"""Hunting objectives that run on their own: kill one monster, or clear the floor she's on.

The decider (a person, or Claude) picks objectives; this module makes the small decisions in a
loop: look at the game, keep the target at Lucky Seven range, face it, throw, step away from
monsters that get too close, loot each kill straight away (or as soon as no monster stands on the
drops), drink a potion when HP gets low, and hand control back when HP is too low, the target is
gone, the floor is clear, or the time is up. Every key goes through the gamelab player, so the
Lie Detector watchdog stops everything the moment the game's anti-bot check appears.

Coordinates: `Sight` positions are in half-size frame pixels (game window / 2); distances given to
moves are game-window pixels. `world` x adds the camera scroll, so positions stay comparable while
the camera follows her.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import mss
import numpy as np

from maplehelper.actions import walk_ms
from maplehelper.config import GameKeys, Timing
from maplehelper.engine import LEFT, RIGHT
from maplehelper.game import COMBAT, MOVEMENT

from . import client, window
from .mobs import SCALE, Mob, MobDetector
from .mapdata import SCREEN_SCALE
from .paths import LAB
from .terrain import find_ropes, her_dot

TAG_FULL = LAB / "tag_template_full.npy"  # her name tag at full resolution (made on first look)
GAME_SIZE = (1920, 1080)  # game window size the templates were made at

# HP and MP bars in the bottom bar, as fractions of the game window (measured at 1920x1080).
HP_BAR = (707 / 1920, 854 / 1920)
MP_BAR = (859 / 1920, 1006 / 1920)
BAR_ROW = 1064 / 1080

# Distances in game-window pixels.
TOO_CLOSE = 90     # a monster nearer than this is about to bump her: step away first
BEST_RANGE = (110, 330)  # throw from here; Lucky Seven was seen reaching ~375 px
STEP_AWAY = 140
MAX_REACH = 370    # throw at targets this far away if she can't get closer (edge of her platform)
EDGE_MARGIN = 25   # never walk closer than this to the edge of the ground she's on
FLOOR_ABOVE = 36   # half-size px: a monster's feet this far above her floor (slimes hop) ...
FLOOR_BELOW = 14   # ... or this far below still count as on her floor
TRACK_SLACK = 60   # half-size px a target may move between two looks (knockback) and still be the same monster


def bars(full: np.ndarray) -> tuple[float, float]:
    """HP and MP as fractions (0-1), read from the bar colours in a full-size game frame."""
    h, w = full.shape[:2]
    row = full[int(BAR_ROW * h)].astype(np.int16)

    def fill(span, colour):
        seg = row[int(span[0] * w):int(span[1] * w)]
        r, g, b = seg[:, 0], seg[:, 1], seg[:, 2]
        on = (r > 150) & (g < 90) & (b < 90) if colour == "red" else (b > 150) & (r < 90)
        return float(on.sum()) / max(1, len(seg))

    return fill(HP_BAR, "red"), fill(MP_BAR, "blue")


def ground_extent(frame: np.ndarray, x: int, floor_y: int) -> tuple[int, int]:
    """Screen x range of the ground she stands on: from her x outwards along the row just under
    her feet, until sky or sea shows (3 pixels in a row). Walls of dirt count as ground, so on a
    floor between cliffs it reaches the walls; on a floating platform it stops at the edges."""
    rows = frame[max(0, floor_y + 2):floor_y + 7].astype(np.int16)
    if rows.size == 0:
        return x, x
    r, g, b = rows[..., 0], rows[..., 1], rows[..., 2]
    sky = ((b > r + 30) & (b >= g - 20)) | ((r > 190) & (g > 190) & (b > 190))
    open_air = sky.mean(axis=0) > 0.5

    def scan(step):
        pos, run = x, 0
        while 0 <= pos + step < frame.shape[1]:
            pos += step
            run = run + 1 if open_air[pos] else 0
            if run >= 3:
                return pos - step * 3
        return pos

    return scan(-1), scan(1)


@dataclass
class Sight:
    t: float
    frame: np.ndarray          # half size
    her: tuple[int, int] | None  # name tag centre; her feet are just above it
    mobs: list[Mob]
    hp: float
    mp: float
    cam: float                 # camera scroll, half-size px; world x = screen x + cam
    ground: tuple[int, int] | None = None  # screen x range of the ground under her
    dot: tuple[float, float] | None = None  # her minimap dot (terrain.her_dot)
    ropes: list = field(default_factory=list)  # terrain.Rope list, full-size px
    map_room: tuple[float, float] | None = None  # screen px to her platform's (left, right) end, from the map

    def room(self, direction: str) -> float:
        """Game px she can walk that way before the edge of her ground: from the map when her platform
        is known (exact), else from the ground seen on screen."""
        if self.map_room is not None:
            edge = self.map_room[1] if direction == RIGHT else self.map_room[0]
            return max(0.0, edge - EDGE_MARGIN)
        if self.her is None or self.ground is None:
            return 0.0
        edge = self.ground[1] - self.her[0] if direction == RIGHT else self.her[0] - self.ground[0]
        return max(0.0, SCALE * edge - EDGE_MARGIN)

    @property
    def floor_y(self) -> int | None:
        """The ground line she stands on (her name tag hangs just below it)."""
        return None if self.her is None else self.her[1] - 8

    def on_floor(self, kinds=None) -> list[Mob]:
        if self.her is None:
            return []
        return [m for m in self.mobs if -FLOOR_ABOVE <= m.feet_y - self.floor_y <= FLOOR_BELOW
                and (kinds is None or m.kind in kinds)]

    def game_dx(self, x: float) -> float:
        """Game-window pixels from her to screen x (+ = to her right)."""
        return SCALE * (x - self.her[0])


class Eyes:
    """Looks at the game without the player (screen capture needs no admin rights).

    Her name tag is matched at full resolution: at half size the score swings between ~0.6 and
    ~0.9 depending on which pixels the downscale keeps, and her name in the minimap scored ~0.6 too.
    """

    def __init__(self, title: str, tag_template: np.ndarray, detector: MobDetector | None = None):
        self.title = title
        self.small_tag = tag_template  # half-size template from calibration, used to find her once
        self.detector = detector or MobDetector()
        self.sct = mss.MSS()
        self.cam = 0.0
        self._strip = None
        self._last: tuple[int, int] | None = None
        self.tag = np.load(TAG_FULL) if TAG_FULL.exists() else None
        self.want_ropes = False  # the navigator turns this on (+~25 ms per look)

    def look(self) -> Sight:
        if window.foreground_title() != self.title:  # the capture sees whatever window is on top
            window.bring_to_front(self.title)
            time.sleep(0.3)
        client_area = window.client_rect(window.find_game(self.title))
        if (client_area["width"], client_area["height"]) != GAME_SIZE:
            raise RuntimeError(f"the game window is {client_area['width']}x{client_area['height']}; monster "
                               f"templates and her name tag were made at {GAME_SIZE[0]}x{GAME_SIZE[1]}")
        full = np.asarray(self.sct.grab(client_area))[:, :, 2::-1]
        hp, mp = bars(full)
        frame = cv2.resize(full, (full.shape[1] // SCALE, full.shape[0] // SCALE), interpolation=cv2.INTER_AREA)
        if self.tag is None:
            self.tag = self._make_full_tag(full)
        her = self._find_her(full)
        mobs, ground = [], None
        if her:
            self._scroll(frame, her)
            mobs = self.detector.detect(frame, rows=(her[1] - 90, her[1] + 20))
            ground = ground_extent(frame, her[0], her[1] - 8)
        return Sight(time.monotonic(), frame, her, mobs, hp, mp, self.cam, ground,
                     her_dot(full), find_ropes(full) if self.want_ropes else [])

    def _make_full_tag(self, full) -> np.ndarray:
        """Find her with the half-size template (trying all four ways of halving the frame), then cut
        a full-resolution template of her name tag there and keep it for next time."""
        best = None
        for oy in (0, 1):
            for ox in (0, 1):
                half = cv2.cvtColor(np.ascontiguousarray(full[oy::SCALE, ox::SCALE]), cv2.COLOR_RGB2GRAY)
                res = cv2.matchTemplate(half[:int(half.shape[0] * 0.9)], self.small_tag, cv2.TM_CCOEFF_NORMED)
                res[:int(half.shape[0] * 0.3), :int(half.shape[1] * 0.17)] = -1  # minimap
                _, score, _, (x, y) = cv2.minMaxLoc(res)
                if best is None or score > best[0]:
                    best = (score, SCALE * x + ox, SCALE * y + oy)
        if best is None or best[0] < 0.75:
            raise RuntimeError("can't find her name tag to build a full-size template; run `calibrate --fresh`")
        # Cut exactly where the half-size template matched, so a name next to hers can't get in.
        th, tw = (SCALE * n for n in self.small_tag.shape)
        tag = cv2.cvtColor(np.ascontiguousarray(full[best[2]:best[2] + th, best[1]:best[1] + tw]), cv2.COLOR_RGB2GRAY)
        np.save(TAG_FULL, tag)
        return tag

    def _find_her(self, full) -> tuple[int, int] | None:
        """Centre of her name tag, in half-size frame pixels. Her name also shows in the minimap,
        the bottom bar and the party list, so of the good matches in the play area, the one nearest
        where she was (or the screen centre, which the camera keeps her near) wins."""
        h, w = full.shape[:2]
        gray = cv2.cvtColor(full[:int(h * 0.9)], cv2.COLOR_RGB2GRAY)  # the bottom bar starts below 90%
        res = cv2.matchTemplate(gray, self.tag, cv2.TM_CCOEFF_NORMED)
        res[:int(h * 0.3), :int(w * 0.17)] = -1  # the minimap (top left)
        th, tw = self.tag.shape
        ref = self._last or (w // 2 // SCALE, int(h * 0.6) // SCALE)
        best = None
        for _ in range(4):
            _, score, _, (x, y) = cv2.minMaxLoc(res)
            if score < 0.7:
                break
            centre = ((x + tw // 2) // SCALE, (y + th // 2) // SCALE)
            dist = abs(centre[0] - ref[0]) + abs(centre[1] - ref[1])
            if best is None or dist < best[0]:
                best = (dist, centre)
            res[max(0, y - 20):y + 20, max(0, x - 60):x + 60] = -1
        self._last = best[1] if best else self._last
        return best[1] if best else None

    def _scroll(self, frame, her) -> None:
        """Follow the camera by how far the ground under her moved since the last look."""
        y0 = min(her[1] + 14, int(frame.shape[0] * 0.88) - 24)
        strip = cv2.cvtColor(frame[y0:y0 + 24], cv2.COLOR_RGB2GRAY).astype(np.float32)
        if self._strip is not None and self._strip[0] == y0:
            win = cv2.createHanningWindow(strip.shape[::-1], cv2.CV_32F)
            (dx, _), response = cv2.phaseCorrelate(self._strip[1], strip, win)
            if response > 0.05 and abs(dx) < frame.shape[1] / 3:
                self.cam -= round(dx * 2) / 2  # scenery moving left = camera moved right
        self._strip = (y0, strip)


@dataclass
class Report:
    kills: dict[str, int] = field(default_factory=dict)
    loot_runs: int = 0
    potions: int = 0
    damage_taken: float = 0.0  # fraction of HP lost overall
    stop: str = ""

    def __str__(self):
        kills = ", ".join(f"{n} {k}" for k, n in self.kills.items()) or "none"
        return (f"kills: {kills}; loot runs: {self.loot_runs}; potions: {self.potions}; "
                f"HP lost: {self.damage_taken:.0%}; stopped because: {self.stop}")


class Hunter:
    def __init__(self, eyes: Eyes, keys: GameKeys, timing: Timing, log: Callable[[str], None] = print,
                 low_hp: float = 0.35, potion_hp: float = 0.5):
        self.eyes, self.keys, self.t, self.log = eyes, keys, timing, log
        self.low_hp, self.potion_hp = low_hp, potion_hp
        self.report = Report()
        self.localizer = None  # set (e.g. by the Navigator) to know her platform's ends from the map
        self.pending_loot: list[tuple[float, float]] = []  # (world x, when)
        self.skipped: list[float] = []  # world x of monsters she can't reach from her platform
        self._last_potion = 0.0
        self._lock_until = 0.0
        self._last_hp: float | None = None

    # --- doing ------------------------------------------------------------------------------

    def _do(self, events: list, name: str) -> None:
        client.submit(events, lead_ms=0, focus=True, name=name)

    def _arrow(self, direction: str) -> str:
        return (self.keys.left if direction == LEFT else self.keys.right).name

    def _wait_lock(self) -> None:
        if (left := self._lock_until - time.monotonic()) > 0:
            time.sleep(left)

    def throw(self, direction: str) -> None:
        """Face `direction`, then Lucky Seven (or a plain attack if it isn't bound)."""
        self._wait_lock()
        key = (self.keys.lucky_seven or self.keys.attack).name
        arrow = self._arrow(direction)
        self._do([[0, arrow, "down"], [MOVEMENT.tap_turn_ms, arrow, "up"],
                  [MOVEMENT.tap_turn_ms + 30, key, "down"], [MOVEMENT.tap_turn_ms + 70, key, "up"]], "throw")
        self._lock_until = time.monotonic() + (COMBAT.attack_lock_ms + 60) / 1000

    def walk(self, direction: str, px: float) -> None:
        self._wait_lock()
        ms = walk_ms(px)
        arrow = self._arrow(direction)
        self._do([[0, arrow, "down"], [ms, arrow, "up"]], "walk")
        time.sleep(MOVEMENT.stop_slide_ms / 1000)

    def sweep_loot(self, direction: str, px: float) -> None:
        """Walk `px` while pressing pick-up every 150 ms, collecting whatever she passes over."""
        self._wait_lock()
        pick = (self.keys.pick_up or self.keys.attack).name
        arrow = self._arrow(direction)
        ms = max(300, walk_ms(px))
        events = [[0, arrow, "down"]]
        events += [e for t in range(0, ms + 300, 150) for e in ([t, pick, "down"], [t + 40, pick, "up"])]
        events.append([ms, arrow, "up"])
        events.sort(key=lambda e: e[0])
        self._do(events, "loot")
        self.report.loot_runs += 1

    def drink(self) -> None:
        if self.keys.hp_potion is None:
            return
        self._wait_lock()
        key = self.keys.hp_potion.name
        self._do([[0, key, "down"], [40, key, "up"]], "potion")
        self._last_potion = time.monotonic()
        self.report.potions += 1

    # --- looking ----------------------------------------------------------------------------

    def look(self) -> Sight:
        sight = self.eyes.look()
        if self.localizer is not None and sight.dot is not None:
            where = self.localizer.where_dot(sight.dot)
            if where.platform is not None:
                p = where.platform
                sight.map_room = ((where.x - p.left) * SCREEN_SCALE, (p.right - where.x) * SCREEN_SCALE)
                if sight.her is not None:
                    # Camera from the map instead of from scrolling scenery (which fails over open sky):
                    # world x (half-size px) = her map x in screen units + offset from her on screen.
                    sight.cam = where.x * SCREEN_SCALE / SCALE - sight.her[0]
        if self._last_hp is not None and sight.hp < self._last_hp:
            self.report.damage_taken += self._last_hp - sight.hp
        self._last_hp = sight.hp
        return sight

    def health_check(self, sight: Sight) -> str | None:
        """Drink if HP is low; returns a reason to stop if it's too low."""
        if sight.hp < self.low_hp:
            return f"HP low ({sight.hp:.0%})"
        if sight.hp < self.potion_hp and time.monotonic() - self._last_potion > 3:
            self.log(f"HP {sight.hp:.0%}: drinking a potion")
            self.drink()
        return None

    # --- objectives -------------------------------------------------------------------------

    def kill(self, target: Mob, sight: Sight, seconds: float = 30) -> str:
        """Attack `target` (as seen in `sight`) until it dies; then loot. Returns how it ended:
        'killed', 'lost', 'timeout', or a reason to stop (low HP)."""
        kind, world = target.kind, target.x + sight.cam
        last_world = world
        deadline, misses, throws = time.monotonic() + seconds, 0, 0
        gone_since: float | None = None
        self.log(f"target: {kind} {sight.game_dx(target.x):+.0f} px away")
        while time.monotonic() < deadline:
            sight = self.look()
            if sight.her is None:
                misses += 1
                if misses > 6:
                    return "lost sight of her"
                continue
            if stop := self.health_check(sight):
                return stop
            found = [m for m in sight.on_floor({kind}) if abs(m.x + sight.cam - last_world) <= TRACK_SLACK]
            if not found:
                misses += 1
                gone_since = gone_since or time.monotonic()
                # Hit monsters flinch, get knocked back and are covered by damage numbers, so it
                # only counts as dead after ~1.2 s of not being seen.
                if misses >= 4 and time.monotonic() - gone_since >= 1.2:
                    if throws:
                        self.report.kills[kind] = self.report.kills.get(kind, 0) + 1
                        self.log(f"{kind} down after {throws} throws: looting")
                        self.loot_at(last_world)
                        return "killed"
                    return "lost"
                time.sleep(0.15)
                continue
            misses, gone_since = 0, None
            target = min(found, key=lambda m: abs(m.x + sight.cam - last_world))
            last_world = target.x + sight.cam
            move = next_move(sight, target)
            self.log(f"  {kind} {sight.game_dx(target.x):+.0f} px, HP {sight.hp:.0%}: {move[0]} {move[1]}"
                     + (f" {move[2]:.0f} px" if len(move) > 2 else ""))
            if move[0] == "skip":
                self.skipped.append(target.x + sight.cam)
                return "unreachable"
            if move[0] == "throw":
                self.throw(move[1])
                throws += 1
            else:
                self.walk(move[1], move[2])
        return "timeout"

    def loot_at(self, world_x: float, since: float | None = None) -> bool:
        """Go over the drops at world_x and pick them up, unless a monster stands on them; then
        they're remembered and collected after a later kill."""
        sight = self.look()
        since = since or time.monotonic()
        if sight.her is None:
            self.pending_loot.append((world_x, since))
            return False
        spot = world_x - sight.cam  # screen x of the drops
        if any(abs(SCALE * (m.x - spot)) < 110 for m in sight.on_floor()):
            self.log("  drops guarded by a monster: later")
            self.pending_loot.append((world_x, since))
            return False
        dx = sight.game_dx(spot)
        direction = RIGHT if dx > 0 else LEFT
        if abs(dx) - 50 > sight.room(direction) + 40:
            self.log("  drops are off her platform: leaving them")
            return True
        if abs(dx) > 60:
            self.walk(direction, min(abs(dx) - 50, sight.room(direction)))
        self.sweep_loot(direction, min(100, max(30, sight.room(direction) - max(0, abs(dx) - 50))))
        return True

    def loot_pending(self) -> None:
        """Try the remembered drops again (loot_at re-queues any that are still guarded)."""
        queue, self.pending_loot = self.pending_loot, []
        for world_x, since in queue:
            if time.monotonic() - since < 90:  # drops vanish after a while
                self.loot_at(world_x, since)

    def clear_floor(self, kinds=None, max_kills: int = 10, seconds: float = 120, span: float = 900) -> Report:
        """Kill the monsters on her floor, nearest first, looting as she goes. Stays within `span`
        game px of where she started. Stops when the floor stays clear for ~3 s."""
        deadline, calm, unseen = time.monotonic() + seconds, 0, 0
        start = None
        while time.monotonic() < deadline and sum(self.report.kills.values()) < max_kills:
            sight = self.look()
            if sight.her is None:  # her name tag hidden (e.g. behind the party HP box at the edge)
                unseen += 1
                if unseen >= 10:
                    self.report.stop = "lost sight of her"
                    return self.report
                time.sleep(0.3)
                continue
            unseen = 0
            if start is None:
                start = sight.her[0] + sight.cam
            if stop := self.health_check(sight):
                self.report.stop = stop
                return self.report
            here = [m for m in sight.on_floor(kinds)
                    if abs(SCALE * (m.x + sight.cam - start)) <= span
                    and not any(abs(m.x + sight.cam - s) < TRACK_SLACK for s in self.skipped)]
            if not here:
                calm += 1
                if self.pending_loot:
                    self.loot_pending()
                if calm >= 6:
                    self.report.stop = "floor clear"
                    return self.report
                time.sleep(0.5)
                continue
            calm = 0
            target = min(here, key=lambda m: abs(m.x - sight.her[0]))
            outcome = self.kill(target, sight)
            if outcome not in ("killed", "lost", "timeout", "unreachable"):
                self.report.stop = outcome
                return self.report
            if self.pending_loot:
                self.loot_pending()
        self.report.stop = "kill limit" if sum(self.report.kills.values()) >= max_kills else "time limit"
        return self.report


def next_move(sight: Sight, target: Mob) -> tuple:
    """What to do about `target` now: ('throw', direction), ('walk', direction, px), or
    ('skip', reason) when it can't be reached from her platform. Never walks off her ground."""
    dx = sight.game_dx(target.x)
    toward = RIGHT if dx > 0 else LEFT
    others = [sight.game_dx(m.x) for m in sight.on_floor() if m is not target]
    threats = [d for d in others if abs(d) < TOO_CLOSE]
    if abs(dx) < TOO_CLOSE or threats:
        # Step away from whatever is closest, unless that walks into another monster or off the edge.
        closest = min([dx] + threats, key=abs)
        direction = LEFT if closest > 0 else RIGHT
        reach = STEP_AWAY + TOO_CLOSE
        blocked = any((0 < d < reach) if direction == RIGHT else (-reach < d < 0) for d in others + [dx])
        step = min(STEP_AWAY, sight.room(direction))
        if not blocked and step >= 60:
            return ("walk", direction, step)
        return ("throw", toward)  # cornered: fight
    if abs(dx) > BEST_RANGE[1]:
        if abs(dx) - sight.room(toward) > MAX_REACH:  # even from the edge it'd be too far
            return ("skip", "out of reach from this platform")
        step = min(300, abs(dx) - (BEST_RANGE[0] + BEST_RANGE[1]) / 2, sight.room(toward))
        if step >= 30:
            return ("walk", toward, step)
    return ("throw", toward)
