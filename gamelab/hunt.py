"""Hunting objectives that run on their own: kill one monster, or clear the monsters in reach.

The decider (a person, or Claude) picks objectives; this module makes the small decisions in a
loop: look at the game, keep the target at the far end of a star's reach, face it, throw a regular
attack (Lucky Seven costs MP and isn't needed on these monsters), step away from monsters that get
close, loot each kill straight away when it fell on her platform (drops elsewhere are remembered for
the tour to collect), drink a potion when HP gets low, and hand control back when HP is too low, the
target is gone, nothing is left in reach, or the time is up. Every key goes through the gamelab
player, so the Lie Detector watchdog stops everything the moment the game's anti-bot check appears.

Positions are in map units (mapdata), from her minimap dot and her name tag: monsters are placed on
the map by their offset from her on screen. She's a ranged job, so targets can be on other
platforms (shots.py): standing where no monster can walk up to her and shooting from there is best.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import mss
import numpy as np

from maplehelper.config import GameKeys, Timing
from maplehelper.engine import LEFT, RIGHT
from maplehelper.game import COMBAT, MOVEMENT

from . import client, shots, window
from .localize import Camera, Localizer, Where
from .mapdata import SCREEN_SCALE, MapModel, Platform
from .mobs import SCALE, Mob, MobDetector
from .paths import LAB
from .terrain import find_ropes, her_dot

TAG_FULL = LAB / "tag_template_full.npy"  # her name tag at full resolution (made on first look)
SHOT_LOG = LAB / "shots.jsonl"            # every throw: where the target was, and whether it was hit
HIT_LOG = LAB / "hits.jsonl"              # hits taken standing still: how far each pushed her
GAME_SIZE = (1920, 1080)  # game window size the templates were made at
UNITS_PER_HALF_PX = SCALE / SCREEN_SCALE  # map units per half-size frame pixel
# Screen px the drawn ground sticks out past a platform's real end. Negative: measured against ropes
# on platforms 32, 30 and 14, the real ends are 21-27 px beyond where the grass seems to end.
GRASS_OVERHANG = -24

# Her EXP number in the bottom bar ("1,385 [8.30%]"), full-size px: it only changes when she gains
# EXP, i.e. when a monster she hit dies. It sits on the bar's plain dark background, so its text pixels
# are identical from look to look otherwise. (The "You received EXP" message at the bottom right was
# tried first, but it's faint and was missed over a bright background.)
EXP_NUMBER = (1060, 1030, 1200, 1056)  # x0, y0, x1, y1

# HP and MP bars in the bottom bar, as fractions of the game window (measured at 1920x1080).
HP_BAR = (707 / 1920, 854 / 1920)
MP_BAR = (859 / 1920, 1006 / 1920)
BAR_ROW = 1064 / 1080

# Distances in map units.
TOO_CLOSE = 110    # a monster on her platform nearer than this is about to touch her: step away first
STEP_AWAY = 120
KNOCKBACK_ROOM = 25  # extra room kept from her platform's ends while monsters share it: standing ~100
                     # units from an end of platform 12, pigs walking into her knocked her off it
JUMP_OVER = 80       # how far a jump started from standing carries her (map units)
MAX_WALK = 300     # longest walk between two looks
TRACK_SLACK = 85   # how far a target may move between two looks (knockback) and still be the same monster
SEARCH_SLACK = 150  # ... and how far it's looked for once it has vanished
LOST_AFTER = 3.0   # s a target may stay unseen (without an EXP message) before she gives up on it
LOOT_REACH = 60    # drops spread about this far around where the monster died
# Where monsters are looked for: from this far above her feet to this far below (bodies stick up
# above their feet, so the band reaches higher than she can shoot).
LOOK_UP = shots.SHOT_UP + 80
LOOK_DOWN = shots.SHOT_DOWN + 15


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


@dataclass(frozen=True)
class Target:
    """A monster placed on the map."""
    mob: Mob
    x: float              # map units, its feet
    y: float
    platform: int | None  # the platform it walks on (monsters never leave it)
    bar: float | None = None  # its HP bar (fraction left), shown once it has been hit; None: unhurt

    @property
    def kind(self) -> str:
        return self.mob.kind


@dataclass
class Sight:
    t: float
    frame: np.ndarray          # half size
    her: tuple[int, int] | None  # name tag centre (half-size px); her feet are just above it
    mobs: list[Mob]
    hp: float
    mp: float
    ground: tuple[int, int] | None = None  # screen x range of the ground under her
    dot: tuple[float, float] | None = None  # her minimap dot (terrain.her_dot)
    ropes: list = field(default_factory=list)  # terrain.Rope list, full-size px
    where: Where | None = None   # her map position (minimap dot, snapped to the platform she's on)
    how: str = "dot"             # how `where.x` was worked out: 'rope', 'edge' (on screen) or 'dot'
    seen: bool = True            # her screen position from her name tag (False: from the dot and camera)
    targets: list[Target] = field(default_factory=list)
    exp: np.ndarray | None = None  # text pixels of her EXP number (EXP_NUMBER)

    @property
    def floor_y(self) -> int | None:
        """The ground line she stands on (her name tag hangs just below it)."""
        return None if self.her is None else self.her[1] - 8

    @property
    def platform(self) -> Platform | None:
        return None if self.where is None else self.where.platform

    def place(self, m: Mob, model: MapModel) -> Target:
        """Put a monster seen on screen onto the map, by its offset from her."""
        x = self.where.x + (m.x - self.her[0]) * UNITS_PER_HALF_PX
        y = self.where.y + (m.feet_y - self.floor_y) * UNITS_PER_HALF_PX
        p = monster_platform(model, m.kind, x, y)
        bar = shots.hp_bar(self.frame, m.x, m.y - m.h // 2)
        return Target(m, x, p.y_at(x) if p else y, None if p is None else p.id, bar)

    def screen_x(self, map_x: float) -> float:
        """Half-size screen x of a map x (at her height)."""
        return self.her[0] + (map_x - self.where.x) / UNITS_PER_HALF_PX

    def room(self, direction: str, margin: float = shots.EDGE_MARGIN) -> float:
        """Map units she can walk that way before the end of her platform (less a margin)."""
        p = self.platform
        if p is None:
            return 0.0
        return max(0.0, (p.right - margin - self.where.x) if direction == RIGHT else (self.where.x - p.left - margin))


def monster_platform(model: MapModel, kind: str, x: float, y: float, reach: float = 50) -> Platform | None:
    """The platform a monster seen at map (x, y) walks on. Monsters never leave the platform they
    spawn on, so only platforms where its kind spawns count (or any spawn platform, if none fits):
    feet read from the screen are ~20 units off at times, and a safe platform 42-60 units above a
    hunting ground once had pigs "on" it, so she backed away from them and off her platform."""
    from .sprites import kind_of
    best = None
    for own_kind in (True, False):
        for pid, names in model.monsters_by_platform.items():
            if own_kind and not any(kind_of(n) == kind for n in names):
                continue
            p = model.platform(pid)
            if not p.covers(x, 25):
                continue
            gap = abs(p.y_at(min(max(x, p.left), p.right)) - y)
            if gap <= reach and (best is None or gap < best[0]):
                best = (gap, p)
        if best:
            return best[1]
    return None


def rope_x(where: Where, s: Sight, model: MapModel) -> float | None:
    """Her x from ropes in view: each is drawn exactly at its map x, and detected to ~2 px."""
    if s.her is None or not s.ropes:
        return None
    her_x = SCALE * s.her[0]
    estimates = []
    for seen in s.ropes:
        for rope in model.ropes:
            expected = her_x + (rope.x - where.x) * SCREEN_SCALE  # where the dot says it should be
            if abs(seen.x - expected) <= 40:
                estimates.append(rope.x - (seen.x - her_x) / SCREEN_SCALE)
    if not estimates:
        return None
    estimates.sort()
    return estimates[len(estimates) // 2]


def edge_x(where: Where, s: Sight) -> float | None:
    """Her x from the visible end(s) of her platform."""
    p = where.platform
    if p is None or s.her is None or s.ground is None:
        return None
    her_x = SCALE * s.her[0]
    left, right = SCALE * s.ground[0] + GRASS_OVERHANG, SCALE * s.ground[1] - GRASS_OVERHANG
    width = s.frame.shape[1] * SCALE
    candidates = []
    if right < width - 12 - GRASS_OVERHANG:  # her platform's right end is on screen
        candidates.append(p.right - (right - her_x) / SCREEN_SCALE)
    if left > 12 + GRASS_OVERHANG:
        candidates.append(p.left + (her_x - left) / SCREEN_SCALE)
    good = [x for x in candidates if abs(x - where.x) <= 45]  # must agree with the dot roughly
    return sum(good) / len(good) if good else None


def refine(where: Where, s: Sight, model: MapModel) -> tuple[Where, str]:
    """Her x along her platform, best source first: a rope in view (ropes are drawn exactly at their
    map x), the visible end of her platform, else the minimap dot (only good to ~11 map units).
    Both screen sources need her name tag."""
    p = where.platform
    if p is None or s.her is None:
        return where, "dot"
    for how, x in (("rope", rope_x(where, s, model)), ("edge", edge_x(where, s))):
        if x is not None and abs(x - where.x) <= 45:
            x = min(max(x, p.left), p.right)
            return Where(x, p.y_at(x), p, where.dot), how
    return where, "dot"


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
        self._last: tuple[int, int] | None = None
        self.tag = np.load(TAG_FULL) if TAG_FULL.exists() else None
        self.want_ropes = False  # the navigator turns this on (+~25 ms per look)

    def look(self) -> Sight:
        """HP/MP, her name tag, the ground under her and her minimap dot; no monsters (see `detect`)."""
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
        ground = ground_extent(frame, her[0], her[1] - 8) if her else None
        s = Sight(time.monotonic(), frame, her, [], hp, mp, ground, her_dot(full),
                  find_ropes(full) if self.want_ropes else [])
        x0, y0, x1, y1 = EXP_NUMBER
        s.exp = full[y0:y1, x0:x1].max(axis=2) > 150
        return s

    def glance(self) -> tuple[tuple[float, float] | None, float]:
        """Just her minimap dot and HP, from two small captures (a few ms): for watching a plan run."""
        from .terrain import MINIMAP_BOX, dot_in_box
        area = window.client_rect(window.find_game(self.title))
        x0, y0, x1, y1 = MINIMAP_BOX
        box = np.asarray(self.sct.grab({"left": area["left"] + x0, "top": area["top"] + y0,
                                        "width": x1 - x0, "height": y1 - y0}))[:, :, 2::-1]
        w, h = area["width"], area["height"]
        a, b = int(HP_BAR[0] * w), int(HP_BAR[1] * w)
        row = np.asarray(self.sct.grab({"left": area["left"] + a, "top": area["top"] + int(BAR_ROW * h),
                                        "width": b - a, "height": 1}))[0, :, 2::-1].astype(np.int16)
        hp = float(((row[:, 0] > 150) & (row[:, 1] < 90) & (row[:, 2] < 90)).sum()) / max(1, b - a)
        return dot_in_box(box), hp

    def detect(self, sight: Sight, up: float = LOOK_UP, down: float = LOOK_DOWN) -> list[Mob]:
        """Monsters from `up` map units above her feet to `down` below (her screen position must be known)."""
        feet, bottom = sight.floor_y, int(sight.frame.shape[0] * 0.88)  # the bottom bar starts below
        return self.detector.detect(sight.frame, rows=(max(0, int(feet - up / UNITS_PER_HALF_PX)),
                                                       min(bottom, int(feet + down / UNITS_PER_HALF_PX))))

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


@dataclass
class Report:
    kills: dict[str, int] = field(default_factory=dict)
    loot_runs: int = 0
    potions: int = 0
    throws: int = 0
    out_of_reach: int = 0      # targets seen that she couldn't reach from her platform
    exp_kills: int = 0         # times her EXP went up: every kill of hers, target or not
    lost: int = 0              # targets that vanished without her EXP going up (hidden, or killed by others)
    damage_taken: float = 0.0  # fraction of HP lost overall
    stop: str = ""

    def __str__(self):
        kills = ", ".join(f"{n} {k}" for k, n in self.kills.items()) or "none"
        return (f"kills: {kills} (EXP gains: {self.exp_kills}, targets lost: {self.lost}); "
                f"throws: {self.throws}; loot runs: {self.loot_runs}; potions: {self.potions}; "
                f"HP lost: {self.damage_taken:.0%}; stopped because: {self.stop}")


@dataclass
class Drops:
    """Loot left where a monster died, to be collected (by the tour) when it's off her platform."""
    x: float
    y: float
    platform: int | None
    since: float


class Hunter:
    def __init__(self, eyes: Eyes, keys: GameKeys, timing: Timing, model: MapModel,
                 log: Callable[[str], None] = print, low_hp: float = 0.35, potion_hp: float = 0.5,
                 skill: bool = False):
        self.eyes, self.keys, self.t, self.model, self.log = eyes, keys, timing, model, log
        self.low_hp, self.potion_hp = low_hp, potion_hp
        self.skill = skill  # throw Lucky Seven instead of the regular attack (costs MP)
        self.localizer = Localizer(model)
        self.camera = Camera(model.bounds)
        eyes.want_ropes = True  # ropes in view pin down her x (+~25 ms per look)
        self.report = Report()
        self.pending_loot: list[Drops] = []
        self.skipped: list[tuple[float, float, float]] = []  # map (x, y, when) of monsters out of reach
        self.facing: str | None = None  # which way she faces (None: unknown), so throws only turn when needed
        self._last_potion = 0.0
        self._lock_until = 0.0
        self._last_hp: float | None = None
        self._last_where: Where | None = None
        self._exp: np.ndarray | None = None  # last look's EXP number (text pixels)
        self.exp_events: list[float] = []  # when her EXP went up
        self._pressed = False  # keys sent since the last look (then a move isn't a knockback)

    # --- doing ------------------------------------------------------------------------------

    def _do(self, events: list, name: str) -> None:
        client.submit(events, lead_ms=0, focus=True, name=name)
        self._pressed = True

    def _arrow(self, direction: str) -> str:
        return (self.keys.left if direction == LEFT else self.keys.right).name

    def _wait_lock(self) -> None:
        if (left := self._lock_until - time.monotonic()) > 0:
            time.sleep(left)

    def throw(self, direction: str) -> None:
        """Face `direction` (if she isn't already), then throw (regular attack; Lucky Seven if `skill`).
        Turning taps the arrow, which also walks her a few units: tapping before every throw walked
        her off the end of a platform after nine throws the same way."""
        self._wait_lock()
        key = (self.keys.lucky_seven if self.skill and self.keys.lucky_seven else self.keys.attack).name
        events, t = [], 0
        if self.facing != direction:
            arrow = self._arrow(direction)
            events, t = [[0, arrow, "down"], [MOVEMENT.tap_turn_ms, arrow, "up"]], MOVEMENT.tap_turn_ms + 30
        self._do(events + [[t, key, "down"], [t + 40, key, "up"]], "throw")
        self.facing = direction
        self._lock_until = time.monotonic() + (COMBAT.attack_lock_ms + 60) / 1000
        self.report.throws += 1

    def walk(self, direction: str, units: float) -> None:
        from .moves import hold_ms
        self._wait_lock()
        arrow = self._arrow(direction)
        self._do([[0, arrow, "down"], [hold_ms(units), arrow, "up"]], "walk")
        self.facing = direction
        time.sleep(MOVEMENT.stop_slide_ms / 1000)

    def jump_over(self, direction: str) -> None:
        """Jump that way from standing (arrow and jump together), over a monster that's too close."""
        self._wait_lock()
        arrow, jump = self._arrow(direction), self.keys.jump.name
        self._do([[0, arrow, "down"], [10, jump, "down"], [50, jump, "up"], [650, arrow, "up"]], "jump_over")
        self.facing = direction
        time.sleep(MOVEMENT.stop_slide_ms / 1000)

    def sweep_loot(self, direction: str, units: float) -> None:
        """Walk `units` (0: stay put) while pressing pick-up every 150 ms, collecting whatever she
        passes over. The walk is exactly `units`: a minimum hold once walked her off a platform."""
        from .moves import hold_ms
        self._wait_lock()
        pick = (self.keys.pick_up or self.keys.attack).name
        arrow = self._arrow(direction)
        ms = hold_ms(units) if units >= 10 else 0
        events = [[0, arrow, "down"], [ms, arrow, "up"]] if ms else []
        events += [e for t in range(0, ms + 450, 150) for e in ([t, pick, "down"], [t + 40, pick, "up"])]
        events.sort(key=lambda e: e[0])
        self._do(events, "loot")
        self.facing = direction
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

    def look(self, mobs: bool = True, wide: bool = False) -> Sight:
        """Her map position from the minimap dot; her screen position from her name tag, or from the
        dot and the camera when the tag is hidden; and (if `mobs`) the monsters she could shoot, or
        (if `wide`) every monster on screen."""
        s = self.eyes.look()
        if s.dot is not None:
            s.where = self.localizer.where_dot(s.dot)
            if s.her is not None:
                s.where, s.how = refine(s.where, s, self.model)
        if s.where is not None:
            if s.her is not None and s.seen:
                self.camera.learn((s.where.x, s.where.y), (SCALE * s.her[0], SCALE * (s.her[1] - 8)))
            else:
                sx, sy = self.camera.to_screen(s.where.x, s.where.y, (s.where.x, s.where.y))
                s.her, s.seen = (round(sx / SCALE), round(sy / SCALE) + 8), False
                s.ground = ground_extent(s.frame, s.her[0], s.her[1] - 8)
        if mobs and s.her is not None and s.where is not None:
            s.mobs = self.eyes.detect(s, up=2000, down=2000) if wide else self.eyes.detect(s)
            s.targets = [s.place(m, self.model) for m in s.mobs]
        if self._exp is not None and s.exp is not None and s.exp.shape == self._exp.shape                 and int((s.exp != self._exp).sum()) >= 3:
            self.exp_events.append(s.t)
            self.report.exp_kills += 1
        self.exp_events = [t for t in self.exp_events if s.t - t < 30]
        self._exp = s.exp
        if self._last_hp is not None and s.hp < self._last_hp:
            self.report.damage_taken += self._last_hp - s.hp
            self._log_hit(s)
        self._last_hp, self._last_where, self._pressed = s.hp, s.where, False
        return s

    def _log_hit(self, s: Sight) -> None:
        """A hit while no key was pressed since the last look: how far it pushed her (knockback)."""
        before = self._last_where
        if self._pressed or before is None or s.where is None or before.platform is None:
            return
        pushed = s.where.x - before.x
        same = s.platform is not None and s.platform.id == before.platform.id
        self.log(f"  hit standing still: pushed {pushed:+.0f} units" + ("" if same else
                 f", off platform {before.platform.id} onto {s.platform.id if s.platform else 'nothing'}"))
        try:
            with HIT_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"pushed": round(pushed, 1), "dy": round(s.where.y - before.y, 1),
                                    "hp": round(self._last_hp - s.hp, 4), "how": s.how,
                                    "platform": before.platform.id, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        except OSError:
            pass

    def health_check(self, sight: Sight) -> str | None:
        """Drink if HP is low; returns a reason to stop if it's too low."""
        if sight.hp < self.low_hp:
            return f"HP low ({sight.hp:.0%})"
        if sight.hp < self.potion_hp and time.monotonic() - self._last_potion > 3:
            self.log(f"HP {sight.hp:.0%}: drinking a potion")
            self.drink()
        return None

    # --- objectives -------------------------------------------------------------------------

    def kill(self, target: Target, seconds: float = 30) -> str:
        """Shoot `target` until it dies; then loot (or remember the drops if they're off her
        platform). Returns how it ended: 'killed', 'lost', 'timeout', 'unreachable', or a reason
        to stop (low HP)."""
        kind, last = target.kind, (target.x, target.y)
        platform = target.platform
        deadline, misses, throws = time.monotonic() + seconds, 0, 0
        gone_since: float | None = None
        first_throw: float | None = None
        flying: list[dict] = []  # throws whose hit isn't confirmed yet (see _score_shots)
        self.log(f"target: {kind} on platform {platform}")
        while time.monotonic() < deadline:
            s = self.look()
            if s.where is None or s.platform is None:
                misses += 1
                if misses > 8:
                    return "lost track of her"
                time.sleep(0.15)
                continue
            if stop := self.health_check(s):
                return stop
            slack = TRACK_SLACK if gone_since is None else SEARCH_SLACK
            found = [t for t in s.targets if t.kind == kind
                     and abs(t.x - last[0]) <= slack and abs(t.y - last[1]) <= 40]
            if not found:
                misses += 1
                gone_since = gone_since or time.monotonic()
                # Dead only if her EXP went up since she started throwing at it. Vanishing alone
                # isn't enough: hit monsters flinch and hide behind damage numbers, and other
                # players kill monsters too (about half the "kills" counted by vanishing alone
                # weren't hers).
                if first_throw is not None and self._take_exp(first_throw):
                    for sh in flying:
                        self._log_shot(sh, sh["due"] <= gone_since + 0.2)
                    self.report.kills[kind] = self.report.kills.get(kind, 0) + 1
                    self.log(f"{kind} down after {throws} throws")
                    self.loot(Drops(last[0], last[1], platform, time.monotonic()))
                    return "killed"
                if time.monotonic() - gone_since >= LOST_AFTER:
                    if throws:
                        self.report.lost += 1
                        self.log(f"{kind} gone without her EXP going up after {throws} throws: not counted")
                    return "lost"
                time.sleep(0.15)
                continue
            misses, gone_since = 0, None
            # An EXP gain while it's still alive was another monster dying (a star hits the first
            # monster in its way): it isn't this one's.
            self.exp_events = [e for e in self.exp_events if s.t - e < 1.5]
            t = min(found, key=lambda t: abs(t.x - last[0]))
            last, platform = (t.x, t.y), t.platform if t.platform is not None else platform
            bar = shots.hp_bar(s.frame, t.mob.x, t.mob.y - t.mob.h // 2)
            flying = self._score_shots(flying, bar)
            threats = [o for o in s.targets if o.platform == s.platform.id and o is not t]
            move = decide(s.where.x, s.where.y, s.platform, t, threats)
            self.log(f"  {kind} {t.x - s.where.x:+.0f},{t.y - s.where.y:+.0f} (bar {_pct(bar)}), "
                     f"HP {s.hp:.0%}: {' '.join(str(v if isinstance(v, str) else round(v)) for v in move)}")
            if move[0] == "skip":
                self.skipped.append((t.x, t.y, time.monotonic()))
                self.report.out_of_reach += 1
                return "unreachable"
            if move[0] == "throw":
                dx, dy = t.x - s.where.x, t.y - s.where.y
                if shots.can_shoot(dx, dy, reach=shots.STAR_REACH) and move[1] == (RIGHT if dx > 0 else LEFT):
                    flying.append({"kind": kind, "dx": int(round(dx)), "dy": int(round(dy)), "her": s.platform.id,
                                   "target": t.platform, "bar": bar, "seen": bool(s.seen),
                                   "due": time.monotonic() + COMBAT.attack_release_ms / 1000
                                   + abs(dx) / shots.STAR_SPEED})
                self.throw(move[1])
                throws += 1
                first_throw = first_throw or time.monotonic()
            elif move[0] == "jump":
                self.jump_over(move[1])
            else:
                self.walk(move[1], move[2])
        return "timeout"

    def _take_exp(self, since: float) -> bool:
        """Use up one EXP gain seen since `since`, if there is one."""
        for i, t in enumerate(self.exp_events):
            if t >= since:
                del self.exp_events[i]
                return True
        return False

    def _score_shots(self, flying: list[dict], bar: float | None) -> list[dict]:
        """A throw hit when the target's HP bar shows up or drops after the star should have arrived;
        it missed when nothing changed within ~0.9 s of that. Returns the throws still undecided."""
        now, left = time.monotonic(), []
        for sh in flying:
            if now >= sh["due"] - 0.1 and bar is not None and (sh["bar"] is None or bar < sh["bar"] - 0.02):
                self._log_shot(sh, True)
            elif now > sh["due"] + 0.9:
                self._log_shot(sh, False)
            else:
                left.append(sh)
        return left

    def _log_shot(self, shot: dict, hit: bool) -> None:
        """Keep a record of every throw: the reach of a star is measured from these."""
        try:
            with SHOT_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**{k: v for k, v in shot.items() if k not in ("bar", "due")}, "hit": bool(hit),
                                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        except OSError:
            pass

    def loot(self, drops: Drops) -> bool:
        """Sweep the drops now if they're on her platform and no monster stands on them; otherwise
        remember them (pending_loot) for later. Returns True when swept."""
        s = self.look()
        if s.where is None or s.platform is None or drops.platform != s.platform.id:
            self.pending_loot.append(drops)
            return False
        if any(t.platform == drops.platform and abs(t.x - drops.x) < TOO_CLOSE for t in s.targets):
            self.log("  drops guarded by a monster: later")
            self.pending_loot.append(drops)
            return False
        self.sweep(s, drops.x)
        return True

    def sweep(self, s: Sight, x: float) -> None:
        """Walk over map x on her platform pressing pick-up: from LOOT_REACH before it to LOOT_REACH past it."""
        dx = x - s.where.x
        direction = RIGHT if dx > 0 else LEFT
        step = min(max(0.0, abs(dx) - LOOT_REACH), s.room(direction))
        if step >= 10:
            self.walk(direction, step)
        room = s.room(direction) - step  # never past her platform's end (less the margin)
        self.sweep_loot(direction, max(0.0, min(2 * LOOT_REACH, room)))

    def loot_pending(self, s: Sight | None = None, max_age: float = 90) -> None:
        """Sweep remembered drops that are on her platform (loot() re-queues any still guarded); the
        rest stay remembered. `s` is a recent look, to tell which ones are on her platform."""
        s = s or self.look(mobs=False)
        here = None if s.platform is None else s.platform.id
        fresh = [d for d in self.pending_loot if time.monotonic() - d.since < max_age]  # drops vanish
        self.pending_loot = [d for d in fresh if d.platform != here]
        for d in fresh:
            if d.platform == here:
                self.loot(d)

    def clear(self, kinds=None, ground: set[int] | None = None, max_kills: int = 10,
              seconds: float = 120, patience: float = 3.0) -> Report:
        """Kill monsters (of `kinds`) on the `ground` platforms, or anywhere if None, and any on her own
        platform: nearest in reach first; walking along her platform to bring others into reach.
        Stops when nothing has been in reach for `patience` seconds (waiting for respawns)."""
        deadline, unseen = time.monotonic() + seconds, 0
        calm_since: float | None = None
        start: int | None = None  # the platform she's on; leaving it means she was knocked off
        while time.monotonic() < deadline and sum(self.report.kills.values()) < max_kills:
            s = self.look()
            if s.where is None or s.platform is None:
                unseen += 1
                if unseen >= 10:
                    self.report.stop = "lost track of her"
                    return self.report
                time.sleep(0.3)
                continue
            unseen = 0
            if stop := self.health_check(s):
                self.report.stop = stop
                return self.report
            start = start if start is not None else s.platform.id
            if s.platform.id != start:
                self.log(f"  she's on platform {s.platform.id} now, not {start} (knocked off?)")
                self.report.stop = f"left platform {start}"
                return self.report
            now = time.monotonic()
            self.skipped = [k for k in self.skipped if now - k[2] < 8]  # monsters move: try again later
            here = [t for t in s.targets if (kinds is None or t.kind in kinds)
                    and (ground is None or t.platform in ground or t.platform == s.platform.id)
                    and not any(abs(t.x - x) < TRACK_SLACK and abs(t.y - y) < 40 for x, y, _ in self.skipped)]
            if not here:
                calm_since = calm_since or now
                if self.pending_loot:
                    self.loot_pending(s)
                if now - calm_since >= patience:
                    self.report.stop = "nothing left in reach"
                    return self.report
                time.sleep(0.5)
                continue
            calm_since = None
            target = pick_target(s.where.x, s.where.y, s.platform, here)
            outcome = self.kill(target, seconds=min(30, deadline - time.monotonic()))
            if outcome not in ("killed", "lost", "timeout", "unreachable"):
                self.report.stop = outcome
                return self.report
            if self.pending_loot:
                self.loot_pending()
        self.report.stop = "kill limit" if sum(self.report.kills.values()) >= max_kills else "time limit"
        return self.report


def _pct(bar: float | None) -> str:
    return "-" if bar is None else f"{bar:.0%}"


def pick_target(x: float, y: float, platform: Platform, targets: list[Target]) -> Target:
    """Monsters on her platform that are close first (they can reach her), then the most hurt one a star
    reaches from where she stands (finish what was started), then the nearest in reach, then the nearest."""
    near = [t for t in targets if t.platform == platform.id and abs(t.x - x) < shots.SHOT_RANGE]
    if near:
        return min(near, key=lambda t: abs(t.x - x))
    in_reach = [t for t in targets if shots.can_shoot(t.x - x, t.y - y)]
    hurt = [t for t in in_reach if t.bar is not None]  # finish the ones already hit first
    if hurt:
        return min(hurt, key=lambda t: (t.bar, abs(t.x - x)))
    return min(in_reach or targets, key=lambda t: abs(t.x - x) + 2 * abs(t.y - y))


def decide(x: float, y: float, platform: Platform, target: Target, threats: list[Target]) -> tuple:
    """What to do about `target` from map (x, y) on `platform`: ('throw', direction),
    ('walk', direction, units), or ('skip', reason) when no spot on her platform is in reach.
    `threats` are the other monsters on her platform. Never walks off her platform."""
    dx, dy = target.x - x, target.y - y
    toward = RIGHT if dx > 0 else LEFT
    mine = threats + ([target] if target.platform == platform.id else [])
    # With monsters on her platform, stay farther from its ends: a hit knocks her back.
    margin = shots.EDGE_MARGIN + (KNOCKBACK_ROOM if mine else 0)
    lo, hi = platform.left + margin, platform.right - margin
    close = [t for t in mine if abs(t.x - x) < TOO_CLOSE]
    if close:
        # Step away from whatever is closest, unless that walks into another monster or off the end.
        nearest = min(close, key=lambda t: abs(t.x - x))
        away = LEFT if nearest.x > x else RIGHT
        room = (hi - x) if away == RIGHT else (x - lo)
        blocked = any(0 < (t.x - x) * (1 if away == RIGHT else -1) < STEP_AWAY + TOO_CLOSE for t in mine)
        if not blocked and room >= 40:
            return ("walk", away, min(STEP_AWAY, room))
        toward_it = RIGHT if nearest.x > x else LEFT
        if abs(nearest.x - x) < shots.MIN_SHOT:
            # Cornered, and too close to throw (the claw would only swipe): jump over it if there's
            # ground beyond it with no other monster there.
            land = x + (JUMP_OVER if toward_it == RIGHT else -JUMP_OVER)
            if lo <= land <= hi and not any(abs(t.x - land) < 60 for t in mine if t is not nearest)                     and abs(land - x) > abs(nearest.x - x) + 30:
                return ("jump", toward_it)
        return ("throw", toward_it)  # cornered: fight
    if shots.can_shoot(dx, dy):
        return ("throw", toward)
    if not -shots.SHOT_UP <= dy <= shots.SHOT_DOWN:
        return ("skip", "not level enough with her platform")
    # Walk to the nearest spot on her platform that has the target in reach (either side of it).
    goals = [min(max(target.x + d, lo), hi)
             for d in (-(shots.SHOT_RANGE - 30), -(shots.MIN_SHOT + 60), shots.MIN_SHOT + 60, shots.SHOT_RANGE - 30)]
    goals = [g for g in goals if shots.can_shoot(target.x - g, dy, reach=shots.SHOT_RANGE + 15)]
    if not goals:
        return ("skip", "out of reach from this platform")
    goal = min(goals, key=lambda g: abs(g - x))
    direction = RIGHT if goal > x else LEFT
    # A monster on her platform in the way gets shot first.
    ahead = [t for t in threats if 0 < (t.x - x) * (1 if direction == RIGHT else -1) < abs(goal - x) + TOO_CLOSE]
    if ahead:
        first = min(ahead, key=lambda t: abs(t.x - x))
        return ("throw", RIGHT if first.x > x else LEFT)
    return ("walk", direction, min(abs(goal - x), MAX_WALK))
