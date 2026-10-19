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
from .loot import ABOVE, BELOW, Item, LootDetector, says_full
from .mapdata import SCREEN_SCALE, MapModel, Platform
from .mobs import SCALE, Mob, MobDetector
from .paths import LAB
from .terrain import MINIMAP_BOX, find_ropes, her_dot

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
MAX_WALK = 300     # longest walk between two looks
TRACK_SLACK = 85   # how far a target may move between two looks (knockback) and still be the same monster
SEARCH_SLACK = 150  # ... and how far it's looked for once it has vanished
LOST_AFTER = 3.0   # s a target may stay unseen (without an EXP message) before she gives up on it
HIDDEN_FOR = 1.8   # s she keeps throwing at a monster she was throwing at once it can't be seen: a hit
                   # one flinches under its damage numbers and the detector loses it for 1-3 s, and
                   # she used to stop throwing until it showed again
EMPTY_PATIENCE = 0.4  # s a fight waits when no monster is seen at all (not even out of reach)
FADING = 1.2       # s a killed monster may still be seen fading out where it died: not thrown at
JUMP_BACK_ROOM = 110  # room she needs behind her (from her platform's end, less the margin) to jump back
JUMP_BACK_CLEAR = 160  # ... with no monster on her platform this close on that side
JUMP_BACK_SPAN = 95  # how far a jump back carries her (85 measured)
JB_RUN_MS = 80     # jump back: the arrow away is held this long before the jump, to get her moving
JB_TURN_MS = 230   # ... and the arrow toward the monster is pressed this long after the start, in mid-air
PILE_PRESSES = 4   # pick-up presses standing on a kill's drops: one item per press, and a pig drops several
LOOT_REACH = 60    # drops spread about this far around where the monster died
PILE_GAP = 15      # drops closer together than this are one pile: picked up standing in one place
STOOD_ON = 15      # she stood on a drop when she stopped within this of it (walks stop ~10 units off)
MOP_UP = 3         # looks after a sweep for drops still on her platform, going back for them
SETTLE = 3.0       # s a kill's drops take to land and show: recorded at 2 fps, they lay on the ground about
                   # 2 s after her EXP went up (the death animation, then they pop up and fall)
EMPTY_LOOKS = 1.5  # s between two looks that must both see a remembered drop spot empty before it's
                   # forgotten: monsters are drawn over drops, and a pig walking over coins hid them
UNPICKABLE_FOR = 90  # s drops she failed to pick up are left alone (they vanish after a while anyway)
FAILED_PICKS = 3   # items in a row that stayed under her with no item picked up: her inventory is full
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


@dataclass(frozen=True)
class Loot:
    """An item or mesos seen on the ground, placed on the map."""
    item: Item
    x: float         # map units
    y: float
    platform: int    # the platform it lies on

    @property
    def mesos(self) -> bool:
        return self.item.mesos


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
    loot: list[Loot] = field(default_factory=list)  # drops seen on the ground (Hunter.see_loot)
    looked: list[tuple[int, float, float]] = field(default_factory=list)  # where loot was looked for:
                                                                          # (platform, map x from, to)

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

    def screen_y(self, map_y: float) -> float:
        """Half-size screen y of a map y (the camera doesn't scale, so offsets from her hold anywhere)."""
        return self.floor_y + (map_y - self.where.y) / UNITS_PER_HALF_PX

    def map_x(self, screen_x: float) -> float:
        return self.where.x + (screen_x - self.her[0]) * UNITS_PER_HALF_PX

    def in_view(self, platform: int, x: float) -> bool:
        """Whether loot was looked for at map x on that platform in this look."""
        return any(p == platform and lo <= x <= hi for p, lo, hi in self.looked)

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


def touch_popup(frame: np.ndarray, her: tuple[int, int]) -> int:
    """Purple pixels above her head (half-size frame): the "MISS" a thief's evasion shows when a monster
    touches her without damage, and the numbers of damage she takes, are both purple, and nothing
    else on screen is. A count of 40+ means a monster touched her."""
    x, y = her
    box = frame[max(0, y - 120):max(0, y - 15), max(0, x - 70):x + 70].astype(np.int16)
    if box.size == 0:
        return 0
    r, g, b = box[..., 0], box[..., 1], box[..., 2]
    return int(((r > 120) & (b > 150) & (g < 110) & (b - g > 80) & (r - g > 50)).sum())


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

    def __init__(self, title: str, tag_template: np.ndarray, detector: MobDetector | None = None,
                 loot: LootDetector | None = None):
        self.title = title
        self.small_tag = tag_template  # half-size template from calibration, used to find her once
        self.detector = detector or MobDetector()
        self._loot = loot
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

    @property
    def loot(self) -> LootDetector:
        """The loot detector, made (from the game's item pictures) on first use."""
        if self._loot is None:
            self._loot = LootDetector()
        return self._loot

    def messages(self) -> str:
        """The pickup log at the bottom right, read by OCR (~0.4 s): "Leather x1 earned. (Etc)", and what
        the game says when a pick-up fails. Empty if the OCR packages (environment.yml) aren't installed."""
        try:
            from maplehelper.ocr import LOOT, shared
        except ImportError:
            return ""
        area = window.client_rect(window.find_game(self.title))
        x, y, w, h = LOOT.region.box((0, 0, area["width"], area["height"]))
        shot = np.asarray(self.sct.grab({"left": area["left"] + x, "top": area["top"] + y, "width": w, "height": h}))
        rows = shared().read(np.ascontiguousarray(shot[:, :, :3]), min_height=LOOT.min_height) or []  # BGR
        return "\n".join(r.text for r in rows)

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
    """Loot left where a monster died, or seen on the ground, to be collected (by the tour) when it's off
    her platform."""
    x: float
    y: float
    platform: int | None
    since: float
    mesos: bool = False  # mesos were seen there (still worth a trip with her inventory full)
    seen: bool = False   # seen on the ground (else: where a kill of hers died)
    empty_since: float | None = None  # when a look first saw the spot empty (see EMPTY_LOOKS)


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
        self.inventory_full = False  # pick-up failed for want of room: items are left, mesos still taken
        self.unpickable: list[tuple[float, int, float]] = []  # (x, platform, when) of drops she couldn't take
        self._failed_picks = 0  # items in a row that stayed under her (see FAILED_PICKS)
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

    def jump_throw(self, direction: str) -> None:
        """Face `direction`, jump straight up and attack in mid-air: in the air the attack throws a
        star even at point-blank range, where on the ground the claw only swipes (9-11 damage against
        30-70 for a star, seen with slimes crowding her in the basin). The macro's F1 combo, with its
        timing (jump, then the attack 100 ms after the jump key)."""
        self._wait_lock()
        key = (self.keys.lucky_seven if self.skill and self.keys.lucky_seven else self.keys.attack).name
        jump = self.keys.jump.name
        events, t = [], 0
        if self.facing != direction:
            arrow = self._arrow(direction)
            events, t = [[0, arrow, "down"], [MOVEMENT.tap_turn_ms, arrow, "up"]], MOVEMENT.tap_turn_ms + 30
        events += [[t, jump, "down"], [t + 40, jump, "up"], [t + 100, key, "down"], [t + 140, key, "up"]]
        self._do(events, "jump_throw")
        self.facing = direction
        self._lock_until = time.monotonic() + (COMBAT.attack_lock_ms + 160) / 1000
        self.report.throws += 1

    def jump_back(self, direction: str) -> None:
        """At a monster too close to throw at (on `direction`'s side): jump away from it, turn toward it
        in mid-air and throw. Thrown at point-blank range, from the ground or straight up, stars often
        missed; from the air she's some 40-60 units farther off, and out of its way. The caller makes
        sure there's room behind her."""
        self._wait_lock()
        key = (self.keys.lucky_seven if self.skill and self.keys.lucky_seven else self.keys.attack).name
        away = self._arrow(LEFT if direction == RIGHT else RIGHT)
        toward, jump = self._arrow(direction), self.keys.jump.name
        events = [[0, away, "down"], [JB_RUN_MS, jump, "down"], [JB_RUN_MS + 40, jump, "up"],
                  [JB_TURN_MS, away, "up"], [JB_TURN_MS, toward, "down"], [JB_TURN_MS + 40, toward, "up"],
                  [JB_TURN_MS + 60, key, "down"], [JB_TURN_MS + 100, key, "up"]]
        self._do(events, "jump_back")
        self.facing = direction
        # Locked until the attack's lock ends, which is after she lands.
        self._lock_until = time.monotonic() + (JB_TURN_MS + 60 + COMBAT.attack_lock_ms) / 1000
        self.report.throws += 1

    def pick_here(self, presses: int = PILE_PRESSES) -> None:
        """Stand and press pick-up `presses` times, 120 ms apart: one item per press."""
        if self.keys.pick_up is None:
            return
        self._wait_lock()
        self._do(self._pickups(0, presses * 120, every=120), "pick")

    def walk(self, direction: str, units: float, pick: bool = True) -> None:
        """Walk `units` that way, pressing pick-up all the while (anything she passes over is looted)."""
        from .moves import hold_ms
        self._wait_lock()
        arrow, ms = self._arrow(direction), hold_ms(units)
        events = [[0, arrow, "down"], [ms, arrow, "up"]]
        if pick and self.keys.pick_up is not None:
            events += self._pickups(0, ms + 200)
        self._do(sorted(events, key=lambda e: e[0]), "walk")
        self.facing = direction
        time.sleep(MOVEMENT.stop_slide_ms / 1000)

    def _pickups(self, start: int, end: int, every: int = 150) -> list[list]:
        name = self.keys.pick_up.name
        return [e for t in range(start, end, every) for e in ([t, name, "down"], [t + 40, name, "up"])]

    def drink(self) -> None:
        if self.keys.hp_potion is None:
            return
        self._wait_lock()
        key = self.keys.hp_potion.name
        self._do([[0, key, "down"], [40, key, "up"]], "potion")
        self._last_potion = time.monotonic()
        self.report.potions += 1

    # --- looking ----------------------------------------------------------------------------

    def look(self, mobs: bool = True, wide: bool = False, loot: bool = False) -> Sight:
        """Her map position from the minimap dot; her screen position from her name tag, or from the
        dot and the camera when the tag is hidden; (if `mobs`) the monsters she could shoot, or (if
        `wide`) every monster on screen; and (if `loot`) the drops on her platform, or (if `wide`) on
        every platform in view."""
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
            if wide:
                s.mobs = self.eyes.detector.detect_wide(s.frame, (0, int(s.frame.shape[0] * 0.88)))
            else:
                s.mobs = self.eyes.detect(s)
            s.targets = [s.place(m, self.model) for m in s.mobs]
        if loot:
            s.loot = self.see_loot(s, wide)
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

    def see_loot(self, s: Sight, wide: bool = False) -> list[Loot]:
        """Drops lying on her platform (or, if `wide`, on every platform in view), placed on the map. Only
        a strip just above each platform's top is searched: drops lie on the ground. With her inventory
        full only mesos are looked for. Where it looked is noted in `s.looked`."""
        if s.her is None or s.where is None:
            return []
        h, w = s.frame.shape[:2]
        bottom = int(h * 0.88)  # the bottom bar starts below
        mini_x, mini_y = MINIMAP_BOX[2] // SCALE, MINIMAP_BOX[3] // SCALE
        detector = self.eyes.loot
        platforms = self.model.platforms if wide else [p for p in (s.platform,) if p is not None]
        found: list[tuple[float, Loot]] = []  # (px between its bottom and the platform's top, loot)
        for p in platforms:
            for x0, x1, high, low in _level_runs(p):
                left, right = max(0.0, s.screen_x(x0) - 4), min(float(w), s.screen_x(x1) + 4)
                top, under = max(0.0, s.screen_y(high) - ABOVE), min(float(bottom), s.screen_y(low) + BELOW)
                if top < mini_y and left < mini_x:
                    left = max(left, float(mini_x))
                if right - left < 12 or under - top < detector.height:
                    continue
                s.looked.append((p.id, s.map_x(left), s.map_x(right)))
                box = (int(left), int(top), int(right), int(under))
                for it in detector.detect(s.frame, box, mesos_only=self.inventory_full):
                    x = min(max(s.map_x(it.x), p.left), p.right)
                    found.append((abs(it.bottom - s.screen_y(p.y_at(x))), Loot(it, x, p.y_at(x), p.id)))
        # Platforms close above one another share strips: an item goes to the one it lies on.
        kept: list[Loot] = []
        for _, loot in sorted(found, key=lambda f: f[0]):
            if not any(abs(loot.item.x - k.item.x) <= 4 and abs(loot.item.y - k.item.y) <= 4 for k in kept):
                kept.append(loot)
        return kept

    def wanted_loot(self, loot: list[Loot]) -> list[Loot]:
        """The drops worth going for: not ones she failed to pick up, and only mesos with her inventory full."""
        now = time.monotonic()
        self.unpickable = [u for u in self.unpickable if now - u[2] < UNPICKABLE_FOR]
        return [d for d in loot if (d.mesos or not self.inventory_full)
                and not any(p == d.platform and abs(x - d.x) <= STOOD_ON for x, p, _ in self.unpickable)]

    def remember_loot(self, s: Sight) -> None:
        """Note what a look (with `loot`) saw, for the tour to collect: drops seen are remembered (a pile per
        LOOT_REACH). A remembered spot is forgotten (taken by someone, or by her on the way) once its drops
        have had time to land (SETTLE) and two looks EMPTY_LOOKS apart saw nothing there: one look can
        miss drops a monster stands in front of."""
        now = time.monotonic()
        seen = self.wanted_loot(s.loot)
        near = lambda d, loot: d.platform == loot.platform and abs(d.x - loot.x) <= LOOT_REACH
        kept = []
        for d in self.pending_loot:
            if now - d.since < SETTLE or not s.in_view(d.platform, d.x) or any(near(d, loot) for loot in seen):
                d.empty_since = None
            elif d.empty_since is None:
                d.empty_since = now
            elif now - d.empty_since >= EMPTY_LOOKS:
                continue
            kept.append(d)
        if gone := len(self.pending_loot) - len(kept):
            self.log(f"  {gone} remembered drop spot(s) seen empty twice: forgotten")
        self.pending_loot = kept
        self._note_loot(seen)

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
        """Pick up the drops now if they're on her platform and no monster stands by them; otherwise
        remember them (pending_loot) for later. Drops seen around the spot are walked to directly. Where
        a kill died she sweeps even when nothing shows: drops can still be falling, or hidden behind a
        monster, and the trip there cost far more. Drops seen earlier that are gone are done with.
        Returns True when done with them."""
        s = self.look(loot=True)
        if s.where is None or s.platform is None or drops.platform != s.platform.id:
            self.pending_loot.append(drops)
            return False
        if any(t.platform == drops.platform and abs(t.x - drops.x) < TOO_CLOSE for t in s.targets):
            self.log("  drops guarded by a monster: later")
            self.pending_loot.append(drops)
            return False
        seen = [d for d in self.wanted_loot(s.loot) if d.platform == drops.platform
                and abs(d.x - drops.x) <= 2 * LOOT_REACH]
        if seen:
            self.sweep_through(s, [Drops(d.x, d.y, d.platform, s.t) for d in seen], margin=0)
        elif drops.seen and time.monotonic() - drops.since >= SETTLE and s.in_view(drops.platform, drops.x):
            self.log(f"  the drops seen at x {drops.x:.0f} are gone")
        else:
            self.sweep_through(s, [drops])
        return True

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
              seconds: float = 120, patience: float = 1.5, approach: bool = True, finish: bool = False,
              far: bool = False) -> Report:
        """Fight like a player: a star hits the nearest monster in reach on the side she faces, so
        she keeps throwing toward the side with the most monsters in reach, one throw per attack
        cooldown, looking around during the cooldown, never idle while one is in reach. She keeps
        them all on one side: one too close makes her step back (or, cornered, fight from where
        she is), never cross over them. With nothing in reach she walks (if `approach`) just near
        enough to the nearest, picking up as she goes; then sweeps up what fell on her platform.
        Kills are her EXP going up. Targets: monsters (of `kinds`) on the `ground` platforms, or
        anywhere if None, and any on her own platform. Stops after `max_kills`, after `seconds`, or
        when nothing has been in reach for `patience` seconds (`EMPTY_PATIENCE` when none is seen at
        all). A monster she was throwing at that goes out of sight is still thrown at for
        `HIDDEN_FOR` seconds: hit, it flinches under its damage numbers, and the detector loses it.
        With `finish`, reaching `max_kills` doesn't end the fight while monsters are still in reach:
        she finishes those (without walking after others), as stopping halfway gave up a good spot.
        Monsters of other kinds on her platform that come close are in her way, and fought too. With
        `far`, monsters (of `kinds`) on any platform in reach are fought too: her darts reach far, and
        their drops are left to the tour (pending_loot). A star hits the first monster in its path,
        whatever its kind, so kills are matched against everything in the path of the throw."""
        deadline, unseen = time.monotonic() + seconds, 0
        thrown: list[list] = []  # [monster, when last seen]: what she's thrown at, kept while hidden
        dying: list[tuple[Target, float]] = []  # kills, and when: still seen fading out for a moment
        calm_since: float | None = None
        start: int | None = None  # the platform she's on; leaving it means she was knocked off
        exp0 = seen_exp = self.report.exp_kills
        aimed: list[Target] = []  # in reach at the last throw: the dead are found among them
        deaths: list[Drops] = []  # kills on her platform, to sweep up
        last_act, last_hidden = None, False
        while time.monotonic() < deadline:
            done = self.report.exp_kills - exp0 >= max_kills
            if done and not finish:
                break
            s = self.look()
            if s.where is None or s.platform is None:
                unseen += 1
                if unseen >= 10:
                    self.report.stop = "lost track of her"
                    return self.report
                time.sleep(0.2)
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
            here, now = s.platform, time.monotonic()
            wanted = [t for t in s.targets if (kinds is None or t.kind in kinds)
                      and (ground is None or t.platform in ground or t.platform == here.id
                           or (far and shots.can_shoot(t.x - s.where.x, t.y - s.where.y)))]
            # Any monster coming close on her platform is in her way (a snail too), and fought.
            wanted += [t for t in s.targets if t not in wanted and t.platform == here.id
                       and abs(t.x - s.where.x) < TOO_CLOSE + 40]
            # A kill is still seen fading out for a moment: throwing at it (and then at its "hidden"
            # ghost) wasted throws after most kills.
            dying = [(d, t) for d, t in dying if now - t < FADING]
            fading = [t for t in s.targets if any(_same(d, [t]) is t for d, _ in dying)]
            wanted = [t for t in wanted if t not in fading]
            for entry in thrown:
                if (same := _same(entry[0], [t for t in s.targets if t not in fading])) is not None:
                    entry[0], entry[1] = same, now
            # Her EXP went up since the last look: those are the kills, among what she was throwing at.
            for _ in range(self.report.exp_kills - seen_exp):
                # Among what the last throw aimed at and what she's thrown at lately: the kill can come
                # in after she's turned to the next one. A kill with no drop spot left its drops.
                dead = _dead_one(aimed + [e[0] for e in thrown if not any(e[0] is a for a in aimed)],
                                 s.targets, s.where.x)
                if dead is None:
                    self.log("  a kill (EXP up) she can't place: its drops are left")
                    break
                if dead in aimed:
                    aimed.remove(dead)
                thrown = [e for e in thrown if e[0] is not dead and _same(dead, [e[0]]) is None]
                dying.append((dead, now))
                wanted = [t for t in wanted if _same(dead, [t]) is not t]
                self.report.kills[dead.kind] = self.report.kills.get(dead.kind, 0) + 1
                drops = Drops(dead.x, dead.y, dead.platform, time.monotonic())
                (deaths if dead.platform == here.id else self.pending_loot).append(drops)
                self.log(f"  {dead.kind} down at x {dead.x:.0f} (platform {dead.platform})")
            seen_exp = self.report.exp_kills
            thrown = [e for e in thrown if now - e[1] < HIDDEN_FOR]
            hidden = [e[0] for e in thrown if e[1] != now]  # not seen this look: still there, hit
            fight = wanted + hidden
            act = stance(s.where.x, s.where.y, here, fight, approach and not done, self.facing,
                         can_jump=lambda a, b: _open_above(self.model, here, a, b))
            if act[:2] != (last_act or ())[:2] or (hidden and act[0] != "wait" and not last_hidden):
                self.log(f"  {_describe(act, fight, s.where.x, hidden)} (HP {s.hp:.0%})")
            last_act, last_hidden = act, bool(hidden)
            if act[0] in ("throw", "jump_throw", "jump_back"):
                calm_since = None
                # Everything in the star's path, whatever its kind: a snail in front of a far pig
                # takes the star (and its EXP isn't the pig's kill).
                aimed = [t for t in [t for t in s.targets if t not in fading] + hidden
                         if (t.x > s.where.x) == (act[1] == RIGHT)
                         and (abs(t.x - s.where.x) <= shots.STAR_REACH if t.platform == here.id
                              else shots.can_shoot(t.x - s.where.x, t.y - s.where.y, shots.STAR_REACH))]
                thrown += [[t, now] for t in aimed if not any(e[0] is t for e in thrown)]
                {"throw": self.throw, "jump_throw": self.jump_throw, "jump_back": self.jump_back}[act[0]](act[1])
            elif act[0] in ("back", "approach"):
                calm_since = None
                self.walk(act[1], act[2])
            else:
                if deaths:  # sweep up what fell on her platform, all in one walk
                    self.sweep_through(s, deaths)
                    deaths = []
                    continue
                if self.pending_loot:
                    self.loot_pending(s)
                if wanted and not calm_since:
                    self.report.out_of_reach += 1  # there are some, just not reachable from here
                if done:  # finished what was in reach after the kill count
                    break
                calm_since = calm_since or time.monotonic()
                wait = patience if fight else min(patience, EMPTY_PATIENCE)  # none seen: move on soon
                if time.monotonic() - calm_since >= wait:
                    self.report.stop = "nothing left in reach"
                    return self.report
                time.sleep(0.1)
        if deaths:
            self.sweep_after(deaths)
        self.report.stop = "kill limit" if self.report.exp_kills - exp0 >= max_kills else "time limit"
        return self.report

    def sweep_after(self, deaths: list[Drops]) -> None:
        """Sweep up a fight's drops once it's over. The fight often ends mid-jump (a jump-throw), where
        a look finds her on no platform, and the sweep used to be skipped: those drops were left. Looks
        again until she has landed; drops on another platform than hers are left to the tour."""
        for _ in range(8):
            s = self.look(mobs=False)
            if s.where is not None and s.platform is not None:
                self.pending_loot += [d for d in deaths if d.platform != s.platform.id]
                self.sweep_through(s, deaths)
                return
            time.sleep(0.15)
        self.pending_loot += deaths

    def sweep_through(self, s: Sight, deaths: list[Drops], margin: float = LOOT_REACH) -> None:
        """Pick up the drops on her platform (`_walk_piles`), then look again for any still there and go
        back for them (`mop_up`)."""
        self.mop_up(self._walk_piles(s, deaths, margin))

    def _walk_piles(self, s: Sight, deaths: list[Drops], margin: float = LOOT_REACH) -> list[float]:
        """Walk over the drops on her platform: from the nearer end of them to the farther one (and
        `margin` on past each end: a kill's drops spread around where it died), pressing pick-up all
        the way, and standing on each pile for PILE_PRESSES presses. A pick-up press takes one item, a
        pig drops several, and walking past at full speed got one or two. Returns where she stood."""
        p = s.platform
        clamp = lambda v: min(max(v, p.left + shots.EDGE_MARGIN), p.right - shots.EDGE_MARGIN)
        piles = _piles([clamp(d.x) for d in deaths if d.platform == p.id])
        if not piles:
            return []
        x = s.where.x
        if abs(x - piles[-1]) < abs(x - piles[0]):
            piles.reverse()
        first, last = piles[0], piles[-1]
        sign = 1 if last >= first else -1
        stops = [(clamp(first - sign * margin), False)] + [(g, True) for g in piles] + [(clamp(last + sign * margin), False)]
        for goal, pile in stops:
            d = goal - x
            if abs(d) >= 10:
                self.walk(RIGHT if d > 0 else LEFT, abs(d))
                x = goal
            if pile:
                self.pick_here()
        self.report.loot_runs += 1
        return piles

    def mop_up(self, stood: list[float] | None = None, rounds: int = MOP_UP) -> None:
        """Look again after a sweep for drops still on her platform, and go back for them: walking past
        at full speed, or stopping a little off a pile, left some. A drop she has stood on twice that's
        still there can't be picked up (`_cant_pick`); drops with a monster by them wait for later.
        `stood`: where the sweep just had her stand picking up."""
        stood, here, tried = list(stood or []), None, []  # tried: items she went back for last round
        for _ in range(rounds):
            time.sleep(0.3)  # a picked-up drop flies to her for a moment
            s = self.look(loot=True)
            if s.where is None or s.platform is None or here not in (None, s.platform.id):
                return  # in the air, or knocked off: whatever is left, the tour sees
            here = s.platform.id
            seen = [d for d in s.loot if d.platform == here]
            if any(not any(abs(d.x - t.x) <= STOOD_ON for d in seen if not d.mesos) for t in tried):
                self._failed_picks = 0  # an item she went back for is gone: she picked it up
            left = self.wanted_loot(seen)
            stuck = [d for d in left if sum(abs(x - d.x) <= STOOD_ON for x in stood) >= 2]
            if stuck:
                self._cant_pick(stuck)
                left = [d for d in left if d not in stuck]
            guarded = [d for d in left if any(t.platform == here and abs(t.x - d.x) < TOO_CLOSE for t in s.targets)]
            if guarded:
                self.log(f"  {len(guarded)} drop(s) by a monster: later")
                self._note_loot(guarded)
                left = [d for d in left if d not in guarded]
            if not left:
                return
            self.log(f"  {_count(left)} still on platform {here}: back for them")
            tried = [d for d in left if not d.mesos]
            stood += self._walk_piles(s, [Drops(d.x, d.y, here, s.t) for d in left], margin=0)

    def _cant_pick(self, stuck: list[Loot]) -> None:
        """Drops she stood on twice, pressing pick-up, that are still there: left alone for a while. Mesos
        always fit, so those are someone else's (drops belong to the killer for a while). Items may be
        too, unless the pickup log says her inventory is full, or items keep staying under her
        (FAILED_PICKS times in a row, with none picked up in between): then items are left, mesos taken."""
        now = time.monotonic()
        self.unpickable += [(d.x, d.platform, now) for d in stuck]
        self.log(f"  can't pick up {_count(stuck)} at x {', '.join(f'{d.x:.0f}' for d in stuck)}: left")
        if self.inventory_full or all(d.mesos for d in stuck):
            return
        self._failed_picks += 1
        read = getattr(self.eyes, "messages", None)
        text = read() if read else ""
        if says_full(text):
            why = f"the game says so: {' / '.join(line for line in text.splitlines() if says_full(line))}"
        elif self._failed_picks >= FAILED_PICKS:
            why = f"items stayed under her {self._failed_picks} times in a row"
        else:
            return
        self.inventory_full = True
        self.log(f"  her inventory is full ({why}): leaving items, picking up mesos only")

    def _note_loot(self, seen: list[Loot]) -> None:
        """Remember drops seen, a pile per LOOT_REACH, for later (the tour collects them)."""
        for loot in seen:
            pile = next((d for d in self.pending_loot if d.platform == loot.platform
                         and abs(d.x - loot.x) <= LOOT_REACH), None)
            if pile is None:
                self.pending_loot.append(Drops(loot.x, loot.y, loot.platform, time.monotonic(), loot.mesos, seen=True))
            else:
                pile.mesos = pile.mesos or loot.mesos


def _piles(xs: list[float]) -> list[float]:
    """Drop spots closer together than PILE_GAP made one pile (their middle), in order."""
    piles: list[list[float]] = []
    for x in sorted(xs):
        if piles and x - piles[-1][-1] < PILE_GAP:
            piles[-1].append(x)
        else:
            piles.append([x])
    return [round(sum(p) / len(p)) for p in piles]


def _count(loot: list[Loot]) -> str:
    """"2 mesos, 1 leather"."""
    kinds: dict[str, int] = {}
    for d in loot:
        kinds[d.item.kind] = kinds.get(d.item.kind, 0) + 1
    return ", ".join(f"{n} {k}" for k, n in kinds.items())


def _level_runs(p: Platform, flat: float = 20) -> list[tuple[float, float, float, float]]:
    """A platform as stretches of nearly level ground: (x from, x to, highest y, lowest y), map units."""
    runs: list[tuple[float, float, float, float]] = []
    for seg in p.segments:
        hi, lo = min(seg.y1, seg.y2), max(seg.y1, seg.y2)
        if runs and max(runs[-1][3], lo) - min(runs[-1][2], hi) <= flat:
            x0, _, top, low = runs[-1]
            runs[-1] = (x0, seg.x2, min(top, hi), max(low, lo))
        else:
            runs.append((seg.x1, seg.x2, hi, lo))
    return runs


def _open_above(model: MapModel, platform: Platform, x0: float, x1: float) -> bool:
    """Whether she can jump anywhere between x0 and x1 on `platform` and land back on it: no other
    platform overhead within her jump height (she'd go up through it and land on top)."""
    from .route import JUMP_HEIGHT
    lo, hi = sorted((x0, x1))
    for q in model.platforms:
        if q.id == platform.id or q.right < lo or q.left > hi:
            continue
        mid = min(max((lo + hi) / 2, q.left), q.right)
        if 0 < platform.y_at(min(max(mid, platform.left), platform.right)) - q.y_at(mid) <= JUMP_HEIGHT + 5:
            return False
    return True


def _same(t: Target, now: list[Target]) -> Target | None:
    """The monster in `now` that is `t` seen again (same kind, near where it was), if any."""
    near = [o for o in now if o.kind == t.kind and abs(o.x - t.x) < TRACK_SLACK and abs(o.y - t.y) < 40]
    return min(near, key=lambda o: abs(o.x - t.x)) if near else None


def _dead_one(aimed: list[Target], now: list[Target], her_x: float) -> Target | None:
    """Which of the monsters she was throwing at died: one no longer seen (nearest her), else the
    nearest (it may still be seen fading out)."""
    gone = [t for t in aimed if not any(o.kind == t.kind and abs(o.x - t.x) < TRACK_SLACK
                                        and abs(o.y - t.y) < 40 for o in now)]
    pool = gone or aimed
    return min(pool, key=lambda t: abs(t.x - her_x)) if pool else None


def stance(x: float, y: float, platform: Platform, wanted: list[Target], approach: bool = True,
           facing: str | None = None, can_jump=None) -> tuple:
    """What to do from map (x, y) on `platform` with these monsters around: ('throw', side),
    ('jump_back', side) at one too close for a throw from the ground (jump away, throw back from the
    air), ('jump_throw', side) the same when cornered (straight up), ('approach', side, units) to
    bring the nearest into reach, or ('wait',). She keeps working the side she faces while it still
    has monsters in reach (switching sides every throw spread the damage over both), and never walks
    past a monster. `can_jump(x0, x1)` says whether jumping over that stretch keeps her on her
    platform: under a platform within her jump height, a jump lands her up on it."""
    can_jump = can_jump or (lambda a, b: True)
    mine = [t for t in wanted if t.platform == platform.id]
    margin = shots.EDGE_MARGIN + (KNOCKBACK_ROOM if mine else 0)
    lo, hi = platform.left + margin, platform.right - margin
    side_of = lambda t: RIGHT if t.x > x else LEFT
    close = [t for t in mine if abs(t.x - x) < shots.MIN_SHOT + 10]
    reach = [t for t in wanted if shots.can_shoot(t.x - x, t.y - y)]
    if close:
        on_facing = [t for t in close if side_of(t) == facing]
        nearest = min(on_facing or close, key=lambda t: abs(t.x - x))
        side = side_of(nearest)
        # Jump away from it and throw back from the air, if there's room behind her and no monster
        # there; cornered, straight up.
        back = -1 if side == RIGHT else 1
        room = (x - lo) if back < 0 else (hi - x)
        behind = [t for t in mine if 0 < (t.x - x) * back < JUMP_BACK_CLEAR]
        if room >= JUMP_BACK_ROOM and not behind and can_jump(x, x + back * JUMP_BACK_SPAN):
            return ("jump_back", side)
        if can_jump(x, x):
            return ("jump_throw", side)
        # A platform just overhead: a jump lands her on it (on 20, five jump-throws at a snail put
        # her up on 19 five times). Step away, else throw from the ground.
        if room >= 40 and not behind:
            return ("back", RIGHT if back > 0 else LEFT, min(STEP_AWAY, room))
        return ("throw", side)
    if reach:
        if facing is not None and any(side_of(t) == facing for t in reach):
            return ("throw", facing)
        right = [t for t in reach if t.x > x]
        left = [t for t in reach if t.x <= x]
        score = lambda side: (len(side), -min((abs(t.x - x) for t in side), default=1e9))
        return ("throw", RIGHT if score(right) > score(left) else LEFT)
    # Off her level and too close sideways (right under or over her), a star misses: step away from
    # it, along her platform, to where it reaches (throwing at it from there used to go on and on).
    under = [t for t in wanted if t.platform != platform.id and -shots.SHOT_UP <= t.y - y <= shots.SHOT_DOWN
             and abs(t.x - x) < shots.min_dx(t.y - y)]
    if under:
        t = min(under, key=lambda t: abs(t.x - x))
        away = LEFT if t.x > x else RIGHT
        goal = t.x + (1 if away == RIGHT else -1) * (shots.min_dx(t.y - y) + 40)
        goal = min(max(goal, lo), hi)
        if abs(goal - x) >= 15 and shots.can_shoot(t.x - goal, t.y - y, reach=shots.SHOT_RANGE + 15):
            return ("back", away, abs(goal - x))
    if approach:
        level = [t for t in wanted if -shots.SHOT_UP <= t.y - y <= shots.SHOT_DOWN]
        if level:
            t = min(level, key=lambda t: abs(t.x - x))
            d = 1 if t.x > x else -1
            goal = min(max(t.x - d * (shots.SHOT_RANGE - 40), lo), hi)
            if (goal - x) * d > 15 and shots.can_shoot(t.x - goal, t.y - y, reach=shots.SHOT_RANGE + 15):
                return ("approach", RIGHT if d > 0 else LEFT, abs(goal - x))
    return ("wait",)


def _describe(act: tuple, wanted: list[Target], x: float, hidden: list[Target] = ()) -> str:
    if act[0] in ("throw", "jump_throw", "jump_back"):
        side = [t for t in wanted if (t.x > x) == (act[1] == RIGHT)]
        verb = {"throw": "throwing", "jump_throw": "jump-throwing", "jump_back": "jumping back, throwing"}[act[0]]
        name = lambda t: f"{t.kind} {t.x - x:+.0f}" + (" hidden" if any(t is h for h in hidden) else "")
        return f"{verb} {act[1]} ({', '.join(name(t) for t in side[:3])})"
    if act[0] in ("back", "approach"):
        return f"{'stepping away' if act[0] == 'back' else 'closing in'} {act[1]} {act[2]:.0f}"
    return "nothing in reach"


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
        # Cornered: fight from here. Jumping over it put the rest of the group on her other side.
        return ("throw", RIGHT if nearest.x > x else LEFT)
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
