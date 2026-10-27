"""The seams between gamelab and the game, faked over a sim.world.World on a fake clock.

- `Clock`: time.monotonic / sleep / time / perf_counter / strftime. Nothing takes time but sleeping and
  the fakes below (a look, a glance, a key plan), so a run is the same every time.
- `FakeEyes`: perception.eyes.Eyes over the world: look() -> an Observation with the screen fields
  filled in as localization (Senses.look, Scene.place, find_loot, pose.refine) inverts them (her name
  tag, the grass under her, the ropes in view, her minimap dot and the other player's), glance() and messages(). Monsters and
  drops are drawn into the frame as data its fake detectors read (Eyes.monsters, monsters_wide, items),
  and hurt monsters' HP bars as pixels, as the game draws them (Eyes.bar reads those); her EXP number as the bar shows
  it (perception.exp_number draws it with the game's glyphs: her level and EXP at the start, World.character, and what
  she has gained), read as in game.
  world.Noise can round the dot to whole pixels and hide her tag, her dot or her EXP number for a while.
- `FakePlayer`: what gamelab.control.client offers (submit, send, done, wait, cancel), running each plan's keys
  on the world: plans run one after another, `end_ms` makes one last longer, every key still held is let
  go at a plan's end, and cancel withdraws a plan not started yet or cuts a running one short. With `alarm_at` set
  the player sees the Lie Detector then, as the real one's watchdog would (client's check_alarm, alarm_up and
  alarm_age too): a plan running stops at it, and after it only what control.player.alarm_refusal lets through plays
  (her pet's food; her retreat's moves, cut at alarm.RETREAT_WINDOW_S).
- `patched`: puts them in place (and points every gamelab path under gamelab/lab at a temporary folder),
  restoring everything afterwards. Anything that would touch the real game, or write anywhere under the
  real gamelab/lab, raises SimViolation.
"""

import builtins
import io
import math
import os
import sys
import time
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from maplehelper.config import GameKeys
from maplehelper.engine import LEFT, RIGHT
from maplehelper.ocr.exp_table import EXP_TO_NEXT

from ..attack.choice import MP_STEPS
from ..behavior.pet import PetCare
from ..control import client
from ..localization.mapdata import MapModel
from ..localization.monsters import FLINCH_S
from ..localization.pose import GRASS_OVERHANG
from ..localization.pose import CAMERA_OFFSET, Camera, Localizer
from ..paths import LAB
from ..perception import exp_number
from ..perception.eyes import Eyes, Observation
from ..perception.heading import Reading
from ..perception.hud import bar_box, draw_hp_bar
from ..perception.loot import Item
from ..perception.messages import ChatLine
from ..perception.match import suppress_overlaps
from ..perception.minimap import OTHER_SHIFT, PICTURE_BASE, Marker
from ..perception.mobs import REFIND_PAD, Mob
from ..perception.terrain import ScreenRope
from ..screen import SCALE
from ..screen import UNITS_PER_HALF_PX
from ..screen import SCREEN_SCALE, STRETCHED
from ..perception.kinds import kind_of
from .world import HEADING_SPELL_S, LEVEL_19, POSE_SPELL_S, Noise, World, exp_pixels

START = 10_000.0          # the fake clock's start (monotonic() is never near 0 in real runs)
EPOCH = 1_800_000_000.0   # what time.time() reads at the start (a fixed date, for strftime)
FRAME = (540, 960)        # half-size game frame
SKY = (60, 140, 230)      # the frame's colour: nothing in it reads as an HP bar, a rope or ground
PLAY_BOTTOM = int(FRAME[0] * 0.88)  # the bottom bar starts below (screen.HUD_TOP)
# The minimap window's rows (game px) on a map whose picture is taller than it shows: on Tree Dungeon, Forest Up North
# II 112 of the picture's 183 (the window from y 72 to 184 at 1366x768; the study's frames). It scrolls the picture to
# keep her dot WINDOW_DOT_ROW rows from its top, stopping at the picture's ends. Rows are the picture's as the game draws
# the dots in it (_picture_row: 7.2 rows above the map's own row there, (y + centerY) / 16, as the dot's origin has
# it). On its frames of 2026-10-24 her dot was 49.5 rows down on 19 (first_000: the picture's corner at y 28, her dot
# at picture row 93.5; one frame: the study took the window to keep her dot in its middle, 56), and on 28 the window
# had stopped at the picture's bottom (the corner at y 1, her dot at row 130.5): from 28 the window shows y -998 and
# below (platforms 17 to 33), and nobody higher up is seen (the hour of live telemetry from 28, 2026-10-24: none of
# 137,645 other players' dots read above -1015, 1,902 within that px of -998; the study: -1001 and down). The first
# map's and West Rocky Mountain II's whole pictures show (their corners at y 72 on all their saved frames).
WINDOW_ROWS = {10002073: 112}
WINDOW_DOT_ROW = 49.5
# Where the game's camera keeps her up and down (map units from the screen's top), on a map where it was seen apart
# from pose.CAMERA_OFFSET's 489 (fitted on the first map): on Tree Dungeon, Forest Up North II 529, as on all 19 of the
# study's frames from 28 (screen y = map y + 579, the labelled monsters' feet on their footholds) and in the hour of
# live telemetry there (the view's top at -579 on 29,127 of 29,146 looks, 2026-10-24). On its one frame from 19 it
# was 545 (screen y = map y + 1184): not understood (the camera's lag after a move, or a dead band; not modelled).
# With 489 a look from 28 found the Horny Mushrooms on 31 above the HUD (feet at row 673), which the game hides
# behind the chat bar (rows 701-718 on the study's frames).
CAMERA_DOWN = {10002073: 529.0}
KINDS = ("pig", "slime", "snail", "orange-mushroom")  # what the real detector has templates for (on the first map;
                                                      # on another, the map's kinds too: detector_kinds)
ITEM_SIZE = {"mesos": (13, 13), "leather": (20, 18)}  # half-size px (w, h)
ITEM_HEIGHT = 22          # the loot detector's tallest template
# Seconds each fake takes (the real ones' cost as gamelab documents it; and a look must cost something,
# or a loop that only looks would never see the clock move).
LOOK_S, LOOK_LAG = 0.10, 0.04   # a look, and when in it the frame is taken (Observation.t)
DETECT_S = 0.03                 # monsters around her level
DETECT_WIDE_S = 0.25            # ... on the whole screen: a wide look takes ~0.35 s in all (navigator.LOOK_AHEAD)
LOOT_S = 0.02             # per strip searched
GLANCE_S = 0.035          # with Navigator.follow's 30 ms sleep, ~15 glances a second (as Navigator.follow says)
MESSAGES_S = 0.4
# The message bar (Eyes.chat_line): a frame from the feed and its pink px (a frame's wait, about), and the OCR of a line
# not read before (perception.messages: 0.28 s CPU a read live)
CHAT_S, CHAT_OCR_S = 0.035, 0.28
EXP_START = (LEVEL_19.level, LEVEL_19.exp)  # her level and EXP when a run starts on the first map (World.character's)
START_LATENCY = 0.01      # a plan starts this long after it's handed over (the player polls its inbox)
RESULT_LATENCY = 0.01     # ... and its result is seen this long after it ends


class FakePetCare(PetCare):
    """behavior.pet.PetCare on the simulated world: her Character Info's pet panel read off the world's pet
    (World.pet_fullness, to the whole point, as the panel shows it), no window opened or closed and no click sent; the
    Pet Food itself goes out as in game (Actions.feed_pet), and the fake clock is its clock."""

    def __init__(self, world, actions, eyes, clock: "Clock", log=print):
        super().__init__(actions, eyes, ocr=object(), log=log, clock=clock.time)
        self.world, self.fake_clock = world, clock

    def _open(self) -> int | None:
        self.world.advance(self.fake_clock.now)
        return int(self.world.pet_fullness())

    def _read(self) -> int | None:
        return self._open()

    def _close(self) -> None:
        pass


class SimViolation(RuntimeError):
    """Something tried to reach the real game (a window, a key) during a simulated run."""


class SimTimeout(BaseException):
    """A simulated run went past its time limit. Not an Exception: a loop that recovers from errors
    (`except Exception`) must not catch it and spin on, since every later step raises it again."""


class Clock:
    def __init__(self, start: float = START, limit: float | None = None, pace: float | None = None,
                 think: float = 0.0):
        self.start = self.now = start
        self.limit = limit  # s after the start; going past it raises SimTimeout
        self.pace = pace    # if set, simulated time also passes in real time, this many times faster
        # Thinking is free on the fake clock: only looks, glances, plans and sleeps take time. With `think`
        # set, the stack's own computing between two of those (measured with the real perf_counter) is
        # charged too, times `think` (1.0: as fast as this computer). Off by default: it makes runs vary.
        self.think = think
        self._mark: float | None = None
        # (period, fn): fn(t) at every frame the game shows, as time passes it (a continuous feed's frames:
        # localization.feed.SyncFeed.tick); None: no frames
        self.frames: tuple[float, Callable[[float], None]] | None = None
        self._next_frame = start

    def charge(self) -> None:
        """Charge the stack's computing since the last fake call (only with `think` set)."""
        if self.think and self._mark is not None:
            spent = _REAL["perf_counter"]() - self._mark
            self._mark = None
            self.advance_to(self.now + spent * self.think)

    def mark(self) -> None:
        """A fake call is over: the stack's own computing starts here."""
        if self.think:
            self._mark = _REAL["perf_counter"]()

    @property
    def elapsed(self) -> float:
        return self.now - self.start

    def advance_to(self, t: float) -> None:
        if t > self.now:
            if self.frames is not None:
                period, fn = self.frames
                while self._next_frame <= t + 1e-9:
                    frame, self._next_frame = self._next_frame, self._next_frame + period
                    if frame >= self.now:
                        self._to(frame)
                        fn(frame)
            self._to(t)

    def _to(self, t: float) -> None:
        if t > self.now:
            if self.pace:  # to watch a run on the mini-map as it happens
                _REAL["sleep"]((t - self.now) / self.pace)
            self.now = t
            if self.limit is not None and self.now - self.start > self.limit:
                raise SimTimeout(f"simulated run passed its {self.limit:.0f} s limit")

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.charge()
        self.advance_to(self.now + max(0.0, seconds))
        self.mark()

    def time(self) -> float:
        return EPOCH + self.elapsed

    def strftime(self, fmt: str, t=None) -> str:
        return _REAL["strftime"](fmt, t if t is not None else _REAL["gmtime"](self.time()))


_REAL = {"monotonic": time.monotonic, "sleep": time.sleep, "time": time.time, "perf_counter": time.perf_counter,
         "strftime": time.strftime, "gmtime": time.gmtime}


# --- perception --------------------------------------------------------------------------------------

class Frame(np.ndarray):
    """A half-size frame that knows what's drawn in it (the fakes' detectors read that, not pixels)."""
    screen: "Screen"


@dataclass
class Screen:
    mobs: list[Mob] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    heading: Reading = Reading(None, why="no pictures")  # which way she faces, as her picture shows it


class FakeDetector:
    """Eyes.detector: monsters drawn in a fake frame, found where a template would fit (the real one
    matches whole templates inside the rows searched)."""

    def __init__(self, kinds=KINDS, clock: Clock | None = None):
        self.templates = {k: [] for k in kinds}
        self.clock = clock
        self.fallbacks = 0

    def detect(self, frame: np.ndarray, rows: tuple[int, int] | None = None) -> list[Mob]:
        if self.clock is not None:
            self.clock.sleep(DETECT_S)
        return self._find(frame, rows)

    def _find(self, frame, rows) -> list[Mob]:
        top, bottom = rows or (0, frame.shape[0])
        found = [m for m in getattr(frame, "screen", Screen()).mobs
                 if m.kind in self.templates and top <= m.y - m.h // 2 and m.y + m.h // 2 <= bottom]
        return suppress_overlaps(found)

    def refind(self, frame: np.ndarray, box: tuple[int, int, int, int], kind: str, template: str = "",
               facing: int = 0, bounds: tuple[int, int, int, int] | None = None, fallback: bool = True) -> Mob | None:
        """A monster drawn of that kind wholly inside the window around `box` (REFIND_PAD), nearest its middle
        (what the template that matched it last would find there: `fallback` makes no difference)."""
        bx0, by0, bx1, by1 = bounds or (0, 0, frame.shape[1], frame.shape[0])
        x0, y0 = max(bx0, box[0] - REFIND_PAD), max(by0, box[1] - REFIND_PAD)
        x1, y1 = min(bx1, box[2] + REFIND_PAD), min(by1, box[3] + REFIND_PAD)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        found = [m for m in getattr(frame, "screen", Screen()).mobs
                 if m.kind == kind and x0 <= m.x - m.w // 2 and m.x + m.w // 2 <= x1
                 and y0 <= m.y - m.h // 2 and m.y + m.h // 2 <= y1]
        if not found:
            self.fallbacks += fallback  # the real one would have tried its kind's whole set
            return None
        return min(found, key=lambda m: (m.x - cx) ** 2 + (m.y - cy) ** 2)

    def detect_wide(self, frame: np.ndarray, rows: tuple[int, int], band: int = 120, step: int = 90) -> list[Mob]:
        if self.clock is not None:
            self.clock.sleep(DETECT_WIDE_S)
        hits = []
        for y0 in range(rows[0], max(rows[0] + 1, rows[1] - band + step), step):
            hits += self._find(frame, (y0, min(rows[1], y0 + band)))
        return suppress_overlaps(hits)


class FakeLoot:
    """Eyes' loot detector: drops drawn in a fake frame, found when wholly inside the box searched."""
    height = ITEM_HEIGHT

    def __init__(self, clock: Clock | None = None):
        self.clock = clock

    def detect(self, frame: np.ndarray, box=None, mesos_only: bool = False) -> list[Item]:
        if self.clock is not None:
            self.clock.sleep(LOOT_S)
        x0, y0, x1, y1 = box or (0, 0, frame.shape[1], frame.shape[0])
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        found = [i for i in getattr(frame, "screen", Screen()).items
                 if (i.mesos or not mesos_only) and x0 <= i.x - i.w // 2 and i.x + i.w // 2 <= int(x1)
                 and y0 <= i.y - i.h // 2 and i.y + i.h // 2 <= int(y1)]
        return suppress_overlaps(found)

    def detect_each(self, frame: np.ndarray, boxes, mesos_only: bool = False) -> list[list[Item]]:
        return [self.detect(frame, box, mesos_only) for box in boxes]  # each charged as a strip of its own


def detector_kinds(model: MapModel) -> tuple[str, ...]:
    """The kinds the fake detector finds on map `model`: KINDS, and the map's own, as the real one at 1920x1080 looks
    for them (sprites.detector_for: the kinds with a hand-made colour gate, and the map's with gates learned against its
    background, perception.gates). Each is found as its kind: the real one, searching kinds of one body as one and
    telling them apart side by side, found all 44 of Tree Dungeon's labelled mushrooms in plain view, none as the other
    kind (Perception)."""
    return KINDS + tuple(sorted({kind_of(s.name) for s in model.spawns if s.name} - set(KINDS)))


def shown_exp(start: tuple[int, int], gained: int) -> tuple[int, int]:
    """Her level and EXP into it, from `start` (level, EXP) with `gained` EXP more: each level's requirement
    (the EXP table) carried into the next."""
    level, exp = start[0], start[1] + gained
    while level in EXP_TO_NEXT and exp >= EXP_TO_NEXT[level]:
        exp -= EXP_TO_NEXT[level]
        level += 1
    return level, exp


class FakeEyes(Eyes):
    def __init__(self, world: World, clock: Clock, model: MapModel, guard: Callable[[], None] | None = None):
        """`guard`: called before every look; it raises when the fakes aren't in place (sim.scenarios.Sim)."""
        self.world, self.clock, self.model = world, clock, model
        # the game's camera: the default offset, or where it was seen keeping her on this map (CAMERA_DOWN)
        self.camera = Camera(model.bounds, (CAMERA_OFFSET[0], CAMERA_DOWN.get(model.id, CAMERA_OFFSET[1])))
        self.loc = Localizer(model)
        self.detector = FakeDetector(detector_kinds(model), clock=clock)
        self._loot = FakeLoot(clock)
        self.want_ropes = False
        self.guard = guard
        self.counts = {"looks": 0, "glances": 0, "tag hidden": 0, "dot hidden": 0}  # for the trace
        self.frame_counts = {"frames": 0, "tag hidden": 0, "dot hidden": 0}  # the feed's frames (observe)
        self._exp_drawn: tuple[int, np.ndarray] | None = None  # her EXP number's box, for the EXP she has gained
        self._bars: dict[tuple, float | None] = {}  # what a bar's box read (`bar`), by its px

    @property
    def noise(self) -> Noise:
        return self.world.noise

    def _now(self) -> None:
        if self.guard is not None:
            self.guard()
        self.clock.charge()
        self.world.advance(self.clock.now)

    def her_on_screen(self) -> tuple[int, int]:
        """Her name tag's centre (half-size px), where the camera puts her."""
        x, y = self.world.her()
        sx, sy = self.camera.to_screen(x, y, (x, y))
        return round(sx / SCALE), round(sy / SCALE) + 8

    def _hp(self) -> float:
        return round(self.world.hp * 147) / 147  # the bar is 147 px long

    def _mp(self) -> float:
        return round(self.world.mp_drawn() * MP_STEPS) / MP_STEPS  # as her MP bar shows it (attack.choice.MP_STEPS)

    def _dot(self, counts: dict | None = None) -> tuple[float, float] | None:
        """Her minimap dot as minimap.her_dot reads it: exact, or (Noise.dot_px) to whole pixels; None
        while something covers the minimap (Noise.dot_hidden), counted in `counts` (her looks' by default)."""
        if Noise.during(self.noise.dot_hidden, self.world.t):
            (self.counts if counts is None else counts)["dot hidden"] += 1
            return None
        dot = self.loc.to_dot(*self.world.her())
        return (float(round(dot[0])), float(round(dot[1]))) if self.noise.dot_px else dot

    def _others(self) -> list[Marker]:
        """Other players' minimap dots as minimap.other_dots reads them: the other players' (World.others_at), each
        drawn OTHER_SHIFT right of and below where her dot would be there, read to whole pixels as hers is
        (Noise.dot_px); none while something covers the minimap (Noise.dot_hidden), none within Noise.dot_under of hers
        (hers is drawn over them), and none outside the minimap window's view (in_window)."""
        if Noise.during(self.noise.dot_hidden, self.world.t):
            return []
        out, hx, hy = [], *self.world.her()
        for at in self.world.others_at():
            if self.noise.dot_under and math.hypot(at[0] - hx, at[1] - hy) < self.noise.dot_under:
                continue
            if not self.in_window(at[1]):
                continue
            x, y = self.loc.to_dot(*at)
            if self.noise.dot_px:
                x, y = float(round(x)), float(round(y))
            out.append(Marker("stranger", x + OTHER_SHIFT, y + OTHER_SHIFT))
        return out

    def in_window(self, y: float) -> bool:
        """Whether map y is in the minimap window's view: everywhere, but on a map whose picture is taller than the
        window shows (WINDOW_ROWS), the rows the window scrolls to with her dot WINDOW_DOT_ROW from its top (stopping at
        the picture's ends)."""
        rows = WINDOW_ROWS.get(self.model.id)
        if rows is None:
            return True
        mm = self.model.minimap
        tall = mm["height"] // 2 ** mm["mag"]
        top = min(max(0.0, self._picture_row(self.world.y) - WINDOW_DOT_ROW), tall - rows)
        return top <= self._picture_row(y) < top + rows

    def _picture_row(self, y: float) -> float:
        """The row of the minimap picture (game px from its top) the game draws a dot in for map y: the dot as it's read
        (Localizer.to_dot, whose origin was fitted with the picture's corner at PICTURE_BASE), 7.2 rows above the map's
        own row in the picture ((y + centerY) / 16): her dot on 28 at 130.2 (130.5 on the study's frames), on 19 93.4
        (93.5)."""
        return self.loc.geo.to_game(self.loc.to_dot(0.0, y)[1]) - PICTURE_BASE[1]

    def _exp_box(self) -> np.ndarray:
        """Her EXP number as the bar shows it, from her level and EXP at the start (World.character) and what she has gained
        (exp_number.draw: the game's glyphs as the game draws them; read only when it changes); with a block of white
        over it while something covers it (Noise.exp_hidden): then it can't be read."""
        gained = self.world.exp
        if self._exp_drawn is None or self._exp_drawn[0] != gained:
            level, exp = shown_exp((self.world.character.level, self.world.character.exp), gained)
            box = exp_number.draw(exp_number.shown(exp, level))
            box.flags.writeable = False
            self._exp_drawn = (gained, box)
        box = self._exp_drawn[1]
        if Noise.during(self.noise.exp_hidden, self.world.t):
            box = box.copy()
            box[:, 20:50] = 255
        return box

    def bar(self, obs: Observation, mob: Mob) -> float | None:
        """Eyes.bar, off the bar drawn over it as the game draws it (perception.hud.draw_hp_bar); with
        Realism.bars_empty a bar up reads empty, as the stack read them live at 1366x768 until 2026-10-25 (hud.hp_bar
        took the black frame around the fill for the bar). Read as live reads it (hud.hp_bar), once for each picture of
        the box it reads and place of the monster in it (the sky around, the bars drawn as the game draws them: few
        pictures), 1.1 ms a read otherwise: 8 of the 18 s a scenario with the feed took."""
        x, top = mob.x, mob.y - mob.h // 2
        x0, y0, x1, y1 = bar_box(obs.frame.shape, x, top, obs.geo)
        key = (np.asarray(obs.frame[y0:y1, x0:x1]).tobytes(), x1 - x0, x - x0, top - y0, obs.geo.client)
        if key not in self._bars:
            if len(self._bars) > 4096:
                self._bars.clear()
            self._bars[key] = super().bar(obs, mob)
        left = self._bars[key]
        return 0.0 if left is not None and self.world.realism.bars_empty else left

    def look(self) -> Observation:
        self._now()
        self.counts["looks"] += 1
        s = self._observation(self.clock.now + LOOK_LAG, self.want_ropes, self.counts)
        self.clock.sleep(LOOK_S)
        return s

    def observe(self, ropes: bool | None = None) -> Observation:
        """The screen now as a look sees it, without the time a look takes: the continuous feed's frames
        (localization.feed.SyncFeed, whose own process pays for them). Not one of her looks: counted apart
        (`frame_counts`)."""
        if self.guard is not None:
            self.guard()
        self.world.advance(self.clock.now)
        self.frame_counts["frames"] += 1
        return self._observation(self.clock.now, self.want_ropes if ropes is None else ropes, self.frame_counts)

    def _observation(self, t: float, ropes: bool, counts: dict) -> Observation:
        """What the screen shows now (the world advanced to the clock), as taken at `t`."""
        w, (x, y) = self.world, self.world.her()
        her = self.her_on_screen()
        frame = _BLANK.copy().view(Frame)  # 0.3 ms, against 2.5 to fill a new one with SKY: the same pixels
        frame.screen = self._draw(frame, her, x, y)
        found = self._ropes(her, x, y) if ropes else []
        seen = not Noise.during(self.noise.tag_hidden, w.t)
        counts["tag hidden"] += not seen
        s = Observation(t=t, frame=frame, tag=her if seen else None, hp=self._hp(), mp=self._mp(),
                        ground=self._ground(her) if seen else None, dot=self._dot(counts), others=self._others(),
                        ropes=found)
        s.exp, s.exp_box = exp_pixels(w.exp), self._exp_box()
        if Noise.during(self.noise.exp_hidden, w.t):  # the cursor over the number changes its pixels as it comes
            s.exp = s.exp.copy()                       # and goes, as in game (and a kill under it the rest)
            s.exp[:, 20:50] = True
        return s

    def glance(self) -> tuple[tuple[float, float] | None, float]:
        self._now()
        self.counts["glances"] += 1
        dot, hp, self.seen_at = self._dot(), self._hp(), self.clock.now
        self.clock.sleep(GLANCE_S)
        return dot, hp

    def _ground(self, her: tuple[int, int]) -> tuple[int, int]:
        """terrain.ground_extent's answer: the screen x range (half size) of the grass she stands on, which
        ends GRASS_OVERHANG px inside her platform's real ends, and the screen's edges where they're off
        it. Under her feet in the air or on a rope there's no grass: (her x, her x)."""
        w = self.world
        if w.mode != "ground" or w.plat is None:
            return her[0], her[0]
        hx, p = SCALE * her[0], w.plat
        left = (hx + (p.left - w.x) * SCREEN_SCALE - GRASS_OVERHANG) / SCALE
        right = (hx + (p.right - w.x) * SCREEN_SCALE + GRASS_OVERHANG) / SCALE
        return max(0, min(her[0], round(left))), min(FRAME[1] - 1, max(her[0], round(right)))

    def chat_line(self) -> ChatLine:
        """The message bar's game line (World.chat: a refusal of hers on a rope), as Eyes.chat_line reads it: its text
        read anew (CHAT_OCR_S) only when it's another line than the last read."""
        self._now()
        line = ChatLine(self.world.chat[1] if self.world.chat is not None else "", None, self.clock.now)
        new = bool(line.text) and line.text != getattr(self, "_chat_read", "")
        self._chat_read = line.text or getattr(self, "_chat_read", "")
        self.clock.sleep(CHAT_S + (CHAT_OCR_S if new else 0.0))
        return line

    def messages(self) -> str:
        self._now()
        text = "\n".join(self.world.recent_messages())
        self.clock.sleep(MESSAGES_S)
        return text

    # --- drawing ------------------------------------------------------------------------------------

    def _draw(self, frame: np.ndarray, her: tuple[int, int], x: float, y: float) -> Screen:
        u, floor = UNITS_PER_HALF_PX, her[1] - 8
        screen = Screen(heading=self._heading())
        for m, mx, my, alive in self.world.visible_monsters():
            if self._unseen(m, alive):
                continue
            w, h = round(2 * m.half / u), round(m.height / u)
            cx, feet = round(her[0] + (mx - x) / u), round(floor + (my - y) / u)
            # Where its feet are, which way it faces and what it's doing, as the templates tell (a dying
            # one's frames aren't among them): walking, standing, or flinching after a hit.
            way, walking = m.faces(self.world.t) if alive else (m.facing, False)
            pose = ("hit1" if m.flinch_at is not None and self.world.t - m.flinch_at < FLINCH_S
                    else "move" if walking else "stand") if alive else ""
            if pose in ("move", "stand") and self._pose_wrong(m):
                pose = "stand" if pose == "move" else "move"
            mob = Mob(m.kind, cx, feet - h // 2, w, h, 0.9, float(cx), float(feet), way, pose)
            if not (0 <= cx - w // 2 and cx + w // 2 <= FRAME[1]
                    and 0 <= mob.y - h // 2 and mob.y + h // 2 <= PLAY_BOTTOM):
                continue
            screen.mobs.append(mob)
            if alive and m.hp < m.hp_max:  # (read empty with Realism.bars_empty: `bar`)
                draw_hp_bar(frame, cx, mob.y - h // 2, m.hp / m.hp_max, STRETCHED.frame_scale)
        for d in self.world.live_drops():
            iw, ih = ITEM_SIZE.get(d.kind, (18, 18))
            cx, bottom = round(her[0] + (d.x - x) / u), round(floor + (d.y - y) / u)
            item = Item(d.kind, cx, bottom - ih // 2, iw, ih, 0.9)
            if 0 <= cx - iw // 2 and cx + iw // 2 <= FRAME[1] and 0 <= item.y - ih // 2 and bottom <= PLAY_BOTTOM:
                screen.items.append(item)
        return screen

    def _heading(self) -> Reading:
        """Which way she faces as perception.heading reads her picture now: the world's facing, matched as her
        pictures matched on her own frames (0.6, 0.4 ahead); not told on a rope (only her back shows), nor on
        Noise.heading_unsure's share of frames (drawn per tick); the other way on Noise.heading_wrong's."""
        w = self.world
        if w.mode == "rope":
            return Reading(None, why="rope")
        if self.noise.heading_unsure and w.chance("heading") < self.noise.heading_unsure:
            return Reading(None, 0.5, 0.02, "close")
        if self.noise.heading_wrong:
            share, score, margin, runs = self.noise.heading_wrong
            draw = w.chance(f"heading_wrong:{self._pose()}", int(w.t // HEADING_SPELL_S)) if runs else \
                w.chance("heading_wrong")
            if draw < share:
                return Reading(LEFT if w.facing == RIGHT else RIGHT, score, margin)
        return Reading(w.facing, 0.6, 0.4)

    def _pose(self) -> str:
        """Her pose as her picture shows it (for Noise.heading_wrong's runs): in the air, attacking, lying down,
        walking or standing."""
        w = self.world
        if w.mode != "ground":
            return w.mode
        if w.attack_ms[0] <= w.ms < w.attack_ms[1]:
            return "attack"
        return "prone" if "down" in w.held else "walk" if w.vx else "stand"

    def heading(self, obs: Observation) -> Reading:
        """Eyes.heading over the world: what her picture showed when the look's frame was drawn (_heading); not told
        where her name tag wasn't seen."""
        screen = getattr(obs.frame, "screen", None)
        if screen is None or obs.tag is None or not getattr(obs, "tag_seen", True):
            return Reading(None, why="no tag")
        return screen.heading

    def learn_heading(self, obs: Observation, side: str, reading: Reading | None = None) -> bool:
        return False  # the world's facing needs no pictures

    def keep_heading(self, obs: Observation, path) -> bool:
        return False  # no pixels of her to keep

    def _pose_wrong(self, m) -> bool:
        """Whether monster `m`'s pose reads wrong now (Noise.pose_wrong), drawn per monster for each spell."""
        w = self.world
        return bool(self.noise.pose_wrong) and w.chance(f"pose:{m.idx}", int(w.t // POSE_SPELL_S)) < self.noise.pose_wrong

    def _unseen(self, m, alive: bool) -> bool:
        """Whether this look misses monster `m` though it shows: hit and hiding under its damage numbers
        (Noise.hit_hidden), on her platform by how near her it is (Noise.near_seen: her body over it, as live) or on
        another (Noise.off_seen), or just not found (Noise.detect_miss). All are drawn per monster from the world's
        seeded chances, the hiding once per hit."""
        noise, w = self.noise, self.world
        if alive and noise.hit_hidden and m.hit_at is not None:
            lo, hi = noise.hit_hidden
            if w.t < m.hit_at + lo + (hi - lo) * w.chance(f"hide:{m.idx}", int(round(m.hit_at * 1000))):
                return True
        if alive and (noise.near_seen or noise.off_seen is not None) and (seen := self._near_seen(m)) is not None:
            return w.chance(f"see:{m.idx}") >= seen
        return bool(noise.detect_miss) and w.chance(f"see:{m.idx}") < noise.detect_miss

    def _near_seen(self, m) -> float | None:
        """The chance a look finds monster `m`: on the platform she's on (or last stood on, in the air) by how near her
        it is (Noise.near_seen's first band that holds it, middle to middle), on another Noise.off_seen; None past the
        bands, or without them (off_seen None)."""
        w = self.world
        her = w.plat or w.last_plat
        if her is None:
            return None
        if m.platform.id != her.id:
            return self.noise.off_seen
        dx = abs(m.at(w.t) - w.x)
        return next((seen for upto, seen in self.noise.near_seen if dx < upto), None)

    def _ropes(self, her: tuple[int, int], x: float, y: float) -> list[ScreenRope]:
        """Ropes in view as terrain.find_ropes reports them: full-size px, at least 90 px of them showing."""
        out = []
        hx, floor = SCALE * her[0], SCALE * (her[1] - 8)
        for r in self.model.ropes:
            rx = round(hx + (r.x - x) * SCREEN_SCALE)
            top = max(0, round(floor + (r.top - y) * SCREEN_SCALE))
            bottom = min(int(1080 * 0.9), round(floor + (r.bottom - y) * SCREEN_SCALE))
            if 0 <= rx < 1920 and bottom - top >= 90:
                out.append(ScreenRope(rx, top, bottom))
        return sorted(out, key=lambda r: (r.x, r.top))


_BLANK = np.full(FRAME + (3,), SKY, np.uint8)


# --- control -----------------------------------------------------------------------------------------

@dataclass
class Plan:
    id: str
    index: int
    name: str
    events: list
    lead: float          # s before its first key
    length: float        # s from its start to its end
    sent: float
    start: float = 0.0
    end: float = 0.0
    state: str = "queued"  # queued | withdrawn | cut
    error: str | None = None
    kind: str | None = None   # client.send's: what it is to the player with the Lie Detector alarm up
    alarm: str | None = None  # the Lie Detector error it's answered with: refused at its start, or stopped by the alarm


@dataclass
class Trace:
    t0: float
    plans: list = field(default_factory=list)    # [t sent, name, events, end_ms]
    cancels: list = field(default_factory=list)  # [t, plan index, withdrawn (not started yet)]
    log: list = field(default_factory=list)

    def at(self, t: float) -> float:
        return round(t - self.t0, 3)


class FakePlayer:
    """gamelab.control.client's submit/send/done/wait/cancel, playing the keys on the world."""

    def __init__(self, world: World, clock: Clock, trace: Trace, keys: GameKeys):
        from ..control.player import allowed_keys
        self.world, self.clock, self.trace = world, clock, trace
        self.keys = keys
        self.allowed = allowed_keys(keys)
        self.plans: dict[str, Plan] = {}
        self.order: list[Plan] = []
        self.free_from = clock.now  # after a cancel: the player takes a moment to notice it and stop
        self.alarm_at: float | None = None  # when the player sees the Lie Detector (the fake clock), if it does
        self.fed: dict[str, float] = {}     # pet food plans it took with the alarm up -> when they started
        self.played: dict[str, tuple] = {}  # plan id -> (its kind, its key and mouse events at clock times) as played
        self.pages: list[list] = []          # whoever is at the PC paged (control.alarm.page): [s, title, text]
        self.kept: list[list] = []           # the recording's saves asked (control.alarm.keep_recent): [s, s, name]

    # --- the Lie Detector (client.check_alarm, alarm_up, alarm_age) -------------------------------------------------

    def alarm_up(self) -> bool:
        return self.alarm_at is not None and self.clock.now >= self.alarm_at

    def check_alarm(self) -> None:
        if self.alarm_up():
            raise client.LieDetectorAlarm(f"Lie Detector seen at {self.trace.at(self.alarm_at):.3f} s")

    def alarm_age(self) -> float | None:
        return self.clock.now - self.alarm_at if self.alarm_up() else None

    def page(self, title: str, text: str) -> None:
        """control.alarm.page in a simulated run (the safety stop's: behavior.retreat.on_alarm): kept, nobody paged."""
        self.pages.append([self.trace.at(self.clock.now), title, text])

    def keep_recent(self, seconds: float, name: str) -> str:
        """control.alarm.keep_recent in a simulated run (out of darts': behavior.retreat.on_alarm): kept, nothing
        saved."""
        self.kept.append([self.trace.at(self.clock.now), seconds, name])
        return f"the recording's last {seconds:g} s saved (simulated: nothing written)"

    # --- plans ---------------------------------------------------------------------------------------------------

    def send(self, events=(), *, fps=0, duration_ms=0, crop=None, scale=1, focus=True, lead_ms=100, name="plan",
             end_ms: int | None = None, kind: str | None = None) -> str:
        from ..control.player import validate
        if kind is None:
            self.check_alarm()  # as client.send: none goes out with the alarm up but the kinds the player takes then
        self.clock.charge()
        events = [list(e) for e in events]
        capture = {"fps": fps, "duration_ms": duration_ms, "scale": scale, "lead_ms": lead_ms}
        plan = {"events": events, "capture": capture, "focus": focus}
        if end_ms is not None:
            plan["end_ms"] = end_ms
        last = max((e[0] for e in events), default=0)
        p = Plan(f"{name}_{len(self.order):04d}", len(self.order), name, events, lead_ms / 1000,
                 (lead_ms + max(last, end_ms or 0, duration_ms)) / 1000, self.clock.now, kind=kind)
        try:
            validate(plan, self.allowed)
        except ValueError as e:
            p.error = str(e)
        self.trace.plans.append([self.trace.at(self.clock.now), name, events, end_ms])
        self.plans[p.id] = p
        self.order.append(p)
        self._schedule(p, self._after(p))
        self.clock.mark()
        return p.id

    def submit(self, events=(), *, fps=0, duration_ms=0, crop=None, scale=1, focus=True, lead_ms=100,
               name="plan", timeout=120, kind: str | None = None) -> dict:
        return self.wait(self.send(events, fps=fps, duration_ms=duration_ms, crop=crop, scale=scale, focus=focus,
                                   lead_ms=lead_ms, name=name, kind=kind), timeout)

    def done(self, plan_id: str) -> bool:
        p = self.plans[plan_id]
        return p.state != "withdrawn" and self.clock.now >= p.end

    def wait(self, plan_id: str, timeout: float = 120) -> dict:
        p = self.plans[plan_id]
        if p.state == "withdrawn":
            raise client.PlayerError(f"plan {plan_id} was withdrawn: no result will come")
        self.clock.charge()
        self.clock.advance_to(p.end + RESULT_LATENCY)
        self.clock.mark()
        if p.error:
            raise client.PlayerError(p.error)
        if p.alarm:
            raise client.LieDetectorAlarm(p.alarm)
        return {"sent": [[round(e[0]), *e[1:]] for e in p.events],
                "problems": ["cancelled"] if p.state == "cut" else []}

    def cancel(self, plan_id: str) -> bool:
        """Withdraw plan `plan_id` if it hasn't started (True), else cut it short now (False). A plan due
        to start at this very moment hasn't pressed anything yet (the world at time t has taken the key
        events before t, not those at t): it's withdrawn."""
        p, now = self.plans[plan_id], self.clock.now
        if p.state == "withdrawn":
            return True
        if now <= p.start:  # not picked up yet: withdrawn
            self.world.withdraw(p.id)
            p.state = "withdrawn"
            withdrawn = True
        elif now < p.end:  # running: stops pressing, lets go of every key
            self.world.advance(now)
            self.world.withdraw(p.id)  # what the world hasn't taken yet won't come
            self.world.release_all(now, p.id)
            p.end, p.state = now, "cut"
            self.free_from = now + START_LATENCY  # ... and only then picks up the next plan
            withdrawn = False
        else:
            withdrawn = False
        self.trace.cancels.append([self.trace.at(now), p.index, withdrawn])
        self._requeue(p.index)
        return withdrawn

    def _after(self, p: Plan) -> float:
        """When plan `p` starts: once handed over, and after the plans before it."""
        before = [q.end for q in self.order[:p.index] if q.state != "withdrawn"]
        return max([p.sent + START_LATENCY, self.free_from] + before)

    def _schedule(self, p: Plan, start: float) -> None:
        p.start, p.end = start, start + p.length
        self.played.pop(p.id, None)
        self.fed.pop(p.id, None)
        if p.error:  # the real player refuses it without pressing anything
            p.end = start
            return
        cutoff = self._cutoff(p, start)
        if cutoff <= start:  # refused for the Lie Detector, nothing pressed
            p.end = start
            return
        played = []
        for e in p.events:
            at = start + p.lead + e[0] / 1000
            if at >= cutoff:  # stopped by the alarm (or her retreat's window): nothing after
                continue
            if len(e) == 3 and e[2] in ("down", "up"):
                self.world.key(at, e[1], e[2] == "down", p.id)
            played.append([round(at, 4), *e[1:]])
        p.end = min(p.end, cutoff)
        self.world.release_all(p.end, p.id)
        self.played[p.id] = (p.kind, played)

    def _cutoff(self, p: Plan, start: float) -> float:
        """When plan `p`, starting at `start`, stops for the Lie Detector (math.inf: it doesn't), as the real player
        has it: one running when it's seen stops then, whatever its kind; one starting after plays only if
        player.alarm_refusal lets it through, a retreat until alarm.RETREAT_WINDOW_S after the sighting. Sets
        `p.alarm` to the error it's answered with."""
        from ..control.alarm import LIE_DETECTOR_ERROR, PET_FOOD, PLAN_KINDS, RETREAT, RETREAT_WINDOW_S
        from ..control.player import alarm_refusal
        p.alarm, seen = None, self.alarm_at
        if seen is None:
            return math.inf
        if start < seen:
            if p.end > seen:
                p.alarm = LIE_DETECTOR_ERROR
            return seen
        if p.kind not in PLAN_KINDS:
            p.alarm = LIE_DETECTOR_ERROR
            return start
        fed = [t for pid, t in self.fed.items() if pid != p.id]
        plan = {"kind": p.kind, "events": p.events, "capture": {"lead_ms": round(p.lead * 1000)}}
        if why := alarm_refusal(plan, self.keys, start - seen, fed, start):
            p.alarm = f"{LIE_DETECTOR_ERROR}: {why}"
            return start
        if p.kind == PET_FOOD:
            self.fed[p.id] = start
        return seen + RETREAT_WINDOW_S if p.kind == RETREAT else math.inf

    def _requeue(self, index: int) -> None:
        """Plans queued after `index` start as soon as the ones before them are done."""
        for q in self.order[index + 1:]:
            if q.state != "queued" or self.clock.now > q.start:
                continue
            start = self._after(q)
            if abs(start - q.start) > 1e-9:
                self.world.withdraw(q.id)
                self._schedule(q, start)


# --- putting the fakes in place ----------------------------------------------------------------------

@contextmanager
def patched(clock: Clock, player: FakePlayer, folder: Path) -> Iterator[None]:
    """Fake time and the player, point every gamelab path under gamelab/lab (module globals, and function
    defaults) at `folder`, and make anything that would reach the game, or write under the real
    gamelab/lab, raise SimViolation; all restored on the way out. Only modules already loaded are
    redirected (sim.scenarios loads them all first); the write guard catches whatever that misses."""
    undo: list[tuple[object, str, object]] = []

    def put(obj, name, value):
        undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def refuse(what):
        def fn(*a, **k):
            raise SimViolation(f"{what} called during a simulated run")
        return fn

    def moved(value):
        if isinstance(value, Path) and (value == LAB or LAB in value.parents):
            return folder / value.relative_to(LAB)
        return value

    try:
        for name in ("monotonic", "sleep", "time", "perf_counter", "strftime"):
            put(time, name, getattr(clock, "monotonic" if name == "perf_counter" else name))
        for name in ("submit", "send", "done", "wait", "cancel", "check_alarm", "alarm_up", "alarm_age"):
            put(client, name, getattr(player, name))
        from ..control import alarm
        put(alarm, "page", player.page)
        put(alarm, "keep_recent", player.keep_recent)
        mods = [m for n, m in list(sys.modules.items()) if n.startswith("gamelab") and not n.startswith("gamelab.sim")]
        for mod in mods:
            for name, value in list(vars(mod).items()):
                if moved(value) is not value:
                    put(mod, name, moved(value))
            for fn in _functions(mod):
                if fn.__defaults__ and any(moved(v) is not v for v in fn.__defaults__):
                    put(fn, "__defaults__", tuple(moved(v) for v in fn.__defaults__))
                if fn.__kwdefaults__ and any(moved(v) is not v for v in fn.__kwdefaults__.values()):
                    put(fn, "__kwdefaults__", {k: moved(v) for k, v in fn.__kwdefaults__.items()})
        for obj, name, value in _write_guards(LAB):
            put(obj, name, value)
        from maplehelper import win32
        from .. import window
        for name in ("send_key", "foreground_title"):
            if hasattr(win32, name):
                put(win32, name, refuse(f"maplehelper.win32.{name}"))
        for name in ("bring_to_front", "find_game", "client_rect", "foreground_title", "move_cursor", "shows",
                     "minimized"):
            if hasattr(window, name):
                put(window, name, refuse(f"gamelab.window.{name}"))
        from ..analysis import recordings
        put(recordings, "record", refuse("gamelab.analysis.recordings.record"))
        yield
    finally:
        for obj, name, value in reversed(undo):
            setattr(obj, name, value)


def _functions(mod: types.ModuleType) -> Iterator[types.FunctionType]:
    """The functions a module defines: its own, and its classes' methods."""
    for value in list(vars(mod).values()):
        if getattr(value, "__module__", None) != mod.__name__:
            continue
        if isinstance(value, types.FunctionType):
            yield value
        elif isinstance(value, type):
            for attr in vars(value).values():
                fn = getattr(attr, "__func__", attr)  # staticmethod / classmethod
                if isinstance(fn, types.FunctionType):
                    yield fn


def _write_guards(root: Path) -> list[tuple[object, str, Callable]]:
    """Stand-ins for open() and the os calls that change files, refusing any path under `root`."""
    top = os.path.normcase(os.path.abspath(root))

    def inside(path) -> bool:
        try:
            p = os.path.normcase(os.path.abspath(os.fsdecode(path)))
        except TypeError:  # a file descriptor
            return False
        return p == top or p.startswith(top + os.sep)

    def check(what: str, *paths) -> None:
        for path in paths:
            if inside(path):
                raise SimViolation(f"{what}({os.fsdecode(path)!r}) during a simulated run: nothing may be "
                                   f"written under {root}")

    real_open = builtins.open
    writes = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def open_(file, mode="r", *a, **k):
        if any(c in mode for c in "wax+"):
            check("open", file)
        return real_open(file, mode, *a, **k)

    def os_open(path, flags, *a, **k):
        if flags & writes:
            check("os.open", path)
        return real["open"](path, flags, *a, **k)

    real = {name: getattr(os, name) for name in ("open", "replace", "rename", "remove", "unlink", "mkdir",
                                                  "rmdir", "utime")}

    def one(name):
        def fn(path, *a, **k):
            check(f"os.{name}", path)
            return real[name](path, *a, **k)
        return fn

    def two(name):
        def fn(src, dst, *a, **k):
            check(f"os.{name}", src, dst)
            return real[name](src, dst, *a, **k)
        return fn

    return ([(builtins, "open", open_), (io, "open", open_), (os, "open", os_open)]
            + [(os, n, two(n)) for n in ("replace", "rename")]
            + [(os, n, one(n)) for n in ("remove", "unlink", "mkdir", "rmdir", "utime")])
