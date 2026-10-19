"""Carrying out route moves in the game: one look, one key plan per move, one look to check.

Each move is a single plan from wherever she stands, the way a player does it: run towards the end
and jump on the way (no stopping at a run-up spot first), run at a rope and jump onto it holding up
(no lining up underneath first), walk and drop in one go. Only climbing down a rope still lines up
first, since she has to stand on its top to grab it.

Her position comes from Senses.look (localization.pose.refine): a rope in view or the visible end of
her platform when her name tag shows, else the minimap dot.
"""

import math
import time

from maplehelper.engine import LEFT, RIGHT

from .. import directions
from ..localization.mapdata import MapModel, Platform
from ..localization.scene import Scene
from ..localization.senses import Senses
from ..motion.physics import CLIMB_SPEED, GRAVITY, JUMP_SPEED, ROPE_REACH, WALK_SPEED, air_time, fly, hold_ms, run_ms
from ..motion.route import JUMP_FROM, Move
from ..screen import SCALE, SCREEN_SCALE
from .actions import Actions, tap


WALK_PX_PER_S = WALK_SPEED * SCREEN_SCALE
SLIDE_MS = 220            # after releasing the arrow she slides to a stop within this
X_TOLERANCE = 8           # map units: close enough when lining up
SPOT_TOLERANCE = 6        # map units: close enough to a run-up spot, or under a rope
ROPE_TOLERANCE = 6        # screen px: close enough to a rope's top to climb down it
ROPE_SEARCH_PX = 70       # screen px: a rope this close to where the map puts it is the one
GRAB_MARGIN = 8           # grab with at least this much rope to spare
MIN_RUN = 7               # map units: the shortest run-up a jump is timed for ...
RUN_UP_SPARE = 10         # ... and backing off for one goes this much further
CLIMB_EXTRA_MS = 500      # up (or down) is held this much longer than the climb takes


def grab_lead(gap: float) -> float | None:
    """How far before a rope to jump (running at her walking speed) to grab it at the top of the jump,
    when its bottom is `gap` above her feet; None if a jump can't reach it."""
    if gap > ROPE_REACH - GRAB_MARGIN:
        return None
    return WALK_SPEED * JUMP_SPEED / GRAVITY  # ~41 units: she's at the top of the jump over the rope


def climb_ms(height: float) -> int:
    """How long to hold up (or down) to climb `height` map units of rope."""
    return int(1000 * height / CLIMB_SPEED) + CLIMB_EXTRA_MS


class Mover:
    """Carries out route moves one at a time, each from a look at where she is."""

    def __init__(self, senses: Senses, actions: Actions, model: MapModel, log=print):
        self.senses, self.actions, self.model, self.log = senses, actions, model, log

    # --- looking ----------------------------------------------------------------------------

    def locate(self, tries: int = 40, mobs: bool = False, wide: bool = False) -> Scene:
        """A look that places her (its `where` is set). Waits up to ~10 s for her minimap dot (a window
        or notification popping up over the minimap hides it)."""
        for i in range(tries):
            s = self.senses.look(mobs=mobs, wide=wide)
            if s.where is not None:
                return s
            if i == 3:
                self.actions.nudge_mouse()  # a tooltip from another program once hid the minimap
            time.sleep(0.25)
        raise RuntimeError("can't find her minimap dot")

    def settle(self, seconds: float = 2.0) -> Scene:
        """Look until she's standing still on a platform (or give up and return the last look)."""
        end = time.monotonic() + seconds
        last = self.locate()
        while time.monotonic() < end:
            time.sleep(0.08)
            s = self.locate()
            if s.where.platform is not None and last.where.platform is s.where.platform \
                    and math.dist(last.where.dot, s.where.dot) < 0.6:
                return s
            last = s
        return last

    # --- walking ----------------------------------------------------------------------------

    def _send(self, events, name):
        """Run a move's key plan (in any order: it's sorted by time here)."""
        self.actions.play(sorted(events, key=lambda e: e[0]), name)
        self.actions.facing = None  # moves turn her; the next throw turns her to its target

    def walk_to(self, x: float, tolerance: float = X_TOLERANCE, tries: int = 4, s: Scene | None = None) -> Scene:
        """Walk along her platform until her feet are at map x."""
        s = s or self.locate()
        for _ in range(tries):
            dx = x - s.where.x
            if abs(dx) <= tolerance:
                return s
            events, _ = self.walk_events(dx, tolerance)
            self._send(events, "walk")
            time.sleep(0.25)  # let her slide to a stop before looking
            s = self.locate()
        return s

    def walk_events(self, dx: float, tolerance: float) -> tuple[list, int]:
        """Key events to walk dx map units (nothing if within tolerance), and when she's stopped."""
        if abs(dx) <= tolerance:
            return [], 0
        arrow = self.actions.arrow(directions.toward(dx))
        ms = hold_ms(abs(dx))
        return [[0, arrow, "down"], [ms, arrow, "up"]], ms + SLIDE_MS

    def line_up_rope(self, rope_x: float) -> Scene:
        """Stand right on top of the rope at map x: walk there, then fine-tune on screen."""
        s = self.walk_to(rope_x, tolerance=12)
        for _ in range(3):
            if not (s.tag and s.tag_seen):
                return s
            her = SCALE * s.tag[0]
            expected = her + (rope_x - s.where.x) * SCREEN_SCALE
            near = [r for r in s.ropes if abs(r.x - expected) <= ROPE_SEARCH_PX]
            if not near:
                return s
            target = min(near, key=lambda r: abs(r.x - expected)).x
            dx = target - her
            if abs(dx) <= ROPE_TOLERANCE:
                return s
            arrow = self.actions.arrow(directions.toward(dx))
            ms = int(1000 * abs(dx) / WALK_PX_PER_S) + (40 if abs(dx) > 20 else 25)
            self._send([[0, arrow, "down"], [ms, arrow, "up"]], "nudge")
            time.sleep(0.2)
            s = self.locate()
        return s

    # --- route moves ------------------------------------------------------------------------

    def do(self, m: Move, s: Scene | None = None) -> Scene:
        """Carry out move `m` from where she is (`s`, or a fresh look) and return a look once she's landed."""
        s = s or self.locate()
        by_kind = {"climb": self._climb, "climb_down": self._climb_down, "jump_up": self._jump_up,
                   "jump": self._run_and_jump, "walk_off": self._walk_off, "drop": self._drop}
        if m.kind not in by_kind:
            raise ValueError(f"unknown move {m.kind}")
        by_kind[m.kind](m, s)
        return self.settle()

    def _climb_down(self, m: Move, s: Scene) -> None:
        """Stand on top of the rope (lining up looks again), then hold down all the way."""
        self.line_up_rope(m.rope_x)
        down = self.actions.keys.down.name
        self._send([[0, down, "down"], [climb_ms(m.rope_bottom - m.rope_top), down, "up"]], "climb_down")

    def _jump_up(self, m: Move, s: Scene) -> None:
        """Walk to the takeoff spot (unless she's there) and jump straight up."""
        events, t = self.walk_events(m.x - s.where.x, 15)
        self._send(events + tap(t, self.actions.keys.jump.name), "jump_up")
        time.sleep(0.45)  # then settle looks for her standing still

    def _walk_off(self, m: Move, s: Scene) -> None:
        """Walk off the end of her platform, on to where she lands."""
        arrow = self.actions.arrow(m.direction)
        ms = hold_ms(abs(m.land_x - s.where.x) + 20)
        self._send([[0, arrow, "down"], [ms, arrow, "up"]], "walk_off")
        time.sleep(0.3)  # then settle looks for her standing still

    def _drop(self, m: Move, s: Scene) -> None:
        """Walk to the drop spot (unless she's there) and drop through her platform: down + jump."""
        down, jump = self.actions.keys.down.name, self.actions.keys.jump.name
        events, t = self.walk_events(m.x - s.where.x, 20)
        self._send(events + [[t, down, "down"], *tap(t + 120, jump), [t + 320, down, "up"]], "drop")
        time.sleep(0.3)  # then settle looks for her standing still

    def _run_and_jump(self, m: Move, s: Scene) -> None:
        """Run towards the end and jump on the way: jump is pressed when she's JUMP_FROM from it."""
        sign = directions.sign(m.direction)
        p = self.model.platform(m.src)
        edge = p.right if sign > 0 else p.left
        takeoff = m.takeoff if m.takeoff is not None else edge - sign * JUMP_FROM
        run = (takeoff - s.where.x) * sign  # how far she runs before jumping
        if run < MIN_RUN:  # too close to the end (or past the takeoff): back up to the run-up start first
            s = self.walk_to(m.x, tolerance=SPOT_TOLERANCE, s=s)
            run = (takeoff - s.where.x) * sign
        arrow, jump = self.actions.arrow(m.direction), self.actions.keys.jump.name
        t = run_ms(max(run, MIN_RUN))
        hold = int(t + 1000 * (air_time(0) or 0.56) + 150)  # the arrow: through a level jump's air time, +150 ms
        self._send([[0, arrow, "down"], *tap(t, jump), [hold, arrow, "up"]], "jump")

    def _climb(self, m: Move, s: Scene) -> None:
        """Grab the rope on the way: run at it and jump holding up (or just walk into it holding up
        when it comes down to her feet). Standing under it already: jump straight up holding up."""
        k = self.actions.keys
        p = self.model.platform(m.src)
        gap = p.y_at(m.rope_x) - m.rope_bottom  # how far above her feet the rope ends
        hold_up = climb_ms(p.y_at(m.rope_x) - m.rope_top)
        dx = m.rope_x - s.where.x
        if abs(dx) <= SPOT_TOLERANCE:
            self._climb_from_under(hold_up)
            return
        sign = 1 if dx > 0 else -1
        walk_in = gap <= 10 and not self.model.portal_near(p, m.rope_x)  # up at a portal would take it
        lead = 0.0 if walk_in else grab_lead(gap)
        if lead is None:
            raise ValueError(f"rope at {m.rope_x:.0f} ends {gap:.0f} above her platform: out of reach")
        run = abs(dx) - lead
        if run < MIN_RUN:  # too close to jump at it: back off to a run-up first (on her platform)
            back = s.where.x - sign * (MIN_RUN - run + RUN_UP_SPARE)
            if not p.covers(back, -10):
                back = m.rope_x  # no room to back off: stand under it instead
            s = self.walk_to(back, tolerance=SPOT_TOLERANCE, s=s)
            if back == m.rope_x:
                # Under it now (within the walk's SPOT_TOLERANCE). Calling _climb again here recursed
                # forever when the walk stopped 5-6 units off, past the 4 it took as "under".
                self._climb_from_under(hold_up)
                return
            dx = m.rope_x - s.where.x
            sign, run = (1 if dx > 0 else -1), abs(dx) - lead
        arrow = self.actions.arrow(directions.toward(sign))
        t = run_ms(max(run, MIN_RUN))
        at_rope = t + int(1000 * lead / WALK_SPEED)
        # Up is held from the jump on (in the air, so a portal on the way can't take her), or when
        # walking into a rope, from shortly before she reaches it.
        up_from = t if lead else max(0, t - 250)
        events = [[0, arrow, "down"], [up_from, k.up.name, "down"], [at_rope + 120, arrow, "up"],
                  [at_rope + hold_up, k.up.name, "up"]]
        if lead:
            events += tap(t, k.jump.name)
        self._send(events, "climb")

    def _climb_from_under(self, hold_up: int) -> None:
        """Standing under the rope: jump straight up holding up."""
        k = self.actions.keys
        self._send([[0, k.up.name, "down"], *tap(40, k.jump.name), [hold_up + 350, k.up.name, "up"]], "climb")

    def off_rope(self, s: Scene) -> Scene:
        """She's on a rope (no platform under her dot): climb up onto the platform if she's near its
        top, else jump off it sideways (towards the side a fall lands on a platform)."""
        k = self.actions.keys
        x, y = s.where.x, s.where.y
        # The rope she's on: the nearest one reaching her height.
        rope = min(self.model.ropes, key=lambda r: abs(r.x - x) + (0 if r.top - 10 <= y <= r.bottom + 10 else 1000))
        if y - rope.top < 70:
            self._send([[0, k.up.name, "down"], [900, k.up.name, "up"]], "off_rope")
        else:
            # Right if a jump off to the right (up, at walking speed) comes down on a platform.
            side = RIGHT if fly(self.model, x + 15, y, WALK_SPEED, -300)[0] is not None else LEFT
            arrow = self.actions.arrow(side)
            self._send([[0, arrow, "down"], [30, k.jump.name, "down"], [80, k.jump.name, "up"], [400, arrow, "up"]],
                       "off_rope")
        return self.settle(2.5)


def platform_text(p: Platform | None) -> str:
    """Platform `p` for the log (None: she's in the air or on a rope)."""
    return "air/rope" if p is None else f"platform {p.id} ({p.left:.0f}..{p.right:.0f}, y {p.top:.0f})"
