"""Carrying out route moves in the game: one look, one key plan per move, one look to check.

Each move is a single plan from wherever she stands, the way a player does it: run towards the end
and jump on the way (no stopping at a run-up spot first), run at a rope and jump onto it holding up
(no lining up underneath first), walk and drop in one go. Only climbing down a rope still lines up
first, since she has to stand on its top to grab it.

Her position comes from Hunter.look (hunt.refine): a rope in view or the visible end of her platform
when her name tag shows, else the minimap dot.
"""

import math
import time
from dataclasses import dataclass

from maplehelper.engine import LEFT, RIGHT

from .hunt import SCALE, Hunter, Sight
from .localize import Localizer, Where
from .mapdata import SCREEN_SCALE, MapModel, Platform
from .route import CLIMB_SPEED, GRAVITY, JUMP_FROM, JUMP_SPEED, ROPE_REACH, WALK_SPEED, Move, air_time

WALK_PX_PER_S = WALK_SPEED * SCREEN_SCALE
# Measured: holding an arrow for t ms moves her ~(12 + 0.209 t) screen px including the slide after
# release, i.e. 8.5 + 0.149 t map units. Shorter than ~30 ms barely registers.
WALK_UNITS_PER_MS = 0.209 / SCREEN_SCALE
WALK_UNITS_FIXED = 12 / SCREEN_SCALE
MIN_HOLD_MS = 30
# When to press jump (ms after the arrow, from standing) to lift off `units` along: she's at
# 0.1485 (t - 120) units after t ms, and lifts off 55 ms after the key (flow.py, measured at 60 fps).
RUN_LAG_MS = 65
SLIDE_MS = 220            # after releasing the arrow she slides to a stop within this
X_TOLERANCE = 8           # map units: close enough when lining up
ROPE_TOLERANCE = 6        # screen px: close enough to a rope's top to climb down it
ARMS = ROPE_REACH - JUMP_SPEED ** 2 / (2 * GRAVITY)  # how far above her feet she grabs a rope (~18)
GRAB_MARGIN = 8           # grab with at least this much rope to spare


def hold_ms(units: float) -> int:
    """How long to hold an arrow to walk `units` map units and come to rest there."""
    return max(MIN_HOLD_MS, round((units - WALK_UNITS_FIXED) / WALK_UNITS_PER_MS))


def run_ms(units: float) -> int:
    """How long after pressing the arrow (from standing) she's `units` along, still running."""
    return round(units / WALK_UNITS_PER_MS + RUN_LAG_MS) if units > 0 else 0


def grab_lead(gap: float) -> float | None:
    """How far before a rope to jump (running at her walking speed) to grab it at the top of the jump,
    when its bottom is `gap` above her feet; None if a jump can't reach it."""
    if gap > ROPE_REACH - GRAB_MARGIN:
        return None
    return WALK_SPEED * JUMP_SPEED / GRAVITY  # ~41 units: she's at the top of the jump over the rope


@dataclass
class Fix:
    """One look: where she is, and how it was worked out."""
    where: Where
    sight: Sight
    how: str              # 'rope' or 'edge' (screen), or 'dot'


class Mover:
    def __init__(self, hunter: Hunter, model: MapModel, localizer: Localizer, log=print):
        self.h, self.model, self.loc, self.log = hunter, model, localizer, log
        hunter.eyes.want_ropes = True

    # --- looking ----------------------------------------------------------------------------

    def fix(self, tries: int = 40, mobs: bool = False, wide: bool = False) -> Fix:
        """A look at where she is. Waits up to ~10 s for her minimap dot (a window or notification
        popping up over the minimap hides it)."""
        for i in range(tries):
            s = self.h.look(mobs=mobs, wide=wide)
            if s.where is not None:
                return Fix(s.where, s, s.how)
            if i == 3:
                # A tooltip from another program once sat over the minimap until the mouse moved:
                # move it inside the game (moves only, never clicks).
                self.h._do([[0, "mouse", 1000, 620], [120, "mouse", 960, 600]], "mouse")
            time.sleep(0.25)
        raise RuntimeError("can't find her minimap dot")

    # --- basic actions ----------------------------------------------------------------------

    def _arrow(self, direction: str) -> str:
        return (self.h.keys.left if direction == LEFT else self.h.keys.right).name

    def _send(self, events, name):
        self.h._do(sorted(events, key=lambda e: e[0]), name)
        self.h.facing = None  # moves turn her; the next throw turns her to its target

    def walk_to(self, x: float, tolerance: float = X_TOLERANCE, tries: int = 4, f: Fix | None = None) -> Fix:
        """Walk along her platform until her feet are at map x."""
        f = f or self.fix()
        for _ in range(tries):
            dx = x - f.where.x
            if abs(dx) <= tolerance:
                return f
            direction = RIGHT if dx > 0 else LEFT
            self._send([[0, self._arrow(direction), "down"], [hold_ms(abs(dx)), self._arrow(direction), "up"]], "walk")
            time.sleep(0.25)  # let her slide to a stop before looking
            f = self.fix()
        return f

    def walk_events(self, dx: float, tolerance: float) -> tuple[list, int]:
        """Key events to walk dx map units (nothing if within tolerance), and when she's stopped."""
        if abs(dx) <= tolerance:
            return [], 0
        arrow = self._arrow(RIGHT if dx > 0 else LEFT)
        ms = hold_ms(abs(dx))
        return [[0, arrow, "down"], [ms, arrow, "up"]], ms + SLIDE_MS

    def line_up_rope(self, rope_x: float) -> Fix:
        """Stand right on top of the rope at map x: walk there, then fine-tune on screen."""
        f = self.walk_to(rope_x, tolerance=12)
        for _ in range(3):
            her = SCALE * f.sight.her[0] if f.sight.her and f.sight.seen else None
            if her is None:
                return f
            expected = her + (rope_x - f.where.x) * SCREEN_SCALE
            near = [r for r in f.sight.ropes if abs(r.x - expected) <= 70]
            if not near:
                return f
            target = min(near, key=lambda r: abs(r.x - expected)).x
            dx = target - her
            if abs(dx) <= ROPE_TOLERANCE:
                return f
            direction = RIGHT if dx > 0 else LEFT
            ms = int(1000 * abs(dx) / WALK_PX_PER_S) + (40 if abs(dx) > 20 else 25)
            self._send([[0, self._arrow(direction), "down"], [ms, self._arrow(direction), "up"]], "nudge")
            time.sleep(0.2)
            f = self.fix()
        return f

    # --- route moves ------------------------------------------------------------------------

    def do(self, m: Move, f: Fix | None = None) -> Fix:
        """Carry out move `m` from where she is (`f`, or a fresh look) and return a look once she's landed."""
        f = f or self.fix()
        k = self.h.keys
        x0 = f.where.x
        if m.kind == "climb":
            self._climb(m, f)
        elif m.kind == "climb_down":
            self.line_up_rope(m.rope_x)
            hold = int(1000 * (m.rope_bottom - m.rope_top) / CLIMB_SPEED) + 500
            self._send([[0, k.down.name, "down"], [hold, k.down.name, "up"]], "climb_down")
        elif m.kind == "jump_up":
            events, t = self.walk_events(m.x - x0, 15)
            self._send(events + [[t, k.jump.name, "down"], [t + 40, k.jump.name, "up"]], "jump_up")
            time.sleep(0.45)
        elif m.kind == "jump":
            self._run_and_jump(m, f)
        elif m.kind == "walk_off":
            arrow = self._arrow(m.direction)
            ms = hold_ms(abs(m.land_x - x0) + 20)
            self._send([[0, arrow, "down"], [ms, arrow, "up"]], "walk_off")
            time.sleep(0.3)
        elif m.kind == "drop":
            events, t = self.walk_events(m.x - x0, 20)
            self._send(events + [[t, k.down.name, "down"], [t + 120, k.jump.name, "down"],
                                 [t + 160, k.jump.name, "up"], [t + 320, k.down.name, "up"]], "drop")
            time.sleep(0.3)
        else:
            raise ValueError(f"unknown move {m.kind}")
        return self.settle()

    def _run_and_jump(self, m: Move, f: Fix) -> None:
        """Run towards the end and jump on the way: jump is pressed when she's JUMP_FROM from it."""
        sign = 1 if m.direction == RIGHT else -1
        p = self.model.platform(m.src)
        edge = p.right if sign > 0 else p.left
        takeoff = edge - sign * JUMP_FROM
        run = (takeoff - f.where.x) * sign  # how far she runs before jumping
        if run < 7:  # too close to the end (or past the takeoff): back up to the run-up start first
            f = self.walk_to(m.x, tolerance=6, f=f)
            run = (takeoff - f.where.x) * sign
        arrow, jump = self._arrow(m.direction), self.h.keys.jump.name
        t = run_ms(max(run, 7))
        hold = int(t + 1000 * (air_time(0) or 0.56) + 150)
        self._send([[0, arrow, "down"], [t, jump, "down"], [t + 40, jump, "up"], [hold, arrow, "up"]], "jump")

    def _climb(self, m: Move, f: Fix) -> None:
        """Grab the rope on the way: run at it and jump holding up (or just walk into it holding up
        when it comes down to her feet). Standing under it already: jump straight up holding up."""
        k = self.h.keys
        p = self.model.platform(m.src)
        gap = p.y_at(m.rope_x) - m.rope_bottom  # how far above her feet the rope ends
        climb_ms = int(1000 * (p.y_at(m.rope_x) - m.rope_top) / CLIMB_SPEED) + 500
        dx = m.rope_x - f.where.x
        if abs(dx) <= 4:
            self._send([[0, k.up.name, "down"], [40, k.jump.name, "down"], [80, k.jump.name, "up"],
                        [climb_ms + 350, k.up.name, "up"]], "climb")
            return
        sign = 1 if dx > 0 else -1
        walk_in = gap <= 10 and not any(abs(pt.x - m.rope_x) < 60 and abs(pt.y - p.y_at(pt.x)) < 40
                                        for pt in self.model.portals)  # up at a portal would take it
        lead = 0.0 if walk_in else grab_lead(gap)
        if lead is None:
            raise ValueError(f"rope at {m.rope_x:.0f} ends {gap:.0f} above her platform: out of reach")
        run = abs(dx) - lead
        if run < 7:  # too close to jump at it: back off to a run-up first (on her platform)
            back = f.where.x - sign * (7 - run + 10)
            if not p.covers(back, -10):
                back = m.rope_x  # no room to back off: stand under it instead
            f = self.walk_to(back, tolerance=6, f=f)
            if back == m.rope_x:
                self._climb(m, f)
                return
            dx = m.rope_x - f.where.x
            sign, run = (1 if dx > 0 else -1), abs(dx) - lead
        arrow = self._arrow(RIGHT if sign > 0 else LEFT)
        t = run_ms(max(run, 7))
        at_rope = t + int(1000 * lead / WALK_SPEED)
        # Up is held from the jump on (in the air, so a portal on the way can't take her), or when
        # walking into a rope, from shortly before she reaches it.
        up_from = t if lead else max(0, t - 250)
        events = [[0, arrow, "down"], [up_from, k.up.name, "down"], [at_rope + 120, arrow, "up"],
                  [at_rope + climb_ms, k.up.name, "up"]]
        if lead:
            events += [[t, k.jump.name, "down"], [t + 40, k.jump.name, "up"]]
        self._send(events, "climb")

    def off_rope(self, f: Fix) -> Fix:
        """She's on a rope (no platform under her dot): climb up onto the platform if she's near its
        top, else jump off it sideways (towards the side a fall lands on a platform)."""
        k = self.h.keys
        x, y = f.where.x, f.where.y
        rope = min(self.model.ropes, key=lambda r: abs(r.x - x) + (0 if r.top - 10 <= y <= r.bottom + 10 else 1000))
        if y - rope.top < 70:
            self._send([[0, k.up.name, "down"], [900, k.up.name, "up"]], "off_rope")
        else:
            from .route import WALK_SPEED, fly
            side = RIGHT if fly(self.model, x + 15, y, WALK_SPEED, -300)[0] is not None else LEFT
            arrow = self._arrow(side)
            self._send([[0, arrow, "down"], [30, k.jump.name, "down"], [80, k.jump.name, "up"], [400, arrow, "up"]],
                       "off_rope")
        return self.settle(2.5)

    def settle(self, seconds: float = 2.0) -> Fix:
        """Look until she's standing still on a platform (or give up and return the last look)."""
        end = time.monotonic() + seconds
        f = self.fix()
        while time.monotonic() < end:
            time.sleep(0.08)
            g = self.fix()
            if g.where.platform is not None and f.where.platform is g.where.platform \
                    and math.dist(f.where.dot, g.where.dot) < 0.6:
                return g
            f = g
        return f


def describe(p: Platform | None) -> str:
    return "air/rope" if p is None else f"platform {p.id} ({p.left:.0f}..{p.right:.0f}, y {p.top:.0f})"
