"""Carrying out route moves in the game, closed loop: look, act, look again.

Her position along a platform comes from two sources:
- the minimap dot (anywhere, but only good to ~11 map units);
- the screen, when the end of her platform is in view: her name tag's distance to that end, which
  is good to a couple of units (and matters for jumps that take off near an end).
Ropes are lined up on screen (the detected rope nearest to where the map says it is).
"""

import time
from dataclasses import dataclass

from maplehelper.engine import LEFT, RIGHT

from .hunt import SCALE, Hunter, Sight
from .localize import Localizer, Where
from .mapdata import SCREEN_SCALE, MapModel, Platform
from .route import CLIMB_SPEED, WALK_SPEED, Move, air_time

WALK_PX_PER_S = WALK_SPEED * SCREEN_SCALE
START_LAG = 0.10          # s between pressing an arrow and moving
X_TOLERANCE = 8           # map units: close enough when lining up
ROPE_TOLERANCE = 6        # screen px: close enough to a rope to grab it


@dataclass
class Fix:
    """One look: where she is, and how it was worked out."""
    where: Where
    sight: Sight
    how: str              # 'edge' (screen) or 'dot'


class Mover:
    def __init__(self, hunter: Hunter, model: MapModel, localizer: Localizer, log=print):
        self.h, self.model, self.loc, self.log = hunter, model, localizer, log
        hunter.eyes.want_ropes = True

    # --- looking ----------------------------------------------------------------------------

    def fix(self, tries: int = 6) -> Fix:
        for _ in range(tries):
            s = self.h.look()
            if s.dot is not None:
                where = self.loc.where_dot(s.dot)
                return self._refine(where, s)
            time.sleep(0.25)
        raise RuntimeError("can't find her minimap dot")

    def _refine(self, where: Where, s: Sight) -> Fix:
        p = where.platform
        if p is None or s.her is None or s.ground is None:
            return Fix(where, s, "dot")
        her_x = SCALE * s.her[0]
        left, right = SCALE * s.ground[0], SCALE * s.ground[1]
        width = s.frame.shape[1] * SCALE
        candidates = []
        if right < width - 12:  # her platform's right end is on screen
            candidates.append(p.right - (right - her_x) / SCREEN_SCALE)
        if left > 12:
            candidates.append(p.left + (her_x - left) / SCREEN_SCALE)
        good = [x for x in candidates if abs(x - where.x) <= 45]  # must agree with the dot roughly
        if not good:
            return Fix(where, s, "dot")
        x = sum(good) / len(good)
        return Fix(Where(x, p.y_at(x), p, where.dot), s, "edge")

    # --- basic actions ----------------------------------------------------------------------

    def _arrow(self, direction: str) -> str:
        return (self.h.keys.left if direction == LEFT else self.h.keys.right).name

    def _send(self, events, name):
        self.h._do(events, name)

    def walk_to(self, x: float, tolerance: float = X_TOLERANCE, tries: int = 4) -> Fix:
        """Walk along her platform until her feet are at map x."""
        f = self.fix()
        for _ in range(tries):
            dx = x - f.where.x
            if abs(dx) <= tolerance:
                return f
            direction = RIGHT if dx > 0 else LEFT
            ms = int(1000 * (abs(dx) / WALK_SPEED + START_LAG)) if abs(dx) > 25 else 60
            self._send([[0, self._arrow(direction), "down"], [ms, self._arrow(direction), "up"]], "walk")
            time.sleep(0.25)
            f = self.fix()
        return f

    def line_up_rope(self, rope_x: float) -> Fix:
        """Stand right under/over the rope at map x: walk there, then fine-tune on screen."""
        f = self.walk_to(rope_x, tolerance=12)
        for _ in range(4):
            her = SCALE * f.sight.her[0] if f.sight.her else None
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

    def do(self, m: Move) -> Fix:
        k = self.h.keys
        if m.kind == "climb":
            f = self.line_up_rope(m.rope_x)
            hold = int(1000 * (0.35 + (f.where.y - m.rope_top) / CLIMB_SPEED)) + 400
            self._send([[0, k.up.name, "down"], [40, k.jump.name, "down"], [80, k.jump.name, "up"],
                        [hold, k.up.name, "up"]], "climb")
            time.sleep(0.3)
        elif m.kind == "climb_down":
            self.line_up_rope(m.rope_x)
            hold = int(1000 * (m.rope_bottom - m.rope_top) / CLIMB_SPEED) + 500
            self._send([[0, k.down.name, "down"], [hold, k.down.name, "up"]], "climb_down")
            time.sleep(0.6)
        elif m.kind == "jump_up":
            self.walk_to(m.x, tolerance=15)
            self._send([[0, k.jump.name, "down"], [40, k.jump.name, "up"]], "jump_up")
            time.sleep(0.8)
        elif m.kind == "jump":
            self.walk_to(m.x, tolerance=8)
            arrow = self._arrow(m.direction)
            air = air_time(0) or 0.56
            hold = int(1000 * (START_LAG + air + 0.15))
            self._send([[0, arrow, "down"], [int(1000 * START_LAG), k.jump.name, "down"],
                        [int(1000 * START_LAG) + 40, k.jump.name, "up"], [hold, arrow, "up"]], "jump")
            time.sleep(0.5)
        elif m.kind == "walk_off":
            f = self.fix()
            arrow = self._arrow(m.direction)
            ms = int(1000 * (abs(m.land_x - f.where.x) / WALK_SPEED + START_LAG + 0.1))
            self._send([[0, arrow, "down"], [ms, arrow, "up"]], "walk_off")
            time.sleep(0.9)
        elif m.kind == "drop":
            self.walk_to(m.x, tolerance=20)
            self._send([[0, k.down.name, "down"], [120, k.jump.name, "down"], [160, k.jump.name, "up"],
                        [320, k.down.name, "up"]], "drop")
            time.sleep(0.9)
        else:
            raise ValueError(f"unknown move {m.kind}")
        return self.settle()

    def settle(self, seconds: float = 2.0) -> Fix:
        """Look until she's standing on a platform (or give up and return the last look)."""
        end = time.monotonic() + seconds
        f = self.fix()
        while f.where.platform is None and time.monotonic() < end:
            time.sleep(0.2)
            f = self.fix()
        return f


def describe(p: Platform | None) -> str:
    return "air/rope" if p is None else f"platform {p.id} ({p.left:.0f}..{p.right:.0f}, y {p.top:.0f})"
