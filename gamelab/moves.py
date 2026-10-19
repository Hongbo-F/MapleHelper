"""Carrying out route moves in the game, closed loop: look, act, look again.

Her position along a platform, best source first:
- a rope in view: ropes are drawn exactly at their map x and detected to ~2 px;
- the visible end of her platform (its real end is ~24 px past where the grass seems to end);
- the minimap dot (anywhere, but only good to ~11 map units).
Precision matters for running jumps, which take off near an end. Ropes are lined up on screen.
"""

import time
from dataclasses import dataclass

from maplehelper.engine import LEFT, RIGHT

from .hunt import SCALE, Hunter, Sight
from .localize import Localizer, Where
from .mapdata import SCREEN_SCALE, MapModel, Platform
from .route import CLIMB_SPEED, RUN_MS, WALK_SPEED, Move, air_time

WALK_PX_PER_S = WALK_SPEED * SCREEN_SCALE
START_LAG = 0.10          # s between pressing an arrow and moving
# Measured: holding an arrow for t ms moves her ~(12 + 0.209 t) screen px including the slide after
# release, i.e. 8.5 + 0.149 t map units. Shorter than ~30 ms barely registers.
WALK_UNITS_PER_MS = 0.209 / SCREEN_SCALE
WALK_UNITS_FIXED = 12 / SCREEN_SCALE
MIN_HOLD_MS = 30
# Screen px the drawn ground sticks out past a platform's real end. Negative: measured against ropes
# on platforms 32, 30 and 14, the real ends are 21-27 px beyond where the grass seems to end.
GRASS_OVERHANG = -24


def hold_ms(units: float) -> int:
    """How long to hold an arrow to walk `units` map units and come to rest there."""
    return max(MIN_HOLD_MS, round((units - WALK_UNITS_FIXED) / WALK_UNITS_PER_MS))
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

    def rope_x(self, where: Where, s: Sight) -> float | None:
        """Her x from ropes in view: each is drawn exactly at its map x, and detected to ~2 px."""
        if s.her is None or not s.ropes:
            return None
        her_x = SCALE * s.her[0]
        estimates = []
        for seen in s.ropes:
            for rope in self.model.ropes:
                expected = her_x + (rope.x - where.x) * SCREEN_SCALE  # where the dot says it should be
                if abs(seen.x - expected) <= 40:
                    estimates.append(rope.x - (seen.x - her_x) / SCREEN_SCALE)
        if not estimates:
            return None
        estimates.sort()
        return estimates[len(estimates) // 2]

    def edge_x(self, where: Where, s: Sight) -> float | None:
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

    def _refine(self, where: Where, s: Sight) -> Fix:
        p = where.platform
        if p is None:
            return Fix(where, s, "dot")
        for how, x in (("rope", self.rope_x(where, s)), ("edge", self.edge_x(where, s))):
            if x is not None and abs(x - where.x) <= 45:
                x = min(max(x, p.left), p.right)
                return Fix(Where(x, p.y_at(x), p, where.dot), s, how)
        return Fix(where, s, "dot")

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
            ms = hold_ms(abs(dx))
            self._send([[0, self._arrow(direction), "down"], [ms, self._arrow(direction), "up"]], "walk")
            time.sleep(0.3)  # let her slide to a stop before looking
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
            self.walk_to(m.x, tolerance=8)  # m.x is RUN_START back from the end
            arrow = self._arrow(m.direction)
            hold = int(RUN_MS + 1000 * (air_time(0) or 0.56) + 150)
            self._send([[0, arrow, "down"], [RUN_MS, k.jump.name, "down"], [RUN_MS + 40, k.jump.name, "up"],
                        [hold, arrow, "up"]], "jump")
            time.sleep(0.5)
        elif m.kind == "walk_off":
            f = self.fix()
            arrow = self._arrow(m.direction)
            ms = hold_ms(abs(m.land_x - f.where.x) + 20)
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
