"""Ropes and ladders as places to wait out monsters. The user, 2026-10-24: "holding on a rope can also be a safe
ground sometimes (less ideal because you can't recover hp while hanging on the rope)".

A rope is a place she can get onto from a platform and back (`refuges`): one whose bottom she grabs jumping straight
up from under it (its bottom within the planner's grab reach above her platform, physics.ROPE_REACH less the
mover's GRAB_MARGIN; she comes back down holding down past its bottom, onto the same platform), or one hanging from
her platform (down onto it from its top; back up over its top: Rope.from_platform). Where on it she hangs
(`Refuge.y`) is the safest height (attack.danger's `on_rope`: a monster that walks its floor touches her there only
where its body, a hop's frames too, reaches hers), of those as safe the one quickest to get to and back from. What it
costs, besides its danger:
- getting on and off takes time (`Refuge.on_s`, `off_s`: the jump to its bottom or the step onto its top, the climb
  at CLIMB_SPEED, and back down: the climb, then the fall from its bottom, or the climb-over at its top), in which
  she throws at nothing;
- she regains no HP there (REGEN_S, a share of her HP a second, standing; a thief's Endure, the skill that lets her
  recover on a rope, is a second job's).
The hunt loop hangs on one only while it waits for respawns and no place along her platform is ROPE_GAIN safer than
it (behavior.hunt, rule 9): never as the default. Pure, from the map (localization.mapdata) and the danger.
"""

import math
from dataclasses import dataclass

from ..localization.mapdata import MapModel, Platform, Rope
from ..localization.monsters import Monster
from ..motion.flow import CLIMB_OVER
from ..motion.physics import CLIMB_SPEED, FALL_SPEED, GRAVITY, JUMP_SPEED, LAGS, ROPE_REACH
from ..motion.route import ROPE_TOP_DY, grab_height, hung_from
from .danger import Danger

GRAB_MARGIN = 8.0    # a rope's bottom she grabs from under it is this much within her reach at least (control.mover's)
TOP_KEEP = 70.0      # hanging from under a rope she keeps her feet this far below its top: nearer, the way off a rope
                     # (control.mover.off_rope) climbs her up over it onto the platform there
END_KEEP = 10.0      # ... and on one from her platform's top, this far above its bottom (past it she lets go)
GRAB_CLIMB = 20.0    # ... and from under one, this far above where she grabs it at least: up is held past the grab (let
                     # go as she got to it, she missed it, and flew on off her platform)
ROPE_IN = 15.0       # a rope got onto from under it is this far in from her platform's ends at least (her half width:
                     # letting go of it she lands back on the platform). Tree Dungeon's ropes over the ends of 20 (1
                     # unit in) and 19 (6 in) aren't: from the one over 20 in the simulator she came down on 21 below
HANG_STEP = 10.0     # map units between the heights on a rope tried
HANG_SLACK = 15.0    # she ends up within this of the height she climbs to (the mover's HANG_TOLERANCE: her dot is good
                     # to ~11 units, the climb's hold to a few): a height counts the worst danger within it, so at a
                     # rope's bottom a monster's top must be this far under her feet too
# Her natural HP recovery standing, a share of her HP a second, which she doesn't get hanging on a rope. A guess: the
# game's base recovery is said to be about 10 HP every 10 s (not measured; nothing of hers recovers more until her
# second job's Endure, which also works on a rope): 1 of her 752 HP a second at level 24.
REGEN_S = 1.0 / 752.0


@dataclass(frozen=True)
class Refuge:
    """Where she'd wait on a rope: `rope`, with her feet at map y `y`, got onto from platform `platform` (and back onto
    it), from its top (`from_top`: it hangs from that platform) or from under its bottom; `on_s` s from standing at its
    x to hanging there, `off_s` s from there to standing on the platform again, and the danger there (attack.danger's
    on_rope, with the look it was worked out from)."""
    rope: Rope
    platform: int
    y: float
    from_top: bool
    on_s: float
    off_s: float
    danger: float = 0.0

    @property
    def x(self) -> float:
        """The rope's map x: where she gets on and comes back down."""
        return self.rope.x


def refuges(model: MapModel, danger: Danger, pid: int, seen: list[Monster] | None = None) -> list[Refuge]:
    """The ropes she can get onto from platform `pid` and back, each at its safest height for her (with the monsters
    `seen` and the spawn points they don't account for; with no look, every spawn point's monster at its point), the
    safest first. None on a platform no rope can be got onto from."""
    p = model.platform(pid)
    out = []
    for rope in model.ropes:
        way = _way(model, p, rope)
        if way is not None:
            out.append(_hang(danger, p, rope, way, seen))
    return sorted(out, key=lambda r: (round(r.danger, 6), r.on_s + r.off_s))


def _way(model: MapModel, p: Platform, rope: Rope) -> str | None:
    """How she gets onto `rope` from platform `p` and back: 'top' (it hangs from p, and she can climb back up over its
    top), 'under' (its bottom within her grab reach above p, and letting go there lands her on p), or None."""
    if not p.covers(rope.x):
        return None
    if rope.from_platform and hung_from(model, rope) is p and abs(p.y_at(rope.x) - rope.top) <= ROPE_TOP_DY:
        return "top"
    gap = p.y_at(rope.x) - rope.bottom  # how far above her feet it ends
    if (0 <= gap <= ROPE_REACH - GRAB_MARGIN and p.left + ROPE_IN <= rope.x <= p.right - ROPE_IN
            and model.platform_below(rope.x, rope.bottom) is p):
        return "under"
    return None


def _hang(danger: Danger, p: Platform, rope: Rope, way: str, seen: list[Monster] | None) -> Refuge:
    """The Refuge on `rope` got onto from `p` (`way`: _way's) at its safest height, the quickest to get to and back
    from of those as safe."""
    floor = p.y_at(rope.x)
    if way == "top":
        lo, hi, start = rope.top + END_KEEP, rope.bottom - END_KEEP, rope.top
    else:
        start = grab_height(floor, rope)
        lo, hi = rope.top + TOP_KEEP, start - GRAB_CLIMB
    ys = [lo + i * HANG_STEP for i in range(int(max(0.0, hi - lo) // HANG_STEP) + 1)] or [hi]
    best = None
    for y in ys:
        on, off = _times(way, rope, floor, start, y)
        here = max(danger.on_rope(rope, y + dy, seen) for dy in (-HANG_SLACK, 0.0, HANG_SLACK))
        key = (round(here, 6), on + off)
        if best is None or key < best[0]:
            best = (key, Refuge(rope, p.id, y, way == "top", round(on, 3), round(off, 3), here))
    return best[1]


def _times(way: str, rope: Rope, floor: float, start: float, y: float) -> tuple[float, float]:
    """Seconds getting onto `rope` to her feet at y (from `start`, where she's on it first), and back onto the floor
    at y `floor` under or at its top."""
    climb = abs(y - start) / CLIMB_SPEED
    if way == "top":  # a step down onto it from its top; back up and over it
        return LAGS.on + climb, (y - rope.top) / CLIMB_SPEED + CLIMB_OVER
    rise = LAGS.jump + JUMP_SPEED / GRAVITY  # the jump to its bottom, up held
    drop = floor - rope.bottom              # letting go at its bottom: the fall onto her platform
    fall = math.sqrt(2 * max(0.0, drop) / GRAVITY)
    if GRAVITY * fall > FALL_SPEED:  # (never on a rope within her jump: under 0.34 s to the fall cap)
        fall = FALL_SPEED / GRAVITY + (drop - FALL_SPEED ** 2 / (2 * GRAVITY)) / FALL_SPEED
    return rise + climb, (rope.bottom - y) / CLIMB_SPEED + fall
