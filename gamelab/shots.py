"""Where her stars reach, and where to shoot a platform's monsters from.

A regular attack throws one star straight ahead. Measured in game (Thicket Around the Beach II):
- in open air the star flew 400 map units (560 screen px) before vanishing;
- it hit a pig 388 units away on a platform 60 units below her (platform 14 -> 17), and killed a pig
  on a platform 60 units above her (platform 3 -> 1): stars aren't stopped by platforms, and the
  target can be lower or higher than she is, within a band;
- it leaves her ~530 ms after the key and flies ~690 units/s;
- right next to a monster, the claw swipes instead of throwing (weak, and the monster can touch her).

All in map units; dy is the target's feet minus hers (+ = below her).
"""

from dataclasses import dataclass

import numpy as np

from .mapdata import MapModel, Platform

STAR_REACH = 390     # the farthest hit seen (388); the star flies 400
SHOT_RANGE = 340     # aim inside this: the target keeps walking during the ~0.5 s wind-up
MIN_SHOT = 90        # closer than this she swipes with the claw instead of throwing
# How far below/above her a target can be and still be hit: 60 both ways measured (hits). Farther
# isn't measured yet; `python -m gamelab hunt shots` lists what the shot log has seen.
SHOT_DOWN = 60
SHOT_UP = 60
# Off her level a star needs more room sideways: the shot log hit monsters 60 above her 7 of 15 times
# at 150-199 units (0 of 1 closer), 70% at 200-249, 85% beyond; 60 below, 56% at 100-149, 79% at
# 150-199, 70-92% beyond. A monster right under or over her isn't hit at all.
MIN_SHOT_ABOVE = 200  # closest sideways a target 60 above her is worth a throw
MIN_SHOT_BELOW = 150  # ... and one 60 below
LEVEL = 15           # within this of her height, a target counts as on her level
STAR_SPEED = 690     # units/s
EDGE_MARGIN = 25     # where she stands to shoot: not closer than this to her platform's end


def min_dx(dy: float) -> float:
    """The closest sideways a target dy below (+) or above (-) her is hit: MIN_SHOT on her level,
    rising to MIN_SHOT_BELOW / MIN_SHOT_ABOVE at 60 below / above."""
    far = MIN_SHOT_BELOW if dy > 0 else MIN_SHOT_ABOVE
    off = min(max(abs(dy) - LEVEL, 0.0) / (60 - LEVEL), 1.0)
    return MIN_SHOT + (far - MIN_SHOT) * off


def can_shoot(dx: float, dy: float, reach: float = SHOT_RANGE) -> bool:
    """Whether a target dx along (either way) and dy below (+) is hit by a star from where she stands."""
    return min_dx(dy) <= abs(dx) <= reach and -SHOT_UP <= dy <= SHOT_DOWN


@dataclass(frozen=True)
class Spot:
    """Where to stand to shoot at a platform: on `platform` at map x `x`."""
    platform: int
    x: float
    coverage: float   # fraction of the target platform's length in reach from there
    safe: bool        # no monster can reach her on that platform (MapModel.safe)


def coverage(target: Platform, x: float, y: float, reach: float = SHOT_RANGE) -> float:
    """Fraction of `target` in reach of a star thrown from (x, y), either way."""
    dy = target.y_at(min(max(x, target.left), target.right)) - y
    if not -SHOT_UP <= dy <= SHOT_DOWN:
        return 0.0
    covered, near = 0.0, min_dx(dy)
    for lo, hi in ((x - reach, x - near), (x + near, x + reach)):
        covered += max(0.0, min(hi, target.right) - max(lo, target.left))
    return covered / max(1.0, target.right - target.left)


def spots(model: MapModel, target: int, step: float = 10) -> list[Spot]:
    """Places to shoot platform `target` from, best first: safe ones (no spawns where she stands),
    then by how much of the target platform is in reach."""
    q = model.platform(target)
    out = []
    for p in model.platforms:
        lo, hi = p.left + EDGE_MARGIN, p.right - EDGE_MARGIN
        if hi < lo:
            continue
        best = None
        for x in np.arange(lo, hi + 0.1, step):
            c = coverage(q, float(x), p.y_at(float(x)))
            if c > 0 and (best is None or c > best[0] + 1e-9):
                best = (c, float(x))
        if best:
            out.append(Spot(p.id, best[1], round(best[0], 3), model.safe(p.id)))
    return sorted(out, key=lambda s: (not s.safe, -s.coverage))


def hp_bar(frame: np.ndarray, x: int, top: int, width: int = 45) -> float | None:
    """A monster's HP bar, shown above it once it has been hit: the fraction still green, or None
    if there's no bar. `frame` is a half-size RGB game frame; (x, top) the monster's centre and top.

    The bar's inside is pure colours, rare in the scenery: green (0, g, 0) for the HP left, black for
    HP lost, and for a moment dark red (r, 0, 0) for the latest damage. It sits ~18 px above the
    monster (half size) and is ~35 px wide."""
    y0, x0 = max(0, top - 30), max(0, x - width)
    box = frame[y0:max(y0, top + 2), x0:x + width].astype(np.int16)
    if box.size == 0:
        return None
    r, g, b = box[..., 0], box[..., 1], box[..., 2]
    green = (g > 120) & (r < 15) & (b < 15)
    # Any shade of pure green (the half-size frame blends green and black where they meet), black,
    # or the dark red of the latest damage.
    inside = (b < 15) & ((r < 15) | ((r > 90) & (g < 50)))
    best = None
    for y in range(inside.shape[0]):
        run, start = 0, 0
        for i, on in enumerate(inside[y]):
            run = run + 1 if on else 0
            if run == 1:
                start = i
            if run >= 22 and (best is None or run > best[0]):
                best = (run, y, start)
    if best is None:
        return None
    run, y, start = best
    return float(green[y, start:start + run].sum()) / run
