"""Where her stars reach: how far, how far below and above her, and how close is too close.

A regular attack throws one star straight ahead. Measured in game (Thicket Around the Beach II):
- in open air the star flew 400 map units (560 screen px) before vanishing;
- it hit a pig 388 units away on a platform 60 units below her (platform 14 -> 17), and killed a pig
  on a platform 60 units above her (platform 3 -> 1): stars aren't stopped by platforms, and the
  target can be lower or higher than she is, within a band;
- it leaves her ~530 ms after the key and flies ~690 units/s;
- right next to a monster, the claw swipes instead of throwing (weak, and the monster can touch her).

All in map units; dy is the target's feet minus hers (+ = below her).
"""


# With Keen Eyes maxed (+200 range by the skill data; the 400 above was measured before), stars on her
# level hit slimes 408, 410 and 452 units off, and missed at 560 and 634 (lab/range_probe.jsonl).
STAR_REACH = 460     # the farthest hit seen (452); somewhere short of 560 they stop hitting
SHOT_RANGE = 410     # aim inside this: the target keeps walking during the ~0.5 s wind-up and the flight
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
OFF_LEVEL_DY = 60    # the height those two were measured at (60 above / below her)
LEVEL = 15           # within this of her height, a target counts as on her level
STAR_SPEED = 690     # units/s
EDGE_MARGIN = 25     # where she stands to shoot: not closer than this to her platform's end
# Where a look searches for monsters: from this far above her feet to this far below (bodies stick up
# above their feet, so the band reaches higher than she can shoot).
LOOK_UP = SHOT_UP + 80
LOOK_DOWN = SHOT_DOWN + 15


def in_band(dy: float) -> bool:
    """Whether a target dy below (+) or above (-) her is level enough with her for a star to hit."""
    return -SHOT_UP <= dy <= SHOT_DOWN


def min_dx(dy: float) -> float:
    """The closest sideways a target dy below (+) or above (-) her is hit: MIN_SHOT on her level,
    rising to MIN_SHOT_BELOW / MIN_SHOT_ABOVE at OFF_LEVEL_DY below / above."""
    far = MIN_SHOT_BELOW if dy > 0 else MIN_SHOT_ABOVE
    off = min(max(abs(dy) - LEVEL, 0.0) / (OFF_LEVEL_DY - LEVEL), 1.0)
    return MIN_SHOT + (far - MIN_SHOT) * off


def can_shoot(dx: float, dy: float, reach: float = SHOT_RANGE) -> bool:
    """Whether a target dx along (either way) and dy below (+) is hit by a star from where she stands."""
    return min_dx(dy) <= abs(dx) <= reach and in_band(dy)
