"""Where her stars reach: how far, how far below and above her, how close is too close, and what an
attack from a jump can hit.

A regular attack throws one star straight ahead. Measured in game (Thicket Around the Beach II):
- in open air the star flew 400 map units (560 screen px) before vanishing;
- it hit a pig 388 units away on a platform 60 units below her (platform 14 -> 17), and killed a pig
  on a platform 60 units above her (platform 3 -> 1): stars aren't stopped by platforms, and the
  target can be lower or higher than she is, within a band;
- it leaves her ~530 ms after the key and flies ~690 units/s;
- right next to a monster, the claw swipes instead of throwing (weak, and the monster can touch her).

What an attack hits is picked when it starts, about 55 ms after its key, not when the star flies (the
star is only a picture): the nearest monster whose body reaches into an area fixed to her feet, from
AREA_DOWN below them to AREA_UP above, up to STAR_REACH ahead. From a jump the area is as high as she is
then. A jump throw's attack starts 42-49 units up, so its area passes over a snail (24-26 tall; a
monster's body is its box in the game data, localization.monsters): in 45 jump-throws at snails 3-77
units away on 2026-10-19, at most one hurt the snail, and the monsters on the platform 60 above took the
stars. On the ground the claw's swing is looked at first: a monster whose body reaches into it is swiped.

All in map units; dy is the target's feet minus hers (+ = below her).
"""

from ..motion import physics

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
# The area an attack picks its target in (see above). Its floor is fitted to the live throws of
# 2026-10-19 (jump-throws missed snails; throws from the ground hit pigs, slimes and Orange Mushrooms 60
# below her): 18-21 below her feet. Of its top, only that monsters 60 above her are in it is known.
AREA_DOWN = 19
AREA_UP = 65
AIR_MARGIN = 6       # how far into the area a body must reach for an attack from a jump to pick it surely:
                     # about one frame of her climb when a jump throw's attack starts
# The claw's swings (game data: the claw's afterImage, swordOL: swingO1-O3 and stabO1) reach from 17 below
# to 62 above her feet, and 73-102 ahead of her (MIN_SHOT is taken as their reach).
CLAW_DOWN = 17
CLAW_UP = 62
STAR_SURE = 125      # from this far on, a throw from the ground surely throws a star at a snail: the widest
                     # swing (swingO3) reaches 102 ahead of her, and half a snail is 19


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


def reaches_into(ahead: float, dy: float, top: float, half: float, far: float, down: float, up: float) -> bool:
    """Whether a monster's body reaches into a box in front of her: from her (0) to `far` ahead, `down`
    below to `up` above her feet. The monster's middle is `ahead` units in front of her, its feet `dy`
    below hers (+ below), and its body `top` tall and `half` its width either side."""
    return -half <= ahead <= far + half and top - dy >= -down and -dy < up


def in_area(ahead: float, dy: float, top: float, rise: float = 0.0, half: float = 0.0) -> bool:
    """Whether an attack can pick a monster `ahead` units in front of her, its feet `dy` below hers as she
    stands (+ below), its body `top` tall (and `half` wide either side), when it starts with her `rise`
    units up in a jump: its body reaches into the area then (AREA_DOWN to AREA_UP about her feet, up to
    STAR_REACH ahead)."""
    return reaches_into(ahead, dy + rise, top, half, STAR_REACH, AREA_DOWN, AREA_UP)


def air_rise(gap_ms: float) -> float:
    """How high she is above the ground she jumped from when an attack keyed `gap_ms` after the jump key
    picks its target: JUMP_LAG after its key, as a jump lifts off, or up to ATTACK_DECIDE_LAG_MS later
    (physics). 49 units at 100 ms (a jump throw), 44 at 85 ms (a jump back), 26 at 40 ms."""
    return physics.jump_rise(gap_ms + physics.JUMP_LAG * 1000 + physics.ATTACK_DECIDE_LAG_MS)


def air_floor(gap_ms: float) -> float:
    """How high above the ground she jumped from the area of an attack keyed `gap_ms` after the jump key
    starts (air_rise less AREA_DOWN): 30 at 100 ms, 25 at 85 ms, 7 at 40 ms."""
    return air_rise(gap_ms) - AREA_DOWN


def air_hits(top: float, gap_ms: float) -> bool:
    """Whether an attack keyed `gap_ms` after the jump key surely picks a monster on her platform that is
    at least `top` tall (localization.monsters.MONSTER_TOP): its body reaches AIR_MARGIN into the area.
    A snail (24) stays under the area of a jump throw (its floor 30 up) and of a jump back (25); keyed
    40 ms after the jump (7), the attack picks every kind."""
    return top >= air_floor(gap_ms) + AIR_MARGIN
