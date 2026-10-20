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

What stops a star is the walls of her own foothold group (mapdata.Segment.group: a floating block's floor
and the walls hanging from its ends, or platforms joined by a step), where they cross a level line SHOT_H
above her feet (higher in a jump): `wall_in_way`. Floors never stop one, and nor do the walls of any
other group, such as the side walls of the block the target stands on. The game data has no foothold
flag for it (306 map images: a foothold has its ends, its neighbours and `piece`, and in 51 maps
`forbidFallDown`); the rule is fitted to the live logs of 2026-10-18 to 10-20:
- from platform 4, no star reached a monster on 2, 60 above her behind the step wall joining the two
  (group (3, 63)), in 3 sessions: 0 of 3 in the shot log, every kill of 40 throws on 4 (sweep log 9),
  0 of 5 in the telemetry (Orange Mushrooms on 2 in view, no HP bar, for 4 s of throwing);
- in the same logs, stars 60 up past the side wall of the block above (another group, from 25 below
  to 60 above her feet) hit: 19 -> 18 13 of 13 decided, 3 -> 1 15 of 19; and stars down past her own
  block's hanging wall (14 -> 17, 25 deep) hit 81%;
- a straight line from her to the target fits neither 4 -> 2 and 14 -> 17 together, nor "every wall
  blocks" 19 -> 18; the level line through her own group's walls fits every pair decided.
The client's collisions in the air also skip walls of other groups (a reconstruction of its physics),
likely the same test. On this map it changes one pair: 4 -> 2.

All in map units; dy is the target's feet minus hers (+ = below her).
"""

import math

from ..localization.mapdata import Platform, Segment
from ..localization.monsters import Monster
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
# What stops a star (see above). The level line runs SHOT_H above her feet: anything from 1 to 59 fits the
# logs (on this map it matters only for steps lower than 60, and there are none), and 28 is where the
# client's target search is centred (MeowDB, Lucky Seven: 28 above her feet).
SHOT_H = 28
# Whether her decisions keep to it. Off, they throw at monsters behind such walls as before, so an in-game
# probe can measure it (the shot log has the rule's verdict either way; the simulator's world keeps to the
# rule regardless).
WALLS_BLOCK = True


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
    """Whether a target dx along (either way) and dy below (+) is hit by a star from where she stands,
    terrain aside (can_hit: with it)."""
    return min_dx(dy) <= abs(dx) <= reach and in_band(dy)


def _crosses(w: Segment, line: float) -> bool:
    """Whether wall `w` crosses the level line at map y `line` (a wall split into footholds, end to end,
    crosses it once: each foothold from its top, inclusive, to its bottom)."""
    return min(w.y1, w.y2) <= line < max(w.y1, w.y2)


def _inside(x: float, p: Platform) -> float:
    """Map x, 1 unit inside platform `p`'s ends."""
    return min(max(x, p.left + 1), p.right - 1)


def wall_between(platform: Platform | None, x: float, y: float, tx: float, rise: float = 0.0,
                 target: Platform | None = None) -> Segment | None:
    """The wall that stops a star from her (map x on `platform`, her feet at y, `rise` units up in a jump)
    before map x `tx`: of the walls of her own foothold group (Platform.walls), the nearest standing
    between them across the level line SHOT_H above her feet; None if the star gets there. Her x is taken
    1 unit inside her platform's ends, and `tx` inside the target's platform (`target`, if known): a
    monster is never past its platform's end, and one at the foot of a wall of her group that rises from
    its own platform's end (the basin's sides, from its floor) isn't behind it."""
    if platform is None:
        return None
    line = y - rise - SHOT_H
    a = _inside(x, platform)
    b = tx if target is None else _inside(tx, target)
    lo, hi = min(a, b), max(a, b)
    walls = [w for w in platform.walls if lo <= w.x1 <= hi and _crosses(w, line)]
    return min(walls, key=lambda w: abs(w.x1 - a), default=None)


def wall_in_way(platform: Platform | None, x: float, y: float, tx: float, rise: float = 0.0,
                target: Platform | None = None) -> Segment | None:
    """wall_between, as her decisions see it: None whatever the terrain with WALLS_BLOCK off."""
    return wall_between(platform, x, y, tx, rise, target) if WALLS_BLOCK else None


def wall_to(platform: Platform | None, x: float, y: float, m: Monster, rise: float = 0.0,
            target: Platform | None = None) -> Segment | None:
    """wall_in_way from map (x, y) on `platform` to monster `m`. `target`: m's platform if known (if m is
    on hers, it's hers)."""
    if target is None and platform is not None and m.platform == platform.id:
        target = platform
    return wall_in_way(platform, x, y, m.x, rise, target)


def can_hit(platform: Platform | None, x: float, y: float, m: Monster, reach: float = SHOT_RANGE,
            rise: float = 0.0, target: Platform | None = None) -> bool:
    """Whether a star from map (x, y) on `platform` hits monster `m`: in reach (can_shoot), and no wall of
    her foothold group in the way (wall_to; `rise`: her height in a jump, `target`: m's platform)."""
    return can_shoot(m.x - x, m.y - y, reach) and wall_to(platform, x, y, m, rise, target) is None


def clear_span(platform: Platform | None, x: float, y: float, rise: float = 0.0) -> tuple[float, float]:
    """The map x a star from her (as in wall_in_way) gets to either way before a wall of her foothold group
    stops it: from the nearest such wall on her left to the nearest on her right (infinite where there's
    none, and either way with WALLS_BLOCK off)."""
    if not WALLS_BLOCK or platform is None:
        return -math.inf, math.inf
    line, a = y - rise - SHOT_H, _inside(x, platform)
    xs = [w.x1 for w in platform.walls if _crosses(w, line)]
    return max((v for v in xs if v <= a), default=-math.inf), min((v for v in xs if v >= a), default=math.inf)


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
