"""Where her stars reach: how far, how far below and above her, how close is too close, what an attack
from the ground or from a jump picks, and what stops a star.

A regular attack throws one star straight ahead. Measured in game (Thicket Around the Beach II):
- in open air the star flew 400 map units (560 screen px) before vanishing;
- it hit a pig 388 units away on a platform 60 units below her (platform 14 -> 17), and killed a pig
  on a platform 60 units above her (platform 3 -> 1): stars aren't stopped by platforms, and the
  target can be lower or higher than she is, within a band;
- it leaves her ~530 ms after the key and flies ~690 units/s;
- right next to a monster, the claw swipes instead of throwing (weak, and the monster can touch her).

What an attack hits is picked when it starts, about 67 ms after its key, not when the star flies (the
star is only a picture), by the monsters' bodies (their boxes in the game data, localization.monsters).
On the ground the claw's swing is looked at first: a monster on her level whose body reaches into it is
swiped (`claw_takes`), so her decisions count on a star at one only from farther off, by its size
(`claw_edge`: a pig from 136 units, a snail from 121). Else the star's target search (`search_band`):
bands SEARCH_STEP wide from SEARCH_START ahead of her out to STAR_RANGE, band k a window from
SEARCH_FLOOR0 - SEARCH_GROW k to SEARCH_TOP0 + SEARCH_GROW k above her feet, so it opens one unit up and
one down for every four ahead; the star goes to a monster whose body overlaps the first band any body
does, the nearest of those. Off her
level a star so needs room sideways, the more the farther off her level and the smaller the monster
(`min_dx`, by its body: `body_of`): 60 below her an Orange Mushroom is picked from 113 units, a pig from
151, a slime from 177, a snail from 246; 60 above, each kind from 191-207. The search is MapleStory
Classic's (this game), as meowdb's Lucky Seven page draws it; the shot log and the live tours' telemetry
agree with its shape, and fit its floor exactly but not its top (SEARCH_TOP0). From a jump it doesn't fit
(40 ms after the jump, stars killed snails 36-66 units away, closer than a search starting 65 ahead
picks), so an attack from a jump keeps to the area that does: a box fixed to her feet, as high as she is
then (`in_area`), from AREA_DOWN below them to AREA_UP above, STAR_RANGE ahead. A jump throw's attack
keyed 100 ms after the jump started 42-49 units up, its area over a snail (24-26 tall): in 45 jump-throws
at snails 3-77 units away on 2026-10-19, at most one hurt the snail, and the monsters on the platform 60
above took the stars.

What stops a star is the walls of her own foothold group (mapdata.Segment.group: a floating block's floor
and the walls hanging from its ends, or platforms joined by a step), where they cross a level line SHOT_H
above her feet (higher in a jump): `wall_in_way`. Floors never stop one, and nor do the walls of any
other group, such as the side walls of the block the target stands on. The game data has no foothold
flag for it (306 map images: a foothold has its ends, its neighbours and `piece`, and in 51 maps
`forbidFallDown`); the rule is fitted to the live logs of 2026-10-18 to 10-20:
- from platform 4, no star reached a monster on 2, 60 above her behind the step wall joining the two
  (group (3, 63)), in 3 sessions: 0 of 3 in the shot log, every kill of 40 throws on 4 (sweep log 9),
  0 of 5 in the telemetry (Orange Mushrooms on 2 in view, no HP bar, for 4 s of throwing). The search
  alone picks them there (286-392 units off when the attacks started, 60 above): it's the wall that stops
  those stars;
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

from ..localization.mapdata import MobBody, Platform, Segment
from ..localization.monsters import MONSTER_HALF, MONSTER_HEIGHT, Monster
from ..motion import physics

# How far a star reaches: 200 for her claw (Lucky Seven's text: "within 200 range in front", skill data)
# and 200 more with Keen Eyes maxed (its `range`), to the near side of a monster's body. On her level,
# stars hit slimes 408, 410 and 452 units off by their middle at the look before the throw (the 452 one
# walking ~25 closer by the pick) and missed at 421, 560 and 634 (lab/range_probe.jsonl); in the live
# tours' telemetry none was hit at 445 or 472.
STAR_RANGE = 400
STAR_REACH = STAR_RANGE + max(MONSTER_HALF.values())  # 434: the farthest off (by its middle) a star picks
                     # the widest kind, a pig: what can be in a throw's path (kill matching)
SHOT_RANGE = 410     # aim inside this: the target keeps walking during the ~0.5 s wind-up and the flight
MIN_SHOT = 90        # closer than this (middle to middle) the claw swipes any monster on her level (CLAW_FAR);
                     # a bigger one farther off too (claw_edge)
# How far below/above her a target can be and still be hit: 60 both ways measured (hits). Farther
# isn't measured yet; `python -m gamelab hunt shots` lists what the shot log has seen.
SHOT_DOWN = 60
SHOT_UP = 60
# The star's target search from the ground (see above): its bands, and their windows above her feet.
SEARCH_START = 65    # the first band starts this far ahead of her (closer, on the ground the claw swipes)
SEARCH_STEP = 20     # each band is this wide ...
SEARCH_GROW = 5      # ... and its window this much higher and lower than the one before (1 in 4)
SEARCH_FLOOR0 = 12   # the first band's window from this far above her feet (meowdb's, and the logs' fit) ...
SEARCH_TOP0 = 21     # ... to this far. meowdb draws 44 (12 to 44: 28 +- 16); fitted to the shot log and the
                     # telemetry the top is 21 (95% about 16-28; 44 is 6.2 log-likelihood worse, and each
                     # prefers the lower top on its own). Too low a top only skips chances above her (a pig
                     # 60 above is picked from 191 units, from 111 with 44), never throws at nothing.
                     # Unmeasured in game: throws from 13 at 12 (60 above her), 110-200 units off, settle it
                     # (`python -m gamelab probe run top-60`, then `probe score`: behavior.probe).
BANDS = math.ceil((STAR_RANGE - SEARCH_START) / SEARCH_STEP)  # 17: the last one 15 wide
LEVEL = 15           # within this of her height, a target counts as on her level
# Between her level and 60 below her the search is unchecked, and the only throws logged there go against it:
# from 0 at 1 (42 below), the shot log of 2026-10-18 has 3 misses in 4 at pigs 124-125 off and a miss at an
# Orange Mushroom 91 off, where the search picks both from 90 (what else was in front wasn't logged then). Its
# fit rests on the throws 60 off her level. So below her, between the two, her decisions keep at least to the
# old ramp (ramp_dx: MIN_SHOT on her level to RAMP_FAR 60 below, a pig 42 below from 126 instead of 90) until
# a probe settles it (`python -m gamelab probe run below-42`, then `probe score`). False: the search alone.
# Above her nothing goes against the search. On this map only 0 -> 1 is that far apart.
RAMP_BELOW = True
RAMP_FAR = 150       # the old ramp's end, 60 below her (fitted to the shot log before the search)
STAR_SPEED = 690     # units/s
EDGE_MARGIN = 25     # where she stands to shoot: not closer than this to her platform's end
# Where a look searches for monsters: from this far above her feet to this far below (bodies stick up
# above their feet, so the band reaches higher than she can shoot).
LOOK_UP = SHOT_UP + 80
LOOK_DOWN = SHOT_DOWN + 15
# The area an attack from a jump picks its target in (see above). Its floor is fitted to the live throws
# of 2026-10-19 (jump-throws missed snails; throws from the ground hit pigs, slimes and Orange Mushrooms
# 60 below her): 18-21 below her feet. Of its top, only that monsters 60 above her are in it is known.
AREA_DOWN = 19
AREA_UP = 65
AIR_MARGIN = 6       # how far into the area a body must reach for an attack from a jump to pick it surely:
                     # about one frame of her climb when a jump throw's attack starts
# The claw's swings (game data: the claw's afterImage, swordOL: swingO1-O3 and stabO1) reach from 17 below
# to 62 above her feet, and 73-102 ahead of her (taken as CLAW_FAR, as the simulator's swing has it). A
# monster on her level whose body reaches into them is swiped: from up to CLAW_FAR plus half its width off
# by its middle (a pig 124, a snail 109), so a throw from the ground at it is a star only from farther off
# (claw_edge). Deciding by the middle (a star from 100 units, whatever the kind), 23% of the simulator's
# swipes came 90-160 units off, and swipes were 40% of the hunt loop's hits, each a fifth of a star's
# damage; by the body they fell by a third (20 seeds). Unmeasured in game: how far off a pig is still
# swiped (by the swings' boxes, 107-136): `python -m gamelab probe run claw-edge`, then `probe score` (a swipe
# takes 7-10% of a pig's or a slime's bar, a star 23-61%).
CLAW_FAR = MIN_SHOT
CLAW_DOWN = 17
CLAW_UP = 62
CLAW_MARGIN = 12     # kept past the swing's reach, for a monster walking at her while the attack starts (a pig,
                     # 125 units/s by the game data, walks 12 in 0.1 s). In the simulator 6 cut the swipes as
                     # much, but gained less in the hunt loop (+1.8% EXP a minute against +5.6%; +1.6% against
                     # +3.2% with the game data's walking speeds), and about the same on the tour.
# What stops a star (see above). The level line runs SHOT_H above her feet: anything from 1 to 59 fits the
# logs (on this map it matters only for steps lower than 60, and there are none). 28 is a stand-in, the middle
# of meowdb's first window (12-44), which the logs' fit doesn't keep (12-21, its middle 16.5); on a map with
# steps 17-27 tall the two would differ. The step-up probe's jump throws and jump backs (the line 26 and 44
# higher: `python -m gamelab probe run step-up --style jump_throw`, `jump_back`) bracket it: from 4 at 2, a
# 60 step, stopped and through put it at 16-34.
SHOT_H = 28
# Whether her decisions keep to it. Off, they throw at monsters behind such walls as before, so an in-game
# probe can measure it (the shot log has the rule's verdict either way; the simulator's world keeps to the
# rule regardless). The probes that measure it (behavior.probe: step-up, step-down, hanging-wall) switch it
# off for their run.
WALLS_BLOCK = True
PIG = (MONSTER_HEIGHT["pig"], MONSTER_HALF["pig"])  # the body of a kind no one knows (body_of)


def in_band(dy: float) -> bool:
    """Whether a target dy below (+) or above (-) her is level enough with her for a star to hit."""
    return -SHOT_UP <= dy <= SHOT_DOWN


# --- the star's target search, from the ground ---------------------------------------------------------

def window(k: int, top0: float | None = None, floor0: float | None = None) -> tuple[float, float]:
    """Band k's window: from how far above her feet (its floor) to how far (its top). `top0`, `floor0`: the
    first window's edges, if not SEARCH_TOP0 and SEARCH_FLOOR0 (the probes' scorer tries others)."""
    top0 = SEARCH_TOP0 if top0 is None else top0
    floor0 = SEARCH_FLOOR0 if floor0 is None else floor0
    return floor0 - SEARCH_GROW * k, top0 + SEARCH_GROW * k


def band_of(ahead: float) -> int | None:
    """The band `ahead` units in front of her is in (None: short of SEARCH_START, or past STAR_RANGE)."""
    if not SEARCH_START <= ahead <= STAR_RANGE:
        return None
    return min(BANDS - 1, int((ahead - SEARCH_START) // SEARCH_STEP))


def search_band(near: float, far: float, bottom: float, top: float, top0: float | None = None,
                floor0: float | None = None) -> int | None:
    """The first band a body overlaps, from `near` to `far` ahead of her and from `bottom` to `top` above
    her feet (body_box). None if the search doesn't pick it. `top0`, `floor0`: as in window."""
    if far < SEARCH_START or near > STAR_RANGE or far < near:
        return None
    for k in range(band_of(max(near, SEARCH_START)), band_of(min(far, STAR_RANGE)) + 1):
        floor, roof = window(k, top0, floor0)
        if bottom < roof and top > floor:
            return k
    return None


def body_box(ahead: float, dy: float, top: float, half: float, lift: float = 0.0) -> tuple:
    """A monster's body as the search sees it, (near, far, bottom, top): its middle `ahead` in front of
    her, its feet `dy` below hers (+ below), `top` tall and `half` wide either side, raised `lift` off its
    feet (a slime in mid-hop: 10-17)."""
    return ahead - half, ahead + half, lift - dy, top - dy


def picks(ahead: float, dy: float, top: float, half: float, lift: float = 0.0) -> bool:
    """Whether the star's search from the ground picks this body (body_box), terrain aside."""
    return search_band(*body_box(ahead, dy, top, half, lift)) is not None


def body_of(kind: str | None, bodies: dict[str, MobBody] | None = None) -> tuple[float, float]:
    """How tall a monster of `kind` is walking, and half its width: what the search goes by. Half its
    width from `bodies` (the map's game data, localization.monsters.bodies) where it has the kind, else
    from MONSTER_HALF; its height from MONSTER_HEIGHT (the game data keeps only how tall a kind is at
    least, as a pig stands: 38 against 46 walking, which 60 below her is 171 units sideways against 151),
    or for a kind the table lacks, that least height. A kind neither knows is taken as a pig."""
    b = (bodies or {}).get(kind)
    top = MONSTER_HEIGHT.get(kind, b.top if b is not None else PIG[0])
    half = b.half if b is not None else MONSTER_HALF.get(kind, PIG[1])
    return top, half


def min_dx(dy: float, top: float = PIG[0], half: float = PIG[1], lift: float = 0.0) -> float:
    """The closest sideways (middle to middle) the search picks a monster dy below (+) or above (-) her
    feet, `top` tall and `half` wide either side, raised `lift` off its feet (default: a walking pig): its
    far side has to reach the first band whose window takes in its body. Infinite if none does. On her
    level, 65 less its half width; the claw swipes that close (shot_dx)."""
    t, b = top - dy, lift - dy  # its top and its bottom, above her feet
    k = 0
    if t <= SEARCH_FLOOR0:      # under the first window: wait for the floor to come down to it
        k = max(k, math.floor((SEARCH_FLOOR0 - t) / SEARCH_GROW) + 1)
    if b >= SEARCH_TOP0:        # over it: wait for the top to come up
        k = max(k, math.floor((b - SEARCH_TOP0) / SEARCH_GROW) + 1)
    return math.inf if k >= BANDS else SEARCH_START + SEARCH_STEP * k - half


def max_dx(half: float = PIG[1]) -> float:
    """The farthest (by its middle) the search picks a monster `half` wide either side: its near side
    within STAR_RANGE."""
    return STAR_RANGE + half


def ramp_dx(dy: float) -> float:
    """The old ramp's closest sideways for a target dy below her (+), between her level and SHOT_DOWN: from
    MIN_SHOT at LEVEL to RAMP_FAR at SHOT_DOWN (RAMP_BELOW). 0 anywhere else."""
    if not LEVEL < dy < SHOT_DOWN:
        return 0.0
    return MIN_SHOT + (RAMP_FAR - MIN_SHOT) * (dy - LEVEL) / (SHOT_DOWN - LEVEL)


def shot_dx(dy: float, kind: str | None = None) -> float:
    """The closest sideways a throw from the ground at a monster of `kind` dy below (+) her throws a star
    that picks it: min_dx by its body (body_of), and never closer than MIN_SHOT (closer, the claw swipes;
    on her own platform her decisions keep a bigger monster out to claw_edge), nor than the old ramp between
    her level and 60 below her (ramp_dx, while RAMP_BELOW)."""
    return max(MIN_SHOT, min_dx(dy, *body_of(kind)), ramp_dx(dy) if RAMP_BELOW else 0.0)


def can_shoot(dx: float, dy: float, reach: float = SHOT_RANGE, kind: str | None = None) -> bool:
    """Whether a throw from where she stands hits a monster of `kind` (default: a pig) dx along (either
    way) and dy below (+) with a star: in band, from shot_dx sideways out to `reach`, and no farther than
    the search picks it (max_dx). Terrain aside (can_hit: with it)."""
    return in_band(dy) and shot_dx(dy, kind) <= abs(dx) <= min(reach, max_dx(body_of(kind)[1]))


def claw_takes(ahead: float, dy: float, top: float, half: float) -> bool:
    """Whether the claw's swing takes a monster `ahead` in front of her (its middle), its feet dy below
    hers, `top` tall and `half` wide either side: on her level (LEVEL), its body reaching into the swing
    (CLAW_FAR ahead of her, CLAW_DOWN below to CLAW_UP above her feet). Off her level it isn't counted on
    (nor swiped in the simulator), though the swings' boxes reach 17 below her feet: in the telemetry of
    2026-10-20 an Orange Mushroom 60 below her and 91 off, its body in them, was passed over by 4 throws, as
    a claw on her level only has it, but from 2, with the step wall down to 4 in the swing too, so it doesn't
    settle it; the decisions never planned on it either. Unmeasured in game: `python -m gamelab probe run
    claw-down` (from 3 at 4) settles it, and `claw-edge` how far off the claw swipes on her level."""
    return abs(dy) <= LEVEL and reaches_into(ahead, dy, top, half, CLAW_FAR, CLAW_DOWN, CLAW_UP)


def claw_edge(half: float) -> float:
    """How far off (middle to middle) a monster on her level `half` wide either side has to be for a throw
    from the ground not to be a claw swipe: its body clear of the swing (CLAW_FAR), CLAW_MARGIN to spare. A
    pig 136, an Orange Mushroom 134, a slime 130, a snail 121 (half widths 34, 32, 28, 19)."""
    return CLAW_FAR + half + CLAW_MARGIN


def star_sure(half: float) -> float:
    """How far off she steps back to for a star from the ground at a monster on her level `half` wide either
    side: 3 past claw_edge, so a step that comes up a little short still clears it."""
    return claw_edge(half) + 3


STAR_SURE = star_sure(MONSTER_HALF["snail"])  # 124: a snail's (it was 125, from the widest swing, 102, and
                                              # half a snail, 19)


def ground_pick(ahead: float, dy: float, top: float, half: float) -> tuple | None:
    """Where a monster (as in claw_takes) comes in the order an attack from the ground takes them, the
    least first, or None if it can't take it: the claw's swing first (the nearest it reaches), then the
    star's search (the first band, the nearest in it). Terrain aside."""
    if claw_takes(ahead, dy, top, half):
        return 0, abs(ahead)
    k = search_band(*body_box(ahead, dy, top, half))
    return None if k is None else (1 + k, abs(ahead))


# --- what stops a star ---------------------------------------------------------------------------------

def _crosses(w: Segment, line: float) -> bool:
    """Whether wall `w` crosses the level line at map y `line` (a wall split into footholds, end to end,
    crosses it once: each foothold from its top, inclusive, to its bottom)."""
    return min(w.y1, w.y2) <= line < max(w.y1, w.y2)


def _inside(x: float, p: Platform) -> float:
    """Map x, 1 unit inside platform `p`'s ends."""
    return min(max(x, p.left + 1), p.right - 1)


def wall_between(platform: Platform | None, x: float, y: float, tx: float, rise: float = 0.0,
                 target: Platform | None = None, shot_h: float | None = None) -> Segment | None:
    """The wall that stops a star from her (map x on `platform`, her feet at y, `rise` units up in a jump)
    before map x `tx`: of the walls of her own foothold group (Platform.walls), the nearest standing
    between them across the level line SHOT_H above her feet; None if the star gets there. Her x is taken
    1 unit inside her platform's ends, and `tx` inside the target's platform (`target`, if known): a
    monster is never past its platform's end, and one at the foot of a wall of her group that rises from
    its own platform's end (the basin's sides, from its floor) isn't behind it. `shot_h`: the line's
    height, if not SHOT_H (the probes' scorer tries others)."""
    if platform is None:
        return None
    line = y - rise - (SHOT_H if shot_h is None else shot_h)
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
    """Whether a star from map (x, y) on `platform` hits monster `m`: in reach by its kind (can_shoot), and
    no wall of her foothold group in the way (wall_to; `rise`: her height in a jump, `target`: m's
    platform)."""
    return can_shoot(m.x - x, m.y - y, reach, m.kind) and wall_to(platform, x, y, m, rise, target) is None


def clear_span(platform: Platform | None, x: float, y: float, rise: float = 0.0) -> tuple[float, float]:
    """The map x a star from her (as in wall_in_way) gets to either way before a wall of her foothold group
    stops it: from the nearest such wall on her left to the nearest on her right (infinite where there's
    none, and either way with WALLS_BLOCK off)."""
    if not WALLS_BLOCK or platform is None:
        return -math.inf, math.inf
    line, a = y - rise - SHOT_H, _inside(x, platform)
    xs = [w.x1 for w in platform.walls if _crosses(w, line)]
    return max((v for v in xs if v <= a), default=-math.inf), min((v for v in xs if v >= a), default=math.inf)


# --- what an attack from a jump picks --------------------------------------------------------------------

def reaches_into(ahead: float, dy: float, top: float, half: float, far: float, down: float, up: float) -> bool:
    """Whether a monster's body reaches into a box in front of her: from her (0) to `far` ahead, `down`
    below to `up` above her feet. The monster's middle is `ahead` units in front of her, its feet `dy`
    below hers (+ below), and its body `top` tall and `half` its width either side."""
    return -half <= ahead <= far + half and top - dy >= -down and -dy < up


def in_area(ahead: float, dy: float, top: float, rise: float = 0.0, half: float = 0.0) -> bool:
    """Whether an attack's star can pick a monster `ahead` units in front of her (its middle), its feet `dy`
    below hers as she stands (+ below), its body `top` tall (and `half` wide either side), when the attack
    starts with her `rise` units up in a jump: its body reaches into the area then (AREA_DOWN to AREA_UP
    about her feet, its near side within STAR_RANGE ahead). From the ground (`rise` 0), the star's search
    picks it (picks)."""
    if rise == 0:
        return picks(ahead, dy, top, half)
    return reaches_into(ahead, dy + rise, top, half, STAR_RANGE, AREA_DOWN, AREA_UP)


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
