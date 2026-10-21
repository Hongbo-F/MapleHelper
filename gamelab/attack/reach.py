"""Where her stars reach: how far, how far below and above her, how close is too close, what an attack
from the ground or from a jump picks, and what stops a star.

A regular attack throws one star straight ahead. Measured in game (Thicket Around the Beach II):
- in open air the star flew 400 map units (560 screen px) before vanishing;
- it hit a pig 388 units away on a platform 60 units below her (platform 14 -> 17), and killed a pig
  on a platform 60 units above her (platform 3 -> 1): stars aren't stopped by platforms, and the
  target can be lower or higher than she is, within a band;
- it leaves her ~530 ms after the key and flies ~690 units/s;
- right next to a monster, the claw swipes instead of throwing (weak, and the monster can touch her).

What an attack hits is picked when it starts, about 65 ms after its key, not when the star flies (the
star is only a picture), by the monsters' bodies (their boxes in the game data, localization.monsters).
From the ground that's the client's own routine (the game decides it in the client, from data it has:
ranges, bodies, the claw's boxes), as a port of the v83 client's decompiled shot code and meowdb's page
on this game give it alike (gamelab/README.md, Attack, has the sources). The claw comes first: each
attack takes one of its two stab stances at random (CLAW_STANCES), and the nearest monster whose body
reaches into that stance's box, cut to CLAW_FAR ahead of her, is swiped instead of shot, on her level or
not (`claw_takes`, `claw_chance`); so her decisions count on a star at one only from farther off, by its
size (`claw_edge`: a pig from 111 units, a snail from 96). Else the star's target search (`search_band`):
bands SEARCH_STEP wide from SEARCH_START ahead of her out to STAR_RANGE, band k's window reaching
(SEARCH_START + SEARCH_STEP k) // SEARCH_SPREAD above and below SHOT_H (`window`: 12 - 5k to 44 + 5k above
her feet), so it opens one unit up and one down for every four ahead; the star goes to a monster whose
body overlaps the first band any body does. Of those the client takes the one that entered the map
first, which can't be seen, so any of them can take it (`take_chances`). Off her level a star so needs
room sideways, the more the farther off her level and the smaller the monster (`min_dx`, by its body:
`body_of`): 60 below her an Orange Mushroom is picked from 113 units, a pig from 151, a slime from 177, a
snail from 246; 60 above, a pig from 111, an Orange Mushroom from 113, a slime from 117, a snail from 126.
The shot log and the live tours' telemetry agree with the search's shape and fit its floor; they lean
to a lower top (SEARCH_TOP0), suggestively, not decisively, which `probe run top-60` settles. From a jump
it doesn't fit (40 ms after the jump, stars killed snails 36-66 units away, closer than a search starting
65 ahead picks), so an attack from a jump keeps to the area that does: a box fixed to her feet, as high as
she is then (`in_area`), from AREA_DOWN below them to AREA_UP above, STAR_RANGE ahead. A jump throw's attack
keyed 100 ms after the jump started 42-49 units up, its area over a snail (24-26 tall): in 45 jump-throws
at snails 3-77 units away on 2026-10-19, at most one hurt the snail, and the monsters on the platform 60
above took the stars. RustMS's port of the v83 client has no case for the air: the claw and the search as
from the ground, from as high as she is, which picks no snail 35-85 off from 21-26 up (`probe run air` weighs
the two; the simulator's world has either, sim.world.AIR_PICK).

What stops a star is checked on the attack's pick alone, as the client does it (RustMS Combat.cpp:1289-1307;
ohmi69's reading of this game's client and meowdb's page on it say the same): the claw and the star's search
pick by the bodies, whatever the terrain, and then one line, from her to the monster picked, is checked.
Stopped, the attack hits nothing: the star ends where it's stopped, and no other monster takes it. So a
monster the attack takes first whose line is stopped leaves those behind it out of a star's reach
(`shielded`). A claw swipe isn't checked at all (the client's whack is a hit up close, with no line). What
stops the line (`wall_between`) is the walls of the foothold group of the platform she last stood on
(mapdata.Segment.group: a floating block's floor and the walls hanging from its ends, or platforms joined by
a step), and only going in at their outer face: a wall drawn downward (y1 < y2) faces right, one drawn
upward faces left (RustMS ShotPath.h:382-445, `blocked`: a line stops only going from a foothold's open side
to its closed side; on this map every block's walls face out from it, and the step between 2 and 4 and the
basin's sides face the floor below them, so it changes nothing here). The walls of other groups never stop
one, such as the side walls of the block the target stands on. The line checked (`LINE`) is the v83
client's, from SHOT_H above her feet to where the star hits its body (`hit_point`), which a floor of any
group stops going down through it (RustMS): so her own platform stops a star at a monster below her unless
she stands near enough its end. Watching her hunt on 2026-10-21, the user saw just that, down a step and from
a higher platform onto a lower one. The other line, a level one SHOT_H above her feet (higher in a jump) to
the monster's x, which no floor can stop (ohmi69, for a basic shot), was the default until then: the live
logs of 2026-10-18 to 10-20 fit both alike, as every throw down in them was made from near an end (299
decided throws, the same verdict on every one). The game data has no foothold flag for it (306 map images: a
foothold has its ends, its neighbours and `piece`, and in 51 maps `forbidFallDown`). In those logs:
- from platform 4, no star reached a monster on 2, 60 above her behind the step wall joining the two
  (group (3, 63)), in 3 sessions: 0 of 3 in the shot log, every kill of 40 throws on 4 (sweep log 9),
  0 of 5 in the telemetry (Orange Mushrooms on 2 in view, no HP bar, for 4 s of throwing). The search
  alone picks them there (286-392 units off when the attacks started, 60 above): it's the wall that stops
  those stars (and the user, watching on 2026-10-21: the step up for sure blocks the star);
- in the same logs, stars 60 up past the side wall of the block above (another group, from 25 below
  to 60 above her feet) hit: 19 -> 18 13 of 13 decided, 3 -> 1 15 of 19; and stars down past her own
  block's hanging wall (14 -> 17, 25 deep) hit 81%, all thrown from near its end, where the v83 line clears
  her floor too;
- "every wall blocks" contradicts 19 -> 18, and a line to the target's feet can't fit 4 -> 2 and 14 -> 17
  together.
The client's collisions in the air also skip walls of other groups (a reconstruction of its physics),
likely the same test. On this map her own group's walls change one pair, 4 -> 2; her own floor stops stars
down at monsters close past her platform's end (from 2 at x 50 at pigs on 4 under 254 off, from 14 at x 690
at pigs on 17 up to 238 off), and from near its end none.

All in map units; dy is the target's feet minus hers (+ = below her).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

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
# How far below/above her a target can be and still be hit: 60 both ways measured (hits). Farther
# isn't measured yet; `python -m gamelab hunt shots` lists what the shot log has seen.
SHOT_DOWN = 60
SHOT_UP = 60
# A shot's row: how far above her feet the star's search is centred, and the line that terrain stops (below)
# runs. It's the client's own constant, in every source that has one: RustMS's port of the v83 client
# (ShotPath.h:20, HEIGHT = 28), meowdb's targeting code for this game (aimHeight 28), ohmi69's reading of this
# game's client (reference_y -28) and Harepacker's of the v95 one (DefaultShootAttackPointYOffset -28).
SHOT_H = 28
# The star's target search from the ground (see above): its bands, and their windows about SHOT_H. The
# client's (RustMS ShotPath.h:26-29 and :311-331, search_steps; meowdb {aimHeight 28, divisor 4, step 20,
# starts: ranged 65} with halfHeight floor(start / 4)): band k's window reaches spread(k) = (65 + 20 k) // 4
# = 16 + 5k above and below SHOT_H, so 12 - 5k to 44 + 5k above her feet.
SEARCH_START = 65    # the first band starts this far ahead of her (where the claw's swipe ends: CLAW_FAR) ...
SEARCH_STEP = 20     # ... each band is this wide ...
SEARCH_SPREAD = 4    # ... and its window reaches one unit up and one down per this many its near edge is ahead
SEARCH_FLOOR0 = SHOT_H - SEARCH_START // SEARCH_SPREAD  # 12: the first band's window from this far above her
                     # feet (the logs fit it: best of the grid, about 4-16 within 2 log-likelihood) ...
SEARCH_TOP0 = SHOT_H + SEARCH_START // SEARCH_SPREAD    # 44: ... to this far. RustMS's own test of it has
                     # the first band 156-188 for her feet at 200, and meowdb's Lucky Seven figure 124 up and
                     # 68 down at its last band (385-400), drawn since 2026-08-22, before RustMS's code. The
                     # logs lean to a lower top: fitted to the shot log and the telemetry it came out 21 (a pig
                     # 60 above her picked from 191 units, not 111), but cleaned fairly the lean is 2-5
                     # log-likelihood. As independent throws that would rule 44 out (the probes' scorer's 95%
                     # bound is 1.92), but its 10-14 throws (60 above her, 150-191 off) are four episodes (3 -> 1
                     # twice, 13 -> 12, 22 -> 21), whose throws go together: all four leaning the same way is 1 in
                     # 16 by chance (p 0.06, one-sided), and in the shot log, without any one of them the lean is
                     # 1-3.5. Suggestive, not decisive; tops 17-28 fit alike. Too high a top has her throw at
                     # monsters above her that the star doesn't take; too low, skip them, and count on stars at
                     # monsters past them that they take.
                     # Unmeasured in game: throws from 13 at 12 (60 above her), 110-200 units off, settle it
                     # (`python -m gamelab probe run top-60`, then `probe score`); this is the one number to
                     # change (to 21) if they find the lower top.
BANDS = math.ceil((STAR_RANGE - SEARCH_START) / SEARCH_STEP)  # 17: the last one 15 wide
LEVEL = 15           # within this of her height, a target counts as on her level
# Between her level and 60 below her the search is unchecked, and the only throws logged there go against it:
# from 0 at 1 (42 below), the shot log of 2026-10-18 has 3 misses in 4 at pigs 124-125 off and a miss at an
# Orange Mushroom 91 off, where the search picks a pig from 71 and an Orange Mushroom from 33 (what else was in
# front wasn't logged then). Its
# fit rests on the throws 60 off her level. So below her, between the two, her decisions keep at least to the
# old ramp (ramp_dx: RAMP_NEAR on her level to RAMP_FAR 60 below, a pig 42 below from 126 instead of 71) until
# a probe settles it (`python -m gamelab probe run below-42`, then `probe score`). False: the search alone.
# Above her nothing goes against the search. On this map only 0 -> 1 is that far apart.
RAMP_BELOW = True
RAMP_NEAR = 90       # the old ramp's start, on her level (the claw's reach as it was taken then) ...
RAMP_FAR = 150       # ... and its end, 60 below her (fitted to the shot log before the search)
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
# The claw up close (see above): a regular attack from the ground first looks for a monster right in front of
# her and swipes it with the claw instead of throwing (9-11 damage against a star's 30-70). In the client
# (RustMS Combat.cpp:1665-1676, find_whack) the attack picks one of the claw's close stances at random, stabO1
# or stabO2 (a claw's info/attack is 7: AttackStances.h:49, picked by CharLook.cpp:641-655; ohmi69 gives a claw
# only the stabs up close, lib_skill-range-viz.js:77 and tabs_formulas.js:78-95, and meowdb's damage guide has a
# claw always stab there), takes that stance's box in the game data (CLAW_BOXES: the claw's afterImage,
# Character/Afterimage/swordOL.img/0, lt and rb as she faces left), cuts its front to CLAW_FAR ahead of her
# (PointBlank.h:16-26, MAX_REACH = 65), and swipes the nearest monster (by distance, feet to feet:
# Combat.cpp:1813-1842) whose body reaches into it, on her level or not: no gate for it. A throw plays a swing
# (swingO1-O3, the claw's ranged list), but the swings' boxes aren't what the claw tests. So the swipe reaches
# from 12-20 to 65 ahead of her and from 15 to 30 above her feet: on her level a monster out to 65 plus half its
# width off by its middle (a pig 99, a snail 84); off it, none whose feet are 30 or more above hers, nor one whose
# top is under 15 above them (walking, a pig more than 31 below her or an Orange Mushroom more than 40; flinching,
# taller, more than 39 and 50). Before, the swings and stabO1 together (73-102 ahead, 17 below to 62 above her
# feet, taken as 90 ahead on her level only) had a pig swiped out to 124. Unmeasured in game: how far off a pig
# is still swiped (`python -m gamelab probe run claw-edge`, then `probe score`: a swipe takes 7-10% of a pig's or
# a slime's bar, a star 23-61%). Deciding by the body, not by the middle alone, cut the simulator's swipes by a
# third (20 seeds, with the old swing).
CLAW_FAR = 65
CLAW_BOXES = {"stabO1": ((-84, -30), (-20, -15)), "stabO2": ((-83, -30), (-12, -16))}
CLAW_STANCES = tuple(CLAW_BOXES)  # one of these per attack, each as likely


@dataclass(frozen=True)
class Claw:
    """How the claw takes a monster up close: each attack takes one of its stances, each as likely, and the
    stance's box (`boxes`: (stance, (lt, rb)), as she faces left, lt.x its front) with its front cut to `far`
    ahead of her takes the nearest monster whose body reaches into it; with `level`, only one within that of
    her height (the client has no such gate). CLAW is the client's; the probes' scorer (behavior.probe) keeps
    the models it tells from it, and the simulator's world goes by whichever CLAW is."""
    boxes: tuple
    far: float
    level: float | None = None

    @property
    def stances(self) -> tuple[str, ...]:
        return tuple(s for s, _ in self.boxes)


CLAW = Claw(tuple(CLAW_BOXES.items()), CLAW_FAR)
CLAW_MARGIN = 12     # kept past the claw's reach, for a monster walking at her while the attack starts (a pig,
                     # 125 units/s by the game data, walks 12 in 0.1 s). In the simulator 6 cut the swipes as
                     # much, but gained less in the hunt loop (+1.8% EXP a minute against +5.6%; +1.6% against
                     # +3.2% with the game data's walking speeds), and about the same on the tour (with the old
                     # swing).
# What stops a star (see above): which line the client checks from her to the monster its attack picked.
# 'v83': the v83 client's, from SHOT_H above her feet at her own x (not the shot's start, 65 ahead) to where the
# star hits the monster's body (hit_point: its top if it's below the row, its feet if above), which a floor of any
# group stops going down through it as well as her group's walls (RustMS ShotPath.h:382-445,
# Combat.cpp:1289-1307). meowdb's page on this game says so too (shooting down through the platform she stands on
# is blocked, up through one isn't), citing no test. 'level': a level line SHOT_H above her feet (raised in a
# jump), from her x to the monster's, which only walls can cross: ohmi69's reading of this game's client checks a
# basic shot straight across from that row (its one source, and one whose basic shot's target, a 1-unit line 28
# up, the logs refute). The logs fit both: every throw down in them was made from near an end, where the two agree
# (299 decided, the same verdict on all), so 'level' was the default until 2026-10-21, when the user, watching her
# hunt, saw her own platform stop her stars at monsters below her, both down a step (as from 2 at 4) and from a
# higher platform at a lower one (as from 14 at 17). So 'v83' is the default. The two differ only below her: by
# the v83 line a star needs room past her platform's end, |dx| > e (28 + t) / 28, e her distance to the end and t
# how far below her feet the monster's top is (a walking pig 60 below, t 14: 1.5 e): from 2 at x 50 pigs on 4
# under 254 off and slimes under 284 are out of reach, from 14 at x 690 pigs on 17 up to 238 off, and from 25 in
# from either end every monster a star picks 60 below her. Unmeasured in game: where past her platform's end the
# star gets through, which the user's report doesn't say (`python -m gamelab probe run step-down`, from 2 at x 50
# at pigs on 4 200-310 off and slimes 230-340, and `hanging-wall`, from 14 at x 706 at pigs on 17 180-250 and
# slimes 190-290: by the v83 line none under 254, 284, 215 and 241 off is hit, and beyond them as many as
# anywhere; then `probe score`, which weighs the two lines on them, and shows the hits by how far off). SHOT_H is
# the client's row, 28; the logs fit any height from 1 to 59 for the level line (on this map it matters only for
# steps lower than 60, and there are none), and the step-up probe's jump throws and jump backs (the line 26 and 44
# higher: `probe run step-up --style jump_throw`, `jump_back`) bracket it: from 4 at 2, a 60 step, stopped and
# through put it at 16-34. Walls of group 0 stop a star from any group in RustMS's port alone (ShotPath.h:398; the
# v40 client's collisions have no such exception): left out, and on this map group 0 is the basin, where it would
# change nothing. Ceilings (x2 < x1), which the walls' rule covers there too, aren't kept (none on this map).
LINE = "v83"
LINES = ("v83", "level")
# Whether her decisions keep to what stops a star. Off, they throw at monsters behind such walls as before, so an
# in-game probe can measure it (the shot log has the rule's verdict either way; the simulator's world keeps to the
# rule regardless). The probes that measure it (behavior.probe: step-up, step-down, hanging-wall) switch it off
# for their run.
WALLS_BLOCK = True
PIG = (MONSTER_HEIGHT["pig"], MONSTER_HALF["pig"])  # the body of a kind no one knows (body_of)


def in_band(dy: float) -> bool:
    """Whether a target dy below (+) or above (-) her is level enough with her for a star to hit."""
    return -SHOT_UP <= dy <= SHOT_DOWN


# --- the star's target search, from the ground ---------------------------------------------------------

def spread(k: int) -> int:
    """How far band k's window reaches above and below SHOT_H, as the client has it: the band's near edge's
    distance ahead of her over SEARCH_SPREAD, in whole units (16 + 5k)."""
    return (SEARCH_START + SEARCH_STEP * k) // SEARCH_SPREAD


def window(k: int, top0: float | None = None, floor0: float | None = None) -> tuple[float, float]:
    """Band k's window: from how far above her feet (its floor) to how far (its top), SHOT_H - spread(k) to
    SHOT_H + spread(k) (12 - 5k to 44 + 5k). `top0`, `floor0`: the first window's edges, if not SEARCH_TOP0
    and SEARCH_FLOOR0 (the probes' scorer tries others); the later windows open from them as the client's
    do."""
    top0 = SEARCH_TOP0 if top0 is None else top0
    floor0 = SEARCH_FLOOR0 if floor0 is None else floor0
    grow = spread(k) - spread(0)
    return floor0 - grow, top0 + grow


def band_of(ahead: float) -> int | None:
    """The band `ahead` units in front of her is in (None: short of SEARCH_START, or past STAR_RANGE)."""
    if not SEARCH_START <= ahead <= STAR_RANGE:
        return None
    return min(BANDS - 1, int((ahead - SEARCH_START) // SEARCH_STEP))


def search_band(near: float, far: float, bottom: float, top: float, top0: float | None = None,
                floor0: float | None = None) -> int | None:
    """The first band a body overlaps, from `near` to `far` ahead of her and from `bottom` to `top` above
    her feet (body_box). None if the search doesn't pick it. Overlap is strict, sideways as well as up and
    down, as the client's (RustMS ShotPath.h:335-339, in_step, IntersectRect's): a body whose far side is
    right at a band's near edge isn't in that band, nor one whose near side is at STAR_RANGE. `top0`,
    `floor0`: as in window."""
    if far <= SEARCH_START or near >= STAR_RANGE or far <= near:
        return None
    first = max(0, int((near - SEARCH_START) // SEARCH_STEP))
    last = min(BANDS - 1, math.ceil((far - SEARCH_START) / SEARCH_STEP) - 1)
    for k in range(first, last + 1):
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
    far side has to reach the first band whose window takes in its body (past that band's near edge: the
    search picks it from just farther off, search_band). Infinite if none does. On her level, 65 less its
    half width; the claw swipes that close (shot_dx)."""
    t, b = top - dy, lift - dy  # its top and its bottom, above her feet
    for k in range(BANDS):
        floor, roof = window(k)
        if b < roof and t > floor:
            return SEARCH_START + SEARCH_STEP * k - half
    return math.inf


def max_dx(half: float = PIG[1]) -> float:
    """The farthest (by its middle) the search picks a monster `half` wide either side: its near side
    within STAR_RANGE (short of it: search_band)."""
    return STAR_RANGE + half


def ramp_dx(dy: float) -> float:
    """The old ramp's closest sideways for a target dy below her (+), between her level and SHOT_DOWN: from
    RAMP_NEAR at LEVEL to RAMP_FAR at SHOT_DOWN (RAMP_BELOW). 0 anywhere else."""
    if not LEVEL < dy < SHOT_DOWN:
        return 0.0
    return RAMP_NEAR + (RAMP_FAR - RAMP_NEAR) * (dy - LEVEL) / (SHOT_DOWN - LEVEL)


def shot_dx(dy: float, kind: str | None = None) -> float:
    """The closest sideways a throw from the ground at a monster of `kind` dy below (+) her throws a star
    that picks it: min_dx by its body (body_of), and never where the claw can take it instead (claw_dx: on
    her level 65 plus half its width; on her own platform her decisions keep a monster out to claw_edge),
    nor closer than the old ramp between her level and 60 below her (ramp_dx, while RAMP_BELOW)."""
    top, half = body_of(kind)
    return max(min_dx(dy, top, half), claw_dx(dy, top, half), ramp_dx(dy) if RAMP_BELOW else 0.0)


def can_shoot(dx: float, dy: float, reach: float = SHOT_RANGE, kind: str | None = None) -> bool:
    """Whether a throw from where she stands hits a monster of `kind` (default: a pig) dx along (either
    way) and dy below (+) with a star: in band, from shot_dx sideways out to `reach`, and no farther than
    the search picks it (max_dx). Terrain aside (can_hit: with it)."""
    return in_band(dy) and shot_dx(dy, kind) <= abs(dx) <= min(reach, max_dx(body_of(kind)[1]))


# --- the claw, up close --------------------------------------------------------------------------------

def claw_box(stance: str, claw: Claw | None = None) -> tuple[float, float, float, float]:
    """The box the client tests for the claw's `stance` (CLAW_STANCES), (near, far) ahead of her and
    (bottom, top) above her feet: its box in the game data (CLAW_BOXES, as she faces left: lt.x is its front)
    with its front cut to CLAW_FAR, as RustMS's point_blank::range has it. stabO1 20-65 ahead and 15-30 above
    her feet, stabO2 12-65 and 16-30. `claw`: another model's stances and cut (Claw), if not CLAW."""
    claw = claw or CLAW
    (left, up), (right, down) = dict(claw.boxes)[stance]
    front = min(max(left, -claw.far), right - 1)
    return -right, -front, -down, -up


def claw_takes(ahead: float, dy: float, top: float, half: float, stance: str, claw: Claw | None = None) -> bool:
    """Whether the claw's `stance` box (claw_box) takes a monster `ahead` in front of her (its middle), its
    feet dy below hers (+ below), `top` tall and `half` wide either side: its body reaches into it, on her
    level or not (the client doesn't ask; a `claw` model with a level gate does). Of those it takes, the
    nearest is swiped instead of shot."""
    claw = claw or CLAW
    near, far, bottom, roof = claw_box(stance, claw)
    return ((claw.level is None or abs(dy) <= claw.level)
            and reaches_into(ahead, dy, top, half, far, -bottom, roof, near))


def claw_chance(ahead: float, dy: float, top: float, half: float, claw: Claw | None = None) -> float:
    """The chance an attack from the ground takes a stance of the claw whose box reaches this monster (as in
    claw_takes): the share of CLAW_STANCES that do, one taken per attack, each as likely. 1 for a monster on
    her level with its near side within 65 of her, 0 for one whose feet are 30 or more above hers or whose top
    is under 15 above them (an Orange Mushroom walking 41 below her); the two boxes differ
    only in slivers (stabO2's starts 12 ahead of her, stabO1's 20, so a snail within a unit of her middle is
    in stabO2's alone; and stabO1's reaches down to 15 above her feet, stabO2's 16), where it's 0.5. It's
    swiped if it's also the nearest of those the stance takes (take_chances). `claw`: as in claw_box."""
    stances = (claw or CLAW).stances
    return sum(claw_takes(ahead, dy, top, half, s, claw) for s in stances) / len(stances)


def claw_dx(dy: float, top: float = PIG[0], half: float = PIG[1]) -> float:
    """How far off sideways (middle to middle) a monster dy below (+) her feet, `top` tall and `half` wide
    either side, is out of the claw's reach in front of her: CLAW_FAR plus half its width if a stance's box
    reaches its height (on her level a pig 99, a snail 84), else 0."""
    fronts = [claw_box(s)[1] for s in CLAW.stances]
    # at a box's front edge, only the monster's height decides whether the box takes it
    reached = [far for s, far in zip(CLAW.stances, fronts) if claw_takes(far, dy, top, half, s)]
    return max(reached) + half if reached else 0.0


def claw_edge(half: float) -> float:
    """How far off (middle to middle) a monster on her level `half` wide either side has to be for a throw
    from the ground not to be a claw swipe: its body clear of the claw's boxes (CLAW_FAR), CLAW_MARGIN to
    spare. A pig 111, an Orange Mushroom 109, a slime 105, a snail 96 (half widths 34, 32, 28, 19)."""
    return CLAW_FAR + half + CLAW_MARGIN


def star_sure(half: float) -> float:
    """How far off she steps back to for a star from the ground at a monster on her level `half` wide either
    side: 3 past claw_edge, so a step that comes up a little short still clears it."""
    return claw_edge(half) + 3


STAR_SURE = star_sure(MONSTER_HALF["snail"])  # 99: a snail's (124 with the swings taken as 90 ahead)


# --- what an attack from the ground takes ------------------------------------------------------------------

def ground_pick(ahead: float, dy: float, top: float, half: float) -> tuple | None:
    """Where a monster (as in claw_takes) comes in the order an attack from the ground takes them, the
    least first, or None if it can't take it, terrain aside: the claw first if a stance of it reaches the
    monster (claw_chance; the nearest, by distance), then the star's search (the first band). Within a band
    the nearest comes first here only for a steady order: the client takes the one that entered the map
    first, which can't be seen (take_chances has any of them take it)."""
    if claw_chance(ahead, dy, top, half) > 0:
        return 0, math.hypot(ahead, dy)
    k = search_band(*body_box(ahead, dy, top, half))
    return None if k is None else (1 + k, abs(ahead))


def take_chances(bodies: list[tuple[float, float, float, float]]) -> list[float]:
    """The chance an attack from the ground takes each of these monsters, (ahead, dy, top, half) each as in
    claw_takes, terrain aside. Each claw stance (one per attack, each as likely) swipes the nearest whose body
    reaches into its box, by distance from her feet to its; with none there the star goes to the search's
    first band, any of the monsters in it as likely as another (the client takes the one that entered the map
    first: RustMS Combat.cpp:1695-1720). So a monster both stances reach, nearest, is surely swiped; one only
    stabO2 reaches, half the time; two in the same band each take half the stars."""
    return [swipe + star for swipe, star in attack_chances(bodies)]


Band = Callable[[tuple[float, float, float, float]], int | None]  # a body (as in claw_takes) -> its search band


def attack_chances(bodies: list[tuple[float, float, float, float]], claw: Claw | None = None,
                   band: Band | None = None) -> list[tuple[float, float]]:
    """take_chances, each monster's chance of the claw's swipe and of the star apart: (swipe, star). `claw`:
    the claw's model (Claw), if not CLAW; `band`: the band of the star's search a body is picked in (None: not
    picked), if not search_band's (the probes' scorer tries other windows)."""
    claw = claw or CLAW
    bands = [search_band(*body_box(*b)) if band is None else band(b) for b in bodies]
    first = min((k for k in bands if k is not None), default=None)
    star = [i for i, k in enumerate(bands) if k is not None and k == first]
    out = [[0.0, 0.0] for _ in bodies]
    share = 1 / len(claw.stances)
    for s in claw.stances:
        swiped = [i for i, b in enumerate(bodies) if claw_takes(*b, s, claw)]
        if swiped:
            out[min(swiped, key=lambda i: math.hypot(bodies[i][0], bodies[i][1]))][0] += share
        else:
            for i in star:
                out[i][1] += share / len(star)
    return [(swipe, st) for swipe, st in out]


# --- what stops a star ---------------------------------------------------------------------------------

def _crosses(w: Segment, line: float) -> bool:
    """Whether wall `w` crosses the level line at map y `line` (a wall split into footholds, end to end,
    crosses it once: each foothold from its top, inclusive, to its bottom)."""
    return min(w.y1, w.y2) <= line < max(w.y1, w.y2)


def faces(w: Segment) -> int:
    """Which way wall `w`'s outer face looks: +1 (right) for one drawn downward (y1 < y2), -1 for one drawn
    upward. The client's open side of a foothold is left of its direction (above a floor drawn left to right),
    so a block's right wall faces right and its left wall left."""
    return 1 if w.y2 > w.y1 else -1


def _stops(w: Segment, a: float, b: float) -> bool:
    """Whether wall `w` stands between map x `a` (her) and `b` (the target), her on its outer side (faces) and
    the target on its inner side or at it: the line goes in at its outer face."""
    f = faces(w)
    return (a - w.x1) * f >= 0 >= (b - w.x1) * f and not (a == b == w.x1)


def _inside(x: float, p: Platform) -> float:
    """Map x, 1 unit inside platform `p`'s ends."""
    return min(max(x, p.left + 1), p.right - 1)


def hit_point(x: float, row: float, d: int, mx: float, feet: float, top: float, half: float,
              lift: float = 0.0) -> tuple[float, float]:
    """Where a star from her at map x `x`, facing `d` (+1 right), hits a monster's body, as the v83 client has
    it (RustMS ShotPath.h:456-489, hit_point): the middle of where the shot's row (1 unit tall at map y `row`,
    from SEARCH_START to STAR_RANGE ahead of her) crosses the body (its middle at map x `mx`, its feet at map y
    `feet`, `top` tall and `half` wide either side, raised `lift` off its feet), moved onto the body's nearest
    edge where they miss. So a body below the row is hit at its top, one above it at its bottom (its feet),
    and one the row crosses at the row; and along it, at the middle of the body's part within the row."""
    lo, hi = sorted((x + d * SEARCH_START, x + d * STAR_RANGE))
    left, right = mx - half, mx + half
    hx = right if lo >= right else left if hi <= left else (max(lo, left) + min(hi, right)) / 2
    return hx, min(max(row, feet - top), feet - lift)


def _hit_on(a: float, row: float, b: float, feet: float, top: float, half: float,
            target: Platform | None) -> tuple[float, float]:
    """hit_point from her at map x `a` (the shot's row at map y `row`) at a body whose middle is at map x `b`
    and feet at map y `feet`, `top` tall and `half` wide either side, facing it; its x taken 1 unit inside the
    monster's platform `target`, if known, as the monster itself is (_inside). Where the row ends inside a
    body hanging over its platform's end, the client's hit point is past that end: from platform 4 a pig on 2
    within 22 of 2's end, 368-410 off, would be hit over the step's edge, short of the step wall, where no star
    from 4 reached one on 2 in the logs and the user saw the step stop them (2026-10-21); and from a jump a
    slime by the basin's side right beside her (the row starts 65 ahead, so its far edge) at the wall's foot."""
    hx, hy = hit_point(a, row, 1 if b >= a else -1, b, feet, top, half)
    return (hx if target is None else _inside(hx, target)), hy


def _side(line: tuple, x: float, y: float) -> float:
    """Which side of a foothold's line (x1, y1, x2, y2) a point is on: negative on its open side, left of
    its direction (above a floor drawn left to right), positive behind it (RustMS ShotPath.h side)."""
    x1, y1, x2, y2 = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def _behind_corner(first: tuple, second: tuple, x: float, y: float) -> bool:
    """Whether a point is behind the corner where foothold `first` runs into `second` (ShotPath.h
    behind_corner, FUN_009b3f1b): behind either where they turn toward their open side, as at a step, and
    behind both where they turn away from it, as at a ledge."""
    behind = _side(first, x, y) > 0, _side(second, x, y) > 0
    return any(behind) if _side(first, second[2], second[3]) <= 0 else all(behind)


def _near_line(footholds, x0: float, y0: float, x1: float, y1: float) -> list[Segment]:
    """The footholds whose bounds touch those of the box from (x0, y0) to (x1, y1), as the client's foothold
    query gives a line's (FUN_00508740)."""
    lo_x, hi_x, lo_y, hi_y = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    return [f for f in footholds if min(f.x1, f.x2) <= hi_x and max(f.x1, f.x2) >= lo_x
            and min(f.y1, f.y2) <= hi_y and max(f.y1, f.y2) >= lo_y]


def sight_stop(platform: Platform, sx: float, sy: float, ex: float, ey: float,
               footholds: list[Segment] | None = None) -> Segment | None:
    """The foothold that stops the v83 client's line of sight from map (sx, sy) to (ex, ey) first, or None
    (RustMS ShotPath.h:382-445, blocked, FUN_00a457d0): any floor of the map (platform.floors) and the walls of
    the foothold group of `platform`, the one she last stood on (platform.walls); each only where the line goes
    from its open side (above a floor, outside a wall) to its closed side. Through a foothold's end, only if the
    line also ends behind the corner it makes with the one joined there (_behind_corner); at a free end it
    passes. The nearest such crossing stops it. `footholds`: those to try, if already narrowed (_near_line)."""
    if sx == ex and sy == ey:
        return None
    shot = (sx, sy, ex, ey)
    if footholds is None:
        footholds = _near_line((*platform.floors, *platform.walls), sx, sy, ex, ey)
    best = None  # (how far along, foothold)
    for f in footholds:
        line = (f.x1, f.y1, f.x2, f.y2)
        start, end = _side(line, sx, sy), _side(line, ex, ey)
        if start > 0 or end < 0 or (start == 0 and end == 0):
            continue
        first, second = _side(shot, f.x1, f.y1), _side(shot, f.x2, f.y2)
        if (first < 0 and second < 0) or (first > 0 and second > 0):
            continue
        if first == 0 and (f.prev is None or not _behind_corner(f.prev, line, ex, ey)):
            continue
        if first != 0 and second == 0 and (f.next is None or not _behind_corner(line, f.next, ex, ey)):
            continue
        t = -start / (end - start)
        if best is None or t < best[0]:
            best = (t, f)
    return None if best is None else best[1]


def wall_between(platform: Platform | None, x: float, y: float, tx: float, rise: float = 0.0,
                 target: Platform | None = None, shot_h: float | None = None, ty: float | None = None,
                 body: tuple[float, float] | None = None, line: str | None = None) -> Segment | None:
    """What stops a star from her (map x on `platform`, the one she last stood on; her feet at y, `rise`
    units up in a jump) on its way to a monster at map x `tx`, by the line LINE (or `line`) has: a wall of her
    own foothold group (Platform.walls) that the line goes into at its outer face, or (the v83 line) a floor
    it goes down through. None if the star gets there. Her x is taken 1 unit inside her platform's ends, and
    `tx` inside the target's platform (`target`, if known): a monster is never past its platform's end, and
    one at the foot of a wall of her group that rises from its own platform's end (the basin's sides, from its
    floor) isn't behind it. The level line runs SHOT_H above her feet (`shot_h`, if not: the probes' scorer
    tries others) across to `tx`; the v83 line from there to the star's hit_point on the monster's body (inside
    `target`, if known: _hit_on): its feet at map y `ty` (by default on `target` at tx, else at her height),
    `body` (how tall, half its width: body_of; a pig's by default)."""
    if platform is None:
        return None
    row = y - rise - (SHOT_H if shot_h is None else shot_h)
    a = _inside(x, platform)
    b = tx if target is None else _inside(tx, target)
    if (LINE if line is None else line) == "level":
        walls = [w for w in platform.walls if _crosses(w, row) and _stops(w, a, b)]
        return min(walls, key=lambda w: abs(w.x1 - a), default=None)
    feet = ty if ty is not None else target.y_at(b) if target is not None else y
    top, half = body or PIG
    return sight_stop(platform, a, row, *_hit_on(a, row, b, feet, top, half, target))


def wall_in_way(platform: Platform | None, x: float, y: float, tx: float, rise: float = 0.0,
                target: Platform | None = None, ty: float | None = None,
                body: tuple[float, float] | None = None) -> Segment | None:
    """wall_between, as her decisions see it: None whatever the terrain with WALLS_BLOCK off."""
    return wall_between(platform, x, y, tx, rise, target, ty=ty, body=body) if WALLS_BLOCK else None


def wall_to(platform: Platform | None, x: float, y: float, m: Monster, rise: float = 0.0,
            target: Platform | None = None) -> Segment | None:
    """wall_in_way from map (x, y) on `platform` to monster `m`, by its body (body_of). `target`: m's platform
    if known (if m is on hers, it's hers)."""
    if target is None and platform is not None and m.platform == platform.id:
        target = platform
    return wall_in_way(platform, x, y, m.x, rise, target, m.y, body_of(m.kind))


def shielded(platform: Platform | None, x: float, y: float, m: Monster, others: list[Monster],
             rise: float = 0.0) -> Monster | None:
    """A monster of `others` that an attack from map (x, y) at monster `m` takes before it and whose line is
    stopped (wall_to), or None: the client checks the line to its pick alone, and stopped the attack hits
    nothing, so no star gets to `m` from there. From the ground (`rise` 0) one in an earlier band of the star's
    search than m's (one in m's own band takes some of the stars, not all); from a jump, one nearer in the
    attack's area (in_area). A monster the claw's stance reaches takes a swipe, which nothing stops: never one
    of those. None too with WALLS_BLOCK off. Bodies by kind, walking (body_of)."""
    if not WALLS_BLOCK or platform is None:
        return None
    d = 1 if m.x >= x else -1

    def order(o: Monster) -> float | None:
        ahead, dy = (o.x - x) * d, o.y - y
        top, half = body_of(o.kind)
        if rise:
            return abs(ahead) if in_area(ahead, dy, top, rise, half) else None
        if claw_chance(ahead, dy, top, half) > 0:
            return None
        return search_band(*body_box(ahead, dy, top, half))

    mine = order(m)
    if mine is None:
        return None
    for o in others:
        if o is m or (o.x - x) * d <= 0:
            continue
        k = order(o)
        if k is not None and k < mine and wall_to(platform, x, y, o, rise) is not None:
            return o
    return None


def can_hit(platform: Platform | None, x: float, y: float, m: Monster, reach: float = SHOT_RANGE,
            rise: float = 0.0, target: Platform | None = None, others: list[Monster] = ()) -> bool:
    """Whether a star from map (x, y) on `platform` hits monster `m`: in reach by its kind (can_shoot), its
    line not stopped (wall_to; `rise`: her height in a jump, `target`: m's platform), and none of `others` (the
    monsters around her) that the attack takes before it stopped either (shielded). Whether m is the pick now
    isn't asked: a monster the attack takes first whose line is clear takes the stars until it's dead, and
    then m's turn comes, so it only orders the kills (her decisions shoot what's in the way first anyway). One
    whose star is stopped dies to none of them, so from here m gets none while it stands there."""
    return (can_shoot(m.x - x, m.y - y, reach, m.kind) and wall_to(platform, x, y, m, rise, target) is None
            and shielded(platform, x, y, m, others, rise) is None)


def clear_length(platform: Platform, x: float, y: float, target: Platform, lo: float, hi: float,
                 body: tuple[float, float] | None = None, step: float = 2.0) -> float:
    """How much of platform `target` from map x `lo` to `hi` a star from her (map x on `platform`, her feet
    at y) gets to by the v83 line (wall_between), at a monster of `body` (how tall, half its width: body_of; a
    pig's) standing there: tried in the middle of every `step` units."""
    top, half = body or PIG
    row, a = y - SHOT_H, _inside(x, platform)
    n = max(1, math.ceil((hi - lo) / step))
    width = (hi - lo) / n
    xs = np.minimum(np.maximum(lo + (np.arange(n) + 0.5) * width, target.left + 1), target.right - 1)  # _inside
    feet = _ys_at(target, xs)
    near = _near_line((*platform.floors, *platform.walls), min(a, lo - half), min(row, float(feet.min()) - top),
                      max(a, hi + half), max(row, float(feet.max())))  # what any of those lines can meet
    hx, hy = _hits_on(a, row, xs, feet, top, half, target)
    return width * int(np.count_nonzero(_sight_clear(near, a, row, hx, hy)))


def _ys_at(p: Platform, xs: np.ndarray) -> np.ndarray:
    """Platform.y_at for many map x on `p` at once, in the same arithmetic (each x's first foothold that has it,
    as Segment.y_at)."""
    ys = np.where(xs < p.left, p.segments[0].y1, p.segments[-1].y2)
    for s in reversed(p.segments):
        y = s.y1 if s.x2 == s.x1 else s.y1 + (xs - s.x1) / (s.x2 - s.x1) * (s.y2 - s.y1)
        ys = np.where((s.x1 <= xs) & (xs <= s.x2), y, ys)
    return ys


def _hits_on(a: float, row: float, b: np.ndarray, feet: np.ndarray, top: float, half: float,
             target: Platform) -> tuple[np.ndarray, np.ndarray]:
    """_hit_on for many bodies at once (their middles at map x `b`, their feet at map y `feet`), in the same
    arithmetic, so with the same answers."""
    d = np.where(b >= a, 1.0, -1.0)
    near, far = a + d * SEARCH_START, a + d * STAR_RANGE
    lo, hi = np.minimum(near, far), np.maximum(near, far)
    left, right = b - half, b + half
    hx = np.where(lo >= right, right, np.where(hi <= left, left, (np.maximum(lo, left) + np.minimum(hi, right)) / 2))
    return np.minimum(np.maximum(hx, target.left + 1), target.right - 1), np.minimum(np.maximum(row, feet - top), feet)


def _sight_clear(footholds: list[Segment], sx: float, sy: float, ex: np.ndarray, ey: np.ndarray) -> np.ndarray:
    """Whether each line of sight from map (sx, sy) to (ex, ey) gets there, none of `footholds` stopping it:
    sight_stop for many lines at once, by the same tests in the same arithmetic (so with the same answers), for
    clear_length, where the spots try tens of thousands (one at a time, every platform's spots took 1.5 s, 1.0 s
    so)."""
    stopped = np.zeros(ex.shape, bool)
    for f in footholds:
        line = (f.x1, f.y1, f.x2, f.y2)
        start = _side(line, sx, sy)
        if start > 0:
            continue
        end = _side(line, ex, ey)
        first, second = _side((sx, sy, ex, ey), f.x1, f.y1), _side((sx, sy, ex, ey), f.x2, f.y2)
        hit = ~(end < 0) & ~((first < 0) & (second < 0)) & ~((first > 0) & (second > 0))
        if start == 0:
            hit &= ~(end == 0)
        at_start = first == 0
        hit &= ~at_start if f.prev is None else ~at_start | _behind_corners(f.prev, line, ex, ey)
        at_end = ~at_start & (second == 0)
        hit &= ~at_end if f.next is None else ~at_end | _behind_corners(line, f.next, ex, ey)
        stopped |= hit
    return ~stopped | ((ex == sx) & (ey == sy))


def _behind_corners(first: tuple, second: tuple, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """_behind_corner for many points at once."""
    behind = _side(first, x, y) > 0, _side(second, x, y) > 0
    return behind[0] | behind[1] if _side(first, second[2], second[3]) <= 0 else behind[0] & behind[1]


def clear_span(platform: Platform | None, x: float, y: float, rise: float = 0.0) -> tuple[float, float]:
    """The map x a star from her (as in wall_in_way, by the level line) gets to either way before a wall of
    her foothold group stops it: from the nearest such wall on her left facing right to the nearest on her
    right facing left (infinite where there's none, and either way with WALLS_BLOCK off). With LINE 'v83' a
    floor stops more, by how far below her a monster is, which this leaves out."""
    if not WALLS_BLOCK or platform is None:
        return -math.inf, math.inf
    line, a = y - rise - SHOT_H, _inside(x, platform)
    walls = [w for w in platform.walls if _crosses(w, line)]
    return (max((w.x1 for w in walls if w.x1 <= a and faces(w) > 0), default=-math.inf),
            min((w.x1 for w in walls if w.x1 >= a and faces(w) < 0), default=math.inf))


# --- what an attack from a jump picks --------------------------------------------------------------------

def reaches_into(ahead: float, dy: float, top: float, half: float, far: float, down: float, up: float,
                 near: float = 0.0) -> bool:
    """Whether a monster's body reaches into a box in front of her: from `near` (her, 0) to `far` ahead,
    `down` below to `up` above her feet (a negative `down`: from that far above them). The monster's middle
    is `ahead` units in front of her, its feet `dy` below hers (+ below), and its body `top` tall and `half`
    its width either side."""
    return near - half <= ahead <= far + half and top - dy >= -down and -dy < up


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
