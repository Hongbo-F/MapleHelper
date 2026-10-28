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
the two; the simulator's world has either, sim.world.AIR_PICK). Since 2026-10-25 the area reaches to the first
window's top over her feet (44), and past it the search's own windows from where her feet are then pick too
(`air_takes`, AIR_SEARCH): they widen with the distance, so a monster on a platform above is hit from high in the
jump up close and from lower, or from the ground, farther off, as the user's play has it. `shots` and `Coverage` say,
for where she stands, every jump and attack key that surely hits each monster: one query for the stance, the hunt
loop, grinding and her trips, the throw from the ground its case with no jump. From 2026-10-27 the area is gone
(AIR_MODEL 'search'; 'widening' puts it back): from a jump the star's search alone picks, from her feet when the
attack starts, as RustMS has it. The live hunts of 2026-10-25 and 26 on Tree Dungeon say so: jump throws in place
at a monster on her platform hurt one under 40 ahead half as often as farther off, and from an air turn's later key,
her feet 52-56 up, one on her level or lower a third of the time (`air_sure`: what an attack keyed so surely picks).

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
from ..localization.monsters import MONSTER_HALF, MONSTER_HEIGHT, MONSTER_TOP, Monster
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
# 60 below her): 18-21 below her feet. Of its top, only that monsters 60 above her are in it is known: from 42-49
# up, their feet 11-18 over hers. So it's the first window's top (SEARCH_TOP0, 44; 65 until 2026-10-25, a guess),
# and past it the search's own windows reach higher the farther ahead (AIR_SEARCH): the user, 2026-10-25, "monster
# 100 units up hit only near the top of the jump. After the jump you can hit monsters that is far away
# horizontally, remember the widening vertical search as horizontal scale out".
AREA_DOWN = 19
AREA_UP = SEARCH_TOP0
# Whether an attack from a jump also picks what the star's search would from where her feet are then (its bands, the
# windows about SHOT_H over her feet widening with the distance: air_takes), the area aside. The user's play on
# Tree Dungeon (watch 2026-10-24 17:05, 4.5 min) bears it out: of her 30 kills of monsters 15+ above her, 23 have a
# look within 0.2 s of the attack's start (put back from the kill's EXP: the star's delay and flight) from where the
# ground's search (UP_BY_SEARCH) or the air's area or bands pick it; by the old area (to 65 above her feet) and the
# ground capped 60 above her (SHOT_UP), 20 (at 0.35 s, 24 against 21; the other 6 no model has, their attack likely
# earlier in a jump the window misses). From a jump, a Green Mushroom 58 above her feet 229 off (in the band's
# window, 74 there; over the area's 44), from 22 at 19, 133 above where she stood (D:/MapleHelper-scratch/ov-jump/
# val_user.py). False: the area alone.
AIR_SEARCH = True
# How an attack starting in the air picks its target (her decisions; the simulator's world as sim.world.AIR_PICK has it):
# 'search', from 2026-10-27, the star's search from her feet when it starts, as from the ground but with no claw and no
# area (air_takes: band k from SEARCH_START + SEARCH_STEP k ahead, its window 12 - 5k to 44 + 5k above her feet then),
# as RustMS's port of the v83 client has no case for the air and this client never whacks in the air (a Lucky Seven key
# from a jump took MP on 93-97% of throws at every distance, 113 of 119 at 0-20 ahead, where from the ground under 90
# ahead only 27% did: the claw's swipe); or 'widening', the area about her feet (AREA_DOWN below them to AREA_UP above,
# from her middle out) and the search's windows past it, as from 2026-10-25. The live hunts of 2026-10-25 23:43 to 10-26
# 23:47 (14 runs on Tree Dungeon, the out-of-darts spells left out) say the search: of 788 jump throws in place at a
# monster on her platform (her feet at the attack's start by physics.pose from where the perception service had her,
# each monster walked on by its track to then: D:/MapleHelper-scratch/clear/diag-slopes/replay.py), 'widening' picked
# the target at 0-40 ahead (her middle to its) for 70% of the no-turn throws' hits where 50% hurt it, and from an air
# turn's 120 ms key (her feet 52-56 up) for 72% where 34% did; the search fits them 111 log-likelihood better (-587
# against -699: D:/MapleHelper-scratch/clear/impl/calib/fitworld.py), and its first window's floor 12 above her feet
# fits best (8 and 16: 7 and 9 worse; diag-slopes/fit_floor.py). So from a jump nothing is picked until the body's far
# side is past SEARCH_START (a mushroom's middle 37 ahead), 65 being where the claw's reach ends and the star's search
# begins (RustMS PointBlank.h MAX_REACH), and a monster lower than her by her height in the jump and the slope is picked
# only farther off: keyed 40 ms after the jump key (her feet 21-26 up) a mushroom whose feet are up to about 10 below
# hers 65-85 ahead, 15 at 85-105; keyed 120 ms (52-56 up), one on her level only from 145 ahead. Planned jumps at
# monsters 40+ above her, both ways alike (36% and 34% hurt among those picked: diag-slopes/fit_top.py).
AIR_MODEL = "search"
AIR_MODELS = ("search", "widening")
# Whether a throw from the ground reaches monsters more than SHOT_UP above her where the star's search picks them (its
# windows widen 5 up for every 20 ahead: 100 above her from about 270 off, 124 at most, at 385-400 ahead), as the
# simulator's world always had it. On by the user's choice, 2026-10-25 ("Turn it on now"), asked whether to wait for
# the top-60 probe: their own play, from the ground, killed Green Mushrooms 69 above her 185-214 off (from 15 at 14),
# 74-76 above 223 off (21 at 19) and 100 above 291 off (24 at 22), where the client's first window's top (SEARCH_TOP0,
# 44) has the search pick them and the logs' 21 wouldn't, and in the simulator it gave the hunt loop +21.8% EXP (16
# seeds) against the cap. The review of 2026-10-25 had it off as that's suggestive, not decisive: the ±0.2 s window put
# back from the kill misplaces where in a jump the attack came (6 of the 30 kills above her, all 61-133 above with
# looks on the ground only, no model explains), the first was airborne 0.09 s on, the third's only look in the window
# was airborne (as she lifted off), and the second has a single look; its shot logs' fit (the bands' floor and the
# lean to 21) wasn't run again. What the probe (`python -m gamelab probe run top-60`, then `probe score`) would still
# settle is that top: at 21 a pig 100 above her is picked from 357 off, not 271, and she'd throw from the ground at
# monsters above her that the star doesn't take, which the simulator, picking by the same windows, can't show (in a
# world with 21, 34% of her throws from the ground hit nothing against 27% capped: README, Attacks from her jumps).
# False: SHOT_UP's cap, 60 above her, as from the review until the user's choice.
UP_BY_SEARCH = True
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
    way) and dy below (+) with a star: in band (above her, with UP_BY_SEARCH, as high as the search's windows
    reach: shot_dx says from how far off), from shot_dx sideways out to `reach`, and no farther than the search
    picks it (max_dx). Terrain aside (can_hit: with it)."""
    return ((in_band(dy) or (UP_BY_SEARCH and dy < 0)) and shot_dx(dy, kind) <= abs(dx)
            <= min(reach, max_dx(body_of(kind)[1])))


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


def ground_sure(half: float) -> tuple[float, float]:
    """The nearest and farthest ahead of her (middle to middle) her attack from the ground surely takes a monster on her
    level `half` wide either side, where it will be when the attack starts: as a star, not the claw's swipe, from its
    claw's edge (claw_edge: a Green or Horny Mushroom 105) to SHOT_RANGE (410). The one near and far limit the stance's
    decisions about the ground go by (attack.stance: too_close, close_to, _from_here, _clear_step through star_sure, a
    landing's clearance), and the seam S1's aim facade replaces: the thief's here. The magician's Magic Claw takes one
    from its search's first band less half the body to its last band's end and half the body (classes.magician.SEARCH:
    50 - half to 300 + half, nothing nearer, never behind her)."""
    return claw_edge(half), float(SHOT_RANGE)


STAR_SURE_MARGIN = 3.0  # units past ground_sure's nearest a step away goes to, so one a little short still clears it


def star_sure(half: float) -> float:
    """How far off she steps back to for a star from the ground at a monster on her level `half` wide either
    side: STAR_SURE_MARGIN past ground_sure's nearest (the claw's edge), so a step that comes up a little short still
    clears it."""
    return ground_sure(half)[0] + STAR_SURE_MARGIN


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
    if known; by default the one its id names on her platform's map (Platform.peer; hers, if it's on hers): the
    star's hit point is taken on it, as the simulator's world and the spots take it (wall_between). Without it,
    from 4 at monsters within 22 of 2's end, 368-410 off, the hit point fell short of the step wall, and her
    decisions threw at them (115 of 118,739 places in reach on the map, every kind, all 4 -> 2: the only ones
    where they and the world's line differed), where the world's stars, and the game's, stop at the step."""
    if target is None and platform is not None:
        target = platform if m.platform == platform.id else platform.peer(m.platform)
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
            return abs(ahead) if in_area(ahead, dy, top, rise, half, o.platform == platform.id) else None
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


def air_takes(ahead: float, dy: float, top: float, half: float, margin: float = 0.0, model: str | None = None) -> bool:
    """Whether an attack starting with her in the air can pick a monster `ahead` units in front of her (its middle),
    its feet `dy` below her feet then (+ below), its body `top` tall and `half` wide either side, by AIR_MODEL (or
    `model`): 'search', its body overlaps its band's window of the star's search from her feet then (search_band: from
    SEARCH_START ahead, the windows about SHOT_H over her feet one unit wider up and down for every SEARCH_SPREAD
    ahead); 'widening', that (with AIR_SEARCH) or its body reaching into the area about her feet (AREA_DOWN below them
    to AREA_UP above, its near side within STAR_RANGE ahead). By `margin`: reaching that far into it, every way (a pick
    that is sure). Of those, the nearest band's is picked ('widening': the nearest)."""
    if (model or AIR_MODEL) == "widening":
        if reaches_into(ahead, dy, top, half - margin, STAR_RANGE, AREA_DOWN - margin, AREA_UP - margin):
            return True
        if not AIR_SEARCH:
            return False
    near, far, bottom, roof = body_box(ahead, dy, top, half)
    return search_band(near + margin, far - margin, bottom + margin, roof - margin) is not None


def in_area(ahead: float, dy: float, top: float, rise: float = 0.0, half: float = 0.0, mine: bool = False,
            model: str | None = None) -> bool:
    """Whether an attack's star can pick a monster `ahead` units in front of her (its middle), its feet `dy`
    below hers as she stands (+ below), its body `top` tall (and `half` wide either side), when the attack
    starts with her `rise` units up in a jump (air_takes, by AIR_MODEL or `model`), from the ground (`rise` 0) as the
    star's search picks it (picks). 'search': on her own platform (`mine`: sloped, a monster on it can be 36 units lower
    than her on Tree Dungeon) or on her level or above it, the search from her feet then; 'widening': the area about
    her feet then, and for one above her level (more than LEVEL over her feet as she stands) the search's widening
    windows too. What her decisions keep as in a jump throw's path (behavior.fight._in_path: what a kill is matched to,
    and what's kept thrown at, hidden, for a while) and what shields others (shielded) go by this. Below her level off
    her platform the windows from a jump reach far monsters only (from 44 up, 104 below her feet at 385-400 ahead),
    which take a star only with none nearer: listed, they were kept thrown at and hidden, and the step-down probe's
    fights on 2 went on to its 120 s cap on seeds 1, 5 and 9 of 1-12 (none at the base; with them left out, as here,
    only 4)."""
    if rise == 0:
        return picks(ahead, dy, top, half)
    if (model or AIR_MODEL) == "search":
        return (mine or dy <= LEVEL) and air_takes(ahead, dy + rise, top, half, model="search")
    if dy < -LEVEL:
        return air_takes(ahead, dy + rise, top, half, model="widening")
    return reaches_into(ahead, dy + rise, top, half, STAR_RANGE, AREA_DOWN, AREA_UP)


def air_sure(ahead: float, dy: float, top: float, half: float, gap_ms: float, margin: float = AIR_MARGIN) -> bool:
    """Whether an attack keyed `gap_ms` after the jump key of a jump in place surely picks a monster `ahead` units in
    front of where she jumped from (its middle where it will be then), its feet `dy` below hers as she stood (+ below:
    the slope, platform.y_at at its x less hers), `top` tall and `half` wide either side: air_takes `margin` into it from
    her feet at both ends of the attack's start (JUMP_LAG after its key, or ATTACK_DECIDE_LAG_MS later: physics). Keyed
    40 ms after the jump key (her feet 21-26 up) a Green or Horny Mushroom (51 tall, 28 wide either side) on her level
    or above it from 44 ahead, 10 below her takeoff from 64, 15 below from 84, 20 below from 104; keyed 120 ms (52-56
    up), on her level from 144, 10 above her from 104, 5 below from 164."""
    early, late = physics.jump_rise(gap_ms + physics.JUMP_LAG * 1000), air_rise(gap_ms)
    return all(air_takes(ahead, dy + rise, top, half, margin) for rise in (early, late))


def air_rise(gap_ms: float) -> float:
    """How high she is above the ground she jumped from when an attack keyed `gap_ms` after the jump key
    picks its target: JUMP_LAG after its key, as a jump lifts off, or up to ATTACK_DECIDE_LAG_MS later
    (physics). 49 units at 100 ms (a jump throw), 44 at 85 ms (a jump back), 26 at 40 ms."""
    return physics.jump_rise(gap_ms + physics.JUMP_LAG * 1000 + physics.ATTACK_DECIDE_LAG_MS)


def air_floor(gap_ms: float) -> float:
    """How high above the ground she jumped from the area of an attack keyed `gap_ms` after the jump key
    starts (air_rise less AREA_DOWN): 30 at 100 ms, 25 at 85 ms, 7 at 40 ms."""
    return air_rise(gap_ms) - AREA_DOWN


def turn_key_ms(top: float) -> int:
    """The attack key of a jump that turns her in the air, at a monster on her platform at least `top` tall (ms after
    the jump key): physics.AIR_TURN_ATTACK_MS, the turn's arrow physics.AIR_TURN_LEAD_MS before it. With AIR_MODEL
    'widening', where its attack still picks such a monster (air_hits), else the latest one from
    physics.JUMP_BACK_ATTACK_MS on that does (a pig: 105, its arrow 45 ms before), or that if none does. With 'search'
    always AIR_TURN_ATTACK_MS: what the search picks goes by how far off a monster is as much as how tall, and keyed
    sooner after the arrow the turn goes the wrong way more often (36% at 25 ms after it, live on 2026-10-25), with her
    feet still 44 up, over a level mushroom's top for the search's first five bands."""
    if AIR_MODEL == "search":
        return physics.AIR_TURN_ATTACK_MS
    for k in range(physics.AIR_TURN_ATTACK_MS, physics.JUMP_BACK_ATTACK_MS, -5):
        if air_hits(top, k):
            return k
    return physics.JUMP_BACK_ATTACK_MS


def air_hits(top: float, gap_ms: float) -> bool:
    """Whether an attack keyed `gap_ms` after the jump key surely picks a monster on her level that is at least `top`
    tall (localization.monsters.MONSTER_TOP), somewhere up close. 'widening' (AIR_MODEL): its body reaches AIR_MARGIN
    into the area (a snail, 24, stays under the area of a jump throw keyed 100 ms after the jump, its floor 30 up, and
    of a jump back, 25; keyed 40 ms after it, 7, the attack picks every kind). 'search': its top AIR_MARGIN over the
    first band's window's floor (SEARCH_FLOOR0) above her feet then (air_rise), as in the band from SEARCH_START ahead:
    keyed 40 ms after the jump (her feet 26 up) a Green or Horny Mushroom (51), not a pig (38) or a slime (40), by 6 and
    4 units, nor a snail (24); keyed 60 ms an Orange Mushroom (55) too; keyed 85 ms or later none."""
    if AIR_MODEL == "search":
        return top - air_rise(gap_ms) >= SEARCH_FLOOR0 + AIR_MARGIN
    return top >= air_floor(gap_ms) + AIR_MARGIN


# --- what her attacks reach: from the ground and from each of her jumps -----------------------------------
# "Which monsters can she hit by jumping" as one query (`shots`, `Coverage`), for the stance, the hunt loop, grinding
# and her trips alike. An attack picks its target when it starts (JUMP_LAG after its key, up to ATTACK_DECIDE_LAG_MS
# later: the same input lag as a jump's lift-off, so it starts about as long after lift-off as its key came after the
# jump key), from where she is then; the star that leaves STAR_DELAY_MS (520) after the key is only a picture. From a
# jump, the nearest monster whose body reaches into the area about her feet or into its band's window of the star's
# search from her feet then (air_takes: the windows about SHOT_H over her feet, 12 - 5k to 44 + 5k above them in band
# k, SEARCH_START + 20 k ahead: one unit wider up and down every SEARCH_SPREAD ahead), her feet wherever the jump has
# them then (physics.pose: her arc over time, for every kind of jump a key plan makes, physics.jump_of); its line is
# checked as from the ground (wall_in_way, from her feet then: as sim.world's sight_stop has it). From the ground the
# special case with no jump: can_hit, the same search from her feet on the ground. So how high a monster can be and
# still be hit follows from her arc and the distance: her feet at most JUMP_HEIGHT (77) over where she jumped from,
# 0.2775 s after lift-off (physics.APEX_S); up close (the area, or the first bands) 44 over her feet, so a monster
# whose feet are 100 above where she stood only with the attack starting 155-400 ms after lift-off (her feet 62 up,
# AIR_MARGIN into it: physics.lift_time), 110 above within 70 ms of the top (207-348 ms), 115 at most (77 + 44 -
# AIR_MARGIN), at the very top; farther off the windows reach higher, so lower in the jump (and from the ground with
# UP_BY_SEARCH, on by the user's choice: a pig 100 above her from 271 off, 110 from 311; capped at SHOT_UP, 60,
# without), up to 124 over her feet at the last band (385-400 ahead), from a jump 201 above where she stood.
KEY_FIRST_MS = physics.JUMP_ATTACK_MS  # the soonest an attack key goes after its jump key (40: in game, keyed so, a
                     # star from the jump killed a snail; keyed at the jump key the game may take the attack first,
                     # on the ground, which nothing has measured)
KEY_LAST_MS = 480    # the latest tried: it starts 545 ms after the jump key, as she comes down to where she left
KEY_STEP_MS = 10     # the attack keys tried, this far apart
KEY_SLACK_MS = 15    # a key counts as sure only if keyed this much sooner or later the attack picks the same monster:
                     # the game sees a key 7.7 ms either side of when it's due on average (body.Lags, one sd), twice
TURN_KEY_MS = physics.AIR_TURN_ATTACK_MS  # a jump that turns her in the air (physics.AIR_TURN_MS: the arrow 60 ms
                     # after the jump key) has its attack key this long after the jump key at the soonest, as the jump
                     # backs do (120: physics.AIR_TURN_LEAD_MS after the arrow; 85 until 2026-10-26, 36% of those went
                     # the wrong way live)
JUMP_KINDS = ("up", "forward", "back")  # the jumps tried (physics.jump_of): in place, toward the throw, away from it


@dataclass(frozen=True)
class Shot:
    """An attack she can make from where she stands, at `target`: from the ground (`jump` None), or from a jump
    (physics.Jump) with the attack key `key_ms` after the jump key (keyed anywhere in `window`, ms after it, the attack
    still picks `target` surely), throwing `side` (+1 right, -1 left); `turn`: an 'up' that turns her in the air first
    (physics.AIR_TURN_MS; a jump back always does). Where she is when it picks (`x`, her feet `rise` over where she
    stood), and where the jump lands her (`land`, `land_x`, `land_s` s after lift-off); `alts`, a jump forward's or
    back's other ways of going (physics.jump_variants: lifting off early or late, slower, the kick short) and where
    each lands, as a route jump's are checked (physics.sure_landing): `lands_sure`, `land_span`."""
    side: int
    target: Monster
    jump: physics.Jump | None = None
    key_ms: int = 0
    window: tuple[int, int] = (0, 0)
    x: float = 0.0
    rise: float = 0.0
    land: Platform | None = None
    land_x: float = 0.0
    land_s: float = 0.0
    turn: bool = False
    alts: tuple = ()  # ((physics.Jump, (platform, x, s)), ...): the jump's variants and their landings

    @property
    def kind(self) -> str:
        """'ground', or the jump's kind ('up', 'forward', 'back')."""
        return "ground" if self.jump is None else self.jump.kind

    @property
    def lands_sure(self) -> bool:
        """Whether every way the jump may go (`alts`) lands on the platform it's meant to (`land`): a kick that leaves
        short (126 or 119 units/s) from under a platform's end lands up on it (review, 2026-10-25)."""
        return self.land is not None and all(land is self.land for _, (land, _, _) in self.alts)

    @property
    def land_span(self) -> tuple[float, float]:
        """The least and the most map x the jump lands at, its variants' (`alts`) with its own."""
        xs = [self.land_x, *(lx for _, (_, lx, _) in self.alts)]
        return min(xs), max(xs)

    def text(self) -> str:
        """The shot for a log line: "a jump forward throwing right, the attack 40 ms in (40-120), 26 up: horny-mushroom
        +180"."""
        way = "right" if self.side > 0 else "left"
        if self.jump is None:
            return f"from the ground {way}: {self.target.kind} {self.target.x - self.x:+.0f}"
        jump = {"up": "a jump in place", "forward": "a jump forward", "back": "a jump back"}.get(self.kind, self.kind)
        return (f"{jump} throwing {way}{' (turning in the air)' if self.turn and self.kind == 'up' else ''}, the "
                f"attack {self.key_ms} ms in ({self.window[0]}-{self.window[1]}), {self.rise:.0f} up: "
                f"{self.target.kind} {self.target.x - self.x:+.0f}")


Where = Callable[[Monster, float], float]  # a monster -> its map x that many s from now (its walk predicted)
Placed = list[tuple[Monster, float, float, float, float]]  # (monster, its x, its feet's y, how tall, half its width)


def _air_pick(px: float, feet: float, side: int, rows: Placed, margin: float) -> Monster | None:
    """The monster an attack from a jump picks from map (px, her feet `feet`), throwing `side`: the nearest whose body
    reaches into the area about her feet or its band's window of the search (air_takes), as sim.world's air_pick; None
    unless it reaches `margin` into it and no other reaching in at all is within `margin` as near (not surely it,
    then). `rows`: the monsters about, placed (Placed)."""
    best = second = None
    for m, mx, my, top, half in rows:
        ahead, dy = (mx - px) * side, my - feet
        if not air_takes(ahead, dy, top, half):
            continue
        row = (abs(ahead), m, ahead, dy, top, half)
        if best is None or row[0] < best[0]:
            best, second = row, best
        elif second is None or row[0] < second[0]:
            second = row
    if best is None:
        return None
    _, m, ahead, dy, top, half = best
    if second is not None and second[0] - best[0] < margin:
        return None
    if not air_takes(ahead, dy, top, half, margin):
        return None
    return m


def shots(model, platform: Platform, x: float, y: float, monsters: list[Monster],
          sizes: dict[str, MobBody] | None = None, kinds: tuple[str, ...] = JUMP_KINDS,
          where: Where | None = None, others: list[Monster] | None = None) -> list[Shot]:
    """Every attack from map (x, y) on `platform` that surely hits one of `monsters`, from the ground and from her
    jumps: one Shot per target and way of attacking, from the ground and for each of `kinds` of jump
    (physics.jump_of) each way she can throw from it ('up' either way, with and without a turn in the air; 'forward'
    toward the throw; 'back' away from it, turning in the air), with the soonest attack key from KEY_FIRST_MS (with a
    turn, turn_key_ms of the target's kind: TURN_KEY_MS, sooner for one too short to pick then, a pig 105 ms) that picks
    the target surely keyed KEY_SLACK_MS sooner or later too, and how much later it still would (Shot.window). (From
    2026-10-26 to the review of that day every kind's turning shots started at TURN_KEY_MS, 120: a pig or a slime on her
    platform, picked only to 105 and 110 ms, had none.) From the ground: can_hit (in reach, nothing stopping its star
    or that of one the attack takes first; of `others`, every monster around her, `monsters` by default). From a
    jump: the attack's pick (the area about her feet, or the search's widening windows: air_takes) from where her feet
    are when the attack starts (physics.pose; _air_pick, AIR_MARGIN into it, at both ends of ATTACK_DECIDE_LAG_MS), of
    all of `others`, and its line not stopped (wall_in_way, from her feet then: stopped, the
    attack hits nothing); the monsters where `where` says they'll be (default: where they are). `sizes`: the kinds'
    bodies (localization.monsters.bodies: the map's game data; body_of's tables by default). Map units, y down."""
    others = monsters if others is None else others
    out: list[Shot] = []
    for m in monsters:
        if can_hit(platform, x, y, m, others=others):
            out.append(Shot(1 if m.x > x else -1, m, x=x))
    wanted = {id(m) for m in monsters}
    near = [m for m in others if abs(m.x - x) <= STAR_RANGE + 150
            and -LOOK_UP - physics.JUMP_HEIGHT <= m.y - y <= LOOK_DOWN + 20]
    if not kinds or not any(id(m) in wanted for m in near):
        return out
    places = {m.platform: platform if m.platform == platform.id else platform.peer(m.platform) for m in near}
    body = {id(m): body_of(m.kind, sizes) for m in near}

    def placed(t: float) -> Placed:
        rows = []
        for m in near:
            mx = m.x if where is None else where(m, t)
            q = places.get(m.platform)
            my = m.y if q is None or where is None else q.y_at(min(max(mx, q.left), q.right))
            rows.append((m, mx, my, *body[id(m)]))
        return rows

    still = placed(0.0) if where is None else None
    lag, decide = physics.JUMP_LAG, physics.ATTACK_DECIDE_LAG_MS / 1000
    step = KEY_STEP_MS // 2
    turns = {m.kind: turn_key_ms(sizes[m.kind].top if sizes and m.kind in sizes else MONSTER_TOP.get(m.kind, PIG[0]))
             for m in near if id(m) in wanted}  # each kind's soonest attack key turning in the air
    turn0 = min(turns.values(), default=TURN_KEY_MS)
    for side in (-1, 1):
        if not any(id(m) in wanted and (m.x - x) * side >= -60 for m in near):
            continue  # (a jump forward can carry her past one 60 behind at most before its attack starts)
        for kind in kinds:
            way = {"up": 0, "forward": side, "back": -side}[kind]
            j = physics.jump_of(kind, way)
            landing = physics.jump_landing(model, platform, x, j)
            alts = tuple((v, physics.jump_landing(model, platform, x, v)) for v in physics.jump_variants(j))
            picks: dict[int, tuple] = {}

            def pick(k: int, j=j, landing=landing, picks=picks, side=side) -> tuple:
                if k not in picks:
                    t0 = k / 1000 + lag
                    p0 = physics.pose(model, platform, x, j, t0, landing)
                    p1 = physics.pose(model, platform, x, j, t0 + decide, landing)
                    m = None
                    if p0[2] and p1[2]:
                        a = still or placed(j.lead_s + t0)
                        b = still or placed(j.lead_s + t0 + decide)
                        m = _air_pick(p0[0], p0[1], side, a, AIR_MARGIN)
                        if m is not None and m is not _air_pick(p1[0], p1[1], side, b, AIR_MARGIN):
                            m = None
                        if m is not None:
                            mx = next(r[1] for r in a if r[0] is m)
                            target = platform if m.platform == platform.id else platform.peer(m.platform)
                            if wall_in_way(platform, p0[0], p0[1], mx, 0.0, target, None, body[id(m)]) is not None:
                                m = None
                    picks[k] = (m, p0)
                return picks[k]

            starts = (((KEY_FIRST_MS, False), (turn0, True)) if kind == "up"
                      else ((turn0, True),) if kind == "back" else ((KEY_FIRST_MS, False),))
            for start, turning in starts:
                seen: set[int] = set()
                k = start
                while k <= KEY_LAST_MS:
                    m, p0 = pick(k)
                    if not p0[2]:
                        break  # landed by then: the attack would start on the ground
                    if (m is not None and id(m) in wanted and id(m) not in seen
                            and (not turning or k >= turns.get(m.kind, TURN_KEY_MS))
                            and all(pick(k + d)[0] is m for d in (-KEY_SLACK_MS, KEY_SLACK_MS))):
                        hi = k
                        while (hi + step <= KEY_LAST_MS and pick(hi + step)[0] is m
                               and pick(hi + step + KEY_SLACK_MS)[0] is m):
                            hi += step
                        seen.add(id(m))
                        out.append(Shot(side, m, j, k, (k, hi), p0[0], y - p0[1], landing[0], landing[1], landing[2],
                                        turn=turning, alts=alts))
                    k += step
    return out


class Coverage:
    """What her attacks reach from one place (shots): by monster, every way to hit it, and the best of each kind."""

    def __init__(self, shot_list: list[Shot]):
        self.shots = shot_list
        self._by: dict[int, list[Shot]] = {}
        for s in shot_list:
            self._by.setdefault(id(s.target), []).append(s)

    def of(self, m: Monster) -> list[Shot]:
        """Every way to hit `m` from here (none: out of reach of any attack from here)."""
        return self._by.get(id(m), [])

    def hits(self, m: Monster, kinds: tuple[str, ...] | None = None) -> bool:
        """Whether some attack from here (of `kinds`: 'ground', 'up', 'forward', 'back'; any by default) hits `m`."""
        return any(kinds is None or s.kind in kinds for s in self.of(m))

    def ground(self, m: Monster) -> bool:
        """Whether a throw from the ground hits `m` (can_hit)."""
        return self.hits(m, ("ground",))

    def best(self, m: Monster, kinds: tuple[str, ...] | None = None) -> Shot | None:
        """The way to hit `m` that's soonest done: from the ground, else the jump with the soonest attack key (in
        place before forward before back, on a tie)."""
        order = {"ground": 0, "up": 1, "forward": 2, "back": 3}
        cands = [s for s in self.of(m) if kinds is None or s.kind in kinds]
        return min(cands, key=lambda s: (s.jump is not None, s.key_ms, order.get(s.kind, 9)), default=None)


def coverage(model, platform: Platform, x: float, y: float, monsters: list[Monster], **kw) -> Coverage:
    """shots from map (x, y) on `platform` at `monsters`, by monster (Coverage). Keywords: shots'."""
    return Coverage(shots(model, platform, x, y, monsters, **kw))
