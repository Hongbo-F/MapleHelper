"""In-game probes of what a star hits (`python -m gamelab probe`): each stands her where one open question
of attack.reach shows, throws only when a single monster can take the star, and logs every throw so the
question can be settled from the log alone (`score`).

The questions (attack.reach, gamelab/README.md, `QUESTIONS`): the target search's window, its top
(SEARCH_TOP0: the client's 44, as a port of the v83 client and meowdb have it, against 21 fitted to the
logs) and its floor by kind; the search between her level and 60 below her (against the old ramp her
decisions keep to there, RAMP_BELOW); whether an attack from a jump picks by the area or as from the ground,
from as high as she is (the v83 client's); what stops a star (her own foothold group's walls against none or
every wall, and the line: the v83 client's, which a floor stops too, against a level line SHOT_H above her
feet, LINE; that her own platform stops a star down past its end the user saw in game on 2026-10-21, and
that the step wall up from 4 stops one, so these probes now tell where and how high); and how far the claw
swipes, on her level (the client's stabs cut to 65 ahead against the box reach had until 2026-10-20, 90
ahead) and off it (the stabs, 15-30 above her feet, against the throw's swings, 17 below to 62 above). A
`Probe` is a place on Thicket Around the Beach II (the platform she stands on, her x, the way she throws),
the monsters it throws at (their kinds, their platforms, how far off sideways by their middle), and how she
throws: from the ground, a jump throw or a jump back, always with the regular attack.

A run (`run`) sends her to her place (the navigator, keeping clear of monsters) and keeps her there. After
every look she throws only when exactly one monster is in front of her (`in_front`: within FRONT by its
middle, from FRONT_UP above her feet to FRONT_DOWN below, all an attack could pick) and it's one the
probe throws at, its middle in the probe's range sideways (`crowd`: or with that many others in front
too; for a probe that asks it, `at_pick`, where it will be when the attack picks: `x_after`). After a
throw she neither moves nor throws until its star's evidence is in: every look up to EVIDENCE_S after the
star was due (longer while the target is hidden, up to HIDDEN_MAX_S: a hit monster flinches under its
damage numbers for 1-3 s). So one throw's evidence is never another's, and a monster hit by one has stopped
flinching (0.6 s, taller) when the next attack picks. She never throws while a monster of her platform is
closer than the probe's range (`in_her_way`: one of a kind it throws at closer than its kind's range, any other
closer than fight.IN_HER_WAY), whatever the crowd: one of a kind the probe doesn't throw at is fought off first
(behavior.fight.clear, her platform only); one it throws at is stepped away from (GIVE_WAY), or waited out where
she can't step. Standing where monsters spawn and throwing
at those of other platforms, she clears her platform first, and again whenever one of its monsters is in front
of her or in her way; a run that fights FIGHT_MAX_S without a throw stops (clears_first); knocked off her
place, she walks or goes back to it, the last walk the way she throws, so that no throw turns her (a turn's
tap walks her on). Nor does she walk or throw from a look a touch's knock may still be moving her in
(Hunter.knocked). Every monster in front counts, its star stopped or not (the client checks the line to its
pick alone, so a walled one still takes the attack). What her decisions take as stopping a star
(reach.WALLS_BLOCK: whether a target behind such a wall, or behind one, is thrown at, and the fights) is
switched for the run: off for the probes that measure it, so it throws at monsters behind such a wall too.

Each throw is a line of lab/probe_geometry.jsonl: the probe, how she threw (and the attack key's delay
after the jump key, `gap_ms`), whether the wall rule was on, where she stood and her foothold group, the
target, every monster in front at the throw (`front`, walls or not; each with the way it faced and its pose,
as the look read them) and at the first look after it, about when the attack picked its target (`pick`: that
look's frame is taken once the throw's key plan is done, about when the attack starts, ~65 ms after its
key), the rule's verdict on the target (`clear`), her EXP gains from the key to the end (`exp`), and for
every monster in front then its `evidence`: where it was, the way it faced and its pose at that look, its HP
bar before the star could land, every bar seen after (with when, from the star's due time), the looks it
wasn't seen in, and whether it was still there at the end.

The scorer (`score`) reads the log with the map's game data. What each monster showed (`shown`, `seen`): a
hit (its HP bar new or lower after the star was due, or killed: her EXP up and it gone), the claw's if it took
no more of its bar than 20 HP's worth (`claw_drop`), else a star's; none (seen with its bar as before); or
can't tell (hidden). A model of the attack (`Model`: reach's, `reach_model`, with one piece changed for each
answer a question weighs) gives every monster in front at the pick its chance of the claw's swipe and of a
star that gets there (`chances`: the claw's stance, one per attack; the search's first band, any monster in
it as likely; from a jump the area, or the search from as high as she is; then the line to the star's pick
alone, stopped and the star lost), by the positions at the pick (a monster seen walking as likely anywhere it
can have walked to within PICK_SLACK_S of the look that logged it: `placings`) and each kind's walking body.
Its likelihood (`likelihood`) is that of every monster's hit or none and every hit's size, with the two hit
rates fitted (a monster taken shows a hit with chance a, one not taken with chance f), as the research fitted
the shot log and the telemetry. Then:
- each question (`QUESTIONS`) over the throws of the probes that ask it: each answer's log-likelihood, on how
  many throws the two best differ, and the answer the throws decide on (DECIDES better than the next, on
  DECIDE_THROWS throws or more where they differ);
- the window's top and floor (SEARCH_TOP0, SEARCH_FLOOR0) fitted to every throw from the ground, and the
  line's height (SHOT_H, raised in a jump: reach.air_rise) to every throw, each by the line in force (reach.LINE:
  a throw whose star it stops took nothing, so its miss says nothing of the window);
- the claw probes' hits by how far off the target was, each the claw's or a star's.
"""

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from maplehelper.engine import LEFT, RIGHT
from maplehelper.game import COMBAT

from .. import directions, paths, telemetry
from ..attack import reach
from ..attack.stance import JUMP_BACK_SPAN, open_above, pct
from ..control import client
from ..control.actions import JB_RUN_MS, JB_THROW_MS, JB_TURN_MS, STAR_DELAY_MS, TAP_MS
from ..localization.mapdata import MapModel, MobBody
from ..localization.monsters import DEFAULT_SPEED, Monster, bodies, same, speeds
from ..localization.scene import Scene
from ..motion import body, physics
from .fight import GUARD_PATIENCE, GUARD_SECONDS, IN_HER_WAY, UNPLACED_LOOKS

if TYPE_CHECKING:
    from ..control.navigator import Navigator
    from .hunter import Hunter

FRONT = 420          # map units: a monster this far in front of her by its middle could take the star (a pig's
                     # body reaches into the search out to 434) ...
FRONT_UP = reach.LOOK_UP  # ... from this far above her feet (140: the search's highest window reaches 124, 101
                     # with the logs' lower top, and a jump's area 65 above her as high as 49 up in the jump) ...
FRONT_DOWN = 140     # ... to this far below (the search's lowest floor is 68 below her feet, and the tallest body
                     # here, an Orange Mushroom, 67). A look searches only LOOK_DOWN (75) below her, unless wide.
STAND_SLACK = 20     # she walks back to her place when she's farther off it than this (a jump back carries her 85)
APPROACH = 12        # ... or faces the other way: then she comes at it walking the way she throws, from at least
                     # this far behind it (a 75 ms hold: no tap short enough to be lost, under 50 ms)
EVIDENCE_S = 0.9     # s after a star was due that a throw's evidence is gathered (as fight._score_shots: a miss) ...
HIDDEN_MAX_S = 2.5   # ... and up to this long while its target is hidden (under its damage numbers: 1-3 s)
EARLY_S = 0.15       # a target seen this long before its star was due, or later, has been seen since (`over`)
BEFORE_HIT_S = (STAR_DELAY_MS - 50) / 1000  # s after the attack key a look still sees every bar as before the
                     # throw: its damage shows STAR_DELAY_MS (520) after the key at the earliest
SETTLED_S = 0.3      # a monster seen this long after the star was due with its bar as before wasn't hit
HURT = 0.02          # a bar this much lower than before is a hit
WAIT_S = 0.1         # s between looks while there's nothing to throw at
# Her own platform's monsters (`_keep_clear`). One closer than the probe's range is in her way (`in_her_way`):
# one of a kind it throws at, on a platform it throws at (hers: claw-edge, air), closer than its kind's range;
# any other closer than fight.IN_HER_WAY. While one is, she doesn't throw (`choose`): a throw then is no probe's.
# Any other is fought off (fight.clear on her platform, up to one kill; she steps away first if she can, as in
# any fight). One it throws at isn't (in the simulator, claw-edge on 21 with its pig and two slimes, the snails on
# 20 and 22, the game's errors, two others allowed in front, 300 s, seeds 1 and 3-6: fighting them off left 1-4
# throws where she made 30-40, and she was knocked off 21 into the basin on the way): she steps away from it
# (GIVE_WAY), to have it past its range's start, and where she can't, waits.
GIVE_WAY = True      # In the simulator (claw-edge, 40 throws, 21's, 20's and 22's monsters, the game's errors, crowd 2,
                     # seeds 1-8 in reach's world and in the old claw's): throwing with one in her way (as until the
                     # review of 2026-10-21), 40 throws took 232-387 s, but 3-11 of each run's 40 had one of 21's
                     # inside 95 in front of her, and those told nothing (the claw's answers differed on 0-5 of them
                     # all told; the rest alone decided the same: 13 runs right, 1 wrong, 2 undecided). With no
                     # throw while one is in her way, standing (False): 450-900 s, and 5 right, 1 wrong, 10
                     # undecided; stepping away: 720-1,500 s (3 of the 16 ran out of their 1,500 s at 33-39
                     # throws), and 11 right, 1 wrong, 4 undecided, her HP as either way (0.53-0.79 left). In game
                     # on 2026-10-21 a pig or slime was inside claw-edge's 95 at 288 of 533 looks
STEP_LEASH = 60      # ... how far from her place she steps away at most ...
END_ROOM = 90        # ... and never nearer than this to her platform's end (a touch's knock is 21-87 units)
STEP_PAST = 10       # ... to have it this far past its range's start
# A probe that throws at monsters on other platforms, standing on one where monsters spawn (clears_first:
# step-down from 2, step-up from 4, claw-down from 3), clears it first: its monsters come back in front of her
# and beside her (2's spawn points are 103 units right of step-down's x 50, and 9, 120 and 234 left of it), and
# one in front takes any star before a monster below does. In game on 2026-10-21 step-down fought 2's monsters
# one at a time as they came close for all of its 150 s, knocked off onto 4 three times, and threw none. So she
# clears it (fight.clear: every monster on it in reach) when she gets to her place, again whenever one of its
# monsters is in front of her or in her way, and waits for a throw in between ...
CLEAR_KILLS = 8      # ... a clear stopping after this many kills (2 has 4 spawn points) ...
CLEAR_S = 30.0       # ... or this long ...
CLEAR_PATIENCE = 1.0  # ... or once nothing on it has been in reach this long
FIGHT_MAX_S = 120.0  # a run stops once she's fought this long (clears and fights off) without a throw: where they
                     # come back faster than she kills them there is nothing to wait for. In the simulator
                     # (step-down for 300 s with 2's and 4's monsters, seeds 1-3): back 8 or 15 s after a kill,
                     # fighting them one at a time as they came close she threw once in the six runs and was
                     # touched 153-205 times a run; clearing, she threw none either and stopped after 126-150 s.
                     # Back after 30 s: one at a time 3 throws on each seed (93-119 touches), clearing 7, 4 and 5
                     # (52-79). How soon they're back in game isn't measured
STYLES = ("throw", "jump_throw", "jump_back")  # from the ground, a jump throw, a jump back (control.Actions)
# The scorer's grids: the first window's top and floor (above her feet), the line's height.
TOP0_GRID = range(0, 81)
FLOOR0_GRID = range(-30, 41)
SHOT_H_GRID = range(0, 101)
LOGS_TOP0 = 21       # the window's top the shot log and the telemetry fit (reach has the client's, 44: 28 + 16)
FITS = 1.92          # values within this much log-likelihood of the best fit (95%, one parameter)
# The two hit rates the likelihood fits are kept to what the logs have measured, or a model saying "not
# taken" would explain a run of hits as well as one saying "taken" (most probe throws have one monster):
A_MIN = 0.3          # a monster the attack takes shows a hit at least this often (72.5% of the shot log's 204
                     # throws hit; the telemetry missed about half the bars)
F_MAX = 0.2          # one it doesn't take shows one at most this often (the telemetry's monsters behind her,
                     # which no star can hit: 7.6% of 224)
RATE_FLOOR = 0.05    # and neither rate within this of 0 or 1: a hit or none is read wrong now and then too (a bar
                     # misread, a monster that walked out of where the look before the pick had it), so where every
                     # throw both answers agree on hits (as in the simulator), one odd none on a throw they differ on
                     # weighs at most log(0.95 / 0.05) = 2.9, under DECIDES (at 0.02, up to 3.9)
CLAW_SPLIT_HP = 20   # a hit that takes less of its target's HP than this was the claw's: 9-11 damage against a star's
                     # 30-70; of a pig's bar 16% (a swipe takes 7-9%, a star 23-55%), of a snail's 44% (a swipe 20-24%,
                     # a star kills it or leaves it under a third) ...
MAX_HP = {"pig": 128, "orange-mushroom": 142, "slime": 115, "snail": 45}  # ... by its kind's HP (game data:
                     # Mob/*.img info/maxHP; a kind not here is taken as a pig)
SIZE_SLIP = 0.05     # the likelihood has a hit's size read wrong this often (a swipe's taken for a star's or back):
                     # the bar is read to a few percent and a star's least damage is 3 times a swipe's most, so this
                     # only keeps one odd bar from deciding a question alone


def claw_drop(kind: str) -> float:
    """How much of the HP bar of a monster of `kind` a hit takes at most to be the claw's (CLAW_SPLIT_HP)."""
    return CLAW_SPLIT_HP / MAX_HP.get(kind, MAX_HP["pig"])


@dataclass(frozen=True)
class Probe:
    """Where she stands, what she throws at and how: one open question of attack.reach."""
    name: str
    asks: str               # the question it settles (`probe list`, the log)
    platform: int           # she stands on this platform ...
    x: float                # ... at this map x ...
    side: str               # ... and throws this way
    at: tuple[int, ...]     # at monsters on these platforms ...
    kinds: tuple[str, ...]  # ... of these kinds ...
    dx: tuple[float, float]  # ... their middle this far off sideways (map units, from her middle)
    by_kind: tuple = ()     # (kind, (lo, hi)): a kind whose range differs from dx
    style: str = "throw"    # how she throws (STYLES)
    throws: int = 20
    wide: bool = False      # looks over the whole screen: the targets stand above what a look's band searches
    crowd: int = 0          # it also throws with this many others in front (`choose`; `probe run --crowd`), never
                            # with one of her platform's in her way (in_her_way)
    walls: bool = True      # whether her decisions keep to the wall rule during the run (off: it's measured)
    at_pick: bool = False   # the range sideways is where a monster will be when the attack picks (`x_after`),
                            # not where the look has it
    expect: str = ""        # what each model predicts
    source: str = ""        # the research's name for it

    def range_of(self, kind: str) -> tuple[float, float]:
        """How far off sideways (by its middle, either way) a monster of `kind` is thrown at."""
        return dict(self.by_kind).get(kind, self.dx)


PROBES = {p.name: p for p in (
    Probe("top-60", "the search window's top (reach.SEARCH_TOP0)", 13, -14.0, LEFT, (12,),
          ("pig", "orange-mushroom"), (110, 200), throws=30, source="geometry P1",
          expect="60 above her, with the client's top (44, reach's) a pig is hit from 111 units and an Orange "
                 "Mushroom from 113; with the logs' fitted one (21), from 191 and 193"),
    Probe("top-120", "the window's top, two platforms up", 15, 76.0, LEFT, (12,), ("pig", "orange-mushroom"),
          (340, 420), wide=True, source="geometry P2",
          expect="120 above her, with the client's top (44) hits from 351 units (an Orange Mushroom from 353); with "
                 "the logs' (21) none"),
    Probe("floor", "the window's floor by kind (reach.SEARCH_FLOOR0)", 14, 824.0, RIGHT, (17,),
          ("orange-mushroom", "pig", "slime"), (90, 200),
          by_kind=(("orange-mushroom", (90, 150)), ("pig", (120, 190)), ("slime", (130, 200))), throws=40,
          source="geometry P3",
          expect="60 below her, by either top: an Orange Mushroom hit from 113 units, a pig from 151 (171 standing), "
                 "a slime from 177 (closer while it hops)"),
    # Facing right from 22 nothing else is in front (24, 60 below, has no spawns; facing left, 21's pigs and
    # slimes 60 above took the stars). A snail closer than 33 touches her, and a touch's knockback (21-87 units)
    # from the right can take her off 22's left end into the basin, so she stands 69 from it and the range
    # starts past touching (--dx 15-30 asks for the rest).
    Probe("air", "what an attack from a jump picks: the area or the search", 22, 30.0, RIGHT, (22,), ("snail",),
          (35, 85), style="jump_throw", throws=40, source="geometry P4",
          expect="by the area (reach.in_area, reach's), hits at every distance; by the v83 client's, the claw and the "
                 "search as from the ground but from 21-26 up, none at all (the stabs' boxes 36-56 above the "
                 "ground, and the search's windows first take a snail's 26 from 86-106 off)"),
    Probe("other-wall", "whether the side wall of another block stops a star", 19, 104.0, RIGHT, (18,),
          ("pig", "orange-mushroom"), (200, 260), source="geometry P5",
          expect="hits as on open ground: the wall of 18 that the line crosses is another foothold group's; if every "
                 "wall stopped a star, none"),
    # Both lines alike here: the v83 line to a monster on 2 ends at its feet, 60 up, so it meets the step wall
    # below its top corner whatever the distance, as the level line does; only a body lifted off its feet right
    # behind the step's edge (a pig's third walking frame lifts 4) would take it over the corner. That the step
    # wall stops a star from the ground the logs had, and the user saw (in game, 2026-10-21); what's left is how
    # high the line runs, which the jump throws and jump backs bracket.
    Probe("step-up", "how high the line a wall stops runs (reach.SHOT_H), at her own group's step wall",
          4, 500.0, LEFT, (2,), ("pig", "orange-mushroom"), (200, 400), throws=10, walls=False,
          source="terrain 1",
          expect="past the 60-unit step wall up to 2, by either line: from the ground (28 up) and from jump throws "
                 "(54 up) no hits, from jump backs (72 up) hits; if her group's walls didn't stop a star, hits from "
                 "all three; again with --x 260, and --style jump_throw, jump_back"),
    # From 2 at x 50, 169 units from its right end: the v83 line goes down through 2's floor before its end,
    # unless the monster is far enough out (a walking pig from 254 off, a slime from 284). That 2's floor stops
    # some is the user's (in game, 2026-10-21); where it lets them through isn't, so the range runs about 55 past
    # it as well as short of it, and 40 throws (about half where the lines differ, as 20 were). An Orange
    # Mushroom (55 tall) is clear from 199, and 4's are all 201 or more off: none is thrown at.
    Probe("step-down", "where her own platform lets a star down past its end: the v83 line or a level one (reach.LINE)",
          2, 50.0, RIGHT, (4,), ("pig", "slime"), (200, 310), by_kind=(("slime", (230, 340)),), throws=40,
          walls=False, source="terrain 2, geometry P5's second",
          expect="by the v83 line (reach's) none at pigs under 254 off or slimes under 284 (2's own floor stops it), "
                 "hits beyond; by a level line hits at every distance; again with --x 195 --dx 160-250: hits by "
                 "either"),
    # 42 below her, between her level and the 60 the search was fitted to: 0 is 1's only such neighbour. Facing
    # left from 25 inside 0's left end, 1 runs on 336 units ahead of her (monsters on 1 can touch her on 0: their
    # bodies reach 38-55 up). An Orange Mushroom 42 below reaches into the stabs' boxes in its taller frames
    # (58-67: half its standing and a quarter of its walking), which the scorer, by its walking body (55), leaves
    # out: thrown at only past the boxes' reach (97), it tells the ramp from the search and nothing else.
    Probe("below-42", "the search between her level and 60 below her, against the old ramp (reach.RAMP_BELOW)",
          0, 1222.0, LEFT, (1,), ("pig", "orange-mushroom", "snail"), (60, 160),
          by_kind=(("orange-mushroom", (100, 160)), ("snail", (130, 220))), throws=30, source="review of the search",
          expect="42 below her, by the search a pig is hit from 71 units (111 standing), an Orange Mushroom at any "
                 "(33 by its body), a snail from 166; by the old ramp every kind from 126. By the swings' boxes "
                 "(17 below to 62 above her feet), a third of the throws at pigs within 99 are swiped. The shot log "
                 "missed pigs 124-125 off 3 times of 4 and an Orange Mushroom 91 off there"),
    # The claw's reach (reach.CLAW_FAR, CLAW_MARGIN), told by how much of the bar a hit takes (claw_drop). On her
    # level, in the middle of 21 (two slimes and a pig spawn there; 174 units to either end, past a knockback).
    # With two others in front: a snail on 20 (60 above) or 22 (60 below) is in front of her nearly always, and in
    # game on 2026-10-21 it threw none in 90 s, a pig or slime in range in front of her at 82 looks and never alone
    # (the snails on 20 and 22 both there at 81). Neither claw reaches 60 above or below her, so a snail there
    # takes a star the same by either. None of 21's in her way, though (choose: one inside 95 in front takes the
    # attack itself, and told nothing); she steps away from them (GIVE_WAY). In the simulator, with 20's, 21's and
    # 22's monsters and the game's errors, alone it threw none in 300 s; with two others, its 40 throws took
    # 12-25 minutes and decided reach's world on 5 of seeds 1-8 and the old claw's on 6, one the other way (the
    # old claw's seed 8, by 5.4), four undecided.
    Probe("claw-edge", "how far off the claw still swipes, by kind (reach.CLAW_FAR, CLAW_MARGIN, claw_edge)", 21,
          -225.0, RIGHT, (21,), ("pig", "slime"), (95, 160), throws=40, crowd=2, source="bodies D",
          expect="by the client's stab boxes (cut to 65 ahead, reach's) and the bodies, swipes (a small bar drop: "
                 "9-11 damage of a pig's 128 HP) up to 99 off a pig and 93 off a slime, stars from there (her "
                 "decisions count on a star from 111 and 105: claw_edge); by the box reach had until 2026-10-20 "
                 "(90 ahead, on her level only), swipes out to 124 and 118"),
    # The claw off her level: the swings' boxes reach 17 below her feet, into a walking pig's or an Orange
    # Mushroom's body 60 below her (46 and 55 tall; 67 in an Orange Mushroom's tallest frame), not a slime's (41);
    # the stabs' boxes, which the client tests, reach down to 15 above her feet, into neither in any frame. So it
    # tells the swings from the stabs, not the stabs from a claw kept to her level. From 35 inside 3's left end,
    # at 4: a pig on 4 is never closer than 81 (4 ends at 669; 69-85 as she stands within STAND_SLACK of her
    # place, at 704-720 in the simulator with the game's errors), and the swings reach one out to 99, so pigs are
    # thrown at only where they'll be within 95 when the attack picks (at_pick: one walking the game data's 125
    # units/s goes on 20 units or so from the look to the pick, and turns back at 4's end). Thrown at by where
    # the look had them (88-100 off, she at 716), a pig was within 97 at the pick on 13-37 of 40 throws with the
    # game's errors, and 40 throws decided the stabs on 1 of seeds 1-8, the swings on 2; by where they'd be, on
    # 32-39, and 80 throws decide both on all 8 (leads 5.8-7.0 and 5.2-8.3; without the errors 6.8-7.2 and 10.1
    # and up). Orange Mushrooms out to 125, where stars from 113 show how often a hit is seen.
    Probe("claw-down", "whether the claw swipes a monster off her level (reach.claw_takes: its stab boxes)", 3,
          716.0, LEFT, (4,), ("pig", "orange-mushroom"), (60, 95), by_kind=(("orange-mushroom", (60, 125)),),
          throws=80, at_pick=True, source="bodies D",
          expect="by the stab boxes (reach's), no hit at a pig (the search picks one 60 below from 151) and stars "
                 "at Orange Mushrooms from 113; by the swings' boxes, a third of the throws at both within 99 and 97 "
                 "swiped (small bar drops)"),
    # From 14 at x 706, 143 units from its right end (25 in from its left; no monster can touch her on 14): the
    # v83 line goes down through 14's floor before its end, unless the monster is far enough out (a walking pig
    # from 215 off, a slime from 241; an Orange Mushroom from 187, closer than any on 17 can be, so none). As
    # step-down's, the range runs past where it lets them through as well as short of it, with 40 throws.
    Probe("hanging-wall", "where her own block lets a star down past its end: the v83 line or a level one (reach.LINE)",
          14, 706.0, RIGHT, (17,), ("pig", "slime"), (180, 250), by_kind=(("slime", (190, 290)),), throws=40,
          walls=False, source="terrain 3",
          expect="by the v83 line (reach's) none at pigs under 215 off (on 17 they come from 189) or slimes under "
                 "241 (from 183), hits beyond; by a level line hits at every distance"),
)}


def _jump_back_out() -> float:
    """How far a jump back has carried her when its attack picks its target (JUMP_LAG after the attack key), by the
    engine with the median lags: control.actions' plan (the arrow away, the jump JB_RUN_MS after it, the turn at
    JB_TURN_MS), replayed on flat ground, the mean over the phases the game's steps can have against the keys."""
    events = [(0.0, "left", True), (JB_RUN_MS / 1000, "jump", True), (JB_TURN_MS / 1000, "left", False),
              (JB_TURN_MS / 1000, "right", True), ((JB_TURN_MS + TAP_MS) / 1000, "right", False)]
    picks = (JB_THROW_MS + physics.JUMP_LAG * 1000) / 1000
    out = [-body.replay(events, picks, physics.SHOE, physics.LAGS, phase=phase).points[-1][1]
           for phase in body.phases()]
    return sum(out) / len(out)


# A jump back's target, as far off as a probe's range reaches at the look, is this much farther off when the attack
# picks (27 units, with the jump keyed 150 ms after the arrow; 25 at 130), and it may walk on meanwhile: from 4
# at 2 (step-up, 200-400 off) jump backs at Orange Mushrooms 387-394 off at the look were 434-445 off at the
# pick, past a star's reach (an Orange Mushroom's to 432 by its middle), and missed for that, not for the wall.
# So a jump back's range ends this much nearer.
JUMP_BACK_OUT = _jump_back_out()


def adjust(probe: Probe, *, x: float | None = None, dx: tuple[float, float] | None = None, style: str | None = None,
           throws: int | None = None, walls: bool | None = None) -> Probe:
    """`probe` with what's given changed (a range sideways given applies to every kind)."""
    changes = {k: v for k, v in (("x", x), ("style", style), ("throws", throws), ("walls", walls)) if v is not None}
    if dx is not None:
        changes.update(dx=tuple(dx), by_kind=())
    return replace(probe, **changes)


def problems(model: MapModel, probe: Probe) -> list[str]:
    """Why `probe` can't run on `model` as it is: her place off her platform, a jump under a platform (she'd
    land on it), no room behind her for a jump back."""
    out = []
    if probe.style not in STYLES:
        out.append(f"no such way to throw: {probe.style} (one of {', '.join(STYLES)})")
    if not 0 <= probe.platform < len(model.platforms):
        return out + [f"no platform {probe.platform} on map {model.id}"]
    p = model.platform(probe.platform)
    if not p.left <= probe.x <= p.right:
        out.append(f"x {probe.x:.0f} isn't on platform {p.id} ({p.left:.0f} to {p.right:.0f})")
    if probe.style != "throw" and not open_above(model, p, probe.x - STAND_SLACK, probe.x + STAND_SLACK):
        out.append(f"a platform overhead at x {probe.x:.0f}: a jump there lands on it")
    behind = (p.right - probe.x) if probe.side == LEFT else (probe.x - p.left)
    if probe.style == "jump_back" and behind < JUMP_BACK_SPAN + reach.EDGE_MARGIN:
        out.append(f"only {behind:.0f} units behind her at x {probe.x:.0f}: a jump back needs "
                   f"{JUMP_BACK_SPAN + reach.EDGE_MARGIN:.0f}")
    lo, hi = probe.dx
    if not 0 <= lo <= hi:
        out.append(f"the range sideways {lo:.0f}-{hi:.0f} isn't one")
    return out


def gap_ms(style: str) -> float | None:
    """How long after the jump key an attack of `style` is keyed (None: from the ground)."""
    return {"jump_throw": physics.JUMP_ATTACK_MS, "jump_back": physics.JUMP_BACK_ATTACK_MS}.get(style)


def rise_of(style: str) -> float:
    """How high she is when an attack of `style` starts (reach.air_rise; 0 from the ground)."""
    gap = gap_ms(style)
    return 0.0 if gap is None else reach.air_rise(gap)


@dataclass
class ProbeReport:
    """How a probe run went (its own reading of each throw's evidence; `score` reads the log)."""
    probe: str
    throws: int = 0
    hits: int = 0       # throws whose target showed a hit
    misses: int = 0     # ... showed none
    unknown: int = 0    # ... couldn't be told (hidden throughout)
    stop: str = ""
    fights: int = 0     # fights with her own platform's monsters (clears and fights off: _keep_clear) ...
    fight_s: float = 0.0  # ... and how long they took

    def __str__(self):
        fought = f"; fought her platform's monsters {self.fights} times ({self.fight_s:.0f} s)" if self.fights else ""
        return (f"{self.probe}: {self.throws} throws, the target hit by {self.hits}, not by {self.misses}, "
                f"{self.unknown} can't tell{fought}; stopped: {self.stop}")


@dataclass
class _Guard:
    """Her own platform's monsters over a run (_keep_clear): whether she clears it first (clears_first) and has
    yet, where she stands while she gives way to one (GIVE_WAY), and how long she's fought since her last throw."""
    clears: bool
    place: float
    cleared: bool = False
    fought: float = 0.0


# --- a run ---------------------------------------------------------------------------------------------

def run(h: "Hunter", nav: "Navigator", probe: Probe, seconds: float = 900, crowd: int | None = None) -> ProbeReport:
    """Run `probe` (see the module's docstring) until it has thrown probe.throws times, for at most
    `seconds`, or until her HP is too low (h.health_check) or she can't be placed or get back to her place.
    `crowd`: how many others may be in front at a throw (None: the probe's own, Probe.crowd).
    Every throw is logged to lab/probe_geometry.jsonl. The wall switch (reach.WALLS_BLOCK) is the probe's for
    the run, and put back after it, however it ends."""
    if trouble := problems(h.model, probe):
        raise ValueError(f"probe {probe.name}: {'; '.join(trouble)}")
    before, reach.WALLS_BLOCK = reach.WALLS_BLOCK, probe.walls
    h.actions.skill = False  # the regular attack: what every log measured, and no MP
    try:
        return _run(h, nav, probe, seconds, probe.crowd if crowd is None else crowd)
    finally:
        reach.WALLS_BLOCK = before


def _run(h: "Hunter", nav: "Navigator", probe: Probe, seconds: float, crowd: int) -> ProbeReport:
    report = ProbeReport(probe.name)
    deadline = time.monotonic() + seconds
    flying: _Flying | None = None
    unplaced = 0
    guard = _Guard(clears_first(h.model, probe), probe.x)
    out = JUMP_BACK_OUT if probe.style == "jump_back" else 0.0
    ranges = ", ".join(f"{k} {lo:.0f}-{hi - out:.0f}" for k in probe.kinds for lo, hi in (probe.range_of(k),))
    h.log(f"probe {probe.name} ({probe.asks}): on {probe.platform} at x {probe.x:.0f}, {probe.style} {probe.side} "
          f"at monsters on {', '.join(map(str, probe.at))} ({ranges} units off); the wall rule "
          f"{'on' if reach.WALLS_BLOCK else 'off'}")
    while flying is not None or (report.throws < probe.throws and time.monotonic() < deadline):
        # Waiting for a lone monster, she can look for minutes without a key, and a key plan is where the alarm
        # is noticed (client.send). A look brings the game to the front: none once the player has seen the
        # Lie Detector.
        client.check_alarm()
        s = h.look(wide=probe.wide)
        if flying is not None and s.where is not None:
            flying.see(s, h.exp_gains)  # in mid-jump too (a jump back's pick, and while she's in the air)
        if s.where is None or s.platform is None:
            unplaced += 1
            if unplaced >= UNPLACED_LOOKS:
                report.stop = "lost track of her"
                break
            if flying is None:
                time.sleep(0.2)
            continue
        unplaced = 0
        if stop := h.health_check(s):
            report.stop = stop
            break
        if flying is not None:
            if not flying.over(s.t):
                continue
            _done(h, report, flying)
            flying = None
            if report.throws >= probe.throws or time.monotonic() >= deadline:
                break
        if h.knocked(s):  # a touch's knock may still be moving her: neither a walk nor a throw from this look
            time.sleep(WAIT_S)
            continue
        if guard.place != probe.x and s.platform.id == probe.platform and not in_her_way(probe, s, probe.x):
            guard.place = probe.x  # what she gave way to has gone from her place: back to it
        if (stop := _to_place(h, nav, replace(probe, x=guard.place), s)) is not None:
            if stop:
                report.stop = stop
                break
            continue
        if _keep_clear(h, probe, s, guard, report):
            if guard.fought >= FIGHT_MAX_S:
                report.stop = f"her platform's monsters kept her fighting {guard.fought:.0f} s without a throw"
                break
            continue
        # The attack key goes down as soon as it's chosen, and the attack picks JUMP_LAG after it (the look's frame
        # is a little older than its time: a few units more for a walking pig, inside the probes' margins).
        lead = time.monotonic() - s.t + physics.JUMP_LAG if probe.at_pick else 0.0
        target = choose(s, probe, crowd, h.model, lead)
        if target is None:
            time.sleep(WAIT_S)
            continue
        flying = _throw(h, s, probe, target)
        report.throws += 1
        guard.fought = 0.0
    if flying is not None:  # stopped while its evidence came in: kept as far as it got
        _done(h, report, flying)
    report.stop = report.stop or ("throws done" if report.throws >= probe.throws else "time limit")
    h.log(f"probe {report}")
    return report


def _to_place(h: "Hunter", nav: "Navigator", probe: Probe, s: Scene) -> str | None:
    """Get her back to her place, facing the way she throws: go there (the navigator) when she's on another
    platform; walk when she's more than STAND_SLACK off her x or faces the other way, the last walk toward the
    way she throws, from APPROACH behind her place at least. None: she's there; "": she moved (look again); else
    why she can't get there."""
    if s.platform.id != probe.platform:
        h.log(f"  on platform {s.platform.id}: going back to {probe.platform} at x {probe.x:.0f}")
        trip = nav.go(probe.platform, x=probe.x, scene=s)
        return "" if trip.result in ("arrived", "in position") else f"couldn't get to her place: {trip.result}"
    off = probe.x - s.where.x
    if abs(off) <= STAND_SLACK and h.actions.facing == probe.side:
        return None
    # Facing the way she throws, no throw turns her: the turn's arrow tap walks her on (7.4 units for 50 ms), and
    # within STAND_SLACK nothing walked her back. In the simulator from x 30 on 22 (`probe run air`), the jump
    # throws after the first came from x 38, 32 units from where the snail came back: inside the 33 it touches her
    # from, and a touch that wasn't evaded knocked her 45 back.
    ahead = off * directions.sign(probe.side)  # how far her place is ahead of her, the way she throws
    if ahead >= APPROACH:
        h.actions.walk(probe.side, ahead)
    else:  # past her place, at it or just short of it: back off first, to come at it the way she throws
        h.actions.walk(directions.opposite(probe.side), APPROACH - ahead)
    return ""


def clears_first(model: MapModel, probe: Probe) -> bool:
    """Whether a run of `probe` clears her platform before it throws (see CLEAR_KILLS): monsters spawn on it (the
    map's spawn points) and it throws at those of other platforms."""
    return probe.platform in model.spawn_platforms and probe.platform not in probe.at


def _thrown_at(probe: Probe, platform: int, m: Monster) -> bool:
    """Whether monster `m` (on `platform`) is of a kind and on a platform `probe` throws at."""
    return platform in probe.at and m.kind in probe.kinds


def in_her_way(probe: Probe, s: Scene, x: float | None = None) -> list[Monster]:
    """The monsters of look `s` on her platform closer to her (or to map x `x`) than the probe's range: one it
    throws at closer than its kind's range starts, any other closer than fight.IN_HER_WAY."""
    here = s.platform.id
    x = s.where.x if x is None else x
    return [m for m in s.monsters if m.platform == here
            and abs(m.x - x) < (probe.range_of(m.kind)[0] if _thrown_at(probe, here, m) else IN_HER_WAY)]


def _keep_clear(h: "Hunter", probe: Probe, s: Scene, guard: _Guard, report: ProbeReport) -> bool:
    """Her own platform's monsters, before a throw (see GIVE_WAY, CLEAR_KILLS): with clears_first, the platform
    cleared when she first gets to her place with any on it, and whenever one is in front of her or in her way;
    else one in her way fought off, or (GIVE_WAY) one she throws at stepped away from (where she can't step, it's
    waited out: `choose` throws at nothing while it's there). Whether she did any."""
    here = s.platform.id
    on = [m for m in s.monsters if m.platform == here]
    close = in_her_way(probe, s)
    if guard.clears and on:
        front = [m for m in in_front(s, probe.side) if m.platform == here]
        if guard.cleared and not front and not close:
            return False
        h.log(f"  clearing platform {here} first: {len(on)} on it, {len(front)} in front of her")
        _fight(h, guard, report, here, max_kills=CLEAR_KILLS, seconds=CLEAR_S, patience=CLEAR_PATIENCE)
        guard.cleared = True
        return True
    if not close:
        return False
    x = s.where.x
    m = min(close, key=lambda m: abs(m.x - x))
    if _thrown_at(probe, here, m):
        if GIVE_WAY and (to := _give_way(probe, s, m)) is not None:
            h.log(f"  {m.kind} close on her platform at {m.x - x:+.0f}: stepping away to x {to:.0f}")
            h.actions.walk(RIGHT if to > x else LEFT, abs(to - x))
            guard.place = to
            return True
        others = [o for o in close if not _thrown_at(probe, here, o)]
        if not others:
            return False
        m = min(others, key=lambda m: abs(m.x - x))
    h.log(f"  {m.kind} close on her platform at {m.x - x:+.0f}: fighting it off first")
    _fight(h, guard, report, here, max_kills=1, seconds=GUARD_SECONDS, patience=GUARD_PATIENCE)
    return True


def _give_way(probe: Probe, s: Scene, m: Monster) -> float | None:
    """Where she steps to away from monster `m`, one the probe throws at, to have it STEP_PAST its range's start:
    None if that's past STEP_LEASH from her place or END_ROOM from her platform's ends, too short a step to take,
    or into another monster's way."""
    p, x = s.platform, s.where.x
    toward = 1 if m.x > x else -1
    to = m.x - toward * (probe.range_of(m.kind)[0] + STEP_PAST)
    lo, hi = max(probe.x - STEP_LEASH, p.left + END_ROOM), min(probe.x + STEP_LEASH, p.right - END_ROOM)
    if not lo <= to <= hi or abs(to - x) < APPROACH:
        return None
    if any(o is not m for o in in_her_way(probe, s, to)):
        return None
    return to


def _fight(h: "Hunter", guard: _Guard, report: ProbeReport, here: int, **kw) -> None:
    """A fight on her platform only, where she stands (fight.clear), timed for the report and FIGHT_MAX_S; not
    going on to finish the monsters she has hurt (finish_wounds off): those are the probe's own targets, and its
    stars at them would be outside the probe's log."""
    began = time.monotonic()
    h.clear(None, ground={here}, approach=False, finish_wounds=False, **kw)
    took = time.monotonic() - began
    report.fights += 1
    report.fight_s += took
    guard.fought += took


def in_front(s: Scene, side: str) -> list[Monster]:
    """The monsters of look `s` that could take the attack of a star she throws `side`: in front of her (a body
    reaching in front of her counts) within FRONT by their middle, from FRONT_UP above her feet to FRONT_DOWN
    below. Whatever stops their stars: the client checks the line to its pick alone, so one behind a wall still
    takes the attack if it comes first, and the star stops at the wall."""
    x, y, d = s.where.x, s.where.y, directions.sign(side)
    return [m for m in s.monsters
            if -reach.body_of(m.kind)[1] < (m.x - x) * d <= FRONT and -FRONT_UP <= m.y - y <= FRONT_DOWN]


def x_after(m: Monster, lead: float, model: MapModel | None = None) -> float:
    """Where monster `m` (its middle) is `lead` s after the look that saw it: seen walking (its pose 'move', its
    way read), its kind's walking speed (monsters.speeds) that long the way it faces, turning back at the ends of
    its platform's walk (its half width inside them, on `model`); else where the look has it."""
    if lead <= 0 or m.pose != "move" or not m.facing:
        return m.x
    walk = speeds(model).get(m.kind, DEFAULT_SPEED) * lead
    if model is None or m.platform is None or not 0 <= m.platform < len(model.platforms):
        return m.x + m.facing * walk
    p, half = model.platform(m.platform), reach.body_of(m.kind)[1]
    lo, hi = p.left + half, max(p.left + half, p.right - half)
    if hi <= lo:
        return lo
    u = (min(max(m.x, lo), hi) - lo + m.facing * walk) % (2 * (hi - lo))  # as it walks there and back
    return lo + (u if u <= hi - lo else 2 * (hi - lo) - u)


def choose(s: Scene, probe: Probe, crowd: int = 0, model: MapModel | None = None,
           lead: float = 0.0) -> Monster | None:
    """What to throw at in look `s`, if anything: the nearest monster in front of her (in_front) that the probe
    throws at (its kind, its platform, its middle in range sideways: for a jump back, its far end JUMP_BACK_OUT
    nearer; with `probe.at_pick`, where it will be when the attack picks, `lead` s after the look: x_after on
    `model`), when it's the only one in front (or with at most `crowd` others). While her decisions keep to what
    stops a star (reach.WALLS_BLOCK), not one whose star is stopped (reach.wall_to), nor one behind another in
    front that the attack takes first whose star is (reach.shielded: with a crowd). Never while a monster of her
    platform is in her way (in_her_way), whatever the crowd: one that close in front takes the attack (the claw's
    or a star) before any in range, one behind her touches her, and the throw is no probe's."""
    if in_her_way(probe, s):
        return None
    front = in_front(s, probe.side)
    x, y, rise = s.where.x, s.where.y, rise_of(probe.style)
    out = JUMP_BACK_OUT if probe.style == "jump_back" else 0.0  # she's that much farther off when it picks

    def off(m: Monster) -> float:
        return abs((x_after(m, lead, model) if probe.at_pick else m.x) - x)
    fits = [m for m in front if m.kind in probe.kinds and m.platform in probe.at
            and probe.range_of(m.kind)[0] <= off(m) <= probe.range_of(m.kind)[1] - out
            and reach.wall_to(s.platform, x, y, m, rise) is None
            and reach.shielded(s.platform, x, y, m, front, rise) is None]
    if not fits or len(front) > 1 + crowd:
        return None
    return min(fits, key=lambda m: abs(m.x - x))


def _throw(h: "Hunter", s: Scene, probe: Probe, target: Monster) -> "_Flying":
    """Throw at `target` as the probe does, and start gathering the evidence."""
    row = _row(h, s, probe, target)
    exp0, keyed = h.exp_gains, []

    def on_throw(when: float, side: str) -> None:
        keyed.append(when)
    if telemetry.recording():
        telemetry.emit("stance", act=probe.style, text=f"probe {probe.name}: {probe.style} {probe.side}",
                       aimed=[{"kind": target.kind, "x": round(target.x, 1), "y": round(target.y, 1)}])
    h.actions.throw_listeners.append(on_throw)
    try:
        getattr(h.actions, probe.style)(probe.side)
    finally:
        h.actions.throw_listeners.remove(on_throw)
    h.report.throws += 1
    key = keyed[0] if keyed else time.monotonic()
    flight = abs(target.x - s.where.x) + (JUMP_BACK_SPAN if probe.style == "jump_back" else 0)
    due = key + COMBAT.attack_release_ms / 1000 + flight / reach.STAR_SPEED
    h.log(f"  {probe.style} {probe.side} at {target.kind} on {target.platform}: {target.x - s.where.x:+.0f},"
          f"{target.y - s.where.y:+.0f} (bar {pct(target.bar)}), {len(row['front'])} in front")
    return _Flying(row, s, target, key, due, exp0)


def _monster(m: Monster, target: bool = False) -> dict:
    """A monster as the probe log keeps it (with the way it faced and its pose, as the look read them)."""
    return {"kind": m.kind, "x": round(m.x, 1), "y": round(m.y, 1), "platform": m.platform, "bar": m.bar,
            "facing": m.facing, "pose": m.pose, **({"target": True} if target else {})}


def _row(h: "Hunter", s: Scene, probe: Probe, target: Monster) -> dict:
    """A throw's line in the probe log, as far as the look before it tells (the rest comes with _Flying)."""
    here, x, y = s.platform, s.where.x, s.where.y
    on = h.model.platform(target.platform) if target.platform is not None else None
    front = in_front(s, probe.side)
    return {"probe": probe.name, "style": probe.style, "gap_ms": gap_ms(probe.style), "walls_block": reach.WALLS_BLOCK,
            "map": h.model.id, "side": probe.side, "x": round(x, 1), "y": round(y, 1), "her": here.id,
            "her_group": list(here.group) if here.group is not None else None,
            "target": _monster(target, True), "dx": round(target.x - x, 1), "dy": round(target.y - y, 1),
            "clear": reach.wall_between(here, x, y, target.x, rise_of(probe.style), target=on, ty=target.y,
                                        body=reach.body_of(target.kind)) is None,
            "front": [_monster(m, m is target) for m in front], "time": time.strftime("%Y-%m-%d %H:%M:%S")}


class _Flying:
    """A probe throw whose evidence is coming in: every monster in front at the throw, and at the first look
    after it (the pick), followed look by look until EVIDENCE_S after its star was due (HIDDEN_MAX_S while
    the target is hidden)."""

    def __init__(self, row: dict, s: Scene, target: Monster, key: float, due: float, exp0: int):
        self.row, self.side, self.key, self.due, self.exp0 = row, row["side"], key, due, exp0
        self.exp = 0
        self.looks = 0  # looks from when the star could have landed on (BEFORE_HIT_S after the key)
        self.picked = False
        self.entries = [self._entry(m, m is target) for m in in_front(s, self.side)]
        if not any(e["target"] for e in self.entries):
            self.entries.append(self._entry(target, True))

    @staticmethod
    def _entry(m: Monster, target: bool) -> dict:
        return {"m": m, "kind": m.kind, "target": target, "at_pick": False, "x": m.x, "y": m.y,
                "platform": m.platform, "facing": m.facing, "pose": m.pose, "bar_before": m.bar, "after": [],
                "unseen": [], "seen_last": True}

    def see(self, s: Scene, exp_gains: int) -> None:
        """A look after the throw that placed her (in the air too; `exp_gains`: the hunter's EXP gains so far).
        The first is the pick: where each monster was then. A look before the star can have landed (BEFORE_HIT_S
        after the key) has each one's bar as before it; no other star of hers is in flight, as the next throw
        waits for this one's evidence. Every look after that is evidence, timed from when the star was due."""
        self.exp = exp_gains - self.exp0
        dt = round(s.t - self.due, 2)
        early = s.t - self.key < BEFORE_HIT_S
        claimed: set[int] = set()
        for e in self.entries:
            near = [o for o in s.monsters if id(o) not in claimed and same(o, e["m"])]
            again = min(near, key=lambda o: abs(o.x - e["m"].x)) if near else None
            e["seen_last"] = again is not None
            if again is not None:
                claimed.add(id(again))
                e["m"] = again
                if not self.picked:
                    e.update(at_pick=True, x=again.x, y=again.y, platform=again.platform, facing=again.facing,
                             pose=again.pose)
                if early:
                    e["bar_before"] = again.bar
                else:
                    e["after"].append([dt, again.bar])
            elif not early:
                e["unseen"].append(dt)
        if not early:
            self.looks += 1
        if self.picked:
            return
        self.picked = True  # the first look after the throw: about when the attack picked its target
        front = in_front(s, self.side)
        for m in front:
            if id(m) not in claimed:  # in front now, not at the throw: it may have taken the star
                e = self._entry(m, False)
                e["at_pick"] = True
                if not early:
                    e["after"].append([dt, m.bar])
                self.entries.append(e)
        self.row["pick"] = {"x": round(s.where.x, 1), "y": round(s.where.y, 1),
                            "platform": None if s.platform is None else s.platform.id,
                            "t": round(s.t - self.key, 3), "front": [_monster(m) for m in front]}

    def over(self, t: float) -> bool:
        """Whether the evidence is all in by `t`: EVIDENCE_S after the star was due, and by then the target
        seen since it was due, or (her EXP up) one of them gone: a kill fades out where it died for as long as
        its death animation runs, a snail 1.26 s. Else up to HIDDEN_MAX_S."""
        dt = t - self.due
        if dt < EVIDENCE_S:
            return False
        if dt >= HIDDEN_MAX_S:
            return True
        if self.exp > 0:
            return any(not e["seen_last"] for e in self.entries)
        return any(when >= -EARLY_S for when, _ in next(e for e in self.entries if e["target"])["after"])

    def finish(self) -> dict:
        """The throw's line for the log, its evidence in."""
        row = self.row
        row["due_s"] = round(self.due - self.key, 3)
        row["exp"], row["looks"] = self.exp, self.looks
        row["evidence"] = [{"kind": e["kind"], "x": round(e["x"], 1), "y": round(e["y"], 1), "platform": e["platform"],
                            "facing": e["facing"], "pose": e["pose"],
                            "target": e["target"], "at_pick": e["at_pick"], "bar_before": e["bar_before"],
                            "after": e["after"], "unseen": e["unseen"], "gone": not e["seen_last"]}
                           for e in self.entries]
        return row


def _done(h: "Hunter", report: ProbeReport, flying: _Flying) -> None:
    """A throw's evidence is in: logged, read, counted."""
    row = flying.finish()
    _log_row(row)
    told = dict((id(e), v) for e, v in outcomes(row))
    target = next(e for e in row["evidence"] if e["target"])
    seen = told[id(target)]
    others = sum(1 for e in row["evidence"] if not e["target"] and told[id(e)])
    if seen is None:
        report.unknown += 1
    elif seen:
        report.hits += 1
    else:
        report.misses += 1
    word = {True: "hit", False: "no hit", None: "can't tell (hidden)"}[seen]
    h.log(f"  -> {word}" + (f"; {others} other(s) hit" if others else "")
          + (f"; EXP +{row['exp']}" if row["exp"] else ""))
    telemetry.emit("event", kind="probe", text=f"probe {row['probe']}: {word}")


def _log_row(row: dict) -> None:
    try:
        with paths.PROBE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass


def listing() -> str:
    """The probes, for `probe list`."""
    lines = []
    for p in PROBES.values():
        ranges = ", ".join(f"{k} {lo:.0f}-{hi:.0f}" for k in p.kinds for lo, hi in (p.range_of(k),))
        lines += [f"{p.name}: {p.asks} ({p.source})",
                  f"    on {p.platform} at x {p.x:.0f}, {p.style} {p.side} at monsters on "
                  f"{', '.join(map(str, p.at))} ({ranges} units off), {p.throws} throws"
                  + ("" if p.walls else ", the wall rule off")
                  + (f", with up to {p.crowd} others in front" if p.crowd else ""),
                  f"    expect: {p.expect}"]
        lines += [f"    `probe score` weighs {q.key}: {' against '.join(name for name, _ in q.answers)}"
                  for q in QUESTIONS.values() if p.name in q.probes]
    return "\n".join(lines)


# --- the scorer ----------------------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    """The probe log's throws (none if there's no log)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shown(e: dict, exp: int, only_gone: bool) -> bool | None:
    """What a monster of a throw's evidence showed after the star was due: True a hit (its HP bar new or
    lower than before, or killed: her EXP up, and it the only one of them gone (`only_gone`)), False none
    (seen SETTLED_S or more after the star was due with its bar as before), None: can't tell (hidden)."""
    before = e["bar_before"]
    if any(b is not None and (before is None or b < before - HURT) for _, b in e["after"]):
        return True
    if exp and e["gone"] and only_gone:
        return True
    if any(dt >= SETTLED_S for dt, _ in e["after"]):
        return False
    return None


def outcomes(row: dict) -> list[tuple[dict, bool | None]]:
    """What each monster of a throw's evidence showed (`shown`)."""
    gone = [e for e in row.get("evidence", ()) if e["gone"]]
    return [(e, shown(e, row.get("exp", 0), len(gone) == 1)) for e in row.get("evidence", ())]


def _her(row: dict) -> tuple[float, float]:
    """Her map x when the attack picked its target (the first look after the throw; for a jump back she's
    on her way by then), and her feet's y on the ground she threw from."""
    pick = row.get("pick")
    return (pick["x"] if pick else row["x"]), row["y"]


# The look that logs where each monster was at the pick (`pick`: the first after the key) isn't taken when the
# attack picks, JUMP_LAG after its key, and by how much isn't measured in game. By the looks' own times it's
# pick.t - JUMP_LAG after it (33 ms with the simulator's looks), but a look's frame is older than its time: in
# game its time is when its reading ends, after the frame was grabbed (eyes.look_at), and the simulator draws its
# monsters LOOK_LAG (40 ms) before it; with the key going down START_LATENCY (10 ms) after the plan is handed
# over, there the look shows them 20 ms before the pick. A pig walking 125 units/s is a few units off either
# way, which at the edge of a box decides what the attack took: claw-edge in the old claw's world on seed 2 (40
# throws), pigs walking away seen 120-125 off, inside the old box (to 124), were 125-128 off at the pick, and the
# stars they took leaned the scorer 2.9 toward the stabs; moved back 4 units by the looks' times, 14.7 toward
# them. So the pick is taken as likely at any of PICK_STEPS moments from PICK_SLACK_S before that look to as
# long after it: 5.3 the old box's way there, and none of test_probe's runs on seeds 1-8 leans the wrong way.
PICK_SLACK_S = 0.04
PICK_STEPS = 5


def _walked(e: dict, dt: float, walk: dict[str, float], model: MapModel | None) -> float:
    """Where monster `e` of a throw's evidence was `dt` after the look that logged it, by its middle: seen walking
    (its pose 'move', its way read), its kind's walking speed (`walk`) that long the way it faced, kept to its
    platform's walk (its half width inside the ends, on `model`); else (standing, flinching, or neither read: a log
    from before they were kept) where that look had it."""
    if not dt or e.get("pose") != "move" or not e.get("facing"):
        return e["x"]
    x = e["x"] + e["facing"] * walk.get(e["kind"], DEFAULT_SPEED) * dt
    if model is not None and e["platform"] is not None and 0 <= e["platform"] < len(model.platforms):
        p, half = model.platform(e["platform"]), reach.body_of(e["kind"])[1]
        x = min(max(x, p.left + half), max(p.left + half, p.right - half))
    return x


def placings(row: dict, model: MapModel | None = None) -> list[list[dict]]:
    """A throw's evidence, each monster where it was at each moment the attack may have picked its target
    (PICK_STEPS from PICK_SLACK_S before the look after the key to as long after it, `_walked`); one placing when
    none of them was seen walking."""
    evidence, walk = list(row.get("evidence", ())), speeds(model)
    if PICK_STEPS < 2 or not any(e.get("pose") == "move" and e.get("facing") for e in evidence):
        return [evidence]
    steps = [PICK_SLACK_S * (2 * i / (PICK_STEPS - 1) - 1) for i in range(PICK_STEPS)]
    return [[{**e, "x": _walked(e, dt, walk, model)} for e in evidence] for dt in steps]


def _rise(row: dict) -> float:
    return 0.0 if row["gap_ms"] is None else reach.air_rise(row["gap_ms"])


# The claw's models the scorer weighs (reach.Claw): the client's close stances, the stabs cut to 65 ahead
# (reach's: STABS); one box from her to 90 ahead and from 17 below to 62 above her feet, on her level only
# (reach's until 2026-10-20: the swings' and stabO1's boxes together, taken as 90 ahead); and the throw's own
# swings, swingO1-O3 (the game data's boxes, Character/Afterimage/swordOL.img/0, as she faces left), cut to 65,
# on her level or not: the cross-validation's synthesis read RustMS's whack so, where its port gives a claw its
# close list, the stabs. On her level the swings reach as far as the stabs; off it, 17 below her feet to 62
# above, where the stabs keep to 15-30 above.
STABS = reach.CLAW
OLD_CLAW = reach.Claw((("swing", ((-90, -62), (0, 17))),), 90, level=reach.LEVEL)
SWINGS = reach.Claw((("swingO1", ((-88, -62), (-18, -6))), ("swingO2", ((-73, -50), (5, 17))),
                     ("swingO3", ((-102, -50), (-22, -10)))), reach.CLAW_FAR)
CLIENT_TOP0 = reach.SHOT_H + reach.SEARCH_START // reach.SEARCH_SPREAD  # 44: the client's first window's top


@dataclass(frozen=True)
class Model:
    """What the scorer takes a throw's attack to do, a piece for each open question of attack.reach: the
    search's first window (its top and floor above her feet, `top0` and `floor0`: SEARCH_TOP0, SEARCH_FLOOR0),
    the claw (reach.Claw), which line terrain stops (`line`: as reach.LINE) and whose walls (`walls`: 'group',
    her own foothold group's, as reach has it; 'none'; or 'every' wall, any group's, across the level line),
    the line's height (`shot_h`: SHOT_H), whether none between her level and 60 below her is picked closer
    than the old ramp (`ramp`: reach.ramp_dx), and how an attack from a jump picks (`air`: 'area', the area
    fixed to her feet, reach.in_area, as reach has it; or 'v83', as from the ground from as high as she is,
    as the simulator's world.AIR_PICK has them). reach_model() is reach's, as it is now."""
    top0: float
    floor0: float
    claw: reach.Claw
    line: str
    shot_h: float
    walls: str = "group"
    ramp: bool = False
    air: str = "area"


def reach_model() -> Model:
    """reach's model of a throw's attack, as reach has it now."""
    return Model(reach.SEARCH_TOP0, reach.SEARCH_FLOOR0, reach.CLAW, reach.LINE, reach.SHOT_H)


def gets_to(row: dict, e: dict, model: MapModel, m: Model, body: dict[str, MobBody]) -> bool:
    """Whether a star of throw `row` gets to monster `e` of its evidence, by `m`'s terrain: no wall stops it
    ('none'); none of the map's walls crosses the level line between her and it ('every'); or (as reach has
    it, wall_between) none of her own foothold group's walls, gone into at their outer face, nor by the v83
    line a floor, stops the line from her to it (`m.line`, `m.shot_h` above her feet, raised in a jump)."""
    if m.walls == "none":
        return True
    x, y = _her(row)
    if m.walls == "every":
        line = y - _rise(row) - m.shot_h
        lo, hi = sorted((x, e["x"]))
        return not any(w.x1 == w.x2 and lo < w.x1 < hi and min(w.y1, w.y2) <= line < max(w.y1, w.y2)
                       for w in model.walls)
    on = model.platform(e["platform"]) if e["platform"] is not None else None
    return reach.wall_between(model.platform(row["her"]), x, y, e["x"], _rise(row), target=on, shot_h=m.shot_h,
                              ty=e["y"], body=reach.body_of(e["kind"], body), line=m.line) is None


def _rises(row: dict) -> tuple[float, ...]:
    """How high she is when a throw's attack picks its target: 0 from the ground; from a jump, as high as she is
    JUMP_LAG after its key or ATTACK_DECIDE_LAG_MS later (physics; reach.air_rise is the later: 21 and 26 units
    for a jump throw keyed 40 ms after the jump key, as the simulator's world and her decisions have it)."""
    gap = row["gap_ms"]
    if gap is None:
        return (0.0,)
    return physics.jump_rise(gap + physics.JUMP_LAG * 1000), reach.air_rise(gap)


def chances(row: dict, model: MapModel, body: dict[str, MobBody], m: Model) -> list[tuple[dict, float, float]]:
    """For each monster of a throw's evidence, the chance `m` gives its attack of swiping it with the claw and
    of throwing it a star that gets to it: (monster, swipe, star). From the ground, as reach.attack_chances
    has it: each claw stance (`m.claw`, one per attack) swipes the nearest whose body reaches into its box,
    else the star goes to one of the search's first band, any of them as likely (`m.top0`, `m.floor0`; with
    `m.ramp`, none below her closer by its middle than the old ramp). From a jump (`m.air`), by the area: the
    star goes to the nearest whose body reaches into it (reach.in_area, as high as she is then, by the least
    height of its kind standing or walking); or ('v83') the claw and the search as from the ground, from as
    high as she is then; each as likely from either height the attack may start at (`_rises`), and at each
    moment it may have picked (`placings`). Then the line to the star's pick alone (gets_to): stopped, the star
    is lost, and no other monster takes it. Positions at the pick (hers `_her`), bodies as the fight goes by
    them (reach.body_of)."""
    entries = list(row.get("evidence", ()))
    x, y = _her(row)
    d, jump = directions.sign(row["side"]), row["gap_ms"] is not None

    def picks(rise: float, shapes: list[tuple]) -> list[tuple[float, float]]:
        if jump and m.air == "area":
            out = [(0.0, 0.0)] * len(entries)
            inside = [i for i, (e, (ahead, dy, top, half)) in enumerate(zip(entries, shapes))
                      if reach.in_area(ahead, dy, body[e["kind"]].top if e["kind"] in body else top, rise, half)]
            if inside:
                out[min(inside, key=lambda i: abs(shapes[i][0]))] = (0.0, 1.0)
            return out
        lifted = [(ahead, dy + rise, top, half) for ahead, dy, top, half in shapes]  # from as high as she is

        def band(b: tuple) -> int | None:
            if m.ramp and not jump and abs(b[0]) < reach.ramp_dx(b[1]):
                return None
            return reach.search_band(*reach.body_box(*b), top0=m.top0, floor0=m.floor0)
        return reach.attack_chances(lifted, m.claw, band)
    each = []
    for placed in placings(row, model):
        shapes = [((p["x"] - x) * d, p["y"] - y, *reach.body_of(p["kind"], body)) for p in placed]
        at = [picks(rise, shapes) for rise in _rises(row)]
        pairs = [tuple(sum(c) / len(at) for c in zip(*one)) for one in zip(*at)]
        each.append([(swipe, star if star and gets_to(row, p, model, m, body) else 0.0)
                     for p, (swipe, star) in zip(placed, pairs)])
    return [(e, *(sum(c) / len(each) for c in zip(*one))) for e, one in zip(entries, zip(*each))]


def seen(row: dict) -> list[tuple[dict, str | None]]:
    """What each monster of a throw's evidence showed (`shown`), as the likelihood reads it: 'none'; a hit, the
    claw's ('claw': it took no more of its bar than claw_drop) or a star's ('star'), or a kill ('hit': either); None,
    can't tell."""
    evidence = row.get("evidence", ())
    only = sum(1 for e in evidence if e["gone"]) == 1
    out = []
    for e, s in outcomes(row):
        if not s:
            out.append((e, None if s is None else "none"))
            continue
        d = drop(e, row.get("exp", 0), only)
        killed = bool(row.get("exp")) and e["gone"] and only
        out.append((e, "hit" if killed or d is None else "claw" if d < claw_drop(e["kind"]) else "star"))
    return out


Chance = Callable[[dict], list[tuple[dict, float, float]]]  # a throw -> (monster, swipe, star) for each


def _grid(lo: float, hi: float) -> list[float]:
    return [round(lo + i / 100, 2) for i in range(round((hi - lo) * 100) + 1)]


A_GRID = _grid(A_MIN, 1 - RATE_FLOOR)
F_GRID = _grid(RATE_FLOOR, F_MAX)


def _ll(groups: dict[tuple, int], a: float, f: float) -> float:
    """The log-likelihood of `groups` ((swipe, star, what it showed) -> monsters) with hit rates a and f."""
    out = 0.0
    for (swipe, star, s), n in groups.items():
        took = swipe + star
        hit = took * a + (1 - took) * f
        p = {"none": 1 - hit, "hit": hit,
             "claw": a * (swipe * (1 - SIZE_SLIP) + star * SIZE_SLIP) + (1 - took) * f / 2,
             "star": a * (swipe * SIZE_SLIP + star * (1 - SIZE_SLIP)) + (1 - took) * f / 2}[s]
        out += n * math.log(p)
    return out


def likelihood(rows: list[dict], chance: Chance) -> tuple[float, dict]:
    """How well a model of what each throw's attack takes (`chance`) fits what the monsters showed (`seen`): the
    log of the chance of every monster's hit or none, one the attack takes (by the claw or a star) showing a
    hit with chance a and one it doesn't with chance f, both fitted (a at least A_MIN, f at most F_MAX, neither
    within RATE_FLOOR of 0 or 1); and of every hit's size, a swipe's the claw's and a star's a star's but for
    SIZE_SLIP of them, one not taken's either as likely. Returns it, and how many monsters were told and hit."""
    groups: dict[tuple, int] = {}
    for row in rows:
        took = {id(e): (swipe, star) for e, swipe, star in chance(row)}
        for e, s in seen(row):
            if s is not None:
                key = (*(round(c, 9) for c in took.get(id(e), (0.0, 0.0))), s)
                groups[key] = groups.get(key, 0) + 1
    counts = {"told": sum(groups.values()), "hits": sum(n for (_, _, s), n in groups.items() if s != "none")}
    if not groups:
        return 0.0, counts
    if all(swipe + star in (0.0, 1.0) for swipe, star, _ in groups):  # each surely taken or not: in closed form
        rate = {}
        for t, lo, hi in ((True, A_MIN, 1 - RATE_FLOOR), (False, RATE_FLOOR, F_MAX)):
            n = [(s != "none", k) for (swipe, star, s), k in groups.items() if (swipe + star == 1) == t]
            total = sum(k for _, k in n)
            rate[t] = min(hi, max(lo, sum(k for h, k in n if h) / total)) if total else lo
        return _ll(groups, rate[True], rate[False]), counts
    return max(_ll(groups, a, f) for a in A_GRID for f in F_GRID), counts


@dataclass
class Fit:
    """A parameter fitted on a grid by likelihood: the best value, the values within FITS of it, and the
    log-likelihood at every value."""
    name: str
    lls: dict[float, float] = field(default_factory=dict)
    throws: int = 0
    decided: int = 0     # monsters whose hit or none was told
    hits: int = 0

    @property
    def best(self) -> float:
        top = max(self.lls.values())
        at = [v for v, ll in self.lls.items() if ll >= top - 1e-9]
        return at[len(at) // 2]  # the middle of a level stretch

    @property
    def fits(self) -> tuple[float, float]:
        top = max(self.lls.values())
        ok = [v for v, ll in self.lls.items() if ll >= top - FITS]
        return min(ok), max(ok)

    def holds(self, value: float) -> bool:
        """Whether `value` fits (within FITS of the best)."""
        return self.lls[value] >= max(self.lls.values()) - FITS


def _fit(name: str, rows: list[dict], grid, model_at: Callable[[float], Model], model: MapModel) -> Fit:
    """`name` fitted over `grid` (and the values reach has now, so they can be compared): reach's model with it
    set to each value (`model_at`)."""
    fit, body, counts = Fit(name, throws=len(rows)), bodies(model), {}
    for v in grid:
        m = model_at(v)
        fit.lls[v], counts = likelihood(rows, lambda r, m=m: chances(r, model, body, m))
    fit.decided, fit.hits = counts.get("told", 0), counts.get("hits", 0)
    return fit


def ground(rows: list[dict]) -> list[dict]:
    """The throws from the ground."""
    return [r for r in rows if r["gap_ms"] is None and r["style"] == "throw"]


def fit_top0(rows: list[dict], model: MapModel) -> Fit:
    """The search window's top (SEARCH_TOP0) fitted to the throws from the ground (reach's terrain kept)."""
    grid = sorted(set(TOP0_GRID) | {reach.SEARCH_TOP0, CLIENT_TOP0, LOGS_TOP0})
    return _fit("SEARCH_TOP0", ground(rows), grid, lambda v: replace(reach_model(), top0=v), model)


def fit_floor0(rows: list[dict], model: MapModel) -> Fit:
    """The window's floor (SEARCH_FLOOR0) fitted to the throws from the ground (the top as reach has it)."""
    grid = sorted(set(FLOOR0_GRID) | {reach.SEARCH_FLOOR0})
    return _fit("SEARCH_FLOOR0", ground(rows), grid, lambda v: replace(reach_model(), floor0=v), model)


def fit_shot_h(rows: list[dict], model: MapModel) -> Fit:
    """The line's height (SHOT_H) fitted to every throw (from the ground and from jumps), by the line in force
    (reach.LINE: the level line's height, or where the v83 line starts above her feet). Until 2026-10-21 it was
    fitted by the level line whatever reach had, so where only a floor stops the line (throws down past her own
    platform's end) nothing bounded it."""
    grid = sorted(set(SHOT_H_GRID) | {reach.SHOT_H})
    return _fit("SHOT_H", rows, grid, lambda v: replace(reach_model(), shot_h=v), model)


@dataclass(frozen=True)
class Question:
    """An open question of attack.reach that probes put to the game: its answers (each reach's model with
    what's given changed: `answers`, (name, changes)), scored over the throws of the probes that ask it."""
    key: str                  # what of reach it settles
    asks: str
    probes: tuple[str, ...]
    answers: tuple[tuple[str, dict], ...]

    def models(self) -> dict[str, Model]:
        """Each answer's model."""
        now = reach_model()
        return {name: replace(now, **changes) for name, changes in self.answers}

    def reach_has(self) -> str | None:
        """The answer reach has now, if one."""
        now = reach_model()
        return next((name for name, changes in self.answers
                     if all(getattr(now, k) == v for k, v in changes.items())), None)

    def rows(self, rows: list[dict]) -> list[dict]:
        """The throws of the probes that ask it."""
        return [r for r in rows if r["probe"] in self.probes]


QUESTIONS = {q.key: q for q in (
    Question("SEARCH_TOP0", "the search window's top", ("top-60", "top-120"),
             (("44, the client's", {"top0": CLIENT_TOP0}), ("21, the logs' fit", {"top0": LOGS_TOP0}))),
    Question("CLAW_FAR", "how far ahead the claw swipes on her level", ("claw-edge",),
             (("the stabs, cut to 65 ahead", {"claw": STABS}),
              ("one box to 90 ahead, her level only (until 2026-10-20)", {"claw": OLD_CLAW}))),
    Question("claw_takes", "whether the claw swipes off her level", ("claw-down", "below-42"),
             (("the stabs, 15-30 above her feet", {"claw": STABS}),
              ("the swings, 17 below to 62 above her feet", {"claw": SWINGS}))),
    Question("RAMP_BELOW", "the search between her level and 60 below her", ("below-42",),
             (("the search alone", {"ramp": False}), ("the old ramp her decisions keep to", {"ramp": True}))),
    Question("LINE", "which line terrain stops", ("step-down", "hanging-wall"),
             (("the v83 line", {"line": "v83"}), ("the level line", {"line": "level"}))),
    Question("WALLS", "whether her own foothold group's walls stop a star", ("step-up",),
             (("they do", {"walls": "group"}), ("no wall does", {"walls": "none"}))),
    Question("OTHER_WALLS", "whether other groups' walls stop a star", ("other-wall",),
             (("only her own group's do", {"walls": "group"}), ("every wall does", {"walls": "every"}))),
    Question("in_area", "how an attack from a jump picks", ("air",),
             (("the area fixed to her feet (reach.in_area)", {"air": "area"}),
              ("as from the ground, from as high as she is (the v83 client's)", {"air": "v83"}))),
)}
DECIDES = 3.0        # an answer this much log-likelihood better than the next is the game's: about 20 to 1 ...
DECIDE_THROWS = 5    # ... on at least this many throws where the two have the attack take something else (one or two
                     # decided below-42 on some seeds in the simulator: a misread bar could have settled a constant)


@dataclass
class Answers:
    """A question's answers scored: the log-likelihood of each (`lls`), over how many throws, and on how many
    of them the two best differ in what the attack takes."""
    question: Question
    lls: dict[str, float]
    throws: int
    differ: int

    @property
    def ranked(self) -> list[str]:
        return sorted(self.lls, key=lambda k: -self.lls[k])

    @property
    def lead(self) -> float:
        """How much better the best answer fits than the next."""
        best, nxt = self.ranked[:2]
        return self.lls[best] - self.lls[nxt]

    @property
    def decided(self) -> str | None:
        """The answer the throws decide on (DECIDES better than the next, on DECIDE_THROWS throws or more where
        they differ), if any."""
        return self.ranked[0] if self.differ >= DECIDE_THROWS and self.lead >= DECIDES else None


def answers(rows: list[dict], model: MapModel, question: Question) -> Answers:
    """`question`'s answers scored over the throws of its probes in `rows`."""
    use, body, models = question.rows(rows), bodies(model), question.models()
    lls = {name: likelihood(use, lambda r, m=m: chances(r, model, body, m))[0] for name, m in models.items()}
    ranked = sorted(lls, key=lambda k: -lls[k])
    a, b = (models[k] for k in ranked[:2])
    differ = sum(1 for r in use
                 if [c[1:] for c in chances(r, model, body, a)] != [c[1:] for c in chances(r, model, body, b)])
    return Answers(question, lls, len(use), differ)


def answers_text(ans: Answers) -> list[str]:
    """`ans` as text: each answer's log-likelihood (reach's marked) and what the throws say."""
    q, now = ans.question, ans.question.reach_has()
    lines = [f"  {q.key}, {q.asks} ({', '.join(q.probes)}: {ans.throws} throws):"]
    if not ans.throws:
        return lines + [f"    no throws yet (`probe run {q.probes[0]}`)"]
    lines += [f"    {name}{' (reach)' if name == now else ''}: {ans.lls[name]:.2f}" for name in ans.ranked]
    best, nxt = ans.ranked[:2]
    if not ans.differ:
        lines.append(f"    -> can't tell: {best} and {nxt} have the attack take the same on every throw")
    elif ans.decided:
        lines.append(f"    -> {best}: {ans.lead:.1f} better than {nxt}, on the {ans.differ} throws they differ on")
    elif ans.differ < DECIDE_THROWS:
        lines.append(f"    -> not decided: {best} {ans.lead:.1f} better than {nxt}, but on only {ans.differ} throws "
                     f"they differ on (decided from {DECIDE_THROWS})")
    else:
        lines.append(f"    -> not decided: {best} only {ans.lead:.1f} better than {nxt} (decided from {DECIDES:g}), "
                     f"on the {ans.differ} throws they differ on")
    return lines


def drop(e: dict, exp: int, only_gone: bool) -> float | None:
    """How much of its HP bar a monster of a throw's evidence lost, if it showed a hit (`shown`): from its bar before
    (full if none) to the lowest after; the whole of what was left if it was killed. None: no hit, or can't tell."""
    if not shown(e, exp, only_gone):
        return None
    before = 1.0 if e["bar_before"] is None else e["bar_before"]
    seen_bars = [b for _, b in e["after"] if b is not None]
    if exp and e["gone"] and only_gone:
        return before
    return before - min(seen_bars) if seen_bars else None


CLAW_COUNTS = ("throws", "claw", "star", "killed", "no hit", "can't tell")


def claw_table(rows: list[dict]) -> dict[tuple, dict[str, int]]:
    """The claw probes' throws (named claw-*) by (probe, the target's kind, its height below her (+), how far off it
    was at the pick in 10-unit bins): how many (CLAW_COUNTS), and what the target showed: a hit the claw's (under
    claw_drop of its bar) or a star's, killed (either), no hit, or can't tell."""
    table: dict[tuple, dict[str, int]] = {}
    for row in rows:
        if not row["probe"].startswith("claw"):
            continue
        found = [(e, s) for e, s in seen(row) if e["target"]]
        if not found:
            continue
        target, s = found[0]
        x, y = _her(row)
        key = (row["probe"], target["kind"], round(target["y"] - y), int(abs(target["x"] - x) // 10) * 10)
        g = table.setdefault(key, dict.fromkeys(CLAW_COUNTS, 0))
        g["throws"] += 1
        g[{None: "can't tell", "none": "no hit", "hit": "killed", "claw": "claw", "star": "star"}[s]] += 1
    return table


def claw_lines(rows: list[dict]) -> list[str]:
    """claw_table, as text."""
    return [f"  {name} {kind} {dy:+d} {lo}-{lo + 10}: " + ", ".join(f"{k} {g[k]}" for k in CLAW_COUNTS)
            for (name, kind, dy, lo), g in sorted(claw_table(rows).items())]


def _target(row: dict) -> tuple[dict | None, bool | None]:
    found = [(e, s) for e, s in outcomes(row) if e["target"]]
    return found[0] if found else (None, None)


def _table(rows: list[dict], key: Callable[[dict], tuple]) -> list[str]:
    """Throws grouped by `key`: how many, and what their targets showed."""
    groups: dict[tuple, list[int]] = {}
    for row in rows:
        _, s = _target(row)
        g = groups.setdefault(key(row), [0, 0, 0, 0])
        g[0] += 1
        g[{True: 1, False: 2, None: 3}[s]] += 1
    return [f"  {' '.join(map(str, k))}: {n} throws, the target hit by {hit}, not by {miss}, {unknown} can't tell"
            for k, (n, hit, miss, unknown) in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0])))]


def _fit_text(fit: Fit, now: float, other: tuple[str, float] | None = None) -> str:
    if not fit.throws or not fit.decided:
        return f"{fit.name}: no throws to fit it to"
    lo, hi = fit.fits
    ends = [w for w, at_end in (("below", lo == min(fit.lls)), ("above", hi == max(fit.lls))) if at_end]
    text = (f"{fit.name}: fits {lo:g} to {hi:g}"
            + (f" (not bounded {' or '.join(ends)} by these throws)" if ends else "")
            + f", best {fit.best:g} (log-likelihood {max(fit.lls.values()):.2f}); "
            f"reach has {now:g} ({fit.lls[now]:.2f}, {'fits' if fit.holds(now) else 'does not fit'})")
    if other is not None:
        name, v = other
        text += f"; {name} {v:g} ({fit.lls[v]:.2f}, {'fits' if fit.holds(v) else 'does not fit'})"
    return text + f". {fit.throws} throws, {fit.decided} monsters told, {fit.hits} hit"


def score(rows: list[dict], model: MapModel) -> str:
    """The probe log scored (see the module's docstring), as text."""
    lines = [f"{len(rows)} probe throws, scored by the {reach.LINE} line (reach.LINE) where a question doesn't weigh "
             f"it; by probe, how she threw, and where she stood (map x):"]
    lines += _table(rows, lambda r: (r["probe"], r["style"], f"x {round(r['x'] / 10) * 10:g}"))
    lines.append("By the target's kind and how far off it was at the pick (20-unit bins, middle to middle):")
    lines += _table(rows, lambda r: (r["probe"], r["style"], r["target"]["kind"], _bin(r)))
    lines.append(f"The probes' questions, each answer's log-likelihood over the throws of the probes that ask it "
                 f"(higher fits better; decided {DECIDES:g} better than the next):")
    for q in QUESTIONS.values():
        if q.rows(rows):
            lines += answers_text(answers(rows, model, q))
    asked = {p for q in QUESTIONS.values() for p in q.probes}
    if unasked := sorted({r["probe"] for r in rows} - asked):
        lines.append(f"  (asking none of them: {', '.join(unasked)})")
    lines.append("The search's window, fitted to every throw from the ground:")
    lines.append("  " + _fit_text(fit_top0(rows, model), reach.SEARCH_TOP0, ("the logs' fit", LOGS_TOP0)))
    lines.append("  " + _fit_text(fit_floor0(rows, model), reach.SEARCH_FLOOR0))
    if claws := claw_lines(rows):
        lines.append(f"The claw, by the target's kind, its height (+ below her) and how far off it was at the pick "
                     f"(a hit taking under {CLAW_SPLIT_HP} HP of its bar is the claw's: "
                     f"{claw_drop('pig'):.0%} of a pig's, {claw_drop('slime'):.0%} of a slime's):")
        lines += claws
    body, now = bodies(model), reach_model()
    blocked = [r for r in rows if _target(r)[0] is not None and not gets_to(r, _target(r)[0], model, now, body)]
    told = [s for r in blocked for _, s in (_target(r),) if s is not None]
    lines.append(f"Behind what stops a star by reach ({reach.LINE} line): {len(blocked)} throws, the target hit by "
                 f"{sum(told)} of {len(told)} told")
    lines.append("  " + _fit_text(fit_shot_h(rows, model), reach.SHOT_H))
    return "\n".join(lines)


def _bin(row: dict) -> str:
    """How far off the target was at the pick, in 20-unit bins."""
    target, _ = _target(row)
    x, _ = _her(row)
    off = abs((target or row["target"])["x"] - x)
    lo = int(off // 20) * 20
    return f"{lo}-{lo + 20}"
