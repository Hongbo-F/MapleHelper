"""The hunt loop: hunt mode of gamelab/docs/hunt-modes.md. One decision after every look, instead of a
tour's stops, loot trips and fights on the way: she shoots what's in reach from a good place, picks up
loot, moves to a better spot, waits a moment for respawns, or heads for the next hunting ground, and a
monster coming into reach or a pile of drops changes the next step at once.

After each look, the first rule that applies decides (the design's numbering; at a ground they're tried
in the order listed, 6 before 3):

   1  close quarters   a target on her platform the claw would swipe (stance.close_to, by its size: a pig
                       within 111, a snail within 96), or nearer than its danger warrants (attack.danger: a
                       Horny Mushroom within 169): stance's answer (jump back, jump-throw, step back)
  6a  valuable pile    a pile with a drop worth VALUABLE mesos or more on its own (looting.Worth: the game
                       data's prices; on the first map a Slime Bubble, a White Potion, a gold coin; on another
                       twice its bronze coin, `valuable`) within VALUABLE_TRAVEL: at once with nothing in reach,
                       travelling or not; with something in reach it waits (EXP before loot, below)
   2  high ground      on a platform she can't safely stand on, with a spot that reaches her ground: back up;
                       also back to the spot from any other platform (after a pile there), unless the
                       ground is in reach from it or she's leaving
   3  shoot            a target in reach from her spot (or the ground she hunts on): throw, step out from
                       under, close in along her platform
   4  in passing       travelling: a target in reach, thrown at from where she is (or a step of PASS_STEP at
                       most out from under one above her: from 22 at 21's), PASSING_S at most per platform per
                       objective (the navigator shoots what stands in the way of a move)
   4b finish           travelling to another platform, done shooting in passing: the monsters she has hurt
                       and not killed (fight.owed), when a throw from here or a walk of fight.FINISH_WALK at
                       most along her platform gets to them (fight.finish_act), FINISH_S at most per
                       objective; she used to leave monsters half hunted
   S  sweep            the loot sweep (behavior.sweep, from 2026-10-30): travelling, the piles within what each may
                       add to her way to where she's going (`_allow`: by its age, and the danger there, from
                       2026-11-01); at a ground with nothing in reach, those within it of a round trip from her
                       spot; leaving it, before she goes, those within it of her way to the next (`_sweep`,
                       `_sweep_act`: a guard shot first, the trip to a pile, its stand, its drops' landing waited
                       for); a fresh pile of hers by her left to her pets a moment (`_pets_first`)
   7  better spot      wanted monsters seen out of reach, and a spot that reaches clearly more of them
   9  wait             monsters about or due back: stand still, for the patience at most, where she is or at the
                       safest place along her platform that still has the ground's spawn points in reach
                       (_wait_place: not between two Horny Mushrooms' spawn points), or on a rope from her
                       platform where nowhere along it is safe enough (_refuge, attack.ropes)
  11  advance          leaving: one trip to the next ground's spot (after the sweep's piles on her way there)

Until 2026-10-30 her drops had five rules of their own (5 pile here, 5b on the way, 6 urgent pile, 8 lull loot, 10 loot
elsewhere), each with its own bounds (PILE_NEAR, DETOUR_S, URGENT_MARGIN, DROP_LIFE, OTHERS_AFTER), and a pile waited
while a monster that counts was within PILE_CLEAR of where she'd go for it. Live on 2026-10-30 (the thief's run,
D:/MapleHelper-scratch/replan/loot) she picked up 199 mesos' worth a minute of the 299 the perception service saw (the
user's own play 306 of 437), and lost 100 a minute: 47% of that waiting on PILE_CLEAR (12-28% of those waits for the
pile's own dying monster), 28% known and never gone for, 15% yielded to something in reach; 48% of it lay within 2 s of
her way while it lasted, and half her departures from a stop came within 3 s of a kill of hers, before its drops landed.
The sweep takes all of those as one choice, until 2026-11-01 priced in her time (the loop's rate, a meso MESO_EXP),
within a budget of her way's time; from then each pile within the seconds it may add to her way (`_allow`: ALLOW_FRESH
soon after its kill, ALLOW_OLD once it's old or more dangerous than where she stands), the most mesos within those.

EXP before loot (`_loot_first`; the user, 2026-10-24: "The hunt should be driven by maximizing the exp gain"): no pile
goes ahead of shooting what's in reach by a fixed priority. The sweep comes after the shooting rules (until 2026-11-01
pricing her time at the more of the loop's rate and what's in reach would give: `_time_worth`). A valuable one (6a)
waits while something is in reach (until the review of 2026-11-01 one at risk by then went first if what waiting would
likely lose of it, at MESO_EXP an meso, was more than the EXP she'd make from what's in reach over the trip there and
back). A trip for a valuable pile is called off when a monster comes into reach: the pile stays known, and is gone for
again once nothing is in reach. In the live hunts of 2026-10-24 (1474 looks with a wanted monster in reach on another
platform than hers), she didn't throw within 1.5 s on 184: on 78 she was picking up a pile, on 43 the valuable pile rule
had her off to one (20 of those deciding it or on the way).

No touch for loot (the user, 2026-10-30: "try not to hit by mobs just for picking up loots"; from 2026-10-30, the sweep):
no pile is stood on where a monster that counts as a danger would meet her standing there (`_guards`: her body's box
against each one's, walking the way perception says it heads at its kind's speed from the game data, turned back at its
platform's ends, over the stand and sweep.STAND_PAST_S more; seen, about though the look missed it, localization.unseen;
never her own kill dying there, ledger.dying, nor the feed's tracks taken for dead, Ledger.dead); one she can shoot from
where she is is shot first ("in general you should kill all the mobs that are in the way": its kill time in the pile's,
its EXP in its worth), any other leaves the pile out until a look sees it clear. Her ways to piles are the navigator's
trips, planned and replanned at every state against every monster, with its shots first at what's in the way. Live on
2026-10-25 all 12 of the Horny Mushrooms' touches and all 3 knocks off 23 came on stays on 23 for drops; until 2026-10-30
a pile waited while one was within PILE_CLEAR (100 units) of where she'd go for it, whatever its way, and was priced by
the HP the rest of its danger would cost (250 EXP a bar). And what her fights go by takes those about her that the look
didn't find (`_unseen`: the feed's hidden tracks, her memory's of the last UNSEEN_S): around her for the stance's danger
and its moves, and those on her platform that count as a danger as targets too (`_with_unseen`).

She leaves a ground by the rate rule (`_leaving`): her EXP rate there falls clearly below what the best
other ground would give, travel included, or nothing is left there and nothing is due back. Not for the ground she'd
pick again at once (the only one: `--grounds 23`, or a value map with one): then it's waiting (`_elsewhere`). Objectives
are suspended, not ended, when a higher rule fires: they keep their route weights and their clock
(OBJECTIVE_S, suspensions included).

What she knows of the monsters off her band is her memory of them (Hunter.memory, localization.memory: every look's,
kept across looks, forgotten by looks that cover their place without them or by age), and every look of hers has the
monsters perception follows on the whole screen (Scene.swept: localization.feed.Feed.look), so the memory is kept up
while she shoots: the ground's monsters out of reach (rule 7) and whether anything is left on it (the fast exit) go by
it, and what guards a pile goes by the whole state. Her own platform's, which the fights react to, are the look's as
before.

The numbers marked "guess" are starting values for the simulator's A/B runs (gamelab.sim.harness); none
is measured in game. Respawns were measured on the live telemetry (analysis.respawn), and didn't settle RESPAWN: its
comment says what it did show; how long drops lie is ledger.SURVIVAL, the live service's.
"""

import random
import time
from collections import Counter
from dataclasses import dataclass, field, replace

from maplehelper.engine import LEFT, RIGHT

from .. import telemetry
from ..attack import reach, ropes, spots
from ..attack.ropes import Refuge
from ..attack.danger import NEWBORN_SEEN, NEWBORN_TOUCH, PASS_AFTER, PASS_BEFORE, RESPAWN_NEAR
from ..attack.spots import GROUND_MARGIN, Spot
from ..attack.stance import (MIN_MOVE, act_text, beside, close_to, drop_limits, jump_hits, open_above, overhead,
                             stance, too_close)
from ..control.actions import PILE_EVERY_MS
from ..control.mover import HANG_TOLERANCE, ON_ROPE_X
from ..control.navigator import ARRIVE_TOLERANCE, RUN_SEEN, RUN_UNITS, TRANSIT, Choice, TripState
from ..localization.monsters import (DEFAULT_SPEED, Monster, Tracker, bodies, speeds, touch_box, tracks_record,
                                     walk_edges, walk_vx)
from ..localization.players import near_rival
from ..localization.scene import Scene
from ..localization.unseen import UNSEEN_S, unseen
from ..motion import physics
from ..motion.bodies import HER_HALF
from ..motion.physics import WALK_SPEED
from ..perception.framecheck import LookLost
from ..perception.kinds import kind_of
from ..perception.loot import MESOS
from ..perception.mobs import Mob
from . import fight, sweep
from .keycheck import Doubt, KeyDoubt
from .chatcheck import ChatDue
from .statcheck import StatsDue
from .fight import THROWS, FightState, Report, kills_text
from .ledger import DROP_LANDS, OWN_S, Pile, dying
from .looting import LOOT_REACH, MIN_WALK, PILE_GAP, SETTLE, STOOD_ON, _piles, expected_worth
from .tour import PRIOR

IN_HER_WAY = fight.IN_HER_WAY  # any monster this close on her platform is a target, whatever its kind
FINISH_S = 8.0         # rule 4b: s of finishing what she has hurt per objective, before (and while) she goes.
                       # Unbounded, in the simulator's basin, where respawns never stop, there was always one
                       # she had hurt in reach: piles and spots ran out their OBJECTIVE_S (52 times in 30 runs of
                       # 20 minutes, against 26) and she picked up 2.9 points less of her drops (and with it,
                       # where the objective was on her own platform, 55)
PASSING_S = 15.0       # rule 4: s of shooting per platform per objective while passing through ...
PASS_STEP = fight.FINISH_WALK  # ... and the longest step out from under a target there (_pass_step)
# A pile is left after at most this many stands or trips (a suspension counts): the third time, drops still where she
# stood twice can't be picked up (looting.cant_pick: someone else's, or her inventory full), the rest stood on once more
PILE_TRIES = 3
SURVIVE_MIN = 0.05     # a pile less likely than this to lie there still (Pile.still: ledger.SURVIVAL) isn't gone for
SPREAD = 30.0          # units either side of where her kill died that its stand walks, its drops not seen yet (live,
                       # neighbouring drops lay 20-25 apart, and a pile of 3 spans about 40)
SPOT_MORE, SPOT_TIMES = 2, 1.5  # rule 7: another spot must reach 2 more monsters, or 1.5 times as many ...
SPOT_TRAVEL = 8.0      # ... within this many s of travel ...
SPOT_REST = 10.0       # ... not sooner than this after taking a spot ...
SPOT_CHANGES = 4       # ... and at most this many times per visit to a ground
# s from a kill until the monster is back (the simulator's): rules 8 and 9 wait for it, and the fast exit leaves a
# ground RESPAWN + 2 s after her last kill there. The live telemetry of 2026-10-19 to 21 says the game doesn't refill a
# platform she empties so (analysis.respawn: README, Respawns and drop life): of 77 platforms a kill of hers emptied,
# each followed while the looks covering it came 2 s apart at most, 12 were seen with a monster again, all within 10.8 s,
# and 80% weren't by 12.56 s (95% interval 67-88%, 23 still followed then; MeowDB's slowest respawn at a spawn point:
# its 5 s cooldown and the 7.56 s tick); none was watched so past 79 s, and too few past 30 s to say more (11 followed:
# 80%). In the simulator, by the same report, 11 of 22 were seen back by 8.8 s, the rest watched only to 6.3 s. What to
# put here instead isn't settled (the median isn't reached, and the game may send a respawn elsewhere on the map, past
# a cap of its monsters, rather than later): it stays until the in-game check (README). It decided 11.5% of the live
# hunt loops' time (rule 8, 6.2%; rule 9, 5.3%) and 20 of their 43 departures from a ground.
RESPAWN = 8.0
PATIENCE = {"ground": 1.0, "spot": 2.0, "none": 0.4}  # rule 9: today's patience values
# Rule 9: waiting, she walks to a safer place along her platform (_wait_place) only for this much less danger there
# (attack.danger: the share of her HP touches would take; the spawn points this look's monsters don't account for may
# fill at any moment). A guess, not measured: under it the walk isn't worth its time and the throws it may cost (on
# 24, the places GROUND_MARGIN from its ends are 0.072 as the spot at 231 is, and the safest of all, 209-210 just past
# 23's Hornies' reach, only 0.005 safer: she stays). On Tree Dungeon's 23 at x -10, on a Horny Mushroom's spawn point
# with the other 105 off, it's 0.28; at -147, 132 and 242 from them, 0.15. Live on 2026-10-24 she waited at x -10
# +- 5 for 49 s, 37% of her waits within 20 units of a spawn point, and respawns landed on her there twice (11% and 19%
# of her HP).
WAIT_GAIN = 0.03
WAIT_STEP = 100.0      # ... in steps of this many units at most (0.7 s at her 149 units/s), a look after each: the walk
                       # is blind (actions.walk), and a respawn or a monster walking in changes where's safest
# Rule 9 on a rope (_refuge: attack.ropes): only where it's this much safer (the danger, a share of the HP she has
# left) than anywhere along her platform, its own danger and the HP she doesn't regain there for HANG_EXPECT_S (the
# live median respawn on Tree Dungeon's 23) counted: getting on and off takes 0.6-1.5 s she throws at nothing, a fresh
# respawn walking at her unanswered meanwhile. A guess, so a rope is no default: on Tree Dungeon's 31, waiting for its
# Horny Mushroom, the rope at -192 (out of its reach, 1.1 s on and off) is 0.10 safer than 31's safest place (0.12),
# where on its Green Mushroom grounds no rope gains more than 0.05 (and 31's foot rules it out: _refuge).
ROPE_GAIN = 0.05
HANG_EXPECT_S = 17.0
HANG_MAX_S = 60.0      # backstop: s on a rope before she gets off whatever comes
ROPE_AGAIN_S = 10.0    # s before another rope, after she got off one for anything but a monster to shoot
OFF_TO_SHOOT = "a monster to shoot, and nothing where she lands"
# A ground's EXP a second (the rate rule, the next ground) counts 1 - GROUND_DANGER x the danger where she'd stand to
# hunt it (its spot's: attack.spots, the spawn points' monsters at their points), never under 0: on Tree Dungeon 23
# shot from 24 (danger 0.07) keeps 86%, 10 hunted standing among its two Horny Mushrooms (0.18) 64%. In the simulator's
# crowd (harness: the whole map, 20 seeds of 20 minutes, Lucky Seven), with the spots by the danger but every ground's
# EXP counted whole she lost 1.39 bars of HP a minute alive, as before them, and 13 of the 20 died (14 before); with
# this 1.03 and 6, and she hunted 23 from 24, 15 from 16 and the floor more, 10 among its Hornies less (README:
# Positioning). At 1.0 she lost 1.12 a minute and 8 died, at 3.0 1.00 and 3 (a run each, before the spots kept
# GROUND_MARGIN). Where nothing counts as a danger (the first map), every ground counts all of its EXP, as before.
GROUND_DANGER = 2.0
PRIOR_S = 20.0         # the rate rule: a ground's expected rate counts as this many s of hunting there
MIN_STAY, MIN_THROWS = 15.0, 6  # ... judged only after this long there, or this many throws
LEAVE_MARGIN = 0.8     # ... leave when her rate is under this share of the best other ground's
VISIT_S = 60.0         # guess: a visit's length, until measured visits replace it
SKIP_S = 120.0         # a ground left with nothing on it is skipped this long
# The next ground (_next_ground): the best EXP a second, travel included, each ground's varied by a random 0.75-1.25.
# Of those within GROUND_TIE of the best, the one she left longest ago (or never got to) goes first, so grounds about
# as good take turns. Off, as nothing measures it: in the harness's hunt loops (preset game, 40 seeds of 20 minutes)
# 0.25 changed not one key (nor 0.1 with route.ROUTES 3), 31 of the 40 runs taking up one ground, the basin, for the
# whole run; live, by the value map, 17 and 1 were within 3% of each other from the basin and from 3, 4 ahead of both,
# and her grounds already varied (4 24 times, 36 20, 17 16, 1 6: README, Route diversity).
GROUND_TIE = 0.0
EMPTY_CONFIRM_S = 3.0  # ... once looks covering it (GROUND_SEEN of it, at least) have seen it empty this long:
GROUND_SEEN = 0.75     # a hit monster hides for 1-3 s, and one look missing it skipped a busy ground
WIDE_EVERY = 2.0       # s between whole-screen looks while nothing is going on
OBJECTIVE_S = 60.0     # backstop: s on one objective, suspensions included
GROUND_S = 600.0       # backstop: s at one ground
STUCK_S = 120.0        # backstop: s without a kill, a pick-up or a new ground before she stands by
# ... the game's key bindings read first (behavior.keycheck, a "stall" doubt), once a stretch: her keys moved in the game
# (an arrow, her jump) can keep her throwing where she stands with no kill (the simulator's runs with her right arrow on
# E or her jump on C: no doubt of her runs or jumps for minutes, her chains none); followed, the hunt goes on. False: she
# stands by at once, as before.
STUCK_CHECKS = True
STALL_WAIT_S = 30.0    # ... its check put off (no quiet moment), the hunt goes on this long more for one
PICK_S = 0.6           # s standing on a pile (4 presses): rule 6a's trips (the sweep's stands are each pile's own)
# Whether her fights, her drops' price and guards and her throws' attack go by the monsters about her that a look didn't
# find (`_unseen`: localization.unseen); off, by the look's alone (and the fight's own hidden ones), as before. With her
# trips' (control.navigator.UNSEEN_TRIPS: the numbers there).
UNSEEN_FIGHTS = True
# ... her throws' attack (behavior.attacks: Lucky Seven at one in its way, the regular attack at "nothing seen in its
# way"; `_unseen_here`, those on her platform). A monster under her body is one the looks miss (none within 25 units in
# 1317 live looks), and a throw at it went out as the regular attack. Taken with the dead ones' tracks in (before the
# review of 2026-10-26: localization.unseen.hidden) it spent 11-13% more MP for her EXP (16 seeds of 10 minutes: the
# review's). With them left out, in the simulator (Tree Dungeon, crowd, `--attack choice`, 16 seeds of 5 minutes; D:/
# MapleHelper-scratch/hunt-ab/B/fix/out/round6.txt), against without: grinding, EXP a minute +9.1% and +7.4% (near-
# misses, zero-walls; 2.4 and 3.7 standard errors), MP for her EXP +1.6% and -3.8%, HP lost -12% and 0%; the hunt loop
# +1.3% and -4.9% EXP (neither 1.6 standard errors), MP for her EXP +4.7% and +5.1% (2 standard errors each), HP +7% and
# +3%: in all, 3% more EXP for 2% more MP an EXP. Only those within 1 s (the review's "strong evidence"): +5.2%, +5.9%,
# +1.0%, -2.0% EXP and +3.8%, -4.0%, +5.0%, +1.2% MP an EXP; within 50 units of her alone, +2.4%, +3.5%, -1.5%, +1.3%
# EXP. Live on 2026-10-25 9 of her 325 throws went out as "nothing seen in its way" (the simulator, near-misses: 82 of
# 955), so live it counts for less either way.
UNSEEN_ATTACK = True
# Rule 6a: a pile with a drop in it worth this many mesos on its own (looting.Worth, the game data's prices: Pile.
# dearest) is gone for at once, ahead of shooting: of the first map's drops a Slime Bubble (50) and a White Potion (35),
# 2-8 times a pile's coins (15) or its other items (the user, watching her hunt on 2026-10-21: "Expensive items
# should attempt to pick immediately"). By a pile's worth all together, two pigs' coins and leather lying together in
# the simulator's basin (50) took her off shooting there 8 times in 3 minutes (loop_basin) ... This is the first
# map's, twice its bronze coin; on another map twice its own (Hunt.valuable, looting.Worth.valuable: on Tree Dungeon
# 66.8, its ores but Bronze and the Magic Potion; on West Rocky Mountain II 51.8, its Opal, Orihalcon, Silver, Emerald
# and Adamantium Ores and a gold coin).
VALUABLE = 30.0
# A bar of her HP's price (`bar_exp`): what a meso is worth to her in EXP. A guess putting EXP first, the user's to
# choose (the loot design: "the user is short on mesos: a reason to consider 1.0"). Until 2026-11-01 the sweep's price
# of her time too (its break-even a second of her way at rate / MESO_EXP: at the live thief's 6.9 EXP a second, 13.9
# mesos), and until the review of that day EXP before loot's (`_loot_first`: an at-risk valuable pile went first when
# what waiting would likely lose of it was worth more at it than what's in reach would give over the trip); from then
# her piles go by the seconds each may add to her way (`_allow`) and wait for what's in reach, not by an exchange the
# user wasn't sure of ("I'm not sure about this currency conversion. My idea is just picking up mesos and items as we
# kill mobs"): it prices her touches alone (HP, potions, mesos).
MESO_EXP = 0.5
# What her potions cost, by what they give: the user's reference prices of 2026-10-31 ("2HP/mesos and 1MP/mesos, use
# these as reference price. We may be well losing money by grinding, which is fine"), HP_PER_MESO HP and MP_PER_MESO MP
# a meso, so a 100-HP Red Potion 50 mesos, the simulator's 250-HP Orange Potion 125 and the 100-MP Blue Potion 100. Cash
# flow isn't a constraint (the user's word), so nothing in the planner keeps mesos back: a bar of her HP is the mesos
# its potions cost at MESO_EXP (`bar_exp`), and her MP isn't priced (her attack mix decides it: attack.choice, mp_floor
# 0). Until then a bar was 250 EXP everywhere (patrol.HP_PRICE, this module's HP_EXP: 3.1 Orange Potions at a guessed
# 160 mesos), now 193.5 at the simulator's 774 HP, 210 at the live thief's 840 and 157 at the magician's 628.
HP_PER_MESO = 2.0
MP_PER_MESO = 1.0
# Leaving a ground, the sweep's piles on her way out (`_leave`) within this many of the sweep's budgets over the whole
# way (sweep.budget_for of the straight way from where she decided to go: DEPART_BUDGETS x max(BUDGET_MIN, half of it)
# more than it, from then on), not one budget at each solve from wherever she is: re-solved at every pass with a fresh
# budget, every pile picked on the way (and every kill's) made room for another, as master's guard had it of the old
# rules ("every kill on a trip made another trip, and she never left"; the review of 2026-10-30: tour_tree_run took 628
# s against master's 534). None: a budget at each solve, as the sweep's first version.
DEPART_BUDGETS: float | None = 2.0
# The sweep's guards (`_about`, `_guards`) count the monsters on a platform just under or over a pile's whose bodies
# meet hers standing on it (Danger.beside: her full box against theirs, as attack.stance's BESIDE), from 2026-10-31;
# False: the pile's own platform's alone, as before.
BESIDE_GUARDS = True
EXP_FIRST = True       # EXP before loot on; off, the piles' fixed priorities as before (6a and 6 ahead of shooting)
VALUABLE_TRAVEL = 15.0  # ... when it's this many s of travel away at most (the urgent piles' 12 s, and a little),
VALUABLE_TRIES = 2      # ... this many times at most (then it's a pile like the others): by one, a monster can stand
                        # (guarded: looting.collect leaves it, not a try), and she can't shoot it from where it lies
# Whether her look is at the whole screen, with the drops on every platform in view, once a kill's of hers on another
# platform have landed (Hunt._far_drops): from a spot the band's drops are her own platform's, and one on the ground
# she shoots was seen only once she stopped shooting. In the simulator (loop_valuable_drop_at_once, seeds 1-16) she
# set off for the Slime Bubble before the mushrooms were dead on 16 seeds with it, on 12 without; the look takes about
# 120 ms more in game than the band's (README: What a look costs), once a far kill.
FAR_DROPS_LOOK = True
# (Rule 5b, piles on the way, from 2026-10-21 to 30: a stop at a pile DETOUR_S (2 s) at most on top of the trip,
# DETOUR_REACH (2.5 s) away at most, DETOURS (3) per objective. In the hunt loops of 2026-10-21 she made 44 trips for piles
# about to vanish, 23 of them for piles a trip of hers had passed within 2 s of before: on the route, only the stop on
# them. The sweep's budget, sweep.BUDGET_MIN and BUDGET_SHARE, has its place.)
# Another player on a ground or by it (localization.players: a rival's minimap dot on it, on its spot's platform, or
# PLAYER_NEAR across and PLAYER_DY up or down of its ends) takes its monsters: choosing the next ground, and in the
# rate rule's best other ground, its EXP a second counts PLAYER_DISCOUNT of itself (1: as if nobody were there). Off:
# in the simulator's game-like world (preset game, 20 seeds of 20 minutes) a discount of 0.85 or 0.7 changed 1-2
# seeds' runs and cost seed 7 27% of its EXP a minute (129.9 -> 128.7 on average; 0.5 on seeds 1-10, 127.9 -> 125.6):
# leaving the basin, she went to 17 (35% of it in reach) rather than to 4, where the other player had just come, and
# spent 5 minutes there, where on 4 she had made more with it beside her. The simulator's other player kills one monster every 25 s, a
# guess; a real one may take far more of a ground. With one killing every 8 s (20 seeds), 0.85 changed no run and 0.7
# one (+0.4% on average): too little to turn it on.
PLAYER_DISCOUNT = 1.0
PLAYER_NEAR = 200.0
PLAYER_DY = 100.0
# Whether a pile no kill of hers explains, with another player by it (players.near_rival), is left to them unless it
# holds a drop worth VALUABLE: theirs, it can't be picked up for a while, and they take it. Off, by the harness (preset
# game, 20 seeds of 20 minutes, against the same code with it off): with the preset's other player (a kill every 25 s)
# it changed 11 seeds' runs, 6 for the better, 130.80 -> 131.16 EXP a minute (+0.4%, standard error 0.8%), but her
# drops picked up fell 83.7% -> 82.7%; with one killing every 8 s (more like a real player), 11 changed, 4 for the
# better, 124.43 -> 122.15 (-1.9%, standard error 1.1%; the worst seed 11% down, and seed 6 fell out of the map at
# 1168 s, a corner of the simulator's walls), drops 85.3% -> 85.8%. (Before her EXP gains named their kinds: 9 seeds
# changed, 4 better, 129.93 -> 129.88, drops 83.7% -> 83.9%: it was on.) The other player there takes its drops 22 s
# after its kill, before they were worth a trip anyway (then OTHERS_AFTER, 30 s), so only piles taken for no kill of
# hers were left; whether a real player's are worth leaving is for the game to say.
PLAYER_PILES = False
# Her pets take what they pass (the user, 2026-11-01: "The pet will pick up whatever thing when the pet walks past
# those loot. But the pet won't proactively go for the loot"; her Snail pets' game data: pickupItem, no sweepForDrop):
# a pile of hers on the platform she's on, its stretch PET_NEAR of her or nearer, is left to them (no candidate:
# `_pets_first`) until she has been by it PET_WAIT s, from its drops' landing at the soonest; what they leave a look
# shows, and the sweep has as before. Live (the perception service's 15 fps records of 2026-10-25 to 30, D:/MapleHelper-
# scratch/replan/loot2/design/pets), her pets' share of the drops she came near without standing on them 25-40% by an
# approach of 60-150 units, the snail tags by her 53-102 units from her (the golden clip's quartiles), and a drop gone a
# median 0.71-0.83 s after her closest approach, p90 2-4 s: PET_NEAR and PET_WAIT are guesses built on those. In the
# simulator's grind with Meat and her pets (`--world meat --world pet-loot`: world.Realism.pets, fitted to the same
# records; 24 seeds of 15 minutes, D:/MapleHelper-scratch/replan/loot2/impl), on top of the allowances (`_allow`): EXP a
# minute +9.8 (se 5.0), kills +3.9 (1.9), loot's worth +2.0 (4.8), her pets' share of her pick-ups 0.31 -> 0.36, her
# drops' share 0.944 -> 0.921 (-0.023, se 0.009), her drops expiring 7.3 -> 10.8 a run; without them EXP +3.6 (6.9),
# loot +3.6 (6.7). Counted from a pile's landing alone (a pile she comes near later hers at once: 12 seeds, without the
# allowances) EXP +3.9 where +7.9; left to them only while she stays (not on her way, nor as she leaves: 12 seeds, with
# the allowances) EXP -1.5 (8.5), loot -2.0 (7.5). Off from the review of 2026-11-01 (D:/MapleHelper-scratch/replan/
# loot2/check): it puts a pile of hers off for her pets, against "piles should be picked up soon after killing", and its
# gain is the simulator's pets': they take 0.14-0.23 more of the drops she comes within 15-90 units of than hers did
# live (by her closest approach), and without them (the harness standard's world) it cut 48 trips for piles short in 8
# runs of 5 minutes, 9 of their 70 drops never picked up, the rest a median 3.7 s later; nothing shows the magician's
# pet picks up at all. On the user's word (README: Loot as she kills). False: none left to them (her pets take what
# they pass all the same: what they leave the looks show, and the sweep has).
PET_LOOT = False
PET_NEAR = 100.0
PET_WAIT = 4.0
PET_KEEP = 64  # (piles whose time by her is kept before the gone ones are dropped)
# What a pile may add to her way (`_allow`: the sweep's allowance of it, sweep.Spot.allow; an itinerary within the sum
# of its piles'), from 2026-11-01, in seconds rather than at MESO_EXP an meso against her time (the user: "I imagine
# piles should be picked up soon after killing. If a pile is about to expire, it probably means they aren't important
# enough to get picked up anyway. I think a real human would bias route planning to pick up piles"; "I'm not sure about
# this currency conversion. My idea is just picking up mesos and items as we kill mobs"): ALLOW_FRESH while it's FRESH_S
# old or younger (from its kill, or its first sighting), down to ALLOW_OLD at OLD_S and ALLOW_OLD after (on her way
# only: no late trips for old far piles, nothing for their expiry), and ALLOW_OLD at the most where the danger at it
# (attack. danger, with the look's monsters) is SAFER_MARGIN over where she stands ("try not to hit by mobs just for
# picking up loots": grind.BACK_MARGIN's, a third of a Horny Mushroom's touch on Tree Dungeon; the HP price of PILE_HP's
# place). Guesses: in the user's own play (the golden clip, 26 drops) her drops went a median 3.7 s after the kill, p90
# 18.9 s, the bot's live a median 6.0 and p90 55.9; its live detours for them a median 0.81 s, p90 1.83 s. In the
# design's prototypes (D:/MapleHelper-scratch/replan/loot2/design: 15-minute grinds with Meat and her pets, 24 seeds)
# ALLOW_FRESH 2 s cost loot (-25 a minute at 5 minutes, se 14.5), OLD_S 30 s and ALLOW_OLD 0.5 picked up less on 24
# seeds, and without SAFER_MARGIN the rules cost EXP (-9.6 a minute, se 6.6). As built
# (D:/MapleHelper-scratch/replan/loot2/impl, the same grind, 24 seeds of 15 minutes, against the sweep at MESO_EXP): EXP
# -3.1 (5.3), loot's worth +15.9 (5.2), her drops' share 0.905 -> 0.944 (+0.039, se 0.008), her drops expiring 12.2 ->
# 7.3 a run, touches -9.8 (4.9); with PET_LOOT EXP +6.7 (5.8), loot +17.9 (5.9), share +0.015 (0.011), touches -5.5
# (4.2), potions -4.6 (3.0). Scaled down by what a pile is worth to her against her map's bronze coin (12 seeds, her
# drops as live: `--world map-drops`) EXP -17.3 (7.8), loot -13.2 (5.2): not taken.
ALLOW_FRESH = 3.0
FRESH_S = OWN_S
OLD_S = 60.0
ALLOW_OLD = 1.0
SAFER_MARGIN = 0.05
# The spawn guard (`_respawn_guard`, from 2026-11-03: the cornered clip's R3, the user's "good to merge"; README, The
# spawn guard): a trip to a pile (`_go_for`: the sweep's, a valuable pile's; one within TRIP_AHEAD_S of her) whose stand
# there (from her arrival, or its drops' landing, to sweep.STAND_PAST_S after it) would come at or after the map's next
# pass (the hunter's spawn clock, localization.spawnclock, once sure) waits where she is for that pass to be seen
# (PASS_SEEN_S after it: '<rule>: the pass') where going has her within attack.danger.RESPAWN_NEAR of more free spawn
# points of monsters whose touch hurts about the pass (PASS_BEFORE before it to PASS_AFTER after: her stand there, her
# walk there along its platform) than waiting does (where she is): then she decides again with what the
# pass brought in sight (the sweep's guards: a newborn shot first from where she is, or the pile left), not walking onto
# the points as they fill (the clip's episodes A to C, her trips planned before the newborns were there), and by one
# herself she waits off it (`_off_points`). Only where what the newborns are expected to add at the user's prices (each
# point's fill at the pass, `_fill`, times NEWBORN_TOUCH less NEWBORN_SEEN, the touch's HP in potions, the share of her
# touches that hurt: behavior.fight.landing_prices) is more than the wait costs her (to the pass seen, at her EXP a
# second): at a fill of 0.27 and two Horny Mushroom points a wait under about 0.9 s, one 0.45 s. The itinerary stays
# as the sweep solved it. And waiting (rule 9, `_safest_along`) she counts each such point's newborn
# (attack.danger.respawn_risk) where she'd be over the next WAIT_AHEAD_S (PASS_BEFORE before a pass to PASS_AFTER
# after it). (The first build, 2026-11-03, left a pile out of the sweep's solve whenever her stand there would meet a
# pass, however far: 20-40 times a run, the itinerary changed each time, piles picked up 16-100 s later or never; in 4
# runs of the standard worlds her EXP a minute 1-9% less. The second held a trip only while her stand would start
# inside the window, PASS_BEFORE to PASS_AFTER about the pass: in the set-piece, loop_tree_respawn_pass, she then went
# before the pass and met the newborns on her way. The third held for any point by her walk or her stand, where she
# waited not set against it: in her standard world, seeds 1-12, her touches by a newborn 74 -> 93, held by a point
# that then filled by her.) False: as before (the danger's free points at every moment, no clock).
RESPAWN_GUARD = True
TRIP_AHEAD_S = 3.0     # guess: s of travel to a pile at most for her stand there to be timed (farther: a guess of s)
PASS_SEEN_S = 0.2      # s after a pass by which a look has its newborns (the spawn watch: at their first frame shown,
                       # alpha 0.065 at the median, perception.spawns; and a look's latency)
WAIT_AHEAD_S = 1.0     # guess: s ahead a place she waits at is held to the pass (a step away takes 0.3-0.7 s, a look)
OFF_MARGIN = 5.0       # units past RESPAWN_NEAR a trip's wait for the pass is made at, by a free point (`_off_points`)
FORGET_LOST_S = 21.0   # s of looks lost (three of the map's passes) after which the spawn clock forgets (`_lost_for`)
# A free point's chance of a newborn at a pass where nothing better is known (the hunt loop: behavior.patrol's prior,
# REFILL_P, 0.27; the grind's own from what it learned, Grind._fill)
FILL_PRIOR = 0.27


def bar_exp(max_hp: float) -> float:
    """What a bar of her HP (`max_hp` HP) costs in EXP: its potions' mesos (HP_PER_MESO HP a meso) at MESO_EXP."""
    return max_hp / HP_PER_MESO * MESO_EXP


def _keep_off(o: "Objective") -> tuple[int, ...]:
    """The platforms a trip to objective o keeps off (Navigator.begin's keep_off): a spot's ground, when the spot is
    off it (its monsters would stop her on the way; route.KEEP_OFF_S)."""
    return (o.ground,) if o.kind == "spot" and o.ground is not None and o.ground != o.platform else ()


@dataclass
class Objective:
    """Where she's heading: a ground's spot, or a pile."""
    kind: str                  # "spot" or "pile"
    platform: int
    x: float | None
    ground: int | None = None  # the ground a spot is for
    pile: Pile | None = None
    spot: Spot | None = None
    weights: dict = field(default_factory=dict)  # its route weights, kept through its suspensions ...
    choice: Choice = field(default_factory=Choice)  # ... and the way its trips take (route.ROUTES)
    # Its clock (OBJECTIVE_S) runs while it's suspended and while rules 1-4 shoot: that bounds a
    # shoot-respawns-on-the-way loop. (Counting only its own trips, in the harness's seed-7 world she tried
    # for platform 4's spot for 270 s and never got there.)
    started: float = field(default_factory=lambda: time.monotonic())

    def age(self) -> float:
        return time.monotonic() - self.started
    trip: TripState | None = None
    passing: dict[int, float] = field(default_factory=dict)  # platform -> s shot there while passing (rule 4)
    finishing: float = 0.0     # s spent finishing what she had hurt (rule 4b)
    why: str = ""              # the rule that set it ("valuable pile", "sweep", ...)
    after: tuple | None = None  # a sweep's pile: where its itinerary goes on to after it (platform, x)
    chain: object | None = None  # a sweep's pile: the chain through it and the next ones (behavior.lootchain), its trip one plan
    detours: int = 0           # piles taken on the way (rule 5b) while it was suspended

    def text(self) -> str:
        what = f"spot for {self.ground}" if self.kind == "spot" else "pile"
        return f"{what} on {self.platform}" + ("" if self.x is None else f" at x {self.x:.0f}")


@dataclass
class Visit:
    """A stay at a hunting ground."""
    ground: int
    spot: Spot
    arrived: float | None = None   # when she got to a spot that reaches it (None: on her way)
    exp: float = 0.0               # EXP her kills there gave
    throws: int = 0
    changes: int = 0               # spots taken after the first (rule 7)
    spot_since: float = 0.0
    kills: list[float] = field(default_factory=list)  # when her kills there were (respawns due back)
    empty_since: float | None = None  # since when looks covering the ground have seen nothing wanted on it
    backs: int = 0                 # trips back to the stop from where she was worse off (behavior.grind: BACK_TRIES)


@dataclass
class HuntReport:
    """What a hunt did, from its start to the end of its trip to stand by (Hunt.run): the run summary, the end of the
    run's log ('hunt done: ...') and its telemetry 'summary' record (`record`).

    Until 2026-11-02 it stopped counting at time-up, before that trip, and missed some of what came before: R1 of
    2026-11-02 (RiceBaby on Tree Dungeon, 00:09:36-00:14:57) said 52 kills, +1,731 off her EXP number and 80 throws,
    where the run had 53, +1,798 and 85 (D:/MapleHelper-scratch/recfix/records). Three causes: her EXP number's first
    reading came with her first kill (the perception service had read it long before, and a hunt that attaches later
    hears only its changes: now each change tells the number before it, localization.worldstate's `before`, which
    Hunter._vitals starts from), so that kill's 26 were lost; the navigator's shots clearing her way (behavior.fight's
    escort: 4 casts) count in the hunter's report, not the hunt's `throws`; and the stand-by trip's own shot, kill and
    41 EXP came after the count stopped. `throws` stays the hunt's own throws (what behavior.grind learns its kill
    times from); `casts` is every throw of hers over the run, the navigator's too. Its 'a minute' figures are over the
    run's minutes, the stand-by trip's too (`minutes`), since its kills are counted; summaries before 2026-11-02 went
    over the hunt's seconds alone, so the same run reads lower by the stand-by's share of its time (the simulator's
    loop_tree_grind_drops, 19 s of 199: 169.6 EXP a minute off her number then, 153.6 now)."""
    seconds: float = 0.0
    grounds: list[int] = field(default_factory=list)  # visited, in order
    kills: dict[str, int] = field(default_factory=dict)
    exp_gains: int = 0
    exp: float = 0.0
    exp_read: int | None = None  # EXP her number rose by, as read (None: never read)
    throws: int = 0
    piles: int = 0             # piles gone after (picked up, or tried)
    on_the_way: int = 0        # EXP gains while travelling to an objective
    rules: Counter = field(default_factory=Counter)  # passes each rule decided
    rule_s: Counter = field(default_factory=Counter)  # ... and the s from each of those passes to the next (Hunt._note)
    stop: str = ""
    standby_s: float = 0.0     # s of the trip to stand by after time-up (counted in all the run's figures)
    standby_kills: int = 0     # ... her kills on it
    casts: int = 0             # every throw of hers over the run (the hunter's report: the navigator's too)
    touches: int = 0           # monsters' touches of her (MISS too) ...
    misses: int = 0            # ... of them MISS
    drinks: int = 0            # potions she drank, her pet's too, as her HP number's rises tell (fight.Report.drinks)
    lowest_hp: float | None = None  # her lowest HP over the run, as her HP number read (None: never read)
    max_hp: float | None = None     # ... of her max HP as read

    @property
    def minutes(self) -> float:
        return (self.seconds + self.standby_s) / 60.0

    def per_minute(self, n: float) -> float | None:
        return n / self.minutes if self.minutes > 0 else None

    def __str__(self):
        read = ("not read" if self.exp_read is None else
                f"+{self.exp_read}" + (f", {self.per_minute(self.exp_read):.1f} a minute" if self.minutes else ""))
        clear = self.casts - self.throws
        a_min = (lambda n: f" ({self.per_minute(n):.1f} a minute)" if self.minutes else "")
        lowest = ("" if self.lowest_hp is None else f", lowest HP {self.lowest_hp:.0f}"
                  + ("" if self.max_hp is None else f" of {self.max_hp:.0f}"))
        return (f"{self.seconds:.0f} s" + (f" and {self.standby_s:.0f} s to stand by" if self.standby_s else "")
                + f"; grounds {self.grounds}; kills: {kills_text(self.kills)} "
                f"(EXP gains {self.exp_gains}, EXP {self.exp:.0f}; off her EXP number {read}"
                + (f"; {self.standby_kills} on her way to stand by" if self.standby_kills else "") + ")"
                + f"; throws {self.casts or self.throws}"
                + (f" ({clear} clearing her way)" if self.casts and clear > 0 else "")
                + f"; piles {self.piles}; touches {self.touches}{a_min(self.touches)}, {self.misses} MISS; "
                f"potions {self.drinks}{a_min(self.drinks)}{lowest}; ended: {self.stop}")

    def record(self) -> dict:
        """The telemetry 'summary' record's data (one at the run's end: Hunt.run)."""
        m = self.minutes
        rate = (lambda n: None if m <= 0 or n is None else round(n / m, 2))
        return {"seconds": round(self.seconds, 1), "standby_s": round(self.standby_s, 1), "kills": dict(self.kills),
                "exp_gains": self.exp_gains, "exp": round(self.exp, 1), "exp_read": self.exp_read,
                "throws": self.throws, "casts": self.casts, "standby_kills": self.standby_kills,
                "piles": self.piles, "on_the_way": self.on_the_way, "touches": self.touches, "misses": self.misses,
                "drinks": self.drinks, "lowest_hp": None if self.lowest_hp is None else round(self.lowest_hp),
                "max_hp": None if self.max_hp is None else round(self.max_hp),
                "a_minute": {"exp_read": rate(self.exp_read), "exp": rate(self.exp), "kills": rate(self.exp_gains),
                             "casts": rate(self.casts), "touches": rate(self.touches), "drinks": rate(self.drinks)},
                "grounds": len(self.grounds), "rules": dict(sorted(self.rules.items())),
                "rule_s": {k: round(v, 1) for k, v in sorted(self.rule_s.items())}, "stop": self.stop}


class Hunt:
    """Hunt mode over a Tour's knowledge of the map (its value map, kinds worth hunting, safe platforms,
    spots and stand-by) and its hunter and navigator."""
    valuable: float = VALUABLE  # rule 6a's: what a drop must be worth on its own, on her map (looting.Worth.valuable)

    def __init__(self, tour, log=print, rng: random.Random | None = None):
        self.tour, self.h, self.nav, self.model = tour, tour.h, tour.nav, tour.nav.model
        self.valuable = self.h.worth.valuable
        self.log = log
        self.rng = rng or tour.rng
        self.kinds = tour.worth
        self.report = HuntReport()
        self.visit: Visit | None = None
        self.obj: Objective | None = None
        self.stack: list[Objective] = []    # suspended objectives, the latest last
        self.fs = FightState()
        self._tops, self._halves = fight.body_sizes(self.model)  # each kind's size (game data), for stance
        self._fs_key = None
        self._owe_move = False               # cut short: move on (one chain) before shooting
        self._skip: dict[int, float] = {}    # ground -> until when it's skipped
        self._left: dict[int, float] = {}    # ground -> when she last left it, having got there (GROUND_TIE)
        self._kills_seen = 0
        self._gone = 0                       # piles she went for that looks have seen gone since (progress)
        self._progress = 0.0
        self._wide_at = -1e9
        self._idle_since: float | None = None
        self._stay_said: tuple | None = None  # the visit, its throws and kills when she last said she stays (_stay)
        self._covers_for, self._covers = None, {}  # what her attacks reach from where she stands, the look's (_cover)
        self._hanging: Refuge | None = None  # the rope she waits on (rule 9: _refuge), while she's on it ...
        self._hung_at = 0.0                  # ... since when
        self._rope_rest = 0.0                # no rope before this (ROPE_AGAIN_S)
        self._rule = ""
        self._noted: tuple[str, float] | None = None  # the last pass's rule and when it was noted (`_note`: rule_s)
        # Leaving a ground: why (None: not leaving), kept while she's off the ground and its spot (see `_at_ground`):
        # the decision itself is made again at every pass at the ground (kept, a weak reason such as "nothing came
        # into reach" lost respawns that came after it). The piles on her way out are the sweep's, within its budget
        # (until 2026-10-30 every pile left when she decided, and only those: going after the drops of monsters shot on
        # the way to them too, every kill on a trip made another trip, and she never left).
        self._exit_reason: str | None = None
        self._depart: tuple[float, float] | None = None  # ... and when she decided, the s her way out may take from then
        # (DEPART_BUDGETS: the straight way and the sweep's budgets)
        self._sweeper = sweep.Sweeper()      # the loot sweep's itinerary she's on (`_sweep`) ...
        self._swept_for, self._swept = None, None  # ... the look (kept: its id is never another's meanwhile) and the
        # way it was solved for last, and its first stop
        self._pile_of: dict[int, Pile] = {}  # ... its spots' piles (sweep.Spot.key: the pile's id)
        self._stood: dict[int, tuple] = {}  # pile (its id) -> (the pile, where she stood on it: looting.cant_pick)
        self._sweep_said = None              # the itinerary last said (its order) and recorded ('sweep')
        self._block_said: dict[int, tuple] = {}  # pile (its id) -> (the pile, whether it was left out unsafe as said)
        self._no_high_until = 0.0            # after a failed trip back up to the spot, not again for a while
        self._tracker = Tracker(self.model)  # the monsters for the mini-map (the navigator sends none with an escort)
        self._s: Scene | None = None         # the last look, for the 'hunt' record
        self._seen: Scene | None = None      # the latest look of all (a trip starts from it if it's still fresh)
        self._published = None
        self._end = float("inf")  # when the running hunt ends (run)
        self._loot_said: dict[int, bool] = {}  # pile (its id) -> whether it went first when she last said (_loot_first)
        self._unseen_for, self._unseen_list = None, []  # the look `_unseen` was worked out for, and its answer

    # --- the loop ------------------------------------------------------------------------------------

    def run(self, minutes: float) -> HuntReport:
        """Hunt for `minutes`, then go to a safe platform and stand by (as a tour ends)."""
        start = time.monotonic()
        end = self._end = start + minutes * 60
        self.h.report = Report()  # one report for the whole hunt
        if self.nav.escort is not None:
            self.nav.escort.kinds = self.kinds
        self._kills_seen = len(self.h.ledger.kills)
        self._gone = self.h.drops.gone_tried
        self._progress = start
        telemetry.emit("tour", order=[], safe=self.tour.safe, grounds=self.tour.hunting_grounds(self._visit_s))
        looter = self.h.looter
        yield_to, about = looter.yield_to, looter.unseen
        if EXP_FIRST:
            looter.yield_to = self._yields  # picking up stops for what's in reach
        looter.unseen = self._unseen  # ... and the monsters about that a look missed guard drops too
        attacks = getattr(self.h, "attacks", None)
        if attacks is not None and UNSEEN_ATTACK:  # ... and a throw's attack goes by those on her platform (attacks)
            attacks.unseen = self._unseen_here
        meter = self.h.meter
        read = meter.total if meter.since is not None else None  # her EXP as read at the start (counted from there)
        regains = self.h.senses.regains
        raises = getattr(self.h, "doubts_raise", None)
        if raises is not None:  # (her regular attack's key doubted: the check runs here, behavior.keycheck)
            self.h.doubts_raise = True
        try:
            # Her first look waits for the window fix's first try at a client it mends (Senses._fixing) as any
            # command's does, and stops the hunt if that didn't set it right; from there on, her looks failing stand
            # her still, and the fix makes its tries there (`regains`: Senses.regain). With `regains` set before it,
            # LookLost (LOST_S, 3 s) came before the fix's STEADY_S (5 s), and `tour --loop` or `--grind` started on a
            # client drawn off ended untried.
            first = self.nav.mover.settle()
            self.h.senses.regains = True
            self._next_ground(first)
            while time.monotonic() < end:
                try:
                    physics.settle()  # (a SPEED her stat window read in the loop made hers: between key plans)
                    if self._pass() == "stop":
                        break
                except LookLost as lost:  # the game doesn't look as expected: stand still until it does (senses)
                    paused = time.monotonic()
                    back = self.h.senses.regain(lost, end)
                    self._paused(time.monotonic() - paused)
                    self._lost_for(time.monotonic() - paused)
                    if back is None:
                        break
                except KeyDoubt as doubt:  # her keys doubted (her regular attack, her casts, her moves, out of
                    # darts due: behavior.keycheck): the check, then on
                    paused, back = time.monotonic(), True
                    try:
                        out = self.h.check_keys(doubt.doubt)
                        if doubt.doubt.kind == "stall" and getattr(out, "what", None) in ("followed", "switched"):
                            self._progress = time.monotonic()  # (her keys put right: the stretch starts again)
                    except LookLost as lost:  # (a look of the check's failed its checks: stand still, as above)
                        lost_at = time.monotonic()
                        back = self.h.senses.regain(lost, end)
                        self._lost_for(time.monotonic() - lost_at)
                    self._paused(time.monotonic() - paused)
                    if back is None:
                        break
                except (ChatDue, StatsDue) as due:  # put off at the start, a quiet moment now: her chat box's
                    paused, back = time.monotonic(), True  # collapse (behavior.chatcheck), her stat window's read
                    try:                                   # (behavior.statcheck)
                        if isinstance(due, ChatDue):
                            self.h.collapse_chat(look=due.look)
                        else:
                            self.h.read_stats()
                    except LookLost as lost:  # (a look of the read's failed its checks: stand still, as above)
                        lost_at = time.monotonic()
                        back = self.h.senses.regain(lost, end)
                        self._lost_for(time.monotonic() - lost_at)
                    self._paused(time.monotonic() - paused)
                    if back is None:
                        break
            if raises is not None:  # (her way off the ground and to stand by: a doubt there holds, as elsewhere)
                self.h.doubts_raise = raises
            self._leave_ground()
        finally:
            looter.yield_to, looter.unseen = yield_to, about
            if attacks is not None:
                attacks.unseen = None
            self.h.senses.regains = regains
            if raises is not None:
                self.h.doubts_raise = raises
        self._over_at = time.monotonic()  # (her kills whose EXP showed by now are the hunt's: _book_after)
        self.report.seconds = time.monotonic() - start
        if (last := getattr(self, "_noted", None)) is not None:  # (the last pass's rule had her to time-up)
            self.report.rule_s[last[0]] += time.monotonic() - last[1]
            self._noted = None
        self._log_memory()
        self.report.stop = self.report.stop or "time up"
        standby = time.monotonic()
        self.tour.stand_by(mode=TRANSIT)
        self.report.standby_s = time.monotonic() - standby
        self._book_after()  # (her kills on the way to stand by: the run's too, HuntReport)
        self._summary(meter, read)
        return self.report

    def _summary(self, meter, read: int | None) -> None:
        """The run's figures that the hunter counted (HuntReport): her EXP number's rise since `read` (as read at the
        start; None: not read then), her throws, touches and potions (the hunter's report, a new one at the start), her
        lowest HP; the telemetry 'summary' record."""
        r, hr = self.report, self.h.report
        if meter.since is not None:
            r.exp_read = meter.total - (read or 0)
        r.casts = max(r.throws, getattr(hr, "throws", 0))
        r.touches, r.misses = getattr(hr, "touches", 0), getattr(hr, "misses", 0)
        r.drinks, r.lowest_hp = getattr(hr, "drinks", 0), getattr(hr, "lowest_hp", None)
        r.max_hp = getattr(self.h, "max_hp", None)
        telemetry.emit("summary", **r.record())

    def _kill_exp(self, k, avg: float) -> tuple[str | None, float]:
        """A kill's kind and EXP: its kind's, placed or named by its gain's size (localization.gains), else `avg`."""
        kind = k.kind or k.named
        return kind, (self.tour.exp.get(kind, avg) if kind is not None else avg)

    def _count_kill(self, kind: str | None, exp: float) -> None:
        self.report.exp_gains += 1
        self.report.exp += exp
        if kind is not None:
            self.report.kills[kind] = self.report.kills.get(kind, 0) + 1

    def _book_after(self) -> list:
        """After her trip to stand by: the kills booked since the hunt loop's last look. Those whose EXP showed before
        the loop was over (`_over_at`; Kill.gained, its frame's time) are the hunt's own, booked as its looks book
        them, from her last look (_book_kills: a visit's or on her way, a grind's credit by platform): the loop's last
        pass can end on one its looks hadn't had yet (a throw dropped for a kill its last glance saw come in:
        loop_tree_grind's Green at 180.3 s, until the review of 2026-11-02 '1 on her way to stand by' and no
        platform's credit). The rest she made on her trip to stand by (the navigator's shots clearing its way),
        counted in the report as on her way. Returns those."""
        kills = self.h.ledger.kills[self._kills_seen:]
        self._kills_seen = len(self.h.ledger.kills)
        over, s = getattr(self, "_over_at", None), getattr(self, "_s", None)
        own = [k for k in kills if over is not None and s is not None and k.gained is not None and k.gained <= over]
        if own:
            self._book_kills(s, own)
        after = [k for k in kills if not any(k is o for o in own)]
        avg = self._avg_exp()
        for k in after:
            self._count_kill(*self._kill_exp(k, avg))
            self.report.on_the_way += 1
        self.report.standby_kills = len(after)
        return after

    def _lost_for(self, seconds: float) -> None:
        """Her looks lost for `seconds` (LookLost: Senses.regain): past FORGET_LOST_S the map's spawn clock forgets the
        passes it saw (the game back may be another server's or channel's, its pass elsewhere: the review of
        2026-11-03), the spawn guard off until it's sure again."""
        clock = getattr(self.h, "spawn_clock", None)
        if clock is not None and seconds > FORGET_LOST_S and clock.passes:
            clock.forget()
            self.log(f"  the spawn clock forgets its passes (looks lost {seconds:.0f} s)")

    def _paused(self, seconds: float) -> None:
        """She stood still `seconds` (LookLost: Senses.regain): her clocks move on by as much, so the pause isn't time
        without progress (STUCK_S: a pause of 125 s ended the hunt as stuck on the first look after it, in the
        simulator's loop_tree_run), on an objective (OBJECTIVE_S, its trip's: TripState.paused), at a ground (its EXP a
        second on disk, MIN_STAY, GROUND_S), on her spot (SPOT_REST), idle (her patience) or on a rope (HANG_MAX_S,
        ROPE_AGAIN_S). The game's own clocks (respawns, drops' lives) ran on meanwhile and aren't moved."""
        self._progress += seconds
        if (last := getattr(self, "_noted", None)) is not None:  # (the pause is no rule's time)
            self._noted = (last[0], last[1] + seconds)
        for o in [self.obj, *self.stack]:
            if o is not None:
                o.started += seconds
                if o.trip is not None:
                    o.trip.paused(seconds)
        if (v := self.visit) is not None:
            v.spot_since += seconds
            if v.arrived is not None:
                v.arrived += seconds
            if v.empty_since is not None:
                v.empty_since += seconds
        if self._idle_since is not None:
            self._idle_since += seconds
        if self._hanging is not None:
            self._hung_at += seconds
        if self._rope_rest:
            self._rope_rest += seconds

    def _pass(self) -> str | None:
        """One look and one step."""
        s = self._look()
        if s.where is None or s.platform is None:
            if self._on_refuge(s):  # waiting on a rope (rule 9)
                return self._while_hanging(s)
            self._hanging = None
            if s.where is not None:  # in the air, or on a rope
                self.nav._onto_platform(s)
            else:
                time.sleep(0.2)
            return None
        self._hanging = None  # on a platform: off any rope she waited on
        self._book(s)
        self._s = s
        self._publish()
        if self._stuck():
            return "stop"
        self.h.health_check(s)  # her potions, and the safety stop (behavior.safety: it raises)
        o = self.obj
        if o is not None and o.age() >= OBJECTIVE_S:
            telemetry.emit("event", kind="backstop", text=f"OBJECTIVE_S: {OBJECTIVE_S:.0f} s on {o.text()}")
            self.log(f"  {o.text()}: {OBJECTIVE_S:.0f} s on it, suspensions included: dropped")
            return self._drop_objective(o, "backstop")
        if o is not None and o.kind == "pile" and o.pile is not None and self._worthless(o.pile):
            # Seen gone, or the inventory filled up on the way: not worth the rest of the trip.
            self.obj = self.stack.pop() if self.stack else None
            return None
        if o is not None and o.why == "valuable pile" and o.pile is not None and (why := self._unsafe(s, o.pile)):
            return self._call_off(o, why)  # a monster would meet her there: it waits, still known
        if o is not None and o.kind == "spot" and o.platform == o.ground == s.platform.id:
            # A ground no standable spot reaches is hunted on itself: anywhere on it she's there (stance
            # closes in). Waiting to reach the spot's x, she was interrupted by close monsters for good.
            self._arrived(o)
        now = time.monotonic()
        targets, visible, hidden = self._targets(s, now)
        if self._close_quarters(s, targets):
            return self._shoot(s, targets, visible, hidden, now, "close quarters", approach=False)
        if self.fs.unsure and self._close_quarters(s, self.fs.unsure) \
                and self._stance(s, targets, visible, hidden, approach=False, log=False)[0] != "wait":
            return self._shoot(s, targets, visible, hidden, now, "close quarters", approach=False)  # (kept off, unshot)
        if (pile := self._valuable_pile(s)) is not None and self._loot_first(s, pile, targets):
            self.log(f"  {_contents(pile)} on platform {pile.platform} at x {pile.x:.0f} (worth {pile.value:.0f}): "
                     f"picking it up now")
            return self._go_for(pile, "valuable pile", s)
        if self._owe_move and self.obj is not None:
            return self._travel("move on first")
        if self.obj is not None:  # travelling: shoot in passing, an urgent pile, else on its way
            return self._while_travelling(s, targets, visible, hidden, now)
        return self._at_ground(s, targets, visible, hidden, now)

    def _while_travelling(self, s: Scene, targets, visible, hidden, now: float) -> str | None:
        o = self.obj
        if o.why == "valuable pile":  # at once: only close quarters (above), her HP and what's in reach that goes
            if not self._loot_first(s, o.pile, targets):  # first now (EXP before loot) stop her
                return self._call_off(o)  # the pile waits
            return self._travel("valuable pile")
        # In passing she shoots her targets, and steps out from under any wanted monster above her (which isn't a
        # target, out of reach: as _interrupt has them, or a trip stopped for one went on, then again). Those are
        # only stepped from, never thrown at, and not one seen fading out where a kill died (this look's kill is
        # among `visible`, which was taken before it was counted): thrown at, a kill took a star, and then its
        # "hidden" ghost more (fight.HIDDEN_FOR), 4 of them after each kill in loop_finishes_what_she_hurt.
        fading = self.fs.fading(visible)
        others = [m for m in visible if (self.kinds is None or m.kind in self.kinds)
                  and not any(m is t for t in targets) and not any(m is f for f in fading)]
        if self._can_pass_shoot(s, targets, others):
            return self._passing_shot(s, targets, others, visible, hidden, now)
        if (owed := self._owed(s, visible + hidden)) is not None:
            return self._finish(s, owed, visible, hidden, now)
        if o.kind == "pile" and o.why == "sweep":  # on her way to one of the sweep's piles
            return self._on_sweep(s, visible, hidden, now)
        if (stop := self._sweep(s, self._dest(o))) is not None:  # the piles worth their time on her way
            return self._sweep_act(s, stop, visible, hidden, now)
        return self._travel(self._rule if self._rule in ("advance", "high ground", "back to spot", "better spot")
                            else "travel")

    def _at_ground(self, s: Scene, targets, visible, hidden, now: float) -> str | None:
        v, here = self.visit, s.platform.id
        leaving = self._exit_reason is not None  # she decided to go (`_leave`)
        if not leaving and not self._standable(here) and v.spot.platform != v.ground \
                and time.monotonic() >= self._no_high_until:
            return self._to_spot(v.spot, "high ground")
        # Off the ground and its spot (after a pile elsewhere): back, unless she can stand here with the
        # ground in reach, or she's leaving it. Without this she hunted the pile's platform under the old
        # visit. Shooting (rule 3) is only from where she can stand, or on the ground or its spot.
        safe_here = here in (v.ground, v.spot.platform) or self._standable(here)
        if not leaving and here not in (v.ground, v.spot.platform) and time.monotonic() >= self._no_high_until \
                and not (safe_here and any(m.platform == v.ground and self._hits(s, m) for m in targets)):
            return self._to_spot(v.spot, "back to spot")
        if safe_here and any(self._hits(s, m) or (m.platform == here) for m in targets):
            act = self._stance(s, targets, visible, hidden, approach=True, log=False)
            if act[0] != "wait":
                return self._shoot(s, targets, visible, hidden, now, "shoot", approach=True)
        wide = self._wide(s)
        if not leaving and (spot := self._better_spot(wide)) is not None:
            v.changes += 1
            return self._to_spot(spot, "better spot")
        # Away from the ground and its spot (a pile elsewhere), the decision stands: from there she can't
        # judge the ground (out of view, it looked empty, and was skipped for nothing).
        away = here not in (v.ground, v.spot.platform)
        reason = self._exit_reason if leaving and away else self._leaving(wide)
        if reason is None:
            self._exit_reason = self._depart = None  # staying after all
            if (stop := self._sweep(wide, (v.spot.platform, v.spot.x))) is not None:  # piles worth a round trip
                return self._sweep_act(wide, stop, visible, hidden, now)
            return self._wait(wide, targets)
        self._exit_reason = reason
        return self._leave(wide, reason, visible, hidden, now)

    def _leave(self, s: Scene, reason: str, visible=(), hidden=(), now: float | None = None) -> None:
        """Leaving the ground: the sweep's piles on her way to the next one first (`_next_dest`: its spot, within
        DEPART_BUDGETS over the whole way: `_sweep_terms`), then rule 11, the next ground."""
        dest = self._next_dest(s)
        if DEPART_BUDGETS is not None and getattr(self, "_depart", None) is None and s.platform is not None:
            direct = (self._arrive(s.platform.id, s.where.x, *dest) or 0.0) if dest is not None else 0.0
            self._depart = (time.monotonic(), direct + DEPART_BUDGETS * sweep.budget_for(direct))
        if (stop := self._sweep(s, dest)) is not None:
            return self._sweep_act(s, stop, list(visible), list(hidden), time.monotonic() if now is None else now)
        self.log(f"  leaving platform {self.visit.ground}: {reason}")
        self._next_ground(s)
        return self._travel("advance")

    # --- looking and bookkeeping ------------------------------------------------------------------

    def _look(self) -> Scene:
        """A look at the band around her, with her platform's drops; at the whole screen, with the drops on every
        platform in view, once a kill's of hers on another platform have landed since the last such look
        (`_far_drops`): from her spot the band's drops are her own platform's, and a valuable drop on the ground
        she shoots wasn't seen until she stopped shooting (rule 6a goes for it at once)."""
        wide = self._far_drops()
        s = self._seen = self.h.look(mobs=True, wide=wide, loot=True)
        if wide and s.where is not None and s.platform is not None:
            self._wide_at = time.monotonic()
        if telemetry.recording() and s.where is not None:
            telemetry.emit("monsters", **tracks_record(self._tracker.update(s.t, s.monsters), self.model))
        return s

    def _far_drops(self) -> bool:
        """Whether a kill's drops of hers, on another platform than the last look put her on, have landed (SETTLE
        after it) since the last whole-screen look."""
        s, now = self._s, time.monotonic()
        here = None if s is None or s.platform is None else s.platform.id
        return FAR_DROPS_LOOK and any(p.hers and p.status == "expected" and p.platform != here
                                      and self._wide_at < p.since + SETTLE <= now for p in self.h.drops.known)

    def _wide(self, s: Scene) -> Scene:
        """A whole-screen look (monsters and drops on every platform in view), at most every WIDE_EVERY s."""
        if time.monotonic() - self._wide_at < WIDE_EVERY:
            return s
        wide = self._seen = self.h.look(mobs=True, wide=True, loot=True)
        if wide.where is None or wide.platform is None:  # dot hidden, or in the air: it searched no monsters
            return s
        self._wide_at = time.monotonic()
        return wide

    def _book(self, s: Scene) -> None:
        """New kills in the ledger (_book_kills), and progress."""
        kills = self.h.ledger.kills[self._kills_seen:]
        self._kills_seen = len(self.h.ledger.kills)
        self._book_kills(s, kills)
        # A pick-up is progress once a look sees the pile she went for gone (standing on one proves
        # nothing: someone else's, or a full inventory). Counted so, the stuck guard can fire.
        gone = self.h.drops.gone_tried
        if gone > self._gone:
            self._gone, self._progress = gone, time.monotonic()
        key = (s.platform.id, id(self.obj))
        if key != self._fs_key:  # a new place or objective: what she was throwing at there is forgotten, but not the
            # kills dying there and her throws' paths (a kill's pile becomes the objective while its monster is still
            # drawn, and that was shot at whole, then its "hidden" ghost: 16 of the 29 throws at the dead live on
            # 2026-10-25 that came after she had the kill, analysis-1025/jumps-perception-skeptic/wasted2.py)
            old = self.fs
            self.fs = FightState(exp_counted=self.h.report.exp_kills, dying=old.dying, keyed=old.keyed)
            self._fs_key = key

    def _book_kills(self, s: Scene, kills: list) -> None:
        """Kills of the hunt's, her look `s`'s: the visit's EXP and kill times, the report, and progress. A kill's EXP
        is its kind's, placed or named by its gain's size (localization.gains), else the average kind's worth
        hunting."""
        avg = self._avg_exp()
        for k in kills:
            kind, exp = self._kill_exp(k, avg)
            self._count_kill(kind, exp)
            v = self.visit
            mine = v is not None and v.arrived is not None and k.t >= v.arrived and (  # not on her way there
                (k.placed and (k.platform == v.ground  # on the ground, or a far kill from it or its spot
                               or (self.obj is None and s.platform.id in (v.ground, v.spot.platform))))
                or (not k.placed and k.ground in (v.ground, v.spot.platform)))  # unplaced: where she was
            if mine:
                v.exp += exp
                if k.placed and k.platform == v.ground:
                    v.kills.append(time.monotonic())
            else:
                self.report.on_the_way += 1
            self._progress = time.monotonic()

    def _log_memory(self) -> None:
        """What her memory of the monsters did over the hunt (the end of the run's log)."""
        c = self.h.memory.counts
        self.log(f"  memory: {c['new']} monsters seen, {c['killed']} killed by her, {c['gone']} forgotten by looks, "
                 f"{c['aged']} by age")
        if said := self.h.attacks.summary():  # her throws by attack (behavior.attacks)
            self.log(f"  attacks: {said}")
        if (save := getattr(self.h.attacks, "save", None)) is not None:
            save()  # (her damage as the bars showed it, kept: behavior.damage)
        if (darts := getattr(self.h, "darts", None)) is not None and (said := darts.summary()):
            self.log(f"  {said}")  # her presses that took no MP, the most in a row (behavior.darts)
        watch = getattr(getattr(self.h, "attacks", None), "watch", None)
        if watch is not None and (said := watch.summary()):
            self.log(f"  {said}")  # her regular presses judged, after a doubt (behavior.keycheck)

    def _remembered(self, s: Scene, platform: int | None) -> list[Monster]:
        """The wanted monsters her memory has on `platform` (None: anywhere) as of look `s`, sure enough to count
        (memory.SURE): this look's among them."""
        return self.h.memory.there(s.t, platform, self.kinds)

    def _avg_exp(self) -> float:
        worth = [self.tour.exp[k] for k in (self.kinds or ())]
        return sum(worth) / len(worth) if worth else 10.0

    def _targets(self, s: Scene, now: float):
        """What's there to shoot: wanted monsters on the ground she hunts, anywhere in reach (far), on her
        platform, and any monster in her way there (fight._targets, with its hidden and fading ones)."""
        ground = {self.visit.ground} if self.visit is not None else None
        targets, visible, hidden = fight._targets(self.h, self.fs, s, self.kinds, ground, True, now)
        self.h.looter.pending.clear()  # drops are the drop store's here (behavior.ledger)
        return self._with_unseen(s, targets, visible, hidden)

    def _with_unseen(self, s: Scene, targets, visible, hidden):
        """Her targets, the monsters seen and the hidden ones (fight._targets'), with the monsters about her that look `s`
        didn't find (`_unseen`: the feed's hidden tracks, her memory's of the last UNSEEN_S, none she threw at already):
        among the hidden ones, so the stance's danger and moves go by them (its `around`), and those on her platform of
        a kind that counts as a danger among the targets too where a throw is worth it (fight.shots_at), so a fight keeps
        off them and shoots them (rule 1 and stance: one on her platform that can touch her first; where no throw is, it
        keeps off them alone: fs.unsure). Not those on other platforms: thrown at where they were
        last seen, they'd be the throws at what's dead or gone that the live hunt of 2026-10-25 wasted 52 of 325 on
        (jumps-perception-skeptic). And the monsters seen on a platform just under or over hers whose bodies meet hers
        (stance's `beside`, BESIDE) among the targets, whatever her ground: her fights keep off them as off one on hers
        (rule 1, the stance's close quarters) and throw at them where a star reaches. Where nothing counts as a danger
        (the first map), as they were."""
        d = getattr(self.h, "danger", None)  # (tests make a hunter without one)
        if s.platform is None or s.where is None or d is None or not d.any:
            return targets, visible, hidden
        here, x = s.platform, s.where.x
        near = [m for m in visible if m.platform != here.id and beside(m, here, x, d)
                and not any(m is t for t in targets)]
        if near:
            targets = targets + near
        more = [m for m in self._unseen(s) if not any(_same(m, h) for h in [*hidden, *visible])]
        if not more:
            return targets, visible, hidden
        mine = [m for m in more if m.platform == here.id and d.share(m.kind) > 0
                and not any(_same(m, t) for t in targets)]
        shot, unsure = fight.shots_at(self.h, s, mine, [m for m in targets if any(m is v for v in visible)])
        self.fs.unsure = [*self.fs.unsure, *unsure]
        return targets + shot, visible, hidden + more

    def _unseen_here(self, s: Scene) -> list[Monster]:
        """Those of `_unseen` on her platform in look `s` whose kind counts as a danger (her throws' attack goes by them:
        behavior.attacks, UNSEEN_ATTACK), where something does."""
        d = getattr(self.h, "danger", None)
        if s.platform is None or d is None or not d.any:
            return []
        return [m for m in self._unseen(s) if m.platform == s.platform.id and d.share(m.kind) > 0]

    def _unseen(self, s: Scene) -> list[Monster]:
        """The monsters about that look `s` didn't find (localization.unseen: the feed's hidden tracks that count, none
        a kill of hers in the ledger is on, her memory's seen within UNSEEN_S), worked out once a look."""
        if not UNSEEN_FIGHTS:
            return []
        if getattr(self, "_unseen_for", None) is not s:
            ledger = getattr(self.h, "ledger", None)
            self._unseen_for, self._unseen_list = s, unseen(s, getattr(self.h, "memory", None), self.model, UNSEEN_S,
                                                            kills=ledger.kills if ledger is not None else ())
        return self._unseen_list

    @property
    def danger(self):
        """The map's danger for her (attack.danger, the hunter's; None: none known)."""
        return self.h.danger

    def _hits(self, s: Scene, m: Monster) -> bool:
        """Whether an attack from where she stands in look `s` hits `m`: a star from the ground (reach.can_hit: in
        reach, and nothing stops its star, nor that of a monster of the look the attack takes first), or one from a
        jump that lands her back on her platform (_jump_hits)."""
        return self._star_hits(s, m) or self._jump_hits(s, m)

    def _star_hits(self, s: Scene, m: Monster) -> bool:
        """Whether a star from the ground where she stands in look `s` hits `m` (reach.can_hit)."""
        return reach.can_hit(s.platform, s.where.x, s.where.y, m, others=s.monsters)

    def _jump_hits(self, s: Scene, m: Monster) -> bool:
        """Whether the attack from one of her jumps from where she stands in look `s` surely hits `m` (attack.reach's
        coverage, every monster of the look about her: _cover) on the stance's terms (stance.jump_hits: the jump
        landing her back where she keeps to however it goes, her body clear of the look's monsters, the danger kept),
        so she doesn't stay to shoot what the stance won't jump at; False with fight.JUMP_SHOTS off."""
        cover = self._cover(s, [m])
        if cover is None or not any(sh.jump is not None for sh in cover().of(m)):
            return False
        actions = getattr(self.h, "actions", None)
        return jump_hits(s.where.x, s.where.y, s.platform, m, cover(), getattr(actions, "facing", None),
                         s.monsters if any(o is m for o in s.monsters) else [*s.monsters, m], self.danger,
                         getattr(self, "_halves", None))

    def _cover(self, s: Scene, targets, around=None):
        """What her attacks reach from where she stands in look `s` at `targets` (fight.jump_cover, as a callable worked
        out when first asked; `around`: the monsters about her, the look's by default), kept for the look: the look's
        own monsters and every other asked of it share one."""
        if not fight.JUMP_SHOTS or not fight._air() or s.where is None or s.platform is None:
            return None
        if getattr(self, "_covers_for", None) is not s:  # (tests make a hunt without __init__)
            self._covers_for, self._covers = s, {}
        around = s.monsters if around is None else around
        key = (tuple(id(m) for m in around),)
        mine = [m for m in targets if not any(m is o for o in around)]
        if mine:  # a monster not of the look (her memory's, a hidden one): its own
            key += tuple(id(m) for m in mine)
        got = self._covers.get(key)
        if got is None:
            got = self._covers[key] = fight.jump_cover(self.model, s, list(around) + mine, list(around) + mine)
        return got

    def _standable(self, pid: int) -> bool:
        """Whether she can stand on platform `pid` to shoot: spots.spot_ok, or it's the visit's spot's (taken as one:
        a place better than hunting on the ground by the map's danger, or the ground itself where none is)."""
        v = self.visit
        return spots.spot_ok(self.model, pid, self.kinds) or (v is not None and pid == v.spot.platform)

    def _stuck(self) -> bool:
        if time.monotonic() - self._progress < STUCK_S:
            return False
        watch = getattr(getattr(self.h, "attacks", None), "watch", None)
        now, asked = time.monotonic(), getattr(self, "_stall_at", None)
        if STUCK_CHECKS and getattr(self.h, "doubts_raise", False) and watch is not None:
            # (her keys moved in the game may be why: the key bindings read first, behavior.keycheck, once in STUCK_S;
            # followed, the hunt goes on from here; its check put off, STALL_WAIT_S more for a quiet moment)
            if (asked is None or now - asked >= STUCK_S) and watch.may_check(now) is None:
                self._stall_at = now
                raise KeyDoubt(Doubt.stall(f"NO KILL, PICK-UP OR NEW GROUND FOR {STUCK_S:.0f} S: her keys may be on "
                                           "other keys in the game", now))
            if (asked is not None and now - asked < STALL_WAIT_S and watch.deferred is not None
                    and watch.deferred.kind == "stall"):
                return False
        self.log(f"  no kill, pick-up or new ground for {STUCK_S:.0f} s: standing by")
        telemetry.emit("event", kind="backstop", text=f"STUCK_S: {STUCK_S:.0f} s without progress")
        self.report.stop = "stuck"
        return True

    # --- shooting -----------------------------------------------------------------------------------

    def _close_quarters(self, s: Scene, monsters) -> bool:
        """Rule 1: one of `monsters` on her platform (or on one just under or over it whose body meets hers: stance's
        `beside`, BESIDE) is as close as stance's own threshold, where it will be when her next attack starts
        (too_close, fight.pace_of: the claw would swipe it, by its size; or nearer than its danger warrants:
        Danger.keep_off)."""
        x, pace = s.where.x, fight.pace_of(self.h, s)
        facing = getattr(getattr(self.h, "actions", None), "facing", None)
        return any(beside(m, s.platform, x, self.danger)
                   and too_close(m, x, self._halves, self.danger, pace, s.platform, (m.x > x) != (facing == RIGHT))
                   for m in monsters)

    def _owed(self, s: Scene, around: list[Monster]) -> list[Monster] | None:
        """Rule 4b: with an objective on another platform (she's set off, or about to: every rule that takes her
        away makes one), and done shooting in passing, the monsters she has hurt and not killed (fight.owed)
        that a throw from here, or a walk of fight.FINISH_WALK at most along her platform, gets to
        (fight.finish_act): what to shoot at to finish them (fight.finish_targets), else None. In the live hunt
        loops she set off for piles, spots and the next ground with monsters half hunted (fight.FINISH_WALK).
        It ends with the kill, once they're out of reach from near her, out of sight, or chased
        ledger.WOUND_S, or with the objective's FINISH_S. Going along her own platform (a pile on it) she
        leaves nothing behind, and at the ground itself shooting (rule 3) goes on as it did: in the simulator's
        basin, where respawns never stop and one she had hurt was always about, finishing before the piles on
        it ran them out of their OBJECTIVE_S."""
        o = self.obj
        if o is None or o.platform == s.platform.id or o.finishing >= FINISH_S:
            return None
        wounds = fight.owed(self.h, s, self.fs)
        if fight.finish_act(self.h, s, wounds, around) is None:
            return None
        return fight.finish_targets(s, wounds, around)

    def _finish(self, s: Scene, owed: list[Monster], visible, hidden, now: float) -> None:
        """Rule 4b: one step at the monsters she owes (_owed), counted against the objective's FINISH_S."""
        t0, o = time.monotonic(), self.obj
        self._shoot(s, owed, visible, hidden, now, "finish", approach=True)
        o.finishing += time.monotonic() - t0
        if o.finishing >= FINISH_S:
            telemetry.emit("event", kind="backstop", text=f"FINISH_S: {FINISH_S:.0f} s finishing for {o.text()}")
        return None

    def _stance(self, s: Scene, targets, visible, hidden, approach: bool, log: bool = True,
                goal: float | None = None) -> tuple:
        """stance's answer for `targets` in look `s` (fight._take_stance's with `log`), with what her jumps' attacks
        reach (_cover: the same for both) and, travelling, where she's going on her platform (`goal`, map x: _toward),
        the way she goes."""
        jumps = self._cover(s, targets, visible + hidden)
        toward = 0 if goal is None else (1 if goal > s.where.x else -1)
        edges, up = self._edges(s)
        limits = fight.move_room(self.h, s)  # (where something counts as a danger, as `edges`: fight.MOVE_LIMITS)
        among = self._among(visible, hidden)
        if log:
            return fight._take_stance(self.h, self.fs, s, targets, visible, hidden, approach, toward=toward,
                                      jumps=jumps, goal=goal, edges=edges, up=up, limits=limits, among=among)
        here = s.platform
        castable = fight.castable_of(self.h)
        return fight.cast_or_wait(self.h, stance(
            s.where.x, s.where.y, here, targets, approach, self.h.actions.facing,
            can_jump=lambda a, b: open_above(self.model, here, a, b), tops=self._tops, halves=self._halves,
            around=visible + hidden, danger=self.danger, jumps=jumps, toward=toward, goal=goal, edges=edges, up=up,
            pace=fight.pace_of(self.h, s), touch=self.fs.touch, limits=limits, moves=self.fs.moves,
            castable=castable, clawed=fight.clawed(self.h, s, visible + hidden), unsure=self.fs.unsure, among=among,
            **fight.wounds_of(self.h, s)), s, castable)

    def _among(self, visible, hidden) -> list[Monster] | None:
        """The stance's `among` (behavior.grind's target-first planner: the others about her; the hunt loop's none)."""
        return None

    def _edges(self, s: Scene) -> tuple:
        """The stance's `edges` and `up` for her in look `s` (attack.stance: drop_limits, overhead): where something
        counts as a danger; (None, None) where nothing does (the first map, as it was)."""
        d = getattr(self.h, "danger", None)
        if d is None or not d.any or s.platform is None or getattr(self, "model", None) is None:
            return None, None
        here, model = s.platform, self.model
        return drop_limits(model, here), (lambda x: overhead(model, here, x))

    def _shoot(self, s: Scene, targets, visible, hidden, now: float, rule: str, approach: bool) -> None:
        """One step of stance's answer: a throw (fight's bookkeeping kept), or a walk."""
        self._note(rule)
        act = self._stance(s, targets, visible, hidden, approach)
        if fight.knocked_move(self.h, s, act):
            return None
        if act[0] in THROWS:
            if not fight.throw_act(self.h, s, act):  # dropped at its last look before the key: the next pass decides
                return None
            if fight.keyless(self.h):  # (her jump away alone: a move)
                self.fs.moves += 1
                self._idle_since = None
                return None
            self.fs.note_throw(now, self.h.actions.attack_at)
            self.fs.moves = 0
            self.h.report.throws += 1
            self.report.throws += 1
            if self.visit is not None:
                self.visit.throws += 1
            self._idle_since = None
        elif act[0] in ("back", "approach"):
            self.fs.moves += fight.moved_away(act)
            self.h.actions.walk(act[1], min(act[2], 300.0))
            self._idle_since = None
        return None

    def _can_pass_shoot(self, s: Scene, targets, others=()) -> bool:
        """Rule 4: a wanted target in reach while travelling, or one a short step brings into reach (_pass_step; of
        the targets and `others`, wanted monsters in view that aren't targets), with this platform's budget not
        spent."""
        if self.obj is None or self.obj.passing.get(s.platform.id, 0.0) >= PASSING_S:
            return False
        return any(self._hits(s, m) for m in targets) or self._pass_step(s, [*targets, *others]) is not None

    def _toward(self, s: Scene) -> float | None:
        """Where along her platform her trip goes next (map x; None: not known, not along it, or under MIN_MOVE off):
        the objective's place on her platform, else where the navigator's route leaves it (its first move's takeoff).
        Its way is the way she's travelling (stance's `toward`); a jump forward in passing lands no farther than it
        (stance's `goal`, PASS_SLACK)."""
        o = self.obj
        if o is None or s.where is None or s.platform is None:
            return None
        x = s.where.x
        if o.platform == s.platform.id:
            goal = o.x
        else:
            goal = self.nav.leaves_at(s.platform, x, o.platform, o.x)
        if goal is None or abs(goal - x) < MIN_MOVE:
            return None
        return goal

    def _pass_step(self, s: Scene, targets) -> tuple | None:
        """In passing, a step along her platform out from under a target above her too close sideways for the
        star's search to pick it (stance's step from under, never closing in), PASS_STEP at most. From 22 a star
        at a monster on 21 (60 above) needs 111-126 units sideways, and 21's right end is 12 units left of 22's
        left end: one near it is in reach only from x 60-75 on 22, and coming up she stands on 22's left part,
        where her way on jumps onto 21 at that end. In the live hunt loops of 2026-10-21, of the 3 times she went
        on from 22 with a monster on 21 in view, 2 had it under the search (77 and 103 units off), and she jumped
        up beside it. Not from over one below her (she shoots those from the spots above them as she gets
        there): with those too, in the simulator's hunt loop (60 runs of 20 minutes) she picked up 88.5% of her
        drops, against 89.0% so and 90.1% without the step."""
        above = [m for m in targets if m.y < s.where.y - reach.LEVEL]
        if not above:
            return None
        here = s.platform
        act = stance(s.where.x, s.where.y, here, above, False, self.h.actions.facing,
                     can_jump=lambda a, b: open_above(self.model, here, a, b), tops=self._tops, halves=self._halves,
                     around=s.monsters, danger=self.danger, pace=fight.pace_of(self.h, s))
        return act if act[0] == "back" and act[2] <= PASS_STEP else None

    def _passing_shot(self, s: Scene, targets, others, visible, hidden, now: float) -> None:
        """Rule 4: a throw at `targets` from where she is, else a step out from under one of them or of `others`
        above her (_pass_step), else on her way."""
        t0 = time.monotonic()
        self._note("in passing")
        act = self._stance(s, targets, visible, hidden, approach=False, goal=self._toward(s))
        if fight.knocked_move(self.h, s, act):
            return None
        if act[0] in THROWS:
            if fight.throw_act(self.h, s, act):  # (dropped at its last look before the key: the next pass decides)
                if fight.keyless(self.h):  # (her jump away alone: a move)
                    self.fs.moves += 1
                    return None
                self.fs.note_throw(now, self.h.actions.attack_at)
                self.fs.moves = 0
                self.h.report.throws += 1
                self.report.throws += 1
        elif (step := self._pass_step(s, [*targets, *others])) is not None:  # out from under one: a star picks it
            if step != act:
                self.h.log(f"  {act_text(step, [*targets, *others], s.where.x, hidden)} (HP {s.hp:.0%})")
            self.h.actions.walk(step[1], step[2])
        else:  # nothing to throw from here without walking after it: on her way
            self.obj.passing[s.platform.id] = PASSING_S
        pid = s.platform.id
        self.obj.passing[pid] = self.obj.passing.get(pid, 0.0) + (time.monotonic() - t0)
        if self.obj.passing[pid] >= PASSING_S:
            telemetry.emit("event", kind="backstop", text=f"PASSING_S: {PASSING_S:.0f} s shooting on platform {pid}")
        return None

    # --- loot: the sweep (from 2026-10-30) ------------------------------------------------------------------------

    def _worthless(self, p: Pile) -> bool:
        """A pile she's going for that isn't worth the rest of the trip: seen gone, or nothing in it she can take (its
        items' tabs full: `_takes`). Not its age: how likely it is to lie there still weighs it in the sweep
        (Pile.still), and the objective's clock bounds the trip."""
        return p.status == "gone" or not self._takes(p)

    def _takes(self, p: Pile) -> bool:
        """Whether pile `p` holds something she can take: mesos, an item whose tab has room (looting.Looter.can_take;
        her inventory by tab, from 2026-10-31), or, none seen in it yet, while an item tab of hers has room."""
        looter = self.h.looter
        if p.mesos:
            return True
        can = getattr(looter, "can_take", None)
        if can is None:  # (a looter of tests': the one flag)
            return not looter.inventory_full
        if p.items:
            return any(can(k) for k in p.items)
        return not looter.items_full

    def _worth(self, p: Pile) -> float:
        """What pile `p` is worth to her (Pile.value) but for the items she can't take (their tabs full): one no look
        has seen yet (her kill's), its kind's expected drops but those (looting.expected_worth; until the review of
        2026-10-31 all of them, a full tab's items too)."""
        looter = self.h.looter
        can = getattr(looter, "can_take", None)
        if can is None or not looter.inventory_full:
            return p.value
        if p.worth is None and p.kind is not None:
            return expected_worth(p.kind, self.h.drops.value, can)
        if not p.items:
            return p.value
        value = self.h.drops.value
        return max(0.0, p.value - sum(value(k) for k in p.items if k != MESOS and not can(k)))

    def _candidate(self, p: Pile, s: Scene | None = None) -> bool:
        """Whether pile `p` is one to go for as of look `s` (None: her last pass's): not seen gone, on a platform,
        stood on or gone for fewer than PILE_TRIES times (a run through it with no stand not counted), hers or past its killer's ownership (OWN_S), something in it
        she can take (`_takes`: her inventory by tab), not another player's (PLAYER_PILES), likely to lie there still
        (Pile.still, SURVIVE_MIN: ledger.SURVIVAL, where until 2026-10-30 a pile was left at DROP_LIFE, 90 s, and
        another's waited OTHERS_AFTER, 30 s), and not left to her pets for now (`_pets_first`)."""
        now = time.monotonic()
        return (p.status != "gone" and p.platform is not None and p.tries - p.runs < PILE_TRIES
                and (p.hers or now - p.since >= OWN_S) and self._takes(p)
                and not self._theirs(p) and p.still(now) >= SURVIVE_MIN and not self._pets_first(p, now, s))

    def _pets_first(self, p: Pile, now: float, s: Scene | None = None) -> bool:
        """PET_LOOT: whether pile `p` is left to her pets now as of look `s` (None: her last pass's): hers, on the
        platform the look has her on, its stretch (`_extent`) within PET_NEAR of her, and she not yet PET_WAIT s by it
        (from its drops' landing at the soonest: DROP_LANDS after its kill; counted afresh once she's farther). Her time
        by a pile is counted from her own looks alone: one made up from another (Scene.what_if: her put where a chain
        ends) reads it and leaves it as it is. (Until the review of 2026-11-01 always as of her last pass's look: on a
        trip the navigator's looks, and the made-up ones, judged a pile by her from where the trip began, and started or
        reset its count from there.)"""
        s = s if s is not None else getattr(self, "_s", None)  # (where she, and her pets with her, are)
        if not PET_LOOT or not p.hers or s is None or s.platform is None or s.where is None \
                or p.platform != s.platform.id:
            return False
        by = self.__dict__.setdefault("_pet_by", {})  # pile (its key) -> since when she's been by it (tests make a
        # hunt without __init__)
        made_up = getattr(s, "what_if", False)
        lo, hi = self._extent(p)
        if max(lo - s.where.x, s.where.x - hi, 0.0) > PET_NEAR:
            if not made_up:
                by.pop(p.key, None)
            return False
        if made_up:
            return now - by.get(p.key, max(now, p.since + DROP_LANDS)) < PET_WAIT
        if len(by) > PET_KEEP:  # (bounded: the piles gone or picked up dropped)
            known = {q.key for q in self.h.drops.known}
            for k in [k for k in by if k not in known]:
                del by[k]
        return now - by.setdefault(p.key, max(now, p.since + DROP_LANDS)) < PET_WAIT

    def _theirs(self, p: Pile) -> bool:
        """Whether pile `p` is another player's (PLAYER_PILES): no kill of hers explains it, it holds nothing
        valuable, and another player is by it in the latest look."""
        s = self._seen
        return (PLAYER_PILES and not p.hers and (p.dearest or 0.0) < self.valuable and s is not None
                and near_rival(s.players, p.x, p.y) is not None)

    def _tried(self, pile: Pile) -> None:
        self.h.drops.tried(pile)
        self.report.piles += 1

    def _travel_to_pile(self, s: Scene, p: Pile) -> float:
        here = s.platform
        if here is not None and here.id == p.platform:
            return abs(p.x - s.where.x) / WALK_SPEED
        t = self.nav.planner.travel(here.id, s.where.x, p.platform) if here is not None else None
        return 1e6 if t is None else t + 1.0

    def _extent(self, p: Pile) -> tuple[float, float]:
        """The stretch of its platform pile `p`'s drops lie on (map x from, to): where a look saw them, else SPREAD either
        side of where her kill died, kept on its platform."""
        if p.xs:
            lo, hi = min(p.xs), max(p.xs)
        else:
            lo, hi = (p.x, p.x) if p.status == "seen" else (p.x - SPREAD, p.x + SPREAD)
        model = getattr(self, "model", None)
        q = None if model is None else model.platform(p.platform)
        return (lo, hi) if q is None else (max(lo, q.left), min(hi, q.right))

    def _stand_s(self, p: Pile) -> float:
        """How long standing on pile `p` takes: a press of pick-up for each drop and one more on each bunch of them
        (PILE_GAP: picked up from one place), PILE_EVERY_MS apart, and the walk over its stretch (`_extent`)."""
        lo, hi = self._extent(p)
        n = max(1, len(p.xs) or len(p.items))
        bunches = len(_piles(list(p.xs))) if p.xs else 1
        return (n + bunches) * PILE_EVERY_MS / 1000 + (hi - lo) / WALK_SPEED

    def _sweep_terms(self, s: Scene, dest: tuple[int, float] | None) -> tuple[float | None, float]:
        """The sweep's terms on her way to `dest` as of look `s`: a budget for its whole itinerary (None: its piles'
        allowances alone, `_allow`) and the least any pile may add to her way (0: its own allowance). In a lull at the
        ground she's at (a round trip back to its spot, nothing wanted left on it, `_nothing_here`, and a kill of hers
        there due back, `_due_back`), her time is worth nothing until the respawn: that wait, as both (rule 8's, lull
        loot, until 2026-10-30: in the simulator's loop_lull_loot, with a 2 s budget for her round trips from 13 to 12's
        drops, she picked up 5 piles in 3 minutes where 20). Leaving it, what's left of her way out's allowance
        (DEPART_BUDGETS). Until 2026-11-01 her time's worth too (`_time_worth`: the loop's rate, or what's in reach)."""
        v = self.visit
        if dest is not None and v is not None and v.arrived is not None and dest == self._home() \
                and self._nothing_here(s) \
                and (due := self._due_back()) is not None and due > 0.0 and self._shoot_rate(s, list(s.monsters))[0] <= 0:
            return max(sweep.BUDGET_MIN, due), due
        depart = getattr(self, "_depart", None)
        if depart is not None and getattr(self, "_exit_reason", None) is not None and s.platform is not None:
            # (leaving: what's left of her way out's allowance, less the straight way on from here: DEPART_BUDGETS)
            direct = (self._arrive(s.platform.id, s.where.x, *dest) or 0.0) if dest is not None else 0.0
            return max(0.0, depart[1] - (time.monotonic() - depart[0]) - direct), 0.0
        return None, 0.0

    def _allow(self, p: Pile, now: float) -> float:
        """The s pile `p` may add to her way (sweep.Spot.allow) by its age (from its kill, or its first sighting):
        ALLOW_FRESH to FRESH_S, ALLOW_OLD from OLD_S, straight between."""
        age = now - p.since
        if age <= FRESH_S:
            return ALLOW_FRESH
        if age >= OLD_S:
            return ALLOW_OLD
        return ALLOW_FRESH + (ALLOW_OLD - ALLOW_FRESH) * (age - FRESH_S) / (OLD_S - FRESH_S)

    def _dest(self, o: "Objective | None") -> tuple[int, float] | None:
        """Where objective `o` takes her (platform, map x), the sweep's way on; None: none."""
        if o is None:
            return None
        if o.x is not None:
            return o.platform, o.x
        q = self.model.platform(o.platform)
        return o.platform, (q.left + q.right) / 2

    def _next_dest(self, s: Scene) -> tuple[int, float] | None:
        """Leaving the ground: the spot of the best other ground she'd go to (`_best_rate`, no draw: the next ground's
        own choice has one), the sweep's way on; None: none."""
        best = self._best_rate(s) if s.platform is not None else None
        if best is None:
            return None
        sp = spots.best_spot(self.model, best[1], self.kinds, self.danger)
        return sp.platform, sp.x

    def _arrive(self, a: int, ax: float, b: int, bx: float) -> float | None:
        """Her route time from (platform a, x) to standing at x on platform b (Planner.arrive: its searches kept by where
        from, rounded), from where she is to the nearest 5 units (the sweep asks it from every look)."""
        return self.nav.planner.arrive(a, 5.0 * round(ax / 5.0), b, bx)

    def _sweep(self, s: Scene, dest: tuple[int, float] | None, say: bool = True) -> "sweep.Stop | None":
        """The loot sweep's next stop from look `s` on her way to `dest` (platform, x; None: nowhere after it), or None:
        of the drop store's candidates as pick spots (`_spots`: their worth, stands, landings, survival, guards and
        allowances), by her route times (`_arrive`), within the way's terms (`_sweep_terms`: in a lull, the wait for the
        respawn; leaving, her way out's budget; behavior.sweep: the itinerary she's on kept unless another is clearly
        better, sweep.Sweeper). Solved once a look for each `dest`; the itinerary said and recorded (a 'sweep' record)
        when it changes (`say`)."""
        if s.where is None or s.platform is None:
            return None
        key = (s, dest)  # (the look itself, not its id: one gone, another may take its id, and a solve for it be
        # taken for the new look's; a run's choices then went by where in memory its looks were)
        last = getattr(self, "_swept_for", None)
        if last is not None and last[0] is s and last[1] == dest:
            return self._swept
        if not hasattr(self, "_sweeper"):  # (tests make a hunt without __init__)
            self._sweeper, self._pile_of, self._stood, self._sweep_said = sweep.Sweeper(), {}, {}, None
        spots_, out = self._spots(s)
        if not spots_:  # (nothing to pick up: no itinerary, nor its terms worked out)
            self._sweeper.drop()
            self._swept_for, self._swept = key, None
            return None
        budget, floor = self._sweep_terms(s, dest)
        if floor > 0.0:  # (a lull's: every pile may take its wait, but one more dangerous than where she is: until the
            # review of 2026-11-01 that one too, SAFER_MARGIN's cap lifted)
            spots_ = [sp if sp.key in self._riskier else replace(sp, allow=max(sp.allow, floor)) for sp in spots_]
        it = self._sweeper.solve(spots_, (s.platform.id, s.where.x), dest, self._arrive, budget,
                                 None if floor > 0.0 else self._sweep_most())  # (a lull's wait its budget alone)
        it.out.update(out)
        self._swept_for, self._swept = key, it.first
        said = (it.keys(), dest)
        if say and said != self._sweep_said:
            self._sweep_said = said
            if it.stops:
                self.log("  sweep: " + ", then ".join(
                    f"{_contents(self._pile_of[st.spot.key])} on {st.spot.platform} at x {st.spot.x:.0f}"
                    + (f" (a {', '.join(g.monster.kind for g in st.spot.guards)} shot first)" if st.spot.guards else "")
                    for st in it.stops) + f": {it.extra:+.1f} s on her way for {it.value:.0f} mesos' worth")
            if telemetry.recording():
                to = self._sweep_to(dest)
                telemetry.emit("sweep", dest=None if dest is None else [dest[0], round(dest[1])],
                               candidates=len(spots_), **it.record(), **({"to": to} if to else {}))
        return it.first

    def _sweep_most(self) -> float | None:
        """The s an itinerary may add to her way at most, whatever its spots' allowances come to (sweep.MOST_S; behavior.
        grind's target-first planner: their sum alone)."""
        return sweep.MOST_S

    def _sweep_to(self, dest: tuple[int, float] | None) -> dict | None:
        """Why the sweep's way goes on to `dest` (its 'sweep' record's `to`, from 2026-11-02: P32 of the recordings'
        analysis, where a leave-sweep's destination couldn't be told from the records): the objective it's the place of
        (`objective`, and the rule that set it, `why`), or the ground's spot she comes back to (`home`); None: neither.
        behavior.grind's says the next stop's choice first (`peek`)."""
        if dest is None:
            return None
        for o in (getattr(self, "obj", None), *reversed(getattr(self, "stack", ()))):
            if o is not None and self._dest(o) == dest:
                return {"objective": o.text(), **({"why": o.why} if o.why else {})}
        return {"home": True} if self._home() == dest else None

    def _sweep_calls(self, s: Scene) -> bool:
        """Whether the sweep has something else for her than the trip she's on (`_interrupt`): on a trip to a spot, a
        pile on her way; on a trip to one of its piles, another (or none any more, or a guard to shoot first). Taken at
        the next pass. On a look made up from another (Scene.what_if: the navigator's, her put where a chain ends) the
        sweep's itinerary, its piles and what it said stay as they were: until the review of 2026-10-30 such a solve
        took the itinerary she was on (sweep.Sweeper: kept unless another is clearly better) for one from a place she
        wasn't at, and logged its "later, ..." from there. Never on the grind's way back to its stop (behavior.grind's
        `_back`, 'back to spot'): from 2026-10-31 (the rebase onto master 58dca1a) a pile on that way cut the trip,
        whose next pass (`_while_travelling`, which didn't sweep on it then) sent her off again, about 900 times in a
        row on the harness's seed 9 (60 s on 19, the objective's clock its end)."""
        o = self.obj
        if o is None or o.kind not in ("spot", "pile") or (o.kind == "pile" and o.why != "sweep") \
                or o.why == "back to spot" or getattr(o, "chain", None) is not None:
            return False
        keep = None
        if getattr(s, "what_if", False) and hasattr(self, "_sweeper"):
            keep = (self._sweeper.current, self._sweeper.dest, self._pile_of, getattr(self, "_swept_for", None),
                    getattr(self, "_swept", None), getattr(self, "_sweep_said", None))
        try:
            first = self._sweep(s, self._dest(o) if o.kind == "spot" else self._after_pile(), say=False)
            if o.kind == "spot":
                return first is not None
            return first is None or self._pile_of.get(first.spot.key) is not o.pile or bool(first.spot.guards)
        finally:
            if keep is not None:
                (self._sweeper.current, self._sweeper.dest, self._pile_of, self._swept_for, self._swept,
                 self._sweep_said) = keep

    def _after_pile(self) -> tuple[int, float] | None:
        """Where she goes on to after the sweep's pile she's going for: where the itinerary it's of goes on to
        (Objective.after: leaving a ground, the next one), else the objective under it, else back to the ground's spot
        (`_home`)."""
        o = self.obj
        if o is not None and o.kind == "pile" and o.after is not None:
            return o.after
        parent = self.stack[-1] if self.stack else None
        return self._dest(parent) if parent is not None else self._home()

    def _on_sweep(self, s: Scene, visible, hidden, now: float) -> None:
        """On her way to the sweep's pile (the objective): the sweep solved again from where she is on her way on
        (`_after_pile`); the same pile first, on to it (at it, its stand or its landing: `_sweep_act`), else what it has
        now instead (another pile, a guard to shoot first, or nothing: the objective under it)."""
        o = self.obj
        stop = self._sweep(s, self._after_pile())
        pile = None if stop is None else self._pile_of.get(stop.spot.key)
        if pile is not o.pile or stop.spot.guards or self._at_pile(s, pile):
            self.obj = self.stack.pop() if self.stack else None
            return None if stop is None else self._sweep_act(s, stop, visible, hidden, now)
        return self._travel("sweep")

    def _home(self) -> tuple[int, float] | None:
        """Where she comes back to at the ground she's at: its spot (the sweep's round trips)."""
        v = self.visit
        return None if v is None else (v.spot.platform, v.spot.x)

    def _spots(self, s: Scene) -> tuple[list["sweep.Spot"], dict]:
        """The drop store's candidates (`_candidate`) as the sweep's pick spots as of look `s`, and why each pile left
        out was (by its key): each where its drops lie (`_extent`, its middle), its worth (Pile.value), its stand
        (`_stand_s`), s until its drops land (an expected pile's: DROP_LANDS after the kill), its survival from now
        (Pile.still), what's to be shot first (`_guards`, by when she'd be there going now: `_travel_to_pile`) and the s
        it may add to her way (`_allow`; ALLOW_OLD at most where the danger there is SAFER_MARGIN over where she
        stands: attack.danger with the look's monsters)."""
        now = time.monotonic()
        out, found = {}, []
        self._pile_of, self._riskier = {}, set()  # (the piles capped for their danger: a lull's wait doesn't lift it)
        d = getattr(self.h, "danger", None)  # (tests make a hunter without one)
        here = None
        if d is not None and d.any and s.platform is not None and s.where is not None:
            here = d.at(s.platform.id, s.where.x, s.monsters)
        for p in self.h.drops.known:
            if not self._candidate(p, s):
                continue
            lo, hi = self._extent(p)
            stand = self._stand_s(p)
            lands = max(0.0, p.since + DROP_LANDS - now) if p.status == "expected" else 0.0
            guards, why = self._guards(s, p, max(self._travel_to_pile(s, p), lands), stand)
            said = None if getattr(s, "what_if", False) else getattr(self, "_block_said", None)  # (made up: unsaid)
            was = None if said is None else said.get(id(p))
            if said is not None and (was is None or was[0] is not p or was[1] != (why is not None)):
                said[id(p)] = (p, why is not None)  # (said as it starts waiting, and when it ends)
                if why is not None:
                    self.log(f"  {_contents(p)} on platform {p.platform} at x {p.x:.0f}: later, {why}")
            if why is not None:
                out[p.key] = "unsafe: " + why
                continue
            self._pile_of[p.key] = p
            worth = self._worth(p)
            allow = self._allow(p, now) * self._share(p, worth)
            if (whole := self._whole(p, s, now)) is not None:  # (behavior.grind's target-first planner: all of them)
                allow = whole
            elif here is not None and d.at(p.platform, (lo + hi) / 2, s.monsters) > here + SAFER_MARGIN:
                allow = min(allow, ALLOW_OLD)  # (more dangerous there than where she is: on her way only)
                self._riskier.add(p.key)
            found.append(sweep.Spot(p.key, p.platform, (lo + hi) / 2, worth, stand, lands,
                                    (lambda dt, p=p: p.still(now, dt)), guards, allow))
        return found, out

    def _whole(self, p: Pile, s: Scene, now: float) -> float | None:
        """The s pile `p` may add to her way whatever it's worth (behavior.grind's target-first planner); None: by its worth
        and the danger there (the hunt loop's: `_allow`)."""
        return None

    def _share(self, p: Pile, worth: float) -> float:
        """The share of pile `p`'s worth (Pile.value) that she can take (`worth`: `_worth`'s, a full tab's items
        nothing): 1 with room for all of it (her inventory by tab, looting.Looter.can_take). Her allowance of a pile
        goes by it (`_spots`: what she'd go out of her way for is what she can take of it), from the review of
        2026-11-01: with her Etc tab full (her drops as live, Meat and her pets, 12 seeds of 5 minutes) the allowances
        alone had her hurt at piles 10.5 times a run against 4.8 with the exchange of mesos for EXP before, for no more
        loot."""
        looter = self.h.looter
        if not looter.inventory_full or p.value <= 0.0:
            return 1.0
        return min(1.0, max(0.0, worth / p.value))

    def _about(self, s: Scene, pid: int, xs: tuple[float, ...] = ()) -> tuple[list[Monster], list[Monster]]:
        """The monsters that count as a danger (attack.danger) on platform `pid` as of look `s`, and those on a platform
        just under or over it whose bodies meet hers standing at one of map xs `xs` there (Danger.beside: a pile's
        stretch; their walks their own platforms', `_reach_of`), as (those the state has on screen, those about that it
        didn't find: `_unseen`), none her kills dying there (ledger.dying: the pile's own monster drawn dying; until
        2026-10-30 12-28% of the piles' waits live were for it) nor of the feed's tracks taken for dead (Ledger.dead).
        [] both where nothing counts (the first map). Until 2026-10-31 her own platform's alone: in the simulator's
        grind on Tree Dungeon a pile on 9 was stood on with Horny Mushrooms walking 8 and 10 by it (D:/MapleHelper-
        scratch/replan/pos: seed 1 at 199.6-203.7 s, four touches at a stand on 9, two from 8 and two from 10)."""
        d = getattr(self.h, "danger", None)  # (tests make a hunter without one)
        if d is None or not d.any:
            return [], []
        ledger = getattr(self.h, "ledger", None)
        kills = ledger.kills if ledger is not None else []
        die, dead = (getattr(ledger, "die", {}), getattr(ledger, "dead", {})) if ledger is not None else ({}, {})
        seen = [*s.monsters, *[m for m in getattr(s, "swept", ()) if not any(m is o for o in s.monsters)]]
        p = self.model.platform(pid) if xs and BESIDE_GUARDS else None

        def counts(m) -> bool:
            if m.platform != pid and (p is None or not any(d.beside(m, p, x) for x in xs)):
                return False
            return d.share(m.kind) > 0 and not dying(kills, m, s.t, seen, die, dead)
        return [m for m in seen if counts(m)], [m for m in self._unseen(s) if counts(m)]

    def _reach_of(self, m: Monster) -> tuple:
        """Monster `m`'s walk for the sweep's checks (sweep.meets): its velocity the way perception says it heads at its
        kind's speed from the game data (monsters.walk_vx; None, its heading not known: either way), its speed, half its
        body's width (monsters.touch_box), where its middle turns on its platform (monsters.walk_edges: the game's walk
        edge, 20 from each end, or the ends themselves for a kind that hops, as the stack's danger has it since master
        072e1ec; until then the platform's ends, 20 units more than the game lets a walker go). One standing or
        flinching (walk_vx 0) may set off either way any time, as one whose heading isn't known (sweep.UNSURE_S of it at
        most): taken as staying put, a stand passed its check at 58 units from a standing one where PILE_CLEAR had kept
        her 100 off, and on the user's own play a standing monster set off at about 0.61 a second (the review of
        2026-10-30)."""
        if getattr(self, "_speeds", None) is None:
            self._speeds, self._bodies = speeds(self.model), bodies(self.model)
        speed = self._speeds.get(m.kind, DEFAULT_SPEED)
        vx = walk_vx(m.kind, getattr(m, "facing", 0), getattr(m, "pose", ""), self._speeds) or None
        q = self.model.platform(m.platform)
        d = getattr(self.h, "danger", None)
        t = None if d is None else d.threats.get(m.kind)
        return (vx, speed, touch_box(m.kind, self._bodies.get(m.kind))[0],
                *walk_edges(q, t is not None and t.hops))

    def _guards(self, s: Scene, p: Pile, at: float, stand: float) -> tuple[tuple, str | None]:
        """What would meet her standing on pile `p` from `at` s from now for `stand` s and sweep.STAND_PAST_S more (her
        body's box against each monster's that counts there, `_about`, walking as `_reach_of` has it: sweep.meets), as
        (the guards to shoot first, why the pile is left out or None: "a horny-mushroom [unseen ]at x 890 on 23 would meet
        her there"): every one that would, if a star from where she is hits each (`_hits`) and the state has it on screen
        (one about that it didn't find, "unseen", isn't thrown at), its kill time (Attacks.kill_s) and its EXP; else the
        pile is left out ("in general you should kill all the mobs that are in the way, try not to hit by mobs just for
        picking up loots": the user, 2026-10-30). Nothing where nothing counts (the first map)."""
        lo, hi = self._extent(p)
        seen, unseen_ = self._about(s, p.platform, (lo, hi))
        if not seen and not unseen_:
            return (), None
        t1 = at + stand + sweep.STAND_PAST_S
        met, rest = [], [*seen, *unseen_]
        while (m := sweep.meets(rest, lo, hi, at, t1, self._reach_of, HER_HALF)) is not None:
            met.append(m)
            rest = [o for o in rest if o is not m]
        if not met:
            return (), None
        met = [m for m in met if self._guards_it(s, p, m, any(m is o for o in seen))]
        if not met:
            return (), None
        for m in met:
            on_screen = any(m is o for o in seen)
            if not on_screen or not self._hits(s, m):
                return (), f"a {m.kind} {'' if on_screen else 'unseen '}at x {m.x:.0f} on {m.platform} would meet her there"
        attacks = getattr(self.h, "attacks", None)
        return tuple(sweep.Guard(m, attacks.kill_s(m) if attacks is not None else 1.0,
                                 self.tour.exp.get(m.kind, self._avg_exp())) for m in met), None

    def _guards_it(self, s: Scene, p: Pile, m: Monster, on_screen: bool) -> bool:
        """Whether monster `m`, that would meet her at pile `p`, counts as guarding it in look `s` (behavior.grind's
        target-first planner: not all do; the hunt loop's all)."""
        return True

    def _respawn_guard(self, s: Scene, pile: Pile, t0: float) -> str | None:
        """Why her trip to pile `pile` waits for the map's pass to be seen (RESPAWN_GUARD, see its comment), her stand
        there from `t0` s after look `s` (`_stand_s` and sweep.STAND_PAST_S long): where going would have her about the
        next pass (PASS_BEFORE before it to PASS_AFTER after: her stand, if it meets that, and her walk there along its
        platform from where she is, at WALK_SPEED) against where waiting would (where she is, or off a point by her:
        `_off_points`), the free spawn points of kinds that count within RESPAWN_NEAR of each (attack.danger's
        respawn_points, by every monster she knows of: `_around`), each one's newborn priced at the user's prices
        (`_fill`, NEWBORN_TOUCH less NEWBORN_SEEN, landing_prices' touch and the share that hurt): she waits where going
        costs more than waiting by more than the wait (to PASS_SEEN_S after the pass, at her EXP a second). None: she
        goes (or no clock sure yet, no danger, her HP not read, nothing to the pass)."""
        clock, d = getattr(self.h, "spawn_clock", None), getattr(self.h, "danger", None)
        if not RESPAWN_GUARD or clock is None or d is None or not d.any or not clock.sure:
            return None
        now = time.monotonic()  # (the looks' clock: Scene.t, the feed's events; a look's frame may be older than now)
        due = clock.next_pass(now - PASS_BEFORE)
        first, last = due - PASS_BEFORE, due + PASS_AFTER  # (the window about it she'd better be off a free point in)
        stand0 = now + t0
        if stand0 + self._stand_s(pile) + sweep.STAND_PAST_S < first:
            return None  # (done before the pass)
        pid, (lo, hi) = pile.platform, self._extent(pile)
        around = self._around(s)
        going = [(lo, hi)] if stand0 <= last else []  # (her stand meets the window)
        here = s.platform.id if s.platform is not None and s.where is not None else None
        if here == pid and stand0 > first:  # (her walk there along it while the window lasts)
            x = s.where.x
            goal = min(max(x, lo), hi)

            def walked(t: float) -> float:
                return x + (goal - x) * min(1.0, max(0.0, (t - now) * WALK_SPEED / max(abs(goal - x), 1e-9)))
            a, b = walked(first), walked(min(last, stand0))
            going.append((min(a, b), max(a, b)))
        go = {q for g0, g1 in going for q in d.respawn_points(pid, g0, g1, around)}
        if not go:
            return None
        stay = set() if here is None or self._off_points(s) is not None else \
            set(d.respawn_points(here, s.where.x, s.where.x, around))
        prices = fight.landing_prices(self.h, d)
        if prices is None:
            return None
        each = self._fill(clock.period) * (NEWBORN_TOUCH - NEWBORN_SEEN) * prices.hurt
        touch = sum(each * prices.touch(kind) for _, kind in go) - sum(each * prices.touch(kind) for _, kind in stay)
        wait = max(0.0, due + PASS_SEEN_S - now)
        if touch <= wait * prices.rate:
            return None
        x, kind = min(go - stay or go, key=lambda q: min(abs(q[0] - lo), abs(q[0] - hi)))
        key = (pid, round(x), round(due, 1))
        if key != getattr(self, "_respawn_said", None):  # (once a point a pass: she looks again every 0.1 s)
            self._respawn_said = key
            if telemetry.recording():
                telemetry.emit("event", kind="respawn_guard", what="trip", platform=pid, x=round(x), point_kind=kind,
                               pass_in=round(due - now, 2), touch=round(touch, 2), wait=round(wait * prices.rate, 2),
                               clock=clock.record())
        return (f"a {kind} spawn point at x {x:.0f} on {pid} may fill at the pass in {due - now:.1f} s (its newborn "
                f"{touch:.1f} EXP more than waiting, the wait {wait * prices.rate:.1f})")

    def _off_points(self, s: Scene) -> tuple[str, float, float] | None:
        """Where the spawn guard's wait for the pass is made when look `s` has her within RESPAWN_NEAR of a free spawn
        point of a kind that counts on her platform: a walk along it (the way, how far, to where) to the nearest place
        OFF_MARGIN past every such point's RESPAWN_NEAR, GROUND_MARGIN in from its ends, with no monster that counts in
        the way (`_in_the_way`) and MIN_MOVE or more off; None: none (she waits where she is, and the guard prices her
        staying there). The walk is blind (actions.walk), so only one that's safe all the way (the review of 2026-11-03:
        uncapped, it walked her 116-230 units across the free points it was keeping her from, at the pass): away from
        every such point by her (none of them on the side she walks to), passing within RESPAWN_NEAR of no other on the
        way, at most WAIT_STEP (rule 9's step) and done before the pass's window opens (WALK_SPEED, PASS_BEFORE before
        the clock's next pass); none from a look a touch's knock may still be moving her in (fight.knocked_move's), as
        rule 9's (`_wait_place`)."""
        d, clock = getattr(self.h, "danger", None), getattr(self.h, "spawn_clock", None)
        if d is None or s.platform is None or s.where is None or clock is None or not clock.sure:
            return None
        knocked = getattr(self.h, "knocked", None)
        if knocked is not None and knocked(s):
            return None
        here, x = s.platform, s.where.x
        points = d.respawn_points(here.id, here.left, here.right, self._around(s))
        near = [px for px, _ in points if abs(px - x) <= RESPAWN_NEAR]
        if not near:
            return None
        now = time.monotonic()
        due = clock.next_pass(now - PASS_BEFORE)
        most = min(WAIT_STEP, WALK_SPEED * (due - PASS_BEFORE - now))  # (there before the window opens)
        best = None
        for g in spots._stand_xs(here.left + GROUND_MARGIN, here.right - GROUND_MARGIN):
            lo, hi = min(x, g), max(x, g)
            if any(abs(px - g) <= RESPAWN_NEAR + OFF_MARGIN for px, _ in points) or abs(g - x) < MIN_MOVE \
                    or abs(g - x) > most or any((px - x) * (g - x) > 0 for px in near) \
                    or any(lo - RESPAWN_NEAR <= px <= hi + RESPAWN_NEAR for px, _ in points if px not in near) \
                    or self._in_the_way(s, x, g):
                continue
            if best is None or abs(g - x) < abs(best - x):
                best = g
        return None if best is None else ((RIGHT if best > x else LEFT), abs(best - x), best)

    def _around(self, s: Scene) -> list[Monster]:
        """Every monster she knows of as of look `s` that keeps a spawn point from filling (the server's SPAWN_CLEAR):
        the look's, those it followed off her band (Scene.swept) and those about that it missed (`_unseen`), none her
        kills dying (ledger.dying) nor the feed's tracks taken for dead (Ledger.dead): the server counts the living."""
        ledger = getattr(self.h, "ledger", None)
        kills = ledger.kills if ledger is not None else []
        die, dead = (getattr(ledger, "die", {}), getattr(ledger, "dead", {})) if ledger is not None else ({}, {})
        seen = [*s.monsters, *[m for m in getattr(s, "swept", ()) if not any(m is o for o in s.monsters)]]
        return [m for m in [*seen, *self._unseen(s)] if not dying(kills, m, s.t, seen, die, dead)]

    def _fill(self, period: float) -> float:
        """A free spawn point's chance of a newborn at the map's next pass (FILL_PRIOR: the hunt loop learns none)."""
        return FILL_PRIOR

    def _respawn_wait(self, s: Scene, pid: int, xs: list[float]) -> list[float]:
        """The share of her HP the newborns of the free spawn points by each place of map xs `xs` on platform `pid` are
        expected to take her waiting there over the next WAIT_AHEAD_S (attack.danger.respawn_risk, the clock's chance
        about its pass; 0 each where RESPAWN_GUARD is off or the clock isn't sure): what rule 9's places weigh too."""
        clock, d = getattr(self.h, "spawn_clock", None), self.danger
        if not RESPAWN_GUARD or clock is None or d is None or not clock.sure:
            return [0.0] * len(xs)
        now = time.monotonic()
        chance = clock.chance(now, now + WAIT_AHEAD_S, -PASS_BEFORE, PASS_AFTER)
        if chance <= 0.0:
            return [0.0] * len(xs)
        fill = self._fill(clock.period)
        points = d.respawn_points(pid, min(xs), max(xs), self._around(s))
        return [d.respawn_risk([q for q in points if abs(q[0] - x) <= RESPAWN_NEAR], fill, chance) for x in xs]

    def _unsafe(self, s: Scene, p: Pile) -> str | None:
        """Why standing on pile `p` once she's there from look `s` would have a monster meet her that she can't shoot
        first from here (`_guards`), or None (rule 6a's pile: it waits)."""
        return self._guards(s, p, self._travel_to_pile(s, p), self._stand_s(p))[1]

    def _at_pile(self, s: Scene, p: Pile) -> bool:
        """Whether look `s` has her at pile `p`: on its platform, by its stretch (a trip's arrival, ARRIVE_TOLERANCE)."""
        if s.platform is None or s.where is None or s.platform.id != p.platform:
            return False
        lo, hi = self._extent(p)
        return lo - ARRIVE_TOLERANCE - STOOD_ON <= s.where.x <= hi + ARRIVE_TOLERANCE + STOOD_ON

    def _sweep_act(self, s: Scene, stop: "sweep.Stop", visible, hidden, now: float) -> None:
        """The sweep's next stop carried out: its guards shot first ('sweep: clear first'), else at the pile its stand
        (`_stand`; its drops' landing waited for, 'sweep: landing'), else a trip there ('sweep': the navigator's, planned
        and replanned at every state as any)."""
        sp = stop.spot
        pile = self._pile_of.get(sp.key)
        if pile is None:
            return None
        if sp.guards:
            guards = [g.monster for g in sp.guards]
            return self._shoot(s, guards, list(visible), list(hidden), now, "sweep: clear first", approach=False)
        if self._at_pile(s, pile):
            if sp.lands > 0.0:
                self._note("sweep: landing")
                self._landing_wait(sp.lands)
                return None
            self._note("sweep")
            return self._stand(s, pile)
        return self._go_for(pile, "sweep", s, after=self._sweeper.dest)

    def _landing_wait(self, lands: float) -> None:
        """By a pile whose drops land in `lands` s: a moment's wait (behavior.grind's whole screen: her target meanwhile)."""
        time.sleep(min(lands, 0.1))

    def _stand(self, s: Scene, pile: Pile) -> None:
        """Pick up pile `p` from where look `s` has her by it: only if nothing would meet her there now (`_guards`, from
        now: else the next pass's sweep decides again: the guard shot first, or the pile left), each bunch of its drops
        seen (PILE_GAP) walked to from the nearer end and stood on, a press of pick-up for each drop and one more (its
        drops not seen yet, the walk over SPREAD either side of where her kill died, pressing all the way); a drop still
        where she stood twice can't be picked up (looting.cant_pick: someone else's, or her inventory full); drops seen
        by it that are all items of a full tab (looting.Looter.can_take) none to take, its try spent and nothing walked
        (from the review of 2026-10-31: they were taken for none seen yet, and its spread walked). One try of
        the pile's (PILE_TRIES); a telemetry event 'pick'. What's left the next look's drops show (the next pass's sweep:
        until 2026-10-30 a mop-up of its own, 0.3 s and a look between its rounds)."""
        here, x = s.platform.id, s.where.x
        guards, why = self._guards(s, pile, 0.0, self._stand_s(pile))
        if why is not None or guards:
            return None
        looter = self.h.looter
        near = [d for d in s.loot if d.platform == here and abs(d.x - pile.x) <= LOOT_REACH]
        seen = looter.wanted(near)
        can = getattr(looter, "can_take", None)
        if not seen and can is not None and any(not d.mesos and not can(d.item.kind) for d in near):
            self._tried(pile)  # (drops there, items of a full tab alone: nothing to take, not its spread walked)
            return None
        kept = self._stood.get(id(pile))
        if kept is None or kept[0] is not pile:
            kept = self._stood[id(pile)] = (pile, [])
        stood = kept[1]
        stuck = [d for d in seen if sum(abs(sx - d.x) <= STOOD_ON for sx in stood) >= 2]
        if stuck:
            looter.cant_pick(stuck)
            seen = [d for d in seen if not any(d is k for k in stuck)]
        if seen:
            xs = sorted(d.x for d in seen)
            bunches = []
            for d in xs:  # (as looting._piles bunches them, with how many in each)
                if bunches and d - bunches[-1][-1] < PILE_GAP:
                    bunches[-1].append(d)
                else:
                    bunches.append([d])
            goals = [(sum(b) / len(b), len(b) + 1) for b in bunches]
        elif pile.status == "expected" and not stood:
            if self._none_landed(s, pile):
                return None
            lo, hi = self._extent(pile)
            goals = [(lo, 1), (hi, 2)]
        else:  # (seen there before, none seen now from by it: gone, as the store will have it)
            self._tried(pile)
            return None
        if abs(x - goals[-1][0]) < abs(x - goals[0][0]):
            goals.reverse()
        actions, x0, first = self.h.actions, x, True
        for goal, presses in goals:
            if abs(goal - x) >= MIN_WALK:
                actions.walk(RIGHT if goal > x else LEFT, abs(goal - x))
                if first and abs(goal - x) >= RUN_UNITS:
                    self._ran(x0, goal)
                x, first = goal, False
            actions.pick_here(presses)
            stood.append(goal)
        if (on := getattr(looter, "stood_on", None)) is not None:  # (the kinds she holds: looting.ITEM_TABS)
            on([d.item.kind for d in seen if not d.mesos], pile)
        self._tried(pile)
        self.h.report.loot_runs += 1
        self._idle_since = None
        if telemetry.recording():
            telemetry.emit("event", kind="pick", platform=here, x=round(pile.x), drops=len(seen),
                           stood=[round(g) for g, _ in goals], tries=pile.tries)
        return None

    def _none_landed(self, s: Scene, pile: Pile) -> bool:
        """Whether look `s`, by a kill's pile no look has seen yet, says its kill dropped nothing and so no stand is made there
        (behavior.grind's target-first planner; the hunt loop's stands walk its spread all the same)."""
        return False

    def _ran(self, x0: float, goal: float) -> None:
        """A stand's walk of RUN_UNITS or more from map x0 to `goal`, told to her moves' watch (keycheck.MoveWatch, as the
        navigator's chains tell it of theirs): whether a glance after it has her dot RUN_SEEN units on. Her arrow keys on
        other keys in the game, her stands' walks never moved her and, standing by her drops, no chain of hers ran to
        doubt them (loop_tree_grind_arrow_key: no key check in its 4.5 minutes, where with the old rules one at 61 s)."""
        moves, senses = getattr(self.h, "moves", None), getattr(self.h, "senses", None)
        if moves is None or senses is None or not hasattr(senses, "glance"):
            return
        dot, _ = senses.glance()
        if dot is not None:
            moves.ran(time.monotonic(), abs(senses.localizer.to_map(dot)[0] - x0) >= RUN_SEEN)

    def _valuable_pile(self, s: Scene) -> Pile | None:
        """Rule 6a: of the piles seen to hold a drop she can take worth `valuable` mesos or more (looting.Worth:
        `_dearest`), VALUABLE_TRAVEL s away at most and gone for at once fewer than VALUABLE_TRIES times, the one worth
        the most to her (`_worth`) per second of travel; None while she's going for one already. Only one where nothing
        would meet her standing on it that she can't shoot first (`_unsafe`). (Until the independent check of
        2026-11-01 by the dearest drop and the worth of all it held, a full tab's items too: with Etc full she rushed a
        pile of a coin and an ore for the coin.)"""
        if self.obj is not None and self.obj.why == "valuable pile":
            return None
        best = None
        for p in self.h.drops.known:
            dear = self._dearest(p)
            if dear is None or dear < self.valuable or p.rushed >= VALUABLE_TRIES or not self._candidate(p, s):
                continue
            t = self._travel_to_pile(s, p) + PICK_S
            worth = self._worth(p)
            if t <= VALUABLE_TRAVEL + PICK_S and (best is None or worth / t > best[0]) and not self._unsafe(s, p):
                best = (worth / t, p)
        return None if best is None else best[1]

    def _dearest(self, p: Pile) -> float | None:
        """The worth of the dearest drop seen in pile `p` that she can take (Pile.dearest but for the items of a full
        tab: looting.Looter.can_take), None with none (or none seen)."""
        looter = self.h.looter
        can = getattr(looter, "can_take", None)
        if can is None or not looter.inventory_full or p.dearest is None:
            return p.dearest
        each = p.each if len(p.each) == len(p.items) else tuple(self.h.drops.value(k) for k in p.items)
        return max((w for k, w in zip(p.items, each) if k == MESOS or can(k)), default=None)

    def _loot_first(self, s: Scene, p: Pile, monsters, say: bool = True) -> bool:
        """EXP before loot: whether pile `p` goes ahead of shooting what's in reach of `monsters` from where she stands
        in look `s` (_shoot_rate): only with nothing in reach. Said in her log when it waits (`say`). Until the review
        of 2026-11-01 one at risk by then (its survival over the kills, another player by it once it was no longer hers
        alone: RIVAL_TAKES, 0.5) went first when what waiting would likely lose of it, at MESO_EXP an meso, was more
        than the EXP she'd make from what's in reach over the trip there and back: an exchange the user wasn't sure of
        ("My idea is just picking up mesos and items as we kill mobs"), and none went first so in the simulator's runs
        (the build's strict EXP first, her drops as live, the same to the digit on 12 seeds of 15 minutes)."""
        if not EXP_FIRST:
            return True
        rate, busy = self._shoot_rate(s, monsters)
        if rate <= 0.0:
            return True
        if say and self._loot_said.get(id(p)) is not False:
            self._loot_said[id(p)] = False
            worth = self._worth(p)  # (but for a full tab's items: the independent check of 2026-11-01)
            self.log(f"  {_contents(p)} on platform {p.platform} at x {p.x:.0f} (worth {worth:.0f}) against {rate:.1f} "
                     f"EXP a second in reach for {busy:.0f} s: shooting first")
            if telemetry.recording():
                telemetry.emit("event", kind="loot_first", first=False, worth=round(worth, 1), rate=round(rate, 2),
                               busy=round(busy, 1), platform=p.platform, x=round(p.x))
        return False

    def _shoot_rate(self, s: Scene, monsters) -> tuple[float, float]:
        """EXP a second she'd make shooting the wanted ones of `monsters` in reach from where she stands in look `s` (as
        rule 3 has them: a star from there hits it, or it's on her platform), and s it takes to kill them all
        (Attacks.kill_s: what each has left, the attack she'd throw); (0, 0) with none."""
        if s.where is None or s.platform is None:
            return 0.0, 0.0
        here = s.platform.id
        near = [m for m in monsters if (self.kinds is None or m.kind in self.kinds)
                and (m.platform == here or self._hits(s, m))]
        if not near:
            return 0.0, 0.0
        busy = sum(self.h.attacks.kill_s(m) for m in near)
        exp = sum(self.tour.exp.get(m.kind, self._avg_exp()) for m in near)
        return (exp / busy if busy > 0 else 0.0), busy

    def _call_off(self, o: Objective, why: str | None = None) -> None:
        """Her trip for a pile ends where she is (something in reach goes first now, or `why`: a monster would meet her
        there, `_unsafe`): the pile stays known, not counted as tried nor rushed, and the objective under it, if any, is
        hers again."""
        p = o.pile
        if o.why == "valuable pile":
            p.rushed = max(0, p.rushed - 1)
        self.log(f"  {_contents(p)} on platform {p.platform} at x {p.x:.0f}: later, "
                 + ("shooting what's in reach first" if why is None else why))
        self.obj = self.stack.pop() if self.stack else None
        return None

    def _yields(self, s: Scene) -> bool:
        """Looter.yield_to while she hunts: whether picking up stops for what's in reach in look `s` (EXP before loot:
        any wanted monster in reach, the pile under her waiting)."""
        return self._shoot_rate(s, s.monsters)[0] > 0.0

    def _go_for(self, pile: Pile, rule: str, s: Scene, after: tuple | None = None, chain=None) -> None:
        """A trip to pile `pile` for rule `rule` (the sweep's, a valuable pile's): its objective, the one she was on
        suspended under it; or, where her stand there would meet the map's next pass by a free spawn point (the spawn
        guard: `_respawn_guard`), nothing yet ('<rule>: the pass'), the next look deciding again."""
        now = time.monotonic()
        at = max(self._travel_to_pile(s, pile),
                 max(0.0, pile.since + DROP_LANDS - now) if pile.status == "expected" else 0.0)
        if at <= TRIP_AHEAD_S and (why := self._respawn_guard(s, pile, at)) is not None:
            if why != getattr(self, "_pass_said", None):
                self._pass_said = why
                self.log(f"  {rule}: {_contents(pile)} on {pile.platform} at x {pile.x:.0f} after the pass: {why}")
            if (step := self._off_points(s)) is not None:  # (by a free point herself: she waits off it)
                self.log(f"  waiting for the pass at x {step[2]:.0f}, off the spawn point")
                self.h.actions.walk(step[0], step[1])
            else:
                time.sleep(0.1)
            self._note(f"{rule}: the pass")  # (she goes once the pass is seen, deciding again with what it brought)
            return None
        parent = self.obj
        if parent is not None:
            self.stack.append(parent)  # suspended
        if rule == "valuable pile":
            pile.rushed += 1
        self.obj = Objective("pile", pile.platform, pile.x, pile=pile, why=rule, after=after, chain=chain)
        if rule == "sweep" and parent is not None:
            # A stop on the way shoots in passing out of the trip's own budget, and leaves finishing to it (once
            # it's done she owes them still): with budgets of their own, its stops shot on and on, and in the
            # simulator's hunt loop the objectives ran out their OBJECTIVE_S 99 times in 60 runs, 87 so (61 without
            # the rule), for 128.3 EXP a minute against 128.7 and 90.1% of her drops against 90.0% (rule 5b's piles on
            # the way, as the sweep's now).
            self.obj.passing, self.obj.finishing = parent.passing, FINISH_S
        return self._travel(rule)

    # --- spots and grounds --------------------------------------------------------------------------

    def _to_spot(self, spot: Spot, rule: str) -> None:
        self._to_spot_obj(spot)
        return self._travel(rule)

    def _to_spot_obj(self, spot: Spot) -> None:
        if self.obj is not None:
            self.stack.append(self.obj)
        self.obj = Objective("spot", spot.platform, spot.x, ground=self.visit.ground, spot=spot)

    def _better_spot(self, s: Scene) -> Spot | None:
        """Rule 7: wanted monsters on the ground out of reach (as her memory has them), and a spot reaching clearly more
        of them."""
        v = self.visit
        if v.changes >= SPOT_CHANGES or time.monotonic() - v.spot_since < SPOT_REST:
            return None
        left = self._remembered(s, v.ground)
        if not left:
            return None
        now_n = sum(self._hits(s, m) for m in left)
        nxt = spots.spot_for(self.model, v.ground, left, v.spot, s.where, self._travel_off(v.ground), self.kinds,
                             danger=self.danger)
        if nxt is None:
            return None
        p = self.model.platform(nxt.platform)
        n = sum(reach.can_hit(p, nxt.x, p.y_at(nxt.x), m, others=s.monsters) for m in left)
        travel = (abs(nxt.x - s.where.x) / WALK_SPEED if nxt.platform == s.platform.id
                  else self._travel_off(v.ground)(s.platform.id, s.where.x, nxt.platform))
        if travel is None or travel > SPOT_TRAVEL or not (n >= now_n + SPOT_MORE or n >= SPOT_TIMES * max(now_n, 1)):
            return None
        return nxt

    def _travel_off(self, ground: int):
        """The planner's route times keeping off `ground` (route.KEEP_OFF_S): to the spots off it it's shot from,
        which a trip to one keeps off it for (_keep_off)."""
        return lambda a, x, b: self.nav.planner.travel(a, x, b, keep_off=(ground,))

    def _nothing_here(self, s: Scene) -> bool:
        """Whether her memory has no wanted monster on the ground as of look `s` (a look missing one hidden under its
        damage numbers doesn't empty it)."""
        return not self._remembered(s, self.visit.ground)

    def _due_back(self) -> float | None:
        """When the next of her kills on this ground is due back (s from now), if one is."""
        now = time.monotonic()
        due = [t + RESPAWN - now for t in self.visit.kills if t + RESPAWN + 2 > now]
        return min(due) if due else None

    def _leaving(self, s: Scene) -> str | None:
        """The rate rule: why she leaves this ground now, or None to stay."""
        v, now = self.visit, time.monotonic()
        if v.arrived is None:
            return None
        here_s = now - v.arrived
        if here_s >= GROUND_S:
            telemetry.emit("event", kind="backstop", text=f"GROUND_S: {GROUND_S:.0f} s at platform {v.ground}")
            return f"{GROUND_S:.0f} s here"
        due = self._due_back()
        covered = self._covers_ground(s)
        if covered and self._nothing_here(s) and due is None:
            if v.empty_since is None:
                v.empty_since = now
            if now - v.empty_since < EMPTY_CONFIRM_S:
                return None  # to be confirmed (not left on patience meanwhile: then it wouldn't be skipped)
            if not self._elsewhere():
                return self._stay("nothing left here")
            self._skip[v.ground] = now + SKIP_S
            return "nothing left here, and nothing due back"
        if covered or due is not None:
            v.empty_since = None
        judged = covered or s.platform.id in (v.ground, v.spot.platform)
        if judged and self._idle_since is not None and now - self._idle_since >= self._patience(s) and due is None:
            if not self._elsewhere():
                return self._stay("nothing came into reach")
            return "nothing came into reach"
        if here_s < MIN_STAY and v.throws < MIN_THROWS:
            return None
        rate, best = self._rate_here(), self._best_rate(s)
        if best is not None and rate < LEAVE_MARGIN * best[0]:
            return f"{rate:.2f} EXP/s here, {best[0]:.2f} to be had at platform {best[1]}"
        return None

    def _elsewhere(self) -> bool:
        """Whether leaving now would take her to another ground, as _next_ground picks: one she isn't skipping, or with
        every one skipped, the soonest back of the others. Not when this one is all there is: live on 2026-10-24 with
        23 the only ground (20261024-165414-tour.jsonl), "nothing came into reach" fired about once a second for 11
        minutes and 23 was picked again at once, each time a visit of about a second and no EXP in the value map (113
        of them: record_visit)."""
        now, here = time.monotonic(), self.visit.ground
        others = [g for g in self.tour.hunting_grounds(self._visit_s) if g != here]
        return any(self._skip.get(g, 0) <= now for g in others) or any(k != here for k in self._skip)

    def _stay(self, why: str) -> None:
        """A reason to leave for the ground she'd pick again (not _elsewhere): she stays, waiting (rule 9), and it's
        progress as picking it again was (STUCK_S: so the stuck guard fires as it did); said once until she throws or
        kills there again."""
        self._progress = time.monotonic()
        v = self.visit
        if self._stay_said != (said := (id(v), v.throws, len(v.kills))):
            self._stay_said = said
            self.log(f"  staying at platform {self.visit.ground}: {why}, and no other ground to go to")
        return None

    def _covers_ground(self, s: Scene) -> bool:
        """Whether look `s` searched GROUND_SEEN of the ground for monsters at least (a sliver at the screen's edge says
        nothing about the monsters on the rest of it): the screen's stretch of it, its level within the rows searched
        (Scene.rows: the band, the screen). Until 2026-10-31 by where drops were looked for (Scene.looked), the screen's
        breadth while every look searched it whole for drops; with the drops' search a strip at a time
        (worldstate.DROPS_EVERY_S, 2026-10-30) a ground was never covered, so "nothing left here" never came and she
        left an empty ground on her patience alone, there and back between two every 4-5 s (loop_tree_run's seeds 3, 6
        and 9 between 19 and 23)."""
        p = self.model.platform(self.visit.ground)
        span = s.searched_x(p.y_at((p.left + p.right) / 2))
        if span is None:
            return False
        return max(0.0, min(span[1], p.right) - max(span[0], p.left)) >= GROUND_SEEN * (p.right - p.left)

    def _patience(self, s: Scene) -> float:
        if not s.monsters:
            return PATIENCE["none"]
        return PATIENCE["ground"] if s.platform.id == self.visit.ground else PATIENCE["spot"]

    def _rate_here(self) -> float:
        v = self.visit
        prior = self.tour.ground_values().get(v.ground, 0.0) / self._visit_s(v.ground)
        return (v.exp + prior * PRIOR_S) / (time.monotonic() - v.arrived + PRIOR_S) * _safe_share(v.spot)

    def _shared(self, g: int, s: Scene) -> float:
        """What of ground `g`'s EXP a second is hers to have as look `s` shows the other players: PLAYER_DISCOUNT with
        a rival of hers on it, on its spot's platform, or by it (PLAYER_NEAR, PLAYER_DY), else all of it."""
        rivals = [pl for pl in s.players if pl.rival]
        if not rivals or PLAYER_DISCOUNT == 1.0:
            return 1.0
        p, spot = self.model.platform(g), spots.spot_at(self.model, g, self.kinds, self.danger)[0]
        for pl in rivals:
            x = min(max(pl.x, p.left), p.right)
            if pl.platform in (g, spot) or (abs(pl.x - x) <= PLAYER_NEAR and abs(pl.y - p.y_at(x)) <= PLAYER_DY):
                return PLAYER_DISCOUNT
        return 1.0

    def _visit_s(self, pid: int) -> float:
        """A visit's expected length: VISIT_S counted as PRIOR visits (as the value map's EXP per visit is),
        blended with the timed visits. Unblended, one empty ground left in 0.6 s looked like the best one:
        its EXP per second jumped several-fold."""
        st = self.tour.stats.get(str(pid), {})
        return (PRIOR * VISIT_S + st.get("seconds", 0.0)) / (PRIOR + st.get("timed", 0))

    def _best_rate(self, s: Scene) -> tuple[float, int] | None:
        """The best other ground's EXP per second, the trip there included."""
        best, now = None, time.monotonic()
        values = self.tour.ground_values()
        for g in self.tour.hunting_grounds(self._visit_s):
            if g == self.visit.ground or self._skip.get(g, 0) > now:
                continue
            sp = spots.best_spot(self.model, g, self.kinds, self.danger)
            t = self.nav.planner.travel(s.platform.id, s.where.x, sp.platform) if s.platform.id != sp.platform else 0.0
            if t is None:
                continue
            rate = values.get(g, 0.0) / (t + self._visit_s(g)) * self._shared(g, s) * _safe_share(sp)
            if best is None or rate > best[0]:
                best = (rate, g)
        return best

    def _next_ground(self, s: Scene) -> None:
        """The ground to hunt next: the best EXP per second, travel included (varied a little; of those about as good,
        GROUND_TIE, the one she left longest ago), and its spot."""
        self._leave_ground()
        self._exit_reason = self._depart = None
        now, values, best, scored = time.monotonic(), self.tour.ground_values(), None, []
        here = s.where.platform if s.where is not None else None
        for g in self.tour.hunting_grounds(self._visit_s):
            if self._skip.get(g, 0) > now or (self.visit is not None and g == self.visit.ground):
                continue
            sp = spots.best_spot(self.model, g, self.kinds, self.danger)
            if here is None:  # on a rope or in the air: no route times, so by value alone
                t = 0.0
            else:
                t = 0.0 if here.id == sp.platform else self.nav.planner.travel(here.id, s.where.x, sp.platform)
            if t is None:
                continue
            score = (values.get(g, 0.0) / (t + self._visit_s(g)) * self.rng.uniform(0.75, 1.25) * self._shared(g, s)
                     * _safe_share(sp))
            scored.append((score, g))
            if best is None or score > best[0]:
                best = (score, g)
        if best is not None and GROUND_TIE > 0:  # about as good: the one she was at longest ago (never: first)
            tied = [sg for sg in scored if sg[0] >= (1 - GROUND_TIE) * best[0]]
            best = min(tied, key=lambda sg: (self._left.get(sg[1], float("-inf")), -sg[0]))
        if best is None:  # every ground skipped: the soonest back
            grounds = self.tour.hunting_grounds(self._visit_s)
            if not self._skip and not grounds:
                raise RuntimeError("no hunting ground on this map")
            failed = self.visit.ground if self.visit is not None else None
            back = [k for k in self._skip if k != failed] or list(self._skip)
            g = min(back, key=self._skip.get) if back else grounds[0]
            self._skip.pop(g, None)
        else:
            g = best[1]
        spot = spots.spot(self.model, g, s.where, self._travel_off(g), self.kinds, danger=self.danger)
        self.visit = Visit(g, spot)
        self.report.grounds.append(g)
        self.stack.clear()
        self.obj = Objective("spot", spot.platform, spot.x, ground=g, spot=spot)
        telemetry.emit("tour", order=list(self.report.grounds), stop=g,
                       spot={"platform": spot.platform, "x": round(spot.x, 1), "coverage": spot.coverage})
        self.log(f"  next: platform {g}, from platform {spot.platform} at x {spot.x:.0f} ({spot.coverage:.0%} in reach)")

    def _leave_ground(self) -> None:
        """What the visit gave goes into the value map (with how long it lasted)."""
        v = self.visit
        if v is None or v.arrived is None:
            return
        self._left[v.ground] = time.monotonic()
        self.tour.record_visit(v.ground, v.exp, time.monotonic() - v.arrived)

    # --- travelling ---------------------------------------------------------------------------------

    def _travel(self, rule: str) -> None:
        """Carry out the objective: one trip, until she gets there, a rule interrupts it (the navigator
        asks `_interrupt` before every chain), or it fails."""
        self._note(rule)
        o = self.obj
        o.trip = self.nav.begin(o.platform, o.x, seconds=max(5.0, OBJECTIVE_S - o.age()),
                                until=self._interrupt, mode=TRANSIT, scene=self._seen, keep_off=_keep_off(o),
                                vary=self._varies(o), on_arrival=self._ends_on_arrival(o),
                                instead_of_waiting=self._instead_of_waiting(o), through=self._through(o),
                                land_near=self._land_near(o))
        o.trip.weights, o.trip.choice = o.weights, o.choice
        chains = o.trip.trip.chains
        while not o.trip.over:
            self.nav.step(o.trip)
            self._owe_move = time.monotonic() < o.trip.post_hit_until  # (just touched: on first, navigator.POST_HIT)
            if o.trip.trip.chains > chains:
                chains = o.trip.trip.chains
        result = o.trip.trip.result
        if o.chain is not None and self._chain_over(o, result):
            return None
        if result == "arrived" or (result == "in position" and o.kind == "spot" and self._on_spot(o)):
            return self._arrived(o)
        if result == "in position":
            return None  # a rule fired: the next pass decides (OBJECTIVE_S is checked there)
        self.log(f"  {o.text()}: {result}")
        return self._drop_objective(o, result)

    def _through(self, o: Objective):
        """What objective o's trip asks for the way on through her drops (TripState.through; behavior.grind's loot chain; the hunt loop's: nothing)."""
        return None

    def _chain_over(self, o: Objective, result: str) -> bool:
        """A trip through her drops over (behavior.grind's loot chain): whether that's dealt with."""
        return False

    def _land_near(self, o: Objective) -> bool:
        """Whether objective o's trip lands for its x (TripState.land_near; behavior.grind's trip to her target's place)."""
        return False

    def _instead_of_waiting(self, o: Objective):
        """What's asked before objective o's trip stands and waits for a monster (TripState.instead_of_waiting; behavior.grind's
        whole screen; the hunt loop's: nothing, she waits)."""
        return None

    def _varies(self, o: Objective) -> bool:
        """Whether objective o's trips may vary their way (route.ROUTES; behavior.grind's trips to a target's place: never)."""
        return o.kind != "pile"

    def _ends_on_arrival(self, o: Objective) -> bool:
        """Whether objective o's trip ends as she reaches its platform, with no walk on to its x (TripState.on_arrival:
        behavior.grind's target-first planner's spot trips; the hunt loop's none)."""
        return False

    def _on_spot(self, o: Objective) -> bool:
        """Stopped 'in position' where she can stand with targets of the ground in reach: as good as there.
        Judged from the look the trip stopped on (`_interrupt` said so to it a moment ago)."""
        s = o.trip.last if o.trip is not None and o.trip.last is not None else self.h.look()
        return (s.platform is not None and self._standable(s.platform.id)
                and any(m.platform == o.ground and self._hits(s, m) for m in s.monsters))

    def _interrupt(self, s: Scene) -> bool:
        """Asked before every chain of a trip: whether a rule calls for her here (close quarters, a valuable pile,
        one she has hurt to finish, a target in reach she may shoot, a pile the sweep has on her way or another than the
        one she's going for, `_sweep_calls`). A valuable pile calls for her only where it goes ahead of what's in reach
        (`_loot_first`, as `_pass` asks it): one that waits stopped the trip "in position" on its first look, no move
        made, and the next pass
        sent her off on it again, again and again, neither shooting nor moving once her time in passing there was spent
        (review, 2026-10-25: 1-4 such trips a run in loop_tree_ore_waits, one followed by 2.9 s without a throw in the
        harness's crowd). In reach, so too: only what the pass would shoot (fight._wanted, as its targets are: on the
        ground, on her platform, or in a star's reach from where she stands). A monster off both that only a jump
        reaches isn't one of its targets, and stopped every trip on its first look: in loop_tree_walls a green mushroom
        on the spot's platform, above her, held her 107 s, the pass deciding 2,373 times and never moving (found
        offline, 2026-10-25)."""
        here = s.platform.id
        if self._close_quarters(s, self._may_stop_her(s)):
            return True
        o = self.obj
        if o is not None and o.kind == "pile" and o.pile is not None and self._worthless(o.pile):
            return True  # not worth it any more (`_pass` drops it)
        if o is not None and o.why == "valuable pile" and o.pile is not None and self._unsafe(s, o.pile):
            return True  # a monster would meet her there now (`_pass` calls it off)
        wanted = [m for m in self._may_stop_her(s) if self.kinds is None or m.kind in self.kinds]
        if o is not None and o.why == "valuable pile":
            # At once: nothing else stops her, but something in reach that goes first now (`_loot_first`)
            return not self._loot_first(s, o.pile, wanted, say=False)
        if (pile := self._valuable_pile(s)) is not None and self._loot_first(s, pile, wanted, say=False):
            return True
        if self._owed(s, s.monsters) is not None:
            return True
        # What's in reach, asked only where it decides (the latency analysis of 2026-11-01, its option d: the jump-shot
        # search for each monster no star from the ground hits was most of this question's time): in passing one a star
        # hits decides it, the rest not searched; with neither her time in passing here left nor a spot's ground to
        # shoot at from here, nothing in reach does.
        passing = o is not None and o.passing.get(here, 0.0) < PASSING_S
        if passing and any(self._star_hits(s, m) for m in wanted):
            return True
        spot = o is not None and o.kind == "spot" and self._standable(here)
        if passing or spot:
            ground = None if self.visit is None else self.visit.ground
            in_reach = [m for m in wanted if self._hits(s, m)
                        and (ground is None or m.platform in (ground, here) or self._star_hits(s, m))]
            if passing and (in_reach or self._pass_step(s, wanted) is not None):
                return True
            if spot and any(m.platform == o.ground for m in in_reach):
                return True
        return self._sweep_calls(s)

    def _may_stop_her(self, s: Scene) -> list[Monster]:
        """The monsters that can stop a trip (`_interrupt`; behavior.grind's whole screen: those it can have as a target)."""
        return s.monsters

    def _arrived(self, o: Objective) -> None:
        if o.kind == "spot":
            v = self.visit
            if v is not None and o.ground == v.ground:
                v.spot, v.spot_since = o.spot or v.spot, time.monotonic()
                if v.arrived is None:
                    v.arrived = time.monotonic()
                    self._progress = time.monotonic()
                    self.log(f"  at platform {v.ground}'s spot (platform {o.platform})")
        elif o.kind == "pile" and o.pile is not None:
            # There: its stand at once, from a look where she got to (`_stand`: only where nothing would meet her
            # there; not landed yet, or a monster by it, the next pass's sweep decides again).
            s = self.h.look(mobs=True, loot=True)
            if s.where is not None and s.platform is not None:
                self._stand(s, o.pile)
            self.h.looter.pending.clear()
        self.obj = self.stack.pop() if self.stack else None
        self._idle_since = None
        return None

    def _drop_objective(self, o: Objective, why: str) -> None:
        if o.kind == "pile" and o.pile is not None:
            self._tried(o.pile)
        if o.kind == "spot" and self.visit is not None and o.ground == self.visit.ground and self.visit.arrived is None:
            self._skip[o.ground] = time.monotonic() + SKIP_S  # couldn't get there: another ground
            self._next_ground(self.nav.mover.settle())
            return None
        if o.kind == "spot":  # the trip back up failed or ran out its clock: not again for a while, so the
            # rate rule gets a look in (without it, high ground was re-created at once, again and again)
            self._no_high_until = time.monotonic() + SPOT_REST
        self.obj = self.stack.pop() if self.stack else None
        return None

    # --- waiting ------------------------------------------------------------------------------------

    def _wait(self, s: Scene, targets) -> None:
        """Rule 9: wait a moment, where she is, on a rope (_refuge), or (a walk along her platform) at a safer place
        (_wait_place)."""
        self._note("wait")
        if self._idle_since is None:
            self._idle_since = time.monotonic()
        if (r := self._refuge(s)) is not None:
            return self._hang(s, r)
        if (step := self._wait_place(s)) is not None:
            side, units, goal, there = step
            self.log(f"  waiting at x {goal:.0f} instead (danger {there:.2f} there)")
            self.h.actions.walk(side, min(units, WAIT_STEP))
            return None
        time.sleep(0.1)
        return None

    def _wait_place(self, s: Scene) -> tuple[str, float, float, float] | None:
        """Rule 9: where along her platform to wait, as a walk (the way, how far, to where, the danger there), or None
        to wait where she is: of the places GROUND_MARGIN from its ends (spots.X_STEP apart) from which a star reaches
        as many of the ground's spawn points as from where she is (_spawns_in_reach), the safest (Danger.at with this
        look's monsters), if it's WAIT_GAIN safer than here and stance.MIN_MOVE away, and the way there doesn't pass
        a monster that counts (_in_the_way). Never on a map where no kind counts as a danger (the first map: she waits
        where she is, as she did), nor from a look a touch's knock may still be moving her in (fight.knocked_move's)."""
        d, x = self.danger, s.where.x
        if d is None or not d.any or self.h.knocked(s):
            return None
        now, best = self._safest_along(s)
        if best is None or now - best[0] < WAIT_GAIN or best[1] < MIN_MOVE:
            return None
        there, units, goal = best
        return (RIGHT if goal > x else LEFT), units, goal, there

    def _safest_along(self, s: Scene) -> tuple[float, tuple[float, float, float] | None]:
        """The danger where she is (Danger.at with this look's monsters), and the safest of _wait_place's places along
        her platform as (its danger, how far, its x), None if there's none."""
        d, here, x = self.danger, s.platform, s.where.x
        xs = [g for g in spots._stand_xs(here.left + GROUND_MARGIN, here.right - GROUND_MARGIN)]
        due = self._respawn_wait(s, here.id, [x, *xs])  # (the spawn guard's newborns about the next pass)
        now = d.at(here.id, x, s.monsters) + due[0]
        need = self._spawns_in_reach(s, x)
        best = None  # (danger, distance, x)
        for g, newborn in zip(xs, due[1:]):
            if self._spawns_in_reach(s, g) < need or self._in_the_way(s, x, g):
                continue
            there = d.at(here.id, g, s.monsters) + newborn
            if best is None or (round(there, 6), abs(g - x)) < (round(best[0], 6), best[1]):
                best = (there, abs(g - x), g)
        return now, best

    # --- waiting on a rope ----------------------------------------------------------------------------

    def _refuge(self, s: Scene) -> Refuge | None:
        """Rule 9: a rope to wait on instead (attack.ropes), or None: one she can get onto from her platform and back
        (by a walk along it that passes no monster that counts: _in_the_way), safer than anywhere along her platform
        (_safest_along, where she is too) by ROPE_GAIN of the HP she has left (a touch takes more of less), what she
        doesn't regain there while she waits (REGEN_S for HANG_EXPECT_S) on top of its own danger; and whose foot,
        where she comes back down to shoot what she waited for, is no more than WAIT_GAIN more dangerous than the
        safest place along her platform as she plans (_foot_floor). On Tree Dungeon that leaves none: 31's rope
        comes down 40 units from its Horny Mushroom's spawn point (0.16 there, 0.12 at 31's safest place; taken, in
        the simulator she made 25% less EXP for the same HP, 8 seeds of 8 minutes on 31 alone), 10's by its two
        (0.23 against 0.18), and over 20 and 19 the ropes hang at the platforms' very ends (attack.ropes.ROPE_IN).
        Only while she waits for more than getting on and off takes: this ground the only one she'd go to, or a
        kill of hers due back no sooner. Never where nothing counts as a danger, nor ROPE_AGAIN_S after she got off
        one for anything but a monster to shoot."""
        d, here, x, now = self.danger, s.platform, s.where.x, time.monotonic()
        if d is None or not d.any or self.h.knocked(s) or now < self._rope_rest:
            return None
        due = self._due_back()
        if due is None and self._elsewhere():
            return None  # she'll leave on her patience (1-2 s): no time for a rope
        options = ropes.refuges(self.model, d, here.id, s.monsters)
        if not options:
            return None
        at, best = self._safest_along(s)
        safest = min(at, best[0]) if best is not None else at
        cost = ropes.REGEN_S * HANG_EXPECT_S
        floor = self._foot_floor(here.id)
        for r in options:  # the safest first
            if due is not None and due < r.on_s + r.off_s:
                continue
            if safest - (r.danger + cost) < ROPE_GAIN * max(s.hp, 0.1) or self._in_the_way(s, x, r.x):
                continue
            if d.at(here.id, r.x) > floor + WAIT_GAIN:
                continue  # she comes back down by a spawn point: there respawns land on her
            return r
        return None

    def _foot_floor(self, pid: int) -> float:
        """The danger of the safest of _wait_place's places along platform `pid` as she plans (every spawn point's
        monster at its point): what the foot of a rope she'd wait on is held to (_refuge)."""
        p, d = self.model.platform(pid), self.danger
        xs = spots._stand_xs(p.left + GROUND_MARGIN, p.right - GROUND_MARGIN) or [(p.left + p.right) / 2]
        return min(d.at(pid, g) for g in xs)

    def _hang(self, s: Scene, r: Refuge) -> None:
        """Rule 9 on a rope: to its x along her platform, onto it and to her height there (control.mover.hang)."""
        at, best = self._safest_along(s)
        safest = min(at, best[0]) if best is not None else at
        self.log(f"  waiting on the rope at x {r.x:.0f} (feet at y {r.y:.0f}; danger {r.danger:.2f} there, "
                 f"{safest:.2f} at best on platform {s.platform.id})")
        telemetry.emit("event", kind="rope", text=f"on the rope at x {r.x:.0f}, y {r.y:.0f}")
        self.nav.mover.hang(r.rope, r.y, r.from_top, s)
        self._hanging, self._hung_at = r, time.monotonic()
        return None

    def _on_refuge(self, s: Scene) -> bool:
        """Whether look `s` (on no platform) has her on the rope she went to wait on (by it, between its ends)."""
        r = self._hanging
        return (r is not None and s.where is not None and abs(s.where.x - r.x) <= ON_ROPE_X
                and r.rope.top - HANG_TOLERANCE <= s.where.y <= r.rope.bottom + HANG_TOLERANCE)

    def _while_hanging(self, s: Scene) -> None:
        """A look while she waits on a rope (a whole-screen one: the ground's monsters): off it (control.mover.unhang)
        when there's a reason to (_off_rope), else a moment more. Potions and the safety stop as anywhere
        (Hunter.health_check)."""
        self._note("wait")
        wide = self.h.look(mobs=True, wide=True, loot=False)
        if wide.where is not None and wide.platform is None:
            s = self._seen = wide
        why = self._off_rope(s)
        if why is None:
            time.sleep(0.1)
            return None
        r = self._hanging
        self.log(f"  off the rope at x {r.x:.0f}: {why}")
        self.nav.mover.unhang(r.rope, s.where.y, r.from_top)
        self._hanging = None
        if why != OFF_TO_SHOOT:
            self._rope_rest = time.monotonic() + ROPE_AGAIN_S
        return None

    def _off_rope(self, s: Scene) -> str | None:
        """Why she gets off the rope she waits on now, or None (Hunter.health_check first: potions, the safety stop):
        the rope no safer than where she'd land; a monster she hunts on the ground (or in reach of where
        she lands), with none where she lands so near that a throw at it would be the claw's swipe (stance.close_to:
        she lets it pass under her; waiting for a Horny Mushroom to walk out of its keep-off, then 162, she hung for
        minutes on 31 in the simulator); nothing due back and another ground to go to; or HANG_MAX_S up there."""
        r, d, now = self._hanging, self.danger, time.monotonic()
        self.h.health_check(s)
        seen = s.monsters
        below = d.at(r.platform, r.x, seen)
        if d.on_rope(r.rope, s.where.y, seen) > below:
            return "no safer than below"
        land = self.model.platform(r.platform)
        wanted = [m for m in seen if (self.kinds is None or m.kind in self.kinds)
                  and (m.platform == self.visit.ground
                       or reach.can_hit(land, r.x, land.y_at(r.x), m, others=seen))]
        if wanted and not any(m.platform == r.platform and close_to(m, r.x, self._halves) for m in seen):
            return OFF_TO_SHOOT
        if self._due_back() is None and self._elsewhere():
            return "nothing due back, and another ground to go to"
        if now - self._hung_at >= HANG_MAX_S:
            return f"{HANG_MAX_S:.0f} s on it"
        return None

    def _in_the_way(self, s: Scene, x: float, g: float) -> bool:
        """Whether a walk along her platform from map x to `g` would touch a monster that counts as a danger: one of
        this look's on her platform ahead of her, up to where its body (its half width and hers) meets her at `g`."""
        d, here = self.danger, s.platform.id
        for m in s.monsters:
            t = d.threats.get(m.kind)
            if m.platform != here or t is None or t.share <= 0:
                continue
            if (x < m.x <= g + t.half + HER_HALF) if g > x else (g - t.half - HER_HALF <= m.x < x):
                return True
        return False

    def _spawns_in_reach(self, s: Scene, x: float) -> int:
        """How many of the visit's ground's spawn points (of the kinds she hunts) a star from map x on her platform
        reaches (reach.can_hit, a monster of its kind standing there; terrain stopping it counts)."""
        here, ground = s.platform, self.visit.ground
        y = here.y_at(x)
        n = 0
        for pid, sp in self.model.spawn_points:
            kind = kind_of(sp.name or str(sp.mob_id))
            if pid != ground or (self.kinds is not None and kind not in self.kinds):
                continue
            q = self.model.platform(pid)
            m = Monster(Mob(kind, 0, 0, 0, 0, 1.0), sp.x, q.y_at(sp.x), pid)
            n += reach.can_hit(here, x, y, m)
        return n

    def _note(self, rule: str) -> None:
        """A pass decided by `rule`: counted, and the time from it to the next pass's note is that rule's (`rule_s`,
        HuntReport; a pause of her looks isn't: `_paused`); a change of rule publishes the 'hunt' record."""
        self.report.rules[rule] += 1
        now = time.monotonic()
        last = getattr(self, "_noted", None)
        if last is not None:
            self.report.rule_s[last[0]] += now - last[1]
        self._noted = (rule, now)
        if rule != self._rule:
            self._rule = rule
            self._publish()

    def _publish(self) -> None:
        """The 'hunt' record for the mini-map (gamelab.ui): why she does what she does. Sent when the rule,
        the objective or the piles change."""
        s = self._s
        if not telemetry.recording() or s is None or s.platform is None:
            return
        first = getattr(self, "_swept", None)  # (the sweep's next pile: 'urgent', the mini-map's red one)
        going = None if first is None else self._pile_of.get(first.spot.key)
        piles = [{"x": round(p.x, 1), "y": round(p.y, 1), "platform": p.platform, "status": p.status,
                  "hers": p.hers, "urgent": p is going, "worth": round(p.value)}
                 for p in self.h.drops.known if self._candidate(p)]
        v, o = self.visit, self.obj
        key = (self._rule, None if o is None else o.text(), None if v is None else v.ground,
               tuple((d["x"], d["platform"], d["status"], d["urgent"]) for d in piles))
        if key == self._published:
            return
        self._published = key
        best = None if v is None else self._best_rate(s)
        telemetry.emit("hunt", mode="hunt", rule=self._rule, objective=key[1], ground=key[2],
                       rate_here=None if v is None or v.arrived is None else round(self._rate_here(), 3),
                       rate_to_beat=None if best is None else [round(best[0], 3), best[1]], piles=piles)


def _same(a: Monster, b: Monster) -> bool:
    """Whether monsters `a` and `b` are one: of one track wherever each has it (2026-11-04 at 00:25:14.839 one track counted
    twice, -52 and -152), or of a kind, on a platform, within memory.MATCH_SLACK (two tracks that near mostly one monster
    the tracker lost and found again under a new id)."""
    ta = getattr(a, "track", None)
    if ta is not None and ta == getattr(b, "track", None):
        return True
    return a.kind == b.kind and a.platform == b.platform and abs(a.x - b.x) <= 40.0


def _safe_share(sp: Spot) -> float:
    """The share of a ground's EXP a second that counts, hunted from spot `sp`: 1 - GROUND_DANGER x its danger, 0 at
    least."""
    return max(0.0, 1.0 - GROUND_DANGER * sp.danger)


def _contents(p: Pile) -> str:
    """What a pile holds, for the log: "slime-bubble, 2 mesos" (as the last look that saw it saw it), or "drops"
    before a look has."""
    if not p.items:
        return "drops"
    return ", ".join(k if n == 1 else f"{n} {k}" for k, n in Counter(p.items).most_common())
