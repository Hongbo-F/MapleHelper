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
                       travelling or not; with something in reach only if it goes first (EXP before loot, below)
   2  high ground      on a platform she can't safely stand on, with a spot that reaches her ground: back up;
                       also back to the spot from any other platform (after a pile there), unless the
                       ground is in reach from it or she's leaving
   6  urgent pile      drops whose remaining life is under the time to get there plus URGENT_MARGIN, within
                       URGENT_TRAVEL; above shooting if it goes first (below), or drops lay until they vanished
                       where respawns never stop
   3  shoot            a target in reach from her spot (or the ground she hunts on): throw, step out from
                       under, close in along her platform
   4  in passing       travelling: a target in reach, thrown at from where she is (or a step of PASS_STEP at
                       most out from under one above her: from 22 at 21's), PASSING_S at most per platform per
                       objective (the navigator shoots what stands in the way of a move)
   4b finish           travelling to another platform, done shooting in passing: the monsters she has hurt
                       and not killed (fight.owed), when a throw from here or a walk of fight.FINISH_WALK at
                       most along her platform gets to them (fight.finish_act), FINISH_S at most per
                       objective; she used to leave monsters half hunted
   5  pile here        drops on her platform within PILE_NEAR, nothing by them or between her and them (also
                       piles she passes while travelling)
  5b  on the way       travelling: a pile a stop at costs DETOUR_S at most more than the trip without it (the
                       planner's times, motion.route.Planner.arrive), DETOUR_REACH away at most, DETOURS per
                       objective at most; she used to leave such piles and go back across the map for them when
                       they were about to vanish
   7  better spot      wanted monsters seen out of reach, and a spot that reaches clearly more of them
   8  lull loot        nothing wanted on this ground and a kill of hers due back: piles she can pick up first
   9  wait             monsters about or due back: stand still, for the patience at most, where she is or at the
                       safest place along her platform that still has the ground's spawn points in reach
                       (_wait_place: not between two Horny Mushrooms' spawn points), or on a rope from her
                       platform where nowhere along it is safe enough (_refuge, attack.ropes)
  10  loot elsewhere   leaving: the piles left when she decided, the most worth per second of travel first
  11  advance          leaving: one trip to the next ground's spot

EXP before loot (`_loot_first`; the user, 2026-10-24: "The hunt should be driven by maximizing the exp gain"): no pile
goes ahead of shooting what's in reach by a fixed priority. One that will still be there when she has killed what's in
reach waits; one at risk by then (`_at_risk`: its life running out, another player by it once it's no longer hers
alone) goes first only if what waiting would likely lose of it, at MESO_EXP an meso, is more than the EXP she'd make
from what's in reach (Attacks.kill_s) over the trip there and back (rules 6a and 6). A trip for a valuable pile is
called off when that no longer holds (a monster comes into reach), and so is picking one up (Looter.yield_to): the pile
stays known, and is gone for again once nothing is in reach. (On a trip for an urgent pile she shoots in passing, as
before: it has URGENT_MARGIN to spare.) In the live hunts of 2026-10-24 (1474 looks with a wanted monster in
reach on another platform than hers), she didn't throw within 1.5 s on 184: on 78 she was picking up a pile, on 43 the
valuable pile rule had her off to one (20 of those deciding it or on the way).

Her drops are priced by the HP getting them would cost (`_pile_block`, `_pile_price`; from 2026-10-26): no pile is
gone for, walked to or stood on with a monster that counts as a danger within PILE_CLEAR of where she'd land, walk and
pick up (seen, or about though the look missed it: localization.unseen; with DUE_BACK, one just due back at its spawn
point after a kill of hers there too); the pile waits, still known, and a trip for one is called off at the chain that
sees it so. What's left of the danger there is the pile's price against what it's worth (EXP before loot: HP_EXP a
bar). Live on 2026-10-25 all 12 of the Horny Mushrooms' touches and all 3 knocks off 23 came on stays on 23 for drops.
And what her fights go by takes those about her that the look didn't find (`_unseen`: the feed's hidden tracks, her
memory's of the last UNSEEN_S): around her for the stance's danger and its moves, and those on her platform that
count as a danger as targets too (`_with_unseen`).

She leaves a ground by the rate rule (`_leaving`): her EXP rate there falls clearly below what the best
other ground would give, travel included, or nothing is left there and nothing is due back. Not for the ground she'd
pick again at once (the only one: `--grounds 23`, or a value map with one): then it's waiting (`_elsewhere`). Objectives
are suspended, not ended, when a higher rule fires: they keep their route weights and their clock
(OBJECTIVE_S, suspensions included).

What she knows of the monsters off her band is her memory of them (Hunter.memory, localization.memory: every look's,
kept across looks, forgotten by looks that cover their place without them or by age), and every look of hers has the
monsters perception follows on the whole screen (Scene.swept: localization.feed.Feed.look), so the memory is kept up
while she shoots: the ground's monsters out of reach (rule 7), whether anything is left on it (the fast exit), and what
guards a pile elsewhere (rule 5b) go by it. Her own platform's, which the fights react to, are the look's as before.

The numbers marked "guess" are starting values for the simulator's A/B runs (gamelab.sim.harness); none
is measured in game. Respawns and how long drops last were measured on the live telemetry (analysis.respawn), and
didn't settle RESPAWN and DROP_LIFE: their comments say what they did show.
"""

import random
import time
from collections import Counter
from dataclasses import dataclass, field

from maplehelper.engine import LEFT, RIGHT

from .. import telemetry
from ..attack import reach, ropes, spots
from ..attack.ropes import Refuge
from ..attack.spots import GROUND_MARGIN, Spot
from ..attack.stance import (MIN_MOVE, act_text, close_to, drop_limits, jump_hits, open_above, overhead, stance,
                             too_close)
from ..control.mover import HANG_TOLERANCE, ON_ROPE_X
from ..control.navigator import TRANSIT, Choice, TripState
from ..localization.monsters import Monster, Tracker, tracks_record
from ..localization.players import near_rival
from ..localization.scene import Scene
from ..localization.unseen import UNSEEN_S, unseen
from ..motion.bodies import HER_HALF
from ..motion.physics import WALK_SPEED
from ..perception.framecheck import LookLost
from ..perception.kinds import kind_of
from ..perception.mobs import Mob
from . import fight
from .keycheck import Doubt, KeyDoubt
from .chatcheck import ChatDue
from .statcheck import StatsDue
from .fight import THROWS, FightState, Report, kills_text
from .ledger import DROP_LANDS, Pile
from .looting import DROPS_LAST, SETTLE, DropSpot
from .tour import PRIOR

IN_HER_WAY = fight.IN_HER_WAY  # any monster this close on her platform is a target, whatever its kind
FINISH_S = 8.0         # rule 4b: s of finishing what she has hurt per objective, before (and while) she goes.
                       # Unbounded, in the simulator's basin, where respawns never stop, there was always one
                       # she had hurt in reach: piles and spots ran out their OBJECTIVE_S (52 times in 30 runs of
                       # 20 minutes, against 26) and she picked up 2.9 points less of her drops (and with it,
                       # where the objective was on her own platform, 55)
PASSING_S = 15.0       # rule 4: s of shooting per platform per objective while passing through ...
PASS_STEP = fight.FINISH_WALK  # ... and the longest step out from under a target there (_pass_step)
PILE_NEAR = 150        # rule 5: drops on her platform this close are picked up from where she is
PILE_GUARD = 110       # ... unless a monster is this close to them (looting.TOO_CLOSE, today's guard)
PILE_TRIES = 2         # a pile is gone after at most this many times (a suspension counts)
URGENT_MARGIN = 30.0   # rule 6: s of slack before a pile's expected end (time for a guard to be shot first) ...
URGENT_TRAVEL = 12.0   # ... for piles at most this far (s of travel): farther ones wait until she leaves a
                       # ground (rule 10); one 84 s round trip for a pile from the basin took a quarter of a run
# s a drop is still worth going for (looting.DROPS_LAST, the code's own assumption; the simulator's lie 180 s). The live
# telemetry of 2026-10-19 to 21 doesn't settle it (analysis.respawn: README, Respawns and drop life): of 337 drops
# followed, 95% were still lying at 60 s (95% interval 89-97%; 7 followed that long, none past 88 s, as she picks
# hers up within seconds and the next kill lands by the rest). Longer than a minute, which 90 and 180 both are.
DROP_LIFE = DROPS_LAST
OTHERS_AFTER = 30.0    # guess: s before drops no kill of hers explains are worth a trip (not hers before)
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
PICK_S = 0.6           # s standing on a pile (4 presses)
# Her drops, priced by the HP getting them would cost (`_pile_block`, `_pile_price`). Live on 2026-10-25 (Tree Dungeon,
# tour --loop 23:43-23:54, 340 EXP a minute; D:/MapleHelper-scratch/analysis-1025/damage-skeptic on23.py, ctx10.py) all
# 12 of the Horny Mushrooms' touches (1.90 of the 3.4 bars of HP she lost) and all 3 knocks off 23 down to 29 (43 s back
# up) came on 6 stays on 23 whose objective was a pile (on 23, 22 or 20: on the way, lull loot, urgent pile, a trip
# there): she landed or walked beside a Horny seen or remembered there (the service's track 130 alone: 6 touches and a
# fall), or one just back at its spawn point (2 of them on top of her), and there she killed no faster (9.4 a minute)
# than from her spot on 24 (11.2). So she doesn't land on a pile's platform for it, walk to it or stand on it with a
# monster that counts as a danger (attack.danger) within PILE_CLEAR of that stretch: one the look found there, one about
# that it didn't (localization.unseen: the feed's hidden tracks, her memory's of the last UNSEEN_S), or (DUE_BACK) a
# spawn point's monster due back. The pile waits, still known, for a look that clears it; a trip for it is called off
# at the chain that sees it not clear (`_call_off`). 100 units, not the analysis's "~150" (a Horny Mushroom walking at
# her 100 units a second is on her in about a second at 150): in the simulator (Tree Dungeon, crowd, `--world
# zero-walls --world near-misses`, `--attack choice`, 8 seeds of 5 minutes, before the review of 2026-10-26) with 150 she lost 0.92 bars of HP a minute alive, drank 13.6
# potions a run and picked up 115 mesos' worth a minute; with 100 0.87, 12.9 and 130 (EXP a minute 297.7 and 307.4);
# without it (and HP_EXP) 1.13, 16.9 and 170. Replayed on the live hunt of 2026-10-25 (78 times she went for a pile),
# 150 has 47 of them wait, 100 44: the stays of both carried 7 of her touches and 1.04 bars of HP, of the 3.4 she lost
# (D:/MapleHelper-scratch/hunt-ab/B/scripts/b1_replay.py). With the feed's tracks of the dead left out (localization.
# unseen.hidden) and no due-back block (DUE_BACK), 17 of the 78 wait at 100 (18 at 150): 79 s of stays that carried 5
# touches and 0.81 bars, 0.61 bars a minute, against 0.24 a minute on the 61 let through, for 20% of the piles' worth
# (D:/MapleHelper-scratch/hunt-ab/B/fix/b1_replay2.py; with ghosts and due-back 0.34 against 0.31: review of 2026-10-26).
PILE_CLEAR = 100.0
# A spawn point's monster is due back from DUE_FROM to DUE_TO s after a kill of hers of its kind on its platform (the
# point of that kind nearest where it died). A spawn point's cooldown is 5 s by MeowDB (and its 7.56 s tick); live on
# 2026-10-25 a monster new to her memory was back on Tree Dungeon's 23 (two Horny Mushroom points) a median 7.2 s after
# her kill there, on 24 11.1 s, 20 15.7 s (exp-time-skeptic refill.py: upper bounds); the user's analysis's "~5-12 s".
# Only her own kills (others' aren't placed). Before DUE_FROM the point is empty: the time to pick up a kill's drops.
# Off (DUE_BACK; review of 2026-10-26): replayed on the live hunt of 2026-10-25 (above) the 20 stays it alone would have
# held carried 1 touch and 0.16 bars in 71 s, 0.14 bars a minute, the safest of all (the 61 let through 0.24, the 17
# the rest holds 0.61), for 980 mesos' worth; in the simulator (the four worlds of the README's table, 8 seeds of 5
# minutes) it made no difference to her HP past the seeds' spread (+10%, -7%, +3%, +14% without it, none 1.3
# standard errors), and grinding she picked up 28% and 10% more without it (D:/MapleHelper-scratch/hunt-ab/B/fix/out/
# round3.txt).
DUE_BACK = False
DUE_FROM = 5.0
DUE_TO = 12.0
# What a bar of her HP is worth in EXP, for a pile's price (EXP before loot, `_loot_first`): her pet drinks Orange
# Potions of 250 HP at its 70% warning (sim.world.HP_POTION, PET_HP: 6 in the 10.95 minutes live on 2026-10-25), about
# 160 mesos each at a shop (a guess: v83's price), so a bar of her 774 HP is 3.1 of them, 500 mesos, 250 EXP at
# MESO_EXP; the knocks off a platform a touch can bring aren't counted (attack.stance keeps her off a long drop's end).
# The user is short of money for potions.
HP_EXP = 250.0
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
# EXP before loot (`_loot_first`): what a meso of drops is worth to her in EXP, against what she'd shoot meanwhile. A
# guess putting EXP first: at it a valuable ore on Tree Dungeon (looting.Worth.valuable 66.8) is worth 33 EXP, under a
# Horny Mushroom's kill (41), and an at-risk pile 4 s there and back goes first when what's in reach makes her under
# 8 EXP a second (the regular attack's 3.7 at a Horny Mushroom; Lucky Seven's 12.5 doesn't let it).
MESO_EXP = 0.5
# s a kill's drops are its killer's alone (the game's ownership: nobody else can pick them up meanwhile; looting.
# cant_pick has seen mesos she stood on twice stay). The simulator's (sim.world.Realism.rival_owns), not measured.
OWN_S = 15.0
RIVAL_TAKES = 0.5      # guess: the chance another player by a pile that isn't hers alone takes it before she's there
EXP_FIRST = True       # EXP before loot on; off, the piles' fixed priorities as before (6a and 6 ahead of shooting)
VALUABLE_TRAVEL = 15.0  # ... when it's this many s of travel away at most (URGENT_TRAVEL's 12, and a little),
VALUABLE_TRIES = 2      # ... this many times at most (then it's a pile like the others): by one, a monster can stand
                        # (guarded: looting.collect leaves it, not a try), and she can't shoot it from where it lies
# Whether her look is at the whole screen, with the drops on every platform in view, once a kill's of hers on another
# platform have landed (Hunt._far_drops): from a spot the band's drops are her own platform's, and one on the ground
# she shoots was seen only once she stopped shooting. In the simulator (loop_valuable_drop_at_once, seeds 1-16) she
# set off for the Slime Bubble before the mushrooms were dead on 16 seeds with it, on 12 without; the look takes about
# 120 ms more in game than the band's (README: What a look costs), once a far kill.
FAR_DROPS_LOOK = True
# Rule 5b: travelling, a pile a stop at costs this many s at most on top of the trip (her way to it, PICK_S on it
# and her way on from it, against her way on from here: Planner.arrive), DETOURS piles at most per objective. In
# the hunt loops of 2026-10-21 (12 runs, 70 minutes) she made 44 trips for piles about to vanish (rule 6), 668 s in
# all, 27 of them of 10 s or more; 23 of those 44 (280 s) were for piles a trip of hers had passed within 2 s of
# before, 15 of them within 0.7 s: on the route, only the stop on them. A look at the end of a chain stops it for one
# within DETOUR_S; the pass that decides allows DETOUR_SLACK more (her real place is a few units off the
# predicted one), so she doesn't stop again and again for one just over. Only for a pile DETOUR_REACH s away at
# most: one near the end of a long trip made the whole trip its own while the trip's clock ran (OBJECTIVE_S). In the
# simulator's hunt loop (preset game, 60 runs of 20 minutes), with no such bound she picked up 89.1% of her drops, for
# 127.8 EXP a minute, and objectives ran out 111 times; with 6 s 89.4%, 127.4 and 99; with 4 s 89.8%, 128.0 and 99;
# with 2.5 s 90.1%, 128.3 and 99 (before the stops shared the trip's budgets, `_go_for`; without the rule 88.9%,
# 128.6 and 61).
DETOUR_S = 2.0
DETOUR_SLACK = 0.5
DETOURS = 3
DETOUR_REACH = 2.5
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
# after its kill, before OTHERS_AFTER makes them worth a trip anyway, so only piles taken for no kill of hers are
# left; whether a real player's are worth leaving is for the game to say.
PLAYER_PILES = False


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
    why: str = ""              # the rule that set it ("valuable pile", "on the way", ...)
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


@dataclass
class HuntReport:
    """What a hunt did."""
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
    stop: str = ""

    def __str__(self):
        read = ("not read" if self.exp_read is None else
                f"+{self.exp_read}" + (f", {self.exp_read * 60 / self.seconds:.1f} a minute" if self.seconds else ""))
        return (f"{self.seconds:.0f} s; grounds {self.grounds}; kills: {kills_text(self.kills)} "
                f"(EXP gains {self.exp_gains}, EXP {self.exp:.0f}; off her EXP number {read}); throws {self.throws}; "
                f"piles {self.piles}; ended: {self.stop}")


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
        # Leaving a ground: the piles to collect first (None: not leaving), the ones left when she decided
        # to go. Only those: going after the drops of monsters shot on the way to them too, every kill on
        # a trip made another trip, and she never left. The decision itself is made again at every pass
        # (kept, a weak reason such as "nothing came into reach" lost respawns that came after it).
        self._exit_piles: list[Pile] | None = None
        self._exit_reason: str | None = None  # why, kept while she's off the ground and its spot (see `_at_ground`)
        self._no_high_until = 0.0            # after a failed trip back up to the spot, not again for a while
        self._tracker = Tracker(self.model)  # the monsters for the mini-map (the navigator sends none with an escort)
        self._s: Scene | None = None         # the last look, for the 'hunt' record
        self._seen: Scene | None = None      # the latest look of all (a trip starts from it if it's still fresh)
        self._published = None
        self._end = float("inf")  # when the running hunt ends (run)
        self._loot_said: dict[int, bool] = {}  # pile (its id) -> whether it went first when she last said (_loot_first)
        self._unseen_for, self._unseen_list = None, []  # the look `_unseen` was worked out for, and its answer
        self._landings: tuple = (None, {})  # the look `_landing` was worked out for, and platform -> where she'd land
        self._block_said: dict[int, bool] = {}  # pile (its id) -> whether it waited when she last said (_pile_block)

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
                    if self._pass() == "stop":
                        break
                except LookLost as lost:  # the game doesn't look as expected: stand still until it does (senses)
                    paused = time.monotonic()
                    back = self.h.senses.regain(lost, end)
                    self._paused(time.monotonic() - paused)
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
                        back = self.h.senses.regain(lost, end)
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
                        back = self.h.senses.regain(lost, end)
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
        self.report.seconds = time.monotonic() - start
        self._log_memory()
        if meter.since is not None:
            self.report.exp_read = meter.total - (read or 0)
        self.report.stop = self.report.stop or "time up"
        self.tour.stand_by(mode=TRANSIT)
        return self.report

    def _paused(self, seconds: float) -> None:
        """She stood still `seconds` (LookLost: Senses.regain): her clocks move on by as much, so the pause isn't time
        without progress (STUCK_S: a pause of 125 s ended the hunt as stuck on the first look after it, in the
        simulator's loop_tree_run), on an objective (OBJECTIVE_S, its trip's: TripState.paused), at a ground (its EXP a
        second on disk, MIN_STAY, GROUND_S), on her spot (SPOT_REST), idle (her patience) or on a rope (HANG_MAX_S,
        ROPE_AGAIN_S). The game's own clocks (respawns, drops' lives) ran on meanwhile and aren't moved."""
        self._progress += seconds
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
        if o is not None and o.kind == "pile" and o.pile is not None and (why := self._pile_block(s, o.pile)):
            return self._call_off(o, why)  # not clear (PILE_CLEAR): it waits, still known
        if o is not None and o.kind == "spot" and o.platform == o.ground == s.platform.id:
            # A ground no standable spot reaches is hunted on itself: anywhere on it she's there (stance
            # closes in). Waiting to reach the spot's x, she was interrupted by close monsters for good.
            self._arrived(o)
        now = time.monotonic()
        targets, visible, hidden = self._targets(s, now)
        if self._close_quarters(s, targets):
            return self._shoot(s, targets, visible, hidden, now, "close quarters", approach=False)
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
        if (pile := self._pile_here(s)) is not None and pile is not self.obj.pile:  # one she passes
            return self._pick_here(s, pile)
        if (way := self._detour(s, DETOUR_SLACK)) is not None:
            extra, pile = way
            self.obj.detours += 1
            self.log(f"  {_contents(pile)} on platform {pile.platform} at x {pile.x:.0f}: on the way ({extra:+.1f} s)")
            return self._go_for(pile, "on the way", s)
        if self.obj.kind != "pile" and self._exit_piles is None and (pile := self._urgent_pile(s)) is not None \
                and self._loot_first(s, pile, targets):
            return self._go_for(pile, "urgent pile", s)
        return self._travel(self._rule if self._rule in ("advance", "loot elsewhere", "urgent pile", "on the way",
                                                         "high ground", "back to spot", "better spot", "lull loot")
                            else "travel")

    def _at_ground(self, s: Scene, targets, visible, hidden, now: float) -> str | None:
        v, here = self.visit, s.platform.id
        leaving = self._exit_piles is not None  # the piles left when she decided to go (`_leave`)
        if not leaving and not self._standable(here) and v.spot.platform != v.ground \
                and time.monotonic() >= self._no_high_until:
            return self._to_spot(v.spot, "high ground")
        # About to expire, a pile outranks shooting: where respawns never stop, something is always in
        # reach, and below shooting the drops lay there until they vanished (a quarter of them). Not while
        # she's leaving: then only the piles left when she decided; her kills on the trips to them made
        # new urgent piles, and she never left.
        if not leaving and (pile := self._urgent_pile(s)) is not None and self._loot_first(s, pile, targets):
            return self._go_for(pile, "urgent pile", s)
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
        if (pile := self._pile_here(s)) is not None:
            return self._pick_here(s, pile)
        wide = self._wide(s)
        if not leaving and (spot := self._better_spot(wide)) is not None:
            v.changes += 1
            return self._to_spot(spot, "better spot")
        if self._nothing_here(wide) and (pile := self._lull_pile(wide)) is not None:
            return self._go_for(pile, "lull loot", wide)
        # Away from the ground and its spot (an exit pile), the decision stands: from there she can't
        # judge the ground (out of view, it looked empty, and was skipped for nothing).
        away = here not in (v.ground, v.spot.platform)
        reason = self._exit_reason if leaving and away else self._leaving(wide)
        if reason is None:
            self._exit_piles = self._exit_reason = None  # staying after all
            return self._wait(wide, targets)
        if self._exit_piles is None:  # decided: the piles left now, and only those (see __init__)
            self._exit_piles = [p for p in self.h.drops.known if self._fresh(p)]
        self._exit_reason = reason
        return self._leave(wide, reason)

    def _leave(self, s: Scene, reason: str) -> None:
        """Rules 10 and 11: the piles left when she decided to go, then the next ground."""
        if (pile := self._best_pile(s, near_only=False, among=self._exit_piles)) is not None:
            return self._go_for(pile, "loot elsewhere", s)
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
        """New kills in the ledger: the visit's EXP and kill times, the report, and progress. A kill's EXP is its
        kind's, placed or named by its gain's size (localization.gains), else the average kind's worth hunting."""
        kills = self.h.ledger.kills[self._kills_seen:]
        self._kills_seen = len(self.h.ledger.kills)
        avg = self._avg_exp()
        for k in kills:
            kind = k.kind or k.named
            exp = self.tour.exp.get(kind, avg) if kind is not None else avg
            self.report.exp_gains += 1
            self.report.exp += exp
            if kind is not None:
                self.report.kills[kind] = self.report.kills.get(kind, 0) + 1
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
        a kind that counts as a danger among the targets too, so a fight keeps off them and shoots them (rule 1 and
        stance: one on her platform that can touch her first). Not those on other platforms: thrown at where they were
        last seen, they'd be the throws at what's dead or gone that the live hunt of 2026-10-25 wasted 52 of 325 on
        (jumps-perception-skeptic). Where nothing counts as a danger (the first map), as they were."""
        d = getattr(self.h, "danger", None)  # (tests make a hunter without one)
        if s.platform is None or s.where is None or d is None or not d.any:
            return targets, visible, hidden
        more = [m for m in self._unseen(s) if not any(_same(m, h) for h in [*hidden, *visible])]
        if not more:
            return targets, visible, hidden
        here = s.platform.id
        mine = [m for m in more if m.platform == here and d.share(m.kind) > 0 and not any(_same(m, t) for t in targets)]
        return targets + mine, visible, hidden + more

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
        """Rule 1: one of `monsters` on her platform is as close as stance's own threshold, where it will be when her
        next attack starts (too_close, fight.pace_of: the claw would swipe it, by its size; or nearer than its danger
        warrants: Danger.keep_off)."""
        x, pace = s.where.x, fight.pace_of(self.h, s)
        facing = getattr(getattr(self.h, "actions", None), "facing", None)
        return any(m.platform == s.platform.id
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
        if log:
            return fight._take_stance(self.h, self.fs, s, targets, visible, hidden, approach, toward=toward,
                                      jumps=jumps, goal=goal, edges=edges, up=up, limits=limits)
        here = s.platform
        castable = fight.castable_of(self.h)
        return fight.cast_or_wait(self.h, stance(
            s.where.x, s.where.y, here, targets, approach, self.h.actions.facing,
            can_jump=lambda a, b: open_above(self.model, here, a, b), tops=self._tops, halves=self._halves,
            around=visible + hidden, danger=self.danger, jumps=jumps, toward=toward, goal=goal, edges=edges, up=up,
            pace=fight.pace_of(self.h, s), touch=self.fs.touch, limits=limits, moves=self.fs.moves,
            castable=castable, **fight.wounds_of(self.h, s)), s, castable)

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

    # --- loot ---------------------------------------------------------------------------------------

    def _worthless(self, p: Pile) -> bool:
        """A pile she's going for that isn't worth the rest of the trip: seen gone, or items with her
        inventory full. Not its age: how long drops last is a guess (DROP_LIFE; the simulator's last
        twice as long), and dropping trips at that age lost 6% of her drops there. The objective's
        clock bounds the trip."""
        return p.status == "gone" or (not p.mesos and self.h.looter.inventory_full)

    def _fresh(self, p: Pile) -> bool:
        now = time.monotonic()
        return (p.status != "gone" and p.platform is not None and now - p.since < DROP_LIFE
                and p.tries < PILE_TRIES and (p.hers or now - p.since >= OTHERS_AFTER)
                and (p.mesos or not self.h.looter.inventory_full) and not self._theirs(p))

    def _theirs(self, p: Pile) -> bool:
        """Whether pile `p` is another player's (PLAYER_PILES): no kill of hers explains it, it holds nothing
        valuable, and another player is by it in the latest look."""
        s = self._seen
        return (PLAYER_PILES and not p.hers and (p.dearest or 0.0) < self.valuable and s is not None
                and near_rival(s.players, p.x, p.y) is not None)

    def _pile_here(self, s: Scene) -> Pile | None:
        """Rule 5: drops seen on her platform within PILE_NEAR, nothing by them, nothing in between."""
        here, x = s.platform.id, s.where.x
        mine = [m for m in s.monsters if m.platform == here]
        if mine and not self._standable(here):
            return None
        for d in sorted((d for d in s.loot if d.platform == here and abs(d.x - x) <= PILE_NEAR),
                        key=lambda d: abs(d.x - x)):
            if any(abs(m.x - d.x) < PILE_GUARD for m in mine):
                continue
            pile = next((p for p in self.h.drops.known if p.platform == here and abs(p.x - d.x) <= 60), None)
            if pile is not None and self._fresh(pile) and not self._pile_block(s, pile):
                return pile
        return None

    def _pick_here(self, s: Scene, pile: Pile) -> None:
        self._note("pile here")
        self._tried(pile)
        self.h.looter.walk_piles(s, [DropSpot(pile.x, pile.y, pile.platform, pile.since, pile.mesos, True)], margin=0)
        self._idle_since = None
        return None

    def _tried(self, pile: Pile) -> None:
        self.h.drops.tried(pile)
        self.report.piles += 1

    def _travel_to_pile(self, s: Scene, p: Pile) -> float:
        here = s.platform
        if here is not None and here.id == p.platform:
            return abs(p.x - s.where.x) / WALK_SPEED
        t = self.nav.planner.travel(here.id, s.where.x, p.platform) if here is not None else None
        return 1e6 if t is None else t + 1.0

    def _urgent_pile(self, s: Scene) -> Pile | None:
        """Rule 6: a pile whose remaining life is under the time to get there, pick it up, and the margin."""
        now, best = time.monotonic(), None
        for p in self.h.drops.known:
            if not self._fresh(p):
                continue
            left = DROP_LIFE - (now - p.since)
            travel = self._travel_to_pile(s, p)
            need = travel + PICK_S + URGENT_MARGIN
            if travel <= URGENT_TRAVEL and need >= left and need - URGENT_MARGIN < left \
                    and (best is None or left < best[0]) and not self._pile_block(s, p):
                best = (left, p)
        return None if best is None else best[1]

    def _lull_pile(self, s: Scene) -> Pile | None:
        """Rule 8: nothing wanted here; a pile she can get to and pick up before the next respawn of her kills
        here is due. With none due there's no lull: the ground is empty and she leaves (rules 10 and 11 take
        the piles). Going after piles on an empty ground, she never left it."""
        due = min((t + RESPAWN for t in self.visit.kills if t + RESPAWN > time.monotonic()), default=None)
        if due is None:
            return None
        return self._best_pile(s, near_only=True, budget=due - time.monotonic())

    def _best_pile(self, s: Scene, near_only: bool, budget: float = 1e6, among: list[Pile] | None = None) -> Pile | None:
        """The pile worth the most per second of travel (Pile.value: what was seen in it, by the game data's
        prices), within `budget` s (of `among`, if given)."""
        best = None
        for p in self.h.drops.known if among is None else among:
            if not self._fresh(p):
                continue
            t = self._travel_to_pile(s, p) + PICK_S
            if t <= budget and (not near_only or t < 20) and (best is None or p.value / t > best[0]) \
                    and not self._pile_block(s, p):
                best = (p.value / t, p)
        return None if best is None else best[1]

    def _valuable_pile(self, s: Scene) -> Pile | None:
        """Rule 6a: of the piles seen to hold a drop worth `valuable` mesos or more (looting.Worth: Pile.dearest),
        VALUABLE_TRAVEL s away at most and gone for at once fewer than VALUABLE_TRIES times, the one worth the most
        per second of travel; None while she's going for one already."""
        if self.obj is not None and self.obj.why == "valuable pile":
            return None
        best = None
        for p in self.h.drops.known:
            if p.dearest is None or p.dearest < self.valuable or p.rushed >= VALUABLE_TRIES or not self._fresh(p):
                continue
            t = self._travel_to_pile(s, p) + PICK_S
            if t <= VALUABLE_TRAVEL + PICK_S and (best is None or p.worth / t > best[0]) and not self._pile_block(s, p):
                best = (p.worth / t, p)
        return None if best is None else best[1]

    def _loot_first(self, s: Scene, p: Pile, monsters, say: bool = True) -> bool:
        """EXP before loot: whether pile `p` goes ahead of shooting what's in reach of `monsters` from where she stands
        in look `s` (_shoot_rate). With nothing in reach, it does, unless what getting it would cost her in HP
        (`_pile_price`) is more than it's worth (at MESO_EXP an meso). Else only if what waiting would likely lose of it
        (its worth x `_at_risk` after the time to kill what's in reach, at MESO_EXP an meso), less that price, is more
        than the EXP she'd make from what's in reach over the trip there and back (and PICK_S on it). Said in her log
        when the answer for a pile changes (`say`)."""
        if not EXP_FIRST:
            return True
        price = self._pile_price(s, p)
        rate, busy = self._shoot_rate(s, monsters)
        if rate <= 0.0:
            return price <= p.value * MESO_EXP
        risk = self._at_risk(s, p, busy)
        to = self._travel_to_pile(s, p)
        trip = 2 * to + PICK_S
        first = risk * p.value * MESO_EXP - price > rate * trip
        if say and self._loot_said.get(id(p)) != first:
            self._loot_said[id(p)] = first
            what = (f"  {_contents(p)} on platform {p.platform} at x {p.x:.0f} (worth {p.value:.0f}, at risk "
                    f"{risk:.0%}" + (f", its HP {price:.0f} EXP" if price else "") + f") against {rate:.1f} EXP a "
                    f"second in reach for {busy:.0f} s, {trip:.1f} s there and back")
            self.log(what + (": the pile first" if first else ": shooting first"))
            if telemetry.recording():
                telemetry.emit("event", kind="loot_first", first=first, worth=round(p.value, 1), risk=round(risk, 2),
                               rate=round(rate, 2), busy=round(busy, 1), trip=round(trip, 1), platform=p.platform,
                               x=round(p.x), price=round(price, 1))
        return first

    def _landing(self, s: Scene, p: Pile) -> float:
        """Where on pile `p`'s platform she'd be when she gets there from look `s`: where she stands, on it; else where
        the quickest route there lands her (its last move's: route.Move.land_x; the trip's own may land elsewhere),
        the pile itself if there's none. Worked out once a look for each platform."""
        here = s.platform
        if here is not None and here.id == p.platform:
            return s.where.x
        if self._landings[0] is not s:
            self._landings = (s, {})
        got = self._landings[1]
        if p.platform not in got:
            path = None if here is None else self.nav.planner.plan(here.id, s.where.x, p.platform)
            got[p.platform] = path[-1].land_x if path else p.x
        return got[p.platform]

    def _touchers(self, s: Scene, pid: int, at: float, stay: float) -> list[tuple[str, float, str]]:
        """The monsters that count as a danger (attack.danger) about platform `pid` as of look `s`, for her there from
        `at` s on for `stay` s: as (kind, map x, why), those the look found on it ('seen'), those about it that it
        didn't ('unseen': `_unseen`), and with DUE_BACK its spawn points whose monster is due back then ('due back':
        DUE_FROM to DUE_TO s after a kill of hers of its kind on it, the point nearest the kill)."""
        d = getattr(self.h, "danger", None)  # (tests make a hunter without one)
        if d is None or not d.any:
            return []
        out = [(m.kind, m.x, "seen") for m in [*s.monsters, *getattr(s, "swept", ())]
               if m.platform == pid and d.share(m.kind) > 0]
        out += [(m.kind, m.x, "unseen") for m in self._unseen(s) if m.platform == pid and d.share(m.kind) > 0]
        now = time.monotonic()
        for k in reversed(self.h.ledger.kills if DUE_BACK else ()):
            if k.t < now - DUE_TO:
                break
            if not k.placed or k.platform != pid or d.share(k.kind) <= 0:
                continue
            if k.t + DUE_FROM <= now + at + stay and now + at <= k.t + DUE_TO:
                xs = [x for q, x, kind in d.points if q == pid and kind == k.kind]
                if xs:
                    out.append((k.kind, min(xs, key=lambda x: abs(x - k.x)), "due back"))
        return out

    def _pile_block(self, s: Scene, p: Pile) -> str | None:
        """Why pile `p` isn't to be gone for from look `s` now, or None: a monster that counts as a danger (`_touchers`:
        seen, about unseen, or with DUE_BACK due back at its spawn point by the time she'd be there) within PILE_CLEAR
        of the stretch of its platform from where she'd land (`_landing`), or stands on it, to the pile (see
        PILE_CLEAR). Said in her log when it changes for a pile. Never where nothing counts as a danger (the first
        map)."""
        d = getattr(self.h, "danger", None)  # (tests make a hunter without one)
        if d is None or not d.any or p.platform is None or s.where is None or s.platform is None:
            return None
        land = self._landing(s, p)
        on_it = s.platform.id == p.platform
        lo, hi = sorted((land, p.x))
        at = 0.0 if on_it else self._travel_to_pile(s, p)
        why = None
        for kind, x, how in self._touchers(s, p.platform, at, PICK_S + (hi - lo) / WALK_SPEED):
            gap = max(lo - x, x - hi, 0.0)
            if gap < PILE_CLEAR:
                why = f"a {kind} {how} {gap:.0f} units from {'her way to it' if hi > lo else 'it'}"
                break
        if self._block_said.get(id(p)) != (why is not None):  # (said when it starts waiting, not as the gap moves)
            self._block_said[id(p)] = why is not None
            if why is not None:
                self.log(f"  {_contents(p)} on platform {p.platform} at x {p.x:.0f}: later, {why}")
        return why

    def _pile_price(self, s: Scene, p: Pile) -> float:
        """What getting pile `p` would cost her in HP as of look `s`, in EXP (HP_EXP a bar): for each monster that counts
        about its platform (`_touchers`), its danger (Danger.of: its touch's share of her HP, by how soon it can be on
        her) where she'd land or at the pile, the worse; 0 where nothing counts as a danger."""
        d = getattr(self.h, "danger", None)  # (tests make a hunter without one)
        if d is None or not d.any or p.platform is None or s.where is None or s.platform is None:
            return 0.0
        q = self.model.platform(p.platform)
        land = self._landing(s, p)
        at = 0.0 if s.platform.id == p.platform else self._travel_to_pile(s, p)
        hp = sum(max(d.of(kind, q, x, q, land), d.of(kind, q, x, q, p.x))
                 for kind, x, _ in self._touchers(s, p.platform, at, PICK_S))
        return hp * HP_EXP

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

    def _at_risk(self, s: Scene, p: Pile, wait: float) -> float:
        """The chance pile `p` is gone if she first shoots for `wait` s, then goes for it: by its life (DROP_LIFE, a
        guess), 0 with URGENT_MARGIN or more of it left once she'd be there, 1 with none, in proportion between (0 if
        it's gone before she could be there going now); RIVAL_TAKES at least if another player is by it (players.
        near_rival) and it isn't hers alone by then (OWN_S; a pile no kill of hers explains never is)."""
        now = time.monotonic()
        left, to = DROP_LIFE - (now - p.since), self._travel_to_pile(s, p) + PICK_S
        if left < to:
            return 0.0
        risk = min(1.0, max(0.0, (wait + to + URGENT_MARGIN - left) / URGENT_MARGIN))
        players = (self._seen or s).players
        if near_rival(players, p.x, p.y) is not None and (not p.hers or now - p.since + wait + to >= OWN_S):
            risk = max(risk, RIVAL_TAKES)
        return risk

    def _call_off(self, o: Objective, why: str | None = None) -> None:
        """Her trip for a pile ends where she is (something in reach goes first now, or `why`: it isn't clear,
        `_pile_block`, which said so): the pile stays known, not counted as tried nor rushed, and the objective under
        it, if any, is hers again."""
        p = o.pile
        if o.why == "valuable pile":
            p.rushed = max(0, p.rushed - 1)
        if why is None:
            self.log(f"  {_contents(p)} on platform {p.platform} at x {p.x:.0f}: later, shooting what's in reach first")
        self.obj = self.stack.pop() if self.stack else None
        return None

    def _yields(self, s: Scene) -> bool:
        """Looter.yield_to while she hunts: whether picking up stops for what's in reach in look `s` (EXP before loot:
        any wanted monster in reach, the pile under her waiting)."""
        return self._shoot_rate(s, s.monsters)[0] > 0.0

    def _detour(self, s: Scene, slack: float = 0.0) -> tuple[float, Pile] | None:
        """Rule 5b: travelling, the pile a stop at costs DETOUR_S at most on top of the trip (and `slack` more):
        her way to it (and its drops' landing, for one not seen yet), PICK_S on it and her way on from it to the
        objective's place, less her way there from here (Planner.arrive), the most worth per second of that first;
        with what it costs. Her ways to the objective's place by the routes its trip takes, keeping off what it
        keeps off (_keep_off: a spot's ground): weighed by the planner's plain routes, from 36 at x 449 a pile at
        779 on her way to 17's spot on 16 was 1.3 s more by the right side, past 17, where the trip goes by the
        left, and 3.8 s more by that. Her way to the pile is the stop's own trip's, which keeps off nothing. Not
        for a pile more than DETOUR_REACH s away, nor one with a monster by it, nor on a platform she can't stand
        on with monsters on it, as rule 5 has them; not while the objective is a pile on the way itself, a
        valuable or an urgent one, nor after DETOURS."""
        o = self.obj
        if o is None or o.why in ("on the way", "valuable pile", "urgent pile") or o.detours >= DETOURS:
            return None
        plan, here, x, keep = self.nav.planner, s.platform.id, s.where.x, _keep_off(o)
        direct = plan.arrive(here, x, o.platform, o.x, keep_off=keep)
        if direct is None:
            return None
        now, best = time.monotonic(), None
        for p in self.h.drops.known:
            if p is o.pile or not self._fresh(p) or self._guarded(s, p) or self._pile_block(s, p):
                continue
            to, on = plan.arrive(here, x, p.platform, p.x), plan.arrive(p.platform, p.x, o.platform, o.x, keep_off=keep)
            if to is None or on is None or to > DETOUR_REACH:
                continue
            lands = max(0.0, p.since + DROP_LANDS - now - to) if p.status == "expected" else 0.0
            extra = to + lands + PICK_S + on - direct
            if extra <= DETOUR_S + slack and (best is None or p.value / max(extra, PICK_S) > best[0]):
                best = (p.value / max(extra, PICK_S), extra, p)
        return None if best is None else best[1:]

    def _guarded(self, s: Scene, p: Pile) -> bool:
        """Whether a monster is by pile `p` (PILE_GUARD), or any on its platform where she can't stand: as her memory
        has them as of look `s` (of any kind: a guard needn't be worth hunting)."""
        there = self.h.memory.there(s.t, p.platform)
        return bool(there) and (not self._standable(p.platform) or any(abs(m.x - p.x) < PILE_GUARD for m in there))

    def _go_for(self, pile: Pile, rule: str, s: Scene) -> None:
        parent = self.obj
        if parent is not None:
            self.stack.append(parent)  # suspended
        if rule == "valuable pile":
            pile.rushed += 1
        self.obj = Objective("pile", pile.platform, pile.x, pile=pile, why=rule)
        if rule == "on the way" and parent is not None:
            # A stop on the way shoots in passing out of the trip's own budget, and leaves finishing to it (once
            # it's done she owes them still): with budgets of their own, its stops shot on and on, and in the
            # simulator's hunt loop the objectives ran out their OBJECTIVE_S 99 times in 60 runs, 87 so (61 without
            # the rule), for 128.3 EXP a minute against 128.7 and 90.1% of her drops against 90.0%.
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
        self._exit_piles = self._exit_reason = None
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
                                vary=o.kind != "pile")
        o.trip.weights, o.trip.choice = o.weights, o.choice
        chains = o.trip.trip.chains
        while not o.trip.over:
            self.nav.step(o.trip)
            self._owe_move = time.monotonic() < o.trip.post_hit_until  # (just touched: on first, navigator.POST_HIT)
            if o.trip.trip.chains > chains:
                chains = o.trip.trip.chains
        result = o.trip.trip.result
        if result == "arrived" or (result == "in position" and o.kind == "spot" and self._on_spot(o)):
            return self._arrived(o)
        if result == "in position":
            return None  # a rule fired: the next pass decides (OBJECTIVE_S is checked there)
        self.log(f"  {o.text()}: {result}")
        return self._drop_objective(o, result)

    def _on_spot(self, o: Objective) -> bool:
        """Stopped 'in position' where she can stand with targets of the ground in reach: as good as there.
        Judged from the look the trip stopped on (`_interrupt` said so to it a moment ago)."""
        s = o.trip.last if o.trip is not None and o.trip.last is not None else self.h.look()
        return (s.platform is not None and self._standable(s.platform.id)
                and any(m.platform == o.ground and self._hits(s, m) for m in s.monsters))

    def _interrupt(self, s: Scene) -> bool:
        """Asked before every chain of a trip: whether a rule calls for her here (close quarters, a valuable pile,
        one she has hurt to finish, a target in reach she may shoot, a pile on the way, an urgent pile). A
        valuable or an urgent pile calls for her only where it goes ahead of what's in reach (`_loot_first`, as `_pass`
        asks it): one that waits stopped the trip "in position" on its first look, no move made, and the next pass
        sent her off on it again, again and again, neither shooting nor moving once her time in passing there was spent
        (review, 2026-10-25: 1-4 such trips a run in loop_tree_ore_waits, one followed by 2.9 s without a throw in the
        harness's crowd). In reach, so too: only what the pass would shoot (fight._wanted, as its targets are: on the
        ground, on her platform, or in a star's reach from where she stands). A monster off both that only a jump
        reaches isn't one of its targets, and stopped every trip on its first look: in loop_tree_walls a green mushroom
        on the spot's platform, above her, held her 107 s, the pass deciding 2,373 times and never moving (found
        offline, 2026-10-25)."""
        here = s.platform.id
        if self._close_quarters(s, s.monsters):
            return True
        o = self.obj
        if o is not None and o.kind == "pile" and o.pile is not None and self._worthless(o.pile):
            return True  # not worth it any more (`_pass` drops it)
        if o is not None and o.kind == "pile" and o.pile is not None and self._pile_block(s, o.pile):
            return True  # not clear any more (`_pass` calls it off)
        wanted = [m for m in s.monsters if self.kinds is None or m.kind in self.kinds]
        if o is not None and o.why == "valuable pile":
            # At once: nothing else stops her, but something in reach that goes first now (`_loot_first`)
            return not self._loot_first(s, o.pile, wanted, say=False)
        if (pile := self._valuable_pile(s)) is not None and self._loot_first(s, pile, wanted, say=False):
            return True
        if self._owed(s, s.monsters) is not None:
            return True
        ground = None if self.visit is None else self.visit.ground
        in_reach = [m for m in wanted if self._hits(s, m)
                    and (ground is None or m.platform in (ground, here) or self._star_hits(s, m))]
        if (in_reach or self._pass_step(s, wanted) is not None) and o is not None \
                and o.passing.get(here, 0.0) < PASSING_S:
            return True
        if o is not None and o.kind == "spot" and self._standable(here) \
                and any(m.platform == o.ground for m in in_reach):
            return True
        if self._detour(s) is not None:
            return True
        return o is not None and o.kind != "pile" and self._exit_piles is None \
            and (pile := self._urgent_pile(s)) is not None and self._loot_first(s, pile, wanted, say=False)

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
            p = o.pile
            # Guarded (a monster by it): not a try. The guard is a target now (it's on her platform), so the
            # next looks shoot it, and the pile is gone for again while it's still worth it.
            if self.h.looter.collect(DropSpot(p.x, p.y, p.platform, p.since, p.mesos, p.status == "seen")):
                self._tried(p)
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
        now = d.at(here.id, x, s.monsters)
        need = self._spawns_in_reach(s, x)
        best = None  # (danger, distance, x)
        for g in spots._stand_xs(here.left + GROUND_MARGIN, here.right - GROUND_MARGIN):
            if self._spawns_in_reach(s, g) < need or self._in_the_way(s, x, g):
                continue
            there = d.at(here.id, g, s.monsters)
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
        self.report.rules[rule] += 1
        if rule != self._rule:
            self._rule = rule
            self._publish()

    def _publish(self) -> None:
        """The 'hunt' record for the mini-map (gamelab.ui): why she does what she does. Sent when the rule,
        the objective or the piles change."""
        s = self._s
        if not telemetry.recording() or s is None or s.platform is None:
            return
        urgent = self._urgent_pile(s)
        piles = [{"x": round(p.x, 1), "y": round(p.y, 1), "platform": p.platform, "status": p.status,
                  "hers": p.hers, "urgent": p is urgent, "worth": round(p.value)}
                 for p in self.h.drops.known if self._fresh(p)]
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
    """Whether monsters `a` and `b` are one: of a kind, on a platform, within memory.MATCH_SLACK."""
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
