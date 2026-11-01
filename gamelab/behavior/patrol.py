"""Grinding's knowledge of the map and its choices (behavior.grind, `tour --loop --grind`): where her kills come from,
where she stands to shoot them, how full each spawn point is likely to be, and the loop over the map that gives the most
EXP a minute. Pure: the map's game data, the planner's route times and her memory of the monsters; nothing here looks or
presses a key. General: nothing is chosen by a platform's or a kind's name.

The map, once (Patrol):
- points: every spawn point of a kind she hunts (Point: its platform, x, kind and the kind's EXP, game data);
- stops (Stop): on each platform from which she can get at spawn points, the x that reaches the most of their EXP: her
  own platform's she walks up to (all of each), another's as much of it as a star reaches from there (attack.spots.
  coverage, about +-60 units up and down: reach.SHOT_UP, SHOT_DOWN), less what touches there would cost her while she
  shoots it (the danger hook, below: STOP_DANGER), so of places reaching as much the safest, not the middle. On Tree
  Dungeon, with the positioning model's danger, 3 of its 26 stops move that way (11, 18 and 22, by 20-77 units, to
  places 0.003-0.014 safer); 23's stays at x 8, between its two Horny Mushroom points (0.29; 0.15 at -147), for the
  Green Mushroom point on 24 a star reaches from there. A stop is where a leg of the loop ends; what comes into reach on
  the way she shoots in passing (behavior.hunt, rule 4); waiting there, she waits at the safest place along it that
  keeps its spawn points in reach (the hunt loop's rule 9: hunt._wait_place);
- the route times between stops (motion.route.Planner.arrive: on Tree Dungeon up the tower is slow, 24 to 6 21 s by its
  ropes, down quick, 6 to 24 8 s).

What she believes (Belief, from her memory of the monsters, localization.memory, at every look): how many monsters each
spawn point has now and at any time ahead (`stock`): the monsters remembered there (each the nearest point of its kind
on its platform), else, for a point a look has seen empty (one that searched a body of its kind standing at the point:
Memory.holds_body) or her kill emptied, what's back since the latest look that saw it empty (`back`: the game fills the
map to its cap at each pass, what died since the pass before back at random spawn points with no monster within 100
units, as measured on Tree Dungeon on 2026-10-29 and 30 and as the server's code has it; at the refill's rate a second,
the same for every free point: `rate`); else, never seen, Q_UNKNOWN. A point is emptied once (her kill, or the first
look to see it empty after a monster was seen there) and full again once a monster is seen there after that. The
refill's rate (Belief.rate) is the map's state, not a constant: what her own kills bring back over the map (`own`, over
its `weight`), and what the other players' do (`others`), learned from what her looks saw of the points they saw free,
each seen again within WATCH_GAP_S free still or with a new monster at it, her kills' passes counted in, weighed by age,
about the prior's REFILL_RATE (until 2026-10-29 a refill clock per point, nothing for 5 s and then 12.8 s times the
points on its platform, scaled by the refills she watched alone: a point that never came back never counted; until
2026-10-30 one rate learned for every free point, whatever died). And what of it is hers (`share`): by a point with
another player by it (their minimap dot, RIVAL_X across and RIVAL_DY up or down), as much as others leave of its kind,
learned from her memory (`take`): of the monsters she saw by another player, those looks forgot (not those forgotten by
age: she never looked again) without her kill, against those she killed or saw there TAKE_KEEP_S after she first saw
them.

The loop (Patrol.best): a closed round of stops (LOOP_MIN to LOOP_MAX), chosen by its EXP a minute in the long run: each
spawn point it reaches gives its EXP (times her share) each time she comes round and finds it back (`caught`: what's
back a cycle after she found it free, at the refill's rate); the cycle is the round's route times, STOP_S a stop and her
shooting (her kill time a point of EXP, KILL_S_PER_EXP, learned from her throws), less what the danger hook says touches
would cost there (hp_price EXP a bar of HP). Grown from each of the SEEDS best stops by adding the stop that raises the
rate most (at its cheapest place in the round), the best round kept. Standing at one stop (camping) is a round of one
(Patrol.camping), which grinding never takes up but logs, to compare. A new round replaces the one she follows only when
it's LOOP_GAIN better by the same belief, and not within LOOP_MIN_S of the last change (no flip-flopping), unless the
last EMPTY_RUN stops she got to gave nothing: then at once.

On Tree Dungeon, Forest Up North II, before she has seen anything (Lucky Seven's kill time, the refill at the prior's
rate): the loop 10, 7, 486 EXP a minute (a round of 24 s: 5 of travel, 17 of shooting; 8 spawn points), where the user
grinding made 484 on 2026-10-24 (with the refill clock until 2026-10-29: 463; 14, 10, 8, 12 461; camping at 23 228,
where her hunt loop camping there made 283 live). Rounds this close are a coin toss: what she sees decides (the next
stop, below), and the refill she learns. With what a round's own kills bring back (from 2026-10-30) the same loop, 515
(a round of 30 s: 23 of shooting), where without it 477 (the magician's stats of 2026-10-30 both).

The next stop (Patrol.choose): along the loop from the last loop stop she was at, the first expected to hold SKIP_FULL
of what it holds full when she gets there (an emptied stretch is skipped until it's due; with none due on the loop, of
the stops due elsewhere the one expected to give the most a second, if more than the loop's next), or on a trip the stop
she's on her way to; and a stop whose monsters she has seen (her memory, not the belief's guesses: a pile of one monster
or more, PULL_MIN) pulls her off it when it pays for its detour against the loop: what it's expected to give (Patrol.
expect: the monsters seen, and the belief's stock at its other points, her share), less what touches there cost her
while she shoots it (Patrol.net: hp_price a bar of HP, by hp_rate, as the loop's rate counts its stops'), PULL_GAIN
times what the loop gives in the time the detour costs her, at the loop's long-run EXP a second (Plan.rate, by the same
belief: the refill as learned, others' share; as of the latest replan, every REPLAN_S). The detour
(Patrol.detour) is its kills (her kill time), STOP_S, and the travel there and on from there to the loop's next stop,
beyond going to that one straight (none for a stop on her way). Of the stops that pay, the one that leaves her the most
EXP over the loop's; on her way to a pile, another takes her there only when it leaves more over the loop's with the
margin (PULL_GAIN times the loop's over its detour) than the pile she's going to does without (detoured to the loop's
next stop alike: two piles each worth the other's detour had her turn from one to the other at every look, 30 ms apart,
in loop_tree_grind_jump_key), and when the pile she's going to no longer pays even without the margin (shot in passing,
gone, taken) she goes back to the loop's next stop ("back to the loop"; it takes the margin to pull her again: until
2026-10-29 she went on to it and left it with nothing, loop_tree_grind_rope_top's 15, 7 of 26 stays in her 01:11 run).
After a pile the loop goes on from its next stop, the one the detour was costed to (Patrol.pulled: also when the pile is
a stop of the loop further on; until 2026-10-29 she went round from the stop a pull took her to, and after a pull on a
trip from the stop after the one she was heading to, which the round then missed). A loop worth nothing by the belief
(its touches costing all it gives) has no bar for a pile to beat: no pulls. The marginal value theorem's bar, as at a
stop (behavior.grind): her time is worth the loop's rate wherever she is, so a pile near her pays for the time it takes
and the travel it adds, not for being nearer than the loop: what's standing in a region of poor spawns doesn't hold her
there, the region's spawns being why the loop isn't there. Until 2026-10-29 a pile pulled at 1.5 times the next stop's
own EXP a second, its travel counted: far from the loop every pile around her beat the climb to it. Live on Tree Dungeon
(the magician, 2026-10-29 01:11, mage-1029-0111.log), a Lie Detector's retreat having left her at the bottom, 28 pulls
in 7 minutes, 21 to its stops 28-33 and 7 to 20-24 (13-101 EXP expected 1-11 s off, 2.3-6.9 EXP/s), 25 of them from the
loop's next stop up the tower (0.7-2.0 EXP/s, 11-31 s away), her loop 14, 13, 10, 8, 6 at 447 EXP a minute: she never
got up the tower, and her kills made 185 EXP a minute (the user: "keeps patrolling the bottom region which has worse
spawn rate"). By this rule none of the 28 pulls (tests/test_grind.py, LIVE_PULLS: the loop's rate as the code had it at
each, 383-479 EXP a minute, and the piles' EXP net of their touches; the most, 69 EXP 2.8 s off on her way up from 31 to
29, and 97 on 23 on her way up to 7, 1.18 times what the loop gives over their 8.3 and 12.6 s): she climbs, shooting in
passing what's in reach on the way (with PULL_GAIN 5 on the old rule, live at 01:26, she went up at once and made 349
EXP a minute over 8.5 minutes). With less margin only piles on her way would (5 at 1.1, 10 with none: going there and on
up adds 0.5 s at most to the climb). A stop of the loop further on counts the same (going there first and on back to the
loop's next): the loop's order holds unless a pile pays for leaving it. What a stop is expected to give (Patrol.expect)
counts a monster her memory has only where she gets at it from there (on the stop's platform, or in a star's reach from
somewhere along it: `gets`), and not one she left the stop with nothing in reach since (`left_at`: she'd been there to
see it); a point her memory has none at, by the share of it a star reaches from the stop.

The danger hook (`Patrol.danger(platform, x)`): how badly she can be touched fighting at a place, in the positioning
model's unit (attack.danger.Danger.at: the share of her HP touches would take there, each monster or spawn point weighed
by how soon it gets to her), which grinding plugs in (behavior.grind: the hunter's, Hunter.danger); she loses
DANGER_HP_S of her HP a second for each unit of it. By default (default_danger), 0 where no monster can touch her
(MapModel.safe), else the
touch shares (touch_shares: the positioning model's, from the game data's touch damage at her level, over her HP) of the
platform's spawn points of kinds whose touch takes DANGER_MIN of her HP or more: by no kind's name. What she loses there is learned as she goes (`hp_rate`: the hook
as a prior, HP_PRIOR_S, and the HP she's seen go there).

The numbers marked "guess" are starting values for the simulator's A/B runs; the rest say where they were measured, or
are a search's resolution.
"""

import bisect
import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..attack import danger as danger_model
from ..attack import reach, spots
from ..attack.reach import EDGE_MARGIN, SHOT_DOWN, SHOT_UP
from ..localization.mapdata import SPAWN_REACH, MapModel
from ..localization.memory import MATCH_SPEED, MATCH_WALK_S, MEMORY_S, SURE
from ..localization.monsters import DEFAULT_SPEED, Monster, speeds
from ..localization.players import Player, near_rival
from ..localization.tracking import SPAWN_NEAR
from ..motion.physics import WALK_SPEED
from ..perception.kinds import kind_of

# s of her shooting a point of a kill's EXP: Lucky Seven live on Tree Dungeon (tour --loop --skill, 2026-10-24): 3.85
# throws a Horny Mushroom's kill (41 EXP) and 2.95 a Green Mushroom's (26), a throw every THROW_S: 3.2 and 2.5 s, 0.079
# and 0.095 s a point. Learned as she hunts from her throws and the EXP they gave (Patrol.learn_kills), this counting
# as KILL_PRIOR_EXP points of EXP: with the regular attack's stars (12.4 throws a Horny Mushroom with the simulator's
# misses) it's about three times as long.
KILL_S_PER_EXP = 0.085
THROW_S = 0.84         # s between her throws (her attack's lock and its margin: 0.79-0.85 s live)
KILL_PRIOR_EXP = 300.0  # guess
STOP_S = 1.0           # guess: s a stop costs beyond its kills (the looks that judge it, settling where she lands)
# The refill of a spawn point she has seen empty (`back`, `not_back`), as the game respawns on Tree Dungeon (measured on
# 2026-10-29 from the perception service's telemetry of 2026-10-28 and 29: 10 files, about 16 hours, 9,957 new monsters
# at spawn points whose platform was in view before them; D:/MapleHelper-scratch/respawn; and on 2026-10-30 against the
# server's own rule, OrionAlpha's LifePool.tryCreateMob, on 19 files, 40 hours: D:/MapleHelper-scratch/respawn-orion,
# RESULTS.txt): the map fills itself to its cap. A pass every TICK_S (fitted on each stretch: 7.09-7.13 s; 73-95% of the
# births within 0.6 s of a tick, the rest the detector's lag) spawns the cap less the monsters alive at points drawn at
# random among those clear (no monster within 100 units of it, nor a point taken before it that pass: OrionAlpha's box,
# sim.world.SPAWN_CLEAR; the user, 2026-10-31), the cap the server's formula (MapModel.mob_cap: the map's area times its
# mobRate; sim.world's top: the measurements; from 2026-10-30 at the user's word, where it was 0.42 of the spawn points
# by a count of the parts of the map she saw). So what comes back is what died since the pass before, hers and the other
# players' kills, spread over the map: at a free point, a pass brings one with a chance of what died over the map's
# weight (Belief.weight: its free points at the cap, a held one never drawn: Tree Dungeon's 31 points, 22 held, 9; until
# 2026-10-31 a held one half as likely, a second monster beside the first, from evidence that didn't hold up), and with
# her alone nothing came back but her own kills (16:35-16:45 on 2026-10-29: nobody else there and nothing of hers dying,
# the top's 12 points and 3 more in view stayed free 4 to 9 minutes, no birth in 38 ticks). Her own kills she knows
# (Belief.own: her kills a second over OWN_S, each its pass's); the other players' she learns (Belief.others: the refill
# a second at a free point net of what her kills brought back). With others there a free point refilled at REFILL_P a
# pass on average (21,285 free point-ticks in view, 5,775 births), by 5-minute stretch from 0.40 to none, with the other
# players (nobody else for a minute: 0.023, 11 births in 486 free point-ticks, each within two ticks of a kill of hers;
# one: 0.10-0.17; more: 0.26-0.31), where until 2026-10-29 each point had its own clock, nothing for 5 s and then n x
# 12.8 s on a platform of n points, and until 2026-10-30 one rate for every free point whatever died. The others' part
# is learned (Belief.others): what her looks saw of the points they saw free, each looked at again within WATCH_GAP_S,
# free still or with a new monster where one born at it can be by then (Belief. born_reach), each the likelihood of
# RATE_GRID's rates by the model, her kills' passes in the gap counted in, weighed by exp(-age / REFILL_MEMORY_S), the
# prior a gamma about REFILL_RATE counting as REFILL_PRIOR_N refills. A round's worth (Patrol.rate) counts its own
# kills: what they bring back over the map, its points' share of it. Until 2026-10-29 the refills she watched were
# averaged against the model's mean and a point that never came back never counted: at 16:39-16:44 the top's points,
# empty for minutes, were each expected back at the model's pace slowed 1.9 times, 45 EXP at stop 7 9 s after she found
# it empty, and she went between 7 and 14 for 4 minutes for nothing (the user: "the character keeps going up where no
# mobs are spawning"); learned so, by 16:42 a free point there refills at 2.5% a tick. What's back may be net of others'
# kills and the cap: a point she left free filling to Q_CEIL in the long run, the refill's rate at first (a chain of two
# states: in at the rate, out at (1 / Q_CEIL - 1) times it). Measured, out of her view a minute to 10 minutes, the free
# points she left had a monster on 0.43-0.53 of them when she was back (in view, 0.48-0.53 net of her kills), where the
# refill alone fills them all; but at 0.5 the loops' long-run rates halve for the long rounds, so the bar a pile must
# beat (the pull) halves too, and replayed on her 01:11 run the pulls to the bottom come back (10 of 32 choices, 1 at
# 1.0, 21 live before the pull's rule). Off (1), as the clock had it: other players' share is by their dots (`share`).
TICK_S = 7.115
OWN_S = 60.0           # guess: her kill rate as of now over this long (Belief.own)
REFILL_P = 0.27
REFILL_RATE = -math.log(1.0 - REFILL_P) / TICK_S  # 0.0442 a second: the others' prior
REFILL_PRIOR_N = 3.0   # guess: the prior counting as this many refills seen ...
REFILL_MEMORY_S = 180.0  # guess: ... what she saw weighed by exp(-age / this): the crowd changed within minutes
WATCH_GAP_S = 20.0     # guess: a point seen free and looked at again within this long: evidence (her looks come in
                       # bursts at the stops, a trip between: the top's 13 s apart when she went between 7 and 14)
RATE_GRID = tuple(0.0005 * (0.3 / 0.0005) ** (k / 47) for k in range(48))  # the rates weighed (a resolution)
Q_CEIL = 1.0
Q_UNKNOWN = 0.35       # a point never seen: this full (live: the uncamped 19-21 had a monster 31-39% of the time, 28%
                       # of all points; the simulator's crowd fills every point)
# Another player by a point (their minimap dot within RIVAL_X across and RIVAL_DY up or down: hunt.PLAYER_NEAR and
# PLAYER_DY) in a look within RIVAL_KEEP_S takes its monsters: of its kind, she gets 1 - take(kind), learned from her
# memory (the monsters she saw by another player: forgotten by looks without a kill of hers, against killed by her or
# seen there TAKE_KEEP_S after she first saw them), TAKE_PRIOR counting as TAKE_N monsters. The crowding study of
# 2026-10-24: other players killed Green Mushrooms within about 4 s and left Horny Mushrooms alone; the simulator's
# camps do the same. Replayed on the user's play (lab/telemetry/20261024-170558-watch.jsonl) it learns 0.50 and 0.33,
# on the 14:01 hunt 0.51 and 0.50.
RIVAL_X = 200.0
RIVAL_DY = 100.0
RIVAL_KEEP_S = 20.0    # guess
TAKE_PRIOR = 0.5       # guess
TAKE_N = 4.0           # guess
TAKE_KEEP_S = 10.0     # guess (the crowding study's Green Mushrooms went within about 4 s)
# The danger hook (see the top): the share of her HP a second she loses fighting at a place, for each unit of its danger.
# Live, hunting 23 (two Horny Mushroom points) on 2026-10-24 she lost 0.28-0.31 bars of HP a minute, 0.005 a second,
# where the positioning model's Danger.at at 23's stop is 0.285: 0.0175. The default's danger there is 0.32 (two Horny
# Mushroom points, 0.161 each): 0.0056 a second.
DANGER_HP_S = 0.0175
# The default (default_danger): a kind counts when a touch by it takes this share of her HP or more, the positioning
# model's (attack.danger.DANGER_MIN: on Tree Dungeon a Horny Mushroom takes 16.1% of her 752 HP at level 24, a Green
# Mushroom 7.2%, a Slime 1.2%; on the first map no kind counts).
DANGER_MIN = danger_model.DANGER_MIN
# What a bar of her HP costs her, in EXP (her pet's potions): the mesos its potions cost at the user's reference price
# (behavior.hunt.bar_exp: HP_PER_MESO 2 HP a meso, at MESO_EXP 0.5), worked out at every look from her max HP as
# perception reads it (behavior.grind's `_look`: Patrol.hp_price), this until it's read (and for a patrol with no
# hunter): the simulator's her (sim.world.LEVEL_25), 774 HP, 387 mesos, 193.5 EXP. From 2026-10-31 ~16:00; before, 250
# (3.1 Orange Potions at a guessed 160 mesos each), and until earlier that day 100, a guess of its own, at which the
# loop and her stays priced her touches at the top of Tree Dungeon's tower at 0.07-0.15 bars a round where she lost 3
# bars a minute on 10 (D:/MapleHelper-scratch/replan/pos).
HP_PRICE = 193.5
# A stop's place along its platform (stops_of) weighs the EXP it reaches against its danger there as the hunt loop
# weighs a ground's (behavior.hunt.GROUND_DANGER: 1 - 2 x the danger, never under 0), from 2026-10-31: on Tree Dungeon
# 7's stop moves from x -95 (1 unit off its Horny Mushroom's point at -94, danger 0.27) to -31 (0.22), 13's from -35 to
# 22, 15's from -95 to -55, 19's from 150 (on its point) to 130, 23's from 8 to -172 (0.28 to 0.13); before she has seen
# anything the loop is 9, 8 (492 EXP a minute with the server's respawn, 2026-10-31; the thief's stats of the harness),
# as with the old weight (before the server's respawn: 10, 8 at 449, for 10, 7). Until then KILL_S_PER_EXP x DANGER_HP_S
# x HP_PRICE (0.149 of its EXP a unit of danger: what touches would cost while she shoots there, by the live hook's
# guesses), which left the stops on spawn points, a respawn there landing on her. With the default hook, the same all
# along a platform, it changes no stop.
STOP_DANGER = 2.0
# ... and blended with the HP she has seen go on each platform (Patrol.learn_hp: her touches, behavior.grind's
# HP_BY_TOUCHES, or the drops of her HP bar between looks), the hook counting as HP_PRIOR_S s there: in the simulator's
# crowd on Tree Dungeon she lost 1.3-1.4 bars a minute alive whatever she did (20 bars a run, her 60 potions), live
# 0.28-0.33; the hook's 0.0175 a unit of danger is the live hunts' on 23, where in the simulator's grind at the top of
# the tower she lost 2.4-2.8 bars a minute at 0.3 and more. A guess, 20 s from 2026-10-31 (60 until then): what she sees
# there counts for as much as the hook after a stop's stay.
HP_PRIOR_S = 20.0
# ... and the hook of a place she hasn't been at scaled by what she has lost against it where she has (Patrol.
# hook_scale), the hook's own word counting as HOOK_PRIOR_HP of HP lost as it said (a guess: about 40 s of shooting at
# a stop of danger 0.3), from the review of 2026-10-31: in the simulator's crowd she learned 2.4-2.8 bars a minute at
# the top of Tree Dungeon's tower against the hook's 0.3, so a stop she had been at soon looked 10 times dearer than
# one she hadn't, and the loop's replans went for the stops she hadn't been at; live the hook is the hunts' own (the
# scale near 1).
HOOK_PRIOR_HP = 0.2
# The loop.
LOOP_MAX = 12          # guess: stops in a round at most ...
LOOP_MIN = 2           # ... and at least: one is camping (Patrol.camping), which grinding doesn't do
SEEDS = 8              # guess: rounds grown, from the stops worth the most on their own (best() about 0.1 s on Tree
                       # Dungeon)
STAND_STEP = 20.0      # units between the places along a platform tried as a stop (a resolution)
# A stop stands clear of its own platform's spawn points (from 2026-10-27, landings with clearance): of the places that
# reach within STOP_SLACK of the most EXP, the one farthest from the nearest of them, up to where her attack from the
# ground surely takes a respawn there (reach.ground_sure: past the claw's edge, 105 for a mushroom), then the most EXP,
# then the nearest its middle. Live the grind stops sat on spawn points (7's 1 unit off its point at -95, 19's 1 off 150, 14's 6, 10's 11,
# 13's, 16's and 32's 13, 8's and 15's 15: D:/MapleHelper-scratch/land/diag-landings), where a respawn comes onto her and
# no attack from there is sure, and in 92% of her close landings on a trip's own platform the stop's x itself was within
# 105 of a monster. Only where the danger hook counts them (not on the first map). With 5% (24's stop 221 -> 201, 71 from its
# Green's point; 13 of Tree Dungeon's 26 stops moved, none nearer its points), grinding made 4.3% less EXP a minute
# (se 2.4%) in the harness's `walkers` world (24 seeds of 5 minutes, the rest of the branch the same), its time within 60
# of a monster 8.6% more: its loops changed ([14, 10, 8, 12] for [10, 7] at loop_tree_grind's start) and the stops
# farther from where the respawns come stood farther from what they shoot. Off (0), as before: the most EXP alone.
STOP_SLACK = 0.0
LOOP_GAIN = 0.15       # guess: a new round replaces hers only this much better by the same belief ...
LOOP_MIN_S = 60.0      # guess: ... and not within this long of the last change, unless ...
EMPTY_RUN = 3          # guess: ... the last this many stops she got to gave nothing (left with nothing in reach, no
                       # EXP there): then the round is looked at again at once and a better one taken whenever. Live at
                       # 16:39-16:44 on 2026-10-29 her loop's stops came up empty 37 times running; the refill learned
                       # from what she sees lags (REFILL_MEMORY_S), the round's own emptiness doesn't
REPLAN_S = 20.0        # guess: s between two looks for a better round
# The next stop.
# A stop of the loop is skipped when it's expected to hold under this share of what it holds full when she gets there
# (an emptied stretch, until it's due): under Q_UNKNOWN, so a stop never seen is never skipped.
SKIP_FULL = 0.25       # guess
# A stop pulls her off the loop (or a trip) when the monsters she has seen there are worth PULL_MIN of a monster of the
# cheapest kind (memory.SURE: one at even odds), and what it's expected to give, net of its touches, is PULL_GAIN times
# the loop's long-run EXP a second over its detour (see the top). The margin multiplies the whole detour, her kill time
# included: a pile's EXP comes at her kill rate (1 / kill_s), the loop's with its travel, so a pile on her way pays
# from PULL_GAIN rate STOP_S / (1 - PULL_GAIN rate kill_s) EXP (twice that a second off it), and none pays at all once
# her kill time times the loop's rate reaches 1 / PULL_GAIN (0.8: a round 80% shooting, net of its touches). The
# thief's (Lucky Seven's 0.085 s a point, Tree Dungeon's loops 7.4-7.7 EXP/s): on her way from 43-53 EXP (a Horny and a
# Green Mushroom), a second off it 87-106. The magician's live on 2026-10-29 (Magic Claw, her casts MP-limited:
# 0.080-0.113 s a point learned, the loop 383-479 EXP a minute, products 0.6-0.86): 32 EXP at 0.094 and 6.4 EXP/s, 70
# at 0.1 and 7, 150 at 0.1 and 7.5, none from 0.107 at 7.5; her kill time moves the bar only so far, the loop's rate
# counting it alike. A margin on the detour's overhead alone (its travel and STOP_S, what's lost if the pile's gone:
# the limit then at her kill rate equal to the loop's) would pull her to 10 of her 28 live pulls (LIVE_PULLS: those on
# her way up), as no margin does by this one: not taken without the harness and a live check. Until 2026-10-29
# PULL_GAIN was 1.5 times the next stop's own EXP a second, its travel counted: with two kills' worth at least asked (a
# guess), 18 of 45 moments of the 14:01 hunt with two monsters or more on another platform went elsewhere, 22 of 34 on
# the user's play: on 22 at 197 s, two seen on 19 2.6 s away (7.1 EXP/s) lost to 10, 15 s up the tower and never seen
# (2.2 EXP/s); one monster has been enough since. By this rule (the user's play replayed every 0.5 s, `python -m
# gamelab patrol`: the moment like it now at 205 s, on 22, two seen on 19 2.3 s off, 46 EXP expected, 10 14.2 s up) 19
# is on her way up to 10 (12.1 s on from it), so its detour is its kills and a stop, 5.0 s, in which the loop (7.4
# EXP/s) gives 37 EXP: they paid, just (by PULL_GAIN), until their touches counted against them (the review of
# 2026-10-29: 3% of their EXP by the danger hook there); now, just not, and she goes on up to 10 from 22 and 21 at
# 204-206 s, as at 197 s; from 23 at 201-204 s and 21 at 207 s, with 47-53 EXP seen there, they pay. Replayed so every
# 5 s, 30 of the 50 choices pull (31 before the touches counted, 46 of the 50 the same; 30 by the old rule, 6 of them to
# the same stop), each to a pile on her way or at most 1.6 s off it; and 5 of the 49 are where the user went next (the
# next platform they stood on within 15 s), 13 by the old rule: the user roamed the piles round them on 19-24
# (D:/MapleHelper-scratch/pull/fix/userplay_agree.txt).
PULL_MIN = SURE
PULL_GAIN = 1.25       # guess: a quarter over the loop's (the pile seen may be gone or taken, the detour longer)
REACH_ROUND = 5.0      # units: a remembered monster's x to this for whether a stop gets at it (Patrol.gets, kept; a
                       # resolution)
DUE_S = 2.0            # guess: s of a wait at a stop for the respawns in reach (Patrol.wait_gain)


@dataclass(frozen=True)
class Point:
    """A spawn point of a kind she hunts."""
    id: int
    platform: int
    x: float
    kind: str
    exp: float


@dataclass(frozen=True)
class Stop:
    """Where a leg of the loop ends: on `platform` at map x `x`, from where she gets at the points `cover` (point id,
    the share of it she gets at: 1 on her own platform, the share of its platform a star reaches on another)."""
    platform: int
    x: float
    cover: tuple[tuple[int, float], ...]


@dataclass
class Plan:
    """A loop and what it gives in the long run."""
    stops: tuple[int, ...]   # Patrol.stops indices, in the order she goes round
    rate: float              # EXP a second (less the danger's cost)
    cycle: float             # s a round takes
    travel: float            # ... of which route times
    kills: float             # ... of which shooting
    exp: float               # EXP a round
    points: int              # spawn points it gets at
    hp: float = 0.0          # bars of HP a round (the danger hook)

    def text(self, plats: Callable[[int], int]) -> str:
        return (f"{[plats(i) for i in self.stops]}: {self.rate * 60:.0f} EXP a minute (a round {self.cycle:.0f} s: "
                f"{self.travel:.0f} s of travel, {self.kills:.0f} of shooting; {self.points} spawn points, "
                f"{self.exp:.0f} EXP)")

    def record(self, plats: Callable[[int], int]) -> dict:
        return {"loop": [plats(i) for i in self.stops], "rate": round(self.rate * 60, 1), "cycle": round(self.cycle, 1),
                "travel": round(self.travel, 1), "shooting": round(self.kills, 1), "points": self.points,
                "exp": round(self.exp, 1)}


def points_of(model: MapModel, kinds: set[str] | None, exp: dict[str, float]) -> list[Point]:
    """The map's spawn points of `kinds` (None: any) that give EXP, each on the platform under it (MapModel's rule:
    monsters_by_platform's)."""
    out = []
    for s in model.spawns:
        p = model.platform_under(s.x, s.y, reach=SPAWN_REACH)
        kind = kind_of(s.name or str(s.mob_id))
        if p is None or (kinds is not None and kind not in kinds) or exp.get(kind, 0) <= 0:
            continue
        out.append(Point(len(out), p.id, float(s.x), kind, float(exp[kind])))
    return out


def stops_of(model: MapModel, points: list[Point], danger: Callable[[int, float], float] | None = None) -> list[Stop]:
    """A stop on each platform from which she gets at points: the x (every STAND_STEP, EDGE_MARGIN from its ends) that
    reaches the most EXP, less STOP_DANGER of it a unit of the `danger` hook's there (None: none), the nearest its
    middle of those; her own platform's points count whole wherever she stands."""
    out = []
    for p in model.places:
        lo, hi = p.left + EDGE_MARGIN, p.right - EDGE_MARGIN
        mid = (p.left + p.right) / 2
        xs = [mid] if hi < lo else [lo + i * STAND_STEP for i in range(int((hi - lo) // STAND_STEP) + 1)] + [hi]
        own = [(q.id, 1.0) for q in points if q.platform == p.id]
        mine = [q for q in points if q.platform == p.id]
        best = None  # (EXP reached, -distance from the middle, x, cover)
        tried = []   # (EXP reached, x, cover) of each place (STOP_SLACK)
        for x in xs:
            y = p.y_at(x)
            cov, kept = list(own), {}
            for q in points:
                if q.platform == p.id:
                    continue
                key = (q.platform, q.kind)
                if key not in kept:
                    kept[key] = spots.coverage(model.platform(q.platform), x, y, p, q.kind)
                if kept[key] > 0:
                    cov.append((q.id, round(kept[key], 3)))
            value = sum(points[i].exp * c for i, c in cov)
            if value > 0 and danger is not None:
                value *= max(0.0, 1.0 - STOP_DANGER * danger(p.id, x))
            if value > 0 and (best is None or (round(value, 6), -abs(x - mid)) > best[:2]):
                best = (round(value, 6), -abs(x - mid), x, tuple(cov))
            if value > 0:
                tried.append((round(value, 6), x, tuple(cov)))
        if best is not None and STOP_SLACK > 0 and mine and danger is not None \
                and any(danger(p.id, q.x) > 0 for q in mine):
            def clear(x: float) -> float:
                return round(min(min(abs(x - q.x), reach.ground_sure(reach.body_of(q.kind)[1])[0]) for q in mine), 1)
            near_best = [t for t in tried if t[0] >= best[0] * (1.0 - STOP_SLACK)]
            v, x, cov = max(near_best, key=lambda t: (clear(t[1]), t[0], -abs(t[1] - mid)))
            best = (v, -abs(x - mid), x, cov)
        if best is not None:
            out.append(Stop(p.id, round(best[2], 1), best[3]))
    return out


def reaching(here, monsters: list[Monster]):
    """The places along platform `here` (every STAND_STEP, EDGE_MARGIN from its ends) from which a star reaches one of
    `monsters` at least (reach.can_hit), as (x, how many)."""
    near = [m for m in monsters if -SHOT_UP - 30 <= m.y - here.y_at(min(max(m.x, here.left), here.right))
            <= SHOT_DOWN + 30]
    if not near:
        return
    lo, hi = here.left + EDGE_MARGIN, here.right - EDGE_MARGIN
    xs = [(lo + hi) / 2] if hi < lo else [lo + i * STAND_STEP for i in range(int((hi - lo) // STAND_STEP) + 1)]
    for x in xs:
        n = sum(reach.can_hit(here, x, here.y_at(x), m) for m in near)
        if n:
            yield x, n


def back(dt: float, rate: float = REFILL_RATE) -> float:
    """The chance a spawn point seen free `dt` s ago has a monster now (see the top): the refill's `rate` (a second) at
    first, Q_CEIL in the long run."""
    if dt <= 0 or rate <= 0:
        return 0.0
    return filled(rate * dt)


def filled(mean: float) -> float:
    """The chance a free point has a monster after `mean` of them were expected to come to it (back's)."""
    return Q_CEIL * (1.0 - math.exp(-mean / Q_CEIL)) if mean > 0 else 0.0


def passes(kills: list[float], t0: float, t1: float) -> float:
    """How many of her `kills` (times, sorted) had the pass that brought them back between t0 and t1: each a pass
    within TICK_S after it (the map's phase not known: evenly)."""
    lo, n = bisect.bisect_left(kills, t0 - TICK_S), 0.0
    for k in kills[lo:]:
        if k >= t1:
            break
        n += max(0.0, min(t1, k + TICK_S) - max(t0, k)) / TICK_S
    return n


def not_back(dt: float, rate: float = REFILL_RATE) -> float:
    """The chance a spawn point seen free `dt` s ago has no monster yet (1 - back)."""
    return 1.0 - back(dt, rate)


def caught(cycle: float, rate: float = REFILL_RATE) -> float:
    """The monsters she kills at a point a round, going round every `cycle` s, in the long run: what's back a cycle
    after she found it free (her kill or nothing there: from free the refill has no memory, so it's the same whichever).
    About cycle x rate when she comes round far more often than it refills, Q_CEIL when far less."""
    return back(cycle, rate)


def refill_median(rate: float = REFILL_RATE) -> float:
    """s until a point seen free has a monster with even odds by the refill alone (its rate; the ceiling aside)."""
    return math.log(2) / rate if rate > 0 else math.inf


def detour_s(exp: float, kill_s: float, far: float, on: float, far_next: float) -> float:
    """s a pile's detour costs her (see the top): shooting its `exp` EXP (`kill_s` s a point), STOP_S, and the travel to
    it (`far` s) and on from it to the stop she was going to (`on` s) beyond going there straight (`far_next` s): none
    for a pile on her way."""
    return exp * kill_s + STOP_S + max(0.0, far + on - far_next)


def pays(exp: float, detour: float, rate: float) -> bool:
    """Whether a pile expected to give `exp` EXP pays for its `detour` s against the loop's `rate` EXP a second: PULL_GAIN
    times what the loop gives meanwhile."""
    return exp >= PULL_GAIN * rate * detour


def touch_shares(model: MapModel, her: danger_model.Her) -> dict[str, float]:
    """Kind -> the share of her HP a touch by it takes, as the positioning model has it (attack.danger.Danger.share:
    the map's touch damage, MapModel.mob_damage, by the game's formula at her level and DEF, over her max HP; 0 under
    DANGER_MIN), for `her` (attack.danger.Her: as perception read her, Hunter.her). {} on a map whose touch damage
    wasn't read: no danger."""
    return {k: t.share for k, t in danger_model.of_map(model, her).threats.items()}


def default_danger(model: MapModel, points: list[Point], touch: dict[str, float]) -> Callable[[int, float], float]:
    """The danger hook's default (see the top): on a platform, the touch shares (`touch`: kind -> share) of its spawn
    points of kinds over DANGER_MIN; 0 where MapModel.safe."""
    on = Counter()
    for q in points:
        if touch.get(q.kind, 0.0) >= DANGER_MIN:
            on[q.platform] += touch[q.kind]

    def danger(pid: int, x: float) -> float:
        return 0.0 if model.safe(pid) else on.get(pid, 0.0)
    return danger


def _mean_rate(loglik: list[float]) -> float:
    """The mean of RATE_GRID's rates by `loglik` (the log likelihood of each) and the prior, a gamma about REFILL_RATE
    counting as REFILL_PRIOR_N refills (on a log grid each rate weighs r d(log r))."""
    a, b = REFILL_PRIOR_N, REFILL_PRIOR_N / REFILL_RATE
    logs = [ll + (a - 1.0) * math.log(r) - b * r for r, ll in zip(RATE_GRID, loglik)]
    top = max(logs)
    ws = [math.exp(v - top) * r for v, r in zip(logs, RATE_GRID)]
    return sum(wi * r for wi, r in zip(ws, RATE_GRID)) / sum(ws)


_PRIOR_MEANS: dict[tuple, float] = {}


def _prior_mean() -> float:
    """The grid's mean of the prior alone, as REFILL_RATE, REFILL_PRIOR_N and RATE_GRID are now (a test or a replay may
    set them): Belief.rate is REFILL_RATE times the grid's mean over it, the prior's exactly with nothing seen. (Until
    the review of 2026-10-29 it was taken once, as the module loaded: REFILL_RATE set to a quarter after gave a
    sixteenth.)"""
    key = (REFILL_RATE, REFILL_PRIOR_N, RATE_GRID)
    if key not in _PRIOR_MEANS:
        _PRIOR_MEANS[key] = _mean_rate([0.0] * len(RATE_GRID))
    return _PRIOR_MEANS[key]


class Belief:
    """How many monsters each spawn point has, from her memory (see the top). `update` at every look."""

    def __init__(self, model: MapModel, points: list[Point]):
        self.model, self.points = model, points
        self._speeds = speeds(model)             # kind -> how fast it walks (units/s: born_reach)
        self._by = {}
        for q in points:
            self._by.setdefault((q.platform, q.kind), []).append(q)
        self.at: dict[int, list] = {}            # point id -> the monsters remembered at it (localization.memory)
        self.empty_since: dict[int, float] = {}  # point id -> since when it's empty (a look without it, her kill)
        self.killed: dict[int, float] = {}       # point id -> her kill that emptied it (refills learned from)
        self.empty_seen: dict[int, float] = {}   # point id -> the latest look that saw it empty
        self.rival_at: dict[int, float] = {}     # point id -> the latest look with another player by it
        self.gone_by: Counter = Counter()        # kind -> monsters by another player gone without her kill
        self.kept_by: Counter = Counter()        # kind -> ... killed by her, or seen there TAKE_KEEP_S on
        self.refills = 0                         # refills watched, in all (the log's)
        self.watched_s = 0.0                     # ... and the s of points watched staying free, in all
        self.kills: list[float] = []             # her kills' times (own, and the passes that bring them back)
        self.spawns = sum(1 for s in model.spawns if model.platform_under(s.x, s.y, reach=SPAWN_REACH) is not None)
        self.cap = model.mob_cap()  # (with her alone: more only with more players than half of it)
        self.weight = self.weight_of(0.0)  # the map's points as a pass draws them, at its cap
        self._loglik = [0.0] * len(RATE_GRID)    # what she saw, as the log likelihood of each of RATE_GRID ...
        self._weighed_t = 0.0                    # ... weighed by age (REFILL_MEMORY_S) as of this
        self._others = REFILL_RATE               # the others' part as of the latest update (others) ...
        self._rate = REFILL_RATE                 # ... and the rate (rate)
        self.t = 0.0
        self._events = 0
        self._kept_ids: set[int] = set()
        self._covered_t = -1e18                  # the latest look's cover taken (Memory.covers)

    # --- what it holds ---------------------------------------------------------------------------------

    @property
    def rate(self) -> float:
        """The refill's rate (a second, a free point; see the top), as of the latest update: what her own kills bring
        back (`own` over the map's `weight`) and the others' part (`others`)."""
        return self._rate

    @property
    def others(self) -> float:
        """The refill's rate (a second, a free point) the other players' kills bring, as of the latest update: the mean
        over RATE_GRID of what she saw (`_seen`, net of her kills' passes), weighed by age (REFILL_MEMORY_S), and the
        prior, a gamma about REFILL_RATE counting as REFILL_PRIOR_N refills."""
        return self._others

    def weight_of(self, free: float) -> float:
        """The map's points a pass draws among, the free ones (a held one never), with `free` of them kept free (a
        round's, whose monsters she kills as they come) and the rest held up to the cap."""
        held = min(max(0.0, self.spawns - free), self.cap)
        return max(1.0, self.spawns - held)

    def own(self, t: float) -> float:
        """Her kills a second over the OWN_S before `t`: the monsters the map's passes bring back for them."""
        return (bisect.bisect_right(self.kills, t) - bisect.bisect_right(self.kills, t - OWN_S)) / OWN_S

    def _learned(self) -> float:
        w = math.exp(-max(0.0, self.t - self._weighed_t) / REFILL_MEMORY_S)
        return REFILL_RATE * _mean_rate([w * ll for ll in self._loglik]) / _prior_mean()

    @property
    def scale(self) -> float:
        """The refill's pace against the prior's (REFILL_RATE over the rate learned): 1 as the prior has it, 2 twice
        as slow (the log's)."""
        return REFILL_RATE / self.rate

    def _seen(self, t: float, gap: float, born: bool) -> None:
        """At time `t`, a point seen free `gap` s before seen again: with a new monster (`born`) or free still. What her
        kills brought back in the gap (their passes in it, over the map's weight) is counted in: the others' part is
        learned."""
        if gap <= 0:
            return
        if t > self._weighed_t:
            w = math.exp(-(t - self._weighed_t) / REFILL_MEMORY_S)
            self._loglik = [w * ll for ll in self._loglik]
            self._weighed_t = t
        mine = passes(self.kills, t - gap, t) / self.weight
        for k, r in enumerate(RATE_GRID):
            got = filled(r * gap + mine)
            self._loglik[k] += math.log(max(1e-12, got if born else 1.0 - got))
        if born:
            self.refills += 1
        else:
            self.watched_s += gap

    def stock(self, i: int, t: float) -> float:
        """How many monsters point `i` is expected to have at time `t` (now or ahead): for one empty, what's back by
        then since the latest look that saw it empty (`back`, at the rate learned)."""
        es = self.at.get(i)
        if es:
            return sum(e.p(t) for e in es)
        since = self.empty_since.get(i)
        if since is not None:
            known = max(since, self.empty_seen.get(i, since))
            return back(t - known, self.rate)
        return Q_UNKNOWN

    def take(self, kind: str) -> float:
        """The share of its kind's monsters another player by a point takes (see the top)."""
        g, k = self.gone_by[kind], self.kept_by[kind]
        return (g + TAKE_PRIOR * TAKE_N) / (g + k + TAKE_N)

    def share(self, i: int, t: float) -> float:
        """The share of point `i`'s monsters that is hers: 1 - take with another player by it lately, else 1."""
        return 1.0 - self.take(self.points[i].kind) if t - self.rival_at.get(i, -1e18) <= RIVAL_KEEP_S else 1.0

    def due(self, i: int, t: float) -> float | None:
        """s from `t` until point `i`, empty, has a monster with even odds by the refill alone (refill_median after
        the latest look that saw it empty, at the rate learned; None: not known to be empty). The refill has no memory:
        a point watched empty is due that long after the latest look, however long it has been empty."""
        since = self.empty_since.get(i)
        if since is None or self.at.get(i):
            return None
        known = max(since, self.empty_seen.get(i, since))
        return known + refill_median(self.rate) - t

    def born_reach(self, kind: str, dt: float) -> float:
        """How far from its spawn point a monster of `kind` born there up to `dt` s before can be seen (a refill: see
        the top): SPAWN_NEAR (tracking's: a new monster that near a point of its kind spawned there) and what it can
        have walked meanwhile, as the memory matches a monster seen again (MATCH_SPEED times its kind's speed, over
        MATCH_WALK_S at most). Until the review of 2026-10-29 SPAWN_NEAR whatever the gap: a monster born while she was
        away had mostly walked off its point by the time she saw it, so the gap went uncounted where the gaps that
        stayed free all counted, and the rate came out low (her looks of 2026-10-29 replayed, pooled over a run: 27-30%
        under the perception service's own count on two runs, 10% on a third; within 8-11% so, of the refills counted
        a monster the service had seen there before 11-15%, 9-14% before: D:/MapleHelper-scratch/respawn/fix)."""
        walk = self._speeds.get(kind, DEFAULT_SPEED) * min(max(0.0, dt), MATCH_WALK_S)
        return SPAWN_NEAR + MATCH_SPEED * walk

    # --- keeping it up ---------------------------------------------------------------------------------

    def update(self, t: float, memory, players: list[Player] | None = None) -> None:
        """At a look at time `t`: the monsters her memory has (localization.memory.Memory) put at their points, her
        kills and the looks that searched a point's monster (since the last update: Memory.covers) emptying points,
        other players by them."""
        self.t = t
        self._take_events(t, memory)
        at: dict[int, list] = {}
        for (pid, kind), qs in self._by.items():
            es = [e for e in memory.entries if e.live and e.platform == pid and e.kind == kind]
            if not es:
                continue
            pairs = sorted((abs(e.x - q.x), e.id, q.id, e) for e in es for q in qs)
            used_e, used_q = set(), set()
            for _, eid, qid, e in pairs:  # one monster a point first, the nearest pairs first
                if eid in used_e or qid in used_q:
                    continue
                used_e.add(eid)
                used_q.add(qid)
                at.setdefault(qid, []).append(e)
            for e in es:  # more monsters than points: each at its nearest
                if e.id not in used_e:
                    at.setdefault(min(qs, key=lambda q: abs(q.x - e.x)).id, []).append(e)
        self.at = at
        for i, es in at.items():  # a monster seen there since it was emptied: full again
            since = self.empty_since.get(i)
            if since is None or not any(e.seen > since for e in es):
                continue
            q, last = self.points[i], self.empty_seen.get(i, -1e18)  # a new one where one born at its point since the
            # latest look that saw it free can be: a refill, seen if that look was lately
            new = [e.first for e in es if e.first > since and abs(e.x - q.x) <= self.born_reach(q.kind, e.seen - last)]
            if new and 0.0 < min(new) - last <= WATCH_GAP_S:
                self._seen(min(new), min(new) - last, True)
            self.empty_since.pop(i, None)
            self.killed.pop(i, None)
        for cover in [c for c in memory.covers if c.t > self._covered_t]:  # a look that searched a point's monster
            for q in self.points:  # ... and nothing sure there: empty
                if any(e.p(cover.t) >= 0.5 for e in at.get(q.id, ())) \
                        or not memory.holds_body(cover, q.platform, q.x, q.kind):
                    continue
                last = self.empty_seen.get(q.id) if q.id in self.empty_since else None
                if last is not None and 0.0 < cover.t - last <= WATCH_GAP_S:  # free still since a look lately
                    self._seen(cover.t, cover.t - last, False)
                self.empty_since.setdefault(q.id, cover.t)
                self.empty_seen[q.id] = max(cover.t, self.empty_seen.get(q.id, -1e18))
            self._covered_t = cover.t
        del self.kills[:bisect.bisect_left(self.kills, t - max(OWN_S, WATCH_GAP_S + TICK_S))]
        self._others = self._learned()
        self._rate = self._others + self.own(t) / self.weight
        if players is not None:
            for q in self.points:
                y = self.model.platform(q.platform).y_at(q.x)
                if near_rival(players, q.x, y, RIVAL_X, RIVAL_DY) is not None:
                    self.rival_at[q.id] = t
        for e in memory.entries:  # by another player, and seen there TAKE_KEEP_S after she first saw it: left to her
            if e.live and e.rival and e.id not in self._kept_ids and e.seen - e.first >= TAKE_KEEP_S:
                self._kept_ids.add(e.id)
                self.kept_by[e.kind] += 1

    def _take_events(self, t: float, memory) -> None:
        """Her memory's new events since the last update: her kills empty their points, and so do another player's
        that the feed saw (taken: localization.memory's `theirs`); the monsters by another player gone or killed teach
        `take`, one she killed left to her, one another took or looks forgot taken."""
        c = memory.counts
        total = c["new"] + c["killed"] + c.get("taken", 0) + c["gone"] + c["aged"]
        new, self._events = min(total - self._events, len(memory.events)), total
        for what, when, e in list(memory.events)[len(memory.events) - new:] if new > 0 else ():
            if what == "killed":
                bisect.insort(self.kills, when)
                qs = self._by.get((e.platform, e.kind))
                if qs:
                    x = e.died_x if e.died_x is not None else e.x
                    q = min(qs, key=lambda q: abs(q.x - x))
                    self.empty_since[q.id] = self.killed[q.id] = when
                    self.empty_seen[q.id] = when
                if e.rival and e.id not in self._kept_ids:
                    self._kept_ids.add(e.id)
                    self.kept_by[e.kind] += 1
            elif what == "taken":  # another player's kill: empty, not her kill to time a refill from
                qs = self._by.get((e.platform, e.kind))
                if qs:
                    q = min(qs, key=lambda q: abs(q.x - e.x))
                    self.empty_since[q.id] = self.empty_seen[q.id] = when
                    self.killed.pop(q.id, None)
                if e.rival and e.id not in self._kept_ids:
                    self._kept_ids.add(e.id)
                    self.gone_by[e.kind] += 1
            elif what == "gone" and e.rival and e.id not in self._kept_ids and e.missed is not None \
                    and e.ended - e.seen <= MEMORY_S:  # forgotten by looks; by age, she never looked again
                self._kept_ids.add(e.id)
                self.gone_by[e.kind] += 1


class Patrol:
    """The map's points and stops, the belief, the loop she follows and the choice of her next stop (see the top).
    `travel(platform, x, platform, x)`: s by the planner's routes (None: no way); `danger(platform, x)`: the hook, the
    danger of fighting there in the positioning model's unit (None: default_danger, with `touch`: kind -> the share of
    her HP a touch takes, None: none, no danger: a hunter's hook is her positioning model's, Hunter.danger)."""

    def __init__(self, model: MapModel, kinds: set[str] | None, exp: dict[str, float],
                 travel: Callable[[int, float, int, float], float | None],
                 danger: Callable[[int, float], float] | None = None, touch: dict[str, float] | None = None):
        self.model = model
        self.points = points_of(model, kinds, exp)
        self.danger = danger or default_danger(model, self.points, {} if touch is None else touch)
        self.stops = stops_of(model, self.points, self.danger)
        self.belief = Belief(model, self.points)
        self.travel = travel
        avg = [q.exp for q in self.points]
        self.avg_exp = sum(avg) / len(avg) if avg else 10.0
        self.min_exp = min(avg, default=10.0)  # the cheapest kind's EXP (PULL_MIN)
        self.left_at: dict[int, float] = {}    # stop -> when she last left it with nothing in reach there
        self.empty_run = 0                     # stops she got to running that gave nothing (EMPTY_RUN)
        self._gets: dict[tuple, bool] = {}
        self.kill_s = KILL_S_PER_EXP   # s of shooting a point of EXP (learn_kills)
        self.plan: Plan | None = None  # the loop she follows
        self.changed = -1e18           # when it last changed
        self.planned = -1e18           # when a better one was last looked for
        self.at_loop = 0               # the loop stop she was at last (index into plan.stops)
        self.pulled: tuple[int, int] | None = None  # a pile she was pulled to, the loop's stop its detour went on to
        self.loops = 0                 # rounds taken up (the first, and every change)
        self.hp_seen: Counter = Counter()  # platform -> the share of her HP she has seen go there (learn_hp) ...
        self.hp_time: Counter = Counter()  # ... over this many s there
        self.hp_hook = 0.0                 # ... and what the danger hook said she'd lose meanwhile (hook_scale)
        self.hp_price = HP_PRICE           # EXP a bar of her HP (behavior.grind sets it from her max HP as read)
        self._legs: dict[tuple[int, int], float | None] = {}

    def platform_of(self, i: int) -> int:
        return self.stops[i].platform

    def stop_on(self, pid: int) -> int | None:
        """The stop on platform `pid`, if there's one."""
        return next((i for i, s in enumerate(self.stops) if s.platform == pid), None)

    def leg(self, i: int, j: int) -> float | None:
        """s from stop i to stop j (planner), kept."""
        if i == j:
            return 0.0
        key = (i, j)
        if key not in self._legs:
            a, b = self.stops[i], self.stops[j]
            self._legs[key] = self.travel(a.platform, a.x, b.platform, b.x)
        return self._legs[key]

    def gets(self, i: int, e) -> bool:
        """Whether at stop i she gets at remembered monster `e` (localization.memory.Remembered): on the stop's
        platform (she walks up to it), or in a star's reach from somewhere along it (grind walks there: `reaching`)."""
        st = self.stops[i]
        if e.platform == st.platform:
            return True
        key = (st.platform, e.platform, e.kind, round(e.x / REACH_ROUND))
        if key not in self._gets:
            self._gets[key] = any(True for _ in reaching(self.model.platform(st.platform), [e.monster]))
        return self._gets[key]

    def learn_hp(self, pid: int, lost: float, dt: float, x: float | None = None) -> None:
        """`dt` s on platform `pid` (at map x `x`: her stop's place there; None, its stop's, or its middle), with `lost`
        of her HP gone meanwhile (hp_rate), and what the danger hook said she'd lose meanwhile (`hook_scale`)."""
        self.hp_seen[pid] += lost
        self.hp_time[pid] += dt
        if x is None:
            st = self.stop_on(pid)
            q = self.model.platform(pid)
            x = self.stops[st].x if st is not None else (q.left + q.right) / 2
        self.hp_hook += DANGER_HP_S * self.danger(pid, x) * dt

    def hook_scale(self) -> float:
        """How many times the danger hook's HP (DANGER_HP_S a unit of danger) she has lost where she's been, over every
        place she learned it at (learn_hp), HOOK_PRIOR_HP of HP lost as the hook said counted in: the hook of a place she
        hasn't been at is the hook times this (hp_rate)."""
        seen = sum(self.hp_seen.values())
        return (HOOK_PRIOR_HP + seen) / (HOOK_PRIOR_HP + self.hp_hook)

    def hp_rate(self, pid: int, x: float) -> float:
        """The share of her HP a second she loses fighting on platform `pid` at x: the danger hook's (DANGER_HP_S a
        unit of it) as scaled by what she has lost against it everywhere (`hook_scale`), as HP_PRIOR_S s there, blended
        with what she has seen go there."""
        hook = DANGER_HP_S * self.danger(pid, x) * self.hook_scale()
        return (hook * HP_PRIOR_S + self.hp_seen[pid]) / (HP_PRIOR_S + self.hp_time[pid])

    def learn_kills(self, throws: int, exp: float) -> None:
        """Her kill time a point of EXP from the throws she made and the EXP her kills gave (KILL_PRIOR_EXP)."""
        self.kill_s = (KILL_S_PER_EXP * KILL_PRIOR_EXP + THROW_S * throws) / (KILL_PRIOR_EXP + exp)

    # --- a round's worth in the long run ---------------------------------------------------------------

    def rate(self, loop: Iterable[int], t: float | None = None, kept: dict[int, float] | None = None) -> Plan | None:
        """What going round `loop` gives in the long run (see the top), with the other players' shares as at time `t`
        (None: the belief's latest); None if a leg has no way. `kept`: stop -> hp_rate there, kept for a search (best:
        the same for every round it weighs; the danger hook's sum over the map's spawn points was nine tenths of its
        time, 0.9 s a search at a low refill rate, where the rounds grow to 12 stops)."""
        loop = tuple(loop)
        t = self.belief.t if t is None else t
        travel = 0.0
        for a, b in zip(loop, loop[1:] + loop[:1]):
            if len(loop) == 1:
                break
            d = self.leg(a, b)
            if d is None:
                return None
            travel += d
        cov: dict[int, tuple[float, int]] = {}  # point -> (share in reach, the stop it's got at from)
        for i in loop:
            for q, c in self.stops[i].cover:
                if q not in cov or c > cov[q][0]:
                    cov[q] = (c, i)
        b = self.belief
        pts = [(self.points[q], c * b.share(q, t), i) for q, (c, i) in cov.items()]
        kept = {} if kept is None else kept
        for i in loop:
            if i not in kept:
                kept[i] = self.hp_rate(self.stops[i].platform, self.stops[i].x)
        hp_s = {i: kept[i] for i in loop}
        full = sum(g * q.exp for q, g, _ in pts)  # EXP a round with every point back (hers) ...
        full_n = sum(g for _, g, _ in pts)  # ... its monsters ...
        full_hp = sum(g * q.exp * self.kill_s * hp_s[i] for q, g, i in pts)  # ... and the HP its shooting costs
        base = travel + STOP_S * len(loop)
        cycle = base + full * self.kill_s  # every point full: the longest a round takes
        # The refill: the others' part and what the round's own kills bring back (below: over the map's points as a
        # pass draws them, the round's free, the rest held up to the cap: what comes back on it is at most what it
        # kills), from where they meet while few of its points are back each round (its kills full_n x the refill)
        weight = max(b.weight_of(full_n), full_n)
        refill = b.others / max(0.05, 1.0 - full_n / weight)
        for _ in range(12):  # what's back in a round (the same share at every point: one rate), the round's shooting
            # for it (cycle only shortens), and the refill with what its kills bring back over the map (her kills a
            # second over the map's weight: a round of a few of its points mostly feeds the rest)
            back_share = caught(cycle, refill)
            shoot = back_share * full * self.kill_s
            cycle = max(base + shoot, 1e-6)
            refill = b.others + back_share * full_n / cycle / weight
        exp, hp = back_share * full, back_share * full_hp
        return Plan(loop, (exp - self.hp_price * hp) / cycle, cycle, travel, shoot, exp, len(pts), hp)

    def camping(self, t: float | None = None) -> Plan | None:
        """The best round of one stop: standing there, each point's monster killed as it comes back (caught). Grinding
        never takes it up (a round has LOOP_MIN stops); it says what camping would give, for the log and the tests."""
        return max((p for i in range(len(self.stops)) if (p := self.rate((i,), t)) is not None),
                   key=lambda p: p.rate, default=None)

    def best(self, t: float | None = None) -> Plan | None:
        """The best round of LOOP_MIN stops or more (see the top): grown from each of the SEEDS stops worth the most on
        their own, by the stop that raises its rate most, the second stop taken whether it does or not."""
        kept: dict[int, float] = {}  # (rate's, for the search)
        alone = [p for i in range(len(self.stops)) if (p := self.rate((i,), t, kept)) is not None]
        seeds = [p.stops[0] for p in sorted(alone, key=lambda p: -p.rate)[:SEEDS]]
        top = None
        for seed in seeds:
            loop, cur = [seed], None
            while len(loop) < LOOP_MAX:
                found = None
                for j in range(len(self.stops)):
                    if j in loop:
                        continue
                    at = self._insert(loop, j)
                    if at is None:
                        continue
                    p = self.rate(loop[:at] + [j] + loop[at:], t, kept)
                    if p is not None and (found is None or p.rate > found.rate):
                        found = p
                if found is None or (cur is not None and found.rate <= cur.rate * (1 + 1e-3)):
                    break
                loop, cur = list(found.stops), found
            if cur is not None and (top is None or cur.rate > top.rate):
                top = cur
        return top

    def _insert(self, loop: list[int], j: int) -> int | None:
        """Where in `loop` stop j adds the least travel (None: no way to it or back)."""
        best = None
        for k in range(len(loop)):
            a, b = loop[k - 1], loop[k]
            ab, aj, jb = self.leg(a, b), self.leg(a, j), self.leg(j, b)
            if aj is None or jb is None or ab is None:
                continue
            extra = aj + jb - (ab if len(loop) > 1 else 0.0)
            if best is None or extra < best[0]:
                best = (extra, k)
        return None if best is None else best[1]

    def replan(self, t: float, force: bool = False) -> str | None:
        """Look for a better round (every REPLAN_S): taken up when it's LOOP_GAIN better than hers by the same belief
        and LOOP_MIN_S after the last change (or she has none, or hers has no way round). Why it changed, or None."""
        run = self.empty_run
        empty = run >= EMPTY_RUN
        if not force and not empty and self.plan is not None and t - self.planned < REPLAN_S:
            return None
        self.planned = t
        if empty:  # the run's look taken: another EMPTY_RUN stops with nothing before the next out of turn (until the
            # review of 2026-10-29 the run lifted REPLAN_S until a stop gave EXP: a search at every stop left, about
            # 1 s each at a low rate, where the loops grow to 12 stops)
            self.empty_run = 0
        new = self.best(t)
        if new is None:
            return None
        if self.plan is None:
            return self._take(new, t, "the first")
        if set(new.stops) == set(self.plan.stops):  # hers in its own order, its rate by the belief now (until
            # 2026-10-29 the same stops found in another order kept the rate it was taken up at: a 2-stop round's two
            # orders are the same round, so a pull's bar and a stop's stayed at the first look's for good)
            self.plan = new if new.stops == self.plan.stops else (self.rate(self.plan.stops, t) or self.plan)
            return None
        now = self.rate(self.plan.stops, t)
        if now is None:
            return self._take(new, t, "hers has no way round")
        if (t - self.changed < LOOP_MIN_S and not empty) or new.rate < (1 + LOOP_GAIN) * now.rate:
            self.plan = now
            return None
        why = f"{new.rate * 60:.0f} EXP a minute against {now.rate * 60:.0f}"
        return self._take(new, t, why + (f"; {run} stops running gave nothing" if empty else ""))

    def _take(self, plan: Plan, t: float, why: str) -> str:
        self.plan, self.changed, self.at_loop, self.pulled = plan, t, 0, None
        self.loops += 1
        self.empty_run = 0
        return why

    def left(self, i: int, t: float, nothing: bool, exp: float) -> None:
        """She left stop i at time t: with nothing in reach there (`nothing`: `left_at`), `exp` EXP got there
        (EMPTY_RUN counts the stops running that gave nothing)."""
        if nothing:
            self.left_at[i] = t
        self.empty_run = self.empty_run + 1 if nothing and exp <= 0 else 0

    # --- the next stop -----------------------------------------------------------------------------------

    def expect(self, i: int, t: float) -> tuple[float, float, float, float]:
        """At stop i at time `t`: the EXP expected in reach (hers: share), of it the EXP of monsters she has seen, the s
        of shooting it takes, and the EXP it holds full (a monster at each point). A point's monsters her memory has
        count each as sure as it is, where she gets at it from the stop (`gets`) and it was seen since she last left
        the stop with nothing in reach (`left_at`); a point it has none at, by its stock and the share of it in reach."""
        b, left = self.belief, self.left_at.get(i, -1e18)
        exp = seen = full = 0.0
        for q, c in self.stops[i].cover:
            g = b.share(q, t) * self.points[q].exp
            es = b.at.get(q)
            if es:
                got = g * sum(e.p(t) for e in es if e.seen > left and self.gets(i, e))
                exp += got
                seen += got
            else:
                exp += g * c * b.stock(q, t)
            full += g * c
        return exp, seen, exp * self.kill_s, full

    def wait_gain(self, i: int, t: float, horizon: float = DUE_S) -> float:
        """EXP a second of waiting `horizon` s at stop i for what's expected back in reach meanwhile (her share; the
        points her memory has no monster at), its kills included. A point watched empty comes back at a steady rate
        (the refill has no memory), so this hardly changes as she waits: at the prior's rate 8.5% of a Horny Mushroom
        in 2 s, 1.5 EXP/s."""
        b, exp = self.belief, 0.0
        for q, c in self.stops[i].cover:
            if not b.at.get(q):
                exp += c * b.share(q, t) * self.points[q].exp * max(0.0, b.stock(q, t + horizon) - b.stock(q, t))
        return exp / (horizon + exp * self.kill_s)

    def due(self, i: int, t: float) -> float | None:
        """s until the soonest respawn due at stop i's points (in reach of it), if one is."""
        due = [d for q, c in self.stops[i].cover if c > 0 and (d := self.belief.due(q, t)) is not None]
        return min(due) if due else None

    def detour(self, j: int, nxt: int, far: float, far_next: float, exp: float) -> float | None:
        """s a detour to stop j `far` s away costs her on her way to stop `nxt` `far_next` s away, with `exp` EXP to
        shoot there (detour_s, her kill time; the route time on from j to `nxt`); None if there's no way on."""
        on = self.leg(j, nxt)
        return None if on is None else detour_s(exp, self.kill_s, far, on, far_next)

    def net(self, j: int, exp: float) -> float:
        """`exp` EXP shot at stop j less what touches there cost her meanwhile (`hp_price` EXP a bar of her HP, at her
        kill time and the HP she loses there a second: hp_rate), as the loop's rate is (Plan.rate)."""
        s = self.stops[j]
        return exp * (1.0 - self.hp_price * self.kill_s * self.hp_rate(s.platform, s.x))

    def choose(self, t: float, here: int | None, x: float | None, current: int | None = None,
               skip: Iterable[int] = (), heading: int | None = None, peek: bool = False) -> tuple[int | None, dict]:
        """The next stop from (platform `here`, map x `x`), having been at stop `current`, never one on a platform in
        `skip` (see the top): (its index, what decided it, for the log and the telemetry). `heading`: the stop she's
        on her way to, which stands in for the loop's next (only a pull takes her elsewhere); `peek`: what she'd do,
        her place in the loop left as it is (a check on the way, a replay)."""
        skip = set(skip)
        if self.plan is None:
            self.replan(t, force=True)
        if self.plan is None:
            return None, {"why": "no stop on this map"}
        loop = list(self.plan.stops)
        pulled = self.pulled if self.pulled is not None and self.pulled[1] in loop else None
        at_loop = self.at_loop  # (kept to itself when peeking)
        if current in loop:  # from the loop's stop she was at; from a pile there, on from the stop its detour went to
            at_loop = (loop.index(pulled[1]) - 1) % len(loop) if pulled is not None and pulled[0] == current \
                else loop.index(current)
        options = {}
        for j in range(len(self.stops)):
            s = self.stops[j]
            if j == current or s.platform in skip:
                continue
            if here is None:
                far = 0.0
            elif here == s.platform:
                far = abs(s.x - x) / WALK_SPEED
            else:
                far = self.travel(here, x, s.platform, s.x)
            if far is None:
                continue
            exp, seen, shoot, full = self.expect(j, t + far)
            options[j] = (far, exp, seen, exp / (far + shoot + STOP_S), full)
        order = [loop[(at_loop + k) % len(loop)] for k in range(1, len(loop) + 1)]
        order = [j for j in order if j in options] or [j for j in sorted(options, key=lambda j: options[j][0])]
        if not order:
            return None, {"why": "no stop to get to"}
        nxt, skipped = None, []
        for j in order:
            if options[j][1] >= SKIP_FULL * options[j][4] and options[j][4] > 0:
                nxt = j
                break
            skipped.append(j)
        why = "the loop"
        if nxt is None:  # nothing due on the loop: of the stops due elsewhere, where the most is expected a second, if
            # more than the loop's next; else the loop's next (until 2026-10-29 the loop's next whatever: at 16:39-16:44
            # on 2026-10-29 that was 14 and 7 in turn, both empty, with the middle unseen for minutes)
            nxt, skipped, why = order[0], [], "the loop, nothing due ahead"
            due = [j for j in options if options[j][4] > 0 and options[j][1] >= SKIP_FULL * options[j][4]]
            most = max(due, key=lambda j: (options[j][3], -options[j][0]), default=None)
            if most is not None and options[most][3] > options[nxt][3]:
                nxt, why = most, "nothing due on the loop: the most expected"
        ahead = nxt  # the loop's next stop: where a pile's detour goes on to
        to_pile = heading is not None and pulled is not None and heading == pulled[0]  # a pile of the loop's, pulled to
        if heading in options:
            nxt, skipped, why = heading, [], "on her way"
            if heading in loop and not to_pile:
                ahead = heading
        rate, far_ahead = self.plan.rate, options[ahead][0]

        def value(j: int) -> tuple[float, float] | None:
            """Stop j's EXP less its touches' cost (net), and its detour's s on to the loop's next (None: no way)."""
            d = self.detour(j, ahead, options[j][0], far_ahead, options[j][1])
            return None if d is None else (self.net(j, options[j][1]), d)
        keep = None  # on her way to a pile: what it leaves her over the loop's, which another must beat, with its margin
        if nxt != ahead and (v := value(nxt)) is not None:
            keep = v[0] - max(0.0, rate) * v[1]
        best = None  # (EXP over what the loop gives meanwhile, the stop, its detour's s)
        for j, (far, exp, seen, _, _) in options.items():
            if rate <= 0:  # the loop's worth nothing by the belief: no bar to beat, she keeps to it
                break
            if j in (nxt, ahead) or seen < PULL_MIN * self.min_exp or (v := value(j)) is None:
                continue
            if not pays(v[0], v[1], rate) or (keep is not None and v[0] - PULL_GAIN * rate * v[1] <= keep):
                continue
            if best is None or v[0] - rate * v[1] > best[0]:
                best = (v[0] - rate * v[1], j, v[1])
        pulled_from = dropped = None
        if best is not None:
            pulled_from, nxt, why = nxt, best[1], "pulled"
        elif keep is not None and keep < 0 and ahead in options:  # the pile she's going to no longer pays for its
            # detour even without the margin (shot in passing, gone, taken): on to the loop's next stop; pulled again
            # only with the margin, so no turning back
            dropped, nxt, why = nxt, ahead, "back to the loop"
        if not peek:  # the loop goes on from its next stop after a pile (the detour's cost), else from the stop chosen
            self.at_loop = at_loop
            if pulled_from is not None:
                self.pulled = None
                if ahead in loop:
                    self.at_loop, self.pulled = (loop.index(ahead) - 1) % len(loop), (nxt, ahead)
            elif not (to_pile and nxt == heading):  # (on her way to a pile still: both kept)
                self.pulled = None
                if nxt in loop:
                    self.at_loop = loop.index(nxt)
        far, exp, seen, gain, _ = options[nxt]
        info = {"why": why, "stop": self.stops[nxt].platform, "x": self.stops[nxt].x, "travel": round(far, 1),
                "expected": round(exp, 1), "seen": round(seen, 1), "gain": round(gain, 2),
                "skipped": [{"stop": self.stops[j].platform, "expected": round(options[j][1], 1),
                             "due": None if (d := self.due(j, t)) is None else round(d, 1)} for j in skipped]}
        if pulled_from is not None:
            info["instead_of"] = {"stop": self.stops[pulled_from].platform, "gain": round(options[pulled_from][3], 2)}
            info["detour"], info["loop_rate"] = round(best[2], 1), round(rate, 2)
            info["ahead"] = self.stops[ahead].platform
        if dropped is not None:
            info["dropped"] = {"stop": self.stops[dropped].platform, "expected": round(options[dropped][1], 1),
                               "over": round(keep, 1)}
        return nxt, info


def choice_text(info: dict) -> str:
    """A next stop's choice (Patrol.choose's info) for the log."""
    out = (f"stop {info['stop']} at x {info['x']:.0f}, {info['travel']:.1f} s away: {info['expected']:.0f} EXP "
           f"expected ({info['seen']:.0f} seen), {info['gain']:.1f} EXP/s with its kills; {info['why']}")
    if "instead_of" in info:
        o = info["instead_of"]
        out += (f" (not stop {o['stop']}: a {info['detour']:.1f} s detour on the way to stop {info['ahead']}, the loop's "
                f"{info['loop_rate']:.1f} EXP/s giving {info['detour'] * info['loop_rate']:.0f} EXP meanwhile)")
    if "dropped" in info:
        o = info["dropped"]
        out += (f" (stop {o['stop']}'s pile no longer pays for its detour: {o['expected']:.0f} EXP expected, "
                f"{-o['over']:.0f} under what the loop gives meanwhile)")
    if info.get("skipped"):
        out += "; skipped " + ", ".join(f"{s['stop']} ({s['expected']:.0f} EXP"
                                         + ("" if s["due"] is None else f", due in {s['due']:.0f} s") + ")"
                                         for s in info["skipped"])
    return out
