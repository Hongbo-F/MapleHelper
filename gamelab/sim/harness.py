"""Results of simulated hunts, and A/B comparisons of hunting policies on the same seeds.

    python -m gamelab.sim.harness run POLICY [--seeds 10] [--minutes 20] [--preset game] [--jobs 4] [--save NAME]
                                             [--costs coded|measured|measured-1366] [--ways N] [--map 10004021]
                                             [--world zero-walls ...] [--grounds 23,...]
                                             [--attack choice|skill|regular] [--think K | --think-clock K]
                                             [--character magician] [--cache]
    python -m gamelab.sim.harness cached POLICY [run's options] [--sha master]   a commit's cached seeds, read
    python -m gamelab.sim.harness ab POLICY [run's options] [--base master] [--metric KEY ...] [--max-seeds 48]
    python -m gamelab.sim.harness compare BASELINE CANDIDATE    two saved runs, seed by seed, against the bar
    python -m gamelab.sim.harness spot RUN [--seeds 3,9]        a saved run's seeds again here: the same to the digit?
    python -m gamelab.sim.harness check                         the bookkeeping agrees with every scenario's world

POLICY: tour, loop, loop-grind (the hunting commands), or walk (movement only, as the live QA walk: _walk).

`--character magician` (from 2026-11-02; the thief by default, or GAMELAB_CHARACTER's) plays the magician: the run
restarts itself with GAMELAB_CHARACTER=magician (her stack: her profile's stats, keys and Magic Claw, attack.magic's
search for every decision, her Shape), the world her as the recordings of 2026-11-02 measured her (sim.scenarios.HERS:
world.MAGE_32, Magic Claw as the game casts it, her walk 148.2 where her profile plans 125) on Tree Dungeon alone, her
damage book as her stack had it then (sim.scenarios.HER_BOOK), `--attack` her profile's (Magic Claw whenever her MP pays,
whatever is asked: ATTACK_HER is the thief's); her runs' results say `character`, and their baselines are cached under
her own key (sim.baseline: the GAMELAB_* environment as resolved). README: The simulator's magician.

`--cache` reads the seeds this checkout's code (its commit's) has already run with the same arguments from the baseline
cache (sim.baseline: D:/MapleHelper-scratch/baselines/harness/<code>/<key>/seed-NNN.json) and runs only the others,
keeping them; a checkout whose code isn't as committed, or a run with `--think` or `--think-clock`, runs them all and
keeps none. `cached` reads a commit's (master's) without running anything, and saves them as a run for `compare`,
which also takes a cache's folder. `ab` runs this checkout against a base commit's cached seeds, 12, then 24 and 48
only while a headline difference (EXP a minute, touches, deaths, falls out of the map, stops, and each --metric) is
within 2 standard errors and not zero (seeds that differ with a mean difference of 0 among them), or has under 2 seeds
with a value on both sides; a --metric no seed's result has is refused (the base's missing seeds run in its own
checkout by D:/MapleHelper-scratch/baselines/base.py, with the environment its key resolved). `spot` is a reviewer's
check of a saved run (or a cache's folder): a seed or two run again with the arguments each result records, the same
to the digit or the keys that differ, instead of the whole table again. README: Checking a change.

A run hunts for a fixed time (the horizon) from one of its map's START_AT, on a world seed, with the policy's own
random choices fixed (POLICY_SEED), so two policies meet the same monsters until their kills differ. On the first
map by default; `--map 10004021` on West Rocky Mountain II, `--map 10002073` on Tree Dungeon, Forest Up North II, with
her as she plays each (sim.scenarios.HER: level 23 and 24, her pet drinking her potions), saved as
POLICY-PRESET-MAP; the presets are the same worlds on every map, but `crowd`, Tree Dungeon's other players
(TREE_CAMPS), which is that map's alone. `--world` switches on what isn't known of the game
there (WORLDS: group 0's walls, the client's sides, the client's rope reach, a touch as a share of PADamage, no
evasion, the monsters' HP bars read empty), each saved as a suffix of the name. `--attack` has the stack choose her
attack a throw as live (behavior.attacks: Lucky Seven but the regular attack to finish a monster and while her MP is
under a cast; Lucky Seven whenever she has the MP; the regular attack always) in a world where her Lucky Seven key
casts it at its MP (ATTACK_HER). Her looks and glances are the feed's (SyncFeed), as live takes them from the
perception service; its costs are localization.feed.COSTS.
What a run gave is read from the simulated world, not from the bot's own reports (those miss kills):

  exp_per_min      EXP she got, per minute of the horizon (a death stops the count)
  kills, rival     her kills, and another player's (Realism.rival_every)
  hp_lost, hp_low  HP lost to touches (each as the world took it: by the monster's touch damage), and
                   her lowest HP; potions; died_at
  casts, mp_used   her Lucky Seven casts and the MP they took (world.CAST_MP each), mp_per_min a minute alive;
                   mp_potions, the MP potions her pet drank for her (world.Her.pet_mp); no_mp, casts pressed with too
                   little MP (nothing thrown); casts_nothing, her casts whose stars hit nothing (MP thrown at
                   nothing: none picked, a dead one, or stopped by terrain); attacks, her attacks with either key;
                   finishes, the regular attack
                   thrown to finish a monster her throws or its bar make low (`--attack choice`: its 'attack'
                   records), and mp_throws, thrown for her MP (as she reckons it, under a cast); mp_per_kill, the
                   MP a kill of hers, and mp_potions_per_min; at_dead, her attacks chosen at a monster dead by then
                   (DEAD_BEFORE_S, DEAD_NEAR: its 'attack' record's target against the world's deaths), casts_at_dead
                   those of them Lucky Seven; by the world's own monster (_at_dead_ids): at_dead_hers,
                   at_dead_others, casts_at_dead_id, and live_taken, the feed's tracks her ledger took for dead that
                   were alive; dropped, the throws dropped at their last look before the key for a
                   kill come in (behavior.hunter: 'dropped' events)
  fell_out_at      when she fell below the map's lowest floor (world.SimFellOut): the run ends there, and
                   counts against the bar as a death does; left_map_at, the same for up held at a portal
                   (world.SimLeftMap)
  touches, hurt    monsters touching her, and those of the touches she didn't evade; misses, those she did (her
                   MISS: no HP, no knock), and touches_told, those the stack was told of (the hunter's 'touch'
                   events: the feed's, from her popups, sim.fakes.FakeTouches, as live from 2026-10-28) and
                   misses_told, the MISSes among them
  trip_s           the seconds she was on the move (from a chain's or a lone move's start, 'chain' or 'move' records,
                   to its 'moved' record), and trip_hurt, trip_hp_lost: the touches that hurt her then, and the HP
                   they took (what her trips cost, against her hunting where she stands)
  retouch_2s       her touches (a MISS too) that came RETOUCH_S (2 s) or less after the one before: a touch again as her
                   invulnerability ends (live in walk B, 2026-10-28: 53 of 148, 34 at 1.43-1.6 s)
  trips            her trips (the navigator's 'trip' records): total, and arrived ('arrived' or 'in position');
                   arrive_share, the share that got there; trip_s_per_arrival, trip_s over the arrivals; touches_by, her
                   touches (a MISS too) by what she was doing: 'chain' (a chain's plan running), 'move' (a move the
                   mover made alone: control.mover), 'off' (neither: her looks, waits, fights, between trips);
                   mover_moves, her moves the mover made alone ('move' records); evades, the navigator's evade steps
                   by kind ('evade' events: control.navigator's evade step, its passes, its contact rule). The `walk`
                   policy's (movement only, as the live QA walk) above all
  drops_share      of her drops whose fate is known by the end (picked up, taken by others: `--world drop-takers`,
                   or lain their whole life), the share she picked up; value_share, the same weighed by what each is
                   worth on the world's map (behavior.looting.worth_for: the game data's prices, mesos by what its
                   monsters drop, or by their amount where the world drew one: `--world map-drops`), and
                   value_per_min, the mesos' worth she picked up a minute: mesos_per_min of it her mesos',
                   items_per_min her items' (items_picked of them); drops_left_per_min, the worth of her drops she
                   left (taken by others, or lain their whole life); cash_per_min, value_per_min less her potions at
                   the user's reference prices (behavior.hunt: HP_PER_MESO HP and MP_PER_MESO MP a meso, by what each
                   of hers gives: the simulator's Orange Potion 125 mesos, Meat 80 with `--world meat`, its Blue
                   100); pick_age, the median age of her drops as she picked them up (s from their landing);
                   drops_expired, those of hers that lay their whole life
  pet_picks        of her drops taken by her, those her pets took (`--world pet-loot`: world.Realism.pets), pet_share
                   of them, pet_value_per_min their worth a minute and her_value_per_min her own key's (value_per_min
                   both); taken_from, where she was when her drops were taken (units from them on their platform,
                   'off' on another or in the air); approach, of her drops she came within 30 ... 300 of without
                   standing on them, those taken within 8 s of it: the tables her pets were fitted by (world.PET_SPEED)
  longest_idle     the longest stretch without a kill or a pick-up, and stuck_s: the time beyond 60 s
                   in such stretches
  safety_stops     the safety stop (behavior.safety: her HP under her reserve, her profile's hits of the map's largest
                   touch in HP (the thief's 3 for now), on two looks or glances 0.25 s apart by her HP number, which
                   the simulator draws (0.6 s by her bar), or under one touch two in a row; until
                   2026-10-28 65% for 2 s, or 25% two looks in a row: her pet's potions not keeping her up): 1 if it
                   ended the run (her retreat to a safe place, then nothing: the
                   hunt ends, as live; the run's time ends with her retreat, alive_s and the idle stretches with it,
                   and it counts against the bar as a death does), and
                   safety_at, when (s; the 'safety_stop' event). Until 2026-10-26 the runs counted her rests on the
                   chair (rests) and the HP lost on her trips to rest (rest_hp_lost, rest_trips); darts_stops and
                   darts_at, out of darts' (behavior.darts: her Lucky Seven presses taking no MP and killing nothing),
                   and darts_out_at, when she ran out (world.Realism.darts: her stars; None: endless, or not out)
  alive_s          the seconds of the horizon she was hunting (to her death, fall, portal or a stop): rates a minute
                   alive
  fresh_hurt       touches that hurt her by a monster within world.FRESH_S of its respawn (from its spawn point),
                   and fresh_hp_lost, the HP they took
  wait_s           seconds the hunt loop spent waiting for respawns (its rules 'wait' and 'lull loot', from its 'hunt'
                   records), and wait_hp_lost, the HP touches took then
  pile_s           seconds the hunt loop spent on her drops (its loot rules, PILE_RULES, or an objective that's a pile,
                   the trips there included: its 'hunt' records), and pile_hurt, pile_hp_lost: the touches that hurt
                   her then and the HP they took (live on 2026-10-25 all 12 of the Horny Mushrooms' touches, 56% of the
                   HP lost, came on stays on Tree Dungeon's 23 for drops); stand_hurt, stand_hp_lost: those off her
                   trips (trip_s), standing, fighting or waiting
  knock_falls      her knocks off a platform onto one more than world.KNOCK_FALL_DY lower (or out of the map), and
                   knock_fall_dy, how far down they took her in all (World.knock_log)
  pet_potions      of her HP potions (potions), those her pet drank (Her.pet: under its warning, world.PET_HP on Tree
                   Dungeon); her MP items are mp_potions
  ropes            how often the hunt loop went to wait on a rope (its rule 9: telemetry events kind 'rope')
  rope_top         at a rope's top: her presses the game refused there that started out of darts' stretch again
                   ('refused'), moves her dot never moved on ('rope_top': control.navigator._rope_top) and her climbs
                   on after a touch ('climb_on': navigator.CLIMB_ON), telemetry events of those kinds; refused, her
                   attacks the world refused on a rope
  backstops        how often each backstop cap fired (telemetry events kind 'backstop')
  nav              her trips, from the stack's log (`navigation`): plans cut short by a hit and off the planned
                   path, replans, misses (landed elsewhere than planned) and falls among them, and her glances; of the
                   cuts for a hit, those for a touch (cut_touch: a MISS too, from 2026-10-28) and for a MISS (cut_miss);
                   chains not sent for her invulnerability spent planning (immune_spent: navigator.IMMUNE_RECHECK)
  coverage         her moves, platforms and plans, for the coverage report (analysis.coverage.tally: what it
                   keeps of the run's telemetry, 3-7 KB a run as saved)
  turning          her throws and jump throws, those that tapped an arrow to turn her first (control.actions: 80 ms
                   more a throw, 205 a jump throw, and a few units walked), her attacks that went the other way from
                   where the stack threw them (world.World.wrong_way), and the frames that told the other way from
                   her last arrow (the hunter's 'heading' events): `--world knock-turns` has a knock turn her,
                   `heading-wrong` and `heading-wrong-sure` her frames read the wrong way
  landings         her landings that moved her (onto another platform, or LANDING_MOVED along hers: World.land_log),
                   by cause ('trip': her trips' legs and falls, 'knock', 'knock-fall': a fall off an end right after a
                   knock's landing, 'fight': an attack played in the flight), and
                   of them those with a live monster on the platform she landed on within 105 and 60 of her, middle to
                   middle ('near', 'close'), and within 105 walking at her ('at'): the live diagnosis's count; and
                   landing_events, her trips' last landings with a price (control.navigator._clear_landing: the monster
                   'shot first', 'waited' out, or 'went' past); moves_away, her stances that moved her away ('back': a
                   step, 'jump_back'; 'away': the steps from a close monster of the 'back', a stack's fights that tell),
                   once a run of looks deciding it

Saved runs go to gamelab/lab/harness/<name>.json (git-ignored, like all game data). Runs are made with
PYTHONHASHSEED=0 and her stats as measured (physics.FIXED_STATS), as golden traces are.
"""

import argparse
import bisect
import dataclasses
import difflib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

from ..analysis import coverage
from ..behavior.hunt import HP_PER_MESO, MP_PER_MESO
from ..behavior.looting import LOOT_REACH, worth_for
from ..paths import LAB, REPO
from .world import (HER_BOX_CLIENT, HER_HALF, HER_HEIGHT, INVULNERABLE_LIVE, KNOCK_PLANNER, LUCKY_SEVEN, MAP_DROPS,
                    MEAT_HP, NEAR_SEEN, OFF_SEEN, POTIONS, ROPE_REACH_CLIENT, SPAWN_FOUND, SPAWN_FOUND_WIDE, TICK_S,
                    TREE_AIR, TREE_DROP_MIX, Camp, Noise, Realism, tab_of)

FOLDER = LAB / "harness"
POLICY_SEED = 7   # the policy's own random choices (route weights, sweep order): the same in every run
FIRST_MAP = 10000012  # (sim.scenarios.MAP_ID)
START_AT = [(36, 300.0), (32, 600.0), (16, 585.0), (14, 760.0), (29, 420.0), (12, -300.0), (21, -300.0),
            (24, 120.0), (4, 450.0), (1, 1200.0)]  # where seed k starts: START_AT[k % len]
# ... on West Rocky Mountain II: its hunting grounds (the bridge 22, the high bridge 14, 17, 19, 10, the ledges 7, 6
# and 9), the floating step 24 under the bridge and stair 30 in the pit.
ROCKY_START_AT = [(22, -700.0), (14, -930.0), (17, -277.5), (19, -700.0), (10, -200.0), (7, -500.0), (30, -1009.0),
                  (24, -946.0), (6, -300.0), (9, -800.0)]
# ... on Tree Dungeon, Forest Up North II: her spot by the portal on 28, the study's best stands (23 at 73, 8 at 137,
# 20 at -95, 19 at 130, 15 at -110, 10 at -103), the mid portal's shelf 12, the floor 33 and 30.
TREE = 10002073
TREE_START_AT = [(28, 247.0), (23, 73.0), (8, 137.0), (20, -95.0), (19, 130.0), (12, 160.0), (33, 100.0), (15, -110.0),
                 (10, -103.0), (30, 60.0)]
STARTS = {FIRST_MAP: START_AT, 10004021: ROCKY_START_AT, TREE: TREE_START_AT}
# Worlds to compare in. "plain" is the simulator's own (every spawn filled, 8 s respawns, nobody else,
# perfect eyes). "game" adds the game's errors and a less crowded, contested map; its values are guesses
# (nothing about respawns, other players or potions is measured), chosen to be harder than "plain"; and arrows
# pressed right after an attack's key swallowed, as measured in game (world.ARROW_LOCK).
_GAME_NOISE = Noise(miss=0.275, walk_jitter=6.0, lag_sd=1.0, tap_lost=0.6, dot_px=True, hit_hidden=(1.0, 3.0),
                    detect_miss=0.1, heading_unsure=0.014)
PRESETS = {
    "plain": (Noise(), Realism()),
    "game": (_GAME_NOISE, Realism(spawn_share=0.6, rival_every=25.0, potions=60, lock_arrows=True)),
    # "game" with the drops seen in game (world.GAME_DROPS: a slime's Slime Bubble one kill in 10, and its other
    # monsters' items; on another map its monsters', world.game_drops) instead of mesos and a pig's leather every time:
    # the hunt loop's valuable piles (hunt.VALUABLE)
    "game-drops": (_GAME_NOISE, Realism(spawn_share=0.6, rival_every=25.0, potions=60, lock_arrows=True,
                                        drops=MAP_DROPS)),
    # "game" with jump throws missing a snail close by as often as in game (world.Noise.air_snail): what
    # stance.JUMP_THROW_SNAILS saves, which "game"'s jump throws, taking every snail their area picks, can't show
    "game-air": (dataclasses.replace(_GAME_NOISE, air_snail=(46.0, 0.35)),
                 Realism(spawn_share=0.6, rival_every=25.0, potions=60, lock_arrows=True)),
    # "game" with 15% of the spawns filled, not 60%: few monsters in the basin, so the trips aren't spent fighting
    # there (in "game" a quarter of the tour's trips time out, 28 of 246 before she set off: analysis.coverage), and
    # what's left to decide where she goes is the routes and the grounds chosen. The coverage test's world
    # (tests/test_coverage.py: every platform monsters spawn on reached).
    "game-sparse": (_GAME_NOISE, Realism(spawn_share=0.15, rival_every=25.0, potions=60, lock_arrows=True)),
}
# The other players on Tree Dungeon, Forest Up North II as the crowding study saw them from 28 (world.Camp; D:/
# MapleHelper-scratch/map3/study/crowding.md, 2026-10-24): Fuiyoohhhh by the portal on 28 killing 29's and 30's Greens
# (one every 7.1 s), BaileyBlake on 21 and siwooda by 22 killing 21's and 22's, TopAce on 23 (whose Hornies lived on), a
# player on 19 and Grasje on 32/33 (Greens and, unseen, Slimes). Five at once (3-6 were in view, mean 3.9); the upper
# tower (6-15), out of the minimap's view from 28, has nobody: unknown.
GREEN, SLIME = "green-mushroom", "slime"
TREE_CAMPS = (Camp(28, 250.0, (29, 30), (GREEN,)), Camp(21, 90.0, (21, 22), (GREEN,)), Camp(23, 0.0, (22,), (GREEN,)),
              Camp(19, 200.0, (19,), (GREEN,)), Camp(33, 120.0, (32, 33), (GREEN, SLIME)))
# Tree Dungeon's crowd: "game"'s errors, other players' dots under hers unseen (Noise.dot_under: 56 units, the study's
# frames), the camps above instead of "game"'s one roaming rival, the map's respawn the server's (Realism.respawn_tick:
# its cap by the formula, the players there counted). Only on that map (PRESET_MAPS). Until 2026-10-31 every spawn
# point was filled and each monster came back on its own clock, 8.8 s after its death ("crowd": MeowDB's mean), 17 s
# ("crowd-17": the live median on 23 after her last kill there) or 30 s ("crowd-slow": the first map's slow refill):
# with the server's respawn those three are one, "crowd".
# From 2026-10-27, with the air pick's noise as fitted to the live jump throws there (world.TREE_AIR: each monster's top
# uniform over its height for it, a point-blank target taking the star up to 30% of the time the nearer the first band,
# one under its window 30%, an air turn lost 26%), the world picking from a jump by the star's search (world.AIR_PICK
# 'search'), so that its jump throws hit and miss where live ones did, by distance and by the slope.
_CROWD_NOISE = dataclasses.replace(_GAME_NOISE, dot_under=56.0, **TREE_AIR)
PRESETS["crowd"] = (_CROWD_NOISE, Realism(camps=TREE_CAMPS, potions=60, lock_arrows=True))
PRESET_MAPS = {"crowd": (TREE,)}
# What isn't known of the game, switched on with `--world` (each a world.Realism field, or her own: world.Her):
# group 0's walls stopping her whatever her group, the map's sides as HeavenClient has them, a rope grabbed as it does
# (82 up at most), a touch's HP as a share of PADamage (as fitted on the first map) rather than meowdb's formula, and
# no evasion at all (the study: at EVA 42 she'd evade no Horny Mushroom), and her potions Red Potions (100 HP: which
# she carries isn't known; world.POTION_HP's 30% otherwise, 226 HP on Tree Dungeon). And how she plays: `lucky-seven`,
# her attacks as Lucky Seven's live (`tour --loop --skill`: world.LUCKY_SEVEN, a Horny Mushroom's kill in about 4
# throws, not 12), the stack unchanged (the simulator acts on its key as on the attack's).
WORLDS = {"zero-walls": {"zero_walls": True}, "joined-walls": {"zero_walls": False}, "client-sides": {"sides": "client"},
          "client-reach": {"rope_reach": ROPE_REACH_CLIENT}, "touch-share": {"defence": None},
          "no-evade": {"evade": 1e-9}, "red-potion": {"potion": 100.0}, "lucky-seven": {"stars": LUCKY_SEVEN},
          "knock-turns": {"knock_turns": True}}
HER_FIELDS = ("defence", "evade", "potion", "stars", "walk_speed")
IDLE_GRACE = 60.0  # s without a kill or a pick-up that still counts as hunting
# ... and her body for touches as the planner has it (`client-body`: 22 either side of her feet and 65 tall,
# motion.bodies, which the planner keeps her clear of monsters by), or as the simulator had it until 2026-11-02
# (`wide-body`: 15 either side and 55 tall, world.HER_HALF, HER_HEIGHT), not the client's touch test (world.HER_TOUCH: a
# line 50 tall over her feet, every world's since).
WORLDS["client-body"] = {"her_box": HER_BOX_CLIENT}
WORLDS["wide-body"] = {"her_box": (HER_HALF, HER_HEIGHT)}
# ... and the monsters' HP bars read empty once they're hurt (`bars-empty`: Realism.bars_empty, sim.fakes.FakeEyes.bar),
# as the stack read them live at 1366x768 until 2026-10-25 (perception.hud.hp_bar took the bar's black frame for its
# fill: 488 of 567 bars read 0), so what a monster has left was her throws' reckoning alone (behavior.attacks).
WORLDS["bars-empty"] = {"bars_empty": True}
# ... and a touch knocking her off a rope (`rope-knocks`: Realism.rope_knocks), as the simulator had it until
# 2026-10-27; the game keeps her on it (11 of 11 hits live on 2026-10-26), and so does the world since.
WORLDS["rope-knocks"] = {"rope_knocks": True}
# ... and her facing read the wrong way off 1% of her frames, in runs of one pose (world.Noise.heading_wrong): as weakly
# as every frame read the wrong way on 2026-10-24 (0.31, 0.24 ahead: never strong, which her throws' taps go by), and,
# the worst case, as surely as she's read right (0.6, 0.4 ahead: strong).
WORLDS["heading-wrong"] = {"heading_wrong": (0.01, 0.31, 0.24, True)}
WORLDS["heading-wrong-sure"] = {"heading_wrong": (0.01, 0.6, 0.4, True)}
# ... and the detector's misses as live (`near-misses`: world.NEAR_SEEN, OFF_SEEN; Noise.near_seen, off_seen): a monster
# on her platform within 25 units of her never found, 7% of looks within 50, 34% within 100, 55% within 150, 60%
# farther; one on another platform 70% (the live hunt of 2026-10-25), where the presets find it 90% of the time wherever
# it is.
WORLDS["near-misses"] = {"near_seen": NEAR_SEEN, "off_seen": OFF_SEEN}
# ... and the monsters the map spawns fading in, seen as late as live sees them (`spawn-fade`: world.SPAWN_FOUND, the
# frame step's spawn watch on, as from 2026-11-01; Noise.spawn_found) or as the whole screen alone saw them before
# (`spawn-fade-wide`: SPAWN_FOUND_WIDE, a median 0.53 s later), where every other world finds a newborn from its birth.
WORLDS["spawn-fade"] = {"spawn_found": SPAWN_FOUND}
WORLDS["spawn-fade-wide"] = {"spawn_found": SPAWN_FOUND_WIDE}
# ... and a newborn touching her from its birth (`newborns-touch`: Realism.newborns_touch), as every world had it until
# 2026-11-02, where it now touches only once it has faded in (world.NEWBORN_SOLID_S: none of the 72 live touches by a
# spawn point came before 0.82 s after the respawn pass).
WORLDS["newborns-touch"] = {"newborns_touch": True}
# ... and her knock as the planner has it (`planner-knock`: Realism.knock, world.KNOCK_PLANNER: motion.physics's 180 out
# and 220 up, landing at the speed she flew; ~58 units on flat ground, 0.22 s in the air), as every world had it until
# 2026-11-02, where it is now the client's (world.KNOCK_VX, KNOCK_VY: 270 and 270, landing at her walk's speed at most;
# ~87, 0.27 s, as R1-R3's 15 fps frames have it).
WORLDS["planner-knock"] = {"knock": KNOCK_PLANNER}
# ... and her invulnerability after a touch as live (`walkers`: INVULNERABLE_LIVE; Realism.invulnerable), with the
# monsters she hits chasing her (Realism.chase, world.AGGRO_S: every world's since 2026-10-31; till then this world's
# alone, by chance, CHASE_TREE; and each monster back 17 s after its own death, RESPAWN_LIVE, until the server's respawn
# became the default the same day). `no-aggro`: the monsters walk as with nobody there, hit or not (as every world but
# `walkers` had them till 2026-10-31).
WORLDS["walkers"] = {"chase": True, "invulnerable": INVULNERABLE_LIVE}
WORLDS["no-aggro"] = {"chase": False}
# The game's respawn is every world's from 2026-10-31 (Realism.respawn_tick: world.TICK_S, SPAWN_CLEAR,
# MapModel.mob_cap): `tick-respawn` is that, kept for the runs saved under its name; `clock-respawn` each monster back
# on its own clock, `respawn` s after it died, as every world had it until then; and with nobody else on the map
# (`solo`: no camps, no rival), as live on 2026-10-29 16:35-16:45.
WORLDS["tick-respawn"] = {"respawn_tick": TICK_S}
WORLDS["clock-respawn"] = {"respawn_tick": 0.0}
WORLDS["solo"] = {"camps": (), "rival_every": 0.0}
# ... and her drops as the perception service saw them live (the loot design, 2026-10-30: D:/MapleHelper-scratch/replan/
# loot): what a kill drops fitted to its sightings (`map-drops`: world.TREE_DROP_MIX, Realism.drop_mix: 0.64 drops a
# kill, 57% mesos, by its kind's amounts, piles of 1-4, 20 units apart on alternating sides), where the default world drops
# one coin a kill; and her drops taken by others (`drop-takers`: Realism.drop_takers, world.others_life: once past her
# ownership, Realism.rival_owns, at world.TAKE_RATE a minute, 0.05: 13% of those she leaves gone by 180 s), where the
# default world leaves them 180 s. Until 2026-10-31 `drop-takers` drew each life from the live service's raw curve (55%
# still there at 60 s, 26% at 120), which counted her pets' pick-ups and the detector's misses as takings.
WORLDS["map-drops"] = {"drop_mix": TREE_DROP_MIX}
WORLDS["drop-takers"] = {"drop_takers": True}
# ... and her inventory (the user, 2026-10-31: "we also need to watch for backpack full scenario"): her Etc tab full
# from the start (`etc-full`), or once she has picked up 6 items into it (`fills-up`: a guess, about a minute of the
# live drops' items), where the default world has room for everything; with `map-drops` for items to fill it.
WORLDS["etc-full"] = {"full_tabs": ("Etc",)}
WORLDS["fills-up"] = {"fills_after": (("Etc", 6),)}
# ... and (from the review of 2026-10-31, nothing had ever made room again) her full tabs with room again at 3 minutes
# (`room-later`: she sold or dropped things, as whoever is at the PC might), with `etc-full` or `fills-up`; in v83 a full
# tab still takes more of a kind she holds a part-filled stack of (Realism.fills_after), in every world.
WORLDS["room-later"] = {"room_at": 180.0}
# ... and the player started before splices (`no-splices`: Realism.splices, sim.fakes.FakePlayer.splices): every chain
# replanned while one runs (control.navigator.REPLAN) a cut and a resend, every key let go between, as live until the
# elevated player is restarted on this code (control.client.can_splice: lab/player.json); the review of 2026-10-30
# found the path exercised by nothing.
WORLDS["no-splices"] = {"splices": False}
# ... and her pet's potions as she carries them on a grind (`meat`: the user, 2026-11-01): Meat (world.MEAT_HP, 160 HP,
# 80 mesos at their prices) from a stock of world.POTIONS (250) a run, where the crowd preset has 60 Orange Potions
# (250 HP), which ran out within 15 minutes.
WORLDS["meat"] = {"potion": MEAT_HP, "potions": POTIONS}
# ... and her two pets taking what they pass as they trail her (`pet-loot`: Realism.pets, world.PET_SPEED ...; the user,
# 2026-11-01: they "will pick up whatever thing when the pet walks past those loot", never going for it), where the
# other worlds' drops are hers to pick up alone: live, 26-29% of the drops taken on her platform went with her 45 units
# or more off, or with her never within 15 of them.
WORLDS["pet-loot"] = {"pets": 2}
# ... and another player's kill of a monster she hurt giving her its EXP times the share of its HP she took
# (`shared-exp`: world.Realism.shared_exp), as live on 2026-11-02 (the hour's 26 shares by the log, mostly +32-35 of a Horny
# Mushroom's 41 after her first claw: the rule's 32.8-36.5; plans/problems.md P43), where the other worlds give nothing.
WORLDS["shared-exp"] = {"shared_exp": True}
# ... and her walk in the world as her planner has it (`walk-planned`: world.Her.walk_speed None, physics.SHOE), where her own
# world walks her at her measured speed (the magician's 148.2 against her profile's 125: world.MAGE_32): what the
# planner's speed error costs her (the roadmap's item 1, P01). The thief's world is the planner's already.
WORLDS["walk-planned"] = {"walk_speed": None}
NOISE_FIELDS = ("heading_wrong", "near_seen", "off_seen", "spawn_found")  # WORLDS' fields that are world.Noise's
# Her with `--attack` (any mode, so the three compare in one world): her Lucky Seven key's attack Lucky Seven as thrown
# live (world.LUCKY_SEVEN; her own stars where the map's Her has them, Her.star and skill_star: LEVEL_25's), her cast's
# MP (Her.cast_mp: LEVEL_25's 14 of 623) and nothing with less, her attack
# key's the regular star; each star's damage drawn against the monster's HP in the game data (world.Her.spread), so a
# monster's HP left is as the stack reckons it (attack.choice: its finishing throws).
ATTACK_HER = {"skill": LUCKY_SEVEN, "spread": True}
# What the simulator's looks take (sim.fakes), to compare against: "coded", the fakes' own (a look 100 ms, as
# the README had them before the speedups of 2026-10-20); "measured", today's, as bench-look and bench-capture
# measured them offline (17 saved frames x 5, medians, after the profiling pass): a look placing her 12.0 ms of
# computing plus the wait for a frame (5.6 ms median with the screen changing) and its copy off the card and
# conversion (2.0), her band 23.8 ms more, the whole screen 83.8, a strip of drops about 8 (her platform's
# 15.6 ms in 1.9 strips). Before the profiling pass (as step 5 measured them): 0.024, 0.006, 0.032, 0.122 and
# 0.008. Glances keep their pace (Eyes.GLANCE_EVERY_S). Her looks and glances come from the feed (SyncFeed), whose
# own costs are localization.feed.COSTS: these are what a look taken by the fakes costs (Hunter's on FakeEyes
# without a feed, the tests'), and the pick-up log's OCR (fakes.MESSAGES_S).
# `measured-1366`: those, and the feed's own as measured on Tree Dungeon's 1366x768 frames with the masked match in C
# and a frame's re-finds at once (localization.feed.COSTS_1366, its "feed" here: COSTS stays the scenarios'). Since
# 2026-10-30 the feed's COSTS are the one pass's at 1366x768, the live size, and COSTS_1366 is COSTS: `measured` and
# `measured-1366` run alike (the name kept for master's cached tables).
COSTS = {
    "coded": {},
    "measured": {"LOOK_S": 0.020, "LOOK_LAG": 0.006, "DETECT_S": 0.024, "DETECT_WIDE_S": 0.084, "LOOT_S": 0.008},
}
COSTS["measured-1366"] = {**COSTS["measured"], "feed": "COSTS_1366"}


# --- policies ------------------------------------------------------------------------------------------

def _safely(h, hunt: Callable[[], None]) -> None:
    """`hunt()` as the command line runs it, in behavior.retreat.on_alarm: at the safety stop (behavior.safety) she goes
    to a safe place, and the run's hunt ends there, as live (nothing hunts after it)."""
    from ..behavior.retreat import on_alarm
    from ..behavior.safety import HuntStop
    try:
        with on_alarm(h, rng=random.Random(POLICY_SEED)):
            hunt()
    except HuntStop:  # (the safety stop, out of darts)
        pass


def _tour(sim) -> None:
    """Today's tour, as `python -m gamelab tour` runs it, started again whenever a sweep ends."""
    h = sim.hunter(potion_hp=0.45)
    tour = sim.tour(h, sim.navigator(h, fight=True, seed=POLICY_SEED), seed=POLICY_SEED)

    def hunt():
        while True:
            tour.run()
    _safely(h, hunt)


def _loop(sim) -> None:
    """The hunt loop (`python -m gamelab tour --loop`), on the same hunter, navigator and map knowledge."""
    from ..behavior.hunt import Hunt
    h = sim.hunter(potion_hp=0.45)
    tour = sim.tour(h, sim.navigator(h, fight=True, seed=POLICY_SEED), seed=POLICY_SEED)

    def hunt():
        while True:
            Hunt(tour, log=tour.log).run(minutes=600)
    _safely(h, hunt)


def _loop_grind(sim) -> None:
    """The hunt loop grinding round the map (`tour --loop --grind`: behavior.grind), on the same hunter, navigator and
    map knowledge as `loop`."""
    from ..behavior.grind import Grind
    h = sim.hunter(potion_hp=0.45)
    tour = sim.tour(h, sim.navigator(h, fight=True, seed=POLICY_SEED), seed=POLICY_SEED)

    def hunt():
        while True:
            Grind(tour, log=tour.log).run(minutes=600)
    _safely(h, hunt)


# The live QA walk's platforms on Tree Dungeon (D:/MapleHelper-scratch/qa/walk.py's PLATFORMS: 7 to 33) and its trips' time
# (behavior.walk's)
WALK_PLATFORMS = range(7, 34)
WALK_TRIP_S = 40.0


def _walk(sim) -> None:
    """Movement only, as the live QA walk of 2026-10-28 (D:/MapleHelper-scratch/qa/walk.py: random `nav` trips, no
    attack), its trips in one navigator that avoids monsters (no escort): behavior.walk's, as `nav --walk` makes them
    (to a platform and an x on it drawn at random, on Tree Dungeon one of WALK_PLATFORMS, 15 in from its ends,
    WALK_TRIP_S each), one after another until the horizon; the safety stop and the Lie Detector as in every policy
    (_safely). Her trips' own numbers: `trips`, `arrive_share`, `trip_s_per_arrival`, `touches_by`, `retouch_2s`,
    `mover_moves`, `evades` (outcomes)."""
    from ..behavior import walk
    h = sim.hunter(potion_hp=0.45)
    nav = sim.navigator(h, fight=False, seed=POLICY_SEED)
    rng = random.Random(POLICY_SEED * 1000 + sim.scenario.seed)
    _safely(h, lambda: walk.walk(nav, sim.model, rng, trip_s=WALK_TRIP_S))


# (`loop-avoid`, the trips to rest keeping clear of monsters, and `loop-hp50`, resting under half her HP, went with the
# rests on 2026-10-26.)
POLICIES = {"tour": _tour, "loop": _loop, "loop-grind": _loop_grind, "walk": _walk}


def _only(policy, grounds: tuple[int, ...]):
    """`policy`, hunting only `grounds` (`--grounds`): its tours' hunting grounds cut to those (all of them if none is
    one by the value map). One ground, as live on Tree Dungeon on 2026-10-24 (23 alone: emptied, the hunt loop left it
    for itself again, and waited there)."""
    def run(sim):
        make = sim.tour

        def tour(*a, **kw):
            t = make(*a, **kw)
            every = t.hunting_grounds
            t.hunting_grounds = lambda visit_s=None: [g for g in every(visit_s) if g in grounds] or list(grounds)
            return t
        sim.tour = tour
        return policy(sim)
    return run


# --- one run -------------------------------------------------------------------------------------------

def scenario(policy: str, seed: int, preset: str = "game", minutes: float = 20.0, think: float = 0.0,
             map_id: int = FIRST_MAP, world: tuple = (), grounds: tuple = (), attack: str | None = None,
             think_clock: float = 0.0):
    """The Scenario for one run: the policy hunting on map `map_id` from its START_AT[seed] (STARTS) in world `seed`
    until the horizon. `think`: charge the stack's own computing to the fake clock (fakes.Clock; the runs then vary a
    little), at the fakes' calls; `think_clock` in its place: so, times it, and at every read of the clock too, as it
    goes (fakes.Clock.reads: a replan's own computing can make it late, navigator.SPLICE_LATE_S); `world`: the switches
    of WORLDS on; `grounds`: the hunting grounds the policy may hunt (_only; all of them by default);
    `attack`: the stack's attack a throw (behavior.attacks.MODES), with her as ATTACK_HER (None: the regular attack,
    her as the map has her)."""
    from . import scenarios
    if map_id not in PRESET_MAPS.get(preset, (map_id,)):
        raise ValueError(f"preset {preset} is map {', '.join(map(str, PRESET_MAPS[preset]))}'s alone")
    if think and think_clock:
        raise ValueError("think or think_clock, not both")
    noise, realism = PRESETS[preset]
    her = scenarios.her_on(map_id)[1]
    for name in world:
        fields = WORLDS[name]
        realism = dataclasses.replace(realism, **{k: v for k, v in fields.items()
                                                  if k not in HER_FIELDS and k not in NOISE_FIELDS})
        her = dataclasses.replace(her, **{k: v for k, v in fields.items() if k in HER_FIELDS})
        noise = dataclasses.replace(noise, **{k: v for k, v in fields.items() if k in NOISE_FIELDS})
    if attack is not None and her.magic is None:  # (the thief's Lucky Seven: the magician's attack is her Magic Claw)
        her = dataclasses.replace(her, **ATTACK_HER)
    fn = POLICIES[policy] if not grounds else _only(POLICIES[policy], tuple(grounds))
    starts = STARTS[map_id]
    return scenarios.Scenario(f"{policy}:{preset}:{seed}" + (f":{map_id}" if map_id != FIRST_MAP else "")
                              + "".join(f":{w}" for w in world) + (f":g{'-'.join(map(str, grounds))}" if grounds else "")
                              + (f":attack-{attack}" if attack else ""),
                              fn,
                              starts[seed % len(starts)], monsters="all", seed=seed, limit=minutes * 60, noise=noise,
                              realism=realism, may_die=True, think=think_clock or think, think_reads=bool(think_clock),
                              map_id=map_id,
                              her=her if world or attack else None, attack=attack,
                              book=scenarios.HER_BOOK if her.magic is not None else None)


def run_one(policy: str, seed: int, preset: str = "game", minutes: float = 20.0, think: float = 0.0,
            costs: str = "coded", ways: int | None = None, map_id: int = FIRST_MAP,
            world: tuple = (), grounds: tuple = (), attack: str | None = None, think_clock: float = 0.0) -> dict:
    """Run one hunt and return what it gave (outcomes). `costs`: what the simulator's looks take (COSTS); `ways`:
    the ways her trips choose among (route.ROUTES; None: as it is); `map_id`: the map (STARTS); `world`: WORLDS on;
    `attack`: the stack's attack a throw; `think`, `think_clock`: the stack's computing charged to the clock (scenario;
    the result has `think_clock` only where it's set)."""
    from .. import character, telemetry
    from ..attack import magic
    from ..localization import feed
    from ..motion import route
    from . import fakes, scenarios
    if not magic.applied():  # (the magician's Magic Claw for every decision, as `tour` sets it: nothing for the thief)
        magic.apply(character.active())
    fakes_costs = {name: value for name, value in COSTS[costs].items() if name != "feed"}
    before = {name: getattr(fakes, name) for name in fakes_costs}
    routes, feed_costs = route.ROUTES, feed.COSTS
    try:
        for name, value in fakes_costs.items():
            setattr(fakes, name, value)
        if "feed" in COSTS[costs]:  # (the feed made for the run takes COSTS as it is then)
            feed.COSTS = getattr(feed, COSTS[costs]["feed"])
        if ways is not None:
            route.ROUTES = ways
        with telemetry.captured() as records:
            trace, sim = scenarios.run_sim(scenario(policy, seed, preset, minutes, think, map_id, world, grounds,
                                                    attack, think_clock))
    finally:
        for name, value in before.items():
            setattr(fakes, name, value)
        route.ROUTES, feed.COSTS = routes, feed_costs
    out = outcomes(sim.world, records, minutes * 60)
    out.update(policy=policy, seed=seed, preset=preset, minutes=minutes, think=think, feed=True, costs=costs,
               ways=routes if ways is None else ways, map=map_id, result=str(trace["result"])[:120])
    if world:
        out["world"] = list(world)
    if grounds:
        out["grounds"] = list(grounds)
    if attack:
        out["attack"] = attack
    if think_clock:
        out["think_clock"] = think_clock
    if (who := character.active_id()) != character.DEFAULT:  # (`--character`: the thief's results as they were)
        out["character"] = who
    out["coverage"] = coverage.tally(records, coverage.graph_for(sim.model))  # her moves, for the coverage report
    f = trace["feed"]
    out["feed_record"] = {"frames": f["frames"], "asked": f["asked"], "busy_s": f["busy_s"],
                          "searches": f.get("searches"), "searching_s": f.get("searching_s"),
                          "senses": f["senses"], "tracker": f["tracker"]}
    out["looks"] = f["senses"].get("looks", 0)
    out["turning"] = turning(trace["plans"], records, sim.world)
    out["attack_kinds"] = attack_kinds(sim.world, records)
    out["nav"] = navigation(trace["log"], sim.model)
    out["nav"]["glances"] = f["senses"].get("glances", 0)
    return out


def turning(plans: list, records: list[dict], world=None) -> dict:
    """Her throws and jump throws (the plans the player ran), those of them that tapped an arrow to turn her first
    (control.actions.Actions._turn), her attacks that went the other way from where the stack threw them (the
    `world`'s count), and the frames that told the other way from her last arrow (telemetry events 'heading':
    behavior.hunter)."""
    throws = [p for p in plans if p[1] in ("throw", "jump_throw")]
    turned = [p for p in throws if any(e[1] in ("left", "right") for e in p[2])
              and not (p[1] == "jump_throw" and _forward(p[2]))]  # (a forward jump's arrow carries her: no tap)
    return {"throws": len(throws), "turned": len(turned),
            "jump_turned": sum(1 for p in turned if p[1] == "jump_throw"),
            "attacks": getattr(world, "attacks_meant", None), "wrong_way": getattr(world, "wrong_way", None),
            "against": sum(1 for r in records if r.get("topic") == "event" and r.get("kind") == "heading")}


def _forward(events: list) -> bool:
    """Whether a jump throw's key plan is a forward one (control.actions.Actions.jump_throw with `way`): its arrow
    held from before its jump key until after it, not a turning tap let go before the jump nor one in the air."""
    jump = min((e[0] for e in events if e[1] in ("alt", "jump") and e[2] == "down"), default=None)
    downs = [e[0] for e in events if e[1] in ("left", "right") and e[2] == "down"]
    ups = [e[0] for e in events if e[1] in ("left", "right") and e[2] == "up"]
    return jump is not None and any(d < jump for d in downs) and any(u > jump for u in ups)


def attack_kinds(world, records: list[dict]) -> dict:
    """What each kind of throw hit, from the world's own log of her attacks (world.World.attack_log): by the key plan
    that pressed it (control.actions: "throw", "jump_throw", "jump_back"; a chain's, "flow"), how many and what came of
    each ("kill", "hit", "swipe", "miss", "stopped", "nothing": `hit_nothing` the last two), those started in the air,
    and of those the ones that picked a monster whose feet were LEVEL or more above hers then (`up`); and the jump
    throws the stack chose, by why (its telemetry 'jump' records' `kind`: attack.stance; from 2026-11-01 its jump throws
    up close too, 'close' and 'tap first', jittered as its shots are), with what came of each
    (`chosen_by`: why and the jump's kind -> outcomes, the record matched to the first jump's attack keyed within
    CHOSEN_KEY_S after it; 'unmatched' if none was)."""
    out: dict = {}
    log = getattr(world, "attack_log", [])
    for _, plan, mode, _, _, outcome, up in log:
        k = out.setdefault(plan, Counter())
        k["n"] += 1
        k[outcome] += 1
        k["air"] += mode == "air"
        k["up"] += up is not None and up >= _LEVEL
        k["hit_nothing"] += outcome in ("stopped", "nothing")
    jumps = [r for r in records if r.get("topic") == "jump"]
    chosen = Counter(r.get("kind") for r in jumps)
    from .fakes import EPOCH
    keyed = sorted((t, outcome) for t, plan, *_rest, outcome, _ in log if plan in ("jump_throw", "jump_back"))
    by: dict[str, Counter] = {}
    i = 0
    for r in sorted(jumps, key=lambda r: r["t"]):
        at = r["t"] - EPOCH
        while i < len(keyed) and keyed[i][0] < at - 1e-6:
            i += 1
        hit = keyed[i][1] if i < len(keyed) and keyed[i][0] <= at + CHOSEN_KEY_S else "unmatched"
        if hit != "unmatched":
            i += 1
        by.setdefault(f"{r.get('kind')}: {r.get('jump')}", Counter())[hit] += 1
    return {"plans": {k: dict(v) for k, v in out.items()}, "chosen": dict(chosen),
            "chosen_by": {k: dict(v) for k, v in by.items()}}


CHOSEN_KEY_S = 1.5  # (attack_kinds) a chosen jump's attack key comes this soon after its record at the latest: its key
                    # at most 0.3 s into the jump (attack.reach.TOP_KEY_MS and JITTER_TOP_MS; 0.48 until 2026-11-01),
                    # after a wait for her lock


_LEVEL = 15.0  # (attack.reach.LEVEL) a pick this far above her feet or more is off her level, above her


def navigation(log: list[str], model) -> dict:
    """How her trips went, from the stack's log (control.navigator): plans cut short by a hit ('cut_hit') and off
    the planned path ('cut_off_path'); replans from where she ended up rather than where planned (a cut, a hit on
    the way, a landing elsewhere: once each, a cut and the landing after it once); misses, the landings elsewhere
    than planned (the trip's own count); and falls, those of them onto a platform more than 10 units lower than
    the one planned, or onto none."""
    out = {"cut_hit": 0, "cut_off_path": 0, "replans": 0, "misses": 0, "falls": 0, "cut_touch": 0, "cut_miss": 0,
           "immune_spent": 0}
    cut = False  # the line before was a cut: the landing it logs next is the same replan
    for line in log:
        out["immune_spent"] += "her invulnerability spent planning" in line
        if "plan cut short" in line:
            out["cut_off_path" if "off the path" in line else "cut_hit"] += 1
            if "touched (" in line:  # (navigator._touch_text)
                out["cut_touch"] += 1
                out["cut_miss"] += "touched (MISS" in line
            out["replans"] += 1
            cut = True
            continue
        if "replanning" in line:
            out["replans"] += not cut
            if (m := _MISSED.search(line)) is not None:
                out["misses"] += 1
                planned = model.platform(int(m["planned"]))
                landed = None if m["landed"] is None else model.platform(int(m["landed"]))
                out["falls"] += landed is None or landed.top > planned.top + 10
        cut = False
    return out


_MISSED = re.compile(r"ended on (?:platform (?P<landed>\d+) \(.*?\)|air/rope) instead of platform (?P<planned>\d+)")


def outcomes(world, records: list[dict], horizon: float) -> dict:
    """What a run gave, from the simulated world (and the telemetry it recorded)."""
    events = [r for r in records if r.get("topic") == "event"]
    stops = [r for r in events if r.get("kind") == "safety_stop"]
    darts = [r for r in events if r.get("kind") == "out_of_darts"]
    end = min(horizon, world.died_at) if world.died_at is not None else horizon
    if world.fell_out_at is not None:
        end = min(end, world.fell_out_at)
    if world.left_map_at is not None:
        end = min(end, world.left_map_at)
    if stops or darts:  # (the hunt ended there, her retreat after the stop its last: nothing hunts on, as live)
        end = min(end, world.t)
    mine = [k for k in world.kill_log if k[3] == "her"]
    drops = [d for d in world.drops if d.owner == "her" and d.since <= horizon]
    known = [d for d in drops if d.taken_by in ("her", "other") or d.since + d.life <= horizon]
    picked = [d for d in known if d.taken_by == "her"]
    coin = worth_for(world.model)  # what drops are worth on the world's map (looting.Worth) ...

    def worth(d) -> float:  # ... mesos by their amount where the world drew one (Realism.drop_mix)
        return float(d.amount) if d.mesos and d.amount else coin(d.kind)
    worth_known, worth_picked = sum(worth(d) for d in known), sum(worth(d) for d in picked)
    left = [d for d in known if d.taken_by != "her"]  # (taken by others, or lain their whole life)
    ages = sorted(d.taken_at - d.since for d in picked)
    times = sorted([0.0] + [k[0] for k in mine] + [d.taken_at for d in world.drops if d.taken_by == "her"] + [end])
    gaps = [b - a for a, b in zip(times, times[1:])]
    waits = _waits(records)
    start = next((r["t"] for r in records if r.get("topic") == "session"), None)
    her = world.character  # (what her potions cost: what each gives at the user's reference prices)
    hp_potion = world._potion() * her.max_hp / HP_PER_MESO
    mp_potion = her.mp_potion / MP_PER_MESO
    return {
        "exp": world.exp,
        "exp_per_min": round(world.exp / (horizon / 60), 2),
        "kills": len(mine),
        "rival": sum(1 for k in world.kill_log if k[3] == "rival"),
        "hp_lost": round(sum(hp for _, hp in world.touch_log), 3),
        "touches": world.touches,
        "hurt": world.touches - world.evaded,
        "misses": world.evaded,
        "touches_told": sum(1 for r in events if r.get("kind") == "touch"),
        "misses_told": sum(1 for r in events if r.get("kind") == "touch" and r.get("touch") == "MISS"),
        "retouch_2s": _retouches(world, end),
        **_travelling(world, records, end),
        **_trips(world, records, end),
        "hp_low": round(world.hp_low, 3),
        "potions": world.potions_used,
        "casts": world.casts,
        "mp_used": round(world.mp_used, 1),
        "mp_per_min": round(world.mp_used / (end / 60), 2) if end else None,
        "mp_potions": world.mp_potions,
        "no_mp": world.no_mp,
        "casts_nothing": world.casts_nothing,
        "attacks": world.attacks,
        "finishes": sum(1 for r in records if r.get("topic") == "attack" and r.get("why") == "finish"),
        "mp_throws": sum(1 for r in records if r.get("topic") == "attack" and r.get("why") == "MP"),
        **_at_dead(world, records),
        **_at_dead_ids(world),
        "mp_per_kill": round(world.mp_used / len(mine), 2) if mine else None,
        "mp_potions_per_min": round(world.mp_potions / (end / 60), 3) if end else None,
        "dropped": sum(1 for r in events if r.get("kind") == "dropped"),
        "died_at": world.died_at,
        "fell_out_at": world.fell_out_at,
        "left_map_at": world.left_map_at,
        "drops": len(drops),
        "drops_share": round(len(picked) / len(known), 3) if known else None,
        "value_share": round(worth_picked / worth_known, 3) if worth_known else None,
        "value_per_min": round(sum(worth(d) for d in drops if d.taken_by == "her") / (horizon / 60), 2),
        "mesos_per_min": round(sum(worth(d) for d in drops if d.taken_by == "her" and d.mesos) / (horizon / 60), 2),
        "items_per_min": round(sum(worth(d) for d in drops if d.taken_by == "her" and not d.mesos) / (horizon / 60), 2),
        "items_picked": sum(1 for d in drops if d.taken_by == "her" and not d.mesos),
        **_pets_took(world, drops, horizon, worth),
        "drops_left_per_min": round(sum(worth(d) for d in left) / (horizon / 60), 2),
        "cash_per_min": round((sum(worth(d) for d in drops if d.taken_by == "her") - world.potions_used * hp_potion
                               - world.mp_potions * mp_potion) / (horizon / 60), 2),
        "pick_age": round(ages[len(ages) // 2], 2) if ages else None,
        "longest_idle": round(max(gaps, default=0.0), 1),
        "stuck_s": round(sum(max(0.0, g - IDLE_GRACE) for g in gaps), 1),
        "safety_stops": len(stops),
        "safety_at": round(stops[0]["t"] - start, 1) if stops and start is not None else None,
        "darts_stops": len(darts),
        "darts_at": round(darts[0]["t"] - start, 1) if darts and start is not None else None,
        "darts_out_at": None if world.darts_out_at is None else round(world.darts_out_at, 1),
        "alive_s": round(end, 1),
        "fresh_hurt": sum(1 for t, hp in world.fresh_log if hp > 0 and t <= end),
        "fresh_hp_lost": round(sum(hp for t, hp in world.fresh_log if t <= end), 3),
        "wait_s": round(sum(max(0.0, min(b, end) - a) for a, b in waits), 1),
        "wait_hp_lost": round(sum(hp for t, hp in world.touch_log if t <= end and any(a <= t <= b for a, b in waits)),
                              3),
        **_on_piles(world, records, end),
        **_full_stands(world, records, end),
        "knock_falls": sum(1 for k in world.knock_log if k[0] <= end),
        "knock_fall_dy": round(sum(k[3] for k in world.knock_log if k[0] <= end), 1),
        "pet_potions": world.pet_potions,
        "ropes": sum(1 for r in events if r.get("kind") == "rope"),
        # (a rope's top: the game refusing her presses, a move her dot never moved on, her climb on after a touch)
        "rope_top": {k: sum(1 for r in events if r.get("kind") == k) for k in ("refused", "rope_top", "climb_on")},
        "refused": getattr(world, "refused", 0),  # her attacks refused on a rope (world.ROPE_REFUSAL)
        # (up close: from 2026-10-27, clearance and slopes)
        "claw_s": round(getattr(world, "claw_ms", 0) / 1000, 1),
        "close_s": round(getattr(world, "close_ms", 0) / 1000, 1),
        "swiped_casts": getattr(world, "swiped_casts", 0),
        "air_turns_lost": getattr(world, "air_turns_lost", 0),
        "picks": picks(world),
        "landings": landings(world, end),
        # (her trips' last landings priced: the monster by one shot first, waited out, or gone past: telemetry events
        # kind 'landing', control.navigator._clear_landing)
        "landing_events": dict(Counter(r.get("text", "").split(":")[0] for r in events if r.get("kind") == "landing")),
        "moves_away": _moves_away(records, end),
        "backstops": dict(Counter(r.get("text", "").split(":")[0] for r in events if r.get("kind") == "backstop")),
        **(_magic(world, end) if world.character.magic is not None else {}),
    }


def _magic(world, end: float) -> dict:
    """The magician's own (world.Her.magic): her casts a minute alive (casts_per_min), her attack keys pressed in the air
    (air_presses: nothing cast), inside the last attack's lock (attack_locked: ignored by the game, so her stack's pace
    is too quick for the world's lock if any), her wand's swings (swings: never, Attacks.no_regular), and her HP and MP
    potions a minute (potions_per_min, mp_potions_per_min is everyone's)."""
    return {"casts_per_min": round(world.casts / (end / 60), 2) if end else None, "air_presses": world.air_presses,
            "attack_locked": world.attack_locked, "swings": world.swings,
            "potions_per_min": round(world.potions_used / (end / 60), 2) if end else None}


TAKEN_BINS = ((15.0, "<=15"), (45.0, "15-45"), (150.0, "45-150"), (math.inf, ">150"))  # (_pets_took) units
APPROACH_BINS = (30.0, 45.0, 60.0, 90.0, 150.0, 300.0)  # ... her closest approach's, past 15 (PICK_REACH)
APPROACH_TAKEN_S = 8.0  # ... and s after it that a drop taken counts as taken by it


def _pets_took(world, drops: list, horizon: float, worth: Callable) -> dict:
    """Her pets' share of her pick-ups (world.Realism.pets: pet_picks, pet_share of her drops taken by her or them,
    pet_value_per_min of the worth, her_value_per_min her own key's), drops_expired (hers that lay their whole life),
    and the tables the pets were fitted by against the live records (world.PET_SPEED; World.path_log): `taken_from`,
    where she was when her drops were taken, units from them on their platform (TAKEN_BINS; 'off' on another, or in the
    air); `approach`, of her drops she came within APPROACH_BINS of (her nearest on their platform while they lay, past
    15) without standing on them, those taken within APPROACH_TAKEN_S of it."""
    path = getattr(world, "path_log", [])
    ts = [p[0] for p in path]
    taken = [d for d in drops if d.taken_by == "her"]
    pets = [d for d in taken if getattr(d, "pet", False)]
    where = {name: 0 for _, name in TAKEN_BINS} | {"off": 0}
    for d in taken:
        j = bisect.bisect_right(ts, d.taken_at) - 1
        if j < 0:
            continue
        _, pid, x = path[j]
        gap = abs(x - d.x)
        where["off" if pid != d.platform else next(name for top, name in TAKEN_BINS if gap <= top)] += 1
    approach = {str(int(b)): {"n": 0, "taken": 0} for b in APPROACH_BINS}
    for d in drops:
        end = d.taken_at if d.taken_by else min(d.since + d.life, horizon)
        best = None
        for i in range(bisect.bisect_left(ts, d.since), bisect.bisect_right(ts, end)):
            if path[i][1] == d.platform and (best is None or abs(path[i][2] - d.x) < best[0]):
                best = (abs(path[i][2] - d.x), path[i][0])
        if best is None or best[0] <= 15.0 or best[0] > APPROACH_BINS[-1]:
            continue
        c = approach[str(int(next(b for b in APPROACH_BINS if best[0] <= b)))]
        c["n"] += 1
        c["taken"] += d.taken_by == "her" and d.taken_at - best[1] <= APPROACH_TAKEN_S
    mins = horizon / 60
    return {"pet_picks": len(pets), "pet_share": round(len(pets) / len(taken), 3) if taken else None,
            "pet_value_per_min": round(sum(worth(d) for d in pets) / mins, 2),
            "her_value_per_min": round(sum(worth(d) for d in taken if not getattr(d, "pet", False)) / mins, 2),
            "drops_expired": sum(1 for d in drops if not d.taken_by and d.since + d.life <= horizon),
            "taken_from": where, "approach": approach}


def _moves_away(records: list[dict], end: float) -> dict:
    """Her stances that move her away (the fights' 'stance' records: a step, 'back', and a jump back), each counted once
    a run of looks deciding it (a look's record repeats the last one's while it's carried out); 'away', the steps from a
    close monster of them (the record's `away`: behavior.fight.moved_away; a stack without it has every step counted
    'back' alone, steps out from under a monster above too)."""
    start = next((r["t"] for r in records if r.get("topic") == "session"), None)
    out, last = Counter(), None
    for r in records:
        if r.get("topic") != "stance":
            continue
        if start is not None and r["t"] - start > end:
            break
        act = r.get("act")
        if act in ("back", "jump_back") and act != last:
            out[act] += 1
            out["away"] += act == "back" and bool(r.get("away"))
        last = act
    return dict(out)


LANDING_MOVED = 30.0  # (landings) units along her platform a landing on it moved her at least, as the live diagnosis's
LANDING_NEAR, LANDING_CLOSE = 105.0, 60.0  # ... and its distances: the claw's edge for a mushroom, and the close episodes'


def landings(world, end: float) -> dict:
    """Her landings that moved her up to `end` (World.land_log), by cause, with a live monster on the platform she
    landed on within LANDING_NEAR and LANDING_CLOSE ('near', 'close'), and within LANDING_NEAR walking at her ('at')."""
    out: dict = {}
    for t, src, xa, dst, xb, cause, near, _, walk in getattr(world, "land_log", ()):
        if t > end or (src == dst and xa is not None and abs(xa - xb) < LANDING_MOVED):
            continue
        c = out.setdefault(cause, {"n": 0, "near": 0, "close": 0, "at": 0})
        c["n"] += 1
        c["near"] += near is not None and near < LANDING_NEAR
        c["close"] += near is not None and near < LANDING_CLOSE
        c["at"] += near is not None and near < LANDING_NEAR and walk == 1
    return out


PICK_BINS = ((0.0, 40.0), (40.0, 60.0), (60.0, 130.0), (130.0, float("inf")))  # (picks) how far ahead, middle to middle


def picks(world) -> dict:
    """Her attacks by how far ahead the nearest monster on her platform in front of her was when each started, and what
    each did to it (world.World.pick_log): by kind ('ground'; from a jump 'straight' or 'turned', the latter an arrow
    toward her back acted on in the air before it; a jump back's 'back') and how far ahead (PICK_BINS, 'none' with
    nothing in front), [attacks, those that hurt that nearest monster (a hit, a swipe or a kill of it), those that hurt
    any, those that were Lucky Seven casts (not the claw's swipe), those that killed that nearest monster (what the live
    hunts' kills, her own alone, are set against: their bar drops have other players' hits in them)]; and from a jump
    'straight' 40-130 ahead by the slope: its feet over 5 lower than the floor under her ('down'), over 5 higher ('up'),
    else 'level'."""
    out: dict = {}
    for t, plan, mode, turned, gap_ms, ahead, drop, nearest, cast, outcome in getattr(world, "pick_log", []):
        kind = ("ground" if mode == "ground" else "back" if plan == "jump_back"
                else "turned" if turned else "straight")
        if ahead is None:
            where = "none"
        else:
            where = next(f"{lo:.0f}-{hi:.0f}" for lo, hi in PICK_BINS if lo <= ahead < hi)
        hurt, any_hit = nearest and outcome in ("hit", "kill", "swipe"), outcome in ("hit", "kill", "swipe")
        for key in ((kind, where),) + (((kind, "slope " + ("down" if drop > 5 else "up" if drop < -5 else "level")),)
                                         if kind == "straight" and ahead is not None and 40 <= ahead < 130 else ()):
            c = out.setdefault(key[0], {}).setdefault(key[1], [0, 0, 0, 0, 0])
            c[0] += 1
            c[1] += hurt
            c[2] += any_hit
            c[3] += bool(cast)
            c[4] += bool(nearest and outcome == "kill")
    return out


DEAD_BEFORE_S = 2.5  # (_at_dead) an attack decided this soon after a death where its target was seen ...
DEAD_NEAR = 40.0     # ... this near it (map units, the same kind on the same platform): at a dead monster. The live
                     # count's (D:/MapleHelper-scratch/analysis-1025/mp-attack-check/atdead.py, the hunt of 2026-10-25
                     # 23:43: 18 casts of Lucky Seven and 4 regular throws so in 11 minutes, the dying monster drawn 14-25
                     # units from where its kill was placed)


SKILL_KEYS = ("lucky_seven", "magic_claw")  # her skill's key in her 'attack' records: the thief's, the magician's


def _at_dead(world, records: list[dict]) -> dict:
    """at_dead and casts_at_dead (see the top): her attacks whose target as the stack chose it (its 'attack' record:
    behavior.attacks, the kind, platform and x it was seen at) had died DEAD_BEFORE_S or less before the choice, within
    DEAD_NEAR of where it was seen (world.death_log: by her or another player). The world's own pick may take another
    monster in its path then; what's counted is the choice made at one dead."""
    from .fakes import EPOCH
    deaths = getattr(world, "death_log", [])
    n = casts = 0
    for r in records:
        if r.get("topic") != "attack" or r.get("x") is None:
            continue
        t = r["t"] - EPOCH
        if any(t - DEAD_BEFORE_S <= dt <= t and kind == r.get("kind") and plat == r.get("platform")
               and abs(x - r["x"]) <= DEAD_NEAR for dt, kind, plat, x in deaths):
            n += 1
            casts += r.get("key") in SKILL_KEYS
    return {"at_dead": n, "casts_at_dead": casts}


def _at_dead_ids(world) -> dict:
    """The throws at the dead by the world's own monsters (World.aim_log, from the Sim's hook on her attack choice: the
    monster of the target's kind on its platform drawn nearest where the stack saw it, as it chose): at_dead_hers, at
    one her attack had killed by then (drawn alive until her star lands: Realism.dies_late), at_dead_others, at another
    player's; casts_at_dead_id, of those Lucky Seven; aims, the choices judged; live_taken, the feed's tracks the ledger
    took for dead (behavior.ledger.Ledger.dead) whose monster drawn there was alive (World.live_taken)."""
    log = getattr(world, "aim_log", [])
    dead = [a for a in log if a[3] in ("her", "rival")]
    return {"at_dead_hers": sum(a[3] == "her" for a in log), "at_dead_others": sum(a[3] == "rival" for a in log),
            "casts_at_dead_id": sum(a[1] in SKILL_KEYS for a in dead), "aims": len(log),
            "live_taken": len(getattr(world, "live_taken", []))}


def _travelling(world, records: list[dict], end: float) -> dict:
    """trip_s, trip_hurt and trip_hp_lost (see the top): her moves, from each 'chain' or 'move' record (a chain sent, a
    move made alone) to the next 'moved' record (chains that follow on are in their first one's), or a move the mover
    couldn't do (a 'cannot' event: nothing made), in world time, up to `end`; and the touches that hurt her then."""
    start = next((r["t"] for r in records if r.get("topic") == "session"), None)
    if start is None:
        return {"trip_s": 0.0, "trip_hurt": 0, "trip_hp_lost": 0.0}
    spans = _trip_spans(records, start, end)
    on = [hp for t, hp in world.touch_log if t <= end and any(a <= t <= b for a, b in spans)]
    return {"trip_s": round(sum(max(0.0, b - a) for a, b in spans if a < end), 1),
            "trip_hurt": sum(1 for hp in on if hp > 0), "trip_hp_lost": round(sum(on), 3)}


def _trip_spans(records: list[dict], start: float, end: float) -> list[tuple[float, float]]:
    """Her moves (_travelling's): from each 'chain' or 'move' record to the next 'moved' record or 'cannot' event, in
    world time (s from `start`, the session's), up to `end`."""
    spans, began = [], None
    for r in records:
        if r.get("topic") in ("chain", "move") and began is None:
            began = r["t"] - start
        elif (r.get("topic") == "moved" or r.get("topic") == "event" and r.get("kind") == "cannot") \
                and began is not None:
            spans.append((began, min(r["t"] - start, end)))
            began = None
    if began is not None:
        spans.append((began, end))
    return spans


RETOUCH_S = 2.0  # (retouch_2s) a touch this soon after the one before is a touch again


def _retouches(world, end: float) -> int:
    """retouch_2s (see the top): her touches (a MISS too: World.touch_log) up to `end` that came RETOUCH_S or less after
    the one before (live in walk B, 53 of 148, 34 of them 1.43-1.6 s on, as her invulnerability ended)."""
    ts = [t for t, _ in world.touch_log if t <= end]
    return sum(1 for a, b in zip(ts, ts[1:]) if b - a <= RETOUCH_S)


def _trips(world, records: list[dict], end: float) -> dict:
    """trips, arrive_share, trip_s_per_arrival, touches_by, mover_moves and evades (see the top): her trips (the
    navigator's 'trip' records up to `end`, all and those that got there: 'arrived' or 'in position'), the seconds on
    the move an arrival, her touches (a MISS too) by what she was doing ('chain': a chain's plan running, 'move': a move
    of the mover's, 'off': neither: looks, waits, fights, between trips), her moves made by the mover alone ('move'
    records), and the navigator's evade steps by kind ('evade' events' `step`)."""
    start = next((r["t"] for r in records if r.get("topic") == "session"), None)
    if start is None:
        return {"trips": {"total": 0, "arrived": 0}, "arrive_share": None, "trip_s_per_arrival": None,
                "touches_by": {}, "mover_moves": 0, "evades": {}}
    ended = [r for r in records if r.get("topic") == "trip" and r["t"] - start <= end]
    arrived = sum(1 for r in ended if r.get("result") in ("arrived", "in position"))
    spans, began = [], None  # (from, to, 'chain' or 'move'), as _trip_spans has them
    for r in records:
        if r.get("topic") in ("chain", "move") and began is None:
            began = (r["t"] - start, "chain" if r["topic"] == "chain" else "move")
        elif (r.get("topic") == "moved" or r.get("topic") == "event" and r.get("kind") == "cannot") \
                and began is not None:
            spans.append((began[0], min(r["t"] - start, end), began[1]))
            began = None
    if began is not None:
        spans.append((began[0], end, began[1]))
    by = Counter({"chain": 0, "move": 0, "off": 0})
    for t, _ in world.touch_log:
        if t <= end:
            by[next((k for a, b, k in spans if a <= t <= b), "off")] += 1
    moving = sum(max(0.0, b - a) for a, b, _ in spans if a < end)
    return {"trips": {"total": len(ended), "arrived": arrived},
            "arrive_share": round(arrived / len(ended), 4) if ended else None,
            "trip_s_per_arrival": round(moving / arrived, 2) if arrived else None,
            "touches_by": dict(by),
            "mover_moves": sum(1 for r in records if r.get("topic") == "move" and r["t"] - start <= end),
            "evades": dict(Counter(str(r.get("step")) for r in records if r.get("topic") == "event"
                                   and r.get("kind") == "evade" and r["t"] - start <= end))}


# The hunt loop's rules for her drops (behavior.hunt: 6a, 6, 5, 5b, 8, 10), for pile_s: while one of them decides, or
# her objective is a pile (a trip for one: its 'hunt' record's objective "pile on ..."), she's on her drops.
PILE_RULES = ("valuable pile", "urgent pile", "pile here", "on the way", "lull loot", "loot elsewhere",
              "sweep", "sweep: landing", "sweep: clear first")  # (the loot sweep's, behavior.sweep, from 2026-10-30)


def _full_stands(world, records: list[dict], end: float) -> dict:
    """pick_refused, the pick-ups the world refused her for want of room; stands_for_nothing, her stands at a pile (the
    hunt loop's 'pick' events, in world time up to `end`) with drops lying within LOOT_REACH of it and none she could
    take: every one an item of a tab full by then and of a kind she had no stack of (World.takes); and of those,
    nothing_known, the stands after the hunt had learned that tab full (its 'inventory' events: the ones it could
    have spared; before, a stand is how it learns). All 0 where her inventory has room for everything."""
    start = next((r["t"] for r in records if r.get("topic") == "session"), None)
    nothing = known = 0
    if start is not None and world.full_at:
        learned: dict[str, list] = {}  # tab -> (t, full) as the hunt learned it
        for r in records:
            if r.get("topic") == "event" and r.get("kind") == "inventory" and r.get("why") != "its stack full too":
                learned.setdefault(r.get("tab"), []).append((r["t"] - start, bool(r.get("full"))))
        for r in records:
            if r.get("topic") != "event" or r.get("kind") != "pick" or r["t"] - start > end:
                continue
            t = r["t"] - start
            here = [d for d in world.drops if d.platform == r.get("platform")
                    and abs(d.x - r.get("x", 0.0)) <= LOOT_REACH and d.since <= t < d.since + d.life
                    and (d.taken_at is None or d.taken_at >= t)]
            if here and not any(d.mesos or world.takes(d.kind, t) for d in here):
                nothing += 1
                known += all(next((full for at, full in reversed(learned.get(tab_of(d.kind), [])) if at <= t), False)
                             for d in here)
    return {"pick_refused": world.pick_refused, "stands_for_nothing": nothing, "nothing_known": known}


def _on_piles(world, records: list[dict], end: float) -> dict:
    """pile_s, pile_hurt and pile_hp_lost, stand_hurt and stand_hp_lost (see the top): the stretches the hunt loop
    was on her drops (from a 'hunt' record with a rule of PILE_RULES or a pile for its objective to the next without),
    in world time up to `end`, and the touches that hurt her then; and those off her trips (_trip_spans)."""
    start = next((r["t"] for r in records if r.get("topic") == "session"), None)
    if start is None:
        return {"pile_s": 0.0, "pile_hurt": 0, "pile_hp_lost": 0.0, "stand_hurt": 0, "stand_hp_lost": 0.0}
    spans, began = [], None
    for r in records:
        if r.get("topic") != "hunt":
            continue
        on = r.get("rule") in PILE_RULES or str(r.get("objective") or "").startswith("pile")
        if on and began is None:
            began = r["t"] - start
        elif not on and began is not None:
            spans.append((began, min(r["t"] - start, end)))
            began = None
    if began is not None:
        spans.append((began, end))
    on = [hp for t, hp in world.touch_log if t <= end and any(a <= t <= b for a, b in spans)]
    trips = _trip_spans(records, start, end)
    off = [hp for t, hp in world.touch_log if t <= end and not any(a <= t <= b for a, b in trips)]
    return {"pile_s": round(sum(max(0.0, b - a) for a, b in spans if a < end), 1),
            "pile_hurt": sum(1 for hp in on if hp > 0), "pile_hp_lost": round(sum(on), 3),
            "stand_hurt": sum(1 for hp in off if hp > 0), "stand_hp_lost": round(sum(off), 3)}


WAITING = ("wait", "lull loot")  # the hunt loop's rules for waiting out respawns (behavior.hunt: 9 and 8)


def _waits(records: list[dict]) -> list[tuple[float, float]]:
    """(start, end) of each stretch the hunt loop's rule was one of WAITING, in world time (s from the run's start):
    from the 'hunt' record that names one to the next that names another rule (or the end)."""
    start = next((r["t"] for r in records if r.get("topic") == "session"), None)
    if start is None:
        return []
    out, began = [], None
    for r in records:
        if r.get("topic") != "hunt":
            continue
        waiting = r.get("rule") in WAITING
        if waiting and began is None:
            began = r["t"] - start
        elif not waiting and began is not None:
            out.append((began, r["t"] - start))
            began = None
    if began is not None:
        out.append((began, float("inf")))
    return out


# --- many runs -----------------------------------------------------------------------------------------

def run_many(policy: str, seeds: int, preset: str, minutes: float, jobs: int = 1, think: float = 0.0,
             costs: str = "coded", ways: int | None = None, map_id: int = FIRST_MAP,
             world: tuple = (), grounds: tuple = (), attack: str | None = None,
             think_clock: float = 0.0) -> list[dict]:
    return run_seeds(policy, list(range(1, seeds + 1)), preset, minutes, jobs, think, costs, ways, map_id, world,
                     grounds, attack, think_clock)


def run_seeds(policy: str, seeds: list[int], preset: str, minutes: float, jobs: int = 1, think: float = 0.0,
              costs: str = "coded", ways: int | None = None, map_id: int = FIRST_MAP, world: tuple = (),
              grounds: tuple = (), attack: str | None = None, think_clock: float = 0.0,
              done: Callable[[int, dict], None] | None = None) -> list[dict]:
    """run_many on the seeds named, their results in that order; `done(seed, result)` as each run ends (in the order
    they end, with `jobs` processes)."""
    args = [(policy, k, preset, minutes, think, costs, ways, map_id, tuple(world), tuple(grounds), attack, think_clock)
            for k in seeds]
    if jobs <= 1 or not args:
        out = []
        for a in args:
            out.append(run_one(*a))
            if done:
                done(a[1], out[-1])
        return out
    from concurrent.futures import as_completed
    from gamelab.sim import harness  # by its module name, so worker processes can find it (not __main__)
    with ProcessPoolExecutor(max_workers=min(jobs, len(args)), initializer=harness._worker) as pool:
        futures = {pool.submit(harness._run_one_args, a): a[1] for a in args}
        got = {}
        for f in as_completed(futures):
            got[futures[f]] = f.result()
            if done:
                done(futures[f], got[futures[f]])
    return [got[k] for k in seeds]


# --- the baseline cache (sim.baseline) -------------------------------------------------------------------

def cache_of(policy: str, preset: str, minutes: float, think: float = 0.0, costs: str = "coded",
             ways: int | None = None, map_id: int = FIRST_MAP, world: tuple = (), grounds: tuple = (),
             attack: str | None = None, think_clock: float = 0.0, code: str | None = None, commit: str | None = None):
    """The baseline cache of a run's arguments on code `code` (baseline.code_id: a commit's; None: this checkout's,
    while its code is as committed), or (None, why there's none). `commit`: the commit, for its meta.json."""
    from . import baseline
    p = baseline.params(run_one, (policy, 0, preset, minutes, think, costs, ways, map_id, tuple(world),
                                  tuple(grounds), attack, think_clock))
    if why := baseline.uncacheable(p):
        return None, why
    if code is None:
        code, why = baseline.identity(REPO)
        if code is None:
            return None, why
        commit = baseline.head(REPO)
    return baseline.Cache(code, p, REPO, commit=commit), ""


def cached_run(policy: str, seeds: list[int], preset: str, minutes: float, jobs: int = 1, think: float = 0.0,
               costs: str = "coded", ways: int | None = None, map_id: int = FIRST_MAP, world: tuple = (),
               grounds: tuple = (), attack: str | None = None, think_clock: float = 0.0,
               say: Callable[[str], None] = print) -> list[dict]:
    """run_seeds through the baseline cache: the seeds this checkout's commit has cached for these arguments read, the
    others run and kept (each as it ends, while the checkout's code is still as committed); all of them as read back
    from their files. A checkout with changes, or a run that isn't deterministic (`think`, `think_clock`), runs them all
    and keeps none, and `say`s why."""
    from . import baseline
    cache, why = cache_of(policy, preset, minutes, think, costs, ways, map_id, world, grounds, attack, think_clock)
    run = (policy, preset, minutes, jobs, think, costs, ways, map_id, world, grounds, attack, think_clock)
    if cache is None:
        say(f"cache: not used: {why}")
        return [baseline.plain(r) for r in run_seeds(policy, seeds, *run[1:])]
    have = {k: r for k in seeds if (r := cache.get(k)) is not None}
    todo = [k for k in seeds if k not in have]
    say(f"cache: {len(have)} of {len(seeds)} seed(s) read from {cache.folder}" + (f", {len(todo)} to run" if todo
                                                                                  else ""))
    lost = []

    def keep(seed: int, result: dict) -> None:
        if baseline.identity(REPO)[0] == cache.code:
            cache.put(seed, result)
        else:
            lost.append(seed)
    ran = dict(zip(todo, run_seeds(policy, todo, *run[1:], done=keep))) if todo else {}
    if lost:
        say(f"cache: the checkout's code changed while it ran: seed(s) {', '.join(map(str, sorted(lost)))} not kept")
    return [have[k] if k in have else baseline.plain(ran[k]) for k in seeds]


def _worker() -> None:
    """A worker process at below-normal priority (feed.quiet): the perception service, at normal, keeps its pace on a
    machine the runs keep busy (it looked at 20.8 frames a second beside 22 busy workers so, 14 at their priority)."""
    from ..localization.feed import quiet
    quiet()


def _run_one_args(a: tuple) -> dict:
    return run_one(*a)


COLUMNS = ("seed", "exp_per_min", "kills", "rival", "hp_lost", "hp_low", "potions", "pet_potions", "mp_per_min",
           "mp_potions", "mp_per_kill", "casts_at_dead", "at_dead", "finishes", "died_at", "fell_out_at",
           "left_map_at",
           "drops_share", "value_share", "value_per_min", "longest_idle", "stuck_s", "safety_stops", "safety_at",
           "darts_stops",
           "alive_s", "hurt", "trip_hurt", "pile_hurt", "stand_hurt", "knock_falls", "fresh_hurt", "wait_s",
           "wait_hp_lost")


def table(results: list[dict]) -> str:
    rows = [" ".join(f"{c:>12}" for c in COLUMNS)]
    for r in results:
        rows.append(" ".join(f"{_fmt(r.get(c)):>12}" for c in COLUMNS))
    rows.append(" ".join(f"{_fmt(v):>12}" for v in ["mean"] + [_mean(results, c) for c in COLUMNS[1:]]))
    caps = Counter()
    for r in results:
        caps.update(r["backstops"])
    rows.append(f"backstops fired: {dict(caps) or 'none'}")
    return "\n".join(rows)


def _mean(results: list[dict], key: str):
    vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
    if key in ("died_at", "fell_out_at", "left_map_at", "safety_at", "darts_at"):
        return sum(1 for r in results if r.get(key) is not None)  # deaths, falls out of the map, portals taken, stops
    return round(statistics.fmean(vals), 3) if vals else None


def _fmt(v) -> str:
    return "-" if v is None else str(v)


# --- the bar -------------------------------------------------------------------------------------------

MIN_PAIRS = 10  # the bars are for 10 seeds or more, in both runs


def _paired(base: list[dict], cand: list[dict]) -> tuple[list[dict], list[dict]]:
    """The runs of the seeds both have, in the same order (means over different worlds compare nothing)."""
    by_seed = {r["seed"]: r for r in base}
    pairs = [(by_seed[r["seed"]], r) for r in cand if r["seed"] in by_seed]
    return [a for a, _ in pairs], [b for _, b in pairs]


def compare(base: list[dict], cand: list[dict]) -> tuple[bool, list[str]]:
    """Whether `cand` passes the bar against `base` (seed by seed, over the seeds both have: MIN_PAIRS at
    least), and why: better EXP a minute on at least 8 of 10 seeds and no seed more than 10% worse; at
    least the drops share on average; no more HP lost on average; no death, no fall out of the map, and no stop (the
    safety stop, out of darts: the hunt ended)."""
    every = cand  # deaths count on every seed, compared or not: "no death" isn't a comparison
    if (maps := _maps(base, cand)) is not None:
        return False, [maps, "does NOT pass the bar"]
    if (who := _characters(base, cand)) is not None:
        return False, [who, "does NOT pass the bar"]
    base, cand = _paired(base, cand)
    if len(cand) < MIN_PAIRS:
        return False, [f"{len(cand)} seed(s) in both runs: the bar needs {MIN_PAIRS}", "does NOT pass the bar"]
    pairs = list(zip(base, cand))
    lines, ok = [], True
    if len(every) > len(cand):
        lines.append(f"{len(every) - len(cand)} candidate seed(s) without a baseline: not compared")
    better = sum(1 for a, b in pairs if b["exp_per_min"] > a["exp_per_min"])
    worst = min((b["exp_per_min"] / a["exp_per_min"] - 1 for a, b in pairs if a["exp_per_min"]), default=0.0)
    need = -(-len(pairs) * 8 // 10)
    lines.append(f"EXP a minute better on {better} of {len(pairs)} seeds (need {need}); worst seed {worst:+.0%}")
    ok &= better >= need and worst >= -0.10
    for key, higher in (("drops_share", True), ("hp_lost", False)):
        a, b = _mean(base, key), _mean(cand, key)
        good = a is not None and b is not None and (b >= a if higher else b <= a)
        lines.append(f"{key}: {a} -> {b} ({'ok' if good else 'worse' if a is not None and b is not None else 'missing'})")
        ok &= good
    deaths = sum(1 for r in every if r.get("died_at") is not None)
    lines.append(f"deaths: {deaths}")
    ok &= deaths == 0
    stopped = [r["seed"] for r in every if r.get("safety_stops") or r.get("darts_stops")]
    if stopped:
        lines.append(f"the hunt stopped (the safety stop, out of darts): seed(s) {', '.join(map(str, stopped))}")
    ok &= not stopped
    fell = [r["seed"] for r in every if r.get("fell_out_at") is not None]
    if fell:
        lines.append(f"fell out of the map: seed(s) {', '.join(map(str, fell))}")
    ok &= not fell
    away = [r["seed"] for r in every if r.get("left_map_at") is not None]
    if away:
        lines.append(f"went through a portal: seed(s) {', '.join(map(str, away))}")
    ok &= not away
    if all("nav" in r for r in base + cand):  # (runs saved before it was counted have none)
        lines.append("her trips, in all (shown only): " + ", ".join(
            f"{key} {sum(r['nav'][key] for r in base)} -> {sum(r['nav'][key] for r in cand)}" for key in NAV))
    lines.append("PASSES the bar" if ok else "does NOT pass the bar")
    return ok, lines


NAV = ("cut_hit", "cut_off_path", "replans", "misses", "falls", "glances")  # navigation()'s counts, and the glances


def _maps(base: list[dict], cand: list[dict]) -> str | None:
    """Why two saved runs can't be compared: they're on different maps (runs saved before the map was kept are the
    first map's), or None."""
    a, b = ({r.get("map", FIRST_MAP) for r in runs} for runs in (base, cand))
    return None if a == b and len(a) == 1 else f"the runs are on different maps ({sorted(a)} and {sorted(b)})"


def _characters(base: list[dict], cand: list[dict]) -> str | None:
    """Why two saved runs can't be compared: they play different characters (runs saved before the character was kept
    are the thief's), or None."""
    from ..character import DEFAULT
    a, b = ({r.get("character", DEFAULT) for r in runs} for runs in (base, cand))
    return None if a == b and len(a) == 1 else f"the runs play different characters ({sorted(a)} and {sorted(b)})"


def save(name: str, results: list[dict]) -> None:
    FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / f"{name}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


def load(name: str) -> list[dict]:
    """A saved run: by its name (gamelab/lab/harness/NAME.json), a .json file's path, or a baseline cache's folder
    (sim.baseline: its seed-NNN.json, in seed order)."""
    from pathlib import Path
    path = Path(name)
    if path.is_dir():
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("seed-*.json"))]
    if path.suffix == ".json" and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads((FOLDER / f"{name}.json").read_text(encoding="utf-8"))


# --- a candidate against master's cached baseline (`ab`) -------------------------------------------------

AB_SEEDS = (12, 24, 48)  # the seeds `ab` takes, more only while a headline difference isn't told from nothing
# The headline metrics, a seed's value each: EXP a minute, touches, and the runs that died, fell out of the map (or
# went through a portal) or stopped (the safety stop, out of darts), as the bar counts them (compare).
HEADLINE = {
    "exp_per_min": lambda r: r.get("exp_per_min"),
    "touches": lambda r: r.get("touches"),
    "deaths": lambda r: int(r.get("died_at") is not None),
    "falls": lambda r: int(r.get("fell_out_at") is not None or r.get("left_map_at") is not None),
    "stops": lambda r: int(bool(r.get("safety_stops") or r.get("darts_stops"))),
}


def metric(result: dict, name: str) -> float | None:
    """A seed's value of a metric: HEADLINE's, or a key of its result (dots for nested ones: `nav.misses`,
    `landings.trip.near`; a count that's absent 0); None where it has none (mp_per_kill without a kill). A name no
    seed's result has at all reads 0 or None everywhere: `ab` refuses those first (has_metric)."""
    if name in HEADLINE:
        return HEADLINE[name](result)
    value, parts = result, name.split(".")
    for part in parts:
        if not isinstance(value, dict):
            return None
        if part not in value:
            return 0 if len(parts) > 1 else None
        value = value[part]
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def has_metric(result: dict, name: str) -> bool:
    """Whether a seed's result has metric `name` at all: HEADLINE's always, else every part of its dotted path (a
    value of None counts: mp_per_kill without a kill)."""
    if name in HEADLINE:
        return True
    value = result
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def _paths(result: dict, prefix: str = "") -> list[str]:
    """Every dotted path to a number in a result (for a mistyped metric's suggestions)."""
    out = []
    for k, v in result.items():
        if isinstance(v, dict):
            out += _paths(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool) or v is None:
            out.append(f"{prefix}{k}")
    return out


def difference(base: list[dict], cand: list[dict], name: str) -> dict:
    """Metric `name` seed by seed (the seeds both have; the differences of those with a value in both): n, the means,
    the mean difference (candidate - base) and its standard error (of the paired differences), t, and whether it's
    'zero' (every seed the same to the digit, its value or its having none), 'too few' (not, but under 2 seeds with a
    value in both: no standard error to tell it by), 'beyond 2 se' or 'within 2 se' (a mean of 0 whose seeds differ,
    +4 touches on one and -4 on another, among them)."""
    base, cand = _paired(base, cand)
    values = [(metric(x, name), metric(y, name)) for x, y in zip(base, cand)]
    pairs = [(a, b) for a, b in values if a is not None and b is not None]
    n = len(pairs)
    d = [b - a for a, b in pairs]
    mean = math.fsum(d) / n if n else 0.0
    se = statistics.stdev(d) / math.sqrt(n) if n > 1 else 0.0
    if values and all(a == b for a, b in values):
        verdict = "zero"
    elif n < 2:
        verdict = "too few"
    else:
        verdict = "beyond 2 se" if abs(mean) > 2 * se else "within 2 se"
    return {"metric": name, "n": n, "base": math.fsum(a for a, _ in pairs) / n if n else None,
            "cand": math.fsum(b for _, b in pairs) / n if n else None, "diff": mean, "se": se,
            "t": mean / se if se else None, "verdict": verdict}


OPEN = ("within 2 se", "too few")  # the verdicts that leave a difference unsettled: `ab` takes more seeds


def ab_next(rows: list[dict], seeds: int, steps: tuple[int, ...] = AB_SEEDS) -> tuple[int | None, str]:
    """`ab`'s rule: after `seeds` seeds, the next count of seeds to take (None: stop) and why. More only while a
    headline difference is within 2 standard errors and not zero, or has too few seeds with a value on both sides
    to tell (OPEN), up to the last of `steps`; stopped there, those are said to be unsettled still."""
    within = [r["metric"] for r in rows if r["verdict"] == "within 2 se"]
    few = [r["metric"] for r in rows if r["verdict"] == "too few"]
    if not within and not few:
        return None, f"every headline difference is beyond 2 se or zero at {seeds} seeds"
    parts = ([f"within 2 se and not zero: {', '.join(within)}"] if within else []) \
        + ([f"too few seeds with a value on both sides: {', '.join(few)}"] if few else [])
    more = next((s for s in steps if s > seeds), None)
    if more is None:
        return None, f"{seeds} seeds, the most (--max-seeds); still unsettled: {'; '.join(parts)}"
    return more, f"at {seeds} seeds {'; '.join(parts)}"


def ab_table(rows: list[dict]) -> str:
    out = [f"{'metric':>22} {'n':>3} {'base':>10} {'cand':>10} {'diff':>9} {'diff %':>7} {'se':>8} {'t':>6}  verdict"]
    for r in rows:
        rel = f"{r['diff'] / r['base']:+.1%}" if r["base"] else "-"
        t = f"{r['t']:+.2f}" if r["t"] is not None else "-"
        out.append(f"{r['metric']:>22} {r['n']:>3} {_num(r['base']):>10} {_num(r['cand']):>10} "
                   f"{r['diff']:>+9.3f} {rel:>7} {r['se']:>8.3f} {t:>6}  {r['verdict']}")
    return "\n".join(out)


def _num(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def _seed_span(seeds: list[int]) -> str:
    return f"{seeds[0]}-{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) else ",".join(map(str, seeds))


# --- checking the bookkeeping ----------------------------------------------------------------------------

def check() -> int:
    """Run every scenario and check that the world's own logs, which outcomes() reads, agree with its
    summary: her kills, pick-ups and drops."""
    from . import scenarios
    bad = 0
    for name, sc in scenarios.SCENARIOS.items():
        trace, sim = scenarios.run_sim(sc)
        w, summary = sim.world, trace["world"]
        kills = sum(1 for k in w.kill_log if k[3] == "her")
        picked = sum(1 for d in w.drops if d.taken_by == "her")
        problems = []
        if kills != sum(summary["kills"].values()):
            problems.append(f"kills {kills} vs {summary['kills']}")
        if picked != sum(summary["picked"].values()):
            problems.append(f"picked {picked} vs {summary['picked']}")
        print(f"{'ok' if not problems else 'DIFFERS'} {name}" + (f": {'; '.join(problems)}" if problems else ""))
        bad += bool(problems)
    return 1 if bad else 0


CHARACTERS = ("thief", "magician")  # the characters the simulator plays (sim.scenarios.HER, HERS)


def _who(argv: list[str], environ=os.environ) -> tuple[str | None, str]:
    """The character a run plays: `--character`, else GAMELAB_CHARACTER (as `ab` hands the base's filler the key's), else
    the thief; (it, "") or (None, why not): one the simulator doesn't play, or the two naming different ones."""
    from ..character import DEFAULT, ENV
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--character")
    named = pre.parse_known_args(argv)[0].character
    env = environ.get(ENV)
    if named and env and env != named:
        return None, f"{ENV} names {env} and --character {named}: one of them"
    who = named or env or DEFAULT
    if who not in CHARACTERS:
        return None, f"the simulator plays {' or '.join(CHARACTERS)}, not {who}"
    return who, ""


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    who, why = _who(argv)
    if who is None:
        print(why)
        return 2
    from ..character import DEFAULT, ENV
    from ..motion.physics import FIXED_STATS
    # The same set order in every run, her stats as measured, not as config.toml has them, and the character it plays
    # (GAMELAB_CHARACTER: her physics and her profile are the process's, gamelab.character)
    if (os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get(FIXED_STATS) != "1"
            or os.environ.get(ENV, DEFAULT) != who):
        env = dict(os.environ, PYTHONHASHSEED="0", **{FIXED_STATS: "1"})
        if who != DEFAULT:
            env[ENV] = who
        return subprocess.call([sys.executable, "-X", "utf8", "-m", "gamelab.sim.harness", *argv], env=env,
                               cwd=str(REPO))
    p = argparse.ArgumentParser(prog="python -m gamelab.sim.harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = _run_options(sub.add_parser("run"))
    r.add_argument("--cache", action="store_true",
                   help="read the seeds this commit has run with these arguments from the baseline cache and run only "
                        "the others, keeping them (sim.baseline; only while the checkout's code is as committed)")
    k = _run_options(sub.add_parser("cached", help="a commit's cached seeds, read: nothing run"))
    k.add_argument("--sha", default="HEAD", help="the commit whose cache to read (default HEAD; master, a SHA): its "
                                                 "code's (baseline.code_id)")
    a = _run_options(sub.add_parser("ab", help="this checkout against a base commit's cached baseline, seeds added "
                                               "while a headline difference isn't told from nothing"))
    a.set_defaults(seeds=AB_SEEDS[0])
    a.add_argument("--base", default="master", help="the base commit (default master)")
    a.add_argument("--metric", action="append", default=[],
                   help="a metric the change targets, beside the headline ones (a result's key: hurt, close_s, "
                        "nav.misses, landings.trip.near; again for another)")
    a.add_argument("--max-seeds", type=int, default=AB_SEEDS[-1], help=f"the most seeds (default {AB_SEEDS[-1]})")
    a.add_argument("--fill", default=os.environ.get("GAMELAB_BASE_FILL"),
                   help="the script that runs the base's missing seeds in its own checkout and caches them (default "
                        "GAMELAB_BASE_FILL, else <baselines>/base.py)")
    c = sub.add_parser("compare")
    c.add_argument("baseline", help="a saved run's name, a .json file, or a baseline cache's folder")
    c.add_argument("candidate")
    s = sub.add_parser("spot", help="a saved run's seeds run again here, each the same to the digit or not")
    s.add_argument("run", help="a saved run's name, a .json file, or a baseline cache's folder")
    s.add_argument("--seeds", type=lambda v: [int(k) for k in v.split(",")],
                   help="the seeds to run again, e.g. 3,9 (default: its first and its last)")
    s.add_argument("--jobs", type=int, default=1)
    s.add_argument("--character", choices=CHARACTERS, default=argparse.SUPPRESS,
                   help="the character the saved run plays (default: GAMELAB_CHARACTER's, else the thief)")
    sub.add_parser("check")
    args = p.parse_args(argv)
    from . import scenarios
    if args.cmd == "spot":
        results = load(args.run)
        if theirs := sorted({r.get("character", DEFAULT) for r in results} - {who}):
            print(f"the run plays {', '.join(theirs)}: spot it with --character {theirs[0]}")
            return 2
        seeds = args.seeds or sorted({min(r["seed"] for r in results), max(r["seed"] for r in results)})
        map_id = results[0].get("map", FIRST_MAP) if results else FIRST_MAP
        if not scenarios.available(map_id):
            print(f"map {map_id} isn't available on this computer (gamedata/data): nothing to run")
            return 2
        return 0 if spot(results, seeds, args.jobs) else 1
    map_id = getattr(args, "map", FIRST_MAP)
    if not scenarios.available(map_id):
        print(f"map {map_id} isn't available on this computer (gamedata/data): nothing to run")
        return 2
    if args.cmd == "check":
        return check()
    if args.cmd == "compare":
        ok, lines = compare(load(args.baseline), load(args.candidate))
        print("\n".join(lines))
        return 0 if ok else 1
    if args.map not in PRESET_MAPS.get(args.preset, (args.map,)):
        print(f"preset {args.preset} is map {', '.join(map(str, PRESET_MAPS[args.preset]))}'s alone")
        return 2
    if who != DEFAULT and (args.map, who) not in scenarios.HERS:
        print(f"the simulator plays {who} on map{'s' if len(scenarios.HERS) > 1 else ''} "
              f"{', '.join(str(m) for m, w in scenarios.HERS if w == who)} alone (sim.scenarios.HERS): --map")
        return 2
    if args.cmd == "cached":
        return _cached(args)
    if args.cmd == "ab":
        return ab(args)
    if args.cache:
        results = cached_run(args.policy, list(range(1, args.seeds + 1)), args.preset, args.minutes, args.jobs,
                             args.think, args.costs, args.ways, args.map, tuple(args.world), args.grounds, args.attack,
                             args.think_clock)
    else:
        results = run_many(args.policy, args.seeds, args.preset, args.minutes, args.jobs, args.think, args.costs,
                           args.ways, args.map, tuple(args.world), args.grounds, args.attack, args.think_clock)
    print(table(results))
    name = args.save or _name(args)
    save(name, results)
    print(f"saved {FOLDER / (name + '.json')}")
    return 0


def _run_options(r: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """A run's options (run, cached, ab)."""
    r.add_argument("policy", choices=sorted(POLICIES))
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--minutes", type=float, default=20.0)
    r.add_argument("--preset", choices=sorted(PRESETS), default="game")
    r.add_argument("--jobs", type=int, default=1)
    thinks = r.add_mutually_exclusive_group()
    thinks.add_argument("--think", type=float, default=0.0,
                        help="charge the stack's computing to the fake clock, times this (1: this computer's speed), "
                             "at the fakes' calls (a read of the clock inside it doesn't move: a replan is never late)")
    thinks.add_argument("--think-clock", type=float, default=0.0, metavar="K",
                        help="as --think K, and at every read of the clock too, as it goes: a replan's own computing "
                             "can make it late (navigator.SPLICE_LATE_S), as live")
    r.add_argument("--save",
                   help="name to save the results under (default: POLICY-PRESET, then -COSTS and -waysN)")
    r.add_argument("--ways", type=int, help="the ways her trips choose among (route.ROUTES; default: route's own)")
    r.add_argument("--costs", choices=sorted(COSTS), default="coded",
                   help="what the simulator's looks take: coded (the fakes' own), measured (today's), measured-1366 "
                        "(and the feed's as measured on Tree Dungeon's 1366x768 frames: feed.COSTS_1366)")
    r.add_argument("--map", type=int, choices=sorted(STARTS), default=FIRST_MAP,
                   help="the map (default: the first, Thicket Around the Beach II; 10004021: West Rocky Mountain II; "
                        "10002073: Tree Dungeon, Forest Up North II)")
    r.add_argument("--world", action="append", choices=sorted(WORLDS), default=[],
                   help="what isn't known of the game, switched on (WORLDS; again for another)")
    r.add_argument("--grounds", type=lambda v: tuple(int(g) for g in v.split(",")), default=(),
                   help="hunt only these grounds, e.g. 23 (default: all of them)")
    r.add_argument("--character", choices=CHARACTERS, default=argparse.SUPPRESS,
                   help="the character the run plays (default: GAMELAB_CHARACTER's, else the thief): the magician as "
                        "the recordings of 2026-11-02 measured her (sim.scenarios.HERS), on Tree Dungeon")
    r.add_argument("--attack", choices=("choice", "skill", "regular"),
                   help="the stack's attack a throw (behavior.attacks), her Lucky Seven key casting it at its MP "
                        "(ATTACK_HER): choice, Lucky Seven but the regular attack to finish a monster and while her MP "
                        "is under a cast (live's default); skill, Lucky Seven while she has the MP; regular, the "
                        "regular attack (default: the regular attack, her as the map has her)")
    return r


def _name(args) -> str:
    """A run's default name to save it under (with --think or --think-clock, saying so: until the review of 2026-11-01
    such a run's name was the plain run's, and saved over it)."""
    return (f"{args.policy}-{args.preset}"
            + (f"-think{args.think:g}" if args.think else "")
            + (f"-think-clock{args.think_clock:g}" if args.think_clock else "")
            + (f"-{args.costs}" if args.costs != "coded" else "")
            + (f"-ways{args.ways}" if args.ways is not None else "")
            + (f"-{args.map}" if args.map != FIRST_MAP else "")
            + "".join(f"-{w}" for w in args.world)
            + (f"-g{'-'.join(map(str, args.grounds))}" if args.grounds else "")
            + (f"-attack-{args.attack}" if args.attack else "")
            + (f"-{_character()}" if _character() != "thief" else ""))


def _character() -> str:
    """The character this process plays (gamelab.character: the harness restarts itself with GAMELAB_CHARACTER)."""
    from .. import character
    return character.active_id()


def _command_line(args) -> list[str]:
    """`run`'s command line for a run's options, but --seeds, --jobs and --save (what `ab` hands the base's filler)."""
    out = ["run", args.policy, "--preset", args.preset, "--minutes", repr(args.minutes), "--costs", args.costs,
           "--map", str(args.map)]
    if args.think:
        out += ["--think", repr(args.think)]
    if args.think_clock:
        out += ["--think-clock", repr(args.think_clock)]
    if args.ways is not None:
        out += ["--ways", str(args.ways)]
    for w in args.world:
        out += ["--world", w]
    if args.grounds:
        out += ["--grounds", ",".join(map(str, args.grounds))]
    if args.attack:
        out += ["--attack", args.attack]
    if _character() != "thief":
        out += ["--character", _character()]
    return out


def _run_args(args) -> tuple:
    """A run's arguments after the policy and seeds (cache_of's)."""
    return (args.preset, args.minutes, args.think, args.costs, args.ways, args.map, tuple(args.world), args.grounds,
            args.attack, args.think_clock)


def _cached(args) -> int:
    """`cached`: seeds 1 to --seeds of commit --sha's cache for these arguments, read (nothing is run), as a table, and
    saved as a run (--save; default the run's name and @SHA) for `compare`. Exit code 1 if seeds are missing."""
    from . import baseline
    try:
        sha = baseline.rev(args.sha, REPO)
        code = baseline.code_id(sha, REPO)
    except ValueError as e:
        print(e)
        return 2
    cache, why = cache_of(args.policy, *_run_args(args), code=code, commit=sha)
    if cache is None:
        print(f"nothing cached: {why}")
        return 2
    seeds = list(range(1, args.seeds + 1))
    missing = [k for k in seeds if cache.get(k) is None]
    print(f"{cache.folder}: {len(seeds) - len(missing)} of {len(seeds)} seed(s) cached (commit {sha[:12]}, code "
          f"{code[:12]})")
    if missing:
        print(f"missing: seed(s) {_seed_span(missing)}; fill them with the base's filler (README, Checking a change):\n"
              f"   python {_filler(os.environ.get('GAMELAB_BASE_FILL'))} --ref {sha} -- "
              f"{' '.join(_command_line(args))} --seeds {args.seeds}")
        return 1
    results = [cache.get(k) for k in seeds]
    print(table(results))
    name = args.save or f"{_name(args)}@{sha[:7]}"
    save(name, results)
    print(f"saved {FOLDER / (name + '.json')}")
    return 0


SPOT_KEYS = ("policy", "preset", "minutes", "think", "costs", "ways", "map", "world", "grounds", "attack",
             "think_clock")


def spot(results: list[dict], seeds: list[int], jobs: int = 1, say: Callable[[str], None] = print) -> bool:
    """A reviewer's spot check of a saved run: its `seeds` run again here with the arguments each result records
    (SPOT_KEYS), and each the same to the digit as saved, or the keys that differ. Whether all were."""
    from . import baseline
    by_seed = {r["seed"]: r for r in results}
    unknown = [k for k in seeds if k not in by_seed]
    if unknown:
        say(f"the run has no seed {_seed_span(unknown)} (it has {_seed_span(sorted(by_seed))})")
        return False
    groups: dict[tuple, list[int]] = {}
    for k in seeds:
        r = by_seed[k]
        groups.setdefault(tuple(json.dumps(r.get(key)) for key in SPOT_KEYS), []).append(k)
    same = True
    for _, ks in groups.items():
        r = by_seed[ks[0]]
        again = run_seeds(r["policy"], ks, r["preset"], r["minutes"], jobs, r.get("think", 0.0),
                          r.get("costs", "coded"), r.get("ways"), r.get("map", FIRST_MAP), tuple(r.get("world", ())),
                          tuple(r.get("grounds", ())), r.get("attack"), r.get("think_clock", 0.0))
        for k, got in zip(ks, again):
            got, want = baseline.plain(got), by_seed[k]
            if got == want:
                say(f"seed {k}: the same to the digit (exp_per_min {got['exp_per_min']}, touches {got['touches']})")
                continue
            same = False
            keys = sorted(key for key in set(got) | set(want) if got.get(key) != want.get(key))
            say(f"seed {k}: DIFFERS in {', '.join(keys[:12])}"
                + (f" and {len(keys) - 12} more" if len(keys) > 12 else "")
                + "".join(f"\n   {key}: saved {_short_value(want.get(key))}, now {_short_value(got.get(key))}"
                          for key in keys[:4]))
    return same


def _short_value(v, limit: int = 160) -> str:
    text = json.dumps(v)
    return text if len(text) <= limit else text[:limit] + " ..."


def _filler(fill: str | None) -> str:
    from . import baseline
    return fill or str(baseline.ROOT / "base.py")


def ab(args) -> int:
    """`ab`: this checkout (the candidate) against commit --base's cached baseline, paired seed by seed, from --seeds
    (12) seeds; 24, then 48 (--max-seeds) only while a headline difference (HEADLINE, and each --metric) is within 2
    standard errors and not zero, or has too few seeds to tell (ab_next, OPEN). A --metric that no seed's result has,
    on either side, is refused after the first seeds (exit 2; the names nearest it said). The base's missing seeds are
    run in its own checkout by the filler (--fill) with the environment its key resolved (baseline.filler_env) and
    cached there; the candidate's are run here (and cached, while its code is as committed). Says the seeds it used
    and why it stopped; both sides saved (--save NAME: NAME and NAME-base; default the run's name and -ab)."""
    from . import baseline
    try:
        base_sha = baseline.rev(args.base, REPO)
        base_code = baseline.code_id(base_sha, REPO)
    except ValueError as e:
        print(e)
        return 2
    cache, why = cache_of(args.policy, *_run_args(args), code=base_code, commit=base_sha)
    if cache is None:
        print(f"no baseline: {why}")
        return 2
    steps = tuple(s for s in AB_SEEDS if args.seeds <= s <= args.max_seeds) or (args.seeds,)
    if steps[0] != args.seeds:
        steps = (args.seeds, *steps)
    names = list(HEADLINE) + [m for m in args.metric if m not in HEADLINE]
    me, mine_why = baseline.identity(REPO)
    print(f"ab: {' '.join(_command_line(args)[1:])}\n"
          f"   candidate {REPO} ({'code ' + me[:12] + ', as committed' if me else mine_why + ': not cached'})\n"
          f"   base {args.base} {base_sha[:12]} (code {base_code[:12]}): {cache.folder}")
    base: list[dict] = []
    cand: list[dict] = []
    n = 0
    while True:
        target = next(s for s in steps if s > n)
        seeds = list(range(n + 1, target + 1))
        missing = [k for k in seeds if cache.get(k) is None]
        if missing:
            filler = _filler(args.fill)
            command = [sys.executable, filler, "--ref", base_sha, "--", *_command_line(args), "--seeds", str(target),
                       "--jobs", str(args.jobs)]
            if not os.path.exists(filler):
                print(f"the base's seeds {_seed_span(missing)} aren't cached and there's no filler ({filler}): run "
                      f"them in a checkout of {base_sha[:12]} with `run ... --cache`, or give --fill")
                return 2
            print(f"base: seed(s) {_seed_span(missing)} not cached: {' '.join(command[1:])}", flush=True)
            # (its environment the one this key resolved: its runs' keys the same, whatever lab/GPU there says)
            done = subprocess.run(command, stdin=subprocess.DEVNULL,
                                  env=baseline.filler_env(cache.meta["env"]["gamelab"]))
            missing = [k for k in seeds if cache.get(k) is None]
            if done.returncode != 0 or missing:
                print(f"the base's filler failed (exit {done.returncode}); seed(s) still missing: "
                      f"{_seed_span(missing) if missing else 'none'}")
                return 2
        base += [cache.get(k) for k in seeds]
        extra = [m for m in names if m not in HEADLINE]
        if n == 0 and (absent := [m for m in extra if not any(has_metric(r, m) for r in base)]):
            print(f"--metric {', '.join(absent)}: in none of the base's results; the candidate's are looked at "
                  f"after its seeds", flush=True)
        cand += cached_run(args.policy, seeds, args.preset, args.minutes, args.jobs, args.think, args.costs,
                           args.ways, args.map, tuple(args.world), args.grounds, args.attack, args.think_clock,
                           say=lambda line: print("candidate " + line, flush=True))
        if n == 0 and (unknown := [m for m in extra if not any(has_metric(r, m) for r in base + cand)]):
            known = sorted({k for r in base + cand for k in _paths(r)})
            for m in unknown:
                near = difflib.get_close_matches(m, known, n=3)
                print(f"--metric {m}: no seed's result has it, on either side (a typo?)"
                      + (f"; did you mean {', '.join(near)}?" if near else ""))
            return 2
        n = target
        rows = [difference(base, cand, m) for m in names]
        print(f"\nseeds {_seed_span(list(range(1, n + 1)))}:\n{ab_table(rows)}", flush=True)
        more, why = ab_next(rows, n, steps)
        if more is None:
            print(f"stopped at {n} seeds (1-{n}): {why}")
            break
        print(f"-> {more} seeds: {why}", flush=True)
    ok, lines = compare(base, cand)
    print("the bar (shown only): " + "; ".join(lines))
    name = args.save or f"{_name(args)}-ab"
    save(name, cand)
    save(f"{name}-base", base)
    print(f"saved {FOLDER / (name + '.json')} and {name}-base.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

