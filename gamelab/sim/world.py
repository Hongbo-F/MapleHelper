"""A deterministic stand-in for the game: her body, the monsters, stars, drops, HP and EXP on a real map
(mapdata.MapModel), in map units, driven by key events on a fake clock (see sim.fakes).

Her body moves as the game moves it, by motion.body's engine (the one the planner's constants come from),
on the world's 5 ms tick, and the game's 30 ms step (body.STEP: STEP_MS):

- the game sees an arrow pressed half a step sooner than body.Lags.on (66 ms) after its key and let go
  half a step sooner than Lags.off (68 ms) after (Lags.seen), and acts on what it has seen at its next step
  (on average Lags.on and Lags.off after the key), so a tap is as long as the game sees it (2 ms longer
  on average, a whole number of steps), and she turns when it sees the press; of the arrows it sees held,
  the one pressed last counts. The steps fall every 30 ms from a phase drawn from the seed (step_phase);
- on the ground, at each step, the arrow it sees speeds her up at 1400 units/s² to her walk speed (her truth: the
  client's formula at her profile's SPEED, physics.PROFILE_SHOE, or her own, Her.walk_speed: World.shoe, apart from her
  planner's, physics.SHOE, which her stat window's SPEED sets: physics.set_speed)
  (turning her round the same way), and none brakes her at 800 (body.ground_step over the step), and she
  runs the step evenly; walking past a platform's end she falls, unless a wall rises there;
- the jump key lifts her off at the step after the game sees it (Lags.jump, 65 ms, after the key on
  average), if she's on the ground then, with the kick: below 0.8 × her speed toward the arrow it sees, she
  gets that much more (body.takeoff). So a jump three steps into a run from standing leaves at 126. With
  down held it
  drops her through the platform physics.DROP_LAG later. A jump key that comes inside the attack lock is
  ignored, and so is one pressed while she's in the air (JUMP_TAKEN_LATE: or one the game takes then);
- in the air she flies body.fly's arc with the arrow the game sees, flown again from where she is when
  that arrow changes: the opposite one brakes her at 200 units/s², the one she flies with does nothing,
  and none drags her at 1 (100 once she falls at the cap). Walls stop her by the game's rule (body.wall_table:
  HeavenClient's, a wall joined to the floor under her just above it, and group 0's, the map's own, anywhere:
  Realism.zero_walls), and the map's sides (Realism.sides). She lands with the speed she has and slides on, unless an
  attack is playing: that stops her
  (jump-backs and jump-throws slid 1.4-2.9 units in game). Below the map's lowest floor she has fallen out
  of the map, and the run ends there (SimFellOut);
- up held at a rope's foot, or anywhere along it in the air, grabs it (its bottom Realism.rope_reach above her
  feet at most); up climbs at her climb (her truth: SPEED + 20 at her profile's SPEED, physics.PROFILE_CLIMB, or her
  own, Her.climb_speed: World.climb) onto
  the platform at its top (a short climb-over, flow.CLIMB_OVER), down climbs down at it too
  and lets go at its bottom; down on a rope's top grabs it from above; the jump key with an arrow jumps
  her off it sideways (ROPE_JUMP); down and jump on a floor whose foothold forbids it (forbidFallDown)
  does nothing, as on the bottom floor;
- monsters walk their platform as fast as the game data says (monster_data.speeds: 125 units/s for a pig,
  43.75 for a snail), deciding as the v83 client's AI does (AI_STEP: stand, left or right for 1-3 s, seeded),
  turning WALK_EDGE from its ends; those that jump (SimJumper: Tree Dungeon's Slimes) jump as it has them, off
  their platform too; a hit that doesn't kill pushes one KNOCK_MOB units back from her and stands it while it
  flinches (MOB_KNOCKBACK), never off its platform, if its damage reaches the monster's `pushed` in the game data
  (push_chance: any damage for the first map's kinds).
  A monster touches her when its body meets hers (HER_TOUCH: as the client tests it, a line 50 tall over her
  feet; Realism.her_box a box about them): the box of the frame it shows then in the game data, walking,
  standing or flinching after a hit, turned
  the way it faces (SimMonster.body), so a hopping slime is off the ground and taller for 38% of its walk.
  One the map has just spawned touches nothing until it has faded in (NEWBORN_SOLID_S after its birth, as live).
  A touch leaves her invulnerable for INVULNERABLE (1.5 s, as the sources and the frames have it), a MISS as much as
  damage. She evades it (a thief's "MISS": nothing else happens) with a chance inversely proportional to the
  monster's accuracy (game data: half a pig's touches, four in five of a snail's), or with her avoid known (Her.avoid)
  by this game's own hit check (game_hit_chance: her level, the monster's accuracy and level); the others take a share
  of her HP by the monster's touch damage (game data:
  about 1.6% for a pig, 0.3% for a snail, give or take 70%; or, her DEF known, by meowdb's formula: Her) and
  knock her back ~87 units on flat ground from standing (the client's push, 270 out and up, added to her own speed
  sideways, landing at her walk's speed at most: KNOCK_VX, KNOCK_VY, KNOCK_ADDS, KNOCK_LANDS_WALKING, as R1-R3's
  frames; the arrow she holds doesn't steer that flight, KNOCK_STEERS). At 0% HP she dies, and the
  run ends (SimDied); up held at a portal (mapdata.PORTAL_BOX: from 100 above it to 25 below, in the air too, as
  HeavenClient has it) takes her through it, out of the map, and the run ends too (SimLeftMap);
- the attack picks its target when it starts, physics.JUMP_LAG after its key (the same input lag as a
  jump's lift-off), from where she is then, and its damage is the server's actions.DAMAGE_SHOWS_MS (0.29 s) after
  the key, on that monster (attack.reach), whatever the distance, before her star even leaves (actions.STAR_DELAY_MS:
  in game it's then seen flying up to 0.6 s more): a kill's EXP and its monster dying then, a hurt one's bar lower,
  its flinch and push (when the game shows those isn't measured). Until 2026-10-26 it landed at STAR_DELAY_MS, where
  live a kill's gain showed 0.15-0.46 s after the key. On the ground the attack takes one of the claw's stances
  (reach.CLAW's, drawn per attack, seeded), and the nearest monster (by distance, feet to feet) whose
  body reaches into its box (reach.claw_takes: cut to reach.CLAW_FAR ahead of her, on her level or not) is
  swiped (a fifth of a star's damage); else the star's search picks (reach.search_band: of the monsters
  whose body overlaps one of its bands, the first band's, one of those drawn at random, seeded, as the
  client's order there, by when each entered the map, can't be seen), so off her level a star needs room
  sideways, by the monster's body (right under or over her, none is hit).
  In the air, the star's search from her feet then (AIR_PICK 'search', the default from 2026-10-27, as
  reach.AIR_MODEL: no claw and no area; with Noise.air_squash a body's top is drawn lower for it, with
  air_short and air_under a monster on her platform short of the first band or under its window takes it now
  and then, and with air_turn_lost an arrow in the air may not turn her: TREE_AIR on Tree Dungeon), or the
  nearest whose body reaches into the attack's area (its near side within reach.STAR_RANGE ahead, from
  reach.AREA_DOWN below to reach.AREA_UP above her feet, wherever she is then) or into its band's window of the star's search from her feet then ('widening': reach.air_takes as
  it was) takes a star (with 'area', the area alone; with 'v83', the claw and the search as from the ground,
  from where she is then). From the ground the Lucky Seven key swipes with the claw as the attack key does, no
  cast's MP taken (Her.claw). For all of these, bodies are their kind's half width either side and as tall as
  their kind walking (monster_data.MONSTER_HEIGHT), taller while they flinch after a hit
  (monster_data.FLINCH_S, flinch tops from the game data); not their frames (World.top says why). So a
  jump throw's attack, starting 46 units up, passes over a snail. All of these pick by the bodies alone,
  whatever the terrain; then, as the client does, the line to a star's pick alone is checked
  (World.sight_stop, reach.wall_between by reach.LINE: by the v83 line, the default since the user saw her
  own platform stop her stars in game on 2026-10-21, the line from reach.SHOT_H above her feet now to where
  the star hits the monster's body, which a wall of the foothold group of the platform she last stood on
  stops, gone into at its outer face, and a floor it goes down through; by the level line, such a wall where
  it crosses the level line reach.SHOT_H above her feet). Stopped, the attack hits nothing, and no
  other monster takes the star: from platform 4 a throw whose search picks a monster on 2 hits nothing
  (live, no star from 4 ever reached one on 2). Kills give EXP
  (model.mob_exp) and, 2 s later, drops on its platform; a killed monster is seen fading out where it
  died as long as its death animation runs (MobBody.die_s: a pig 0.66 s, a snail 1.26), and respawns 8 s
  later where it spawned (Realism.respawn), or, with Realism.respawn_tick, at the map's pass, which fills the map to its
  cap (the server's formula: MapModel.mob_cap) at random spawn points with no monster within 100 units (TICK_S,
  SPAWN_CLEAR: the server's as OrionAlpha has it);
- the pick-up key takes the nearest drop within reach (items only while her inventory has room), the
  HP potion key gives 30% HP; her pet (Her.pet), if she has one, drinks a potion for her right after a touch leaves
  her under its warning. (Until 2026-10-26 the chair key sat her down, +3% HP a second, until an arrow stood her up:
  the stack rested on it.)

Her HP, her level and EXP when a run starts, her evasion, her pet and what her attack does (her regular attack's star,
or Lucky Seven as thrown live: LUCKY_SEVEN) are `Her`'s: LEVEL_19 on the first map, LEVEL_23 on West Rocky Mountain II,
LEVEL_24 on Tree Dungeon, Forest Up North II. A monster of a kind the first map
hasn't takes as many stars as its HP and defence in the game data say (hits_of), and its body is as tall and wide as
its frames there (_body_of).

`Noise` (off by default) adds the errors of the real game, each one seeded and tied to the moment it
happens, not to how often anything was looked at: stars that miss, walks that stop a few units early or
late, lags spread as measured (so a jump keyed 80 ms after the arrow leaves with the kick or without it),
short arrow taps that turn her without walking her, her minimap dot read to whole pixels, and spells when
her name tag or her dot can't be seen.

Time is kept in whole milliseconds on a 5 ms tick, so the state at any moment depends only on the key
events, never on when or how often it was looked at.
"""

import bisect
import copy
import functools
import json
import math
import os
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gamedata.paths import OUT
from maplehelper.config import GameKeys
from maplehelper.engine import LEFT, RIGHT
from maplehelper.game import COMBAT

from .. import directions
from ..attack import reach
from ..attack.danger import touch_taken
from ..behavior import pet as pet_care
from ..character import Search
from ..classes.magician import SEARCH as CLAW_SEARCH
from ..control.actions import DAMAGE_SHOWS_MS, STAR_DELAY_MS
from ..localization import monsters as monster_data
from ..localization.mapdata import PORTAL_BOX, PORTAL_EXIT, MapModel, Platform, Rope
from ..motion import bodies as her_body
from ..motion import body, flow, physics
from ..motion.physics import DROP_LAG, GRAB_SURE, JUMP_HEIGHT, JUMP_LAG, LAGS, ROPE_REACH
from ..perception import drops as drop_data
from ..perception.kinds import kind_of

TICK_MS = 5              # the world's tick, which is also the engine's own step in the air (body.TICK)
STEP_MS = round(body.STEP * 1000)  # the game's step: it acts on keys, and sets her speed on the ground, every
                                   # 30 ms (body.STEP)
# Her climb, up and down (World.climb, her_climb): the client's SPEED + 20 at her profile's SPEED
# (physics.PROFILE_CLIMB: the thief's 121, the magician's 120; she climbed 119.9 up and 119.6 down on 2026-11-02). Until
# 2026-11-02 up at the planner's flow.ROPE_SPEED (130, the minimap's traces) and down at its physics.CLIMB_SPEED (117).
# A climb ends in an animation onto the platform at the rope's top: flow.CLIMB_OVER is how long after the
# top she makes a standing start, which the game then sees an arrow already held for Lags.on later. So she
# stands still for both (0.30 s), less the wait for the game's next step that comes after it (12.5 ms on
# average from a tick), and with the arrow held from before the top is running at full speed 0.35 s after
# it, as in the two climbs traced on the minimap (flow).
CLIMB_OVER = flow.CLIMB_OVER + LAGS.on - (STEP_MS - TICK_MS) / 2000
ARMS = ROPE_REACH - JUMP_HEIGHT  # she grabs a rope this far above her feet (by default: Realism.rope_reach)
GRAB_X = 10.0            # ... within this of its x
ROPE_TOP_REACH = 15.0    # down grabs a rope whose top is within this of her platform
REGRAB_AFTER = 0.3       # s after leaving a rope before up grabs one again
# Jumping off a rope sideways: the decompile's 1.3 × her walking speed out and half a jump's speed up
# (194 and 277 units/s: Shoe.rope_vx, rope_vy). The one rope jump traced on the minimap (`ropejump`) went
# 178 ± 18 units/s sideways and rose about 16-20 units, which fits these better than what the simulator
# used before (149 out, 220 up: a 12-unit rise), by about 2 sd: weak, so False puts those back.
ROPE_JUMP_SHOE = True


def rope_jump(shoe: body.Shoe) -> tuple[float, float]:
    """A jump off a rope sideways with `shoe` (the world's: World.shoe), out and up, units/s (ROPE_JUMP_SHOE)."""
    return (shoe.rope_vx * shoe.vmax, shoe.rope_vy * shoe.vjump) if ROPE_JUMP_SHOE else (shoe.vmax, 220.0)


ROPE_JUMP = rope_jump(physics.PROFILE_SHOE)  # (her profile's: the world's own is World.rope_jump)
# Her body for the world's touches (Realism.her_box), (half its width, its height) about her feet. By default the
# client's touch test (from 2026-11-02): RustMS's copy of HeavenClient, aligned with the v83 client, has a monster touch
# her where its frame's box (Mob::is_in_range: the bounds of the frame it shows, as SimMonster.body has them) meets
# her as a line 50 tall, from where she was a frame before to where she is, with no width of its own
# (MapMobs::find_colliding; D:/MapleHelper-scratch/spawn-fade/mech/rustms). The world (World._touch) tests the line at
# her x now alone, the box's edges not touching: from one 5 ms tick to the next she goes 0.75 units at her walk (1.35 in
# a knock's flight), so the sweep and the edge are left out; the -22..22 x 65 box of the client's
# FUN_0045183b (MobAttackRules.h `user_body`) is where a monster's own attacks land, which Tree Dungeon's don't have.
# Live it fits the touches as her invulnerability ends: of BugBaby's grinds of 2026-10-30 (02:03-02:49, 16:31-17:37:
# the perception service's tracks; D:/MapleHelper-scratch/recfix/sim/live_boxtest.py), with the nearest seen monster of
# her platform 0-25 units off her (middle to middle) 1.40 s after a touch the next touch came by 1.75 s 81% of the time
# (n 78), 25-35 off 33% (n 27), 35-45 off 46% (n 24), 45-60 off 15% (n 34); in the world with 15 either side of her
# 100%, 100%, 89% and 59% (12 runs of the standard grind, world truth: sim_boxtest.py), with this line 98%, 75%, 56% and
# 53% (what's left there is her chasers walking in). Until then 15 either side and 55 tall (HER_HALF, HER_HEIGHT: the
# harness's `--world wide-body`; the stack's hop was timed in game with the 15, flow.HOP_HALF), or the planner's
# (`--world client-body`): 22 either side, 65 tall (motion.bodies, which the stack plans by: her trips,
# attack.danger), the safe side for keeping clear.
HER_TOUCH = (0.0, 50.0)
HER_HALF = 15.0
HER_HEIGHT = 55.0
HER_BOX_CLIENT = (her_body.HER_HALF, her_body.HER_TALL)
# A jump key the game takes in the air does nothing. When it takes one isn't measured: the 18 jumps recorded
# lifted off 45-95 ms after their key, and a key pressed while she was in the air (or held from before) never
# jumped her again on landing. Taken when it acts on it (Lags.jump after the key, as the world's clock
# counts), a key pressed in the last 65 ms of a fall jumps her once she lands; taken as the key goes out,
# it is lost. False takes it as the key goes out, as the simulator did before it ran on the engine: the
# harsher of the two for a plan that presses the next jump early, so a plan that copes here copes in game.
JUMP_TAKEN_LATE = False
STAR_DELAY = STAR_DELAY_MS / 1000
ATTACK_LOCK = COMBAT.attack_lock_ms / 1000
# An arrow pressed while her attack plays doesn't move her (Realism.lock_arrows). In game on 2026-10-21, of 13
# chains sent within 0.35 s of the look after a throw (their first arrow about 0.3 s after its attack key), 6
# first moved her 0.73-0.92 s after they went out, about 1.0 s after the key, with the arrow still held; and 2
# never did, a step let go before then (two walk-offs off platform 3's left end onto 4, the arrow held 0.48-0.52 s
# from about 0.3 s after the key: her dot never moved, though the next look had her 7 units on in the first, and
# the walk-off sent again 0.9 s later went). With the first arrow held back to 0.6 s after the key she still set
# off late: timed by when her dot went off the end of the platform in the walk-offs (less the fall to her dot's
# first drop, and her run to the end by the engine), 17
# whose first arrow went out 0.59-0.62 s after the key were acted on 0.75-0.88 s after it by her dot (the median
# 0.81: 0.21 s after going out, where an arrow with no attack before it is 0.09 s, the median of 73), so about
# 0.79 s after the key by the game's clock; the 3 that went out 0.79-0.82 s after the key went at once (0.03-0.09).
# (By her dot's first move, a pixel per 11 units, it read as 0.85 s.)
# So an arrow pressed (as it goes out) less than ARROW_LOCK after the key of an attack the game took is seen
# ARROW_HELD after that key, if it's still held then, and else never: the game acts on it at its next step, about
# 0.79 s after the key on average (ARROW_LOCK is ARROW_HELD less the time the game takes to see an arrow,
# Lags.seen, so a press just after it is seen as late). Fitted so, of the chains for the jump from 16 up onto 14
# held back only to 0.6 s (nav_jump_after_kill, seeds 1-16), 14 of the 15 that kill the pig fall to 25 as 8 of 10
# did in game; with 0.835 (fitted to the dot's first move) 5 did and 10 were cut short on 16. Those pressed early
# in the attack seemed to wait longer (about 1.0 s, by the dot's first move), which no plan does now; none was
# pressed from 0.63 to 0.78 s after the key, which is unmeasured. Arrows pressed before the key and held through
# it aren't measured (no plan does that): they move her as ever. (0.55 and 1.0 s before the hold-back's 0.6 s was
# seen to fall short; 0.835 until the review of 2026-10-21.)
ARROW_HELD = 0.77
ARROW_LOCK = ARROW_HELD - body.Lags.seen(LAGS.on)  # 0.719
# Her HP: her HP bar read "HP[642/642]" in game (2026-10-18, at level 19; maplehelper.ocr.parse's test).
MAX_HP = 642.0
# A touch takes TOUCH_SHARE of the monster's touch damage (PADamage in the game data: snail 3, slime 13,
# pig 15, Orange Mushroom 17) from her HP, give or take TOUCH_SPREAD of that (uniform, seeded by the
# touch's tick, like Noise). Fitted to lab/hits.jsonl: its 100 hits of up to 5 px on the basin and platform
# 21, where only pigs and slimes spawn, took 1 px of the 147 px HP bar 30 times, 2 px 32, 3 px 28, 4 px 9
# and 5 px once (2.19 px on average: 9.6 HP, 0.68 of their PADamage); ±70% gives 26 / 33 / 30 / 9 / 0
# (and 2 too small to show). So a pig takes 1.6% of her HP on average, a slime 1.4%, an Orange Mushroom
# 1.8%, a snail 0.3%. How the game takes her defence off isn't known: a share of PADamage is the simplest
# form that fits, and no touch by a snail or an Orange Mushroom alone has been logged.
TOUCH_SHARE = 0.68
TOUCH_SPREAD = 0.7
TOUCH_DAMAGE = 0.014     # a touch by a monster without game data: the logged hits' median (2 px, 1.4%)
# ... or (Her.defence) as meowdb's page on this game's formulas has it (meowdb.com/msclassic/formulas/damage-taken,
# read 2026-10-24): raw = PADamage x (1.2 + 0.2 u), taken = raw x (1 - DEF / (DEF + 300 + 2 raw)) - (DEF / 25 + her
# level / 10), cut to whole HP, at least 1. Her DEF isn't known: DEFENCE is what makes that formula give the logged
# hits' 9.56 HP on average at level 19 (pigs and slimes half each: 9.48; 70 gives 9.88, 80 9.09), taken as hers at 23
# too. Against a share it's the shape that differs: little for a weak monster, more for a strong one (at level 23,
# a Red Snail 3.6 HP, a Stump 5.8, a Dark Stump 31.3, an Axe Stump 78.9, where 0.68 of their PADamage is 6.1, 7.5,
# 23.1 and 51.0), and its spread is the roll's ±8%, not ±70%.
DEFENCE = 75.0
# A thief's evasion: a touch shows "MISS" and takes nothing, without a knockback, with a chance of
# EVADE_ACC over the monster's accuracy (acc in the game data: snail 33, slime 53, pig 54, Orange Mushroom
# 62). Inversely proportional is the form MapleStory's dodge rule is usually given in (not measured here).
# Fitted to the 13 touches logged hopping over monsters (lab/hop_log and hop_log2: 7 MISS, 6 with damage),
# pigs by their speed (most walked at the tracker's 70 units/s clamp, which only pigs and Orange Mushrooms
# reach): a pig's touches are evaded half the time, a slime's 51%, an Orange Mushroom's 44%, a snail's 82%.
# Seeded by the touch's tick, like Noise.
EVADE_ACC = 27.0
EVADE = 0.5              # ... a monster without game data
# ... or, with her avoid known (Her.avoid), this game's own hit check (game_hit_chance: meowdb's page on this game's
# damage taken, its 'incoming-hit' calculator, meowdb.com/msclassic/formulas/damage-taken, read 2026-10-28; the same in
# a reading of this game's client binary, ohmi69/osms_datamine_dashboard tabs/formulas-mob-hit.js, "outcome routine
# 0x140268140"; no job term): with her level L over the monster's G = max(0, L - its level), its accuracy as
# A = acc x 100 / (5 (G + 51)) against her avoid as E = avoid / (1 + avoid / 80) / (1 + G / 40), a roll R uniform in
# 1 +- f, f = 0.15 + 0.2 / (1 + exp((A - E) / 12)), hits where A R >= E (where E is past A (1 + f) a rescue roll of
# 0.03 exp((A - E) / 18), within 0.02-0.03, instead), and one that failed hits 8% of the time all the same (a MISS at
# most 90.2%). Her avoid isn't on her HUD (her stat window's): fitted to her MISS shares on the frames of 2026-10-24..28
# (README, Touches: by eye, the popups over her) by most likelihood, one avoid for both kinds at a level, which the
# formula fits within their counts' spread: at level 25 58.1 (one sd 55.9-60.4; Green Mushrooms MISS 36 of 71 touches,
# 52.5% by the fit, Horny 13 of 46, 26.3%), at 28 70.6 (68.5-72.8; Green 109 of 152, 70.9%; Horny 35 of 83, 43.9%).
# Where EVADE_ACC had a Green's touches MISS 34% and a Horny's 29% at any level.
LEVEL_25_AVOID = 58.0
# s a touch leaves her invulnerable, a MISS as much as damage, every touch starting it afresh: as the sources have it
# and the frames agree once read by their frame steps (every world's from 2026-11-03, the user's "yes" to the touch
# study's two changes as the defaults; the study, D:/MapleHelper-scratch/recfix/touch-study/verify-invulnerability-1.5:
# gaps.py, contact.txt): RustMS invincible.set_for(1500), meowdb's "1.5 s", a MISS the same. Live's re-touches at the
# floor, 88 of them on four 15 fps sources (R1-R3, BugBaby's October hunts and walk check), are 21, 22 and 23 frames
# apart 0, 40 and 48 times, where 1.45 s gives about 22%, 70% and 8% and 1.5 s 0%, 46% and 54%; his 20 fps telemetry's
# at a median 1.501 s; on R1's frames 1883-1884 a Horny on her line 1.400-1.467 s after the touch before touched her only
# at 1885. The stack plans by perception.touches.INVULNERABLE_S (the navigator's immune windows, Senses), left at 1.45
# (a live change, to be measured on its own): the world is the game, the stack's figure its belief.
INVULNERABLE_SOURCES = 1.5
INVULNERABLE = INVULNERABLE_SOURCES
# ... and as measured live (Realism.invulnerable; the world `--world walkers` has it, sim.harness): first from the HP
# drops alone (none of 800 within 1.25 s of the one before, 2026-10-25/26: D:/MapleHelper-scratch/land/diag-move-away/
# touches.py), 1.25 until the popups were read (2026-10-28), then the frames' least gap, now the same.
INVULNERABLE_LIVE = INVULNERABLE_SOURCES
# ... and as every world had it from 2026-10-28 until 2026-11-03 (`--world invulnerable-1.45`, to price the change):
# the frames' least gap, 1.435 and 1.441, two 22-frame gaps their timestamps' jitter pulled down (the stack's
# perception.touches.INVULNERABLE_S, which every world took then); until 2026-10-28 1.0 coded, and 1.25 from the HP drops
# alone.
INVULNERABLE_LEAST_GAP = 1.45
# s a touch's popup over her shows (sim.fakes.FakeTouches reads it off the frame's Screen.popups): v83's 400 ms opaque
# and 600 fading (RustMS DamageLayout.h:16-24); unconfirmed in this game. The stack tells a touch the frame it first
# shows.
POPUP_S = 1.0
# Monsters chasing her once she hits them (Realism.chase): aggro, as the game has it. A player's hit gives the monster to
# that player's client "with chase" (OrionAlpha, a replica of Nexon's early Korean server: Mob.onMobHit,
# LifePool.changeMobController, MobCtrl.Active_Req; once its controller hasn't hurt it for 5 s, to whoever else did in
# those 5 s: Mob.onMobMove), and that client chases its player (HeavenClient, a reimplementation of the v83 client from
# its code: Mob::chase, FUN_007f73d8) AGGRO_S from then, 3 to 4 minutes: every CHASE_THINK_MS decision it decides
# which way to walk, at her, or on the way it chose while she's within AGGRO_PASS across (so it walks past her that far
# and more before turning back, and where its platform's end turned it, back to that end: Mob::think), never standing, at its walk speed (info/chaseSpeed is for a kind with a chase animation:
# none of Tree Dungeon's or the first map's kinds has one, nor info/firstAttack, a kind's chase on sight: gamedata),
# along its own platform wherever she is (turning at its ends as its walks do). Each hit of hers starts it anew, after
# the hit's flinch (knock). Unhurt it walks as ever. The user, 2026-10-31: "Mobs move on their own until you hit them,
# then you draw their aggro and they will charge towards until certain period, hit the mob again refreshes that
# period". Until then any monster on her platform within 300 of her walked its stretches at her by chance, 0.2 unhurt
# and 0.6 hurt, stood 0.5-1.5 s on her and walked a stretch away (CHASE_TREE, fitted to the perception service's tracks
# of 2026-10-27: within 100 of her, monsters walked at her 50% of the time unhurt, 63% hurt): the unhurt ones' 50% is
# their own walk's (the world's walks with nobody there had 46%).
AGGRO_S = (180.0, 240.0)
AGGRO_PASS = 100.0
# The monsters' AI as the v83 client runs it (RustMS's copy of HeavenClient, annotated with the client's functions, read
# 2026-10-31: Mob::step_body, think, walk, chase; the user, 2026-10-31: "Yes, align them"): it steps every AI_STEP s
# (FUN_007f72d9), and a decision holds a whole number of its steps: unhurt (r + 1000) // 90 of them, r uniform in
# 0-1999 (1.0-3.0 s), standing, walking left or walking right, a third each (FUN_007f8025, FUN_007f81c8); chasing
# (r + 800) // 90, r in 0-399 (0.72-1.17 s). A hit (Realism.chase) is a chase anew (Mob::chase: its timer and its way
# reset): at her from the hit's flinch's end. Until then the simulator's own: standing 35% of its stretches (0.5-2.5 s)
# and walking 65% (1-4 s), its chase every 0.8-1.2 s to its aggro's end, after an end's turn on the way it was going, and
# a hit while it chased kept its way.
AI_STEP = 0.09
IDLE_THINK_MS = (1000, 2000)   # (least, span): (r + least) // 90 steps, r uniform in 0 to span - 1
CHASE_THINK_MS = (800, 400)
# A walking monster (move ability 1: no jump, no flight) turns WALK_EDGE along the foothold at its platform's end from
# the end (FootholdTree::get_walk_edge, FUN_007f8d64: localization.monsters.walk_edges, the stack's predictions' too),
# where the simulator had its body's half width (28 for Tree Dungeon's kinds).
WALK_EDGE = monster_data.WALK_EDGE
walk_edges = monster_data.walk_edges
# Her landings as the live diagnosis counted them (World.land_log): an attack played in the flight if keyed from
# ATTACK_BEFORE_TAKEOFF_MS before she left the ground (a jump throw's key comes 40-120 ms after the jump key, and
# her feet leave the ground about JUMP_LAG after it), and a fall within KNOCK_FALL_MS of a knock's landing the knock's
# (in the harness's runs, 15 of the 76 'trip' landings within 105 of a monster in 8 runs took off 0.1-0.3 s after a
# knock's landing: its slide, or a step's arrow held through it, took her off an end; D:/MapleHelper-scratch/land/fix/
# landdiag.py).
ATTACK_BEFORE_TAKEOFF_MS = 150
KNOCK_FALL_MS = 500
# Its knockback, the world's own from 2026-11-02 (Realism.knock), apart from the planner's (motion.physics.KNOCK_VX,
# KNOCK_VY: the navigator's KNOCK_AIR_S): away from the monster and up at 270 units/s each (KNOCK_VX, KNOCK_VY: the v83
# client's push as RustMS's port of it has it, "away from the mob and up at 270 px/s", rising about 18 and landing about
# 73 away from standing; D:/MapleHelper-scratch/spawn-fade/mech/rustms/agent_docs_client.md.txt), 0.27 s in the air on
# flat ground; the push sideways added to the speed she has, not past 270 (KNOCK_ADDS: the port's "each axis adds to the
# speed without going past the push"), so walking at the monster at 150 she flies off at 120; and landing with her speed
# along the ground at most her walk's (KNOCK_LANDS_WALKING), braked from there as ever: from standing ~73 units in the
# air and a ~14 slide, ~87 in all. R1-R3's lossless 15 fps frames agree (the replay kit's dumps of master 40ba190's
# perception on them: D:/MapleHelper-scratch/recfix/sim-fix/knock_fit.py, knock_fit.txt; the 9 touches that hurt her
# standing and landed her where she stood): 18 units a frame in the air (270 a second), 7 of the flights fitted at
# 262-270 out (the others 200 and 250), all 9 at 274-280 up (a median rise of 19.1 seen: 270's is 18.2), landing a
# median 73.4 away; then 11-19.5 more (a median 15.1, 5 of 7: one slid 6, one walked back) at 8, 4 and 2 units a frame,
# as her walk braked at 800 slides (14, at 8.2, 4.5 and 2.7), where 270 braked by the engine (1600 down to her walk,
# then 800) slides 30; 84-93 in all; the 12 hurting touches with her walking at the monster at 122-157 a second over the 2 frames before flew her off
# at 24-188 (a median 93; knock_moving.py, knock_moving.txt), the 26 standing at 196-274 (24 at 249-274). Until then
# the planner's 180 and 220 set outright (KNOCK_PLANNER: `--world planner-knock`), ~40 in the air (0.22 s) and a ~19
# slide, ~58 in all, fitted to hits.jsonl's pushes on her own level (quartiles 21 / 58 / 87, from the looks, not
# frames). Those were all hits with no key pressed since the look before (Hunter._log_hit), so what an arrow held does
# to the flight isn't measured (the walking ones' 93 against the 120 the push added gives would be the opposite arrow's
# braking at 200 units/s², held through the flight; their keys aren't in the dumps). False: the arrow does nothing until
# she lands, as the fit had it (and the simulator before it ran on the engine); on the ground it walks her as ever.
KNOCK_STEERS = False
KNOCK_VX, KNOCK_VY = 270.0, 270.0
KNOCK_ADDS = True
KNOCK_LANDS_WALKING = True
KNOCK_PLANNER = (physics.KNOCK_VX, physics.KNOCK_VY, False, False)  # (vx, vy, KNOCK_ADDS, KNOCK_LANDS_WALKING) as until
                                                                    # 2026-11-02
HITS = {"pig": 3, "orange-mushroom": 3, "slime": 2, "snail": 1}  # HP in stars
# A kind HITS doesn't have takes as many stars as hers need on average to kill it (stars_to_kill, rounded): a star of
# hers did 30-70 damage on the first map (a claw swipe 9-11: README, Attack; at level 23 not measured yet), less the
# monster's defence as meowdb's page on this game's formulas has it (meowdb.com/msclassic/formulas/damage, read
# 2026-10-24: damage x 100 / (DEF + 100), DEF its PDDamage in the game data, 30 for every stump: their stars do
# 23-54), and no level penalty (none of them is above her level). So a Red Snail (68 HP) takes 2 stars, a Stump
# (133 HP) 4, a Dark Stump (236) 7, an Axe Stump (371) 10. HITS' own kinds stay as they were fitted (by this a slime,
# 115 HP and DEF 10, would take 3).
STAR_DAMAGE = (30, 70)
HITS_UNKNOWN = 2         # ... and a kind without game data
# A hit that doesn't kill (a star or a swipe) pushes the monster away from her at KNOCK_MOB_V, slowing at
# KNOCK_MOB_DECEL on the ground (v83's push, 130 px/s, slowed by 40000 / its mass of 100: Mob::flinch_body,
# step_body): KNOCK_MOB units in all over KNOCK_MOB_S (21 units, 0.33 s), stopping at its walk's end. It faces
# her and stands in its hit1 frame until monster_data.FLINCH_S after the hit (0.6 s, that frame's delay), its AI
# waiting the hit out; then the rest of its decision standing, or hit with Realism.chase at her at once. In the game
# data `pushed` is 1 for all four kinds: any damage pushes them. Measured once: a slime walking at her, swiped on
# the basin floor (lab/recordings/do_attack2, static camera, 30 fps), went 13 half-px (18 units) back over about
# 0.25 s and stood about 0.6 s before it walked at her again (until 2026-10-31 the push: KNOCK_MOB 18, KNOCK_MOB_S
# 0.25, evenly, facing as before it, its next decision drawn after the flinch); a pig hit in `fight1` lost about 80
# units of its walk (moving camera, so no distance). False puts the monsters back to walking through hits. A
# walker stays on its platform: none is pushed off an end (pigs, which jump, have been seen on platforms where none
# spawn: SimJumper).
MOB_KNOCKBACK = True
KNOCK_MOB_V = 130.0
KNOCK_MOB_DECEL = 400.0
KNOCK_MOB = KNOCK_MOB_V ** 2 / (2 * KNOCK_MOB_DECEL)
KNOCK_MOB_S = KNOCK_MOB_V / KNOCK_MOB_DECEL


def think_steps(rng: random.Random, ms: tuple[int, int]) -> int:
    """A decision's AI steps (AI_STEP each): (r + least) // 90, r uniform in 0 to span - 1 (IDLE_THINK_MS,
    CHASE_THINK_MS)."""
    return (rng.randrange(ms[1]) + ms[0]) // 90


def pushed_x(start: float, to: float, tau: float) -> float:
    """Where a push from `start` toward `to` (as far as it goes) has a monster `tau` s after the hit."""
    went = KNOCK_MOB_V * tau - KNOCK_MOB_DECEL * tau * tau / 2 if tau < KNOCK_MOB_S else KNOCK_MOB
    return start + math.copysign(min(went, abs(to - start)), to - start)
CLAW = 0.2               # a claw swipe's damage, in stars (9-11 against a star's 30-70)
CLAW_DAMAGE = (9, 11)
STAR_MEAN = sum(STAR_DAMAGE) / 2  # a star's damage on average: the unit of a monster's HP with Her.spread
# What one of her attacks does, in stars (Her.stars): 1, her regular attack's star as above (and a swipe its CLAW of
# one); LUCKY_SEVEN, Lucky Seven as she threw it live on Tree Dungeon, Forest Up North II (tour --loop --skill,
# 2026-10-24, 26.9 minutes): 3.85 throws a Horny Mushroom's kill and 2.95 a Green's, by the kills her EXP gains named,
# where the regular attack's 9 and 5 stars (HITS' fit, stars_to_kill) take 12.4 and 6.9 with preset game's misses. At
# 3 stars a cast they take 4.1 and 2.8 there. Lucky Seven has no stab of its own up close (the stab is the regular
# attack's: README, Attack), so with it a swipe's pick takes a cast's damage too. With `stars` it's every attack's, and
# its MP isn't kept; as Her.skill, the Lucky Seven key's alone, its MP kept (below).
LUCKY_SEVEN = 3.0
# Her MP (Her.mp; kept from 2026-10-24, when the stack began choosing Lucky Seven or the regular attack a throw:
# behavior.attacks). With Her.skill set, the Lucky Seven key's attack is Lucky Seven: SKILL_STARS stars of Her.skill /
# SKILL_STARS of the regular star each, CAST_MP a cast, and with less MP than that the key throws nothing (no attack, no
# lock), as in game. She regains MP_REGEN a second all the while, and her pet (Her.pet_mp) drinks an MP potion of hers,
# MP_POTION, when a cast leaves her under its warning. All from the live telemetry of 2026-10-24 on Tree Dungeon
# (level 24, 601 MP):
# - MP_REGEN: her bar rose 92.6 MP in 286 s of the 20:57 run (the regular attack alone: no casts), a px (5.8 MP) about
#   every 20 s;
# - CAST_MP: attack.choice.cast_mp at her Lucky Seven's level (10: 11 MP); her bar fell a median 11.4 a cast (99 casts);
# - MP_POTION: the median rise of her bar over 97 of the pet's drinks (in two clusters, 127-145 and 173-203);
# - PET_MP: a warning in System Options' 5% steps; the reading before each drink was 0.058-0.067 of her bar (35-40 MP),
#   a cast after it under 5%.
# At level 25 (from 2026-10-25: LEVEL_25) her own: her Lucky Seven's level 16, 14 MP a cast (Her.cast_mp, as
# attack.choice.cast_mp has it; her bar fell 14.05 MP a cast over 364 one-cast stretches of the hunts of 2026-10-25 23:43
# to 01:03), and her pet's MP potions the 100-MP Blue Potions she carried then (Her.mp_potion: the first 20 drinks of the
# hunt of 23:43 each 16.4-16.8 px of 104 of her 601-623 MP, 95-100; analysis-1025/mp-attack-check/drinksize.py).
SKILL_STARS = 2
CAST_MP = 11.0
MP_REGEN = 0.32
MP_POTION = 185.0
PET_MP = 0.05
# Her MP bar is drawn this long after a cast's key without the cast's MP, and the drink of her pet's it brings
# (World.mp_drawn: what her looks read; the cast is taken off her MP at its key, as the game decides on it then): live
# on 2026-10-25 (the hunt of 23:43) a cast's MP was first seen off her bar 0.21-0.53 s after its key, a median 0.35,
# over 24 isolated casts, her looks then 5-10 a second (D:/MapleHelper-scratch/analysis-1025/mp-attack-check/
# nodrop.py): the server's, as a kill's gain (actions.DAMAGE_SHOWS_MS). Until the review of 2026-10-26 it was drawn at
# the key, and her reckoning took a cast off a frame shown under 0.21 s after its key that showed it already, one cast
# low (behavior.attacks; right live): 2.4 regular throws a run for 'MP' with a cast's MP hers.
MP_SHOWS_MS = DAMAGE_SHOWS_MS
# The magician's Magic Claw (Her.magic, SimMagic; from 2026-11-02, the roadmap of the day's three recordings, its item 8:
# D:/MapleHelper-scratch/rec-analysis/plans/ROADMAP.md, plan-simulator.md S5, critique-simulator.md), as the game casts
# it, not as her stack believes it (classes.magician's placeholders, attack.magic's swaps of attack.reach), so the
# world shows where her stack's belief is off. RiceBaby at level 32 on Tree Dungeon, master 40ba190's grinds of
# 2026-11-02 00:09-00:53 (R1-R3, D:/MapleHelper-scratch/golden-clips/ricebaby-grind-20261102-*: lossless frames at 15 fps
# and the hour's telemetry):
# - its pick: the client's search for an aimed spell (classes.magician.SEARCH: from 50 ahead of her to 300 in bands 20
#   wide, band k's window 28 -/+ (50 + 20 k) // 4 over her feet, the first band any body overlaps, one in it drawn),
#   through walls and floors alike (no line: the user, 2026-10-27, settled), from the ground only (a press in the air
#   does nothing: no MP, no lock; the user, settled), never behind her nor within its first 50, no swipe up close;
# - CLAW_HIT: a hit's damage before the monster's defence, a whole number from 157 to 181, two a cast on the one
#   monster (attackCount 2): fitted by most likelihood to the hour's 261 first bars of a Horny Mushroom (403 HP, its
#   MDDamage 0) under full within 1.5 s of her cast at one (D:/MapleHelper-scratch/recfix/sim-mage/scripts/claw_fit.py:
#   p10 322, median 334, p90 359 a cast; within 2 log-likelihood 156-157 to 180-182), as her own damage book has it
#   (lab/characters/magician/damage.json's fit at 00:51: 157-183). So a Horny always takes 2 casts (at most 362 a cast)
#   and a Green Mushroom (233 HP, MDDamage 30) 1 (at least 314 x 100 / 130 = 241): R1-R3's kills, 2 casts a Horny and
#   1 a Green. Against MDDamage as against PDDamage (x 100 / (MDD + 100)), the stack's own prior (attack.choice.need):
#   not measured (only the Green has any, and one cast kills it whatever the form);
# - CLAW_LOCK_S: the next cast taken 0.80 s after the key (the port's, RustMS: 800 ms of swingO1-O3 frames). The hour's
#   783 casts came back to back at 0.846-0.854 s (219) or 0.92-0.96 s (199), one frame apart at 15 fps, none under 0.80
#   (plans/work/simulator/mage_facts.txt): a bound, the lock at most about 0.85 s less the key's jitter;
# - CLAW_SHOWS_S: her hit shows on the monster (its flinch and push, its numbers, its death drawn: Realism.dies_late)
#   0.53 s after the key, the same wherever it is (no flight): R2's frames, a kill's monster seen alive 8 frames after
#   the key at the median (4-10) and dying from the next (rec-analysis/r23-verify-monsters/out/castgap2.txt);
# - CLAW_HIDDEN: a monster her cast hurt is lost under her hit's effect (83 x 94, 570 ms [data]) 0.4-1.0 s after it shows
#   (Noise.hit_hidden's for her, where the presets have the star's 1-3 s): R2's castgap, re-found 0.8-1.6 s after the key,
#   a median about 1.2;
# - her MP: 20 a cast (Magic Claw's mpCon at its level 20 [data]; Her.cast_mp), and with less the game refuses it with
#   MP_REFUSAL on the message bar (no cast, no lock), as live. Her pet's MP potion 187 MP (Her.mp_potion: R2's frames,
#   her HUD's number 60 -> 247, 71 -> 258 and 83 -> 270, D:/MapleHelper-scratch/recfix/sim-mage/mp_drinks2.png), drunk
#   under 6% of her 1549 MP (MAGE_PET_MP: the three at 60-83, none seen at 94; her bar reads 0.0 under 74 MP). Her MP's
#   own rise (Her.mp_regen, MAGE_MP_REGEN): her HUD's number ticks 18 MP every 10.0 s (R2's frames: 30 on that grid in
#   300 s; 1.8 MP/s, as the hour's stretches with no cast by her bar, 1.82: scripts/mp_regen.py, mp_ticks.py), and the
#   hour's balance asks 2.4 (783 casts x 20 = 15,660 MP, less the bar's net fall of 655 and a level's refill of 968, less
#   48 drinks of 187 = 8,976: 5,061 MP in 2,112 s; the rest of it +9 MP steps seen twice on R2's numbers, not explained):
#   the balance's, so her MP potions a minute come out as live;
# - her regular attack (a wand's swing up close) does nothing here: its damage isn't known and her stack never keys it
#   (Attacks.no_regular); counted (World.swings).
MP_REFUSAL = "You don't have enough MP to use this skill."
CLAW_HIT = (157, 181)
CLAW_LOCK_S = 0.80
CLAW_SHOWS_S = 0.53
CLAW_HIDDEN = (0.4, 1.0)
MAGE_MP_REGEN = 2.4
MAGE_PET_MP = 0.06
MAGE_MP_POTION = 187.0


@dataclass(frozen=True)
class SimMagic:
    """Magic Claw as the world casts it (Her.magic; the comment above CLAW_HIT has each from where): its search, its
    hits a cast and a hit's damage before the monster's defence (MDDamage), its lock, when its hit shows on the monster,
    how long that monster is lost under it, and its misses (none measured: her damage book judged 4 casts, 2 of them
    misses, too few to fit; 0)."""
    search: Search = CLAW_SEARCH
    hits: int = 2
    hit: tuple[int, int] = CLAW_HIT
    lock_s: float = CLAW_LOCK_S
    shows_s: float = CLAW_SHOWS_S
    hidden: tuple[float, float] = CLAW_HIDDEN
    miss: float = 0.0


def magic_band(s: Search, near: float, far: float, bottom: float, top: float) -> int | None:
    """The first band of search `s` (character.Search: Magic Claw's, SimMagic.search) a body overlaps, from `near` to
    `far` ahead of her and from `bottom` to `top` above her feet (reach.body_box), or None: as reach.search_band runs the
    thief's star search, by the search's own numbers (never attack.reach's, which attack.magic sets to her stack's
    belief), the overlap strict as the client's (RustMS ShotPath.h:335-339)."""
    if far <= s.start or near >= s.range or far <= near:
        return None
    first = max(0, int((near - s.start) // s.step))
    last = min(s.bands - 1, math.ceil((far - s.start) / s.step) - 1)
    for k in range(first, last + 1):
        floor, roof = s.window(k)
        if bottom < roof and top > floor:
            return k
    return None
# A touch by a monster within this many s of its respawn came from its spawn point, onto her or beside her (World.
# fresh_log): what a waiting place by a spawn point costs.
FRESH_S = 1.5
# ... and one within this many s of it, a newborn as the live hour's were counted (the cornered study's verify-R3: a
# touch within 4 s of a newborn's first sighting within 70 of her; World.newborn_log): what the spawn guard is for.
NEWBORN_S = 4.0
# ... but a hit pushes it (and it flinches) only if its damage, after the monster's defence (x 100 / (PDDamage + 100),
# as STAR_DAMAGE's), reaches its `pushed` in the game data: 1 for every kind of the first map (any damage, as above),
# 40 for a Horny Mushroom and 25 for a Green Mushroom on Tree Dungeon, Forest Up North II, 30 for an Axe Stump and 25
# for a Dark Stump on West Rocky Mountain II (both DEF 30). So two in three of her stars push a Horny Mushroom (30-70
# against DEF 10: 44 and up), a Green every one, an Axe Stump 78% (39 and up), a Dark Stump 93%, and no claw swipe
# pushes a stump; the rest leave it walking at her. Drawn per hit (push_chance), seeded by its tick. Not measured: the
# study of Tree Dungeon has "a hit under 40 doesn't knock back or flinch a Horny" from the game data alone.
PUSH_BY_DAMAGE = True
# How the world's attack from a jump picks its target: 'area', the nearest in the area fixed to her feet
# (reach.in_area, as her decisions have it: the 40 ms jump throws' kills of snails 36-66 off on 2026-10-20 fit
# it); or 'v83', as from the ground, from where she is then (the claw's stance, then the star's search), as
# RustMS's port of the v83 client has it, with no case for the air. Neither fits the live hunts of 2026-10-21:
# of 78 jump throws at a snail on her platform, 6 of 29 under 46 units off hurt or killed it, 20 of 36 at 46-84
# and 8 of 12 at 85-130 (and 6 of 8 throws from the ground at 85-130), where 'area' takes every one of them and
# 'v83' none at 35-85. Noise.air_snail has the area's pick of a snail that close miss as often as that (preset
# `game-air`), so the harness can weigh stance.JUMP_THROW_SNAILS. Which picture holds, `probe run air` asks: this
# lets the probe's scorer be checked in a world of either. 'widening', the default since 2026-10-25: the nearest whose
# body reaches into the area (now to reach.AREA_UP, 44, over her feet) or into its band's window of the star's search
# from her feet then (reach.air_takes, as her decisions have it): the windows widen with the distance, so a monster on
# a platform above is picked from low in the jump far off and only near its top up close, as the user's play of
# 2026-10-24 has it (reach.AIR_SEARCH). Up close on her own platform it's the area, as before.
# 'search', the default from 2026-10-27, as her decisions have it (reach.AIR_MODEL): the star's search from her feet
# then, with no claw and no area (World.air_search). 'widening' picked the target of 70% of the live in-place jump
# throws at 0-40 ahead and of 72% of those that turned in the air where 50% and 34% hurt it (788 throws of 2026-10-25
# and 26 on Tree Dungeon), so the harness counted those throws as hits and could never show what keeping clear of a
# monster gains. With the noise that stands for what the live positions don't tell (Noise.air_squash, air_short,
# air_under, air_turn_lost: TREE_AIR), the world on the live throws' geometry gives their kills by distance and by the
# slope within about 8 points of live's (README, Clearance and slopes).
AIR_PICK = "search"
# In the air, an arrow the game acts on at the same step (STEP_MS) as an attack's key turns her too late for that
# attack: it goes the way she faced before (the turn shows, the star doesn't follow). Live on 2026-10-25 (the hunt of
# 23:43) 10 of 28 jumps that turned her in the air threw the wrong way, the attack 25 ms after the arrow, and none of 68
# throws from the ground with it 80 ms after (analysis-1025/jumps-perception-skeptic/airturn2.py); how the game
# decides isn't known, and this has 25 ms apart fail one time in six (5 ms of the step's 30 shared), 60 ms never.
# Before 2026-10-26 the world turned her as it saw the arrow, whatever came at that step: 2 of 201 such jumps went
# the wrong way in the harness (master's hunt loop on Tree Dungeon, 8 seeds of 10 minutes). False: as before.
AIR_TURN_SAME_STEP = True
DROP_DELAY = 2.0
CLOSE_DX = 60.0          # (World.close_ms) a monster this near her, middle to middle: the live diagnosis's close episodes
# s from a kill until the monster is back at its spawn point (a guess; Realism.respawn). Not the game's, by its live
# telemetry of 2026-10-19 to 21 (analysis.respawn: README, Respawns and drop life): there a platform she emptied,
# watched with no gap over 2 s, had a monster again within 12.56 s on an estimated 20% (95% interval 12-33%; 12 of 77
# seen back, 23 still watched then), and too few were watched past 30 s to say more; in this world 11 of 22 were seen
# back by 8.8 s, the rest watched only to 6.3 s (52% not back by 5 s, 26-72%, against the game's 90%: the same report on
# 8 runs of preset game). The game's basin had a monster at 12% of its spawn points, none 45% of the time, this world's
# 27% and 7%. Kept until the game's refill is settled (a delay, or MeowDB's cap of the map's monsters, a respawn going to
# any free spawn point): the in-game check there.
RESPAWN = 8.0
RESPAWN_LIVE = 17.0  # ... and the live median on Tree Dungeon's 23 after her last kill there (1.7-57 s, telemetry of
                     # 2026-10-24: README, Positioning; harness preset crowd-17): `--world walkers` has it
# The game's respawn as measured on Tree Dungeon (2026-10-29 and 30: behavior.patrol's top, D:/MapleHelper-scratch/
# respawn and respawn-orion, RESULTS.txt): not a monster's own clock but the map's, filling it to its cap, as the
# server's LifePool.tryCreateMob does in OrionAlpha (a replica of Nexon's early Korean server): a pass every TICK_S (88%
# of the births within a second of the fitted ticks; the server's 7 s at the least) spawns the cap less the monsters
# alive, each at a spawn point drawn at random among the clear ones: no live monster within SPAWN_CLEAR of it across and
# up or down, nor a point found clear before it in that pass (OrionAlpha's 200 x 200 box; the user, 2026-10-31: "I feel
# the orionalpha's rule actually holds"). Until then a held point was drawn half as likely and got a second monster
# beside the first, from births beside a monster already there: the two such clips, looked at frame by frame, were
# spawns at the platform's other point 115 px away (the newborn credited to the point nearest where it was first
# tracked, after it had walked), and a spawn tracked once fading in and again solid reads the same; so they showed
# nothing against the box (D:/MapleHelper-scratch/respawn-orion/RESULTS.txt, 6). The cap is the server's formula
# (MapModel.mob_cap: the map's area times its mobRate, more only with more players than half of it; Tree Dungeon II 22):
# with her alone (10-29 16:37-16:46) nothing came back but what she killed, 17-19 free points watched 5 minutes without
# a kill and without a birth; the parts of the map she saw held 11-15 then, but the map was never seen whole without a
# kill (hers or another's) between, so its count isn't known; with others 0.39-0.55 a point at every count of players, 1
# to 6+ (MeowDB's 75% and 5% more a player: not seen). Until the user's call of 2026-10-30 ("Let's use the formula... we
# don't know how many mobs are on the map at any given time") CAP_SHARE, 0.42 of the points, from that count; the
# formula's from then on (Realism.respawn_cap None), a share still a knob. Until 2026-10-30 each free point came back
# with a chance at a tick (0.27), the same whatever died. Realism.respawn_tick (`--world tick-respawn`), the map
# starting at its cap; Realism.respawn's own clock otherwise.
TICK_S = 7.115
SPAWN_CLEAR = 100.0      # map units about a spawn point, across and up or down, that must hold no monster for it to fill
FADE = 0.8               # s a killed monster is still seen fading out where it died, for a kind whose death
                         # animation the game data doesn't have (else its own: MobBody.die_s, 0.66-1.26 s)
# A monster her attack killed is drawn alive, walking on with its bar up, until her star lands (Realism.dies_late): its
# death animation began DIE_LANDS_S after the killing key and as her distance to it over DIE_FLIGHT units a second
# more (live on 2026-10-26, the Lie Detector's recording of 23:10: 23 of 26 kills within 0.15 s of that; a median 0.89
# s after the key, 0.60 after its gain: D:/MapleHelper-scratch/kplace/diag-signs), then it stops where it got to and
# plays it; the frames' detector has no die1 pictures, and lost it DIE_LOST_S after it stopped (the service's last
# sighting of it 0.07 s before die1 began, seen 0.85 s in at the median on the 52 of 142 deaths re-found in die1/0: a
# tenth of a second or two) where the game's death animation still shows it.
DIE_LANDS_S = 0.75
DIE_FLIGHT = 1490.0
DIE_LOST_S = 0.17
# The death watch in the world (FakeEyes.deaths, perception.deaths' stand-in): a death drawn is seen DEATH_SEEN_AFTER s
# after it began, on DEATH_SEEN of them (drawn once a death): live, 118 of 142 deaths on a track (83%), a median 0.14 s
# after die1 began.
DEATH_SEEN = 0.83
DEATH_SEEN_AFTER = 0.14
# A monster the map spawns fades in (Noise.spawn_found): from nothing to solid in SPAWN_FADE_S (alpha 1.224 a second,
# 223 births on Tree Dungeon's lossless clips: perception.spawns.FADE_PER_S), and a look finds it only once its alpha
# has reached the one drawn for it from a table of the share of newborns first found by then (each one's own draw, its
# index and birth: then seen as any monster is, the rest of the noise as ever), its alpha in what the look sees
# (Mob.fading) while it fades. Measured on those clips' 323 traced births (D:/MapleHelper-scratch/spawn-fade/integ,
# cdf.py): with the frame step's spawn watch (perception.spawns, worldstate.SPAWNS) the watch or the whole screen first
# found them at alpha 0.098 at the median (p10 0.039, p90 0.66: the ones whose start was hidden, under her, players or
# effects, as late as the whole screen finds them), SPAWN_FOUND; the whole screen alone (before 2026-11-01) at 0.647
# (p10 0.41, p90 0.89), SPAWN_FOUND_WIDE: 0.53 s later. A table's pairs (alpha, share first found by then), from the
# faintest any was found at.
SPAWN_FADE_S = 0.817
# ... and it touches her only once it has faded in (Realism.newborns_touch False: every world's from 2026-11-02, the
# user's decision 2 of the recordings' roadmap; till then it touched from its birth, `--world newborns-touch`). Live,
# RiceBaby's hour of 2026-11-02 00:17:48-00:53:06 (her 170 touches, all hurting, against the 7.115-s grid of the
# respawn passes the spawn watch's 1,251 fading sightings fit; D:/MapleHelper-scratch/rec-analysis/plans/work/simulator/
# touch_phase.py, touch_phase.txt): of the 72 with her within 45 of a spawn point of her platform none came 0-0.75 s
# after a pass, the earliest at 0.82 s (five at 0.82-0.84), where about 5-6 were to be expected at the rate away from
# the passes (Poisson's P(0) about 0.003), then 19 at 0.75-1.0 s. The one touch 0.24 s after a newborn's start away
# from a point (00:48:18.6) had her at a rope's top 175 under it. So 0.82 s from the newborn's alpha 0, the fade's
# end. A newborn walks while it fades (the world starts its walk at its birth: perception.spawns, 42 units by solid at
# the median) and her attacks can hit it (spawns.FOLLOW has its hit frames), both as before.
NEWBORN_SOLID_S = SPAWN_FADE_S
SPAWN_FOUND = ((0.02, 0.0), (0.05, 0.204), (0.1, 0.579), (0.2, 0.734), (0.3, 0.737), (0.4, 0.762), (0.5, 0.793),
               (0.6, 0.861), (0.7, 0.926), (0.8, 0.957), (0.9, 0.988), (1.0, 1.0))
SPAWN_FOUND_WIDE = ((0.2, 0.0), (0.3, 0.006), (0.4, 0.093), (0.5, 0.207), (0.6, 0.409), (0.7, 0.576), (0.8, 0.771),
                    (0.9, 0.904), (1.0, 1.0))
POSE_SPELL_S = 1.0       # s a monster's pose reads right or wrong at a time (Noise.pose_wrong)
HEADING_SPELL_S = 1.0    # s one pose of hers reads the wrong way at a time (Noise.heading_wrong in runs): a pose her
                         # pictures read wrong reads so on every frame while she holds it
DROP_LIFE = 180.0        # s drops lie before they vanish (a guess: in game they lie over a minute, 95% at 60 s of 337
                         # followed, and the hunts' telemetry can't say more; analysis.respawn)
PICK_REACH = 15.0
POTION_HP = 0.30         # of her HP a potion heals where what it heals isn't known (Her.potion): as on the first map
# Her pet's potions (Her.pet), as meowdb's page on this game's pets has them (meowdb.com/msclassic/formulas/pets,
# read 2026-10-24): it drinks when "HP after the hit x 100 / MaxHP < Warning", the warning a multiple of 5% (System
# Options), one potion she picked, from her stock; potions "at least 0.2 seconds apart (unconfirmed)". So the pet
# drinks right after a touch that leaves her under the warning (when after it isn't known: at once), one potion a
# touch, never sooner than PET_GAP after the last; with none left it drinks nothing, nor too hungry (below).
PET_GAP = 0.2
KNOCK_FALL_DY = 30.0     # units: a knock that lands her this much lower than where it came from, on another platform,
                         # knocked her off it (World.knock_log); a step down between two platforms of a run is less
# Her pet's fullness (with a pet: Her.pet), as behavior.pet follows it: HUNGRY lost every TICK_S (the game data's 1
# for her Snail pets; 70-74 s a point live on 2026-10-24), FOOD (30) for a Pet Food, sent home at PACKED_AT or under
# (then it drinks nothing). At a run's start PET_FULLNESS, not known: under pet.FEED_BELOW (70), so the pet care's
# first read feeds it once (60 -> 90), as she fed it at the start before it was read. The pet care reads it off the
# world (sim.fakes.FakePetCare: no window, no OCR, no clicks).
PET_FULLNESS = 60.0
# Her pets as looters (Realism.pets: the harness's `--world pet-loot`; the user, 2026-11-01: "Has pick-up skills but not
# exactly item pouch/meso magnet. The pet will pick up whatever thing when the pet walks past those loot. But the pet
# won't proactively go for the loot"): her Snail pets (Item/Pet 5000054, 5000033: pickupItem 1, no sweepForDrop nor
# longRange in the game data) trail her on her platform and take what they pass, never going for it (SimPet,
# World._pets_step). Each walks toward her at PET_SPEED until it is its PET_FOLLOW from her (the first's the first
# number), then wanders within PET_WANDER of her at half that, a new place every PET_WANDER_S; on another platform than
# hers PET_HOP_S, it's on hers, its PET_FOLLOW behind her; it takes any drop she could (hers, or another's past its
# ownership, that fits her tabs: World.takes) lying on its platform within PET_REACH of it, as her own pick-up (the same
# log line, her tabs). Live (the perception service's 15 fps records of 2026-10-25 to 30, D:/MapleHelper-scratch/replan/
# loot2/design/pets): of her drops she came within 15-30 units of without standing on them, 56% vanished with her more
# than 15 off, 46% / 40% / 32% / 25% by her closest approach of 30-45 / 45-60 / 60-90 / 90-150, a median 0.71-0.83 s
# after it, 2-4% with her on another platform; in the golden clip the snail tags by her were 53 / 73 / 102 units from
# her (quartiles), under her 48% of the time (trailing her climbs). v83's follow isn't read from the client: the numbers
# are a coarse grid's (10 settings x 4 seeds, then 12) against those tables. With them the simulator's pets take 31% of
# her pick-ups, by her closest approach 0.96 / 0.73 / 0.63 / 0.63 / 0.29 where live 0.73 / 0.57 / 0.49 / 0.40 / 0.32
# (the same shape, about 0.15 high), and 7% of her drops are taken 45-150 units from her where live 23% (no setting
# tried passed 11%: the simulator's grind walks onto its piles more than the live bots did).
PET_SPEED = 80.0         # units a second (60 / 80 / 120 tried)
PET_FOLLOW = (40.0, 70.0)  # units from her each stops at (25/45 to 90/150 tried)
PET_WANDER = 80.0        # units either side of her it wanders within (40 / 80 / 140 tried)
PET_WANDER_S = (1.0, 3.0)  # s between its wander's places, drawn
PET_HOP_S = 1.0          # s she's on another platform before it's there too (1 and 3 tried)
PET_REACH = 15.0         # units from it a drop is taken within (15 and 25 tried; her own PICK_REACH)
PET_STEP_MS = 50         # ms between its steps
PET_SCAN_S = 0.25        # s between its looks at the drops lying (one landed since is passed over until then)
PATH_EVERY_MS = 100      # ms between two of her places in World.path_log (the harness's tables of where she was)
# What isn't known of the game's physics, which the simulator can run either way (Realism.sides, zero_walls,
# rope_reach; the defaults are the simulator's own, as its golden traces were recorded with):
# - The map's sides. The simulator's: its platforms' outermost ends, which stop her in the air only. HeavenClient's
#   (the open-source v83 client RustMS ports, whose step motion.body's is; Gameplay/Physics/FootholdTree.cpp, read
#   2026-10-24: `walls = { leftw + 25, rightw - 25 }`): SIDE_INSET in from its footholds' leftmost and rightmost x,
#   which stop her on the ground too, wherever no foothold of her own stops her first. On the first map -464 and 1454
#   for the platforms' -489 and 1479 (live, in 14,869 placements of 2026-10-19 to 21, she was never further than -432
#   and 1446; in the simulator she stands at 34's left end, -489, in tour_run); on West Rocky Mountain II -1637 and 88
#   for -1662 and 113 (22's left end is at -1640, 22 units in from the platforms' side, and fighting there she was
#   knocked off it and fell out of the map in 8 of 10 of the harness's hunt loops and 5 of 10 tours, preset game, 20
#   minutes; none with the client's sides); on Tree Dungeon, Forest Up North II -475 and 565 for -500 and 590, both
#   outside the tower (-220..310), where nothing is under her either way.
# - The walls of foothold group 0 (a map's own: on Tree Dungeon, Forest Up North II the tower's, its shelves' and its
#   floor's; its floating platforms are other groups, with no walls). Her own group's walls alone stopped her until
#   2026-10-31 (body.OWN_GROUP_WALLS, from a reconstruction of the client's movement); RustMS's port has group 0's
#   stop every group's stars (README, Attack). They stop her body too, whatever is under her (Realism.zero_walls, on
#   top of HeavenClient's rule, body.wall_table: the default from 2026-10-31):
#   live on Tree Dungeon (the perception service's scenes of 2026-10-29 and 30, 387,443 placements; D:/MapleHelper-
#   scratch/tower) in the air along the tower's walls her place by a cue (her tag or body, not carried) was at the
#   wall's x, 310 or -220 to a unit, 351 times, and past it 3 (318, 328, -222: the body cue's spread), never on
#   anything outside, where she stood within 20 of 24's right end (the wall) 7,031 times; falling along it she landed
#   on 19 at 310, on 13 at -220. With her own group's walls alone (as until then), walked or knocked off the
#   outer end of a floating platform there (11's left end, 13 from the tower's wall, 23's 23, 24's right end at it,
#   25's right end 34), or off a rope near the wall (the one at -192, 28 from it: off a rope no wall stops her,
#   body.ROPE_GROUP), she fell past the wall and out of the map: loop_tree_run, 5 of seeds 1-13 on 2026-10-31.
#   HeavenClient's rule alone (`--world joined-walls`: a wall stops her if it's joined to the foothold under her, its
#   prev or the one before, where it overlaps the 50 units above her feet: FootholdTree::get_wall) lets her through 143
#   of those 351 live placements, over 70 units above the floor under her.
# - How far above her feet a rope's bottom can be for a jump to grab it: physics.ROPE_REACH, 95 (her jump's 77 and
#   18 of arms: a guess from the start, never measured; the highest grab tried in game was a rope 69 up, on the first
#   map); HeavenClient grabs a rope when a point 5 above her feet is on it (Gameplay/MapleMap/MapInfo.cpp,
#   Ladder::inrange, read 2026-10-24), 82 up at the top of her jump (physics.GRAB_SURE). Only one climb on the three
#   maps is between the two: from 30 up the rope at -8 on Tree Dungeon, Forest Up North II (94.3 up), on every way up
#   from the bottom of the tower; jumping for it and missing, she passes up through 29 (58 above 30 there) and lands
#   on it.
SIDE_INSET = 25.0
ROPE_REACH_CLIENT = GRAB_SURE
MESSAGE_FOR = 5.0        # s a pickup message stays in the log on screen
FULL_MESSAGE = "You can't get anymore items."


def tab_of(kind: str) -> str:
    """The inventory tab an item of `kind` goes into in the simulator: Use for a potion, Etc for the rest (as its
    pickup log says them; behavior.looting.tab_of reads the game data's, which agrees for Tree Dungeon's drops)."""
    return "Use" if kind.endswith("-potion") else "Etc"


# The game's pickup log cuts a long item name off with "..." (live on 2026-10-26, tests/fixtures/
# pickup_log_lines_2026-10-26.json: "Horny Mushroo... x1 earned. (Etc)", "Green Mushroo...", "Arrows for Cros...";
# "Arrows for Bows", "Orange Potion" and "Emerald Ore" whole): by the px of its font, here by letters, a name of more
# than LOG_NAME_MAX cut to its first LOG_NAME_CUT (Tree Dungeon's caps as the game cuts them; until the review of
# 2026-11-01 every name whole, so the looter's reading of the log was never tried on a cut one in the simulator).
LOG_NAME_MAX = 15
LOG_NAME_CUT = 13


def log_name(kind: str) -> str:
    """An item of `kind` as the pickup log names it (LOG_NAME_MAX)."""
    name = kind.replace("-", " ").title()
    return name if len(name) <= LOG_NAME_MAX else name[:LOG_NAME_CUT].rstrip() + "..."
# The message bar's line when the game refuses an attack on a rope (live on 2026-10-26: perception.messages); it stays
# up until another line replaces it (none does in the simulator)
ROPE_REFUSAL = "That cannot be used while you're on a rope or ladder."
ITEM_OF = {"pig": "leather"}  # (the first map's pigs; a kind not in it drops mesos alone)
# What a kill drops in a world closer to the game's (Realism.drops; the harness's preset `game-drops`): for each
# monster kind, each drop and the chance a kill drops it, as seen where her kills died on the first map in the hunts of
# 2026-10-21 (of the kills whose place a look saw drops at 2-12 s after, within 60 units: 52 slimes, 45 pigs, 23 Orange
# Mushrooms, 19 snails; an item there of another kind's, a slime's Squishy Liquid by a pig in the basin, is left
# out). The default world drops mesos, and leather from a pig (ITEM_OF), every time.
GAME_DROPS = (
    ("slime", "mesos", 0.75), ("slime", "squishy-liquid", 0.65), ("slime", "slime-bubble", 0.10),
    ("slime", "red-potion", 0.04),
    ("pig", "mesos", 0.80), ("pig", "pig's-head", 0.07), ("pig", "leather", 0.02),
    ("orange-mushroom", "mesos", 0.70), ("orange-mushroom", "orange-mushroom-cap", 0.70),
    ("snail", "mesos", 0.63), ("snail", "snail-shell", 0.47), ("snail", "red-potion", 0.11),
)
# Realism.drops of a world that drops what its own map's monsters drop (`game_drops`): the harness's `game-drops`.
MAP_DROPS = "map"
# On another map (`game_drops`) a monster whose drops were seen in game on the first map (GAME_DROPS: the slime, ...)
# drops what it dropped there, and its other drops (perception.drops: a slime's Orihalcon and Sapphire Ore) at OSMLib's
# reference rates. One not seen in game drops mesos this often (the first map's four kinds' 63-80% of their piles with
# coins, on average) and each item of its at OSMLib's rate; where OSMLib gives none (the Magic Potion, a Fragment of
# Magic, a Stump's Bronze Ore: 4 of Tree Dungeon's 22 rows, 4 of West Rocky Mountain II's 25), as rarely as the rarest
# it gives that monster (every one's ores', 0.2%), or NO_RATE where it gives it none. At 1% those four had made
# valuable drops (66.8 and up) 0.0177 a kill on Tree Dungeon (weighed by spawn points), 80% of it theirs, the same ore
# 1% from one monster and 0.3% from another; at the rarest, 0.0064. Neither is the game's: OSMLib's "may not represent
# MapleStory Classic World rates", and the first map's slimes dropped Squishy Liquid at 65% where it gives 40%, a Slime
# Bubble at 10% for its 3%.
MESOS_SHARE = 0.72
NO_RATE = 0.002


@dataclass(frozen=True)
class DropMix:
    """What a kill drops in a world fitted to what the perception service saw where her kills died live (Realism.
    drop_mix; the harness's `--world map-drops`): her thief's run on Tree Dungeon of 2026-10-30 02:03-02:48 (the loot
    design's measurements, D:/MapleHelper-scratch/replan/loot: drops_fate.py, spread.py, dupes.py, expcount.py) gave 0.64
    drops seen a kill of hers, 57% of them mesos, its piles of 1, 2, 3 and 4 or more drops 61, 23, 8 and 6% of them (of
    the scenes' clusters 6,609, 2,538, 884 and 625), neighbouring drops 20-25 units apart (3,649 of the 6,739 gaps under
    70), where the default world drops one coin a kill at +-10 of where it died. Each kill's pile is empty, or has
    `sizes`' drops (drawn by their shares), `per_kill` a kill on average; each drop mesos with `mesos` of a chance, its
    amount drawn from its kind's range (perception.drops' reports: a Horny Mushroom 40-53, a Green 25-30, a slime 10-20),
    else one of its kind's items (perception.drops: by OSMLib's reference rates, an item with none at the rarest's);
    the first where the monster died, each next `gap` farther out on alternating sides (+20, -20, +40, ...: how the
    game pops them out). What is seen counts what the detector missed as not dropped, so this is a floor."""
    per_kill: float = 0.64
    sizes: tuple = ((1, 0.61), (2, 0.23), (3, 0.08), (4, 0.08))  # (drops, share of the piles with any)
    mesos: float = 0.57
    gap: float = 20.0

    def any_share(self) -> float:
        """The share of kills whose pile has any drop: per_kill over the drops a pile has on average."""
        mean = sum(n * p for n, p in self.sizes) / sum(p for _, p in self.sizes)
        return min(1.0, self.per_kill / mean)


TREE_DROP_MIX = DropMix()
# Her drops taken by others (Realism.drop_takers; the harness's `--world drop-takers`), as the live evidence has it (the
# loot designs of 2026-10-31: D:/MapleHelper-scratch/replan/loot/diag, live_km.txt, and design): hers alone for
# Realism.rival_owns (the killer's ownership: HeavenMS's 15 s, not measured on this server), then taken by someone else
# at TAKE_RATE a minute, else lying DROP_LIFE (HeavenMS's 180 s; one live expiry pass took four drops 198-228 s old).
# The perception service's own, her pets' pick-ups (71% of her drops that vanished did so 15 units or more from her, the
# pickup log saying "earned" by her pets and not by her) and the detector's misses taken out: 95%, 88% and 86% of her
# drops still lying at 15, 60 and 90 s, at most about 0.1 a minute left to others, and of 1,580 vanishings of hers timed
# to within 1 s 5 with another player within 60 units, all under 15 s old. TAKE_RATE 0.05 is a guess inside that bound.
# Until 2026-10-31 each drop's life was drawn from the live service's raw curve (HER_DROPS_LAST: 55% still there at 60
# s, 26% at 120, none at 180), which counted her pets' pick-ups and the detector's misses as takings: about 4 times her
# real loss.
TAKE_RATE = 0.05
# ... and (from the review of 2026-10-31) the rest gone at the server's first expiry pass past DROP_LIFE, the passes
# EXPIRE_PASS_S apart from a time of the run's own (Drop.life): the one live pass took four drops 198-228 s old, so its
# passes come 48 s apart at least, and 60 is a guess. Until then they went at DROP_LIFE.
EXPIRE_PASS_S = 60.0


def expiry_life(since: float, phase: float, every: float | None = None) -> float:
    """How long a drop down at `since` lies untaken: to the first of the server's expiry passes (at `phase` and every
    EXPIRE_PASS_S) past DROP_LIFE."""
    every = EXPIRE_PASS_S if every is None else every
    return phase + math.ceil((since + DROP_LIFE - phase) / every) * every - since


def others_life(u: float, owns: float, rate: float = TAKE_RATE) -> float:
    """When one of her drops is taken by others (s from its landing), `u` uniform in (0, 1]: `owns` (her ownership) and
    an exponential at `rate` a minute after it (DROP_LIFE and more: it lies its whole life)."""
    return owns + (-math.log(max(u, 1e-12)) / (rate / 60.0) if rate > 0 else math.inf)


def game_drops(model) -> tuple:
    """What a kill drops on map `model` in a world closer to the game's: (monster kind, drop kind, chance), as
    GAME_DROPS. The first map's are GAME_DROPS, as seen in game; another's are its monsters': a monster seen in game
    (GAME_DROPS' kinds) as seen there, the rest (perception.drops) mesos MESOS_SHARE of kills, and each of their other
    items at its reference rate, or their rarest one's where it has none (NO_RATE where they have none); a monster
    neither knows drops mesos alone."""
    if getattr(model, "id", None) == drop_data.FIRST_MAP:
        return GAME_DROPS
    seen: dict[str, list] = {}
    for row in GAME_DROPS:
        seen.setdefault(row[0], []).append(row)
    out = []
    for name in drop_data.monsters_of(model):
        kind, known = kind_of(name), drop_data.MONSTERS.get(name)
        rows = list(seen.get(kind, [(kind, "mesos", MESOS_SHARE)]))
        if known is not None:
            rarest, had = min(known.rates.values(), default=NO_RATE), {drop for _, drop, _ in rows}
            rows += [(kind, kind_of(item), known.rates.get(item, rarest)) for item in known.items
                     if kind_of(item) not in had]
        out += rows
    return tuple(out)

# Another player (Realism.rival_every) hunts one place at a time: it kills a monster within RIVAL_REACH across and
# RIVAL_DY up or down of where it stands, every Realism.rival_every s on average, and walks to another (by a live
# monster, RIVAL_NEAR from it at most: a platform as likely as it has monsters alive) after Realism.rival_stay s
# there on average, or as soon as nothing is left in its reach. It walks RIVAL_SPEED straight across the map, as its
# minimap dot goes from one place to the other, and kills nothing on the way. Its dot is on the minimap (FakeEyes),
# so the stack sees where it is. (Before, the rival killed a monster anywhere on the map, and had no place.)
RIVAL_REACH = 400.0
RIVAL_DY = 60.0
RIVAL_NEAR = 150.0
RIVAL_SPEED = 125.0
# Other players camping (Realism.camps), as the crowding study saw them on Tree Dungeon, Forest Up North II from her
# spot on 28 (D:/MapleHelper-scratch/map3/study/crowding.md, 2026-10-24: 19 frames 2 s apart, 35.4 s): 3-6 others in the
# minimap's view at once, each staying by 1-3 neighbouring platforms for 80 s or more, killing the Green Mushrooms there
# (8 kills seen, all Greens; of the 7 whose appearing was seen, 5 died 2.1-6.3 s later) and leaving the Horny Mushrooms
# (none of 3-4 died; one took 192 and lived), their drops gone within one 2 s step. A camp (Camp) stands at a place, its
# dot on the minimap there, and kills each monster of its kinds on the platforms it hunts CAMP_KILL s after it shows (at
# the start, or respawned), drawn per monster from a generator of its own; its drops are taken CAMP_TAKES s after they
# land. It doesn't move, and kills nothing it doesn't hunt.
CAMP_KILL = (2.0, 6.0)
CAMP_TAKES = 1.0


class SimDied(Exception):
    """Her HP reached 0: she's dead, and a simulated run ends there (nothing she does counts any more)."""


class SimLeftMap(SimDied):
    """Up held at a portal (World._through_portal: in its box, mapdata.PORTAL_BOX, while she isn't on a rope, sitting
    or attacking, on the ground or in the air, as HeavenClient takes one; whether the game takes one in the air isn't
    known): she went through it to another map, and a simulated run ends there. No plan holds up within 60 units of
    one (mapdata.MapModel.portal_near, the spawn points' too, and an exit up to 125 below: the climbs, flow and the
    mover keep it): on Tree Dungeon, Forest Up North II nowhere right of x 118 on 28 and 12 (their exits at 282 and
    281, the spawn points from 178)."""


class SimFellOut(SimDied):
    """She fell below the map's lowest floor (a knock or a flight through a wall of another foothold group
    with nothing under it): she'd fall for good, and a simulated run ends there, named, not left standing
    still in the air for the rest of it. What the game does then isn't known (the map's bottom isn't a
    floor in its data)."""


TAP_LOST_MS = 50         # an arrow tap this short (as sent) can be lost (Noise.tap_lost)
# The detector's misses near her (Noise.near_seen) as 2026-10-25's perception had them (the harness's
# `--world near-misses-1025`; `near-misses` until 2026-11-03), live on Tree Dungeon (23:43-23:54, the perception
# service's 10 fps records, lab/telemetry/20261025-210747-perceive.jsonl; D:/MapleHelper-scratch/analysis-1025/
# damage-skeptic occl.py): of the frames in which it followed a monster on her platform, the share that found it, with
# her not attacking in the 1.5 s before (a monster her stars hit hides under its damage numbers: Noise.hit_hidden), was
# 7% within 50 units, 34% at 50-100 and 55% at 100-150 (attacking: 12%, 20%, 44%; 63% at 125-150 either way); and of
# the hunter's 1317 searched looks none found one within 25 units of her (nor at 00:31, 3029, nor in the user's own
# play, 1753): her body drawn over it, 22 units either side of her and 28 of a mushroom. Not only her body: 34% at
# 50-100. Past 150 on her platform (review of 2026-10-26: D:/MapleHelper-scratch/hunt-ab/review-b-measure/scripts/
# offplat.py, D:/MapleHelper-scratch/hunt-ab/B/fix/detect_bands.py) 49-52% (45-74% by 50 units, 65% in all, without the
# tracks that ended dead or dropped: not a monster there), 60%; and off her platform (OFF_SEEN_1025) 56-68% by 50
# units, 60% in all, 64-75% and 71% without those, 70%: the preset's detect_miss (10%) found 89% of them.
NEAR_SEEN_1025 = ((25.0, 0.0), (50.0, 0.07), (100.0, 0.34), (150.0, 0.55), (float("inf"), 0.6))
OFF_SEEN_1025 = 0.7
# ... and as the perception of 2026-11-02 finds them (the harness's `--world near-misses` from 2026-11-03, the user's
# "yes" to the touch study's two changes as the defaults; `near-misses-1102` before, its name kept; the touch study of
# 2026-11-03, D:/MapleHelper-scratch/recfix/touch-study/proto: detect_bands.py, pooled.py, their .txt): the replay kit's
# dumps of 92b09c3's perception on R1-R3 (her) and on BugBaby's frame sets (his play of
# 10-30 and the alarm clips of 10-26..10-30), by the same definition: of the frames in which the tracker follows a
# monster on her platform, the share that found it, her on the ground, the frames within 3 s of a hit seen on it left
# out (its bar's, her throws': the world hides those on its own, Noise.hit_hidden, CLAW_HIDDEN), and the tracker's tails
# it dropped or lost off screen (maybe no monster there; those that ended in its death kept). Hers and his: within 25
# of her 38% and 39%, 25-50 66% and 57%, 50-100 84% and 74%, 100-150 79% and 88%, farther 95% and 89%, another platform
# 91% and 92%; pooled by their frames (415, 626, 1023, 603, 988; 52,691 off it) below. Counting every hidden frame (the
# tails too) 31%, 46%, 66%, 74%, 91%, 87%; only those found again 49%, 73%, 85%, 87%, 95%, 93%. Where 2026-10-25's
# detector (NEAR_SEEN_1025) never found one within 25 of her.
NEAR_SEEN_1102 = ((25.0, 0.39), (50.0, 0.63), (100.0, 0.80), (150.0, 0.83), (float("inf"), 0.93))
OFF_SEEN_1102 = 0.91
# ... its misses in runs, as live's come (Noise.near_spell, off_spell; runs_live.py, spell_fit.py, their .txt): of two
# frames in a row of one track (the same frames as above), found after found 85-99% of the time, after unfound 2-13%,
# where a draw a look (NEAR_SEEN_1025's) gives the band's share after either (39-93%). A draw a spell of n frames, its
# phase drawn, gives 1 - (1 - p)/n and p/n: by most likelihood (hers and his pooled) n 5 within 25 of her (0.33 s; 95%
# 4-6), 11 at 25-50 (0.73 s; 8-16), 14 at 50-100 (0.93 s; 10-21), 16 at 100-150 (1.07 s; 10-30), 7 farther (0.47 s;
# 5-11), 8 on another platform (0.53 s; 7-8).
NEAR_SPELL_1102 = ((25.0, 0.33), (50.0, 0.73), (100.0, 0.93), (150.0, 1.07), (float("inf"), 0.47))
OFF_SPELL_1102 = 0.53
# ... and with the exact finder beside the search (perception.exact; the harness's `near-misses-exact`, Noise.exact: not
# the default until the finder has run live): by the same definition (the touch study's detect_bands.py, its D shares,
# master's the same numbers again) on the frame step's replays of R1-R3 with the finder on (D:/MapleHelper-scratch/unseen/
# fix/bands.txt, her own frames): 100%, 100%, 99.3%, 99.4%, 100%, another platform 98.6%, its misses in spells of a frame
# near her, 0.6 s on another platform. On BugBaby's lossless clip of 2026-11-02 22:49 (fix/gaps.py, by chains of its
# finds): within 100 x 70 of her 96.4% of the monster-frames, a miss 2 frames on average; farther 92%, 6 frames. Pooled by
# their frames, what's below. The search's misses it fills in; a monster hit isn't lost under its numbers or her effect
# (those cover part of it, 14 of 14 found in the hand sample: Noise.hit_hidden off with it). Not measured live.
NEAR_SEEN_1104 = ((100.0, 0.98), (float("inf"), 0.97))
OFF_SEEN_1104 = 0.97
NEAR_SPELL_1104 = ((100.0, 0.13), (float("inf"), 0.4))
OFF_SPELL_1104 = 0.5
# The detector's misses near her as live (the harness's `near-misses`, the scenarios' _NEAR): today's perception's.
NEAR_SEEN, OFF_SEEN, NEAR_SPELL, OFF_SPELL = NEAR_SEEN_1102, OFF_SEEN_1102, NEAR_SPELL_1102, OFF_SPELL_1102
# The air pick's noise on Tree Dungeon (Noise.air_squash, air_short, air_short_ramp, air_under, air_turn_lost), fitted by
# likelihood to the live jump throws in place at a monster on her platform of 2026-10-25 23:43 to 10-26 23:47, all 838 of
# them with the target's track at the pick and ahead of her middle (530 straight, 308 turning in the air; every track of
# her platform there: D:/MapleHelper-scratch/clear/diag-clearance/attacks.jsonl, the review's full set), by their kills
# (an EXP gain of the target's kind, hers alone: P = picked * K + B, K by kind) and by their bar drops (which other
# players' hits drop too: P = picked * H + B): D:/MapleHelper-scratch/clear/fix/live/fitsoft.py, fitramp.py. Each
# monster's top drawn uniform over its height (air_squash 1.0: the search picks one the likelier the farther its top
# reaches into a window, which a hard edge at the kind's height doesn't give: live the straight throws whose target's top
# reached 0-10 into the deepest window of its bands hurt it 60% of the time, 10-20 71%, 20-30 82%, 30 or more 90%,
# overlap.py), a target short of the first band taking the star 0.3 * max(0, 1 - short / 25), one under a window 0.3
# (by key the same: 0.3 at 40 ms and at 120 by kills), an air turn lost 26% (the live share facing away after one).
# Log-likelihood by kills -449.2 against -466.6 for TREE_AIR as it was from 2026-10-27 (flat 0.3 and 0.3: fitted on the
# 788 of diag-slopes' replay, which left out the throws whose target its frames didn't show) and -478.6 for the search
# alone; by the bar drops -353.1 against -370.0 and -380.4. With it the world on the live geometry gives the straight
# throws' slope (40-130 ahead, the target's feet over 5 lower than hers, level, over 5 higher) kills 32/38/50% against
# live 31/38/58 (TREE_AIR as it was: 34/36/38), hurt 62/66/80 against 66/74/89 (bins2.py), and 0-10, 10-20, 20-30, 30-40
# ahead kills 7/9/18/27% against live 3/8/16/37 (as it was 17/16/18/23). Picked, a cast's target was hurt 75% of the time
# by the fit (H; the world's 1 - Noise.miss 72.5%: Noise.miss stays) and killed K 0.7 (Green), 0.45 (Horny Mushroom).
TREE_AIR = {"air_squash": 1.0, "air_short": 0.3, "air_short_ramp": 25.0, "air_under": 0.3, "air_turn_lost": 0.26}


@dataclass(frozen=True)
class Noise:
    """The real game's errors, all off by default. Each draw is seeded by the world's seed and the
    moment it's for (a star's tick, a key's tick), and the spells are fixed stretches of time, so a
    run is still the same every time, and a change that only adds or drops a look changes nothing here
    (detect_miss is drawn per look, so with it on, an extra look does change what later looks see)."""
    miss: float = 0.0            # chance a star that would hit misses (lab/shots.jsonl: 72.5% of 204 hit)
    regular_miss: float = 0.0    # ... and a regular attack's star, on top of it (behavior.keycheck's doubt with her keys right)
    miss_unseen: float = 0.0     # chance a missed star's MISS popup over its monster isn't seen (World.misses: under
                                 # other numbers, a window); 1: none is (the doubt as before her attack's MISS was read)
    walk_jitter: float = 0.0     # units: letting go of an arrow stops her up to this much early or late
    lag_sd: float = 0.0          # the lags' spread around their medians, times the measured one (body.Lags:
                                 # sd 7.7 ms of when the game sees each key, which the wait for its next
                                 # step makes the 11-13 measured), drawn for each key; 1: as measured. 0:
                                 # every lag is its median. (On the game's steps a tap seen held less than a
                                 # step can fall between two and not move her: a 20 ms tap 27% of the time)
    tap_lost: float = 0.0        # chance an arrow tap of TAP_LOST_MS or less turns her without walking her
                                 # (the game sees it pressed, then let go at once): in the recordings, of
                                 # five taps of 20-50 ms, three moved her 0 (0.6; research 4.7)
    dot_px: bool = False         # her minimap dot read to whole pixels (11.4 map units each), not exact
    tag_hidden: tuple = ()       # (from, to) s since the start: her name tag can't be seen (behind a box)
    unfound: tuple = ()          # (from, to) s: the localizer hasn't found her (live's first frames, or after a gap in
                                 # its frames: her tag, plate and body nowhere): her place by her dot alone, as live's
                                 # then (Senses._by_scene's not found), her tag not seen
    dot_hidden: tuple = ()       # (from, to) s: her minimap dot can't be seen (a tooltip over the minimap)
    exp_hidden: tuple = ()       # (from, to) s: her EXP number can't be read (the cursor or a window over it: its
                                 # pixels change as it comes and as it goes)
    hit_hidden: tuple = ()       # (shortest, longest) s a monster hit by a star (and not killed) can't be
                                 # seen: in game it flinches under its damage numbers and the detector loses
                                 # it for 1-3 s (README); () off
    detect_miss: float = 0.0     # chance a look doesn't find a monster that shows (per monster and look)
    pose_wrong: float = 0.0      # share of the time a monster's pose reads wrong (walking as standing, standing
                                 # as walking), in spells of POSE_SPELL_S drawn per monster: in game the template
                                 # that matched said so for 18% of the sightings of monsters walking and 22% of
                                 # those standing (3 recordings with the camera still), mostly a frame at a time
                                 # but for up to 1.4 s on end, most of the time wrong in such long stretches
    dot_under: float = 0.0       # units: another player's minimap dot this near hers isn't seen (hers is drawn over
                                 # it: on Tree Dungeon's frames theirs vanished within about 3.5 minimap px, 56 units,
                                 # her yellow core whole in all 20); 0 off
    air_snail: tuple = ()        # (near, take): an attack from a jump whose area picks a snail (AIR_PICK 'area')
                                 # under `near` units ahead takes it only with chance `take`, else hits nothing.
                                 # In the live hunts of 2026-10-21 such jump throws hurt or killed the snail 6
                                 # times in 29 under 46 units off, against 28 in 48 at 46-130: (46, 0.35); () off
    heading_unsure: float = 0.0  # share of frames whose picture of her can't tell which way she faces (perception.
                                 # heading), drawn per frame: on the old recordings 1.4% of the frames read with
                                 # her own pictures, 1 of 20 at 1366x768, each scene read with pictures of the
                                 # others (on a rope it never can: FakeEyes)
    turned: tuple = ()           # ((at, side), ...): s since the start when something that isn't her keys turns her
                                 # (her player's own arrow, ...): her frame shows it, her keys don't know
    near_seen: tuple = ()        # ((up to units, chance), ...): a monster on the platform she's on (or last stood on)
                                 # this near her, middle to middle, is found by a look only with that chance (drawn
                                 # per monster and look, as detect_miss), the first band that holds it; farther, as
                                 # detect_miss has it. NEAR_SEEN: as live (NEAR_SEEN_1025: 2026-10-25's); () off
    off_seen: float | None = None  # ... and one on another platform with this chance (OFF_SEEN: as live); None: as
                                   # detect_miss has it
    near_spell: tuple = ()       # ((up to units, s), ...): with near_seen, whether a look finds a monster on her
                                 # platform drawn once a spell of that many s (the first band that holds it, its phase
                                 # drawn per monster), not per look: live's misses come in runs (NEAR_SPELL_1102); ()
                                 # per look
    off_spell: float = 0.0       # ... and on another platform, with off_seen (OFF_SPELL_1102); 0 per look
    exact: bool = False          # the feed has the exact finder (its misses count: tracking.Track.blind): NEAR_SEEN_1104
    spawn_found: tuple = ()      # ((alpha, share), ...): a monster back at its spawn point fades in (SPAWN_FADE_S) and
                                 # isn't found before its alpha reaches one drawn from this table, its alpha in what's
                                 # seen while it fades (SPAWN_FOUND: as live with the spawn watch; SPAWN_FOUND_WIDE: the
                                 # whole screen alone); () off: drawn solid, and found, from its birth
    air_short: float = 0.0       # chance a monster on the platform she last stood on whose body is short of the star's
                                 # search's first band (point-blank: its far side within reach.SEARCH_START ahead of her)
                                 # takes the star of an attack from a jump all the same, nearer than the search's pick
                                 # (AIR_PICK 'search'; drawn per attack and monster, the nearest first) ...
    air_under: float = 0.0       # ... and one in a band's reach sideways but under its window (too low for her height
                                 # then): what the live positions (the tracks' error, the monster's walk and her drift
                                 # to the attack's start) don't tell. TREE_AIR: fitted to the live close-up jump throws
    air_short_ramp: float | None = None  # with it, air_short falls off with how far short of the first band the body
                                 # is: air_short * max(0, 1 - short / air_short_ramp)
    air_squash: float = 0.0      # an attack from a jump (AIR_PICK 'search') takes each monster's top up to this share
                                 # of its height lower (drawn per attack and monster, uniform): the likelier picked the
                                 # farther its top reaches into the search's window. TREE_AIR: 1.0, fitted
    air_turn_lost: float = 0.0   # chance an arrow the game acts on in the air doesn't turn her (the attack goes the way
                                 # she faced, and her frame shows her so): live 63 of 246 jump throws that turned her in
                                 # the air faced away from the target after them (26%), the attack 60 ms after the arrow
    heading_wrong: tuple = ()    # (share, score, margin, runs): that share of frames read her facing the wrong way,
                                 # matching `score`, `margin` ahead; `runs` True: drawn per pose of hers for each
                                 # HEADING_SPELL_S (a pose her pictures don't know), False: per frame. Of the frames
                                 # read the wrong way on 2026-10-24 none was strong (heading.Reading.strong): with
                                 # pictures of walks and jumps alone, 9 of 426 frames of throws, at 0.31, 0.24 ahead;
                                 # () off

    @staticmethod
    def during(spells: tuple, t: float) -> bool:
        return any(a <= t < b for a, b in spells)

    @staticmethod
    def found_at(table: tuple, u: float) -> float:
        """The alpha a newborn is first found at, for a draw u in [0, 1) from spawn_found's `table` (its cumulative
        shares, linear between its pairs)."""
        a0, c0 = table[0]
        for a1, c1 in table[1:]:
            if u < c1:
                return a0 + (a1 - a0) * (u - c0) / (c1 - c0) if c1 > c0 else a1
            a0, c0 = a1, c1
        return table[-1][0]


@dataclass(frozen=True)
class Realism:
    """How busy and how hard the map is. The defaults are the simulator's own world (the map at its cap from the start,
    filled again at the map's respawn passes as the server fills it: respawn_tick; nobody else hunting, endless
    potions). Until 2026-10-31 (the user: "Use the tick default everywhere") every spawn point was filled and each
    monster came back 8 s after its own death (RESPAWN, now `respawn_tick` 0), denser and easier than the game: tour_run
    made about 143 EXP a minute where a sweep in game made about 76. The other values are knobs for comparing policies
    in a world closer to the game's; none of them is measured but lock_arrows and the respawn."""
    respawn: float = RESPAWN     # with respawn_tick 0: s from a kill until the monster is back at its spawn point
    respawn_tick: float = TICK_S  # s between the map's respawn passes (TICK_S: the game's, as measured): at each the
                                 # map filled to its cap (None: the server's formula, MapModel.mob_cap, by the players
                                 # there, her, the camps and the rival; a number: that share of its spawn points) at
                                 # random points clear of monsters by SPAWN_CLEAR; 0: `respawn` s after its death, each
                                 # on its own (the default until 2026-10-31; harness `--world clock-respawn`)
    respawn_cap: float | None = None
    # A scenario's set-up of the map's pass (with respawn_tick): the monsters of these indices (the scenario's, in its
    # order) dead long since at the start, their points free for the first pass, and that pass this many s from the
    # start (None: at a phase drawn, as every run's); the spawn guard's set-piece (loop_tree_respawn_pass)
    vacant: tuple = ()
    first_pass: float | None = None
    full_tabs: tuple = ()        # her inventory's tabs full from the start ("Use", "Etc": a drop of one refused with
                                 # FULL_MESSAGE; the harness's `--world etc-full`); (): room for everything
    fills_after: tuple = ()      # ((tab, n), ...): a tab full once she has picked up n items into it (the harness's
                                 # `--world fills-up`); a kind she picked into it before still fits (a part-filled
                                 # stack: v83's full tab has no free slot, her stacks take more), a new kind doesn't
    room_at: float | None = None  # s: her full tabs have room again then (she sold or dropped things: the harness's
                                  # `--world room-later`); None: never
    spawn_share: float = 1.0     # share of the map's spawn points that have a monster (seeded)
    hurt: tuple = ()             # ((kind, share), ...): the map's monsters of that kind there at the start have that
                                 # share of their HP left, their bars up (hurt by someone before her: as by the crowd
                                 # live); back whole once killed. A scenario's set-up: a monster to finish certain from
                                 # the start (the regular attack's doubt: loop_tree_grind_attack_key)
    rival_every: float = 0.0     # s between another player's kills, on average (0: nobody else)
    rival_owns: float = 15.0     # s a rival's drops can't be picked up by her
    rival_takes: float = 20.0    # s after landing that a rival picks up its own drops
    rival_stay: float = 90.0     # s a rival hunts one place before it walks to another, on average (RIVAL_REACH)
    potions: int | None = None   # potions she has (None: endless)
    darts: int | None = None     # throwing stars she has (None: endless): a Lucky Seven cast takes SKILL_STARS, a
                                 # regular throw one (a claw swipe none); with too few the Lucky Seven key does
                                 # nothing (no MP, no attack, no lock: as live on 2026-10-26 01:28, behavior.darts) and
                                 # the attack key only swipes (the stab, if a monster is in its box: else nothing)
    drops: tuple | str = ()      # what a kill drops: (monster kind, drop kind, chance) (GAME_DROPS), drawn per
                                 # kill from a generator of their own, or MAP_DROPS: the world's map's (game_drops);
                                 # (): mesos, and a pig's leather (ITEM_OF)
    drop_mix: DropMix | None = None  # what a kill drops, fitted to the live service's sightings (TREE_DROP_MIX: the
                                     # harness's `--world map-drops`), from a generator of its own; None: `drops`
    drop_takers: bool = False    # her drops taken by others once past rival_owns, at TAKE_RATE a minute (others_life:
                                 # the harness's `--world drop-takers`), from a generator of its own, the rest gone at
                                 # the server's first expiry pass past DROP_LIFE (EXPIRE_PASS_S); False: they lie
                                 # DROP_LIFE
    splices: bool = True         # the player takes splices (sim.fakes.FakePlayer.splices, control.client.can_splice);
                                 # False: a player started before them, every replanned chain a cut and a resend (the
                                 # harness's `--world no-splices`: the elevated player live until it's restarted)
    camps: tuple = ()            # other players camping (Camp, CAMP_KILL): each kills the monsters of its kinds on
                                 # its platforms soon after they show
    shared_exp: bool = False     # another player's kill of a monster she hurt gives her its EXP times the share of its
                                 # HP she took (whether the game does isn't known: the study's session 2 asks)
    lock_arrows: bool = False    # an arrow pressed within ARROW_LOCK of an attack's key moves her only from
                                 # ARROW_HELD after it, if still held (as measured in game on 2026-10-21;
                                 # control.actions holds a plan's first arrow back for it). Preset game and
                                 # the noisy scenarios have it; the default world lets her walk at once
    # What isn't known of the game's physics (SIDE_INSET says what each is):
    sides: str = "platforms"     # the map's sides: 'platforms' (its platforms' outer ends, in the air) or 'client'
                                 # (SIDE_INSET in from its footholds' extent, on the ground too)
    zero_walls: bool = True      # the walls of foothold group 0 stop her whatever HeavenClient's rule says (body.
                                 # wall_table; as live: the comment above SIDE_INSET); False: its rule alone
                                 # (`--world joined-walls`)
    rope_reach: float = ROPE_REACH  # how far above her feet a rope's bottom can be for her to grab it
                                    # (ROPE_REACH_CLIENT: HeavenClient's)
    her_box: tuple[float, float] | None = None  # her body for touches, (half its width, its height): None, HER_TOUCH
                                                # (the client's touch test: a line 50 tall, from 2026-11-02);
                                                # (HER_HALF, HER_HEIGHT), as until then; HER_BOX_CLIENT, the
                                                # planner's (motion.bodies)
    rope_knocks: bool = False    # a touch knocks her off a rope (as the simulator had it until 2026-10-27); False: she
                                 # stays on it, her HP taken and her keys still held (up climbing on), as live on
                                 # 2026-10-26: 11 of 11 hits on a rope left her on it (behavior.darts' false stops)
    knock_turns: bool = False    # a touch's knock turns her to face the monster: HeavenClient's Player::damage doesn't
                                 # (it never calls set_direction), and in the old recordings she never turned without
                                 # an arrow, but no recording has a knock with no arrow held (README, Her heading)
    bars_empty: bool = False     # a hurt monster's HP bar read empty, whatever it has left (sim.fakes.FakeEyes.bar): as
                                 # the stack read them at 1366x768 until 2026-10-25 (perception.hud.hp_bar: 83 of 87
                                 # bars seen in a run read 0; harness `--world bars-empty`)
    air_pick: str | None = None  # how an attack from a jump picks (AIR_PICK's names; None: AIR_PICK): a scenario of what
                                 # an earlier picture had (probe_air: the area's) keeps to it
    damage_ms: int | None = None  # ms after an attack's key that its damage is applied (None: the server's,
                                  # actions.DAMAGE_SHOWS_MS; actions.STAR_DELAY_MS: at the star's release, as the
                                  # world had it until 2026-10-26 and the probes' scenarios still do: their scorer,
                                  # probe.BEFORE_HIT_S, is built on it)
    dies_late: bool = True       # a monster her attack killed is drawn alive until her star lands (DIE_LANDS_S,
                                 # DIE_FLIGHT), then dying where it got to, and her frames lose it DIE_LOST_S on, as
                                 # live on 2026-10-26; False: drawn dying where it was at the kill (the server's time),
                                 # its bar gone, from then (as until 2026-10-27)
    chase: bool = True           # monsters she hits chase her AGGRO_S, each hit anew (aggro, as the game has it; the
                                 # default from 2026-10-31); False: they walk as with nobody there, hit or not (the
                                 # harness's `--world no-aggro`)
    invulnerable: float = INVULNERABLE  # s a touch leaves her invulnerable (INVULNERABLE_LIVE: as live)
    newborns_touch: bool = False  # a monster the map has just spawned touches her from its birth (as the world had it
                                  # until 2026-11-02: the harness's `--world newborns-touch`); False: only once it has
                                  # faded in, NEWBORN_SOLID_S after (as live)
    hit_inside: bool = False      # a star's line checked to its hit point taken 1 unit inside the monster's platform
                                  # alone (as every world had it until 2026-11-03: the harness's `--world hit-inside`);
                                  # False: the client's own hit point too where it's past the platform's end (its body
                                  # over a wall rising from there, up close: sight_stop, reach.CLIENT_HIT_POINT)
    skill_row: float | None = None  # how far ahead of her a skill's (Lucky Seven's) shot row starts for the line's hit
                                  # points (reach.hit_point's `start`): None, reach.SEARCH_START's 65 as for the regular
                                  # attack (the stack's assumption, R6's); 0.0 at her x, RustMS's port's for a skill
                                  # (the harness's `--world hit-row-0`; the review of 2026-11-03)
    knock: tuple[float, float, bool, bool] | None = None  # her knock (vx, vy, whether its push sideways adds to her
                                  # speed, whether she lands from it at her walk's speed at most): None, the world's own
                                  # (KNOCK_VX, KNOCK_VY, KNOCK_ADDS, KNOCK_LANDS_WALKING: the client's, as R1-R3's frames
                                  # have it, from 2026-11-02); KNOCK_PLANNER, the planner's (motion.physics) set
                                  # outright, as until then (the harness's `--world planner-knock`)
    # The game's own key bindings where they aren't her keys' (behavior.keycheck): (her GameKeys field, the key the game
    # has it on, None: on no key), e.g. (("attack", "a"),): her configured attack key then does nothing (as live on
    # 2026-10-28 00:30, the user's own play having moved it), A attacks. The backslash opens the KEY BINDINGS window
    # (World.key_window: what the simulator's eyes draw, sim.fakes.FakeEyes._capture) and Esc closes it; Esc with it
    # closed opens the game's MENU (counted: World.menu_opened); keys pressed while it's open do nothing (counted: an
    # assumption about the game, not seen).
    game_keys: tuple = ()
    # Her stat window (behavior.statcheck): her stats key (`stat_key`: S, ABILITY as live; None: on no key) toggles it,
    # drawn from the game data's pictures with her stats (sim.fakes.FakeEyes._capture: perception.stat_window.compose,
    # at its DRAW_AT), its DETAIL panel opened by a click on its button (`stat_detail`: open with the window from the
    # first, as a game that remembers it would), closed by Esc as the KEY BINDINGS window is; `stat_stuck`: neither her
    # stats key nor Esc closes it (a test's: the window may be open). Keys work as ever with it open (it isn't modal).
    stat_key: str | None = "s"
    stat_detail: bool = False
    stat_stuck: bool = False
    # Her chat box (behavior.chatcheck): None, the simulator's eyes read none (sim.fakes.FakeEyes: Observation.chat None,
    # as before); "expanded" or "collapsed", the box as the game shows it at the start (perception.chatbox.ChatBox in
    # every look), her chat key (', CHAT+ as live) toggling it; `chat_stuck`: the key does nothing to it (a test's).
    chat: str | None = None
    chat_stuck: bool = False
    pets: int = 0                # her pets out, looting as they trail her (PET_SPEED ...; the harness's `--world
                                 # pet-loot`); 0: none takes anything (her pet's potions and food are Her.pet's)


@dataclass(frozen=True)
class Camp:
    """Another player camping (Realism.camps, CAMP_KILL): where it stands (its minimap dot), the platforms whose
    monsters it kills, and of which kinds."""
    stand: int
    x: float
    hunts: tuple[int, ...]
    kinds: tuple[str, ...]


@dataclass(frozen=True)
class Her:
    """Her as the simulator plays her: her HP (what a touch takes is a share of it), her level and EXP when a run
    starts (her EXP number: sim.fakes), how a touch's damage is worked out, and her pet. LEVEL_19 is her on the first
    map, where the golden traces and the harness's runs were fitted; LEVEL_23 on West Rocky Mountain II (from
    2026-10-23: level 23, HP 730 and MP 584 by her bars there, her pet drinking her potions). Her MP is kept where
    `mp` is set (LEVEL_24): then the Lucky Seven key's casts cost it (`skill`, CAST_MP)."""
    max_hp: float = MAX_HP
    level: int = 19
    exp: int = 421               # EXP into her level (as on the saved frames of 2026-10-18)
    defence: float | None = None  # her DEF in meowdb's touch formula (DEFENCE); None: a share of PADamage, as fitted
    pet: float | None = None     # her pet's HP warning: it drinks a potion of hers when a touch leaves her under this
                                 # share of her HP (PET_GAP); None: no pet
    evade: float = 0.0           # her evasion against a touch, as EVADE_ACC (her chance is this over the monster's
                                 # accuracy); 0: EVADE_ACC, as fitted
    avoid: float | None = None   # her avoid (her stat window's): a touch MISSes by this game's hit check
                                 # (game_hit_chance, against the monster's accuracy and level); None: `evade`
    potion: float | None = None  # HP a potion of hers heals (her key's and her pet's); None: POTION_HP of her HP
    stars: float = 1.0           # what one of her attacks does, in stars: 1 the regular attack (a swipe CLAW of one);
                                 # LUCKY_SEVEN, Lucky Seven as thrown live (a swipe's pick as much)
    mp: float = 0.0              # her MP when full (MP_REGEN, CAST_MP); 0: not kept
    skill: float | None = None   # what the Lucky Seven key's attack does, in stars (LUCKY_SEVEN; CAST_MP a cast, and
                                 # nothing with less MP); None: as the attack key's (`stars`), whatever her MP
    pet_mp: float | None = None  # her pet's MP warning (PET_MP): it drinks an MP potion of hers (MP_POTION) when a cast
                                 # leaves her under this share of her MP; None: it doesn't
    spread: bool = False         # each of her stars' damage drawn (STAR_DAMAGE, a whole number as likely each, less the
                                 # monster's defence) against its HP in the game data, not a star's worth each
    star: tuple[int, int] | None = None  # with `spread`, her regular attack's star before defence (None: STAR_DAMAGE)
    skill_star: tuple[int, int] | None = None  # ... and each of Lucky Seven's SKILL_STARS stars (None: `skill` /
                                               # SKILL_STARS of the regular star)
    cast_mp: float = CAST_MP     # MP a Lucky Seven cast takes
    mp_potion: float = MP_POTION  # MP an MP potion her pet drinks gives
    claw: tuple[int, int] | None = None  # with `spread`, a claw swipe's damage before defence (None: CLAW_DAMAGE). With
                                 # `skill`, the Lucky Seven key swipes up close as the attack key does, its cast's MP
                                 # and stars not taken (World._uncast): live, a Lucky Seven press with the target 20-90
                                 # ahead on her platform took MP 54 times in 200 (27%; 76% at 100-120, where the claw's
                                 # boxes end), and by eye on the recordings of 2026-10-26 every press at one on or under
                                 # her was a swipe of 15-24 or a MISS, no MP
    walk_speed: float | None = None  # map units a second the world walks her at (her_shoe: the game's Physics.img shoe at
                                 # the SPEED that walks her so, her JUMP the planner's); None: her truth, the client's
                                 # formula at her profile's SPEED (physics.PROFILE_SHOE: the thief's 151.25), whatever
                                 # her planner has (until 2026-11-02 the planner's own, physics.SHOE). Apart from the
                                 # planner's, a wrong speed in her profile shows in the world (her steps overshoot, her
                                 # jumps after a run carry farther than planned), as it did live (MAGE_32)
    climb_speed: float | None = None  # ... and climbs her at, up and down (her_climb); None: her truth, SPEED + 20 at her
                                 # profile's SPEED (physics.PROFILE_CLIMB: the thief's 121)
    planned: tuple[float, float] | None = None  # her planner's walk and climb (units/s) held at these after her hunter's
                                 # start, apart from her stat window's SPEED (physics.set_planned: Sim.hunter), the
                                 # world moving her at its own (walk_speed, climb_speed): what a planner's error costs
                                 # her (sim.harness's `planner-v83`); None: her planner by her stat window's SPEED, as
                                 # live (in the simulator her profile's)
    mp_regen: float = MP_REGEN   # MP a second she regains all the while (Her.mp kept)
    magic: SimMagic | None = None  # her skill key's attack Magic Claw as the game casts it (SimMagic: the magician's,
                                 # MAGE_32), her MP its ammunition (`cast_mp` a cast, nothing with less, MP_REFUSAL
                                 # said); None: the thief's attacks as above


def her_shoe(her: Her) -> body.Shoe:
    """The shoe the world walks and jumps her with: her truth, the client's at her profile's SPEED and JUMP
    (physics.PROFILE_SHOE: never her planner's, which her stat window's SPEED sets), unless Her.walk_speed says
    otherwise, then the game's Physics.img shoe at the SPEED that walks her at it (walkSpeed x SPEED / 100), her JUMP the
    planner's. Only her walk differs: the acceleration, braking and the air's drag are the game data's for everyone."""
    if her.walk_speed is None:
        return physics.PROFILE_SHOE
    speed = 100.0 * her.walk_speed / body.PHYSICS_IMG["walkSpeed"]
    return body.Shoe.from_physics(body.PHYSICS_IMG, speed, physics.CHARACTER.jump)


def her_climb(her: Her) -> float:
    """How fast (units/s) the world climbs her, up and down: her truth, SPEED + 20 at her profile's SPEED
    (physics.PROFILE_CLIMB), unless Her.climb_speed says otherwise."""
    return physics.PROFILE_CLIMB if her.climb_speed is None else float(her.climb_speed)


LEVEL_19 = Her()
# Her pet's warning isn't known (a multiple of 5% in System Options): 45% is where the tour and the hunt loop have her
# drink herself (Hunter's potion_hp in Sim.tour_stack and sim.harness), so the pet does what her own potions did.
LEVEL_23 = Her(max_hp=730.0, level=23, exp=31_736, defence=DEFENCE, pet=0.45)
# On Tree Dungeon, Forest Up North II (from 2026-10-24): level 24, HP 752 and MP 601 (the study's: her bars), her EXP
# into the level not read yet (0). Her MP kept: her casts of Lucky Seven take CAST_MP each, her pet drinks for her under
# PET_MP (MP_REGEN, MP_POTION); with `skill` None (the default) the Lucky Seven key throws as the attack key does. A
# touch by meowdb's formula at DEFENCE: a Horny Mushroom (PADamage 110) takes 111-131 HP,
# 15-17% of hers (the study's 113-134 at DEF 64), a Green (54) 49-59, a Slime 7-9; by a share of PADamage as fitted
# on the first map (`defence` None: the harness's `--touch share`) 75, 37 and 9, give or take 70%. Which holds isn't
# known: one Horny touch's px on her HP bar (17 px of 104 by the formula, 10 by the share, at 1366x768) tells.
# Her potions there, as the live hunt of 2026-10-25 had them (23:43-23:54, the perception service's 10 fps HP readings,
# lab/telemetry/20261025-210747-perceive.jsonl; D:/MapleHelper-scratch/analysis-1025/damage-skeptic threshold.py): her pet
# drank an HP potion only when a touch left her under 70% (every reading under 0.70 was followed by a rise within 0.12-
# 0.16 s, none over it: 0.712-0.721 held 1.5 s, 0.769 for 52 s), 6 times in 10.95 minutes, each 230-250 HP (one ended
# uncapped at 0.971 from 0.65: 250 HP of her 774, an Orange Potion's: PET_HP, HP_POTION). Before, the pet's warning was
# 0.45 (where the stack had her drink herself) and a potion POTION_HP's 30%, 226 HP: the simulator drank later and
# less a potion. (The 15 smaller rises of that run, about 100 HP each, came 0.4-0.6 s before an MP refill from an
# empty bar: the MP item's, not modelled.) The harness's `--world red-potion` has her drink Red Potions (100 HP).
PET_HP = 0.70
HP_POTION = 250.0
# Her pet's HP potion as she grinds from 2026-11-01 (the user: "Usually 200 or 300 pieces of meat (150HP)"): Meat (item
# 2010003), which heals 160 HP by the game data (Item/Consume 0201.img: spec hp 160, the Red Potion's 100, the
# Orange's 250), as live (63 rises of exactly 160 HP on 2026-10-30 at 22:14, the magician's pet: lab/telemetry/
# 20261030-221417-perceive-part2.jsonl; D:/MapleHelper-scratch/replan/loot2/design), where the user remembered 150;
# POTIONS of them a run, the middle of their 200-300. The harness's `--world meat` (Her.potion, Realism.potions), her
# pet's warning still PET_HP (measured with Orange Potions on the thief: with Meat it isn't known). The crowd preset's
# 60 Orange Potions ran out within 15 minutes, the safety stop ending 7 of 12 of the standard grind's runs.
MEAT_HP = 160.0
POTIONS = 250
LEVEL_24 = Her(max_hp=752.0, level=24, exp=0, defence=DEFENCE, pet=PET_HP, mp=601.0, pet_mp=PET_MP, potion=HP_POTION)
# ... and from 2026-10-25 at level 25: HP 774 and MP 623 (her bars' numbers), her stars as read off the monsters' bars
# then (attack.choice.STAR and SKILL_STAR: the regular star 20-140, Lucky Seven's two of 96-192), drawn so with `spread`
# (harness.ATTACK_HER), her Lucky Seven at level 16 (14 MP), her pet's MP potions Blue Potions (100) and its HP
# potions as above.
LEVEL_25 = Her(max_hp=774.0, level=25, exp=0, defence=DEFENCE, pet=PET_HP, mp=623.0, pet_mp=PET_MP, potion=HP_POTION,
               star=(20, 140), skill_star=(96, 192), cast_mp=14.0, mp_potion=100.0, claw=(15, 24), avoid=LEVEL_25_AVOID)
# The magician, RiceBaby, as she grinds Tree Dungeon from 2026-11-02 (ROADMAP item 8; the user, that morning: today's
# measurements are the simulator's truth for her), from R1-R3 and the hour's telemetry (00:17:48-00:53:06; the comment
# above CLAW_HIT has her Magic Claw and her MP):
# - level 32, HP 858, MP 1549, W.DEF 75, AVOID 16: her stat window, read 40 times that night (lab/characters/magician/
#   her_stats.json; INT 143, MAGIC 126). At AVOID 16 every touch hurts (game_hit_chance: 0 MISS in the hour's 170
#   touches, the formula's 8% and less on Tree's kinds), at DEF 75 a Horny Mushroom's 110-130 and a Green's 50-58, as
#   live (attack.danger.touch_taken, meowdb's formula);
# - her walk the client's, 150 units/s (walk_speed: MAGE_WALK, below), as her planner plans it from item 1 (her
#   profile's SPEED 100): the world walks her at its own all the same (Her.walk_speed), so a planner's error shows;
#   her climb the client's too, 120 up and down (climb_speed: MAGE_CLIMB), as her planner plans it;
# - her pet drinks Meat, 160 HP (MEAT_HP: on R2's frames her HUD's number 606 -> 766, 601 -> 761, 639 -> 799 and
#   562 -> 722, the frame after a touch; D:/MapleHelper-scratch/recfix/sim-mage/hp_seq.png, hp_seq2.png; the hour's 92
#   rises all 160; the game data's spec hp 160, where the user remembered 150), under 75% of her HP (MAGE_PET_HP: the
#   hour's 55 drops drunk at, all at 0.748 or under, none at 0.752 or over, held 1.5-30 s: critique-simulator-work/
#   pet_threshold.py; her live profile still has 0.55); her one pet, a Snail, taking nothing (her pets' pick-ups aren't
#   known: Realism.pets 0);
# - her HP's own regeneration isn't kept (the thief's isn't either): live it covered about a tenth of her touches' HP.
MAGE_PET_HP = 0.75
# Her walk in the world: the client's, 125 x (SPEED + 20)/100 at her stat window's SPEED 100 (motion.body.SPEED_BONUS,
# meowdb's measured formula; README "Her walk speed"): 150 units/s. Her steady limit with each walk's braking tail and
# Tree's slopes taken out is 149.4-149.7 (the hour's 66 walks: median 149.44, 90% 149.1-150.0; mean 149.67): the world
# brakes her by itself, and walks her slopes at flat ground's speed (it has no slope rule, as the planner hasn't). From
# the integration of 2026-11-02 (work/rec-all); before, the world walked her at 148.2, the hour's raw median of 81
# steady walks with both in it (P01), while her planner planned v83's 125.
MAGE_WALK = body.Shoe.from_physics(body.PHYSICS_IMG, 100.0, 100.0, bonus=body.SPEED_BONUS).vmax  # 150.0
MAGE_CLIMB = physics.climb_of(100.0)  # 120.0: her climb, SPEED + 20 (she climbed 119.9 up and 119.6 down that night)
MAGE_32 = Her(max_hp=858.0, level=32, exp=0, defence=75.0, avoid=16.0, pet=MAGE_PET_HP, potion=MEAT_HP, mp=1549.0,
              cast_mp=20.0, mp_regen=MAGE_MP_REGEN, pet_mp=MAGE_PET_MP, mp_potion=MAGE_MP_POTION, walk_speed=MAGE_WALK,
              climb_speed=MAGE_CLIMB, magic=SimMagic())


# --- monsters ----------------------------------------------------------------------------------------

MOBS = OUT / "mobs"      # what the simulator reads of each monster from the game data, cached as JSON (as
                         # mapdata caches maps): the simulator runs the same off the caches alone
FRAME_MS = 100           # how long a frame shows when the game data gives it no delay (the client's default)


@dataclass(frozen=True)
class Frame:
    """One frame of a monster's animation in the game data: how long it shows (delay), and its body then
    (its box, lt/rb, which is what touches her): from `lo` to `hi` sideways of its feet as drawn, facing
    left, and from `bottom` to `top` above them (a hop lifts its bottom off the ground)."""
    ms: int
    lo: float
    hi: float
    bottom: float
    top: float


@dataclass(frozen=True)
class MobInfo:
    """A monster as the game data has it: how much a touch by it takes (PADamage, before her defence), its
    accuracy (acc), which her evasion goes against, and its frames standing, walking and flinching after a
    hit (hit1). A walk cycle can lift it: a slime is off the ground for 400 of its 1,050 ms (its box 10-17
    above its feet, up to 66 tall where it stands 42-44), a pig 4 units for 300 of 500, an Orange Mushroom
    3 for 120 of 480 (67 tall where it stands 55-58). Its HP (maxHP) and defence (PDDamage) say how many stars
    kill it (stars_to_kill); 0 where not known."""
    pad: float
    acc: float
    stand: tuple[Frame, ...] = ()
    move: tuple[Frame, ...] = ()
    hit: Frame | None = None
    hp: float = 0.0
    pdd: float = 0.0
    pushed: float = 1.0          # the least damage that pushes it back (and flinches it): PUSH_BY_DAMAGE

    @staticmethod
    def frame_at(frames: tuple[Frame, ...], ms: float) -> Frame:
        """The frame an animation of `frames` (looping) shows `ms` after it began."""
        k = ms % sum(f.ms for f in frames)
        for f in frames:
            if k < f.ms:
                return f
            k -= f.ms
        return frames[-1]


@functools.lru_cache(maxsize=None)
def mob_info(mob_id: int) -> MobInfo | None:
    """Monster `mob_id`'s MobInfo, from the game data (cached in MOBS), or None where there's none."""
    cache = MOBS / f"{mob_id:07d}.json"
    raw = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else None
    if raw is None or "move" not in raw or "pushed" not in raw:  # not cached yet, or before its frames, HP or push
                                                                 # were kept
        fresh = _read_mob(mob_id)
        if fresh is None and (raw is None or "move" not in raw):
            return None
        if fresh is not None:  # (without the game data copy, a cache from before its HP was kept does as it is)
            raw = fresh
            cache.parent.mkdir(parents=True, exist_ok=True)
            part = cache.with_suffix(f".{os.getpid()}.part")  # whole, or not there: harness workers read it too
            part.write_text(json.dumps(raw), encoding="utf-8")
            os.replace(part, cache)
    stand, move = (tuple(Frame(*f) for f in raw[k]) for k in ("stand", "move"))
    return MobInfo(raw["pad"], raw["acc"], stand or move, move or stand, raw["hit"] and Frame(*raw["hit"]),
                   raw.get("hp", 0.0), raw.get("pdd", 0.0), raw.get("pushed", 1.0))


def _read_mob(mob_id: int) -> dict | None:
    """MobInfo's fields for monster `mob_id`, from its image in the game data copy (None without it). A
    monster that borrows another's frames (info/link) takes them from that one's image; an animation
    marked zigzag plays forward then back."""
    try:
        from gamedata.canvas import _mob_pack
        from gamedata.wz import Canvas
        pack = _mob_pack()
        tree = pack.image(f"Mob/{mob_id:07d}.img")
        info = tree.get("info") or {}
        if info.get("link") and not tree.get("stand"):
            tree = pack.image(f"Mob/{int(info['link']):07d}.img")
    except (OSError, KeyError, ValueError):  # no copy of the game data, or no such monster in it
        return None

    def frames(action: str) -> list[list]:
        node = tree.get(action) or {}
        out = []
        for index, f in sorted(node.items(), key=lambda kv: (len(kv[0]), kv[0])):
            lt, rb = (f.props.get("lt"), f.props.get("rb")) if isinstance(f, Canvas) else (None, None)
            if index.isdigit() and lt is not None and rb is not None:  # y is up negative in the data
                out.append([int(f.props.get("delay") or FRAME_MS), float(lt.x), float(rb.x), float(-rb.y),
                            float(-lt.y)])
        return out + out[-2:0:-1] if node.get("zigzag") else out

    hit = frames("hit1")
    return {"pad": float(info.get("PADamage") or 0), "acc": float(info.get("acc") or 0),
            "stand": frames("stand"), "move": frames("move"), "hit": hit[0] if hit else None,
            "hp": float(info.get("maxHP") or 0), "pdd": float(info.get("PDDamage") or 0),
            "pushed": float(info.get("pushed") or 1)}


def touch_damage(info: MobInfo | None, u: float, her: Her = LEVEL_19) -> float:
    """The share of her HP a touch by a monster with `info` takes, `u` in [0, 1) placing it within the
    spread: TOUCH_SHARE of its touch damage, give or take TOUCH_SPREAD; or, with her DEF known (Her.defence),
    as meowdb's formula has it (DEFENCE)."""
    if info is None or info.pad <= 0:
        return TOUCH_DAMAGE
    if her.defence is None:
        return info.pad * TOUCH_SHARE * (1 + TOUCH_SPREAD * (2 * u - 1)) / her.max_hp
    return touch_taken(info.pad, her.defence, her.level, u) / her.max_hp  # (the stack's own: attack.danger)


@functools.lru_cache(maxsize=None)
def stars_to_kill(hp: float, pdd: float = 0.0) -> float:
    """How many of her stars kill a monster of `hp` HP and defence `pdd` on average: each star a whole number of
    damage from STAR_DAMAGE's least to its most, as likely each, less the defence (x 100 / (pdd + 100))."""
    lo, hi = STAR_DAMAGE
    need = hp * (pdd + 100) / 100      # in damage before the defence
    sums, stars = {0: 1.0}, 0.0       # the chance of each total of the stars so far, among those short of `need`
    while sums:
        stars += sum(sums.values())   # the chance one more star is needed
        nxt: dict[int, float] = {}
        for total, chance in sums.items():
            for k in range(lo, hi + 1):
                if total + k < need:
                    nxt[total + k] = nxt.get(total + k, 0.0) + chance / (hi - lo + 1)
        sums = nxt
    return stars


def hits_of(kind: str, info: MobInfo | None) -> int:
    """A monster's HP in stars: HITS' for its kind, else by its HP and defence in the game data (stars_to_kill,
    rounded), else HITS_UNKNOWN."""
    if kind in HITS:
        return HITS[kind]
    if info is None or info.hp <= 0:
        return HITS_UNKNOWN
    return max(1, round(stars_to_kill(info.hp, info.pdd)))


def evade_chance(info: MobInfo | None, her: Her | None = None, mob_level: float | None = None) -> float:
    """The chance she evades a touch by a monster with `info` (of `mob_level`, the game data's): with her avoid known
    (Her.avoid) and its level, 1 less this game's hit check (game_hit_chance); else EVADE_ACC (or her own, Her.evade)
    over its accuracy."""
    if info is None or info.acc <= 0:
        return EVADE
    if her is not None and her.avoid is not None and mob_level is not None:
        return 1.0 - game_hit_chance(info.acc, mob_level, her.avoid, her.level)
    return min(1.0, (her.evade if her is not None and her.evade else EVADE_ACC) / info.acc)


def game_hit_chance(acc: float, mob_level: float, avoid: float, level: float) -> float:
    """The chance a touch by a monster of accuracy `acc` and level `mob_level` hits her (avoid `avoid`, level `level`):
    this game's hit check (see LEVEL_25_AVOID)."""
    gap = max(0.0, level - mob_level)
    score = acc * 100 / ((gap + 51) * 5)
    eff = avoid / (avoid / 80 + 1) / (gap / 40 + 1)
    spread = 0.15 + 0.2 / (1 + math.exp((score - eff) / 12))
    roll = 1.0 if eff <= 0 else min(1.0, max(0.0, (1 + spread - eff / score) / (2 * spread)))
    rescue = min(0.03, max(0.02, 0.03 * math.exp((score - eff) / 18))) if eff > (1 + spread) * score else 0.0
    return roll + (1 - roll) * (rescue + (1 - rescue) * 0.08)


@functools.lru_cache(maxsize=None)
def push_chance(pushed: float, pdd: float, swipe: bool = False) -> float:
    """The chance a hit of hers pushes back a monster of `pushed` and defence `pdd` (PUSH_BY_DAMAGE): of her star's
    damage (STAR_DAMAGE, or a claw swipe's, CLAW_DAMAGE), each whole number as likely, the share that reaches `pushed`
    after the defence."""
    lo, hi = CLAW_DAMAGE if swipe else STAR_DAMAGE
    return sum(1 for d in range(lo, hi + 1) if d * 100 / (pdd + 100) >= pushed) / (hi - lo + 1)


@dataclass
class SimMonster:
    """A monster walking its platform. Its motion is a list of segments drawn from its own rng (walk
    left, walk right or stand, for a while), evaluated lazily, so where it is at a moment doesn't depend
    on how often it was looked at. Its body at a moment is the box of the frame it shows then (body)."""
    idx: int
    name: str
    kind: str
    platform: Platform
    home: float               # spawn x (where it respawns)
    rng: random.Random
    speed: float = 125.0
    half: float = 34.0
    height: float = 46.0
    hp_max: float = 1.0        # in stars
    hp: float = 1.0
    alive: bool = True
    died_at: float | None = None
    died_x: float = 0.0
    hit_at: float | None = None  # when a star last hit it without killing it (Noise.hit_hidden) ...
    flinch_at: float | None = None  # ... and pushed it back: it flinches (PUSH_BY_DAMAGE)
    flinch_top: float = 54.0   # how tall it is while it flinches
    fade: float = FADE         # s it's still seen once killed, fading out where it died
    drawn_alive: float | None = None  # killed (Realism.dies_late): drawn alive until then, its bar `bar_drawn` ...
    bar_drawn: float = 1.0
    killed_by: str = ""        # ... and who killed it ("her", "rival")
    info: MobInfo | None = None  # its touch damage, accuracy and frames (game data), if known
    # current motion segment: from t0 at x0, heading d (-1, 0, +1) until t1
    t0: float = 0.0
    x0: float = 0.0
    d: int = 0
    t1: float = 0.0
    facing: int = -1           # the way it faced when it last stopped walking (-1: left, as it's drawn) ...
    anim0: float = 0.0         # ... and when it began to stand or walk (its animation's start)
    pushed: tuple[float, float] | None = None  # a hit's (when, from x): pushed from there to x0 (knock)
    life: int = 0              # how often it has come back (a camp's kill is for the life it was drawn in)
    born: float | None = None  # when it last came back (None: there from the start)
    # Chasing her once she hits it (Realism.chase): her x for a stretch starting at t (World._chase_at; None: no her to
    # chase), until when it chases her (None: not hit this life), the way its chase goes (0: to be decided), and its
    # chase's own rng (its walk otherwise as ever)
    chase: Callable[["SimMonster", float], float | None] | None = None
    aggro_until: float | None = None
    chase_dir: int = 0
    chase_rng: random.Random | None = None
    point: int | None = None   # the spawn point it came from (World._spawn's order; None: its idx), one the map's pass
                               # made there (Realism.respawn_tick) while the point's own was still fading or walked off

    @property
    def spawn(self) -> int:
        return self.idx if self.point is None else self.point

    @property
    def lo(self) -> float:
        return walk_edges(self.platform)[0]

    @property
    def hi(self) -> float:
        return walk_edges(self.platform)[1]

    def start(self, t: float, x: float) -> None:
        self.t0, self.x0, self.anim0, self.pushed = t, min(max(x, self.lo), self.hi), t, None
        self._draw()

    def _draw(self) -> None:
        if self.aggro_until is not None and self.t0 < self.aggro_until and self._chase_her():
            return
        self.t1 = self.t0 + think_steps(self.rng, IDLE_THINK_MS) * AI_STEP
        self.d = self.rng.randrange(3) - 1

    def _chase_her(self) -> bool:
        """Realism.chase (AGGRO_S): this decision, CHASE_THINK_MS long, walks at her, or on the way it chose while she's
        within AGGRO_PASS across (back to its platform's end if that turned it); False with no her to chase (its own
        walk then)."""
        hx = self.chase(self, self.t0)
        if hx is None:
            return False
        if self.chase_dir == 0 or abs(hx - self.x0) >= AGGRO_PASS:
            self.chase_dir = 1 if hx >= self.x0 else -1
        self.d = self.chase_dir
        self.t1 = self.t0 + think_steps(self.chase_rng, CHASE_THINK_MS) * AI_STEP
        return True

    def aggro(self, t: float) -> None:
        """Hit by her at t (Realism.chase): it chases her AGGRO_S from now, its way decided afresh, at once, or once a
        hit's flinch is over (knock)."""
        if self.chase is None:
            return
        self.at(t)
        self.chase_dir = 0
        self.aggro_until = t + self.chase_rng.uniform(*AGGRO_S)
        if self.pushed is not None and self.pushed[0] == self.t0:
            self.t1 = min(self.t1, self.t0 + monster_data.FLINCH_S)  # (flinching: it chases from the flinch's end)
            return
        (x, d), walked = self._fold(t), self.d
        self.t0, self.x0, self.facing = t, x, d or self.facing
        self._draw()
        if bool(self.d) != bool(walked):
            self.anim0 = t

    def _fold(self, t: float) -> tuple[float, int]:
        """Where it is at t within the current segment, and which way it's then going (bouncing at the ends)."""
        if not self.d:
            if self.pushed is not None and self.pushed[0] == self.t0 and t < self.t0 + KNOCK_MOB_S:
                return pushed_x(self.pushed[1], self.x0, t - self.t0), 0
            return self.x0, 0
        span = self.hi - self.lo
        if span <= 0:
            return self.lo, self.d
        u = (self.x0 - self.lo) + self.d * self.speed * (t - self.t0)
        m = u % (2 * span)
        return (self.lo + m, self.d) if m <= span else (self.lo + 2 * span - m, -self.d)

    def ghost(self) -> "SimMonster":
        """A copy to look ahead with (`at` a later time): its own rng copied, so the monster itself goes on drawing
        the same segments, frame by frame. Advanced itself, a monster her attack killed was drawn off its own path from
        the kill to her star's landing on 18% of 2,000 such kills (a median 27 units, the most 108: review of
        2026-10-27, review-behaviour/simjump.py), the frames between evaluating the newer segment backwards."""
        g = copy.copy(self)
        g.rng = random.Random()
        g.rng.setstate(self.rng.getstate())
        if self.chase_rng is not None:  # (and its chase's: Realism.chase)
            g.chase_rng = random.Random()
            g.chase_rng.setstate(self.chase_rng.getstate())
        return g

    def at(self, t: float) -> float:
        while t >= self.t1:
            (x, d), walked = self._fold(self.t1), self.d
            flinched = self.pushed is not None and self.pushed[0] == self.t0
            self.t0, self.x0, self.facing = self.t1, x, d or self.facing
            self._draw()
            if flinched or bool(self.d) != bool(walked):  # its stand or walk starts over (after a hit too)
                self.anim0 = self.t0
        return self._fold(t)[0]

    def faces(self, t: float) -> tuple[int, bool]:
        """Which way it faces at t (+1 right, -1 left), and whether it's walking then (standing or flinching:
        the way it faced, `facing`). `at(t)` first."""
        way = self._fold(t)[1]
        return way or self.facing, bool(way)

    def y(self, x: float) -> float:
        return self.platform.y_at(x)

    def knock(self, t: float, away: int) -> None:
        """Hit at t, from her side: pushed KNOCK_MOB units `away` (+1/-1) over KNOCK_MOB_S, as far as its
        walk goes, facing her, standing until FLINCH_S after the hit and then the rest of its decision (its AI waits
        the hit out). Its walk after is drawn then, as at the end of any of its segments."""
        self.at(t)
        x, _ = self._fold(t)
        left = max(0.0, self.t1 - t)
        self.pushed, self.anim0, self.facing = (t, x), t, -away
        self.t0, self.x0, self.d = t, min(max(x + away * KNOCK_MOB, self.lo), self.hi), 0
        self.t1 = t + monster_data.FLINCH_S + left

    def body(self, t: float, flinching: bool = False) -> tuple[float, float, float, float]:
        """Its body at t: from `lo` to `hi` sideways of its feet and from `bottom` to `top` above them. With
        game data, the box of the frame it shows then (walking, standing, or its hit1 frame while it
        flinches after a hit), turned the way it faces; without, its half width either side, as tall as
        its kind (or its flinch top)."""
        self.at(t)
        if self.info is None or not self.info.move:
            return -self.half, self.half, 0.0, max(self.height, self.flinch_top) if flinching else self.height
        _, d = self._fold(t)
        if flinching and self.info.hit is not None:
            f = self.info.hit
        else:
            f = MobInfo.frame_at(self.info.move if self.d else self.info.stand, (t - self.anim0) * 1000)
        return (f.lo, f.hi, f.bottom, f.top) if (d or self.facing) < 0 else (-f.hi, -f.lo, f.bottom, f.top)


# Monsters that jump (a jump action in their game data, MapModel.mob_hops: Tree Dungeon's Slimes): v83's move ability
# 2 (RustMS's HeavenClient: Mob::step_body, think, think_jump, jump; FootholdTree::limit_jumper, get_wall_ahead;
# Physics::jump), where the simulator walked them as any other (the user, 2026-10-31: "Yes, align them"). Its AI as a
# walker's (AI_STEP, IDLE_THINK_MS, CHASE_THINK_MS), and: one idle decision in JUMP_IDLE a jump (the way it walks, or
# straight up standing), after which it decides again; a rising wall within JUMP_AHEAD ahead (at its platform's end)
# jumped at while its decision lasts, and pressed against one, back the other way; a jump of those that lands as high
# as it left turns it round; chasing her, one AI step in JUMP_CHASE a jump while she's over a floor of another group,
# higher, that reaches across over it. Walking: within its reach (its platform's group's and its home group's across,
# less JUMPER_MARGIN a side where wider than twice that), turning round at its limits; at its platform's end, on down a
# step joined to it (unless chasing her over another group), else it stops and jumps off it the client's tick later
# (EDGE_JUMP_S; chasing her lower, at a free end or a step half the time, and the other half off it walking). Its jump:
# up at the jump speed (555: 77 units high), the way it holds at its walk speed (from standing at 0.8 of it: the kick),
# through the air as her flight (body.fly: its shoe, the walls that stop her and the map's sides, World._flight's),
# landing on the way it jumped; in the air its AI only counts its decision down. Hit in the air, it's stood where it
# lands (Mob::flinch_body); its push as a walker's, as far as its platform and its reach go. False: they walk as the
# others.
JUMPERS = True
JUMP_IDLE = 5
JUMP_CHASE = 3
JUMP_AHEAD = 40.0
JUMPER_MARGIN = 20.0
EDGE_JUMP_S = 0.008
JUMPER_KEEP_S = 10.0  # the stretches kept behind it, for a look a little back in time


@dataclass
class JumpSpace:
    """What a jumping monster needs of its world: the map, the walls and sides that stop it in the air (World._flight's),
    each foothold group's reach across (x from, to) and a shoe for each walk speed."""
    model: MapModel
    walls: tuple
    sides: tuple[float, float] | None
    reach: dict
    shoes: dict

    @classmethod
    def of(cls, model: MapModel, walls, sides) -> "JumpSpace":
        reach: dict = {}
        for g, xs in ([(p.group, (p.left, p.right)) for p in model.platforms]
                      + [(w.group, (w.x1, w.x2)) for w in model.walls if w.group is not None]):
            lo, hi = reach.get(g, (min(xs), max(xs)))
            reach[g] = (min(lo, *xs), max(hi, *xs))
        return cls(model, walls, sides, reach, {})

    def shoe(self, speed: float) -> body.Shoe:
        """A monster's shoe: walking at `speed`, jumping at the jump speed (100%). No SPEED_BONUS: the client's hidden
        20 Speed is a character's (motion.body); a monster walks v83's 125 x (1 + its speed/100), as Tree Dungeon's at
        speed -20 walked 99 and 98 in the recordings of 2026-11-02, against 100."""
        if speed not in self.shoes:
            self.shoes[speed] = body.Shoe.from_physics(body.PHYSICS_IMG, 100 * speed / body.PHYSICS_IMG["walkSpeed"],
                                                       100.0)
        return self.shoes[speed]

    def group_below(self, x: float, y: float):
        """The foothold group of the floor at or under (x, y) (None: none)."""
        under = [q for q in self.model.platforms if q.covers(x) and q.y_at(x) >= y - 1]
        return min(under, key=lambda q: q.y_at(x)).group if under else None


def end_link(p: Platform, way: int) -> str:
    """What joins platform p's end that way (+1 right, -1 left): "free" (nothing), "rising" (a wall up from it),
    "step" (a wall down from it), "on" (a foothold on across)."""
    if way > 0:
        link = max(p.segments, key=lambda s: max(s.x1, s.x2)).next
    else:
        link = min(p.segments, key=lambda s: min(s.x1, s.x2)).prev
    if link is None:
        return "free"
    x1, y1, x2, y2 = link
    if x2 > x1:
        return "on"
    return "rising" if (y2 < y1 if way > 0 else y2 > y1) else "step"


@dataclass
class SimJumper(SimMonster):
    """A monster that jumps (JUMPERS): moved step by step of its AI (AI_STEP) and event by event between them (its
    walk reaching its platform's end or its reach, a jump asked for there, its landing, its hit's end), as the
    stretches it goes through (`segs`: ("stand", t0, x, platform), ("walk", t0, x, way, platform), ("press", t0, x,
    way, platform): walking into its platform's end, ("hit", t0, from x, to x, platform), ("air", t0, flight, way)),
    kept JUMPER_KEEP_S behind the latest. `platform` is the one it stands on, or jumped from in the air."""
    space: JumpSpace | None = None
    target: Callable[["SimJumper", float], tuple[float, float] | None] | None = None
    home_platform: Platform | None = None
    segs: list | None = None
    ai_t: float = 0.0          # its AI's next step
    thinktimer: int = 0
    movedir: int = 0
    jumpdir: int = 0
    checkjump: bool = False
    jumpy: float = 0.0
    jump_at: float | None = None  # a jump asked for at its platform's end
    hit_until: float | None = None
    last: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (t, x, y) of its last place worked out

    def start(self, t: float, x: float) -> None:
        if self.home_platform is None:
            self.home_platform = self.platform
        self.platform = p = self.home_platform
        x = p.clamp(x)
        self.segs, self.ai_t = [("stand", t, x, p)], t + AI_STEP
        self.thinktimer = self.movedir = self.jumpdir = 0
        self.checkjump, self.jump_at, self.hit_until, self.pushed = False, None, None, None
        self.t0, self.x0, self.d, self.anim0 = t, x, 0, t
        self.last = (t, x, p.y_at(x))

    def ghost(self) -> "SimJumper":
        g = super().ghost()
        g.segs = list(self.segs)
        return g

    # --- where it is ---------------------------------------------------------------------------------------------

    def at(self, t: float) -> float:
        self._run(t)
        x, y, _, _ = self._place(t)
        self.last = (t, x, y)
        return x

    def y(self, x: float) -> float:
        _, lx, ly = self.last
        return ly if abs(x - lx) < 1e-6 else self.platform.y_at(self.platform.clamp(x))

    def in_air(self, t: float) -> bool:
        self.at(t)
        return self._seg(t)[0] == "air"

    def faces(self, t: float) -> tuple[int, bool]:
        self.at(t)
        s = self._seg(t)
        if s[0] in ("walk", "press"):
            return s[3] or self.facing, bool(s[3])
        if s[0] == "air":
            return s[3] or self.facing, True
        return self.facing, False

    def body(self, t: float, flinching: bool = False) -> tuple[float, float, float, float]:
        way, walking = self.faces(t)
        if self.info is None or not self.info.move:
            return -self.half, self.half, 0.0, max(self.height, self.flinch_top) if flinching else self.height
        if flinching and self.info.hit is not None:
            f = self.info.hit
        else:
            f = MobInfo.frame_at(self.info.move if walking else self.info.stand, (t - self.anim0) * 1000)
        return (f.lo, f.hi, f.bottom, f.top) if way < 0 else (-f.hi, -f.lo, f.bottom, f.top)

    def _seg(self, t: float):
        for s in reversed(self.segs):
            if s[1] <= t:
                return s
        return self.segs[0]

    def _place(self, t: float) -> tuple[float, float, int, Platform | None]:
        """(x, y, the way it holds: its AI's input, the platform under it: None in the air) at t."""
        s = self._seg(t)
        kind, t0 = s[0], s[1]
        if kind == "stand":
            return s[2], s[3].y_at(s[2]), 0, s[3]
        if kind in ("walk", "press"):
            x = s[2] + s[3] * self.speed * max(0.0, t - t0) if kind == "walk" else s[2]
            return x, s[4].y_at(s[4].clamp(x)), s[3], s[4]
        if kind == "hit":
            x = pushed_x(s[2], s[3], max(0.0, t - t0))
            return x, s[4].y_at(s[4].clamp(x)), 0, s[4]
        path = s[2].path
        u = max(0.0, t - t0) / body.TICK
        i = int(u)
        if i >= len(path) - 1:
            x, y = path[-1]
        else:
            k = u - i
            x = path[i][0] + (path[i + 1][0] - path[i][0]) * k
            y = path[i][1] + (path[i + 1][1] - path[i][1]) * k
        return x, y, s[3], None

    # --- its events ----------------------------------------------------------------------------------------------

    def _run(self, t: float) -> None:
        while True:
            s = self.segs[-1]
            if self.hit_until is not None:
                due = [(self.hit_until, "hit end")]
            else:
                due = [(self.ai_t, "ai")]
                if s[0] == "walk":
                    due.append(self._walk_end(s))
                elif s[0] == "air":
                    due.append((s[1] + s[2].seconds, "land"))
                if self.jump_at is not None:
                    due.append((self.jump_at, "jump"))
            te, what = min(due)
            if te > t:
                break
            if what == "ai":
                self.ai_t = te + AI_STEP
                if s[0] == "air":
                    self.thinktimer -= 1
                else:
                    self._think(te)
            elif what == "hit end":
                self.hit_until = None
                x, _, _, p = self._place(te)
                self._add(("stand", te, x, p))
            elif what == "land":
                self._land(te)
            elif what == "jump":
                self._jump(te, self._place(te)[2], check=False)
            else:
                self._at_end(te, what)
        if len(self.segs) > 1 and self.segs[1][1] < t - JUMPER_KEEP_S:
            keep = [s for s in self.segs if s[1] >= t - JUMPER_KEEP_S]
            self.segs = [self._seg(t - JUMPER_KEEP_S), *keep] if keep else self.segs[-1:]

    def _add(self, s) -> None:
        moving = s[0] in ("walk", "press", "air") and s[3] != 0 or s[0] == "air"
        before = self.segs[-1]
        if s[0] == "hit" or moving != (before[0] in ("walk", "press", "air") and before[3] != 0 or before[0] == "air"):
            self.anim0 = s[1]
        self.segs.append(s)
        if s[0] != "air":
            self.platform = s[-1]
        self.d = s[3] if s[0] in ("walk", "press", "air") else 0
        if self.d:
            self.facing = self.d

    def _limits(self) -> tuple[float, float]:
        """Its reach across: its platform's group's and its home group's, less JUMPER_MARGIN a side where wider than
        twice that (FootholdTree::limit_jumper)."""
        here = self.space.reach.get(self.platform.group, (self.platform.left, self.platform.right))
        home = self.space.reach.get(self.home_platform.group, here)
        lo, hi = min(here[0], home[0]), max(here[1], home[1])
        return (lo + JUMPER_MARGIN, hi - JUMPER_MARGIN) if hi - lo > 2 * JUMPER_MARGIN else (lo, hi)

    def _walk_end(self, s) -> tuple[float, str]:
        """When a walk reaches its platform's end ("end") or its reach ("turn")."""
        _, t0, x0, way, p = s
        lo, hi = self._limits()
        if way > 0:
            stop, what = (p.right, "end") if p.right <= hi else (hi, "turn")
        else:
            stop, what = (p.left, "end") if p.left >= lo else (lo, "turn")
        return t0 + max(0.0, (stop - x0) * way) / self.speed, what

    def _chasing(self, t: float) -> tuple[float, float] | None:
        """Her place (x, her feet's y) while it chases her (Realism.chase), else None."""
        if self.target is None or self.aggro_until is None or t >= self.aggro_until or self.hit_until is not None:
            return None
        return self.target(self, t)

    def _walk(self, t: float, way: int) -> None:
        """Mob::walk: stand (0) or walk that way; pressed to its platform's end, it holds that way (the jump asked
        for there goes that way)."""
        x, _, now, p = self._place(t)
        s = self.segs[-1]
        if s[0] == "press":
            if way != now:
                self._add(("press", t, x, way, p) if way else ("stand", t, x, p))
            return
        if way != now or s[0] not in ("walk", "stand"):
            self._add(("walk", t, x, way, p) if way else ("stand", t, x, p))

    def _jump(self, t: float, way: int, check: bool) -> None:
        x, y, _, _ = self._place(t)
        s = self.segs[-1]
        shoe = self.space.shoe(self.speed)
        v = s[3] * self.speed if s[0] == "walk" else 0.0
        self.jumpdir, self.checkjump, self.jumpy, self.jump_at = way, check, y, None
        self._fly(t, x, y, body.takeoff(v, way, shoe) if way else v, -shoe.vjump, way)

    def _fly(self, t: float, x: float, y: float, vx: float, vy: float, way: int) -> None:
        f = body.fly(self.space.model, x, y, vx, vy, shoe=self.space.shoe(self.speed), arrow=way, walls=self.space.walls,
                     sides=self.space.sides, path=True)
        self._add(("air", t, f, way))

    def _land(self, t: float) -> None:
        f = self.segs[-1][2]
        if f.platform is None:  # (nothing to land on within the flight's time: back where it came from)
            self.start(t, self.home)
            return
        way = self.jumpdir
        self._add(("walk", t, f.land_x, way, f.platform) if way else ("stand", t, f.land_x, f.platform))

    def _at_end(self, t: float, what: str) -> None:
        """Its walk at its reach (round) or its platform's end (on, down a step, off walking, or a jump asked for)."""
        _, _, _, way, p = self.segs[-1]
        x, y, _, _ = self._place(t)
        if what == "turn":
            self._add(("walk", t, x, -way, p))
            return
        link = end_link(p, way)
        her = self._chasing(t)
        lo, hi = self._limits()
        inreach = lo < p.left if way < 0 else p.right < hi
        stepdown = her is None or self.space.group_below(*her) == p.group
        edgejump = her is None or her[1] <= y or self.chase_rng.random() < 0.5
        if link == "on" or link == "step" and inreach and stepdown or link in ("free", "step") and not edgejump:
            self.jumpdir, self.checkjump = way, False
            self._fly(t, x, y, way * self.speed, 0.0, way)  # (on, or off it walking)
            return
        self._add(("press", t, x, way, p))
        self.jump_at = t + EDGE_JUMP_S

    def _think(self, t: float) -> None:
        """One AI step on the ground: think_jump, then think (Mob::think)."""
        x, y, way, p = self._place(t)
        her = self._chasing(t)
        if self.checkjump:
            self.checkjump = False
            if way and abs(y - self.jumpy) < 0.5:
                way = -way
                self._walk(t, way)
        end = p.right if way > 0 else p.left
        if way and end_link(p, way) == "rising" and (end - x) * way <= JUMP_AHEAD:
            if (end - x) * way < 1:
                way = -way
                self._walk(t, way)
            elif self.thinktimer > 0:
                self._jump(t, way, check=True)
                self.thinktimer -= 1
                return
        if her is not None and self.thinktimer > 0:
            g = self.space.group_below(*her)
            reach = self.space.reach.get(g)
            if (g is not None and g != p.group and her[1] < y and reach is not None and reach[0] < x < reach[1]
                    and self.chase_rng.randrange(JUMP_CHASE) == 0):
                self.movedir, self.thinktimer = 0, -1
                self._jump(t, way, check=True)
                return
        if self.thinktimer > 0:
            self.thinktimer -= 1
            return
        if her is not None:
            self.thinktimer = think_steps(self.chase_rng, CHASE_THINK_MS)
            if self.movedir == 0 or abs(her[0] - x) >= AGGRO_PASS:
                self.movedir = 1 if her[0] >= x else -1
            self._walk(t, self.movedir)
        elif self.rng.randrange(JUMP_IDLE) == 0:
            self._jump(t, way, check=False)
            return
        else:
            self.thinktimer = think_steps(self.rng, IDLE_THINK_MS)
            self.movedir = self.rng.randrange(3) - 1
            self._walk(t, self.movedir)
        self.thinktimer -= 1

    # --- hit ---------------------------------------------------------------------------------------------------

    def knock(self, t: float, away: int) -> None:
        self.at(t)
        x, _, _, p = self._place(t)
        if p is None:  # (in the air: stood where it lands)
            f = self.segs[-1][2]
            if f.platform is None:
                self.start(t, self.home)
                x, p = self.home, self.platform
            else:
                x, p = f.land_x, f.platform
        lo, hi = self._limits()
        to = min(max(x + away * KNOCK_MOB, p.left, lo), p.right, hi)
        self.jump_at, self.checkjump = None, False
        self.ai_t = t + monster_data.FLINCH_S + max(0.0, self.ai_t - t)  # (its AI waits the hit out)
        self.hit_until = t + monster_data.FLINCH_S
        self._add(("hit", t, x, to, p))
        self.pushed, self.facing = (t, x), -away

    def aggro(self, t: float) -> None:
        if self.chase is None:
            return
        self.at(t)
        self.aggro_until = t + self.chase_rng.uniform(*AGGRO_S)
        self.thinktimer, self.movedir = 0, 0  # (Mob::chase: it decides at its AI's next step)


@dataclass
class Drop:
    kind: str           # "mesos" or an item kind
    x: float
    y: float
    platform: int
    since: float        # when it lands (shows) on the ground
    gone: bool = False
    owner: str = "her"  # "her", or "rival": another player's kill (Realism.rival_every)
    taken_at: float | None = None  # when it was picked up ...
    taken_by: str = ""             # ... and by whom ("her", "rival", or "other": Realism.drop_takers)
    amount: int = 0     # mesos: how many (Realism.drop_mix); 0: not drawn (the coin's worth: behavior.looting.Worth)
    life: float = DROP_LIFE  # s it lies unless taken: DROP_LIFE, or to the expiry pass past it (Realism.drop_takers)
    pet: bool = False   # taken by a pet of hers (Realism.pets: taken_by "her", as her own)

    @property
    def mesos(self) -> bool:
        return self.kind == "mesos"


@dataclass
class SimPet:
    """A pet of hers (Realism.pets, PET_SPEED ...): the platform it's on and its x there, since when she's been on
    another (None: she isn't), and the place it wanders to and when it picks another."""
    platform: int | None
    x: float
    off: float | None = None
    goal: float = 0.0
    next: float = 0.0


@dataclass
class Rival:
    """Another player (Realism.rival_every): where it hunts, or the walk it's on to another place."""
    x: float
    y: float
    platform: int | None          # where it hunts (None: walking to another place)
    goal: tuple | None = None     # (platform, x, y) it walks to ...
    start: tuple = (0.0, 0.0)     # ... from (x, y) ...
    leaves: float = 0.0           # ... setting off then ...
    arrives: float = 0.0          # ... and getting there then (world s)


# --- the world ---------------------------------------------------------------------------------------

@dataclass
class Event:
    ms: int
    seq: int
    kind: str           # 'down' / 'up' (keys), 'release' (all keys), or an internal one
    data: object = None
    tag: str = ""       # the plan a key event belongs to
    at: float = 0.0     # clock time it was sent for (before rounding up to a tick)


class World:
    def __init__(self, model: MapModel, keys: GameKeys, start: tuple[int, float], seed: int = 1,
                 monsters="all", hp: float = 1.0, inventory_full: bool = False, facing: str = RIGHT,
                 origin: float = 0.0, drops: list[tuple[str, int, float]] = (), hanging: float | None = None,
                 noise: Noise | None = None, realism: Realism | None = None, step_phase: int | None = None,
                 character: Her = LEVEL_19):
        """Her standing on platform start[0] at x start[1] (or, with `hanging`, on the rope nearest that x,
        her feet at y `hanging`). `monsters`: None, "all" (the map's spawns), a set of platform ids (the
        spawns on those), or [(name, platform, x)]. `drops`: [(kind, platform, x)] lying there already.
        `step_phase`: the ms (a whole tick, below STEP_MS) of the game's first step; by default drawn from
        the seed (from an rng of its own: the world's other draws don't move). `character`: her HP, level and pet
        (Her)."""
        self.model = model
        self.character = character
        # Her walk as the world moves her (Her.walk_speed: her_shoe), apart from the planner's physics.SHOE, and her jump
        # off a rope sideways with it; her climb (Her.climb_speed: her_climb)
        self.shoe = her_shoe(character)
        self.rope_jump = rope_jump(self.shoe)
        self.climb = her_climb(character)
        self.origin = origin       # clock time of ms 0
        self.ms = 0
        self.seed = seed
        if step_phase is None:
            step_phase = random.Random(f"step:{seed}").randrange(0, STEP_MS, TICK_MS)
        self.step_phase = step_phase
        self.rng = random.Random(seed)
        self.noise = noise or Noise()
        self.realism = realism or Realism()
        # The knobs draw from rngs of their own, so with them off the world is the same as without them.
        self._rival_rng = random.Random(f"rival:{seed}")
        self._drop_rng = random.Random(f"drops:{seed}")  # Realism.drops' draws (the others' unchanged by them)
        self._camp_rng = random.Random(f"camps:{seed}")  # Realism.camps' draws
        self._tick_rng = random.Random(f"ticks:{seed}")  # Realism.respawn_tick's draws
        self._mix_rng = random.Random(f"mix:{seed}")      # Realism.drop_mix's draws
        self._taker_rng = random.Random(f"takers:{seed}")  # Realism.drop_takers' draws
        self._pet_rng = random.Random(f"pets:{seed}")      # Realism.pets' draws
        self._expiry_at = random.Random(f"expiry:{seed}").uniform(0.0, EXPIRE_PASS_S)  # ... and its expiry passes'
        # What a kill drops (Realism.drops; with MAP_DROPS, the map's)
        self._drop_table = game_drops(model) if self.realism.drops == MAP_DROPS else tuple(self.realism.drops)
        self.potions_left = self.realism.potions
        self.darts_left = self.realism.darts  # her throwing stars (Realism.darts; None: endless) ...
        self.no_darts = 0                     # ... and her attacks keyed with too few of them
        self.darts_out_at: float | None = None  # ... the first of those (world s)
        self.potions_used = 0           # potions drunk, by her key or by her pet
        self.pet_potions = 0            # ... of them her pet's (Her.pet)
        self.casts = 0                  # Lucky Seven casts (Her.mp kept), the MP they took, the MP potions her pet
        self.mp_used = 0.0              # drank for her (Her.pet_mp), and the casts pressed with too little MP (none
        self.mp_potions = 0             # thrown)
        self.no_mp = 0
        self._mp_unshown: list[tuple[int, float]] = []  # (the tick her bar shows it at, its MP share): MP_SHOWS_MS
        self.casts_nothing = 0          # ... and of her casts, those whose stars hit nothing (none picked, a dead one,
                                        # or stopped by terrain): MP thrown at nothing
        self.attacks = 0                # attacks her keys made, either key
        self.attack_locked = 0          # attack keys pressed inside the last attack's lock (ignored by the game)
        self.air_presses = 0            # (Her.magic) her attack keys pressed in the air: nothing (cast from the ground
        self.swings = 0                 # only), and her regular attack's swings (her wand: no damage kept)
        # Every attack the game took, as it came out (_attack): (when its key went out, s of the run; the name of the
        # key plan that pressed it, as the stack's player was sent it; "ground" or "air" when it started; how far she
        # was above the platform she last stood on then; the kind of the monster it picked, or None; what came of it:
        # "kill", "hit", "swipe" (the claw's, not killing), "miss" (Noise.miss), "stopped" (its line: sight_stop) or
        # "nothing" (no monster picked, or one dead by then); how far above her feet the pick's were, + above). What
        # each kind of throw hit (sim.harness: `attack_kinds`).
        self.attack_log: list[tuple] = []
        self.on_attack = None  # called with each attack_log row as it's logged (sim.scenarios.Sim: 'sim_attack')
        # Her stars' MISS popups (as the game draws one over a monster her star missed, where it was as the star landed:
        # FakeEyes.misses): (s it shows from, map x, the monster's feet's y, its height), Noise.miss_unseen's left out
        self.misses: list[tuple[float, float, float, float]] = []
        self._key_tag = ""              # the plan of the key being taken (_handle), for attack_log
        self.pet_drank_ms = -10_000     # when her pet last drank one
        self.pet_fed = 0                # times she fed her pet (the pet food key)
        self.pet_full = PET_FULLNESS     # her pet's fullness (Her.pet) as of pet_since (pet_fullness)
        self.pet_since = 0.0
        self.hp_low = hp                # her lowest HP so far
        self.kill_log: list[tuple[float, str, int, str]] = []  # (t, kind, platform, "her" or "rival")
        self.death_log: list[tuple[float, str, int, float]] = []  # ... and where each died: (t, kind, platform, x)
        # Her stack's throws' targets as it chose them, by the world's monster (aimed_at: t, attack key, its index,
        # 'alive', 'her' or 'rival' for one dead by then, 'none'), and the feed's tracks her stack took for dead that
        # were of a monster alive then (live_taken: t, its index), from the Sim's hooks (sim.scenarios: Sim.hunter)
        self.aim_log: list[tuple[float, str, int | None, str]] = []
        self.live_taken: list[tuple[float, int]] = []
        self.touch_log: list[tuple[float, float]] = []  # (t, the share of her HP it took: 0 if evaded) for every touch
        self.fresh_log: list[tuple[float, float]] = []  # ... of those, the touches by a monster within FRESH_S of its
                                                        # respawn (from its spawn point)
        self.newborn_log: list[tuple[float, float]] = []  # ... and within NEWBORN_S of it (the spawn guard's newborns:
                                                          # behavior.hunt._respawn_guard)
        # ... and where each was: (t, her mode then: ground, air or rope, her x, her feet's y), one for each of
        # touch_log's; her climbs up a rope onto the platform at its top (climb_log: t, the rope's x, the platform);
        # and her grabs of a rope (grab_log: t, the rope's x, her feet's height under its top then). The live hour of
        # 2026-11-02 had 19-23 of its 170 touches at the top of the rope at x -80 over 68 climbs up it (sim.harness:
        # `rope_tops`).
        self.touch_place: list[tuple[float, str, float, float]] = []
        self.climb_log: list[tuple[float, float, int]] = []
        self.grab_log: list[tuple[float, float, float]] = []
        # Her knocks off a platform (a touch's flight landing her on another one more than KNOCK_FALL_DY lower than the
        # one she was knocked from, or out of the map): (t of the touch, from, onto (None: out of the map), how far
        # down). Live on 2026-10-25 three knocks off Tree Dungeon's 23 dropped her 389 units to 29, 43 s of the hunt
        # back up (sim.harness: `knock_falls`).
        self.knock_log: list[tuple[float, int, int | None, float]] = []
        self._knocked_from: tuple[float, int, float] | None = None  # the touch's (t, platform, her y) while it flies
        self.actions = _actions(keys, self.realism.game_keys)
        # her chair and SIT, which nothing presses in this world: where the window draws them (binding_view)
        self.unused = {k.name: what for what, k in (("chair", keys.chair), ("sit", keys.sit)) if k is not None
                       and k.name not in self.actions}
        self.actions.setdefault("\\", "set_key")  # (the KEY BINDINGS window's keys: _key_window)
        self.actions.setdefault("esc", "esc")
        if self.realism.stat_key is not None:
            self.actions.setdefault(self.realism.stat_key, "stat")  # (her stat window's key: _stat_window)
        if self.realism.chat is not None:
            self.actions.setdefault("'", "chat")  # (her chat key: _chat_box)
        self.chat_box = self.realism.chat  # her chat box as the game shows it (Realism.chat), her chat key's presses
        self.chat_presses = 0
        self.stat_window = False  # her stat window open (Realism.stat_key) ...
        self.stat_detail = False  # ... its DETAIL panel with it ...
        self.stat_windows = 0     # ... how often it was opened, and the clicks on its DETAIL button
        self.stat_clicks = 0
        self.key_window = False   # the KEY BINDINGS window open (Realism.game_keys) ...
        self.key_windows = 0      # ... how often it was opened, the keys pressed while it was (doing nothing) and Esc
        self.keys_in_window = 0   # pressed with it closed (the game's MENU opens)
        self.menu_opened = 0
        self.held: dict[str, float] = {}  # action -> the clock time it went down (as sent)
        self._events: list[tuple[int, int, Event]] = []
        self._seq = 0
        self.log: list[str] = []        # notable world events, for the trace
        # her
        p = model.platform(start[0])
        self.mode = "ground"            # ground | air | rope
        self.plat: Platform | None = p
        self.last_plat: Platform = p    # the platform she last stood on: its foothold group's walls stop stars
        self.rope: Rope | None = None
        self.flight: body.Flight | None = None  # in the air: the arc she's on (body.fly, with its path) ...
        self.flight_ms = 0              # ... from this tick (its first point) ...
        self.flight_arrow = 0           # ... flown with this arrow held ...
        self.flight_skip: Platform | None = None  # ... through this platform (dropping)
        self.group = p.group            # her foothold group: the walls of it stop her in the air (body.fly)
        self.knocked = False            # in the air from a monster's touch (KNOCK_STEERS)
        self.fell_out_at: float | None = None  # when she fell below the map's lowest floor (SimFellOut)
        self.left_map_at: float | None = None  # when she went through a portal (SimLeftMap)
        self.bottom = max(q.bottom for q in model.platforms)  # its lowest floor (until 2026-11-01 its ends')
        self.x, self.y = float(start[1]), p.y_at(start[1])
        self.vx = self.vy = 0.0         # her speed (units/s; y down)
        self.glide = 0.0                # on the ground: how far she moves each tick of the game's step (none from a
                                        # landing until the next step: body.LANDING_ENDS_STEP)
        self.step_arrow = 0             # the arrow the game moved her with at its last step on the ground
        self.facing = facing
        self._air_turn: tuple[int, str] | None = None  # the step an arrow last turned her in the air at, her way before
        self.air_turn_ms = -10_000      # the tick an arrow toward her back was last acted on in the air (turned or
                                        # not) ...
        self.air_turns_lost = 0         # ... and those that didn't turn her (Noise.air_turn_lost)
        self.lift_ms = -10_000          # the tick she last lifted off (a jump, off a rope)
        self.swiped_casts = 0           # Lucky Seven keys the claw's swipe took (World._uncast: no cast)
        # Up close (the clearance harness counts, sim.harness: `claw_s`, `close_s`): ticks she stood or flew with a
        # live monster on the platform she last stood on within the claw's reach of her (reach.CLAW_FAR plus half its
        # width, middle to middle: a mushroom 93), and within CLOSE_DX
        self.claw_ms = 0
        self.close_ms = 0
        # Every attack's pick (_aim), with what came of it (_attack): (key s, plan, "ground" or "air", turned in the air
        # in this flight before it, the attack key's ms after her lift-off, the nearest live monster on the platform she
        # last stood on ahead of her then the way the stack threw it (`meant`; her facing where no throw listener
        # tells): how far ahead (middle to middle) and how much lower its feet were than the floor under her (+ lower:
        # the slope) or None, whether it was the pick, whether it was a Lucky Seven cast (not one the claw's swipe took:
        # _uncast), the outcome as attack_log's)
        self.pick_log: list[tuple] = []
        self.seen: dict[str, int] = {}  # the arrows the game sees held -> the order it saw them pressed in
        self._seen_count = 0
        self._seen_due: dict[str, int] = {}  # each arrow's latest change the game is yet to see (its tick)
        self.still_ms = 0               # she can't move before this tick (climbing over a rope's top)
        self.left_rope_ms = -10_000
        self.attack_ms = (-10_000, -10_000)  # the last attack's animation: from its start to its lock's end
        self.hp = hp
        self.mp = 1.0
        self.exp = 0
        self.attack_lock_ms = -10_000
        self.attack_key_at = -10_000.0  # when the key of the last attack the game took went out (clock time)
        self.invulnerable_ms = -10_000
        # The game data's level of each kind on the map (MapModel.mob_level, by kind): her hit check's (evade_chance)
        self._mob_levels = {kind_of(name): level for name, level in getattr(model, "mob_level", {}).items()}
        # ... and each kind's defence against magic (MapModel.mob_mdd, MDDamage): her Magic Claw's (Her.magic)
        self._mob_mdd = {kind_of(name): mdd for name, mdd in getattr(model, "mob_mdd", {}).items()}
        self.inventory_full = inventory_full
        # Her inventory's tabs (Realism.full_tabs, fills_after; `inventory_full`: every item tab): those full, when each
        # filled (s), and the items picked into each
        self.full_at: dict[str, float] = {tab: 0.0 for tab in (("Use", "Etc") if inventory_full else ())
                                          + tuple(self.realism.full_tabs)}
        self._fills = dict(self.realism.fills_after)
        self.tab_items: dict[str, int] = {}
        self.tab_kinds: dict[str, dict[str, float]] = {}  # ... and their kinds, when each first (her stacks: a full tab
                                                          # takes more of those)
        self.room_t: float | None = None  # when her full tabs had room again (Realism.room_at)
        self.pick_refused = 0  # pick-ups refused for want of room
        self.kills: dict[str, int] = {}
        self.touches = 0                # monsters touching her (evaded ones too)
        self.evaded = 0
        # The ways the stack means its attacks to go, (when its key goes out, way), from the Sim's throw listener
        # (control.actions: before the plan plays, so its key goes then or later), and of the attacks the game took
        # that one was meant for, how many went the other way (her facing as each started: _aim)
        self.meant: list[tuple[float, str]] = []
        self.attacks_meant = 0
        self.wrong_way = 0
        self.picked: dict[str, int] = {}
        self.messages: list[tuple[float, str]] = []
        self.pets: list[SimPet] | None = None  # her pets (Realism.pets), out from their first step (_pets_step) ...
        self._pet_drops: list[Drop] = []       # ... the drops lying as they last looked (PET_SCAN_S), and when
        self._pet_scan_t = -1.0
        self.path_log: list[tuple[float, int | None, float]] = []  # (s, the platform she stood on or None, x), every
                                                                   # PATH_EVERY_MS
        self.chat: tuple[float, str] | None = None  # the message bar's game line, and since when (ROPE_REFUSAL)
        self.refused = 0  # her attacks refused on a rope
        self.portals = [pt for pt in model.portals if pt.kind == PORTAL_EXIT]
        self.sides = _sides(model, self.realism.sides)
        self.arms = self.realism.rope_reach - JUMP_HEIGHT  # she grabs a rope this far above her feet (ARMS)
        self.dead = False
        self.died_at: float | None = None
        zero = self.realism.zero_walls
        self._blocked = {pid: (_wall_at(model, q, q.left, -1, zero), _wall_at(model, q, q.right, 1, zero))
                         for pid, q in enumerate(model.platforms)}
        self._near = {q.id: [o.id for o in model.platforms if o.right >= q.left - 60 and o.left <= q.right + 60
                             and -70 <= o.top - q.top <= 70] for q in model.platforms}
        if hanging is not None:
            self._grab(min(model.ropes, key=lambda r: abs(r.x - start[1])))
            self.y = min(max(hanging, self.rope.top), self.rope.bottom)
            self.log.clear()
        # Her landings (_land), as the live diagnosis counted them: (t, the platform she took off from or None off a rope
        # or with no takeoff seen, its x, the platform she landed on, x, "knock", "fight" (an attack played in the
        # flight) or "trip", the nearest live monster on the platform she landed on, middle to middle, or None, its kind,
        # +1 walking at her, -1 away, 0 standing). sim.harness: `landings`.
        self.land_log: list[tuple] = []
        self._took_off: tuple[int | None, float, int] | None = None
        self._knock_landed_ms = -10 ** 9  # (her last knock's landing: _log_landing's knock-falls)
        self.monsters: list[SimMonster] = []
        self.by_platform: dict[int, list[SimMonster]] = {}  # (the walkers by their platform; the jumpers apart: _on)
        self.jumpers: list[SimJumper] = []
        self._jump_space: JumpSpace | None = None
        self._kinds: tuple | None = None  # the game data's bodies, speeds and mob ids (_monster)
        self._spawn(monsters)
        self.points = len(self.monsters)  # the map's spawn points (one monster each at the start)
        hurt = dict(self.realism.hurt)
        for m in self.monsters:  # (Realism.hurt: a share of its HP left, its bar drawn as any hurt one's: drawn_bar)
            if m.kind in hurt:
                m.hp = m.hp_max * hurt[m.kind]
        self.drops: list[Drop] = [Drop(kind, x, model.platform(pid).y_at(x), pid, 0.0) for kind, pid, x in drops]
        self.rival: Rival | None = None
        if self.realism.rival_every > 0 and self.monsters:
            self.rival = self._rival_start()
            self._push(self._rival_due(), "rival")
            self._push(self._rival_stays(), "rival_moves")
        if self.realism.respawn_tick > 0 and self.monsters:  # the map's pass, from a phase of its own; the map at its
            # cap from the start (the monsters over it there from the start, dead long since)
            self.respawn_ticks = self.respawn_births = self.respawn_blocked = 0
            over = len(self.monsters) - self._respawn_cap()
            for m in self._tick_rng.sample(self.monsters, max(0, over)):
                m.alive, m.died_at, m.died_x = False, -1e6, m.home
            for i in self.realism.vacant:  # (a scenario's: Realism.vacant, first_pass)
                m = self.monsters[i]
                m.alive, m.died_at, m.died_x = False, -1e6, m.home
            first = self._tick_rng.uniform(0.0, self.realism.respawn_tick)
            self._push(int((first if self.realism.first_pass is None else self.realism.first_pass) * 1000), "map_tick")
        for m in self.monsters:
            self._camps_watch(m)
        for at, side in self.noise.turned:
            self._push(int(at * 1000), "turned", side)

    # --- set-up -------------------------------------------------------------------------------------

    def _spawn(self, monsters) -> None:
        if monsters is None or monsters == []:
            return
        if monsters == "all" or isinstance(monsters, (set, frozenset)):
            wanted = []
            share = random.Random(f"spawn:{self.seed}")
            for s in self.model.spawns:
                p = self.model.platform_under(s.x, s.y, reach=60)
                if p is not None and (monsters == "all" or p.id in monsters):
                    if self.realism.spawn_share < 1.0 and share.random() >= self.realism.spawn_share:
                        continue
                    wanted.append((s.name or str(s.mob_id), p.id, s.x))
        else:
            wanted = list(monsters)
        for i, (name, pid, x, *first) in enumerate(wanted):
            m = self._monster(i, name, pid, float(x), random.Random(self.rng.random()))
            m.start(0.0, float(x))
            if first and not isinstance(m, SimJumper):  # (name, platform, x, s): it stands that long first, then its
                m.d, m.t1 = 0, float(first[0])           # own walk (a set-piece's look, as live's tracks had them)
            self.monsters.append(m)
            self._keep(m)

    def _monster(self, i: int, name: str, pid: int, x: float, rng: random.Random) -> SimMonster:
        """A monster of `name` at spawn x `x` on platform `pid`, its body, speed, HP and info the game data's (not
        started)."""
        if self._kinds is None:
            ids = {kind_of(s.name): s.mob_id for s in self.model.spawns if s.name}  # by kind ...
            ids.update({s.name: s.mob_id for s in self.model.spawns if s.name})      # ... and by name
            self._kinds = (monster_data.bodies(self.model), monster_data.speeds(self.model), ids)
        bodies, speeds, ids = self._kinds
        kind = kind_of(name)
        mob_id = ids.get(name, ids.get(kind))
        info = None if mob_id is None else mob_info(int(mob_id))
        half, height = _body_of(kind, bodies.get(kind), info)
        hits = float(hits_of(kind, info))
        if self.character.magic is not None and info is not None and info.hp > 0:  # (her Magic Claw: its damage
            hits = info.hp * (self._mob_mdd.get(kind, 0.0) + 100) / 100 / STAR_MEAN  # before MDDamage, in STAR_MEANs)
        elif self.character.spread and info is not None and info.hp > 0:  # in average stars, not whole ones
            hits = info.hp * (info.pdd + 100) / 100 / STAR_MEAN
        hops = self.model.mob_hops
        jumps = JUMPERS and bool(hops[name] if name in hops else any(v for k, v in hops.items() if kind_of(k) == kind))
        m = (SimJumper if jumps else SimMonster)(i, name, kind, self.model.platform(pid), x, rng,
                                                 speeds.get(kind, monster_data.DEFAULT_SPEED), half, height, hits, hits)
        if jumps:
            if self._jump_space is None:
                self._jump_space = JumpSpace.of(self.model, body.wall_table(self.model, self.realism.zero_walls),
                                                self.sides)
            m.space, m.target = self._jump_space, self._target_at
        if kind in bodies:
            m.flinch_top = bodies[kind].flinch_top
            m.fade = bodies[kind].die_s or FADE
        m.info = info
        if self.realism.chase:  # (chasing her once she hits it: Realism.chase)
            m.chase, m.chase_rng = self._chase_at, random.Random(f"chase:{self.seed}:{i}")
        return m

    # --- time -----------------------------------------------------------------------------------------

    @property
    def t(self) -> float:
        """World seconds since its start."""
        return self.ms / 1000

    def ms_of(self, clock_t: float) -> int:
        """The tick a clock time falls on (events take effect on the tick at or after them)."""
        ms = (clock_t - self.origin) * 1000
        return int(math.ceil(ms / TICK_MS - 1e-6)) * TICK_MS

    def step_at(self, ms: int) -> int:
        """The game's step at or after tick `ms` (when it acts on what it has seen by then)."""
        return ms + (self.step_phase - ms) % STEP_MS

    def _acts(self, at: float, lag: float) -> int:
        """The step at which the game acts on a key that went out at clock time `at`, `lag` s on average
        after it (a Lags value: it sees it Lags.seen(lag) after the key, and acts at its next step)."""
        return self.step_at(self.ms_of(at + body.Lags.seen(lag)))

    def advance(self, clock_t: float) -> None:
        """Run the world up to clock time `clock_t` (whole ticks at or before it). Raises SimDied once
        she's dead: whatever looks next finds the run over."""
        end = int(math.floor(((clock_t - self.origin) * 1000) / TICK_MS + 1e-6)) * TICK_MS
        while self.ms < end and not self.dead and self.fell_out_at is None and self.left_map_at is None:
            self._tick()
        if self.left_map_at is not None:
            raise SimLeftMap(f"she went through a portal at {self.left_map_at:.3f} s (x {self.x:.0f})")
        if self.fell_out_at is not None:
            raise SimFellOut(f"she fell out of the map at {self.fell_out_at:.3f} s (x {self.x:.0f})")
        if self.dead:
            raise SimDied(f"she died at {self.died_at:.3f} s ({self.touches} touches)")

    def chance(self, what: str, ms: int | None = None) -> float:
        """A noise draw in [0, 1) for `what` at tick `ms` (now by default): the same every run."""
        return random.Random(f"{what}:{self.seed}:{self.ms if ms is None else ms}").random()

    def _push(self, ms: int, kind: str, data=None, tag: str = "", at: float = 0.0) -> Event:
        self._seq += 1
        e = Event(max(ms, self.ms), self._seq, kind, data, tag, at)
        bisect.insort(self._events, (e.ms, e.seq, e), key=lambda v: (v[0], v[1]))
        return e

    def key(self, clock_t: float, name: str, down: bool, tag: str = "") -> None:
        """A key going down/up at clock time `clock_t` (in the future, or now)."""
        self._push(self.ms_of(clock_t), "down" if down else "up", name, tag, clock_t)

    def release_all(self, clock_t: float, tag: str = "") -> None:
        """The player lets go of every key at `clock_t` (a plan's end, or a cancel)."""
        self._push(self.ms_of(clock_t), "release", None, tag, clock_t)

    def call_at(self, clock_t: float, fn) -> None:
        """Call fn(world) when the world gets to clock time `clock_t` (at its tick, before it moves her): a test's look
        at the world at a given moment, whatever the stack is doing then."""
        self._push(self.ms_of(clock_t), "call", fn, "", clock_t)

    def withdraw_from(self, tag: str, clock_t: float) -> None:
        """Forget the key events of plan `tag` at clock time `clock_t` or later, and its releases whenever (a splice took
        over from it then: its end's letting go of every key doesn't come)."""
        self._events = [v for v in self._events if not (v[2].tag == tag and (
            v[2].kind == "release" or v[2].kind in ("down", "up", "click") and v[2].at >= clock_t - 1e-9))]

    def withdraw(self, tag: str) -> None:
        """Forget the key events of plan `tag` the world hasn't taken yet."""
        self._events = [v for v in self._events if not (v[2].tag == tag and v[2].kind in ("down", "up", "release",
                                                                                        "click"))]

    # --- one tick ---------------------------------------------------------------------------------------

    def _tick(self) -> None:
        while self._events and self._events[0][0] <= self.ms:
            _, _, e = self._events.pop(0)
            self._handle(e)
        dt = TICK_MS / 1000
        if self.mode == "ground":
            self._ground(dt)
        elif self.mode == "air":
            self._air()
        else:
            self._climb(dt)
        if "up" in self.held:
            self._through_portal()
        self._touch()
        self._closeness()
        if self.character.mp > 0 and self.mp < 1.0:
            self.mp = min(1.0, self.mp + self.character.mp_regen * dt / self.character.mp)
        self.ms += TICK_MS
        if self.ms % PATH_EVERY_MS == 0:
            self.path_log.append((self.t, self.plat.id if self.mode == "ground" and self.plat is not None else None,
                                  self.x))
        if self.realism.pets and self.ms % PET_STEP_MS == 0:
            self._pets_step(PET_STEP_MS / 1000)

    def _through_portal(self) -> bool:
        """Up held: in an exit's box (mapdata.PORTAL_BOX) while she isn't on a rope or attacking (or sitting: the
        simulator has no chair), she goes through it and leaves the map (SimLeftMap), on the ground or in the air
        (HeavenClient's Stage::update)."""
        if self.left_map_at is not None or self.mode == "rope" or self.ms < self.attack_lock_ms:
            return False
        for pt in self.portals:
            if abs(pt.x - self.x) <= PORTAL_BOX[0] and pt.y - PORTAL_BOX[1] <= self.y <= pt.y + PORTAL_BOX[2]:
                where = "on the ground" if self.mode == "ground" else "in the air"
                self.note(f"up at portal {pt.name} (to map {pt.to_map}) {where}: she leaves the map")
                self.left_map_at = self.t
                self.held.clear()
                self.seen.clear()
                return True
        return False

    def _handle(self, e: Event) -> None:
        if e.kind in ("down", "up"):
            action = self.actions.get(e.data)
            if action is None or self.dead:
                return
            if e.kind == "down" and action in ("set_key", "esc"):
                self._key_window(action)
                return
            if e.kind == "down" and self.key_window and action not in self.held:
                self.keys_in_window += 1  # (the window open: the key does nothing)
                self.note(f"{e.data} pressed with the KEY BINDINGS window open: nothing")
                return
            if action == "stat":
                if e.kind == "down":
                    self._stat_window()
                return
            if action == "chat":
                if e.kind == "down":
                    self._chat_box()
                return
            if e.kind == "down":
                if action not in self.held:
                    self.held[action] = self._sent(e)
                    self._key_tag = e.tag
                    self._pressed(action, self.held[action])
            elif (since := self.held.pop(action, None)) is not None:
                self._released(action, self._sent(e), since)
        elif e.kind == "release":
            for action in list(self.held):
                self._released(action, self._sent(e), self.held.pop(action))
        elif e.kind == "seen":
            self._seen(*e.data)
        elif e.kind == "unlock":
            action, since = e.data
            if self.held.get(action) == since:  # the press swallowed by her attack, still held: seen now
                self.note(f"{action} still held: seen now")
                self._seen_due[action] = max(self.ms, self._seen_due.get(action, -10_000))
                self._seen(action, True)
        elif e.kind == "call":
            e.data(self)
        elif e.kind == "click":
            self._click(*e.data)
        elif e.kind == "turned":
            self.turn(e.data)
        elif e.kind == "liftoff":
            self._liftoff(e.data)
        elif e.kind == "drop":
            if self.mode == "ground" and self.plat is not None and self.plat.id == e.data \
                    and self.plat.drop_ok(self.x):
                d = self.arrow()
                f = self._flight(self.x, self.y + 1, self.vx, 0.0, d, skip=self.plat)
                if f.platform is not None:
                    self._fly(f, self.ms, d, skip=self.plat)
                    self.note(f"drops through platform {e.data}")
        elif e.kind == "aim":
            self._aim(*e.data)
        elif e.kind == "star":
            self._attack(*e.data)
        elif e.kind == "struck":
            self._struck(*e.data)
        elif e.kind == "loot":
            self.drops += e.data
        elif e.kind == "respawn":
            self._respawn(e.data)
        elif e.kind == "map_tick":
            self._map_tick()
            self._push(self.ms + int(self.realism.respawn_tick * 1000), "map_tick")
        elif e.kind == "rival":
            self._push(self._rival_kill(), "rival")
        elif e.kind == "rival_moves":
            if self.rival.goal is None:
                self._rival_go()
            self._push(self._rival_stays(), "rival_moves")
        elif e.kind == "rival_arrives":
            r = self.rival
            if r.goal is not None and self.ms >= self._rival_arrival():
                r.platform, r.x, r.y = r.goal
                r.goal = None
                self.note(f"another player gets to platform {r.platform} at x {r.x:.0f}")
        elif e.kind == "camp_kill":
            m, life = e.data
            if m.alive and m.life == life:
                self._others_kill(m, self._camp_rng, CAMP_TAKES)
        elif e.kind == "rival_takes":
            for d in e.data:
                if not d.gone:
                    d.gone, d.taken_at, d.taken_by = True, self.t, "rival"
        elif e.kind == "others_take":  # (Realism.drop_takers: her drop's life ends, taken by someone else)
            for d in e.data:
                if not d.gone:
                    d.gone, d.taken_at, d.taken_by = True, self.t, "other"

    def _key_window(self, action: str) -> None:
        """The backslash (`set_key`) or Esc pressed: the KEY BINDINGS window opened or closed (Esc with it closed: the
        game's MENU)."""
        if action == "set_key":
            self.key_window = not self.key_window
            self.key_windows += self.key_window
            self.note("the KEY BINDINGS window " + ("opens" if self.key_window else "closes (the backslash)"))
        elif self.key_window:
            self.key_window = False
            self.note("the KEY BINDINGS window closes (Esc), nothing changed")
        elif self.stat_window:
            if self.realism.stat_stuck:
                self.note("Esc: her stat window stays open (stuck)")
            else:
                self.stat_window = self.stat_detail = False
                self.note("her stat window closes (Esc)")
        else:
            self.menu_opened += 1
            self.note("Esc with no window open: the game's MENU opens")

    def _stat_window(self) -> None:
        """Her stats key pressed: her stat window opened (its DETAIL panel with it as Realism.stat_detail has it) or
        closed (not with Realism.stat_stuck)."""
        if self.stat_window:
            if self.realism.stat_stuck:
                self.note("her stats key: her stat window stays open (stuck)")
                return
            self.stat_window = self.stat_detail = False
            self.note("her stat window closes (her stats key)")
            return
        self.stat_window, self.stat_detail = True, self.realism.stat_detail
        self.stat_windows += 1
        self.note("her stat window opens" + (" with its DETAIL panel" if self.stat_detail else ""))

    def _chat_box(self) -> None:
        """Her chat key pressed: her chat box expanded or collapsed, the other way (not with Realism.chat_stuck)."""
        self.chat_presses += 1
        if self.realism.chat_stuck:
            self.note(f"her chat key: her chat box stays {self.chat_box} (stuck)")
            return
        self.chat_box = "collapsed" if self.chat_box == "expanded" else "expanded"
        self.note(f"her chat box {self.chat_box} (her chat key)")

    def click(self, clock_t: float, x: int, y: int, tag: str = "") -> None:
        """A left click at client px (x, y) at clock time `clock_t` (sim.fakes.FakePlayer: the pointer where the
        plan's last mouse move put it)."""
        self._push(self.ms_of(clock_t), "click", (x, y), tag, clock_t)

    def _click(self, x: int, y: int) -> None:
        """A left click: on her stat window's DETAIL button (where perception.stat_window.compose draws it) the DETAIL
        panel opens or closes; anywhere else nothing (counted with it)."""
        from ..perception import stat_window as sw
        self.stat_clicks += 1
        mx, my = sw.DRAW_AT["main"]
        bx, by, bw, bh = sw.DETAIL_BUTTON
        if self.stat_window and mx + bx <= x < mx + bx + bw and my + by <= y < my + by + bh:
            self.stat_detail = not self.stat_detail
            self.note(f"a click on DETAIL: its panel {'opens' if self.stat_detail else 'closes'}")
        else:
            self.note(f"a click at ({x}, {y}): nothing")

    def binding_view(self) -> dict[str, int | str]:
        """The keys' functions as the KEY BINDINGS window draws them (perception.keybindings: key name -> its function
        picture's id, her Lucky Seven's and pet food's pictures by name, kb.PICTURES, or PICTURE for another item's),
        from what each key does in this world."""
        from ..perception import keybindings as kb
        icons = {"attack": 52, "jump": 53, "pick": 50, "left": 300, "right": 301, "up": 302, "down": 303,
                 "set_key": 9, "esc": 46, "stat": 2, "skill": "lucky_seven", "pet_food": "pet_food", "potion": kb.PICTURE,
                 "chair": "chair", "sit": 51}
        shown = {key: icons[what] for key, what in self.unused.items()}
        return {**shown, **{key: icons[action] for key, action in self.actions.items() if action in icons}}

    def _sent(self, e: Event) -> float:
        """The clock time key event `e` went out: when it was sent for (its lags count from then, not from
        the tick it falls on), or now, for one sent for a moment already past."""
        now = self.origin + self.ms / 1000
        return e.at if e.at > now - TICK_MS / 1000 else now

    def note(self, text: str) -> None:
        self.log.append(f"{self.t:8.3f} {text}")

    # --- keys -------------------------------------------------------------------------------------------

    def arrow(self) -> int:
        """The arrow the game sees held (the one it saw pressed later, if both are), as -1/0/+1."""
        if not self.seen:
            return 0
        return directions.sign(max(self.seen, key=self.seen.get))

    def _lag(self, what: str, median: float, sd: float) -> float:
        """s from a key going out (`what`, at this tick) to the game acting on it: the measured median
        (body.Lags), spread by Noise.lag_sd times the measured sd, drawn for that key and tick."""
        if not self.noise.lag_sd:
            return median
        z = random.Random(f"lag:{what}:{self.seed}:{self.ms}").gauss(0.0, 1.0)
        return max(0.0, median + max(-3.0, min(3.0, z)) * sd * self.noise.lag_sd)

    def _seen_later(self, action: str, down: bool, at: float, lag: float) -> None:
        """The game acts on arrow `action` going down/up at clock time `at` at its step `lag` s after it on
        average (_acts). One key's changes are seen in the order they happen: let go and pressed again at
        once, it stays held."""
        ms = max(self._acts(at, lag), self._seen_due.get(action, -10_000))
        self._seen_due[action] = ms
        self._push(ms, "seen", (action, down))

    def _seen(self, action: str, down: bool) -> None:
        """The game sees an arrow go down (she turns that way) or up. In the air the step it turned her at, and the way
        she faced before, are kept (AIR_TURN_SAME_STEP)."""
        if self.dead:
            return
        if not down:
            self.seen.pop(action, None)
            return
        self._seen_count += 1
        self.seen[action] = self._seen_count
        side = LEFT if action == "left" else RIGHT
        if self.mode == "air" and side != self.facing:
            self._air_turn = (self.ms, self.facing)
            self.air_turn_ms = self.ms
            if self.noise.air_turn_lost and self.chance("air_turn_lost") < self.noise.air_turn_lost:
                self.air_turns_lost += 1  # (Noise.air_turn_lost: held all the same, for her steering)
                self.note(f"the arrow {side} in the air doesn't turn her: she faces {self.facing}")
                return
        self.facing = side

    def turn(self, side: str, why: str = "turned from outside") -> None:
        """Something that isn't her keys turns her `side` (Noise.turned, Realism.knock_turns): her frame shows it."""
        if side != self.facing and not self.dead:
            self.facing = side
            self.note(f"{why}: she faces {side}")

    def _pressed(self, action: str, at: float) -> None:
        ground = self.mode == "ground"
        if action in ("left", "right"):
            if self.realism.lock_arrows and at - self.attack_key_at < ARROW_LOCK - 1e-9:
                # Pressed while her attack plays: seen only ARROW_HELD after its key, if still held then.
                self.note(f"{action} pressed {at - self.attack_key_at:.2f} s after the attack's key: swallowed")
                self._push(self.ms_of(self.attack_key_at + ARROW_HELD), "unlock", (action, at))
                return
            self._seen_later(action, True, at, self._lag(action, LAGS.on, LAGS.sd_on))
        elif action == "jump":
            if self.ms < self.attack_lock_ms:
                return
            if "down" in self.held and self.mode != "rope":  # on the ground: a drop; in the air: nothing
                if ground:
                    self._push(self.ms + int(DROP_LAG * 1000), "drop", self.plat.id)
                return
            if self.mode == "air" and not JUMP_TAKEN_LATE:
                return  # pressed in the air: lost
            # It lifts her off at the game's step Lags.jump later on average, if she's on the ground then
            # (_liftoff); with JUMP_TAKEN_LATE, one pressed just before she lands jumps her once she has.
            self._push(self._acts(at, self._lag("jump", LAGS.jump, LAGS.sd_jump)), "liftoff", self.mode)
        elif action in ("attack", "skill"):
            if self.mode == "rope":  # the game refuses it, and says so on the message bar
                self.refused += 1
                self.chat = (self.t, ROPE_REFUSAL)
                return
            if self.ms < self.attack_lock_ms:
                self.attack_locked += 1  # (inside the last attack's lock: the game ignores it)
                return
            if self.character.magic is not None:
                self._press_magic(action, at)
                return
            skill = action == "skill" and self.character.skill is not None
            if skill and self.darts_left is not None and self.darts_left < SKILL_STARS:
                self.no_darts += 1
                self.darts_out_at = self.t if self.darts_out_at is None else self.darts_out_at
                self.note("Lucky Seven with no throwing stars left: nothing")
                return  # (no MP taken, nothing thrown, no lock)
            if skill and self.character.mp > 0 and not self._cast():
                return  # too little MP: nothing thrown, no lock
            if skill and self.darts_left is not None:
                self.darts_left -= SKILL_STARS
            self.attack_lock_ms = self.ms + int(ATTACK_LOCK * 1000)
            self.attack_key_at = at
            self.attacks += 1
            step = self._acts(at, self._lag(action, LAGS.on, LAGS.sd_on))  # the step the game acts on it at
            self._push(self.ms_of(at + JUMP_LAG), "aim", (self.ms, skill, self._key_tag, step))  # it starts then
        elif action == "up":
            if self._through_portal():
                return
            if ground:
                self._grab_from_ground()
        elif action == "down":
            if ground:
                self._grab_from_top()
        elif action == "pick":
            if ground:
                self._pick()
        elif action == "potion":
            if self.potions_left is not None:
                if self.potions_left <= 0:
                    self.note("no potion left")
                    return
                self.potions_left -= 1
            self.potions_used += 1
            self.hp = min(1.0, self.hp + self._potion())
            self.note("drinks a potion")
        elif action == "pet_food":
            self.pet_fed += 1
            self.pet_full, self.pet_since = min(100.0, self.pet_fullness() + pet_care.FOOD), self.t
            self.note("feeds her pet")

    def _released(self, action: str, at: float, since: float) -> None:
        """Key `action`, down since clock time `since`, let go at clock time `at`."""
        if action in ("left", "right"):
            if (self.noise.tap_lost and (at - since) * 1000 <= TAP_LOST_MS + 1e-6 and action in self._seen_due
                    and self.chance("tap") < self.noise.tap_lost):
                # A short tap lost: the game sees it let go as it sees it pressed (it turns her), and it doesn't
                # walk her.
                self._push(self._seen_due[action], "seen", (action, False))
                return
            lag = self._lag(f"{action} up", LAGS.off, LAGS.sd_off)
            if self.noise.walk_jitter and not {"left", "right"} & self.held.keys():
                # letting go of the last arrow: she runs on a little longer or stops a little sooner
                lag += (2 * self.chance("walk") - 1) * self.noise.walk_jitter / self.shoe.vmax
            self._seen_later(action, False, at, lag)

    # --- her motion -------------------------------------------------------------------------------------

    def _ground(self, dt: float) -> None:
        p = self.plat
        if "up" in self.held and self._grab_from_ground():
            return
        if "down" in self.held and self._grab_from_top():
            return
        if self.ms == self.step_at(self.ms):  # the game's step: her speed for the step to come (body.replay)
            lying = "down" in self.held
            d = 0 if lying or self.ms < self.still_ms else self.arrow()
            self.step_arrow = d
            if not d and not self.vx:
                self.glide = 0.0
                return
            self.vx, dx = body.ground_step(self.vx, d, self.shoe, STEP_MS / 1000)
            self.glide = dx * TICK_MS / STEP_MS
        if not self.glide:
            return
        nx = self.x + self.glide
        lo, hi = self.sides
        if self.realism.sides == "client" and not lo <= nx <= hi:  # the map's side stops her as a wall would
            self.x, self.vx, self.glide = min(max(nx, lo), hi), 0.0, 0.0
            self.y = p.y_at(self.x)
            return
        if nx > p.right or nx < p.left:
            right = nx > p.right
            edge = p.right if right else p.left
            if self._blocked[p.id][1 if right else 0]:
                # Against the wall, a hair inside the end: on her side of it, so a flight from here (even one
                # flown on in mid-air, with another arrow) goes into it only toward it (body.fly).
                self.x, self.vx, self.glide = edge - (1e-6 if right else -1e-6), 0.0, 0.0
                self.y = p.y_at(self.x)
                return
            # off the end: in the air from the end of this tick
            d = self.step_arrow
            self._fly(self._flight(nx, p.y_at(edge) - 1, self.vx, 0.0, d), self.ms + TICK_MS, d)
            return
        self.x, self.y = nx, p.y_at(nx)

    def _flight(self, x: float, y: float, vx: float, vy: float, arrow: int, skip: Platform | None = None,
                airborne: bool = False) -> body.Flight:
        """Her arc from (x, y) at (vx, vy) with `arrow` held: body.fly's, step by step on the world's tick,
        with the map's sides stopping her like walls (the planner never jumps at them), and the walls by the
        game's rule (body.wall_table: those joined to the floor under her, and with Realism.zero_walls group 0's)."""
        walls = body.wall_table(self.model, self.realism.zero_walls)
        return body.fly(self.model, x, y, vx, vy, shoe=self.shoe, arrow=arrow, skip=skip, walls=walls,
                        sides=self.sides, dt=TICK_MS / 1000, path=True, airborne=airborne)

    def _steer(self) -> int:
        """The arrow that steers her in the air: the one the game sees, but none in a knock's flight (unless
        KNOCK_STEERS)."""
        return 0 if self.knocked and not KNOCK_STEERS else self.arrow()

    def _launch(self, vx: float, vy: float, from_ms: int) -> None:
        """Into the air from where she is, at (vx, vy), from tick `from_ms` (this one, or the next when it
        happens after this tick's motion)."""
        y = self.y if self.mode != "ground" else self.plat.y_at(self.x)
        d = self._steer()
        self.lift_ms = from_ms
        self._fly(self._flight(self.x, y, vx, vy, d), from_ms, d)

    def _fly(self, f: body.Flight, from_ms: int, arrow: int, skip: Platform | None = None) -> None:
        if self.mode != "air":  # (where she left the ground or the rope: land_log)
            self._took_off = (self.plat.id if self.mode == "ground" and self.plat is not None else None, self.x,
                              from_ms)
        self.mode, self.plat, self.rope, self.flight, self.flight_ms = "air", None, None, f, from_ms
        self.flight_arrow, self.flight_skip = arrow, skip
        self.vx, self.vy = f.speeds[0]

    def _liftoff(self, pressed_on: str) -> None:
        """The game takes a jump key pressed while she was `pressed_on` ground, in the air or on a rope:
        on the ground it jumps her, with the kick toward the arrow it sees (body.takeoff); on the rope it
        was pressed on, with an arrow, it jumps her off sideways. Anywhere else it does nothing."""
        d = self.arrow()
        if self.mode == "ground" and pressed_on != "rope":
            self.knocked = False
            self._launch(body.takeoff(self.vx, d, self.shoe), -self.shoe.vjump, self.ms)
        elif self.mode == "rope" and pressed_on == "rope" and d:
            self._leave_rope()
            self._launch(d * self.rope_jump[0], -self.rope_jump[1], self.ms)

    def _air(self) -> None:
        f = self.flight
        d = self._steer()
        if d != self.flight_arrow:  # the game sees another arrow: she flies on from here with it
            k = min(max(0, (self.ms - self.flight_ms) // TICK_MS), len(f.path) - 1)
            (x, y), (vx, vy) = f.path[k], f.speeds[k]
            f = self._flight(x, y, vx, vy, d, skip=self.flight_skip, airborne=k > 0)
            self._fly(f, self.ms, d, self.flight_skip)
        k = (self.ms + TICK_MS - self.flight_ms) // TICK_MS
        if k >= len(f.path) - 1:
            if f.platform is not None:
                self._land(f.platform, f.land_x, f.vx_land)
                return
            # No floor under her within the flight's 4 s (a long fall): she flies on from where it ends.
            (x, y), (vx, vy) = f.path[-1], f.speeds[-1]
            f = self._flight(x, y, vx, vy, d, skip=self.flight_skip, airborne=True)
            self._fly(f, self.ms, d, self.flight_skip)
            k = 1
        k = min(k, len(f.path) - 1)
        (self.x, self.y), (self.vx, self.vy) = f.path[k], f.speeds[k]
        if f.platform is None and self.y > self.bottom:  # below the map's lowest floor, with none to land on
            self.fell_out_at = self.t
            if self.knocked and self._knocked_from is not None:
                t0, src, y0 = self._knocked_from
                self.knock_log.append((t0, src, None, round(self.y - y0, 1)))
            self.held.clear()
            self.seen.clear()
            self.note(f"falls out of the map at x {self.x:.0f} (below its lowest floor)")
            return
        if "up" in self.held and self.ms - self.left_rope_ms >= REGRAB_AFTER * 1000:
            for r in self.model.ropes:
                if abs(self.x - r.x) <= GRAB_X and r.top <= self.y and self.y - self.arms <= r.bottom:
                    self._grab(r)
                    return

    def _land(self, q: Platform, x: float, vx: float) -> None:
        """Down on platform q at x, going sideways at vx (0 if a wall stopped her): she slides on as the
        ground brakes her (or runs on, with the arrow), except while an attack plays: it stops her. Five
        jump-backs and jump-throws landing during their attack slid 1.4-2.9 units, where braking alone
        has 7.6-11.7 (research: 60 fps recordings). Landing from a knock she goes on at her walk's speed at most
        (KNOCK_LANDS_WALKING: R1-R3's frames). Between the game's steps she stands where she lands until the next
        (body.LANDING_ENDS_STEP)."""
        if self.knocked and self._knocked_from is not None:
            t0, src, y0 = self._knocked_from
            if q.id != src and q.y_at(x) - y0 > KNOCK_FALL_DY:
                self.knock_log.append((t0, src, q.id, round(q.y_at(x) - y0, 1)))
        if self.knocked and self._knock()[3]:
            vx = min(max(vx, -self.shoe.vmax), self.shoe.vmax)
        self._knocked_from = None
        self._log_landing(q, x)
        self.mode, self.plat, self.flight, self.last_plat, self.group, self.knocked = "ground", q, None, q, q.group, False
        self.x, self.y = x, q.y_at(x)
        start, end = self.attack_ms
        self.vx, self.vy = (0.0 if start <= self.ms < end else vx), 0.0
        self.glide = 0.0 if body.LANDING_ENDS_STEP else self.vx * TICK_MS / 1000  # until the game's next step

    def _log_landing(self, q: Platform, x: float) -> None:
        """A landing on q at x, as land_log has it (the takeoff _fly saw, and what's about on q). Her fall off an end
        within KNOCK_FALL_MS of a knock's landing is the knock's ("knock-fall": its slide, or a step's arrow held through
        it, took her off)."""
        took, self._took_off = self._took_off, None
        start, _ = self.attack_ms
        cause = ("knock" if self.knocked else
                 "fight" if took is not None and took[2] - ATTACK_BEFORE_TAKEOFF_MS <= start <= self.ms else
                 "knock-fall" if took is not None and took[2] - self._knock_landed_ms <= KNOCK_FALL_MS else "trip")
        if cause == "knock":
            self._knock_landed_ms = self.ms
        t, near = self.t, None
        for m in self._on(q.id):
            if m.alive:
                mx = m.at(t)
                if near is None or abs(mx - x) < near[0]:
                    way, walking = m.faces(t)
                    near = (abs(mx - x), m.kind, 0 if not walking else (1 if (x - mx) * way > 0 else -1))
        self.land_log.append((round(t, 3), None if took is None else took[0], None if took is None else round(took[1], 1),
                              q.id, round(x, 1), cause, *((None, None, None) if near is None else
                                                         (round(near[0], 1), near[1], near[2]))))

    def _keep(self, m: SimMonster) -> None:
        """A new monster among the walkers on its platform (by_platform), or the jumpers."""
        if isinstance(m, SimJumper):
            self.jumpers.append(m)
        else:
            self.by_platform.setdefault(m.platform.id, []).append(m)

    def _on(self, pid: int) -> list[SimMonster]:
        """The monsters on platform `pid` now: its walkers, and the jumpers standing on it or in the air from it."""
        here = self.by_platform.get(pid, [])
        if not self.jumpers:
            return here
        t, here = self.t, list(here)
        for j in self.jumpers:
            if j.alive:
                j.at(t)
            if j.platform.id == pid:
                here.append(j)
        return here

    def _target_at(self, m: SimMonster, t: float) -> tuple[float, float] | None:
        """Realism.chase: her place (x, her feet's y) for a jumper's chase, None once she's dead."""
        return None if self.dead else (self.x, self.y)

    def _chase_at(self, m: SimMonster, t: float) -> float | None:
        """Realism.chase: her x for monster m's chase stretch starting at t, wherever she is (it walks along its own
        platform at her x), None once she's dead."""
        return None if self.dead else self.x

    def _grab_from_ground(self) -> bool:
        """Up at a rope's foot: onto it."""
        if self.ms - self.left_rope_ms < REGRAB_AFTER * 1000:
            return False
        for r in self.model.ropes:
            if abs(self.x - r.x) <= GRAB_X and r.top < self.y - 5 and self.y - self.arms <= r.bottom:
                self._grab(r)
                return True
        return False

    def _grab_from_top(self) -> bool:
        """Down standing on a rope's top: onto it, climbing down."""
        if self.ms - self.left_rope_ms < REGRAB_AFTER * 1000:
            return False
        for r in self.model.ropes:
            if abs(self.x - r.x) <= GRAB_X and abs(r.top - self.y) <= ROPE_TOP_REACH:
                self._grab(r)
                self.y = r.top + 1
                return True
        return False

    def _grab(self, r: Rope) -> None:
        self.mode, self.rope, self.plat, self.flight, self.group = "rope", r, None, None, body.ROPE_GROUP
        self.knocked = False
        self.x, self.y, self.vx, self.vy = r.x, min(max(self.y, r.top), r.bottom), 0.0, 0.0
        self.grab_log.append((round(self.t, 3), r.x, round(self.y - r.top, 1)))
        self.note(f"grabs the rope at x {r.x:.0f}")

    def _leave_rope(self) -> None:
        self.left_rope_ms = self.ms
        self.rope = None

    def _climb(self, dt: float) -> None:
        r = self.rope
        if "up" in self.held and "down" not in self.held:
            self.y -= self.climb * dt
            if self.y <= r.top:
                self.y = r.top
                top = next((p for p in self.model.places
                            if p.covers(r.x, 4) and abs(p.y_at(r.x) - r.top) <= 20), None)
                if top is not None and r.from_platform:
                    self._leave_rope()
                    self.mode, self.plat, self.last_plat, self.group = "ground", top, top, top.group
                    self.x = min(max(r.x, top.left), top.right)
                    self.y, self.vx, self.glide = top.y_at(self.x), 0.0, 0.0
                    self.still_ms = self.ms + int(round(CLIMB_OVER * 1000))
                    self.climb_log.append((round(self.t, 3), r.x, top.id))
                    self.note(f"climbs onto platform {top.id}")
        elif "down" in self.held and "up" not in self.held:
            self.y += self.climb * dt
            if self.y >= r.bottom:
                self.y = r.bottom
                self._leave_rope()
                self._launch(0.0, 0.0, self.ms + TICK_MS)  # falling from the end of this tick

    # --- monsters and her -------------------------------------------------------------------------------

    def _closeness(self) -> None:
        """The ticks a live monster on the platform she last stood on is within the claw's reach of her (claw_ms) and
        within CLOSE_DX (close_ms), middle to middle."""
        if self.mode == "rope" or self.dead:
            return
        t, x, claw, close = self.t, self.x, False, False
        for m in self._on(self.last_plat.id):
            if m.alive:
                dx = abs(m.at(t) - x)
                claw = claw or dx < reach.CLAW_FAR + m.half
                close = close or dx < CLOSE_DX
        self.claw_ms += TICK_MS * claw
        self.close_ms += TICK_MS * close

    def _touch(self) -> None:
        if self.ms < self.invulnerable_ms or not self.monsters or self.dead:
            return
        t = self.t
        half, tall = self.realism.her_box or HER_TOUCH
        if self.mode == "ground":
            pids = self._near[self.plat.id]
        else:
            pids = [q.id for q in self.model.platforms if q.covers(self.x, 60) and -70 <= q.y_at(self.x) - self.y <= 70]
        near = [m for pid in pids for m in self._on(pid)]
        near += [j for j in self.jumpers if j.alive and j not in near and j.in_air(t)]  # (in the air from elsewhere)
        solid = 0.0 if self.realism.newborns_touch else NEWBORN_SOLID_S
        for m in near:
            if not m.alive or m.born is not None and t - m.born < solid:  # (fading in: it touches nothing yet)
                continue
            mx = m.at(t)
            lo, hi, bottom, top = self.body(m)
            if self.x + half <= mx + lo or self.x - half >= mx + hi:
                continue
            my = m.y(mx)
            if self.y > my - top and self.y - tall < my - bottom:
                self._hit_by(m, mx)
                return

    def _hit_by(self, m: SimMonster, mx: float) -> None:
        self.touches += 1
        self.invulnerable_ms = self.ms + int(round(self.realism.invulnerable * 1000))
        evaded = self.chance("evade") < evade_chance(m.info, self.character, self._mob_levels.get(m.kind))
        damage = 0.0 if evaded else touch_damage(m.info, self.chance("damage"), self.character)
        self.touch_log.append((self.t, damage))
        self.touch_place.append((self.t, self.mode, round(self.x, 1), round(self.y, 1)))
        if m.born is not None and self.t - m.born < FRESH_S:
            self.fresh_log.append((self.t, damage))
        if m.born is not None and self.t - m.born < NEWBORN_S:
            self.newborn_log.append((self.t, damage))
        if evaded:
            self.evaded += 1
            self.note(f"touched by {m.kind} #{m.idx} at x {mx:.0f}: MISS")
            return
        self.hp = max(0.0, self.hp - damage)
        self.hp_low = min(self.hp_low, self.hp)
        away = 1 if self.x >= mx else -1
        self.note(f"touched by {m.kind} #{m.idx} at x {mx:.0f}: HP {self.hp:.1%}")
        if self.hp <= 1e-9:
            self.dead, self.died_at = True, self.t
            self.held.clear()
            self.seen.clear()
            self.note("HP 0: she died")
            return
        self._pet_drinks()
        if self.mode == "rope" and not self.realism.rope_knocks:
            return  # (on the rope still: no knock, her keys held)
        if self.mode == "rope":
            self._leave_rope()
        if self.realism.knock_turns:
            self.turn(LEFT if away > 0 else RIGHT, f"knocked by {m.kind} #{m.idx}")
        if not self.knocked or self._knocked_from is None:  # (a touch in a knock's flight: from where the first was)
            self._knocked_from = (self.t, self.last_plat.id, self.last_plat.y_at(self.last_plat.clamp(self.x)))
        self.knocked = True
        push, vy, adds, _ = self._knock()
        vx = away * push if not adds else (min(self.vx + push, push) if away > 0 else max(self.vx - push, -push))
        self._launch(vx, -vy, self.ms + TICK_MS)  # after this tick's motion

    def _knock(self) -> tuple[float, float, bool, bool]:
        """Her knock as this world has it (Realism.knock): (vx, vy, whether its push sideways adds to the speed she
        has, not past vx, whether she lands from it at her walk's speed at most)."""
        return self.realism.knock or (KNOCK_VX, KNOCK_VY, KNOCK_ADDS, KNOCK_LANDS_WALKING)

    def _potion(self) -> float:
        """The share of her HP a potion of hers heals: Her.potion over her HP, or POTION_HP where that isn't known."""
        her = self.character
        return POTION_HP if her.potion is None else her.potion / her.max_hp

    def _pet_drinks(self) -> None:
        """Her pet's auto potion, right after a touch took HP (Her.pet; PET_GAP): below its warning, a potion from
        her stock (Realism.potions), as her own heals (_potion)."""
        pet = self.character.pet
        if pet is None or self.hp >= pet - 1e-9 or self.ms - self.pet_drank_ms < PET_GAP * 1000:
            return
        if self.pet_fullness() <= pet_care.PACKED_AT:
            self.note("her pet is too hungry to drink a potion: sent home")
            return
        if self.potions_left is not None:
            if self.potions_left <= 0:
                self.note("her pet has no potion left to drink")
                return
            self.potions_left -= 1
        self.potions_used += 1
        self.pet_potions += 1
        self.pet_drank_ms = self.ms
        self.hp = min(1.0, self.hp + self._potion())
        self.note(f"her pet drinks a potion for her: HP {self.hp:.1%}")

    def pet_fullness(self) -> float:
        """Her pet's fullness now: pet.HUNGRY lost at every pet.TICK_S since it was last fed (or the start)."""
        ticks = int((self.t - self.pet_since) // pet_care.TICK_S)
        return max(0.0, self.pet_full - ticks * pet_care.HUNGRY)

    def flinching(self, m: SimMonster) -> bool:
        """Whether monster `m` flinches now: for FLINCH_S after a star or a swipe hit it."""
        return m.flinch_at is not None and self.t - m.flinch_at < monster_data.FLINCH_S

    def body(self, m: SimMonster) -> tuple[float, float, float, float]:
        """Monster `m`'s body now (SimMonster.body): its frame's box, which is what touches her."""
        return m.body(self.t, self.flinching(m))

    def top(self, m: SimMonster) -> float:
        """How tall monster `m` is for an attack to pick it: as tall as its kind walking (MONSTER_HEIGHT, as
        the decisions' reach.body_of has it), taller while it flinches. Not its frame's top (body): the star's
        search was fitted by these heights, and the decisions work out its reach by them. By its frames a pig
        standing 60 below her (38-40 tall) would be picked by the search only from 171 units sideways, not
        151, and by a jump's area (its floor 19 below her feet, reach.AREA_DOWN) not at all, where in game
        pigs 60 below were hit. The attack and the frames don't agree there yet (README: Found offline)."""
        return max(m.height, m.flinch_top) if self.flinching(m) else m.height

    def nearest_in(self, facing: str, far: float, down: float, up: float) -> SimMonster | None:
        """The live monster nearest her whose body reaches into a box on her `facing` side, from her to
        `far` ahead and `down` below to `up` above her feet (reach.reaches_into): the attack's area from a
        jump, whatever the terrain (_aim checks the line to it)."""
        t, d, best = self.t, directions.sign(facing), None
        for m in self.monsters:
            if not m.alive:
                continue
            mx = m.at(t)
            ahead, dy = (mx - self.x) * d, m.y(mx) - self.y
            if not reach.reaches_into(ahead, dy, self.top(m), m.half, far, down, up):
                continue
            if best is None or abs(ahead) < best[0]:
                best = (abs(ahead), m)
        return None if best is None else best[1]

    def air_pick(self, facing: str) -> SimMonster | None:
        """The live monster an attack starting with her in the air picks on her `facing` side (AIR_PICK 'widening'):
        the nearest her whose body reaches into the area about her feet or into its band's window of the star's
        search from her feet now (reach.air_takes' 'widening', whatever reach.AIR_MODEL is), whatever the terrain (_aim
        checks the line to it)."""
        t, d, best = self.t, directions.sign(facing), None
        for m in self.monsters:
            if not m.alive:
                continue
            mx = m.at(t)
            ahead, dy = (mx - self.x) * d, m.y(mx) - self.y
            if not reach.air_takes(ahead, dy, self.top(m), m.half, model="widening"):
                continue
            if best is None or abs(ahead) < best[0]:
                best = (abs(ahead), m)
        return None if best is None else best[1]

    def air_search(self, facing: str, key_ms: int) -> SimMonster | None:
        """The live monster an attack starting with her in the air picks on her `facing` side (AIR_PICK 'search'): the
        star's search from her feet now (search_pick), no claw, no area, as reach.AIR_MODEL has it. With the noise
        (Noise.air_short, air_under), a monster on the platform she last stood on that the search doesn't pick, in front
        of her and nearer than its pick, takes the star with that chance (drawn per attack and monster, the nearest
        first): its body short of the first band (point-blank; with Noise.air_short_ramp, the less the farther short
        it is), or in a band's reach sideways but under its window. With Noise.air_squash each monster's top is that
        share of its height lower at most for it (drawn per attack and monster, uniform): the search picks one the
        likelier the farther its top reaches into a window."""
        top_of = self.top
        if (squash := self.noise.air_squash):
            def top_of(m: SimMonster) -> float:
                return self.top(m) * (1.0 - squash * self.chance(f"air_squash:{m.idx}", key_ms))
        pick = self.search_pick(facing, top_of)
        qs, qu, ramp = self.noise.air_short, self.noise.air_under, self.noise.air_short_ramp
        if not (qs or qu):
            return pick
        t, d = self.t, directions.sign(facing)
        limit = math.inf if pick is None else (pick.at(t) - self.x) * d
        near = []
        for m in self.monsters:
            if not m.alive or m is pick:
                continue
            mx = m.at(t)
            if m.platform.id != self.last_plat.id:
                continue
            ahead = (mx - self.x) * d
            lo, hi, bottom, top = reach.body_box(ahead, m.y(mx) - self.y, top_of(m), m.half)
            if hi <= 0 or ahead >= limit or reach.search_band(lo, hi, bottom, top) is not None:
                continue
            short = reach.SEARCH_START - hi
            near.append((ahead, m, qu if short < 0 else qs * max(0.0, 1.0 - short / ramp) if ramp else qs))
        for _, m, q in sorted(near, key=lambda c: c[0]):
            if q and self.chance(f"air_near:{m.idx}", key_ms) < q:
                return m
        return pick

    def _uncast(self) -> None:
        """A Lucky Seven key that swiped with the claw (on the ground, a monster in the stance's box): its cast's MP and
        stars, taken at its key (_cast), given back, as the game takes none (Her.claw). Her pet's drink at that cast, if
        any, stays drunk."""
        her = self.character
        if her.mp > 0 and self._mp_unshown and self.casts > 0:
            ms, d = self._mp_unshown.pop()
            self.mp = min(1.0, self.mp + her.cast_mp / her.mp)
            self.casts -= 1
            self.mp_used -= her.cast_mp
            if abs(d + her.cast_mp / her.mp) > 1e-9:  # (the pet's drink with it: shown as it would have been)
                self._mp_unshown.append((ms, d + her.cast_mp / her.mp))
        if self.darts_left is not None:
            self.darts_left += SKILL_STARS
        self.swiped_casts += 1

    def swiped(self, facing: str, stance: str) -> SimMonster | None:
        """The live monster the claw's `stance` swipes on her `facing` side: the nearest her (by distance,
        feet to feet, as the client has it) whose body reaches into the stance's box (reach.claw_takes), on
        her level or not, whatever the terrain (the client's whack has no line to check)."""
        t, d, best = self.t, directions.sign(facing), None
        for m in self.monsters:
            if not m.alive:
                continue
            mx = m.at(t)
            ahead, dy = (mx - self.x) * d, m.y(mx) - self.y
            if not reach.claw_takes(ahead, dy, self.top(m), m.half, stance):
                continue
            if best is None or math.hypot(ahead, dy) < best[0]:
                best = (math.hypot(ahead, dy), m)
        return None if best is None else best[1]

    def search_pick(self, facing: str, top: Callable[[SimMonster], float] | None = None) -> SimMonster | None:
        """The live monster the star's search from the ground picks on her `facing` side, whatever the
        terrain (_aim checks the line to it): of those whose body overlaps one of its bands
        (reach.search_band), the first band's; of those, one drawn at random (seeded, per attack), as the
        client takes the one that entered the map first, which nothing tells. `top`: how tall each is for it
        (World.top by default)."""
        t, d, band, first = self.t, directions.sign(facing), None, []
        top = top or self.top
        for m in self.monsters:
            if not m.alive:
                continue
            mx = m.at(t)
            ahead = (mx - self.x) * d
            k = reach.search_band(*reach.body_box(ahead, m.y(mx) - self.y, top(m), m.half))
            if k is None:
                continue
            if band is None or k < band:
                band, first = k, [m]
            elif k == band:
                first.append(m)
        if len(first) > 1:
            return first[int(self.chance("band") * len(first))]
        return first[0] if first else None

    def sight_stop(self, m: SimMonster, skill: bool = False):
        """What stops the star's line from her now to monster `m` (reach.wall_between, by reach.LINE: from the
        foothold group of the platform she last stood on, her feet where they are now, in a jump too; m's body
        as the attack picks it, World.top), or None. To the client's own hit point too, whatever her decisions take
        (reach.CLIENT_HIT_POINT): past a wall rising from its platform's end where its body is over it, up close; not
        with Realism.hit_inside. A skill's (`skill`: Lucky Seven's) row from Realism.skill_row ahead of her if set."""
        mx = m.at(self.t)
        row = self.realism.skill_row if skill else None
        return reach.wall_between(self.last_plat, self.x, self.y, mx, target=m.platform, ty=m.y(mx),
                                  body=(self.top(m), m.half), client=not self.realism.hit_inside, row_start=row)

    def _press_magic(self, action: str, at: float) -> None:
        """Her attack key (`action` 'attack', her wand's swing, or 'skill', Magic Claw) pressed outside the lock, as the
        game takes it for the magician (Her.magic): in the air nothing at all (no MP, no lock, no attack: her casts are
        from the ground only); a cast with too little MP refused (_cast: MP_REFUSAL on the message bar, no lock); else
        the attack locks her for its lock (SimMagic.lock_s; a swing the thief's ATTACK_LOCK) and starts JUMP_LAG after
        its key (_aim_magic)."""
        magic = self.character.magic
        what = "Magic Claw" if action == "skill" else "her swing"
        if self.mode != "ground":
            self.air_presses += 1
            self.note(f"{what} pressed in the air ({self._key_tag or 'no plan'}): nothing")
            return
        skill = action == "skill"
        if skill and not self._cast():
            return  # too little MP: refused, said (MP_REFUSAL), no lock
        self.attack_lock_ms = self.ms + int(round((magic.lock_s if skill else ATTACK_LOCK) * 1000))
        self.attack_key_at = at
        self.attacks += 1
        step = self._acts(at, self._lag(action, LAGS.on, LAGS.sd_on))
        self._push(self.ms_of(at + JUMP_LAG), "aim", (self.ms, skill, self._key_tag, step))

    def magic_pick(self, facing: str) -> SimMonster | None:
        """The live monster Magic Claw's search from her feet now picks on her `facing` side (SimMagic.search: the
        client's for an aimed spell, by the bodies as the star's search takes them, World.top; whatever the terrain,
        nothing stops it): of those whose body overlaps one of its bands (magic_band), the first band's; of those, one
        drawn at random (seeded, per attack), as search_pick."""
        s = self.character.magic.search
        t, d, band, first = self.t, directions.sign(facing), None, []
        for m in self.monsters:
            if not m.alive:
                continue
            mx = m.at(t)
            k = magic_band(s, *reach.body_box((mx - self.x) * d, m.y(mx) - self.y, self.top(m), m.half))
            if k is None:
                continue
            if band is None or k < band:
                band, first = k, [m]
            elif k == band:
                first.append(m)
        if len(first) > 1:
            return first[int(self.chance("band") * len(first))]
        return first[0] if first else None

    def _aim_magic(self, key_ms: int, skill: bool, tag: str) -> None:
        """Her attack keyed at `key_ms` starts (Her.magic): Magic Claw picks by its search from where she is now
        (magic_pick), her wand's swing nothing (its damage not kept); its damage the server's DAMAGE_SHOWS_MS after the
        key (_attack_magic), on that monster wherever it gets to."""
        magic = self.character.magic
        self.attack_ms = (self.ms, key_ms + int(round((magic.lock_s if skill else ATTACK_LOCK) * 1000)))
        facing = self.facing
        due = [way for when, way in self.meant if when <= self.attack_key_at + 1e-6]
        meant = due[-1] if due else facing
        if due:
            self.meant = [(when, way) for when, way in self.meant if when > self.attack_key_at + 1e-6]
            self.attacks_meant += 1
            if facing != due[-1]:
                self.wrong_way += 1
                self.note(f"attack meant {due[-1]} goes {facing}")
        target = self.magic_pick(facing) if skill else None
        if not skill:
            self.swings += 1
        up = None if target is None else self.y - target.y(target.at(self.t))
        near = self._nearest_ahead(meant)
        pick = (False, None, None if near is None else round(near[0], 1), None if near is None else round(near[1], 1),
                near is not None and near[2] is target, skill)
        how = (key_ms / 1000, tag.rsplit("_", 1)[0], self.mode, 0.0, None if target is None else target.kind,
               None if up is None else round(up, 1), pick)
        damage = DAMAGE_SHOWS_MS if self.realism.damage_ms is None else self.realism.damage_ms
        self._push(key_ms + damage, "star", (facing, target, False, None, skill, how, key_ms))

    def _attack_magic(self, facing: str, m: SimMonster | None, skill: bool, how: tuple | None,
                      key_ms: int | None) -> None:
        """Magic Claw's damage (Her.magic), the server's DAMAGE_SHOWS_MS after its key, on the monster it picked when
        it started: SimMagic.hits hits each drawn from SimMagic.hit before the monster's MDDamage (its HP is kept in
        damage before that defence: _monster), or a miss (SimMagic.miss); its death then, or its flinch, push and
        hiding as her hit shows on it (SimMagic.shows_s after the key, no flight: _struck)."""
        magic = self.character.magic
        what = "Magic Claw" if skill else "her swing"
        if m is None or not m.alive:
            self.note(f"{what} {facing}: no hit")
            self._logged(how, "nothing")
            self.casts_nothing += skill
            return
        mx = m.at(self.t)
        dx = mx - self.x
        shows = self.ms if key_ms is None else max(self.ms, key_ms + int(round(magic.shows_s * 1000)))
        if magic.miss and self.chance("miss") < magic.miss:
            self.note(f"Magic Claw misses {m.kind} #{m.idx} at {dx:+.0f}")
            self._logged(how, "miss")
            feet = m.platform.y_at(m.platform.clamp(mx))
            self.misses = [p for p in self.misses if p[0] > self.t - 5.0] + [(shows / 1000, mx, feet, m.height)]
            return
        lo, hi = magic.hit
        damage = sum(lo + int(self.chance(f"damage:{k}") * (hi - lo + 1)) for k in range(magic.hits)) / STAR_MEAN
        before = m.hp
        m.hp = round(m.hp - damage, 6)
        self._logged(how, "hit" if m.hp > 1e-9 else "kill")
        if m.hp > 1e-9:
            self.note(f"Magic Claw hits {m.kind} #{m.idx} at {dx:+.0f} ({m.hp * STAR_MEAN:.0f} of "
                      f"{m.hp_max * STAR_MEAN:.0f} left before its defence)")
            self._push(shows, "struck", (m, False, "Magic Claw"))
            return
        self._kill(m, mx, "Magic Claw", key_ms, min(1.0, max(0.0, before / m.hp_max)))

    def _aim(self, key_ms: int, skill: bool = False, tag: str = "", step: int | None = None) -> None:
        """The attack keyed at `key_ms` starts: it picks its target from where she is now, and its damage
        is the server's DAMAGE_SHOWS_MS after the key (_attack). On the ground it takes one of the claw's stances (drawn,
        seeded: reach.CLAW's), and the claw swipes the nearest monster that stance's box reaches (swiped);
        else the star's search picks (search_pick). In the air, the nearest in the attack's area or its band's
        window of the search from her feet takes the star (AIR_PICK 'widening': air_pick), the nearest in the area
        alone ('area'), or as on the ground from where she is then ('v83'). Each picks by the bodies
        alone; then, as the client does, the line to a star's pick alone is checked (sight_stop): stopped, the
        attack hits nothing, and no other monster takes it. In the air an arrow acted on at its key's `step` turned
        her too late for it (AIR_TURN_SAME_STEP): it goes the way she faced before. The magician's (Her.magic):
        _aim_magic."""
        if self.character.magic is not None:
            self._aim_magic(key_ms, skill, tag)
            return
        self.attack_ms = (self.ms, key_ms + int(ATTACK_LOCK * 1000))  # its animation plays until the lock ends
        facing = self.facing
        if (AIR_TURN_SAME_STEP and step is not None and self._air_turn is not None and self._air_turn[0] == step
                and self.mode == "air"):
            facing = self._air_turn[1] or facing
            self.note(f"turned {self.facing} at the step of the attack's key: it goes {facing}")
        due = [way for when, way in self.meant if when <= self.attack_key_at + 1e-6]
        meant = due[-1] if due else facing  # (the way the stack threw it: pick_log's nearest is that way)
        if due:  # (the latest meant by then: one whose key the game didn't take is passed over)
            self.meant = [(when, way) for when, way in self.meant if when > self.attack_key_at + 1e-6]
            self.attacks_meant += 1
            if facing != due[-1]:
                self.wrong_way += 1
                self.note(f"attack meant {due[-1]} goes {facing}")
        starless = not skill and self.darts_left is not None and self.darts_left < 1  # (Realism.darts: none left)
        air_pick = self.realism.air_pick or AIR_PICK
        if self.mode == "ground" or (air_pick == "v83" and self.mode == "air"):
            stances = reach.CLAW.stances
            target = self.swiped(facing, stances[int(self.chance("stance") * len(stances))])
            swipe = target is not None
            if swipe and skill and self.mode == "ground":
                self._uncast()  # the Lucky Seven key swipes up close as the attack key does: no cast (Her.claw)
                skill = False
            if target is None and not starless:
                target = self.search_pick(facing)
        else:
            target = (self.air_search(facing, key_ms) if air_pick == "search"
                      else self.air_pick(facing) if air_pick == "widening"
                      else self.nearest_in(facing, reach.STAR_RANGE, reach.AREA_DOWN, reach.AREA_UP))
            swipe = False
            if target is not None and target.kind == "snail" and self.noise.air_snail:
                near, take = self.noise.air_snail
                if abs(target.at(self.t) - self.x) < near and self.chance("air_snail", key_ms) >= take:
                    target = None  # over it (Noise.air_snail): the star hits nothing
        if starless and not swipe:
            target = None  # no star to throw: the claw's swing alone
            self.no_darts += 1
            self.darts_out_at = self.t if self.darts_out_at is None else self.darts_out_at
        elif not skill and not swipe and self.darts_left is not None:
            self.darts_left -= 1
        stop = None if target is None or swipe else self.sight_stop(target, skill)
        if stop is not None:
            target, stop = None, (target, stop)
        rise = self.last_plat.y_at(self.last_plat.clamp(self.x)) - self.y  # her height above where she last stood
        picked = target if stop is None else stop[0]
        up = None if picked is None else self.y - picked.y(picked.at(self.t))  # its feet above hers, + above
        near = self._nearest_ahead(meant)
        turned = self.mode == "air" and self.air_turn_ms >= self.lift_ms
        pick = (turned, key_ms - self.lift_ms if self.mode == "air" else None,
                None if near is None else round(near[0], 1), None if near is None else round(near[1], 1),
                near is not None and near[2] is picked, skill)
        how = (key_ms / 1000, tag.rsplit("_", 1)[0], self.mode, round(rise, 1),
               None if picked is None else picked.kind, None if up is None else round(up, 1), pick)
        damage = DAMAGE_SHOWS_MS if self.realism.damage_ms is None else self.realism.damage_ms
        self._push(key_ms + damage, "star", (facing, target, swipe, stop, skill, how, key_ms))

    def _attack(self, facing: str, m: SimMonster | None, swipe: bool, stop: tuple | None = None,
                skill: bool = False, how: tuple | None = None, key_ms: int | None = None) -> None:
        """The attack's damage, the server's DAMAGE_SHOWS_MS after its key (`key_ms`): on the monster it picked when it
        started (_aim), a claw swipe or a star; none if what stops a star stopped it (`stop`: the monster picked, and
        the foothold). Its HP (and bar), or its death and her EXP, then; its flinch, its push and its damage numbers (it
        hides under them: Noise.hit_hidden) as the star lands, STAR_DELAY after the key (_struck). `skill`: Lucky
        Seven's (Her.skill), its stars' damage on a swipe's pick too. `how`: what _aim knew of it, for attack_log.
        The magician's (Her.magic): _attack_magic."""
        if self.character.magic is not None:
            self._attack_magic(facing, m, skill, how, key_ms)
            return
        if stop is not None:
            picked, f = stop
            where = f"the wall at x {f.x1:.0f}" if f.x1 == f.x2 else f"the floor at y {f.y1:.0f}"
            self.note(f"star thrown {facing} at {picked.kind} #{picked.idx}: stopped by {where}")
            self._logged(how, "stopped")
            self.casts_nothing += skill
            return
        if m is None or not m.alive:
            self.note(f"star thrown {facing}: no hit")
            self._logged(how, "nothing")
            self.casts_nothing += skill
            return
        mx = m.at(self.t)
        dx = mx - self.x
        what = "claw swipe" if swipe else "star"
        if not swipe and (self.noise.miss and self.chance("miss") < self.noise.miss or not skill and self.noise.regular_miss
                          and self.chance("regular_miss") < self.noise.regular_miss):
            self.note(f"star misses {m.kind} #{m.idx} at {dx:+.0f}")
            self._logged(how, "miss")
            if not (self.noise.miss_unseen and self.chance("miss_unseen") < self.noise.miss_unseen):
                lands = self.ms if key_ms is None else max(self.ms, key_ms + int(STAR_DELAY * 1000))
                feet = m.platform.y_at(m.platform.clamp(mx))
                self.misses = [p for p in self.misses if p[0] > self.t - 5.0] + [(lands / 1000, mx, feet, m.height)]
            return
        her = self.character
        stars = her.skill if skill else her.stars
        if her.spread:
            damage = self._damage(stars, swipe and stars == 1.0)
        else:
            damage = stars if stars != 1.0 else CLAW if swipe else 1.0
        before = m.hp
        m.hp = round(m.hp - damage, 6)
        self._logged(how, "swipe" if swipe and m.hp > 1e-9 else "hit" if m.hp > 1e-9 else "kill")
        if m.hp > 1e-9:
            self.note(f"{what} hits {m.kind} #{m.idx} at {dx:+.0f} ({m.hp:g}/{m.hp_max:g} left)")
            lands = self.ms if key_ms is None else max(self.ms, key_ms + int(STAR_DELAY * 1000))
            self._push(lands, "struck", (m, swipe, what))
            return
        self._kill(m, mx, what, key_ms, min(1.0, max(0.0, before / m.hp_max)))

    def _struck(self, m: SimMonster, swipe: bool, what: str) -> None:
        """A hit's star lands on monster `m` (STAR_DELAY after its key, its damage the server's already: _attack): its
        damage numbers (it hides under them: Noise.hit_hidden), and its flinch and push if its damage reaches its
        `pushed` (PUSH_BY_DAMAGE), and its chase of her (Realism.chase). Nothing if it's dead by then."""
        if not m.alive:
            return
        m.hit_at = self.t
        if self._pushes(m, swipe):
            mx = m.at(self.t)
            m.flinch_at = self.t
            if MOB_KNOCKBACK:
                m.knock(self.t, 1 if mx >= self.x else -1)
        else:
            self.note(f"the {what} at {m.kind} #{m.idx}: not pushed")
        m.aggro(self.t)  # (Realism.chase: it chases her from now)

    def _logged(self, how: tuple | None, outcome: str) -> None:
        """An attack's outcome into attack_log (`how`: _aim's (key time, plan, ground or air, her rise, the kind picked,
        how far above her feet it was, what pick_log keeps of it)), and pick_log."""
        if how is not None:
            t, plan, mode, rise, kind, up, pick = how
            self.attack_log.append((round(t, 3), plan, mode, rise, kind, outcome, up))
            self.pick_log.append((round(t, 3), plan, mode, *pick, outcome))
            if self.on_attack is not None:
                self.on_attack(self.attack_log[-1])

    def _nearest_ahead(self, facing: str) -> tuple[float, float, "SimMonster"] | None:
        """The nearest live monster on the platform she last stood on ahead of her (its middle in front of hers): how far
        ahead, how much lower its feet are than the floor under her (+ lower), and it; None with none (pick_log)."""
        t, d, p, best = self.t, directions.sign(facing), self.last_plat, None
        floor = p.y_at(p.clamp(self.x))
        for m in self._on(p.id):
            if m.alive:
                mx = m.at(t)
                ahead = (mx - self.x) * d
                if ahead >= 0 and (best is None or ahead < best[0]):
                    best = (ahead, m.y(mx) - floor, m)
        return best

    def _damage(self, stars: float, swipe: bool) -> float:
        """An attack's damage drawn (Her.spread), in average stars (STAR_MEAN): a claw swipe's (CLAW_DAMAGE), one star's
        (STAR_DAMAGE), or SKILL_STARS stars of `stars` / SKILL_STARS of one each; every star a whole number from its
        least to its most, as likely each, seeded by the tick."""
        her = self.character
        lo, hi = (her.claw or CLAW_DAMAGE) if swipe else (her.star or STAR_DAMAGE)
        n, each = (1, 1.0) if swipe or stars == 1.0 else (SKILL_STARS, stars / SKILL_STARS)
        if not swipe and stars != 1.0 and her.skill_star is not None:
            (lo, hi), each = her.skill_star, 1.0
        lo, hi = round(lo * each), round(hi * each)
        total = sum(lo + int(self.chance(f"damage:{k}") * (hi - lo + 1)) for k in range(n))
        return total / STAR_MEAN

    def _cast(self) -> bool:
        """A Lucky Seven cast's MP (CAST_MP) taken, if she has it (Her.mp): then her pet may drink for her (Her.pet_mp).
        False with too little: nothing is thrown. Her bar shows both MP_SHOWS_MS on (mp_drawn)."""
        her = self.character
        if self.mp * her.mp < her.cast_mp - 1e-9:
            self.no_mp += 1
            if her.magic is not None:  # (the magician's: the game says so on the message bar)
                self.chat = (self.t, MP_REFUSAL)
            self.note(f"{'Magic Claw' if her.magic is not None else 'Lucky Seven'} with {self.mp * her.mp:.0f} MP: not "
                      f"enough, nothing {'cast' if her.magic is not None else 'thrown'}")
            return False
        before = self.mp
        self.mp -= her.cast_mp / her.mp
        self.casts += 1
        self.mp_used += her.cast_mp
        if her.pet_mp is not None and self.mp < her.pet_mp and self.pet_fullness() > pet_care.PACKED_AT:
            self.mp = min(1.0, self.mp + her.mp_potion / her.mp)
            self.mp_potions += 1
            self.note(f"her pet drinks an MP potion for her: MP {self.mp:.1%}")
        self._mp_unshown.append((self.ms + MP_SHOWS_MS, self.mp - before))
        return True

    def mp_drawn(self) -> float:
        """Her MP as her bar draws it now: less the casts (and her pet's drinks) it doesn't show yet (MP_SHOWS_MS)."""
        self._mp_unshown = [(ms, d) for ms, d in self._mp_unshown if ms > self.ms]
        return min(1.0, max(0.0, self.mp - sum(d for _, d in self._mp_unshown)))

    def _pushes(self, m: SimMonster, swipe: bool) -> bool:
        """Whether this hit pushes monster `m` back (PUSH_BY_DAMAGE: drawn by push_chance, seeded by the tick); her
        Magic Claw's (Her.magic) by its hit's damage (SimMagic.hit) after the monster's MDDamage: on Tree Dungeon every
        one of hers (157 and up, against a Horny Mushroom's `pushed` 40 and a Green's 25 after its 30)."""
        if not PUSH_BY_DAMAGE or m.info is None or m.info.pushed <= 1:
            return True
        if (magic := self.character.magic) is not None:
            lo, hi = magic.hit
            mdd = self._mob_mdd.get(m.kind, 0.0)
            p = sum(1 for d in range(lo, hi + 1) if d * 100 / (mdd + 100) >= m.info.pushed) / (hi - lo + 1)
            return p >= 1.0 or (p > 0.0 and self.chance(f"push:{m.idx}") < p)
        p = push_chance(m.info.pushed, m.info.pdd, swipe)
        return p >= 1.0 or (p > 0.0 and self.chance(f"push:{m.idx}") < p)

    def _kill(self, m: SimMonster, mx: float, what: str, key_ms: int | None = None, bar: float = 1.0) -> None:
        t = self.t
        m.alive, m.died_at, m.died_x, m.killed_by = False, t, mx, "her"
        if self.realism.dies_late and key_ms is not None:  # drawn alive, walking on, until her star lands
            lands = key_ms / 1000 + (DIE_LANDS_S + abs(mx - self.x) / DIE_FLIGHT if self.character.magic is None
                                     else self.character.magic.shows_s)  # (her Magic Claw: no flight)
            if lands > t:  # (where it gets to on a copy: its own segments go on as its frames are drawn)
                m.drawn_alive, m.bar_drawn = lands, bar
                m.died_at, m.died_x = lands, m.ghost().at(lands)
        mx = m.died_x  # (drawn alive until her star lands, Realism.dies_late: it dies, and its drops pop out, there)
        self.kills[m.kind] = self.kills.get(m.kind, 0) + 1
        self.kill_log.append((t, m.kind, m.platform.id, "her"))
        self.death_log.append((t, m.kind, m.platform.id, mx))
        self.exp += self.model.mob_exp.get(m.name, 1)
        self.note(f"{what} kills {m.kind} #{m.idx} at x {mx:.0f} (platform {m.platform.id})")
        p = m.platform
        drops = []
        if self.realism.drop_mix is not None:
            drops = self._mixed_drops(m, mx, t + DROP_DELAY)
        else:
            rng = self._drop_rng if self._drop_table else self.rng
            for kind in self._drop_kinds(m.kind, rng):
                x = min(max(mx + rng.uniform(-10, 10), p.left + 3), p.right - 3)
                drops.append(Drop(kind, x, p.y_at(x), p.id, t + DROP_DELAY))
        self._push(self.ms + int(DROP_DELAY * 1000), "loot", drops)
        if self.realism.drop_takers:  # (taken by others once she owns them no more, at TAKE_RATE a minute)
            for d in drops:
                life = others_life(1.0 - self._taker_rng.random(), self.realism.rival_owns)
                d.life = expiry_life(d.since, self._expiry_at)
                if life < d.life:
                    self._push(self.ms + int((DROP_DELAY + life) * 1000), "others_take", [d])
        if self.realism.respawn_tick <= 0:
            self._push(self.ms + int(self.realism.respawn * 1000), "respawn", m)

    def _respawn(self, m: SimMonster) -> None:
        """Monster `m` back at its spawn point."""
        m.alive, m.hp, m.died_at, m.hit_at, m.flinch_at = True, m.hp_max, None, None, None
        m.drawn_alive, m.bar_drawn, m.killed_by = None, 1.0, ""
        m.aggro_until, m.chase_dir = None, 0
        m.life += 1
        m.born = self.t
        m.start(self.t, m.home)
        self._camps_watch(m)

    def _respawn_cap(self) -> int:
        """The most monsters the map has at once: the server's formula by the players there (MapModel.mob_cap), or
        Realism.respawn_cap of its spawn points."""
        r = self.realism
        if r.respawn_cap is None:
            return self.model.mob_cap(1 + len(r.camps) + (1 if r.rival_every > 0 else 0))
        return round(self.points * min(1.0, r.respawn_cap))

    def _map_tick(self) -> None:
        """The map's respawn pass (Realism.respawn_tick), as the server's (OrionAlpha's LifePool.tryCreateMob): the cap
        less the monsters alive (or still drawn so) spawned at spawn points drawn at random among the clear ones, a
        point clear with no live monster within SPAWN_CLEAR of it across and up or down, nor a point found clear before
        it this pass (the points in the map's order, as the server goes through them); at each a monster of the point's
        whose death is over (drawn and faded) back, else a new one made (its own still fading out, or alive and walked
        off past SPAWN_CLEAR)."""
        t, rng = self.t, self._tick_rng
        self.respawn_ticks += 1
        room = self._respawn_cap() - sum(1 for m in self.monsters if m.alive or m.drawn_alive is not None and
                                         t < m.drawn_alive)
        if room <= 0:
            return
        # (where each is, from a copy: its own walk is drawn as it's looked at, aimed at where she is then with
        # Realism.chase, so a look of the pass's own would move it)
        near = [(x, g.y(x)) for m in self.monsters if m.alive for g in (m.ghost(),) for x in (g.at(t),)]
        clear = []
        for p in range(self.points):  # (the map's first monsters are its points', in its order: _spawn)
            home = self.monsters[p]
            x, y = home.home, home.platform.y_at(home.home)
            if any(abs(nx - x) < SPAWN_CLEAR and abs(ny - y) < SPAWN_CLEAR for nx, ny in near):
                self.respawn_blocked += 1
                continue
            clear.append(p)
            near.append((x, y))
        rng.shuffle(clear)
        for p in clear[:room]:
            back = [m for m in self.monsters if m.spawn == p and not m.alive and m.died_at is not None
                    and t >= m.died_at + m.fade]
            if back:
                self._respawn(back[0])
            else:
                first = next(m for m in self.monsters if m.spawn == p)
                m = self._monster(len(self.monsters), first.name, first.platform.id, first.home,
                                  random.Random(rng.random()))
                m.point, m.born = p, t
                m.start(t, first.home)
                self.monsters.append(m)
                self._keep(m)
                self._camps_watch(m)
            self.respawn_births += 1

    def _mixed_drops(self, m: "SimMonster", mx: float, lands: float) -> list[Drop]:
        """What a kill of monster `m` at map x `mx` drops by Realism.drop_mix (DropMix), landing at `lands`: none, or a
        pile of its sizes' drops, each mesos (an amount from its kind's range) or one of its kind's items, out from
        where it died on alternating sides `gap` apart, kept on its platform."""
        mix, rng, p = self.realism.drop_mix, self._mix_rng, m.platform
        if rng.random() >= mix.any_share():
            return []
        n = rng.choices([k for k, _ in mix.sizes], weights=[w for _, w in mix.sizes])[0]
        known = drop_data.MONSTERS.get(m.name) if m.name else None
        lo, hi = known.mesos if known is not None and known.mesos is not None else (10, 20)
        items = [] if known is None else [(kind_of(i), known.rates.get(i, min(known.rates.values(), default=NO_RATE)))
                                          for i in known.items]
        side = 1 if rng.random() < 0.5 else -1
        out = []
        for i in range(n):
            off = 0.0 if i == 0 else side * mix.gap * ((i + 1) // 2) * (1 if i % 2 else -1)
            x = min(max(mx + off + rng.uniform(-3.0, 3.0), p.left + 3), p.right - 3)
            if not items or rng.random() < mix.mesos:
                out.append(Drop("mesos", x, p.y_at(x), p.id, lands, amount=rng.randint(lo, hi)))
            else:
                out.append(Drop(rng.choices([k for k, _ in items], weights=[w for _, w in items])[0], x, p.y_at(x),
                                p.id, lands))
        return out

    def _drop_kinds(self, kind: str, rng: random.Random) -> list[str]:
        """What a kill of a monster of this kind drops: by Realism.drops' chances (drawn from `rng`), else mesos
        and its item (ITEM_OF)."""
        if not self._drop_table:
            return ["mesos"] + ([ITEM_OF[kind]] if kind in ITEM_OF else [])
        return [drop for mob, drop, chance in self._drop_table if mob == kind and rng.random() < chance]

    def _rival_due(self) -> int:
        """The tick of another player's next kill: every Realism.rival_every s on average."""
        return self.ms + int(self.realism.rival_every * self._rival_rng.uniform(0.5, 1.5) * 1000)

    def _rival_stays(self) -> int:
        """The tick another player next walks to another place: every Realism.rival_stay s on average."""
        return self.ms + int(self.realism.rival_stay * self._rival_rng.uniform(0.5, 1.5) * 1000)

    def rival_at(self) -> tuple[float, float] | None:
        """Where the other player is now (map units, as its minimap dot places it), or None if nobody else hunts."""
        r = self.rival
        if r is None:
            return None
        if r.goal is None:
            return r.x, r.y
        f = 1.0 if r.arrives <= r.leaves else min(1.0, max(0.0, (self.t - r.leaves) / (r.arrives - r.leaves)))
        return r.start[0] + (r.goal[1] - r.start[0]) * f, r.start[1] + (r.goal[2] - r.start[1]) * f

    def _rival_place(self, m: "SimMonster") -> tuple[int, float, float]:
        """A place to hunt by monster `m`: on its platform, RIVAL_NEAR from it at most."""
        p, mx = m.platform, m.at(self.t)
        x = min(max(mx + self._rival_rng.uniform(-RIVAL_NEAR, RIVAL_NEAR), p.left), p.right)
        return p.id, x, p.y_at(x)

    def _rival_start(self) -> Rival:
        """Where the other player starts: by a monster alive, drawn as _rival_go draws one."""
        pid, x, y = self._rival_place(self._rival_pick([m for m in self.monsters if m.alive]))
        return Rival(x, y, pid)

    def _rival_pick(self, live: list) -> "SimMonster":
        """A monster to go and hunt by, any of `live` alike (so a platform as likely as it has monsters alive)."""
        return live[self._rival_rng.randrange(len(live))]

    def _rival_arrival(self) -> int:
        """The tick the other player gets where it's walking to."""
        return int(math.ceil(self.rival.arrives * 1000 / TICK_MS - 1e-6)) * TICK_MS

    def _rival_go(self, away: bool = False) -> None:
        """The other player sets off for another place (with `away`, by a monster out of its reach)."""
        r, (x, y) = self.rival, self.rival_at()
        live = [m for m in self.monsters if m.alive and not (away and self._rival_reaches(m))]
        if not live:
            return
        pid, gx, gy = self._rival_place(self._rival_pick(live))
        r.goal, r.start, r.platform = (pid, gx, gy), (x, y), None
        r.leaves, r.arrives = self.t, self.t + math.hypot(gx - x, gy - y) / RIVAL_SPEED
        self._push(self._rival_arrival(), "rival_arrives")

    def _rival_reaches(self, m: "SimMonster") -> bool:
        """Whether the other player, hunting where it stands, reaches monster `m` (RIVAL_REACH, RIVAL_DY)."""
        r, mx = self.rival, m.at(self.t)
        return r.goal is None and abs(mx - r.x) <= RIVAL_REACH and abs(m.y(mx) - r.y) <= RIVAL_DY

    def _rival_kill(self) -> int:
        """Another player kills a monster in its reach: no EXP for her, and drops she can't pick up for
        Realism.rival_owns s, which the rival takes Realism.rival_takes s after they land. With none in its reach it
        sets off for another place and kills there once it's there; walking, it kills nothing. Returns the tick of
        its next kill."""
        r = self.rival
        if r.goal is None and not any(m.alive and self._rival_reaches(m) for m in self.monsters):
            self._rival_go(away=True)
        if r.goal is not None:  # on its way: its next kill once it's there
            return max(self._rival_due(), self._rival_arrival() + TICK_MS)
        live = [m for m in self.monsters if m.alive and self._rival_reaches(m)]
        if not live:
            return self._rival_due()
        self._others_kill(live[self._rival_rng.randrange(len(live))], self._rival_rng, self.realism.rival_takes)
        return self._rival_due()

    def _others_kill(self, m: "SimMonster", rng: random.Random, takes: float) -> None:
        """Another player kills monster `m` (a rival or a camp): no EXP for her (but her share of it, with
        Realism.shared_exp), drops drawn from `rng` that she can't pick up for Realism.rival_owns s and that it takes
        `takes` s after they land, and the monster's respawn."""
        t, mx = self.t, m.at(self.t)
        m.alive, m.died_at, m.died_x, m.killed_by = False, t, mx, "rival"
        self.kill_log.append((t, m.kind, m.platform.id, "rival"))
        self.death_log.append((t, m.kind, m.platform.id, mx))
        self.note(f"another player kills {m.kind} #{m.idx} at x {mx:.0f} (platform {m.platform.id})")
        if self.realism.shared_exp and m.hp < m.hp_max:
            share = round(self.model.mob_exp.get(m.name, 1) * (m.hp_max - m.hp) / m.hp_max)
            self.exp += share
            self.note(f"her share of its EXP: {share}")
        p, drops = m.platform, []
        for kind in self._drop_kinds(m.kind, rng):
            x = min(max(mx + rng.uniform(-10, 10), p.left + 3), p.right - 3)
            drops.append(Drop(kind, x, p.y_at(x), p.id, t + DROP_DELAY, owner="rival"))
        self._push(self.ms + int(DROP_DELAY * 1000), "loot", drops)
        self._push(self.ms + int((DROP_DELAY + takes) * 1000), "rival_takes", drops)
        if self.realism.respawn_tick <= 0:
            self._push(self.ms + int(self.realism.respawn * 1000), "respawn", m)

    def _camps_watch(self, m: "SimMonster") -> None:
        """Monster `m` shows (at the start, or respawned): each camp hunting its platform and kind kills it CAMP_KILL
        s from now (if it's still that life of it then)."""
        for c in self.realism.camps:
            if m.platform.id in c.hunts and m.kind in c.kinds:
                self._push(self.ms + int(self._camp_rng.uniform(*CAMP_KILL) * 1000), "camp_kill", (m, m.life))

    def others_at(self) -> list[tuple[float, float]]:
        """Where the other players are now (map units, as their minimap dots place them): the rival (rival_at) and
        the camps (Realism.camps)."""
        out = [] if (r := self.rival_at()) is None else [r]
        for c in self.realism.camps:
            p = self.model.platform(c.stand)
            out.append((c.x, p.y_at(c.x)))
        return out

    def _pick(self) -> None:
        t = self.t
        near = sorted((d for d in self.drops if not d.gone and d.since <= t < d.since + d.life
                       and d.platform == self.plat.id and abs(d.x - self.x) <= PICK_REACH
                       and (d.owner == "her" or t >= d.since + self.realism.rival_owns)),
                      key=lambda d: (abs(d.x - self.x), d.x))
        if self.realism.room_at is not None and t >= self.realism.room_at and self.room_t is None:
            self.room_t = t  # (Realism.room_at: room in every tab again, for good)
            self.note("her tabs have room again")
        for d in near:
            tab = None if d.mesos else tab_of(d.kind)
            if tab is None or not self.tab_full(tab, t) or d.kind in self.tab_kinds.get(tab, {}):
                self._take(d, t)
                return
        if near:
            self.pick_refused += 1
            self.messages.append((t, FULL_MESSAGE))
            self.note("can't pick up: inventory full")

    def _take(self, d: Drop, t: float, pet: bool = False) -> None:
        """Drop `d` taken at t, by her pick-up key or by a pet of hers (`pet`: Realism.pets), hers either way: its line
        in the pickup log ("You have gained mesos", "... x1 earned. (Etc)": a pet's in game too), into her tab."""
        tab = None if d.mesos else tab_of(d.kind)
        d.gone, d.taken_at, d.taken_by, d.pet = True, t, "her", pet
        self.picked[d.kind] = self.picked.get(d.kind, 0) + 1
        self.messages.append((t, f"You have gained mesos (+{d.amount or 15})" if d.mesos else
                              f"{log_name(d.kind)} x1 earned. ({tab})"))
        self.note(f"{'her pet picks' if pet else 'picks'} up {d.kind} at x {d.x:.0f}")
        if tab is not None:
            self.tab_kinds.setdefault(tab, {}).setdefault(d.kind, t)
            self.tab_items[tab] = self.tab_items.get(tab, 0) + 1
            if self.tab_items[tab] >= self._fills.get(tab, math.inf):
                self.full_at[tab] = t
                self.note(f"her {tab} tab is full now")

    def _pets_step(self, dt: float) -> None:
        """Her pets' step (Realism.pets, every PET_STEP_MS; out where she is at the first): each on her platform walks
        to its PET_FOLLOW from her at PET_SPEED, or within it wanders by her (PET_WANDER, at half that), is on hers
        PET_HOP_S after she's on another (its PET_FOLLOW behind her), and takes what it's on that she could take (hers
        or past its owner's time, fitting her tabs: World.takes) as her pick-up does (_take). On a rope or in the air
        she leaves them where they are."""
        t = self.t
        if self.pets is None:
            pid = None if self.plat is None else self.plat.id
            self.pets = [SimPet(pid, self.x, goal=self.x) for _ in range(self.realism.pets)]
        here = self.plat if self.mode == "ground" else None
        if t - self._pet_scan_t >= PET_SCAN_S:
            self._pet_drops = [d for d in self.drops if not d.gone and d.since + d.life > t]
            self._pet_scan_t = t
        for k, pet in enumerate(self.pets):
            follow = PET_FOLLOW[k % len(PET_FOLLOW)]
            if here is not None and here.id != pet.platform:
                if pet.off is None:
                    pet.off = t
                if t - pet.off >= PET_HOP_S:  # (on hers now, behind her)
                    side = 1.0 if self.facing == RIGHT else -1.0
                    pet.platform, pet.x = here.id, min(max(self.x - side * follow, here.left), here.right)
                    pet.off, pet.next = None, t
            elif here is not None:
                pet.off = None
            if pet.platform is None:
                continue
            if here is not None and here.id == pet.platform:
                gap = self.x - pet.x
                if abs(gap) > follow:
                    goal, speed = self.x - math.copysign(follow, gap), PET_SPEED
                else:
                    if t >= pet.next:
                        pet.goal = self.x + self._pet_rng.uniform(-PET_WANDER, PET_WANDER)
                        pet.next = t + self._pet_rng.uniform(*PET_WANDER_S)
                    goal, speed = pet.goal, PET_SPEED / 2
                step = max(-speed * dt, min(speed * dt, goal - pet.x))
                pet.x = min(max(pet.x + step, here.left), here.right)
            for d in self._pet_drops:
                if (not d.gone and d.platform == pet.platform and d.since <= t < d.since + d.life
                        and abs(d.x - pet.x) <= PET_REACH
                        and (d.owner == "her" or t >= d.since + self.realism.rival_owns) and self.takes(d.kind, t)):
                    self._take(d, t, pet=True)

    def tab_full(self, tab: str, t: float) -> bool:
        """Whether her inventory's `tab` was full at t (Realism.full_tabs, fills_after; room again from room_t)."""
        return self.full_at.get(tab, math.inf) <= t and (self.room_t is None or t < self.room_t)

    def takes(self, kind: str, t: float) -> bool:
        """Whether a drop of `kind` fitted her inventory at t: mesos, a tab with room, or a kind she had a stack of."""
        if kind == "mesos":
            return True
        tab = tab_of(kind)
        return not self.tab_full(tab, t) or self.tab_kinds.get(tab, {}).get(kind, math.inf) <= t

    # --- what can be seen -------------------------------------------------------------------------------

    def her(self) -> tuple[float, float]:
        return self.x, self.y

    def live_drops(self) -> list[Drop]:
        """The drops lying on the ground now (landed, not taken, not vanished yet)."""
        t = self.t
        return [d for d in self.drops if not d.gone and d.since <= t < d.since + d.life]

    def visible_monsters(self) -> list[tuple[SimMonster, float, float, bool]]:
        """Every monster that shows: (monster, x, y, alive as drawn); killed ones fade out where they died, those her
        attack killed drawn alive until her star lands (Realism.dies_late: SimMonster.drawn_alive)."""
        t, out = self.t, []
        for m in self.monsters:
            if m.alive or (m.drawn_alive is not None and t < m.drawn_alive):
                x = m.at(t)
                out.append((m, x, m.y(x), True))
            elif m.died_at is not None and 0.0 <= t - m.died_at < m.fade:
                out.append((m, m.died_x, m.y(m.died_x), False))
        return out

    def fading(self, m: SimMonster) -> float | None:
        """Monster `m`'s alpha while it fades in at its spawn point after coming back (Noise.spawn_found on:
        SPAWN_FADE_S from its birth); None: drawn solid (there from the start, its fade over, or the noise off)."""
        if not self.noise.spawn_found or m.born is None or not m.alive:
            return None
        a = (self.t - m.born) / SPAWN_FADE_S
        return a if 0.0 <= a < 1.0 else None

    def drawn_bar(self, m: SimMonster) -> float | None:
        """The HP bar monster `m` is drawn with (its share left; None: none drawn, unhurt): a killed one drawn alive
        (Realism.dies_late) as it was before the killing hit."""
        if not m.alive:
            return m.bar_drawn if m.bar_drawn < 1.0 else None
        return m.hp / m.hp_max if m.hp < m.hp_max else None

    def dying_at(self, kind: str, x: float, y: float, near: float = 20.0) -> "SimMonster | None":
        """The killed monster of `kind` drawn dying now within `near` of (x, y) (its death begun DEATH_SEEN_AFTER ago
        at least, still drawn), if any: what the death watch (FakeEyes.deaths) can see."""
        t = self.t
        for m in self.monsters:
            if (not m.alive and m.kind == kind and m.died_at is not None and DEATH_SEEN_AFTER <= t - m.died_at < m.fade
                    and abs(m.died_x - x) <= near and abs(m.y(m.died_x) - y) <= near):
                return m
        return None

    def aimed_at(self, kind: str, platform: int | None, x: float, key: str) -> None:
        """Her stack chose a throw's target (behavior.attacks' record): the monster of that kind on that platform drawn
        nearest x then, within 40 units (alive, drawn alive though killed, or dying), into `aim_log`: (t, key, its
        index, 'alive', 'hers' or 'rival' for one dead by then; 'none' where none is)."""
        best, d = None, 40.0
        for m, mx, my, _ in self.visible_monsters():
            if m.kind == kind and (platform is None or m.platform.id == platform) and abs(mx - x) <= d:
                best, d = m, abs(mx - x)
        state = "none" if best is None else "alive" if best.alive else best.killed_by or "rival"
        self.aim_log.append((round(self.t, 3), key, None if best is None else best.idx, state))

    def recent_messages(self) -> list[str]:
        t = self.t
        return [text for when, text in self.messages if t - when < MESSAGE_FOR]

    def summary(self) -> dict:
        """Her end state, for the trace."""
        return {"x": round(self.x, 1), "y": round(self.y, 1), "mode": self.mode, "dead": self.dead,
                "platform": None if self.plat is None else self.plat.id, "hp": round(self.hp, 3),
                "exp": self.exp, "kills": dict(sorted(self.kills.items())), "touches": self.touches,
                "evaded": self.evaded, **({"fell_out_at": round(self.fell_out_at, 3)} if self.fell_out_at else {}),
                **({"left_map_at": round(self.left_map_at, 3)} if self.left_map_at is not None else {}),
                **({"pet_potions": self.pet_potions} if self.pet_potions else {}),
                **({"casts": self.casts, "mp_potions": self.mp_potions} if self.casts else {}),
                **({"no_mp": self.no_mp, "air_presses": self.air_presses, "swings": self.swings,
                    "attack_locked": self.attack_locked} if self.character.magic is not None else {}),
                **({"darts": self.darts_left, "no_darts": self.no_darts} if self.darts_left is not None else {}),
                **({"refused": self.refused} if self.refused else {}),
                **({"key_windows": self.key_windows, "keys_in_window": self.keys_in_window,
                    "menu_opened": self.menu_opened, "key_window_open": self.key_window}
                   if self.key_windows or self.menu_opened else {}),
                **({"stat_windows": self.stat_windows, "stat_clicks": self.stat_clicks,
                    "stat_window_open": self.stat_window} if self.stat_windows or self.stat_clicks else {}),
                **({"chat_box": self.chat_box, "chat_presses": self.chat_presses} if self.realism.chat is not None
                   else {}),
                "picked": dict(sorted(self.picked.items())),
                "drops_left": sorted([d.kind, d.platform, round(d.x)] for d in self.live_drops())}


def _body_of(kind: str, b, info: MobInfo | None) -> tuple[float, float]:
    """A monster's half width and height (how tall it walks), as an attack picks it (World.top): the tables'
    (monster_data.MONSTER_HALF, MONSTER_HEIGHT) for their kinds, else from the game data as those were made (half its
    widest stand or move frame, MobBody.half from the map; the top of its first move frame), else a pig-sized 34 by
    46. On West Rocky Mountain II: a Stump, Dark Stump and Axe Stump 33 by 51, a Red Snail 20.5 by 34."""
    if kind in monster_data.MONSTER_HALF:
        return monster_data.MONSTER_HALF[kind], monster_data.MONSTER_HEIGHT.get(kind, 46.0)
    half = b.half if b is not None else 34.0
    return half, info.move[0].top if info is not None and info.move else 46.0


ACTIONS = (("left", "left"), ("right", "right"), ("up", "up"), ("down", "down"), ("jump", "jump"),
           ("attack", "attack"), ("lucky_seven", "skill"), ("magic_claw", "skill"), ("pick_up", "pick"),
           ("hp_potion", "potion"), ("pet_food", "pet_food"))


def _actions(keys: GameKeys, game: tuple = ()) -> dict[str, str]:
    """Key name -> what it does: her keys', but where the game has one of them on another key (Realism.game_keys: her
    GameKeys field, the key, None for none). (The World adds the backslash, the KEY BINDINGS window's 'set_key', and
    Esc.)"""
    out = {}
    for field_name, action in ACTIONS:
        k = getattr(keys, field_name, None)
        if k is not None:
            out[k.name] = action
    named = dict(ACTIONS)
    for field_name, key in game:
        action = named[field_name]
        out = {k: a for k, a in out.items() if a != action}
        if key is not None:
            out[key] = action
    return out


def _sides(model: MapModel, how: str = "platforms") -> tuple[float, float]:
    """The map's sides (Realism.sides, SIDE_INSET): its platforms' outermost ends ('platforms'), or SIDE_INSET in from
    the leftmost and rightmost x of its footholds ('client')."""
    if how == "platforms":
        return min(q.left for q in model.platforms), max(q.right for q in model.platforms)
    if how != "client":
        raise ValueError(f"no such sides: {how!r}")
    xs = [v for q in model.platforms for f in q.segments for v in (f.x1, f.x2)] + \
        [v for w in model.walls for v in (w.x1, w.x2)]
    return min(xs) + SIDE_INSET, max(xs) - SIDE_INSET


def _wall_at(model: MapModel, p: Platform, edge: float, way: int, zero: bool = True) -> bool:
    """Whether a wall rises at this end of platform `p` that stops her going `way` into it (body.wall_table: one joined
    to its end, or with `zero` a wall of group 0 there): she walks into it, not off the end."""
    return body.stops(body.wall_table(model, zero), edge, p.y_at(edge) - 2, way)


def exp_pixels(exp: int) -> np.ndarray:
    """Her EXP number's text pixels (hud.EXP_NUMBER): a pattern that changes whenever her EXP does."""
    return np.random.default_rng(exp).random((26, 140)) > 0.5
