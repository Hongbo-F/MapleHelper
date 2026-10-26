"""A deterministic stand-in for the game: her body, the monsters, stars, drops, HP and EXP on a real map
(mapdata.MapModel), in map units, driven by key events on a fake clock (see sim.fakes).

Her body moves as the game moves it, by motion.body's engine (the one the planner's constants come from),
on the world's 5 ms tick, and the game's 30 ms step (body.STEP: STEP_MS):

- the game sees an arrow pressed half a step sooner than body.Lags.on (66 ms) after its key and let go
  half a step sooner than Lags.off (68 ms) after (Lags.seen), and acts on what it has seen at its next step
  (on average Lags.on and Lags.off after the key), so a tap is as long as the game sees it (2 ms longer
  on average, a whole number of steps), and she turns when it sees the press; of the arrows it sees held,
  the one pressed last counts. The steps fall every 30 ms from a phase drawn from the seed (step_phase);
- on the ground, at each step, the arrow it sees speeds her up at 1400 units/s² to physics.WALK_SPEED
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
  and none drags her at 1 (100 once she falls at the cap). Only the walls of her foothold group stop her
  (body.OWN_GROUP_WALLS: the group of the platform she last stood on; off a rope, none; with Realism.zero_walls
  group 0's too), and the map's sides (Realism.sides). She lands with the speed she has and slides on, unless an
  attack is playing: that stops her
  (jump-backs and jump-throws slid 1.4-2.9 units in game). Below the map's lowest floor she has fallen out
  of the map, and the run ends there (SimFellOut);
- up held at a rope's foot, or anywhere along it in the air, grabs it (its bottom Realism.rope_reach above her
  feet at most); up climbs at flow.ROPE_SPEED onto
  the platform at its top (a short climb-over, flow.CLIMB_OVER), down climbs down at physics.CLIMB_SPEED
  and lets go at its bottom; down on a rope's top grabs it from above; the jump key with an arrow jumps
  her off it sideways (ROPE_JUMP); down and jump on a floor whose foothold forbids it (forbidFallDown)
  does nothing, as on the bottom floor;
- monsters walk their platform as fast as the game data says (monster_data.speeds: 125 units/s for a pig,
  43.75 for a snail), turning and stopping now and then (seeded); a hit that doesn't kill pushes one
  KNOCK_MOB units back from her and stands it while it flinches (MOB_KNOCKBACK), never off its platform, if its
  damage reaches the monster's `pushed` in the game data (push_chance: any damage for the first map's kinds).
  A monster touches her when its body overlaps hers (HER_HALF either side of her feet, HER_HEIGHT tall):
  the box of the frame it shows then in the game data, walking, standing or flinching after a hit, turned
  the way it faces (SimMonster.body), so a hopping slime is off the ground and taller for 38% of its walk.
  A touch leaves her a second's invulnerability. She evades it (a thief's "MISS": nothing else happens)
  with a chance inversely proportional to the monster's accuracy (game data: half a pig's touches, four
  in five of a snail's); the others take a share of her HP by the monster's touch damage (game data:
  about 1.6% for a pig, 0.3% for a snail, give or take 70%; or, her DEF known, by meowdb's formula: Her) and
  knock her back ~58 units (the median
  logged push; the arrow she holds doesn't steer that flight, KNOCK_STEERS). At 0% HP she dies, and the
  run ends (SimDied); up held at a portal (mapdata.PORTAL_BOX: from 100 above it to 25 below, in the air too, as
  HeavenClient has it) takes her through it, out of the map, and the run ends too (SimLeftMap);
- the attack picks its target when it starts, physics.JUMP_LAG after its key (the same input lag as a
  jump's lift-off), from where she is then, and its damage lands actions.STAR_DELAY_MS after the key, on
  that monster (attack.reach), whatever the distance (in game the star is then seen flying up to 0.6 s more;
  whether its damage shows when it gets there isn't measured, so the shot log allows for both:
  behavior.fight.LANDS_FROM_S). On the ground the attack takes one of the claw's stances
  (reach.CLAW's, drawn per attack, seeded), and the nearest monster (by distance, feet to feet) whose
  body reaches into its box (reach.claw_takes: cut to reach.CLAW_FAR ahead of her, on her level or not) is
  swiped (a fifth of a star's damage); else the star's search picks (reach.search_band: of the monsters
  whose body overlaps one of its bands, the first band's, one of those drawn at random, seeded, as the
  client's order there, by when each entered the map, can't be seen), so off her level a star needs room
  sideways, by the monster's body (right under or over her, none is hit).
  In the air, the nearest whose body reaches into the attack's area (its near side within
  reach.STAR_RANGE ahead, from reach.AREA_DOWN below to reach.AREA_UP above her feet, wherever she is
  then) or into its band's window of the star's search from her feet then (AIR_PICK 'widening', the
  default: reach.air_takes) takes a star (with 'area', the area alone; with 'v83', the claw and the search
  as from the ground, from where she is then). For all of these, bodies are their kind's half width either side and as tall as
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
  later where it spawned;
- the pick-up key takes the nearest drop within reach (items only while her inventory has room), the
  HP potion key gives 30% HP, the chair key sits her down (+3% HP/s) until an arrow stands her up; her pet
  (Her.pet), if she has one, drinks a potion for her right after a touch leaves her under its warning.

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
import functools
import json
import math
import os
import random
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
from ..control.actions import STAR_DELAY_MS
from ..localization import monsters as monster_data
from ..localization.mapdata import PORTAL_BOX, PORTAL_EXIT, MapModel, Platform, Rope
from ..motion import bodies as her_body
from ..motion import body, flow
from ..motion.physics import (CLIMB_SPEED, DROP_LAG, GRAB_SURE, JUMP_HEIGHT, JUMP_LAG, LAGS, ROPE_REACH, SHOE,
                               WALK_SPEED)
from ..perception import drops as drop_data
from ..perception.kinds import kind_of

TICK_MS = 5              # the world's tick, which is also the engine's own step in the air (body.TICK)
STEP_MS = round(body.STEP * 1000)  # the game's step: it acts on keys, and sets her speed on the ground, every
                                   # 30 ms (body.STEP)
CLIMB_UP = flow.ROPE_SPEED
CLIMB_DOWN = CLIMB_SPEED
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
ROPE_JUMP = ((SHOE.rope_vx * SHOE.vmax, SHOE.rope_vy * SHOE.vjump) if ROPE_JUMP_SHOE else (WALK_SPEED, 220.0))
# Her body for the world's touches: 15 either side of her feet and 55 tall, as the simulator has had it (the hop was
# timed in game with the 15: flow.HOP_HALF), or the client's (Realism.her_box, `--world client-body`): 22 either side,
# 65 tall (motion.bodies, which the stack plans by: her trips, attack.danger).
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
INVULNERABLE = 1.0
# Its knockback: ~40 units in the air (0.22 s) and a ~19-unit slide on landing, ~58 in all on flat ground
# (hits.jsonl's pushes on her own level: quartiles 21 / 58 / 87 units). Those are all hits with no key
# pressed since the look before (Hunter._log_hit), so what an arrow held does to the flight isn't measured.
# By the engine's air steering the opposite arrow would brake it at 200 units/s² (45 units on flat ground
# instead of 56, toward the monster) and the one held away would carry her on, landing at full speed (65).
# False: the arrow does nothing until she lands, as the fit had it (and the simulator before it ran on the
# engine); on the ground it walks her as ever.
KNOCK_STEERS = False
KNOCK_VX, KNOCK_VY = 180.0, 220.0
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
# A hit that doesn't kill (a star or a swipe) pushes the monster KNOCK_MOB units away from her over
# KNOCK_MOB_S, and it stands in its hit1 frame until monster_data.FLINCH_S after the hit (0.6 s, that
# frame's delay); then it walks or stands as its own draws say. In the game data `pushed` is 1 for all four
# kinds: any damage pushes them. Measured once: a slime walking at her, swiped on the basin floor
# (lab/recordings/do_attack2, static camera, 30 fps), went 13 half-px (18 units) back over about 0.25 s
# and stood about 0.6 s before it walked at her again; a pig hit in `fight1` lost about 80 units of its
# walk (moving camera, so no distance). One sample, so False puts the monsters back to walking through
# hits. Pushed from where it turns (its body at the platform's end), a monster stays on its platform:
# none is pushed off an end (pigs have been seen on platforms where none spawn; how they get there isn't
# known).
MOB_KNOCKBACK = True
KNOCK_MOB = 18.0
KNOCK_MOB_S = 0.25
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
# A touch by a monster within this many s of its respawn came from its spawn point, onto her or beside her (World.
# fresh_log): what a waiting place by a spawn point costs.
FRESH_S = 1.5
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
AIR_PICK = "widening"
# In the air, an arrow the game acts on at the same step (STEP_MS) as an attack's key turns her too late for that
# attack: it goes the way she faced before (the turn shows, the star doesn't follow). Live on 2026-10-25 (the hunt of
# 23:43) 10 of 28 jumps that turned her in the air threw the wrong way, the attack 25 ms after the arrow, and none of 68
# throws from the ground with it 80 ms after (analysis-1025/jumps-perception-skeptic/airturn2.py); how the game
# decides isn't known, and this has 25 ms apart fail one time in six (5 ms of the step's 30 shared), 60 ms never.
# Before 2026-10-26 the world turned her as it saw the arrow, whatever came at that step: 2 of 201 such jumps went
# the wrong way in the harness (master's hunt loop on Tree Dungeon, 8 seeds of 10 minutes). False: as before.
AIR_TURN_SAME_STEP = True
DROP_DELAY = 2.0
# s from a kill until the monster is back at its spawn point (a guess; Realism.respawn). Not the game's, by its live
# telemetry of 2026-10-19 to 21 (analysis.respawn: README, Respawns and drop life): there a platform she emptied,
# watched with no gap over 2 s, had a monster again within 12.56 s on an estimated 20% (95% interval 12-33%; 12 of 77
# seen back, 23 still watched then), and too few were watched past 30 s to say more; in this world 11 of 22 were seen
# back by 8.8 s, the rest watched only to 6.3 s (52% not back by 5 s, 26-72%, against the game's 90%: the same report on
# 8 runs of preset game). The game's basin had a monster at 12% of its spawn points, none 45% of the time, this world's
# 27% and 7%. Kept until the game's refill is settled (a delay, or MeowDB's cap of the map's monsters, a respawn going to
# any free spawn point): the in-game check there.
RESPAWN = 8.0
FADE = 0.8               # s a killed monster is still seen fading out where it died, for a kind whose death
                         # animation the game data doesn't have (else its own: MobBody.die_s, 0.66-1.26 s)
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
# Her pet's fullness (with a pet: Her.pet), as behavior.pet follows it: HUNGRY lost every TICK_S (the game data's 1
# for her Snail pets; 70-74 s a point live on 2026-10-24), FOOD (30) for a Pet Food, sent home at PACKED_AT or under
# (then it drinks nothing). At a run's start PET_FULLNESS, not known: under pet.FEED_BELOW (70), so the pet care's
# first read feeds it once (60 -> 90), as she fed it at the start before it was read. The pet care reads it off the
# world (sim.fakes.FakePetCare: no window, no OCR, no clicks).
PET_FULLNESS = 60.0
CHAIR_HP_PER_S = 0.03
STAND_UP = 0.2           # s the arrow press that stands her up from the chair takes
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
#   floor's; its floating platforms are other groups, with no walls). By the client's movement as reconstructed
#   (body.OWN_GROUP_WALLS), only her own group's walls stop her; RustMS's port has group 0's stop every group's stars
#   (README, Attack). Whether they stop her body is unknown: walked or knocked off the outer end of a floating platform
#   there (11's left end, 13 from the tower's wall, 23's 23, 25's right end 34), or off a rope near the wall (the one
#   at -192, 28 from it: off a rope no wall stops her, body.ROPE_GROUP), she falls past the wall and out of the map, or
#   slides down it onto the shelf below. (HeavenClient has neither: there a wall stops her if it's chained to the
#   foothold below her, its prev or the one before, FootholdTree::get_wall, whatever its group.)
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


@dataclass(frozen=True)
class Noise:
    """The real game's errors, all off by default. Each draw is seeded by the world's seed and the
    moment it's for (a star's tick, a key's tick), and the spells are fixed stretches of time, so a
    run is still the same every time, and a change that only adds or drops a look changes nothing here
    (detect_miss is drawn per look, so with it on, an extra look does change what later looks see)."""
    miss: float = 0.0            # chance a star that would hit misses (lab/shots.jsonl: 72.5% of 204 hit)
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
    heading_wrong: tuple = ()    # (share, score, margin, runs): that share of frames read her facing the wrong way,
                                 # matching `score`, `margin` ahead; `runs` True: drawn per pose of hers for each
                                 # HEADING_SPELL_S (a pose her pictures don't know), False: per frame. Of the frames
                                 # read the wrong way on 2026-10-24 none was strong (heading.Reading.strong): with
                                 # pictures of walks and jumps alone, 9 of 426 frames of throws, at 0.31, 0.24 ahead;
                                 # () off

    @staticmethod
    def during(spells: tuple, t: float) -> bool:
        return any(a <= t < b for a, b in spells)


@dataclass(frozen=True)
class Realism:
    """How busy and how hard the map is. The defaults are the simulator's own world (every spawn point
    filled, an 8 s respawn, nobody else hunting, endless potions), which is denser and easier than the
    game: tour_run makes about 143 EXP a minute where a sweep in game made about 76. The other values are
    knobs for comparing policies in a world closer to the game's; none of them is measured but lock_arrows (and the
    respawn, which the live telemetry says isn't the game's: RESPAWN)."""
    respawn: float = RESPAWN     # s from a kill until the monster is back at its spawn point
    spawn_share: float = 1.0     # share of the map's spawn points that have a monster (seeded)
    rival_every: float = 0.0     # s between another player's kills, on average (0: nobody else)
    rival_owns: float = 15.0     # s a rival's drops can't be picked up by her
    rival_takes: float = 20.0    # s after landing that a rival picks up its own drops
    rival_stay: float = 90.0     # s a rival hunts one place before it walks to another, on average (RIVAL_REACH)
    potions: int | None = None   # potions she has (None: endless)
    drops: tuple | str = ()      # what a kill drops: (monster kind, drop kind, chance) (GAME_DROPS), drawn per
                                 # kill from a generator of their own, or MAP_DROPS: the world's map's (game_drops);
                                 # (): mesos, and a pig's leather (ITEM_OF)
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
    zero_walls: bool = False     # the walls of foothold group 0 stop her whatever her group
    rope_reach: float = ROPE_REACH  # how far above her feet a rope's bottom can be for her to grab it
                                    # (ROPE_REACH_CLIENT: HeavenClient's)
    her_box: tuple[float, float] | None = None  # her body for touches, (half its width, its height): None, HER_HALF
                                                # and HER_HEIGHT; HER_BOX_CLIENT, the client's (motion.bodies)
    knock_turns: bool = False    # a touch's knock turns her to face the monster: HeavenClient's Player::damage doesn't
                                 # (it never calls set_direction), and in the old recordings she never turned without
                                 # an arrow, but no recording has a knock with no arrow held (README, Her heading)
    bars_empty: bool = False     # a hurt monster's HP bar read empty, whatever it has left (sim.fakes.FakeEyes.bar): as
                                 # the stack read them at 1366x768 until 2026-10-25 (perception.hud.hp_bar: 83 of 87
                                 # bars seen in a run read 0; harness `--world bars-empty`)


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
# Which potion she carries on the new maps isn't known: POTION_HP's 30% is 219 HP at 730 and 226 at 752, between a Red
# Potion's 100 and an Orange Potion's 250 (the game data; the Tree Dungeon study's monsters_drops.md); the harness's
# `--world red-potion` has her drink Red Potions.
LEVEL_24 = Her(max_hp=752.0, level=24, exp=0, defence=DEFENCE, pet=0.45, mp=601.0, pet_mp=PET_MP)
# ... and from 2026-10-25 at level 25: HP 774 and MP 623 (her bars' numbers), her stars as read off the monsters' bars
# then (attack.choice.STAR and SKILL_STAR: the regular star 20-140, Lucky Seven's two of 96-192), drawn so with `spread`
# (harness.ATTACK_HER), her Lucky Seven at level 16 (14 MP) and her pet's MP potions Blue Potions (100).
LEVEL_25 = Her(max_hp=774.0, level=25, exp=0, defence=DEFENCE, pet=0.45, mp=623.0, pet_mp=PET_MP, star=(20, 140),
               skill_star=(96, 192), cast_mp=14.0, mp_potion=100.0)


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


def evade_chance(info: MobInfo | None, her: Her | None = None) -> float:
    """The chance she evades a touch by a monster with `info`: EVADE_ACC (or her own, Her.evade) over its accuracy."""
    if info is None or info.acc <= 0:
        return EVADE
    return min(1.0, (her.evade if her is not None and her.evade else EVADE_ACC) / info.acc)


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

    @property
    def lo(self) -> float:
        return self.platform.left + self.half

    @property
    def hi(self) -> float:
        return max(self.lo, self.platform.right - self.half)

    def start(self, t: float, x: float) -> None:
        self.t0, self.x0, self.anim0, self.pushed = t, min(max(x, self.lo), self.hi), t, None
        self._draw()

    def _draw(self) -> None:
        r = self.rng.random()
        self.d = 0 if r < 0.35 else (-1 if r < 0.675 else 1)
        self.t1 = self.t0 + (self.rng.uniform(0.5, 2.5) if self.d == 0 else self.rng.uniform(1.0, 4.0))

    def _fold(self, t: float) -> tuple[float, int]:
        """Where it is at t within the current segment, and which way it's then going (bouncing at the ends)."""
        if not self.d:
            if self.pushed is not None and self.pushed[0] == self.t0 and t < self.t0 + KNOCK_MOB_S:
                return self.pushed[1] + (self.x0 - self.pushed[1]) * (t - self.t0) / KNOCK_MOB_S, 0
            return self.x0, 0
        span = self.hi - self.lo
        if span <= 0:
            return self.lo, self.d
        u = (self.x0 - self.lo) + self.d * self.speed * (t - self.t0)
        m = u % (2 * span)
        return (self.lo + m, self.d) if m <= span else (self.lo + 2 * span - m, -self.d)

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
        walk goes, standing until FLINCH_S after the hit (it faces the way it did). Its walk after is drawn
        then, as at the end of any of its segments."""
        self.at(t)
        x, d = self._fold(t)
        self.pushed, self.anim0, self.facing = (t, x), t, d or self.facing
        self.t0, self.x0, self.d = t, min(max(x + away * KNOCK_MOB, self.lo), self.hi), 0
        self.t1 = t + monster_data.FLINCH_S

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
    taken_by: str = ""             # ... and by whom ("her" or "rival")

    @property
    def mesos(self) -> bool:
        return self.kind == "mesos"


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
        # What a kill drops (Realism.drops; with MAP_DROPS, the map's)
        self._drop_table = game_drops(model) if self.realism.drops == MAP_DROPS else tuple(self.realism.drops)
        self.potions_left = self.realism.potions
        self.potions_used = 0           # potions drunk, by her key or by her pet
        self.pet_potions = 0            # ... of them her pet's (Her.pet)
        self.casts = 0                  # Lucky Seven casts (Her.mp kept), the MP they took, the MP potions her pet
        self.mp_used = 0.0              # drank for her (Her.pet_mp), and the casts pressed with too little MP (none
        self.mp_potions = 0             # thrown)
        self.no_mp = 0
        self.attacks = 0                # attacks her keys made, either key
        # Every attack the game took, as it came out (_attack): (when its key went out, s of the run; the name of the
        # key plan that pressed it, as the stack's player was sent it; "ground" or "air" when it started; how far she
        # was above the platform she last stood on then; the kind of the monster it picked, or None; what came of it:
        # "kill", "hit", "swipe" (the claw's, not killing), "miss" (Noise.miss), "stopped" (its line: sight_stop) or
        # "nothing" (no monster picked, or one dead by then); how far above her feet the pick's were, + above). What
        # each kind of throw hit (sim.harness: `attack_kinds`).
        self.attack_log: list[tuple] = []
        self._key_tag = ""              # the plan of the key being taken (_handle), for attack_log
        self.pet_drank_ms = -10_000     # when her pet last drank one
        self.pet_fed = 0                # times she fed her pet (the pet food key)
        self.pet_full = PET_FULLNESS     # her pet's fullness (Her.pet) as of pet_since (pet_fullness)
        self.pet_since = 0.0
        self.hp_low = hp                # her lowest HP so far
        self.kill_log: list[tuple[float, str, int, str]] = []  # (t, kind, platform, "her" or "rival")
        self.death_log: list[tuple[float, str, int, float]] = []  # ... and where each died: (t, kind, platform, x)
        self.touch_log: list[tuple[float, float]] = []  # (t, the share of her HP it took: 0 if evaded) for every touch
        self.fresh_log: list[tuple[float, float]] = []  # ... of those, the touches by a monster within FRESH_S of its
                                                        # respawn (from its spawn point)
        self.actions = _actions(keys)
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
        self.bottom = max(max(q.y_at(q.left), q.y_at(q.right)) for q in model.platforms)  # its lowest floor
        self._walls: dict = {}          # foothold group -> the walls that stop her with it (body.Walls)
        self.x, self.y = float(start[1]), p.y_at(start[1])
        self.vx = self.vy = 0.0         # her speed (units/s; y down)
        self.glide = 0.0                # on the ground: how far she moves each tick of the game's step (none from a
                                        # landing until the next step: body.LANDING_ENDS_STEP)
        self.step_arrow = 0             # the arrow the game moved her with at its last step on the ground
        self.facing = facing
        self._air_turn: tuple[int, str] | None = None  # the step an arrow last turned her in the air at, her way before
        self.seen: dict[str, int] = {}  # the arrows the game sees held -> the order it saw them pressed in
        self._seen_count = 0
        self._seen_due: dict[str, int] = {}  # each arrow's latest change the game is yet to see (its tick)
        self.still_ms = 0               # she can't move before this tick (climbing over a rope's top, standing up)
        self.left_rope_ms = -10_000
        self.attack_ms = (-10_000, -10_000)  # the last attack's animation: from its start to its lock's end
        self.sitting = False
        self.hp = hp
        self.mp = 1.0
        self.exp = 0
        self.attack_lock_ms = -10_000
        self.attack_key_at = -10_000.0  # when the key of the last attack the game took went out (clock time)
        self.invulnerable_ms = -10_000
        self.inventory_full = inventory_full
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
        self.portals = [pt for pt in model.portals if pt.kind == PORTAL_EXIT]
        self.sides = _sides(model, self.realism.sides)
        self.arms = self.realism.rope_reach - JUMP_HEIGHT  # she grabs a rope this far above her feet (ARMS)
        self.dead = False
        self.died_at: float | None = None
        self._blocked = {pid: (_wall_at(model, q, q.left), _wall_at(model, q, q.right))
                         for pid, q in enumerate(model.platforms)}
        self._near = {q.id: [o.id for o in model.platforms if o.right >= q.left - 60 and o.left <= q.right + 60
                             and -70 <= o.top - q.top <= 70] for q in model.platforms}
        if hanging is not None:
            self._grab(min(model.ropes, key=lambda r: abs(r.x - start[1])))
            self.y = min(max(hanging, self.rope.top), self.rope.bottom)
            self.log.clear()
        self.monsters: list[SimMonster] = []
        self.by_platform: dict[int, list[SimMonster]] = {}
        self._spawn(monsters)
        self.drops: list[Drop] = [Drop(kind, x, model.platform(pid).y_at(x), pid, 0.0) for kind, pid, x in drops]
        self.rival: Rival | None = None
        if self.realism.rival_every > 0 and self.monsters:
            self.rival = self._rival_start()
            self._push(self._rival_due(), "rival")
            self._push(self._rival_stays(), "rival_moves")
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
        bodies, speeds = monster_data.bodies(self.model), monster_data.speeds(self.model)
        ids = {kind_of(s.name): s.mob_id for s in self.model.spawns if s.name}  # by kind ...
        ids.update({s.name: s.mob_id for s in self.model.spawns if s.name})      # ... and by name
        for i, (name, pid, x) in enumerate(wanted):
            kind = kind_of(name)
            mob_id = ids.get(name, ids.get(kind))
            info = None if mob_id is None else mob_info(int(mob_id))
            half, height = _body_of(kind, bodies.get(kind), info)
            hits = float(hits_of(kind, info))
            if self.character.spread and info is not None and info.hp > 0:  # in average stars, not whole ones
                hits = info.hp * (info.pdd + 100) / 100 / STAR_MEAN
            m = SimMonster(i, name, kind, self.model.platform(pid), float(x), random.Random(self.rng.random()),
                           speeds.get(kind, monster_data.DEFAULT_SPEED), half, height, hits, hits)
            if kind in bodies:
                m.flinch_top = bodies[kind].flinch_top
                m.fade = bodies[kind].die_s or FADE
            m.info = info
            m.start(0.0, float(x))
            self.monsters.append(m)
            self.by_platform.setdefault(pid, []).append(m)

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

    def withdraw(self, tag: str) -> None:
        """Forget the key events of plan `tag` the world hasn't taken yet."""
        self._events = [v for v in self._events if not (v[2].tag == tag and v[2].kind in ("down", "up", "release"))]

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
        if self.sitting:
            self.hp = min(1.0, self.hp + CHAIR_HP_PER_S * dt)
        if self.character.mp > 0 and self.mp < 1.0:
            self.mp = min(1.0, self.mp + MP_REGEN * dt / self.character.mp)
        self.ms += TICK_MS

    def _through_portal(self) -> bool:
        """Up held: in an exit's box (mapdata.PORTAL_BOX) while she isn't on a rope, sitting or attacking, she goes
        through it and leaves the map (SimLeftMap), on the ground or in the air (HeavenClient's Stage::update)."""
        if self.left_map_at is not None or self.mode == "rope" or self.sitting or self.ms < self.attack_lock_ms:
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
        elif e.kind == "loot":
            self.drops += e.data
        elif e.kind == "respawn":
            m = e.data
            m.alive, m.hp, m.died_at, m.hit_at, m.flinch_at = True, m.hp_max, None, None, None
            m.life += 1
            m.born = self.t
            m.start(self.t, m.home)
            self._camps_watch(m)
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
        """The game sees an arrow go down (she turns that way, and it stands her up from the chair) or up. In the air
        the step it turned her at, and the way she faced before, are kept (AIR_TURN_SAME_STEP)."""
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
        self.facing = side
        if self.sitting:
            self.sitting = False
            self.still_ms = self.ms + int(STAND_UP * 1000)

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
            if ground and self.sitting:
                self.sitting = False
                self.still_ms = self.ms + int(STAND_UP * 1000)
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
            if self.ms < self.attack_lock_ms or self.mode == "rope":
                return
            skill = action == "skill" and self.character.skill is not None
            if skill and self.character.mp > 0 and not self._cast():
                return  # too little MP: nothing thrown, no lock
            self.attack_lock_ms = self.ms + int(ATTACK_LOCK * 1000)
            self.attack_key_at = at
            self.sitting = False
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
        elif action == "chair":
            if ground and abs(self.vx) < 1:
                self.sitting = not self.sitting
                self.note("sits on the chair" if self.sitting else "stands up from the chair")

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
                lag += (2 * self.chance("walk") - 1) * self.noise.walk_jitter / WALK_SPEED
            self._seen_later(action, False, at, lag)

    # --- her motion -------------------------------------------------------------------------------------

    def _ground(self, dt: float) -> None:
        p = self.plat
        if "up" in self.held and self._grab_from_ground():
            return
        if "down" in self.held and self._grab_from_top():
            return
        if self.ms == self.step_at(self.ms):  # the game's step: her speed for the step to come (body.replay)
            lying = "down" in self.held or self.sitting
            d = 0 if lying or self.ms < self.still_ms else self.arrow()
            self.step_arrow = d
            if not d and not self.vx:
                self.glide = 0.0
                return
            self.vx, dx = body.ground_step(self.vx, d, SHOE, STEP_MS / 1000)
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
        with the map's sides stopping her like walls (the planner never jumps at them), and the walls of
        her foothold group."""
        walls = self._walls.get(self.group)
        if walls is None:
            walls = self._walls[self.group] = body.vertical_walls(_walls_for(self.model, self.group,
                                                                              self.realism.zero_walls))
        return body.fly(self.model, x, y, vx, vy, shoe=SHOE, arrow=arrow, skip=skip, walls=walls,
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
        self._fly(self._flight(self.x, y, vx, vy, d), from_ms, d)

    def _fly(self, f: body.Flight, from_ms: int, arrow: int, skip: Platform | None = None) -> None:
        self.mode, self.plat, self.rope, self.flight, self.flight_ms = "air", None, None, f, from_ms
        self.flight_arrow, self.flight_skip = arrow, skip
        self.vx, self.vy = f.speeds[0]
        self.sitting = False

    def _liftoff(self, pressed_on: str) -> None:
        """The game takes a jump key pressed while she was `pressed_on` ground, in the air or on a rope:
        on the ground it jumps her, with the kick toward the arrow it sees (body.takeoff); on the rope it
        was pressed on, with an arrow, it jumps her off sideways. Anywhere else it does nothing."""
        d = self.arrow()
        if self.mode == "ground" and pressed_on != "rope":
            self.knocked = False
            self._launch(body.takeoff(self.vx, d, SHOE), -SHOE.vjump, self.ms)
        elif self.mode == "rope" and pressed_on == "rope" and d:
            self._leave_rope()
            self._launch(d * ROPE_JUMP[0], -ROPE_JUMP[1], self.ms)

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
        has 7.6-11.7 (research: 60 fps recordings). Between the game's steps she stands where she lands until
        the next (body.LANDING_ENDS_STEP)."""
        self.mode, self.plat, self.flight, self.last_plat, self.group, self.knocked = "ground", q, None, q, q.group, False
        self.x, self.y = x, q.y_at(x)
        start, end = self.attack_ms
        self.vx, self.vy = (0.0 if start <= self.ms < end else vx), 0.0
        self.glide = 0.0 if body.LANDING_ENDS_STEP else self.vx * TICK_MS / 1000  # until the game's next step

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
        self.sitting = False
        self.note(f"grabs the rope at x {r.x:.0f}")

    def _leave_rope(self) -> None:
        self.left_rope_ms = self.ms
        self.rope = None

    def _climb(self, dt: float) -> None:
        r = self.rope
        if "up" in self.held and "down" not in self.held:
            self.y -= CLIMB_UP * dt
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
                    self.note(f"climbs onto platform {top.id}")
        elif "down" in self.held and "up" not in self.held:
            self.y += CLIMB_DOWN * dt
            if self.y >= r.bottom:
                self.y = r.bottom
                self._leave_rope()
                self._launch(0.0, 0.0, self.ms + TICK_MS)  # falling from the end of this tick

    # --- monsters and her -------------------------------------------------------------------------------

    def _touch(self) -> None:
        if self.ms < self.invulnerable_ms or not self.monsters or self.dead:
            return
        t = self.t
        half, tall = self.realism.her_box or (HER_HALF, HER_HEIGHT)
        if self.mode == "ground":
            pids = self._near[self.plat.id]
        else:
            pids = [q.id for q in self.model.platforms if q.covers(self.x, 60) and -70 <= q.y_at(self.x) - self.y <= 70]
        for pid in pids:
            for m in self.by_platform.get(pid, ()):
                if not m.alive:
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
        self.invulnerable_ms = self.ms + int(INVULNERABLE * 1000)
        evaded = self.chance("evade") < evade_chance(m.info, self.character)
        damage = 0.0 if evaded else touch_damage(m.info, self.chance("damage"), self.character)
        self.touch_log.append((self.t, damage))
        if m.born is not None and self.t - m.born < FRESH_S:
            self.fresh_log.append((self.t, damage))
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
        if self.mode == "rope":
            self._leave_rope()
        if self.realism.knock_turns:
            self.turn(LEFT if away > 0 else RIGHT, f"knocked by {m.kind} #{m.idx}")
        self.knocked = True
        self._launch(away * KNOCK_VX, -KNOCK_VY, self.ms + TICK_MS)  # after this tick's motion

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
        search from her feet now (reach.air_takes), whatever the terrain (_aim checks the line to it)."""
        t, d, best = self.t, directions.sign(facing), None
        for m in self.monsters:
            if not m.alive:
                continue
            mx = m.at(t)
            ahead, dy = (mx - self.x) * d, m.y(mx) - self.y
            if not reach.air_takes(ahead, dy, self.top(m), m.half):
                continue
            if best is None or abs(ahead) < best[0]:
                best = (abs(ahead), m)
        return None if best is None else best[1]

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

    def search_pick(self, facing: str) -> SimMonster | None:
        """The live monster the star's search from the ground picks on her `facing` side, whatever the
        terrain (_aim checks the line to it): of those whose body overlaps one of its bands
        (reach.search_band), the first band's; of those, one drawn at random (seeded, per attack), as the
        client takes the one that entered the map first, which nothing tells."""
        t, d, band, first = self.t, directions.sign(facing), None, []
        for m in self.monsters:
            if not m.alive:
                continue
            mx = m.at(t)
            ahead = (mx - self.x) * d
            k = reach.search_band(*reach.body_box(ahead, m.y(mx) - self.y, self.top(m), m.half))
            if k is None:
                continue
            if band is None or k < band:
                band, first = k, [m]
            elif k == band:
                first.append(m)
        if len(first) > 1:
            return first[int(self.chance("band") * len(first))]
        return first[0] if first else None

    def sight_stop(self, m: SimMonster):
        """What stops the star's line from her now to monster `m` (reach.wall_between, by reach.LINE: from the
        foothold group of the platform she last stood on, her feet where they are now, in a jump too; m's body
        as the attack picks it, World.top), or None."""
        mx = m.at(self.t)
        return reach.wall_between(self.last_plat, self.x, self.y, mx, target=m.platform, ty=m.y(mx),
                                  body=(self.top(m), m.half))

    def _aim(self, key_ms: int, skill: bool = False, tag: str = "", step: int | None = None) -> None:
        """The attack keyed at `key_ms` starts: it picks its target from where she is now, and its damage
        lands STAR_DELAY after the key (_attack). On the ground it takes one of the claw's stances (drawn,
        seeded: reach.CLAW's), and the claw swipes the nearest monster that stance's box reaches (swiped);
        else the star's search picks (search_pick). In the air, the nearest in the attack's area or its band's
        window of the search from her feet takes the star (AIR_PICK 'widening': air_pick), the nearest in the area
        alone ('area'), or as on the ground from where she is then ('v83'). Each picks by the bodies
        alone; then, as the client does, the line to a star's pick alone is checked (sight_stop): stopped, the
        attack hits nothing, and no other monster takes it. In the air an arrow acted on at its key's `step` turned
        her too late for it (AIR_TURN_SAME_STEP): it goes the way she faced before."""
        self.attack_ms = (self.ms, key_ms + int(ATTACK_LOCK * 1000))  # its animation plays until the lock ends
        facing = self.facing
        if (AIR_TURN_SAME_STEP and step is not None and self._air_turn is not None and self._air_turn[0] == step
                and self.mode == "air"):
            facing = self._air_turn[1] or facing
            self.note(f"turned {self.facing} at the step of the attack's key: it goes {facing}")
        due = [way for when, way in self.meant if when <= self.attack_key_at + 1e-6]
        if due:  # (the latest meant by then: one whose key the game didn't take is passed over)
            self.meant = [(when, way) for when, way in self.meant if when > self.attack_key_at + 1e-6]
            self.attacks_meant += 1
            if facing != due[-1]:
                self.wrong_way += 1
                self.note(f"attack meant {due[-1]} goes {facing}")
        if self.mode == "ground" or (AIR_PICK == "v83" and self.mode == "air"):
            stances = reach.CLAW.stances
            target = self.swiped(facing, stances[int(self.chance("stance") * len(stances))])
            swipe = target is not None
            if target is None:
                target = self.search_pick(facing)
        else:
            target = (self.air_pick(facing) if AIR_PICK == "widening"
                      else self.nearest_in(facing, reach.STAR_RANGE, reach.AREA_DOWN, reach.AREA_UP))
            swipe = False
            if target is not None and target.kind == "snail" and self.noise.air_snail:
                near, take = self.noise.air_snail
                if abs(target.at(self.t) - self.x) < near and self.chance("air_snail", key_ms) >= take:
                    target = None  # over it (Noise.air_snail): the star hits nothing
        stop = None if target is None or swipe else self.sight_stop(target)
        if stop is not None:
            target, stop = None, (target, stop)
        rise = self.last_plat.y_at(self.last_plat.clamp(self.x)) - self.y  # her height above where she last stood
        picked = target if stop is None else stop[0]
        up = None if picked is None else self.y - picked.y(picked.at(self.t))  # its feet above hers, + above
        how = (key_ms / 1000, tag.rsplit("_", 1)[0], self.mode, round(rise, 1),
               None if picked is None else picked.kind, None if up is None else round(up, 1))
        self._push(key_ms + int(STAR_DELAY * 1000), "star", (facing, target, swipe, stop, skill, how))

    def _attack(self, facing: str, m: SimMonster | None, swipe: bool, stop: tuple | None = None,
                skill: bool = False, how: tuple | None = None) -> None:
        """The attack's damage, STAR_DELAY after its key: on the monster it picked when it started (_aim),
        a claw swipe or a star; none if what stops a star stopped it (`stop`: the monster picked, and the
        foothold). `skill`: Lucky Seven's (Her.skill), its stars' damage on a swipe's pick too. `how`: what _aim
        knew of it, for attack_log."""
        if stop is not None:
            picked, f = stop
            where = f"the wall at x {f.x1:.0f}" if f.x1 == f.x2 else f"the floor at y {f.y1:.0f}"
            self.note(f"star thrown {facing} at {picked.kind} #{picked.idx}: stopped by {where}")
            self._logged(how, "stopped")
            return
        if m is None or not m.alive:
            self.note(f"star thrown {facing}: no hit")
            self._logged(how, "nothing")
            return
        mx = m.at(self.t)
        dx = mx - self.x
        what = "claw swipe" if swipe else "star"
        if not swipe and self.noise.miss and self.chance("miss") < self.noise.miss:
            self.note(f"star misses {m.kind} #{m.idx} at {dx:+.0f}")
            self._logged(how, "miss")
            return
        her = self.character
        stars = her.skill if skill else her.stars
        if her.spread:
            damage = self._damage(stars, swipe and stars == 1.0)
        else:
            damage = stars if stars != 1.0 else CLAW if swipe else 1.0
        m.hp = round(m.hp - damage, 6)
        self._logged(how, "swipe" if swipe and m.hp > 1e-9 else "hit" if m.hp > 1e-9 else "kill")
        if m.hp > 1e-9:
            m.hit_at = self.t
            if self._pushes(m, swipe):
                m.flinch_at = self.t
                if MOB_KNOCKBACK:
                    m.knock(self.t, 1 if mx >= self.x else -1)
                self.note(f"{what} hits {m.kind} #{m.idx} at {dx:+.0f} ({m.hp:g}/{m.hp_max:g} left)")
            else:
                self.note(f"{what} hits {m.kind} #{m.idx} at {dx:+.0f} ({m.hp:g}/{m.hp_max:g} left): not pushed")
            return
        self._kill(m, mx, what)

    def _logged(self, how: tuple | None, outcome: str) -> None:
        """An attack's outcome into attack_log (`how`: _aim's (key time, plan, ground or air, her rise, the kind picked,
        how far above her feet it was))."""
        if how is not None:
            t, plan, mode, rise, kind, up = how
            self.attack_log.append((round(t, 3), plan, mode, rise, kind, outcome, up))

    def _damage(self, stars: float, swipe: bool) -> float:
        """An attack's damage drawn (Her.spread), in average stars (STAR_MEAN): a claw swipe's (CLAW_DAMAGE), one star's
        (STAR_DAMAGE), or SKILL_STARS stars of `stars` / SKILL_STARS of one each; every star a whole number from its
        least to its most, as likely each, seeded by the tick."""
        her = self.character
        lo, hi = CLAW_DAMAGE if swipe else (her.star or STAR_DAMAGE)
        n, each = (1, 1.0) if swipe or stars == 1.0 else (SKILL_STARS, stars / SKILL_STARS)
        if not swipe and stars != 1.0 and her.skill_star is not None:
            (lo, hi), each = her.skill_star, 1.0
        lo, hi = round(lo * each), round(hi * each)
        total = sum(lo + int(self.chance(f"damage:{k}") * (hi - lo + 1)) for k in range(n))
        return total / STAR_MEAN

    def _cast(self) -> bool:
        """A Lucky Seven cast's MP (CAST_MP) taken, if she has it (Her.mp): then her pet may drink for her (Her.pet_mp).
        False with too little: nothing is thrown."""
        her = self.character
        if self.mp * her.mp < her.cast_mp - 1e-9:
            self.no_mp += 1
            self.note(f"Lucky Seven with {self.mp * her.mp:.0f} MP: not enough, nothing thrown")
            return False
        self.mp -= her.cast_mp / her.mp
        self.casts += 1
        self.mp_used += her.cast_mp
        if her.pet_mp is not None and self.mp < her.pet_mp and self.pet_fullness() > pet_care.PACKED_AT:
            self.mp = min(1.0, self.mp + her.mp_potion / her.mp)
            self.mp_potions += 1
            self.note(f"her pet drinks an MP potion for her: MP {self.mp:.1%}")
        return True

    def _pushes(self, m: SimMonster, swipe: bool) -> bool:
        """Whether this hit pushes monster `m` back (PUSH_BY_DAMAGE: drawn by push_chance, seeded by the tick)."""
        if not PUSH_BY_DAMAGE or m.info is None or m.info.pushed <= 1:
            return True
        p = push_chance(m.info.pushed, m.info.pdd, swipe)
        return p >= 1.0 or (p > 0.0 and self.chance(f"push:{m.idx}") < p)

    def _kill(self, m: SimMonster, mx: float, what: str) -> None:
        t = self.t
        m.alive, m.died_at, m.died_x = False, t, mx
        self.kills[m.kind] = self.kills.get(m.kind, 0) + 1
        self.kill_log.append((t, m.kind, m.platform.id, "her"))
        self.death_log.append((t, m.kind, m.platform.id, mx))
        self.exp += self.model.mob_exp.get(m.name, 1)
        self.note(f"{what} kills {m.kind} #{m.idx} at x {mx:.0f} (platform {m.platform.id})")
        p = m.platform
        drops = []
        rng = self._drop_rng if self._drop_table else self.rng
        for kind in self._drop_kinds(m.kind, rng):
            x = min(max(mx + rng.uniform(-10, 10), p.left + 3), p.right - 3)
            drops.append(Drop(kind, x, p.y_at(x), p.id, t + DROP_DELAY))
        self._push(self.ms + int(DROP_DELAY * 1000), "loot", drops)
        self._push(self.ms + int(self.realism.respawn * 1000), "respawn", m)

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
        m.alive, m.died_at, m.died_x = False, t, mx
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
        near = sorted((d for d in self.drops if not d.gone and d.since <= t < d.since + DROP_LIFE
                       and d.platform == self.plat.id and abs(d.x - self.x) <= PICK_REACH
                       and (d.owner == "her" or t >= d.since + self.realism.rival_owns)),
                      key=lambda d: (abs(d.x - self.x), d.x))
        for d in near:
            if d.mesos or not self.inventory_full:
                d.gone, d.taken_at, d.taken_by = True, t, "her"
                self.picked[d.kind] = self.picked.get(d.kind, 0) + 1
                self.messages.append((t, "You have gained mesos (+15)" if d.mesos else
                                      f"{d.kind.replace('-', ' ').title()} x1 earned. "
                                      + ("(Use)" if d.kind.endswith("-potion") else "(Etc)")))
                self.note(f"picks up {d.kind} at x {d.x:.0f}")
                return
        if near:
            self.messages.append((t, FULL_MESSAGE))
            self.note("can't pick up: inventory full")

    # --- what can be seen -------------------------------------------------------------------------------

    def her(self) -> tuple[float, float]:
        return self.x, self.y

    def live_drops(self) -> list[Drop]:
        """The drops lying on the ground now (landed, not taken, not vanished yet)."""
        t = self.t
        return [d for d in self.drops if not d.gone and d.since <= t < d.since + DROP_LIFE]

    def visible_monsters(self) -> list[tuple[SimMonster, float, float, bool]]:
        """Every monster that shows: (monster, x, y, alive); killed ones fade out where they died."""
        t, out = self.t, []
        for m in self.monsters:
            if m.alive:
                x = m.at(t)
                out.append((m, x, m.y(x), True))
            elif m.died_at is not None and t - m.died_at < m.fade:
                out.append((m, m.died_x, m.y(m.died_x), False))
        return out

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


def _actions(keys: GameKeys) -> dict[str, str]:
    """Key name -> what it does."""
    out = {}
    for field_name, action in (("left", "left"), ("right", "right"), ("up", "up"), ("down", "down"),
                               ("jump", "jump"), ("attack", "attack"), ("lucky_seven", "skill"),
                               ("pick_up", "pick"), ("chair", "chair"), ("hp_potion", "potion"),
                               ("pet_food", "pet_food")):
        k = getattr(keys, field_name, None)
        if k is not None:
            out[k.name] = action
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


def _walls_for(model: MapModel, group, zero: bool = False) -> list:
    """The walls that stop her with foothold group `group` (body.walls_for), and with `zero` (Realism.zero_walls) those
    of every layer's group 0 too, whatever her group: off a rope too (body.ROPE_GROUP, where none stops her)."""
    walls = body.walls_for(model, group)
    if zero and group != body.EVERY_WALL and body.OWN_GROUP_WALLS:
        walls = walls + [w for w in model.walls if w.group is not None and w.group[1] == 0 and w.group != group]
    return walls


def _wall_at(model: MapModel, p: Platform, edge: float) -> bool:
    """Whether a wall of its foothold group rises at this end of platform `p` (it stops her there)."""
    level = p.y_at(edge)
    return any(w.x1 == w.x2 and abs(w.x1 - edge) < 1 and min(w.y1, w.y2) < level - 2 <= max(w.y1, w.y2)
               for w in body.walls_for(model, p.group))


def exp_pixels(exp: int) -> np.ndarray:
    """Her EXP number's text pixels (hud.EXP_NUMBER): a pattern that changes whenever her EXP does."""
    return np.random.default_rng(exp).random((26, 140)) > 0.5
