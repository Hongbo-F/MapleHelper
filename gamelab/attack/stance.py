"""What to do about the monsters around her: throw, jump-throw, jump back and throw, step away, or
close in. Pure decisions from map positions; the fight loop (behavior.fight) carries them out.

`stance` is the policy for a group (fight.clear, what tours use) and `decide` the one for a single
target (fight.kill, `hunt kill`). Both take a monster on her platform as too close while a throw from the
ground at it would be a claw swipe, by its size (`close_to`, reach.claw_edge: a pig within 111 units, a
snail within 96), decide at least within TOO_CLOSE; stance closes in to SHOT_RANGE - 40 of a target where
decide walks to the nearest of SHOT_RANGE - 30 and 20 past the claw's edge from it (off her level, a band
past reach.shot_dx), either side.
`pick_target` says which monster to shoot first, `open_above` whether a jump keeps her on her
platform, and `act_text` and `pct` word it for the log. Thresholds built on reach's constants are
read from reach at call time, and so are the attacks' timings (physics.JUMP_ATTACK_MS and
JUMP_BACK_ATTACK_MS): whether a jump throw or a jump back can hit a kind follows from them.
Toward an end of her platform with a long drop under it (drop_limits: the next platform LONG_DROP or more down, or
none, and no wall of her foothold group there), her steps away and jumps back keep DROP_ROOM from it (`edges`), and
under a platform overhead an escape may be the jump up onto it (`up`: overhead), by the danger there.
Of the monsters in reach from the ground, one on her platform whose kind counts as a danger is thrown at first.
A monster whose star terrain stops (reach.wall_to: a wall of her own foothold group, as for one on
platform 2 for her on 4, from anywhere on her platform; or a floor, her own for one below her unless she
stands near enough its end, as the user saw in game on 2026-10-21) is out of reach from there: neither
policy throws at it, or walks along her platform for it but to where its line is clear (_clear_from). So
is one behind a monster the attack takes first whose star is stopped (reach.shielded: the client checks the
line to its pick alone), so all of them go by every monster around her (`around`).
Off her level, how much room a star needs sideways goes by the monster's kind (reach.shot_dx: the star's
search picks a small or a short one from farther off).
A snail too close for a star from the ground is never jump-thrown at (JUMP_THROW_SNAILS): she steps back to
where a throw from the ground is a star if her platform has the room, else stabs it (not to a short platform's
far end: SNAIL_FAR_END).
With the map's danger (attack.danger), the group policy lets a monster of a kind that hits hard come only as near
as its danger warrants (Danger.keep_off: a Horny Mushroom 169 units on Tree Dungeon; no kind on the first map, and
on West Rocky Mountain II an Axe Stump's 87 is inside the claw's reach, 110), and then does whichever of a jump back, a
step away and a throw from where she is leaves her the least danger (_escape: the monsters around her and the spawn
points they don't account for; a jump back toward another one, or onto a spawn point, isn't an escape, nor a move
into or through a monster's body), a step only if it's clearly safer (ESCAPE_THROW: it doesn't throw). From
where she is it's a throw from the ground, a jump throw only where that would be a claw swipe: a jump in place by one
walks it into her as she lands.
From 2026-10-27 (clearance kept ahead of time: the block of constants before JumpPlan) who's too close, and what each
throw would take, go by where each monster will be when the attack starts (`Pace`: her lock, the look's age, its walk),
from a jump by the star's search alone (reach.AIR_MODEL), on the slope as it is (reach.air_sure): up close the throw
from where she stands is the one whose attack surely takes the monster (`_from_here`: from the ground past the claw's
reach; a jump in place, keyed 40 ms into the jump if she faces it, else turning in the air where that picks it, else
tapping the arrow on the ground first), else a step to where the next one will (`_clear_step`), else a jump back, else
the claw from the ground within CLAW_ONLY, the jump in place beyond it.
With what her jumps' attacks reach (`jumps`: attack.reach.coverage), nothing in reach from the ground has her
jump-throw at a monster on a platform above where an attack from a jump surely picks it (_jump_shot), in place,
forward or back, its attack key as soon in the jump as that pick is sure (JumpPlan); and on her way somewhere
(`toward`), a jump forward along it that hits one goes before a throw from the ground (PASSING_JUMPS).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

from maplehelper.engine import LEFT, RIGHT
from maplehelper.game import MOVEMENT

from .. import directions
from ..localization.mapdata import MapModel, MobBody, Platform
from ..localization.monsters import DEFAULT_SPEED, MONSTER_SPEED, MONSTER_TOP, Monster, touch_box
from ..motion import physics
from ..motion import bodies as her_body
from ..motion.bodies import HER_HALF
from ..motion.physics import JUMP_HEIGHT, WALK_SPEED
from . import reach
from .danger import KEEP_DANGER, Danger


# Distances in map units.
TOO_CLOSE = 110    # a monster on her platform nearer than this is about to touch her: step away first
STEP_AWAY = 120    # how far she steps away from a monster too close
STEP_ROOM = 40     # the least room worth stepping away into (with less she fights from where she is)
KNOCKBACK_ROOM = 25  # extra room kept from her platform's ends while monsters share it: standing ~100
                     # units from an end of platform 12, pigs walking into her knocked her off it
MAX_WALK = 300     # longest walk between two looks
MIN_MOVE = 15      # stance: a shorter step or approach isn't worth it
CLEAR_STEP = 10    # where a star's line is stopped from where she'd go, the places along her platform tried
                   # instead, this far apart (by the v83 line a floor stops it from part of her platform only)
SHOOT_SLACK = 6.0  # a walk's place to shoot from (_reached_from) only where her attack takes it this far either way
                   # of it too: the walk is open loop (Actions.walk holds the key for its time), live (2026-10-28,
                   # 86 steps) they ended a median 3 off their goal, half of them short by 1 or more, and the harness's
                   # walk jitter is 6 either way; the nearest place in reach lay on its edge (a median 6.5 inside it on
                   # Tree Dungeon's trip-ending legs, 11% under 3), and 5 of 24 simulated walks to it ended out of reach
JUMP_BACK_ROOM = 110  # room she needs behind her (from her platform's end, less the margin) to jump back
JUMP_BACK_CLEAR = 160  # ... with no monster on her platform this close on that side
JUMP_BACK_SPAN = 95  # how far a jump back carries her (by the engine 87-97 units with the lags spread as measured,
                     # 94 at the median: control.actions.JB_RUN_MS)
# Escaping a monster that hits hard (_escape), a step away (STEP_AWAY) is taken over a jump back or a throw from where
# she is only if it leaves her this much less danger (attack.danger: the share of her HP touches would take): they
# throw, the step doesn't (0.8 s without one live). A guess, at what the throw buys: two in three of a Horny
# Mushroom's stars push it back (sim.world.PUSH_BY_DAMAGE), and 40 units more between them is 0.01-0.02 less danger
# 150-200 off. By danger alone the step always won (it goes 120 to the jump's 95: a lone Horny 150 off, 0.042 against
# 0.049), and she never threw.
ESCAPE_THROW = 0.03
# Whether she jump-throws at a monster too close for a star from the ground that's too short for a jump back's
# attack to pick (a snail: reach.air_hits). By the attack's area keyed 40 ms after the jump (physics.JUMP_ATTACK_MS)
# it's picked, and in game on 2026-10-20 one such star killed a snail. But the user, watching her hunt on
# 2026-10-21, saw jump throws mostly miss snails, "especially on short platform[s]", and the day's telemetry says
# so: of 26 jump throws at a snail under 46 units off on her platform (in reach of a stab from the ground, out of
# the star's search), 5 hurt or killed it; at 46-84 off 18 of 33, at 85-130 6 of 10. RustMS's port of the v83
# client, from as high as she is then (21-26 up), picks no snail 35-85 off. False: never; she steps back to where
# a throw from the ground is a star, if her platform has the room, else stabs it from the ground (the claw, weak
# but always hitting: 5 swipes kill one). `python -m gamelab probe run air` weighs the two pictures of the attack
# from a jump.
JUMP_THROW_SNAILS = False
SMALL_STEP_REACT_S = 0.45  # s from the end of a step back to her next attack's pick (the slide, a look, the key's
                           # lag): a monster walking at her closes in that much more (a snail 43.75 units/s: 20)
# Whether that step back from a snail may go past the 50 she keeps from her platform's ends with monsters about
# (stand limits), to SMALL_STEP_END from an end: on the 168-unit platforms 3, 20, 22, 31 and 33, to the far end. In
# the simulator it lost to stabbing the snail where she stands. With preset game's noise, one snail too close on
# her platform and a fight to its kill (every 20 units along those five platforms, the snail 20-80 units off either
# side, seeds 1-8: 1,120 starts), of the 187 starts where it took her to the far end she was knocked off her platform
# 27 times against 9, and killed the snail in 6.5 s on average (20 for none in 20 s) against 5.7 (within 4 s 25%
# against 16%, within 10 s 94% against 97%): on a short platform the snail is soon back at her, and there a touch
# knocks her off the end. Over all 1,120: 4.6 s against 4.5, knocked off 75 times against 57. On the long platforms
# 1 and 34 (432 starts, seeds 1-3) a step back within her room killed it in 3.0 s against 3.3 for stabbing first,
# with half the touches (183 against 331), so she still steps back where the room is. Not measured in game. False:
# on a short platform she stabs it.
SNAIL_FAR_END = False
SMALL_STEP_END = 40  # ... and how near her platform's end that step may take her then: not to EDGE_MARGIN's 25, the
                     # step's walk is planned before her last attack's lock runs out, and a touch meanwhile knocked
                     # her 40 units toward the end and a step of 110 took her off 3 (nav_fight_up, seed 1). Of that
                     # scenario's seeds 1-48, 36 arrived with 25, 38 with 35, 41 with 40 and with 50, with the step
                     # to the far end on; with it off (SNAIL_FAR_END False) this isn't used: 37 arrive with any

_THROW_VERBS = {"throw": "throwing", "jump_throw": "jump-throwing", "jump_back": "jumping back, throwing"}
# A jump throw at what no throw from the ground reaches from where she stands (_jump_shot: a monster on a platform above
# her, up close only from high in the jump, farther off from lower: attack.reach.shots, the windows widening with the
# distance): she takes one if it lands her back on her platform, where she keeps to, however a jump forward or back goes
# (lifting off 12 units early or late, 10% slower, the kick short at 126 or 119: physics.jump_variants, as a route
# jump's landing is checked), her body meeting no monster's on the way (motion.bodies, each where it can be by then:
# MonsterTrack.reach's spread, at its speed; in place, the monsters off her platform: right under one above, her box
# meets it), and, forward or back, where the danger is at most JUMP_DANGER more than where she stands (attack.danger: a
# guess, a tenth of a Horny Mushroom's touch, 16% of her HP). The choice among them (the choice model): every one of
# these throws once, and a throw's lock (0.73 s, the game's: COMBAT.attack_lock_ms) is most of what it costs, the jump's
# flight (0.62 s) inside it; so what decides is what it gains: a throw from the ground when anything's in reach from
# there (no jump's lock extra, JUMP_THROW_LOCK_EXTRA_MS), else the jump whose attack comes soonest at the target
# pick_target's order takes first (hurt first, then the nearest), in place before forward before back (the least ground
# covered: the least danger), forward along the way she's going first; travelling, a jump forward along her way that
# hits one over a throw from the ground (PASSING_JUMPS: the throw and 84 units of her way, 0.56 s of her walk, in one
# lock).
JUMP_DANGER = 0.016
ARC_STEP = 0.02  # s between the places along a jump her body is checked at (as motion.flow's danger, DANGER_STEP)
# In passing (stance's `toward`: travelling that way), a jump forward that way whose attack hits a monster takes her
# on as she shoots (a jump's 83 units, 0.56 s of her walk, for an attack 40 ms later and a lock 0.1 s longer than a
# throw's from the ground): taken over a throw from the ground at a monster that way, on the same terms as above, and
# only if it lands her no more than PASS_SLACK past where she's going on her platform (the objective's place there, or
# where her route leaves it: behavior.hunt's _toward): a rope's foot or a jump's takeoff 15-80 ahead, overshot, has her
# walk back (review, 2026-10-25), which a throw from the ground doesn't.
PASSING_JUMPS = True
PASS_SLACK = MIN_MOVE  # units: a shorter walk back isn't worth a step (stance's own MIN_MOVE)
# Toward an end of her platform with a long drop under it, her steps away and jumps back from monsters on it keep this
# far from it (`edges`: drop_limits). A touch's knock takes her 53-79 units: live on 2026-10-25 on Tree Dungeon (the
# perception service's 10 fps records, D:/MapleHelper-scratch/analysis-1025/damage-skeptic knock.py) 53, 55, 58, 63, 66
# and 67 on flat ground over 0.8 s, and three knocks took her off 23's left end from 52, 53 and 79 units in, 389 units
# down to 29 and 43 s back up (with 2 touches on 29), each after she had stepped away left (53, 120, then 120 and 55
# units: the step's limit was EDGE_MARGIN and KNOCKBACK_ROOM, 50, from any end). An end's drop is long when the next
# platform under it is LONG_DROP or more down, or there's none, and no wall of her foothold group rises from it (it
# stops her: Platform.walls; 24's right end, the tower's wall): 23's left end (389 down to 29), not its right (37 to 24)
# nor 24's left (74 to 25: back up in 2-3.6 s live). Guesses past the measured knocks: 100 units, the farthest knock and
# a little.
DROP_ROOM = 100.0
LONG_DROP = 100.0
END_PAST = 5.0       # units past a platform's end where what's under it is looked for (end_drop)
# Under a platform overhead, the jump up onto it is an escape (_escape's `up`); off, as before. With DROP_ROOM, in the
# simulator (Tree Dungeon, crowd, zero-walls, near-misses, 8 seeds of 5 minutes): 0.92 bars of HP lost a minute against
# 0.97 without both, her knocks off a platform 7.5 a run against 8.1, 982 units down against 997.
ESCAPE_UP = True
# One in reach on her platform whose kind counts as a danger is thrown at first (_throw_side); off, as before. With
# spots.SPAWN_CLEAR, in the simulator as above: touches off her trips 7.85 a minute against 8.90 without both, 0.92 bars
# of HP lost a minute against 1.02.
TOUCHER_FIRST = True
# Her wounds first (the magician's: `hurt`, from 2026-10-29). Live that night 166 of 1,497 first casts at a whole monster
# left it alive with no second cast for 8 s (a Green Mushroom at about a tenth), and in the 3 s after, 58% of them she
# cast at another, 39% she was touched: the one she hurt walks at her, and inside her search's first 50 her cast can't
# pick it (D:/MapleHelper-scratch/castfollow). The user: "yes we should finish a monster, but we should also watch out for
# inmennent danger from a close by mob". So with `hurt` (behavior.attacks.Attacks.hurt: what a monster she has thrown at
# has left as she reckons it, alive), a throw goes the way of the most hurt in reach she casts at (_throw_side); and with
# one she hurt unseen this look (`hidden_hurt`: under her hit's effect or the feed's lost track, seen within
# WOUND_HIDDEN_S) in reach one way and none she casts at that way, she waits for it rather than turn to another. Unless a
# monster on her platform that can touch her would be too close for a throw from the ground by the attack after her
# next (`_imminent`: too_close's IMMINENT_S later): that one first where she casts at it, and no wait. Off: as before.
WOUNDS_FIRST = True
IMMINENT_S = 0.84        # s: her next throw's lock (attack.choice.THROW_S), so by the attack after it
WOUND_HIDDEN_S = 0.8     # s unseen that a wound of hers is waited for (her hit's effect over it: 570 ms)
# R1 of the cornered study (2026-11-03, D:/MapleHelper-scratch/cornered: the user's clip of 2026-11-02 22:51, "cornered
# character making bad attack move"): the looks are blind at her body (a monster seen 0-20 off her is seen in the next
# look 19-42% of the time live, against ~90% past 60), and a press from the ground goes to the claw whenever a monster's
# body is in its box in front of her, whatever she aimed at. Live in that hour 59 of 586 ground Lucky Sevens aimed 105
# or more off with nothing seen within 105 took no MP, and every one checked on the frames was the claw on a monster the
# looks missed 0-70 off (15-25 a swipe, no star), all but those at newborns fading in within 2 s of a touch the looks
# had no monster for (the MISS of 22:50:43.7 known at the look of 22:50:44.1, which changed nothing). So a touch of hers
# with no monster the look found about her that could have made it (`clawed`: behavior.fight.clawed, the navigator's R7
# stand-in's test, Navigator._toucher: TOUCHER_S, TOUCHER_NEAR; a stand-in of one the look didn't find, hidden or
# unseen, is a monster the looks miss) has a throw at one past the claw's edge go from a jump in place
# where its attack surely picks it (_from_here without the ground: reach.air_sure, CLEAR_MARGIN farther), the air having
# no claw: its star takes the one the looks miss if that one's body reaches the search's window (37 or more ahead, the
# 35-70 of the wall corner of 22:49:57), else the one she aimed at. With none sure, from the ground as before (the
# claw's swipe, if it's there). Off: as before.
CLAWED_AIR = True
# ... and up close (CLAWED_CLOSE, R1 extended, 2026-11-03): where the throw from where she stands at the one too close
# would be a jump in place that surely picks nothing (her jump's attack picks a Horny or Green Mushroom only from 54
# ahead: jump_reach; _close_quarters' last resort), a sure jump throw that way at one past it (_from_air: its star takes
# the one on her if its body reaches the search's window, else the one past it). In the simulator, its stand-ins among
# the monsters about her (localization.unseen.UNSEEN_LIVE), a Horny Mushroom on her at a wall end had her jump in place
# at it 13 times in 16 s for nothing, touched 8 times (the thief's standard command, seed 6:
# D:/MapleHelper-scratch/cornered-fix/r1/log_live_6.txt, 78-94 s). Off: the jump in place, as before.
CLAWED_CLOSE = True

# --- clearance, kept ahead of time (from 2026-10-27) -------------------------------------------------------------------
# The user, 2026-10-27, watching her hunt Tree Dungeon: "I frequently find the player standing at the same X pos as the
# mob, doing jump hit but couldn't hit the mob because no minimal clearance is maintained", and "jump attack is
# extremely hard to hit a mob on a down slop, even slightly". The live hunts of 2026-10-25 23:43 to 10-26 23:47 (14 on
# Tree Dungeon, 189 minutes and 6,209 attacks out of darts' spells: D:/MapleHelper-scratch/clear/diag-clearance) bear it
# out: from a jump the attack picks by the star's search alone (reach.AIR_MODEL), so a jump throw in place hurt a monster
# under 40 ahead 40% of the time (8% of the casts at an unhurt Green killed it under 40 ahead, 53% at 60-90), one turning
# her in the air (the attack 120 ms after the jump key, her feet 52-56 up) a third of the time at one level with her or
# lower, and from the ground the claw swipes one within its reach (a Lucky Seven press took MP 27% of the time under 90
# ahead, the rest swipes of 15-24). And she got that close: a monster on her platform within 60 of her 8.7 s a minute,
# 27% of her attacks, 30% of her touches; 41% of those episodes began with a landing, 11% with a monster walking in
# during her lock (Greens and Hornies walk 67-70% of the time at a median 95 units/s, and from the look her ground throw
# was decided on to its attack's start was 0.66 s: 86 of 632 ground throws decided as stars had the target under 93 at
# the pick, 51 of their 80 casts swipes). So her decisions go by where each monster will be when the attack starts
# (`Pace`: her lock and the look's age, the attack's own key, its walk by its track), a throw counts only where its
# attack surely picks the monster then (reach.air_sure from a jump, on the slope as it is; past the claw's edge from the
# ground), and up close she steps to where the next one will, or with no such step the claw swipes (CLAW_ONLY).
GROUND_PICK_S = physics.JUMP_LAG + physics.ATTACK_DECIDE_LAG_MS / 2000  # s from a throw's key to its attack's start
TAP_S = (MOVEMENT.tap_turn_ms + 30) / 1000  # a throw from the ground that taps the arrow first has its key this much
                                            # later (control.actions: the tap, and TURN_SETTLE_MS)
CLEAR_MARGIN = 10.0  # units: a jump throw in place counts only at a monster this much farther than where its attack
                     # surely picks it (reach.air_sure, AIR_MARGIN into the search's window): the tracks' error. Live the
                     # in-place throws without a turn hurt 40% of the time at 0-40 ahead, 79% at 40-60 (a mushroom is
                     # surely picked from 44), 81% at 60-130: 54 for a level mushroom
CLAW_ONLY = 15.0     # units, middle to middle: nearer than this (on her), with no step that clears it, she swipes it
                     # with the claw from the ground (15-24 a swipe, no MP), not a jump throw in place: by eye on the
                     # recordings of 2026-10-26 those hurt one on or under her 0 times in 9 (within 25), and by the
                     # service's tracks at the pick her casts in place killed one 0-10 ahead 1 time in 31 (3%), 10-20 2
                     # in 25, 20-30 5 in 32 (16%), 30-40 16 in 43 (37%, as at 40-130): a cast 20-30 ahead does about
                     # twice a swipe's damage on average, 0-10 about as much for its 14 MP. In the harness (Tree, crowd,
                     # 8 seeds of 5 minutes, the world as first fitted) 30 made 361.1 EXP a minute on the hunt loop
                     # where 15 made 370.6, 358.3 grinding where 347.8: the same within the seeds' noise
# Whether her attack has point-blank answers up close: a jump in place, a tap first, a jump back throwing, the claw from
# the ground (the thief's: `_from_here`, `_close_quarters`). Off for an attack whose search picks nothing within its
# first stretch and nothing in the air (the magician's Magic Claw: her profile's Policy.close_quarters 'touch', set by
# attack.magic for her process): up close she casts from the ground only where her search surely picks the monster when
# the cast starts (CLEAR_MARGIN past its start), else steps to where it will (_clear_step, judged from the ground
# alone), jumps back (the jump away alone: control.actions.Actions.ground_only), casts that way at another her search
# picks from where she stands, or waits ('wait'), never a cast that picks nothing. Live on 2026-10-29 her close
# quarters' jump throws went out as casts from the ground (Actions.ground_only): 106 of her 540 casts of 01:46-02:03,
# 62% of them showing nothing (D:/MapleHelper-scratch/mcasts).
POINT_BLANK = True
STEP_SETTLE_S = 0.25  # s after a step's walk before her next keys: the slide (STOP_RUN) and the look after it
# A monster walking at her stops where its body meets hers (its half and HER_HALF from her x), on the side of her it was
# seen on: it touches her there, and the touch's knock (about 58 units) leaves it on that side. Until the reviews of
# 2026-10-27 it walked on through her, and right after a throw (her lock still 0.7-0.8 s) a walker within about 85 in
# front had her tap round and throw at her empty back (86 such throws in its Tree goldens, none in master's). Replayed
# on the 839 live close-up decisions whose target's track at the pick is known (D:/MapleHelper-scratch/clear/fin/live/
# pace_stop.py, walks matched to the service's tracks record nearest the look as pace_of does), the 217 walking at her
# that reach her by the pick were a median 49 from her middle then, 207 on the side the look saw them: stopped so, 19.8
# off on average (walked through her 32.6, stopped at her x 29.5, where they were seen 27.1; all 839: 15.5, 18.8, 18.0,
# 20.1). False: as before.
PACE_STOP = True
# A monster whose walk nothing tells (no track, or one seen too briefly) stays where it was seen; True walks it at her at
# its kind's speed (Pace.speeds: the game data's, 100 units/s for the mushrooms; live a median 95), the worst case, as
# until the reviews of 2026-10-27: of the same decisions those 46 were 20.8 off on average where they were seen, 42.8
# walked at her, 28.2 walked at her and stopped where they meet her (fin/live/walk_at.py).
WALK_AT = False
TRACK_MATCH = 25.0   # units: a look's monster is its track's (Scene.tracks) within this of the track's x, its kind and
                     # platform the same (behavior.fight.pace_of)
TRACK_SEEN = 1       # ... and that track's speed counts once it was seen in this many frames: from its first, its
                     # speed the game data's the way it heads (localization.tracking, since 2026-10-30; 3 until
                     # then, its speed filtered)
# A monster her running attack was at stands from when its damage shows until FLINCH_HOLD_S after the key (Pace.hold):
# live, walkers 120-300 off walking at her when she threw at them came on 19 units by 0.3 s after the key, 17 by 0.6, 19 by
# 0.9 and -1 by 1.2 when hurt (n 16-28), where those her throw didn't hurt came on 19, 36, 43 and 50 (D:/MapleHelper-
# scratch/land/diag-move-away/flinch.py); the simulator's monsters stand so (an 18-unit push, then 0.6 s).
FLINCH_HOLD_S = 0.9

# --- landings with clearance, and moving away to attack (from 2026-10-27) ----------------------------------------------
# The user, 2026-10-27, asked whether landing with clearance from monsters should be the next fix: "Please do, the
# character should actively try to move away from the mob to attack it." Live (the 14 Tree Dungeon hunts of 2026-10-25
# 23:43 to 10-26 23:47, 189 minutes: D:/MapleHelper-scratch/land/diag-landings) she landed within 105 of a monster on the
# platform she came down on 3.9 times a minute (2.4 by live tracks alone), and from 69% of those landings the stance had
# no attack sure to take it; once down, the room behind her was a median 49, inside the 50 she kept from an end with
# monsters about (-14: no step at all in 77%), so she seldom could move away (a step that clears it 10% of the time, a
# jump back 0.1%), 27 of 40 close cases seen by eye against the tower's wall. A not-sure landing cost 5.7 EXP (se 1.7) and
# 0.13 touches (se 0.07) in the 3 s after it, against a sure one. So a landing is priced by what her first attack there
# would take (`landing`: the navigator's chains choose where they land by it, the stance's own jumps forward and back
# too), and up close she moves away to where her next attack surely takes the monster when it starts (_clear_step), to
# WALL_ROOM from a wall end, EDGE_MARGIN from a short drop (move_limits), never onto a spawn point the look's monsters
# don't account for (a respawn there lands on her: 55 of the 93 first clearing steps live ended within 50 of one), and
# at most MAX_MOVES in a row without a throw.
LANDING_SETTLE_S = STEP_SETTLE_S  # s from a landing to her first keys there: its slide, and the look after it
LANDING_PRICE_S = 1.36  # s of hunting a landing with no attack sure to take the nearest monster costs: a prior from the
                        # live numbers above (5.7 EXP at her 6.85 EXP a second, 0.83 s, and 0.13 touches at TOUCH_EXP
                        # each, 0.53 s)
MEET_PRICE_S = 2.86     # ... more where one walks into her before her first attack there: a touch (TOUCH_EXP, 4.1 s,
                        # but for the 30% she evades)
# What a touch costs her in EXP: the live Horny Mushroom's 112 HP (her touches of 2026-10-25) in potions at the user's
# reference prices (behavior.hunt: HP_PER_MESO 2 HP a meso, 56 mesos, at MESO_EXP 0.5), from 2026-10-31; until then
# about 25, a guess (LANDING_PRICE_S 1.2, MEET_PRICE_S 2.5). Its numbers written here (this package doesn't read
# behavior's): tests/test_landing.py pins them to hunt's prices. These are Tree Dungeon's live numbers, the defaults
# only: from the review of 2026-10-31 (one map's monster and rate frozen into general constants) a landing is priced by
# `Prices` where the caller has them (behavior.fight's FightOnTheWay.landings: the kind's touch, attack.danger's
# Danger.taken, of her max HP as read, at the user's prices; her hunt's EXP a second; the share of her touches that
# hurt), LANDING_LOST_S of her time and LANDING_TOUCHES touches for a landing not sure, a touch for one that meets her.
TOUCH_EXP = 28.0
LANDING_LOST_S = 0.83   # s of her hunting a not-sure landing lost (live 5.7 EXP at her 6.85 EXP a second) ...
LANDING_TOUCHES = 0.13  # ... and its touches more than a sure one's (live)
HURT_SHARE = 0.7        # of her touches that hurt (live: 30% evaded), until her own say otherwise


@dataclass(frozen=True)
class Prices:
    """What a landing costs her, s of her hunting (`landing`): `touch(kind)` EXP a touch of `kind` takes in potions,
    `rate` her EXP a second (over 0, behavior.fight.landing_prices bounding it), `hurt` the share of her touches that
    hurt."""
    touch: Callable[[str], float]
    rate: float
    hurt: float = HURT_SHARE

    def landing_s(self, kind: str) -> float:
        """A landing with no attack sure to take the nearest (`kind`): LANDING_LOST_S and LANDING_TOUCHES touches."""
        return LANDING_LOST_S + LANDING_TOUCHES * self.touch(kind) / self.rate

    def meet_s(self, kind: str) -> float:
        """... more where it walks into her before her first attack there: a touch, but for those she evades."""
        return self.hurt * self.touch(kind) / self.rate
WALL_ROOM = HER_HALF + 3.0  # units a move away keeps from a wall end of her platform (a knock can't take her off it):
                            # her half width and the slack of her dot
MAX_MOVES = 2          # guess: steps away in a row without a throw between (her decisions are made again at each look)
# R8 (the evade work, 2026-10-28): touched, she gets clear of the monster on her before her invulnerability ends
# (FIGHT_CLEAR). Live in walk B 53 of 148 touches came within 2 s of the one before, 34 of them as the window ended; in the
# simulator's hunt loop 68% of her touches came off her trips, and a fight stayed on a Horny Mushroom for three windows
# (D:/MapleHelper-scratch/evade/design/trace_loop2.txt). With the window on at the look (Pace.window, over
# CLEAR_MIN_S), the nearest monster on her platform too close for a throw from the ground in contact (its body on hers
# then, CONTACT_SLACK more) and not about to die (its HP bar over KILL_BAR: a throw may well take it), and a kind that
# counts (danger): the escape (_escape) decides, staying put CLEAR_STAY dearer than its danger there, whatever the
# kind's keep_off. False: as before.
FIGHT_CLEAR = True
CLEAR_MIN_S = 0.2
CONTACT_SLACK = 10.0
KILL_BAR = 0.25
CLEAR_STAY = 0.05
# From a monster that hits hard (_escape), the steps tried: STEP_AWAY or the room there is, as before, and (ESCAPE_STEPS)
# every ESCAPE_STEP from STEP_ROOM up, the safest end first: with the room a wall end leaves her (move_limits) the full
# step could end by another monster or a spawn point where a shorter one is safe (loop_tree_grind at 1.0 s: from a Horny
# Mushroom 18 off master's stack stepped 63, the room it had; with the room to 120 the branch threw from where she stood).
ESCAPE_STEPS = True
ESCAPE_STEP = 20
# Up close, the throw from where she stands at whichever monster reaches her first (the cornered study's R8, 2026-11-03),
# not at the nearest on the side she faces: of the nearest too close on each side (_close_quarters), where both throws
# from where she stands surely take theirs (_from_here), the one whose body meets hers first as `pace` walks them
# (_contact_s, by her attack after next: IMMINENT_S), then the most hurt (its HP bar: finish what was started, as
# pick_target and _jump_shot have it), then the side she faces, then the nearer; and where the one she faces has only
# an unsure jump in place left (no step, no jump back: the other hems her in), the sure throw the other way in its place.
# Live in BugBaby's hour of 2026-11-02 22:46-23:46 (D:/MapleHelper-scratch/cornered, episode A at 22:50:27-42 on 7) 6
# jumps in place went at a monster they didn't surely take with a sure one standing on her other side, none doing
# damage; replayed on A's looks (ep-A/replay/proto_fix.py) the other way's jump throw comes instead. The thief's alone
# (POINT_BLANK: the magician's casts up close are as they were). False: as before.
FIRST_TO_REACH = True
# Her full box against theirs (from 2026-10-31): her fights count a monster on a platform just under or over hers as on
# hers where its body meets hers (`beside`, Danger.beside: a kind that counts, its floor less than its top under her
# feet or less than her height over them, as her trips and the danger have it): who's too close (too_close, the hunt
# loop's rule 1), the moves that would take her into its body (_into), who's thrown at first (_throw_side) and who's
# imminent (_imminent), each walking its own platform (walks_on), a star from the ground at one sure only where the
# star's search picks it there (_star_takes); one off her platform too close only by its danger to her where it walks,
# its platform's ends and all (Danger.of: what keep_off is on hers), or on her. In the simulator's grind on Tree Dungeon
# (D:/MapleHelper-scratch/replan/pos) the touches from such a one went from 11 to 23 a run of 5 minutes with the loot
# sweep, on 9 between 8 and 10 above all (40-50 apart), where her fights went by her own platform's alone. Off: in the
# harness's standard grind (12 seeds of 5 minutes, paired against the branch before; the sweep's guards and her way back
# to the stop on in both, pos/impl/out/cmp_A2.txt, cmp_E.txt) on, 86.3 touches a run and 357 EXP a minute, off 81.2 and
# 373 (kills 50.7 against 52.8): her fights kept off what could touch her from under or over her and shot less, and were
# touched no less (the danger she escapes to, Danger.at, has them already). True: on. On from 2026-10-31 ~16:00, the
# user's choice of "more exp and loot": at their prices, the rest as first built (BACK off, her trips' touches charged
# to the stop she's going to, the hook unscaled; 12 seeds, D:/MapleHelper-scratch/replan/loot/price), on made 19.7 EXP a
# minute more (se 14.5) with 4.3 touches that hurt fewer (se 2.9) and 0.55 bars of HP less (se 0.47).
BESIDE = True
AWAY = "away"  # a step away from a close monster ('back', side, units, AWAY): what MAX_MOVES counts
HELD_STEP = 30.0  # units a step off a spot of the magician's held goes (held_step): past behavior.attacks.HOLD_MOVED's
                  # 20 as her x reads it, a band and a half of her search


@dataclass(frozen=True)
class Pace:
    """How soon her next attack picks its target, and how the monsters walk meanwhile (clearance kept ahead of time):
    `lead`, s from the look's frame to when her next keys can go (what's left of her attack lock, and the look's age);
    `walk`, a monster's speed along its platform (units/s, + right) as its track has it, or None where nothing tells
    (it stays where it was seen; with WALK_AT it walks at her, at `speeds` of its kind, DEFAULT_SPEED for one it
    lacks); `halves`, half each kind's width (where one walking at her meets her: PACE_STOP). Without `walk` the
    monsters stay where they were seen, as before."""
    lead: float = 0.0
    walk: Callable[[Monster], float | None] | None = None
    speeds: dict[str, float] | None = None
    halves: dict[str, float] | None = None
    hold: Callable[[Monster], tuple[float, float] | None] | None = None  # (from, until) s after the look it stands:
                                                                          # her running attack's hit (FLINCH_HOLD_S)
    window: float = 0.0  # s left of her invulnerability after a touch at the look (Senses.invulnerable_until): R8

    def at(self, m: Monster, x: float, t: float, platform: Platform | None = None) -> float:
        """Where monster `m` will be (map x) `t` s after the look, her at map x `x`: as its walk has it, within
        `platform`'s ends if it's on it; one walking at her stops where its body meets hers (PACE_STOP), on the side
        of her it was seen on."""
        if self.walk is None:
            return m.x
        v = self.walk(m)
        if v is None:
            if not WALK_AT:
                return m.x
            v = (self.speeds or {}).get(m.kind, DEFAULT_SPEED) * (1.0 if m.x < x else -1.0)
        walked = t
        if self.hold is not None and (held := self.hold(m)) is not None:  # her attack's hit stands it a while
            stops = max(0.0, held[0])
            walked = min(t, stops) + max(0.0, t - max(stops, held[1]))
        px = m.x + v * walked
        if PACE_STOP and v and (m.x - x) * v < 0:  # walking at her: no nearer than where it meets her, if not yet
            s = 1.0 if m.x > x else -1.0
            px = x + s * max((px - x) * s, min((m.x - x) * s, half_of(m.kind, self.halves) + HER_HALF))
        if platform is not None and m.platform == platform.id:
            px = min(max(px, platform.left), platform.right)
        return px

    def meets(self, m: Monster, x: float, t: float, platform: Platform | None = None, half: float | None = None) -> bool:
        """Whether monster `m`'s body (`half` its width either side; by default its kind's, `halves`) meets hers at map
        x `x` `t` s after the look (where `at` has it): a touch, unless she's still invulnerable from the last."""
        half = half_of(m.kind, self.halves) if half is None else half
        return abs(self.at(m, x, t, platform) - x) <= half + HER_HALF + 1e-6


NO_PACE = Pace()


def jump_pick_s(key_ms: float) -> float:
    """s from a jump's key to its attack's start, the attack keyed `key_ms` after it (JUMP_LAG after the key, half
    ATTACK_DECIDE_LAG_MS on average)."""
    return key_ms / 1000 + GROUND_PICK_S


def too_close(t: Monster, x: float, halves: dict[str, float] | None = None, danger: Danger | None = None,
              pace: Pace = NO_PACE, platform: Platform | None = None, turn: bool = False) -> bool:
    """Whether monster `t` (on her platform) is too close to map x for her next throw from the ground to be a star, as
    it will be when that attack starts (`pace`: her lock and the look's age, and its walk; a tap first, `turn`, TAP_S
    later): the claw would swipe it (close_to, by half its width: `halves`), or with `danger` nearer than its kind's
    danger warrants (Danger.keep_off). The hunt loop's rule 1 and the stance go by it."""
    at = pace.lead + GROUND_PICK_S + (TAP_S if turn else 0.0)
    own = walks_on(t, platform)
    mx = pace.at(t, x, at, own)
    gap = abs(mx - x)
    if platform is not None and t.platform != platform.id:
        # One beside her (BESIDE): no claw takes it, and its platform's ends may keep it off her: on her when its body
        # is (CONTACT_SLACK), else by its danger to her where it walks (Danger.of: what keep_off is on her platform)
        if danger is None or own is None:
            return False
        return gap <= half_of(t.kind, halves) + HER_HALF + CONTACT_SLACK \
            or danger.of(t.kind, own, mx, platform, x) >= KEEP_DANGER
    return gap < reach.ground_sure(half_of(t.kind, halves))[0] \
        or (danger is not None and gap < danger.keep_off(t.kind))


def walks_on(t: Monster, platform: Platform | None) -> Platform | None:
    """The platform monster `t` walks, whose ends Pace.at keeps it to: `platform` (hers) if it's on it, else its own
    (Platform.peer: one on a platform just under or over hers, `beside` her), None where that isn't known."""
    if platform is None or t.platform == platform.id:
        return platform
    return platform.peer(t.platform)


def beside(t: Monster, platform: Platform, x: float, danger: Danger | None) -> bool:
    """Whether monster `t` counts as on her `platform` for her body at map x (BESIDE): on it, or with `danger` (where
    something counts) on a platform just under or over it whose monster's body meets hers there (Danger.beside)."""
    if t.platform == platform.id:
        return True
    return BESIDE and danger is not None and danger.any and danger.beside(t, platform, x)


@dataclass(frozen=True)
class Landing:
    """What a landing leaves her (`landing`): how far the nearest monster on the platform she lands on is when her first
    attack there starts, middle to middle (`gap`; None with none), its kind, whether one walks into her before then
    (`meets`), whether that attack surely takes the nearest (`sure`), and the landing's price in s of hunting (`price_s`:
    0 for a clear one)."""
    gap: float | None
    kind: str | None
    meets: bool
    sure: bool
    price_s: float


CLEAR_LANDING = Landing(None, None, False, True, 0.0)


def landing(platform: Platform, x: float, near, halves: dict[str, float] | None = None,
            tops: dict[str, float] | None = None, danger: Danger | None = None,
            can_jump: Callable[[float, float], bool] | None = None, air: bool = True,
            prices: Prices | None = None) -> Landing:
    """A landing at map x on `platform`, with the monsters on it `near`, each (kind, map x when she lands, map x when her
    first attack there starts: LANDING_SETTLE_S and GROUND_PICK_S after the landing; as its walk has it, one walking at
    her stopping where it meets her, Pace's PACE_STOP). One whose body meets hers at either, or that passes her between,
    `meets` (a touch). Her first attack surely takes the nearest then (`sure`) if a throw from the ground does there
    (reach.ground_sure: past its claw's edge) or, with no platform overhead (`can_jump`), a jump in place keyed 40 ms after
    the jump key does (reach.air_sure, CLEAR_MARGIN farther: jump_reach on her level; the slope where it will be; its line
    clear, _jump_clear: not into a wall its body is over), and one
    of a kind that hits hard (`danger`) is at least its keep-off off (Danger.keep_off: a Horny Mushroom 169). Its price
    LANDING_PRICE_S where it isn't sure, MEET_PRICE_S more where one meets her (with `prices`, Prices.landing_s and
    meet_s for the nearest's kind: hers, not Tree Dungeon's); 0 for a clear landing, with nobody on
    the platform, or with the nearest past where her attack from the ground reaches (ground_sure's farthest: nothing to
    take there, nor coming at her by then; the reviews of 2026-10-27: on Tree Dungeon's floor, 530 wide, a Green 500 off
    was priced as one 52 off). What surely takes it is reach's for her attack (ground_sure), and the jump in place only
    for an attack she makes from a jump (`air`: the thief's Lucky Seven; not the magician's Magic Claw, cast from the
    ground only: character.AttackSpec.air), so the aim facade can give another character's (S1)."""
    best, meets = None, False
    y = platform.y_at(min(max(x, platform.left), platform.right))
    for kind, at_land, at_attack in near:
        reach_x = half_of(kind, halves) + HER_HALF
        if abs(at_land - x) <= reach_x + 1e-6 or abs(at_attack - x) <= reach_x + 1e-6 \
                or (at_land - x) * (at_attack - x) < 0:
            meets = True
            gap = min(abs(at_attack - x), reach_x)
        else:
            gap = abs(at_attack - x)
        if best is None or gap < best[0]:
            best = (gap, kind, at_attack)
    if best is None:
        return CLEAR_LANDING
    gap, kind, mx = best
    half = half_of(kind, halves)
    top = (tops or MONSTER_TOP).get(kind, MONSTER_TOP["pig"])
    near_ground, far_ground = reach.ground_sure(half)
    if gap > far_ground and not meets:  # (out of her reach, and not on her by her first attack: nothing to it)
        return Landing(round(gap, 1), kind, False, True, 0.0)
    sure = near_ground <= gap
    if not sure and air and (can_jump is None or can_jump(x, x)):
        dy = platform.y_at(min(max(mx, platform.left), platform.right)) - y
        sure = reach.air_sure(gap - CLEAR_MARGIN, dy, top, half, physics.JUMP_ATTACK_MS) \
            and _jump_clear(platform, x, mx, platform, y + dy, top, half, physics.JUMP_ATTACK_MS)
    if sure and danger is not None and gap < danger.keep_off(kind):
        sure = False
    if prices is None:
        price = (0.0 if sure else LANDING_PRICE_S) + (MEET_PRICE_S if meets else 0.0)
    else:
        price = (0.0 if sure else prices.landing_s(kind)) + (prices.meet_s(kind) if meets else 0.0)
    return Landing(round(gap, 1), kind, meets, sure, price)


def first_attack_s(landed: float) -> float:
    """When her first attack after a landing at `landed` (s) starts: its slide and the look after it (LANDING_SETTLE_S),
    and a throw from the ground's key to its attack's start (GROUND_PICK_S, the thief's: character.Shape.pick_s is the
    profile's, which S1 gives another character's by). What `landing_of` and the navigator's judge
    (behavior.fight.FightOnTheWay.landings) both price a landing by."""
    return landed + LANDING_SETTLE_S + GROUND_PICK_S


def landing_of(platform: Platform, x: float, monsters: list[Monster], lead: float, pace: Pace = NO_PACE,
               halves: dict[str, float] | None = None, tops: dict[str, float] | None = None,
               danger: Danger | None = None, can_jump: Callable[[float, float], bool] | None = None,
               air: bool = True, prices: Prices | None = None) -> Landing:
    """`landing` for a landing at map x on `platform` `lead` s after the look, the monsters on it of `monsters` (a look's)
    walked on by `pace` (where each will be when she lands, and when her first attack there starts: first_attack_s)."""
    after = first_attack_s(lead)
    near = [(m.kind, pace.at(m, x, lead, platform), pace.at(m, x, after, platform))
            for m in monsters if m.platform == platform.id]
    return landing(platform, x, near, halves, tops, danger, can_jump, air, prices)


def move_limits(model: MapModel, platform: Platform) -> tuple[float, float]:
    """The stretch of `platform` her moves away up close keep to (stance's `limits`): WALL_ROOM from an end a wall of its
    foothold group rises from (a knock can't take her off there: end_drop 0), reach.EDGE_MARGIN from one with a short
    drop under it, drop_limits' DROP_ROOM from one with a long drop. Where she kept 50 from every end with monsters about
    (EDGE_MARGIN and KNOCKBACK_ROOM): live she was pinned against Tree Dungeon's wall ends (24's right end 6 of 8 times
    by eye), 61 from it at the median, with no step left. On one too narrow, as drop_limits."""
    def work() -> tuple[float, float]:
        def margin(side: int) -> float:
            drop = end_drop(model, platform, side)
            return WALL_ROOM if drop == 0.0 else DROP_ROOM if drop >= LONG_DROP else reach.EDGE_MARGIN
        lo, hi = platform.left + margin(-1), platform.right - margin(1)
        return drop_limits(model, platform) if lo > hi else (lo, hi)
    return _per_platform(_MOVES, model, platform, work)


_MOVES: dict[tuple[int, int], tuple] = {}  # (id(model), platform) -> (the model, move_limits' answer)


def _per_platform(cache: dict, model: MapModel, platform: Platform, work: Callable[[], tuple]) -> tuple:
    """`work()`'s answer for `platform` of `model`, worked out once (`cache`: (id(model), platform id) -> (the model, the
    answer), the model kept so its id can't go to another one while it's there; cleared past 4096)."""
    key = (id(model), platform.id)
    got = cache.get(key)
    if got is None or got[0] is not model:
        if len(cache) > 4096:
            cache.clear()
        got = cache[key] = (model, work())
    return got[1]


@dataclass(frozen=True)
class Here:
    """The throw she'd make from where she stands at a monster too close (_from_here): the stance's act, and whether its
    attack surely takes it (`sure`); `gap`, how far ahead of her it will be when the attack starts (middle to middle)."""
    act: tuple
    sure: bool
    gap: float


@dataclass(frozen=True)
class JumpPlan:
    """How a jump throw or a jump back goes (control.actions.Actions.jump_throw, jump_back): the attack key `key_ms`
    after the jump key; `way` +1/-1 a jump forward that way (0: in place); `air_turn` False: turning, the tap goes
    before the jump (an attack too soon in the jump to turn first: a short monster); why it was chosen (`why`: the
    telemetry 'jump' record's kind) and the shot it makes (attack.reach.Shot), if worked out; `room`, the keys about
    `key_ms` a jitter may key it at where it has no shot (a jump throw up close: _from_here), each as sure as it, and
    the monster it's for (`target`: the shot's is its own). `keys`: those, or its shot's window. `sure` False: a room
    whose keys nothing checked (a jump attack keyed as the stance's own, given the jitter's whole room as it's carried
    out: behavior.fight.jittered's JITTER_ALL, the review of 2026-11-01)."""
    key_ms: int = physics.JUMP_ATTACK_MS
    way: int = 0
    air_turn: bool = True
    why: str = ""
    shot: reach.Shot | None = None
    room: tuple[int, int] | None = None
    target: Monster | None = None
    sure: bool = True

    @property
    def keys(self) -> tuple[int, int] | None:
        """The attack keys (ms after the jump key) its jitter may key it at (behavior.fight.jittered): `room`, else its
        shot's window (reach.Shot.window), else None (no jitter: the stance's own timing)."""
        if self.room is not None:
            return self.room
        return None if self.shot is None else self.shot.window


def plan_of(act: tuple) -> JumpPlan | None:
    """A jump throw's or jump back's plan (JumpPlan), if the stance gave one (else its defaults)."""
    return act[2] if len(act) > 2 and isinstance(act[2], JumpPlan) else None


def _side_of(t: Monster, x: float) -> str:
    """Which side of map x monster `t` is on (LEFT when right at x)."""
    return RIGHT if t.x > x else LEFT


def _nearest(monsters: list[Monster], x: float) -> Monster:
    """The monster nearest to map x along the platforms (the first of equally near ones)."""
    return min(monsters, key=lambda t: abs(t.x - x))


def _goal_reach() -> float:
    """How far from its target a spot she walks to may be: a little past SHOT_RANGE."""
    return reach.SHOT_RANGE + 15


def _clear_from(platform: Platform, x: float, y: float, t: Monster, lo: float, hi: float,
                around: list[Monster] | None = None, sizes: dict[str, MobBody] | None = None) -> float | None:
    """The nearest place to map x along `platform`, from `lo` to `hi` (every CLEAR_STEP units), from where a
    star reaches monster `t` (reach.can_shoot, out to _goal_reach) and terrain doesn't stop it (reach.wall_to),
    or None. A wall of her foothold group stops it from anywhere on her platform; by the v83 line (reach.LINE)
    her own floor stops a star at a monster below unless she stands near enough its end. With `around` (the
    monsters around her), nor does a monster the attack takes first have its star stopped there (reach.can_hit's
    shield): from 14 at x 690 a pig on 17 at 880, in her floor's shadow, takes the stars at one at 1000, and
    from x 729 on the search doesn't pick it (under 151 off), so the one at 1000 gets them. `sizes`: its body by the
    map's game data (the magician's stance), else reach's tables."""
    xs = [lo + i * CLEAR_STEP for i in range(int((hi - lo) // CLEAR_STEP) + 1)] + [hi]
    if around is None:
        ok = [g for g in xs if reach.can_shoot(t.x - g, t.y - y, reach=_goal_reach(), kind=t.kind, sizes=sizes)
              and reach.wall_to(platform, g, y, t) is None]
    else:
        ok = [g for g in xs if reach.can_hit(platform, g, y, t, reach=_goal_reach(), others=around, sizes=sizes)]
    return min(ok, key=lambda g: abs(g - x), default=None)


def shooting_place(platform: Platform, x: float, y: float, t: Monster, most: float,
                   around: list[Monster]) -> float | None:
    """Where along `platform` within `most` units of map x her attack from the ground takes monster `t` standing there
    (_reached_from: the fights' own rule at that place's height, within where she stands to shoot with monsters about:
    _stand_limits), the nearest, with no monster of her platform within TOO_CLOSE of the walk there (her body's way:
    those are the fights' to deal with); None if there's none, or it's under MIN_MOVE off (shooting from where she
    stands is the caller's). The navigator's shot first at a monster by a landing (behavior.fight.FightOnTheWay.
    shoot_from, shot_place)."""
    return place_to_shoot(platform, x, y, t, most, around)[0]


def place_to_shoot(platform: Platform, x: float, y: float, t: Monster, most: float,
                   around: list[Monster]) -> tuple[float | None, str]:
    """shooting_place's place, and where there's none why not, for the log and the 'landing' event (behavior.fight.
    FightOnTheWay.shot_place): (the place, ''), or (None, why): no throw from the ground takes it from anywhere along
    `platform` within where she stands to shoot, with how far under or over her (at `y`) it is (from Tree Dungeon's 21
    one on 23, 117-122 under it, past a star's band, reach.SHOT_DOWN: live, 2026-10-28 00:38:36), or only from a stretch
    narrower than a walk's slack (SHOOT_SLACK either way); only from farther than `most` (the nearest such place said);
    only from under MIN_MOVE off; or a monster of her platform in the walk's way. From the ground only: a jump throw
    from where she stands is FightOnTheWay.shot_place's to ask (reach.shots there; at every place along her platform
    it'd cost a coverage each)."""
    lo, hi = _stand_limits(platform, crowded=any(o.platform == platform.id for o in around))
    g = _reached_from(platform, x, t, max(lo, x - most), min(hi, x + most), around)
    if g is None:
        far = None if x - most <= lo and x + most >= hi else _reached_from(platform, x, t, lo, hi, around)
        if far is None:
            dy = t.y - y
            where = f"it's {abs(dy):.0f} {'under' if dy > 0 else 'over'} her"
            edge = _reached_from(platform, x, t, lo, hi, around, slack=0.0)
            if edge is not None:
                return None, (f"a throw from the ground takes it only from a sliver of platform {platform.id} "
                              f"(x {edge:.0f}, under SHOOT_SLACK of a walk's end either way; {where})")
            return None, f"no throw from the ground takes it from anywhere on platform {platform.id} ({where})"
        return None, (f"a throw from the ground takes it only from x {far:.0f} on platform {platform.id}, "
                      f"{abs(far - x):.0f} off")
    if abs(g - x) < MIN_MOVE:
        return None, f"a throw from the ground takes it from x {g:.0f}, under MIN_MOVE off"
    if any(o.platform == platform.id and min(x, g) - TOO_CLOSE <= o.x <= max(x, g) + TOO_CLOSE for o in around):
        return None, f"a monster of platform {platform.id} in the walk's way to x {g:.0f}"
    return g, ""


def _reached_from(platform: Platform, x: float, t: Monster, lo: float, hi: float, around: list[Monster],
                  slack: float | None = None) -> float | None:
    """The nearest place to map x along `platform`, from `lo` to `hi` (every CLEAR_STEP units, as _clear_from's), from
    where her attack takes monster `t` standing there, and `slack` (SHOOT_SLACK) either way of it too, within her
    platform (a walk there may stop short or go on a little): the fights' own rule (reach.can_hit at the shot's reach,
    as behavior.fight.FightOnTheWay.clear_for has it from there), her feet at the platform's height at each place. Not
    _clear_from's (her height where she is now, out to _goal_reach: a walk's goal the fights' next look decides on
    again): on 25's slope (10 lower at 140 than at 89) that had a Green on 24 at 260 in reach from 89 for her at 140,
    where standing at 89 it isn't, so the shot first's walk ended where clear_for found no throw (from 2026-10-28).
    Without the slack (as first built that day) the nearest place lay on the reach's edge, and a walk a few units short
    ended out of it (the review's: 5 of 24 simulated walks of 138, 23 to a Green on 22)."""
    slack = SHOOT_SLACK if slack is None else slack

    def takes(g: float) -> bool:
        return reach.can_hit(platform, g, platform.y_at(g), t, others=around)

    def sure(g: float) -> bool:
        ends = {min(max(g + d, platform.left), platform.right) for d in (-slack, slack)} if slack > 0 else set()
        return takes(g) and all(takes(v) for v in ends - {g})
    xs = [lo + i * CLEAR_STEP for i in range(int((hi - lo) // CLEAR_STEP) + 1)] + [hi]
    return next((g for g in sorted(xs, key=lambda g: abs(g - x)) if sure(g)), None)


def _stand_limits(platform: Platform, crowded: bool) -> tuple[float, float]:
    """The stretch of `platform` she keeps to: EDGE_MARGIN from its ends, and KNOCKBACK_ROOM more while
    monsters share it (`crowded`): a hit knocks her back."""
    margin = reach.EDGE_MARGIN + (KNOCKBACK_ROOM if crowded else 0)
    return platform.left + margin, platform.right - margin


def jump_room(platform: Platform) -> tuple[float, float]:
    """The most of `platform` a jump shot she takes lands her on: _jump_shot keeps a jump's landing to _stand_limits'
    stretch, crowded (KNOCKBACK_ROOM narrower) or not, whoever asks (the stance, jump_hits), so a jump landing her
    elsewhere makes none and its attack keys needn't be tried (behavior.fight.jump_cover: reach.shots' `lands`)."""
    return _stand_limits(platform, crowded=False)


def end_drop(model: MapModel, platform: Platform, side: int) -> float:
    """How far she'd fall knocked off `platform`'s end on `side` (-1 its left, +1 its right): down to the highest
    platform under the map x just past it (inf with none), 0 where a wall of its foothold group rises from that end
    (Platform.walls: it stops her)."""
    e = platform.left if side < 0 else platform.right
    y = platform.y_at(e)
    for w in platform.walls:
        if w.x1 == w.x2 and abs(w.x1 - e) <= 2 and min(w.y1, w.y2) < y - 1 and max(w.y1, w.y2) > y - her_body.HER_TALL:
            return 0.0
    x = e + side * END_PAST  # where her feet leave it, knocked past it
    under = [q.y_at(x) - y for q in model.places if q.id != platform.id and q.covers(x) and q.y_at(x) > y + 1]
    return min(under, default=float("inf"))


def drop_limits(model: MapModel, platform: Platform) -> tuple[float, float]:
    """The stretch of `platform` her steps away and jumps back keep to (stance's `edges`): DROP_ROOM in from an end with
    a long drop under it (end_drop LONG_DROP or more), the end itself at another; worked out once a platform. On one
    too narrow for that, as far from the long drops as it goes: its middle between two (a step only toward it; review
    of 2026-10-26: Tree Dungeon's 18, 27 and 30, 187-188 units wide, drop 125-254 units at both ends, and the stretch
    was the whole platform), the other end with one."""
    def work() -> tuple[float, float]:
        left, right = end_drop(model, platform, -1) >= LONG_DROP, end_drop(model, platform, 1) >= LONG_DROP
        lo = platform.left + (DROP_ROOM if left else 0.0)
        hi = platform.right - (DROP_ROOM if right else 0.0)
        if lo > hi:
            lo = hi = (platform.left + platform.right) / 2 if left and right else platform.right if left \
                else platform.left
        return lo, hi
    return _per_platform(_DROPS, model, platform, work)


_DROPS: dict[tuple[int, int], tuple] = {}  # (id(model), platform) -> (the model, drop_limits' answer)


def overhead(model: MapModel, platform: Platform, x: float) -> Platform | None:
    """The platform a jump from map x on `platform` lands her on: the nearest one overhead within her jump height (she
    goes up through it and lands on top: open_above's); None where a jump lands her back on hers."""
    best = None
    for q in model.places:  # (not a twin of hers: mapdata.MapModel.twins)
        if q.id == platform.id or not q.covers(x):
            continue
        dy = platform.y_at(min(max(x, platform.left), platform.right)) - q.y_at(x)
        if 0 < dy <= JUMP_HEIGHT + 5 and (best is None or dy < best[0]):
            best = (dy, q)
    return None if best is None else best[1]


def open_above(model: MapModel, platform: Platform, x0: float, x1: float) -> bool:
    """Whether she can jump anywhere between x0 and x1 on `platform` and land back on it: no other
    platform overhead within her jump height (she'd go up through it and land on top)."""
    lo, hi = sorted((x0, x1))
    for q in model.places:  # (not a twin of hers, 0.15 units above her: mapdata.MapModel.twins)
        if q.id == platform.id or q.right < lo or q.left > hi:
            continue
        mid = min(max((lo + hi) / 2, q.left), q.right)
        if 0 < platform.y_at(min(max(mid, platform.left), platform.right)) - q.y_at(mid) <= JUMP_HEIGHT + 5:
            return False
    return True


def stance(x: float, y: float, platform: Platform, wanted: list[Monster], approach: bool = True,
           facing: str | None = None, can_jump: Callable[[float, float], bool] | None = None,
           tops: dict[str, float] | None = None, halves: dict[str, float] | None = None,
           around: list[Monster] | None = None, danger: Danger | None = None,
           jumps: Callable[[], reach.Coverage] | None = None, toward: int = 0, goal: float | None = None,
           edges: tuple[float, float] | None = None, up: Callable[[float], Platform | None] | None = None,
           pace: Pace = NO_PACE, touch: list[Monster] = (), limits: tuple[float, float] | None = None,
           moves: int = 0, castable: Callable[[Monster], bool] | None = None,
           hurt: Callable[[Monster], float | None] | None = None, hidden_hurt: list[Monster] = (),
           clawed: bool = False) -> tuple:
    """What to do from map (x, y) on `platform` with these monsters around: ('throw', side),
    ('jump_back', side) at one too close for a throw from the ground (jump away, throw back from the
    air), ('jump_throw', side) the same when cornered (straight up), ('back', side, units[, AWAY]) to step away,
    ('approach', side, units) to bring the nearest into reach, or ('wait',). She keeps working the side
    she faces while it still has monsters in reach (switching sides every throw spread the damage over
    both), and never walks past a monster. `can_jump(x0, x1)` says whether jumping over that stretch
    keeps her on her platform: under a platform within her jump height, a jump lands her up on it.
    A monster on her platform is too close for a throw from the ground while the claw would swipe it
    (close_to). `tops`: how tall each kind is at least, `halves`: half its width (the map's game data,
    localization.monsters.bodies; MONSTER_TOP and MONSTER_HALF by default). `around`: every monster around
    her, of any kind (`wanted` by default): one the attack takes first whose star is stopped shields those
    behind it (reach.shielded). `danger`: the map's (attack.danger), for how near a monster that hits hard may come
    (Danger.keep_off) and where she goes from it (_escape). `jumps`: what her attacks reach from here, from the ground
    and from her jumps (attack.reach.coverage, worked out when first asked): with nothing in reach from the ground, a
    jump throw whose attack surely hits a wanted monster from the air comes before a step or a walk (_jump_shot: a
    monster on a platform above her, out of a star's reach from the ground), ('jump_throw', side, JumpPlan)
    or ('jump_back', side, JumpPlan); `toward` (+1/-1: she's travelling that way, to map x `goal` on her platform if
    given), a jump forward that way that hits one, landing short of `goal` (PASS_SLACK), comes before a throw from the
    ground (PASSING_JUMPS). `edges`: the stretch of her platform her steps away and jumps back keep to (drop_limits:
    DROP_ROOM from an end with a long drop under it); `up(x)`: the platform a jump from x lands her on (overhead), for
    an escape (_escape). `pace`: how soon her next attack picks and how the monsters walk meanwhile: who's too close
    goes by where each will be then (too_close), and up close what each throw would take then (_from_here). `touch`:
    monsters about her that count for her body alone (the danger where she'd stand, her jumps' arcs and landings, an
    escape's ways), not as targets or as taking her stars (behavior.fight.TOUCH_NEAR: her kill's monster not seen dead
    yet, which may be a live one). `limits`: where her moves away up close keep to instead (move_limits: nearer a wall
    end, from 2026-10-27); `moves`: how many moves away she has made since her last throw (MAX_MOVES: then no step),
    counted only where something counts as a danger (`danger`'s any: not on the first map, whose stance is as it was).
    A step away from a close monster comes as ('back', side, units, AWAY), and _step_from_under's as ('back', side,
    units): the fights count only the first toward MAX_MOVES. `castable`: whether her attack goes at a monster at all
    (behavior.attacks.Attacks.refused: the magician's casts, none at one unseen, dead, likely dead or held; None: every
    one): one it won't go at is never thrown at, nor taken as sure from where she stands up close, but counts as ever
    for her body, the danger and who's too close. Without POINT_BLANK (the magician's), up close with nothing she casts
    at from where she stands (_close_quarters' 'wait'), a throw that way at another in reach she casts at. With
    `castable` her reach is by each kind's body in the map's game data (`tops`, `halves`: reach.can_hit's `sizes`), as
    her casts' search is (ledger.in_path's: a mushroom 28 either side, where reach's table has a pig's 34), so the
    stance neither throws where that search picks nothing (a throw her cast would send none with) nor stays there: off
    her level, nearer than it picks one, she steps out from under it (_step_from_under; the review of 2026-10-29: 60
    over her, the stance took one 116-122 off for in reach where her search picks from 123). `hurt`: what a monster she
    has thrown at has left (its HP's share as she reckons it, alive; None: none of hers), and `hidden_hurt` hers unseen
    this look as last seen: her wounds first, with the imminent danger before them (WOUNDS_FIRST). `clawed`: a monster
    the looks miss is on her (a touch with none the look found about her that could have made it: behavior.fight
    .clawed), so a throw from the ground would go to the claw: a throw at one in reach goes from a jump in place where
    its attack surely picks it (CLAWED_AIR: _from_air), and up close so does one past the one too close where the throw
    from where she stands would be a jump in place sure of nothing (CLAWED_CLOSE)."""
    can_jump = can_jump or (lambda a, b: True)
    counts = danger is not None and danger.any
    moves = moves if counts else 0
    around = wanted if around is None else around
    body = around + [m for m in touch if not any(m is o for o in around)]
    mine = [t for t in wanted if beside(t, platform, x, danger)]  # (hers, and one under or over her: BESIDE)
    lo, hi = _stand_limits(platform, crowded=bool(mine))
    # a throw from the ground would be a claw swipe, or it hits too hard, as it will be when the attack starts
    close = [t for t in mine if too_close(t, x, halves, danger, pace, platform, _side_of(t, x) != facing)]
    sizes = _sizes(tops, halves) if castable is not None else None  # (her casts' search: the game's bodies)
    reachable = [t for t in wanted if reach.can_hit(platform, x, y, t, others=around, sizes=sizes)]
    if close:
        slo, shi = (lo, hi) if edges is None else (max(lo, edges[0]), min(hi, edges[1]))
        if limits is not None:
            slo, shi = max(platform.left, limits[0]), min(platform.right, limits[1])
        act, unsure = _close_quarters_here(x, close, mine, facing, slo, shi, can_jump, tops or MONSTER_TOP, halves,
                                           (platform.left + SMALL_STEP_END, platform.right - SMALL_STEP_END)
                                           if SNAIL_FAR_END else None, danger, platform, body, up, y, pace, moves,
                                           seen=around, castable=castable, clawed=clawed)
        if clawed and CLAWED_CLOSE and CLAWED_AIR and POINT_BLANK and act[0] == "jump_throw" and unsure:
            # (an unsure jump in place, a tap first or not: past the one on her instead)
            past = [t for t in reachable if not any(t is c for c in close)]
            air = _from_air(x, y, act[1], past, facing, can_jump, tops, halves, platform, pace) if past else None
            if air is not None:
                return air
        if act[0] == "wait" and not POINT_BLANK:  # (her: none she casts at up close; another in reach that she does)
            others = [t for t in reachable if not any(t is c for c in close) and (castable is None or castable(t))]
            if others:
                return ("throw", _throw_side(x, others, facing, platform, danger))
        return act
    # (`castable` asked only where its answer decides: behavior.fight.Castable keeps those it refused, for the log)
    in_reach = [t for t in reachable if castable is None or castable(t)]
    if in_reach:
        if toward and PASSING_JUMPS and jumps is not None:
            jump = _jump_shot(x, y, platform, wanted, jumps(), lo, hi, facing, body, danger, halves, toward,
                              kinds=("forward",), why="passing", goal=goal, pace=pace, tops=tops, can_jump=can_jump)
            if jump is not None:
                return jump
        imminent = _imminent(x, platform, body, danger, pace, halves) if WOUNDS_FIRST and hurt is not None else None
        side = _throw_side(x, in_reach, facing, platform, danger, hurt if WOUNDS_FIRST else None, imminent)
        if WOUNDS_FIRST and hidden_hurt and imminent is None:  # (hers unseen just now: waited for, not turned from)
            waited = [m for m in hidden_hurt if reach.can_hit(platform, x, y, m, others=around, sizes=sizes)]
            if waited:
                way = _side_of(min(waited, key=lambda m: abs(m.x - x)), x)
                if not any(_side_of(t, x) == way for t in in_reach):
                    return ("wait",)
        if clawed and CLAWED_AIR and POINT_BLANK:  # (a monster the looks miss on her: the ground's press is the claw's)
            air = _from_air(x, y, side, in_reach, facing, can_jump, tops, halves, platform, pace)
            if air is not None:
                return air
        return ("throw", side)
    if jumps is not None:
        jump = _jump_shot(x, y, platform, wanted, jumps(), lo, hi, facing, body, danger, halves, toward, pace=pace,
                          tops=tops, can_jump=can_jump)
        if jump is not None:
            return jump
    act = _step_from_under(x, y, platform, wanted, lo, hi, around, sizes)
    if act is None and approach:
        act = _close_in(x, y, platform, wanted, lo, hi, around, sizes)
    return act or ("wait",)


def _sizes(tops: dict[str, float] | None, halves: dict[str, float] | None) -> dict[str, MobBody] | None:
    """Each kind's body as reach.body_of takes it from the map's game data (MobBody) out of the stance's `tops` and
    `halves` (behavior.fight.body_sizes: localization.monsters.bodies), or None without them: what the magician's
    reach goes by (stance's `castable`), as her casts' search does (ledger.in_path's `sizes`)."""
    if not tops or not halves:
        return None
    return {k: MobBody(tops[k], tops[k], halves[k]) for k in halves if k in tops}


def air_gap(act: str | tuple) -> float | None:
    """How long after its jump key a stance's attack key comes ('jump_throw', 'jump_back': its plan's, plan_of, else
    physics'), or None for a throw from the ground: how high she is when the attack picks its target follows from it
    (reach.air_rise)."""
    plan = plan_of(act) if isinstance(act, tuple) else None
    name = act[0] if isinstance(act, tuple) else act
    if plan is not None and name in ("jump_throw", "jump_back"):
        return plan.key_ms
    return {"jump_throw": physics.JUMP_ATTACK_MS, "jump_back": physics.JUMP_BACK_ATTACK_MS}.get(name)


def _jump_shot(x: float, y: float, platform: Platform, wanted: list[Monster], cover: reach.Coverage, lo: float,
               hi: float, facing: str | None, around: list[Monster], danger: Danger | None,
               halves: dict[str, float] | None, toward: int = 0, kinds: tuple[str, ...] = reach.JUMP_KINDS,
               why: str = "", goal: float | None = None, pace: Pace | None = None,
               tops: dict[str, float] | None = None,
               can_jump: Callable[[float, float], bool] | None = None) -> tuple | None:
    """A jump throw at one of `wanted` from map (x, y) on `platform` (stance's): of the shots `cover` has at them
    (attack.reach.shots) of `kinds`, those whose jump lands her back on her platform between `lo` and `hi` (where
    she keeps to), a jump forward or back however it goes (reach.Shot.lands_sure, land_span: lifting off early or
    late, slower, the kick short, as a route jump's landing is checked), her body meeting no monster's on the way
    (_arc_meets), a jump forward or back neither landing nor passing into one (_into) nor, with `danger`, landing
    where the danger is more than JUMP_DANGER over where she stands; an 'up' as she'd make it (turning in the air if
    she doesn't face that way by her keys, `facing`). Of those, at the target pick_target's order would take first
    (hurt first, then the nearest), the one soonest done: in place, then forward (`toward` first: she's going that
    way; in passing, `why` 'passing', only that way, and landing no more than PASS_SLACK past `goal`, where she's
    going on her platform, if given), then back, the soonest attack key first. None if there's none. `why`: the
    plan's (else 'above' at a monster off her level above her, 'reach' at one that isn't). With `pace` (from
    2026-10-27) and a `danger` where something counts (Danger.any: not on the first map), a jump forward or back only
    where it lands with clearance from the monsters on her platform (`landing`, as they walk by then: none that walks
    into her, her first attack there sure to take the nearest; `can_jump`: no jump in place counts under a platform
    overhead there)."""
    here_danger = None
    mine = ([m for m in around if m.platform == platform.id]
            if pace is not None and danger is not None and danger.any else [])
    best = None
    for t in wanted:
        for sh in cover.of(t):
            if sh.jump is None or sh.kind not in kinds:
                continue
            side = RIGHT if sh.side > 0 else LEFT
            if sh.kind == "up" and sh.turn != (facing != side):
                continue
            if why == "passing" and sh.jump.way != toward:
                continue
            if not sh.lands_back(platform, lo, hi):
                continue
            near, far = sh.land_span
            if sh.kind != "up":
                end = far if sh.jump.way > 0 else near  # the farthest along her way it may take her
                if why == "passing" and goal is not None and (end - goal) * sh.jump.way > PASS_SLACK:
                    continue
                if _into(x, end, platform, around, halves):
                    continue
                if danger is not None:
                    if here_danger is None:
                        here_danger = danger.at(platform.id, x, around)
                    if any(danger.at(platform.id, lx, around) > here_danger + JUMP_DANGER
                           for lx in {sh.land_x, near, far}):
                        continue
                if mine and landing_of(platform, sh.land_x, mine, pace.lead + sh.jump.lead_s + physics.JUMP_LAG
                                      + sh.land_s, pace, halves, tops, danger, can_jump).price_s > 0:
                    continue
            if _arc_meets(x, y, platform, sh, around, halves):
                continue
            hurt = t.bar is not None
            order = {"up": 0, "forward": 1 if not toward or sh.jump.way == toward else 2, "back": 3}[sh.kind]
            rank = (not hurt, t.bar if hurt else 0.0, abs(t.x - x) + 2 * abs(t.y - y), order, sh.key_ms)
            if best is None or rank < best[0]:
                best = (rank, sh, t)
    if best is None:
        return None
    _, sh, t = best
    side = RIGHT if sh.side > 0 else LEFT
    reason = why or ("above" if t.y < y - reach.LEVEL else "reach")
    if sh.kind == "back":
        return ("jump_back", side, JumpPlan(sh.key_ms, -sh.side, True, reason, sh))
    return ("jump_throw", side, JumpPlan(sh.key_ms, sh.jump.way, True, reason, sh))


def jump_hits(x: float, y: float, platform: Platform, m: Monster, cover: reach.Coverage, facing: str | None = None,
              around: list[Monster] | None = None, danger: Danger | None = None,
              halves: dict[str, float] | None = None) -> bool:
    """Whether the stance would jump-throw at `m` from map (x, y) on `platform` with nothing in reach from the ground
    (_jump_shot's terms: a shot `cover` has at it landing her back where she keeps to however it goes, her body clear
    of every monster `around` her, the danger kept; as she'd make it, facing `facing`): what the hunt loop and grinding
    count as in reach from a jump (behavior.hunt's _jump_hits), so they don't stay to shoot what the stance won't.
    Where she keeps to as with monsters on her platform if any of `around` is on it."""
    around = [m] if around is None else around
    lo, hi = _stand_limits(platform, crowded=any(t.platform == platform.id for t in around))
    return _jump_shot(x, y, platform, [m], cover, lo, hi, facing, around, danger, halves) is not None


def _arc_meets(x: float, y: float, platform: Platform, sh: reach.Shot, around: list[Monster],
               halves: dict[str, float] | None) -> bool:
    """Whether her body (motion.bodies) over shot `sh`'s jump from map (x, y), from the plan's first key until she
    lands, meets a monster's of `around`, each where it can be by then (from where it was seen, either way at its
    kind's speed, MONSTER_SPEED: DEFAULT_SPEED for a kind the table lacks; its box for touches, touch_box), every
    ARC_STEP; a jump forward or back so too as its slowest variant goes (the one landing nearest: Shot.alts), longer
    at each place. A jump in place, against the monsters off her platform: up into the one she throws at on a
    platform above (review, 2026-10-25: right under it, her box met a pig 100 above her from 0.14 s to 0.54 s after
    the jump key, before the star left); where she stands, what walks into her is the fight's (her keeping off),
    jumping or not."""
    j = sh.jump
    flights = [(j, (sh.land, sh.land_x, sh.land_s))]
    if sh.alts:
        flights.append(min(sh.alts, key=lambda a: abs(a[1][1] - x)))
    boxes = []
    for m in around:
        if sh.kind == "up" and m.platform == platform.id:
            continue
        q = platform if m.platform == platform.id else platform.peer(m.platform)
        if q is None:
            continue
        half, tall = touch_box(m.kind)
        if halves and m.kind in halves:
            half = halves[m.kind]
        boxes.append((m, q, half, tall, MONSTER_SPEED.get(m.kind, DEFAULT_SPEED)))
    if not boxes:
        return False
    for jj, landing in flights:
        if landing[0] is None:
            continue
        t, end = -jj.lead_s, physics.JUMP_LAG + landing[2]
        while t <= end + 1e-9:
            fx, fy, _ = _pose(platform, x, jj, max(t, -jj.lead_s), landing)
            since = t + jj.lead_s
            for m, q, half, tall, speed in boxes:
                lo, hi = max(q.left, m.x - speed * since), min(q.right, m.x + speed * since)
                feet = q.y_at(min(max(fx, lo), hi))
                if her_body.touches(fx, fy, lo, hi, half, feet, tall):
                    return True
            t += ARC_STEP
    return False


def _pose(platform: Platform, x: float, j: physics.Jump, t: float, landing: tuple) -> tuple[float, float, bool]:
    """Her feet t s after jump `j`'s key (from -j.lead_s, the plan's first key) from map x on `platform`:
    physics.pose without the walls (the shot's own landing, worked out with them, ends it)."""
    land, lx, s = landing
    if t <= physics.JUMP_LAG:
        dx, _ = j.arc(t)
        x1 = min(max(x + dx, platform.left), platform.right)
        return x1, platform.y_at(x1), False
    tau = t - physics.JUMP_LAG
    if land is not None and tau >= s:
        return lx, land.y_at(lx), False
    x0 = min(max(x + j.way * j.run, platform.left), platform.right)
    return x0 + j.vx * tau, platform.y_at(x0) - physics.rise_after(tau), True


def half_of(kind: str, halves: dict[str, float] | None = None) -> float:
    """Half a monster's width by its kind: from `halves` (the map's game data) if it has the kind, else
    MONSTER_HALF, else a pig's (reach.body_of)."""
    return (halves or {}).get(kind, reach.body_of(kind)[1])


def close_to(t: Monster, x: float, halves: dict[str, float] | None = None) -> bool:
    """Whether monster `t` (on her level) is too close to map x for a throw from the ground at it to be a
    star: the claw would swipe it (reach.claw_edge, by half its width: `halves`, as half_of). By its middle
    alone (a star from 100 units, whatever the kind), the claw swiped pigs, slimes and Orange Mushrooms she
    counted on a star for (in the simulator, out to 118-124 units off with the claw's old reach, 90 ahead;
    by the client's, 65 ahead, out to 93-99)."""
    return abs(t.x - x) < reach.ground_sure(half_of(t.kind, halves))[0]


def _close_quarters(*args, **kw) -> tuple:
    """_close_quarters_here's act alone."""
    return _close_quarters_here(*args, **kw)[0]


def _close_quarters_here(x: float, close: list[Monster], mine: list[Monster], facing: str | None, lo: float,
                         hi: float, can_jump: Callable[[float, float], bool], tops: dict[str, float],
                         halves: dict[str, float] | None = None, ends: tuple[float, float] | None = None,
                         danger: Danger | None = None, platform: Platform | None = None,
                         around: list[Monster] | None = None, up: Callable[[float], Platform | None] | None = None,
                         y: float | None = None, pace: Pace = NO_PACE, moves: int = 0,
                         seen: list[Monster] | None = None, castable: Callable[[Monster], bool] | None = None,
                         clawed: bool = False) -> tuple[tuple, bool]:
    """A monster on her platform too close for a throw from the ground (`close`, as it will be when her next attack
    starts: too_close; `mine`: all on her platform), the nearest one then deciding (on the side she faces, if any will
    be there: one walking at her stops where it meets her, on its side, Pace's PACE_STOP; with FIRST_TO_REACH, the
    nearest the other way where the throws at both are sure and it reaches her first, or is more hurt, and the sure throw
    at it in place of an unsure jump in place at the one she faces). The throw from where she
    stands (_from_here, `pace`: where each will be when its attack starts) if its attack surely
    takes it: from the ground past the claw's reach, else a jump in place keyed so that it does (reach.air_sure: her feet
    then, its body on the slope as it stands then), at once if she faces it, else turning in the air where that picks
    it, or tapping the arrow on the ground first and jumping keyed early (JumpPlan air_turn False). Else a step away to
    where her next throw would surely take it, its walk meanwhile allowed for (_clear_step: against one walking at her
    no step does on these platforms, 54 units gained for each 149 walked); else a jump back, if there's room behind her
    and no monster there (its own attack, turning in the air, picks one on her level only from about 145 off: it buys
    room); else the throw from where she is all the same: the claw's swipe from the ground within CLAW_ONLY of her or
    under a platform overhead, else the jump in place. `lo`, `hi`: where she keeps to with monsters about (`ends`, with
    SNAIL_FAR_END, for a kind no jump in place picks on her level: a snail, on a short platform its far end). With
    `danger` (on `platform`, the monsters `around` her), one of a kind she keeps further off than the claw's reach
    (Danger.keep_off) is escaped as _escape has it (`up`: the platform overhead, stance's). `y`: her feet (map y; her
    platform's y_at at x by default). With `danger`, no step, escape or jump back ends on a spawn point of her platform
    the monsters the look saw (`seen`: `around` less her kill's, which may be the one due back there) don't account for
    (Danger.free_points: a respawn there lands on her), and after `moves` moves away in a row without a throw
    (MAX_MOVES) no step, an escape's neither (a jump back, which throws, still). A step away is ('back', side, units,
    AWAY). Without POINT_BLANK (the magician's) the throw from where she stands is a cast from the ground her search
    surely picks it with (_from_here), and with none, `castable` saying no to it, or her search not picking it then,
    no throw at it: a step, a jump back (the jump away alone) or ('wait',), never the claw's swipe nor a jump in place
    (which went out as a cast from the ground that picked nothing). `clawed` (stance's): no throw from the ground at it
    is sure (CLAWED_AIR: the claw takes it). Returns the act and whether it's the throw from where she stands whose
    attack surely takes nothing (_from_here's `sure` False: an unsure jump in place, a tap first or not, for stance's
    CLAWED_CLOSE; JumpPlan.sure is its jitter room's, not its target's: the review of 2026-11-03)."""
    at = pace.lead + GROUND_PICK_S  # where each will be when a throw from the ground would start (its side, how near)
    where = {id(t): pace.at(t, x, at, walks_on(t, platform)) for t in close}

    def side_of(t: Monster) -> str:
        return RIGHT if where[id(t)] > x or (where[id(t)] == x and t.x > x) else LEFT
    on_facing = [t for t in close if side_of(t) == facing]
    nearest = min(on_facing or close, key=lambda t: abs(where[id(t)] - x))
    if y is None:
        y = nearest.y if platform is None else platform.y_at(min(max(x, platform.left), platform.right))

    def throw_at(t: Monster) -> Here:
        args = (x, y, side_of(t), t, facing, can_jump, tops.get(t.kind, MONSTER_TOP["pig"]), half_of(t.kind, halves),
                mine, halves, platform, pace)
        h = _from_here(*args)
        if clawed and CLAWED_AIR and h.sure and h.act[0] == "throw":  # (a monster the looks miss on her: the claw's)
            h = _touched(_from_here(*args, ground_ok=False))
        return h
    here = throw_at(nearest)
    alt = None  # (FIRST_TO_REACH: the sure throw at the nearest the other way, for an unsure jump at `nearest`)
    if FIRST_TO_REACH and POINT_BLANK and platform is not None:
        other = [t for t in close if side_of(t) != side_of(nearest)]
        if other:
            o = min(other, key=lambda t: abs(where[id(t)] - x))
            there = throw_at(o)
            if there.sure and not here.sure:
                alt = there if here.act[0] == "jump_throw" else None
            elif there.sure:
                until = pace.lead + GROUND_PICK_S + IMMINENT_S

                def first(t: Monster) -> tuple:
                    return (_contact_s(t, x, pace, platform, halves, until), 1.0 if t.bar is None else t.bar,
                            side_of(t) != facing, abs(where[id(t)] - x))
                if first(o) < first(nearest):
                    nearest, here = o, there
    side = side_of(nearest)
    away = directions.opposite(side)
    back = directions.sign(away)
    room = (hi - x) if away == RIGHT else (x - lo)
    behind = [t for t in mine if 0 < (t.x - x) * back < JUMP_BACK_CLEAR]
    top = tops.get(nearest.kind, MONSTER_TOP["pig"])  # a kind of unknown size: a pig's
    half = half_of(nearest.kind, halves)
    others = mine if around is None else around
    if not POINT_BLANK and here.sure and castable is not None and not castable(nearest):
        here = Here(("wait",), False, here.gap)  # (her: no cast at one she won't cast at, however her search picks it)
    stepping = moves < MAX_MOVES
    spawns = ([(sx, half_of(kind, halves)) for sx, kind in danger.free_points(platform.id, others if seen is None
                                                                              else seen)]
              if danger is not None and danger.any and platform is not None else [])
    near_ground = reach.ground_sure(half)[0]
    touched = FIGHT_CLEAR and pace.window > CLEAR_MIN_S and danger is not None and danger.any \
        and danger.share(nearest.kind) > 0 and abs(where[id(nearest)] - x) <= half + HER_HALF + CONTACT_SLACK \
        and (nearest.bar is None or nearest.bar > KILL_BAR)
    if danger is not None and platform is not None and (danger.keep_off(nearest.kind) > near_ground or touched):
        act = _escape(x, side, room, can_jump, danger, platform, others, here, halves, up, stepping, spawns,
                      stay=CLEAR_STAY if touched else 0.0)
        if alt is not None and act is here.act:
            return alt.act, False
        return act, act is here.act and not here.sure
    if here.sure:
        return here.act, False
    jumpable = jump_reach(top, half) < near_ground
    if JUMP_THROW_SNAILS and not jumpable and jump_reach(top, half) < 2 * near_ground and can_jump(x, x):
        # (cornered or not, a kind the jump's attack picks at all near her: a snail from 143, not one 5 tall)
        return (("jump_throw", side) if facing == side
                else ("jump_throw", side, JumpPlan(physics.JUMP_ATTACK_MS, 0, False, "tap first"))), True
    far = room
    if ends is not None and not jumpable:
        far = max(room, (ends[1] - x) if away == RIGHT else (x - ends[0]))
    step = (_clear_step(x, y, nearest, away, far, top, half, platform, others, halves, pace, can_jump, spawns, danger)
            if stepping else None)  # (no endless backing off: her decisions are made again at each look)
    if step is not None and not behind:
        return ("back", away, step, AWAY), False
    if room >= JUMP_BACK_ROOM and not behind and can_jump(x, x + back * JUMP_BACK_SPAN) and jumpable \
            and not _on_spawn(x + back * JUMP_BACK_SPAN, spawns):
        return ("jump_back", side), False
    if stepping and not can_jump(x, x) and room >= STEP_ROOM and not behind \
            and not _on_spawn(x + back * min(STEP_AWAY, room), spawns):
        # A platform just overhead: a jump lands her on it (on 20, five jump-throws at a snail put her up on 19 five
        # times). Step away, else throw from the ground.
        return ("back", away, min(STEP_AWAY, room), AWAY), False
    if not jumpable:
        return (("throw", side), False) if POINT_BLANK else (here.act, not here.sure)  # (her: no claw to swipe it with)
    return (here.act, not here.sure) if alt is None else (alt.act, False)


def _contact_s(t: Monster, x: float, pace: Pace, platform: Platform | None, halves: dict[str, float] | None,
               until: float) -> float:
    """s after the look until monster `t`'s body meets hers at map x (Pace.meets: as `pace` walks it, on its own
    platform, walks_on), tried every ARC_STEP up to `until`; 0 if it's on her already, inf if not by then (it stands, walks
    away, or comes on too slowly): which of two monsters reaches her first (FIRST_TO_REACH)."""
    own = walks_on(t, platform)
    half = half_of(t.kind, halves)
    s = 0.0
    while s <= until + 1e-9:
        if pace.meets(t, x, s, own, half):
            return s
        s += ARC_STEP
    return math.inf


def jump_reach(top: float, half: float) -> float:
    """How far ahead of her (middle to middle) a jump in place keyed 40 ms after the jump key surely picks a monster on
    her level `top` tall and `half` wide either side, CLEAR_MARGIN farther (reach.air_sure; inf if within STAR_RANGE
    none is): a Green or Horny Mushroom 54, a slime 73, a pig 88; a snail 143, past the claw's edge (96): a jump in place
    never takes one the claw would swipe, so she doesn't jump at one (JUMP_THROW_SNAILS) nor back from one."""
    key = (top, half, physics.JUMP_ATTACK_MS, reach.AIR_MODEL, CLEAR_MARGIN)
    if key not in _JUMP_REACH:
        a = next((float(a) for a in range(0, int(reach.STAR_RANGE))
                  if reach.air_sure(a, 0.0, top, half, physics.JUMP_ATTACK_MS)), math.inf)
        _JUMP_REACH[key] = a + CLEAR_MARGIN
    return _JUMP_REACH[key]


_JUMP_REACH: dict[tuple, float] = {}  # (top, half, the jump throw's key, reach's air model, CLEAR_MARGIN) -> jump_reach


def _from_here(x: float, y: float, side: str, m: Monster, facing: str | None, can_jump: Callable[[float, float], bool],
               top: float, half: float, mine: list[Monster], halves: dict[str, float] | None,
               platform: Platform | None, pace: Pace, ground_ok: bool = True) -> Here:
    """The throw from map (x, y) on `platform` at monster `m`, on `side` of her, too close (_close_quarters), and whether
    its attack surely takes it where `pace` has it when the attack starts: from the ground (after a tap, TAP_S later,
    unless she faces `side`) if every monster of `mine` that way is then past the claw's edge (reach.claw_edge, by half
    its width: `halves`); else a jump in place (`can_jump`: none under a platform overhead) whose attack surely picks it
    then (reach.air_sure, CLEAR_MARGIN farther: `top` tall, `half` wide, its feet on the slope where it will be) and
    whose star's line to it is clear then (with reach.CLIENT_HIT_POINT: reach.wall_in_way at both ends of the decide
    lag, to the client's own hit point too: a monster at the foot of a wall rising from her platform's end, its body
    over it, takes no star from a jump in place within about 70 of it, the line going into the wall), keyed
    40 ms after the jump if she faces it, else turning in the air (physics.AIR_TURN_ATTACK_MS: her feet 52-56 up then,
    a mushroom on her level picked from 144 ahead) or tapping the arrow first and jumping once the tap's walk has
    stopped (physics.JUMP_THROW_TURN_MS, then 40 ms: JumpPlan air_turn False), whichever picks it, the air turn first.
    With none of those sure (`sure` False): within CLAW_ONLY of her, or with no jump, the claw from the ground; else the
    jump in place keyed 40 ms after the jump (tapping first if she must turn). Without POINT_BLANK (the magician's): a
    cast from the ground (a tap first if she must turn) where her search surely picks it when the cast starts, its body
    CLEAR_MARGIN past the search's start (reach.picks, by `top` and `half`, its feet on the slope then) and clear of
    hers (`half` and HER_HALF apart, middle to middle: the review of 2026-10-29, a Green Mushroom 33-49 off took a cast
    with their bodies overlapping, where master stepped back from 35), else ('wait',), not sure: no jump in place, no
    claw; a step away or the jump away comes first (_close_quarters). `ground_ok` False: no throw from the ground is sure
    (the claw takes it: stance's `clawed`, _from_air)."""
    d = directions.sign(side)
    turn = facing != side
    own = walks_on(m, platform)

    def gap(t: float) -> float:
        return (pace.at(m, x, t, own) - x) * d

    def drop(t: float) -> float:
        if platform is None or m.platform != platform.id:
            return m.y - y
        mx = pace.at(m, x, t, platform)
        return platform.y_at(min(max(mx, platform.left), platform.right)) - y

    tg = pace.lead + (TAP_S if turn else 0.0) + GROUND_PICK_S
    ground = ("throw", side)
    g0 = gap(tg)
    if not POINT_BLANK:  # (her search from the ground, or nothing: no claw, no cast in the air, none on her)
        sure = g0 >= half + HER_HALF and reach.picks(g0 - CLEAR_MARGIN, drop(tg), top, half)
        return Here(ground if sure else ("wait",), sure, g0)
    swiped = [o for o in mine if (platform is None or o.platform == platform.id)
              and -half_of(o.kind, halves) < (pace.at(o, x, tg, platform) - x) * d
              < reach.ground_sure(half_of(o.kind, halves))[0]]
    level = platform is None or m.platform == platform.id  # (one beside her off her platform: the star's own reach)
    if ground_ok and g0 >= reach.ground_sure(half)[0] and not swiped \
            and (level or _star_takes(platform, x, y, m, pace.at(m, x, tg, own))):
        return Here(ground, True, g0)
    jump = ("jump_throw", side)
    tap = ("jump_throw", side, JumpPlan(physics.JUMP_ATTACK_MS, 0, False, "tap first"))
    if not can_jump(x, x):
        return Here(ground, False, g0)
    if not turn:
        tries = ((jump, physics.JUMP_ATTACK_MS, 0.0),)
    else:
        tries = ((jump, physics.AIR_TURN_ATTACK_MS, 0.0),
                 (tap, physics.JUMP_ATTACK_MS, physics.JUMP_THROW_TURN_MS / 1000))

    def clear(k: int, t: float) -> bool:  # (the line from her feet to it as an attack keyed k starts, `t`: _jittered_up)
        return platform is None or all(
            reach.wall_in_way(platform, x, y, x + d * gap(t), physics.jump_rise(k + physics.JUMP_LAG * 1000 + lag), own,
                              y + drop(t), (top, half)) is None for lag in (0, physics.ATTACK_DECIDE_LAG_MS))

    for act, key, wait in tries:
        t = pace.lead + wait + jump_pick_s(key)
        if reach.air_sure(gap(t) - CLEAR_MARGIN, drop(t), top, half, key) \
                and (not reach.CLIENT_HIT_POINT or clear(key, t)):  # (its own key's line too: R6)
            return Here(_jittered_up(act, m, key, wait, gap, drop, top, half, pace, clear), True, gap(t))
    if g0 < CLAW_ONLY:
        return Here(ground, False, g0)
    act, key, wait = tries[-1]
    return Here(act, False, gap(pace.lead + wait + jump_pick_s(key)))


def _from_air(x: float, y: float, side: str, in_reach: list[Monster], facing: str | None,
              can_jump: Callable[[float, float], bool], tops: dict[str, float] | None,
              halves: dict[str, float] | None, platform: Platform, pace: Pace) -> tuple | None:
    """A throw that way at the nearest of `in_reach` on `side` of her from a jump in place whose attack surely picks it
    (_from_here without the ground: reach.air_sure, CLEAR_MARGIN farther, at once if she faces it, else turning in the
    air or tapping first, its jitter's room and line checked), for stance's `clawed` (a monster the looks miss on her:
    from the ground the claw takes the press); its plan says why ('touched'). None where no such jump is sure (under a
    platform overhead, a target out of the air's reach)."""
    way = [t for t in in_reach if _side_of(t, x) == side]
    if not way:
        return None
    m = min(way, key=lambda t: abs(t.x - x))
    top = (tops or MONSTER_TOP).get(m.kind, MONSTER_TOP["pig"])
    here = _touched(_from_here(x, y, side, m, facing, can_jump, top, half_of(m.kind, halves), [], halves, platform,
                               pace, ground_ok=False))
    return here.act if here.sure and here.act[0] == "jump_throw" else None


def _touched(here: Here) -> Here:
    """_from_here's answer without the ground (stance's `clawed`) with its sure jump's plan saying why ('touched', its
    own why after it: 'touched, tap first'): the 'jump' records' kind, and her log's."""
    plan = plan_of(here.act)
    if not here.sure or here.act[0] != "jump_throw" or plan is None:
        return here
    why = "touched" if plan.why in ("", "close") else f"touched, {plan.why}"
    return Here((here.act[0], here.act[1], replace(plan, why=why), *here.act[3:]), here.sure, here.gap)


# A jump throw up close's later keys (the jitter's room) have their lines to it checked too (_jittered_up), as reach.shots'
# picks are and the world's attacks: until the review of 2026-11-01 its pick alone (reach.air_sure), and on a slope her
# own floor or one just over her stopped the star of a later key where the key's hit (on Tree Dungeon's 33 at x 127, a
# Horny Mushroom 107 off, keyed 40 the world's star hit, 41-55 stopped: 22 of 290 of the world's close throws with a
# room, 185 of 4,150 of their keys), a pattern of the jitter's own making for the later analysis. Checked at both ends of
# the decide lag (physics.ATTACK_DECIDE_LAG_MS), as air_sure checks the pick: at its start alone, 2 of 290, 10 of 3,940
# keys; at both, none of 3,490 (D:/MapleHelper-scratch/replan/speed/fix/world_rooms.py). False: as before.
ROOM_LINE = True


def _jittered_up(act: tuple, m: Monster, key: int, wait: float, gap: Callable[[float], float],
                 drop: Callable[[float], float], top: float, half: float, pace: Pace,
                 clear: Callable[[int, float], bool] = lambda k, t: True) -> tuple:
    """A jump throw up close at `m` (_from_here's `act`, its attack keyed `key` ms after the jump key, the jump `wait` s
    after the plan's start: a tap first) with the keys its jitter may key it at (JumpPlan.room: behavior.fight.jittered,
    the user's "a bit jitter" of 2026-11-01, as her jump shots': reach.JITTER_FIRST_MS later at most, never sooner), those
    after it whose attack surely picks the monster too as `pace` has it walk (reach.air_sure, CLEAR_MARGIN farther, at
    every reach.KEY_CHECK_MS) and whose line to it is clear then (ROOM_LINE: `clear`, the key and when its attack picks;
    reach.wall_in_way from her feet at both ends of the decide lag), up to the first that doesn't. Its plan says 'close'
    (or its own why: 'tap first'), and that it's for `m`."""
    plan = plan_of(act)
    hi = key
    while hi + reach.KEY_CHECK_MS <= key + reach.JITTER_FIRST_MS:
        k = hi + reach.KEY_CHECK_MS
        t = pace.lead + wait + jump_pick_s(k)
        if not reach.air_sure(gap(t) - CLEAR_MARGIN, drop(t), top, half, k) or (ROOM_LINE and not clear(k, t)):
            break
        hi = k
    return (act[0], act[1], JumpPlan(key, 0, plan.air_turn if plan is not None else True,
                                     plan.why if plan is not None else "close", room=(key, hi), target=m))


def _star_takes(platform: Platform | None, x: float, y: float, m: Monster, mx: float) -> bool:
    """Whether a star from the ground at map (x, y) on `platform` takes monster `m` (off her platform: one beside her,
    BESIDE) where it will stand, at map x `mx` on its own platform (walks_on): reach.can_hit, in the star's search and its
    line not stopped (her own floor stops one under her unless she stands near enough its end)."""
    own = walks_on(m, platform)
    my = own.y_at(min(max(mx, own.left), own.right)) if own is not None else m.y
    return reach.can_hit(platform, x, y, replace(m, x=mx, y=my), target=own)


def _clear_step(x: float, y: float, m: Monster, away: str, room: float, top: float, half: float,
                platform: Platform | None, around: list[Monster], halves: dict[str, float] | None,
                pace: Pace, can_jump: Callable[[float, float], bool] | None = None,
                spawns: list[tuple[float, float]] = (), danger: Danger | None = None) -> float | None:
    """The shortest step away from monster `m` (on `away`'s side of her; `room` there, MIN_MOVE to STEP_AWAY) after which
    her next throw at it surely takes it, as `pace` has it walk meanwhile: the step's walk (physics.hold_ms) and
    STEP_SETTLE_S, then (facing away from it) a tap and a throw from the ground past its claw's edge (reach.star_sure),
    or the tap, physics.JUMP_THROW_TURN_MS and a jump in place keyed 40 ms after it (reach.air_sure, CLEAR_MARGIN
    farther, its line clear: _jump_clear; only where `can_jump` there). Never into a monster's body on her platform
    (_into; with `danger`, one
    beside her too: BESIDE), nor to within a respawn's body of one of `spawns` ((map x, half the width of its kind):
    spawn points of her platform no monster about accounts for, where a respawn lands on her). `m` on a platform just
    under or over hers walks its own (walks_on), and a star from the ground takes it only where it would take it there
    (_star_takes). None if no step within `room` does: against one walking at her (95 units/s) she gains 54 units for
    each 149 walked, less what it walks while she starts and stops."""
    d = directions.sign(away)
    own = walks_on(m, platform)
    level = platform is None or m.platform == platform.id
    most = min(room, STEP_AWAY)
    tried = [float(s) for s in range(int(MIN_MOVE), int(most) + 1, int(CLEAR_STEP))]
    if most >= MIN_MOVE and (not tried or tried[-1] < most):
        tried.append(float(most))  # (all the room there is, as _clear_from tries its far end)
    for step in tried:
        gx = x + d * step
        if platform is not None and _into(x, gx, platform, around, halves, danger):
            return None
        if _on_spawn(gx, spawns):
            continue
        done = pace.lead + physics.hold_ms(step) / 1000 + STEP_SETTLE_S
        star_t = done + TAP_S + GROUND_PICK_S
        if pace.meets(m, gx, star_t, own, half):
            continue  # (on her there when her throw starts: a touch, no clearance)
        mx = pace.at(m, gx, star_t, own)
        if (gx - mx) * d >= reach.star_sure(half) and (level or _star_takes(platform, gx, platform.y_at(gx), m, mx)):
            return step
        if not POINT_BLANK or (can_jump is not None and not can_jump(gx, gx)):
            continue  # (her: no attack from a jump, the step has to clear it for one from the ground)
        jump_t = done + physics.JUMP_THROW_TURN_MS / 1000 + jump_pick_s(physics.JUMP_ATTACK_MS)
        if pace.meets(m, gx, jump_t, own, half):
            continue
        mx = pace.at(m, gx, jump_t, own)
        dy = (m.y - y if platform is None or m.platform != platform.id
              else platform.y_at(min(max(mx, platform.left), platform.right))
              - platform.y_at(min(max(gx, platform.left), platform.right)))
        if reach.air_sure((gx - mx) * d - CLEAR_MARGIN, dy, top, half, physics.JUMP_ATTACK_MS) \
                and _jump_clear(platform, gx, mx, own, m.y, top, half, physics.JUMP_ATTACK_MS):
            return step
    return None


def _jump_clear(platform: Platform | None, x: float, mx: float, own: Platform | None, my: float, top: float,
                half: float, key: int) -> bool:
    """Whether the star of a jump in place from map x on `platform` (her feet on it there), its attack keyed `key` ms
    after the jump key, gets to a monster at map x `mx` (on `own`, its feet on it there; at map y `my` where that isn't
    known), `top` tall and `half` wide either side: its line clear at both ends of the decide lag (reach.wall_in_way, to
    the client's own hit point too), as _from_here's own key's (R6). Always, without reach.CLIENT_HIT_POINT (as before)."""
    if not reach.CLIENT_HIT_POINT or platform is None:
        return True
    y = platform.y_at(min(max(x, platform.left), platform.right))
    feet = own.y_at(min(max(mx, own.left), own.right)) if own is not None else my
    return all(reach.wall_in_way(platform, x, y, mx, physics.jump_rise(key + physics.JUMP_LAG * 1000 + lag), own,
                                 feet, (top, half)) is None for lag in (0, physics.ATTACK_DECIDE_LAG_MS))


def _escape(x: float, side: str, room: float, can_jump: Callable[[float, float], bool], danger: Danger,
            platform: Platform, around: list[Monster], here: Here, halves: dict[str, float] | None = None,
            up: Callable[[float], Platform | None] | None = None, stepping: bool = True,
            spawns: list[tuple[float, float]] = (), stay: float = 0.0) -> tuple:
    """From a monster on her platform that hits hard, on `side` of her, too near (Danger.keep_off): of a jump back
    (JUMP_BACK_SPAN away, throwing back; with JUMP_BACK_ROOM of `room` behind her), a step away (STEP_AWAY, or the
    room there is, STEP_ROOM at least) and the throw from where she is (`here`: _from_here's), the one leaving her the
    least danger (Danger.at, with the monsters `around` her and the spawn points they don't account for), the first of
    those of equal danger; the step, which doesn't throw, only if it's ESCAPE_THROW safer than the others, and so a
    throw from where she is whose attack wouldn't surely take it (`here.sure` False: a swipe, or a jump in place at one
    too near for the star's search). A monster behind her makes the way back worse by its own danger (with a Horny
    Mushroom on each side of her on Tree Dungeon's 23 the old rule, no monster within JUMP_BACK_CLEAR behind, let a jump
    back 1 time in 332), and a move whose way or landing would take her into a monster's body (any kind's, on her platform: _into, by
    half its width, `halves`) is no escape. Under a platform overhead (`up(x)`: where a jump from here lands her),
    the jump up onto it, throwing back at the monster as she rises, is one too, by the danger where she lands there
    (the monsters on hers that can't reach it count for nothing there: Danger.walk): on Tree Dungeon's 23 left of x 10,
    under 22, a jump back or a jump throw in place lands her on 22, so the old rule left her a step toward the drop at
    23's left end or a throw (all three of the live knocks off it, 2026-10-25, began there: DROP_ROOM). `room`: the
    room behind her within where her steps keep to (stance's `edges`, or `limits`). Not `stepping` (MAX_MOVES moves
    away in a row), no step. With ESCAPE_STEPS the step's every ESCAPE_STEP from STEP_ROOM up are weighed too, the
    safest end first (of equal danger the shortest). Neither a step nor a jump back ends within a respawn's body of one
    of `spawns` (_close_quarters': spawn points no monster seen accounts for). `stay`: the throw from where she is that
    much dearer (FIGHT_CLEAR: touched, the monster on her)."""
    away = directions.opposite(side)
    back = directions.sign(away)
    options = []  # (danger there, order, act)
    jump = x + back * JUMP_BACK_SPAN
    if room >= JUMP_BACK_ROOM and can_jump(x, jump) and not _into(x, jump, platform, around, halves, danger) \
            and not _on_spawn(jump, spawns):
        options.append((danger.at(platform.id, jump, around), 0, ("jump_back", side)))
    most = min(STEP_AWAY, room)
    steps = sorted({most} | ({float(v) for v in range(int(STEP_ROOM), int(most), int(ESCAPE_STEP))} if ESCAPE_STEPS
                             else set()))
    for step in (steps if stepping and room >= STEP_ROOM else ()):  # (the safest end of a step, ESCAPE_STEPS)
        if not _into(x, x + back * step, platform, around, halves, danger) \
                and not _on_spawn(x + back * step, spawns):
            options.append((danger.at(platform.id, x + back * step, around) + ESCAPE_THROW, 1,
                            ("back", away, step, AWAY)))
    options.append((danger.at(platform.id, x, around) + (0.0 if here.sure else ESCAPE_THROW) + stay, 2, here.act))
    if ESCAPE_UP and up is not None and (q := up(x)) is not None:
        options.append((danger.at(q.id, x, around), 3, ("jump_throw", side, JumpPlan(why=f"up onto {q.id}"))))
    return min(options, key=lambda o: (round(o[0], 6), o[1]))[2]


def held_step(x: float, platform: Platform, side: str, around: list[Monster], halves: dict[str, float] | None = None,
              danger: Danger | None = None, limits: tuple[float, float] | None = None) -> tuple | None:
    """A step of HELD_STEP away from `side` from map x on `platform`, where her casts that way took no effect from
    there (the magician's spot held: behavior.attacks.Spot; behavior.fight.cast_or_wait), new evidence for the next
    (her HOLD_MOVED along lifts it): within where she keeps to with monsters about (_stand_limits; `limits`, as
    stance's), into no monster's body of `around` on her platform (_into, `halves`), to no more danger than where she
    stands (`danger`: Danger.at). ('back', side, HELD_STEP), or None: then she waits (the review of 2026-10-29: with
    only held ones in reach she stood there)."""
    away = directions.opposite(side)
    goal = x + directions.sign(away) * HELD_STEP
    lo, hi = _stand_limits(platform, crowded=any(m.platform == platform.id for m in around))
    if limits is not None:
        lo, hi = max(lo, limits[0]), min(hi, limits[1])
    if not lo <= goal <= hi or _into(x, goal, platform, around, halves, danger):
        return None
    if danger is not None and danger.any and (danger.at(platform.id, goal, around)
                                              > danger.at(platform.id, x, around) + 1e-9):
        return None
    return ("back", away, HELD_STEP)


def _on_spawn(gx: float, spawns: list[tuple[float, float]]) -> bool:
    """Whether her body at map x `gx` is within a respawn's body of one of `spawns` ((map x, half its kind's width))."""
    return any(abs(gx - sx) <= sh + HER_HALF for sx, sh in spawns)


def _into(x: float, goal: float, platform: Platform, around: list[Monster], halves: dict[str, float] | None,
          danger: Danger | None = None) -> bool:
    """Whether a move along `platform` from map x to `goal` would take her into a monster's body: one of `around` on it
    ahead of her, up to where its body (half its width: half_of, and her HER_HALF) meets hers at `goal` (hunt's
    _in_the_way, for any kind: a touch, and its knock, whatever it takes); with `danger`, one on a platform just under
    or over hers whose body meets hers where it is too (`beside`, BESIDE: a kind that counts)."""
    for m in around:
        if m.platform != platform.id and not beside(m, platform, min(max(m.x, platform.left), platform.right), danger):
            continue
        r = half_of(m.kind, halves) + HER_HALF
        if (x < m.x <= goal + r) if goal > x else (goal - r <= m.x < x):
            return True
    return False


def _throw_side(x: float, in_reach: list[Monster], facing: str | None, platform: Platform | None = None,
                danger: Danger | None = None, hurt: Callable[[Monster], float | None] | None = None,
                imminent: Monster | None = None) -> str:
    """Which way to throw at the monsters in reach: toward `imminent` if it's one of them (_imminent: about to be too
    close); else, with `hurt` (her wounds: WOUNDS_FIRST), toward the one of them with the least of its HP left as she
    reckons it, then the nearer; else with `danger`, toward the nearest of them on her `platform` whose kind counts as a
    danger (it can touch her: live on 2026-10-25 her spot's own Green Mushroom on Tree Dungeon's 24 made 14 of her 34
    touches, 3 while she shot 23's Horny Mushrooms the other way); else the way she faces while that side has any, else
    the side with more of them, then the one with the nearer."""
    if imminent is not None and any(t is imminent for t in in_reach):
        return _side_of(imminent, x)
    if hurt is not None:
        wounds = [(share, t) for t in in_reach if (share := hurt(t)) is not None]
        if wounds:
            return _side_of(min(wounds, key=lambda w: (w[0], abs(w[1].x - x)))[1], x)
    if TOUCHER_FIRST and danger is not None and platform is not None:
        mine = [t for t in in_reach if beside(t, platform, x, danger) and danger.share(t.kind) > 0]
        if mine:
            return _side_of(_nearest(mine, x), x)
    if facing is not None and any(_side_of(t, x) == facing for t in in_reach):
        return facing
    right = [t for t in in_reach if t.x > x]
    left = [t for t in in_reach if t.x <= x]

    def score(group: list[Monster]) -> tuple:
        return len(group), -min((abs(t.x - x) for t in group), default=1e9)

    return RIGHT if score(right) > score(left) else LEFT


def _imminent(x: float, platform: Platform, around: list[Monster], danger: Danger | None, pace: Pace,
              halves: dict[str, float] | None) -> Monster | None:
    """The nearest monster on her `platform` whose kind can touch her (`danger`'s share over 0; any without one) that
    would be too close for a throw from the ground by the attack after her next (too_close, IMMINENT_S later: as `pace`
    walks it), or None: the danger her wounds don't come before (WOUNDS_FIRST)."""
    later = replace(pace, lead=pace.lead + IMMINENT_S)
    near = [t for t in around if beside(t, platform, x, danger) and (danger is None or danger.share(t.kind) > 0)
            and too_close(t, x, halves, danger, later, platform)]
    return _nearest(near, x) if near else None


def _step_from_under(x: float, y: float, platform: Platform, wanted: list[Monster], lo: float,
                     hi: float, around: list[Monster] = (), sizes: dict[str, MobBody] | None = None) -> tuple | None:
    """Off her level and too close sideways (right under or over her), a star misses: step away from
    the nearest such monster, along her platform, to where a star reaches it (reach.shot_dx, by its
    kind; throwing at it from right over it used to go on and on), or where its star isn't stopped if it is
    from there (_clear_from). None if there's none (one whose star terrain stops from all along her platform
    doesn't count), or no such step. `around`: as stance's; `sizes`: its body by the map's game data (stance's, for
    the magician), else reach's tables."""
    def goal_of(t: Monster) -> tuple[str, float] | None:
        away = LEFT if t.x > x else RIGHT
        goal = min(max(t.x + directions.sign(away) * (reach.shot_dx(t.y - y, t.kind, sizes) + 40), lo), hi)
        if reach.can_hit(platform, goal, y, t, reach=_goal_reach(), others=around, sizes=sizes):
            return away, goal
        goal = _clear_from(platform, x, y, t, lo, hi, around, sizes)
        return None if goal is None else (RIGHT if goal > x else LEFT, goal)

    under = [(t, g) for t in wanted if t.platform != platform.id and reach.in_band(t.y - y)
             and abs(t.x - x) < reach.shot_dx(t.y - y, t.kind, sizes) and (g := goal_of(t)) is not None]
    if not under:
        return None
    t, (side, goal) = min(under, key=lambda tg: abs(tg[0].x - x))
    if abs(goal - x) >= MIN_MOVE and reach.can_hit(platform, goal, y, t, reach=_goal_reach(), others=around,
                                                   sizes=sizes):
        return ("back", side, abs(goal - x))
    return None


def _close_in(x: float, y: float, platform: Platform, wanted: list[Monster], lo: float,
              hi: float, around: list[Monster] = (), sizes: dict[str, MobBody] | None = None) -> tuple | None:
    """Walk toward the nearest monster level enough to shoot, to SHOT_RANGE - 40 from it (never past it), or
    where terrain doesn't stop its star if it does from there (reach.wall_to, _clear_from: a wall of her
    foothold group does from anywhere on her platform, and then she doesn't walk for it). None if there's
    none, or it can't be brought into reach from her platform. `around`, `sizes`: as _step_from_under's."""
    def goal_of(t: Monster) -> float | None:
        goal = min(max(t.x - directions.sign(_side_of(t, x)) * (reach.SHOT_RANGE - 40), lo), hi)
        if reach.can_hit(platform, goal, y, t, reach=_goal_reach(), others=around, sizes=sizes):
            return goal
        return _clear_from(platform, x, y, t, lo, hi, around, sizes)

    level = [(t, g) for t in wanted if reach.in_band(t.y - y) and (g := goal_of(t)) is not None]
    if not level:
        return None
    t, goal = min(level, key=lambda tg: abs(tg[0].x - x))
    side = _side_of(t, x)
    if ((goal - x) * directions.sign(side) > MIN_MOVE
            and reach.can_hit(platform, goal, y, t, reach=_goal_reach(), others=around, sizes=sizes)):
        return ("approach", side, abs(goal - x))
    return None


def act_text(act: tuple, wanted: list[Monster], x: float, hidden: list[Monster],
             refused: tuple[str, Monster | None] | None = None) -> str:
    """A stance as a log line: "throwing right (pig +120, slime +300 hidden)". `refused`: why her cast goes at none,
    and the body it would have (the magician's: behavior.fight.cast_or_wait, behavior.attacks.Attacks.refused), for a
    stance that sends no cast: "no cast: no effect from here (green-mushroom +120)" for a wait (the review of
    2026-10-29: it said "nothing in reach" with a held one in reach), said after a move."""
    why = ""
    if refused is not None:
        m = refused[1]
        why = f"no cast: {refused[0]}" + ("" if m is None else f" ({m.kind} {m.x - x:+.0f})")
    if act[0] in _THROW_VERBS and refused is None:
        side = [t for t in wanted if (t.x > x) == (act[1] == RIGHT)]

        def name(t: Monster) -> str:
            return f"{t.kind} {t.x - x:+.0f}" + (" hidden" if any(t is h for h in hidden) else "")

        plan = plan_of(act)
        how = "" if plan is None else (f" {'forward ' if plan.way and act[0] == 'jump_throw' else ''}"
                                       f"at {plan.key_ms} ms{f', {plan.why}' if plan.why else ''}")
        return f"{_THROW_VERBS[act[0]]} {act[1]}{how} ({', '.join(name(t) for t in side[:3])})"
    if act[0] in ("back", "approach"):
        return f"{'stepping away' if act[0] == 'back' else 'closing in'} {act[1]} {act[2]:.0f}" + (
            f"; {why}" if why else "")
    if act[0] in _THROW_VERBS:  # (the magician's jump away: no cast in the air)
        return f"{'jumping back' if act[0] == 'jump_back' else _THROW_VERBS[act[0]]} {act[1]}; {why}"
    return why or "nothing in reach"


def pct(bar: float | None) -> str:
    """An HP bar for the log: "40%", or "-" when none is shown (unhurt)."""
    return "-" if bar is None else f"{bar:.0%}"


def pick_target(x: float, y: float, platform: Platform, targets: list[Monster],
                around: list[Monster] | None = None) -> Monster:
    """Monsters on her platform that are close first (they can reach her), then the most hurt one a star
    reaches from where she stands (finish what was started), then the nearest in reach, then the nearest.
    `around`: as stance's (`targets` by default)."""
    near = [t for t in targets if t.platform == platform.id and abs(t.x - x) < reach.SHOT_RANGE]
    if near:
        return _nearest(near, x)
    around = targets if around is None else around
    in_reach = [t for t in targets if reach.can_hit(platform, x, y, t, others=around)]
    hurt = [t for t in in_reach if t.bar is not None]  # finish the ones already hit first
    if hurt:
        return min(hurt, key=lambda t: (t.bar, abs(t.x - x)))
    return min(in_reach or targets, key=lambda t: abs(t.x - x) + 2 * abs(t.y - y))


def decide(x: float, y: float, platform: Platform, target: Monster, threats: list[Monster],
           halves: dict[str, float] | None = None, around: list[Monster] | None = None) -> tuple:
    """What to do about `target` from map (x, y) on `platform`: ('throw', direction),
    ('walk', direction, units), or ('skip', reason) when no spot on her platform is in reach (or terrain
    stops the target's star from all of them: a wall of her group, or a floor, her own for one below her
    unless she can stand near enough its end; or from every one a monster the attack takes first has its star
    stopped and none is MIN_MOVE or more away). `threats` are the other monsters on her platform; `around`:
    every monster around her (as stance's; `threats` by default). Never walks off her platform. One on her
    platform is too close within TOO_CLOSE, or while the claw would swipe it (close_to: by half its width,
    `halves`, as half_of)."""
    around = threats if around is None else around
    dx, dy = target.x - x, target.y - y
    mine = threats + ([target] if target.platform == platform.id else [])
    lo, hi = _stand_limits(platform, crowded=bool(mine))
    close = [t for t in mine if abs(t.x - x) < TOO_CLOSE or close_to(t, x, halves)]
    if close:
        # Step away from whatever is closest, unless that walks into another monster or off the end.
        nearest = _nearest(close, x)
        away = LEFT if nearest.x > x else RIGHT
        room = (hi - x) if away == RIGHT else (x - lo)
        blocked = any(0 < (t.x - x) * directions.sign(away) < STEP_AWAY + TOO_CLOSE for t in mine)
        if not blocked and room >= STEP_ROOM:
            return ("walk", away, min(STEP_AWAY, room))
        # Cornered: fight from here. Jumping over it put the rest of the group on her other side.
        return ("throw", _side_of(nearest, x))
    if reach.can_hit(platform, x, y, target, others=around):
        return ("throw", directions.toward(dx))
    if not reach.in_band(dy):
        return ("skip", "not level enough with her platform")
    # Walk to the nearest spot on her platform that has the target in reach (either side of it): near it,
    # 20 past where the claw would swipe it (reach.claw_edge, by its size; it was 150 for every kind), or off
    # her level one band of the star's search past where it first picks the target (reach.shot_dx, by its
    # kind: a pig standing needs that band more than one walking).
    far = reach.SHOT_RANGE - 30
    near = max(reach.claw_edge(half_of(target.kind, halves)) + 20,
               reach.shot_dx(dy, target.kind) + reach.SEARCH_STEP)
    goals = [min(max(target.x + d, lo), hi) for d in (-far, -near, near, far)]
    goals = [g for g in goals if reach.can_shoot(target.x - g, dy, reach=_goal_reach(), kind=target.kind)]
    if not goals:
        return ("skip", "out of reach from this platform")
    stops = [reach.wall_to(platform, g, y, target) for g in goals]
    goals = ([g for g, stop in zip(goals, stops) if stop is None]
             or [g for g in [_clear_from(platform, x, y, target, lo, hi)] if g is not None])
    if not goals:
        wall = stops[0].x1 == stops[0].x2
        return ("skip", "behind a wall of her platform's foothold group" if wall
                else "a floor in the way from all along her platform")
    # Of those, where no monster the attack takes first has its star stopped, if there's one; else the nearest
    # place on her platform where none has (_clear_from, with the monsters around her: one in her floor's shadow,
    # picked first from farther back, isn't picked from nearer her platform's end); else one at least MIN_MOVE
    # off, as monsters move. Already at one so shielded (she'd tap back and forth on the spot until the kill's
    # patience ran out), the target is out of reach from here for now.
    clear = [g for g in goals if reach.shielded(platform, g, y, target, around) is None]
    if not clear and (g := _clear_from(platform, x, y, target, lo, hi, around)) is not None:
        clear = [g]
    goals = clear or [g for g in goals if abs(g - x) >= MIN_MOVE]
    if not goals:
        return ("skip", "behind a monster whose star is stopped")
    goal = min(goals, key=lambda g: abs(g - x))
    if abs(goal - x) < 1:
        # At the spot already (her platform's end keeps her from a nearer one): the star gets there from here.
        return ("throw", directions.toward(dx))
    direction = RIGHT if goal > x else LEFT
    # A monster on her platform in the way gets shot first.
    ahead = [t for t in threats if 0 < (t.x - x) * directions.sign(direction) < abs(goal - x) + TOO_CLOSE]
    if ahead:
        return ("throw", _side_of(_nearest(ahead, x), x))
    return ("walk", direction, min(abs(goal - x), MAX_WALK))
