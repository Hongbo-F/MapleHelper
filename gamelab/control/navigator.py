"""Navigation: plan a route over the map (motion.route.Planner), carry it out like a player
(motion.flow.Flow: several moves in one continuous key plan), and replan from wherever she actually
ends up.

  look (dot + screen + monsters)  ->  plan (route.Planner)  ->  chain the moves (flow.Flow)
        ^                                                              |
        +--------------- check where she landed  <-  one key plan for the whole chain

One look per chain instead of several per move: she keeps running through landings, jumps on the way
and grabs ropes on the run. Moves the chain can't predict (climbing down a rope, a run-up she'd have
to back off for) are done on their own, closed loop (control.mover.Mover).

Monsters, without an escort (the default): the look before each chain covers the whole screen. A
chain stops before a leg a monster could touch her on (flow.danger: her body's box all along the way her
feet go, on the ground, in the air and on a rope, against each monster's wherever it can be by then, on any
platform); that move then costs extra in the route planner so a way around is taken if there is one, else she
waits for the monster to move off (a few looks, while she can stand where she is), else she evades (from 2026-10-28,
EVADE: a refuge along her platform or a move off it, waiting there while it stays safe, the contact rule), and then
goes past it as a chain planned without it, watched as any (never the mover for a move a monster is in the way of; the
mover's own moves watched too: WATCH_MOVES). A touch during a move (the feed's: her popup,
MISS or damage, perception.touches; else lost HP) cuts it short: she replans from wherever she was pushed to, without
counting it as a missed move, and while her invulnerability after it lasts no touch counts on the legs she plans
(IMMUNE: she may pass a monster then).

With an escort (hunting tours: behavior.fight.FightOnTheWay), each trip has a mode (go(mode=...)):
`hunt` (the default) shoots what's in reach before each chain and any monster in the way of a move along
her platform; `transit` (the hunt loop's trips) only shoots what's in the way. Either way her chains keep clear of
the monsters whose touch counts (the escort's `threat`: on Tree Dungeon the mushrooms, on the first map none) as they
do without an escort: a chain stops short of a leg one could touch her on, and she shoots it first if a star reaches
it from where she stands (the escort's `clear_for`), else the move costs extra so a way around is taken, else she
hops it (one in the way of a run along her platform), else waits for it to move off, then goes anyway. The monsters
for that are the whole state's: every one the look's frame had on screen (`trip_monsters`: since 2026-10-30 every frame
perception gives has the whole screen's, so a fight's band look keeps its band for the fights and the chains plan by all
of them), each walking the way perception says it heads at its kind's speed from the game data (`_walking`), and with
her memory of the monsters (`memory`: a hunter's) the ones it has there that the look couldn't have found, last seen
off its view, are kept clear of too (`remembered`). Without an escort the mode is `avoid`, as described above.

With a continuous feed (the hunting commands' default) her glances while a chain runs carry the monsters it follows: one
the chain's legs ahead of her now meet, that the look it was planned from hadn't, stops the chain as a hit does while she
runs along a platform with room to stop (REACT), and the next pass meets it.
"""

import math
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import ClassVar, NamedTuple, Protocol

from .. import character, directions, telemetry
from ..localization.mapdata import MapModel, Platform, Rope
from ..localization.memory import EDGE, MATCH_SLACK, MATCH_SPEED, MATCH_WALK_S, SURE
from ..localization.monsters import (DEFAULT_SPEED, MONSTER_HEIGHT, WALK_EDGE, Monster, MonsterTrack, Tracker,
                                     touch_box, tracks_record, walk_edges)
from ..localization.pose import ROPE_TOP_X
from ..localization.scene import Scene
from ..localization.rate import HZ
from ..localization.senses import Senses
from ..localization.tracking import HIDDEN, HIDE_S, LIVE
from ..localization.unseen import hidden
from ..localization.worldstate import to_scene
from ..motion import bodies as her_body
from ..motion import flow, physics, route
from ..motion.bodies import Climb
from ..motion.flow import (CLIMB_OVER, END_LINGER, MAX_LEGS, ROPE_SPEED, UP_EXTRA, Body, Flow, Leg, Span, chain_record,
                           ms)
from ..motion.physics import (CLIMB_SPEED, DROP_LAG, GRAVITY, JUMP_LAG, KNOCK_VX, KNOCK_VY, LAND_SLIDE, STOP_S,
                              WALK_SPEED)
from ..perception.framecheck import LookLost
from ..perception.kinds import kind_of
from ..perception.touches import INVULNERABLE_S, MISS
from ..motion.flow import Keys, _ground
from ..motion.route import Move, Planner, keeping_off, route_record
from . import client
from .actions import Actions
from .mover import MoveCut, MoveRefused, Mover, platform_text

ON_WAY_TRIES = 5    # monsters engaged per platform while passing through (respawns could keep her forever)
MAX_FIGHTS = 3      # fights in a row clearing the way, without a move between: then she moves on regardless
UNSHOT_S = 2.0      # guess: s a monster the shot first threw nothing at isn't tried again by a trip through her drops (_meet)
AVOID_SECONDS = 6.0  # extra route cost of a move a monster is in the way of (so a way around wins) ...
AVOID_MEMORY_S = 4   # ... for this long after the monster was seen there
# The extra route cost of a move the mover couldn't do from where she stood (TripState.cannot), for the rest of the trip:
# a climb up a rope ending 90-95 above her platform (route.GRAB_CHANCE: the route's reach, ROPE_REACH 95, against the
# mover's, ROPE_REACH less GRAB_MARGIN) can't be chained (no sure grab) nor done alone (mover.grab_lead). The quickest
# way never takes one on Tree Dungeon, Forest Up North II; with a jump from 30 up to 29 costing extra for a Green
# Mushroom on 29, a hunt loop's trip took the climb from 30 up the rope at -8 to 27, which ends 94 above 30, and the
# mover raised (simulator, loop_tree_walls seeds 1, 3, 4, 6, 7, 11).
CANNOT_S = 60.0
AROUND_TRIES = 8     # ways tried at most, with an escort, before the next look (Navigator._around)
WAIT_LOOKS = 6       # looks spent waiting for a monster to move off before going anyway
WAIT_S = 0.3         # between those looks
WAIT_AHEAD = 2.4     # with an escort, she waits only for a monster that, walking as seen, is clear of the move started
                     # this much later at most (about WAIT_LOOKS of waiting: Navigator._worth_waiting)
# With an escort, the longest her chains keep stopping short of the monsters whose touch counts (shots at them, ways
# around, waits) without a move that gets her where it was going: then she goes on regardless, as before her chains
# kept clear of them (Navigator._meet). From the floor of Tree Dungeon (33), whose Slimes and Greens respawn every 8.8 s
# among the crowd, a trip to rest spent its 180 s there shooting the Greens on 32 one after another, waiting, and
# fighting the Slimes in her way between (simulator, loop_tree_run), and she sat down where she was, among them. Counted
# again from every move she set off on, it still did: knocked back down to the floor each time, every pass there met
# a Green again, and each look again gave the Slimes in her way another fight (loop_tree_walls, seeds 1, 3, 5 and 10
# of 1-10, against none before her chains kept clear).
MEET_S = 8.0
# _meet takes a look with the monsters before going past one regardless, and shoots it if a star reaches it
# (_last_shot); off, as before. In the simulator it made no difference past the seeds' spread: 297.7 EXP a minute
# against 307.4 without, 0.92 bars of HP a minute either way, touches on her trips 2.38 a minute against 2.08, falls 2.1
# a run against 1.5 (Tree Dungeon, crowd, zero-walls, near-misses, 8 seeds of 5 minutes, standard errors about 5% of
# EXP). On because the live run's two touches after a pass regardless came from a Green Mushroom she had seen; to watch.
LAST_SHOT = True
# s: passes regardless of one move (its kind and platforms) within this of the last are one episode (the backstop
# event's `episode`: _pass_record), so their touches are counted once an episode; the analysis of 2026-11-02 counted the
# touches within 3.5 s of a pass as its (P04), and a cut-and-replan cycle passes the same move again within it.
PASS_EPISODE_S = 3.5
SETTLE_S = 0.35      # right after a plan she may still be sliding or landing this long
ARRIVE_TOLERANCE = 20  # map units: close enough to the x she's sent to
# Chains follow on from each other without a stop (follow): late in one, a look around decides whether
# another may follow; once she's back on the ground for good, a look places her and the next chain is
# planned from there and queued, to start the moment the running one ends.
LOOK_AHEAD = 0.9     # s before a chain's last landing that the look around is taken (~0.2 s; whole screen ~0.4)
CHECK_AFTER = 0.05   # s after she's back on the ground for good that the look placing her is taken ...
CHECK_LEAD = 0.25    # ... at the latest this long before the running plan ends: that look (live: 110 ms, 131 at
                     # p90, without the monster search), the next chain planned from it (~5 ms, 22 at p90) and
                     # sent (~8 ms) fit, with time for a second look if the first still caught her landing.
                     # (With the monster search the look took 205 ms live, and none of 15 chains followed on.)
RUN_ON_ROOM = 50     # units of run she keeps in hand before the route's next move when a plan runs on past
                     # its last leg (a running jump takes off up to flow.JUMP_FROM_FAR before its edge)
QUEUE_SLACK = 0.03   # s of timing error a chain planned from that look starts with
SEND_S = 0.03        # s before the running plan ends that the next chain must be sent by (planned: ~5 ms, 22 at p90)
MIN_GROUND_PX = 6    # half-size px (1920x1080; the look's geometry converts them) of ground under her feet that show
                     # she's standing (in the air: none)
END_WATCH = 0.1      # s before a plan with nothing queued after it ends: glances stop, its end is waited for ...
END_POLL_S = 0.005   # ... this closely (noticed late, the next look and plan came ~70 ms later)
FRESH_S = 0.3        # s a look stays good enough to plan from, if no key plan was sent since
AFTER_STILL_S = 0.35  # s at most waited for her to stand still after a chain, before the next pass's look
                      # (a stop slides her ~0.26 s; knocked about, she isn't held up longer)
PLAN_START_S = 0.015  # a chain's plan is taken to start this long after it's sent
GLANCE_S = 0.03      # between glances at her dot while a plan runs, or while she recovers
# What her glances show is judged by how long it lasts, not by how many glances show it, so it means the same however
# often she's glanced at (with a feed, as often as it looks: senses.FEED_GLANCE_EVERY_S). These were counts of glances
# at the pace glances had when they were set, 63 ms apart (eyes.GLANCE_EVERY_S): n of them are (n - 1) x 63 ms, less
# a game frame (16.7 ms), since each glance sees the newest frame when it's taken, up to a frame before.
OFF_PATH_S = 0.109   # s off the planned path, from the first glance in a row off it, that cut a plan short (3
                     # glances) ...
OFF_LEG_S = 0.046    # ... or this long once a glance among them had her where the leg doesn't go (2 glances)
STILL = 0.6          # minimap px: her dot within this on both axes of where it stood ...
STILL_S = 0.109      # ... in every glance for this long has stopped: she stands on the platform under it (3 glances).
                     # For 63 ms a dot can sit still in the air too: near the top of a jump, or at full speed within a
                     # minimap pixel (11-16 units; 11.4 on this map: 149 units/s covers 9 in 63 ms, 16 in 109)
RECOVER_STILL_S = 0.046  # s her dot stands still for `recover` to take it she's stopped moving (2 glances)
# Standing still where her run along a platform starts while the plan has her this far along it (Span.x_at) is
# off the path too. In game on 2026-10-21 chains whose first arrow went down 0.6 s after a throw's key set off
# about 0.2 s late, the jump at the end of the run went at its planned time from 15-30 units short, and she fell
# off: 8 of 10 jumps from 16 up onto 14 fell to 25. Her dot stood still at the start for the first 0.35-0.48 s
# while the plan had her running. (She speeds up from standing, so the plan's straight line runs ahead of her
# by up to 15 units early on; her dot is read to whole minimap pixels.)
OFF_SPAN = 30
HIT = 0.004          # HP lost (fraction) that counts as a hit, where no touch was told (a feed's states without one)
# A touch of hers (the feed's: Senses.touched_since, perception.touches, MISS or damage; README, Touches) told in a
# frame after the look a chain was planned from cuts it short as a hit always has, every key let go, and she replans
# from where she is: a MISS too, though it neither hurts nor moves her (it's where a monster's body met hers, and the
# next touch takes HP as often as not). Before 2026-10-28 only her HP dropping did, so a MISS (half her touches live)
# changed nothing and she went on into the same monster. Her HP dropping with no touch told (HIT) still does.
TOUCH_CUTS = True
# Her invulnerability after a touch (perception.touches.INVULNERABLE_S from its popup: Senses.invulnerable_until) in the
# chains planned and watched (flow.Flow.immune: no touch counts on a leg before then), less IMMUNE_MARGIN_S (a popup's
# frame comes up to a frame after the touch, a chain may start later than planned): a chain may pass a monster while it
# lasts, and a monster coming into its way then isn't one. False: as before, every leg against every monster.
IMMUNE = True
IMMUNE_MARGIN_S = 0.15
# The window is taken as the pass plans (its start then: now, or her attack lock's end, and PLAN_START_S), and the
# planning's own time comes out of it before the chain goes: the review of 2026-10-28 found the route to the chain 0.15 s
# or more on 24 of 239 chains of the 10-28 live tours (0.59 s at the most), so a leg could count on her being
# invulnerable up to 0.44 s after it was over. As it goes (`_immunity_spent`) the window is taken again, and a chain with
# a leg that meets one of the monsters it keeps clear of in the part the planning spent isn't sent: the pass looks again
# and plans with what's left ("her invulnerability spent planning ..."). False: the window as planned.
IMMUNE_RECHECK = True
# After a touch (a chain or one of the mover's moves cut short by one, or by her HP dropping) no fight starts on the way
# (the escort's shots in reach, `_shoot_on_the_way`, and in her way, `_clear_the_way`) and the trip's `until` isn't asked
# while her invulnerability after it lasts (TripState.post_hit_until: Senses.invulnerable_until less IMMUNE_MARGIN_S,
# INVULNERABLE_S from now where no popup was told): that window is what lets her chains pass the monster that just
# touched her (IMMUNE), and a fight then throws it away. Fighting from wherever a hit knocked her kept her fighting there:
# in the simulator a 12 s trip from 12 across to 17 (nav_fight_across) ran out its 180 s. Until 2026-10-30 the next pass
# after a cut took a quick look without monsters (TripState.quick) and planned its chain from the tracker's last
# monsters (Tracker.last), for one chain: live a median 1.2 s old (p90 3.3, max 4.8; 39 of the thief's 449 chains of
# 2026-10-30), the monsters the look's own frame had left out (D:/MapleHelper-scratch/motion/chain_looks_out.txt). Now
# the next chain is planned from the newest state's monsters, all of them.
POST_HIT = True
# A hit while she climbs a rope of a chain (_climbing) cuts the plan as any hit does, every key let go, up too, and the
# game doesn't knock her off a rope (live on 2026-10-26 she stayed on after 11 of 11 hits on ropes, the dot by the rope
# and still after each): she hung where it caught her. At 22:58:08.596 (Tree Dungeon, the rope at x 111 from 19 up to
# 18) a Green Mushroom walking on 18 touched her 47 under its top, 0.22 s before the plan had her there, and she hung
# about 30 under 18's surface for 25 s (lab/recent/20261026-225844-out-of-darts), her dot snapped onto 18 and her Lucky
# Seven presses refused there; that night 9 of the 56 climbs up that rope were cut by a hit, 5 of them left so, and
# 4 of the 27 up the rope at -8 (2). With CLIMB_ON up goes down again at once, held to the top and over it (_climb_on);
# off, as before. CLIMB_ON_X: map units her dot may be off the rope's x, and past its bottom; CLIMB_ON_UNDER: under its
# top she must be at least (half a minimap px: on the review's grinding run it went "up again" at -2 and -12, she was
# off the rope already, and the hold kept her by the mushroom on the platform).
CLIMB_ON = True
CLIMB_ON_X = 20
CLIMB_ON_UNDER = 8
# A chain whose glances had her dot where the first had it for STUCK_S or more (a jump or a run there moves it a minimap
# px within 0.1-0.25 s), or a move whose looks all had it where it began (Mover.still), from the top of a rope hanging
# from her platform: she's on the rope, not the platform (_rope_top). Of the chains the tour's telemetry has with 8
# glances or more where her dot never moved, 9 of 10 were at a rope's top on 18 (the other a drop at 00:19:07 on
# 10-26). ROPE_TOP off: as before (the hunt replanned from 18 there: 10 moves in 15.4 s at 22:19:38). STUCK_GAP: her dot
# more than this under the platform's surface too (map units): of the 51,391 glances of the tour's Tree Dungeon
# telemetry (10-24 13:49 to 10-26 23:47), those standing still 0.5 s or more within 16 of a rope's top read her dot
# 9.2 under the platform at the most (119 glances at 5 spots), the hangs 14.8, 19.3, 22.2 and 30.8
# (D:/MapleHelper-scratch/rope/review-numbers/glance_flips2.txt). A jump_up straight up from the rope's top that falls
# back where it began has her dot never move on the mover's settling looks either: standing, that's not the rope.
STUCK_S = 0.4
STUCK_GAP = 12
ROPE_TOP = True
# A chain from a rope's top cut short for her dot standing where its plan runs her ("off the path (still ...)", 0.12 s
# in) is a chain whose dot never moved too, however soon (ROPE_CUT, from the evade work's review, 2026-10-29): on the
# rope under 30's top at 146, every chain from there was cut so after 0.12 s, short of STUCK_S, "ended on platform 30
# instead", 112 times in 4 minutes (the harness's walk, seed 9, TOUCHER_ALL off). False: STUCK_S as before.
ROPE_CUT = True
# A drop that ended on no platform, on a rope (ROPE_DROP, from the evade work's review, 2026-10-29): it went down by a
# rope's top, of the platform she dropped through or the one she landed on with down still held, and onto the rope; the
# next pass planned the same drop from the same spot (the hunt loop's seed 10, from 28 at 146 onto 30 by its rope's top
# at 146: 12 times in 57 s, "ended on air/rope instead of platform 30", and seed 18's mover drop from 18 by the rope at
# 111, 11 times). It costs CANNOT_S for the rest of the trip, as a move the mover can't do does. False: as before.
ROPE_DROP = True
# A chain that ends in a drop, judged from a look of before the fall (DROP_RELOOK, 2026-10-31): `recover` takes the frame
# its glances saw her standing still in, and before a drop she lies still, down held, on the platform she drops through,
# so its look could be of that frame, her on it. "Ended on platform 28 instead of platform 30", and the next pass planned
# the same drop from 28 with her on 30 already: down + jump took her through 30 too, onto 32, or (at 146) down 30's rope
# by its top (loop-grind, Tree Dungeon's crowd: seeds 6, 9 and 10, 7 of the 9 misses of seeds 1-12 once the wall-slide
# walk-offs took her to 28). When the look has her on the platform she dropped through, it looks again DROP_RELOOK_S on,
# the fall over, and she's judged by that. False: as before.
DROP_RELOOK = True
DROP_RELOOK_S = 0.35
# What her dot shows of a chain's jumps and runs, told to the hunter's watch of her moves' keys (`moves`:
# behavior.keycheck.MoveWatch; the user, 2026-10-28: her keys moved in the game are followed): a planned jump's lift-off
# lifted her if a glance from it to LIFT_BY_S after has her dot LIFT_UNITS over the highest it had LIFT_BEFORE_S before
# (the dot lags: live it showed the jump's top 0.3-0.7 s after the lift-off planned, the telemetry's chains and glances of
# 2026-10-24 to 28: D:/MapleHelper-scratch/akey2/moves); a run on the ground RUN_UNITS long moved her if her dot went
# RUN_SEEN within it and RUN_AFTER_S after; a chain cut short for her dot standing where it runs her ("off the path
# (still at x ...)") didn't move her. Told only with GLANCES_MIN glances in a jump's window (3 in a run's), seen
# LIFT_SEEN_S (RUN_SEEN_S) before the chain was cut short, and not from a rope's top (her dot snapped onto the
# platform: _rope_top).
LIFT_UNITS = 40.0
LIFT_BY_S = 1.0
LIFT_BEFORE_S = 0.35
RUN_UNITS = 60.0
RUN_SEEN = 30.0
RUN_AFTER_S = 0.5
GLANCES_MIN = 2
LIFT_KINDS = ("jump", "jump_up")  # the moves whose jumps are told (a climb's grab of a rope goes with up alone)
LIFT_SEEN_S = 0.7     # a jump's window seen this long at least (a chain cut short sooner: not told) ...
RUN_SEEN_S = 0.5      # ... and a run's (her walk goes 75 units in it, RUN_SEEN 30), or one cut short once her walk
RUN_CUT_UNITS = 45.0  # would have gone this far (3 px of her dot): moved if her dot went half of it
# Looks at most, each after waiting up to 2 s for her to land and stand still (Mover.settle), for one that puts
# her on a platform, when a look shows her in the air: a slime's touch as she walked off platform 3 onto 4
# knocked her back into the 12-unit gap between the blocks at x 669-681, down which she fell to the basin for
# 2.5 s, and with one look the trip ended "lost: not on any platform" while she still fell (simulator,
# nav_monsters_long_way on seeds 2 and 4).
FALL_LOOKS = 2
# A fighting trip's monsters (Navigator.trip_monsters) are the whole state's since 2026-10-30 (the user: "make motion
# planing work on the 15fps whole screen result"): every frame perception gives has every monster on screen (localization.
# worldstate: one pass a frame), and a fight's band look carries those off its band (Scene.swept, Feed.look). Before, the
# band look's frame was searched again in the tour's process over the rows the chain's platforms take (Eyes.monsters, 23
# ms at 1366x768, as high as a jump over the highest and the tallest monster, as low as the lowest and the tallest under
# it: TRIP_LOOK_BELOW 15), on nearly every plan look, without the feed's track ids or which way they head; and off-band
# monsters on that frame (Scene.swept) went unread by the planner.
# A fighting trip's chains keep clear too of the monsters her memory has where the chain's legs go that the look didn't
# find (Navigator.remembered: Hunter.memory, localization.memory; at memory.SURE), seen within this long: where one can
# be by now is wherever it can have walked since at its kind's speed, either way (its walking not known), 200 units at
# 2 s. Longer shuts most of a platform to her chains for little (a guess). Those in view too, since 2026-10-26: live on
# 2026-10-25 a Horny Mushroom the looks had missed on 23 for 1.3 s (at 120-150 units, while she stood on 24) was dropped
# from the plan of her jump onto 23 because 23 was in view and searched, and she landed beside it (6 touches and a fall
# to 29 followed: the service's track 130; the detector finds a monster on her platform in 7-12% of frames within 50
# units, 44-55% beyond 100: localization.unseen). Before, with a look missing a monster 10% of the time wherever it
# was, the simulator had them in view too a cost: on Tree Dungeon (crowd, zero-walls, the regular attack, 8 seeds of
# 10 minutes) she was hurt on her trips 29.5 times a run against 28.3 without, lost as much HP on them (2.42 bars),
# and spent 12% longer on them (MEET_S fired 25 times against 9). Kept also: the feed's tracks out of sight (HIDDEN,
# their walking known: Navigator.hidden_tracks), and after a chain cut for a monster coming into its way, that
# monster's track for the next chain (TripState.came: the look after the cut can miss it as the one before did). With
# the detector's misses near her as live (sim.world.NEAR_SEEN_1025), these and the hunt loop's fights by what the looks
# missed (behavior.hunt.UNSEEN_FIGHTS) made 297.7 EXP a minute against 270.9 without, lost 0.92 bars of HP a minute
# against 1.04, and her trips were touched 2.38 times a minute against 3.25, their plans cut by a hit or a monster coming
# 8.9 times a run against 27.8 (in the simulator: Tree Dungeon, crowd, zero-walls, near-misses, `--attack choice`, 8
# seeds of 5 minutes; D:/MapleHelper-scratch/hunt-ab/B/ablate-a1-table.txt).
TRIP_MEMORY_S = 2.0
UNSEEN_TRIPS = True  # those in view, the feed's hidden tracks and the cut's track (above); off, as before
# How long a monster's walk as perception has it head (the feed's track: _walking, _track_of) is taken as seen; past that
# it may have turned or set off either way at its kind's speed (MonsterTrack.trust, reach). None: all along, as the
# tracker's measured walks were. Predicted from one detection on the user's own play it was 4.0 units off at 0.5 s
# (median; p90 37.6: localization.monsters.walk_vx), and the simulator's monsters stand 0.5-2.5 s and walk 1-4 s at a
# time; but 0.6 s made the harness's standard grind worse on seeds 13-24 (EXP a minute 305.8 against 325.2, touches 81.5
# against 75.6, on her trips 20.3 against 18.8: more chains stopped short and passed regardless; D:/MapleHelper-scratch/
# replan/rp/out/var_HEADING_TRUST_S=0.6_13.txt).
HEADING_TRUST_S: float | None = None
# A monster perception has standing or flinching (a stand or hit1 frame: monsters.walk_vx 0, under STILL_VX) is planned
# as one whose heading isn't known: either way at its kind's speed, STAND_UNSURE_S of it at the most (MonsterTrack.unsure),
# not as standing where it was (until the review of 2026-10-30 it was planned within TURN_SLACK and 20 units a second of
# its spot, 45 units at 1 s, where on the user's own play a standing monster set off at about 0.61 a second, 46% within
# 1 s, and the predictor was 68 units off at 1 s at p90: D:/MapleHelper-scratch/replan/p15/clip_motion_out.txt). Capped,
# since the chain's watch sees it set off at the next state and reacts (_known: not as the plan had it): a jump's flight
# and the reaction (a state, the splice's lead, the game's key effect) take about this long, where nothing she does can
# keep clear of it. None: all along, as a look's monster seen once.
STAND_UNSURE_S: float | None = 1.0
# With a continuous feed (Senses.glanced) her glances while a chain runs carry the monsters it follows, re-found in
# every frame with which way and how fast they walk (localization.tracking). A monster one of the chain's legs ahead of
# her now meets (flow.danger over the rest of the plan, REACT_LEAD_S on at the soonest), of the kinds the chain keeps
# clear of (with an escort, those whose touch counts: Escort.threat; without, any), cuts the plan short while she's on
# the ground, as a hit does: the next pass looks with the monsters and meets it (a shot, a way around, a wait). Only one
# the look the chain was planned from hadn't (walked in, spawned, or off that look's view: Navigator._known): one it had
# was planned against, or past on purpose (after waiting for it, or regardless: _meet). Only while she runs along a
# platform with room to stop on it (_room_to_stop): not in the air, on a rope or at a jump's lift-off, where letting go
# of the keys changes nothing she can use or drops her; once per monster (the feed's track) in REACT_AGAIN_S. The glances
# keep their pace and chains follow on as before (the hunt loop with the feed on Tree Dungeon, 4 seeds of 5 minutes: 2
# chains followed on either way; 51 stops, 122 chains against 89).
REACT = True
REACT_LEAD_S = 0.25  # s: a stop from a run slides her about 0.26 s (AFTER_STILL_S), so a touch sooner comes anyway
REACT_AGAIN_S = 3.0  # s
# A monster whose replan was dropped as too late with no room to stop (SPLICE_LATE_S) is asked about again at the next
# state LATE_RETRIES times in a row; dropped late again, it's marked for LATE_AGAIN_S, then asked about again (the review
# of 2026-11-01, its finding 3: the load that makes a replan late makes the next one late too, and every state, 67 ms
# apart, paid a whole replan while the monster came, its touch's and off-path checks waiting). None of the harness's
# runs has dropped one at 1x or 3x of its computing charged to the clock (--think-clock): only its tests do.
LATE_RETRIES = 1
LATE_AGAIN_S = REACT_AGAIN_S / 2  # s
STILL_VX = 15.0      # units/s: a followed monster slower than this stands (tracking's velocities, filtered frame to frame)
REACT_ROOM = 50.0    # units of platform ahead of her run she needs to stop on (her 149 units/s for AFTER_STILL_S's
                     # slide: 39, and a little)
# Replanning at every state while a chain runs (REPLAN, from 2026-10-30; the user: "ideally we probably want to regenerate
# motion plan at every frame so any hit/new spawn will cause motion plan to react immediately"). Every state perception
# gives (localization.rate.HZ: 15 a second, each with the whole screen's monsters) is a glance of the chain's watch
# (_watching), and on it:
# - a monster the running chain's legs still to come now meet (REACT's: one the plan didn't know, walked in or spawned),
#   wherever she is, has the chain replanned from where she'll be when new keys can act (_splice_point: SPLICE_LEAD_S on,
#   on the ground where the state has her; in a flight or past a takeoff's key, once she's down: the next state asks
#   again) against the whole state and handed over in place of the running plan's rest (_take_over: client.splice, the
#   arrow held kept held, no stop; a player without splices: cut and sent at once); only where nothing chains does she
#   stop as before (REACT's cut, with room to stop);
# - every BETTER_EVERY_S, the chain replanned so, taken where the running chain stops short (a monster, a move it can't
#   predict, cut off for length with nothing to follow on) and the new one goes on past where it stops (a chain that
#   stopped short of a monster gone since runs on), or where its route isn't the running one's and the route planner
#   costs it SWITCH_GAIN_S less; SWITCH_MIN_S after the last switch at the soonest; never the same way over again (the
#   same moves of the route: a replan one state later took the same route 99.2% of the time on the user's own play,
#   its takeoffs moving by the flow's variety, which isn't a reason);
# - a touch still cuts the plan (the game knocks her about whatever keys are held), and the next pass plans at once
#   from the first state that has her on the ground again (KNOCK_REPLAN: _knock_landing), not after she has stood still
#   and a look of its own (recover).
# Only a pass's own chain is replanned so (not an evade step's, a pass regardless or a hop over a monster: what they
# leave out on purpose, the replan wouldn't). The splice: control.client.splice, control.player (Player.run); the
# simulator's fake player as the real one (sim.fakes.FakePlayer.splice). False: as before.
REPLAN = True
STATE_S = 1.0 / HZ           # s between perception's states (15 a second)
SPLICE_LEAD_S = 0.04         # s after now new keys can take over at the soonest: the player sees a splice within its wait
                             # step (player.WAIT_STEP_S, 15 ms) and its last spin (FINAL_S, 20 ms), the file ~2 ms
KEY_COMMIT_S = 0.068         # s: a takeoff whose key goes this little after a splice's time is the game's already (it
                             # acts on a key 66-68 ms after it's sent: senses.KEY_EFFECT_S): the splice goes from where it
                             # lands her
KEY_LEAD = {"jump": JUMP_LAG, "jump_up": JUMP_LAG, "hop": JUMP_LAG, "climb": JUMP_LAG, "drop": DROP_LAG,
            "walk_off": STOP_S}  # s from a leg's takeoff key (a walk-off's: the arrow let go to stop short) to its flight


@physics.follows
def _follow() -> None:
    """Her walk and climb as set (physics.set_speed, which rebinds the names taken from it): flow's climb up
    (ROPE_SPEED, flow's own follower's), a walk-off's lead (KEY_LEAD) and a touch's knock standing (KNOCK_REACH: its
    landing slide is her walk's; the final review of 2026-11-02 found it left at the import's) made again of them."""
    global ROPE_SPEED, KNOCK_REACH
    ROPE_SPEED = flow.ROPE_SPEED
    KEY_LEAD["walk_off"] = physics.STOP_S
    KNOCK_REACH = KNOCK_PUSH * 2 * KNOCK_PUSH / GRAVITY + physics.LAND_SLIDE


BETTER_EVERY_S = 5 * STATE_S  # s between looks for a better way (a guess: 3 a second)
SWITCH_GAIN_S = 0.25         # s sooner a better way must get her there (a guess) ...
SWITCH_MIN_S = 0.5           # ... and s after the last switch at the soonest (a guess)
SPLICE_DRIFT = 20.0          # map units: a replan goes from where the state has her if she's this near where the plan has
                             # her (else her off-path checks deal with it: OFF_SPAN)
# Where a running chain has her at a splice, on its leg's run from the leg's start with the arrow held all along: by the
# chain's own physics (flow._ground: the lag before the game sees the arrow, her speeding up from what she had), not the
# run's span (a straight line from where it starts to where it ends, for monster checks: its mean speed and places off
# by up to RUN_LAG of a run, 18 units, early on). From the review of 2026-10-31: the splice's start was the span's, and
# 10 of the 28 trip landings elsewhere than planned in the harness's standard grind (48 seeds) came within 3 s of one
# (the speed it gave her off the physics' by a median 29 units a second, 112 at p90); with this and KNOCK_STILL, 2 of 8,
# master's 8 in all. False: the span's, as before.
SPLICE_PHYSICS = True
SPLICE_LATE_S = 0.05         # s: a replan done so late that the player would take it this much past its plan time 0, or
                             # past a takeoff's key of the running plan's (KEY_COMMIT_S), is dropped (_switch, _dropped:
                             # for a monster coming she stops for it where there's room, else the next state asks again,
                             # the monster not taken as handled; a better way's next look asks again): its start would
                             # be behind her (7.5 units at her run), or the takeoff's key let go by the splice while its
                             # plan has her running on (the review of 2026-10-30: a replan slower than about 108 ms;
                             # measured p99 36 ms, the most 51 at Idle priority)
KNOCK_REPLAN = True
KNOCK_AIR_S = 2 * KNOCK_VY / GRAVITY - STATE_S  # s after a touch's popup her knock's flight lasts on flat ground (0.22 s)
                                                # less a state: no state sooner has her landed
KNOCK_LAND_S = 0.8           # s at most a landing is waited for (then as before: recover)
LANDED_DY = 6.0              # map units: on a platform at the height the state before had her, she's landed
# ... and planned from once her slide has ended (KNOCK_STILL, from the review of 2026-10-31): a knock lands her going
# ~140 units a second away from the monster, and she slides ~19 units braking, where the chain is planned from standing.
# Planned from the first state on the ground, 73% of the 540 landings planned from in the harness's standard grind (48
# seeds of 5 minutes) had her going over 50 units a second (a median 108, the world's), and 18 of her trips' 28
# landings elsewhere than planned came right after one (the jump off 16 for 15 run off 16's end and down to 20, 421
# units, twice on seed 39), where master's 8 in all. A state on the ground within KNOCK_STILL_DX of the one before
# (45 units a second: the last fifteenth of a second of the slide's braking) has it over. False: the first state on
# the ground, as before.
KNOCK_STILL = True
KNOCK_STILL_DX = 3.0         # map units from one state to the next (15 a second)
COMING ="a monster coming into its way"  # (the cut's reason starts so: Navigator._watch, _carry_out_chain)
ASKED = "the trip's until, at a landing"  # (Navigator._landing_calls: no miss either)
TOUCHED = "touched"  # ... and a touch's (or her HP dropping): _touch_text
# Landings with clearance (from 2026-10-27; the user: "the character should actively try to move away from the mob to
# attack it"). With an escort that prices landings (Escort.landings: attack.stance.landing, what her first attack where a
# leg lands surely takes, the monsters walked on to then), a chain's moves are done where they land the cheapest
# (flow.chain's `price`: their other takeoffs, the route's other moves onto the same platform, the walk on to where she's
# sent counted; a landing she runs on from by its touch alone), and where a trip's last landing still has a price the
# monster making it so is shot first from where she stands, else from along her platform within SHOOT_WALK
# (LANDING_WALK), only where her attack reaches it (Escort.shot_place: from 2026-10-28, when live from 21 one on 23, past
# a star's band, was said to be shot first and nothing was thrown; from where she stands from the ground or a jump, from
# along her platform from the ground with SHOOT_SLACK either way; a try that threw nothing, Escort.throws, not a shot),
# else waited out if its walk clears the landing within
# WAIT_AHEAD (LAND_WAITS looks at most), else she goes (_clear_landing; not _meet's: walking to shoot first there, 25 up
# onto 24 with a Green Mushroom by 24's middle was hurt 14 times in 8 seeds and 56 s on its trips where master's stack
# 8 and 17 s, D:/MapleHelper-scratch/land/fix/shootfirst.py). Live (the 14 Tree Dungeon hunts of 2026-10-25 23:43 to
# 10-26 23:47:
# D:/MapleHelper-scratch/land/diag-landings/trips.py) 520 of her trips' landings came within 105 of a monster on the
# platform she landed on; at 50% of them _meet had planned the chain as if it weren't there (NO_WAIT 243 times,
# WAIT_LOOKS 17), at 42% she had waited for it, at 4% shot it first; 43% of the not-sure ones could have landed sure by
# the same kind of move from another takeoff (a median 49 units over), 12% by a star from her takeoff, 20% by one from
# along her platform (a median 53 units' walk), 6% by waiting up to 3 s. False: as before.
LANDING_CLEAR = True
LANDING_WALK = True  # ... from along her platform, where no star reaches it from where she stands; False: not walked to
SHOOT_WALK = 150.0   # units: the longest walk along her platform to shoot the monster by a landing first (1.0 s: a
                     # guess; live a median 53 units' walk would have done, diag-landings)
LAND_WAITS = 4       # guess: looks spent in a row waiting for one to clear a landing (WAIT_S apart), before going anyway
# _meet's shot first at a monster in the way of the route's first move, where no star reaches it from where she stands:
# from along her platform within SHOOT_WALK, where her attack reaches it (Escort.shot_place, shoot_from: as the landing's
# shot first), before a way around, a hop or a wait (the user, 2026-10-30: "in general you should kill all the mobs that
# are in the way"; since her chains are planned by the whole state's monsters, MEET_S fired 68 times in the harness's 24
# seeds where it hadn't, NO_WAIT 909 against 419: what's in her way now is seen, and more of it out of a star's reach from
# where she stands: D:/MapleHelper-scratch/replan/rp). False: from where she stands alone, as before.
MEET_WALK = True
# A chain planned on the move (_queue_next) prices its moves' landings by their own ways and the route's other moves
# between the same platforms only, not wider takeoffs (flow.chain's `spread`): it must be planned before the running
# plan ends (SEND_S), and with them a priced chain took a median 5.7 ms more, 20 at the most, 48 with the suite running
# alongside (the reviews of 2026-10-27, review-behaviour/chaintime.py), against SEND_S's 30; without them 0.7 ms, 3 at
# p90, 3.9 at the most, with the harness running alongside (D:/MapleHelper-scratch/land/fix/chaintime.py, SPREAD=0, where
# with them 10, 45 and 62). True: wider takeoffs too.
QUEUE_SPREAD = False

# The evade step (from 2026-10-28; the user, watching a movement-only walk: "Looks like there was no attempt to avoid
# hitting the mobs", "there was no attempt to even evade"). Live in walk B (D:/MapleHelper-scratch/qawalk/report) 57% of
# her 148 touches came after _wait_for_monster handed a move a monster was in the way of to the mover (control.mover:
# it doesn't look at monsters, nothing watched it, no touch stopped it), where her spot was judged unsafe to wait on or
# her WAIT_LOOKS ran out; she never stepped back or turned away. Now a move she can't wait for (her spot unsafe, the
# waits spent, or waiting can't clear the way: _worth_waiting) gets, in order (_evade, `evade` events and log lines):
# the contact rule (CONTACT, below); a refuge: the nearest spot along her platform within EVADE_WALK, EVADE_STEP_MIN
# away at least and EVADE_IN in from its ends, that a step there and standing there EVADE_HOLD_S (and to her
# invulnerability's end and STAND_PAST_S, R3) keep clear of every monster (flow.danger: the step's checks start after
# her invulnerability too, so right after a touch the step may pass the monster), else a single move off her platform
# that chains clear and lands where she can stand as long, the trip EVADE_MOVE_S dearer at most (a retreat after a stop:
# no bound), the quickest; ties between spots as near broken by the hunter's danger model (attack.danger, spawn points
# too) where the navigator has it (`danger`); a step to a refuge is a chain watched as any (a touch cuts it), and she
# looks again from there, waiting WAIT_S between looks while it stays safe, another refuge only once it doesn't
# (EVADE_REFUGES a blocked move at most); EVADE_S from the first refuge decision (a retreat: the time it has left), or
# no refuge, or EVADE_DEADLINE_S left of the trip, and she goes past (_go_past: the chain planned without the monsters
# in the way of the move, one at a time, PASS_TRIES at most, else blind; said, an `evade` event 'pass' and a backstop
# event NO_REFUGE, EVADE_S, EVADE_REFUGES or DEADLINE), and the mover only for a move no chain can plan at all, watched
# (R2: Mover.watch). With an escort (_meet) the same refuge before her last shot and the pass regardless. In the
# simulator's prototype (D:/MapleHelper-scratch/evade/design, proto2.py: a walk of random trips as live, Tree Dungeon's
# crowd, walkers, 8 seeds of 5 minutes) the bundle (refuges, refuge moves 3 s dearer at most, the planner's monsters as
# below, the contact rule) cut her touches by 16.2 a run of 119 (se 3.5), her touches within 2 s of the one before by
# 16.9 of 74, and her arrivals from 118 of 138 trips to 106 of 131; refuge moves of any cost cut 22.4 but arrived 72 of
# 115; keeping away up to 8 s doubled her touches in the walk's replays (hence the short budget). False: as before.
EVADE = True
EVADE_WALK = 150.0     # units along her platform a refuge spot may be (1 s of walking)
EVADE_STEP_MIN = 8.0   # ... this far from where she stands at least (here was judged unsafe)
EVADE_IN = 22.0        # ... and this far in from its ends
SPOT_STEP = 5.0        # ... the spots tried this far apart, the nearest first
EVADE_HOLD_S = 1.5     # s she must be able to stand where a refuge takes her, once there
EVADE_MOVE_S = 1.5     # s a refuge move off her platform may cost the trip more at most (math.inf: no bound)
REFUGE_MOVES = True    # refuges off her platform too (False: spots along it only)
EVADE_S = 3.5          # s a blocked move gets from its first block (BUDGET_FROM_BLOCK), its waits in it: then past
EVADE_REFUGES = 4      # refuges a blocked move gets at most
EVADE_DEADLINE_S = 4.0  # s: a trip with this little left goes past at once ...
HURRY_SLACK = 0.0       # ... or with less left than its route's time (Move.seconds) this many times over and that
                        # (0: off)
PASS_TRIES = 3         # monsters left out of a pass one at a time, before it's planned blind
WORTH_WAIT = True      # without an escort she waits only while waiting can clear the way (_worth_waiting)
SAFE_PASS = False      # safe where she stands, waits spent or no use: past it at once (a refuge only from an unsafe spot)
BUDGET_FROM_BLOCK = True  # EVADE_S counted from the move's first block, its waits in it (False: from its first refuge)
AROUND_ALL = True      # R4: a way around costs every move of it the same monsters block (_around), not the first's alone
PASS_COMMIT_S = 5.0    # s after a pass no refuge move takes her back onto the platform she passed from (she went
                       # up past a monster, met another there, and a refuge move took her back down: to and fro)
# R3: the stand check behind waiting (_safe_to_stand) runs to her invulnerability's end and STAND_PAST_S more, at least
# 1 s: with a second or more of it left, a second's check was vacuous (no touch counts before its end) and she
# "waited" in contact. The contact rule (CONTACT): where a monster seen live (its sighting CONTACT_LIVE_S old at most)
# touches her within CONTACT_S whatever she does now (standing, or a run either way), and she isn't invulnerable, that
# touch is taken as given, and the chain planned with the window that follows it (its time + INVULNERABLE_S, less
# IMMUNE_MARGIN_S): she's got clear of it within the window instead of stalling where she stands; once in
# CONTACT_AGAIN_S unless a touch was told since (the prototype: 3.6 fewer touches a run, 4.4 fewer touches again).
STAND_PAST_S = 0.5
STAND_WINDOW = True  # (R3's stand check to the window's end; False: a second, as before)
CONTACT = True
CONTACT_S = 0.25
CONTACT_LIVE_S = 0.5
CONTACT_AGAIN_S = 2.0
# R5: the planner's monsters without an escort (UNSEEN_AVOID), as with one since 2026-10-26 (UNSEEN_TRIPS): her memory's
# where the chain goes (`remembered`), the feed's tracks out of sight (`hidden_tracks`) and the last cut's monster
# (TripState.came); each look's monster walking as the feed follows it where the look's tracker had none (51% of the
# danger entries in walk B); and a monster the look placed on no platform (in the air, a slime's hop: walk B's 345)
# placed on the platform under its feet (PLACE_REACH). The ghost guard: a track out of sight on her (its box on hers)
# with no touch told in GHOST_S, while she can be touched, is none (walk B: half the chains flagged had a track sitting
# on her at the plan's start; with every track, 127 of 210 chains planned nothing: a stall). False: as before.
UNSEEN_AVOID = True
PLACE_REACH = 40.0
GHOST_S = 1.0
# R2: the mover's moves (those no chain plans) watched as a chain is (Mover.watch): cut on a touch told since the move
# began (a MISS too), her HP dropping, or her standing on a platform the move neither leaves nor goes to (MOVE_OFF_S);
# the navigator replans from a quick look. Its run-ups and back-offs, and the walk to where she's sent on arrival, are
# checked against the monsters (Mover.clear): a back-off into one climbs from under the rope instead (a climb), or the
# move is refused and costs AVOID_SECONDS as a monster in its way does (not CANNOT_S). False: as before.
WATCH_MOVES = True
MOVE_OFF_S = 0.109
REFUSALS = 3  # a move refused this many times in a row (no refuge meanwhile, a wait each) goes unchecked
# R7: a trip that dodges from a touch (go's `toucher`: her retreat, behavior.retreat) keeps clear of the monster that
# touched her as its first chains plan: where no monster of the look is on her platform within TOUCHER_NEAR of her, a
# stand-in for it at her x (its walking not known), of the map's kind that touches hardest (PADamage), while the touch
# (a look's or glance's, not `crowded`) is TOUCHER_S old at most. In the prototype's retreats after the safety stop
# (D:/MapleHelper-scratch/evade/design/trace_loop2.txt) the first chain off 8 was planned without the Horny Mushroom
# that had touched her three times, hidden under her sprite, and it touched her again 0.4 s later.
TOUCHER_S = 2.0
TOUCHER_NEAR = 40.0
# R6: before a chain queued on the move whose first leg is a drop or a climb starts (2 glances before), its flight and
# landing, or its climb and the rope's top, are checked against the feed's newest tracks (the glance's: live ones, and
# those out of sight QUEUE_SEEN_S at most): one that meets it withdraws it, and she stops where the running one ends
# (QUEUE_CHECK). After a touch on a rope (CLIMB_ON), the rope's top is looked at first (CLIMB_LOOK): a monster the feed
# has there by the time she'd be up, and she goes down the rope again instead of up into it. False: as before.
QUEUE_CHECK = True
QUEUE_SEEN_S = 1.0
CLIMB_LOOK = True
# ROPE_HANG (from the three recordings of RiceBaby grinding on Tree Dungeon on 2026-11-02, the roadmap's item 9, P08:
# D:/MapleHelper-scratch/rec-analysis/plans): while she climbs a rope up to the platform at its top, a chain's climb or
# the mover's lone climb move, her glances check the way up from the newest state: where the rest of the climb, the
# climb-over and STAND_PAST_S standing at the top would meet a monster (_way_up: flow.danger, her box all the way,
# against every monster the state has whose touch counts, on any platform, each walking as followed, her invulnerability
# counted), she stops on the rope HANG_UNDER under its top (the cut lets go of every key; the game keeps her where she
# is on a rope) and hangs there (_hang), where no monster's body reaches her, looking at every state; once the way up is
# clear she climbs the rest, as her climb on after a touch holds it (_climb_on), and after HANG_S she goes up regardless
# (a backstop event, as a pass regardless). A touch on the rope with the top taken (CLIMB_LOOK) hangs too, down to
# HANG_UNDER first, where it went down the whole rope. Live that hour 19-23 of the 170 touches were at the top of the
# rope at x -80 from 11 to 10 (11 of them on the rope, 16-48 under its top: R2 #8, #24, R3 #34 on their frames, a Horny
# Mushroom on 10 beside the rope's top each time), over 68 climbs up it, 24 of them past a Horny regardless (NO_WAIT)
# and about 13 the mover's lone climb move, which nothing watched for the top; she can't cast on a rope ("That cannot be
# used while you're on a rope or ladder.", the game's line for any skill), so nothing else is to be done there. False:
# as before.
# What it does in the simulator's thief world (the standard grind, HANG_S below; the review of 2026-11-02,
# D:/MapleHelper-scratch/recfix/review-rope: hangs.py, anal.py; counted again after its fixes on seeds 1-96,
# D:/MapleHelper-scratch/recfix/rope-fix): the way up seldom clears. Of 476 hangs 434 went up regardless after HANG_S,
# and 434 were followed by a touch at that rope (on it, or standing within 40 of it after the climb) within 3 s of the
# hang's end; none was touched in a hang. A Horny at its point on 10 (x -109, 29 from the rope at -80) or a Green at its
# point on 18 (x 39, 47 from the rope at -8) is within the 50 of her box and its (22 and 28) of the rope's top: standing
# there it blocks the top for good (at 7dd45bf, at the ends of the regardless hangs of seeds 1-24 a monster's middle was
# within 50 of the rope on the top platform in 94 of 115, 60 of them chasing her: the walkers world's aggro). Much of
# what the hang saves (touches -6.6%, that hurt -8.2%, at a rope's top -19%) comes from what follows it: the rest of the
# climb goes up in one push, unwatched (hang_up), and a touch in it is the cut's (POST_HIT: no fight while her
# invulnerability lasts), so the base's runs of 3-5 touches at one rope's top within 6 s (0.71 a run, about 2.7 touches)
# are nearly gone (0.04 a run, at 7dd45bf); hanging only where _climb_on did gave the base back. The wait does the rest:
# without it (HANG_S 0) touches +5.6% (t +2.65) on seeds 1-96.
ROPE_HANG = True
# Her feet this far under the rope's top at least where she hangs: she's 65 tall (bodies.HER_TALL, the client's box), so
# a monster standing on a flat top can't reach her body, with 15 more for her dot's error (to about 11 units).
# HANG_FLOOR: past this (her feet nearer the top) there's no hanging, the climb goes on as before. Both are for a top
# that doesn't dip under the rope's top within a monster's reach of it; where it does she hangs that much deeper
# (_hang_drop: the review of 2026-11-02 found 18 sloping down to -883.3 at x 61, 9.7 under the top of the rope at 111,
# where a Green Mushroom on 18 reaches her down to 75 under, inside the band's 70: there the band is 80-98 under; 4.8
# deeper under 146's top, 1.5 under -192's, none under -80's and -8's). The cut leaves her between HANG_FLOOR and
# HANG_UNDER and a frame's climb (8.7) under the top. On Tree Dungeon's long ropes (190 long: -80 from 11 to 10, 111
# and -8 up to 18) she grabs them 163-176 under the top, on the short ones (115-116: 146 from 32, -192 from 31, -8 from
# 30 up to 27) 104-116 under (route.grab_height), all under the band, and -8 from 29 81 under, in it: so she hangs on
# the short ones too (the review's harness runs at -192 and 146, test_replan's late replan at 146).
HANG_UNDER = 80.0
HANG_FLOOR = 70.0
# The cut is decided at the glance before the one that would be too late: from her dot's height (on the rope above where
# she grabbed it, and not in the air by the glance's state: in the jump at it no cut), less what she climbs in the
# glance's frame age and the game's lag in letting go of up (motion.body.Lags.off, 68 ms), at flow.ROPE_SPEED (her climb,
# SPEED + 20 by her stat window: the thief's 121, hers 120, as she climbed; until 2026-11-02 the planner's 130, so she
# stopped a little lower then).
HANG_LAG_S = 0.068
# HANG_S: the most she hangs. The roadmap's 4 s (about a touch's price, attack.stance.MEET_PRICE_S 2.86 s, and the climb
# down it saves), 2 s and 0 (the stop under the top, then up the rest at once, unwatched, a touch the cut's), on the
# harness's standard thief grind (loop-grind, Tree, crowd, choice, measured, zero-walls, near-misses, walkers, 5 minutes)
# against master 40ba190's cached seeds 1-48 (D:/MapleHelper-scratch/recfix: rope, review-rope, rope-fix): 4 s (at
# 7dd45bf) EXP a minute -4.2% (t -2.21), touches -9.3% (t -3.31), that hurt -12.3% (t -3.90), 18.3 s of hanging a run,
# a trip's arrival +23%; 2 s EXP -0.3% (t -0.14), touches -6.6% (t -2.38), that hurt -8.2% (t -2.53), at a rope's top
# -19% (seeds 1-24), 9.8 s of hanging, a trip's arrival +10%; 0 EXP +3.3% (t +1.89), touches -3.8% (t -1.29). 0 against
# 2 s, paired on seeds 1-96: EXP +1.8% (t +1.21), touches +5.6% (t +2.65), that hurt +5.3% (t +2.28), a trip's arrival
# -13% (on 1-48 EXP +3.6%, t +1.83, the review's +3.5% at 7dd45bf; 49-96 took it back). 2 s: the touches its wait saves
# are there at 96 seeds, the EXP 0 would earn isn't (the user's rule: EXP and loot over fewer touches; theirs to change,
# one constant). On the live hour 8 of the 11 hangs that hindsight says would have cleared within 4 s cleared within 2
# (the oracle below).
HANG_S = 2.0
# The monsters as the hang judges the way up: each walking as followed (TURN_SLACK and 20 units a second of spread, as
# every chain plans them), but one standing may set off either way for HANG_UNSURE_S at the most, not the chains'
# STAND_UNSURE_S (1 s): a Horny standing anywhere on 10's left part (-163 to -32, its Horny point at -109) is within
# 50 + 100 of the rope's top at -80, so at 1 s hardly a hang there ended before HANG_S (a first rope-top scenario, two
# Horny Mushrooms by the top, seeds 1-12: 18 of 19 hangs up regardless after 4 s; 12 of 19 at 0.3 s). On the hour's 35
# climbs that perception saw 100 or less under the top, the top was free for the 1.3 s of the rest of the climb, the
# climb-over and STAND_PAST_S at once on 20 (1 touched), and on the other 15 (all 15 touched) the monsters as they
# really went freed it within 1 s on 5, 2 s on 8, 4 s on 11 (D:/MapleHelper-scratch/recfix/rope/oracle.py, oracle.txt):
# the hang is worth what it judges free soon, so standing ones are taken as they stand but for about a state and a
# key's lag of setting off. Taken as standing (0) the harness's standard grind did no better (24 seeds, HANG_S 2 s:
# EXP a minute -2.0%, touches at a rope's top -22.1%, against -0.6% and -21.3% at 0.3; 4.7 of its 5.2 hangs a run up
# regardless where 4.8 of 5.3: what blocks the way up there is mostly one at the top or walking by it).
HANG_UNSURE_S = 0.3
HUNG = "hung on the rope"  # (the cut's reason starts so: Navigator._watching, _carry_out_chain, _carry_out_move)
# The user, 2026-11-02 (~12:45), asked whether she hangs 2 s or 4 s under a rope's top: "If the mob is moving and has an
# expectation to clear the area soon, then we may wait a bit longer, but otherwise 2s is what we wait. And sometimes it
# may make sense to just take the damage if there are a lot EXP to earn by killing off mobs above." So HANG_S is the
# wait, and two rules about it, judged at the hang's start and at every look of it (`_hang_judge`):
# HANG_LONGER: where every monster blocking the way up (_way_up's: the planner's reach) walks, and the state's monsters'
# likeliest walks (MonsterTrack.at: the way each is followed walking, at its kind's speed from the game data, turning at
# its platform's walk edges, as the simulator's v83 AI walks one through a decision of 1-3 s) have the way up clear and
# staying clear (the rest of the climb, the climb-over and STAND_PAST_S at the top, her box all the way) d s from now
# (`_walks_off`), she hangs till d and HANG_MARGIN_S from now, where that's past HANG_S and HANG_HORIZON_S or less from
# the hang's start, and where her hang place stays clear meanwhile (_hang_safe); re-judged at every look (d as the newest
# state has it, the margin from the first look that has it clear already; a look with no such clear has HANG_S, up at
# once if that's past). Once a monster it hangs longer for
# turns or stops (off its predicted walk by its heading as followed, away from its walk edges: `_off_walk`), or her hang
# place isn't clear that long, back to HANG_S from the hang's start for the rest of the hang (up at once if that's
# past). Telemetry 'rope_hang' events: step 'longer' when it first goes past HANG_S (the predicted clear, the walkers),
# 'back' when it's put back for good (why); a hang that ends at its longer wait 'waited' (up as predicted clear, no
# backstop event). False: HANG_S, as before.
# The horizon and margin by the harness (D:/MapleHelper-scratch/recfix/rope2: hangs2.py, compare2.py, perhang.py;
# HANG_LONGER alone against the 2 s hang, work/rec-all 560bd57's cached seeds, 1-24 paired, the thief's standard grind
# and her standard world): HANG_HORIZON_S 3 s, HANG_MARGIN_S 0.3 s: the thief's EXP a minute +2.3% (t +1.30) and his
# touches at a rope's top -11.2% (t -1.94), hers +0.1% (+0.12) and -9.0% (-1.49), better on both measures in both
# worlds. 4 s: the thief -0.2% and -9.9%, she +1.0% and -18.0% (-1.83); 2.5 s hardly ever longer (one of her 24 seeds
# changed); a margin of 0 (up at the first look with the way up predicted clear) the thief -1.8% and -6.4%, she +1.8%
# and -17.2%; 0.6 s the thief +2.5% and -9.4%, she +0.1% and -4.9%. None of these is told from nothing at 24 seeds. A
# walker's decision lasts 1-3 s in the v83 AI and a third of the next ones stand and a third turn, so a walk predicted
# more than about a second past HANG_S seldom holds: at 4 s 84 of the thief's 106 longer hangs were put back for a
# turn or a stop (52 of 69 at 3 s; hers 43 of 69 and 26 of 45), and the longer wait was lost time. Of her 92 longer
# hangs over seeds 1-48 (both rules) 1 ended at its own end ('waited'), 11 with the way judged clear.
# On, the user's rule (2026-11-02 ~21:40, shown it measured worse for her than the 2 s hang on the merged world: "keep
# the rope rule on"; off from the final review until then). The harness's `--world hang-2s` prices it (both rules off,
# the 2 s hang alone), `no-hang` the hang itself. Why it turned from better (work/rec-rope2, on 560bd57's world) to
# worse on the merged world, and what was fixed (D:/MapleHelper-scratch/recfix/rope-on: hangs3.py's knobs, her
# standard world, seeds 1-48 paired): the climb turns it, her climb up 130 in the world and planned up at 130 (flow's)
# and 117 (physics') as at 560bd57 against the client's 120 both ways (the knob C560): both rules against the 2 s hang
# EXP +3.1% (t +2.12) there, -4.1% (t -2.55) here; here the loss is in the knocks off 10 by the top of -80 (10 -> 13,
# 300 units down, a climb of ~9 s back) after hangs with a Horny chasing her by the top (hit by her: the user's monster
# rule), 47 over the 48 runs with the 2 s hang alone, 84 with both rules. Of the longer waits for a walker that was chasing
# her 63 of 72 were put back for its turn (none cleared): a monster she hurt follows her, so a walk of it is no walk
# off (`_walks_off`: none longer for one, its HP bar seen). Only on a navigator with a price (a hunter's), never a
# retreat's: its trip's plan has no time for a longer wait (behavior.retreat's RETREAT_S).
HANG_LONGER = True
HANG_HORIZON_S = 3.0
HANG_MARGIN_S = 0.3
WALK_OFF_STEP = 0.1    # s between the set-offs `_walks_off` tries (a resolution)
OFF_WALK_EDGE = 25.0   # units: a monster this near its walk edge (or predicted so) turning or stopping is at its end
# HANG_TAKE: at the hang's start and at every look, she climbs on at once (step 'took') where the rest of the wait
# costs her more than the touch it waits out: its s at what her time is worth up there (TopPrice.rate: the hunter's,
# Hunter.top_price: with behavior.hunter.TOP_RATE_ABOVE, on from 2026-11-02, what the grind expects on the platform at
# the rope's top from the monsters there, their EXP and her kill time, the user's "a lot EXP to earn by killing off
# mobs above", none where it has no stop there; off, the hunt's EXP a second) against the touch: a touch of the kind
# blocking in potions at the user's prices (HP_PER_MESO 2 HP a meso at MESO_EXP: behavior.fight.landing_prices, as
# her landings are priced) times the share of her touches that hurt, and the s a knock there would lose her at that
# rate (`_knock_cost`: none on the rope, where the game keeps her; standing at the top the knock's flight, and where it
# lands her off the platform the planner's way back up). Never where her HP as last read is under the safety stop's
# reserve and every touch it's priced as (TopPrice.margin: each touch's HP by its kind, the second at the top's too, and
# after a touch on the rope that one, `_take_now`), nor without a price (a navigator a hunter didn't make: a
# retreat's). The Lie Detector and the safety stop are raised at the hang's glances, whatever it chose. A touch taken
# so, in the climb after, is the cut's (POST_HIT), as after any hang. A touch
# on the rope with the top taken (_climb_on, CLIMB_LOOK) is up again at once so, where it went down to hang. False: as
# before. By the harness (as HANG_LONGER's): the touch taken at 34% of the thief's hangs and 28% of hers over seeds
# 1-48 (on 1-24, 68 of their 77 a Green Mushroom's: hers 11.4 EXP against her 2 s at 8.5 EXP a second, his 8.8 at 7.2).
# Both rules against the 2 s hang, seeds 1-48: her EXP a minute +3.5% (t +2.32), touches -3.9% (-1.25), her hanging
# -24%; the thief's EXP +0.9% (+0.38), touches +7.6% (+2.42), that hurt +5.8% (+1.65), touches at a rope's top level
# (+1.2%). Against no hang (ROPE_HANG off): her EXP -2.3% (-1.27), touches -8.4% (-2.34), at a rope's top -18.5%
# (-1.83); his -2.1% (-0.82), -2.9% (-1.14), -21.6% (-2.73). With the grind's rate on the platform at the top
# (TOP_RATE_ABOVE, 24 seeds, the margin then from every look) the touch was taken at 39% and 67% of the hangs, at 8.8
# and 13 EXP a second on average, and both did worse than at the hunt's rate: EXP -2.5% and -2.4% (t -0.64, -1.51),
# her touches +7.4% (+1.85). On, the user's rule, with HANG_LONGER (its note). Its price after the merged world's
# measures (D:/MapleHelper-scratch/recfix/rope-on): the touch taken costs what follows it at the top too, a second touch
# where it or another monster meets her standing where she climbs over once her invulnerability after the first is over
# (`_touched_again`: to TAKE_AGAIN_S past it, the next pass's reaction: on the merged world the second touches at the
# top came 1.45-1.9 s after the first), with its knock; the knock's reach the client's (KNOCK_REACH), where the
# planner's 53.7 put her on 10 after a knock left from the top of -80 (83 from 10's end) that took her off it; and a
# touch on the rope with the top taken (_climb_on) is the touch at the top she'd climb into, standing, where it was
# priced as one on the rope (no knock).
# Its HP check (the review of 2026-11-03, D:/MapleHelper-scratch/recfix/rope-on-review and rope-fix): it had her HP over
# the reserve and one touch of the kind in the way, where the take is priced as two touches when a monster stays at the
# top (the second by any kind), so from her HP between the reserve and one touch and two she'd take both and end under
# the reserve (her reserve 2 touches of 132-150: from 400-450 HP a two-touch take ends one touch from 0, the monster that
# made it two still at the top); the thief took 1 of his 2 so. And `_climb_on`'s take came right after the touch on the
# rope, which her HP as read may not have yet (her bar settles HP_SETTLE_S after the popup: perception.hp_number).
HANG_TAKE = True
TAKE_AGAIN_S = 1.0     # s past her invulnerability's end she's taken to stand where she climbed over (_touched_again)
HP_SETTLE_S = 0.37     # s after a touch's popup her HP as read has it (the bar within a px by then: perception.hp_number)
# A touch's knock standing, as the client throws her (sim.world.KNOCK_VX: away and up at 270 units/s, R1-R3's 15 fps
# frames: a median 73.4 units through the air, then her slide from her walk's speed at most): ~87 units at hers (the
# frames' 84-93), where the planner's KNOCK_VX and KNOCK_VY (its knock's flight's time, KNOCK_AIR_S) give 53.7.
KNOCK_PUSH = 270.0
KNOCK_REACH = KNOCK_PUSH * 2 * KNOCK_PUSH / GRAVITY + LAND_SLIDE  # (_follow's: its slide hers)
KNOCK_LOST_S = 30.0    # s a knock off the top costs where the planner has no way back (a bound)
# After the evade work's review (2026-10-29: D:/MapleHelper-scratch/evade/review; the fixes' runs and numbers in
# D:/MapleHelper-scratch/evade/fix). ARRIVE_STAND: for a trip whose x is what it's for (go's arrive_waits: her
# retreat's place, TripState.hold_x), the walk on arrival to where she's sent (WATCH_MOVES' _arrival_clear) is refused
# only where standing where she is as long and STAND_PAST_S more is clear, else she walks (a Lie Detector retreat onto
# 28 refused its walk from x 146 to 207 for a Green Mushroom on 30 that was already touching her at 146, and "arrived"
# where it went on touching her with the hunt stopped: loop_tree_lie_detector_28, seed 9); refused, she waits there a
# look at a time (arrive_waits) for it to clear. A trip that stops short of its x says so (Trip.short: her retreat
# tells it apart). For any trip it cost the walk touches (12 seeds: 3.9 more a run, se 1.6, 5.0 more within 2 s):
# there any x on the platform does. False: refused whenever the walk meets one.
ARRIVE_STAND = True
# PASS_WINDOW: a pass (_go_past, _within_window) leaves the monster in the way out only within the window its first touch
# of her on the pass's chain opens (that touch's plan time, her invulnerability now counted, INVULNERABLE_S on, less
# IMMUNE_MARGIN_S): a chain in its box past that is planned again with it counting from then (MonsterTrack.counts_from),
# that chain's own first touch checked the same way; only where nothing chains so, without it at all, then blind, as
# before. (Left out for the whole chain, her retreat after a stop was planned lying on 15 for a drop, a Horny Mushroom
# walking at her, past each window: touched 1.45 s apart and dead, the review's tour seed 7; that one it doesn't save:
# the touch the chain takes comes before the drop and cuts it.) In the walk level (12 seeds, off: 2.5 touches a run
# fewer, se 3.5). False: left out for the whole chain.
PASS_WINDOW = True
# The review's cycles (a jump up from 25 to 24 under a Green Mushroom refuged, waited for 3.5 s and passed into 4 times
# in one trip: its walk 19), measured, both off. BLOCK_MEMORY: the evade step's budget (EVADE_S from the first block)
# and its refuges are the blocked move's (by its key), kept over passes and chains until she makes the move or
# BLOCK_FORGET_S go by without a block of it, and (BLOCK_NO_WAIT) a block of it past its budget goes past at once, no
# wait; PASS_AVOID: a pass of it that's touched makes it dearer (st.avoid, AVOID_MEMORY_S: another way if there is
# one). Against both off, the harness's walk (24 seeds, D:/MapleHelper-scratch/evade/fix/abl/g-*): both on 3.2 touches
# a run more (se 1.7), 3.3 more within 2 s of the one before, her trips getting there as often (0.6 points fewer);
# BLOCK_MEMORY alone 6.4 more (se 1.3), 8.3 more within 2 s, 1.4 points more trips there (it passes sooner); PASS_AVOID
# alone 4.2 fewer (se 2.6) but 4.9 points fewer trips there (se 2.0: a dearer way, and trips out of time). False (as
# committed): the budget and refuges start again at every pass and every chain with legs, as before.
BLOCK_MEMORY = False
BLOCK_FORGET_S = 10.0
BLOCK_NO_WAIT = True
PASS_AVOID = False
# REFUGE_WORTH (the review's, measured, off): at a refuge she waits only while waiting can clear the way
# (_worth_waiting), else she goes past at once: the walk's touches 11.3 a run more (12 seeds, se 3.7), 6.5 more that hurt
# (as SAFE_PASS before it: a refuge's wait pays off past what _worth_waiting can see).
REFUGE_WORTH = False
# TOUCHER_ALL (the review's, measured, off): every trip without an escort keeps clear of the monster that touched her
# last (R7's stand-in at her x, TOUCHER_S), not only her retreat's (a refuge step just after a touch was judged clear
# without the Green Mushroom hidden under her sprite: the review's walk 1, 146.8 s). Its walking not known, its reach
# left no refuge within reach and turned spots into passes: the walk's touches 9.3 a run more (12 seeds, se 3.2), 8.4
# more that hurt; standing at her x (TOUCHER_STAND), 2.4 more (se 3.7). Left to perception (the track under her
# sprite kept). False: her retreat's alone.
TOUCHER_ALL = False
TOUCHER_STAND = True  # ... its stand-in standing at her x (False: its walking not known, as her retreat's)
# ARRIVE_CHECK_POST_HIT (a stopgap until the per-step planner, the user, 2026-11-03: "One-line patch for her now, then
# planner"): in the window after a touch (POST_HIT: no fight, no `until`) an escorted trip's walk
# on arrival is checked as an unescorted one's (_arrival_clear): a walk a monster would touch after her invulnerability
# arrives short where she stands. Off, the escort's walk on to its x went unchecked: after a touch on the -80 rope she
# walked through the Horny Mushroom chasing her to stop 10's x -138, 25 from 10's end, and its next touch knocked her off
# onto 13, again and again (D:/MapleHelper-scratch/arrival/REPORT.md). Measured (her standard world, 48 paired seeds, 5
# minutes; D:/MapleHelper-scratch/arrival/vtrace, planner/fix): on 68344b2 her knock-offs 10 -> 13 after a -80 climb 76
# -> 29 (t -4.00), long falls 213 -> 160, EXP a minute 459.8 -> 475.3 (+3.4%, t +2.87), touches level; on 71620d4
# 459.7 -> 475.5. The thief's not helped (61 -> 54, EXP -2.8%, t -1.34: kept at the top, his jump throws in place there
# are knocked off in the air): off for him. None: by the character playing (_arrive_check_post_hit: on where her profile
# has no point-blank answers up close, Policy.close_quarters 'touch', the magician's); True or False: for anyone.
ARRIVE_CHECK_POST_HIT: bool | None = None


def _arrive_check_post_hit() -> bool:
    """Whether ARRIVE_CHECK_POST_HIT holds for the character playing (character.active())."""
    if ARRIVE_CHECK_POST_HIT is not None:
        return ARRIVE_CHECK_POST_HIT
    return character.active().policy.close_quarters == "touch"

# How a trip treats monsters (Navigator.go's `mode`).
AVOID = "avoid"      # no escort: whole-screen looks, ways around, waits, hops, chains planned on the move
HUNT = "hunt"        # the escort shoots what's in reach before each chain, and what's in the way of a move
TRANSIT = "transit"  # the escort shoots only what's in the way
MODES = (AVOID, HUNT, TRANSIT)
FIGHTING = (HUNT, TRANSIT)  # the modes the escort fights in


class Escort(Protocol):
    """Fighting on the way (behavior.fight.FightOnTheWay): what the navigator asks between moves."""

    def target(self, s: Scene) -> Monster | None:
        """The monster to shoot from where she stands, if any is in reach."""

    def shoot(self) -> int:
        """Fight what's in reach for a moment; returns the kills."""

    def in_way(self, s: Scene, x: float) -> Monster | None:
        """The monster on her platform in her way to map x that clear_way would shoot (None if none)."""

    def clear_way(self, s: Scene, x: float, finish: bool = True) -> int | None:
        """Shoot a monster on her platform in her way to map x (where her next move starts, or where she's
        sent to on arrival); the kills, or None if none was in the way. `finish`: finishing what she has
        hurt too (not in transit)."""

    def threat(self, kind: str) -> bool:
        """Whether her chains keep clear of monsters of `kind` (a touch by one counts): they stop short of a leg one
        could touch her on, and the navigator has it shot (clear_for), goes around it, hops it or waits."""

    def clear_for(self, s: Scene, m: Monster, finish: bool = True, jump: bool = False) -> int | None:
        """Shoot monster `m`, one that could touch her on the chain ahead, if a star reaches it from where she
        stands (`jump`: or one from a jump, as shot_place has it): the kills, or None if none does. `finish`: as
        clear_way's."""

    def throws(self) -> int:
        """Her throws so far: a shot first that threw tells apart from one that didn't (the kills don't: a hit isn't
        one, and a fight's looks may not have it in reach after all)."""

    def wont_have(self, s: Scene, m: Monster) -> bool:
        """Whether no fight would take monster `m` as things are (taken for dying, set aside by the plan)."""

    def shoot_from(self, s: Scene, m: Monster, most: float, finish: bool = True) -> int | None:
        """Walk along her platform, `most` units at most, to the nearest place a star reaches monster `m` from, and
        shoot it there: the kills, or None if there's no such place."""

    def shot_place(self, s: Scene, m: Monster, most: float) -> tuple[float | None, str]:
        """Where her attack reaches monster `m` from (the fights' own reach): (her x, '') from where she stands, from
        the ground or a jump (clear_for's with `jump`), (a place within `most` along her platform, '') shoot_from's,
        else (None, why not). Nothing sent."""

    def landings(self, tracks: list) -> Callable[[Leg], tuple[float, MonsterTrack | None]] | None:
        """How a leg's landing is priced against `tracks` (the chain's): the price (s: 0 where her first attack there
        surely takes the nearest, nobody walking into her first), and the track making it so; None: not priced."""


@dataclass(frozen=True)
class TopPrice:
    """What her time up a rope is worth and what a touch at its top costs her (HANG_TAKE: Navigator's `price`, the
    hunter's Hunter.top_price): `rate` EXP a second, `touch` EXP a touch of the kind takes (its HP in potions at the
    user's prices, times the share of her touches that hurt), `margin` her HP as last read over the safety stop's
    reserve (HP; None: not read) and `read` when (its frame's time), `hp` the HP a touch of the kind takes, `largest` the
    largest touch's of the map's (a touch whose kind isn't told); `words` where the rate comes from."""
    rate: float
    touch: float
    margin: float | None
    words: str = ""
    hp: float = 0.0
    largest: float = 0.0
    read: float | None = None


@dataclass
class _Wait:
    """A hang's wait as judged so far (Navigator._hang_judge): `s` from its start (HANG_S, or longer: HANG_LONGER),
    the walks it was last longer for (`walks`: [(the feed's track id, the track as predicted, its plan time's start)],
    None till then), whether it was ever longer (`said`: its 'longer' event), whether it was put back for good
    (`back`: one of them off its walk), and when the way up is predicted clear (`clear`: kept while each look has it
    clear already, so HANG_MARGIN_S counts from then)."""
    s: float = 0.0
    walks: list | None = None
    said: bool = False
    back: bool = False
    clear: float | None = None  # (when the way up was last predicted clear, time.monotonic; None: not)


@dataclass
class Trip:
    """How a trip (Navigator.go) went."""
    result: str = ""      # 'arrived', 'in position', or why she didn't get there
    moves: list[str] = field(default_factory=list)
    misses: int = 0       # moves that didn't land where planned (not counting hits)
    hits: int = 0         # moves during which a monster hit her
    kills: int = 0        # monsters shot dead on the way
    chains: int = 0       # key plans that carried out several moves in one go
    waits: int = 0        # looks spent waiting for a monster to move off
    replans: int = 0      # chains replanned and handed over while one ran (REPLAN)
    time: dict = field(default_factory=lambda: {"look": 0.0, "think": 0.0, "move": 0.0})  # s spent
    short: ClassVar[str] = ""  # arrived short of the x she was sent to, and why (ARRIVE_STAND), set on the trip; "":
                               # not (no field: the trip's record as before)


class KeyNames(Mapping):
    """Her moves' key names (flow.Keys': left, right, jump, up, down, and "pick" for her pick-up) as her actions have
    them at each ask: a key followed to where the game has it (behavior.keycheck, Actions.follow) is in the next chain.
    Until 2026-10-28 they were taken at the navigator's start."""
    FIELDS = {"left": "left", "right": "right", "jump": "jump", "up": "up", "down": "down", "pick": "pick_up"}

    def __init__(self, actions):
        self.actions = actions

    def __getitem__(self, name: str) -> str:
        key = getattr(self.actions.keys, self.FIELDS[name], None) if name in self.FIELDS else None
        if key is None:
            raise KeyError(name)
        return key.name

    def __iter__(self):
        return (n for n, f in self.FIELDS.items() if getattr(self.actions.keys, f, None) is not None)

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _Running(NamedTuple):
    """A chain's key plan, sent to the player (running, or queued to start when the one before ends)."""
    plan_id: str
    legs: list[Leg]
    start: float  # when it starts (time.monotonic)
    more: bool    # another chain may follow on from where it ends
    tail: float = 0.0  # s its plan runs on past its last leg (time to place her before the next one)
    length: float | None = None  # s its plan lasts, as sent (its last key can come after its last leg)
    known: tuple = ()  # the monsters the look it was planned from had (MonsterTrack, in its plan's time): REACT
    parent: str | None = None  # spliced in: the plan it takes over from, until the player has taken it (client.taken)
    why: str = ""      # why its chain ended (flow.chain's: 'done', 'monster', 'long', 'unpredictable')
    arrows: tuple = ()  # its arrow keys' events as sent ((ms, -1 left / +1 right, down)), and the one held at its start
                        # ((0, d, True) first): which arrow is held when (_splice_point; () not known: by the legs)

    @property
    def ends(self) -> float:
        """When its plan is over (time.monotonic)."""
        return self.start + (self.length if self.length is not None else self.legs[-1].end.t + self.tail)


class _Ahead(NamedTuple):
    """The look around late in a running chain, for the chain that may follow it."""
    scene: Scene         # with her put where the running chain ends
    tracks: tuple = ()   # the monsters the next chain keeps clear of (with an escort, those whose touch counts) ...
    watch: tuple = ()    # ... and, with an escort, every one it mustn't come near (else she stops, and it fights them)


@dataclass
class Choice:
    """Which of its ways a trip takes (route.choose, at its first plan: Navigator._route), kept through its replans
    and, in the hunt loop, its objective's later trips: the move the way goes through (None: the quickest), until
    she has set off on it, she is past it (on a platform the way reaches from it on: `beyond`), or the way from where
    she is got too dear or would take her back (onto a platform twice, or onto one she has set off from since it
    chose, `left`, that the quickest from there doesn't go onto); and how much dearer than the quickest it may be (s:
    route.STRETCH of the quickest when it chose)."""
    made: bool = False
    via: Move | None = None
    leeway: float = 0.0
    beyond: frozenset = frozenset()
    left: set = field(default_factory=set)


@dataclass
class TripState:
    """A trip in progress (Navigator.begin, then Navigator.step until it's over): where it goes and how,
    and what it keeps from one look to the next."""
    dst: int
    x: float | None = None
    deadline: float = 0.0  # time.monotonic() it times out at
    max_misses: int = 6
    until: Callable[[Scene], bool] | None = None
    health: Callable[[Scene], None] = lambda s: None
    mode: str = AVOID
    trip: Trip = field(default_factory=Trip)  # how it's going so far (returned at the end)
    weights: dict[tuple, float] = field(default_factory=dict)  # its random move weights, kept through its replans
    choice: Choice = field(default_factory=Choice)  # the way it takes, of the ways the route planner offers
    over: bool = False
    fights: int = 0      # clearing the way, in a row without a move between (see MAX_FIGHTS)
    tries: dict[int, int] = field(default_factory=dict)  # platform -> monsters engaged there on the way
    # (kind, src, dst, x) of moves a monster was in the way of -> when: rebuilt at every plan, less
    # those older than AVOID_MEMORY_S (the route planner's penalty reads the current one)
    avoid: dict[tuple, float] = field(default_factory=dict)
    waited: int = 0      # looks spent in a row waiting for a monster to move off
    moved_at: float = 0.0  # when the last plan ended (she may still be sliding)
    # Touched (a chain or a move cut short by a touch, or her HP dropping): up to when her invulnerability after it lasts
    # (Senses.invulnerable_until less IMMUNE_MARGIN_S), no fight starts on the way and `until` isn't asked (POST_HIT)
    post_hit_until: float = -1e18
    switched_at: float = -1e18  # when a running chain was last replanned and handed over (REPLAN: SWITCH_MIN_S)
    fresh: Scene | None = None  # a look taken for the next pass already (after a chain, or the caller's)
    last: Scene | None = None   # the last pass's look (the one `until` said 'in position' to, if it did)
    keep_off: frozenset = frozenset()  # platforms its routes go onto only at a cost (route.KEEP_OFF_S): the hunting
                                       # ground a spot it goes to shoots
    vary: bool = True  # it chooses among the planner's ways (route.ROUTES); not a trip with a deadline (drops, a
                       # retreat): that goes the quickest way, by its weights
    shots: int = 0       # shots at a monster that could touch her on the way, in a row without a move between
                         # (MAX_FIGHTS at most: _meet)
    blocked: float | None = None  # since when its chains have stopped short of such a monster without a move between
                                  # that got her where it was going (MEET_S: _meet), None while they don't
    path: list[Move] = field(default_factory=list)  # the route the running chain was planned from (_look_ahead)
    sighted: list[tuple[MonsterTrack, Monster]] = field(default_factory=list)  # the last pass's tracks of the
                                                                              # monsters it saw, each with the sighting
    cannot: set = field(default_factory=set)  # moves the mover couldn't do from where she stood (CANNOT_S)
    refused: dict = field(default_factory=dict)  # moves the mover refused for a monster (Mover.clear) -> how often
    stopped: bool = False  # the last chain was cut short for a monster coming into its way (REACT): not a miss
    came: object | None = None  # ... the feed's track of that monster (tracking.Track), kept for the next chain
    thought_to: float = 0.0  # up to when thinking is counted (Trip.time: _chain, called again for a pass's replans)
    land_waits: int = 0  # looks spent in a row waiting for a monster to clear the chain's last landing (LAND_WAITS)
    # The evade step (EVADE): when the blocked move's first refuge decision came (None: none since a move was planned
    # clear), the refuges taken for it, the last one's (platform, x) while she's there, the contact rule's last touch
    # taken as given (time.monotonic), and the trip's budget: EVADE_S, and how much dearer a refuge move may make it
    evade_since: float | None = None
    refuges: int = 0
    refuge: tuple[int, float] | None = None
    contact_at: float = -1e18
    evade_s: float | None = None   # (None: EVADE_S)
    refuge_s: float | None = None  # (None: EVADE_MOVE_S)
    passed: tuple[int, float] | None = None  # the platform her last pass set off from, and when (PASS_COMMIT_S)
    toucher: bool = False        # its chains keep clear of the monster that touched her last (TOUCHER_S: R7)
    deadline_pass: bool = True   # with EVADE_DEADLINE_S left it goes past; False (a retreat after a stop): it doesn't
    # BLOCK_MEMORY: the blocked move evade_since and refuges are for (its key), the time it was last blocked, and the
    # other blocked moves' (key -> (evade_since, refuges, last blocked)); why the evade step gave way to a pass (_why_past)
    evade_key: tuple | None = None
    blocked_at: float = -1e18
    blocks: dict = field(default_factory=dict)
    past_why: str | None = None
    arrive_waits: int = 0  # looks it may wait on arrival for the walk on to x to clear (ARRIVE_STAND: go's) ...
    hold_x: bool = False   # ... and whether its x is what it's for (a retreat's place: go's arrive_waits given)
    # Its arrival walk on to x (Mover.walk_to): its walks, [units asked, hold ms] each, and the look it ended with, for the
    # 'trip' record (`_trip_end`: where she was as it ended)
    walked: list | None = None
    walked_to: Scene | None = None
    # The target-first planner's trip (behavior.targeting: a grind's spot trip): over as she reaches its platform, wherever
    # she is on it: no walk on to x, no shot clearing it; what she does there is the planner's
    on_arrival: bool = False
    # (the target-first planner's whole screen) Asked of her look and the platform a move or a landing of this trip goes to,
    # before she stands and waits for a monster to move off it: True ends the trip there, 'in position', and the hunt has
    # something else for her
    instead_of_waiting: Callable[[Scene, int], bool] | None = None
    stop_waiting: bool = False
    land_near: bool = False  # (the whole screen's trip to her target's place) the quickest with her walk on to x, as that place's time was reckoned
    # (the loot chain, behavior.lootchain) Asked of her look: the way on, her drops in order ((platform, x, s to stand there), ...), or
    # [] for done. The trip is one chain of the route's moves and walks ('walk', flow.Flow._walk) through them, cut, or spliced as her
    # other trips' are; it's over with that plan, 'in position', and what's left is the hunt's next look's to say.
    through: Callable[[Scene], list[tuple[int, float, float]]] | None = None
    stops: list[tuple[int, float, float]] = field(default_factory=list)  # ... as last asked, less the walks made (`_gone_by`)
    ran: list[tuple[int, float, float]] = field(default_factory=list)  # ... the stretches her chain ran her over: (platform, x from, x to)
    walk_blocked: bool = False  # ... a monster in the way of its first walk: nothing sent (`_meet`)
    chain_why: str = ""         # ... why the plan ended: a cut's, else why its chain did ('done', 'monster', ...)

    def __post_init__(self):
        self.evade_s = EVADE_S if self.evade_s is None else self.evade_s
        self.refuge_s = EVADE_MOVE_S if self.refuge_s is None else self.refuge_s

    def end(self, result: str) -> Trip:
        """The trip, over: `result` says how it ended."""
        self.trip.result, self.over = result, True
        return self.trip

    def paused(self, seconds: float) -> None:
        """She stood still `seconds` (LookLost: Senses.regain): the trip's clocks move on by as much, its deadline and
        how long its chains have stopped short of a monster (MEET_S), and a look taken before isn't used, nor the
        monsters seen before (the review of 2026-10-24: a quick look after a cut once planned from the monsters of before
        the pause). Where moves were in a monster's way (`avoid`) ages as the game's own clocks do, and so does her
        invulnerability after a touch (`post_hit_until`)."""
        self.deadline += seconds
        if self.blocked is not None:
            self.blocked += seconds
        self.fresh, self.sighted = None, []


class Navigator:
    """Gets her onto a platform: plans the route, chains its moves, and replans from wherever she ends up."""
    _clear_of: tuple = (None, ())  # (`first_clear`: the look its tracks were last taken for, and them)

    def __init__(self, senses: Senses, actions: Actions, model: MapModel, *, log=print,
                 health: Callable[[Scene], None] | None = None, escort: Escort | None = None,
                 variety: float | None = None, rng: random.Random | None = None, memory=None, moves=None,
                 danger: Callable[[], object] | None = None, price: Callable[[int, str], TopPrice | None] | None = None):
        """`health(scene)`, asked at every pass's look: her potions, and the safety stop, which it raises (a hunter's
        health_check: behavior.safety); `escort` fights on the way (without one she avoids monsters instead).
        `variety`: the moves' random weights (route.NOISE by default; 0 for none). `memory`: her memory of the
        monsters (localization.memory.Memory, a hunter's), whose monsters where a fighting trip's chain goes it keeps
        clear of too (`remembered`); None, only the looks'. `moves`: what her dot showed of her chains' jumps and runs
        is told to it (a hunter's behavior.keycheck.MoveWatch: `_keys_seen`); None: to nothing. `danger`: the hunter's
        danger model as of each ask (attack.danger.Danger: its `at`), which breaks ties between refuges as near
        (EVADE); None: the nearest on the left first. `price(platform, kind)`: what her time on that platform is worth
        and a touch by `kind` costs her (TopPrice: a hunter's Hunter.top_price), for her hang under a rope's top
        (HANG_TAKE); None: no touch taken."""
        self.senses, self.actions, self.model, self.log = senses, actions, model, log
        self.moves = moves
        self.danger = danger
        self.price = price
        self._took: dict = {}  # (the last touch taken's numbers: _take_now, for its 'rope_hang' event)
        self.memory = memory
        self.health = health or (lambda s: None)
        self.escort = escort
        self.mode = HUNT if escort is not None else AVOID  # a trip's mode unless go() says otherwise
        self._mode = self.mode                             # the running trip's
        self.planner = Planner(model)
        self._epoch = physics.EPOCH  # her walk and climb the planner's moves were made at (_follow_walk)
        self.mover = Mover(senses, actions, model, log)
        self.rng = rng or random.Random()
        # Like a player, not the same way every time: each trip weighs every move by a random 1 to
        # 1 + variety, so near-equal variants take turns (the basin's two ropes, a drop or a walk-off), moves
        # start from varied spots (flow.Flow's rng), and with route.ROUTES over 1 it chooses among the ways
        # the planner offers (route.choose: _route).
        self.variety = route.NOISE if variety is None else variety
        self._weights: dict[tuple, float] = {}
        self._used: dict[tuple, float] = {}  # move -> when she last set off on it (route.STALE_S) ...
        self._since = time.monotonic()       # ... (one she never set off on: since it began)
        # her keys' names as her actions have them at every chain (KeyNames: followed keys too), pick-up pressed all
        # through her runs (loot on the way)
        self.flow = Flow(model, KeyNames(actions), rng=self.rng if self.variety else None)
        self.tracker = Tracker(model)  # which way the monsters walk, from look to look
        self._checked: Scene | None = None  # follow: the last look placing her for the next chain
        self._threat_kinds: frozenset | None = None  # the map's kinds whose touch counts (_threats), once worked out
        self._siblings: dict[tuple[int, int], list[Move]] | None = None  # (src, dst) -> the route's moves (_pricing)
        self._reacted: dict[int, float] = {}  # the feed's track id -> when a plan was last cut for it (REACT)
        self._replanned = False  # whether the last _switch replanned with nothing to take (REPLAN: _watching marks so)
        self._late = None  # the last _switch's replan dropped as too late: (why, its ms, s late), said by _dropped
        self._late_in_row: dict[int, int] = {}  # the feed's track id -> its replans dropped late in a row, no room
        self._came = None  # the feed's track of the monster the last cut was for (REACT: _coming), for TripState.came
        self._cut_plan: tuple | None = None  # where the last cut cut: (the running plan's legs, its plan time then)
        self._unshot: dict = {}  # a track the shot first threw nothing at, on a trip through her drops -> until when it isn't tried again
        self.keep_avoid = False  # whether where moves were in a monster's way goes on from one trip to the next (the loot chain's: a trip's a plan)
        self.landing_asks = False  # whether a chain's landings get the question a pass asks first (behavior.targeting.stay_or_go: `_landing_calls`)
        self._asked_on: int | None = None  # ... the platform that last got it
        self._asked_n = 0  # ... with so many monsters on the screen
        self._avoided: dict = {}  # ... those moves -> when
        self._tails: dict[str, float] = {}    # a chain's plan id -> s it runs on past its last key (_ended)
        # The last chain's glances, if her dot stood where the first had it all along: that dot and for how long
        # (None: it moved, or no glance had it): _rope_top
        self.chain_still: tuple[tuple[float, float], float] | None = None
        self.cut_still = False  # (the last chain cut short for her dot standing where it runs her: ROPE_CUT)
        self._tracks_at = time.monotonic()  # when the last pass's monsters were made (their plan time 0: _chain_monsters)
        # Passes regardless, by move (kind, src, dst): [its episode's number, when the last was, passes in it] (the
        # backstop events' episodes: _pass_record), and the episodes so far
        self._episodes: dict[tuple, list] = {}
        self._episode_n = 0

    @property
    def fight(self) -> bool:
        """Fighting on the way (the escort's hunt or transit mode) rather than avoiding monsters."""
        return self._mode in FIGHTING

    def leaves_at(self, here: Platform, x: float, dst: int, dst_x: float | None = None) -> float | None:
        """Where along platform `here` (from map x) the quickest way onto platform `dst` leaves it (its first move's
        takeoff, or where that starts), or `dst_x` if `dst` is `here`; None if there's no way."""
        if dst == here.id:
            return dst_x
        path = self.planner.plan(here.id, x, dst)
        if not path:
            return None
        first = path[0]
        return first.takeoff if first.takeoff is not None else first.x

    def go(self, dst: int, x: float | None = None, seconds: float = 180, max_misses: int = 6,
           until: Callable[[Scene], bool] | None = None, *, mode: str | None = None,
           scene: Scene | None = None, keep_off=(), vary: bool = True, stand_until: float | None = None,
           evade_s: float | None = None, refuge_s: float | None = None, toucher: bool = False,
           deadline_pass: bool = True, arrive_waits: int = 0) -> Trip:
        """Get onto platform `dst` (and to map x on it, if given). `until` is asked before every chain,
        with a look that includes monsters (late in the chain before, when it follows on from that one):
        True ends the trip early ('in position'). `mode` (AVOID, HUNT or TRANSIT; the navigator's own by
        default) says how the trip treats monsters. `scene`: the caller's last look, used for the first
        pass instead of a new one if it's fresh (no keys sent since). `keep_off`: platforms its routes go onto
        only at a cost (TripState.keep_off). `vary`: it chooses among the planner's ways (route.ROUTES; TripState.
        vary). While its looks fail (LookLost) it stands still up to its deadline, or `stand_until` (time.monotonic) if
        that's sooner: a retreat's window (behavior.retreat). `evade_s` and `refuge_s`: its evade step's budget and how
        much dearer a refuge move may make it (EVADE_S, EVADE_MOVE_S; a retreat's own); `toucher`: its chains keep clear
        of the monster that touched her last (TOUCHER_S); `deadline_pass`: it goes past a monster with EVADE_DEADLINE_S
        left (False: it waits at a refuge to the end, a retreat after a stop); `arrive_waits`: looks it waits on its
        platform at most, a WAIT_S each, where the walk on to x isn't clear and standing is (ARRIVE_STAND: a
        retreat's), before it ends short of x (0: at once)."""
        st = self.begin(dst, x, seconds, max_misses, until, mode=mode, scene=scene, keep_off=keep_off, vary=vary,
                        evade_s=evade_s, refuge_s=refuge_s)
        st.toucher, st.deadline_pass, st.arrive_waits, st.hold_x = toucher, deadline_pass, arrive_waits, arrive_waits > 0
        while not st.over:
            try:
                self.step(st)
            except LookLost as lost:  # the game doesn't look as expected: stand still until it does, up to the trip's
                stood = time.monotonic()  # deadline (then the command stops, saying why), as the hunt loop does
                until = st.deadline if stand_until is None else min(st.deadline, stand_until)
                if self.senses.regain(lost, until) is None:
                    raise
                st.paused(time.monotonic() - stood)
        return st.trip

    def begin(self, dst: int, x: float | None = None, seconds: float = 180, max_misses: int = 6,
              until: Callable[[Scene], bool] | None = None, *, mode: str | None = None,
              scene: Scene | None = None, keep_off=(), vary: bool = True, evade_s: float | None = None,
              refuge_s: float | None = None, on_arrival: bool = False,
              instead_of_waiting: Callable[[Scene, int], bool] | None = None,
              through: Callable[[Scene], list] | None = None, land_near: bool = False) -> TripState:
        """A trip to platform `dst`, as go() makes it, to be carried out one step() at a time. A twin is the platform
        it lies on (mapdata.MapModel.twins): she's there on either."""
        dst = self.model.place_of(dst)
        mode = mode or self.mode
        if mode not in MODES or (mode in FIGHTING and self.escort is None):
            raise ValueError(f"no {mode!r} trip without an escort" if mode in MODES else f"unknown mode {mode!r}")
        if mode in FIGHTING:  # the kinds whose touch counts, worked out now rather than between a look and its chain
            self._threats()   # (when the escort read the game data itself the first read took 535 ms on Tree
                              # Dungeon, a chain then planned from a look that old: review, 2026-10-24; Hunter.danger
                              # has it from the map's cache since)
        st = TripState(dst, x, time.monotonic() + seconds, max_misses, until, self.health, mode,
                       keep_off=frozenset(keep_off), vary=vary, evade_s=evade_s, refuge_s=refuge_s,
                       on_arrival=on_arrival, instead_of_waiting=instead_of_waiting, through=through, land_near=land_near)
        if self.keep_avoid:
            st.avoid = dict(self._avoided)
        # Her last look will do for the first pass if nothing has moved her since: one look less before she
        # sets off. Only a look with the monsters the mode looks for (whole screen when avoiding them), and
        # one that shows her standing (_onto_platform).
        if scene is not None and scene.searched and (mode in FIGHTING or scene.wide) and self.fresh(scene) \
                and standing(scene):
            st.fresh = scene
        return st

    def _follow_walk(self) -> None:
        """At a pass's start, where no chain of hers runs: a change of her walk and climb proposed made (physics.settle:
        her stat window's SPEED read in the hunt loop, behavior.hunter), and the route graph made again if her walk or
        climb is another than its moves were made at (physics.EPOCH: their takeoffs, landings and costs), with what was
        worked out of its moves (the pricing's siblings). A chain planned before keeps its timings: its keys are the
        player's, and it has ended by now."""
        physics.settle()
        if self._epoch != physics.EPOCH:
            self.planner, self._epoch, self._siblings = Planner(self.model), physics.EPOCH, None

    def fresh(self, s: Scene) -> bool:
        """Whether look `s` still shows where she is: taken FRESH_S ago at most, with no key plan sent since."""
        return s.where is not None and s.t >= self.actions.sent_at and time.monotonic() - s.t <= FRESH_S

    def step(self, st: TripState) -> Trip | None:
        """One pass of trip `st`: a look, then what it calls for (stop, get off a rope, shoot what's in
        reach or in the way, arrive, wait for a monster, or carry out a chain of moves or one move).
        Returns the trip once it's over (and sends its 'trip' record), else None."""
        if st.over:
            return st.trip
        self._follow_walk()
        self._mode, self._weights = st.mode, st.weights
        try:
            trip = self._pass(st)
        finally:
            self._mode = self.mode
        if trip is not None:
            telemetry.emit("trip", dst=st.dst, result=trip.result, **_trip_end(st))
        return trip

    def _pass(self, st: TripState) -> Trip | None:
        if not time.monotonic() < st.deadline:
            last = st.last.where if st.last is not None else None
            at = st.refuge is not None and last is not None and last.platform is not None \
                and last.platform.id == st.refuge[0] and abs(last.x - st.refuge[1]) <= ARRIVE_TOLERANCE
            return st.end("timeout (at a refuge)" if at else "timeout")
        if st.trip.misses >= st.max_misses:  # (an evade step's chain missed last: _carry_evade)
            return st.end("too many misses")
        # The Lie Detector seen: the trip stops here, keys or not (behavior.retreat); a retreat's own trip goes on (its
        # plans are the kind the player takes then).
        if getattr(self.actions, "kind", None) is None:
            client.check_alarm()
        # Look, and deal with what's here.
        if st.stop_waiting:  # (the hunt has something else for her: `instead_of_waiting`)
            return st.end("in position")
        s, looked = self._look(st)
        st.last = s
        st.health(s)  # her potions, and the safety stop (a hunter's health_check raises it: behavior.safety)
        fights = not self._post_hit(st)  # (no fight, nor `until`, while her invulnerability after a touch lasts)
        if fights and st.until is not None and s.where.platform is not None and st.until(s):
            return st.end("in position")
        s = self._onto_platform(s)
        if s.where.platform is None:
            return st.end("lost: not on any platform")
        if st.through is not None:
            st.stops = st.through(s)
            if not st.stops:
                return st.end("arrived")
        elif s.where.platform.id == st.dst and st.on_arrival:
            return st.end("arrived")
        elif s.where.platform.id == st.dst:
            # There: only a monster between her and the x she's sent to is shot first. Bursts at
            # whatever is in reach here (as on the way) took whole trips: a 60 s trip to drops on the
            # basin spent all of it shooting respawns there (tour_hunt_basin).
            if fights and st.x is not None and self._clear_the_way(st, s, st.x):
                return None  # look again after the fight
            if st.x is not None and self._arrival_clear(st, s):
                st.walked_to = self.mover.walk_to(st.x, tolerance=ARRIVE_TOLERANCE, s=s)
                st.walked = list(getattr(self.mover, "walked", None) or ())
            elif st.trip.short and st.arrive_waits > 0:  # (ARRIVE_STAND: standing here clear, a look again)
                st.arrive_waits -= 1
                st.trip.waits += 1
                st.trip.short = ""
                self.log(f"  waiting for the way on to x {st.x:.0f} to clear")
                time.sleep(WAIT_S)
                return None
            return st.end("arrived")
        if fights and self._shoot_on_the_way(st, s):
            return None  # look again after the fight

        # Plan the route, and chain as many of its moves as can be predicted (no touch counting on its legs while her
        # invulnerability after a touch lasts: IMMUNE, from when the chain can start at the latest, her lock held back).
        lock = max(getattr(self.actions, "lock_until", 0.0), getattr(self.actions, "arrow_until", 0.0))
        self._immunity(max(time.monotonic(), lock) + PLAN_START_S)
        path, planned_at = self._plan(st, s, st.dst)
        if not path:
            return st.end(f"no route from platform {s.where.platform.id} to {st.dst}")
        if fights and self._clear_the_way(st, s, path[0].x):
            return None  # look again after the fight
        monsters, near = self._chain_monsters(st, s, path)
        legs, keys, why = self._chain(st, s, path, monsters, looked)
        if legs:  # (the way is clear, or clear to where it's met: the evade step starts again from nothing; with
            # BLOCK_MEMORY a blocked move's budget is kept till she makes it: _made)
            st.refuge = None
            if not BLOCK_MEMORY:
                st.evade_since, st.refuges = None, 0
        met = None
        if self.fight:
            met = self._meet(st, s, path, monsters, legs, why, planned_at, looked)
            if met is True:
                return None  # look again: after the shot, for a way around, or once she's waited a moment
            if met is not None:
                legs, keys, why = met  # a hop over it
            elif self._clear_landing(st, s, path, monsters, legs, why):
                return None  # look again: after the shot, or once she's waited a moment
        elif self._wait_for_monster(st, s, path, monsters, legs, why, planned_at, looked):
            return None  # look again: for a way around, once she's waited a moment, or after a refuge or a pass

        # Her invulnerability as it is when the chain goes (IMMUNE): the planning's time came out of it.
        if legs and self._immunity_spent(legs):
            return None  # look again, and plan with what's left of it

        # Move: the chain in one plan, or else the first move on its own.
        hp = s.hp
        if legs and st.through is not None and not keys.sorted():  # (walks of nothing: no plan to send)
            st.walk_blocked = True
            return st.end("in position")
        if legs:
            # With an escort the chain keeps clear only of the monsters whose touch counts (on the first map none: it
            # fights them instead), but its run on past its end (follow's tail) mustn't take her into any.
            if near is None:
                near = self.tracker.update(s.t, s.monsters, seen_at=s.t - time.monotonic())
            st.path = path
            self.chain_still = None
            moved, planned, after = self._carry_out_chain(st, s, path, legs, keys, why, st.dst, near,
                                                          replan=REPLAN and met is None)
            still = self.chain_still is not None and (self.chain_still[1] >= STUCK_S or ROPE_CUT and self.cut_still)
        elif st.through is not None and path[0].kind == "walk":  # (nothing to chain to her drops: the hunt says what instead)
            st.walk_blocked = True
            return st.end("in position")
        else:
            st.stopped = False
            moved, planned, after = self._carry_out_move(st, s, path, monsters)
            if not moved:
                if _key(path[0]) in st.refused:  # refused for a monster: a refuge, else a wait (then unchecked)
                    self._refused_step(st, s, path, monsters, looked)
                return None  # one the mover can't do: another way (st.cannot)
            still = self.mover.still is True
        if still and ROPE_TOP:
            after = self._rope_top(st, s, after, jumped=bool(legs) and any(leg.air for leg in legs)
                                   or bool(legs) and ROPE_CUT and self.cut_still)
        self._made(st, moved, after.where.platform is not None and after.where.platform.id == planned.id)
        if self._note_landing(st.trip, moved, planned, after, hp, st.stopped, since=s.t):  # a miss
            st.trip.misses += 1
            self._rope_drop(st, moved, planned, after)
            if st.trip.misses >= st.max_misses:
                return st.end("too many misses")
        return st.end("in position") if st.through is not None else None

    # --- the steps of a pass (_pass) ---------------------------------------------------------------

    def _look(self, st: TripState) -> tuple[Scene, float]:
        """A look at her that includes the monsters (the whole screen when avoiding them; a fight's band look carries
        those off its band, Scene.swept: every state has the whole screen's), or the one taken for this pass already
        (st.fresh). Returns it and when it was over (thinking is counted from then). (Until 2026-10-30 a quick one
        without monsters right after a chain was cut short: TripState.quick, POST_HIT.)"""
        if st.fresh is not None:
            s, st.fresh = st.fresh, None
            return s, time.monotonic()
        t0 = time.monotonic()
        s = self.mover.locate(mobs=True, wide=not self.fight)
        t1 = time.monotonic()
        st.trip.time["look"] += t1 - t0
        return s, t1

    def _shoot_on_the_way(self, st: TripState, s: Scene) -> bool:
        """In hunt mode: shoot a monster in reach from where she stands, ON_WAY_TRIES times at most per
        platform (respawns could keep her there forever). True if she fought."""
        here = s.where.platform
        if self._mode == HUNT and st.tries.get(here.id, 0) < ON_WAY_TRIES \
                and (mob := self.escort.target(s)):
            st.tries[here.id] = st.tries.get(here.id, 0) + 1
            self.log(f"  {mob.kind} in reach on platform {mob.platform}: shooting it on the way")
            st.trip.kills += self.escort.shoot()
            if st.tries[here.id] == ON_WAY_TRIES:
                telemetry.emit("event", kind="backstop", text=f"ON_WAY_TRIES: no more bursts on platform {here.id}")
            return True
        return False

    def _plan(self, st: TripState, s: Scene, dst: int) -> tuple[list[Move], float]:
        """The route from where she stands to `dst` (empty if there's none), and when it was planned. A
        move a monster was in the way of less than AVOID_MEMORY_S ago costs extra, so a way around wins
        if there is one."""
        now = time.monotonic()
        st.avoid = {key: t for key, t in st.avoid.items() if now - t < AVOID_MEMORY_S}
        if self.keep_avoid:
            self._avoided = st.avoid
        path, offered = self._route(st, s.where.platform.id, s.where.x, dst)
        if path and telemetry.recording():
            telemetry.emit("route", **_route_record(st, dst, path, s.where.platform.id, s.where.x, offered))
        return path, now

    def _route(self, st: TripState, src: int, x: float, dst: int) -> tuple[list[Move] | None, tuple | None]:
        """Trip `st`'s route from (platform src, x) to `dst`, weighed by its weights and penalized as `_penalty`
        says; and, at its first plan with route.ROUTES over 1, the ways it chose among, the chances they were drawn
        with and which it took (route_record's `ways`). At its first plan the trip chooses among the planner's ways
        (Planner.alternatives, route.choose: a way through a move she hasn't set off on for route.STALE_S has
        route.FLOOR at least). Replanned, it goes the way it chose (Planner.through the way's move) until she has set
        off on that move, unless from where she is that costs more than the quickest by more than the ways offered
        could (route.STRETCH of the quickest's time when it chose: a share of what's left would drop a way whose
        detour is near the end, as 23 -> 17 by the rope hops at the end of a trip from 35), or she is past the move
        (on a platform the way reaches from it on), or the way from where she is goes back (_goes_back): then, as
        with the quickest chosen, the quickest from where she is. (Kept whatever it took, from 34 to 2 by the hops,
        ended on 17 by the climb past 23's end, she was sent down to 23 to hop back up; off the way at the quickest's
        landings, 677 of 5,112 replans that kept a way chosen on the map went back onto a platform she had been on.) A trip
        `through` her drops: its way through them (`_way_through`)."""
        if st.through is not None:
            return self._way_through(st, src, x), None
        penalty = self._penalty(st)
        if route.ROUTES <= 1 or not st.vary:
            return self.planner.plan(src, x, dst, weight=self.weight, penalty=penalty, dst_x=st.x if st.land_near else None), None
        c = st.choice
        if not c.made:
            c.made = True
            ways = self.planner.alternatives(src, x, dst, penalty, self.weight)
            if not ways:
                return None, None
            i, chances = route.choose(ways, self.rng, stale=self._stale)
            c.via, c.leeway = ways[i].via, route.STRETCH * ways[0].cost
            if i:
                v = ways[i].via
                c.beyond = frozenset(m.dst for m in ways[i].moves[ways[i].moves.index(v):])
                self.log(f"  of {len(ways)} ways, the one through {v.kind} {v.src}->{v.dst} "
                         f"({ways[i].cost / ways[0].cost - 1:+.0%}, chance {chances[i]:.0%})")
            return list(ways[i].moves), (ways, chances, i)
        best = self.planner.quickest(src, x, dst, penalty, self.weight)
        if c.via is not None and best is not None:
            way = None if src in c.beyond else self.planner.through(src, x, dst, c.via, penalty, self.weight)
            if way is not None and way.cost - best.cost <= c.leeway and not _goes_back(way, best, src, c.left):
                return list(way.moves), None
            c.via = None
        return (None if best is None else list(best.moves)), None

    def _way_through(self, st: TripState, src: int, x: float) -> list[Move]:
        """Trip `st`'s route from (platform src, x) through its stops (`through`): to each the route's moves, weighed and
        penalized as any trip's, the quickest with her walk on from where they land her, then a 'walk' to it, as far as there's a
        way. A walk's x is its goal, as a move's is where it starts (what her way to the route's next move is asked of it)."""
        path, place = [], self.model.place_of
        src, penalty = place(src), self._penalty(st)
        st.stops = [(place(p), g, w) for p, g, w in st.stops]  # (a twin's the platform it lies on: as her moves have it, `_gone_by`)
        for pid, goal, stand in st.stops:
            if pid != src:
                moves = self.planner.plan(src, x, pid, penalty, self.weight, dst_x=goal)
                if moves is None:
                    break
                path += moves
                src, x = pid, moves[-1].land_x
            path.append(Move("walk", pid, pid, goal, goal, directions.toward(goal - x), stand))
            x = goal
        return path

    def _stale(self, m: Move) -> bool:
        """Whether she hasn't set off on move m for route.STALE_S (this navigator's memory: since it began, if
        never)."""
        return time.monotonic() - self._used.get(_key(m), self._since) >= route.STALE_S

    def _made(self, st: TripState, moved: list[Move], arrived: bool = True) -> None:
        """Note the moves she set off on (route.STALE_S), and the trip's way (_gone_by); the fights and shots at a monster
        in the way start again from none (MAX_FIGHTS), and once they got her where they were going (`arrived`: not cut
        short by a touch, say), so does the time her chains have been stopping short of one (MEET_S), and (BLOCK_MEMORY)
        the evade step's budget of each move made."""
        now = time.monotonic()
        for m in moved:
            self._used[_key(m)] = now
        self._gone_by(st, moved)
        st.fights, st.shots, st.land_waits = 0, 0, 0
        if arrived:
            st.blocked = None
            if BLOCK_MEMORY:
                for m in moved:
                    st.blocks.pop(_key(m), None)
                    if st.evade_key == _key(m):
                        st.evade_key, st.evade_since, st.refuges = None, None, 0

    @staticmethod
    def _gone_by(st: TripState, moved: list[Move]) -> None:
        """The moves she set off on, for the way the trip chose: the platforms she set them off from are behind her
        (Choice.left); with its move among them, it has gone the way it chose, and goes on by the quickest. A walk made is
        the stop it goes to done, on a trip `through` her drops."""
        for m in moved:
            if m.kind == "walk" and st.stops and st.stops[0][:2] == (m.src, m.land_x):
                st.stops = st.stops[1:]
        c = st.choice
        if c.via is None:
            return
        c.left.update(m.src for m in moved)
        if any(_key(m) == _key(c.via) for m in moved):
            c.via = None

    @staticmethod
    def _penalty(st: TripState) -> Callable[[Move], float]:
        """The trip's route penalty: AVOID_SECONDS for a move a monster was in the way of lately (st.avoid),
        route.KEEP_OFF_S for one onto a platform it keeps off (st.keep_off), and CANNOT_S for one the mover couldn't
        do (st.cannot)."""
        off = keeping_off(st.keep_off, st.dst)
        return lambda m: (AVOID_SECONDS if _key(m) in st.avoid else 0.0) + off(m) \
            + (CANNOT_S if _key(m) in st.cannot else 0.0)

    def _clear_the_way(self, st: TripState, s: Scene, x: float) -> bool:
        """With an escort: shoot a monster on her platform in her way to map x (where the route's first move
        starts, or where she's sent to on arrival), up to MAX_FIGHTS times in a row without a move between, and
        then not again till she has moved (_made); its kills count in the trip's. In transit the fight doesn't go on to
        finish what she has hurt (FightOnTheWay.clear_way). True if she fought. (The count started again once it had
        run out, so a chain then stopped short of a monster whose touch counts, _meet, set off three more fights at the
        floor's Slimes: from Tree Dungeon's floor a trip to rest ran out its 180 s that way, simulator, loop_tree_walls
        seed 10.)"""
        if not self.fight:
            return False
        m = self.escort.in_way(s, x) if st.through is not None else None
        if m is not None and self._no_shot(s, m):  # (MAX_FIGHTS is a trip's and a chain's a trip: without this for ever, 0.46 s a time)
            return False
        thrown = self.escort.throws() if m is not None else 0
        if st.fights < MAX_FIGHTS and (kills := self.escort.clear_way(s, x, finish=st.mode != TRANSIT)) is not None:
            st.fights += 1
            st.trip.kills += kills
            if m is not None and m.track is not None and self.escort.throws() == thrown:
                self._unshot[m.track] = time.monotonic() + UNSHOT_S
            return True
        if st.fights == MAX_FIGHTS:
            telemetry.emit("event", kind="backstop", text="MAX_FIGHTS: moving on past the monster in the way")
            st.fights += 1  # (once: no more fights in her way till she has moved, _made)
        elif st.fights < MAX_FIGHTS:
            st.fights = 0  # nothing in her way
        return False

    def _no_shot(self, s: Scene, m: Monster) -> bool:
        """On a trip through her drops: whether a fight for monster `m` would say "nothing in reach" and cost her 0.4 s for nothing: the
        fights won't have it (Escort.wont_have), or one has just thrown nothing at it (UNSHOT_S)."""
        return self.escort.wont_have(s, m) or self._unshot.get(m.track, -1e18) > time.monotonic()

    def _chain_monsters(self, st: TripState, s: Scene, path: list[Move],
                        start: float | None = None) -> tuple[list, list | None]:
        """The monsters the chain keeps clear of, and those its run on past its end mustn't take her into (follow's
        tail; None: not tracked here, so the caller tracks the look's if it needs them): the look's state's, all of them
        (trip_monsters: the whole screen's), each walking as perception has it head at its kind's speed from the game
        data (_walking), with her memory's where the chain goes that the look didn't find (`remembered`), the feed's
        out of sight (`hidden_tracks`) and the one the last cut was for. Without an escort, every one of them both ways.
        With an escort, those of the kinds whose touch counts (Escort.threat), and every one of them for the tail; on a
        map where no kind counts (the first map), none, as before: the chain is planned blind (it fights them instead);
        its tail and whether the next chain follows on (_queue_next) are judged against every one, by her box
        (motion.bodies). While her invulnerability after a touch lasts (POST_HIT) the pass's sightings aren't kept
        (`st.sighted`: no shot first at one of them, _meet's, _shoot_ahead's or _clear_landing's). `start`: when the
        chain's plan time 0 is (time.monotonic; now by default): a chain replanned while one runs (REPLAN: _switch), which
        leaves the pass's own (its sightings, the cut's monster, the mover's plan time, the 'monsters' record) as they
        are."""
        if self.fight and not self._threats():
            return [], None
        replanned, start = start is not None, time.monotonic() if start is None else start
        platforms = _platforms(path)
        unseen = self.fight or UNSEEN_AVOID  # (without an escort too: R5)
        seen = self._on_platforms(self.trip_monsters(s)) if self.fight or UNSEEN_AVOID else s.monsters
        tracks = self.tracker.update(s.t, seen, seen_at=s.t - start)
        placed = [m for m in seen if m.platform is not None]  # (as the tracker keeps them)
        if self.fight or UNSEEN_AVOID:  # walking as the feed has them head, at their kind's speed
            tracks = self._walking(s, tracks, placed)
        if not replanned:
            st.sighted = [] if self._post_hit(st) else list(zip(tracks, placed))
        if unseen:  # and those her memory has there that the look didn't find (not into the tracker's)
            tracks = tracks + self.remembered(s, platforms, seen, s.t - start)
        if unseen:  # and the feed's out of sight, and the one the last cut was for (not into the tracker's)
            tracks = tracks + self.hidden_tracks(s, platforms, tracks, None if replanned else st.came, start=start)
        if not replanned:
            st.came = None
        if (st.toucher or TOUCHER_ALL and not self.fight)                 and (toucher := self._toucher(s, tracks, standing=not st.toucher and TOUCHER_STAND)) is not None:
            # (R7: her retreat; TOUCHER_ALL, every trip without an escort)
            tracks = tracks + [toucher]
        if not replanned:
            self._tracks_at = start  # (plan time 0 of these: the mover's checks, Mover.clear)
            if telemetry.recording():
                telemetry.emit("monsters", **tracks_record(tracks, self.model))
        if not self.fight:
            return tracks, tracks
        return [t for t in tracks if self.escort.threat(t.kind)], tracks

    def _threats(self) -> bool:
        """Whether any kind of the map's monsters counts (Escort.threat), worked out once. A kind that counts and hops
        (MapModel.mob_hops) is said: its tracks don't follow it off its platform (MonsterTrack.reach), and no map of
        hers has one yet."""
        if self._threat_kinds is None:
            kinds = {kind_of(name) for names in self.model.monsters_by_platform.values() for name in names}
            self._threat_kinds = frozenset(k for k in kinds if self.escort is not None and self.escort.threat(k))
            if hoppers := sorted({kind_of(n) for n, hops in self.model.mob_hops.items() if hops} & self._threat_kinds):
                self.log(f"  kinds that hop and whose touch counts ({', '.join(hoppers)}): her trips plan as if they "
                         f"never left their platforms (MonsterTrack.reach)")
        return bool(self._threat_kinds)

    def trip_monsters(self, s: Scene) -> list[Monster]:
        """The monsters of look `s`'s state that a trip's chains are planned by (with an escort, those whose touch counts
        are kept clear of: its chains stop short of a leg one of them could touch her on): all of them, the whole
        screen's (every frame perception gives has them since 2026-10-30): a whole-screen look's own, a band look's (a
        fight's: her stars reach 60 up and down, and the fights decide by the band) with those off its band (Scene.swept,
        Feed.look). None searched again: until 2026-10-30 a band look's frame was searched again in this process over
        the rows the chain's platforms take (Eyes.monsters, 23 ms at 1366x768). What her memory has off the look's view
        comes on top (`remembered`)."""
        return list(s.monsters) + [m for m in getattr(s, "swept", ()) if not any(m is k for k in s.monsters)]

    def remembered(self, s: Scene, platforms, seen: list[Monster], seen_at: float) -> list[MonsterTrack]:
        """The monsters her memory has (`memory`, at memory.SURE) on `platforms` (where a fighting trip's chain goes)
        that look `s` didn't find (`seen`: one of a kind on the same platform within what it can have walked since, as
        the memory matches a sighting, is it), seen within TRIP_MEMORY_S before it: off its view, and in it too (the
        detector misses one near her more often than not: TRIP_MEMORY_S). As tracks as of their last sighting
        (`seen_at`: the look's time in the plan's, less their age), their walking not known: wherever they can have
        walked since at their kind's speed. None without a memory."""
        if self.memory is None or s.where is None:
            return []
        view = None
        if not UNSEEN_TRIPS and s.tag is not None and getattr(s.frame, "shape", None) is not None:
            view = s.map_x(0) + EDGE, s.map_x(s.frame.shape[1]) - EDGE  # (as before: off its view only)
        out = []
        for e in self.memory.alive(s.t, SURE):
            age = s.t - e.seen
            if e.platform not in platforms or not 0 < age <= TRIP_MEMORY_S:
                continue
            half, top = touch_box(e.kind, self.tracker.body.get(e.kind))
            if view is not None and view[0] <= e.x - half and e.x + half <= view[1]:
                continue
            speed = self.tracker.walking.get(e.kind, DEFAULT_SPEED)
            slack = MATCH_SLACK + MATCH_SPEED * speed * min(age, MATCH_WALK_S)
            if any(m.kind == e.kind and m.platform == e.platform and abs(m.x - e.x) <= slack for m in seen):
                continue
            out.append(MonsterTrack(e.platform, e.x, speed, 0, half, None, seen_at - age,
                                    MONSTER_HEIGHT.get(e.kind, top), 0, e.kind, top))
        return out

    def _toucher(self, s: Scene, tracks: list[MonsterTrack], standing: bool = False) -> MonsterTrack | None:
        """R7: a stand-in for the monster that touched her last (Senses.touch, not `crowded`, TOUCHER_S old at most)
        where look `s` has none of `tracks` on her platform within TOUCHER_NEAR of her: at her x as of the touch's
        popup, its walking not known (`standing`: standing there, TOUCHER_ALL's), of the map's kind that touches hardest
        (its game data's PADamage); else None."""
        touch = fresh_touch(self.senses, s)
        w = s.where
        if touch is None or any(t.platform == w.platform.id and abs(t.x - w.x) <= TOUCHER_NEAR for t in tracks):
            return None
        names = {n for ns in self.model.monsters_by_platform.values() for n in ns}
        if not names:
            return None
        kind = kind_of(max(names, key=lambda n: (self.model.mob_damage.get(n, 0.0), n)))
        half, top = touch_box(kind, self.tracker.body.get(kind))
        return MonsterTrack(w.platform.id, w.x, self.tracker.walking.get(kind, DEFAULT_SPEED), 0, half,
                            0.0 if standing else None, touch["at"] - time.monotonic(), MONSTER_HEIGHT.get(kind, top), 0,
                            kind, top)

    def _on_platforms(self, monsters: list[Monster]) -> list[Monster]:
        """A look's monsters, one placed on no platform (in the air: a slime's hop, a knock; R5) put on the platform under
        its feet within PLACE_REACH (MapModel.platform_under), if there's one."""
        out = []
        for m in monsters:
            if m.platform is None and (q := self.model.platform_under(m.x, m.y, reach=PLACE_REACH)) is not None:
                m = replace(m, platform=q.id)
            out.append(m)
        return out

    def _walking(self, s: Scene, tracks: list[MonsterTrack], placed: list[Monster]) -> list[MonsterTrack]:
        """A look's tracks (`tracks`, one each of `placed`) walking as the feed follows each (Monster.track, the look's
        Scene.tracks): the way perception has it head at its kind's speed from the game data (tracking's velocity,
        monsters.walk_vx: 0 in a stand frame or flinching; under STILL_VX it stands), as _track_of has a track, over
        what the tracker measured between looks (the user, 2026-10-30: "Perception should only give mob heading, speed
        should come from data-mined game files"); one standing or flinching, either way at its speed STAND_UNSURE_S at
        the most (its heading not known: it may set off any time); one the feed doesn't follow as the tracker has it.
        Until 2026-10-30 only without an escort (R5), and only where the tracker couldn't tell yet (vx None): a fighting
        trip's monster seen once was planned for wherever it could walk either way."""
        feed = {tr.id: tr for tr in (getattr(s, "tracks", None) or ())}
        if not feed:
            return tracks
        out = []
        for t, m in zip(tracks, placed):
            tr = feed.get(getattr(m, "track", None))
            if tr is not None and getattr(tr, "vx", None) is not None:
                if abs(tr.vx) > STILL_VX:
                    vx = max(-t.speed, min(t.speed, tr.vx))
                    t = replace(t, vx=vx, heading=1 if vx > 0 else -1, track=tr.id, trust=HEADING_TRUST_S)
                elif _may_set_off(tr):  # (standing or flinching: either way, STAND_UNSURE_S at the most)
                    t = replace(t, vx=None, heading=0, track=tr.id, unsure=STAND_UNSURE_S)
                else:  # (dying, or its pose not told: where it was, as before)
                    t = replace(t, vx=0.0, heading=0, track=tr.id, trust=HEADING_TRUST_S)
            out.append(t)
        return out

    def _ghost(self, s: Scene, tr, now: float) -> bool:
        """The ghost guard (R5, without an escort): the feed's track `tr`, out of sight GHOST_S or more, on her (its box
        on hers where look `s` has her), with no touch told in GHOST_S and her not invulnerable: it isn't there (it would
        have touched her)."""
        w = s.where
        if self.fight or w is None or w.platform is None or tr.platform != w.platform.id or now - tr.last_seen < GHOST_S:
            return False
        half, _ = touch_box(tr.kind, self.tracker.body.get(tr.kind))
        if abs(tr.x - w.x) > half + her_body.HER_HALF:
            return False
        touch = getattr(self.senses, "touch", None)
        told = touch.get("told", -1e18) if isinstance(touch, dict) else -1e18
        until = getattr(self.senses, "invulnerable_until", None)
        return s.t - told > GHOST_S and (until is None or until() <= now)

    def hidden_tracks(self, s: Scene, platforms, tracks: list[MonsterTrack], came=None,
                      start: float | None = None) -> list[MonsterTrack]:
        """The monsters the feed follows out of sight on screen as of look `s` that count as about (localization.unseen.
        hidden: last seen within its TRACK_S, none that a kill of hers is on: her memory's `kills`, the ledger's) on
        `platforms`, and `came` (the feed's track of the monster the last chain was cut for, kept for this one:
        TripState.came; seen within tracking.HIDE_S) wherever the feed has it, as tracks walking as followed
        (_track_of), of the kinds the chain keeps clear of; none that one of `tracks` (the look's and her memory's, by
        kind and platform within what it can have walked since: memory.MATCH_SLACK) is. `start`: when the plan they're
        for starts (time.monotonic; now by default)."""
        out: list[MonsterTrack] = []
        if not UNSEEN_TRIPS:
            return out
        now = time.monotonic()
        start = now if start is None else start
        kills = getattr(self.memory, "kills", None)
        cands = hidden(s, kills() if kills is not None else (), platforms)
        if came is not None and not any(tr.id == came.id for tr in cands) and 0 <= s.t - came.last_seen <= HIDE_S:
            cands.append(came)
        for tr in cands:
            if tr.platform is None or tr.platform not in platforms \
                    or (self.escort is not None and not self.escort.threat(tr.kind)):
                continue
            if UNSEEN_AVOID and self._ghost(s, tr, now):
                continue
            speed = self.tracker.walking.get(tr.kind, DEFAULT_SPEED)
            age = now - tr.last_seen
            slack = MATCH_SLACK + MATCH_SPEED * speed * min(age, MATCH_WALK_S)
            p = self.model.platform(tr.platform)
            if any(mo.kind == tr.kind and mo.platform == tr.platform and abs(mo.at(0.0, p) - tr.x) <= slack
                   for mo in tracks + out):
                continue
            out.append(self._track_of(tr, start))
        return out

    def _start(self, st: TripState, s: Scene) -> Body:
        """Her body where a chain from look `s` starts: standing there, still settling right after a plan (the first
        jump or drop waits: SETTLE_S)."""
        settle = max(0.0, SETTLE_S - (time.monotonic() - st.moved_at))
        return Body(0.0, s.where.x, s.where.platform, 0, settle > 0, settle)

    def _chain(self, st: TripState, s: Scene, path: list[Move], monsters: list, looked: float,
               hops: bool | None = None, priced: list | None = None) -> tuple:
        """As many of the route's moves as can be predicted, chained from where she stands (flow.chain:
        the legs, their keys, and why the chain ended). The time since the look (`looked`), or since the last chain
        planned from it (a pass's replans: _meet), counts as thinking. `hops`: over a monster in the way of a run (by
        default only without an escort). Its last landing priced against `priced` (_pricing; `monsters` by default: one
        gone past regardless is kept clear of no more, but she still lands as clear of it as she can)."""
        # Right after a plan she may still be sliding or landing: the first jump/drop waits. A chain cut off
        # for length doesn't stop her at its end: the next one may follow on (follow).
        price, alts, goal = self._pricing(st, monsters if priced is None else priced)
        legs, keys, why = self.flow.chain(path, self._start(st, s), monsters, keep_going=True,
                                          hops=not self.fight if hops is None else hops, price=price, alts=alts,
                                          goal=goal)
        self._kept = list(monsters)  # (what it keeps clear of: _immunity_spent)
        now = time.monotonic()
        st.trip.time["think"] += now - max(looked, st.thought_to)
        st.thought_to = now
        return legs, keys, why

    def _pricing(self, st: TripState, monsters: list) -> tuple:
        """What flow.chain prices a chain's landings by (LANDING_CLEAR, with an escort that prices them: Escort.
        landings): the price of a leg's landing against `monsters` (s; with none, the spawn points the escort's judge
        prices too), the route's other moves between the same platforms (Planner.moves), and where she's sent on the
        trip's platform ((platform, x), or None); all None without."""
        landings = getattr(self.escort, "landings", None) if LANDING_CLEAR and self.fight else None
        judge = landings(list(monsters)) if landings is not None else None
        if judge is None:
            return None, None, None
        if self._siblings is None:
            self._siblings = {}
            for m in self.planner.moves:
                self._siblings.setdefault((m.src, m.dst), []).append(m)
        cannot = st.cannot if ROPE_DROP else ()  # (a drop that ended on a rope isn't another way either: ROPE_DROP)
        return ((lambda leg, last=True: judge(leg, last)[0]),
                (lambda m: [o for o in self._siblings.get((m.src, m.dst), []) if _key(o) not in cannot]),
                None if st.x is None else (st.dst, st.x))

    def _clear_landing(self, st: TripState, s: Scene, path: list[Move], monsters: list, legs: list[Leg],
                       why: str) -> bool:
        """With an escort, a chain that ends the trip (onto its platform) landing with a price (Escort.landings: her first
        attack there not sure to take the nearest monster, or one walking into her first), the cheapest flow.chain
        found: the monster making it so is shot first where her attack reaches it (Escort.shot_place, the fights' own
        reach): from where she stands, from the ground or a jump (Escort.clear_for with `jump`), else from along her
        platform within SHOOT_WALK (Escort.shoot_from: "shooting it first from x N"), MAX_FIGHTS tries in a row at most
        as _meet's; a try that threw nothing (Escort.throws: the fight's or the walk's look not having it in reach after
        all) is said so ("none thrown at the K: ..."), counted as a try, and looked at again, not as a shot; where
        there's no such shot, the log says why ("no shot first, no throw from the ground takes it from anywhere on
        platform 21 (it's 125 under her), nor one from a jump where she stands") and nothing is sent; then it's waited
        out if, walking as seen, it clears the landing within WAIT_AHEAD (LAND_WAITS looks at most, where none can reach
        her meanwhile); else she goes ("landing on P by the K anyway: why"). Each a 'landing' event (shot first, none
        thrown, waited, went), with why there was no shot (`shot`) and why she didn't wait (`why`). True to look
        again."""
        landings = getattr(self.escort, "landings", None)
        if not (LANDING_CLEAR and legs and why == "done" and landings is not None and monsters) \
                or legs[-1].end.platform.id != st.dst or legs[-1].move.kind == "walk":
            return False
        judge = landings(list(monsters))
        if judge is None:
            return False
        price, mo = judge(legs[-1])
        if price <= 0 or mo is None:
            return False
        q = legs[-1].end.platform
        kind = mo.kind or "monster"
        seen = next((m for t, m in st.sighted if t is mo), None)
        # The shot first: only one her attack can make (Escort.shot_place, the fights' own reach), from where she stands
        # or from along her platform. Live on 2026-10-28 at 00:38:36 the drop 21 -> 23 (117 down, past a star's band)
        # logged "shooting it first" with nothing to throw: the line came before clear_for and shoot_from, which both
        # found no throw. `shot`: why there's none (the log's and the events').
        if seen is None:
            shot = "not in this look"  # (just touched, POST_HIT, or a track the look didn't have: nothing to aim at)
        elif st.shots >= MAX_FIGHTS:
            shot = f"{st.shots} shots in a row already (MAX_FIGHTS)"
        else:
            place, shot = self.escort.shot_place(s, seen, SHOOT_WALK if LANDING_WALK else 0.0)
            if place is None:
                self.log(f"  {seen.kind} at x {seen.x:.0f} by her landing on {q.id}: no shot first, {shot}")
            else:
                finish = st.mode != TRANSIT
                walk = abs(place - s.where.x) >= 1e-6
                where = f" from x {place:.0f}" if walk else ""
                self.log(f"  {seen.kind} at x {seen.x:.0f} by her landing on {q.id}: shooting it first{where}")
                before = self.escort.throws()
                kills = self.escort.shoot_from(s, seen, SHOOT_WALK, finish=finish) if walk \
                    else self.escort.clear_for(s, seen, finish=finish, jump=True)
                thrown = self.escort.throws() - before
                if kills is not None:
                    # A try either way (MAX_FIGHTS bounds them; she may have walked or stepped, and time has passed:
                    # look again), a shot only where it threw (as first built every try was 'shot first', thrown or
                    # not: the review's, a walk a few units short of its place, or the fight's own looks without it).
                    st.shots += 1
                    st.trip.kills += kills
                    if thrown or kills:
                        telemetry.emit("event", kind="landing", text=f"shot first: {seen.kind} on {q.id} by the "
                                       f"landing{where}", price=round(price, 2), at=round(place, 1), throws=thrown,
                                       kills=kills)
                    else:
                        none = "the walk's look didn't have it in reach" if walk \
                            else "the fight's looks didn't have it in reach"
                        self.log(f"  none thrown at the {seen.kind} by her landing on {q.id}: {none}")
                        telemetry.emit("event", kind="landing", text=f"none thrown: {seen.kind} on {q.id} by the "
                                       f"landing{where}", price=round(price, 2), at=round(place, 1), why=none)
                    return True
                shot = "her attack didn't reach it after all"  # (shot_place said it would: not expected)
        # Then waited out, a bounded wait (LAND_WAITS looks), where its walk as seen clears the landing within
        # WAIT_AHEAD and none can reach her meanwhile; else she goes, priced.
        if st.land_waits >= LAND_WAITS:
            why = f"waited {st.land_waits} looks for it already (LAND_WAITS)"
        elif not self._safe_to_stand(s.where.x, s.where.platform, path[0], monsters):
            why = "not safe to stand and wait here"
        else:
            why = f"its walk as seen doesn't clear the landing within {WAIT_AHEAD:.1f} s"
            delay, last = WAIT_S, legs[-1]
            while delay <= WAIT_AHEAD + 1e-9:
                # (the same landing `delay` later: the monsters walked on, their sightings as old as they are; no
                # monster seen making it priced then, what the looks don't have aside)
                if judge(replace(last, end=replace(last.end, t=last.end.t + delay)))[1] is None:
                    if st.instead_of_waiting is not None and st.instead_of_waiting(s, q.id):
                        st.stop_waiting = True
                        return True
                    st.land_waits += 1
                    st.trip.waits += 1
                    self.log(f"  waiting for the {kind} by her landing on {q.id} to move off ({st.land_waits})")
                    telemetry.emit("event", kind="landing", text=f"waited: {mo.kind} on {q.id} by the landing",
                                   shot=shot, price=round(price, 2), clears_s=round(delay, 2))
                    time.sleep(WAIT_S)
                    return True
                delay += WAIT_S
        self.log(f"  landing on {q.id} by the {kind} anyway (price {price:.1f} s): {why}")
        telemetry.emit("event", kind="landing", text=f"went: {mo.kind} on {q.id} by the landing (price {price:.1f} s)",
                       shot=shot, why=why, price=round(price, 2))
        return False

    def _meet(self, st: TripState, s: Scene, path: list[Move], monsters: list, legs: list[Leg], why: str,
              planned_at: float, looked: float):
        """With an escort: a monster whose touch counts could touch her on the route's first move (nothing chained
        because of it: flow.in_the_way; one on a later leg is shot from here if a star reaches it, _shoot_ahead, else
        met where the chain stops). She shoots it if a star reaches it from where she stands (Escort.clear_for;
        MAX_FIGHTS shots in a row at most, without a move between), else the move costs extra from `planned_at` on so
        a way around is taken if there is one (AVOID_SECONDS, as without an escort), else she hops it (one in the way of
        a run along her platform: flow's hop), else she waits for it to move off (WAIT_LOOKS looks at most, only where
        none can reach her meanwhile, and only while waiting can clear the way: _worth_waiting), else she goes anyway:
        the chain planned as if that monster weren't there (the others still kept clear of). Her chains stop short of
        such monsters MEET_S at most without a move that gets her where it was going (this one or one on a later leg):
        then she goes regardless, the chain planned blind (_regardless). True to look again; a
        chain (legs, keys, why) to carry out (a hop over it, or past it regardless); None to go on as planned."""
        if why != "monster":
            st.waited = 0
            return None
        st.blocked = time.monotonic() if st.blocked is None else st.blocked
        if time.monotonic() - st.blocked >= MEET_S:
            nxt = path[min(len(path) - 1, sum(1 for leg in legs if leg.move.kind != "hop"))]
            return self._regardless(st, s, path, monsters, looked, None,
                                    f"MEET_S: {nxt.kind} -> {nxt.dst} past the monsters after {MEET_S:.0f} s", blind=True)
        if legs:  # stopped short of a later leg: shot from here if a star reaches it, else met where the chain stops
            st.waited = 0
            if st.through is not None:  # (her drops: as far as it's clear, and the next plan from there)
                return None
            return True if self._shoot_ahead(st, s, path, monsters, legs) else None
        move, start, linger = path[0], self._start(st, s), END_LINGER if len(path) == 1 else 0.0
        hit = self.flow.in_the_way(move, start, monsters, linger)
        if hit is None and not linger:  # the move is clear, standing where it ends isn't (flow.chain ends short of it)
            linger = END_LINGER
            hit = self.flow.in_the_way(move, start, monsters, linger)
        mo = hit[0] if hit is not None else None
        kind = mo.kind if mo is not None and mo.kind else "monster"
        seen = next((m for t, m in st.sighted if t is mo), None)  # as the look saw it (none just after a touch)
        for step in ("shoot", "around"):
            if step == "shoot" and seen is not None and st.shots < MAX_FIGHTS:
                if st.through is not None and self._no_shot(s, seen):
                    continue  # (a chain through her drops: no 0.4 s for "nothing in reach": the fights' own look won't have it)
                thrown = self.escort.throws() if st.through is not None else 0
                kills = self.escort.clear_for(s, seen, finish=st.mode != TRANSIT)
                place = getattr(self.escort, "shot_place", None) if kills is None and MEET_WALK else None
                if place is not None and place(s, seen, SHOOT_WALK)[0] is not None:  # (from along her platform)
                    kills = self.escort.shoot_from(s, seen, SHOOT_WALK, finish=st.mode != TRANSIT)
                if kills is not None:
                    st.shots += 1
                    st.trip.kills += kills
                    if st.through is not None and seen.track is not None and self.escort.throws() == thrown:
                        self._unshot[seen.track] = time.monotonic() + UNSHOT_S
                    return True
            if step == "around" and _key(move) not in st.avoid and move.kind != "walk":
                self.log(f"  a {kind} could touch her on {move.kind} -> {move.dst}: looking for a way around")
                self._around(st, s, move, monsters, planned_at)
                if self.keep_avoid:  # (no new look for it: this one plans the way around, a look's wait less)
                    st.fresh = s
                return True
        hopped = self._chain(st, s, path, monsters, looked, hops=True)
        if hopped[0]:
            return hopped
        if move.kind == "walk":  # (her drops: never past a monster regardless; no chain now, the hunt says what instead)
            st.walk_blocked = st.stop_waiting = True
            return True
        safe = worth = None  # (why she didn't wait, as far as it was asked: the backstop's record says)
        if st.waited < WAIT_LOOKS:
            safe = self._safe_to_stand(s.where.x, s.where.platform, move, monsters)
            worth = self._worth_waiting(move, start, monsters, linger, mo) if safe else None  # (None: not asked)
            if worth and st.instead_of_waiting is not None and st.instead_of_waiting(s, move.dst):
                st.stop_waiting = True
                return True
            if worth:
                st.waited += 1
                st.trip.waits += 1
                self.log(f"  waiting for the {kind} to move off ({st.waited})")
                time.sleep(WAIT_S)
                return True
        # (only where she can't stand and wait: with an escort, from a safe spot a refuge is waiting on for nothing)
        if EVADE and not self._safe_to_stand(s.where.x, s.where.platform, move, monsters)                 and self._evade_step(st, s, path, monsters, looked):
            return True  # look again: from the refuge, or after the chain the contact rule planned
        last_shot = LAST_SHOT and mo is not None
        if last_shot and self._last_shot(st, s, mo, planned_at):
            return True  # look again after the shot
        return self._regardless(st, s, path, monsters, looked, mo,
                                f"WAIT_LOOKS: {move.kind} -> {move.dst} past a {kind} regardless" if st.waited >= WAIT_LOOKS
                                else f"NO_WAIT: {move.kind} -> {move.dst} past a {kind} regardless",
                                why_not={"safe": safe, "worth": worth, "last_shot": last_shot})

    def _shoot_ahead(self, st: TripState, s: Scene, path: list[Move], monsters: list, legs: list[Leg]) -> bool:
        """A chain that stops short of a later leg for a monster whose touch counts (`legs`: as far as it goes): that
        monster shot from where she stands, if a star reaches it (Escort.clear_for; MAX_FIGHTS shots in a row at most,
        as _meet's), rather than from where the chain stops, which may be no better placed (and nearer it). True if
        she shot."""
        if st.shots >= MAX_FIGHTS:
            return False
        nxt = sum(1 for leg in legs if leg.move.kind != "hop")  # (hops are the chain's own, not the route's)
        if nxt >= len(path):
            return False
        hit = self.flow.in_the_way(path[nxt], legs[-1].end, monsters, END_LINGER if nxt == len(path) - 1 else 0.0) \
            or self.flow.in_the_way(path[nxt], legs[-1].end, monsters, END_LINGER)  # (or standing where it ends)
        seen = next((m for t, m in st.sighted if hit is not None and t is hit[0]), None)
        if seen is None:
            return False
        kills = self.escort.clear_for(s, seen, finish=st.mode != TRANSIT)
        if kills is None:
            return False
        st.shots += 1
        st.trip.kills += kills
        return True

    def _last_shot(self, st: TripState, s: Scene, mo: MonsterTrack, planned_at: float) -> bool:
        """_meet's shot before going past monster `mo` regardless: a look with the monsters (the pass's may have had
        none: just after a touch `st.sighted` is empty and she doesn't shoot, POST_HIT, and one her memory or the feed's hidden tracks have is in
        none), and if it has `mo` (one of its kind on its platform where it can be by now, MATCH_SLACK on: MonsterTrack.
        reach) and a star from where she stands reaches it, she shoots it (Escort.clear_for; MAX_FIGHTS shots in a row
        at most, as _meet's). True if she did. Live on 2026-10-25 28 passes regardless (NO_WAIT, WAIT_LOOKS) were
        followed by 4 touches within 3 s, two into a Green Mushroom she had seen (24's, from 25: 23:49:42.7, 23:50:05.7;
        D:/MapleHelper-scratch/analysis-1025/damage-skeptic): the shot needed it in the pass's own sightings."""
        if st.shots >= MAX_FIGHTS or self.escort is None or mo.platform is None or not mo.kind or self._post_hit(st):
            return False
        look = self.mover.locate(mobs=True, wide=not self.fight)
        if look.where is None or look.where.platform is None or look.where.platform.id != s.where.platform.id:
            return False
        p = self.model.platform(mo.platform)
        lo, hi = mo.reach(time.monotonic() - planned_at, p)
        near = [m for m in look.monsters if m.kind == mo.kind and m.platform == mo.platform
                and lo - MATCH_SLACK <= m.x <= hi + MATCH_SLACK]
        if not near:
            return False
        m = min(near, key=lambda m: abs(m.x - (lo + hi) / 2))
        kills = self.escort.clear_for(look, m, finish=st.mode != TRANSIT)
        if kills is None:
            return False
        st.shots += 1
        st.trip.kills += kills
        return True

    def _regardless(self, st: TripState, s: Scene, path: list[Move], monsters: list, looked: float,
                    mo: MonsterTrack | None, why: str, blind: bool = False, why_not: dict | None = None):
        """_meet's last resort: the chain planned as if monster `mo` weren't there (the others still kept clear of), or,
        `blind` (MEET_S), as before her chains kept clear of anything; None (the move on its own) if that chains nothing
        either. A backstop event says `why`. (Past MEET_S, kept clear of the others, the chain stopped short of the next
        one, where it could still touch her: up from Tree Dungeon's floor (33) she jumped onto the ledge 32 beside a
        Green Mushroom, short of the rope to 30 another walked the top of, was knocked back down, and did the same
        again for the 180 s of a trip to rest: simulator, loop_tree_run seed 4. Blind on every backstop, though, her
        hunt loop's trips were touched 14% more often than with only `mo` left out: harness, 20 seeds.)
        Its backstop event says why it came to this too (`_pass_record`; `why_not`: why _meet didn't wait, as far as it
        asked: safe to stand, worth waiting, and whether its last shot was tried)."""
        if st.through is not None:  # (her drops: a move past it regardless but no walk: as far as the route's moves, the next plan from there)
            path = path[:next((i for i, m in enumerate(path) if m.kind == "walk"), len(path))]
            if not path:
                return None
        telemetry.emit("event", kind="backstop", text=why, **self._pass_record(st, s, path[0] if path else None, mo,
                                                                               why_not))
        st.waited = 0
        st.refuge = None
        if not BLOCK_MEMORY:  # (kept till she makes the move: _made)
            st.evade_since, st.refuges = None, 0
        st.passed = (s.where.platform.id, time.monotonic())
        regardless = self._chain(st, s, path, [] if blind else [t for t in monsters if t is not mo], looked,
                                 priced=monsters)
        if not regardless[0] and EVADE and not blind:  # (never the mover for a move a monster is in the way of: blind)
            regardless = self._chain(st, s, path, [], looked, priced=monsters)
        return regardless if regardless[0] else None

    def _pass_record(self, st: TripState, s: Scene, move: Move | None, mo: MonsterTrack | None,
                     why_not: dict | None = None, waited: int | None = None) -> dict:
        """A pass regardless as its backstop event records it (from 2026-11-02; P04 and P28 of the recordings'
        analysis: 46-47 of the hour's 170 touches came within 3.5 s of one, and why each happened couldn't be told):
        the move (`move`: kind, src, dst, x); the monster in its way (`blocker`: kind, its platform and x, and where
        that is from hers, `rel`: same, above or below; none for MEET_S's blind pass); the looks waited for it
        (`waited`), the s her chains had stopped short of such monsters (`blocked_s`, MEET_S's clock), the shots at
        it in a row (`shots`); why she didn't wait (`why_not`: _meet's); and its episode (`episode`, a number, and
        `nth` in it): passes of one move within PASS_EPISODE_S of the last are one episode, so the touches after them
        are counted once an episode."""
        now = time.monotonic()
        out: dict = {"waited": st.waited if waited is None else waited, "shots": st.shots,
                     "blocked_s": None if st.blocked is None else round(now - st.blocked, 2)}
        if move is not None:
            out["move"] = [move.kind, move.src, move.dst, round(move.x, 1)]
            key, eps = (move.kind, move.src, move.dst), self.__dict__.setdefault("_episodes", {})
            ep = eps.get(key)
            if ep is None or now - ep[1] > PASS_EPISODE_S:
                self._episode_n = getattr(self, "_episode_n", 0) + 1
                ep = [self._episode_n, now, 0]
            ep[1], ep[2] = now, ep[2] + 1
            eps[key] = ep
            out["episode"], out["nth"] = ep[0], ep[2]
        if mo is not None:
            here = s.where.platform if s.where is not None else None
            rel = None
            q = None if here is None else here if mo.platform == here.id else here.peer(mo.platform)
            if q is not None:
                rel = "same" if q is here else "above" if q.top < here.top else "below"
            out["blocker"] = {"kind": mo.kind or None, "platform": mo.platform, "x": round(mo.x, 1), "rel": rel}
        if why_not:
            out["why_not"] = why_not
        return out

    def _worth_waiting(self, move: Move, start: Body, monsters: list, linger: float, mo: MonsterTrack | None) -> bool:
        """Whether waiting a moment can clear the way of `move` of the monster `mo` that could touch her on it: its
        walking isn't known yet (one look: the next tells which way it goes), it stands (monsters stand and walk by
        turns: it walks off sooner or later), or, walking as it was seen, it's clear of the move started WAIT_AHEAD s
        later at most. Waiting out a Green Mushroom walking on 18, which a climb from 19 or 20 reaches 1.9 s on, her
        looks were spent for nothing: wherever it was seen heading it could still be at the rope's top by then
        (25 + 20 units/s either way of where it's heading: MonsterTrack.reach); not waiting for one standing under a drop
        from 22 onto 23, she came down on it (simulator, seeds 1-16)."""
        if mo is None or not mo.vx:
            return True
        delay = WAIT_S
        while delay <= WAIT_AHEAD + 1e-9:
            later = [replace(t, seen_at=t.seen_at - delay) for t in monsters]
            if self.flow.in_the_way(move, start, later, linger) is None:
                return True
            delay += WAIT_S
        return False

    def _wait_for_monster(self, st: TripState, s: Scene, path: list[Move], monsters: list, legs: list[Leg],
                          why: str, planned_at: float, looked: float = 0.0) -> bool:
        """A monster in the way of the route's first move (nothing chained because of it): look for a way around (the
        move costs extra from `planned_at` on, and with EVADE each move of the way offered then that the same monsters
        block: _around), else wait for it to go, WAIT_LOOKS looks at most, where no monster can reach her meanwhile
        (_safe_to_stand) and, with EVADE, while waiting can clear the way (_worth_waiting); else the evade step (EVADE:
        _evade, a refuge or past it). True to look again; False to go on (with `legs`, or the move alone: the mover's,
        only for a move no chain can plan)."""
        move = path[0]
        if legs or why != "monster":
            st.waited = 0
            return False
        start, linger = self._start(st, s), END_LINGER if len(path) == 1 else 0.0
        if EVADE:
            self._evade_for(st, move)
        if EVADE and BUDGET_FROM_BLOCK and st.evade_since is None:  # (its waits count in the evade step's budget)
            st.evade_since = time.monotonic()
        safe_here = self._safe_to_stand(s.where.x, s.where.platform, move, monsters)
        hurry = EVADE and HURRY_SLACK > 0 and self._due(st, path)  # (no time for waits either: past it)
        # (BLOCK_MEMORY: a move whose budget a pass already spent gets no more waits: past it at once)
        spent = EVADE and BLOCK_MEMORY and BLOCK_NO_WAIT and st.evade_since is not None \
            and time.monotonic() - st.evade_since >= st.evade_s
        if st.waited < WAIT_LOOKS and safe_here and not hurry and not spent and (
                not (EVADE and WORTH_WAIT) or self._worth_waiting(move, start, monsters, linger,
                                                                  self._blocker(move, start, monsters, linger))):
            if _key(move) not in st.avoid:
                self.log(f"  a monster could get in the way of {move.kind} -> {move.dst}: looking for a way around")
                if EVADE and AROUND_ALL:
                    self._around(st, s, move, monsters, planned_at)
                else:
                    st.avoid[_key(move)] = planned_at
                return True
            st.waited += 1
            st.trip.waits += 1
            self.log(f"  waiting for the monster to move off ({st.waited})")
            time.sleep(WAIT_S)
            return True
        if not EVADE:
            st.waited = 0
            return False
        if safe_here and SAFE_PASS:  # (no refuge is safer than here: past it now, waiting more can't clear the way)
            return self._go_past(st, s, path, monsters, looked,
                                 f"WAIT_LOOKS: {st.waited} looks waited" if st.waited >= WAIT_LOOKS
                                 else "NO_WAIT: waiting can't clear the way")
        return self._evade(st, s, path, monsters, looked)

    def _blocker(self, move: Move, start: Body, monsters: list, linger: float) -> MonsterTrack | None:
        """The monster that stops a chain before `move` from `start` (flow.in_the_way, with `linger` there, else standing
        where it ends END_LINGER: chain's own), or None."""
        hit = self.flow.in_the_way(move, start, monsters, linger)
        if hit is None and not linger:
            hit = self.flow.in_the_way(move, start, monsters, END_LINGER)
        return None if hit is None else hit[0]

    def _first_blocker(self, path: list[Move], start: Body, monsters: list) -> MonsterTrack | None:
        """The monster that stops a chain of `path` from `start` short of its first move: walking the chain on (each
        move's first way clear of `monsters`, as flow.chain picks), the one in the way of the first move one is (a
        chain stopped there backs off every leg whose end one could reach her standing at, END_LINGER: so the first
        move's end being unsafe stops it before it too); else the one by the first move's end; None if none is."""
        b, by_end = start, None
        for i, m in enumerate(path[:MAX_LEGS]):
            linger = END_LINGER if i == len(path) - 1 else 0.0
            hit = self.flow.in_the_way(m, b, monsters, linger)
            if hit is not None:
                return hit[0]
            leg = None
            for v in Flow.VARIANTS:
                lg = self.flow.leg(m, b, Keys(self.flow.key_names), v)
                if lg is not None and self.flow.danger(lg, monsters, linger) is None:
                    leg = lg
                    break
            if leg is None:
                break
            if i == 0 and (stand := self.flow.danger(leg, monsters, linger=END_LINGER)) is not None:
                by_end = stand[0]
            b = leg.end
        return by_end

    # --- the evade step (EVADE) ------------------------------------------------------------------------------------

    def _evade(self, st: TripState, s: Scene, path: list[Move], monsters: list, looked: float) -> bool:
        """Without an escort, the route's first move blocked by a monster and waiting not allowed: the evade step
        (_evade_step: the contact rule, waiting at a refuge that stays safe, a refuge), else past it (_go_past). True
        to look again; False for the mover (a move no chain plans, even blind)."""
        if self._evade_step(st, s, path, monsters, looked):
            return True
        return self._go_past(st, s, path, monsters, looked, self._why_past(st, path))

    @staticmethod
    def _evade_for(st: TripState, move: Move) -> None:
        """BLOCK_MEMORY: trip `st`'s evade step state (evade_since, refuges) made blocked move `move`'s: as it was at its
        last block, unless that's BLOCK_FORGET_S ago or more (then none), the other move's put by."""
        if not BLOCK_MEMORY:
            return
        now, k = time.monotonic(), _key(move)
        if st.evade_key != k:
            if st.evade_key is not None:
                st.blocks[st.evade_key] = (st.evade_since, st.refuges, st.blocked_at)
                st.evade_since, st.refuges, st.blocked_at = st.blocks.pop(k, (None, 0, now))
            elif k in st.blocks:
                st.evade_since, st.refuges, st.blocked_at = st.blocks.pop(k)
            else:  # (none blocked since the last move made: as it is)
                st.blocked_at = now
            st.evade_key = k
        if now - st.blocked_at > BLOCK_FORGET_S:
            st.evade_since, st.refuges = None, 0
        st.blocked_at = now

    @staticmethod
    def _due(st: TripState, path: list[Move] | None = None) -> bool:
        """Whether the trip has too little time left to spend any on the evade step: EVADE_DEADLINE_S, or with
        HURRY_SLACK its route's time (`path`'s moves) that many times over and EVADE_DEADLINE_S; never for a trip that
        doesn't go past for its deadline (`deadline_pass`: a retreat after a stop)."""
        if not st.deadline_pass:
            return False
        left = st.deadline - time.monotonic()
        need = HURRY_SLACK * sum(m.seconds for m in path or ()) if HURRY_SLACK else 0.0
        return left <= EVADE_DEADLINE_S + need

    def _why_past(self, st: TripState, path: list[Move] | None = None) -> str:
        """Why the evade step gives way to a pass: its backstop's name, and why."""
        if st.past_why is not None:
            why, st.past_why = st.past_why, None
            return why
        now = time.monotonic()
        spent = now - (st.evade_since if st.evade_since is not None else now)
        if spent >= st.evade_s:
            return f"EVADE_S: {spent:.1f} s blocked"
        if self._due(st, path):
            return f"DEADLINE: {max(0.0, st.deadline - now):.1f} s left of the trip"
        if st.refuges >= EVADE_REFUGES:
            return f"EVADE_REFUGES: {st.refuges} refuges taken"
        return "NO_REFUGE: no refuge along her platform nor a move off it"

    def _evade_step(self, st: TripState, s: Scene, path: list[Move], monsters: list, looked: float) -> bool:
        """The evade step for the route's first move, blocked by a monster and waiting not allowed (EVADE): the contact
        rule (_contact: a touch she can't avoid taken as given, the chain planned with the window after it); else, at the
        refuge she went to and while it stays safe, a look again WAIT_S on; else a refuge (_refuge: a step along her
        platform, or a move off it), gone to as a watched chain; each within the budget (TripState.evade_s from the
        move's first block, BUDGET_FROM_BLOCK; EVADE_REFUGES; EVADE_DEADLINE_S left of the trip, _due). True if she did
        one (look again). With BLOCK_MEMORY the budget and refuges are the blocked move's (_evade_for); with REFUGE_WORTH
        at the refuge she waits only while waiting can clear the way (_worth_waiting), else it's past it now."""
        self._evade_for(st, path[0])
        now = time.monotonic()
        if st.evade_since is None:
            st.evade_since = now
        spent = now - st.evade_since
        move, p, x = path[0], s.where.platform, s.where.x
        if CONTACT and (hit := self._contact(st, s, move, monsters)) is not None:
            mo, t = hit
            st.contact_at = now
            self.flow.immune = t + INVULNERABLE_S - IMMUNE_MARGIN_S
            legs, keys, why = self._chain(st, s, path, monsters, looked)
            if legs:
                self._carry_evade(st, s, path, legs, keys, why, monsters, "contact",
                                  f"{move.kind} -> {move.dst}: the {mo.kind or 'monster'} touches her within "
                                  f"{CONTACT_S:g} s whatever she does: through in the window after it "
                                  f"({self.flow.immune:.2f} s)")
                return True
        budget = spent < st.evade_s and not self._due(st, path)
        if budget and st.refuge is not None and st.refuge[0] == p.id and abs(st.refuge[1] - x) <= ARRIVE_TOLERANCE \
                and self._safe_to_stand(x, p, move, monsters):
            if REFUGE_WORTH and not self.fight:
                start, linger = self._start(st, s), END_LINGER if len(path) == 1 else 0.0
                if not self._worth_waiting(move, start, monsters, linger, self._blocker(move, start, monsters, linger)):
                    st.past_why = f"NO_WAIT: at the refuge {spent:.1f} s, waiting can't clear the way"
                    return False
            st.trip.waits += 1
            self.log(f"  at the refuge at x {x:.0f}: waiting for the way to clear ({spent:.1f} s of {st.evade_s:g})")
            time.sleep(WAIT_S)
            return True
        if budget and st.refuges < EVADE_REFUGES and (ref := self._refuge(st, s, monsters)) is not None:
            step, legs, keys, text, to = ref
            st.refuges += 1
            st.refuge = to
            self._carry_evade(st, s, path, legs, keys, "done", monsters, step,
                              f"{text} (refuge {st.refuges}, {move.kind} -> {move.dst} blocked {spent:.1f} s)",
                              refuge=True)
            return True
        return False

    def _stand_until(self, t: float, hold: float) -> float:
        """Plan time a stand from `t` must be clear to: `hold` on, and her invulnerability's end and STAND_PAST_S at
        least (R3: no touch counts before then, so a stand ending earlier is no check)."""
        return max(t + hold, self.flow.immune + STAND_PAST_S) if STAND_WINDOW else t + hold

    def _refuge(self, st: TripState, s: Scene, monsters: list) -> tuple | None:
        """Where the evade step goes from look `s`: ('spot', legs, keys, why, (platform, x)) the nearest spot along her
        platform within EVADE_WALK (EVADE_STEP_MIN away at least, EVADE_IN in from its ends, every SPOT_STEP; as near,
        the one the hunter's danger model rates safer: `danger`) that a step there (flow._step) and standing there
        EVADE_HOLD_S (_stand_until) keep clear of every monster; else ('move', ...) the quickest single move off her
        platform that chains clear (flow.chain, no hop) and lands where she can stand as long, the trip st.refuge_s
        dearer at most; None if there's neither."""
        p, x0 = s.where.platform, s.where.x
        start = self._start(st, s)
        lo, hi = p.left + EVADE_IN, p.right - EVADE_IN
        spots = [lo + i * SPOT_STEP for i in range(int(max(0.0, hi - lo) // SPOT_STEP) + 1)]
        spots = [x for x in spots if EVADE_STEP_MIN <= abs(x - x0) <= EVADE_WALK]
        model = self._danger_model()

        def rank(x: float) -> tuple:
            near = round(abs(x - x0) / SPOT_STEP)
            return near, (model.at(p.id, x, s.monsters) if model is not None else 0.0), x

        for x in sorted(spots, key=rank):
            k = Keys(self.flow.key_names)
            t_stop, x_stop = Flow._step(Body(start.t, x0, p), x, k)
            d = "right" if x_stop > x0 else "left"
            step = Leg(Move("step", p.id, p.id, x0, x_stop, d), Body(start.t, x0, p), Body(t_stop, x_stop, p),
                       [Span(p.id, start.t, t_stop, x0, x_stop)])
            if self.flow.danger(step, monsters) is not None:
                continue
            until = self._stand_until(t_stop, EVADE_HOLD_S)
            stand = Leg(step.move, step.end, Body(until, x_stop, p), [Span(p.id, t_stop, until, x_stop, x_stop)])
            if self.flow.danger(stand, monsters) is not None:
                continue
            return "spot", [step], k, f"a step {d} to x {x_stop:.0f} on platform {p.id}", (p.id, x_stop)
        here = self.planner.quickest(p.id, x0, st.dst) if math.isfinite(st.refuge_s) else None
        best = None
        back = st.passed[0] if st.passed is not None and time.monotonic() - st.passed[1] < PASS_COMMIT_S else None
        for m in (self.planner.moves if REFUGE_MOVES or not math.isfinite(st.refuge_s) else ()):
            if m.src != p.id or m.dst == back:
                continue
            legs, keys, why = self.flow.chain([m], start, monsters, keep_going=False, hops=False)
            if not legs or why != "done":
                continue
            e = legs[-1].end
            if e.platform.id != st.dst and math.isfinite(st.refuge_s):
                there = self.planner.quickest(e.platform.id, e.x, st.dst)
                if here is None or there is None or e.t + there.cost > here.cost + st.refuge_s:
                    continue
            until = self._stand_until(e.t, EVADE_HOLD_S)
            stand = Leg(m, e, Body(until, e.x, e.platform), [Span(e.platform.id, e.t, until, e.x, e.x)])
            if self.flow.danger(stand, monsters) is not None:
                continue
            if best is None or e.t < best[1][-1].end.t:
                best = ("move", legs, keys, f"{m.kind} -> platform {m.dst} (x {e.x:.0f})", (e.platform.id, e.x))
        return best

    def _danger_model(self):
        """The hunter's danger model (attack.danger.Danger) where the navigator has one and a kind on the map counts,
        else None."""
        try:
            model = self.danger() if self.danger is not None else None
        except Exception:  # (one that can't be made: no tie-break, the nearest first)
            return None
        return model if model is not None and getattr(model, "any", False) else None

    def _contact(self, st: TripState, s: Scene, move: Move, monsters: list) -> tuple[MonsterTrack, float] | None:
        """The contact rule (CONTACT): a monster seen live (its sighting CONTACT_LIVE_S old at most) that touches her
        within CONTACT_S whatever she does now (standing, or a run either way: flow.danger), while she isn't
        invulnerable: it and when; None otherwise, and within CONTACT_AGAIN_S of the last touch it took as given unless
        a touch was told since."""
        if self.flow.immune > 0:
            return None
        if time.monotonic() - st.contact_at < CONTACT_AGAIN_S and self._touch_since(st.contact_at) is None:
            return None
        live = [t for t in monsters if t.seen_at >= -CONTACT_LIVE_S]
        if not live:
            return None
        p, x = s.where.platform, s.where.x
        b = Body(0.0, x, p)
        stand = Leg(move, b, Body(CONTACT_S, x, p), [Span(p.id, 0.0, CONTACT_S, x, x)])
        hit = self.flow.danger(stand, live)
        if hit is None:
            return None
        for d in (-1, 1):
            x1 = p.clamp(_ground(b, d, CONTACT_S)[0])
            run = Leg(move, b, Body(CONTACT_S, x1, p, run=d), [Span(p.id, 0.0, CONTACT_S, x, x1)])
            if self.flow.danger(run, live) is None:
                return None
        return hit

    def _go_past(self, st: TripState, s: Scene, path: list[Move], monsters: list, looked: float, why: str) -> bool:
        """Past the monsters in the way of the route's first move: the chain planned without them, one at a time
        (_blocker: the one that stops it, then the next), PASS_TRIES at most, else blind; said (`why`: the backstop's
        name and why), an `evade` event 'pass' and a backstop event, carried out and watched as any chain. False if
        nothing chains even so (the mover's move: watched, WATCH_MOVES). With PASS_WINDOW each is left out first only
        while her invulnerability lasts (_pass_windows: her window now, or the one its touch opens; then one touch more),
        and wholly only if nothing chains so."""
        move, start = path[0], self._start(st, s)
        others, past, legs, keys, chained, within = list(monsters), [], [], None, "done", None
        for _ in range(PASS_TRIES):
            mo = self._first_blocker(path, start, others)
            if mo is None:
                break
            past.append(mo)
            others = [t for t in others if t is not mo]
            legs, keys, chained = self._chain(st, s, path, others, looked, priced=monsters)
            if legs and PASS_WINDOW and (windowed := self._within_window(st, s, path, legs, others, mo, looked,
                                                                         monsters)) is not None:
                got, within = windowed
                if got is not None:
                    legs, keys, chained = got
            if legs:
                break
        blind = not legs
        if blind:
            legs, keys, chained = self._chain(st, s, path, [], looked, priced=monsters)
        waited = st.waited
        st.waited = 0
        st.refuge = None
        if not BLOCK_MEMORY:  # (kept till she makes the move: _made)
            st.evade_since, st.refuges = None, 0
        st.passed = (s.where.platform.id, time.monotonic())
        names = ", ".join(sorted({t.kind or "monster" for t in past})) or "monster"
        text = f"{why}; {move.kind} -> {move.dst} past the {names}" + (" (blind)" if blind else "") \
            + (f" (clear of it from {within:.2f} s)" if within is not None else "")
        telemetry.emit("event", kind="backstop", text=text,
                       **self._pass_record(st, s, move, past[0] if past else None, waited=waited))
        if not legs:
            self.log(f"  {text}: nothing chains; the move alone")
            return False
        self._carry_evade(st, s, path, legs, keys, chained, monsters, "pass", text)
        return True

    def _within_window(self, st: TripState, s: Scene, path: list[Move], legs: list[Leg], others: list,
                       mo: MonsterTrack, looked: float, monsters: list) -> tuple | None:
        """PASS_WINDOW: a pass whose chain `legs` was planned without monster `mo` in the way (`others` kept clear of):
        the chain that keeps clear of `mo` once the window its first touch of it opens is over (that touch's plan time,
        her invulnerability now counted, INVULNERABLE_S on less IMMUNE_MARGIN_S): `legs` if they do, else the chain
        planned with `mo` counting from then (its own first touch checked the same way, once more); ((legs, keys, why),
        that time), or None where `legs` never meet `mo` or nothing chains so (the pass leaves it out wholly, as
        before)."""
        w = INVULNERABLE_S - IMMUNE_MARGIN_S
        t = self._first_touch(legs, mo)
        if t is None:
            return None
        until = t + w
        if self._first_touch(legs, replace(mo, counts_from=until)) is None:
            return None, until  # (clear of it after its window as planned: kept)
        for _ in range(2):
            got = self._chain(st, s, path, others + [replace(mo, counts_from=until)], looked, priced=monsters)
            if not got[0]:
                return None
            t2 = self._first_touch(got[0], mo)
            if t2 is None or t2 + w >= until - 1e-9:
                return got, until
            until = t2 + w  # (it touches her sooner: her window ends sooner)
        return None

    def _first_touch(self, legs: list[Leg], mo: MonsterTrack) -> float | None:
        """When monster `mo` could first touch her on the chain of `legs` (plan time, her invulnerability counted: flow's
        danger, standing END_LINGER where it ends), or None."""
        for i, leg in enumerate(legs):
            hit = self.flow.danger(leg, [mo], linger=END_LINGER if i == len(legs) - 1 else 0.0)
            if hit is not None:
                return hit[1]
        return None

    def _carry_evade(self, st: TripState, s: Scene, path: list[Move], legs: list[Leg], keys, why: str, monsters: list,
                     step: str, text: str, refuge: bool = False) -> None:
        """An evade step's chain (`step`: 'spot' or 'move' for a refuge, 'contact', 'pass'), said (the log, an `evade`
        event) and carried out as a pass's chain is (_carry_out_chain: watched, cut on a touch), her landing noted. A
        refuge's moves don't count as getting her where the trip goes (MEET_S goes on: _made's `arrived`), and a step
        along her platform is no move of the route's."""
        self.log(f"  {step}: {text}")
        telemetry.emit("event", kind="evade", step=step, text=text)
        hp = s.hp
        st.path = path
        self.chain_still = None
        moved, planned, after = self._carry_out_chain(st, s, path, legs, keys, why, st.dst, monsters)
        if ROPE_TOP and self.chain_still is not None and (self.chain_still[1] >= STUCK_S or ROPE_CUT and self.cut_still):
            # (her dot never moved: on a rope under its top, not on the platform: off it, as _pass's chains are)
            after = self._rope_top(st, s, after, jumped=any(leg.air for leg in legs) or ROPE_CUT and self.cut_still)
        landed = after.where.platform is not None and after.where.platform.id == planned.id
        if PASS_AVOID and step == "pass" and (self._touch_since(s.t) is not None or after.hp < hp - HIT):
            # (a pass that's touched: its move dearer for a while, another way if there's one)
            st.avoid[_key(path[0])] = time.monotonic()
        self._made(st, [m for m in moved if m.kind != "step"], landed and not refuge)
        if self._note_landing(st.trip, moved, planned, after, hp, st.stopped, since=s.t):
            st.trip.misses += 1
            self._rope_drop(st, moved, planned, after)

    def _rope_drop(self, st: TripState, moved: list[Move], planned: Platform, after: Scene) -> None:
        """ROPE_DROP: a drop onto `planned` that ended on no platform (a miss: on a rope, `after` placing her on none)
        costs CANNOT_S for the rest of the trip (st.cannot): another way, if there's one, not the same drop again."""
        if not ROPE_DROP or after.where is None or after.where.platform is not None:
            return
        for m in moved:
            if m.kind == "drop" and m.dst == planned.id and _key(m) not in st.cannot:
                st.cannot.add(_key(m))
                self.log(f"    the drop {m.src} -> {m.dst} from x {m.x:.0f} ended on a rope: another way")
                telemetry.emit("event", kind="cannot", text=f"drop -> {m.dst}: ended on a rope")

    def _around(self, st: TripState, s: Scene, move: Move, monsters: list, planned_at: float) -> None:
        """Move `move` costs extra from `planned_at` on (st.avoid), and so does each move of the way the route planner
        offers then that the same monsters stop a chain before (as far as a chain plans it, without hops), AROUND_TRIES
        at most, without a look between: the next look plans the way around if there's one clear of them, else the
        same move again, which she then hops, waits out or goes on regardless. On Tree Dungeon a floor has several
        jumps up onto the platform above, each from its own takeoff: one look each, a trip to rest from 33 spent 1.1 s
        finding them all in the way of the same Green Mushroom (simulator, loop_tree_run). And a way around judged by
        its first move alone went round to the other rope up to 18, whose top the same Green walked too, and back,
        four times in a minute (19 -> 18 with a Green on 18, seeds 1-16: 9 waits and 23 s a trip on average)."""
        start = self._start(st, s)
        for _ in range(AROUND_TRIES):
            st.avoid[_key(move)] = planned_at
            path = self.planner.plan(s.where.platform.id, s.where.x, st.dst, weight=self.weight,
                                     penalty=self._penalty(st))
            if not path or _key(path[0]) in st.avoid:
                return
            legs, _, why = self.flow.chain(path, start, monsters, keep_going=True, hops=False)
            if why != "monster":
                return
            move = path[len(legs)]

    def _carry_out_chain(self, st: TripState, s: Scene, path: list[Move], legs: list[Leg], keys, why: str,
                         dst: int, monsters=(), replan: bool = False) -> tuple[list[Move], Platform, Scene]:
        """The chain in one plan, followed on by those planned on the move (follow), and (`replan`: a pass's own
        chain, REPLAN) replanned as it runs. Returns the moves made, the platform she should have ended on, and a look
        once it's over. Cut short, she replans from where she is (from where a touch knocked her: POST_HIT)."""
        immune = self.flow.immune
        self.log(f"  {platform_text(s.where.platform)} at x {s.where.x:.0f} ({s.how}): "
                 + _chain_text(legs, len(path), why) + (f" (invulnerable for {immune:.2f} s)" if immune else ""))
        t2 = time.monotonic()
        cut, legs = self.follow(legs, keys, s.hp, dst, why, st.trip, st, path, monsters, since=s.t, replan=replan)
        st.trip.time["move"] += time.monotonic() - t2
        st.moved_at = time.monotonic()
        if st.through is None or _arrows_in(legs):  # (a stand's presses alone: her facing, and the throw's retap, as they were)
            self.actions.facing = None
        if st.through is not None:
            st.ran += _ran_over(legs, self._cut_plan if cut else None)
            st.chain_why = cut or why
        after = (self._after_through(s) if st.through is not None and not cut else None) or self._after_chain(cut, st.moved_at, st)
        last = legs[-1].move
        if DROP_RELOOK and not cut and last.kind == "drop" and after.where.platform is not None \
                and after.where.platform.id == last.src:
            after = self._after_drop(st)
        # Cut short by a touch, she replans from where it knocked her, and while her invulnerability after it lasts no
        # fight starts on the way nor is `until` asked (POST_HIT): she moves on from there first. Stopped short for a
        # monster (one coming into the chain's way, or a hang under a rope's top, ROPE_HANG, after which she climbed on
        # up: the chain's moves past the climb not made), it's no miss.
        st.stopped = cut.startswith((COMING, HUNG, ASKED))
        if cut.startswith(TOUCHED) or cut.startswith(HUNG) and self._touch_since(s.t) is not None:  # (told late)
            self._touched(st)
        st.came, self._came = (self._came if st.stopped else None), None
        return [leg.move for leg in legs], legs[-1].end.platform, after

    def _carry_out_move(self, st: TripState, s: Scene, path: list[Move],
                        monsters=()) -> tuple[list[Move], Platform, Scene]:
        """The route's first move on its own, closed loop (control.mover). Returns it (as a list), the
        platform it goes to, and a look once it's done; none, where she is and `s`, if the mover can't do it
        from here (it costs CANNOT_S for the rest of the trip: st.cannot). With WATCH_MOVES its key plans are watched
        (Mover.watch: _move_watch) and cut on a touch, her HP dropping, or her standing on a platform the move doesn't
        go by: she replans from where she is, as after a chain cut short; its run-ups and back-offs are checked against
        `monsters` (the pass's: Mover.clear), and one into a monster refuses the move (Mover.MoveRefused), which then
        costs AVOID_SECONDS as a move a monster is in the way of (st.avoid), not CANNOT_S."""
        move = path[0]
        self.log(f"  {platform_text(s.where.platform)} at x {s.where.x:.0f} ({s.how}): {move.kind} "
                 f"{move.direction} -> platform {move.dst}  [{len(path)} moves to go]")
        if telemetry.recording():  # (as a chain's 'chain' record: she sets off; 'moved' once it's done)
            telemetry.emit("move", kind=move.kind, src=move.src, dst=move.dst, x=round(move.x, 1))
        t2 = time.monotonic()
        if WATCH_MOVES:
            self.mover.watch = self._move_watch(s, move)
            # (refused REFUSALS times in a row: unchecked, as a pass goes past)
            self.mover.clear = self._walk_clear(s, list(monsters)) if st.refused.get(_key(move), 0) < REFUSALS \
                else None
        try:
            after = self.mover.do(move, s)
        except MoveCut as cut:  # cut short as a chain is: she replans from where she is
            if str(cut).startswith(HUNG) and getattr(self, "_hang_at", None) is not None:  # (ROPE_HANG: the hang, up)
                text, self._hang_at = self._hang(*self._hang_at), None
                touch = self._hang_touch  # (a touch in it, or in the climb after, is the cut's: POST_HIT)
                cut = MoveCut(f"{HUNG}: {text}" if touch is None
                              else f"{_touch_text(touch, s.hp, s.hp)} ({HUNG}: {text})")
            st.trip.time["move"] += time.monotonic() - t2
            st.moved_at = time.monotonic()
            after = self.recover()
            self.log(f"    {cut}: the move cut short; on {platform_text(after.where.platform)} at x "
                     f"{after.where.x:.0f} after {time.monotonic() - t2:.2f} s: replanning from there")
            telemetry.emit("event", kind="move_cut", text=f"{move.kind} -> {move.dst}: {cut}")
            if str(cut).startswith(TOUCHED) or str(cut).startswith(HUNG) and self._touch_since(s.t) is not None:
                self._touched(st)
            return [move], self.model.platform(move.dst), after
        except MoveRefused as e:  # a back-off into a monster: another way, as for a monster in the way
            self.log(f"    refused: {e}; another way")
            st.avoid[_key(move)] = time.monotonic()
            st.refused[_key(move)] = st.refused.get(_key(move), 0) + 1
            if telemetry.recording():  # (nothing made: no 'moved' record ends the 'move' one)
                telemetry.emit("event", kind="cannot", text=f"{move.kind} -> {move.dst}: refused: {e}")
            return [], s.where.platform, s
        except ValueError as e:  # one it can't do from here (CANNOT_S): another way, from the next look
            self.log(f"    can't: {e}; another way")
            st.cannot.add(_key(move))
            if telemetry.recording():  # (nothing made: no 'moved' record ends the 'move' one)
                telemetry.emit("event", kind="cannot", text=f"{move.kind} -> {move.dst}: {e}")
            return [], s.where.platform, s
        finally:
            self.mover.watch = self.mover.clear = None
        st.trip.time["move"] += time.monotonic() - t2
        st.moved_at = time.monotonic()
        st.refused.pop(_key(move), None)
        return [move], self.model.platform(move.dst), after

    def _refused_step(self, st: TripState, s: Scene, path: list[Move], monsters: list, looked: float) -> None:
        """The mover refused the route's first move for a monster by its run-up or back-off (Mover.clear): the evade
        step's refuge (_evade_step), else a wait (WAIT_S); refused REFUSALS times in a row, it goes unchecked next
        (_carry_out_move), said, a backstop event."""
        move = path[0]
        if EVADE and self._evade_step(st, s, path, monsters, looked):
            return
        if st.refused.get(_key(move), 0) >= REFUSALS:
            text = f"REFUSALS: {move.kind} -> {move.dst} refused {st.refused[_key(move)]} times: unchecked"
            self.log(f"  {text}")
            telemetry.emit("event", kind="backstop", text=text)
            return
        st.trip.waits += 1
        self.log(f"  waiting for the monster by its run-up to move off ({st.refused[_key(move)]})")
        time.sleep(WAIT_S)

    def _move_watch(self, s: Scene, move: Move) -> Callable[[], str | None]:
        """What watches a move of the mover's from look `s` (WATCH_MOVES: Mover.watch, between its key plan's glances):
        a glance, and why to cut it short, or None: a touch told after the look (TOUCH_CUTS, a MISS too) or her HP
        lower than it had it (HIT), as a chain's; or her standing MOVE_OFF_S on a platform the move neither leaves nor
        goes to (a knock, a fall past it); or, a climb up a rope to the platform at its top (ROPE_HANG), the way up from
        her height on it not clear (_hang_due: `_hang_at` says where she hangs, the caller hangs her there)."""
        since, hp0, by = s.t, s.hp, {move.src, move.dst}
        stood = [None]
        self._hang_at = None
        rope = None
        if ROPE_HANG and move.kind == "climb":
            rope = next((r for r in self.model.ropes
                         if (r.x, r.top, r.bottom) == (move.rope_x, move.rope_top, move.rope_bottom)), None)
        grab = route.grab_height(self.model.platform(move.src).y_at(move.rope_x), rope) if rope is not None else None

        def watch() -> str | None:
            self.senses.wait_frame(GLANCE_S)
            dot, hp = self.senses.glance()
            touch = self._touch_since(since) if TOUCH_CUTS else None
            if touch is not None or hp < hp0 - HIT:
                return _touch_text(touch, hp0, hp)
            stood[0] = _stood(stood[0], self.senses.glanced_at, dot)
            if stood[0] is not None and self.senses.glanced_at - stood[0][0] >= MOVE_OFF_S:
                mx, my = self.senses.localizer.to_map(dot)
                on = self.model.platform_under(mx, my - 6, reach=16)
                if on is not None and on.id not in by:
                    return f"off the move (on {on.id}, planned {move.dst})"
            if rope is not None and (due := self._hang_due(rope, grab, dot, self.senses.glanced_at)) is not None:
                self._hang_at = due
                return HUNG
            return None
        return watch

    def _walk_clear(self, s: Scene, monsters: list) -> Callable[[float, float], str | None]:
        """What checks a walk of the mover's along her platform from map x0 to x1, set off now (Mover.clear: its
        run-ups and back-offs): why it isn't clear (a monster of `monsters`, the pass's, could touch her on it or as she
        stands there after: flow.danger, her invulnerability counted), or None."""
        p = s.where.platform

        def clear(x0: float, x1: float) -> str | None:
            if p is None or not monsters:
                return None
            t0 = max(0.0, time.monotonic() - self._tracks_at)
            t1 = t0 + abs(x1 - x0) / WALK_SPEED + SETTLE_S
            leg = Leg(Move("walk", p.id, p.id, x0, x1, "right" if x1 > x0 else "left"), Body(t0, x0, p),
                      Body(t1, x1, p), [Span(p.id, t0, t1, x0, x1)])
            hit = self.flow.danger(leg, monsters, linger=STAND_PAST_S)
            if hit is None:
                return None
            return f"a {hit[0].kind or 'monster'} on {hit[0].platform} by the walk from x {x0:.0f} to {x1:.0f}"
        return clear

    def first_clear(self, s: Scene, x0: float, goals: list[float]) -> float | None:
        """The first of `goals` a walk along her platform from map x0 to it, set off as soon as her lock lets her, is clear of
        look `s`'s monsters (`_walk_clear`: flow.danger, her invulnerability counted); None: none. The target-first planner's
        check of a walk (behavior.targeting), the look's tracks taken once however often it asks (a second update in one look
        has no time gone: no heading)."""
        if self._clear_of[0] is not s:
            self._clear_of = (s, self.tracker.update(s.t, s.monsters, seen_at=s.t - time.monotonic()))
            self._tracks_at = time.monotonic()
        monsters = self._clear_of[1]
        lock = max(getattr(self.actions, "lock_until", 0.0), getattr(self.actions, "arrow_until", 0.0))
        self._immunity(max(time.monotonic(), lock) + PLAN_START_S)
        clear = self._walk_clear(s, monsters)
        return next((g for g in goals if clear(x0, g) is None), None)

    def _arrival_clear(self, st: TripState, s: Scene) -> bool:
        """Without an escort (WATCH_MOVES), whether the walk to where she's sent on the trip's platform (look `s`) keeps
        clear of its monsters (the look's, tracked; her invulnerability counted): else she stays where she is, said, the
        trip arrived short (Trip.short). With ARRIVE_STAND, for a trip whose x is what it's for (TripState.hold_x), she
        stays only where standing there as long and STAND_PAST_S more is clear (_stand_clear), else walks on regardless;
        with TOUCHER_ALL the monster that touched her last is counted (_toucher). With an escort, the monster in her way
        is shot first (_clear_the_way): always; with ARRIVE_CHECK_POST_HIT, in the window after a touch, as without one."""
        if self.fight and self._post_hit(st) and _arrive_check_post_hit():
            kept, self._mode = self._mode, AVOID
            try:
                return self._arrival_clear(st, s)
            finally:
                self._mode = kept
        if self.fight or not WATCH_MOVES or st.x is None or abs(st.x - s.where.x) <= ARRIVE_TOLERANCE:
            return True
        monsters = self.tracker.update(s.t, self._on_platforms(s.monsters) if UNSEEN_AVOID else s.monsters,
                                       seen_at=s.t - time.monotonic())
        if TOUCHER_ALL and (toucher := self._toucher(s, monsters, standing=TOUCHER_STAND)) is not None:
            monsters = monsters + [toucher]
        self._tracks_at = time.monotonic()
        lock = max(getattr(self.actions, "lock_until", 0.0), getattr(self.actions, "arrow_until", 0.0))
        self._immunity(max(time.monotonic(), lock) + PLAN_START_S)
        why = self._walk_clear(s, monsters)(s.where.x, st.x)
        if why is None:
            return True
        if ARRIVE_STAND and st.hold_x                 and not self._stand_clear(s, monsters, abs(st.x - s.where.x) / WALK_SPEED + SETTLE_S):
            self.log(f"  on platform {st.dst} at x {s.where.x:.0f}: {why}, but standing here is no safer: on to x "
                     f"{st.x:.0f}")
            return True
        self.log(f"  on platform {st.dst} at x {s.where.x:.0f}, not walked on to x {st.x:.0f}: {why}")
        telemetry.emit("event", kind="evade", step="arrival", text=f"not walked on to x {st.x:.0f}: {why}")
        st.trip.short = f"on {st.dst} at x {s.where.x:.0f}, not walked on to x {st.x:.0f}: {why}"
        return False

    def _stand_clear(self, s: Scene, monsters: list, seconds: float) -> bool:
        """ARRIVE_STAND: whether standing where look `s` has her, from now (the tracks' plan time since _tracks_at) for
        `seconds` and STAND_PAST_S more (to her invulnerability's end and STAND_PAST_S at least) keeps clear of
        `monsters` (flow.danger)."""
        p, x = s.where.platform, s.where.x
        t0 = max(0.0, time.monotonic() - self._tracks_at)
        until = max(t0 + seconds, self.flow.immune) + STAND_PAST_S
        leg = Leg(Move("walk", p.id, p.id, x, x, ""), Body(t0, x, p), Body(until, x, p), [Span(p.id, t0, until, x, x)])
        return self.flow.danger(leg, monsters) is None

    def _onto_platform(self, s: Scene) -> Scene:
        """`s` if she's standing on a platform; else (in the air, or hanging on a rope) a look once she's
        landed or got off the rope, whose platform is None if that didn't work either. In the air over a
        platform her dot puts her on it: a chain planned from there as if she stood ran its first steps in
        the air. In the harness (loop, seed 30) a trip right after a jump throw ran for a drop from 33 onto
        35 mostly in mid-jump, dropped 30 units short, onto the basin by its cliff, and a pig's knock took
        her through the cliff and out of the map. A look still in the air once she's settled waits for her to
        land again (FALL_LOOKS)."""
        if s.where.platform is not None:
            if standing(s):
                return s
            for _ in range(FALL_LOOKS):  # in the air: once she's landed and stopped, a look with the monsters
                self.mover.settle(2)
                s = self.mover.locate(mobs=True, wide=not self.fight)
                if s.where.platform is not None:
                    return s
        s = self.mover.settle(2)
        if s.where.platform is None:  # hanging on a rope
            self.log(f"  on a rope at x {s.where.x:.0f}: getting off")
            s = self.mover.off_rope(s)
        return s

    def _rope_top(self, st: TripState, s: Scene, after: Scene, jumped: bool = False) -> Scene:
        """After a chain or a move from look `s` whose keys never moved her dot (`_pass`: the chain's glances for
        STUCK_S, or every look of the mover's), with `s` placing her at the top of a rope hanging from her platform
        (rope_at_top) and `after` her dot where it was: she's on that rope under its top, not on the platform (a jump
        or an arrow there does nothing; her dot snaps onto the platform from up to 33 under it, and from 17 under it by
        a rope since: localization.pose.ROPE_TOP_GAP), as at 22:19:38 on 2026-10-26, when 10 moves from rope 4's top
        "ended on platform 18 instead" over 15.4 s, her dot 14.8 under 18 at every chain start (Tree Dungeon; the
        3,039 chain starts' standing gaps: at most 14.0, none by a rope's top over 12). Her dot more than STUCK_GAP
        under the platform's surface only (standing by a rope's top, 9.2 at the most), unless the chain had her `jumped`
        (EVADE: a jump moves her dot two minimap px or more within 0.3 s, so one that never moved it never left the
        rope; the evade step's passes from a rope's top, "ended on 18 instead of platform 17" again and again where her
        dot was 12 or less under 18: the harness's walk, seed 34, 99 times). Off it (Mover.off_rope: up, onto the
        platform), and a look once she's there; else `after`."""
        rope = None if s.where is None else rope_at_top(self.model, s.where)
        if rope is None or after.where is None or after.where.dot is None or s.where.dot is None \
                or abs(after.where.dot[0] - s.where.dot[0]) > STILL or abs(after.where.dot[1] - s.where.dot[1]) > STILL:
            return after
        x, y = self.senses.localizer.to_map(after.where.dot)
        p = s.where.platform
        if y - p.y_at(p.clamp(x)) <= STUCK_GAP and not (jumped and (EVADE or ROPE_CUT and self.cut_still)):
            return after
        text = (f"her dot never moved: on the rope at x {rope.x:.0f}, {y - rope.top:.0f} under its top, not on "
                f"platform {s.where.platform.id}")
        self.log(f"    {text}: getting off")
        telemetry.emit("event", kind="rope_top", text=text)
        st.fresh = None  # (the look after the chain, for the next pass: she's moved since)
        return self.mover.off_rope(after, rope, at=(x, y))

    def _touch_since(self, since: float | None) -> dict | None:
        """The newest touch of hers told in a frame shown after `since` (Senses.touched_since: MISS or damage), or None
        (none, no `since`, or a stand-in for Senses without it)."""
        touched = getattr(self.senses, "touched_since", None)
        return None if since is None or touched is None else touched(since)

    def _touched(self, st: TripState) -> None:
        """A chain or a move of trip `st` was cut short by a touch (or her HP dropping): no fight on the way nor `until`
        till her invulnerability after it is over (POST_HIT: Senses.invulnerable_until, less IMMUNE_MARGIN_S;
        INVULNERABLE_S from now where no popup was told)."""
        if not POST_HIT:
            return
        until = getattr(self.senses, "invulnerable_until", None)
        end = until() if until is not None else -1e18
        if end <= time.monotonic():  # (her HP dropped with no popup told: from now)
            end = time.monotonic() + INVULNERABLE_S
        st.post_hit_until = max(st.post_hit_until, end - IMMUNE_MARGIN_S)

    @staticmethod
    def _post_hit(st: TripState) -> bool:
        """Whether trip `st` is in the window after a touch where no fight starts and `until` isn't asked (POST_HIT)."""
        return time.monotonic() < st.post_hit_until

    def _immunity(self, start: float) -> float:
        """Her invulnerability after her last touch (Senses.invulnerable_until, less IMMUNE_MARGIN_S) as plan time of a
        chain that starts at `start` (time.monotonic): set as flow.immune (no touch counts on its legs before then) and
        returned; 0 once it's over by then, or without IMMUNE."""
        until = getattr(self.senses, "invulnerable_until", None)
        self.flow.immune = max(0.0, until() - IMMUNE_MARGIN_S - start) if IMMUNE and until is not None else 0.0
        return self.flow.immune

    def _immunity_spent(self, legs: list[Leg]) -> bool:
        """IMMUNE_RECHECK: her invulnerability's window taken again for a chain about to go (from now, or her lock's end,
        and PLAN_START_S); where planning spent more of it than IMMUNE_MARGIN_S covers and a leg meets one of the monsters
        the chain keeps clear of (the last flow.chain's, `_kept`) before the window it was planned in ended (flow.danger
        at what's left), True: said, a telemetry event 'immune', and the chain isn't sent. flow.immune is what's left."""
        planned = self.flow.immune
        if not (IMMUNE and IMMUNE_RECHECK) or planned <= 0:
            return False
        lock = max(getattr(self.actions, "lock_until", 0.0), getattr(self.actions, "arrow_until", 0.0))
        left = self._immunity(max(time.monotonic(), lock) + PLAN_START_S)
        if left >= planned - 1e-9:
            return False
        kept = getattr(self, "_kept", [])
        for i, leg in enumerate(legs):
            met = self.flow.danger(leg, kept) if kept else None
            if met is not None and met[1] < planned:
                text = (f"her invulnerability spent planning ({planned - left:.2f} s of {planned:.2f}): leg {i + 1}'s "
                        f"{leg.move.kind} meets the {met[0].kind or 'monster'} {met[1]:.2f} s in; planned again")
                self.log(f"  {text}")
                telemetry.emit("event", kind="immune", text=text, planned=round(planned, 3), left=round(left, 3))
                return True
        return False

    def _safe_to_stand(self, x: float, here: Platform, move: Move, monsters) -> bool:
        """Whether no monster can reach her standing at x for the next second (as a leg that stays put), and with EVADE
        to her invulnerability's end and STAND_PAST_S at least (R3: with a second of it left, a second's stand was no
        check, and she "waited" in contact)."""
        until = self._stand_until(0.0, 1.0) if EVADE else 1.0
        return not self.flow.danger(
            Leg(move, Body(0.0, x, here), Body(until, x, here), [Span(here.id, 0.0, until, x, x)]), monsters)

    def _after_through(self, s: Scene) -> Scene | None:
        """The trip `through` her drops over, not cut short: her dot's last glance for where it left her, as a look's data (the hunt's
        next look says the rest, one frame on: a look here would make that two); None with no glance to go by."""
        dot, hp = self.senses.glance()
        where = None if dot is None else self.senses.localizer.where_dot(dot)
        return None if where is None or where.platform is None else replace(s, where=where, hp=hp)

    def _after_chain(self, cut: str, moved_at: float, st: TripState) -> Scene:
        """A look once a chain's plan is over: once she's stopped moving if it was cut short (`cut`),
        else once she's on a platform. Not cut short, it's the look the next pass needs (with the monsters):
        two looks in a row there were half the stop between chains."""
        landed, self._landed = getattr(self, "_landed", None), None
        if cut and landed is not None:  # (KNOCK_REPLAN: the first state with her on the ground again, the next pass's)
            touch = getattr(self.senses, "touch", None)
            since = f" {landed.t - touch['at']:.2f} s after the touch" if isinstance(touch, dict) else ""
            self.log(f"    {cut}: plan cut short; on {platform_text(landed.where.platform)} at x {landed.where.x:.0f}, "
                     f"the knock over{since}: replanning from there at once")
            st.fresh = landed
            return landed
        if cut:
            after = self.recover()
            self.log(f"    {cut}: plan cut short; on {platform_text(after.where.platform)} at x "
                     f"{after.where.x:.0f} after {time.monotonic() - moved_at:.2f} s: replanning from there")
            return after
        # Once she's stopped sliding (AFTER_STILL_S at most): planned from a look at the plan's end, she slid on
        # up to 27 units past where the next chain started her and its jumps missed; from a look a set 0.16 s
        # in (when the next pass's own look came before), misses rose by a quarter in the simulator.
        after = self.recover(longest=AFTER_STILL_S, mobs=True, wide=not self.fight)
        if after.where.platform is None:
            return self.mover.settle(1.5)
        st.fresh = after
        return after

    def _after_drop(self, st: TripState) -> Scene:
        """A look again DROP_RELOOK_S after a chain's drop, once the look after it had her on the platform she dropped
        through (from a frame of before the fall: DROP_RELOOK): once she's stopped moving, of a frame after the first."""
        time.sleep(DROP_RELOOK_S)
        after = self.recover(longest=AFTER_STILL_S, mobs=True, wide=not self.fight)
        if after.where.platform is None:
            return self.mover.settle(1.5)
        self.log(f"    looked again after the drop's fall: on {platform_text(after.where.platform)}")
        st.fresh = after
        return after

    def _note_landing(self, trip: Trip, moved: list[Move], planned: Platform, after: Scene, hp: float,
                      stopped: bool = False, since: float | None = None) -> bool:
        """Record the moves made and where she landed in trip.moves (and a touch in trip.hits: one told in a frame after
        `since`, the look they were planned from, MISS or damage; or her HP lower than `hp` with none told), and log
        it if that's not `planned`. True if it's a miss: elsewhere than planned, and not because a
        monster touched her or (`stopped`) she stopped short for one coming into the chain's way (the caller counts
        misses)."""
        landed = after.where.platform
        touch = self._touch_since(since)
        hit = touch is not None or after.hp < hp - HIT
        moved = [m for m in moved if m.kind not in ("step", "walk")]  # (an evade step's step along her platform, a loot chain's walk: no route move)
        if telemetry.recording():  # (for the coverage report: analysis.coverage)
            telemetry.emit("moved", moves=[{"kind": m.kind, "src": m.src, "dst": m.dst, "x": round(m.x, 1)}
                                           for m in moved],
                           planned=planned.id, landed=landed.id if landed else None, hit=hit,
                           **({"touch": touch["kind"]} if touch is not None else {}))
        if moved:
            trip.moves += [f"{m.kind} {m.src}->{m.dst}" for m in moved]
            trip.moves[-1] += f": landed on {landed.id if landed else None}" + (" (hit)" if hit else "")
        if hit:
            trip.hits += 1
        if landed is not None and landed.id == planned.id:
            return False
        if hit:
            self.log(f"    {_touch_text(touch, hp, after.hp)} on the way: on {platform_text(landed)}; replanning")
            return False
        if stopped:  # (where the cut left her: said by _after_chain)
            return False
        self.log(f"    ended on {platform_text(landed)} instead of platform {planned.id}; replanning")
        return True

    def follow(self, legs, keys, hp: float, dst: int, why: str, trip: Trip, st: TripState | None = None,
               path: list[Move] | None = None, monsters=(), since: float | None = None,
               replan: bool = False) -> tuple[str, list]:
        """Run a chain's plan and keep going like a player, with no stop before the next chain: late in
        it, a look around decides whether one may follow (_look_ahead); once she's back on the ground for
        good, a look places her and the next chain is planned from there (_queue_next), queued so the
        player starts it the moment this one finishes. Meanwhile her dot and HP are watched ~15 times a
        second: a touch told after `since` (the frame time of the look it was planned from: TOUCH_CUTS, MISS or
        damage) or her HP lower than `hp` (HIT), or her being on another platform than planned for a few glances,
        cuts the running plan and the queued one short (the player lets go of every key). Returns why it was cut
        ('' if not) and all the legs sent. (`path`: the route the chain is from; `st`: its trip; `monsters`: the
        ones the chain keeps clear of; `replan`: replanned as it runs, REPLAN.)"""

        def send(chain_legs, chain_keys, why: str, tail: float = 0.0) -> tuple[str, float]:
            """Hand a chain's plan to the player: its id, and how long it lasts (s)."""
            trip.chains += 1
            if telemetry.recording():
                telemetry.emit("chain", **chain_record(chain_legs, why))
            # The plan lasts until she's predicted to land (or the next chain starts), so the look
            # after it doesn't catch her still falling; and its tail past that, if another may follow.
            events = chain_keys.sorted()
            end_ms = max(max(e[0] for e in events), ms(chain_legs[-1].end.t + tail))
            self.actions.sent_at = time.monotonic()
            plan_id = client.send(events, lead_ms=0, focus=True, name="flow", end_ms=end_ms,
                                  kind=getattr(self.actions, "kind", None))
            self._tails[plan_id] = (end_ms - max(e[0] for e in events)) / 1000  # (its run on past its last key)
            return plan_id, end_ms / 1000

        tail = self._tail(legs, path, monsters) if why == "long" and st is not None else None
        more, tail = tail is not None, tail or 0.0
        if why == "long" and not more:
            _no_follow("no room to look where it lands")
        if why == "long" and not more:
            keys.set_arrow(legs[-1].end.t, 0)  # nothing follows: she stops where it ends (held on, she ran past it)
        # Right after a throw, a jump pressed inside the attack lock is ignored, and an arrow pressed within about
        # 0.62 s of its key (from 0.63 to 0.78 s it isn't measured) moves her only from about 0.79 s after it: the
        # chain waits just enough for both (Actions.hold_back: its first arrow 0.8 s after the key, ARROW_LOCK_MS).
        # (A chain queued on the move starts when this one ends, past any lock.)
        self.actions.hold_back(keys.sorted())
        plan_id, length = send(legs, keys, why, tail)
        current = _Running(plan_id, legs, time.monotonic() + PLAN_START_S, more, tail, length, tuple(monsters), why=why,
                           arrows=self._arrows(keys.sorted()))
        self._sent_plans = [current.plan_id]  # every plan this chain sent: cancelled if anything fails
        self._replanning = replan and st is not None
        self.actions.pressed = True
        try:
            return self._watch(current, hp, dst, send, st, since)
        except BaseException:
            # Whatever went wrong here (Ctrl+C included), the plans sent must not play on unwatched.
            for pid in self._sent_plans:
                try:
                    client.cancel(pid)
                except Exception:
                    pass
            self._ended(None, cut=True)
            raise

    def _watch(self, current: _Running, hp: float, dst: int, send, st: TripState | None,
               since: float | None = None) -> tuple[str, list]:
        """follow's loop: watch the running chain, queue the next, cut both short when she's touched (after `since`)
        or hit, or off her path."""
        queued, sent, cut = None, list(current.legs), ""
        stood = None  # since when, and where, her dot has stood (_stood)
        off_since, off_leg = None, False  # the first glance in a row off the planned path, and whether one of them
                                          # had her where the leg doesn't go
        ahead = None  # the look around for what follows the running chain, once taken
        over = False  # the running chain's plan has ended (its result read)
        self.chain_still = None  # her dot where the first glance had it all along, and for how long (_rope_top)
        self.cut_still = False  # ... and whether the chain was cut short for her dot standing where it runs her (ROPE_CUT)
        first, moved, first_at, last_at = None, False, 0.0, 0.0
        dot = None
        glanced: list[tuple[float, tuple]] = []  # every glance's (frame time, dot), for _keys_seen
        last = getattr(st, "last", None)  # (the pass's look: a chain from a rope's top tells the watch nothing)
        top = last is not None and last.where is not None and rope_at_top(self.model, last.where) is not None
        checked, withdrawn = None, []  # the queued chain checked against the feed's newest tracks (QUEUE_CHECK)
        try:
            return self._watching(current, hp, dst, send, st, since, queued, sent, cut, stood, off_since, off_leg,
                                  ahead, over, first, moved, first_at, last_at, dot, glanced, top, checked, withdrawn)
        finally:
            for pid in withdrawn:  # (taken by the player before it was withdrawn: it lets go at once, its result read)
                client.wait(pid)

    def _watching(self, current, hp, dst, send, st, since, queued, sent, cut, stood, off_since, off_leg, ahead, over,
                  first, moved, first_at, last_at, dot, glanced, top, checked, withdrawn) -> tuple[str, list]:
        """_watch's loop, its state handed in (the queued chains it withdrew, `withdrawn`, waited for by _watch)."""
        replan = getattr(self, "_replanning", False)
        better_at = time.monotonic() + BETTER_EVERY_S  # (the next look for a better way: REPLAN)
        touch = None
        while True:
            # The look around goes late in the chain: from where she starts, where she's going can be
            # off screen (from platform 34 the basin's slimes were under the status bar).
            lands = current.start + current.legs[-1].end.t
            if current.more and queued is None:
                if ahead is None and lands - time.monotonic() <= LOOK_AHEAD:
                    ahead = self._look_ahead(st, current, dst)
                    if ahead is None:
                        current = current._replace(more=False)  # she stops where it ends
                elif ahead is not None and time.monotonic() >= self._check_at(current):
                    queued, ahead = self._queue_next(st, current, ahead, dst, send), None
                    if queued is None:
                        current = current._replace(more=False)
                    else:
                        sent += queued.legs
                        self._sent_plans.append(queued.plan_id)
            if over:
                if not top:
                    self._keys_seen(current, glanced)
                if queued is None:
                    break
                current, queued, over = queued, None, False
                top = False
                continue
            if queued is None and not current.more:
                # Nothing follows: she stops where its last leg ends (the rest of a tail kept for the look placing
                # her isn't needed: run on, it put her 34 units past where she was to stop, and the next chain
                # from there dropped her into the basin), and that's noticed at once (the next look and plan wait).
                last = current.start + current.legs[-1].end.t
                if current.tail and time.monotonic() >= last:
                    if not client.cancel(current.plan_id):
                        client.wait(current.plan_id)
                    self._ended(current.plan_id, cut=True)
                    break
                if (last if current.tail else current.ends) - time.monotonic() <= END_WATCH:
                    if client.done(current.plan_id):
                        client.wait(current.plan_id)
                        self._ended(current.plan_id)
                        break
                    time.sleep(END_POLL_S)
                    continue
            if current.more and queued is None and ahead is not None:
                due = self._check_at(current) - time.monotonic()
                if due < 2 * GLANCE_S:  # the look placing her is due before a glance would be over: wait for it
                    time.sleep(max(0.0, due))
                    continue
            self.senses.wait_frame(GLANCE_S)
            dot, now_hp = self.senses.glance()
            seen_at = self.senses.glanced_at
            if current.parent is not None and (taken := client.taken(current.plan_id)) is not None:
                if not taken:  # (the plan it was to take over from was over first: she stopped where that ends)
                    self.log("    the replanned chain wasn't taken in time (its plan was over): she stops there")
                    telemetry.emit("event", kind="replan", text="not taken: the plan it was for was over first")
                    self._ended(current.plan_id, cut=True)
                    self._cut_plan = (current.legs, 0.0)  # (none of it ran)
                    cut = "off the path (the replanned chain wasn't taken in time)"
                    break
                current = current._replace(parent=None)
            if QUEUE_CHECK and queued is not None and checked is not queued \
                    and queued.start - time.monotonic() <= 2 * GLANCE_S:
                checked = queued
                if (why_q := self._queued_meets(queued)) is not None:
                    self.log(f"    {why_q}: the chain queued after this one withdrawn")
                    telemetry.emit("event", kind="evade", step="queued", text=why_q)
                    if not client.cancel(queued.plan_id):
                        withdrawn.append(queued.plan_id)
                    sent = sent[:len(sent) - len(queued.legs)]
                    queued, current = None, current._replace(more=False)
                    continue
            if dot is not None:
                glanced.append((seen_at, dot))
            stood = _stood(stood, seen_at, dot)
            if dot is not None:
                if first is None:
                    first, first_at = dot, seen_at
                elif abs(dot[0] - first[0]) > STILL or abs(dot[1] - first[1]) > STILL:
                    moved = True
                last_at = seen_at
            if dot is not None and telemetry.recording():
                gx, gy = self.senses.localizer.to_map(dot)
                telemetry.emit("glance", x=round(gx, 1), y=round(gy, 1), hp=round(now_hp, 3))
            touch = self._touch_since(since) if TOUCH_CUTS else None
            if touch is not None or now_hp < hp - HIT:
                cut = _touch_text(touch, hp, now_hp)
            elif dot is not None:
                on, want, leg, behind = self._placed(dot, seen_at - stood[0] >= STILL_S, current, seen_at)
                if on is not None and on.id not in (leg.start.platform.id, leg.end.platform.id):
                    # Standing where this leg doesn't go (e.g. a drop that fell one platform too far):
                    # stop sooner, before the plan's next keys fire there.
                    off_since, off_leg = seen_at if off_since is None else off_since, True
                elif on is not None and (want is not None and on.id != want or behind is not None):
                    off_since = seen_at if off_since is None else off_since
                else:
                    off_since, off_leg = None, False
                if off_since is not None and seen_at - off_since >= (OFF_LEG_S if off_leg else OFF_PATH_S):
                    where = f"still at x {behind:.0f}" if behind is not None else f"on {on.id if on else 'nothing'}"
                    cut = f"off the path ({where}, planned {want or leg.end.platform.id})"
            if not cut and self.landing_asks and self._landing_calls(st, current, queued):
                cut = ASKED
            hang = None
            if not cut and ROPE_HANG and (hang := self._hang_check(current, dot, seen_at)) is not None:
                cut = HUNG  # (said once the hang is over, below)
            switched = None
            if not cut and REACT and (coming := self._coming(current, room=not replan)) is not None:
                mo, t, tid = coming
                why = f"{COMING} ({mo.kind} on {mo.platform}, {t - (time.monotonic() - current.start):.2f} s ahead)"
                self._replanned, self._late = False, None
                switched = self._switch(st, current, dst, why, "monster") if replan else None
                if switched is None and (not replan or self._room_to_stop(current, time.monotonic() - current.start)):
                    cut = why
                if self._late is not None:  # (replanned too late: she stops for it where there's room, else asked)
                    self._dropped("monster", "she stops for it" if cut else self._late_again(tid))
                if self._late is None or cut:  # (taken, or cut for: its late replans no longer in a row)
                    self._late_in_row.pop(tid, None)
                if switched is not None or cut or self._replanned:
                    # (acted on, or replanned with nothing to take: not again for REACT_AGAIN_S; else, in a flight or
                    # past a takeoff's key, not where the plan has her, or its replan dropped as too late with no room
                    # to stop, the next state asks again, as _switch says: marked first, it went unheeded for 3 s, 1-3
                    # times a 5-minute run, the review of 2026-10-30; a replan dropped late was marked so until the
                    # latency analysis of 2026-11-01, and she ran on into it)
                    self._reacted[tid] = time.monotonic()
            elif not cut and replan and time.monotonic() >= better_at:
                better_at = time.monotonic() + BETTER_EVERY_S
                self._late = None
                switched = self._switch(st, current, dst, "a better way", "better")
                if self._late is not None:  # (the next look is BETTER_EVERY_S on, as after any other)
                    self._dropped("better", f"the next look for a better way asks again ({BETTER_EVERY_S:.2f} s on)")
            if switched is not None:  # (handed over: the running plan's rest is the new chain's)
                if not top:
                    self._keys_seen(current, glanced, until=switched.start)
                if queued is not None:  # (planned to follow the running plan's end: withdrawn)
                    if not client.cancel(queued.plan_id):
                        withdrawn.append(queued.plan_id)
                    sent = sent[:len(sent) - len(queued.legs)]
                    queued = None
                at = (switched.start if switched.parent is not None else time.monotonic()) - current.start
                kept = [leg for leg in current.legs if leg.end.t <= at + 1e-6]  # (made: the leg she's on is the new one's)
                sent = sent[:len(sent) - len(current.legs)] + kept + list(switched.legs)
                self._sent_plans.append(switched.plan_id)
                current, ahead, checked, over, top = switched, None, None, False, False
                stood, off_since, off_leg, glanced = None, None, False, []
                first, moved = None, False
                continue
            if cut:
                self._cut_plan = (current.legs, seen_at - current.start)
                if not top:
                    still = cut.startswith("off the path (still")  # (her dot standing where the plan runs her)
                    self.cut_still = still
                    self._keys_seen(current, glanced, until=seen_at, runs=not still)
                    if still and self.moves is not None:
                        self.moves.ran(seen_at, False)  # (the chain's run, once)
                climbing = self._climbing(current, dot) if cut.startswith(TOUCHED) and CLIMB_ON else None
                for pid in [current.plan_id] + ([queued.plan_id] if queued else []):
                    if not client.cancel(pid):
                        client.wait(pid)
                self._ended(current.plan_id, cut=True)
                if queued:
                    sent = sent[:len(sent) - len(queued.legs)]
                if hang is not None:  # (ROPE_HANG: a touch in it, or in the climb after, is the cut's: POST_HIT)
                    text = self._hang(*hang)
                    touch = self._hang_touch
                    cut = f"{HUNG}: {text}" if touch is None else f"{_touch_text(touch, hp, now_hp)} ({HUNG}: {text})"
                if climbing is not None:
                    self._climb_on(*climbing, touch=touch)
                elif cut.startswith(TOUCHED) and REPLAN and KNOCK_REPLAN:  # (the next pass plans from where it lands her)
                    self._landed = self._knock_landing(touch)
                break
            if client.done(current.plan_id):
                client.wait(current.plan_id)
                self._ended(current.plan_id)
                over = True  # the queued chain takes over, or the next one is still to come (see above)
        if first is not None and not moved:
            self.chain_still = (first, last_at - first_at)
        if not cut and not top and not over:  # (the last chain's end: over, its glances seen through; cut: told above)
            self._keys_seen(current, glanced)
        return cut, sent

    def _landing_calls(self, st: TripState | None, running: _Running, queued: _Running | None) -> bool:
        """Whether she has just landed on a platform the running chain goes on from, and what a pass asks first (the trip's
        `until`) calls for her there, from a real look: nothing else asks there (`_watching`'s checks are all of her route), and
        72% of the landings she passed live were inside a chain, at a third of those her own chooser shooting at once
        if asked (README: "Where she hunts"). Once a landing and again for a look with more monsters than it was asked with (the first look
        hasn't all of them: of 82 landings she passed live with 2 or more in reach, 39 had them only at a later look); not where the chain set
        off from, nor its end (the pass asks), nor while her invulnerability after a touch lasts (POST_HIT)."""
        state = getattr(self.senses, "glanced", None)
        if st is None or st.until is None or state is None or state.where is None or state.mode != "ground":
            self._asked_on = None
            return False
        on = state.where[2]
        if on is None or (on == self._asked_on and len(state.monsters) <= self._asked_n):
            return False
        self._asked_on, self._asked_n = on, len(state.monsters)
        if on == running.legs[0].start.platform.id or self._post_hit(st) \
                or (on == running.legs[-1].end.platform.id and not running.more and queued is None):
            return False
        s = to_scene(state, self.model, self.senses.band, mobs=True, wide=True)
        s.landing = True
        return s.where is not None and s.where.platform is not None and s.searched and bool(st.until(s))

    # --- replanning while a chain runs (REPLAN) ------------------------------------------------------

    def _switch(self, st: TripState | None, running: _Running, dst: int, why: str, kind: str) -> _Running | None:
        """REPLAN: the running chain replanned from where she'll be when new keys can act (_splice_point: SPLICE_LEAD_S
        on), against the newest state's monsters, all of them (the glance's: `_state_scene`, its whole screen, each
        walking as perception has it head at its kind's speed from the game data, her memory's and the feed's out of
        sight), its route the trip's (_route), its landings priced as the pass prices them, no wider takeoffs (flow.chain's
        `spread`: within a state's budget), and handed over in place of the rest of the running plan (`_take_over`).
        Only from the ground, where the state has her on the platform the plan has her on then and within SPLICE_DRIFT
        of where it has her, her start put where she is (the plan's place shifted by what the state measured): in a
        flight, or past a takeoff's key (`_splice_point` past now's), not yet: the next state asks again once she's down
        (planned from a predicted landing, a landing 12.5 units off made the next running jump leave 12.5 units early,
        and one late put the next chain's keys in the air: _queue_next's). `kind` 'monster': a monster coming into the
        running chain's way (REACT): taken if anything chains. 'better': taken where the running chain stops short
        (for a monster, a move it can't predict, or cut off for length with nothing to follow on) and the new one goes
        the same way on past where it stops; or the new one's route isn't the running one's and costs SWITCH_GAIN_S
        less by the route planner's own costs (Planner.quickest against the running way, Planner.through its next
        move: the same estimates both); SWITCH_MIN_S after the last switch at the soonest either way. None: nothing taken
        (no state, she's in the air or not where the plan has her, no route, nothing chains, or not worth it; or done
        too late, SPLICE_LATE_S: `_late` says so, and the caller says what comes of it, _dropped). `_replanned` once it
        replans, for 'monster': a monster replanned for with nothing to take is handled (_watching marks it), one whose
        replan was dropped as too late isn't."""
        state = getattr(self.senses, "glanced", None)
        if st is None or state is None or not getattr(state, "placed", False):
            return None
        now = time.monotonic()
        if kind == "better" and now - st.switched_at < SWITCH_MIN_S:
            return None
        soonest = now + SPLICE_LEAD_S - running.start
        point = self._splice_point(running, soonest)
        if point is None or point[0] > soonest + 1e-6:
            return None  # (in a flight or past a takeoff's key: asked again once she's down)
        at, body = point
        if at > running.legs[-1].end.t + 1e-6 or (body.platform.id == dst and st.through is None):
            return None  # (past its end, or she gets there with it: the next pass goes on from there)
        mx, _, on, _ = state.where
        seen = self._splice_point(running, state.t - running.start)
        if on != body.platform.id or seen is None or seen[1].platform is not body.platform                 or abs(mx - seen[1].x) > SPLICE_DRIFT:
            return None  # (not where the plan has her: her off-path checks deal with it)
        p = body.platform
        body = replace(body, x=p.clamp(body.x + (mx - seen[1].x)))
        s = self._state_scene(state, body)
        if s is None:
            return None
        self._replanned = True  # (from here on it's replanned: a monster it was for isn't asked about again at once,
                                # unless the replan is dropped as too late, below)
        t0 = time.perf_counter()
        start = running.start + at  # when the new plan's keys take over (its plan time 0)
        kept = [leg.move for leg in running.legs if leg.end.t <= at + 1e-6]
        self._gone_by(st, kept)  # (the moves she has set off on: the way the trip chose, Choice)
        path, _ = self._route(st, p.id, body.x, dst)
        if not path:
            return None
        rest = [leg.move for leg in running.legs if leg.end.t > at + 1e-6 and leg.move.kind != "hop"]
        same = bool(rest) and _same_moves(path, rest)
        if kind == "better":
            stops = running.why in ("monster", "unpredictable") or running.why == "long" and not running.more
            if same and not (stops and len(path) > len(rest)):
                return None  # (the same way, and the running chain doesn't stop short of where this one goes)
            if same and self._stop_calls(st, state, running):
                return None  # (where it stops, the trip's pass has her fight or end there: no running on past it)
            if not same and st.through is not None:
                return None  # (her drops' order is the sweep's: no other way to weigh)
            if not same:
                penalty = self._penalty(st)
                old = self.planner.through(p.id, body.x, dst, rest[0], penalty, self.weight) if rest else None
                new = self.planner.quickest(p.id, body.x, dst, penalty, self.weight)
                if old is None or new is None or not new.cost + SWITCH_GAIN_S < old.cost:
                    return None
                why = f"{why} ({old.cost - new.cost:.2f} s sooner by route)"
        monsters, near = self._chain_monsters(st, s, path, start=start)
        self._immunity(start)
        price, alts, goal = self._pricing(st, monsters)
        spliced = client.can_splice()
        # (spliced, the arrow her start holds stays held: the chain's keys let it go where it doesn't run on; until its
        # review a chain that turned or stopped at once pressed the other arrow with this one held on, and ran her off
        # 11's end: flow.chain's `held`. Cut and sent, every key is let go and the chain presses its own.)
        legs, keys, why_c = self.flow.chain(path, body, monsters, keep_going=True, hops=not self.fight, price=price,
                                            alts=alts, goal=goal, spread=False, held=body.run if spliced else 0)
        if not legs or (st.through is not None and not keys.sorted()):
            return None
        if kind == "better" and same:
            done = sum(1 for leg in legs if leg.move.kind != "hop")
            if done <= len(rest) and why_c == running.why:
                return None  # (it stops where the running one does: nothing gained)
            why = f"{why}: on past where it stopped ({running.why})"
        think_ms = (time.perf_counter() - t0) * 1000
        late = time.monotonic() + SPLICE_LEAD_S - start  # (s past its plan time 0 the player would take it)
        if late > 0.0:
            gone = self._splice_point(running, at + late)
            if late > SPLICE_LATE_S or gone is None or gone[0] > at + late + 1e-6:
                # (nothing taken, and for a monster nothing handled: _watching cuts where there's room, else the next
                # state asks again; until the latency analysis of 2026-11-01 its monster stayed marked replanned for
                # REACT_AGAIN_S, while the log said the next state asks again)
                self._replanned, self._late = False, (why, think_ms, late)
                return None
        return self._take_over(st, running, at, path, legs, keys, why_c, why, kind, near if near is not None else monsters,
                               think_ms, spliced)

    def _late_again(self, tid: int) -> str:
        """What comes of the feed's track `tid`'s replan dropped as too late with no room to stop: the next state asks
        about it again, LATE_RETRIES times in a row; then it's marked for LATE_AGAIN_S (REACT_AGAIN_S's mark set back),
        asked about again after that."""
        n = self._late_in_row.get(tid, 0) + 1
        if n <= LATE_RETRIES:
            self._late_in_row[tid] = n
            return "the next state asks again"
        self._late_in_row.pop(tid, None)
        self._reacted[tid] = time.monotonic() - (REACT_AGAIN_S - LATE_AGAIN_S)
        return f"asked about again {LATE_AGAIN_S:.1f} s on (dropped late {n} times in a row)"

    def _dropped(self, kind: str, then: str) -> None:
        """The last _switch's replan of `kind`, dropped as too late (`_late`: SPLICE_LATE_S), said with what comes of it
        (`then`: her stop for the monster, or when it's asked again): the log and a 'replan' event."""
        why, think_ms, late = self._late
        self.log(f"    {why}: replanned too late ({think_ms:.0f} ms, {late * 1000:.0f} ms past its start): "
                 f"dropped, {then}")
        if telemetry.recording():
            telemetry.emit("event", kind="replan", text="dropped: too late", of=kind, then=then,
                           think_ms=round(think_ms, 2), late_ms=round(late * 1000, 1))

    def _take_over(self, st: TripState, running: _Running, at: float, path: list[Move], legs: list[Leg], keys,
                   why_c: str, why: str, kind: str, known, think_ms: float, spliced: bool | None = None) -> _Running:
        """The new chain handed over in place of the running plan's rest at its plan time `at`: spliced (client.splice:
        the arrow her new start holds kept held, no key let go between) where the player takes splices
        (client.can_splice), else the running plan cut and the new one sent at once, its keys from `at` on (a player
        started before splices: every key let go for the cut, 15-35 ms). Said (the log, a 'chain' and a 'replan' record),
        and the running chain the new one's (_Running: `parent` the plan it takes over from, until the player has taken
        it: client.taken)."""
        trip = st.trip
        # A chain cut off for length runs on into the next, planned on the move (follow's tail: _look_ahead,
        # _queue_next), where there's room; else she stops where it ends, as follow has it.
        tail = self._tail(legs, path, known) if why_c == "long" else None
        if why_c == "long" and tail is None:
            keys.set_arrow(legs[-1].end.t, 0)
        events = keys.sorted()
        last_key = max(e[0] for e in events)
        end_ms = max(last_key, ms(legs[-1].end.t + (tail or 0.0)))
        body = legs[0].start
        held = [self.flow.key_names[directions.toward(body.run)]] if body.run else []
        plan_kind = getattr(self.actions, "kind", None)
        spliced = client.can_splice() if spliced is None else spliced  # (as the chain was planned for: `held`)
        if spliced:
            plan_id = client.splice(running.plan_id, events, ms(at), held, end_ms=end_ms, name="flow", kind=plan_kind)
            start, parent = running.start + at, running.plan_id
        else:
            if not client.cancel(running.plan_id):
                client.wait(running.plan_id)
            self._ended(running.plan_id, cut=True)
            begins = time.monotonic() + PLAN_START_S
            shift = ms(max(0.0, running.start + at - begins))  # (a flight's landing: its keys from then on)
            events = [[e[0] + shift, *e[1:]] for e in events]
            plan_id = client.send(events, lead_ms=0, focus=True, name="flow", end_ms=end_ms + shift, kind=plan_kind)
            start, parent = begins + shift / 1000, None
        self.actions.sent_at = time.monotonic()
        self._tails[plan_id] = (end_ms - last_key) / 1000  # (its run on past its last key: _ended)
        trip.chains += 1
        trip.replans += 1
        st.switched_at = time.monotonic()
        st.path = path
        if telemetry.recording():
            telemetry.emit("chain", **chain_record(legs, why_c))
            telemetry.emit("replan", why=why, kind=kind, at=round(at, 3), spliced=spliced, legs=len(legs),
                           think_ms=round(think_ms, 2), first=f"{legs[0].move.kind} {legs[0].move.src}->{legs[0].move.dst}")
        self.log(f"    {why}: replanned from {at:.2f} s into the plan ({'spliced' if spliced else 'cut and sent'}), "
                 f"on {body.platform.id} at x {body.x:.0f}: " + _chain_text(legs, len(path), why_c))
        return _Running(plan_id, legs, start, tail is not None, tail or 0.0, end_ms / 1000, tuple(known), parent=parent,
                        why=why_c, arrows=self._arrows(events, body.run if spliced else 0))

    def _arrows(self, events, held: int = 0) -> tuple:
        """A plan's arrow keys' events ((ms, -1 left / +1 right, down)), the arrow held at its start (`held`: a splice's
        start body's run) first: _Running.arrows."""
        names = {self.flow.key_names.get(directions.LEFT): -1, self.flow.key_names.get(directions.RIGHT): 1}
        out = [(0, held, True)] if held else []
        for e in events:
            d = names.get(e[1])
            if d is not None and len(e) >= 3 and e[2] in ("down", "up"):
                out.append((int(e[0]), d, e[2] == "down"))
        return tuple(out)

    def _splice_point(self, running: _Running, t: float) -> tuple[float, Body] | None:
        """Where the running plan can be taken over from at its plan time t at the soonest: (its plan time then, her body
        then as a new plan's start, at its time 0). On the ground in a run or standing on a leg (its spans and stands):
        t, there, the arrow the run holds held (Body.run) at its speed; else, from where a flight or a climb the leg is
        in lands her (its end), and from that leg's takeoff key on less KEY_COMMIT_S (the game has it before new keys
        could act: KEY_LEAD: a jump's key, a drop's, a walk-off's stop) too: nothing changes a flight. None past the plan's
        end."""
        for leg in running.legs:
            if leg.end.t <= t + 1e-9:
                continue
            off = [a.t0 for a in leg.air] + [c.t0 for c in leg.rope]
            if off and t >= min(off) - KEY_LEAD.get(leg.move.kind, KEY_LEAD["jump"]) - KEY_COMMIT_S:
                return leg.end.t, replace(leg.end, t=0.0, early_arrow=None)
            for sp in leg.spans:
                if sp.t0 - 1e-9 <= t <= sp.t1 + 1e-9:
                    dx, dt = sp.x1 - sp.x0, sp.t1 - sp.t0
                    v = max(-WALK_SPEED, min(WALK_SPEED, dx / dt)) if abs(dx) >= 1 and dt > 0 else 0.0
                    d = (0 if abs(dx) < 1 else (1 if dx > 0 else -1)) if not running.arrows \
                        else _arrow_at(running.arrows, t)
                    # (the arrow the plan's keys hold then, not the way the span goes: she slides on after letting
                    # go, and a splice there that took the slide for a run held its arrow on with no key to let it go
                    # in the new plan: up the tower from 13 she ran off 11's end and out of the map, its review's
                    # fix of 2026-10-30)
                    b0 = leg.start
                    # (the arrow as of its press's whole ms, as the plan's keys have it, flow.ms: a leg from 0.4006 s
                    # has it down at 401 ms, which sp.t0 + 1e-4 missed; the review of 2026-11-01)
                    if SPLICE_PHYSICS and d and abs(sp.t0 - b0.t) < 1e-6 and abs(sp.x0 - b0.x) < 1e-6 \
                            and _arrow_at(running.arrows, ms(sp.t0) / 1000) == d:
                        x, v = _ground(b0, d, t)  # (the leg's run from its start, its arrow held all along)
                        return t, Body(0.0, x, self.model.platform(sp.platform), d, v=v)
                    return t, Body(0.0, sp.x_at(t), self.model.platform(sp.platform), d, v=v if d else 0.0)
            for sp in leg.stands:
                if sp.t0 - 1e-9 <= t <= sp.t1 + 1e-9:
                    return t, Body(0.0, sp.x_at(t), self.model.platform(sp.platform))
            if t < leg.start.t:
                return leg.start.t, replace(leg.start, t=0.0, early_arrow=None)
            return None
        return None

    def _stop_calls(self, st: TripState, state, running: _Running) -> bool:
        """Whether, where the running chain stops (its last leg's end), what the trip's pass asks first would call for
        her there: the trip's `until` (the hunt loop's: a target in reach she may shoot, a pile, close quarters:
        Hunt._interrupt) or, in hunt mode, a monster in reach to shoot on the way (Escort.target), as the state has the
        monsters; not while her invulnerability after a touch lasts (POST_HIT: neither is asked then). A chain that stops
        there isn't run on past it (`_switch`'s 'better'): the pass decides there, as it would have."""
        if self._post_hit(st):
            return False
        end = running.legs[-1].end
        there = self._state_scene(state, end)
        if there is None:
            return True  # (not known: as it was, she stops)
        there.what_if = True  # (made up: what the hunt's until decides there isn't kept, Hunt._sweep_calls)
        if st.until is not None and st.until(there):
            return True
        return self._mode == HUNT and st.tries.get(end.platform.id, 0) < ON_WAY_TRIES             and self.escort is not None and self.escort.target(there) is not None

    def _state_scene(self, state, body: Body) -> Scene | None:
        """A glance's state (worldstate.WorldState) as a look with the whole screen's monsters (worldstate.to_scene), her
        put where `body` has her; None where it can't place them (her name tag not found)."""
        s = to_scene(state, self.model, self.senses.band, mobs=True, wide=True)
        if s.where is None or not s.searched:
            return None
        p = body.platform
        s.where = replace(s.where, x=body.x, y=p.y_at(p.clamp(body.x)), platform=p)
        return s

    def _knock_landing(self, touch: dict | None) -> Scene | None:
        """KNOCK_REPLAN: after a touch cut the running plan, the first state that has her on the ground again, as a look
        with the monsters (Senses.look of that very state: no wait), for the next pass to plan from at once (no wait for
        her to stand still, no look of its own: recover's 0.13 s median, p90 0.33, and its look's frame, live): from
        KNOCK_AIR_S after the touch's popup (her knock's flight), the scene's mode 'ground' where it has one, else her
        place on a platform at the height the state before had it (LANDED_DY), and the look standing (`standing`); with
        KNOCK_STILL, once her slide has ended too (a state on the ground within KNOCK_STILL_DX of the one before it).
        None if none does within KNOCK_LAND_S (then she's waited for as before: recover)."""
        since = touch.get("at", time.monotonic()) if isinstance(touch, dict) else time.monotonic()
        end, prev, last = time.monotonic() + KNOCK_LAND_S, None, None
        while time.monotonic() < end:
            self.senses.wait_frame(GLANCE_S)
            self.senses.glance()
            state = getattr(self.senses, "glanced", None)
            w = getattr(state, "where", None)
            if state is None or w is None or not getattr(state, "placed", False):
                continue
            mode = getattr(state, "mode", None)
            on = w[2] is not None and (mode == "ground" if mode is not None else
                                       prev is not None and abs(prev[1] - w[1]) <= LANDED_DY)
            prev = w
            if KNOCK_STILL:  # (her slide over: this state's x by the last other state's on the ground)
                seq = getattr(state, "seq", None)
                if last is not None and seq == last[0]:
                    continue  # (the same state glanced again)
                still = on and last is not None and abs(w[0] - last[1]) <= KNOCK_STILL_DX
                last = (seq, w[0]) if on else None
                on = still
            if state.t < since + KNOCK_AIR_S or not on:
                continue
            s = self.senses.look(mobs=True, wide=not self.fight, after=state.t)
            if s.where is not None and s.where.platform is not None and standing(s):
                return s
        return None

    def _keys_seen(self, running: _Running, glanced: list, until: float | None = None, runs: bool = True) -> None:
        """What her dot showed of a chain's jumps and runs (the glances `glanced`: frame times and dots), told to
        `moves` (see LIFT_UNITS): each planned jump's lift-off (a jump's or a jump up's: a climb's onto a rope she can
        grab with up alone) and each run of RUN_UNITS, seen for LIFT_SEEN_S and RUN_SEEN_S at least (not past `until`,
        when the chain was cut short: off its path with her dot standing, a run that didn't move her is cut so)."""
        if self.moves is None or not glanced:
            return
        to_map = self.senses.localizer.to_map
        pts = [(t, *to_map(d)) for t, d in glanced]
        end = float("inf") if until is None else until
        for leg in running.legs:
            if leg.move.kind in LIFT_KINDS:
                for arc in leg.air:
                    if arc.vy >= 0:  # (a fall: off an edge, a drop)
                        continue
                    t0 = running.start + arc.t0
                    t1 = min(t0 + LIFT_BY_S, end)
                    before = [y for t, _, y in pts if t0 - LIFT_BEFORE_S <= t <= t0]
                    after = [y for t, _, y in pts if t0 < t <= t1]
                    if t1 - t0 >= LIFT_SEEN_S and before and len(after) >= GLANCES_MIN:
                        self.moves.jumped(t0, min(after) <= min(before) - LIFT_UNITS)
            for sp in leg.spans if runs else ():
                if abs(sp.x1 - sp.x0) < RUN_UNITS:
                    continue
                t0 = running.start + sp.t0
                t1 = min(running.start + sp.t1 + RUN_AFTER_S, end)
                xs = [x for t, x, _ in pts if t0 <= t <= t1]
                # (seen for RUN_SEEN_S, or cut short once her walk would have gone RUN_CUT_UNITS: half of it seen)
                need = RUN_SEEN if t1 - t0 >= RUN_SEEN_S else (t1 - t0) * WALK_SPEED / 2
                if (t1 - t0 >= RUN_SEEN_S or (t1 - t0) * WALK_SPEED >= RUN_CUT_UNITS) and len(xs) >= 3:
                    self.moves.ran(t0, max(xs) - min(xs) >= need)

    def _queued_meets(self, queued: _Running) -> str | None:
        """QUEUE_CHECK: why the chain queued to start next mustn't (its first leg a drop or a climb, its flight and
        landing or its climb and the rope's top meeting a monster the feed has now: the last glance's tracks, live or out
        of sight QUEUE_SEEN_S at most, of the kinds the chain keeps clear of, walking as followed; her invulnerability
        counted), or None."""
        leg = queued.legs[0]
        state = getattr(self.senses, "glanced", None)
        if leg.move.kind not in ("drop", "climb") or state is None or not getattr(state, "placed", False):
            return None
        tracks = [self._track_of(tr, queued.start) for tr in state.tracks
                  if tr.platform is not None and (tr.state == LIVE or tr.state == HIDDEN
                                                  and state.t - tr.last_seen <= QUEUE_SEEN_S)
                  and (not self.fight or self.escort.threat(tr.kind))]
        if not tracks:
            return None
        off = min([a.t0 for a in leg.air] + [c.t0 for c in leg.rope], default=leg.start.t)
        self._immunity(queued.start)
        hit = self.flow.danger(leg, tracks, linger=END_LINGER if leg is queued.legs[-1] else 0.0, after=off)
        if hit is None:
            return None
        return (f"a {hit[0].kind or 'monster'} on {hit[0].platform} by its {leg.move.kind} -> {leg.move.dst} "
                f"({hit[1] - (time.monotonic() - queued.start):.2f} s ahead, the newest tracks)")

    def _climbing(self, running: _Running, dot) -> tuple[float, float, float] | None:
        """Whether she's climbing a rope of the running plan now (its Climb from the grab to UP_EXTRA past the top, and
        her last glance's dot by the rope and CLIMB_ON_UNDER or more under its top, or no dot and the plan's height
        so): the rope's top, her height on it (her dot's, else the plan's) and its x; None if not."""
        t = time.monotonic() - running.start
        for leg in running.legs:
            for c in leg.rope:
                if not c.t0 <= t <= c.t1 + UP_EXTRA:
                    continue
                if dot is None:
                    y = c.at(t)[1]
                    return (c.y1, y, c.x) if y - c.y1 >= CLIMB_ON_UNDER else None
                mx, my = self.senses.localizer.to_map(dot)
                if abs(mx - c.x) <= CLIMB_ON_X and c.y1 + CLIMB_ON_UNDER <= my <= c.y0 + CLIMB_ON_X:
                    return c.y1, my, c.x
        return None

    def _top_taken(self, x: float, top: float, up_s: float) -> str | None:
        """CLIMB_LOOK: why the top of the rope at map x (its top at map y `top`) isn't clear for her to climb on to it
        (a monster the feed has now on the platform there, walking as followed, whose box can be on hers at the top
        from her invulnerability's end, less IMMUNE_MARGIN_S, to `up_s` and STAND_PAST_S on), or None; None too while
        she gets up within the window with STAND_PAST_S to spare (the touch that cut the climb left her one: up through
        the monster then is safe, and the next pass gets her clear of it)."""
        tr = self._top_taker(x, top, up_s)
        return None if tr is None else f"a {tr.kind or 'monster'} by its top"

    def _top_taker(self, x: float, top: float, up_s: float):
        """_top_taken's monster (the feed's track: tracking.Track), or None."""
        state = getattr(self.senses, "glanced", None)
        q = self.model.platform_under(x, top - 1, reach=20)
        if state is None or q is None:
            return None
        now = time.monotonic()
        until = getattr(self.senses, "invulnerable_until", None)
        window = max(0.0, until() - IMMUNE_MARGIN_S - now) if until is not None else 0.0
        if up_s + STAND_PAST_S <= window:
            return None
        for tr in state.tracks:
            if tr.platform != q.id or tr.state not in (LIVE, HIDDEN) or state.t - tr.last_seen > QUEUE_SEEN_S \
                    or (self.fight and not self.escort.threat(tr.kind)):
                continue
            mo = self._track_of(tr, now)
            lo, hi = mo.reach_over(min(window, up_s), up_s + STAND_PAST_S, q)
            if lo - mo.half - her_body.HER_HALF <= x <= hi + mo.half + her_body.HER_HALF:
                return tr
        return None

    def _climb_on(self, top: float, y: float, x: float | None = None, touch: dict | None = None) -> None:
        """A touch while she climbs a rope (_climbing: a MISS too; `touch` the one told, Senses.touched_since's, None for
        her HP dropping alone): the cut let go of every key, up too, and the game keeps her on the rope when she's
        touched there (live on 2026-10-26, 11 of 11 hits on a rope), where she'd hang below its top (see CLIMB_ON); up
        again, to the top and onto the platform there, held as a chain's climb holds it (the climb from her height at
        flow.ROPE_SPEED, the step onto the platform in it, and UP_EXTRA: 0.75 s from 52 under, where mover.climb_ms's
        0.945 s had the same mushroom touch her again on the platform 9 ms before it ended in loop_tree_walls), waited
        for, so the next pass looks once she's there. With the top taken (CLIMB_LOOK) and ROPE_HANG, down to HANG_UNDER
        under it (and the rope's _hang_drop; where she's higher) and a hang there till the way up is clear (_hang), where
        the hang place is clear of every monster (_hang_safe); else down the rope to its bottom, as before. Where the
        touch is worth taking (HANG_TAKE: the wait's HANG_S against the touch, `_take_now`, her HP as read less `touch`
        where it may not have it yet) up again at once instead, a 'rope_hang' event of no hang ('took', 0 s)."""
        k = self.actions.keys.up.name
        hold = ms(max(0.0, y - top) / ROPE_SPEED + UP_EXTRA)
        if CLIMB_LOOK and x is not None and (why := self._top_taken(x, top, hold / 1000)) is not None:
            rope = min(self.model.ropes, key=lambda r: abs(r.x - x) + abs(r.top - top))
            down = self.actions.keys.down.name
            at = max(y, rope.top + HANG_UNDER + self._hang_drop(rope))
            shown = math.inf if touch is None else touch.get("at", math.inf)  # (none told: her HP dropping alone)
            if ROPE_HANG and HANG_TAKE and (q := route.hung_from(self.model, rope)) is not None \
                    and (taker := self._top_taker(x, top, hold / 1000)) is not None \
                    and (took := self._take_now(rope, y, HANG_S, self._top_touch(rope, q, y, taker), touched=shown)):
                self.log(f"    touched on the rope {y - top:.0f} under its top, {why}: {took}: up again ({hold} ms)")
                telemetry.emit("event", kind="climb_on", text=f"touched on the rope {y - top:.0f} under its top, {why}: "
                                                              f"up again, the touch taken")
                telemetry.emit("event", kind="rope_hang", step="hang", text=f"{why}: touched on the rope",
                               x=rope.x, under=round(y - rope.top, 1))
                telemetry.emit("event", kind="rope_hang", step="took", text=took, x=rope.x,
                               under=round(y - rope.top, 1), s=0.0, **self._took)
                self.actions.play([[0, k, "down"], [hold, k, "up"]], "climb_on")
                return
            if ROPE_HANG and at <= rope.bottom and route.hung_from(self.model, rope) is not None \
                    and self._hang_safe(rope, at):
                self.log(f"    touched on the rope {y - top:.0f} under its top, {why}: "
                         + (f"down to {at - rope.top:.0f} under it to hang there" if at > y else "hanging there"))
                telemetry.emit("event", kind="climb_on", text=f"touched on the rope {y - top:.0f} under its top, {why}: "
                                                              f"hanging {at - rope.top:.0f} under it")
                if at > y + 1:
                    self.actions.play([[0, down, "down"], [ms((at - y) / CLIMB_SPEED), down, "up"]], "hang_down")
                self._hang(rope, at, why)
                return
            hold = ms(max(0.0, rope.bottom - y) / ROPE_SPEED + UP_EXTRA)
            self.log(f"    touched on the rope {y - top:.0f} under its top, {why}: down again ({hold} ms)")
            telemetry.emit("event", kind="climb_on", text=f"touched on the rope {y - top:.0f} under its top, {why}: "
                                                          f"down again")
            self.actions.play([[0, down, "down"], [hold, down, "up"]], "climb_down")
            return
        self.log(f"    touched on the rope {y - top:.0f} under its top: up again ({hold} ms)")
        telemetry.emit("event", kind="climb_on", text=f"touched on the rope {y - top:.0f} under its top: up again")
        self.actions.play([[0, k, "down"], [hold, k, "up"]], "climb_on")

    def _top_touch(self, rope: Rope, q: Platform, y: float, taker) -> tuple[MonsterTrack, float]:
        """HANG_TAKE after a touch on the rope with the top taken (_climb_on): the touch she'd climb into, by `taker`
        (the feed's track _top_taker found) on her way up `rope` from her feet at map y onto platform `q`, and when
        (plan time from now): its first meeting with her once the invulnerability the touch on the rope left her is
        over (flow.danger, STAND_PAST_S standing at the top), at the climb-over's end where the planner's reach has none;
        standing at the top, so its knock counts (_knock_cost), where it was priced as a touch on the rope."""
        mo, now = self._track_of(taker, time.monotonic()), time.monotonic()
        self._immunity(now)
        leg = self._up_leg(rope, q, y)
        meet = self.flow.danger(leg, [mo], linger=STAND_PAST_S)
        return mo, (leg.end.t if meet is None else meet[1])

    # --- hanging under a rope's top (ROPE_HANG) ------------------------------------------------------------------

    def _hang_check(self, running: _Running, dot, seen_at: float | None) -> tuple[Rope, float, str] | None:
        """ROPE_HANG: whether the running chain's climb (the plan's Climb whose time is now: from half a second before
        the grab, since she may grab sooner than planned, to UP_EXTRA past the top) stops on the rope now, by her last
        glance (`dot`, its frame shown at `seen_at`): _hang_due, from where the plan has her grab it. None if not."""
        if dot is None or seen_at is None:
            return None
        t = time.monotonic() - running.start
        for leg in running.legs:
            for c in leg.rope:
                if c.t0 - 0.5 <= t <= c.t1 + UP_EXTRA:
                    rope = next((r for r in self.model.ropes if abs(r.x - c.x) <= 1 and abs(r.top - c.y1) <= 1), None)
                    return None if rope is None else self._hang_due(rope, c.y0, dot, seen_at)
        return None

    def _hang_due(self, rope: Rope, grab: float, dot, seen_at: float | None) -> tuple[Rope, float, str] | None:
        """ROPE_HANG: whether she stops climbing `rope` now, by her dot `dot` (its frame shown at `seen_at`), having
        grabbed it with her feet at about map y `grab`: on it (within CLIMB_ON_X of it, above where she grabs it, and
        not in the air by the glance's state: in the jump at it she isn't yet, and the jump's top is 5 over the grab),
        where the cut would stop her (her dot's height less what she climbs in the frame's age and HANG_LAG_S, at
        flow.ROPE_SPEED) is HANG_UNDER or less under its top by the next frame, but HANG_FLOOR or more now (both deeper
        by the rope's _hang_drop); the way up from where she is isn't clear (_way_up) and where she'd stop is
        (_hang_safe). The rope, where she'd stop (map y) and why; None if not."""
        if dot is None or seen_at is None or route.hung_from(self.model, rope) is None:
            return None
        mx, my = self.senses.localizer.to_map(dot)
        if abs(mx - rope.x) > CLIMB_ON_X or not rope.top <= my <= grab + 5:
            return None
        if getattr(getattr(self.senses, "glanced", None), "mode", None) == "air":  # (the scene's: the jump at it)
            return None
        stop, drop = my - ROPE_SPEED * (max(0.0, time.monotonic() - seen_at) + HANG_LAG_S), self._hang_drop(rope)
        if stop - ROPE_SPEED / HZ > rope.top + HANG_UNDER + drop or stop < rope.top + HANG_FLOOR + drop:
            return None  # (the next frame decides, or too late: on up, as before)
        why = self._way_up(rope, my)
        if why is None or not self._hang_safe(rope, stop):
            return None
        return rope, stop, why

    def _hang_drop(self, rope: Rope) -> float:
        """ROPE_HANG: how much deeper than HANG_UNDER and HANG_FLOOR she hangs under `rope`'s top: how far the surface
        of the platform at its top (route.hung_from) dips under the rope's top where a monster standing on it reaches
        her box (its middle inside the platform's walk edges, monsters.WALK_EDGE, and within HER_HALF and the widest
        touch box of the map's monsters of the rope: 50 on Tree Dungeon); 0 where it doesn't dip (HANG_UNDER is for a
        flat top). Tree Dungeon: 9.7 under the top of 111 (18 slopes down to -883.3 at x 61), 4.8 under 146's (30), 1.5
        under -192's (29), none under the others' (-80's, -8's). Kept for each rope."""
        drops = self.__dict__.setdefault("_drops", {})
        if (key := (rope.x, rope.top)) not in drops:
            q, drop = route.hung_from(self.model, rope), 0.0
            if q is not None:
                half = max((touch_box(kind_of(name), body)[0] for name, body in self.model.mob_bodies.items()),
                           default=touch_box("")[0])
                lo = max(rope.x - her_body.HER_HALF - half, q.left + WALK_EDGE)
                hi = min(rope.x + her_body.HER_HALF + half, q.right - WALK_EDGE)
                if lo <= hi:  # (the surface is straight between its footholds' ends)
                    xs = [lo, hi] + [x for seg in q.segments for x in (seg.x1, seg.x2) if lo < x < hi]
                    drop = max(0.0, *(q.y_at(x) - rope.top for x in xs))
            drops[key] = drop
        return drops[key]

    def _newest_tracks(self, now: float | None = None) -> list[MonsterTrack]:
        """The monsters of the newest state (the last glance's: Senses.glanced), as the move planner sees them from now
        (`now`, time.monotonic: their plan time's start; None: now) (_track_of), one standing setting off either way
        HANG_UNSURE_S at the most: live, or out of sight QUEUE_SEEN_S at most, on a platform; with an escort those whose
        touch counts (as _queued_meets has them)."""
        state = getattr(self.senses, "glanced", None)
        if state is None:
            return []
        now = time.monotonic() if now is None else now
        tracks = [self._track_of(tr, now) for tr in state.tracks
                  if tr.platform is not None and (tr.state == LIVE or tr.state == HIDDEN
                                                  and state.t - tr.last_seen <= QUEUE_SEEN_S)
                  and (not self.fight or self.escort.threat(tr.kind))]
        return [replace(mo, unsure=HANG_UNSURE_S) if mo.unsure is not None else mo for mo in tracks]

    def _up_leg(self, rope: Rope, q: Platform, y: float) -> Leg:
        """The rest of a climb up `rope` from her feet at map y set off now (plan time 0): at physics.CLIMB_SPEED (SPEED +
        20 by her stat window: the thief's 121, hers 120 units/s, as she climbed), and her climb-over (flow.CLIMB_OVER)
        onto platform `q` at its top."""
        up = max(0.0, y - rope.top) / CLIMB_SPEED
        return Leg(Move("climb", q.id, q.id, rope.x, rope.x, rope_x=rope.x, rope_top=rope.top, rope_bottom=rope.bottom),
                   Body(0.0, rope.x, q), Body(up + CLIMB_OVER, rope.x, q), rope=[Climb(0.0, up, rope.x, y, rope.top)],
                   stands=[Span(q.id, up, up + CLIMB_OVER, rope.x, rope.x)])

    def _way_up(self, rope: Rope, y: float) -> str | None:
        """ROPE_HANG: why the rest of a climb up `rope` from her feet at map y isn't clear now (a monster and its
        platform: _way_up_hit's), or None."""
        hit = self._way_up_hit(rope, y)
        return None if hit is None else _blocked(hit)

    def _way_up_hit(self, rope: Rope, y: float) -> tuple[MonsterTrack, float] | None:
        """ROPE_HANG: the first monster the rest of a climb up `rope` from her feet at map y set off now would meet, and
        when (plan time from now), or None: the climb (_up_leg) and STAND_PAST_S standing on the platform at its top, her
        box all the way (flow.danger) against every monster of the newest state (_newest_tracks: on any platform, so a
        walker on 9 by its end over the top of the rope at -80 too), each walking as followed; her invulnerability
        counted."""
        q, tracks = route.hung_from(self.model, rope), self._newest_tracks()
        if q is None or not tracks:
            return None
        self._immunity(time.monotonic())
        return self.flow.danger(self._up_leg(rope, q, y), tracks, linger=STAND_PAST_S)

    def _hang_safe(self, rope: Rope, y: float, s: float = HANG_S) -> bool:
        """ROPE_HANG: whether hanging on `rope` with her feet at map y for `s` (HANG_S) keeps her box clear of every
        monster of the newest state (_newest_tracks, on any platform: one at the rope's foot too; flow.danger)."""
        q, tracks = route.hung_from(self.model, rope), self._newest_tracks()
        if q is None or not tracks:
            return True
        self._immunity(time.monotonic())
        leg = Leg(Move("climb", q.id, q.id, rope.x, rope.x, rope_x=rope.x, rope_top=rope.top, rope_bottom=rope.bottom),
                  Body(0.0, rope.x, q), Body(s, rope.x, q), rope=[Climb(0.0, s, rope.x, y, y)])
        return self.flow.danger(leg, tracks) is None

    def _hang(self, rope: Rope, y: float, why: str) -> str:
        """ROPE_HANG: she hangs on `rope` (every key let go: the game keeps her there), her feet at about map y, until
        the way up from there is clear (_way_up, at every glance with a state, her height her dot's while it's by the
        rope; one without a state tells nothing, where the review of 2026-11-03 had it end the hang as clear), a touch
        is told (her invulnerability then takes her past it), or her wait is over: HANG_S (a backstop event: up
        regardless, as a pass regardless goes), or longer where the monsters in the way are predicted to walk off
        (HANG_LONGER: up as predicted clear); or at once, at its start or at any look, where the rest of the wait costs
        her more than the touch (HANG_TAKE: `_hang_judge`); then up the rest, held as _climb_on holds it, waited for.
        The Lie Detector seen at a glance stops it there (LieDetectorAlarm, as _pass's check: no key goes out, and the
        hunt stops then, not up to HANG_S later), but on a retreat's own trip. Each hang is a telemetry event
        'rope_hang' at its start (step 'hang') and its end ('clear', 'touched', 'regardless', 'waited' or 'took': how
        long, what blocked the way; the wait's longer budget, the touch's numbers), and in it 'longer' and 'back'
        (HANG_LONGER). How it ended, for the cut's reason; `_hang_touch` the touch told from its start to the end of
        the climb after it (None: none), which the caller takes as the cut's (POST_HIT: no fight while her
        invulnerability after it lasts, so the next chain passes the monster in it, as after a touch on the rope;
        taken as no touch, nav_tree_rope_hit, from 19 up to 16, fought the Green Mushroom by the top after the hang,
        was knocked back down to 19 and climbed again, 105 s and 37 touches with HANG_S 4 s, where 7.1 s and 1 so). A
        touch told only after it (the popup's lag) is taken so by the caller too (_carry_out_chain, _carry_out_move)."""
        start, under = time.monotonic(), y - rope.top
        since = getattr(self.senses, "glanced_at", None)
        self.log(f"    {why}: hanging on the rope at x {rope.x:.0f}, {under:.0f} under its top")
        telemetry.emit("event", kind="rope_hang", step="hang", text=f"{why}: hanging {under:.0f} under the top of "
                                                                    f"the rope at x {rope.x:.0f}",
                       x=rope.x, under=round(under, 1))
        how, block, wait = "regardless", why, _Wait(HANG_S)
        judged = (HANG_LONGER or HANG_TAKE) and self.price is not None  # (a retreat's navigator has none: HANG_S)
        took = self._hang_judge(rope, y, start, wait, self._way_up_hit(rope, y)) if judged else None
        while took is None and time.monotonic() - start < wait.s:
            self.senses.wait_frame(GLANCE_S)
            dot, _ = self.senses.glance()
            if getattr(self.actions, "kind", None) is None:  # (the Lie Detector seen: the hunt stops now, as _pass)
                client.check_alarm()
            if dot is not None:
                mx, my = self.senses.localizer.to_map(dot)
                if abs(mx - rope.x) <= CLIMB_ON_X and rope.top <= my <= rope.bottom + CLIMB_ON_X:
                    y = my
            if self._touch_since(since) is not None:
                how = "touched"
                break
            if getattr(self.senses, "glanced", True) is None:  # (no state: no news, not the way up clear; LookLost
                continue                                       # once there's been none GLANCE_LOST_S: Senses.glance)
            if (hit := self._way_up_hit(rope, y)) is None:
                how = "clear"
                break
            block = _blocked(hit)
            if judged:
                took = self._hang_judge(rope, y, start, wait, hit)
        held = time.monotonic() - start
        if took is not None:
            how = "took"
        elif how == "regardless" and wait.s > HANG_S:
            how = "waited"
        text = {"clear": f"the way up clear after {held:.1f} s", "touched": f"touched after {held:.1f} s",
                "regardless": f"up regardless after {held:.1f} s ({block})",
                "waited": f"up after {held:.1f} s, the way up predicted clear by then ({block})",
                "took": f"up after {held:.1f} s, {took}"}[how]
        if how == "regardless":
            telemetry.emit("event", kind="backstop", text=f"HANG_S: up the rope at x {rope.x:.0f} regardless: {block}")
        hold = ms(max(0.0, y - rope.top) / ROPE_SPEED + UP_EXTRA)
        self.log(f"    hung {held:.1f} s, {text}: up the rest ({hold} ms)")
        more = dict(self._took) if how == "took" else {}
        if wait.s > HANG_S:
            more["wait"] = round(wait.s, 2)
        telemetry.emit("event", kind="rope_hang", step=how, text=text, x=rope.x, under=round(y - rope.top, 1),
                       s=round(held, 2), **more)
        k = self.actions.keys.up.name
        self.actions.play([[0, k, "down"], [hold, k, "up"]], "hang_up")
        self._hang_touch = self._touch_since(since)
        return text

    def _hang_judge(self, rope: Rope, y: float, start: float, wait: _Wait,
                    hit: tuple[MonsterTrack, float] | None) -> str | None:
        """A hang's wait (`wait`, the hang started at `start`, her feet at map y) judged at its start and at every look,
        `hit` the newest state's first monster in the way up (_way_up_hit's): HANG_LONGER's budget, then HANG_TAKE's
        choice: why the touch is taken (she climbs at once), or None (she hangs on)."""
        now = time.monotonic()
        if HANG_LONGER and not wait.back and hit is not None:
            off = self._off_walk(wait, now) if wait.walks else None
            got = None if off is not None else                 self._walks_off(rope, y, max(0.0, start + HANG_HORIZON_S - HANG_MARGIN_S - now))
            clear = None if got is None else now + got[0]
            if got is not None and got[0] <= 0 and wait.clear is not None and wait.clear <= now:
                clear = wait.clear  # (predicted clear since then: the margin from then, not from every look)
            if clear is not None and clear + HANG_MARGIN_S - start > HANG_HORIZON_S + 1e-9:
                clear = None
            wait.clear = clear
            until = None if clear is None else clear + HANG_MARGIN_S - start
            if until is not None and until > HANG_S and not self._hang_safe(rope, y, until - (now - start)):
                until, off = None, "her hang place not clear that long"
            if until is not None and until > HANG_S:
                if not wait.said:
                    names = ", ".join(f"a {mo.kind or 'monster'} on {mo.platform} at {mo.vx:+.0f}"
                                      for _, mo, _ in got[1])
                    text = (f"{_blocked(hit)}, walking off ({names}): the way up predicted clear "
                            f"{until - HANG_MARGIN_S:.1f} s into the hang, hanging {until:.1f} s")
                    self.log(f"    {text}")
                    telemetry.emit("event", kind="rope_hang", step="longer", text=text, x=rope.x,
                                   at=round(now - start, 2), clear=round(until - HANG_MARGIN_S, 2), wait=round(until, 2))
                    wait.said = True
                wait.s, wait.walks = until, got[1]
            elif off is not None and wait.walks is not None:  # (off its walk: HANG_S from the start, for good)
                wait.s, wait.walks, wait.back = HANG_S, None, True
                text = f"{off}: back to HANG_S ({HANG_S:g} s)"
                self.log(f"    {text}")
                telemetry.emit("event", kind="rope_hang", step="back", text=text, x=rope.x, at=round(now - start, 2))
            else:  # (no clear predicted past HANG_S within HANG_HORIZON_S at this look: HANG_S, its walks still watched)
                wait.s = HANG_S
                if got is not None and wait.walks is not None:
                    wait.walks = got[1]
        if HANG_TAKE and hit is not None:
            return self._take_now(rope, y, start + wait.s - now, hit)
        return None

    def _walks_off(self, rope: Rope, y: float, within: float) -> tuple[float, list] | None:
        """HANG_LONGER: when the way up `rope` from her feet at map y is predicted clear, s from now (0 to `within`,
        every WALK_OFF_STEP), and the walks it's predicted by ([(the feed's track id, the track, now)] of the monsters
        in the way now): where every monster in the way now (flow.danger with the planner's reach, as _way_up's) walks,
        the first set-off whose climb, climb-over and STAND_PAST_S at the top (_up_leg) meets none of the newest state's
        monsters where each likeliest is by then (MonsterTrack.at: the way it's followed walking, at its kind's speed
        from the game data, turning at its walk edges; one standing where it stands), her invulnerability counted; 0 and
        none with none in the way now. None if one in the way now stands (or may set off either way) or chases her
        (`_chasing`: one she's hurt), or none is clear within `within`."""
        q, now = route.hung_from(self.model, rope), time.monotonic()
        tracks = self._newest_tracks(now)
        if q is None or not tracks or within < 0:
            return None
        immune = self._immunity(now)
        leg = self._up_leg(rope, q, y)
        way = [self.flow.danger(leg, [mo], linger=STAND_PAST_S) is not None for mo in tracks]  # (in the way now)
        if not any(way):
            return 0.0, []
        if any(w and (not mo.vx or self._chasing(mo)) for w, mo in zip(way, tracks)):
            return None
        tracks = [replace(mo, vx=math.copysign(mo.speed, mo.vx)) if mo.vx else mo for mo in tracks]
        path = self.flow.path(leg, STAND_PAST_S)
        end = path[-1][0] + within
        near = []  # (those whose box can get near her way up by then: flow.danger's bound)
        for mo in tracks:
            p = self.model.platform(mo.platform)
            reach = mo.half + her_body.HER_HALF + mo.speed * end
            if p.left - reach < rope.x < p.right + reach and y > p.top - mo.tall \
                    and rope.top - her_body.HER_TALL < her_body.low_end(p):
                near.append((mo, p))
        d = 0.0
        while d <= within + 1e-9:
            if not any(t + d >= immune and her_body.touches(x, fy, c, c, mo.half, p.y_at(c), mo.tall)
                       for t, x, fy in path for mo, p in near for c in (mo.at(t + d, p),)):
                return d, [(mo.track, mo, now) for w, mo in zip(way, tracks) if w]
            d += WALK_OFF_STEP
        return None

    def _chasing(self, mo: MonsterTrack) -> bool:
        """HANG_LONGER: whether monster `mo` (one of the newest state's) chases her, so a walk of it is no walk off: one
        she's hurt (the feed's track of it has its HP bar seen: tracking.Track.bar), as the user's monster rule has it
        (a hit draws a chase, 3-4 minutes, each hit anew: it walks at her, turning back past her, wherever she is above
        or below it). On the merged world 63 of the 72 longer waits for one chasing her were put back for its turn, and
        none of them cleared (D:/MapleHelper-scratch/recfix/rope-on)."""
        state = getattr(self.senses, "glanced", None)
        if state is None or mo.track is None:
            return False
        tr = next((tr for tr in state.tracks if getattr(tr, "id", None) == mo.track), None)
        return tr is not None and getattr(tr, "bar", None) is not None

    def _off_walk(self, wait: _Wait, now: float) -> str | None:
        """HANG_LONGER: why a monster the longer wait is for is off its predicted walk at the newest state (the feed's
        track of it): standing, or walking the other way from where its prediction has it walk now, but within
        OFF_WALK_EDGE of its walk edges, where it turns as predicted (give or take a frame); None if none is (one gone
        from the state is out of the way)."""
        state = getattr(self.senses, "glanced", None)
        if state is None:
            return None
        by_id = {getattr(tr, "id", None): tr for tr in state.tracks}
        for tid, mo, t0 in wait.walks:
            tr = by_id.get(tid) if tid is not None else None
            if tr is None or tr.state not in (LIVE, HIDDEN):
                continue
            p = self.model.platform(mo.platform)
            lo, hi = walk_edges(p)
            at, on = mo.at(now - t0, p), mo.at(now - t0 + 0.05, p)
            if min(at - lo, hi - at) < OFF_WALK_EDGE or min(tr.x - lo, hi - tr.x) < OFF_WALK_EDGE:
                continue
            if abs(tr.vx) <= STILL_VX:
                return f"the {tr.kind or 'monster'} on {tr.platform} stopped"
            if (tr.vx > 0) != (on > at):
                return f"the {tr.kind or 'monster'} on {tr.platform} turned"
        return None

    def _take_now(self, rope: Rope, y: float, left: float, hit: tuple[MonsterTrack, float | None],
                  touched: float | None = None) -> str | None:
        """HANG_TAKE: why she takes the touch of `hit`'s monster on the way up `rope` from her feet at map y (and when,
        plan time from now: _way_up_hit's; None: on the rope) rather than hang `left` s more, or None: the wait's s at
        what her time is worth up there (TopPrice.rate, `price`'s) more than the touches' EXP (the second at the top's
        by its kind: `_touched_again`) and their knocks' s at that rate (`_knock_cost`), her HP as last read over the
        safety stop's reserve by every touch it's priced as (TopPrice.margin: each one's HP by its kind) and, after a
        touch on the rope (`touched`: its popup's frame time, math.inf for her HP dropping alone; _climb_on's), that one
        too unless her HP was read HP_SETTLE_S or more after it (the map's largest touch: its kind isn't told). The
        numbers in `_took` (the 'rope_hang' event's)."""
        q, price = route.hung_from(self.model, rope), getattr(self, "price", None)
        if not HANG_TAKE or price is None or q is None or left <= 0:
            return None
        mo, t = hit
        got = price(q.id, mo.kind)
        if got is None or got.margin is None:
            return None
        knock = self._knock_cost(rope, q, y, mo, t)
        again = self._touched_again(rope, q, y, mo, t)  # (a second touch at the top once her invulnerability is over)
        second = None if again is None else got if again[0].kind == mo.kind else price(q.id, again[0].kind)
        if again is not None:
            knock += self._knock_cost(rope, q, y, *again)
        touches = 1 if again is None else 2
        need = got.hp + (0.0 if second is None else second.hp)
        if touched is not None and (got.read is None or got.read < touched + HP_SETTLE_S):
            need += got.largest  # (the touch on the rope just now, not in her HP as read yet)
        if second is None and again is not None or got.margin < need:
            return None
        exp = got.touch + (0.0 if second is None else second.touch)
        wait, cost = left * got.rate, exp + knock * got.rate
        if wait <= cost:
            return None
        self._took = {"left": round(left, 2), "rate": round(got.rate, 2), "wait_exp": round(wait, 1),
                      "touch_exp": round(exp, 1), "knock_s": round(knock, 2), "touches": touches,
                      "margin": round(got.margin), "need": round(need)}
        return (f"the touch taken: {left:.1f} s more of the wait at {got.rate:.1f} EXP a second ({got.words}), "
                f"{wait:.0f} EXP, over a {mo.kind or 'monster'}'s touch" + (" and a second at the top" if again else "")
                + f", {exp:.0f} EXP" + (f" and its knock's {knock:.1f} s" if knock else "")
                + f"; her HP {got.margin:.0f} over the reserve, {need:.0f} the touches'")

    def _touched_again(self, rope: Rope, q: Platform, y: float, mo: MonsterTrack,
                       t: float | None) -> tuple[MonsterTrack, float] | None:
        """HANG_TAKE: the monster that touches her again after the touch by `mo` taken at plan time t (None: on the
        rope now) on her way up `rope` from her feet at map y, and when (plan time from now): `mo` or any of the newest
        state's (_newest_tracks: the others at the top too), from her invulnerability's end after it (INVULNERABLE_S less
        IMMUNE_MARGIN_S) on, over the rest of the climb, the climb-over onto platform `q` and standing there to
        TAKE_AGAIN_S past that end (flow.danger, the planner's reach of each); None where none does. (Of the touches
        taken on her world with `mo` alone checked, 9 of 21 had a Horny chasing her at the top as the hang began.)"""
        inv = (0.0 if t is None else t) + INVULNERABLE_S - IMMUNE_MARGIN_S
        leg = self._up_leg(rope, q, y)
        kept = self.flow.immune
        self.flow.immune = max(kept, inv)
        try:
            return self.flow.danger(leg, [mo] + self._newest_tracks(),
                                    linger=max(STAND_PAST_S, inv + TAKE_AGAIN_S - leg.end.t))
        finally:
            self.flow.immune = kept

    def _knock_cost(self, rope: Rope, q: Platform, y: float, mo: MonsterTrack, t: float | None) -> float:
        """HANG_TAKE: s a touch by `mo` at plan time t of the way up `rope` from her feet at map y (None: on the rope)
        costs her in its knock: none on the rope (the game keeps her there: CLIMB_ON); standing on platform `q` at its
        top, the knock's flight (KNOCK_AIR_S) where it lands her on `q` (KNOCK_REACH away from the monster), else that
        and the planner's way back up from where she falls to (KNOCK_LOST_S where it has none)."""
        if t is None or t <= max(0.0, y - rope.top) / CLIMB_SPEED:
            return 0.0
        land = rope.x + (KNOCK_REACH if rope.x >= mo.at(t, self.model.platform(mo.platform)) else -KNOCK_REACH)
        if q.covers(land):
            return KNOCK_AIR_S
        below = self.model.platform_below(land, q.y_at(q.clamp(land)))
        back = None if below is None else self.planner.arrive(below.id, below.clamp(land), q.id, rope.x)
        return KNOCK_AIR_S + (KNOCK_LOST_S if back is None else back)

    def _ended(self, plan_id: str | None, cut: bool = False) -> None:
        """A chain's plan is over, its result read: its last key went its run on past it before now (send's), or now,
        cut short (every key let go: Actions.ended)."""
        tail = self._tails.pop(plan_id, 0.0)
        if hasattr(self.actions, "ended"):  # (a stand-in for Actions may not have it)
            self.actions.ended(0.0 if cut else tail)

    def _coming(self, running: _Running, room: bool = True) -> tuple[MonsterTrack, float, int] | None:
        """A monster the feed follows (the last glance's state: Senses.glanced) that the running chain's legs ahead of
        her now meet, REACT_LEAD_S on at the soonest, while she's on the ground with room to stop (`room`; else
        wherever she is: REPLAN replans from where she'll be): it, when (plan time) and its track's id; None if none,
        without a feed's glance, or (`room`) in the air or on a rope. Only the kinds the chain keeps clear of (with an
        escort, those whose touch counts), each once in REACT_AGAIN_S (see REACT)."""
        state = getattr(self.senses, "glanced", None)  # (a stand-in for Senses may not have it)
        if state is None or not state.placed or (self.fight and not self._threats()):
            return None
        now = time.monotonic()
        t = now - running.start
        if room and not self._room_to_stop(running, t):
            return None
        tracks, ids, raw, seen = [], [], [], state.t - running.start
        for tr in state.tracks:
            if tr.state not in (LIVE, HIDDEN) or tr.platform is None or state.t - tr.last_seen > REACT_AGAIN_S \
                    or now - self._reacted.get(tr.id, -1e18) < REACT_AGAIN_S:
                continue
            if (self.fight and not self.escort.threat(tr.kind)) or self._known(tr, running.known, seen):
                continue
            tracks.append(self._track_of(tr, running.start))
            ids.append(tr.id)
            raw.append(tr)
        if not tracks:
            return None
        self._immunity(running.start)  # (no touch counts while her invulnerability lasts: IMMUNE)
        for leg in running.legs:
            if leg.end.t < t + REACT_LEAD_S:
                continue
            if (hit := self.flow.danger(leg, tracks, after=t + REACT_LEAD_S)) is not None:
                i = next(i for i, mo in enumerate(tracks) if mo is hit[0])
                self._came = raw[i]  # (the next chain keeps clear of it: TripState.came)
                return hit[0], hit[1], ids[i]
        return None

    def _room_to_stop(self, running: _Running, t: float) -> bool:
        """Whether, t s into the running plan, she's running along a platform (or standing on it) with room to stop
        before its end: REACT_ROOM ahead of her the way she runs. Not in the air, on a rope, or while she waits for a
        jump to lift off or a drop to go through; and near an end, letting go of the keys slid her off it: with the
        reaction from anywhere on the ground, on Tree Dungeon (crowd, 8 seeds of 10 minutes) she missed 37 landings in
        one run where she had missed 4, 31 of them falls."""
        if self.flow.airborne(running.legs, t):
            return False
        for leg in running.legs:
            for sp in leg.spans:
                if sp.t0 <= t <= sp.t1:
                    p, x, go = self.model.platform(sp.platform), sp.x_at(t), sp.x1 - sp.x0
                    return (x + REACT_ROOM <= p.right) if go > 0 else (x - REACT_ROOM >= p.left) if go < 0 \
                        else p.left <= x <= p.right
        return False

    def _known(self, tr, known: tuple, t: float) -> bool:
        """Whether the look the running chain was planned from had the monster the feed follows as `tr` (tracking.Track),
        as of plan time t: one of its kind on its platform (`known`) that, walking as that look had it, can be within
        MATCH_SLACK of where the feed has it, and walks as the plan had it (`_as_planned`: one planned standing that has
        set off since, or planned walking one way that now walks the other or stands, isn't: the plan's walk of it no
        longer holds, the review of 2026-10-30: one that set off was within its reach and MATCH_SLACK, 85 units at 1 s,
        long after it had left the plan's spread). The chain was planned against it then, or past it on purpose (a way
        through after waiting for it, or regardless: _meet); REACT is for those the look hadn't."""
        p = self.model.platform(tr.platform)
        for mo in known:
            if mo.platform == tr.platform and mo.kind == tr.kind:
                lo, hi = mo.reach(t, p)
                if lo - MATCH_SLACK <= tr.x <= hi + MATCH_SLACK and _as_planned(mo, tr):
                    return True
        return False

    def _track_of(self, tr, start: float) -> MonsterTrack:
        """A monster the feed follows (tracking.Track) as the move planner sees it, as of its last sighting in the plan
        that started at `start` (time.monotonic): walking at its velocity as followed (to its kind's speed at most), or
        standing (under STILL_VX: where it may start walking either way, STAND_UNSURE_S of it at the most, as a look's
        standing one: _walking)."""
        speed = self.tracker.walking.get(tr.kind, DEFAULT_SPEED)
        half, top = touch_box(tr.kind, self.tracker.body.get(tr.kind))
        walks = abs(tr.vx) > STILL_VX
        unsure = not walks and _may_set_off(tr)
        vx = max(-speed, min(speed, tr.vx)) if walks else None if unsure else 0.0
        heading = (1 if vx > 0 else -1) if walks else 0
        return MonsterTrack(tr.platform, tr.x, speed, heading, half, vx, tr.last_seen - start,
                            MONSTER_HEIGHT.get(tr.kind, top), tr.facing, tr.kind, top, track=getattr(tr, "id", None),
                            trust=HEADING_TRUST_S, unsure=STAND_UNSURE_S if unsure else None)

    def _placed(self, dot, still: bool, running: _Running,
                seen_at: float | None = None) -> tuple[Platform | None, int | None, Leg, float | None]:
        """Where a glance at her dot puts her against the running plan: the platform she stands on
        (None if none, or her dot isn't `still`: it hasn't stood where it is for STILL_S), the
        platform she should be on by then (None: in the air, or about to take off or land), the leg she's
        on, and her x if she's standing where her run along that platform starts while the plan has her
        OFF_SPAN or more along it (else None). While the plan has her in the air (Flow.airborne) she stands
        nowhere: near the top of a jump her dot can sit still over a platform she's flying past. All of it as of
        the glance's frame (`seen_at`, when it was shown; None: now): at the perception's 15 frames a second a glance
        can be up to a period and a pass old (about 85 ms; 42 at 30 until 2026-10-30), and judged at now the plan had
        her a few units on from where the frame showed her."""
        t = (time.monotonic() if seen_at is None else seen_at) - running.start
        want = self.flow.expected(running.legs, t)
        # On a platform only when the dot sits at its height and has stopped: in the air the dot
        # passes 40-60 units under the platforms above, which read as "on" them before.
        mx, my = self.senses.localizer.to_map(dot)
        on = self.model.platform_under(mx, my - 6, reach=16) if still and not self.flow.airborne(running.legs, t) \
            else None
        leg = next((lg for lg in running.legs if lg.start.t <= t <= lg.end.t + 0.3), running.legs[-1])
        span = self.flow.span_at(running.legs, t) if on is not None else None
        behind = None
        if span is not None and span.platform == on.id and abs(mx - span.x0) <= OFF_SPAN / 2 \
                and abs(span.x_at(t) - mx) > OFF_SPAN:
            behind = mx
        return on, want, leg, behind

    def weight(self, m: Move) -> float:
        """This trip's random weight for a move (see `variety`)."""
        key = _key(m)
        if key not in self._weights:
            self._weights[key] = 1.0 + self.variety * self.rng.random()
        return self._weights[key]

    def recover(self, longest: float = 0.8, mobs: bool = False, wide: bool = False) -> Scene:
        """Right after a plan: wait until she's stopped moving (a hit knocks her back and she can't be
        steered for a moment; a stop slides her on), glancing at her dot, then one look to place her (with
        the monsters, if `mobs`; on the whole screen, if `wide`). With a feed, that look is of the frame the last
        glance saw her standing still in, or a later one (Senses.look's `after`): taken anew, it waited for the feed's
        next frame, 41 ms (median; 20 for a look taken here at today's costs) in the simulator's hunt loop."""
        end, stood, still = time.monotonic() + longest, None, None
        while time.monotonic() < end:
            dot, _ = self.senses.glance()
            stood = _stood(stood, self.senses.glanced_at, dot)
            if stood is not None and self.senses.glanced_at - stood[0] >= RECOVER_STILL_S:
                w = self.senses.localizer.where_dot(dot)
                if w.platform is not None:
                    still = self.senses.glanced_at if getattr(self.senses, "glanced", None) is not None else None
                    break
            self.senses.wait_frame(GLANCE_S)
        return self.mover.locate(mobs=mobs, wide=wide, after=still)

    # --- chains that follow on (follow) ------------------------------------------------------------

    def _tail(self, legs: list[Leg], path: list[Move] | None, monsters=()) -> float | None:
        """How long a chain's plan runs on past its last leg so that, before it ends, a look can place her
        on the platform it ends on and the next chain be planned from there (CHECK_AFTER, CHECK_LEAD):
        standing, or running on with the arrow held short of the route's next move and clear of the
        `monsters` the chain keeps clear of. None if there's no room for that: she stops where it ends."""
        last = legs[-1]
        end = last.end
        tail = max(0.0, _ground_from(last) + CHECK_AFTER + CHECK_LEAD - end.t)
        if not tail:
            return tail
        if end.run:
            done = sum(1 for leg in legs if leg.move.kind != "hop")  # hops are the chain's own, not the route's
            nxt = path[done] if path is not None and done < len(path) else None
            if nxt is None or (nxt.x - end.x) * end.run - RUN_ON_ROOM < tail * WALK_SPEED:
                return None
        if monsters:  # the run on (or the wait) mustn't take her into a monster
            x = end.x + end.run * WALK_SPEED * tail
            on = Leg(last.move, end, replace(end, t=end.t + tail, x=x),
                     [Span(end.platform.id, end.t, end.t + tail, end.x, x)])
            if self.flow.danger(on, list(monsters)):
                return None
        return tail

    def _check_at(self, running: _Running) -> float:
        """When the look placing her for the next chain is taken: once she's back on the ground for good,
        as late as still leaves CHECK_LEAD before the running plan ends."""
        last = running.legs[-1]
        return running.start + max(_ground_from(last) + CHECK_AFTER, last.end.t + running.tail - CHECK_LEAD)

    def _look_ahead(self, st: TripState | None, running: _Running, dst: int) -> _Ahead | None:
        """The look around late in the running chain (she's on the move), for the chain that may follow.
        None if she's to stop where it ends: she'll be there, or, with an escort, something there calls
        for her: the trip's `until`, or (hunt mode) a monster in reach to shoot. The look's monsters are
        judged from where the chain ends."""
        end = running.legs[-1].end
        if st is None or (end.platform.id == dst and not (st.through is not None and _after(running, st.path))):
            return None
        s = self.senses.look(mobs=True, wide=not self.fight)
        if s.where is None:
            return None
        if st.through is not None and s.where.platform is not None and not self._post_hit(st) and st.until is not None and st.until(s):
            return _no_follow("the trip's until, as she goes")
        p = end.platform
        there = replace(s, where=replace(s.where, x=end.x, y=p.y_at(end.x), platform=p), what_if=True)
        threats = self.fight and self._threats()  # (where the next chain's legs go: trip_monsters)
        ahead = {p.id} | _platforms(_after(running, st.path))
        seen = self._on_platforms(self.trip_monsters(s)) if threats else s.monsters
        tracks = tuple(self.tracker.update(s.t, seen, seen_at=s.t - running.ends))  # as of the next start
        if threats:  # and those her memory has there off the look's view
            tracks += tuple(self.remembered(s, ahead, seen, s.t - running.ends))
        if not self.fight:  # avoiding them: the next chain keeps clear of them
            return _Ahead(there, tracks)
        if st.until is not None and not self._post_hit(st) and st.until(there):  # (not while she's invulnerable after a
            return _no_follow("the trip's until, where it ends")                  # touch: POST_HIT, as _pass)
        if self._mode == HUNT and st.tries.get(p.id, 0) < ON_WAY_TRIES and self.escort.target(there):
            return _no_follow("a monster in reach where it ends")
        keep = tuple(t for t in tracks if self.escort.threat(t.kind)) if threats else ()
        return _Ahead(there, keep, watch=tracks)

    def _queue_next(self, st: TripState, running: _Running, ahead: _Ahead, dst: int, send) -> _Running | None:
        """Looks until one places her standing on the platform the running chain ends on (_landed_on), and the
        next chain planned from where that puts her when the running plan ends, sent to start then. None if
        she isn't placed there while the running plan still has SEND_S to go, nothing can be chained, or
        (with an escort) a monster is in the way of the next move or could get near the chain: she stops, and
        the next pass deals with it. (Planned from the predicted end instead, a landing 12.5 units off made
        the next running jump leave 12.5 units early; landing late, still in the air or sliding to a stop
        once her plan had ended, the next chain's keys came too early and she fell off.)"""
        end = running.legs[-1].end
        p, starts = end.platform, running.ends
        s = self._landed_on(p, starts - SEND_S)  # a look started later wouldn't leave time to plan and send
        if s is None:
            last = self._checked
            seen = "none" if last is None or last.where is None else (
                f"platform {last.where.platform.id if last.where.platform else None}, ground {last.ground}, "
                f"tag seen {last.tag_seen}")
            return _no_follow(f"not seen standing on it in time (last look: {seen}; "
                              f"{time.monotonic() - starts:+.3f} s from the plan's end)")
        # Where she'll be when the running plan ends: running on with the arrow held, or standing.
        x = min(max(s.where.x + end.run * WALK_SPEED * max(0.0, starts - s.t), p.left), p.right)
        # The running chain's moves are made as far as the way chosen goes: with its move among them she has gone
        # that way (planned through it still, from 9 after walking off 5 on the way through 12 from 19 to 3 she
        # jumped back onto 5 to walk off it again), and the platforms they set off from are behind her.
        self._gone_by(st, [leg.move for leg in running.legs])
        path, _ = self._route(st, p.id, x, dst)
        if not path:
            return None
        here = replace(ahead.scene, where=replace(ahead.scene.where, x=x, y=p.y_at(x)))
        if self.fight and self.escort.in_way(here, path[0].x) is not None:
            return _no_follow("a monster in the way of the next move")
        # It starts when the running plan ends, or at once if that's over: the monsters are predicted for then.
        start = max(starts, time.monotonic() + PLAN_START_S)
        self._immunity(start)  # (no touch counts on its legs while her invulnerability lasts: IMMUNE)
        tracks = [replace(m, seen_at=m.seen_at - (start - starts)) for m in ahead.tracks]
        price, alts, goal = self._pricing(st, tracks)
        legs, keys, why = self.flow.chain(path, Body(0.0, x, p, end.run, False, QUEUE_SLACK), tracks,
                                          keep_going=True, hops=not self.fight, price=price, alts=alts, goal=goal,
                                          spread=QUEUE_SPREAD)
        if not legs or (st.through is not None and not keys.sorted()):
            return None  # (a monster whose touch counts in the way of the next move: the next pass meets it, _meet)
        # With an escort, chains keep clear only of the monsters whose touch counts (it fights the others instead, at
        # the stops). One that follows on skips a stop: only where none of the monsters seen can get near it. (Queued
        # regardless, misses went up by half in the simulator's hunt-mode trips: touches knock her off her jumps.)
        watch = [replace(m, seen_at=m.seen_at - (start - starts)) for m in ahead.watch]
        if watch and any(self.flow.danger(leg, watch) for leg in legs):
            return _no_follow("a monster could get near the next chain")
        if time.monotonic() >= starts:  # the running plan is over: she's sliding to a stop
            return _no_follow(f"planned too late ({time.monotonic() - starts:+.3f} s from the plan's end)")
        telemetry.emit("event", kind="follow", text="followed on")
        st.path = path
        tail = self._tail(legs, path, watch if self.fight else tracks) if why == "long" else None
        if why == "long" and tail is None:
            keys.set_arrow(legs[-1].end.t, 0)  # nothing will follow it: she stops where it ends
        self.log(f"    then from platform {p.id} at x {x:.0f} (planned on the move): "
                 + _chain_text(legs, len(path), why))
        if telemetry.recording():
            telemetry.emit("route", **_route_record(st, dst, path, p.id, x))
        plan_id, length = send(legs, keys, why, tail or 0.0)
        return _Running(plan_id, legs, start, tail is not None, tail or 0.0, length,
                        tuple(watch if self.fight else tracks), arrows=self._arrows(keys.sorted()))

    def _landed_on(self, p: Platform, until: float) -> Scene | None:
        """Looks until one places her standing on platform `p` (now and then she lands later than predicted,
        or is still climbing over a rope's top), or until `until` (time.monotonic) has passed: that look, or
        None. Standing: ground under her feet on screen (in the air there's none: the minimap dot alone put
        her on a platform she was still falling past), or her name tag hidden and the dot on `p`."""
        while True:
            s = self._checked = self.senses.look(mobs=False)
            if s.where is not None and s.where.platform is not None and s.where.platform.id == p.id and standing(s):
                return s
            if time.monotonic() >= until:
                return None


def rope_at_top(model: MapModel, where) -> Rope | None:
    """The rope hanging from the platform `where` has her on (route.hung_from) within localization.pose.ROPE_TOP_X of
    her x (hanging at the rope at x 111 her name tag placed her at 113.3, her dot alone at 101.6), if any."""
    p = where.platform
    if p is None:
        return None
    near = [r for r in model.ropes if abs(r.x - where.x) <= ROPE_TOP_X and route.hung_from(model, r) is p]
    return min(near, key=lambda r: abs(r.x - where.x)) if near else None


def fresh_touch(senses, s: Scene, slack: float = 0.0) -> dict | None:
    """The touch of hers R7's stand-in goes by (Navigator._toucher; the fights' too, behavior.fight.clawed): `senses`'s
    newest (Senses.touch), not `crowded`, its popup TOUCHER_S before look `s` at most, her placed on a platform; else
    None. `slack`: s its popup may read after the look's frame (the fights': a popup's `at` is rounded to the ms,
    perception.touches.Touch.data, so one told in the look's own frame can read up to 0.5 ms after it; R7's own 0)."""
    touch = getattr(senses, "touch", None)
    w = s.where
    if not isinstance(touch, dict) or touch.get("crowded") or w is None or w.platform is None \
            or not -slack <= s.t - touch.get("at", -1e18) <= TOUCHER_S:
        return None
    return touch


def standing(s: Scene) -> bool:
    """Whether look `s` shows her standing, as far as it tells: ground under her feet on screen (in the air
    there's none: the minimap dot alone puts her on a platform she's falling past or jumping over), unless her
    name tag is hidden or no ground was read."""
    return not s.tag_seen or s.ground is None or s.ground[1] - s.ground[0] >= s.geo.fpx(MIN_GROUND_PX)


def _touch_text(touch: dict | None, hp: float, now: float) -> str:
    """Why a chain was cut short (or a move ended elsewhere) for her being touched: the touch told (Senses.touch: MISS,
    the damage's number) and her HP from `hp` to `now`; or with none told, her HP dropping alone."""
    if touch is None:
        return f"{TOUCHED} (HP {hp:.0%} -> {now:.0%})"
    kind = touch.get("kind")
    what = "MISS" if kind == MISS else (f"damage {touch['text']}" if touch.get("text") else str(kind))
    return f"{TOUCHED} ({what}: HP {hp:.0%} -> {now:.0%})"


def _no_follow(why: str) -> None:
    """Note (telemetry) why a chain cut off for length isn't followed on; None, for the caller to return."""
    telemetry.emit("event", kind="follow", text=why)
    return None


def _key(m: Move) -> tuple:
    """A move's identity across replans."""
    return (m.kind, m.src, m.dst, round(m.x))


def _goes_back(way: route.Way, best: route.Way, src: int, left: set) -> bool:
    """Whether a trip's chosen way, replanned from platform src (Planner.through), takes her back: onto a platform
    twice (src included), or onto one she has set off from since the trip chose (`left`) that the quickest from there
    (`best`) doesn't go onto (knocked back below it, the quickest goes over it again too)."""
    onto = [m.dst for m in way.moves]
    quickest = {m.dst for m in best.moves}
    return len(set(onto) | {src}) < len(onto) + 1 or any(p in left and p not in quickest for p in onto)


def _route_record(st: TripState, dst: int, path: list[Move], src: int, x: float, offered: tuple | None = None) -> dict:
    """Trip `st`'s route `path` to `dst`, planned from platform src at x, as telemetry ('route'), with what its
    plan weighed besides the moves' times: the platforms it keeps off and the moves a monster was in the way of; and
    the ways it chose among (`offered`: Navigator._route's), if it did."""
    return route_record(dst, path, start=(src, x), keep_off=st.keep_off - {dst}, avoid=sorted(st.avoid),
                        ways=offered)


def _arrows_in(legs: list[Leg]) -> bool:
    """Whether a chain's `legs` have her move at all (a 'walk' leg with its goal where she stands has no arrow in its keys)."""
    return any(leg.move.kind != "walk" or abs(leg.end.x - leg.start.x) >= 1.0 or leg.start.run for leg in legs)


def _ran_over(legs: list[Leg], cut: tuple | None = None) -> list[tuple[int, float, float]]:
    """The stretches of ground a chain's `legs` ran her over: (platform, x from, x to), pick-up pressed all the way; `cut`, (the
    running plan's legs, plan time): those only as far as then."""
    out = []
    for leg in legs:
        until = cut[1] if cut is not None and any(leg is lg for lg in cut[0]) else math.inf
        out += [(sp.platform, sp.x0, sp.x_at(min(until, sp.t1))) for sp in leg.spans if sp.t0 < until]
    return out


def _trip_end(st: TripState) -> dict:
    """Where trip `st` left her, for its 'trip' record (from 2026-11-02): the platform (`on`) and x of its last look
    (her arrival walk's own, Mover.walk_to's, if it walked on to x), and that walk (`walk`: [units asked, hold ms] each
    of its walks). A trip that said 'arrived' while she was on another platform, off the end of the one it was going to
    after its arrival walk, is told by `on` against `dst` (15 of the 210 'arrived' trips in the hour of 2026-11-02:
    P02 of the recordings' analysis; analysis.walkfit counts them)."""
    s = st.walked_to if st.walked_to is not None else st.last
    w = None if s is None else s.where
    out = {} if w is None else {"on": None if w.platform is None else w.platform.id, "x": round(w.x, 1)}
    if st.walked:
        out["walk"] = st.walked
    return out


def _platforms(path: list[Move]) -> set[int]:
    """The platforms the first moves of a route (as many as a chain takes: flow.MAX_LEGS) go from and onto."""
    return {p for m in path[:MAX_LEGS] for p in (m.src, m.dst)}


def _after(running: "_Running", path: list[Move]) -> list[Move]:
    """The moves of `path` (the route the running chain was planned from) after the chain's: hops are the chain's own,
    not the route's."""
    return path[sum(1 for leg in running.legs if leg.move.kind != "hop"):]


def _arrow_at(arrows: tuple, t: float) -> int:
    """The arrow a plan's keys hold at its plan time t (s): -1 left, +1 right, 0 none (_Running.arrows; both held, the
    one pressed last)."""
    held: dict[int, int] = {}
    for k, (ms_, d, down) in enumerate(arrows):
        if ms_ / 1000 > t + 1e-9:
            break
        if down:
            held[d] = k
        else:
            held.pop(d, None)
    return max(held, key=held.get) if held else 0


def _same_moves(a: list[Move], b: list[Move]) -> bool:
    """Whether two routes go the same way as far as both go (each move's kind, from and onto: not where it takes off)."""
    return all((m.kind, m.src, m.dst) == (n.kind, n.src, n.dst) for m, n in zip(a, b))


def _blocked(hit: tuple[MonsterTrack, float]) -> str:
    """What blocks the way up a rope (Navigator._way_up_hit's monster), in words: "a horny-mushroom on 10 by its top"."""
    return f"a {hit[0].kind or 'monster'} on {hit[0].platform} by its top"


def _may_set_off(tr) -> bool:
    """Whether a monster the feed follows standing (tracking.Track: vx under STILL_VX) may set off any time: alive, its
    last detection a stand or flinch frame (monsters.walk_vx 0). Not one dying (the tracker's DYING: a kill of hers on
    it, or its death seen; a corpse never walks, and planned so the landing by one dying of her star became a move in
    the way of her jump, shot at again and again with nothing to shoot: nav_tree_landing_shot_first, its review's fix),
    nor one whose pose the frame didn't tell (its vx kept from before)."""
    return getattr(tr, "state", LIVE) in (LIVE, HIDDEN) and getattr(tr, "pose", "stand") in ("stand", "hit1")


def _as_planned(mo: MonsterTrack, tr) -> bool:
    """Whether the feed's track `tr` (tracking.Track) walks as a plan's monster `mo` had it: planned either way (its
    heading not known: standing, as perception had it, or seen once), so far as it hasn't set off since (one planned
    standing, `unsure`, that walks now hasn't: STAND_UNSURE_S); planned walking (vx), the same way still, not standing
    (one stopped is where the plan has it gone by)."""
    walks = abs(getattr(tr, "vx", 0.0) or 0.0) > STILL_VX
    if mo.vx is None:
        return not (walks and mo.unsure is not None)
    if not mo.vx:
        return not walks
    return walks and (tr.vx > 0) == (mo.vx > 0)


def _ground_from(leg: Leg) -> float:
    """When (s into its plan) she's back on the ground for good on the platform a leg ends on: where her
    last run along it starts, or where the leg ends (a landing, or the top of a climb)."""
    sp = leg.spans[-1] if leg.spans else None
    if sp is not None and sp.platform == leg.end.platform.id and sp.t1 >= leg.end.t - 1e-6:
        return sp.t0
    return leg.end.t


def _still(dot, last) -> bool:
    """Whether her minimap dot is where it was (within STILL on both axes of `last`: None if there was none)."""
    return last is not None and abs(dot[0] - last[0]) < STILL and abs(dot[1] - last[1]) < STILL


def _stood(stood: tuple[float, tuple] | None, t: float, dot) -> tuple[float, tuple] | None:
    """Since when her dot has stood where it is, and where it stood then, once a glance at `t` (its frame's time) saw
    it at `dot` (None: hidden), from `stood` as the glances before had it (None: none, or the last didn't see it):
    within STILL of where it stood, it stands there still; else it stands here from now."""
    if dot is None:
        return None
    return stood if stood is not None and _still(dot, stood[1]) else (t, dot)


def _chain_text(legs, to_go: int, why: str) -> str:
    """A chain's moves for the log."""
    return (", ".join(f"{leg.move.kind}->{leg.move.dst}" if leg.move.kind != "hop" else f"hop@{leg.move.x:.0f}"
                      for leg in legs) + f"  [{to_go} moves to go; chain ended: {why}]")
