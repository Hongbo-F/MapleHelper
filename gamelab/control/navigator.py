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
waits for the monster to move off (a few looks), and then goes anyway. A hit during a move shows as lost HP:
she replans from wherever she was pushed to, without counting it as a missed move.

With an escort (hunting tours: behavior.fight.FightOnTheWay), each trip has a mode (go(mode=...)):
`hunt` (the default) shoots what's in reach before each chain and any monster in the way of a move along
her platform; `transit` (the hunt loop's trips) only shoots what's in the way. Either way her chains keep clear of
the monsters whose touch counts (the escort's `threat`: on Tree Dungeon the mushrooms, on the first map none) as they
do without an escort: a chain stops short of a leg one could touch her on, and she shoots it first if a star reaches
it from where she stands (the escort's `clear_for`), else the move costs extra so a way around is taken, else she
hops it (one in the way of a run along her platform), else waits for it to move off, then goes anyway. The monsters
for that are searched where the chain's legs go
(`trip_monsters`: her fights look at the band around her), and with her memory of the monsters (`memory`: a hunter's)
the ones it has there that the look couldn't have found, last seen off its view, are kept clear of too (`remembered`).
Without an escort the mode is `avoid`, as described above.

With a continuous feed (the hunting commands' default) her glances while a chain runs carry the monsters it follows: one
the chain's legs ahead of her now meet, that the look it was planned from hadn't, stops the chain as a hit does while she
runs along a platform with room to stop (REACT), and the next pass meets it.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import NamedTuple, Protocol

from .. import telemetry
from ..localization.mapdata import MapModel, Platform, Rope
from ..localization.memory import EDGE, MATCH_SLACK, MATCH_SPEED, MATCH_WALK_S, SURE
from ..localization.monsters import (DEFAULT_SPEED, MONSTER_HEIGHT, Monster, MonsterTrack, Tracker, bodies, touch_box,
                                     tracks_record)
from ..localization.pose import ROPE_TOP_X
from ..localization.scene import Scene
from ..localization.senses import Senses
from ..localization.tracking import HIDDEN, HIDE_S, LIVE
from ..localization.unseen import hidden
from ..motion import bodies as her_body
from ..motion import route
from ..motion.flow import END_LINGER, MAX_LEGS, ROPE_SPEED, UP_EXTRA, Body, Flow, Leg, Span, chain_record, ms
from ..motion.physics import JUMP_HEIGHT, WALK_SPEED
from ..perception.framecheck import LookLost
from ..perception.kinds import kind_of
from ..motion.route import Move, Planner, keeping_off, route_record
from . import client
from .actions import Actions
from .mover import Mover, platform_text

ON_WAY_TRIES = 5    # monsters engaged per platform while passing through (respawns could keep her forever)
MAX_FIGHTS = 3      # fights in a row clearing the way, without a move between: then she moves on regardless
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
HIT = 0.004          # HP lost (fraction) that counts as a hit
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
# Looks at most, each after waiting up to 2 s for her to land and stand still (Mover.settle), for one that puts
# her on a platform, when a look shows her in the air: a slime's touch as she walked off platform 3 onto 4
# knocked her back into the 12-unit gap between the blocks at x 669-681, down which she fell to the basin for
# 2.5 s, and with one look the trip ended "lost: not on any platform" while she still fell (simulator,
# nav_monsters_long_way on seeds 2 and 4).
FALL_LOOKS = 2
# Where a fighting trip's monsters are searched (Navigator.trip_monsters): as high above her feet as the chain's
# highest platform, a jump over it (JUMP_HEIGHT), her body (motion.bodies.HER_TALL) and the map's tallest monster
# reach, and as low as its lowest platform, the map's tallest monster (one on a platform under it whose head reaches
# her feet there) and this much more (a monster's feet a little under the surface by its sprite: the detector finds
# only whole ones). Without the tallest monster below, on 20 of 472 random chains on Tree Dungeon a Green Mushroom
# standing on the step under the chain's lowest platform, 24-56 below it, could touch her and wasn't searched for
# (review, 2026-10-24: from 20 at x -78 walking off onto 21, one on 22 at x -64 reached her 1.8 s on).
TRIP_LOOK_BELOW = 15
SAME_SIGHTING = 5  # map units: a monster found again on the same frame within this of one found before is it
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
# the detector's misses near her as live (sim.world.NEAR_SEEN), these and the hunt loop's fights by what the looks
# missed (behavior.hunt.UNSEEN_FIGHTS) made 297.7 EXP a minute against 270.9 without, lost 0.92 bars of HP a minute
# against 1.04, and her trips were touched 2.38 times a minute against 3.25, their plans cut by a hit or a monster coming
# 8.9 times a run against 27.8 (in the simulator: Tree Dungeon, crowd, zero-walls, near-misses, `--attack choice`, 8
# seeds of 5 minutes; D:/MapleHelper-scratch/hunt-ab/B/ablate-a1-table.txt).
TRIP_MEMORY_S = 2.0
UNSEEN_TRIPS = True  # those in view, the feed's hidden tracks and the cut's track (above); off, as before
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
STILL_VX = 15.0      # units/s: a followed monster slower than this stands (tracking's velocities, filtered frame to frame)
REACT_ROOM = 50.0    # units of platform ahead of her run she needs to stop on (her 149 units/s for AFTER_STILL_S's
                     # slide: 39, and a little)
COMING = "a monster coming into its way"  # (the cut's reason starts so: Navigator._watch, _carry_out_chain)
# Landings with clearance (from 2026-10-27; the user: "the character should actively try to move away from the mob to
# attack it"). With an escort that prices landings (Escort.landings: attack.stance.landing, what her first attack where a
# leg lands surely takes, the monsters walked on to then), a chain's moves are done where they land the cheapest
# (flow.chain's `price`: their other takeoffs, the route's other moves onto the same platform, the walk on to where she's
# sent counted; a landing she runs on from by its touch alone), and where a trip's last landing still has a price the
# monster making it so is shot first from where she stands, else from along her platform within SHOOT_WALK
# (LANDING_WALK), only where her attack reaches it (Escort.shot_place: from 2026-10-28, when live from 21 one on 23, past
# a star's band, was said to be shot first and nothing was thrown), else waited out if its walk clears the landing within
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
# A chain planned on the move (_queue_next) prices its moves' landings by their own ways and the route's other moves
# between the same platforms only, not wider takeoffs (flow.chain's `spread`): it must be planned before the running
# plan ends (SEND_S), and with them a priced chain took a median 5.7 ms more, 20 at the most, 48 with the suite running
# alongside (the reviews of 2026-10-27, review-behaviour/chaintime.py), against SEND_S's 30; without them 0.7 ms, 3 at
# p90, 3.9 at the most, with the harness running alongside (D:/MapleHelper-scratch/land/fix/chaintime.py, SPREAD=0, where
# with them 10, 45 and 62). True: wider takeoffs too.
QUEUE_SPREAD = False

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

    def clear_for(self, s: Scene, m: Monster, finish: bool = True) -> int | None:
        """Shoot monster `m`, one that could touch her on the chain ahead, if a star reaches it from where she
        stands: the kills, or None if none does. `finish`: as clear_way's."""

    def shoot_from(self, s: Scene, m: Monster, most: float, finish: bool = True) -> int | None:
        """Walk along her platform, `most` units at most, to the nearest place a star reaches monster `m` from, and
        shoot it there: the kills, or None if there's no such place."""

    def shot_place(self, s: Scene, m: Monster, most: float) -> tuple[float | None, str]:
        """Where her attack reaches monster `m` from (the fights' own reach): (her x, '') from where she stands
        (clear_for's), (a place within `most` along her platform, '') shoot_from's, else (None, why not). Nothing sent."""

    def landings(self, tracks: list) -> Callable[[Leg], tuple[float, MonsterTrack | None]] | None:
        """How a leg's landing is priced against `tracks` (the chain's): the price (s: 0 where her first attack there
        surely takes the nearest, nobody walking into her first), and the track making it so; None: not priced."""


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
    time: dict = field(default_factory=lambda: {"look": 0.0, "think": 0.0, "move": 0.0})  # s spent


class _Running(NamedTuple):
    """A chain's key plan, sent to the player (running, or queued to start when the one before ends)."""
    plan_id: str
    legs: list[Leg]
    start: float  # when it starts (time.monotonic)
    more: bool    # another chain may follow on from where it ends
    tail: float = 0.0  # s its plan runs on past its last leg (time to place her before the next one)
    length: float | None = None  # s its plan lasts, as sent (its last key can come after its last leg)
    known: tuple = ()  # the monsters the look it was planned from had (MonsterTrack, in its plan's time): REACT

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
    quick: bool = False  # just cut short: replan from a fast look with the monsters already seen
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
    stopped: bool = False  # the last chain was cut short for a monster coming into its way (REACT): not a miss
    came: object | None = None  # ... the feed's track of that monster (tracking.Track), kept for the next chain
    thought_to: float = 0.0  # up to when thinking is counted (Trip.time: _chain, called again for a pass's replans)
    land_waits: int = 0  # looks spent in a row waiting for a monster to clear the chain's last landing (LAND_WAITS)

    def end(self, result: str) -> Trip:
        """The trip, over: `result` says how it ended."""
        self.trip.result, self.over = result, True
        return self.trip

    def paused(self, seconds: float) -> None:
        """She stood still `seconds` (LookLost: Senses.regain): the trip's clocks move on by as much, its deadline and
        how long its chains have stopped short of a monster (MEET_S), and a look taken before isn't used, nor the
        monsters seen before: the next pass looks for them afresh even if a cut-short chain had it take a quick look
        (`quick`, which plans from the tracker's last monsters: those from before the pause, none that came during it,
        and with an escort no fight for that chain; review, 2026-10-24). Where moves were in a monster's way (`avoid`)
        ages as the game's own clocks do."""
        self.deadline += seconds
        if self.blocked is not None:
            self.blocked += seconds
        self.fresh, self.quick, self.sighted = None, False, []


class Navigator:
    """Gets her onto a platform: plans the route, chains its moves, and replans from wherever she ends up."""

    def __init__(self, senses: Senses, actions: Actions, model: MapModel, *, log=print,
                 health: Callable[[Scene], None] | None = None, escort: Escort | None = None,
                 variety: float | None = None, rng: random.Random | None = None, memory=None):
        """`health(scene)`, asked at every pass's look: her potions, and the safety stop, which it raises (a hunter's
        health_check: behavior.safety); `escort` fights on the way (without one she avoids monsters instead).
        `variety`: the moves' random weights (route.NOISE by default; 0 for none). `memory`: her memory of the
        monsters (localization.memory.Memory, a hunter's), whose monsters where a fighting trip's chain goes it keeps
        clear of too (`remembered`); None, only the looks'."""
        self.senses, self.actions, self.model, self.log = senses, actions, model, log
        self.memory = memory
        self.health = health or (lambda s: None)
        self.escort = escort
        self.mode = HUNT if escort is not None else AVOID  # a trip's mode unless go() says otherwise
        self._mode = self.mode                             # the running trip's
        self.planner = Planner(model)
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
        names = {k: getattr(actions.keys, k).name for k in ("left", "right", "jump", "up", "down")}
        if actions.keys.pick_up is not None:
            names["pick"] = actions.keys.pick_up.name  # pressed all through her runs: loot on the way
        self.flow = Flow(model, names, rng=self.rng if self.variety else None)
        self.tracker = Tracker(model)  # which way the monsters walk, from look to look
        self._checked: Scene | None = None  # follow: the last look placing her for the next chain
        self._threat_kinds: frozenset | None = None  # the map's kinds whose touch counts (_threats), once worked out
        self._tallest_box: float | None = None       # the map's tallest monster box (_tallest), once worked out
        self._siblings: dict[tuple[int, int], list[Move]] | None = None  # (src, dst) -> the route's moves (_pricing)
        self._reacted: dict[int, float] = {}  # the feed's track id -> when a plan was last cut for it (REACT)
        self._came = None  # the feed's track of the monster the last cut was for (REACT: _coming), for TripState.came
        self._tails: dict[str, float] = {}    # a chain's plan id -> s it runs on past its last key (_ended)
        # The last chain's glances, if her dot stood where the first had it all along: that dot and for how long
        # (None: it moved, or no glance had it): _rope_top
        self.chain_still: tuple[tuple[float, float], float] | None = None

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
           scene: Scene | None = None, keep_off=(), vary: bool = True, stand_until: float | None = None) -> Trip:
        """Get onto platform `dst` (and to map x on it, if given). `until` is asked before every chain,
        with a look that includes monsters (late in the chain before, when it follows on from that one):
        True ends the trip early ('in position'). `mode` (AVOID, HUNT or TRANSIT; the navigator's own by
        default) says how the trip treats monsters. `scene`: the caller's last look, used for the first
        pass instead of a new one if it's fresh (no keys sent since). `keep_off`: platforms its routes go onto
        only at a cost (TripState.keep_off). `vary`: it chooses among the planner's ways (route.ROUTES; TripState.
        vary). While its looks fail (LookLost) it stands still up to its deadline, or `stand_until` (time.monotonic) if
        that's sooner: a retreat's window (behavior.retreat)."""
        st = self.begin(dst, x, seconds, max_misses, until, mode=mode, scene=scene, keep_off=keep_off, vary=vary)
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
              scene: Scene | None = None, keep_off=(), vary: bool = True) -> TripState:
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
                       keep_off=frozenset(keep_off), vary=vary)
        # Her last look will do for the first pass if nothing has moved her since: one look less before she
        # sets off. Only a look with the monsters the mode looks for (whole screen when avoiding them), and
        # one that shows her standing (_onto_platform).
        if scene is not None and scene.searched and (mode in FIGHTING or scene.wide) and self.fresh(scene) \
                and standing(scene):
            st.fresh = scene
        return st

    def fresh(self, s: Scene) -> bool:
        """Whether look `s` still shows where she is: taken FRESH_S ago at most, with no key plan sent since."""
        return s.where is not None and s.t >= self.actions.sent_at and time.monotonic() - s.t <= FRESH_S

    def step(self, st: TripState) -> Trip | None:
        """One pass of trip `st`: a look, then what it calls for (stop, get off a rope, shoot what's in
        reach or in the way, arrive, wait for a monster, or carry out a chain of moves or one move).
        Returns the trip once it's over (and sends its 'trip' record), else None."""
        if st.over:
            return st.trip
        self._mode, self._weights = st.mode, st.weights
        try:
            trip = self._pass(st)
        finally:
            self._mode = self.mode
        if trip is not None:
            telemetry.emit("trip", dst=st.dst, result=trip.result)
        return trip

    def _pass(self, st: TripState) -> Trip | None:
        if not time.monotonic() < st.deadline:
            return st.end("timeout")
        # The Lie Detector seen: the trip stops here, keys or not (behavior.retreat); a retreat's own trip goes on (its
        # plans are the kind the player takes then).
        if getattr(self.actions, "kind", None) is None:
            client.check_alarm()
        # Look, and deal with what's here.
        s, looked = self._look(st)
        st.last = s
        st.health(s)  # her potions, and the safety stop (a hunter's health_check raises it: behavior.safety)
        if st.until is not None and s.where.platform is not None and st.until(s):
            return st.end("in position")
        s = self._onto_platform(s)
        if s.where.platform is None:
            return st.end("lost: not on any platform")
        if s.where.platform.id == st.dst:
            # There: only a monster between her and the x she's sent to is shot first. Bursts at
            # whatever is in reach here (as on the way) took whole trips: a 60 s trip to drops on the
            # basin spent all of it shooting respawns there (tour_hunt_basin).
            if st.x is not None and self._clear_the_way(st, s, st.x):
                return None  # look again after the fight
            if st.x is not None:
                self.mover.walk_to(st.x, tolerance=ARRIVE_TOLERANCE, s=s)
            return st.end("arrived")
        if self._shoot_on_the_way(st, s):
            return None  # look again after the fight

        # Plan the route, and chain as many of its moves as can be predicted.
        path, planned_at = self._plan(st, s, st.dst)
        if not path:
            return st.end(f"no route from platform {s.where.platform.id} to {st.dst}")
        if self._clear_the_way(st, s, path[0].x):
            return None  # look again after the fight
        monsters, near = self._chain_monsters(st, s, path)
        st.quick = False
        legs, keys, why = self._chain(st, s, path, monsters, looked)
        if self.fight:
            met = self._meet(st, s, path, monsters, legs, why, planned_at, looked)
            if met is True:
                return None  # look again: after the shot, for a way around, or once she's waited a moment
            if met is not None:
                legs, keys, why = met  # a hop over it
            elif self._clear_landing(st, s, path, monsters, legs, why):
                return None  # look again: after the shot, or once she's waited a moment
        elif self._wait_for_monster(st, s, path[0], monsters, legs, why, planned_at):
            return None  # look again: for a way around, or once she's waited a moment

        # Move: the chain in one plan, or else the first move on its own.
        hp = s.hp
        if legs:
            # With an escort the chain keeps clear only of the monsters whose touch counts (on the first map none: it
            # fights them instead), but its run on past its end (follow's tail) mustn't take her into any.
            if near is None:
                near = self.tracker.update(s.t, s.monsters, seen_at=s.t - time.monotonic())
            st.path = path
            self.chain_still = None
            moved, planned, after = self._carry_out_chain(st, s, path, legs, keys, why, st.dst, near)
            still = self.chain_still is not None and self.chain_still[1] >= STUCK_S
        else:
            st.stopped = False
            moved, planned, after = self._carry_out_move(st, s, path)
            if not moved:
                return None  # one the mover can't do: another way (st.cannot)
            still = self.mover.still is True
        if still and ROPE_TOP:
            after = self._rope_top(st, s, after)
        self._made(st, moved, after.where.platform is not None and after.where.platform.id == planned.id)
        if self._note_landing(st.trip, moved, planned, after, hp, st.stopped):  # a miss
            st.trip.misses += 1
            if st.trip.misses >= st.max_misses:
                return st.end("too many misses")
        return None

    # --- the steps of a pass (_pass) ---------------------------------------------------------------

    def _look(self, st: TripState) -> tuple[Scene, float]:
        """A look at her: a quick one right after a chain was cut short (the monsters already seen will
        do), else one that includes the monsters (the whole screen when avoiding them), or the one taken
        for this pass already (st.fresh). Returns it and when it was over (thinking is counted from then)."""
        if st.fresh is not None and not st.quick:
            s, st.fresh = st.fresh, None
            return s, time.monotonic()
        st.fresh = None
        t0 = time.monotonic()
        s = self.mover.locate() if st.quick else self.mover.locate(mobs=True, wide=not self.fight)
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
        landings, 677 of 5,112 replans that kept a way chosen on the map went back onto a platform she had been on.)"""
        penalty = self._penalty(st)
        if route.ROUTES <= 1 or not st.vary:
            return self.planner.plan(src, x, dst, weight=self.weight, penalty=penalty), None
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

    def _stale(self, m: Move) -> bool:
        """Whether she hasn't set off on move m for route.STALE_S (this navigator's memory: since it began, if
        never)."""
        return time.monotonic() - self._used.get(_key(m), self._since) >= route.STALE_S

    def _made(self, st: TripState, moved: list[Move], arrived: bool = True) -> None:
        """Note the moves she set off on (route.STALE_S), and the trip's way (_gone_by); the fights and shots at a monster
        in the way start again from none (MAX_FIGHTS), and once they got her where they were going (`arrived`: not cut
        short by a touch, say), so does the time her chains have been stopping short of one (MEET_S)."""
        now = time.monotonic()
        for m in moved:
            self._used[_key(m)] = now
        self._gone_by(st, moved)
        st.fights, st.shots, st.land_waits = 0, 0, 0
        if arrived:
            st.blocked = None

    @staticmethod
    def _gone_by(st: TripState, moved: list[Move]) -> None:
        """The moves she set off on, for the way the trip chose: the platforms she set them off from are behind her
        (Choice.left); with its move among them, it has gone the way it chose, and goes on by the quickest."""
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
        if st.fights < MAX_FIGHTS and (kills := self.escort.clear_way(s, x, finish=st.mode != TRANSIT)) is not None:
            st.fights += 1
            st.trip.kills += kills
            return True
        if st.fights == MAX_FIGHTS:
            telemetry.emit("event", kind="backstop", text="MAX_FIGHTS: moving on past the monster in the way")
            st.fights += 1  # (once: no more fights in her way till she has moved, _made)
        elif st.fights < MAX_FIGHTS:
            st.fights = 0  # nothing in her way
        return False

    def _chain_monsters(self, st: TripState, s: Scene, path: list[Move]) -> tuple[list, list | None]:
        """The monsters the chain keeps clear of, and those its run on past its end mustn't take her into (follow's
        tail; None: not tracked here, so the caller tracks the look's if it needs them). After a quick look, the last
        look's, moved on by the time since; else this look's, tracked from look to look (which way they walk). Without
        an escort, every one of them both ways. With an escort, those of the kinds whose touch counts (Escort.threat),
        searched for where the chain's legs go (trip_monsters), and every one of them for the tail; on a map where no
        kind counts (the first map), none, as before: the chain is planned blind (it fights them instead); its tail and
        whether the next chain follows on (_queue_next) are judged against every one, by her box (motion.bodies)."""
        if self.fight and not self._threats():
            return [], None
        platforms = _platforms(path)
        if st.quick:  # the last look's monsters, moved on by the time since
            looked, seen_then = self.tracker.last
            tracks = [replace(mo, seen_at=looked - time.monotonic()) for mo in seen_then]
            st.sighted = []
        else:
            seen = self.trip_monsters(s, platforms) if self.fight else s.monsters
            tracks = self.tracker.update(s.t, seen, seen_at=s.t - time.monotonic())
            st.sighted = list(zip(tracks, [m for m in seen if m.platform is not None]))  # (as the tracker keeps them)
            if self.fight:  # and those her memory has there that the look didn't find (not into the tracker's)
                tracks = tracks + self.remembered(s, platforms, seen, s.t - time.monotonic())
        if self.fight:  # and the feed's out of sight, and the one the last cut was for (not into the tracker's)
            tracks = tracks + self.hidden_tracks(s, platforms, tracks, st.came)
        st.came = None
        if not st.quick and telemetry.recording():
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

    def trip_monsters(self, s: Scene, platforms) -> list[Monster]:
        """The monsters seen by look `s` that a trip with an escort keeps clear of (those whose touch counts: its
        chains stop short of a leg one of them could touch her on), where its chain's legs go (`platforms`: the
        platforms of the route's next moves): the look's own if it searched the whole screen, or its band (Senses.band)
        takes in from as high as she gets over them (the highest, a jump over it, her body and the map's tallest
        monster: TRIP_LOOK_BELOW) to as low as the lowest and the tallest monster under it; else its frame searched
        again over all of that (Eyes.monsters: a band as tall, in game about a band's time more, 23 ms at the game's own
        1366x768), with the look's own kept too (a feed's frame may be gone by then: Feed.frame_of gives a blank one). A
        fight looks at the band around her (her stars reach 60 up and down), so most of a trip's looks are band looks
        (`_look`). What her memory has there off the look's view comes on top (`remembered`)."""
        if s.wide or not s.searched or s.where is None or s.tag is None:
            return list(s.monsters)
        ps = [self.model.platform(p) for p in platforms]
        if not ps:
            return list(s.monsters)
        y, tallest = s.where.y, self._tallest()
        up = max(y - p.top for p in ps) + JUMP_HEIGHT + her_body.HER_TALL + tallest
        down = max(her_body.low_end(p) - y for p in ps) + tallest + TRIP_LOOK_BELOW
        band_up, band_down = self.senses.band
        if up <= band_up and down <= band_down:
            return list(s.monsters)
        eyes = self.senses.eyes
        found = [s.place(m, self.model, eyes.bar(s, m)) for m in eyes.monsters(s, max(up, band_up), max(down, band_down))]
        kept = [m for m in s.monsters if not any(_same_sighting(m, f) for f in found)]
        return kept + found

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

    def hidden_tracks(self, s: Scene, platforms, tracks: list[MonsterTrack], came=None) -> list[MonsterTrack]:
        """The monsters the feed follows out of sight on screen as of look `s` that count as about (localization.unseen.
        hidden: last seen within its TRACK_S, none that a kill of hers is on: her memory's `kills`, the ledger's) on
        `platforms`, and `came` (the feed's track of the monster the last chain was cut for, kept for this one:
        TripState.came; seen within tracking.HIDE_S) wherever the feed has it, as tracks walking as followed
        (_track_of), of the kinds the chain keeps clear of; none that one of `tracks` (the look's and her memory's, by
        kind and platform within what it can have walked since: memory.MATCH_SLACK) is."""
        out: list[MonsterTrack] = []
        if not UNSEEN_TRIPS:
            return out
        now = time.monotonic()
        kills = getattr(self.memory, "kills", None)
        cands = hidden(s, kills() if kills is not None else (), platforms)
        if came is not None and not any(tr.id == came.id for tr in cands) and 0 <= s.t - came.last_seen <= HIDE_S:
            cands.append(came)
        for tr in cands:
            if tr.platform is None or tr.platform not in platforms \
                    or (self.escort is not None and not self.escort.threat(tr.kind)):
                continue
            speed = self.tracker.walking.get(tr.kind, DEFAULT_SPEED)
            age = now - tr.last_seen
            slack = MATCH_SLACK + MATCH_SPEED * speed * min(age, MATCH_WALK_S)
            p = self.model.platform(tr.platform)
            if any(mo.kind == tr.kind and mo.platform == tr.platform and abs(mo.at(0.0, p) - tr.x) <= slack
                   for mo in tracks + out):
                continue
            out.append(self._track_of(tr, now))
        return out

    def _tallest(self) -> float:
        """The tallest box (for touches: localization.monsters.touch_box) of the map's monsters."""
        if self._tallest_box is None:
            body = bodies(self.model)
            names = {name for names in self.model.monsters_by_platform.values() for name in names}
            self._tallest_box = max((touch_box(kind_of(n), body.get(kind_of(n)))[1] for n in names), default=60.0)
        return self._tallest_box

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
        return ((lambda leg, last=True: judge(leg, last)[0]), (lambda m: self._siblings.get((m.src, m.dst), [])),
                None if st.x is None else (st.dst, st.x))

    def _clear_landing(self, st: TripState, s: Scene, path: list[Move], monsters: list, legs: list[Leg],
                       why: str) -> bool:
        """With an escort, a chain that ends the trip (onto its platform) landing with a price (Escort.landings: her first
        attack there not sure to take the nearest monster, or one walking into her first), the cheapest flow.chain
        found: the monster making it so is shot first where her attack reaches it (Escort.shot_place, the fights' own
        reach): from where she stands (Escort.clear_for), else from along her platform within SHOOT_WALK (Escort.
        shoot_from: "shooting it first from x N"), MAX_FIGHTS shots in a row at most as _meet's; where there's no such
        shot, the log says why ("no shot first, her attack reaches it from nowhere on platform 21 (136 under her)") and
        nothing is sent; then it's waited out if, walking as seen, it clears the landing within WAIT_AHEAD (LAND_WAITS
        looks at most, where none can reach her meanwhile); else she goes ("landing on P by the K anyway: why"). Each a
        'landing' event (shot first, waited, went), with why there was no shot (`shot`) and why she didn't wait (`why`).
        True to look again."""
        landings = getattr(self.escort, "landings", None)
        if not (LANDING_CLEAR and legs and why == "done" and landings is not None and monsters) \
                or legs[-1].end.platform.id != st.dst:
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
            shot = "not in this look"  # (a quick look's, or a track the look didn't have: nothing to aim at)
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
                kills = self.escort.shoot_from(s, seen, SHOOT_WALK, finish=finish) if walk \
                    else self.escort.clear_for(s, seen, finish=finish)
                if kills is not None:
                    st.shots += 1
                    st.trip.kills += kills
                    telemetry.emit("event", kind="landing", text=f"shot first: {seen.kind} on {q.id} by the landing"
                                   f"{where}", price=round(price, 2), at=round(place, 1))
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
            return True if self._shoot_ahead(st, s, path, monsters, legs) else None
        move, start, linger = path[0], self._start(st, s), END_LINGER if len(path) == 1 else 0.0
        hit = self.flow.in_the_way(move, start, monsters, linger)
        if hit is None and not linger:  # the move is clear, standing where it ends isn't (flow.chain ends short of it)
            linger = END_LINGER
            hit = self.flow.in_the_way(move, start, monsters, linger)
        mo = hit[0] if hit is not None else None
        kind = mo.kind if mo is not None and mo.kind else "monster"
        seen = next((m for t, m in st.sighted if t is mo), None)  # as the look saw it (none after a quick look)
        for step in ("shoot", "around"):
            if step == "shoot" and seen is not None and st.shots < MAX_FIGHTS:
                kills = self.escort.clear_for(s, seen, finish=st.mode != TRANSIT)
                if kills is not None:
                    st.shots += 1
                    st.trip.kills += kills
                    return True
            if step == "around" and _key(move) not in st.avoid:
                self.log(f"  a {kind} could touch her on {move.kind} -> {move.dst}: looking for a way around")
                self._around(st, s, move, monsters, planned_at)
                return True
        hopped = self._chain(st, s, path, monsters, looked, hops=True)
        if hopped[0]:
            return hopped
        if st.waited < WAIT_LOOKS and self._safe_to_stand(s.where.x, s.where.platform, move, monsters) \
                and self._worth_waiting(move, start, monsters, linger, mo):
            st.waited += 1
            st.trip.waits += 1
            self.log(f"  waiting for the {kind} to move off ({st.waited})")
            time.sleep(WAIT_S)
            return True
        if LAST_SHOT and mo is not None and self._last_shot(st, s, mo, planned_at):
            return True  # look again after the shot
        return self._regardless(st, s, path, monsters, looked, mo,
                                f"WAIT_LOOKS: {move.kind} -> {move.dst} past a {kind} regardless" if st.waited >= WAIT_LOOKS
                                else f"NO_WAIT: {move.kind} -> {move.dst} past a {kind} regardless")

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
        none: after a cut, a quick look's `st.sighted` is empty, and one her memory or the feed's hidden tracks have is in
        none), and if it has `mo` (one of its kind on its platform where it can be by now, MATCH_SLACK on: MonsterTrack.
        reach) and a star from where she stands reaches it, she shoots it (Escort.clear_for; MAX_FIGHTS shots in a row
        at most, as _meet's). True if she did. Live on 2026-10-25 28 passes regardless (NO_WAIT, WAIT_LOOKS) were
        followed by 4 touches within 3 s, two into a Green Mushroom she had seen (24's, from 25: 23:49:42.7, 23:50:05.7;
        D:/MapleHelper-scratch/analysis-1025/damage-skeptic): the shot needed it in the pass's own sightings."""
        if st.shots >= MAX_FIGHTS or self.escort is None or mo.platform is None or not mo.kind:
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
                    mo: MonsterTrack | None, why: str, blind: bool = False):
        """_meet's last resort: the chain planned as if monster `mo` weren't there (the others still kept clear of), or,
        `blind` (MEET_S), as before her chains kept clear of anything; None (the move on its own) if that chains nothing
        either. A backstop event says `why`. (Past MEET_S, kept clear of the others, the chain stopped short of the next
        one, where it could still touch her: up from Tree Dungeon's floor (33) she jumped onto the ledge 32 beside a
        Green Mushroom, short of the rope to 30 another walked the top of, was knocked back down, and did the same
        again for the 180 s of a trip to rest: simulator, loop_tree_run seed 4. Blind on every backstop, though, her
        hunt loop's trips were touched 14% more often than with only `mo` left out: harness, 20 seeds.)"""
        telemetry.emit("event", kind="backstop", text=why)
        st.waited = 0
        regardless = self._chain(st, s, path, [] if blind else [t for t in monsters if t is not mo], looked,
                                 priced=monsters)
        return regardless if regardless[0] else None

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

    def _wait_for_monster(self, st: TripState, s: Scene, move: Move, monsters: list, legs: list[Leg],
                          why: str, planned_at: float) -> bool:
        """A monster in the way of the first move (nothing chained because of it): look for a way around
        (the move costs extra from `planned_at` on), else wait for it to go, WAIT_LOOKS looks at most.
        Waiting only makes sense where no monster can reach her meanwhile. True to look again."""
        safe_here = self._safe_to_stand(s.where.x, s.where.platform, move, monsters)
        if not legs and why == "monster" and st.waited < WAIT_LOOKS and safe_here:
            if _key(move) not in st.avoid:
                st.avoid[_key(move)] = planned_at
                self.log(f"  a monster could get in the way of {move.kind} -> {move.dst}: looking for a way around")
                return True
            st.waited += 1
            st.trip.waits += 1
            self.log(f"  waiting for the monster to move off ({st.waited})")
            time.sleep(WAIT_S)
            return True
        st.waited = 0
        return False

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
                         dst: int, monsters=()) -> tuple[list[Move], Platform, Scene]:
        """The chain in one plan, followed on by those planned on the move (follow). Returns the moves
        made, the platform she should have ended on, and a look once it's over. Cut short, she replans
        from a quick look."""
        self.log(f"  {platform_text(s.where.platform)} at x {s.where.x:.0f} ({s.how}): "
                 + _chain_text(legs, len(path), why))
        t2 = time.monotonic()
        cut, legs = self.follow(legs, keys, s.hp, dst, why, st.trip, st, path, monsters)
        st.trip.time["move"] += time.monotonic() - t2
        st.moved_at = time.monotonic()
        self.actions.facing = None
        after = self._after_chain(cut, st.moved_at, st)
        # Cut short, she replans from a quick look, without monsters: with an escort that skips the fights
        # (and `until`) for one chain, so she moves on from where a hit knocked her first. Fighting from
        # there instead (a look with monsters) kept her fighting wherever she was knocked to: in the
        # simulator a 12 s trip from 12 across to 17 (nav_fight_across) ran out its 180 s.
        st.stopped = cut.startswith(COMING)
        st.quick = bool(cut) and not st.stopped  # (cut for a monster coming: the next look has it, or TripState.came)
        st.came, self._came = (self._came if st.stopped else None), None
        return [leg.move for leg in legs], legs[-1].end.platform, after

    def _carry_out_move(self, st: TripState, s: Scene, path: list[Move]) -> tuple[list[Move], Platform, Scene]:
        """The route's first move on its own, closed loop (control.mover). Returns it (as a list), the
        platform it goes to, and a look once it's done; none, where she is and `s`, if the mover can't do it
        from here (it costs CANNOT_S for the rest of the trip: st.cannot)."""
        move = path[0]
        self.log(f"  {platform_text(s.where.platform)} at x {s.where.x:.0f} ({s.how}): {move.kind} "
                 f"{move.direction} -> platform {move.dst}  [{len(path)} moves to go]")
        if telemetry.recording():  # (as a chain's 'chain' record: she sets off; 'moved' once it's done)
            telemetry.emit("move", kind=move.kind, src=move.src, dst=move.dst, x=round(move.x, 1))
        t2 = time.monotonic()
        try:
            after = self.mover.do(move, s)
        except ValueError as e:  # one it can't do from here (CANNOT_S): another way, from the next look
            self.log(f"    can't: {e}; another way")
            st.cannot.add(_key(move))
            if telemetry.recording():  # (nothing made: no 'moved' record ends the 'move' one)
                telemetry.emit("event", kind="cannot", text=f"{move.kind} -> {move.dst}: {e}")
            return [], s.where.platform, s
        st.trip.time["move"] += time.monotonic() - t2
        st.moved_at = time.monotonic()
        return [move], self.model.platform(move.dst), after

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

    def _rope_top(self, st: TripState, s: Scene, after: Scene) -> Scene:
        """After a chain or a move from look `s` whose keys never moved her dot (`_pass`: the chain's glances for
        STUCK_S, or every look of the mover's), with `s` placing her at the top of a rope hanging from her platform
        (rope_at_top) and `after` her dot where it was: she's on that rope under its top, not on the platform (a jump
        or an arrow there does nothing; her dot snaps onto the platform from up to 33 under it, and from 17 under it by
        a rope since: localization.pose.ROPE_TOP_GAP), as at 22:19:38 on 2026-10-26, when 10 moves from rope 4's top
        "ended on platform 18 instead" over 15.4 s, her dot 14.8 under 18 at every chain start (Tree Dungeon; the
        3,039 chain starts' standing gaps: at most 14.0, none by a rope's top over 12). Her dot more than STUCK_GAP
        under the platform's surface only (standing by a rope's top, 9.2 at the most). Off it (Mover.off_rope: up, onto
        the platform), and a look once she's there; else `after`."""
        rope = None if s.where is None else rope_at_top(self.model, s.where)
        if rope is None or after.where is None or after.where.dot is None or s.where.dot is None \
                or abs(after.where.dot[0] - s.where.dot[0]) > STILL or abs(after.where.dot[1] - s.where.dot[1]) > STILL:
            return after
        x, y = self.senses.localizer.to_map(after.where.dot)
        p = s.where.platform
        if y - p.y_at(p.clamp(x)) <= STUCK_GAP:
            return after
        text = (f"her dot never moved: on the rope at x {rope.x:.0f}, {y - rope.top:.0f} under its top, not on "
                f"platform {s.where.platform.id}")
        self.log(f"    {text}: getting off")
        telemetry.emit("event", kind="rope_top", text=text)
        st.fresh = None  # (the look after the chain, for the next pass: she's moved since)
        return self.mover.off_rope(after, rope, at=(x, y))

    def _safe_to_stand(self, x: float, here: Platform, move: Move, monsters) -> bool:
        """Whether no monster can reach her standing at x for the next second (as a leg that stays put)."""
        return not self.flow.danger(
            Leg(move, Body(0.0, x, here), Body(1.0, x, here), [Span(here.id, 0.0, 1.0, x, x)]), monsters)

    def _after_chain(self, cut: str, moved_at: float, st: TripState) -> Scene:
        """A look once a chain's plan is over: once she's stopped moving if it was cut short (`cut`),
        else once she's on a platform. Not cut short, it's the look the next pass needs (with the monsters):
        two looks in a row there were half the stop between chains."""
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

    def _note_landing(self, trip: Trip, moved: list[Move], planned: Platform, after: Scene, hp: float,
                      stopped: bool = False) -> bool:
        """Record the moves made and where she landed in trip.moves (and a hit in trip.hits), and log
        it if that's not `planned`. True if it's a miss: elsewhere than planned, and not because a
        monster hit her or (`stopped`) she stopped short for one coming into the chain's way (the caller counts
        misses)."""
        landed = after.where.platform
        hit = after.hp < hp - HIT
        if telemetry.recording():  # (for the coverage report: analysis.coverage)
            telemetry.emit("moved", moves=[{"kind": m.kind, "src": m.src, "dst": m.dst, "x": round(m.x, 1)}
                                           for m in moved],
                           planned=planned.id, landed=landed.id if landed else None, hit=hit)
        trip.moves += [f"{m.kind} {m.src}->{m.dst}" for m in moved]
        trip.moves[-1] += f": landed on {landed.id if landed else None}" + (" (hit)" if hit else "")
        if hit:
            trip.hits += 1
        if landed is not None and landed.id == planned.id:
            return False
        if hit:
            self.log(f"    hit on the way ({hp:.0%} -> {after.hp:.0%}): on {platform_text(landed)}; replanning")
            return False
        if stopped:  # (where the cut left her: said by _after_chain)
            return False
        self.log(f"    ended on {platform_text(landed)} instead of platform {planned.id}; replanning")
        return True

    def follow(self, legs, keys, hp: float, dst: int, why: str, trip: Trip, st: TripState | None = None,
               path: list[Move] | None = None, monsters=()) -> tuple[str, list]:
        """Run a chain's plan and keep going like a player, with no stop before the next chain: late in
        it, a look around decides whether one may follow (_look_ahead); once she's back on the ground for
        good, a look places her and the next chain is planned from there (_queue_next), queued so the
        player starts it the moment this one finishes. Meanwhile her dot and HP are watched ~15 times a
        second: a hit, or her being on another platform than planned for a few glances, cuts the running
        plan and the queued one short (the player lets go of every key). Returns why it was cut ('' if
        not) and all the legs sent. (`path`: the route the chain is from; `st`: its trip; `monsters`: the
        ones the chain keeps clear of.)"""

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
        current = _Running(plan_id, legs, time.monotonic() + PLAN_START_S, more, tail, length, tuple(monsters))
        self._sent_plans = [current.plan_id]  # every plan this chain sent: cancelled if anything fails
        self.actions.pressed = True
        try:
            return self._watch(current, hp, dst, send, st)
        except BaseException:
            # Whatever went wrong here (Ctrl+C included), the plans sent must not play on unwatched.
            for pid in self._sent_plans:
                try:
                    client.cancel(pid)
                except Exception:
                    pass
            self._ended(None, cut=True)
            raise

    def _watch(self, current: _Running, hp: float, dst: int, send, st: TripState | None) -> tuple[str, list]:
        """follow's loop: watch the running chain, queue the next, cut both short when she's hit or off
        her path."""
        queued, sent, cut = None, list(current.legs), ""
        stood = None  # since when, and where, her dot has stood (_stood)
        off_since, off_leg = None, False  # the first glance in a row off the planned path, and whether one of them
                                          # had her where the leg doesn't go
        ahead = None  # the look around for what follows the running chain, once taken
        over = False  # the running chain's plan has ended (its result read)
        self.chain_still = None  # her dot where the first glance had it all along, and for how long (_rope_top)
        first, moved, first_at, last_at = None, False, 0.0, 0.0
        dot = None
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
                if queued is None:
                    break
                current, queued, over = queued, None, False
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
            if now_hp < hp - HIT:
                cut = f"hit ({hp:.0%} -> {now_hp:.0%})"
            elif dot is not None:
                on, want, leg, behind = self._placed(dot, seen_at - stood[0] >= STILL_S, current)
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
            if not cut and REACT and (coming := self._coming(current)) is not None:
                mo, t, tid = coming
                self._reacted[tid] = time.monotonic()
                cut = f"{COMING} ({mo.kind} on {mo.platform}, {t - (time.monotonic() - current.start):.2f} s ahead)"
            if cut:
                climbing = self._climbing(current, dot) if cut.startswith("hit") and CLIMB_ON else None
                for pid in [current.plan_id] + ([queued.plan_id] if queued else []):
                    if not client.cancel(pid):
                        client.wait(pid)
                self._ended(current.plan_id, cut=True)
                if queued:
                    sent = sent[:len(sent) - len(queued.legs)]
                if climbing is not None:
                    self._climb_on(*climbing)
                break
            if client.done(current.plan_id):
                client.wait(current.plan_id)
                self._ended(current.plan_id)
                over = True  # the queued chain takes over, or the next one is still to come (see above)
        if first is not None and not moved:
            self.chain_still = (first, last_at - first_at)
        return cut, sent

    def _climbing(self, running: _Running, dot) -> tuple[float, float] | None:
        """Whether she's climbing a rope of the running plan now (its Climb from the grab to UP_EXTRA past the top, and
        her last glance's dot by the rope and CLIMB_ON_UNDER or more under its top, or no dot and the plan's height
        so): the rope's top and her height on it (her dot's, else the plan's); None if not."""
        t = time.monotonic() - running.start
        for leg in running.legs:
            for c in leg.rope:
                if not c.t0 <= t <= c.t1 + UP_EXTRA:
                    continue
                if dot is None:
                    y = c.at(t)[1]
                    return (c.y1, y) if y - c.y1 >= CLIMB_ON_UNDER else None
                mx, my = self.senses.localizer.to_map(dot)
                if abs(mx - c.x) <= CLIMB_ON_X and c.y1 + CLIMB_ON_UNDER <= my <= c.y0 + CLIMB_ON_X:
                    return c.y1, my
        return None

    def _climb_on(self, top: float, y: float) -> None:
        """A hit while she climbs a rope (_climbing): the cut let go of every key, up too, and the game keeps her on the
        rope when she's touched there (live on 2026-10-26, 11 of 11 hits on a rope), where she'd hang below its top
        (see CLIMB_ON); up again, to the top and onto the platform there, held as a chain's climb holds it (the climb
        from her height at flow.ROPE_SPEED, the step onto the platform in it, and UP_EXTRA: 0.75 s from 52 under, where
        mover.climb_ms's 0.945 s had the same mushroom touch her again on the platform 9 ms before it ended in
        loop_tree_walls), waited for, so the next pass looks once she's there."""
        k = self.actions.keys.up.name
        hold = ms(max(0.0, y - top) / ROPE_SPEED + UP_EXTRA)
        self.log(f"    hit on the rope {y - top:.0f} under its top: up again ({hold} ms)")
        telemetry.emit("event", kind="climb_on", text=f"hit on the rope {y - top:.0f} under its top: up again")
        self.actions.play([[0, k, "down"], [hold, k, "up"]], "climb_on")

    def _ended(self, plan_id: str | None, cut: bool = False) -> None:
        """A chain's plan is over, its result read: its last key went its run on past it before now (send's), or now,
        cut short (every key let go: Actions.ended)."""
        tail = self._tails.pop(plan_id, 0.0)
        if hasattr(self.actions, "ended"):  # (a stand-in for Actions may not have it)
            self.actions.ended(0.0 if cut else tail)

    def _coming(self, running: _Running) -> tuple[MonsterTrack, float, int] | None:
        """A monster the feed follows (the last glance's state: Senses.glanced) that the running chain's legs ahead of
        her now meet, REACT_LEAD_S on at the soonest, while she's on the ground: it, when (plan time) and its track's
        id; None if none, without a feed's glance, in the air or on a rope. Only the kinds the chain keeps clear of
        (with an escort, those whose touch counts), each once in REACT_AGAIN_S (see REACT)."""
        state = getattr(self.senses, "glanced", None)  # (a stand-in for Senses may not have it)
        if state is None or not state.placed or (self.fight and not self._threats()):
            return None
        now = time.monotonic()
        t = now - running.start
        if not self._room_to_stop(running, t):
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
        MATCH_SLACK of where the feed has it. The chain was planned against it then, or past it on purpose (a way
        through after waiting for it, or regardless: _meet); REACT is for those the look hadn't."""
        p = self.model.platform(tr.platform)
        for mo in known:
            if mo.platform == tr.platform and mo.kind == tr.kind:
                lo, hi = mo.reach(t, p)
                if lo - MATCH_SLACK <= tr.x <= hi + MATCH_SLACK:
                    return True
        return False

    def _track_of(self, tr, start: float) -> MonsterTrack:
        """A monster the feed follows (tracking.Track) as the move planner sees it, as of its last sighting in the plan
        that started at `start` (time.monotonic): walking at its velocity as followed (to its kind's speed at most), or
        standing (under STILL_VX: where it may start walking either way, as a look's standing one)."""
        speed = self.tracker.walking.get(tr.kind, DEFAULT_SPEED)
        half, top = touch_box(tr.kind, self.tracker.body.get(tr.kind))
        vx = max(-speed, min(speed, tr.vx)) if abs(tr.vx) > STILL_VX else 0.0
        heading = (1 if vx > 0 else -1) if vx else 0
        return MonsterTrack(tr.platform, tr.x, speed, heading, half, vx, tr.last_seen - start,
                            MONSTER_HEIGHT.get(tr.kind, top), tr.facing, tr.kind, top)

    def _placed(self, dot, still: bool, running: _Running) -> tuple[Platform | None, int | None, Leg, float | None]:
        """Where a glance at her dot puts her against the running plan: the platform she stands on
        (None if none, or her dot isn't `still`: it hasn't stood where it is for STILL_S), the
        platform she should be on by now (None: in the air, or about to take off or land), the leg she's
        on, and her x if she's standing where her run along that platform starts while the plan has her
        OFF_SPAN or more along it (else None). While the plan has her in the air (Flow.airborne) she stands
        nowhere: near the top of a jump her dot can sit still over a platform she's flying past."""
        t = time.monotonic() - running.start
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
        if st is None or end.platform.id == dst:
            return None
        s = self.senses.look(mobs=True, wide=not self.fight)
        if s.where is None:
            return None
        p = end.platform
        there = replace(s, where=replace(s.where, x=end.x, y=p.y_at(end.x), platform=p))
        threats = self.fight and self._threats()  # (where the next chain's legs go: trip_monsters)
        ahead = {p.id} | _platforms(_after(running, st.path))
        seen = self.trip_monsters(s, ahead) if threats else s.monsters
        tracks = tuple(self.tracker.update(s.t, seen, seen_at=s.t - running.ends))  # as of the next start
        if threats:  # and those her memory has there off the look's view
            tracks += tuple(self.remembered(s, ahead, seen, s.t - running.ends))
        if not self.fight:  # avoiding them: the next chain keeps clear of them
            return _Ahead(there, tracks)
        if st.until is not None and st.until(there):
            return _no_follow("the trip's until, where it ends")
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
        tracks = [replace(m, seen_at=m.seen_at - (start - starts)) for m in ahead.tracks]
        price, alts, goal = self._pricing(st, tracks)
        legs, keys, why = self.flow.chain(path, Body(0.0, x, p, end.run, False, QUEUE_SLACK), tracks,
                                          keep_going=True, hops=not self.fight, price=price, alts=alts, goal=goal,
                                          spread=QUEUE_SPREAD)
        if not legs:
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
                        tuple(watch if self.fight else tracks))

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


def standing(s: Scene) -> bool:
    """Whether look `s` shows her standing, as far as it tells: ground under her feet on screen (in the air
    there's none: the minimap dot alone puts her on a platform she's falling past or jumping over), unless her
    name tag is hidden or no ground was read."""
    return not s.tag_seen or s.ground is None or s.ground[1] - s.ground[0] >= s.geo.fpx(MIN_GROUND_PX)


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


def _platforms(path: list[Move]) -> set[int]:
    """The platforms the first moves of a route (as many as a chain takes: flow.MAX_LEGS) go from and onto."""
    return {p for m in path[:MAX_LEGS] for p in (m.src, m.dst)}


def _same_sighting(a: Monster, b: Monster) -> bool:
    """Whether two monsters found on the same frame and placed on the map are one: of a kind, standing within
    SAME_SIGHTING of each other."""
    return a.kind == b.kind and abs(a.x - b.x) <= SAME_SIGHTING and abs(a.y - b.y) <= SAME_SIGHTING


def _after(running: "_Running", path: list[Move]) -> list[Move]:
    """The moves of `path` (the route the running chain was planned from) after the chain's: hops are the chain's own,
    not the route's."""
    return path[sum(1 for leg in running.legs if leg.move.kind != "hop"):]


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
