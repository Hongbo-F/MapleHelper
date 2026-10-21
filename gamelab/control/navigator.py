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
chain stops before a leg a monster could walk into (flow.danger); that move then costs extra in the
route planner so a way around is taken if there is one, else she waits for the monster to move off
(a few looks), and then goes anyway. A hit during a move shows as lost HP: she replans from wherever
she was pushed to, without counting it as a missed move.

With an escort (hunting tours: behavior.fight.FightOnTheWay), each trip has a mode (go(mode=...)):
`hunt` (the default) shoots what's in reach before each chain and any monster in the way of a move along
her platform; `transit` (the trip to rest) only shoots what's in the way, and low HP is no reason to stop
(potions are still drunk). Without an escort the mode is `avoid`, as described above.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import NamedTuple, Protocol

from .. import telemetry
from ..localization.mapdata import MapModel, Platform
from ..localization.monsters import Monster, Tracker, tracks_record
from ..localization.scene import Scene
from ..localization.senses import Senses
from ..motion.flow import Body, Flow, Leg, Span, chain_record, ms
from ..motion.physics import WALK_SPEED
from ..motion.route import Move, Planner, route_record
from . import client
from .actions import Actions
from .mover import Mover, platform_text

ON_WAY_TRIES = 5    # monsters engaged per platform while passing through (respawns could keep her forever)
MAX_FIGHTS = 3      # fights in a row clearing the way, without a move between: then she moves on regardless
AVOID_SECONDS = 6.0  # extra route cost of a move a monster is in the way of (so a way around wins) ...
AVOID_MEMORY_S = 4   # ... for this long after the monster was seen there
WAIT_LOOKS = 6       # looks spent waiting for a monster to move off before going anyway
WAIT_S = 0.3         # between those looks
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
MIN_GROUND_PX = 6    # half-size px of ground under her feet that show she's standing (in the air: none)
END_WATCH = 0.1      # s before a plan with nothing queued after it ends: glances stop, its end is waited for ...
END_POLL_S = 0.005   # ... this closely (noticed late, the next look and plan came ~70 ms later)
FRESH_S = 0.3        # s a look stays good enough to plan from, if no key plan was sent since
AFTER_STILL_S = 0.35  # s at most waited for her to stand still after a chain, before the next pass's look
                      # (a stop slides her ~0.26 s; knocked about, she isn't held up longer)
PLAN_START_S = 0.015  # a chain's plan is taken to start this long after it's sent
GLANCE_S = 0.03      # between glances at her dot while a plan runs, or while she recovers
OFF_PATH = 3         # glances off the planned path (counting 2 where the leg doesn't go) that cut a plan short
STILL = 0.6          # minimap px: a dot that moved less than this on both axes has stopped ...
STILL_GLANCES = 3    # ... for this many glances in a row: she stands on the platform under it. Over 2 glances
                     # (63 ms apart) a dot can sit still in the air too: near the top of a jump, or at full
                     # speed within a minimap pixel (11-16 units: 149 units/s covers 9 between two glances)
# Standing still where her run along a platform starts while the plan has her this far along it (Span.x_at) is
# off the path too. In game on 2026-10-21 chains whose first arrow went down 0.6 s after a throw's key set off
# about 0.2 s late, the jump at the end of the run went at its planned time from 15-30 units short, and she fell
# off: 8 of 10 jumps from 16 up onto 14 fell to 25. Her dot stood still at the start for the first 0.35-0.48 s
# while the plan had her running. (She speeds up from standing, so the plan's straight line runs ahead of her
# by up to 15 units early on; her dot is read to whole minimap pixels.)
OFF_SPAN = 30
HIT = 0.004          # HP lost (fraction) that counts as a hit
# Looks at most, each after waiting up to 2 s for her to land and stand still (Mover.settle), for one that puts
# her on a platform, when a look shows her in the air: a slime's touch as she walked off platform 3 onto 4
# knocked her back into the 12-unit gap between the blocks at x 669-681, down which she fell to the basin for
# 2.5 s, and with one look the trip ended "lost: not on any platform" while she still fell (simulator,
# nav_monsters_long_way on seeds 2 and 4).
FALL_LOOKS = 2

# How a trip treats monsters (Navigator.go's `mode`).
AVOID = "avoid"      # no escort: whole-screen looks, ways around, waits, hops, chains planned on the move
HUNT = "hunt"        # the escort shoots what's in reach before each chain, and what's in the way of a move
TRANSIT = "transit"  # the escort shoots only what's in the way; low HP doesn't stop the trip (potions go on)
TRANSIT_AVOID = "transit-avoid"  # as avoid (no escort, ways around monsters), and as transit, low HP doesn't
                                 # stop the trip (`tour --rest-avoid`: the trips to rest and to stand by)
MODES = (AVOID, HUNT, TRANSIT, TRANSIT_AVOID)
FIGHTING = (HUNT, TRANSIT)  # the modes the escort fights in


class Escort(Protocol):
    """Fighting on the way (behavior.fight.FightOnTheWay): what the navigator asks between moves."""

    def target(self, s: Scene) -> Monster | None:
        """The monster to shoot from where she stands, if any is in reach."""

    def shoot(self) -> int:
        """Fight what's in reach for a moment; returns the kills."""

    def in_way(self, s: Scene, x: float) -> Monster | None:
        """The monster on her platform in her way to map x that clear_way would shoot (None if none)."""

    def clear_way(self, s: Scene, x: float) -> int | None:
        """Shoot a monster on her platform in her way to map x (where her next move starts, or where she's
        sent to on arrival); the kills, or None if none was in the way."""


@dataclass
class Trip:
    """How a trip (Navigator.go) went."""
    result: str = ""      # 'arrived', 'in position', a reason to stop (low HP), or why she didn't get there
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

    @property
    def ends(self) -> float:
        """When its plan is over (time.monotonic)."""
        return self.start + (self.length if self.length is not None else self.legs[-1].end.t + self.tail)


class _Ahead(NamedTuple):
    """The look around late in a running chain, for the chain that may follow it."""
    scene: Scene         # with her put where the running chain ends
    tracks: tuple = ()   # the monsters the next chain keeps clear of (none with an escort) ...
    watch: tuple = ()    # ... or, with an escort, that it mustn't come near (else she stops, and it fights them)


@dataclass
class TripState:
    """A trip in progress (Navigator.begin, then Navigator.step until it's over): where it goes and how,
    and what it keeps from one look to the next."""
    dst: int
    x: float | None = None
    deadline: float = 0.0  # time.monotonic() it times out at
    max_misses: int = 6
    until: Callable[[Scene], bool] | None = None
    health: Callable[[Scene], str | None] = lambda s: None
    mode: str = AVOID
    trip: Trip = field(default_factory=Trip)  # how it's going so far (returned at the end)
    weights: dict[tuple, float] = field(default_factory=dict)  # its random move weights, kept through its replans
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

    def end(self, result: str) -> Trip:
        """The trip, over: `result` says how it ended."""
        self.trip.result, self.over = result, True
        return self.trip


class Navigator:
    """Gets her onto a platform: plans the route, chains its moves, and replans from wherever she ends up."""

    def __init__(self, senses: Senses, actions: Actions, model: MapModel, *, log=print,
                 health: Callable[[Scene], str | None] | None = None,
                 potions: Callable[[Scene], None] | None = None, escort: Escort | None = None,
                 variety: float = 0.35, rng: random.Random | None = None):
        """`health(scene)` returns a reason to stop (low HP), and may drink a potion; `potions(scene)` only
        drinks (transit trips); `escort` fights on the way (without one she avoids monsters instead)."""
        self.senses, self.actions, self.model, self.log = senses, actions, model, log
        self.health = health or (lambda s: None)
        self.potions = potions or (lambda s: None)
        self.escort = escort
        self.mode = HUNT if escort is not None else AVOID  # a trip's mode unless go() says otherwise
        self._mode = self.mode                             # the running trip's
        self.planner = Planner(model)
        self.mover = Mover(senses, actions, model, log)
        self.rng = rng or random.Random()
        # Like a player, not the same way every time: each trip weighs every move by a random 1 to
        # 1 + variety, so routes within ~variety of the quickest take turns (the basin's two ropes, a
        # drop or a walk-off), and moves start from varied spots (flow.Flow's rng).
        self.variety = variety
        self._weights: dict[tuple, float] = {}
        names = {k: getattr(actions.keys, k).name for k in ("left", "right", "jump", "up", "down")}
        if actions.keys.pick_up is not None:
            names["pick"] = actions.keys.pick_up.name  # pressed all through her runs: loot on the way
        self.flow = Flow(model, names, rng=self.rng if variety else None)
        self.tracker = Tracker(model)  # which way the monsters walk, from look to look
        self._checked: Scene | None = None  # follow: the last look placing her for the next chain

    @property
    def fight(self) -> bool:
        """Fighting on the way (the escort's hunt or transit mode) rather than avoiding monsters."""
        return self._mode in FIGHTING

    def go(self, dst: int, x: float | None = None, seconds: float = 180, max_misses: int = 6,
           until: Callable[[Scene], bool] | None = None, *, mode: str | None = None,
           scene: Scene | None = None) -> Trip:
        """Get onto platform `dst` (and to map x on it, if given). `until` is asked before every chain,
        with a look that includes monsters (late in the chain before, when it follows on from that one):
        True ends the trip early ('in position'). `mode` (AVOID, HUNT or TRANSIT; the navigator's own by
        default) says how the trip treats monsters. `scene`: the caller's last look, used for the first
        pass instead of a new one if it's fresh (no keys sent since)."""
        st = self.begin(dst, x, seconds, max_misses, until, mode=mode, scene=scene)
        while not st.over:
            self.step(st)
        return st.trip

    def begin(self, dst: int, x: float | None = None, seconds: float = 180, max_misses: int = 6,
              until: Callable[[Scene], bool] | None = None, *, mode: str | None = None,
              scene: Scene | None = None) -> TripState:
        """A trip to platform `dst`, as go() makes it, to be carried out one step() at a time."""
        mode = mode or self.mode
        if mode not in MODES or (mode in FIGHTING and self.escort is None):
            raise ValueError(f"no {mode!r} trip without an escort" if mode in MODES else f"unknown mode {mode!r}")
        # Transit: low HP is why she's going (to rest), so potions only.
        health = self.potions if mode in (TRANSIT, TRANSIT_AVOID) else self.health
        st = TripState(dst, x, time.monotonic() + seconds, max_misses, until, health, mode)
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
        # Look, and deal with what's here.
        s, looked = self._look(st)
        st.last = s
        if stop := self._reason_to_stop(s, st.until, st.health):
            return st.end(stop)
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
        monsters = self._chain_monsters(st, s)
        st.quick = False
        legs, keys, why = self._chain(st, s, path, monsters, looked)
        if self._wait_for_monster(st, s, path[0], monsters, legs, why, planned_at):
            return None  # look again: for a way around, or once she's waited a moment

        # Move: the chain in one plan, or else the first move on its own.
        hp = s.hp
        if legs:
            # With an escort the chain keeps clear of no monster (it fights them instead), but its run on past
            # its end (follow's tail) mustn't take her into one.
            near = monsters if not self.fight else self.tracker.update(s.t, s.monsters, seen_at=s.t - time.monotonic())
            moved, planned, after = self._carry_out_chain(st, s, path, legs, keys, why, st.dst, near)
        else:
            moved, planned, after = self._carry_out_move(st, s, path)
        if self._note_landing(st.trip, moved, planned, after, hp):  # a miss
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

    def _reason_to_stop(self, s: Scene, until: Callable[[Scene], bool] | None,
                        health: Callable[[Scene], str | None]) -> str | None:
        """Why the trip ends at this look, if it does: `health`'s reason (low HP), or 'in position' when
        she's on a platform and `until` says so."""
        if stop := health(s):
            return stop
        if until is not None and s.where.platform is not None and until(s):
            return "in position"
        return None

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
        path = self.planner.plan(s.where.platform.id, s.where.x, dst, weight=self.weight,
                                 penalty=lambda m: AVOID_SECONDS if _key(m) in st.avoid else 0.0)
        if path and telemetry.recording():
            telemetry.emit("route", **route_record(dst, path))
        return path, now

    def _clear_the_way(self, st: TripState, s: Scene, x: float) -> bool:
        """With an escort: shoot a monster on her platform in her way to map x (where the route's first move
        starts, or where she's sent to on arrival), up to MAX_FIGHTS times in a row without a move between;
        its kills count in the trip's. True if she fought."""
        if not self.fight:
            return False
        if st.fights < MAX_FIGHTS and (kills := self.escort.clear_way(s, x)) is not None:
            st.fights += 1
            st.trip.kills += kills
            return True
        if st.fights >= MAX_FIGHTS:
            telemetry.emit("event", kind="backstop", text="MAX_FIGHTS: moving on past the monster in the way")
        st.fights = 0
        return False

    def _chain_monsters(self, st: TripState, s: Scene) -> list:
        """The monsters the chain keeps clear of: none with an escort (it fights them instead); after a
        quick look, the last look's, moved on by the time since; else this look's, tracked from look to
        look (which way they walk)."""
        if self.fight:
            return []
        if st.quick:  # the last look's monsters, moved on by the time since
            looked, seen_then = self.tracker.last
            return [replace(mo, seen_at=looked - time.monotonic()) for mo in seen_then]
        monsters = self.tracker.update(s.t, s.monsters, seen_at=s.t - time.monotonic())
        if telemetry.recording():
            telemetry.emit("monsters", **tracks_record(monsters, self.model))
        return monsters

    def _chain(self, st: TripState, s: Scene, path: list[Move], monsters: list, looked: float) -> tuple:
        """As many of the route's moves as can be predicted, chained from where she stands (flow.chain:
        the legs, their keys, and why the chain ended). The time since the look (`looked`) counts as
        thinking."""
        # Right after a plan she may still be sliding or landing: the first jump/drop waits. A chain cut off
        # for length doesn't stop her at its end: the next one may follow on (follow).
        settle = max(0.0, SETTLE_S - (time.monotonic() - st.moved_at))
        legs, keys, why = self.flow.chain(path, Body(0.0, s.where.x, s.where.platform, 0, settle > 0, settle),
                                          monsters, keep_going=True, hops=not self.fight)
        st.trip.time["think"] += time.monotonic() - looked
        return legs, keys, why

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
        st.quick = bool(cut)
        return [leg.move for leg in legs], legs[-1].end.platform, after

    def _carry_out_move(self, st: TripState, s: Scene, path: list[Move]) -> tuple[list[Move], Platform, Scene]:
        """The route's first move on its own, closed loop (control.mover). Returns it (as a list), the
        platform it goes to, and a look once it's done."""
        move = path[0]
        self.log(f"  {platform_text(s.where.platform)} at x {s.where.x:.0f} ({s.how}): {move.kind} "
                 f"{move.direction} -> platform {move.dst}  [{len(path)} moves to go]")
        t2 = time.monotonic()
        after = self.mover.do(move, s)
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

    def _note_landing(self, trip: Trip, moved: list[Move], planned: Platform, after: Scene, hp: float) -> bool:
        """Record the moves made and where she landed in trip.moves (and a hit in trip.hits), and log
        it if that's not `planned`. True if it's a miss: elsewhere than planned, and not because a
        monster hit her (the caller counts misses)."""
        landed = after.where.platform
        hit = after.hp < hp - HIT
        trip.moves += [f"{m.kind} {m.src}->{m.dst}" for m in moved]
        trip.moves[-1] += f": landed on {landed.id if landed else None}" + (" (hit)" if hit else "")
        if hit:
            trip.hits += 1
        if landed is not None and landed.id == planned.id:
            return False
        if hit:
            self.log(f"    hit on the way ({hp:.0%} -> {after.hp:.0%}): on {platform_text(landed)}; replanning")
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
            return client.send(events, lead_ms=0, focus=True, name="flow", end_ms=end_ms), end_ms / 1000

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
        current = _Running(plan_id, legs, time.monotonic() + PLAN_START_S, more, tail, length)
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
            raise

    def _watch(self, current: _Running, hp: float, dst: int, send, st: TripState | None) -> tuple[str, list]:
        """follow's loop: watch the running chain, queue the next, cut both short when she's hit or off
        her path."""
        queued, sent, cut = None, list(current.legs), ""
        dots, off = [], 0  # the last glances' dots; off: glances in a row off the planned path
        ahead = None  # the look around for what follows the running chain, once taken
        over = False  # the running chain's plan has ended (its result read)
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
                    break
                if (last if current.tail else current.ends) - time.monotonic() <= END_WATCH:
                    if client.done(current.plan_id):
                        client.wait(current.plan_id)
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
            if dot is not None and telemetry.recording():
                gx, gy = self.senses.localizer.to_map(dot)
                telemetry.emit("glance", x=round(gx, 1), y=round(gy, 1), hp=round(now_hp, 3))
            if now_hp < hp - HIT:
                cut = f"hit ({hp:.0%} -> {now_hp:.0%})"
            elif dot is not None:
                on, want, leg, behind = self._placed(dot, dots, current)
                if on is not None and on.id not in (leg.start.platform.id, leg.end.platform.id):
                    # Standing where this leg doesn't go (e.g. a drop that fell one platform too far):
                    # stop before the plan's next keys fire there.
                    off += 2
                elif on is not None and (want is not None and on.id != want or behind is not None):
                    off += 1
                else:
                    off = 0
                if off >= OFF_PATH:
                    where = f"still at x {behind:.0f}" if behind is not None else f"on {on.id if on else 'nothing'}"
                    cut = f"off the path ({where}, planned {want or leg.end.platform.id})"
            dots = [*dots, dot][-(STILL_GLANCES - 1):] if dot is not None else []
            if cut:
                for pid in [current.plan_id] + ([queued.plan_id] if queued else []):
                    if not client.cancel(pid):
                        client.wait(pid)
                if queued:
                    sent = sent[:len(sent) - len(queued.legs)]
                break
            if client.done(current.plan_id):
                client.wait(current.plan_id)
                over = True  # the queued chain takes over, or the next one is still to come (see above)
        return cut, sent

    def _placed(self, dot, dots: list, running: _Running) -> tuple[Platform | None, int | None, Leg, float | None]:
        """Where a glance at her dot puts her against the running plan: the platform she stands on
        (None if none, or the dot moved in the last STILL_GLANCES glances: `dots` are the ones before), the
        platform she should be on by now (None: in the air, or about to take off or land), the leg she's
        on, and her x if she's standing where her run along that platform starts while the plan has her
        OFF_SPAN or more along it (else None). While the plan has her in the air (Flow.airborne) she stands
        nowhere: near the top of a jump her dot can sit still over a platform she's flying past."""
        t = time.monotonic() - running.start
        want = self.flow.expected(running.legs, t)
        seen = [*dots, dot][-STILL_GLANCES:]
        still = len(seen) == STILL_GLANCES and all(_still(b, a) for a, b in zip(seen, seen[1:]))
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
        the monsters, if `mobs`; on the whole screen, if `wide`)."""
        end, last = time.monotonic() + longest, None
        while time.monotonic() < end:
            dot, _ = self.senses.glance()
            if dot is not None and _still(dot, last):
                w = self.senses.localizer.where_dot(dot)
                if w.platform is not None:
                    break
            last = dot
            self.senses.wait_frame(GLANCE_S)
        return self.mover.locate(mobs=mobs, wide=wide)

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
        tracks = tuple(self.tracker.update(s.t, s.monsters, seen_at=s.t - running.ends))  # as of the next start
        if not self.fight:  # avoiding them: the next chain keeps clear of them
            return _Ahead(there, tracks)
        if st.until is not None and st.until(there):
            return _no_follow("the trip's until, where it ends")
        if self._mode == HUNT and st.tries.get(p.id, 0) < ON_WAY_TRIES and self.escort.target(there):
            return _no_follow("a monster in reach where it ends")
        return _Ahead(there, watch=tracks)

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
        path = self.planner.plan(p.id, x, dst, weight=self.weight,
                                 penalty=lambda m: AVOID_SECONDS if _key(m) in st.avoid else 0.0)
        if not path:
            return None
        here = replace(ahead.scene, where=replace(ahead.scene.where, x=x, y=p.y_at(x)))
        if self.fight and self.escort.in_way(here, path[0].x) is not None:
            return _no_follow("a monster in the way of the next move")
        # It starts when the running plan ends, or at once if that's over: the monsters are predicted for then.
        start = max(starts, time.monotonic() + PLAN_START_S)
        tracks = [replace(m, seen_at=m.seen_at - (start - starts)) for m in ahead.tracks]
        legs, keys, why = self.flow.chain(path, Body(0.0, x, p, end.run, False, QUEUE_SLACK), tracks,
                                          keep_going=True, hops=not self.fight)
        if not legs:
            return None
        # With an escort, chains are planned blind (it fights the monsters instead, at the stops). One that
        # follows on skips a stop: only where none of the monsters seen can get near it. (Queued regardless,
        # misses went up by half in the simulator's hunt-mode trips: touches knock her off her jumps.)
        watch = [replace(m, seen_at=m.seen_at - (start - starts)) for m in ahead.watch]
        if watch and any(self.flow.danger(leg, watch) for leg in legs):
            return _no_follow("a monster could get near the next chain")
        if time.monotonic() >= starts:  # the running plan is over: she's sliding to a stop
            return _no_follow(f"planned too late ({time.monotonic() - starts:+.3f} s from the plan's end)")
        telemetry.emit("event", kind="follow", text="followed on")
        tail = self._tail(legs, path, tracks or watch) if why == "long" else None
        if why == "long" and tail is None:
            keys.set_arrow(legs[-1].end.t, 0)  # nothing will follow it: she stops where it ends
        self.log(f"    then from platform {p.id} at x {x:.0f} (planned on the move): "
                 + _chain_text(legs, len(path), why))
        if telemetry.recording():
            telemetry.emit("route", **route_record(dst, path))
        plan_id, length = send(legs, keys, why, tail or 0.0)
        return _Running(plan_id, legs, start, tail is not None, tail or 0.0, length)

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


def standing(s: Scene) -> bool:
    """Whether look `s` shows her standing, as far as it tells: ground under her feet on screen (in the air
    there's none: the minimap dot alone puts her on a platform she's falling past or jumping over), unless her
    name tag is hidden or no ground was read."""
    return not s.tag_seen or s.ground is None or s.ground[1] - s.ground[0] >= MIN_GROUND_PX


def _no_follow(why: str) -> None:
    """Note (telemetry) why a chain cut off for length isn't followed on; None, for the caller to return."""
    telemetry.emit("event", kind="follow", text=why)
    return None


def _key(m: Move) -> tuple:
    """A move's identity across replans."""
    return (m.kind, m.src, m.dst, round(m.x))


def _ground_from(leg: Leg) -> float:
    """When (s into its plan) she's back on the ground for good on the platform a leg ends on: where her
    last run along it starts, or where the leg ends (a landing, or the top of a climb)."""
    sp = leg.spans[-1] if leg.spans else None
    if sp is not None and sp.platform == leg.end.platform.id and sp.t1 >= leg.end.t - 1e-6:
        return sp.t0
    return leg.end.t


def _still(dot, last) -> bool:
    """Whether her minimap dot has stopped since the last glance (`last`: None if there was none)."""
    return last is not None and abs(dot[0] - last[0]) < STILL and abs(dot[1] - last[1]) < STILL


def _chain_text(legs, to_go: int, why: str) -> str:
    """A chain's moves for the log."""
    return (", ".join(f"{leg.move.kind}->{leg.move.dst}" if leg.move.kind != "hop" else f"hop@{leg.move.x:.0f}"
                      for leg in legs) + f"  [{to_go} moves to go; chain ended: {why}]")
