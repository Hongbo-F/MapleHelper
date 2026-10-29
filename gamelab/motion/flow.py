"""Moving like a player: several route moves turned into one continuous key timeline.

A player doesn't stop between moves. She runs along a platform and jumps on the way, keeps running
when she lands, jumps onto the next platform, jumps at a rope and holds up, and only looks around
when something unexpected happens. This module predicts her body over time with the game's physics
(physics.fly) and her walking as the engine has it (motion.body's closed forms, with the median lags:
speeding up from standing, running on and sliding after a release, the kick at a jump), and writes the
keys for a chain of moves as one plan:

  body now -> leg 1 (keys, where she'll be when) -> leg 2 -> ... -> one look at the end

Each move can be done a few ways (keep running on landing, land and stop, jump straight up, step back
for a run-up first), and the chain picks the way the next move can follow on from. Moves it can't
predict well enough (climbing down a rope, which needs her lined up on its top) end the chain; the
navigator has control.mover do those on their own, closed loop.

Monsters: each leg is checked against the monsters it's given (danger): her body's box (motion.bodies: the
client's, 22 either side of her feet and 65 tall) all along the way her feet go, her runs on the ground, her jumps,
falls and climbs, where she stands meanwhile (a jump's lift-off, a drop, a landing's slide, a rope's climb-over) and
where the chain ends, against each monster's box (its kind's widest and tallest frame) wherever it can be by then:
where it's walking to (from two looks), or anywhere it could get at its walking speed when only one look caught it,
on its own platform, whichever platform that is (a monster where a jump lands, on a platform a climb or a jump
passes, on one just over where she runs). A monster in the way of a run is hopped over: a running jump timed so the
top of the jump is over where it will most likely be, landing on the same platform clear of it, as a player does.
Where that can't be timed (too close, no room to land), the chain stops before that leg, and the navigator shoots
it (with an escort), goes another way or waits.

All positions in map units, times in seconds from the start of the plan.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from maplehelper.engine import LEFT, RIGHT

from .. import directions
from ..localization.mapdata import MapModel, Platform
from ..localization.monsters import MonsterTrack
from . import bodies, body
from .bodies import Arc, Climb
from .physics import (DROP_LAG, FALL_SPEED, GRAVITY, JUMP_HEIGHT, JUMP_LAG, JUMP_SPEED, LAGS, LAND_MARGIN, LAND_SLIDE,
                      RUN_LAG, SHOE, STOP_RUN, STOP_S, WALK_SPEED, fly, flight, hold_ms, sure_jump_ms, sure_landing)
from .route import DROP_STEP, JUMP_FROM, LANDING_ROOM, ROPE_ARMS, ROPE_CLEAR, Move, drop_clear, drop_spots

# Her measured lags and slides (RUN_LAG, JUMP_LAG, DROP_LAG, STOP_RUN, STOP_S, LAND_SLIDE) describe her body,
# so they're in physics, with how they were measured (still importable from here). Her walk, stops and
# takeoffs are worked out here from the engine's closed forms (motion.body), from whatever speed she has.
DROP_SLACK = 0.1     # s of extra timing error a drop adds
ARROW_LAG = 0.1      # s before a landing the arrow is let go for her to stop there: the game acts on it 68
                     # ms later (body.Lags; sd 12 with the wait for its step), so by 2.5 sd it has before she
                     # lands, and she slides from her landing speed (in the air with no arrow she slows by 1
                     # unit/s² at most)
AFTER_LANDING = 0.03  # s after a predicted landing before the next jump key (in the air it's ignored)
SLACK_PER_LEG = 0.025  # s of timing error each predicted landing may add
MAX_SLACK = 0.15     # ... counted up to this (beyond it waiting longer only makes her sluggish)
TAP = 0.04           # s a jump/down tap is held
PICK_EVERY = 0.15    # s between pick-up presses while she runs
MIN_RUNUP = 15.0     # units of run before a running jump from standing (keyed 155 ms after the arrow: full speed)
JUMP_FROM_FAR = 45.0  # running jumps take off between JUMP_FROM and this far before the end
# A chained jump lands with room to spare from where she really leaves the ground (Flow.sure_takeoff), or it's
# taken from another spot (every TAKEOFF_STEP units of its takeoffs, nearest the one picked first): LANDING_ROOM
# (9) in from the ends of the platform it lands on, and on it still if she leaves TAKEOFF_SLOP units early or
# late (and nothing nearer than that to her platform's end), 10% slower or at 0.85 x her speed (physics.
# sure_landing). The route checks each jump so at its own takeoff with 12 units' slop; chained, the pick up to
# 10 units further back (from along a platform; 12 off an end), the kick's wait and the landing before it moved
# the takeoff and used that up: of the 81 route jumps on this map, 40 had takeoffs the chains could pick that
# came down 2-7 units onto the platform and missed it from 8 units further back or 6% slower. In game on
# 2026-10-21 jumps from short platforms took off too early (the player's eye), and the jumps 32 -> 29, 29 ->
# 28 and 16 -> 14 that came down short were planned to land 5-16 units onto their platforms. Since the route
# places every jump's takeoff for room (route._placed: 55 jumps, each from along its platform, up a staircase
# step 15 units from the end), the chains take each from the move's takeoff or up to 10 before it, the least
# room 16.5 units; on 4 of them (jumps down from along a platform) only the move's own takeoff lands surely,
# and that's the one taken.
TAKEOFF_SLOP = 8.0
TAKEOFF_STEP = 2.0
# Standing this close to where a move wants her she doesn't step: a step that short is an arrow tap of
# under 50 ms (physics.hold_ms), and of three 50 ms taps in game two didn't move her and one moved her 7.1
# units. A longer step from standing is a hold the engine works out, whatever its length (body.hold_for:
# 1925 (t + 0.012)² units for a hold of t s up to about 95 ms, 7.7 + 149 t beyond), where it used to be a
# short hold below 35 units (STOP_RUN + 10) and a run with a release above.
STEP_ACCEPT = 8
JUST_PASSED = 2      # units: running, a spot she's this little past still counts as reached
EDGE_ROOM = 5        # units she keeps in from a platform's ends when stepping back or taking off
# Drops: at least route.ROPE_CLEAR (30) from the top of a rope hanging from her platform, and route.DROP_ROOM
# (8) from a foothold that forbids it; spots every route.DROP_STEP units (route.drop_spots).
DROP_JUMP_AFTER = 0.1  # s after down that a drop's jump goes in ...
DROP_DOWN_HOLD = 0.18  # ... and down is let go (as soon as the drop has started)
GRAB_LEAD = WALK_SPEED * JUMP_SPEED / GRAVITY  # ~41: jump this far before a rope to be at the top over it
APEX = JUMP_HEIGHT   # 77 units: how high a jump rises
# Two climbs traced on the minimap (60 fps): from grabbing the rope to standing on the platform at its
# top she covers the distance at an effective 130 units/s, the step onto the platform included
# (predicted 2.64 s and 5.20 s, seen 2.61 s and 5.09 s; a third: 4.49 s, seen 4.41 s). Up is let go a
# while past the predicted top. Climbing over the top is an animation: with the next arrow already
# held from just before the top (it does nothing on a rope), she walked off ~0.34 s after arriving;
# with the arrow pressed after arriving, only ~0.4 s after the press. Jumping off the rope sideways
# just below the top doesn't help: it has next to no lift (she fell 360 units). So the next move's
# arrow goes down just before the top, and she's modelled as starting to run 0.35 s after it.
ROPE_SPEED = 130.0
UP_EXTRA = 0.35
CLIMB_OVER = 0.23    # her body after a climb: a standing start this long after the top (+ RUN_LAG = 0.35)
PRE_ARROW = 0.1      # s before the predicted top that the next move's arrow is pressed ...
# ... unless it's a step: an arrow let go this soon after where the plan has her standing after the climb
# stays at its planned time. Pressed early, a step's walk starts with the climb-over, whenever that comes,
# while its release is timed from the plan's standing start. In game on 2026-10-21, of the 25 chains that
# climbed 36 -> 32 and stepped back on 32 for the jump onto 29 (a step of 18-30 units, held 104-170 ms) before
# 01:13, her dot went 8-30 units past the stop planned (575 or more, for 557-567) in 13, the jump off 32 left
# that much further on, and 17 of the 25 came down in the basin: 8 short of 29, 9 short of 28 off the jump
# after it, whose run on 29 started that much further on too. With the step pressed at its planned time (a
# live patch), the 6 chains after stopped at their spot or a few units short (her dot under 558 in 5) and
# landed every jump. A run after a climb (held longer) still goes down early: she gets going as the
# climb-over ends, as the plan has her.
STEP_EARLY_MAX_MS = 250
HOP_HALF = 15        # half her width as the hop was timed with in game (hop: 7 hops took no damage), not her body:
                     # what a monster touches her by in a plan is her box, the client's (motion.bodies: 22 either side,
                     # 65 tall), which danger goes by, as attack.danger does
DANGER_STEP = 0.05   # s between the moments danger checks her at (the shortest overlap of her box and a monster's:
                     # falling at the cap past one about 0.18 s, sideways at both their speeds about 0.4 s)
APEX_T = JUMP_SPEED / GRAVITY  # s from leaving the ground to the top of a jump (0.28)
HOP_HORIZON = 3.0    # s ahead a hop over a monster is looked for
END_LINGER = 1.0     # s she may stand where a chain ends: a monster mustn't reach her there either
MAX_LEGS = 5
EXPECT_MARGIN = 0.12  # s around a takeoff or landing when where she should be isn't said (Flow.expected)
MAX_SECONDS = 3.5    # a chain this long gets a fresh look before the next one
HOLD_MERGE_MS = 5    # a key held again within this many ms of its release stays down


@dataclass(frozen=True)
class Body:
    """Her predicted state at a moment of the plan."""
    t: float
    x: float
    platform: Platform
    run: int = 0          # running that way with the arrow held (+1 right, -1 left; the game sees it), or 0
    landed: bool = False  # she touches down at t (the next jump/drop key must come after that)
    slack: float = 0.0    # how far off t may be by now (prediction errors add up leg by leg)
    early_arrow: float | None = None  # after a climb: the next arrow may go down this early
    v: float | None = None  # her speed along the ground (units/s, + right): None for full speed that way if
                            # `run`, else standing. A jump lands her as fast as she left (and it's the
                            # kick that decides that), so she can run on from a landing below full speed

    def __post_init__(self):
        if self.v is None:
            object.__setattr__(self, "v", self.run * WALK_SPEED)

    @property
    def ready_at(self) -> float:
        """The earliest a jump or drop key counts: at t, or just after she has landed (a key pressed in
        the air is ignored), allowing for the timing error built up so far."""
        return self.t + (AFTER_LANDING + min(self.slack, MAX_SLACK) if self.landed else 0.0)


@dataclass(frozen=True)
class Span:
    """Her on the ground of one platform between two moments, moving linearly (for monster checks)."""
    platform: int
    t0: float
    t1: float
    x0: float
    x1: float

    def x_at(self, t: float) -> float:
        if self.t1 <= self.t0:
            return self.x0
        return self.x0 + (self.x1 - self.x0) * min(1.0, max(0.0, (t - self.t0) / (self.t1 - self.t0)))


@dataclass
class Leg:
    """One move as planned: her body where it starts and ends, her runs on the ground (spans), and the rest of the
    way her feet go, for touches (danger): her flights (air), her climbs up a rope (rope), and where she stands on the
    ground meanwhile (stands: waiting for a jump to lift off or a drop to go through, a landing and its slide, the
    climb-over at a rope's top), which nothing else plans by."""
    move: Move
    start: Body
    end: Body
    spans: list[Span] = field(default_factory=list)
    air: list[Arc] = field(default_factory=list)
    rope: list[Climb] = field(default_factory=list)
    stands: list[Span] = field(default_factory=list)


class Keys:
    """Key events for one plan, with the arrow tracked so a run carries on across legs. Events are
    [ms, key name, 'down' or 'up'], or [ms, key name, 'hold', end ms] for a hold (see sorted)."""

    def __init__(self, names: dict[str, str]):
        self.names = names        # left/right/jump/up/down -> key name
        self.events: list[list] = []
        self.arrow: int = 0       # -1 left held, +1 right held, 0 none

    def copy(self) -> "Keys":
        k = Keys(self.names)
        k.events, k.arrow = list(self.events), self.arrow
        return k

    def set_arrow(self, t: float, d: int) -> None:
        """From t, hold the arrow that way (+1 right, -1 left) or none (0): the one held is let go, and
        the new one pressed a millisecond later."""
        if d == self.arrow:
            return
        at = ms(t)
        if self.arrow:
            self.events.append([at, self.names[directions.toward(self.arrow)], "up"])
        if d:
            self.events.append([at + (1 if self.arrow else 0), self.names[directions.toward(d)], "down"])
        self.arrow = d

    def press_arrow_early(self, t: float, early: float) -> None:
        """Move the arrow press at t (if there is one) to `early`, if it starts a run: one let go
        STEP_EARLY_MAX_MS or less after t is a step, and stays where it is (see STEP_EARLY_MAX_MS)."""
        at, arrows = ms(t), (self.names[LEFT], self.names[RIGHT])
        for i, e in enumerate(self.events):
            if e[0] == at and e[2] == "down" and e[1] in arrows:
                up = min((u[0] for u in self.events if u[1] == e[1] and u[2] == "up" and u[0] > at), default=None)
                if up is None or up - at > STEP_EARLY_MAX_MS:
                    self.events[i] = [ms(early), e[1], e[2]]  # (a list of its own: copies share the old one)
                break

    def tap(self, t: float, key: str, hold: float = TAP) -> None:
        self.hold(t, t + hold, key)

    def hold(self, t0: float, t1: float, key: str) -> None:
        self.events.append([ms(t0), self.names[key], "hold", ms(t1)])

    def sorted(self) -> list[list]:
        """The key events in time order. Overlapping holds of a key merge into one: a key pressed
        again before an earlier hold of it ends stays down (otherwise the earlier release would let
        go of the later hold: a second rope climb's up was released 4 ms after it was pressed)."""
        out = [e for e in self.events if e[2] != "hold"]
        holds = [e for e in self.events if e[2] == "hold"]
        # Keys in the order of their first hold: presses at the same ms come out in this order.
        merged: dict[str, list[list[int]]] = {e[1]: [] for e in holds}
        for t0, name, _, t1 in sorted(holds, key=lambda e: e[0]):
            spans = merged[name]
            if spans and t0 <= spans[-1][1] + HOLD_MERGE_MS:
                spans[-1][1] = max(spans[-1][1], t1)
            else:
                spans.append([t0, t1])
        for name, spans in merged.items():
            for down, up in spans:
                out += [[down, name, "down"], [up, name, "up"]]
        return sorted(out, key=lambda e: (e[0], e[2] == "down"))


def ms(t: float) -> int:
    """Seconds as whole milliseconds (never negative), as key plans count them."""
    return max(0, round(t * 1000))


def _sign(v: float) -> int:
    return 1 if v > 0 else -1


def _rise_time(h: float) -> float:
    """s from leaving the ground until a jump straight up has risen h units (APEX_T past its top)."""
    if h <= 0:
        return 0.0
    if h >= APEX:
        return APEX_T
    return (JUMP_SPEED - math.sqrt(JUMP_SPEED ** 2 - 2 * GRAVITY * h)) / GRAVITY


def _brake(v: float, tau: float) -> tuple[float, float]:
    """(how far, speed) with no arrow for tau s from speed v (signed): she brakes to a stop."""
    if not v:
        return 0.0, 0.0
    s, t = _sign(v), min(max(0.0, tau), abs(v) / SHOE.drag)
    return v * t - s * SHOE.drag * t * t / 2, v - s * SHOE.drag * t


def _seen(b: Body, d: int) -> tuple[float, float, float]:
    """When the game sees arrow d (±1) held, from body b with it pressed at b.t if it wasn't held yet (it
    sees it LAGS.on later), and where she is and how fast she goes then: running the other way she runs on
    till then (that arrow, let go at b.t, is seen let go later still), standing she stands."""
    if d == b.run:
        return b.t, b.x, b.v
    if b.run:
        return (b.t + LAGS.on, b.x + body.run_x(LAGS.on, b.v, b.run, SHOE),
                body.run_speed(LAGS.on, b.v, b.run, SHOE))
    dx, v = _brake(b.v, LAGS.on)
    return b.t + LAGS.on, b.x + dx, v


def _ground(b: Body, d: int, t: float) -> tuple[float, float]:
    """Where she is along b's platform at t, and how fast she goes then (signed), with arrow d held from b.t
    on (0: none from then; pressed then if it wasn't already): the engine's closed forms (body), with the
    median lags. The platform's ends aren't looked at."""
    if not d:  # let go at b.t: she runs on until the game sees that, then brakes
        dx, v = (body.run_x(LAGS.off, b.v, b.run, SHOE), body.run_speed(LAGS.off, b.v, b.run, SHOE)) \
            if b.run else (0.0, b.v)
        tau = t - b.t - (LAGS.off if b.run else 0.0)
        if tau < 0:
            return b.x + body.run_x(t - b.t, b.v, b.run, SHOE), body.run_speed(t - b.t, b.v, b.run, SHOE)
        sx, v = _brake(v, tau)
        return b.x + dx + sx, v
    t0, x0, v0 = _seen(b, d)
    if t < t0:  # the arrow not seen yet: she goes on as she was
        if b.run:
            return b.x + body.run_x(t - b.t, b.v, b.run, SHOE), body.run_speed(t - b.t, b.v, b.run, SHOE)
        dx, v = _brake(b.v, t - b.t)
        return b.x + dx, v
    return x0 + body.run_x(t - t0, v0, d, SHOE), body.run_speed(t - t0, v0, d, SHOE)


def _tap_pick_up(k: Keys, legs: list[Leg]) -> None:
    """Pick up whatever she runs over, as a player taps the key all the way."""
    for leg in legs:
        for sp in leg.spans:
            t = sp.t0
            while t < sp.t1:
                k.tap(t, "pick")
                t += PICK_EVERY


class Flow:
    """Plans chains of route moves on a map as one key timeline each."""

    def __init__(self, model: MapModel, key_names: dict[str, str], rng=None):
        self.model = model
        self.key_names = key_names
        self.rng = rng  # varies where moves start (None: always the same, e.g. in tests)
        # Her invulnerability after a touch (perception.touches.INVULNERABLE_S): plan time (s from the chain's start)
        # before which no monster's touch counts on a leg (`danger`); 0, none. Whoever plans or watches a chain sets it
        # for that chain's time (control.navigator: Navigator._immunity).
        self.immune = 0.0

    def pick(self, near: float, far: float) -> float:
        """A spot between `near` (the quickest) and `far`, usually close to `near`; `near` without rng."""
        if self.rng is None:
            return near
        return near + (far - near) * self.rng.random() ** 2

    # --- running -----------------------------------------------------------------------------

    @staticmethod
    def run_to(b: Body, d: int, x: float) -> float | None:
        """When she's at x running that way from body b, the arrow pressed at b.t (None if it's
        behind her): from standing she starts LAGS.on after the key and speeds up to full speed in 7.9
        units, running the other way she first runs on, brakes and turns (body.run_time)."""
        dist = (x - b.x) * d
        if b.run == d and dist < 0:
            return b.t + dist / WALK_SPEED if dist >= -JUST_PASSED else None
        if dist < 0:
            return None
        t0, x0, v0 = _seen(b, d)
        tau = body.run_time((x - x0) * d, v0, d, SHOE)
        return None if tau is None else t0 + tau

    # --- one leg per move ------------------------------------------------------------------------
    #
    # Each move can be done a few ways ("variants"); the chain picks the one the next move can follow:
    #   run      keep running when she lands (the arrow stays held);
    #   stop     the same jump, but the arrow is let go just before she lands, so she stands still
    #            ~14 units on (for a next move that needs her standing, or going the other way);
    #   stand    (jump_up) stop under the platform above and jump straight up.

    VARIANTS = ("run", "stop", "stand")

    @staticmethod
    def liftoff(b: Body, d: int, t_off: float) -> tuple[float, float, float, float]:
        """When to press jump for her to leave the ground at t_off, running that way (d; 0: straight up)
        from body b with that arrow held from b.t, or as soon after as she can: never before she has landed
        (a press in the air is ignored), and with the arrow pressed at b.t never in the kick's blind spot
        (physics.sure_jump_ms: keyed there she could leave at 126 units/s, three of the game's steps into her
        run, short of where she's meant to land). Turning round first, she gets to a standstill |v|/1400 s later than standing would.
        Returns (the press, when she leaves the ground, where, and her sideways speed then: what she has
        on the ground, with the kick if that's below 0.8 × her walking speed, body.takeoff)."""
        press = max(t_off - JUMP_LAG, b.ready_at)
        if d and b.run != d:  # a new arrow: how soon after it the jump comes decides how fast she leaves
            turn = abs(b.v) / SHOE.acc if b.run == -d else 0.0
            press = b.t + turn + sure_jump_ms((press - b.t - turn) * 1000) / 1000
        t = press + JUMP_LAG
        x, v = _ground(b, d, t)
        return press, t, x, body.takeoff(v, d, SHOE)

    def press_jump(self, k: Keys, b: Body, d: int, t_off: float) -> tuple[float, float, float]:
        """Press jump for her to leave the ground at t_off (liftoff). Returns when she leaves the ground,
        where, and her sideways speed then."""
        press, t, x, vx = self.liftoff(b, d, t_off)
        k.tap(press, "jump")
        return t, x, vx

    def leg(self, m: Move, b: Body, k: Keys, variant: str = "run") -> Leg | None:
        """Move m done the `variant` way from body b, its keys added to k; None where that way doesn't
        work. Climbing down a rope (and a hop, planned by `hop`) isn't predicted: it ends a chain."""
        if b.platform.id != m.src:
            return None
        plan = {"jump": self._jump, "walk_off": self._walk_off, "jump_up": self._jump_up, "drop": self._drop,
                "climb": self._climb}.get(m.kind)
        return None if plan is None else plan(m, b, k, variant)

    def _landing(self, m: Move, b: Body, k: Keys, variant: str, t_land: float, lx: float, land: Platform,
                 d: int, v_land: float, spans: list[Span], arc: Arc | None = None) -> Leg:
        """End a jump/fall leg, landing at v_land (signed): running on (arrow held) or, for 'stop',
        standing a little further on, where she slides to from v_land (13.9 units from full speed). Either
        way she stands where she lands until the game's next step first (body.landing_wait: 15 ms on
        average), so a run on from a landing, and the jump at the end of it, is that much behind the arc's
        landing (in game 3.3 units at full speed, 22 ms: body.LANDING_ENDS_STEP). `arc`: the flight, kept with the
        leg, as are the landing and its slide (Leg.stands)."""
        slack, air = b.slack + SLACK_PER_LEG, [arc] if arc is not None else []
        t_on = t_land + body.landing_wait()
        if variant == "stop" and d:
            k.set_arrow(t_land - ARROW_LAG, 0)  # the game sees it just before she lands
            x = lx + _sign(v_land) * body.slide(v_land, SHOE)
            if not land.covers(x, -3):  # she'd slide off the end: she stops 3 units in from it
                x = min(max(x, land.left + 3), land.right - 3)
            end = Body(t_on + abs(v_land) / SHOE.drag, x, land, slack=slack)
            return Leg(m, b, end, spans, air, stands=[Span(land.id, t_land, end.t, lx, x)])
        return Leg(m, b, Body(t_on, lx, land, run=d, landed=True, slack=slack, v=v_land), spans, air,
                   stands=[Span(land.id, t_land, t_on, lx, lx)])

    def _runup(self, b: Body, d: int, takeoff: float, k: Keys) -> Body | None:
        """Make sure she can run at `takeoff` going that way: stop if she's running the other way,
        step back if she's standing too close. Returns her body ready to run (or None: no room)."""
        if b.run == d:
            return b
        a = b.platform
        if b.run == -d:  # stop first
            k.set_arrow(b.t, 0)
            b = Body(b.t + body.stop_time(b.v, SHOE, LAGS), b.x + b.run * body.stop_distance(b.v, SHOE, LAGS), a,
                     slack=b.slack)
        room = (takeoff - b.x) * d
        if room >= MIN_RUNUP:
            return b
        back = b.x - d * (MIN_RUNUP + 8 - room)  # a run-up 8 units longer than the least, give or take 4
        if not a.covers(back, -EDGE_ROOM):
            return None
        step = self._stop_within(b, back - 4, back + 4, back, k)
        if step is None:
            return None
        return Body(step[0], step[1], a, slack=b.slack)

    def _jump(self, m: Move, b: Body, k: Keys, variant: str) -> Leg | None:
        """A running jump, from the first of its takeoffs (_takeoffs) that lands with room to spare where she
        really leaves the ground (sure_takeoff)."""
        if variant == "stand":
            return None
        a, d = b.platform, directions.sign(m.direction)
        edge = a.right if d > 0 else a.left
        for takeoff in self._takeoffs(m, b, d, edge):
            trial = k.copy()
            leg = self._jump_from(m, b, trial, variant, takeoff)
            if leg is not None:
                k.events, k.arrow = trial.events, trial.arrow
                return leg
        return None

    def _takeoffs(self, m: Move, b: Body, d: int, edge: float) -> list[float]:
        """Where a running jump may take off, the one to try first first: from along the platform, the move's
        takeoff or up to 10 units before it; off the end, JUMP_FROM to JUMP_FROM_FAR before it (players jump
        from a little further back too). With rng the first is picked at random, usually near the move's own,
        the rest every TAKEOFF_STEP units by how near they are to it. Running that way past one already, she
        jumps now instead (if she's more than 12 units from the end)."""
        base, near, far = (m.takeoff, 0.0, 10.0) if m.takeoff is not None else (edge, JUMP_FROM, JUMP_FROM_FAR)
        first = self.pick(near, far)
        back = [first] + sorted((near + i * TAKEOFF_STEP for i in range(int((far - near) / TAKEOFF_STEP) + 1)),
                                key=lambda v: abs(v - first))
        out: list[float] = []
        for v in back:
            x = base - d * v
            if b.run == d and (x - b.x) * d < -JUST_PASSED:
                if (edge - b.x) * d <= 12:
                    continue
                x = b.x  # already past this spot, still room: now
            if all(abs(x - o) > 0.5 for o in out):
                out.append(x)
        return out

    def _jump_from(self, m: Move, b: Body, k: Keys, variant: str, takeoff: float) -> Leg | None:
        """The running jump of _jump from `takeoff` (or from where she really leaves the ground, which can be
        later and further on), if it lands surely (sure_takeoff); its keys added to k."""
        a, d = b.platform, directions.sign(m.direction)
        edge = a.right if d > 0 else a.left
        start = self._runup(b, d, takeoff, k)
        if start is None:
            return None
        t = self.run_to(start, d, takeoff)
        if t is None:
            return None
        k.set_arrow(start.t, d)
        # Later than planned, and further on, if she had to land first or wait out the kick's blind spot.
        t_off, takeoff, vx = self.press_jump(k, start, d, t)
        if (edge - takeoff) * d < TAKEOFF_SLOP:
            return None
        land, lx, ta, v_land = flight(self.model, takeoff, a.y_at(takeoff), vx, -JUMP_SPEED, group=a.group)
        if land is None or land.id != m.dst or not self.sure_takeoff(a, takeoff, vx, land, lx):
            return None
        return self._landing(m, b, k, variant, t_off + ta, lx, land, d, v_land,
                             [Span(a.id, b.t, t_off, b.x, takeoff)],
                             Arc(t_off, t_off + ta, takeoff, a.y_at(takeoff), lx, land.y_at(lx), -JUMP_SPEED))

    def sure_takeoff(self, a: Platform, x: float, vx: float, land: Platform, lx: float) -> bool:
        """Whether a jump off platform a from x at vx, which lands on `land` at lx, does so with room to spare:
        LANDING_ROOM in from its ends, and still on it if she leaves TAKEOFF_SLOP units early or late, 10%
        slower or as slow as physics.SLOW_TAKEOFF (sure_landing)."""
        return land.covers(lx - LANDING_ROOM) and land.covers(lx + LANDING_ROOM) and \
            sure_landing(self.model, x, a.y_at(x), vx, -JUMP_SPEED, land, slop=TAKEOFF_SLOP, group=a.group)

    def _walk_off(self, m: Move, b: Body, k: Keys, variant: str) -> Leg | None:
        if variant == "stand":
            return None
        a, d = b.platform, directions.sign(m.direction)
        edge = a.right if d > 0 else a.left
        start = self._runup(b, d, edge, k) if b.run == -d else b
        if start is None:
            return None
        t = self.run_to(start, d, edge)
        if t is None:
            return None
        _, v = _ground(start, d, t)  # (from standing close to the end: not yet at full speed)
        land, lx, ta, v_land = flight(self.model, edge + d, a.y_at(edge) - 1, v, 0.0, group=a.group)
        if land is None or land.id != m.dst:
            return None
        k.set_arrow(start.t, d)
        return self._landing(m, b, k, variant, t + ta, lx, land, d, v_land, [Span(a.id, b.t, t, b.x, edge)],
                             Arc(t, t + ta, edge + d, a.y_at(edge) - 1, lx, land.y_at(lx), 0.0))

    def _jump_up(self, m: Move, b: Body, k: Keys, variant: str, at: float | None = None) -> Leg | None:
        a, q = b.platform, self.model.platform(m.dst)
        if variant != "stand" and at is None:
            # A running jump on the way, if one lands on the platform above with room to spare. Takeoffs
            # every 5 units (up to 400 on): running, from where she is; from standing, towards the
            # move's spot, from 15 units on.
            d = b.run if b.run else (_sign(m.x - b.x) if abs(m.x - b.x) > 15 else 0)
            if not d:
                return None
            first = b.x if b.run == d else b.x + d * 15
            x = first
            while a.covers(x, -EDGE_ROOM) and abs(x - first) <= 400:
                t = self.run_to(b, d, x)
                if t is None:
                    return None
                press, t_off, x_off, vx = self.liftoff(b, d, t)
                if abs(t_off - t) < 0.02:  # leaves the ground where planned
                    land, lx, ta, v_land = flight(self.model, x_off, a.y_at(x_off), vx, -JUMP_SPEED,
                                                  group=a.group)
                    if land is q and q.covers(lx - d * LAND_MARGIN) and q.covers(lx + d * 30):
                        k.set_arrow(b.t, d)
                        k.tap(press, "jump")
                        return self._landing(m, b, k, variant, t_off + ta, lx, q, d, v_land,
                                             [Span(a.id, b.t, t_off, b.x, x_off)],
                                             Arc(t_off, t_off + ta, x_off, a.y_at(x_off), lx, q.y_at(lx), -JUMP_SPEED))
                x += d * 5
            return None
        # Stop anywhere under it and jump straight up.
        lo, hi = max(a.left, q.left) + LAND_MARGIN, min(a.right, q.right) - LAND_MARGIN
        near = min(max(b.x, lo), hi)
        # With rng, from the nearest spot under it towards the far end of the overlap; else the planner's (or `at`:
        # where a landing with clearance is, _spread).
        prefer = at if at is not None else self.pick(near, hi if near - lo < hi - near else lo) if self.rng else m.x
        if at is not None and SPREAD_FROM_SPOT:
            # (_spread's: from that spot itself. As first built `at` was only where she'd rather stop, and standing or
            # sliding to a stop anywhere under the platform she jumped from there: every spot gave the same jump.)
            lo = hi = min(max(at, lo), hi)
        stop = self._stop_within(b, lo, hi, prefer, k)
        if stop is None:
            return None
        t, x = stop
        land, lx, ta = fly(self.model, x, a.y_at(x), 0.0, -JUMP_SPEED, group=a.group)
        if land is not q:
            return None
        stopped = t
        t, _, _ = self.press_jump(k, Body(t, x, a, landed=b.landed and t == b.t, slack=b.slack), 0, t + JUMP_LAG)
        return Leg(m, b, Body(t + ta, lx, q, landed=True, slack=b.slack + SLACK_PER_LEG), [Span(a.id, b.t, t, b.x, x)],
                   [Arc(t, t + ta, x, a.y_at(x), lx, q.y_at(lx), -JUMP_SPEED)], stands=[Span(a.id, stopped, t, x, x)])

    def _drop(self, m: Move, b: Body, k: Keys, variant: str, at: float | None = None) -> Leg | None:
        if variant != "run":
            return None
        a, below = b.platform, self.model.platform(m.dst)
        # Not at the top of a rope hanging from her platform (down there climbs onto the rope), nor where its
        # foothold forbids dropping through (forbidFallDown), or near one.
        clear = drop_spots(self.model, a, below)
        if not clear:
            return None
        prefer = min(clear, key=lambda x: (abs(x - b.x), abs(x - m.x)) if at is None else abs(x - at))
        if self.rng is not None and at is None:  # not always the first spot: somewhere further over it now and then
            far = max(clear, key=lambda x: abs(x - b.x))
            # (pick draws again for every spot, around the nearest one)
            prefer = min(clear, key=lambda x: abs(x - self.pick(prefer, far)))
        # She may stop anywhere in the clear stretch around that spot; with `at` (_spread's), at that spot itself. (As
        # first built at any of _spread's spots standing anywhere in the stretch she dropped from where she stood: live on
        # 2026-10-28 at 00:38:36 the drop 21 -> 23 from 42, by a Horny Mushroom, every spot from 42, where from 60, a
        # step of 18, it landed clear of it.)
        spots = set(clear)
        run_lo = run_hi = prefer
        while not (at is not None and SPREAD_FROM_SPOT) and run_lo - DROP_STEP in spots:
            run_lo -= DROP_STEP
        while not (at is not None and SPREAD_FROM_SPOT) and run_hi + DROP_STEP in spots:
            run_hi += DROP_STEP
        stop = self._stop_within(b, run_lo, run_hi, prefer, k)
        if stop is not None and not drop_clear(self.model, a, stop[1], below):
            # Standing a tap's step outside the stretch (STEP_ACCEPT) does for other moves, not here: within
            # ROPE_CLEAR of a rope's top (down there climbs onto it). She steps into the stretch first, to its
            # nearest end.
            stop = self._step(b, min(max(b.x, run_lo), run_hi), k)
        if stop is None:
            return None
        t, x = stop
        land, lx, ta = fly(self.model, x, a.y_at(x) + 1, 0.0, 0.0, skip=a, group=a.group)
        if land is None or land.id != m.dst:
            return None
        t = max(t, b.ready_at)
        k.hold(t, t + DROP_DOWN_HOLD, "down")
        k.tap(t + DROP_JUMP_AFTER, "jump")
        t_off = t + DROP_JUMP_AFTER + DROP_LAG
        # Lying down there till she goes through (her box taken as standing: lying, the client's is 46 wide on the side
        # she faces and 31 tall), then the fall.
        return Leg(m, b, Body(t_off + ta, lx, land, landed=True, slack=b.slack + SLACK_PER_LEG + DROP_SLACK),
                   [Span(a.id, b.t, t, b.x, x)], [Arc(t_off, t_off + ta, x, a.y_at(x) + 1, lx, land.y_at(lx), 0.0)],
                   stands=[Span(a.id, t, t_off, x, x)])

    def _climb(self, m: Move, b: Body, k: Keys, variant: str) -> Leg | None:
        if variant != "run":
            return None
        a, top = b.platform, self.model.platform(m.dst)
        rx, gap = m.rope_x, a.y_at(m.rope_x) - m.rope_bottom
        # Out of reach of a jump (and her arms), or by a portal: up there would take it.
        if gap > APEX + 10 or self.model.portal_near(a, rx):
            return None
        if abs(rx - b.x) <= 4 and not b.run:
            # Straight up under it: she grabs it once her feet are ROPE_ARMS below its bottom (route, the
            # simulator's world.ARMS), on her feet at its bottom (grab_h). Timed 0.15 s after leaving the ground,
            # the plan had her at the top about 0.1 s after she got there under the ropes 40 above 6, 30 and 36
            # (grabbed 0.045 s up), and a run off the top to a jump taking off 15 units from the end (route.
            # TAKEOFF_EDGE) went off the end before its key: 11 of seeds 1-24 from under 723's rope to 1. When the
            # game grabs isn't measured: with her feet at the rope's bottom it would be 0.085 s up, 0.04 s later.
            t, grab_h, ran_to = b.t, min(APEX, max(0.0, gap)), b.x
            t_lift = self.press_jump(k, b, 0, t + JUMP_LAG)[0]
            t_rope = t_lift + _rise_time(gap - ROPE_ARMS)
            arc = Arc(t_lift, t_rope, b.x, a.y_at(b.x), rx, a.y_at(rx) - grab_h, -JUMP_SPEED)
            waits = [Span(a.id, t, t_lift, b.x, b.x)]  # (the jump's lift-off)
        else:
            d = _sign(rx - b.x)
            ran_to = rx - d * GRAB_LEAD
            start = self._runup(b, d, ran_to, k)
            if start is None:
                return None
            t = self.run_to(start, d, ran_to)
            if t is None:
                return None
            k.set_arrow(start.t, d)
            t, ran_to, vx = self.press_jump(k, start, d, t)
            t_rope = t + max(0.0, (rx - ran_to) * d) / max(abs(vx), 1.0)  # over the rope (at the top of the jump)
            grab_h = APEX
            k.set_arrow(t_rope + 0.12, 0)  # let go of the arrow once she's on the rope
            arc, waits = Arc(t, t_rope, ran_to, a.y_at(ran_to), rx, a.y_at(rx) - grab_h, -JUMP_SPEED), []
        t_top = t_rope + max(0.0, (a.y_at(rx) - grab_h) - m.rope_top) / ROPE_SPEED
        k.hold(t, t_top + UP_EXTRA, "up")
        end = Body(t_top + CLIMB_OVER, rx, top, slack=b.slack + SLACK_PER_LEG, early_arrow=t_top - PRE_ARROW)
        # Up the rope from where she grabs it to its top's platform (the climb-over: standing there, Leg.stands).
        return Leg(m, b, end, [Span(a.id, b.t, t, b.x, ran_to)], [arc],
                   [Climb(t_rope, t_top, rx, a.y_at(rx) - grab_h, m.rope_top)],
                   waits + [Span(top.id, t_top, end.t, rx, rx)])

    def _stop_within(self, b: Body, lo: float, hi: float, prefer: float, k: Keys) -> tuple[float, float] | None:
        """Bring her to a stop anywhere in [lo, hi] (nearest `prefer` when she has a choice): right
        away if she's there (or standing a tap's step from it, STEP_ACCEPT), by releasing the arrow now
        if that leaves her there, else at the spot. Returns (when she's stopped, where)."""
        if lo > hi:
            return None
        target = min(max(prefer, lo), hi)
        if not b.run:
            if lo - STEP_ACCEPT < b.x < hi + STEP_ACCEPT:
                return b.t, b.x
            return self._step(b, target, k)
        natural = b.x + b.run * body.stop_distance(b.v, SHOE, LAGS)
        if lo <= natural <= hi:
            k.set_arrow(b.t, 0)
            return b.t + body.stop_time(b.v, SHOE, LAGS), natural
        if (target - natural) * b.run > 0:  # the spot is further on
            return self._stop_at(b, target, k)
        return None

    @staticmethod
    def _step(b: Body, x: float, k: Keys) -> tuple[float, float]:
        """From standing, step to x: hold the arrow just long enough (physics.hold_ms, the engine's).
        Returns (when she's stopped, where)."""
        held = hold_ms(abs(x - b.x))
        d = _sign(x - b.x)
        k.set_arrow(b.t, d)
        k.set_arrow(b.t + held / 1000, 0)
        return b.t + body.hold_time(held, SHOE, LAGS), b.x + d * body.hold_distance(held, SHOE, LAGS)

    @staticmethod
    def _stop_at(b: Body, x: float, k: Keys) -> tuple[float, float] | None:
        """Running that way, bring her to a stop at x, further on than she'd slide to if let go now:
        let go of the arrow once she's held it long enough for that (body.held_for: a stop's length
        early at full speed). Returns (when she's stopped, where)."""
        d = b.run
        seen = body.held_for((x - b.x) * d, b.v * d, SHOE)  # how long the game must see it held from now
        t = b.t + seen - LAGS.off
        if t < b.t:
            return None
        k.set_arrow(b.t, d)  # (held already in this plan; a plan that follows on from a run presses it again)
        k.set_arrow(t, 0)
        _, v = _ground(b, d, b.t + seen)
        return b.t + seen + abs(v) / SHOE.drag, x

    # --- monsters ------------------------------------------------------------------------------

    def path(self, leg: Leg, linger: float = 0.0) -> list[tuple[float, float, float]]:
        """Where her feet are over a leg, (t, map x, map y), every DANGER_STEP s of each part of it and at each part's
        end, in time order: her runs and where she stands on the ground (at the platform's height there), her
        flights and her climbs; and, with `linger`, standing that long where it ends."""
        ground = list(leg.spans) + list(leg.stands)
        if linger:
            e = leg.end
            ground.append(Span(e.platform.id, e.t, e.t + linger, e.x, e.x))
        pieces = [(sp.t0, sp.t1, sp) for sp in ground] + [(a.t0, a.t1, a) for a in leg.air] \
            + [(c.t0, c.t1, c) for c in leg.rope]
        out = []
        for t0, t1, piece in sorted(pieces, key=lambda p: (p[0], p[1])):
            n = max(0, math.ceil((t1 - t0) / DANGER_STEP - 1e-9))
            for i in range(n + 1):
                t = min(t1, t0 + i * DANGER_STEP)
                if isinstance(piece, Span):
                    x = piece.x_at(t)
                    out.append((t, x, self.model.platform(piece.platform).y_at(x)))
                elif isinstance(piece, Arc):
                    out.append((t, *piece.at(t, GRAVITY, FALL_SPEED)))
                else:
                    out.append((t, *piece.at(t)))
        return out

    def danger(self, leg: Leg, monsters: list[MonsterTrack], linger: float = 0.0,
               after: float | None = None) -> tuple[MonsterTrack, float] | None:
        """The first monster that could touch her on this leg (and, if `linger`, while she stands that long where it
        ends), and when: her box (motion.bodies) wherever her feet are (path), against each monster's box wherever
        it can be by then (MonsterTrack.reach: along its own platform, whichever that is). None if none could.
        `after`: only from that plan time on (the leg's rest, for a plan running: Navigator._coming); and none before
        `immune` (her invulnerability after a touch: a leg may pass a monster while it lasts)."""
        if not monsters:
            return None
        path = self.path(leg, linger)
        after = self.immune if after is None else max(after, self.immune)
        if after > 0:
            path = [p for p in path if p[0] >= after]
        if not path:
            return None
        lo_x, hi_x = min(x for _, x, _ in path), max(x for _, x, _ in path)
        lo_y, hi_y = min(y for _, _, y in path), max(y for _, _, y in path)
        near = []  # only those whose box can get near where she goes, over the leg (a walker turning back included)
        for mo in monsters:
            q = self.model.platform(mo.platform)
            a, b = mo.reach_over(path[0][0], path[-1][0], q)
            reach_x = mo.half + bodies.HER_HALF
            if a - reach_x < hi_x and b + reach_x > lo_x \
                    and hi_y > q.top - mo.tall and lo_y - bodies.HER_TALL < bodies.low_end(q):
                near.append((mo, q))
        for t, x, y in path:
            for mo, q in near:
                lo, hi = mo.reach(t, q)
                if bodies.touches(x, y, lo, hi, mo.half, q.y_at(min(max(x, lo), hi)), mo.tall):
                    return mo, t
        return None

    def in_the_way(self, m: Move, b: Body, monsters: list[MonsterTrack],
                   linger: float = 0.0) -> tuple[MonsterTrack, float] | None:
        """The monster that stops a chain before move m from body b (chain's why 'monster'), and when it could touch
        her: on the first of the move's ways (VARIANTS) that can be done at all. None if any way meets none."""
        first = None
        for v in self.VARIANTS:
            leg = self.leg(m, b, Keys(self.key_names), v)
            if leg is not None:
                hit = self.danger(leg, monsters, linger)
                if hit is None:
                    return None
                first = first or hit
        return first

    # --- hopping over monsters ------------------------------------------------------------------

    def hop(self, b: Body, mo: MonsterTrack, d: int, k: Keys) -> Leg | None:
        """A running jump over a monster in her way on her platform, landing on the same platform.

        By the body sizes in the game data her jump can't clear a pig or slime outright (only a
        snail): timed with the top of the jump over them, 6 of 8 hops over slimes walking at her
        touched them (4 for damage, 2 MISS). So the jump is timed so she can't touch it on the way
        up: she leaves the ground just outside its reach plus the distance it takes her to climb to
        its height. Any contact then comes past its middle on the way down, where the knockback (away
        from the monster) pushes her on instead of back. Timed this way, 7 hops over monsters standing
        or walking at her took no damage (4 touches were MISS: she's a thief, with high evasion)."""
        a = b.platform
        if mo.platform != a.id or not d:
            return None
        if mo.vx is not None and mo.vx * d > 10:
            # Walking away from her, she closes on it at only ~80 units/s: she comes down short of
            # its middle and gets pushed back (2 of 3 such hops took damage, one pushed back 33 units).
            return None

        def her_x(t: float) -> float:
            return _ground(b, d, t)[0]

        closing = WALK_SPEED + max(0.0, -(mo.vx or 0.0) * d)  # how fast the gap shrinks
        disc = JUMP_SPEED ** 2 - 2 * GRAVITY * mo.height
        rise = (JUMP_SPEED - disc ** 0.5) / GRAVITY if disc > 0 else APEX_T  # s to clear its height
        gap = mo.half + HOP_HALF + closing * rise + 3   # distance to its middle at takeoff
        t = b.t
        while t < b.t + HOP_HORIZON and (mo.at(t, a) - her_x(t)) * d > gap:
            t += 0.01
        if t >= b.t + HOP_HORIZON:
            return None  # it keeps ahead of her
        earliest = b.ready_at + JUMP_LAG
        if t < earliest:
            return None  # already too close to time it
        press, t_off, x_off, vx = self.liftoff(b, d, t)  # (from standing: maybe later, out of the kick's blind spot)
        if (mo.at(t_off, a) - x_off) * d < gap - 15 or not a.covers(x_off, -EDGE_ROOM):
            return None  # too close by then, or no room
        land, lx, ta, v_land = flight(self.model, x_off, a.y_at(x_off), vx, -JUMP_SPEED, group=a.group)
        if land is not a or (lx - mo.at(t_off + ta, a)) * d < 0 or not self.sure_takeoff(a, x_off, vx, a, lx):
            return None  # off the platform (or too near its end to be sure), or she'd come down short of it
        k.set_arrow(b.t, d)
        k.tap(press, "jump")
        move = Move("hop", a.id, a.id, x_off, lx, directions.toward(d))
        # Running on from where she lands, as any landing (_landing): from the game's next step after it.
        t_on = t_off + ta + body.landing_wait()
        return Leg(move, b, Body(t_on, lx, a, run=d, landed=True, slack=b.slack + SLACK_PER_LEG, v=v_land),
                   [Span(a.id, b.t, t_off, b.x, x_off)],
                   [Arc(t_off, t_off + ta, x_off, a.y_at(x_off), lx, a.y_at(lx), -JUMP_SPEED)],
                   stands=[Span(a.id, t_off + ta, t_on, lx, lx)])

    def _hop_then(self, m: Move, b: Body, k: Keys, variant: str, monsters: list[MonsterTrack],
                  linger: float) -> tuple[Leg, Leg, Keys] | None:
        """The move done with a hop over the monster in the way of its run, if that works out."""
        trial = k.copy()
        leg = self.leg(m, b, trial, variant)
        hit = None if leg is None else self.danger(leg, monsters, linger)
        if hit is None:
            return None
        mo, t_hit = hit
        span = next((sp for sp in leg.spans if sp.platform == mo.platform and sp.t0 - 1e-6 <= t_hit <= sp.t1 + 1e-6), None)
        d = 0 if span is None or abs(span.x1 - span.x0) < 1 else (1 if span.x1 > span.x0 else -1)
        hk = k.copy()
        jump = self.hop(b, mo, d, hk)
        others = [o for o in monsters if o is not mo]  # it's behind her once she's over it
        if jump is None or self.danger(jump, others):
            return None
        after = self.leg(m, jump.end, hk, variant)
        if after is None or self.danger(after, others, linger):
            return None
        return jump, after, hk

    # --- landings with clearance (from 2026-10-27) -----------------------------------------------

    SPREAD = 40.0        # guess: units either way of a move's own takeoff (or spot) that _spread tries ...
    SPREAD_STEP = 5.0    # ... this far apart (a running jump's takeoff)
    JUMP_UP_STEP = 10.0  # ... a jump straight up's, from anywhere under the platform above (its landing moves as much)
    DROP_EVERY = 5       # ... and every this many of a drop's spots (route.drop_spots: DROP_STEP apart)

    def _priced(self, m: Move, b: Body, k: Keys, order: tuple, options: list, monsters, linger: float,
                price: Callable[[Leg], float], alts, goal: tuple[int, float] | None, last: bool = True,
                spread: bool = True) -> list:
        """The ways of doing move m from body b, the cheapest first by when she stands where she lands, the landing's
        `price` and her walk on from there to `goal` (WALK_SPEED: where she's sent, or the next move's spot): where the
        first of `options` (chain's, the ways keeping clear of touches) lands with no price, `options` as they are;
        else they and, for `m` and the route's other moves between the same platforms (`alts`: their own ways, and
        with `spread` the ways from other takeoffs, _spread), each kept clear of touches as chain's are (danger,
        `linger`), and as they are still unless one of those lands priced less than the first. The chain's `last` leg
        (she stops where it ends) is judged where she touches down and where she comes to stand, the dearer of the two:
        a running one's stop and slide on (_at_rest; until the reviews of 2026-10-27 at its touchdown alone, 14 units
        short, so 'run' won the ties by its slide's 0.19 s); another (she runs on from it) by `price(leg, last=False)`:
        the price of a landing she passes through."""
        def priced(leg: Leg) -> float:
            if not last:  # (a leg she runs on from: the price of a landing passed through, `last` False)
                return price(leg, last=False)
            return max(price(leg), price(_at_rest(leg))) if leg.end.run else price(leg)

        def cost(leg: Leg) -> float:
            end = _at_rest(leg) if last else leg
            walk = (abs(end.end.x - goal[1]) / WALK_SPEED if goal is not None and goal[0] == end.end.platform.id
                    else 0.0)
            return end.end.t + priced(leg) + walk
        first = priced(options[0][0])
        if first <= 0:
            return options
        found = list(options)
        for mm in [m] + [a for a in (alts(m) if alts is not None else ()) if a is not m]:
            for v in order:
                tried = [] if mm is m else [(self.leg(mm, b, trial, v), trial) for trial in (k.copy(),)]
                if spread:
                    tried += self._spread(mm, b, k, v)
                for leg, trial in tried:
                    if leg is not None and not self.danger(leg, monsters, linger=linger):
                        found.append((leg, trial))
        if min(priced(leg) for leg, _ in found) >= first - 1e-9:
            return options  # (nothing lands any cheaper: as it was)
        return sorted(found, key=lambda o: cost(o[0]))

    def _spread(self, m: Move, b: Body, k: Keys, variant: str) -> list[tuple[Leg, Keys]]:
        """Move m done the `variant` way from takeoffs other than its own, each with its keys (added to a copy of k):
        a running jump from SPREAD either way of its takeoff (from along the platform) or its end's (sure_takeoff, as
        _jump's), a jump straight up from anywhere under the platform above, a drop from anywhere over the one below,
        every SPREAD_STEP, JUMP_UP_STEP and DROP_EVERY spots; none for the moves that have one way only (a walk-off, a
        climb: their own way is chain's, or _priced's for the route's other moves)."""
        a = b.platform
        if a.id != m.src:
            return []
        out = []
        if m.kind == "jump" and variant != "stand":
            d = directions.sign(m.direction)
            edge = a.right if d > 0 else a.left
            base = m.takeoff if m.takeoff is not None else edge - d * (JUMP_FROM + JUMP_FROM_FAR) / 2
            n = int(self.SPREAD // self.SPREAD_STEP)
            for i in range(-n, n + 1):
                x = base + i * self.SPREAD_STEP
                if not a.covers(x, -EDGE_ROOM):
                    continue
                trial = k.copy()
                leg = self._jump_from(m, b, trial, variant, x)
                if leg is not None:
                    out.append((leg, trial))
        elif m.kind == "jump_up" and variant == "stand":
            q = self.model.platform(m.dst)
            lo, hi = max(a.left, q.left) + LAND_MARGIN, min(a.right, q.right) - LAND_MARGIN
            x = lo
            while x <= hi + 1e-6:
                trial = k.copy()
                leg = self._jump_up(m, b, trial, variant, at=x)
                if leg is not None:
                    out.append((leg, trial))
                x += self.JUMP_UP_STEP
        elif m.kind == "drop" and variant == "run":
            for x in drop_spots(self.model, a, self.model.platform(m.dst))[::self.DROP_EVERY]:
                trial = k.copy()
                leg = self._drop(m, b, trial, variant, at=x)
                if leg is not None:
                    out.append((leg, trial))
        return out

    # --- following a chain ---------------------------------------------------------------------

    @staticmethod
    def expected(legs: list[Leg], t: float) -> int | None:
        """The platform she should be on (on the ground) t s into the plan, or None while she's in
        the air, on a rope, or within a moment of a takeoff or landing."""
        for i, leg in enumerate(legs):
            m0 = EXPECT_MARGIN + min(leg.start.slack, 2 * MAX_SLACK)  # timings drift as the chain goes on
            for sp in leg.spans:
                if sp.t0 + m0 <= t <= sp.t1 - m0:
                    return sp.platform
            m1 = EXPECT_MARGIN + min(leg.end.slack, 2 * MAX_SLACK)
            nxt = legs[i + 1].spans[0].t0 if i + 1 < len(legs) and legs[i + 1].spans else None
            # Where the leg ends: for good after the last leg, else from a landing until the next run.
            if leg.end.t + m1 <= t and (nxt is None or (t <= nxt - m1 and leg.end.landed)):
                return leg.end.platform.id
        return None

    @staticmethod
    def airborne(legs: list[Leg], t: float) -> bool:
        """Whether she's in the air t s into the plan: from leaving the ground on a jump, a hop, a walk-off or a
        drop until a moment after she lands (EXPECT_MARGIN, and the slack built up). A climb's flight onto the
        rope isn't counted: once there she's on the rope, and standing below it, she missed it."""
        for leg in legs:
            if leg.move.kind == "climb":
                continue
            off = leg.spans[-1].t1 if leg.spans else leg.start.t
            if off <= t <= leg.end.t + EXPECT_MARGIN + min(leg.end.slack, 2 * MAX_SLACK):
                return True
        return False

    @staticmethod
    def span_at(legs: list[Leg], t: float) -> Span | None:
        """Her run along a platform t s into the plan (away from its ends as `expected` has them), or None."""
        for leg in legs:
            m0 = EXPECT_MARGIN + min(leg.start.slack, 2 * MAX_SLACK)
            for sp in leg.spans:
                if sp.t0 + m0 <= t <= sp.t1 - m0:
                    return sp
        return None

    # --- chains ----------------------------------------------------------------------------------

    def chain(self, moves: list[Move], start: Body, monsters=(), keep_going: bool = False,
              hops: bool = True, price: Callable[[Leg], float] | None = None,
              alts: Callable[[Move], list[Move]] | None = None,
              goal: tuple[int, float] | None = None, spread: bool = True) -> tuple[list[Leg], Keys, str]:
        """As many of `moves` as can be done in one go from `start`: their legs, the keys, and why
        the chain stopped ('done', 'unpredictable', 'monster' or 'long'). Each move is done the way
        (variant) the next move can follow on from. A monster in the way of a run is hopped over
        (`hops`); the chain stops before a leg one could still get in the way of, and at a move it
        can't predict. Stopped for a monster, it ends where none could reach her standing END_LINGER
        (she stands there for the next look and plan): short of a leg end that one could, as far back as
        it takes. With `keep_going`, a chain cut off for length doesn't stop her at its end: the next
        chain carries on from there. With `price` (from 2026-10-27: the navigator's, attack.stance.landing's price of
        where a leg lands, s; `price(leg, last=False)` for a landing she runs on from, the route's last move's
        `price(leg)`), a move whose first way lands with a price is done where it lands the cheapest by its time
        and that price (_priced: its other ways, `alts` (the route's other moves between the same platforms) and with
        `spread` wider takeoffs; the walk on counted, to `goal` ((platform, map x) where she's going on it) for the
        route's last move, else to the next move's spot), the way the next move follows on from; else as before. Every
        move, not only the route's last (PRICE_EVERY_MOVE): a chain's landings by monsters in the simulator came at
        moves before the route's last, its chain cut there or not, as often as at it (the reviews of 2026-10-27)."""
        k, legs, b, why = Keys(self.key_names), [], start, "done"
        monsters = list(monsters)
        marks = []  # (legs so far, keys, body) as each move begins: where a chain stopped for a monster may end
        for i, m in enumerate(moves):
            if i >= MAX_LEGS or b.t > MAX_SECONDS:
                why = "long"
                break
            marks.append((len(legs), k, b))
            # The last leg is done the way the next move can follow on from: from this chain, or (keep_going)
            # from the next one, planned on the move. Done the stop way, the next chain started from standing.
            nxt = moves[i + 1] if i + 1 < len(moves) and (keep_going or i + 1 < MAX_LEGS) else None
            order = ("stop", "run", "stand") if nxt is None else self.VARIANTS
            linger = END_LINGER if nxt is None else 0.0
            options = []
            for v in order:
                trial = k.copy()
                leg = self.leg(m, b, trial, v)
                if leg is not None and not self.danger(leg, monsters, linger=linger):
                    options.append((leg, trial))
            if price is not None and options and (nxt is None or PRICE_EVERY_MOVE):
                options = self._priced(m, b, k, order, options, monsters, linger, price, alts,
                                       goal if nxt is None else (m.dst, nxt.takeoff if nxt.takeoff is not None
                                                                 else nxt.x),
                                       last=nxt is None, spread=spread)
            if not options:
                hopped = None
                if hops:  # a hop over the monster in the way, done the first way (variant) it works out
                    for v in order:
                        hopped = self._hop_then(m, b, k, v, monsters, linger)
                        if hopped:
                            break
                if hopped:
                    jump, leg, k = hopped
                    legs += [jump, leg]
                    b = leg.end
                    continue
                why = "monster" if any(self.leg(m, b, k.copy(), v) for v in order) else "unpredictable"
                break

            def follows(leg: Leg) -> bool:
                """Whether the next move (if any) can be done some way from where this leg leaves her."""
                return nxt is None or any(self.leg(nxt, leg.end, Keys(self.key_names), v) for v in self.VARIANTS)

            leg, trial = next(((lg, tr) for lg, tr in options if follows(lg)), options[0])
            if b.early_arrow is not None:  # just climbed: hold the arrow through the climb-over
                trial.press_arrow_early(b.t, b.early_arrow)
            legs.append(leg)
            k, b = trial, leg.end
        if why == "monster" and legs:
            # Stopped short of a leg a monster could touch her on: not where one could reach her as she stands
            # there. (Stopped on 32, a ledge of Tree Dungeon's floor, under a climb a Green Mushroom walked the top
            # of, she was knocked off it by the Green beside her and back down among the floor's Slimes: simulator,
            # loop_tree_walls.)
            marks.pop()  # (the move it stopped at)
            while legs and self.danger(legs[-1], monsters, linger=END_LINGER):
                n, k, b = marks.pop()
                legs = legs[:n]
        if legs and not (keep_going and why == "long"):
            k.set_arrow(b.t, 0)  # stop at the end (she slides a little)
        if legs and "pick" in self.key_names:
            _tap_pick_up(k, legs)
        return legs, k, why


def _at_rest(leg: Leg) -> Leg:
    """Leg `leg` as it ends where she comes to stand: running at its end, the stop chain() ends it with (the arrow let
    go there, body.stop_distance and stop_time on), within its platform; else as it is."""
    e = leg.end
    if not e.run:
        return leg
    p = e.platform
    x = min(max(e.x + e.run * body.stop_distance(e.v, SHOE, LAGS), p.left), p.right)
    return replace(leg, end=replace(e, t=e.t + body.stop_time(e.v, SHOE, LAGS), x=x, run=0, v=0.0, landed=False))


# Every move of a chain priced where it lands (flow.chain's `price`), not only the route's last: False, the route's last
# alone (as first built, 2026-10-27).
PRICE_EVERY_MOVE = True

# _spread's drops and jumps straight up from other spots (_drop's, _jump_up's `at`) taken from that spot itself (from
# 2026-10-28): False, as first built, from where she stands or stops anywhere in the move's clear stretch, so every spot
# gave the same move (live, 2026-10-28 00:38:36: the drop 21 -> 23 from 42 by a Horny Mushroom, where from 60, a step of
# 18, it lands clear). In the harness (Tree, crowd, walkers; master 1cb0c3f against it, D:/MapleHelper-scratch/lshot/fix)
# it leans the wrong way in grinding, the default: 1.2% less EXP (se 0.9%, t -1.4; on the first 48 seeds 2.1% less,
# t -1.9), its trips 2.9% longer (t 2.1, the only row past 2), their landings within 105 of a monster 4.7% more (t 1.4),
# within 60 5.7% more, by one walking at her 9.3% more (t 1.6): the opposite of what it's for; the hunt loop level (0.8%
# more EXP, se 1.7%, 48 seeds). So off (the review's, 2026-10-28), as master's, until the user decides or a live A/B
# shows it buys something: True to try it (four Tree goldens differ then, D:/MapleHelper-scratch/lshot/mine/golden-spread).
SPREAD_FROM_SPOT = False


def chain_record(legs: list[Leg], why: str) -> dict:
    """A chain of moves in one key plan as telemetry data ('chain'): where she runs on the ground (spans)
    and where each leg ends, as predicted."""
    return {"why": why,
            "text": ", ".join(f"{leg.move.kind}->{leg.move.dst}" for leg in legs) + f" ({why})",
            "legs": [{"kind": leg.move.kind, "src": leg.move.src, "dst": leg.move.dst,
                      "spans": [[sp.platform, round(sp.x0, 1), round(sp.x1, 1)] for sp in leg.spans],
                      "end": {"t": round(leg.end.t, 3), "x": round(leg.end.x, 1), "platform": leg.end.platform.id}}
                     for leg in legs]}
