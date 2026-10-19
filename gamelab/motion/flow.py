"""Moving like a player: several route moves turned into one continuous key timeline.

A player doesn't stop between moves. She runs along a platform and jumps on the way, keeps running
when she lands, jumps onto the next platform, jumps at a rope and holds up, and only looks around
when something unexpected happens. This module predicts her body over time with the game's physics
(physics.fly) and her measured walking, and writes the keys for a chain of moves as one plan:

  body now -> leg 1 (keys, where she'll be when) -> leg 2 -> ... -> one look at the end

Each move can be done a few ways (keep running on landing, land and stop, jump straight up, step back
for a run-up first), and the chain picks the way the next move can follow on from. Moves it can't
predict well enough (climbing down a rope, which needs her lined up on its top) end the chain; the
navigator has control.mover do those on their own, closed loop.

Monsters: she doesn't fight while travelling. Each leg's runs on the ground are checked against the
monsters seen: where they're walking to (from two looks), or anywhere they could get at their
walking speed when only one look caught them. A monster in the way of a run is hopped over: a
running jump timed so the top of the jump is over where it will most likely be, landing on the same
platform clear of it, as a player does. Where that can't be timed (too close, no room to land), the
chain stops before that leg and the navigator waits or goes another way.

All positions in map units, times in seconds from the start of the plan.
"""

from dataclasses import dataclass, field

from maplehelper.engine import LEFT, RIGHT

from ..localization.mapdata import MapModel, Platform
from ..localization.monsters import MonsterTrack
from .physics import (DROP_LAG, GRAVITY, JUMP_HEIGHT, JUMP_LAG, JUMP_SPEED, LAND_MARGIN, LAND_SLIDE, RUN_LAG,
                      STOP_RUN, STOP_S, TURN_EXTRA, WALK_SPEED, fly)
from .route import JUMP_FROM, Move

# Her measured lags and slides (RUN_LAG, TURN_EXTRA, JUMP_LAG, DROP_LAG, STOP_RUN, STOP_S, LAND_SLIDE)
# describe her body, so they're in physics, with how they were measured (still importable from here).
DROP_SLACK = 0.1     # s of extra timing error a drop adds
ARROW_LAG = 0.1      # s she runs on after the arrow is released (input lag)
AFTER_LANDING = 0.03  # s after a predicted landing before the next jump key (in the air it's ignored)
SLACK_PER_LEG = 0.025  # s of timing error each predicted landing may add
MAX_SLACK = 0.15     # ... counted up to this (beyond it waiting longer only makes her sluggish)
TAP = 0.04           # s a jump/down tap is held
PICK_EVERY = 0.15    # s between pick-up presses while she runs
MIN_RUNUP = 15.0     # units of run before a running jump from standing
JUMP_FROM_FAR = 45.0  # running jumps take off between JUMP_FROM and this far before the end
SHORT_STEP = STOP_RUN + 10  # units: closer than this she steps there (a short hold), no run and slide
JUST_PASSED = 2      # units: running, a spot she's this little past still counts as reached
EDGE_ROOM = 5        # units she keeps in from a platform's ends when stepping back or taking off
ROPE_CLEAR = 30      # drop at least this far from the top of a rope hanging from her platform
DROP_STEP = 2        # units between the drop spots tried
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
PRE_ARROW = 0.1      # s before the predicted top that the next move's arrow is pressed
HER_HALF = 15        # half her body's width
APEX_T = JUMP_SPEED / GRAVITY  # s from leaving the ground to the top of a jump (0.28)
HOP_HORIZON = 3.0    # s ahead a hop over a monster is looked for
END_LINGER = 1.0     # s she may stand where a chain ends: a monster mustn't reach her there either
MAX_LEGS = 5
MAX_SECONDS = 3.5    # a chain this long gets a fresh look before the next one
HOLD_MERGE_MS = 5    # a key held again within this many ms of its release stays down


@dataclass(frozen=True)
class Body:
    """Her predicted state at a moment of the plan."""
    t: float
    x: float
    platform: Platform
    run: int = 0          # running that way at full speed with the arrow held (+1 right, -1 left), or 0
    landed: bool = False  # she touches down at t (the next jump/drop key must come after that)
    slack: float = 0.0    # how far off t may be by now (prediction errors add up leg by leg)
    early_arrow: float | None = None  # after a climb: the next arrow may go down this early

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
    """One move as planned: her body where it starts and ends, and her runs on the ground (spans)."""
    move: Move
    start: Body
    end: Body
    spans: list[Span] = field(default_factory=list)


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

    def adopt(self, other: "Keys") -> None:
        """Take over another plan's events and arrow (a trial copy that worked out)."""
        self.events, self.arrow = other.events, other.arrow

    def set_arrow(self, t: float, d: int) -> None:
        """From t, hold the arrow that way (+1 right, -1 left) or none (0): the one held is let go, and
        the new one pressed a millisecond later."""
        if d == self.arrow:
            return
        at = ms(t)
        if self.arrow:
            self.events.append([at, self.names[_dir_name(self.arrow)], "up"])
        if d:
            self.events.append([at + (1 if self.arrow else 0), self.names[_dir_name(d)], "down"])
        self.arrow = d

    def press_arrow_early(self, t: float, early: float) -> None:
        """Move the arrow press at t (if there is one) to `early`."""
        at = ms(t)
        for e in self.events:
            if e[0] == at and e[2] == "down" and e[1] in (self.names[LEFT], self.names[RIGHT]):
                e[0] = ms(early)
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


def _dir_sign(direction: str) -> int:
    """A move's direction ('left' or 'right') as a sign along x."""
    return 1 if direction == RIGHT else -1


def _dir_name(d: int) -> str:
    """A sign along x as a direction: the arrow for it."""
    return RIGHT if d > 0 else LEFT


def _start_lag(b: Body, d: int) -> float:
    """How far a run that way from body b lags running at full speed from b.t: none when she already
    is, a standing start's lag, more when she's turning round."""
    return 0.0 if b.run == d else RUN_LAG + (TURN_EXTRA if b.run == -d else 0.0)


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

    def pick(self, near: float, far: float) -> float:
        """A spot between `near` (the quickest) and `far`, usually close to `near`; `near` without rng."""
        if self.rng is None:
            return near
        return near + (far - near) * self.rng.random() ** 2

    # --- running -----------------------------------------------------------------------------

    @staticmethod
    def run_to(b: Body, d: int, x: float) -> float | None:
        """When she's at x running that way from body b, the arrow pressed at b.t (None if it's
        behind her)."""
        dist = (x - b.x) * d
        if b.run == d:
            return b.t + dist / WALK_SPEED if dist >= -JUST_PASSED else None
        if dist < 0:
            return None
        return b.t + _start_lag(b, d) + dist / WALK_SPEED

    # --- one leg per move ------------------------------------------------------------------------
    #
    # Each move can be done a few ways ("variants"); the chain picks the one the next move can follow:
    #   run      keep running when she lands (the arrow stays held);
    #   stop     the same jump, but the arrow is let go just before she lands, so she stands still
    #            ~13 units on (for a next move that needs her standing, or going the other way);
    #   stand    (jump_up) stop under the platform above and jump straight up.

    VARIANTS = ("run", "stop", "stand")

    @staticmethod
    def press_jump(k: Keys, t_off: float, b: Body) -> float:
        """Press jump so she leaves the ground at t_off, or as soon after as she can: never before
        she has landed, since a press in the air is ignored. Returns when she leaves the ground."""
        press = max(t_off - JUMP_LAG, b.ready_at)
        k.tap(press, "jump")
        return press + JUMP_LAG

    def leg(self, m: Move, b: Body, k: Keys, variant: str = "run") -> Leg | None:
        """Move m done the `variant` way from body b, its keys added to k; None where that way doesn't
        work. Climbing down a rope (and a hop, planned by `hop`) isn't predicted: it ends a chain."""
        if b.platform.id != m.src:
            return None
        plan = {"jump": self._jump, "walk_off": self._walk_off, "jump_up": self._jump_up, "drop": self._drop,
                "climb": self._climb}.get(m.kind)
        return None if plan is None else plan(m, b, k, variant)

    def _landing(self, m: Move, b: Body, k: Keys, variant: str, t_land: float, lx: float, land: Platform,
                 d: int, spans: list[Span]) -> Leg:
        """End a jump/fall leg: running on (arrow held) or, for 'stop', standing a little further on."""
        slack = b.slack + SLACK_PER_LEG
        if variant == "stop" and d:
            k.set_arrow(t_land - ARROW_LAG, 0)  # takes effect as she lands
            x = lx + d * LAND_SLIDE
            if not land.covers(x, -3):  # she'd slide off the end: she stops 3 units in from it
                x = min(max(x, land.left + 3), land.right - 3)
            return Leg(m, b, Body(t_land + STOP_S - ARROW_LAG, x, land, slack=slack), spans)
        return Leg(m, b, Body(t_land, lx, land, run=d, landed=True, slack=slack), spans)

    def _runup(self, b: Body, d: int, takeoff: float, k: Keys) -> Body | None:
        """Make sure she can run at `takeoff` going that way: stop if she's running the other way,
        step back if she's standing too close. Returns her body ready to run (or None: no room)."""
        if b.run == d:
            return b
        a = b.platform
        if b.run == -d:  # stop first
            k.set_arrow(b.t, 0)
            b = Body(b.t + STOP_S, b.x - d * STOP_RUN, a, slack=b.slack)
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
        if variant == "stand":
            return None
        a, d = b.platform, _dir_sign(m.direction)
        edge = a.right if d > 0 else a.left
        if m.takeoff is not None:  # a jump from along the platform (up to 10 units early with rng)
            takeoff = m.takeoff - d * self.pick(0.0, 10.0)
        else:
            takeoff = edge - d * self.pick(JUMP_FROM, JUMP_FROM_FAR)  # players jump from a little further back too
        if b.run == d and (takeoff - b.x) * d < -JUST_PASSED:
            if (edge - b.x) * d <= 12:
                return None
            takeoff = b.x  # already past the usual spot, still room: now
        start = self._runup(b, d, takeoff, k)
        if start is None:
            return None
        t = self.run_to(start, d, takeoff)
        if t is None:
            return None
        k.set_arrow(start.t, d)
        t_off = self.press_jump(k, t, start)
        takeoff += d * (t_off - t) * WALK_SPEED  # later than planned if she had to land first
        if (edge - takeoff) * d < 3:
            return None
        land, lx, ta = fly(self.model, takeoff, a.y_at(takeoff), d * WALK_SPEED, -JUMP_SPEED)
        if land is None or land.id != m.dst:
            return None
        return self._landing(m, b, k, variant, t_off + ta, lx, land, d, [Span(a.id, b.t, t_off, b.x, takeoff)])

    def _walk_off(self, m: Move, b: Body, k: Keys, variant: str) -> Leg | None:
        if variant == "stand":
            return None
        a, d = b.platform, _dir_sign(m.direction)
        edge = a.right if d > 0 else a.left
        start = self._runup(b, d, edge, k) if b.run == -d else b
        if start is None:
            return None
        t = self.run_to(start, d, edge)
        if t is None:
            return None
        land, lx, ta = fly(self.model, edge + d, a.y_at(edge) - 1, d * WALK_SPEED, 0.0)
        if land is None or land.id != m.dst:
            return None
        k.set_arrow(start.t, d)
        return self._landing(m, b, k, variant, t + ta, lx, land, d, [Span(a.id, b.t, t, b.x, edge)])

    def _jump_up(self, m: Move, b: Body, k: Keys, variant: str) -> Leg | None:
        a, q = b.platform, self.model.platform(m.dst)
        if variant != "stand":
            # A running jump on the way, if one lands on the platform above with room to spare. Takeoffs
            # every 5 units (up to 400 on): running, from where she is; from standing, towards the
            # move's spot, from 15 units on.
            d = b.run if b.run else (_sign(m.x - b.x) if abs(m.x - b.x) > 15 else 0)
            if not d:
                return None
            first = b.x if b.run == d else b.x + d * 15
            x = first
            while a.covers(x, -EDGE_ROOM) and abs(x - first) <= 400:
                land, lx, ta = fly(self.model, x, a.y_at(x), d * WALK_SPEED, -JUMP_SPEED)
                if land is q and q.covers(lx - d * LAND_MARGIN) and q.covers(lx + d * 30):
                    t = self.run_to(b, d, x)
                    if t is None:
                        return None
                    trial = k.copy()
                    trial.set_arrow(b.t, d)
                    if abs(self.press_jump(trial, t, b) - t) < 0.02:  # leaves the ground where planned
                        k.adopt(trial)
                        return self._landing(m, b, k, variant, t + ta, lx, q, d, [Span(a.id, b.t, t, b.x, x)])
                x += d * 5
            return None
        # Stop anywhere under it and jump straight up.
        lo, hi = max(a.left, q.left) + LAND_MARGIN, min(a.right, q.right) - LAND_MARGIN
        near = min(max(b.x, lo), hi)
        # With rng, from the nearest spot under it towards the far end of the overlap; else the planner's.
        prefer = self.pick(near, hi if near - lo < hi - near else lo) if self.rng else m.x
        stop = self._stop_within(b, lo, hi, prefer, k)
        if stop is None:
            return None
        t, x = stop
        land, lx, ta = fly(self.model, x, a.y_at(x), 0.0, -JUMP_SPEED)
        if land is not q:
            return None
        t = self.press_jump(k, t + JUMP_LAG, Body(t, x, a, landed=b.landed and t == b.t, slack=b.slack))
        return Leg(m, b, Body(t + ta, lx, q, landed=True, slack=b.slack + SLACK_PER_LEG), [Span(a.id, b.t, t, b.x, x)])

    def _drop(self, m: Move, b: Body, k: Keys, variant: str) -> Leg | None:
        if variant != "run":
            return None
        a, below = b.platform, self.model.platform(m.dst)
        lo, hi = max(a.left, below.left) + LAND_MARGIN, min(a.right, below.right) - LAND_MARGIN
        # Not at the top of a rope hanging from her platform (its top within 15 units of it): down there
        # climbs onto the rope.
        ropes = [r.x for r in self.model.ropes if abs(r.top - a.y_at(r.x)) < 15 and a.covers(r.x)]
        clear = [x for x in range(int(lo), int(hi) + 1, DROP_STEP) if all(abs(x - rx) > ROPE_CLEAR for rx in ropes)]
        if not clear:
            return None
        prefer = min(clear, key=lambda x: (abs(x - b.x), abs(x - m.x)))
        if self.rng is not None:  # not always the first spot: somewhere further over it now and then
            far = max(clear, key=lambda x: abs(x - b.x))
            # (pick draws again for every spot, around the nearest one)
            prefer = min(clear, key=lambda x: abs(x - self.pick(prefer, far)))
        # She may stop anywhere in the clear stretch around that spot.
        spots = set(clear)
        run_lo = run_hi = prefer
        while run_lo - DROP_STEP in spots:
            run_lo -= DROP_STEP
        while run_hi + DROP_STEP in spots:
            run_hi += DROP_STEP
        stop = self._stop_within(b, run_lo, run_hi, prefer, k)
        if stop is None:
            return None
        t, x = stop
        land, lx, ta = fly(self.model, x, a.y_at(x) + 1, 0.0, 0.0, skip=a)
        if land is None or land.id != m.dst:
            return None
        t = max(t, b.ready_at)
        k.hold(t, t + DROP_DOWN_HOLD, "down")
        k.tap(t + DROP_JUMP_AFTER, "jump")
        t_off = t + DROP_JUMP_AFTER + DROP_LAG
        return Leg(m, b, Body(t_off + ta, lx, land, landed=True, slack=b.slack + SLACK_PER_LEG + DROP_SLACK),
                   [Span(a.id, b.t, t, b.x, x)])

    def _climb(self, m: Move, b: Body, k: Keys, variant: str) -> Leg | None:
        if variant != "run":
            return None
        a, top = b.platform, self.model.platform(m.dst)
        rx, gap = m.rope_x, a.y_at(m.rope_x) - m.rope_bottom
        # Out of reach of a jump (and her arms), or by a portal: up there would take it.
        if gap > APEX + 10 or any(abs(p.x - rx) < 60 and abs(p.y - a.y_at(p.x)) < 40 for p in self.model.portals):
            return None
        if abs(rx - b.x) <= 4 and not b.run:
            t, grab_h, ran_to = b.t, min(APEX, max(0.0, gap)), b.x   # straight up under it
            t_rope = self.press_jump(k, t + JUMP_LAG, b) + 0.15  # grabs it 0.15 s after leaving the ground
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
            t = self.press_jump(k, t, start)
            t_rope = t + GRAB_LEAD / WALK_SPEED
            grab_h = APEX
            k.set_arrow(t_rope + 0.12, 0)  # let go of the arrow once she's on the rope
        t_top = t_rope + max(0.0, (a.y_at(rx) - grab_h) - m.rope_top) / ROPE_SPEED
        k.hold(t, t_top + UP_EXTRA, "up")
        return Leg(m, b, Body(t_top + CLIMB_OVER, rx, top, slack=b.slack + SLACK_PER_LEG, early_arrow=t_top - PRE_ARROW),
                   [Span(a.id, b.t, t, b.x, ran_to)])

    def _stop_within(self, b: Body, lo: float, hi: float, prefer: float, k: Keys) -> tuple[float, float] | None:
        """Bring her to a stop anywhere in [lo, hi] (nearest `prefer` when she has a choice): right
        away if she's there, by releasing the arrow now if that leaves her there, else at the spot.
        Returns (when she's stopped, where)."""
        if lo > hi:
            return None
        target = min(max(prefer, lo), hi)
        if not b.run:
            if lo <= b.x <= hi:
                return b.t, b.x
            dist = abs(target - b.x)
            if dist < SHORT_STEP:  # hold the arrow just long enough (walking model)
                # ms, by the measured walk (8.5 + 0.1486 per ms: physics.hold_ms's model, rounded)
                held = max(30.0, (dist - 8.5) / 0.1486)
                d = _sign(target - b.x)
                k.set_arrow(b.t, d)
                k.set_arrow(b.t + held / 1000, 0)
                return b.t + held / 1000 + STOP_S, target
            return self._stop_at(b, target, k)
        natural = b.x + b.run * STOP_RUN
        if lo <= natural <= hi:
            k.set_arrow(b.t, 0)
            return b.t + STOP_S, natural
        if (target - natural) * b.run > 0:  # the spot is further on
            return self._stop_at(b, target, k)
        return None

    def _stop_at(self, b: Body, x: float, k: Keys, tolerance: float = 12) -> tuple[float, float] | None:
        """Bring her to a stop at x (within tolerance): release the arrow a slide's length early.
        Returns (when she's stopped, where)."""
        if abs(x - b.x) <= tolerance and not b.run:
            return b.t, b.x
        d = _sign(x - b.x)
        if b.run == -d:
            return None
        if b.run != d and abs(x - b.x) < SHORT_STEP:
            return None  # a short shuffle: leave it to the closed loop
        t = self.run_to(b, d, x - d * STOP_RUN)
        if t is None:
            return None
        k.set_arrow(b.t, d)
        k.set_arrow(t, 0)
        return t + STOP_S, x

    # --- monsters ------------------------------------------------------------------------------

    def danger(self, leg: Leg, monsters: list[MonsterTrack], linger: float = 0.0) -> tuple[MonsterTrack, float] | None:
        """The first monster that could touch her on this leg's runs (and, if `linger`, while she
        stands that long where it ends), and when."""
        spans = list(leg.spans)
        if linger:
            e = leg.end
            spans.append(Span(e.platform.id, e.t, e.t + linger, e.x, e.x))
        for span in spans:
            p = self.model.platform(span.platform)
            for mo in monsters:
                if mo.platform != span.platform:
                    continue
                touch = mo.half + HER_HALF
                t = span.t0
                while t <= span.t1 + 1e-9:  # checked every 0.05 s
                    lo, hi = mo.reach(t, p)
                    x = span.x_at(t)
                    if lo - touch < x < hi + touch:
                        return mo, t
                    t += 0.05
        return None

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
            return b.x + d * WALK_SPEED * max(0.0, t - b.t - _start_lag(b, d))

        closing = WALK_SPEED + max(0.0, -(mo.vx or 0.0) * d)  # how fast the gap shrinks
        disc = JUMP_SPEED ** 2 - 2 * GRAVITY * mo.height
        rise = (JUMP_SPEED - disc ** 0.5) / GRAVITY if disc > 0 else APEX_T  # s to clear its height
        gap = mo.half + HER_HALF + closing * rise + 3   # distance to its middle at takeoff
        t = b.t
        while t < b.t + HOP_HORIZON and (mo.at(t, a) - her_x(t)) * d > gap:
            t += 0.01
        if t >= b.t + HOP_HORIZON:
            return None  # it keeps ahead of her
        earliest = b.ready_at + JUMP_LAG
        if t < earliest or (mo.at(t, a) - her_x(t)) * d < gap - 15:
            return None  # already too close to time it
        x_off = her_x(t)
        if not a.covers(x_off, -EDGE_ROOM):
            return None
        land, lx, ta = fly(self.model, x_off, a.y_at(x_off), d * WALK_SPEED, -JUMP_SPEED)
        if land is not a or (lx - mo.at(t + ta, a)) * d < 0:
            return None  # off the platform, or she'd come down short of it
        k.set_arrow(b.t, d)
        t_off = self.press_jump(k, t, b)
        move = Move("hop", a.id, a.id, x_off, lx, _dir_name(d))
        return Leg(move, b, Body(t_off + ta, lx, a, run=d, landed=True, slack=b.slack + SLACK_PER_LEG),
                   [Span(a.id, b.t, t_off, b.x, x_off)])

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

    # --- following a chain ---------------------------------------------------------------------

    @staticmethod
    def expected(legs: list[Leg], t: float) -> int | None:
        """The platform she should be on (on the ground) t s into the plan, or None while she's in
        the air, on a rope, or within a moment of a takeoff or landing."""
        margin = 0.12
        for i, leg in enumerate(legs):
            m0 = margin + min(leg.start.slack, 2 * MAX_SLACK)  # timings drift as the chain goes on
            for sp in leg.spans:
                if sp.t0 + m0 <= t <= sp.t1 - m0:
                    return sp.platform
            m1 = margin + min(leg.end.slack, 2 * MAX_SLACK)
            nxt = legs[i + 1].spans[0].t0 if i + 1 < len(legs) and legs[i + 1].spans else None
            # Where the leg ends: for good after the last leg, else from a landing until the next run.
            if leg.end.t + m1 <= t and (nxt is None or (t <= nxt - m1 and leg.end.landed)):
                return leg.end.platform.id
        return None

    # --- chains ----------------------------------------------------------------------------------

    def chain(self, moves: list[Move], start: Body, monsters=(), keep_going: bool = False,
              hops: bool = True) -> tuple[list[Leg], Keys, str]:
        """As many of `moves` as can be done in one go from `start`: their legs, the keys, and why
        the chain stopped ('done', 'unpredictable', 'monster' or 'long'). Each move is done the way
        (variant) the next move can follow on from. A monster in the way of a run is hopped over
        (`hops`); the chain stops before a leg one could still get in the way of, and at a move it
        can't predict. With `keep_going`, a chain cut off for length doesn't stop her at its end:
        the next chain carries on from there."""
        k, legs, b, why = Keys(self.key_names), [], start, "done"
        monsters = list(monsters)
        for i, m in enumerate(moves):
            if i >= MAX_LEGS or b.t > MAX_SECONDS:
                why = "long"
                break
            nxt = moves[i + 1] if i + 1 < len(moves) and i + 1 < MAX_LEGS else None
            order = ("stop", "run", "stand") if nxt is None else self.VARIANTS
            linger = END_LINGER if nxt is None else 0.0
            options = []
            for v in order:
                trial = k.copy()
                leg = self.leg(m, b, trial, v)
                if leg is not None and not self.danger(leg, monsters, linger=linger):
                    options.append((leg, trial))
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
        if legs and not (keep_going and why == "long"):
            k.set_arrow(b.t, 0)  # stop at the end (she slides a little)
        if legs and "pick" in self.key_names:
            _tap_pick_up(k, legs)
        return legs, k, why


def chain_record(legs: list[Leg], why: str) -> dict:
    """A chain of moves in one key plan as telemetry data ('chain'): where she runs on the ground (spans)
    and where each leg ends, as predicted."""
    return {"why": why,
            "text": ", ".join(f"{leg.move.kind}->{leg.move.dst}" for leg in legs) + f" ({why})",
            "legs": [{"kind": leg.move.kind, "src": leg.move.src, "dst": leg.move.dst,
                      "spans": [[sp.platform, round(sp.x0, 1), round(sp.x1, 1)] for sp in leg.spans],
                      "end": {"t": round(leg.end.t, 3), "x": round(leg.end.x, 1), "platform": leg.end.platform.id}}
                     for leg in legs]}
