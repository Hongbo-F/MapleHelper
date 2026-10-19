"""Moving like a player: several route moves turned into one continuous key timeline.

A player doesn't stop between moves. She runs along a platform and jumps on the way, keeps running
when she lands, jumps onto the next platform, jumps at a rope and holds up, and only looks around
when something unexpected happens. This module predicts her body over time with the game's physics
(route.fly) and her measured walking, and writes the keys for a chain of moves as one plan:

  body now -> leg 1 (keys, where she'll be when) -> leg 2 -> ... -> one look at the end

Each move can be done a few ways (keep running on landing, land and stop, jump straight up, step back
for a run-up first), and the chain picks the way the next move can follow on from. Moves it can't
predict well enough (climbing down a rope, which needs her lined up on its top) end the chain; the
navigator does those on their own, closed loop.

Monsters: she doesn't fight while travelling. Each leg's runs on the ground are checked against the
monsters seen, which may walk along their platform (in the direction they were seen going, else
either way) at up to their walking speed. The chain stops before a leg one of them could get in the
way of; the navigator then routes around it or waits for it to move off. Jumping over them doesn't
work here: by the game data a pig is 67 units wide and 46 tall, and her jump stays above 46 units
for only ~52 units of travel.

All positions in map units, times in seconds from the start of the plan.
"""

from dataclasses import dataclass, field

from maplehelper.engine import LEFT, RIGHT

from .mapdata import MapModel, Platform
from .route import CLIMB_SPEED, GRAVITY, JUMP_FROM, JUMP_SPEED, LAND_MARGIN, WALK_SPEED, Move, fly

# Measured on platform 34 with the camera fixed (map edge), 60 fps: from standing, holding an arrow
# puts her at 148.5 (t - 0.12) units after t s; a jump lifts off 55 ms after the key; after the arrow
# is released she runs on ~0.1 s (input lag) and slides, stopping ~27 units on, ~0.3 s later. A jump
# key pressed while she's still in the air is ignored, and holding it doesn't jump again on landing.
RUN_LAG = 0.12       # s: a standing start lags full-speed running by this much
TURN_EXTRA = 0.08    # s more when she was running the other way
JUMP_LAG = 0.055     # s from the jump key to leaving the ground
DROP_LAG = 0.28      # s from the jump key (down held) to falling through (0.28-0.34 in 4 drops on the
                     # minimap at 60 fps, whether or not she had just stopped running)
DROP_SLACK = 0.1     # s of extra timing error a drop adds
STOP_RUN = 27.0      # units she goes on after the arrow is released at full speed ...
STOP_S = 0.3         # ... until she stands still
AFTER_LANDING = 0.03  # s after a predicted landing before the next jump key (in the air it's ignored)
SLACK_PER_LEG = 0.025  # s of timing error each predicted landing may add
TAP = 0.04           # s a jump/down tap is held
LAND_SLIDE = 13.0    # units she slides on landing with no arrow held
MIN_RUNUP = 15.0     # units of run before a running jump from standing
ROPE_CLEAR = 30      # drop at least this far from a rope hanging below
GRAB_LEAD = WALK_SPEED * JUMP_SPEED / GRAVITY  # ~41: jump this far before a rope to be at the top over it
APEX = JUMP_SPEED ** 2 / (2 * GRAVITY)         # 77 units
ONTO_TOP = 0.45      # s to climb from a rope's top onto the platform (with some margin)
HER_HALF = 15        # half her body's width
MAX_LEGS = 5
MAX_SECONDS = 6.0

# Walking speeds (map units/s), from each monster's speed stat in the game data (pig and Orange
# Mushroom 0, slime -20, snail -65) applied to a base of 70; an upper bound for where they can get.
MONSTER_SPEED = {"pig": 70.0, "orange-mushroom": 70.0, "slime": 56.0, "snail": 25.0}
# Half their body width (map units), from the lt/rb of their sprites in the game data.
MONSTER_HALF = {"pig": 34.0, "orange-mushroom": 32.0, "slime": 28.0, "snail": 19.0}
TURN_SLACK = 25      # units a monster seen walking one way may still come back (they turn now and then)


@dataclass(frozen=True)
class Body:
    """Her predicted state at a moment of the plan."""
    t: float
    x: float
    platform: Platform
    run: int = 0          # running that way at full speed with the arrow held (+1 right, -1 left), or 0
    landed: bool = False  # she touches down at t (the next jump/drop key must come after that)
    slack: float = 0.0    # how far off t may be by now (prediction errors add up leg by leg)


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
    move: Move
    start: Body
    end: Body
    spans: list[Span] = field(default_factory=list)


@dataclass(frozen=True)
class Monster:
    """A monster seen on a platform: where it is (map x), how fast it can walk, which way it was
    seen walking (+1/-1, or 0 if unknown) and half its body width."""
    platform: int
    x: float
    speed: float = 70.0
    heading: int = 0
    half: float = 34.0

    def reach(self, t: float, p: Platform) -> tuple[float, float]:
        """Where its centre can be t s after it was seen (it never leaves its platform)."""
        go = self.speed * t
        lo = self.x - (go if self.heading <= 0 else min(go, TURN_SLACK))
        hi = self.x + (go if self.heading >= 0 else min(go, TURN_SLACK))
        return max(p.left, lo), min(p.right, hi)


class Keys:
    """Key events for one plan, with the arrow tracked so a run carries on across legs."""

    def __init__(self, names: dict[str, str]):
        self.names = names        # left/right/jump/up/down -> key name
        self.events: list[list] = []
        self.arrow: int = 0       # -1 left held, +1 right held, 0 none

    def copy(self) -> "Keys":
        k = Keys(self.names)
        k.events, k.arrow = list(self.events), self.arrow
        return k

    def set_arrow(self, t: float, d: int) -> None:
        if d == self.arrow:
            return
        ms = _ms(t)
        if self.arrow:
            self.events.append([ms, self.names[RIGHT if self.arrow > 0 else LEFT], "up"])
        if d:
            self.events.append([ms + (1 if self.arrow else 0), self.names[RIGHT if d > 0 else LEFT], "down"])
        self.arrow = d

    def tap(self, t: float, key: str, hold: float = TAP) -> None:
        self.events += [[_ms(t), self.names[key], "down"], [_ms(t + hold), self.names[key], "up"]]

    def hold(self, t0: float, t1: float, key: str) -> None:
        self.events += [[_ms(t0), self.names[key], "down"], [_ms(t1), self.names[key], "up"]]

    def sorted(self) -> list[list]:
        return sorted(self.events, key=lambda e: (e[0], e[2] == "down"))


def _ms(t: float) -> int:
    return max(0, round(t * 1000))


def _sign(v: float) -> int:
    return 1 if v > 0 else -1


class Flow:
    def __init__(self, model: MapModel, key_names: dict[str, str]):
        self.model = model
        self.key_names = key_names

    # --- running -----------------------------------------------------------------------------

    @staticmethod
    def run_to(b: Body, d: int, x: float) -> float | None:
        """When she's at x running that way from body b, the arrow pressed at b.t (None if it's
        behind her)."""
        dist = (x - b.x) * d
        if b.run == d:
            return b.t + dist / WALK_SPEED if dist >= -2 else None
        if dist < 0:
            return None
        lag = RUN_LAG + (TURN_EXTRA if b.run == -d else 0.0)
        return b.t + lag + dist / WALK_SPEED

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
        press = max(t_off - JUMP_LAG, b.t + (AFTER_LANDING + b.slack if b.landed else 0.0))
        k.tap(press, "jump")
        return press + JUMP_LAG

    def leg(self, m: Move, b: Body, k: Keys, variant: str = "run") -> Leg | None:
        if b.platform.id != m.src:
            return None
        fn = getattr(self, "_" + m.kind, None)
        return None if fn is None else fn(m, b, k, variant)

    def _landing(self, m: Move, b: Body, k: Keys, variant: str, t_land: float, lx: float, land: Platform,
                 d: int, spans: list[Span]) -> Leg:
        """End a jump/fall leg: running on (arrow held) or, for 'stop', standing a little further on."""
        slack = b.slack + SLACK_PER_LEG
        if variant == "stop" and d:
            k.set_arrow(t_land - 0.1, 0)  # takes effect as she lands
            x = lx + d * LAND_SLIDE
            if not land.covers(x, -3):
                x = min(max(x, land.left + 3), land.right - 3)
            return Leg(m, b, Body(t_land + STOP_S - 0.1, x, land, 0, False, slack), spans)
        return Leg(m, b, Body(t_land, lx, land, d, True, slack), spans)

    def _runup(self, b: Body, d: int, takeoff: float, k: Keys) -> Body | None:
        """Make sure she can run at `takeoff` going that way: stop if she's running the other way,
        step back if she's standing too close. Returns her body ready to run (or None: no room)."""
        if b.run == d:
            return b
        a = b.platform
        if b.run == -d:  # stop first
            k.set_arrow(b.t, 0)
            b = Body(b.t + STOP_S, b.x - d * STOP_RUN, a, 0, False, b.slack)
        room = (takeoff - b.x) * d
        if room >= MIN_RUNUP:
            return b
        back = b.x - d * (MIN_RUNUP + 8 - room)
        if not a.covers(back, -5):
            return None
        step = self._stop_within(b, back - 4, back + 4, back, k)
        if step is None:
            return None
        return Body(step[0], step[1], a, 0, False, b.slack)

    def _jump(self, m: Move, b: Body, k: Keys, variant: str) -> Leg | None:
        if variant == "stand":
            return None
        a, d = b.platform, (1 if m.direction == RIGHT else -1)
        edge = a.right if d > 0 else a.left
        takeoff = edge - d * JUMP_FROM
        if b.run == d and (takeoff - b.x) * d < -2:
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
        a, d = b.platform, (1 if m.direction == RIGHT else -1)
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
            # A running jump on the way, if one lands on the platform above with room to spare.
            d = b.run if b.run else (_sign(m.x - b.x) if abs(m.x - b.x) > 15 else 0)
            if not d:
                return None
            first = b.x if b.run == d else b.x + d * 15
            x = first
            while a.covers(x, -5) and abs(x - first) <= 400:
                land, lx, ta = fly(self.model, x, a.y_at(x), d * WALK_SPEED, -JUMP_SPEED)
                if land is q and q.covers(lx - d * LAND_MARGIN) and q.covers(lx + d * 30):
                    t = self.run_to(b, d, x)
                    if t is None:
                        return None
                    trial = k.copy()
                    trial.set_arrow(b.t, d)
                    if abs(self.press_jump(trial, t, b) - t) < 0.02:  # leaves the ground where planned
                        k.events, k.arrow = trial.events, trial.arrow
                        return self._landing(m, b, k, variant, t + ta, lx, q, d, [Span(a.id, b.t, t, b.x, x)])
                x += d * 5
            return None
        # Stop anywhere under it and jump straight up.
        lo, hi = max(a.left, q.left) + LAND_MARGIN, min(a.right, q.right) - LAND_MARGIN
        stop = self._stop_within(b, lo, hi, m.x, k)
        if stop is None:
            return None
        t, x = stop
        land, lx, ta = fly(self.model, x, a.y_at(x), 0.0, -JUMP_SPEED)
        if land is not q:
            return None
        t = self.press_jump(k, t + JUMP_LAG, Body(t, x, a, 0, b.landed and t == b.t, b.slack))
        return Leg(m, b, Body(t + ta, lx, q, 0, True, b.slack + SLACK_PER_LEG), [Span(a.id, b.t, t, b.x, x)])

    def _drop(self, m: Move, b: Body, k: Keys, variant: str) -> Leg | None:
        if variant != "run":
            return None
        a, below = b.platform, self.model.platform(m.dst)
        lo, hi = max(a.left, below.left) + LAND_MARGIN, min(a.right, below.right) - LAND_MARGIN
        # Not next to a rope hanging below: falling past it with down held, she grabbed it.
        ys = (a.y_at(m.x), below.y_at(m.x))
        ropes = [r.x for r in self.model.ropes if r.top < ys[1] and r.bottom > ys[0]]
        clear = [x for x in range(int(lo), int(hi) + 1, 2) if all(abs(x - rx) > ROPE_CLEAR for rx in ropes)]
        if not clear:
            return None
        prefer = min(clear, key=lambda x: (abs(x - b.x), abs(x - m.x)))
        lo = hi = prefer
        while lo - 2 in clear:
            lo -= 2
        while hi + 2 in clear:
            hi += 2
        stop = self._stop_within(b, lo, hi, prefer, k)
        if stop is None:
            return None
        t, x = stop
        land, lx, ta = fly(self.model, x, a.y_at(x) + 1, 0.0, 0.0, skip=a)
        if land is None or land.id != m.dst:
            return None
        t = max(t, b.t + (AFTER_LANDING + b.slack if b.landed else 0.0))
        k.hold(t, t + 0.18, "down")  # let go as soon as the drop has started
        k.tap(t + 0.1, "jump")
        t_off = t + 0.1 + DROP_LAG
        return Leg(m, b, Body(t_off + ta, lx, land, 0, True, b.slack + SLACK_PER_LEG + DROP_SLACK),
                   [Span(a.id, b.t, t, b.x, x)])

    def _climb(self, m: Move, b: Body, k: Keys, variant: str) -> Leg | None:
        if variant != "run":
            return None
        a, top = b.platform, self.model.platform(m.dst)
        rx, gap = m.rope_x, a.y_at(m.rope_x) - m.rope_bottom
        if gap > APEX + 10 or any(abs(p.x - rx) < 60 and abs(p.y - a.y_at(p.x)) < 40 for p in self.model.portals):
            return None
        if abs(rx - b.x) <= 4 and not b.run:
            t, grab_h, ran_to = b.t, min(APEX, max(0.0, gap)), b.x   # straight up under it
            t_rope = self.press_jump(k, t + JUMP_LAG, b) + 0.15
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
            k.set_arrow(t_rope + 0.12, 0)
        climb = max(0.0, (a.y_at(rx) - grab_h) - m.rope_top) / CLIMB_SPEED
        t_top = t_rope + climb + ONTO_TOP
        k.hold(t, t_top, "up")
        return Leg(m, b, Body(t_top + 0.05, rx, top, 0, False, b.slack + SLACK_PER_LEG),
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
            if dist < STOP_RUN + 10:  # a short step: hold the arrow just long enough (walking model)
                ms = max(30.0, (dist - 8.5) / 0.1486)
                d = _sign(target - b.x)
                k.set_arrow(b.t, d)
                k.set_arrow(b.t + ms / 1000, 0)
                return b.t + ms / 1000 + STOP_S, target
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
        if b.run != d and abs(x - b.x) < STOP_RUN + 10:
            return None  # a short shuffle: leave it to the closed loop
        t = self.run_to(b, d, x - d * STOP_RUN)
        if t is None:
            return None
        k.set_arrow(b.t, d)
        k.set_arrow(t, 0)
        return t + STOP_S, x

    # --- monsters ------------------------------------------------------------------------------

    def danger(self, leg: Leg, monsters: list[Monster], linger: float = 0.0) -> tuple[Monster, float] | None:
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
                while t <= span.t1 + 1e-9:
                    lo, hi = mo.reach(t, p)
                    x = span.x_at(t)
                    if lo - touch < x < hi + touch:
                        return mo, t
                    t += 0.05
        return None

    # --- following a chain ---------------------------------------------------------------------

    @staticmethod
    def expected(legs: list[Leg], t: float) -> int | None:
        """The platform she should be on (on the ground) t s into the plan, or None while she's in
        the air, on a rope, or within a moment of a takeoff or landing."""
        margin = 0.12
        for i, leg in enumerate(legs):
            m0 = margin + leg.start.slack  # timings drift as the chain goes on
            for sp in leg.spans:
                if sp.t0 + m0 <= t <= sp.t1 - m0:
                    return sp.platform
            m1 = margin + leg.end.slack
            nxt = legs[i + 1].spans[0].t0 if i + 1 < len(legs) and legs[i + 1].spans else None
            if leg.end.t + m1 <= t and (nxt is None or t <= nxt - m1) and leg.end.landed:
                return leg.end.platform.id
            if nxt is None and leg.end.t + m1 <= t:
                return leg.end.platform.id
        return None

    # --- chains ----------------------------------------------------------------------------------

    def chain(self, moves: list[Move], start: Body, monsters=()) -> tuple[list[Leg], Keys, str]:
        """As many of `moves` as can be done in one go from `start`: their legs, the keys, and why
        the chain stopped ('done', 'unpredictable', 'monster' or 'long'). Each move is done the way
        (variant) the next move can follow on from; the chain stops at a move it can't predict, and
        before a leg a monster could get in the way of."""
        k, legs, b, why = Keys(self.key_names), [], start, "done"
        monsters = list(monsters)
        for i, m in enumerate(moves):
            if i >= MAX_LEGS or b.t > MAX_SECONDS:
                why = "long"
                break
            nxt = moves[i + 1] if i + 1 < len(moves) and i + 1 < MAX_LEGS else None
            order = ("stop", "run", "stand") if nxt is None else self.VARIANTS
            options = []
            for v in order:
                trial = k.copy()
                leg = self.leg(m, b, trial, v)
                if leg is not None and not self.danger(leg, monsters, linger=1.0 if nxt is None else 0.0):
                    options.append((leg, trial))
            if not options:
                why = "monster" if any(self.leg(m, b, k.copy(), v) for v in order) else "unpredictable"
                break

            def follows(leg: Leg) -> bool:
                return nxt is None or any(self.leg(nxt, leg.end, Keys(self.key_names), v) for v in self.VARIANTS)

            leg, trial = next(((lg, tr) for lg, tr in options if follows(lg)), options[0])
            legs.append(leg)
            k, b = trial, leg.end
        if legs:
            k.set_arrow(b.t, 0)  # stop at the end (she slides a little)
        return legs, k, why
