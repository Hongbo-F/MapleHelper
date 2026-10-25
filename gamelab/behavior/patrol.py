"""Grinding's knowledge of the map and its choices (behavior.grind, `tour --loop --grind`): where her kills come from,
where she stands to shoot them, how full each spawn point is likely to be, and the loop over the map that gives the most
EXP a minute. Pure: the map's game data, the planner's route times and her memory of the monsters; nothing here looks or
presses a key. General: nothing is chosen by a platform's or a kind's name.

The map, once (Patrol):
- points: every spawn point of a kind she hunts (Point: its platform, x, kind and the kind's EXP, game data);
- stops (Stop): on each platform from which she can get at spawn points, the x that reaches the most of their EXP: her
  own platform's she walks up to (all of each), another's as much of it as a star reaches from there (attack.spots.
  coverage, about +-60 units up and down: reach.SHOT_UP, SHOT_DOWN). A stop is where a leg of the loop ends; what comes
  into reach on the way she shoots in passing (behavior.hunt, rule 4);
- the route times between stops (motion.route.Planner.arrive: on Tree Dungeon up the tower is slow, 24 to 6 21 s by its
  ropes, down quick, 6 to 24 8 s).

What she believes (Belief, from her memory of the monsters, localization.memory, at every look): how many monsters
each spawn point has now and at any time ahead (`stock`): the monsters remembered there (each the nearest point of its
kind on its platform), else, for a point a look has seen empty or her kill emptied, its refill clock (`not_back`:
nothing for REFILL_D0, then back at REFILL_TAU per point on its platform), else, never seen, Q_UNKNOWN. And what of it is hers
(`share`): by a point with another player by it (their minimap dot, RIVAL_X across and RIVAL_DY up or down), as much as
others leave of its kind, learned from her memory (`take`): of the monsters she saw by another player, those gone
without her kill against those she killed or saw stay TAKE_KEEP_S.

The loop (Patrol.best): a closed round of stops (LOOP_MIN to LOOP_MAX), chosen by its EXP a minute in the long run: each
spawn point it reaches gives its EXP (times her share) each time she comes round and finds it back (`caught`: a round
of cycle s catches 1 / sum over k of not_back(k cycle) monsters a point); the cycle is the round's route times, STOP_S
a stop and her shooting (her kill time a point of EXP, KILL_S_PER_EXP, learned from her throws), less what the danger
hook says touches would cost there (HP_PRICE EXP a bar of HP). Grown from each of the SEEDS best stops by adding the
stop that raises the rate most (at its cheapest place in the round), the best round kept. Standing at one stop
(camping) is a round of one (Patrol.camping), which grinding never takes up but logs, to compare. A new round
replaces the one she follows only when it's LOOP_GAIN better by the same belief, and not within LOOP_MIN_S of the last
change (no flip-flopping).

On Tree Dungeon, Forest Up North II, before she has seen anything (Lucky Seven's kill time): the loop 14, 10, 8, 12, 466
EXP a minute (a round of 50 s: 12 of travel, 33 of shooting; 14 spawn points), where the user grinding made 484 on
2026-10-24; 19-24 round the platforms by the safe ones, 337; camping at 23 229, where her hunt loop camping there made
283 live (her kills from 23 and 24 on 22, 23 and 24).

The next stop (Patrol.choose): along the loop from the last loop stop she was at, the first expected to hold SKIP_FULL
of what it holds full when she gets there (an emptied stretch is skipped until it's due); and a stop whose monsters she
has seen (her memory, not the belief's guesses) pulls her off the loop when it holds PULL_MIN kills' worth and gives
PULL_GAIN times the EXP a second (travel and its kills included) of the loop's next.

The danger hook (`Patrol.danger(platform, x)`, a share of her HP a second spent fighting there): by default 0 where no
monster can touch her (MapModel.safe), else HP_S_PER_SPAWN for each spawn point on that platform of a kind she keeps off
(attack.stance.KEEP_OFF). The positioning model (attack.danger, branch work/positioning) can be plugged in there. What
she loses there is learned as she goes (`hp_rate`: the hook as a prior, HP_PRIOR_S, and the HP she's seen go there).

The numbers marked "guess" are starting values for the simulator's A/B runs; the rest say where they were measured.
"""

import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache

from ..attack import spots
from ..attack.reach import EDGE_MARGIN
from ..attack.stance import KEEP_OFF
from ..localization.mapdata import SPAWN_REACH, MapModel
from ..localization.players import Player, near_rival
from ..motion.physics import WALK_SPEED
from ..perception.kinds import kind_of

# s of her shooting a point of a kill's EXP: Lucky Seven live on Tree Dungeon (tour --loop --skill, 2026-10-24): 3.85
# throws a Horny Mushroom's kill (41 EXP) and 2.95 a Green Mushroom's (26), a throw every THROW_S: 3.2 and 2.5 s, 0.079
# and 0.095 s a point. Learned as she hunts from her throws and the EXP they gave (Patrol.learn_kills), this counting
# as KILL_PRIOR_EXP points of EXP: with the regular attack's stars (12.4 throws a Horny Mushroom with the simulator's
# misses) it's about three times as long.
KILL_S_PER_EXP = 0.085
THROW_S = 0.84         # s between her throws (her attack's lock and its margin: 0.79-0.85 s live)
KILL_PRIOR_EXP = 300.0
STOP_S = 1.0           # guess: s a stop costs beyond its kills (the looks that judge it, settling where she lands)
# The refill of a spawn point she emptied (not_back): the live respawn report on Tree Dungeon (2026-10-24, six runs, 133
# platforms her kills emptied): none back within 5 s in 91%, a median of 13.9 s for a platform's first monster back
# (95% 11.3-16.2), 0.74 not back at 8 s, 0.55 at 12.56 s; nothing for REFILL_D0, then exp(-(dt - D0) / tau) fits it
# with tau REFILL_TAU (0.79 at 8 s, 0.55 at 12.56 s). A platform's first monster back is the soonest of its points, so a
# point on a platform with n of them comes back at n x REFILL_TAU. Scaled by what she sees (Belief.scale): of the points
# her kills emptied and looks watched until a monster came back, the time it took against the model's median.
REFILL_D0 = 5.0
REFILL_TAU = 12.8
REFILL_PRIOR_N = 5     # ... the model counting as this many such refills
WATCH_GAP_S = 2.0      # ... watched: a look saw it empty at most this long before the one that saw it back
Q_UNKNOWN = 0.35       # a point never seen: this full (live: the uncamped 19-21 had a monster 31-39% of the time, 28%
                       # of all points; the simulator's crowd fills every point)
# Another player by a point (their minimap dot within RIVAL_X across and RIVAL_DY up or down: hunt.PLAYER_NEAR and
# PLAYER_DY) in a look within RIVAL_KEEP_S takes its monsters: of its kind, she gets 1 - take(kind), learned from her
# memory (the monsters she saw by another player: gone without a kill of hers, against killed by her or seen there
# TAKE_KEEP_S on), TAKE_PRIOR counting as TAKE_N monsters. The crowding study of 2026-10-24: other players killed Green
# Mushrooms within about 4 s and left Horny Mushrooms alone; the simulator's camps do the same.
RIVAL_X = 200.0
RIVAL_DY = 100.0
RIVAL_KEEP_S = 20.0    # guess
TAKE_PRIOR = 0.5       # guess
TAKE_N = 4.0
TAKE_KEEP_S = 10.0
# The danger hook's default: a share of her HP a second fighting on a platform with a spawn point of a kind she keeps
# off (stance.KEEP_OFF): live, hunting 23 (two Horny Mushroom points) on 2026-10-24 she lost 0.28-0.31 bars of HP a
# minute, 0.0025 a second a point. What a bar costs her, in EXP (her pet's potions, and the risk): HP_PRICE, a guess.
HP_S_PER_SPAWN = 0.0025
HP_PRICE = 100.0
# ... and blended with the HP she has seen go on each platform (Patrol.learn_hp: the drops of her HP bar between looks
# there), the hook counting as HP_PRIOR_S s there: in the simulator's crowd on Tree Dungeon she lost 1.3-1.4 bars a
# minute alive whatever she did (20 bars a run, her 60 potions), live 0.28-0.33.
HP_PRIOR_S = 60.0
# The loop.
LOOP_MAX = 12          # stops in a round at most ...
LOOP_MIN = 2           # ... and at least: one is camping (Patrol.camping), which grinding doesn't do
SEEDS = 8              # rounds grown, from the stops worth the most on their own
STAND_STEP = 20.0      # units between the places along a platform tried as a stop
CAUGHT_TERMS = 400     # rounds summed at most for what she catches at a point (caught)
LOOP_GAIN = 0.15       # guess: a new round replaces hers only this much better by the same belief ...
LOOP_MIN_S = 60.0      # ... and not within this long of the last change
REPLAN_S = 20.0        # s between two looks for a better round
# The next stop.
# A stop of the loop is skipped when it's expected to hold under this share of what it holds full when she gets there
# (an emptied stretch, until it's due): under Q_UNKNOWN, so a stop never seen is never skipped.
SKIP_FULL = 0.25
PULL_MIN = 2.0         # guess: a stop pulls her off the loop holding this many kills' worth she has seen (her kinds'
                       # average EXP) ...
PULL_GAIN = 1.5        # guess: ... giving this many times the loop's next stop's EXP a second
DUE_S = 2.0            # guess: at a stop, a respawn in reach due within this many s is waited for


@dataclass(frozen=True)
class Point:
    """A spawn point of a kind she hunts."""
    id: int
    platform: int
    x: float
    kind: str
    exp: float


@dataclass(frozen=True)
class Stop:
    """Where a leg of the loop ends: on `platform` at map x `x`, from where she gets at the points `cover` (point id,
    the share of it she gets at: 1 on her own platform, the share of its platform a star reaches on another)."""
    platform: int
    x: float
    cover: tuple[tuple[int, float], ...]


@dataclass
class Plan:
    """A loop and what it gives in the long run."""
    stops: tuple[int, ...]   # Patrol.stops indices, in the order she goes round
    rate: float              # EXP a second (less the danger's cost)
    cycle: float             # s a round takes
    travel: float            # ... of which route times
    kills: float             # ... of which shooting
    exp: float               # EXP a round
    points: int              # spawn points it gets at
    hp: float = 0.0          # bars of HP a round (the danger hook)

    def text(self, plats: Callable[[int], int]) -> str:
        return (f"{[plats(i) for i in self.stops]}: {self.rate * 60:.0f} EXP a minute (a round {self.cycle:.0f} s: "
                f"{self.travel:.0f} s of travel, {self.kills:.0f} of shooting; {self.points} spawn points, "
                f"{self.exp:.0f} EXP)")

    def record(self, plats: Callable[[int], int]) -> dict:
        return {"loop": [plats(i) for i in self.stops], "rate": round(self.rate * 60, 1), "cycle": round(self.cycle, 1),
                "travel": round(self.travel, 1), "shooting": round(self.kills, 1), "points": self.points,
                "exp": round(self.exp, 1)}


def points_of(model: MapModel, kinds: set[str] | None, exp: dict[str, float]) -> list[Point]:
    """The map's spawn points of `kinds` (None: any) that give EXP, each on the platform under it (MapModel's rule:
    monsters_by_platform's)."""
    out = []
    for s in model.spawns:
        p = model.platform_under(s.x, s.y, reach=SPAWN_REACH)
        kind = kind_of(s.name or str(s.mob_id))
        if p is None or (kinds is not None and kind not in kinds) or exp.get(kind, 0) <= 0:
            continue
        out.append(Point(len(out), p.id, float(s.x), kind, float(exp[kind])))
    return out


def stops_of(model: MapModel, points: list[Point]) -> list[Stop]:
    """A stop on each platform from which she gets at points: the x (every STAND_STEP, EDGE_MARGIN from its ends) that
    reaches the most EXP, the nearest its middle of those; her own platform's points count whole wherever she stands."""
    out = []
    for p in model.places:
        lo, hi = p.left + EDGE_MARGIN, p.right - EDGE_MARGIN
        mid = (p.left + p.right) / 2
        xs = [mid] if hi < lo else [lo + i * STAND_STEP for i in range(int((hi - lo) // STAND_STEP) + 1)] + [hi]
        own = [(q.id, 1.0) for q in points if q.platform == p.id]
        best = None  # (EXP reached, -distance from the middle, x, cover)
        for x in xs:
            y = p.y_at(x)
            cov, kept = list(own), {}
            for q in points:
                if q.platform == p.id:
                    continue
                key = (q.platform, q.kind)
                if key not in kept:
                    kept[key] = spots.coverage(model.platform(q.platform), x, y, p, q.kind)
                if kept[key] > 0:
                    cov.append((q.id, round(kept[key], 3)))
            value = sum(points[i].exp * c for i, c in cov)
            if value > 0 and (best is None or (round(value, 6), -abs(x - mid)) > best[:2]):
                best = (round(value, 6), -abs(x - mid), x, tuple(cov))
        if best is not None:
            out.append(Stop(p.id, round(best[2], 1), best[3]))
    return out


def not_back(dt: float, n: int = 1, scale: float = 1.0) -> float:
    """The chance a spawn point emptied `dt` s ago has no monster yet, on a platform with `n` points (see the top)."""
    d0 = REFILL_D0 * scale
    return 1.0 if dt < d0 else math.exp(-(dt - d0) / (max(1, n) * REFILL_TAU * scale))


def caught(cycle: float, n: int = 1, scale: float = 1.0) -> float:
    """The monsters she kills at a point a round, going round every `cycle` s, in the long run (on a platform with `n`
    points): after a kill it's back R s later (not_back), and she kills it at her first round after that, ceil(R /
    cycle) rounds on; so one every cycle x sum over k >= 0 of not_back(k cycle) s on average. 1 a round when it's always
    back in time; about cycle / its mean refill when she comes round far more often (waiting at one, camping, the same).
    Kept by the cycle to a tenth of a second (a loop's search asks it thousands of times)."""
    return _caught(round(cycle, 1), max(1, n), round(scale, 3))


@lru_cache(maxsize=20000)
def _caught(cycle: float, n: int, scale: float) -> float:
    total, k = 0.0, 0
    while k < CAUGHT_TERMS:
        s = not_back(k * cycle, n, scale)
        if s < 1e-4:
            break
        total += s
        k += 1
    return 1.0 / max(total, 1.0)


def refill_median(n: int = 1, scale: float = 1.0) -> float:
    """s until a point emptied is back with even odds (not_back's median)."""
    return (REFILL_D0 + max(1, n) * REFILL_TAU * math.log(2)) * scale


def default_danger(model: MapModel, points: list[Point]) -> Callable[[int, float], float]:
    """The danger hook's default (see the top): HP_S_PER_SPAWN for each point of a KEEP_OFF kind on the platform, 0
    where MapModel.safe."""
    n = Counter(q.platform for q in points if q.kind in KEEP_OFF)

    def danger(pid: int, x: float) -> float:
        return 0.0 if model.safe(pid) else HP_S_PER_SPAWN * n.get(pid, 0)
    return danger


class Belief:
    """How many monsters each spawn point has, from her memory (see the top). `update` at every look."""

    def __init__(self, model: MapModel, points: list[Point]):
        self.model, self.points = model, points
        self.n_on = Counter(q.platform for q in points)
        self._by = {}
        for q in points:
            self._by.setdefault((q.platform, q.kind), []).append(q)
        self.at: dict[int, list] = {}            # point id -> the monsters remembered at it (localization.memory)
        self.empty_since: dict[int, float] = {}  # point id -> since when it's empty (a look without it, her kill)
        self.killed: dict[int, float] = {}       # point id -> her kill that emptied it (refills learned from)
        self.empty_seen: dict[int, float] = {}   # point id -> the latest look that saw it empty
        self.rival_at: dict[int, float] = {}     # point id -> the latest look with another player by it
        self.gone_by: Counter = Counter()        # kind -> monsters by another player gone without her kill
        self.kept_by: Counter = Counter()        # kind -> ... killed by her, or seen there TAKE_KEEP_S on
        self.refills: list[float] = []           # refills watched, against the model's median
        self.t = 0.0
        self._events = 0
        self._kept_ids: set[int] = set()
        self._covered_t = -1e18

    # --- what it holds ---------------------------------------------------------------------------------

    @property
    def scale(self) -> float:
        """The refill model's time scale: 1, blended with the refills watched (REFILL_PRIOR_N)."""
        return (REFILL_PRIOR_N + sum(self.refills)) / (REFILL_PRIOR_N + len(self.refills))

    def stock(self, i: int, t: float) -> float:
        """How many monsters point `i` is expected to have at time `t` (now or ahead)."""
        es = self.at.get(i)
        if es:
            return sum(e.p(t) for e in es)
        since = self.empty_since.get(i)
        if since is not None:
            return 1.0 - not_back(t - since, self.n_on[self.points[i].platform], self.scale)
        return Q_UNKNOWN

    def seen(self, i: int, t: float) -> float:
        """Of stock, the monsters she has seen (her memory's), not the model's guesses."""
        return sum(e.p(t) for e in self.at.get(i, ()))

    def take(self, kind: str) -> float:
        """The share of its kind's monsters another player by a point takes (see the top)."""
        g, k = self.gone_by[kind], self.kept_by[kind]
        return (g + TAKE_PRIOR * TAKE_N) / (g + k + TAKE_N)

    def share(self, i: int, t: float) -> float:
        """The share of point `i`'s monsters that is hers: 1 - take with another player by it lately, else 1."""
        return 1.0 - self.take(self.points[i].kind) if t - self.rival_at.get(i, -1e18) <= RIVAL_KEEP_S else 1.0

    def due(self, i: int, t: float) -> float | None:
        """s from `t` until point `i`, empty, is back with even odds (None: not known to be empty)."""
        since = self.empty_since.get(i)
        if since is None or self.at.get(i):
            return None
        return since + refill_median(self.n_on[self.points[i].platform], self.scale) - t

    # --- keeping it up ---------------------------------------------------------------------------------

    def update(self, t: float, memory, players: list[Player] | None = None) -> None:
        """At a look at time `t`: the monsters her memory has (localization.memory.Memory) put at their points, her
        kills and the looks that covered a platform (since the last update) emptying points, other players by them."""
        self.t = t
        self._take_events(t, memory)
        at: dict[int, list] = {}
        for (pid, kind), qs in self._by.items():
            es = [e for e in memory.entries if e.live and e.platform == pid and e.kind == kind]
            if not es:
                continue
            pairs = sorted((abs(e.x - q.x), e.id, q.id, e) for e in es for q in qs)
            used_e, used_q = set(), set()
            for _, eid, qid, e in pairs:  # one monster a point first, the nearest pairs first
                if eid in used_e or qid in used_q:
                    continue
                used_e.add(eid)
                used_q.add(qid)
                at.setdefault(qid, []).append(e)
            for e in es:  # more monsters than points: each at its nearest
                if e.id not in used_e:
                    at.setdefault(min(qs, key=lambda q: abs(q.x - e.x)).id, []).append(e)
        self.at = at
        for i, es in at.items():
            since = self.empty_since.get(i)
            back = [e for e in es if since is not None and e.first > since]
            if since is not None and back:
                first = min(e.first for e in back)
                if i in self.killed and first - self.empty_seen.get(i, -1e18) <= WATCH_GAP_S:
                    watched = (self.empty_seen[i] + first) / 2 - self.killed[i]
                    n = self.n_on[self.points[i].platform]
                    self.refills.append(max(0.0, watched) / refill_median(n))
                self.empty_since.pop(i, None)
                self.killed.pop(i, None)
        for q in self.points:  # a look covering its platform since the last update, nothing sure there: empty
            covered = memory.covered.get(q.platform)
            if covered is None or covered <= self._covered_t:
                continue
            if not any(e.p(covered) >= 0.5 for e in at.get(q.id, ())):
                self.empty_since.setdefault(q.id, covered)
                self.empty_seen[q.id] = covered
        self._covered_t = max([self._covered_t, *memory.covered.values()])
        if players is not None:
            for q in self.points:
                y = self.model.platform(q.platform).y_at(q.x)
                if near_rival(players, q.x, y, RIVAL_X, RIVAL_DY) is not None:
                    self.rival_at[q.id] = t
        for e in memory.entries:  # by another player, and still there TAKE_KEEP_S on: left to her
            if e.live and e.rival and e.id not in self._kept_ids and t - e.first >= TAKE_KEEP_S:
                self._kept_ids.add(e.id)
                self.kept_by[e.kind] += 1

    def _take_events(self, t: float, memory) -> None:
        """Her memory's new events since the last update: her kills empty their points; the monsters by another player
        gone or killed teach `take`."""
        c = memory.counts
        total = c["new"] + c["killed"] + c["gone"] + c["aged"]
        new, self._events = min(total - self._events, len(memory.events)), total
        for what, when, e in list(memory.events)[len(memory.events) - new:] if new > 0 else ():
            if what == "killed":
                qs = self._by.get((e.platform, e.kind))
                if qs:
                    x = e.died_x if e.died_x is not None else e.x
                    q = min(qs, key=lambda q: abs(q.x - x))
                    self.empty_since[q.id] = self.killed[q.id] = when
                    self.empty_seen[q.id] = when
                if e.rival and e.id not in self._kept_ids:
                    self._kept_ids.add(e.id)
                    self.kept_by[e.kind] += 1
            elif what == "gone" and e.rival and e.id not in self._kept_ids:
                self._kept_ids.add(e.id)
                self.gone_by[e.kind] += 1


class Patrol:
    """The map's points and stops, the belief, the loop she follows and the choice of her next stop (see the top).
    `travel(platform, x, platform, x)`: s by the planner's routes (None: no way); `danger(platform, x)`: the hook, a
    share of her HP a second spent fighting there (None: default_danger)."""

    def __init__(self, model: MapModel, kinds: set[str] | None, exp: dict[str, float],
                 travel: Callable[[int, float, int, float], float | None],
                 danger: Callable[[int, float], float] | None = None):
        self.model = model
        self.points = points_of(model, kinds, exp)
        self.stops = stops_of(model, self.points)
        self.belief = Belief(model, self.points)
        self.travel = travel
        self.danger = danger or default_danger(model, self.points)
        avg = [q.exp for q in self.points]
        self.avg_exp = sum(avg) / len(avg) if avg else 10.0
        self.kill_s = KILL_S_PER_EXP   # s of shooting a point of EXP (learn_kills)
        self.plan: Plan | None = None  # the loop she follows
        self.changed = -1e18           # when it last changed
        self.planned = -1e18           # when a better one was last looked for
        self.at_loop = 0               # the loop stop she was at last (index into plan.stops)
        self.loops = 0                 # rounds taken up (the first, and every change)
        self.hp_seen: Counter = Counter()  # platform -> the share of her HP she has seen go there (learn_hp) ...
        self.hp_time: Counter = Counter()  # ... over this many s there
        self._legs: dict[tuple[int, int], float | None] = {}

    def platform_of(self, i: int) -> int:
        return self.stops[i].platform

    def stop_on(self, pid: int) -> int | None:
        """The stop on platform `pid`, if there's one."""
        return next((i for i, s in enumerate(self.stops) if s.platform == pid), None)

    def leg(self, i: int, j: int) -> float | None:
        """s from stop i to stop j (planner), kept."""
        if i == j:
            return 0.0
        key = (i, j)
        if key not in self._legs:
            a, b = self.stops[i], self.stops[j]
            self._legs[key] = self.travel(a.platform, a.x, b.platform, b.x)
        return self._legs[key]

    def learn_hp(self, pid: int, lost: float, dt: float) -> None:
        """`dt` s on platform `pid`, with `lost` of her HP gone meanwhile (hp_rate)."""
        self.hp_seen[pid] += lost
        self.hp_time[pid] += dt

    def hp_rate(self, pid: int, x: float) -> float:
        """The share of her HP a second she loses fighting on platform `pid` at x: the danger hook, as HP_PRIOR_S s
        there, blended with what she has seen go there."""
        return (self.danger(pid, x) * HP_PRIOR_S + self.hp_seen[pid]) / (HP_PRIOR_S + self.hp_time[pid])

    def learn_kills(self, throws: int, exp: float) -> None:
        """Her kill time a point of EXP from the throws she made and the EXP her kills gave (KILL_PRIOR_EXP)."""
        self.kill_s = (KILL_S_PER_EXP * KILL_PRIOR_EXP + THROW_S * throws) / (KILL_PRIOR_EXP + exp)

    # --- a round's worth in the long run ---------------------------------------------------------------

    def rate(self, loop: Iterable[int], t: float | None = None) -> Plan | None:
        """What going round `loop` gives in the long run (see the top), with the other players' shares as at time `t`
        (None: the belief's latest); None if a leg has no way."""
        loop = tuple(loop)
        t = self.belief.t if t is None else t
        travel = 0.0
        for a, b in zip(loop, loop[1:] + loop[:1]):
            if len(loop) == 1:
                break
            d = self.leg(a, b)
            if d is None:
                return None
            travel += d
        cov: dict[int, tuple[float, int]] = {}  # point -> (share in reach, the stop it's got at from)
        for i in loop:
            for q, c in self.stops[i].cover:
                if q not in cov or c > cov[q][0]:
                    cov[q] = (c, i)
        b = self.belief
        pts = [(self.points[q], c * b.share(q, t), i) for q, (c, i) in cov.items()]
        hp_s = {i: self.hp_rate(self.stops[i].platform, self.stops[i].x) for i in loop}
        scale = b.scale
        base = travel + STOP_S * len(loop)
        cycle = base + sum(g * q.exp * self.kill_s for q, g, _ in pts)  # every point full: the longest a round takes
        for _ in range(8):  # what's back in a round, and the round's shooting for it (cycle only shortens)
            got = [(q, g * caught(cycle, b.n_on[q.platform], scale), i) for q, g, i in pts]
            shoot = sum(n * q.exp * self.kill_s for q, n, _ in got)
            cycle = max(base + shoot, 1e-6)
        exp = sum(n * q.exp for q, n, _ in got)
        hp = sum(n * q.exp * self.kill_s * hp_s[i] for q, n, i in got)
        return Plan(loop, (exp - HP_PRICE * hp) / cycle, cycle, travel, shoot, exp, len(pts), hp)

    def camping(self, t: float | None = None) -> Plan | None:
        """The best round of one stop: standing there, each point's monster killed as it comes back (caught). Grinding
        never takes it up (a round has LOOP_MIN stops); it says what camping would give, for the log and the tests."""
        return max((p for i in range(len(self.stops)) if (p := self.rate((i,), t)) is not None),
                   key=lambda p: p.rate, default=None)

    def best(self, t: float | None = None) -> Plan | None:
        """The best round of LOOP_MIN stops or more (see the top): grown from each of the SEEDS stops worth the most on
        their own, by the stop that raises its rate most, the second stop taken whether it does or not."""
        alone = [p for i in range(len(self.stops)) if (p := self.rate((i,), t)) is not None]
        seeds = [p.stops[0] for p in sorted(alone, key=lambda p: -p.rate)[:SEEDS]]
        top = None
        for seed in seeds:
            loop, cur = [seed], None
            while len(loop) < LOOP_MAX:
                found = None
                for j in range(len(self.stops)):
                    if j in loop:
                        continue
                    at = self._insert(loop, j)
                    if at is None:
                        continue
                    p = self.rate(loop[:at] + [j] + loop[at:], t)
                    if p is not None and (found is None or p.rate > found.rate):
                        found = p
                if found is None or (cur is not None and found.rate <= cur.rate * (1 + 1e-3)):
                    break
                loop, cur = list(found.stops), found
            if cur is not None and (top is None or cur.rate > top.rate):
                top = cur
        return top

    def _insert(self, loop: list[int], j: int) -> int | None:
        """Where in `loop` stop j adds the least travel (None: no way to it or back)."""
        best = None
        for k in range(len(loop)):
            a, b = loop[k - 1], loop[k]
            ab, aj, jb = self.leg(a, b), self.leg(a, j), self.leg(j, b)
            if aj is None or jb is None or ab is None:
                continue
            extra = aj + jb - (ab if len(loop) > 1 else 0.0)
            if best is None or extra < best[0]:
                best = (extra, k)
        return None if best is None else best[1]

    def replan(self, t: float, force: bool = False) -> str | None:
        """Look for a better round (every REPLAN_S): taken up when it's LOOP_GAIN better than hers by the same belief
        and LOOP_MIN_S after the last change (or she has none, or hers has no way round). Why it changed, or None."""
        if not force and self.plan is not None and t - self.planned < REPLAN_S:
            return None
        self.planned = t
        new = self.best(t)
        if new is None:
            return None
        if self.plan is None:
            return self._take(new, t, "the first")
        if set(new.stops) == set(self.plan.stops):
            self.plan = new if new.stops == self.plan.stops else self.plan
            return None
        now = self.rate(self.plan.stops, t)
        if now is None:
            return self._take(new, t, "hers has no way round")
        if t - self.changed < LOOP_MIN_S or new.rate < (1 + LOOP_GAIN) * now.rate:
            self.plan = now
            return None
        return self._take(new, t, f"{new.rate * 60:.0f} EXP a minute against {now.rate * 60:.0f}")

    def _take(self, plan: Plan, t: float, why: str) -> str:
        self.plan, self.changed, self.at_loop = plan, t, 0
        self.loops += 1
        return why

    # --- the next stop -----------------------------------------------------------------------------------

    def expect(self, i: int, t: float) -> tuple[float, float, float, float]:
        """At stop i at time `t`: the EXP expected in reach (hers: share), of it the EXP of monsters she has seen, the s
        of shooting it takes, and the EXP it holds full (a monster at each point)."""
        b = self.belief
        exp = seen = full = 0.0
        for q, c in self.stops[i].cover:
            g = c * b.share(q, t) * self.points[q].exp
            exp += g * b.stock(q, t)
            seen += g * b.seen(q, t)
            full += g
        return exp, seen, exp * self.kill_s, full

    def due(self, i: int, t: float) -> float | None:
        """s until the soonest respawn due at stop i's points (in reach of it), if one is."""
        due = [d for q, c in self.stops[i].cover if c > 0 and (d := self.belief.due(q, t)) is not None]
        return min(due) if due else None

    def choose(self, t: float, here: int | None, x: float | None, current: int | None = None,
               skip: Iterable[int] = ()) -> tuple[int | None, dict]:
        """The next stop from (platform `here`, map x `x`), having been at stop `current`, never one on a platform in
        `skip` (see the top): (its index, what decided it, for the log and the telemetry)."""
        skip = set(skip)
        if self.plan is None:
            self.replan(t, force=True)
        if self.plan is None:
            return None, {"why": "no stop on this map"}
        loop = list(self.plan.stops)
        if current in loop:
            self.at_loop = loop.index(current)
        options = {}
        for j in range(len(self.stops)):
            s = self.stops[j]
            if j == current or s.platform in skip:
                continue
            if here is None:
                far = 0.0
            elif here == s.platform:
                far = abs(s.x - x) / WALK_SPEED
            else:
                far = self.travel(here, x, s.platform, s.x)
            if far is None:
                continue
            exp, seen, shoot, full = self.expect(j, t + far)
            options[j] = (far, exp, seen, exp / (far + shoot + STOP_S), full)
        order = [loop[(self.at_loop + k) % len(loop)] for k in range(1, len(loop) + 1)]
        order = [j for j in order if j in options] or [j for j in sorted(options, key=lambda j: options[j][0])]
        if not order:
            return None, {"why": "no stop to get to"}
        nxt, skipped = None, []
        for j in order:
            if options[j][1] >= SKIP_FULL * options[j][4] and options[j][4] > 0:
                nxt = j
                break
            skipped.append(j)
        why = "the loop"
        if nxt is None:
            nxt, skipped, why = order[0], [], "the loop, nothing due ahead"
        best = max((j for j in options if j != nxt and options[j][2] >= PULL_MIN * self.avg_exp),
                   key=lambda j: options[j][3], default=None)
        if best is not None and options[best][3] >= PULL_GAIN * options[nxt][3]:
            pulled_from, nxt, why = nxt, best, "pulled"
        else:
            pulled_from = None
        if nxt in loop:
            self.at_loop = loop.index(nxt)
        far, exp, seen, gain, _ = options[nxt]
        info = {"why": why, "stop": self.stops[nxt].platform, "x": self.stops[nxt].x, "travel": round(far, 1),
                "expected": round(exp, 1), "seen": round(seen, 1), "gain": round(gain, 2),
                "skipped": [{"stop": self.stops[j].platform, "expected": round(options[j][1], 1),
                             "due": None if (d := self.due(j, t)) is None else round(d, 1)} for j in skipped]}
        if pulled_from is not None:
            info["instead_of"] = {"stop": self.stops[pulled_from].platform, "gain": round(options[pulled_from][3], 2)}
        return nxt, info


def choice_text(info: dict) -> str:
    """A next stop's choice (Patrol.choose's info) for the log."""
    out = (f"stop {info['stop']} at x {info['x']:.0f}, {info['travel']:.1f} s away: {info['expected']:.0f} EXP "
           f"expected ({info['seen']:.0f} seen), {info['gain']:.1f} EXP/s with its kills; {info['why']}")
    if "instead_of" in info:
        o = info["instead_of"]
        out += f" (stop {o['stop']}'s {o['gain']:.1f} EXP/s)"
    if info.get("skipped"):
        out += "; skipped " + ", ".join(f"{s['stop']} ({s['expected']:.0f} EXP"
                                         + ("" if s["due"] is None else f", due in {s['due']:.0f} s") + ")"
                                         for s in info["skipped"])
    return out
