"""The loot sweep: which of the drops she knows of she picks up on her way from where she is to where she's going, and in
what order (the user, 2026-10-30: "For loot, please be more intelligent and heuristic, before leaving an area you should
try to pick as much as possible without too much impact on travel time, and in general you should kill all the mobs that
are in the way, try not to hit by mobs just for picking up loots"; 2026-11-01: "I imagine piles should be picked up soon
after killing. If a pile is about to expire, it probably means they aren't important enough to get picked up anyway";
the designs and their measurements: D:/MapleHelper-scratch/replan/loot, loot2). Pure, as behavior.patrol is: the hunt
loop gives it the pick spots (the drop store's piles, each with what it's worth, when its drops land, how long it's
likely to lie, what guards it and how much it may cost her way) and her route times (motion.route.Planner.arrive), and
it gives back an itinerary.

An orienteering problem in seconds, not in an exchange of mesos for EXP: of the candidates, the set and the order that
pick up the most of

    sum of worth_i x P(still there when she gets there | there now)

(her pets' pick-ups and hers alike: what's left of a pile is what a look sees), within what they may cost her way: each
spot's own `allow` (s it may add to her way on its own: the hunt's, by the pile's age and the danger there,
Hunt._allow), and the whole itinerary's `extra` (the time the route through them and on to where she's going takes over
going there straight, the landings waited for in it, not its guards' kill times: those are EXP) within the sum of its
spots' allowances and MOST_S, and within `budget` where one is given (the hunt loop's lull, without MOST_S, and its way
out). Of those that pick up as much (to GAIN_TIE), the quickest. Until 2026-11-01 the most of worth x MESO_EXP and the
guards' EXP less her time's worth (the loop's rate x extra) within a budget of max(BUDGET_MIN, BUDGET_SHARE x the
straight way), each spot CAND_S (5 s) at most: the user wasn't sure of that exchange ("My idea is just picking up mesos
and items as we kill mobs"), and at a stop its 2 s budget never fetched a fresh coin 160 units off (the independent
check of 2026-11-01, finding 2). Every spot's stand is checked against the monsters that count where she'd stand, each
walking the way perception says it heads at its kind's speed from the game data, turned back at its platform's ends
(`meets`: her body's box against theirs over the stand and STAND_PAST_S more); one that meets a monster she can shoot
first from where she is has the shot in its time (a Guard: "kill all the mobs that are in the way"), one that meets
another is left out for this solve ("try not to hit by mobs just for picking up loots"). Her legs between spots are the
navigator's trips, planned and replanned at every state against every monster as any trip (control.navigator: REPLAN,
its shots first at what's in its way).

Solved exactly in its travel: of at most MAX_CAND candidates (the most worth for their insertion's time first, each
alone within its allowance), a dynamic program over subsets with the last spot visited (the least time to have picked up
a set ending at a spot), each set's quickest order weighed by the survival of its drops at its arrivals (they change by
a few % over the seconds a sweep takes). On the live thief run of 2026-10-30 the prototype (sweep_proto.py, without
monsters or survival) solved 353 times at a median 0.62 ms (p99 4.9, the most 8.4), 3 candidates at the median, 8 at the
most.

The itinerary changes only for one that picks up KEEP_SHARE more (`Sweeper.solve`), while it keeps within its
allowances and KEEP_S: solved again at every look, two piles each worth the other's detour turned her from one to the
other at every look (patrol's warning case, loop_tree_grind_jump_key).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

# The hunt loop's budgets (behavior.hunt: its way out, DEPART_BUDGETS; a lull's wait, at least this): s a sweep may add
# to her way at least, or BUDGET_SHARE of the straight way, the more. The sweep's own cap until 2026-11-01 (live on
# 2026-10-30 her lost drops' worth was a marginal 49, 33, 24, 14, 10 and 6.6 a second of extra travel at <=0.6, 0.6-1,
# 1-1.5, 1.5-2, 2-3 and 3-5 s: insertion.py; 48% of the worth she lost lay within 2 s of her way, 66% within 3).
BUDGET_MIN = 2.0
BUDGET_SHARE = 0.5
ALLOW = BUDGET_MIN   # s a spot may add to her way where its maker gives none (Spot.allow)
# s an itinerary may add to her way at most, its guards' kills not counted, whatever its spots' allowances come to
# (`plan`'s `most`; None: their sum alone, the hunt loop's lull): two fresh piles' (hunt.ALLOW_FRESH), from the review
# of 2026-11-01 ("we shouldn't wait too long and take a long detour to pick them up"): their sum alone (up to MAX_CAND x
# 3 s) had her plan +11.6 and then +18.2 s on a 2.6 s way to her next stop for 260 mesos (the harness's seed 17 at 136
# s), over 10 s in 1 of 8 of its runs of 5 minutes and in 3 of 8 with her pets. A guess, measured in the README.
MOST_S = 6.0
MAX_CAND = 8         # candidates at most (the dynamic program's 2^8 sets x 8 ends)
KEEP_SHARE = 0.2     # the itinerary she's on is kept unless another picks up this much more ...
KEEP_S = 0.5         # ... while it costs her way no more than this over its allowances
GAIN_TIE = 0.1       # mesos: itineraries that pick up as much to this are told apart by their time
STAND_PAST_S = 0.5   # s after a stand that its check covers (she steps off, a monster walking in meanwhile)
CONTACT_MARGIN = 10.0  # units more between her body's box and a monster's than touching, for a stand
STEP_S = 0.1         # s between the times a stand's check looks at a monster's place
# s a monster whose heading isn't known (her memory's, about though the look missed it, a frame that doesn't tell, or
# one standing or flinching, which may set off any time: hunt._reach_of, since the review of 2026-10-30)
# is taken to walk, either way at its kind's speed, for a stand's check: the next looks tell (the sweep is solved again
# at every one, and the stand is checked again from the look she's there on). Taken for as long as a stand 3 s off is
# away, it was anywhere within 300 units of where it was last seen, and in the simulator's camping hunt loop
# (loop_tree_mp_runs_out) every pile on the ground she shot was left (none of 26 picked up, where the old PILE_CLEAR's
# 100 units had them all); at 1 s, 100 units, as PILE_CLEAR had them.
UNSURE_S = 1.0


@dataclass(frozen=True)
class Guard:
    """A monster to shoot first before a spot's stand (it'd meet her there), from where she is: how long its kill takes
    (Attacks.kill_s) and what it gives."""
    monster: object
    kill_s: float
    exp: float


@dataclass(frozen=True)
class Spot:
    """A pick spot: where she stands to pick up a pile (map x on `platform`), what it's worth in mesos, how long the
    stand takes (s), s from now until its drops lie on the ground (0: they do), the chance they're still there `dt` s
    from now (`still`, given they are now), what to shoot first (`guards`), and the s it may add to her way (`allow`:
    on its own, and its share of an itinerary's)."""
    key: object
    platform: int
    x: float
    worth: float
    stand: float
    lands: float = 0.0
    still: Callable[[float], float] = lambda dt: 1.0
    guards: tuple[Guard, ...] = ()
    allow: float = ALLOW

    @property
    def clear_s(self) -> float:
        return sum(g.kill_s for g in self.guards)


@dataclass
class Stop:
    """A spot of an itinerary: when the stand starts (s from now: its landing waited for, its guards shot first) and ends."""
    spot: Spot
    arrive: float
    leave: float


@dataclass
class Itinerary:
    """The spots to pick up, in order, then on to where she's going: the time it adds to her way (`extra`), what it
    picks up (`value`, mesos: each spot's worth by its chance of lying there still when she's there), the straight way's
    time, the budget it kept to (None: its allowances alone), and why each candidate it didn't take was left out (`out`:
    by Spot.key)."""
    stops: list[Stop] = field(default_factory=list)
    extra: float = 0.0
    value: float = 0.0
    direct: float = 0.0
    budget: float | None = None
    out: dict = field(default_factory=dict)

    @property
    def first(self) -> Stop | None:
        return self.stops[0] if self.stops else None

    def keys(self) -> tuple:
        return tuple(st.spot.key for st in self.stops)

    @property
    def clear(self) -> float:
        """The s of its guards' kills (EXP: not charged to its allowances)."""
        return sum(st.spot.clear_s for st in self.stops)

    @property
    def allowed(self) -> float:
        """The s its spots may add to her way together (each Spot.allow)."""
        return sum(st.spot.allow for st in self.stops)

    def record(self) -> dict:
        """The 'sweep' record's (telemetry): the order chosen (each stop's platform, x, arrival, worth, guards and
        allowance), its numbers, and why the rest were left out."""
        return {"order": [[st.spot.platform, round(st.spot.x), round(st.arrive, 2), round(st.spot.worth, 1),
                           len(st.spot.guards), round(st.spot.allow, 2)] for st in self.stops],
                "extra": round(self.extra, 2), "value": round(self.value, 2), "direct": round(self.direct, 2),
                "budget": None if self.budget is None else round(self.budget, 2),
                "out": dict(sorted(((str(k), v) for k, v in self.out.items())))}


Arrive = Callable[[int, float, int, float], float | None]


def budget_for(direct: float) -> float:
    """What the hunt loop's budgets give a way that takes `direct` s straight: BUDGET_MIN at least, BUDGET_SHARE of
    it."""
    return max(BUDGET_MIN, BUDGET_SHARE * direct)


def plan(spots: list[Spot], here: tuple[int, float], dest: tuple[int, float] | None, arrive: Arrive,
         budget: float | None = None, most: float | None = MOST_S) -> Itinerary:
    """The best itinerary through `spots` from `here` (platform, x) to `dest` (platform, x; None: nowhere after, the
    route ends at its last spot), by her route times `arrive(platform, x, platform, x)` (s, or None: no way): the most
    mesos likely picked up, each spot within its `allow` of her way on its own and all of them within the sum of
    theirs and `most` (None: the sum alone; their guards' kill times not counted), and within `budget` s more than the
    straight way where one is given; of those as good (GAIN_TIE), the quickest. The empty itinerary if none fits."""
    direct = 0.0 if dest is None else (arrive(*here, *dest) or 0.0) if (here[0], here[1]) != dest else 0.0
    out = Itinerary(direct=direct, budget=budget)
    memo: dict = {}

    def t(a: tuple[int, float], b: tuple[int, float]) -> float | None:
        if a == b:
            return 0.0
        k = (a[0], a[1], b[0], b[1])
        if k not in memo:
            memo[k] = arrive(a[0], a[1], b[0], b[1])
        return memo[k]

    def on(p: tuple[int, float]) -> float | None:  # from a spot on to where she's going
        return 0.0 if dest is None else t(p, dest)

    cands = []
    for sp in spots:
        at = (sp.platform, sp.x)
        to, after = t(here, at), on(at)
        if to is None or after is None:
            out.out[sp.key] = "no way"
            continue
        start = max(to, sp.lands) + sp.clear_s
        ins = start + sp.stand + after - direct - sp.clear_s  # (on its own, but its guards' kills)
        if ins > sp.allow + 1e-9:
            out.out[sp.key] = f"cost {ins:.1f} s over {sp.allow:.1f}"
            continue
        if sp.still(start) <= 0.0:
            out.out[sp.key] = "expired"
            continue
        cands.append((sp.worth * sp.still(start) / max(ins, 0.1), sp))
    cands.sort(key=lambda c: -c[0])
    for _, sp in cands[MAX_CAND:]:
        out.out[sp.key] = "not among the best"
    cands = [sp for _, sp in cands[:MAX_CAND]]
    n = len(cands)
    if not n:
        return out
    places = [(sp.platform, sp.x) for sp in cands]
    # best[mask][j]: (time she leaves spot j, having picked the spots of mask, j last; the order; the drops' worth)
    best: list[dict[int, tuple[float, tuple[int, ...], float]]] = [dict() for _ in range(1 << n)]
    for j, sp in enumerate(cands):
        start = max(t(here, places[j]), sp.lands) + sp.clear_s
        best[1 << j][j] = (start + sp.stand, (j,), sp.worth * sp.still(start))
    for mask in range(1, 1 << n):
        for j, (leave, order, gain) in list(best[mask].items()):
            for k in range(n):
                if mask & (1 << k):
                    continue
                step = t(places[j], places[k])
                if step is None:
                    continue
                sp = cands[k]
                start = max(leave + step, sp.lands) + sp.clear_s
                there = start + sp.stand
                m2 = mask | (1 << k)
                got = best[m2].get(k)
                if got is None or there < got[0]:
                    best[m2][k] = (there, order + (k,), gain + sp.worth * sp.still(start))
    chosen, value, extra, key = None, 0.0, 0.0, None
    for mask in range(1, 1 << n):
        each = [cands[k] for k in range(n) if mask & (1 << k)]
        allowed, clear = _cap(sum(sp.allow for sp in each), most), sum(sp.clear_s for sp in each)
        for j, (leave, order, gain) in best[mask].items():
            after = on(places[j])
            if after is None:
                continue
            ex = leave + after - direct
            if ex - clear > allowed + 1e-9 or (budget is not None and ex > budget + 1e-9):
                continue
            k2 = (round(gain / GAIN_TIE), -ex)
            if key is None or k2 > key:
                chosen, value, extra, key = order, gain, ex, k2
    used = set(chosen or ())
    for j, sp in enumerate(cands):
        if j not in used:
            out.out[sp.key] = "cost"
    if chosen is None:
        return out
    stops, at, clock = [], here, 0.0
    for j in chosen:
        sp = cands[j]
        start = max(clock + t(at, places[j]), sp.lands) + sp.clear_s
        stops.append(Stop(sp, start, start + sp.stand))
        at, clock = places[j], start + sp.stand
    out.stops, out.value, out.extra = stops, value, extra
    return out


def _cap(allowed: float, most: float | None) -> float:
    return allowed if most is None else min(allowed, most)


def again(it: Itinerary, spots: list[Spot], here: tuple[int, float], dest: tuple[int, float] | None,
          arrive: Arrive, most: float | None = MOST_S) -> Itinerary | None:
    """Itinerary `it`'s order of spots as they are now (`spots`, by key; None if one of them is gone from it or no longer
    has a way, or it now costs her way more than its allowances, `most` at most, and KEEP_S), from `here` on to `dest`:
    its times, extra and value again."""
    by = {sp.key: sp for sp in spots}
    if any(k not in by for k in it.keys()):
        return None
    stops, at, clock = [], here, 0.0
    gain = 0.0
    for k in it.keys():
        sp = by[k]
        step = 0.0 if at == (sp.platform, sp.x) else arrive(*at, sp.platform, sp.x)
        if step is None:
            return None
        start = max(clock + step, sp.lands) + sp.clear_s
        stops.append(Stop(sp, start, start + sp.stand))
        gain += sp.worth * sp.still(start)
        at, clock = (sp.platform, sp.x), start + sp.stand
    direct = 0.0 if dest is None else (arrive(*here, *dest) or 0.0) if here != dest else 0.0
    after = 0.0 if dest is None else (0.0 if at == dest else arrive(*at, *dest))
    if after is None:
        return None
    kept = Itinerary(stops, clock + after - direct, gain, direct, it.budget)
    return None if kept.extra - kept.clear > _cap(kept.allowed, most) + KEEP_S else kept


class Sweeper:
    """The itinerary she's on, kept until another is clearly better (KEEP_SHARE, KEEP_S): `solve` at every look."""

    def __init__(self):
        self.current: Itinerary | None = None
        self.dest: tuple[int, float] | None = None

    def solve(self, spots: list[Spot], here: tuple[int, float], dest: tuple[int, float] | None, arrive: Arrive,
              budget: float | None = None, most: float | None = MOST_S) -> Itinerary:
        new = plan(spots, here, dest, arrive, budget, most)
        cur = self.current
        if cur is not None and cur.stops and self.dest == dest and new.keys() != cur.keys():
            kept = again(cur, spots, here, dest, arrive, most)
            if kept is not None:
                kept.budget = budget
            if kept is not None and kept.value > 0.0 and (budget is None or kept.extra <= budget + KEEP_S) \
                    and new.value <= kept.value * (1.0 + KEEP_SHARE):
                kept.out = new.out
                self.current = kept
                return kept
        self.current, self.dest = new, dest
        return new

    def drop(self) -> None:
        """No itinerary any more (she's off doing something else: the next solve starts afresh)."""
        self.current = None


# --- what a stand meets ---------------------------------------------------------------------------------------------

def bounce(x0: float, v: float, t: float, lo: float, hi: float) -> float:
    """Where something walking from x0 at v units a second along [lo, hi], turning back at its ends, is t s on."""
    if hi <= lo or v == 0.0:
        return min(max(x0, lo), hi)
    span = hi - lo
    p = (min(max(x0, lo), hi) - lo + v * t) % (2 * span)
    return lo + (p if p <= span else 2 * span - p)


def meets(monsters, lo: float, hi: float, t0: float, t1: float, reach: Callable[[object], tuple[float, float, float]],
          her_half: float, margin: float = CONTACT_MARGIN) -> object | None:
    """The first of `monsters` that comes within touching of her standing anywhere from map x `lo` to `hi` (on their
    platform) from t0 to t1 s from now, and `margin` more; None if none does. `reach(m)`: (its velocity, units a second,
    + right, or None where its heading isn't known: either way at its speed; its speed; half its body's width; its
    platform's ends, where it turns back; one whose heading isn't known walks UNSURE_S at most). Sampled every
    STEP_S."""
    for m in monsters:
        vx, speed, half, left, right = reach(m)
        near = half + her_half + margin
        steps = max(1, math.ceil((t1 - t0) / STEP_S))
        for i in range(steps + 1):
            t = t0 + (t1 - t0) * i / steps
            if vx is None:  # either way, at its speed (UNSURE_S of it at most): the nearest it can be
                a, b = max(left, m.x - speed * min(t, UNSURE_S)), min(right, m.x + speed * min(t, UNSURE_S))
                gap = max(lo - b, a - hi, 0.0)
            else:
                x = bounce(m.x, vx, t, left, right)
                gap = max(lo - x, x - hi, 0.0)
            if gap < near:
                return m
    return None
