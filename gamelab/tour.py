"""A hunting tour of a map: sweep the platforms monsters spawn on (nearest next, varied a little, the
order then tidied to cut travel), shoot each one
clear, collect the drops, rest on the chair at a safe platform when HP runs low, and finish standing
by on a safe platform.

A platform is safe when no monster can touch her there (MapModel.safe): none spawn on it (monsters
stay on the platform they spawn on), and none walk a platform just underneath. Her
stars reach other platforms (shots.py), so each hunting platform is shot from a safe platform when
one has enough of it in reach (e.g. platform 17 from platform 14, 60 units above it); only then does
she go down to pick up what dropped, and back up to keep shooting. Otherwise she hunts on the
platform itself. On the way there, she starts shooting as soon as a target is in reach.
"""

import json
import random
import time
from dataclasses import dataclass, field

import numpy as np

from . import shots
from .hunt import Hunter, Report, Sight
from .mapdata import MONSTER_REACH
from .nav import Navigator
from .paths import LAB
from .route import WALK_SPEED
from .sprites import kind_of

MIN_COVERAGE = 0.3   # shoot from another platform only if this much of the hunting platform is in reach
NEAR_BEST = 0.7      # spots reaching at least this share of the best one's count as good: the nearest wins
LOOT_AGE = 90        # s: drops older than this are left (they vanish, and others may take them)
GROUNDS = LAB / "grounds.json"  # per map and hunting platform: visits, and the EXP they gave
WORTH = 0.25         # monsters giving under this share of the map's best EXP aren't hunted: snails, 2 vs 15
SEEN_SHARE = 0.3     # share of a platform's spawn EXP a visit gives (platform 36: 13-52 of 100), the prior
PRIOR = 2            # visits the spawn data counts for, against what visits really gave
MIN_VALUE = 0.15     # grounds expected to give under this share of the best one's EXP per visit are skipped


@dataclass
class TourReport:
    visited: list[int] = field(default_factory=list)
    kills: dict[str, int] = field(default_factory=dict)
    throws: int = 0
    exp_gains: int = 0    # times her EXP went up while hunting at the stops (kills of hers)
    on_the_way: int = 0   # monsters shot dead while travelling
    rests: int = 0
    loot_trips: int = 0
    stop: str = ""

    def __str__(self):
        kills = ", ".join(f"{n} {k}" for k, n in self.kills.items()) or "none"
        return (f"visited platforms {self.visited}; kills: {kills} (EXP gains: {self.exp_gains}), "
                f"plus {self.on_the_way} on the way; throws: {self.throws}; "
                f"loot trips: {self.loot_trips}; rests: {self.rests}; ended: {self.stop}")


class Tour:
    def __init__(self, hunter: Hunter, nav: Navigator, log=print, rng: random.Random | None = None,
                 rest_below: float = 0.5, rest_to: float = 0.9):
        self.h, self.nav, self.log = hunter, nav, log
        self.rng = rng or random.Random()
        self.rest_below, self.rest_to = rest_below, rest_to
        self.report = TourReport()
        model = nav.model
        self.safe = [p.id for p in model.platforms if model.safe(p.id) and p.right - p.left >= 120]
        self.exp = {kind_of(name): e for name, e in model.mob_exp.items()}  # kind -> EXP per kill
        best = max(self.exp.values(), default=0)
        self.worth = {k for k, e in self.exp.items() if e >= WORTH * best} or None  # kinds worth hunting
        self.stats = _load_grounds(model.id)

    def ground_values(self) -> dict[int, float]:
        """Expected EXP from a visit, per platform that monsters worth hunting spawn on (all of them
        ones `hunt` can recognize: it can't see the others coming). The spawn data's guess
        (SEEN_SHARE of its spawns' EXP) counts as PRIOR visits, blended with what visits really gave.
        By the data, platforms 34 and 35 spawn slimes, but 11 visits found only snails there."""
        known = set(self.h.eyes.detector.templates)
        out = {}
        for pid, names in self.nav.model.monsters_by_platform.items():
            kinds = [kind_of(n) for n in names]
            if not all(k in known for k in kinds):
                continue
            prior = SEEN_SHARE * sum(self.exp.get(k, 10) for k in kinds if self.worth is None or k in self.worth)
            if prior <= 0:
                continue
            st = self.stats.get(str(pid), {})
            out[pid] = (PRIOR * prior + st.get("exp", 0)) / (PRIOR + st.get("visits", 0))
        return out

    def hunting_grounds(self) -> list[int]:
        """The platforms worth a visit: expected to give at least MIN_VALUE of the best one's EXP."""
        values = self.ground_values()
        best = max(values.values(), default=0)
        return [pid for pid, v in values.items() if v >= MIN_VALUE * best]

    def _record(self, pid: int, r: Report) -> None:
        """Remember what a visit to `pid` gave, so the sweep learns where the monsters really are."""
        exp = sum(self.exp.get(k, 0) * n for k, n in r.kills.items())
        worth = [self.exp[k] for k in (self.worth or ()) if k in self.exp]
        exp += max(0, r.exp_kills - sum(r.kills.values())) * (sum(worth) / len(worth) if worth else 0)
        st = self.stats.setdefault(str(pid), {"visits": 0, "exp": 0})
        st["visits"] += 1
        st["exp"] = round(st["exp"] + exp, 1)
        _save_grounds(self.nav.model.id, self.stats)

    def spot_ok(self, platform: int) -> bool:
        """Whether she can stand on `platform` to shoot from it: no monster can touch her there
        (MapModel.safe), or only ones not worth hunting can (the snails on 3, 20 and 22: those she
        shoots when they come close)."""
        m = self.nav.model
        if m.safe(platform):
            return True
        if self.worth is None:
            return False
        p = m.platform(platform)
        for q, names in m.monsters_by_platform.items():
            other = m.platform(q)
            lo, hi = max(p.left, other.left), min(p.right, other.right)
            near = q == platform or (lo < hi and 0 < other.y_at((lo + hi) / 2) - p.y_at((lo + hi) / 2) <= MONSTER_REACH)
            if near and any(kind_of(n) in self.worth for n in names):
                return False
        return True

    def spot(self, pid: int, on_it: bool = False) -> shots.Spot:
        """Where to shoot platform `pid` from: a spot on another platform she can stand on (spot_ok)
        with enough of `pid` in reach; of those about as good as the best, the nearest to her. Coming
        from below she shoots up from a platform underneath, from above down: it used to be the one
        above every time. Else (or if `on_it`) the middle of the platform itself."""
        f = self.nav.mover.fix()
        here = f.where.platform
        good = [] if on_it else [sp for sp in shots.spots(self.nav.model, pid) if sp.platform != pid
                                 and sp.coverage >= MIN_COVERAGE and self.spot_ok(sp.platform)]
        if good:
            best = max(sp.coverage for sp in good)

            def cost(sp: shots.Spot) -> float:
                if here is None or sp.platform == here.id:
                    return abs(sp.x - f.where.x) / WALK_SPEED
                t = self.nav.planner.travel(here.id, f.where.x, sp.platform)
                return 1e6 if t is None else t

            sp = min((sp for sp in good if sp.coverage >= NEAR_BEST * best), key=cost)
            if cost(sp) < 1e6:
                return shots.Spot(sp.platform, sp.x, sp.coverage, True)
        p = self.nav.model.platform(pid)
        return shots.Spot(pid, (p.left + p.right) / 2, 1.0, False)

    def spot_for(self, pid: int, left: list, current: shots.Spot) -> shots.Spot | None:
        """A spot off `pid`, on a platform she can stand on, that reaches more of the monsters `left`
        on it than where she is, so she keeps shooting from safety instead of going down into the
        pile. The one reaching the most, then the nearest."""
        m = self.nav.model
        f = self.nav.mover.fix()

        def reached(x: float, y: float) -> int:
            return sum(shots.can_shoot(t.x - x, t.y - y) for t in left)

        cur = m.platform(current.platform)
        now = reached(current.x, cur.y_at(current.x)) if current.platform != pid else 0
        best = None
        for p in m.platforms:
            if p.id == pid or not self.spot_ok(p.id):
                continue
            for x in np.arange(p.left + shots.EDGE_MARGIN, p.right - shots.EDGE_MARGIN + 0.1, 10):
                n = reached(float(x), p.y_at(float(x)))
                if n <= now or (best is not None and n < best[0]):
                    continue
                if f.where.platform is None:
                    continue
                t = (abs(x - f.where.x) / WALK_SPEED if p.id == f.where.platform.id
                     else self.nav.planner.travel(f.where.platform.id, f.where.x, p.id))
                if t is not None and (best is None or (n, -t) > (best[0], -best[1])):
                    best = (n, t, p.id, float(x))
        if best is None:
            return None
        q, p = m.platform(pid), m.platform(best[2])
        return shots.Spot(best[2], best[3], round(shots.coverage(q, best[3], p.y_at(best[3])), 3), True)

    def nearest_safe(self) -> int:
        f = self.nav.mover.settle(3)  # she may still be falling (knocked off a platform)
        here = f.where.platform
        if here is not None and here.id in self.safe:
            return here.id
        best = None
        for pid in self.safe:
            path = self.nav.planner.plan(here.id, f.where.x, pid) if here else None
            if path is not None:
                cost = sum(m.seconds for m in path) + sum(abs(a.land_x - b.x) / WALK_SPEED
                                                          for a, b in zip(path, path[1:]))
                if best is None or cost < best[0]:
                    best = (cost, pid)
        if best is None:
            raise RuntimeError("no safe platform reachable")
        return best[1]

    def rest(self, max_seconds: float = 120) -> None:
        """Go to the nearest safe platform, sit on the chair until HP is back, and stand up."""
        pid = self.nearest_safe()
        self.log(f"resting on platform {pid}")
        self.nav.hunt_on_way = False  # low on HP: straight there
        try:
            trip = self.nav.go(pid)
        finally:
            self.nav.hunt_on_way = True
        if trip.result != "arrived":
            self.log(f"  couldn't reach a safe platform ({trip.result}); resting here")
        self.report.rests += 1
        k = self.h.keys
        if k.chair is not None:
            self.h._do([[0, k.chair.name, "down"], [40, k.chair.name, "up"]], "chair")
        end = time.monotonic() + max_seconds
        while time.monotonic() < end:
            time.sleep(5)
            hp = self.h.look(mobs=False).hp
            self.log(f"  HP {hp:.0%}")
            if hp >= self.rest_to:
                break
        arrow = k.right.name  # tapping an arrow stands her up
        self.h._do([[0, arrow, "down"], [50, arrow, "up"]], "stand")
        self.h.facing = "right"

    def hunt(self, pid: int, kinds=None, kills: int = 5, seconds: float = 90) -> Report:
        """Shoot platform `pid` clear like a player keeping the high (or low) ground: from the spot
        chosen, every monster in reach is killed before anything else (the kill count only decides
        whether to go after more); monsters left out of reach are shot from another spot that
        reaches them, if there is one. Only then are the drops collected; she goes onto the platform
        itself only when no spot reaches what's left. Killing some, then walking down into the rest
        to pick up the drops, gave up the safe spot."""
        end = time.monotonic() + seconds
        total = Report()
        spot = self.spot(pid)
        arrived = False
        for _ in range(5):
            how = "from platform" if spot.platform != pid else "on the platform"
            self.log(f"  shooting platform {pid} {how} {spot.platform} at x {spot.x:.0f} "
                     f"({spot.coverage:.0%} in reach)")

            def in_reach(s: Sight) -> bool:
                # Only where she can stand (or the spot itself): on the way to platform 2 her route
                # crosses platform 4, and stopping there put her in its crowd for the whole stop.
                here = s.platform
                if here is None or (here.id != spot.platform and not self.spot_ok(here.id)):
                    return False
                return any(t.platform == pid and (kinds is None or t.kind in kinds)
                           and shots.can_shoot(t.x - s.where.x, t.y - s.where.y) for t in s.targets)

            trip = self.nav.go(spot.platform, x=spot.x, until=in_reach, seconds=max(20, end - time.monotonic()))
            self.report.on_the_way += trip.kills
            if trip.result not in ("arrived", "in position"):
                total.stop = f"didn't get there: {trip.result}"
                break
            arrived = True
            if trip.result == "in position":
                self.log("  targets in reach on the way: shooting from here")
            self.h.report = Report()
            # Waiting for respawns looked idle and slowed the sweep: move on soon when nothing's there.
            r = self.h.clear(kinds, ground={pid}, max_kills=max(1, kills - sum(total.kills.values())),
                             seconds=max(5, end - time.monotonic()), patience=2.0 if spot.safe else 1.0,
                             finish=True, far=True)
            for kind, n in r.kills.items():
                total.kills[kind] = total.kills.get(kind, 0) + n
            total.throws += r.throws
            total.loot_runs += r.loot_runs
            total.potions += r.potions
            total.exp_kills += r.exp_kills
            total.lost += r.lost
            total.stop = r.stop
            if r.stop.startswith("HP low") or time.monotonic() >= end:
                break
            if r.stop.startswith("left platform"):
                continue  # knocked off: go back to the spot
            s = self.h.look(wide=True)
            left = [t for t in s.targets if t.platform == pid and (kinds is None or t.kind in kinds)]
            enough = sum(total.kills.values()) >= kills
            if left and not enough:
                nxt = self.spot_for(pid, left, spot)
                if nxt is not None:
                    self.log(f"  {len(left)} left on platform {pid}, out of reach here: over to platform "
                             f"{nxt.platform} at x {nxt.x:.0f}")
                    spot = nxt
                    continue
            trips = self.report.loot_trips
            self.collect_loot()
            if not left:
                self.log(f"  nothing left on platform {pid}: moving on")
                break
            if enough:
                break
            if not r.throws and self.report.loot_trips == trips:
                # Nothing thrown and she hasn't moved: going back would only repeat that. In the basin
                # a whole-screen look saw pigs that the fight's own looks never did, 4 times running.
                self.log(f"  {len(left)} seen on platform {pid} but none came into reach: moving on")
                break
            if spot.platform != pid:
                self.log(f"  {len(left)} left on platform {pid} that no spot reaches: onto the platform")
                spot = self.spot(pid, on_it=True)
            else:
                self.log(f"  {len(left)} still on platform {pid}: back at it")
        self.collect_loot()
        if arrived:  # what the visit gave, so later sweeps know where the monsters really are
            self._record(pid, total)
        return total

    def collect_loot(self) -> None:
        """Go to the drops left on other platforms and pick them up: her kills' (far kills' can lie across
        the map) and any seen on the ground. A whole-screen look first, and again after each trip, notes
        the drops in view and forgets remembered ones seen gone (Hunter.remember_loot). The nearest by
        route goes first. With her inventory full, only to mesos seen."""
        self._look_for_loot()
        for d in self.h.pending_loot:  # a kill placed on no platform: its drops fell onto the one below
            if d.platform is None and (p := self.nav.model.platform_under(d.x, d.y, reach=120)) is not None:
                d.platform, d.y = p.id, p.y_at(d.x)
        while self.h.pending_loot:
            fresh = [d for d in self.h.pending_loot if time.monotonic() - d.since < LOOT_AGE and d.platform is not None
                     and (d.mesos or not self.h.inventory_full)]
            self.h.pending_loot = []
            if not fresh:
                return
            f = self.nav.mover.fix()
            fresh.sort(key=lambda d: self._travel_to(f.where, d))
            d, rest = fresh[0], fresh[1:]
            self.log(f"  collecting drops on platform {d.platform} at x {d.x:.0f}")
            self.report.loot_trips += 1
            trip = self.nav.go(d.platform, x=d.x, seconds=60)
            self.report.on_the_way += trip.kills
            self.h.pending_loot = rest
            if trip.result != "arrived":
                self.log(f"    couldn't get there ({trip.result})")
                continue
            if not self.h.loot(d):  # guarded: shoot the guard, then try again
                self.h.clear(ground={d.platform}, max_kills=2, seconds=20)
                self.h.loot_pending()
            self._look_for_loot()

    def _look_for_loot(self) -> None:
        """A look at every platform in view for drops (about 0.2 s), noted for collect_loot."""
        self.h.remember_loot(self.h.look(mobs=False, wide=True, loot=True))

    def _travel_to(self, where, d) -> float:
        """Seconds to drops `d` from `where`, by route (straight-line walking if there's no route known)."""
        here = where.platform
        if here is not None and here.id != d.platform:
            t = self.nav.planner.travel(here.id, where.x, d.platform)
            if t is not None:
                return t + abs(d.x - where.x) / WALK_SPEED * 0.2
        return (abs(d.x - where.x) + abs(d.y - where.y)) / WALK_SPEED

    def sweep(self, grounds: list[int], stops: int, here: tuple[int, float]) -> tuple[list[int], float]:
        """An order to visit `stops` of the hunting grounds that keeps travelling short, like a player
        sweeping the map: nearest next (by route time from where she'll be, judged with a random
        +-25% so it isn't the same sweep every time), then improved by swapping stretches of it
        (2-opt) on the real route times. Returns the order and its travel time (s, by the model)."""
        spots = {pid: self._spot_at(pid) for pid in grounds}
        cache: dict[tuple, float] = {}

        def travel(a: tuple[int, float], pid: int) -> float:
            key = (a[0], round(a[1]), pid)
            if key not in cache:
                t = self.nav.planner.travel(a[0], a[1], spots[pid][0])
                cache[key] = 1e6 if t is None else t
            return cache[key]

        order, at, left = [], here, [g for g in grounds if travel(here, g) < 1e6]
        while left and len(order) < stops:
            nxt = min(left, key=lambda g: travel(at, g) * self.rng.uniform(0.75, 1.25))
            order.append(nxt)
            left.remove(nxt)
            at = spots[nxt]

        def total(seq: list[int]) -> float:
            t, a = 0.0, here
            for g in seq:
                t += travel(a, g)
                a = spots[g]
            return t

        better = True
        while better:
            better = False
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    trial = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    if total(trial) < total(order) - 0.5:
                        order, better = trial, True
        return order, total(order)

    def _spot_at(self, pid: int) -> tuple[int, float]:
        """Where she'll stand to hunt platform `pid`: the best spot she can stand on, else its middle."""
        for sp in shots.spots(self.nav.model, pid):
            if sp.platform != pid and sp.coverage >= MIN_COVERAGE and self.spot_ok(sp.platform):
                return sp.platform, sp.x
        p = self.nav.model.platform(pid)
        return pid, (p.left + p.right) / 2

    def run(self, stops: int | None = None, kinds=None, kills_per_stop: int = 5,
            seconds_per_stop: float = 90) -> TourReport:
        """Sweep the hunting grounds (all of them worth a visit, or the `stops` nearest), hunting the
        monsters worth it (`kinds`, by default all but those giving little EXP, like snails: they're
        only shot when in her way)."""
        grounds = self.hunting_grounds()
        kinds = self.worth if kinds is None else kinds
        self.nav.kinds = kinds
        f = self.nav.mover.fix()
        here = (f.where.platform.id, f.where.x) if f.where.platform else (grounds[0], 0.0)
        n = len(grounds) if stops is None else min(stops, len(grounds))
        order, travel = self.sweep(grounds, n, here)
        blind = sorted(self._order_cost(self.rng.sample(grounds, n), here)[1] for _ in range(9))[4]
        self.log(f"tour: sweep {order}, about {travel:.0f} s of travel ({n} hunting grounds picked at random: "
                 f"{blind:.0f} s); safe platforms: {self.safe}")
        left = list(order)
        while left:
            pid = left.pop(0)
            if self.h.look(mobs=False).hp < self.rest_below:
                self.rest()
            self.log(f"heading to platform {pid}")
            result = self.hunt(pid, kinds, kills_per_stop, seconds_per_stop)
            self.report.visited.append(pid)
            for kind, n in result.kills.items():
                self.report.kills[kind] = self.report.kills.get(kind, 0) + n
            self.report.throws += result.throws
            self.report.exp_gains += result.exp_kills
            self.log(f"  platform {pid}: {result}")
            if result.stop.startswith("HP low"):
                self.rest()
            # Far kills' drops can take her across the map: the rest of the sweep is planned again from
            # wherever she ended up.
            f = self.nav.mover.fix()
            if left and f.where is not None and f.where.platform is not None:
                left = self.sweep(left, len(left), (f.where.platform.id, f.where.x))[0]
        self.rest_and_stand_by()
        return self.report

    def _order_cost(self, seq: list[int], here: tuple[int, float]) -> tuple[list[int], float]:
        t, a = 0.0, here
        for g in seq:
            spot = self._spot_at(g)
            t += self.nav.planner.travel(a[0], a[1], spot[0]) or 0.0
            a = spot
        return seq, t

    def rest_and_stand_by(self) -> None:
        if self.h.look(mobs=False).hp < self.rest_to:
            self.rest()
            k = self.h.keys
            if k.chair is not None:  # stay seated while standing by
                self.h._do([[0, k.chair.name, "down"], [40, k.chair.name, "up"]], "chair")
        else:
            pid = self.nearest_safe()
            self.nav.go(pid)
        self.report.stop = "standing by on a safe platform"


def _load_grounds(map_id: int) -> dict:
    try:
        return json.loads(GROUNDS.read_text(encoding="utf-8")).get(str(map_id), {})
    except (OSError, ValueError):
        return {}


def _save_grounds(map_id: int, stats: dict) -> None:
    try:
        everything = json.loads(GROUNDS.read_text(encoding="utf-8")) if GROUNDS.exists() else {}
    except (OSError, ValueError):
        everything = {}
    everything[str(map_id)] = stats
    try:
        GROUNDS.parent.mkdir(parents=True, exist_ok=True)
        GROUNDS.write_text(json.dumps(everything, indent=1), encoding="utf-8")
    except OSError:
        pass
