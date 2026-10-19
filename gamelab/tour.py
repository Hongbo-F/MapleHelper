"""A hunting tour of a map: visit the platforms monsters spawn on in random order, shoot each one
clear, collect the drops, rest on the chair at a safe platform when HP runs low, and finish standing
by on a safe platform.

A platform is safe when no monster can touch her there (MapModel.safe): none spawn on it (monsters
stay on the platform they spawn on), and none walk a platform just underneath. Her
stars reach other platforms (shots.py), so each hunting platform is shot from a safe platform when
one has enough of it in reach (e.g. platform 17 from platform 14, 60 units above it); only then does
she go down to pick up what dropped, and back up to keep shooting. Otherwise she hunts on the
platform itself. On the way there, she starts shooting as soon as a target is in reach.
"""

import random
import time
from dataclasses import dataclass, field

from . import shots
from .hunt import Hunter, Report, Sight
from .nav import Navigator
from .route import WALK_SPEED
from .sprites import kind_of

MIN_COVERAGE = 0.4   # shoot from a safe platform only if this much of the hunting platform is in reach
LOOT_AGE = 90        # s: drops older than this are left (they vanish, and others may take them)


@dataclass
class TourReport:
    visited: list[int] = field(default_factory=list)
    kills: dict[str, int] = field(default_factory=dict)
    throws: int = 0
    rests: int = 0
    loot_trips: int = 0
    stop: str = ""

    def __str__(self):
        kills = ", ".join(f"{n} {k}" for k, n in self.kills.items()) or "none"
        return (f"visited platforms {self.visited}; kills: {kills}; throws: {self.throws}; "
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

    def hunting_grounds(self, min_spawns: int = 2) -> list[int]:
        """Platforms with at least `min_spawns` spawn points, all of monsters `hunt` can recognize (it
        can't see the others coming, so it would let them walk up and hit her)."""
        known = set(self.h.eyes.detector.templates)
        return [pid for pid, names in self.nav.model.monsters_by_platform.items()
                if len(names) >= min_spawns and all(kind_of(n) in known for n in names)]

    def spot(self, pid: int, on_it: bool = False) -> shots.Spot:
        """Where to shoot platform `pid` from: the safe spot with the most of it in reach (if enough),
        else (or if `on_it`) the middle of the platform itself."""
        f = self.nav.mover.fix()
        here = f.where.platform
        for s in ([] if on_it else shots.spots(self.nav.model, pid)):
            if not s.safe or s.coverage < MIN_COVERAGE:
                break
            if here is None or s.platform == here.id or self.nav.planner.plan(here.id, f.where.x, s.platform):
                return s
        p = self.nav.model.platform(pid)
        return shots.Spot(pid, (p.left + p.right) / 2, 1.0, False)

    def nearest_safe(self) -> int:
        f = self.nav.mover.fix()
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
        trip = self.nav.go(pid)
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
        """Shoot platform `pid` clear from the best spot, collecting the drops between rounds."""
        end = time.monotonic() + seconds
        total = Report()
        on_it = False  # hunt on the platform itself (the safe spot left its monsters out of reach)
        for _ in range(4):
            spot = self.spot(pid, on_it)
            how = "from safe platform" if spot.platform != pid else "on the platform"
            self.log(f"  shooting platform {pid} {how} {spot.platform} at x {spot.x:.0f} "
                     f"({spot.coverage:.0%} in reach)")

            def in_reach(s: Sight) -> bool:
                return any(t.platform == pid and (kinds is None or t.kind in kinds)
                           and shots.can_shoot(t.x - s.where.x, t.y - s.where.y) for t in s.targets)

            trip = self.nav.go(spot.platform, x=spot.x, until=in_reach, seconds=max(20, end - time.monotonic()))
            if trip.result not in ("arrived", "in position"):
                total.stop = f"didn't get there: {trip.result}"
                break
            if trip.result == "in position":
                self.log("  targets in reach on the way: shooting from here")
            self.h.report = Report()
            # From a safe spot, waiting for monsters to respawn or walk into reach costs nothing.
            r = self.h.clear(kinds, ground={pid}, max_kills=kills - sum(total.kills.values()),
                             seconds=max(5, end - time.monotonic()), patience=10 if spot.safe else 4)
            for kind, n in r.kills.items():
                total.kills[kind] = total.kills.get(kind, 0) + n
            total.throws += r.throws
            total.loot_runs += r.loot_runs
            total.potions += r.potions
            total.stop = r.stop
            self.collect_loot()
            if r.stop.startswith("HP low") or sum(total.kills.values()) >= kills or time.monotonic() >= end:
                break
            if r.stop.startswith("left platform"):
                continue  # knocked off: go back to the spot
            if spot.safe and r.out_of_reach:
                self.log("  monsters out of reach from the safe spot: hunting on the platform itself")
                on_it = True
            elif not r.throws and not self.h.pending_loot:  # nothing there to shoot: move on
                break
        return total

    def collect_loot(self) -> None:
        """Go to drops left on other platforms (nearest first) and sweep them up."""
        while self.h.pending_loot:
            fresh = [d for d in self.h.pending_loot if time.monotonic() - d.since < LOOT_AGE and d.platform is not None]
            self.h.pending_loot = []
            if not fresh:
                return
            f = self.nav.mover.fix()
            fresh.sort(key=lambda d: abs(d.x - f.where.x) + abs(d.y - f.where.y))
            d, rest = fresh[0], fresh[1:]
            self.log(f"  collecting drops on platform {d.platform} at x {d.x:.0f}")
            self.report.loot_trips += 1
            trip = self.nav.go(d.platform, x=d.x, seconds=40)
            self.h.pending_loot = rest
            if trip.result != "arrived":
                self.log(f"    couldn't get there ({trip.result})")
                continue
            if not self.h.loot(d):  # guarded: shoot the guard, then try again
                self.h.clear(ground={d.platform}, max_kills=2, seconds=20)
                self.h.loot_pending()

    def run(self, stops: int = 3, kinds=None, kills_per_stop: int = 5, seconds_per_stop: float = 90) -> TourReport:
        grounds = self.hunting_grounds()
        order = self.rng.sample(grounds, min(stops, len(grounds)))
        self.log(f"tour: {order} (safe platforms: {self.safe})")
        for pid in order:
            if self.h.look(mobs=False).hp < self.rest_below:
                self.rest()
            self.log(f"heading to platform {pid}")
            result = self.hunt(pid, kinds, kills_per_stop, seconds_per_stop)
            self.report.visited.append(pid)
            for kind, n in result.kills.items():
                self.report.kills[kind] = self.report.kills.get(kind, 0) + n
            self.report.throws += result.throws
            self.log(f"  platform {pid}: {result}")
            if result.stop.startswith("HP low"):
                self.rest()
        self.rest_and_stand_by()
        return self.report

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
