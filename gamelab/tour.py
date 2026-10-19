"""A hunting tour of a map: visit the platforms monsters spawn on in random order, clear each one
(looting as she goes), rest on the chair at a safe platform when HP runs low, and finish standing by
on a safe platform.

A platform is safe when no monster spawns on it: monsters stay on the platform they spawn on.
"""

import random
import time
from dataclasses import dataclass, field

from .hunt import Hunter
from .nav import Navigator
from .route import WALK_SPEED
from .sprites import kind_of


@dataclass
class TourReport:
    visited: list[int] = field(default_factory=list)
    kills: dict[str, int] = field(default_factory=dict)
    rests: int = 0
    stop: str = ""

    def __str__(self):
        kills = ", ".join(f"{n} {k}" for k, n in self.kills.items()) or "none"
        return f"visited platforms {self.visited}; kills: {kills}; rests: {self.rests}; ended: {self.stop}"


class Tour:
    def __init__(self, hunter: Hunter, nav: Navigator, log=print, rng: random.Random | None = None,
                 rest_below: float = 0.5, rest_to: float = 0.9):
        self.h, self.nav, self.log = hunter, nav, log
        self.rng = rng or random.Random()
        self.rest_below, self.rest_to = rest_below, rest_to
        self.report = TourReport()
        model = nav.model
        self.safe = [p.id for p in model.platforms if model.spawn_platforms.get(p.id, 0) == 0
                     and p.right - p.left >= 120]

    def hunting_grounds(self, min_spawns: int = 2) -> list[int]:
        """Platforms with at least `min_spawns` spawn points, all of monsters `hunt` can recognize (it
        can't see the others coming, so it would let them walk up and hit her)."""
        known = set(self.h.eyes.detector.templates)
        return [pid for pid, names in self.nav.model.monsters_by_platform.items()
                if len(names) >= min_spawns and all(kind_of(n) in known for n in names)]

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
            hp = self.h.look().hp
            self.log(f"  HP {hp:.0%}")
            if hp >= self.rest_to:
                break
        arrow = k.right.name  # tapping an arrow stands her up
        self.h._do([[0, arrow, "down"], [50, arrow, "up"]], "stand")

    def run(self, stops: int = 3, kinds=None, kills_per_stop: int = 5, seconds_per_stop: float = 90) -> TourReport:
        grounds = self.hunting_grounds()
        order = self.rng.sample(grounds, min(stops, len(grounds)))
        self.log(f"tour: {order} (safe platforms: {self.safe})")
        for pid in order:
            if self.h.look().hp < self.rest_below:
                self.rest()
            self.log(f"heading to platform {pid}")
            trip = self.nav.go(pid)
            if trip.result != "arrived":
                self.log(f"  didn't get there: {trip.result}")
                continue
            self.report.visited.append(pid)
            p = self.nav.model.platform(pid)
            self.h.report = type(self.h.report)()
            result = self.h.clear_floor(kinds, max_kills=kills_per_stop, seconds=seconds_per_stop,
                                        span=(p.right - p.left) * 1.40625)
            for kind, n in result.kills.items():
                self.report.kills[kind] = self.report.kills.get(kind, 0) + n
            self.log(f"  platform {pid}: {result}")
            if result.stop.startswith("HP low"):
                self.rest()
        self.rest_and_stand_by()
        return self.report

    def rest_and_stand_by(self) -> None:
        if self.h.look().hp < self.rest_to:
            self.rest()
            k = self.h.keys
            if k.chair is not None:  # stay seated while standing by
                self.h._do([[0, k.chair.name, "down"], [40, k.chair.name, "up"]], "chair")
        else:
            pid = self.nearest_safe()
            self.nav.go(pid)
        self.report.stop = "standing by on a safe platform"
