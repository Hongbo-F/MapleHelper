"""Navigation: plan a route over the map (route.Planner), carry it out move by move (moves.Mover),
and replan from wherever she actually ends up when a move doesn't land where planned.

  look (localize.Localizer: minimap dot + screen)  ->  plan (route.Planner: platforms, ropes, jumps)
        ^                                                   |
        +----------- check where she landed  <-  do one move (moves.Mover)
"""

import time
from dataclasses import dataclass, field

from .hunt import Hunter
from .localize import Localizer
from .mapdata import MapModel
from .moves import Mover, describe
from .route import Planner


@dataclass
class Trip:
    result: str = ""
    moves: list[str] = field(default_factory=list)
    misses: int = 0       # moves that didn't land where planned


class Navigator:
    def __init__(self, hunter: Hunter, model: MapModel, log=print):
        self.h, self.model, self.log = hunter, model, log
        self.loc = Localizer(model)
        hunter.localizer = self.loc  # fighting then knows exactly how far she can step on her platform
        self.planner = Planner(model)
        self.mover = Mover(hunter, model, self.loc, log)

    def go(self, dst: int, x: float | None = None, seconds: float = 180, max_misses: int = 6) -> Trip:
        """Get onto platform `dst` (and to map x on it, if given)."""
        trip, deadline = Trip(), time.monotonic() + seconds
        while time.monotonic() < deadline:
            f = self.mover.fix()
            if stop := self.h.health_check(f.sight):
                trip.result = stop
                return trip
            here = f.where.platform
            if here is None:
                f = self.mover.settle(3)
                here = f.where.platform
                if here is None:
                    trip.result = "lost: not on any platform"
                    return trip
            if here.id == dst:
                if x is not None:
                    self.mover.walk_to(x, tolerance=20)
                trip.result = "arrived"
                return trip
            path = self.planner.plan(here.id, f.where.x, dst)
            if not path:
                trip.result = f"no route from platform {here.id} to {dst}"
                return trip
            move = path[0]
            self.log(f"  {describe(here)} at x {f.where.x:.0f} ({f.how}): {move.kind} {move.direction} -> "
                     f"platform {move.dst}  [{len(path)} moves to go]")
            after = self.mover.do(move)
            landed = after.where.platform
            trip.moves.append(f"{move.kind} {move.src}->{move.dst}: landed on {landed.id if landed else None}")
            if landed is None or landed.id != move.dst:
                trip.misses += 1
                self.log(f"    landed on {describe(landed)} instead; replanning")
                if trip.misses >= max_misses:
                    trip.result = "too many misses"
                    return trip
        trip.result = "timeout"
        return trip
