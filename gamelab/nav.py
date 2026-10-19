"""Navigation: plan a route over the map (route.Planner), carry it out move by move (moves.Mover),
and replan from wherever she actually ends up when a move doesn't land where planned.

  look (localize.Localizer: minimap dot + screen)  ->  plan (route.Planner: platforms, ropes, jumps)
        ^                                                   |
        +----------- check where she landed  <-  do one move (moves.Mover)

She hunts on the way: before each move, any monster a star reaches from where she stands (on any
platform) is shot dead first, a few per platform, so travelling earns EXP too.

Monsters knock her back when they touch her, which can throw her off a run-up, a rope or a ledge.
Before a move that walks along a platform with monsters on it, any monster in the way is shot first
(she outranges them all). A hit during a move shows as lost HP: the trip notes it and replans from
wherever she was pushed to, without counting it as a missed move.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import shots
from .hunt import TOO_CLOSE, Hunter, Sight, pick_target
from .mapdata import MapModel
from .moves import Fix, Mover, describe
from .route import Move, Planner

GUARD_SECONDS = 15  # longest spent shooting a monster out of the way
ON_WAY_TRIES = 5    # monsters engaged per platform while passing through (respawns could keep her forever)


@dataclass
class Trip:
    result: str = ""
    moves: list[str] = field(default_factory=list)
    misses: int = 0       # moves that didn't land where planned (not counting hits)
    hits: int = 0         # moves during which a monster hit her
    kills: int = 0        # monsters shot dead on the way


class Navigator:
    def __init__(self, hunter: Hunter, model: MapModel, log=print):
        self.h, self.model, self.log = hunter, model, log
        self.loc = hunter.localizer
        self.planner = Planner(model)
        self.mover = Mover(hunter, model, self.loc, log)
        self.hunt_on_way = True  # shoot monsters in reach between moves (off while going to rest)

    def go(self, dst: int, x: float | None = None, seconds: float = 180, max_misses: int = 6,
           until: Callable[[Sight], bool] | None = None) -> Trip:
        """Get onto platform `dst` (and to map x on it, if given). `until` is asked after every move,
        with a look that includes monsters: True ends the trip early ('in position')."""
        trip, deadline = Trip(), time.monotonic() + seconds
        fights = 0  # in a row, without a move between: after a few she moves on regardless
        tries: dict[int, int] = {}  # platform -> monsters engaged there on the way
        while time.monotonic() < deadline:
            f = self.mover.fix(mobs=True)
            if stop := self.h.health_check(f.sight):
                trip.result = stop
                return trip
            if until is not None and f.where.platform is not None and until(f.sight):
                trip.result = "in position"
                return trip
            here = f.where.platform
            if here is None:
                f = self.mover.settle(3)
                here = f.where.platform
                if here is None:
                    trip.result = "lost: not on any platform"
                    return trip
            if self.hunt_on_way and tries.get(here.id, 0) < ON_WAY_TRIES and (t := self.in_reach(f)):
                tries[here.id] = tries.get(here.id, 0) + 1
                self.log(f"  {t.kind} in reach on platform {t.platform}: shooting it on the way")
                if self.h.kill(t, seconds=GUARD_SECONDS) == "killed":
                    trip.kills += 1
                continue
            if here.id == dst:
                if x is not None:
                    self.mover.walk_to(x, tolerance=20, f=f)
                trip.result = "arrived"
                return trip
            path = self.planner.plan(here.id, f.where.x, dst)
            if not path:
                trip.result = f"no route from platform {here.id} to {dst}"
                return trip
            move = path[0]
            if fights < 3 and self.clear_way(f, move):
                fights += 1
                continue  # look again after the fight
            fights = 0
            self.log(f"  {describe(here)} at x {f.where.x:.0f} ({f.how}): {move.kind} {move.direction} -> "
                     f"platform {move.dst}  [{len(path)} moves to go]")
            hp = f.sight.hp
            after = self.mover.do(move, f)
            landed = after.where.platform
            hit = after.sight.hp < hp - 0.004
            trip.moves.append(f"{move.kind} {move.src}->{move.dst}: landed on {landed.id if landed else None}"
                              + (" (hit)" if hit else ""))
            if hit:
                trip.hits += 1
            if landed is None or landed.id != move.dst:
                if hit:
                    self.log(f"    hit on the way ({hp:.0%} -> {after.sight.hp:.0%}): on {describe(landed)}; replanning")
                    continue
                trip.misses += 1
                self.log(f"    landed on {describe(landed)} instead; replanning")
                if trip.misses >= max_misses:
                    trip.result = "too many misses"
                    return trip
        trip.result = "timeout"
        return trip

    def in_reach(self, f: Fix):
        """The monster to shoot from where she stands, if any is in reach (pick_target's choice)."""
        s, here = f.sight, f.where.platform
        reach = [t for t in s.targets if shots.can_shoot(t.x - f.where.x, t.y - f.where.y)]
        return pick_target(f.where.x, f.where.y, here, reach) if reach else None

    def clear_way(self, f: Fix, move: Move) -> bool:
        """Shoot a monster standing on her platform between her and where the move starts (or too
        close to her). Returns True if she fought."""
        s = f.sight
        lo, hi = sorted((f.where.x, move.x))
        here = f.where.platform
        in_way = [t for t in s.targets if t.platform == here.id
                  and lo - TOO_CLOSE <= t.x <= hi + TOO_CLOSE]
        if not in_way:
            return False
        t = min(in_way, key=lambda t: abs(t.x - f.where.x))
        self.log(f"  {t.kind} in the way at x {t.x:.0f}: shooting it first")
        self.h.kill(t, seconds=GUARD_SECONDS)
        return True
