"""Where to stand to shoot a platform's monsters: the spots on other platforms that have the most of it
in reach, and of those, one she can stand on safely and get to quickly. In reach counts only what terrain
doesn't stop a star at (reach.wall_in_way): from platform 4, nothing on 2 (a wall of her foothold group),
and nothing down close past her platform's end (her own floor, by the v83 line, reach.LINE). So to shoot
down, a place near her platform's end has the most in reach: 17 is shot from 14 at x 756, where 716 had as
much by the level line.

A spot is good to stand on (`spot_ok`) when no monster can touch her there (MapModel.safe), or only
ones not worth hunting can (the snails on 3, 20 and 22: those she shoots when they come close). Of the
spots about as good as the best, she takes the nearest: coming up from the basin she shoots 21, 18 and
12 from the platforms below them, and from above she shoots down (it used to be the one above every
time).

`spots` lists the places to shoot a platform from; `spot` picks one for her where she is, `spot_for`
one that reaches more of the monsters still there, and `spot_at` one for planning a sweep.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from ..localization.mapdata import MapModel, Platform, runs_under
from ..localization.monsters import Monster
from ..localization.pose import Where
from ..motion.physics import WALK_SPEED
from ..perception.kinds import kind_of
from . import reach
from .reach import EDGE_MARGIN, SHOT_DOWN, SHOT_RANGE, SHOT_UP, body_of, can_hit, clear_span, max_dx, shot_dx

MIN_COVERAGE = 0.3   # shoot from another platform only if this much of the hunting platform is in reach
NEAR_BEST = 0.7      # spots reaching at least this share of the best one's count as good: the nearest wins
X_STEP = 10          # map units between the places along a platform tried as spots
UNREACHABLE = 1e6    # s: what a spot with no route to it costs

Travel = Callable[[int, float, int], float | None]  # seconds from (platform, x) to a platform, None: no route


@dataclass(frozen=True)
class Spot:
    """Where to stand to shoot at a platform: on `platform` at map x `x`."""
    platform: int
    x: float
    coverage: float   # fraction of the target platform's length in reach from there
    safe: bool        # no monster can reach her on that platform (MapModel.safe)


def coverage(target: Platform, x: float, y: float, here: Platform | None = None, kind: str | None = None) -> float:
    """Fraction of `target` in reach of a star thrown from (x, y), either way, at a monster of `kind` (by
    its body, reach.shot_dx; a pig by default): on platform `here`, only as far as the walls of its
    foothold group let a star (reach.clear_span), and by the v83 line only where no floor stops it either
    (reach.clear_length)."""
    dy = target.y_at(min(max(x, target.left), target.right)) - y
    if not -SHOT_UP <= dy <= SHOT_DOWN:
        return 0.0
    covered, near, far = 0.0, shot_dx(dy, kind), min(SHOT_RANGE, max_dx(body_of(kind)[1]))
    left, right = clear_span(here, x, y)
    for lo, hi in ((max(x - far, left), x - near), (x + near, min(x + far, right))):
        lo, hi = max(lo, target.left), min(hi, target.right)
        if hi <= lo:
            continue
        if reach.LINE == "v83" and here is not None and reach.WALLS_BLOCK:
            covered += reach.clear_length(here, x, y, target, lo, hi, body_of(kind))
        else:
            covered += hi - lo
    return covered / max(1.0, target.right - target.left)


def _stand_xs(lo: float, hi: float) -> list[float]:
    """The map x from `lo` to `hi` to try standing at: every X_STEP units (the 0.1 keeps `hi` in)."""
    return [float(x) for x in np.arange(lo, hi + 0.1, X_STEP)]


def spots(model: MapModel, target: int) -> list[Spot]:
    """Places to shoot platform `target` from, best first: safe ones (no spawns where she stands),
    then by how much of the target platform is in reach. Worked out once per map and platform (`_KEPT`), and
    again if what reach's answers that coverage goes by are switched (`_switches`: a probe switches
    reach.WALLS_BLOCK for its run, and the probes' tests put the other answers in their worlds)."""
    kept = _KEPT.get(id(model))
    if kept is None or kept[0] is not model:
        if len(_KEPT) >= MODELS_KEPT:
            _KEPT.clear()
        kept = _KEPT[id(model)] = (model, {})
    key = (target, *_switches())
    found = kept[1].get(key)
    if found is None:
        found = kept[1][key] = tuple(_spots(model, target))
    return list(found)


# spots' answers by id(model), with the model (kept, so its id can't go to another one while it's here) and
# the answers by target platform and reach's switches (_switches). They depend on the map, on what stops a star
# (her own foothold group's walls, unless a probe switches that off; and by which line, reach.LINE) and on what
# the attack picks (the search's window, the claw, the old ramp: coverage's shot_dx), and take 3.6 ms a platform
# to work out: the hunt loop asked for every hunting ground's after most looks (its rate rule, and the 'hunt'
# record for the mini-map), 30-60 ms between a look and her next keys.
_KEPT: dict[int, tuple[MapModel, dict[tuple, tuple[Spot, ...]]]] = {}
MODELS_KEPT = 4  # maps whose spots are kept (a run has one; tests make many)


def _switches() -> tuple:
    """What of reach coverage goes by that a run or a test may switch: the wall rule and its line, the
    search window's edges, the claw's model and the old ramp."""
    return (reach.WALLS_BLOCK, reach.LINE, reach.SEARCH_TOP0, reach.SEARCH_FLOOR0, reach.CLAW, reach.RAMP_BELOW)


def _spots(model: MapModel, target: int) -> list[Spot]:
    ground = model.platform(target)
    out = []
    for p in model.platforms:
        lo, hi = p.left + EDGE_MARGIN, p.right - EDGE_MARGIN
        if hi < lo:
            continue
        best = None  # (coverage, x): the first x of the best coverage on p
        for x in _stand_xs(lo, hi):
            c = coverage(ground, x, p.y_at(x), p)
            if c > 0 and (best is None or c > best[0] + 1e-9):
                best = (c, x)
        if best:
            out.append(Spot(p.id, best[1], round(best[0], 3), model.safe(p.id)))
    return sorted(out, key=lambda s: (not s.safe, -s.coverage))


def spot_ok(model: MapModel, platform: int, worth: set[str] | None) -> bool:
    """Whether she can stand on `platform` to shoot from it: no monster can touch her there, or only
    ones not worth hunting can (`worth`: the kinds that are; None: all are)."""
    if model.safe(platform):
        return True
    if worth is None:
        return False
    p = model.platform(platform)
    for q, names in model.monsters_by_platform.items():
        other = model.platform(q)
        if (q == platform or runs_under(other, p)) and any(kind_of(n) in worth for n in names):
            return False
    return True


def _standable(model: MapModel, pid: int, worth: set[str] | None) -> Iterator[Spot]:
    """spots(model, pid) on other platforms that reach at least MIN_COVERAGE of `pid` and that she can
    stand on (spot_ok), best first."""
    return (sp for sp in spots(model, pid)
            if sp.platform != pid and sp.coverage >= MIN_COVERAGE and spot_ok(model, sp.platform, worth))


def _middle(model: MapModel, pid: int) -> float:
    """The map x of the middle of platform `pid`."""
    p = model.platform(pid)
    return (p.left + p.right) / 2


def spot(model: MapModel, pid: int, here: Where, travel: Travel, worth: set[str] | None,
         on_it: bool = False) -> Spot:
    """Where to shoot platform `pid` from, for her at `here`: a spot on another platform she can stand
    on with enough of `pid` in reach; of those about as good as the best, the nearest to her. Else (or
    if `on_it`) the middle of the platform itself."""
    at = here.platform
    good = [] if on_it else list(_standable(model, pid, worth))
    if good:
        best = max(sp.coverage for sp in good)

        def cost(sp: Spot) -> float:
            """Seconds to get to `sp` from `here` (UNREACHABLE: no route)."""
            if at is None or sp.platform == at.id:
                return abs(sp.x - here.x) / WALK_SPEED
            t = travel(at.id, here.x, sp.platform)
            return UNREACHABLE if t is None else t

        sp = min((sp for sp in good if sp.coverage >= NEAR_BEST * best), key=cost)
        if cost(sp) < UNREACHABLE:
            return sp
    return Spot(pid, _middle(model, pid), 1.0, model.safe(pid))


def spot_for(model: MapModel, pid: int, left: list[Monster], current: Spot, here: Where, travel: Travel,
             worth: set[str] | None) -> Spot | None:
    """A spot off `pid`, on a platform she can stand on, that reaches more of the monsters `left` on it
    than where she is (`current`), so she keeps shooting from safety instead of going down into the
    pile. The one reaching the most, then the nearest. None if there's none, or she isn't on a
    platform (no telling how far a spot is)."""

    on = {t.platform: model.platform(t.platform) for t in left if t.platform is not None}

    def reached(p: Platform, x: float, y: float) -> int:
        return sum(can_hit(p, x, y, t, target=on.get(t.platform), others=left) for t in left)

    cur = model.platform(current.platform)
    now = reached(cur, current.x, cur.y_at(current.x)) if current.platform != pid else 0
    best = None  # (monsters reached, seconds to get there, platform, x)
    for p in model.platforms:
        if p.id == pid or not spot_ok(model, p.id, worth):
            continue
        for x in _stand_xs(p.left + EDGE_MARGIN, p.right - EDGE_MARGIN):
            n = reached(p, x, p.y_at(x))
            if n <= now or (best is not None and n < best[0]):
                continue
            if here.platform is None:
                continue
            t = (abs(x - here.x) / WALK_SPEED if p.id == here.platform.id
                 else travel(here.platform.id, here.x, p.id))
            if t is not None and (best is None or (n, -t) > (best[0], -best[1])):
                best = (n, t, p.id, x)
    if best is None:
        return None
    _, _, stand, x = best
    ground, p = model.platform(pid), model.platform(stand)
    return Spot(stand, x, round(coverage(ground, x, p.y_at(x), p), 3), model.safe(stand))


def spot_at(model: MapModel, pid: int, worth: set[str] | None) -> tuple[int, float]:
    """Where she'll stand to hunt platform `pid` (for planning a sweep): the best spot she can stand on,
    else its middle."""
    sp = next(_standable(model, pid, worth), None)
    if sp is not None:
        return sp.platform, sp.x
    return pid, _middle(model, pid)
