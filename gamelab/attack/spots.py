"""Where to stand to shoot a platform's monsters: the spots on other platforms that have the most of it
in reach, and of those, one she can stand on safely and get to quickly. In reach counts only what terrain
doesn't stop a star at (reach.wall_in_way): from platform 4, nothing on 2 (a wall of her foothold group),
and nothing down close past her platform's end (her own floor, by the v83 line, reach.LINE). So to shoot
down, a place near her platform's end has the most in reach: 17 is shot from 14 at x 756, where 716 had as
much by the level line.

A spot is good to stand on (`spot_ok`) when no monster can touch her there (MapModel.safe), or only
ones not worth hunting can (the snails on 3, 20 and 22: those she shoots when they come close). With the map's
danger (attack.danger: the share of her HP touches would take there, from its spawn points' monsters), so is one
on any other platform that's better than hunting on the ground itself: its share of the ground in reach, less
DANGER_WEIGHT times its danger, over the ground's whole reach less the danger of the safest place on it. On Tree
Dungeon 23 (two Horny Mushrooms, 16% of her HP a touch) is shot from 24 at x 231 (95% of it in reach), where only
24's Green Mushroom (7.2%) touches her, and the Hornies, which never leave 23, can't. Of the spots about as
good as the best (by that score), she takes the nearest: coming up from the basin she shoots 21, 18 and 12 from
the platforms below them, and from above she shoots down (it used to be the one above every time). With none,
she hunts the ground standing on it, from its safest place (the middle where nothing counts as a danger, as on
the first map, where no kind does).

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
from .danger import Danger
from .reach import EDGE_MARGIN, SHOT_DOWN, SHOT_RANGE, SHOT_UP, body_of, can_hit, clear_span, max_dx, shot_dx
from .stance import KNOCKBACK_ROOM

MIN_COVERAGE = 0.3   # shoot from another platform only if this much of the hunting platform is in reach
NEAR_BEST = 0.7      # spots scoring at least this share of the best one's count as good: the nearest wins
X_STEP = 10          # map units between the places along a platform tried as spots
UNREACHABLE = 1e6    # s: what a spot with no route to it costs
# A spot's score: the share of the ground in reach from it less this times its danger (attack.danger: the share of her
# HP its touches would take, the spawn points' monsters at their points). A guess, not measured (nor swept in the
# simulator), which says what a ground's reach is worth against HP: at 1 a place where a Horny Mushroom's touch
# (16% of her HP) is about to come is worth 16% of a ground in reach less: on Tree Dungeon 24 at 231 (95% of 23 in
# reach, 24's Green by her: danger 0.07, score 0.88) wins over 23's own safest place (0.15 at x -147: 0.85), and 8 at
# 157 (all of 7 in reach, 0.17: 0.83) over 7's (0.25: 0.75).
DANGER_WEIGHT = 1.0
GROUND_MARGIN = EDGE_MARGIN + KNOCKBACK_ROOM  # hunting a ground standing on it, she keeps this far from its ends (as
                                              # stance does with monsters about: a touch knocks her back), and so on a
                                              # spot with monsters that count as a danger (_margin): on 24, 211 had 24's
                                              # Green knock her off its left end in the simulator

Travel = Callable[[int, float, int], float | None]  # seconds from (platform, x) to a platform, None: no route


@dataclass(frozen=True)
class Spot:
    """Where to stand to shoot at a platform: on `platform` at map x `x`."""
    platform: int
    x: float
    coverage: float   # fraction of the target platform's length in reach from there
    safe: bool        # no monster can reach her on that platform (MapModel.safe)
    danger: float = 0.0  # the danger there (attack.danger, the spawn points' monsters at their points; 0 without it)

    @property
    def score(self) -> float:
        """How good a place to shoot from: its coverage less DANGER_WEIGHT times its danger."""
        return self.coverage - DANGER_WEIGHT * self.danger


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


def spots(model: MapModel, target: int, danger: Danger | None = None) -> list[Spot]:
    """Places to shoot platform `target` from, best first: safe ones (no spawns where she stands),
    then by their score (how much of the target platform is in reach, less DANGER_WEIGHT times the danger there,
    with `danger`). Worked out once per map and platform (`_KEPT`), and again if what reach's answers that coverage
    goes by are switched (`_switches`: a probe switches reach.WALLS_BLOCK for its run, and the probes' tests put
    the other answers in their worlds)."""
    return list(_kept(model, ("spots", target, *_her(danger), *_switches()),
                      lambda: tuple(_spots(model, target, danger))))


def _kept(model: MapModel, key: tuple, make: Callable):
    """`make()`'s answer for `model` and `key`, worked out once (_KEPT)."""
    kept = _KEPT.get(id(model))
    if kept is None or kept[0] is not model:
        if len(_KEPT) >= MODELS_KEPT:
            _KEPT.clear()
        kept = _KEPT[id(model)] = (model, {})
    found = kept[1].get(key)
    if found is None:
        found = kept[1][key] = make()
    return found


def _her(danger: Danger | None) -> tuple:
    """What of `danger` the kept answers go by: her (a map's Danger is its model's and hers), or nothing without it."""
    return () if danger is None or not danger.any else (danger.her,)


# spots' answers by id(model), with the model (kept, so its id can't go to another one while it's here) and
# the answers by what they are, target platform, her (_her) and reach's switches (_switches). They depend on the
# map, on what stops a star (her own foothold group's walls, unless a probe switches that off; and by which line,
# reach.LINE), on what the attack picks (the search's window, the claw, the old ramp: coverage's shot_dx) and on her
# (the danger), and take 3.6 ms a platform to work out (with the danger, every ground of Tree Dungeon 1.1 s at the
# hunt loop's first look): the hunt loop asked for every hunting ground's after most looks (its rate rule, and the
# 'hunt' record for the mini-map), 30-60 ms between a look and her next keys.
_KEPT: dict[int, tuple[MapModel, dict[tuple, object]]] = {}
MODELS_KEPT = 4  # maps whose spots are kept (a run has one; tests make many)


def _switches() -> tuple:
    """What of reach coverage goes by that a run or a test may switch: the wall rule and its line, the
    search window's edges, the claw's model and the old ramp."""
    return (reach.WALLS_BLOCK, reach.LINE, reach.SEARCH_TOP0, reach.SEARCH_FLOOR0, reach.CLAW, reach.RAMP_BELOW)


def _spots(model: MapModel, target: int, danger: Danger | None = None) -> list[Spot]:
    grid = _grid(model, target, danger)
    out = []
    for p in model.places:
        lo, hi = p.left + EDGE_MARGIN, p.right - EDGE_MARGIN
        if hi < lo:
            continue
        best = None  # (score, x): the first x of the best score on p
        for x in _xs_within(_stand_xs(lo, hi), p, _margin(p, danger)):
            c, d = grid.get((p.id, x), (0.0, 0.0))
            if c > 0 and (best is None or c - DANGER_WEIGHT * d > best[0] + 1e-9):
                best = (c - DANGER_WEIGHT * d, x)
        if best:
            c, d = grid[(p.id, best[1])]
            out.append(Spot(p.id, best[1], round(c, 3), model.safe(p.id), round(d, 3)))
    return sorted(out, key=lambda s: (not s.safe, -s.score))


def _grid(model: MapModel, target: int, danger: Danger | None = None) -> dict[tuple[int, float], tuple[float, float]]:
    """(platform, x) -> (coverage of `target` from there, the danger there) for every place along every platform
    tried as a spot (X_STEP apart, EDGE_MARGIN from its ends) that has some of it in reach; the danger is 0 without
    `danger`. Kept (_KEPT): spot_for looks places up in it."""
    def make() -> dict:
        ground, out = model.platform(target), {}
        for p in model.places:
            lo, hi = p.left + EDGE_MARGIN, p.right - EDGE_MARGIN
            for x in _stand_xs(lo, hi) if hi >= lo else ():
                c = coverage(ground, x, p.y_at(x), p)
                if c > 0:
                    out[(p.id, x)] = (c, _danger_at(model, danger, p.id, x))
        return out
    return _kept(model, ("grid", target, *_her(danger), *_switches()), make)


def _danger_at(model: MapModel, danger: Danger | None, pid: int, x: float) -> float:
    """The danger at map x on platform `pid` as she plans (the spawn points' monsters at their points), kept for every
    target's grid (0 without `danger`)."""
    if danger is None or not danger.any:
        return 0.0
    at = _kept(model, ("danger", *_her(danger)), dict)
    if (pid, x) not in at:
        at[(pid, x)] = danger.at(pid, x)
    return at[(pid, x)]


def _margin(p: Platform, danger: Danger | None) -> float:
    """How far from platform `p`'s ends a spot on it is: GROUND_MARGIN where monsters that count as a danger spawn on it
    (a touch knocks her back), else EDGE_MARGIN (as everywhere where nothing counts: the first map)."""
    if danger is None or not danger.any or not any(q == p.id for q, _, _ in danger.points):
        return EDGE_MARGIN
    return GROUND_MARGIN


def _xs_within(xs: list[float], p: Platform, margin: float) -> list[float]:
    """Those of `xs` `margin` or more from platform `p`'s ends."""
    return [x for x in xs if p.left + margin <= x <= p.right - margin]


def on_ground(model: MapModel, pid: int, danger: Danger | None = None) -> tuple[float, float]:
    """Where she stands to hunt platform `pid` on it, and the danger there: its safest place (GROUND_MARGIN from its
    ends, the spawn points' monsters at their points: on Tree Dungeon 23's left part, 132 units from the nearer Horny
    Mushroom's point, not its middle between the two), the nearest to its middle of equally safe ones; its middle
    where nothing counts as a danger (as on the first map)."""
    mid = _middle(model, pid)
    if danger is None or not danger.any:
        return mid, 0.0

    def make() -> tuple[float, float]:
        p = model.platform(pid)
        xs = _stand_xs(p.left + GROUND_MARGIN, p.right - GROUND_MARGIN) or [mid]
        at = {x: danger.at(pid, x) for x in xs}
        x = min(xs, key=lambda x: (round(at[x], 6), abs(x - mid)))
        return x, at[x]
    return _kept(model, ("ground", pid, *_her(danger)), make)


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


def _standable(model: MapModel, pid: int, worth: set[str] | None, danger: Danger | None = None) -> Iterator[Spot]:
    """spots(model, pid) on other platforms that reach at least MIN_COVERAGE of `pid` and that she can
    stand on (spot_ok, or with `danger` better than hunting on the ground: _better), best first."""
    ground = _ground_score(model, pid, danger)
    return (sp for sp in spots(model, pid, danger)
            if sp.platform != pid and sp.coverage >= MIN_COVERAGE
            and (spot_ok(model, sp.platform, worth) or _better(sp.score, ground)))


def _ground_score(model: MapModel, pid: int, danger: Danger | None) -> float | None:
    """What hunting platform `pid` standing on it scores (all of it in reach, less DANGER_WEIGHT times the danger of its
    safest place: on_ground), or None where nothing counts as a danger (only spot_ok's spots are then taken)."""
    if danger is None or not danger.any:
        return None
    return 1.0 - DANGER_WEIGHT * on_ground(model, pid, danger)[1]


def _better(score: float, ground: float | None) -> bool:
    """Whether a spot off the ground scoring `score` beats hunting on it (`ground`: _ground_score)."""
    return ground is not None and score > ground


def _middle(model: MapModel, pid: int) -> float:
    """The map x of the middle of platform `pid`."""
    p = model.platform(pid)
    return (p.left + p.right) / 2


def spot(model: MapModel, pid: int, here: Where, travel: Travel, worth: set[str] | None,
         on_it: bool = False, danger: Danger | None = None) -> Spot:
    """Where to shoot platform `pid` from, for her at `here`: a spot on another platform she can stand
    on with enough of `pid` in reach; of those about as good as the best (by their score), the nearest to her.
    Else (or if `on_it`) the platform itself, at its safest place (on_ground: its middle without `danger`)."""
    at = here.platform
    good = [] if on_it else list(_standable(model, pid, worth, danger))
    if good:
        best = max(sp.score for sp in good)

        def cost(sp: Spot) -> float:
            """Seconds to get to `sp` from `here` (UNREACHABLE: no route)."""
            if at is None or sp.platform == at.id:
                return abs(sp.x - here.x) / WALK_SPEED
            t = travel(at.id, here.x, sp.platform)
            return UNREACHABLE if t is None else t

        sp = min((sp for sp in good if sp.score >= best - (1 - NEAR_BEST) * abs(best)), key=cost)
        if cost(sp) < UNREACHABLE:
            return sp
    x, d = on_ground(model, pid, danger)
    return Spot(pid, x, 1.0, model.safe(pid), round(d, 3))


def spot_for(model: MapModel, pid: int, left: list[Monster], current: Spot, here: Where, travel: Travel,
             worth: set[str] | None, danger: Danger | None = None) -> Spot | None:
    """A spot off `pid`, on a platform she can stand on (spot_ok, or with `danger` a place better than hunting on
    the ground: _better), that reaches more of the monsters `left` on it than where she is (`current`), so she
    keeps shooting from safety instead of going down into the pile. The one reaching the most, then the safest,
    then the nearest. None if there's none, or she isn't on a platform (no telling how far a spot is)."""

    on = {t.platform: model.platform(t.platform) for t in left if t.platform is not None}

    def reached(p: Platform, x: float, y: float) -> int:
        return sum(can_hit(p, x, y, t, target=on.get(t.platform), others=left) for t in left)

    cur = model.platform(current.platform)
    now = reached(cur, current.x, cur.y_at(current.x)) if current.platform != pid else 0
    ground = _ground_score(model, pid, danger)
    grid = _grid(model, pid, danger) if ground is not None else {}
    best = None  # (monsters reached, less danger, seconds to get there, platform, x)
    for p in model.places:
        if p.id == pid:
            continue
        ok = spot_ok(model, p.id, worth)
        for x in _xs_within(_stand_xs(p.left + EDGE_MARGIN, p.right - EDGE_MARGIN), p, _margin(p, danger)):
            c, d = grid.get((p.id, x), (0.0, 0.0))
            if not ok and not _better(c - DANGER_WEIGHT * d, ground):
                continue
            n = reached(p, x, p.y_at(x))
            if n <= now or (best is not None and n < best[0]):
                continue
            if here.platform is None:
                continue
            t = (abs(x - here.x) / WALK_SPEED if p.id == here.platform.id
                 else travel(here.platform.id, here.x, p.id))
            if t is not None and (best is None or (n, -round(d, 3), -t) > (best[0], best[1], -best[2])):
                best = (n, -round(d, 3), t, p.id, x)
    if best is None:
        return None
    _, safety, _, stand, x = best
    ground_p, p = model.platform(pid), model.platform(stand)
    return Spot(stand, x, round(coverage(ground_p, x, p.y_at(x), p), 3), model.safe(stand), -safety)


def spot_at(model: MapModel, pid: int, worth: set[str] | None, danger: Danger | None = None) -> tuple[int, float]:
    """Where she'll stand to hunt platform `pid` (for planning a sweep): the best spot she can stand on,
    else its safest place (on_ground: its middle without `danger`)."""
    sp = best_spot(model, pid, worth, danger)
    return sp.platform, sp.x


def best_spot(model: MapModel, pid: int, worth: set[str] | None, danger: Danger | None = None) -> Spot:
    """spot_at's spot, with its coverage and danger."""
    sp = next(_standable(model, pid, worth, danger), None)
    if sp is not None:
        return sp
    x, d = on_ground(model, pid, danger)
    return Spot(pid, x, 1.0, model.safe(pid), round(d, 3))
