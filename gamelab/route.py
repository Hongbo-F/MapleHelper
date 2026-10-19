"""Routes across a map: which moves link which platforms, and the quickest way from here to there.

Moves between platforms (all in map units, using the game's physics and her measured speed):
- climb: a rope or ladder whose bottom she can reach from a platform, up to the platform it hangs from;
- climb_down: from a rope's top down to its bottom, falling onto what's below;
- jump_up: straight up through a platform overhead (floating platforms can be jumped through from below);
- jump: a running jump off one platform's end onto another (up, level or down);
- walk_off: walk off a platform's end and fall onto the platform below;
- drop: down + jump to fall through the platform she stands on.
Walking along a platform links its moves; routes are found with Dijkstra on time.
"""

import bisect
import heapq
import math
from dataclasses import dataclass

from .mapdata import MapModel, Platform

WALK_SPEED = 149.0       # map units/s: her measured walk (209 screen px/s at 1.40625 px per unit)
CLIMB_SPEED = 117.0      # map units/s: measured 0.165 screen px/ms
JUMP_SPEED = 555.0       # physics: jumpSpeed
GRAVITY = 2000.0         # physics: gravityAcc
FALL_SPEED = 670.0       # physics: fallSpeed (terminal)
JUMP_HEIGHT = JUMP_SPEED ** 2 / (2 * GRAVITY)   # 77 map units
JUMP_MARGIN = 12         # don't plan jumps that need the last few units of height
ROPE_REACH = 95          # she can grab a rope that ends this far above her feet (jump + arms)
LAND_MARGIN = 18         # land at least this far onto a platform
# Running jumps: she stands RUN_START back from the end, runs, and presses jump RUN_MS later, which
# (after ~100 ms before she moves) puts the takeoff about JUMP_FROM back. Standing closer, she walked
# off the end before the jump started (5 of 5 tries); from 30-40 back it landed.
RUN_START = 40
RUN_MS = 150
JUMP_FROM = 33
# Seconds added to a move's cost for how often it goes wrong (a miss costs ~2-3 s to recover from).
# Moving one move at a time, running jumps missed often and this was 1.5 s; chained (flow.py) about
# one move in 15 lands elsewhere, mostly drops and straight-up jumps timed off.
RISK = {"jump": 0.25, "walk_off": 0.2, "drop": 0.25}
ALONG_STEP = 10      # running jumps are tried from every this many units along a platform (20 missed
                     # the 30-unit window of takeoffs from platform 4 up onto 2, so 2 had no route) ...
ALONG_SPREAD = 60    # ... and kept this far apart ...
ALONG_KEEP = 3       # ... up to this many (plus the last) per landing platform


@dataclass(frozen=True)
class Move:
    kind: str            # climb | climb_down | jump_up | jump | walk_off | drop
    src: int             # platform ids
    dst: int
    x: float             # where on src she starts the move
    land_x: float        # where on dst she ends up
    direction: str = ""  # left/right for jumps and walk_off
    seconds: float = 1.0
    rope_x: float | None = None
    rope_top: float | None = None
    rope_bottom: float | None = None
    takeoff: float | None = None  # running jumps from along a platform: where she leaves the ground


_FLY_CACHE: dict[tuple, tuple] = {}
_WALLS: dict[int, tuple] = {}
# The caches are keyed by id(model), so the models are kept: a freed model's id is reused by the next
# one, which then got the old one's walls and landings (tests failed depending on their order).
_MODELS: dict[int, MapModel] = {}


def _walls(model: MapModel) -> tuple[list[float], list[tuple[float, float]]]:
    """The model's vertical walls sorted by x: their xs, and each one's (top, bottom)."""
    key = id(model)
    _MODELS[key] = model
    if key not in _WALLS:
        ws = sorted((w for w in model.walls if w.x1 == w.x2), key=lambda w: w.x1)
        _WALLS[key] = ([w.x1 for w in ws], [(min(w.y1, w.y2) - 1, max(w.y1, w.y2) + 1) for w in ws])
    return _WALLS[key]


def fly(model: MapModel, x: float, y: float, vx: float, vy: float, skip: Platform | None = None,
        dt: float = 0.004, limit: float = 4.0) -> tuple[Platform | None, float, float]:
    """Where she lands from (x, y) moving at (vx, vy) (map units/s, y down): the first platform whose
    surface her feet cross on the way down. Walls stop her sideways. `skip` is a platform she falls
    through (dropping). Returns (platform, landing x, seconds). Results are cached: chains try the
    same jumps many times over."""
    key = (id(model), round(x, 2), round(y, 2), vx, vy, None if skip is None else skip.id, dt, limit)
    _MODELS[id(model)] = model
    if key in _FLY_CACHE:
        return _FLY_CACHE[key]
    wall_xs, wall_ys = _walls(model)
    # Only platforms she could reach in the time: within her sideways travel, at or below the top
    # of the jump.
    reach = abs(vx) * limit + 1
    top = y - (vy * vy / (2 * GRAVITY) if vy < 0 else 0) - 1
    plats = [q for q in model.platforms if q is not skip and q.right >= x - reach and q.left <= x + reach
             and max(q.y_at(q.left), q.y_at(q.right)) >= top]
    t, result = 0.0, None
    while t < limit:
        nx, ny, nvy = x + vx * dt, y + vy * dt, min(vy + GRAVITY * dt, FALL_SPEED)
        if vx:
            lo, hi = (x, nx) if nx >= x else (nx, x)
            i = bisect.bisect_left(wall_xs, lo)
            while i < len(wall_xs) and wall_xs[i] <= hi:
                if wall_ys[i][0] <= ny <= wall_ys[i][1]:
                    nx, vx = x, 0.0  # bumped into a wall
                    break
                i += 1
        if (vy > 0 or nvy > 0) and t > 0 and ny >= y:
            for q in plats:
                if q.covers(nx) and y <= q.y_at(nx) <= ny + 0.01:
                    result = (q, nx, t + dt)
                    break
            if result:
                break
        x, y, vy, t = nx, ny, nvy, t + dt
    result = result or (None, x, t)
    _FLY_CACHE[key] = result
    return result


def sure_landing(model: MapModel, x: float, y: float, vx: float, vy: float, land: Platform,
                 slop: float = 12.0) -> bool:
    """Whether a jump from (x, y) still lands on `land` if she leaves the ground `slop` units early or
    late, or 10% slower: a jump down from platform 25 meant for the basin landed on platform 33,
    whose edge its arc passed close by."""
    sign = 1 if vx > 0 else -1 if vx < 0 else 0
    for dx, k in ((-slop, 1.0), (slop, 1.0), (0.0, 0.9)):
        x0 = x + sign * dx
        got, _, _ = fly(model, x0, y, vx * k, vy)
        if got is not land:
            return False
    return True


def air_time(dy_up: float) -> float | None:
    """Seconds from takeoff until she comes down through a height dy_up above takeoff (negative = below)."""
    disc = JUMP_SPEED ** 2 - 2 * GRAVITY * dy_up
    if disc < 0:
        return None
    return (JUMP_SPEED + math.sqrt(disc)) / GRAVITY


def fall_time(dy: float) -> float:
    return math.sqrt(2 * max(0.0, dy) / GRAVITY)


def moves(model: MapModel) -> list[Move]:
    out: list[Move] = []
    plats = model.platforms

    def landing(x: float, y: float) -> Platform | None:
        return model.platform_below(x, y)

    for rope in model.ropes:
        top = next((p for p in plats if p.covers(rope.x, 4) and abs(p.y_at(rope.x) - rope.top) <= 20), None)
        for p in plats:
            if not p.covers(rope.x) or p is top:
                continue
            gap = p.y_at(rope.x) - rope.bottom  # how far the rope ends above her feet
            if -5 <= gap <= ROPE_REACH and top is not None:
                out.append(Move("climb", p.id, top.id, rope.x, rope.x,
                                seconds=0.6 + (p.y_at(rope.x) - rope.top) / CLIMB_SPEED,
                                rope_x=rope.x, rope_top=rope.top, rope_bottom=rope.bottom))
        if top is not None:
            below = landing(rope.x, rope.bottom)
            if below is not None:
                out.append(Move("climb_down", top.id, below.id, rope.x, rope.x,
                                seconds=(rope.bottom - rope.top) / CLIMB_SPEED + fall_time(below.y_at(rope.x) - rope.bottom),
                                rope_x=rope.x, rope_top=rope.top, rope_bottom=rope.bottom))

    # Jumps, walk-offs and drops: simulated with the game's physics, so the landing platform is
    # wherever her feet first cross a platform on the way down (not the one aimed at).
    for a in plats:
        # Straight up through each platform overhead, where they overlap.
        for b in plats:
            if b is a:
                continue
            lo, hi = max(a.left, b.left) + LAND_MARGIN, min(a.right, b.right) - LAND_MARGIN
            if lo > hi:
                continue
            x = (lo + hi) / 2
            rise = a.y_at(x) - b.y_at(x)
            if not 8 < rise <= JUMP_HEIGHT - JUMP_MARGIN:
                continue
            land, land_x, t = fly(model, x, a.y_at(x), 0.0, -JUMP_SPEED)
            if land is b:
                out.append(Move("jump_up", a.id, b.id, x, land_x, seconds=t + 0.2))
        for direction, edge, sign in (("right", a.right, 1), ("left", a.left, -1)):
            # Running jump off this end.
            start, takeoff = edge - sign * RUN_START, edge - sign * JUMP_FROM
            if a.covers(start):
                land, land_x, t = fly(model, takeoff, a.y_at(takeoff), sign * WALK_SPEED, -JUMP_SPEED)
                if land is not None and land is not a and land.covers(land_x - sign * LAND_MARGIN * 0.5) \
                        and land.covers(land_x + sign * LAND_MARGIN * 0.5) \
                        and sure_landing(model, takeoff, a.y_at(takeoff), sign * WALK_SPEED, -JUMP_SPEED, land):
                    out.append(Move("jump", a.id, land.id, start, land_x, direction,
                                    seconds=t + 0.2 + RUN_MS / 1000))
            # Walking off this end: from just inside it when a wall rises there (it stops her), else
            # from just past it. A cliff dropping from the end (34 down to 36) isn't in her way, but
            # starting inside it read as bumping into the cliff's top.
            level = a.y_at(edge)
            rising = any(w.x1 == w.x2 and abs(w.x1 - edge) < 1 and min(w.y1, w.y2) < level - 2 <= max(w.y1, w.y2)
                         for w in model.walls)
            land, land_x, t = fly(model, edge + sign * (-1 if rising else 0.5), a.y_at(edge) - 1,
                                  sign * WALK_SPEED, 0.0)
            if land is not None and land is not a:
                out.append(Move("walk_off", a.id, land.id, edge - sign * 5, land_x, direction, seconds=0.3 + t))
        # Running jumps from along the platform, not only its ends: a player jumps from wherever that
        # reaches another platform, up, across or down. A few takeoffs per landing platform, spread out.
        for direction, sign in (("right", 1), ("left", -1)):
            found: dict[int, list[tuple[float, float, float]]] = {}
            x = a.left + JUMP_FROM
            while x <= a.right - JUMP_FROM:
                land, land_x, t = fly(model, x, a.y_at(x), sign * WALK_SPEED, -JUMP_SPEED)
                if land is not None and land is not a and land.covers(land_x - sign * LAND_MARGIN * 0.5) \
                        and land.covers(land_x + sign * LAND_MARGIN * 0.5) \
                        and sure_landing(model, x, a.y_at(x), sign * WALK_SPEED, -JUMP_SPEED, land):
                    found.setdefault(land.id, []).append((x, land_x, t))
                x += ALONG_STEP
            for land_id, options in found.items():
                kept: list[tuple[float, float, float]] = []
                for opt in options:
                    if all(abs(opt[0] - k[0]) >= ALONG_SPREAD for k in kept):
                        kept.append(opt)
                for takeoff, land_x, t in kept[:ALONG_KEEP] + kept[ALONG_KEEP:][-1:]:
                    start = takeoff - sign * (RUN_START - JUMP_FROM)
                    if a.covers(start):
                        out.append(Move("jump", a.id, land_id, start, land_x, direction,
                                        seconds=t + 0.2 + RUN_MS / 1000, takeoff=takeoff))
        # Down to a platform that walking off an end reaches anyway: walk off, as a player does; a
        # running jump down there is a needless jump.
        offs = {(m.direction, m.dst) for m in out if m.kind == "walk_off" and m.src == a.id}
        out = [m for m in out if not (m.kind == "jump" and m.src == a.id and (m.direction, m.dst) in offs
                                      and model.platform(m.dst).y_at(m.land_x) > a.y_at(m.x) + 10)]
        # Dropping through (down + jump) to each different platform underneath.
        seen = set()
        for b in plats:
            if b is a:
                continue
            lo, hi = max(a.left, b.left) + LAND_MARGIN, min(a.right, b.right) - LAND_MARGIN
            if lo > hi:
                continue
            x = (lo + hi) / 2
            land, land_x, t = fly(model, x, a.y_at(x) + 1, 0.0, 0.0, skip=a)
            if land is not None and land.id not in seen:
                seen.add(land.id)
                out.append(Move("drop", a.id, land.id, x, land_x, seconds=0.4 + t))
    return out


class Planner:
    def __init__(self, model: MapModel):
        self.model = model
        self.moves = moves(model)
        self.by_src: dict[int, list[Move]] = {}
        for m in self.moves:
            self.by_src.setdefault(m.src, []).append(m)

    def plan(self, src: int, x: float, dst: int, dst_x: float | None = None,
             penalty=None, weight=None) -> list[Move] | None:
        """Quickest list of moves from (platform src, x) to platform dst (and walking to dst_x).
        `penalty(move)` adds seconds to a move (e.g. one a monster is in the way of); `weight(move)`
        scales its cost (a little randomness lets near-equal routes take turns)."""
        found = self._search(src, x, dst, penalty, weight)
        return None if found is None else found[0]

    def travel(self, src: int, x: float, dst: int) -> float | None:
        """Seconds the quickest route from (platform src, x) to platform dst takes (by the model)."""
        found = self._search(src, x, dst, None, None)
        return None if found is None else found[1]

    def _search(self, src: int, x: float, dst: int, penalty, weight) -> tuple[list[Move], float] | None:
        start = (src, round(x))
        best = {start: 0.0}
        back: dict[tuple, tuple] = {}
        heap = [(0.0, start)]
        while heap:
            t, (pid, px) = heapq.heappop(heap)
            if t > best.get((pid, px), math.inf):
                continue
            if pid == dst:
                path, node = [], (pid, px)
                while node in back:
                    node, move = back[node]
                    path.append(move)
                return path[::-1], t
            for m in self.by_src.get(pid, []):
                step = abs(m.x - px) / WALK_SPEED + m.seconds + RISK.get(m.kind, 0.0)
                cost = t + step * (weight(m) if weight is not None else 1.0)
                if penalty is not None:
                    cost += penalty(m)
                node = (m.dst, round(m.land_x))
                if cost < best.get(node, math.inf):
                    best[node] = cost
                    back[node] = ((pid, px), m)
                    heapq.heappush(heap, (cost, node))
        return None

    def reachable(self, src: int) -> set[int]:
        seen, todo = {src}, [src]
        while todo:
            for m in self.by_src.get(todo.pop(), []):
                if m.dst not in seen:
                    seen.add(m.dst)
                    todo.append(m.dst)
        return seen
