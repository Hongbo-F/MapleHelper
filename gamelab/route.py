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

import heapq
import math
from dataclasses import dataclass

from .mapdata import MapModel, Platform

WALK_SPEED = 149.0       # map units/s: her measured walk (209 screen px/s at 1.40625 px per unit)
CLIMB_SPEED = 117.0      # map units/s: measured 0.165 screen px/ms
JUMP_SPEED = 555.0       # physics: jumpSpeed
GRAVITY = 2000.0         # physics: gravityAcc
JUMP_HEIGHT = JUMP_SPEED ** 2 / (2 * GRAVITY)   # 77 map units
JUMP_MARGIN = 12         # don't plan jumps that need the last few units of height
ROPE_REACH = 95          # she can grab a rope that ends this far above her feet (jump + arms)
LAND_MARGIN = 18         # land at least this far onto a platform
EDGE_MARGIN = 12         # take off this far before a platform's end


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

    for a in plats:
        for b in plats:
            if a is b:
                continue
            # Straight up through b from a, where they overlap.
            lo, hi = max(a.left, b.left) + LAND_MARGIN, min(a.right, b.right) - LAND_MARGIN
            if lo <= hi:
                x = (lo + hi) / 2
                rise = a.y_at(x) - b.y_at(x)
                if 8 < rise <= JUMP_HEIGHT - JUMP_MARGIN:
                    out.append(Move("jump_up", a.id, b.id, x, x, seconds=air_time(rise) + 0.2))
            # Running jumps off a's ends.
            for direction, edge, sign in (("right", a.right, 1), ("left", a.left, -1)):
                takeoff = edge - sign * EDGE_MARGIN
                near = b.left if sign > 0 else b.right
                gap = (near - edge) * sign
                if gap < -LAND_MARGIN:  # b doesn't start beyond this end
                    continue
                land_x = near + sign * LAND_MARGIN
                if not b.covers(land_x):
                    continue
                rise = a.y_at(takeoff) - b.y_at(land_x)
                if rise > JUMP_HEIGHT - JUMP_MARGIN:
                    continue
                t = air_time(rise)
                if t is None or WALK_SPEED * t < (land_x - takeoff) * sign:
                    continue
                # Don't plan a jump that would land on something closer underneath first.
                first = model.platform_below(land_x, b.y_at(land_x) - 1)
                if rise < 0 and first is not None and first.id != b.id and first.y_at(land_x) < b.y_at(land_x):
                    continue
                out.append(Move("jump", a.id, b.id, takeoff, land_x, direction, seconds=t + 0.2))

    for a in plats:
        for direction, edge, sign in (("right", a.right, 1), ("left", a.left, -1)):
            x = edge + sign * 25
            if any(w.x1 == w.x2 and abs(w.x1 - edge) < 30 and min(w.y1, w.y2) < a.y_at(edge) - 5 for w in model.walls):
                continue  # a wall rises at this end: can't walk off
            b = landing(x, a.y_at(edge))
            if b is not None:
                out.append(Move("walk_off", a.id, b.id, edge - sign * 5, x, direction,
                                seconds=0.3 + fall_time(b.y_at(x) - a.y_at(edge))))
        # Drop through to each platform underneath (once per platform, mid-overlap).
        seen = set()
        for b in plats:
            if b is a or b.id in seen:
                continue
            lo, hi = max(a.left, b.left) + LAND_MARGIN, min(a.right, b.right) - LAND_MARGIN
            if lo > hi:
                continue
            x = (lo + hi) / 2
            if landing(x, a.y_at(x)) is b:
                seen.add(b.id)
                out.append(Move("drop", a.id, b.id, x, x, seconds=0.4 + fall_time(b.y_at(x) - a.y_at(x))))
    return out


class Planner:
    def __init__(self, model: MapModel):
        self.model = model
        self.moves = moves(model)
        self.by_src: dict[int, list[Move]] = {}
        for m in self.moves:
            self.by_src.setdefault(m.src, []).append(m)

    def plan(self, src: int, x: float, dst: int, dst_x: float | None = None) -> list[Move] | None:
        """Quickest list of moves from (platform src, x) to platform dst (and walking to dst_x)."""
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
                return path[::-1]
            for m in self.by_src.get(pid, []):
                cost = t + abs(m.x - px) / WALK_SPEED + m.seconds
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
