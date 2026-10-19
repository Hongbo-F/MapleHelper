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
from collections.abc import Callable
from dataclasses import dataclass

from ..localization.mapdata import MapModel, Platform
from .physics import (CLIMB_SPEED, JUMP_HEIGHT, JUMP_SPEED, LAND_MARGIN, ROPE_REACH, WALK_SPEED, fall_time,
                      fly, sure_landing)


JUMP_MARGIN = 12         # don't plan jumps that need the last few units of height
MIN_JUMP_UP_RISE = 8     # a platform overhead must be more than this above to jump up onto it
# Running jumps: she stands RUN_START back from the end, runs, and presses jump RUN_MS later, which
# (after ~100 ms before she moves) puts the takeoff about JUMP_FROM back. Standing closer, she walked
# off the end before the jump started (5 of 5 tries); from 30-40 back it landed.
RUN_START = 40
RUN_MS = 150
JUMP_FROM = 33
LANDING_ROOM = LAND_MARGIN * 0.5  # a running jump must land at least this far in from the landing platform's ends
NEEDLESS_DROP = 10       # a running jump down more than this, where walking off gets there too, is needless
# Ropes: a rope hangs from the platform within ROPE_TOP_DX sideways and ROPE_TOP_DY up or down of its
# top, and can be grabbed from a platform it ends at most ROPE_REACH above (or ROPE_PAST below).
ROPE_TOP_DX = 4
ROPE_TOP_DY = 20
ROPE_PAST = 5
# Seconds each kind of move costs on top of its simulated flight, climb or fall.
CLIMB_EXTRA_S = 0.6
JUMP_EXTRA_S = 0.2
WALK_OFF_EXTRA_S = 0.3
DROP_EXTRA_S = 0.4
# Seconds added to a move's cost for how often it goes wrong (a miss costs ~2-3 s to recover from).
# Moving one move at a time, running jumps missed often and this was 1.5 s; chained (flow.py) about
# one move in 15 lands elsewhere, mostly drops and straight-up jumps timed off.
RISK = {"jump": 0.25, "walk_off": 0.2, "drop": 0.25}
ALONG_STEP = 10      # running jumps are tried from every this many units along a platform (20 missed
                     # the 30-unit window of takeoffs from platform 4 up onto 2, so 2 had no route) ...
ALONG_SPREAD = 60    # ... and kept this far apart ...
ALONG_KEEP = 3       # ... up to this many (plus the last) per landing platform

SIDES = (("right", 1), ("left", -1))  # a direction and its sign along x


@dataclass(frozen=True)
class Move:
    """One move from platform src to platform dst. flow adds 'hop' (a running jump over a monster,
    landing on the same platform) to the kinds below."""
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


def moves(model: MapModel) -> list[Move]:
    """Every move the physics allows on the map: the rope moves, then each platform's own moves."""
    out = _rope_moves(model)
    # Jumps, walk-offs and drops: simulated with the game's physics, so the landing platform is
    # wherever her feet first cross a platform on the way down (not the one aimed at).
    for a in model.platforms:
        mine = _jump_up_moves(model, a) + _end_moves(model, a) + _along_jumps(model, a)
        out += _without_needless_jumps(model, a, mine) + _drop_moves(model, a)
    return out


def _rope_moves(model: MapModel) -> list[Move]:
    """Climbing each rope from every platform she can grab it from, and climbing down it."""
    out: list[Move] = []
    for rope in model.ropes:
        top = next((p for p in model.platforms
                    if p.covers(rope.x, ROPE_TOP_DX) and abs(p.y_at(rope.x) - rope.top) <= ROPE_TOP_DY), None)
        if top is None:
            continue
        for p in model.platforms:
            if not p.covers(rope.x) or p is top:
                continue
            gap = p.y_at(rope.x) - rope.bottom  # how far the rope ends above her feet
            if -ROPE_PAST <= gap <= ROPE_REACH:
                out.append(Move("climb", p.id, top.id, rope.x, rope.x,
                                seconds=CLIMB_EXTRA_S + (p.y_at(rope.x) - rope.top) / CLIMB_SPEED,
                                rope_x=rope.x, rope_top=rope.top, rope_bottom=rope.bottom))
        below = model.platform_below(rope.x, rope.bottom)
        if below is not None:
            out.append(Move("climb_down", top.id, below.id, rope.x, rope.x,
                            seconds=(rope.bottom - rope.top) / CLIMB_SPEED + fall_time(below.y_at(rope.x) - rope.bottom),
                            rope_x=rope.x, rope_top=rope.top, rope_bottom=rope.bottom))
    return out


def _jump_up_moves(model: MapModel, a: Platform) -> list[Move]:
    """Straight up from a through each platform overhead, in the middle of where they overlap."""
    out: list[Move] = []
    for b in model.platforms:
        if b is a:
            continue
        x = _overlap_middle(a, b)
        if x is None:
            continue
        rise = a.y_at(x) - b.y_at(x)
        if not MIN_JUMP_UP_RISE < rise <= JUMP_HEIGHT - JUMP_MARGIN:
            continue
        land, land_x, t = fly(model, x, a.y_at(x), 0.0, -JUMP_SPEED)
        if land is b:
            out.append(Move("jump_up", a.id, b.id, x, land_x, seconds=t + JUMP_EXTRA_S))
    return out


def _end_moves(model: MapModel, a: Platform) -> list[Move]:
    """Off each end of a (right, then left): a running jump, and walking off."""
    out: list[Move] = []
    for direction, sign in SIDES:
        edge = a.right if sign > 0 else a.left
        # Running jump off this end.
        start, takeoff = edge - sign * RUN_START, edge - sign * JUMP_FROM
        if a.covers(start):
            jump = _sure_running_jump(model, a, takeoff, sign)
            if jump is not None:
                land, land_x, t = jump
                out.append(Move("jump", a.id, land.id, start, land_x, direction, seconds=_running_jump_seconds(t)))
        # Walking off this end: from just inside it when a wall rises there (it stops her), else
        # from just past it. A cliff dropping from the end (34 down to 36) isn't in her way, but
        # starting inside it read as bumping into the cliff's top.
        level = a.y_at(edge)
        rising = any(w.x1 == w.x2 and abs(w.x1 - edge) < 1 and min(w.y1, w.y2) < level - 2 <= max(w.y1, w.y2)
                     for w in model.walls)
        land, land_x, t = fly(model, edge + sign * (-1 if rising else 0.5), a.y_at(edge) - 1,
                              sign * WALK_SPEED, 0.0)
        if land is not None and land is not a:
            # She starts 5 units in from the end.
            out.append(Move("walk_off", a.id, land.id, edge - sign * 5, land_x, direction,
                            seconds=WALK_OFF_EXTRA_S + t))
    return out


def _along_jumps(model: MapModel, a: Platform) -> list[Move]:
    """Running jumps from along a, not only its ends: a player jumps from wherever that reaches another
    platform, up, across or down. A few takeoffs per landing platform, spread out (right, then left)."""
    out: list[Move] = []
    for direction, sign in SIDES:
        found: dict[int, list[tuple[float, float, float]]] = {}  # landing platform -> (takeoff, land_x, t)
        x = a.left + JUMP_FROM
        while x <= a.right - JUMP_FROM:
            jump = _sure_running_jump(model, a, x, sign)
            if jump is not None:
                land, land_x, t = jump
                found.setdefault(land.id, []).append((x, land_x, t))
            x += ALONG_STEP
        for land_id, options in found.items():
            kept: list[tuple[float, float, float]] = []
            for opt in options:
                if all(abs(opt[0] - k[0]) >= ALONG_SPREAD for k in kept):
                    kept.append(opt)
            for takeoff, land_x, t in kept[:ALONG_KEEP] + kept[ALONG_KEEP:][-1:]:  # the first few and the last
                start = takeoff - sign * (RUN_START - JUMP_FROM)
                if a.covers(start):
                    out.append(Move("jump", a.id, land_id, start, land_x, direction,
                                    seconds=_running_jump_seconds(t), takeoff=takeoff))
    return out


def _without_needless_jumps(model: MapModel, a: Platform, mine: list[Move]) -> list[Move]:
    """a's moves (mine) without running jumps down to a platform that walking off an end reaches
    anyway: she walks off, as a player does."""
    offs = {(m.direction, m.dst) for m in mine if m.kind == "walk_off"}
    return [m for m in mine if not (m.kind == "jump" and (m.direction, m.dst) in offs
                                     and model.platform(m.dst).y_at(m.land_x) > a.y_at(m.x) + NEEDLESS_DROP)]


def _drop_moves(model: MapModel, a: Platform) -> list[Move]:
    """Dropping through a (down + jump) to each different platform underneath."""
    out: list[Move] = []
    seen = set()
    for b in model.platforms:
        if b is a:
            continue
        x = _overlap_middle(a, b)
        if x is None:
            continue
        land, land_x, t = fly(model, x, a.y_at(x) + 1, 0.0, 0.0, skip=a)
        if land is not None and land.id not in seen:
            seen.add(land.id)
            out.append(Move("drop", a.id, land.id, x, land_x, seconds=DROP_EXTRA_S + t))
    return out


def _overlap_middle(a: Platform, b: Platform) -> float | None:
    """The middle of where a and b overlap, LAND_MARGIN in from both ends (None: they overlap less)."""
    lo, hi = max(a.left, b.left) + LAND_MARGIN, min(a.right, b.right) - LAND_MARGIN
    return None if lo > hi else (lo + hi) / 2


def _sure_running_jump(model: MapModel, a: Platform, x: float, sign: int) -> tuple[Platform, float, float] | None:
    """Where a running jump leaving a at x (going that way) lands, if it surely lands on another platform
    with room to spare: (platform, landing x, seconds in the air)."""
    land, land_x, t = fly(model, x, a.y_at(x), sign * WALK_SPEED, -JUMP_SPEED)
    if land is not None and land is not a and land.covers(land_x - sign * LANDING_ROOM) \
            and land.covers(land_x + sign * LANDING_ROOM) \
            and sure_landing(model, x, a.y_at(x), sign * WALK_SPEED, -JUMP_SPEED, land):
        return land, land_x, t
    return None


def _running_jump_seconds(t: float) -> float:
    """A running jump's cost: t s in the air, its run-up and the jump's own extra."""
    return t + JUMP_EXTRA_S + RUN_MS / 1000


class Planner:
    """The map's moves, and the quickest routes over them."""

    def __init__(self, model: MapModel):
        self.model = model
        self.moves = moves(model)
        self.by_src: dict[int, list[Move]] = {}
        for m in self.moves:
            self.by_src.setdefault(m.src, []).append(m)

    def plan(self, src: int, x: float, dst: int, penalty: Callable[[Move], float] | None = None,
             weight: Callable[[Move], float] | None = None) -> list[Move] | None:
        """Quickest list of moves from (platform src, x) onto platform dst (None: there's no way): the
        route ends wherever she lands on dst.
        `penalty(move)` adds seconds to a move (e.g. one a monster is in the way of); `weight(move)`
        scales its cost (a little randomness lets near-equal routes take turns)."""
        found = self._search(src, x, dst, penalty, weight)
        return None if found is None else found[0]

    def travel(self, src: int, x: float, dst: int) -> float | None:
        """Seconds the quickest route from (platform src, x) to platform dst takes (by the model)."""
        found = self._search(src, x, dst, None, None)
        return None if found is None else found[1]

    def _search(self, src: int, x: float, dst: int, penalty: Callable[[Move], float] | None,
                weight: Callable[[Move], float] | None) -> tuple[list[Move], float] | None:
        """Dijkstra over (platform, x) nodes: the quickest moves to dst and their cost in seconds."""
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
        """The platforms some route leads to from src (src included)."""
        seen, todo = {src}, [src]
        while todo:
            for m in self.by_src.get(todo.pop(), []):
                if m.dst not in seen:
                    seen.add(m.dst)
                    todo.append(m.dst)
        return seen


def route_record(dst: int, moves: list[Move]) -> dict:
    """A planned route as telemetry data ('route')."""
    return {"dst": dst, "moves": [{"kind": m.kind, "src": m.src, "dst": m.dst, "x": round(m.x, 1),
                                   "land_x": round(m.land_x, 1)} for m in moves]}
