"""Routes across a map: which moves link which platforms, and the quickest way from here to there.

Moves between platforms (all in map units, using the game's physics and her measured speed):
- climb: a rope or ladder whose bottom she can reach from a platform, up to the platform it hangs from;
- climb_down: from a rope's top down to its bottom, falling onto what's below;
- jump_up: straight up through a platform overhead (floating platforms can be jumped through from below);
- jump: a running jump off one platform's end onto another (up, level or down);
- walk_off: walk off a platform's end and fall onto the platform below;
- drop: down + jump to fall through the platform she stands on (not where its foothold forbids it).
In the air only walls of the foothold group she leaves stop her (body.OWN_GROUP_WALLS). Walking along a
platform links its moves; routes are found with Dijkstra on time.
"""

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass

from maplehelper.engine import LEFT, RIGHT

from .. import directions
from ..localization.mapdata import MapModel, Platform
from .body import slide, walls_for
from .physics import (CLIMB_SPEED, JUMP_HEIGHT, JUMP_SPEED, LAND_MARGIN, ROPE_REACH, SHOE, WALK_SPEED,
                      fall_time, flight, fly, sure_landing)


JUMP_MARGIN = 12         # don't plan jumps that need the last few units of height
MIN_JUMP_UP_RISE = 8     # a platform overhead must be more than this above to jump up onto it
# Running jumps: she stands RUN_START back from the end, runs, and presses jump RUN_MS later, which
# (after ~100 ms before she moves) puts the takeoff about JUMP_FROM back. Standing closer, she walked
# off the end before the jump started (5 of 5 tries); from 30-40 back it landed. (That was the mover's
# old open-loop timing, from standing. JUMP_FROM is where a jump off an end is looked for; where it takes
# off is placed below.)
RUN_START = 40
RUN_MS = 150
JUMP_FROM = 33
# Where a running jump takes off (_placed). Of tonight's 293 chained jumps up onto the next step of a
# staircase (60 up, the next platform's end 12 across: 3 -> 1, 19 -> 18, 29 -> 28, 32 -> 29 and the like;
# telemetry of 2026-10-21), the jumps that took off 33-36 units back from the end missed 4 of 117 times,
# 36-39 back 5 of 75, 39-42 back 9 of 47 and 42-45 back 8 of 36, coming down short of the platform: from
# there the jump lands 16 units onto it at most, 4 when flow takes off 12 further back (JUMP_FROM to
# flow.JUMP_FROM_FAR, "players jump from a little further back too"). By the landing planned, 16 of 102
# that were to land under 12 in from an end missed, and 1 of 61 of those 20 or more in. So a jump's takeoff
# is placed where it still lands TAKEOFF_ROOM_WANTED in if she takes off up to TAKEOFF_SPREAD earlier (flow
# takes off that much earlier than the move's takeoff with rng: Flow._jump) and TAKEOFF_SLOP early or late
# on top of that (sure_landing's slop); if no takeoff manages that, where she has the most room with the
# slop alone. And never nearer the end than TAKEOFF_EDGE, so that leaving TAKEOFF_SLOP late still leaves the
# 3 units flow keeps. Up the staircase that's 15 back (it was 33): she lands 34 in, 24 with flow's spread.
TAKEOFF_EDGE = 15
TAKEOFF_SPREAD = 10.0
TAKEOFF_SLOP = 12.0
TAKEOFF_ROOM_WANTED = 20  # landing room enough (no better takeoff is sought)
TAKEOFF_SEARCH = 30       # how far from where a jump was found its takeoff may be moved
RUN_UP = 20               # a running jump starts this far before its takeoff (mover.MIN_RUN 12 and some)
LANDING_ROOM = LAND_MARGIN * 0.5  # a running jump must land at least this far in from the landing platform's ends,
                                  # and a walk-off this far more than her slide from its far end
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
# Where she drops: down on a rope's top climbs onto the rope, and on a foothold that forbids it (forbidFallDown)
# the drop does nothing. So not within ROPE_CLEAR of the top of a rope hanging from her platform, nor within
# DROP_ROOM of a foothold that forbids it: where she stands to drop is good to about that (flow.STEP_ACCEPT,
# the units within which a plan takes her as standing where it wants her; the mover's walks).
ROPE_CLEAR = 30
DROP_ROOM = 8
# Nor where she comes down within KNOCK_CARRY of an end of the platform below with nothing under it (void_ends:
# on this map 36's, the basin's cliffs). In the air from another block she keeps its foothold group, whose walls
# don't stop her (body.OWN_GROUP_WALLS), and a touch's knock as she comes down carries her up to this far sideways
# before she lands (the simulator's knock, 180 units/s and 220 up, given as high as the tallest body here reaches,
# an Orange Mushroom's 67: 0.39 s in the air). In the harness, knocks off drops from 31 at 185 (56 from 36's left
# cliff) and from 33 at 907 (44 from its right one) carried her through the cliff and out of the map.
KNOCK_CARRY = 70
DROP_STEP = 2        # units between the drop spots tried
ALONG_STEP = 10      # running jumps are tried from every this many units along a platform (20 missed
                     # the 30-unit window of takeoffs from platform 4 up onto 2, so 2 had no route) ...
ALONG_SPREAD = 60    # ... and kept this far apart ...
ALONG_KEEP = 3       # ... up to this many (plus the last) per landing platform
TIMES_KEPT = 256     # Planner.times: starts whose route times to every platform are kept (a few KB each)

SIDES = tuple((d, directions.sign(d)) for d in (RIGHT, LEFT))  # a direction and its sign along x


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
        out += _without_needless_jumps(model, a, _distinct(mine)) + _drop_moves(model, a)
    return out


def _distinct(mine: list[Move]) -> list[Move]:
    """a's moves without a jump that takes off where one before it does for the same platform (a jump off an
    end and one found along the platform are placed at the same takeoff when that lands with the most room)."""
    seen: set[tuple] = set()
    out = []
    for m in mine:
        key = (m.kind, m.dst, m.direction, m.takeoff)
        if m.kind != "jump" or key not in seen:
            seen.add(key)
            out.append(m)
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
        land, land_x, t = fly(model, x, a.y_at(x), 0.0, -JUMP_SPEED, group=a.group)
        if land is b:
            out.append(Move("jump_up", a.id, b.id, x, land_x, seconds=t + JUMP_EXTRA_S))
    return out


def _end_moves(model: MapModel, a: Platform) -> list[Move]:
    """Off each end of a (right, then left): a running jump, and walking off."""
    out: list[Move] = []
    for direction, sign in SIDES:
        edge = a.right if sign > 0 else a.left
        # Running jump off this end: looked for from JUMP_FROM back, taking off where it lands with room (_placed).
        if a.covers(edge - sign * RUN_START):
            jump = _sure_running_jump(model, a, edge - sign * JUMP_FROM, sign)
            if jump is not None:
                placed = _placed(model, a, edge - sign * JUMP_FROM, sign, jump[0])
                if placed is not None:
                    out.append(placed)
        # Walking off this end: from just inside it when a wall of her group rises there (it stops her),
        # else from just past it. A cliff dropping from the end (34 down to 36) isn't in her way, but
        # starting inside it read as bumping into the cliff's top.
        level = a.y_at(edge)
        rising = any(w.x1 == w.x2 and abs(w.x1 - edge) < 1 and min(w.y1, w.y2) < level - 2 <= max(w.y1, w.y2)
                     for w in walls_for(model, a.group))
        land, land_x, t, v_land = flight(model, edge + sign * (-1 if rising else 0.5), a.y_at(edge) - 1,
                                         sign * WALK_SPEED, 0.0, group=a.group)
        # She lands at full speed (unless a wall stopped her), so she must be able to stop on it: her slide
        # and LANDING_ROOM more before its far end. Walking off 10's right end, 10 units from 15's right
        # end, she slid off it onto 18 every time in the simulator.
        if land is not None and land is not a \
                and land.covers(land_x + ((v_land > 0) - (v_land < 0)) * (slide(v_land, SHOE) + LANDING_ROOM)):
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
                placed = _placed(model, a, takeoff, sign, model.platform(land_id))
                if placed is not None:
                    out.append(placed)  # (two placed at the same takeoff: _distinct keeps one)
    return out


def _without_needless_jumps(model: MapModel, a: Platform, mine: list[Move]) -> list[Move]:
    """a's moves (mine) without running jumps down to a platform that walking off an end reaches
    anyway: she walks off, as a player does."""
    offs = {(m.direction, m.dst) for m in mine if m.kind == "walk_off"}
    return [m for m in mine if not (m.kind == "jump" and (m.direction, m.dst) in offs
                                     and model.platform(m.dst).y_at(m.land_x) > a.y_at(m.x) + NEEDLESS_DROP)]


def _drop_moves(model: MapModel, a: Platform) -> list[Move]:
    """Dropping through a (down + jump) onto each platform underneath, from the middle of where the two
    overlap (LAND_MARGIN in from both ends), if that's where she lands. Taken from the overlap with any
    platform, the first that landed her on one won: 27 -> 36 dropped from the middle of 27's overlap with 2
    (above it), 6 units from the basin's cliff at 36's left end, and 33 -> 36 from its overlap with 1, 6
    from the cliff at its right end. In the air there, the walls of 36's group don't stop her (body.
    OWN_GROUP_WALLS), and a touch's knock carried her through them and out of the map (harness, tour seed
    25 and loop seed 19). Where something else is under the middle (28 under 27's overlap with 36), there's
    no drop: the stretch between is no wider than her slop (from 27, 12 units at that cliff). Where the game
    won't drop her through a (forbidFallDown), or a knock could take her past a cliff of b (KNOCK_CARRY), she
    drops from the nearest spot to the middle clear of both."""
    out: list[Move] = []
    for b in model.platforms:
        if b is a:
            continue
        x = _overlap_middle(a, b)
        if x is None:
            continue
        cliffs = void_ends(model, b)
        if not (a.drop_ok(x, DROP_ROOM) and _clear_of(cliffs, x)):
            lo, hi = max(a.left, b.left) + LAND_MARGIN, min(a.right, b.right) - LAND_MARGIN
            ok = [s for s in range(math.ceil(lo), math.floor(hi) + 1, DROP_STEP)
                  if a.drop_ok(s, DROP_ROOM) and _clear_of(cliffs, s)]
            if not ok:
                continue
            x = float(min(ok, key=lambda s: abs(s - x)))  # the nearest spot where it will
        land, land_x, t = fly(model, x, a.y_at(x) + 1, 0.0, 0.0, skip=a, group=a.group)
        if land is b:
            out.append(Move("drop", a.id, b.id, x, land_x, seconds=DROP_EXTRA_S + t))
    return out


def rope_tops(model: MapModel, a: Platform) -> list[float]:
    """The xs of the ropes hanging from platform a (their tops within 15 units of it)."""
    return [r.x for r in model.ropes if abs(r.top - a.y_at(r.x)) < 15 and a.covers(r.x)]


def drop_clear(model: MapModel, a: Platform, x: float, below: Platform | None = None) -> bool:
    """Whether down + jump at x drops her through platform a: not within ROPE_CLEAR of a rope's top (down
    there climbs onto it), nor on a foothold that forbids it; and onto `below` (if given), not within
    KNOCK_CARRY of a cliff of it (void_ends)."""
    return a.drop_ok(x) and all(abs(x - rx) > ROPE_CLEAR for rx in rope_tops(model, a)) \
        and (below is None or _clear_of(void_ends(model, below), x))


def drop_spots(model: MapModel, a: Platform, below: Platform) -> list[int]:
    """Where on a she can drop through it onto `below`: every DROP_STEP units where they overlap (LAND_MARGIN
    in from the ends), not within ROPE_CLEAR of a rope's top, nor within DROP_ROOM of a foothold that
    forbids dropping, nor within KNOCK_CARRY of a cliff of `below` (void_ends)."""
    lo, hi = max(a.left, below.left) + LAND_MARGIN, min(a.right, below.right) - LAND_MARGIN
    ropes, cliffs = rope_tops(model, a), void_ends(model, below)
    return [x for x in range(int(lo), int(hi) + 1, DROP_STEP)
            if all(abs(x - rx) > ROPE_CLEAR for rx in ropes) and a.drop_ok(x, DROP_ROOM) and _clear_of(cliffs, x)]


def void_ends(model: MapModel, q: Platform) -> list[float]:
    """The ends of platform q past which nothing is under her, inside the map's sides (which stop her): past
    one, a fall goes below the map's lowest floor. On this map 36's two, the basin's cliffs."""
    lo, hi = min(p.left for p in model.platforms), max(p.right for p in model.platforms)
    return [end for end, out in ((q.left, -1), (q.right, 1))
            if lo <= end + out <= hi and model.platform_below(end + out, q.y_at(end)) is None]


def _clear_of(cliffs: list[float], x: float) -> bool:
    """Whether landing at x keeps KNOCK_CARRY from each of `cliffs` (void_ends)."""
    return all(abs(x - c) >= KNOCK_CARRY for c in cliffs)


def _overlap_middle(a: Platform, b: Platform) -> float | None:
    """The middle of where a and b overlap, LAND_MARGIN in from both ends (None: they overlap less)."""
    lo, hi = max(a.left, b.left) + LAND_MARGIN, min(a.right, b.right) - LAND_MARGIN
    return None if lo > hi else (lo + hi) / 2


def _sure_running_jump(model: MapModel, a: Platform, x: float, sign: int) -> tuple[Platform, float, float] | None:
    """Where a running jump leaving a at x (going that way) lands, if it surely lands on another platform
    with room to spare: (platform, landing x, seconds in the air)."""
    land, land_x, t = fly(model, x, a.y_at(x), sign * WALK_SPEED, -JUMP_SPEED, group=a.group)
    if land is not None and land is not a and land.covers(land_x - sign * LANDING_ROOM) \
            and land.covers(land_x + sign * LANDING_ROOM) \
            and sure_landing(model, x, a.y_at(x), sign * WALK_SPEED, -JUMP_SPEED, land, group=a.group):
        return land, land_x, t
    return None


def _room(model: MapModel, a: Platform, x: float, sign: int, land: Platform) -> float | None:
    """How far in from the nearer end of `land` a running jump leaving a at x (going that way) lands on it;
    None if it lands elsewhere, or nowhere."""
    got, lx, _ = fly(model, x, a.y_at(x), sign * WALK_SPEED, -JUMP_SPEED, group=a.group)
    return min(lx - land.left, land.right - lx) if got is land else None


# A takeoff's spread: how much earlier (further back from the end she jumps off) than planned she may leave
# the ground, flow's spread and her slop together (_placed); and her slop alone, either way.
_SPREAD = (-TAKEOFF_SLOP, 0.0, TAKEOFF_SPREAD, TAKEOFF_SPREAD + TAKEOFF_SLOP)
_SLOP = (-TAKEOFF_SLOP, 0.0, TAKEOFF_SLOP)


def _placed(model: MapModel, a: Platform, x0: float, sign: int, land: Platform) -> Move | None:
    """The running jump off a onto `land` found taking off at x0 (going that way), taking off where it lands
    with the most room (TAKEOFF_*, above): within TAKEOFF_SEARCH of x0 and TAKEOFF_EDGE in from the end she
    jumps off at least, where she lands TAKEOFF_ROOM_WANTED in whether she leaves the ground on time,
    TAKEOFF_SLOP late or up to TAKEOFF_SPREAD + TAKEOFF_SLOP early (or else the most in under the worst of
    those); failing that, under her slop alone. Of the takeoffs as good, the nearest x0. It has to be a sure
    running jump there (_sure_running_jump). None if no takeoff is."""
    edge = a.right if sign > 0 else a.left
    xs = [x for x in (x0 + sign * i for i in range(-TAKEOFF_SEARCH, TAKEOFF_SEARCH + 1, 2))
          if a.covers(x) and (edge - x) * sign >= TAKEOFF_EDGE]
    known: dict[float, float | None] = {}

    def room(x: float) -> float | None:
        if x not in known:
            known[x] = _room(model, a, x, sign, land)
        return known[x]

    def worst(x: float, offsets) -> float | None:
        # (a takeoff behind the platform's other end can't happen: not counted)
        rooms = [room(x - sign * o) for o in offsets if a.covers(x - sign * o)]
        return None if not rooms or None in rooms else min(min(rooms), TAKEOFF_ROOM_WANTED)

    for offsets in (_SPREAD, _SLOP):
        rated = [(r, x) for x in xs if (r := worst(x, offsets)) is not None]
        for _, x in sorted(rated, key=lambda rx: (-rx[0], abs(rx[1] - x0))):
            jump = _sure_running_jump(model, a, x, sign)
            if jump is not None and jump[0] is land:
                return Move("jump", a.id, land.id, a.clamp(x - sign * RUN_UP), jump[1],
                            directions.toward(sign), seconds=_running_jump_seconds(jump[2]), takeoff=x)
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
        self._times: dict[tuple[int, int], dict[int, float]] = {}  # travel's, by where from (TIMES_KEPT)
        self._landings: dict[tuple[int, int], dict[int, list[tuple[int, float]]]] = {}  # arrive's (TIMES_KEPT)

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
        return self.times(src, x).get(dst)

    def times(self, src: int, x: float) -> dict[int, float]:
        """Seconds the quickest route from (platform src, x) to each platform it can get to takes: one
        search to every platform (the same numbers as a search for each, which stops at its platform),
        kept for the TIMES_KEPT starts asked for last. A search starts from x rounded (`_search`), so
        that's the start. The hunt loop asks for the route time to each hunting ground after most looks,
        from where she stands: 0.7 ms a search, against 1.6 for one to every platform."""
        start = (src, round(x))
        found = self._times.pop(start, None)
        if found is None:
            found = self._every(*start)
            if len(self._times) >= TIMES_KEPT:
                del self._times[next(iter(self._times))]  # the one asked for longest ago
        self._times[start] = found
        return found

    def arrive(self, src: int, x: float, dst: int, dst_x: float | None = None) -> float | None:
        """Seconds from (platform src, x) to standing at map x `dst_x` on platform dst (None: anywhere on it, as
        `travel`): the quickest of the routes landing anywhere on dst, with the walk along it from where each
        lands (`travel` ends wherever she lands). None if there's no way there. The searches are kept as
        `times`' are (by where from, rounded): a pile's from its place are asked for again and again (the hunt
        loop's piles on the way: hunt._detour), and one to every platform has about 210 landings on this map."""
        if dst_x is None:
            return self.travel(src, x, dst)
        start = (src, round(x))
        found = self._landings.pop(start, None)
        if found is None:
            found = self._landed(*start)
            if len(self._landings) >= TIMES_KEPT:
                del self._landings[next(iter(self._landings))]
        self._landings[start] = found
        at = found.get(dst)
        return None if not at else min(t + abs(px - dst_x) / WALK_SPEED for px, t in at)

    def _landed(self, src: int, px: int) -> dict[int, list[tuple[int, float]]]:
        """`_every`'s search, keeping every landing: each platform's [(x landed at, cost)], the start among
        them (where she stands, at no cost)."""
        best = {(src, px): 0.0}
        heap = [(0.0, (src, px))]
        while heap:
            t, (pid, at) = heapq.heappop(heap)
            if t > best.get((pid, at), math.inf):
                continue
            for m in self.by_src.get(pid, []):
                cost = t + abs(m.x - at) / WALK_SPEED + m.seconds + RISK.get(m.kind, 0.0)
                node = (m.dst, round(m.land_x))
                if cost < best.get(node, math.inf):
                    best[node] = cost
                    heapq.heappush(heap, (cost, node))
        out: dict[int, list[tuple[int, float]]] = {}
        for (pid, at), t in best.items():
            out.setdefault(pid, []).append((at, t))
        return out

    def _every(self, src: int, px: int) -> dict[int, float]:
        """`_search` without a goal, to the end: each platform's cost, as `_search` would return it for
        that platform (the first time a node on it comes off the heap). No penalty, no weights."""
        best = {(src, px): 0.0}
        heap = [(0.0, (src, px))]
        out: dict[int, float] = {}
        while heap:
            t, (pid, px) = heapq.heappop(heap)
            if t > best.get((pid, px), math.inf):
                continue
            if pid not in out:
                out[pid] = t
            for m in self.by_src.get(pid, []):
                step = abs(m.x - px) / WALK_SPEED + m.seconds + RISK.get(m.kind, 0.0)
                cost = t + step
                node = (m.dst, round(m.land_x))
                if cost < best.get(node, math.inf):
                    best[node] = cost
                    heapq.heappush(heap, (cost, node))
        return out

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
