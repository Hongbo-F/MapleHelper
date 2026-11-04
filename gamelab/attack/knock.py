"""Where a touch's knock puts her, and what that costs her: the planner's knock term (the arrival study's prerequisite A,
D:/MapleHelper-scratch/arrival/REPORT.md). The user, 2026-11-03: "Motion planning should consider the current game
situation and dynamically calculate the best action to perform"; a candidate action's predicted touch is priced by
where its knock lands her (`landing`) and what getting back costs (`cost`), not by a margin kept from an end.

`landing(model, platform, x, phase, side, vx)`: the knock flown as the simulator's world flies it (sim.world:
World._hit_by, _launch, _land, _ground), from her feet at map x on `platform`:
- away from the monster (`side`: +1 right, -1 left) and up at KNOCK_VX and KNOCK_VY (sim.world's, the client's push as
  R1-R3's frames have it), the push sideways added to her speed `vx` but not past it (KNOCK_ADDS, `pushed`): standing
  or going away she flies off at 270, going at the monster at her walk at 270 less it;
- from the ground (`phase` GROUND) or in mid-jump from her height over her floor (`phase` a number, map units: the
  world resets her upward speed to the push, so only the height matters, rising or falling): the thief's jump throws;
- through the air by the engine (motion.physics.flight: body.fly's steps, in C, kept out of the planner's cache), no
  arrow (a knock's flight isn't steered: KNOCK_STEERS), the game's walls (body.wall_table with group 0's: the world's
  zero_walls) and the map's sides (its platforms' outer ends: the world's 'platforms' sides);
- landing at her walk's speed at most (KNOCK_LANDS_WALKING), then braked a game step at a time (body.ground_step, no
  arrow) to a stop, off an end that has no wall and down to what's under it (the same again), or stopped where she
  lands while an attack plays (`slides` False: World._land).
It answers (the platform she comes to rest on, None out of the map; her x there; how far below her floor at x, map
units, y down); `flown` the whole answer (Flown: its seconds too). The world's own knocks, staged, come to rest there
(tests/test_knock.py: the same platform, x within 0.1). Each platform's answers are tabled for its map and her shoe the
first time they're asked for: every GRID units from its left end to its right, each side, GROUND and AIR's heights,
standing and at her walk at the monster (TOWARD), a row of ~45 flights at a time (2-3 ms in C; `prepare` does a map's
all, ~2.5 s for Tree Dungeon); a lookup interpolates between the grid points, heights and speeds either side where both
come to rest on the same platform, else takes the nearer (~3 us, 5-10 us between heights or speeds). `fly` is the same
worked out (~30 us), and `from_state` a knock from any state of hers the table doesn't hold: mid-move (a route's jump,
drop or fall, past her platform's end), or faster than her walk.

`cost(model, planner, platform, x, flown, rate, touch_exp, ...)`: the price, in EXP, of a knock landing her at `flown`
from (platform, x) now: FREE where she stays on her platform (the flight and the walk back along it are the touch's,
whatever the knock); else the seconds lost (the flight and fall, the route planner's way back to where she was,
route.Planner.arrive, and HANG_S under each rope top on that way with monsters at it, `busy`) at her EXP a second,
`rate`, and the touches expected on the way back (TOP_TOUCH at each busy top, `touch_rate` a second elsewhere) at
`touch_exp` each (a touch's HP at the user's prices: behavior.hunt.bar_exp's share). KNOCK_LOST_S stands for a way
back the planner hasn't.

The stack's three other knock models disagree with this one (README "Her knock landings: flown, tabled and priced"):
the navigator's flat KNOCK_REACH (87 units on any slope: control.navigator._knock_cost), stance's end_drop (what's
under 5 units past an end, END_PAST) with margins of 25, 50 and 100 from the ends. Live, her standing knocks come to
rest where this puts them; her knocks walking at the monster fly shorter than the world's (open, the same section).
Pure: the map, the physics, no clock, no keys. Not wired into decisions yet (the planner's builder will).
"""

import bisect
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..localization.mapdata import MapModel, Platform
from ..motion import body, physics

# The world's knock (sim.world.KNOCK_VX, KNOCK_VY, KNOCK_ADDS, KNOCK_LANDS_WALKING; tests/test_knock.py pins them):
# away from the monster and up at 270 units/s each, the push sideways added to the speed she has, not past it.
KNOCK_VX = 270.0
KNOCK_VY = 270.0
GRID = 5.0                 # map units between the tabled x of a platform (its left end, then every GRID, its right end)
# Heights (map units over her floor at x) of the mid-jump knocks tabled: a jump from standing peaks at
# physics.JUMP_HEIGHT (77). The world resets her upward speed to KNOCK_VY whatever it was, so a height is the whole of a
# jump's phase there (rising or falling alike). Every 10 units: between two, where she comes to rest on the same
# platform, interpolated (README "Her knock landings": 97% of random mid-jump knocks on the table's platform).
AIR = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 77.0)
GROUND = "ground"
PHASES = (GROUND,) + AIR
# Her speed at the monster as it touches her (`vx`): the push adds to it (KNOCK_ADDS), so going at it at her walk she
# flies off at 270 less her walk (~120), standing or going away at 270. Tabled at both, interpolated between (the
# launch is linear in it); faster than her walk (off a rope jump) as at her walk (from_state flies it as it is).
TOWARD = (0.0, 1.0)  # (standing or away, at her walk toward it): her speed at it as a share of her walk
LIMIT = 4.0                # s a flight is flown before she's taken to be out of the map (the planner's flights')
# Her way back's prices (`cost`). HANG_S is control.navigator's (the hang under a rope's top with a monster at it, the
# user's upper limit: tests/test_knock.py pins it), taken whole where a monster stands at the top: on the baseline's
# -80 climbs (68344b2, 48 seeds; D:/MapleHelper-scratch/planner/knock/tops.py) a monster within TOP_REACH of the top as
# she got there made the climb longer at the median by 1.8 s for her (3.40 s from the grab against 1.61) and 0.9 for the
# thief (3.40 against 2.52), and touched her on the rope or within 1 s at the top in 96% (151 of 157) and him in 97%
# (168 of 173), against 13% and 12% with none. KNOCK_LOST_S is the navigator's bound for a knock with no way back.
HANG_S = 2.0
TOP_TOUCH = 0.95
TOP_REACH = 60.0
KNOCK_LOST_S = 30.0


@dataclass(frozen=True)
class Flown:
    """Where a knock leaves her: the platform she comes to rest on (None: out of the map), her x on it, how far below
    her floor where it took her she is (map units, y down; inf out of the map), s from the touch to then, and whether
    she left her platform by sliding off its end after landing on it (`slid`)."""
    platform: Platform | None
    x: float
    fall: float
    seconds: float
    slid: bool = False

    def off(self, model: MapModel, platform: Platform) -> bool:
        """Whether this knock takes her off `platform` (another place of the map, or out of it)."""
        return self.platform is None or model.place_of(self.platform.id) != model.place_of(platform.id)


@dataclass(frozen=True)
class Price:
    """What a knock landing costs her (`cost`): EXP in all, the seconds lost (flight and fall, the way back, hangs),
    the touches expected on the way back, the planner's way back (s; None: none, KNOCK_LOST_S taken), the hangs' s, and
    how many busy rope tops that way climbs to."""
    exp: float
    seconds: float = 0.0
    touches: float = 0.0
    back: float | None = 0.0
    hang: float = 0.0
    tops: int = 0


FREE = Price(0.0)


def sides(model: MapModel) -> tuple[float, float]:
    """The map's sides as the world has them by default (sim.world._sides, 'platforms'): its platforms' outer ends."""
    return min(q.left for q in model.platforms), max(q.right for q in model.platforms)


def _blocked(model: MapModel, p: Platform, way: int) -> bool:
    """Whether a wall stops her sliding off `p`'s end `way` (-1 left, +1 right): the world's _wall_at (the game's rule,
    group 0's walls too)."""
    edge = p.left if way < 0 else p.right
    return body.stops(body.wall_table(model, True), edge, p.y_at(edge) - 2, way)


def _flight(model: MapModel, p: Platform, x: float, y: float, vx: float, vy: float) -> tuple:
    """The world's flight from (x, y) at (vx, vy), no arrow: (platform or None, landing x, s, sideways speed then). In
    C by physics.flight (the game's walls, with group 0's: the world's zero_walls), kept out of the planner's cache; the
    map's sides stop her too, which only a flight ending past them can tell (no arrow, her sideways speed never turns),
    flown again by body.fly with them."""
    q, lx, t, vl = physics.flight(model, x, y, vx, vy, group=p.group, arrow=0, limit=LIMIT, cache=False)
    lo, hi = sides(model)
    if not lo <= lx <= hi:
        f = body.fly(model, x, y, vx, vy, shoe=physics.SHOE, arrow=0, walls=body.wall_table(model, True),
                     sides=(lo, hi), dt=body.TICK, limit=LIMIT)
        q, lx, t, vl = f.platform, f.land_x, f.seconds, f.vx_land
    return q, lx, t, vl


def pushed(vx: float, side: int) -> float:
    """Her speed sideways as a knock `side` launches her from speed vx: the push added to it, not past the push
    (KNOCK_ADDS)."""
    push = side * KNOCK_VX
    return min(vx + push, push) if side > 0 else max(vx + push, push)


def _start(p: Platform, x: float, phase, side: int, toward: float, shoe: body.Shoe) -> tuple[float, float]:
    """Her feet's y and her speed sideways as the knock `side` launches her from x on `p` in `phase`, going at the
    monster at `toward` of her walk."""
    floor = p.y_at(x)
    y = floor if phase == GROUND else floor - float(phase)
    return y, pushed(-side * toward * shoe.vmax, side)


def fly(model: MapModel, platform: Platform, x: float, phase=GROUND, side: int = 1, vx: float = 0.0,
        slides: bool = True, shoe: body.Shoe | None = None) -> Flown:
    """Where a touch's knock from her feet at map x on `platform` leaves her, worked out (not tabled; ~30 us):
    `landing`'s, all of it. `phase`: GROUND or her height over her floor in mid-jump (map units); `side` the knock's
    way, away from the monster (+1 right, -1 left); `vx` her speed sideways then (units/s, signed: going at the monster
    the push adds to it); `slides` False while an attack plays as she lands (it stops her there); `shoe` her walk's
    (physics.SHOE)."""
    shoe = shoe or physics.SHOE
    x = platform.clamp(x)
    floor = platform.y_at(x)
    y = floor if phase == GROUND else floor - float(phase)
    return _launched(model, platform, x, y, pushed(vx, side), floor, shoe)[0 if slides else 1]


def from_state(model: MapModel, platform: Platform, x: float, y: float, vx: float, side: int, slides: bool = True,
               shoe: body.Shoe | None = None) -> Flown:
    """A knock `side` from any state of hers, worked out (~40 us): her feet at map (x, y) (on the ground or anywhere in
    the air: a route's jump, drop or fall, past her platform's end), going sideways at vx; `platform` the one she last
    stood on (its walls' group, and what `fall` is measured from: its floor under her, as the world's knock_log). For
    the knocks the table doesn't hold: mid-move, or in mid-jump with a speed sideways."""
    shoe = shoe or physics.SHOE
    return _launched(model, platform, x, y, pushed(vx, side), platform.y_at(platform.clamp(x)), shoe)[
        0 if slides else 1]


def _both(model: MapModel, platform: Platform, x: float, phase, side: int, toward: float,
          shoe: body.Shoe) -> tuple[Flown, Flown]:
    """The table's answers at x, sliding and not (an attack playing as she lands), from the one flight."""
    x = platform.clamp(x)
    y, vx = _start(platform, x, phase, side, toward, shoe)
    return _launched(model, platform, x, y, vx, platform.y_at(x), shoe)


def _launched(model: MapModel, platform: Platform, x: float, y: float, vx: float, floor: float,
              shoe: body.Shoe) -> tuple[Flown, Flown]:
    """The knock launched from (x, y) at vx sideways and KNOCK_VY up, to rest (sliding, and stopped where she lands)."""
    q, lx, t, v = _flight(model, platform, x, y, vx, -KNOCK_VY)
    if q is None:
        out = Flown(None, lx, math.inf, t)
        return out, out
    still = Flown(q, lx, q.y_at(lx) - floor, t)
    v = min(max(v, -shoe.vmax), shoe.vmax)  # (KNOCK_LANDS_WALKING)
    step, ticks = body.STEP, round(body.STEP / body.TICK)
    t += body.landing_wait(step) if v else 0.0  # (she stands where she lands until the game's next step)
    slid = False
    while v:
        v, dx = body.ground_step(v, 0, shoe, step)
        t += step
        if q.left <= lx + dx <= q.right:  # (the world moves her a tick at a time: off an end, from the tick past it)
            lx += dx
            continue
        for _ in range(ticks):
            nx = lx + dx / ticks
            if q.left <= nx <= q.right:
                lx = nx
                continue
            way = 1 if nx > q.right else -1
            edge = q.right if way > 0 else q.left
            if _blocked(model, q, way):  # against the wall at that end
                lx, v = edge, 0.0
                break
            q2, lx2, t2, v2 = _flight(model, q, nx, q.y_at(edge) - 1, v, 0.0)  # off the end: down to what's under it
            t += t2
            slid = True
            if q2 is None:
                return Flown(None, lx2, math.inf, t, True), still
            q, lx, v = q2, lx2, v2
            break
    return Flown(q, lx, q.y_at(lx) - floor, t, slid), still


class _Table:
    """One platform's knocks: for each phase, side and speed at the monster (TOWARD's), (slid, still) Flown at its
    grid's x, each row worked out the first time it's asked for (~45 flights: 2-3 ms in C)."""

    def __init__(self, model: MapModel, platform: Platform, shoe: body.Shoe):
        self.model, self.platform, self.shoe = model, platform, shoe
        self.left, self.right = platform.left, platform.right
        n = int(math.floor((self.right - self.left) / GRID))
        xs = [self.left + i * GRID for i in range(n + 1)]
        if xs[-1] < self.right:
            xs.append(self.right)
        self.xs = xs
        self.rows: dict[tuple, list[tuple[Flown, Flown]]] = {}

    def row(self, phase, side: int, toward: float) -> list[tuple[Flown, Flown]]:
        key = (phase, side, toward)
        got = self.rows.get(key)
        if got is None:
            got = self.rows[key] = [_both(self.model, self.platform, x, phase, side, toward, self.shoe)
                                    for x in self.xs]
        return got

    def every_row(self) -> None:
        for phase in PHASES:
            for side in (-1, 1):
                for toward in TOWARD:
                    self.row(phase, side, toward)

    def at(self, x: float, phase, side: int, toward: float, slides: bool) -> Flown:
        row = self.row(phase, side, toward)
        k = 0 if slides else 1
        if len(self.xs) == 1:
            return row[0][k]
        x = min(max(x, self.left), self.right)
        i = min(max(bisect.bisect_right(self.xs, x) - 1, 0), len(self.xs) - 2)
        x0, x1 = self.xs[i], self.xs[i + 1]
        return _between(row[i][k], row[i + 1][k], (x - x0) / (x1 - x0))


def _between(a: Flown, b: Flown, u: float) -> Flown:
    """a and b's answer at u of the way from a to b (0..1): interpolated where both come to rest on the same platform
    the same way (both slid off an end to it, or neither), else the nearer's (out of the map too)."""
    if a.platform is b.platform and a.platform is not None and a.slid == b.slid:
        return Flown(a.platform, a.x + (b.x - a.x) * u, a.fall + (b.fall - a.fall) * u,
                     a.seconds + (b.seconds - a.seconds) * u, a.slid)
    return a if u <= 0.5 else b


_TABLES: dict[tuple, tuple] = {}  # (id(model), platform id, shoe) -> (the model, its _Table): as stance's _per_platform


def table(model: MapModel, platform: Platform, shoe: body.Shoe | None = None) -> _Table:
    """`platform`'s knocks for `model` and her `shoe` (physics.SHOE), worked out once (the model kept so its id can't
    go to another one while it's here; cleared past 4096)."""
    shoe = shoe or physics.SHOE
    key = (id(model), platform.id, shoe)
    got = _TABLES.get(key)
    if got is None or got[0] is not model:
        if len(_TABLES) > 4096:
            _TABLES.clear()
        got = _TABLES[key] = (model, _Table(model, platform, shoe))
    return got[1]


def prepare(model: MapModel, shoe: body.Shoe | None = None) -> None:
    """Every place's knocks worked out now (a map's load), not at their first lookups."""
    for p in model.places:
        table(model, p, shoe).every_row()


def _at_height(t: _Table, x: float, phase, side: int, toward: float, slides: bool) -> Flown:
    """The table's answer at a height between AIR's (or on the ground): interpolated as _between."""
    if phase == GROUND:
        return t.at(x, GROUND, side, toward, slides)
    h = float(phase)
    if h <= 0:
        return t.at(x, GROUND, side, toward, slides)
    if h >= AIR[-1]:
        return t.at(x, AIR[-1], side, toward, slides)
    j = bisect.bisect_left(AIR, h)
    if AIR[j] == h:
        return t.at(x, AIR[j], side, toward, slides)
    lo_h, lo = (0.0, GROUND) if j == 0 else (AIR[j - 1], AIR[j - 1])
    return _between(t.at(x, lo, side, toward, slides), t.at(x, AIR[j], side, toward, slides),
                    (h - lo_h) / (AIR[j] - lo_h))


def flown(model: MapModel, platform: Platform, x: float, phase=GROUND, side: int = 1, vx: float = 0.0,
          slides: bool = True, shoe: body.Shoe | None = None) -> Flown:
    """`landing`'s whole answer (Flown: its seconds and how it left her platform too), from the table (`fly`'s at every
    GRID units of x, AIR's heights and TOWARD's speeds, interpolated between)."""
    shoe = shoe or physics.SHOE
    t = table(model, platform, shoe)
    side = 1 if side > 0 else -1
    u = min(max(-vx * side / shoe.vmax, 0.0), 1.0)  # her speed at the monster, a share of her walk
    if u <= 0.0 or u >= 1.0:
        return _at_height(t, x, phase, side, float(u >= 1.0), slides)
    return _between(_at_height(t, x, phase, side, 0.0, slides), _at_height(t, x, phase, side, 1.0, slides), u)


def landing(model: MapModel, platform: Platform, x: float, phase=GROUND, side: int = 1, vx: float = 0.0,
            slides: bool = True) -> tuple[Platform | None, float, float]:
    """Where a touch's knock from her feet at map x on `platform` puts her: (the platform she comes to rest on, None
    out of the map; her x; how far below her floor at x, map units, y down). `phase` GROUND or her height over her
    floor in mid-jump (map units), `side` the knock's way (+1 right, -1 left: away from the monster), `vx` her speed
    sideways as it touches her (signed units/s: going at it, the push adds to it), `slides` False while an attack
    plays as she lands. From the per-platform table (`flown`); `from_state` for a knock off her platform's span (a
    route's move) or faster than her walk."""
    f = flown(model, platform, x, phase, side, vx, slides)
    return f.platform, f.x, f.fall


def zone(model: MapModel, platform: Platform, end: int, phase=GROUND, vx: float = 0.0, step: float = 1.0,
         shoe: body.Shoe | None = None) -> float:
    """How far in from `platform`'s `end` (-1 left, +1 right) a knock toward that end still takes her off it (`fly`'s,
    worked out every `step` units over 150 from the end): the knock-off zone's width; 0 where none does (a wall)."""
    edge = platform.left if end < 0 else platform.right
    widest, d = 0.0, 0.0
    while d <= min(150.0, platform.right - platform.left):
        if fly(model, platform, edge - end * d, phase, end, vx, True, shoe).off(model, platform):
            widest = d
        d += step
    return widest


def busy_tops(model: MapModel, monsters: Iterable, reach: float = TOP_REACH) -> Callable[[object], bool]:
    """`cost`'s `busy` from the monsters seen (each with .platform and .x): a climb (route.Move) is busy where one
    stands on the platform at its top within `reach` of where she climbs over."""
    tops = [(model.place_of(m.platform), m.x) for m in monsters if getattr(m, "platform", None) is not None]

    def busy(move) -> bool:
        dst = model.place_of(move.dst)
        return any(pid == dst and abs(mx - move.land_x) <= reach for pid, mx in tops)
    return busy


def cost(model: MapModel, planner, platform: Platform, x: float, landed: Flown, rate: float, touch_exp: float, *,
         busy: Callable[[object], bool] | None = None, hang_s: float = HANG_S, top_touch: float = TOP_TOUCH,
         touch_rate: float = 0.0, lost_s: float = KNOCK_LOST_S) -> Price:
    """What a knock that leaves her at `landed` (`flown`'s) from map x on `platform` costs her now, in EXP: FREE where
    she stays on her platform; else her `rate` (EXP a second) over the seconds it loses her (its flight and fall,
    `planner.arrive`'s way back to x, `hang_s` under each rope top on that way `busy` says has monsters at it) and
    `touch_exp` (EXP a touch's HP costs her) over the touches expected on the way (`top_touch` at each busy top,
    `touch_rate` a second of the rest). `lost_s` stands for the way back where the planner has none (out of the map
    too)."""
    if not landed.off(model, platform):
        return FREE
    if landed.platform is None:
        seconds = landed.seconds + lost_s
        touches = touch_rate * lost_s
        return Price(seconds * rate + touches * touch_exp, seconds, touches, None, 0.0, 0)
    back = planner.arrive(landed.platform.id, landed.x, platform.id, x)
    tops = 0
    if busy is not None and back is not None:
        way = planner.plan(landed.platform.id, landed.x, platform.id) or []
        tops = sum(1 for m in way if m.kind == "climb" and busy(m))
    hang = hang_s * tops
    walk = lost_s if back is None else back
    seconds = landed.seconds + walk + hang
    touches = top_touch * tops + touch_rate * walk
    return Price(seconds * rate + touches * touch_exp, seconds, touches, back, hang, tops)
