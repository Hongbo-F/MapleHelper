"""Her body as the game moves it: the client's movement step by step (walking, braking, turning, the
jump and its kick, gravity and the fall cap, the little steering she has in the air), the lags between
a key and the game acting on it, and closed forms of it on flat ground for planning. Pure: no files,
no clock, no keys. Map units and seconds; x right, y down (as mapdata). `d` is the arrow the game
sees: -1 (left), 0 (none) or +1 (right).

Where it comes from:
- The constants are the game's own, from Physics.img (the map model's `physics`; PHYSICS_IMG is a copy
  a test checks against it): forces over the client's mass of 100 are accelerations.
- Her SPEED and JUMP stats (config.toml [character], read by motion.physics) scale her walk and jump:
  SPEED 119.2 walks her at 149 units/s, as measured (148-151 in the walks recorded).
- The step is a v83 client decompile's, as RustMS ported it. That is secondhand, and this client is
  newer, so it is used where the 60 fps recordings agree: speeding up at 1400 units/s² from standing
  (five starts), braking at 800 (seven stops), the kick at the jump (five jump-backs), braking at 200
  in the air with the opposite arrow (five holds). Turning on the ground, the fall-cap drag and the
  rope side-jump are the decompile's alone.
- The lags are the medians of the recordings (Lags), with their spread.

With these fixed lags the engine follows 18 tracked recordings to 2.49 units RMS in x and 1.58 in y,
where the model it replaces (a standing start that switches to full speed, an even slide, nothing in
the air) is off by 3.20 and 2.72 (`python -m gamelab.analysis.physics_check`).
"""

import bisect
import math
from dataclasses import dataclass, field

# Map.wz/Physics.img, as the game data has it (every map shares it).
PHYSICS_IMG = {
    "walkForce": 140000.0, "walkSpeed": 125.0, "walkDrag": 80000.0, "slipForce": 60000.0, "slipSpeed": 120.0,
    "floatDrag1": 100000.0, "floatDrag2": 10000.0, "floatCoefficient": 0.01, "swimForce": 120000.0,
    "swimSpeed": 140.0, "flyForce": 120000.0, "flySpeed": 200.0, "gravityAcc": 2000.0, "fallSpeed": 670.0,
    "jumpSpeed": 555.0, "maxFriction": 2.0, "minFriction": 0.05, "swimSpeedDec": 0.9,
    "flyJumpDec": 0.3499999940395355,
}
MASS = 100.0     # the client's mass for a character (decompile)
# The game's stat limits (decompile): SPEED counts from 80% to 140%, JUMP from 80% to 123%.
SPEED_RANGE = (80.0, 140.0)
JUMP_RANGE = (80.0, 123.0)
# s per step. The game's own step isn't known. It isn't the 16.7 ms frame: three jump-backs took off at
# 121-127 units/s, and at 1400 units/s² a 16.7 ms step only has 117 and 140 there. With lags fitted per
# recording, a 5 ms step follows the eight recordings with jumps to 0.96 / 1.00 units RMS (x / y), 8 ms
# to 1.14 / 1.03, 16.7 ms to 1.96 / 1.52. 5 ms is also the simulator's tick.
TICK = 0.005


@dataclass(frozen=True)
class Shoe:
    """How the game moves her: her limits and accelerations in units/s and units/s²."""
    vmax: float           # walking speed: walkSpeed × SPEED/100 (149)
    acc: float            # speeding up, or turning round, with an arrow on the ground: walkForce/mass (1400)
    drag: float           # braking with no arrow on the ground, or above vmax: walkDrag/mass (800)
    vjump: float          # the jump's upward speed: jumpSpeed × JUMP/100 (555: 77 units high)
    gravity: float        # gravityAcc (2000)
    fall: float           # the fall-speed cap: fallSpeed (670)
    air_acc: float        # an arrow in the air: 2 × floatDrag2/mass (200), toward ...
    air_speed: float      # ... at most this speed that way: walkSpeed/walkForce × floatDrag2 (8.9). So the
                          # arrow she runs with does nothing in the air, and the opposite one brakes her.
    air_drag: float       # no arrow in the air: floatDrag2/mass × floatCoefficient (1) ...
    air_drag_fall: float  # ... or floatDrag2/mass (100) once she falls at the cap
    kick: float = 0.8     # the jump: below kick × vmax toward the arrow, she gets kick × vmax more (capped at vmax)
    rope_vx: float = 1.3  # jumping off a rope sideways: rope_vx × vmax out ...
    rope_vy: float = 0.5  # ... and rope_vy × vjump up (decompile; the minimap's one rope jump agrees, weakly)

    @classmethod
    def from_physics(cls, physics: dict, speed: float, jump: float, fs: float = 1.0) -> "Shoe":
        """Her shoe from Physics.img's values, her SPEED and JUMP stats (%) and the map's friction
        `fs` (info/fs; 1 on every map in the game data: icy maps would have it below 1)."""
        speed = min(max(speed, SPEED_RANGE[0]), SPEED_RANGE[1])
        jump = min(max(jump, JUMP_RANGE[0]), JUMP_RANGE[1])
        friction = min(max(fs, physics["minFriction"]), physics["maxFriction"])
        friction = friction if friction >= 1 else friction / 2  # decompile; untested (no map has fs)
        float_drag = physics["floatDrag2"] / MASS
        return cls(vmax=physics["walkSpeed"] * speed / 100, acc=physics["walkForce"] * fs / MASS,
                   drag=physics["walkDrag"] * friction / MASS, vjump=physics["jumpSpeed"] * jump / 100,
                   gravity=physics["gravityAcc"], fall=physics["fallSpeed"], air_acc=2 * float_drag,
                   air_speed=physics["walkSpeed"] / physics["walkForce"] * physics["floatDrag2"],
                   air_drag=float_drag * physics["floatCoefficient"], air_drag_fall=float_drag)


@dataclass(frozen=True)
class Lags:
    """s from a key going out to the game acting on it: medians of the 60 and 30 fps recordings, and
    their spread (sd). Only their differences matter to a key plan (jump - on is +7 ms, off - on +12 ms);
    the game's frame, its drawing and the capture add to all of them alike."""
    on: float = 0.060     # an arrow pressed -> she starts to speed up (10 standing starts: 45-70 ms)
    off: float = 0.072    # an arrow let go -> she starts to brake (7 stops: 52-95 ms)
    jump: float = 0.067   # the jump key -> she leaves the ground (18 jumps: 45-95 ms, mean 65.5)
    drop: float = 0.28    # the jump key with down held -> she falls through (4 drops: 0.28-0.34 s)
    sd_on: float = 0.010
    sd_off: float = 0.011
    sd_jump: float = 0.012


# --- one step ---------------------------------------------------------------------------------------

def _accelerate(v: float, a: float, limit: float, dt: float) -> float:
    """v moved by a·dt toward ±limit (a's sign), only while it is short of it."""
    if a > 0:
        return min(v + a * dt, limit) if v < limit else v
    if a < 0:
        return max(v + a * dt, -limit) if v > -limit else v
    return v


def _decelerate(v: float, limit: float, drag: float, dt: float) -> float:
    """v moved by drag·dt back toward ±limit, only while |v| is above it."""
    if v > limit:
        return max(v - drag * dt, limit)
    if v < -limit:
        return min(v + drag * dt, -limit)
    return v


def ground_step(v: float, d: int, shoe: Shoe, dt: float = TICK) -> tuple[float, float]:
    """One step on flat ground: her new speed, and how far she moved (the mean of the two speeds over
    dt, as the client does). With an arrow she speeds up toward it at shoe.acc (turning round too, with
    no extra braking); with none she brakes at shoe.drag."""
    nv = _decelerate(v, shoe.vmax, shoe.drag, dt)  # faster than she walks (off a rope jump): back down
    nv = _decelerate(nv, 0.0, shoe.drag, dt) if d == 0 else _accelerate(nv, d * shoe.acc, shoe.vmax, dt)
    return nv, (v + nv) / 2 * dt


def air_step(vx: float, vy: float, d: int, shoe: Shoe, dt: float = TICK) -> tuple[float, float, float, float]:
    """One step in the air: her new speeds, and how far she moved (dx, dy; y down). Gravity up to the
    fall cap; an arrow steers at shoe.air_acc toward shoe.air_speed that way (so it only ever brakes a
    jump that way at speed); no arrow, the air drags her sideways a little (a lot at the fall cap)."""
    nvy = _decelerate(vy, shoe.fall, shoe.air_drag_fall, dt)  # faster than the cap: back down to it
    nvy = _accelerate(nvy, shoe.gravity, shoe.fall, dt)
    if d:
        nvx = _accelerate(vx, d * shoe.air_acc, shoe.air_speed, dt)
    else:
        nvx = _decelerate(vx, 0.0, shoe.air_drag if nvy < shoe.fall else shoe.air_drag_fall, dt)
    return nvx, nvy, (vx + nvx) / 2 * dt, (vy + nvy) / 2 * dt


def takeoff(v: float, d: int, shoe: Shoe) -> float:
    """Her sideways speed leaving the ground, from her ground speed v with arrow d seen as the jump is
    (the kick): slower than shoe.kick × vmax that way, she gets kick × vmax more, then at most vmax.
    So from standing she leaves at 119 (not 149); a jump just after the arrow (her speed 30-119) leaves
    at full speed; one just below full speed (119-149) gets no kick and leaves as she was going. In the
    jump-backs, keyed 80 ms after the arrow: 121, 124 and 127 with no kick, 142 with one (the arrow seen
    late: she had barely moved)."""
    if d:
        if d * v < shoe.kick * shoe.vmax:
            v += d * shoe.kick * shoe.vmax
        if d * v > shoe.vmax:
            v = d * shoe.vmax
    return v


# --- closed forms on flat ground ----------------------------------------------------------------------

def _run(u: float, tau: float, shoe: Shoe) -> tuple[float, float]:
    """(how far, speed) along an arrow held tau s as the game sees it, from speed u that way (negative:
    going the other way, so she brakes and turns first)."""
    vm, s = shoe.vmax, 0.0
    if u > vm:  # dragged back down to vmax
        t1 = (u - vm) / shoe.drag
        if tau <= t1:
            return u * tau - shoe.drag * tau * tau / 2, u - shoe.drag * tau
        s, tau, u = (u + vm) / 2 * t1, tau - t1, vm
    if u < vm:
        t1 = (vm - u) / shoe.acc
        if tau <= t1:
            return s + u * tau + shoe.acc * tau * tau / 2, u + shoe.acc * tau
        s, tau = s + (u + vm) / 2 * t1, tau - t1
    return s + vm * tau, vm


def run_x(tau: float, v0: float, d: int, shoe: Shoe) -> float:
    """How far she has moved (x, signed) tau s after the game sees arrow d held, from speed v0 (signed):
    from standing 700τ² up to full speed (0.106 s, 7.9 units on), then 149 a second. Running the other
    way, she brakes and turns at shoe.acc first (0.21 s from full speed to full speed back, going 7.9
    units on first)."""
    return d * _run(d * v0, max(0.0, tau), shoe)[0]


def run_speed(tau: float, v0: float, d: int, shoe: Shoe) -> float:
    """Her speed (signed) tau s after the game sees arrow d held, from speed v0 (signed): run_x's."""
    return d * _run(d * v0, max(0.0, tau), shoe)[1]


def run_time(dist: float, v0: float, d: int, shoe: Shoe) -> float | None:
    """When (s after the game sees arrow d held, from speed v0) she has gone `dist` that way, on her way
    there: turning round, the second time she passes it. None for a spot behind her that she never
    reaches while turning."""
    u, vm, a = d * v0, shoe.vmax, shoe.acc
    s = t = 0.0
    if u > vm:
        t1 = (u - vm) / shoe.drag
        s1 = (u + vm) / 2 * t1
        if dist <= s1:
            return (u - math.sqrt(max(0.0, u * u - 2 * shoe.drag * dist))) / shoe.drag
        s, t, u = s1, t1, vm
    if u < vm:
        t1 = (vm - u) / a
        s1 = (u + vm) / 2 * t1
        if dist - s <= s1:
            disc = u * u + 2 * a * (dist - s)
            return None if disc < 0 else t + (-u + math.sqrt(disc)) / a
        s, t = s + s1, t + t1
    return t + (dist - s) / vm


def slide(v: float, shoe: Shoe) -> float:
    """How far she slides braking from speed v with no arrow (landing, or once a release is seen):
    v²/2·drag, 13.9 units from full speed."""
    return v * v / (2 * shoe.drag)


def stop_distance(v: float, shoe: Shoe, lags: Lags) -> float:
    """How far she goes on after letting go of the arrow she's running with at speed v (|v|): she keeps
    going for the release's lag (speeding up still, below full speed), then slides. 24.6 units from full
    speed (seven 30 and 60 fps stops: 23.4-28.1)."""
    s, u = _run(abs(v), lags.off, shoe)
    return s + slide(u, shoe)


def stop_time(v: float, shoe: Shoe, lags: Lags) -> float:
    """s from letting go of the arrow at speed v until she stands still: 0.26 from full speed."""
    _, u = _run(abs(v), lags.off, shoe)
    return lags.off + u / shoe.drag


def hold_distance(ms: float, shoe: Shoe, lags: Lags) -> float:
    """How far an arrow held `ms` from standing moves her, the slide included. The game sees it held 12
    ms longer (off - on): a hold of t s moves her 1925 (t + 0.012)² units below about 0.095 s, and
    7.7 + 149 t above. Short taps are bimodal in game: of three 50 ms taps two moved her 0 and one 7.1
    (the model: 7.4)."""
    seen = ms / 1000 + lags.off - lags.on
    if seen <= 0:
        return 0.0
    s, u = _run(0.0, seen, shoe)
    return s + slide(u, shoe)


def hold_time(ms: float, shoe: Shoe, lags: Lags) -> float:
    """s from an arrow's key going down, held `ms` from standing, until she stands still again (the game
    sees it let go lags.off after its key, and she brakes from what she reached): 0.23 s for a 50 ms tap,
    which has her at 87 units/s when the game sees it let go."""
    _, u = _run(0.0, max(0.0, ms / 1000 + lags.off - lags.on), shoe)
    return ms / 1000 + lags.off + u / shoe.drag


def held_for(units: float, v0: float, shoe: Shoe) -> float:
    """How long (s) the game must see an arrow held, from speed v0 that way (0 to vmax), for her to come
    to rest `units` on once it sees it let go (0: she slides that far anyway). From standing, 0.106 s
    (full speed) has her rest 21.8 units on; each 1/149 s more, a unit more."""
    vm, a = shoe.vmax, shoe.acc
    u = min(max(v0, 0.0), vm)
    if units <= slide(u, shoe):
        return 0.0
    at_full = (vm * vm - u * u) / (2 * a) + slide(vm, shoe)  # held until just full speed, then let go
    if units > at_full:
        return (vm - u) / a + (units - at_full) / vm
    # Let go while speeding up: u τ + a τ²/2 of run, then a slide from u + a τ.
    qa, qb, qc = a / 2 + a * a / (2 * shoe.drag), u + u * a / shoe.drag, slide(u, shoe) - units
    return (-qb + math.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)


def hold_for(units: float, shoe: Shoe, lags: Lags) -> int:
    """How long (ms) to hold an arrow from standing to move `units` and stop there: hold_distance's
    inverse (39 ms for 5 units, 90 for 20, 284 for 50)."""
    if units <= 0:
        return 0
    return max(0, round((held_for(units, 0.0, shoe) - (lags.off - lags.on)) * 1000))


def _normal_below(x: float, mean: float, sd: float) -> float:
    return 0.5 * (1 + math.erf((x - mean) / (sd * math.sqrt(2)))) if sd > 0 else float(x > mean)


def short_takeoff_chance(gap_ms: float, shoe: Shoe, lags: Lags, slower: float = 0.94) -> float:
    """The chance that a jump keyed gap_ms after the arrow, from standing, leaves the ground slower than
    `slower` × vmax (140 of her 149), with the lags as spread as measured. The kick gives full speed
    while her ground speed is 0.2-0.8 × vmax: too early (the arrow not seen yet, or barely) she leaves
    at 119 or straight up; just below full speed (0.8-1 × vmax) she gets no kick. A flat jump at 121
    is about 15 units shorter than at 149."""
    mean = gap_ms / 1000 + lags.jump - lags.on  # how long the game has seen the arrow when it jumps her
    sd = math.hypot(lags.sd_jump, lags.sd_on)
    full = shoe.vmax / shoe.acc
    kicked = (slower - shoe.kick) * full  # below this the kick still leaves her short
    unkicked = shoe.kick * full           # from here no kick ...
    reached = slower * full               # ... and short until here
    p = _normal_below(kicked, mean, sd)
    return p + max(0.0, _normal_below(reached, mean, sd) - _normal_below(unkicked, mean, sd))


def jump_window_ok(gap_ms: float, shoe: Shoe, lags: Lags, risk: float = 0.05) -> bool:
    """Whether a jump keyed gap_ms after the arrow, from standing, surely leaves at full speed (at most
    `risk` that it's short): 35-52 ms (the kick), or 120 ms and later (full speed by then). The
    jump-back keys it 80 ms after the arrow: a 35% chance of leaving short, and three of the five
    recorded left at 121-127."""
    return short_takeoff_chance(gap_ms, shoe, lags) <= risk


# --- in the air over a map ----------------------------------------------------------------------------

Walls = tuple[list[float], list[tuple[float, float]]]  # vertical walls by x: their xs, each one's (top, bottom)


def vertical_walls(segments) -> Walls:
    """The vertical ones of a map's wall segments, sorted by x, as fly takes them: their xs, and each
    one's top and bottom (a unit beyond its ends)."""
    ws = sorted((w for w in segments if w.x1 == w.x2), key=lambda w: w.x1)
    return [w.x1 for w in ws], [(min(w.y1, w.y2) - 1, max(w.y1, w.y2) + 1) for w in ws]


@dataclass(frozen=True)
class Flight:
    """Where an arc through the air ends: the platform (None: none within the time), where, after how
    long, and her sideways speed as she lands (0 after a wall stopped her). `path`: her (x, y) after
    every step from the start, when asked for, and `speeds` her (vx, vy) then (a flight can be carried
    on from any step of it, with another arrow: the simulator does when the arrow changes mid-air)."""
    platform: object
    land_x: float
    seconds: float
    vx_land: float
    path: tuple[tuple[float, float], ...] = field(default=(), repr=False)
    speeds: tuple[tuple[float, float], ...] = field(default=(), repr=False)


def _lands_at(q, x: float, y: float, nx: float, ny: float) -> float | None:
    """Where a step down from (x, y) to (nx, ny) puts her feet on platform q (None: it doesn't): where it
    ends, if q is under that and the step went through its surface; or, if the step starts over q and ends
    past its end, where it went through q's surface before the end. Checked only where the step ends, a
    knock in the air took her through the basin's corner (36's right end, under the foot of the wall
    above it) and out of the map (harness, tour seed 10)."""
    if q.covers(nx):
        return nx if y <= q.y_at(nx) <= ny + 0.01 else None
    if nx == x or not q.covers(x):
        return None
    end = q.right if nx > x else q.left
    before = y - q.y_at(x)                                    # above the surface: below 0
    after = y + (ny - y) * (end - x) / (nx - x) - q.y_at(end)  # at the end: at or below it
    if before > 0 or after < -0.01:
        return None
    return x + (end - x) * (-before / (after - before) if after > before else 1.0)


def fly(model, x: float, y: float, vx: float, vy: float, *, shoe: Shoe, arrow: int = 0, skip=None,
        walls: Walls | None = None, sides: tuple[float, float] | None = None, dt: float = TICK,
        limit: float = 4.0, path: bool = False, airborne: bool = False) -> Flight:
    """Where she lands from (x, y) moving at (vx, vy) with `arrow` held all the way: the first platform
    whose surface her feet cross on the way down (_lands_at). Walls stop her sideways (`walls`: the
    model's, worked out each call if not given; `sides`: the map's left and right limits, which stop her
    too). `skip` is a platform she falls through (dropping). A flight leaving the ground can't land on
    its first step; one carried on from mid-air (`airborne`) can. Not cached."""
    wall_xs, wall_ys = walls if walls is not None else vertical_walls(model.walls)
    # Only platforms she could reach in the time: within her sideways travel, at or below the top of the jump.
    reach = max(abs(vx), shoe.air_speed if arrow else 0.0) * limit + 1
    top = y - (vy * vy / (2 * shoe.gravity) if vy < 0 else 0) - 1
    plats = [q for q in model.platforms if q is not skip and q.right >= x - reach and q.left <= x + reach
             and max(q.y_at(q.left), q.y_at(q.right)) >= top]
    points = [(x, y)] if path else None
    speeds = [(vx, vy)] if path else None
    t = 0.0
    while t < limit:
        nvx, nvy, dx, dy = air_step(vx, vy, arrow, shoe, dt)
        nx, ny = x + dx, y + dy
        if nx != x:
            lo, hi = (x, nx) if nx > x else (nx, x)
            i = bisect.bisect_left(wall_xs, lo)
            while i < len(wall_xs) and wall_xs[i] <= hi:
                if wall_ys[i][0] <= ny <= wall_ys[i][1]:
                    nx, nvx = x, 0.0  # bumped into a wall
                    break
                i += 1
            if sides is not None and not sides[0] <= nx <= sides[1]:
                nx, nvx = min(max(nx, sides[0]), sides[1]), 0.0
        if (vy > 0 or nvy > 0) and (t > 0 or airborne) and ny >= y:
            for q in plats:
                at = _lands_at(q, x, y, nx, ny)
                if at is not None:
                    if points is not None:
                        points.append((at, q.y_at(at)))
                        speeds.append((nvx, nvy))
                    return Flight(q, at, t + dt, nvx, tuple(points or ()), tuple(speeds or ()))
        x, y, vx, vy, t = nx, ny, nvx, nvy, t + dt
        if points is not None:
            points.append((x, y))
            speeds.append((vx, vy))
    return Flight(None, x, t, vx, tuple(points or ()), tuple(speeds or ()))


# --- a key timeline on flat ground ------------------------------------------------------------------

@dataclass
class Replay:
    """Her body under a key timeline on flat ground (replay): (t, x, y) after every step, and each
    lift-off's (t, her speed on the ground, her sideways speed leaving it)."""
    points: list[tuple[float, float, float]]
    takeoffs: list[tuple[float, float, float]]


def replay(events, until: float, shoe: Shoe, lags: Lags, *, start: float = -0.1, dt: float = TICK,
           floors=()) -> Replay:
    """Her body on flat ground under a key timeline, as the game moves it: from standing at x 0 on a floor
    at y 0 (y down: negative in the air), from `start` to `until` s. `events` are (t s, action, down) as
    the keys went out, the action "left", "right" or "jump" (others are ignored). The game sees an
    arrow's press lags.on later, its release lags.off later and a jump lags.jump later; of the arrows it
    sees held, the one pressed last counts. A jump it sees while she's in the air is ignored (and
    holding the key doesn't jump her again on landing). `floors`: (from t, y) pairs, the floor she lands
    on from then on (a recording that lands on another platform)."""
    seen = []
    for t, action, down in events:
        if action in ("left", "right"):
            seen.append((t + (lags.on if down else lags.off), action, down))
        elif action == "jump" and down:
            seen.append((t + lags.jump, "jump", True))
    seen.sort(key=lambda e: e[0])
    floors = sorted(floors)
    held: dict[str, int] = {}
    order = 0
    x = y = vx = vy = 0.0
    ground, floor = True, 0.0
    points, takeoffs = [], []
    j = 0
    for k in range(int((until - start) / dt + 1e-9) + 1):
        t = start + k * dt  # counted, not summed, so a step's time doesn't drift
        jump = False
        while j < len(seen) and seen[j][0] <= t + 1e-9:
            _, action, down = seen[j]
            if action == "jump":
                jump = True
            elif down:
                order += 1
                held[action] = order
            else:
                held.pop(action, None)
            j += 1
        d = 0
        if held:
            d = 1 if max(held, key=held.get) == "right" else -1
        for since, level in floors:
            if t >= since:
                floor = level
        if ground and jump:
            takeoffs.append((t, vx, takeoff(vx, d, shoe)))
            vx, vy, ground = takeoffs[-1][2], -shoe.vjump, False
        if ground:
            vx, dx = ground_step(vx, d, shoe, dt)
            x += dx
        else:
            vx, vy, dx, dy = air_step(vx, vy, d, shoe, dt)
            x, y = x + dx, y + dy
            if vy > 0 and y >= floor:
                y, vy, ground = floor, 0.0, True
        points.append((t + dt, x, y))
    return Replay(points, takeoffs)
