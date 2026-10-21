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
  newer, so it is used where the recordings agree: speeding up at 1400 units/s² from standing (five
  starts), braking at 800 (seven stops), the kick at the jump (five jump-backs, 34 jumps of the sweeps),
  braking at 200 in the air with the opposite arrow (five holds). Turning on the ground, the fall-cap drag
  and the rope side-jump are the decompile's alone.
- The game applies it once every STEP (30 ms): the jump sweeps of 2026-10-21 (STEP, below). replay and
  the simulator move her so; the closed forms are their mean over where the steps fall against the keys.
- The lags are the medians of the recordings (Lags), with their spread.

With these fixed lags, its path the mean over the steps' phase, the engine follows 18 tracked recordings
(the 60 fps ones before the sweeps) to 1.94 units RMS in x and 1.67 in y, where the 5 ms engine with the
lags then was off by 2.49 and 1.58, and the model before it (a standing start that switches to full
speed, an even slide, nothing in the air) by 3.20 and 2.72 (`python -m gamelab.analysis.physics_check`).
Without do_ledge2 (a moving camera at 30 fps, which the research set aside): 1.82 and 1.57, against 2.28
and 1.45 and 2.49 and 2.76. The gain in x is the jump-backs keyed 80 ms after the arrow (0.9-1.7 against
4.3-5.6): the steps make them kicked or not by where they fall, half and half. With lags and the phase
fitted to each recording: 0.89 and 0.75. And the sweeps' 18 tracked jumps, with each one's own lags, take
off after as many steps as hers, as fast to 4 units/s, and land within 2 units of where she did
(tests/test_recordings_physics.py).
"""

import bisect
import functools
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
# s per step of the game's movement: it acts on the keys it has seen, and changes her speed on the ground
# (and lifts her off), once every STEP. In game on 2026-10-21 (18 jumps from standing keyed 40-160 ms after
# the arrow, tracked at 30 fps) her ground run at lift-off sat in five tight groups, 0.45-0.96, 2.34-2.79,
# 5.19-6.54, 10.09-10.68 and 14.04-15.23 units, which are 1-5 steps of 30 ms at 1400 units/s² (0.63, 2.52,
# 5.67, 9.80, 14.27), with nothing between; and every jump that lifted off three steps after the arrow
# left at 124.7-126.9 units/s (3 × 42: too fast for the kick, short of full speed), the only speed below
# full seen but the 119 of a jump the arrow's own step lifts off (16 more jumps, by their distance).
# With a 5 ms step the unkicked ones would spread over 119-149 by when the jump came. The frames at 60 fps
# move evenly all the same: the game draws her between its steps.
STEP = 0.030
# s per step of the engine's own integration: her flight through the air (body.fly, which the planner's
# landings and the simulator's flights are), and the simulator's tick. In the air the game's arc is the
# same at its steps; landing within 5 ms of where her feet cross a floor keeps the planner's landings
# on platform edges as they were.
TICK = 0.005
# Landing between two of the game's steps, she stands where she lands until the next one, keeping her speed:
# the landing takes the rest of its step. In game on 2026-10-21, between landing and letting go of the arrow
# she ran 3.3 ± 0.8 units less than the engine had her running on from where she landed (all 18 jumps of the
# sweep, 1.6-4.8; landing where the arc says, to 1.7 units), which this makes 2.2 on average (0 to 4.5 by when
# the next step comes). False: she runs on at her landing speed until the next step (0 less on average).
LANDING_ENDS_STEP = True


def landing_wait(step: float = STEP) -> float:
    """s she stands where she lands before the game's next step moves her on, on average (LANDING_ENDS_STEP):
    half a step, 15 ms (the planner's run on from a landing starts that much later)."""
    return step / 2 if LANDING_ENDS_STEP else 0.0


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
    """s from a key going out to the game acting on it, on average: the game sees the key, and acts on it
    at its next step (STEP), half a step later on average. The medians of the recordings, which include
    that wait: what the planner's closed forms take. The step engine (replay, and the simulator) has the
    game see each key half a step sooner than this (seen), spread by sd_*, and act at its next step: the
    wait adds 8.7 ms of spread (30/√12), which with sd 7.7 makes the 11-13 measured. Only their differences
    matter to a key plan; the game's drawing and the capture add to all of them alike."""
    on: float = 0.066     # an arrow pressed -> she starts to speed up (18 starts at 30 fps: 37-86 ms, sd 12;
                          # 10 at 60 fps before: 60)
    off: float = 0.068    # an arrow let go -> she starts to brake (18 stops at 30 fps, by how far she went
                          # on: 45-92 ms, sd 13; 7 at 60 fps: 72)
    jump: float = 0.065   # the jump key -> she leaves the ground (18 jumps at 30 fps: 64, 47-88, sd 11; 18 at
                          # 60 fps: 67, 45-95): between the two, which follows both sets' arcs best
    drop: float = 0.28    # the jump key with down held -> she falls through (4 drops: 0.28-0.34 s)
    sd_on: float = 0.0077   # the spread of when the game sees the key, before the wait for its step
    sd_off: float = 0.0077
    sd_jump: float = 0.0077

    @staticmethod
    def seen(lag: float, step: float = STEP) -> float:
        """When (s after its key, on average) the game sees a key it acts on `lag` after it on average."""
        return lag - step / 2


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
    at full speed; one just below full speed (119-149) gets no kick and leaves as she was going: on the
    game's steps that is 126, three steps into her run (all five such of the sweeps: 124.7-126.9). In the
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
    going for the release's lag (speeding up still, below full speed), then slides. 24.0 units from full
    speed (18 stops at 30 fps: 24.0 ± 2.0; seven at 30 and 60 fps before: 23.4-28.1)."""
    s, u = _run(abs(v), lags.off, shoe)
    return s + slide(u, shoe)


def stop_time(v: float, shoe: Shoe, lags: Lags) -> float:
    """s from letting go of the arrow at speed v until she stands still: 0.25 from full speed (braked a step
    at a time, the last step's to a stop: 13 ms more on average)."""
    _, u = _run(abs(v), lags.off, shoe)
    return lags.off + u / shoe.drag


@functools.lru_cache(maxsize=256)
def steps_run(k: int, shoe: Shoe, step: float = STEP) -> float:
    """How far she goes from standing with an arrow the game acts on for k of its steps, then none, braked
    to rest a step at a time: 1.8, 7.0, 15.7 and 23.7 units for one to four steps of 30 ms, then 4.47 (a
    step at full speed) more for each."""
    v = x = 0.0
    for i in range(k):
        v, dx = ground_step(v, 1, shoe, step)
        x += dx
        if v >= shoe.vmax:  # at full speed: every step more is as long
            return x + (k - i - 1) * shoe.vmax * step + _braked(v, shoe, step)
    return x + _braked(v, shoe, step)


def _braked(v: float, shoe: Shoe, step: float) -> float:
    """How far she slides from speed v with no arrow, braked a step at a time (13.94 from full speed)."""
    x = 0.0
    while v > 0:
        v, dx = ground_step(v, 0, shoe, step)
        x += dx
    return x


def hold_distance(ms: float, shoe: Shoe, lags: Lags, step: float = STEP) -> float:
    """How far an arrow held `ms` from standing moves her, the slide included, on average over the game's
    steps. The game sees it held 2 ms longer (off - on), z of its steps, and acts on it for floor(z) of them
    or one more, as its steps fall (the chance of the more is z's fraction): so on average the line between
    the two (steps_run). Short taps are bimodal: a 50 ms tap moves her 1.8 or 7.0 (5.6 on average; in game,
    of three 50 ms taps two moved her 0 and one 7.1). From 118 ms on, 6.2 + 149 t for a hold of t s."""
    seen = ms / 1000 + lags.off - lags.on
    if seen <= 0:
        return 0.0
    z = seen / step
    k = int(z)
    return steps_run(k, shoe, step) + (z - k) * (steps_run(k + 1, shoe, step) - steps_run(k, shoe, step))


def hold_time(ms: float, shoe: Shoe, lags: Lags) -> float:
    """s from an arrow's key going down, held `ms` from standing, until she stands still again (the game
    acts on its release lags.off after its key, and she brakes from what she reached): 0.21 s for a 50 ms
    tap, which has her at 73 units/s when the game acts on its release (on the game's steps, 13 ms more on
    average: it brakes her to a stop a step at a time)."""
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


def hold_for(units: float, shoe: Shoe, lags: Lags, step: float = STEP) -> int:
    """How long (ms) to hold an arrow from standing to move `units` and stop there, on average over the
    game's steps: hold_distance's inverse (46 ms for 5 units, 104 for 20, 294 for 50)."""
    if units <= 0:
        return 0
    k = 0
    while steps_run(k + 1, shoe, step) < units:
        k += 1
    lo, hi = steps_run(k, shoe, step), steps_run(k + 1, shoe, step)
    seen = (k + (units - lo) / (hi - lo)) * step
    return max(0, round((seen - (lags.off - lags.on)) * 1000))


def _below_mean(c: float, mean: float, sd: float) -> float:
    """E[max(0, c - X)] for X normal (mean, sd)."""
    if sd <= 0:
        return max(0.0, c - mean)
    z = (c - mean) / sd
    return sd * (z * 0.5 * (1 + math.erf(z / math.sqrt(2))) + math.exp(-z * z / 2) / math.sqrt(2 * math.pi))


def steps_held_chances(gap_ms: float, lags: Lags, step: float = STEP) -> dict[int, float]:
    """The chance of each number of the game's steps k between the one that acts on an arrow and the one
    that acts on a jump keyed gap_ms after it: k = floor(z + U), z = (gap + seen jump - seen arrow) / step,
    U the step grid's phase (uniform), the lags spread as measured. For a given z that is floor(z) or one
    more, so each k's chance is the triangle max(0, 1 - |z - k|) averaged over z."""
    mean = (gap_ms / 1000 + lags.jump - lags.on) / step
    sd = math.hypot(lags.sd_jump, lags.sd_on) / step
    lo, hi = math.floor(mean - 6 * sd) - 1, math.ceil(mean + 6 * sd) + 1
    r = {k: _below_mean(k, mean, sd) for k in range(lo - 1, hi + 2)}
    return {k: max(0.0, r[k + 1] - 2 * r[k] + r[k - 1]) for k in range(lo, hi + 1)}


def short_takeoff_chance(gap_ms: float, shoe: Shoe, lags: Lags, slower: float = 0.94,
                         step: float = STEP) -> float:
    """The chance that a jump keyed gap_ms after the arrow, from standing, leaves the ground slower than
    `slower` × vmax (140 of her 149), with the lags as spread as measured and the game's steps
    (steps_held_chances). k steps of the arrow have her at k × 42 units/s (1400 × 30 ms): lifted off by
    the arrow's own step (k 0) she leaves at 119 (the kick from standing), after one or two at full speed
    (kicked from 42 or 84), after three at 126 (too fast for the kick, short of full speed: 13 units short
    of a full jump's landing), after four or more at full speed. Before the arrow's step (k below 0) she
    leaves straight up."""
    short = 0.0
    for k, p in steps_held_chances(gap_ms, lags, step).items():
        if k < 0:
            v = 0.0
        else:
            v = takeoff(min(k * shoe.acc * step, shoe.vmax), 1, shoe)
        if v < slower * shoe.vmax:
            short += p
    return short


def jump_window_ok(gap_ms: float, shoe: Shoe, lags: Lags, risk: float = 0.05) -> bool:
    """Whether a jump keyed gap_ms after the arrow, from standing, surely leaves at full speed (at most
    `risk` that it's short): 40-52 ms by the median lags (the kick, after one or two steps of the arrow), or
    129 ms and later (four steps). In between she leaves at 126 after three steps, more often than not from 80
    to 110 ms: in game, of the jumps keyed 80, 100 and 120 ms after the arrow, 2 of 4, 3 of 4 and 3 of 4 left
    short."""
    return short_takeoff_chance(gap_ms, shoe, lags) <= risk


def blind_ends_ms(shoe: Shoe, lags: Lags, risk: float = 0.05, longest: int = 400) -> int:
    """The soonest gap (whole ms after the arrow) from which a jump from standing keyed then, or any time
    later, leaves short at most `risk` of the time: where the blind spot of three steps ends."""
    t = longest
    while t > 0 and short_takeoff_chance(t - 1, shoe, lags) <= risk:
        t -= 1
    return t


# --- in the air over a map ----------------------------------------------------------------------------

Walls = tuple[list[float], list[tuple[float, float]]]  # vertical walls by x: their xs, each one's (top, bottom)

# Which walls stop her: only those of her own foothold group (mapdata.Segment.group), the group of the
# foothold she last stood on. A jump or a fall through her platform keeps it, landing sets it, and a rope
# gives her none (ROPE_GROUP). Floors of every group land her. A reconstruction of the client's code has its
# airborne collision skip a wall of any other group (ThatMapleDev's "Player Physics" gist), and so did she
# in the live runs (telemetry, 2026-10-19 and 20; her minimap dot fitted to this engine's arcs): six flights
# went across another group's wall (15-37 units inside its span; one just under the floor of the block it
# hangs from, which she'd have landed on above it), and each came down within 13 units of where this puts
# it, against 54-142 units off (or on another platform) with every wall stopping her. Three walk-offs off
# platform 3's right end went through platform 1's wall 12 units on, planned for the basin, and landed on 17
# at 1,005-1,015 (this: 1,004-1,008; stopped by it: the basin, or 869 round its foot); two jumps from 19
# that came in under 18's floor went through its wall onto 24 at 198 (this: 185-198; stopped: 140-144); one
# from 22 went through 21's wall onto 34 at -138 (this: -138; stopped: -56). None was seen stopped by one.
# False: every wall stops her, whatever its group (as before).
OWN_GROUP_WALLS = True
EVERY_WALL = "every wall"  # a group that has every wall stop her (whatever OWN_GROUP_WALLS says)
# Her group off a rope: none, so no wall stops her. (The decompile gives a rope group 0 on the rope's own
# page, 4 on this map, which no foothold is on; read as the group number alone, it is the basin's (3, 0),
# whose walls are nowhere near a rope here. Untested in game.)
ROPE_GROUP = None


def vertical_walls(segments) -> Walls:
    """The vertical ones of a map's wall segments, sorted by x, as fly takes them: their xs, and each
    one's top and bottom (a unit beyond its ends)."""
    ws = sorted((w for w in segments if w.x1 == w.x2), key=lambda w: w.x1)
    return [w.x1 for w in ws], [(min(w.y1, w.y2) - 1, max(w.y1, w.y2) + 1) for w in ws]


def walls_for(model, group) -> list:
    """The walls of the map that stop her with foothold group `group` (Segment.group; ROPE_GROUP off a
    rope, EVERY_WALL for all of them): her own group's (OWN_GROUP_WALLS), or else all of them."""
    if group == EVERY_WALL or not OWN_GROUP_WALLS:
        return model.walls
    return model.walls_of(group)


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


def _side_at(platforms, x: float, y: float) -> int:
    """Which side of a wall at exactly x she is on, standing at (x, y): at the right end of the platform
    under her feet she's on its left (-1), at its left end on its right (+1); 0 if neither (along a
    platform, or in the air: there a wall at her x stops her either way)."""
    for q in platforms:
        if q.covers(x) and abs(q.y_at(x) - y) <= 1.5:
            if x == q.right:
                return -1
            if x == q.left:
                return 1
    return 0


def fly(model, x: float, y: float, vx: float, vy: float, *, shoe: Shoe, arrow: int = 0, skip=None,
        group=EVERY_WALL, walls: Walls | None = None, sides: tuple[float, float] | None = None,
        dt: float = TICK, limit: float = 4.0, path: bool = False, airborne: bool = False) -> Flight:
    """Where she lands from (x, y) moving at (vx, vy) with `arrow` held all the way: the first platform
    whose surface her feet cross on the way down (_lands_at), whatever its group. Walls stop her sideways:
    `walls`, or else those that stop her with foothold `group` (walls_for: the group of the platform she
    leaves; by default every wall), worked out each call; `sides`, the map's left and right limits, stop
    her too. A wall at exactly her start, at the end of the platform she stands on (where a wall stops
    her walking), stops her only going into it: she's on her platform's side of it, and a jump away from
    it leaves (counted as in her way both ways, she went straight up against it). `skip` is a platform
    she falls through (dropping). A flight leaving the ground can't land on its first step; one carried on
    from mid-air (`airborne`) can. Not cached."""
    wall_xs, wall_ys = walls if walls is not None else vertical_walls(walls_for(model, group))
    # Only platforms she could reach in the time: within her sideways travel, at or below the top of the jump.
    reach = max(abs(vx), shoe.air_speed if arrow else 0.0) * limit + 1
    top = y - (vy * vy / (2 * shoe.gravity) if vy < 0 else 0) - 1
    plats = [q for q in model.platforms if q is not skip and q.right >= x - reach and q.left <= x + reach
             and max(q.y_at(q.left), q.y_at(q.right)) >= top]
    x0 = x
    i = bisect.bisect_left(wall_xs, x)
    side = _side_at(model.platforms, x, y) if i < len(wall_xs) and wall_xs[i] == x and not airborne else 0
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
                if wall_ys[i][0] <= ny <= wall_ys[i][1] and not (side and wall_xs[i] == x == x0
                                                                 and (nx - x) * side > 0):
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
    """Her body under a key timeline on flat ground (replay): (t, x, y) after every tick, and each
    lift-off's (t, her speed on the ground, her sideways speed leaving it)."""
    points: list[tuple[float, float, float]]
    takeoffs: list[tuple[float, float, float]]


def phases(step: float = STEP, dt: float = TICK) -> list[float]:
    """The phases of the game's step grid replay can take, one per tick of a step (0, 5, ..., 25 ms)."""
    return [i * dt for i in range(max(1, round(step / dt)))]


def replay(events, until: float, shoe: Shoe, lags: Lags, *, start: float = -0.1, dt: float = TICK,
           step: float = STEP, phase: float = 0.0, floors=()) -> Replay:
    """Her body on flat ground under a key timeline, as the game moves it: from standing at x 0 on a floor
    at y 0 (y down: negative in the air), from `start` to `until` s, every dt. `events` are (t s, action,
    down) as the keys went out, the action "left", "right" or "jump" (others are ignored). The game
    sees an arrow's press Lags.seen(lags.on) after its key, its release Lags.seen(lags.off) after and a
    jump Lags.seen(lags.jump) after, and acts on what it has seen at its steps, at `phase` + n × `step`
    (a multiple of dt from `start`): there it lifts her off, and sets her speed on the ground for the step
    to come, which she runs evenly (as the game draws her); of the arrows it sees held, the one pressed
    last counts. A jump it sees while she's in the air is ignored (and holding the key doesn't jump her
    again on landing). In the air she flies every dt, with the arrow it last acted on; landed between
    steps, she waits for the next (LANDING_ENDS_STEP). Each phase is one way the game's steps can
    fall against the keys: `phases` lists them, and the closed forms are their mean. `floors`: (from t,
    y) pairs, the floor she lands on from then on (a recording that lands on another platform)."""
    seen = []
    for t, action, down in events:
        if action in ("left", "right"):
            seen.append((t + Lags.seen(lags.on if down else lags.off, step), action, down))
        elif action == "jump" and down:
            seen.append((t + Lags.seen(lags.jump, step), "jump", True))
    seen.sort(key=lambda e: e[0])
    floors = sorted(floors)
    per = max(1, round(step / dt))  # ticks a step
    first = round((phase - start) / dt) % per  # the first tick that is a step
    held: dict[str, int] = {}
    order = 0
    x = y = vx = vy = glide = 0.0
    ground, floor, d = True, 0.0, 0
    points, takeoffs = [], []
    j = 0
    for k in range(int((until - start) / dt + 1e-9) + 1):
        t = start + k * dt  # counted, not summed, so a step's time doesn't drift
        for since, level in floors:
            if t >= since:
                floor = level
        if (k - first) % per == 0:  # the game's step
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
            if ground and jump:
                takeoffs.append((t, vx, takeoff(vx, d, shoe)))
                vx, vy, ground = takeoffs[-1][2], -shoe.vjump, False
            if ground:
                vx, dx = ground_step(vx, d, shoe, per * dt)
                glide = dx / per
        if ground:
            x += glide
        else:
            vx, vy, dx, dy = air_step(vx, vy, d, shoe, dt)
            x, y = x + dx, y + dy
            if vy > 0 and y >= floor:
                y, vy, ground, glide = floor, 0.0, True, 0.0 if LANDING_ENDS_STEP else vx * dt
        points.append((t + dt, x, y))
    return Replay(points, takeoffs)
