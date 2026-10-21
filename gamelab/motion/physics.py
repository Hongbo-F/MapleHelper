"""Her physics as the planner takes it: the game's jump, gravity and fall speeds, her walking (with its
lags and slides) and climbing (and how long to hold an arrow to walk or run a distance), and where a
jump or fall lands (flown by motion.body's engine against the map's platforms and the walls of her
foothold group, as the simulator flies her). All in map units. The walk, the jump and their lags come
from motion.body, the engine that moves her as the game does.
"""

import itertools
import math
import os
from collections import OrderedDict
from dataclasses import replace

from maplehelper import config

from . import body
from .. import native
from ..localization.mapdata import MapModel, Platform
from ..paths import CONFIG

# Her stats (config.toml [character]), and how the game moves her with them: motion.body's engine, its
# shoe from the game's Physics.img and its lags measured. With FIXED_STATS set in the environment, the
# measured defaults (config.Character()) whatever the file says: the tests set it (tests/conftest.py), and
# the simulator's golden traces and harness restart themselves with it, so editing [character] for her gear
# changes neither what they check nor what they compare. (`python -m gamelab sim`, for watching, takes the
# file's, as the hunt does.)
FIXED_STATS = "GAMELAB_FIXED_STATS"


def stats(path=CONFIG, environ=os.environ) -> config.Character:
    """Her stats: config.toml's [character], or the measured defaults when FIXED_STATS is set."""
    return config.Character() if environ.get(FIXED_STATS) else config.character(path)


CHARACTER = stats()
SHOE = body.Shoe.from_physics(body.PHYSICS_IMG, CHARACTER.speed, CHARACTER.jump)
LAGS = body.Lags()

WALK_SPEED = SHOE.vmax   # map units/s: 125 x SPEED/100, 149 at her 119.2 (measured 148-151: 209 screen px/s)
CLIMB_SPEED = 117.0      # map units/s: measured 0.165 screen px/ms
JUMP_SPEED = SHOE.vjump  # physics: jumpSpeed x JUMP/100 (555)
GRAVITY = SHOE.gravity   # physics: gravityAcc (2000)
FALL_SPEED = SHOE.fall   # physics: fallSpeed (670, terminal)
JUMP_HEIGHT = JUMP_SPEED ** 2 / (2 * GRAVITY)   # 77 map units
ROPE_REACH = 95          # she can grab a rope that ends this far above her feet (jump + arms)
LAND_MARGIN = 18         # land at least this far onto a platform
# Her walk and jump in the planner's terms, from the engine with the median lags (body.Lags: the jump sweeps
# at 30 fps and the 60 fps recordings, the wait for the game's 30 ms step included). From standing the game
# acts on an arrow 66 ms after the key and speeds her up at 1400 units/s² to full speed, so after t s she's
# 149 (t - 0.119) units along. Let go, she runs on until it acts on the release (68 ms), then brakes at 800
# units/s². A jump lifts off 65 ms after its key. A jump key pressed while she's still in the air is ignored,
# and holding it doesn't jump again on landing.
RUN_LAG = LAGS.on + WALK_SPEED / (2 * SHOE.acc)  # 0.119 s: a standing start lags full-speed running by this much
# ... and a start from running the other way at full speed lags a standing one by this much more: braking
# and turning at 1400 units/s² (body.run_time; the decompile's, unmeasured). It was 0.08 s, from a loose
# measurement. Nothing plans with it now (flow turns her by body.run_time): it is kept for its name.
TURN_EXTRA = body.run_time(100.0, -WALK_SPEED, 1, SHOE) - body.run_time(100.0, 0.0, 1, SHOE)  # 0.16 s
JUMP_LAG = LAGS.jump  # 0.065 s from the jump key to leaving the ground
DROP_LAG = LAGS.drop  # 0.28 s from the jump key (down held) to falling through (0.28-0.34 in 4 drops on the
                      # minimap at 60 fps, whether or not she had just stopped running)
STOP_RUN = body.stop_distance(WALK_SPEED, SHOE, LAGS)  # 24.0 units she goes on after the arrow is released at
                                                        # full speed (measured 24.0 ± 2.0 in 18 stops) ...
STOP_S = body.stop_time(WALK_SPEED, SHOE, LAGS)        # ... until she stands still, 0.25 s later
LAND_SLIDE = body.slide(WALK_SPEED, SHOE)              # 13.9 units she slides on landing with no arrow held
# Holding an arrow t ms from standing moves her 6.2 + 0.149 t map units, the slide after release included,
# once t is past about 118 ms (she has reached full speed by then: the game sees a hold 2 ms longer than it
# is); a shorter hold, on average over the game's steps, the line between 1.8, 7.0, 15.7 and 23.7 units for
# one to four steps of it seen held (body.hold_distance, hold_ms). One hold at a time it is one of the two:
# a 50 ms tap moves her 1.8 or 7.0 units (5.6 on average), as bimodal as in game (of three, two moved her 0
# and one 7.1). Shorter than MIN_HOLD_MS isn't sent.
WALK_UNITS_PER_MS = WALK_SPEED / 1000
WALK_UNITS_FIXED = body.hold_distance(1000, SHOE, LAGS) - WALK_SPEED  # 6.2
MIN_HOLD_MS = 30
# When to press jump (ms after the arrow, from standing) to lift off `units` along: she's at
# 0.149 (t - 119) units after t ms, and lifts off 65 ms after the key (JUMP_LAG), so
# RUN_LAG_MS = RUN_LAG - JUMP_LAG (54).
RUN_LAG_MS = round((RUN_LAG - JUMP_LAG) * 1000)
# A jump from standing, keyed t ms after the arrow, leaves at full speed only outside the blind spot
# (body.short_takeoff_chance): the game's steps of 30 ms have her at 42, 84, 126 and 149 units/s after one to
# four steps of the arrow, and the kick takes the first two to full speed, not the third. So a jump the arrow's
# own step lifts off leaves at 119, one after one or two steps at full speed (the kick), after three at 126
# (landing 13 units short of a full jump), after four or more at full speed. In game on 2026-10-21 (34 jumps
# keyed 0-160 ms after the arrow, from standing on platform 34), of those keyed 80, 100 and 120 ms after it 2
# of 4, 3 of 4 and 3 of 4 left short, and none of the 10 keyed 130 ms or later. Full speed by her own run is
# what the planner times jumps from standing for. The kick from a near standstill worked in all 4 jumps keyed
# 40 ms after the arrow and 3 of 4 at 60, which the step model puts at 5% and 13% short by the median lags (up
# to 27% and 21% within what the sweeps allow): not reliable enough to plan on, so KICK_JUMPS stays off. True
# lets the planner key jumps from standing in the kick's window too (body.jump_window_ok: 40-52 ms after the
# arrow by the median lags), lifting off within 2 units of where she stands, and keys the jump-back 52 ms after
# its arrow (control.actions.JB_RUN_MS), the window's best by the sweeps' fit: about 20 more runs there in game
# should confirm it first.
KICK_JUMPS = False
# The difference of the jump's lag and the arrow's (when the game sees them, its step aside), as the two jump
# sweeps of 2026-10-21 have it: the step model fitted to all 34 jumps' outcomes (and the steps of the 18
# tracked) puts it at -7 ms, sd 10; within 2 log-likelihood of the best, from -13 to -1 ms, its spread from
# 4 to 17 ms. The median lags have -1 (body.Lags: 65 - 66), and the spread of two keys seen, 10.9.
SWEEP_LAG_GAP = (-0.013, -0.001)
SWEEP_LAG_GAP_SD = (0.004, 0.017)
BLIND_RISK = 0.05  # the most jumps from standing that may leave short
# ms after the arrow from which a jump from standing leaves at full speed by her own run (four steps of the
# arrow or more) but for at most BLIND_RISK of jumps, wherever in that range the lags' gap is: 150. By the
# median lags 129 (138 for 1%), by the sweeps' best fit 134; at the old 118, 20% leave short by the median lags,
# 33% by the best fit and up to 50% within its range. No jump from standing is keyed sooner than this after its
# arrow (sure_jump_ms), but in the kick's window with KICK_JUMPS.
BLIND_ENDS_MS = max(body.blind_ends_ms(SHOE, replace(LAGS, jump=LAGS.on + gap, sd_on=sd / math.sqrt(2),
                                                     sd_jump=sd / math.sqrt(2)), BLIND_RISK)
                    for gap in SWEEP_LAG_GAP for sd in SWEEP_LAG_GAP_SD)
# Attacks from a jump (control.actions times her key plans with these; attack.reach works out what such an
# attack can hit). An attack picks its target when it starts, about JUMP_LAG after its key, the same input
# lag as a jump's lift-off: so when it starts she has been in the air for about as long as the attack key
# came after the jump key.
JUMP_ATTACK_MS = 40        # a jump throw's attack key comes this long after the jump key. In game on 2026-10-20,
                           # at a snail 20-60 units away: keyed 100 ms after the jump (the macro's F1 combo), the
                           # star went over it; keyed 40 ms after, a star hit it for 78 and killed it
JUMP_BACK_ATTACK_MS = 85   # ... and a jump back's: right after she turns in mid-air (actions.JB_THROW_MS 235
                           # less JB_RUN_MS 150, when the jump key goes down)
ATTACK_DECIDE_LAG_MS = 11  # it can start up to this much later than that (60 fps recordings jump_throw_0 and
                           # _1, keys 100 ms apart: her attack pose first shows 91-110 ms after lift-off)
# fly's landings, the FLY_CACHE_SIZE most recently used. Each is worked out from its start rounded as its
# key is (x and y to 0.01 units, speeds to 0.1 units/s), so it is a function of the key alone: which call
# came first, or which landings were dropped, can't change an answer. (Worked out from the exact start, a
# landing cached by one call was handed to every later one that rounded the same, so the simulator's
# scenarios each had to start from the same cache to give the same trace.) On this map the planner's own
# jumps and falls are 5,726 landings (about 2.6 MB) since its jumps' takeoffs are placed for room (route._placed;
# 3,128 before, and the 8-minute simulated hunt loop loop_run added none).
FLY_CACHE_SIZE = 20_000  # landings (about 9 MB when full)
MODELS_KEPT = 4          # maps fly knows at a time (their walls, and the serial numbers keying their landings)
_FLY_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
# The last MODELS_KEPT models fly was used on, by id(model): each kept with its serial number (its landings'
# key), the walls that stop her by foothold group (body.walls_for), and the same laid out for the flight in C
# (native: the platforms, and the walls by group; made the first time she flies there with the DLL in use).
# Keyed by id(model) alone, a freed model's id was reused by the next one, which then got the old one's walls
# and landings (tests failed depending on their order). A model is kept while it's in here, so its id can't be
# taken, and a model seen again after it was let go gets a new serial: no landing cached for another model can
# ever be handed to it.
_MODELS: OrderedDict[int, tuple[MapModel, int, dict, dict]] = OrderedDict()
_SERIALS = itertools.count()
_SHOES: dict[body.Shoe, object] = {}  # each shoe as the C flight takes it (native.shoe_of)


def _known(model: MapModel, group) -> tuple[int, body.Walls, dict]:
    """The model's serial number, the vertical walls that stop her with foothold `group`, sorted by x (their
    xs, and each one's (top, bottom)), and its layouts for the C flight (_native_flight's)."""
    known = _MODELS.get(id(model))
    if known is None or known[0] is not model:
        known = _MODELS[id(model)] = (model, next(_SERIALS), {}, {})
    _MODELS.move_to_end(id(model))
    if len(_MODELS) > MODELS_KEPT:
        _MODELS.popitem(last=False)
    walls = known[2].get(group)
    if walls is None:
        walls = known[2][group] = body.vertical_walls(body.walls_for(model, group))
    return known[1], walls, known[3]


def fly(model: MapModel, x: float, y: float, vx: float, vy: float, skip: Platform | None = None,
        dt: float = body.TICK, limit: float = 4.0, group=body.EVERY_WALL,
        arrow: int | None = None) -> tuple[Platform | None, float, float]:
    """Where she lands from (x, y) moving at (vx, vy) (map units/s, y down): the first platform whose
    surface her feet cross on the way down. Walls stop her sideways: those of foothold `group` (the
    platform's she leaves, Platform.group; body.ROPE_GROUP off a rope), unless body.OWN_GROUP_WALLS is
    off; by default every wall. `skip` is a platform she falls through (dropping). Returns (platform,
    landing x, seconds). The start is rounded (x and y to 0.01 units, the speeds to 0.1 units/s) and
    results are cached by it: chains try the same jumps many times over."""
    return flight(model, x, y, vx, vy, skip, dt, limit, group, arrow)[:3]


def flight(model: MapModel, x: float, y: float, vx: float, vy: float, skip: Platform | None = None,
           dt: float = body.TICK, limit: float = 4.0, group=body.EVERY_WALL,
           arrow: int | None = None) -> tuple[Platform | None, float, float, float]:
    """fly's landing, and her sideways speed as she lands. Flown by the engine (body.fly), the one
    integration the simulator flies her by too, with `arrow` held all the way: by default the way she
    leaves (none straight up), as a plan holds it. Held, it does nothing to her speed at the speeds she
    jumps with (it steers toward 8.9 units/s at most); none, and the air drags her a little (a lot at the
    fall cap). Her speed as she lands is 0 after a wall stopped her. On the ground she slides on from it,
    or runs on."""
    if not body.OWN_GROUP_WALLS:
        group = body.EVERY_WALL  # (one set of landings, whatever group is asked for)
    serial, walls, laid_out = _known(model, group)
    x, y, vx, vy = round(x, 2), round(y, 2), round(vx, 1), round(vy, 1)
    if arrow is None:
        arrow = (vx > 0) - (vx < 0)
    key = (serial, group, x, y, vx, vy, arrow, None if skip is None else skip.id, dt, limit)
    result = _FLY_CACHE.get(key)
    if result is not None:
        _FLY_CACHE.move_to_end(key)
        return result
    # The same flight in C (native.c: fly), when it's built: 0.0xx ms a flight, against 1.0 in Python.
    result = _native_flight(model, laid_out, group, walls, x, y, vx, vy, skip, dt, limit, arrow) if limit > 0 else None
    if result is None:
        f = body.fly(model, x, y, vx, vy, shoe=SHOE, arrow=arrow, skip=skip, walls=walls, dt=dt, limit=limit)
        result = (f.platform, f.land_x, f.seconds, f.vx_land)
    _FLY_CACHE[key] = result
    if len(_FLY_CACHE) > FLY_CACHE_SIZE:
        _FLY_CACHE.popitem(last=False)
    return result


def _native_flight(model: MapModel, laid_out: dict, group, walls: body.Walls, x: float, y: float, vx: float,
                   vy: float, skip: Platform | None, dt: float, limit: float,
                   arrow: int) -> tuple[Platform | None, float, float, float] | None:
    """flight's body.fly in C (native.fly), with the model's platforms and the group's walls laid out for it once
    (`laid_out`, kept with the model in _MODELS); None if the DLL isn't there. The answers are body.fly's to the
    bit (the same steps in the same order, IEEE double), and of the same types: where no step changed her x or
    her speed (a wall at her start stopping her every step; no arrow and no speed), Python hands back the start's
    own, an int if it was one, and so does this. (With no step at all, limit 0, it would be the start's whatever
    it was: flight leaves that to Python.)"""
    if native.lib() is None:
        return None
    m = laid_out.get("platforms")
    if m is None:
        m = laid_out["platforms"] = native.Map(model.platforms)
    w = laid_out.get(group)
    if w is None:
        w = laid_out[group] = native.Walls(walls)
    shoe = _SHOES.get(SHOE)
    if shoe is None:
        shoe = _SHOES[SHOE] = native.shoe_of(SHOE)
    s = -1 if skip is None else next((i for i, q in enumerate(model.platforms) if q is skip), -1)
    found = native.fly(m, w, x, y, vx, vy, int(arrow), dt, limit, shoe, s)
    if found is None:
        return None
    q, lx, t, vl, same_x, same_vx = found
    return None if q < 0 else model.platforms[q], x if same_x else lx, t, vx if same_vx else vl


# A route jump has to land on its platform at this share of her walking speed too: a jump from standing
# keyed in the blind spot leaves at 126 (three of the game's steps into her run: 0.846 × 149), and one the
# mover keys (control.mover, MIN_RUN on) does so for 0.01% of jumps by the median lags (up to 5% within what
# the jump sweeps of 2026-10-21 allow).
SLOW_TAKEOFF = 0.85


def sure_landing(model: MapModel, x: float, y: float, vx: float, vy: float, land: Platform,
                 slop: float = 12.0, group=body.EVERY_WALL) -> bool:
    """Whether a jump from (x, y) still lands on `land` if she leaves the ground `slop` units early or
    late, or 10% slower, or as slow as SLOW_TAKEOFF: a jump down from platform 25 meant for the basin
    landed on platform 33, whose edge its arc passed close by. (`group`: fly's.)"""
    sign = 1 if vx > 0 else -1 if vx < 0 else 0
    for dx, k in ((-slop, 1.0), (slop, 1.0), (0.0, 0.9), (0.0, SLOW_TAKEOFF)):
        x0 = x + sign * dx
        got, _, _ = fly(model, x0, y, vx * k, vy, group=group)
        if got is not land:
            return False
    return True


def jump_rise(ms: float) -> float:
    """How high her feet are above the ground she jumped from, `ms` after the jump key of a jump straight
    up: still on it until lift-off (JUMP_LAG), then the game's arc (77 units up 277 ms after lift-off,
    back down 555 ms after it)."""
    t = ms / 1000 - JUMP_LAG
    return max(0.0, JUMP_SPEED * t - GRAVITY * t * t / 2) if t > 0 else 0.0


def air_time(dy_up: float) -> float | None:
    """Seconds from takeoff until she comes down through a height dy_up above takeoff (negative = below)."""
    disc = JUMP_SPEED ** 2 - 2 * GRAVITY * dy_up
    if disc < 0:
        return None
    return (JUMP_SPEED + math.sqrt(disc)) / GRAVITY


def fall_time(dy: float) -> float:
    """Seconds to fall dy units from rest, by gravity alone (0 for dy <= 0)."""
    return math.sqrt(2 * max(0.0, dy) / GRAVITY)


def hold_ms(units: float) -> int:
    """How long to hold an arrow to walk `units` map units and come to rest there (body.hold_for: 46 ms
    for 5 units, 104 for 20, 294 for 50), MIN_HOLD_MS at least."""
    return max(MIN_HOLD_MS, body.hold_for(units, SHOE, LAGS))


def sure_jump_ms(gap: float) -> float:
    """The soonest a jump from standing can be keyed, `gap` ms after the arrow or later, and surely leave
    the ground at full speed (KICK_JUMPS above): BLIND_ENDS_MS (150) after the arrow for any gap up to that,
    the gap itself from there on. With KICK_JUMPS, a gap in the kick's window too (before three of the game's
    steps, and short at most BLIND_RISK of the time by the median lags)."""
    def sure(t: float) -> bool:
        if t >= BLIND_ENDS_MS:
            return True
        return KICK_JUMPS and t < 3 * body.STEP * 1000 and body.jump_window_ok(t, SHOE, LAGS, BLIND_RISK)
    if gap >= 0 and sure(gap):
        return gap
    t = max(0, math.ceil(gap))
    while not sure(t):
        t += 1
    return t


def run_ms(units: float) -> int:
    """How long after pressing the arrow (from standing) to press jump to lift off `units` along, still
    running; never in the blind spot, though (sure_jump_ms: keyed there, the jump can find her three steps
    into her run and leave at 126). So a run-up shorter than about 14 units is keyed 150 ms in and lifts her
    off about 14 units along, at full speed."""
    if units <= 0:
        return 0
    return sure_jump_ms(round(units / WALK_UNITS_PER_MS + RUN_LAG_MS))
