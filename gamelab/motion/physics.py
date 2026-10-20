"""Her physics as the planner takes it: the game's jump, gravity and fall speeds, her walking (with its
lags and slides) and climbing (and how long to hold an arrow to walk or run a distance), and where a
jump or fall lands (simulated against the map's platforms and walls). All in map units. The walk, the
jump and their lags come from motion.body, the engine that moves her as the game does.
"""

import bisect
import itertools
import math
from collections import OrderedDict

from maplehelper import config

from . import body
from ..localization.mapdata import MapModel, Platform
from ..paths import CONFIG

# Her stats (config.toml [character]), and how the game moves her with them: motion.body's engine, its
# shoe from the game's Physics.img and its lags measured.
CHARACTER = config.character(CONFIG)
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
# Her walk and jump in the planner's terms, from the engine with the median lags (body.Lags; the 60 and 30
# fps recordings). From standing the game sees an arrow 60 ms after the key and speeds her up at 1400
# units/s² to full speed, so after t s she's 149 (t - 0.113) units along. Let go, she runs on until the
# game sees the release (72 ms), then brakes at 800 units/s². A jump lifts off 67 ms after its key (18
# jumps: 45-95 ms). A jump key pressed while she's still in the air is ignored, and holding it doesn't jump
# again on landing.
RUN_LAG = LAGS.on + WALK_SPEED / (2 * SHOE.acc)  # 0.113 s: a standing start lags full-speed running by this much
JUMP_LAG = LAGS.jump  # 0.067 s from the jump key to leaving the ground
DROP_LAG = LAGS.drop  # 0.28 s from the jump key (down held) to falling through (0.28-0.34 in 4 drops on the
                      # minimap at 60 fps, whether or not she had just stopped running)
STOP_RUN = body.stop_distance(WALK_SPEED, SHOE, LAGS)  # 24.6 units she goes on after the arrow is released at
                                                        # full speed (measured 23.4-28.1) ...
STOP_S = body.stop_time(WALK_SPEED, SHOE, LAGS)        # ... until she stands still, 0.26 s later
LAND_SLIDE = body.slide(WALK_SPEED, SHOE)              # 13.9 units she slides on landing with no arrow held
# Holding an arrow t ms from standing moves her 7.7 + 0.149 t map units, the slide after release included,
# once t is past about 95 ms (she has reached full speed by then: the game sees a hold 12 ms longer than it
# is); a shorter hold, 1925 (t/1000 + 0.012)² (body.hold_distance, hold_ms). Taps of 50 ms or less are
# bimodal in game: of three, two moved her 0 and one 7.1. Shorter than MIN_HOLD_MS isn't sent.
WALK_UNITS_PER_MS = WALK_SPEED / 1000
WALK_UNITS_FIXED = body.hold_distance(1000, SHOE, LAGS) - WALK_SPEED  # 7.7
MIN_HOLD_MS = 30
# When to press jump (ms after the arrow, from standing) to lift off `units` along: she's at
# 0.149 (t - 113) units after t ms, and lifts off 67 ms after the key (JUMP_LAG), so
# RUN_LAG_MS = RUN_LAG - JUMP_LAG (46).
RUN_LAG_MS = round((RUN_LAG - JUMP_LAG) * 1000)
# Attacks from a jump (control.actions times her key plans with these; attack.reach works out what such an
# attack can hit). An attack picks its target when it starts, about JUMP_LAG after its key, the same input
# lag as a jump's lift-off: so when it starts she has been in the air for about as long as the attack key
# came after the jump key.
JUMP_ATTACK_MS = 40        # a jump throw's attack key comes this long after the jump key. In game on 2026-10-20,
                           # at a snail 20-60 units away: keyed 100 ms after the jump (the macro's F1 combo), the
                           # star went over it; keyed 40 ms after, a star hit it for 78 and killed it
JUMP_BACK_ATTACK_MS = 85   # ... and a jump back's: right after she turns in mid-air (actions.JB_THROW_MS 165
                           # less JB_RUN_MS 80, when the jump key goes down)
ATTACK_DECIDE_LAG_MS = 11  # it can start up to this much later than that (60 fps recordings jump_throw_0 and
                           # _1, keys 100 ms apart: her attack pose first shows 91-110 ms after lift-off)
# fly's landings, the FLY_CACHE_SIZE most recently used. Each is worked out from its start rounded as its
# key is (x and y to 0.01 units, speeds to 0.1 units/s), so it is a function of the key alone: which call
# came first, or which landings were dropped, can't change an answer. (Worked out from the exact start, a
# landing cached by one call was handed to every later one that rounded the same, so the simulator's
# scenarios each had to start from the same cache to give the same trace.) On this map the planner's own
# jumps and falls are 2,718 landings (about 1 MB); a 60-minute simulated hunt loop added 1 more.
FLY_CACHE_SIZE = 20_000  # landings (about 9 MB when full)
MODELS_KEPT = 4          # maps fly knows at a time (their walls, and the serial numbers keying their landings)
_FLY_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
# The last MODELS_KEPT models fly was used on, by id(model): each kept with its serial number (its landings'
# key) and its walls. Keyed by id(model) alone, a freed model's id was reused by the next one, which then got
# the old one's walls and landings (tests failed depending on their order). A model is kept while it's in
# here, so its id can't be taken, and a model seen again after it was let go gets a new serial: no landing
# cached for another model can ever be handed to it.
_MODELS: OrderedDict[int, tuple[MapModel, int, tuple]] = OrderedDict()
_SERIALS = itertools.count()


def _known(model: MapModel) -> tuple[int, tuple[list[float], list[tuple[float, float]]]]:
    """The model's serial number, and its vertical walls sorted by x: their xs, and each one's (top, bottom)."""
    known = _MODELS.get(id(model))
    if known is not None and known[0] is model:
        _MODELS.move_to_end(id(model))
        return known[1], known[2]
    ws = sorted((w for w in model.walls if w.x1 == w.x2), key=lambda w: w.x1)
    walls = ([w.x1 for w in ws], [(min(w.y1, w.y2) - 1, max(w.y1, w.y2) + 1) for w in ws])
    serial = next(_SERIALS)
    _MODELS[id(model)] = (model, serial, walls)
    _MODELS.move_to_end(id(model))
    if len(_MODELS) > MODELS_KEPT:
        _MODELS.popitem(last=False)
    return serial, walls


def fly(model: MapModel, x: float, y: float, vx: float, vy: float, skip: Platform | None = None,
        dt: float = 0.004, limit: float = 4.0) -> tuple[Platform | None, float, float]:
    """Where she lands from (x, y) moving at (vx, vy) (map units/s, y down): the first platform whose
    surface her feet cross on the way down. Walls stop her sideways. `skip` is a platform she falls
    through (dropping). Returns (platform, landing x, seconds). The start is rounded (x and y to 0.01
    units, the speeds to 0.1 units/s) and results are cached by it: chains try the same jumps many times
    over."""
    return flight(model, x, y, vx, vy, skip, dt, limit)[:3]


def flight(model: MapModel, x: float, y: float, vx: float, vy: float, skip: Platform | None = None,
           dt: float = 0.004, limit: float = 4.0) -> tuple[Platform | None, float, float, float]:
    """fly's landing, and her sideways speed as she lands: what she left with (with the arrow held, the
    air does nothing to it), or 0 after a wall stopped her. On the ground she slides on from it, or runs
    on."""
    serial, walls = _known(model)
    x, y, vx, vy = round(x, 2), round(y, 2), round(vx, 1), round(vy, 1)
    key = (serial, x, y, vx, vy, None if skip is None else skip.id, dt, limit)
    result = _FLY_CACHE.get(key)
    if result is not None:
        _FLY_CACHE.move_to_end(key)
        return result
    result = _fly(model, walls, x, y, vx, vy, skip, dt, limit)
    _FLY_CACHE[key] = result
    if len(_FLY_CACHE) > FLY_CACHE_SIZE:
        _FLY_CACHE.popitem(last=False)
    return result


def _fly(model: MapModel, walls: tuple[list[float], list[tuple[float, float]]], x: float, y: float, vx: float,
         vy: float, skip: Platform | None, dt: float, limit: float) -> tuple[Platform | None, float, float, float]:
    """flight, worked out (not cached)."""
    wall_xs, wall_ys = walls
    # Only platforms she could reach in the time: within her sideways travel, at or below the top
    # of the jump.
    reach = abs(vx) * limit + 1
    top = y - (vy * vy / (2 * GRAVITY) if vy < 0 else 0) - 1
    plats = [q for q in model.platforms if q is not skip and q.right >= x - reach and q.left <= x + reach
             and max(q.y_at(q.left), q.y_at(q.right)) >= top]
    t = 0.0
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
                    return q, nx, t + dt, vx
        x, y, vy, t = nx, ny, nvy, t + dt
    return None, x, t, vx


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
    """How long to hold an arrow to walk `units` map units and come to rest there (body.hold_for: 39 ms
    for 5 units, 90 for 20, 284 for 50), MIN_HOLD_MS at least."""
    return max(MIN_HOLD_MS, body.hold_for(units, SHOE, LAGS))


def run_ms(units: float) -> int:
    """How long after pressing the arrow (from standing) to press jump to lift off `units` along, still
    running; never in the kick's blind spot, though (body.jump_window_ok: keyed there, the jump finds her
    just below full speed and leaves at 119-149; keyed 93 ms after the arrow, 22% leave below 134, the 10%
    slower that route.sure_landing allows). So a run-up shorter than about 12 units is keyed about 120 ms
    in and lifts her off up to 11 units along, at full speed: within the 12 units early or late every
    route jump allows."""
    if units <= 0:
        return 0
    t = round(units / WALK_UNITS_PER_MS + RUN_LAG_MS)
    while not body.jump_window_ok(t, SHOE, LAGS):
        t += 1
    return t
