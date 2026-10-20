"""Her physics: the game's jump, gravity and fall speeds, her measured walking (with its lags and
slides) and climbing (and how long to hold an arrow to walk or run a distance), and where a jump or
fall lands (simulated against the map's platforms and walls). All in map units.
"""

import bisect
import itertools
import math
from collections import OrderedDict

from ..localization.mapdata import MapModel, Platform
from ..screen import SCREEN_SCALE


WALK_SPEED = 149.0       # map units/s: her measured walk (209 screen px/s at 1.40625 px per unit)
CLIMB_SPEED = 117.0      # map units/s: measured 0.165 screen px/ms
JUMP_SPEED = 555.0       # physics: jumpSpeed
GRAVITY = 2000.0         # physics: gravityAcc
FALL_SPEED = 670.0       # physics: fallSpeed (terminal)
JUMP_HEIGHT = JUMP_SPEED ** 2 / (2 * GRAVITY)   # 77 map units
ROPE_REACH = 95          # she can grab a rope that ends this far above her feet (jump + arms)
LAND_MARGIN = 18         # land at least this far onto a platform
# Measured on platform 34 with the camera fixed (map edge), 60 fps: from standing, holding an arrow
# puts her at 148.5 (t - 0.12) units after t s; a jump lifts off 55 ms after the key; after the arrow
# is released she runs on ~0.1 s (input lag) and slides, stopping ~27 units on, ~0.3 s later. A jump
# key pressed while she's still in the air is ignored, and holding it doesn't jump again on landing.
RUN_LAG = 0.12       # s: a standing start lags full-speed running by this much
TURN_EXTRA = 0.08    # s more when she was running the other way
JUMP_LAG = 0.055     # s from the jump key to leaving the ground
DROP_LAG = 0.28      # s from the jump key (down held) to falling through (0.28-0.34 in 4 drops on the
                     # minimap at 60 fps, whether or not she had just stopped running)
STOP_RUN = 27.0      # units she goes on after the arrow is released at full speed ...
STOP_S = 0.3         # ... until she stands still
LAND_SLIDE = 13.0    # units she slides on landing with no arrow held
# Measured: holding an arrow for t ms moves her ~(12 + 0.209 t) screen px including the slide after
# release, i.e. 8.5 + 0.149 t map units. Shorter than ~30 ms barely registers.
WALK_UNITS_PER_MS = 0.209 / SCREEN_SCALE
WALK_UNITS_FIXED = 12 / SCREEN_SCALE
MIN_HOLD_MS = 30
# When to press jump (ms after the arrow, from standing) to lift off `units` along: she's at
# 0.1485 (t - 120) units after t ms, and lifts off 55 ms after the key (JUMP_LAG), so
# RUN_LAG_MS = RUN_LAG - JUMP_LAG (measured at 60 fps).
RUN_LAG_MS = 65
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
         vy: float, skip: Platform | None, dt: float, limit: float) -> tuple[Platform | None, float, float]:
    """fly, worked out (not cached)."""
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
                    return q, nx, t + dt
        x, y, vy, t = nx, ny, nvy, t + dt
    return None, x, t


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
    """How long to hold an arrow to walk `units` map units and come to rest there."""
    return max(MIN_HOLD_MS, round((units - WALK_UNITS_FIXED) / WALK_UNITS_PER_MS))


def run_ms(units: float) -> int:
    """How long after pressing the arrow (from standing) she's `units` along, still running."""
    return round(units / WALK_UNITS_PER_MS + RUN_LAG_MS) if units > 0 else 0
