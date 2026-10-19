"""Her physics: the game's jump, gravity and fall speeds, her measured walking (with its lags and
slides) and climbing (and how long to hold an arrow to walk or run a distance), and where a jump or
fall lands (simulated against the map's platforms and walls). All in map units.
"""

import bisect
import math

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
_FLY_CACHE: dict[tuple, tuple] = {}  # fly's results by its arguments (the model by id)
_WALLS: dict[int, tuple] = {}        # _walls' results by id(model)
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
    """Seconds to fall dy units from rest, by gravity alone (0 for dy <= 0)."""
    return math.sqrt(2 * max(0.0, dy) / GRAVITY)


def hold_ms(units: float) -> int:
    """How long to hold an arrow to walk `units` map units and come to rest there."""
    return max(MIN_HOLD_MS, round((units - WALK_UNITS_FIXED) / WALK_UNITS_PER_MS))


def run_ms(units: float) -> int:
    """How long after pressing the arrow (from standing) she's `units` along, still running."""
    return round(units / WALK_UNITS_PER_MS + RUN_LAG_MS) if units > 0 else 0
