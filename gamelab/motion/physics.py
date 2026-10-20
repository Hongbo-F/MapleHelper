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

from maplehelper import config

from . import body
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
# Her walk and jump in the planner's terms, from the engine with the median lags (body.Lags; the 60 and 30
# fps recordings). From standing the game sees an arrow 60 ms after the key and speeds her up at 1400
# units/s² to full speed, so after t s she's 149 (t - 0.113) units along. Let go, she runs on until the
# game sees the release (72 ms), then brakes at 800 units/s². A jump lifts off 67 ms after its key (18
# jumps: 45-95 ms). A jump key pressed while she's still in the air is ignored, and holding it doesn't jump
# again on landing.
RUN_LAG = LAGS.on + WALK_SPEED / (2 * SHOE.acc)  # 0.113 s: a standing start lags full-speed running by this much
# ... and a start from running the other way at full speed lags a standing one by this much more: braking
# and turning at 1400 units/s² (body.run_time; the decompile's, unmeasured). It was 0.08 s, from a loose
# measurement. Nothing plans with it now (flow turns her by body.run_time): it is kept for its name.
TURN_EXTRA = body.run_time(100.0, -WALK_SPEED, 1, SHOE) - body.run_time(100.0, 0.0, 1, SHOE)  # 0.16 s
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
# A jump from standing, keyed t ms after the arrow, leaves at full speed only outside the kick's blind spot
# (body.jump_window_ok): keyed 35-52 ms after it, the kick takes her from the 30-80 units/s she has reached
# to full speed; keyed 118 ms or more after it, she has reached full speed herself; in between she leaves
# at 119-149 (three of the five jump-backs recorded at 60 fps, keyed 80 ms after the arrow, left at 121-127).
# Full speed by running rests on the walk the recordings fit (1400 units/s², the lags); full speed by the
# kick from a near standstill rests on the decompile and one recording (jb_k190: the arrow seen a step
# before the jump, at 23 units/s, and it left at 142). So the planner times jumps from standing for the
# first only, until an in-game sweep of the gap (probe P1) shows the kick: True lets it key them 35-52 ms
# after the arrow too, lifting off within 2 units of where she stands, and keys the jump-back 40 ms after its
# arrow (control.actions.JB_RUN_MS), not 130.
KICK_JUMPS = False
# ms after the arrow (from standing) at which the jump, by the median lags, finds her at the kick's limit
# (0.8 × her walking speed): the blind spot is around and after it (78).
_KICK_ENDS_MS = (SHOE.kick * SHOE.vmax / SHOE.acc - (LAGS.jump - LAGS.on)) * 1000
# Attacks from a jump (control.actions times her key plans with these; attack.reach works out what such an
# attack can hit). An attack picks its target when it starts, about JUMP_LAG after its key, the same input
# lag as a jump's lift-off: so when it starts she has been in the air for about as long as the attack key
# came after the jump key.
JUMP_ATTACK_MS = 40        # a jump throw's attack key comes this long after the jump key. In game on 2026-10-20,
                           # at a snail 20-60 units away: keyed 100 ms after the jump (the macro's F1 combo), the
                           # star went over it; keyed 40 ms after, a star hit it for 78 and killed it
JUMP_BACK_ATTACK_MS = 85   # ... and a jump back's: right after she turns in mid-air (actions.JB_THROW_MS 215
                           # less JB_RUN_MS 130, when the jump key goes down)
ATTACK_DECIDE_LAG_MS = 11  # it can start up to this much later than that (60 fps recordings jump_throw_0 and
                           # _1, keys 100 ms apart: her attack pose first shows 91-110 ms after lift-off)
# fly's landings, the FLY_CACHE_SIZE most recently used. Each is worked out from its start rounded as its
# key is (x and y to 0.01 units, speeds to 0.1 units/s), so it is a function of the key alone: which call
# came first, or which landings were dropped, can't change an answer. (Worked out from the exact start, a
# landing cached by one call was handed to every later one that rounded the same, so the simulator's
# scenarios each had to start from the same cache to give the same trace.) On this map the planner's own
# jumps and falls are 3,128 landings (about 1.4 MB), and the 8-minute simulated hunt loop loop_run adds none.
FLY_CACHE_SIZE = 20_000  # landings (about 9 MB when full)
MODELS_KEPT = 4          # maps fly knows at a time (their walls, and the serial numbers keying their landings)
_FLY_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
# The last MODELS_KEPT models fly was used on, by id(model): each kept with its serial number (its landings'
# key) and the walls that stop her by foothold group (body.walls_for). Keyed by id(model) alone, a freed
# model's id was reused by the next one, which then got the old one's walls and landings (tests failed
# depending on their order). A model is kept while it's in here, so its id can't be taken, and a model seen
# again after it was let go gets a new serial: no landing cached for another model can ever be handed to it.
_MODELS: OrderedDict[int, tuple[MapModel, int, dict]] = OrderedDict()
_SERIALS = itertools.count()


def _known(model: MapModel, group) -> tuple[int, body.Walls]:
    """The model's serial number, and the vertical walls that stop her with foothold `group`, sorted by x:
    their xs, and each one's (top, bottom)."""
    known = _MODELS.get(id(model))
    if known is None or known[0] is not model:
        known = _MODELS[id(model)] = (model, next(_SERIALS), {})
    _MODELS.move_to_end(id(model))
    if len(_MODELS) > MODELS_KEPT:
        _MODELS.popitem(last=False)
    walls = known[2].get(group)
    if walls is None:
        walls = known[2][group] = body.vertical_walls(body.walls_for(model, group))
    return known[1], walls


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
    serial, walls = _known(model, group)
    x, y, vx, vy = round(x, 2), round(y, 2), round(vx, 1), round(vy, 1)
    if arrow is None:
        arrow = (vx > 0) - (vx < 0)
    key = (serial, group, x, y, vx, vy, arrow, None if skip is None else skip.id, dt, limit)
    result = _FLY_CACHE.get(key)
    if result is not None:
        _FLY_CACHE.move_to_end(key)
        return result
    f = body.fly(model, x, y, vx, vy, shoe=SHOE, arrow=arrow, skip=skip, walls=walls, dt=dt, limit=limit)
    result = (f.platform, f.land_x, f.seconds, f.vx_land)
    _FLY_CACHE[key] = result
    if len(_FLY_CACHE) > FLY_CACHE_SIZE:
        _FLY_CACHE.popitem(last=False)
    return result


# A route jump has to land on its platform at this share of her walking speed too: a jump the mover keys
# from standing (control.mover, MIN_RUN on) leaves slower than 0.85 × 149 (127) only with the lags 2.8 sd
# off (0.3%), and one keyed in the kick's blind spot leaves at 119-149 (0.8-1 ×).
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
    """How long to hold an arrow to walk `units` map units and come to rest there (body.hold_for: 39 ms
    for 5 units, 90 for 20, 284 for 50), MIN_HOLD_MS at least."""
    return max(MIN_HOLD_MS, body.hold_for(units, SHOE, LAGS))


def sure_jump_ms(gap: float) -> float:
    """The soonest a jump from standing can be keyed, `gap` ms after the arrow or later, and surely leave
    the ground at full speed (KICK_JUMPS above): 118 ms after the arrow for any gap up to that, the gap
    itself from there on."""
    def sure(t: float) -> bool:
        return body.jump_window_ok(t, SHOE, LAGS) and (KICK_JUMPS or t >= _KICK_ENDS_MS)
    if gap >= 0 and sure(gap):
        return gap
    t = max(0, math.ceil(gap))
    while not sure(t):
        t += 1
    return t


def run_ms(units: float) -> int:
    """How long after pressing the arrow (from standing) to press jump to lift off `units` along, still
    running; never in the kick's blind spot, though (sure_jump_ms: keyed there, the jump finds her just
    below full speed and leaves at 119-149; keyed 93 ms after the arrow, 22% leave below 134, the 10% slower
    that route.sure_landing allows). So a run-up shorter than about 12 units is keyed 118 ms in and lifts
    her off up to 11 units along, at full speed: within the 12 units early or late every route jump
    allows."""
    if units <= 0:
        return 0
    return sure_jump_ms(round(units / WALK_UNITS_PER_MS + RUN_LAG_MS))
