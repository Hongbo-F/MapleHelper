"""Her physics as the planner takes it: the game's jump, gravity and fall speeds, her walking (with its
lags and slides) and climbing (and how long to hold an arrow to walk or run a distance), and where a
jump or fall lands (flown by motion.body's engine against the map's platforms and the walls of her
foothold group, as the simulator flies her). All in map units. The walk, the jump and their lags come
from motion.body, the engine that moves her as the game does.
"""

import bisect
import itertools
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, replace

from maplehelper import config

from . import body
from .. import character, native
from ..localization.mapdata import MapModel, Platform
from ..paths import CONFIG

# Her stats (config.toml [character]), and how the game moves her with them: motion.body's engine, its
# shoe from the game's Physics.img and its lags measured. With FIXED_STATS set in the environment, the
# measured defaults (config.Character()) whatever the file says: the tests set it (tests/conftest.py), and
# the simulator's golden traces and harness restart themselves with it, so editing [character] for her gear
# changes neither what they check nor what they compare. (`python -m gamelab sim`, for watching, takes the
# file's, as the hunt does.) Hers are the character's this process plays (gamelab.character: the thief unless
# --character, GAMELAB_CHARACTER, `gamelab character use` or [characters] default names another; under FIXED_STATS
# only GAMELAB_CHARACTER, and that character's class's fixed copy).
FIXED_STATS = "GAMELAB_FIXED_STATS"


def stats(path=CONFIG, environ=os.environ) -> config.Character:
    """Her stats: config.toml's [character], or the measured defaults when FIXED_STATS is set (the thief's; another
    character's own, gamelab.character.stats)."""
    return character.stats(path, environ)


def shoe_of(stats: config.Character) -> body.Shoe:
    """A character's shoe from her stats: Physics.img's (body.PHYSICS_IMG), her SPEED with the Speed this client adds
    to a character's (body.SPEED_BONUS: she walks 125 x (SPEED + 20)/100) and her JUMP."""
    return body.Shoe.from_physics(body.PHYSICS_IMG, stats.speed, stats.jump, bonus=body.SPEED_BONUS)


CHARACTER = character.active().stats  # (who plays, resolved once: stats() as the process has it, her profile's)
SHOE = shoe_of(CHARACTER)
LAGS = body.Lags()

WALK_SPEED = SHOE.vmax   # map units/s: 125 x (SPEED + 20)/100 (body.SPEED_BONUS): the magician's 150 at her stat
                         # window's 100 (she walks 149.4), the thief's 149 at her profile's 99.2 (she walks 151.2)
CLIMB_SPEED = 117.0      # map units/s: measured 0.165 screen px/ms (the thief at 1920x1080; the magician climbs 119.9)


def record() -> dict:
    """Her physics as this process plans her moves, for the telemetry 'session' record (from 2026-11-02, so a run's
    records say what walk its timed holds were planned at: analysis.walkfit measures her walk against it): who plays
    (`character`, `class`), her stats' SPEED and JUMP (%), the Speed the client adds to hers (`bonus`: body.SPEED_BONUS,
    from 2026-11-02's walk fix) and the walk and climb they give (units/s)."""
    c = character.resolved() or character.active()
    return {"character": c.id, "class": c.klass, "speed": CHARACTER.speed, "jump": CHARACTER.jump,
            "bonus": body.SPEED_BONUS, "walk": round(WALK_SPEED, 2), "climb": CLIMB_SPEED}
JUMP_SPEED = SHOE.vjump  # physics: jumpSpeed x JUMP/100 (555)
GRAVITY = SHOE.gravity   # physics: gravityAcc (2000)
FALL_SPEED = SHOE.fall   # physics: fallSpeed (670, terminal)
JUMP_HEIGHT = JUMP_SPEED ** 2 / (2 * GRAVITY)   # 77 map units
ROPE_REACH = 95          # she can grab a rope that ends this far above her feet (jump + arms: a guess, never measured;
                         # the highest grab tried in game was a rope 69 up)
# ... and surely this far: HeavenClient (the open-source v83 client) grabs a rope when a point 5 above her feet is on it
# (Gameplay/MapleMap/MapInfo.cpp, Ladder::inrange, read 2026-10-24), 82 up at the top of her jump. Between the two,
# which the game does isn't known (route.GRAB_CHANCE).
GRAB_SURE = JUMP_HEIGHT + 5
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
# Her knock when a monster touches her (from sim.world, where it was fitted: hits.jsonl's pushes on her own level, quartiles
# 21 / 58 / 87 units; ~40 units in the air for 0.22 s and a ~19-unit slide on landing): away from the monster at KNOCK_VX,
# up at KNOCK_VY (control.navigator: when the state after a touch can have her on the ground again, KNOCK_AIR_S).
KNOCK_VX, KNOCK_VY = 180.0, 220.0
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
JUMP_BACK_ATTACK_MS = 85   # ... and a jump back's at the soonest: right after she turns in mid-air (the arrow toward
                           # the monster AIR_TURN_MS after the jump key), what decides which kinds one can pick at all
                           # (attack.stance); keyed AIR_TURN_LEAD_MS after the arrow wherever the kind allows
                           # (AIR_TURN_ATTACK_MS, below: reach.turn_key_ms)
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
# key), the walls that stop her (body.walls_by: the game's rule, or every wall), and the same laid out for the flight
# in C (native: the platforms, and the walls; made the first time she flies there with the DLL in use).
# Keyed by id(model) alone, a freed model's id was reused by the next one, which then got the old one's walls
# and landings (tests failed depending on their order). A model is kept while it's in here, so its id can't be
# taken, and a model seen again after it was let go gets a new serial: no landing cached for another model can
# ever be handed to it.
_MODELS: OrderedDict[int, tuple[MapModel, int, dict, dict]] = OrderedDict()
_SERIALS = itertools.count()
_SHOES: dict[body.Shoe, object] = {}  # each shoe as the C flight takes it (native.shoe_of)


def _rule(group):
    """The key of the walls that stop her flying from foothold `group`: every wall (body.EVERY_WALL, or without
    body.JOINED_WALLS), or the game's rule (body.wall_table, with or without body.ZERO_WALLS), whatever her group."""
    if group == body.EVERY_WALL or not body.JOINED_WALLS:
        return body.EVERY_WALL
    return "rule", body.ZERO_WALLS


def _known(model: MapModel, group) -> tuple[int, body.Walls, dict]:
    """The model's serial number, the walls that stop her flying from foothold `group` (body.walls_by, as body.fly
    takes them), and its layouts for the C flight (_native_flight's)."""
    known = _MODELS.get(id(model))
    if known is None or known[0] is not model:
        known = _MODELS[id(model)] = (model, next(_SERIALS), {}, {})
    _MODELS.move_to_end(id(model))
    if len(_MODELS) > MODELS_KEPT:
        _MODELS.popitem(last=False)
    rule = _rule(group)
    walls = known[2].get(rule)
    if walls is None:
        walls = known[2][rule] = body.walls_by(model, body.EVERY_WALL if rule == body.EVERY_WALL else group)
    return known[1], walls, known[3]


def fly(model: MapModel, x: float, y: float, vx: float, vy: float, skip: Platform | None = None,
        dt: float = body.TICK, limit: float = 4.0, group=body.EVERY_WALL,
        arrow: int | None = None) -> tuple[Platform | None, float, float]:
    """Where she lands from (x, y) moving at (vx, vy) (map units/s, y down): the first platform whose
    surface her feet cross on the way down. Walls stop her sideways: by the game's rule (body.wall_table, for
    any `group`: the platform's she leaves, Platform.group, or body.ROPE_GROUP off a rope, as callers say it), or
    every wall for body.EVERY_WALL, the default. `skip` is a platform she falls through (dropping). Returns (platform,
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
    group = _rule(group)  # (one set of landings for every group: the walls are the game's rule's)
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
        m = laid_out["platforms"] = native.Map(model.places)  # (a twin is never landed on: MapModel.twins)
    w = laid_out.get(group)
    if w is None:
        w = laid_out[group] = native.Walls(walls)
    shoe = _SHOES.get(SHOE)
    if shoe is None:
        shoe = _SHOES[SHOE] = native.shoe_of(SHOE)
    s = -1 if skip is None else next((i for i, q in enumerate(model.places) if q is skip), -1)
    found = native.fly(m, w, x, y, vx, vy, int(arrow), dt, limit, shoe, s)
    if found is None:
        return None
    q, lx, t, vl, same_x, same_vx = found
    return None if q < 0 else model.places[q], x if same_x else lx, t, vx if same_vx else vl


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
    return max(0.0, rise_after(t)) if t > 0 else 0.0


# --- her jumps over time ------------------------------------------------------------------------------
# Where she is, from the jump key on, in each kind of jump a key plan makes: what the attack from a jump picks
# (attack.reach.shots: where she is when it starts), and where she lands. The arc is the engine's (body.air_step:
# gravity from JUMP_SPEED up, capped at FALL_SPEED; with the arrow held the way she flies her speed stays as it was,
# with none the air drags her 1 unit/s² sideways, under 0.2 units over a jump), in closed form: the engine's step
# moves her by the mean of two speeds, exact under a steady pull, so the two agree to rounding but for where a wall
# stops her (`pose`, as body.fly). An attack in the air leaves her arc as it is: the 60 fps jump throws jump_throw_0
# and _1 (the attack 100 ms after the jump key) rose 78-79 units and came down within 0.1 units of where they left,
# and the jump backs jb_k190 and jb_k290 (the attack 110 and 210 ms after the jump key, mid-flight) came down 78.7
# and 78.3 units away, as the engine flies them without one (rec_measure, D:/MapleHelper-scratch/jump-attack). She
# lands where the arc meets a floor, and an attack still playing then stops her there (jump backs and jump throws
# slid 1.4-2.9 units in game: sim.world; with none she slides on, LAND_SLIDE).
APEX_S = JUMP_SPEED / GRAVITY  # 0.2775 s from lift-off to the top of her jump, JUMP_HEIGHT (77.0) over where she left ...
AIR_S = 2 * APEX_S             # ... and 0.555 s until she is back down at that height (a jump on flat ground)
CAP_S = (JUMP_SPEED + FALL_SPEED) / GRAVITY  # 0.6125 s after lift-off she falls at the cap (FALL_SPEED), 35 under
KICK_SPEED = SHOE.kick * WALK_SPEED  # 119: leaving from standing with the arrow the game sees as it sees the jump
# A jump forward from standing: the jump key this long after the arrow, in the kick's window by the median lags
# (body.jump_window_ok, 40-52 ms): one or two of the game's steps of the arrow, then the kick to full speed (149), and
# 4.5% leave at 126 by the median lags (three steps: body.short_takeoff_chance). In game (60 fps, the kick sweep of
# 2026-10-21, kick2_40_l and _r) jumps keyed 40 ms after the arrow came down 82.7 and 85.1 units from where they lifted
# off (149 x AIR_S is 82.7), within 2 units of where she stood; keyed 60 ms after it, 71 (kick2_60_r: three steps,
# 126). Sooner, the arrow can come after the jump's step: from standing she leaves straight up, and the arrow in the
# air does nothing (87% of jumps keyed with the arrow leave short of full speed by the median lags).
FORWARD_JUMP_MS = 40
# A jump in place that has to turn her: the arrow toward the target tapped this long after the jump key, once she's off
# the ground (the jump is seen at its step 50 ms after its key, the arrow 51 ms after its own: 60 ms later it's seen
# 61 ms after the jump, two steps at least, but for 0.2% of jumps by the spread of the two). Seen before the jump's
# step, the arrow would kick her off at 119 toward the monster (tapped 20 ms after the jump key, by the engine over
# the game's steps a third of such jumps go 75 units forward); seen after it, it turns her in the air and steers
# her at most 8.9 units/s: by the engine she comes down 3.8 units on (2.7-4.3 over the steps' phase; an attack
# playing then stops her, without the slide). The jump backs turn so, the arrow 60 ms after their jump key
# (control.actions.JB_TURN_MS). Tapped before the jump instead (as until 2026-10-25), the tap walks her: jumped 80 ms
# after it, she leaves at 18-84 units/s and comes down 12-53 units on (33 on average); 205 ms after it
# (actions.JUMP_THROW_TURN_MS), standing again, she comes down 5.3 on (1.8-7.0), the attack 245 ms after
# the tap (D:/MapleHelper-scratch/ov-jump/carry.py).
AIR_TURN_MS = 60
# ... and the attack key this long after that arrow at the soonest: live on 2026-10-25 (the hunt of 23:43), with the
# attack 25 ms after the arrow (85 ms after the jump key), 10 of the 28 jumps that turned her in the air threw the wrong
# way by her frames (36%; none unreadable) and none of those 10 killed, where 10 of the 18 that turned did; the throws
# from the ground that tapped first, their attack 80 ms after the arrow, turned on all 68 readable ones
# (D:/MapleHelper-scratch/analysis-1025/jumps-perception-skeptic/airturn2.py). So 60: the attack 120 ms after the jump
# key (AIR_TURN_ATTACK_MS), where her air attack still picks a Horny or a Green Mushroom (51 tall; reach.air_hits up to
# about 145 ms) but not a pig (38) or a slime (40), picked only to about 105 and 110 ms: at those the attack goes as
# late as it still picks them (reach.turn_key_ms), sooner than this after the arrow.
AIR_TURN_LEAD_MS = 60
AIR_TURN_ATTACK_MS = AIR_TURN_MS + AIR_TURN_LEAD_MS
# A jump throw that taps the arrow to turn her on the ground first (control.actions.Actions.jump_throw with its air
# turn off) jumps this long after the tap, once the tap's walk has stopped (control.actions has why; attack.stance times
# the attack's start by it: a tap first keeps the attack's key 40 ms after the jump key, where an air turn's comes 120 ms
# after it, her feet 30 units higher).
JUMP_THROW_TURN_MS = 205


def rise_after(t: float) -> float:
    """How high her feet are over where she lifted off, t s after lift-off (negative once she's below it): the
    game's arc, up at JUMP_SPEED under GRAVITY, falling at FALL_SPEED from CAP_S on."""
    if t <= CAP_S:
        return JUMP_SPEED * t - GRAVITY * t * t / 2
    return JUMP_SPEED * CAP_S - GRAVITY * CAP_S * CAP_S / 2 - FALL_SPEED * (t - CAP_S)


def lift_time(rise: float) -> tuple[float, float] | None:
    """When (s after lift-off) her feet are `rise` over where she lifted off, on the way up and on the way down
    (None if her jump doesn't get that high: JUMP_HEIGHT)."""
    disc = JUMP_SPEED ** 2 - 2 * GRAVITY * rise
    if disc < 0:
        return None
    return (JUMP_SPEED - math.sqrt(disc)) / GRAVITY, (JUMP_SPEED + math.sqrt(disc)) / GRAVITY


@dataclass(frozen=True)
class Jump:
    """One kind of jump as a key plan makes it, from standing: how she leaves the ground, timed from the jump key.
    `kind`: 'up' (in place: no arrow but a turn's tap in the air), 'forward' (the arrow held toward `way`, the jump
    FORWARD_JUMP_MS after it: the kick), 'back' (control.actions' jump back: away from what she throws at, the jump
    actions.JB_RUN_MS after the arrow), 'run' (running already at full speed). `way`: which way she goes (+1 right,
    -1 left, 0 up); `vx`: her sideways speed leaving the ground; `lead_s`: s from the plan's first key (the arrow) to
    the jump key; `run`: how far along `way` she has gone from where she stood by lift-off (JUMP_LAG after the jump
    key)."""
    kind: str
    way: int
    vx: float
    lead_s: float = 0.0
    run: float = 0.0

    def arc(self, t: float) -> tuple[float, float]:
        """Where her feet are t s after the jump key, against where she stood: (how far along x, signed; how high),
        on flat ground and with nothing in her way (pose has the map)."""
        if t <= JUMP_LAG:
            return self.way * self.run * max(0.0, (t + self.lead_s) / (JUMP_LAG + self.lead_s)), 0.0
        tau = t - JUMP_LAG
        return self.way * self.run + self.vx * tau, rise_after(tau)


def jump_of(kind: str, way: int = 0, lead_ms: float | None = None) -> Jump:
    """The jump `kind` makes toward `way` (+1 right, -1 left; 'up' none), by the engine with the median lags: her speed
    and run at lift-off from the arrow's lead (body.run_x, body.takeoff). 'forward' leaves at full speed 1.1 units on
    (the kick: FORWARD_JUMP_MS after the arrow); 'back', JB_RUN_MS (150) after its arrow, at full speed 14.3 units on;
    'run' at full speed from where the jump key goes."""
    if kind == "up" or not way:
        return Jump("up", 0, 0.0)
    if kind == "run":
        return Jump("run", way, way * WALK_SPEED, 0.0, WALK_SPEED * JUMP_LAG)
    lead = (FORWARD_JUMP_MS if kind == "forward" else BLIND_ENDS_MS) if lead_ms is None else lead_ms
    tau = lead / 1000 + JUMP_LAG - LAGS.on  # how long the game has seen the arrow held when she lifts off
    v = body.takeoff(body.run_speed(tau, 0.0, way, SHOE), way, SHOE)
    return Jump(kind, way, v, lead / 1000, abs(body.run_x(tau, 0.0, way, SHOE)))


LANDING_SLOP = 12.0  # how many units early or late a jump from standing may lift off (sure_landing's default)


def jump_variants(j: Jump, slop: float = LANDING_SLOP) -> list[Jump]:
    """The ways jump `j` may go otherwise, as sure_landing tries a route jump's: lifting off `slop` units early or late
    along its way, 10% slower, as slow as SLOW_TAKEOFF (126: three of the game's steps of the arrow, the blind spot),
    and a jump forward from standing at KICK_SPEED too (119: the arrow seen at the jump's own step, 4% of kicks keyed
    FORWARD_JUMP_MS after it by the median lags, up to 27% within what the sweeps allow: KICK_JUMPS' reason to stay
    off). A jump in place has none (it lands where she stood)."""
    if not j.way:
        return []
    out = [replace(j, run=j.run - slop), replace(j, run=j.run + slop), replace(j, vx=j.vx * 0.9),
           replace(j, vx=j.vx * SLOW_TAKEOFF)]
    if j.kind == "forward":
        out.append(replace(j, vx=j.way * KICK_SPEED))
    return out


def jump_landing(model: MapModel, platform: Platform, x: float, j: Jump) -> tuple[Platform | None, float, float]:
    """Where jump `j` from map x on `platform` lands: (platform, x, s after lift-off), flown by the engine (flight,
    the walls of her platform's foothold group) from where she lifts off (her run first, within her platform)."""
    x0 = min(max(x + j.way * j.run, platform.left), platform.right)
    land, lx, s, _ = flight(model, x0, platform.y_at(x0), j.vx, -JUMP_SPEED, group=platform.group,
                            arrow=j.way)
    return land, lx, s


def pose(model: MapModel, platform: Platform, x: float, j: Jump, t: float,
         landing: tuple[Platform | None, float, float] | None = None) -> tuple[float, float, bool]:
    """Her feet t s after jump `j`'s key, from map x on `platform`: (map x, map y, whether she's in the air). The arc
    (Jump.arc) over where she lifts off, stopped sideways by a wall of her platform's group where her feet meet it, as
    body.fly stops her; from its landing (`landing`: jump_landing's, worked out if not given) on her feet are there."""
    landing = landing if landing is not None else jump_landing(model, platform, x, j)
    land, lx, s = landing
    if t <= JUMP_LAG:
        dx, _ = j.arc(t)
        x1 = min(max(x + dx, platform.left), platform.right)
        return x1, platform.y_at(x1), False
    x0 = min(max(x + j.way * j.run, platform.left), platform.right)
    y0 = platform.y_at(x0)
    tau = t - JUMP_LAG
    if land is not None and tau >= s:
        return lx, land.y_at(lx), False
    nx = x0 + j.vx * tau
    if j.vx:
        _, (xs, ys, ways), _ = _known(model, platform.group)
        lo, hi = sorted((x0, nx))
        i = bisect.bisect_left(xs, lo)
        hits = []
        while i < len(xs) and xs[i] <= hi:
            if xs[i] != x0:
                feet = y0 - rise_after((xs[i] - x0) / j.vx)
                if ys[i][0] <= feet <= ys[i][1] and ways[i] * j.vx >= 0:
                    hits.append(xs[i])
            i += 1
        if hits:
            nx = min(hits) if j.vx > 0 else max(hits)
    return nx, y0 - rise_after(tau), True


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
