"""The offline simulator (gamelab.sim): its physics against the planner's own (physics.fly, the measured walk),
the fake player's queueing and cancelling, the fakes put in place and taken away again (and nothing
reaching the real player or gamelab/lab), and scenarios that give the same trace every time and show
what they're there for. Synthetic maps need no game data; the real-map tests are skipped without it, and
fail (not skip) when it's there but doesn't load."""

import dataclasses
import os
import sys
import time
import types

import numpy as np
import pytest

from gamelab import paths
from gamelab.control import client
from gamelab.localization import mapdata
from gamelab.motion import body, flow, physics
from gamelab.behavior.hunter import Hunter
from gamelab.motion.physics import hold_ms
from gamelab.paths import LAB
from gamelab.sim import fakes, golden, scenarios
from gamelab.sim import world as world_mod
from gamelab.sim.fakes import Clock, FakeEyes, FakePlayer, SimViolation, Trace, patched
from gamelab.sim.world import DROP_LIFE, Noise, SimDied, SimFellOut, World, exp_pixels
from maplehelper.config import GameKeys
from maplehelper.keys import parse as K
from gamelab.attack import reach
from gamelab.behavior import hunter
from gamelab.localization import monsters
from gamelab.perception.kinds import kind_of

KEYS = GameKeys(left=K("left"), right=K("right"), jump=K("alt"), attack=K("f"), up=K("up"), down=K("down"),
                pick_up=K("z"), chair=K("d"), hp_potion=K("r"))

#   y 300  [A: 0..200]                 [D: 600..800]   (D is up a rope from C)
#   y 360           [B: 220..400]
#   y 500  [C: 0 .......................... 800] with walls rising at both ends
#   rope at x 700 from D (y 300) down to y 450: 50 above C, so it takes a jump to grab


def seg(fid, x1, y1, x2, y2, prev=0, next=0):
    return str(fid), {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "prev": prev, "next": next}


def tree():
    footholds = {"1": {
        "1": dict([seg(1, 0, 300, 100, 300, 0, 2), seg(2, 100, 300, 200, 300, 1, 0)]),
        "2": dict([seg(3, 220, 360, 400, 360)]),
        "3": dict([seg(4, 0, 400, 0, 500, 0, 5), seg(5, 0, 500, 800, 500, 4, 6), seg(6, 800, 500, 800, 400, 5, 0)]),
        "4": dict([seg(7, 600, 300, 800, 300)]),
    }}
    return {
        "info": {"VRLeft": -100, "VRTop": 0, "VRRight": 900, "VRBottom": 600},
        "foothold": footholds,
        "ladderRope": {"0": {"l": 0, "uf": 1, "x": 700, "y1": 300, "y2": 450}},
        "life": {"0": {"type": "m", "id": "0000002", "x": 300, "y": 360, "cy": 360, "rx0": 230, "rx1": 390}},
        "miniMap": {"width": 1000, "height": 600, "centerX": 100, "centerY": 0, "mag": 4},
    }


@pytest.fixture(scope="module")
def model():
    return mapdata.parse(1, tree(), "Test Map", {}, {"2": "Snail"}, {"2": 2})


def plat(model, y, x):
    return next(p for p in model.platforms if p.covers(x) and abs(p.y_at(x) - y) < 1)


def world(model, pid, x, **kw):
    kw.setdefault("monsters", None)
    return World(model, KEYS, (pid, x), **kw)


PHASES = range(0, world_mod.STEP_MS, world_mod.TICK_MS)  # every phase the game's steps can have in the world


def tap(w, t, name, hold):
    w.key(t, name, True)
    w.key(t + hold, name, False)


# --- physics -----------------------------------------------------------------------------------------

def test_her_arc_is_the_engines_and_lands_where_the_planner_says(model):
    # The world flies her by body.fly; the planner still predicts with physics.fly (its own step rule,
    # a unit higher at the top): with the arrow she jumps with held, as the planner's moves do, the same
    # platforms a unit or so apart. (Let go, a long fall at the fall cap drags her back: 2.5 units shorter
    # from B's end down to C.)
    a, b, c = plat(model, 300, 100), plat(model, 360, 300), plat(model, 500, 400)
    for x, y, vx, vy, skip in [(150, 300, 149.0, -555.0, None), (201, 299, 149.0, 0.0, None),
                               (390, 360, 149.0, -555.0, None), (300, 361, 0.0, 0.0, b),
                               (700, 500, -149.0, -555.0, None), (20, 500, -149.0, -555.0, None),
                               (100, 301, 0.0, 0.0, a)]:
        want = physics.fly(model, x, y, vx, vy, skip)
        got = body.fly(model, x, y, vx, vy, shoe=physics.SHOE, skip=skip, arrow=(vx > 0) - (vx < 0))
        assert got.platform is want[0] and got.land_x == pytest.approx(want[1], abs=1.5)
        assert got.seconds == pytest.approx(want[2], abs=0.02)
    assert c is not None
    # A jump straight up from standing in the world: body.fly's arc, step by step on the world's tick.
    w = world(model, c.id, 300.0)
    tap(w, 0.0, "alt", 0.04)
    w.advance(0.1)
    f = body.fly(model, 300.0, 500.0, 0.0, -physics.SHOE.vjump, shoe=physics.SHOE, path=True)
    assert w.flight.path == f.path and w.flight.platform is f.platform is c


def test_walking_is_the_engines(model):
    # The game acts on the arrow at its next step after it sees it, 66 ms after the key on average, and speeds
    # her up at 1400 units/s² a step at a time; let go, it acts on that 68 ms after on average and brakes her
    # at 800: at each phase of its steps body.replay's walk, and over them the closed forms (body.hold_distance,
    # and flow's run model).
    c = plat(model, 500, 400)
    for ms in (300, 600, 1000):
        rests, at_06 = [], []
        for phase in PHASES:
            w = world(model, c.id, 100.0, step_phase=phase)
            tap(w, 0.0, "right", ms / 1000)
            r = body.replay([(0.0, "right", True), (ms / 1000, "right", False)], 3.0, physics.SHOE, physics.LAGS,
                            start=0.0, phase=phase / 1000)
            path = np.array(r.points)
            for t in (0.05, 0.1, 0.15, 0.3, 0.6):
                w.advance(t)
                assert w.x - 100 == pytest.approx(np.interp(t, path[:, 0], path[:, 1]), abs=1e-6)
            at_06.append(w.x - 100)
            w.advance(3.0)
            assert w.vx == 0 and w.mode == "ground"
            rests.append(w.x - 100)
        if ms == 1000:
            assert np.mean(at_06) == pytest.approx(physics.WALK_SPEED * (0.6 - physics.RUN_LAG), abs=0.5)
        assert np.mean(rests) == pytest.approx(body.hold_distance(ms, physics.SHOE, physics.LAGS), abs=0.5)
        assert abs(hold_ms(np.mean(rests)) - ms) <= 10  # physics.hold_ms: the hold for what she walked


def test_taps_move_her_as_long_as_the_game_sees_them_held(model):
    c = plat(model, 500, 400)
    for hold in (0.107, 0.05, 0.02):  # 20.8, 5.6 and 1.3 units by the engine, over the game's steps
        moved = []
        for phase in PHASES:
            w = world(model, c.id, 100.0, step_phase=phase)
            tap(w, 0.0, "left", hold)
            w.advance(2.0)
            assert w.facing == "left"
            moved.append(100 - w.x)
        # (keyed at 0, the press is seen on the tick 4 ms after its lag: a hold of 105 ms, not 109)
        assert np.mean(moved) == pytest.approx(body.hold_distance(hold * 1000, physics.SHOE, physics.LAGS), abs=1.2)
        assert len({round(v, 2) for v in moved}) == 2  # whole steps of it, one more or one less
    # With the lags spread (three times as measured, to see it in a few seeds), a tap the game sees let go
    # before it saw it pressed is lost: it turns her without moving her.
    moved = []
    for seed in range(30):
        w = world(model, c.id, 100.0, seed=seed, noise=Noise(lag_sd=3.0))
        tap(w, 0.0, "left", 0.02)
        w.advance(2.0)
        assert w.facing == "left"
        moved.append(100 - w.x)
    assert min(moved) == 0 and max(moved) > 3


def test_a_jump_from_standing_gets_the_kick_unless_she_is_nearly_at_full_speed(model):
    # Her sideways speed leaving the ground (body.takeoff) with the arrow pressed `gap` s before the jump key,
    # at each phase of the game's steps: it acts on the arrow 66 ms after its key on average and on the jump
    # 65 ms after its own, at its steps, which have her at 42, 84, 126 and 149 after one to four of the arrow.
    # Lifted off by the arrow's own step she leaves at 119 (or straight up, if the jump's step came first),
    # after one or two kicked to full speed, after three at 126 (as in game: 3 of 4 keyed 100 ms after it).
    c = plat(model, 500, 400)
    for gap, speeds in ((None, {0.0}), (0.0, {0.0, 119.2}), (0.045, {149.0}), (0.08, {126.0, 149.0}),
                        (0.2, {149.0})):
        seen = set()
        for phase in PHASES:
            w = world(model, c.id, 100.0, step_phase=phase)
            if gap is not None:
                w.key(0.0, "right", True)
            tap(w, gap or 0.0, "alt", 0.04)
            w.advance((gap or 0.0) + 0.12)
            assert w.mode == "air", (gap, phase)
            seen.add(round(w.flight.speeds[0][0], 1))
        assert seen == speeds, gap
    # Keyed 80 ms after the arrow, as the jump-back was, it is two outcomes: kicked to full speed, or not
    # kicked, three steps in (the five recorded: 121, 124, 127, 142 and 146 units/s).
    speeds = []
    for seed in range(40):
        w = world(model, c.id, 100.0, seed=seed, noise=Noise(lag_sd=1.0))
        w.key(0.0, "right", True)
        tap(w, 0.08, "alt", 0.04)
        w.advance(0.3)
        speeds.append(w.flight.speeds[0][0])
    assert min(speeds) < 130 and max(speeds) == 149.0 and sum(v == 149.0 for v in speeds) >= 10


def test_in_the_air_the_opposite_arrow_brakes_her_and_her_own_does_nothing(model):
    c = plat(model, 500, 400)

    def jump(then):
        w = world(model, c.id, 100.0, step_phase=0)
        w.key(0.0, "right", True)
        tap(w, 0.5, "alt", 0.04)    # at full speed: lifts off at the step at 0.57, lands about 0.555 s later
        then(w)
        w.advance(1.0)
        assert w.mode == "air"
        return w

    held = jump(lambda w: None)
    braked = jump(lambda w: tap(w, 0.7, "left", 0.04))
    let_go = jump(lambda w: w.key(0.55, "right", False))
    assert held.vx == 149.0  # the arrow she flies with does nothing at speed
    # left acted on from the step at 0.78 s to the one at 0.81: 30 ms at 200 units/s²
    assert held.vx - braked.vx == pytest.approx(200 * 0.030, abs=0.1) and braked.facing == "left"
    assert held.vx - let_go.vx == pytest.approx(1 * (1.0 - 0.63), abs=0.01)  # no arrow: 1 unit/s² of drag


def test_landing_while_an_attack_plays_stops_her(model):
    c = plat(model, 500, 400)

    def jump(attack: bool):
        w = world(model, c.id, 100.0, step_phase=0)
        w.key(0.0, "right", True)
        tap(w, 0.5, "alt", 0.04)
        w.key(0.6, "right", False)  # let go in the air: she'd slide 13.9 on landing
        if attack:
            tap(w, 0.6, "f", 0.04)  # its animation plays until 1.33 s; she lands at 1.13
        w.advance(1.12)
        assert w.mode == "air"
        land_x = w.flight.land_x
        w.advance(3.0)
        return w, land_x

    (slid, at), (stopped, at2) = jump(False), jump(True)
    assert slid.x - at == pytest.approx(physics.LAND_SLIDE, abs=0.5)
    assert stopped.x == at2 and at2 == pytest.approx(at, abs=0.01)


def test_an_arrow_pressed_right_after_an_attack_moves_her_only_once_it_stops_swallowing_it(model):
    """In game (2026-10-21) an arrow pressed within about 0.55 s of an attack's key didn't move her: still
    held, she set off about 1.0 s after the key, and let go before then she didn't move at all; pressed
    0.575 s after the key, it moved her at once (world.ARROW_LOCK, ARROW_HELD; Realism.lock_arrows)."""
    from gamelab.sim.world import ARROW_HELD, Realism
    c = plat(model, 500, 400)

    def walk(lock: bool, press: float, hold: float, until: float) -> float:
        w = world(model, c.id, 100.0, realism=Realism(lock_arrows=lock))
        tap(w, 0.0, "f", 0.04)
        tap(w, press, "right", hold)
        w.advance(until)
        return w.x - 100.0

    assert walk(True, 0.3, 1.0, ARROW_HELD) == 0 and walk(True, 0.3, 1.0, 1.3) > 15  # still held: from 1.0 s
    assert walk(True, 0.3, 0.5, 2.0) == 0                                          # let go before: nothing
    assert walk(True, 0.6, 0.3, 0.75) > 0                                           # pressed after: at once
    assert walk(False, 0.3, 0.3, 0.45) > 0                                          # the default world: at once
    w = world(model, c.id, 100.0, realism=Realism(lock_arrows=True))
    w.attack_lock_ms = 10_000  # an attack key the game doesn't take (inside the lock) swallows nothing
    tap(w, 0.0, "f", 0.04)
    tap(w, 0.3, "right", 0.3)
    w.advance(0.45)
    assert w.x > 100.0


def test_a_jump_back_leaves_at_full_speed_and_lands_about_90_units_away(model):
    # control.actions' key plan, on the engine with the lags spread as measured: keyed 150 ms after the arrow
    # the jump is past the blind spot (at 80 ms, half left at 126 and landed 68-89 units away).
    from gamelab.control import actions
    c = plat(model, 500, 400)
    events = [[0, "right", "down"], *actions.tap(actions.JB_RUN_MS, "alt"), [actions.JB_TURN_MS, "right", "up"],
              *actions.tap(actions.JB_TURN_MS, "left"), *actions.tap(actions.JB_THROW_MS, "f")]
    lands, speeds = [], []
    for seed in range(30):
        w = world(model, c.id, 300.0, facing="left", seed=seed, noise=Noise(lag_sd=1.0))
        for t, name, how in sorted(events, key=lambda e: e[0]):
            w.key(t / 1000, name, how == "down")
        while w.mode == "ground" and w.t < 0.4:
            w.advance(w.t + 0.005)
        speeds.append(w.flight.speeds[0][0])  # as she leaves the ground (the turn's arrow then brakes her)
        w.advance(1.5)
        lands.append(w.x - 300)
    assert min(speeds) >= 0.94 * physics.WALK_SPEED
    assert 86 <= min(lands) and max(lands) <= 97 and sorted(lands)[15] == pytest.approx(94, abs=1)


def test_after_a_climb_she_runs_off_as_the_minimap_traced_it(model):
    # The arrow held from before the top: she stands still through the climb-over (flow.CLIMB_OVER) and the
    # game's lag for the arrow, then speeds up: at full speed 0.35 s after the top, as flow plans it.
    c, d = plat(model, 500, 400), plat(model, 300, 700)
    ran = []
    for phase in PHASES:
        w = World(model, KEYS, (c.id, 700.0), monsters=None, hanging=400.0, step_phase=phase)
        w.key(0.0, "up", True)
        w.key(0.5, "right", True)  # the top comes 100 / 130 = 0.77 s in
        w.advance(1.0)
        top = next(float(line.split()[0]) for line in w.log if "climbs onto platform" in line)
        w.advance(top + 0.6)
        assert w.plat is d
        ran.append(w.x - 700)
    # flow's model: a standing start (RUN_LAG, 0.119 s) CLIMB_OVER (0.23 s) after the top, on average over the
    # game's steps (each a step's run, 4.5 units, from the next)
    assert np.mean(ran) == pytest.approx(physics.WALK_SPEED * (0.6 - flow.CLIMB_OVER - physics.RUN_LAG), abs=0.5)


def test_a_rope_side_jump_goes_out_fast_with_a_little_lift(model):
    c = plat(model, 500, 400)
    w = World(model, KEYS, (c.id, 700.0), monsters=None, hanging=400.0)
    tap(w, 0.0, "alt", 0.04)
    w.advance(0.3)
    assert w.mode == "rope"  # no arrow: the jump key does nothing on a rope
    w.key(0.3, "left", True)
    tap(w, 0.33, "alt", 0.04)  # seen at 0.36 and 0.4 (on the ticks)
    w.advance(0.41)
    assert w.mode == "air"
    assert w.flight.speeds[0] == pytest.approx((-1.3 * physics.WALK_SPEED, -0.5 * physics.JUMP_SPEED))
    w.advance(3.0)
    assert w.plat is c and w.x < 700 - 100


def test_a_running_jump_lands_where_the_engine_and_the_planner_say(model):
    a, b = plat(model, 300, 100), plat(model, 360, 300)
    w = world(model, a.id, 40.0, step_phase=0)
    w.key(0.0, "right", True)
    tap(w, 0.93, "alt", 0.04)  # lifts off at the game's step 1.005 s in (JUMP_LAG, 65 ms, on average): down onto B
    w.advance(1.01)
    assert w.mode == "air"
    x0, y0 = w.flight.path[0]
    assert x0 - 40 == pytest.approx(physics.WALK_SPEED * (0.93 + physics.JUMP_LAG - physics.RUN_LAG), abs=0.5)
    f = body.fly(model, x0, y0, physics.WALK_SPEED, -physics.JUMP_SPEED, shoe=physics.SHOE, arrow=1)
    land, land_x, seconds = physics.fly(model, x0, y0, physics.WALK_SPEED, -physics.JUMP_SPEED)
    assert f.platform is land is b and f.land_x == pytest.approx(land_x, abs=1.5)
    w.key(1.01, "right", False)  # let go in the air: the game acts on it at 1.08 s, and she flies on from there
    w.advance(1.1)
    g = w.flight
    assert g.platform is b and g.land_x == pytest.approx(f.land_x, abs=0.5)
    # ... with no arrow: dragged at 1 unit/s², and at 100 for the last few steps, falling at the cap (670)
    assert 140 < g.vx_land < 146
    w.advance(3.0)
    assert w.plat is b and w.vx == 0
    assert w.x == pytest.approx(g.land_x + body.slide(g.vx_land, physics.SHOE), abs=0.01)  # 13 units on


def test_a_jump_key_pressed_in_the_air_is_ignored(model, monkeypatch):
    c = plat(model, 500, 400)
    w = world(model, c.id, 300.0)
    tap(w, 0.0, "alt", 0.04)  # lifts off at 0.07 s, lands at 0.625
    tap(w, 0.3, "alt", 0.04)
    w.advance(0.3)
    flight = w.flight
    w.advance(0.4)
    assert w.flight is flight
    w.advance(2.0)
    assert w.plat is c and w.x == 300
    # Pressed 25 ms before she lands: lost, as one pressed earlier (world.JUMP_TAKEN_LATE off) ...
    for late, again in ((False, False), (True, True)):
        monkeypatch.setattr(world_mod, "JUMP_TAKEN_LATE", late)
        w = world(model, c.id, 300.0)
        tap(w, 0.0, "alt", 0.04)
        tap(w, 0.6, "alt", 0.04)
        w.advance(0.66)
        assert w.mode == "ground"
        w.advance(0.7)
        assert (w.mode == "air") == again  # ... or taken 42 ms after she landed, when the game acts on it


def test_down_and_jump_drops_through_onto_the_platform_below(model):
    a, c = plat(model, 300, 100), plat(model, 500, 400)
    w = world(model, a.id, 100.0)
    w.key(0.0, "down", True)
    tap(w, 0.1, "alt", 0.04)
    w.key(0.3, "down", False)
    w.advance(0.1 + 0.27)
    assert w.mode == "ground"  # the drop starts DROP_LAG after the key
    w.advance(2.0)
    assert w.plat is c and w.x == 100


def test_walking_off_an_end_falls_and_a_wall_stops_her(model):
    a, b, c = plat(model, 300, 100), plat(model, 360, 300), plat(model, 500, 400)
    w = world(model, a.id, 150.0)
    w.key(0.0, "right", True)
    w.advance(3.0)
    want = physics.fly(model, 200.5, 299, physics.WALK_SPEED, 0.0)[0]
    assert want in (b, c) and w.plat in (b, c)
    w = world(model, c.id, 700.0)
    w.key(0.0, "right", True)
    w.advance(2.0)
    assert w.plat is c and w.x == pytest.approx(800, abs=1e-5) and w.mode == "ground"  # (a hair inside the end)
    # Against it, a jump the other way leaves it.
    w.key(2.0, "right", False)
    tap(w, 2.2, "left", 0.25)  # let go in the air: what she goes is the jump's (and the slide's)
    tap(w, 2.35, "alt", 0.04)
    w.advance(3.5)
    assert w.plat is c and w.x < 800 - 60


def test_down_and_jump_on_a_floor_that_forbids_it_does_nothing():
    t = tree()
    for f in t["foothold"]["1"]["1"].values():  # A
        f["forbidFallDown"] = 1
    m = mapdata.parse(1, t, "Test Map", {})
    a = plat(m, 300, 100)
    w = world(m, a.id, 100.0)
    w.key(0.0, "down", True)
    tap(w, 0.1, "alt", 0.04)
    w.key(0.3, "down", False)
    w.advance(2.0)
    assert w.plat is a and w.mode == "ground" and w.x == 100


def test_in_the_air_only_walls_of_her_foothold_group_stop_her(monkeypatch):
    # A ledge L (0..200, y 300), and 12 units past its right end the wall hanging 85 units under the left end
    # of a block H (212..600, y 240) of another group, as off platform 3's right end on the real map; F is
    # 600 below L, a basin floor 1200 below.
    t = tree()
    t["info"].update(VRLeft=-200, VRRight=1200, VRBottom=1600)
    t["foothold"] = {"1": {
        "10": dict([seg(1, 0, 300, 200, 300), seg(2, 200, 300, 200, 325)]),
        "11": dict([seg(3, 212, 325, 212, 240, 0, 4), seg(4, 212, 240, 600, 240, 3, 0)]),
        "12": dict([seg(5, 212, 900, 600, 900)]),
        "13": dict([seg(6, -100, 1500, 1000, 1500)]),
    }}
    t["life"], t["ladderRope"] = {}, {}
    m = mapdata.parse(1, t, "Test Map", {})
    ledge, block, floor, basin = plat(m, 300, 100), plat(m, 240, 400), plat(m, 900, 400), plat(m, 1500, 400)

    def walk_off() -> World:  # the arrow let go once she's off the end
        w = world(m, ledge.id, 150.0)
        tap(w, 0.0, "right", 0.6)
        w.advance(3.0)
        return w

    w = walk_off()
    assert w.plat is floor and w.x > 300 and w.group == floor.group  # through H's wall, and on
    monkeypatch.setattr(body, "OWN_GROUP_WALLS", False)
    w = walk_off()  # every wall: H's stops her, and she comes down by it (in the basin, or round its foot)
    assert w.x < 220
    monkeypatch.setattr(body, "OWN_GROUP_WALLS", True)
    # Her own group's walls still stop her: from C, a running jump into the wall rising at C's right end.
    c = plat(model_c := mapdata.parse(1, tree(), "Test Map", {}), 500, 400)
    w = world(model_c, c.id, 740.0)
    w.key(0.0, "right", True)
    tap(w, 0.2, "alt", 0.04)
    w.advance(2.0)
    assert w.plat is c and 799 < w.x <= 800 and w.group == c.group


def test_up_and_a_jump_grab_the_rope_climb_it_and_down_lets_her_off_at_the_bottom(model):
    c, d = plat(model, 500, 400), plat(model, 300, 700)
    w = world(model, c.id, 700.0)
    w.key(0.0, "up", True)
    tap(w, 0.05, "alt", 0.04)
    w.advance(0.4)
    assert w.mode == "rope"
    w.advance(3.0)
    assert w.plat is d and w.x == 700 and w.mode == "ground"
    w.key(3.0, "up", False)
    w.key(3.1, "down", True)
    w.advance(3.3)
    assert w.mode == "rope"
    w.advance(6.0)
    assert w.plat is c and w.x == 700


def test_a_star_kills_gives_exp_and_drops_that_can_be_picked_up(model):
    c = plat(model, 500, 400)
    w = world(model, c.id, 100.0, monsters=[("Snail", c.id, 400.0)], facing="right")
    tap(w, 0.0, "f", 0.04)
    w.advance(0.5)
    assert w.monsters[0].alive  # the star leaves ~0.52 s after the key
    w.advance(0.6)
    assert not w.monsters[0].alive and w.kills == {"snail": 1} and w.exp == 2
    assert int((exp_pixels(0) != exp_pixels(w.exp)).sum()) >= 3
    w.advance(3.0)
    (d,) = w.live_drops()
    assert d.mesos and abs(d.x - w.monsters[0].died_x) <= 10
    w.x = d.x + 10
    tap(w, 3.0, "z", 0.04)
    w.advance(3.2)
    assert w.live_drops() == [] and w.picked == {"mesos": 1}


def test_a_star_goes_the_way_she_faces_and_not_past_its_reach(model):
    c = plat(model, 500, 400)
    w = world(model, c.id, 100.0, monsters=[("Snail", c.id, 400.0)], facing="left")
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert w.monsters[0].alive and w.monsters[0].hp == 1
    w = still(world(model, c.id, 100.0, monsters=[("Snail", c.id, 100 + reach.STAR_REACH + 40)], facing="right"))
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert w.monsters[0].alive and w.monsters[0].hp == 1


def still(w):
    """Monsters that stand where they are (for tests that aim at them)."""
    for m in w.monsters:
        m.speed = 0.0
    return w


def _jump_throw(w, gap: float = 0.1):
    """A jump-throw from where she stands: the jump key, and the attack key `gap` s after it."""
    w.invulnerable_ms = 10_000  # no touches in the way
    tap(w, 0.0, "alt", 0.04)
    tap(w, gap, "f", 0.04)
    w.advance(1.0)


def test_a_jump_throw_attack_starts_too_high_for_a_snail_but_not_for_a_pig(model):
    # The attack picks its target when it starts, JUMP_LAG after its key as a jump lifts off: 100 ms after the
    # jump key she's 46 units up, and the area it picks from starts 19 below her feet, over a snail.
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 300.0, monsters=[("Snail", c.id, 330.0)], facing="right"))
    _jump_throw(w)
    assert w.monsters[0].alive and w.monsters[0].hp == 1 and any("star thrown right: no hit" in l for l in w.log)
    w = still(world(model, c.id, 300.0, monsters=[("Pig", c.id, 330.0)], facing="right"))
    _jump_throw(w)
    assert w.monsters[0].hp == 2  # a pig is tall enough: a star
    # The attack 40 ms after the jump key starts 21 up: a star kills the snail.
    w = still(world(model, c.id, 300.0, monsters=[("Snail", c.id, 330.0)], facing="right"))
    _jump_throw(w, 0.04)
    assert not w.monsters[0].alive and w.kills == {"snail": 1}


def test_with_the_v83_air_pick_a_jump_throw_picks_as_from_the_ground_from_where_she_is(model, monkeypatch):
    # world.AIR_PICK 'v83' (the other answer `probe run air` weighs): the attack 40 ms after the jump key starts
    # 21 up, and the claw and the search go from there as from the ground. A snail 30 off is under the
    # stabs' boxes and short of the search's first band, so the star hits nothing (by the area it dies); a pig
    # 120 off, its body reaching above the first window's floor, takes a star.
    c = plat(model, 500, 400)
    monkeypatch.setattr(world_mod, "AIR_PICK", "v83")
    w = still(world(model, c.id, 300.0, monsters=[("Snail", c.id, 330.0)], facing="right"))
    _jump_throw(w, 0.04)
    assert w.monsters[0].alive and w.monsters[0].hp == 1 and any("star thrown right: no hit" in l for l in w.log)
    w = still(world(model, c.id, 300.0, monsters=[("Pig", c.id, 420.0)], facing="right"))
    _jump_throw(w, 0.04)
    assert w.monsters[0].hp == 2


def test_the_world_s_claw_is_reach_s_model(model, monkeypatch):
    # The world swipes by reach.CLAW: with the one box reach had until 2026-10-20 (90 ahead of her, on her level
    # only: the other answer `probe run claw-edge` weighs), a pig 120 off is swiped, where the client's stabs
    # (cut to 65) leave it to a star.
    c = plat(model, 500, 400)
    old = reach.Claw((("swing", ((-90, -62), (0, 17))),), 90, level=15)
    for claw, hp in ((reach.CLAW, 2), (old, 3 - world_mod.CLAW)):
        monkeypatch.setattr(reach, "CLAW", claw)
        w = still(world(model, c.id, 300.0, monsters=[("Pig", c.id, 420.0)], facing="right"))
        tap(w, 0.0, "f", 0.04)
        w.advance(1.0)
        assert w.monsters[0].hp == pytest.approx(hp)


def test_the_ground_attack_swipes_what_the_claw_reaches(model):
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 300.0, monsters=[("Snail", c.id, 330.0)], facing="right"))
    w.invulnerable_ms = 10_000
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert w.monsters[0].hp == pytest.approx(1 - world_mod.CLAW)  # a snail beside her: swiped, every time
    w = still(world(model, c.id, 300.0, monsters=[("Pig", c.id, 340.0), ("Pig", c.id, 500.0)], facing="right"))
    w.invulnerable_ms = 10_000
    tap(w, 0.0, "f", 0.04)      # from the ground, a monster that close takes a claw swipe, not a star
    w.advance(1.0)
    assert w.monsters[0].hp == pytest.approx(3 - world_mod.CLAW) and w.monsters[1].hp == 3
    assert any("claw swipe hits pig #0" in line for line in w.log)
    # Either stab stance reaches 65 ahead of her, into a pig's body (34 either side) up to 99 off: a star
    # beyond.
    for x, swiped in ((395.0, True), (405.0, False)):
        w = still(world(model, c.id, 300.0, monsters=[("Pig", c.id, x)], facing="right"))
        tap(w, 0.0, "f", 0.04)
        w.advance(1.0)
        assert w.monsters[0].hp == pytest.approx(3 - (world_mod.CLAW if swiped else 1)), x


def test_a_flinching_snail_is_tall_enough_for_a_jump_throw(model):
    # Hit, a snail flinches for 0.6 s, 33 units tall: the jump-throw right after a claw swipe gets it.
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 300.0, monsters=[("Snail", c.id, 330.0)], facing="right"))
    w.invulnerable_ms = 10_000
    tap(w, 0.0, "f", 0.04)          # swiped at 0.52 s
    tap(w, 0.75, "alt", 0.04)       # after the attack lock: the jump, and the attack 100 ms later
    tap(w, 0.85, "f", 0.04)         # (it starts at 0.905 s, while the snail still flinches)
    w.advance(2.0)
    assert not w.monsters[0].alive and any("star kills snail" in line for line in w.log)


def test_a_jump_throw_at_a_snail_hits_an_orange_mushroom_on_the_platform_above(model):
    # She's on B with a snail beside her, A 60 above to her left: the jump-throw's attack, picking its
    # target 46 units up, goes over the snail to the mushroom up there, however close to the side.
    a, b = plat(model, 300, 100), plat(model, 360, 300)
    w = still(world(model, b.id, 300.0, monsters=[("Snail", b.id, 270.0), ("Orange Mushroom", a.id, 150.0)],
                    facing="left"))
    _jump_throw(w)
    assert w.monsters[0].hp == 1 and w.monsters[1].hp == 2
    # From the ground the claw swipes the snail, and the mushroom (60 above, 150 to the side) is left alone.
    w = still(world(model, b.id, 300.0, monsters=[("Snail", b.id, 270.0), ("Orange Mushroom", a.id, 150.0)],
                    facing="left"))
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert w.monsters[0].hp == pytest.approx(1 - world_mod.CLAW) and w.monsters[1].hp == 3


def test_a_star_hits_the_first_monster_in_its_path_but_passes_one_right_under_her(model):
    a, b = plat(model, 300, 100), plat(model, 360, 300)  # B is 60 below A, to its right
    w = still(world(model, a.id, 190.0, monsters=[("Orange Mushroom", b.id, 250.0), ("Orange Mushroom", b.id, 380.0)],
                    facing="right"))
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    # 60 below and 60 along is right under her (the star's search picks an Orange Mushroom from 113
    # sideways), and the claw doesn't swipe off her level: the star goes over it to the next one.
    assert w.monsters[0].hp == 3 and w.monsters[1].hp == 2
    # A snail 60 below and 190 along isn't hit: it's picked only from 246 (26 tall, its top 34 below her feet).
    w = still(world(model, a.id, 190.0, monsters=[("Snail", b.id, 380.0)], facing="right"))
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert w.monsters[0].hp == 1
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0, monsters=[("Snail", c.id, 400.0), ("Snail", c.id, 250.0)]))
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert [m.alive for m in w.monsters] == [True, False]  # the nearer one takes it


def test_the_star_search_picks_by_body(model):
    # A (y 300) is 60 above B (y 360). From A a pig on B is picked from 151 units sideways (walking, 46 tall:
    # its far side in the band whose window takes it in), an Orange Mushroom from 113; from B, a pig up on A
    # from 111.
    a, b, c = plat(model, 300, 100), plat(model, 360, 300), plat(model, 500, 400)

    def throw(w):
        tap(w, 0.0, "f", 0.04)
        w.advance(1.0)
        return w.monsters[0].hp

    for x, hp in ((330.0, 3), (355.0, 2)):  # 140 and 165 along
        assert throw(still(world(model, a.id, 190.0, monsters=[("Pig", b.id, x)], facing="right"))) == hp, x
    assert throw(still(world(model, a.id, 190.0, monsters=[("Orange Mushroom", b.id, 310.0)], facing="right"))) == 2
    for her, hp in ((266.0, 3), (286.0, 2)):  # 100 and 120 along
        assert throw(still(world(model, b.id, her, monsters=[("Pig", a.id, 166.0)], facing="left"))) == hp, her
    # On her level, out to 400 to its near side: a pig 430 along (its near side 396) is hit, one 440 along isn't.
    for x, hp in ((530.0, 2), (540.0, 3)):
        assert throw(still(world(model, c.id, 100.0, monsters=[("Pig", c.id, x)], facing="right"))) == hp, x


def test_the_star_goes_to_any_monster_of_the_searchs_first_band(model):
    # Two pigs 200 and 210 along, both in band 5 (their near sides 165 and 175 ahead): the client takes the one
    # that entered the map first, which nothing tells, so the world draws one per attack (seeded). Over a
    # dozen seeds each takes some of the stars; a third one, 300 along (band 10), never does.
    c = plat(model, 500, 400)
    hit = set()
    for seed in range(1, 13):
        w = still(world(model, c.id, 100.0, monsters=[("Pig", c.id, 300.0), ("Pig", c.id, 310.0),
                                                      ("Pig", c.id, 400.0)], facing="right", seed=seed))
        tap(w, 0.0, "f", 0.04)
        w.advance(1.0)
        hit |= {i for i, m in enumerate(w.monsters) if m.hp < m.hp_max}
    assert hit == {0, 1}


def test_a_flinching_pig_below_is_picked_closer(model):
    # Hit, a pig flinches 54 tall for 0.6 s: 60 below her the search picks it from 111 sideways, not 151.
    a, b = plat(model, 300, 100), plat(model, 360, 300)
    w = still(world(model, a.id, 190.0, monsters=[("Pig", b.id, 315.0)], facing="right"))  # 125 along
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert w.monsters[0].hp == 3 and any("star thrown right: no hit" in line for line in w.log)
    w.monsters[0].hit_at = w.t  # as if a star had hit it just now
    tap(w, 1.0, "f", 0.04)
    w.advance(2.0)
    assert w.monsters[0].hp == 2


def test_a_wall_of_her_own_foothold_group_stops_the_star(model):
    # The step map of test_attack: she's on A (y 500), below the step wall at 200 joining it to B (y 440,
    # the same group). From the ground the star doesn't get past the wall: an Orange Mushroom up on B, 220
    # along and 60 above (in reach by distance), is picked and isn't hit: the star stops at the wall.
    from test_attack import step_map
    steps, a, b, _ = step_map()
    w = still(world(steps, a.id, 100.0, monsters=[("Orange Mushroom", b.id, 320.0)], facing="right"))
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert w.monsters[0].hp == 3
    assert any("star thrown right at orange-mushroom #0: stopped by the wall at x 200" in line for line in w.log)
    # Jumping, the line it runs at rises with her: keyed 100 ms after the jump the attack starts 46 up, its
    # line 74 above the platform, over the step's top (60). And from up on B a star goes down past the wall.
    w = still(world(steps, a.id, 150.0, monsters=[("Orange Mushroom", b.id, 360.0)], facing="right"))
    _jump_throw(w)
    assert w.monsters[0].hp == 2
    w = still(world(steps, b.id, 300.0, monsters=[("Orange Mushroom", a.id, 90.0)], facing="left"))
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert w.monsters[0].hp == 2


def test_a_star_stopped_on_its_way_to_the_pick_hits_no_other_monster():
    # test_attack's shield map: from A at 10 a pig up on B at 215 (past the step wall, the search's band 5) and
    # a pig on L 60 below at 170 (short of it, band 6). The search picks the one on B whatever the wall, its
    # line is checked, stopped, and the attack hits nothing: the one on L, its line clear, doesn't get the star
    # the wall stopped (the client doesn't refill its pick). Alone, the one on L is hit.
    from test_attack import shield_map
    steps, a, b, low = shield_map()
    w = still(world(steps, a.id, 10.0, monsters=[("Pig", b.id, 215.0), ("Pig", low.id, 170.0)], facing="right"))
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert [m.hp for m in w.monsters] == [m.hp_max for m in w.monsters]
    assert any("at pig #0: stopped by the wall at x 200" in line for line in w.log)
    w = still(world(steps, a.id, 10.0, monsters=[("Pig", low.id, 170.0)], facing="right"))
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert w.monsters[0].hp < w.monsters[0].hp_max


def test_with_the_v83_line_her_floor_stops_a_star_down_close_past_its_end(monkeypatch):
    # test_attack's ledge map: from H at 100 (200 from its end) at a pig on L 60 below, 280 off. The level line
    # gets there; by the v83 line (reach.LINE) her own floor stops it (the line drops 28 to its level in 2/3 of
    # the way to the pig's top: past the end only from 300 off), and from 200 the star gets there again.
    from test_attack import ledge_map
    ledges, h, low, up = ledge_map()

    def hit(x):
        w = still(world(ledges, h.id, x, monsters=[("Pig", low.id, 380.0)], facing="right"))
        tap(w, 0.0, "f", 0.04)
        w.advance(1.0)
        return w.monsters[0].hp < w.monsters[0].hp_max, w.log

    assert hit(100.0)[0]
    monkeypatch.setattr(reach, "LINE", "v83")
    got, log = hit(100.0)
    assert not got and any("at pig #0: stopped by the floor at y 500" in line for line in log)
    assert hit(200.0)[0]


def test_a_monster_touching_her_hurts_and_knocks_her_back(model, monkeypatch):
    monkeypatch.setattr(world_mod, "EVADE", 0.0)
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 400.0, monsters=[("Pig", c.id, 420.0)]))
    w.advance(0.05)
    assert w.touches == 1 and w.hp == pytest.approx(1 - world_mod.TOUCH_DAMAGE) and w.mode == "air"
    w.advance(0.9)  # invulnerable for a second
    assert w.touches == 1 and w.plat is c and w.vx == 0
    assert 45 <= 400 - w.x <= 70  # knocked back ~58 units, away from it (lab/hits.jsonl's median push)


def test_the_arrow_she_holds_doesnt_steer_a_knocks_flight(model, monkeypatch):
    # The logged pushes (~58 units) were all taken with no key pressed; what an arrow does to the flight
    # isn't measured, so it does nothing until she lands (world.KNOCK_STEERS); by the engine's air steering
    # the arrow toward the monster would brake her at 200 units/s².
    monkeypatch.setattr(world_mod, "EVADE", 0.0)
    c = plat(model, 500, 400)

    def landed(arrow: str | None) -> float:
        w = still(world(model, c.id, 400.0, monsters=[("Pig", c.id, 420.0)]))
        if arrow:
            w.key(0.0, arrow, True)
        w.advance(0.05)
        assert w.touches == 1 and w.mode == "air"
        while w.mode == "air":
            w.advance(w.t + 0.005)
        return w.x

    free = landed(None)
    assert landed("right") == free and landed("left") == free  # toward it, away from it: as with none
    monkeypatch.setattr(world_mod, "KNOCK_STEERS", True)
    assert landed("right") > free + 2  # braked toward the monster: she lands nearer it


def test_falling_below_the_lowest_floor_ends_the_run_named():
    # Two ledges with nothing under the gap: walked off the first, she falls past every floor. The run
    # ends there (SimFellOut, a SimDied), where she used to hang in the air at the end of her flight's 4 s.
    t = tree()
    t["foothold"] = {"1": {"1": dict([seg(1, 0, 300, 200, 300)]), "2": dict([seg(2, 400, 300, 600, 300)])}}
    t["ladderRope"], t["life"] = {}, {}
    m = mapdata.parse(1, t, "Test Map", {})
    w = world(m, plat(m, 300, 100).id, 150.0)
    w.key(0.0, "right", True)
    with pytest.raises(SimFellOut):
        w.advance(30.0)
    assert w.fell_out_at < 1.0 and w.y > 300 and "falls out of the map" in w.log[-1]
    assert w.summary()["fell_out_at"] == w.fell_out_at
    with pytest.raises(SimDied):  # whatever looks next finds the run over
        w.advance(31.0)


def test_a_short_tap_can_be_lost_turning_her_without_walking_her(model):
    # Of five taps of 20-50 ms recorded, three moved her 0 (Noise.tap_lost): the lag spread alone loses
    # almost none of them.
    c = plat(model, 500, 400)
    moved, turned = [], []
    for seed in range(40):
        w = world(model, c.id, 400.0, seed=seed, noise=Noise(tap_lost=0.6), facing="right")
        tap(w, 0.1, "left", 0.04)
        w.advance(1.0)
        moved.append(w.x < 399)
        turned.append(w.facing == "left")
    assert all(turned) and 0.2 < 1 - sum(moved) / 40 < 0.9
    w = world(model, c.id, 400.0, noise=Noise(tap_lost=1.0))
    tap(w, 0.1, "left", 0.2)  # longer than TAP_LOST_MS: walks her
    w.advance(1.0)
    assert w.x < 380


def test_a_touch_takes_hp_by_the_monsters_touch_damage_and_is_evaded_by_its_accuracy(model, monkeypatch):
    monkeypatch.setattr(world_mod, "evade_chance", lambda info: 0.0)
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 400.0, monsters=[("Pig", c.id, 420.0)]))
    w.monsters[0].info = world_mod.MobInfo(pad=15.0, acc=54.0)
    w.advance(0.05)
    (t, took), = w.touch_log
    assert took == pytest.approx(world_mod.touch_damage(w.monsters[0].info, w.chance("damage", round(t * 1000))))
    assert w.hp == pytest.approx(1 - took)
    pig, snail = world_mod.MobInfo(15.0, 54.0), world_mod.MobInfo(3.0, 33.0)
    # 0.68 of PADamage over her 642 HP, give or take 70% (lab/hits.jsonl): 1.6% a pig, 0.3% a snail
    assert world_mod.touch_damage(pig, 0.5) == pytest.approx(15 * 0.68 / 642)
    assert world_mod.touch_damage(pig, 0.0) == pytest.approx(0.3 * 15 * 0.68 / 642)
    assert world_mod.touch_damage(snail, 0.5) < 0.004 < 0.014 < world_mod.touch_damage(pig, 0.5)
    assert world_mod.touch_damage(None, 0.9) == world_mod.TOUCH_DAMAGE  # no game data: the logged median
    monkeypatch.undo()
    assert world_mod.evade_chance(pig) == pytest.approx(0.5)  # as logged hopping over pigs
    assert world_mod.evade_chance(snail) == pytest.approx(27 / 33)
    assert world_mod.evade_chance(None) == world_mod.EVADE


def test_the_game_data_gives_each_kind_its_touch_damage_and_accuracy():
    m = scenarios.real_map()
    if m is None:
        pytest.skip("the map's game data isn't on this computer")
    infos = {kind_of(s.name): world_mod.mob_info(s.mob_id) for s in m.spawns}
    if any(i is None for i in infos.values()):
        pytest.skip("the monsters' game data isn't on this computer")
    assert {k: (i.pad, i.acc) for k, i in infos.items()} == {
        "snail": (3, 33), "slime": (13, 53), "pig": (15, 54), "orange-mushroom": (17, 62)}
    w = World(m, KEYS, (36, 300.0), monsters="all")
    assert {mo.kind: mo.info for mo in w.monsters} == infos
    assert infos["slime"] == SLIME and infos["orange-mushroom"] == MUSHROOM  # the frames the tests above use


F = world_mod.Frame
# The slime's and the Orange Mushroom's frames as the game data has them (Mob/0000007.img, 0000009.img):
# a slime's walk lifts it 16-17 units for its 300 ms hop (and 10 for 100 ms), and is 66 tall then.
SLIME = world_mod.MobInfo(13.0, 53.0, stand=(F(200, -27, 24, 0, 43), F(300, -28, 24, 0, 42), F(250, -26, 23, 0, 44)),
                          move=(F(200, -29, 26, 0, 41), F(50, -31, 26, 0, 40), F(150, -27, 19, 16, 66),
                                F(150, -27, 19, 17, 66), F(100, -27, 21, 10, 58), F(100, -28, 25, 0, 42),
                                F(300, -27, 24, 0, 43)),
                          hit=F(600, -23, 26, 0, 46))
MUSHROOM = world_mod.MobInfo(17.0, 62.0, stand=(F(180, -27, 36, 0, 58), F(180, -27, 36, 0, 55)),
                             move=(F(180, -32, 32, 0, 55), F(120, -31, 31, 3, 67), F(180, -31, 32, 0, 55)),
                             hit=F(600, -28, 34, 0, 65))


def _put(w, mode, x, y):
    w.mode, w.plat, w.x, w.y = mode, None if mode == "air" else w.plat, x, y
    w.invulnerable_ms, w.touches = -10_000, 0


def test_a_monster_s_body_is_its_frame_s_box_turned_the_way_it_faces(model):
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0, monsters=[("Slime", c.id, 400.0)]))
    m = w.monsters[0]
    m.info, m.t0, m.d, m.t1, m.anim0 = SLIME, 0.0, 1, 1e9, 0.0  # walking right (on the spot) for good
    assert m.body(0.1) == (-26, 29, 0, 41)       # its first frame, turned to face right
    assert m.body(0.3) == (-19, 27, 16, 66)      # 250-400 ms into its walk it's in the air
    assert m.body(1.05 + 0.3) == m.body(0.3)     # the walk loops every 1,050 ms
    assert m.body(0.3, flinching=True) == (-26, 23, 0, 46)  # hit: its hit1 frame
    m.t1 = 1.0
    m.at(1.0)
    m.d, m.t1, m.anim0 = 0, 1e9, 1.0             # then it stands, still facing right
    assert m.body(1.1) == (-24, 27, 0, 43)
    m.facing = -1
    assert m.body(1.1) == (-27, 24, 0, 43)       # facing left, as it's drawn
    m.info = None                                # no game data: its kind's half width and height
    assert m.body(1.1) == (-m.half, m.half, 0.0, m.height)


def test_her_jump_clears_a_slime_on_the_ground_but_not_one_mid_hop(model):
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0, monsters=[("Slime", c.id, 400.0)]))
    m = w.monsters[0]
    m.info, m.t0, m.d, m.t1, m.anim0 = SLIME, 0.0, 1, 1e9, 0.0
    for ms, touched in ((100, False), (300, True), (500, True), (800, False)):
        w.ms = ms
        _put(w, "air", 420.0, c.y_at(420) - 50)  # her feet 50 units up, beside its front
        w._touch()
        assert w.touches == int(touched), ms


def test_a_monster_touches_her_by_its_body_as_it_faces(model):
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0, monsters=[("Orange Mushroom", c.id, 400.0)]))
    m = w.monsters[0]
    m.info, m.t0, m.d, m.t1 = MUSHROOM, 0.0, 0, 1e9  # standing for good
    for facing, touched in ((-1, True), (1, False)):  # its cap reaches 36 behind it, 27 in front
        m.facing = facing
        _put(w, "ground", 400.0 + 34 + world_mod.HER_HALF, c.y_at(400))
        w._touch()
        assert w.touches == int(touched), facing


def test_a_hit_pushes_a_monster_back_and_stands_it_while_it_flinches(model, monkeypatch):
    c = plat(model, 500, 400)

    def shot(push):
        monkeypatch.setattr(world_mod, "MOB_KNOCKBACK", push)
        w = world(model, c.id, 300.0, monsters=[("Pig", c.id, 600.0)], facing="right")
        m = w.monsters[0]
        m.t0, m.x0, m.d, m.t1 = 0.0, 600.0, -1, 1e9  # walking at her for good
        tap(w, 0.0, "f", 0.04)
        w.advance(0.53)
        return m

    m = shot(True)
    hit = m.hit_at
    at_hit = 600.0 - m.speed * hit
    assert m.hp == 2 and hit == pytest.approx(0.52)  # a star (of 3) lands 520 ms after the key
    assert m.at(hit + 0.125) == pytest.approx(at_hit + 9)  # pushed away from her ...
    assert m.at(hit + 0.3) == m.at(hit + 0.59) == pytest.approx(at_hit + 18)  # ... 18 units, then it stands
    assert m.d == 0
    m = shot(False)  # without: it walks on through the hit
    assert m.at(hit + 0.3) == pytest.approx(600.0 - m.speed * (hit + 0.3))
    w = world(model, c.id, 700.0, monsters=[("Pig", c.id, 800.0)])
    m = w.monsters[0]
    m.knock(0.0, 1)
    assert m.at(0.5) == m.hi  # at its platform's end it stays on it


def test_simulated_monsters_walk_the_maps_speeds_and_stand_through_a_hits_push():
    """The game data's walking speeds (MapModel.mob_speeds, localization.monsters.speeds) and the push of a
    hit: a pig on a map whose data says 120 units/s (walkSpeed 100, speed stat +20; the table says 125)
    walks at 120, the tracker expects as much, a hit pushes it 18 units away from her over 0.25 s and stands
    it until FLINCH_S after the hit, and then it walks at 120 again."""
    class Walks:  # its draws after the stand: walk right, for 4 s
        def random(self):
            return 0.9

        def uniform(self, lo, hi):
            return hi

    fast = mapdata.parse(1, tree(), "Test Map", {"walkSpeed": 100.0}, {"2": "Snail", "8": "Pig"}, {"2": 2},
                         mob_speeds={"8": 20})
    assert monsters.speeds(fast)["pig"] == 120.0 and monsters.Tracker(fast).walking["pig"] == 120.0
    c = plat(fast, 500, 400)
    w = world(fast, c.id, 300.0, monsters=[("Pig", c.id, 600.0)])
    m = w.monsters[0]
    assert m.speed == 120.0
    m.t0, m.x0, m.d, m.t1, m.rng = 0.0, 600.0, -1, 1e9, Walks()  # walking at her
    assert m.at(1.0) == pytest.approx(480.0)
    m.knock(1.0, 1)
    assert m.at(1.125) == pytest.approx(489.0) and m.at(1.25) == m.at(1.0 + monsters.FLINCH_S - 0.01) == 498.0
    assert m.faces(1.3) == (-1, False)  # standing, still facing her
    t = 1.0 + monsters.FLINCH_S
    assert m.at(t + 0.2) == pytest.approx(498.0 + 120.0 * 0.2) and m.faces(t + 0.2) == (1, True)


def test_she_evades_some_touches_and_dies_at_no_hp(model, monkeypatch):
    c = plat(model, 500, 400)
    monkeypatch.setattr(world_mod, "EVADE", 1.0)
    w = world(model, c.id, 400.0, monsters=[("Pig", c.id, 420.0)])
    w.advance(0.05)
    assert w.touches == w.evaded == 1 and w.hp == 1.0 and w.mode == "ground"  # MISS: nothing happens
    monkeypatch.setattr(world_mod, "EVADE", 0.0)
    w = world(model, c.id, 400.0, monsters=[("Pig", c.id, 420.0)], hp=0.01)
    with pytest.raises(SimDied):
        w.advance(0.05)
    assert w.dead and w.hp == 0
    with pytest.raises(SimDied):  # whatever looks next finds the run over
        w.advance(1.0)


def test_a_drop_that_has_vanished_cant_be_picked_up_however_seldom_it_was_looked_at(model):
    c = plat(model, 500, 400)
    w = world(model, c.id, 300.0, drops=[("mesos", c.id, 305.0)])
    w.advance(DROP_LIFE + 1)    # no look in between
    tap(w, DROP_LIFE + 1, "z", 0.04)
    w.advance(DROP_LIFE + 2)
    assert w.picked == {} and w.live_drops() == []


def test_a_potion_and_the_chair_give_hp_back_and_an_arrow_stands_her_up(model):
    c = plat(model, 500, 400)
    w = world(model, c.id, 100.0, hp=0.4)
    tap(w, 0.0, "r", 0.04)
    w.advance(0.1)
    assert w.hp == pytest.approx(0.7)
    tap(w, 0.2, "d", 0.04)
    w.advance(10.2)
    assert w.sitting and w.hp == pytest.approx(1.0)
    tap(w, 10.3, "right", 0.05)
    w.advance(10.4)
    assert not w.sitting


def test_monsters_walk_their_platform_the_same_way_however_often_looked_at(model):
    b = plat(model, 360, 300)
    w1, w2 = world(model, b.id, 300.0, monsters="all", seed=3), world(model, b.id, 300.0, monsters="all", seed=3)
    seen = []
    for t in np.arange(0.0, 20.0, 0.37):
        seen.append(w1.monsters[0].at(float(t)))
    assert w2.monsters[0].at(19.98) == w1.monsters[0].at(19.98)
    assert all(b.left + w1.monsters[0].half - 1e-6 <= x <= b.right - w1.monsters[0].half + 1e-6 for x in seen)
    assert max(seen) - min(seen) > 20


# --- the fake player -------------------------------------------------------------------------------

def _player(model):
    clock = Clock()
    c = plat(model, 500, 400)
    w = World(model, KEYS, (c.id, 100.0), monsters=None, origin=clock.now)
    trace = Trace(clock.now)
    return clock, w, FakePlayer(w, clock, trace, KEYS), trace


def test_plans_run_one_after_another_and_end_ms_makes_one_last_longer(model):
    clock, w, player, trace = _player(model)
    a = player.send([[0, "right", "down"], [100, "right", "up"]], lead_ms=0, name="a", end_ms=300)
    b = player.send([[0, "left", "down"], [50, "left", "up"]], lead_ms=0, name="b")
    pa, pb = player.plans[a], player.plans[b]
    assert pb.start == pytest.approx(pa.end) and pa.end - pa.start == pytest.approx(0.3)
    assert not player.done(a)
    player.wait(a)
    assert player.done(a) and not player.done(b)
    player.wait(b)
    assert clock.now >= pb.end and player.done(b)
    assert [p[1] for p in trace.plans] == ["a", "b"]


def test_cancel_withdraws_a_plan_not_started_and_cuts_a_running_one(model):
    clock, w, player, trace = _player(model)
    a = player.send([[0, "right", "down"], [2000, "right", "up"]], lead_ms=0, name="a")
    b = player.send([[0, "up", "down"], [100, "up", "up"]], lead_ms=0, name="b")
    clock.sleep(0.5)
    w.advance(clock.now)
    assert "right" in w.held
    assert player.cancel(a) is False  # running: cut short, every key let go now
    assert player.cancel(b) is True           # not started: withdrawn
    clock.sleep(3.0)
    w.advance(clock.now)
    assert w.held == {} and w.mode == "ground"
    assert player.done(a) and not player.done(b)
    with pytest.raises(client.PlayerError):
        player.wait(b)
    assert [c[2] for c in trace.cancels] == [False, True]


def test_keys_still_held_are_let_go_when_a_plan_ends(model):
    clock, w, player, _ = _player(model)
    player.submit([[0, "right", "down"], [200, "z", "down"], [240, "z", "up"]], lead_ms=0, name="run")
    w.advance(clock.now + 0.01)
    assert w.held == {}


def test_a_plan_due_to_start_the_moment_it_is_cancelled_is_withdrawn(model):
    clock, w, player, trace = _player(model)
    player.send([[0, "right", "down"], [300, "right", "up"]], lead_ms=0, name="a")
    b = player.send([[0, "alt", "down"], [40, "alt", "up"]], lead_ms=0, name="b")
    clock.advance_to(player.plans[b].start)  # b starts now: nothing of it has been pressed yet
    w.advance(clock.now)
    assert player.cancel(b) is True
    w.advance(clock.now + 2)
    assert w.mode == "ground" and "jump" not in w.held and trace.cancels[-1][2] is True


def test_a_plan_the_real_player_would_refuse_fails_the_same_way(model):
    clock, w, player, _ = _player(model)
    with pytest.raises(client.PlayerError):
        player.submit([[0, "enter", "down"], [40, "enter", "up"]], lead_ms=0, name="bad")
    with pytest.raises(client.PlayerError):
        player.submit([[100, "right", "down"], [0, "right", "up"]], lead_ms=0, name="unsorted")


def test_the_fakes_go_in_and_come_out_again(model, tmp_path):
    clock, w, player, _ = _player(model)
    real = (time.monotonic, time.sleep, client.send, paths.SHOT_LOG)
    from maplehelper import win32
    with patched(clock, player, tmp_path):
        t0 = time.monotonic()
        time.sleep(1.5)
        assert time.monotonic() - t0 == pytest.approx(1.5) and t0 == clock.now - 1.5
        assert client.send == player.send
        assert tmp_path in paths.SHOT_LOG.parents and tmp_path in client.INBOX.parents
        with pytest.raises(SimViolation):
            win32.send_key(K("left"), True)
        assert time.strftime("%Y") == time.strftime("%Y", time.gmtime(fakes.EPOCH + clock.elapsed))
    assert (time.monotonic, time.sleep, client.send, paths.SHOT_LOG) == real


def test_nothing_may_be_written_under_gamelab_lab_during_a_run(model, tmp_path):
    clock, w, player, _ = _player(model)
    fake = types.ModuleType("gamelab._sim_test_module")  # a module with a lab path only as a default

    def save(name: str, folder=LAB / "telemetry"):
        return folder / name
    save.__module__ = fake.__name__
    fake.save = save
    sys.modules[fake.__name__] = fake
    try:
        with patched(clock, player, tmp_path):
            assert save("x") == tmp_path / "telemetry" / "x"  # function defaults are redirected too
            for write in (lambda: open(LAB / "sim-test.txt", "w"), lambda: (LAB / "sim-test.txt").write_text("x"),
                          lambda: (LAB / "sim-test-dir").mkdir(), lambda: os.replace(tmp_path / "a", LAB / "b")):
                with pytest.raises(SimViolation):
                    write()
            (tmp_path / "ok.txt").write_text("fine")  # anywhere else is fine
            assert (tmp_path / "ok.txt").read_text() == "fine"
        assert save("x") == LAB / "telemetry" / "x"
        open(tmp_path / "after.txt", "w").close()
    finally:
        del sys.modules[fake.__name__]
    assert not (LAB / "sim-test.txt").exists() and not (LAB / "sim-test-dir").exists()


def test_the_stack_runs_only_with_the_fakes_in_place(model):
    sc = scenarios.Scenario("guard", lambda sim: None, (plat(model, 500, 400).id, 100.0), limit=50)
    sim = scenarios.Sim(model, sc)
    with pytest.raises(SimViolation):
        sim.hunter()  # not entered: gamelab.control.client is the real one
    with scenarios.Sim(model, sc) as sim:
        h = sim.hunter()
        h.actions.walk("right", 30)
        assert sim.trace.plans and sim.trace.plans[0][1] == "walk"
    real_submit = client.submit
    with pytest.raises(SimViolation):
        h.actions.walk("right", 30)  # built inside, used after: refused before any key is sent
    with pytest.raises(SimViolation):
        h.look()
    assert client.submit is real_submit


# --- seeing ------------------------------------------------------------------------------------------

def test_a_look_places_her_the_monsters_and_the_drops_where_the_world_has_them(model):
    b = plat(model, 360, 300)  # monsters are placed on platforms their kind spawns on (monsters.monster_platform)
    clock = Clock()
    w = World(model, KEYS, (b.id, 240.0), monsters=[("Snail", b.id, 380.0)], origin=clock.now,
              drops=[("mesos", b.id, 300.0)])
    eyes = FakeEyes(w, clock, model)
    h = Hunter(eyes, KEYS, model, log=lambda m: None)
    s = h.look(loot=True)
    assert s.where.x == pytest.approx(240, abs=1) and s.platform is b and s.hp == 1.0
    (t,) = s.monsters
    assert t.kind == "snail" and t.platform == b.id and t.x == pytest.approx(w.monsters[0].at(0.0), abs=1.5)
    (d,) = s.loot
    assert d.mesos and d.x == pytest.approx(300, abs=1.5) and d.platform == b.id
    assert clock.elapsed > 0  # looking takes time
    assert s.how == "rope"  # her x from the rope in view
    eyes.want_ropes = False
    s = h.look()
    assert s.how == "edge" and s.where.x == pytest.approx(240, abs=1)  # ... or from her platform's end


def test_noise_rounds_the_dot_hides_her_tag_and_dot_and_makes_stars_miss(model):
    b = plat(model, 360, 300)
    clock = Clock()
    noise = Noise(dot_px=True, tag_hidden=((0.0, 1.0),), dot_hidden=((2.0, 3.0),))
    w = World(model, KEYS, (b.id, 243.0), monsters=[("Snail", b.id, 380.0)], origin=clock.now, noise=noise)
    eyes = FakeEyes(w, clock, model)
    h = Hunter(eyes, KEYS, model, log=lambda m: None)
    s = h.look()
    assert s.dot == (round(s.dot[0]), round(s.dot[1])) and not s.tag_seen  # placed by the camera instead
    assert s.where.x == pytest.approx(243, abs=6) and s.monsters  # monsters placed from there all the same
    clock.advance_to(clock.start + 2.5)
    assert h.look().where is None and eyes.glance()[0] is None and eyes.counts["dot hidden"] == 2
    clock.advance_to(clock.start + 3.5)
    assert h.look().tag_seen and eyes.counts["tag hidden"] == 1
    c = plat(model, 500, 400)
    outcomes = []
    for seed in range(12):  # each star's miss is drawn from the seed and its tick: the same every time
        w = still(World(model, KEYS, (c.id, 100.0), seed=seed, monsters=[("Pig", c.id, 300.0)],
                        noise=Noise(miss=0.5)))
        tap(w, 0.0, "f", 0.04)
        w.advance(1.0)
        outcomes.append(w.monsters[0].hp)
    assert set(outcomes) == {2.0, 3.0}


# --- scenarios -----------------------------------------------------------------------------------------

def _synthetic(sim):
    h = sim.hunter()
    nav = sim.navigator(h)
    return nav.go(plat(sim.model, 300, 700).id)


def test_a_scenario_gives_the_same_trace_every_time(model):
    c = plat(model, 500, 400)
    sc = scenarios.Scenario("synthetic", _synthetic, (c.id, 100.0), monsters="all", seed=2, limit=200,
                            noise=Noise(dot_px=True, walk_jitter=4.0, miss=0.3, lag_sd=1.0))
    first, second = scenarios.run_scenario(sc, model), scenarios.run_scenario(sc, model)
    assert first["result"]["result"] == "arrived" and first["plans"]
    assert first == second


def test_map_data_that_is_there_but_fails_to_load_fails_instead_of_skipping(monkeypatch):
    def broken(map_id):
        raise TypeError("a refactor broke mapdata")
    monkeypatch.setattr(scenarios, "available", lambda: True)
    monkeypatch.setattr(scenarios, "_MODEL", [])
    monkeypatch.setattr(mapdata, "load", broken)
    with pytest.raises(TypeError):
        scenarios.real_map()
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # golden.main runs here, not in a fresh process
    with pytest.raises(TypeError):
        golden.main(["check", "hunt_kill"])
    monkeypatch.setattr(scenarios, "available", lambda: False)
    assert scenarios.real_map() is None
    assert golden.main(["check"]) == 2  # nothing could be compared: never a pass


REAL = pytest.mark.skipif(not scenarios.available(), reason="needs the map's game data (gamedata/data)")


@REAL
@pytest.mark.parametrize("name", list(scenarios.SCENARIOS))
def test_real_map_scenarios_show_what_they_are_there_for(name):
    trace = scenarios.run(name)
    assert not scenarios.unmet(trace), (scenarios.unmet(trace), trace["result"])


@REAL
def test_without_the_arrows_hold_back_the_walk_off_right_after_a_kill_doesnt_move_her(monkeypatch):
    """nav_walk_off_after_kill with no wait for her attack to stop swallowing arrows (actions.ARROW_LOCK_MS 0):
    the walk-off goes 0.34 s after her last throw's key, she stays on 3 as in game, and only the walk-off sent
    again once she's seen there lands."""
    from gamelab.control import actions
    monkeypatch.setattr(actions, "ARROW_LOCK_MS", 0)
    trace = scenarios.run("nav_walk_off_after_kill")
    assert set(scenarios.unmet(trace)) == {"the walk-off right after the kill lands",
                                           "no log line says 'instead of platform 4'"}
    assert trace["result"]["moves"] == ["walk_off 3->4: landed on 3", "walk_off 3->4: landed on 4"]


def _climb_then_jump(seed: int, x: float) -> dict:
    """nav_climb_then_jump from (36, x) on `seed`: its result."""
    sc = scenarios.SCENARIOS["nav_climb_then_jump"]
    return scenarios.run_scenario(dataclasses.replace(sc, seed=seed, start=(36, x)))["result"]


@REAL
def test_a_climb_a_step_back_and_two_jumps_land_on_every_seed():
    """nav_climb_then_jump on seeds 1-32 from x 300, 470 and 760 (under the rope from either side), the lags
    spread: every trip climbs 36 -> 32 and jumps on to 29 and 28 without a miss (28 of 144 trips missed one at
    the branch before: flow.STEP_EARLY_MAX_MS, TAKEOFF_SLOP)."""
    for x in (300.0, 470.0, 760.0):
        for seed in range(1, 33):
            r = _climb_then_jump(seed, x)
            assert r["result"] == "arrived" and r["misses"] == 0, (x, seed, r["moves"])


@REAL
def test_a_step_after_a_climb_pressed_before_the_top_throws_the_jumps_after_it(monkeypatch):
    """With every arrow after a climb pressed before the top again, the step back on 32 included
    (flow.STEP_EARLY_MAX_MS -1), some of nav_climb_then_jump's seeds miss 28 again: the step walks from the
    climb-over on, and the run on 29 to the jump onto 28 starts further on (7 of seeds 1-48, as in game)."""
    monkeypatch.setattr(flow, "STEP_EARLY_MAX_MS", -1)
    missed = [seed for seed in range(1, 25) if _climb_then_jump(seed, 470.0)["misses"]]
    assert missed, "every seed landed every move"


@REAL
def test_a_real_map_scenario_is_deterministic():
    assert scenarios.run("nav_monsters_long_way") == scenarios.run("nav_monsters_long_way")


@REAL
@pytest.mark.parametrize("name,topics", [
    ("nav_monsters_basin", {"session", "scene", "route", "chain", "monsters", "glance", "trip", "log"}),
    ("hunt_clear_basin", {"session", "scene", "drops", "stance", "event", "log"}),
])
def test_recording_telemetry_changes_nothing_she_does(name, topics):
    from gamelab import telemetry
    plain = scenarios.run(name)
    with telemetry.captured() as records:
        recorded = scenarios.run(name)
    assert recorded == plain  # the same key plans, log and world: telemetry only reads what's there
    assert topics <= {r["topic"] for r in records}


@REAL
def test_a_telemetry_failure_stops_recording_not_the_run():
    """A record that can't be written (a full disk, say) stops recording; the run goes on unchanged."""
    from gamelab import telemetry
    plain = scenarios.run("nav_monsters_basin")

    def full_disk(record):
        if record["topic"] == "glance":
            raise OSError(28, "No space left on device")
    before, telemetry._sink = telemetry._sink, full_disk
    try:
        recorded = scenarios.run("nav_monsters_basin")
        assert not telemetry.recording()  # recording stopped at the first failure
    finally:
        telemetry._sink = before
    assert recorded == plain


def _stepped_go(self, dst, x=None, seconds=180, max_misses=6, until=None, *, mode=None, scene=None):
    """Navigator.go written out as the hunt loop drives a trip: begin(), then step() until it's over."""
    st = self.begin(dst, x, seconds, max_misses, until, mode=mode, scene=scene)
    while (trip := self.step(st)) is None:
        pass
    return trip


@pytest.mark.parametrize("name", [n for n in scenarios.SCENARIOS if n.startswith("nav_")])
def test_a_trip_stepped_one_pass_at_a_time_is_the_same_trip(name, monkeypatch):
    if scenarios.real_map() is None:
        pytest.skip("needs the real map's data")
    from gamelab.control.navigator import Navigator
    want = scenarios.as_json(scenarios.run(name))
    monkeypatch.setattr(Navigator, "go", _stepped_go)
    assert scenarios.as_json(scenarios.run(name)) == want


def test_a_simulated_monster_faces_the_way_it_last_walked():
    """What FakeEyes reports as a monster's facing: the way it walks, turned at its platform's ends, and
    while it stands, the way it last walked."""
    import random

    from gamelab.localization.mapdata import Platform, Segment
    m = world_mod.SimMonster(0, "Pig", "pig", Platform(0, [Segment(0, 500, 400, 500)]), 200.0, random.Random(3),
                             speed=125.0)
    m.start(0.0, 200.0)
    walked = stood = 0
    last = None
    for i in range(1, 2000):
        t = i * 0.01
        x = m.at(t)
        way, walking = m.faces(t)
        if last is not None and walking and abs(x - last) > 0.5 and m.lo + 2 < x < m.hi - 2:  # not mid-bounce
            assert (x > last) == (way > 0)
            walked += 1
        if not walking:
            assert way == m.facing and way in (1, -1)
            stood += 1
        last = x
    assert walked > 300 and stood > 300
