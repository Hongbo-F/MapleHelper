"""The offline simulator (gamelab.sim): its physics against the planner's own (physics.fly, the measured walk),
the fake player's queueing and cancelling, the fakes put in place and taken away again (and nothing
reaching the real player or gamelab/lab), and scenarios that give the same trace every time and show
what they're there for. Synthetic maps need no game data; the real-map tests are skipped without it, and
fail (not skip) when it's there but doesn't load."""

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
from gamelab.sim.world import DROP_LIFE, Noise, SimDied, World, exp_pixels
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
    # The game sees the arrow 60 ms after the key and speeds her up at 1400 units/s²; let go, it sees that
    # 72 ms later and brakes her at 800 (body.run_x, body.hold_distance).
    c = plat(model, 500, 400)
    for ms in (300, 600, 1000):
        w = world(model, c.id, 100.0)
        tap(w, 0.0, "right", ms / 1000)
        if ms == 1000:
            for t in (0.05, 0.1, 0.15, 0.3, 0.6):
                w.advance(t)
                assert w.x - 100 == pytest.approx(body.run_x(t - physics.LAGS.on, 0.0, 1, physics.SHOE), abs=0.01)
            assert w.x - 100 == pytest.approx(physics.WALK_SPEED * (0.6 - physics.RUN_LAG), abs=0.5)  # flow's run model
        w.advance(3.0)
        assert w.vx == 0 and w.mode == "ground"
        assert w.x - 100 == pytest.approx(body.hold_distance(ms, physics.SHOE, physics.LAGS), abs=1)
        assert abs(hold_ms(w.x - 100) - ms) <= 10  # physics.hold_ms: the hold for what she walked


def test_taps_move_her_as_long_as_the_game_sees_them_held(model):
    c = plat(model, 500, 400)
    for hold in (0.107, 0.05, 0.02):  # 23.7, 7.4 and 2.0 units by the engine
        w = world(model, c.id, 100.0)
        tap(w, 0.0, "left", hold)
        w.advance(2.0)
        assert 100 - w.x == pytest.approx(body.hold_distance(hold * 1000, physics.SHOE, physics.LAGS), abs=1)
        assert w.facing == "left"
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
    # Her sideways speed leaving the ground (body.takeoff) with the arrow pressed `gap` s before the jump key:
    # the game sees the arrow 60 ms after its key and the jump 67 ms after its own.
    c = plat(model, 500, 400)
    for gap, vx in ((None, 0.0), (0.0, 14 + 119.2), (0.045, 149.0), (0.08, 126.0), (0.2, 149.0)):
        w = world(model, c.id, 100.0)
        if gap is not None:
            w.key(0.0, "right", True)
        tap(w, gap or 0.0, "alt", 0.04)
        w.advance((gap or 0.0) + 0.1)
        assert w.mode == "air" and w.flight.speeds[0][0] == pytest.approx(vx, abs=0.01), gap
    # Keyed 80 ms after the arrow, as the jump-back is, the lags' spread makes it two outcomes: kicked to
    # full speed, or not kicked just below it (the five recorded: 121, 124, 127, 142 and 146 units/s).
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
        w = world(model, c.id, 100.0)
        w.key(0.0, "right", True)
        tap(w, 0.5, "alt", 0.04)    # at full speed: lifts off at 0.57, lands about 0.555 s later
        then(w)
        w.advance(1.0)
        assert w.mode == "air"
        return w

    held = jump(lambda w: None)
    braked = jump(lambda w: tap(w, 0.7, "left", 0.04))
    let_go = jump(lambda w: w.key(0.55, "right", False))
    assert held.vx == 149.0  # the arrow she flies with does nothing at speed
    # left seen from 0.76 s to 0.815 (on the tick): 55 ms at 200 units/s²
    assert held.vx - braked.vx == pytest.approx(200 * 0.055, abs=0.1) and braked.facing == "left"
    assert held.vx - let_go.vx == pytest.approx(1 * (1.0 - 0.625), abs=0.01)  # no arrow: 1 unit/s² of drag


def test_landing_while_an_attack_plays_stops_her(model):
    c = plat(model, 500, 400)

    def jump(attack: bool):
        w = world(model, c.id, 100.0)
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


def test_a_jump_back_leaves_at_full_speed_and_lands_about_90_units_away(model):
    # control.actions' key plan, on the engine with the lags spread as measured: keyed 130 ms after the arrow
    # the jump is past the kick's blind spot (at 80 ms, 29% left below 140 and landed 68-89 units away).
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
    assert 84 <= min(lands) and max(lands) <= 97 and sorted(lands)[15] == pytest.approx(90, abs=1)


def test_after_a_climb_she_runs_off_as_the_minimap_traced_it(model):
    # The arrow held from before the top: she stands still through the climb-over (flow.CLIMB_OVER) and the
    # game's lag for the arrow, then speeds up: at full speed 0.35 s after the top, as flow plans it.
    c, d = plat(model, 500, 400), plat(model, 300, 700)
    w = World(model, KEYS, (c.id, 700.0), monsters=None, hanging=400.0)
    w.key(0.0, "up", True)
    w.key(0.5, "right", True)  # the top comes 100 / 130 = 0.77 s in
    w.advance(1.0)
    top = next(float(line.split()[0]) for line in w.log if "climbs onto platform" in line)
    w.advance(top + 0.6)
    # flow's model: a standing start (RUN_LAG, 0.113 s) CLIMB_OVER (0.23 s) after the top
    assert w.plat is d and w.x - 700 == pytest.approx(physics.WALK_SPEED * (0.6 - flow.CLIMB_OVER - physics.RUN_LAG), abs=0.3)


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
    w = world(model, a.id, 40.0)
    w.key(0.0, "right", True)
    tap(w, 0.93, "alt", 0.04)  # lifts off JUMP_LAG (67 ms) later, on the tick at 1.0 s: down onto B
    w.advance(1.01)
    assert w.mode == "air"
    x0, y0 = w.flight.path[0]
    assert x0 - 40 == pytest.approx(physics.WALK_SPEED * (1.0 - physics.RUN_LAG), abs=0.5)
    f = body.fly(model, x0, y0, physics.WALK_SPEED, -physics.JUMP_SPEED, shoe=physics.SHOE, arrow=1)
    land, land_x, seconds = physics.fly(model, x0, y0, physics.WALK_SPEED, -physics.JUMP_SPEED)
    assert f.platform is land is b and f.land_x == pytest.approx(land_x, abs=1.5)
    w.key(1.01, "right", False)  # let go in the air: the game sees it at 1.085 s, and she flies on from there
    w.advance(1.1)
    g = w.flight
    assert g.platform is b and g.land_x == pytest.approx(f.land_x, abs=0.5)
    # ... with no arrow: dragged at 1 unit/s², and at 100 for the last few steps, falling at the cap (670)
    assert 140 < g.vx_land < 146
    w.advance(3.0)
    assert w.plat is b and w.vx == 0
    assert w.x == pytest.approx(g.land_x + body.slide(g.vx_land, physics.SHOE), abs=0.01)  # 13 units on


def test_a_jump_key_the_game_takes_in_the_air_is_ignored(model):
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
    # Pressed 25 ms before she lands, the game takes it 42 ms after: she jumps again.
    w = world(model, c.id, 300.0)
    tap(w, 0.0, "alt", 0.04)
    tap(w, 0.6, "alt", 0.04)
    w.advance(0.66)
    assert w.mode == "ground"
    w.advance(0.7)
    assert w.mode == "air"


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
    assert w.plat is c and w.x == 800 and w.mode == "ground"


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
    # The swing reaches 90 ahead of her, into a pig's body (34 either side) up to 124 off: a star beyond.
    for x, swiped in ((420.0, True), (430.0, False)):
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
    # from 191.
    a, b, c = plat(model, 300, 100), plat(model, 360, 300), plat(model, 500, 400)

    def throw(w):
        tap(w, 0.0, "f", 0.04)
        w.advance(1.0)
        return w.monsters[0].hp

    for x, hp in ((330.0, 3), (355.0, 2)):  # 140 and 165 along
        assert throw(still(world(model, a.id, 190.0, monsters=[("Pig", b.id, x)], facing="right"))) == hp, x
    assert throw(still(world(model, a.id, 190.0, monsters=[("Orange Mushroom", b.id, 310.0)], facing="right"))) == 2
    for her, hp in ((346.0, 3), (366.0, 2)):  # 180 and 200 along
        assert throw(still(world(model, b.id, her, monsters=[("Pig", a.id, 166.0)], facing="left"))) == hp, her
    # On her level, out to 400 to its near side: a pig 430 along (its near side 396) is hit, one 440 along isn't.
    for x, hp in ((530.0, 2), (540.0, 3)):
        assert throw(still(world(model, c.id, 100.0, monsters=[("Pig", c.id, x)], facing="right"))) == hp, x


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
    # along and 60 above (in reach by distance), isn't hit, and nothing else is.
    from test_attack import step_map
    steps, a, b, _ = step_map()
    w = still(world(steps, a.id, 100.0, monsters=[("Orange Mushroom", b.id, 320.0)], facing="right"))
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert w.monsters[0].hp == 3 and any("star thrown right: no hit" in line for line in w.log)
    # Jumping, the line it runs at rises with her: keyed 100 ms after the jump the attack starts 46 up, its
    # line 74 above the platform, over the step's top (60). And from up on B a star goes down past the wall.
    w = still(world(steps, a.id, 150.0, monsters=[("Orange Mushroom", b.id, 360.0)], facing="right"))
    _jump_throw(w)
    assert w.monsters[0].hp == 2
    w = still(world(steps, b.id, 300.0, monsters=[("Orange Mushroom", a.id, 90.0)], facing="left"))
    tap(w, 0.0, "f", 0.04)
    w.advance(1.0)
    assert w.monsters[0].hp == 2


def test_a_monster_touching_her_hurts_and_knocks_her_back(model, monkeypatch):
    monkeypatch.setattr(world_mod, "EVADE", 0.0)
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 400.0, monsters=[("Pig", c.id, 420.0)]))
    w.advance(0.05)
    assert w.touches == 1 and w.hp == pytest.approx(1 - world_mod.TOUCH_DAMAGE) and w.mode == "air"
    w.advance(0.9)  # invulnerable for a second
    assert w.touches == 1 and w.plat is c and w.vx == 0
    assert 45 <= 400 - w.x <= 70  # knocked back ~58 units, away from it (lab/hits.jsonl's median push)


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


def _stepped_go(self, dst, x=None, seconds=180, max_misses=6, until=None, *, mode=None):
    """Navigator.go written out as the hunt loop drives a trip: begin(), then step() until it's over."""
    st = self.begin(dst, x, seconds, max_misses, until, mode=mode)
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
