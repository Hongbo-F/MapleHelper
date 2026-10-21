"""The movement engine (motion.body): her shoe from the game's Physics.img, one step on the ground and in
the air, the jump's kick, the closed forms the planner uses against the steps they stand for, and flights
over a map. The numbers are the measured ones (gamelab/README.md, Motion planning); a small made-up map
needs no game data, the real map's tests are skipped without it."""

import random

import numpy as np
import pytest

from maplehelper import config

from gamelab.localization import mapdata
from gamelab.motion import body, physics, route
from gamelab.motion.body import Lags, Shoe
from gamelab.sim import scenarios

from test_navigation import by_y, seg, tree

SHOE = Shoe.from_physics(body.PHYSICS_IMG, speed=119.2, jump=100)  # config.toml's defaults
LAGS = Lags()
REAL = pytest.mark.skipif(not scenarios.available(), reason="needs the map's game data (gamedata/data)")


@pytest.fixture
def model():
    return mapdata.parse(1, tree(), "Test Map", {})


def steps(v, d, seconds, dt=body.TICK):
    """(speed, distance) after `seconds` of ground_step with arrow d, from speed v."""
    x = 0.0
    for _ in range(round(seconds / dt)):
        v, dx = body.ground_step(v, d, SHOE, dt)
        x += dx
    return v, x


def mean_replay(events, until, dt=0.001):
    """replay's (t, x, y) from 0 s, the mean over every phase of the game's steps at dt (what the closed
    forms stand for), and each phase's."""
    paths = [np.array(body.replay(events, until, SHOE, LAGS, start=0.0, dt=dt, phase=ph).points)
             for ph in body.phases(dt=dt)]
    return np.mean(paths, axis=0), paths


# --- her shoe ----------------------------------------------------------------------------------------

@REAL
def test_the_physics_is_the_game_datas():
    assert scenarios.real_map().physics == body.PHYSICS_IMG


def test_her_shoe_from_physics_img_and_her_stats():
    assert SHOE.vmax == pytest.approx(149.0)  # 125 x 1.192: she walks 148-151 units/s in the recordings
    assert (SHOE.acc, SHOE.drag, SHOE.vjump, SHOE.gravity, SHOE.fall) == (1400, 800, 555, 2000, 670)
    assert (SHOE.air_acc, SHOE.air_drag, SHOE.air_drag_fall) == (200, 1, 100)
    assert SHOE.air_speed == pytest.approx(8.93, abs=0.01)
    # The game counts SPEED up to 140% and JUMP up to 123% (Haste can't take her past them).
    hasted = Shoe.from_physics(body.PHYSICS_IMG, speed=160, jump=130)
    assert hasted.vmax == pytest.approx(175) and hasted.vjump == pytest.approx(555 * 1.23)


def test_physics_takes_her_stats_from_the_config():
    assert physics.SHOE == Shoe.from_physics(body.PHYSICS_IMG, physics.CHARACTER.speed, physics.CHARACTER.jump)


def test_the_tests_take_her_measured_stats_whatever_the_config_says(tmp_path):
    # Editing config.toml's [character] for her gear mustn't break the tests, change the golden traces or
    # the harness's runs: those take the measured stats (FIXED_STATS); the hunt takes the file's.
    assert physics.CHARACTER == config.Character()
    f = tmp_path / "config.toml"
    f.write_text("[character]\nspeed = 125\njump = 110\n", encoding="utf-8")
    assert physics.stats(f, environ={}) == config.Character(speed=125, jump=110)
    assert physics.stats(f, environ={physics.FIXED_STATS: "1"}) == config.Character()


def test_turning_from_a_run_lags_a_standing_start_by_the_braking():
    # Kept for its name: from full speed the other way she brakes 0.106 s at 1400 (7.9 units on), then
    # speeds up as from standing.
    assert physics.TURN_EXTRA == pytest.approx(1.5 * 149 / 1400, abs=1e-3)


# --- on the ground --------------------------------------------------------------------------------------

def test_a_standing_start_speeds_up_at_1400_then_runs_at_149():
    v, x = steps(0.0, 1, 0.105)
    assert v == pytest.approx(147, abs=0.1) and x == pytest.approx(7.7, abs=0.1)  # 1400 t, 700 t²
    assert steps(0.0, 1, 0.11)[0] == SHOE.vmax  # full speed 0.106 s after the game sees the arrow
    # From the key: the game acts on the arrow LAGS.on later on average, so she's 149 (t - 0.119) along for t
    # past ~0.17 s, over the phases of its steps; at each, up to half a step's run (2.2 units) either way.
    mean, paths = mean_replay([(0.0, "right", True)], 1.0)
    later = mean[:, 0] >= 0.2
    assert np.abs(mean[later, 1] - 149 * (mean[later, 0] - 0.1192)).max() < 0.1 and not mean[:, 2].any()
    assert max(np.abs(p[later, 1] - mean[later, 1]).max() for p in paths) < 2.3


def test_letting_go_she_goes_on_24_units():
    assert body.slide(149, SHOE) == pytest.approx(13.9, abs=0.05)  # 149²/1600, on landing too
    assert body.stop_distance(149, SHOE, LAGS) == pytest.approx(24.0, abs=0.1)  # measured 24.0 ± 2.0 (18 stops)
    assert body.stop_time(149, SHOE, LAGS) == pytest.approx(0.254, abs=0.001)
    mean, paths = mean_replay([(0.0, "right", True), (1.0, "right", False)], 2.0)
    i = np.searchsorted(mean[:, 0], 1.0)
    assert mean[-1, 1] - mean[i, 1] == pytest.approx(24.0, abs=0.1)
    assert all(p[-1, 1] - p[i, 1] == pytest.approx(24.0, abs=2.3) for p in paths)  # half a step's run either way


def test_turning_round_brakes_and_turns_at_1400():
    # From full speed one way to full speed the other: 0.21 s, going 7.9 units on first.
    assert body.run_time(0.0, -149, 1, SHOE) == pytest.approx(0.213, abs=0.001)
    assert body.run_x(0.1064, -149, 1, SHOE) == pytest.approx(-7.93, abs=0.01)
    v, x = steps(-149.0, 1, 0.215)
    assert v == SHOE.vmax and x == pytest.approx(0.3, abs=0.5)


@pytest.mark.parametrize("v0, d", [(0.0, 1), (60.0, 1), (-149.0, 1), (149.0, -1), (200.0, 1), (149.0, 1)])
def test_run_x_and_run_time_are_the_steps_closed_form(v0, d):
    for tau in (0.03, 0.08, 0.15, 0.4, 1.0):
        v, x = steps(v0, d, tau)
        assert body.run_x(tau, v0, d, SHOE) == pytest.approx(x, abs=0.05)
        dist = d * body.run_x(tau, v0, d, SHOE)
        if v * d > 0:  # on her way there (turning round, she first passes it going the other way)
            assert body.run_time(dist, v0, d, SHOE) == pytest.approx(tau, abs=1e-6)


def test_holds_follow_the_tap_model():
    # The game sees a hold 2 ms longer than it is (off - on), and she brakes from what she reached.
    # On average over the game's steps: the line between whole steps of it seen held (1.8, 7.0, 15.7, 23.7).
    assert [round(body.steps_run(k, SHOE), 2) for k in range(6)] == [0, 1.8, 7.02, 15.66, 23.73, 28.2]
    assert body.hold_distance(50, SHOE, LAGS) == pytest.approx(1.8 + (52 / 30 - 1) * 5.22, abs=0.01)  # 5.6
    assert body.hold_distance(300, SHOE, LAGS) == pytest.approx(6.15 + 0.149 * 300, abs=0.05)
    assert body.hold_distance(-20, SHOE, LAGS) == 0
    assert body.hold_for(5, SHOE, LAGS) == 46 and body.hold_for(20, SHOE, LAGS) == 104
    assert body.hold_for(50, SHOE, LAGS) == pytest.approx(294, abs=2)
    for units in (2, 5, 12, 20, 21.8, 30, 50, 120):
        # (whole ms: half a ms is up to 0.2 units around full speed, where she's quickest to brake from)
        assert body.hold_distance(body.hold_for(units, SHOE, LAGS), SHOE, LAGS) == pytest.approx(units, abs=0.25)
    for ms in (30, 60, 90, 150, 400):  # the closed form is the mean of what the steps give over their phase
        mean, paths = mean_replay([(0.0, "right", True), (ms / 1000, "right", False)], 1.5)
        assert mean[-1, 1] == pytest.approx(body.hold_distance(ms, SHOE, LAGS), abs=0.05)
    # One phase at a time a short tap is bimodal: a 50 ms tap is held one step of speeding up or two (as in
    # game: of three 50 ms taps, two moved her 0 and one 7.1).
    _, paths = mean_replay([(0.0, "right", True), (0.05, "right", False)], 1.0)
    assert sorted({round(p[-1, 1], 1) for p in paths}) == [1.8, 7.0]


@pytest.mark.parametrize("v0", [0.0, 60.0, 127.0, 149.0])
def test_held_for_brings_her_to_rest_where_asked_from_any_speed(v0):
    # flow plans its stops with it: the game must see the arrow held this long from v0, then she slides.
    for units in (5.0, 15.0, 21.8, 30.0, 80.0):
        tau = body.held_for(units, v0, SHOE)
        v, x = steps(v0, 1, round(tau / 0.0001) * 0.0001, dt=0.0001)
        if units <= body.slide(v0, SHOE):
            assert tau == 0  # she slides that far anyway
            continue
        assert x + body.slide(v, SHOE) == pytest.approx(units, abs=0.05)
        assert body.run_speed(tau, v0, 1, SHOE) == pytest.approx(v, abs=0.2)
    # From standing, hold_for is that over the game's steps (within 5 ms), the game seeing a hold 2 ms longer.
    assert body.hold_for(20, SHOE, LAGS) == pytest.approx(round(1000 * body.held_for(20, 0.0, SHOE)) - 2, abs=5)


def test_a_tap_has_her_standing_still_again_when_hold_time_says():
    # A 50 ms tap (a turn): seen held 52 ms, 73 units/s when the game acts on its release 118 ms after the key,
    # braked to a stop 91 ms later. Over the game's steps she is within a quarter unit of where she comes to
    # rest by then (the steps brake her 24 units/s at a time, the last one to a stop: 13 ms later on average).
    assert body.hold_time(50, SHOE, LAGS) == pytest.approx(0.209, abs=0.0005)
    mean, _ = mean_replay([(0.0, "right", True), (0.05, "right", False)], 0.5)
    assert mean[-1, 1] - np.interp(body.hold_time(50, SHOE, LAGS), mean[:, 0], mean[:, 1]) < 0.25


# --- the jump ----------------------------------------------------------------------------------------

def test_the_kick():
    assert body.takeoff(0, 1, SHOE) == pytest.approx(119.2)    # from standing: 0.8 x 149
    assert body.takeoff(23.3, 1, SHOE) == pytest.approx(142.5)  # jb_k190: the arrow seen a step before
    assert body.takeoff(100, 1, SHOE) == 149                    # kicked past full speed: capped
    assert body.takeoff(130, 1, SHOE) == 130                    # jb_k165/450: just below full speed, no kick
    assert body.takeoff(149, 1, SHOE) == 149
    assert body.takeoff(-149, 1, SHOE) == pytest.approx(-29.8)  # running the other way
    assert body.takeoff(60, 0, SHOE) == 60                      # no arrow: as she was going


def test_a_jump_keyed_in_the_kicks_blind_spot_is_flagged():
    # The game's steps: after k steps of the arrow she runs at 42 k (to 149), and the kick takes 42 and 84 to
    # full speed, not 126. In game (2026-10-21) jumps keyed 80, 100 and 120 ms after the arrow left short 2,
    # 3 and 3 times of 4; none of 10 keyed 130-160 ms after it, nor of 4 at 40.
    assert body.short_takeoff_chance(80, SHOE, LAGS) == pytest.approx(0.58, abs=0.01)
    assert body.short_takeoff_chance(100, SHOE, LAGS) == pytest.approx(0.62, abs=0.01)
    assert body.short_takeoff_chance(120, SHOE, LAGS) == pytest.approx(0.16, abs=0.01)
    assert body.short_takeoff_chance(150, SHOE, LAGS) < 0.001
    assert body.short_takeoff_chance(40, SHOE, LAGS) == pytest.approx(0.045, abs=0.005)
    assert body.short_takeoff_chance(0, SHOE, LAGS) == pytest.approx(0.87, abs=0.01)  # 119 from the arrow's step
    assert [body.jump_window_ok(g, SHOE, LAGS) for g in (0, 20, 40, 50, 60, 80, 100, 112, 130, 166)] == \
        [False, False, True, True, False, False, False, False, True, True]
    assert body.blind_ends_ms(SHOE, LAGS) == 129


def test_the_steps_chances_are_the_triangle_over_the_lags_spread():
    # With no spread, a gap of 1.25 steps between the game seeing the arrow and the jump is 1 step or 2 by the
    # phase of its grid (3 in 4 and 1 in 4); with the spread, each k's chance is that averaged.
    still = body.Lags(sd_on=0.0, sd_jump=0.0)
    gap = 1.25 * body.STEP * 1000 - (still.jump - still.on) * 1000
    chances = {k: p for k, p in body.steps_held_chances(gap, still).items() if p > 1e-9}
    assert chances == pytest.approx({1: 0.75, 2: 0.25})
    spread = body.steps_held_chances(100, LAGS)
    assert sum(spread.values()) == pytest.approx(1.0) and max(spread, key=spread.get) == 3
    # And replayed over the grid's phases, a jump 80 ms after the arrow leaves as the chances say.
    events = [(0.0, "right", True), (0.08, "jump", True), (0.12, "jump", False), (0.8, "right", False)]
    speeds = [abs(body.replay(events, 0.4, SHOE, still, start=0.0, dt=0.001, phase=ph).takeoffs[0][2])
              for ph in body.phases(dt=0.001)]
    p = body.short_takeoff_chance(80, SHOE, still)
    assert sum(v < 140 for v in speeds) / len(speeds) == pytest.approx(p, abs=0.04)
    assert {round(v) for v in speeds} == {126, 149}


def test_a_jump_key_in_the_air_is_ignored_and_holding_it_doesnt_jump_again():
    events = [(0.0, "jump", True), (0.04, "jump", False), (0.3, "jump", True), (0.9, "jump", False)]  # held_jump
    r = body.replay(events, 1.5, SHOE, LAGS, start=0.0)  # the game's steps at 0, 30, 60 ms ...
    assert len(r.takeoffs) == 1 and r.takeoffs[0][0] == pytest.approx(0.060) and r.takeoffs[0][2] == 0
    top = min(y for _, _, y in r.points)
    assert top == pytest.approx(-77.0, abs=0.1) and r.points[-1][2] == 0  # 555²/4000 up, and back down


# --- in the air ----------------------------------------------------------------------------------------

def air(vx, vy, d, seconds):
    for _ in range(round(seconds / body.TICK)):
        vx, vy, _, _ = body.air_step(vx, vy, d, SHOE)
    return vx, vy


def test_steering_in_the_air_brakes_at_200_and_never_speeds_her_up():
    assert air(149.0, -555.0, -1, 0.04)[0] == pytest.approx(149 - 8, abs=0.5)  # the jump-backs: -4 to -15
    assert air(149.0, -555.0, 1, 0.3)[0] == 149       # the arrow she runs with: nothing (jb_k450: 124 -> 124)
    assert air(0.0, -555.0, 1, 0.3)[0] == pytest.approx(SHOE.air_speed)  # straight up: a drift of 8.9
    assert 148 <= air(149.0, -555.0, 0, 0.555)[0] < 149  # no arrow: 1 unit/s² of drag
    vx, vy = air(149.0, 670.0, 0, 0.1)
    assert vy == 670 and vx == pytest.approx(139, abs=0.01)  # at the fall cap: 100 units/s²


def test_a_running_jump_lands_where_the_arc_says_to_a_step(model):
    a, b = by_y(model, 500, 400), by_y(model, 360, 300)
    f = body.fly(model, 500, 500, 149.0, -555.0, shoe=SHOE, arrow=1)
    exact = 500 + 149 * 0.555  # flat: 2 x 555 / 2000 s in the air
    assert f.platform is a and exact <= f.land_x <= exact + 149 * body.TICK
    assert f.vx_land == 149 and f.seconds == pytest.approx(0.555, abs=body.TICK)
    braked = body.fly(model, 500, 500, 149.0, -555.0, shoe=SHOE, arrow=-1)
    assert braked.land_x == pytest.approx(f.land_x - 200 * 0.555 ** 2 / 2, abs=1)  # 30.8 shorter
    assert braked.vx_land == pytest.approx(149 - 200 * 0.555, abs=1.5)
    # Off A's end down onto B, 60 lower: her arc crosses B's surface on the way down.
    down = body.fly(model, 150, 300, 149.0, -555.0, shoe=SHOE, arrow=1, path=True)
    assert down.platform is b and down.land_x == pytest.approx(150 + 149 * 0.648, abs=1)
    assert down.path[0] == (150, 300) and down.path[-1] == (down.land_x, 360)
    assert len(down.path) == round(down.seconds / body.TICK) + 1


def test_a_flight_carried_on_from_any_step_is_the_same_flight(model):
    # The simulator flies her on from where she is when the arrow the game sees changes: carried on with
    # the same arrow, from any step (its speeds kept with its path), a flight lands where it did, when it did.
    a, b = by_y(model, 500, 400), by_y(model, 360, 300)
    for x, vx, arrow in ((150.0, 149.0, 1), (150.0, 149.0, 0), (500.0, -120.0, -1)):
        f = body.fly(model, x, 300 if x == 150 else 500, vx, -555.0, shoe=SHOE, arrow=arrow, path=True)
        assert len(f.speeds) == len(f.path) and f.platform in (a, b)
        for k in range(1, len(f.path) - 1, 7):
            (x1, y1), (vx1, vy1) = f.path[k], f.speeds[k]
            g = body.fly(model, x1, y1, vx1, vy1, shoe=SHOE, arrow=arrow, airborne=True)
            assert g.platform is f.platform and g.land_x == pytest.approx(f.land_x, abs=1e-6), k
            assert k * body.TICK + g.seconds == pytest.approx(f.seconds, abs=1e-6) and g.vx_land == pytest.approx(f.vx_land)


def test_walls_and_the_maps_sides_stop_her(model):
    c = by_y(model, 500, 400)
    f = body.fly(model, 790, 499, 149.0, -555.0, shoe=SHOE, arrow=0)
    assert f.platform is c and 799 < f.land_x < 800 and f.vx_land == 0  # up against the wall at 800
    walls = body.vertical_walls(model.walls)
    assert walls == ([0, 800], [(399, 501), (399, 501)])
    open_sky = body.fly(model, 790, 380, 149.0, -555.0, shoe=SHOE, walls=([], []), sides=(-100, 850))
    assert open_sky.platform is None and open_sky.land_x == 850  # past the wall's top: the side stops her


def test_a_wall_at_her_start_stops_her_only_going_into_it(model):
    # Standing at C's right end against the wall that rises there (where the simulator stops her walking): a
    # jump away from it leaves, one into it goes straight up. (Counted in her way both ways, the jump away
    # went straight up too, and landed where it started.)
    c = by_y(model, 500, 400)
    away = body.fly(model, 800, 500, -149.0, -555.0, shoe=SHOE, arrow=-1)
    assert away.platform is c and away.land_x == pytest.approx(800 - 149 * 0.555, abs=1)
    into = body.fly(model, 800, 500, 149.0, -555.0, shoe=SHOE, arrow=1)
    assert into.platform is c and into.land_x == 800 and into.vx_land == 0
    assert physics.fly(model, 800, 500, -149.0, -555.0)[1] == away.land_x
    # In the air nothing says which side of a wall at her x she's on: it stops her either way.
    assert body.fly(model, 800, 450, -149.0, 0.0, shoe=SHOE, airborne=True).land_x == 800


def test_in_the_air_only_her_own_foothold_groups_walls_stop_her(model, monkeypatch):
    # C's walls are C's group's: they stop her jumping off C, not someone come from A (another group's).
    a, c = by_y(model, 300, 100), by_y(model, 500, 400)
    assert body.walls_for(model, c.group) == model.walls and body.walls_for(model, a.group) == []
    assert body.walls_for(model, body.ROPE_GROUP) == [] and body.walls_for(model, body.EVERY_WALL) == model.walls
    own = body.fly(model, 790, 499, 149.0, -555.0, shoe=SHOE, group=c.group)
    assert own.platform is c and 799 < own.land_x < 800 and own.vx_land == 0
    other = body.fly(model, 790, 499, 149.0, -555.0, shoe=SHOE, group=a.group, sides=(-100, 850))
    assert other.platform is None and other.land_x == 850  # through the wall, out to the map's side
    monkeypatch.setattr(body, "OWN_GROUP_WALLS", False)  # every wall stops her, as before
    assert body.walls_for(model, a.group) == model.walls
    assert body.fly(model, 790, 499, 149.0, -555.0, shoe=SHOE, group=a.group).land_x == own.land_x


def _ledge_map():
    """Platform 3's right end on the real map, made up: a ledge L (0..200, y 300) whose walk-off passes,
    12 units on, the wall hanging 85 units under the left end of a block H (212..600, y 240) of another
    group; F (212..600) is 600 below L, and a basin floor under everything is 1200 below."""
    t = tree()
    t["info"].update(VRLeft=-200, VRRight=1200, VRBottom=1600)
    t["foothold"] = {"1": {
        "10": dict([seg(1, 10, 1, 0, 300, 200, 300), seg(1, 10, 2, 200, 300, 200, 325)]),
        "11": dict([seg(1, 11, 3, 212, 325, 212, 240, 0, 4), seg(1, 11, 4, 212, 240, 600, 240, 3, 0)]),
        "12": dict([seg(1, 12, 5, 212, 900, 600, 900)]),
        "13": dict([seg(1, 13, 6, -100, 1500, 1000, 1500)]),
    }}
    t["life"] = {}
    m = mapdata.parse(1, t, "Test Map", {})
    return m, by_y(m, 300, 100), by_y(m, 900, 400), by_y(m, 1500, 400)


def test_walking_off_through_another_groups_wall_lands_beyond_it(monkeypatch):
    m, ledge, floor, basin = _ledge_map()
    f = body.fly(m, 200.5, 299, 149.0, 0.0, shoe=SHOE, arrow=1, group=ledge.group)
    assert f.platform is floor and f.land_x == pytest.approx(200.5 + 149 * f.seconds, abs=1) and f.vx_land == 149
    # Every wall stopping her, she comes down along H's wall: into the basin with the arrow let go (where
    # the planner, which took no notice of the arrow before it flew her by the engine, had platform 3's
    # walk-off going: into the basin, 36, where in game she landed on 17 three times out of three); with the
    # arrow held, the air steers her round the wall's foot onto F, 8 units past it.
    every = body.fly(m, 200.5, 299, 149.0, 0.0, shoe=SHOE, arrow=0)
    assert every.platform is basin and 211 < every.land_x < 212
    round_foot = body.fly(m, 200.5, 299, 149.0, 0.0, shoe=SHOE, arrow=1)
    assert round_foot.platform is floor and 212 < round_foot.land_x < 221 and round_foot.vx_land < 9

    def walk_off() -> list[tuple[int, int]]:  # where walking off L's right end lands, by the planner
        return [(mv.dst, round(mv.land_x)) for mv in route.moves(m)
                if (mv.kind, mv.src, mv.direction) == ("walk_off", ledge.id, "right")]

    assert walk_off() == [(floor.id, round(f.land_x))]  # the planner flies her by the engine
    monkeypatch.setattr(body, "OWN_GROUP_WALLS", False)
    assert walk_off() == [(floor.id, round(round_foot.land_x))]


def test_a_step_through_a_platforms_corner_lands_on_it(model):
    # Falling at 180 units/s sideways onto B (220..400, y 360) from 5 units above: she lands on it exactly
    # when her arc crosses its surface by its end, even when the step that does it ends past the end.
    # (The planner's own integration looked only at where each step ends: dozens of these went through B.)
    b = by_y(model, 360, 300)
    for i in range(300):
        x0 = 386 + i * 0.01
        cross = x0 + 180 * (2 * 5 / 2000) ** 0.5  # where the arc comes down through y 360
        f = body.fly(model, x0, 355.0, 180.0, 0.0, shoe=SHOE, arrow=1)
        if abs(cross - 400) > 0.02:
            assert (f.platform is b) == (cross < 400), x0
        assert physics.fly(model, x0, 355.0, 180.0, 0.0)[0] is f.platform  # the planner's is the engine's


@REAL
def test_a_knock_in_the_air_by_the_basins_end_lands_in_the_basin():
    # Harness tour seed 10: knocked mid jump-throw at x 887, 46 units up, she went through the corner of
    # 36's right end (951) under the foot of the wall above it, and fell out of the map for good.
    m = scenarios.real_map()
    f = body.fly(m, 887.45, 529.46, 180.0, -220.0, shoe=physics.SHOE)
    assert f.platform.id == 36 and 950 < f.land_x <= 951
    land, x, _ = physics.fly(m, 887.45, 529.46, 180.0, -220.0)  # (its own integration had none: out of the map)
    assert land.id == 36 and 950 < x <= 951


def test_long_falls_with_the_arrow_let_go_land_shorter():
    t = tree()
    t["foothold"]["5"] = {"9": dict([seg(1, 9, 30, -2000, 1000, 3000, 1000)])}  # a floor 500 below C
    m = mapdata.parse(1, t, "Test Map", {})
    held = body.fly(m, 801, 499, 149.0, 0.0, shoe=SHOE, arrow=1, walls=([], []))
    free = body.fly(m, 801, 499, 149.0, 0.0, shoe=SHOE, arrow=0, walls=([], []))
    assert held.platform is free.platform is not None
    # At the fall cap (670, reached 0.335 s and 112 units down) the air drags her at 100 units/s² with no
    # arrow: 0.58 s more down to the floor, 16.8 units shorter.
    assert held.land_x - free.land_x == pytest.approx(16.9, abs=1)
    assert held.vx_land == 149 and free.vx_land == pytest.approx(149 - 0.3 - 58, abs=1.5)


@REAL
@pytest.mark.parametrize("own_group", [True, False])
def test_the_planners_moves_land_where_the_engine_has_them(own_group, monkeypatch):
    """Every move the route planner finds lands where the engine has it, on the same platform within 0.05
    units, by either wall rule: the planner flies her by the engine (physics.flight is body.fly from its
    start rounded to 0.01 units, with the arrow held the way she leaves). It had an integration of its own
    before, by the speed at each step's start (a unit higher at the top: 78.1 for 77.0), which took no
    notice of the arrow: with every wall stopping her, its walk-off off platform 3 slid down platform 1's
    wall into the basin (36), where the engine's arrow steers her round the wall's foot onto 17. In game
    she went through that wall (OWN_GROUP_WALLS)."""
    monkeypatch.setattr(body, "OWN_GROUP_WALLS", own_group)
    m = scenarios.real_map()
    far, other = [], []
    for mv in route.moves(m):
        a, sign = m.platform(mv.src), 1 if mv.direction == "right" else -1
        edge = a.right if sign > 0 else a.left
        if mv.kind == "jump":
            x = mv.takeoff if mv.takeoff is not None else edge - sign * route.JUMP_FROM
            start, arrow, skip = (x, a.y_at(x), sign * physics.WALK_SPEED, -physics.JUMP_SPEED), sign, None
        elif mv.kind == "jump_up":
            start, arrow, skip = (mv.x, a.y_at(mv.x), 0.0, -physics.JUMP_SPEED), 0, None
        elif mv.kind == "drop":
            start, arrow, skip = (mv.x, a.y_at(mv.x) + 1, 0.0, 0.0), 0, a
        elif mv.kind == "walk_off":
            level = a.y_at(edge)
            rising = any(w.x1 == w.x2 and abs(w.x1 - edge) < 1 and min(w.y1, w.y2) < level - 2 <= max(w.y1, w.y2)
                         for w in body.walls_for(m, a.group))
            x = edge + sign * (-1 if rising else 0.5)
            start, arrow, skip = (x, level - 1, sign * physics.WALK_SPEED, 0.0), sign, None
        else:
            continue
        f = body.fly(m, *start, shoe=physics.SHOE, arrow=arrow, skip=skip, group=a.group)
        if f.platform is None or f.platform.id != mv.dst:
            other.append((mv.kind, mv.src, mv.dst, f.platform and f.platform.id))
        elif abs(f.land_x - mv.land_x) > 0.05:
            far.append((mv.kind, mv.src, mv.dst))
    assert other == [] and far == []


@REAL
def test_a_drop_lands_her_well_inside_the_platform_it_lands_on():
    """Every drop goes from the middle of where her platform overlaps the one it lands on (LAND_MARGIN in from
    both ends). Taken from its overlap with any platform, 27 -> 36 went from 27's overlap with 2 (above it), 6
    units from the basin's cliff at 36's left end, and 33 -> 36 from 6 units off its right one. In the air there
    the walls of 36's group don't stop her (body.OWN_GROUP_WALLS), and a touch's knock carried her through them
    and out of the map (harness, tour seed 25 and loop seed 19). From 27 only the 12 units by that cliff drop
    her onto 36 (28 is under the rest), so 27 has no drop to 36. Nor does one come down within KNOCK_CARRY (70)
    of those cliffs, 36's ends, the only ends here with nothing under them: dropped from 33 at 907, 44 from the
    right one, a pig's knock still took her through it."""
    m = scenarios.real_map()
    assert [(q.id, end) for q in m.platforms for end in route.void_ends(m, q)] == [(36, 129), (36, 951)]
    drops = [mv for mv in route.moves(m) if mv.kind == "drop"]
    for mv in drops:
        q = m.platform(mv.dst)
        assert q.left + physics.LAND_MARGIN <= mv.land_x <= q.right - physics.LAND_MARGIN, mv
        assert all(abs(mv.land_x - end) >= route.KNOCK_CARRY for end in route.void_ends(m, q)), mv
    into_basin = {mv.src: mv.x for mv in drops if mv.dst == 36}
    assert 27 not in into_basin and into_basin[33] == 881 and into_basin[31] == 225
    assert route.drop_spots(m, m.platform(33), m.platform(36)) == [879, 881]


@REAL
def test_off_platform_3s_right_end_she_goes_through_platform_1s_wall_onto_17():
    # Three times out of three in game (telemetry, 2026-10-20), landing at 1,005-1,015 by her dot.
    m = scenarios.real_map()
    p3 = m.platform(3)
    assert p3.group != m.platform(1).group
    f = body.fly(m, 849.5, -926, 149.0, 0.0, shoe=physics.SHOE, arrow=1, group=p3.group)
    assert f.platform.id == 17 and f.land_x == pytest.approx(1008, abs=2)
    assert physics.fly(m, 849.5, -926, 149.0, 0.0, group=p3.group)[0].id == 17
    # Every wall stopping her: down it into the basin with the arrow let go, or (held) round its foot onto 17
    # at 869, 140 units short of where she came down in game.
    assert physics.fly(m, 849.5, -926, 149.0, 0.0, arrow=0)[0].id == 36
    land, x, _ = physics.fly(m, 849.5, -926, 149.0, 0.0)
    assert land.id == 17 and x == pytest.approx(869, abs=2)
    kinds = {(mv.kind, mv.dst) for mv in route.moves(m) if mv.src == 3 and mv.direction == "right"}
    assert ("walk_off", 17) in kinds and ("walk_off", 36) not in kinds


def test_physics_fly_and_body_fly_land_alike_on_random_running_jumps(model):
    rng = random.Random(0)
    for _ in range(50):
        x = rng.uniform(0, 800)
        sign = rng.choice((-1, 1))
        land, lx, _ = physics.fly(model, x, 500, sign * 149.0, -555.0)
        f = body.fly(model, x, 500, sign * 149.0, -555.0, shoe=SHOE, arrow=sign)
        assert f.platform is land
        assert f.land_x == pytest.approx(lx, abs=0.006)  # (physics.fly starts from x rounded to 0.01)
