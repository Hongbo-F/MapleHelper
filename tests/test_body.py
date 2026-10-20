"""The movement engine (motion.body): her shoe from the game's Physics.img, one step on the ground and in
the air, the jump's kick, the closed forms the planner uses against the steps they stand for, and flights
over a map. The numbers are the measured ones (gamelab/README.md, Motion planning); a small made-up map
needs no game data, the real map's tests are skipped without it."""

import random

import pytest

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


# --- on the ground --------------------------------------------------------------------------------------

def test_a_standing_start_speeds_up_at_1400_then_runs_at_149():
    v, x = steps(0.0, 1, 0.105)
    assert v == pytest.approx(147, abs=0.1) and x == pytest.approx(7.7, abs=0.1)  # 1400 t, 700 t²
    assert steps(0.0, 1, 0.11)[0] == SHOE.vmax  # full speed 0.106 s after the game sees the arrow
    # From the key: the arrow is seen LAGS.on later, so she's 149 (t - 0.113) along for t past ~0.17 s.
    r = body.replay([(0.0, "right", True)], 1.0, SHOE, LAGS, start=0.0)
    for t, x, y in r.points:
        if t >= 0.2:
            assert x == pytest.approx(149 * (t - 0.113), abs=0.3) and y == 0
    assert r.takeoffs == []


def test_letting_go_she_goes_on_24_6_units():
    assert body.slide(149, SHOE) == pytest.approx(13.9, abs=0.05)  # 149²/1600, on landing too
    assert body.stop_distance(149, SHOE, LAGS) == pytest.approx(24.6, abs=0.2)  # measured 23.4-28.1
    assert body.stop_time(149, SHOE, LAGS) == pytest.approx(0.26, abs=0.005)
    r = body.replay([(0.0, "right", True), (1.0, "right", False)], 2.0, SHOE, LAGS, start=0.0)
    at_release = next(x for t, x, _ in r.points if t >= 1.0)
    assert r.points[-1][1] - at_release == pytest.approx(24.6, abs=0.8)  # within a step's travel


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
    # The game sees a hold 12 ms longer than it is (off - on), and she brakes from what she reached.
    assert body.hold_distance(50, SHOE, LAGS) == pytest.approx(1925 * 0.062 ** 2, abs=0.05)  # 7.4; game: 7.1 once
    assert body.hold_distance(300, SHOE, LAGS) == pytest.approx(7.74 + 0.149 * 300, abs=0.05)
    assert body.hold_distance(-20, SHOE, LAGS) == 0
    assert body.hold_for(5, SHOE, LAGS) == pytest.approx(39, abs=1)
    assert body.hold_for(50, SHOE, LAGS) == pytest.approx(283, abs=2)
    for units in (2, 5, 12, 20, 21.8, 30, 50, 120):
        # (whole ms: half a ms is up to 0.2 units around full speed, where she's quickest to brake from)
        assert body.hold_distance(body.hold_for(units, SHOE, LAGS), SHOE, LAGS) == pytest.approx(units, abs=0.25)
    for ms in (30, 60, 90, 150, 400):  # the closed form is what the steps give (1 ms steps: a release can
        # come up to a step late, and around 100 ms each ms held is 0.4 units)
        r = body.replay([(0.0, "right", True), (ms / 1000, "right", False)], 1.5, SHOE, LAGS, start=0.0, dt=0.001)
        assert r.points[-1][1] == pytest.approx(body.hold_distance(ms, SHOE, LAGS), abs=0.5)


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
    # From standing that is hold_for, the game seeing a hold 12 ms longer than it is.
    assert body.hold_for(20, SHOE, LAGS) == round(1000 * body.held_for(20, 0.0, SHOE)) - 12


def test_a_tap_has_her_standing_still_again_when_hold_time_says():
    # A 50 ms tap (a turn): seen held 62 ms, 87 units/s when the game sees it let go 122 ms after the key,
    # braked to a stop 109 ms later.
    assert body.hold_time(50, SHOE, LAGS) == pytest.approx(0.2305, abs=0.0005)
    r = body.replay([(0.0, "right", True), (0.05, "right", False)], 0.5, SHOE, LAGS, start=0.0, dt=0.001)
    last = next(t for (t, x, _), (_, nx, _) in zip(r.points, r.points[1:]) if nx == x and t > 0.1)
    assert last == pytest.approx(body.hold_time(50, SHOE, LAGS), abs=0.002)


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
    # 80 ms after the arrow (the jump-back's timing): 35% of jumps leave below 140 (three of five did).
    assert body.short_takeoff_chance(80, SHOE, LAGS) == pytest.approx(0.35, abs=0.02)
    assert body.short_takeoff_chance(40, SHOE, LAGS) == pytest.approx(0.03, abs=0.01)
    assert body.short_takeoff_chance(0, SHOE, LAGS) == pytest.approx(0.69, abs=0.02)  # often not seen yet
    assert [body.jump_window_ok(g, SHOE, LAGS) for g in (0, 20, 40, 50, 60, 80, 100, 112, 130, 166)] == \
        [False, False, True, True, False, False, False, False, True, True]


def test_a_jump_key_in_the_air_is_ignored_and_holding_it_doesnt_jump_again():
    events = [(0.0, "jump", True), (0.04, "jump", False), (0.3, "jump", True), (0.9, "jump", False)]  # held_jump
    r = body.replay(events, 1.5, SHOE, LAGS, start=0.0)
    assert len(r.takeoffs) == 1 and r.takeoffs[0][0] == pytest.approx(0.067, abs=body.TICK) and r.takeoffs[0][2] == 0
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
    assert physics.fly(m, 200.5, 299, 149.0, 0.0, group=ledge.group)[0] is floor
    # Every wall stopping her, she'd come down along H's wall into the basin (as the planner had platform 3's
    # walk-off going into the basin, 36, where in game she landed on 17 three times out of three). (With the
    # arrow let go: held, it steers her round the foot of the wall onto F.)
    every = body.fly(m, 200.5, 299, 149.0, 0.0, shoe=SHOE, arrow=0)
    assert every.platform is basin and 211 < every.land_x < 212
    assert physics.fly(m, 200.5, 299, 149.0, 0.0)[0] is basin
    monkeypatch.setattr(body, "OWN_GROUP_WALLS", False)
    assert physics.fly(m, 200.5, 299, 149.0, 0.0, group=ledge.group)[0] is basin

    def walk_off() -> list[int]:  # where walking off L's right end lands, by the planner
        return [mv.dst for mv in route.moves(m) if (mv.kind, mv.src, mv.direction) == ("walk_off", ledge.id, "right")]

    assert walk_off() == [basin.id]
    monkeypatch.setattr(body, "OWN_GROUP_WALLS", True)
    assert walk_off() == [floor.id]


def test_a_step_through_a_platforms_corner_lands_on_it(model):
    # Falling at 180 units/s sideways onto B (220..400, y 360) from 5 units above: she lands on it exactly
    # when her arc crosses its surface by its end, even when the step that does it ends past the end.
    b = by_y(model, 360, 300)
    leaks = 0
    for i in range(300):
        x0 = 386 + i * 0.01
        cross = x0 + 180 * (2 * 5 / 2000) ** 0.5  # where the arc comes down through y 360
        f = body.fly(model, x0, 355.0, 180.0, 0.0, shoe=SHOE, arrow=1)
        if abs(cross - 400) > 0.02:
            assert (f.platform is b) == (cross < 400), x0
        if (cross < 400) and physics.fly(model, x0, 355.0, 180.0, 0.0)[0] is not b:
            leaks += 1  # physics.fly checks only where each step ends
    assert leaks > 20


@REAL
def test_a_knock_in_the_air_by_the_basins_end_lands_in_the_basin():
    # Harness tour seed 10: knocked mid jump-throw at x 887, 46 units up, she went through the corner of
    # 36's right end (951) under the foot of the wall above it, and fell out of the map for good.
    m = scenarios.real_map()
    f = body.fly(m, 887.45, 529.46, 180.0, -220.0, shoe=physics.SHOE)
    assert f.platform.id == 36 and 950 < f.land_x <= 951
    assert physics.fly(m, 887.45, 529.46, 180.0, -220.0)[0] is None


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
def test_the_planners_moves_fly_the_same_with_the_arrow_held(own_group, monkeypatch):
    """Every move the route planner finds with physics.fly lands on the same platform with the engine, the
    same walls stopping her (her foothold group's). The arcs differ only by the integration: the engine
    moves her by the mean of the step's two speeds, as the client does, physics.fly by the speed at its
    start, which lifts her about a unit higher at the top (78.1 for 77.0). With every wall stopping her
    (OWN_GROUP_WALLS off, as before), where one stops a walk-off the arrow held pushes her on at 8.9
    units/s below its foot, as the client's air steering does: off platform 3 she clears the foot of
    platform 1's wall and lands on 17, not in the basin (36). In game she went through that wall."""
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
        _, old_x, _ = physics.fly(m, *start, skip, group=a.group)
        if f.platform is None or f.platform.id != mv.dst:
            other.append((mv.kind, mv.src, mv.dst, f.platform and f.platform.id))
        elif abs(f.land_x - old_x) > (1.1 if mv.kind == "jump" else 0.5):
            far.append((mv.kind, mv.src, mv.dst))
    if own_group:
        assert other == [] and far == []
    else:
        assert other == [("walk_off", 3, 36, 17)]
        assert sorted(far) == [("walk_off", 13, 20), ("walk_off", 19, 24), ("walk_off", 22, 34)]  # the wall push


@REAL
def test_off_platform_3s_right_end_she_goes_through_platform_1s_wall_onto_17():
    # Three times out of three in game (telemetry, 2026-10-20), landing at 1,005-1,015 by her dot.
    m = scenarios.real_map()
    p3 = m.platform(3)
    assert p3.group != m.platform(1).group
    f = body.fly(m, 849.5, -926, 149.0, 0.0, shoe=physics.SHOE, arrow=1, group=p3.group)
    assert f.platform.id == 17 and f.land_x == pytest.approx(1008, abs=2)
    assert physics.fly(m, 849.5, -926, 149.0, 0.0, group=p3.group)[0].id == 17
    assert physics.fly(m, 849.5, -926, 149.0, 0.0)[0].id == 36  # every wall: down it into the basin
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
        assert f.land_x == pytest.approx(lx, abs=1.1)
