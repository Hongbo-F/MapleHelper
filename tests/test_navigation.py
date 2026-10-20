"""Map model, localization and route planning on a small made-up map (no game data needed)."""

import pytest


from gamelab.localization.pose import Localizer
from gamelab.motion.physics import JUMP_HEIGHT, air_time
from gamelab.motion.route import Planner
from gamelab.localization import mapdata

#   y 300  [A: 0..200]                 [D: 600..800]   (D is up a rope from C)
#   y 360           [B: 220..400]
#   y 500  [C: 0 .......................... 800] with walls at both ends
#   rope at x 700 from D (y 300) down to y 450, grabbable from C (50 above its floor)


def seg(layer, group, fid, x1, y1, x2, y2, prev=0, next=0):
    return str(fid), {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "prev": prev, "next": next}


def tree():
    footholds = {"1": {
        "1": dict([seg(1, 1, 1, 0, 300, 100, 300, 0, 2), seg(1, 1, 2, 100, 300, 200, 300, 1, 0)]),
        "2": dict([seg(1, 2, 3, 220, 360, 400, 360)]),
        "3": dict([seg(1, 3, 4, 0, 400, 0, 500, 0, 5), seg(1, 3, 5, 0, 500, 800, 500, 4, 6),
                   seg(1, 3, 6, 800, 500, 800, 400, 5, 0)]),
        "4": dict([seg(1, 4, 7, 600, 300, 800, 300)]),
    }}
    return {
        "info": {"VRLeft": -100, "VRTop": 0, "VRRight": 900, "VRBottom": 600},
        "foothold": footholds,
        "ladderRope": {"0": {"l": 0, "uf": 1, "x": 700, "y1": 300, "y2": 450}},
        "portal": {"0": {"pn": "out", "pt": 2, "x": 50, "y": 500, "tm": 123}},
        "life": {"0": {"type": "m", "id": "0000002", "x": 300, "y": 360, "cy": 360, "rx0": 230, "rx1": 390}},
        "miniMap": {"width": 1000, "height": 600, "centerX": 100, "centerY": 0, "mag": 4},
    }


@pytest.fixture
def model():
    return mapdata.parse(1, tree(), "Test Map", {})


def by_y(model, y, x):
    return next(p for p in model.platforms if p.covers(x) and abs(p.y_at(x) - y) < 1)


def test_footholds_join_into_platforms_and_walls(model):
    a = by_y(model, 300, 100)
    assert (a.left, a.right) == (0, 200)  # two footholds joined
    assert len(model.platforms) == 4
    assert len(model.walls) == 2


def test_platforms_and_walls_keep_their_foothold_group(model):
    a, c = by_y(model, 300, 100), by_y(model, 500, 400)
    assert a.group == (1, 1) and c.group == (1, 3)
    assert {(w.x1, w.group) for w in model.walls} == {(0, (1, 3)), (800, (1, 3))}
    assert model.walls_of(c.group) == model.walls and model.walls_of(a.group) == [] and model.walls_of(None) == []
    # A step joined by a wall is one group with two platforms (like platforms 2 and 4 on the real map).
    t = tree()
    t["info"]["VRRight"] = 1400
    t["foothold"]["2"] = {"7": dict([seg(2, 7, 20, 1000, 300, 1100, 300, 0, 21), seg(2, 7, 21, 1100, 300, 1100, 240, 20, 22),
                                      seg(2, 7, 22, 1100, 240, 1200, 240, 21, 0)])}
    m = mapdata.parse(1, t, "Test Map", {})
    low, high = by_y(m, 300, 1050), by_y(m, 240, 1150)
    assert low is not high and low.group == high.group == (2, 7)
    assert [(w.x1, w.y1, w.y2) for w in m.walls_of((2, 7))] == [(1100, 300, 240)]


def test_a_floor_that_forbids_dropping_through_it_gets_no_drop(model):
    # forbidFallDown (on 14,197 footholds in 51 of the game data's maps, none on this one): the game won't
    # drop her through that foothold, so no drop through it is planned.
    assert not any(p.no_drop for p in model.platforms)
    a = by_y(model, 300, 100)
    assert any(x.kind == "drop" and x.src == a.id for x in Planner(model).moves)
    t = tree()
    for f in t["foothold"]["1"]["1"].values():  # A, both its footholds
        f["forbidFallDown"] = 1
    m = mapdata.parse(1, t, "Test Map", {})
    a, c = by_y(m, 300, 100), by_y(m, 500, 400)
    assert a.no_drop and not a.drop_ok(100) and not c.no_drop and c.drop_ok(400)
    moves = Planner(m).moves
    assert not any(x.kind == "drop" and x.src == a.id for x in moves)
    assert any(x.kind == "walk_off" and x.src == a.id for x in moves)  # she still gets down off its ends
    # One foothold of the two: only along that one (3 platforms of the 2,100 on those maps are like this).
    t = tree()
    t["foothold"]["1"]["1"]["2"]["forbidFallDown"] = 1  # A's 100..200
    a = by_y(mapdata.parse(1, t, "Test Map", {}), 300, 100)
    assert not a.no_drop and a.drop_ok(50) and a.drop_ok(100) and not a.drop_ok(150) and not a.drop_ok(250)
    # ... and not near it: where she stands to drop is good to a few units.
    assert a.drop_ok(92, room=8) and not a.drop_ok(93, room=8) and not a.drop_ok(100, room=8)
    # The drop onto C goes from the nearest spot to the middle of their overlap (100) clear of it.
    [drop] = [x for x in Planner(mapdata.parse(1, t, "Test Map", {})).moves if x.kind == "drop" and x.src == a.id]
    assert drop.x == 92


def test_platform_lookups(model):
    c = by_y(model, 500, 400)
    assert model.platform_under(400, 490).id == c.id
    assert model.platform_below(300, 200).y_at(300) == 360  # B is first under A's height there
    assert model.spawn_platforms == {by_y(model, 360, 300).id: 1}


def test_physics_helpers():
    assert JUMP_HEIGHT == pytest.approx(77, abs=1)
    assert air_time(0) == pytest.approx(0.555, abs=0.01)
    assert air_time(100) is None  # higher than a jump reaches


def test_routes_use_the_rope_and_jumps(model):
    planner = Planner(model)
    a, b, c, d = (by_y(model, 300, 100), by_y(model, 360, 300), by_y(model, 500, 400), by_y(model, 300, 700))
    up = planner.plan(c.id, 100, d.id)
    assert [m.kind for m in up] == ["climb"] and up[0].rope_x == 700
    down = planner.plan(a.id, 100, c.id)
    assert down and down[-1].dst == c.id
    # A and B are 140-200 above C: out of jump reach, and no rope leads there.
    assert planner.reachable(c.id) == {c.id, d.id}
    # B is lower than A (360 vs 300) and just past A's right end: a running jump or walking off.
    assert planner.plan(a.id, 150, b.id)[0].kind in ("jump", "walk_off")


def test_localizer_round_trip(model):
    loc = Localizer(model, origin=(10.0, 20.0))
    assert loc.per_px == pytest.approx(16 / 1.40625)
    dot = loc.to_dot(400, 500)
    assert loc.to_map(dot) == pytest.approx((400, 500))
    where = loc.where_dot(loc.to_dot(400, 480))  # dot a little above the floor still finds C
    assert where.platform.y_at(400) == 500 and where.y == 500
    loc.anchor(dot, 410, 500)
    assert loc.to_map(dot) == pytest.approx((410, 500))


def test_camera_follows_her_but_stops_at_the_map_edges():
    from gamelab.localization.pose import Camera, VIEW
    cam = Camera((0, 0, 3000, 2000), offset=(680, 490))
    assert cam.to_screen(1500, 1000, at=(1500, 1000)) == pytest.approx((680 * 1.40625, 490 * 1.40625))
    # Near the left edge the camera stops at 0, so she's drawn left of her usual spot.
    assert cam.to_screen(100, 1000, at=(100, 1000))[0] == pytest.approx(100 * 1.40625)
    assert cam.clamped(100, 1000) == (True, False)
    assert cam.corner(2990, 1990) == pytest.approx((3000 - VIEW[0], 2000 - VIEW[1]))


def test_camera_learns_her_spot_only_where_it_isnt_stopped():
    from gamelab.localization.pose import Camera
    cam = Camera((0, 0, 3000, 2000), offset=(680, 490))
    cam.learn((1500, 1000), (700 * 1.40625, 500 * 1.40625), rate=1.0)
    assert cam.offset == pytest.approx((700, 500))
    cam.learn((100, 1000), (50, 600 * 1.40625), rate=1.0)  # x stopped at the edge: only y learns
    assert cam.offset == pytest.approx((700, 600))


def test_run_timing_and_rope_grab_lead():
    from gamelab.control.mover import grab_lead
    from gamelab.motion.physics import RUN_LAG_MS, run_ms
    from gamelab.motion.physics import WALK_SPEED
    assert RUN_LAG_MS == 46  # she's 149 (t - 0.113) along t s in, and lifts off 67 ms after the key
    assert run_ms(30) == pytest.approx(247, abs=1)  # jump key 247 ms in: lifts off 30 units along
    assert run_ms(149) == pytest.approx(1000 + RUN_LAG_MS, abs=10)
    # 7 units along is 93 ms in, in the kick's blind spot (she'd leave just below full speed): a little
    # later instead, at full speed, 11 units along.
    assert run_ms(7) == 118 and run_ms(10) == 118 and run_ms(12) == 127
    assert grab_lead(41) == pytest.approx(WALK_SPEED * 0.2775, abs=1)  # top of the jump over the rope
    assert grab_lead(120) is None


def test_a_jump_from_standing_is_keyed_where_it_surely_leaves_at_full_speed(monkeypatch):
    from gamelab.motion import body, physics
    from gamelab.motion.physics import LAGS, SHOE, sure_jump_ms
    # Up to 118 ms after the arrow the jump is put off to then (the kick's blind spot ends there); later is
    # as it comes. Every gap it gives leaves short at most 5% of the time.
    assert [sure_jump_ms(g) for g in (0, 40, 80, 117.2, 118, 130, 200)] == [118, 118, 118, 118, 118, 130, 200]
    assert all(body.short_takeoff_chance(sure_jump_ms(g), SHOE, LAGS) <= 0.05 for g in range(0, 300, 7))
    # With the kick trusted (probe P1), 35-52 ms after the arrow will do too: full speed from where she stands.
    monkeypatch.setattr(physics, "KICK_JUMPS", True)
    assert [sure_jump_ms(g) for g in (0, 40, 52, 53, 80)] == [34, 40, 52, 118, 118]


def test_a_route_jump_lands_even_if_she_leaves_15_percent_slow(model, monkeypatch):
    # (as a jump the mover keys from standing can, rarely: physics.SLOW_TAKEOFF)
    from gamelab.motion import physics
    from gamelab.motion.physics import JUMP_SPEED, WALK_SPEED, fly, sure_landing
    b = by_y(model, 360, 300)
    lands = lambda x, k=1.0: fly(model, x, 300, k * WALK_SPEED, -JUMP_SPEED)[0] is b
    # Takeoffs from A onto B that land 12 units early or late and 10% slow, but not 15% slow.
    edge = [x for x in range(100, 200) if lands(x - 12) and lands(x + 12) and lands(x, 0.9) and not lands(x, 0.85)]
    assert edge
    assert not sure_landing(model, edge[0], 300, WALK_SPEED, -JUMP_SPEED, b)
    monkeypatch.setattr(physics, "SLOW_TAKEOFF", 0.9)
    assert sure_landing(model, edge[0], 300, WALK_SPEED, -JUMP_SPEED, b)


def test_a_platform_just_above_monsters_is_not_safe():
    t = tree()
    # A small platform 40 units above B (where the snail spawns), and C/D far from any spawn.
    t["foothold"]["1"]["5"] = dict([seg(1, 5, 8, 250, 320, 350, 320)])
    model = mapdata.parse(1, t, "Test Map", {})
    low = by_y(model, 320, 300)
    assert not model.safe(low.id)                      # B's monsters reach 40 units up
    assert model.safe(by_y(model, 500, 400).id)        # C: no spawns, nothing just underneath
    assert not model.safe(by_y(model, 360, 300).id)    # B itself: the snail spawns there


def test_route_weights_let_near_equal_routes_take_turns(model):
    planner = Planner(model)
    a, c = by_y(model, 300, 100), by_y(model, 500, 400)
    assert planner.travel(a.id, 100, c.id) > 0
    # Two ways down from A: walking off its end or dropping through it. Weighting one of them up
    # makes the other win.
    first = planner.plan(a.id, 100, c.id)[0]
    other = planner.plan(a.id, 100, c.id, weight=lambda m: 3.0 if m.kind == first.kind else 1.0)
    assert other[0].kind != first.kind


def test_route_times_from_one_search_are_each_platforms_own_search(model):
    """Planner.times searches to every platform at once: the same seconds as a search for each, from x
    rounded as a search starts; kept, and the oldest start goes past TIMES_KEPT."""
    from gamelab.motion import route
    planner = Planner(model)
    for p in model.platforms:
        for x in (p.left, p.left + 10.4, (p.left + p.right) / 2, p.right - 0.6, p.right):
            times = planner.times(p.id, x)
            for q in model.platforms:
                found = planner._search(p.id, x, q.id, None, None)
                assert planner.travel(p.id, x, q.id) == (None if found is None else found[1])
                assert times.get(q.id) == (None if found is None else found[1])
    assert planner.times(0, 10.4) is planner.times(0, 9.6)  # both start from x 10
    planner._times.clear()
    for x in range(route.TIMES_KEPT + 1):
        planner.times(0, x)
    assert len(planner._times) == route.TIMES_KEPT and (0, 0) not in planner._times


@pytest.mark.skipif(not __import__("gamelab.sim.scenarios", fromlist=["available"]).available(),
                    reason="needs the map's game data (gamedata/data)")
def test_route_times_on_the_real_map_are_each_platforms_own_search():
    import random
    from gamelab.sim import scenarios
    model = scenarios.real_map()
    planner, rng = Planner(model), random.Random(1)
    for _ in range(60):
        p = rng.choice(model.platforms)
        x = rng.uniform(p.left, p.right)
        for q in model.platforms:
            found = planner._search(p.id, x, q.id, None, None)
            assert planner.travel(p.id, x, q.id) == (None if found is None else found[1])


def flights(model, rng, n):
    """Random flights on a map: from on, above and beside its platforms (at their ends and at walls too), walking,
    running, jumping and falling (past the fall cap too), through a platform or not, with the arrow held her way,
    the other way or none, stopped by every wall or by a foothold group's, with the planner's step and others; and
    a fifth through a platform's corner (just above its surface by an end, going out and down)."""
    from gamelab.motion import body
    walls = sorted({w.x1 for w in model.walls if w.x1 == w.x2}) or [0.0]
    out = []
    for _ in range(n):
        p = rng.choice(model.platforms)
        x = rng.choice([rng.uniform(p.left - 80, p.right + 80), float(round(rng.uniform(p.left, p.right))),
                        int(rng.uniform(p.left, p.right)), p.left, p.right, float(p.right), rng.choice(walls)])
        y = p.y_at(min(max(x, p.left), p.right)) - rng.choice([0, 1, 0.5, 0.004, rng.uniform(0, 400)])
        vx = rng.choice([0.0, -0.0, 0, 149.0, -149.0, 149, 134.1, 8.9, 5, rng.uniform(-400, 400)])
        vy = rng.choice([0.0, -555.0, 670.0, 900.0, rng.uniform(-700, 700)])
        group = rng.choice([body.EVERY_WALL, p.group, p.group, body.ROPE_GROUP, rng.choice(model.platforms).group])
        if rng.random() < 0.2:
            way = rng.choice([-1, 1])
            x = (p.right if way > 0 else p.left) - way * rng.uniform(0, 4)
            y, vx, vy = p.y_at(x) - rng.uniform(0, 3), way * rng.uniform(100, 700), rng.uniform(0, 700)
        dt, limit = rng.choice([(body.TICK, 4.0), (body.TICK, 4.0), (0.004, 4.0), (0.002, 1.0), (0.01, 0.5),
                                (body.TICK, 0)])
        out.append((x, y, vx, vy, rng.choice([None, None, p, rng.choice(model.platforms)]), dt, limit, group,
                    rng.choice([None, None, -1, 0, 1])))
    return out


def same_flights(model, cases, monkeypatch):
    """physics.flight (body.fly's flight) with the C flight and without it: the same platform, x, t and speed at
    landing, to the bit (a zero's sign too) and the type, with each wall rule (body.OWN_GROUP_WALLS)."""
    from collections import OrderedDict

    from gamelab import native
    from gamelab.motion import body, physics
    for own in (True, False):
        monkeypatch.setattr(body, "OWN_GROUP_WALLS", own)
        got = []
        for use in (True, False):
            monkeypatch.setattr(native, "USE", use)
            monkeypatch.setattr(physics, "_FLY_CACHE", OrderedDict())
            monkeypatch.setattr(physics, "_MODELS", OrderedDict())
            got.append([physics.flight(model, *c) for c in cases])
        for c, a, b in zip(cases, *got):
            assert a[0] is b[0] and [(type(v), repr(v)) for v in a[1:]] == [(type(v), repr(v)) for v in b[1:]], c


def test_her_flight_in_c_is_pythons(model, monkeypatch):
    import random
    from gamelab import native
    if native.lib() is None:
        pytest.skip(f"gamelab's C pieces aren't here: {native.why}")
    same_flights(model, flights(model, random.Random(2), 1500), monkeypatch)


@pytest.mark.skipif(not __import__("gamelab.sim.scenarios", fromlist=["available"]).available(),
                    reason="needs the map's game data (gamedata/data)")
def test_her_flight_in_c_is_pythons_on_the_real_map(monkeypatch):
    import random
    from gamelab import native
    from gamelab.sim import scenarios
    if native.lib() is None:
        pytest.skip(f"gamelab's C pieces aren't here: {native.why}")
    same_flights(scenarios.real_map(), flights(scenarios.real_map(), random.Random(3), 1500), monkeypatch)


def test_jumps_need_a_sure_landing(model):
    from gamelab.motion.physics import JUMP_SPEED, WALK_SPEED, fly, sure_landing
    a = by_y(model, 300, 100)
    land, _, _ = fly(model, 167, 300, WALK_SPEED, -JUMP_SPEED)
    assert land is not None and sure_landing(model, 167, 300, WALK_SPEED, -JUMP_SPEED, land)
    assert not sure_landing(model, 167, 300, WALK_SPEED, -JUMP_SPEED, a)  # it doesn't land back on A


def test_a_landing_is_worked_out_from_the_rounded_start_so_the_cache_never_changes_it(model, monkeypatch):
    from collections import OrderedDict
    from gamelab.motion import physics
    from gamelab.motion.physics import JUMP_SPEED, WALK_SPEED, fly
    monkeypatch.setattr(physics, "_FLY_CACHE", OrderedDict())
    monkeypatch.setattr(physics, "FLY_CACHE_SIZE", 50)
    # Two starts that round the same get the same landing, whichever came first: the one from the start rounded.
    early = fly(model, 166.996, 300, WALK_SPEED, -JUMP_SPEED)
    physics._FLY_CACHE.clear()
    late = fly(model, 167.004, 300, WALK_SPEED + 0.04, -JUMP_SPEED)
    physics._FLY_CACHE.clear()
    assert early == late == fly(model, 167, 300, WALK_SPEED, -JUMP_SPEED)
    # The cache keeps the last 50; a landing it let go is worked out again, the same.
    first = [fly(model, x, 300, WALK_SPEED, -JUMP_SPEED) for x in range(0, 200, 2)]
    assert len(physics._FLY_CACHE) == 50
    assert [fly(model, x, 300, WALK_SPEED, -JUMP_SPEED) for x in range(0, 200, 2)] == first


def test_a_map_never_gets_the_walls_or_landings_of_one_let_go_before_it(monkeypatch):
    # Keyed by id(model) alone, a freed model's id went to the next one made, with its walls and landings.
    from collections import OrderedDict
    from gamelab.motion import physics
    from gamelab.motion.physics import JUMP_SPEED, WALK_SPEED, fly
    monkeypatch.setattr(physics, "_MODELS", OrderedDict())

    def lands_on_its_own(i: int) -> None:  # a new map, with B 5 units lower than on map i - 1
        t = tree()
        for f in t["foothold"]["1"]["2"].values():
            f["y1"] = f["y2"] = 360 + 5 * i
        m = mapdata.parse(1, t, "Test Map", {})
        land, x, _ = fly(m, 167, 300, WALK_SPEED, -JUMP_SPEED)
        assert any(land is p for p in m.platforms) and land.y_at(x) == 360 + 5 * i

    for i in range(3 * physics.MODELS_KEPT):
        lands_on_its_own(i)
        assert len(physics._MODELS) <= physics.MODELS_KEPT
    monkeypatch.setattr(physics, "id", lambda o: 1, raising=False)  # each map takes the id of the one before
    for i in range(3 * physics.MODELS_KEPT, 4 * physics.MODELS_KEPT):
        lands_on_its_own(i)


def test_a_tour_sweeps_nearest_first_then_tidies_the_order():
    import random
    from types import SimpleNamespace
    from gamelab.behavior.tour import Tour
    xs = {1: 100.0, 2: 900.0, 3: 200.0, 4: 800.0}  # grounds along a line; travel = distance
    planner = SimpleNamespace(travel=lambda src, x, dst: abs(xs[dst] - x) / 100)
    tour = Tour.__new__(Tour)
    tour.nav, tour.rng = SimpleNamespace(planner=planner), random.Random(0)
    tour._spot_at = lambda pid: (pid, xs[pid])
    order, travel = tour.sweep([1, 2, 3, 4], 4, (0, 0.0))
    assert order == [1, 3, 4, 2] and travel == pytest.approx(9.0)


def test_she_walks_off_a_cliff_but_not_through_a_wall():
    # A ledge at y 300 ends in a cliff down to y 450 (one foothold chain, like platforms 34 and 36).
    # Higher up at the same x, a wall rises at the end of a shelf at y 150.
    t = tree()
    t["info"]["VRRight"] = 1400
    t["foothold"]["1"]["5"] = dict([seg(1, 5, 8, 1000, 300, 1100, 300, 0, 9), seg(1, 5, 9, 1100, 300, 1100, 450, 8, 10),
                                    seg(1, 5, 10, 1100, 450, 1300, 450, 9, 0)])
    t["foothold"]["1"]["6"] = dict([seg(1, 6, 11, 1020, 150, 1100, 150, 0, 12), seg(1, 6, 12, 1100, 150, 1100, 100, 11, 0)])
    m = mapdata.parse(1, t, "Test Map", {})
    ledge, low, shelf = by_y(m, 300, 1050), by_y(m, 450, 1200), by_y(m, 150, 1050)
    moves = Planner(m).moves
    assert any(x.kind == "walk_off" and x.src == ledge.id and x.dst == low.id and x.direction == "right" for x in moves)
    assert not any(x.kind == "walk_off" and x.src == shelf.id and x.direction == "right" for x in moves)


def test_down_to_a_platform_a_walk_off_reaches_she_walks_off(model):
    a, b = by_y(model, 300, 100), by_y(model, 360, 300)
    kinds = {m.kind for m in Planner(model).moves if m.src == a.id and m.dst == b.id and m.direction == "right"}
    assert "walk_off" in kinds and "jump" not in kinds


def test_a_walk_off_must_land_where_she_can_stop():
    # Off A's right end at full speed she comes down on B 37 units on, then slides 13.9 more: on a B 30 units
    # long she'd slide off its far end (as walking off 10's right end onto 15 on the real map did, every time
    # in the simulator), so that's no move; on B as it is (180 long), it is.
    t = tree()
    t["foothold"]["1"]["2"] = dict([seg(1, 2, 3, 220, 360, 250, 360)])
    m = mapdata.parse(1, t, "Test Map", {})
    a, b = by_y(m, 300, 100), by_y(m, 360, 230)
    assert not any(x.kind == "walk_off" and x.src == a.id and x.dst == b.id for x in Planner(m).moves)
    m = mapdata.parse(1, tree(), "Test Map", {})
    a, b = by_y(m, 300, 100), by_y(m, 360, 230)
    assert any(x.kind == "walk_off" and x.src == a.id and x.dst == b.id for x in Planner(m).moves)


def _stop(rounds, seen, loot_after=(), elsewhere=(), log=None):
    """A tour stop with faked fights: `rounds` gives each fight's throws, `seen` how many monsters a
    whole-screen look then sees on the platform, `loot_after` the rounds after which drops were
    collected elsewhere. Returns how many fights ran."""
    from types import SimpleNamespace
    from gamelab.attack.spots import Spot
    from gamelab.behavior.fight import Report
    from gamelab.behavior.tour import Tour, TourReport
    from gamelab.localization.monsters import Monster
    from gamelab.perception.mobs import Mob
    tour, fights = Tour.__new__(Tour), []
    tour.log, tour.report = (lambda m: None), TourReport()
    tour.spot = lambda pid, on_it=False: Spot(pid, 100, 1.0, False)
    tour.nav = SimpleNamespace(go=lambda *a, **k: SimpleNamespace(result="arrived", kills=0))
    pig = Monster(Mob("pig", 0, 0, 30, 30, 0.9), 300, 500, 7)

    def clear(*a, **k):
        fights.append(1)
        return Report(throws=rounds[len(fights) - 1], stop="nothing left in reach")

    def collect():
        if len(fights) in loot_after:
            tour.report.loot_trips += 1

    tour.h = SimpleNamespace(report=Report(), clear=clear,
                             look=lambda **k: SimpleNamespace(monsters=[pig] * seen[len(fights) - 1]))
    tour.collect_loot = collect
    tour._record = lambda pid, r: None
    moves = iter(elsewhere)
    tour.spot_for = lambda pid, left, spot: next(moves, None)
    if log is not None:
        tour.log = log.append
    tour.hunt(7, kills=5, seconds=60)
    return len(fights)


def test_a_stop_moves_on_once_nothing_is_left_there():
    assert _stop([3, 3, 3], seen=[0, 0, 0]) == 1                  # cleared: no going back to wait
    assert _stop([2, 2, 2], seen=[0, 0, 0], loot_after=[1]) == 1  # looted elsewhere: on to the next


def test_a_stop_goes_back_only_for_monsters_it_can_get_at():
    assert _stop([2, 2, 0, 0], seen=[1, 1, 1, 1]) == 3  # back while throwing; a fight with none moves on
    assert _stop([0, 0, 0], seen=[2, 2, 2]) == 1        # seen from afar, never in reach: no looping


def test_grounds_are_ranked_by_exp_and_learn_from_visits(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from gamelab import paths
    from gamelab.behavior.fight import Report
    from gamelab.behavior.tour import Tour
    monkeypatch.setattr(paths, "GROUNDS", tmp_path / "grounds.json")
    spawns = {1: ["Snail", "Snail", "Snail"], 2: ["Pig", "Pig"], 3: ["Slime", "Snail"], 4: ["Pig", "Orange Mushroom"]}
    model = SimpleNamespace(id=5, mob_exp={"Snail": 2, "Pig": 13, "Slime": 12, "Orange Mushroom": 15},
                            monsters_by_platform=spawns, platforms=[], safe=lambda pid: False)
    hunter = SimpleNamespace(senses=SimpleNamespace(eyes=SimpleNamespace(
        kinds={"snail", "pig", "slime", "orange-mushroom"})))
    tour = Tour(hunter, SimpleNamespace(model=model), log=lambda m: None)
    assert tour.worth == {"pig", "slime", "orange-mushroom"}  # snails: 2 EXP against 15
    assert sorted(tour.hunting_grounds()) == [2, 3, 4]         # a snail-only platform isn't one
    # Visits to 3 keep finding nothing worth it (only its snails turn up): it drops out.
    for _ in range(6):
        tour._record(3, Report())
    assert sorted(tour.hunting_grounds()) == [2, 4]
    # ... and that's remembered for the next tour.
    again = Tour(hunter, SimpleNamespace(model=model), log=lambda m: None)
    assert sorted(again.hunting_grounds()) == [2, 4] and again.stats["3"]["visits"] == 6
    # A visit's kills count by their EXP.
    tour._record(4, Report(kills={"pig": 2, "orange-mushroom": 1}, exp_kills=3))
    assert tour.stats["4"]["exp"] == 41
    # The hunt loop's visits also say how long they lasted (its rate rule reads the average).
    tour.record_visit(2, 26.0, 40.0)
    tour.record_visit(2, 13.0, 20.0)
    assert {k: tour.stats["2"][k] for k in ("visits", "exp", "timed", "seconds")} == \
        {"visits": 2, "exp": 39.0, "timed": 2, "seconds": 60.0}
    assert Tour(hunter, SimpleNamespace(model=model), log=lambda m: None).stats["2"]["seconds"] == 60.0


def test_monsters_left_out_of_reach_are_shot_from_another_spot_before_looting():
    from gamelab.attack.spots import Spot
    log = []
    # After the first fight 2 are left out of reach; a spot on platform 9 reaches them: she goes there
    # and fights again, and only then collects the drops.
    assert _stop([2, 2, 0], seen=[2, 0, 0], elsewhere=[Spot(9, 300, 0.0, True)], log=log) == 2
    assert any("over to platform 9" in m for m in log)
    assert log.index(next(m for m in log if "over to platform 9" in m)) < len(log) - 1


def test_tour_seed_seeds_the_route_choices_too(monkeypatch):
    """`tour --seed N` seeds the navigator (route weights, takeoff spots) as well as the sweep's order, as
    the simulator does: the same seed gives the same tour."""
    import random
    from types import SimpleNamespace

    from gamelab import __main__ as cli
    from gamelab.behavior import tour
    rngs = {}

    class Tour:
        def __init__(self, hunter, nav, log, rng, rest_below, rest_avoid=False):
            rngs["tour"] = rng

        def run(self, **kw):
            return "done"

    hunter = SimpleNamespace(model=SimpleNamespace(id=1),
                             navigator=lambda fighting, rng=None: rngs.setdefault("nav", rng))
    monkeypatch.setattr(cli, "_hunter", lambda args, **kw: hunter)
    monkeypatch.setattr(cli, "_record", lambda command, map_id: None)
    monkeypatch.setattr(cli, "_game_title", lambda: "game")
    monkeypatch.setattr(cli.window, "bring_to_front", lambda title: None)
    monkeypatch.setattr(tour, "Tour", Tour)
    assert cli.cmd_tour(cli.build_parser().parse_args(["tour", "--seed", "7"])) == 0
    want = random.Random(7).random()
    assert rngs["nav"].random() == want and rngs["tour"].random() == want


def test_a_trip_mode_needs_an_escort_unless_it_avoids_monsters(model):
    """go(mode=...): avoid (no escort), hunt (the escort's bursts and monsters in the way) or transit (the
    trip to rest); hunting or transit without an escort, or an unknown mode, is refused before any key."""
    from types import SimpleNamespace

    from gamelab.control.navigator import AVOID, HUNT, TRANSIT, Navigator
    key = lambda name: SimpleNamespace(name=name)
    keys = SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), up=key("up"), down=key("down"),
                           pick_up=None)
    alone = Navigator(None, SimpleNamespace(keys=keys), model)
    escorted = Navigator(None, SimpleNamespace(keys=keys), model, escort=SimpleNamespace())
    assert (alone.mode, escorted.mode) == (AVOID, HUNT) and not alone.fight and escorted.fight
    for mode in (HUNT, TRANSIT, "sneak"):
        with pytest.raises(ValueError):
            alone.go(0, mode=mode)
    with pytest.raises(ValueError):
        escorted.go(0, mode="sneak")


def test_the_trips_to_rest_and_stand_by_keep_their_modes():
    """Tour._calm_mode: to rest, transit with an escort and transit-avoid without one (low HP never ends
    the trip either way: it did without an escort, and she sat down among the monsters); to stand by, the
    navigator's own mode (the tour's), or avoid; with rest_avoid, transit-avoid both ways."""
    from types import SimpleNamespace

    from gamelab.behavior.tour import Tour
    from gamelab.control.navigator import AVOID, TRANSIT, TRANSIT_AVOID
    tour = object.__new__(Tour)
    tour.rest_avoid = False
    tour.nav = SimpleNamespace(escort=SimpleNamespace())
    assert (tour._calm_mode(TRANSIT), tour._calm_mode(None)) == (TRANSIT, None)
    tour.nav = SimpleNamespace(escort=None)
    assert (tour._calm_mode(TRANSIT), tour._calm_mode(None)) == (TRANSIT_AVOID, AVOID)
    tour.rest_avoid = True
    assert tour._calm_mode(TRANSIT) == tour._calm_mode(None) == TRANSIT_AVOID


def test_transit_with_avoidance_keeps_clear_of_monsters_and_only_drinks_potions(model):
    """TRANSIT_AVOID (`tour --rest-avoid`): no escort needed, no fighting on the way, and low HP doesn't end
    the trip (potions only, as in transit)."""
    from types import SimpleNamespace

    from gamelab.control.navigator import TRANSIT_AVOID, Navigator
    key = lambda name: SimpleNamespace(name=name)
    keys = SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), up=key("up"), down=key("down"),
                           pick_up=None)
    health, potions = (lambda s: "HP low"), (lambda s: None)
    for escort in (None, SimpleNamespace()):
        nav = Navigator(None, SimpleNamespace(keys=keys), model, health=health, potions=potions, escort=escort)
        st = nav.begin(0, mode=TRANSIT_AVOID)
        assert st.mode == TRANSIT_AVOID and st.health is potions
        nav._mode = st.mode
        assert not nav.fight


def _navigator(model, escort=None):
    from types import SimpleNamespace

    from gamelab.control.navigator import Navigator
    key = lambda name: SimpleNamespace(name=name)
    keys = SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), up=key("up"), down=key("down"),
                           pick_up=None)
    return Navigator(None, SimpleNamespace(keys=keys, sent_at=0.0), model, escort=escort)


def test_a_chain_runs_on_past_its_end_only_as_far_as_the_next_move_and_the_monsters_allow(model):
    """Navigator._tail: a chain another may follow runs on long enough after she's back on the ground for a
    look placing her and the next chain's plan (CHECK_AFTER + CHECK_LEAD): standing, or running on with the
    arrow held if that leaves RUN_ON_ROOM before the next move; not at all when her last run along the
    platform already leaves time for it. With no room for it (short of the next move, no next move, or a
    monster in the way of the run on), None: she stops where the chain ends."""
    from gamelab.control.navigator import CHECK_AFTER, CHECK_LEAD, RUN_ON_ROOM
    from gamelab.localization.monsters import MonsterTrack
    from gamelab.motion.flow import Body, Leg, Span
    from gamelab.motion.physics import WALK_SPEED
    from gamelab.motion.route import Move
    a, c = by_y(model, 300, 100), by_y(model, 500, 400)
    nav = _navigator(model)
    need = CHECK_AFTER + CHECK_LEAD
    landing = Leg(Move("walk_off", a.id, c.id, 195, 230, "right"), Body(0.0, 150, a), Body(1.0, 240, c, run=1, landed=True),
                  [Span(a.id, 0.0, 0.3, 150, 195)])
    far, near = Move("walk_off", c.id, a.id, 700, 700, "right"), Move("walk_off", c.id, a.id, 300, 300, "right")
    assert nav._tail([landing], [landing.move, far]) == pytest.approx(need)  # room enough: the whole need
    assert 300 - 240 - RUN_ON_ROOM < need * WALK_SPEED
    assert nav._tail([landing], [landing.move, near]) is None  # the next move is too close to run on
    assert nav._tail([landing], [landing.move]) is None  # no next move: nothing to run on for
    standing = Leg(landing.move, landing.start, Body(1.0, 240, c, run=0, landed=True), landing.spans)
    assert nav._tail([standing], [standing.move]) == pytest.approx(need)
    pig = MonsterTrack(c.id, 240 + WALK_SPEED * need + 30, speed=0.0)
    assert nav._tail([landing], [landing.move, far], [pig]) is None  # it would run her into it
    running = Leg(Move("walk_off", c.id, a.id, 700, 700, "right"), Body(0.0, 240, c, run=1),
                  Body(1.0, 389, c, run=1), [Span(c.id, 0.0, 1.0, 240, 389)])
    assert nav._tail([running], [running.move, far]) == 0.0  # she's on the ground long before its end


def test_a_trip_starts_from_the_callers_look_while_it_still_shows_where_she_is(model, monkeypatch):
    """Navigator.begin(scene=...): the first pass uses the caller's last look (one look less before she sets
    off) if no key plan was sent since and it's fresh, with the monsters the mode looks for."""
    from types import SimpleNamespace

    import numpy as np

    from gamelab.control.navigator import AVOID, FRESH_S, HUNT
    from gamelab.localization.pose import Where
    from gamelab.localization.scene import Scene
    c = by_y(model, 500, 400)
    t = 100.0
    look = Scene(t=t, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0, where=Where(400, 500, c, (0, 0)),
                 searched=True)
    escorted = _navigator(model, escort=SimpleNamespace())
    clock = {"now": t + 0.1}
    import gamelab.control.navigator as nv
    monkeypatch.setattr(nv.time, "monotonic", lambda: clock["now"])
    assert escorted.begin(c.id, mode=HUNT, scene=look).fresh is look
    escorted.actions.sent_at = t + 0.05  # a plan went out after it
    assert escorted.begin(c.id, mode=HUNT, scene=look).fresh is None
    escorted.actions.sent_at, clock["now"] = 0.0, t + FRESH_S + 0.1  # too old
    assert escorted.begin(c.id, mode=HUNT, scene=look).fresh is None
    clock["now"] = t + 0.1
    alone = _navigator(model)
    assert alone.begin(c.id, mode=AVOID, scene=look).fresh is None  # avoiding: only a whole-screen look will do
    look.wide = True
    assert alone.begin(c.id, mode=AVOID, scene=look).fresh is look


def test_a_hit_cuts_the_running_chain_and_withdraws_the_one_queued_after_it(model, monkeypatch):
    """Navigator._watch: once the next chain is queued (after the look placing her), a hit cuts the running
    plan short and withdraws the queued one before it starts; the legs sent leave out the withdrawn chain's."""
    from types import SimpleNamespace

    import gamelab.control.navigator as nv
    from gamelab.localization.pose import Localizer
    from gamelab.motion.flow import Body, Leg, Span
    from gamelab.motion.route import Move
    c = by_y(model, 500, 400)
    nav = _navigator(model)
    localizer = Localizer(model)
    hp = {"now": 1.0}
    nav.senses = SimpleNamespace(glance=lambda: (localizer.to_dot(150, 500), hp["now"]), wait_frame=nv.time.sleep,
                                 localizer=localizer)
    nav.flow.expected = lambda legs, t: c.id
    walk = Leg(Move("walk_off", c.id, c.id, 300, 300, "right"), Body(0.0, 100, c, run=1), Body(0.5, 175, c, run=1),
               [Span(c.id, 0.0, 0.5, 100, 175)])
    more = Leg(walk.move, Body(0.0, 175, c, run=1), Body(0.5, 250, c, run=1), [Span(c.id, 0.0, 0.5, 175, 250)])
    calls = {"cancel": [], "wait": []}
    monkeypatch.setattr(nv.client, "done", lambda pid: False)
    monkeypatch.setattr(nv.client, "cancel", lambda pid: calls["cancel"].append(pid) or pid == "queued")
    monkeypatch.setattr(nv.client, "wait", lambda pid, timeout=120: calls["wait"].append(pid) or {})
    start = nv.time.monotonic()
    monkeypatch.setattr(nav, "_look_ahead", lambda st, running, dst: nv._Ahead(None))
    monkeypatch.setattr(nav, "_check_at", lambda running: start)

    def queue_next(st, running, ahead, dst, send):
        hp["now"] = 0.5  # hit right after the next chain was queued
        return nv._Running("queued", [more], running.ends, False)
    monkeypatch.setattr(nav, "_queue_next", queue_next)
    nav._sent_plans = ["running"]
    running = nv._Running("running", [walk], start, True, 0.2)
    cut, sent = nav._watch(running, 1.0, by_y(model, 300, 100).id, send=None, st=SimpleNamespace())
    assert cut.startswith("hit") and sent == [walk]
    assert calls["cancel"] == ["running", "queued"] and calls["wait"] == ["running"]  # the queued one withdrawn


def test_a_look_searches_for_ropes_where_the_maps_ropes_can_show(model, tmp_path, monkeypatch):
    """Ropes are looked for only around where her minimap dot puts the map's ropes on screen (ROPE_PAD):
    she's placed by the rope there as with the whole screen searched, and a rope-coloured post elsewhere
    isn't searched."""
    import numpy as np

    from gamelab import paths
    from gamelab.analysis import bench
    from gamelab.localization import senses
    rng = np.random.default_rng(0)
    tag = np.full((20, 72), 40, np.uint8)
    tag[5:15, 4:68] = np.where(rng.random((10, 64)) < 0.45, 235, 40)
    np.save(tmp_path / "tag.npy", tag)
    monkeypatch.setattr(paths, "TAG_FULL", tmp_path / "tag.npy")
    full = np.full((1080, 1920, 3), (90, 150, 230), np.uint8)  # sky
    full[590:610, 924:996] = tag[..., None]                  # her name tag, centre (960, 600)
    dot = Localizer(model).to_dot(500, 500)                  # she's on C at x 500 ...
    full[round(dot[1]) - 2:round(dot[1]) + 3, round(dot[0]) - 2:round(dot[0]) + 3] = (255, 230, 40)  # ... her dot
    for x in (300, 1243):  # the map's rope at 700 shows at about 1239; nothing is at 300 on the map
        full[200:900, x - 2:x + 2] = (200, 180, 140)
    bgra = np.dstack([full[..., ::-1], np.full((1080, 1920), 255, np.uint8)])

    def look():
        eyes = bench.SavedEyes(tag[::2, ::2])
        eyes.show(bgra)
        return senses.Senses(eyes, model, (100, 100)).look(mobs=False)

    near = look()
    monkeypatch.setattr(senses, "ROPE_PAD", None)
    whole = look()
    assert [r.x for r in near.ropes] == [1243] and [r.x for r in whole.ropes] == [300, 1243]
    assert near.how == whole.how == "rope" and near.where.x == whole.where.x == pytest.approx(700 - 283 / 1.40625)


def test_the_tracker_keeps_only_the_latest_speeds_of_each_kind():
    """Tracker.speeds is for `nav`'s summary: a look every frame would add one a monster each time, so
    only the latest SPEED_SAMPLES are kept."""
    from gamelab.localization import monsters
    from gamelab.localization.monsters import Monster, Tracker
    from gamelab.perception.mobs import Mob
    tracker, n, x = Tracker(), monsters.SPEED_SAMPLES, 300.0
    for i in range(n + 51):  # a pig pacing back and forth, a look every 0.5 s ...
        if i:  # ... 20 units the other way each time (40 units/s), and over the last 50 looks 30 (60 units/s)
            x += (20 if i <= n else 30) * (1 if i % 2 else -1)
        tracker.update(0.5 * i, [Monster(Mob("pig", 0, 0, 30, 30, 0.9), x, 500, 7)])
    v = tracker.speeds["pig"]
    assert len(v) == n and list(v)[-50:] == [60.0] * 50 and list(v)[:-50] == [40.0] * (n - 50)


def walker(x: float, facing: int = 0, pose: str = "move", kind: str = "pig", platform: int = 7):
    from gamelab.localization.monsters import Monster
    from gamelab.perception.mobs import Mob
    return Monster(Mob(kind, 0, 0, 30, 30, 0.9, facing=facing, pose=pose), x, 500, platform, None, facing, pose)


def test_a_monster_seen_walking_heads_the_way_it_faces_from_its_first_sighting(monkeypatch):
    from gamelab.localization import monsters
    from gamelab.localization.mapdata import Platform, Segment
    from gamelab.localization.monsters import TURN_SLACK, Tracker
    assert not monsters.FACING_HEADING  # off by default: keeping clear of monsters went worse in the simulator
    monkeypatch.setattr(monsters, "FACING_HEADING", True)
    p = Platform(7, [Segment(0, 500, 1000, 500)])
    right, standing, unknown = Tracker().update(10.0, [walker(400, 1), walker(600, 1, "stand"), walker(800)])
    assert (right.heading, right.vx, right.facing) == (1, None, 1)
    lo, hi = right.reach(1.0, p)
    assert lo == 400 - TURN_SLACK and hi == pytest.approx(400 + right.speed)  # it spreads forward
    assert standing.heading == unknown.heading == 0  # facing that way, but not walking; or not known
    assert standing.reach(1.0, p) == pytest.approx((600 - standing.speed, 600 + standing.speed))
    tracker = Tracker()
    tracker.update(10.0, [walker(400, 1)])
    again, = tracker.update(10.1, [walker(403, 1)])  # too soon to tell from the looks: still its facing
    assert (again.heading, again.vx) == (1, None)
    again, = tracker.update(10.5, [walker(395, 1)])  # seen walking back (a knock back): the looks win
    assert again.heading == -1 and again.vx < 0
    monkeypatch.setattr(monsters, "FACING_HEADING", False)  # as before: a heading takes two looks
    assert Tracker().update(10.0, [walker(400, 1)])[0].heading == 0


def test_two_standing_in_move_frames_are_matched_to_themselves_as_the_nearest():
    # In game the pose is the matched template's, wrong for about a fifth of the sightings: two pigs standing
    # 100 apart in move frames, facing each other, were each matched to the other by the way they faced
    # (WALK_MATCH), and said to walk toward each other at 125 units/s. By default the nearest is matched.
    from gamelab.localization import monsters
    from gamelab.localization.monsters import Tracker
    assert not monsters.WALK_MATCH
    pigs = [walker(0.0, 1), walker(100.0, -1)]
    tracker = Tracker()
    tracker.update(10.0, pigs)
    assert [(t.heading, t.vx) for t in tracker.update(10.8, pigs)] == [(0, 0.0), (0, 0.0)]
    monsters.WALK_MATCH = True
    try:
        tracker = Tracker()
        tracker.update(10.0, pigs)
        assert [t.vx for t in tracker.update(10.8, pigs)] == pytest.approx([-125.0, 125.0])  # swapped, as it was
    finally:
        monsters.WALK_MATCH = False


def test_two_of_a_kind_walking_apart_are_told_apart_by_which_way_they_face(monkeypatch):
    from gamelab.localization import monsters
    from gamelab.localization.monsters import Tracker
    monkeypatch.setattr(monsters, "WALK_MATCH", True)  # off by default (see above)
    # Two pigs 60 apart walk past each other: a second later each is nearer where the other was.
    a, b = walker(400, 1), walker(460, -1)
    tracker = Tracker()
    tracker.update(10.0, [a, b])
    a2, b2 = walker(470, 1), walker(390, -1)  # they've crossed
    ta, tb = tracker.update(11.0, [a2, b2])
    assert (ta.vx, tb.vx) == (70.0, -70.0)  # each matched to itself, not to the other
    # Not known to be walking: the nearest, as before (each taken for the other, as if they'd hardly moved).
    still_a, still_b = walker(400), walker(460)
    tracker = Tracker()
    tracker.update(10.0, [still_a, still_b])
    ta, tb = tracker.update(11.0, [walker(a2.x), walker(b2.x)])
    assert (ta.vx, tb.vx) == (10.0, -10.0)
    # Walking farther than MATCH_DX since the last look, it can still be the same one.
    tracker = Tracker()
    tracker.update(10.0, [a])
    far = walker(400 + 0.9 * tracker.last[1][0].speed, 1)
    assert tracker.update(11.0, [far])[0].vx > 0
    tracker = Tracker()
    tracker.update(10.0, [still_a])
    assert tracker.update(11.0, [walker(far.x)])[0].vx is None  # not known to be walking: a new one


def test_a_walker_that_stopped_is_still_matched_to_itself():
    from gamelab.localization.monsters import Tracker
    a, b = walker(400, 1), walker(480, 0, "stand")
    tracker = Tracker()
    tracker.update(10.0, [a, b])
    a2, b2 = walker(405, 1, "stand"), walker(480, 0, "stand")  # a stopped just after the look
    assert [t.vx for t in tracker.update(10.7, [a2, b2])] == [0.0, 0.0]
