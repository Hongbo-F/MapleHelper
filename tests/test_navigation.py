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


def twin_tree():
    """The made-up map with a twin: T (200..400, y 499.9) lies on C's floor, a chain of its own joined into C's
    (as 11 lies on 14 on West Rocky Mountain II), and E (150..450, y 430) is over both."""
    t = tree()
    t["foothold"]["1"]["3"]["10"] = {"x1": 200, "y1": 499.9, "x2": 400, "y2": 499.9, "prev": 0, "next": 5}
    t["foothold"]["1"]["5"] = dict([seg(1, 5, 11, 150, 430, 450, 430)])
    return t


def test_a_platform_lying_on_anothers_floor_is_that_platform_for_everything(model):
    assert model.twins == {} and model.places is model.platforms  # none on the made-up map (none on the first map)
    m = mapdata.parse(1, twin_tree(), "Test Map", {})
    c, e = by_y(m, 500, 600), by_y(m, 430, 300)
    twin = next(p for p in m.platforms if (p.left, p.right) == (200, 400))
    assert m.twins == {twin.id: c.id} and m.place_of(twin.id) == c.id and m.place_of(c.id) == c.id
    assert twin not in m.places and len(m.places) == len(m.platforms) - 1
    # Where she is: on C, also where T is (0.1 above it), from her dot or a fall.
    assert m.platform_under(300, 499).id == c.id and m.platform_below(300, 450).id == c.id
    assert m.safe(twin.id) == m.safe(c.id)
    # No move from or onto T; between C and T she walks: a trip to T is a trip to C.
    planner = Planner(m)
    assert not any(twin.id in (mv.src, mv.dst) for mv in planner.moves)
    assert planner.plan(c.id, 600, twin.id) == [] and planner.plan(twin.id, 300, c.id) == []
    assert twin.id in planner.reachable(e.id) and planner.travel(e.id, 300, twin.id) == planner.travel(e.id, 300, c.id)
    # A drop off E lands on C, and none goes through C under T: standing there she may be on T, and a drop takes
    # her through T alone.
    drops = [mv for mv in planner.moves if mv.kind == "drop"]
    assert any(mv.src == e.id and mv.dst == c.id for mv in drops)
    assert not any(mv.src == c.id and 200 - 8 <= mv.x <= 400 + 8 for mv in drops)
    from gamelab.motion.route import droppable
    assert not droppable(m, c, 300) and not droppable(m, c, 405, room=8) and droppable(m, c, 600)
    # A landing from E's jumps is on C, never T.
    assert not any(mv.dst == twin.id for mv in planner.moves)


def trap_tree(gap: float = 44):
    """A pit, as West Rocky Mountain II's stairs: S (-400..-200, y 480) over R (-300..-gap, y 540), each jumped onto from
    the other; the pit Q (0..300, y 600, a wall up from its right end) that R's jump goes down onto, and whose only way
    back up is a jump off its left end onto R, `gap` across and 60 up (44 is 27 -> 25's)."""
    return {
        "info": {"VRLeft": -500, "VRTop": 0, "VRRight": 500, "VRBottom": 800},
        "foothold": {"1": {
            "1": dict([seg(1, 1, 1, -400, 480, -200, 480)]),
            "2": dict([seg(1, 2, 2, -300, 540, -gap, 540)]),
            "3": dict([seg(1, 3, 3, 0, 600, 300, 600, 0, 4), seg(1, 3, 4, 300, 600, 300, 400, 3, 0)]),
        }},
        "miniMap": {"width": 1000, "height": 800, "centerX": 500, "centerY": 0, "mag": 4},
    }


def test_a_pit_whose_only_way_out_is_a_jump_from_its_very_end_gets_that_jump_by_its_chance():
    from gamelab.motion import route
    m = mapdata.parse(1, trap_tree(44), "Trap", {})
    s, r, q = by_y(m, 480, -300), by_y(m, 540, -100), by_y(m, 600, 100)
    planner = Planner(m)
    [out] = [mv for mv in planner.moves if mv.chance < 1]
    # Off Q's left end up onto R lands from its first 17 units at full speed (none by the sure rules: 9 units of room,
    # and still on R taking off 12 early or late): from the middle of them, 0.90 by her takeoff's spread.
    assert (out.kind, out.src, out.dst, out.direction) == ("jump", q.id, r.id, "left")
    assert out.takeoff == 9.0 and out.x == 29.0 and -60 < out.land_x < r.right
    assert out.chance == pytest.approx(0.898, abs=0.001)
    assert out.seconds == pytest.approx(route._running_jump_seconds(0.408) + (1 / out.chance - 1) * route.MISS_S,
                                        abs=0.02)
    assert all(planner.reachable(p.id) == {s.id, r.id, q.id} for p in (s, r, q))
    assert [mv.dst for mv in planner.plan(q.id, 200, s.id)] == [r.id, s.id]
    # 6 units further, it lands from 13 (0.72); 10 further, from 9 (0.53); 16 further, from 3: not worth trying
    # (route.ESCAPE_CHANCE), and the pit stays one.
    chances = {}
    for gap in (50, 54, 60):
        mm = mapdata.parse(1, trap_tree(gap), "Trap", {})
        chances[gap] = [round(mv.chance, 2) for mv in Planner(mm).moves if mv.chance < 1]
    assert chances == {50: [0.72], 54: [0.53], 60: []}


def step_tree(gap: float = 25, rise: float = 64):
    """A (0..300, y 600) and B, `rise` up and `gap` past A's right end (300 + gap..500), over a floor C (-500..1000, y
    800) with a rope up to each: West Rocky Mountain II's 17 and 16 (25 across, 64 up)."""
    return {
        "info": {"VRLeft": -500, "VRTop": 0, "VRRight": 1000, "VRBottom": 900},
        "foothold": {"1": {
            "1": dict([seg(1, 1, 1, 0, 600, 300, 600)]),
            "2": dict([seg(1, 2, 2, 300 + gap, 600 - rise, 500, 600 - rise)]),
            "3": dict([seg(1, 3, 3, -500, 800, 1000, 800)]),
        }},
        "ladderRope": {"0": {"l": 0, "uf": 1, "x": 100, "y1": 600, "y2": 760},
                       "1": {"l": 0, "uf": 1, "x": 450, "y1": 600 - rise, "y2": 760}},
        "miniMap": {"width": 1500, "height": 900, "centerX": 500, "centerY": 0, "mag": 4},
    }


def test_a_jump_off_an_end_that_lands_only_from_nearer_than_jump_from_is_found():
    # From 33 back (route.JUMP_FROM) the jump up onto B comes down short of it or with no room to spare; from 15-21
    # back it surely lands. Looked for at JUMP_FROM alone it was never found, and A got to B by nothing.
    from gamelab.motion import route
    m = mapdata.parse(1, step_tree(), "Step", {})
    a, b = by_y(m, 600, 100), by_y(m, 536, 400)
    assert route._sure_running_jump(m, a, a.right - route.JUMP_FROM, 1) is None
    [up] = [mv for mv in Planner(m).moves if mv.src == a.id and mv.dst == b.id]
    assert up.kind == "jump" and up.direction == "right" and up.chance == 1
    assert route.TAKEOFF_EDGE <= a.right - up.takeoff < route.JUMP_FROM and up.x == up.takeoff - route.RUN_UP
    assert b.left + route.LANDING_ROOM <= up.land_x
    # Nearer than TAKEOFF_EDGE alone (36 across): still none.
    m = mapdata.parse(1, step_tree(36), "Step", {})
    a, b = by_y(m, 600, 100), by_y(m, 536, 400)
    assert not any(mv.src == a.id and mv.dst == b.id for mv in Planner(m).moves)


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


def test_her_dot_snaps_onto_the_nearer_of_two_platforms_one_over_the_other():
    # Localizer.where_dot: of the platforms her dot could be on, the one whose surface is nearest its height. Taking the
    # closest at or below a point DOT_SNAP_LIFT over it, a dot 2 units over the lower of two 34 apart went onto the
    # upper (Tree Dungeon's 28 over 30 at x 157: a drop 28 -> 30 that had landed judged missed and done again, 2026-10-31).
    t = tree()
    t["foothold"]["1"]["20"] = dict([seg(1, 20, 40, 1000, 700, 1200, 700)])
    t["foothold"]["1"]["21"] = dict([seg(1, 21, 41, 1000, 666, 1200, 666)])
    m = mapdata.parse(1, t, "Test Map", {})
    loc = Localizer(m, origin=(10.0, 20.0))
    for y, on in ((698, 700), (702, 700), (690, 700), (664, 666), (668, 666), (676, 666)):
        assert loc.where_dot(loc.to_dot(1100, y)).platform.y_at(1100) == on, y


def test_a_drop_lands_out_of_reach_of_a_rope_hanging_from_the_platform_below_where_it_can():
    # route.drop_spots and _drop_moves: down still held as she lands within LAND_ROPE_CLEAR of a rope hanging from the
    # platform below climbs her onto it, so of the spots she can drop from, those clear of it, if any (Tree Dungeon's
    # drop 28 -> 30 by 30's rope at 146, 2026-10-31: none clear there, its stretch 130-149).
    from gamelab.motion.route import LAND_ROPE_CLEAR, drop_spots
    t = tree()
    t["foothold"]["1"]["22"] = dict([seg(1, 22, 42, 1400, 300, 1700, 300)])
    t["foothold"]["1"]["23"] = dict([seg(1, 23, 43, 1400, 400, 1700, 400)])
    t["ladderRope"]["1"] = {"l": 0, "uf": 1, "x": 1550, "y1": 400, "y2": 480}
    m = mapdata.parse(1, t, "Test Map", {})
    a, b = by_y(m, 300, 1550), by_y(m, 400, 1550)
    spots = drop_spots(m, a, b)
    assert spots and all(abs(x - 1550) > LAND_ROPE_CLEAR for x in spots) and min(spots) < 1550 < max(spots)
    [drop] = [x for x in Planner(m).moves if x.kind == "drop" and (x.src, x.dst) == (a.id, b.id)]
    assert abs(drop.x - 1550) > LAND_ROPE_CLEAR and abs(drop.x - 1550) <= LAND_ROPE_CLEAR + 2  # (the nearest clear)
    # A stretch all within it: every spot, as before.
    t["foothold"]["1"]["22"] = dict([seg(1, 22, 42, 1520, 300, 1580, 300)])
    m = mapdata.parse(1, t, "Test Map", {})
    a, b = by_y(m, 300, 1550), by_y(m, 400, 1550)
    assert drop_spots(m, a, b) and all(abs(x - 1550) <= LAND_ROPE_CLEAR for x in drop_spots(m, a, b))


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
    assert RUN_LAG_MS == 55  # she's 151.25 (t - 0.120) along t s in (the thief's prior), and lifts off 65 ms after the key
    assert run_ms(30) == pytest.approx(253, abs=1)  # jump key 253 ms in: lifts off 30 units along
    assert run_ms(WALK_SPEED) == pytest.approx(1000 + RUN_LAG_MS, abs=10)
    # 7 units along is 101 ms in, in the blind spot (three of the game's steps into her run, she'd leave at
    # 126): later instead, at full speed, 14 units along.
    assert run_ms(7) == 150 and run_ms(12) == 150 and run_ms(15) == 154
    assert grab_lead(41) == pytest.approx(WALK_SPEED * 0.2775, abs=1)  # top of the jump over the rope
    assert grab_lead(120) is None


def test_a_jump_from_standing_is_keyed_where_it_surely_leaves_at_full_speed(monkeypatch):
    from gamelab.motion import body, physics
    from gamelab.motion.physics import LAGS, SHOE, sure_jump_ms
    # Up to 150 ms after the arrow the jump is put off to then (the blind spot ends there, wherever in what the
    # jump sweeps allow the lags' gap is); later is as it comes. Every gap it gives leaves short at most 5% of
    # the time, by the median lags at most 0.2%.
    assert physics.BLIND_ENDS_MS == 150
    assert [sure_jump_ms(g) for g in (0, 40, 80, 120, 130, 149.2, 150, 160, 200)] == [150] * 7 + [160, 200]
    assert all(body.short_takeoff_chance(sure_jump_ms(g), SHOE, LAGS) <= 0.002 for g in range(0, 300, 7))
    worst = body.Lags(jump=LAGS.on - 0.013, sd_on=0.017 / 2 ** 0.5, sd_jump=0.017 / 2 ** 0.5)
    assert body.short_takeoff_chance(150, SHOE, worst) <= 0.05 < body.short_takeoff_chance(149, SHOE, worst)
    # MapleHelper's own running jumps wait as long (it can't import gamelab).
    from maplehelper import actions as helper_actions
    assert helper_actions.RUN_JUMP_AFTER_MS >= physics.BLIND_ENDS_MS
    # With the kick trusted (KICK_JUMPS), 40-52 ms after the arrow will do too: full speed from where she stands.
    monkeypatch.setattr(physics, "KICK_JUMPS", True)
    assert [sure_jump_ms(g) for g in (0, 40, 52, 53, 80)] == [40, 40, 52, 150, 150]


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


def test_a_trip_takes_a_way_by_a_softmax_over_their_times_with_a_floor_for_stale_moves():
    """route.chances: a way dearer than the quickest by the temperature's share of its time weighs 1/e of it; a way
    through a move she hasn't made for STALE_S (and the quickest doesn't make) has the floor at least, the others
    scaled down for it, half the chances at most between them. One way: no draw (the trip's random numbers go on
    as before)."""
    import math
    import random
    from gamelab.motion import route
    from gamelab.motion.route import Move, Way
    a, b, c = (Move("jump", 1, 2, x, 0.0) for x in (0.0, 10.0, 20.0))
    ways = [Way((a,), 10.0), Way((b,), 11.0, b), Way((c,), 12.0, c)]
    p = route.chances(ways, temperature=0.1, floor=0.0)
    assert p[1] / p[0] == pytest.approx(math.exp(-1)) and p[2] / p[0] == pytest.approx(math.exp(-2))
    assert sum(p) == pytest.approx(1.0)
    q = route.chances(ways, temperature=0.1, floor=0.3, stale=lambda m: m is c)
    assert q[2] == pytest.approx(0.3) and q[1] / q[0] == pytest.approx(p[1] / p[0]) and sum(q) == pytest.approx(1.0)
    assert route.chances(ways, temperature=0.1, floor=0.3, stale=lambda m: m is a) == pytest.approx(p)  # the quickest's
    assert sum(route.chances(ways, temperature=0.1, floor=0.4, stale=lambda m: True)[1:]) == pytest.approx(0.5)
    rng = random.Random(5)
    drawn = [route.choose(ways, rng, temperature=0.1, floor=0.0)[0] for _ in range(4000)]
    assert [drawn.count(i) / 4000 for i in range(3)] == pytest.approx(p, abs=0.03)
    rng, state = random.Random(5), random.Random(5).getstate()
    assert route.choose(ways[:1], rng) == (0, [1.0]) and rng.getstate() == state


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


def test_arriving_at_an_x_is_the_quickest_landing_and_the_walk_from_it(model):
    """Planner.arrive: to stand at an x on a platform, the quickest of the routes landing anywhere on it with the walk
    along it from there (`travel` ends wherever she lands); on her own platform, the walk (or a way round, if
    quicker); with no x, `travel`'s."""
    from gamelab.motion.physics import WALK_SPEED
    planner = Planner(model)
    for p in model.platforms:
        for q in model.platforms:
            for x in (q.left + 5, (q.left + q.right) / 2, q.right - 5):
                t = planner.arrive(p.id, p.left + 20, q.id, x)
                path = planner.plan(p.id, p.left + 20, q.id)
                if path is None and p.id != q.id:
                    assert t is None
                    continue
                if p.id == q.id:
                    assert t <= abs(x - round(p.left + 20)) / WALK_SPEED + 1e-9
                else:  # never more than the quickest route's landing and the walk from it
                    landing = planner.travel(p.id, p.left + 20, q.id)
                    assert landing - 1e-9 <= t <= landing + abs(x - round(path[-1].land_x)) / WALK_SPEED + 1e-9
        assert planner.arrive(p.id, p.left + 20, p.id) == planner.travel(p.id, p.left + 20, p.id)


def test_arriving_keeps_off_what_a_trip_keeps_off(model):
    """Planner.arrive with `keep_off`: route.KEEP_OFF_S more for each move onto one of those platforms (but the one
    she goes to), as `travel` and the navigator's trips have it: never less than without, the same going onto it, and
    never under `travel`'s keeping off the same (that ends wherever she lands)."""
    planner = Planner(model)
    for k in model.platforms:
        for p in model.platforms:
            for q in model.platforms:
                x = (q.left + q.right) / 2
                plain = planner.arrive(p.id, p.left + 20, q.id, x)
                kept = planner.arrive(p.id, p.left + 20, q.id, x, keep_off=(k.id,))
                if plain is None:
                    assert kept is None
                    continue
                assert kept >= plain - 1e-9 and (q.id != k.id or kept == plain)
                assert planner.travel(p.id, p.left + 20, q.id, keep_off=(k.id,)) <= kept + 1e-9
                assert planner.arrive(p.id, p.left + 20, q.id, keep_off=(k.id,)) == \
                    planner.travel(p.id, p.left + 20, q.id, keep_off=(k.id,))


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
    landing, to the bit (a zero's sign too) and the type, with each wall rule (body.JOINED_WALLS)."""
    from collections import OrderedDict

    from gamelab import native
    from gamelab.motion import body, physics
    for own in (True, False):
        monkeypatch.setattr(body, "JOINED_WALLS", own)
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


@pytest.mark.skipif(not __import__("gamelab.sim.scenarios", fromlist=["available"]).available(10002073),
                    reason="needs Tree Dungeon's game data (gamedata/data)")
def test_her_flight_in_c_is_pythons_on_the_real_map(monkeypatch):
    import random
    from gamelab import native
    from gamelab.sim import scenarios
    if native.lib() is None:
        pytest.skip(f"gamelab's C pieces aren't here: {native.why}")
    tree = scenarios.real_map(scenarios.TREE)
    same_flights(tree, flights(tree, random.Random(3), 1500), monkeypatch)


def test_jumps_need_a_sure_landing(model):
    from gamelab.motion.physics import JUMP_SPEED, WALK_SPEED, fly, sure_landing
    a = by_y(model, 300, 100)
    land, _, _ = fly(model, 167, 300, WALK_SPEED, -JUMP_SPEED)
    assert land is not None and sure_landing(model, 167, 300, WALK_SPEED, -JUMP_SPEED, land)
    assert not sure_landing(model, 167, 300, WALK_SPEED, -JUMP_SPEED, a)  # it doesn't land back on A


def test_a_rope_just_past_a_platforms_end_is_jumped_at_from_it():
    """A platform E (380..480, y 240) hangs a rope 15 units past B's right end, its bottom 60 above B: a running
    jump off B grabs it (a climb from B to E, taking off GRAB_LEAD before the rope). 45 units past the end it's out
    of GRAB_PAST: no climb."""
    from gamelab.motion import route
    for past, found in ((15, True), (45, False)):
        t = tree()
        t["foothold"]["1"]["5"] = dict([seg(1, 5, 8, 380, 240, 480, 240)])
        t["ladderRope"]["1"] = {"l": 0, "uf": 1, "x": 400 + past, "y1": 240, "y2": 300}
        model = mapdata.parse(1, t, "Test Map", {})
        b, e = by_y(model, 360, 300), by_y(model, 240, 420)
        climbs = [m for m in route.moves(model) if (m.kind, m.src, m.dst) == ("climb", b.id, e.id)]
        assert bool(climbs) == found
        if found:
            m, = climbs
            assert m.rope_x == 400 + past and m.direction == "right"
            assert m.x == pytest.approx(m.rope_x - route.GRAB_LEAD)


def test_a_rope_jump_from_under_a_rope_beyond_a_sure_grab_lands_by_the_grabs_chance():
    """A platform F (540..640, y 400) under the arc of a jump to the left off the rope at 700, reached from the floor by
    that rope jump alone: with the rope's bottom 50 above the floor its grab is sure; 90 above, beyond physics.GRAB_SURE
    (82) but within ROPE_REACH (95), the move lands route.GRAB_CHANCE of the time and costs its misses (MISS_S each),
    as the climb up that rope does (her grab 18 lower on the rope then: at its bottom, 410, not 428)."""
    from gamelab.motion import route
    jumps = {}
    for bottom in (450, 410):
        t = tree()
        t["ladderRope"]["0"]["y2"] = bottom
        t["foothold"]["1"]["5"] = dict([seg(1, 5, 8, 540, 400, 640, 400)])
        model = mapdata.parse(1, t, "Test Map", {})
        floor, f = by_y(model, 500, 300), by_y(model, 400, 600)
        jumps[bottom], = [m for m in route.moves(model) if (m.kind, m.src, m.dst) == ("rope_jump", floor.id, f.id)]
    assert (jumps[450].chance, jumps[410].chance) == (1.0, route.GRAB_CHANCE)
    assert jumps[410].seconds - jumps[450].seconds == pytest.approx(route.MISS_S - 18 / route.CLIMB_SPEED, abs=0.01)


def test_a_landing_is_worked_out_from_the_rounded_start_so_the_cache_never_changes_it(model, monkeypatch):
    from collections import OrderedDict
    from gamelab.motion import physics
    from gamelab.motion.physics import JUMP_SPEED, fly
    WALK_SPEED = 149.0  # (a walk whose 0.1 units/s rounding a start 0.04 more stays within: the thief's until 2026-11-02)
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
                            monsters_by_platform=spawns, platforms=[], places=[], safe=lambda pid: False)
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


def test_the_hunt_loop_keeps_the_grounds_worth_a_visit_by_their_exp_a_second(monkeypatch, tmp_path):
    # The hunt loop stays at a ground while it gives. One visit of 600 s to 2 (240 EXP) puts its EXP a visit at
    # 85, and every other ground under MIN_VALUE of that: by EXP a visit the loop, leaving 2 empty, could only
    # go back to 2 (harness seed 7: 1,150 of its 1,200 s on the basin). A second of 2 is worth 0.36 EXP, and
    # 4's 0.14 and 3's 0.06 are worth a visit by that.
    from types import SimpleNamespace

    from gamelab import paths
    from gamelab.behavior.hunt import Hunt
    from gamelab.behavior.tour import Tour
    monkeypatch.setattr(paths, "GROUNDS", tmp_path / "grounds.json")
    spawns = {1: ["Snail", "Snail", "Snail"], 2: ["Pig", "Pig"], 3: ["Slime", "Snail"], 4: ["Pig", "Orange Mushroom"]}
    model = SimpleNamespace(id=5, mob_exp={"Snail": 2, "Pig": 13, "Slime": 12, "Orange Mushroom": 15},
                            monsters_by_platform=spawns, platforms=[], places=[], safe=lambda pid: False)
    hunter = SimpleNamespace(senses=SimpleNamespace(eyes=SimpleNamespace(
        kinds={"snail", "pig", "slime", "orange-mushroom"})))
    tour = Tour(hunter, SimpleNamespace(model=model), log=lambda m: None)
    tour.record_visit(2, 240.0, 600.0)
    loop = object.__new__(Hunt)
    loop.tour = tour
    assert round(tour.ground_values()[2], 1) == 85.2 and loop._visit_s(2) == 240.0 and loop._visit_s(4) == 60.0
    assert tour.hunting_grounds() == [2]                              # the tour's sweep: by EXP a visit
    assert sorted(tour.hunting_grounds(loop._visit_s)) == [2, 3, 4]   # the hunt loop's: by EXP a second


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
        def __init__(self, hunter, nav, log, rng):
            rngs["tour"] = rng

        def run(self, **kw):
            return "done"

    hunter = SimpleNamespace(model=SimpleNamespace(id=1),
                             navigator=lambda fighting, rng=None: rngs.setdefault("nav", rng))
    monkeypatch.setattr(cli, "_hunter", lambda args, **kw: hunter)
    monkeypatch.setattr(cli, "_record", lambda command, map_id: None)
    monkeypatch.setattr(cli, "_game_title", lambda: "game")
    monkeypatch.setattr(cli.window, "bring_to_front", lambda title, wait=0.0: True)
    monkeypatch.setattr(tour, "Tour", Tour)
    assert cli.cmd_tour(cli.build_parser().parse_args(["tour", "--seed", "7"])) == 0
    want = random.Random(7).random()
    assert rngs["nav"].random() == want and rngs["tour"].random() == want


@pytest.mark.parametrize("argv", [["tour"], ["tour", "--loop"], ["nav", "3"], ["hunt", "clear"], ["hunt", "kill"],
                                  ["probe", "run", "top-60"]])
def test_a_hunting_command_brings_the_game_forward_and_stops_if_it_wont_come(argv, monkeypatch, capsys):
    """Before its first look and plan, a hunting command brings the game to the front and waits for it
    (FRONT_WAIT_S): the player refuses every plan while another window is in front (live on 2026-10-21 at
    01:01:30, a tour's first plan was refused, 'game not in front'). If it doesn't come, the command says so
    and stops there: nothing is looked at, no plan goes out."""
    from types import SimpleNamespace

    from gamelab import __main__ as cli
    from gamelab.behavior import probe, tour
    from gamelab.localization import mapdata
    asked, started = [], []

    def bring(title, wait=0.0):
        asked.append((title, wait))
        return False

    def refuse(*a, **k):
        raise AssertionError("went on with the game not in front")
    hunter = SimpleNamespace(model=SimpleNamespace(id=1), navigator=lambda *a, **k: SimpleNamespace(),
                             look=refuse, clear=refuse, kill=refuse)
    monkeypatch.setattr(cli, "_hunter", lambda args, **kw: hunter)
    monkeypatch.setattr(cli, "_record", lambda command, map_id: started.append(command))
    monkeypatch.setattr(cli, "_game_title", lambda: "game")
    monkeypatch.setattr(cli.window, "bring_to_front", bring)
    monkeypatch.setattr(cli.window, "foreground_title", lambda: "Claude")
    monkeypatch.setattr(tour, "Tour", refuse)
    monkeypatch.setattr(probe, "run", refuse)
    monkeypatch.setattr(probe, "problems", lambda model, p: [])
    monkeypatch.setattr(mapdata, "load", lambda map_id: None)
    args = cli.build_parser().parse_args(argv)
    assert args.func(args) == 1
    assert asked == [("game", cli.FRONT_WAIT_S)] and len(started) == 1
    assert "'Claude' is in front" in capsys.readouterr().out


def test_bring_to_front_says_whether_the_game_came_and_switches_to_it_half_way(monkeypatch):
    """window.bring_to_front: SetForegroundWindow (attached to the window in front's input), then a wait of
    up to `wait` s for the game to be in front; with SWITCH_FALLBACK, from half way it also switches to it as
    Alt+Tab does. With no wait (the elevated player's call) it only says whether it's in front now."""
    import importlib.util

    from gamelab import paths
    spec = importlib.util.spec_from_file_location("window_copy", paths.REPO / "gamelab" / "window.py")
    win = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(win)  # a copy of its own: the tests' window module refuses every call
    calls, clock, front = [], [0.0], {"title": "Claude"}

    class User32:
        def __getattr__(self, name):
            return lambda *a: calls.append(name) or 0
    monkeypatch.setattr(win, "user32", User32())
    monkeypatch.setattr(win, "kernel32", User32())
    monkeypatch.setattr(win, "find_game", lambda title: 7)
    monkeypatch.setattr(win, "foreground_title", lambda: front["title"])
    monkeypatch.setattr(win.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(win.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    assert win.bring_to_front("game") is False and "SetForegroundWindow" in calls
    assert "SwitchToThisWindow" not in calls
    calls.clear()
    assert win.bring_to_front("game", wait=1.0) is False and clock[0] >= 1.0
    assert calls.count("SwitchToThisWindow") == 1
    calls.clear()

    def switch_brings_it(*a):
        calls.append("SwitchToThisWindow")
        front["title"] = "game"
    monkeypatch.setattr(win.user32.__class__, "SwitchToThisWindow", staticmethod(switch_brings_it), raising=False)
    assert win.bring_to_front("game", wait=1.0) is True and calls.count("SwitchToThisWindow") == 1
    calls.clear()
    assert win.bring_to_front("game", wait=1.0) is True and calls == []  # in front already: nothing to do


def test_a_trip_mode_needs_an_escort_unless_it_avoids_monsters(model):
    """go(mode=...): avoid (no escort), hunt (the escort's bursts and monsters in the way) or transit (the
    hunt loop's trips); hunting or transit without an escort, or an unknown mode, is refused before any key."""
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


def test_the_nearest_safe_platform_is_found_from_where_she_is_once_off_a_rope():
    """Tour.nearest_safe with her hanging on a rope: off it first (Navigator._onto_platform), then the one she's on if
    it's safe, else the quickest by route from it."""
    from types import SimpleNamespace

    from gamelab.behavior.tour import Tour
    on_rope = SimpleNamespace(where=SimpleNamespace(platform=None, x=-80.0))
    off = SimpleNamespace(where=SimpleNamespace(platform=SimpleNamespace(id=10), x=-80.0))
    got, planned = [], []
    tour = object.__new__(Tour)
    tour.nav = SimpleNamespace(mover=SimpleNamespace(settle=lambda n: on_rope),
                               _onto_platform=lambda s: got.append(s) or off,
                               planner=SimpleNamespace(plan=lambda a, x, b: planned.append((a, x, b))))
    tour.safe = {10, 25}
    assert tour.nearest_safe() == 10 and got == [on_rope] and not planned
    tour.safe = {25}
    with pytest.raises(RuntimeError):  # (no route planned to 25: none reachable)
        tour.nearest_safe()
    assert planned == [(10, -80.0, 25)]


def test_the_trip_to_stand_by_goes_in_the_mode_asked_and_avoids_monsters_without_an_escort():
    """Tour.stand_by: over to the nearest safe platform in the mode asked (the navigator's own by default: the tour's;
    the hunt loop's transit), avoiding monsters without an escort, and there she stands: no chair, no waiting for her
    HP (Tour.rest went on 2026-10-26: the user, "Just remove the sit recover hp functionality")."""
    from types import SimpleNamespace

    from gamelab.behavior.tour import Tour, TourReport
    from gamelab.control.actions import Actions
    from gamelab.control.navigator import AVOID, TRANSIT
    went = []
    tour = object.__new__(Tour)
    tour.report, tour.nearest_safe = TourReport(), lambda: 25
    for escort, mode, want in ((SimpleNamespace(), None, None), (SimpleNamespace(), TRANSIT, TRANSIT),
                               (None, None, AVOID), (None, TRANSIT, AVOID)):
        tour.nav = SimpleNamespace(escort=escort, go=lambda pid, mode=None: went.append((pid, mode))
                                   or SimpleNamespace(kills=1))
        tour.stand_by(mode)
        assert went[-1] == (25, want)
    assert tour.report.stop == "standing by on a safe platform" and tour.report.on_the_way == 4
    assert not any(hasattr(Tour, name) for name in ("rest", "rest_and_stand_by", "_calm_mode"))
    assert not any(hasattr(Actions, name) for name in ("sit", "stand"))


def test_every_trip_asks_after_her_health_at_every_pass_whatever_its_mode(model):
    """Navigator.begin: the hunter's health_check (her potions, and the safety stop, which it raises: behavior.safety)
    is every trip's, in every mode. Until 2026-10-26 a transit trip only drank (low HP was why she went to rest, no
    reason to stop), and `tour --rest-avoid`'s transit-avoid with it."""
    from types import SimpleNamespace

    from gamelab.control import navigator
    from gamelab.control.navigator import AVOID, HUNT, TRANSIT, Navigator
    assert navigator.MODES == (AVOID, HUNT, TRANSIT)
    key = lambda name: SimpleNamespace(name=name)
    keys = SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), up=key("up"), down=key("down"),
                           pick_up=None)
    health = lambda s: None
    for escort, modes in ((None, (AVOID,)), (SimpleNamespace(threat=lambda kind: False), (AVOID, HUNT, TRANSIT))):
        nav = Navigator(None, SimpleNamespace(keys=keys), model, health=health, escort=escort)
        for mode in modes:
            assert nav.begin(0, mode=mode).health is health


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


def test_a_look_in_the_air_waits_for_her_to_land_again_after_a_long_fall(model):
    """Navigator._onto_platform: a look that shows her in the air waits for her to land (Mover.settle) and looks
    again; a look still in the air then (a knock down the gap between two blocks: 2.5 s to the basin) waits
    once more (FALL_LOOKS) rather than ending the trip 'lost: not on any platform' with her still falling."""
    from types import SimpleNamespace

    import numpy as np

    from gamelab.control.navigator import FALL_LOOKS
    from gamelab.localization.pose import Where
    from gamelab.localization.scene import Scene
    c = by_y(model, 500, 400)
    nav = _navigator(model)

    def scene(platform, tag_seen=True, ground=None):
        return Scene(t=0.0, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0,
                     where=Where(400, 500, platform, (0, 0)), tag_seen=tag_seen, ground=ground)
    falling, landed = scene(None), scene(c)
    looks = [falling] * (FALL_LOOKS - 1) + [landed]
    settled = []
    nav.mover = SimpleNamespace(settle=lambda seconds: settled.append(seconds) or falling,
                                locate=lambda mobs=False, wide=False: looks.pop(0))
    in_air = scene(c, ground=(10, 12))  # her dot over C, no ground under her feet on screen
    assert nav._onto_platform(in_air) is landed and len(settled) == FALL_LOOKS


def test_in_transit_the_fights_in_her_way_dont_go_on_to_finish_what_she_has_hurt(model):
    """Navigator._clear_the_way: on a hunting trip the escort's fight off a monster in her way finishes what she
    has hurt (FightOnTheWay.clear_way's `finish`); in transit (a trip to rest at low HP, the hunt loop's trips,
    whose rule 4b finishes within FINISH_S) it doesn't. Finishing there, the rest trips' fights in the basin ran
    to their GUARD_SECONDS twice as often, and loop_rest_hp_low's trip to rest ran out on 3 of seeds 1-72."""
    from types import SimpleNamespace

    from gamelab.control.navigator import HUNT, TRANSIT
    asked = []
    escort = SimpleNamespace(clear_way=lambda s, x, finish=True: asked.append(finish) or 1, threat=lambda kind: False)
    nav = _navigator(model, escort=escort)
    c = by_y(model, 500, 400)
    for mode in (HUNT, TRANSIT):
        st = nav.begin(c.id, mode=mode)
        assert nav._clear_the_way(st, SimpleNamespace(), 300.0) and st.trip.kills == 1
    assert asked == [True, False]


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
    escorted = _navigator(model, escort=SimpleNamespace(threat=lambda kind: False))
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


def _glancing(localizer, glance):
    """Senses as Navigator._watch and recover use them, glancing with `glance` (her dot and HP), each glance's frame
    shown as it's taken."""
    from types import SimpleNamespace

    import gamelab.control.navigator as nv
    senses = SimpleNamespace(wait_frame=nv.time.sleep, localizer=localizer, glanced_at=None)

    def glanced():
        senses.glanced_at = nv.time.monotonic()
        return glance()
    senses.glance = glanced
    return senses


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
    nav.senses = _glancing(localizer, lambda: (localizer.to_dot(150, 500), hp["now"]))
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
    assert cut.startswith("touched") and sent == [walk]
    assert calls["cancel"] == ["running", "queued"] and calls["wait"] == ["running"]  # the queued one withdrawn


def test_a_miss_told_after_the_look_cuts_the_running_chain_as_a_hit_does(model, monkeypatch):
    """Navigator._watch (navigator.TOUCH_CUTS): a touch of hers told after the look the chain was planned from
    (Senses.touched_since: here a MISS, her HP the same) cuts it short as a hit does, every plan cancelled; one told
    before that look doesn't, nor any with TOUCH_CUTS off."""
    from types import SimpleNamespace

    import gamelab.control.navigator as nv
    from gamelab.localization.pose import Localizer
    from gamelab.motion.flow import Body, Leg, Span
    from gamelab.motion.route import Move
    from gamelab.perception.touches import MISS, Touch
    c = by_y(model, 500, 400)
    localizer = Localizer(model)
    walk = Leg(Move("walk_off", c.id, c.id, 300, 300, "right"), Body(0.0, 100, c, run=1), Body(0.5, 175, c, run=1),
               [Span(c.id, 0.0, 0.5, 100, 175)])
    monkeypatch.setattr(nv.client, "cancel", lambda pid: True)
    monkeypatch.setattr(nv.client, "wait", lambda pid, timeout=120: {})

    def watched(told_at: float, since: float) -> str:
        nav = _navigator(model)
        nav.senses = _glancing(localizer, lambda: (localizer.to_dot(150, 500), 1.0))
        touch = {**Touch(told_at, MISS, MISS).data(), "told": told_at}
        nav.senses.touched_since = lambda t: touch if touch["told"] > t else None
        nav.flow.expected = lambda legs, t: c.id
        start = nv.time.monotonic()
        done = {"n": 0}

        def finished(pid):  # (the plan ends of itself after a few glances)
            done["n"] += 1
            return done["n"] > 5
        monkeypatch.setattr(nv.client, "done", finished)
        nav._sent_plans = ["running"]
        cut, _ = nav._watch(nv._Running("running", [walk], start, False, 0.0, 0.5), 1.0, c.id, send=None,
                            st=SimpleNamespace(), since=since)
        return cut
    now = nv.time.monotonic()
    assert watched(now, now - 0.1) == "touched (MISS: HP 100% -> 100%)"
    assert watched(now - 0.2, now - 0.1) == ""  # (told before the look it was planned from)
    monkeypatch.setattr(nv, "TOUCH_CUTS", False)
    assert watched(now, now - 0.1) == ""


def test_a_running_chain_is_cut_when_her_glances_have_no_state_to_act_on(model, monkeypatch):
    """Navigator.follow: a glance that has had no state to act on for senses.GLANCE_LOST_S raises LookLost (perception
    down, the game not on screen as expected): the plan running is cut, every key let go (client.cancel), and her last
    key is now, before it's raised on (the trip stands still: Navigator.go, Senses.regain)."""
    from types import SimpleNamespace

    import gamelab.control.navigator as nv
    from gamelab.localization.pose import Localizer
    from gamelab.motion.flow import Body, Leg, Span
    from gamelab.motion.route import Move
    from gamelab.perception.framecheck import LookLost, Problem
    c = by_y(model, 500, 400)
    nav = _navigator(model)
    localizer = Localizer(model)
    seen = iter([(localizer.to_dot(100, 500), 1.0), (localizer.to_dot(106, 500), 1.0)])

    def glance():
        try:
            return next(seen)
        except StopIteration:
            raise LookLost(Problem("perception", "the perception service's process ended")) from None
    nav.senses = _glancing(localizer, glance)
    nav.flow.expected = lambda legs, t: c.id
    ended = []
    nav.actions.hold_back = lambda events: None
    nav.actions.ended = lambda tail_s=0.0: ended.append((nv.time.monotonic(), tail_s))
    walk = Leg(Move("walk_off", c.id, c.id, 300, 300, "right"), Body(0.0, 100, c, run=1), Body(1.0, 250, c, run=1),
               [Span(c.id, 0.0, 1.0, 100, 250)])
    keys = SimpleNamespace(sorted=lambda: [[0, "right", "down"], [1000, "right", "up"]])
    cancelled = []
    monkeypatch.setattr(nv.client, "send", lambda events, **kw: "running")
    monkeypatch.setattr(nv.client, "done", lambda pid: False)
    monkeypatch.setattr(nv.client, "cancel", lambda pid: cancelled.append(pid) or True)
    with pytest.raises(LookLost):
        nav.follow([walk], keys, 1.0, c.id, "end", trip=SimpleNamespace(chains=0))
    assert cancelled == ["running"] and [tail for _, tail in ended] == [0.0]


def _watch_still_dot(model, monkeypatch, legs, x, y, every=0.03):
    """Navigator._watch over a plan of `legs` (nothing queued after it) while her dot sits still at map (x, y), on a
    clock of its own, glanced at every `every` s (each glance's frame shown as it's taken): why it was cut ('' if
    not), and when (s into the plan; the plan's end if it wasn't)."""
    from types import SimpleNamespace

    import gamelab.control.navigator as nv
    from gamelab.localization.pose import Localizer
    clock = {"now": 100.0}
    monkeypatch.setattr(nv, "time", SimpleNamespace(monotonic=lambda: clock["now"],
                                                     sleep=lambda s: clock.update(now=clock["now"] + max(0.0, s))))
    nav = _navigator(model)
    localizer = Localizer(model)
    nav.senses = _glancing(localizer, lambda: (localizer.to_dot(x, y), 1.0))
    nav.senses.wait_frame = lambda timeout: nv.time.sleep(every)
    start = nv.time.monotonic()
    cut_at = []
    monkeypatch.setattr(nv.client, "done", lambda pid: nv.time.monotonic() - start > legs[-1].end.t + 0.2)
    monkeypatch.setattr(nv.client, "cancel", lambda pid: cut_at.append(nv.time.monotonic() - start) or True)
    monkeypatch.setattr(nv.client, "wait", lambda pid, timeout=120: {})
    nav._sent_plans = ["running"]
    cut, _ = nav._watch(nv._Running("running", legs, start, False), 1.0, legs[-1].end.platform.id, send=None,
                        st=SimpleNamespace())
    return cut, (cut_at[0] if cut_at else legs[-1].end.t)


def test_standing_still_where_her_run_starts_cuts_the_plan_short(model, monkeypatch):
    """Navigator._watch: her dot still (for STILL_S) where a run along her platform starts while the
    plan has her OFF_SPAN or more along it cuts the plan short (in game, a chain whose first arrow the game
    took 0.2 s late jumped at its planned time from 15-30 units short and fell off), and before the jump at
    the end of the run; where the plan has her, it doesn't."""
    from gamelab.control.navigator import OFF_SPAN
    from gamelab.motion.flow import Body, Leg, Span
    from gamelab.motion.route import Move
    c = by_y(model, 500, 400)
    run = Leg(Move("walk_off", c.id, c.id, 300, 300, "right"), Body(0.0, 100, c), Body(1.0, 250, c, run=1),
              [Span(c.id, 0.0, 1.0, 100, 250)])
    cut, at = _watch_still_dot(model, monkeypatch, [run], 100, 500)
    assert cut.startswith("off the path (still at x 100") and OFF_SPAN / 150 < at < 0.6
    standing = Leg(run.move, Body(0.0, 100, c), Body(1.0, 100, c), [Span(c.id, 0.0, 1.0, 100, 100)])
    assert _watch_still_dot(model, monkeypatch, [standing], 100, 500)[0] == ""


def test_what_her_glances_show_is_judged_by_how_long_it_lasts_not_by_how_many_show_it(model, monkeypatch):
    """Navigator._watch: standing still where her run starts while the plan has her running is cut short OFF_PATH_S
    after the plan has her OFF_SPAN along (her dot still since it began, for STILL_S and more), at the first glance
    after that, whether she's glanced at 30 times a second (a feed's frames) or every 63 ms (as before): 0.37 and
    0.38 s in. Counted in glances (3 for both), at 30 a second it was cut 0.30 s in."""
    from gamelab.control.navigator import OFF_PATH_S, OFF_SPAN, STILL_S
    from gamelab.motion.flow import Body, Leg, Span
    from gamelab.motion.route import Move
    c = by_y(model, 500, 400)
    run = Leg(Move("walk_off", c.id, c.id, 300, 300, "right"), Body(0.0, 100, c), Body(1.0, 250, c, run=1),
              [Span(c.id, 0.0, 1.0, 100, 250)])
    behind = OFF_SPAN / 150  # (150 units/s along it)
    assert STILL_S < behind
    for every in (1 / 30, 0.063):
        cut, at = _watch_still_dot(model, monkeypatch, [run], 100, 500, every)
        assert cut.startswith("off the path (still at x 100")
        assert behind + OFF_PATH_S <= at < behind + OFF_PATH_S + 2 * every


def _watch_coming(model, monkeypatch, legs, tracks, known=()):
    """Navigator._watch over a plan of `legs` while her dot follows it and each glance comes from a feed whose state
    follows `tracks` (tracking.Track, re-found in every frame), on a clock of its own, glanced at every 1/30 s: why it
    was cut ('' if not), and when (s into the plan). `known`: the monsters the look it was planned from had."""
    from types import SimpleNamespace

    import gamelab.control.navigator as nv
    from gamelab.localization.pose import Localizer
    clock = {"now": 100.0}
    monkeypatch.setattr(nv, "time", SimpleNamespace(monotonic=lambda: clock["now"],
                                                     sleep=lambda s: clock.update(now=clock["now"] + max(0.0, s))))
    nav = _navigator(model)
    localizer = Localizer(model)
    start = nv.time.monotonic()

    def glance():
        t = nv.time.monotonic() - start
        x, p = next(((sp.x_at(t), sp.platform) for leg in legs for sp in leg.spans if sp.t0 <= t <= sp.t1),
                    (legs[-1].end.x, legs[-1].end.platform.id))
        for tr in tracks:
            tr.last_seen = nv.time.monotonic()
        senses.glanced = SimpleNamespace(placed=True, t=nv.time.monotonic(), tracks=tracks)
        return localizer.to_dot(x, model.platform(p).y_at(x)), 1.0
    senses = _glancing(localizer, glance)
    nav.senses = senses
    nav.senses.wait_frame = lambda timeout: nv.time.sleep(1 / 30)
    nav.flow.expected = lambda legs, t: None
    cut_at = []
    monkeypatch.setattr(nv.client, "done", lambda pid: nv.time.monotonic() - start > legs[-1].end.t + 0.2)
    monkeypatch.setattr(nv.client, "cancel", lambda pid: cut_at.append(nv.time.monotonic() - start) or True)
    monkeypatch.setattr(nv.client, "wait", lambda pid, timeout=120: {})
    nav._sent_plans = ["running"]
    cut, _ = nav._watch(nv._Running("running", legs, start, False, known=tuple(known)), 1.0,
                        legs[-1].end.platform.id, send=None, st=SimpleNamespace())
    return cut, (cut_at[0] if cut_at else None)


def test_a_monster_the_feed_sees_coming_into_the_running_chain_cuts_it_short_while_she_can_stop(model, monkeypatch):
    """Navigator._watch with a feed's glances (REACT): a monster it follows that one of the running chain's legs ahead
    of her now meets cuts the plan short, REACT_LEAD_S or more before the touch; not one the look the chain was planned
    from had (planned against, or past on purpose), not without room to stop before her platform's end (she'd slide
    off it), and not once it's too late."""
    from gamelab.control.navigator import COMING, REACT_LEAD_S
    from gamelab.localization.monsters import MonsterTrack
    from gamelab.localization.tracking import Track
    from gamelab.motion.flow import Body, Leg, Span
    from gamelab.motion.route import Move
    c = by_y(model, 500, 400)  # 0..800
    run = Leg(Move("walk_off", c.id, c.id, 700, 700, "right"), Body(0.0, 100, c, run=1), Body(2.0, 400, c, run=1),
              [Span(c.id, 0.0, 2.0, 100, 400)])
    pig = lambda x: Track(1, "pig", c.id, x, 500.0, None, 100.0, 100.0)
    cut, at = _watch_coming(model, monkeypatch, [run], [pig(470.0)])
    assert cut.startswith(COMING) and "pig on" in cut and at is not None and at < 2.0 - REACT_LEAD_S
    assert _watch_coming(model, monkeypatch, [run], [pig(470.0)],
                         known=[MonsterTrack(c.id, 460.0, kind="pig")])[0] == ""  # the look had it
    assert _watch_coming(model, monkeypatch, [run], [pig(2000.0)])[0] == ""  # nowhere near
    # Room to stop: running right along 0..800, only up to REACT_ROOM short of its end; in the air, never.
    from gamelab.control.navigator import REACT_ROOM, _Running
    nav = _navigator(model)
    to_end = Leg(run.move, Body(0.0, 500, c, run=1), Body(2.0, 800, c, run=1), [Span(c.id, 0.0, 2.0, 500, 800)])
    running = _Running("running", [to_end], 0.0, False)
    at_room = (800 - REACT_ROOM - 500) / 150  # (150 units/s along it)
    assert nav._room_to_stop(running, at_room - 0.05) and not nav._room_to_stop(running, at_room + 0.05)
    a, b = by_y(model, 300, 100), by_y(model, 360, 300)
    jump = Leg(Move("jump", a.id, b.id, 150, 260, "right"), Body(0.0, 150, a),
               Body(0.7, 260, b, run=1, landed=True), [Span(a.id, 0.0, 0.1, 150, 165)])
    assert not nav._room_to_stop(_Running("running", [jump], 0.0, False), 0.4)


def test_her_dot_stands_where_it_stood_until_it_moves_away_from_there():
    """navigator._stood: since when, and where, her dot has stood: within STILL of where it stood (on both axes) it
    stands there still, however it creeps; once it's further from there, it stands where it is from then; a glance
    without it starts over."""
    from gamelab.control.navigator import STILL, _stood
    stood = _stood(None, 1.0, (10.0, 20.0))
    assert stood == (1.0, (10.0, 20.0)) and _stood(stood, 1.1, (10.0, 20.0)) is stood
    assert _stood(stood, 1.2, (10.0 + 0.9 * STILL, 20.0 - 0.9 * STILL)) is stood  # (the start of a creep it allows)
    assert _stood(stood, 1.3, (10.0 + 1.1 * STILL, 20.0)) == (1.3, (10.0 + 1.1 * STILL, 20.0))
    assert _stood(stood, 1.3, (10.0, 21.0)) == (1.3, (10.0, 21.0))  # a pixel down
    assert _stood(stood, 1.4, None) is None and _stood(None, 1.5, (10.0, 20.0)) == (1.5, (10.0, 20.0))


def test_her_dot_still_over_another_platform_counts_only_once_the_plan_has_her_down(model, monkeypatch):
    """Navigator._watch: while the plan has her in the air (Flow.airborne: a jump from A onto B), her dot
    sitting still over C isn't her standing on C (near the top of a jump it can sit still over what she
    flies past); once the plan has her landed, it is, and the plan is cut short."""
    from gamelab.motion.flow import EXPECT_MARGIN, Body, Leg, Span
    from gamelab.motion.route import Move
    a, b = by_y(model, 300, 100), by_y(model, 360, 300)
    jump = Leg(Move("jump", a.id, b.id, 150, 260, "right"), Body(0.0, 150, a),
               Body(0.7, 260, b, run=1, landed=True), [Span(a.id, 0.0, 0.1, 150, 165)])
    on = Leg(Move("walk_off", b.id, b.id, 380, 380, "right"), jump.end, Body(1.5, 380, b, run=1),
             [Span(b.id, 0.7, 1.5, 260, 380)])  # and runs on along B
    cut, at = _watch_still_dot(model, monkeypatch, [jump, on], 500, 500)  # over C, below them both
    assert cut.startswith(f"off the path (on {by_y(model, 500, 500).id}") and 0.7 + EXPECT_MARGIN <= at < 1.2


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


# --- a monster's box for touches (motion.bodies) ------------------------------------------------------------

def test_a_monsters_box_for_touches_is_its_widest_and_tallest_frame(model):
    """localization.monsters.touch_box, which each track keeps (MonsterTrack.half, .tall): the first map's tables for
    their kinds (so its decisions are as they were), as tall as it flinches if the game data says so; a Horny
    Mushroom from the map's game data, 28 either side and 56 tall as it flinches (52 walking); a kind nothing knows,
    a pig walking."""
    from gamelab.localization.mapdata import MobBody
    from gamelab.localization.monsters import Tracker, touch_box
    assert touch_box("horny-mushroom", MobBody(51.0, 56.0, 28.0)) == (28.0, 56.0)
    assert touch_box("pig") == (34.0, 46.0) and touch_box("pig", MobBody(38.0, 54.0, 34.5)) == (34.0, 54.0)
    assert touch_box("axe-stump") == (34.0, 46.0)
    tree_like = mapdata.parse(1, tree(), "Test Map", {}, mob_names={"19": "Horny Mushroom"},
                              mob_bodies={"19": {"top": 51.0, "flinch_top": 56.0, "half": 28.0}})
    b = by_y(tree_like, 360, 300)
    [track] = Tracker(tree_like).update(0.0, [walker(300, kind="horny-mushroom", platform=b.id)])
    assert (track.kind, track.half, track.tall, track.height) == ("horny-mushroom", 28.0, 56.0, 56.0)
    [pig] = Tracker().update(0.0, [walker(300, platform=b.id)])  # the first map's: as before for a hop (46)
    assert (pig.half, pig.height, pig.tall) == (34.0, 46.0, 54.0)


# --- trips with an escort keep clear of the monsters whose touch counts (Navigator._meet) ---------------------

def _meeting(model, monkeypatch, escort, monster_at, mode="transit"):
    """A trip with an escort from C at x 200 to D (up the rope at 700), and the look it plans from: one Horny Mushroom
    standing at `monster_at` (platform, x), tracked as the pass tracks it. Returns the navigator, the trip, the look,
    the route, and the chain's monsters."""
    import numpy as np

    import gamelab.control.navigator as nv
    from gamelab.localization.pose import Where
    from gamelab.localization.scene import Scene
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    nav = _navigator(model, escort=escort)
    monkeypatch.setattr(nv.time, "sleep", lambda s: None)
    st = nav.begin(d.id, mode=mode)
    st.moved_at = -1e9  # not just after a plan: nothing to settle
    seen = walker(monster_at[1], kind="horny-mushroom", pose="stand", platform=monster_at[0])
    s = Scene(t=0.0, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0,
              where=Where(200, 500, c, (0, 0)), searched=True, wide=True, monsters=[seen])
    path = Planner(model).plan(c.id, 200, d.id)
    monsters, _ = nav._chain_monsters(st, s, path)
    nav.mover.locate = lambda mobs=False, wide=False: s  # (_meet's look before going regardless: _last_shot)
    return nav, st, s, path, monsters


def _escort(shots=None):
    """An escort for which every kind counts (Escort.threat), shooting with `shots` (its kills; None: no star reaches)."""
    from types import SimpleNamespace
    asked = []
    return SimpleNamespace(threat=lambda kind: True, asked=asked,
                           clear_for=lambda s, m, finish=True: asked.append((m, finish)) or shots)


def test_a_fighting_trip_reads_which_kinds_count_before_its_first_look(model):
    """Navigator.begin: with an escort, the kinds whose touch counts (Escort.threat: the danger model's shares since the
    branches met; when the escort read the game data itself a first read took half a second) are worked out as the
    trip begins, not between its first look and chain; without one, never."""
    from types import SimpleNamespace
    d = by_y(model, 300, 700)
    nav = _navigator(model, escort=SimpleNamespace(threat=lambda kind: True))
    nav.begin(d.id, mode="avoid")
    assert nav._threat_kinds is None
    nav.begin(d.id, mode="transit")
    assert nav._threat_kinds is not None


def test_a_kind_that_counts_and_hops_is_said_its_tracks_keep_to_their_platforms():
    """Navigator._threats: MonsterTrack.reach keeps a monster on its platform, and attack.danger has a kind that hops
    leave it. No kind whose touch counts hops on her three maps (test_danger); on a map with one (test_danger's Hopper,
    a Horny Mushroom's touch that hops) the navigator says so, once; with only the Horny Mushroom, nothing."""
    from types import SimpleNamespace

    from test_danger import build
    said = []
    hopper = build([(0, 500, 800, 500)], [("99", 300, 500), ("19", 600, 500)])
    nav = _navigator(hopper, escort=SimpleNamespace(threat=lambda kind: True))
    nav.log = said.append
    assert nav._threats() and nav._threats()
    assert len(said) == 1 and "(hopper)" in said[0] and "MonsterTrack.reach" in said[0], said
    said.clear()
    nav = _navigator(hopper, escort=SimpleNamespace(threat=lambda kind: kind == "horny-mushroom"))
    nav.log = said.append
    assert nav._threats() and said == []


def test_thinking_is_counted_once_however_many_chains_a_pass_plans(model, monkeypatch):
    """Navigator._chain: the time since the look counts as thinking, and a replan from the same look (_meet's hop and
    regardless) counts only the time since the chain before."""
    import gamelab.control.navigator as nv
    c = by_y(model, 500, 400)
    nav, st, s, path, monsters = _meeting(model, monkeypatch, _escort(), (c.id, 100.0))
    clock = {"now": 10.0}
    monkeypatch.setattr(nv.time, "monotonic", lambda: clock["now"])
    nav._chain(st, s, path, monsters, 9.5)
    clock["now"] = 10.25
    nav._chain(st, s, path, monsters, 9.5)
    assert st.trip.time["think"] == pytest.approx(0.75)
    clock["now"] = 12.0
    nav._chain(st, s, path, monsters, 11.5)  # a new look's: from it
    assert st.trip.time["think"] == pytest.approx(1.25)


def test_a_trip_with_an_escort_shoots_first_a_monster_that_could_touch_her_on_the_way(model, monkeypatch):
    """Navigator._meet: the run along C to the rope passes a Horny Mushroom standing at 450 (the chain stops before
    the move: flow.in_the_way). A star reaches it from where she stands (the escort's clear_for): she shoots it, and
    looks again; in transit the fight doesn't finish what she has hurt. MAX_FIGHTS shots in a row at most."""
    from gamelab.control.navigator import MAX_FIGHTS
    c = by_y(model, 500, 400)
    escort = _escort(shots=1)
    nav, st, s, path, monsters = _meeting(model, monkeypatch, escort, (c.id, 450))
    legs, keys, why = nav._chain(st, s, path, monsters, 0.0)
    assert legs == [] and why == "monster" and len(monsters) == 1 and monsters[0].kind == "horny-mushroom"
    assert nav._meet(st, s, path, monsters, legs, why, 0.0, 0.0) is True
    assert st.shots == 1 and st.trip.kills == 1 and escort.asked == [(s.monsters[0], False)]
    st.shots = MAX_FIGHTS  # as many in a row as it may: a way around instead
    assert nav._meet(st, s, path, monsters, legs, why, 0.0, 0.0) is True and len(escort.asked) == 1 and st.avoid
    nav._made(st, [path[0]])  # a move made: the shots start again from none
    assert st.shots == 0


def test_a_monster_in_the_way_no_star_reaches_from_where_she_stands_is_shot_from_along_her_platform(model, monkeypatch):
    """navigator.MEET_WALK (2026-10-30, the user: "in general you should kill all the mobs that are in the way"): where
    no star reaches the monster in the way of the route's first move from where she stands (Escort.clear_for: None),
    she walks along her platform to where her attack reaches it, SHOOT_WALK at most (Escort.shot_place, shoot_from: as
    the landing's shot first), before a way around; where there's no such place, a way around as before; off, as
    before."""
    import gamelab.control.navigator as nv
    c = by_y(model, 500, 400)
    escort = _escort(shots=None)
    walked = []
    escort.shot_place = lambda s, m, most: (430.0, "") if most == nv.SHOOT_WALK else (None, "")
    escort.shoot_from = lambda s, m, most, finish=True: walked.append((m, most, finish)) or 1
    nav, st, s, path, monsters = _meeting(model, monkeypatch, escort, (c.id, 450))
    legs, keys, why = nav._chain(st, s, path, monsters, 0.0)
    assert nav._meet(st, s, path, monsters, legs, why, 0.0, 0.0) is True
    assert walked == [(s.monsters[0], nv.SHOOT_WALK, False)] and st.shots == 1 and st.trip.kills == 1 and not st.avoid
    escort.shot_place = lambda s, m, most: (None, "no throw from the ground takes it")
    assert nav._meet(st, s, path, monsters, legs, why, 0.0, 0.0) is True and st.avoid and len(walked) == 1
    monkeypatch.setattr(nv, "MEET_WALK", False)
    st.avoid.clear()
    escort.shot_place = lambda s, m, most: (430.0, "")
    assert nav._meet(st, s, path, monsters, legs, why, 0.0, 0.0) is True and st.avoid and len(walked) == 1


def test_out_of_reach_a_monster_on_the_way_is_gone_around_hopped_or_waited_for(model, monkeypatch):
    """Navigator._meet with no star reaching it: the move costs extra so a way around is taken (st.avoid: here there's
    none, so the next pass plans the same move); then, one in the way of a run along her platform, she hops it
    (flow's hop: the chain hops over it, then climbs); one she can't hop (at the rope's top), she waits for it to
    move off (WAIT_LOOKS looks), then goes on regardless (a backstop: the chain planned as if it weren't there). One
    seen walking up and down D (its walking known: two looks) she doesn't wait for: wherever it heads, it can be at
    the rope's top as she gets there, so waiting can't clear the way."""
    from dataclasses import replace

    from gamelab import telemetry
    from gamelab.control import navigator as nv
    from gamelab.control.navigator import WAIT_LOOKS, _key
    from gamelab.motion import flow
    # (the hop timed for one standing, as first built: this look's sighting is as old as the clock, and walking at her
    # since then, flow.HOP_AT_HER, it would be at C's far end; test_evade has a hop timed for one walking at her)
    monkeypatch.setattr(flow, "HOP_AT_HER", False)
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    escort = _escort(shots=None)
    nav, st, s, path, monsters = _meeting(model, monkeypatch, escort, (c.id, 450))
    legs, keys, why = nav._chain(st, s, path, monsters, 0.0)
    assert nav._meet(st, s, path, monsters, legs, why, 0.0, 0.0) is True
    assert _key(path[0]) in st.avoid and escort.asked  # asked first: no star reaches it
    hopped = nav._meet(st, s, path, monsters, legs, why, 0.0, 0.0)
    assert isinstance(hopped, tuple) and [leg.move.kind for leg in hopped[0]] == ["hop", "climb"]
    nav, st, s, path, monsters = _meeting(model, monkeypatch, _escort(shots=None), (d.id, 700))
    legs, keys, why = nav._chain(st, s, path, monsters, 0.0)
    assert legs == [] and why == "monster" and monsters[0].vx is None  # (one look: which way it walks isn't known)
    assert nav._meet(st, s, path, monsters, legs, why, 0.0, 0.0) is True and st.avoid  # around first ...
    for i in range(WAIT_LOOKS):  # ... then waits, where nothing can reach her meanwhile
        assert nav._meet(st, s, path, monsters, legs, why, 0.0, 0.0) is True and st.waited == i + 1
    regardless = nav._meet(st, s, path, monsters, legs, why, 0.0, 0.0)  # then on regardless
    assert [leg.move.kind for leg in regardless[0]] == ["climb"] and st.waited == 0
    walking = [replace(monsters[0], vx=100.0)]  # seen walking on D: no waiting
    with telemetry.captured() as records:
        assert [leg.move.kind for leg in nav._meet(st, s, path, walking, legs, why, 0.0, 0.0)[0]] == ["climb"]
    assert st.waited == 0
    (rec,) = [r for r in records if r.get("kind") == "backstop"]  # (its record: safe to stand, not worth waiting)
    assert rec["why_not"]["safe"] is True and rec["why_not"]["worth"] is False
    standing = [replace(monsters[0], vx=0.0)]  # seen standing: it walks off sooner or later, so she waits
    assert nav._meet(st, s, path, standing, legs, why, 0.0, 0.0) is True and st.waited == 1
    monkeypatch.setattr(nav, "_safe_to_stand", lambda *a, **k: False)  # nowhere safe to stand: worth it isn't asked
    monkeypatch.setattr(nv, "EVADE", False)
    with telemetry.captured() as records:
        assert nav._meet(st, s, path, standing, legs, why, 0.0, 0.0)[0]
    (rec,) = [r for r in records if r.get("kind") == "backstop"]
    assert rec["why_not"]["safe"] is False and rec["why_not"]["worth"] is None


def test_past_meet_s_without_a_move_that_arrives_she_goes_on_regardless(model, monkeypatch):
    """Navigator._meet's MEET_S: her chains stop short of a monster whose touch counts for that long at most, counted
    from the first stop and not started again by a move cut short (a touch knocked her back: _made's `arrived`); then
    she goes on regardless, the chain planned blind as before they kept clear of anything (a backstop), and a move
    that gets her where it was going starts the count again. (Started again by every move, a trip to rest from Tree
    Dungeon's floor ran out its 180 s there: simulator, loop_tree_walls.)"""
    from gamelab.control.navigator import MEET_S
    c = by_y(model, 500, 400)
    nav, st, s, path, monsters = _meeting(model, monkeypatch, _escort(shots=1), (c.id, 450))
    legs, keys, why = nav._chain(st, s, path, monsters, 0.0)
    assert nav._meet(st, s, path, monsters, legs, why, 0.0, 0.0) is True and st.blocked is not None  # a shot first
    nav._made(st, [path[0]], arrived=False)  # set off, knocked back
    assert st.blocked is not None and st.shots == 0
    st.blocked -= MEET_S
    regardless = nav._meet(st, s, path, monsters, legs, why, 0.0, 0.0)
    blind = nav._chain(st, s, path, [], 0.0)
    assert regardless[0] and [leg.move for leg in regardless[0]] == [leg.move for leg in blind[0]]
    nav._made(st, [path[0]])
    assert st.blocked is None


def test_where_no_kind_counts_an_escorts_chains_are_planned_as_before(model, monkeypatch):
    """With an escort on a map where no kind's touch counts (the first map: Escort.threat), the chain keeps clear of
    no monster (it fights them instead), and its tail's are tracked as before (None: by the caller)."""
    from types import SimpleNamespace
    c = by_y(model, 500, 400)
    escort = SimpleNamespace(threat=lambda kind: False)
    nav, st, s, path, monsters = _meeting(model, monkeypatch, escort, (c.id, 450))
    assert monsters == [] and nav._chain_monsters(st, s, path) == ([], None)
    assert nav._chain(st, s, path, monsters, 0.0)[2] != "monster"


def test_a_fighting_trip_plans_by_the_whole_states_monsters_with_no_second_search(model, monkeypatch):
    """Navigator.trip_monsters (2026-10-30, the user: "make motion planing work on the 15fps whole screen result"): a
    band look (a fight's) carries the state's monsters off its band (Scene.swept: every state has the whole screen's),
    and the chain is planned by all of them, a band look's own and those off it, with no search of the frame again (the
    eyes' monsters() never asked: until then a band look was searched again over the chain's platforms' rows, 23 ms);
    a whole-screen look as it is. Each walks as the feed has it head at its kind's speed from the game data (_walking:
    the feed's track over the tracker's look-to-look measure, with an escort too); one off the band is kept clear of
    where the chain goes."""
    from dataclasses import replace
    from types import SimpleNamespace

    from gamelab.localization.monsters import DEFAULT_SPEED
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    asked = []
    nav, st, s, path, _ = _meeting(model, monkeypatch, _escort(), (c.id, 450))
    speed = nav.tracker.walking.get("horny-mushroom", DEFAULT_SPEED)  # (the game data's, where the map has it)
    nav.senses = SimpleNamespace(band=(140.0, 75.0), eyes=SimpleNamespace(
        monsters=lambda *a, **k: asked.append(a) or [], bar=lambda s, m: None))
    above = replace(walker(650, 1, kind="horny-mushroom", platform=d.id), track=7)  # on D, 200 above: off her band
    s.wide, s.swept = False, [above]
    s.tracks = [SimpleNamespace(id=7, vx=-speed, kind="horny-mushroom", state="live", platform=d.id)]
    assert nav.trip_monsters(s) == s.monsters + [above] and not asked
    monsters, tail = nav._chain_monsters(st, s, path)
    assert [(m.platform, m.x) for m in monsters] == [(c.id, 450.0), (d.id, 650.0)] and not asked
    walking = monsters[1]  # facing right in its picture, the feed's track heading left: the feed's
    assert walking.vx == pytest.approx(-speed) and walking.heading == -1
    assert walking.track == 7 and monsters[0].track is None  # (the look's own one has no track of the feed's)
    s.wide, s.monsters, s.swept = True, s.monsters + [above], []
    assert nav.trip_monsters(s) == s.monsters and not asked


def test_after_a_touch_no_fight_starts_nor_is_until_asked_while_her_invulnerability_lasts(model, monkeypatch):
    """navigator.POST_HIT (2026-10-30, in place of the quick look after a cut, TripState.quick): a chain cut short by a
    touch leaves the trip in the window her invulnerability after it lasts (Senses.invulnerable_until, less
    IMMUNE_MARGIN_S; INVULNERABLE_S from now where no popup was told), and while it lasts no fight starts on the way (the
    escort's shot in reach) and `until` isn't asked, and the pass's sightings aren't kept (no shot first at one); the
    pass still plans, from the look's monsters. Once it's over both are asked again."""
    import time
    from types import SimpleNamespace

    import gamelab.control.navigator as nv
    from gamelab.perception.touches import INVULNERABLE_S
    c = by_y(model, 500, 400)
    asked = []
    escort = _escort()
    escort.target = lambda s: asked.append("target")
    nav, st, s, path, _ = _meeting(model, monkeypatch, escort, (c.id, 450), mode="hunt")
    nav.actions.kind = "retreat"  # (the Lie Detector alarm file isn't read)
    end = time.monotonic() + 1.0
    nav.senses = SimpleNamespace(invulnerable_until=lambda: end)

    class Planned(Exception):
        pass

    def plan(st, s, dst):
        asked.append("plan")
        raise Planned
    monkeypatch.setattr(nav, "_look", lambda st: (s, 0.0))
    monkeypatch.setattr(nav, "_onto_platform", lambda s: s)
    monkeypatch.setattr(nav, "_plan", plan)
    st.until = lambda s: asked.append("until") or False
    nav._touched(st)
    assert st.post_hit_until == pytest.approx(end - nv.IMMUNE_MARGIN_S)
    with pytest.raises(Planned):
        nav._pass(st)
    assert asked == ["plan"]
    nav._chain_monsters(st, s, path)
    assert st.sighted == []  # (no shot first at what the look has, _meet's)
    st.post_hit_until = time.monotonic() - 1e-3  # the window over
    with pytest.raises(Planned):
        nav._pass(st)
    assert asked == ["plan", "until", "target", "plan"]
    nav._chain_monsters(st, s, path)
    assert len(st.sighted) == 1
    nav.senses = SimpleNamespace(invulnerable_until=lambda: -1e18)  # her HP dropped, no popup told: from now
    nav._touched(st)
    assert st.post_hit_until == pytest.approx(time.monotonic() + INVULNERABLE_S - nv.IMMUNE_MARGIN_S, abs=0.05)


def test_a_fighting_trip_keeps_clear_of_the_monsters_her_memory_has_where_its_chain_goes(model, monkeypatch):
    """Navigator.remembered: with her memory (a hunter's: Hunter.navigator), the monsters it has on the platforms the
    chain goes from and onto that the look didn't find, seen within TRIP_MEMORY_S, are kept clear of too, as of their
    last sighting and their walking not known (wherever they can have walked since): off its view (this look's frame is
    a pixel wide) and, since 2026-10-26, in it too (live on 2026-10-25 a Horny Mushroom the looks missed on 23, in view,
    was dropped from the plan of her jump onto 23 beside it); not one the look found (as the memory matches a sighting),
    one seen too long ago or one where the chain doesn't go; nothing without a memory. The tracker's tracks stay the
    looks'."""
    import time
    from types import SimpleNamespace

    import numpy as np

    from gamelab.control.navigator import TRIP_MEMORY_S
    a, c, d = by_y(model, 300, 100), by_y(model, 500, 400), by_y(model, 300, 700)
    nav, st, s, path, monsters = _meeting(model, monkeypatch, _escort(), (c.id, 450))
    assert len(monsters) == 1 and {c.id, d.id} <= {m.src for m in path} | {m.dst for m in path}
    s.t = 10.0

    def entry(pid, x, seen):
        return SimpleNamespace(kind="horny-mushroom", platform=pid, x=x, seen=seen)
    hidden = entry(d.id, 600.0, 9.0)  # 1 s ago on D, not in this look
    found = entry(c.id, 470.0, 9.5)  # the look's own (walked 20 units since)
    old = entry(d.id, 650.0, 10.0 - TRIP_MEMORY_S - 0.5)
    away = entry(a.id, 100.0, 9.0)
    nav.memory = SimpleNamespace(alive=lambda t, min_p: [hidden, found, old, away])
    monsters, tail = nav._chain_monsters(st, s, path)
    assert [(m.platform, m.x) for m in monsters] == [(c.id, 450.0), (d.id, 600.0)] and tail == monsters
    m = monsters[1]
    assert m.heading == 0 and m.vx is None and m.seen_at == pytest.approx(s.t - time.monotonic() - 1.0, abs=0.5)
    assert m.reach(m.seen_at + 1.0, d) == pytest.approx((max(d.left, 600.0 - m.speed), min(d.right, 600.0 + m.speed)))
    assert len(nav.tracker.last[1]) == 1  # the tracker's: the look's alone
    s.frame = np.zeros((1, 2000, 3), np.uint8)  # a view 2000 px wide around her: D's in it, where the look searched
    assert len(nav._chain_monsters(st, s, path)[0]) == 2  # missed in view: kept
    nav.memory = None
    assert len(nav._chain_monsters(st, s, path)[0]) == 1


def test_before_going_past_a_monster_regardless_she_looks_with_the_monsters_and_shoots_it_if_a_star_reaches(
        model, monkeypatch):
    """Navigator._last_shot (LAST_SHOT): with nothing chained for a monster that could touch her, no way around, no hop
    and no point waiting (seen walking), she doesn't go on regardless at once: a look with the monsters, and if it has
    that one where a star from here reaches it, she shoots it (the pass's own look may have had none: after a cut, a
    quick look's). Live on 2026-10-25 28 passes regardless were followed by 4 touches within 3 s, two into a Green
    Mushroom she had seen."""
    from dataclasses import replace

    from gamelab.control import navigator as nv
    d = by_y(model, 300, 700)
    escort = _escort(shots=1)
    nav, st, s, path, monsters = _meeting(model, monkeypatch, escort, (d.id, 700))
    walking = [replace(monsters[0], vx=100.0)]  # no point waiting for it
    legs, keys, why = nav._chain(st, s, path, walking, 0.0)
    st.sighted = []  # as after a quick look: the pass's shot has nothing to aim at
    assert nav._meet(st, s, path, walking, legs, why, 0.0, 0.0) is True and st.avoid  # around first
    asked = len(escort.asked)
    assert nav._meet(st, s, path, walking, legs, why, 0.0, 0.0) is True  # then the look, and the shot
    assert len(escort.asked) == asked + 1 and escort.asked[-1][0] is s.monsters[0] and st.shots == 1
    monkeypatch.setattr(nv, "LAST_SHOT", False)
    st.shots = 0
    regardless = nav._meet(st, s, path, walking, legs, why, 0.0, 0.0)  # as before: on regardless
    assert isinstance(regardless, tuple) and len(escort.asked) == asked + 1


def test_a_fighting_trip_keeps_clear_of_the_feeds_hidden_monsters_and_of_the_one_its_last_cut_was_for(
        model, monkeypatch):
    """Navigator.hidden_tracks (UNSEEN_TRIPS): the feed's tracks out of sight on screen (HIDDEN, seen within HIDE_S) on
    the chain's platforms, walking as followed, are kept clear of too, but not one the look found nor one of a kind that
    doesn't count; and after a chain cut for a monster coming into its way, that monster's track (TripState.came) for
    the next chain, wherever the feed has it."""
    import time
    from types import SimpleNamespace

    from gamelab.control import navigator as nv
    from gamelab.localization.tracking import HIDDEN, LIVE, Track
    from gamelab.perception.mobs import Mob
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    escort = _escort()
    nav, st, s, path, monsters = _meeting(model, monkeypatch, escort, (c.id, 450))
    now = time.monotonic()
    s.t = now

    def track(i, x, pid, state=HIDDEN, vx=0.0, kind="horny-mushroom"):
        return Track(i, kind, pid, x, 300.0, Mob(kind, 0, 0, 30, 30, 0.9), now - 5, now - 1.0, vx=vx, state=state)
    s.tracks = [track(1, 650.0, d.id, vx=-50.0), track(2, 460.0, c.id), track(3, 750.0, d.id, state=LIVE)]
    got = nav.hidden_tracks(s, {c.id, d.id}, monsters)
    assert [(mo.platform, mo.x, mo.vx) for mo in got] == [(d.id, 650.0, -50.0)]  # (#2 is the look's, #3 is live)
    st.came = track(4, 790.0, d.id, state=LIVE, kind="green-mushroom")
    kept, _ = nav._chain_monsters(st, s, path)
    assert sorted((mo.platform, mo.x) for mo in kept) == sorted([(c.id, 450.0), (d.id, 650.0), (d.id, 790.0)])
    assert st.came is None  # for one chain
    escort.threat = lambda kind: kind != "horny-mushroom"
    assert nav.hidden_tracks(s, {c.id, d.id}, []) == []
    monkeypatch.setattr(nv, "UNSEEN_TRIPS", False)
    escort.threat = lambda kind: True
    assert nav.hidden_tracks(s, {c.id, d.id}, []) == [] and SimpleNamespace


def test_a_glance_is_judged_against_the_plan_at_its_frames_time(monkeypatch):
    """Navigator._placed: where the plan should have her as of the glance's frame (`seen_at`), not as of now: at the
    perception's 15 frames a second a glance can be up to a period and a pass old (about 85 ms; 42 at 30 until
    2026-10-30)."""
    from types import SimpleNamespace

    from gamelab.control import navigator as nav_mod
    from gamelab.control.navigator import Navigator, _Running
    asked = []
    nav = Navigator.__new__(Navigator)
    nav.flow = SimpleNamespace(expected=lambda legs, t: asked.append(round(t, 6)), airborne=lambda legs, t: True,
                               span_at=lambda legs, t: None)
    nav.senses = SimpleNamespace(localizer=SimpleNamespace(to_map=lambda dot: (0.0, 0.0)))
    leg = SimpleNamespace(start=SimpleNamespace(t=0.0), end=SimpleNamespace(t=2.0))
    running = _Running("p", [leg], 100.0, False)
    monkeypatch.setattr(nav_mod, "time", SimpleNamespace(monotonic=lambda: 100.6))
    nav._placed((0.0, 0.0), False, running, seen_at=100.52)
    nav._placed((0.0, 0.0), False, running)
    assert asked == [0.52, 0.6]
