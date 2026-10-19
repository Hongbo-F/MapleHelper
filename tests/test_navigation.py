"""Map model, localization and route planning on a small made-up map (no game data needed)."""

import pytest

from gamelab import mapdata
from gamelab.localize import Localizer
from gamelab.route import JUMP_HEIGHT, Planner, air_time

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
    from gamelab.localize import VIEW, Camera
    cam = Camera((0, 0, 3000, 2000), offset=(680, 490))
    assert cam.to_screen(1500, 1000, at=(1500, 1000)) == pytest.approx((680 * 1.40625, 490 * 1.40625))
    # Near the left edge the camera stops at 0, so she's drawn left of her usual spot.
    assert cam.to_screen(100, 1000, at=(100, 1000))[0] == pytest.approx(100 * 1.40625)
    assert cam.clamped(100, 1000) == (True, False)
    assert cam.corner(2990, 1990) == pytest.approx((3000 - VIEW[0], 2000 - VIEW[1]))


def test_camera_learns_her_spot_only_where_it_isnt_stopped():
    from gamelab.localize import Camera
    cam = Camera((0, 0, 3000, 2000), offset=(680, 490))
    cam.learn((1500, 1000), (700 * 1.40625, 500 * 1.40625), rate=1.0)
    assert cam.offset == pytest.approx((700, 500))
    cam.learn((100, 1000), (50, 600 * 1.40625), rate=1.0)  # x stopped at the edge: only y learns
    assert cam.offset == pytest.approx((700, 600))


def test_run_timing_and_rope_grab_lead():
    from gamelab.moves import RUN_LAG_MS, grab_lead, run_ms
    from gamelab.route import WALK_SPEED
    assert run_ms(7) == pytest.approx(112, abs=3)  # jump key 112 ms in: lifts off 7 units along
    assert run_ms(149) == pytest.approx(1000 + RUN_LAG_MS, abs=10)
    assert grab_lead(41) == pytest.approx(WALK_SPEED * 0.2775, abs=1)  # top of the jump over the rope
    assert grab_lead(120) is None


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


def test_jumps_need_a_sure_landing(model):
    from gamelab.route import WALK_SPEED, JUMP_SPEED, fly, sure_landing
    a = by_y(model, 300, 100)
    land, _, _ = fly(model, 167, 300, WALK_SPEED, -JUMP_SPEED)
    assert land is not None and sure_landing(model, 167, 300, WALK_SPEED, -JUMP_SPEED, land)
    assert not sure_landing(model, 167, 300, WALK_SPEED, -JUMP_SPEED, a)  # it doesn't land back on A


def test_a_tour_sweeps_nearest_first_then_tidies_the_order():
    import random
    from types import SimpleNamespace
    from gamelab.tour import Tour
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


def _stop(rounds, seen, loot_after=(), elsewhere=(), log=None):
    """A tour stop with faked fights: `rounds` gives each fight's throws, `seen` how many monsters a
    whole-screen look then sees on the platform, `loot_after` the rounds after which drops were
    collected elsewhere. Returns how many fights ran."""
    from types import SimpleNamespace
    from gamelab.hunt import Report, Target
    from gamelab.mobs import Mob
    from gamelab.shots import Spot
    from gamelab.tour import Tour, TourReport
    tour, fights = Tour.__new__(Tour), []
    tour.log, tour.report = (lambda m: None), TourReport()
    tour.spot = lambda pid, on_it=False: Spot(pid, 100, 1.0, False)
    tour.nav = SimpleNamespace(go=lambda *a, **k: SimpleNamespace(result="arrived", kills=0))
    pig = Target(Mob("pig", 0, 0, 30, 30, 0.9), 300, 500, 7)

    def clear(*a, **k):
        fights.append(1)
        return Report(throws=rounds[len(fights) - 1], stop="nothing left in reach")

    def collect():
        if len(fights) in loot_after:
            tour.report.loot_trips += 1

    tour.h = SimpleNamespace(pending_loot=[], report=Report(), clear=clear,
                             look=lambda **k: SimpleNamespace(targets=[pig] * seen[len(fights) - 1]))
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
    from gamelab import tour as tour_mod
    from gamelab.hunt import Report
    from gamelab.mapdata import MapModel
    from gamelab.tour import Tour
    monkeypatch.setattr(tour_mod, "GROUNDS", tmp_path / "grounds.json")
    spawns = {1: ["Snail", "Snail", "Snail"], 2: ["Pig", "Pig"], 3: ["Slime", "Snail"], 4: ["Pig", "Orange Mushroom"]}
    model = SimpleNamespace(id=5, mob_exp={"Snail": 2, "Pig": 13, "Slime": 12, "Orange Mushroom": 15},
                            monsters_by_platform=spawns, platforms=[], safe=lambda pid: False)
    hunter = SimpleNamespace(eyes=SimpleNamespace(detector=SimpleNamespace(
        templates={"snail": 0, "pig": 0, "slime": 0, "orange-mushroom": 0})))
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


def test_monsters_left_out_of_reach_are_shot_from_another_spot_before_looting():
    from gamelab.shots import Spot
    log = []
    # After the first fight 2 are left out of reach; a spot on platform 9 reaches them: she goes there
    # and fights again, and only then collects the drops.
    assert _stop([2, 2, 0], seen=[2, 0, 0], elsewhere=[Spot(9, 300, 0.0, True)], log=log) == 2
    assert any("over to platform 9" in m for m in log)
    assert log.index(next(m for m in log if "over to platform 9" in m)) < len(log) - 1
