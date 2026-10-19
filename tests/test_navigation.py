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
