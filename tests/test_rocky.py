"""The second map, West Rocky Mountain II (10004021), from its game data: where her minimap dot places her, the
platforms that are one ground, and the routes over its stairs and floating steps. Each test needs the map's parsed
game data (gamedata/data/out/maps), as the first map's do."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab.localization import mapdata
from gamelab.localization.pose import Localizer
from gamelab.perception import minimap
from gamelab.perception.minimap import MinimapPicture
from gamelab.screen import NATIVE, of_frame

FIXTURES = Path(__file__).parent / "fixtures"
ROCKY = 10004021
FIRST = 10000012


def load(map_id: int = ROCKY) -> mapdata.MapModel:
    if not (mapdata.CACHE / f"{map_id}.json").is_file():
        pytest.skip("needs the map's parsed game data (gamedata/data/out/maps)")
    return mapdata.load(map_id)


def crop(name: str) -> np.ndarray:
    """A saved minimap (MINIMAP_BOX, RGB): tests/test_minimap.py's crop."""
    return np.asarray(Image.open(FIXTURES / f"minimap_{name}.png").convert("RGB")).copy()


# --- where her dot places her ----------------------------------------------------------------------------------

def picture(map_id: int = ROCKY) -> np.ndarray:
    pic = mapdata.minimap_picture(map_id)
    if pic is None:
        pytest.skip("needs the game data copy (gamedata/data/copy)")
    return pic


def test_her_dot_places_her_where_she_stood_once_the_minimap_picture_is_found():
    # On the floating step 24 (-1013..-877, both its ends on screen) the ropes at -1463 and -303 in view put her at
    # -938; on stair 29 (-923..-833) she stood near its left end. Read as if the picture were where the first map's is
    # (minimap.MinimapPicture: here it's 5 game px further left), her dot reads 80 units further right than with the
    # first map's origin alone, which put her at 24's left end (-1013, clamped) and 38 left of stair 29, on no platform.
    model = load()
    for name, pid, x in (("rocky", 24, -946), ("rocky_stairs", 29, -881)):
        box = crop(name)
        dot = minimap.dot_in_box(box)
        before = Localizer(model).where_dot(dot)
        finder = MinimapPicture(picture())
        assert finder.find(box) is not None
        sx, sy = finder.shift()
        where = Localizer(model).where_dot((dot[0] - sx, dot[1] - sy))
        assert where.platform is not None and where.platform.id == pid and where.x == pytest.approx(x, abs=1)
        assert Localizer(model).to_map((dot[0] - sx, dot[1]))[0] - Localizer(model).to_map(dot)[0] == pytest.approx(80)
        assert before.platform is None or before.platform.id != pid or before.x < where.x - 60


def test_at_1366x768_the_picture_is_found_and_her_dot_places_her_on_stair_28():
    # The game at its own size (tests/fixtures/native: she stands on stair 28, -833..-743, where the ropes in view at
    # screen x 941 and 1174 put her name tag at map x -792): the picture's corner at game px (42, 72), 5 left of where
    # the first map's is (minimap.PICTURE_BASE, 47 and 72). Her dot then reads -800 on 28; with the first map's origin
    # alone, -880, on no platform.
    model = load()
    for name in ("n28_000", "n28_008", "native_first"):
        full = np.asarray(Image.open(FIXTURES / "native" / f"{name}.png").convert("RGB"))
        geo = of_frame(full)
        x0, y0, x1, y1 = minimap.minimap_box(geo)
        finder = MinimapPicture(picture(), geo)
        assert geo == NATIVE and finder.find(full[y0:y1, x0:x1]) == (42, 72)
        assert MinimapPicture(picture(FIRST), geo).find(full[y0:y1, x0:x1]) is None  # the first map's picture
        dot = minimap.her_dot(full, geo)
        loc = Localizer(model, geo=geo)
        assert loc.where_dot(dot).platform is None
        sx, sy = finder.shift()
        where = loc.where_dot((dot[0] - sx, dot[1] - sy))
        assert where.platform.id == 28 and where.x == pytest.approx(-800, abs=1) and abs(where.x + 792) < 16


def test_on_the_first_map_the_picture_is_where_the_origin_was_fitted():
    load(FIRST)
    for name in ("stranger", "party"):
        finder = MinimapPicture(picture(FIRST))
        assert finder.find(crop(name)) is not None and finder.shift() == (0.0, 0.0)


# --- one place: 11 on 14, 21 on 22 -----------------------------------------------------------------------------

def test_11_and_21_lie_on_14_and_22_and_are_them():
    model = load()
    assert model.twins == {11: 14, 21: 22}
    assert [p.id for p in model.places] == [p for p in range(33) if p not in (11, 21)]
    for twin, host, x in ((11, 14, -1300.0), (21, 22, -1300.0)):
        t, h = model.platform(twin), model.platform(host)
        assert h.left < t.left and t.right <= h.right and abs(t.y_at(x) - h.y_at(x)) < 0.2
        assert t.group == h.group == (5, 0)  # one foothold group: their chains join (119 -> 88, 76 -> 73)
        assert model.platform_under(x, t.y_at(x) - 1).id == host  # her dot there: on the long one
    # Their spawns are the long ones': 14 has 11's three, 22 has 21's two.
    assert model.spawn_platforms[14] == 10 and model.spawn_platforms[22] == 14
    assert 11 not in model.spawn_platforms and 21 not in model.spawn_platforms
    first = load(FIRST)
    assert first.twins == {} and first.places is first.platforms


def test_no_route_jumps_between_a_twin_and_the_platform_it_lies_on():
    from gamelab.motion.route import Planner, droppable
    model = load()
    planner = Planner(model)
    assert not any({m.src, m.dst} & {11, 21} for m in planner.moves)
    for a, b in ((14, 11), (11, 14), (22, 21), (21, 22)):
        assert planner.plan(a, -1300, b) == []  # there already: she walks
    # No drop under 21 (standing there she may be on 21, and a drop takes her through it alone, onto 22): 22's drop
    # onto 31 went from -1353 before; 31 lies all under 21, so there's none.
    assert not droppable(model, model.platform(22), -1353) and droppable(model, model.platform(22), -1506)
    assert not any(m.kind == "drop" and m.src == 22 and -1412 <= m.x <= -1141 for m in planner.moves)


# --- the pit of stairs and the way out of it ------------------------------------------------------------------

def test_the_stairs_are_jumped_up_and_the_way_out_of_the_pit_is_27_onto_25():
    from gamelab.motion.physics import JUMP_HEIGHT
    from gamelab.motion.route import Planner
    model = load()
    planner = Planner(model)
    jumps = {(m.src, m.dst) for m in planner.moves if m.kind == "jump" and m.chance == 1}
    for low, high in ((32, 31), (31, 30), (30, 29), (29, 28), (28, 27), (27, 26)):
        assert (low, high) in jumps  # 60 up each, 17 to spare under her jump's 77
        assert model.platform(low).top - model.platform(high).top == 60 and JUMP_HEIGHT - 60 > 16
    [out] = [m for m in planner.moves if m.chance < 1]
    assert (out.kind, out.src, out.dst, out.direction) == ("jump", 27, 25, "left")
    assert -743 < out.takeoff < -726 and out.chance == pytest.approx(0.90, abs=0.01)
    assert model.platform(25).left < out.land_x < model.platform(25).right
    route = planner.plan(30, -1009, 19)
    assert [(m.src, m.dst) for m in route][:4] == [(30, 29), (29, 28), (28, 27), (27, 25)] and route[-1].dst == 19


def test_every_place_gets_to_every_other_but_20_under_22s_slope():
    from gamelab.motion.route import Planner
    model = load()
    planner = Planner(model)
    every = set(range(len(model.platforms)))
    for p in range(len(model.platforms)):
        assert every - planner.reachable(p) == ({20} if p != 20 else set()), p
    # 20 (-473..-442, y 2385) lies 0-6.4 units under 22's foothold from -481 to -442: a fall there lands on 22 first,
    # and walking left on 22 takes 22's own foothold (83's prev is 82); off 20 she walks onto 22, or off its left end
    # into the pit (26).
    assert {m.dst for m in planner.moves if m.src == 20} == {22, 26}
    first = load(FIRST)
    fp = Planner(first)
    assert all(fp.reachable(p.id) == set(range(len(first.platforms))) for p in first.platforms)
    assert not any(m.chance < 1 for m in fp.moves)


def test_the_jumps_that_land_only_from_near_an_end_are_found_and_the_steps_on_the_right_are_a_way_up(monkeypatch):
    # 17 (-391..-164) onto 16 (-139..1, 64 up, 25 across) surely lands from the last 20 units of 17 alone; looked for at
    # route.JUMP_FROM (33 back) only, it and four more weren't found (three more pairs of platforms, and a fourth
    # takeoff off 7's left end onto 5), and from 17 to 10 went round by 22 and the rope to 14.
    from gamelab.motion import route
    model = load()
    planner = route.Planner(model)
    jumps = {(m.src, m.dst, m.direction) for m in planner.moves if m.kind == "jump"}
    new = {(17, 16, "right"), (15, 17, "left"), (5, 4, "right"), (3, 5, "left")}
    # (179: since a step onto a floor rising the way she flies lands her, body._lands_at, the walk-off off 8's left end
    # lands on 10, not through it onto 17, and the two running jumps from 8 onto 10 are needless; 181 before)
    assert new <= jumps and len(planner.moves) == 179
    [up] = [m for m in planner.moves if (m.kind, m.src, m.dst) == ("jump", 17, 16)]
    assert up.takeoff == -164 - route.TAKEOFF_EDGE and model.platform(16).left + route.LANDING_ROOM <= up.land_x
    assert route._sure_running_jump(model, model.platform(17), -164 - route.JUMP_FROM, 1) is None
    trip = planner.plan(17, -277.5, 10)
    assert [(m.src, m.dst) for m in trip] == [(17, 16), (16, 13), (13, 10)]
    assert planner.travel(17, -277.5, 10) == pytest.approx(3.4, abs=0.1)
    monkeypatch.setattr(route, "END_LOOK", (route.JUMP_FROM,))  # looked for at JUMP_FROM alone, as before
    before = route.Planner(load())
    assert not new & {(m.src, m.dst, m.direction) for m in before.moves if m.kind == "jump"}
    assert len(before.moves) == 175
    assert before.travel(17, -277.5, 10) == pytest.approx(22.2, abs=0.1)


def test_in_the_simulator_she_gets_out_of_the_pit_up_the_rope_from_14_and_onto_11_from_22():
    # The trips the live runs of 2026-10-23 missed, with her dot placed right: up the rope from 14 to 7 (it ran past
    # 14's right end, 80 units further on than planned, and fell to 17 twice), from the pit to 19 (no route), and from
    # under 11 to 11 (jumps between 11 and 14, landings on 11 counted as elsewhere).
    from gamelab.sim.scenarios import Scenario, _nav, run_scenario
    model = load()
    for start, dst, made in (((14, -930.0), 7, "climb 14->7"), ((30, -1009.0), 19, "jump 27->25"),
                             ((22, -1300.0), 11, "climb 22->14")):
        trace = run_scenario(Scenario(f"rocky_{start[0]}_{dst}", _nav(dst), start, limit=120), model)
        result = trace["result"]
        assert isinstance(result, dict) and result["result"] == "arrived" and result["misses"] == 0, result
        assert any(m.startswith(made) for m in result["moves"]), result["moves"]
