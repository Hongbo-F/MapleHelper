"""Where a touch's knock puts her, and what that costs her (attack.knock: the arrival study's prerequisite A, 2026-11-03,
D:/MapleHelper-scratch/arrival/REPORT.md). The knock flown as the simulator's world flies it, from her feet: the world's
own knocks, staged on Tree Dungeon's game data, come to rest where knock.fly says; the per-platform table against the
places report's zones (D:/MapleHelper-scratch/arrival/places/zones.txt) and the harness's measured outcomes at the -80
rope top (68344b2, 48 seeds: arrival/sim/runs/*_knockx.txt); the price of a knock landing her off her platform."""

import random
import time
from types import SimpleNamespace

import pytest

from gamelab.attack import knock
from gamelab.control import navigator as nav_mod
from gamelab.motion import physics
from gamelab.motion.route import Planner
from gamelab.sim import scenarios
from gamelab.sim import world as world_mod

from test_sim import KEYS

TREE = 10002073


@pytest.fixture(scope="module")
def tree_map():
    m = scenarios.real_map(TREE)
    if m is None:
        pytest.skip("needs Tree Dungeon, Forest Up North II's game data (gamedata/data)")
    return m


@pytest.fixture(scope="module")
def planner(tree_map):
    return Planner(tree_map)


def test_its_knock_is_the_worlds_and_its_hang_and_bound_the_navigators(tree_map):
    assert (knock.KNOCK_VX, knock.KNOCK_VY) == (world_mod.KNOCK_VX, world_mod.KNOCK_VY)
    assert world_mod.KNOCK_ADDS and world_mod.KNOCK_LANDS_WALKING and not world_mod.KNOCK_STEERS
    assert world_mod.Realism().knock is None and world_mod.Realism().zero_walls and world_mod.Realism().sides == "platforms"
    assert knock.sides(tree_map) == world_mod._sides(tree_map, "platforms")
    assert knock.HANG_S == nav_mod.HANG_S and knock.KNOCK_LOST_S == nav_mod.KNOCK_LOST_S


def _world_knock(m, pid: int, x: float, side: int, height: float = 0.0):
    """The world's knock staged as World._hit_by launches it (standing, or in mid-jump `height` over her floor with no
    speed sideways), run to rest with no key held."""
    p = m.platform(pid)
    w = world_mod.World(m, KEYS, (pid, x), monsters=None)
    if height:
        w.mode, w.plat, w.y, w.vx, w.vy = "air", None, p.y_at(x) - height, 0.0, 0.0
    w.knocked, w._knocked_from = True, (w.t, pid, p.y_at(x))
    w._launch(side * world_mod.KNOCK_VX, -world_mod.KNOCK_VY, w.ms)
    w.advance(4.0)
    return w


def test_the_worlds_own_knocks_come_to_rest_where_it_says(tree_map):
    """84 knocks of the world staged from random places of Tree Dungeon's platforms (standing and in mid-jump, each
    way): each comes to rest on knock.fly's platform, within 0.1 units of its x (a slide off an end: the world's next
    flight starts a tick on)."""
    rng = random.Random(7)
    places = [p for p in tree_map.places if p.id > 5]  # (0-5: the town outside the tower's walls)
    for _ in range(84):
        p = rng.choice(places)
        x, side, h = round(rng.uniform(p.left, p.right), 1), rng.choice((-1, 1)), rng.choice((0.0, 0.0, 25.0, 60.0))
        w = _world_knock(tree_map, p.id, x, side, h)
        f = knock.fly(tree_map, p, x, h or knock.GROUND, side, shoe=w.shoe)
        assert w.mode == "ground" and w.vx == 0 and f.platform is w.plat, (p.id, x, side, h)
        assert f.x == pytest.approx(w.x, abs=0.1), (p.id, x, side, h)


def test_walking_at_the_monster_the_push_adds_to_her_walk_as_the_worlds(tree_map):
    """Going at the monster at her walk the world flies her off at 270 less it (test_sim's
    test_a_knocks_push_adds_to_the_speed_she_has); going away, at 270 (the push isn't passed)."""
    walk = physics.SHOE.vmax
    for side in (-1, 1):
        assert knock.pushed(-side * walk, side) == pytest.approx(side * (world_mod.KNOCK_VX - walk))
        assert knock.pushed(side * walk, side) == side * world_mod.KNOCK_VX == knock.pushed(0.0, side)
    p = tree_map.platform(10)
    standing = knock.fly(tree_map, p, 52.0, knock.GROUND, 1)  # (on the grid: 10's left end -163, every 5)
    walking = knock.fly(tree_map, p, 52.0, knock.GROUND, 1, vx=-walk)
    assert 80 < standing.x - 52 < 95 and 30 < walking.x - 52 < 50
    assert knock.flown(tree_map, p, 52.0, knock.GROUND, 1, vx=-walk) == walking
    half = knock.flown(tree_map, p, 52.0, knock.GROUND, 1, vx=-walk / 2)
    assert walking.x < half.x < standing.x


@pytest.mark.parametrize("pid, end, width", [(10, -1, 68), (10, 1, 89), (18, -1, 85), (23, -1, 89), (25, -1, 97),
                                             (18, 1, 68), (30, 1, 76)])
def test_the_knock_off_zones_are_the_places_reports(tree_map, pid, end, width):
    """The stretch in from an end a standing knock toward it leaves the platform from (the places report's, flown at
    270/270 with physics.LAND_SLIDE for her slide): 68-97 on Tree, where the navigator's flat KNOCK_REACH says 87
    everywhere."""
    assert knock.zone(tree_map, tree_map.platform(pid), end) == pytest.approx(width, abs=1.5)


@pytest.mark.parametrize("pid, end, onto, below", [(11, 1, 14, (110, 125)), (16, 1, 19, (360, 380)),
                                                   (26, -1, 29, (225, 240)), (10, -1, 13, (295, 310))])
def test_from_25_in_a_knock_lands_far_below_where_end_drop_says_short(tree_map, pid, end, onto, below):
    """end_drop's 5-unit probe calls 11 R (52), 16 R (52) and 26 L (70) short; the knock flies past what's under them:
    11 R onto 14 120 below, 16 R onto 19, 26 L onto 29 (the places report's); 10 L onto 13, not 11 (236)."""
    p = tree_map.platform(pid)
    x = p.right - 25 if end > 0 else p.left + 25
    got, lx, fall = knock.landing(tree_map, p, x, knock.GROUND, end)
    assert got.id == onto and below[0] <= fall <= below[1]


def test_at_the_minus_80_rope_top_a_knock_from_the_right_keeps_her_on_10_and_from_100_in_it_takes_her_off(tree_map):
    """The harness (68344b2, 48 seeds each, arrival/sim/runs/*_knockx.txt): touched from her right on the ground at
    -85..-75 she stayed on 10 every time (0 of 46), at x -100 or less she went off every time (113 of 113); in the air
    at -85..-75 (the thief's jump throws in place) off 12 of 14. Every unit of those stretches, from the table."""
    p = tree_map.platform(10)
    for x in range(-85, -74):
        assert knock.landing(tree_map, p, x, knock.GROUND, -1)[0] is p
    for x in range(-163, -99):
        assert knock.landing(tree_map, p, x, knock.GROUND, -1)[0] is not p
    for h in (30.0, 45.0, 60.0, 77.0):
        for x in range(-85, -74):
            assert knock.landing(tree_map, p, x, h, -1)[0] is not p
    # The navigator's flat knock says the opposite at the top: -80 - 87 is past 10's end at -163.
    assert -80 - nav_mod.KNOCK_REACH < p.left


def _pid(f: knock.Flown) -> int | None:
    return None if f.platform is None else f.platform.id


def test_the_table_is_the_flight_at_its_grid_and_close_between(tree_map):
    """At the grid's x and AIR's heights the table holds fly's answers. Between them, over every platform of the tower
    (200 random x, side, phase and speed at the monster each): the platform she comes to rest on is fly's in 99% of the
    knocks from the ground and 96% in mid-jump (where it changes between two grid points: one of theirs); 90% of her x
    within 2 units of fly's where it is."""
    rng = random.Random(11)
    p = tree_map.platform(10)
    for x in knock.table(tree_map, p).xs[::7]:
        for phase in knock.PHASES:
            assert knock.flown(tree_map, p, x, phase, -1) == knock.fly(tree_map, p, x, phase, -1)
    same, n, errs = {"ground": 0, "air": 0}, {"ground": 0, "air": 0}, []
    for p in tree_map.places:
        if p.id <= 5:
            continue
        t = knock.table(tree_map, p)
        for _ in range(200):
            x, side = rng.uniform(p.left, p.right), rng.choice((-1, 1))
            phase = rng.choice((knock.GROUND, rng.uniform(1.0, 77.0)))
            vx = rng.choice((0.0, 0.0, -side * physics.SHOE.vmax, side * 60.0))
            got, want = knock.flown(tree_map, p, x, phase, side, vx), knock.fly(tree_map, p, x, phase, side, vx)
            kind = "air" if isinstance(phase, float) else "ground"
            n[kind] += 1
            if _pid(got) == _pid(want):
                same[kind] += 1
                errs.append(abs(got.x - want.x))
            elif not isinstance(phase, float):
                i = max(j for j, v in enumerate(t.xs) if v <= x)
                assert _pid(got) in {_pid(knock.fly(tree_map, p, v, phase, side, vx)) for v in t.xs[i:i + 2]}
    assert same["ground"] / n["ground"] > 0.99 and same["air"] / n["air"] > 0.96
    assert sorted(errs)[int(0.9 * len(errs))] < 2.0


def test_an_attack_playing_as_she_lands_stops_her_where_she_lands(tree_map):
    p = tree_map.platform(10)
    slid = knock.fly(tree_map, p, 0.0, knock.GROUND, 1)
    still = knock.fly(tree_map, p, 0.0, knock.GROUND, 1, slides=False)
    assert still.platform is slid.platform is p and 70 < still.x < 80 and 10 < slid.x - still.x < 16


def test_a_knock_on_her_platform_costs_nothing(tree_map, planner):
    p = tree_map.platform(10)
    f = knock.flown(tree_map, p, -80.0, knock.GROUND, -1)
    assert f.platform is p and knock.cost(tree_map, planner, p, -80.0, f, 15.0, 28.0) == knock.FREE


def test_off_10_at_its_end_she_pays_the_flight_and_the_way_back_and_the_hang_where_the_top_is_busy(tree_map, planner):
    """From her stop at -138 (her 2026-11-03 grind's), knocked left onto 13: the flight and fall, then the planner's
    way back up to -138 (13 -> 11 -> the -80 rope -> 10) at her rate; with a Horny standing by the rope's top, HANG_S
    more and TOP_TOUCH touches at the top."""
    p = tree_map.platform(10)
    f = knock.flown(tree_map, p, -138.0, knock.GROUND, -1)
    assert f.platform.id == 13 and f.fall > 290
    back = planner.arrive(13, f.x, 10, -138.0)
    assert back is not None and 3.0 < back < 8.0
    free = knock.cost(tree_map, planner, p, -138.0, f, 15.0, 28.0)
    assert free.seconds == pytest.approx(f.seconds + back) and free.touches == 0 and free.tops == 0
    assert free.exp == pytest.approx(15.0 * (f.seconds + back))
    horny = SimpleNamespace(platform=10, x=-70.0)
    busy = knock.cost(tree_map, planner, p, -138.0, f, 15.0, 28.0, busy=knock.busy_tops(tree_map, [horny]))
    assert busy.tops == 1 and busy.hang == knock.HANG_S
    assert busy.exp == pytest.approx(15.0 * (f.seconds + back + knock.HANG_S) + knock.TOP_TOUCH * 28.0)
    far = SimpleNamespace(platform=10, x=100.0)  # (not at the top: no hang)
    assert knock.cost(tree_map, planner, p, -138.0, f, 15.0, 28.0, busy=knock.busy_tops(tree_map, [far])) == free


def test_with_no_way_back_she_pays_the_navigators_bound(tree_map):
    p = tree_map.platform(10)
    f = knock.flown(tree_map, p, -138.0, knock.GROUND, -1)
    nowhere = SimpleNamespace(arrive=lambda *a, **k: None, plan=lambda *a, **k: None)
    got = knock.cost(tree_map, nowhere, p, -138.0, f, 10.0, 28.0)
    assert got.back is None and got.seconds == pytest.approx(f.seconds + knock.KNOCK_LOST_S)


def test_a_lookup_is_cheap_once_its_platform_is_tabled(tree_map):
    p = tree_map.platform(10)
    knock.table(tree_map, p)
    xs = [random.Random(5).uniform(p.left, p.right) for _ in range(2000)]
    t0 = time.perf_counter()
    for x in xs:
        knock.landing(tree_map, p, x, 33.0, -1)
    per = (time.perf_counter() - t0) / len(xs)
    assert per < 2e-4  # (measured ~5 us a lookup between two heights; a loose bound for slow machines)
