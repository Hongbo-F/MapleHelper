"""The simulator on the third map, Tree Dungeon, Forest Up North II (10002073), and what isn't known of the game's
physics there, which the world can run either way (world.Realism: the map's sides, the walls of foothold group 0,
how far up a rope can be grabbed). Synthetic maps need no game data; the real-map tests are skipped without it."""

import pytest

from gamelab.localization import mapdata
from gamelab.motion import body
from gamelab.sim import scenarios
from gamelab.sim import world as world_mod
from gamelab.sim.world import ROPE_REACH_CLIENT, Realism, SimFellOut, World

from test_sim import KEYS, plat, seg, tap, tree

TREE = 10002073


def tree_map() -> mapdata.MapModel:
    m = scenarios.real_map(TREE)
    if m is None:
        pytest.skip("needs Tree Dungeon, Forest Up North II's game data (gamedata/data)")
    return m


def tower() -> mapdata.MapModel:
    """A tower like Tree Dungeon's: walls of group 0 at x 0 and 600 from y 0 down to its floor (y 500, group 0), a
    floating block of another group (40..300, y 300) with no walls, and under the block's left end a shelf of group 0
    against the left wall (0..120, y 400); a rope from the block (x 250) down to 90 above the floor; and outside the
    tower, above it, a wall cap (-300..-200, y -100), as Tree Dungeon's caps above its VR widen its footholds."""
    t = tree()
    t["info"].update(VRLeft=-100, VRRight=700, VRBottom=600)
    t["foothold"] = {"1": {
        "0": dict([seg(1, 0, 0, 0, 400), seg(2, 0, 400, 0, 500, 0, 3), seg(3, 0, 500, 600, 500, 2, 4),
                   seg(4, 600, 500, 600, 0, 3, 0), seg(5, 0, 400, 120, 400), seg(7, -300, -100, -200, -100)]),
        "5": dict([seg(6, 40, 300, 300, 300)]),
    }}
    t["life"] = {}
    t["ladderRope"] = {"0": {"l": 0, "uf": 1, "x": 250, "y1": 300, "y2": 410}}
    return mapdata.parse(1, t, "Test Tower", {})


def world(m, pid, x, **kw):
    kw.setdefault("monsters", None)
    return World(m, KEYS, (pid, x), **kw)


# --- what isn't known of the game's physics ------------------------------------------------------------------------

def test_off_a_floating_blocks_outer_end_she_falls_past_the_towers_wall_unless_group_0s_walls_stop_her():
    m = tower()
    block, shelf, floor = plat(m, 300, 100), plat(m, 400, 60), plat(m, 500, 300)
    assert block.group != shelf.group == floor.group and block.group[1] != 0 and shelf.group[1] == 0

    def walk_off(realism=None) -> World:
        w = world(m, block.id, 100.0, realism=realism)
        tap(w, 0.0, "left", 1.0)
        try:
            w.advance(4.0)
        except SimFellOut:
            pass
        return w

    w = walk_off()  # her own group's walls only (body.OWN_GROUP_WALLS): out past x 0, nothing under her
    assert w.fell_out_at is not None and w.x < 0
    w = walk_off(Realism(zero_walls=True))  # group 0's wall stops her, and she comes down by it onto the shelf
    assert w.fell_out_at is None and w.plat is shelf and 0 <= w.x < 10
    assert world_mod._walls_for(m, block.group, zero=True) != world_mod._walls_for(m, block.group)
    assert world_mod._walls_for(m, shelf.group, zero=True) == world_mod._walls_for(m, shelf.group)


def test_the_maps_sides_are_its_platforms_ends_or_the_clients_25_in_from_its_footholds(monkeypatch):
    m = tower()
    assert world_mod._sides(m) == (-300, 600) and world_mod._sides(m, "client") == (-275, 575)
    with pytest.raises(ValueError):
        world_mod._sides(m, "walls")
    cap = plat(m, -100, -250)
    for realism, end in ((None, -300.0), (Realism(sides="client"), -275.0)):  # walking left on the cap: stopped at
        w = world(m, cap.id, -250.0, realism=realism)                        # its end (the side there), or 25 in
        tap(w, 0.0, "left", 1.5)
        w.advance(4.0)
        assert w.plat is cap and w.x == pytest.approx(end, abs=0.01)
    if scenarios.real_map(TREE) is not None:
        assert world_mod._sides(tree_map()) == (-500, 590) and world_mod._sides(tree_map(), "client") == (-475, 565)


def test_a_rope_90_up_is_grabbed_at_the_planners_reach_and_not_at_the_clients():
    m = tower()
    floor, block = plat(m, 500, 300), plat(m, 300, 100)
    assert ROPE_REACH_CLIENT == pytest.approx(82, abs=0.01)

    def jump_for_it(realism=None) -> World:
        w = world(m, floor.id, 250.0, realism=realism)
        w.key(0.0, "up", True)
        tap(w, 0.05, "alt", 0.04)
        w.advance(4.0)
        return w

    w = jump_for_it()  # 90 up: within ROPE_REACH's 95, she grabs it and climbs onto the block
    assert w.plat is block and w.mode == "ground"
    w = jump_for_it(Realism(rope_reach=ROPE_REACH_CLIENT))  # a point 5 above her feet never gets to it
    assert w.plat is floor and w.mode == "ground"
    assert not any("grabs the rope" in line for line in w.log)
    assert body.OWN_GROUP_WALLS  # (the world's default, as fitted on the first map's flights)


# --- the planner on Tree Dungeon, Forest Up North II ---------------------------------------------------------------

def test_the_planner_gets_onto_7_from_8_and_up_from_the_bottom_by_29():
    from gamelab.motion.route import GRAB_CHANCE, MISS_S, Planner
    m = tree_map()
    planner = Planner(m)
    by = {(mv.kind, mv.src, mv.dst): mv for mv in planner.moves}
    # 7 (two Horny Mushrooms) by a running jump off 8's left end (route.END_LOOK, from West Rocky Mountain II's 17 onto
    # 16): the study's brute force lands it from takeoffs at x 24-48 on 8 (2-26 units in from its end at 22)
    jump = by["jump", 8, 7]
    assert 24 <= jump.takeoff <= 48 and m.platform(7).covers(jump.land_x) and jump.chance == 1.0
    assert ("jump", 25, 29) in by
    # The rope at -8 from 30 ends 94.3 above it: beyond a sure grab (82), within ROPE_REACH (95)
    climb = by["climb", 30, 27]
    assert climb.chance == GRAB_CHANCE and climb.seconds == pytest.approx(by["climb", 29, 27].seconds + 0.49
                                                                          + (1 / GRAB_CHANCE - 1) * MISS_S, abs=0.01)
    assert [f"{mv.kind} {mv.src}->{mv.dst}" for mv in planner.plan(30, 60.0, 27)] == ["jump 30->29", "climb 29->27"]
    up = planner.plan(33, 100.0, 6)
    assert "climb 30->27" not in [f"{mv.kind} {mv.src}->{mv.dst}" for mv in up] and up[-1].dst == 6
    assert [mv for mv in planner.moves if mv.kind == "climb" and mv.chance < 1] == [climb]


def test_on_the_other_maps_every_climb_is_a_sure_grab():
    from gamelab.motion.route import Planner
    for map_id in (scenarios.MAP_ID, scenarios.ROCKY):
        m = scenarios.real_map(map_id)
        if m is None:
            continue
        assert all(mv.chance == 1.0 for mv in Planner(m).moves if mv.kind == "climb")


def test_jumping_for_the_rope_from_30_at_the_clients_reach_she_misses_and_lands_on_29():
    m = tree_map()
    p29, p27 = m.platform(29), m.platform(27)
    for realism, lands in ((None, p27), (Realism(rope_reach=ROPE_REACH_CLIENT), p29)):
        w = world(m, 30, -8.0, realism=realism)
        w.key(0.0, "up", True)
        tap(w, 0.05, "alt", 0.04)
        w.advance(4.0)
        assert w.plat is lands and w.mode == "ground", (realism, w.plat)


# --- her at level 25 (24 until 2026-10-26), and the map's monsters ---------------------------------------------------

def test_she_is_level_25_there_with_her_pet_her_new_keys_and_her_stars_as_measured():
    from gamelab.attack import choice
    from gamelab.control.timeline import SIM_KEYS_PET
    from gamelab.sim.world import DEFENCE, LEVEL_24, LEVEL_25
    assert scenarios.her_on(TREE) == (SIM_KEYS_PET, LEVEL_25)
    assert (LEVEL_25.max_hp, LEVEL_25.level, LEVEL_25.defence, LEVEL_25.mp) == (774, 25, DEFENCE, 623)
    assert LEVEL_25.star == choice.STAR and LEVEL_25.skill_star == choice.SKILL_STAR  # (the stack reckons as drawn)
    assert LEVEL_25.cast_mp == choice.cast_mp(choice.SKILL_LEVEL) == 14 and LEVEL_25.mp_potion == 100
    assert (LEVEL_24.max_hp, LEVEL_24.level, LEVEL_24.mp, LEVEL_24.star) == (752, 24, 601, None)
    # Her pet drinks an Orange Potion (250 HP) under 70%, as live on 2026-10-25 (world.PET_HP, HP_POTION)
    for her in (LEVEL_24, LEVEL_25):
        assert (her.pet, her.potion) == (0.70, 250.0)


def test_a_horny_mushrooms_touch_takes_111_to_131_hp_by_the_formula_and_75_give_or_take_70_percent_by_a_share():
    import dataclasses
    from gamelab.sim.world import LEVEL_24, MobInfo, touch_damage
    horny, green, slime = MobInfo(110.0, 94.0), MobInfo(54.0, 79.0), MobInfo(13.0, 53.0)
    hp = [round(touch_damage(horny, u / 100, LEVEL_24) * 752) for u in range(100)]
    assert (min(hp), max(hp)) == (111, 131)
    assert {round(touch_damage(green, u, LEVEL_24) * 752) for u in (0.0, 0.999)} == {49, 59}
    assert {round(touch_damage(slime, u, LEVEL_24) * 752) for u in (0.0, 0.999)} == {7, 9}
    share = dataclasses.replace(LEVEL_24, defence=None)  # the harness's --touch share
    assert touch_damage(horny, 0.5, share) * 752 == pytest.approx(110 * 0.68)
    assert touch_damage(horny, 0.0, share) * 752 == pytest.approx(110 * 0.68 * 0.3)


def test_her_evasion_goes_against_the_monsters_accuracy():
    import dataclasses
    from gamelab.sim.world import EVADE_ACC, LEVEL_24, MobInfo, evade_chance
    horny = MobInfo(110.0, 94.0)
    assert evade_chance(horny) == evade_chance(horny, LEVEL_24) == pytest.approx(EVADE_ACC / 94)  # 29%
    assert evade_chance(horny, dataclasses.replace(LEVEL_24, evade=9.4)) == pytest.approx(0.1)


def test_a_hit_pushes_a_monster_back_only_from_its_pushed_damage():
    from gamelab.sim.world import push_chance
    assert push_chance(1.0, 10.0) == push_chance(1.0, 0.0, swipe=True) == 1.0  # the first map's kinds: any damage
    assert push_chance(40.0, 10.0) == pytest.approx(27 / 41)   # a Horny Mushroom: 44-70 of her 30-70, after DEF 10
    assert push_chance(25.0, 0.0) == 1.0                       # a Green Mushroom: every star
    assert push_chance(30.0, 30.0) == pytest.approx(32 / 41)   # an Axe Stump (DEF 30): 39-70
    assert push_chance(25.0, 30.0) == pytest.approx(38 / 41)   # a Dark Stump: 33-70
    assert push_chance(30.0, 30.0, swipe=True) == 0.0          # no claw swipe (9-11) pushes a stump


def test_a_star_that_doesnt_push_a_horny_mushroom_leaves_it_walking_at_her(monkeypatch):
    m = tree_map()
    p = m.platform(23)
    for roll, pushed in ((0.99, False), (0.0, True)):
        monkeypatch.setattr(World, "chance", lambda self, what, ms=None, roll=roll: roll if what.startswith("push")
                            else 0.5)
        w = world(m, 23, 0.0, monsters=[("Horny Mushroom", 23, 150.0)], facing="right")
        w.monsters[0].speed = 0.0
        w.invulnerable_ms = 10_000
        tap(w, 0.0, "f", 0.04)
        w.advance(0.7)
        h = w.monsters[0]
        assert h.alive and h.hp == h.hp_max - 1 and h.hit_at is not None
        assert (h.flinch_at is not None) == pushed and w.flinching(h) == pushed
        assert any("not pushed" in line for line in w.log) == (not pushed)
        assert (h.x0 > 150.0) == pushed  # pushed 18 units away from her, or where it was
    assert p.covers(150.0)


# --- the portals -----------------------------------------------------------------------------------------------------

def test_up_held_at_a_portal_takes_her_out_of_the_map_in_the_air_too_and_no_planned_climb_is_near_one():
    """As HeavenClient takes a portal (mapdata.PORTAL_BOX): up held with her feet within 25 of it across and from 100
    above it to 25 below, on the ground or in the air, unless she's on a rope, sitting or attacking."""
    from gamelab.motion.route import Planner
    from gamelab.sim.world import SimLeftMap
    m = tree_map()
    out00 = next(pt for pt in m.portals if pt.name == "out00")
    w = world(m, 28, out00.x - 10)
    w.key(0.0, "up", True)
    with pytest.raises(SimLeftMap):
        w.advance(0.5)
    assert w.summary()["left_map_at"] == 0.0
    for dx, taken in ((10, True), (40, False)):  # up pressed in the air, at the top of a jump straight up
        w = world(m, 28, out00.x - dx)
        tap(w, 0.0, "alt", 0.05)
        w.key(0.3, "up", True)
        if taken:
            with pytest.raises(SimLeftMap):
                w.advance(1.0)
            assert w.summary()["left_map_at"] == 0.3 and any("in the air" in line for line in w.log)
        else:
            w.advance(1.0)
    w = world(m, 28, out00.x - 60)  # up held walking into its box: taken there
    w.key(0.0, "up", True)
    w.key(0.1, "right", True)
    with pytest.raises(SimLeftMap):
        w.advance(2.0)
    assert out00.x - 26 < w.x <= out00.x - 24
    # No move of the planner's holds up within 60 of either, nor over one (MapModel.portal_near): no rope is near one.
    for mv in Planner(m).moves:
        if mv.kind in ("climb", "rope_jump"):
            assert not m.portal_near(m.platform(mv.src), mv.rope_x), mv
    for pid in (28, 12):
        assert m.portal_near(m.platform(pid), 240.0) and not m.portal_near(m.platform(pid), 110.0)


def test_a_scenario_that_takes_a_portal_or_falls_out_of_the_map_is_unmet_as_a_death_is():
    """scenarios.unmet: the world counts a portal taken (left_map_at) and a fall out of the map (fell_out_at) apart from
    a death, and none of them is her alive at the end, unless the scenario `may_die`."""
    sc = scenarios.Scenario("t", lambda sim: None, (28, 247.0), map_id=TREE)
    for world, why in (({"dead": True}, "she died"), ({"dead": False, "left_map_at": 3.0}, "she went through a portal"),
                       ({"dead": False, "fell_out_at": 4.0}, "she fell out of the map"), ({"dead": False}, None)):
        trace = {"scenario": "t", "result": "r", "world": world}
        assert scenarios.unmet(trace, sc) == ([f"{why}: r"] if why else [])
        assert scenarios.unmet(trace, scenarios.Scenario("t", sc.fn, sc.start, may_die=True, map_id=TREE)) == []


# --- other players ---------------------------------------------------------------------------------------------------

def _camp_world(m, **kw):
    from gamelab.sim.world import Camp
    camps = (Camp(28, 250.0, (29, 30), ("green-mushroom",)),)
    return world(m, 23, 0.0, monsters={29, 30, 23}, realism=Realism(camps=camps, respawn=8.8, **kw))


def test_a_camp_kills_the_greens_it_hunts_2_to_6_s_after_they_show_and_leaves_the_rest():
    m = tree_map()
    w = _camp_world(m)
    w.invulnerable_ms = 10**9  # (the Horny Mushrooms on 23, by her: they don't count here)
    w.advance(30.0)
    kills = [k for k in w.kill_log if k[3] == "rival"]
    first = sorted(t for t, _, _, _ in kills)[:2]
    assert {k[1] for k in kills} == {"green-mushroom"} and {k[2] for k in kills} == {29, 30}
    assert all(2.0 <= t <= 6.0 for t in first)  # the two there from the start
    assert len(kills) >= 4  # and again once they're back (8.8 s), 2-6 s after
    assert all(m_.alive for m_ in w.monsters if m_.kind == "horny-mushroom")
    taken = [d for d in w.drops if d.owner == "rival" and d.gone]
    assert taken and all(d.taken_by == "rival" and d.taken_at - d.since <= 1.0 + 1e-6 for d in taken)
    assert w.others_at() == [(250.0, m.platform(28).y_at(250.0))]


def test_with_shared_exp_a_monster_she_hurt_and_another_player_killed_gives_her_her_share():
    m = tree_map()
    for shared, exp in ((False, 0), (True, round(26 * 2 / 5))):
        w = _camp_world(m, shared_exp=shared)
        g = next(x for x in w.monsters if x.kind == "green-mushroom")
        g.hp -= 2  # two of her five stars
        w.invulnerable_ms = 10**9
        w.advance(6.5)
        assert not g.alive and w.exp == exp


def test_other_players_dots_show_in_the_minimap_windows_view_and_not_under_hers():
    import dataclasses
    from gamelab.sim import fakes
    from gamelab.sim.world import Camp, Noise
    m = tree_map()
    camps = (Camp(29, -100.0, (29,), ()), Camp(28, 240.0, (28,), ()), Camp(13, -100.0, (13,), ()))
    clock = fakes.Clock()
    w = World(m, KEYS, (28, 262.0), monsters=None, origin=clock.now, realism=Realism(camps=camps))
    eyes = fakes.FakeEyes(w, clock, m)
    # From 28 the window has stopped at the picture's bottom: y -998 and below (the dots' rows, 7.2 above the map's own
    # rows in the picture: fakes.FakeEyes._picture_row), not -1113 as the map's rows put it.
    assert eyes.in_window(m.platform(29).top) and eyes.in_window(-990) and not eyes.in_window(-1005)
    assert len(eyes._others()) == 2  # 13 (y -1190) is out of the window's view from 28
    w.noise = dataclasses.replace(w.noise, dot_under=56.0)
    assert len(eyes._others()) == 1  # the one 22 units from her is under her dot
    w.y = m.platform(13).top  # up on 13 the window has scrolled: 13 in view, 28 and 29 not
    assert eyes.in_window(-1190) and not eyes.in_window(-49)
    assert [eyes.in_window(y) for y in (-1200, -50)] == [True, False]
    if (rocky := scenarios.real_map(scenarios.ROCKY)) is not None:  # its whole picture shows
        eyes = fakes.FakeEyes(World(rocky, KEYS, (22, -700.0), monsters=None, origin=clock.now), clock, rocky)
        assert eyes.in_window(-5000) and eyes.in_window(5000)


def test_the_camera_keeps_her_529_down_there_so_from_28_the_horny_mushrooms_on_31_are_behind_the_chat_bar():
    """fakes.CAMERA_DOWN: on the study's frames from 28 screen y = map y + 579 (her feet at row 529, not the first map's
    489): 31's Horny Mushrooms stand at rows 701-718, under the HUD's top (675.5), and aren't searched; 30's Greens
    are."""
    from gamelab.screen import STRETCHED
    from gamelab.sim import fakes
    m = tree_map()
    clock = fakes.Clock()
    w = world(m, 28, 247.0, monsters=[("Horny Mushroom", 31, -152.0), ("Green Mushroom", 30, 64.0)], origin=clock.now)
    eyes = fakes.FakeEyes(w, clock, m)
    assert eyes.camera.offset[1] == 529.0
    assert eyes.camera.to_screen(247.0, w.y, (247.0, w.y))[1] == pytest.approx(529.0 * STRETCHED.stretch)
    assert [mob.kind for mob in eyes.look().frame.screen.mobs] == ["green-mushroom"]


def test_with_the_maps_drops_its_kills_drop_what_its_detectors_look_for_and_the_fake_eyes_name_them_so():
    """The harness's `game-drops` on Tree Dungeon (world.MAP_DROPS: its monsters' drops, perception.drops): a Horny
    Mushroom's kills drop its cap 40% of the time and coins 72%, and nothing the map's loot detector doesn't look for
    (drops.items_for); the fake eyes draw a drop where it lies, named as that detector names it, and the fake monster
    detector looks for the map's kinds, as its learned gates do (perception.gates)."""
    from gamelab.perception import drops
    from gamelab.perception.kinds import kind_of
    from gamelab.sim import fakes
    from gamelab.sim.world import MAP_DROPS, MESOS_SHARE, Drop
    m = tree_map()
    clock = fakes.Clock()
    w = world(m, 28, 247.0, monsters=[("Green Mushroom", 30, 64.0)], origin=clock.now, realism=Realism(drops=MAP_DROPS))
    looked_for = {kind_of(name) for name in drops.items_for(m)} | {"mesos"}
    kills = [w._drop_kinds("horny-mushroom", w._drop_rng) for _ in range(4000)]
    assert set().union(*kills) <= looked_for
    assert sum("horny-mushroom-cap" in k for k in kills) / len(kills) == pytest.approx(0.40, abs=0.03)
    assert sum("mesos" in k for k in kills) / len(kills) == pytest.approx(MESOS_SHARE, abs=0.03)
    assert {"horny-mushroom", "green-mushroom", "slime"} <= set(fakes.detector_kinds(m))
    p = m.platform(28)
    w.drops.append(Drop("horny-mushroom-cap", 200.0, p.y_at(200.0), p.id, w.t))
    eyes = fakes.FakeEyes(w, clock, m)
    seen = eyes.look().frame.screen
    assert [i.kind for i in seen.items] == ["horny-mushroom-cap"]
    assert [mob.kind for mob in seen.mobs] == ["green-mushroom"]


# --- landing on a floor that rises the way she flies ---------------------------------------------------------------

def test_a_step_onto_a_floor_that_rises_the_way_she_flies_lands_her():
    from gamelab.motion.body import _lands_at
    t = tree()
    t["foothold"] = {"1": {"1": dict([seg(1, 0, 500, 200, 480)])}}  # rises 0.1 a unit to the right, as 33 does
    t["life"], t["ladderRope"] = {}, {}
    floor = mapdata.parse(1, t, "Test Map", {}).platforms[0]
    # From 0.05 above the surface under her at x 100 (499.95 against 490: 9.95 below it at 100.75, where the step ends,
    # whose surface is at 489.925) to 0.5 below it: she was above it where the step started and is below it where it
    # ends, as HeavenClient's crossing has it; weighed against the surface at its end alone, she went through.
    y0 = floor.y_at(100.0) - 0.05
    assert _lands_at(floor, 100.0, y0, 100.75, floor.y_at(100.75) + 0.5) == 100.75
    assert y0 > floor.y_at(100.75)  # (below the surface at the step's end when it starts)
    assert _lands_at(floor, 100.0, floor.y_at(100.0) + 1.0, 100.75, floor.y_at(100.75) + 2.0) is None  # under it
    assert _lands_at(floor, 100.75, floor.y_at(100.75) - 0.05, 100.0, floor.y_at(100.0) + 0.5) == 100.0  # downhill too


def test_knocked_right_off_30_she_lands_on_tree_dungeons_floor_not_through_it():
    # The floor (33) rises to the right, 248 to 225. In the hunt loop (seed 4, 16 s; 5 minutes from 28) a touch knocked
    # her right off 30 at x 143.7 (180 out, 220 up): she came down at x 266 and, weighed against the surface at each
    # step's end alone, through the floor and out of the map. (Of 6 five-minute hunt loops from 28, 4 fell out of the
    # map, two of them inside the tower: none of those now.)
    m = tree_map()
    floor = m.platform(33)
    assert floor.y_at(-220) > floor.y_at(310)
    w = world(m, 30, 143.7)
    w._launch(180.0, -220.0, w.ms)
    w.advance(2.0)
    assert w.plat is floor and w.fell_out_at is None and 250 < w.x < 300


def test_knocked_off_the_rope_by_the_towers_left_wall_she_falls_out_unless_group_0s_walls_stop_her():
    # The rope at -192 (29 down to 31) hangs 28 from the tower's wall at -220. Off a rope no wall stops her (body.
    # ROPE_GROUP): knocked left off it she went out of the tower (loop_tree_run's seed 10, 328 s); with group 0's walls
    # stopping her (Realism.zero_walls) she comes down by the wall onto 31.
    m = tree_map()
    for realism, out in ((None, True), (Realism(zero_walls=True), False)):
        w = world(m, 31, -192.0, hanging=0.0, realism=realism)
        assert w.mode == "rope" and w.group is None
        w._leave_rope()
        w.knocked = True
        w._launch(-180.0, -220.0, w.ms)
        try:
            w.advance(4.0)
        except SimFellOut:
            pass
        assert (w.fell_out_at is not None) == out
        if not out:
            assert w.plat is m.platform(31) and w.x == pytest.approx(-220, abs=1)


# --- the harness there ---------------------------------------------------------------------------------------------

def test_the_harness_runs_tree_dungeon_with_its_crowd_and_the_unknowns_switched_on():
    import dataclasses
    from gamelab.sim import harness
    from gamelab.sim.world import LEVEL_25
    sc = harness.scenario("loop", 1, "crowd", map_id=TREE, world=("zero-walls", "touch-share"))
    assert sc.start == harness.TREE_START_AT[1] and sc.name == "loop:crowd:1:10002073:zero-walls:touch-share"
    assert sc.realism.camps == harness.TREE_CAMPS and sc.realism.respawn == 8.8 and sc.realism.zero_walls
    assert sc.noise.dot_under == 56.0 and sc.realism.spawn_share == 1.0 and sc.realism.rival_every == 0
    for preset, respawn in (("crowd-17", 17.0), ("crowd-slow", 30.0)):  # the live median on 23, the first map's refill
        other = harness.scenario("loop", 1, preset, map_id=TREE)
        assert other.realism.respawn == respawn and other.realism.camps == harness.TREE_CAMPS
    assert sc.her == dataclasses.replace(LEVEL_25, defence=None)
    assert harness.scenario("loop", 1, "game", map_id=TREE).her is None  # as she is there (scenarios.HER)
    assert harness.scenario("tour", 2, map_id=TREE, world=("client-reach",)).realism.rope_reach == ROPE_REACH_CLIENT
    assert harness.scenario("loop", 1, map_id=TREE, world=("red-potion",)).her == dataclasses.replace(LEVEL_25,
                                                                                                    potion=100.0)
    with pytest.raises(ValueError, match="10002073's alone"):
        harness.scenario("loop", 1, "crowd")  # the camps stand on Tree Dungeon's platforms
    assert harness.main(["run", "loop", "--preset", "crowd", "--seeds", "1"]) == 2


def test_a_short_harness_run_on_tree_dungeon():
    tree_map()
    from gamelab.sim import harness
    out = harness.run_one("loop", 1, "crowd", minutes=0.5, map_id=TREE, world=("zero-walls",))
    assert out["map"] == TREE and out["world"] == ["zero-walls"] and "left_map_at" in out and out["coverage"]


def test_the_harness_has_the_hunt_loop_with_low_hp_at_half_to_compare_against():
    from types import SimpleNamespace
    from gamelab.sim import harness
    made = []

    class Stop(Exception):
        pass

    def hunter(low_hp, potion_hp):
        made.append((low_hp, potion_hp))
        raise Stop
    sim = SimpleNamespace(hunter=hunter)
    for policy, low in (("loop", 0.3), ("loop-hp50", 0.5)):
        with pytest.raises(Stop):
            harness.POLICIES[policy](sim)
        assert made[-1] == (low, 0.45)
