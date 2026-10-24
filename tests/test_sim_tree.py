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
