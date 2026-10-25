"""What her attacks reach from the ground and from her jumps (attack.reach.shots, Coverage), on synthetic maps: her
arc over time for each kind of jump (motion.physics), the attack's pick from where she is when it starts (the area
about her feet, and the star's search's windows widening with the distance: reach.air_takes), and the world's
attack from a jump going by the same model (sim.world.AIR_PICK 'widening')."""

import pytest

from gamelab.attack import reach
from gamelab.localization import mapdata
from gamelab.localization.monsters import Monster
from gamelab.motion import physics
from gamelab.perception.mobs import Mob
from gamelab.sim import world as world_mod
from gamelab.sim.world import World

from test_sim import KEYS, still, tap


def seg(fid, x1, y1, x2, y2, prev=0, next=0):
    return str(fid), {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "prev": prev, "next": next}


def build(groups: dict) -> mapdata.MapModel:
    """A map of these foothold groups ({group: [seg, ...]}), every one in layer 1."""
    tree = {"info": {"VRLeft": -200, "VRTop": -400, "VRRight": 1600, "VRBottom": 700},
            "foothold": {"1": {g: dict(segs) for g, segs in groups.items()}},
            "miniMap": {"width": 1000, "height": 600, "centerX": 100, "centerY": 0, "mag": 4}}
    return mapdata.parse(1, tree, "Test", {}, {"1": "Pig", "2": "Snail"}, {"1": 3, "2": 2})


#   y 400            [up: 300 ......................... 1100]   (a floating block: its own group, no walls)
#   y 500  [floor: 0 ............................................... 1500]
LEDGE = build({"1": [seg(1, 0, 500, 1500, 500)], "2": [seg(2, 300, 400, 1100, 400)]})


def plat(model, y, x):
    return next(p for p in model.platforms if p.covers(x) and abs(p.y_at(x) - y) < 1)


def pig(x, y, platform, facing=0, pose=""):
    return Monster(Mob("pig", 0, 0, 30, 30, 0.9, facing=facing, pose=pose), x, y, platform)


def cover(model, p, x, monsters, **kw):
    return reach.coverage(model, p, x, p.y_at(x), monsters, **kw)


# --- her arc -------------------------------------------------------------------------------------------------------

def test_her_jump_peaks_77_up_0_2775_s_after_lift_off_and_lands_0_555_s_after_it():
    assert physics.APEX_S == pytest.approx(0.2775) and physics.JUMP_HEIGHT == pytest.approx(77.0, abs=0.01)
    assert physics.rise_after(physics.APEX_S) == pytest.approx(77.0, abs=0.01)
    assert physics.rise_after(physics.AIR_S) == pytest.approx(0.0, abs=1e-6)
    up, down = physics.lift_time(62)
    assert (up, down) == pytest.approx((0.155, 0.400), abs=0.001)
    assert physics.lift_time(78) is None
    # the jump key's lift-off comes JUMP_LAG after it: jump_rise counts from the key
    assert physics.jump_rise(physics.JUMP_LAG * 1000 + physics.APEX_S * 1000) == pytest.approx(77.0, abs=0.01)


def test_each_kind_of_jump_lands_where_the_engine_flies_it():
    floor = plat(LEDGE, 500, 100)
    x = 100
    land, lx, s = physics.jump_landing(LEDGE, floor, x, physics.jump_of("up"))
    assert land is floor and lx == pytest.approx(x, abs=0.5) and s == pytest.approx(physics.AIR_S, abs=0.02)
    fwd = physics.jump_of("forward", 1)
    assert fwd.vx == pytest.approx(physics.WALK_SPEED, abs=1)  # the kick: full speed (149) at lift-off
    land, lx, s = physics.jump_landing(LEDGE, floor, x, fwd)
    assert land is floor and lx - x == pytest.approx(84, abs=3)  # in game 82.7-85.1 (kick2_40_l, _r)
    land, lx, s = physics.jump_landing(LEDGE, floor, x + 300, physics.jump_of("back", -1))
    assert land is floor and x + 300 - lx == pytest.approx(physics.WALK_SPEED * physics.AIR_S + 14, abs=4)
    # where she is over the jump: the arc's top over where she lifted off, at its half-way along a jump forward
    px, py, air = physics.pose(LEDGE, floor, x, fwd, physics.JUMP_LAG + physics.APEX_S)
    assert air and 500 - py == pytest.approx(77.0, abs=0.01) and px - x == pytest.approx(42.5, abs=2)


def test_a_jump_onto_a_platform_above_lands_on_it_and_one_off_a_ledge_on_the_floor_below():
    steps = build({"1": [seg(1, 0, 500, 1500, 500)], "2": [seg(2, 300, 440, 700, 440)]})
    floor, block = plat(steps, 500, 250), plat(steps, 440, 400)
    land, lx, s = physics.jump_landing(steps, floor, 250, physics.jump_of("forward", 1))
    assert land is block and lx > 300  # 60 up: within her jump's 77
    land, lx, s = physics.jump_landing(steps, block, 680, physics.jump_of("forward", 1))
    assert land is floor and lx > 700 and s > physics.AIR_S  # off the block's end, down to the floor


# --- the airborne query ---------------------------------------------------------------------------------------------

def test_a_monster_100_up_and_close_is_hit_only_with_the_attack_near_the_top_of_the_jump():
    # The user, 2026-10-25: "monster 100 units up hit only near the top of the jump". 50 ahead (in the area, its far
    # side in the search's first band), its feet must be under 44 over hers: her feet 62 up (AIR_MARGIN to spare),
    # 155-400 ms after lift-off, and the attack starts about as long after lift-off as its key came after the jump
    # key. (A body farther off reaches into later bands, whose windows are higher: 80 ahead a pig's far side is in
    # the third, 54 over her feet, from 135 ms.)
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 450)
    m = pig(450, 400, up.id)
    c = cover(LEDGE, floor, 400, [m])
    assert not c.ground(m)
    ups = [s for s in c.of(m) if s.kind == "up" and not s.turn]
    assert ups, c.shots
    s = ups[0]
    assert 150 <= s.key_ms and s.window[1] <= 400 and s.rise >= 60
    # and no key outside that: each key tried, its pick
    for key in range(40, 480, 10):
        p = physics.pose(LEDGE, floor, 400, physics.jump_of("up"), key / 1000 + physics.JUMP_LAG)
        took = p[2] and reach.air_takes(50, 400 - p[1], 46, 34)
        assert took == (500 - p[1] > 56), key  # her feet over 56 up: the monster's under 44 over them


def test_the_same_height_farther_off_is_hit_lower_in_the_jump_or_from_the_ground():
    # "After the jump you can hit monsters that is far away horizontally, remember the widening vertical search as
    # horizontal scale out": the windows reach 5 higher every 20 ahead, so 100 up is in reach lower in the jump
    # farther off, and from the ground from 271 (a pig).
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 600)
    keys = {}
    for dx in (80, 160, 220):
        m = pig(400 + dx, 400, up.id)
        c = cover(LEDGE, floor, 400, [m])
        assert not c.ground(m)
        keys[dx] = c.best(m, ("up",)).key_ms
    assert keys[80] > keys[160] > keys[220], keys
    far = pig(400 + 290, 400, up.id)
    assert cover(LEDGE, floor, 400, [far]).ground(far)
    assert reach.can_shoot(290, -100) and not reach.can_shoot(250, -100)


def test_with_up_by_search_off_the_ground_is_capped_60_above_her(monkeypatch):
    monkeypatch.setattr(reach, "UP_BY_SEARCH", False)
    assert not reach.can_shoot(290, -100) and reach.can_shoot(200, -60)


def test_too_high_for_the_widest_window_and_the_jump_is_out_of_reach():
    # The last band's window reaches 124 over her feet (385-400 ahead), and her feet 77 up at the top: 201 over where
    # she stood. A block 210 up is out of reach, and one 125 up close (115 at most up close: 77 + 44 - AIR_MARGIN) of
    # the ground and a jump in place; a jump back carries her away from it, out to where the windows reach higher.
    high = build({"1": [seg(1, 0, 500, 1500, 500)], "2": [seg(2, 300, 290, 1100, 290)],
                  "3": [seg(3, 120, 375, 280, 375)]})
    floor = plat(high, 500, 100)
    far = pig(100 + 395, 290, plat(high, 290, 495).id)
    close = pig(100 + 50, 375, plat(high, 375, 150).id)
    c = cover(high, floor, 100, [far, close])
    assert not c.of(far) and not c.hits(close, ("ground", "up", "forward"))
    back = c.best(close, ("back",))
    assert back is not None and back.side == 1 and 100 - back.x > 40 and back.rise > 70
    # 190 up at 395 off: from the top of the jump the last band reaches it
    reachable = build({"1": [seg(1, 0, 500, 1500, 500)], "2": [seg(2, 300, 310, 1100, 310)]})
    floor = plat(reachable, 500, 100)
    m = pig(100 + 395, 310, plat(reachable, 310, 495).id)
    assert cover(reachable, floor, 100, [m]).hits(m, ("up",))


def test_a_jump_forward_reaches_one_past_a_gap():
    # She stands at her platform's end; a pig across a gap is 440 off, past a star's reach (its near side 406 off):
    # from the ground and in place, out of reach; a jump forward over the gap carries her on as the attack starts.
    gap = build({"1": [seg(1, 0, 500, 400, 500)], "2": [seg(2, 470, 500, 1300, 500)]})
    here, there = plat(gap, 500, 300), plat(gap, 500, 900)
    m = pig(395 + 440, 500, there.id)
    c = cover(gap, here, 395, [m])
    assert not c.ground(m) and not c.hits(m, ("up",))
    s = c.best(m, ("forward",))
    assert s is not None and s.side == 1 and s.land is there and s.x > 395 and s.land_x > 470


def test_a_wall_of_her_group_blocks_from_the_ground_and_from_the_jump():
    # Her floor's group has a wall at x 600 rising to y 250, facing her; a pig on a block beyond it, 100 up.
    walled = build({"1": [seg(1, 0, 500, 600, 500, 0, 2), seg(2, 600, 500, 600, 250, 1, 0)],
                    "2": [seg(3, 610, 400, 1000, 400)]})
    floor, block = plat(walled, 500, 300), plat(walled, 400, 700)
    m = pig(700, 400, block.id)
    c = cover(walled, floor, 550, [m])
    assert not c.of(m)
    # the same without the wall: in reach from the jump
    open_ = build({"1": [seg(1, 0, 500, 600, 500)], "2": [seg(3, 610, 400, 1000, 400)]})
    assert cover(open_, plat(open_, 500, 300), 550, [pig(700, 400, plat(open_, 400, 700).id)]).shots


def test_a_monster_walking_is_placed_where_it_will_be_when_the_attack_picks():
    # A pig on her floor 452 off (its near side past a star's 400) walking at her at 100 units/s: by the attack's
    # pick from a jump (0.1-0.5 s on) it has come 10-50 nearer, into reach; standing there, it's out of reach.
    floor = plat(LEDGE, 500, 400)
    m = pig(400 + 452, 500, floor.id)

    def toward(mo, t):
        return mo.x - 100 * t

    assert not cover(LEDGE, floor, 400, [m]).hits(m, ("ground", "up"))
    walking = cover(LEDGE, floor, 400, [m], where=toward)
    assert walking.hits(m, ("up",)) and not walking.ground(m)


# --- the world goes by the same model -------------------------------------------------------------------------------

def world(model, pid, x, **kw):
    kw.setdefault("monsters", None)
    return World(model, KEYS, (pid, x), **kw)


@pytest.mark.parametrize("dx", [80, 160])
def test_the_world_s_attack_from_a_jump_hits_where_the_query_says(dx):
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 400 + dx)
    m = pig(400 + dx, 400, up.id)
    s = cover(LEDGE, floor, 400, [m]).best(m, ("up",))
    lo, hi = s.window
    for key, hits in ((lo, True), (hi, True), (max(physics.JUMP_ATTACK_MS, lo - 60), False)):
        if key == lo - 60 and lo - 60 < physics.JUMP_ATTACK_MS:
            continue
        w = still(world(LEDGE, floor.id, 400.0, monsters=[("Pig", up.id, 400.0 + dx)], facing="right"))
        w.invulnerable_ms = 10_000
        tap(w, 0.0, "alt", 0.04)
        tap(w, key / 1000, "f", 0.04)
        w.advance(1.5)
        assert (w.monsters[0].hp < w.monsters[0].hp_max) == hits, (key, w.log[-5:])


def test_with_the_area_alone_the_world_misses_far_high_ones(monkeypatch):
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 650)
    monkeypatch.setattr(world_mod, "AIR_PICK", "area")
    w = still(world(LEDGE, floor.id, 400.0, monsters=[("Pig", up.id, 650.0)], facing="right"))
    w.invulnerable_ms = 10_000
    tap(w, 0.0, "alt", 0.04)
    tap(w, 0.05, "f", 0.04)  # her feet 50 up at the pick: the pig's 50 over them, over the area's 44
    w.advance(1.5)
    assert w.monsters[0].hp == w.monsters[0].hp_max


# --- the stance takes them -------------------------------------------------------------------------------------------

def test_the_stance_jump_throws_at_a_monster_above_with_the_attack_when_it_surely_picks_it():
    from maplehelper.engine import LEFT, RIGHT
    from gamelab.attack.stance import plan_of, stance
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 450)
    m = pig(450, 400, up.id)
    for facing in (RIGHT, LEFT):
        act = stance(400, 500, floor, [m], facing=facing, jumps=lambda: cover(LEDGE, floor, 400, [m]))
        plan = plan_of(act)
        assert act[:2] == ("jump_throw", RIGHT) and plan.way == 0 and plan.why == "above", act
        assert plan.shot.window[0] <= plan.key_ms <= plan.shot.window[1] and plan.shot.rise >= 60
        assert plan.shot.turn == (facing == LEFT)
    # without what her jumps reach: as before, nothing she can do from here
    assert stance(400, 500, floor, [m], facing=RIGHT)[0] in ("wait", "approach", "back")


def test_on_her_way_a_jump_forward_that_hits_one_goes_before_a_throw_from_the_ground():
    from maplehelper.engine import RIGHT
    from gamelab.attack.stance import plan_of, stance
    floor = plat(LEDGE, 500, 400)
    m = pig(400 + 300, 500, floor.id)
    jumps = lambda: cover(LEDGE, floor, 400, [m])
    assert stance(400, 500, floor, [m], facing=RIGHT, jumps=jumps) == ("throw", RIGHT)
    act = stance(400, 500, floor, [m], facing=RIGHT, jumps=jumps, toward=1)
    assert act[:2] == ("jump_throw", RIGHT) and plan_of(act).way == 1 and plan_of(act).why == "passing"
    assert stance(400, 500, floor, [m], facing=RIGHT, jumps=jumps, toward=-1) == ("throw", RIGHT)  # not her way
