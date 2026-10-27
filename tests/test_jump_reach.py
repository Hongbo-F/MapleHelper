"""What her attacks reach from the ground and from her jumps (attack.reach.shots, Coverage), on synthetic maps: her
arc over time for each kind of jump (motion.physics), the attack's pick from where she is when it starts (the area
about her feet, and the star's search's windows widening with the distance: reach.air_takes), and the world's
attack from a jump going by the same model (sim.world.AIR_PICK 'widening')."""

from pathlib import Path

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


def test_the_same_height_farther_off_is_hit_lower_in_the_jump_or_from_the_ground(monkeypatch):
    # "After the jump you can hit monsters that is far away horizontally, remember the widening vertical search as
    # horizontal scale out": the windows reach 5 higher every 20 ahead, so 100 up is in reach lower in the jump
    # farther off: 290 off with the attack keyed at once, and from the ground there too, the search's windows reaching
    # it from 271 (a pig: UP_BY_SEARCH, on by the user's choice, 2026-10-25). With the switch off, her decisions keep
    # to SHOT_UP (60) above her from the ground, and 290 off is a jump's only.
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 600)
    keys = {}
    for dx in (80, 160, 220, 290):
        m = pig(400 + dx, 400, up.id)
        c = cover(LEDGE, floor, 400, [m])
        assert c.ground(m) == (dx == 290), dx
        keys[dx] = c.best(m, ("up",)).key_ms
    assert keys[80] > keys[160] > keys[220] >= keys[290] == reach.KEY_FIRST_MS, keys
    assert reach.can_shoot(290, -100) and not reach.can_shoot(250, -100)
    far = pig(400 + 290, 400, up.id)
    monkeypatch.setattr(reach, "UP_BY_SEARCH", False)
    c = cover(LEDGE, floor, 400, [far])
    assert not c.ground(far) and c.best(far, ("up",)).key_ms == reach.KEY_FIRST_MS


def test_the_ground_reaches_above_60_by_the_search_and_the_switch_caps_it(monkeypatch):
    # The user, 2026-10-25, "Turn it on now": their own play killed Green Mushrooms 69-100 above her from the ground,
    # where the client's top (44) has the search pick them. The search's windows alone, then: a pig 100 above her from
    # 271 off, 120 from 351 (the last bands; 124 at most, at 385-400 ahead). Off (UP_BY_SEARCH False, should the
    # top-60 probe find the logs' 21), SHOT_UP's cap: 60 above her at most.
    assert reach.UP_BY_SEARCH
    assert reach.can_shoot(290, -100) and not reach.can_shoot(260, -100) and reach.can_shoot(200, -60)
    assert reach.can_shoot(360, -120) and not reach.can_shoot(340, -120)
    monkeypatch.setattr(reach, "UP_BY_SEARCH", False)
    assert not reach.can_shoot(290, -100) and not reach.can_shoot(360, -120) and reach.can_shoot(200, -60)


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


def test_a_turning_jumps_attack_at_one_on_her_platform_starts_at_its_kinds_turn_key(monkeypatch):
    """reach.shots' jumps that turn her in the air ('up' turning, 'back') start at reach.turn_key_ms of the target's
    kind. By the star's search from her feet in the air (reach.AIR_MODEL 'search', from 2026-10-27) every kind's is
    120 ms, and a pig on her level 40 or 100 off is picked by no turning jump in place until her way down (450 ms in
    and later: her feet 55 up at 120 ms, the pig's top 17 under them); a jump back at one 150 off picks it at 120. By the
    area ('widening'): a pig (38 tall at least) at 105 ms, the latest that still picks it, its arrow 45 ms before; never
    sooner than that for any kind. With 120 for every kind (from 2026-10-26 to the review of that day, TURN_KEY_MS), a
    pig on her platform had no turning shot until her way down, 430 ms in."""
    from gamelab.localization.monsters import MONSTER_TOP
    p = plat(LEDGE, 500, 200)
    for dx in (40, 100):
        turning = [s for s in reach.shots(LEDGE, p, 200.0, 500.0, [pig(200.0 + dx, 500.0, p.id)]) if s.turn]
        assert all(s.key_ms >= 435 for s in turning), dx
    assert any(s.kind == "back" and s.key_ms == 120 for s in reach.shots(LEDGE, p, 200.0, 500.0,
                                                                         [pig(350.0, 500.0, p.id)]))
    monkeypatch.setattr(reach, "AIR_MODEL", "widening")
    for dx in (40, 100):
        target = pig(200.0 + dx, 500.0, p.id)
        turning = [s for s in reach.shots(LEDGE, p, 200.0, 500.0, [target]) if s.turn]
        assert turning and min(s.key_ms for s in turning) == reach.turn_key_ms(MONSTER_TOP["pig"]) == 105, dx
        assert all(s.key_ms >= 105 for s in turning)
        with monkeypatch.context() as mp:
            mp.setattr(reach, "turn_key_ms", lambda top: 120)
            assert min(s.key_ms for s in reach.shots(LEDGE, p, 200.0, 500.0, [target]) if s.turn) > 400, dx


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


# --- what a live run's telemetry says of them ----------------------------------------------------------------------

def test_a_runs_jump_records_are_counted_with_what_the_looks_after_them_saw():
    from gamelab.analysis import jumps

    def look(t, bar, x=150.0):
        return {"t": t, "topic": "scene", "her": {"x": 50.0},
                "monsters": [{"kind": "green-mushroom", "platform": 22, "x": x, "bar": bar}]}

    def jump(t, kind="above", how="up"):
        return {"t": t, "topic": "jump", "kind": kind, "jump": how, "key_ms": 155, "target": "green-mushroom",
                "platform": 22, "x": 50.0, "dx": 100.0, "up": 100.0}
    rows = [look(99.9, None), jump(100.0), look(100.8, 0.6),                          # its bar shows: a hit
            look(199.9, 0.6), jump(200.0, "passing", "forward"), look(200.5, 0.6),
            {"t": 200.9, "topic": "exp", "kinds": ["green-mushroom"]},                 # its EXP: a kill
            look(299.9, 0.4), jump(300.0), look(300.8, 0.4, x=400.0)]                 # nothing seen
    got = jumps.outcomes(rows)
    assert [j["outcome"] for j in got] == ["hit", "kill", "nothing seen"]
    text = jumps.table(got, minutes=5.0)
    assert "above    up          2  0.40" in text and "passing  forward     1" in text


def test_a_jumps_kill_is_a_gain_soon_after_its_attack_key_not_one_after_a_later_throws():
    """analysis.jumps (KILL_AFTER): a kill's EXP gain shows 0.15-0.46 s after the key of the throw that made it (live on
    2026-10-25): a gain 0.3 s after a jump's attack key (its record's time and key_ms) is its kill; one 1.2 s after it
    but 0.3 s after a ground throw's key at the same kind is that throw's, not the jump's (until 2026-10-26 any gain
    within 1.6 s of the jump record was, the hunter's own lagging ones)."""
    from gamelab.analysis import jumps

    def jump(t):
        return {"t": t, "topic": "jump", "kind": "above", "jump": "up", "key_ms": 100, "target": "horny-mushroom",
                "platform": 23, "x": 0.0, "dx": 100.0, "up": 60.0}

    def attack(t, kind="horny-mushroom"):
        return {"t": t, "topic": "attack", "key": "lucky_seven", "kind": kind}

    gain = lambda t: {"t": t, "topic": "exp", "gain": 41, "kinds": ["horny-mushroom"]}
    rows = [jump(10.0), attack(10.002), attack(10.95), jump(20.0), attack(20.002), attack(20.9)]
    got = jumps.outcomes(rows, gains=[gain(10.4), gain(21.3)])
    assert [j["outcome"] for j in got] == ["kill", "nothing seen"] and got[0]["key"] == pytest.approx(10.1)
    assert jumps.kills_of([gain(21.3)], jumps.keys_of(rows)) == {20.9: 21.3}  # the ground throw's
    assert jumps.outcomes(rows, gains=[gain(10.05)])[0]["outcome"] == "nothing seen"  # (too soon: not its)


def test_the_jumps_of_the_hunt_of_2026_10_25_2343_as_the_perception_services_records_have_them():
    """analysis.jumps on the supervised hunt of 2026-10-25 23:43 (its six jump records and the records about them, the
    hunter's and the perception service's: tests/fixtures/jumps_2026-10-25), as the analysis of that night checked them
    jump by jump (D:/MapleHelper-scratch/analysis-1025/checked.txt): 'reach' nothing (the turn failed); 'above' a hit
    (#119's bar 0.22 1.0 s on), nothing (#119 dying already) and a hit (#156's bar 0.46 0.8 s on, in the service's
    scenes only: by the hunter's looks 'nothing seen', and a kill by master's tool, a ground throw's gain 1.4 s on);
    'passing' two kills (gains 0.37 and 0.34 s after their keys)."""
    from gamelab.analysis import jumps
    path = Path(__file__).parent / "fixtures" / "jumps_2026-10-25" / "20261025-234327-tour.jsonl"
    rows = jumps._rows(path)
    gains, scenes = jumps.service_records(path, rows)
    got = [(j["kind"], j["outcome"]) for j in jumps.outcomes(rows, gains, scenes)]
    assert got == [("reach", "nothing seen"), ("above", "hit"), ("above", "nothing seen"), ("above", "hit"),
                   ("passing", "kill"), ("passing", "kill")]
    assert [j["outcome"] for j in jumps.outcomes(rows, gains)][3] == "nothing seen"  # (the hunter's looks)


# --- the stance takes them -------------------------------------------------------------------------------------------

def test_the_stance_jump_throws_at_a_monster_above_with_the_attack_when_it_surely_picks_it():
    # A pig 100 above her 150 off: out of reach from the ground (SHOT_UP), in reach from a jump in place 95 ms on (her
    # feet 34 up), her body clear of it however it walks meanwhile (125 units/s either way: _arc_meets).
    from maplehelper.engine import LEFT, RIGHT
    from gamelab.attack.stance import plan_of, stance
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 550)
    m = pig(550, 400, up.id)
    for facing in (RIGHT, LEFT):
        act = stance(400, 500, floor, [m], facing=facing, jumps=lambda: cover(LEDGE, floor, 400, [m]))
        plan = plan_of(act)
        assert act[:2] == ("jump_throw", RIGHT) and plan.way == 0 and plan.why == "above", act
        assert plan.shot.window[0] <= plan.key_ms <= plan.shot.window[1] and 20 <= plan.shot.rise <= 60
        assert plan.shot.turn == (facing == LEFT)
    # without what her jumps reach: as before, nothing she can do from here
    assert stance(400, 500, floor, [m], facing=RIGHT)[0] in ("wait", "approach", "back")
    # and in the world: the pig hurt, her body never touched
    w = still(world(LEDGE, floor.id, 400.0, monsters=[("Pig", up.id, 550.0)], facing="right"))
    w.invulnerable_ms = 0
    plan = plan_of(stance(400, 500, floor, [m], facing=RIGHT, jumps=lambda: cover(LEDGE, floor, 400, [m])))
    tap(w, 0.0, "alt", 0.04)
    tap(w, plan.key_ms / 1000, "f", 0.04)
    w.advance(1.5)
    assert w.monsters[0].hp < w.monsters[0].hp_max and w.touches == 0, w.log[-5:]


@pytest.mark.parametrize("dx", [0, 30, 50, 60])
def test_the_stance_never_jumps_up_into_a_monster_overhead(dx):
    # Right under a pig 100 above her (her box 22 either side, its touch box 34, and as far as it can walk by then),
    # a jump in place meets it before the star leaves: in the world it touched her 0.16 s after the jump key from 0 to
    # 30 off (review, 2026-10-25). The query has the shot (near the top of the jump); the stance doesn't take it.
    from maplehelper.engine import RIGHT
    from gamelab.attack import stance as stance_mod
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 400 + dx)
    m = pig(400 + dx, 400, up.id)
    c = cover(LEDGE, floor, 400, [m])
    ups = [s for s in c.of(m) if s.kind == "up"]
    if dx >= 50:
        assert ups  # (right overhead, at 0 or 30, its far side short of the search's first band from her feet then)
    assert all(stance_mod._arc_meets(400, 500, floor, s, [m], None) for s in ups)
    act = stance_mod.stance(400, 500, floor, [m], facing=RIGHT, jumps=lambda: c)
    plan = stance_mod.plan_of(act)
    assert plan is None or plan.shot.kind != "up", act
    # one on her own platform is the fight's (her keeping off): a jump in place doesn't take her any nearer it
    mine = pig(400 + dx + 80, 500, floor.id)
    assert not any(stance_mod._arc_meets(400, 500, floor, s, [mine], None) for s in ups)


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


# --- the stance's safety checks on a jump forward or back ----------------------------------------------------------

def _passing():
    """She stands at 400 on LEDGE's floor travelling right, a pig on the floor 300 on (in reach from the ground): the
    jump forward in passing (landing about 84 on) and the shot it makes."""
    from gamelab.attack.stance import plan_of, stance
    from maplehelper.engine import RIGHT
    floor = plat(LEDGE, 500, 400)
    m = pig(700, 500, floor.id)
    c = cover(LEDGE, floor, 400, [m])
    act = stance(400, 500, floor, [m], facing=RIGHT, jumps=lambda: c, toward=1)
    assert act[0] == "jump_throw" and plan_of(act).way == 1
    return floor, m, c, plan_of(act).shot


def _shot_of(floor, m, cover_, lo=25.0, hi=1475.0, around=None, danger=None, goal=None):
    """stance._jump_shot's jump forward in passing (as _passing's), where she keeps to `lo`..`hi`."""
    from gamelab.attack.stance import _jump_shot
    from maplehelper.engine import RIGHT
    return _jump_shot(400, 500, floor, [m], cover_, lo, hi, RIGHT, [m] if around is None else around, danger, None,
                      toward=1, kinds=("forward",), why="passing", goal=goal)


def test_a_jump_forward_is_taken_only_if_however_it_goes_it_lands_where_she_keeps_to():
    floor, m, c, sh = _passing()
    near, far = sh.land_span
    assert near < sh.land_x < far and sh.lands_sure  # 12 units early or late, slower, the kick short: all on the floor
    assert _shot_of(floor, m, c, hi=far) is not None
    assert _shot_of(floor, m, c, hi=far - 1) is None  # her keeping off the platform's end: lo..hi
    assert _shot_of(floor, m, c, lo=near + 1) is None


def test_a_jump_forward_whose_short_kick_lands_on_the_platform_above_is_not_taken():
    # Under a platform 60 up whose end is at 450: the kick at full speed (149) lands her on the floor at 484, at 119
    # (the arrow seen at the jump's own step) up on the platform (review, 2026-10-25), as physics.sure_landing finds
    # of the same jump; with the platform's end at 440 every way lands on the floor, and she takes it.
    from gamelab.attack.stance import plan_of, stance
    from maplehelper.engine import RIGHT
    for end, sure in ((450, False), (440, True)):
        model = build({"1": [seg(1, 0, 500, 1500, 500)], "2": [seg(2, 200, 440, end, 440)]})
        floor = plat(model, 500, 600)
        m = pig(780, 500, floor.id)
        c = reach.coverage(model, floor, 400.0, 500, [m])
        fwd = [s for s in c.of(m) if s.kind == "forward"]
        assert fwd and all(s.land is floor and s.lands_sure == sure for s in fwd), end
        j = physics.jump_of("forward", 1)
        assert physics.sure_landing(model, 400 + j.run, 500, j.vx, -physics.JUMP_SPEED, floor,
                                    group=floor.group) == sure
        act = stance(400, 500, floor, [m], facing=RIGHT, jumps=lambda: c, toward=1)
        assert (act[0] == "jump_throw" and plan_of(act).way == 1) == sure, (end, act)


def test_a_jump_forward_that_passes_into_a_monster_above_its_arc_is_not_taken():
    # A block 60 up from 480 to 530 that her arc comes down under (her feet 13 up at 480, every way she may go under
    # its floor there, her head over it): with a pig on it her box meets the pig's on the way (_arc_meets), so no jump
    # forward; without, she takes it.
    from gamelab.attack import stance as stance_mod
    from maplehelper.engine import RIGHT
    model = build({"1": [seg(1, 0, 500, 1500, 500)], "2": [seg(2, 480, 440, 530, 440)]})
    floor, block = plat(model, 500, 400), plat(model, 440, 505)
    m = pig(700, 500, floor.id)
    on_block = pig(505, 440, block.id)
    c = reach.coverage(model, floor, 400.0, 500, [m])  # (with the block's pig about, the air's pick takes it first)
    fwd = [s for s in c.of(m) if s.kind == "forward"]
    assert fwd and all(s.land is floor and s.lands_sure for s in fwd)
    assert all(stance_mod._arc_meets(400, 500, floor, s, [m, on_block], None) for s in fwd)
    assert not any(stance_mod._arc_meets(400, 500, floor, s, [m], None) for s in fwd)
    act = stance_mod.stance(400, 500, floor, [m], facing=RIGHT, jumps=lambda: c, toward=1, around=[m, on_block])
    assert not (act[0] == "jump_throw" and stance_mod.plan_of(act) and stance_mod.plan_of(act).way), act
    act = stance_mod.stance(400, 500, floor, [m], facing=RIGHT, jumps=lambda: c, toward=1)
    assert act[0] == "jump_throw" and stance_mod.plan_of(act).way == 1, act
    # a monster on her platform where she'd land: _into
    floor, m, c, sh = _passing()
    assert _shot_of(floor, m, c, around=[m, pig(sh.land_x + 20, 500, floor.id)]) is None


def test_a_jump_forward_lands_no_more_dangerous_than_jump_danger_over_where_she_stands():
    from gamelab.attack import stance as stance_mod
    floor, m, c, sh = _passing()

    class Danger:  # (attack.danger's interface the stance reads)
        def __init__(self, landing):
            self.landing = landing

        def keep_off(self, kind):
            return 0.0

        def at(self, pid, x, around):
            return self.landing if x > 450 else 0.0

    assert _shot_of(floor, m, c, danger=Danger(stance_mod.JUMP_DANGER)) is not None
    assert _shot_of(floor, m, c, danger=Danger(stance_mod.JUMP_DANGER + 0.001)) is None


def test_in_passing_a_jump_forward_lands_no_more_than_pass_slack_past_where_she_is_going():
    from gamelab.attack import stance as stance_mod
    floor, m, c, sh = _passing()
    far = sh.land_span[1]
    assert _shot_of(floor, m, c, goal=far - stance_mod.PASS_SLACK) is not None
    assert _shot_of(floor, m, c, goal=far - stance_mod.PASS_SLACK - 1) is None  # (a rope's foot 60 on: she'd walk back)
    act = stance_mod.stance(400, 500, floor, [m], facing="right", jumps=lambda: c, toward=1, goal=460.0)
    assert act == ("throw", "right"), act


def test_the_hunt_travels_toward_its_objectives_place_or_where_her_route_leaves_her_platform():
    from types import SimpleNamespace
    from gamelab.behavior.hunt import MIN_MOVE, Hunt
    h = object.__new__(Hunt)
    s = SimpleNamespace(where=SimpleNamespace(x=400.0), platform=SimpleNamespace(id=1))
    asked = []

    def leaves_at(here, x, dst, dst_x=None):
        asked.append((here.id, x, dst, dst_x))
        return 250.0
    h.nav = SimpleNamespace(leaves_at=leaves_at)
    h.obj = None
    assert h._toward(s) is None
    h.obj = SimpleNamespace(platform=1, x=600.0)
    assert h._toward(s) == 600.0
    h.obj = SimpleNamespace(platform=1, x=400.0 + MIN_MOVE - 1)
    assert h._toward(s) is None  # (too near to be going anywhere)
    h.obj = SimpleNamespace(platform=1, x=None)
    assert h._toward(s) is None
    h.obj = SimpleNamespace(platform=7, x=90.0)
    assert h._toward(s) == 250.0 and asked == [(1, 400.0, 7, 90.0)]
    h.nav = SimpleNamespace(leaves_at=lambda here, x, dst, dst_x=None: None)
    assert h._toward(s) is None


def test_the_navigator_says_where_the_quickest_way_leaves_her_platform():
    from gamelab.control.navigator import Navigator
    from gamelab.motion.route import Planner
    steps = build({"1": [seg(1, 0, 500, 1500, 500)], "2": [seg(2, 300, 440, 700, 440)]})
    floor, block = plat(steps, 500, 100), plat(steps, 440, 400)
    nav = object.__new__(Navigator)
    nav.planner = Planner(steps)
    assert nav.leaves_at(floor, 100.0, floor.id, 160.0) == 160.0
    first = nav.planner.plan(floor.id, 100.0, block.id)[0]
    at = nav.leaves_at(floor, 100.0, block.id, 500.0)
    assert at == (first.takeoff if first.takeoff is not None else first.x) and 100.0 <= at <= 300.0
    alone = build({"1": [seg(1, 0, 500, 1500, 500)], "2": [seg(2, 300, 200, 700, 200)]})  # 300 up: no way there
    nav.planner = Planner(alone)
    assert nav.leaves_at(plat(alone, 500, 100), 100.0, plat(alone, 200, 400).id) is None


@pytest.mark.parametrize("dx,hits", [(50, False), (150, True)])
def test_the_hunt_counts_in_reach_from_a_jump_only_what_the_stance_would_jump_at(dx, hits):
    # A pig 100 above her: right overhead the query has a shot near the top of the jump, but her body would meet it
    # (the stance won't take it), so the hunt loop and grinding don't stay to shoot it (Hunt._jump_hits); 150 off,
    # in place from early in the jump, they do.
    from types import SimpleNamespace
    from maplehelper.engine import RIGHT
    from gamelab.behavior.hunt import Hunt
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 400 + dx)
    m = pig(400 + dx, 400, up.id)
    assert any(s.kind == "up" for s in cover(LEDGE, floor, 400, [m]).of(m))
    h = object.__new__(Hunt)
    h.h = SimpleNamespace(actions=SimpleNamespace(facing=RIGHT), danger=None)
    h.model = LEDGE
    s = SimpleNamespace(where=SimpleNamespace(x=400.0, y=500.0), platform=floor, monsters=[m])
    assert h._jump_hits(s, m) == hits and h._hits(s, m) == hits


def test_her_kill_not_seen_dead_by_her_counts_for_her_body_alone():
    """A monster in the stance's `touch` (behavior.fight.TOUCH_NEAR: one her kill is bound to with no death seen yet,
    live now and then, by her) is kept clear of as one `around` is: no jump forward that lands into it. But it's no
    target, and doesn't take her stars: from the ground she throws at the one beyond it."""
    from gamelab.attack.stance import stance
    from maplehelper.engine import RIGHT
    floor, m, c, sh = _passing()
    by = pig(sh.land_x + 20, 500, floor.id)
    assert stance(400, 500, floor, [m], facing=RIGHT, jumps=lambda: c, toward=1)[0] == "jump_throw"
    assert stance(400, 500, floor, [m], facing=RIGHT, jumps=lambda: c, toward=1, touch=[by]) == ("throw", RIGHT)
