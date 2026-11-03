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
    # to SHOT_UP (60) above her from the ground, and 290 off is a jump's only. Of the two keys (from 2026-11-01, the
    # user's: at once, at the top) the top's nearer (until then 175 ms at 80 off, 95 at 160, 60 at 220: the soonest
    # sure key from 40 on); turning in the air, its soonest (120: 52 up) from 160 off.
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 600)
    keys, turning = {}, {}
    for dx in (80, 160, 220, 290):
        m = pig(400 + dx, 400, up.id)
        c = cover(LEDGE, floor, 400, [m])
        assert c.ground(m) == (dx == 290), dx
        keys[dx] = min(s.key_ms for s in c.of(m) if s.kind == "up" and not s.turn)
        turning[dx] = min(s.key_ms for s in c.of(m) if s.kind == "up" and s.turn)
    assert keys == {80: reach.TOP_KEY_MS, 160: reach.TOP_KEY_MS, 220: reach.TOP_KEY_MS, 290: reach.KEY_FIRST_MS}
    assert turning == {80: reach.TOP_KEY_MS, 160: reach.TURN_KEY_MS, 220: reach.TURN_KEY_MS, 290: reach.TURN_KEY_MS}
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
    120 ms, and a pig on her level 40 or 100 off is picked by no turning jump in place (her feet 55 up at 120 ms, the
    pig's top 17 under them; at the top, 77: until 2026-11-01, when every key to 480 was tried, from 450 ms in on her
    way down); a jump back at one 150 off picks it at 120. By the area ('widening'): a pig (38 tall at least) at 105
    ms, the latest that still picks it, its arrow 45 ms before; never sooner than that for any kind. With 120 for every
    kind (from 2026-10-26 to the review of that day, TURN_KEY_MS), a pig on her platform had no turning shot (until
    2026-11-01: none until her way down, 430 ms in)."""
    from gamelab.localization.monsters import MONSTER_TOP
    p = plat(LEDGE, 500, 200)
    for dx in (40, 100):
        assert not [s for s in reach.shots(LEDGE, p, 200.0, 500.0, [pig(200.0 + dx, 500.0, p.id)]) if s.turn], dx
    assert any(s.kind == "back" and s.key_ms == 120 for s in reach.shots(LEDGE, p, 200.0, 500.0,
                                                                         [pig(350.0, 500.0, p.id)]))
    monkeypatch.setattr(reach, "AIR_MODEL", "widening")
    for dx in (40, 100):
        target = pig(200.0 + dx, 500.0, p.id)
        turning = [s for s in reach.shots(LEDGE, p, 200.0, 500.0, [target]) if s.turn]
        assert turning and min(s.key_ms for s in turning) == reach.turn_key_ms(MONSTER_TOP["pig"]) == 105, dx
        assert all(s.key_ms == 105 and s.window[0] == 105 for s in turning)  # (never sooner: its jitter later only)
        with monkeypatch.context() as mp:
            mp.setattr(reach, "turn_key_ms", lambda top: 120)
            assert not [s for s in reach.shots(LEDGE, p, 200.0, 500.0, [target]) if s.turn], dx


def test_each_jump_tries_two_attack_keys_at_once_and_at_its_top(monkeypatch):
    """The user (2026-11-01): "You only need two jump attacks: one with jump and immediately attack, another with jump
    and attack at mid air peak." reach.shots tries each jump and side at its soonest key (KEY_FIRST_MS; turning in the
    air, TURN_KEY_MS) and the top's (TOP_KEY_MS: 278, the attack starting physics.APEX_S after lift-off, her feet
    JUMP_HEIGHT up) and nowhere else but the slack and the jitter's room about them (until then every key from the
    soonest to 480, 5 ms apart); every shot at one of them, its window within that key's jitter."""
    assert reach.TOP_KEY_MS == round(physics.APEX_S * 1000) == 278
    assert physics.jump_rise(reach.TOP_KEY_MS + physics.JUMP_LAG * 1000) == pytest.approx(physics.JUMP_HEIGHT,
                                                                                         abs=0.01)
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 600)
    ms = [pig(560, 400, up.id), pig(480, 500, floor.id), pig(250, 400, up.id), pig(700, 400, up.id)]
    asked = set()
    real = physics.pose

    def pose(model, platform, x, j, t, landing=None):
        asked.add(round((t - physics.JUMP_LAG) * 1000))  # (the key, or 11 ms later: ATTACK_DECIDE_LAG_MS)
        return real(model, platform, x, j, t, landing)
    monkeypatch.setattr(physics, "pose", pose)
    got = reach.shots(LEDGE, floor, 400.0, 500.0, ms)
    monkeypatch.undo()
    reach_of = {reach.KEY_FIRST_MS: (0, reach.JITTER_FIRST_MS), reach.TURN_KEY_MS: (0, reach.JITTER_FIRST_MS),
                reach.TOP_KEY_MS: (reach.JITTER_TOP_MS, reach.JITTER_TOP_MS)}
    slack, decide = reach.KEY_SLACK_MS, physics.ATTACK_DECIDE_LAG_MS
    near = lambda k: any(base - lo - slack <= k <= base + hi + slack + decide for base, (lo, hi) in reach_of.items())
    assert asked and all(near(k) for k in asked), sorted(asked)
    assert {reach.KEY_FIRST_MS, reach.TURN_KEY_MS, reach.TOP_KEY_MS} <= asked
    jumps = [sh for sh in got if sh.jump is not None]
    assert {sh.key_ms for sh in jumps} == {reach.KEY_FIRST_MS, reach.TURN_KEY_MS, reach.TOP_KEY_MS}, jumps
    for sh in jumps:
        lo, hi = reach_of[sh.key_ms]
        assert sh.key_ms - lo <= sh.window[0] <= sh.key_ms <= sh.window[1] <= sh.key_ms + hi, sh.text()
        assert (sh.key_ms == reach.TURN_KEY_MS) <= sh.turn  # (the turn's soonest only turning)


def test_a_jump_shots_window_is_its_jitters_room_every_key_in_it_sure():
    """Shot.window (from 2026-11-01): the keys its jitter may key it at (behavior.fight.jittered), each picking the
    target surely: the top's JITTER_TOP_MS either side of it at a pig 100 above her (the world's attack hits it keyed
    anywhere in it), the soonest's up to JITTER_FIRST_MS later, never sooner; narrowed where the pick isn't sure all
    over that, to the key alone for a pig on her level 60 off (sure at 40 ms, not keyed 5 ms later: its window then,
    as the old scan's, 40-40): kept, with no jitter, rather than dropped."""
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 480)
    m = pig(480, 400, up.id)
    top = min((s for s in cover(LEDGE, floor, 400, [m]).of(m) if s.kind == "up" and not s.turn),
              key=lambda s: s.key_ms)
    assert top.window == (reach.TOP_KEY_MS - reach.JITTER_TOP_MS, reach.TOP_KEY_MS + reach.JITTER_TOP_MS)
    for key in range(top.window[0], top.window[1] + 1, 4):
        w = still(world(LEDGE, floor.id, 400.0, monsters=[("Pig", up.id, 480.0)], facing="right"))
        w.invulnerable_ms = 10_000
        tap(w, 0.0, "alt", 0.04)
        tap(w, key / 1000, "f", 0.04)
        w.advance(1.5)
        assert w.monsters[0].hp < w.monsters[0].hp_max, (key, w.log[-5:])
    level = pig(460, 500, floor.id)
    first = [s for s in cover(LEDGE, floor, 400, [level]).of(level) if s.kind == "up" and not s.turn]
    assert [(s.key_ms, s.window) for s in first] == [(reach.KEY_FIRST_MS, (reach.KEY_FIRST_MS, reach.KEY_FIRST_MS))]
    far = pig(550, 480, floor.id)  # (20 up, 150 off: sure all over the soonest's room)
    first = [s for s in cover(LEDGE, floor, 400, [far]).of(far) if s.kind == "up" and not s.turn]
    assert [s.window for s in first] == [(reach.KEY_FIRST_MS, reach.KEY_FIRST_MS + reach.JITTER_FIRST_MS)]


def test_her_jump_attacks_key_is_its_shots_jittered_within_its_window():
    """behavior.fight.jittered: a jump attack's key drawn as it's carried out, uniform to the ms over its shot's window
    (the top's 258-298, the soonest's 40-55: never sooner than its key), from the hunter's own generator (`jitter`:
    seeded, the same draws again); none without one (a test's hunter) or a shot (the stance's own timing)."""
    import random
    from types import SimpleNamespace
    from maplehelper.engine import RIGHT
    from gamelab.attack.stance import JumpPlan, plan_of
    from gamelab.behavior.fight import jittered
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 560)
    m, level = pig(560, 400, up.id), pig(550, 480, floor.id)
    top = min((s for s in cover(LEDGE, floor, 400, [m]).of(m) if s.kind == "up" and not s.turn), key=lambda s: s.key_ms)
    soon = min((s for s in cover(LEDGE, floor, 400, [level]).of(level) if s.kind == "up" and not s.turn),
               key=lambda s: s.key_ms)
    for sh, lo, hi in ((top, 258, 298), (soon, 40, 55)):
        assert sh.window == (lo, hi)
        act = ("jump_throw", RIGHT, JumpPlan(sh.key_ms, 0, True, "above", sh))
        h = SimpleNamespace(jitter=random.Random(7))
        keys = [plan_of(jittered(h, act)).key_ms for _ in range(3000)]
        assert min(keys) == lo and max(keys) == hi and set(keys) == set(range(lo, hi + 1))
        assert abs(sum(keys) / len(keys) - (lo + hi) / 2) < 1.0
        again = SimpleNamespace(jitter=random.Random(7))
        assert [plan_of(jittered(again, act)).key_ms for _ in range(3000)] == keys
        assert jittered(SimpleNamespace(), act) is act and jittered(SimpleNamespace(jitter=None), act) is act
        kept = jittered(h, act)
        assert kept[:2] == act[:2] and plan_of(kept).shot is sh and plan_of(kept).way == 0
    plain = ("jump_throw", RIGHT, JumpPlan())
    assert jittered(SimpleNamespace(jitter=random.Random(1)), plain) is plain


def test_a_jump_attack_is_keyed_as_jittered_and_its_records_say_so(monkeypatch):
    """fight.throw_act: the key plan has the attack key at the jittered key after the jump key; the 'jump' record has
    its base key, jitter, the key sent, its window, the turn, her rise then (the arc's at the key sent), and `n`; the
    'jump_keys' record after its keys how long after the jump key the player pressed the attack key (Actions.key_gap:
    its answer's 'sent'), and that they went."""
    import random
    from types import SimpleNamespace
    from collections import Counter
    from maplehelper.engine import RIGHT
    from gamelab import telemetry
    from gamelab.attack.stance import JumpPlan
    from gamelab.behavior import fight
    from gamelab.control import actions as actions_mod
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 560)
    m = pig(560, 400, up.id)
    sh = min((s for s in cover(LEDGE, floor, 400, [m]).of(m) if s.kind == "up" and not s.turn), key=lambda s: s.key_ms)
    plans = []

    def submit(events, **kw):  # the player: each key pressed 0.4 ms late, the attack's 1.6 ms
        plans.append(events)
        return {"sent": [[e[0] + (2.0 if e[1] == "f" else 0.4), *e[1:]] for e in events], "problems": []}
    monkeypatch.setattr(actions_mod.client, "submit", submit)
    key = lambda name: SimpleNamespace(name=name)
    a = actions_mod.Actions(SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), attack=key("f"),
                                            lucky_seven=None, pick_up=key("z"), hp_potion=None, chair=None))
    a.facing = RIGHT
    lines = []
    h = SimpleNamespace(actions=a, jitter=random.Random(5), log=lines.append, jump_counts=Counter(), danger=None)
    s = SimpleNamespace(where=SimpleNamespace(x=400.0, y=500.0), platform=floor, monsters=[m])
    want = random.Random(5).randint(*sh.window)
    with telemetry.captured() as records:
        assert fight.throw_act(h, s, ("jump_throw", RIGHT, JumpPlan(sh.key_ms, 0, True, "above", sh)))
    (events,) = plans
    at = lambda k: next(e[0] for e in events if e[1] == k and e[2] == "down")
    assert at("f") - at("alt") == want != sh.key_ms
    jump = next(r for r in records if r["topic"] == "jump")
    assert (jump["base_ms"], jump["key_ms"], jump["jitter_ms"], jump["window"], jump["turn"], jump["n"]) == (
        reach.TOP_KEY_MS, want, want - reach.TOP_KEY_MS, list(sh.window), False, 1)
    assert jump["rise"] == pytest.approx(physics.rise_after(want / 1000), abs=0.2)
    keys = next(r for r in records if r["topic"] == "jump_keys")
    assert keys == {**keys, "n": 1, "thrown": True, "pressed_ms": pytest.approx(want + 1.6)}
    assert any(f"keyed at {want} ({want - reach.TOP_KEY_MS:+d})" in line for line in lines), lines
    assert a.key_gap() == pytest.approx(want + 1.6)
    a.sent = [[0.0, "alt", "down"]]  # (no attack key in the answer: not known)
    assert a.key_gap() is None


def test_a_jump_throw_up_close_is_jittered_over_the_keys_the_stance_finds_as_sure(monkeypatch):
    """attack.stance._from_here (from 2026-11-01): a jump throw at a pig too close to throw at from the ground (95 or 105
    off on her floor) is keyed 40 ms after the jump (facing it; tapping first if not), its plan's room (JumpPlan.room)
    the keys after it, up to reach.JITTER_FIRST_MS, whose attack surely picks it too (reach.air_sure, CLEAR_MARGIN
    farther, as the key itself), 40-45 here (her feet rise past its top); fight.jittered draws over it, and its 'jump'
    record says 'close' (or 'tap first'), base 40, the jitter, the room, the pig 105 off; not sure, no room."""
    import random
    from types import SimpleNamespace
    from collections import Counter
    from maplehelper.engine import LEFT, RIGHT
    from gamelab import telemetry
    from gamelab.attack.stance import CLEAR_MARGIN, plan_of, stance
    from gamelab.behavior import fight
    from gamelab.control import actions as actions_mod
    from gamelab.localization.monsters import MONSTER_HALF, MONSTER_TOP
    floor = plat(LEDGE, 500, 400)
    for dx in (95, 105):
        m = pig(400 + dx, 500, floor.id)
        for facing, why in ((RIGHT, "close"), (LEFT, "tap first")):
            act = stance(400, 500, floor, [m], facing=facing)
            plan = plan_of(act)
            assert act[:2] == ("jump_throw", RIGHT) and plan.why == why and plan.key_ms == 40 and plan.shot is None
            assert plan.target is m
            lo, hi = plan.room
            assert lo == 40 and hi == 45 and plan.keys == plan.room
            for k in range(lo, hi + 1, reach.KEY_CHECK_MS):  # (every key of it as sure as the key: a pig standing)
                assert reach.air_sure(dx - CLEAR_MARGIN, 0.0, MONSTER_TOP["pig"], MONSTER_HALF["pig"], k), (dx, k)
            drawn = SimpleNamespace(jitter=random.Random(dx))
            draws = {plan_of(fight.jittered(drawn, act)).key_ms for _ in range(200)}
            assert draws == set(range(lo, hi + 1))
    m = pig(505, 500, floor.id)
    plans = []
    monkeypatch.setattr(actions_mod.client, "submit",
                        lambda events, **kw: plans.append(events) or {"sent": [list(e) for e in events]})
    key = lambda name: SimpleNamespace(name=name)
    a = actions_mod.Actions(SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), attack=key("f"),
                                            lucky_seven=None, pick_up=key("z"), hp_potion=None, chair=None))
    a.facing = RIGHT
    h = SimpleNamespace(actions=a, jitter=random.Random(2), log=lambda line: None, jump_counts=Counter(), danger=None,
                        model=LEDGE)
    s = SimpleNamespace(where=SimpleNamespace(x=400.0, y=500.0), platform=floor, monsters=[m])
    act = stance(400, 500, floor, [m], facing=RIGHT)
    want = random.Random(2).randint(40, 45)
    with telemetry.captured() as records:
        assert fight.throw_act(h, s, act)
    (events,) = plans
    at = lambda k: next(e[0] for e in events if e[1] == k and e[2] == "down")
    assert at("f") - at("alt") == want
    jump = next(r for r in records if r["topic"] == "jump")
    assert (jump["kind"], jump["jump"], jump["base_ms"], jump["key_ms"], jump["jitter_ms"], jump["window"],
            jump["turn"], jump["target"], jump["dx"]) == ("close", "up", 40, want, want - 40, [40, 45], False, "pig",
                                                         105.0)
    assert next(r for r in records if r["topic"] == "jump_keys")["pressed_ms"] == want
    assert plan_of(stance(400, 500, floor, [pig(475, 500, floor.id)], facing=RIGHT)) is None or \
        plan_of(stance(400, 500, floor, [pig(475, 500, floor.id)], facing=RIGHT)).room is None  # (not sure)


def test_every_jump_attack_is_jittered_and_recorded_those_keyed_unsure_over_the_whole_room(monkeypatch):
    """behavior.fight.jittered's JITTER_ALL (the review of 2026-11-01, its F2): a jump attack the stance keyed at its
    own key, with no shot and no room (a jump throw up close not sure, at a snail; a jump back; 'tap first' not sure),
    is keyed over the soonest's whole room from the key her actions press (40 to 55; a jump back's turn 120 to 135, a
    jump throw turning in the air the same), its plan not sure; its 'jump' record says 'unsure' (or its plan's why), not
    sure, the jump's kind ('back' for a jump back, which turns her), the base and the jitter, and 'jump_keys' its press.
    Without the look, as before: the stance's key, no record; without JITTER_ALL, the stance's key, recorded (its room
    the key alone)."""
    import random
    from types import SimpleNamespace
    from collections import Counter
    from maplehelper.engine import LEFT, RIGHT
    from gamelab import telemetry
    from gamelab.attack.stance import JumpPlan, plan_of
    from gamelab.behavior import fight
    from gamelab.control import actions as actions_mod
    floor = plat(LEDGE, 500, 400)
    m = pig(470, 500, floor.id)
    plans = []
    monkeypatch.setattr(actions_mod.client, "submit",
                        lambda events, **kw: plans.append(events) or {"sent": [list(e) for e in events]})
    key = lambda name: SimpleNamespace(name=name)
    s = SimpleNamespace(where=SimpleNamespace(x=400.0, y=500.0), platform=floor, monsters=[m])

    def thrown(act, facing, seed=3):
        a = actions_mod.Actions(SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), attack=key("f"),
                                                lucky_seven=None, pick_up=key("z"), hp_potion=None, chair=None))
        a.facing = facing
        h = SimpleNamespace(actions=a, jitter=random.Random(seed), log=lambda line: None, jump_counts=Counter(),
                            danger=None, model=LEDGE)
        plans.clear()
        with telemetry.captured() as records:
            assert fight.throw_act(h, s, act)
        (events,) = plans
        at = lambda k: next(e[0] for e in events if e[1] == k and e[2] == "down")
        jump = next((r for r in records if r["topic"] == "jump"), None)
        keys = next((r for r in records if r["topic"] == "jump_keys"), None)
        return at("f") - at("alt"), jump, keys
    turn = fight.reach.turn_key_ms(fight.MONSTER_TOP["pig"])  # (a pig's: 120 by the search's model)
    cases = ((("jump_throw", RIGHT), RIGHT, 40, "unsure", "up", False),
             (("jump_throw", RIGHT), LEFT, max(40, turn), "unsure", "up", True),
             (("jump_back", RIGHT), LEFT, max(turn, physics.JUMP_BACK_ATTACK_MS), "jump back", "back", True),
             (("jump_throw", RIGHT, JumpPlan(40, 0, False, "tap first")), LEFT, 40, "tap first", "up", False))
    for act, facing, base, why, kind, turned in cases:
        seen = set()
        for seed in range(40):
            gap, jump, keys = thrown(act, facing, seed)
            assert base <= gap <= base + reach.JITTER_FIRST_MS, (act, gap)
            seen.add(gap)
            assert (jump["kind"], jump["jump"], jump["base_ms"], jump["key_ms"], jump["jitter_ms"], jump["window"],
                    jump["sure"], jump["turn"]) == (why, kind, base, gap, gap - base,
                                                    [base, base + reach.JITTER_FIRST_MS], False, turned), jump
            assert keys["n"] == jump["n"] and keys["pressed_ms"] == gap
        assert len(seen) > 8  # (drawn over the room)
    sure = ("jump_throw", RIGHT, JumpPlan(40, 0, True, "close", room=(40, 45)))
    gap, jump, _ = thrown(sure, RIGHT)
    assert 40 <= gap <= 45 and jump["sure"] is True  # (a sure one's room as before)
    h = SimpleNamespace(actions=SimpleNamespace(faces=lambda d: True), jitter=random.Random(1), model=LEDGE)
    plain = ("jump_throw", RIGHT)
    assert fight.jittered(h, plain) is plain  # (no look: as before)
    assert plan_of(fight.jittered(h, plain, s)).room == (40, 55) and not plan_of(fight.jittered(h, plain, s)).sure
    monkeypatch.setattr(fight, "JITTER_ALL", False)
    for act, facing, base, why, kind, turned in cases:
        gap, jump, keys = thrown(act, facing)
        assert gap == base and (jump["kind"], jump["jitter_ms"], jump["window"], jump["sure"]) == (
            why, 0, [base, base], False) and keys["pressed_ms"] == base


def test_a_jump_throw_up_close_keeps_its_jitter_to_keys_whose_line_is_clear(monkeypatch):
    """attack.stance._jittered_up (ROOM_LINE, the review of 2026-11-01): on Tree Dungeon's slopes a close jump throw's
    later keys had their star stopped by her floor or one just over her where the key's hit (on 33 at x 126.9 facing
    left at a Horny Mushroom 107 off, keyed 40 the world's star hits, 41-55 stopped; on 13 at -139 facing right at one
    108 off, 40-45 hit, 46-55 stopped): the room now ends before the first key whose line, at either end of the decide
    lag, is stopped, and every key of it hits in the world; without the check, the old rooms. (Its first key's line is
    checked too since the cornered study's R6, 2026-11-03: below, with reach.CLIENT_HIT_POINT on.)"""
    from maplehelper.engine import LEFT, RIGHT
    from gamelab.attack import reach
    from gamelab.attack import stance as st
    from gamelab.behavior.fight import bodies
    model = mapdata.load(10002073)
    tops = {k: b.top for k, b in bodies(model).items()}
    monkeypatch.setattr(reach, "CLIENT_HIT_POINT", False)  # (the first key's line unchecked, as then)

    def room(pid, x, mx, facing):
        p = model.platform(pid)
        m = Monster(Mob("horny-mushroom", 0, 0, 30, 30, 0.9), mx, p.y_at(mx), pid)
        plan = st.plan_of(st.stance(x, p.y_at(x), p, [m], facing=facing, tops=tops, around=[m]))
        assert plan.why == "close" and plan.key_ms == 40 and plan.target is m
        return plan.room

    def hits(pid, x, mx, facing, key):
        w = still(World(model, KEYS, (pid, x), monsters=[("Horny Mushroom", pid, mx)], facing=facing))
        w.invulnerable_ms = 10_000
        tap(w, 0.0, "alt", 0.04)
        tap(w, key / 1000, "f", 0.04)
        w.advance(1.5)
        return w.attack_log[-1][5]
    cases = ((33, 126.9, 126.9 - 107.2, LEFT, (40, 40), (40, 55)), (13, -139.0, -31.0, RIGHT, (40, 40), (40, 55)))
    for pid, x, mx, facing, now, old in cases:
        got = room(pid, x, mx, facing)
        assert got == now, (pid, got)
        assert all(hits(pid, x, mx, facing, k) == "hit" for k in range(got[0], got[1] + 1)), pid
        assert hits(pid, x, mx, facing, old[1]) == "stopped"  # (a key the old room had)
    monkeypatch.setattr(st, "ROOM_LINE", False)
    assert [room(pid, x, mx, facing) for pid, x, mx, facing, _, _ in cases] == [old for *_, old in cases]
    # The first key's own line at both ends of the decide lag too (reach.CLIENT_HIT_POINT): on 33 its late end is the
    # later keys' start, which her floor stops: no sure jump from there, a step to where a throw is sure (from the ground,
    # 122 off); on 13, as before.
    monkeypatch.setattr(st, "ROOM_LINE", True)
    monkeypatch.setattr(reach, "CLIENT_HIT_POINT", True)
    p = model.platform(33)
    m = Monster(Mob("horny-mushroom", 0, 0, 30, 30, 0.9), 126.9 - 107.2, p.y_at(126.9 - 107.2), 33)
    act = st.stance(126.9, p.y_at(126.9), p, [m], facing=LEFT, tops=tops, around=[m])
    assert act[:2] == ("back", RIGHT)
    x = 126.9 + act[2]
    there = st.stance(x, p.y_at(x), p, [m], facing=LEFT, tops=tops, around=[m])
    assert there == ("throw", LEFT)
    assert room(13, -139.0, -31.0, RIGHT) == (40, 40)


def test_a_simulated_runs_jitter_is_drawn_from_its_seed():
    """sim.scenarios.Sim.hunter: her jitter's generator seeded by the scenario's seed, so a run is the same when run
    again (its draws the same), another seed's others."""
    from gamelab.sim import scenarios
    sc = scenarios.SCENARIOS["nav_tree_up_by_29"]
    draws = {}
    for seed in (sc.seed, sc.seed, sc.seed + 1):
        sim = scenarios.Sim(mapdata.load(sc.map_id), sc if seed == sc.seed else
                            __import__("dataclasses").replace(sc, seed=seed))
        with sim:
            draws.setdefault(seed, []).append([sim.hunter().jitter.randint(0, 1000) for _ in range(5)])
    assert draws[sc.seed][0] == draws[sc.seed][1] != draws[sc.seed + 1][0]


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
    # A pig on her floor 430 off (its near side past a star's 400) walking at her at 100 units/s: by the attack's
    # pick from a jump keyed at once (0.1 s on) it has come 10 nearer, into reach; standing there, it's out of reach.
    # (Until 2026-11-01, with every key to 480 tried, 452 off: 50 nearer on her way down.)
    floor = plat(LEDGE, 500, 400)
    m = pig(400 + 430, 500, floor.id)

    def toward(mo, t):
        return mo.x - 100 * t

    assert not cover(LEDGE, floor, 400, [m]).hits(m, ("ground", "up"))
    walking = cover(LEDGE, floor, 400, [m], where=toward)
    assert walking.hits(m, ("up",)) and not walking.ground(m)


def test_a_search_asks_where_each_monster_will_be_once_a_time():
    # reach.shots (from 2026-11-01): the monsters' places at a time (`where`: behavior.fight.walk_on's, walking each
    # along its platform) worked out once a search, where both sides of a kind of jump asked them again, the times of
    # its attack keys the same either way.
    from collections import Counter
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 500)
    ms = [pig(560, 400, up.id, facing=1, pose="move"), pig(250, 400, up.id), pig(700, 500, floor.id)]
    asked = Counter()

    def walking(m, t):
        asked[(id(m), t)] += 1
        return m.x - 40 * t

    got = cover(LEDGE, floor, 400, ms, where=walking)
    assert any(s.jump is not None for s in got.shots) and asked and max(asked.values()) == 1
    assert len({t for _, t in asked}) * len(ms) == len(asked)  # (every monster placed at every time asked)


def test_jumps_not_landing_her_back_where_a_jump_shot_may_have_no_keys_tried_and_the_answers_are_the_same():
    # reach.shots' `lands` (from 2026-11-01: behavior.fight.jump_cover asks with stance.jump_room): a jump landing her
    # anywhere but back on her platform within it however it goes (reach.jump_lands_back) has none of its attack keys
    # tried, so no shot; every other shot the same, field for field, in the same order; and what the stance does and
    # what the hunt counts in reach from a jump (stance.jump_hits) the same from what's left: stance._jump_shot takes
    # no shot of such a jump, crowded (KNOCKBACK_ROOM) or not. A block 60 up over the floor from 300 to 700: from
    # under it her jumps land on it, from by its ends some of their ways do, from farther off none.
    from gamelab.attack import stance as stance_mod
    from maplehelper.engine import LEFT, RIGHT
    steps = build({"1": [seg(1, 0, 500, 1500, 500)], "2": [seg(2, 300, 440, 700, 440)]})
    floor, block = plat(steps, 500, 100), plat(steps, 440, 400)
    room = stance_mod.jump_room(floor)
    assert room == stance_mod._stand_limits(floor, crowded=False)
    lo, hi = stance_mod._stand_limits(floor, crowded=True)
    assert room[0] <= lo and hi <= room[1]
    dropped = kept = 0
    for x in range(30, 1480, 45):
        for ms in ([pig(x + 120, 440, block.id)], [pig(x - 150, 440, block.id), pig(x + 200, 500, floor.id)],
                   [pig(x + 60, 500, floor.id), pig(x - 90, 440, block.id)]):
            full = reach.coverage(steps, floor, x, 500, ms)
            cut = reach.coverage(steps, floor, x, 500, ms, lands=room)
            assert cut.shots == [s for s in full.shots if s.jump is None or s.lands_back(floor, *room)], x
            dropped += len(full.shots) - len(cut.shots)
            kept += sum(s.jump is not None for s in cut.shots)
            for facing in (LEFT, RIGHT):
                for m in ms:
                    assert (stance_mod.jump_hits(x, 500, floor, m, full, facing, ms)
                            == stance_mod.jump_hits(x, 500, floor, m, cut, facing, ms)), (x, m.x)
                for toward in (0, 1, -1):
                    assert (stance_mod.stance(x, 500, floor, ms, facing=facing, jumps=lambda: full, toward=toward)
                            == stance_mod.stance(x, 500, floor, ms, facing=facing, jumps=lambda: cut,
                                                 toward=toward)), (x, facing, toward)
    assert dropped > 100 and kept > 100


def test_her_jump_cover_tries_only_the_jumps_that_land_her_back_where_a_jump_shot_may(monkeypatch):
    from types import SimpleNamespace
    from gamelab.attack import stance as stance_mod
    from gamelab.behavior import fight
    floor = plat(LEDGE, 500, 400)
    asked = []
    monkeypatch.setattr(fight, "_air", lambda: True)
    monkeypatch.setattr(reach, "coverage", lambda *a, **kw: asked.append(kw) or reach.Coverage([]))
    s = SimpleNamespace(where=SimpleNamespace(x=400.0, y=500.0), platform=floor)
    fight.jump_cover(LEDGE, s, [pig(600, 400, 1)], [])()
    assert asked[0]["lands"] == stance_mod.jump_room(floor)


# --- the world goes by the same model -------------------------------------------------------------------------------

def world(model, pid, x, **kw):
    kw.setdefault("monsters", None)
    return World(model, KEYS, (pid, x), **kw)


@pytest.mark.parametrize("dx", [80, 160])
def test_the_world_s_attack_from_a_jump_hits_where_the_query_says(dx):
    # A pig 100 above her: the top's key (from 2026-11-01) and every key its jitter may go at hit it in the world (her
    # facing it, no turn), the soonest key doesn't (her feet 21 up then).
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 400 + dx)
    m = pig(400 + dx, 400, up.id)
    s = min((s for s in cover(LEDGE, floor, 400, [m]).of(m) if s.kind == "up" and not s.turn), key=lambda s: s.key_ms)
    lo, hi = s.window
    assert s.key_ms == reach.TOP_KEY_MS and (lo, hi) == (reach.TOP_KEY_MS - reach.JITTER_TOP_MS,
                                                         reach.TOP_KEY_MS + reach.JITTER_TOP_MS)
    for key, hits in ((lo, True), (s.key_ms, True), (hi, True), (reach.KEY_FIRST_MS, False)):
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


def _jitter_run(outcome_of, n=60, base=None, seed=3):
    """A run's records of `n` jump attacks with jitter (fight.log_jump's 'jump', throw_act's 'jump_keys') and the
    simulator's own outcome of each ('sim_attack'), `outcome_of(jitter, i)` saying which; every 7th not thrown."""
    import random
    from gamelab.sim.fakes import EPOCH
    rng, rows = random.Random(seed), [{"t": EPOCH, "topic": "session"}]
    for i in range(n):
        b = base if base is not None else (reach.TOP_KEY_MS if i % 2 else reach.KEY_FIRST_MS)
        room = (-reach.JITTER_TOP_MS, reach.JITTER_TOP_MS) if b == reach.TOP_KEY_MS else (0, reach.JITTER_FIRST_MS)
        j = rng.randint(*room)
        t = EPOCH + 10.0 * (i + 1)
        rows.append({"t": t, "topic": "jump", "kind": "above", "jump": "up", "side": 1, "turn": False,
                     "key_ms": b + j, "base_ms": b, "jitter_ms": j, "window": [b + room[0], b + room[1]],
                     "rise": 50.0, "target": "pig", "dx": 120.0, "up": 80.0, "platform": 5, "x": 0.0, "n": i + 1})
        thrown = i % 7 != 3
        rows.append({"t": t + 0.9, "topic": "jump_keys", "n": i + 1, "thrown": thrown,
                     "pressed_ms": b + j if thrown else None})
        if thrown:
            rows.append({"t": t + 0.6, "topic": "sim_attack", "key": t + 0.3 + (b + j) / 1000, "plan": "jump_throw",
                         "mode": "air", "rise": 50.0, "kind": "pig", "outcome": outcome_of(j, i), "up": 80.0})
        rows.append({"t": t + 2.0, "topic": "sim_attack", "key": t + 2.0, "plan": "throw", "mode": "ground",
                     "rise": 0.0, "kind": "pig", "outcome": "kill", "up": 0.0})  # (a throw from the ground: not its)
    return rows


def test_the_jitter_report_tabulates_jump_attacks_by_jitter_and_asks_whether_it_predicts_a_hit(tmp_path):
    """analysis.jumps' jitter report (`python -m gamelab jitter`): each 'jump' record with jitter joined to its
    'jump_keys' (not thrown: no outcome taken from the next attack) and, in a simulated run, to the simulator's own
    outcome of its attack ('sim_attack', its first jump attack keyed within MATCH_S); by base key, kind and jitter bin;
    and the permutation test: a hit only keyed early at the top (jitter under 0) has a small p, one at random a
    large one."""
    import random
    from gamelab.__main__ import main
    from gamelab.analysis import jumps
    rows = _jitter_run(lambda j, i: "kill" if j < 0 else "nothing", n=120, base=reach.TOP_KEY_MS)
    got = jumps.jitter_rows(rows)
    assert len(got) == 120 and sum(not r["thrown"] for r in got) == 17
    assert all(r["outcome"] == ("not thrown" if not r["thrown"] else "kill" if r["jitter_ms"] < 0 else "nothing")
               and r["source"] == "sim" for r in got)
    assert all(r["pressed_jitter"] == r["jitter_ms"] for r in got if r["thrown"])
    table = jumps.jitter_table(got)
    bins = [line.split()[4] for line in table.splitlines()[1:] if line.split()[:4] == ["278", "up", "no", "shot"]]
    assert bins == ["-20", "-15", "-10", "-5", "+0", "+5", "+10", "+15", "+20"], table
    test = jumps.jitter_test(got)
    p = float(test.splitlines()[1].split()[-1])
    assert p < 0.01, test
    flip = random.Random(11)
    rows = _jitter_run(lambda j, i: "kill" if flip.random() < 0.5 else "miss", n=120)
    got = jumps.jitter_rows(rows)
    assert {jumps._bin(r) for r in got if r["base_ms"] == reach.KEY_FIRST_MS} == {0, 5, 10, 15}
    test = jumps.jitter_test(got)
    assert all(float(line.split()[-1]) > 0.05 for line in test.splitlines()[1:3]), test
    path = tmp_path / "20261101-120000-sim-x.jsonl"
    path.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "report.txt"
    assert main(["jitter", str(path), "--text", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "the simulator's own" in text and "does the jitter predict a hit?" in text and "  40 up" in text


def test_the_jitter_test_shuffles_within_each_room_and_counts_presses_as_they_went():
    """analysis.jumps (the reviews of 2026-11-01): a jitter drawn from its own room (narrowed where the search found
    later keys unsure) is shuffled only among attacks of the same room, so a target's geometry (here a narrow room's
    attacks hitting a third of the time, a whole room's two thirds, whatever their jitter) passes for no jitter's effect,
    where the old test, by group alone, saw a pattern; the jitter is the press's (`pressed_ms` less the base: a press 2 ms late bins as +2 more); an
    attack pressed more than PRESS_OFF_MS off its key sent (her actions' turn in the air: 80 ms) isn't counted; a
    record not sure (`sure` False: behavior.fight.JITTER_ALL) is its own group; each 'jump_keys' goes with the
    'jump' just before it of its `n` (two hunters in one file, both numbering from 1); a record with no target doesn't
    stop the looks' outcomes."""
    import random
    from gamelab.analysis import jumps
    from gamelab.sim.fakes import EPOCH
    rng, rows = random.Random(4), [{"t": EPOCH, "topic": "session"}]
    for i in range(400):
        narrow = i % 2 == 0
        room = (0, 5) if narrow else (0, reach.JITTER_FIRST_MS)
        j = rng.randint(*room)
        t = EPOCH + 10.0 * (i + 1)
        n = i % 200 + 1  # (a second hunter from the 201st: its own numbers from 1)
        rows.append({"t": t, "topic": "jump", "kind": "close", "jump": "up", "side": 1, "turn": False,
                     "key_ms": 40 + j, "base_ms": 40, "jitter_ms": j, "window": [40 + room[0], 40 + room[1]],
                     "rise": 22.0, "target": "pig", "dx": 100.0, "up": 0.0, "platform": 5, "x": 0.0,
                     "sure": True, "n": n})
        rows.append({"t": t + 0.5, "topic": "jump_keys", "n": n, "thrown": True, "pressed_ms": 40 + j + 2.0})
        rows.append({"t": t + 0.6, "topic": "sim_attack", "key": t + 0.3, "plan": "jump_throw", "mode": "air",
                     "rise": 22.0, "kind": "pig", "up": 0.0,
                     "outcome": "kill" if rng.random() < (1 / 3 if narrow else 2 / 3) else "nothing"})
    got = jumps.jitter_rows(rows)
    assert len(got) == 400 and all(r["pressed_ms"] == r["key_ms"] + 2.0 for r in got)  # (each its own press)
    assert all(r["pressed_jitter"] == r["jitter_ms"] + 2.0 and not r["otherwise"] for r in got)
    old = jumps._perm_p([[(float(r["jitter_ms"]), r["outcome"] in jumps.HITS) for r in got]], 500, random.Random(0))
    assert old[1] < 0.01  # (by group alone: the whole rooms' larger jitters "predict" their hits)
    test = jumps.jitter_test(got)
    assert float(test.splitlines()[-1].split(" p ")[1].split()[0]) > 0.05 and "nothing to test" not in test, test
    assert {jumps._room(r) for r in got} == {(0, 5), (0, 15)} and jumps._whole(got[0]) == (0, 15)
    assert (jumps.TOP_KEY_MS, jumps.WHOLE_TOP, jumps.WHOLE_FIRST) == (  # (reach's own, not imported there)
        reach.TOP_KEY_MS, (-reach.JITTER_TOP_MS, reach.JITTER_TOP_MS), (0, reach.JITTER_FIRST_MS))
    assert {jumps._bin(r) for r in got} == {0, 5, 10, 15}
    table = jumps.jitter_table(got)
    assert "narrow" in table.splitlines()[0]
    turned = [{**r, "pressed_ms": r["pressed_ms"] + 80.0} if r["topic"] == "jump_keys" and r["t"] < EPOCH + 30.0
              else r for r in rows]  # (the first two turned in the air at their actions' own key)
    off = jumps.jitter_rows(turned)
    assert sum(r["otherwise"] for r in off) == 2 and len(jumps._counted(off)) == 398
    unsure = {**got[1], "sure": False, "kind": "unsure"}
    assert jumps._group(unsure)[3] == "unsure" and jumps._group(got[1])[3] == "close"
    blind = {"t": EPOCH + 5.0, "topic": "jump", "kind": "jump back", "jump": "back", "key_ms": 125, "target": None,
             "platform": None, "x": 10.0, "dx": None, "up": None, "n": 1}
    assert jumps.outcomes([blind])[0]["outcome"] == "nothing seen"


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
    # A pig 100 above her 150 off: out of reach from the ground (SHOT_UP), in reach from a jump in place at its top (the
    # top's key, from 2026-11-01: her feet 77 up), or turning in the air at the turn's soonest (120 ms, 52 up; until
    # then the soonest sure key either way, 95 ms on, her feet 34 up), her body clear of it however it walks meanwhile
    # (125 units/s either way: _arc_meets).
    from maplehelper.engine import LEFT, RIGHT
    from gamelab.attack.stance import plan_of, stance
    floor, up = plat(LEDGE, 500, 400), plat(LEDGE, 400, 550)
    m = pig(550, 400, up.id)
    for facing in (RIGHT, LEFT):
        act = stance(400, 500, floor, [m], facing=facing, jumps=lambda: cover(LEDGE, floor, 400, [m]))
        plan = plan_of(act)
        assert act[:2] == ("jump_throw", RIGHT) and plan.way == 0 and plan.why == "above", act
        if facing == RIGHT:  # facing it: at the top (the soonest key, 40, from 21 up, doesn't pick it)
            assert plan.key_ms == reach.TOP_KEY_MS and plan.shot.window == (reach.TOP_KEY_MS - reach.JITTER_TOP_MS,
                                                                            reach.TOP_KEY_MS + reach.JITTER_TOP_MS)
        else:  # turning in the air: its soonest key, 52 up, picks it
            assert plan.key_ms == reach.TURN_KEY_MS and plan.shot.window == (reach.TURN_KEY_MS,
                                                                             reach.TURN_KEY_MS + reach.JITTER_FIRST_MS)
        assert plan.shot.rise == pytest.approx(physics.rise_after(plan.key_ms / 1000), abs=0.1)
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
    # of the same jump; with the platform's end at 440 every way lands on the floor, and she takes it. (Her planner at
    # the 149 these were found at, SPEED 99.2: physics.set_speed, as her stat window's read sets it; at the thief's
    # 151.25 the end that catches the short kick is 455-460.)
    from gamelab.attack.stance import plan_of, stance
    from maplehelper.engine import RIGHT
    physics.set_speed(99.2)
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
