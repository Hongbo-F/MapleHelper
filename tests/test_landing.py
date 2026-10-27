"""Landings with clearance, and moving away to attack (from 2026-10-27): a landing priced by what her first attack there
surely takes (attack.stance.landing), a trip's last move chained where it lands the cheapest (motion.flow.chain's `price`:
behavior.fight.FightOnTheWay.landings), the grind's stops clear of their platforms' spawn points (behavior.patrol), up
close her moves away nearer a wall end (stance.move_limits), never onto a spawn point nobody accounts for, at most
stance.MAX_MOVES in a row, her running attack's target standing a while in her pace (Pace.hold), and the simulator's
monsters walking at her (sim.world: Realism.chase, invulnerable; the harness's `walkers` and `landings`)."""

import random
from types import SimpleNamespace

import pytest

from maplehelper.engine import LEFT, RIGHT

from gamelab.attack import danger as danger_mod
from gamelab.attack import reach
from gamelab.attack import stance as st
from gamelab.behavior import fight, patrol
from gamelab.localization import mapdata
from gamelab.localization.mapdata import Platform, Segment
from gamelab.localization.monsters import Monster, MonsterTrack
from gamelab.motion.flow import Body, Flow
from gamelab.motion.route import Planner
from gamelab.perception.mobs import Mob
from gamelab.sim import harness
from gamelab.sim import world as world_mod

from test_sim import model, plat, world  # noqa: F401 (model: the module's fixture)

TREE = 10002073
GREEN, HORNY = "green-mushroom", "horny-mushroom"
FLAT = Platform(0, [Segment(0, 500, 1000, 500)])
TOPS, HALVES = {GREEN: 51.0, HORNY: 51.0}, {GREEN: 28.0, HORNY: 28.0}
KEYS = {"left": "left", "right": "right", "jump": "alt", "up": "up", "down": "down"}


@pytest.fixture(scope="module")
def tree_map():
    return mapdata.load(TREE)


def mon(x, platform=0, kind=GREEN, y=500.0):
    return Monster(Mob(kind, 0, 0, 30, 30, 0.9), x, y, platform)


def test_a_landing_is_priced_by_what_her_first_attack_there_surely_takes(tree_map):
    # Her first attack after landing at 500: where the nearest will be then. A Green standing 52 off (its body clear of
    # hers, 28 and 22): neither a throw from the ground (the claw would swipe it: its edge 105) nor a jump in place (sure
    # from 54) surely takes it; 30 off, it's on her.
    near = st.landing(FLAT, 500, [(GREEN, 552, 552)], HALVES, TOPS)
    assert not near.sure and not near.meets and near.price_s == st.LANDING_PRICE_S and near.gap == 52
    on = st.landing(FLAT, 500, [(GREEN, 530, 530)], HALVES, TOPS)
    assert on.meets and not on.sure and on.price_s == st.LANDING_PRICE_S + st.MEET_PRICE_S
    # 60 off a jump in place keyed 40 ms after the jump key does; under a platform (no jump in place) it doesn't.
    assert st.landing(FLAT, 500, [(GREEN, 560, 560)], HALVES, TOPS).price_s == 0
    assert st.landing(FLAT, 500, [(GREEN, 560, 560)], HALVES, TOPS, can_jump=lambda a, b: False).price_s > 0
    # Past its claw's edge a throw from the ground does (reach.ground_sure), and nobody there is clear.
    assert reach.ground_sure(28.0) == (105.0, float(reach.SHOT_RANGE))
    assert st.landing(FLAT, 500, [(GREEN, 620, 620)], HALVES, TOPS).sure
    assert st.landing(FLAT, 500, [], HALVES, TOPS) == st.CLEAR_LANDING
    # One that walks into her before her attack (on her by then, or passing her) meets her: a touch, priced more.
    walker = st.landing(FLAT, 500, [(GREEN, 580, 510)], HALVES, TOPS)
    assert walker.meets and walker.price_s == st.LANDING_PRICE_S + st.MEET_PRICE_S
    assert st.landing(FLAT, 500, [(GREEN, 560, 440)], HALVES, TOPS).meets
    # A kind that hits hard: only from its keep-off (a Horny Mushroom 169 on Tree Dungeon), where she'd escape it.
    d = danger_mod.of_map(tree_map)
    assert d.keep_off(HORNY) == pytest.approx(169, abs=1)
    assert st.landing(FLAT, 500, [(HORNY, 650, 650)], HALVES, TOPS).sure
    assert not st.landing(FLAT, 500, [(HORNY, 650, 650)], HALVES, TOPS, danger=d).sure
    # From a look: each walked on as the stance's pace has it (here at 95 units/s, at her: 0.35 s from landing to her
    # attack's start, 33 units).
    pace = st.Pace(0.0, lambda m: -95.0, {GREEN: 100.0}, HALVES)
    got = st.landing_of(FLAT, 500, [mon(700)], 0.5, pace, HALVES, TOPS)
    assert got.gap == pytest.approx(700 - 95 * (0.5 + st.LANDING_SETTLE_S + st.GROUND_PICK_S) - 500, abs=0.2)


def test_her_moves_away_keep_to_a_wall_end_more_closely(tree_map):
    p24, p23 = tree_map.platform(24), tree_map.platform(23)
    # 24's right end is the tower's wall (end_drop 0), its left a short drop to 25 (74): HER_HALF and the slack from the
    # wall, EDGE_MARGIN from the short drop, where she kept 50 from both with monsters about (and still keeps DROP_ROOM
    # from 23's long drop to 29 on its left).
    assert st.end_drop(tree_map, p24, 1) == 0.0 and 0 < st.end_drop(tree_map, p24, -1) < st.LONG_DROP
    assert st.move_limits(tree_map, p24) == (p24.left + reach.EDGE_MARGIN, p24.right - st.WALL_ROOM)
    assert st.move_limits(tree_map, p23) == (p23.left + st.DROP_ROOM, p23.right - reach.EDGE_MARGIN)
    # On 24 at 255, a Green standing 25 off on her left: within the old limits (260) no step at all; now one of 30 to
    # where a jump in place surely takes it, right up to the wall.
    x, green = 255.0, mon(230.0, 24, y=p24.y_at(230.0))
    kw = dict(tops=TOPS, halves=HALVES, edges=st.drop_limits(tree_map, p24))
    old = st.stance(x, p24.y_at(x), p24, [green], False, LEFT, **kw)
    assert old[0] != "back"
    new = st.stance(x, p24.y_at(x), p24, [green], False, LEFT, limits=st.move_limits(tree_map, p24), **kw)
    assert new[0] == "back" and new[1] == RIGHT and st.MIN_MOVE <= new[2] <= 30
    # At most MAX_MOVES moves away in a row without a throw: then she throws from where she is.
    assert st.stance(x, p24.y_at(x), p24, [green], False, LEFT, limits=st.move_limits(tree_map, p24),
                     moves=st.MAX_MOVES, **kw)[0] in fight.THROWS


def test_a_step_never_ends_on_a_spawn_point_nobody_accounts_for(tree_map):
    # A Green standing 20 off: a step of 40 clears it for a jump in place (60 off then); a spawn point there whose
    # monster the look doesn't have (a respawn lands on her) rules out the steps that end within its body of it.
    green = mon(520)
    step = st._clear_step(500, 500, green, LEFT, 300, 51, 28, FLAT, [green], HALVES, st.NO_PACE)
    assert step is not None
    gx = 500 - step
    later = st._clear_step(500, 500, green, LEFT, 300, 51, 28, FLAT, [green], HALVES, st.NO_PACE, spawns=[(gx, 28.0)])
    assert later is None or abs(500 - later - gx) > 28 + st.HER_HALF
    # Danger.free_points: the spawn points of a platform the look's monsters don't account for (24's Green's, at 272,
    # free with no Green seen on 24, taken by one seen anywhere on it).
    d = danger_mod.of_map(tree_map)
    assert [round(x) for x, _ in d.free_points(24, [])] == [272]
    assert d.free_points(24, [mon(200, 24)]) == []


def test_her_running_attack_stands_its_target_a_while():
    # Pace.hold: from when its damage shows to FLINCH_HOLD_S after the key it stands (the live walkers she threw at came
    # on 19 units and stopped), then walks on.
    held = st.Pace(0.0, lambda m: -100.0, hold=lambda m: (0.2, 0.7))
    green = mon(700)
    assert held.at(green, 500, 0.1) == pytest.approx(690)
    assert held.at(green, 500, 0.6) == pytest.approx(680)
    assert held.at(green, 500, 1.0) == pytest.approx(650)
    # pace_of holds the monster her last throw was at (the hunter's `aim`, her actions' last key), walked on as its
    # track had it to the look; nothing for another monster, nor once the hold is over.
    now = fight.time.monotonic()
    h = SimpleNamespace(actions=SimpleNamespace(lock_until=0.0, attack_at=now - 0.1), model=None,
                        aim=(now - 0.1, GREEN, 0, 710.0, -100.0), danger=SimpleNamespace(any=True))
    s = SimpleNamespace(t=now, monsters=[green, mon(300)], tracks=())
    pace = fight.pace_of(h, s)
    assert pace.hold(green) == pytest.approx((0.19, 0.8), abs=0.01)
    assert pace.hold(mon(300)) is None
    h.aim = (now - 1.0, GREEN, 0, 710.0, -100.0)
    h.actions.attack_at = now - 1.0
    assert fight.pace_of(h, s).hold is None
    # Where nothing counts as a danger (the first map) her decisions are as they were: no hold.
    h.aim, h.actions.attack_at, h.danger = (now - 0.1, GREEN, 0, 710.0, -100.0), now - 0.1, SimpleNamespace(any=False)
    assert fight.pace_of(h, s).hold is None


def _judge(tree_map, tracks):
    """FightOnTheWay.landings' price, from a hunter with Tree Dungeon's danger."""
    h = SimpleNamespace(danger=danger_mod.of_map(tree_map), model=tree_map)
    return fight.FightOnTheWay(h).landings(tracks)


def test_a_trips_last_move_lands_where_her_first_attack_is_sure(tree_map):
    # 25 up onto 24: the route's jump straight up lands her at 226, 54 from a Green standing at 280 (by 24's spawn point,
    # 272), 10 down 24's slope from her: a jump in place takes it surely only from 74 (64 and CLEAR_MARGIN), a throw from
    # the ground from 105. The chain's touches don't have it (gone past regardless, the navigator's _meet: its price
    # still does), so as before it lands there; priced, it lands her where a jump in place surely takes it, from farther
    # left under 24 (its overlap with 25 is 176-276).
    p25 = tree_map.platform(25)
    path = [m for m in Planner(tree_map).moves if (m.kind, m.src, m.dst) == ("jump_up", 25, 24)]
    green = MonsterTrack(24, 280.0, speed=0.0, half=28.0, vx=0.0, height=56.0, kind=GREEN, top=56.0)
    flow = Flow(tree_map, KEYS)
    start = Body(0.0, 140.0, p25)
    plain, _, why = flow.chain(path, start, [])
    assert why == "done" and plain[-1].end.x == pytest.approx(226, abs=4)
    judge = _judge(tree_map, [green])
    assert judge(plain[-1])[0] > 0 and judge(plain[-1])[1] is green
    priced, _, why = flow.chain(path, start, [], price=lambda leg: judge(leg)[0])
    assert why == "done" and judge(priced[-1])[0] == 0
    assert 280 - priced[-1].end.x >= 74 and priced[-1].end.platform.id == 24
    # With where she's sent on 24 (`goal`), the walk on from the landing counts: the clear landing nearest it.
    to = flow.chain(path, start, [], price=lambda leg: judge(leg)[0], goal=(24, 180.0))[0]
    assert judge(to[-1])[0] == 0 and to[-1].end.x <= priced[-1].end.x + 1e-6
    # Where the first way lands clear, the chain is as it was (no price, nothing else tried).
    far = MonsterTrack(24, 300.0, speed=0.0, half=28.0, vx=0.0, height=56.0, kind=GREEN, top=56.0)
    judge_far = _judge(tree_map, [far])
    a, _, _ = flow.chain(path, start, [])
    b, _, _ = flow.chain(path, start, [], price=lambda leg: judge_far(leg)[0])
    assert judge_far(a[-1])[0] == 0 and [(leg.end.x, leg.end.t) for leg in a] == [(leg.end.x, leg.end.t) for leg in b]
    # No danger counting (the first map): no price.
    assert fight.FightOnTheWay(SimpleNamespace(danger=None, model=tree_map)).landings([green]) is None


def test_grind_stops_stand_clear_of_their_platforms_spawn_points(tree_map, monkeypatch):
    kinds, exp = {GREEN, HORNY}, {GREEN: 26.0, HORNY: 41.0}
    points = patrol.points_of(tree_map, kinds, exp)
    d = danger_mod.of_map(tree_map)

    def stops():
        return {s.platform: s.x for s in patrol.stops_of(tree_map, points, lambda pid, x: d.at(pid, x))}

    def clear(pid, x):
        return min((abs(x - q.x) for q in points if q.platform == pid), default=1e9)
    now = stops()
    monkeypatch.setattr(patrol, "STOP_SLACK", 0.0)
    before = stops()
    # 24's stop from 221 (51 from its Green's point at 272) to 201 (71); none nearer its own points than before.
    assert before[24] == 221 and now[24] == 201
    assert all(clear(pid, now[pid]) >= clear(pid, before[pid]) - 1e-6 for pid in now)
    assert sum(clear(pid, now[pid]) > clear(pid, before[pid]) + 1 for pid in now) >= 8


def test_the_worlds_monsters_walk_at_her_once_hurt(model):  # noqa: F811
    # Realism.chase: a monster on her platform within range walks its stretches at her, by its chance unhurt or hurt;
    # off, as with nobody there. Hurt (her stars), a Snail 300 off on C walks at her every stretch with chance 1.
    c = plat(model, 500, 400)
    for chase, at in (((), None), ((400.0, 0.0, 1.0), True)):
        w = world(model, c.id, 100.0, monsters=[("Snail", c.id, 400.0)], facing="right",
                  realism=world_mod.Realism(chase=chase, invulnerable=1.25))
        m = w.monsters[0]
        m.hp = m.hp_max / 2
        m.t1 = 0.0  # (its next stretch drawn now: hurt)
        m.at(0.001)
        if at:
            assert m.d == -1 and m.chase_into == pytest.approx(world_mod.HER_HALF + m.half - world_mod.CHASE_OVERLAP)
        else:
            assert m.chase is None
    # Her invulnerability after a touch is the knob's (INVULNERABLE_LIVE in `walkers`, with the live median respawn),
    # and the harness has the world.
    assert harness.WORLDS["walkers"] == {"chase": world_mod.CHASE_TREE, "invulnerable": world_mod.INVULNERABLE_LIVE,
                                         "respawn": world_mod.RESPAWN_LIVE}
    assert (world_mod.INVULNERABLE_LIVE, world_mod.RESPAWN_LIVE) == (1.25, 17.0)


def test_the_harness_counts_her_landings_by_monsters():
    # World.land_log's landings that moved her (onto another platform, or LANDING_MOVED along hers), by cause, near a
    # monster (105) and close (60), walking at her.
    log = [(1.0, 3, 10.0, 4, 50.0, "trip", 40.0, GREEN, 1), (2.0, 4, 50.0, 4, 60.0, "fight", 20.0, GREEN, 0),
           (3.0, 4, 60.0, 4, 120.0, "fight", 80.0, GREEN, -1), (4.0, None, None, 4, 100.0, "knock", None, None, None),
           (9.0, 3, 0.0, 5, 0.0, "trip", 10.0, GREEN, 1)]
    got = harness.landings(SimpleNamespace(land_log=log), 5.0)
    assert got == {"trip": {"n": 1, "near": 1, "close": 1, "at": 1}, "fight": {"n": 1, "near": 1, "close": 0, "at": 0},
                   "knock": {"n": 1, "near": 0, "close": 0, "at": 0}}
