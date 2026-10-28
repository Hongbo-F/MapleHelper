"""Landings with clearance, and moving away to attack (from 2026-10-27): a landing priced by what her first attack there
surely takes (attack.stance.landing), a chain's moves done where they land the cheapest (motion.flow.chain's `price`:
behavior.fight.FightOnTheWay.landings, what the looks don't have priced less), the monster by a trip's last landing shot
first, walked to and shot, or waited out (control.navigator._clear_landing, FightOnTheWay.shoot_from), the grind's stops
clear of their platforms' spawn points (behavior.patrol), up close her moves away nearer a wall end (stance.move_limits),
never onto a spawn point nobody accounts for, at most stance.MAX_MOVES in a row where something counts, her running
attack's target standing a while in her pace once her hit shows (Pace.hold), and the simulator's monsters walking at her
(sim.world: Realism.chase, invulnerable; the harness's `walkers` and `landings`)."""

import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from maplehelper.engine import LEFT, RIGHT

from gamelab.attack import danger as danger_mod
from gamelab.attack import reach
from gamelab import telemetry
from gamelab.attack import stance as st
from gamelab.behavior import fight, patrol
from gamelab.control import navigator as nav_mod
from gamelab.localization import mapdata
from gamelab.localization.mapdata import Platform, Segment
from gamelab.localization.monsters import Monster, MonsterTrack
from gamelab.localization.pose import Where
from gamelab.motion import flow as flow_mod
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


def mon(x, platform=0, kind=GREEN, y=500.0, bar=None):
    return Monster(Mob(kind, 0, 0, 30, 30, 0.9), x, y, platform, bar=bar)


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
    # Past its claw's edge a throw from the ground does (reach.ground_sure), and nobody there is clear; nor does one past
    # where her attack reaches at all cost anything (on Tree Dungeon's floor, 530 wide, a Green 500 off was priced as
    # one 52 off: the reviews of 2026-10-27).
    assert reach.ground_sure(28.0) == (105.0, float(reach.SHOT_RANGE))
    assert reach.star_sure(28.0) == reach.ground_sure(28.0)[0] + reach.STAR_SURE_MARGIN
    assert st.landing(FLAT, 500, [(GREEN, 620, 620)], HALVES, TOPS).sure
    assert st.landing(FLAT, 500, [], HALVES, TOPS) == st.CLEAR_LANDING
    for far in (900, 1000):
        assert st.landing(FLAT, 500, [(GREEN, far, far)], HALVES, TOPS).price_s == 0
    # The jump in place counts only for an attack she makes from a jump (the thief's; not the magician's Magic Claw,
    # cast from the ground only: character.AttackSpec.air).
    assert st.landing(FLAT, 500, [(GREEN, 560, 560)], HALVES, TOPS, air=False).price_s == st.LANDING_PRICE_S
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
    assert new[0] == "back" and new[1] == RIGHT and st.MIN_MOVE <= new[2] <= 30 and new[3:] == (st.AWAY,)
    assert fight.moved_away(new) and not fight.moved_away(new[:3])  # (a step out from under one above isn't counted)
    # At most MAX_MOVES moves away in a row without a throw, where something counts as a danger: then she throws from
    # where she is. On the first map (no danger) the count is nothing to the stance: as it was.
    d = danger_mod.of_map(tree_map)
    assert st.stance(x, p24.y_at(x), p24, [green], False, LEFT, limits=st.move_limits(tree_map, p24), danger=d,
                     moves=st.MAX_MOVES, **kw)[0] in fight.THROWS
    assert st.stance(x, p24.y_at(x), p24, [green], False, LEFT, limits=st.move_limits(tree_map, p24),
                     moves=st.MAX_MOVES, **kw)[:3] == new[:3]


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
    # track had it from the decision's look until her hit shows; till a look can show her hit (DAMAGE_SHOWS_MS) half
    # the window, then only if its bar is lower than at the decision; nothing for another monster, nor once it's over.
    now = fight.time.monotonic()
    shows = fight.actions_mod.DAMAGE_SHOWS_MS / 1000
    key = now - 0.1
    h = SimpleNamespace(actions=SimpleNamespace(lock_until=0.0, attack_at=key), model=None,
                        aim=fight.Aim(key, GREEN, 0, 710.0, -100.0, key - 0.05, None), danger=SimpleNamespace(any=True))
    green = mon(700)  # (walked on 10 units at the look, 0.15 s after the decision's)
    s = SimpleNamespace(t=now, monsters=[green, mon(300)], tracks=())
    pace = fight.pace_of(h, s)
    assert pace.hold(green) == pytest.approx((shows - 0.1, shows + (st.FLINCH_HOLD_S - shows) / 2 - 0.1), abs=0.01)
    assert pace.hold(mon(300)) is None
    # A look 0.75 s after the key, the Green where the hold has it (it stood from when her hit showed): held, but only
    # with its bar down (hurt); unhurt, it walks on as its track had it.
    later = now + 0.65
    for bar, held in ((None, False), (0.6, True)):
        stood = mon(710.0 - 100.0 * (shows + 0.05), bar=bar)
        got = fight.pace_of(h, SimpleNamespace(t=later, monsters=[stood], tracks=())).hold(stood)
        assert (got is not None) == held
        if held:
            assert got == pytest.approx((shows - 0.75, st.FLINCH_HOLD_S - 0.75), abs=0.01)
    h.aim = fight.Aim(now - 1.0, GREEN, 0, 710.0, -100.0, now - 1.05, None)
    h.actions.attack_at = now - 1.0
    assert fight.pace_of(h, s).hold is None
    # Where nothing counts as a danger (the first map) her decisions are as they were: no hold.
    h.aim, h.actions.attack_at = fight.Aim(key, GREEN, 0, 710.0, -100.0, key - 0.05, None), key
    h.danger = SimpleNamespace(any=False)
    assert fight.pace_of(h, s).hold is None
    # A jump back's throw (which rarely takes anything) holds nothing.
    h.danger = SimpleNamespace(any=True)
    fight._note_aim(h, SimpleNamespace(where=SimpleNamespace(x=600.0), platform=FLAT, monsters=[green], t=now),
                    ("jump_back", RIGHT), None)
    assert h.aim is None
    fight._note_aim(h, SimpleNamespace(where=SimpleNamespace(x=600.0), platform=FLAT, monsters=[green], t=now, tracks=()),
                    ("throw", RIGHT), None)
    assert h.aim.kind == GREEN and h.aim.x == 700 and h.aim.seen == now


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
    before = stops()  # (off by default: in the harness it cost grinding 4.3% of its EXP, patrol.STOP_SLACK)
    monkeypatch.setattr(patrol, "STOP_SLACK", 0.05)
    now = stops()
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


def test_a_chains_moves_land_where_they_cost_least_what_the_looks_miss_priced_less(tree_map, monkeypatch):
    # FightOnTheWay.landings: a track seen within STALE_S priced in full, an older one (as likely a ghost) and the spawn
    # points no track of the platform accounts for (a monster the looks missed, or a respawn) UNSEEN_PRICE of that, with
    # no track making it so (nothing to shoot first or wait out).
    p25 = tree_map.platform(25)
    up = [m for m in Planner(tree_map).moves if (m.kind, m.src, m.dst) == ("jump_up", 25, 24)]
    flow = Flow(tree_map, KEYS)
    plain = flow.chain(up, Body(0.0, 140.0, p25), [])[0]
    assert plain[-1].end.x == pytest.approx(229, abs=4)

    def green(x, seen_at=-0.1, vx=0.0, pid=24):
        return MonsterTrack(pid, x, speed=0.0, half=28.0, vx=vx, height=56.0, kind=GREEN, top=56.0, seen_at=seen_at)
    fresh, stale = green(280.0), green(280.0, seen_at=-(fight.STALE_S + 0.5))
    price, mo = _judge(tree_map, [fresh])(plain[-1])
    assert price == st.LANDING_PRICE_S and mo is fresh
    assert _judge(tree_map, [stale])(plain[-1]) == (pytest.approx(fight.UNSEEN_PRICE * st.LANDING_PRICE_S), None)
    alone = _judge(tree_map, [])  # (24's Green's point at 272, nobody seen on 24)
    assert alone(plain[-1])[0] > 0 and alone(plain[-1])[1] is None
    pairs = {}
    for m in Planner(tree_map).moves:
        pairs.setdefault((m.src, m.dst), []).append(m)
    priced = flow.chain(up, Body(0.0, 140.0, p25), [], price=lambda leg: alone(leg)[0],
                        alts=lambda mm: pairs.get((mm.src, mm.dst), []))[0]
    assert alone(priced[-1])[0] == 0 and priced[-1].end.x < plain[-1].end.x - 20
    # Planned on the move (navigator.QUEUE_SPREAD: flow.chain's `spread` False) no wider take-offs are tried: here none of
    # the route's own ways lands cheaper, so as it was.
    near_x = flow.chain(up, Body(0.0, 140.0, p25), [], price=lambda leg: alone(leg)[0],
                        alts=lambda mm: pairs.get((mm.src, mm.dst), []), spread=False)[0][-1].end.x
    assert near_x == pytest.approx(plain[-1].end.x) and not nav_mod.QUEUE_SPREAD
    monkeypatch.setattr(fight, "UNSEEN_PRICE", 0.0)  # (0: the seen ones alone, an older one left out)
    assert _judge(tree_map, [stale])(plain[-1]) == (0.0, None) and _judge(tree_map, [])(plain[-1]) == (0.0, None)
    # Every move of a chain is priced, not only the route's last (flow.PRICE_EVERY_MOVE), one she runs on from by its
    # touch alone (`last` False: FightOnTheWay.landings' MID_MEETS): 25 -> 24 -> 23 with a Green standing at 230 on
    # 24, where the route's jump up lands her on it, lands on 24 clear of its body by another way, the next move
    # following on from there; as first built, the route's last alone, as it was.
    path = Planner(tree_map).plan(25, 140.0, 23)
    assert [(m.kind, m.dst) for m in path] == [("jump_up", 24), ("jump", 23)]
    judge = _judge(tree_map, [green(230.0)])
    a = flow.chain(path, Body(0.0, 140.0, p25), [])[0]
    b = flow.chain(path, Body(0.0, 140.0, p25), [], price=lambda leg, last=True: judge(leg, last)[0],
                   alts=lambda mm: pairs.get((mm.src, mm.dst), []))[0]
    assert judge(a[0], last=False)[0] == st.MEET_PRICE_S and judge(b[0], last=False)[0] == 0
    assert judge(a[0])[0] == st.LANDING_PRICE_S + st.MEET_PRICE_S
    assert [leg.end.platform.id for leg in b] == [24, 23]
    flow_mod.PRICE_EVERY_MOVE = False
    try:
        c = flow.chain(path, Body(0.0, 140.0, p25), [], price=lambda leg, last=True: judge(leg, last)[0],
                       alts=lambda mm: pairs.get((mm.src, mm.dst), []))[0]
        assert c[0].end.x == pytest.approx(a[0].end.x)
    finally:
        flow_mod.PRICE_EVERY_MOVE = True
    # Where nothing lands any cheaper the chain is as it was (no takeoff or variant changed for nothing), and a running
    # last leg is judged where she comes to stand (its stop's slide on: flow._at_rest).
    same = flow.chain(up, Body(0.0, 140.0, p25), [], price=lambda leg: 1.0,
                      alts=lambda mm: pairs.get((mm.src, mm.dst), []))[0]
    assert same[-1].end.x == pytest.approx(plain[-1].end.x) and same[-1].end.t == pytest.approx(plain[-1].end.t)
    moving = replace(plain[-1], end=replace(plain[-1].end, run=1, v=149.0))
    assert flow_mod._at_rest(moving).end.x > plain[-1].end.x + 10 and flow_mod._at_rest(moving).end.run == 0


class _Escort:
    """A stand-in escort for _clear_landing: its judge, and what its shots give."""

    def __init__(self, judge, clear=None, walked=None):
        self.judge, self.clear, self.walked, self.calls = judge, clear, walked, []

    def landings(self, tracks):
        return self.judge

    def clear_for(self, s, m, finish=True):
        self.calls.append(("clear_for", m))
        return self.clear

    def shoot_from(self, s, m, most, finish=True):
        self.calls.append(("shoot_from", m, most))
        return self.walked


def test_the_monster_by_a_trips_last_landing_is_shot_first_walked_to_or_waited_out(tree_map, monkeypatch):
    # Navigator._clear_landing, the trip's last landing still priced by a monster the look saw: shot from where she
    # stands if a star reaches it, else walked to along her platform (SHOOT_WALK at most) and shot, MAX_FIGHTS in a row
    # at most; else waited out where it clears the landing within WAIT_AHEAD (LAND_WAITS looks, only where none can reach
    # her meanwhile); else she goes. Each a 'landing' event.
    monkeypatch.setattr(nav_mod.time, "sleep", lambda s: None)
    p25 = tree_map.platform(25)
    up = [m for m in Planner(tree_map).moves if (m.kind, m.src, m.dst) == ("jump_up", 25, 24)]
    legs = Flow(tree_map, KEYS).chain(up, Body(0.0, 140.0, p25), [])[0]
    track = MonsterTrack(24, 250.0, half=28.0, vx=0.0, kind=GREEN, seen_at=-0.1)
    seen = mon(250.0, 24)
    s = SimpleNamespace(where=Where(140.0, p25.y_at(140.0), p25, (0.0, 0.0)), platform=p25, monsters=[seen])
    landed = legs[-1].end.t

    def run(judge, clear=None, walked=None, safe=True, shots=0, waits=0):
        st_ = nav_mod.TripState(dst=24, mode=nav_mod.HUNT)
        st_.sighted, st_.shots, st_.land_waits = [(track, seen)], shots, waits
        esc, logs = _Escort(judge, clear, walked), []
        me = SimpleNamespace(escort=esc, log=logs.append, _safe_to_stand=lambda *a: safe)
        with telemetry.captured() as rec:
            got = nav_mod.Navigator._clear_landing(me, st_, s, up, [track], legs, "done")
        return got, st_, esc, logs, [r["text"].split(":")[0] for r in rec if r.get("kind") == "landing"]

    def always(leg):
        return st.LANDING_PRICE_S, track

    def clears(leg):  # (it walks off the landing within WAIT_AHEAD)
        return (0.0, None) if leg.end.t > landed + 0.5 else (st.LANDING_PRICE_S, track)
    got, st_, esc, _, ev = run(always, clear=1)
    assert got and st_.shots == 1 and st_.trip.kills == 1 and esc.calls == [("clear_for", seen)] and ev == ["shot first"]
    got, st_, esc, _, ev = run(always, walked=0)
    assert got and st_.shots == 1 and [c[0] for c in esc.calls] == ["clear_for", "shoot_from"]
    assert esc.calls[1][2] == nav_mod.SHOOT_WALK and ev == ["shot first"]
    got, st_, esc, logs, ev = run(clears)
    assert got and st_.land_waits == 1 and st_.trip.waits == 1 and ev == ["waited"] and "move off" in logs[-1]
    assert "by her landing on 24: shooting it first" in logs[0]
    nav_mod.LANDING_WALK = False  # (not walked to: only shot from where she stands)
    try:
        got, _, esc, _, _ = run(always, walked=0)
        assert [c[0] for c in esc.calls] == ["clear_for"]
    finally:
        nav_mod.LANDING_WALK = True
    assert not run(clears, safe=False)[0] and not run(clears, waits=nav_mod.LAND_WAITS)[0]
    got, _, _, _, ev = run(always)  # (standing on: nothing for it but to go)
    assert not got and ev == ["went"]
    got, _, esc, _, _ = run(always, clear=1, shots=nav_mod.MAX_FIGHTS)  # (shot at MAX_FIGHTS times already)
    assert not got and esc.calls == []
    assert not run(lambda leg: (0.3, None))[0]  # (priced by what the looks don't have: nothing to shoot or wait out)
    # Not a chain ending the trip (onto another platform, or cut short): as before.
    me = SimpleNamespace(escort=_Escort(always, clear=1), log=print, _safe_to_stand=lambda *a: True)
    assert not nav_mod.Navigator._clear_landing(me, nav_mod.TripState(dst=23, mode=nav_mod.HUNT), s, up, [track], legs,
                                                "done")
    assert not nav_mod.Navigator._clear_landing(me, nav_mod.TripState(dst=24, mode=nav_mod.HUNT), s, up, [track], legs,
                                                "monster")


def test_shooting_it_first_from_along_her_platform(tree_map):
    # FightOnTheWay.shoot_from: from 25 at 140, a Green on 24 at 260 by her landing is in a star's reach from 89
    # (attack.stance.shooting_place): she walks left 51, looks, finds it again, and shoots it there (clear_for). Not
    # past `most`, nor with a monster of her platform in the walk's way; 0 if the walk's look doesn't find it.
    p25 = tree_map.platform(25)
    g = mon(260.0, 24, y=tree_map.platform(24).y_at(260.0))
    assert st.shooting_place(p25, 140.0, p25.y_at(140.0), g, 150.0, [g]) == pytest.approx(89)
    assert st.shooting_place(p25, 140.0, p25.y_at(140.0), g, 40.0, [g]) is None
    assert st.shooting_place(p25, 140.0, p25.y_at(140.0), g, 150.0, [g, mon(110.0, 25, y=p25.y_at(110.0))]) is None
    walks, shot = [], []
    after = SimpleNamespace(where=Where(89.0, p25.y_at(89.0), p25, (0.0, 0.0)), platform=p25,
                            monsters=[mon(265.0, 24, y=tree_map.platform(24).y_at(265.0))])
    h = SimpleNamespace(actions=SimpleNamespace(walk=lambda side, units: walks.append((side, units))),
                        look=lambda: after, log=lambda m: None)
    esc = fight.FightOnTheWay(h)
    esc.clear_for = lambda s, m, finish=True: shot.append(m) or 1
    s = SimpleNamespace(where=Where(140.0, p25.y_at(140.0), p25, (0.0, 0.0)), platform=p25, monsters=[g])
    assert esc.shoot_from(s, g, 150.0) == 1 and walks == [(LEFT, pytest.approx(51))] and shot == [after.monsters[0]]
    after.monsters = []
    assert esc.shoot_from(s, g, 150.0) == 0
    assert esc.shoot_from(s, g, 40.0) is None


def test_her_escape_and_jump_back_keep_off_spawn_points_nobody_accounts_for(tree_map):
    # From a Horny Mushroom 18 off on 23 her escape jumps back; with the jump back's landing on a free spawn point a step
    # of 40, and with every step's end on one the throw from where she stands (the spawn rule, as _clear_step's).
    d = danger_mod.of_map(tree_map)
    p23 = tree_map.platform(23)
    horny = mon(78.0, 23, kind=HORNY, y=p23.y_at(78.0))
    here = st.Here(("throw", RIGHT), False, 18.0)
    room = 60.0 - st.move_limits(tree_map, p23)[0]

    def esc(spawns):
        return st._escape(60.0, RIGHT, room, lambda a, b: True, d, p23, [horny], here, HALVES, spawns=spawns)
    jump = [(60.0 - st.JUMP_BACK_SPAN, 28.0)]
    assert esc([]) == ("jump_back", RIGHT)
    assert esc(jump) == ("back", LEFT, 40.0, st.AWAY)
    assert esc(jump + [(60.0 - s, 0.0) for s in range(40, 160, 20)]) == ("throw", RIGHT)


def test_the_worlds_walkers_hover_on_reaching_her_then_walk_off(model):  # noqa: F811
    # Realism.chase: one that has walked into her (or is on her) stands there CHASE_HOVER s, then walks a stretch away
    # from her (CHASE_LEAVE): live, those that reached her hovered 0-40 off for 0.5-1.5 s and walked back (by eye).
    c = plat(model, 500, 400)
    w = world(model, c.id, 100.0, monsters=[("Snail", c.id, 400.0)], facing="right",
              realism=world_mod.Realism(chase=(400.0, 0.0, 1.0), invulnerable=1.25))
    m = w.monsters[0]
    m.hp = m.hp_max / 2
    m.x0, m.t1 = w.x + m.chase_into, 0.0  # (on her: its body CHASE_OVERLAP into hers)
    m.at(0.001)
    assert m.d == 0 and m.chase_next == 2
    assert world_mod.CHASE_HOVER[0] <= m.t1 - m.t0 <= world_mod.CHASE_HOVER[1]
    m.at(m.t1 + 0.001)
    assert m.d == 1 and world_mod.CHASE_LEAVE[0] <= m.t1 - m.t0 <= world_mod.CHASE_LEAVE[1]
    # A hit (her star's knock) has it decide afresh.
    m.chase_next = 2
    m.knock(m.t0 + 0.1, 1)
    assert m.chase_next == 0


def test_a_fall_right_after_a_knocks_landing_is_the_knocks(model):  # noqa: F811
    # World._log_landing: a knock's landing 'knock', a fall off an end within KNOCK_FALL_MS of it (its slide, or a step's
    # arrow held through it) 'knock-fall', later ones 'trip'; the harness counts them apart.
    c = plat(model, 500, 400)
    w = world(model, c.id, 100.0, monsters=[("Snail", c.id, 400.0)])
    w.knocked = True
    w._log_landing(c, 120.0)
    w.knocked = False
    w._took_off = (c.id, 150.0, w.ms + world_mod.KNOCK_FALL_MS - 100)
    w._log_landing(c, 300.0)
    w._took_off = (c.id, 300.0, w.ms + world_mod.KNOCK_FALL_MS + 100)
    w._log_landing(c, 500.0)
    assert [r[5] for r in w.land_log] == ["knock", "knock-fall", "trip"]
    got = harness.landings(w, 10.0)
    assert set(got) == {"knock", "knock-fall", "trip"} and got["trip"]["n"] == 1


def test_the_horny_escape_takes_the_safest_end_of_a_step(tree_map, monkeypatch):
    # stance._escape with ESCAPE_STEPS: of the steps every ESCAPE_STEP from STEP_ROOM up to the room there is, the one
    # whose end is the least danger (here 100 of 120: on 23 at 23, a Horny Mushroom 18 off on her right, another at -187);
    # without, the room there is (STEP_AWAY at most), as before.
    d = danger_mod.of_map(tree_map)
    p23 = tree_map.platform(23)
    near, far = mon(41.0, 23, kind=HORNY, y=p23.y_at(41.0)), mon(-187.0, 23, kind=HORNY, y=p23.y_at(-187.0))
    here = st.Here(("throw", RIGHT), False, 18.0)
    room = 23.0 - st.move_limits(tree_map, p23)[0]

    def esc():
        return st._escape(23.0, RIGHT, room, lambda a, b: True, d, p23, [near, far], here, HALVES)
    got = esc()
    monkeypatch.setattr(st, "ESCAPE_STEPS", False)
    old = esc()
    assert got[:2] == old[:2] == ("back", LEFT) and got[2] < old[2] == st.STEP_AWAY
    assert d.at(23, 23.0 - got[2], [near, far]) <= d.at(23, 23.0 - old[2], [near, far])


def test_her_moves_away_are_counted_and_a_throw_starts_the_count_again(monkeypatch):
    # fight._carry_out: a step away from a close one counts toward MAX_MOVES (FightState.moves), a step out from under one
    # above doesn't, and a throw puts the count back to none.
    walks = []
    h = SimpleNamespace(actions=SimpleNamespace(walk=lambda side, units: walks.append((side, units)), attack_at=1.0),
                        report=SimpleNamespace(throws=0))
    state = fight.FightState()
    for act in (("back", LEFT, 30.0, st.AWAY), ("back", LEFT, 20.0), ("back", RIGHT, 40.0, st.AWAY)):
        fight._carry_out(h, state, SimpleNamespace(), act, [], 0.0, 1.5, False)
    assert state.moves == 2 and len(walks) == 3
    monkeypatch.setattr(fight, "throw_act", lambda h, s, act: True)
    fight._carry_out(h, state, SimpleNamespace(), ("throw", RIGHT), [], 0.0, 1.5, False)
    assert state.moves == 0 and h.report.throws == 1
