"""Landings with clearance, and moving away to attack (from 2026-10-27): a landing priced by what her first attack there
surely takes (attack.stance.landing), a chain's moves done where they land the cheapest (motion.flow.chain's `price`:
behavior.fight.FightOnTheWay.landings, what the looks don't have priced less), the monster by a trip's last landing shot
first, walked to and shot, or waited out (control.navigator._clear_landing, FightOnTheWay.shoot_from), the shot only
where her attack reaches it from where she'll stand, else said why not (FightOnTheWay.shot_place, stance.place_to_shoot:
from 2026-10-28, the live shoot-first that threw nothing), a drop's and a jump straight up's other takeoffs from their own
spots (flow._drop, _jump_up), the grind's stops clear of their platforms' spawn points (behavior.patrol), up close her
moves away nearer a wall end (stance.move_limits), never onto a spawn point nobody accounts for, at most stance.MAX_MOVES
in a row where something counts, her running attack's target standing a while in her pace once her hit shows
(Pace.hold), and the simulator's monsters walking at her (sim.world: Realism.chase, invulnerable; the harness's
`walkers` and `landings`)."""

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


HER = danger_mod.Her(752.0, 24.0)  # what a touch is measured against: her max HP and level as her HUD read them at level
                                   # 24 (752), her DEF the stack's prior (perception's live: behavior.bootstrap)


def mon(x, platform=0, kind=GREEN, y=500.0, bar=None):
    return Monster(Mob(kind, 0, 0, 30, 30, 0.9), x, y, platform, bar=bar)


def test_a_landing_and_a_meeting_are_priced_by_a_touch_at_the_users_prices():
    """stance.TOUCH_EXP: the live Horny Mushroom's 112 HP in potions at the user's reference prices (behavior.hunt's
    HP_PER_MESO, at MESO_EXP); MEET_PRICE_S a touch at her live 6.85 EXP a second but for the 30% she evades,
    LANDING_PRICE_S the live landing's 5.7 EXP and 0.13 touches."""
    from gamelab.behavior import hunt
    assert st.TOUCH_EXP == 112.0 / hunt.HP_PER_MESO * hunt.MESO_EXP
    assert st.MEET_PRICE_S == round(st.TOUCH_EXP / 6.85 * 0.7, 2)
    assert st.LANDING_PRICE_S == round((5.7 + 0.13 * st.TOUCH_EXP) / 6.85, 2)
    assert st.LANDING_LOST_S == round(5.7 / 6.85, 2) and st.HURT_SHARE == 0.7


def test_her_landings_are_priced_by_her_touches_hp_and_rate_not_tree_dungeons():
    """The review of 2026-10-31 (one map's monster and rate frozen into general constants): with her prices
    (stance.Prices: behavior.fight.landing_prices) a landing is priced by the kind's touch (attack.danger's
    Danger.taken of her max HP as read, at the user's reference prices), her EXP a second (Hunter.exp_rate, else the
    live 6.85 wherever no grind loop sets one) and the share of her touches that hurt (her own, as 10 touches at the
    live 70% would be); stance's constants until her max HP is read. The live Horny Mushroom's numbers give the
    constants back."""
    from gamelab.behavior import fight, hunt
    touch_share = 112.0 / 840.0
    danger = SimpleNamespace(taken=lambda kind: touch_share if kind == "horny-mushroom" else 0.05)
    h = SimpleNamespace(max_hp=840.0, exp_rate=None, report=SimpleNamespace(touches=0, misses=0))
    p = fight.landing_prices(h, danger)
    assert p.touch("horny-mushroom") == pytest.approx(112.0 / hunt.HP_PER_MESO * hunt.MESO_EXP) and p.rate == 6.85
    assert p.meet_s("horny-mushroom") == pytest.approx(st.MEET_PRICE_S, abs=0.01)
    assert p.landing_s("horny-mushroom") == pytest.approx(st.LANDING_PRICE_S, abs=0.01)
    h.exp_rate, h.report = 3.4, SimpleNamespace(touches=30, misses=0)  # (half the rate, every touch of hers hurt)
    q = fight.landing_prices(h, danger)
    assert q.hurt == pytest.approx((30 + 7) / 40) and q.meet_s("horny-mushroom") > 2.5 * p.meet_s("horny-mushroom")
    assert q.meet_s("green-mushroom") < q.meet_s("horny-mushroom")
    assert fight.landing_prices(SimpleNamespace(max_hp=None), danger) is None
    near = [(GREEN, 530.0, 530.0)]  # (a Green by her where she lands: not sure, and a meeting)
    flat = FLAT
    plain = st.landing(flat, 500, near, HALVES, TOPS, air=False)
    priced = st.landing(flat, 500, near, HALVES, TOPS, air=False, prices=q)
    assert plain.meets and plain.price_s == st.LANDING_PRICE_S + st.MEET_PRICE_S
    assert priced.price_s == pytest.approx(q.landing_s(GREEN) + q.meet_s(GREEN))


def test_her_landings_rate_is_kept_in_a_band_about_the_prior():
    """The independent check of 2026-11-01: a landing's price goes by 1 / her EXP a second, and a loop's rate early in a
    grind or one near 0 priced it without bound. Her rate is taken within fight.RATE_BAND of RATE_PRIOR either way, and
    RATE_PRIOR where the hunt has none or one not over 0."""
    from gamelab.behavior import fight
    danger = SimpleNamespace(taken=lambda kind: 112.0 / 840.0)
    h = SimpleNamespace(max_hp=840.0, exp_rate=None, report=SimpleNamespace(touches=0, misses=0))
    low, high = fight.RATE_PRIOR / fight.RATE_BAND, fight.RATE_PRIOR * fight.RATE_BAND
    for rate, taken in ((None, fight.RATE_PRIOR), (0.0, fight.RATE_PRIOR), (-2.0, fight.RATE_PRIOR),
                        (float("nan"), fight.RATE_PRIOR), (0.01, low), (0.2, low), (5.4, 5.4), (8.14, 8.14),
                        (90.0, high)):
        h.exp_rate = rate
        p = fight.landing_prices(h, danger)
        assert p.rate == pytest.approx(taken)
        assert p.meet_s(HORNY) <= st.MEET_PRICE_S * fight.RATE_BAND + 0.01


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
    d = danger_mod.of_map(tree_map, HER)
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
    d = danger_mod.of_map(tree_map, HER)
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
    d = danger_mod.of_map(tree_map, HER)
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
    h = SimpleNamespace(danger=danger_mod.of_map(tree_map, HER), model=tree_map)
    return fight.FightOnTheWay(h).landings(tracks)


def test_a_trips_last_move_lands_where_her_first_attack_is_sure(tree_map, monkeypatch):
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
    # Where the first way lands clear, the chain is as it was (no price, nothing else tried): a Green at 300, 74 off, by
    # the hit point inside 24 (reach.CLIENT_HIT_POINT off). By the client's own (the cornered study's R6), its body over
    # the tower's wall at 310 by 18, a jump throw's star from the landing (229) goes into the wall: not sure, priced.
    far = MonsterTrack(24, 300.0, speed=0.0, half=28.0, vx=0.0, height=56.0, kind=GREEN, top=56.0)
    judge_far = _judge(tree_map, [far])
    a, _, _ = flow.chain(path, start, [])
    assert judge_far(a[-1])[0] > 0
    monkeypatch.setattr(reach, "CLIENT_HIT_POINT", False)
    b, _, _ = flow.chain(path, start, [], price=lambda leg: judge_far(leg)[0])
    assert judge_far(a[-1])[0] == 0 and [(leg.end.x, leg.end.t) for leg in a] == [(leg.end.x, leg.end.t) for leg in b]
    # No danger counting (the first map): no price.
    assert fight.FightOnTheWay(SimpleNamespace(danger=None, model=tree_map)).landings([green]) is None


def test_a_meeting_with_a_monster_of_no_known_kind_is_priced_as_the_maps_hardest_touch(tree_map):
    """The review of 2026-11-01: with her own prices (fight.landing_prices) a touch of a kind the map's game data
    doesn't have (a track of no kind, '', which the judge takes for a 'pig') was priced 0 (Danger.taken), so a meeting
    with one cost nothing: no shot first, no wait at a trip's last landing, where stance's flat MEET_PRICE_S had priced
    it. It's the map's hardest-touching kind's now (the navigator's stand-in for the monster that touched her,
    Navigator._toucher): the Horny Mushroom's on Tree Dungeon."""
    d = danger_mod.of_map(tree_map, HER)
    assert d.taken("") == d.taken("pig") == d.taken(HORNY) == max(d.takes.values()) > d.taken(GREEN) > 0
    h = SimpleNamespace(danger=d, model=tree_map, max_hp=HER.hp, exp_rate=None,
                        report=SimpleNamespace(touches=0, misses=0))
    p = fight.landing_prices(h, d)
    assert p.meet_s("") == p.meet_s(HORNY) > 0.5 * st.MEET_PRICE_S
    # 25 up onto 24 (the test before's way), one of no kind standing where it lands her: the meeting priced.
    path = [m for m in Planner(tree_map).moves if (m.kind, m.src, m.dst) == ("jump_up", 25, 24)]
    legs, _, _ = Flow(tree_map, KEYS).chain(path, Body(0.0, 140.0, tree_map.platform(25)), [])
    x = legs[-1].end.x
    nobody = MonsterTrack(24, x, speed=0.0, half=28.0, vx=0.0, height=56.0, kind="", top=56.0)
    price, by = fight.FightOnTheWay(h).landings([nobody])(legs[-1])
    assert by is nobody and price == pytest.approx(p.landing_s(HORNY) + p.meet_s(HORNY))


def test_grind_stops_stand_clear_of_their_platforms_spawn_points(tree_map, monkeypatch):
    kinds, exp = {GREEN, HORNY}, {GREEN: 26.0, HORNY: 41.0}
    points = patrol.points_of(tree_map, kinds, exp)
    d = danger_mod.of_map(tree_map, HER)

    def stops():
        return {s.platform: s.x for s in patrol.stops_of(tree_map, points, lambda pid, x: d.at(pid, x))}

    def clear(pid, x):
        return min((abs(x - q.x) for q in points if q.platform == pid), default=1e9)
    before = stops()  # (off by default: in the harness it cost grinding 4.3% of its EXP, patrol.STOP_SLACK)
    monkeypatch.setattr(patrol, "STOP_SLACK", 0.05)
    now = stops()
    # 16's stop from 89 (13 from its Green's point at 76) to 166 (90); none nearer its own points than before. (With the
    # stops weighing danger as the hunt loop's grounds since 2026-10-31, patrol.STOP_DANGER, 24's stays at 221, 51 from
    # its Green's point: 201 is in 23's Horny Mushrooms' reach, 0.23, where until then it went there.)
    assert before[16] == 89 and now[16] == 166 and before[24] == now[24] == 221
    assert all(clear(pid, now[pid]) >= clear(pid, before[pid]) - 1e-6 for pid in now)
    assert sum(clear(pid, now[pid]) > clear(pid, before[pid]) + 1 for pid in now) >= 5


def test_the_worlds_monsters_chase_her_only_once_she_hits_them(model):  # noqa: F811
    # Realism.chase (aggro, as the game has it: world.AGGRO_S): unhurt, a monster walks as with nobody there, the knob on
    # or off (its chase draws from its own rng); hit by her, it walks at her every stretch (CHASE_THINK_MS each, never
    # standing), on past her AGGRO_PASS and a stretch's walk at the most before turning back, until its aggro's end, a
    # hit again starting it anew; then its own walk again.
    c = plat(model, 500, 400)
    walks = []
    for chase in (False, True):
        w = world(model, c.id, 400.0, monsters=[("Snail", c.id, 700.0)], facing="right",
                  realism=world_mod.Realism(chase=chase, invulnerable=1.25))
        m = w.monsters[0]
        assert (m.chase is None) is not chase
        walks.append([(m.at(i * 0.1), m.d) for i in range(600)])
    assert walks[0] == walks[1] and any(d == 0 for _, d in walks[1])
    t = 60.0
    x0 = m.at(t)
    m.aggro(t)
    assert t + world_mod.AGGRO_S[0] <= m.aggro_until <= t + world_mod.AGGRO_S[1]
    longest = (sum(world_mod.CHASE_THINK_MS) - 1) // 90 * world_mod.AI_STEP  # (a chase decision's most: 1.17 s)
    xs, stretches = [], set()
    for i in range(1, 1201):  # (two minutes of it)
        xs.append(m.at(t + i * 0.1))
        stretches.add((m.t0, m.t1, m.d))
    assert all(d != 0 and t1 - t0 <= longest + 1e-9 for t0, t1, d in stretches)
    met = next(i for i, x in enumerate(xs) if abs(x - 400.0) <= m.speed * 0.1)
    assert (met + 1) * 0.1 <= abs(x0 - 400.0) / m.speed + 0.2  # (straight at her)
    after = [x - 400.0 for x in xs[met:]]
    past = world_mod.AGGRO_PASS + m.speed * longest
    assert max(abs(dx) for dx in after) <= past + 1e-6
    assert min(after) < -world_mod.AGGRO_PASS + 5 and max(after) > world_mod.AGGRO_PASS - 5  # (on past her, both ways)
    # A hit again starts it anew; once it's over, its own walk (standing too).
    m.aggro(t + 100.0)
    until = m.aggro_until
    assert t + 100.0 + world_mod.AGGRO_S[0] <= until <= t + 100.0 + world_mod.AGGRO_S[1]
    ends, own = set(), set()
    for i in range(1, int((until - t - 100.0 + 60.0) * 10)):
        m.at(t + 100.0 + i * 0.1)
        (own if m.t0 >= until else ends).add((m.t0, m.t1, m.d))
    assert until <= max(t1 for _, t1, _ in ends) <= until + longest + 1e-9  # (its last chase decision run out)
    assert any(d == 0 for _, _, d in own) and all(d != 0 for _, _, d in ends)
    # Her x wherever she is (it walks along its own platform at it); none once she's dead.
    w.x = 123.0
    assert w._chase_at(m, w.t) == 123.0
    w.dead = True
    assert w._chase_at(m, w.t) is None
    # Her invulnerability after a touch is the knob's (INVULNERABLE_LIVE in `walkers`; its live median respawn until
    # the server's became every world's on 2026-10-31), and the harness has the world. From her popups read
    # (2026-10-28) it was the frames' least gap, 1.45 s, everywhere (perception.touches.INVULNERABLE_S, the stack's
    # still); from the touch study (2026-11-03) the sources' 1.5, which the frames agree with read by their frame
    # steps (world.INVULNERABLE_SOURCES), the default world's too.
    assert harness.WORLDS["walkers"] == {"chase": True, "invulnerable": world_mod.INVULNERABLE_LIVE}
    # Every world's monsters chase her once she hits them (from 2026-10-31); `no-aggro` the old walk, hit or not.
    assert world_mod.Realism().chase is True and harness.WORLDS["no-aggro"] == {"chase": False}
    assert (world_mod.INVULNERABLE_LIVE, world_mod.RESPAWN_LIVE) == (1.5, 17.0)
    assert world_mod.INVULNERABLE == world_mod.Realism().invulnerable == 1.5


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
    """A stand-in escort for _clear_landing: its judge, and what its shots give (`clear`: from where she stands;
    `walked`: from along her platform, 51 off; neither: her attack reaches it from nowhere, shot_place's None), each
    shot throwing `thrown` (0: a try whose fight or walk's look didn't have it in reach after all)."""

    def __init__(self, judge, clear=None, walked=None, thrown=1):
        self.judge, self.clear, self.walked, self.calls = judge, clear, walked, []
        self.thrown, self.n = thrown, 0

    def throws(self):
        return self.n

    def landings(self, tracks):
        return self.judge

    def shot_place(self, s, m, most):
        if self.clear is not None:
            return s.where.x, ""
        if self.walked is not None and most > 0:
            return s.where.x - 51.0, ""
        return None, "her attack reaches it from nowhere on platform 25 (stand-in)"

    def clear_for(self, s, m, finish=True, jump=False):
        self.calls.append(("clear_for", m, jump))
        self.n += self.thrown if self.clear is not None else 0
        return self.clear

    def shoot_from(self, s, m, most, finish=True):
        self.calls.append(("shoot_from", m, most))
        self.n += self.thrown if self.walked is not None else 0
        return self.walked


def test_the_monster_by_a_trips_last_landing_is_shot_first_walked_to_or_waited_out(tree_map, monkeypatch):
    # Navigator._clear_landing, the trip's last landing still priced by a monster the look saw: shot from where she
    # stands if her attack reaches it there, else walked to along her platform (SHOOT_WALK at most) and shot where it
    # does (Escort.shot_place), MAX_FIGHTS in a row at most; where it reaches it from nowhere, the log says so and why
    # and nothing is sent (from 2026-10-28: live it said "shooting it first" and threw nothing); then waited out where it
    # clears the landing within WAIT_AHEAD (LAND_WAITS looks, only where none can reach her meanwhile); else she goes,
    # the log saying why. One 'landing' event a decision, with why there was no shot and why she went.
    monkeypatch.setattr(nav_mod.time, "sleep", lambda s: None)
    p25 = tree_map.platform(25)
    up = [m for m in Planner(tree_map).moves if (m.kind, m.src, m.dst) == ("jump_up", 25, 24)]
    legs = Flow(tree_map, KEYS).chain(up, Body(0.0, 140.0, p25), [])[0]
    track = MonsterTrack(24, 250.0, half=28.0, vx=0.0, kind=GREEN, seen_at=-0.1)
    seen = mon(250.0, 24)
    s = SimpleNamespace(where=Where(140.0, p25.y_at(140.0), p25, (0.0, 0.0)), platform=p25, monsters=[seen])
    landed = legs[-1].end.t

    records = []

    def run(judge, clear=None, walked=None, safe=True, shots=0, waits=0, thrown=1):
        st_ = nav_mod.TripState(dst=24, mode=nav_mod.HUNT)
        st_.sighted, st_.shots, st_.land_waits = [(track, seen)], shots, waits
        esc, logs = _Escort(judge, clear, walked, thrown), []
        me = SimpleNamespace(escort=esc, log=logs.append, _safe_to_stand=lambda *a: safe)
        with telemetry.captured() as rec:
            got = nav_mod.Navigator._clear_landing(me, st_, s, up, [track], legs, "done")
        records[:] = [r for r in rec if r.get("kind") == "landing"]
        return got, st_, esc, logs, [r["text"].split(":")[0] for r in records]

    def always(leg):
        return st.LANDING_PRICE_S, track

    def clears(leg):  # (it walks off the landing within WAIT_AHEAD)
        return (0.0, None) if leg.end.t > landed + 0.5 else (st.LANDING_PRICE_S, track)
    got, st_, esc, logs, ev = run(always, clear=1)
    assert got and st_.shots == 1 and st_.trip.kills == 1 and esc.calls == [("clear_for", seen, True)]
    assert ev == ["shot first"] and records[0]["throws"] == 1 and records[0]["kills"] == 1
    assert logs == ["  green-mushroom at x 250 by her landing on 24: shooting it first"]
    # A try that threw nothing (the fight's own looks didn't have it in reach after all: live, 00:44:09, clear_for's
    # fight logged "nothing in reach" three times): counted as a try (MAX_FIGHTS) and looked at again, but said so, not
    # a 'shot first' (as first built it was).
    got, st_, esc, logs, ev = run(always, clear=0, thrown=0)
    assert got and st_.shots == 1 and st_.trip.kills == 0 and ev == ["none thrown"]
    assert logs[-1] == "  none thrown at the green-mushroom by her landing on 24: the fight's looks didn't have it in reach"
    got, st_, esc, logs, ev = run(always, walked=0, thrown=0)
    assert got and st_.shots == 1 and ev == ["none thrown"] and records[0]["at"] == 89.0
    assert logs[-1].endswith("the walk's look didn't have it in reach")
    got, st_, esc, logs, ev = run(always, clear=0)  # (thrown, not killed: a shot)
    assert got and ev == ["shot first"] and records[0]["throws"] == 1 and records[0]["kills"] == 0
    got, st_, esc, logs, ev = run(always, walked=0)  # (walked to and shot: shoot_from alone, from x 89)
    assert got and st_.shots == 1 and [c[0] for c in esc.calls] == ["shoot_from"]
    assert esc.calls[0][2] == nav_mod.SHOOT_WALK and ev == ["shot first"] and records[0]["at"] == 89.0
    assert logs == ["  green-mushroom at x 250 by her landing on 24: shooting it first from x 89"]
    got, st_, esc, logs, ev = run(clears)  # (no shot: said so and why, nothing sent; waited out)
    assert got and st_.land_waits == 1 and st_.trip.waits == 1 and ev == ["waited"] and esc.calls == []
    assert logs[0].endswith("by her landing on 24: no shot first, her attack reaches it from nowhere on platform 25 "
                            "(stand-in)") and "move off (1)" in logs[-1] and len(logs) == 2  # (the stand-in's words)
    assert records[0]["shot"].startswith("her attack reaches it from nowhere") and records[0]["clears_s"] > 0
    nav_mod.LANDING_WALK = False  # (not walked to: only shot from where she stands)
    try:
        got, _, esc, _, ev = run(always, walked=0)
        assert not got and esc.calls == [] and ev == ["went"]
        got, _, esc, _, _ = run(always, clear=1)
        assert [c[0] for c in esc.calls] == ["clear_for"] and esc.calls[0][2]
    finally:
        nav_mod.LANDING_WALK = True
    got, _, _, logs, ev = run(clears, safe=False)
    assert not got and ev == ["went"] and records[0]["why"] == "not safe to stand and wait here"
    got, _, _, logs, ev = run(clears, waits=nav_mod.LAND_WAITS)
    assert not got and ev == ["went"] and "(LAND_WAITS)" in records[0]["why"] and "(LAND_WAITS)" in logs[-1]
    got, _, esc, logs, ev = run(always)  # (standing on: nothing for it but to go)
    assert not got and ev == ["went"] and esc.calls == []
    assert logs[-1] == (f"  landing on 24 by the green-mushroom anyway (price {st.LANDING_PRICE_S:.1f} s): its walk as "
                        f"seen doesn't clear the landing within {nav_mod.WAIT_AHEAD:.1f} s")
    assert records[0]["price"] == st.LANDING_PRICE_S and "nowhere" in records[0]["shot"]
    got, _, esc, _, ev = run(always, clear=1, shots=nav_mod.MAX_FIGHTS)  # (shot at MAX_FIGHTS times already)
    assert not got and esc.calls == [] and "(MAX_FIGHTS)" in records[0]["shot"]
    assert not run(lambda leg: (0.3, None))[0]  # (priced by what the looks don't have: nothing to shoot or wait out)
    # Not a chain ending the trip (onto another platform, or cut short): as before.
    me = SimpleNamespace(escort=_Escort(always, clear=1), log=print, _safe_to_stand=lambda *a: True)
    assert not nav_mod.Navigator._clear_landing(me, nav_mod.TripState(dst=23, mode=nav_mod.HUNT), s, up, [track], legs,
                                                "done")
    assert not nav_mod.Navigator._clear_landing(me, nav_mod.TripState(dst=24, mode=nav_mod.HUNT), s, up, [track], legs,
                                                "monster")


def test_shooting_it_first_from_along_her_platform(tree_map):
    # FightOnTheWay.shoot_from: from 21 at 42.3, a Green on 19 at 195 by her landing (78 over her) is out of her attack's
    # reach from there and in it from 17 standing there (attack.stance.shooting_place: the fights' own rule at that
    # place's height): she walks left 25, looks, finds it again, and shoots it there (clear_for). Not past `most`, nor
    # with a monster of her platform in the walk's way; 0 if the walk's look doesn't find it.
    p21, p19 = tree_map.platform(21), tree_map.platform(19)
    g = mon(195.0, 19, y=p19.y_at(195.0))
    assert not reach.can_hit(p21, 42.3, p21.y_at(42.3), g, others=[g])
    assert st.shooting_place(p21, 42.3, p21.y_at(42.3), g, 150.0, [g]) == pytest.approx(17)
    assert reach.can_hit(p21, 17.0, p21.y_at(17.0), g, others=[g])
    assert st.shooting_place(p21, 42.3, p21.y_at(42.3), g, 15.0, [g]) is None
    assert st.shooting_place(p21, 42.3, p21.y_at(42.3), g, 150.0, [g, mon(30.0, 21, y=p21.y_at(30.0))]) is None
    walks, shot = [], []
    after = SimpleNamespace(where=Where(17.0, p21.y_at(17.0), p21, (0.0, 0.0)), platform=p21,
                            monsters=[mon(200.0, 19, y=p19.y_at(200.0))])
    h = SimpleNamespace(actions=SimpleNamespace(walk=lambda side, units: walks.append((side, units))),
                        look=lambda: after, log=lambda m: None)
    esc = fight.FightOnTheWay(h)
    esc.clear_for = lambda s, m, finish=True, jump=False: shot.append((m, jump)) or 1
    s = SimpleNamespace(where=Where(42.3, p21.y_at(42.3), p21, (0.0, 0.0)), platform=p21, monsters=[g])
    assert esc.shoot_from(s, g, 150.0) == 1 and walks == [(LEFT, pytest.approx(25.3))]
    assert shot == [(after.monsters[0], True)]  # (from a jump too, where she ends up: clear_for's `jump`)
    after.monsters = []
    assert esc.shoot_from(s, g, 150.0) == 0
    assert esc.shoot_from(s, g, 15.0) is None
    # Where she'll stand, not where she is: on 25's slope (10 higher at 89 than at 140) the place was judged from her
    # height at 140, out to _goal_reach, and from 140 a Green on 24 at 260 was "in reach from 89", where standing at 89
    # it isn't (as first built the walk ended where clear_for found no throw); from nowhere on 25 is it, standing there.
    p25, p24 = tree_map.platform(25), tree_map.platform(24)
    g24 = mon(260.0, 24, y=p24.y_at(260.0))
    assert st._clear_from(p25, 140.0, p25.y_at(140.0), g24, 69.0, 251.0, [g24]) == pytest.approx(89)
    assert not reach.can_hit(p25, 89.0, p25.y_at(89.0), g24, others=[g24])
    assert st.shooting_place(p25, 140.0, p25.y_at(140.0), g24, 150.0, [g24]) is None
    walks.clear()
    at89 = SimpleNamespace(where=Where(89.0, p25.y_at(89.0), p25, (0.0, 0.0)), platform=p25, monsters=[g24])
    real = fight.FightOnTheWay(SimpleNamespace(actions=h.actions, look=lambda: at89, log=lambda m: None))
    s25 = SimpleNamespace(where=Where(140.0, p25.y_at(140.0), p25, (0.0, 0.0)), platform=p25, monsters=[g24])
    assert real.shoot_from(s25, g24, 150.0) is None and walks == []  # (as first built: walked 51, then no throw, 0)
    assert st.place_to_shoot(p25, 140.0, p25.y_at(140.0), g24, 150.0, [g24]) == \
        (None, "no throw from the ground takes it from anywhere on platform 25 (it's 76 over her)")
    # The walk's place SHOOT_SLACK inside the reach, not on its edge (the walk is open loop: live it ended a median 3
    # off its goal, half of them short; the harness's jitter 6): from 23 at -100 a Green on 22 at -60 from 48, where as
    # first built 38, which a throw takes it from and 37.5 doesn't (in the simulator 5 of 24 walks to 38 ended out of
    # reach, 2 of 24 to 48, both with the Green walking: D:/MapleHelper-scratch/lshot/mine/walk_shot.py).
    p23, p22 = tree_map.platform(23), tree_map.platform(22)
    g22 = mon(-60.0, 22, y=p22.y_at(-60.0))
    assert st.shooting_place(p23, -100.0, p23.y_at(-100.0), g22, 150.0, [g22]) == pytest.approx(48)
    assert st._reached_from(p23, -100.0, g22, -100.0, 50.0, [g22], slack=0.0) == pytest.approx(40)
    assert all(reach.can_hit(p23, x, p23.y_at(x), g22, others=[g22]) for x in (42.0, 48.0, 54.0))
    assert not reach.can_hit(p23, 37.5, p23.y_at(37.5), g22, others=[g22])


def test_her_escape_and_jump_back_keep_off_spawn_points_nobody_accounts_for(tree_map):
    # From a Horny Mushroom 18 off on 23 her escape jumps back; with the jump back's landing on a free spawn point a step
    # of 40, and with every step's end on one the throw from where she stands (the spawn rule, as _clear_step's).
    d = danger_mod.of_map(tree_map, HER)
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


def test_a_monster_she_hits_chases_her_after_its_flinch_and_forgets_it_when_back(model):  # noqa: F811
    # Realism.chase: her star's hit (World._struck) starts its chase; pushed, it flinches first (knock: FLINCH_S
    # standing), then walks at her. Back after its death, it walks its own way. The knob off, a hit changes nothing.
    c = plat(model, 500, 400)
    for chase in (False, True):
        w = world(model, c.id, 400.0, monsters=[("Snail", c.id, 600.0)], facing="right",
                  realism=world_mod.Realism(chase=chase, invulnerable=1.25))
        m = w.monsters[0]
        m.at(0.0)
        w._struck(m, False, "star")
        assert (m.aggro_until is not None) is chase
    assert m.flinch_at == 0.0 and m.pushed is not None and m.d == 0  # (pushed: its hit's damage this Snail's)
    assert m.t1 == pytest.approx(world_mod.monster_data.FLINCH_S)
    m.at(m.t1 + 0.01)
    assert m.d == -1 and m.chase_dir == -1  # (at her, from its flinch's end)
    m.alive, m.died_at = False, 1.0
    w._respawn(m)
    assert m.aggro_until is None and m.chase_dir == 0


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
    d = danger_mod.of_map(tree_map, HER)
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


# --- the live shoot-first that threw nothing (2026-10-28 00:38:36, from 2026-10-28) ------------------------------------

def _live_looks() -> dict:
    """The live looks' monsters on the route's platforms (grind-1028-0030.log 1061-1078; the scenes' sweeps), by when they
    were taken: on Tree Dungeon's 21 at x 42.3, a hunt trip to stop 23 (x 8) by the drop 21 -> 23; a Horny Mushroom on 23
    walking left (64.3, 24.3, -12.7: the third by its own clock 0.370 s after the second, the live track's vx -100, the
    tracker's clamp at the kind's speed), another standing at -131.7, Greens on 19 and 24."""
    def m(kind, x, y, pid, facing):
        return Monster(Mob(kind, 0, 0, 30, 30, 0.9, facing=facing), x, y, pid)

    def at(x, y):
        return [m(GREEN, 208.3, -644.3, 19, -1), m(HORNY, -131.7, -442.9, 23, 1), m(GREEN, 261.3, -423.1, 24, -1),
                m(HORNY, x, y, 23, -1)]
    return {36.499: at(64.3, -454.0), 36.878: at(24.3, -450.5), 37.248: at(-12.7, -446.3)}


def _live_pass(tree_map, monkeypatch, at=36.878, age=0.075):
    """The live trip's pass planned from look `at` (`age` s old then: the telemetry's scene record and the plan's log
    line), as the navigator plans and prices it (flow.chain with FightOnTheWay.landings, the route's other moves), through
    Navigator._clear_landing with the real FightOnTheWay on a hunter that records its fights and walks (none done): (what
    it returned, its log, its 'landing' events, the fights and walks sent, the chain's legs, the last leg's price)."""
    monkeypatch.setattr(nav_mod.time, "sleep", lambda s: None)
    from gamelab.localization.monsters import Tracker
    fights, walks = [], []
    h = SimpleNamespace(danger=danger_mod.of_map(tree_map, HER), model=tree_map, log=lambda text: None,
                        clear=lambda *a, **kw: fights.append(kw), report=SimpleNamespace(exp_kills=0),
                        actions=SimpleNamespace(walk=lambda side, units: walks.append(units)))
    esc, tracker, flow = fight.FightOnTheWay(h), Tracker(tree_map), Flow(tree_map, KEYS)
    moves = Planner(tree_map).moves
    path = [min((m for m in moves if (m.kind, m.src, m.dst) == ("drop", 21, 23)), key=lambda m: abs(m.x - 86.0))]
    pairs = {}
    for m in moves:
        pairs.setdefault((m.src, m.dst), []).append(m)
    p21 = tree_map.platform(21)
    for t, seen in _live_looks().items():
        tracks = tracker.update(t, seen, seen_at=-age)
        if t == at:
            break
    threats = [mo for mo in tracks if esc.threat(mo.kind)]
    judge = esc.landings(list(threats))
    legs, _, why = flow.chain(path, Body(0.0, 42.3, p21), threats, keep_going=True, hops=False,
                              price=lambda leg, last=True: judge(leg, last)[0],
                              alts=lambda mm: pairs.get((mm.src, mm.dst), []), goal=(23, 8.0))
    trip = nav_mod.TripState(dst=23, x=8.0, mode=nav_mod.HUNT)
    trip.sighted = list(zip(tracks, seen))
    s = SimpleNamespace(where=Where(42.3, p21.y_at(42.3), p21, (0.0, 0.0)), platform=p21, monsters=seen)
    logs = []
    me = SimpleNamespace(escort=esc, log=logs.append, _safe_to_stand=lambda *a: True)
    with telemetry.captured() as rec:
        got = nav_mod.Navigator._clear_landing(me, trip, s, path, threats, legs, why)
    return (got, logs, [r for r in rec if r.get("kind") == "landing"], fights, walks, legs,
            judge(legs[-1]) if legs else None)


def test_no_shot_first_is_said_where_no_attack_of_hers_reaches_it(tree_map, monkeypatch):
    # The live case: the drop 21 -> 23 from where she stood (42.3; the drop's other takeoffs left out, as 1cb0c3f's chain
    # had none: flow._spread's), its landing priced by the walker (24.3 walking left at its 100 a second: 129 off by her
    # first attack, inside its keep-off of 169). 23 is 117-122 under 21, past a star's band (reach.SHOT_DOWN), so her
    # attack reaches it from nowhere on 21. 1cb0c3f logged "shooting it first" there and sent nothing (clear_for and
    # shoot_from both found no throw); now the log says there's no shot and why, nothing is sent, and the walk as seen
    # clears the landing within WAIT_AHEAD, so she waits a look (its 'landing' event says why there was no shot).
    monkeypatch.setattr(flow_mod.Flow, "_spread", lambda self, *a: [])
    got, logs, ev, fights, walks, legs, price = _live_pass(tree_map, monkeypatch)
    assert legs[-1].end.x == pytest.approx(42.3) and price[0] == st.LANDING_PRICE_S
    assert not any("shooting it first" in line for line in logs)
    assert got and fights == [] and walks == [] and [r["text"].split(":")[0] for r in ev] == ["waited"]
    assert "horny-mushroom at x 24 by her landing on 23: no shot first, no throw from the ground takes it from " \
           "anywhere on platform 21 (it's " in logs[0] and "under her), nor one from a jump where she stands" in logs[0]
    assert "to move off (1)" in logs[1]
    assert ev[0]["shot"].startswith("no throw from the ground takes it from anywhere on platform 21")


def test_shot_place_is_where_the_fights_reach_has_her_attack_take_it(tree_map):
    # FightOnTheWay.shot_place, by the fights' own rule (reaches: reach.can_hit, clear_for's) from where she stands, else
    # stance.place_to_shoot along her platform within `most` (shoot_from's), else why not. From 21 nothing on 23: from
    # no x of 21 does a star reach the live walker, 23 lying more than reach.SHOT_DOWN under it everywhere.
    p21, p23 = tree_map.platform(21), tree_map.platform(23)
    esc = fight.FightOnTheWay(SimpleNamespace(danger=danger_mod.of_map(tree_map, HER), model=tree_map))
    for seen in _live_looks().values():
        s = SimpleNamespace(where=Where(42.3, p21.y_at(42.3), p21, (0.0, 0.0)), platform=p21, monsters=seen)
        place, why = esc.shot_place(s, seen[-1], nav_mod.SHOOT_WALK)
        assert place is None and not esc.reaches(s, seen[-1], jump=True) and "from anywhere on platform 21" in why
        assert not any(reach.can_hit(p21, x, p21.y_at(x), seen[-1], others=seen)
                       for x in range(int(p21.left), int(p21.right) + 1))
    assert min(p23.y_at(x) - p21.y_at(x) for x in range(int(p21.left), int(p21.right) + 1)) > reach.SHOT_DOWN
    # From 23 at -137 (a Horny Mushroom at 176 on it: within where she stands to shoot with monsters about) a Green on 22
    # at -200: from -57 along it (-67 the reach's edge, SHOOT_SLACK inside it), not within 40 (the nearest place said),
    # nor walked to at all with `most` 0; from -67 itself, from where she stands; with another on 23 at -100, in the
    # walk's way.
    p22 = tree_map.platform(22)
    g, crowd = mon(-200.0, 22, y=p22.y_at(-200.0)), mon(176.0, 23, kind=HORNY, y=p23.y_at(176.0))

    def at(x, others=()):
        return SimpleNamespace(where=Where(x, p23.y_at(x), p23, (0.0, 0.0)), platform=p23,
                               monsters=[g, crowd, *others])
    assert esc.shot_place(at(-137.0), g, 150.0) == (pytest.approx(-57), "")
    assert esc.shot_place(at(-137.0), g, 40.0) == \
        (None, "a throw from the ground takes it only from x -57 on platform 23, 80 off, nor one from a jump where she "
               "stands")
    assert esc.shot_place(at(-137.0), g, 0.0)[0] is None and esc.shot_place(at(-67.0), g, 0.0) == (-67.0, "")
    assert esc.shot_place(at(-137.0, [mon(-100.0, 23, y=p23.y_at(-100.0))]), g, 150.0) == \
        (None, "a monster of platform 23 in the walk's way to x -57, nor one from a jump where she stands")


def test_shot_place_takes_a_jump_throw_from_where_she_stands_as_the_fights_do(tree_map, monkeypatch):
    # FightOnTheWay.reaches with `jump` (jump_reaches: stance.jump_hits over jump_cover, Hunt._jump_hits' terms): from 25
    # at 100 a Green on 24 at 250 is out of reach of every throw from the ground on 25, and the fight jump-throws at it in
    # place (in the simulator 4 of 9 such tries killed it with one; at each, shot_place had said 'from nowhere': the
    # review's, D:/MapleHelper-scratch/lshot/review-correctness/probes/jump_clear.py; on the fix 9 of 9 agree,
    # lshot/mine/jump_clear.out). So the shot first goes from where she stands (clear_for with `jump`); _meet's
    # clear_for as before, from the ground alone. Not where her attack isn't made from a jump (the magician's Magic
    # Claw: fight._air), nor from 25 at 140 with it at 260 (no jump's attack surely picks it there).
    p25, p24 = tree_map.platform(25), tree_map.platform(24)
    esc = fight.FightOnTheWay(SimpleNamespace(danger=danger_mod.of_map(tree_map, HER), model=tree_map,
                                              actions=SimpleNamespace(facing=None)))

    def at(x, gx):
        g = mon(gx, 24, y=p24.y_at(gx))
        return SimpleNamespace(where=Where(x, p25.y_at(x), p25, (0.0, 0.0)), platform=p25, monsters=[g]), g
    s, g = at(100.0, 250.0)
    assert not esc.reaches(s, g) and esc.reaches(s, g, jump=True) and esc.shot_place(s, g, 150.0) == (100.0, "")
    assert st.place_to_shoot(p25, 100.0, p25.y_at(100.0), g, 150.0, [g])[0] is None
    s2, g2 = at(140.0, 260.0)
    assert esc.shot_place(s2, g2, 150.0) == \
        (None, "no throw from the ground takes it from anywhere on platform 25 (it's 76 over her), nor one from a jump "
               "where she stands")
    fights = []
    esc.h.clear, esc.h.log, esc.h.report = (lambda *a, **kw: fights.append(kw)), (lambda text: None), \
        SimpleNamespace(exp_kills=0, throws=0)
    assert esc.clear_for(s, g) is None and fights == []  # (_meet's: from the ground alone)
    assert esc.clear_for(s, g, jump=True) == 0 and len(fights) == 1 and esc.throws() == 0
    monkeypatch.setattr(fight, "_air", lambda: False)
    assert esc.shot_place(s, g, 150.0) == \
        (None, "no throw from the ground takes it from anywhere on platform 25 (it's 85 over her)")


def test_no_shot_first_at_one_dying(tree_map):
    # FightOnTheWay.shot_place: a monster the hunter's ledger has dead or dying (ledger.dying) is no target for any fight,
    # so no shot first at it (nav_tree_landing_shot_first, as recorded until 2026-10-28: _meet's clear_for killed the
    # Horny Mushroom by her landing, its body dying there still priced it, and the landing's shot first fired twice at
    # it, each fight "nothing in reach", 1.3 s, both logged "shooting it first").
    p24, p23 = tree_map.platform(24), tree_map.platform(23)
    horny = Monster(Mob(HORNY, 0, 0, 30, 30, 0.9), 21.0, p23.y_at(21.0), 23, track=7)
    s = SimpleNamespace(where=Where(224.0, p24.y_at(224.0), p24, (0.0, 0.0)), platform=p24, monsters=[horny])
    h = SimpleNamespace(danger=danger_mod.of_map(tree_map, HER), model=tree_map, actions=SimpleNamespace(facing=None),
                        ledger=SimpleNamespace(kills=[], dead=set()))
    esc = fight.FightOnTheWay(h)
    assert esc.shot_place(s, horny, nav_mod.SHOOT_WALK) == (224.0, "")  # (a star from 24 reaches it)
    h.ledger.dead.add(7)
    assert esc.dying(s, horny) and esc.shot_place(s, horny, nav_mod.SHOOT_WALK) ==         (None, "it's dying (a kill of hers on it): no target")


def test_a_drops_and_a_jump_ups_other_takeoffs_are_tried_from_their_own_spots(tree_map, monkeypatch):
    # flow._spread's drops from other spots over the platform below and jumps straight up from other spots under the one
    # above: as first built `at` was only where she'd rather stop (_drop, _jump_up), and standing anywhere in the clear
    # stretch she went from where she stood, so every spot gave the same move. From each spot itself now. The live pass
    # (2026-10-28 00:38:36) then drops from 60, a step of 18 right, and lands clear: by her first attack the walker 181
    # off, the standing one 192 (cost 1.41 s against 2.15 from 42.3), so nothing to shoot first or wait out: she goes.
    from gamelab.motion.flow import Keys
    monkeypatch.setattr(flow_mod, "SPREAD_FROM_SPOT", True)  # (off by default from the review of 2026-10-28)
    flow = Flow(tree_map, KEYS)
    drop = [m for m in Planner(tree_map).moves if (m.kind, m.src, m.dst) == ("drop", 21, 23)][0]
    drops = flow._spread(drop, Body(0.0, 42.3, tree_map.platform(21)), Keys(flow.key_names), "run")
    assert len({round(leg.spans[-1].x1) for leg, _ in drops}) > 5  # (within a tap's step of her: from where she is)
    monkeypatch.setattr(flow_mod, "SPREAD_FROM_SPOT", False)  # (as first built: every spot from where she stood)
    assert {round(leg.spans[-1].x1, 1) for leg, _ in flow._spread(drop, Body(0.0, 42.3, tree_map.platform(21)),
                                                                   Keys(flow.key_names), "run")} == {42.3}
    monkeypatch.setattr(flow_mod, "SPREAD_FROM_SPOT", True)
    up = [m for m in Planner(tree_map).moves if (m.kind, m.src, m.dst) == ("jump_up", 25, 24)][0]
    ups = flow._spread(up, Body(0.0, 230.0, tree_map.platform(25)), Keys(flow.key_names), "stand")
    assert len({round(leg.end.x) for leg, _ in ups}) > 5  # (under 24, their overlap 176-276)
    got, logs, ev, fights, walks, legs, price = _live_pass(tree_map, monkeypatch)
    assert legs[-1].end.x == pytest.approx(60.1, abs=1.0) and legs[-1].spans[-1].x1 == pytest.approx(60.0, abs=1.0)
    assert price == (0.0, None) and not got and ev == [] and logs == [] and fights == [] and walks == []
