"""The danger of a place (attack.danger) on synthetic maps: a spawn point between two places, a kind that hits hard
against a weak one, a monster on another platform that can't come over and one that hops, one walking a platform just
under hers, how far a fight keeps a monster off, and a look's monsters standing for their spawn points. And what's
decided by it: where she shoots a ground from (attack.spots), what she does about one too near (attack.stance), where
she waits for respawns and how much of a ground's EXP counts (behavior.hunt)."""

import math
from types import SimpleNamespace

import pytest

from gamelab.attack import danger, spots, stance
from gamelab.attack.danger import HER_TALL, HOP_CHANCE, THREAT_S, Danger, Her, soon, touch_taken
from gamelab.localization import mapdata
from gamelab.localization.monsters import Monster
from gamelab.localization.pose import Where
from gamelab.motion.bodies import HER_HALF
from gamelab.perception.mobs import Mob

HER = Her(hp=752.0, level=24.0)
# Tree Dungeon's kinds (game data: PADamage 110, 54, 13; the mushrooms never hop, slimes do; all walk 100 units/s,
# 28 units either side of their middle, 51 tall) and a hard-hitting one that hops, which no map of hers has.
NAMES = {"19": "Horny Mushroom", "13": "Green Mushroom", "7": "Slime", "99": "Hopper"}
DAMAGE = {"19": 110, "13": 54, "7": 13, "99": 110}
HOPS = {"19": False, "13": False, "7": True, "99": True}
BODIES = {i: {"top": 51.0, "flinch_top": 60.0, "half": 28.0} for i in NAMES}
SPEEDS = {i: -20 for i in NAMES}  # 125 x (1 - 20 / 100): 100 units/s
REACH = 28.0 + HER_HALF          # a monster's middle this near hers touches her


def build(footholds: list[tuple[float, float, float, float]], lives: list[tuple[str, float, float]]):
    """A map of these footholds (each a platform of its own group) and spawn points (monster id, x, y)."""
    tree = {"info": {"VRLeft": -500, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
            "foothold": {"1": {str(i): {str(i): {"x1": a, "y1": b, "x2": c, "y2": d, "prev": 0, "next": 0}}
                               for i, (a, b, c, d) in enumerate(footholds, 1)}},
            "life": {str(i): {"type": "m", "id": mob, "x": x, "y": y, "cy": y, "rx0": x, "rx1": x}
                     for i, (mob, x, y) in enumerate(lives)},
            "miniMap": {"width": 2000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4}}
    ids = {mob for mob, _, _ in lives}  # the game data the map has: its own monsters'

    def mine(table: dict) -> dict:
        return {i: v for i, v in table.items() if i in ids}
    return mapdata.parse(1, tree, "Test", {}, NAMES, mine({"19": 41, "13": 26, "7": 12, "99": 41}), mine(BODIES), {},
                         mine(SPEEDS), mine(DAMAGE), mine(HOPS))


def seen(kind: str, x: float, y: float, pid: int) -> Monster:
    return Monster(Mob(kind, 0, 0, 30, 30, 0.9), x, y, pid)


def on(model, x, y):
    return model.platform_under(x, y).id


def test_a_touch_takes_what_meowdbs_formula_says_and_weak_kinds_count_for_nothing():
    d = Danger(build([(0, 500, 800, 500)], [("19", 300, 500), ("13", 400, 500), ("7", 500, 500)]), HER)
    horny, green, slime = (d.threats[k] for k in ("horny-mushroom", "green-mushroom", "slime"))
    assert touch_taken(110, 75, 24, 0.0) == 111 and touch_taken(110, 75, 24, 0.999) == 131  # the simulator's 111-131
    assert horny.share == pytest.approx(121 / 752) and green.share == pytest.approx(54 / 752)
    assert slime.share == 0.0 and d.share("slime") == 0.0  # 9 of 752: under DANGER_MIN
    assert horny.speed == 100.0 and not horny.hops and slime.hops
    # At level 19 with 642 HP (the first map's), a pig-sized touch (PADamage 15) takes 11 HP: nothing counts there.
    assert touch_taken(15, 75, 19) / 642 < danger.DANGER_MIN


def test_between_two_spawn_points_is_more_dangerous_than_beside_one():
    model = build([(0, 500, 800, 500)], [("19", 300, 500), ("19", 500, 500)])
    d, g = Danger(model, HER), on(model, 400, 500)
    share = d.share("horny-mushroom")
    between, end = d.at(g, 400), d.at(g, 75)
    assert between == pytest.approx(2 * share * soon((100 - REACH) / 100))
    assert end == pytest.approx(share * (soon((225 - REACH) / 100) + soon((425 - REACH) / 100)))
    assert between > 1.5 * end
    assert d.at(g, 300) == pytest.approx(share * (1 + soon((200 - REACH) / 100)))  # one comes back on top of her


def test_a_monster_that_never_leaves_its_platform_cant_come_over_one_that_hops_can():
    # The ground 0..400; another platform 60 below it, 50 to its right (no overlap): the Horny Mushroom walks only its
    # own; a kind that hops gets onto the other, HOP_CHANCE as likely.
    model = build([(0, 500, 400, 500), (450, 560, 900, 560)], [("19", 200, 500), ("99", 300, 500)])
    d, other = Danger(model, HER), on(model, 600, 560)
    assert d.of("horny-mushroom", model.platform(on(model, 200, 500)), 200, model.platform(other), 600) == 0.0
    walk = (400 - 300) + 50 + (600 - 450) - REACH  # to its platform's end, across, along hers
    assert d.at(other, 600) == pytest.approx(HOP_CHANCE * d.share("hopper") * soon(walk / 100))
    far = build([(0, 500, 400, 500), (500, 560, 900, 560)], [("99", 300, 500)])  # 100 across: beyond a hop
    assert Danger(far, HER).at(on(far, 600, 560), 600) == 0.0


def test_a_kind_that_hops_from_a_platform_over_hers_leaves_it_by_an_end():
    # Hers 0..800 at y 500; one 100 above it at 200..500, over its middle: a hopper right above her can't drop through
    # its platform, it walks to the nearer end (here the left, 100 off) and down, then along hers to her.
    model = build([(0, 500, 800, 500), (200, 400, 500, 400)], [("99", 300, 400)])
    d, here, up = Danger(model, HER), on(model, 400, 500), on(model, 300, 400)
    walk = (300 - 200) + (300 - 200) - REACH
    assert d.of("hopper", model.platform(up), 300, model.platform(here), 300) == pytest.approx(
        HOP_CHANCE * d.share("hopper") * soon(walk / 100))
    assert d.at(here, 300) < HOP_CHANCE * d.share("hopper")  # not as if it dropped onto her
    # From one wider than hers by more than a hop on each side it never gets down onto hers.
    wide = build([(300, 500, 500, 500), (0, 400, 800, 400)], [("99", 400, 400)])
    assert Danger(wide, HER).at(on(wide, 400, 500), 400) == 0.0


def test_no_kind_whose_touch_counts_hops_on_her_maps():
    """Her trips' monster tracks never leave their platforms (localization.monsters.MonsterTrack.reach) while the danger
    model has a kind that hops leave its own (Danger.walk): that holds only while no kind whose touch counts hops. On
    her three maps at her level there: Tree Dungeon's mushrooms and West Rocky Mountain II's Axe Stump count and walk;
    the slimes, pigs and Orange Mushrooms hop and count for nothing. A map where one does fails here first."""
    from gamelab.sim import scenarios
    for map_id, her in ((10002073, Her(752.0, 24.0)), (10004021, Her(730.0, 23.0)), (10000012, Her(642.0, 19.0))):
        if not scenarios.available(map_id):
            continue
        d = Danger(mapdata.load(map_id), her)
        assert not [k for k, t in d.threats.items() if t.share > 0 and t.hops], map_id


def test_a_map_whose_touch_damage_cant_be_read_says_so(monkeypatch, capsys):
    from gamelab.sim import scenarios
    if not scenarios.available(10002073):
        pytest.skip("needs Tree Dungeon's game data (gamedata/data)")

    def broken(ids):
        raise OSError("no game data copy")
    monkeypatch.setattr(mapdata, "_mob_touch", broken)
    model = mapdata.load(10002073)
    assert not model.mob_damage and not Danger(model, HER).any
    assert "map 10002073: no touch damage" in capsys.readouterr().err


def test_one_walking_a_platform_just_under_or_over_hers_touches_her_where_they_overlap():
    # Her platform 300..700 at y 500; one 30 below it at 600..1000 (its monster's top, 51, reaches into her), one
    # 70 above at 0..400 (beyond her height, HER_TALL: it walks over her head).
    model = build([(300, 500, 700, 500), (600, 530, 1000, 530), (0, 430, 400, 430)],
                  [("19", 800, 530), ("19", 200, 430)])
    d, here = Danger(model, HER), on(model, 500, 500)
    assert 30 < 51 and 70 > HER_TALL
    assert d.at(here, 650) == pytest.approx(d.share("horny-mushroom") * soon((800 - 650 - REACH) / 100))
    assert d.at(here, 600 - 28 - REACH - 1) == 0.0  # where its middle can't get near enough under her
    assert d.at(here, 350) == 0.0  # under the one above: not touched


def test_a_fight_keeps_a_monster_that_hits_hard_further_off_than_the_claw():
    d = Danger(build([(0, 500, 800, 500)], [("19", 300, 500)]), HER)
    share = d.share("horny-mushroom")
    expected = REACH + 100 * THREAT_S * math.log(share / danger.KEEP_DANGER)
    assert d.keep_off("horny-mushroom") == pytest.approx(expected) and 155 < expected < 170
    assert d.of("horny-mushroom", d.model.platform(0), 0 + expected, d.model.platform(0), 0) == pytest.approx(
        danger.KEEP_DANGER)
    assert d.keep_off("green-mushroom") == 0.0 and d.keep_off("slime") == 0.0 and d.keep_off("pig") == 0.0


def test_a_look_s_monsters_stand_for_their_spawn_points_and_the_rest_may_fill():
    model = build([(0, 500, 800, 500)], [("19", 300, 500), ("19", 500, 500)])
    d, g = Danger(model, HER), on(model, 400, 500)
    share = d.share("horny-mushroom")
    far = seen("horny-mushroom", 750, 500, g)
    assert d.at(g, 100, seen=[]) == pytest.approx(d.at(g, 100))  # none seen: both may fill
    # One seen at 750: it counts where it is, and each spawn point is half as likely to bring one.
    one = d.at(g, 100, seen=[far])
    assert one == pytest.approx(share * soon((750 - 100 - REACH) / 100) + 0.5 * d.at(g, 100))
    # Both seen: only where they are.
    assert d.at(g, 100, seen=[far, far]) == pytest.approx(2 * share * soon((750 - 100 - REACH) / 100))
    # A kind that doesn't count, or one on no platform, changes nothing.
    assert d.at(g, 100, seen=[seen("slime", 120, 500, g), seen("horny-mushroom", 120, 500, None)]) == \
        pytest.approx(d.at(g, 100))


def test_the_danger_is_made_once_per_map_and_her():
    model = build([(0, 500, 800, 500)], [("19", 300, 500)])
    assert danger.of_map(model, HER) is danger.of_map(model, HER)
    assert danger.of_map(model, Her(hp=2000.0, level=60.0)).share("horny-mushroom") < 0.1
    assert not Danger(build([(0, 500, 800, 500)], [("7", 300, 500)]), HER).any  # only slimes: nothing counts


def test_the_simulators_touches_are_the_same_formula():
    from gamelab.sim import world
    info = world.MobInfo(110.0, 94.0, (), (), None)
    for u in (0.0, 0.3, 0.999):
        assert world.touch_damage(info, u, world.LEVEL_24) == touch_taken(110, world.DEFENCE, 24, u) / 752
    assert world.DEFENCE == danger.DEFENCE


# --- where she stands to shoot a ground (attack.spots), how near she lets one come (attack.stance), where she waits ---

def spot_map():
    """A ground (0..400, y 500) with two Horny Mushrooms; beside it, 10 to its right and 30 below, a platform with a
    Green Mushroom at its far end: its monsters can't reach the ground, nor the Hornies it."""
    return build([(0, 500, 400, 500), (410, 530, 900, 530)], [("19", 150, 500), ("19", 250, 500), ("13", 850, 530)])


def test_a_ground_is_shot_from_where_its_monsters_cant_reach_when_that_is_safer():
    model = spot_map()
    d, g, side = Danger(model, HER), on(model, 200, 500), on(model, 600, 530)
    worth = {"horny-mushroom", "green-mushroom"}
    here = Where(100.0, 500.0, model.platform(g), (0.0, 0.0))
    travel = lambda a, x, b: 1.0  # noqa: E731
    assert not spots.spot_ok(model, side, worth)  # a Green, worth hunting, walks it
    assert spots.spot(model, g, here, travel, worth) == spots.Spot(g, 200.0, 1.0, False)  # as before: its middle
    sp = spots.spot(model, g, here, travel, worth, danger=d)
    assert sp.platform == side and sp.coverage >= spots.MIN_COVERAGE and sp.score > 1 - spots.on_ground(model, g, d)[1]
    assert sp.x >= 410 + spots.GROUND_MARGIN  # its Green can knock her back: as far from its end as on a ground
    assert spots.spot_at(model, g, worth, d) == (sp.platform, sp.x) and spots.best_spot(model, g, worth, d) == sp
    # Hunted on itself (on_it, or with no such spot), from its safest place: away from both spawn points.
    x, there = spots.on_ground(model, g, d)
    assert x in (50.0, 350.0) and there < d.at(g, 200.0) and spots.on_ground(model, g) == (200.0, 0.0)
    assert spots.spot(model, g, here, travel, worth, on_it=True, danger=d).x == x


def test_where_no_kind_counts_spots_are_as_they_were():
    model = build([(0, 500, 400, 500), (450, 530, 900, 530)], [("7", 150, 500), ("7", 250, 500), ("7", 850, 530)])
    d, g = Danger(model, HER), on(model, 200, 500)
    here = Where(100.0, 500.0, model.platform(g), (0.0, 0.0))
    assert not d.any
    assert spots.spot(model, g, here, lambda a, x, b: 1.0, None, danger=d) == spots.spot(model, g, here,
                                                                                         lambda a, x, b: 1.0, None)
    assert spots.spots(model, g, d) == spots.spots(model, g) and spots.spot_at(model, g, None, d) == (g, 200.0)


def test_a_monster_that_hits_hard_is_kept_off_and_escaped_to_the_least_danger():
    model = build([(0, 500, 1000, 500)], [("19", 650, 500), ("19", 330, 500)])
    d, p = Danger(model, HER), model.platform(0)
    front = seen("horny-mushroom", 650, 500, p.id)
    # 150 off: past the claw's reach, inside its keep-off (162): without the danger a throw, with it a jump back.
    assert stance.stance(500, 500, p, [front]) == ("throw", "right")
    assert stance.stance(500, 500, p, [front], danger=d) == ("jump_back", "right")
    # Another 170 behind her: the jump back would land her 75 from it, a step 50: she throws from where she is.
    behind = seen("horny-mushroom", 330, 500, p.id)
    assert stance.stance(500, 500, p, [front, behind], danger=d) == ("throw", "right")
    # Inside the claw's reach a throw from the ground would be a swipe: from where she is, a jump throw in place, whose
    # attack 40 ms into the jump surely picks it 90 off (from 44); tapping the arrow on the ground first where she
    # doesn't surely face it (an air turn's attack, 120 ms into the jump, picks one on her level only from 144).
    near = seen("horny-mushroom", 590, 500, p.id)
    assert stance.stance(500, 500, p, [near, behind], danger=d, facing="right") == ("jump_throw", "right")
    assert stance.stance(500, 500, p, [near, behind], danger=d) == (
        "jump_throw", "right", stance.JumpPlan(40, 0, False, "tap first"))
    # A Green Mushroom (7.2%) is let come to the claw's reach, as before.
    green = seen("green-mushroom", 650, 500, p.id)
    assert stance.stance(500, 500, p, [green], danger=d) == stance.stance(500, 500, p, [green]) == ("throw", "right")


def test_escaping_with_the_way_back_clear_she_jumps_back_and_throws_and_never_steps_into_another_monster():
    # A lone Horny Mushroom on open ground, its spawn point ahead of her (and another further on): a jump back (which
    # throws) wins over a step away, 120 to its 95 and a little safer for it, at any distance inside the keep-off.
    for spawns in ([("19", 650, 500)], [("19", 650, 500), ("19", 900, 500)]):
        model = build([(0, 500, 1000, 500)], spawns)
        d, p = Danger(model, HER), model.platform(0)
        for off in (60, 100, 140, 150):
            front = seen("horny-mushroom", 500 + off, 500, p.id)
            assert stance.stance(500, 500, p, [front], danger=d) == ("jump_back", "right"), (spawns, off)
    # A Green Mushroom behind her: no move whose way or landing takes her into its body (its half 28 and her 22,
    # motion.bodies).
    model = build([(0, 500, 1000, 500)], [("19", 630, 500), ("13", 300, 500)])
    d, p = Danger(model, HER), model.platform(0)
    front = seen("horny-mushroom", 630, 500, p.id)
    for behind_x in (385, 350):  # 115 and 150 behind: the step (120) ends past or 30 from its middle
        act = stance.stance(500, 500, p, [front], around=[front, seen("green-mushroom", behind_x, 500, p.id)],
                            danger=d)
        assert act[0] != "back", (behind_x, act)
        if act[0] == "jump_back":  # 95 back: clear of it only with it 150 behind (55 > 50)
            assert behind_x == 350
    edge = 350 + stance.half_of("green-mushroom") + HER_HALF  # (a pig's half without the map's game data: 34)
    assert stance._into(500, edge - 1, p, [seen("green-mushroom", 350, 500, p.id)], None)
    assert not stance._into(500, edge + 1, p, [seen("green-mushroom", 350, 500, p.id)], None)
    assert not stance._into(500, edge - 1, p, [seen("green-mushroom", 350, 500, p.id + 1)], None)  # another platform's


def _waiting(model, kinds):
    from gamelab.behavior.hunt import Hunt, Visit
    g = on(model, 400, 500)
    h = object.__new__(Hunt)
    h.model, h.kinds = model, kinds
    h.h = SimpleNamespace(danger=Danger(model, HER), knocked=lambda s: False)
    h.visit = Visit(g, spots.Spot(g, 400.0, 1.0, False))
    h.tour = SimpleNamespace(hunting_grounds=lambda visit_s=None: [g], stats={})  # the only ground
    h._skip, h._rope_rest, h._hanging = {}, 0.0, None  # (no rope on these maps: rule 9 on one never comes up)
    return h, SimpleNamespace(platform=model.platform(g), where=SimpleNamespace(x=400.0, y=500.0), monsters=[], hp=1.0)


def test_waiting_she_steps_off_between_two_spawn_points_of_a_kind_that_hits_hard():
    from gamelab.behavior.hunt import WAIT_GAIN
    model = build([(0, 500, 800, 500)], [("19", 300, 500), ("19", 500, 500)])
    h, s = _waiting(model, {"horny-mushroom"})
    side, units, goal, there = h._wait_place(s)
    assert (side, units) == (("left", 400 - goal) if goal < 400 else ("right", goal - 400))
    assert min(abs(goal - 300), abs(goal - 500)) >= 150  # off both, out of a fresh one's reach
    assert h._spawns_in_reach(s, goal) == h._spawns_in_reach(s, 400.0) == 2  # both still in reach
    assert h.danger.at(s.platform.id, 400.0, []) - there >= WAIT_GAIN
    s.where.x = goal  # there already: she waits where she is
    assert h._wait_place(s) is None
    weak = build([(0, 500, 800, 500)], [("7", 300, 500), ("7", 500, 500)])  # slimes: nothing counts, as before
    h, s = _waiting(weak, {"slime"})
    assert h._wait_place(s) is None


def test_waiting_she_never_walks_past_a_monster_that_hits_hard_and_looks_again_after_a_step():
    from gamelab.behavior.hunt import WAIT_STEP
    model = build([(0, 500, 800, 500)], [("19", 300, 500), ("19", 500, 500)])
    h, s = _waiting(model, {"horny-mushroom"})
    side, units, goal, _ = h._wait_place(s)  # by an end (the left: as safe as the right, and first)
    assert goal < 200 and side == "left" and units > WAIT_STEP  # a walk of several steps, a look after each
    # One of them back at 300, between her and there: she stays on her side of it, or goes the other way.
    s.monsters = [seen("horny-mushroom", 300.0, 500.0, s.platform.id)]
    step = h._wait_place(s)
    assert step is None or step[2] > 400
    assert h._in_the_way(s, 400.0, 100.0) and not h._in_the_way(s, 400.0, 700.0)
    assert not h._in_the_way(s, 400.0, 360.0) and h._in_the_way(s, 400.0, 340.0)  # its body and hers meet at 343


def test_waiting_she_walks_one_step_at_most_then_looks_again():
    from gamelab.behavior.hunt import WAIT_STEP, HuntReport
    model = build([(0, 500, 800, 500)], [("19", 300, 500), ("19", 500, 500)])
    h, s = _waiting(model, {"horny-mushroom"})
    walks, lines = [], []
    h.h.actions = SimpleNamespace(walk=lambda side, units: walks.append((side, units)))
    h.report, h._rule, h._idle_since, h._s, h.log = HuntReport(), "", None, None, lines.append
    side, units, goal, _ = h._wait_place(s)
    assert units > WAIT_STEP  # the place is several steps off ...
    h._wait(s, [])
    assert walks == [(side, WAIT_STEP)] and h.report.rules["wait"] == 1  # ... she walks one, and the next look decides
    assert f"waiting at x {goal:.0f} instead" in lines[0]
    s.where.x = goal + (60 if side == "left" else -60)  # 60 short of it: the rest of the way
    h._wait(s, [])
    assert walks[-1] == (side, 60.0)


def test_a_ground_counts_less_of_its_exp_the_more_dangerous_its_stand():
    from gamelab.behavior import hunt
    assert hunt._safe_share(spots.Spot(1, 0.0, 1.0, False)) == 1.0
    assert hunt._safe_share(spots.Spot(1, 0.0, 1.0, False, 0.1)) == pytest.approx(1 - hunt.GROUND_DANGER * 0.1)
    assert hunt._safe_share(spots.Spot(1, 0.0, 1.0, False, 0.9)) == 0.0


def test_of_two_grounds_as_good_she_goes_to_the_one_with_the_safer_stand(monkeypatch):
    """_next_ground and the rate rule (_best_rate) weigh a ground's EXP a second by the danger where she'd stand to hunt
    it (_safe_share): of two grounds giving as much, the one shot from a safe spot; and the one among Horny Mushrooms
    only when it gives enough more."""
    from gamelab.behavior import hunt
    from gamelab.behavior.hunt import Hunt, HuntReport, Visit
    values = {7: 20.0, 8: 20.0}
    stands = {7: spots.Spot(7, 0.0, 1.0, False, 0.18), 8: spots.Spot(9, 0.0, 0.9, True, 0.0)}
    h = object.__new__(Hunt)
    h.tour = SimpleNamespace(ground_values=lambda: values, hunting_grounds=lambda visit_s=None: list(values) + [33],
                             stats={})
    h.nav = SimpleNamespace(planner=SimpleNamespace(travel=lambda src, x, dst: 10.0))
    h.model, h.kinds, h.rng, h.log = None, None, SimpleNamespace(uniform=lambda a, b: 1.0), lambda text: None
    h._skip, h._left, h.visit, h.stack, h.obj, h.report = {}, {}, None, [], None, HuntReport()
    h.h = SimpleNamespace(danger=object())
    monkeypatch.setattr(hunt.spots, "best_spot", lambda model, g, kinds, danger=None: stands.get(g, stands[8]))
    monkeypatch.setattr(hunt.spots, "spot", lambda model, g, where, travel, kinds, danger=None: stands[g])
    s = SimpleNamespace(where=SimpleNamespace(platform=SimpleNamespace(id=33), x=0.0), players=[])
    h._next_ground(s)
    assert h.visit.ground == 8  # as much EXP, from a safe spot
    values[7] = 20.0 / hunt._safe_share(stands[7]) * 1.05  # 5% more than its danger costs: worth it
    h.visit = None
    h._next_ground(s)
    assert h.visit.ground == 7
    # The rate rule, hunting 33: the best other ground's rate counts its stand's danger the same way.
    values[7] = 20.0
    h.visit = Visit(33, spots.Spot(33, 0.0, 1.0, False))
    rate, g = h._best_rate(SimpleNamespace(platform=SimpleNamespace(id=33), where=s.where, players=[]))
    assert g == 8 and rate == pytest.approx(20.0 / (10.0 + h._visit_s(8)))


def test_her_steps_away_keep_drop_room_from_an_end_with_a_long_drop_and_none_from_another():
    """stance.drop_limits (`edges`): DROP_ROOM in from an end with the next platform under it LONG_DROP or more down (or
    none), the end itself where the drop is short; an escape's step toward a long drop stops DROP_ROOM from it, and with
    less room than that, there's no step that way (live on 2026-10-25 three knocks took her off Tree Dungeon's 23 from
    52-79 units in, each after a step toward its left end)."""
    model = build([(0, 500, 1000, 500), (1000, 540, 1300, 540)], [("19", 300, 500)])  # nothing under its left end
    d, p = Danger(model, HER), model.platform_under(500, 500)
    assert stance.end_drop(model, p, -1) == math.inf and stance.end_drop(model, p, 1) == 40
    assert stance.drop_limits(model, p) == (stance.DROP_ROOM, 1000)
    horny = seen("horny-mushroom", 300, 500, p.id)
    for x in (150.0, 180.0, 230.0):  # escaping left from one on her right
        mine = seen("horny-mushroom", x + 60, 500, p.id)
        act = stance.stance(x, 500, p, [mine], around=[mine], danger=d, edges=stance.drop_limits(model, p))
        assert not (act[0] == "back" and x - act[2] < stance.DROP_ROOM), (x, act)
        assert not (act[0] == "jump_back" and x - stance.JUMP_BACK_SPAN < stance.DROP_ROOM), (x, act)
        old = stance.stance(x, 500, p, [mine], around=[mine], danger=d)  # without: stepped toward the drop
        assert old[:2] in (("back", "left"), ("jump_back", "right")), (x, old)
    assert horny.platform == p.id


def test_on_a_platform_too_narrow_for_the_drop_room_her_steps_go_only_toward_its_middle():
    """stance.drop_limits on a platform too narrow to keep DROP_ROOM from both its ends, each with a long drop under it
    (Tree Dungeon's 18, 27 and 30: 187-188 units, drops of 125-254 units; review of 2026-10-26): its middle, so a step
    away goes only toward the middle, never toward either end (the whole platform before: a step left from x 120 on a
    180-unit one went to 50, 50 units from its end); with one long end, the other end."""
    model = build([(0, 500, 180, 500), (600, 500, 680, 500), (680, 540, 900, 540)], [("19", 90, 500)])
    d, p = Danger(model, HER), model.platform_under(90, 500)
    assert stance.end_drop(model, p, -1) == math.inf and stance.end_drop(model, p, 1) == math.inf
    assert stance.drop_limits(model, p) == (90.0, 90.0)
    for x, monster in ((120.0, 165.0), (60.0, 15.0), (100.0, 150.0), (80.0, 30.0)):
        mine = seen("horny-mushroom", monster, 500, p.id)
        act = stance.stance(x, 500, p, [mine], around=[mine], danger=d, edges=stance.drop_limits(model, p))
        if act[0] == "back":
            end = x + (act[2] if act[1] == "right" else -act[2])
            assert abs(end - 90.0) <= abs(x - 90.0) + 1e-6, (x, act)  # toward the middle, if at all
        assert act[0] != "jump_back" or abs(x - 90.0) >= stance.JUMP_BACK_SPAN, (x, act)
        old = stance.stance(x, 500, p, [mine], around=[mine], danger=d, edges=(p.left, p.right))
        to = x - old[2] if old[0] == "back" and old[1] == "left" else x + old[2] if old[0] == "back" else x
        assert min(to - p.left, p.right - to) < min(x - p.left, p.right - x) <= stance.DROP_ROOM, (x, old)  # nearer
    short = model.platform_under(640, 500)  # 80 wide, its left end a long drop, its right 40 down to the next
    assert stance.end_drop(model, short, -1) == math.inf and stance.end_drop(model, short, 1) == 40
    assert stance.drop_limits(model, short) == (680.0, 680.0)


def test_under_a_platform_overhead_an_escape_can_be_the_jump_up_onto_it():
    """stance._escape's `up` (overhead): on her platform under another one within her jump height, a jump (back, or in
    place) lands her up on it; the jump up, throwing back as she rises, is an escape by the danger where she'd land
    there, where a Horny Mushroom on her platform can't reach her (60 above: over its 51 units)."""
    model = build([(0, 500, 1000, 500), (0, 440, 400, 440)], [("19", 300, 500)])
    d = Danger(model, HER)
    her, over = model.platform_under(700, 500), model.platform_under(200, 440)
    assert stance.overhead(model, her, 200) is over and stance.overhead(model, her, 700) is None
    assert not stance.open_above(model, her, 200, 200)
    mine = seen("horny-mushroom", 260, 500, her.id)
    up = lambda x: stance.overhead(model, her, x)  # noqa: E731
    act = stance.stance(200, 500, her, [mine], around=[mine], danger=d, can_jump=lambda a, b: stance.open_above(
        model, her, a, b), up=up, edges=stance.drop_limits(model, her))
    assert act[0] == "jump_throw" and act[1] == "right" and stance.plan_of(act).why == f"up onto {over.id}"
    assert d.at(over.id, 200, [mine]) < d.at(her.id, 200, [mine])
    before = stance.stance(200, 500, her, [mine], around=[mine], danger=d,
                           can_jump=lambda a, b: stance.open_above(model, her, a, b))
    assert before[0] != "jump_throw" or stance.plan_of(before) is None


def test_of_the_monsters_in_reach_one_on_her_platform_that_counts_is_thrown_at_first():
    """stance._throw_side with the danger (TOUCHER_FIRST): toward a monster in reach on her platform whose kind counts,
    whichever way she faces (live on 2026-10-25 her spot's Green Mushroom on 24 made 14 of her 34 touches, 3 while she
    shot 23's Horny Mushrooms the other way); without the danger, the way she faces as before."""
    model = build([(0, 500, 1000, 500), (1000, 520, 1600, 520)], [("13", 700, 500), ("19", 1300, 520)])
    d, p = Danger(model, HER), model.platform_under(500, 500)
    green = seen("green-mushroom", 330, 500, p.id)
    low = seen("horny-mushroom", 720, 520, model.platform_under(1300, 520).id)
    assert stance.stance(500, 500, p, [green, low], facing="right") == ("throw", "right")
    assert stance.stance(500, 500, p, [green, low], facing="right", danger=d) == ("throw", "left")


def test_a_spot_keeps_clear_of_its_platforms_spawn_points_where_that_is_worth_it():
    """attack.spots._off_spawns (SPAWN_CLEAR): of the places tried as a spot along a platform, those 50 units or more from
    its spawn points of kinds that count (a monster back at its point is on her before she can shoot it); with none,
    the nearest ones past the knock room from an end with a short drop under it (here its left, 75 over the platform
    under it: Tree Dungeon's 24 over 25; its right a long drop); taken only within the touch's share of the best score,
    else the places as they were. Where nothing counts, as they were."""
    model = build([(176, 425, 310, 425), (44, 500, 276, 500)], [("13", 272, 425)])
    d, p = Danger(model, HER), model.platform_under(250, 425)
    xs = [231.0, 241.0, 251.0]  # GROUND_MARGIN from its ends, all within 50 of 272
    assert spots._off_spawns(model, p, d, xs, lambda x: 1.0 - abs(x - 231) / 1000) == [221.0]
    assert spots._off_spawns(model, p, d, xs, lambda x: 1.0 - abs(x - 231) / 50) == xs  # 221 scores 0.8: not worth it
    assert spots._off_spawns(model, p, d, [181.0, 191.0, 231.0], lambda x: 1.0) == [181.0, 191.0]
    assert spots._off_spawns(model, p, None, xs, lambda x: 1.0) == xs
