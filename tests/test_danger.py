"""The danger of a place (attack.danger) on synthetic maps: a spawn point between two places, a kind that hits hard
against a weak one, a monster on another platform that can't come over and one that hops, one walking a platform just
under hers, how far a fight keeps a monster off, and a look's monsters standing for their spawn points."""

import math

import pytest

from gamelab.attack import danger
from gamelab.attack.danger import HER_TALL, HOP_CHANCE, THREAT_S, Danger, Her, soon, touch_taken
from gamelab.localization import mapdata
from gamelab.localization.monsters import Monster
from gamelab.motion.flow import HER_HALF
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
