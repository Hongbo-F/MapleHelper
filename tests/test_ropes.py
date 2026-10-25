"""Ropes as places to wait out monsters (attack.ropes, attack.danger's on_rope), on synthetic maps: which kinds can
reach her on a rope (a walker under it only as high as its body, a hopper as high as its hop's frames, a flier
anywhere), and the ropes she can get onto from a platform and back, at what height and what it takes."""

import math

import pytest

from gamelab.attack import ropes
from gamelab.attack.danger import HER_TALL, THREAT_S, Danger, Her
from gamelab.localization import mapdata
from gamelab.motion.flow import HER_HALF

from test_danger import BODIES, DAMAGE, HOPS, NAMES, SPEEDS, seen

HER = Her(hp=752.0, level=24.0)
TALL = {"19": 52.0, "13": 52.0, "7": 66.0, "99": 52.0, "98": 40.0}  # game data: the mushrooms 52, a slime's hop 66


def build(footholds, lives, rope_list=(), fliers=()):
    """A map of these footholds (each a platform of its own group), spawn points (monster id, x, y) and ropes (x, top,
    bottom, from its platform); `fliers`: monster ids that fly ("98" stands for one: no map of hers has a flier)."""
    names = dict(NAMES, **{"98": "Flier"})
    tree = {"info": {"VRLeft": -500, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
            "foothold": {"1": {str(i): {str(i): {"x1": a, "y1": b, "x2": c, "y2": d, "prev": 0, "next": 0}}
                               for i, (a, b, c, d) in enumerate(footholds, 1)}},
            "life": {str(i): {"type": "m", "id": mob, "x": x, "y": y, "cy": y, "rx0": x, "rx1": x}
                     for i, (mob, x, y) in enumerate(lives)},
            "ladderRope": {str(i): {"x": x, "y1": top, "y2": bottom, "l": 0, "uf": int(up)}
                           for i, (x, top, bottom, up) in enumerate(rope_list, 1)},
            "miniMap": {"width": 2000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4}}
    ids = {mob for mob, _, _ in lives}

    def mine(table: dict) -> dict:
        return {i: v for i, v in table.items() if i in ids}
    bodies, speeds = dict(BODIES, **{"98": BODIES["99"]}), dict(SPEEDS, **{"98": -20})
    damage, hops = dict(DAMAGE, **{"98": 110}), dict(HOPS, **{"98": False})
    return mapdata.parse(1, tree, "Test", {}, names, mine({i: 41 for i in names}), mine(bodies), {}, mine(speeds),
                         mine(damage), mine(hops), mine({i: i in fliers for i in names}), mine(TALL))


def on(model, x, y):
    return model.platform_under(x, y).id


# A floor 0..800 at y 500, a platform over its left part (150..350 at y 250) the rope at 250 hangs from, down to 440:
# 60 above the floor, grabbed from it jumping straight up (her jump 77, and the arms).
FLOOR, OVER = (0, 500, 800, 500), (150, 250, 350, 250)
ROPE = (250, 250, 440, True)


def test_a_walker_under_a_rope_reaches_her_only_as_high_as_its_body_a_hopper_as_its_hop_a_flier_anywhere():
    model = build([FLOOR, OVER], [("19", 250, 500), ("7", 300, 500)], [ROPE])
    d, rope = Danger(model, HER), model.ropes[0]
    horny = d.threats["horny-mushroom"]
    assert horny.tall == 52.0 and d.threats["slime"].tall == 66.0
    # The Horny Mushroom right under the rope: its top 52 over the floor, her feet must be above it to be out of reach.
    assert d.on_rope(rope, 500 - 52 + 1) == pytest.approx(d.share("horny-mushroom"))  # her feet 51 up: touched
    assert d.on_rope(rope, 500 - 52 - 1) == 0.0                                        # 53 up: out of its reach
    assert d.on_rope(rope, 400) == 0.0 and d.on_rope(rope, 300) == 0.0
    # Hanging low on a rope down to 30 over the floor, it touches her once its walk brings its body under hers: from
    # 250 away, a walk of 250 less its half width and hers.
    far = build([FLOOR, OVER], [("19", 500, 500)], [(250, 250, 470, True)])
    df = Danger(far, HER)
    assert df.on_rope(far.ropes[0], 470) == pytest.approx(
        df.share("horny-mushroom") * math.exp(-(250 - 28 - HER_HALF) / 100 / THREAT_S))
    # A hopper reaches as high as its hop's frames (66): it would count if it counted (a slime: 1.2%, under DANGER_MIN).
    t, floor = d.threats["slime"], model.platform(on(model, 100, 500))
    assert d.walk_rope(t, floor, 250, rope, 500 - 60) is not None
    assert d.walk_rope(t, floor, 250, rope, 500 - 70) is None
    # A flier goes straight at her, whatever the floors: 100 off sideways and level with her, 100 - 28 - 15 to go.
    flier = build([FLOOR, OVER], [("98", 350, 500)], [ROPE], fliers={"98"})
    dfl = Danger(flier, HER)
    t = dfl.threats["flier"]
    assert t.flies
    way = dfl.walk_rope(t, flier.platform(on(flier, 100, 500)), 350, flier.ropes[0], 500 - t.tall / 2 + HER_TALL / 2)
    assert way == (pytest.approx(100 - 28 - HER_HALF), 1.0)
    assert dfl.on_rope(flier.ropes[0], 300) > 0  # high up the rope too, unlike a walker


def test_the_ropes_she_can_get_onto_from_a_platform_and_back_at_their_safest_height():
    model = build([FLOOR, OVER], [("19", 250, 500)], [ROPE])
    d, floor, over = Danger(model, HER), on(model, 100, 500), on(model, 250, 250)
    (under,) = ropes.refuges(model, d, floor)
    assert not under.from_top and under.x == 250 and under.platform == floor
    # Grabbed at the top of her jump (feet 72 up: 428), held up past it, out of the Horny Mushroom's reach and its
    # slack (52 + HANG_SLACK under her feet) and below TOP_KEEP from its top.
    assert 250 + ropes.TOP_KEEP <= under.y <= 428 - ropes.GRAB_CLIMB and under.danger == 0.0
    assert 500 - under.y >= 52 + ropes.HANG_SLACK
    assert 0.3 < under.on_s < 1.5 and 0.2 < under.off_s < 1.5
    (top,) = ropes.refuges(model, d, over)  # from the platform it hangs from: down onto it, and back up over its top
    assert top.from_top and top.platform == over and 250 < top.y <= 440 - ropes.END_KEEP
    # A rope at a platform's very end (ROPE_IN) isn't one: letting go of it she came down beside it.
    edge = build([FLOOR, OVER], [("19", 250, 500)], [(795, 250, 440, False)])
    assert ropes.refuges(edge, Danger(edge, HER), on(edge, 100, 500)) == []
    # Nor one out of her reach above her (95 up, less GRAB_MARGIN).
    high = build([FLOOR, OVER], [("19", 250, 500)], [(250, 250, 400, False)])
    assert ropes.refuges(high, Danger(high, HER), on(high, 100, 500)) == []


def test_how_high_a_kind_reaches_and_whether_it_flies_come_with_the_map():
    from gamedata.paths import COPY
    if not (COPY / "Packs" / "Mob_00000.ms").is_file():
        pytest.skip("needs the game data copy (gamedata/data/copy)")
    data = mapdata._mob_touch({"7", "13", "19"})  # a slime (its hop's frames), the Green and Horny Mushrooms
    assert data["mob_tall"] == {"7": 66.0, "13": 52.0, "19": 52.0}
    assert data["mob_flies"] == {"7": False, "13": False, "19": False}
