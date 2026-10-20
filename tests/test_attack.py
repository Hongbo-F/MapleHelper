"""Attack decisions from map positions: where a star reaches, which monster first, what to do about the
monsters around her, what an attack from a jump can hit, what stops a star, and where to stand to shoot
a platform."""

import math

import pytest

from maplehelper.engine import LEFT, RIGHT

from gamelab.attack import reach, spots
from gamelab.attack.stance import TOO_CLOSE, decide, pick_target, stance
from gamelab.localization import mapdata, monsters
from gamelab.localization.mapdata import Platform, Segment
from gamelab.localization.monsters import Monster, monster_platform
from gamelab.motion import physics
from gamelab.perception.mobs import Mob

PLAT = Platform(0, [Segment(0, 500, 1000, 500)])      # her platform
BELOW = Platform(1, [Segment(1100, 560, 1600, 560)])  # 60 below, to the right


def target(x, y=500, platform=0, kind="pig", bar=None):
    return Monster(Mob(kind, 0, 0, 30, 30, 0.9), x, y, platform, bar)


# --- one target (kill) ---------------------------------------------------------------------------

def test_throws_from_the_far_end_of_her_reach():
    assert decide(500, 500, PLAT, target(500 + reach.SHOT_RANGE - 10), []) == ("throw", RIGHT)
    assert decide(500, 500, PLAT, target(200), []) == ("throw", LEFT)


def test_shoots_a_monster_on_a_lower_platform_from_hers():
    assert decide(950, 500, PLAT, target(1250, 560, platform=1), []) == ("throw", RIGHT)


def test_walks_just_into_reach_when_too_far():
    move = decide(300, 500, PLAT, target(800), [])
    assert move[:2] == ("walk", RIGHT)
    assert reach.MIN_SHOT < 800 - (300 + move[2]) <= reach.SHOT_RANGE


def test_backs_off_from_a_monster_too_close_on_her_platform():
    assert decide(500, 500, PLAT, target(500 + TOO_CLOSE - 30), [])[:2] == ("walk", LEFT)


def test_a_monster_on_another_platform_is_no_threat_but_needs_room_to_throw():
    # Right above it: not a threat (it can't come up), but too close to throw at: walk off to throw
    # (the way there's room).
    move = decide(980, 500, PLAT, target(950, 560, platform=1), [])
    assert move[:2] == ("walk", LEFT)


def test_fights_when_cornered_between_monsters():
    t, behind = target(560), target(400)
    assert decide(500, 500, PLAT, t, [behind]) == ("throw", RIGHT)


def test_never_walks_off_her_platform():
    assert decide(900, 500, PLAT, target(1500, 560, platform=1), [])[0] == "skip"


def test_skips_targets_too_far_above_or_below():
    assert decide(500, 500, PLAT, target(700, 700, platform=2), []) == ("skip", "not level enough with her platform")


def test_shoots_a_monster_in_the_way_first():
    in_way = target(300)
    move = decide(100, 500, PLAT, target(950), [in_way])
    assert move == ("throw", RIGHT)


def test_cornered_she_fights_from_where_she_is():
    t = target(PLAT.right - 70)  # she's at the right end, the monster right next to her
    assert decide(PLAT.right - 30, 500, PLAT, t, []) == ("throw", LEFT)  # not jumping to its other side


# --- which one ---------------------------------------------------------------------------------------

def test_picks_close_monsters_on_her_platform_before_far_ones():
    near_me, far = target(650), target(1300, 560, platform=1)
    assert pick_target(500, 500, PLAT, [far, near_me]) is near_me


def test_finishes_hurt_monsters_first():
    fresh, hurt = target(700), target(750, bar=0.4)
    assert pick_target(500, 500, Platform(9, [Segment(0, 400, 1000, 400)]), [fresh, hurt]) is hurt


# --- a group (clear) ---------------------------------------------------------------------------------

def test_she_throws_toward_the_side_with_more_monsters_in_reach():
    group = [target(700), target(780)]
    lone = target(250)
    assert stance(500, 500, PLAT, group + [lone]) == ("throw", RIGHT)


def test_one_too_close_gets_a_jump_back_throw_not_a_claw_swipe():
    # Room behind her: jump away from it and throw back from the air.
    assert stance(500, 500, PLAT, [target(560), target(700)]) == ("jump_back", RIGHT)
    # Cornered at the left end, or another monster behind her: straight up. No crossing either way.
    assert stance(60, 500, PLAT, [target(110)]) == ("jump_throw", RIGHT)
    assert stance(500, 500, PLAT, [target(560), target(400)]) == ("jump_throw", RIGHT)


def test_she_keeps_working_the_side_she_faces():
    lone_right, pair_left = target(700), [target(300), target(250)]
    assert stance(500, 500, PLAT, [lone_right] + pair_left, facing=RIGHT) == ("throw", RIGHT)
    assert stance(500, 500, PLAT, [lone_right] + pair_left) == ("throw", LEFT)


def test_she_closes_in_only_as_far_as_needed_and_waits_when_nothing_is_there():
    act = stance(100, 500, PLAT, [target(900)])
    assert act[:2] == ("approach", RIGHT) and 900 - (100 + act[2]) == pytest.approx(reach.SHOT_RANGE - 40)
    assert stance(100, 500, PLAT, []) == ("wait",)


def test_a_monster_right_under_her_makes_her_step_away_not_throw():
    below = target(530, y=560, platform=1)       # 60 below, 30 to the side: a star misses it
    act = stance(500, 500, PLAT, [below])
    assert act[:2] == ("back", LEFT) and reach.can_shoot(530 - (500 - act[2]), 60)
    assert not reach.can_shoot(30, 60) and reach.can_shoot(30 + 150, 60) and not reach.can_shoot(170, -60)


def test_under_a_low_platform_she_does_not_jump_at_a_monster():
    # A jump would carry her up onto the platform overhead: step away, or (cornered) throw from the ground.
    assert stance(500, 500, PLAT, [target(560)], can_jump=lambda a, b: False) == ("back", LEFT, 120)
    assert stance(60, 500, PLAT, [target(110)], can_jump=lambda a, b: False) == ("throw", RIGHT)


# --- what an attack from a jump can hit ----------------------------------------------------------------

def test_an_attack_keyed_late_in_a_jump_starts_too_high_for_a_snail():
    # Keyed 100 ms after the jump key (the macro's F1 combo), she's 49 units up when the attack picks its
    # target (plus the lags), so its area's floor is 30 up: a snail (24 at least) is under it, the others
    # reach into it. In game the star went over the snail. The jump back's attack (85 ms) too.
    assert reach.air_floor(100) == pytest.approx(30.3, abs=0.1)
    assert reach.air_floor(physics.JUMP_BACK_ATTACK_MS) == pytest.approx(25.1, abs=0.1)
    tops = monsters.MONSTER_TOP
    for gap in (100, physics.JUMP_BACK_ATTACK_MS):
        assert not reach.air_hits(tops["snail"], gap)
        assert tops["snail"] < reach.air_floor(gap) + reach.AIR_MARGIN
    assert all(reach.air_hits(tops[k], 100) for k in ("pig", "slime", "orange-mushroom"))
    # A jump throw's attack is keyed 40 ms after the jump (killed a snail in game): every kind is hit.
    assert physics.JUMP_ATTACK_MS == 40
    assert reach.air_floor(physics.JUMP_ATTACK_MS) == pytest.approx(6.7, abs=0.1)
    assert all(reach.air_hits(top, physics.JUMP_ATTACK_MS) for top in tops.values())


def test_with_the_attack_keyed_late_a_cornered_snail_is_shot_from_the_ground(monkeypatch):
    monkeypatch.setattr(physics, "JUMP_ATTACK_MS", 100)
    snail = target(110, kind="snail")
    # Cornered at the left end (no room to step back far enough for a star): from where she is, the claw
    # swipes it, and every swipe hits. A jump throw's attack would go over it.
    assert stance(60, 500, PLAT, [snail]) == ("throw", RIGHT)
    assert stance(60, 500, PLAT, [snail], can_jump=lambda a, b: False) == ("throw", RIGHT)
    # With room behind her: never a jump back (its attack starts too high too), but a step back to where a
    # throw from the ground is a star, and then the star.
    act = stance(500, 500, PLAT, [target(560, kind="snail")])
    assert act == ("back", LEFT, reach.STAR_SURE - 60)
    assert stance(500 - act[2], 500, PLAT, [target(560, kind="snail")]) == ("throw", RIGHT)
    # Another monster behind her: no step back either.
    assert stance(500, 500, PLAT, [target(560, kind="snail"), target(400)]) == ("throw", RIGHT)
    # Pigs, slimes and Orange Mushrooms as before.
    for kind in ("pig", "slime", "orange-mushroom"):
        assert stance(500, 500, PLAT, [target(560, kind=kind)]) == ("jump_back", RIGHT)
        assert stance(60, 500, PLAT, [target(110, kind=kind)]) == ("jump_throw", RIGHT)


def test_with_the_attack_keyed_40_ms_after_the_jump_a_cornered_snail_gets_a_jump_throw():
    # Everything follows from the attack's timing: at 40 ms (JUMP_ATTACK_MS) the jump throw's attack starts
    # low enough.
    assert stance(60, 500, PLAT, [target(110, kind="snail")]) == ("jump_throw", RIGHT)
    # Never a jump back at a snail: its attack has to come after the turn in mid-air (85 ms).
    assert stance(500, 500, PLAT, [target(560, kind="snail")]) == ("jump_throw", RIGHT)
    assert stance(500, 500, PLAT, [target(560)]) == ("jump_back", RIGHT)
    # The kinds' sizes can come from the map's game data instead of the tables.
    assert stance(60, 500, PLAT, [target(110, kind="snail")], tops={"snail": 5.0}) == ("throw", RIGHT)


def test_monster_tables_match_the_game_data():
    """localization.monsters' tables against the boxes of the monsters' frames in the game data (as
    mapdata reads them with the map). Skipped without the game data copy."""
    from gamedata.paths import COPY
    if not (COPY / "Packs" / "Mob_00000.ms").is_file():
        pytest.skip("needs the game data copy (gamedata/data/copy)")
    data = mapdata._mob_bodies({"2", "7", "8", "9"})  # snail, slime, pig, Orange Mushroom
    for mob_id, kind in (("2", "snail"), ("7", "slime"), ("8", "pig"), ("9", "orange-mushroom")):
        body = data[mob_id]
        assert body["top"] == monsters.MONSTER_TOP[kind]
        assert body["flinch_top"] == monsters.MONSTER_FLINCH_TOP[kind]
        assert abs(body["half"] - monsters.MONSTER_HALF[kind]) <= 0.5
    assert monsters.bodies()["snail"].top == 24  # without a map: the tables


# --- where to shoot from -------------------------------------------------------------------------------

def test_coverage_and_spots_prefer_safe_platforms():
    tree = {
        "info": {"VRLeft": 0, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
        "foothold": {"1": {
            "1": {"1": {"x1": 0, "y1": 500, "x2": 400, "y2": 500, "prev": 0, "next": 0}},    # safe, 60 above
            "2": {"2": {"x1": 420, "y1": 560, "x2": 900, "y2": 560, "prev": 0, "next": 0}},  # hunting ground
            "3": {"3": {"x1": 300, "y1": 900, "x2": 1500, "y2": 900, "prev": 0, "next": 0}},  # far below
        }},
        "life": {"0": {"type": "m", "id": "8", "x": 600, "y": 560, "cy": 560, "rx0": 450, "rx1": 880},
                 "1": {"type": "m", "id": "8", "x": 700, "y": 560, "cy": 560, "rx0": 450, "rx1": 880}},
        "miniMap": {"width": 2000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4},
    }
    model = mapdata.parse(1, tree, "Test", {})
    ground = model.platform_under(600, 550)
    best = spots.spots(model, ground.id)[0]
    assert best.safe and model.platform(best.platform).y_at(best.x) == 500
    assert best.coverage == pytest.approx(spots.coverage(ground, best.x, 500), abs=1e-3)
    assert best.coverage > 0.35  # off her level the star needs 150-200 units sideways (reach.min_dx)


def test_monsters_are_placed_only_on_platforms_they_spawn_on():
    tree = {
        "info": {"VRLeft": 0, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
        "foothold": {"1": {
            "1": {"1": {"x1": 300, "y1": 458, "x2": 420, "y2": 458, "prev": 0, "next": 0}},  # safe, 42 above
            "2": {"2": {"x1": 200, "y1": 500, "x2": 900, "y2": 500, "prev": 0, "next": 0}},  # pigs spawn here
        }},
        "life": {"0": {"type": "m", "id": "8", "x": 600, "y": 500, "cy": 500, "rx0": 250, "rx1": 880}},
        "miniMap": {"width": 2000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4},
    }
    model = mapdata.parse(1, tree, "Test", {}, {"8": "Pig"})
    # Feet read 15 units too high: closer to the safe platform, but pigs only walk the lower one.
    assert monster_platform(model, "pig", 350, 465).y_at(350) == 500


# --- what stops a star -----------------------------------------------------------------------------------

def _fh(x1, y1, x2, y2, prev=0, nxt=0):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "prev": prev, "next": nxt}


def step_map():
    """Group (1, 1): floor A (0..200, y 500), the step wall at x 200 (from 500 up to 440), floor B
    (200..400, y 440), like platforms 4 and 2 on the real map. Group (1, 2): block C (450..600, y 400),
    its walls hanging 25 below its ends. Returns the model and A, B, C."""
    tree = {"info": {"VRLeft": -100, "VRTop": 0, "VRRight": 800, "VRBottom": 800},
            "foothold": {"1": {"1": {"1": _fh(0, 500, 200, 500, 0, 2), "2": _fh(200, 500, 200, 440, 1, 3),
                                     "3": _fh(200, 440, 400, 440, 2, 0)},
                               "2": {"4": _fh(450, 425, 450, 400, 0, 5), "5": _fh(450, 400, 600, 400, 4, 6),
                                     "6": _fh(600, 400, 600, 425, 5, 0)}}}}
    model = mapdata.parse(1, tree, "Steps", {})
    a, b, c = (next(p for p in model.platforms if p.left == left) for left in (0, 200, 450))
    return model, a, b, c


def test_a_wall_of_her_own_foothold_group_stops_a_star():
    model, a, b, c = step_map()
    assert a.group == b.group == (1, 1) and a.walls == b.walls == tuple(model.walls_of(a.group))
    assert [w.x1 for w in a.walls] == [200] and sorted(w.x1 for w in c.walls) == [450, 600]
    # Below the step: the wall rises 0..60 above her feet, across the line 28 above them.
    up = target(300, 440, platform=b.id)
    assert reach.can_shoot(300 - 100, -60) and not reach.can_hit(a, 100, 500, up)
    assert reach.wall_in_way(a, 100, 500, 300).x1 == 200
    assert reach.clear_span(a, 100, 500) == (-math.inf, 200)
    # Above it, the wall hangs below her feet: down past it the star gets there.
    assert reach.wall_in_way(b, 210, 440, 100) is None and reach.can_hit(b, 300, 440, target(100, 500, platform=a.id))
    assert reach.clear_span(b, 300, 440) == (-math.inf, math.inf)


def test_walls_of_other_groups_and_floors_let_stars_through():
    model, a, b, c = step_map()
    # From B at 300 (the line at y 412) to C's top, 40 above her: C's own wall at 450 hangs from 400 to 425,
    # across the line, but it's another group's (the side of the block a target stands on).
    assert any(min(w.y1, w.y2) <= 440 - reach.SHOT_H < max(w.y1, w.y2) for w in c.walls if w.x1 == 450)
    assert reach.wall_in_way(b, 300, 440, 560) is None and reach.can_hit(b, 300, 440, target(560, 400, platform=c.id))
    # Floors never stop one: from C down to B, past C's end and its hanging wall (C's own group).
    assert reach.wall_in_way(c, 470, 400, 300) is None


def test_a_jump_raises_the_line_a_star_runs_at():
    model, a, b, c = step_map()
    # A jump throw's attack starts 26 up (the line 54 above her feet): still under the 60 step's top. A jump
    # back's, 44 up (72): over it.
    assert reach.wall_in_way(a, 100, 500, 300, rise=reach.air_rise(physics.JUMP_ATTACK_MS)) is not None
    assert reach.wall_in_way(a, 100, 500, 300, rise=reach.air_rise(physics.JUMP_BACK_ATTACK_MS)) is None


def test_a_monster_is_never_past_its_platforms_end():
    model, a, b, c = step_map()
    # Read 9 units past A's end (where the step wall rises): on A, so on her side of it.
    near_end = target(209, 500, platform=a.id)
    assert reach.can_hit(a, 100, 500, near_end) and reach.wall_in_way(a, 100, 500, 209, target=a) is None
    assert reach.wall_in_way(a, 100, 500, 209) is not None  # its platform unknown: taken where it was seen
    # On B, right at the step's top: behind it.
    assert reach.wall_in_way(a, 100, 500, 200, target=b) is not None


def test_walls_block_can_be_switched_off_for_probes(monkeypatch):
    model, a, b, c = step_map()
    monkeypatch.setattr(reach, "WALLS_BLOCK", False)
    assert reach.wall_in_way(a, 100, 500, 300) is None and reach.can_hit(a, 100, 500, target(300, 440, platform=b.id))
    assert reach.clear_span(a, 100, 500) == (-math.inf, math.inf)
    assert reach.wall_between(a, 100, 500, 300) is not None  # what the shot log records either way


def test_what_stops_a_star_on_the_real_map():
    """Thicket Around the Beach II (skipped without its parsed game data): from platform 4 a star never
    reaches 2, behind the step wall of their group (3, 63); the shots that hit in the live logs, past other
    groups' walls and down past her own block's, are clear."""
    if not (mapdata.CACHE / "10000012.json").is_file():
        pytest.skip("needs the map's parsed game data (gamedata/data/out/maps)")
    model = mapdata.load(10000012)
    p = model.platform
    assert p(2).group == p(4).group == (3, 63)
    assert all(reach.wall_in_way(p(4), x, p(4).y_at(x), tx, target=p(2)) is not None
               for x in range(219, 670, 10) for tx in range(-309, 220, 10))
    for her, x, there, txs in ((19, -14, 18, (150, 300, 380)), (19, 52, 18, (300, 439)), (3, 706, 1, (870, 930)),
                               (3, 723, 1, (900, 950)), (14, 716, 17, (900, 1050)), (14, 737, 17, (918, 1088)),
                               (2, 200, 4, (300, 450)), (22, 16, 21, (-150, -300))):
        for tx in txs:
            assert reach.wall_in_way(p(her), x, p(her).y_at(x), tx, target=p(there)) is None, (her, x, there, tx)
