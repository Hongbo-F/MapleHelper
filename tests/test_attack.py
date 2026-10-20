"""Attack decisions from map positions: where a star reaches, which monster first, what to do about the
monsters around her, what the star's search from the ground picks, what an attack from a jump can hit,
what stops a star, and where to stand to shoot a platform."""

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


def test_a_throw_from_the_ground_is_a_star_only_past_the_claws_reach_by_the_monsters_size():
    # The claw's swing (90 ahead of her) takes a body that reaches into it: a pig up to 124 off by its
    # middle, a snail up to 109. Her decisions count on a star 12 past that (reach.claw_edge).
    edge = {kind: reach.claw_edge(half) for kind, half in monsters.MONSTER_HALF.items()}
    assert edge == {"pig": 136, "orange-mushroom": 134, "slime": 130, "snail": 121}
    assert reach.claw_takes(124, 0, 46, 34) and not reach.claw_takes(125, 0, 46, 34)
    assert reach.star_sure(19) == reach.STAR_SURE == 124 and reach.star_sure(34) == 139
    # A pig 115 away on her platform isn't a throw from the ground (the claw would swipe it, where by its
    # middle she used to count on a star from 100): a jump back with room behind her, straight up with a
    # monster there; one target (decide) she steps away from. From 140 it's a star.
    pig = target(615)
    assert stance(500, 500, PLAT, [pig]) == ("jump_back", RIGHT)
    assert stance(500, 500, PLAT, [pig, target(350)]) == ("jump_throw", RIGHT)
    assert decide(500, 500, PLAT, pig, []) == ("walk", LEFT, 120)
    assert stance(500, 500, PLAT, [target(640)]) == ("throw", RIGHT)
    assert decide(500, 500, PLAT, target(640), []) == ("throw", RIGHT)
    # A snail 115 away is too close as well (121), one 125 away isn't; by the map's game data (half 18.5)
    # one 120.8 away isn't either (120.5). A slime 125 away is (130), an Orange Mushroom 135 away isn't (134).
    assert stance(500, 500, PLAT, [target(615, kind="snail")]) == ("jump_throw", RIGHT)
    assert stance(500, 500, PLAT, [target(625, kind="snail")]) == ("throw", RIGHT)
    assert stance(500, 500, PLAT, [target(620.8, kind="snail")]) == ("jump_throw", RIGHT)
    assert stance(500, 500, PLAT, [target(620.8, kind="snail")], halves={"snail": 18.5}) == ("throw", RIGHT)
    assert stance(500, 500, PLAT, [target(625, kind="slime")]) == ("jump_back", RIGHT)
    assert stance(500, 500, PLAT, [target(635, kind="orange-mushroom")]) == ("throw", RIGHT)


# --- the star's search from the ground -------------------------------------------------------------------

def test_the_star_search_opens_one_in_four():
    # Bands 20 wide from 65 ahead of her out to 400, each one's window 5 higher and 5 lower than the last.
    assert reach.window(0) == (12, reach.SEARCH_TOP0) and reach.window(16) == (-68, reach.SEARCH_TOP0 + 80)
    assert reach.band_of(64) is None and reach.band_of(65) == 0 and reach.band_of(84.9) == 0
    assert reach.band_of(400) == 16 and reach.band_of(401) is None and reach.BANDS == 17
    # A body across several bands is taken in the first whose window it reaches into: 30 to 70 above her
    # feet, from 80 to 150 ahead (bands 0-4), it's under band 2's top (31), not band 1's (26).
    assert reach.search_band(80, 150, 30, 70) == 2
    assert reach.search_band(10, 60, 0, 40) is None and reach.search_band(405, 440, 0, 40) is None


def test_min_dx_is_where_the_search_first_picks_a_body():
    # The closed form against the search itself, every half unit sideways, for each kind's body (and a pig
    # flinching): from min_dx out to max_dx, its near side within 400.
    for dy in (-120, -60, -42, -20, 0, 20, 42, 60, 120):
        for top, half in ((26, 18.5), (41, 28.5), (46, 34.5), (55, 32), (54, 34)):
            picked = [a / 2 for a in range(0, 1000) if reach.picks(a / 2, dy, top, half)]
            if not picked:
                assert reach.min_dx(dy, top, half) == math.inf, (dy, top)
                continue
            assert abs(picked[0] - reach.min_dx(dy, top, half)) <= 0.5, (dy, top)
            assert picked[-1] == reach.max_dx(half), (dy, top)


def test_off_level_a_star_needs_more_room_the_smaller_the_monster():
    # By the game data's bodies (walking: MONSTER_HEIGHT; half widths 18.5-34.5), 60 below her:
    body = {"pig": mapdata.MobBody(38, 54, 34.5), "slime": mapdata.MobBody(40, 46, 28.5),
            "orange-mushroom": mapdata.MobBody(55, 65, 32), "snail": mapdata.MobBody(24, 33, 18.5)}
    below = {k: reach.min_dx(60, *reach.body_of(k, body)) for k in body}
    assert below == {"pig": 150.5, "slime": 176.5, "orange-mushroom": 113, "snail": 246.5}
    assert reach.min_dx(-60, *reach.body_of("pig", body)) == 190.5
    assert reach.min_dx(0, *reach.body_of("pig", body)) == 30.5 and reach.max_dx(34.5) == 434.5
    # A pig flinching after a hit (54 tall) is picked from 111, one standing (38) from 171. 120 above her
    # none is.
    assert reach.min_dx(60, 54, 34) == 111 and reach.min_dx(60, 38, 34) == 171
    assert all(reach.min_dx(-120, *reach.body_of(k)) == math.inf for k in body)
    # Her decisions: never closer than the claw's reach, and a kind she doesn't know is taken as a pig.
    assert reach.shot_dx(0) == reach.MIN_SHOT and reach.shot_dx(60, "orange-mushroom") == 113
    assert reach.shot_dx(60, "dragon") == reach.shot_dx(60, "pig") == 151
    assert reach.can_shoot(120, 60, kind="orange-mushroom") and not reach.can_shoot(120, 60)
    assert not reach.can_shoot(170, 60, kind="slime") and reach.can_shoot(180, 60, kind="slime")
    assert reach.can_shoot(410, 0, kind="snail") and not reach.can_shoot(420, 0, reach=500, kind="snail")


def test_from_the_ground_the_claw_first_then_the_search_band_by_band():
    # The claw's swing takes what it reaches on her level (90 ahead, into a pig's body up to 124 off), not
    # an Orange Mushroom right under her, 60 below.
    assert reach.claw_takes(120, 0, 46, 34) and not reach.claw_takes(130, 0, 46, 34)
    assert not reach.claw_takes(60, 60, 55, 32)
    # Then the search, its first band first: a slime 60 below 200 along (band 7) before a pig on her level
    # 300 along (band 10); one 60 below 150 along isn't picked at all (it needs 177).
    assert reach.ground_pick(30, 0, 26, 19) == (0, 30)
    assert reach.ground_pick(200, 60, 41, 28) == (8, 200) < reach.ground_pick(300, 0, 46, 34) == (11, 300)
    assert reach.ground_pick(150, 60, 41, 28) is None
    # An attack from a jump keeps to the area fixed to her feet: as high as she is then, a pig 60 above her
    # 100 along is in it; from the ground, the search doesn't pick it (it needs 191).
    assert reach.in_area(100, -60, 46, rise=reach.air_rise(physics.JUMP_ATTACK_MS), half=34)
    assert not reach.in_area(100, -60, 46, half=34)


def test_off_level_she_gives_a_star_the_room_its_kind_needs():
    # 60 below her: an Orange Mushroom 130 along is thrown at, and so is a pig 160 along; a slime 160 along
    # isn't (it needs 177): she steps back 40 past where it's picked, or (one target) walks to a band past it.
    mushroom = target(630, 560, platform=1, kind="orange-mushroom")
    assert decide(500, 500, PLAT, mushroom, []) == ("throw", RIGHT)
    assert stance(500, 500, PLAT, [mushroom]) == ("throw", RIGHT)
    assert stance(500, 500, PLAT, [target(660, 560, platform=1)]) == ("throw", RIGHT)
    slime = target(660, 560, platform=1, kind="slime")
    assert stance(500, 500, PLAT, [slime]) == ("back", LEFT, pytest.approx(177 + 40 - 160))
    move = decide(500, 500, PLAT, slime, [])
    assert move[:2] == ("walk", LEFT) and 660 - (500 - move[2]) == pytest.approx(177 + reach.SEARCH_STEP)


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
    assert act == ("back", LEFT, reach.star_sure(monsters.MONSTER_HALF["snail"]) - 60)
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
    dying = mapdata._mob_die_s({"2", "7", "8", "9"})
    for mob_id, kind in (("2", "snail"), ("7", "slime"), ("8", "pig"), ("9", "orange-mushroom")):
        body = data[mob_id]
        assert body["top"] == monsters.MONSTER_TOP[kind]
        assert body["flinch_top"] == monsters.MONSTER_FLINCH_TOP[kind]
        assert abs(body["half"] - monsters.MONSTER_HALF[kind]) <= 0.5
        assert dying[mob_id] == monsters.MONSTER_DIE_S[kind]  # its death animation: 0.66 s for a pig
    assert monsters.bodies()["snail"].top == 24  # without a map: the tables
    assert monsters.bodies()["snail"].die_s == 1.26
    # The map's cache is read again when it was made before bodies kept their death animations (a key of
    # its own, so code from before reads the new cache as it did).
    assert mapdata.load(10000012).mob_bodies["Pig"].die_s == 0.66


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


def test_she_doesnt_throw_at_or_walk_for_a_monster_behind_a_wall_of_her_group(monkeypatch):
    model, a, b, c = step_map()
    up = target(300, 440, platform=b.id)  # up on B: in reach by distance from A at 100, behind the step
    assert decide(100, 500, a, up, []) == ("skip", "behind a wall of her platform's foothold group")
    assert stance(100, 500, a, [up]) == ("wait",)
    assert pick_target(100, 500, a, [up, target(390, 440, platform=b.id, bar=0.5)]) is up  # none in reach: nearest
    # Down from B the star gets there: thrown, from B, at the monster on A.
    assert decide(300, 440, b, target(100, 500, platform=a.id), []) == ("throw", LEFT)
    # Where to shoot B from: nothing of it from A (by distance, from 191 sideways for a pig 60 above: 109 of
    # its 200 units).
    assert spots.coverage(b, 100, 500, a) == 0 and spots.coverage(b, 100, 500) == pytest.approx(0.545)
    # With the rule off (an in-game probe), as before.
    monkeypatch.setattr(reach, "WALLS_BLOCK", False)
    assert decide(100, 500, a, up, []) == ("throw", RIGHT) and stance(100, 500, a, [up]) == ("throw", RIGHT)


def test_no_spot_on_platform_4_for_platform_2():
    """On the real map (skipped without its parsed game data): by distance a spot on 4 (x 244) has 41% of
    platform 2 in reach, but it's all behind the step wall; 2 is shot from itself."""
    if not (mapdata.CACHE / "10000012.json").is_file():
        pytest.skip("needs the map's parsed game data (gamedata/data/out/maps)")
    model = mapdata.load(10000012)
    assert [sp.platform for sp in spots.spots(model, 2)] == [2]
    assert spots.coverage(model.platform(2), 244, model.platform(4).y_at(244)) == pytest.approx(0.415, abs=0.002)
    assert spots.coverage(model.platform(2), 244, model.platform(4).y_at(244), model.platform(4)) == 0


def test_a_monster_seen_past_its_platforms_end_is_placed_on_it():
    # Read from the screen, a pig seems 15 units past the right end of the platform it walks on: it's put
    # at the end (past the step at the end of platform 2 one would seem in front of the wall it's behind).
    import numpy as np
    from gamelab.localization.pose import Where
    from gamelab.localization.scene import Scene
    from gamelab.screen import UNITS_PER_HALF_PX
    tree = {"info": {"VRLeft": 0, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
            "foothold": {"1": {"2": {"2": _fh(200, 500, 900, 500)}}},
            "life": {"0": {"type": "m", "id": "8", "x": 600, "y": 500, "cy": 500, "rx0": 250, "rx1": 880}},
            "miniMap": {"width": 2000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4}}
    model = mapdata.parse(1, tree, "Test", {}, {"8": "Pig"})
    s = Scene(t=0.0, frame=np.zeros((10, 10, 3), np.uint8), tag=(500, 300), hp=1.0, mp=1.0,
              where=Where(800.0, 500.0, model.platforms[0], (0.0, 0.0)))
    seen = Mob("pig", round(500 + 115 / UNITS_PER_HALF_PX), s.floor_y - 15, 40, 30, 0.9)
    assert s.map_x(seen.x) > 910
    m = s.place(seen, model, None)
    assert (m.x, m.y, m.platform) == (900, 500, model.platforms[0].id)
