"""Attack decisions from map positions: where a star reaches, which monster first, what to do about the
monsters around her, what the star's search from the ground picks, what an attack from a jump can hit,
what stops a star, and where to stand to shoot a platform."""

import math

import pytest

from maplehelper.engine import LEFT, RIGHT

from gamelab.attack import reach, spots
from gamelab.attack import stance as stance_mod
from gamelab.attack.stance import TOO_CLOSE, decide, pick_target, stance
from gamelab.localization import mapdata, monsters
from gamelab.localization.mapdata import Platform, Segment
from gamelab.localization.monsters import Monster, monster_platform
from gamelab.motion import physics
from gamelab.perception.mobs import Mob


def no_room(act: tuple) -> tuple:
    """A stance's act with its jump throw's jitter room left out (attack.stance.JumpPlan.room, from 2026-11-01: a jump
    throw up close's own), for comparing its plan as before."""
    plan = stance_mod.plan_of(act)
    return act if plan is None else (*act[:2], stance_mod.replace(plan, room=None), *act[3:])

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
    assert reach.shot_dx(0) < 800 - (300 + move[2]) <= reach.SHOT_RANGE


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


def test_one_too_close_she_steps_to_where_her_next_throw_takes_it():
    # A pig 60 off: from the ground the claw swipes it, and from a jump in place nothing is picked until the body's far
    # side is past the star's search's first band (reach.AIR_MODEL 'search'): a pig, its top 12 over her feet 40 ms into
    # the jump, only from the search's third band, 78 ahead (88 with CLEAR_MARGIN: stance.jump_reach). Standing, it's
    # stepped from to where her next throw surely takes it (a jump in place after a tap on the ground, 35 back), not
    # jumped back from (the jump back's attack, after its turn in the air, picks one on her level only from about 145).
    assert stance_mod.jump_reach(38, 34) == 88
    assert stance(500, 500, PLAT, [target(560), target(700)])[:3] == ("back", LEFT, 35)
    # Cornered at the left end, or another monster behind her: straight up all the same, tapping the arrow on the ground
    # first if she must turn (its attack 40 ms into the jump: an air turn's, 120 ms, starts 30 units higher), as by her
    # keys she faces it at once; under CLAW_ONLY (15: on her) the claw from the ground. No crossing either way.
    tap = ("jump_throw", RIGHT, stance_mod.JumpPlan(40, 0, False, "tap first"))
    assert no_room(stance(60, 500, PLAT, [target(110)])) == tap
    assert stance_mod.plan_of(stance(60, 500, PLAT, [target(110)])).room is None  # (not sure: no jitter)
    assert stance(60, 500, PLAT, [target(110)], facing=RIGHT)[:2] == ("jump_throw", RIGHT)
    assert no_room(stance(500, 500, PLAT, [target(560), target(400)])) == tap
    assert stance(60, 500, PLAT, [target(70)], facing=RIGHT) == ("throw", RIGHT)


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
    assert not reach.can_shoot(30, 60) and reach.can_shoot(30 + 150, 60)
    assert not reach.can_shoot(100, -60) and reach.can_shoot(120, -60)  # 60 above her, a pig from 111


def test_under_a_low_platform_she_does_not_jump_at_a_monster():
    # A jump would carry her up onto the platform overhead: step away, to where a throw from the ground is a star (a
    # pig standing 60 off: 55 back, 115 off, past its claw's edge, 114, and the 111 a tap first leaves it at), or
    # (cornered) throw from the ground.
    assert stance(500, 500, PLAT, [target(560)], can_jump=lambda a, b: False)[:3] == ("back", LEFT, 55)
    assert stance(60, 500, PLAT, [target(110)], can_jump=lambda a, b: False) == ("throw", RIGHT)


def test_a_throw_from_the_ground_is_a_star_only_past_the_claws_reach_by_the_monsters_size():
    # Each of the claw's two stab stances (one per attack) takes a body that reaches into its box, cut to 65
    # ahead of her: on her level a pig up to 99 off by its middle, a snail up to 84. Her decisions count on a
    # star 12 past that (reach.claw_edge).
    assert {s: reach.claw_box(s) for s in reach.CLAW_STANCES} == {"stabO1": (20, 65, 15, 30),
                                                                  "stabO2": (12, 65, 16, 30)}
    edge = {kind: reach.claw_edge(half) for kind, half in monsters.MONSTER_HALF.items()}
    assert edge == {"pig": 111, "orange-mushroom": 109, "slime": 105, "snail": 96}
    for s in reach.CLAW_STANCES:
        assert reach.claw_takes(99, 0, 46, 34, s) and not reach.claw_takes(100, 0, 46, 34, s)
    assert reach.claw_chance(99, 0, 46, 34) == 1 and reach.claw_chance(100, 0, 46, 34) == 0
    assert reach.star_sure(19) == reach.STAR_SURE == 99 and reach.star_sure(34) == 114
    # A pig 105 away on her platform isn't a throw from the ground (the claw would swipe it, where by its
    # middle she used to count on a star from 100): a jump in place, whose attack surely picks it (from 88), tapping
    # first as she doesn't surely face it; one target (decide) she steps away from. From 115 it's a star.
    pig = target(605)
    tap = ("jump_throw", RIGHT, stance_mod.JumpPlan(40, 0, False, "tap first"))
    assert no_room(stance(500, 500, PLAT, [pig])) == tap
    room = stance_mod.plan_of(stance(500, 500, PLAT, [pig])).room  # (sure: its jitter's, 40 ms and later, each sure)
    assert room[0] == 40 and 40 <= room[1] <= 40 + reach.JITTER_FIRST_MS
    assert no_room(stance(500, 500, PLAT, [pig, target(350)])) == tap
    assert stance(500, 500, PLAT, [pig], facing=RIGHT)[:2] == ("jump_throw", RIGHT)
    assert decide(500, 500, PLAT, pig, []) == ("walk", LEFT, 120)
    assert stance(500, 500, PLAT, [target(615)]) == ("throw", RIGHT)
    assert decide(500, 500, PLAT, target(615), []) == ("throw", RIGHT)
    # A snail 90 away is too close as well (96: she steps back from it), one 100 away isn't; by the map's game
    # data (half 18.5) one 95.8 away isn't either (95.5). A slime 100 away is (105), an Orange Mushroom 110 away
    # isn't (109).
    assert stance(500, 500, PLAT, [target(590, kind="snail")])[:2] == ("back", LEFT)
    assert stance(500, 500, PLAT, [target(600, kind="snail")]) == ("throw", RIGHT)
    assert stance(500, 500, PLAT, [target(595.8, kind="snail")])[:2] == ("back", LEFT)
    assert stance(500, 500, PLAT, [target(595.8, kind="snail")], halves={"snail": 18.5}) == ("throw", RIGHT)
    assert no_room(stance(500, 500, PLAT, [target(600, kind="slime")])) == tap
    assert stance(500, 500, PLAT, [target(610, kind="orange-mushroom")]) == ("throw", RIGHT)


def test_the_claw_takes_what_its_stance_reaches_on_her_level_or_off_it():
    # No gate for her level: a body that reaches 15-30 above her feet is in the boxes (a pig 20 below her, its
    # top 26 above her feet; an Orange Mushroom 40 below, its top 15 up, just), one that doesn't isn't (an
    # Orange Mushroom 42 below, its top 13 up; a pig 42 above, its feet over the boxes' tops). The boxes differ only in slivers: a snail within a unit of her
    # middle is in stabO2's alone (it starts 12 ahead of her, stabO1's 20), so half the attacks swipe it.
    assert reach.claw_chance(60, 20, 46, 34) == 1
    assert reach.claw_chance(60, 42, 55, 32) == 0 and reach.claw_chance(30, -42, 46, 34) == 0
    assert reach.claw_chance(50, 40, 55, 32) == 0.5 and reach.claw_chance(50, 41, 55, 32) == 0
    assert reach.claw_chance(0.5, 0, 26, 19) == 0.5 and reach.claw_chance(1.5, 0, 26, 19) == 1
    assert reach.claw_dx(0) == 99 and reach.claw_dx(0, 26, 19) == 84
    assert reach.claw_dx(60) == reach.claw_dx(-42) == 0


# --- the star's search from the ground -------------------------------------------------------------------

def test_the_star_search_opens_one_in_four():
    # Bands 20 wide from 65 ahead of her out to 400, each one's window reaching a unit up and one down from
    # SHOT_H (28 above her feet) for every 4 its near edge is ahead of her, as the client has it: 12 to 44,
    # then each 5 higher and 5 lower than the last, to -68 to 124 (meowdb's Lucky Seven figure).
    assert reach.window(0) == (12, 44) == (reach.SEARCH_FLOOR0, reach.SEARCH_TOP0)
    assert reach.window(16) == (-68, 124) and reach.spread(16) == 96
    assert all(reach.window(k) == (reach.SHOT_H - reach.spread(k), reach.SHOT_H + reach.spread(k))
               for k in range(reach.BANDS))
    assert reach.window(1, top0=21) == (7, 26)  # the logs' lower top, opening as the client's does
    assert reach.band_of(64) is None and reach.band_of(65) == 0 and reach.band_of(84.9) == 0
    assert reach.band_of(400) == 16 and reach.band_of(401) is None and reach.BANDS == 17
    # A body across several bands is taken in the first whose window it reaches into: 50 to 90 above her
    # feet, from 80 to 150 ahead (bands 0-4), it's under band 2's top (54), not band 1's (49).
    assert reach.search_band(80, 150, 50, 90) == 2
    assert reach.search_band(10, 60, 0, 40) is None and reach.search_band(405, 440, 0, 40) is None
    # Overlap is strict, as the client's (in_step): a body whose far side is right at band 4's near edge (145)
    # isn't in it, nor one whose near side is at 400 in the last band.
    assert reach.search_band(77, 145, 60, 106) is None and reach.search_band(77, 145.5, 60, 106) == 4
    assert reach.search_band(20, 65, 12, 40) is None and reach.search_band(400, 430, 0, 40) is None
    assert reach.search_band(399.5, 430, 0, 40) == 16


def test_min_dx_is_where_the_search_first_picks_a_body():
    # The closed form against the search itself, every half unit sideways, for each kind's body (and a pig
    # flinching): from just past min_dx out to just short of max_dx, its near side within 400 (overlap is strict).
    for dy in (-120, -60, -42, -20, 0, 20, 42, 60, 120):
        for top, half in ((26, 18.5), (41, 28.5), (46, 34.5), (55, 32), (54, 34)):
            picked = [a / 2 for a in range(0, 1000) if reach.picks(a / 2, dy, top, half)]
            if not picked:
                assert reach.min_dx(dy, top, half) == math.inf, (dy, top)
                continue
            assert reach.min_dx(dy, top, half) < picked[0] <= reach.min_dx(dy, top, half) + 0.5, (dy, top)
            assert reach.max_dx(half) - 0.5 <= picked[-1] < reach.max_dx(half), (dy, top)


def test_off_level_a_star_needs_more_room_the_smaller_the_monster():
    # By the game data's bodies (walking: MONSTER_HEIGHT; half widths 18.5-34.5), 60 below her:
    body = {"pig": mapdata.MobBody(38, 54, 34.5), "slime": mapdata.MobBody(40, 46, 28.5),
            "orange-mushroom": mapdata.MobBody(55, 65, 32), "snail": mapdata.MobBody(24, 33, 18.5)}
    below = {k: reach.min_dx(60, *reach.body_of(k, body)) for k in body}
    assert below == {"pig": 150.5, "slime": 176.5, "orange-mushroom": 113, "snail": 246.5}
    assert reach.min_dx(-60, *reach.body_of("pig", body)) == 110.5
    assert reach.min_dx(0, *reach.body_of("pig", body)) == 30.5 and reach.max_dx(34.5) == 434.5
    # A pig flinching after a hit (54 tall) is picked from 111, one standing (38) from 171. 120 above her the
    # last bands pick each kind (a pig from 351), out of band (in_band: 60 is measured) but where her decisions
    # take the search's own reach above her (reach.UP_BY_SEARCH, on by the user's choice: tested in test_jump_reach).
    assert reach.min_dx(60, 54, 34) == 111 and reach.min_dx(60, 38, 34) == 171
    assert reach.min_dx(-120, *reach.body_of("pig")) == 351 and not reach.in_band(-120)
    assert reach.can_shoot(360, -120) and not reach.can_shoot(340, -120)
    # Her decisions: never where the claw would swipe it (on her level a pig to 99), and a kind she doesn't
    # know is taken as a pig.
    assert reach.shot_dx(0) == 99 and reach.shot_dx(0, "snail") == 84
    assert reach.shot_dx(60, "orange-mushroom") == 113
    assert reach.shot_dx(60, "dragon") == reach.shot_dx(60, "pig") == 151
    assert reach.can_shoot(120, 60, kind="orange-mushroom") and not reach.can_shoot(120, 60)
    assert not reach.can_shoot(170, 60, kind="slime") and reach.can_shoot(180, 60, kind="slime")


def test_between_her_level_and_60_below_her_decisions_keep_to_the_old_ramp(monkeypatch):
    # 42 below her (0 -> 1) the search picks a pig from 71, an Orange Mushroom from 33, a slime from 97, but the
    # only throws logged there missed pigs 124-125 off and an Orange Mushroom 91 off. Her decisions keep to the
    # old ramp there (126) until a probe settles it; the search alone with the switch off. 60 below, and above
    # her, the search alone either way: 42 above her a pig is picked from 31, its feet just under the first
    # window's top (44).
    assert reach.ramp_dx(42) == 126 and reach.ramp_dx(15) == reach.ramp_dx(60) == reach.ramp_dx(-42) == 0
    assert [reach.shot_dx(42, k) for k in ("pig", "orange-mushroom", "slime", "snail")] == [126, 126, 126, 166]
    assert not reach.can_shoot(120, 42, kind="pig") and reach.can_shoot(130, 42, kind="pig")
    assert reach.shot_dx(60, "orange-mushroom") == 113 and reach.shot_dx(-42, "pig") == 31
    monkeypatch.setattr(reach, "RAMP_BELOW", False)
    assert [reach.shot_dx(42, k) for k in ("pig", "orange-mushroom", "slime", "snail")] == [71, 33, 97, 166]
    assert reach.can_shoot(410, 0, kind="snail") and not reach.can_shoot(420, 0, reach=500, kind="snail")


def test_the_claw_is_a_model_of_stances_their_cut_and_a_level_gate():
    # reach.CLAW is the client's (the two stabs, cut to 65 ahead); another model is tried without changing it,
    # as the probes' scorer does. The one box reach had until 2026-10-20 (90 ahead, 17 below to 62 above her
    # feet, on her level only) swipes a pig out to 124 on her level and none 20 below; the throw's swings, cut
    # to 65, reach a pig 60 below in one stance of three (swingO2, down to 17 below her feet), the stabs never.
    old = reach.Claw((("swing", ((-90, -62), (0, 17))),), 90, level=reach.LEVEL)
    assert reach.claw_box("swing", old) == (0, 90, -17, 62)
    assert reach.claw_chance(124, 0, 46, 34, old) == 1 and reach.claw_chance(125, 0, 46, 34, old) == 0
    assert reach.claw_chance(60, 20, 46, 34, old) == 0 and reach.claw_chance(60, 20, 46, 34) == 1
    swings = reach.Claw((("swingO1", ((-88, -62), (-18, -6))), ("swingO2", ((-73, -50), (5, 17))),
                         ("swingO3", ((-102, -50), (-22, -10)))), reach.CLAW_FAR)
    assert reach.claw_box("swingO2", swings) == (-5, 65, -17, 50)
    assert reach.claw_chance(90, 60, 46, 34, swings) == pytest.approx(1 / 3) and reach.claw_chance(90, 60, 46, 34) == 0
    # Each monster's chance of the swipe and of the star apart: a snail by her middle is in stabO2's box alone,
    # so half the attacks swipe it and the other half throw the pig in the first band a star.
    assert reach.attack_chances([(0.5, 0, 26, 19), (200, 0, 46, 34)]) == [(0.5, 0.0), (0.0, 0.5)]
    assert reach.attack_chances([(90, 60, 46, 34), (200, 0, 46, 34)], swings) == [(1 / 3, 0.0), (0.0, 2 / 3)]


def test_from_the_ground_the_claw_first_then_the_search_band_by_band():
    # The claw takes what its stance reaches (65 ahead, into a pig's body up to 99 off), not an Orange
    # Mushroom right under her, 60 below.
    assert reach.claw_chance(95, 0, 46, 34) == 1 and reach.claw_chance(105, 0, 46, 34) == 0
    assert reach.claw_chance(60, 60, 55, 32) == 0
    # Then the search, its first band first: a slime 60 below 200 along (band 7) before a pig on her level
    # 300 along (band 10); one 60 below 150 along isn't picked at all (it needs 177).
    assert reach.ground_pick(30, 0, 26, 19) == (0, 30)
    assert reach.ground_pick(200, 60, 41, 28) == (8, 200) < reach.ground_pick(300, 0, 46, 34) == (11, 300)
    assert reach.ground_pick(150, 60, 41, 28) is None
    # Of the monsters it can take, each stance swipes the nearest in its box; else any monster in the search's
    # first band takes the star, as likely as another (the client's order there can't be seen).
    assert reach.take_chances([(50, 0, 46, 34), (60, 0, 26, 19)]) == [1, 0]
    assert reach.take_chances([(200, 0, 46, 34), (205, 0, 46, 34), (300, 0, 46, 34)]) == [0.5, 0.5, 0]
    assert reach.take_chances([(0.5, 0, 26, 19), (200, 0, 46, 34)]) == [0.5, 0.5]
    assert reach.take_chances([(150, 60, 41, 28)]) == [0]
    # An attack from a jump keeps to the area fixed to her feet: as high as she is then, a pig 60 above her
    # 100 along is in it; from the ground, the search doesn't pick it (it needs 111).
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
    assert stance(500, 500, PLAT, [slime])[:3] == ("back", LEFT, pytest.approx(177 + 40 - 160))
    move = decide(500, 500, PLAT, slime, [])
    assert move[:2] == ("walk", LEFT) and 660 - (500 - move[2]) == pytest.approx(177 + reach.SEARCH_STEP)


# --- what an attack from a jump can hit ----------------------------------------------------------------

def test_from_a_jump_the_star_search_alone_picks_from_her_feet_then():
    # reach.AIR_MODEL 'search' (2026-10-27): an attack starting in the air picks as the star's search from her feet
    # then, no claw and no area: a Green or Horny Mushroom (51 tall, 28 wide either side) on her level only once its
    # far side is past the first band's 65 ahead, by AIR_MARGIN: from 44 ahead keyed 40 ms after the jump key (her feet
    # 21-26 up); 10 below her takeoff (a slope) from 64, 15 below from 84; keyed 120 ms (an air turn's: 52-56 up) only
    # from 144 on her level, from 104 where it's 10 above her.
    def first(dy, key):
        return next(a for a in range(400) if reach.air_sure(a, dy, 51, 28, key))
    assert [first(dy, 40) for dy in (-10, 0, 10, 15)] == [44, 44, 64, 84]
    assert [first(dy, 120) for dy in (-10, 0)] == [104, 144]
    assert not reach.air_takes(20, 26, 51, 28) and reach.air_takes(50, 26, 51, 28)
    # Of the kinds on her level, keyed 40 ms after the jump the first band takes a mushroom only (air_hits); a turning
    # jump's attack goes 120 ms after the jump key whatever the kind (turn_key_ms).
    tops = monsters.MONSTER_TOP
    assert reach.air_hits(51, 40) and not any(reach.air_hits(tops[k], 40) for k in ("pig", "slime", "snail"))
    assert reach.turn_key_ms(38) == reach.turn_key_ms(24) == physics.AIR_TURN_ATTACK_MS


def test_an_attack_keyed_late_in_a_jump_starts_too_high_for_a_snail(monkeypatch):
    # With the area ('widening', from 2026-10-25 to 10-27): keyed 100 ms after the jump key (the macro's F1 combo),
    # she's 49 units up when the attack picks its target (plus the lags), so its area's floor is 30 up: a snail (24 at
    # least) is under it, the others reach into it. In game the star went over the snail. The jump back's attack (85
    # ms) too.
    monkeypatch.setattr(reach, "AIR_MODEL", "widening")
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
    # throw from the ground is a star (reach.star_sure, 99: in CLEAR_STEP steps, 45 to 105 off; and as far as the snail
    # walks at her meanwhile, with a pace: stance.Pace), and then the star.
    act = stance(500, 500, PLAT, [target(560, kind="snail")])
    assert act[:3] == ("back", LEFT, 45) and 560 - (500 - act[2]) >= reach.star_sure(monsters.MONSTER_HALF["snail"])
    assert stance(500 - act[2], 500, PLAT, [target(560, kind="snail")]) == ("throw", RIGHT)
    pace = stance_mod.Pace(0.3, lambda m: -monsters.MONSTER_SPEED["snail"], {"snail": monsters.MONSTER_SPEED["snail"]})
    walked = stance(500, 500, PLAT, [target(560, kind="snail")], pace=pace)
    assert walked[:2] == ("back", LEFT) and walked[2] > act[2]
    # Another monster behind her: no step back either.
    assert stance(500, 500, PLAT, [target(560, kind="snail"), target(400)]) == ("throw", RIGHT)
    # Pigs, slimes and Orange Mushrooms standing: a step to where her next throw takes each (a jump in place's attack
    # keyed this late picks none of them up close), cornered the claw within CLAW_ONLY, else the jump in place.
    for kind in ("pig", "slime", "orange-mushroom"):
        assert stance(500, 500, PLAT, [target(560, kind=kind)])[:2] == ("back", LEFT)
        assert stance(60, 500, PLAT, [target(80, kind=kind)]) == ("throw", RIGHT)


def test_a_snail_too_close_is_never_jump_thrown_at_but_stepped_back_from_or_stabbed(monkeypatch):
    # The attack's area keyed 40 ms after the jump (JUMP_ATTACK_MS) takes a snail in, but in game on 2026-10-21
    # jump throws mostly missed snails (5 of 26 under 46 units off hurt one): cornered at the left end, the claw
    # stabs it from the ground, and every stab hits.
    assert stance(60, 500, PLAT, [target(110, kind="snail")]) == ("throw", RIGHT)
    # With room behind her: a step back past where a throw from the ground is a star, then the star.
    assert stance(500, 500, PLAT, [target(560, kind="snail")])[:2] == ("back", LEFT)
    assert stance(500, 500, PLAT, [target(560)])[:3] == ("back", LEFT, 35)  # a pig: to where a jump in place takes it
    # On a short platform (168 units, as 3, 20 and 22) with a monster about she keeps 50 from its ends: where only
    # its far end has the room she stabs it (SNAIL_FAR_END: in the simulator the snail was soon back at her there,
    # and a touch knocked her off the end).
    short = Platform(3, [Segment(681, -925, 849, -925)])
    snail = target(820, -925, platform=3, kind="snail")
    assert stance(780, -925, short, [snail]) == ("throw", RIGHT)
    assert stance(745, -925, short, [target(770, -925, platform=3, kind="snail")]) == ("throw", RIGHT)
    # With SNAIL_FAR_END the step back goes on to 40 from its end (SMALL_STEP_END).
    monkeypatch.setattr(stance_mod, "SNAIL_FAR_END", True)
    act = stance(780, -925, short, [snail])
    assert act[:2] == ("back", LEFT) and short.left + stance_mod.SMALL_STEP_END <= 780 - act[2] < short.left + 50
    assert stance(780 - act[2], -925, short, [snail]) == ("throw", RIGHT)
    assert stance(745, -925, short, [target(770, -925, platform=3, kind="snail")]) == ("throw", RIGHT)  # nor there
    # Never a jump back at a snail either: its attack has to come after the turn in mid-air (85 ms).
    # JUMP_THROW_SNAILS brings back the jump throw at one, cornered or not.
    monkeypatch.setattr(stance_mod, "JUMP_THROW_SNAILS", True)
    assert stance(60, 500, PLAT, [target(110, kind="snail")])[:2] == ("jump_throw", RIGHT)
    assert stance(500, 500, PLAT, [target(560, kind="snail")], facing=RIGHT)[:2] == ("jump_throw", RIGHT)
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
    stats = mapdata._mob_speeds({"2", "7", "8", "9"})
    assert stats == {"2": -65, "7": -20, "8": 0, "9": 0}
    for mob_id, kind in (("2", "snail"), ("7", "slime"), ("8", "pig"), ("9", "orange-mushroom")):
        assert mapdata.mob_walk_speed(stats[mob_id], {"walkSpeed": 125.0}) == monsters.MONSTER_SPEED[kind]


def test_monster_speeds_come_with_the_map(tmp_path, monkeypatch):
    """How fast a monster walks: the map's walkSpeed times 1 + its speed stat / 100. A map cached before the
    stats were kept gets them added (here from a stand-in for the game data)."""
    import json
    tree = {"info": {}, "foothold": {"1": {"1": {"1": {"x1": 0, "y1": 500, "x2": 900, "y2": 500, "prev": 0, "next": 0}}}},
            "life": {"0": {"type": "m", "id": "2", "x": 600, "y": 500, "cy": 500}}, "miniMap": {}}
    model = mapdata.parse(1, tree, "Test", {"walkSpeed": 120.0}, {"2": "Snail"}, mob_speeds={"2": -65})
    assert model.mob_speeds == {"Snail": pytest.approx(42.0)}
    assert monsters.speeds(model)["snail"] == pytest.approx(42.0) and monsters.speeds()["snail"] == 43.75
    raw = {"id": 1, "name": "Test", "tree": tree, "physics": {"walkSpeed": 125.0}, "mob_names": {"2": "Snail"},
           "mob_exp": {"2": 2}, "mob_bodies": {}, "mob_die_s": {}}
    monkeypatch.setattr(mapdata, "CACHE", tmp_path)
    (tmp_path / "1.json").write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(mapdata, "_mob_speeds", lambda ids: {i: -65.0 for i in ids})
    assert mapdata.load(1).mob_speeds == {"Snail": 43.75}
    assert json.loads((tmp_path / "1.json").read_text(encoding="utf-8"))["mob_speeds"] == {"2": -65.0}
    assert [p.name for p in tmp_path.iterdir()] == ["1.json"]


def test_touch_damage_hops_and_spawn_points_come_with_the_map(tmp_path, monkeypatch):
    """A monster's touch damage (PADamage) and whether it hops (a jump action in its Mob data), by name, and each spawn
    point with its platform. A map cached before those were kept gets them as it's read (here from a stand-in for the
    game data), and the cache stays as it was."""
    import json
    tree = {"info": {}, "foothold": {"1": {"1": {"1": {"x1": 0, "y1": 500, "x2": 900, "y2": 500, "prev": 0, "next": 0}}}},
            "life": {"0": {"type": "m", "id": "2", "x": 600, "y": 500, "cy": 500},
                     "1": {"type": "m", "id": "7", "x": 300, "y": 500, "cy": 500},
                     "2": {"type": "m", "id": "7", "x": 300, "y": 900, "cy": 900}},  # on no platform
            "miniMap": {}}
    model = mapdata.parse(1, tree, "Test", {}, {"2": "Snail", "7": "Slime"}, mob_damage={"2": 3, "7": 13},
                          mob_hops={"2": False, "7": True})
    assert model.mob_damage == {"Snail": 3.0, "Slime": 13.0} and model.mob_hops == {"Snail": False, "Slime": True}
    assert [(pid, s.x, s.name) for pid, s in model.spawn_points] == [(0, 600, "Snail"), (0, 300, "Slime")]
    assert model.monsters_by_platform == {0: ["Snail", "Slime"]}
    raw = {"id": 1, "name": "Test", "tree": tree, "physics": {}, "mob_names": {"2": "Snail", "7": "Slime"},
           "mob_exp": {"2": 2, "7": 12}, "mob_bodies": {}, "mob_die_s": {}, "mob_speeds": {"2": -65, "7": -20}}
    monkeypatch.setattr(mapdata, "CACHE", tmp_path)
    (tmp_path / "1.json").write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(mapdata, "_mob_touch", lambda ids: {"mob_damage": {i: 9.0 for i in ids},
                                                            "mob_hops": {i: i == "7" for i in ids}})
    loaded = mapdata.load(1)
    assert loaded.mob_damage == {"Snail": 9.0, "Slime": 9.0} and loaded.mob_hops == {"Snail": False, "Slime": True}
    assert json.loads((tmp_path / "1.json").read_text(encoding="utf-8")) == raw  # not written back
    from gamedata.paths import COPY
    if not (COPY / "Packs" / "Mob_00000.ms").is_file():
        pytest.skip("needs the game data copy (gamedata/data/copy)")
    monkeypatch.undo()
    data = mapdata._mob_touch({"2", "7", "8", "9", "13", "19"})  # the first map's, Green and Horny Mushrooms
    assert data["mob_damage"] == {"2": 3.0, "7": 13.0, "8": 15.0, "9": 17.0, "13": 54.0, "19": 110.0}
    assert data["mob_hops"] == {"2": False, "7": True, "8": True, "9": True, "13": False, "19": False}


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
    # Worked out once per map: the same as working them out again, and a list of its own each time.
    assert spots.spots(model, ground.id) == spots._spots(model, ground.id)
    first = spots.spots(model, ground.id)
    first.clear()
    assert spots.spots(model, ground.id) == spots._spots(model, ground.id) != []
    other = mapdata.parse(1, tree, "Test", {})  # another map's are its own
    assert spots.spots(other, ground.id) == spots._spots(other, ground.id)
    assert spots._KEPT[id(other)][0] is other


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
    # At a pig up on B: a jump throw's attack starts 26 up (the line 54 above her feet): still under the 60 step's
    # top. A jump back's, 44 up (72): over it. By either line (the v83 line ends at the pig's feet, 60 up, from 54
    # up, or crosses its body at 72).
    for line in reach.LINES:
        assert reach.wall_between(a, 100, 500, 300, rise=reach.air_rise(physics.JUMP_ATTACK_MS), target=b,
                                  line=line) is not None, line
        assert reach.wall_between(a, 100, 500, 300, rise=reach.air_rise(physics.JUMP_BACK_ATTACK_MS), target=b,
                                  line=line) is None, line
    assert reach.wall_in_way(a, 100, 500, 300, rise=reach.air_rise(physics.JUMP_ATTACK_MS), target=b) is not None
    assert reach.wall_in_way(a, 100, 500, 300, rise=reach.air_rise(physics.JUMP_BACK_ATTACK_MS), target=b) is None


def test_a_monster_is_never_past_its_platforms_end():
    model, a, b, c = step_map()
    # Read 9 units past A's end (where the step wall rises): on A, so on her side of it.
    near_end = target(209, 500, platform=a.id)
    assert reach.can_hit(a, 100, 500, near_end) and reach.wall_in_way(a, 100, 500, 209, target=a) is None
    assert reach.wall_in_way(a, 100, 500, 209) is not None  # its platform unknown: taken where it was seen
    # On B, right at the step's top: behind it.
    assert reach.wall_in_way(a, 100, 500, 200, target=b) is not None


def test_her_decisions_take_a_monster_on_its_own_platform_by_its_id():
    # A wide floor A (-400..200, y 500), the step wall at 200 and B (200..400, y 440) above it. From A at -190 a
    # pig on B at 205, 395 off: the star's row ends inside its body, and the client's hit point (the middle of the
    # body's part within the row, 190.5) is on A's side of the wall. Taken on B, as the pig is (Platform.peer, by
    # the monster's id), the wall stops it, as the world's line does (it knows the pig's platform): from 4 at the
    # monsters at 2's end her decisions threw where the world's stars, and the game's, stop at the step.
    tree = {"info": {"VRLeft": -500, "VRTop": 0, "VRRight": 800, "VRBottom": 800},
            "foothold": {"1": {"1": {"1": _fh(-400, 500, 200, 500, 0, 2), "2": _fh(200, 500, 200, 440, 1, 3),
                                     "3": _fh(200, 440, 400, 440, 2, 0)}}}}
    model = mapdata.parse(1, tree, "Wide step", {})
    a, b = (next(p for p in model.platforms if p.left == left) for left in (-400, 200))
    assert a.peer(b.id) is b and a.peer(None) is None and a.peer(99) is None
    pig = target(205, 440, platform=b.id)
    assert reach.hit_point(-190, 500 - reach.SHOT_H, 1, 205, 440, *reach.PIG)[0] < 200
    assert reach.wall_between(a, -190, 500, 205, ty=440) is None  # its platform unknown: short of the wall
    assert reach.wall_to(a, -190, 500, pig) == reach.wall_between(a, -190, 500, 205, target=b, ty=440) == a.walls[0]
    assert not reach.can_hit(a, -190, 500, pig) and stance(-190, 500, a, [pig])[0] != "throw"
    alone = Platform(a.id, a.segments, a.walls, a.floors)  # no map attached (peers): as before, where it was seen
    assert reach.wall_to(alone, -190, 500, pig) is None


def test_walls_block_can_be_switched_off_for_probes(monkeypatch):
    model, a, b, c = step_map()
    monkeypatch.setattr(reach, "WALLS_BLOCK", False)
    assert reach.wall_in_way(a, 100, 500, 300) is None and reach.can_hit(a, 100, 500, target(300, 440, platform=b.id))
    assert reach.clear_span(a, 100, 500) == (-math.inf, math.inf)
    assert reach.wall_between(a, 100, 500, 300) is not None  # what the shot log records either way


def test_spots_are_worked_out_again_when_walls_are_switched(monkeypatch):
    """spots keeps its answers per map, platform and wall switch (reach.WALLS_BLOCK): a probe that switches
    what stops a star gets the spots of its own rule, and switched back, the ones before."""
    model, a, b, c = step_map()
    on = spots.spots(model, b.id)
    assert a.id not in [sp.platform for sp in on]  # nothing of B from A: the step wall is in the way
    monkeypatch.setattr(reach, "WALLS_BLOCK", False)
    off = spots.spots(model, b.id)
    assert a.id in [sp.platform for sp in off] and off == spots._spots(model, b.id)
    monkeypatch.setattr(reach, "WALLS_BLOCK", True)
    assert spots.spots(model, b.id) == on


def test_spots_are_worked_out_again_when_reach_s_other_answers_are_switched(monkeypatch):
    """The spots go by the line terrain stops (reach.LINE) and by what the attack picks (the search's window,
    the claw, the old ramp) too, which the probes' tests switch for their worlds: switched, spots answers as
    worked out afresh, and switched back, as before."""
    model, a, b, c = step_map()
    v83 = spots.spots(model, a.id)
    monkeypatch.setattr(reach, "LINE", "level")  # from C down at A, its own floor no longer stops the line close in
    level = spots.spots(model, a.id)
    assert level != v83 and level == spots._spots(model, a.id)
    monkeypatch.setattr(reach, "LINE", "v83")
    monkeypatch.setattr(reach, "SEARCH_FLOOR0", -20)
    lower = spots.spots(model, a.id)
    assert lower != v83 and lower == spots._spots(model, a.id)
    monkeypatch.undo()
    assert spots.spots(model, a.id) == v83


def test_she_doesnt_throw_at_or_walk_for_a_monster_behind_a_wall_of_her_group(monkeypatch):
    model, a, b, c = step_map()
    up = target(300, 440, platform=b.id)  # up on B: in reach by distance from A at 100, behind the step
    assert decide(100, 500, a, up, []) == ("skip", "behind a wall of her platform's foothold group")
    assert stance(100, 500, a, [up]) == ("wait",)
    assert pick_target(100, 500, a, [up, target(390, 440, platform=b.id, bar=0.5)]) is up  # none in reach: nearest
    # Down from B the star gets there: thrown, from B, at the monster on A.
    assert decide(300, 440, b, target(100, 500, platform=a.id), []) == ("throw", LEFT)
    # Where to shoot B from: nothing of it from A (by distance, from 111 sideways for a pig 60 above: 189 of
    # its 200 units).
    assert spots.coverage(b, 100, 500, a) == 0 and spots.coverage(b, 100, 500) == pytest.approx(0.945)
    # With the rule off (an in-game probe), as before.
    monkeypatch.setattr(reach, "WALLS_BLOCK", False)
    assert decide(100, 500, a, up, []) == ("throw", RIGHT) and stance(100, 500, a, [up]) == ("throw", RIGHT)


def test_a_wall_stops_a_star_only_going_in_at_its_outer_face():
    # A wall standing in the middle of a group, drawn downward (its outer face looks right), across the line 28
    # above the floor: from its right a star at a monster on its left stops there; from its left one goes on
    # through its back (RustMS's port of the client: only a line from a foothold's open side to its closed one).
    tree = {"info": {"VRLeft": -100, "VRTop": 0, "VRRight": 600, "VRBottom": 700},
            "foothold": {"1": {"1": {"1": _fh(0, 500, 400, 500), "2": _fh(200, 440, 200, 480)}}}}
    model = mapdata.parse(1, tree, "Fin", {})
    a = model.platforms[0]
    (w,) = a.walls
    assert reach.faces(w) == 1
    for line in reach.LINES:
        assert reach.wall_between(a, 300, 500, 100, target=a, line=line) is w, line
        assert reach.wall_between(a, 100, 500, 300, target=a, line=line) is None, line
    assert reach.clear_span(a, 300, 500) == (200, math.inf)
    assert reach.clear_span(a, 100, 500) == (-math.inf, math.inf)
    # The step map's step wall is drawn upward, from A up to B: its face looks left, at A, below it.
    model, a, b, c = step_map()
    assert [reach.faces(w) for w in a.walls] == [-1] and sorted(reach.faces(w) for w in c.walls) == [-1, 1]


def ledge_map():
    """Her block H (group (1, 1): 0..300 at y 500, its walls hanging 25 below its ends), the ground L under
    it (group (1, 2): -200..800 at y 560, 60 below), a block U (group (1, 3): 350..700 at y 440, 60 above H),
    a ledge G (group (1, 4): 420..460 at y 505) and a ledge F (group (1, 5): 420..470 at y 452). Returns the
    model and H, L, U."""
    tree = {"info": {"VRLeft": -300, "VRTop": 0, "VRRight": 900, "VRBottom": 800},
            "foothold": {"1": {"1": {"1": _fh(0, 525, 0, 500, 0, 2), "2": _fh(0, 500, 300, 500, 1, 3),
                                     "3": _fh(300, 500, 300, 525, 2, 0)},
                               "2": {"4": _fh(-200, 560, 800, 560)},
                               "3": {"5": _fh(350, 465, 350, 440, 0, 6), "6": _fh(350, 440, 700, 440, 5, 7),
                                     "7": _fh(700, 440, 700, 465, 6, 0)},
                               "4": {"8": _fh(420, 505, 460, 505)},
                               "5": {"9": _fh(420, 452, 470, 452)}}}}
    model = mapdata.parse(1, tree, "Ledges", {})
    h, low, up = (next(p for p in model.platforms if p.left == left) for left in (0, -200, 350))
    return model, h, low, up


def test_the_v83_line_runs_to_where_the_star_hits_and_floors_stop_it_going_down(monkeypatch):
    # LINE 'v83' (RustMS's port of the client): from 28 above her feet to the hit point, a pig's top 60 below
    # her. From H at 100, 200 from its end, her own floor stops a star at a pig on L under 300 off (1.5 x 200:
    # the line drops 28 to her floor's level for 42 to the pig's top), and the level line doesn't.
    model, h, low, up = ledge_map()
    assert reach.LINE == "v83"  # the default since the user saw her own platform stop her stars (2026-10-21)
    assert reach.hit_point(100, 472, 1, 400, 560, 46, 34) == (400, 514)  # its top: the pig is below the row
    assert reach.hit_point(100, 472, 1, 300, 440, 46, 34) == (300, 440)  # its feet: above the row
    assert reach.hit_point(100, 472, 1, 300, 500, 46, 34) == (300, 472)  # the row: it crosses the body
    assert reach.hit_point(100, 472, 1, 140, 500, 46, 34) == (169.5, 472)  # its part past the shot's start, 65 on

    def stop(x, tx, there=low, y=500, here=h, line="v83"):
        return reach.wall_between(here, x, y, tx, target=there, ty=there.y_at(tx), body=reach.PIG, line=line)

    assert stop(100, 300) in h.segments and stop(100, 399) in h.segments and stop(100, 401) is None
    assert stop(100, 300, line="level") is None
    assert stop(100, -100) is None  # the other way, past her floor's end at 0, 100 from it: from 150 off
    # Through its end exactly (the line from 100 to a pig at 400 meets the floor's level at 300): it ends past
    # the hanging wall there, so it isn't behind that corner (a ledge: behind both, the floor and the wall).
    assert stop(100, 400) is None
    # From 280, 19 from her end: clear down to L at 480, but for the ledge G (another group's floor), which the
    # line goes down through at 437.
    g = next(p for p in model.platforms if (p.left, p.top) == (420, 505))
    assert stop(280, 480) in g.segments
    assert stop(280, 540) is None
    # Up through a floor it goes on: from H at 250 to a pig up on U at 560, through the ledge F from below.
    assert stop(250, 560, there=up) is None
    # A wall of her group at its outer face stops it as the level line does: the step map's, from A up at B.
    steps, a, b, _ = step_map()
    assert stop(100, 300, there=b, here=a) == stop(100, 300, there=b, here=a, line="level") == a.walls[0]
    assert stop(300, 100, there=a, y=440, here=b) is None  # down past the wall's back, clear of B's floor
    # Her decisions keep to the switch: a pig on L at 380 is thrown at from 100 by the v83 line only from past
    # 140 (where 380 - x > 1.5 (300 - x)). She walks there: to 145, the nearest of the places 10 apart from 25,
    # her platform's limit (stance._clear_from). By the level line she throws from 100.
    pig = target(380, 560, platform=low.id)
    assert not reach.can_hit(h, 135, 500, pig) and reach.can_hit(h, 145, 500, pig)
    assert stance(100, 500, h, [pig]) == ("approach", RIGHT, 45)
    assert decide(100, 500, h, pig, []) == ("walk", RIGHT, 109)  # to its own near goal, 171 off: clear
    monkeypatch.setattr(reach, "LINE", "level")
    assert reach.can_hit(h, 100, 500, pig) and stance(100, 500, h, [pig]) == ("throw", RIGHT)


def test_with_the_v83_line_a_spot_covers_only_what_her_floor_lets_a_star_down_to(monkeypatch):
    # From H at 100, a pig on L 60 below is in reach 151-410 off either way: 149 + 259 of L's 1000 units.
    # By the v83 line her floor stops a star under 300 off to the right (200 to her end) and none to the
    # left (100 to it: under 150).
    model, h, low, up = ledge_map()
    assert spots.coverage(low, 100, 500, h) == pytest.approx(0.149 + 0.110, abs=0.003)
    assert spots.coverage(low, 100, 500) == pytest.approx(0.408)  # her platform unknown: nothing stops it
    monkeypatch.setattr(reach, "LINE", "level")
    assert spots.coverage(low, 100, 500, h) == pytest.approx(0.408)


def test_clear_length_answers_as_its_lines_one_at_a_time():
    # reach.clear_length tries every star's line at once (reach._sight_clear): the length of the target, tried in
    # the middle of every 2 units, that a star gets to by the v83 line, as wall_between has it one line at a time,
    # with every answer the same (the ledge map's floors, corners and another group's ledge).
    def one_at_a_time(here, x, y, there, lo, hi, body=reach.PIG):
        n = max(1, math.ceil((hi - lo) / 2))
        width = (hi - lo) / n
        xs = [reach._inside(lo + (i + 0.5) * width, there) for i in range(n)]
        return width * sum(reach.wall_between(here, x, y, t, target=there, ty=there.y_at(t), body=body, line="v83")
                           is None for t in xs)

    model, h, low, up = ledge_map()
    cases = [(h, x, 500, there, lo, hi, body) for x in (0, 25, 100, 160, 240, 290, 300)
             for there, lo, hi in ((low, -200, 0), (low, 150, 560), (low, 300, 800), (up, 350, 700))
             for body in (reach.PIG, (25, 19))]
    for here, x, y, there, lo, hi, body in cases:
        assert reach.clear_length(here, x, y, there, lo, hi, body) == one_at_a_time(here, x, y, there, lo, hi, body)


def test_a_lines_footholds_come_from_its_platforms_grid_the_same_ones_in_the_same_order():
    # reach._Sight (from 2026-11-01): the footholds a line of sight is tried against (sight_stop's, clear_length's)
    # from the SIGHT_CELL cells its box touches, not all of a platform's floors and walls: the same ones in the same
    # order as _near_line over all of them (the nearest crossing's tie goes to the first), on Tree Dungeon's tower (396
    # for each of its 19 platforms of the walls' group) for a star's lines from her place on every platform, level,
    # plumb and up to 450 across and 250 up or down, and a box over more cells than there are footholds; so every
    # sight_stop answers the same, the simulator's world's own (World.sight_stop: wall_between's) with them.
    import random

    from gamelab.sim import scenarios
    if not scenarios.available(10002073):
        pytest.skip("no Tree Dungeon game data")
    model = mapdata.load(10002073)
    rng, stops = random.Random(11), 0
    for p in model.platforms:
        every = (*p.floors, *p.walls)
        for _ in range(250):
            x = rng.uniform(p.left, p.right)
            sx, sy = x, p.y_at(x) - rng.choice((reach.SHOT_H, 0, 77 + reach.SHOT_H))
            ex = sx + rng.choice((0.0, rng.uniform(-450, 450)))
            ey = sy + rng.choice((0.0, rng.uniform(-250, 250)))
            near = reach._near_line(every, sx, sy, ex, ey)
            got = reach._sight(p).near(sx, sy, ex, ey)
            assert len(got) == len(near) and all(a is b for a, b in zip(got, near))
            stop = reach.sight_stop(p, sx, sy, ex, ey, near)
            assert reach.sight_stop(p, sx, sy, ex, ey) is stop
            stops += stop is not None
        whole = reach._sight(p).near(-2000, -3000, 2000, 1000)
        assert len(whole) == len(every) and all(a is b for a, b in zip(whole, every))
    assert stops > 500  # (lines that something stops, not only open air)
    # made once a platform, anew where its floors or walls are others
    _, h, low, _ = ledge_map()
    grid = reach._sight(h)
    assert reach._sight(h) is grid and grid.near(-5, 495, 60, 520) == [h.segments[0], h.walls[0]]  # (4 cells of 7)
    h.walls = ()  # (her block's walls gone)
    assert reach._sight(h) is not grid and reach._sight(h).near(-5, 495, 60, 520) == [h.segments[0]]


def shield_map():
    """step_map's A, step wall and B (group (1, 1)), and a ledge L (group (1, 2): 60..300 at y 560), 60 below
    A. Returns the model and A, B, L."""
    tree = {"info": {"VRLeft": -100, "VRTop": 0, "VRRight": 800, "VRBottom": 800},
            "foothold": {"1": {"1": {"1": _fh(0, 500, 200, 500, 0, 2), "2": _fh(200, 500, 200, 440, 1, 3),
                                     "3": _fh(200, 440, 400, 440, 2, 0)},
                               "2": {"4": _fh(60, 560, 300, 560)}}}}
    model = mapdata.parse(1, tree, "Shield", {})
    a, b, low = (next(p for p in model.platforms if p.left == left) for left in (0, 200, 60))
    return model, a, b, low


def test_a_monster_the_attack_takes_first_whose_star_is_stopped_shields_those_behind_it(monkeypatch):
    # The client checks the line to its pick alone, and stopped, the attack hits nothing (RustMS Combat.cpp
    # 1290-1307; ohmi69 and meowdb on this game: the slot isn't refilled). From A at 10, a pig up on B at 215,
    # just past the step wall, is in the search's band 5; a pig on L 60 below at 170, short of the wall, in band
    # 6 (it needs room sideways). The star goes to the one on B and stops at the wall: the one on L, in reach and
    # its line clear, gets none. On the level line: by the v83 line A's own floor stops the star at the one on L
    # too (160 off, 190 from A's end; test_her_floor_s_shadow_shields_those_behind_it has that).
    monkeypatch.setattr(reach, "LINE", "level")
    model, a, b, low = shield_map()
    up, under = target(215, 440, platform=b.id), target(170, 560, platform=low.id)
    assert reach.search_band(*reach.body_box(205, -60, *reach.PIG)) == 5
    assert reach.search_band(*reach.body_box(160, 60, *reach.PIG)) == 6
    assert reach.can_hit(a, 10, 500, under) and not reach.can_hit(a, 10, 500, up)
    assert reach.shielded(a, 10, 500, under, [up, under]) is up
    assert not reach.can_hit(a, 10, 500, under, others=[up, under])
    assert stance(10, 500, a, [under]) == ("throw", RIGHT)
    assert stance(10, 500, a, [under], around=[up, under]) == ("wait",)
    # In the same band they'd take the stars by turns: no shield.
    assert reach.shielded(a, 10, 500, target(180, 560, platform=low.id), [target(235, 440, platform=b.id)]) is None
    # Off (an in-game probe), as before.
    monkeypatch.setattr(reach, "WALLS_BLOCK", False)
    assert reach.can_hit(a, 10, 500, under, others=[up, under])


def test_decide_never_walks_on_the_spot(monkeypatch):
    # From A at 25 (its left end's stand limit) at a pig on L at 190, 60 below, behind the pig up on B at 215
    # that the search takes first (its star stopped by the step wall): every spot near enough to reach the
    # one on L is shielded too, and she's at the nearest. She skips it; she used to "walk" 0 units, a tap back
    # and forth on the spot for as long as the kill's patience lasted. On the level line (by the v83 line A's
    # floor stops the star at the one on L from all of A).
    monkeypatch.setattr(reach, "LINE", "level")
    model, a, b, low = shield_map()
    up, under = target(215, 440, platform=b.id), target(190, 560, platform=low.id)
    assert reach.shielded(a, 25, 500, under, [up, under]) is up
    assert decide(25, 500, a, under, [], around=[up, under]) == ("skip", "behind a monster whose star is stopped")
    assert decide(25, 500, a, under, []) == ("throw", RIGHT)
    # At her platform's end with a target 415 off (past SHOT_RANGE, within a star's reach): no nearer spot, so
    # she throws from there.
    assert decide(975, 500, PLAT, target(1390, 560, platform=1), []) == ("throw", RIGHT)
    monkeypatch.undo()
    skip = ("skip", "a floor in the way from all along her platform")
    assert decide(25, 500, a, under, [], around=[up, under]) == skip


def test_her_floor_s_shadow_shields_those_behind_it():
    # By the v83 line, from H at 100 (200 from its end) a pig on L at 330 (230 off, the search's band 6) is in her
    # floor's shadow: its star is stopped (from past 240 only is it clear, but from there it's under 151 off, so
    # never). A pig on L at 420 (320 off, band 11), its line clear, gets no star while the one at 330 is picked
    # first. From 179 on, the one at 330 (151 off) isn't picked, and the one at 420 is, its line clear: stance
    # walks her to 185, the nearest place 10 apart from 25 from where a star gets to it (stance._clear_from with
    # the monsters around her), and decide to its near goal, 171 off (249).
    model, h, low, up = ledge_map()
    shade, far = target(330, 560, platform=low.id), target(420, 560, platform=low.id)
    assert reach.wall_to(h, 100, 500, shade) in h.segments and reach.wall_to(h, 100, 500, far) is None
    assert reach.shielded(h, 100, 500, far, [shade, far]) is shade
    assert not reach.can_hit(h, 100, 500, far, others=[shade, far]) and reach.can_hit(h, 100, 500, far)
    assert not any(reach.can_hit(h, g, 500, shade) for g in range(25, 276))
    assert [g for g in range(25, 276) if reach.can_hit(h, g, 500, far, others=[shade, far])] == list(range(179, 270))
    assert stance(100, 500, h, [shade, far]) == ("approach", RIGHT, 85)
    assert decide(100, 500, h, far, [], around=[shade, far]) == ("walk", RIGHT, 149)


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
def test_a_monster_is_placed_where_its_feet_are():
    """Scene.place puts a monster where the template that matched says its feet are (its origin), not at
    the template's centre and bottom; and the look's telemetry says which way it faces."""
    import numpy as np

    from gamelab.localization.pose import Where
    from gamelab.localization.scene import Scene
    from gamelab.screen import UNITS_PER_HALF_PX as U
    tree = {
        "info": {"VRLeft": 0, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
        "foothold": {"1": {"1": {"1": {"x1": 0, "y1": 500, "x2": 1000, "y2": 500, "prev": 0, "next": 0}}}},
        "life": {"0": {"type": "m", "id": "8", "x": 600, "y": 500, "cy": 500, "rx0": 50, "rx1": 950}},
        "miniMap": {"width": 2000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4},
    }
    model = mapdata.parse(1, tree, "Test", {}, {"8": "Pig"})
    s = Scene(t=0.0, frame=np.zeros((1, 1, 3), np.uint8), tag=(400, 308), hp=1.0, mp=1.0,
              where=Where(300.0, 500.0, model.platform(0), (0, 0)))  # her feet at y 300 on screen
    centred = s.place(Mob("pig", 500, 280, 46, 30, 0.9), model, None)
    assert centred.x == pytest.approx(300 + 100 * U) and centred.facing == 0
    hop = Mob("pig", 500, 280, 46, 30, 0.9, ox=507.0, feet=301.5, facing=1, pose="move", frame="move/2")
    placed = s.place(hop, model, None)  # its feet 7 px right of the picture's middle, below the picture
    assert placed.x == pytest.approx(300 + 107 * U) and placed.y == 500 and placed.platform == 0
    assert (placed.facing, placed.pose) == (1, "move")
    s.monsters = [centred, placed]
    assert [("facing" in m, m.get("facing")) for m in s.record()["monsters"]] == [(False, None), (True, 1)]


def test_a_monster_with_its_feet_on_a_platform_goes_on_it_whatever_spawns_there(monkeypatch):
    """Monsters leave the platforms they spawn on (a pig on 26 in after_rest.png). With its feet from the
    template's origin and her on the ground, a monster goes on the platform right under its feet; otherwise
    the spawn rule stands (feet from a template's bottom were up to 20 units off)."""
    import numpy as np

    from gamelab.localization.pose import Where
    from gamelab.localization.scene import Scene
    from gamelab.screen import UNITS_PER_HALF_PX as U
    tree = {
        "info": {"VRLeft": 0, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
        "foothold": {"1": {
            "1": {"1": {"x1": 300, "y1": 460, "x2": 520, "y2": 460, "prev": 0, "next": 0}},  # no spawns, 40 above
            "2": {"2": {"x1": 0, "y1": 500, "x2": 1000, "y2": 500, "prev": 0, "next": 0}},   # pigs spawn here
        }},
        "life": {"0": {"type": "m", "id": "8", "x": 600, "y": 500, "cy": 500, "rx0": 50, "rx1": 950}},
        "miniMap": {"width": 2000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4},
    }
    model = mapdata.parse(1, tree, "Test", {}, {"8": "Pig"})
    high, low = model.platform_under(400, 450), model.platform_under(700, 490)
    assert monster_platform(model, "pig", 400, 461, exact=True) is high      # its feet 1 unit below it
    assert monster_platform(model, "pig", 400, 461) is low                   # the spawn rule
    assert monster_platform(model, "pig", 400, 480, exact=True) is low       # 20 off: nothing that close
    monkeypatch.setattr(monsters, "FEET_SNAP", None)
    assert monster_platform(model, "pig", 400, 461, exact=True) is low
    monkeypatch.undo()
    s = Scene(t=0.0, frame=np.zeros((1, 1, 3), np.uint8), tag=(400, 308), hp=1.0, mp=1.0, ground=(300, 500),
              where=Where(300.0, 500.0, low, (0, 0)))  # she stands on the ground, her feet at y 300 on screen
    feet = 300 + (461 - 500) / U
    pig = Mob("pig", 470, round(feet) - 15, 46, 30, 0.9, ox=470.0, feet=feet)
    assert s.grounded and s.place(pig, model, None).platform == high.id
    assert s.place(Mob("pig", 470, round(feet) - 15, 46, 30, 0.9), model, None).platform == low.id  # its bottom
    s.ground = (400, 402)  # in the air: no ground under her
    assert not s.grounded and s.place(pig, model, None).platform == low.id
