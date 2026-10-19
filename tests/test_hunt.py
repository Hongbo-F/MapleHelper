"""Hunting controller pieces that don't need the game: bars, platform edges, move choice, detection."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab import shots
from gamelab.hunt import TOO_CLOSE, Target, bars, decide, ground_extent, pick_target
from gamelab.mapdata import Platform, Segment
from gamelab.mobs import ASSETS, Mob, MobDetector
from maplehelper.engine import LEFT, RIGHT

FIXTURES = Path(__file__).parent / "fixtures"

SKY = (60, 140, 230)
GRASS = (90, 170, 40)


def test_bars_read_the_fill_fraction():
    full = np.zeros((1080, 1920, 3), np.uint8)
    full[1060:1069, 707:707 + 121] = (220, 30, 30)  # HP 121/147 px = 523/642 in game
    full[1060:1069, 859:859 + 74] = (30, 60, 220)   # MP half
    hp, mp = bars(full)
    assert hp == pytest.approx(121 / 147, abs=0.02)
    assert mp == pytest.approx(0.5, abs=0.02)


def test_ground_extent_stops_at_the_platform_edges():
    frame = np.full((300, 600, 3), SKY, np.uint8)
    frame[202:230, 100:250] = GRASS  # platform top at y=202
    assert ground_extent(frame, 180, 200) == pytest.approx((100, 249), abs=1)


PLAT = Platform(0, [Segment(0, 500, 1000, 500)])      # her platform
BELOW = Platform(1, [Segment(1100, 560, 1600, 560)])  # 60 below, to the right


def target(x, y=500, platform=0, kind="pig"):
    return Target(Mob(kind, 0, 0, 30, 30, 0.9), x, y, platform)


def test_throws_from_the_far_end_of_her_reach():
    assert decide(500, 500, PLAT, target(500 + shots.SHOT_RANGE - 10), []) == ("throw", RIGHT)
    assert decide(500, 500, PLAT, target(200), []) == ("throw", LEFT)


def test_shoots_a_monster_on_a_lower_platform_from_hers():
    assert decide(950, 500, PLAT, target(1250, 560, platform=1), []) == ("throw", RIGHT)


def test_walks_just_into_reach_when_too_far():
    move = decide(300, 500, PLAT, target(800), [])
    assert move[:2] == ("walk", RIGHT)
    assert shots.MIN_SHOT < 800 - (300 + move[2]) <= shots.SHOT_RANGE


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


def test_picks_close_monsters_on_her_platform_before_far_ones():
    near_me, far = target(650), target(1300, 560, platform=1)
    assert pick_target(500, 500, PLAT, [far, near_me]) is near_me


def test_coverage_and_spots_prefer_safe_platforms():
    from gamelab import mapdata
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
    best = shots.spots(model, ground.id)[0]
    assert best.safe and model.platform(best.platform).y_at(best.x) == 500
    assert best.coverage == pytest.approx(shots.coverage(ground, best.x, 500), abs=1e-3)
    assert best.coverage > 0.35  # off her level the star needs 150-200 units sideways (shots.min_dx)


def test_hp_bar_reads_the_green_fraction():
    frame = np.full((200, 200, 3), SKY, np.uint8)
    frame[87:93, 68:112] = (165, 165, 165)  # border
    frame[88:92, 70:100] = (0, 196, 0)    # 30 px green
    frame[88:92, 100:105] = (140, 0, 0)   # 5 px red (latest damage)
    frame[88:92, 105:110] = (0, 0, 0)     # 5 px black
    assert shots.hp_bar(frame, 90, 100) == pytest.approx(0.75)
    assert shots.hp_bar(np.full((200, 200, 3), SKY, np.uint8), 90, 100) is None


def test_detector_ignores_scenery():
    frame = np.asarray(Image.open(FIXTURES / "game_scene.png").convert("RGB"))  # daffodils, leaves, sky
    half = np.asarray(Image.fromarray(frame).resize((frame.shape[1] // 2, frame.shape[0] // 2)))
    assert MobDetector().detect(half) == []


def test_detector_finds_a_pasted_monster():
    frame = np.full((200, 300, 3), SKY, np.uint8)
    frame[150:] = GRASS
    rgba = np.asarray(Image.open(ASSETS / "pig_0.png"))
    small = np.asarray(Image.fromarray(rgba).resize((rgba.shape[1] // 2, rgba.shape[0] // 2)))
    h, w = small.shape[:2]
    region = frame[150 - h:150, 100:100 + w]
    alpha = small[..., 3:] > 0
    frame[150 - h:150, 100:100 + w] = np.where(alpha, small[..., :3], region)
    found = MobDetector().detect(frame)
    assert [m.kind for m in found] == ["pig"]
    assert abs(found[0].x - (100 + w // 2)) <= 3



def test_blank_windows_dont_crowd_out_a_monster_in_an_odd_sized_band():
    # The fight's band was 151 rows. Halving an odd size smeared faint colour into blank windows,
    # whose silhouette scores came out a fake 1.0 and took every check: in the basin, snails and
    # pigs at her level were never seen. Here: pig-coloured specks (like flowers), a pig on the right.
    rgba = np.asarray(Image.open(ASSETS / "pig_0.png"))
    pig = np.asarray(Image.fromarray(rgba).resize((rgba.shape[1] // 2, rgba.shape[0] // 2)))
    ph, pw = pig.shape[:2]
    detector = MobDetector()
    for seed in range(4):
        rng = np.random.default_rng(seed)
        frame = np.full((151, 960, 3), SKY, np.uint8)
        frame[121:] = GRASS
        for _ in range(60):
            y, x = rng.integers(0, 117), rng.integers(0, 700)
            frame[y:y + rng.integers(2, 9), x:x + rng.integers(2, 12)] = (240, 160, 150)
        region = frame[121 - ph:121, 760:760 + pw]
        frame[121 - ph:121, 760:760 + pw] = np.where(pig[..., 3:] > 0, pig[..., :3], region)
        found = detector.detect(frame, rows=(0, 151))
        assert any(m.kind == "pig" and abs(m.x - (760 + pw // 2)) <= 4 for m in found), seed

def test_cornered_she_fights_from_where_she_is():
    t = target(PLAT.right - 70)  # she's at the right end, the monster right next to her
    assert decide(PLAT.right - 30, 500, PLAT, t, []) == ("throw", LEFT)  # not jumping to its other side


def test_she_throws_toward_the_side_with_more_monsters_in_reach():
    from gamelab.hunt import stance
    group = [target(700), target(780)]
    lone = target(250)
    assert stance(500, 500, PLAT, group + [lone]) == ("throw", RIGHT)


def test_one_too_close_gets_a_jump_back_throw_not_a_claw_swipe():
    from gamelab.hunt import stance
    # Room behind her: jump away from it and throw back from the air.
    assert stance(500, 500, PLAT, [target(560), target(700)]) == ("jump_back", RIGHT)
    # Cornered at the left end, or another monster behind her: straight up. No crossing either way.
    assert stance(60, 500, PLAT, [target(110)]) == ("jump_throw", RIGHT)
    assert stance(500, 500, PLAT, [target(560), target(400)]) == ("jump_throw", RIGHT)


def test_she_keeps_working_the_side_she_faces():
    from gamelab.hunt import stance
    lone_right, pair_left = target(700), [target(300), target(250)]
    assert stance(500, 500, PLAT, [lone_right] + pair_left, facing=RIGHT) == ("throw", RIGHT)
    assert stance(500, 500, PLAT, [lone_right] + pair_left) == ("throw", LEFT)


def test_she_closes_in_only_as_far_as_needed_and_waits_when_nothing_is_there():
    from gamelab.hunt import stance
    act = stance(100, 500, PLAT, [target(900)])
    assert act[:2] == ("approach", RIGHT) and 900 - (100 + act[2]) == pytest.approx(shots.SHOT_RANGE - 40)
    assert stance(100, 500, PLAT, []) == ("wait",)


def test_monsters_are_placed_only_on_platforms_they_spawn_on():
    from gamelab import mapdata
    from gamelab.hunt import monster_platform
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


def test_finishes_hurt_monsters_first():
    fresh, hurt = target(700), Target(Mob("pig", 0, 0, 30, 30, 0.9), 750, 500, 0, bar=0.4)
    assert pick_target(500, 500, Platform(9, [Segment(0, 400, 1000, 400)]), [fresh, hurt]) is hurt


class _Eyes:
    """Hands out prepared looks (no game needed)."""
    want_ropes = False

    def __init__(self, sights):
        self.sights = iter(sights)

    def look(self):
        return next(self.sights)


def test_a_kill_is_confirmed_by_her_exp_number_changing():
    from gamelab import mapdata
    from gamelab.hunt import Hunter, Sight
    from maplehelper.config import GameKeys, Timing
    model = mapdata.parse(1, {"info": {"VRLeft": 0, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
                              "miniMap": {"centerX": 0, "centerY": 0, "mag": 4}}, "Test", {})
    before = np.zeros((26, 140), bool)
    after = before.copy()
    after[5:15, 100:104] = True  # a digit changed
    looks = [Sight(t, np.zeros((10, 10, 3), np.uint8), None, [], 1.0, 1.0, exp=e)
             for t, e in ((1.0, before), (1.5, before), (2.0, after), (2.5, after))]
    h = Hunter(_Eyes(looks), GameKeys.__new__(GameKeys), Timing.__new__(Timing), model, log=lambda m: None)
    for _ in looks:
        h.look(mobs=False)
    assert h.exp_events == [2.0] and h.report.exp_kills == 1
    assert not h._take_exp(since=2.5)   # gains before she started throwing aren't this target's
    assert h._take_exp(since=1.8) and h.exp_events == []


def test_touches_show_as_purple_text_above_her():
    from gamelab.hunt import touch_popup
    frame = np.full((300, 400, 3), SKY, np.uint8)
    her = (200, 200)
    assert touch_popup(frame, her) == 0
    frame[120:135, 180:220] = (200, 40, 220)  # "MISS" (or damage taken) floating above her head
    assert touch_popup(frame, her) >= 40


def _fighter(monkeypatch, looks, on_look=None):
    """A Hunter whose looks come from `looks` (lists of targets), 0.3 s apart on a fake clock; her
    throws are recorded instead of pressed."""
    from types import SimpleNamespace
    from gamelab import hunt, mapdata
    from maplehelper.config import GameKeys, Timing
    model = mapdata.parse(1, {"info": {"VRLeft": 0, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
                              "miniMap": {"centerX": 0, "centerY": 0, "mag": 4}}, "Test", {})
    h = hunt.Hunter(_Eyes([]), GameKeys.__new__(GameKeys), Timing.__new__(Timing), model, log=lambda m: None)
    clock, seq, throws = [0.0], iter(looks), []

    def look(*a, **k):
        clock[0] += 0.3
        n = len(throws)
        if on_look:
            on_look(h, clock[0])
        return SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT, hp=1.0,
                               targets=next(seq, []), throws=n)

    monkeypatch.setattr(hunt.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(hunt.time, "sleep", lambda s: None)
    h.look, h.health_check = look, lambda s: None
    h.throw = h.jump_throw = h.jump_back = lambda side: throws.append((round(clock[0], 1), side))
    h.sweep_through = lambda s, deaths: None
    h.mop_up = lambda stood=None, rounds=0: None  # looking for loot is tested on its own (test_loot.py)
    return h, throws


def test_she_keeps_throwing_at_a_monster_hidden_after_a_hit(monkeypatch):
    from gamelab.hunt import HIDDEN_FOR
    h, throws = _fighter(monkeypatch, [[target(700)]])  # seen once, then lost under its damage numbers
    r = h.clear(patience=2.0)
    assert len(throws) == 1 + round(HIDDEN_FOR / 0.3 - 1) and {side for _, side in throws} == {RIGHT}
    assert r.stop == "nothing left in reach"


def test_her_exp_going_up_ends_the_throws_at_a_hidden_monster(monkeypatch):
    def kill_at(h, t):
        if abs(t - 0.9) < 1e-6:
            h.report.exp_kills += 1
    h, throws = _fighter(monkeypatch, [[target(700)]], on_look=kill_at)
    r = h.clear(patience=2.0)
    assert len(throws) == 2 and r.kills == {"pig": 1}


def test_with_no_monster_in_sight_she_moves_on_soon(monkeypatch):
    from gamelab.hunt import EMPTY_PATIENCE
    h, throws = _fighter(monkeypatch, [])
    h.clear(patience=2.0)
    from gamelab import hunt
    assert hunt.time.monotonic() <= 0.3 + EMPTY_PATIENCE + 0.3 and not throws
    # One seen, just out of reach: she waits the full patience for it to walk in.
    far = [[target(980, y=200)]] * 20
    h, throws = _fighter(monkeypatch, far)
    h.clear(patience=2.0, approach=False)
    assert hunt.time.monotonic() >= 2.0 and not throws


def test_drops_are_swept_once_she_lands_or_left_to_the_tour(monkeypatch):
    from types import SimpleNamespace
    from gamelab.hunt import Drops
    h, _ = _fighter(monkeypatch, [])
    swept = []
    h.sweep_through = lambda s, deaths: swept.append([d.x for d in deaths])
    air = SimpleNamespace(where=None, platform=None)
    ground = SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT)
    looks = iter([air, air, ground])
    h.look = lambda *a, **k: next(looks)
    deaths = [Drops(600, 500, 0, 0.0), Drops(700, 560, 1, 0.0)]  # one on her platform, one below
    h.sweep_after(deaths)
    assert swept == [[600, 700]] and [d.x for d in h.pending_loot] == [700]
    h.pending_loot, swept[:] = [], []
    h.look = lambda *a, **k: air  # never lands (on a rope, say): the tour collects them later
    h.sweep_after(deaths)
    assert not swept and len(h.pending_loot) == 2


def test_she_stands_on_each_pile_pressing_pick_up(monkeypatch):
    from types import SimpleNamespace
    from gamelab.hunt import Drops, LOOT_REACH
    h, _ = _fighter(monkeypatch, [])
    del h.sweep_through  # the real one
    did = []
    h.walk = lambda direction, units, pick=True: did.append(("walk", direction, round(units)))
    h.pick_here = lambda presses=4: did.append(("pick",))
    s = SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT)
    h.sweep_through(s, [Drops(600, 500, 0, 0.0), Drops(400, 500, 0, 0.0)])
    # Nearer end first (both 100 away: the first found), over each pile, stopping on it.
    walks = [d for d in did if d[0] == "walk"]
    assert did.count(("pick",)) == 2
    assert sum(u for _, d, u in walks if d == RIGHT) - sum(u for _, d, u in walks if d == LEFT) in (
        600 + LOOT_REACH - 500, 400 - LOOT_REACH - 500)


def test_a_kill_after_she_moved_on_to_the_next_target_still_gets_its_drops(monkeypatch):
    # She threw at a pig on the right, then turned to one on the left; the right one's death (EXP up)
    # comes in after that, when the last throw aimed only left.
    def kill_at(h, t):
        if abs(t - 1.2) < 1e-6:
            h.report.exp_kills += 1
    looks = [[target(700)], [target(450)], [target(450)], [target(450)], [target(450)]]
    h, throws = _fighter(monkeypatch, looks, on_look=kill_at)
    swept = []
    h.sweep_after = lambda deaths: swept.extend(d.x for d in deaths)
    h.clear(max_kills=1, patience=2.0)
    assert [side for _, side in throws][:2] == [RIGHT, LEFT]  # then at the one on her left
    assert h.report.kills == {"pig": 1} and swept == [700]   # not the one still in plain sight


def test_after_the_kill_count_she_finishes_what_is_in_reach(monkeypatch):
    def kill_at(h, t):
        if abs(t - 0.6) < 1e-6:
            h.report.exp_kills += 1
    looks = [[target(700), target(250)]] * 2 + [[target(250)]] * 3 + [[]] * 5
    h, throws = _fighter(monkeypatch, looks, on_look=kill_at)
    h.sweep_after = lambda deaths: None
    r = h.clear(max_kills=1, patience=2.0, finish=True)
    assert len(throws) >= 3 and LEFT in [side for _, side in throws]  # the one on the left too
    assert r.stop == "kill limit"
    h2, throws2 = _fighter(monkeypatch, looks, on_look=kill_at)
    h2.sweep_after = lambda deaths: None
    h2.clear(max_kills=1, patience=2.0)  # without finish: stops at the kill
    assert len(throws2) < len(throws)


def test_a_monster_right_under_her_makes_her_step_away_not_throw():
    from gamelab.hunt import stance
    below = target(530, y=560, platform=1)       # 60 below, 30 to the side: a star misses it
    act = stance(500, 500, PLAT, [below])
    assert act[:2] == ("back", LEFT) and shots.can_shoot(530 - (500 - act[2]), 60)
    assert not shots.can_shoot(30, 60) and shots.can_shoot(30 + 150, 60) and not shots.can_shoot(170, -60)


def test_with_far_on_she_shoots_monsters_on_any_platform_in_reach(monkeypatch):
    far_pig = target(700, y=560, platform=7)  # another platform, 60 below and 200 off: in reach
    h, throws = _fighter(monkeypatch, [[far_pig]] * 3)
    h.clear(ground={3}, patience=0.5, far=True)
    assert throws and throws[0][1] == RIGHT
    h, throws = _fighter(monkeypatch, [[far_pig]] * 3)
    h.clear(ground={3}, patience=0.5)  # without far: not one of hers
    assert not throws


def test_a_kill_is_matched_to_the_first_monster_in_the_star_path(monkeypatch):
    # A snail (not hunted) stands between her and the pig she throws at: the star hits the snail,
    # and the EXP that comes in is the snail's, not the pig's.
    def kill_at(h, t):
        if abs(t - 0.6) < 1e-6:
            h.report.exp_kills += 1
    snail, pig = target(650, kind="snail"), target(780)
    looks = [[snail, pig], [pig], [pig]] + [[]] * 6
    h, throws = _fighter(monkeypatch, looks, on_look=kill_at)
    h.sweep_after = lambda deaths: None
    h.clear(kinds={"pig"}, max_kills=1, patience=0.5)
    assert h.report.kills == {"snail": 1}


def test_under_a_low_platform_she_does_not_jump_at_a_monster():
    from gamelab.hunt import stance
    # A jump would carry her up onto the platform overhead: step away, or (cornered) throw from the ground.
    assert stance(500, 500, PLAT, [target(560)], can_jump=lambda a, b: False) == ("back", LEFT, 120)
    assert stance(60, 500, PLAT, [target(110)], can_jump=lambda a, b: False) == ("throw", RIGHT)


def test_jump_back_turns_and_throws_just_before_landing():
    from types import SimpleNamespace
    from gamelab import hunt
    from gamelab.flow import JUMP_LAG
    from gamelab.route import air_time
    key = lambda name: SimpleNamespace(name=name)
    h = hunt.Hunter.__new__(hunt.Hunter)
    h.keys = SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), attack=key("f"), lucky_seven=None)
    h.skill, h.facing, h._lock_until, h.report = False, None, 0.0, hunt.Report()
    sent = []
    h._do = lambda events, name: sent.extend(events)
    h.jump_back(RIGHT)  # the monster is on her right: she jumps left
    assert [e[0] for e in sent] == sorted(e[0] for e in sent)  # the player refuses unsorted plans
    at = lambda k, how: next(t for t, name, h_ in sent if name == k and h_ == how)
    lift = at("alt", "down") + 1000 * JUMP_LAG
    land = lift + 1000 * air_time(0)
    apex = lift + 1000 * air_time(0) / 2
    assert at("left", "down") < at("alt", "down")              # moving away when she jumps
    lift_off = at("alt", "down") + 1000 * JUMP_LAG
    assert lift_off < at("right", "down") < at("f", "down") < apex  # turned in the air, then the key
    leaves = at("f", "down") + hunt.STAR_DELAY_MS               # ... the star leaves ~520 ms later
    assert 0 <= land - leaves <= 40                            # just before landing
    assert at("left", "up") <= at("right", "down")             # no longer holding away when she lands
