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
    assert best.coverage > 0.5


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


def test_jumps_over_a_monster_too_close_when_cornered():
    t = target(PLAT.right - 70)  # she's at the right end, the monster right next to her
    assert decide(PLAT.right - 30, 500, PLAT, t, []) == ("jump", LEFT)


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
