"""Seeing loot on the ground, and going back for it, without the game: synthetic item pictures pasted
into synthetic frames, as test_perception does for monsters."""

import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from maplehelper.config import GameKeys

from gamelab import screen
from gamelab.behavior.hunter import Hunter
from gamelab.behavior.looting import EMPTY_LOOKS, DropSpot
from gamelab.localization import mapdata
from gamelab.localization.pose import Where
from gamelab.localization.scene import Loot, Scene, find_loot
from gamelab.perception.eyes import Eyes
from gamelab.perception.loot import Item, LootDetector, says_full

SKY = (60, 140, 230)
GRASS = (90, 170, 40)
DIRT = (120, 105, 50)
ROOT = Path(__file__).resolve().parents[1]
HERS, UPPER = 1, 0  # her platform's id in _model(), and the one above


def coin(size: int = 32, colour=(175, 120, 72)) -> np.ndarray:
    """A coin like a bronze meso: a disc with a darker rim and a cross (the leaf) on it, RGBA."""
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2 + 0.5, xx - size / 2 + 0.5)
    c = np.array(colour)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[r < size / 2 - 1] = (*c, 255)
    rgba[(r >= size / 2 - 4) & (r < size / 2 - 1)] = (*(c * 0.65), 255)
    cross = ((np.abs(xx - size / 2) < 3) | (np.abs(yy - size / 2) < 3)) & (r < size / 2 - 7)
    rgba[cross] = (*(c * 0.75), 255)
    return rgba


def leather(colour=(110, 60, 45)) -> np.ndarray:
    """A ragged brown hide, RGBA."""
    rgba = np.zeros((40, 44, 4), np.uint8)
    c = np.array(colour)
    rgba[6:36, 4:40] = (*c, 255)
    rgba[6:14, 4:14, 3] = 0       # torn corners
    rgba[28:36, 30:40, 3] = 0
    rgba[16:22, 10:34] = (*(c * 0.6), 255)  # folds
    rgba[26:30, 8:24] = (*(c * 0.6), 255)
    return rgba


@pytest.fixture
def detector(tmp_path) -> LootDetector:
    Image.fromarray(coin(), "RGBA").save(tmp_path / "mesos_bronze0.png")
    Image.fromarray(leather(), "RGBA").save(tmp_path / "leather_0.png")
    return LootDetector(tmp_path)


def paste(frame: np.ndarray, rgba: np.ndarray, cx: int, bottom: int) -> None:
    """Draw a full-size picture into a half-size frame, its bottom on row `bottom`: drawn at full size,
    then halved, as the game's picture is (its edges mix with what's behind)."""
    h, w = rgba.shape[:2]
    hh, hw = (h + 1) // 2, (w + 1) // 2
    y0, x0 = bottom - hh, cx - hw // 2
    big = cv2.resize(frame[y0:bottom, x0:x0 + hw], (2 * hw, 2 * hh), interpolation=cv2.INTER_NEAREST)
    on = rgba[..., 3] >= 128
    big[2 * hh - h:, :w][on] = rgba[..., :3][on]
    frame[y0:bottom, x0:x0 + hw] = cv2.resize(big, (hw, hh), interpolation=cv2.INTER_AREA)


def scene(height: int = 540, width: int = 960, ground: int = 300) -> np.ndarray:
    frame = np.full((height, width, 3), SKY, np.uint8)
    frame[ground:ground + 6] = GRASS
    frame[ground + 6:] = DIRT
    return frame


def test_finds_coins_and_items_lying_on_the_ground(detector):
    frame = scene()
    paste(frame, coin(), 300, 300)
    paste(frame, leather(), 520, 300)
    found = detector.detect(frame, (0, 270, 960, 306))
    assert sorted((i.kind, i.x) for i in found) == [("leather", 520), ("mesos", 300)]
    assert all(abs(i.bottom - 300) <= 1 for i in found)
    assert detector.detect(frame, (0, 270, 960, 306), mesos_only=True)[0].kind == "mesos"


def test_a_shape_in_the_wrong_colour_is_not_loot(detector):
    # The normalised match ignores tint: a grey-green stone the shape of a coin matches it as well as a
    # coin does. In saved frames the sea matched the Blue Potion's shape, UI buttons the White Potion's;
    # their colours never did.
    frame = scene()
    paste(frame, coin(colour=(90, 110, 95)), 300, 300)
    assert detector.detect(frame, (0, 270, 960, 306)) == []
    assert detector.detect(scene(), (0, 270, 960, 306)) == []


def test_an_item_with_a_thin_string_is_found(tmp_path):
    # The Slime Bubble: a green ball on a 1-pixel string. At half size the string is half string, half
    # background, so colours are checked only where the picture covers whole pixels (the ball).
    rgba = np.zeros((36, 38, 4), np.uint8)
    yy, xx = np.mgrid[:36, :38]
    rgba[(np.abs(np.hypot(yy - 12, xx - 14) - 10) < 0.7)] = (40, 40, 30, 255)  # the string's loop
    ball = np.hypot(yy - 27, xx - 28) < 6
    rgba[ball] = (60, 210, 60, 255)
    rgba[ball & (np.hypot(yy - 25, xx - 26) < 2)] = (220, 255, 220, 255)       # its shine
    Image.fromarray(rgba, "RGBA").save(tmp_path / "slime-bubble_0.png")
    frame = scene()
    paste(frame, rgba, 400, 300)
    found = LootDetector(tmp_path).detect(frame, (0, 270, 960, 306))
    assert [(i.kind, abs(i.x - 400) <= 2) for i in found] == [("slime-bubble", True)]


def test_full_inventory_messages():
    assert says_full("You can't get anymore items.")  # what the game said when her Etc tab was full
    assert says_full("Your inventory is full.")
    assert says_full("You can't get any more items.")
    assert says_full("Unable to pick up the item.")
    assert not says_full("Squishy Liquid x1 earned. (Etc)")
    assert not says_full("You have gained mesos (+13)")


# --- seeing loot from a look, and going back for it -------------------------------------------------

def _model():
    tree = {
        "info": {"VRLeft": 0, "VRTop": 0, "VRRight": 3000, "VRBottom": 1000},
        "foothold": {"1": {
            "1": {"1": {"x1": 600, "y1": 500, "x2": 1400, "y2": 500, "prev": 0, "next": 0}},   # hers
            "2": {"2": {"x1": 1500, "y1": 440, "x2": 1900, "y2": 440, "prev": 0, "next": 0}},  # 60 above
        }},
        "miniMap": {"width": 3000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4},
    }
    return mapdata.parse(1, tree, "Test", {})


class _Eyes(Eyes):
    """Eyes with a loot detector and a pickup log, but no game."""
    want_ropes = False

    def __init__(self, loot=None, text=""):
        self._loot, self.text = loot, text

    def messages(self):
        return self.text


def _hunter(eyes=None) -> Hunter:
    return Hunter(eyes or _Eyes(), GameKeys.__new__(GameKeys), _model(), log=lambda m: None)


def _scene(model, frame, x=1000.0) -> Scene:
    """Her at map x on her platform (y 500), her feet on screen row 300 at column 480."""
    p = model.platform_under(x, 500)
    return Scene(t=0.0, frame=frame, tag=(480, 308), hp=1.0, mp=1.0, where=Where(x, 500, p, (0, 0)))


def test_drops_are_placed_on_the_platform_they_lie_on(detector):
    h = _hunter(_Eyes(detector))
    hers, upper = h.model.platform_under(1000, 500), h.model.platform_under(1600, 440)
    frame = scene()
    u = screen.UNITS_PER_HALF_PX
    paste(frame, coin(), 480 + round(100 / u), 300)                       # on hers, 100 to the right
    frame[300 - round(60 / u):300 - round(60 / u) + 6, 480 + round(500 / u):] = GRASS  # the one above
    paste(frame, leather(), 480 + round(600 / u), 300 - round(60 / u))    # on it
    s = _scene(h.model, frame)
    near = find_loot(s, h.model, h.senses.eyes)
    assert [(d.item.kind, d.platform) for d in near] == [("mesos", hers.id)]
    assert near[0].x == pytest.approx(1100, abs=3) and near[0].y == 500
    every = find_loot(_scene(h.model, frame), h.model, h.senses.eyes, wide=True)
    assert sorted((d.item.kind, d.platform) for d in every) == [("leather", upper.id), ("mesos", hers.id)]
    assert s.in_view(hers.id, 1100) and not s.in_view(upper.id, 1600)


def _loot(kind, x, platform=HERS):
    return Loot(Item(kind, 0, 0, 16, 16, 0.9), x, 500, platform)


def _mopper(monkeypatch, looks, text="", monsters=()):
    """A Hunter whose looks (with loot) come from `looks` (lists of Loot); her sweeps are recorded."""
    h = _hunter(_Eyes(text=text))
    here = h.model.platform_under(1000, 500)
    assert here.id == HERS
    seq, sweeps = iter(looks), []
    monkeypatch.setattr(time, "sleep", lambda s: None)
    h.look = lambda *a, **k: SimpleNamespace(where=Where(1000, 500, here, (0, 0)), platform=here, t=0.0,
                                             loot=next(seq, []), monsters=list(monsters))

    def walk_piles(s, deaths, margin):
        sweeps.append(sorted(round(d.x) for d in deaths))
        return [d.x for d in deaths]

    h.looter.walk_piles = walk_piles
    return h, sweeps


def test_she_goes_back_for_drops_still_seen_after_a_sweep(monkeypatch):
    # Seen after the fight's sweep: a coin and a hide; after going back, the hide is still there (a pig
    # drops several and one press takes one); then nothing.
    h, sweeps = _mopper(monkeypatch, [[_loot("mesos", 1100), _loot("leather", 1180)], [_loot("leather", 1180)], []])
    h.looter.mop_up(stood=[1000])
    assert sweeps == [[1100, 1180], [1180]] and not h.looter.unpickable and not h.looter.inventory_full


def test_drops_on_another_platform_are_not_hers_to_mop_up(monkeypatch):
    h, sweeps = _mopper(monkeypatch, [[_loot("mesos", 1600, platform=UPPER)]])
    h.looter.mop_up()
    assert sweeps == []


def test_drops_by_a_monster_wait_for_later(monkeypatch):
    h, sweeps = _mopper(monkeypatch, [[_loot("mesos", 1100)]], monsters=[SimpleNamespace(platform=HERS, x=1150)])
    h.looter.mop_up()
    assert sweeps == [] and [round(d.x) for d in h.looter.pending] == [1100]


def test_a_drop_she_stood_on_twice_is_left(monkeypatch):
    h, sweeps = _mopper(monkeypatch, [[_loot("mesos", 1100)]] * 3)
    h.looter.mop_up()
    assert sweeps == [[1100], [1100]]
    assert [round(u[0]) for u in h.looter.unpickable] == [1100] and not h.looter.inventory_full  # mesos: someone else's
    assert h.looter.wanted([_loot("mesos", 1105)]) == []


def test_items_that_stay_under_her_with_the_game_saying_so_mean_a_full_inventory(monkeypatch):
    h, _ = _mopper(monkeypatch, [[_loot("leather", 1100), _loot("mesos", 900)]] * 2 + [[_loot("leather", 1100)]],
                   text="Your inventory is full.")
    h.looter.mop_up()
    assert h.looter.inventory_full
    # From now on items are left where they lie; mesos are still picked up.
    assert [d.item.kind for d in h.looter.wanted([_loot("leather", 700), _loot("mesos", 760)])] == ["mesos"]


def test_without_the_message_items_stuck_three_times_running_mean_a_full_inventory(monkeypatch):
    failed, unpickable = 0, []
    for n, x in enumerate((1100, 900, 700)):
        h, _ = _mopper(monkeypatch, [[_loot("leather", x)]] * 3)
        h.looter._failed_picks, h.looter.unpickable = failed, unpickable
        h.looter.mop_up()
        failed, unpickable = h.looter._failed_picks, h.looter.unpickable
        assert h.looter.inventory_full == (n == 2)


def test_an_item_picked_up_resets_the_count_of_failed_picks(monkeypatch):
    h, _ = _mopper(monkeypatch, [[_loot("leather", 1100)], []])
    h.looter._failed_picks = 2
    h.looter.mop_up()
    assert h.looter._failed_picks == 0


def test_remembered_drops_seen_gone_twice_are_forgotten_and_drops_seen_elsewhere_noted(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    h = _hunter()
    hers, upper = h.model.platform_under(1000, 500).id, h.model.platform_under(1600, 440).id
    h.looter.pending = [DropSpot(1000, 500, hers, 90.0), DropSpot(2500, 500, hers, 90.0),  # in view, out of view
                      DropSpot(1200, 500, hers, 99.0)]                                 # in view, still landing
    s = SimpleNamespace(loot=[_loot("mesos", 1610, upper), _loot("leather", 1640, upper)],
                        in_view=lambda pid, x: pid == hers and 700 <= x <= 1300)
    h.looter.remember(s)  # one empty look isn't enough: a pig walking over coins hid them in game
    assert [(round(d.x), d.platform, d.mesos, d.on_ground) for d in h.looter.pending] == [
        (1000, hers, False, False), (2500, hers, False, False), (1200, hers, False, False), (1610, upper, True, True)]
    clock[0] += EMPTY_LOOKS
    h.looter.remember(SimpleNamespace(loot=[], in_view=s.in_view))
    assert [round(d.x) for d in h.looter.pending] == [2500, 1200, 1610]  # 1200's drops may still be landing


def test_where_a_kill_died_she_sweeps_even_when_nothing_shows(monkeypatch):
    h, sweeps = _mopper(monkeypatch, [[], []])
    h.looter.walk_piles = lambda s, deaths, margin: sweeps.append(("walk", [round(d.x) for d in deaths])) or []
    h.looter.mop_up = lambda stood=None: None
    here = h.model.platform_under(1000, 500)
    s = SimpleNamespace(where=Where(1000, 500, here, (0, 0)), platform=here, t=0.0, loot=[], monsters=[],
                        in_view=lambda pid, x: True)
    h.look = lambda *a, **k: s
    assert h.looter.collect(DropSpot(1100, 500, HERS, -100.0))               # a kill's spot: hidden, or still falling
    assert h.looter.collect(DropSpot(1300, 500, HERS, -100.0, on_ground=True))    # seen before, gone now: taken
    assert sweeps == [("walk", [1100])]


def test_a_mop_up_sweep_stands_right_on_each_drop(monkeypatch):
    h = _hunter()
    did = []
    h.actions.walk = lambda direction, units: did.append(("walk", direction, round(units)))
    h.actions.pick_here = lambda presses=4: did.append(("pick",))
    here = h.model.platform_under(1000, 500)
    s = SimpleNamespace(where=Where(1000, 500, here, (0, 0)), platform=here)
    stood = h.looter.walk_piles(s, [DropSpot(1100, 500, here.id, 0), DropSpot(1108, 500, here.id, 0),
                              DropSpot(1200, 500, here.id, 0)], margin=0)
    assert stood == [1104, 1200]  # two drops 8 apart are one pile
    assert did == [("walk", "right", 104), ("pick",), ("walk", "right", 96), ("pick",)]


def _touring(h, travel=None):
    """A Tour around Hunter `h` whose trips are recorded; `travel` gives route times (platform -> s)."""
    from gamelab.behavior.tour import Tour, TourReport
    trips = []
    h.looter.collect = lambda d: True
    h.look = lambda *a, **k: SimpleNamespace(loot=[], in_view=lambda pid, x: False)
    tour = Tour.__new__(Tour)
    tour.h, tour.log, tour.report = h, lambda m: None, TourReport()
    here = SimpleNamespace(x=1000, y=500, platform=SimpleNamespace(id=HERS))
    tour.nav = SimpleNamespace(
        mover=SimpleNamespace(locate=lambda: SimpleNamespace(where=here)),
        planner=SimpleNamespace(travel=lambda a, x, b: (travel or {}).get(b)),
        model=h.model,
        go=lambda pid, x, seconds: trips.append((pid, x)) or SimpleNamespace(result="arrived", kills=0))
    return tour, trips


def test_drops_of_a_kill_placed_on_no_platform_are_fetched_from_the_one_below():
    # A monster seen mid-jump, or its feet misread, is placed on no platform; its drops fall onto the
    # platform below, and used to be left (collect_loot skips drops without a platform).
    h = _hunter()
    h.looter.pending = [DropSpot(1100, 470, None, 1e12)]  # 30 above her platform's top
    tour, trips = _touring(h)
    tour.collect_loot()
    assert trips == [(HERS, 1100)]


def test_the_tour_goes_for_mesos_only_once_her_inventory_is_full():
    h = _hunter()
    h.looter.inventory_full = True
    h.looter.pending = [DropSpot(1000, 500, 5, 1e12), DropSpot(1600, 440, 6, 1e12, mesos=True)]
    tour, trips = _touring(h)
    tour.collect_loot()
    assert trips == [(6, 1600)]


def test_far_drops_are_collected_nearest_by_route_first():
    # A far kill's drops on platform 7, just below her but a long way round; platform 8 is farther
    # across, but a short route.
    h = _hunter()
    h.looter.pending = [DropSpot(1100, 700, 7, 1e12), DropSpot(1900, 300, 8, 1e12)]
    tour, trips = _touring(h, travel={7: 25.0, 8: 6.0})
    tour.collect_loot()
    assert trips == [(8, 1900), (7, 1100)]


# --- the game's own pictures, on frames saved in game (local only: game data stays off git) ----------

LAB = ROOT / "gamelab" / "lab"
COPY = ROOT / "gamedata" / "data" / "copy"


@pytest.mark.skipif(not (COPY / "Item").is_dir() or not (LAB / "z2_loot2.png").exists(),
                    reason="needs the game data copy and saved frames")
def test_finds_the_drops_in_frames_saved_in_game():
    d = LootDetector()

    def half(name):
        im = np.asarray(Image.open(LAB / name).convert("RGB"))
        return cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA) if im.shape[1] == 1920 else im

    ground = (0, 375, 960, 411)  # the basin floor's strip (its top at row 405)
    kinds = lambda name: sorted(i.kind for i in d.detect(half(name), ground))
    assert kinds("z2_loot1.png") == ["mesos"] * 3
    assert kinds("z2_loot2.png") == ["leather"] + ["mesos"] * 3
    assert kinds("loot_0_before.png") == ["mesos", "squishy-liquid"]
    assert kinds("loot_0_after.png") == []
