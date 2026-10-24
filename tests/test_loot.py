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


def test_coins_say_which_coin_they_are_drawn_as(tmp_path):
    # The amount picks the coin (bronze, gold): what they're worth (behavior.looting.worth) goes by it.
    Image.fromarray(coin(), "RGBA").save(tmp_path / "mesos_bronze0.png")
    Image.fromarray(coin(colour=(235, 190, 40)), "RGBA").save(tmp_path / "mesos_gold1.png")
    Image.fromarray(leather(), "RGBA").save(tmp_path / "leather_0.png")
    frame = scene()
    paste(frame, coin(), 300, 300)
    paste(frame, coin(colour=(235, 190, 40)), 600, 300)
    paste(frame, leather(), 800, 300)
    found = LootDetector(tmp_path).detect(frame, (0, 270, 960, 306))
    assert sorted((i.x, i.kind, i.coin) for i in found) == [(300, "mesos", "bronze"), (600, "mesos", "gold"),
                                                             (800, "leather", None)]


def test_drops_in_several_boxes_searched_together_are_each_boxs(detector):
    frame = scene()
    for x in (100, 300, 700):
        paste(frame, coin(), x, 300)
    paste(frame, leather(), 520, 300)
    boxes = [(0, 270, 400, 306), (250, 270, 960, 306), (600, 280, 800, 306), (0, 0, 20, 20), (480, 250, 560, 306)]
    for mesos_only in (False, True):
        each = detector.detect_each(frame, boxes, mesos_only)
        assert each == [detector.detect(frame, box, mesos_only) for box in boxes]
    assert [len(found) for found in detector.detect_each(frame, boxes)] == [2, 3, 1, 0, 1]


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


def _map(map_id, *names):
    """A map's model as drops.items_for reads it: its id, and its spawn points' monster names."""
    return SimpleNamespace(id=map_id, spawns=[SimpleNamespace(name=n) for n in names])


def test_each_map_looks_for_its_own_monsters_drops():
    from gamelab.perception import drops, loot
    # The first map's, as they were: what was seen there in game, and the potions.
    first = _map(drops.FIRST_MAP, "Snail", "Slime", "Pig", "Orange Mushroom")
    assert drops.items_for(first) == loot.ITEMS == (
        "Squishy Liquid", "Slime Bubble", "Leather", "Pig's Head", "Pig's Ribbon", "Orange Mushroom Cap",
        "Snail Shell", "Red Potion", "Orange Potion", "White Potion", "Blue Potion")
    # Tree Dungeon's: its monsters' drops in the order they spawn (each item once), then the potions not named yet.
    tree = _map(10002073, "Horny Mushroom", "Green Mushroom", "Horny Mushroom", "Slime")
    assert drops.items_for(tree) == (
        "Horny Mushroom Cap", "Orange Potion", "Bronze Ore", "Blue Potion", "Magic Potion", "Topaz Ore",
        "Sapphire Ore", "Fragment of Magic", "Green Mushroom Cap", "Silver Ore", "Red Potion", "Emerald Ore",
        "Orihalcon Ore", "Slime Bubble", "Squishy Liquid", "White Potion")
    assert drops.monsters_of(tree) == {"Horny Mushroom": 2, "Green Mushroom": 1, "Slime": 1}
    # A monster the table doesn't know adds nothing, and is named.
    odd = _map(5, "Stump", "Jr. Necki")
    assert drops.items_for(odd) == drops.MONSTERS["Stump"].items + ("Orange Potion", "White Potion", "Blue Potion")
    assert drops.unknown(odd) == ["Jr. Necki"] and drops.unknown(first) == []


def test_a_drop_whose_kind_cant_be_a_file_name_is_refused(tmp_path):
    # "omok-piece:-slime_0.png" would be an alternate data stream of a file "omok-piece" on Windows, and no folder
    # listing shows it: the template would be gone without a word.
    from gamelab.perception import drops, loot
    assert not drops.safe("Omok Piece: Slime") and drops.safe("Pig's Head")
    assert all(drops.safe(n) for m in drops.MONSTERS.values() for n in m.items)
    assert all(drops.safe(n) for items in drops.PINNED.values() for n in items)
    with pytest.raises(ValueError, match="Omok Piece"):
        loot.build(tmp_path, 1.0, ("Squishy Liquid", "Omok Piece: Slime"))
    assert list(tmp_path.iterdir()) == []


def test_a_detector_holds_its_maps_drops_only(tmp_path):
    # Every map's drops are built into one folder a size: a map's detector takes its own from it, and mesos.
    Image.fromarray(coin(), "RGBA").save(tmp_path / "mesos_bronze0.png")
    Image.fromarray(leather(), "RGBA").save(tmp_path / "leather_0.png")
    Image.fromarray(leather(colour=(40, 90, 160)), "RGBA").save(tmp_path / "horny-mushroom-cap_0.png")
    assert {t.kind for t in LootDetector(tmp_path).templates} == {"mesos", "leather", "horny-mushroom-cap"}
    held = LootDetector(tmp_path, items=("Leather",))
    assert {t.kind for t in held.templates} == {"mesos", "leather"} and held.items == ("Leather",)


def _native_folder(tmp_path, pictures: dict) -> Path:
    """A folder of made-up drop pictures at the game's own size (scale.json 1.0, as loot.build makes at stretch 1)."""
    import json
    for name, rgba in pictures.items():
        Image.fromarray(rgba, "RGBA").save(tmp_path / f"{name}.png")
    (tmp_path / "scale.json").write_text(json.dumps({"px_per_game_px": 1.0}), encoding="utf-8")
    return tmp_path


def test_a_drop_drawn_partly_see_through_is_matched_on_the_px_drawn_as_they_are(tmp_path):
    # Tree Dungeon's Fragment of Magic is a crystal in a glow the game blends with what lies behind it. At the game's own
    # size its template is the px it draws as they are: there it scores 1.00 whatever lies behind the glow.
    from gamelab.analysis.pasted import paste
    from gamelab.screen import NATIVE
    rgba = np.zeros((30, 30, 4), np.uint8)
    yy, xx = np.mgrid[:30, :30]
    r = np.hypot(yy - 14.5, xx - 14.5)
    rgba[r < 14] = (150, 220, 255, 140)                         # the glow, half see-through
    rgba[r < 9] = (90, 120, 230, 255)                           # the crystal
    rgba[(r < 9) & (np.abs(xx - yy) < 2)] = (240, 240, 255, 255)  # its shine
    det = LootDetector(_native_folder(tmp_path, {"fragment-of-magic_0": rgba}), NATIVE)
    t = det.templates[0]
    assert t.solid.sum() == (rgba[..., 3] == 255).sum()
    for back in ((30, 30, 30), (230, 200, 120), (60, 140, 230)):
        frame = np.zeros((200, 300, 3), np.uint8)
        frame[:] = back
        paste(frame, rgba, 120, 100)
        found = det.detect(frame)
        assert [(i.kind, i.x, i.y, i.score) for i in found] == [("fragment-of-magic", 135, 115, 1.0)], back


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


def test_her_senses_tell_her_eyes_the_maps_drops(monkeypatch):
    # The loot detector, made on first use, looks for what the map's monsters drop (none spawn on this one: the potions).
    from gamelab.perception import eyes as eyes_module
    from gamelab.perception.drops import POTIONS
    made = []
    monkeypatch.setattr(eyes_module, "LootDetector", lambda **kw: made.append(kw) or "a detector")
    eyes = _Eyes()
    _hunter(eyes)
    assert eyes.drop_names == POTIONS
    assert eyes.loot_detector() == "a detector" and made == [{"geo": eyes.geo, "items": POTIONS}]


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
                                             loot=next(seq, []), monsters=list(monsters), players=[])

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
                        in_view=lambda pid, x: pid == hers and 700 <= x <= 1300, players=[])
    h.looter.remember(s)  # one empty look isn't enough: a pig walking over coins hid them in game
    assert [(round(d.x), d.platform, d.mesos, d.on_ground) for d in h.looter.pending] == [
        (1000, hers, False, False), (2500, hers, False, False), (1200, hers, False, False), (1610, upper, True, True)]
    clock[0] += EMPTY_LOOKS
    h.looter.remember(SimpleNamespace(loot=[], in_view=s.in_view, players=[]))
    assert [round(d.x) for d in h.looter.pending] == [2500, 1200, 1610]  # 1200's drops may still be landing


def test_the_tour_doesnt_come_back_for_drops_by_another_player_unless_one_is_worth_it(monkeypatch):
    """looting.PLAYER_PILES (off by default): drops seen with another player by them (400 units across) aren't
    remembered for later, unless one is worth PLAYER_PILE_WORTH (a Slime Bubble); with her party's leader there, or
    nobody, they are."""
    from gamelab.behavior import looting
    from gamelab.behavior.looting import PLAYER_PILE_WORTH, worth
    from gamelab.localization.players import Player
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(looting, "PLAYER_PILES", True)
    assert worth("slime-bubble") >= PLAYER_PILE_WORTH > worth("mesos")
    seen = [_loot("mesos", 1610, UPPER), _loot("slime-bubble", 1900, UPPER), _loot("leather", 2400, UPPER)]
    for kind, kept in (("stranger", [1900, 2400]), ("party", [1610, 1900, 2400]), (None, [1610, 1900, 2400])):
        h = _hunter()
        players = [] if kind is None else [Player(1, kind, 1700.0, 440.0, UPPER, (0.0, 0.0), 0.0, 1.0, seen=2)]
        h.looter.remember(SimpleNamespace(loot=seen, in_view=lambda pid, x: False, players=players))
        assert [round(d.x) for d in h.looter.pending] == kept, kind


def test_where_a_kill_died_she_sweeps_even_when_nothing_shows(monkeypatch):
    h, sweeps = _mopper(monkeypatch, [[], []])
    h.looter.walk_piles = lambda s, deaths, margin: sweeps.append(("walk", [round(d.x) for d in deaths])) or []
    h.looter.mop_up = lambda stood=None: None
    here = h.model.platform_under(1000, 500)
    s = SimpleNamespace(where=Where(1000, 500, here, (0, 0)), platform=here, t=0.0, loot=[], monsters=[],
                        in_view=lambda pid, x: True, players=[])
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
    """A Tour around Hunter `h` whose trips are recorded; `travel` gives route times (platform -> s). A trip for drops
    goes the quickest way (not one of the ways a trip may choose among: route.ROUTES)."""
    from gamelab.behavior.tour import Tour, TourReport
    trips = []
    h.looter.collect = lambda d: True
    h.look = lambda *a, **k: SimpleNamespace(loot=[], in_view=lambda pid, x: False, players=[])
    tour = Tour.__new__(Tour)
    tour.h, tour.log, tour.report = h, lambda m: None, TourReport()
    here = SimpleNamespace(x=1000, y=500, platform=SimpleNamespace(id=HERS))
    tour.nav = SimpleNamespace(
        mover=SimpleNamespace(locate=lambda: SimpleNamespace(where=here)),
        planner=SimpleNamespace(travel=lambda a, x, b: (travel or {}).get(b)),
        model=h.model,
        go=lambda pid, x, seconds, vary=True: trips.append((pid, x) if not vary else ("varied", pid, x))
        or SimpleNamespace(result="arrived", kills=0))
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


def test_a_drop_is_worth_its_shop_price_and_mesos_by_their_coin():
    from gamelab.behavior.looting import COINS, PRICES, UNKNOWN_WORTH, worth
    assert worth("slime-bubble") == 50 and worth("squishy-liquid") == 6 and worth("red-potion") == 5
    assert worth("mesos") == worth("mesos", "bronze") == COINS["bronze"] and worth("mesos", "gold") == 75
    assert worth("a scroll the table doesn't know") == UNKNOWN_WORTH == max(PRICES.values())


@pytest.mark.skipif(not (COPY / "Item").is_dir(), reason="needs the game data copy")
def test_the_prices_are_the_game_data_s():
    """looting.PRICES against each item's info/price in the game data (the first id of its name with one)."""
    from gamedata.wz import Package
    from gamelab.behavior.looting import PRICES
    from gamelab.perception.kinds import kind_of
    from gamelab.perception.loot import ITEMS, item_ids
    ids, found = item_ids(COPY), {}
    for folder in ("Etc", "Consume"):
        pack = Package(COPY / "Item" / folder / f"{folder}_000.wz")
        for name in ITEMS:
            for item in ids.get(name, []):
                if kind_of(name) in found:
                    break
                try:
                    entry = pack.image(f"{item // 10_000:04d}.img").get(f"{item:08d}")
                except (KeyError, ValueError):
                    continue
                if entry is not None and "price" in entry.get("info", {}):
                    found[kind_of(name)] = entry["info"]["price"]
    assert found == PRICES


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


# --- the new maps' drops, the game's own pictures, on frames of those maps (Tree Dungeon's stay in
# D:/MapleHelper-scratch: skipped without them) ------------------------------------------------------------------

TREE_DUNGEON = Path("D:/MapleHelper-scratch/map3/native")  # 10002073 at 1366x768: 20 frames, no drop on any
WEST_ROCKY = ROOT / "tests" / "fixtures" / "native"         # 10004021 at 1366x768: 3 frames, no drop on any


@pytest.mark.skipif(not (COPY / "Item").is_dir(), reason="needs the game data copy")
@pytest.mark.parametrize("map_id,frames", [(10004021, WEST_ROCKY), (10002073, TREE_DUNGEON)])
def test_a_new_maps_drops_drawn_as_the_game_draws_them_are_found_and_nothing_else_is(map_id, frames):
    """Each of the map's drops (perception.drops) and the coins' frames, the game's own pictures drawn 1:1 onto frames
    of the map at the game's own 1366x768, is found in the strip above where it lies, as its kind and where drawn; and
    on the frames as they are no drop is found anywhere above the bottom bar."""
    from gamelab.analysis import pasted
    from gamelab.localization import mapdata
    from gamelab.perception import loot
    from gamelab.perception.drops import items_for
    from gamelab.screen import HUD_TOP, NATIVE
    files = sorted(frames.glob("*.png")) if frames.is_dir() else []
    if not files:
        pytest.skip(f"needs frames of map {map_id} in {frames}")
    items = items_for(mapdata.load(map_id))
    det = loot.LootDetector(geo=NATIVE, items=items)
    assert {t.kind for t in det.templates} == loot.kinds_of(items)
    backgrounds = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    result = pasted.drops(backgrounds, {"native": (det, NATIVE)}, loot.folder_for(NATIVE.stretch), each=2, seed=3,
                          kinds=loot.kinds_of(items))
    tally = result["tally"]["native"]
    assert set(tally) == loot.kinds_of(items)
    for kind, t in tally.items():
        assert (t.found, t.as_drawn, t.extra) == (t.drawn, t.drawn, 0) and min(t.scores) == 1.0, kind
        assert max(max(abs(dx), abs(dy)) for dx, dy in t.errors) <= 0.5, kind
    assert result["clean"]["native"] == []
    for full in backgrounds:
        assert det.detect(full, (0, 0, full.shape[1], int(full.shape[0] * HUD_TOP))) == []
