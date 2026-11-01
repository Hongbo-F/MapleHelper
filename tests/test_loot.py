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

import testdata

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
    from gamelab.perception import drops
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
    assert drops.unknown(odd) == ["Jr. Necki"]
    assert drops.unknown_note(odd).startswith("drops not known for Jr. Necki ") and drops.unknown_note(tree) is None


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


def test_a_maps_detector_builds_only_the_pictures_missing(tmp_path, monkeypatch):
    # The folder is every map's, and a test run's workers make their maps' detectors at once: one map's build must not
    # write over the pictures another's detector is reading (the coins and potions are every map's).
    from gamelab.perception import loot
    calls = []
    monkeypatch.setattr(loot, "build", lambda folder, stretch, items, coins=True: calls.append((items, coins)))
    for name in ("squishy-liquid_0", "mesos_bronze0"):
        (tmp_path / f"{name}.png").write_bytes(b"")
    assert loot.ensure(tmp_path, 1.0, ("Squishy Liquid",)) == tmp_path and calls == []
    loot.ensure(tmp_path, 1.0, ("Squishy Liquid", "Leather"))
    (tmp_path / "mesos_bronze0.png").unlink()
    loot.ensure(tmp_path, 1.0, ("Squishy Liquid",))
    assert calls == [(("Leather",), False), ((), True)]


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


def test_of_two_recoloured_pictures_found_at_one_place_the_nearer_colours_win():
    # An Iron Ore's picture is a Silver Ore's recoloured: on brightness both score alike where either lies, and their
    # colours are 12 apart, under PIXEL_SLACK. Where a monster hides part of a Silver Ore, Iron's could score higher.
    from gamelab.perception.loot import SAME_SHAPE, _recoloured
    silver, iron = Item("silver-ore", 100, 50, 25, 21, 0.951), Item("iron-ore", 100, 51, 25, 21, 0.953)
    assert _recoloured([(silver, 6.0), (iron, 15.0)]) == [silver]
    assert _recoloured([(silver, 16.0), (iron, 4.0)]) == [iron]
    # Not one picture recoloured (another size), apart (2 px), or shapes clearly apart: the scores decide, as before.
    for other in (Item("iron-ore", 100, 50, 22, 22, 0.953), Item("iron-ore", 102, 50, 25, 21, 0.953),
                  Item("iron-ore", 100, 50, 25, 21, 0.951 + SAME_SHAPE + 0.001)):
        assert _recoloured([(silver, 6.0), (other, 15.0)]) == [silver, other]


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


def test_her_inventory_is_full_by_tab_and_an_item_earned_into_one_says_it_has_room(monkeypatch):
    """Her inventory by tab (looting.ITEM_TABS, from 2026-10-31): Etc items stuck under her with the game's full message
    (which names no tab) mark her Etc tab full, not her Use tab: Etc items are left, a Use item and mesos still wanted,
    and drops still looked for as all kinds (items_full only once every item tab is). Later a Use item stuck with the
    message and an "earned. (Etc)" line after it in the same read (her pets' pick-up, say): Use full, Etc with room
    again. And a tab full TAB_RETRY_S ago is taken as having room again."""
    from gamelab.behavior import looting
    full = "You can't get anymore items."
    h, _ = _mopper(monkeypatch, [[_loot("leather", 1100), _loot("mesos", 900)]] * 2 + [[_loot("leather", 1100)]],
                   text=full)
    h.looter.mop_up()
    assert set(h.looter.full) == {"Etc"} and h.looter.inventory_full and not h.looter.items_full
    left = h.looter.wanted([_loot("leather", 700), _loot("red-potion", 720), _loot("mesos", 760)])
    assert [d.item.kind for d in left] == ["red-potion", "mesos"]
    h.senses.eyes.text = full + "\nLeather x1 earned. (Etc)"
    h.looter.cant_pick([_loot("red-potion", 720)])
    assert set(h.looter.full) == {"Use"} and h.looter.can_take("leather") and not h.looter.can_take("red-potion")
    when, why = h.looter.full["Use"]
    h.looter.full["Use"] = (when - looting.TAB_RETRY_S, why)
    assert h.looter.can_take("red-potion") and not h.looter.full


def test_the_pickup_log_is_read_in_its_order(monkeypatch):
    """The review of 2026-10-31: the log's lines oldest first (looting.ITEM_TABS), so the leather that filled her Etc tab
    ("earned") and then the next one refused, both in the read, mark Etc full (until then any "earned. (Etc)" line in the
    read kept it from being marked); the leather is refused (its stack full too) though earned."""
    full = "You can't get anymore items."
    h, _ = _mopper(monkeypatch, [], text="Leather x1 earned. (Etc)\n" + full)
    h.looter.cant_pick([_loot("leather", 1100)])
    assert set(h.looter.full) == {"Etc"} and not h.looter.can_take("leather") and h.looter.can_take("red-potion")


def test_a_full_tab_still_takes_a_kind_she_holds_a_stack_of(monkeypatch):
    """v83: a full tab has no free slot, but a stack below its most takes more (looting.ITEM_TABS, the review of
    2026-10-31). The kinds she stood on while their tab had room (Looter.stood_on: she holds them now) are still taken
    once Etc is full on another kind's refusal; a kind she doesn't hold isn't; one of hers that stays under her twice
    with Etc full is refused (no read of the log: its stack full too), and an "earned" line of it takes it back."""
    full = "You can't get anymore items."
    h, _ = _mopper(monkeypatch, [], text=full)
    h.looter.stood_on(["horny-mushroom-cap", "mesos"])
    assert h.looter.held == {"horny-mushroom-cap"}
    h.looter.cant_pick([_loot("leather", 1100)])
    assert set(h.looter.full) == {"Etc"} and not h.looter.can_take("leather")
    assert h.looter.can_take("horny-mushroom-cap") and not h.looter.can_take("green-mushroom-cap")
    reads = []
    h.senses.eyes.messages = lambda: reads.append(1) or full
    h.looter.cant_pick([_loot("horny-mushroom-cap", 1100)])
    assert not h.looter.can_take("horny-mushroom-cap") and not reads and set(h.looter.full) == {"Etc"}
    h.senses.eyes.messages = lambda: "Horny Mushroom Cap x1 earned. (Etc)\n" + full
    h.looter.cant_pick([_loot("red-potion", 720)])
    assert h.looter.can_take("horny-mushroom-cap") and set(h.looter.full) == {"Etc", "Use"}


def test_a_kind_is_hers_once_the_looks_show_it_gone_from_the_pile_she_stood_on(monkeypatch):
    """Looter.stood_on with the pile (the hunt's stands): a kind counts as one she holds a stack of (`held`) only once
    the looks after show it gone from that pile (Looter.on_look, after the drop store's: the pile seen without it, or
    gone), not on the stand alone: with her Etc tab full from the start, every kind she stood on before she learned it
    was taken for hers (the review of 2026-10-31); one still lying there isn't, and a stand the looks don't confirm
    within PICK_SEEN_S is let go."""
    from gamelab.behavior import looting
    from gamelab.behavior.ledger import Pile
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    h, _ = _mopper(monkeypatch, [])
    took = Pile(500.0, 0.0, HERS, 90.0, True, status="seen", last_seen=99.0, items=("horny-mushroom-cap",))
    left = Pile(700.0, 0.0, HERS, 90.0, True, status="seen", last_seen=99.0, items=("leather", "green-mushroom-cap"))
    gone = Pile(900.0, 0.0, HERS, 90.0, True, status="seen", last_seen=99.0, items=("blue-mushroom-cap",))
    late = Pile(1100.0, 0.0, HERS, 90.0, True, status="seen", last_seen=99.0, items=("snail-shell",))
    for p in (took, left, gone, late):
        h.looter.stood_on(list(p.items), p)
    assert not h.looter.held
    clock[0] = 101.0
    took.items, took.last_seen = (), 101.0
    left.items, left.last_seen = ("leather",), 101.0
    gone.status = "gone"
    h.looter.on_look(None)
    assert h.looter.held == {"horny-mushroom-cap", "green-mushroom-cap", "blue-mushroom-cap"}
    clock[0] = 100.0 + looting.PICK_SEEN_S + 1.0
    h.looter.on_look(None)
    assert not h.looter._stands and "snail-shell" not in h.looter.held


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
    remembered for later, unless one is valuable on the map (looting.Worth.valuable: a Slime Bubble); with her party's
    leader there, or nobody, they are."""
    from gamelab.behavior import looting
    from gamelab.behavior.looting import FIRST, worth
    from gamelab.localization.players import Player
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(looting, "PLAYER_PILES", True)
    assert worth("slime-bubble") >= FIRST.valuable > worth("mesos")
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
    from gamelab.behavior.looting import COINS, UNKNOWN_WORTH, worth
    assert worth("slime-bubble") == 50 and worth("squishy-liquid") == 6 and worth("red-potion") == 5
    assert worth("mesos") == worth("mesos", "bronze") == COINS["bronze"] and worth("mesos", "gold") == 75
    # Not the dearest drop known (the Magic Potion's 250: every drop not known would be valuable), the first map's.
    assert worth("a scroll the table doesn't know") == UNKNOWN_WORTH == 50 == worth("magic-potion") / 5


def test_on_each_map_mesos_are_worth_what_its_monsters_drop():
    """looting.coins_for: a coin stands for the mean of the amounts it's drawn for (bronze under 50, gold 50-99) that the
    map's monsters drop (perception.drops), by their spawn points; a drop is valuable at twice a bronze coin."""
    from gamelab.behavior.looting import worth_for
    tree = worth_for(_map(10002073, *["Horny Mushroom"] * 15, *["Green Mushroom"] * 14, *["Slime"] * 2))
    assert tree.coins == {"bronze": 33.4, "gold": 51.5} and tree.valuable == 66.8
    assert tree("mesos") == 33.4 and tree("mesos", "gold") == 51.5 and tree("sapphire-ore") == 120
    assert tree("slime-bubble") < tree.valuable < tree("topaz-ore")  # a bubble is under two of its coins there


def test_every_maps_drops_have_a_price():
    from gamelab.behavior.looting import PRICES
    from gamelab.perception import drops
    from gamelab.perception.kinds import kind_of
    names = {n for m in drops.MONSTERS.values() for n in m.items} | set(drops.POTIONS)
    names |= {n for items in drops.PINNED.values() for n in items}
    assert {kind_of(n) for n in names} == set(PRICES)


@pytest.mark.skipif(not (COPY / "Item").is_dir(), reason="needs the game data copy")
def test_a_drops_picture_is_written_whole_as_the_games(tmp_path):
    """loot.build at the game's own size: the game's picture as it is (the Fragment of Magic's glow partly see-through),
    written under another name and renamed (nothing of it left behind), and the coins only when asked for."""
    from gamelab.perception import loot
    loot.build(tmp_path, 1.0, ("Fragment of Magic",), coins=False)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["fragment-of-magic_0.png", "scale.json"]
    alpha = np.asarray(Image.open(tmp_path / "fragment-of-magic_0.png").convert("RGBA"))[..., 3]
    assert ((alpha > 0) & (alpha < 255)).sum() > 100
    loot.build(tmp_path, 1.0, ())
    assert {loot._kind(f) for f in tmp_path.glob("*.png")} == {"fragment-of-magic", "mesos"}
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.skipif(not (COPY / "Item").is_dir(), reason="needs the game data copy")
def test_the_prices_are_the_game_data_s():
    """looting.PRICES against each item's info/price in the game data (the first id of its name with one), every map's
    drops'."""
    from gamedata.wz import Package
    from gamelab.behavior.looting import PRICES
    from gamelab.perception import drops
    from gamelab.perception.kinds import kind_of
    from gamelab.perception.loot import item_ids
    names = sorted({n for m in drops.MONSTERS.values() for n in m.items} | set(drops.POTIONS)
                   | {n for items in drops.PINNED.values() for n in items})
    ids, found = item_ids(COPY), {}
    for folder in ("Etc", "Consume"):
        pack = Package(COPY / "Item" / folder / f"{folder}_000.wz")
        for name in names:
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


# --- the new maps' drops, the game's own pictures, on frames of those maps (Tree Dungeon's are test data:
# tests/testdata.py) -----------------------------------------------------------------------------------------------

TREE_DUNGEON = testdata.TREE_DUNGEON  # 10002073 at 1366x768: 24 frames of 2026-10-28 (a test data set), no drop on any
WEST_ROCKY = ROOT / "tests" / "fixtures" / "native"  # 10004021 at 1366x768: 3 frames, no drop on any (a background)


def frames_of(where) -> list[Path]:
    """The frames of a folder, or of a test data set (its name): those its labels name, where they are
    (testdata.labelled: the test fails without them)."""
    return sorted(where.glob("*.png")) if isinstance(where, Path) else sorted(testdata.labelled(where)[1].values())


@pytest.mark.skipif(not (COPY / "Item").is_dir(), reason="needs the game data copy")
@pytest.mark.parametrize("map_id,frames", [(10002073, TREE_DUNGEON)])
def test_a_new_maps_drops_drawn_as_the_game_draws_them_are_found_and_nothing_else_is(map_id, frames):
    """Each of the map's drops (perception.drops) and the coins' frames, the game's own pictures drawn 1:1 onto frames
    of the map at the game's own 1366x768, is found in the strip above where it lies, as its kind and where drawn; and
    on the frames as they are no drop is found anywhere above the chat bar (row 700, where the search goes down to)."""
    from gamelab.analysis import pasted
    from gamelab.localization import mapdata
    from gamelab.perception import loot
    from gamelab.perception.drops import items_for
    from gamelab.screen import NATIVE
    files = frames_of(frames)
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
        assert det.detect(full, (0, 0, full.shape[1], int(full.shape[0] * NATIVE.hud_top))) == []


@pytest.mark.skipif(not (COPY / "Item").is_dir(), reason="needs the game data copy")
def test_tree_dungeons_drops_seen_in_game_are_found_as_their_kinds_and_nothing_else_is():
    """The drops lying on frames saved in game on Tree Dungeon (tests/fixtures/tree_dungeon_drops.json: 13 drops, 5 Horny
    Mushroom Caps, a Bronze Ore, a Green Mushroom Cap, an Orange Potion, 4 bronze coins and a gold one, seen 39 times in
    plain view on five frames 24 s apart; players, pets and the portal's light about them), searched for over the whole
    screen above the chat bar by the map's detector: every one in plain view is found as its kind, within 2 px of where
    it was found when they were labelled (the labels' places are the detector's, checked by eye; their kinds were read by
    eye), and nothing where none lies. The first map's detector found the coins and the potion alone, 12 of the 39
    sightings, 6 of the 13 drops."""
    from gamelab.localization import mapdata
    from gamelab.perception import loot
    from gamelab.screen import NATIVE
    labels, frames = testdata.labelled(testdata.TREE_DUNGEON_DROPS)
    det = loot.detector_for(mapdata.load(10002073), NATIVE)
    shown, found_shown = 0, 0
    for name, drops in labels.items():
        frame = np.asarray(Image.open(frames[name]).convert("RGB"))
        found = det.detect(frame, (0, 0, frame.shape[1], 700))

        def near(d, i, px):
            return (i.kind, i.coin) == (d["kind"], d.get("coin")) and abs(i.x - d["x"]) <= px and abs(i.y - d["y"]) <= px

        for d in drops:
            if "optional" not in d:
                shown += 1
                found_shown += any(near(d, i, 2) for i in found)
        assert all(any(near(d, i, 4) for d in drops) for i in found), (name, found)  # nothing that isn't there
    assert (found_shown, shown) == (39, 39)
    # the same drops in every frame: the camera moved up and down, not across
    assert len({(d["kind"], d.get("coin"), d["x"]) for ds in labels.values() for d in ds if "optional" not in d}) == 13


@pytest.mark.skipif(not (COPY / "Item").is_dir(), reason="needs the game data copy")
def test_a_silver_ore_partly_hidden_is_not_taken_for_an_iron_ore():
    """The game's Silver Ore drawn on West Rocky Mountain II, its left 5 columns hidden (a pale patch: a monster in front
    of it): Iron Ore's picture, the same picture recoloured, scored higher on brightness there, and its colours passed."""
    from gamelab.analysis import pasted
    from gamelab.perception import loot
    from gamelab.screen import NATIVE
    det = loot.LootDetector(geo=NATIVE, items=("Silver Ore", "Iron Ore"))
    ore = {n: rgba for n, rgba, _ in pasted.pictures(loot.folder_for(1), {"silver-ore"})}["silver-ore_0"]
    frame = np.array(Image.open(WEST_ROCKY / "n28_000.png").convert("RGB"))
    pasted.paste(frame, ore, 200, 300)
    frame[300:321, 200:205] = 220
    assert [i.kind for i in det.detect(frame, (160, 280, 265, 327))] == ["silver-ore"]


@pytest.mark.skipif(not (COPY / "Item").is_dir(), reason="needs the game data copy")
def test_a_drops_twins_are_the_items_whose_picture_is_its_picture_recoloured():
    """loot.twins_of: the ores beside a Silver Ore in the game data whose colours are within TWIN_SLACK of its own (an
    Iron Ore 12.1 off, a Bronze 24.0, a Mithril 30.7, an Orihalcon 37.2; Adamantium and Gold are 50.7 and 54.3), a
    Magic Potion's the Warrior Potion (19.6), and the first map's drops none: its potions are one another's (all held),
    the snail shells differ in shape, and the caps, the leather and the slimes' drops have no picture of their size."""
    from gamelab.perception import loot
    twins = loot.twins_of(("Silver Ore", "Magic Potion") + loot.ITEMS)
    assert [name for name, _ in twins["Silver Ore"]] == ["Bronze Ore", "Iron Ore", "Mithril Ore", "Orihalcon Ore"]
    assert [name for name, _ in twins["Magic Potion"]][0] == "Warrior Potion"
    held = set(loot.ITEMS)
    assert all(name in held for item in loot.ITEMS for name, _ in twins[item])


@pytest.mark.skipif(not (COPY / "Item").is_dir(), reason="needs the game data copy")
def test_a_twin_of_a_tree_dungeon_drop_is_not_taken_for_it():
    """Each twin of Tree Dungeon's drops that the map's detector doesn't look for (an Iron Ore, an Aquamarine, a Diamond,
    an Opal, a Warrior Potion, ...), the game's picture drawn 1:1 onto a frame of the map: nothing found. Its colours had
    passed the held kind's check, and an Iron Ore was found as a Silver Ore (0.998), an Aquamarine as an Emerald (0.953),
    a Diamond and an Opal as a Sapphire (0.993, 0.929), a Warrior Potion as a Magic Potion (0.996). The held kinds drawn
    the same way are still found as themselves."""
    from gamelab.analysis import pasted
    from gamelab.perception import loot
    from gamelab.screen import NATIVE
    files = frames_of(TREE_DUNGEON)
    det = loot.detector_for(mapdata.load(10002073), NATIVE)
    folder, held = loot.folder_for(NATIVE.stretch), {t.kind for t in det.templates}
    listed = {k for t in held - {loot.MESOS} for k in (folder / loot.TWINS / f"{t}.txt").read_text().split()}
    assert {"iron-ore", "aquamarine-ore", "diamond-ore", "opal-ore", "warrior-potion"} <= listed - held
    background = np.asarray(Image.open(files[0]).convert("RGB"))
    for kind in sorted(listed - held) + sorted(held & {"silver-ore", "emerald-ore", "sapphire-ore", "magic-potion"}):
        frame = background.copy()
        rgba = np.asarray(Image.open(folder / (loot.TWINS if kind not in held else ".") / f"{kind}_0.png"))
        pasted.paste(frame, rgba, 400, 300)
        found = det.detect(frame, (370, 270, 430 + rgba.shape[1], 330 + rgba.shape[0]))
        assert [i.kind for i in found] == ([kind] if kind in held else []), kind
