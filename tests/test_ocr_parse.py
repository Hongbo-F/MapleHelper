import re

import pytest

from maplehelper.ocr.engine import Row
from maplehelper.ocr.item_names import ITEM_NAMES
from maplehelper.ocr.parse import level_from_exp, match_name, match_prefix, parse_hud, parse_loot


def rows(*texts):
    return [Row(t, 0.99, 0, i * 20.0, 12, 0, 0) for i, t in enumerate(texts)]


# Made-up EXP table (level -> EXP needed for the next level), only for these tests.
EXP_TABLE = {9: 21000, 10: 27000, 11: 34000}


def test_hud_reads_hp_mp_exp():
    r = parse_hud(rows("HP 1234/1500", "MP 567/890", "EXP 12345 [45.67%]"))
    assert (r.hp, r.hp_max, r.mp, r.mp_max) == (1234, 1500, 567, 890)
    assert (r.exp, r.exp_percent) == (12345, 45.67)


def test_hud_folds_misread_digits_and_separators():
    r = parse_hud(rows("HP 1,234 / 15OO", "M P 56l/89O", "EXP 12345[45,67%]"))
    assert (r.hp, r.hp_max, r.mp, r.mp_max) == (1234, 1500, 561, 890)
    assert r.exp_percent == 45.67


def test_hud_drops_impossible_values():
    r = parse_hud(rows("HP 1600/1500", "EXP 100 [150.00%]", "MP 5/0"))
    assert r.hp is None and r.mp is None
    assert r.exp is None and r.exp_percent is None


def test_not_the_bar():
    assert parse_hud(rows("Cash Shop", "Menu")) is None
    assert parse_hud([]) is None


def test_level_from_exp_needs_one_clear_match():
    assert level_from_exp(13500, 50.0, EXP_TABLE) == 10        # 13500 / 50% = 27000
    assert level_from_exp(13600, 50.0, EXP_TABLE) == 10        # within 2%
    assert level_from_exp(15000, 50.0, EXP_TABLE) is None      # 30000 matches nothing
    assert level_from_exp(270, 0.5, EXP_TABLE) is None         # under 1%: too coarse
    assert level_from_exp(100, 50.0, {1: 200, 2: 202}) is None  # two levels fit


def test_hud_level_prefers_exp_over_stylized_digits():
    r = parse_hud(rows("LV. 00", "HP 50/50", "EXP 13500 [50.00%]"), EXP_TABLE)
    assert (r.level, r.level_source) == (10, "exp")


def test_hud_level_read_is_kept_only_when_exp_fits():
    assert parse_hud(rows("LV. 11", "HP 50/50", "EXP 100 [0.30%]"), EXP_TABLE).level == 11
    assert parse_hud(rows("LV. 9", "HP 50/50", "EXP 25000 [0.50%]"), EXP_TABLE).level is None  # 25000 > 21000
    assert parse_hud(rows("LV. 7", "HP 50/50"), None).level == 7


# With the real Classic World table (the default): level 19 needs 16,680 EXP for level 20.

def test_classic_level_19_from_exp_although_read_as_09():
    r = parse_hud(rows("LV. 09", "HP 812/900", "MP 300/420", "EXP 13500 [80.93%]"))
    assert (r.level, r.level_source) == (19, "exp")


def test_classic_level_digits_repaired_when_percent_is_too_small():
    # Right after a level up the percentage is too coarse to pin the level; the read "09" is repaired to 19,
    # which the EXP fits (9 can't hold 50 EXP at 0.29%).
    r = parse_hud(rows("LV.09", "HP 812/900", "EXP 50 [0.29%]"))
    assert (r.level, r.level_source) == (19, "read")


def test_classic_level_digits_alone_left_of_the_gauges():
    bar = [Row("09", 0.9, 30, 20, 18, 20, 40), Row("Rogue", 0.99, 90, 20, 14, 60, 120),
           Row("HP 812/900", 1.0, 400, 20, 14, 350, 450), Row("EXP 50 [0.29%]", 1.0, 800, 20, 14, 740, 860)]
    assert parse_hud(bar).level == 19


def test_real_bar_read():
    # What OCR read from the bar in the game (1920x1080 window); the character name is a placeholder.
    bar = [Row("LV. 09", 0.97, 452, 40, 44, 400, 504), Row("Rogue", 1.0, 541, 30, 30, 508, 574),
           Row("HP[642/6421", 0.93, 753, 29, 36, 698, 808), Row("MP[516/516]", 0.98, 906, 30, 37, 850, 962),
           Row("EXP.248[1.49%]", 0.93, 1074, 29, 28, 1010, 1138), Row("Name1", 1.0, 549, 50, 30, 508, 590)]
    r = parse_hud(bar)
    assert (r.hp, r.hp_max, r.mp, r.mp_max) == (642, 642, 516, 516)  # "]" read as "1" in 6421
    assert (r.exp, r.exp_percent) == (248, 1.49)
    assert (r.level, r.level_source) == (19, "exp")  # the boxed "19" reads as "09"


def test_hp_bracket_fix_only_applies_to_an_unclosed_bracket():
    assert parse_hud(rows("HP[642/6421]")).hp_max == 6421
    assert parse_hud(rows("HP 642/6421")).hp_max == 6421


def test_classic_level_unknown_when_nothing_fits():
    assert parse_hud(rows("LV. 09", "HP 812/900", "EXP 99999 [0.29%]")).level is None


@pytest.mark.parametrize("text, kind, amount, sure", [
    ("You received EXP (+21)", "exp", 21, True),
    ("You received EXP (t21)", "exp", 21, False),
    ("You recel ved EXP [+1,234]", "exp", 1234, True),
    ("You reeived EXP (+28)", "exp", 28, True),      # read like this in the game
    ("You recieved EXP (+28)", "exp", 28, True),
    ("You have gained mesos (+27)", "mesos", 27, True),
    ("You have gained rnesos (+27)", "mesos", 27, True),
    ("You have gained mes0s (443)", "mesos", 43, False),  # "+" read as "4"
    ("You have gained mes", None, None, None),            # cut off: no amount yet
])
def test_loot_amounts(text, kind, amount, sure):
    line = parse_loot(text)
    if kind is None:
        assert line is None
    else:
        assert (line.kind, line.amount, line.sure) == (kind, amount, sure)


@pytest.mark.parametrize("text, name, qty, sure", [
    ("Leaf x1 earned. (Etc)", "Leaf", 1, True),
    ("Blue Snail Shell xl eamed. (Etc)", "Blue Snail Shell", 1, False),
    ("Wax Doll x12 earmed (Etc)", "Wax Doll", 12, True),
    ("Red Potion ×3 earned.", "Red Potion", 3, True),
])
def test_loot_items(text, name, qty, sure):
    line = parse_loot(text)
    assert (line.kind, line.item, line.quantity, line.sure) == ("item", name, qty, sure)


def test_loot_item_name_matched_to_known_names():
    line = parse_loot("Bl ue Snai1 She11 x1 earned. (Etc)", ["Blue Snail Shell", "Red Snail Shell"])
    assert line.item == "Blue Snail Shell"


# The game cuts long item names off with "..."; with Classic World's item names they're completed.

@pytest.mark.parametrize("text", [
    "Bubbling's Hug...x1 earned. (Etc)",   # read like this in the game
    "Bubbling's Hug…x1 earned. (Etc)",
    "Bubbling's Hug.. x1 earned. (Etc)",
])
def test_cut_item_name_is_completed(text):
    line = parse_loot(text, ITEM_NAMES)
    assert (line.item, line.quantity, line.cut) == ("Bubbling's Huge Bubble", 1, False)


@pytest.mark.parametrize("text", [
    "Bubbiing's Hug...x1 earned. (Etc)",   # l read as i
    "Bubblng's Hug...x1 earned. (Etc)",    # a letter dropped
    "Bubb1ings Hug...xl earned. (Etc)",
])
def test_misread_cut_item_name_is_completed(text):
    assert parse_loot(text, ITEM_NAMES).item == "Bubbling's Huge Bubble"


def test_ambiguous_cut_item_name_stays_as_read():
    # Brown Hard Leather Boots, Pants or Top.
    line = parse_loot("Brown Hard Lea...x1 earned.", ITEM_NAMES)
    assert (line.item, line.cut) == ("Brown Hard Lea", True)


def test_item_names_are_the_games():
    # meowdb tells apart items that share a name with suffixes the game never shows ("Red Potion (Alt)").
    assert not [n for n in ITEM_NAMES if re.search(r"\((?:Alt(?: \d+)?|Warrior|Magician|Bowman|Thief)\)$", n)]
    assert {"Red Potion", "Dark Marble", "The Proof of a Hero", "Compensation Box"} <= set(ITEM_NAMES)
    assert all(n == n.strip() for n in ITEM_NAMES)


@pytest.mark.parametrize("text, name", [
    # Each was several suffixed names on meowdb, so the cut-off name looked ambiguous.
    ("The Proof of a H...x1 earned. (Etc)", "The Proof of a Hero"),
    ("Return Scroll - Nea...x1 earned. (Use)", "Return Scroll - Nearest Town"),
])
def test_cut_item_name_shared_by_several_items(text, name):
    assert parse_loot(text, ITEM_NAMES).item == name


def test_cut_item_name_is_never_the_name_it_shows():
    # A name the game cut off is longer than what it shows: not "Dark Brown Stealer" itself.
    assert parse_loot("Dark Brown Stealer...x1 earned.", ITEM_NAMES).item == "Dark Brown Stealer Pants"


def test_cut_item_name_without_names_or_a_match():
    line = parse_loot("Bubbling's Hug...x1 earned. (Etc)")
    assert (line.item, line.cut) == ("Bubbling's Hug", True)
    line = parse_loot("Qwerty Zxcvb...x2 earned.", ITEM_NAMES)
    assert (line.item, line.quantity, line.cut) == ("Qwerty Zxcvb", 2, True)
    assert parse_loot("Leaf x1 earned. (Etc)", ITEM_NAMES).cut is False


def test_match_prefix():
    names = ["Blue Snail Shell", "Red Snail Shell", "Snail Shell", "Orange Mushroom Cap"]
    assert match_prefix("Orange Mush", names) == "Orange Mushroom Cap"
    assert match_prefix("0range Mushr0", names) == "Orange Mushroom Cap"
    assert match_prefix("Snail She", names) == "Snail Shell"
    assert match_prefix("Snail Shell", names) is None  # nothing longer starts with it
    assert match_prefix("Red Snail Sh", names) == "Red Snail Shell"
    assert match_prefix("Blue Sn", names) == "Blue Snail Shell"
    assert match_prefix("Snail", names) is None        # too short to go on


def test_other_text_is_not_loot():
    assert parse_loot("Henesys") is None
    assert parse_loot("Quest completed (1/3)") is None


@pytest.mark.parametrize("text", ["Name1", "Name2", "米", "Party", "Lv. 19 Rogue"])
def test_party_panel_lines_are_not_loot(text):
    # The party panel sits inside the pickup box: member names with HP bars, the leader's icon read as "米".
    assert parse_loot(text) is None


def test_match_name():
    names = ["Henesys Hunting Ground I", "Henesys Hunting Ground II", "Ellinia", "Perion"]
    assert match_name("ellinia", names) == "Ellinia"
    assert match_name("Peri0n", names) == "Perion"
    assert match_name("Elinia", names) == "Ellinia"
    assert match_name("Henesys Hunting Ground", names) is None  # I and II score too close to tell
    assert match_name("Kerning City", names) is None
