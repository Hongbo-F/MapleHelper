import pytest

from maplehelper.ocr.engine import Row
from maplehelper.ocr.parse import level_from_exp, match_name, parse_hud, parse_loot


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


@pytest.mark.parametrize("text, kind, amount, sure", [
    ("You received EXP (+21)", "exp", 21, True),
    ("You received EXP (t21)", "exp", 21, False),
    ("You recel ved EXP [+1,234]", "exp", 1234, True),
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


def test_other_text_is_not_loot():
    assert parse_loot("Henesys") is None
    assert parse_loot("Quest completed (1/3)") is None


def test_match_name():
    names = ["Henesys Hunting Ground I", "Henesys Hunting Ground II", "Ellinia", "Perion"]
    assert match_name("ellinia", names) == "Ellinia"
    assert match_name("Peri0n", names) == "Perion"
    assert match_name("Elinia", names) == "Ellinia"
    assert match_name("Henesys Hunting Ground", names) is None  # I and II score too close to tell
    assert match_name("Kerning City", names) is None
