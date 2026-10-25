"""Which attack a throw is (attack.choice: Lucky Seven, or the regular attack to finish a monster and while her MP is
under a cast), and the hunter's bookkeeping behind it (behavior.attacks: the monster a throw takes, the HP it has
left, her MP), on synthetic looks: no map image, no simulator."""

import dataclasses
import json
import time
from types import SimpleNamespace

import pytest

from gamedata.paths import TABLES
from maplehelper.keys import parse as K
from gamelab.attack import choice
from gamelab.attack.choice import REGULAR, can_cast, cast_mp, finishes, lucky_seven, pick
from gamelab.behavior.attacks import Attacks
from gamelab.control.actions import Actions
from gamelab.localization.monsters import Monster
from gamelab.perception.mobs import Mob

from test_sim import KEYS

L7 = lucky_seven(10)
MAX_MP = 601.0


def test_lucky_seven_is_three_regular_stars_and_kills_a_horny_mushroom_in_about_three_and_a_half_throws():
    assert (L7.stars, L7.lo, L7.hi, L7.mp) == (2, 45, 105, 11.0)
    assert L7.mean() == pytest.approx(3 * REGULAR.mean())
    assert REGULAR.throws(233) == pytest.approx(5.18, abs=0.01) and L7.throws(233) == pytest.approx(2.03, abs=0.01)
    assert REGULAR.throws(403, 10) == pytest.approx(9.40, abs=0.01) and L7.throws(403, 10) == pytest.approx(3.44, abs=0.01)
    assert REGULAR.throws(30) == 1.0 and REGULAR.one_throw(30) == 1.0 and REGULAR.one_throw(71) == 0.0
    assert REGULAR.throws(0) == 0.0


def test_the_regular_attack_finishes_what_it_takes_about_as_fast_as_lucky_seven():
    # A Green Mushroom (no defence) from 40 HP left: a regular star kills it 76% of the time, 1.24 throws against
    # Lucky Seven's 1; from 41 it's 1.27, more than FINISH_SLACK over.
    assert finishes(40, 0, REGULAR, L7) and not finishes(41, 0, REGULAR, L7)
    assert finishes(36, 10, REGULAR, L7) and not finishes(37, 10, REGULAR, L7)  # a Horny Mushroom: DEF 10
    assert not finishes(233, 0, REGULAR, L7)
    assert REGULAR.one_throw(40) == pytest.approx(31 / 41)


def test_a_cast_only_with_a_px_of_mp_more_than_it_costs():
    px = MAX_MP / choice.MP_STEPS  # 5.8 MP
    assert can_cast((11 + px) / MAX_MP, MAX_MP, 11) and not can_cast((11 + px - 0.1) / MAX_MP, MAX_MP, 11)
    assert not can_cast(2 / 104, MAX_MP, 11) and can_cast(3 / 104, MAX_MP, 11)  # 2 px of her bar (11.6 MP): not sure


def test_pick_lucky_seven_but_to_finish_with_mp_and_for_something_in_its_way():
    full = 1.0
    assert pick(233, 0, full, MAX_MP, L7) == (L7, "damage")
    assert pick(30, 0, full, MAX_MP, L7) == (REGULAR, "finish")
    assert pick(233, 0, 0.02, MAX_MP, L7) == (REGULAR, "MP")       # 12 MP read: less a px, under a cast
    assert pick(None, 0, full, MAX_MP, L7) == (REGULAR, "nothing in its way")
    assert pick(30, 0, full, MAX_MP, L7, mode="skill") == (L7, "skill")  # Lucky Seven whenever she has the MP ...
    assert pick(30, 0, 0.0, MAX_MP, L7, mode="skill") == (REGULAR, "MP")  # ... and only then
    assert pick(233, 0, full, MAX_MP, L7, mode="regular") == (REGULAR, "regular")
    assert pick(233, 0, full, MAX_MP, None) == (REGULAR, "regular")  # no Lucky Seven key
    assert pick(233, 0, None, MAX_MP, L7) == (L7, "damage")  # MP not read: as if she has it


def test_lucky_sevens_mp_a_cast_is_the_game_datas():
    path = TABLES / "skills.json"
    if not path.is_file():
        pytest.skip("no game data tables (python -m gamedata tables)")
    levels = json.loads(path.read_text(encoding="utf-8"))["4001003"]["levels"]
    assert choice.LUCKY_SEVEN_MP == tuple(levels[str(k)]["mpCon"] for k in range(1, len(levels) + 1))
    assert cast_mp(10) == 11 and cast_mp(0) == 8 and cast_mp(99) == 16


# --- the hunter's bookkeeping ----------------------------------------------------------------------------

KEYS7 = dataclasses.replace(KEYS, lucky_seven=K("g"))  # (test_sim's keys, with Lucky Seven on g)
MODEL = SimpleNamespace(mob_hp={"Green Mushroom": 233.0, "Horny Mushroom": 403.0},
                        mob_pdd={"Green Mushroom": 0.0, "Horny Mushroom": 10.0})


def _look(monsters, mp=1.0, t=None):
    """A look of her standing at x 0 on platform 1 (y 0), with `monsters` (kind, x, bar)."""
    p = SimpleNamespace(id=1, walls=(), floors=(), peers=(), left=-500.0, right=500.0, y_at=lambda x: 0.0)
    ms = [Monster(Mob(k, 0, 0, 0, 0, 1.0), x, 0.0, 1, bar) for k, x, bar in monsters]
    return SimpleNamespace(t=time.monotonic() if t is None else t, mp=mp, searched=True,
                           where=SimpleNamespace(x=0.0, y=0.0, platform=p), platform=p, monsters=ms)


def _attacks(mode="choice", **kw):
    said = []
    a = Attacks(MODEL, KEYS7, mode, max_mp=MAX_MP, level=10, log=said.append, **kw)
    return a, said


def test_she_casts_at_a_fresh_monster_and_finishes_it_with_the_regular_attack(monkeypatch):
    monkeypatch.setattr(choice, "MISS", 0.0)  # her throws counted at their whole average damage
    a, said = _attacks()
    s = _look([("green-mushroom", 250.0, None)])
    a.on_look(s)
    keys = []
    for _ in range(3):
        keys.append(a.key("right"))
        a.on_look(_look([("green-mushroom", 250.0, None)]))
    # 233 HP: a cast (150 on average) leaves 83, too much for a regular star; a second leaves none to reckon with (1)
    assert keys[:2] == [KEYS7.lucky_seven.name] * 2 and keys[2] == KEYS7.attack.name
    assert a.counts["lucky_seven"] == 2 and a.counts["finish"] == 1
    assert "Lucky Seven: green-mushroom ~233/233 HP (MP 595)" in said[0]
    assert "the regular attack: green-mushroom ~1/233 HP; finishing it" in said[2]


def test_a_bar_read_between_empty_and_full_sets_what_is_left(monkeypatch):
    a, _ = _attacks()
    a.on_look(_look([("horny-mushroom", 200.0, 0.08)]))  # never thrown at: its bar, 32 of its 403 HP
    assert a.key("right") == KEYS7.attack.name
    a2, _ = _attacks()
    a2.on_look(_look([("horny-mushroom", 200.0, 0.0)]))  # read empty (as on Tree Dungeon live): its whole HP
    assert a2.key("right") == KEYS7.lucky_seven.name
    assert a2.left(a2._last.monsters[0]) > 400 - 0.725 * 150 * 100 / 110 - 1


def test_the_regular_attack_with_mp_under_a_cast_and_with_nothing_in_the_way():
    a, said = _attacks()
    a.on_look(_look([("green-mushroom", 250.0, None)], mp=2 / 104))
    assert a.key("right") == KEYS7.attack.name and a.counts["MP"] == 1
    assert "MP 6 < a cast's 11" in said[-1]
    a.on_look(_look([("green-mushroom", 250.0, None)]))  # MP back
    assert a.key("left") == KEYS7.attack.name and a.counts["nothing in its way"] == 1  # nothing that way
    r, _ = _attacks("regular")
    r.on_look(_look([("green-mushroom", 250.0, None)]))
    assert r.key("right") == KEYS7.attack.name


def test_unseen_under_its_damage_numbers_the_monster_she_threw_at_is_still_the_target():
    a, _ = _attacks()
    a.on_look(_look([("horny-mushroom", 250.0, None)]))
    a.key("right")
    a.on_look(_look([]))  # hidden: no monster in the look
    assert a.target("right") is not None and a.target("left") is None


def test_actions_ask_the_choice_for_each_throws_key():
    act = Actions(KEYS7, skill=True)
    asked = []
    act.choose = lambda side: asked.append(side) or KEYS7.attack.name
    assert act._attack_key("left") == KEYS7.attack.name and asked == ["left"]
    act.choose = None
    assert act._attack_key("left") == KEYS7.lucky_seven.name
    assert Actions(KEYS7)._attack_key("left") == KEYS7.attack.name  # no skill: the regular attack, whatever
