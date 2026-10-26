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
from gamelab.behavior.attacks import LANDS_S, SHOWS_S, Attacks
from gamelab.behavior.ledger import in_path
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
    # 233 HP: a cast (150 on average) leaves 83, too much for a regular star; after a second it's dead 97% of the
    # time, and if it isn't, it has 11 left on average: a regular star's
    assert keys[:2] == [KEYS7.lucky_seven.name] * 2 and keys[2] == KEYS7.attack.name
    assert a.counts["lucky_seven"] == 2 and a.counts["finish"] == 1
    assert "Lucky Seven: green-mushroom ~233/233 HP (MP 595)" in said[0]
    assert "the regular attack: green-mushroom ~11/233 HP if alive (97% dead); finishing it" in said[2]


def test_seen_alive_after_her_throws_landed_what_would_have_killed_it_is_ruled_out():
    """With misses (choice.MISS), a Green Mushroom seen walking after two casts landed most likely took one of them,
    not both: about 100 HP left, a cast's work, where her throws' average damage alone left 16."""
    a, said = _attacks()
    t0 = time.monotonic()
    a.on_look(_look([("green-mushroom", 250.0, None)], t=t0))
    a.key("right")
    a.key("right")
    m = a._last.monsters[0]
    assert a.reckon(m).dead == pytest.approx(0.51, abs=0.01)  # not seen since: maybe dead ...
    assert a.left(m) == pytest.approx(104, abs=1)  # ... and if it isn't, a cast's work left
    a.on_look(_look([("green-mushroom", 250.0, None)], t=t0 + 1.0 + LANDS_S))  # alive after both
    left = a.reckon(a._last.monsters[0])
    assert left.dead == 0.0 and left.hp == pytest.approx(104, abs=1)
    assert a.key("right") == KEYS7.lucky_seven.name and a.counts["finish"] == 0


def test_a_bar_up_though_read_empty_says_not_all_her_throws_missed():
    """On Tree Dungeon at 1366x768 a hurt monster's bar reads empty, but it's up: her damage shows. Seen walking with it
    after two casts landed, the Green took one or both (about 80 HP left, not 104 with the chance both missed, 15%);
    with a third cast out that may have shown by then and not landed, the bar may be that one's, and says nothing."""
    a, _ = _attacks()
    t0 = time.monotonic()
    a.on_look(_look([("green-mushroom", 250.0, None)], t=t0))
    a.key("right")
    a.key("right")
    a.on_look(_look([("green-mushroom", 250.0, 0.0)], t=t0 + 1.0 + LANDS_S))  # alive after both, its bar up
    left = a.reckon(a._last.monsters[0])
    assert left.dead == 0.0 and left.p[-1] == 0.0 and left.hp == pytest.approx(80, abs=1)
    assert choice.Left.of(233).hit() is None  # (no throw reckoned in it: nothing to go by)
    b, _ = _attacks()
    b.on_look(_look([("green-mushroom", 250.0, None)], t=t0))
    b.key("right")
    b.key("right")
    b.marks[0].pending[-1] = (t0 + 1.0, b.marks[0].pending[-1][1])  # the second cast keyed 1 s later
    b.on_look(_look([("green-mushroom", 250.0, 0.0)], t=t0 + 1.05 + SHOWS_S))  # it may have shown: the first unsure
    assert b.reckon(b._last.monsters[0]).p[-1] > 0.0


def test_a_throw_takes_the_next_in_its_way_past_one_likely_dead():
    a, _ = _attacks()
    monkey = choice.MISS
    try:
        choice.MISS = 0.0
        a.on_look(_look([("green-mushroom", 250.0, None), ("horny-mushroom", 255.0, None)]))
        assert [m.kind for m in in_path(a._last, "right", a._last.monsters)] == ["green-mushroom", "horny-mushroom"]
        a.key("right")
        a.key("right")  # two sure casts at the Green: dead 97%
        assert a.target("right").kind == "horny-mushroom"
    finally:
        choice.MISS = monkey


def test_a_bar_read_between_empty_and_full_sets_what_is_left(monkeypatch):
    a, _ = _attacks()
    a.on_look(_look([("horny-mushroom", 200.0, 0.08)]))  # never thrown at: its bar, 32 of its 403 HP
    assert a.key("right") == KEYS7.attack.name
    a2, _ = _attacks()
    a2.on_look(_look([("horny-mushroom", 200.0, 0.0)]))  # read empty (as at 1366x768 until 2026-10-25): its whole HP
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


def test_a_casts_mp_is_off_at_its_throw_and_a_frame_shown_before_it_showed_reads_no_more():
    """A look that doesn't wait for her keys' effect (senses.KEYS_WAIT "moves" or "none") can be of a frame shown
    before a throw's key, right after it: its bar still reads the MP the cast took. The cast's MP is taken off at its
    throw (Attacks.on_throw), and such a bar is believed only where it's lower, as is one shown before MP_SHOWS_S after
    it (live a cast's MP first showed 0.21-0.53 s after its key); one shown then is her MP again (her pet's potion,
    say). (With KEYS_WAIT "moves", in loop_tree_mp_runs_out she cast with 5 MP, nothing thrown: the look after her
    last cast, shown before its key, read the 12 it took.)"""
    from gamelab.behavior.attacks import MP_SHOWS_S
    a, _ = _attacks()
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=20 / MAX_MP))
    assert a.key("right") == KEYS7.lucky_seven.name
    at = time.monotonic() + 0.08  # its attack key, 80 ms into the plan (a tap first)
    a.on_throw(at, "right")
    assert a.mp == pytest.approx(9 / MAX_MP)
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=20 / MAX_MP, t=at - 0.02))  # shown before its key
    assert a.mp == pytest.approx(9 / MAX_MP)
    assert a.key("right") == KEYS7.attack.name and a.counts["MP"] == 1
    a.on_throw(at + 1.0, "right")  # (the regular attack: no MP)
    assert a.mp == pytest.approx(9 / MAX_MP)
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=0.5, t=at + 0.3))  # too soon for the cast's MP to show
    assert a.mp == pytest.approx(9 / MAX_MP)
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=0.5, t=at + MP_SHOWS_S))
    assert a.mp == 0.5 and a.key("right") == KEYS7.lucky_seven.name


def test_a_zero_read_under_her_pets_mp_warning_is_her_mp_as_reckoned():
    """Live on 2026-10-25 her MP bar read 0 for up to 0.6 s after a cast took her under her pet's 5% warning, from 6-9
    px (of 104), while she had about 20-35 MP: 11 regular throws went out for 'MP 0'. Such a 0 (within WARNING_S of a
    read of WARNING_STEPS or less) is her last read less her casts since: from 7 px (40 MP at 601) and a cast (11), 29
    MP, a cast's; from 3 px (17), 6: the regular attack. A 0 after a read of more, or read longer, is believed."""
    from gamelab.behavior.attacks import MP_SHOWS_S, WARNING_S
    t0 = time.monotonic()
    for px, key in ((7, KEYS7.lucky_seven.name), (3, KEYS7.attack.name)):
        a, said = _attacks()
        a.on_look(_look([("horny-mushroom", 250.0, None)], mp=px / 104, t=t0))
        a.key("right")
        a.on_throw(t0 + 0.05, "right")
        a.on_look(_look([("horny-mushroom", 250.0, None)], mp=0.0, t=t0 + 0.05 + MP_SHOWS_S))  # the warning's 0
        assert a.mp == pytest.approx(px / 104 - 11 / MAX_MP) and a.key("right") == key, px
        assert a.counts["MP 0 under the warning"] == 1 and not any("MP 0 <" in line for line in said)
    a, _ = _attacks()
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=7 / 104, t=t0))
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=0.0, t=t0 + WARNING_S + 0.1))  # too long since: believed
    assert a.mp == 0.0 and a.key("right") == KEYS7.attack.name
    b, _ = _attacks()
    b.on_look(_look([("horny-mushroom", 250.0, None)], mp=20 / 104, t=t0))
    b.on_look(_look([("horny-mushroom", 250.0, None)], mp=0.0, t=t0 + 0.5))  # from 20 px: not the warning's
    assert b.mp == 0.0 and b.counts["MP 0 under the warning"] == 0


def test_a_ground_throws_cast_is_taken_off_at_its_own_key():
    """Actions.throw asks its attack key before it tells the throw listeners (as jump_throw does): the cast's MP goes
    off at its own key, not at the next throw's (with that one's key time)."""
    a, _ = _attacks()
    sent = []
    act = Actions(KEYS7, skill=True)
    act.choose = a.key
    act.throw_listeners.append(a.on_throw)
    act.play = lambda events, name, kind=None: sent.append((name, events))
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=1.0))
    act.facing = "right"
    act.throw("right")
    assert sent[0][1][0][1] == KEYS7.lucky_seven.name
    assert a.mp == pytest.approx(1.0 - 11 / MAX_MP) and a._cast is None and a._cast_at == act._attack_at


def test_a_monster_dying_where_a_kill_was_placed_is_no_throws_target_nor_marked():
    """Live on 2026-10-25 the dying monster stayed drawn 0.6-1.3 s after its kill was placed (its bar None, or as it
    was), and was taken from the path again whole: 18 casts and 4 regular throws at the dead in 11 minutes. One of a
    placed kill's kind on its platform within DYING_DX of it (ledger.dying) isn't the target: the next in the path is,
    or none (the regular attack: nothing in its way), and no mark is made on it; nor is it the target once it's gone
    (the hidden fallback), and the fallback skips a mark likely dead (DEAD_LIKELY) as the path does."""
    from gamelab.behavior.ledger import Kill
    kills = []
    a, said = _attacks(kills=lambda: kills)
    now = time.monotonic()
    a.on_look(_look([("green-mushroom", 250.0, None)], t=now - 0.4))
    a.key("right")  # a cast at it: a mark
    kills.append(Kill(now - 0.1, "green-mushroom", 1, 245.0, 0.0, 1))  # its kill placed 5 units off
    a.on_look(_look([("green-mushroom", 262.0, None), ("horny-mushroom", 330.0, None)], t=now))  # still drawn
    assert a.marks == []  # (the kill dropped its mark)
    assert a.target("right").kind == "horny-mushroom"
    a.on_look(_look([("green-mushroom", 262.0, None)], t=now))  # alone in the path, dying
    assert a.key("right") == KEYS7.attack.name and a.counts["nothing in its way"] == 1 and a.marks == []
    a.aimed = a._last.monsters[0]  # a jump's own target: dying too
    assert a.key("right") == KEYS7.attack.name and a.marks == []
    b, _ = _attacks()
    b.on_look(_look([("green-mushroom", 250.0, None)]))
    for _ in range(3):  # three casts: likely dead
        b.key("right")
    b.on_look(_look([]))  # unseen
    assert b.target("right") is None and b.key("right") == KEYS7.attack.name


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
