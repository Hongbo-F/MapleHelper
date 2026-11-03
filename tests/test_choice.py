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

L7 = lucky_seven(10)  # (11 MP a cast: the MP tests' numbers, at her 601 of level 24)
MAX_MP = 601.0


def test_lucky_seven_is_two_stars_of_its_measured_damage_at_her_level_and_kills_a_green_mushroom_nine_times_in_ten():
    """Her stars as read off the monsters' bars live at level 25 (attack.choice.STAR, SKILL_STAR): Lucky Seven two of
    96-192 at its level 16 (14 MP), 3.6 regular stars of 20-140, scaled by its damage in the game data at another
    level (96% at 10 against 120%): one cast kills a Green Mushroom 91% of the time, a Horny Mushroom takes two."""
    l16 = lucky_seven(16)
    assert (l16.stars, l16.lo, l16.hi, l16.mp) == (2, 96, 192, 14.0) and (L7.lo, L7.hi, L7.mp) == (77, 154, 11.0)
    assert l16.mean() == pytest.approx(3.6 * REGULAR.mean()) and l16.one_throw(233) == pytest.approx(0.91, abs=0.005)
    assert REGULAR.throws(233) == pytest.approx(3.50, abs=0.01) and l16.throws(233) == pytest.approx(1.09, abs=0.01)
    assert REGULAR.throws(403, 10) == pytest.approx(6.14, abs=0.01) and l16.throws(403, 10) == pytest.approx(2.01, abs=0.01)
    assert REGULAR.throws(20) == 1.0 and REGULAR.one_throw(20) == 1.0 and REGULAR.one_throw(141) == 0.0
    assert REGULAR.throws(0) == 0.0


def test_lucky_sevens_damage_by_level_is_the_game_datas():
    path = TABLES / "skills.json"
    if not path.is_file():
        pytest.skip("no game data tables (python -m gamedata tables)")
    levels = json.loads(path.read_text(encoding="utf-8"))["4001003"]["levels"]
    assert choice.LUCKY_SEVEN_DAMAGE == tuple(levels[str(k)]["damage"] for k in range(1, len(levels) + 1))


def test_the_regular_attack_finishes_what_it_takes_about_as_fast_as_lucky_seven():
    # A Green Mushroom (no defence) from 49 HP left: a regular star kills it 76% of the time, 1.24 throws against
    # Lucky Seven's 1; from 50 it's more than FINISH_SLACK over (40 with the regular star of 30-70 before 2026-10-26).
    assert finishes(49, 0, REGULAR, L7) and not finishes(50, 0, REGULAR, L7)
    assert finishes(44, 10, REGULAR, L7) and not finishes(45, 10, REGULAR, L7)  # a Horny Mushroom: DEF 10
    assert not finishes(233, 0, REGULAR, L7) and not finishes(80, 0)  # (her Lucky Seven at SKILL_LEVEL by default)
    assert REGULAR.one_throw(49) == pytest.approx(92 / 121)


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
    # 233 HP: a cast (231 on average at level 10) kills it 48% of the time, and if it doesn't, it has 27 left on
    # average: a regular star's
    assert keys[0] == KEYS7.lucky_seven.name and keys[1:] == [KEYS7.attack.name] * 2
    assert a.counts["lucky_seven"] == 1 and a.counts["finish"] == 2
    assert "Lucky Seven: green-mushroom ~233/233 HP (MP 595)" in said[0]
    assert "the regular attack: green-mushroom ~27/233 HP if alive (48% dead); finishing it" in said[1]


def test_seen_alive_after_her_throws_landed_what_would_have_killed_it_is_ruled_out():
    """With misses (choice.MISS), a Green Mushroom seen walking after two casts landed most likely took one of them,
    not both: about 82 HP left, a cast's work."""
    a, said = _attacks()
    t0 = time.monotonic()
    a.on_look(_look([("green-mushroom", 250.0, None)], t=t0))
    a.key("right")
    a.key("right")
    m = a._last.monsters[0]
    assert a.reckon(m).dead == pytest.approx(0.72, abs=0.01)  # not seen since: likely dead ...
    assert a.left(m) == pytest.approx(82, abs=1)  # ... and if it isn't, a cast's work left
    a.on_look(_look([("green-mushroom", 250.0, None)], t=t0 + 1.0 + LANDS_S))  # alive after both
    left = a.reckon(a._last.monsters[0])
    assert left.dead == 0.0 and left.hp == pytest.approx(82, abs=1)
    assert a.key("right") == KEYS7.lucky_seven.name and a.counts["finish"] == 0


def test_a_bar_up_though_read_empty_says_not_all_her_throws_missed():
    """On Tree Dungeon at 1366x768 a hurt monster's bar reads empty, but it's up: her damage shows. Seen walking with it
    after two casts landed, the Green took one or both (about 27 HP left, not 82 with the chance both missed, 8%);
    with a third cast out that may have shown by then and not landed, the bar may be that one's, and says nothing."""
    a, _ = _attacks()
    t0 = time.monotonic()
    a.on_look(_look([("green-mushroom", 250.0, None)], t=t0))
    a.key("right")
    a.key("right")
    a.on_look(_look([("green-mushroom", 250.0, 0.0)], t=t0 + 1.0 + LANDS_S))  # alive after both, its bar up
    left = a.reckon(a._last.monsters[0])
    assert left.dead == 0.0 and left.p[-1] == 0.0 and left.hp == pytest.approx(27, abs=1)
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
    assert a2.left(a2._last.monsters[0]) > 400 - 0.725 * L7.mean() * 100 / 110 - 1


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


def test_a_casts_mp_is_off_at_its_throw_and_a_frame_shown_before_it_could_show_reads_it_less():
    """A look that doesn't wait for her keys' effect (senses.KEYS_WAIT "moves" or "none") can be of a frame shown
    before a throw's key, right after it: its bar still reads the MP the cast took. The cast's MP is taken off at its
    throw (Attacks.on_throw), and a bar read off a frame shown under MP_FIRST_S after it is less the cast; under
    MP_SHOWS_S, less it as far as her reckoning goes (live a cast's MP first showed 0.21-0.53 s after its key): her
    pet's potion shows at once, less the cast, and in full once the cast surely shows. (With KEYS_WAIT "moves", in
    loop_tree_mp_runs_out she cast with 5 MP, nothing thrown: the look after her last cast, shown before its key, read
    the 12 it took.)"""
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
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=0.5, t=at + 0.1))  # the pet's potion: the cast can't show
    assert a.mp == pytest.approx(0.5 - 11 / MAX_MP)
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=0.5, t=at + 0.3))  # it may show: her reckoning stands
    assert a.mp == pytest.approx(0.5 - 11 / MAX_MP)
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=0.5, t=at + MP_SHOWS_S))
    assert a.mp == 0.5 and a.key("right") == KEYS7.lucky_seven.name


def test_her_mp_follows_her_pets_drink_and_presses_that_took_none_while_she_casts_on():
    """With a look 0.2-0.34 s after each key (live), every read comes inside MP_SHOWS_S of a cast. Believed then only
    where it was lower (review of 2026-10-26), her MP as reckoned only went down: under the warning a drink read 0.25 s
    after her next cast's key stayed 1.3 px, and she threw the regular attack for 'MP' with 126 MP; over 40 casts with
    1 press in 10 taking none she reckoned 35 MP against 103. Now the drink is believed less that cast, and the
    reckoning keeps within a cast and a px of her bar."""
    mon = [("horny-mushroom", 250.0, None)]
    said = []
    a = Attacks(MODEL, KEYS7, "choice", max_mp=623.0, level=16, log=said.append)
    t = time.monotonic()
    a.on_look(_look(mon, mp=6 / 104, t=t))              # 6 px: 36 MP
    assert a.key("right") == KEYS7.lucky_seven.name
    a.on_throw(t + 0.05, "right")                        # under the pet's warning
    a.on_look(_look(mon, mp=0.0, t=t + 0.35))            # its 0: not believed
    assert a.key("right") == KEYS7.lucky_seven.name     # ~22 MP: a cast's
    a.on_throw(t + 0.90, "right")
    a.on_look(_look(mon, mp=21 / 104, t=t + 1.15))       # the drink shows: 21 px (126 MP)
    assert a.mp == pytest.approx(21 / 104 - 14 / 623)
    assert a.key("right") == KEYS7.lucky_seven.name and a.counts["MP"] == 0
    b = Attacks(MODEL, KEYS7, "choice", max_mp=623.0, level=16, log=lambda line: None)
    true, t = 600.0, time.monotonic()
    b.on_look(_look(mon, mp=round(true / 623 * 104) / 104, t=t))
    for i in range(40):
        assert b.key("right") == KEYS7.lucky_seven.name
        b.on_throw(t + 0.05, "right")
        true -= 0 if i % 10 == 9 else 14  # one press in ten takes none
        true += 0.17                        # her regen
        b.on_look(_look(mon, mp=int(true / 623 * 104) / 104, t=t + 0.35))  # the bar as drawn (whole px)
        assert -1e-9 <= true - b.mp * 623 <= 14 + 623 / 104 + 1e-9, i
        t += 0.85


def test_her_mp_off_the_newest_frame_before_a_key_shows_her_pets_drink_since_the_look():
    """Attacks.glance (Hunter._still, before a throw's key): her pet's drink shows about 0.3 s after the cast that
    brought it, and the look a throw is decided on 0.2-0.34 s after the key before; a drink shown between the look and
    the key is read off the feed's newest frame: Lucky Seven, not the regular attack for 'MP' (the harness: 1.25 a run
    so, her MP back)."""
    mon = [("horny-mushroom", 250.0, None)]
    a = Attacks(MODEL, KEYS7, "choice", max_mp=623.0, level=16, log=lambda line: None)
    t = time.monotonic()
    a.on_look(_look(mon, mp=4 / 104, t=t))  # 24 MP
    a.key("right")
    a.on_throw(t + 0.05, "right")  # 10 MP left: under the pet's warning, it drinks
    a.on_look(_look(mon, mp=4 / 104, t=t + 0.2))  # the look before the next key: the cast not shown yet
    assert a.mp == pytest.approx(4 / 104 - 14 / 623)
    a.glance(t + 0.5, 21 / 104)  # the drink shown, with the cast (126 MP: 112 as reckoned, the cast maybe not shown)
    assert a.mp == pytest.approx(21 / 104 - 14 / 623)
    assert a.key("right") == KEYS7.lucky_seven.name and a.counts["MP"] == 0
    a.glance(t + 0.7, None)  # (not read: nothing changes)
    assert a.mp == pytest.approx(21 / 104 - 14 / 623)


def test_a_bar_read_lower_than_her_reckoning_is_believed_at_once():
    """A cast her reckoning missed (or MP taken otherwise) shows as a bar lower than she reckons: believed, less the
    casts it can't show yet, whatever the casts it may."""
    a, _ = _attacks()
    t = time.monotonic()
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=0.8, t=t))
    a.key("right")
    a.on_throw(t + 0.05, "right")
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=0.5, t=t + 0.35))  # (the cast may show)
    assert a.mp == pytest.approx(0.5)
    a.key("right")
    a.on_throw(t + 0.40, "right")
    a.on_look(_look([("horny-mushroom", 250.0, None)], mp=0.4, t=t + 0.45))  # (that one can't)
    assert a.mp == pytest.approx(0.4 - 11 / MAX_MP)


def test_a_zero_read_under_her_pets_mp_warning_is_her_mp_as_reckoned():
    """Live on 2026-10-25 her MP bar read 0 for up to 0.6 s after a cast took her under her pet's 5% warning, from 6-9
    px (of 104), while she had about 20-35 MP: 11 regular throws went out for 'MP 0'. Such a 0 (within WARNING_S of a
    read above 0, it or her MP as reckoned since WARNING_STEPS or less) is her last read less her casts since: from 7
    px (40 MP at 601) and a cast (11), 29 MP, a cast's; from 3 px (17), 6: the regular attack; from 10 px and two
    casts, 36 (replayed live, two such 0s were believed by the read alone). Counted once a run of 0s. A 0 after a read
    of more, or read longer, is believed."""
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
    c, said = _attacks()  # from 10 px (58 MP) and two casts, 36 (6.2 px) as reckoned: the warning's too
    c.on_look(_look([("horny-mushroom", 250.0, None)], mp=10 / 104, t=t0))
    for k in range(2):
        c.key("right")
        c.on_throw(t0 + 0.05 + 0.85 * k, "right")
    for k in range(3):  # three looks of 0 in a row: one run of them, counted once
        c.on_look(_look([("horny-mushroom", 250.0, None)], mp=0.0, t=t0 + 1.2 + 0.1 * k))
    assert c.mp == pytest.approx(10 / 104 - 22 / MAX_MP) and c.key("right") == KEYS7.lucky_seven.name
    assert c.counts["MP 0 under the warning"] == 1
    assert "her MP bar's 0 under her pet's warning not believed 1 times" in c.summary()
    assert "MP 0 under" not in c.summary().split(";")[0]  # (apart from the regular attack's reasons)
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
    assert a.mp == pytest.approx(1.0 - 11 / MAX_MP) and a._cast is None and a._casts[-1][0] == act._attack_at


def test_a_monster_dying_where_a_kill_was_placed_is_no_throws_target_nor_marked():
    """Live on 2026-10-25 the dying monster stayed drawn 0.6-1.3 s after its kill was placed (its bar None, or as it
    was), and was taken from the path again whole: 18 casts and 4 regular throws at the dead in 11 minutes. One of a
    placed kill's kind on its platform within DYING_DX of it (ledger.dying) isn't the target: the next in the path is,
    or none (the regular attack: nothing in its way), and no mark is made on it; nor is it the target once it's gone
    (the hidden fallback), and the fallback takes a mark not likely dead (DEAD_LIKELY) first, the likeliest if all are,
    as the path does."""
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
    b, said = _attacks()
    b.on_look(_look([("horny-mushroom", 240.0, None)]))
    b.key("right")  # a cast at the Horny
    b.on_look(_look([("green-mushroom", 250.0, None)]))  # the Horny unseen: hidden
    for _ in range(3):  # three casts at the Green: likely dead
        b.key("right")
    b.on_look(_look([]))  # both unseen
    assert b.target("right").kind == "horny-mushroom"  # not likely dead: first
    b.marks = [mk for mk in b.marks if mk.m.kind == "green-mushroom"]
    assert b.target("right").kind == "green-mushroom"  # all likely dead: the likeliest ...
    b.key("right")
    assert "if alive" in said[-1]  # ... its attack by what it has if alive


def test_a_throw_takes_a_last_look_before_its_key_and_a_kill_come_in_of_a_kind_in_its_way_drops_it():
    """Hunter._still (Actions.still): once her lock is waited out, before any key of a throw, the feed's newest state
    after her keys' effect, and its events not drained yet (the next look's): an EXP gain named (by a copy of her
    meter) for a kind in the throw's way drops it: no key sent, a 'dropped' line; one of a kind not in its way, none,
    or nothing in its way lets it go. Live on 2026-10-25 17 throws went out after a gain that showed before their key
    (the look they were decided on a median 0.57 s old at it)."""
    from gamelab.behavior.hunter import Hunter
    from gamelab.localization import mapdata
    from gamelab.localization.gains import ExpMeter
    from gamelab.localization.tracking import Event
    from gamelab.perception.exp_number import ExpReading
    from test_fight import EMPTY_MAP, _Eyes

    class Feed:
        def __init__(self, events):
            self.events = events

        def newest(self, after, timeout=0.0):
            return SimpleNamespace(t=time.monotonic())

        def pending_events(self, upto=None):
            return list(self.events)

    def hunter(gain, kind_in_way):
        said = []
        h = Hunter(_Eyes([]), KEYS7, mapdata.parse(1, EMPTY_MAP, "Test", {}), log=said.append, attack="choice")
        h.meter = ExpMeter({"Green Mushroom": 26, "Horny Mushroom": 41})
        t = time.monotonic()
        h.meter.update(t - 5.0, ExpReading(1000, None, (25,)), False)  # her EXP as read before
        h.senses._feed = Feed([] if gain is None else
                              [Event("exp_gain", t - 0.1, None, {"exp": 1000 + gain, "levels": [25]})])
        h.attacks.on_look(_look([] if kind_in_way is None else [(kind_in_way, 250.0, None)]))
        sent = []
        h.actions.play = lambda events, name, kind=None: sent.append(name)
        h.actions.facing = "right"
        return h, said, sent

    for gain, kind, goes in ((41, "horny-mushroom", False), (41, "green-mushroom", True), (None, "horny-mushroom", True),
                             (41, None, True), (26, "green-mushroom", False)):
        h, said, sent = hunter(gain, kind)
        assert h.actions.throw("right") is goes and (sent == ["throw"]) is goes, (gain, kind)
        assert any("a kill came in before the key" in line for line in said) is not goes
    h, said, sent = hunter(41, "horny-mushroom")
    h.attacks.aimed = None
    assert h.actions.jump_throw("right") is False and sent == []


def test_unseen_under_its_damage_numbers_the_monster_she_threw_at_is_still_the_target():
    a, _ = _attacks()
    a.on_look(_look([("horny-mushroom", 250.0, None)]))
    a.key("right")
    a.on_look(_look([]))  # hidden: no monster in the look
    assert a.target("right") is not None and a.target("left") is None


def test_a_monster_about_that_the_look_missed_takes_a_throw_its_way_as_the_looks_own():
    """Attacks.unseen (the hunt loop's Hunt._unseen_here, hunt.UNSEEN_ATTACK): a monster under her body the look didn't
    find (none within 25 units in 1317 live looks) is in the throw's way, so it's Lucky Seven at it; with nothing about
    either, the regular attack ("nothing seen in its way", no MP thrown at nothing). The dead ones' tracks aren't among
    them (localization.unseen.hidden: tests/test_hunt.py)."""
    a, said = _attacks()
    a.on_look(_look([]))
    assert a.key("right") == KEYS7.attack.name and a.counts["nothing in its way"] == 1
    under = _look([("horny-mushroom", 15.0, None)]).monsters[0]
    a.unseen = lambda s: [under]
    assert a.target("right") is under and a.target("left") is None
    assert a.key("right") == KEYS7.lucky_seven.name and "Lucky Seven: horny-mushroom" in said[-1]


def test_actions_ask_the_choice_for_each_throws_key():
    act = Actions(KEYS7, skill=True)
    asked = []
    act.choose = lambda side: asked.append(side) or KEYS7.attack.name
    assert act._attack_key("left") == KEYS7.attack.name and asked == ["left"]
    act.choose = None
    assert act._attack_key("left") == KEYS7.lucky_seven.name
    assert Actions(KEYS7)._attack_key("left") == KEYS7.attack.name  # no skill: the regular attack, whatever


# --- what she knows of a kill as soon as a frame can show it (2026-10-26) ----------------------------------------------

def test_a_throw_whose_kill_would_have_shown_by_a_frame_with_no_gain_since_didnt_kill():
    """Attacks.heard (Hunter._vitals at every look, Hunter._still at a throw's key): a frame shown KILL_SHOWN_S after a
    throw's attack key, with no EXP gain since that can be its target's kill, rules out what the throw would have killed
    it with, seen or not; its damage stays taken off. Live on 2026-10-26 the look a throw was decided on came 0.2-0.34 s
    after the key before, a kill ruled out only by a look LANDS_S after its throw that saw its target: 232 of 652 throws
    went out reckoned 50% dead or more, 224 of them at monsters alive at the key, and at one unseen under its damage
    numbers the reckoning only grew with her throws (66% -> 89% -> 96%)."""
    from gamelab.behavior.attacks import KILL_SHOWN_S
    from gamelab.behavior.ledger import Kill
    t0 = time.monotonic()
    # The frame KILL_SHOWN_S after the second cast's key, from that key as on_throw has it. Written t0 + 0.9 +
    # KILL_SHOWN_S, it was a float an ulp under (t0 + 0.05 + 0.85) + KILL_SHOWN_S wherever t0 (time.monotonic: the
    # machine's uptime) lies between 2**16 and 2**17 s, 18-36 hours up, and Attacks._unkilled's `heard >= key +
    # KILL_SHOWN_S` took it for a frame before (failing so on master 40ba190 and every branch of 2026-11-02)
    shown = (t0 + 0.05 + 0.85 * 1) + KILL_SHOWN_S

    def two_casts(kills=None):
        a, said = _attacks(kills=(lambda: kills) if kills is not None else None)
        a.on_look(_look([("green-mushroom", 250.0, None)], t=t0))
        for k in range(2):
            assert a.key("right") == KEYS7.lucky_seven.name
            a.on_throw(t0 + 0.05 + 0.85 * k, "right")
        a.on_look(_look([], t=t0 + 1.0))  # unseen since: under its damage numbers
        return a, a.marks[0].m

    a, m = two_casts()
    a.heard(t0 + 1.0, [])  # the second cast's kill can't have shown yet: the first's surely has, and didn't
    left = a.reckon(m)
    assert 0.0 < left.dead < 0.6 and a.left(m) == pytest.approx(82, abs=1)  # (unheard 72%: the first's ruled out)
    a.heard(shown, [])  # nor the second's
    assert a.reckon(m).dead == 0.0 and a.left(m) == pytest.approx(82, abs=1)  # (what it has if alive: as it was)
    b, m = two_casts()
    assert b.reckon(m).dead == pytest.approx(0.72, abs=0.01)  # (unheard: as before)
    b.heard(shown, [(t0 + 1.2, frozenset({"horny-mushroom"}))])  # a gain named another kind
    assert b.reckon(m).dead == 0.0
    for kinds in (frozenset({"green-mushroom"}), None):  # one that can be its kill (or any: a share, unread) ...
        c, m = two_casts()
        c.heard(shown, [(t0 + 1.2, kinds)])
        # ... the second cast's (0.3 s after its key), not the first's (1.15 s): as heard at t0 + 1.0, alive after it
        assert c.reckon(m).dead == pytest.approx(left.dead) and c.left(m) == pytest.approx(82, abs=1), kinds
    kills = []
    d, m = two_casts(kills)
    d.heard(shown, [(t0 + 1.2, frozenset({"green-mushroom"}))])
    kills.append(Kill(t0 + 1.35, "green-mushroom", 1, -200.0, 0.0, 1, "green-mushroom", t0 + 1.2))  # placed elsewhere
    d.on_look(_look([], t=t0 + 1.35))
    assert d.reckon(m).dead == 0.0  # (its gain was that one's)


def test_a_gain_can_be_a_throws_kill_only_as_soon_after_its_key_as_a_kill_shows():
    """Attacks._unkilled (review, 2026-10-27): a gain can be a throw's kill only GAIN_AFTER_KEY (0.1-0.6 s) after its
    key, when its kill's would show: one of no kind known 0.7 s after a cast, or 0.05 s after (the throw before's),
    rules that cast's kill out all the same; one 0.3 s after doesn't. Any gain since counted at first, and replayed
    through the grind of 2026-10-26 21:34, a gain of no kind known 0.66 s after a cast kept the one before it 72% dead:
    the next five throws at that Horny Mushroom, alive, went out the regular attack (live it took 4 casts)."""
    from gamelab.localization.monsters import GAIN_AFTER_KEY
    t0 = time.monotonic()

    def one_cast(after):
        a, _ = _attacks()
        a.on_look(_look([("green-mushroom", 250.0, None)], t=t0))
        assert a.key("right") == KEYS7.lucky_seven.name
        a.on_throw(t0 + 0.05, "right")
        a.on_look(_look([], t=t0 + 0.2))  # unseen since: under its damage numbers
        a.heard(t0 + 1.05, [] if after is None else [(t0 + 0.05 + after, None)])  # (None: a share, or not read yet)
        return a.reckon(a.marks[0].m).dead

    assert one_cast(None) == 0.0
    assert one_cast(0.3) == pytest.approx(0.35, abs=0.02)  # (a cast at level 10 kills a Green Mushroom so, its misses in)
    assert one_cast(GAIN_AFTER_KEY[1] + 0.1) == 0.0 and one_cast(GAIN_AFTER_KEY[0] - 0.05) == 0.0


def test_a_throws_pending_time_is_its_attack_keys_and_the_next_look_waits_for_its_kill_to_show():
    """Attacks.on_throw: the throw `key` chose for a target is pending at its attack key's time (a jump throw's 40-460 ms
    after the choice), and `shown_by` is KILL_SHOWN_S after it (Senses.shows: the look the next throw is decided on is
    of that frame); a throw at nothing changes neither."""
    from gamelab.behavior.attacks import KILL_SHOWN_S
    a, _ = _attacks()
    t = time.monotonic()
    a.on_look(_look([("horny-mushroom", 250.0, None)], t=t))
    assert a.shown_by() < t
    a.key("right")
    a.on_throw(t + 0.3, "right")  # (a jump throw's attack key, 0.3 s after its key was chosen)
    mk = a.marks[0]
    assert mk.pending[-1][0] == mk.last == t + 0.3 and a.shown_by() == t + 0.3 + KILL_SHOWN_S
    a.key("left")  # nothing that way
    a.on_throw(t + 1.2, "left")
    assert a.shown_by() == t + 0.3 + KILL_SHOWN_S and len(mk.pending) == 1


def test_the_attack_is_chosen_with_the_chance_its_target_is_dead():
    """Attacks.key, DEAD_CHOICE: choice.finishes on the whole reckoning (what Lucky Seven saves counts only if it's
    alive): at a Green Mushroom 85% dead after three casts, unseen since, the regular attack; by what it has if alive
    alone (as until 2026-10-26), Lucky Seven. Heard alive, Lucky Seven either way."""
    import copy

    from gamelab.behavior import attacks as attacks_mod
    from gamelab.behavior.attacks import KILL_SHOWN_S
    a, said = _attacks()
    t0 = time.monotonic()
    a.on_look(_look([("green-mushroom", 250.0, None)], t=t0))
    for k in range(3):
        a.aimed = a.marks[0].m if a.marks else a._last.monsters[0]
        a.key("right")
        a.on_throw(t0 + 0.05 + 0.85 * k, "right")
    a.on_look(_look([], t=t0 + 2.0))
    m = a.marks[0].m
    assert a.reckon(m).dead > 0.5
    b = copy.deepcopy(a)
    assert a.key("right") == KEYS7.attack.name and "finishing it" in said[-1] and "dead)" in said[-1]
    try:
        attacks_mod.DEAD_CHOICE = False
        assert b.key("right") == KEYS7.lucky_seven.name
    finally:
        attacks_mod.DEAD_CHOICE = True
    c = copy.deepcopy(b)
    c.heard(t0 + 2.0 + KILL_SHOWN_S, [])
    assert c.reckon(m).dead == 0.0 and c.key("right") == KEYS7.lucky_seven.name


def test_the_kinds_a_change_of_her_exp_can_be():
    """hunter.kinds_of: a change's gains' kinds (Attacks.heard: what can be a throw's kill), or any (None) where one
    named none, it couldn't be read yet (no gain), or no kill of the map's explains it (a share of one another player
    hurt too: maybe her target's)."""
    from gamelab.behavior.hunter import kinds_of
    from gamelab.localization.gains import Gain
    assert kinds_of([Gain(1.0, 26, ("green-mushroom",))]) == frozenset({"green-mushroom"})
    assert kinds_of([Gain(1.0, 67, ("green-mushroom", "horny-mushroom"))]) == {"green-mushroom", "horny-mushroom"}
    assert kinds_of([]) is None and kinds_of([Gain(1.0, 8, ())]) is None and kinds_of([Gain(1.0, None, (None,))]) is None


def test_the_hunter_tells_the_attack_choice_her_exp_at_every_look_and_at_a_throws_key():
    """Hunter: its looks' EXP changes (Hunter._vitals) and the feed's gains not drained yet at a throw's key
    (Hunter._still) go to Attacks.heard with the kinds each can be; with the attack choice, the look a throw is decided
    on waits for the last throw's kill to show (Senses.shows: Attacks.shown_by)."""
    from gamelab.behavior.hunter import Hunter
    from gamelab.localization import mapdata
    from gamelab.localization.gains import ExpMeter
    from gamelab.localization.tracking import Event
    from gamelab.perception.exp_number import ExpReading
    from test_fight import EMPTY_MAP, _Eyes

    t = time.monotonic()

    class Feed:
        events = [Event("exp_gain", t - 0.1, None, {"exp": 1041, "levels": [25]})]

        def newest(self, after, timeout=0.0):
            return SimpleNamespace(t=t)

        def pending_events(self, upto=None):
            return list(self.events)

    h = Hunter(_Eyes([]), KEYS7, mapdata.parse(1, EMPTY_MAP, "Test", {}), log=lambda line: None, attack="choice")
    assert h.senses.shows == h.attacks.shown_by
    assert Hunter(_Eyes([]), KEYS7, mapdata.parse(1, EMPTY_MAP, "Test", {}), log=lambda line: None,
                  attack="regular").senses.shows() < t  # (no choice: no marks, no wait)
    h.meter = ExpMeter({"Green Mushroom": 26, "Horny Mushroom": 41})
    h.meter.update(t - 5.0, ExpReading(1000, None, (25,)), False)
    h.senses._feed = Feed()
    h.attacks.on_look(_look([("green-mushroom", 250.0, None)]))  # a Green in the way: a Horny's kill doesn't drop it
    h.actions.play = lambda events, name, kind=None: None
    h.actions.facing = "right"
    assert h.actions.throw("right") is True
    assert h.attacks._heard == t and h.attacks._gains == {t - 0.1: frozenset({"horny-mushroom"})}
    s = _look([], t=t + 0.5)
    s.events = Feed.events + [Event("exp_gain", t + 0.4, None, {"exp": 1067, "levels": [25]})]
    s.hp, s.exp, s.exp_box = 1.0, None, None
    h._vitals(s)
    assert h.attacks._heard == t + 0.5 and h.attacks._gains[t + 0.4] == frozenset({"green-mushroom"})


def test_with_the_feed_the_dead_are_known_by_their_tracks_and_a_live_neighbour_keeps_its_mark():
    """With the feed's ids (Monster.track): a kill placed on a track (ledger.Kill.track) closes that one's mark, not the
    mark of the one beside it however near (its damage kept: not "~404 HP" again); a track the ledger takes for dead
    (Ledger.dead: another player's kill seen, say) is no target, and its mark is closed at once, sightings or not."""
    from gamelab.behavior.ledger import Kill

    def look(monsters, t=None):
        s = _look([(k, x, bar) for k, x, bar, _ in monsters], t=t)
        s.monsters = [dataclasses.replace(m, track=tid) for m, (_, _, _, tid) in zip(s.monsters, monsters)]
        return s

    kills, dead = [], set()
    a, _ = _attacks(kills=lambda: kills, dead=lambda: dead)
    now = time.monotonic()
    a.on_look(look([("green-mushroom", 250.0, None, 1), ("green-mushroom", 262.0, None, 2)], t=now - 0.9))
    a.key("right")                                    # a mark on 1 ...
    a.aimed = a._last.monsters[1]
    a.key("right")                                    # ... and one on 2
    assert sorted(mk.m.track for mk in a.marks) == [1, 2]
    kills.append(Kill(now - 0.1, "green-mushroom", 1, 262.0, 0.0, 1, track=1))  # its place by 2's: still 1's
    a.on_look(look([("green-mushroom", 251.0, None, 1), ("green-mushroom", 262.0, None, 2)], t=now))
    assert [mk.m.track for mk in a.marks] == [2] and a.target("right").track == 2
    dead.add(2)  # 2 found dead (another player's kill seen): no target, its mark closed, seen as it is
    a.on_look(look([("green-mushroom", 251.0, None, 1), ("green-mushroom", 262.0, None, 2)], t=now + 0.1))
    assert a.marks == [] and a.target("right") is None
