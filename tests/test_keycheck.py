"""Her regular attack's key doubted, checked and set right (behavior.keycheck). The watch on synthetic presses and
looks; the live runs of 2026-10-25 to 28 replayed through it (tests/fixtures/regular_live_2026-10-28.json: at 00:31 on
2026-10-28, the game's Attack moved off Ctrl, the doubt at 00:31:19.8; never in the other 15 runs); the check with a
stand-in game (its window drawn from the game's pictures, perception.keybindings.compose): her stars, the window
opened, read and closed, each outcome, the Lie Detector halfway, the gap between checks, the magician. End to end in
the simulator: loop_tree_grind_attack_key, loop_tree_attack_unbound, loop_tree_attack_misses and
loop_tree_attack_no_darts (tests/test_sim.py runs every scenario)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gamelab import character
from gamelab.behavior import keycheck
from gamelab.behavior.darts import DartWatch, OutOfDarts
from gamelab.behavior.keycheck import (CHECK_GAP_S, CHECKS_MAX, DOUBT_PRESSES, KeyCheck, KeyDoubt, RegularWatch,
                                       regular_reach)
from gamelab.control import alarm, client
from gamelab.localization.monsters import Monster
from gamelab.perception import keybindings as kb
from gamelab.sim import scenarios
from maplehelper.keys import parse as K

FIXTURES = Path(__file__).parent / "fixtures"


def mon(kind="horny-mushroom", x=100.0, bar=0.09, platform=23, y=-450.0, track=None) -> Monster:
    return Monster(SimpleNamespace(kind=kind), x, y, platform, bar, track=track)


def press_and_look(w: RegularWatch, t: float, m: Monster, bar_after: float | None = None, **kw) -> float:
    """A regular press at `t` at `m`, then looks 0.4 and 1.3 s after it seeing it with `bar_after` (its own bar by
    default): the time after."""
    w.pressed(t, m, **kw)
    seen = mon(m.kind, m.x, m.bar if bar_after is None else bar_after, m.platform, m.y, m.track)
    w.look(t + 0.4, [seen])
    w.look(t + 1.3, [seen])
    return t + 1.3


# --- the watch ------------------------------------------------------------------------------------------------------

def test_five_regular_presses_without_effect_in_a_row_are_the_doubt_once():
    w, t = RegularWatch(), 0.0
    for i in range(DOUBT_PRESSES - 1):
        t = press_and_look(w, t, mon())
        assert w.due() is None
    t = press_and_look(w, t, mon())
    d = w.due()
    assert d is not None and len(d.presses) == 5 and d.first == 0.0 and w.due() is None
    assert "TOOK NO EFFECT 5 TIMES IN A ROW (horny-mushroom, its bar 9% throughout" in d.text()
    assert w.counts["none"] == 5 and w.longest == 5 and w.run == []  # (its run starts again)


def test_an_effect_ends_the_run():
    w, t = RegularWatch(), 0.0
    for _ in range(4):
        t = press_and_look(w, t, mon())
    t = press_and_look(w, t, mon(), bar_after=0.05)  # its bar lower: her damage
    assert w.counts["effect"] == 1 and w.run == []
    for _ in range(4):
        t = press_and_look(w, t, mon())
    assert w.due() is None and w.longest == 4


@pytest.mark.parametrize("what", ["kill", "gain", "gain of another kind", "party share"])
def test_a_kill_placed_on_it_or_a_gain_naming_its_kind_is_an_effect_a_share_isnt(what):
    w, t = RegularWatch(), 0.0
    for _ in range(4):
        t = press_and_look(w, t, mon())
    m = mon()
    w.pressed(t, m)
    if what == "kill":
        w.killed(t + 0.5, "horny-mushroom", 23, 100.0)
    else:
        w.heard([(t + 0.45, {"gain": frozenset({"horny-mushroom"}), "gain of another kind": frozenset({"slime"}),
                             "party share": None}[what])])
    w.look(t + 1.3, [m])
    effect = what in ("kill", "gain")
    assert (w.counts["effect"] == 1) == effect
    assert (w.due() is None) == effect


@pytest.mark.parametrize("why, kw", [
    ("on a rope", {"roped": True}), ("out of reach", {"reach_ok": False}),
    ("its HP not known", {"known": False}), ("dying", {"dying": True}), ("a jump's", {"ground": False})])
def test_presses_that_cant_show_what_they_did_are_passed_over(why, kw, monkeypatch):
    monkeypatch.setattr(keycheck, "JUMPS_JUDGED", False)  # (a jump throw's left out as designed: below, judged)
    w, t = RegularWatch(), 0.0
    for _ in range(4):
        t = press_and_look(w, t, mon())
    t = press_and_look(w, t, mon(), **kw)
    assert w.counts[f"passed: {why}"] == 1 and len(w.run) == 4 and w.due() is None
    t = press_and_look(w, t, mon())  # (passed over: the run goes on)
    assert w.due() is not None


def test_a_jump_throw_s_press_is_judged_too():
    """keycheck.JUMPS_JUDGED: left out, the simulator's runs with the game's Attack moved never doubted in 3 of 12 (their
    presses mostly jump throws at a monster she couldn't finish); live, judged, they add 4 effects and no press
    without one."""
    assert keycheck.JUMPS_JUDGED
    w, t = RegularWatch(), 0.0
    for _ in range(5):
        t = press_and_look(w, t, mon(), ground=False)
    assert w.due() is not None


def test_no_bar_nothing_in_its_way_a_touch_or_a_cast_s_effect_are_passed_over():
    w = RegularWatch()
    w.pressed(0.0, mon(bar=None))
    w.pressed(1.0, None)
    assert w.counts["passed: no bar to judge by"] == 1 and w.counts["passed: nothing in its way"] == 1
    m = mon()
    w.pressed(2.0, m)
    w.touched(1.7)  # within KNOCK_S before the key
    w.look(3.3, [m])
    assert w.counts["passed: touched"] == 1
    w.pressed(4.0, m)
    w.cast(4.3, m)  # a cast at it after the press, before its bar showed lower
    w.look(5.3, [mon(bar=0.04)])
    assert w.counts["passed: a cast's effect maybe"] == 1 and w.counts["effect"] == 0 and w.run == []


def test_a_bar_higher_than_its_own_is_another_monster_s_and_a_press_unjudged_in_time_is_passed():
    w = RegularWatch()
    w.pressed(0.0, mon(bar=0.11))
    w.look(1.3, [mon(bar=0.24)])
    assert w.pending and w.pending[0].verdict is None
    w.look(2.1, [mon(bar=0.24)])
    assert w.counts["passed: not judged in time"] == 1


def test_a_target_followed_by_its_track_or_walking_unseen_for_a_while():
    w = RegularWatch()
    w.pressed(0.0, mon(x=0.0, track=7))
    w.look(1.3, [mon(x=200.0, track=7), mon(x=0.0, track=8)])  # (its track, wherever it walked)
    assert w.counts["none"] == 1
    w.pressed(2.0, mon(x=0.0))
    w.look(3.3, [mon(x=130.0)])  # 130 off: TRACK_SLACK and WALK_S x 1.3 s
    assert w.counts["none"] == 2
    w.pressed(4.0, mon(x=0.0))
    w.look(5.3, [mon(x=130.0), mon(x=-130.0)])  # two that could be it: not judged
    w.look(6.1, [mon(x=130.0), mon(x=-130.0)])
    assert w.counts["none"] == 2 and w.counts["passed: not judged in time"] == 1


def test_held_off_and_its_targets():
    w = RegularWatch()
    m, other = mon(x=0.0), mon(x=500.0)
    assert not w.holds(m)
    w.no_regular_at.append(m)
    assert w.holds(mon(x=10.0)) and not w.holds(other) and not w.holds(None)
    w.held = True
    assert w.holds(other)
    w.held, w.off = False, "ATTACK on no key"
    assert w.holds(other)


def test_checks_a_run_and_the_gap_between_them():
    w = RegularWatch()
    assert w.may_check(0.0) is None
    w.checked(0.0)
    assert "none within 300 s" in w.may_check(CHECK_GAP_S - 1)
    assert w.may_check(CHECK_GAP_S) is None
    w.checked(400.0), w.checked(800.0)
    assert len(w.checks) == CHECKS_MAX and "3 checks this run already" in w.may_check(2000.0)


def test_the_summary_says_nothing_until_a_doubt():
    w, t = RegularWatch(), 0.0
    for _ in range(4):
        t = press_and_look(w, t, mon())
    assert w.summary() == ""
    press_and_look(w, t, mon())
    w.due()
    assert w.summary().startswith("her regular presses: 5 judged, 5 took no effect, 5 in a row")


# --- the live runs ----------------------------------------------------------------------------------------------

def _replay(run: dict, kinds: list[str], model) -> tuple[list, RegularWatch]:
    """A run's events (tests/fixtures/regular_live_2026-10-28.json: its doc) through a RegularWatch as the hunter feeds
    it: each look that searched (her place, its monsters), each attack record's press at the monster it names in the
    last look (by its x; one of its kind if the record has none), from the ground if the stance before it threw from
    there, her kills placed, her EXP's gains, her touches; the doubts (time, Doubt)."""
    w = RegularWatch(reach_ok=regular_reach)
    t, last, out = run["start"], None, []
    for row in run["events"]:
        t += row[0] / 100
        what = row[1]
        if what == "s":
            x, y, plat, mons = row[2:]
            ms = [Monster(SimpleNamespace(kind=kinds[m[0]]), m[1], m[2], m[3], None if m[4] is None else m[4] / 1000)
                  for m in mons]
            p = None if plat is None else model.platform(plat)
            last = SimpleNamespace(t=t, where=SimpleNamespace(x=x, y=y, platform=p), platform=p, monsters=ms)
            if p is not None:
                w.look(t, ms, last)
                if (d := w.due()) is not None:
                    out.append((t, d))
        elif what == "a":
            key, kind, x, act = row[2:]
            target = None
            if last is not None and kind is not None:
                same = [m for m in last.monsters if m.kind == kinds[kind]]
                same = [m for m in same if abs(m.x - x) < 1.0] if x is not None else same if len(same) == 1 else []
                target = same[0] if same else None
            if key == "lucky_seven":
                w.cast(t, target)
            else:
                ok = target is not None and last.platform is not None and regular_reach(last, target)
                w.pressed(t, target, ground=act == "throw", roped=last is None or last.platform is None, reach_ok=ok)
        elif what == "g":
            w.heard([(t, None if row[2] is None else frozenset(kinds[i] for i in row[2]))])
        elif what == "k":
            w.killed(t, kinds[row[2]], row[3], row[4])
        elif what == "h":
            w.touched(t)
    return out, w


@pytest.fixture(scope="module")
def live():
    model = scenarios.real_map(scenarios.TREE)
    if model is None:
        pytest.skip("needs Tree Dungeon's game data (gamedata/data)")
    fx = json.loads((FIXTURES / "regular_live_2026-10-28.json").read_text(encoding="utf-8"))
    return {name: _replay(run, fx["kinds"], model) for name, run in fx["runs"].items()}, fx


AT_0031 = 1793172660.0  # 2026-10-28 00:31:00 where the runs were (the telemetry's time.time())


def _clock(t: float) -> str:
    """00:31:SS.s for a time in that minute."""
    return f"00:31:{int(t - AT_0031):02d}.{int((t - AT_0031) * 10) % 10}"


def test_at_00_31_on_2026_10_28_the_doubt_comes_at_00_31_19_8(live):
    """The game's Attack off Ctrl from before 00:30:49 to about 00:33:00: 88 regular presses in the run, 20 of them in
    00:31:14-33 at one Horny Mushroom, its bar 9%. The doubt at the look of 00:31:19.8 (its 5th press keyed at 00:31:18.5),
    13 s before her last useless press (00:31:33.0)."""
    runs, fx = live
    doubts, w = runs["20261028-003049"]
    t, d = doubts[0]
    assert _clock(t) == "00:31:19.7" and len(d.presses) == 5
    assert [_clock(p.t) for p in d.presses] == ["00:31:14.2", "00:31:15.9", "00:31:16.7", "00:31:17.6", "00:31:18.4"]
    assert {p.kind for p in d.presses} == {"horny-mushroom"} and {round(p.after, 2) for p in d.presses} == {0.09}
    assert fx["runs"]["20261028-003049"]["regular"] == 88


def test_the_other_runs_never_doubt_two_in_a_row_at_the_most(live):
    """15 runs of 2026-10-25 to 28 with the keys right (245 minutes, 595 regular presses): no doubt; 60 judged, 5 of
    them without effect, 2 in a row at the most (the doubt at 5)."""
    runs, fx = live
    others = {name: v for name, v in runs.items() if name != "20261028-003049"}
    assert len(others) == 15 and sum(fx["runs"][n]["regular"] for n in others) == 595
    for name, (doubts, w) in others.items():
        assert not doubts and w.longest <= 2, (name, w.longest)
    judged = sum(w.counts["effect"] + w.counts["none"] for _, w in others.values())
    none = sum(w.counts["none"] for _, w in others.values())
    assert (judged, none) == (60, 5)


# --- the check, with a stand-in game ------------------------------------------------------------------------------

class Game:
    """The game's window as the check sees it: the backslash opens the KEY BINDINGS window (unless `opens` is False),
    Esc closes it (`stuck`: not the first Esc), Esc with nothing open the MENU; its frame the window drawn from the
    game's pictures with `bindings`."""

    def __init__(self, bindings=None, opens=True, stuck=0, open_at_start=False, alarm_after=None):
        self.bindings = dict(kb.LIVE if bindings is None else bindings)
        self.opens, self.stuck, self.open = opens, stuck, open_at_start
        self.alarm_after = alarm_after  # plans after which the player refuses (the Lie Detector)
        self.plans, self.menu = [], 0

    def play(self, events, name, **kw):
        if self.alarm_after is not None and len(self.plans) >= self.alarm_after:
            raise client.LieDetectorAlarm("Lie Detector seen")
        self.plans.append((name, [e[1] for e in events if len(e) == 3 and e[2] == "down"]))
        for e in events:
            if len(e) == 3 and e[2] == "down":
                if e[1] == "\\" and self.opens:
                    self.open = True
                elif e[1] == "esc":
                    if not self.open:
                        self.menu += 1
                    elif self.stuck:
                        self.stuck -= 1
                    else:
                        self.open = False

    def capture(self):
        return kb.compose(self.bindings, 2) if self.open else np.full((768, 1366, 3), (90, 140, 70), np.uint8)


class Hunter:
    """What KeyCheck asks of a hunter: its attack choice's watch, her darts' watch, her actions' keys and plays, her
    looks and her eyes' whole frame; rekey and regular_off as the Hunter's (behavior.hunter)."""

    def __init__(self, game: Game, mp_shown_at=None):
        self.game, self.lines = game, []
        self.log = self.lines.append
        self.attacks = SimpleNamespace(watch=RegularWatch(), skill=None, mp=None, max_mp=623.0)
        self.darts = DartWatch()
        self.darts.mp_shown_at = mp_shown_at
        keys = SimpleNamespace(attack=K("ctrl"), lucky_seven=K("shift"), jump=K("alt"), pick_up=K("z"),
                               pet_food=K("u"), up=K("up"), down=K("down"), left=K("left"), right=K("right"),
                               chair=None, sit=None)
        self.actions = SimpleNamespace(keys=keys, play=game.play, attack_at=0.0)
        self.senses = SimpleNamespace(eyes=SimpleNamespace(_capture=game.capture))
        self.rekeyed, self.off = None, None

    def look(self):
        return SimpleNamespace(t=0.0, where=SimpleNamespace(x=0.0, y=0.0), monsters=[], platform=None)

    def rekey(self, key):
        self.rekeyed = key

    def regular_off(self, why):
        self.off = why

    def hold_regular(self, on=True):
        self.attacks.watch.held = on


def _doubt(t0: float = 1000.0) -> keycheck.Doubt:
    return keycheck.Doubt([keycheck.Press(t0 + i, mon(), 0.09, after=0.09, at=t0 + i + 1.3) for i in range(5)], t0 + 6)


@pytest.fixture
def quiet(monkeypatch):
    """No sleeping, no paging (kept), no alarm."""
    pages = []
    monkeypatch.setattr(keycheck.time, "sleep", lambda s: None)
    monkeypatch.setattr(alarm, "page", lambda title, text: pages.append((title, text)))
    monkeypatch.setattr(client, "check_alarm", lambda: None)
    return pages


def test_attack_on_another_key_her_regular_attack_is_that_key_paged(quiet):
    g = Game({**kb.LIVE, "ctrl": None, "a": 52})
    h = Hunter(g, mp_shown_at=1001.0)
    out = KeyCheck(h).run(_doubt())
    assert out.what == "switched" and out.key == "a" and h.rekeyed == "a" and out.stars == "shown"
    assert [p for p in g.plans] == [("key_bindings_mouse", []), ("key_bindings", ["\\"]),
                                    ("key_bindings_close", ["esc"])]
    assert not g.open and g.menu == 0
    assert any(line.startswith("ATTACK KEY: THE GAME HAS ATTACK ON A, NOT CTRL (Key Bindings, preset 2): her regular "
                               "attack is A for the rest of this run (config.toml unchanged); CTRL does nothing now")
               for line in h.lines)
    assert len(quiet) == 1 and quiet[0][0] == "MapleStory: the attack key"


def test_attack_on_one_of_her_other_keys_is_switched_and_the_clash_said(quiet):
    h = Hunter(Game({**kb.LIVE, "ctrl": None, "shift": 52}), mp_shown_at=1001.0)
    out = KeyCheck(h).run(_doubt())
    assert out.what == "switched" and out.key == "shift" and "SHIFT is her lucky_seven key too" in out.text


@pytest.mark.parametrize("bindings, tray, says", [
    ({"ctrl": None}, None, "ON NO KEY"),
    ({"ctrl": None, "esc": 52}, None, "ON ESC (NOT A KEY TO PRESS)"),
    ({"ctrl": None, "f1": 52}, None, "ON F1 (NOT A KEY TO PRESS)")])
def test_attack_on_no_key_or_one_not_to_press_no_more_regular_attacks(quiet, bindings, tray, says):
    h = Hunter(Game({**kb.LIVE, **bindings}), mp_shown_at=1001.0)
    out = KeyCheck(h).run(_doubt())
    assert out.what == "off" and h.off and h.rekeyed is None and says in out.text
    assert "no more regular attacks this run (Lucky Seven only)" in out.text and len(quiet) == 1


def test_attack_on_ctrl_as_configured_its_targets_skipped_nobody_paged(quiet):
    h = Hunter(Game(), mp_shown_at=1001.0)
    d = _doubt()
    out = KeyCheck(h).run(d)
    assert out.what == "as configured" and h.rekeyed is None and h.off is None and not quiet
    assert "ATTACK IS ON CTRL AS CONFIGURED" in out.text and "misses? out of reach?" in out.text
    assert h.attacks.watch.holds(mon()) and not h.attacks.watch.holds(mon(x=900.0))


def test_the_window_not_opening_sends_no_esc_and_turns_the_regular_attack_off(quiet):
    g = Game(opens=False)
    h = Hunter(g, mp_shown_at=1001.0)
    out = KeyCheck(h).run(_doubt())
    assert [name for name, _ in g.plans] == ["key_bindings_mouse", "key_bindings"]  # (no Esc: the MENU would open)
    assert g.menu == 0 and out.what == "off" and "NOT READ (not open)" in out.text


def test_open_already_it_s_read_without_the_backslash(quiet):
    g = Game(open_at_start=True)
    out = KeyCheck(Hunter(g, mp_shown_at=1001.0)).run(_doubt())
    assert [keys for _, keys in g.plans] == [[], ["esc"]] and out.what == "as configured" and not g.open


def test_a_second_esc_if_it_s_still_open_then_said_and_paged(quiet):
    g = Game(stuck=1)
    KeyCheck(Hunter(g, mp_shown_at=1001.0)).run(_doubt())
    assert [keys for _, keys in g.plans][-2:] == [["esc"], ["esc"]] and not g.open and not quiet
    g = Game(stuck=5)
    h = Hunter(g, mp_shown_at=1001.0)
    KeyCheck(h).run(_doubt())
    assert [keys for _, keys in g.plans].count(["esc"]) == 2 and g.open
    assert any("KEY BINDINGS window may still be open" in t for _, t in quiet)


def test_the_lie_detector_with_the_window_open_says_so_and_goes_on(quiet):
    g = Game(alarm_after=2)  # (the mouse, the backslash; then the player refuses)
    h = Hunter(g, mp_shown_at=1001.0)
    with pytest.raises(client.LieDetectorAlarm):
        KeyCheck(h).run(_doubt())
    assert g.open and any("close it with Esc or CANCEL" in t for _, t in quiet)


def test_her_stars_not_known_without_a_lucky_seven_key_and_none_for_the_magician(quiet):
    h = Hunter(Game(), mp_shown_at=None)
    out = KeyCheck(h).run(_doubt())
    assert out.stars == "not known" and out.what == "as configured"
    from gamelab.classes import magician
    with character.using(magician.profile()):
        h = Hunter(Game(), mp_shown_at=None)
        assert KeyCheck(h).run(_doubt()).stars == "no stars"
        assert any("none to run out of" in line for line in h.lines)


def test_no_check_within_the_gap_its_targets_held_and_past_three_checks_off(quiet, monkeypatch):
    h = Hunter(Game(), mp_shown_at=1001.0)
    now = [5000.0]
    monkeypatch.setattr(keycheck.time, "monotonic", lambda: now[0])
    check = KeyCheck(h)
    assert check.run(_doubt()).what == "as configured"
    now[0] += 60
    out = check.run(_doubt(5050.0))
    assert out.what == "held" and "the last check 60 s ago" in out.text and h.off is None
    for _ in range(2):
        now[0] += CHECK_GAP_S
        assert check.run(_doubt(now[0] - 10)).what == "as configured"
    now[0] += CHECK_GAP_S
    out = check.run(_doubt(now[0] - 10))
    assert out.what == "off" and "3 checks this run already" in out.text and h.off


def test_her_stars_test_five_casts_without_mp_is_out_of_darts(quiet, monkeypatch):
    """No cast's MP since the doubt's first press: up to 5 Lucky Seven casts at a monster in reach, judged by her MP bar;
    5 that took none (untouched) are out of darts, the stop as it is."""
    h = Hunter(Game(), mp_shown_at=None)
    from gamelab.attack import choice
    h.attacks.skill, h.attacks.mp = choice.lucky_seven(16), 0.5
    h.hit_at = None
    check = KeyCheck(h)
    monkeypatch.setattr(check, "_cast_side", lambda s, d: "right")
    keyed = []
    monkeypatch.setattr(check, "_cast", lambda side: keyed.append(side) or True)
    monkeypatch.setattr(check, "_mp_shown", lambda key: False)
    h.darts.share = 0.5
    with pytest.raises(OutOfDarts) as e:
        check.run(_doubt())
    assert len(keyed) == 5 and "then 5 Lucky Seven casts took no MP" in str(e.value)
    h2 = Hunter(Game(), mp_shown_at=None)
    h2.attacks.skill, h2.attacks.mp, h2.hit_at, h2.darts.share = choice.lucky_seven(16), 0.5, None, 0.5
    check = KeyCheck(h2)
    monkeypatch.setattr(check, "_cast_side", lambda s, d: "right")
    monkeypatch.setattr(check, "_cast", lambda side: True)
    shown = iter([False, True])
    monkeypatch.setattr(check, "_mp_shown", lambda key: next(shown))
    assert check.run(_doubt()).stars == "test"


def test_key_doubt_isn_t_a_hunt_stop():
    from gamelab.behavior.safety import HuntStop
    assert not issubclass(KeyDoubt, HuntStop) and "TOOK NO EFFECT" in str(KeyDoubt(_doubt()))


def test_held_a_regular_throw_is_lucky_seven_with_the_mp_for_a_cast_else_none():
    """behavior.attacks.Attacks.key with the watch holding the regular attack: Lucky Seven while she has the MP for a
    cast, no throw (None) without it, the regular attack again once let go; with the regular attack alone her
    Actions' own presses are told to the watch at their throw (on_throw), their target as a throw that way takes."""
    from maplehelper.config import GameKeys
    from maplehelper.engine import RIGHT
    from gamelab.behavior.attacks import Attacks
    model = SimpleNamespace(mob_hp={"Horny Mushroom": 403.0}, mob_pdd={"Horny Mushroom": 10.0})
    keys = GameKeys(left=K("left"), right=K("right"), jump=K("alt"), attack=K("ctrl"), lucky_seven=K("shift"))
    lines = []
    a = Attacks(model, keys, "regular", max_mp=623.0, level=16.0, log=lines.append)
    a.watch = RegularWatch(held=True)
    a.mp = 0.5
    assert a.key(RIGHT) == "shift" and "the regular attack held (its key doubted)" in lines[-1]
    a.on_throw(1.0, RIGHT)
    a.mp = 0.01
    assert a.key(RIGHT) is None and "no throw: the regular attack is held" in lines[-1]
    a.watch.held = False
    assert a.key(RIGHT) == "ctrl"
    a.on_throw(2.0, RIGHT)
    a.on_throw(3.0, RIGHT)  # (not asked: the regular attack alone, her Actions' own key)
    assert a.watch.counts["passed: nothing in its way"] == 2
