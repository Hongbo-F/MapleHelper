"""Her regular attack's key doubted, checked and set right (behavior.keycheck). The watch on synthetic presses and
looks; the live runs of 2026-10-25 to 28 replayed through it (tests/fixtures/regular_live_2026-10-28.json: at 00:31 on
2026-10-28, the game's Attack moved off Ctrl, the doubt at 00:31:20.9; never in the other 15 runs); the check with a
stand-in game (its window drawn from the game's pictures, perception.keybindings.compose): a place and a quiet moment
(put off and tried again), her stars, the window opened, read and closed (late, slow to close, partly off, not
recognised, no frames, another size), each outcome, her moves' keys, the Lie Detector halfway, the safety stop, the gap
between checks, the magician. End to end in the simulator: loop_tree_grind_attack_key, loop_tree_attack_unbound,
loop_tree_attack_misses and loop_tree_attack_no_darts (tests/test_sim.py runs every scenario)."""

import json
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gamelab import character
from gamelab.behavior import keycheck
from gamelab.behavior.darts import DartWatch, OutOfDarts
from gamelab.behavior.keycheck import (CHECK_GAP_S, CHECKS_MAX, DEFER_GAP_S, DEFERS_MAX, DOUBT_PRESSES, KeyCheck,
                                       KeyCheckStop, KeyDoubt, RegularWatch, regular_reach)
from gamelab.behavior.safety import SafetyStop
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


def test_an_unhurt_target_is_judged_a_bar_up_after_is_an_effect_none_twice_no_effect():
    """keycheck.UNHURT_JUDGED (the review of 2026-10-28: with the regular attack alone and its key wrong no monster is
    ever hurt, so no press at one with no bar up was ever judged): its bar up by the judging look, an effect; still none
    on two looks UNHURT_AGAIN_S apart, no effect; five of those the doubt."""
    assert keycheck.UNHURT_JUDGED
    w = RegularWatch()
    w.pressed(0.0, mon(bar=None, x=900.0))
    w.look(0.7, [mon(bar=0.93, x=900.0)])
    assert w.counts["effect"] == 1
    t = 2.0
    for i in range(5):
        w.pressed(t, mon(bar=None))
        w.look(t + 0.7, [mon(bar=None)])  # (once: a bar missed in one look isn't none)
        assert len(w.run) == i
        w.look(t + 1.1, [mon(bar=None)])
        assert len(w.run) == i + 1
        t += 2.0
    d = w.due()
    assert d is not None and w.counts["none"] == 5 and "its bar no bar throughout" in d.text()


def test_a_bar_missed_in_one_look_is_the_one_seen_before_it():
    """Live at 00:31:14.7 the Horny Mushroom at 9% read with no bar between two reads of it at 9%: a press then is at
    9% (BAR_MEMORY_S), not at an unhurt one whose bar then shows (an effect it wasn't)."""
    w = RegularWatch()
    w.look(0.0, [mon(bar=0.09)])
    w.pressed(0.8, mon(bar=None, x=110.0))
    w.look(1.5, [mon(bar=0.09, x=115.0)])
    assert w.counts["none"] == 1 and w.counts["effect"] == 0
    assert w.run[0].bar == 0.09 and not w.run[0].unhurt


def test_no_bar_nothing_in_its_way_a_touch_or_a_cast_s_effect_are_passed_over(monkeypatch):
    monkeypatch.setattr(keycheck, "UNHURT_JUDGED", False)
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


def test_a_target_with_another_monster_nearer_its_way_isn_t_judged():
    """regular_reach (keycheck._in_front): another monster her star reaches nearer that way, or one up close her claw
    swipes, takes the attack, so the target's bar doesn't move with her keys right (loop_tree_ore_waits doubted so: its
    stars took a Horny Mushroom hidden under its damage numbers, which RegularWatch.hidden keeps, 1.8 s)."""
    model = scenarios.real_map(scenarios.TREE)
    if model is None:
        pytest.skip("needs Tree Dungeon's game data (gamedata/data)")
    p = model.platform(23)
    target = mon(x=-100.0, y=p.y_at(-100.0), platform=23)

    def look(*others):
        x = 100.0
        return SimpleNamespace(where=SimpleNamespace(x=x, y=p.y_at(x), platform=p), platform=p,
                               monsters=[target, *others])
    assert regular_reach(look(), target)
    assert not regular_reach(look(mon(x=0.0, y=p.y_at(0.0), platform=23)), target)      # (nearer, the star's)
    assert not regular_reach(look(mon(x=75.0, y=p.y_at(75.0), platform=23)), target)    # (up close, the claw's)
    assert regular_reach(look(mon(x=300.0, y=p.y_at(300.0), platform=23)), target)      # (the other way)
    w = RegularWatch()
    w.look(0.0, [target, mon(x=0.0)])
    w.look(1.0, [target])
    assert [m.x for m in w.hidden(1.2, [target])] == [0.0] and w.hidden(2.0, [target]) == []


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
    there (in reach with the monsters the watch saw lately counted: RegularWatch.hidden, as Attacks counts them), her
    kills placed, her EXP's gains, her touches; the doubts (time, Doubt)."""
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
                about = SimpleNamespace(where=last.where, platform=last.platform,
                                        monsters=last.monsters + w.hidden(t, last.monsters))  # (as Attacks has it)
                ok = target is not None and last.platform is not None and regular_reach(about, target)
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


def test_at_00_31_on_2026_10_28_the_doubt_comes_at_00_31_20_9(live):
    """The game's Attack off Ctrl from before 00:30:49 to about 00:33:00: 88 regular presses in the run, 20 of them in
    00:31:14-33 at one Horny Mushroom, its bar 9%. The doubt at the look of 00:31:20.9 (its 5th press keyed at
    00:31:19.3; the one of 00:31:15.0 at it as its bar was missed in the look before, BAR_MEMORY_S; those of 14.2 and
    18.4 passed over, a monster seen lately nearer its way, and at its look), 12 s before her last useless press
    (00:31:33.0)."""
    runs, fx = live
    doubts, w = runs["20261028-003049"]
    t, d = doubts[0]
    assert _clock(t) == "00:31:20.9" and len(d.presses) == 5
    assert [_clock(p.t) for p in d.presses] == ["00:31:15.0", "00:31:15.9", "00:31:16.7", "00:31:17.6", "00:31:19.3"]
    assert {p.kind for p in d.presses} == {"horny-mushroom"} and {p.bar_text() for p in d.presses} == {"9%"}
    assert fx["runs"]["20261028-003049"]["regular"] == 88


def test_the_other_runs_never_doubt_two_in_a_row_at_the_most(live):
    """15 runs of 2026-10-25 to 28 with the keys right (245 minutes, 595 regular presses): no doubt; 66 judged (unhurt
    targets and bars missed in a look judged too, none with a monster nearer its way), 4 of them without effect, 2 in a
    row at the most (the doubt at 5)."""
    runs, fx = live
    others = {name: v for name, v in runs.items() if name != "20261028-003049"}
    assert len(others) == 15 and sum(fx["runs"][n]["regular"] for n in others) == 595
    for name, (doubts, w) in others.items():
        assert not doubts and w.longest <= 2, (name, w.longest)
    judged = sum(w.counts["effect"] + w.counts["none"] for _, w in others.values())
    none = sum(w.counts["none"] for _, w in others.values())
    assert (judged, none) == (66, 4)


# --- the check, with a stand-in game ------------------------------------------------------------------------------

PLAIN = (90, 140, 70)


class Game:
    """The game's window as the check sees it: the backslash opens the KEY BINDINGS window (unless `opens` is False),
    shown from the `open_after`-th frame after it, its frames failing their checks for `blind` frames once it's open;
    Esc closes it (`stuck`: not the first Esc; `slow`: frames still showing it after it closed), Esc with nothing open the
    MENU; its frame the window drawn from the game's pictures with `bindings` at `at` (`cover_title`: something over its
    title), on a client of `size`."""

    def __init__(self, bindings=None, opens=True, stuck=0, open_at_start=False, alarm_after=None, open_after=0,
                 blind=0, at=kb.OPEN_AT, cover_title=False, slow=0, size=(1366, 768)):
        self.bindings = dict(kb.LIVE if bindings is None else bindings)
        self.opens, self.stuck, self.open = opens, stuck, open_at_start
        self.alarm_after = alarm_after  # plans after which the player refuses (the Lie Detector)
        self.open_after, self.blind, self.at, self.cover_title, self.slow, self.size = (
            open_after, blind, at, cover_title, slow, size)
        self.plans, self.menu = [], 0
        self.since, self.closing, self._blind = None, 0, 0

    def play(self, events, name, **kw):
        if self.alarm_after is not None and len(self.plans) >= self.alarm_after:
            raise client.LieDetectorAlarm("Lie Detector seen")
        self.plans.append((name, [e[1] for e in events if len(e) == 3 and e[2] == "down"]))
        for e in events:
            if len(e) == 3 and e[2] == "down":
                if e[1] == "\\" and self.opens:
                    self.open, self.since, self._blind = True, 0, self.blind
                elif e[1] == "esc":
                    if not self.open:
                        self.menu += 1
                    elif self.stuck:
                        self.stuck -= 1
                    else:
                        self.open, self.closing = False, self.slow

    def capture(self):
        if self.open and self._blind > 0:
            self._blind -= 1
            raise RuntimeError("the minimap where it isn't")
        shown = (self.open and (self.since is None or self.since >= self.open_after)) or self.closing > 0
        if self.since is not None:
            self.since += 1
        if self.closing > 0:
            self.closing -= 1
        frame = np.full((768, 1366, 3), PLAIN, np.uint8)
        if shown:
            frame = kb.compose(self.bindings, 2, at=self.at, frame=frame)
            if self.cover_title:
                x, y = self.at
                frame[y:y + 24, x:x + kb.WINDOW[0]] = (200, 60, 60)
        if self.size != (1366, 768):
            big = np.full((self.size[1], self.size[0], 3), PLAIN, np.uint8)
            big[:768, :1366] = frame
            return big
        return frame


class Hunter:
    """What KeyCheck asks of a hunter: its attack choice's watch, her darts' watch, her actions' keys and plays, her
    looks (`monsters` what they see; `place`: her platform, or None) and her eyes' whole frame; hold_regular, hold_at,
    rekey and regular_off as the Hunter's (behavior.hunter)."""

    def __init__(self, game: Game, mp_shown_at=None):
        self.game, self.lines = game, []
        self.log = self.lines.append
        self.attacks = SimpleNamespace(watch=RegularWatch(), skill=None, mp=None, max_mp=623.0,
                                       target=lambda side: None)
        self.darts = DartWatch()
        self.darts.mp_shown_at = mp_shown_at
        keys = SimpleNamespace(attack=K("ctrl"), lucky_seven=K("shift"), jump=K("alt"), pick_up=K("z"),
                               pet_food=K("u"), up=K("up"), down=K("down"), left=K("left"), right=K("right"),
                               chair=None, sit=None)
        self.actions = SimpleNamespace(keys=keys, play=game.play, attack_at=0.0)
        self.senses = SimpleNamespace(eyes=SimpleNamespace(_capture=game.capture))
        self.rekeyed, self.off = None, None
        self.hit_at, self.hp_watch = None, None
        self.monsters, self.place, self.looks = [], SimpleNamespace(id=23), 0
        self.retreats = 0

    def look(self):
        self.looks += 1
        return SimpleNamespace(t=time.monotonic(), where=SimpleNamespace(x=0.0, y=0.0), monsters=list(self.monsters),
                               platform=self.place)

    def on_rope(self, s):
        return False

    def knocked(self, s):
        return False

    def rekey(self, key):
        self.rekeyed = key
        self.actions.keys.attack = K(key)
        self.attacks.watch.held = False

    def regular_off(self, why):
        self.off = why
        self.attacks.watch.held = False

    def hold_regular(self, on=True):
        self.attacks.watch.held = on

    def hold_at(self, targets):
        self.attacks.watch.no_regular_at.extend(targets)
        self.attacks.watch.held = False


def _doubt(t0: float = 1000.0) -> keycheck.Doubt:
    return keycheck.Doubt([keycheck.Press(t0 + i, mon(), 0.09, after=0.09, at=t0 + i + 1.3) for i in range(5)], t0 + 6)


@pytest.fixture
def quiet(monkeypatch, tmp_path):
    """No sleeping, no paging (kept), no alarm, no retreat (counted), the window's frames kept in a scratch folder."""
    from gamelab import paths
    monkeypatch.setattr(paths, "SHOTS", tmp_path)
    pages = []
    monkeypatch.setattr(keycheck.time, "sleep", lambda s: None)
    monkeypatch.setattr(alarm, "page", lambda title, text: pages.append((title, text)))
    monkeypatch.setattr(client, "check_alarm", lambda: None)
    from gamelab.behavior import retreat

    def away(h, seen, why, log=None, rng=None):
        h.retreats += 1
        if getattr(h, "clear_after_retreat", True):
            h.monsters = []
        return SimpleNamespace(result="reached")
    monkeypatch.setattr(retreat, "away", away)
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


def test_attack_on_her_lucky_seven_s_key_stops_the_hunt(quiet):
    """The game's ATTACK on Shift, her Lucky Seven's key (a picture: where her Lucky Seven went can't be told off the
    window): her casts would attack, so the hunt stops (KeyCheckStop), not switched."""
    h = Hunter(Game({**kb.LIVE, "ctrl": None, "shift": 52}), mp_shown_at=1001.0)
    with pytest.raises(KeyCheckStop) as e:
        KeyCheck(h).run(_doubt())
    assert "SHIFT is her lucky seven key" in str(e.value) and h.rekeyed is None and not e.value.window
    assert e.value.exit_code == keycheck.EXIT


@pytest.mark.parametrize("bindings, says", [
    ({"ctrl": 53, "alt": 52}, "JUMP is on CTRL"),       # (the review's: Attack and Jump swapped)
    ({"ctrl": None, "alt": 52}, "JUMP is on no key"),
])
def test_attack_on_her_jump_key_stops_the_hunt(quiet, bindings, says):
    """The review's case: the game's ATTACK on Alt, her jump key, was switched to and she jumped on a key that attacks.
    Her moves' keys not as she has them stop the hunt (not followed: her trips' chains and the player's retreat under
    the Lie Detector go by config.toml's)."""
    h = Hunter(Game({**kb.LIVE, **bindings}), mp_shown_at=1001.0)
    with pytest.raises(KeyCheckStop) as e:
        KeyCheck(h).run(_doubt())
    assert "her jump key ALT shows ATTACK" in str(e.value) and says in str(e.value) and h.rekeyed is None


def test_an_arrow_without_its_move_stops_the_hunt_whatever_attack_is_on(quiet):
    h = Hunter(Game({**kb.LIVE, "left": None}), mp_shown_at=1001.0)
    with pytest.raises(KeyCheckStop) as e:
        KeyCheck(h).run(_doubt())
    assert "her left key LEFT shows nothing" in str(e.value) and "MOVE LEFT is on no key" in str(e.value)


def test_attack_on_her_pick_up_key_is_switched_and_said(quiet):
    """Her pick-up presses attack then (no loot picked up, nothing worse): switched, said and paged."""
    h = Hunter(Game({**kb.LIVE, "ctrl": None, "z": 52, "a": 50}), mp_shown_at=1001.0)
    out = KeyCheck(h).run(_doubt())
    assert out.what == "switched" and out.key == "z" and "Z IS HER PICK-UP KEY TOO" in out.text
    assert "PICK UP IS ON A IN THE GAME" in out.text and "Z IS HER PICK-UP KEY TOO" in quiet[0][1]


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
    assert h.attacks.watch.holds(mon()) and not h.attacks.watch.holds(mon(x=900.0)) and not h.attacks.watch.held


def test_as_configured_its_targets_held_with_the_regular_attack_alone_too(quiet):
    """The review's probe: after "ATTACK IS ON CTRL AS CONFIGURED ... no more regular attacks at that horny-mushroom"
    in mode 'regular' her Actions keyed Ctrl at it themselves (not routed). Hunter.hold_at routes every throw's key
    through her attack choice, which holds it there (Lucky Seven)."""
    from maplehelper.config import GameKeys
    from maplehelper.engine import RIGHT
    from gamelab.behavior.attacks import Attacks
    from gamelab.behavior.hunter import Hunter as RealHunter
    from gamelab.control.actions import Actions
    g = Game()
    h = Hunter(g, mp_shown_at=1001.0)
    model = SimpleNamespace(mob_hp={"Horny Mushroom": 403.0}, mob_pdd={"Horny Mushroom": 10.0})
    keys = GameKeys(left=K("left"), right=K("right"), jump=K("alt"), attack=K("ctrl"), lucky_seven=K("shift"),
                    pick_up=K("z"), pet_food=K("u"), up=K("up"), down=K("down"))
    h.attacks = Attacks(model, keys, "regular", max_mp=623.0, level=16.0, log=h.lines.append)
    h.attacks.watch, h.attacks.mp = RegularWatch(), 0.5
    h.actions = Actions(keys, skill=False)
    h.actions.play = g.play
    h._attack_mode = "regular"
    for name in ("hold_regular", "hold_at", "regular_off", "rekey", "_route"):
        setattr(h, name, types.MethodType(getattr(RealHunter, name), h))
    d = _doubt()
    assert KeyCheck(h).run(d).what == "as configured"
    h.attacks.aimed = d.targets[0]
    assert h.actions.routed and h.actions._attack_key(RIGHT, ground=True) == "shift"


def test_the_window_not_opening_sends_no_esc_and_turns_the_regular_attack_off(quiet):
    g = Game(opens=False)
    h = Hunter(g, mp_shown_at=1001.0)
    out = KeyCheck(h).run(_doubt())
    assert [name for name, _ in g.plans] == ["key_bindings_mouse", "key_bindings"]  # (no Esc: the MENU would open)
    assert g.menu == 0 and out.what == "off" and "NOT READ (not open)" in out.text
    assert "if it's open in the game, close it with Esc or CANCEL" in quiet[0][1]  # (the backslash went out)


def test_a_window_opening_late_is_seen_at_the_last_frame_read_and_closed(quiet):
    """The review's case: opening later than SETTLE_S + AGAIN_S (1.0 s) after the backslash, it was left open with no
    Esc. LATE_S: one more frame 2 s after it."""
    g = Game(open_after=2)
    out = KeyCheck(Hunter(g, mp_shown_at=1001.0)).run(_doubt())
    assert out.what == "as configured" and not g.open and g.menu == 0
    assert [keys for _, keys in g.plans] == [[], ["\\"], ["esc"]]


@pytest.mark.parametrize("game, why", [
    (dict(blind=3), "no frame to tell by"),                    # open, but its frames fail their checks
    (dict(cover_title=True), "though the frame changed"),      # open, its title covered: not recognised
])
def test_a_window_the_check_can_t_tell_is_closed_stops_the_hunt(quiet, game, why):
    """The backslash sent and the window not seen closed: it may be open, so the hunt stops (KeyCheckStop, paged by
    retreat.on_alarm as every stop: "close it with Esc or CANCEL"); no Esc sent unseen (the MENU)."""
    g = Game(**game)
    h = Hunter(g, mp_shown_at=1001.0)
    with pytest.raises(KeyCheckStop) as e:
        KeyCheck(h).run(_doubt())
    assert e.value.window and "close it with Esc or CANCEL" in str(e.value)
    assert ["esc"] not in [keys for _, keys in g.plans] and g.menu == 0
    assert any(why in line for line in h.lines)
    assert e.value.page()[0] == "MapleStory: the key bindings window"


def test_a_window_dragged_partly_off_the_frame_is_closed(quiet):
    """The review's case: dragged past x 730 its whole title isn't in the frame; its name is (keybindings.find_part):
    open, not read, closed with Esc; the regular attack off (not read)."""
    g = Game(at=(900, 174))
    out = KeyCheck(Hunter(g, mp_shown_at=1001.0)).run(_doubt())
    assert out.what == "off" and "partly off the frame" in out.text and not g.open and g.menu == 0


def test_another_client_size_sends_nothing(quiet):
    """The review's case: a 1920x1080 client got the backslash and never Esc. Now nothing is sent at all."""
    g = Game(size=(1920, 1080))
    out = KeyCheck(Hunter(g, mp_shown_at=1001.0)).run(_doubt())
    assert g.plans == [] and not g.open and out.what == "off" and "1920x1080" in out.text


def test_open_already_it_s_read_without_the_backslash(quiet):
    g = Game(open_at_start=True)
    out = KeyCheck(Hunter(g, mp_shown_at=1001.0)).run(_doubt())
    assert [keys for _, keys in g.plans] == [[], ["esc"]] and out.what == "as configured" and not g.open


def test_a_second_esc_only_if_two_frames_after_the_first_still_show_it(quiet):
    g = Game(stuck=1)
    KeyCheck(Hunter(g, mp_shown_at=1001.0)).run(_doubt())
    assert [keys for _, keys in g.plans][-2:] == [["esc"], ["esc"]] and not g.open and g.menu == 0
    g = Game(slow=1)  # (closed, but the first frame after Esc still shows it: no second Esc, no MENU)
    h = Hunter(g, mp_shown_at=1001.0)
    KeyCheck(h).run(_doubt())
    assert [keys for _, keys in g.plans].count(["esc"]) == 1 and not g.open and g.menu == 0
    g = Game(stuck=5)
    h = Hunter(g, mp_shown_at=1001.0)
    with pytest.raises(KeyCheckStop):
        KeyCheck(h).run(_doubt())
    assert [keys for _, keys in g.plans].count(["esc"]) == 2 and g.open
    assert any("STILL OPEN after Esc twice" in line for line in h.lines)


def test_the_lie_detector_with_the_window_open_says_so_and_goes_on(quiet):
    g = Game(alarm_after=2)  # (the mouse, the backslash; then the player refuses)
    h = Hunter(g, mp_shown_at=1001.0)
    with pytest.raises(client.LieDetectorAlarm):
        KeyCheck(h).run(_doubt())
    assert g.open and any("close it with Esc or CANCEL" in t for _, t in quiet)


def test_a_monster_near_after_the_retreat_puts_it_off_then_a_quiet_look_raises_it_again(quiet, monkeypatch):
    """The review's case: the window opened with a monster on her (nothing looked again after the retreat). A monster
    still within 250 after it: no backslash, put off, her regular presses held; Hunter.health_check raises its doubt
    again at a quiet look DEFER_GAP_S later (RegularWatch.again); after DEFERS_MAX, off for the run."""
    g = Game()
    h = Hunter(g, mp_shown_at=1001.0)
    h.monsters, h.clear_after_retreat = [mon(x=100.0, y=0.0)], False
    check, d = KeyCheck(h), _doubt()
    out = check.run(d)
    assert out.what == "put off" and g.plans == [] and h.retreats == 1 and h.attacks.watch.held
    w = h.attacks.watch
    assert w.deferred is d and w.again(w.deferred_at + 1.0) is None and w.again(w.deferred_at + DEFER_GAP_S) is d
    for _ in range(DEFERS_MAX - 1):
        assert check.run(d).what == "put off"
    out = check.run(d)
    assert out.what == "off" and "NO QUIET MOMENT" in out.text and h.off and g.plans == []
    assert not w.checks  # (none ran: the window never opened)


def test_a_monster_up_close_after_her_stars_test_puts_it_off(quiet, monkeypatch):
    """The review's second case: one that walks up during her stars' test is near at the backslash. The look right
    before it (_quiet) sees it: one more retreat, still near, put off."""
    monkeypatch.setattr(keycheck, "QUIET_S", 0.0)
    g = Game()
    h = Hunter(g, mp_shown_at=1001.0)
    check = KeyCheck(h)
    check._stars = lambda d: (setattr(h, "monsters", [mon(x=20.0, y=0.0)]), setattr(h, "clear_after_retreat", False),
                              "shown")[-1]
    out = check.run(_doubt())
    assert out.what == "put off" and g.plans == [] and h.retreats == 1


def test_the_safety_stop_comes_at_a_look_of_the_check(quiet):
    h = Hunter(Game(), mp_shown_at=1001.0)
    stop = SafetyStop(0.4, 2.1, 3, 0.65)
    h.hp_watch = SimpleNamespace(due=lambda: stop)
    with pytest.raises(SafetyStop):
        KeyCheck(h).run(_doubt())
    assert h.game.plans == []


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


def _stars_hunter(game=None):
    from gamelab.attack import choice
    h = Hunter(game or Game(), mp_shown_at=None)
    h.attacks.skill, h.attacks.mp, h.darts.share = choice.lucky_seven(16), 0.5, 0.5
    return h


def test_her_stars_five_test_casts_without_mp_say_maybe_out_not_the_stop(quiet, monkeypatch):
    """The review's finding: live, with her stars, 7-8 casts in a row took no MP under the test's own conditions (a
    monster on her unseen). So 5 test casts that took no MP and hurt nothing say "maybe out", not OutOfDarts: ATTACK on
    Ctrl as configured then turns the regular attack off for the run (paged); out of darts stays its own watch's."""
    h = _stars_hunter()
    check = KeyCheck(h)
    keyed = []
    monkeypatch.setattr(check, "_cast_side", lambda s, d: ("right", mon()))
    monkeypatch.setattr(check, "_cast", lambda side: keyed.append(side) or True)
    monkeypatch.setattr(check, "_evidence", lambda key, m: None)
    out = check.run(_doubt())
    assert len(keyed) == 5 and out.stars == "maybe out" and out.what == "off" and h.off == "her stars may be out"
    assert "AND HER STARS MAY BE OUT" in out.text and len(quiet) == 1
    assert any("MAYBE OUT (5 test casts took no MP and hurt nothing" in line for line in h.lines)


def test_her_stars_test_out_of_darts_only_with_its_own_watch_due(quiet, monkeypatch):
    h = _stars_hunter()
    h.darts.shown = h.darts.presses  # (its stretch with the test's casts as long as the stop's)
    check = KeyCheck(h)
    monkeypatch.setattr(check, "_cast_side", lambda s, d: ("right", mon()))
    monkeypatch.setattr(check, "_cast", lambda side: True)
    monkeypatch.setattr(check, "_evidence", lambda key, m: None)
    with pytest.raises(OutOfDarts):
        check.run(_doubt())
    assert h.game.plans == []  # (the window never opened)


def test_her_stars_a_cast_that_took_mp_or_hurt_its_target_says_she_has_them(quiet, monkeypatch):
    h = _stars_hunter()
    check = KeyCheck(h)
    monkeypatch.setattr(check, "_cast_side", lambda s, d: ("right", mon()))
    monkeypatch.setattr(check, "_cast", lambda side: True)
    shown = iter([None, "lowered its target's bar"])
    monkeypatch.setattr(check, "_evidence", lambda key, m: next(shown))
    assert check.run(_doubt()).stars == "test"


def test_her_stars_test_ends_at_a_touch(quiet, monkeypatch):
    """The review's case: a monster on her back swiped at every cast for 8 s. A touch since the test began ends it
    (not known)."""
    h = _stars_hunter()
    check = KeyCheck(h)
    monkeypatch.setattr(check, "_cast_side", lambda s, d: ("right", mon()))

    def cast(side):
        h.hit_at = time.monotonic() + 0.01
        return True
    monkeypatch.setattr(check, "_cast", cast)
    monkeypatch.setattr(check, "_evidence", lambda key, m: None)
    out = check.run(_doubt())
    assert out.stars == "not known" and any("touched during the test, after 1 cast" in line for line in h.lines)


def test_a_test_cast_goes_with_no_monster_up_close_either_way_nor_at_a_rope_s_top():
    h = _stars_hunter()
    far = mon(x=300.0, y=0.0)
    h.attacks.target = lambda side: far if side == "right" else None
    check = KeyCheck(h)
    p = SimpleNamespace(id=23)
    model = scenarios.real_map(scenarios.TREE)
    plat = model.platform(23) if model is not None else p

    def look(monsters, x=0.0):
        return SimpleNamespace(t=0.0, where=SimpleNamespace(x=x, y=0.0), monsters=monsters, platform=plat)
    behind = mon(x=-30.0, y=0.0)  # (on her back, the other way)
    assert check._cast_side(look([far, behind]), _doubt()) == "a monster up close"
    h.darts.at_top = True
    assert check._cast_side(look([far]), _doubt()) == "not on a platform"


def test_key_doubt_isn_t_a_hunt_stop_key_check_stop_is():
    from gamelab.behavior.safety import HuntStop
    assert not issubclass(KeyDoubt, HuntStop) and "TOOK NO EFFECT" in str(KeyDoubt(_doubt()))
    assert issubclass(KeyCheckStop, HuntStop) and KeyCheckStop.exit_code == 6


def test_held_a_regular_throw_is_lucky_seven_with_the_mp_for_a_cast_else_none():
    """behavior.attacks.Attacks.key with the watch holding the regular attack: Lucky Seven at a target while she has the
    MP for a cast, no throw (None) without it or with nothing in its way (no cast spent at nothing: the review), the
    regular attack again once let go; with the regular attack alone her Actions' own presses are told to the watch at
    their throw (on_throw), their target as a throw that way takes."""
    from maplehelper.config import GameKeys
    from maplehelper.engine import RIGHT
    from gamelab.behavior.attacks import Attacks
    model = SimpleNamespace(mob_hp={"Horny Mushroom": 403.0}, mob_pdd={"Horny Mushroom": 10.0})
    keys = GameKeys(left=K("left"), right=K("right"), jump=K("alt"), attack=K("ctrl"), lucky_seven=K("shift"))
    lines = []
    a = Attacks(model, keys, "regular", max_mp=623.0, level=16.0, log=lines.append)
    a.watch = RegularWatch(held=True)
    a.mp = 0.5
    target = mon(kind="horny-mushroom", x=200.0)
    a.aimed = target
    assert a.key(RIGHT) == "shift" and "the regular attack held (its key doubted)" in lines[-1]
    a.on_throw(1.0, RIGHT)
    assert a.key(RIGHT) is None and "nothing is in its way (no Lucky Seven cast at nothing)" in lines[-1]
    a.watch.off = "ATTACK on no key"
    a.aimed = target
    assert a.key(RIGHT) == "shift" and "the regular attack off this run" in lines[-1]
    a.mp, a.aimed = 0.01, target
    assert a.key(RIGHT) is None and "no throw: the regular attack is off this run" in lines[-1]
    a.watch.held, a.watch.off = False, None
    assert a.key(RIGHT) == "ctrl"
    a.on_throw(2.0, RIGHT)
    a.on_throw(3.0, RIGHT)  # (not asked: the regular attack alone, her Actions' own key)
    assert a.watch.counts["passed: nothing in its way"] == 2


def test_hunter_health_check_raises_a_put_off_check_again_at_a_quiet_look_only():
    """behavior.hunter.Hunter.health_check: a check put off (RegularWatch.deferred) comes due again as KeyDoubt at the
    first look DEFER_GAP_S or more later that keycheck.unquiet passes (a monster near: not then), in the hunt loop only
    (`doubts_raise`)."""
    from gamelab.behavior.hunter import Hunter as RealHunter
    h = SimpleNamespace(drink_if_low=lambda s: None, hp_watch=SimpleNamespace(due=lambda: None), darts=DartWatch(),
                        attacks=SimpleNamespace(watch=RegularWatch()), doubts_raise=True, log=lambda line: None,
                        on_rope=lambda s: False, knocked=lambda s: False, _doubted=None)
    w, d = h.attacks.watch, _doubt()
    w.deferred, w.deferred_at = d, time.monotonic() - DEFER_GAP_S - 1
    place = SimpleNamespace(id=23)

    def look(monsters):
        return SimpleNamespace(t=time.monotonic(), where=SimpleNamespace(x=0.0, y=0.0), monsters=monsters,
                               platform=place)
    check = types.MethodType(RealHunter.health_check, h)
    check(look([mon(x=50.0, y=0.0)]))  # (a monster near: not yet)
    assert w.deferred is d
    h.doubts_raise = False
    check(look([]))  # (not the hunt loop: not raised)
    assert w.deferred is d
    h.doubts_raise = True
    with pytest.raises(KeyDoubt) as e:
        check(look([]))
    assert e.value.doubt is d and w.deferred is None


# --- `gamelab keys` (the live check's tool) -----------------------------------------------------------------------

def test_keys_says_where_attack_is_only_for_a_window_read(capsys):
    """The review's case: a window not read (the Controller tab, dragged off) printed "ATTACK IS NOT ON CTRL ... on no
    key", which could have the user rebind a key that's fine."""
    from gamelab.__main__ import _say_keys
    _say_keys(kb.Reading(True, "the Controller tab", (177, 174)), {}, "ctrl")
    out = capsys.readouterr().out
    assert "not read: the Controller tab" in out and "ATTACK IS NOT ON" not in out and "isn't known" in out
    _say_keys(kb.Reader().read(kb.compose({**kb.LIVE, "ctrl": None, "a": 52}, 2)), {}, "ctrl")
    assert "ATTACK IS NOT ON CTRL (config.toml [keys] attack): the game has it A" in capsys.readouterr().out


@pytest.mark.parametrize("game, frames_after_esc, want, escs", [
    (dict(), 18, "read and closed (Esc x1)", 1),
    (dict(stuck=1), 18, "read and closed (Esc x2)", 2),
    (dict(stuck=1), 0, "not known whether it closed (no frames after Esc x1)", 1),   # (no Esc unseen: the MENU)
    (dict(stuck=5), 18, "STILL OPEN after two Esc", 2),
    (dict(open_after=30), 18, "the window wasn't seen open", 0),
])
def test_keys_record_sends_a_second_esc_only_with_the_window_seen_still_open(monkeypatch, tmp_path, capsys, game,
                                                                              frames_after_esc, want, escs):
    import gamelab.__main__ as cli
    g = Game(**game)
    monkeypatch.setattr(cli, "SHOTS", tmp_path)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)

    def submit(events, fps=0, duration_ms=0, name="", **kw):
        g.play(events, name)
        n = int(duration_ms / 1000 * fps) if not (name.startswith("keybindings_close") and not frames_after_esc) else 0
        return {"frames": [g.capture() for _ in range(n)], "times": None}
    monkeypatch.setattr(cli.client, "submit", submit)
    hunter = SimpleNamespace(senses=SimpleNamespace(eyes=SimpleNamespace(_capture=g.capture)))
    cli._keys_recorded(hunter, kb.Reader(), "ctrl")
    out = capsys.readouterr().out
    assert want in out, out
    assert [keys for _, keys in g.plans].count(["esc"]) == escs and g.menu == 0
