"""Out of darts (behavior.darts): her Lucky Seven presses taking no MP and killing nothing, more in a row than a monster
standing on her takes of her claw's swipes, end the hunt as the safety stop does: she goes to a safe place, the log
says so in capitals, whoever is at the PC is paged, the recording's last minutes are saved, the command exits with its
own code. The live hunts of 2026-10-25 23:43 to 2026-10-26 01:28 replayed through it (tests/fixtures/
darts_live_2026-10-26.json): no stop while she had her darts; the presses after 01:28:17 counted. In the simulator, end
to end: loop_tree_out_of_darts (tests/test_sim.py runs every scenario)."""

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from gamelab.attack.choice import MP_STEPS
from gamelab.behavior import darts, retreat
from gamelab.behavior.darts import EXIT, PRESSES, READ_EVERY, DartWatch, OutOfDarts, presses_for, swipes
from gamelab.behavior.hunter import Hunter
from gamelab.behavior.safety import HpWatch, HuntStop
from gamelab.control import alarm
from gamelab.perception.loot import says_full, says_no_darts

FIXTURES = Path(__file__).parent / "fixtures"
TREE_HP = {"Horny Mushroom": 403.0, "Green Mushroom": 233.0, "Slime": 115.0}  # Tree Dungeon's, the game data's
TREE_PDD = {"Horny Mushroom": 10.0, "Green Mushroom": 0.0, "Slime": 10.0}


def _px(px: int) -> float:
    return px / MP_STEPS


def _stretch(w: DartWatch, n: int, t0: float = 0.0, px: int = 40, every: float = 0.9) -> float:
    """`n` presses `every` s apart from t0 with her bar at `px` throughout (a read after each): the time after."""
    t = t0
    for _ in range(n):
        w.cast(t)
        t += every
        w.see(t, _px(px))
    w.see(t + 0.6, _px(px))
    return t + 0.6


def test_presses_for_the_map_tree_dungeon_s_20_more_where_its_toughest_monster_takes_more_swipes():
    # Tree Dungeon's 20: twice and more the 9 in a row with her darts live. Its toughest, a Horny Mushroom (403 HP,
    # defence 10), takes 21.1 presses of her swipes (21 a press, x 100/110), a Green Mushroom (233) 11.1.
    assert presses_for(TREE_HP, TREE_PDD) == PRESSES == 20
    assert swipes(403.0, 10.0) == pytest.approx(21.11, abs=0.01) and swipes(233.0) == pytest.approx(11.1, abs=0.01)
    assert presses_for({"Snail": 15.0}, {}) == presses_for({}, {}) == PRESSES  # weak monsters: as many
    # West Rocky Mountain II's Axe Stump (371 HP, defence 30) takes 23.0: 20 x 23.0 / 21.1, 22.
    assert presses_for({"Axe Stump": 371.0, "Stump": 133.0}, {"Axe Stump": 30.0, "Stump": 30.0}) == 22
    assert (EXIT, READ_EVERY) == (5, 12) and issubclass(OutOfDarts, HuntStop)


def test_a_stretch_of_presses_that_took_no_mp_stops_the_hunt_once():
    w = DartWatch(20)
    w.see(-1.0, _px(40))
    t = _stretch(w, 19)
    assert w.shown == 19 and w.due(t) is None
    t = _stretch(w, 1, t)
    stop = w.due(t)
    assert isinstance(stop, OutOfDarts) and stop.presses == 20 and stop.mp == pytest.approx(_px(40))
    assert str(stop).startswith("20 Lucky Seven presses over 1") and "took no MP and killed nothing" in str(stop)
    assert "out of throwing stars" in str(stop) and not stop.told
    title, text = stop.page()
    assert title == "MapleStory: out of darts" and "safe place" in text and "throwing stars" in text
    _stretch(w, 5, t)
    assert w.due() is None and w.fired  # once
    # A press keyed under MP_SHOWS_S before the last read isn't counted yet: its cast may not show.
    w = DartWatch(3)
    w.see(0.0, _px(40))
    for t in (0.1, 1.0, 1.9):
        w.cast(t)
    w.see(1.2, _px(40))
    w.see(2.2, _px(40))
    assert w.shown == 2 and w.due() is None
    w.see(2.5, _px(40))
    assert w.shown == 3 and w.due() is not None


def test_a_cast_a_drink_a_kill_or_a_low_read_starts_it_again_a_gap_drops_what_no_look_saw():
    def watched(*steps):
        """4 presses, `steps` (each given the time, giving the time after it), 3 presses."""
        w = DartWatch(5)
        w.see(-1.0, _px(40))
        t = _stretch(w, 4)
        for step in steps:
            t = step(w, t) or t + 0.1
        return w, _stretch(w, 3, t + 0.1)
    # Without: 7 presses in a row, the stop due.
    w, t = watched()
    assert w.shown == 7 and w.due(t) is not None
    # Her bar falls 2 px (a cast's MP shown): it went out, the presses before don't count.
    w, t = watched(lambda w, t: w.see(t + 0.1, _px(38)))
    assert w.shown == 3 and w.due(t) is None
    # ... a px is a flicker, or a regen tick and a cast at most: it goes on.
    w, t = watched(lambda w, t: w.see(t + 0.1, _px(39)))
    assert w.shown == 7
    # Her regen raises the highest read: a fall from there is a cast.
    w, t = watched(lambda w, t: w.see(t + 0.1, _px(41)), lambda w, t: w.see(t + 0.2, _px(39)))
    assert w.shown == 3
    # A drink of her pet's (could hide casts), a kill of hers, her MP too low to tell by.
    for step in (lambda w, t: w.see(t + 0.1, _px(57)), lambda w, t: w.kill(t + 0.1),
                 lambda w, t: w.see(t + 0.1, _px(4))):
        w, t = watched(step)
        assert w.shown <= 3 and w.due(t) is None
    # A gap in the reads (looks failing: the game not in front?) drops the presses keyed since the last read (no look
    # saw the game take them), not those before it, nor its highest read: a trip between fights doesn't end it.
    w, t = watched(lambda w, t: (w.cast(t + 0.1), w.see(t + 2.8, _px(40)))[1] or t + 2.8)
    assert w.shown == 7 and w.due(t) is not None
    w, t = watched(lambda w, t: (w.cast(t + 0.1), w.see(t + 2.8, _px(38)))[1] or t + 2.8)  # a cast in it: shown
    assert w.shown == 3
    # A read of 0 (her pet's warning) ends it too (the next read begins the next); one not read (None) changes nothing.
    w, t = watched(lambda w, t: w.see(t + 0.1, 0.0))
    assert w.shown == 2
    w, t = watched(lambda w, t: w.see(t + 0.1, None))
    assert w.shown == 7


def test_the_pickup_log_is_read_on_a_long_stretch_and_its_word_stops_the_hunt_sooner():
    w = DartWatch(32)
    w.see(-1.0, _px(40))
    t = _stretch(w, READ_EVERY - 1)
    assert not w.asks()
    t = _stretch(w, 1, t)
    assert w.asks()
    w.hear("You received EXP (+41)\nBonus EXP for PARTY (+2)")
    assert not w.asks() and w.due(t) is None  # it says nothing of it: on to the stretch's 32
    t = _stretch(w, READ_EVERY, t)
    assert w.asks()
    w.hear("You don't have enough throwing stars.")
    stop = w.due(t)
    assert stop is not None and stop.told and stop.presses == 2 * READ_EVERY
    assert str(stop).startswith("the game says she's out of throwing stars (You don't have enough throwing stars.)")
    # Its word comes with READ_EVERY presses at least: a kill since starts the count again.
    w = DartWatch(32)
    w.see(-1.0, _px(40))
    t = _stretch(w, READ_EVERY)
    w.hear("Not enough throwing stars")
    w.kill(t)
    assert w.told is None and w.due(t) is None


def test_the_game_s_words_for_no_stars_and_none_of_the_pickup_logs_lines():
    for said in ("You don't have enough throwing stars.", "Not enough Throwing Stars", "You have no stars left",
                 "No more darts.", "You are out of throwing stars", "Ran out of stars", "Throwing stars are used up",
                 "Lucky Seven requires throwing stars", "You do not have any throwing stars", "NotenoughThrowingStars"):
        assert says_no_darts(said), said
    for said in ("Subi Throwing Stars x1 earned. (Use)", "You received EXP (+41)", "Bonus EXP for PARTY (+2)",
                 "You have gained mesos (+20)", "Not enough MP", "Horny Mushroo... x1 earned. (Etc)",
                 "Go to a safe place before the", "You can't get anymore items."):
        assert not says_no_darts(said), said
    # The pickup log of the live hunts (the Lie Detector recordings' frames, a second apart, her darts with her): every
    # line OCR read there, none of them saying so (nor that her inventory was full).
    kept = json.loads((FIXTURES / "pickup_log_lines_2026-10-26.json").read_text(encoding="utf-8"))
    assert kept["frames"] == 1413 and len(kept["lines"]) == 739
    assert not [line for line in kept["lines"] if says_no_darts(line) or says_full(line)]


def _live():
    """The five live hunts' events (the fixture's): run -> ([t, kind, px]...), its `darts_out` (or None)."""
    runs = json.loads((FIXTURES / "darts_live_2026-10-26.json").read_text(encoding="utf-8"))["runs"]
    out = {}
    for name, r in runs.items():
        t, evs = 0.0, []
        for row in r["events"]:
            t += row[0] / 100
            evs.append((t, row[1], row[2] if len(row) > 2 else None))
        out[name] = (evs, r.get("darts_out"))
    return out


def _replay(evs, presses, until=None):
    """Events through a DartWatch as the hunter feeds it: the first stop (its time and presses), the longest stretch."""
    w = DartWatch(presses)
    for t, kind, px in evs:
        if until is not None and t > until:
            break
        if kind == "c":
            w.cast(t)
        elif kind == "k":
            w.kill(t)
        else:
            w.see(t, None if px is None else _px(px))
            if (stop := w.due(t)) is not None:
                return (round(t, 1), stop.presses), w.longest
    return None, w.longest


def test_the_live_hunts_never_stop_while_she_had_her_darts():
    """Five hunts, 79 minutes, 2,293 presses with her darts: 9 in a row at the most took no MP (her claw swiping a
    Green Mushroom standing on her: the module's doc), so no stop at Tree Dungeon's 20, nor at 12."""
    live = _live()
    evs, out = live["01:04"]
    assert sum(1 for evs, _ in live.values() for e in evs if e[1] == "c") == 2316
    assert sum(1 for e in evs if e[1] == "c" and e[0] > out) == 23  # (her presses after her last star)
    for name, (evs, out) in live.items():
        for presses in (presses_for(TREE_HP, TREE_PDD), 12):
            stop, longest = _replay(evs, presses, until=out)
            assert stop is None and longest[0] <= 9, (name, presses, longest)


def test_the_presses_after_her_last_star_at_01_28_are_counted():
    """The grind of 01:04: her last cast that took MP shown at 01:28:17; 22 presses after it took none over 20 s until
    she went for her drops and the user stopped the hunt (01:28:47). At Tree Dungeon's 20 the stop comes 19.4 s after
    it (01:28:36.4, 10 s before the user's stop); at 16, 15 s; at 24 or 32 none before she stopped pressing."""
    evs, out = _live()["01:04"]
    stop, longest = _replay(evs, presses_for(TREE_HP, TREE_PDD))
    assert stop[1] == 20 and stop[0] - out == pytest.approx(19.4, abs=0.1)  # (01:28:36.4)
    assert _replay(evs, 16)[0][0] - out == pytest.approx(15.2, abs=0.1)
    for presses in (24, 32):
        stop, longest = _replay(evs, presses)
        assert stop is None and longest[0] == 22 and longest[1] == pytest.approx(19.8, abs=0.1)


def _hunter(said="You received EXP (+26)"):
    h = object.__new__(Hunter)
    h.drink_if_low, h.hp_watch, h.darts = (lambda s: None), HpWatch(), DartWatch(READ_EVERY + 2)
    h.logged, h.reads = [], []
    h.log = h.logged.append

    def messages():
        h.reads.append(said)
        return said
    h.senses = SimpleNamespace(eyes=SimpleNamespace(messages=messages))
    return h


def test_her_health_check_reads_the_pickup_log_on_a_long_stretch_and_raises_the_stop_with_it():
    h = _hunter()
    h.darts.see(-1.0, _px(40))
    t = _stretch(h.darts, READ_EVERY)
    s = SimpleNamespace(t=t, hp=1.0)
    assert h.health_check(s) is None and len(h.reads) == 1
    assert h.logged == [f"  {READ_EVERY} Lucky Seven presses in a row took no MP: the pickup log reads "
                        "You received EXP (+26)"]
    _stretch(h.darts, 2, t)
    with pytest.raises(OutOfDarts) as e:
        h.health_check(s)
    assert len(h.reads) == 2 and e.value.said == "You received EXP (+26)" and not e.value.told
    assert str(e.value).endswith("; the pickup log then: You received EXP (+26)")
    assert h.health_check(s) is None  # once
    # The game's word for it: the stop at the read.
    h = _hunter("Not enough throwing stars.")
    h.darts.see(-1.0, _px(40))
    _stretch(h.darts, READ_EVERY)
    with pytest.raises(OutOfDarts) as e:
        h.health_check(s)
    assert e.value.told and len(h.reads) == 1
    # A read that fails is said, and taken as nothing.
    h = _hunter()

    def broken():
        raise RuntimeError("no frame")
    h.senses = SimpleNamespace(eyes=SimpleNamespace(messages=broken))
    assert h._messages() == "" and h.logged == ["  the pickup log not read (RuntimeError: no frame)"]


def test_her_casts_and_her_bar_reach_the_watch_through_her_attack_choice():
    from gamelab.behavior.attacks import Attacks
    a = object.__new__(Attacks)
    a.darts, a._cast, a.mp, a._casts, a._read, a._warned = DartWatch(), 0.02, 0.5, [], None, False
    a.counts, a._keyed, a._swipe = Counter(), None, False  # (no throw at a target chosen: on_throw's pending time)
    a._ground = True
    a.on_throw(1.0, "left")
    assert a.darts.keys == [1.0]
    a.on_throw(2.0, "left")  # the regular attack (no cast chosen): not a press of Lucky Seven
    assert a.darts.keys == [1.0]
    a._cast, a._swipe = 0.02, True  # from the ground at a monster the claw reaches: no MP whatever her darts
    a.on_throw(2.5, "left")
    assert a.darts.keys == [1.0] and a.darts.swiped_presses == 1
    a._cast, a._swipe, a._ground = 0.02, False, False  # a jump throw's, keyed in the air (Actions.keyed_ground)
    a.on_throw(2.7, "left")
    assert a.darts.keys == [1.0, 2.7] and a.darts.aired == [2.7]
    a.glance(3.0, 0.4)
    assert a.darts.read_at == 3.0 and a.darts.top == round(0.4 * MP_STEPS)


def test_at_the_stop_she_retreats_she_is_paged_and_the_recording_is_kept(monkeypatch):
    said, paged, went, kept = [], [], [], []
    h = SimpleNamespace(log=said.append)

    def away(hunter, seen, why, log=None, rng=None):
        went.append(why)
        return retreat.Retreat(0.0, result="arrived")
    monkeypatch.setattr(retreat, "away", away)
    monkeypatch.setattr(retreat.alarm, "page", lambda title, text: paged.append(title))
    monkeypatch.setattr(retreat.alarm, "keep_recent", lambda s, name: kept.append((s, name)) or "kept: lab/recent/x")
    with pytest.raises(OutOfDarts):
        with retreat.on_alarm(h) as watch:
            raise OutOfDarts(32, 28.4, 0.35, said="")
    assert went == [retreat.Why("OUT OF DARTS", "Out of darts", "after the stop")]
    assert said[0].startswith("OUT OF DARTS: 32 Lucky Seven presses over 28 s took no MP and killed nothing")
    assert said[0].endswith("The hunt ends: to a safe place (no more throws)")
    assert paged == ["MapleStory: out of darts"] and kept == [(darts.KEEP_S, "out-of-darts")]
    assert said[-1] == "OUT OF DARTS: kept: lab/recent/x" and watch.kept == "kept: lab/recent/x"


def test_keeping_the_recording_never_raises(monkeypatch):
    from gamelab.control import rolling

    def none_running(seconds, name):
        raise rolling.NotRunning("no recorder is running")
    monkeypatch.setattr(rolling, "ask", none_running)
    assert alarm.keep_recent(90, "out-of-darts") == "the recording not saved (NotRunning: no recorder is running)"
    monkeypatch.setattr(rolling, "ask", lambda seconds, name: {"error": "busy"})
    assert alarm.keep_recent(90, "out-of-darts") == "the recording not saved: busy"
    monkeypatch.setattr(rolling, "ask", lambda seconds, name: {"seconds": 90.0, "folder": "lab/recent/a-out-of-darts"})
    assert alarm.keep_recent(90, "out-of-darts") == "the recording's last 90 s saved: lab/recent/a-out-of-darts"


def test_a_hunting_command_stopped_by_it_exits_with_its_own_code(monkeypatch, capsys):
    from gamelab import __main__ as cli

    def tour(args):
        raise OutOfDarts(32, 28.4, 0.35)
    monkeypatch.setattr(cli, "cmd_tour", tour)
    assert cli.main(["tour", "--loop"]) == EXIT
    out = capsys.readouterr().out
    assert out.startswith("OUT OF DARTS: 32 Lucky Seven presses over 28 s") and "Give her throwing stars" in out


def test_a_press_from_the_ground_at_a_monster_the_claw_reaches_is_no_press_for_out_of_darts():
    # Up close the Lucky Seven key swipes as the attack key does (no MP whatever her darts): the attack choice tells
    # the watch so (Attacks._clawed, from where she stood in the last look; Actions.keyed_ground), and a Horny Mushroom
    # pinning her at a wall's end (21 swipes from whole) no longer stops the hunt with her stars in her bag.
    from gamelab.behavior.attacks import Attacks
    from gamelab.localization.monsters import Monster
    from gamelab.perception.mobs import Mob
    a = object.__new__(Attacks)
    a._last = SimpleNamespace(where=SimpleNamespace(x=500.0, y=500.0))
    on_her, far = (Monster(Mob("horny-mushroom", 0, 0, 30, 30, 0.9), x, 500.0, 0) for x in (530.0, 700.0))
    a.from_ground = lambda: True
    assert a._clawed(on_her) and not a._clawed(far)
    a.from_ground = lambda: False  # a jump throw's: its star is a cast's, MP and all
    assert not a._clawed(on_her)
    watch = DartWatch()
    for i in range(PRESSES + 5):
        watch.cast(float(i), swipe=True)
    assert watch.keys == [] and watch.swiped_presses == PRESSES + 5 and "the claw reaches: not counted" in watch.summary()


def test_presses_within_touched_s_of_a_touch_are_no_presses_for_out_of_darts():
    # The simulator's thief seed 6 of the pile-up study (e040fb9, 241 s): a Horny Mushroom on her at platform 19's right
    # end, lost under her own swipes' damage numbers and never seen, touched her every 1.7 s while every press swiped it
    # (0.85 s apart, no MP): 21 presses over 17 s stopped the hunt with her stars in her bag. A touch, told a look late,
    # takes the presses keyed since it off the stretch and keeps those to come off it for TOUCHED_S.
    from gamelab.behavior.darts import TOUCHED_S
    w, t, touch = DartWatch(PRESSES), 0.0, 0.3
    for _ in range(3 * PRESSES):
        w.cast(t)
        t += 0.85
        if t >= touch + 0.4:  # (told at the next look after it)
            w.touched(touch)
            touch += 1.7
        w.see(t, _px(40))
        assert w.due(t) is None
    assert w.longest[0] <= 1 and w.touched_presses >= 3 * PRESSES - 1
    assert f"within {TOUCHED_S:g} s of a touch of hers: not counted" in w.summary()
    # ... and once the touches stop (the one on her dead, or gone), her presses count again: out of darts stops it
    w.see(t + TOUCHED_S, _px(40))
    stop = None
    t = _stretch(w, PRESSES, t0=t + TOUCHED_S + 0.1)
    stop = w.due(t)
    assert stop is not None and stop.presses == PRESSES


def test_a_touch_of_any_kind_reaches_out_of_darts_watch_from_the_hunter():
    from gamelab.behavior.fight import Report
    from gamelab.perception.touches import DAMAGE, MISS, Touch
    h = Hunter.__new__(Hunter)  # (only what _touch_told uses)
    h.hit_at = h.hit_kind = h.knock_at = h.touch_at = h._last_touch = None
    h.report, h.log, h._last_where = Report(), (lambda text: None), None
    h.actions = SimpleNamespace(pressed=True)
    h.attacks = SimpleNamespace()
    h.darts = DartWatch(PRESSES)
    h.darts.keys = [9.0, 9.9, 10.4, 10.9]
    h._touch_told(SimpleNamespace(t=10.5, hp=0.9, where=None), Touch(10.0, MISS, MISS).data())
    assert h.darts.touched_at == 10.0 and h.darts.keys == [9.0, 9.9] and h.darts.touched_presses == 2
    h.darts.cast(11.95, ground=False)  # (a jump throw's, keyed in the air: a cast's whatever is on her)
    h._touch_told(SimpleNamespace(t=12.0, hp=0.8, where=None), Touch(11.9, DAMAGE, "54").data())
    assert h.darts.touched_at == 11.9 and h.darts.keys == [9.0, 9.9, 11.95] and h.darts.touched_presses == 2
    h.darts.cast(12.5, ground=False)
    h.darts.cast(12.9)
    assert h.darts.keys == [9.0, 9.9, 11.95, 12.5] and h.darts.touched_presses == 3
