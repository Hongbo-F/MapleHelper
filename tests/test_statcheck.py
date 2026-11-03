"""Her stat window opened, read and closed at a hunt's start (behavior.statcheck), in the simulator with the window drawn
from the game data's pictures (sim.world: Realism.stat_key, stat_detail, stat_stuck; sim.fakes.FakeEyes._capture) and
read as in game but for the OCR (FakeEyes.stat_ocr): the flow (the quiet moment, S, DETAIL, the read, S to close, Esc
only with it still open), its refusals (not calibrated, no safe key, another client size, not due), what stops it (the
window may be open, the Lie Detector, the safety stop), and what the read gives her stats and her reserve (the user,
2026-10-28 ~18:20: "You can open character stats panel to check the defense")."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from maplehelper.config import GameKeys
from maplehelper.keys import parse as parse_key

from gamelab import paths
from gamelab.behavior import statcheck
from gamelab.behavior.bootstrap import SESSION_S, HerStats
from gamelab.behavior.statcheck import StatCheck, StatWindowStop, stats_key
from gamelab.control import client
from gamelab.perception import stat_window as sw
from gamelab.sim import harness, scenarios
from gamelab.sim.world import LEVEL_25, Realism

TREE = scenarios.TREE
REAL = pytest.mark.skipif(not scenarios.available(TREE), reason="needs Tree Dungeon's game data (gamedata/data)")
QUIET = set(range(0, 17))  # (the map's monsters on 0-16 only: nothing near her on 23 at the start)


def _sc(fn, start=(23, 73.0), monsters=QUIET, seed=1, **realism):
    return scenarios.Scenario("stat_test", fn, start, monsters=monsters, seed=seed, limit=200, map_id=TREE,
                              attack="choice", her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER),
                              realism=Realism(zero_walls=True, **realism))


def _start(calibrated=True, before=None, start=(23, 73.0), monsters=QUIET, **realism):
    """A hunt's start in the simulator (Hunter.read_stats at_start, as gamelab.__main__'s _hunter): the outcome or the
    stop, the sim, the hunter, the stat window's plans' keys and the trace's log."""
    got = {}

    def fn(sim):
        if calibrated:
            scenarios._stat_calibrated(sim)
        h = sim.tour_stack().h
        got["h"] = h
        if before is not None:
            before(sim, h)
        try:
            got["out"] = h.read_stats(at_start=True)
        except BaseException as e:  # (the stop, the Lie Detector)
            got["raised"] = e
        sim.world.advance(sim.clock.now)
        got["keys"] = [k for _, name, events, _ in sim.trace.plans if name.startswith("stat_window")
                       for k in (e[1] for e in events if len(e) == 3 and e[2] == "down")]
        got["log"] = list(sim.trace.log)
        got["world"] = sim.world.summary()
        got["pages"] = list(sim.player.pages)
        return {}
    scenarios.run_sim(_sc(fn, start=start, monsters=monsters, **realism))
    return got


@REAL
def test_s_detail_read_s_and_her_def_from_it():
    """The read at a quiet start: S, the window seen, a click on DETAIL, its panel seen, read, S, closed; her DEF the
    window's (the world's 75), kept with her level and the run's time; her reserve made again with it (453 -> 393: 3
    Horny Mushroom touches of 131 at DEF 75, level 25), said after the read's line."""
    got = _start()
    out, h, log = got["out"], got["h"], got["log"]
    assert out.what == "read" and out.sent == ("mouse", "s", "click DETAIL", "s") and got["keys"] == ["s", "lbutton", "s"]
    assert out.reading.wdef == 75 and out.reading.level == 25 and out.reading.max_hp == 774
    assert h.stats.defence.source == "stat window" and h.stats.defence.low == 75.0
    assert h.stats.window_read["level"] == 25 and h.stats.window_read["read"]["acc"] == 85
    w = got["world"]
    assert w["stat_windows"] == 1 and not w["stat_window_open"] and w.get("menu_opened", 0) == 0
    read = next(i for i, line in enumerate(log) if "her stat window: W.DEF 75, level 25, HP 774/774" in line)
    assert "the safety stop now under 393 HP of 774 (51%: 3 Horny Mushroom touches of 131) (was 453 HP" in log[read + 1]
    assert h.hp_watch.reserve.limit(h.max_hp).startswith("under 393 HP of 774")
    assert not got["pages"]
    # Kept in her stats file: a hunt started again this session doesn't open it again (said once)
    kept = json.loads(paths.STATS.read_text(encoding="utf-8")) if paths.STATS.is_file() else None
    assert kept is None or kept["window"]["low"] == 75.0


@REAL
def test_detail_open_with_it_no_click():
    got = _start(stat_detail=True)
    assert got["out"].what == "read" and got["keys"] == ["s", "s"] and got["out"].reading.wdef == 75


@REAL
def test_it_didnt_open_nothing_more_sent():
    """Her stats key does nothing (Realism.stat_key None: ABILITY on no key in the game): the frame unchanged where it
    opens, so it didn't open: no Esc (with nothing open Esc opens the game's MENU), her DEF as it was."""
    got = _start(stat_key=None)
    assert got["out"].what == "not opened" and got["keys"] == ["s"]
    assert any("it didn't open after S (nothing changed where it opens): not read, nothing more sent" in line
               for line in got["log"])
    assert got["h"].stats.defence.source == "prior" and got["world"].get("menu_opened", 0) == 0


@REAL
def test_a_window_that_wont_close_stops_the_hunt():
    """The window still open after S and Esc twice (Realism.stat_stuck): it may be open, StatWindowStop (her retreat in
    a hunt; at a start nothing is hunted), said in capitals, exit code 7; Esc only with it seen open."""
    got = _start(stat_stuck=True)
    stop = got["raised"]
    assert isinstance(stop, StatWindowStop) and stop.exit_code == statcheck.EXIT == 7
    assert str(stop).startswith("her stat window may be open in the game (her stats key sent, it wasn't seen closed): "
                                "close it with S or Esc; it read W.DEF 75")
    assert got["keys"] == ["s", "lbutton", "s", "esc", "esc"]
    assert any("her stat window: STILL OPEN after S and Esc twice" in line for line in got["log"])
    assert stop.page()[0] == "MapleStory: her stat window" and stop.after == "Nothing was hunted."
    assert got["pages"] and got["pages"][-1][1] == "MapleStory: her stat window"  # (at a start: paged by it)


@REAL
def test_not_calibrated_nothing_sent_said_once():
    got = _start(calibrated=False)
    assert got["out"].what == "skipped" and got["keys"] == []
    said = [line for line in got["log"] if "her stat window not read" in line]
    assert len(said) == 1 and "its reader not calibrated on this client yet: run `python -m gamelab stats --read " \
                               "--record` once" in said[0]
    h = got["h"]
    assert h.read_stats(at_start=True).what == "skipped" and h.stats.defence.source == "prior"


@REAL
def test_put_off_near_a_monster_and_not_waited_there():
    """A monster within 250 of her at the start: put off at once (nothing moves her for it), nothing sent; pending for
    the hunt loop (StatsDue at a quiet look DEFER_GAP_S on)."""
    got = _start(monsters="all")
    out, h = got["out"], got["h"]
    assert out.what == "put off" and got["keys"] == [] and h.stat_check.pending
    assert out.text.startswith("her stat window put off (green-mushroom, horny-mushroom within 250 of her)")
    first = float(next(line for line in got["log"] if "put off" in line).split()[0])
    assert first < 0.5  # (the first look: not 3 s standing there)
    at = h.stat_check.pending_at
    assert not h.stat_check.again(at + statcheck.DEFER_GAP_S - 0.1) and h.stat_check.again(at + statcheck.DEFER_GAP_S)


def test_put_off_where_no_hunt_loop_follows_is_not_this_run(monkeypatch):
    """`loop` False at a start (gamelab.__main__'s _hunter: `hunt kill`, a plain `tour`, `probe run`, not `tour
    --loop`): a read put off is not this run, said so, not pending (the review of 2026-10-30, of the chat box's check,
    which had it from here: "again at a quiet moment in the hunt" was said where nothing tried again). `loop` None: as
    before (a start's put-off pending)."""
    monkeypatch.setattr(statcheck.telemetry, "emit", lambda *a, **k: None)
    lines = []
    sc = StatCheck(SimpleNamespace(log=lines.append, doubts_raise=False))
    sc.due = lambda force=False: None
    sc._quiet = lambda: "green-mushroom within 250 of her"
    out = sc.run(at_start=True, loop=False)
    assert out.what == "put off" and not sc.pending and sc.loop is False
    assert lines == ["  her stat window not read: no quiet moment for it (green-mushroom within 250 of her): not this "
                     "run"]
    sc = StatCheck(SimpleNamespace(log=lines.append, doubts_raise=False))
    sc.due = lambda force=False: None
    sc._quiet = lambda: "green-mushroom within 250 of her"
    assert sc.run(at_start=True).what == "put off" and sc.pending
    assert lines[-1] == "  her stat window put off (green-mushroom within 250 of her): again at a quiet moment in the hunt"


@REAL
def test_another_client_size_nothing_sent():
    def bigger(sim, h):
        h.senses.eyes._capture = lambda: np.zeros((1080, 1920, 3), np.uint8)
    got = _start(before=bigger)
    assert got["out"].what == "not opened" and got["keys"] == []
    assert any("the client is 1920x1080, not 1366x768: not opened" in line for line in got["log"])


@REAL
def test_the_lie_detector_halfway_says_it_may_be_open():
    """The Lie Detector seen just after her stats key went out: the player refuses the rest (client.LieDetectorAlarm,
    as every plan's), and a window that may be open then is said and paged before the alarm goes on."""
    def alarm_soon(sim, h):
        sim.player.alarm_at = sim.clock.now + 0.9
    got = _start(before=alarm_soon)
    assert isinstance(got["raised"], client.LieDetectorAlarm)
    assert any("STAT WINDOW: her stat window may still be open in the game: close it with S or Esc" in line
               for line in got["log"])
    assert got["pages"] and got["pages"][-1][1] == "MapleStory: her stat window"


@REAL
def test_the_lie_detector_during_her_stats_key_says_it_may_be_open():
    """The review of 2026-10-28: the player stops a plan at the Lie Detector and answers it with its error, so a stats
    key that went down before the alarm raised out of its plan before the read knew it had gone out: an open window
    went unsaid through the test. Now it may be open from before the plan: said and paged."""
    def alarm_in_s(sim, h):
        real = h.actions.play

        def play(events, name, *a, **kw):
            out = real(events, name, *a, **kw)
            if name == "stat_window":  # (S down and up, then the player's error)
                raise client.LieDetectorAlarm("Lie Detector seen")
            return out
        h.actions.play = play
    got = _start(before=alarm_in_s)
    assert isinstance(got["raised"], client.LieDetectorAlarm) and got["keys"] == ["s"]
    assert got["world"]["stat_window_open"]
    assert any("STAT WINDOW: her stat window may still be open in the game: close it with S or Esc" in line
               for line in got["log"])
    assert got["pages"] and got["pages"][-1][1] == "MapleStory: her stat window"


@REAL
def test_the_safety_stop_at_every_look_of_it():
    """Her HP under her reserve at the start (no potion left): the safety stop at the read's own looks, nothing
    sent."""
    from gamelab.behavior.safety import SafetyStop

    def low(sim, h):
        sim.world.hp = 0.3
    got = _start(before=low, potions=0)
    assert isinstance(got["raised"], SafetyStop) and got["keys"] == []


def _keys(**kw):
    base = dict(left=parse_key("left"), right=parse_key("right"), jump=parse_key("alt"), attack=parse_key("ctrl"))
    return GameKeys(**{**base, **{k: parse_key(v) for k, v in kw.items()}})


def test_her_stats_key_from_her_remembered_bindings_else_s(tmp_path, monkeypatch):
    """S where nothing remembered says otherwise; ABILITY's key in her remembered key bindings (control.keymap's
    lab/keybindings-<her id>.json, 'functions'); refused where it's one of her own keys, Esc or the backslash, or her
    remembered bindings have ABILITY on no key."""
    monkeypatch.setattr(paths, "LAB", tmp_path)
    assert stats_key(_keys()) == ("s", "S, ABILITY's in her key bindings of 2026-10-28 (none remembered since)")
    key, why = stats_key(_keys(chair="s"))
    assert key is None and why == ("S (S, ABILITY's in her key bindings of 2026-10-28 (none remembered since)) is her "
                                   "chair key too: not pressed for it")
    (tmp_path / "keybindings-thief.json").write_text(json.dumps({"read_at": "2026-10-29 20:00:00", "functions": {
        "ctrl": "ATTACK", "k": "ABILITY", "s": "SKILL"}}), encoding="utf-8")
    assert stats_key(_keys()) == ("k", "ABILITY's in her remembered key bindings (keybindings-thief.json, read "
                                       "2026-10-29 20:00:00)")
    (tmp_path / "keybindings-thief.json").write_text(json.dumps({"functions": {"ctrl": "ATTACK", "esc": "ABILITY"}}),
                                                       encoding="utf-8")
    assert stats_key(_keys())[0] is None and "isn't a key to press for it" in stats_key(_keys())[1]
    (tmp_path / "keybindings-thief.json").write_text(json.dumps({"functions": {"ctrl": "ATTACK"}}), encoding="utf-8")
    assert stats_key(_keys()) == (None, "ABILITY on no key in her remembered key bindings (keybindings-thief.json, "
                                        "read ?)")


def test_her_def_known_this_session_or_read_again(tmp_path):
    """bootstrap.HerStats.window_due: none read; read at her level this session (the perception service's run, under
    SESSION_S): not again; at another level, in another session, over SESSION_S ago, or her newest 8 touches since not
    fitting it: again."""
    st = HerStats(tmp_path / "s.json", session="1")
    assert st.window_due(28, "1") == "no stat window read yet"
    t0 = st.run_start - 1000.0  # (read 1000 s before this run; `now` the time it's judged at)
    st.see_window(111, at=t0, level=28, session="1")
    assert st.window_due(28, "1", now=t0 + 60) is None
    assert st.window_due(29, "1", now=t0 + 60) == "her W.DEF 111 read at level 28; she's level 29 now"
    assert st.window_due(28, "2", now=t0 + 60).startswith("her W.DEF 111 read in another session")
    assert st.window_due(28, None, now=t0 + SESSION_S + 1).startswith("her W.DEF 111 read 3.0 h ago")
    rows = [{"t": t0 + 100 + i, "hp": 150, "level": 28, "pads": [110.0]} for i in range(8)]  # (DEF 0's touches)
    assert "of her newest 8 touches since it was read fit her W.DEF 111 (her gear changed?)" in st.window_due(
        28, "1", rows, now=t0 + 200)
    again = HerStats(tmp_path / "s.json", session="1")  # (kept: the next run in the same session knows it)
    again.see_level(28, HerStats.LV_FROM)
    assert again.window_due(28, "1") is None and again.defence.low == 111


def _read_into(values, hud_max_hp=None, bars=(800 / 840, 100 / 679)):
    """Hunter._stats_read of a read of `values` in the simulator's hunter (its HUD: level 25, max HP `hud_max_hp`, None:
    not read), her bars as the read's last look had them (`bars`: HP, MP; None: no look): the hunter, what was said."""
    got = {}

    def fn(sim):
        h = sim.tour_stack().h
        said = []
        h.log = said.append
        h.stats.max_hp = hud_max_hp
        h.stat_check = SimpleNamespace(scene=None if bars is None else SimpleNamespace(hp=bars[0], mp=bars[1]),
                                       _session=lambda: None)
        h._stats_read(sw.StatReading(at={"main": (0, 0), "detail": (0, 0)}, values=values), None)
        got["h"], got["said"] = h, " / ".join(said)
        return {}
    scenarios.run_sim(_sc(fn))
    return got["h"], got["said"]


@REAL
def test_what_the_read_gives_her_stats():
    """Her max HP from the window while her HUD's number hasn't read, her HP bar bearing it out; her max MP her casts
    are reckoned at (config.toml's 623 until then: said); her SPEED her walk and climb's prior from then on, by the
    client's formula (motion.calibrate: nothing measured yet), said with the change."""
    h, said = _read_into({"wdef": 111, "level": 25, "hp": (800, 840), "mp": (100, 679), "speed": 130, "jump": 100})
    assert h.max_hp == 840.0 and h.max_mp == 679.0 and h.attacks.max_mp == 679.0
    assert "her max HP 840 by her stat window" in said
    assert "her max MP 679 by her stat window (was 623, config.toml [character] mp): her casts reckoned at it" in said
    assert ("her SPEED 130 by her stat window: her walk 151.25 -> 187.5 and climb 121 -> 150 units/s from her next move "
            "(the client's formula)") in said
    assert h.motion.panel.speed == 130.0 and h.motion.prior_of("walk") == 187.5
    assert h.stats.defence.low == 111.0 and h.stats.defence.source == "stat window" and h.level == 25


@REAL
def test_a_read_her_hud_disagrees_with_is_not_taken():
    """The review of 2026-10-28 (a misread taken as read): the window's level or max HP not her HUD's (the game shows
    the same numbers in both: the panel's cells or the OCR off), none of it taken, said; her DEF, max HP and casts as
    they were."""
    h, said = _read_into({"wdef": 111, "level": 26, "hp": (840, 840), "mp": (100, 679)}, hud_max_hp=840)
    assert h.stats.defence.source == "prior" and h.max_mp == 623.0
    assert ("her stat window's read not taken: it says level 26, her HUD's LV. digits 25 (a misread?); her DEF as it "
            "was, DEF not known") in said
    h, said = _read_into({"wdef": 111, "level": 25, "hp": (8840, 8840)}, hud_max_hp=840)
    assert h.stats.defence.source == "prior" and "it says max HP 8840, her HUD's HP number 840 (a misread?)" in said


@REAL
def test_its_max_hp_and_mp_only_where_something_bears_them_out():
    """Her HUD's number not read: the window's max HP only where her HP bar bears it out (at full it can't tell a
    misread max: 8840 / 8840 isn't taken) or her stats file has it from her HUD as last read; its max MP where her MP
    bar bears it out or within MP_SPAN of what her casts are reckoned at (6790 / 6790 isn't). Her W.DEF taken all the
    same (her HUD read nothing to check it by)."""
    h, said = _read_into({"wdef": 111, "level": 25, "hp": (8840, 8840), "mp": (6790, 6790)}, bars=(1.0, 1.0))
    assert h.max_hp is None and h.max_mp == 623.0 and h.stats.defence.low == 111.0
    assert "her stat window's max HP 8840 not taken (at full her bar can't tell a misread max: not borne out)" in said
    assert ("her stat window's max MP 6790 not taken (not within x1.25 of 623 as known, at full her bar can't tell a "
            "misread max: not borne out): her casts reckoned at 623 as before") in said
    h, said = _read_into({"wdef": 111, "level": 25, "hp": (800, 840), "mp": (100, 679)}, bars=(0.5, 0.5))
    assert h.max_hp is None and "her bar at 50% against its 95%: not borne out" in said
    assert h.max_mp == 679.0  # (within x1.25 of 623)
    assert statcheck.borne((840, 840), None, 840) is None and statcheck.borne((900, 840), 1.0, 840) is not None


def test_the_kept_max_hp_bears_out_the_windows(tmp_path):
    """HerStats.kept_max_hp: her stats file's max HP (her HUD's, as last read) is what bears the window's out at full
    HP."""
    path = tmp_path / "her_stats.json"
    st = HerStats(path)
    st.see_max_hp(840)
    assert HerStats(path).kept_max_hp == 840 and HerStats(tmp_path / "none.json").kept_max_hp is None
    assert statcheck.borne((840, 840), 1.0, HerStats(path).kept_max_hp) is None


def test_a_hunting_commands_start_reads_it_after_her_max_hp(monkeypatch):
    """gamelab.__main__'s _hunter: her max HP waited for, then her stat window read (at_start); her max HP not read off
    her HUD, the window's taken before HpNotRead is raised; not a command that hunts (the pet), not read."""
    from gamelab import __main__ as cli
    from gamelab.behavior import hunter as hunter_mod
    from gamelab.behavior.safety import HpNotRead
    from gamelab.localization import mapdata, service
    from gamelab.screen import STRETCHED
    from test_worldstate import small_map

    class Client:
        def __init__(self, name, **kw):
            self.welcome, self.why = {"pid": 1, "epoch": 1, "map": kw["map_id"]}, ""

        def ready(self, timeout):
            return True
    order = []

    def made(eyes, keys, model, log, feed, **kw):
        h = SimpleNamespace(hp_why="why", too_strong=lambda: order.append("strong"), max_hp=made.hud,
                            senses=SimpleNamespace(window_fix=None))
        h.wait_max_hp = lambda: order.append("wait") or h.max_hp is not None
        h.read_stats = lambda at_start, loop=None: order.append(("read", at_start)) or setattr(
            h, "max_hp", h.max_hp if h.max_hp is not None else made.window)
        return h
    made.hud, made.window = 840, 840
    monkeypatch.setattr(service, "ServiceClient", Client)
    monkeypatch.setattr(cli, "_eyes", lambda model, title, perception, restore: SimpleNamespace(log=print,
                                                                                               geo=STRETCHED))
    monkeypatch.setattr(cli, "_fit_start", lambda title, fix: None)
    monkeypatch.setattr(mapdata, "load", lambda map_id: small_map())
    monkeypatch.setattr(hunter_mod, "Hunter", made)
    try:
        cli._hunter(cli.build_parser().parse_args(["tour", "--loop"]))
        assert order == ["wait", ("read", True), "strong"]
        made.hud, order[:] = None, []  # (her HUD's number not read: the window's max HP)
        cli._hunter(cli.build_parser().parse_args(["tour", "--loop"]))
        assert order == ["wait", ("read", True), "strong"]
        made.window, order[:] = None, []  # (nor the window's)
        with pytest.raises(HpNotRead):
            cli._hunter(cli.build_parser().parse_args(["tour", "--loop"]))
        assert order == ["wait", ("read", True)]
        order[:] = []
        cli._hunter(cli.build_parser().parse_args(["tour", "--loop"]), throws=False)  # (the pet: it hunts nothing)
        assert order == []
    finally:
        cli._PERCEPTION.clear()


def test_the_stop_and_the_outcome_words():
    stop = StatWindowStop("her stat window may be open in the game")
    assert stop.label == "STAT WINDOW" and stop.exit_code == 7 and "Close her stat window" in stop.advice
    assert stop.page() == ("MapleStory: her stat window", "gamelab stopped: her stat window may be open in the game.")
    assert statcheck.READS_MAX == 2 and statcheck.DEFERS_MAX == 6


def test_the_mouse_goes_off_the_calibrated_window():
    """The pointer goes to MOUSE_AT unless the calibration puts a panel there (16 px clear): then the first place of
    MOUSE_ELSE off every panel."""
    def check(**at):
        cal = sw.Calibration("copy", {k: {"at": list(v)} for k, v in at.items()})
        return StatCheck(SimpleNamespace(log=lambda m: None), cal=cal)._mouse_at()
    assert check(main=(560, 150)) == statcheck.MOUSE_AT == (1100, 300)
    assert check(main=(1050, 200)) == statcheck.MOUSE_ELSE[0] == (1300, 420)
    assert check(main=(1050, 200), detail=(1235, 200)) == statcheck.MOUSE_ELSE[1] == (120, 420)


# --- `gamelab stats --read --record` (the calibrating capture: the live check's tool) ---------------------------------

class StatGame:
    """A stand-in game for the capture: her stats key toggles the window (`opens`: it does at all; `newer`: a window
    the game data's pictures don't match, its labels and values read by a stand-in OCR), a click on DETAIL opens its
    panel (`detail`: open with the window), Esc closes it; `stuck` closing keys do nothing first. Every plan's keys
    kept."""

    def __init__(self, opens=True, detail=False, stuck=0, newer=False, size=(1366, 768)):
        self.opens, self.detail_first, self.stuck, self.newer, self.size = opens, detail, stuck, newer, size
        self.open, self.detail, self.pointer, self.plans, self.drawn, self.menu = False, False, (0, 0), [], None, 0
        self.base = np.full((size[1], size[0], 3), (90, 140, 70), np.uint8)

    def capture(self):
        if not self.open or self.size != (1366, 768):
            self.drawn = None
            return self.base.copy()
        if self.newer:
            import test_stat_window as tsw
            frame, ocr, _, _ = tsw._newer_ui(self.base)
            self.drawn = ocr
            return frame
        d = sw.compose({**dict(name="BugBaby", level=28, hp=(840, 840), mp=(138, 679)), "wdef": 111, "acc": 85},
                       base=self.base, detail=self.detail)
        self.drawn = d.ocr
        return d.frame

    def ocr(self, crop, box):
        return [] if self.drawn is None else self.drawn(crop, box)

    def play(self, events, name):
        self.plans.append((name, [e[1] for e in events if len(e) == 3 and e[2] == "down"]))
        for e in events:
            if len(e) == 4:
                self.pointer = (e[2], e[3])
            elif e[2] != "down":
                continue
            elif e[1] == "s" and self.opens:
                if self.open and self.stuck:
                    self.stuck -= 1
                elif self.open:
                    self.open = self.detail = False
                else:
                    self.open, self.detail = True, self.detail_first
            elif e[1] == "esc":
                if not self.open:
                    self.menu += 1
                elif self.stuck:
                    self.stuck -= 1
                else:
                    self.open = self.detail = False
            elif e[1] == "lbutton" and self.open:
                x, y = sw.detail_button(sw.DRAW_AT["main"])
                if abs(self.pointer[0] - x) <= 20 and abs(self.pointer[1] - y) <= 8:
                    self.detail = True


def _record(monkeypatch, tmp_path, capsys, game, click=True, level=28, max_hp=840):
    import gamelab.__main__ as cli
    monkeypatch.setattr(cli, "SHOTS", tmp_path / "shots")
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    monkeypatch.setattr(sw, "_shared_rows", game.ocr)

    def submit(events, fps=0, duration_ms=0, name="", **kw):  # (as client.wait answers: one numpy array of them)
        game.play(events, name)
        frames = [game.capture() for _ in range(int(duration_ms / 1000 * fps))]
        return {"frames": np.stack(frames), "times": np.arange(len(frames)) * 1000.0 / fps} if frames else {}
    monkeypatch.setattr(cli.client, "submit", submit)
    hunter = SimpleNamespace(senses=SimpleNamespace(eyes=SimpleNamespace(_capture=game.capture), feed=None),
                             touches=SimpleNamespace(rows=[]), stats=SimpleNamespace(level=level, max_hp=max_hp))
    code = cli._stats_recorded(hunter, "s", "S, ABILITY's", click=click)
    return code, capsys.readouterr().out, [k for _, keys in game.plans for k in keys]


def test_record_calibrates_off_the_game_datas_pictures(monkeypatch, tmp_path, capsys):
    """S, the window seen by the game data's pictures, a click on DETAIL, read, S, closed: calibrated (lab/
    stat_window.json: the copy's pictures, where each panel opened, DETAIL by a click), her W.DEF kept as hers."""
    code, out, keys = _record(monkeypatch, tmp_path, capsys, StatGame())
    assert code == 0 and keys == ["s", "lbutton", "s"], out
    assert "her stat window: W.DEF 111, level 28, HP 840/840, MP 138/679, ACC 85" in out and "read and closed (S)" in out
    cal = sw.Calibration.load()
    assert cal.source == "copy" and cal.place("main") == sw.DRAW_AT["main"] and cal.place("detail") == sw.DRAW_AT[
        "detail"] and not cal.detail_with_main and cal.key == "s"
    assert json.loads(paths.STATS.read_text(encoding="utf-8"))["window"]["low"] == 111.0
    folder = next((tmp_path / "shots").iterdir())
    assert {"before.png", "open.png", "detail.png", "closed1.png", "report.json"} <= {p.name for p in folder.iterdir()}


@pytest.mark.parametrize("game, click, want, keys, calibrated", [
    (dict(detail=True), True, "read and closed (S)", ["s", "s"], True),
    (dict(), False, "--no-click: her W.DEF isn't read", ["s", "s"], True),
    (dict(stuck=1), True, "read and closed (ESC)", ["s", "lbutton", "s", "esc"], True),
    (dict(stuck=5), True, "STILL OPEN after S and Esc twice", ["s", "lbutton", "s", "esc", "esc"], False),
    (dict(opens=False), True, "nothing opened after S (no window where the frame changed)", ["s"], False),
    (dict(size=(1920, 1080)), True, "the client is 1920x1080, not 1366x768: nothing sent", [], False),
])
def test_record_closes_only_what_it_saw_and_calibrates_only_once_closed(monkeypatch, tmp_path, capsys, game, click,
                                                                       want, keys, calibrated):
    g = StatGame(**game)
    code, out, sent = _record(monkeypatch, tmp_path, capsys, g, click)
    assert want in out and sent == keys and g.menu == 0, out
    assert (sw.Calibration.load() is not None) == calibrated
    if game.get("detail"):
        assert sw.Calibration.load().detail_with_main


def test_record_calibrates_a_newer_window_off_its_labels(monkeypatch, tmp_path, capsys):
    """A window the game data's pictures don't match: the frame against the one before tells it opened; cut from the
    live frame, its cells by its labels (the OCR: a stand-in here), read with them, closed by S (the cut templates no
    longer found): calibrated "live"."""
    code, out, keys = _record(monkeypatch, tmp_path, capsys, StatGame(newer=True))
    assert keys == ["s", "s"] and "that the game data's pictures don't match: calibrating from its labels" in out, out
    cal = sw.Calibration.load()
    assert cal is not None and cal.source == "live" and "read and closed (S)" in out
    assert "her stat window: W.DEF 111, level 28" in out and code == 0
    assert (paths.STAT_TEMPLATES / "main.png").is_file()


def test_record_stopped_halfway_says_the_window_may_be_open(monkeypatch, tmp_path, capsys):
    """The review of 2026-10-28: a crash after her stats key (its frames' truth, as keys --record's in 4c4bfa0) left the
    window open with nothing said. Anything raised after it went out: "MAY BE OPEN" said, the report written with the
    error, the error raised on; nothing more sent (no Esc unseen)."""
    def broken(*a, **kw):
        raise ValueError("the OCR fell over")
    monkeypatch.setattr(sw, "read", broken)
    g = StatGame()
    with pytest.raises(ValueError):
        _record(monkeypatch, tmp_path, capsys, g)
    out = capsys.readouterr().out
    assert "HER STAT WINDOW MAY BE OPEN IN THE GAME: close it with S or Esc" in out and g.open
    assert [k for _, keys in g.plans for k in keys] == ["s", "lbutton"] and g.menu == 0
    report = json.loads(next((tmp_path / "shots").iterdir()).joinpath("report.json").read_text(encoding="utf-8"))
    assert report["error"] == "ValueError: the OCR fell over" and "MAY BE OPEN" in report["result"]
    assert sw.Calibration.load() is None


def test_record_a_read_her_hud_disagrees_with_calibrates_nothing(monkeypatch, tmp_path, capsys):
    """Its level or max HP not her HUD's: read and closed, but not calibrated (a hunt would refuse every read), her
    W.DEF not kept; what to send said."""
    code, out, keys = _record(monkeypatch, tmp_path, capsys, StatGame(), level=27)
    assert code == 1 and keys == ["s", "lbutton", "s"] and "read and closed (S)" in out
    assert "not calibrated: its read says level 28, her level as read 27" in out and "stats --calibrate" in out
    assert sw.Calibration.load() is None and not paths.STATS.is_file()


def test_record_with_the_window_open_already_sends_nothing(monkeypatch, tmp_path, capsys):
    g = StatGame()
    g.open = True
    code, out, keys = _record(monkeypatch, tmp_path, capsys, g)
    assert code == 2 and keys == [] and "her stat window is open already" in out


@REAL
def test_a_level_up_in_the_hunt_reads_it_again_at_a_quiet_look():
    """Her W.DEF read at level 25 and her level read 26 in the hunt: the read again at the hunt loop's next quiet look
    (StatsDue from health_check, at once), READS_MAX a run; not outside the hunt loop."""
    from gamelab.behavior.statcheck import StatsDue
    got = {}

    def level(sim, n):  # (her HUD's digits and her stat window show level n from now on)
        for eyes in (sim.eyes, sim.frames):
            eyes.shown_level = lambda: n
            eyes._lv_drawn = None

    def fn(sim):
        scenarios._stat_calibrated(sim)
        h = sim.tour_stack().h
        assert h.read_stats(at_start=True).what == "read" and h.stats.window_read["level"] == 25
        sc = h.stat_check
        level(sim, 26)
        while h.level != 26:
            s = h.look()
        got["pending"] = sc.pending
        h.doubts_raise = False
        h.health_check(s)  # (not the hunt loop: not raised)
        h.doubts_raise = True
        try:
            h.health_check(h.look())
        except StatsDue:
            got["raised"] = True
        out = h.read_stats()
        got["second"] = (out.what, h.stats.window_read["level"])
        level(sim, 27)
        while h.level != 27:
            h.look()
        got["third"] = sc.pending  # (READS_MAX reached)
        got["log"] = [line for line in sim.trace.log if "her level 26 now" in line]
        return {}
    scenarios.run_sim(_sc(fn))
    assert got["pending"] and got.get("raised") and got["second"] == ("read", 26) and not got["third"]
    assert got["log"] and got["log"][0].endswith("her level 26 now, her W.DEF read at level 25: her stat window again "
                                                 "at a quiet moment in the hunt")


@REAL
def test_a_window_that_wont_close_in_the_hunt_loop_ends_the_hunt_with_her_retreat():
    """Put off at the start, read by the hunt loop, and it won't close: StatWindowStop out of the hunt loop, her retreat
    to a safe place (behavior.retreat.on_alarm, as every HuntStop), paged once by it. (Seed 2 since the evade work,
    2026-10-29: on seed 1 her fights' jumps back off a monster on her after a touch, stance.FIGHT_CLEAR, left no quiet
    moment for the read in the minute; of seeds 1-6, 2-6 meet all of this.)"""
    trace, sim = scenarios.run_sim(_sc(scenarios.stat_run(1.0), monsters="all", seed=2, stat_stuck=True))
    r = trace["result"]
    assert r["start"] == ["put off", []] and r["stopped"].startswith("STAT WINDOW: her stat window may be open")
    assert [p[1] for p in r["pages"]] == ["MapleStory: her stat window"]
    assert any("STAT WINDOW: retreat" in line or "STAT WINDOW" in line for line in trace["log"])
    keys = [k for _, _, ks in r["window"] for k in ks]
    assert keys == ["s", "lbutton", "s", "esc", "esc"] and trace["world"].get("menu_opened", 0) == 0
