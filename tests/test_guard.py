"""A stop that parks and a guard between hunts (behavior.guard; behavior.retreat.park_place, told; behavior.safety.StopAsked),
her potions and her death on the phone (behavior.vitals). The user, 2026-11-05: "you should build both, but the platform
picking should be dynamically chosen based on the current map"."""

import json
import time
from datetime import datetime
from types import SimpleNamespace

import pytest

from gamelab import paths
from gamelab.attack import danger
from gamelab.behavior import guard as g
from gamelab.behavior import retreat, vitals as v
from gamelab.behavior.guard import (ALERT, BACK_S, BLIND_S, EPISODE_S, HAND_S, OFF, ON, PARK_AGAIN_S, PARK_MOST, Guard, Pass,
                                    Seen)
from gamelab.behavior.retreat import Place, Retreat, park_place
from gamelab.behavior.safety import HuntStop, StopAsked
from gamelab.behavior.vitals import AGAIN_S, DEAD_S, MP_OUT_S, Vitals
from gamelab.control import client
from gamelab.localization import mapdata
from gamelab.motion.route import Planner
from gamelab.sim import scenarios

from gamelab.control.actions import Actions
from gamelab.control.timeline import SIM_KEYS_PET

from test_sim import tree

TREE = scenarios.TREE


def tree_keys():
    return SIM_KEYS_PET


@pytest.fixture(autouse=True)
def lab(tmp_path, monkeypatch):
    """The lab's files of all this in the test's own folder."""
    for name in ("VITALS", "HUNT", "STOP_HUNT", "PARK_NOTE", "PARKED", "GUARD", "GUARD_MODE"):
        monkeypatch.setattr(paths, name, tmp_path / getattr(paths, name).name)
    monkeypatch.setattr(g, "LOOP_STOP", tmp_path / "loop" / "STOP")
    return tmp_path


class Now:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _guard(t=1000.0):
    now, pushes, parks = Now(t), [], []
    return Guard(lambda title, text, critical: pushes.append((title, text, critical)), lambda: parks.append(now.t),
                 log=lambda m: None, clock=now), now, pushes, parks


# --- the guard's decisions ---------------------------------------------------------------------------------------------

def test_touched_with_the_game_in_front_and_the_player_up_she_parks_once_and_a_critical_alert_says_so():
    gd, now, pushes, parks = _guard()
    assert gd.act(Seen(touched=True, hp="HP 900 of 1244")) == "park" and parks == [1000.0]
    assert [(t, c) for t, _, c in pushes] == [("Parking her", True)] and "HP 900 of 1244" in pushes[0][1]
    now.t += 1
    assert gd.act(Seen()) is None and parks == [1000.0]  # nothing new: nothing


def test_nothing_setting_it_off_nothing_happens():
    gd, _, pushes, parks = _guard()
    assert gd.act(Seen()) is None and not pushes and not parks and gd.state == {}


def test_her_hp_under_the_stop_rule_sets_it_off_untouched_too():
    gd, _, pushes, parks = _guard()
    assert gd.act(Seen(low=True)) == "park" and "stop rule" in pushes[0][1]


def test_dead_nothing_is_sent_and_alive_it_is_set_off_afresh():
    gd, now, pushes, parks = _guard()
    assert gd.act(Seen(touched=True, dead=True)) is None and not parks and not pushes
    now.t += 500
    assert gd.act(Seen(touched=True)) == "park"


def test_the_lie_detector_up_no_moves_nothing_lost():
    gd, now, pushes, parks = _guard()
    assert gd.act(Seen(touched=True, alarm=True)) is None and not parks and not pushes
    assert gd.state["pending"]  # (it may be wanted once the alarm's over)
    now.t += 1
    assert gd.act(Seen()) == "park"


def test_played_by_hand_she_is_left_alone_and_only_the_stop_rule_is_told_once():
    gd, now, pushes, parks = _guard()
    assert gd.act(Seen(touched=True, by_hand=True)) is None and not pushes and not parks
    assert gd.act(Seen(low=True, by_hand=True)) is None and not parks
    assert [t for t, _, _ in pushes] == ["HP low"] and "no moves" in pushes[0][1]
    now.t += 1
    assert gd.act(Seen(low=True, by_hand=True)) is None and len(pushes) == 1
    now.t += 1
    assert gd.act(Seen(by_hand=True)) is None and not parks


def test_game_not_in_front_one_alert_and_she_parks_at_its_return():
    gd, now, pushes, parks = _guard()
    assert gd.act(Seen(touched=True, front=False)) is None and not parks
    assert [t for t, _, _ in pushes] == ["She can't park yet"] and "isn't in front" in pushes[0][1]
    now.t += 30
    assert gd.act(Seen(touched=True, front=False)) is None and len(pushes) == 1
    now.t += 30
    assert gd.act(Seen(front=True)) == "park" and parks


def test_the_player_down_one_alert_it_is_never_started_and_she_parks_once_it_is_up():
    gd, now, pushes, parks = _guard()
    assert gd.act(Seen(touched=True, player=False)) is None and not parks
    assert "player isn't running" in pushes[0][1]
    now.t += 5
    assert gd.act(Seen(player=False)) is None and len(pushes) == 1
    assert gd.act(Seen(player=True)) == "park"


def test_a_park_that_fails_waits_its_cool_down_and_gives_up_with_one_alert():
    gd, now, pushes, parks = _guard()
    assert gd.act(Seen(touched=True)) == "park"
    now.t += PARK_AGAIN_S - 1
    assert gd.act(Seen(touched=True)) is None
    for _ in range(PARK_MOST - 1):
        now.t += PARK_AGAIN_S
        assert gd.act(Seen(touched=True)) == "park"
    now.t += PARK_AGAIN_S
    assert gd.act(Seen(touched=True)) is None and len(parks) == PARK_MOST
    assert pushes[-1][0] == "She can't park"
    now.t += PARK_AGAIN_S
    assert gd.act(Seen(touched=True)) is None and len(parks) == PARK_MOST and len(pushes) == 2


def test_nothing_false_a_second_after_the_last_park_starts():
    gd, now, pushes, parks = _guard()
    for _ in range(PARK_MOST):
        assert gd.act(Seen(touched=True)) == "park"
        now.t += PARK_AGAIN_S
    now.t -= PARK_AGAIN_S - 1
    assert gd.act(Seen(touched=True)) is None and [t for t, _, _ in pushes] == ["Parking her"]


def test_a_park_still_wanted_outlasts_the_episode_however_long_the_game_is_away():
    gd, now, pushes, parks = _guard()
    assert gd.act(Seen(touched=True, front=False)) is None
    now.t += EPISODE_S * 10
    assert gd.act(Seen(front=False)) is None and gd.state["pending"] and len(pushes) == 1
    now.t += 1
    assert gd.act(Seen()) == "park"


def test_she_waits_while_the_game_is_only_just_back():
    gd, now, pushes, parks = _guard()
    assert gd.act(Seen(touched=True, held=True)) is None and not parks and gd.state["pending"]
    now.t += 1
    assert gd.act(Seen()) == "park"


def test_a_stops_note_alone_is_told_as_a_regular_push_a_hunt_that_died_as_a_critical_one():
    gd, _, pushes, _ = _guard()
    gd.act(Seen(note="`stop`", front=False))
    assert pushes[-1][0] == "She can't park yet" and not pushes[-1][2]
    gd, _, pushes, _ = _guard()
    gd.state = {}
    gd.act(Seen(note="`tour` ended without parking her", urgent=True, front=False))
    assert pushes[-1][2] and "`tour` ended" in pushes[-1][1]


def test_dead_a_note_is_over():
    gd, _, _, parks = _guard()
    paths.PARK_NOTE.write_text("{}")
    assert gd.act(Seen(note="`stop`", dead=True)) is None and not paths.PARK_NOTE.exists() and not parks


def test_an_episode_ends_and_the_next_is_told_afresh():
    gd, now, pushes, parks = _guard()
    gd.act(Seen(touched=True))
    now.t += EPISODE_S
    assert gd.act(Seen()) is None and gd.state.get("episode") is None
    now.t += 1
    assert gd.act(Seen(touched=True)) == "park" and len(pushes) == 2


def test_off_nothing_and_alert_only_the_push_and_no_moves():
    gd, now, pushes, parks = _guard()
    assert gd.act(Seen(touched=True), OFF) is None and not pushes and gd.state == {}
    assert gd.act(Seen(touched=True), ALERT) == "alert" and not parks
    assert "alert only" in pushes[0][1]
    now.t += 1
    assert gd.act(Seen(touched=True), ALERT) is None and len(pushes) == 1


def test_its_state_survives_the_pet_feeder_being_stopped_for_the_park(lab):
    gd, now, pushes, parks = _guard()
    gd.act(Seen(touched=True))
    again, _, more, parks2 = _guard(now.t + 10)  # (a new process: the babysitter stopped the feeder, then started it)
    assert again.act(Seen(touched=True)) is None and not more and not parks2  # (its cool-down, the alert told)
    assert json.loads(paths.GUARD.read_text())["parks"] == 1


def test_a_stops_note_parks_her_even_untouched_and_played_by_hand_it_is_over(lab):
    gd, now, pushes, parks = _guard()
    assert gd.act(Seen(note="`stop`")) == "park" and not pushes[-1][2]  # (the user asked: a regular push)
    paths.PARK_NOTE.write_text("{}")
    now.t += PARK_AGAIN_S
    assert gd.act(Seen(note="`stop`", by_hand=True)) is None and not paths.PARK_NOTE.exists() and len(parks) == 1


# --- played by hand: Windows' last input, against the player's own keys ------------------------------------------------

class Hands:
    """A pass of the guard with a clock, the game's window and Windows' last input in the test's hands."""

    def __init__(self, monkeypatch, tmp_path, state=None):
        paths.GUARD.unlink(missing_ok=True)
        self.gd, self.now, self.pushes, self.parks = _guard(100000.0)
        self.foreground, self.input_at, self.state = "game", self.now.t - 1000.0, state or _state()
        self.log = tmp_path / "player.log"
        self.log.write_text("", encoding="utf-8")
        monkeypatch.setattr(client, "alarm_up", lambda: False)
        monkeypatch.setattr(client, "player_up", lambda: True)
        monkeypatch.setattr(g.window, "foreground_title", lambda: self.foreground)
        monkeypatch.setattr(g.window, "last_input_age", lambda: self.now.t - self.input_at)
        monkeypatch.setattr(g.paths, "PLAYER_LOG", self.log)
        self.p = Pass(self.gd, lambda: self.state, lambda st: st.hp < 0.2, "game", Vitals())

    def player_key(self):
        """A key of the player's, now: Windows' last input moves, and its log has the plan (whole seconds)."""
        self.input_at = self.now.t
        self.log.write_text(time.strftime("%H:%M:%S", time.localtime(self.now.t)) + " ran 1_pet_food_ab: ok\n", encoding="utf-8")

    def run(self, seconds):
        """A pass a second; the first 'park' or None."""
        for _ in range(int(seconds)):
            self.now.t += 1.0
            if self.p.tick() == "park":
                return "park"
        return None

    def touch(self):
        self.now.t += 1.0
        self.state = _state(told=self.now.t)
        return self.p.tick()

    def touched_twice(self):
        self.state = _state(told=1.0)
        self.p.tick()
        self.touch()
        return self.touch()


def test_a_key_played_by_hand_counts_for_its_time_and_the_players_own_keys_dont(monkeypatch, tmp_path):
    h = Hands(monkeypatch, tmp_path)
    h.run(1)
    h.player_key()
    h.run(1)
    assert h.touched_twice() == "park"  # (the player's key: nobody's playing)
    h = Hands(monkeypatch, tmp_path)
    h.run(1)
    h.input_at = h.now.t
    h.run(1)
    assert h.touched_twice() is None and not h.parks  # (a hand)
    h.run(HAND_S + 1)
    assert not h.parks and h.p.played_at
    assert h.touched_twice() == "park"  # (over HAND_S ago)


def test_the_pets_food_tap_after_a_key_doesnt_hide_it(monkeypatch, tmp_path):
    # (review, 2026-11-05: a key 8 s ago, a tap 3 s ago, and two touches parked her against whoever was playing)
    h = Hands(monkeypatch, tmp_path)
    h.run(1)
    h.input_at = h.now.t
    h.run(5)
    h.player_key()
    h.run(3)
    assert h.touched_twice() is None and not h.parks


def test_input_with_the_game_behind_isnt_playing_her(monkeypatch, tmp_path):
    h = Hands(monkeypatch, tmp_path)
    h.foreground = "work"
    h.run(1)
    for _ in range(5):
        h.input_at = h.now.t
        h.run(1)
    assert h.p.played_at == 0.0


def test_the_click_that_brings_the_game_back_isnt_playing_her_and_a_stops_note_parks_her(monkeypatch, tmp_path):
    # (both reviews, 2026-11-05: the game only comes back by a person's click, which read as playing: the note was dropped)
    h = Hands(monkeypatch, tmp_path)
    h.foreground = "work"
    paths.PARK_NOTE.write_text(json.dumps({"t": 1.0, "why": "`stop`"}))
    assert h.run(120) is None and [t for t, _, _ in h.pushes] == ["She can't park yet"]
    h.foreground = "game"
    h.input_at = h.now.t + 1.0  # (the click)
    back = h.now.t
    assert h.run(2 * BACK_S - 1) is None and not h.parks and paths.PARK_NOTE.exists()  # (held, for whoever's come to play)
    assert h.run(2) == "park" and h.now.t - back <= 2 * BACK_S + 1


def test_touched_while_away_she_parks_at_the_games_return_after_the_click(monkeypatch, tmp_path):
    h = Hands(monkeypatch, tmp_path)
    h.touched_twice()  # (the park's first: the game in front)
    h.parks.clear()
    h.foreground = "work"
    h.now.t += PARK_AGAIN_S
    h.touch()
    h.touch()
    assert h.run(EPISODE_S * 2) is None and not h.parks and h.gd.state["pending"]
    h.foreground = "game"
    h.input_at = h.now.t + 1.0
    assert h.run(2 * BACK_S + 1) == "park"


def test_back_and_playing_she_is_left_alone_and_the_note_is_over(monkeypatch, tmp_path):
    h = Hands(monkeypatch, tmp_path)
    h.foreground = "work"
    paths.PARK_NOTE.write_text(json.dumps({"t": 1.0, "why": "`stop`"}))
    h.run(30)
    h.foreground = "game"
    for _ in range(int(3 * BACK_S)):  # (a key a second from the click on)
        h.input_at = h.now.t + 1.0
        assert h.run(1) is None
    assert not h.parks and not paths.PARK_NOTE.exists()


def test_known_parked_her_hp_low_alone_parks_her_no_more(monkeypatch, tmp_path):
    # (both reviews: after a safety stop with her potions out, three no-op parks and a false "She can't park")
    h = Hands(monkeypatch, tmp_path, _state(hp=0.1))
    paths.PARKED.write_text(json.dumps({"t": h.now.t, "there": True, "said": "parked on a rope"}))
    assert h.run(PARK_AGAIN_S * 2) is None and not h.parks and not h.pushes
    h.state = _state(hp=0.1, told=1.0)
    h.p.tick()
    assert h.touch() is None
    assert h.touch() == "park"  # (touched there: it's no safe place after all)


def test_not_known_parked_her_hp_low_parks_her_and_a_park_that_failed_counts_for_nothing(monkeypatch, tmp_path):
    h = Hands(monkeypatch, tmp_path, _state(hp=0.1))
    paths.PARKED.write_text(json.dumps({"t": h.now.t, "there": False, "said": "not parked"}))
    assert h.run(3) == "park"
    h = Hands(monkeypatch, tmp_path, _state(hp=0.1))
    paths.PARKED.write_text(json.dumps({"t": h.now.t, "there": True, "said": "parked"}))
    h.run(1)
    h.input_at = h.now.t  # (played since: no longer known)
    assert h.run(HAND_S) is None and [t for t, _, _ in h.pushes] == ["HP low"]
    assert h.run(3) == "park"


# --- a hunt that dies, and no state ------------------------------------------------------------------------------------

def test_a_hunt_killed_or_crashed_is_a_note_at_once_and_its_clean_end_isnt(monkeypatch, tmp_path):
    h = Hands(monkeypatch, tmp_path)
    paths.HUNT.write_text(json.dumps({"pid": 4242, "command": "tour"}))
    monkeypatch.setattr(client, "_alive", lambda pid: False)
    assert h.run(1) == "park" and not paths.HUNT.exists()
    assert h.pushes[0][2] and "`tour` ended without parking her" in h.pushes[0][1]
    monkeypatch.setattr(g.os, "getpid", lambda: 4242)
    paths.PARK_NOTE.unlink()
    paths.HUNT.write_text(json.dumps({"pid": 4242, "command": "tour"}))
    g.over()
    assert not paths.HUNT.exists()
    g.left_behind()
    assert not paths.PARK_NOTE.exists()


def test_a_hunt_killed_with_the_game_behind_is_told_at_once_with_no_state_to_go_by(monkeypatch, tmp_path):
    h = Hands(monkeypatch, tmp_path)
    h.state, h.foreground = None, "work"
    paths.HUNT.write_text(json.dumps({"pid": 4242, "command": "tour"}))
    monkeypatch.setattr(client, "_alive", lambda pid: False)
    assert h.run(2) is None and [(t, c) for t, _, c in h.pushes] == [("She can't park yet", True)]


def test_a_pid_given_again_isnt_the_hunt(monkeypatch):
    paths.HUNT.write_text(json.dumps({"pid": 4242, "started": 111, "command": "tour"}))
    monkeypatch.setattr(client, "_alive", lambda pid: True)
    monkeypatch.setattr(client, "started", lambda pid: 111)
    assert g.running_hunt()["command"] == "tour"
    monkeypatch.setattr(client, "started", lambda pid: 222)
    assert g.running_hunt() is None


def test_no_usable_state_for_minutes_one_regular_push_unless_shes_known_parked(monkeypatch, tmp_path):
    h = Hands(monkeypatch, tmp_path)
    h.state, h.foreground = None, "work"
    assert h.run(BLIND_S - 2) is None and not h.pushes
    h.run(4 + 600)
    assert [(t, c) for t, _, c in h.pushes] == [("The guard can't see her", False)] and "isn't in front" in h.pushes[0][1]
    h.state = _state()
    h.run(1)
    h.state = None
    h.run(BLIND_S + 2)
    assert len(h.pushes) == 2  # (back, and blind again)
    h = Hands(monkeypatch, tmp_path)
    h.state, h.foreground = None, "work"
    paths.PARKED.write_text(json.dumps({"t": h.now.t, "there": True, "said": "parked"}))
    h.run(BLIND_S + 60)
    assert not h.pushes


def test_a_hunt_claims_its_entry_a_live_ones_stays_a_parks_goes_and_only_an_old_request_is_cleared(monkeypatch):
    monkeypatch.setattr(client, "_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(client, "started", lambda pid: None)
    paths.HUNT.write_text(json.dumps({"pid": 4242, "command": "tour"}))
    paths.STOP_HUNT.write_text(json.dumps({"t": time.time()}))
    g.claim("nav")
    assert json.loads(paths.HUNT.read_text())["command"] == "tour" and paths.STOP_HUNT.exists()
    paths.HUNT.write_text(json.dumps({"pid": 4242, "command": "nav-park"}))
    for path in (paths.PARK_NOTE, paths.PARKED, paths.GUARD):
        path.write_text("{}")
    g.claim("tour")
    assert json.loads(paths.HUNT.read_text())["pid"] == g.os.getpid()
    assert paths.STOP_HUNT.exists()  # (fresh: this hunt's first look reads it)
    assert not paths.PARK_NOTE.exists() and not paths.PARKED.exists() and not paths.GUARD.exists()
    paths.STOP_HUNT.write_text(json.dumps({"t": time.time() - g.STALE_S - 1}))
    paths.PARK_NOTE.write_text("{}")
    paths.HUNT.unlink()
    g.claim("nav-park")
    assert not paths.STOP_HUNT.exists() and paths.PARK_NOTE.exists()  # (a park keeps the note until it's told)


def test_only_a_hunting_command_claims_the_entry(tmp_path, monkeypatch):
    from gamelab import __main__ as cli
    monkeypatch.setattr(cli.telemetry, "record_to", lambda command: tmp_path / "t")
    monkeypatch.setattr(cli.telemetry, "emit", lambda *a, **k: None)
    monkeypatch.setattr("gamelab.gcstats.report_every", lambda *a, **k: None)
    for command in ("stats", "keys", "pet"):
        cli._record(command, 1)
        assert not paths.HUNT.exists()
    for command in ("tour", "hunt-kill", "probe-x", "nav", "nav-park"):
        cli._record(command, 1)
        assert json.loads(paths.HUNT.read_text())["command"] == command
        paths.HUNT.unlink()


def test_the_players_last_key_is_its_logs_last_plan_time_only_stamps_across_midnight(tmp_path):
    log = tmp_path / "player.log"
    log.write_text("23:59:58 ran 235958_flow_0520: ok\n23:59:59 spliced 235959_flow_f407 -> x at 784 ms\n"
                   "00:00:01 player waiting\n", encoding="utf-8")
    at = datetime(2026, 11, 6, 0, 0, 5).timestamp()
    assert g.player_last_key(at, log) == datetime(2026, 11, 5, 23, 59, 59).timestamp()
    assert g.player_last_key(datetime(2026, 11, 5, 23, 59, 59).timestamp() + 2, log) == datetime(2026, 11, 5, 23, 59, 59).timestamp()
    log.write_text("no plan\n", encoding="utf-8")
    assert g.player_last_key(at, log) is None
    assert g.player_last_key(at, tmp_path / "none") is None


# --- a pass: perception's newest state alone ---------------------------------------------------------------------------

def _state(hp=0.9, mp=0.5, hp_of=None, told=None, crowded=False):
    touch = None if told is None else {"told": told, "crowded": crowded}
    return SimpleNamespace(hp=hp, mp=mp, hp_of=hp_of, touch=touch)


def test_a_pass_tells_a_new_touch_from_an_old_one_and_not_one_crowded(monkeypatch):
    gd, now, pushes, parks = _guard()
    monkeypatch.setattr(client, "alarm_up", lambda: False)
    monkeypatch.setattr(client, "player_up", lambda: True)
    monkeypatch.setattr(g, "running_hunt", lambda: None)
    monkeypatch.setattr(g.window, "foreground_title", lambda: "game")
    monkeypatch.setattr(g.window, "last_input_age", lambda: None)
    states = [_state(told=5.0), _state(told=5.0), _state(told=6.0, crowded=True), _state(told=7.0), _state(told=47.0),
              _state(told=50.0)]
    p = Pass(gd, lambda: states.pop(0), lambda st: False, "game", Vitals())
    assert p.tick() is None and p.tick() is None and p.tick() is None and not parks  # (from before it; the same; crowded)
    assert p.tick() is None and p.tick() is None and not parks  # (lone popups: over 30 s apart)
    assert p.tick() == "park"


def test_a_pass_needs_her_hp_low_twice_and_with_no_state_it_says_nothing(monkeypatch):
    gd, now, pushes, parks = _guard()
    monkeypatch.setattr(client, "alarm_up", lambda: False)
    monkeypatch.setattr(client, "player_up", lambda: True)
    monkeypatch.setattr(g, "running_hunt", lambda: None)
    monkeypatch.setattr(g.window, "foreground_title", lambda: "game")
    monkeypatch.setattr(g.window, "last_input_age", lambda: None)
    states = [None, _state(hp=0.1), _state(hp=0.9), _state(hp=0.1), _state(hp=0.1)]
    p = Pass(gd, lambda: states.pop(0), lambda st: st.hp < 0.2, "game", Vitals())
    assert [p.tick() for _ in range(4)] == [None, None, None, None] and not parks
    assert p.tick() == "park"


def test_a_pass_reads_the_switch_each_time_and_a_hunt_running_holds_it(lab, monkeypatch):
    gd, now, pushes, parks = _guard()
    monkeypatch.setattr(client, "alarm_up", lambda: False)
    monkeypatch.setattr(client, "player_up", lambda: True)
    monkeypatch.setattr(g.window, "foreground_title", lambda: "game")
    monkeypatch.setattr(g.window, "last_input_age", lambda: None)
    p = Pass(gd, lambda: _state(hp=0.1), lambda st: True, "game", Vitals())
    paths.GUARD_MODE.write_text("off")
    assert g.read_mode() == OFF and p.tick() is None and p.lows == 0
    paths.GUARD_MODE.write_text("alert")
    assert g.read_mode() == ALERT
    paths.GUARD_MODE.unlink()
    assert g.read_mode() == ON
    monkeypatch.setattr(g, "running_hunt", lambda: {"pid": 1, "command": "tour"})
    assert p.tick() is None and p.tick() is None and not parks
    monkeypatch.setattr(g, "running_hunt", lambda: None)
    assert p.tick() == "park"


def test_the_pet_feeders_wait_has_the_guards_passes_and_one_that_fails_stops_nothing(monkeypatch, capsys):
    from gamelab import __main__ as cli
    now, ticks = [0.0], []

    class Pass_:
        def tick(self):
            ticks.append(now[0])
            if len(ticks) == 2:
                raise ValueError("no state")
    monkeypatch.setattr(cli.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(cli.time, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    cli._guarded(Pass_(), 5.0)
    assert ticks == [0.0, 1.0, 2.0, 3.0, 4.0] and capsys.readouterr().out.count("guard: a pass failed") == 1
    now[0] = 0.0
    cli._guarded(None, 60.0)
    assert now[0] == 60.0


def test_the_guards_pass_reads_her_hp_against_the_stop_rule_as_the_hunters_reserve_has_it(monkeypatch):
    from gamelab import __main__ as cli
    from gamelab.control import notify
    pushed = []
    monkeypatch.setattr(notify, "phone", lambda *a, **k: pushed.append(a))
    hunter = SimpleNamespace(max_hp=1000, hp_watch=SimpleNamespace(threshold=lambda most: 378.0 if most else 0.5),
                             senses=SimpleNamespace(feed=SimpleNamespace(latest=lambda want, timeout, ask: None)))
    p = cli._guard_pass(hunter, "game")
    assert p.latest() is None
    assert p.low(SimpleNamespace(hp=0.9, hp_of=(300, 1244))) and not p.low(SimpleNamespace(hp=0.2, hp_of=(400, 1244)))
    assert p.low(SimpleNamespace(hp=0.3, hp_of=None)) and not p.low(SimpleNamespace(hp=0.4, hp_of=None))  # (300, 400 of 1000)
    p.guard.push("t", "x", True)
    assert pushed == [("t", "x", True)]


# --- `stop` ------------------------------------------------------------------------------------------------------------

def test_stop_with_a_hunt_running_asks_it_and_says_what_came_of_it(lab, monkeypatch):
    (lab / "loop").mkdir()
    paths.HUNT.write_text(json.dumps({"pid": 4242, "command": "tour"}))
    alive = [True, True, False, False]
    monkeypatch.setattr(client, "_alive", lambda pid: alive.pop(0) if alive else False)
    cut = []
    monkeypatch.setattr(client, "cut_running", lambda: cut.append(1))
    said = []

    def sleep(s):  # (the hunt, meanwhile)
        assert paths.STOP_HUNT.exists()
        paths.PARKED.write_text(json.dumps({"t": time.time() + 1, "there": True, "said": "parked on platform 27"}))
    assert g.stop("game", said.append, sleep=sleep) == 0
    assert said[-1] == "parked on platform 27" and g.LOOP_STOP.exists() and cut


def test_stop_with_no_hunt_parks_her_and_with_the_game_away_leaves_the_note(lab, monkeypatch):
    monkeypatch.setattr(client, "alarm_up", lambda: False)
    monkeypatch.setattr(g.window, "foreground_title", lambda: "game")
    said, went = [], []

    def park():
        went.append(1)
        paths.PARKED.write_text(json.dumps({"t": time.time() + 1, "there": True, "said": "stays on platform 26"}))
    assert g.stop("game", said.append, park=park) == 0 and went and said == ["stays on platform 26"]
    monkeypatch.setattr(g.window, "foreground_title", lambda: "another")
    said.clear()
    went.clear()
    assert g.stop("game", said.append, park=park) == 0 and not went and paths.PARK_NOTE.exists()
    assert "isn't in front" in said[0]


def test_stop_looks_again_after_parking_for_a_grind_the_loop_began_meanwhile(lab, monkeypatch):
    monkeypatch.setattr(client, "alarm_up", lambda: False)
    monkeypatch.setattr(g.window, "foreground_title", lambda: "game")
    alive = [True, False]
    monkeypatch.setattr(client, "_alive", lambda pid: alive.pop(0) if alive else False)
    monkeypatch.setattr(client, "cut_running", lambda: None)
    said = []

    def park():
        paths.HUNT.write_text(json.dumps({"pid": 4242, "command": "tour"}))
    def sleep(s):
        pass
    alive[:] = [True, True, False]
    assert g.stop("game", said.append, park=park, sleep=sleep) == 1
    assert paths.PARK_NOTE.exists() and any("asked `tour`" in m for m in said)


def test_stop_cuts_no_plan_with_the_alarm_up(lab, monkeypatch):
    paths.HUNT.write_text(json.dumps({"pid": 4242, "command": "tour"}))
    monkeypatch.setattr(client, "alarm_up", lambda: True)
    monkeypatch.setattr(client, "_alive", lambda pid: False)
    alive = [True, False]
    monkeypatch.setattr(client, "_alive", lambda pid: alive.pop(0) if alive else False)
    monkeypatch.setattr(client, "cut_running", lambda: pytest.fail("the Lie Detector's cursor and OK are plans too"))
    assert g.stop("game", lambda m: None, sleep=lambda s: None) == 1 and paths.PARK_NOTE.exists()


def test_with_the_guard_off_stop_doesnt_say_it_parks_her(lab, monkeypatch):
    paths.GUARD_MODE.write_text("off")
    monkeypatch.setattr(client, "alarm_up", lambda: False)
    monkeypatch.setattr(g.window, "foreground_title", lambda: "another")
    said = []
    g.stop("game", said.append)
    assert "the guard is off" in said[0]


def test_stop_with_the_alarm_up_moves_nothing(lab, monkeypatch):
    monkeypatch.setattr(client, "alarm_up", lambda: True)
    went, said = [], []
    assert g.stop("game", said.append, park=lambda: went.append(1)) == 0 and not went and not paths.PARK_NOTE.exists()


def test_a_stop_asked_goes_to_the_parks_place_pages_nobody_and_says_how_it_came_out(monkeypatch):
    said, paged, went = [], [], []
    h = SimpleNamespace(log=said.append)

    def away(hunter, seen, why, log=None, rng=None):
        went.append(why)
        out = Retreat(0.0, Place(27, None, "a safe platform", 3.0), "arrived", 4.0, (27, 80.0))
        return out
    monkeypatch.setattr(retreat, "away", away)
    monkeypatch.setattr(retreat.alarm, "page", lambda title, text: paged.append(title))
    paths.STOP_HUNT.write_text("{}")
    paths.PARK_NOTE.write_text("{}")
    with pytest.raises(StopAsked):
        with retreat.on_alarm(h):
            assert h.stoppable
            raise StopAsked()
    assert went[0].park and went[0].seconds == retreat.PARK_S and went[0].event == "Stop asked" and not paged
    row = json.loads(paths.PARKED.read_text())
    assert row["there"] and row["said"] == "parked on platform 27 (a safe platform), 4.0 s on"
    assert not paths.STOP_HUNT.exists() and not paths.PARK_NOTE.exists()


def test_a_stop_asked_with_the_game_away_ends_with_a_note_and_no_retreat(monkeypatch):
    went = []
    monkeypatch.setattr(retreat, "away", lambda *a, **k: went.append(a))
    with pytest.raises(StopAsked):
        with retreat.on_alarm(SimpleNamespace(log=lambda m: None)):
            raise StopAsked(later=True)
    assert not went and paths.PARK_NOTE.exists() and not json.loads(paths.PARKED.read_text())["there"]


def test_a_stops_look_reads_the_request_and_a_look_with_nothing_to_act_on_waits_then_ends(monkeypatch):
    from gamelab.behavior.hunter import Hunter
    h = Hunter.__new__(Hunter)
    h.stoppable, h.retreating, h.asked, h._stop_wait = True, False, False, 0.0
    h._stop_now(None)  # (none asked)
    paths.STOP_HUNT.write_text("{}")
    with pytest.raises(StopAsked):
        h._stop_now(None)
    h.retreating = True
    h._stop_now(None)  # (her retreat isn't cut by it)
    h.retreating, h.stoppable = False, False
    h._stop_now(None)  # (a pet feeder's hunter: it's for no command that doesn't hunt)
    h.stoppable = True
    now = [100.0]
    monkeypatch.setattr("gamelab.behavior.hunter.time.monotonic", lambda: now[0])
    h._stop_waiting(None)
    now[0] += 59.0
    h._stop_waiting(None)
    now[0] += 2.0
    with pytest.raises(StopAsked) as e:
        h._stop_waiting(None)
    assert e.value.later


def test_her_retreats_plans_never_bring_the_game_forward(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "INBOX", tmp_path)
    pid = client.send([[0, "left", "down"]], name="flow", kind=client.RETREAT)
    assert json.loads((tmp_path / f"{pid}.json").read_text())["focus"] is False
    monkeypatch.setattr(client, "check_alarm", lambda: None)
    pid = client.send([[0, "left", "down"]], name="flow")
    assert json.loads((tmp_path / f"{pid}.json").read_text())["focus"] is True


def test_a_stop_cuts_what_the_player_is_playing_and_what_waits_withdrawing_nothing(tmp_path, monkeypatch):
    # (a plan withdrawn from another process: its sender's `wait` had no result for 120 s, then a PlayerError no retreat
    # follows: the review of 2026-11-05)
    monkeypatch.setattr(client, "INBOX", tmp_path)
    (tmp_path / "a_flow.running").write_text("")
    (tmp_path / "b_throw.json").write_text("{}")
    client.cut_running()
    assert (tmp_path / "a_flow.cancel").exists() and (tmp_path / "b_throw.cancel").exists()
    assert (tmp_path / "b_throw.json").exists()


def test_a_plan_sent_after_the_stop_isnt_sent(monkeypatch):
    # (`cut_running` can't catch it: 145 ms after the request, a Lucky Seven in the simulator)
    actions = Actions(tree_keys())
    sent = []
    monkeypatch.setattr(client, "submit", lambda *a, **k: sent.append(a) or {})
    actions.before_play = lambda: (_ for _ in ()).throw(StopAsked())
    with pytest.raises(StopAsked):
        actions.play([[0, "shift", "down"], [40, "shift", "up"]], "throw")
    assert not sent
    away = actions.moves_only()
    assert away.before_play is None  # (her retreat's moves go on)
    away.play([[0, "left", "down"], [40, "left", "up"]], "flow")
    assert sent


def test_a_senses_look_with_hands_off_doesnt_bring_the_game_forward():
    from gamelab.localization.senses import Senses
    from gamelab.perception.framecheck import LookLost, Problem
    brought = []
    s = Senses.__new__(Senses)
    s._feed = SimpleNamespace(look=lambda *a: None, problem=lambda: Problem("window", "the game isn't in front"),
                              bring_forward=brought.append)
    s.shows, s.keyed = lambda: -1e18, lambda: -1e18
    s._looked, s.settled, s.shape = 0.0, 0.0, SimpleNamespace(shows_wait_s=0.0)
    s.waits, s.window_fix, s.log, s.failed, s._failing, s._set_right, s.regains = [], None, lambda m: None, {"looks": 0, "lost": 0}, None, -1e18, False
    s.hands_off = lambda: True
    s._failing = (time.monotonic() - 100.0, "the game isn't in front")
    with pytest.raises(LookLost):
        s.look()
    assert not brought
    s.hands_off = lambda: False
    with pytest.raises(LookLost):
        s.look()
    assert brought


def test_nav_park_with_the_game_not_in_front_sends_nothing_and_leaves_the_note(monkeypatch, capsys):
    from gamelab import __main__ as cli
    monkeypatch.setattr(cli, "_game_title", lambda: "game")
    monkeypatch.setattr(cli.window, "foreground_title", lambda: "another")
    monkeypatch.setattr(cli, "_hunter", lambda *a, **k: pytest.fail("no hunter: nothing to send keys"))
    monkeypatch.setattr(cli.window, "bring_to_front", lambda *a, **k: pytest.fail("she never steals the focus"))
    assert cli.cmd_nav(SimpleNamespace(park=True, walk=None, platform=None, tell=False)) == 1
    assert paths.PARK_NOTE.exists() and "isn't in front" in capsys.readouterr().out


def test_nav_park_with_a_hunt_running_sends_nothing(monkeypatch, capsys):
    from gamelab import __main__ as cli
    paths.HUNT.write_text(json.dumps({"pid": 4242, "command": "tour"}))
    monkeypatch.setattr(client, "_alive", lambda pid: True)
    monkeypatch.setattr(cli, "_game_title", lambda: pytest.fail("no further"))
    assert cli.cmd_nav(SimpleNamespace(park=True, walk=None, platform=None, tell=False)) == 1
    assert "`tour` is running" in capsys.readouterr().out


def test_a_minimized_game_a_park_says_so_and_a_pet_feeders_start_waits_never_bringing_it_back(monkeypatch):
    from gamelab import __main__ as cli
    monkeypatch.setattr(cli.window, "find_game", lambda title: 1)
    monkeypatch.setattr(cli.window, "minimized", lambda hwnd: True)
    monkeypatch.setattr(cli.window, "bring_to_front", lambda *a, **k: pytest.fail("she never steals the focus"))
    with pytest.raises(RuntimeError):
        cli._game_geometry("game", restore=False, wait=False)
    pushed = []
    monkeypatch.setattr("gamelab.control.notify.phone", lambda *a, **k: pushed.append(a))
    cli._said_minimized("game")
    assert not pushed  # (nothing wanted)
    paths.HUNT.write_text(json.dumps({"pid": 4242, "command": "tour"}))
    monkeypatch.setattr(client, "_alive", lambda pid: False)
    cli._said_minimized("game")
    assert pushed[0][0] == "She can't park yet" and "minimized" in pushed[0][1] and pushed[0][2]


def test_nav_parks_map_is_the_perception_services_not_a_map_default(tmp_path, monkeypatch):
    from gamelab import __main__ as cli
    from gamelab.localization import service
    info = tmp_path / "perceive.json"
    monkeypatch.setattr(service, "info_path", lambda saved=None: info)
    args = SimpleNamespace(map=10000012)
    assert not cli._park_map(args) and args.map == 10000012  # (no service: it says so, she stays)
    info.write_text(json.dumps({"pid": 1, "map": 10002073}))
    assert cli._park_map(args) and args.map == 10002073


def test_the_guards_switch_by_its_command(capsys):
    from gamelab import __main__ as cli
    assert cli.cmd_guard(SimpleNamespace(mode="alert")) == 0 and g.read_mode() == ALERT
    assert cli.cmd_guard(SimpleNamespace(mode="off")) == 0 and g.read_mode() == OFF
    assert cli.cmd_guard(SimpleNamespace(mode=None)) == 0 and "off" in capsys.readouterr().out.splitlines()[-1]
    assert cli.cmd_guard(SimpleNamespace(mode="on")) == 0 and g.read_mode() == ON and not paths.GUARD_MODE.exists()


# --- the place ---------------------------------------------------------------------------------------------------------

def _tree():
    m = scenarios.real_map(TREE)
    if m is None:
        pytest.skip("needs Tree Dungeon's game data")
    return m


def _place(m, pid, x, monsters=(), her=danger.Her(1244, 35, 117)):
    h = SimpleNamespace(danger=danger.of_map(m, her))
    nav = SimpleNamespace(model=m, planner=Planner(m))
    return park_place(h, nav, SimpleNamespace(where=SimpleNamespace(platform=m.platform(pid), x=x), monsters=list(monsters)))


def test_the_place_is_from_the_scene_a_safe_platform_or_a_safe_rope_however_far():
    m = _tree()
    for pid, x in ((11, -38.5), (19, 309.5), (33, 0.0), (9, 100.0), (23, 200.0)):
        p = _place(m, pid, x)
        assert p.why in ("a safe platform", "a safe rope") and p.seconds > 0 and p.danger == 0.0
    assert _place(m, 26, 150.0).why == "here: a safe platform"


def test_a_map_nothing_can_touch_her_on_she_stays_and_it_says_so():
    t = tree()
    t.pop("life")
    m = mapdata.parse(1, t, "A town", {}, {}, {})
    assert not m.spawn_platforms
    p = _place(m, m.places[0].id, 100.0, her=danger.Her(642, 19))
    assert (p.seconds, p.why) == (0.0, "here: nothing on this map can touch her")


def test_a_map_with_no_safe_place_she_stays_and_it_says_so(monkeypatch):
    m = _tree()
    monkeypatch.setattr(retreat, "wholly_safe", lambda model, pid: False)
    monkeypatch.setattr(danger.Danger, "on_rope", lambda self, rope, y, seen=None: 0.5)
    monkeypatch.setattr(danger.Danger, "at", lambda self, pid, x, seen=None: 0.5)
    p = _place(m, 33, 0.0)
    assert (p.platform, p.x, p.seconds, p.why) == (33, 0.0, 0.0, "here: no safe place on this map")


def test_no_map_id_nor_platform_of_a_map_in_the_place_picking():
    import inspect

    from gamelab.behavior import walk
    assert not hasattr(walk, "PARK")
    for source in (inspect.getsource(park_place), inspect.getsource(retreat.choose), inspect.getsource(retreat.park)):
        assert "10002073" not in source and "TREE" not in source


class _Look:
    """A mover's looks, in turn."""

    def __init__(self, *looks):
        self.looks = list(looks)

    def locate(self, **kw):
        return self.looks.pop(0) if len(self.looks) > 1 else self.looks[0]


def _at(x, pid=19, conf=0.99):
    return SimpleNamespace(where=SimpleNamespace(x=x, platform=SimpleNamespace(id=pid)), confidence=conf)


def test_a_park_sets_off_only_from_a_sure_place(monkeypatch):
    # (2026-11-05 11:18:45: 'carried', 0.32, as she fell past her name tag's reach)
    monkeypatch.setattr(retreat.time, "sleep", lambda s: None)
    nav = SimpleNamespace(mover=_Look(_at(309.5, conf=0.32), _at(250.0), _at(200.0), _at(198.0)))
    assert retreat.sure_look(nav).where.x == 198.0
    now = [0.0]

    def tick():
        now[0] += 1.0
        return now[0]
    monkeypatch.setattr(retreat.time, "monotonic", tick)
    assert retreat.sure_look(SimpleNamespace(mover=_Look(_at(1.0, conf=0.3)))) is None


# --- a retreat's go by go ----------------------------------------------------------------------------------------------

class _Mover:
    """Looks in turn; a rope's way off."""

    def __init__(self, *looks):
        self.looks = list(looks)

    def locate(self, **kw):
        return self.looks.pop(0) if len(self.looks) > 1 else self.looks[0]

    def off_rope(self, s):
        return self.locate()


def _hunted(monkeypatch, *looks, why=None, there=()):
    """`_away` with its navigator faked: `there`, the places chosen go by go."""
    places = list(there)
    nav = SimpleNamespace(mover=_Mover(*looks), model=SimpleNamespace(ropes=[SimpleNamespace(x=-80.0, top=-1469.0, bottom=-1277.0)]),
                          go=lambda *a, **k: SimpleNamespace(result="arrived", short=""),
                          _onto_platform=lambda s: s)
    monkeypatch.setattr(retreat, "Navigator", lambda *a, **k: nav)
    monkeypatch.setattr(retreat, "park_place", lambda h, nav, s, hang=True: places.pop(0))
    monkeypatch.setattr(retreat, "choose", lambda h, nav, s, left, hang=True: places.pop(0))
    monkeypatch.setattr(retreat.time, "sleep", lambda s: None)
    monkeypatch.setattr(retreat.telemetry, "emit", lambda *a, **k: None)
    h = SimpleNamespace(actions=SimpleNamespace(moves_only=lambda: None), senses=None, model=None, log=lambda m: None,
                        danger=None, memory=None)
    return retreat._away(h, time.monotonic(), why or retreat.PARK)


def _on(pid, x=0.0, t=0.0):
    return SimpleNamespace(where=SimpleNamespace(platform=SimpleNamespace(id=pid), x=x, y=0.0), confidence=0.99, t=t, monsters=[], mode="ground")


def _off(x, y, mode, dot=(1.0, 1.0)):
    return SimpleNamespace(where=SimpleNamespace(platform=None, x=x, y=y, dot=dot), confidence=0.99, t=0.0, monsters=[], mode=mode)


def test_knocked_onto_another_safe_place_she_ended_there_and_its_said(monkeypatch):
    # (the review: first go to a rope, knocked onto wholly safe 27, and `stop` said "not parked")
    out = _hunted(monkeypatch, _on(5), _on(5), _on(27), _on(27), there=[
        Place(9, 5.0, "a safe platform", 3.0), Place(27, 0.0, "here: a safe platform", 0.0)])
    assert out.place.platform == 27 and out.there() and out.said().startswith("parked on platform 27")


def test_a_look_not_sure_is_another_go_not_the_end_of_the_park(monkeypatch):
    # (the review: it raised, and one failed look ended the park on 2.7% of the real states)
    looks = [None, _on(27)]
    monkeypatch.setattr(retreat, "sure_look", lambda nav: looks.pop(0))
    out = _hunted(monkeypatch, _on(27), there=[Place(27, 0.0, "here: a safe platform", 0.0)])
    assert out.there() and len(looks) == 0


def test_a_mid_jump_isnt_hanging_and_isnt_there(monkeypatch):
    # (live 2026-11-05 18:14:51: "she stays on the rope at x -80" on looks reading 'air' at x -96..-99, y -1279..-1310; she
    # landed on 11 at x -100, and took MISSes there)
    nav = SimpleNamespace(mover=_Look(_off(-96.0, -1300.0, "air")), model=SimpleNamespace(ropes=[SimpleNamespace(x=-80.0, top=-1469.0, bottom=-1277.0)]))
    assert retreat.hanging(nav, _off(-96.0, -1300.0, "air")) is None
    assert retreat.hanging(nav, _off(-80.0, -1300.0, "rope")) is None  # (a look on 'rope', then one on 'air')
    monkeypatch.setattr(retreat.time, "sleep", lambda s: None)
    nav.mover = _Look(_off(-80.0, -1329.0, "rope"))
    still, rope = retreat.hanging(nav, _off(-80.0, -1329.0, "rope"))
    assert rope.x == -80.0
    out = Retreat(0.0, Place(None, -80.0, "here: a safe rope", 0.0))
    retreat._end(out, 0.0, lambda m: None, None, _off(-96.0, -1300.0, "air"))
    assert out.air and not out.there()
    retreat._end(out, 0.0, lambda m: None, None, _off(-80.0, -1329.0, "rope"))
    assert not out.air and out.there()


def test_ropes_off_she_doesnt_stay_on_one_nor_is_sent_to_one(monkeypatch):
    # (`ROPES` False if her pet's food doesn't work on a rope: a safety stop with nobody there would hang for hours)
    import dataclasses
    went = []
    monkeypatch.setattr(retreat, "hanging", lambda nav, s: (s, SimpleNamespace(x=-80.0)))
    monkeypatch.setattr(retreat, "safe_rope", lambda h, rope, y, seen: Place(None, rope.x, "here: a safe rope", 0.0))
    monkeypatch.setattr(retreat, "rope_risk", lambda *a: 0.0)
    out = _hunted(monkeypatch, _off(-80.0, -1329.0, "rope"), why=retreat.LIE_DETECTOR)
    assert out.result == "stayed"
    monkeypatch.setattr(retreat, "hanging", lambda nav, s: (s, SimpleNamespace(x=-80.0)) if not went else None)
    off = dataclasses.replace(retreat.LIE_DETECTOR, ropes=False)
    out = _hunted(monkeypatch, _off(-80.0, -1329.0, "rope"), _on(11), why=off, there=[Place(11, 0.0, "here: the safest within reach", 0.0)])
    assert out.place.why != "here: a safe rope"
    assert all(w.ropes is retreat.ROPES for w in (retreat.LIE_DETECTOR, retreat.PARK, retreat.SAFETY, retreat.why_of(StopAsked())))


def test_the_players_words_about_another_window_stay_off_the_phone(monkeypatch):
    def refused(*a, **k):
        raise client.PlayerError("game not in front (foreground: 'My secret document')")
    logged = []
    nav = SimpleNamespace(mover=_Mover(_on(5)), model=SimpleNamespace(ropes=[]), go=refused, _onto_platform=lambda s: s)
    monkeypatch.setattr(retreat, "Navigator", lambda *a, **k: nav)
    monkeypatch.setattr(retreat, "choose", lambda h, nav, s, left, hang=True: Place(9, 5.0, "a safe platform", 3.0))
    monkeypatch.setattr(retreat.telemetry, "emit", lambda *a, **k: None)
    h = SimpleNamespace(actions=SimpleNamespace(moves_only=lambda: None), senses=None, model=None, log=logged.append,
                        danger=None, memory=None)
    out = retreat._away(h, time.monotonic(), retreat.LIE_DETECTOR)
    assert "secret" not in out.said() and "secret" not in out.result and any("secret" in m for m in logged)


def test_a_hunt_taking_over_ends_a_park(monkeypatch):
    paths.HUNT.write_text(json.dumps({"pid": 4242, "command": "tour"}))
    out = _hunted(monkeypatch, _on(5))
    assert out.result == "a hunt took over" and not out.there()
    paths.HUNT.write_text(json.dumps({"pid": g.os.getpid(), "command": "tour"}))
    out = _hunted(monkeypatch, _on(27), there=[Place(27, 0.0, "here: a safe platform", 0.0)])
    assert out.there()


def test_any_stops_retreat_and_the_lie_detectors_are_told(monkeypatch):
    monkeypatch.setattr(retreat, "away", lambda *a, **k: Retreat(0.0, Place(27, None, "a safe platform", 3.0), "arrived", 4.0, (27, 80.0)))
    monkeypatch.setattr(retreat, "retreat", lambda *a, **k: None)
    monkeypatch.setattr(retreat.alarm, "page", lambda *a: None)
    from gamelab.behavior.safety import SafetyStop
    h = SimpleNamespace(log=lambda m: None)
    with pytest.raises(SafetyStop):
        with retreat.on_alarm(h):
            raise SafetyStop(300.0, 2.0, 5, 378.0)
    assert json.loads(paths.PARKED.read_text())["there"] and not paths.PARK_NOTE.exists()
    with pytest.raises(client.LieDetectorAlarm):
        with retreat.on_alarm(h):
            raise client.LieDetectorAlarm("seen")
    assert not json.loads(paths.PARKED.read_text())["there"] and paths.PARK_NOTE.exists()  # (none went: the guard, once it's over)


def test_a_parking_height_keeps_clear_of_the_models_edge():
    # (the review, 2026-11-05: the rope at 146, the most-picked, aimed at y 62 with the model's zero from 45, and she ends up
    # within 15 of her aim: live at 47 and 51)
    m = _tree()
    d = danger.every_kind(danger.of_map(m, danger.Her(1244, 35, 117)))
    from gamelab.attack import ropes
    of = lambda pid, *margin: next(r for r in ropes.refuges(m, d, pid, None, *margin) if r.rope.x == 146)
    assert of(30, retreat.ROPE_MARGIN).y - 45.0 >= retreat.ROPE_MARGIN and of(30, retreat.ROPE_MARGIN).danger == 0.0
    assert of(32, retreat.ROPE_MARGIN).y == 62.0 and of(32, retreat.ROPE_MARGIN).danger == 0.0  # (from under, her grab's limit)
    assert of(30).y == 62.0 and of(32).y == 62.0  # (a hunt's, as ever)


# --- in the simulator --------------------------------------------------------------------------------------------------

def _stop_in_a_grind(run_after: float, frames=None):
    """`gamelab stop` `run_after` s into a grind on Tree Dungeon (its request as a file, as live: the fake player cuts no
    plan for it, the command's client.cut_running being another process's): how it stopped, her keys from then on."""
    def fn(sim):
        from gamelab.behavior.grind import Grind
        tour = sim.tour_stack()
        asked = sim.clock.now + run_after
        if frames is not None:
            sim.fail_frames(sim.clock.elapsed + run_after - 2.0, sim.clock.elapsed + 900.0, frames)
        sim.world.call_at(asked, lambda w: paths.STOP_HUNT.write_text("{}"))
        out = {}
        try:
            with retreat.on_alarm(tour.h) as watch:
                Grind(tour, log=sim.log).run(minutes=5)
        except StopAsked as e:
            out.update(later=e.later, after=sim.clock.now - asked, retreat=watch.retreat and watch.retreat.record())
        ends = {q.id: q.end for q in sim.player.plans.values()}
        k = sim.keys
        moves = {k.left.name, k.right.name, k.up.name, k.down.name, k.jump.name}
        out["keys"] = sorted((round(t - asked, 3), kind, *ev) for pid, (kind, events) in sim.player.played.items()
                             for t, *ev in events if asked <= t < ends.get(pid, 1e18))
        out["moves"] = sorted(moves)
        out["safe"] = scenarios.safe_there(sim)
        out["parked"] = json.loads(paths.PARKED.read_text()) if paths.PARKED.exists() else None
        out["note"] = paths.PARK_NOTE.exists()
        out["left"] = paths.STOP_HUNT.exists()
        return out
    import dataclasses
    base = scenarios.SCENARIOS["loop_tree_lie_detector_grind"]
    return scenarios.run_scenario(dataclasses.replace(base, fn=fn, limit=300, seed=1))["result"]


def test_in_the_simulator_a_stop_in_a_grind_her_moves_alone_and_she_parks_and_it_says_so():
    _tree()
    out = _stop_in_a_grind(37.0)
    assert out["later"] is False and out["safe"] and out["retreat"]["why"] in ("a safe platform", "a safe rope")
    assert out["parked"]["there"] and out["parked"]["said"].startswith("parked on") and not out["note"] and not out["left"]
    assert all(e[2] in out["moves"] and e[1] == "retreat" for e in out["keys"] if e[0] > 1.0)  # (the plan playing then, a moment)


def test_in_the_simulator_a_stop_with_the_game_away_ends_the_hunt_in_a_minute_with_a_note_and_no_keys():
    from gamelab.perception.framecheck import Problem
    _tree()
    out = _stop_in_a_grind(37.0, Problem("window", "the game isn't in front"))
    assert out["later"] is True and 55.0 <= out["after"] <= 70.0
    assert out["note"] and not out["left"] and not out["parked"]["there"] and out["retreat"] is None
    assert not [e for e in out["keys"] if e[0] > 1.5]


# --- her potions and her death -----------------------------------------------------------------------------------------

def _vitals():
    pushes = []
    return Vitals(lambda title, text, critical: pushes.append((title, critical))), pushes


def _hp(w, *steps):
    """Her HP number over time, a state 0.5 s before each change too."""
    last = None
    for t, cur in steps:
        if last is not None:
            w.see(last / 1244, 0.5, (last, 1244), t=t - 0.5)
        w.see(cur / 1244, 0.5, (cur, 1244), t=t)
        last = cur


def test_hp_potions_out_two_touches_under_her_pets_threshold_with_no_potion_a_regular_push_once():
    w, pushes = _vitals()
    # a working pet: 739 of 1244, 899 0.13 s on
    _hp(w, (0.0, 949), (9.0, 855), (20.0, 739), (20.13, 899), (40.0, 739), (40.13, 899))
    assert not pushes
    # 2026-11-05 11:26: 744, 754 (regenerating back over her pet's 60%), 644, 538
    _hp(w, (60.0, 899), (62.0, 744), (75.0, 754), (79.0, 644))
    assert not pushes
    _hp(w, (79.0, 644), (83.0, 538))
    assert pushes == [("HP potions not working", False)]
    _hp(w, (83.0, 538), (90.0, 440))
    assert len(pushes) == 1


def test_a_potion_working_again_rearms_it_after_its_time():
    w, pushes = _vitals()
    _hp(w, (0.0, 800), (3.0, 700), (6.0, 600))
    assert len(pushes) == 1
    _hp(w, (10.0, 500), (13.0, 400))  # (it goes on: no more)
    assert len(pushes) == 1
    w.see(600 / 1244, 0.5, (600, 1244), t=13.2)  # (a potion: it works)
    assert w.state["hp"] is None and w.state["told_hp"] == 6.0
    _hp(w, (20.0, 600), (23.0, 500), (26.0, 400))  # (out again at once: not within AGAIN_S)
    assert len(pushes) == 1
    w.see(600 / 1244, 0.5, (600, 1244), t=26.2)
    _hp(w, (AGAIN_S + 10.0, 600), (AGAIN_S + 13.0, 500), (AGAIN_S + 16.0, 400))
    assert len(pushes) == 2


def test_mp_potions_out_under_its_floor_for_a_minute_with_no_potion():
    w, pushes = _vitals()
    # working: the bar reads 0 for a frame before each potion (2026-11-05 07:48: 0.0, then 0.19)
    for t, mp in ((0.0, 0.08), (1.0, 0.0), (1.1, 0.19), (30.0, 0.07), (30.5, 0.0), (30.6, 0.19)):
        w.see(0.9, mp, t=t)
    assert not pushes
    t = 100.0
    while t < 100.0 + MP_OUT_S + 1:
        w.see(0.9, 0.02, t=t)
        t += 1.0
    assert pushes == [("MP potions not working", False)]


def test_a_cast_regained_and_spent_keeps_it_under_and_the_mp_rising_slowly_is_no_potion():
    w, pushes = _vitals()
    for i in range(70):
        w.see(0.9, 0.01 + 0.02 * (i % 3) / 2, t=float(i))
    assert pushes


def test_dead_once_critical_and_alive_again_it_rearms():
    w, pushes = _vitals()
    w.see(0.5, 0.5, (600, 1244), t=0.0)
    w.see(0.0, 0.5, (0, 1244), t=1.0)
    assert w.zero and not pushes  # (a look's misread 0 lasts a frame: told once it's lasted)
    w.see(0.5, 0.5, (600, 1244), t=1.2)
    assert not w.zero and not pushes
    for t in (2.0, 3.0, 2.0 + DEAD_S, 10.0, 3000.0):
        w.see(0.0, 0.5, (0, 1244), t=t)
    assert pushes == [("She died", True)]
    w.see(0.3, 0.5, (400, 1244), t=3100.0)
    for t in (3200.0, 3200.0 + DEAD_S):
        w.see(0.0, 0.5, (0, 1244), t=t)
    assert pushes == [("She died", True), ("She died", True)]


def test_what_was_told_survives_a_new_process():
    w, pushes = _vitals()
    for t in (0.0, DEAD_S):
        w.see(0.0, 0.5, (0, 1244), t=t)
    assert len(pushes) == 1
    again, more = _vitals()
    for t in (100.0, 100.0 + DEAD_S):
        again.see(0.0, 0.5, (0, 1244), t=t)
    assert not more and again.zero


def test_nothing_goes_without_a_push_and_a_state_of_no_bars_says_nothing():
    w = Vitals()
    w.see(None, None, t=1.0)
    assert w.state == {} and not w.zero
    for t in (0.0, DEAD_S):
        w.see(0.0, 0.0, t=t)
    assert w.state["dead"] == DEAD_S


def test_the_pushes_go_as_the_lab_has_them(monkeypatch):
    from gamelab.control import notify
    sent = []
    monkeypatch.setattr(notify, "load_bark", lambda: {"server": "x", "key": "k", "ring": False})
    monkeypatch.setattr(notify, "bark", lambda cfg, title, text, level, log=None, sound=None: sent.append((level, sound)))
    notify.phone("t", "x", True)
    notify.phone("t", "x")
    assert sent == [("critical", notify.BARK_SOUND), ("active", None)]


def test_the_stop_itself_is_a_stop_the_command_line_knows():
    assert issubclass(StopAsked, HuntStop) and StopAsked().exit_code == 9 and StopAsked().page() is None
