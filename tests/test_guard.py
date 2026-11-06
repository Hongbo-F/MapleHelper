"""A stop that parks and a guard between hunts (behavior.guard; behavior.retreat.park_place, told; behavior.safety.StopAsked),
her potions and her death on the phone (behavior.vitals). The user, 2026-11-05: "you should build both, but the platform
picking should be dynamically chosen based on the current map"."""

import json
import math
import time
from datetime import datetime
from types import SimpleNamespace

import pytest

from gamelab import paths
from gamelab.attack import danger
from gamelab.behavior import guard as g
from gamelab.behavior import retreat, vitals as v
from gamelab.behavior.guard import ALERT, EPISODE_S, HAND_S, OFF, ON, PARK_AGAIN_S, PARK_MOST, Guard, Pass, Seen
from gamelab.behavior.retreat import Place, Retreat, park_place
from gamelab.behavior.safety import HuntStop, StopAsked
from gamelab.behavior.vitals import AGAIN_S, DEAD_S, MP_OUT_S, Vitals
from gamelab.control import client
from gamelab.localization import mapdata
from gamelab.motion.route import Planner
from gamelab.sim import scenarios

from test_sim import tree

TREE = scenarios.TREE


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
    assert gd.act(Seen(note=True)) == "park"
    paths.PARK_NOTE.write_text("{}")
    now.t += PARK_AGAIN_S
    assert gd.act(Seen(note=True, by_hand=True)) is None and not paths.PARK_NOTE.exists() and len(parks) == 1


# --- played by hand: Windows' last input, against the player's own keys ------------------------------------------------

def test_input_counts_only_with_the_game_in_front_within_the_time_and_later_than_the_players_last_key():
    now = 5000.0
    assert g.played_by_hand(now, True, 3.0, None)
    assert g.played_by_hand(now, True, HAND_S, now - 200.0)
    assert not g.played_by_hand(now, False, 3.0, None)
    assert not g.played_by_hand(now, True, HAND_S + 1, None)
    assert not g.played_by_hand(now, True, None, None)
    assert not g.played_by_hand(now, True, 5.0, now - 4.0)  # the player's key, not a hand: it updates the same clock
    assert g.played_by_hand(now, True, 5.0, now - 50.0)


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


def test_a_stop_cuts_what_the_player_is_playing_and_withdraws_what_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "INBOX", tmp_path)
    (tmp_path / "a_flow.running").write_text("")
    (tmp_path / "b_throw.json").write_text("{}")
    (tmp_path / "a_flow.splice").write_text("{}")
    client.cut_running()
    assert (tmp_path / "a_flow.cancel").exists() and not (tmp_path / "b_throw.json").exists()
    assert not (tmp_path / "a_flow.splice").exists()


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
    with pytest.raises(ValueError):
        retreat.sure_look(SimpleNamespace(mover=_Look(_at(1.0, conf=0.3))))


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
    assert math.isinf(math.inf)
