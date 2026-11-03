"""A plan that takes over from the running one (a splice: control.client.splice, control.player's Player.run): the
navigator's chains replanned while one runs (control.navigator.REPLAN) are handed over with no key let go between."""

import json
import os
import threading
import time

import pytest

from gamelab.control import client
from gamelab.control import player as player_mod
from gamelab.control.alarm import LIE_DETECTOR_ERROR
from gamelab.control.player import Alarm, Player, allowed_keys


def _player(monkeypatch, tmp_path, flag=None, keys=None):
    """A Player whose keys go to `sent` (name, down) and whose inbox and outbox are tmp_path's, the game in front."""
    from maplehelper.keys import parse
    sent = []
    inbox, outbox = tmp_path / "inbox", tmp_path / "outbox"
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    for mod in (player_mod, client):
        monkeypatch.setattr(mod, "INBOX", inbox)
        monkeypatch.setattr(mod, "OUTBOX", outbox)
    monkeypatch.setattr(player_mod, "ALARM", tmp_path / "ALARM.json")
    monkeypatch.setattr(client, "ALARM", tmp_path / "ALARM.json")
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "client_rect",
                        lambda hwnd: {"left": 0, "top": 0, "width": 1366, "height": 768})
    monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "Game")
    monkeypatch.setattr(player_mod.win32, "send_key", lambda key, down: sent.append((key.name, down)))
    allowed = allowed_keys(keys) if keys is not None else {k: parse(k) for k in ("right", "left", "alt", "z", "f")}
    return Player(allowed, "Game", flag, keys=keys, log=lambda m: None), sent, inbox, outbox


def _splice_later(inbox, running: str, delay: float, plan: dict) -> threading.Timer:
    """Write a splice into plan `running` after `delay` s, as client.splice does (its tmp file, then in place)."""
    def put():
        tmp = inbox / f"{running}.splice.tmp"
        tmp.write_text(json.dumps(plan), encoding="utf-8")
        tmp.replace(inbox / f"{running}.splice")
    timer = threading.Timer(delay, put)
    timer.start()
    return timer


def test_a_splice_takes_over_at_its_time_with_the_arrow_held_kept_down(monkeypatch, tmp_path):
    """Mid-plan, a splice into it takes over at its at_ms of the running plan's clock: the running plan's keys after it
    don't go, the arrow both hold isn't let go (no 'up' for it, no second 'down'), a key the running plan holds that the
    new one doesn't is let go, and the new plan's events play from then on (its clock's start the splice's time). The
    running plan is answered at once with SPLICED (the new id, its at_ms, how late), the new one as any (the last plan's
    result, its id for serve), each with its own .running while it plays."""
    p, sent, inbox, outbox = _player(monkeypatch, tmp_path)
    old = {"id": "old", "events": [[0, "right", "down"], [100, "z", "down"], [400, "z", "up"], [900, "right", "up"]],
           "capture": {"lead_ms": 0}}
    new = {"id": "new", "at_ms": 250, "held": ["right"], "events": [[0, "right", "down"], [100, "alt", "down"],
                                                                     [140, "alt", "up"], [300, "right", "up"]],
           "capture": {"lead_ms": 0}}
    (inbox / "old.running").write_text("")
    _splice_later(inbox, "old", 0.12, new)
    started = time.perf_counter()
    result = p.run(old)
    took = time.perf_counter() - started
    assert result["id"] == "new" and result["problems"] == []
    assert [e[1:] for e in result["sent"]] == [["right", "down"], ["alt", "down"], ["alt", "up"], ["right", "up"]]
    assert sent == [("right", True), ("z", True), ("z", False), ("alt", True), ("alt", False), ("right", False)]
    assert 0.5 <= took < 0.8  # (250 ms into the old plan, then the new one's 300)
    answered = json.loads((outbox / "old.json").read_text())
    assert answered["problems"] == [client.SPLICED] and answered["spliced"]["to"] == "new"
    assert answered["spliced"]["at_ms"] == 250 and answered["spliced"]["late_ms"] < 20
    assert [e[1:] for e in answered["sent"]] == [["right", "down"], ["z", "down"]]
    assert not (inbox / "old.running").exists() and (inbox / "new.running").exists()  # (serve answers the new one)
    assert not (inbox / "old.splice").exists()


def test_a_splice_at_a_plans_end_carries_it_on_with_no_key_let_go_between(monkeypatch, tmp_path):
    """A splice whose time is the running plan's end (or past it): taken as its continuation, the keys held until it, so
    the arrow held through the end isn't let go (the end_ms queue let go of every key between two plans)."""
    p, sent, inbox, outbox = _player(monkeypatch, tmp_path)
    old = {"id": "old", "events": [[0, "right", "down"]], "end_ms": 200, "capture": {"lead_ms": 0}}
    new = {"id": "next", "at_ms": 260, "held": ["right"], "events": [[100, "right", "up"]], "capture": {"lead_ms": 0}}
    (inbox / "old.running").write_text("")
    _splice_later(inbox, "old", 0.05, new)
    started = time.perf_counter()
    result = p.run(old)
    assert result["id"] == "next" and 0.33 <= time.perf_counter() - started < 0.6
    assert sent == [("right", True), ("right", False)]  # (down once, up once: nothing let go at the old plan's end)
    assert json.loads((outbox / "old.json").read_text())["spliced"]["to"] == "next"


def test_a_splice_seen_late_takes_over_at_once_and_says_how_late(monkeypatch, tmp_path):
    p, sent, inbox, outbox = _player(monkeypatch, tmp_path)
    old = {"id": "old", "events": [[0, "right", "down"], [600, "right", "up"]], "capture": {"lead_ms": 0}}
    new = {"id": "late", "at_ms": 50, "held": ["right"], "events": [[0, "right", "down"], [50, "right", "up"]],
           "capture": {"lead_ms": 0}}
    (inbox / "old.running").write_text("")
    _splice_later(inbox, "old", 0.2, new)
    p.run(old)
    late = json.loads((outbox / "old.json").read_text())["spliced"]["late_ms"]
    assert 120 <= late < 260, late
    assert sent == [("right", True), ("right", False)]


def test_a_malformed_or_disallowed_splice_is_refused_and_the_running_plan_plays_on(monkeypatch, tmp_path):
    """A splice is checked as any plan (player.validate: its keys allowed, its times sorted): refused, it's answered with
    the error under its own id and dropped, and the running plan plays on to its end as it was."""
    p, sent, inbox, outbox = _player(monkeypatch, tmp_path)
    old = {"id": "old", "events": [[0, "right", "down"], [300, "right", "up"]], "capture": {"lead_ms": 0}}
    bad = {"id": "bad", "at_ms": 100, "held": [], "events": [[0, "ctrl", "down"], [40, "ctrl", "up"]],
           "capture": {"lead_ms": 0}}
    (inbox / "old.running").write_text("")
    _splice_later(inbox, "old", 0.03, bad)
    result = p.run(old)
    assert "id" not in result and result["problems"] == []
    assert sent == [("right", True), ("right", False)]
    assert "not allowed" in json.loads((outbox / "bad.json").read_text())["error"]


def test_the_alarm_stops_a_spliced_plan_and_with_it_up_no_splice_but_its_kinds_is_taken(monkeypatch, tmp_path):
    """The Lie Detector's alarm halts a spliced plan as any (every key let go), and a splice that comes with the alarm up
    is refused unless it's of a kind that goes then (alarm.PLAN_KINDS: alarm_refusal), so nothing else is pressed."""
    flag = Alarm()
    p, sent, inbox, outbox = _player(monkeypatch, tmp_path, flag)
    old = {"id": "old", "events": [[0, "right", "down"], [2000, "right", "up"]], "capture": {"lead_ms": 0}}
    new = {"id": "new", "at_ms": 100, "held": ["right"], "events": [[0, "right", "down"], [1500, "right", "up"]],
           "capture": {"lead_ms": 0}}
    (inbox / "old.running").write_text("")
    _splice_later(inbox, "old", 0.03, new)
    threading.Timer(0.3, flag.set).start()
    result = p.run(old)
    assert result["id"] == "new" and result["error"] == LIE_DETECTOR_ERROR
    assert sent == [("right", True), ("right", False)]  # (let go at the alarm, nothing after)
    # With the alarm up already: a splice of no kind is refused, nothing pressed
    p2, sent2, inbox2, outbox2 = _player(monkeypatch, tmp_path / "two")
    p2.alarm.set()
    refused = p2._splice_refusal({"id": "x", "events": [[0, "right", "down"], [40, "right", "up"]],
                                  "capture": {"lead_ms": 0}}, during=True)
    assert refused == LIE_DETECTOR_ERROR and sent2 == []


def test_the_client_hands_a_splice_over_and_withdraws_one_not_taken(monkeypatch, tmp_path):
    """client.splice writes `{running}.splice` (no *.json: never a plan of its own), refused with the alarm up as `send`
    is; cancelling a splice the player hasn't taken withdraws it and cuts the plan it was to take over from short;
    `taken` reads that plan's result: SPLICED to it, taken; over without it, never (the splice withdrawn)."""
    p, sent, inbox, outbox = _player(monkeypatch, tmp_path)
    monkeypatch.setattr(client, "check_alarm", lambda: None)
    (inbox / "a.running").write_text("")
    sid = client.splice("a", [[0, "right", "down"]], 120, ["right"], end_ms=300, name="flow")
    assert (inbox / "a.splice").exists() and not list(inbox.glob("*.json"))
    plan = json.loads((inbox / "a.splice").read_text())
    assert plan["id"] == sid and plan["at_ms"] == 120 and plan["held"] == ["right"] and plan["end_ms"] == 300
    assert client.taken(sid) is None  # (its plan still runs)
    assert client.cancel(sid) is True  # withdrawn: no result will come for it ...
    assert not (inbox / "a.splice").exists() and (inbox / "a.cancel").exists()  # ... and the plan it was for cut
    (outbox / "a.json").write_text(json.dumps({"sent": [], "problems": ["cancelled"]}))
    assert client.taken(sid) is False
    # Taken: the plan it was for answered SPLICED to it
    (inbox / "b.running").write_text("")
    sid2 = client.splice("b", [], 0, [], name="flow")
    (inbox / "b.splice").unlink()  # (the player claimed it)
    (outbox / "b.json").write_text(json.dumps({"sent": [], "problems": [client.SPLICED], "spliced": {"to": sid2}}))
    assert client.taken(sid2) is True
    # Over first, never taken: the splice withdrawn
    (inbox / "c.running").write_text("")
    sid3 = client.splice("c", [], 0, [], name="flow")
    (outbox / "c.json").write_text(json.dumps({"sent": [], "problems": []}))
    assert client.taken(sid3) is False and not (inbox / "c.splice").exists()
    monkeypatch.setattr(client, "check_alarm", lambda: (_ for _ in ()).throw(client.LieDetectorAlarm("up")))
    with pytest.raises(client.LieDetectorAlarm):
        client.splice("c", [], 0, [])


def test_can_splice_follows_the_players_info_file(monkeypatch, tmp_path):
    """client.can_splice: a player that takes splices says so in paths.PLAYER_INFO as it starts (its pid, alive); none,
    a dead pid, or one that doesn't say splice: no (the navigator cuts and sends at once)."""
    info = tmp_path / "player.json"
    monkeypatch.setattr(client, "PLAYER_INFO", info)
    monkeypatch.setattr(client, "_INFO", {"stamp": None, "can": False})
    assert client.can_splice() is False
    info.write_text(json.dumps({"splice": 1, "pid": os.getpid(), "started": time.time()}))
    assert client.can_splice() is True
    time.sleep(0.02)
    info.write_text(json.dumps({"splice": 1, "pid": 2 ** 30, "started": time.time(), "x": 1}))  # (no such process)
    assert client.can_splice() is False
    info.unlink()
    assert client.can_splice() is False


def test_a_splice_during_a_mouse_trail_takes_over_before_the_next_event(monkeypatch, tmp_path):
    """A trail waits through the plan's own wait: a splice due during it takes over before the old plan's next key."""
    p, sent, inbox, outbox = _player(monkeypatch, tmp_path)
    old = {"id": "old", "events": [[0, "mouse", 100, 100], [0, "f", "down"]], "capture": {"lead_ms": 0}}
    new = {"id": "new", "at_ms": 0, "held": [], "events": [[0, "alt", "down"], [40, "alt", "up"]],
           "capture": {"lead_ms": 0}}
    (inbox / "old.running").write_text("")
    (inbox / "old.splice").write_text(json.dumps(new))
    monkeypatch.setattr(player_mod.window, "move_cursor", lambda x, y, wait: wait(time.perf_counter()))

    result = p.run(old)

    assert result["id"] == "new" and result["problems"] == []
    assert sent == [("alt", True), ("alt", False)]  # the old plan's f never goes after its interrupted mouse move
    answered = json.loads((outbox / "old.json").read_text())
    assert answered["sent"] == [] and answered["problems"] == [client.SPLICED]
