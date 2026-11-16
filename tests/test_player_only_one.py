"""One key player at a time (control.player.only_one): a second refuses to start before it touches the inbox, so the
running one's plans, its claimed one's marker among them, aren't taken for left over; `player --elevate` asks for no
UAC click while one runs."""

import json
import os
import sys

import pytest

from gamelab import __main__ as cli
from gamelab.control import client
from gamelab.control import player as player_mod


@pytest.mark.skipif(sys.platform != "win32", reason="a Windows named mutex")
def test_a_second_hold_of_the_players_mutex_is_refused():
    name = f"Local\\gamelab-player-test-{os.getpid()}"
    first = player_mod.only_one(name)
    try:
        assert first not in (None, True)
        assert player_mod.only_one(name) is None
    finally:
        import ctypes
        ctypes.WinDLL("kernel32").CloseHandle(first)
    second = player_mod.only_one(name)  # (the first let go: free again)
    assert second not in (None, True)
    ctypes.WinDLL("kernel32").CloseHandle(second)


def test_a_second_player_leaves_the_inbox_and_the_log_alone(monkeypatch, tmp_path):
    inbox, outbox, log = tmp_path / "inbox", tmp_path / "outbox", tmp_path / "player.log"
    inbox.mkdir()
    running, queued = inbox / "a.running", inbox / "b.json"
    running.write_text("{}", encoding="utf-8")
    queued.write_text("{}", encoding="utf-8")
    info = tmp_path / "player.json"
    info.write_text(json.dumps({"pid": 4242}), encoding="utf-8")
    for name, value in (("LAB", tmp_path), ("INBOX", inbox), ("OUTBOX", outbox), ("PLAYER_LOG", log),
                        ("PLAYER_INFO", info), ("STOP", tmp_path / "STOP"), ("ALARM", tmp_path / "ALARM.json")):
        monkeypatch.setattr(player_mod, name, value)
    monkeypatch.setattr(player_mod, "only_one", lambda name=player_mod.MUTEX: None)
    monkeypatch.setattr(player_mod, "_rotate_log", lambda *a: pytest.fail("the running player's log moved"))
    monkeypatch.setattr(player_mod, "Watchdog", lambda *a, **k: pytest.fail("a second watchdog started"))
    player_mod.serve()
    assert running.exists() and queued.exists()
    assert info.read_text(encoding="utf-8") == json.dumps({"pid": 4242})
    said = log.read_text(encoding="utf-8")
    assert "refusing to start: another player is running (pid 4242)" in said
    assert "player stopped" not in said  # (`stop-player` reads that line as the running one's end)


def _elevate(monkeypatch, up: bool) -> list:
    """`player --elevate` with a player running or not (player.running), its ShellExecuteW calls kept."""
    monkeypatch.setattr(client, "check_alarm", lambda: None)
    monkeypatch.setattr(player_mod, "running", lambda name=None: up)
    shell = []
    monkeypatch.setattr(cli.ctypes, "windll", type("W", (), {"shell32": type("S", (), {
        "ShellExecuteW": staticmethod(lambda *a: shell.append(a) or 42)})})(), raising=False)
    assert cli.cmd_player(type("A", (), {"elevate": True})()) == 0
    return shell


def test_player_elevate_asks_no_uac_click_while_one_runs(monkeypatch, capsys):
    assert _elevate(monkeypatch, up=True) == []
    assert "a player is already running" in capsys.readouterr().out


def test_player_elevate_asks_for_the_uac_click_with_none_running(monkeypatch, capsys):
    shell = _elevate(monkeypatch, up=False)
    assert len(shell) == 1 and shell[0][1] == "runas" and shell[0][3] == "-X utf8 -m gamelab player"
    assert "player starting as administrator" in capsys.readouterr().out


@pytest.mark.skipif(sys.platform != "win32", reason="a Windows named mutex")
def test_running_asks_the_mutex_not_the_players_file(monkeypatch, tmp_path):
    """A player.json left behind, its pid now another process's (this one's), isn't a player running."""
    info = tmp_path / "player.json"
    info.write_text(json.dumps({"pid": os.getpid(), "started": 0}), encoding="utf-8")
    monkeypatch.setattr(client, "PLAYER_INFO", info)
    assert client.player_up() and not player_mod.running()
    held = player_mod.only_one(player_mod.MUTEX)
    try:
        assert player_mod.running()
    finally:
        player_mod._let_go(held)
    assert not player_mod.running()


@pytest.mark.skipif(sys.platform != "win32", reason="a Windows named mutex")
def test_a_player_that_ended_lets_its_mutex_go(monkeypatch, tmp_path):
    for name, value in (("LAB", tmp_path), ("INBOX", tmp_path / "inbox"), ("OUTBOX", tmp_path / "outbox"),
                        ("PLAYER_LOG", tmp_path / "player.log"), ("STOP", tmp_path / "STOP")):
        monkeypatch.setattr(player_mod, name, value)
    monkeypatch.setattr(player_mod, "alarm_up", lambda: True)  # (it ends at once: "refusing to start")
    player_mod.serve()
    assert "a Lie Detector alarm hasn't been cleared" in (tmp_path / "player.log").read_text(encoding="utf-8")
    again = player_mod.only_one(player_mod.MUTEX)
    assert again not in (None, True)
    player_mod._let_go(again)
