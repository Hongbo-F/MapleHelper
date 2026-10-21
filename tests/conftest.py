import copy
import os

import pytest

# Her stats as measured, not config.toml's [character] (gamelab.motion.physics.FIXED_STATS): set before
# anything imports gamelab, which works out her walk and jumps from them once, when it's first imported.
os.environ["GAMELAB_FIXED_STATS"] = "1"

# A fixed config for tests, so editing config.toml for her keys never breaks them.
BASE_CONFIG = {
    "game_window_title": "MapleStory",
    "keys": {"left": "left", "right": "right", "jump": "alt", "attack": "ctrl",
             "lucky_seven": "shift"},
    "hotkeys": {"jump_throw": "f1", "back_off": "f2", "hop_back": "f3", "jump_back_shot": "f5"},
    "control": {"abort": "esc", "slow_mode": "f12"},
    "timing": {"throw_gap": 650},
}


@pytest.fixture
def config_data():
    return copy.deepcopy(BASE_CONFIG)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """No test may press a key, touch the game window, capture the screen (DXGI desktop duplication or mss),
    send a plan to the player, page anyone, or write under the real gamelab/lab (learned stats, logs, the Lie
    Detector alarm). A test that needs one of these replaces it with its own fake, which takes precedence."""
    from maplehelper import win32

    from gamelab import capture, paths, window
    from gamelab.control import alarm
    from gamelab.control import client
    from gamelab.sim.fakes import _write_guards

    def refuse(what):
        def fn(*a, **k):
            raise AssertionError(f"{what} called in a test")
        return fn

    monkeypatch.setattr(win32, "send_key", refuse("maplehelper.win32.send_key"))
    monkeypatch.setattr(win32, "foreground_title", refuse("maplehelper.win32.foreground_title"))
    for name in ("bring_to_front", "move_cursor", "find_game", "client_rect", "foreground_title", "shows",
                 "minimized"):
        monkeypatch.setattr(window, name, refuse(f"gamelab.window.{name}"))
    monkeypatch.setattr(capture, "Dxgi", refuse("gamelab.capture.Dxgi (desktop duplication)"))
    monkeypatch.setattr(capture, "_open_mss", refuse("gamelab.capture's mss"))
    monkeypatch.setattr(capture, "_SHARED", None)  # no grabber outlives its test
    monkeypatch.setattr(client, "INBOX", tmp_path / "inbox")  # plans go nowhere a player looks
    monkeypatch.setattr(client, "OUTBOX", tmp_path / "outbox")
    monkeypatch.setattr(alarm, "page", refuse("gamelab.control.alarm.page"))
    for obj, name, value in _write_guards(paths.LAB):
        monkeypatch.setattr(obj, name, value)


@pytest.fixture
def old_route_takeoffs(monkeypatch):
    """The route's running jumps as they were before their takeoffs were placed for room (route._placed, from the
    user's watching on 2026-10-21), as the game played with them on the night of the live fixes: off an end looked
    for and left to flow from route.JUMP_FROM back (flow takes off JUMP_FROM to JUMP_FROM_FAR from the end), its
    run from RUN_START back; along a platform from where it was found, its run RUN_START - JUMP_FROM before it.
    The map's 81 route jumps of then, exactly (test_flow). For the tests that pin what a fix of that night does at
    the takeoffs it was found with."""
    from gamelab import directions
    from gamelab.motion import route
    state = {"end": False}
    end_moves = route._end_moves

    def ends(model, a):
        state["end"] = True
        try:
            return end_moves(model, a)
        finally:
            state["end"] = False

    def placed(model, a, x0, sign, land):
        jump = route._sure_running_jump(model, a, x0, sign)
        if jump is None or jump[0] is not land:
            return None
        edge = a.right if sign > 0 else a.left
        start = edge - sign * route.RUN_START if state["end"] else x0 - sign * (route.RUN_START - route.JUMP_FROM)
        if not a.covers(start):
            return None
        return route.Move("jump", a.id, land.id, start, jump[1], directions.toward(sign),
                          seconds=route._running_jump_seconds(jump[2]), takeoff=None if state["end"] else x0)

    monkeypatch.setattr(route, "_end_moves", ends)
    monkeypatch.setattr(route, "_placed", placed)
