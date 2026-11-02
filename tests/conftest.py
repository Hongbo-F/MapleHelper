import copy
import os
from pathlib import Path

import pytest

# Her stats as measured, not config.toml's [character] (gamelab.motion.physics.FIXED_STATS): set before
# anything imports gamelab, which works out her walk and jumps from them once, when it's first imported.
os.environ["GAMELAB_FIXED_STATS"] = "1"
os.environ.pop("GAMELAB_CHARACTER", None)  # ... and the thief's, whoever the shell names (gamelab.character.ENV)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # 3 GB a process otherwise (gamelab/__init__.py)
os.environ.setdefault("GAMELAB_GPU", "0")  # the re-finds' GPU off unless a test turns it on, the lab's GPU file or not
# No Lie Detector alert or text to the user's phone from the lab's bark.json and smtp.json (gamelab.control.notify
# .NO_PUSH_ENV): the suite in the main checkout sent one from a test's fake screen on 2026-10-29 ~05:35.
os.environ["GAMELAB_NO_PUSH"] = "1"

# A fixed config for tests, so editing config.toml for her keys never breaks them.
BASE_CONFIG = {
    "game_window_title": "MapleStory",
    "keys": {"left": "left", "right": "right", "jump": "alt", "attack": "ctrl",
             "lucky_seven": "shift"},
    "hotkeys": {"jump_throw": "f1", "back_off": "f2", "hop_back": "f3", "jump_back_shot": "f5"},
    "control": {"abort": "esc", "slow_mode": "f12"},
    "timing": {"throw_gap": 650},
}


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true",
                     help="also run the tests marked slow: minutes more of simulation, for before a commit")


def pytest_xdist_auto_num_workers(config):
    """-n auto (pyproject.toml's addopts): 8 workers for the suite or a folder of it, which take 10 minutes on one;
    one process when the run names its files or tests, as a worker takes about 4 s to start. 8, not one a core: each
    commits about half a GB."""
    return 0 if all(Path(arg.split("::")[0]).is_file() for arg in config.args) else 8


def pytest_collection_modifyitems(config, items):
    """Without --slow the tests marked slow aren't run: the long simulations and sweeps (README: Tests)."""
    if config.getoption("--slow"):
        return
    slow = [item for item in items if item.get_closest_marker("slow")]
    if slow:
        config.hook.pytest_deselected(items=slow)
        items[:] = [item for item in items if not item.get_closest_marker("slow")]


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
                 "minimized", "on_screen", "over", "overs", "describe", "window_rect", "inset", "monitor_of", "full_screen"):
        monkeypatch.setattr(window, name, refuse(f"gamelab.window.{name}"))
    monkeypatch.setattr(capture, "Dxgi", refuse("gamelab.capture.Dxgi (desktop duplication)"))
    monkeypatch.setattr(capture, "_open_mss", refuse("gamelab.capture's mss"))
    monkeypatch.setattr(capture, "_SHARED", None)  # no grabber outlives its test
    monkeypatch.setattr(client, "INBOX", tmp_path / "inbox")  # plans go nowhere a player looks
    monkeypatch.setattr(client, "OUTBOX", tmp_path / "outbox")
    # Nor is the lab's own Lie Detector alarm read: a worktree whose lab was copied while the alarm was up had every
    # plan a test sent refused (and serve() refused to start).
    from gamelab.control import player
    for mod in (paths, alarm, client, player):
        monkeypatch.setattr(mod, "ALARM", tmp_path / "ALARM.json")
    monkeypatch.setattr(alarm, "page", refuse("gamelab.control.alarm.page"))
    from gamelab.localization import service
    monkeypatch.setattr(service, "_start_detached", refuse("gamelab.localization.service._start_detached (the "
                                                           "perception service started)"))
    # The size fix's tries an hour and its lock are the test's own (a worker's test beside another's took its mutex).
    from gamelab.control import window_size
    monkeypatch.setattr(window_size, "WINDOW_TRIES", tmp_path / "window_tries.json")
    monkeypatch.setattr(window_size, "LOCK", f"Local\\gamelab-window-fix-test-{os.getpid()}-{tmp_path.name}")
    # Nor the recorders' mutex (the one `gamelab recorder` or the perception service's recording holds on the PC).
    from gamelab.control import rolling
    monkeypatch.setattr(rolling, "MUTEX", f"Local\\gamelab-rolling-recorder-test-{os.getpid()}-{tmp_path.name}")
    # Nor `gamelab character use`'s choice of who plays (the tests play the thief unless they say otherwise), and
    # who plays as a test resolves her is its own: the process's is the thief's, her physics loaded as hers first.
    from gamelab import character
    from gamelab.motion import physics  # noqa: F401 (character.active: the thief, whatever a test plays later)
    monkeypatch.setattr(character, "CHOSEN", tmp_path / "character.json")
    monkeypatch.setattr(character, "_ACTIVE", list(character._ACTIVE))
    # Nor her touches as her HP number measured them live (lab/touches.jsonl: the safety stop's reserve rises with them):
    # a test's hunter has none but its own.
    monkeypatch.setattr(paths, "TOUCHES", tmp_path / "touches.jsonl")
    # ... nor her stats as perception read them (lab/her_stats.json: her DEF kept), behavior.bootstrap.
    monkeypatch.setattr(paths, "STATS", tmp_path / "her_stats.json")
    # ... nor the stat window's live calibration (lab/stat_window.json: a hunt opens the window only with one),
    # behavior.statcheck, nor a remembered key binding's stats key
    monkeypatch.setattr(paths, "STAT_WINDOW", tmp_path / "stat_window.json")
    monkeypatch.setattr(paths, "STAT_TEMPLATES", tmp_path / "stat_window")
    # ... nor her damage as the bars showed it live (lab/characters/<hers>/damage.json, behavior.damage): a test's hunter
    # learns from its own looks alone, its fits made at once (the same every run)
    from gamelab.behavior import damage
    monkeypatch.setattr(paths, "DAMAGE", tmp_path / "damage.json")
    monkeypatch.setattr(paths.Files, "damage", property(
        lambda self: paths.DAMAGE if not self.folder else tmp_path / f"damage-{self.folder}.json"), raising=False)
    monkeypatch.setattr(damage, "THREADED", False)
    for obj, name, value in _write_guards(paths.LAB):
        monkeypatch.setattr(obj, name, value)
