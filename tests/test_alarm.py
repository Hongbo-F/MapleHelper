"""Lie Detector auto-stop: spotting it, stopping the player, and refusing to run until cleared."""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab.control import alarm, client
from gamelab.perception import lie_detector
from gamelab.perception.lie_detector import LieDetectorSpotter
from gamelab.__main__ import LIE_DETECTOR_EXIT, main
from gamelab.control.player import LIE_DETECTOR_ERROR, Player, Watchdog, _sleep_until

FIXTURES = Path(__file__).parent / "fixtures"


def img(name):
    return np.asarray(Image.open(FIXTURES / name).convert("RGB"))


SCENE = img("game_scene.png")            # daffodils and other yellow scenery
WINDOW = img("lie_detector_window.png")  # the real Lie Detector countdown window (top part)
NEAR_MISS = img("near_miss.png")         # the key bindings window's orange CANCEL button


def scene_with_window():
    frame = np.zeros((700, 1100, 3), np.uint8)
    frame[:SCENE.shape[0], :SCENE.shape[1]] = SCENE
    frame[300:300 + WINDOW.shape[0], 600:600 + WINDOW.shape[1]] = WINDOW
    return frame


@pytest.fixture
def alarm_file(tmp_path, monkeypatch):
    path = tmp_path / "ALARM.json"
    monkeypatch.setattr(alarm, "ALARM", path)
    monkeypatch.setattr(client, "ALARM", path)
    return path


def test_spots_the_lie_detector_anywhere_on_screen():
    spotter = LieDetectorSpotter()
    assert spotter.seen(scene_with_window())
    assert spotter.score(WINDOW) > 0.95


def test_spots_the_mouse_mini_game_and_the_truth_booth():
    # The mini-game was only captured at half size: enlarge it back to the game's size.
    half = img("lie_detector_minigame_half.png")
    mini_game = np.asarray(Image.fromarray(half).resize((half.shape[1] * 2, half.shape[0] * 2), Image.BILINEAR))
    spotter = LieDetectorSpotter()
    assert spotter.scores(mini_game)["mini-game"] >= 1
    assert spotter.seen(mini_game)
    assert spotter.scores(img("truth_booth_timer.png"))["truth booth"] >= 1


SEEN_IN_GAME = sorted((FIXTURES / "lie_detector").glob("*.png"))  # <date>_<time>_<form>.png, see its README
FORMS = {"countdown": "countdown", "mini_game": "mini-game", "truth_booth": "truth booth"}


@pytest.mark.parametrize("png", SEEN_IN_GAME, ids=lambda p: p.stem)
def test_spots_every_lie_detector_seen_in_game(png):
    form = FORMS[png.stem.split("_", 2)[2]]
    assert LieDetectorSpotter().scores(img(png))[form] >= 1


@pytest.mark.parametrize("png", SEEN_IN_GAME, ids=lambda p: p.stem)
def test_spots_every_lie_detector_seen_in_game_at_the_game_s_own_size_too(png):
    # Since 2026-10-23 the game's window is 1366x768, pixel for pixel: the 1920x1080 screens shrunk to it.
    form = FORMS[png.stem.split("_", 2)[2]]
    native = np.asarray(Image.fromarray(img(png)).resize((1366, 768), Image.BOX))
    assert LieDetectorSpotter().scores(native)[form] >= 1


def test_the_lie_detectors_seen_in_game_are_all_there():
    assert len(SEEN_IN_GAME) >= 6


@pytest.mark.parametrize("frame", [SCENE, NEAR_MISS, np.zeros((540, 960, 3), np.uint8)])
def test_ignores_yellow_scenery_and_other_windows(frame):
    assert not LieDetectorSpotter().seen(frame)


def numpy_yellow(rgb):
    """lie_detector.yellow as it was: numpy's tests on int16 copies."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return ((r > 180) & (g > 140) & (b < 110) & (r - b > 110)).astype(np.float32)


def test_the_title_yellow_is_the_same_as_numpys_on_every_colour():
    g, b = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8), indexing="ij")
    for r0 in range(0, 256, 16):
        picture = np.empty((16 * 256, 256, 3), np.uint8)
        picture[..., 0] = np.repeat(np.arange(r0, r0 + 16, dtype=np.uint8), 256)[:, None]
        picture[..., 1], picture[..., 2] = np.tile(g, (16, 1)), np.tile(b, (16, 1))
        new = lie_detector.yellow(picture)
        assert new.dtype == np.float32 and np.array_equal(new, numpy_yellow(picture))


def test_spotter_scores_are_the_same_as_with_numpys_yellow(monkeypatch):
    half = img("lie_detector_minigame_half.png")
    mini_game = np.asarray(Image.fromarray(half).resize((half.shape[1] * 2, half.shape[0] * 2), Image.BILINEAR))
    frames = [scene_with_window(), WINDOW, mini_game, img("truth_booth_timer.png"), SCENE, NEAR_MISS,
              np.zeros((540, 960, 3), np.uint8)]
    new = [LieDetectorSpotter().scores(f) for f in frames]
    monkeypatch.setattr(lie_detector, "yellow", numpy_yellow)
    old = LieDetectorSpotter()
    assert new == [old.scores(f) for f in frames]


def test_watchdog_grabs_the_game_as_an_rgb_picture_of_its_own(monkeypatch):
    from gamelab.control import player as player_mod
    rng = np.random.default_rng(0)
    bgra = rng.integers(0, 256, (1080, 1920, 4), np.uint8)
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "minimized", lambda hwnd: False)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 1920, "height": 1080})

    class Screen:
        def grab(self, box):
            return bgra

    frame = Watchdog("MapleStory", threading.Event(), spotter=object())._grab(Screen())
    assert frame.flags.c_contiguous and np.array_equal(frame, bgra[:, :, 2::-1])


def test_watchdog_says_why_it_has_no_picture_the_game_minimized_or_gone(monkeypatch):
    # Minimized, the game's client area isn't 0 x 0 (432x54 on 2026-10-24): Windows is asked.
    from gamelab.control import player as player_mod
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "minimized", lambda hwnd: True)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": -32000, "top": -32000, "width": 432,
                                                                        "height": 54})
    dog = Watchdog("MapleStory", threading.Event(), spotter=object())
    assert dog._grab(None) == "minimized"

    def no_game(title):
        raise RuntimeError("no window titled 'MapleStory'")

    monkeypatch.setattr(player_mod.window, "find_game", no_game)
    assert dog._grab(None) == "no window"


def test_watchdog_sets_the_alarm_first_then_reports_once(tmp_path):
    frames = iter([SCENE, SCENE, scene_with_window(), scene_with_window(), SCENE])
    flag, reports, said = threading.Event(), [], []
    dog = Watchdog("MapleStory", flag, grab=lambda: next(frames), on_alarm=lambda frame: reports.append(flag.is_set()),
                   folder=tmp_path, log=said.append)
    for t in range(5):
        dog.tick(t * 0.25)
    assert flag.is_set() and reports == [True]  # the alarm was already up when it reported, and once a cycle
    assert dog.recording is not None and dog.recording.grabs == 3  # and the watchdog goes on, recording it
    dog.recording.finish("stopped")
    assert dog.recording.join(timeout=10)


def test_watchdog_keeps_watching_while_all_is_well(tmp_path):
    flag = threading.Event()
    dog = Watchdog("MapleStory", flag, grab=lambda: SCENE, interval=0.01, on_alarm=lambda f: None, folder=tmp_path,
                   log=lambda m: None)
    dog.start()
    time.sleep(0.1)
    assert dog.is_alive() and not flag.is_set()
    dog.stopped.set()
    dog.join(timeout=2)


def test_player_refuses_plans_once_the_alarm_is_up():
    flag = threading.Event()
    flag.set()
    # Returns before looking for the game window or pressing anything.
    assert Player({}, "no such window", flag).run({"events": [[0, "f", "down"]]}) == {"error": LIE_DETECTOR_ERROR}


def test_waits_end_early_when_the_alarm_goes_off():
    flag = threading.Event()
    threading.Timer(0.05, flag.set).start()
    started = time.perf_counter()
    assert _sleep_until(time.perf_counter() + 5, flag) is False
    assert time.perf_counter() - started < 0.5
    assert _sleep_until(time.perf_counter() + 0.01) is True


def test_raised_alarm_blocks_gamelab_until_cleared(alarm_file, monkeypatch):
    pages = []
    monkeypatch.setattr(alarm, "page", lambda title, text: pages.append(title))
    info = alarm.raise_alarm(scene_with_window())
    assert pages == ["MapleStory Lie Detector on screen"]
    assert json.loads(alarm_file.read_text())["reason"] == "lie_detector"
    assert Path(info["screenshot"]).exists()
    Path(info["screenshot"]).unlink()

    with pytest.raises(client.LieDetectorAlarm):
        client.submit([], name="x")
    assert main(["act", "100"]) == LIE_DETECTOR_EXIT  # refused before touching the game
    assert main(["map"]) == LIE_DETECTOR_EXIT         # its look would bring the game to the front
    assert main(["player", "--elevate"]) == LIE_DETECTOR_EXIT

    assert main(["clear-alarm"]) == 0
    assert not alarm_file.exists()
    client.check_alarm()  # no longer raises


def test_a_plan_can_be_cut_short_and_lets_go_of_its_keys(tmp_path, monkeypatch):
    from gamelab.control import player as player_mod
    from maplehelper.keys import parse
    sent = []
    monkeypatch.setattr(player_mod, "INBOX", tmp_path)
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 1920, "height": 1080})
    monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "Game")
    monkeypatch.setattr(player_mod.win32, "send_key", lambda key, down: sent.append((key.name, down)))
    p = Player({"f": parse("f")}, "Game")
    threading.Timer(0.1, lambda: (tmp_path / "plan1.cancel").touch()).start()
    started = time.perf_counter()
    result = p.run({"id": "plan1", "events": [[0, "f", "down"], [3000, "f", "up"]], "capture": {"lead_ms": 0}})
    assert time.perf_counter() - started < 1.0
    assert result["problems"] == ["cancelled"]
    assert sent == [("f", True), ("f", False)]           # pressed, then let go when cut short
    assert not (tmp_path / "plan1.cancel").exists()       # the signal is used up


def test_a_plan_can_last_past_its_last_key(tmp_path, monkeypatch):
    from gamelab.control import player as player_mod
    from maplehelper.keys import parse
    monkeypatch.setattr(player_mod, "INBOX", tmp_path)
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 1920, "height": 1080})
    monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "Game")
    monkeypatch.setattr(player_mod.win32, "send_key", lambda key, down: None)
    p = Player({"f": parse("f")}, "Game")
    started = time.perf_counter()
    p.run({"id": "p2", "events": [[0, "f", "down"], [10, "f", "up"]], "end_ms": 300, "capture": {"lead_ms": 0}})
    assert 0.28 <= time.perf_counter() - started < 0.6
    with pytest.raises(ValueError):
        player_mod.validate({"events": [[0, "f", "down"], [500, "f", "up"]], "end_ms": 100}, {"f": parse("f")})
