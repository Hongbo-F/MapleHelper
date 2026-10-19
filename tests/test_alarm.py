"""Lie Detector auto-stop: spotting it, stopping the player, and refusing to run until cleared."""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab import alarm, client
from gamelab.__main__ import LIE_DETECTOR_EXIT, main
from gamelab.player import LIE_DETECTOR_ERROR, Player, Watchdog, _sleep_until

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
    spotter = alarm.LieDetectorSpotter()
    assert spotter.seen(scene_with_window())
    assert spotter.score(WINDOW) > 0.95


@pytest.mark.parametrize("frame", [SCENE, NEAR_MISS, np.zeros((540, 960, 3), np.uint8)])
def test_ignores_yellow_scenery_and_other_windows(frame):
    assert not alarm.LieDetectorSpotter().seen(frame)


def test_watchdog_sets_the_alarm_first_then_reports_once():
    frames = iter([SCENE, SCENE, scene_with_window(), scene_with_window()])
    flag, reports = threading.Event(), []
    dog = Watchdog("MapleStory", flag, grab=lambda: next(frames), interval=0.01,
                   on_alarm=lambda frame: reports.append(flag.is_set()))
    dog.start()
    dog.join(timeout=5)
    assert not dog.is_alive()
    assert flag.is_set() and reports == [True]  # the alarm was already up when it reported


def test_watchdog_keeps_watching_while_all_is_well():
    flag = threading.Event()
    dog = Watchdog("MapleStory", flag, grab=lambda: SCENE, interval=0.01, on_alarm=lambda f: None)
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
    assert main(["player", "--elevate"]) == LIE_DETECTOR_EXIT

    assert main(["clear-alarm"]) == 0
    assert not alarm_file.exists()
    client.check_alarm()  # no longer raises
