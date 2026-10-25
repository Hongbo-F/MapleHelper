"""The YOLO Lie Detector shadow: suggestions and recording only, never input."""

import json
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gamelab.control.lie_recorder import INDEX, SUMMARY
from gamelab.control.player import ShadowWorker, Watchdog
from gamelab.perception.lie_detector_yolo import FORM, YoloPatternShadow


class Values:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        boxes = SimpleNamespace(xyxy=Values(self.boxes), conf=Values([0.9] * len(self.boxes)))
        return [SimpleNamespace(boxes=boxes)]


def test_yolo_returns_original_pixel_pattern_centres_and_never_needs_an_input_api():
    model = FakeModel([[10, 20, 30, 50], [100, 200, 140, 260], [1, 2, 3, 4], [50, 60, 70, 90]])
    shadow = YoloPatternShadow(model=model, device=0)
    frame = np.zeros((300, 200, 3), np.uint8)
    frame[..., 0] = 255  # RGB red must reach Ultralytics' ndarray interface as BGR blue

    score, suggestions = shadow.check(frame)

    assert score == 1.0
    assert suggestions[0] == {"class": "pattern", "box": [10, 20, 30, 50], "center": [20.0, 35.0],
                              "confidence": 0.9}
    assert suggestions[1]["center"] == [120.0, 230.0]
    call = model.calls[0]
    assert call["source"][0, 0].tolist() == [0, 0, 255]
    assert call["imgsz"] == 640 and call["conf"] == 0.5 and call["device"] == 0


def test_full_frames_use_the_native_size_board_crop_and_restore_frame_coordinates():
    model = FakeModel([[10, 20, 30, 50]])
    shadow = YoloPatternShadow(model=model)
    _, suggestions = shadow.check(np.zeros((768, 1366, 3), np.uint8))
    assert model.calls[0]["source"].shape == (500, 750, 3)
    assert suggestions[0]["box"] == [318.0, 154.0, 338.0, 184.0]
    assert suggestions[0]["center"] == [328.0, 169.0]


def test_fewer_than_four_detections_do_not_mark_an_active_pattern_board():
    shadow = YoloPatternShadow(model=FakeModel([[0, 0, 10, 10]] * 3))
    score, suggestions = shadow.check(np.zeros((20, 20, 3), np.uint8))
    assert score == 0.75 and len(suggestions) == 3


def test_warming_up_loads_the_model_and_checks_a_blank_game_frame_once():
    model = FakeModel([])
    YoloPatternShadow(model=model).warm()
    assert len(model.calls) == 1 and model.calls[0]["source"].shape == (500, 750, 3)


# --- the watchdog's shadow, on a thread of its own (control.player.ShadowWorker) ---

@pytest.fixture(autouse=True)
def _shadows_stop():
    """The shadow threads a test's watchdogs started (it starts with them) stopped after it."""
    yield
    for thread in threading.enumerate():
        if isinstance(thread, ShadowWorker):
            thread.stopped.set()


class Spotter:
    """A made-up frame's red value is the countdown's score, its blue the Truth Booth's (1 = seen)."""

    def scores(self, frame):
        return {"countdown": float(frame[0, 0, 0]), "truth booth": float(frame[0, 0, 2])}


class Shadow:
    """A made-up frame's green value is the patterns it finds (4: a board)."""

    def __init__(self):
        self.calls = 0
        self.suggestion = {"class": "pattern", "box": [1, 2, 3, 4], "center": [2.0, 3.0],
                           "confidence": 0.9}

    def check(self, frame):
        self.calls += 1
        count = int(frame[0, 0, 1])
        return count / 4, [self.suggestion] * count


class Gated(Shadow):
    """Each check held until the test lets one through (`let_go`): the shadow as slow as the test says."""

    def __init__(self):
        super().__init__()
        self.gate = threading.Semaphore(0)
        self.started = []  # the green value of each frame it began to check

    def check(self, frame):
        self.started.append(int(frame[0, 0, 1]))
        assert self.gate.acquire(timeout=20)
        return super().check(frame)


def game(red=0, green=0, blue=0):
    frame = np.zeros((48, 64, 3), np.uint8)
    frame[0, 0] = red, green, blue
    return frame


def make_dog(tmp_path, shadow, screen, **kw):
    said = []
    kw = {"on_alarm": lambda frame: None, "end_s": 0.5, **kw}
    dog = Watchdog("MapleStory", threading.Event(), spotter=Spotter(), shadow=shadow, grab=lambda: screen["frame"],
                   folder=tmp_path, log=said.append, **kw)
    return dog, said


def until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end
        time.sleep(0.001)


def settled(dog):
    """Until the shadow has checked what it was handed (what's back waits for the watchdog's next look)."""
    until(dog.shadow_worker.idle)


def let_go(dog, shadow):
    """One held check through, and until it's back (the shadow on to the next frame handed it, if any)."""
    k = len(shadow.started)
    shadow.gate.release()
    until(lambda: dog.shadow_worker.idle() or len(shadow.started) > k)


def written(rec):
    assert rec.join(timeout=10)
    index = [json.loads(line) for line in (rec.folder / INDEX).read_text(encoding="utf-8").splitlines()]
    return index, json.loads((rec.folder / SUMMARY).read_text(encoding="utf-8"))


def test_the_shadow_keeps_a_cycle_going_and_each_check_is_recorded_with_its_frame(tmp_path):
    shadow, screen = Shadow(), {}
    frames = [game(), game(red=1, green=4), game(green=4), game(green=4), game(), game(), game()]
    dog, _ = make_dog(tmp_path, shadow, screen)
    for n, f in enumerate(frames):
        screen["frame"] = f
        dog.tick(n * 0.25)
        settled(dog)  # its check back before the next look: a look late
    assert shadow.calls == 6  # not on the ordinary game frame before the template started the cycle
    assert dog.recording is None and dog.cycle_t0 is None
    [rec] = dog.ended
    index, summary = written(rec)
    shots = [r for r in index if "file" in r]
    checks = [r for r in index if "shadow" in r]
    assert len(shots) == 6 and all(set(r["scores"]) == {"countdown", "truth booth"} for r in shots)
    assert [r["seen"] for r in shots] == [True] + [False] * 5  # the spotter's: the notice at its first frame only
    # Each check a line of its own, after its frame's, with its frame's t and number, a look late; the last frame's
    # came back after the cycle's end, and is left.
    assert [(r["i"], r["t"], r["seen"], len(r["suggestions"]), r["late"]) for r in checks] == [
        (0, 0.0, True, 4, 0.25), (1, 0.25, True, 4, 0.25), (2, 0.5, True, 4, 0.25), (3, 0.75, False, 0, 0.25),
        (4, 1.0, False, 0, 0.25)]
    assert [index.index(r) for r in checks] == [1, 3, 5, 7, 9]
    assert summary["shadow"] == {"checks": 5, "suggestions": 12, "late_max_s": 0.25, "lost": 0}
    assert summary["forms"][FORM] == {"first": 0.0, "last": 0.5, "frames": 3}
    # The board kept it going past the notice: over at 1.25 s (its last board at 0.5, then 0.5 s without), where the
    # spotter alone ended it at 0.75.
    assert summary["first_seen"] == 0.0 and summary["last_seen"] == 0.5 and summary["duration_s"] == 1.25
    assert summary["end"] == "gone"


def test_the_shadow_alone_cannot_start_a_cycle(tmp_path):
    shadow = Shadow()
    dog, _ = make_dog(tmp_path, shadow, {"frame": game(green=4)})
    dog.tick(0.0)
    settled(dog)
    dog.tick(0.25)
    assert shadow.calls == 0 and not dog.alarm.is_set() and dog.recording is None


def test_a_shadow_failure_does_not_disable_template_alarms_or_frame_recording(tmp_path):
    class Broken:
        def check(self, frame):
            raise RuntimeError("GPU unavailable")

    dog, said = make_dog(tmp_path, Broken(), {"frame": game(red=1)})
    dog.tick(0.0)
    assert dog.alarm.is_set() and dog.recording is not None
    settled(dog)
    dog.tick(0.25)
    assert sum("the YOLO shadow check failed (RuntimeError: GPU unavailable)" in m for m in said) == 1
    rec = dog.recording
    rec.finish("stopped")
    index, summary = written(rec)
    assert [r.get("file") for r in index] == ["00000.jpg", "00001.jpg"] and summary["shadow"]["checks"] == 0


def test_a_slow_shadow_never_holds_up_the_recording_4_frames_a_second_through_its_check(tmp_path):
    """2026-10-24, 21:02: the first sighting's check, on the watchdog's thread, loaded torch and the model there; the
    recording has frame 0, then nothing until 16.1 s, by when the Truth Booth showed. A check that takes 10 s now: the
    frames go on 4 a second through it, and the watchdog stops at once when asked, the check still out."""
    class Slow(Shadow):
        def check(self, frame):
            self.calls += 1
            time.sleep(10)
            return 1.0, []

    shadow = Slow()
    dog, _ = make_dog(tmp_path, shadow, {"frame": game(red=1)}, end_s=60.0)
    dog.start()
    time.sleep(3.0)
    dog.stopped.set()
    dog.join(timeout=5)
    assert not dog.is_alive() and shadow.calls == 1
    [rec] = dog.ended
    index, summary = written(rec)
    times = [r["t"] for r in index if "file" in r]
    assert 11 <= len(times) <= 14 and summary["end"] == "stopped" and summary["shadow"]["checks"] == 0
    assert np.median(np.diff(times)) == pytest.approx(0.25, abs=0.02) and max(np.diff(times)) < 0.4


def test_the_shadow_s_checks_come_back_late_and_the_cycle_still_ends_and_reads_right(tmp_path):
    """A check out for 3 looks at a time (the shadow takes the newest frame when it's free): the cycle's end waits for
    its check of a frame of the stretch without the notice, and the board it saw there keeps the cycle going; the Truth
    Booth (the spotter's) is the outcome as ever; each check's line has its frame's t and number."""
    shadow, screen = Gated(), {"frame": game(red=1, green=4)}
    dog, _ = make_dog(tmp_path, shadow, screen, end_s=1.0)
    dog.tick(0.0)  # the notice: the cycle starts; the shadow checks its frame, and holds it
    until(lambda: len(shadow.started) == 1)
    screen["frame"] = game(green=4)  # the notice gone, the pattern board up (nothing the spotter knows)
    dog.tick(0.25)
    dog.tick(0.5)
    let_go(dog, shadow)  # 0 s's check back; on to 0.5 s's (0.25 s's never checked: the newest goes)
    dog.tick(0.75)
    dog.tick(1.0)
    dog.tick(1.25)  # 1 s without the notice (since 0.25): over by the spotter, but 0.5 s's check is out
    assert dog.cycle_t0 is not None
    let_go(dog, shadow)  # 0.5 s's: a board; on to 1.25 s's
    dog.tick(1.5)
    assert dog.cycle_t0 is not None and dog.gone_since == 1.5  # without it from 1.5 s, as late as the check
    screen["frame"] = game(blue=1)  # sent to the Truth Booth
    let_go(dog, shadow)  # 1.25 s's: a board; on to 1.5 s's
    dog.tick(1.75)
    screen["frame"] = game()
    t = 2.0
    while dog.cycle_t0 is not None:
        while not dog.shadow_worker.idle():
            let_go(dog, shadow)
        dog.tick(t)
        assert t <= 3.0
        t += 0.25
    assert t == 3.25  # the Truth Booth last seen at 1.75: 1 s without it from 2.0
    while not dog.shadow_worker.idle():
        let_go(dog, shadow)
    [rec] = dog.ended
    index, summary = written(rec)
    number = {r["t"]: r["i"] for r in index if "file" in r}
    checks = [r for r in index if "shadow" in r]
    assert all(number[r["t"]] == r["i"] for r in checks)
    assert [(r["t"], r["seen"], r["late"]) for r in checks[:4]] == [(0.0, True, 0.75), (0.5, True, 1.0),
                                                                     (1.25, True, 0.5), (1.5, True, 0.5)]
    assert not any(r["seen"] for r in checks[4:]) and summary["shadow"]["late_max_s"] == 1.0
    assert summary["forms"][FORM] == {"first": 0.0, "last": 1.5, "frames": 4}
    assert summary["forms"]["truth booth"] == {"first": 1.75, "last": 1.75, "frames": 1}
    assert summary["truth_booth"] is True and summary["outcome"] == "sent to the Truth Booth"
    assert summary["first_seen"] == 0.0 and summary["last_seen"] == 1.75 and summary["end"] == "gone"


def test_a_check_that_hangs_holds_the_cycle_s_end_no_longer_than_shadow_wait_s(tmp_path, monkeypatch):
    from gamelab.control import player
    monkeypatch.setattr(player, "SHADOW_WAIT_S", 0.5)
    shadow, screen = Gated(), {"frame": game(red=1)}
    dog, _ = make_dog(tmp_path, shadow, screen, end_s=1.0)
    dog.tick(0.0)
    let_go(dog, shadow)
    screen["frame"] = game()
    dog.tick(0.25)  # without it from here; this frame's check hangs
    until(lambda: len(shadow.started) == 2)
    for t in (0.5, 0.75, 1.0, 1.25, 1.5):
        dog.tick(t)
        assert dog.cycle_t0 is not None, t  # due at 1.25, waiting for the check
    dog.tick(1.75)  # 0.5 s past due
    assert dog.cycle_t0 is None and dog.ended[0].end == "gone"
    shadow.gate.release(2)  # (and the frame handed it last)


def test_a_board_the_shadow_sees_after_the_alarm_is_cleared_raises_it_again(tmp_path, monkeypatch):
    """As the spotter's sight does (the cycle not over): gamelab's commands go on refusing. Not for a frame grabbed
    before the clear: the board may be gone since."""
    from gamelab.control import alarm, player
    path, pages = tmp_path / "ALARM.json", []
    for mod in (alarm, player):
        monkeypatch.setattr(mod, "ALARM", path)
    monkeypatch.setattr(alarm, "page", lambda title, text: pages.append(title))
    shadow, screen = Shadow(), {"frame": game(red=1, green=4)}
    dog, said = make_dog(tmp_path, shadow, screen, on_alarm=alarm.raise_alarm, end_s=5.0)
    dog.tick(0.0)
    assert path.exists() and len(pages) == 1
    screen["frame"] = game(green=4)
    for t in (0.25, 0.5):
        settled(dog)
        dog.tick(t)
    alarm.clear_alarm()  # (after 0.5 s's look)
    settled(dog)
    dog.tick(0.75)  # 0.5 s's board: grabbed before the clear
    assert not path.exists() and len(pages) == 1
    settled(dog)
    dog.tick(1.0)  # 0.75 s's: after it
    assert path.exists() and len(pages) == 2
    assert any("cleared while the Lie Detector is still on screen: raised again" in m for m in said)
    dog.recording.finish("stopped")


def test_the_shadow_warms_up_from_the_start_and_the_frames_go_on_meanwhile(tmp_path):
    loaded = threading.Event()

    class Warming(Shadow):
        def warm(self):
            assert loaded.wait(10)

    shadow, screen = Warming(), {"frame": game(red=1, green=4)}
    dog, said = make_dog(tmp_path, shadow, screen)
    for t in (0.0, 0.25, 0.5):  # warming up: the cycle recorded all the same, its frames handed over in turn
        dog.tick(t)
    assert shadow.calls == 0 and dog.recording.numbered == 3 and not dog.shadow_worker.ready.is_set()
    loaded.set()
    until(dog.shadow_worker.ready.is_set)
    settled(dog)
    assert shadow.calls == 1  # the newest only
    assert any(m.startswith("the YOLO shadow is ready: loaded and a first check in") for m in said)
    dog.tick(0.75)
    rec = dog.recording
    rec.finish("stopped")
    index, _ = written(rec)
    assert [(r["t"], r["i"]) for r in index if "shadow" in r] == [(0.5, 2)]


def test_a_shadow_that_cant_be_loaded_is_turned_off_and_the_watching_goes_on(tmp_path):
    class Missing(Shadow):
        def warm(self):
            raise ModuleNotFoundError("No module named 'ultralytics'")

    shadow, screen = Missing(), {"frame": game(red=1, green=4)}
    dog, said = make_dog(tmp_path, shadow, screen)
    until(dog.shadow_worker.ready.is_set)
    assert dog.shadow_worker.off
    assert any("the YOLO shadow couldn't be loaded (ModuleNotFoundError" in m for m in said)
    dog.tick(0.0)
    dog.tick(0.25)
    assert dog.alarm.is_set() and dog.recording.numbered == 2 and shadow.calls == 0 and dog.shadow_worker.idle()
    dog.recording.finish("stopped")


def test_a_shadow_check_of_a_recording_ended_or_too_far_behind_is_not_kept(tmp_path):
    from gamelab.control.lie_recorder import Recording

    class Full:
        """The writer's queue full (the disk behind)."""

        def put_nowait(self, item):
            raise queue.Full

        def get(self, timeout):
            time.sleep(timeout)
            raise queue.Empty

    rec = Recording(tmp_path, log=lambda m: None)
    rec._queue, real = Full(), rec._queue
    assert not rec.add_shadow(0.0, 0, FORM, 1.0, [], late=0.5) and rec.shadow_lost == 1
    rec._queue = real
    assert rec.add_shadow(0.0, 0, FORM, 1.0, [], late=0.5)
    rec.finish("gone")
    assert not rec.add_shadow(0.25, 1, FORM, 1.0, [], late=0.5)
    assert rec.join(timeout=10)
    summary = json.loads((rec.folder / SUMMARY).read_text(encoding="utf-8"))
    assert summary["shadow"] == {"checks": 1, "suggestions": 0, "late_max_s": 0.5, "lost": 1}
