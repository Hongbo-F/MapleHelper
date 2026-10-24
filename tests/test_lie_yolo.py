"""The YOLO Lie Detector shadow: suggestions and recording only, never input."""

import json
import threading
from types import SimpleNamespace

import numpy as np

from gamelab.control.lie_recorder import INDEX, SUMMARY
from gamelab.control.player import Watchdog
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


class Spotter:
    def scores(self, frame):
        return {"countdown": float(frame[0, 0, 0])}


class Shadow:
    def __init__(self):
        self.calls = 0
        self.suggestion = {"class": "pattern", "box": [1, 2, 3, 4], "center": [2.0, 3.0],
                           "confidence": 0.9}

    def check(self, frame):
        self.calls += 1
        count = int(frame[0, 0, 1])
        return count / 4, [self.suggestion] * count


def game(red=0, green=0):
    frame = np.zeros((48, 64, 3), np.uint8)
    frame[0, 0, 0], frame[0, 0, 1] = red, green
    return frame


def test_shadow_extends_an_existing_episode_and_records_every_suggestion(tmp_path, monkeypatch):
    from gamelab.control import player
    monkeypatch.setattr(player, "ALARM", tmp_path / "ALARM.json")
    shadow = Shadow()
    frames = iter([game(), game(red=1, green=4), game(green=4), game(green=4), game(), game(), game()])
    dog = Watchdog("MapleStory", threading.Event(), spotter=Spotter(), shadow=shadow,
                   grab=lambda: next(frames), on_alarm=lambda frame: None, folder=tmp_path,
                   log=lambda message: None, end_s=0.5)

    for n in range(7):
        dog.tick(n * 0.25)

    assert shadow.calls == 6  # not called on the ordinary game frame before the template started the cycle
    assert dog.recording is None
    [rec] = dog.ended
    assert rec.join(timeout=10)
    index = [json.loads(line) for line in (rec.folder / INDEX).read_text(encoding="utf-8").splitlines()]
    assert len(index) == 6
    assert index[0]["seen"] is True and len(index[0]["suggestions"]) == 4
    assert index[1]["scores"]["countdown"] == 0.0 and index[1]["scores"][FORM] == 1.0
    assert index[2]["seen"] is True  # YOLO keeps the episode alive after the notice disappears
    assert index[3]["seen"] is False and index[3]["suggestions"] == []
    summary = json.loads((rec.folder / SUMMARY).read_text(encoding="utf-8"))
    assert summary["shadow"] == {"checks": 6, "suggestions": 12}
    assert summary["forms"][FORM] == {"first": 0.0, "last": 0.5, "frames": 3}


def test_shadow_alone_cannot_start_an_alarm(tmp_path):
    shadow = Shadow()
    dog = Watchdog("MapleStory", threading.Event(), spotter=Spotter(), shadow=shadow,
                   grab=lambda: game(green=4), on_alarm=lambda frame: None, folder=tmp_path,
                   log=lambda message: None)
    dog.tick(0.0)
    assert shadow.calls == 0 and not dog.alarm.is_set() and dog.recording is None


def test_a_shadow_failure_does_not_disable_template_alarms_or_frame_recording(tmp_path, monkeypatch):
    from gamelab.control import player
    monkeypatch.setattr(player, "ALARM", tmp_path / "ALARM.json")

    class Broken:
        def check(self, frame):
            raise RuntimeError("GPU unavailable")

    said = []
    dog = Watchdog("MapleStory", threading.Event(), spotter=Spotter(), shadow=Broken(),
                   grab=lambda: game(red=1), on_alarm=lambda frame: None, folder=tmp_path,
                   log=said.append)
    dog.tick(0.0)
    assert dog.alarm.is_set() and dog.recording is not None
    assert any("YOLO shadow check failed" in message for message in said)
    dog.recording.finish("stopped")
    assert dog.recording.join(timeout=10)
    index = (dog.recording.folder / INDEX).read_text(encoding="utf-8")
    assert '"file": "00000.jpg"' in index and "suggestions" not in index
