"""The Lie Detector's whole cycle recorded (control.lie_recorder): every frame the watchdog grabs from the first sight
until it has been gone a while, with an index and a summary, under a disk cap. All in temp folders, with made-up
frames and scores; nothing touches the game."""

import json
import os
import threading
import time

import cv2
import numpy as np
import pytest

from gamelab.control import lie_recorder
from gamelab.control.lie_recorder import INDEX, SUMMARY, Recording

GONE = {"countdown": 0.4, "mini-game": 0.5, "truth booth": 0.6}
COUNTDOWN = {**GONE, "countdown": 1.3}
MINI_GAME = {**GONE, "mini-game": 1.01}
BOOTH = {**GONE, "truth booth": 1.17}


def frame(value=90, w=64, h=48):
    out = np.full((h, w, 3), value, np.uint8)
    out[:, : w // 2, 0] = 250  # some colour: what's written is RGB turned to the JPEG's BGR and back
    return out


def lines(folder):
    return [json.loads(s) for s in (folder / INDEX).read_text(encoding="utf-8").splitlines()]


def finished(rec, end="gone"):
    rec.finish(end)
    assert rec.join(timeout=10)
    return json.loads((rec.folder / SUMMARY).read_text(encoding="utf-8"))


def test_a_cycle_seen_then_gone_saves_every_frame_an_index_and_a_summary(tmp_path):
    said = []
    rec = Recording(tmp_path, {"screenshot": "lab/lie_detector_101500.png"}, log=said.append, end_s=1.0)
    plan = [COUNTDOWN] * 4 + [GONE] + [MINI_GAME] * 3 + [GONE] * 5  # a moment between the two windows
    for n, scores in enumerate(plan):
        rec.add(n * 0.25, frame(n * 10), scores)
    summary = finished(rec)
    assert rec.folder.parent == tmp_path and len(rec.folder.name) == len("20261024-101500")
    assert sorted(p.name for p in rec.folder.glob("*.jpg")) == [f"{i:05d}.jpg" for i in range(13)]
    index = lines(rec.folder)
    assert [r["t"] for r in index] == [n * 0.25 for n in range(13)]
    assert [r["seen"] for r in index] == [s is not GONE for s in plan]
    assert index[0] == {"t": 0.0, "i": 0, "seen": True, "scores": COUNTDOWN, "file": "00000.jpg"}
    # The pictures are the frames as grabbed (RGB), JPEG'd.
    back = cv2.cvtColor(cv2.imread(str(rec.folder / "00005.jpg")), cv2.COLOR_BGR2RGB)
    assert back.shape == (48, 64, 3) and np.abs(back.astype(int) - frame(50)).mean() < 2
    assert summary["first_seen"] == 0.0 and summary["last_seen"] == 1.75 and summary["seen_s"] == 1.75
    assert summary["duration_s"] == 3.0 and summary["frames"] == 13 and summary["grabs"] == 13
    assert summary["forms"] == {"countdown": {"first": 0.0, "last": 0.75, "frames": 4},
                                "mini-game": {"first": 1.25, "last": 1.75, "frames": 3}}
    assert summary["truth_booth"] is False and summary["outcome"] == "answered" and summary["end"] == "gone"
    assert summary["alarm_screenshot"] == "lab/lie_detector_101500.png" and summary["window"] == [64, 48]
    assert summary["write_errors"] == 0 and summary["dropped"] == 0 and summary["not_showing"] == {}
    assert summary["jpeg_quality"] == lie_recorder.JPEG_QUALITY and summary["end_stretch_s"] == 1.0
    assert said[-1].startswith("Lie Detector recording ended (gone): 13 frames over 3 s") and "answered" in said[-1]


def test_the_truth_booth_is_the_outcome_when_it_shows(tmp_path):
    rec = Recording(tmp_path, log=lambda m: None)
    for n, scores in enumerate([MINI_GAME, MINI_GAME, GONE, GONE, BOOTH, BOOTH, GONE]):
        rec.add(n * 0.25, frame(), scores)
    summary = finished(rec)
    assert summary["truth_booth"] is True and summary["outcome"] == "sent to the Truth Booth"
    assert summary["forms"]["truth booth"] == {"first": 1.0, "last": 1.25, "frames": 2}


def test_one_still_up_when_it_ends_has_no_outcome_yet(tmp_path):
    rec = Recording(tmp_path, log=lambda m: None)
    rec.add(0.0, frame(), COUNTDOWN)
    assert finished(rec, "cap")["outcome"].startswith("unknown")


def test_a_game_window_gone_or_minimized_is_recorded_as_such_with_no_frame(tmp_path):
    rec = Recording(tmp_path, log=lambda m: None)
    rec.add(0.0, frame(), COUNTDOWN)
    rec.add(0.25, None, why="minimized")
    rec.add(0.5, None, why="no window")
    rec.add(0.75, frame(), COUNTDOWN)
    summary = finished(rec)
    assert lines(rec.folder)[1:3] == [{"t": 0.25, "game": "minimized"}, {"t": 0.5, "game": "no window"}]
    assert summary["frames"] == 2 and summary["grabs"] == 4
    assert summary["not_showing"] == {"minimized": 1, "no window": 1}
    assert len(list(rec.folder.glob("*.jpg"))) == 2


def test_a_frame_that_cant_be_written_is_logged_once_and_the_recording_goes_on(tmp_path, monkeypatch):
    said = []
    real = cv2.imencode
    fail = {2, 3}

    def imencode(ext, img, params):
        if int(img[0, -1, 0]) in fail:  # (the frame's own value: frames 2 and 3)
            raise OSError(28, "No space left on device")
        return real(ext, img, params)

    monkeypatch.setattr(lie_recorder.cv2, "imencode", imencode)
    rec = Recording(tmp_path, log=said.append)
    for n in range(6):
        rec.add(n * 0.25, frame(n), COUNTDOWN)
    summary = finished(rec)
    assert summary["write_errors"] == 2 and summary["frames"] == 4
    assert sorted(p.name for p in rec.folder.glob("*.jpg")) == ["00000.jpg", "00001.jpg", "00004.jpg", "00005.jpg"]
    index = lines(rec.folder)
    assert len(index) == 6 and "No space left" in index[2]["error"] and "file" not in index[2]
    assert sum("couldn't write frame" in m for m in said) == 1  # once, not 4 times a second
    assert "2 frames not written" in said[-1]


def test_adding_never_waits_for_the_disk_a_full_queue_drops_frames_and_says_how_many(tmp_path, monkeypatch):
    taken, stuck, said = threading.Event(), threading.Event(), []
    real = Recording._write_one

    def slow(self, record, frame):
        taken.set()
        stuck.wait(10)
        real(self, record, frame)

    monkeypatch.setattr(Recording, "_write_one", slow)
    rec = Recording(tmp_path, log=said.append)
    rec.add(0.0, frame(), COUNTDOWN)
    assert taken.wait(5)  # the writer has the first and is stuck on it (a disk that hangs)
    started = time.perf_counter()
    for n in range(1, lie_recorder.QUEUE_ITEMS + 6):
        rec.add(n * 0.25, frame(), COUNTDOWN)
    assert time.perf_counter() - started < 1.0
    stuck.set()
    for _ in range(250):  # the disk catches up
        if rec._queue.qsize() < lie_recorder.QUEUE_ITEMS:
            break
        time.sleep(0.02)
    rec.add(99.0, frame(), COUNTDOWN)
    summary = finished(rec)
    # The writer had taken one when it stuck; the queue held QUEUE_ITEMS; the rest went.
    assert summary["dropped"] == 5 and summary["frames"] == lie_recorder.QUEUE_ITEMS + 2
    assert lines(rec.folder)[-1]["dropped_before"] == 5
    assert sum("disk is behind" in m for m in said) == 1


def test_a_recording_that_cant_make_its_folder_raises_for_the_watchdog_to_log(tmp_path):
    blocker = tmp_path / "lie_detector"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    with pytest.raises(OSError):
        Recording(blocker / "x", log=lambda m: None)


def _fake_recording(root, name, nbytes, summary=True):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "00000.jpg").write_bytes(b"x" * nbytes)
    (folder / INDEX).write_text("{}\n", encoding="utf-8")
    if summary:
        (folder / SUMMARY).write_text("{}", encoding="utf-8")
    return folder


def test_the_oldest_recordings_go_past_the_cap_never_the_one_being_written(tmp_path):
    a = _fake_recording(tmp_path, "20261020-010000", 400)
    b = _fake_recording(tmp_path, "20261021-010000", 400)
    c = _fake_recording(tmp_path, "20261022-010000", 400)
    n = lie_recorder.size_of(a)  # 400 and the index's and summary's few bytes
    assert 400 < n < 420
    assert lie_recorder.prune(tmp_path, keep_bytes=2 * n + 10, dry_run=True) == [(a, n)]
    assert a.exists()
    assert lie_recorder.prune(tmp_path, keep_bytes=10_000) == []
    # The one being written is left out of the sum and never deleted, whatever its size.
    d = _fake_recording(tmp_path, "20261023-010000", 5000, summary=False)
    assert lie_recorder.prune(tmp_path, keep_bytes=n + 10, spare=d) == [(a, n), (b, n)]
    assert not a.exists() and not b.exists() and c.exists() and d.exists()
    assert lie_recorder.prune(tmp_path, keep_bytes=0, spare=d) == [(c, n)] and d.exists()
    assert lie_recorder.prune(tmp_path / "none") == []


def test_a_new_recording_deletes_the_oldest_past_the_cap(tmp_path):
    old = _fake_recording(tmp_path, "20261020-010000", 3000)
    keep = _fake_recording(tmp_path, "20261021-010000", 300)
    said = []
    rec = Recording(tmp_path, log=said.append, keep_bytes=1000)
    for _ in range(100):  # the pruning runs on a thread of its own
        if not old.exists():
            break
        time.sleep(0.02)
    finished(rec)
    assert not old.exists() and keep.exists() and rec.folder.exists()
    assert any("deleted the 1 oldest" in m for m in said)


def test_whether_a_recording_is_still_being_written(tmp_path):
    assert lie_recorder.going(tmp_path) is None
    rec = Recording(tmp_path, log=lambda m: None)
    rec.add(0.0, frame(), COUNTDOWN)
    for _ in range(250):
        if (rec.folder / INDEX).exists():
            break
        time.sleep(0.02)
    assert lie_recorder.going(tmp_path) == rec.folder
    assert lie_recorder.going(tmp_path, now=time.time() + lie_recorder.LIVE_S + 1) is None  # the player stopped
    finished(rec)
    assert lie_recorder.going(tmp_path) is None
    assert lie_recorder.summary_of(rec.folder)["frames"] == 1 and lie_recorder.summary_of(tmp_path) is None
    # Two in one second (they can't be: END_S apart at least) still get folders of their own.
    other = lie_recorder._new_folder(tmp_path)
    again = lie_recorder._new_folder(tmp_path)
    assert other != again and other.exists() and again.exists()
    os.utime(again, (time.time() - 60, time.time() - 60))
    assert lie_recorder.going(tmp_path) is None  # made a minute ago, never written to
