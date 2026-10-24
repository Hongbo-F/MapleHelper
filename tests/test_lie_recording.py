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


# --- the watchdog records each cycle (control.player.Watchdog) ---

from gamelab.control.player import Watchdog  # noqa: E402

FORMS = [GONE, COUNTDOWN, MINI_GAME, BOOTH]


class Spotter:
    """Scores a made-up frame by its green value: 0 nothing, 1 the countdown, 2 the mini-game, 3 the Truth Booth."""

    def scores(self, frame):
        return FORMS[int(frame[0, 0, 1])]


def game(kind):
    """A made-up frame the fake spotter scores as `kind` (an index into FORMS), or why there's none (a str)."""
    if isinstance(kind, str):
        return kind
    out = frame(40)
    out[..., 1] = kind
    return out


class Screen:
    """What the fake grab hands the watchdog, set as the test goes."""

    def __init__(self, kind=0):
        self.kind = kind

    def __call__(self):
        return game(self.kind)


def make_dog(tmp_path, screen, flag=None, on_alarm=None, end_s=1.0, cap_s=60.0, **kw):
    said, alarms = [], []
    flag = flag or threading.Event()

    def raised(frame):
        alarms.append(flag.is_set())
        return {"screenshot": "shot.png"}

    dog = Watchdog("MapleStory", flag, spotter=Spotter(), grab=screen, on_alarm=on_alarm or raised,
                   folder=tmp_path / "lie_detector", log=said.append, end_s=end_s, cap_s=cap_s, **kw)
    return dog, flag, said, alarms


def play(dog, screen, kinds, t0=0.0, step=0.25):
    """One look a `step` for each kind on screen in turn; returns the time of the next look."""
    t = t0
    for kind in kinds:
        screen.kind = kind
        dog.tick(t)
        t += step
        end = time.monotonic() + 10
        while dog.recording is not None and not dog.recording._queue.empty():  # looks 250 ms apart, not 0
            assert time.monotonic() < end
            time.sleep(0.001)
    return t


def written(dog):
    """The recordings the watchdog ended, once their writers are done: (folder, summary, index)."""
    out = []
    for rec in dog.ended:
        assert rec.join(timeout=10)
        out.append((rec.folder, json.loads((rec.folder / SUMMARY).read_text(encoding="utf-8")), lines(rec.folder)))
    return out


def test_a_cycle_seen_for_a_while_then_gone_is_recorded_from_its_first_sight_to_its_end(tmp_path):
    screen = Screen()
    dog, flag, said, alarms = make_dog(tmp_path, screen)
    t = play(dog, screen, [0, 0, 0])
    assert not flag.is_set() and dog.recording is None and not (tmp_path / "lie_detector").exists()
    t = play(dog, screen, [1] * 6 + [2] * 4, t)  # the countdown, then the mini-game's notice
    assert flag.is_set() and alarms == [True] and dog.recording is not None
    t = play(dog, screen, [0] * 4, t)  # gone: 0.75 s after its first look without it, not yet the 1 s
    assert dog.recording is not None
    play(dog, screen, [0, 0], t)        # 1.0 s: over
    assert dog.recording is None and dog.cycle_t0 is None
    [(folder, summary, index)] = written(dog)
    assert len(index) == 15 and summary["frames"] == 15  # 10 seen, then 5 gone (the last at the end stretch)
    assert index[0]["t"] == 0.0 and index[-1]["t"] == 3.5 and [r["seen"] for r in index] == [True] * 10 + [False] * 5
    assert summary["first_seen"] == 0.0 and summary["last_seen"] == 2.25 and summary["duration_s"] == 3.5
    assert summary["forms"] == {"countdown": {"first": 0.0, "last": 1.25, "frames": 6},
                                "mini-game": {"first": 1.5, "last": 2.25, "frames": 4}}
    assert summary["outcome"] == "answered" and summary["end"] == "gone" and summary["truth_booth"] is False
    assert summary["alarm_screenshot"] == "shot.png" and summary["window"] == [64, 48]
    assert flag.is_set()  # still locked: the alarm is lifted elsewhere (the player's rule)
    assert any(m.startswith("recording the Lie Detector") and str(folder) in m for m in said)
    assert any(m.startswith("Lie Detector recording ended (gone): 15 frames") for m in said)
    first = [i for i, m in enumerate(said) if m.startswith("LIE DETECTOR seen: input stopped")]
    assert first and first[0] < [i for i, m in enumerate(said) if m.startswith("recording the")][0]


def test_a_flicker_shorter_than_the_end_stretch_is_the_same_cycle(tmp_path):
    screen = Screen()
    dog, flag, said, alarms = make_dog(tmp_path, screen)
    t = play(dog, screen, [1, 1, 0, 0, 1, 0, 0, 0, 2, 2])  # gone 0.5 s and 0.75 s: under the 1 s
    play(dog, screen, [0] * 5, t)
    [(folder, summary, index)] = written(dog)
    assert alarms == [True] and len(index) == 15
    assert summary["first_seen"] == 0.0 and summary["last_seen"] == 2.25


def test_the_truth_booth_is_the_outcome_when_a_failed_test_sends_her_there(tmp_path):
    screen = Screen()
    dog, *_ = make_dog(tmp_path, screen)
    t = play(dog, screen, [2, 2, 0, 0, 3, 3, 3])
    play(dog, screen, [0] * 5, t)
    [(_, summary, _)] = written(dog)
    assert summary["truth_booth"] is True and summary["outcome"] == "sent to the Truth Booth"
    assert summary["forms"]["truth booth"] == {"first": 1.0, "last": 1.5, "frames": 3}


def test_the_recording_stops_at_the_cap_and_the_cycle_goes_on_unrecorded(tmp_path):
    screen = Screen()
    dog, flag, said, alarms = make_dog(tmp_path, screen, cap_s=2.0)
    t = play(dog, screen, [1] * 12)  # 3 s of it
    assert dog.recording is None and dog.cycle_t0 is not None and flag.is_set()
    t = play(dog, screen, [1, 1, 0, 0, 0, 0, 0], t)  # still the same cycle: no new recording, no second alarm
    assert alarms == [True] and dog.cycle_t0 is None
    [(folder, summary, index)] = written(dog)
    assert summary["end"] == "cap" and summary["frames"] == 9 and index[-1]["t"] == 2.0  # 0 to 2 s
    assert summary["outcome"].startswith("unknown")
    assert len(list((tmp_path / "lie_detector").iterdir())) == 1


def test_a_game_window_gone_or_minimized_mid_cycle_is_recorded_as_such_not_as_the_lie_detector_gone(tmp_path):
    screen = Screen()
    dog, flag, said, alarms = make_dog(tmp_path, screen)
    t = play(dog, screen, [1, 1] + ["minimized"] * 8 + ["no window"] * 4)  # 3 s without the game: not gone
    assert dog.recording is not None
    t = play(dog, screen, [0, 0, "minimized", 0, 0, 0, 0], t)  # the stretch starts over after the minimized look
    assert dog.recording is not None  # 0.75 s since then
    play(dog, screen, [0], t)
    [(folder, summary, index)] = written(dog)
    assert summary["not_showing"] == {"minimized": 9, "no window": 4} and summary["frames"] == 9
    assert index[2] == {"t": 0.5, "game": "minimized"} and index[10] == {"t": 2.5, "game": "no window"}
    assert summary["end"] == "gone" and summary["outcome"] == "answered"


def test_a_frame_that_cant_be_written_doesnt_stop_the_recording_or_the_watchdog(tmp_path, monkeypatch):
    real = cv2.imencode

    def imencode(ext, img, params):  # the mini-game's frames can't be written
        if img[1, 1, 1] == 2:
            raise PermissionError(13, "Permission denied")
        return real(ext, img, params)

    monkeypatch.setattr(lie_recorder.cv2, "imencode", imencode)
    screen = Screen()
    dog, flag, said, alarms = make_dog(tmp_path, screen)
    t = play(dog, screen, [1, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0])
    [(folder, summary, index)] = written(dog)
    assert summary["write_errors"] == 3 and summary["frames"] == 8 and len(index) == 11
    assert sum("couldn't write frame 2" in m for m in said) == 1
    play(dog, screen, [1], t)  # and the watchdog still sees the next one
    assert dog.recording is not None and len(alarms) == 2


def test_the_alarm_is_set_and_raised_before_anything_of_the_recording_is_written(tmp_path):
    screen, events = Screen(), []
    folder = tmp_path / "lie_detector"
    flag = threading.Event()

    def raised(frame):
        events.append(("alarm set", flag.is_set()))
        events.append(("anything written", folder.exists() and any(folder.rglob("*"))))
        return None

    dog, *_ = make_dog(tmp_path, screen, flag=flag, on_alarm=raised)
    play(dog, screen, [1])
    assert events == [("alarm set", True), ("anything written", False)]
    assert dog.recording is not None and dog.recording.folder.exists()


def test_a_failed_alarm_or_recording_doesnt_stop_the_watchdog(tmp_path):
    screen = Screen()

    def broken(frame):
        raise OSError(28, "No space left on device")

    dog, flag, said, _ = make_dog(tmp_path, screen, on_alarm=broken)
    (tmp_path / "lie_detector").write_text("a file where the folder should be", encoding="utf-8")
    play(dog, screen, [1, 1, 0, 0, 0, 0, 0])
    assert flag.is_set() and dog.recording is None and dog.cycle_t0 is None
    assert any("raising the alarm failed" in m for m in said)
    assert any("starting the Lie Detector recording failed" in m for m in said)


def test_the_watchdog_thread_records_at_4_a_second_and_finishes_its_recording_when_stopped(tmp_path):
    screen = Screen(1)
    dog, flag, said, alarms = make_dog(tmp_path, screen, interval=0.05)
    dog.start()
    time.sleep(0.6)
    dog.stopped.set()
    dog.join(timeout=10)
    assert not dog.is_alive()
    [(folder, summary, index)] = written(dog)
    assert summary["end"] == "stopped" and summary["outcome"].startswith("unknown")
    gaps = np.diff([r["t"] for r in index])
    assert 8 <= len(index) <= 14 and np.median(gaps) == pytest.approx(0.05, abs=0.02)  # from each look's start
