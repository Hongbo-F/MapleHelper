"""The Lie Detector's whole cycle recorded (control.lie_recorder): every frame the watchdog grabs from the first sight
until it has been gone a while, with an index and a summary, under a disk cap. All in temp folders, with made-up
frames and scores; nothing touches the game."""

import contextlib
import functools
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


def test_a_grab_added_as_the_writer_finds_the_queue_empty_is_still_written(tmp_path, monkeypatch):
    # The watchdog adds its last grab and ends the recording in the same look: the writer, between finding the
    # queue empty and looking whether it was ended, had quit with that grab unwritten and uncounted.
    found_empty, go_on = threading.Event(), threading.Event()
    armed = {"on": False}

    class Queue(lie_recorder.queue.Queue):
        def get(self, block=True, timeout=None):
            try:
                return super().get(block, timeout)
            except lie_recorder.queue.Empty:
                if armed["on"]:
                    armed["on"] = False
                    found_empty.set()  # the writer has found it empty ...
                    go_on.wait(5)      # ... and Python switches to the watchdog before the writer goes on
                raise

    monkeypatch.setattr(lie_recorder.queue, "Queue", Queue)
    rec = Recording(tmp_path, log=lambda m: None)
    rec.add(0.0, frame(), COUNTDOWN)
    for _ in range(250):
        if (rec.folder / INDEX).exists():
            break
        time.sleep(0.02)
    armed["on"] = True
    assert found_empty.wait(5)
    rec.add(0.25, frame(), COUNTDOWN)  # the cycle's last look ...
    rec.finish("cap")                  # ... and its end
    go_on.set()
    assert rec.join(timeout=10)
    summary = lie_recorder.summary_of(rec.folder)
    assert summary["grabs"] == 2 and summary["frames"] == 2 and len(lines(rec.folder)) == 2


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


def test_one_that_cant_be_deleted_wholly_counts_only_what_is_left_of_it(tmp_path):
    a = tmp_path / "20261020-010000"
    a.mkdir()
    (a / "00000.jpg").write_bytes(b"x" * 2990)
    (a / "00001.jpg").write_bytes(b"x" * 10)
    b = _fake_recording(tmp_path, "20261021-010000", 280)
    c = _fake_recording(tmp_path, "20261022-010000", 280)
    with open(a / "00001.jpg", "rb"):  # a frame open in a viewer: Windows won't delete it
        gone = lie_recorder.prune(tmp_path, keep_bytes=1000)
    # The rest of the oldest went; it and the two newer ones (about 300 bytes each) fit the 1,000 kept.
    assert gone == [] and lie_recorder.size_of(a) == 10 and b.exists() and c.exists()
    nb, nc = lie_recorder.size_of(b), lie_recorder.size_of(c)
    assert lie_recorder.prune(tmp_path, keep_bytes=0) == [(a, 10), (b, nb), (c, nc)]  # closed, the rest of it goes
    assert not a.exists()


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
    """A made-up frame the fake spotter scores as `kind` (an index into FORMS), or why there's none (a str); for
    ("covered", kind), that frame with another window over the game."""
    if isinstance(kind, tuple):
        return game(kind[1]), kind[0]
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


def test_a_game_with_another_window_over_it_is_checked_but_not_saved_nor_counted_as_the_lie_detector_gone(tmp_path):
    screen = Screen()
    dog, flag, said, alarms = make_dog(tmp_path, screen)
    covered = ("covered", 0)
    t = play(dog, screen, [("covered", 1)])  # seen round a window over the game: raised as ever
    assert flag.is_set() and alarms == [True]
    t = play(dog, screen, [1] + [covered] * 8, t)  # 2 s covered: not gone
    assert dog.recording is not None
    t = play(dog, screen, [("covered", 2), 0, 0, 0, 0], t)  # seen through it again; then gone 0.75 s
    assert dog.recording is not None
    play(dog, screen, [0], t)  # 1 s
    [(folder, summary, index)] = written(dog)
    assert summary["not_showing"] == {"covered": 10} and summary["frames"] == 6 and summary["grabs"] == 16
    assert len(list(folder.glob("*.jpg"))) == 6  # only the game's own pictures
    assert index[0] == {"t": 0.0, "game": "covered", "seen": True, "scores": COUNTDOWN}
    assert index[2] == {"t": 0.5, "game": "covered", "seen": False, "scores": GONE}
    assert index[1]["file"] == "00000.jpg" and index[11]["file"] == "00001.jpg"
    assert summary["first_seen"] == 0.0 and summary["last_seen"] == 2.5
    assert summary["forms"] == {"countdown": {"first": 0.0, "last": 0.25, "frames": 2},
                                "mini-game": {"first": 2.5, "last": 2.5, "frames": 1}}
    assert summary["end"] == "gone" and summary["window"] == [64, 48]


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


# --- the player stays up through it, locked, and stops once it's cleared and the screen clear (control.player) ---

from types import SimpleNamespace  # noqa: E402

from gamelab.control import alarm, client  # noqa: E402
from gamelab.control import player as player_mod  # noqa: E402
from gamelab.control.player import LIE_DETECTOR_ERROR, Player  # noqa: E402


@pytest.fixture
def pages(monkeypatch):
    """Paging counted instead of done: the titles."""
    out = []
    monkeypatch.setattr(alarm, "page", lambda title, text: out.append(title))
    return out


@pytest.fixture
def alarm_file(tmp_path, monkeypatch, pages):
    """The alarm file in tmp_path for everyone who reads it (and paging counted)."""
    path = tmp_path / "ALARM.json"
    for mod in (alarm, client, player_mod):
        monkeypatch.setattr(mod, "ALARM", path)
    return path


def refused(flag):
    """Whether the player refuses a plan now (it does before looking for the game or pressing anything)."""
    try:
        return Player({}, "no such window", flag).run({"events": [[0, "f", "down"]]}) == {"error": LIE_DETECTOR_ERROR}
    except ValueError:  # it went on to check the plan (its key isn't allowed here): not refused
        return False


def test_after_the_lie_detector_plans_are_refused_until_the_player_is_started_again(tmp_path, alarm_file, pages):
    screen = Screen()
    dog, flag, said, _ = make_dog(tmp_path, screen, on_alarm=alarm.raise_alarm)
    t = play(dog, screen, [0, 0])
    assert not refused(flag)
    t = play(dog, screen, [1, 1, 2], t)
    assert refused(flag) and alarm_file.exists() and pages == ["MapleStory Lie Detector on screen"]
    # Gone well past the end stretch, the alarm not cleared: refused, and the player doesn't stop.
    t = play(dog, screen, [0] * 12, t)
    assert dog.cycle_t0 is None and refused(flag) and not dog.cleared.is_set()
    assert any(m.endswith("its cycle is over (waiting for `clear-alarm`, then the player stops)") for m in said)
    alarm.clear_alarm()  # a person, the screen clear already: the player stops at the next look
    t = play(dog, screen, [0], t)
    assert dog.cleared.is_set() and "the player stops" in said[-1]
    # It never takes plans again by itself: a file any program can delete would be all that stood between the keys
    # and a Lie Detector the spotter can't see. Only a player started again (the user's UAC click) does.
    play(dog, screen, [0] * 8, t)
    assert refused(flag)
    assert len(written(dog)) == 1


def test_cleared_while_it_is_still_on_screen_it_is_raised_again(tmp_path, alarm_file, pages):
    screen = Screen()
    dog, flag, said, _ = make_dog(tmp_path, screen, on_alarm=alarm.raise_alarm)
    t = play(dog, screen, [1, 1, 0, 0])  # the countdown, then half a second without it (under the 1 s)
    alarm.clear_alarm()                  # cleared mid-cycle
    t = play(dog, screen, [3, 3], t)     # the Truth Booth within the end stretch: the same cycle
    assert alarm_file.exists() and len(pages) == 2 and dog.ended == []
    assert any("cleared while the Lie Detector is still on screen: raised again" in m for m in said)
    with pytest.raises(client.LieDetectorAlarm):
        client.check_alarm()  # gamelab's commands go on refusing (exit code 3)
    t = play(dog, screen, [0] * 5, t)  # its cycle over, the alarm up: the player doesn't stop
    assert dog.cycle_t0 is None and not dog.cleared.is_set() and refused(flag)
    alarm.clear_alarm()
    play(dog, screen, [0], t)
    assert dog.cleared.is_set() and len(pages) == 2
    [(_, summary, _)] = written(dog)
    assert summary["outcome"] == "sent to the Truth Booth"


def test_cleared_while_it_is_up_but_not_seen_the_player_stops_at_the_cycle_s_end_sending_nothing(
        tmp_path, alarm_file, pages):
    # (The figure tracking: on screen, but nothing the spotter knows.) Nothing raises it again, and the player stops
    # once the end stretch has run out: nothing is sent, and only a person can start it again.
    screen = Screen()
    dog, flag, said, _ = make_dog(tmp_path, screen, on_alarm=alarm.raise_alarm)
    t = play(dog, screen, [2, 0])
    alarm.clear_alarm()
    t = play(dog, screen, [0, 0, 0], t)  # 0.75 s
    assert not dog.cleared.is_set()
    play(dog, screen, [0], t)  # 1 s
    assert dog.cleared.is_set() and refused(flag) and len(pages) == 1


def test_a_second_lie_detector_before_the_alarm_is_cleared_is_a_new_cycle_paged_and_recorded(
        tmp_path, alarm_file, pages):
    screen = Screen()
    dog, flag, said, _ = make_dog(tmp_path, screen, on_alarm=alarm.raise_alarm)
    t = play(dog, screen, [1, 1] + [0] * 5)
    assert dog.cycle_t0 is None
    play(dog, screen, [3, 3] + [0] * 5, t)
    assert flag.is_set() and alarm_file.exists() and len(pages) == 2 and not dog.cleared.is_set()
    first, second = written(dog)
    assert first[0] != second[0] and second[1]["outcome"] == "sent to the Truth Booth"
    assert first[1]["outcome"] == "answered"


def test_an_alarm_file_that_cant_be_written_is_still_paged(tmp_path, monkeypatch, pages):
    blocker = tmp_path / "lab_is_a_file"
    blocker.write_text("x", encoding="utf-8")
    for mod in (alarm, client, player_mod):  # its folder can't be made, as a full or failing disk would refuse it
        monkeypatch.setattr(mod, "ALARM", blocker / "ALARM.json")
    with pytest.raises(OSError):
        alarm.raise_alarm(frame())
    assert pages == ["MapleStory Lie Detector on screen"]  # raised before the paging, it had told nobody
    screen = Screen()
    dog, flag, said, _ = make_dog(tmp_path, screen, on_alarm=alarm.raise_alarm)
    play(dog, screen, [1, 0])
    assert flag.is_set() and len(pages) == 2 and any("raising the alarm failed" in m for m in said)
    dog.recording.finish("stopped")
    assert dog.recording.join(timeout=10)


def test_an_alarm_file_that_cant_be_looked_at_is_up(tmp_path, monkeypatch):
    # Path.exists answers False for any error (an os.stat that raises "the name is wrong", WinError 123), which
    # had read as a person clearing it.
    bad = tmp_path / "bad?name" / "ALARM.json"
    with pytest.raises(OSError):
        os.stat(bad)
    assert not bad.exists()
    monkeypatch.setattr(player_mod, "ALARM", bad)
    assert player_mod.alarm_up() is True
    monkeypatch.setattr(player_mod, "ALARM", tmp_path / "ALARM.json")
    assert player_mod.alarm_up() is False
    (tmp_path / "ALARM.json").write_text("{}", encoding="utf-8")
    assert player_mod.alarm_up() is True


def test_an_alarm_file_that_couldnt_be_written_keeps_the_player_locked(tmp_path, alarm_file):
    # No alarm file means no person can be said to have cleared it: up and locked until it's stopped.
    screen = Screen()
    dog, flag, said, _ = make_dog(tmp_path, screen, on_alarm=lambda frame: None)
    play(dog, screen, [1] + [0] * 20)
    assert flag.is_set() and dog.cycle_t0 is None and not alarm_file.exists() and not dog.cleared.is_set()
    assert any("couldn't be written: locked until it's stopped" in m for m in said)


def fake_lab(tmp_path, monkeypatch, screen, end_s=0.4, cap_s=60.0):
    """serve() with everything outside faked: the lab in tmp_path (the inbox and outbox the client's, as
    tests/conftest.py points them), the game a fake screen, the config made up."""
    for name, path in (("LAB", tmp_path), ("INBOX", client.INBOX), ("OUTBOX", client.OUTBOX),
                       ("STOP", tmp_path / "STOP"), ("PLAYER_LOG", tmp_path / "player.log")):
        monkeypatch.setattr(player_mod, name, path)
    monkeypatch.setattr(player_mod.config, "load",
                        lambda path, combos: SimpleNamespace(keys=None, game_window_title="MapleStory"))
    monkeypatch.setattr(player_mod.win32, "precise_timers", contextlib.nullcontext)
    monkeypatch.setattr(player_mod.win32, "enable_dpi_awareness", lambda: None)
    monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "MapleStory")
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "client_rect",
                        lambda hwnd: {"left": 0, "top": 0, "width": 64, "height": 48})
    monkeypatch.setattr(player_mod, "Watchdog", functools.partial(
        Watchdog, spotter=Spotter(), grab=screen, interval=0.02, end_s=end_s, cap_s=cap_s,
        folder=tmp_path / "lie_detector"))


class Served:
    """serve() on a thread of its own, in fake_lab. A plan with no keys runs through Player.run as any would."""

    def __init__(self, tmp_path, monkeypatch, screen, end_s=0.4, cap_s=60.0):
        self.screen, self.end_s, self.lab = screen, end_s, tmp_path
        fake_lab(tmp_path, monkeypatch, screen, end_s, cap_s)
        self.thread = threading.Thread(target=player_mod.serve, name="served-player")
        self.thread.start()
        self.wait_for(lambda: "player started" in self.log())

    def log(self):
        try:
            return (self.lab / "player.log").read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def wait_for(self, cond, timeout=10.0):
        end = time.monotonic() + timeout
        while not cond():
            assert time.monotonic() < end, f"timed out; the player's log:\n{self.log()}"
            time.sleep(0.01)

    def send(self, name):
        """A plan put in the inbox as the client does (but past its check of the alarm); its answer's file."""
        client.INBOX.mkdir(parents=True, exist_ok=True)
        tmp = client.INBOX / f"{name}.tmp"
        tmp.write_text(json.dumps({"events": [], "capture": {"lead_ms": 0}, "focus": False}), encoding="utf-8")
        tmp.replace(client.INBOX / f"{name}.json")
        return client.OUTBOX / f"{name}.json"

    def plan(self, name):
        """A plan sent (`send`), and the player's answer."""
        out = self.send(name)
        self.wait_for(out.exists)
        result = client._read_result(out)
        out.unlink()
        return result

    def ended(self, timeout=10.0):
        """Wait for the player to stop by itself; its log."""
        self.thread.join(timeout=timeout)
        assert not self.thread.is_alive(), f"still running; the player's log:\n{self.log()}"
        assert self.log().rstrip().endswith("player stopped")  # (what `python -m gamelab stop` waits for)
        return self.log()

    def stop(self):
        """Stop it (`python -m gamelab stop`), if it's still running."""
        if self.thread.is_alive():
            (self.lab / "STOP").touch()
            self.ended(timeout=30)


def test_the_player_stays_up_through_the_lie_detector_recording_it_and_stops_once_it_is_cleared(
        tmp_path, monkeypatch, alarm_file, pages):
    screen = Screen()
    served = Served(tmp_path, monkeypatch, screen)
    try:
        assert "error" not in served.plan("before")
        screen.kind = 1
        served.wait_for(alarm_file.exists)
        assert served.plan("during")["error"] == LIE_DETECTOR_ERROR
        with pytest.raises(client.LieDetectorAlarm):  # the stack's side, as before
            client.submit([], focus=False, timeout=5)
        screen.kind = 0
        served.wait_for(lambda: "its cycle is over" in served.log())
        time.sleep(2 * served.end_s)  # gone long enough, not cleared: still up, still refusing
        assert served.thread.is_alive() and served.plan("gone")["error"] == LIE_DETECTOR_ERROR
        alarm.clear_alarm()
        log = served.ended()
    finally:
        served.stop()
    assert "the Lie Detector alarm was cleared and none has been on screen for 0.4 s: the player stops" in log
    assert "stopping after the Lie Detector, its alarm cleared: start the player again" in log
    [folder] = (tmp_path / "lie_detector").iterdir()
    assert json.loads((folder / SUMMARY).read_text(encoding="utf-8"))["end"] == "gone"
    assert log.count("LIE DETECTOR seen") == 1 and log.count("recording the Lie Detector") == 1
    assert log.count("Lie Detector recording ended (gone)") == 1


def test_the_player_wont_start_with_the_alarm_up(tmp_path, monkeypatch, alarm_file):
    alarm_file.write_text(json.dumps({"reason": "lie_detector", "time": "2026-10-24 10:15:00"}), encoding="utf-8")
    fake_lab(tmp_path, monkeypatch, Screen(2))
    player_mod.serve()  # returns at once
    log = (tmp_path / "player.log").read_text(encoding="utf-8")
    assert "refusing to start: a Lie Detector alarm hasn't been cleared" in log and "player started" not in log
    assert not (tmp_path / "lie_detector").exists()


def test_the_idle_limit_waits_for_a_lie_detector_s_recording_to_end(tmp_path, monkeypatch, alarm_file, pages):
    # Idle 24 hours with the Lie Detector come in the last minute, the player had stopped in the middle of its
    # recording, nothing left watching.
    monkeypatch.setattr(player_mod, "IDLE_LIMIT_S", 0.5)
    served = Served(tmp_path, monkeypatch, Screen(1), cap_s=1.5)  # on screen the whole time
    try:
        log = served.ended()
    finally:
        served.stop()
    [folder] = (tmp_path / "lie_detector").iterdir()
    assert json.loads((folder / SUMMARY).read_text(encoding="utf-8"))["end"] == "cap"  # not "stopped"
    assert "stopping with the Lie Detector alarm up (locked)" in log


def test_the_player_takes_no_plan_before_its_watchdog_has_looked_at_the_game(tmp_path, monkeypatch, alarm_file,
                                                                             pages):
    looking = threading.Event()

    class Slow(Screen):
        """The first grab held until the test lets it go."""

        def __call__(self):
            looking.wait(10)
            return super().__call__()

    served = Served(tmp_path, monkeypatch, Slow(1))  # the Lie Detector on screen as it starts (the Truth Booth, say)
    try:
        out = served.send("first")
        time.sleep(0.3)
        assert not out.exists()  # waiting for the first look
        looking.set()
        served.wait_for(out.exists)
        assert client._read_result(out)["error"] == LIE_DETECTOR_ERROR  # seen before any key of it
    finally:
        served.stop()


def test_the_player_stops_if_its_watchdog_does(tmp_path, monkeypatch):
    class Dead(Watchdog):
        def run(self):  # as when mss can't be opened: the thread ends at once
            pass

    fake_lab(tmp_path, monkeypatch, Screen())
    monkeypatch.setattr(player_mod, "Watchdog", functools.partial(Dead, spotter=Spotter(), grab=Screen()))
    thread = threading.Thread(target=player_mod.serve, name="served-player")
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    log = (tmp_path / "player.log").read_text(encoding="utf-8")
    assert "the Lie Detector watch has stopped: stopping, as nothing would see one" in log
    assert log.rstrip().endswith("player stopped")


def test_a_lie_detector_seen_while_a_plan_waits_on_the_window_brings_nothing_forward_and_sends_nothing(monkeypatch):
    # Seen during the resize's wait (or the bring-to-front's), the plan stops before the game is brought forward and
    # before its first key: those waits weren't watched (the keys' own are).
    from maplehelper.keys import parse
    flag, brought = threading.Event(), []
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "resize_client", lambda hwnd, w, h: flag.set())  # seen meanwhile
    monkeypatch.setattr(player_mod.window, "bring_to_front", lambda title: brought.append(title))
    monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "another window")
    plan = {"id": "p", "events": [[0, "f", "down"], [10, "f", "up"]], "window_client": [1366, 768], "focus": True}
    assert Player({"f": parse("f")}, "MapleStory", flag).run(plan) == {"error": LIE_DETECTOR_ERROR}
    assert brought == []  # (and no key: conftest refuses win32.send_key)
    flag.clear()
    monkeypatch.setattr(player_mod.window, "bring_to_front", lambda title: (brought.append(title), flag.set()))
    plan.pop("window_client")
    assert Player({"f": parse("f")}, "MapleStory", flag).run(plan) == {"error": LIE_DETECTOR_ERROR}
    assert brought == ["MapleStory"]


# --- clear-alarm waits for the recording to end, and says how it went ---

def test_clear_alarm_waits_for_the_lie_detector_s_recording_to_end(tmp_path, monkeypatch, alarm_file, capsys):
    from gamelab.__main__ import LIE_DETECTOR_EXIT, main
    monkeypatch.setattr(lie_recorder, "LIE_RECORDINGS", tmp_path / "lie_detector")
    alarm.raise_alarm(frame())
    rec = Recording(tmp_path / "lie_detector", log=lambda m: None)
    rec.add(0.0, frame(), COUNTDOWN)
    for _ in range(250):
        if (rec.folder / INDEX).exists():
            break
        time.sleep(0.02)
    # Cleared now, a running player would stop a minute after the last sight, the person maybe still at it.
    assert main(["clear-alarm"]) == LIE_DETECTOR_EXIT
    out = capsys.readouterr().out
    assert out.startswith(f"the Lie Detector's recording is still going ({rec.folder}): it ends once the game has "
                          "shown no Lie Detector for 60 s")
    assert "The alarm stays up" in out and alarm_file.exists()
    rec.add(0.25, frame(), GONE)
    finished(rec)
    assert main(["clear-alarm"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("cleared the Lie Detector alarm from ") and not alarm_file.exists()
    assert f"the last recording: {rec.folder} (answered: 2 frames over 0 s, ended: gone)" in out
    assert "a running player stops now" in out and "python -m gamelab player --elevate" in out
    assert main(["clear-alarm"]) == 0
    assert capsys.readouterr().out.startswith("no Lie Detector alarm is set")
