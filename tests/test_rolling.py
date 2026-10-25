"""The rolling recording (control.rolling): the game's window 10 times a second, the last 5 minutes kept as JPEG in
memory, a gap wherever the game doesn't show, saved on demand through the inbox and on a Lie Detector alarm into its
recording, memory and disk bounded; and the sheet and video of a saved part. With a fake window and a fake capture,
a fake clock, and temp folders: nothing touches the game or the screen."""

import json
import os
import threading
import time

import cv2
import numpy as np
import pytest

from gamelab import window
from gamelab.__main__ import main
from gamelab.analysis import recordings
from gamelab.control import alarm, lie_recorder, rolling
from gamelab.control.rolling import BEFORE, INDEX, SUMMARY, Item, Recorder, Ring
from gamelab.perception import framecheck

AREA = {"left": 100, "top": 50, "width": 64, "height": 48}


class Clock:
    """time.time's stand-in: a tick's worth on at each `step` (near the real time, as status files are read by it).
    The steps are summed apart from the start: added to 1.8e9 s one by one, 0.1 s steps drift 3e-4 s in 3,000."""

    def __init__(self, t: float | None = None):
        self.start, self.since = (time.time() - 400 if t is None else t), 0.0

    def __call__(self) -> float:
        return self.start + self.since

    @property
    def t(self) -> float:
        return self()

    @t.setter
    def t(self, value: float) -> None:
        self.start, self.since = value, 0.0

    def step(self, dt: float = 0.1) -> None:
        self.since += dt


class Game:
    """The game window as a test sets it: `why` "" shows it at `area`; else that's why not."""

    def __init__(self):
        self.why, self.area, self.checks = "", dict(AREA), 0

    def check(self):
        self.checks += 1
        return (None, self.why) if self.why else (dict(self.area), "")


class Grab:
    """Grabs BGRA frames (as mss gives them), each a shade lighter; `during` runs inside a grab (the window covered
    or moved meanwhile)."""

    def __init__(self):
        self.n, self.areas, self.during = 0, [], None

    def __call__(self, area):
        self.n += 1
        self.areas.append(area)
        if self.during:
            self.during()
        return bgra(self.n)


def bgra(n: int, w: int = 64, h: int = 48) -> np.ndarray:
    out = np.full((h, w, 4), (n * 3) % 256, np.uint8)
    out[:, : w // 2, 2] = 250  # some red: the JPEG keeps BGR's order
    out[..., 3] = 255
    return out


@pytest.fixture
def alarm_file(tmp_path, monkeypatch):
    path = tmp_path / "ALARM.json"
    monkeypatch.setattr(alarm, "ALARM", path)
    return path


def recorder(tmp_path, clock=None, game=None, grab=None, **kw) -> Recorder:
    said = []
    kw.setdefault("log", said.append)
    rec = Recorder(game or Game(), grab or Grab(), box=tmp_path / "recorder", lie_root=tmp_path / "lie_detector",
                   recent=tmp_path / "recent", clock=clock or Clock(), **kw)
    rec.said = said
    return rec


def run_ticks(rec: Recorder, n: int, poll: bool = False) -> None:
    for _ in range(n):
        rec.tick()
        if poll:
            rec.poll()
        rec.clock.step(1 / rec.fps)


def lines(folder):
    return [json.loads(s) for s in (folder / INDEX).read_text(encoding="utf-8").splitlines()]


def summary(folder):
    return json.loads((folder / SUMMARY).read_text(encoding="utf-8"))


def test_the_ring_keeps_exactly_the_last_5_minutes_of_moments_at_10_a_second(tmp_path):
    rec = recorder(tmp_path)
    run_ticks(rec, 6000)  # 10 minutes
    items = list(rec.ring.items)
    assert len(items) == 3000 and rec.ring.frames == 3000
    assert items[-1].t - items[0].t == pytest.approx(299.9) and rec.ring.span() == pytest.approx(300.0)
    assert all(b.t - a.t == pytest.approx(0.1, abs=1e-6) for a, b in zip(items, items[1:]))
    # The newest 3,000 grabs, in order: each frame decodes to the grab it was.
    first = cv2.imdecode(np.frombuffer(items[0].jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert np.abs(first.astype(int) - bgra(3001)[..., :3]).mean() < 2 and items[0].size == (64, 48)
    assert rec.grab.n == 6000 and rec.ring.trimmed == 0
    assert len(rec.ring.last(30)) == 300 and rec.ring.last(30)[0] is items[2700]
    rec.close()


def test_a_frame_is_kept_only_while_the_game_shows_else_a_gap_says_why(tmp_path):
    game, grab = Game(), Grab()
    rec = recorder(tmp_path, game=game, grab=grab)
    plan = ["", "", "not in front", "covered", "minimized", "no window", "off the screen", "", "covered", ""]
    for why in plan:
        game.why = why
        rec.tick()
        rec.clock.step()
    items = list(rec.ring.items)
    assert [it.why for it in items] == plan
    assert [it.jpeg is not None for it in items] == [not why for why in plan]
    assert grab.n == plan.count("")  # never grabbed while the game didn't show: nothing of another window
    assert rec.counts == {"frame": 4, "covered": 2, "not in front": 1, "minimized": 1, "no window": 1,
                          "off the screen": 1}
    # The log says when it starts and stops showing.
    said = [m.split(" ", 1)[1] for m in rec.said]
    assert said[:3] == ["recorder: the game is on screen: recording", "recorder: not recording: not in front",
                        "recorder: not recording: covered"]
    rec.close()


def test_a_window_over_the_game_or_the_game_moving_during_the_grab_drops_the_frame(tmp_path):
    game, grab = Game(), Grab()
    rec = recorder(tmp_path, game=game, grab=grab)
    grab.during = lambda: setattr(game, "why", "covered")
    assert rec.tick() == Item(rec.clock(), None, "covered")  # grabbed, then the look after said covered: not kept
    game.why = ""
    grab.during = lambda: game.area.update(left=game.area["left"] + 26)
    rec.clock.step()
    assert rec.tick().why == "moved" and rec.ring.frames == 0
    grab.during = None
    rec.clock.step()
    assert rec.tick().jpeg is not None and grab.n == 3 and game.checks == 6  # a look before each grab and after
    rec.close()


def test_a_grab_that_fails_is_a_gap_and_the_recording_goes_on(tmp_path):
    def broken(area):
        raise OSError("BitBlt failed")
    rec = recorder(tmp_path, grab=broken)
    rec.tick()
    assert rec.ring.items[-1].why == "capture failed" and rec.failures == 1
    assert any("a grab failed (OSError: BitBlt failed" in m for m in rec.said)
    rec.close()


def test_the_game_window_is_asked_as_the_eyes_ask_it_wholly_shown_in_front_with_nothing_over_it(monkeypatch):
    state = {"minimized": False, "on_screen": True, "front": "MapleStory", "shows": True}
    grids = []
    monkeypatch.setattr(window, "find_game", lambda title: 7)
    monkeypatch.setattr(window, "minimized", lambda hwnd: state["minimized"])
    monkeypatch.setattr(window, "client_rect", lambda hwnd: dict(AREA))
    monkeypatch.setattr(window, "on_screen", lambda area: state["on_screen"])
    monkeypatch.setattr(window, "foreground_title", lambda: state["front"])
    monkeypatch.setattr(window, "shows", lambda hwnd, area, grid=None: grids.append(grid) or state["shows"])
    game = rolling.GameWindow("MapleStory")
    assert game.check() == (AREA, "") and grids == [framecheck.COVER_GRID]
    for key, value, why in (("shows", False, "covered"), ("front", "Claude", "not in front"),
                            ("on_screen", False, "off the screen"), ("minimized", True, "minimized")):
        state[key] = value
        assert game.check() == (None, why)

    def gone(title):
        raise RuntimeError("no window titled 'MapleStory'")
    monkeypatch.setattr(window, "find_game", gone)
    assert game.check() == (None, "no window")


def test_save_recent_asks_the_running_recorder_through_its_inbox_and_gets_the_folder(tmp_path):
    rec = recorder(tmp_path)
    game = rec.game
    run_ticks(rec, 100)
    game.why = "not in front"
    run_ticks(rec, 15)
    game.why = ""
    run_ticks(rec, 20)
    rec.clock.t = time.time()  # (the status is read against the real clock)
    rec.write_status()
    got = {}
    asking = threading.Thread(target=lambda: got.update(rolling.ask(5, "pop up?", box=tmp_path / "recorder")))
    asking.start()
    while asking.is_alive():  # the recorder's loop: a request is taken at the next poll
        rec.poll()
        time.sleep(0.01)
    assert "folder" in got, got
    folder = tmp_path / "recent" / os.path.basename(got["folder"])
    assert folder.name.endswith("-pop-up") and folder.is_dir()
    assert got["seconds"] == 5.0 and got["frames"] == 35 and got["gaps"] == {"not in front": 1.5}
    assert got["window"] == [64, 48] and got["bytes"] == sum(p.stat().st_size for p in folder.glob("*.jpg"))
    index = lines(folder)
    assert [r.get("gap") for r in index if "gap" in r] == ["not in front"]
    assert index[0]["i"] == 0 and index[0]["file"] == "00000.jpg" and index[0]["t"] == 0.0
    assert index[15] == {"gap": "not in front", "t": 1.5, "until": 2.9, "ticks": 15}
    assert index[16]["t"] == 3.0 and index[-1] == {"i": 34, "file": "00034.jpg", "t": 4.9,
                                                    "at": rolling.clock_time(rec.ring.items[-1].t)}
    s = summary(folder)
    assert s["kind"] == "save-recent" and s["name"] == "pop-up" and s["ticks"] == 50 and s["fps"] == 10.0
    assert s["jpeg_quality"] == lie_recorder.JPEG_QUALITY
    assert not list((tmp_path / "recorder" / "inbox").iterdir())  # taken, answered and cleaned up
    assert not list((tmp_path / "recorder" / "outbox").iterdir())
    back = cv2.imread(str(folder / "00034.jpg"))  # the 120th grab (none while the game wasn't in front)
    assert np.abs(back.astype(int) - bgra(120)[..., :3]).mean() < 2
    rec.close()


def test_save_recent_says_so_when_no_recorder_runs_or_none_takes_the_request(tmp_path):
    box = tmp_path / "recorder"
    with pytest.raises(rolling.NotRunning, match="no recorder is running"):
        rolling.ask(box=box)
    box.mkdir()
    st = {"updated": time.time() - 60, "stopped": False}
    (box / "status.json").write_text(json.dumps(st), encoding="utf-8")
    with pytest.raises(rolling.NotRunning, match="older than 30 s"):
        rolling.ask(box=box)
    st["updated"] = time.time()
    (box / "status.json").write_text(json.dumps(st), encoding="utf-8")
    with pytest.raises(rolling.NotRunning, match="didn't take the request"):
        rolling.ask(box=box, taken_s=0.2)
    assert not list((box / "inbox").iterdir())  # withdrawn


def test_a_lie_detector_alarm_saves_the_5_minutes_before_it_into_the_player_s_recording(tmp_path, alarm_file):
    rec = recorder(tmp_path)
    run_ticks(rec, 3500, poll=True)  # nearly 6 minutes, the alarm looked for every 0.5 s
    seen = rec.clock()
    alarm_file.write_text(json.dumps({"reason": "lie_detector", "time": "2026-10-24 21:50:00"}), encoding="utf-8")
    os.utime(alarm_file, (seen, seen))
    run_ticks(rec, 5, poll=True)  # it's seen at the next look
    assert rec.jobs.unfinished_tasks == 1
    run_ticks(rec, 15)  # the player pages, then starts its recording
    player = tmp_path / "lie_detector" / time.strftime(lie_recorder.STAMP, time.localtime(seen + 1))
    player.mkdir(parents=True)
    assert rec.flush(10)
    before = player / BEFORE
    s = summary(before)
    assert s["kind"] == "before the Lie Detector" and s["alarm"]["time"] == "2026-10-24 21:50:00"
    assert s["player_recording"] == str(player)
    # The 5 minutes up to the alarm (3,000 moments as it was seen; its moment 0.5 s at most before), and on until the
    # player's recording began.
    assert 3000 <= s["ticks"] <= 3020 and s["alarm_t"] == pytest.approx(300.0, abs=0.6)
    assert s["frames"] == s["ticks"] == len(list(before.glob("*.jpg"))) and s["seconds"] == s["ticks"] / 10
    assert any("saved" in m and "before the Lie Detector" in m for m in rec.said)
    # The same alarm isn't saved again; the player's own folder is left as it was, but for the part.
    run_ticks(rec, 20, poll=True)
    assert rec.flush(10) and sorted(p.name for p in player.iterdir()) == [BEFORE]
    rec.close()


def test_an_alarm_with_no_recording_of_the_player_s_goes_to_the_recent_folder(tmp_path, alarm_file):
    rec = recorder(tmp_path, folder_wait=0.3)
    run_ticks(rec, 50, poll=True)
    alarm_file.write_text(json.dumps({"reason": "lie_detector (seen by watch)", "time": "2026-10-24 21:51:00"}),
                          encoding="utf-8")
    run_ticks(rec, 5, poll=True)
    assert rec.flush(10)
    (folder,) = (tmp_path / "recent").iterdir()
    assert folder.name.endswith("-lie-detector") and summary(folder)["player_recording"] is None
    assert summary(folder)["frames"] >= 50
    assert any("no recording of the player's began" in m for m in rec.said)
    rec.close()


def test_an_alarm_up_as_it_starts_is_not_saved_but_the_next_one_is(tmp_path, alarm_file):
    alarm_file.write_text(json.dumps({"reason": "lie_detector", "time": "2026-10-24 20:00:00"}), encoding="utf-8")
    rec = recorder(tmp_path, folder_wait=0.2)
    run_ticks(rec, 20, poll=True)
    assert rec.flush(5) and not (tmp_path / "recent").exists()
    alarm_file.write_text(json.dumps({"reason": "unknown"}), encoding="utf-8")  # (read as it's written: no time yet)
    run_ticks(rec, 10, poll=True)
    assert rec.jobs.unfinished_tasks == 0
    alarm_file.write_text(json.dumps({"reason": "lie_detector", "time": "2026-10-24 21:00:00"}), encoding="utf-8")
    run_ticks(rec, 10, poll=True)
    assert rec.flush(5) and len(list((tmp_path / "recent").iterdir())) == 1
    rec.close()


def test_memory_is_bounded_the_oldest_frames_go_past_the_cap(tmp_path):
    ring = Ring(max_bytes=10_000)
    for k in range(100):
        ring.add(Item(k / 10, b"x" * 1000))
        ring.add(Item(k / 10 + 0.05, None, "covered"))
        assert ring.bytes <= 10_000
    assert ring.frames == 10 and ring.trimmed == 90  # younger than 5 minutes, gone for the cap
    assert ring.items[0].jpeg is not None and ring.items[0].t == 9.0
    clock_set_back = Ring(max_items=50)
    for k in range(200):
        clock_set_back.add(Item(1000.0 - k, None, "covered"))
    assert len(clock_set_back.items) == 50


def test_disk_is_bounded_the_oldest_saves_go_and_a_lie_detector_cycle_counts_its_part_before(tmp_path):
    recent = tmp_path / "recent"
    for n, stamp in enumerate(("20261024-100000", "20261024-110000", "20261024-120000")):
        rolling.write_part(recent / stamp, [Item(1000.0 + k / 10, b"y" * 1000, size=(4, 4)) for k in range(10)],
                           {"kind": "save-recent"})
    rec = recorder(tmp_path, recent_keep=25_000)
    run_ticks(rec, 10)
    rec.clock.t = time.time()
    rec.write_status()
    got = {}
    asking = threading.Thread(target=lambda: got.update(rolling.ask(1, box=tmp_path / "recorder")))
    asking.start()
    while asking.is_alive():
        rec.poll()
        time.sleep(0.01)
    names = sorted(p.name for p in recent.iterdir())
    assert names[:2] == ["20261024-110000", "20261024-120000"] and len(names) == 3  # the oldest went
    assert any("deleted the 1 oldest saves" in m for m in rec.said)
    # The Lie Detector's recordings are pruned counting the part before each.
    cycle = tmp_path / "lie_detector" / "20261024-090000"
    rolling.write_part(cycle / BEFORE, [Item(1.0, b"z" * 5000, size=(4, 4))], {"kind": "before the Lie Detector"})
    (cycle / "00000.jpg").write_bytes(b"j" * 100)
    assert lie_recorder.size_of(cycle) == 5100 + (cycle / BEFORE / INDEX).stat().st_size \
        + (cycle / BEFORE / SUMMARY).stat().st_size
    assert lie_recorder.prune(tmp_path / "lie_detector", keep_bytes=1000, dry_run=True)[0][0] == cycle
    rec.close()


def test_run_ticks_10_times_a_second_and_leaves_a_status_saying_it_stopped(tmp_path):
    rec = recorder(tmp_path, clock=time.time)
    rec.run(0.55)
    assert 5 <= len(rec.ring.items) <= 7
    st = rolling.read_status(tmp_path / "recorder")
    assert st["stopped"] is True and st["showing"] is True and st["frames"] == len(rec.ring.items)
    assert rolling.running(tmp_path / "recorder") is None
    text = "\n".join(rolling.status_lines(tmp_path / "recorder"))
    assert text.startswith("recorder stopped") and "the game on screen: recording" in text
    assert "frames a second" in text and "of a core" in text


def test_the_status_is_written_though_a_reader_has_it_open_for_a_moment(tmp_path, monkeypatch):
    rec = recorder(tmp_path)
    rec.write_status()
    path = tmp_path / "recorder" / "status.json"
    monkeypatch.setattr(rolling, "STATUS_TRIES", 2)
    with open(path, encoding="utf-8"):  # `recorder --status` reading it: Windows won't replace it meanwhile
        rec.write_status(stopped=True)
    assert rolling.read_status(tmp_path / "recorder")["stopped"] is False
    assert "couldn't write its status" in rec.said[-1]
    monkeypatch.setattr(rolling, "STATUS_TRIES", 200)
    held = open(path, encoding="utf-8")
    threading.Timer(0.05, held.close).start()
    rec.write_status(stopped=True)  # tried again until the reader let go
    assert rolling.read_status(tmp_path / "recorder")["stopped"] is True
    rec.close()


def test_a_saved_part_makes_a_sheet_and_a_video_and_so_does_a_lie_detector_recording(tmp_path):
    items = [Item(1000.0 + k / 10, rolling.encode(bgra(k)), size=(64, 48)) for k in range(30)]
    items[10:15] = [Item(it.t, None, "covered") for it in items[10:15]]
    folder = tmp_path / "part"
    rolling.write_part(folder, items, {"kind": "save-recent"})
    rec = recordings.load_folder(folder)
    assert len(rec.frames) == 25 and rec.gaps == [(1.0, 1.4, "covered")]
    assert main(["sheet", str(folder), "--every", "5", "--shrink", "2"]) == 0
    sheet = cv2.imread(str(folder / "sheet.png"))
    assert sheet.shape == (24, 8 * 32, 3)  # 5 tiles of 32x24 in a row of 8
    assert main(["sheet", str(folder), "--video"]) == 0
    video = cv2.VideoCapture(str(folder / "video.mp4"))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 30 and video.get(cv2.CAP_PROP_FPS) == pytest.approx(10)
    shown = [video.read()[1] for _ in range(30)]
    video.release()
    assert abs(shown[12][-4:].mean() - recordings.GAP_GREY) < 15  # the gap's grey card (its text in the middle)
    assert abs(shown[20][-4:].mean() - recordings.GAP_GREY) > 30
    # A Lie Detector recording (4 grabs a second, a grab with no frame a line of its own) reads the same way.
    cycle = tmp_path / "cycle"
    cycle.mkdir()
    cv2.imwrite(str(cycle / "00000.jpg"), bgra(5)[..., :3])
    cv2.imwrite(str(cycle / "00001.jpg"), bgra(9)[..., :3])
    (cycle / INDEX).write_text("\n".join(json.dumps(r) for r in (
        {"t": 0.0, "i": 0, "seen": True, "scores": {}, "file": "00000.jpg"}, {"t": 0.25, "game": "covered"},
        {"t": 0.5, "i": 1, "seen": False, "scores": {}, "file": "00001.jpg"})) + "\n{\"t\": 0.7", encoding="utf-8")
    rec = recordings.load_folder(cycle)
    assert [(t, f.name, label) for t, f, label in rec.frames] == [(0.0, "00000.jpg", "0.00s"), (0.5, "00001.jpg", "0.50s")]
    assert rec.gaps == [(0.25, 0.25, "covered")]
    assert recordings.folder_video(rec, cycle / "video.mp4") == 6


def test_only_one_recorder_runs_at_a_time():
    name = f"Local\\gamelab-rolling-recorder-test-{os.getpid()}"
    first = rolling.only_one(name)
    assert first is not None and rolling.only_one(name) is None
