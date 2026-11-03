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
REAL_OVER, REAL_OVERS, REAL_SHOWS, REAL_DESCRIBE = (window.over, window.overs, window.shows,
                                                   window.describe)  # (before conftest refuses them)


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
                   recent=tmp_path / "recent", clips=tmp_path / "alarm_clips", unknown=tmp_path / "booth_unknown",
                   clock=clock or Clock(), **kw)
    rec.said = said
    return rec


def run_ticks(rec: Recorder, n: int, poll: bool = False) -> None:
    for _ in range(n):
        rec.tick()
        if poll:
            rec.poll()
        rec.clock.step(1 / rec.fps)


def a_part(folder, *sizes, kind="x"):
    """A saved part on disk whose frames take these many bytes each (written as they are: what pruning counts)."""
    folder.mkdir(parents=True)
    for i, n in enumerate(sizes):
        (folder / f"{i:05d}.jpg").write_bytes(b"b" * n)
    (folder / INDEX).write_text("".join(json.dumps({"i": i, "file": f"{i:05d}.jpg", "t": i / 10}) + "\n"
                                        for i in range(len(sizes))), encoding="utf-8")
    (folder / SUMMARY).write_text(json.dumps({"kind": kind}), encoding="utf-8")


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
    # The newest 3,000 grabs, in order: each frame's pixels are the grab's, exactly (lossless since 2026-10-25).
    assert np.array_equal(items[0].pixels(), bgra(3001)[..., :3]) and items[0].size == (64, 48)
    assert rec.grab.n == 6000 and rec.ring.trimmed == 0
    assert len(rec.ring.last(30)) == 300 and rec.ring.last(30)[0] is items[2700]
    rec.close()


def test_a_frame_packed_into_the_kept_bgr_picture_packs_as_a_fresh_one_and_reads_back_as_grabbed():
    """pack's BGR picture is kept from frame to frame (a new one a frame was 3 MB of fresh memory, its page faults
    zeroed by the kernel): what it packs is a fresh conversion's, whatever size came before."""
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 256, (48, 64, 4), np.uint8), rng.integers(0, 256, (30, 40, 4), np.uint8)]
    for frame in frames + frames[::-1] + [frames[0][..., :3]]:
        packed = rolling.pack(frame)
        assert packed == rolling._compress(memoryview(np.ascontiguousarray(frame[..., :3])).cast("B"))
        item = Item(0.0, packed, size=(frame.shape[1], frame.shape[0]))
        assert np.array_equal(item.pixels(), frame[..., :3])


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
    assert [it.data is not None for it in items] == [not why for why in plan]
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
    assert rec.tick().data is not None and grab.n == 3 and game.checks == 6  # a look before each grab and after
    rec.close()


def test_a_grab_that_fails_is_a_gap_and_the_recording_goes_on(tmp_path):
    def broken(area):
        raise OSError("BitBlt failed")
    rec = recorder(tmp_path, grab=broken)
    rec.tick()
    assert rec.ring.items[-1].why == "capture failed" and rec.failures == 1
    assert any("a grab failed (OSError: BitBlt failed" in m for m in rec.said)
    rec.close()


def test_the_game_window_is_asked_what_is_over_it_to_note_a_gap_only_when_all_of_it_is(monkeypatch):
    """Since 2026-11-01 another program's window over part of the game leaves the frame kept, the boxes of the client
    it covers noted (`over`); in front or not isn't asked. A game wholly under such windows is a gap ("covered")."""
    state = {"minimized": False, "on_screen": True, "overs": []}
    monkeypatch.setattr(window, "find_game", lambda title: 7)
    monkeypatch.setattr(window, "minimized", lambda hwnd: state["minimized"])
    monkeypatch.setattr(window, "client_rect", lambda hwnd: dict(AREA))
    monkeypatch.setattr(window, "on_screen", lambda area: state["on_screen"])
    monkeypatch.setattr(window, "overs", lambda hwnd, area: list(state["overs"]))
    monkeypatch.setattr(window, "describe", lambda hwnd: f"a window of class 'Toast' ({hwnd})")
    game = rolling.GameWindow("MapleStory")
    assert game.check() == (AREA, "") and game.over == [] and game.detail() == ""
    # A 30x20 box over the client's top-left corner (AREA: 64x48 at 100,50), cut to the client.
    state["overs"] = [(9, (90, 40, 120, 60))]
    assert game.check() == (AREA, "") and game.over == [(0, 0, 20, 10)]
    assert game.detail() == "by a window of class 'Toast' (9)"
    # Two, the nearest named; nearly all of it under them still a frame.
    state["overs"] = [(9, (90, 40, 120, 60)), (11, (100, 55, 300, 300))]
    assert game.check() == (AREA, "") and game.over == [(0, 0, 20, 10), (0, 5, 64, 48)]
    # All of it under them: a gap.
    state["overs"] = [(9, (90, 40, 170, 100))]
    assert game.check() == (None, "covered") and game.over == [] and game.detail() == "by a window of class 'Toast' (9)"
    state["overs"] = []
    for key, value, why in (("on_screen", False, "off the screen"), ("minimized", True, "minimized")):
        state[key] = value
        assert game.check() == (None, why) and game.over == []

    def gone(title):
        raise RuntimeError("no window titled 'MapleStory'")
    monkeypatch.setattr(window, "find_game", gone)
    assert game.check() == (None, "no window")


def test_another_window_over_part_of_the_game_leaves_the_frame_as_grabbed_with_what_it_covered_noted(tmp_path):
    """The user, 2026-11-01: "keep the frame if it's only partially covered ... save metadata on what part was covered
    by other window" (seven Lie Detectors' clips of 2026-10-29 kept no frame: a window over a corner of the game made
    every frame "covered"). The boxes over it, asked before the grab and after it, go with the frame (Item.over, its
    share: Item.covered) into the index; the summary counts such frames."""
    game, grab = Game(), Grab()
    rec = recorder(tmp_path, game=game, grab=grab)
    game.over = [(0, 0, 16, 12)]
    item = rec.tick()
    assert item.why == "" and item.covered == pytest.approx(16 * 12 / (64 * 48)) and item.over == ((0, 0, 16, 12),)
    assert np.array_equal(item.pixels(), bgra(1)[..., :3])  # the frame as grabbed
    assert rec.part_covered == 1 and rec.said[-1].endswith("recorder: recording, another window over part of the game")
    # A window that came during the grab (the look after it) is noted too.
    game.over = []
    grab.during = lambda: setattr(game, "over", [(48, 36, 64, 48)])
    rec.clock.step()
    item = rec.tick()
    assert item.over == ((48, 36, 64, 48),) and np.array_equal(item.pixels(), bgra(2)[..., :3])
    # All of it under such windows: a gap.
    grab.during, game.over = None, [(0, 0, 64, 48)]
    rec.clock.step()
    assert rec.tick() == Item(rec.clock(), None, "covered")
    game.over = []
    rec.clock.step()
    assert rec.tick().covered == 0.0 and rec.said[-1].endswith("recorder: the game is on screen: recording")
    s = rolling.write_part(tmp_path / "part", list(rec.ring.items), {"kind": "x"})
    index = lines(tmp_path / "part")
    assert [r.get("covered") for r in index] == [0.062, 0.062, None, None]
    assert index[0]["over"] == [[0, 0, 16, 12]] and index[1]["over"] == [[48, 36, 64, 48]] and "over" not in index[3]
    assert s["part_covered_frames"] == 2 and s["frames"] == 3 and s["gaps"] == {"covered": 0.1}
    rec.close()


class Desktop:
    """Windows' answers about the top windows (user32's and dwmapi's, as window.over and window.shows ask them):
    `windows` topmost first, each a dict: `hwnd`, `box` (as drawn: left, top, right, bottom), and as wanted `rect`
    (GetWindowRect's, with the invisible borders; else the box), `pid` (the game's is 1), `visible`, `iconic`,
    `cloaked`, `through` (clicks pass through it: WindowFromPoint skips it), `cls`, `no_dwm` (no drawn box)."""

    def __init__(self, windows):
        self.order = [w["hwnd"] for w in windows]
        self.w = {w["hwnd"]: w for w in windows}

    def GetWindow(self, hwnd, cmd):  # noqa: N802 (Windows' names)
        assert cmd == window.GW_HWNDPREV
        i = self.order.index(hwnd)
        return self.order[i - 1] if i else None

    def IsWindowVisible(self, hwnd):  # noqa: N802
        return self.w[hwnd].get("visible", True)

    def IsIconic(self, hwnd):  # noqa: N802
        return self.w[hwnd].get("iconic", False)

    def GetWindowThreadProcessId(self, hwnd, pid):  # noqa: N802
        pid._obj.value = self.w[hwnd].get("pid", 2)
        return 1

    def GetWindowRect(self, hwnd, rect):  # noqa: N802
        w, r = self.w[hwnd], rect._obj
        r.left, r.top, r.right, r.bottom = w.get("rect", w["box"])
        return 1

    def GetClassNameW(self, hwnd, buf, n):  # noqa: N802
        buf.value = self.w[hwnd].get("cls", "Fake")
        return len(buf.value)

    def DwmGetWindowAttribute(self, hwnd, attr, out, size):  # noqa: N802
        w = self.w[hwnd]
        if attr == window.DWMWA_CLOAKED:
            out._obj.value = int(w.get("cloaked", False))
            return 0
        assert attr == window.DWMWA_EXTENDED_FRAME_BOUNDS
        if w.get("no_dwm"):
            return -2147024809  # E_INVALIDARG
        r = out._obj
        r.left, r.top, r.right, r.bottom = w["box"]
        return 0

    def WindowFromPoint(self, point):  # noqa: N802
        for hwnd in self.order:
            w = self.w[hwnd]
            x0, y0, x1, y1 = w["box"]
            shown = w.get("visible", True) and not w.get("iconic") and not w.get("cloaked")
            if shown and not w.get("through") and x0 <= point.x < x1 and y0 <= point.y < y1:
                return hwnd
        return None

    def GetAncestor(self, hwnd, flags):  # noqa: N802
        return hwnd


def test_a_window_between_the_cover_grid_s_points_is_seen_in_the_z_order_above_the_game(tmp_path, monkeypatch):
    """The reviewer's three boxes over the real client area (2448,206 1366x768) pass window.shows at COVER_GRID's
    points: a window over the client's left 8 px, a 200x150 box between the points, a 300x100 notification; so does an
    overlay clicks pass through. window.over sees each, and the recorder keeps no frame and names the window. Windows
    not shown over the game (hidden, minimized, cloaked, only their invisible borders over it, the game's own, below
    it) are left out."""
    area = {"left": 2448, "top": 206, "width": 1366, "height": 768}
    l, t = area["left"], area["top"]
    game_box = (l, t, l + 1366, t + 768)
    the_game = {"hwnd": 1, "pid": 1, "box": game_box}
    covers = {"left 8 px": (l - 500, t + 100, l + 8, t + 400),
              "200x150 between the points": (l + 18, t + 18, l + 218, t + 168),
              "300x100 notification": (l + 1000, t + 654, l + 1300, t + 754),
              "overlay": game_box}

    def desktop(*windows):
        desk = Desktop(list(windows))
        monkeypatch.setattr(window, "user32", desk)
        monkeypatch.setattr(window, "dwmapi", desk)
        return desk

    for what, box in covers.items():
        desktop({"hwnd": 50, "box": box, "through": what == "overlay"}, the_game)
        assert REAL_SHOWS(1, area, grid=framecheck.COVER_GRID), what  # the grid's points miss it
        assert REAL_OVER(1, area) == 50, what
    # Not over it: hidden, minimized, cloaked (another virtual desktop), only its invisible borders over the client
    # (the box as drawn ends at its left edge), the game's own, an empty box, a window below the game.
    beside = (l - 400, t, l, t + 300)
    desk = desktop({"hwnd": 60, "box": game_box, "visible": False}, {"hwnd": 61, "box": game_box, "iconic": True},
                   {"hwnd": 62, "box": game_box, "cloaked": True},
                   {"hwnd": 63, "box": beside, "rect": (beside[0] - 7, t, beside[2] + 7, t + 307)},
                   {"hwnd": 64, "box": game_box, "pid": 1}, {"hwnd": 65, "box": (l + 10, t + 10, l + 10, t + 50)},
                   the_game, {"hwnd": 70, "box": game_box})
    assert REAL_OVER(1, area) is None
    desk.w[63]["no_dwm"] = True  # no box as drawn: GetWindowRect's, its borders over the client
    assert REAL_OVER(1, area) == 63
    # Every one of them, the nearest above the game first (what the recorder blacks out).
    desktop({"hwnd": 50, "box": covers["300x100 notification"]}, {"hwnd": 51, "box": covers["left 8 px"]},
            {"hwnd": 52, "box": game_box, "visible": False}, the_game)
    assert REAL_OVERS(1, area) == [(51, covers["left 8 px"]), (50, covers["300x100 notification"])]
    # The recorder's look at the game: the frame kept as grabbed, the notification's box noted, and its log naming
    # the window over it.
    desktop({"hwnd": 50, "box": covers["300x100 notification"], "cls": "Windows.UI.Core.CoreWindow"}, the_game)
    for name, fn in (("over", REAL_OVER), ("overs", REAL_OVERS), ("describe", REAL_DESCRIBE),
                     ("find_game", lambda title: 1), ("minimized", lambda hwnd: False),
                     ("client_rect", lambda hwnd: dict(area)), ("on_screen", lambda area: True)):
        monkeypatch.setattr(window, name, fn)
    grab = lambda box: np.full((box["height"], box["width"], 4), 200, np.uint8)  # noqa: E731
    rec = recorder(tmp_path, game=rolling.GameWindow("MapleStory"), grab=grab)
    item = rec.tick()
    assert item.why == "" and item.over == ((1000, 654, 1300, 754),)
    assert item.covered == pytest.approx(300 * 100 / (1366 * 768)) and (item.pixels() == 200).all()
    assert rec.said[-1].endswith("recorder: recording, another window over part of the game (by a window of class "
                                 f"'Windows.UI.Core.CoreWindow' at {l + 1000},{t + 654} 300x100)")
    rec.close()


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
    assert s["jpeg_quality"] == rolling.JPEG_QUALITY
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
    assert rec.pending_clips and rec.jobs.unfinished_tasks == 0  # nothing written while the Lie Detector is up ...
    run_ticks(rec, 15)  # the player pages, then starts its recording
    player = tmp_path / "lie_detector" / time.strftime(lie_recorder.STAMP, time.localtime(seen + 1))
    player.mkdir(parents=True)
    run_ticks(rec, 100, poll=True)
    assert rec.flush(60) and not (player / BEFORE).exists()  # ... while the cycle goes on
    _cycle_over(player, 4.0)  # last seen 4 s after its first sight: over 10 s after that, 15 s after the alarm at most
    run_ticks(rec, 40, poll=True)
    assert rec.flush(60)  # (3,000 frames written as JPEG: 10 s, more with the PC busy)
    before = player / BEFORE
    s = summary(before)
    assert s["kind"] == "before the Lie Detector" and s["alarm"]["time"] == "2026-10-24 21:50:00"
    assert s["player_recording"] == str(player)
    # The 5 minutes up to the alarm (3,000 moments as it was seen; its moment 0.5 s at most before), and its first 5 s
    # (BEFORE_TAIL_S: the player's own frames begin about 1.5 s after it).
    assert 3044 <= s["ticks"] <= 3051 and s["alarm_t"] == pytest.approx(300.0, abs=0.6)
    assert s["frames"] == s["ticks"] == len(list(before.glob("*.jpg"))) and s["seconds"] == s["ticks"] / 10
    assert any("saved" in m and "before the Lie Detector" in m for m in rec.said)
    # The same alarm isn't saved again; the player's own folder is left as it was, but for the part.
    run_ticks(rec, 20, poll=True)
    assert rec.flush(10) and sorted(p.name for p in player.iterdir()) == [BEFORE, SUMMARY]
    rec.close()


def test_an_alarm_save_keeps_the_lie_detector_s_recordings_to_their_budget_whatever_the_player_counts(tmp_path,
                                                                                                     alarm_file):
    """A player from before the part before (its size_of counted a folder's files only) never counts the `before/`
    parts: the recorder prunes the Lie Detector's recordings itself when it writes one, the parts counted, the
    player's recording under way spared however big."""
    root = tmp_path / "lie_detector"
    for stamp, size in (("20261020-100000", 4000), ("20261021-100000", 3000), ("20261022-100000", 2000)):
        a_part(root / stamp / BEFORE, size)
        (root / stamp / "00000.jpg").write_bytes(b"j" * 100)  # all an old player counts of it
    rec = recorder(tmp_path, lie_keep=7500)  # the two newer, about 6.3 KB with their index and summary, fit
    run_ticks(rec, 50, poll=True)
    seen = rec.clock()
    alarm_file.write_text(json.dumps({"reason": "lie_detector", "time": "2026-10-24 22:30:00"}), encoding="utf-8")
    os.utime(alarm_file, (seen, seen))
    player = root / time.strftime(lie_recorder.STAMP, time.localtime(seen + 1))
    player.mkdir()
    (player / "00000.jpg").write_bytes(b"p" * 50_000)  # the recording under way: spared, and not in the sum
    run_ticks(rec, 50, poll=True)
    _cycle_over(player, 1.0)
    run_ticks(rec, 120, poll=True)  # (written once the cycle is over)
    assert rec.flush(10)
    assert sorted(p.name for p in root.iterdir()) == ["20261021-100000", "20261022-100000", player.name]
    assert (player / BEFORE / SUMMARY).exists() and (player / "00000.jpg").exists()
    assert any("deleted the 1 oldest Lie Detector recordings" in m for m in rec.said)
    rec.close()


def test_an_alarm_with_no_recording_of_the_player_s_goes_to_the_recent_folder(tmp_path, alarm_file):
    rec = recorder(tmp_path, folder_wait=0.3)
    run_ticks(rec, 50, poll=True)
    alarm_file.write_text(json.dumps({"reason": "lie_detector (seen by watch)", "time": "2026-10-24 21:51:00"}),
                          encoding="utf-8")
    os.utime(alarm_file, (rec.clock(), rec.clock()))
    run_ticks(rec, 20, poll=True)
    assert rec.flush(10) and not (tmp_path / "recent").exists()  # the alarm up: the cycle going
    alarm_file.unlink()  # cleared: with no recording of the player's, its cycle's end
    run_ticks(rec, 5, poll=True)
    assert rec.flush(10)
    (folder,) = (tmp_path / "recent").iterdir()
    assert folder.name.endswith("-lie-detector") and summary(folder)["player_recording"] is None
    assert summary(folder)["frames"] >= 50
    assert any("no recording of the player's began" in m for m in rec.said)
    rec.close()


def test_an_alarm_for_the_image_quiz_or_the_mini_game_s_notice_is_saved_by_itself(tmp_path, alarm_file, monkeypatch):
    """On 2026-10-25 the image quiz raised no alarm, and was kept only because the user asked for `save-recent`. The
    player's alarm for it (control.alarm.raise_alarm, its `form`) is saved as any, named for the form."""
    monkeypatch.setattr(alarm, "page", lambda title, text: None)
    rec = recorder(tmp_path, folder_wait=0.3)
    run_ticks(rec, 50, poll=True)
    alarm.raise_alarm(None, form="quiz warning")
    os.utime(alarm_file, (rec.clock(), rec.clock()))  # (the fake clock's now)
    run_ticks(rec, 20, poll=True)
    alarm_file.unlink()
    run_ticks(rec, 5, poll=True)
    assert rec.flush(10)
    (folder,) = (tmp_path / "recent").iterdir()
    assert folder.name.endswith("-lie-detector-quiz-warning")
    assert summary(folder)["alarm"]["form"] == "quiz warning" and summary(folder)["frames"] >= 50
    assert any("Lie Detector alarm (lie_detector, quiz warning, " in m for m in rec.said)
    rec.close()


def _alarm_at(rec, alarm_file, form="mini-game notice"):
    seen = rec.clock()
    alarm_file.write_text(json.dumps({"reason": "lie_detector", "time": "2026-10-25 00:55:10", "form": form}),
                          encoding="utf-8")
    os.utime(alarm_file, (seen, seen))
    return seen


def _cycle_over(player, last_seen):
    """The player's recording of the cycle summed up: the Lie Detector last seen `last_seen` s after its first sight
    (the folder's name)."""
    (player / SUMMARY).write_text(json.dumps({"last_seen": last_seen}), encoding="utf-8")


def test_the_truth_booth_of_unknown_cause_has_the_5_minutes_before_it_saved_at_once(tmp_path, alarm_file):
    """The user, 2026-11-03: "if we entered a truth booth without triggering any of the mini-game/challenge, that
    probably means we missed some kind of check, we should immediately save a 5min-clip and mark them unknown reason
    that led to truth booth for later analysis". The alarm's `cause` (the watchdog's lie_recorder.UNKNOWN_CAUSE): the 5
    minutes before it written at once, lossless, to `unknown`, named and summed up as such, the player's recording
    named; its cycle's clip as any once it's over, and no `before/` then (it's this part)."""
    rec = recorder(tmp_path)
    run_ticks(rec, 3500, poll=True)  # nearly 6 minutes
    seen = rec.clock()
    alarm_file.write_text(json.dumps({"reason": "lie_detector", "time": "2026-11-03 12:00:00", "form": "truth booth",
                                      "cause": lie_recorder.UNKNOWN_CAUSE}), encoding="utf-8")
    os.utime(alarm_file, (seen, seen))
    player = tmp_path / "lie_detector" / time.strftime(lie_recorder.STAMP, time.localtime(seen + 1))
    player.mkdir(parents=True)
    run_ticks(rec, 5, poll=True)
    assert rec.flush(60)  # written while the Truth Booth is up
    (part,) = (tmp_path / "booth_unknown").iterdir()
    assert part.name.endswith("-" + rolling.UNKNOWN_NAME)
    s = summary(part)
    assert s["kind"] == "before a Truth Booth of unknown cause" and s["cause"] == lie_recorder.UNKNOWN_CAUSE
    assert s["alarm"]["form"] == "truth booth" and s["player_recording"] == str(player) and s["format"] == "png"
    assert 2995 <= s["ticks"] <= 3001 and s["alarm_t"] == pytest.approx(300.0, abs=0.6)
    assert s["frames"] == s["ticks"] == len(list(part.glob("*.png")))
    assert any("no test seen before it" in m for m in rec.said)
    assert not (tmp_path / "alarm_clips").exists() and not (player / BEFORE).exists() and rec.pending_clips
    _cycle_over(player, 30.0)
    run_ticks(rec, 400, poll=True)  # its last sight's 5 s past
    assert rec.flush(60) and not rec.pending_clips
    (clip,) = (tmp_path / "alarm_clips").iterdir()
    assert summary(clip)["kind"] == "the alarm's clip" and not (player / BEFORE).exists()
    assert len(list((tmp_path / "booth_unknown").iterdir())) == 1
    rec.close()


def test_an_alarm_gets_a_clip_of_its_own_from_2_s_before_it_to_its_cycle_s_end(tmp_path, alarm_file):
    """The user, 2026-10-25: "For an alarmed lie detector save, save T-10s to T+1min automatically, regardless of
    [the] lie detector's own save or not. I need this as a 2nd line of defense"; since 2026-11-01 to the cycle's end,
    written then (the user: "no more 60s, save all recordings you need at the 10s mark"); since 2026-11-03 from 2 s
    before the Lie Detector's first frame ("Only include frames that are 2s before the bottom right count down"), here
    none showing it (the made-up frames): 2 s before the alarm. With no recording of the player's, the cycle ends when
    the alarm is cleared."""
    rec = recorder(tmp_path, folder_wait=0.3)
    run_ticks(rec, 400, poll=True)
    seen = _alarm_at(rec, alarm_file)
    run_ticks(rec, 300, poll=True)  # 30 s on, the alarm still up: not yet, and nor is the part before it
    assert rec.flush(10) and not (tmp_path / "alarm_clips").exists() and not (tmp_path / "recent").exists()
    alarm_file.unlink()
    run_ticks(rec, 1, poll=True)
    assert rec.flush(10)
    (clip,) = (tmp_path / "alarm_clips").iterdir()
    s = summary(clip)
    assert clip.name.endswith("-lie-detector-mini-game-notice") and s["kind"] == "the alarm's clip"
    assert s["alarm"]["form"] == "mini-game notice" and s["asked"] == [-2.0, pytest.approx(30.0, abs=0.15)]
    assert s["cut_short"] is False and s["player_recording"] is None and s["first_shown"] is None
    assert s["alarm_t"] == pytest.approx(2.0, abs=0.15) and 319 <= s["ticks"] <= 322 and s["frames"] == s["ticks"]
    assert lines(clip)[-1]["t"] == pytest.approx(32.0, abs=0.15)
    # Lossless: PNG, each frame exactly the grab it was.
    assert s["format"] == "png" and s["jpeg_quality"] is None and lines(clip)[0]["file"] == "00000.png"
    shot = cv2.imread(str(clip / "00000.png"))
    grab_n = next(k for k in range(256) if k * 3 % 256 == shot[0, -1, 0])  # (Grab's frames: a shade a grab)
    assert np.array_equal(shot, bgra(grab_n)[..., :3])
    # ... besides the part before it, its own way (here no recording of the player's began: to recent/), JPEG.
    (before,) = (tmp_path / "recent").iterdir()
    assert summary(before)["kind"] == "before the Lie Detector" and summary(before)["format"] == "jpeg"
    assert summary(before)["alarm_t"] == pytest.approx(40.0, abs=0.6)  # all the ring held: 40 s
    assert any("its own clip from 2 s before its first frame to 5 s after its test's last sight" in m for m in rec.said)
    rec.close()
    assert len(list((tmp_path / "alarm_clips").iterdir())) == 1  # written once
    assert lines(clip)[0]["at"] == rolling.clock_time(seen - 1.9)  # its first moment, 2 s before the alarm's


def test_an_alarm_s_clip_and_part_before_are_written_5_s_after_the_cycle_s_last_sight(tmp_path, alarm_file):
    """Since 2026-11-01 the clip is the Lie Detector's only video (the player's recording keeps no frames): it runs
    while the player's recording of the cycle does (its folder there, no summary yet), and it and the part before are
    written once the cycle is summed up, ending CLIP_TAIL_S after its last sight (the user: "no more 60s, save all
    recordings you need at the 10s mark"; 5 s since 2026-11-03), from the folder's name in a summary of before then
    (no `first_seen_at`, no `test_last`)."""
    rec = recorder(tmp_path)
    run_ticks(rec, 100, poll=True)
    seen = _alarm_at(rec, alarm_file)
    player = tmp_path / "lie_detector" / time.strftime(lie_recorder.STAMP, time.localtime(seen + 1))
    player.mkdir(parents=True)
    run_ticks(rec, 300, poll=True)  # 30 s on, the cycle still going: nothing written
    assert rec.flush(30) and not (tmp_path / "alarm_clips").exists() and not (player / BEFORE).exists()
    _cycle_over(player, 30.0)  # the cycle over: last seen 30 s after its first sight
    run_ticks(rec, 30, poll=True)
    assert rec.flush(30) and not (tmp_path / "alarm_clips").exists()  # (33 s on: the last sight's 5 s not past)
    run_ticks(rec, 40, poll=True)
    assert rec.flush(30) and (player / BEFORE / SUMMARY).exists()
    (clip,) = (tmp_path / "alarm_clips").iterdir()
    s = summary(clip)
    end = rolling.stamp_time(player.name) + 30.0 + rolling.CLIP_TAIL_S  # (the folder's name: its first sight)
    assert s["asked"][0] == -2.0 and s["asked"][1] == pytest.approx(end - seen, abs=1e-3) and 35 < s["asked"][1] <= 36
    assert s["player_recording"] == str(player) and s["cut_short"] is False
    assert lines(clip)[-1]["at"] == rolling.clock_time(max(it.t for it in rec.ring.items if it.t <= end))
    assert abs(s["ticks"] - 10 * (2 + s["asked"][1])) <= 2
    rec.close()


def test_an_alarm_s_clip_runs_from_2_s_before_its_first_frame_to_5_s_after_its_test_s_last_sight(tmp_path, alarm_file):
    """The user, 2026-11-03: "Only include frames that are 2s before the bottom right count down", "Only include
    frames of up to 5s once the board disappears" (the clip of 00:01 had run 10 s before the alarm to 85 s into the
    Truth Booth). The first frame showing the Lie Detector is found on the clip's frames (`shows`: here the grabs from
    the 98th on, 0.3 s before the alarm); the end from the player's recording: its first sight's time and its test's
    last sight, whatever was seen after (the Truth Booth)."""
    grab = Grab()
    shown = []

    def shows(px):
        n = next(k for k in range(256) if k * 3 % 256 == px[0, -1, 0])  # (Grab's frames: a shade a grab)
        shown.append(n)
        return n >= 98

    rec = recorder(tmp_path, grab=grab, shows=shows)
    run_ticks(rec, 100, poll=True)  # grabs 1 to 100: the alarm at the 101st's time
    seen = _alarm_at(rec, alarm_file)
    player = tmp_path / "lie_detector" / time.strftime(lie_recorder.STAMP, time.localtime(seen + 1))
    player.mkdir(parents=True)
    run_ticks(rec, 300, poll=True)
    (player / SUMMARY).write_text(json.dumps({"first_seen_at": seen + 0.4, "test_last": 20.0, "last_seen": 60.0}),
                                  encoding="utf-8")
    run_ticks(rec, 1, poll=True)
    assert rec.flush(30)
    (clip,) = (tmp_path / "alarm_clips").iterdir()
    s = summary(clip)
    first = seen - 0.3  # grab 98's time
    assert s["first_shown"] == rolling.day_time(first) and s["asked"][0] == pytest.approx(-2.3, abs=1e-3)
    assert s["asked"][1] == pytest.approx(25.4, abs=1e-3)  # 0.4 + 20 + 5
    assert lines(clip)[0]["at"] == rolling.clock_time(first - 1.9) and s["alarm_t"] == pytest.approx(2.2, abs=0.01)
    end = seen + 0.4 + 20.0 + rolling.CLIP_TAIL_S
    assert lines(clip)[-1]["at"] == rolling.clock_time(max(it.t for it in rec.ring.items if it.t <= end))
    assert max(shown) == 98 and min(shown) >= 70  # looked for from 3 s before the alarm, up to the first showing it
    assert np.array_equal(cv2.imread(str(clip / "00000.png")), bgra(79)[..., :3])  # grab 79: 1.9 s before grab 98
    rec.close()


def test_a_clip_starting_inside_a_chain_starts_with_its_frame_made_whole():
    """The perception service's frames are kept as differences, a whole frame a second (a chain): a clip that starts
    inside one starts with its frame made whole, in the chain's own order (RGB), exactly as shown."""
    frames = [np.random.default_rng(n).integers(0, 256, (48, 64, 3), np.uint8) for n in range(5)]  # RGB, as shown
    items = [rolling.Item(float(n), rolling.pack(f) if n == 0 else rolling.pack_diff(f, frames[n - 1]), size=(64, 48),
                          key=n == 0, rgb=True, seq=n) for n, f in enumerate(frames)]
    out = rolling.starting_at(items, 3)
    assert [it.seq for it in out] == [3, 4] and out[0].key and out[0].rgb and not out[1].key
    assert [np.array_equal(px, frames[k][..., ::-1]) for k, (_, px) in zip((3, 4), rolling.decode(out))] == [True] * 2
    assert rolling.starting_at(items, 0) == items and rolling.starting_at(items, 5) == []


def test_an_alarm_s_clip_ends_3_minutes_after_it_however_long_the_cycle_runs(tmp_path, alarm_file):
    rec = recorder(tmp_path)
    run_ticks(rec, 100, poll=True)
    seen = _alarm_at(rec, alarm_file)
    player = tmp_path / "lie_detector" / time.strftime(lie_recorder.STAMP, time.localtime(seen + 1))
    player.mkdir(parents=True)  # (never summed up: a cycle sent to the Truth Booth, the player stopped)
    run_ticks(rec, 1795, poll=True)
    assert rec.flush(30) and not (tmp_path / "alarm_clips").exists()
    run_ticks(rec, 10, poll=True)
    assert rec.flush(30)
    (clip,) = (tmp_path / "alarm_clips").iterdir()
    assert summary(clip)["asked"] == [-2.0, rolling.CLIP_MAX_S] and abs(summary(clip)["ticks"] - 1820) <= 2
    rec.close()


def test_an_alarm_s_clip_keeps_its_moments_however_little_the_ring_keeps(tmp_path, alarm_file):
    rec = recorder(tmp_path, folder_wait=0.3, ring=Ring(seconds=20.0))  # the ring holds 20 s; the clip needs 70
    run_ticks(rec, 300, poll=True)
    _alarm_at(rec, alarm_file, form="quiz warning")
    run_ticks(rec, 600, poll=True)
    alarm_file.unlink()
    run_ticks(rec, 1, poll=True)
    assert rec.flush(10)
    (clip,) = (tmp_path / "alarm_clips").iterdir()
    s = summary(clip)
    assert clip.name.endswith("-lie-detector-quiz-warning")
    assert 619 <= s["ticks"] <= 622 and s["alarm_t"] == pytest.approx(2.0, abs=0.15)
    rec.close()


def test_an_alarm_s_clip_is_written_cut_short_when_the_recorder_stops(tmp_path, alarm_file):
    rec = recorder(tmp_path, folder_wait=0.3)
    run_ticks(rec, 200, poll=True)
    _alarm_at(rec, alarm_file)
    run_ticks(rec, 50, poll=True)
    rec.close()
    (clip,) = (tmp_path / "alarm_clips").iterdir()
    s = summary(clip)
    assert s["cut_short"] is True and 69 <= s["ticks"] <= 71 and s["alarm_t"] == pytest.approx(2.0, abs=0.15)


def test_an_alarm_s_clips_are_kept_to_their_budget(tmp_path, alarm_file):
    root = tmp_path / "alarm_clips"
    for stamp in ("20261020-100000-lie-detector", "20261021-100000-lie-detector"):
        a_part(root / stamp, 4000)
    rec = recorder(tmp_path, folder_wait=0.3, clips_keep=6000)  # the newer old one fits beside the next, not both
    run_ticks(rec, 50, poll=True)
    _alarm_at(rec, alarm_file)
    run_ticks(rec, 20, poll=True)
    alarm_file.unlink()
    run_ticks(rec, 5, poll=True)
    assert rec.flush(10)
    names = sorted(p.name for p in root.iterdir())
    assert names[0] == "20261021-100000-lie-detector" and len(names) == 2
    assert any("deleted the 1 oldest alarm clips" in m for m in rec.said)
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
    os.utime(alarm_file, (rec.clock(), rec.clock()))
    run_ticks(rec, 20, poll=True)
    alarm_file.unlink()
    run_ticks(rec, 5, poll=True)
    assert rec.flush(5) and len(list((tmp_path / "recent").iterdir())) == 1
    rec.close()


def test_memory_is_bounded_the_oldest_frames_go_past_the_cap(tmp_path):
    ring = Ring(max_bytes=10_000)
    for k in range(100):
        ring.add(Item(k / 10, b"x" * 1000))
        ring.add(Item(k / 10 + 0.05, None, "covered"))
        assert ring.bytes <= 10_000
    assert ring.frames == 10 and ring.trimmed == 90  # younger than 5 minutes, gone for the cap
    assert ring.items[0].data is not None and ring.items[0].t == 9.0
    clock_set_back = Ring(max_items=50)
    for k in range(200):
        clock_set_back.add(Item(1000.0 - k, None, "covered"))
    assert len(clock_set_back.items) == 50


def test_disk_is_bounded_the_oldest_saves_go_and_a_lie_detector_cycle_counts_its_part_before(tmp_path):
    recent = tmp_path / "recent"
    for n, stamp in enumerate(("20261024-100000", "20261024-110000", "20261024-120000")):
        a_part(recent / stamp, *[1000] * 10, kind="save-recent")
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
    a_part(cycle / BEFORE, 5000, kind="before the Lie Detector")
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
    items = [Item(1000.0 + k / 10, rolling.pack(bgra(k)), size=(64, 48)) for k in range(30)]
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
