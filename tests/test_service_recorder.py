"""The rolling recording kept by the perception service (control.recording, from 2026-10-26): its own frames, every other
one, each kept as its difference from the one before (a whole frame every second), read back exactly; a chain
going whole from the ring and a save starting at a whole frame; a moment of its own (the recorder's tick: a grab, or a
gap and why) while the frame loop has no frame of the game, and while it gives fewer than the old recorder's 10 a
second (the floor); only frames of the game, none with another window over it (a gap: covered), none not asked about;
never waiting on the frame loop; the alarm's `before/` and clip and `save-recent` as the recorder wrote them, with each
frame's number; one recorder at a time (the mutex), the service saying it waits; and the service feeding it, its
screen's frames asked about the windows over the game. Fake windows, grabs and clocks, temp folders: nothing touches
the game or the screen."""

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gamelab.control import rolling
from gamelab.control.rolling import BEFORE, INDEX, SUMMARY, Item, Ring
from gamelab.control import recording
from gamelab.control.recording import Recording
from gamelab.perception.framecheck import Problem
from gamelab.perception.frames import Frame

from test_rolling import Clock, Game, Grab, bgra

W, H = 64, 48


def frame_n(n: int) -> np.ndarray:
    """The service's frame n (RGB): a background that stays, a block that moves and a corner that counts, so that the
    differences are mostly zeros and never all."""
    rng = np.random.default_rng(7)
    out = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    x = (3 * n) % (W - 8)
    out[10:18, x:x + 8] = (250, 20, (n * 37) % 256)
    out[0, 0] = (n % 256, (n // 256) % 256, 1)
    return out


def made(tmp_path, n=0, **kw) -> Recording:
    said = []
    kw.setdefault("clock", Clock())
    rec = Recording("MapleStory", game=kw.pop("game", Game()), grab=kw.pop("grab", Grab()),
                    mutex=f"Local\\gamelab-rolling-test-{os.getpid()}-{tmp_path.name}-{n}", log=said.append,
                    box=tmp_path / "recorder", lie_root=tmp_path / "lie_detector", recent=tmp_path / "recent",
                    clips=tmp_path / "alarm_clips", **kw)
    rec.said = said
    return rec


def feed(rec: Recording, frames: range, poll: bool = False) -> list[Item]:
    """Frames of the service (`frame_n`), 15 a second on the recorder's clock, encoded as the encoder thread does."""
    out = []
    for n in frames:
        out.append(rec.encode(rec.clock(), frame_n(n), n, 1))
        if poll:
            rec.recorder.poll()
        rec.clock.step(1 / rec.fps)
    return out


def bgr(n: int) -> np.ndarray:
    return cv2.cvtColor(frame_n(n), cv2.COLOR_RGB2BGR)


def test_frames_kept_as_differences_read_back_exactly_with_a_whole_frame_every_second(tmp_path):
    rec = made(tmp_path)
    assert rec.take()
    items = feed(rec, range(1, 47))
    assert [i for i, it in enumerate(items) if it.key] == [0, 15, 30, 45]  # (1 s at 15 a second)
    assert all(it.rgb and it.size == (W, H) and it.seq == n and it.run == 1 for n, it in zip(range(1, 47), items))
    whole, diff = len(rolling.pack_whole(frame_n(2))), len(items[1].data)
    assert diff < whole  # (mostly zeros)
    # Bit-exact: each frame as shown (BGR, as a grab's), a whole one alone, a difference with its chain.
    got = list(rolling.decode(list(rec.recorder.ring.items)))
    assert len(got) == 46 and all(np.array_equal(px, bgr(n)) for n, (_, px) in zip(range(1, 47), got))
    assert np.array_equal(items[15].pixels(), bgr(16))
    with pytest.raises(ValueError):
        items[16].pixels()
    # Saved as a clip is (PNG: lossless), each file the frame as shown, the index with each frame's number.
    summary = rolling.write_part(tmp_path / "part", list(rec.recorder.ring.items), {"kind": "x"}, rec.fps, png=True,
                                 encoders=3)
    assert summary["frames"] == 46 and summary["sources"] == {"perception service": 46} and summary["gaps"] == {}
    index = [json.loads(s) for s in (tmp_path / "part" / INDEX).read_text(encoding="utf-8").splitlines()]
    assert [r["seq"] for r in index] == list(range(1, 47)) and {r["run"] for r in index} == {1}
    assert all(np.array_equal(cv2.imread(str(tmp_path / "part" / r["file"])), bgr(r["seq"])) for r in index)
    rec.close()


def test_a_ring_of_chains_lets_a_chain_go_whole_and_a_save_starts_at_a_whole_frame():
    ring = Ring(seconds=2.0, max_items=1000)
    t, items = 1000.0, []
    for n in range(60):  # 4 s at 15 a second, a whole frame every 15th
        it = Item(t + n / 15, b"x" * 10, size=(1, 1), key=n % 15 == 0)
        assert ring.add(it)
        items.append(it)
    kept = list(ring.items)
    # The oldest chain goes once its last moment is out: 2 s or up to a chain's more, from a whole frame.
    assert kept[0].key and kept[0] is items[30] and kept[-1] is items[59] and len(kept) == 30
    assert ring.frames == 30 and ring.bytes == 300 and ring.trimmed == 0
    last = ring.last(0.5)  # 0.5 s asked: from its chain's whole frame
    assert last[0] is items[45] and last[-1] is items[59]
    assert ring.after(items[50].t)[0] is items[45]
    # A difference goes in only after the frame before it; with another moment between, it's refused.
    assert not ring.add(Item(t + 5, b"y", size=(1, 1), key=False), after=items[58])
    assert ring.add(Item(t + 5, b"y", size=(1, 1), key=False), after=items[59])
    # The cap takes whole chains, younger than 2 s or not.
    small = Ring(seconds=60.0, max_bytes=250)
    for n in range(60):
        small.add(Item(t + n / 15, b"x" * 10, size=(1, 1), key=n % 15 == 0))
    assert small.items[0].key and len(small.items) == 15 and small.trimmed == 45


def test_a_moment_of_its_own_or_a_gap_between_two_frames_starts_a_new_chain(tmp_path):
    rec = made(tmp_path)
    assert rec.take()
    feed(rec, range(1, 5))
    rec.recorder.tick()  # (a moment of the recorder's own: grabbed, BGR, whole)
    rec.clock.step(1 / rec.fps)
    after = feed(rec, range(5, 8))
    assert after[0].key and not after[1].key
    rec.recorder.game.why = "covered"
    rec.recorder.tick()  # a gap
    rec.clock.step(1 / rec.fps)
    again = feed(rec, range(8, 10))
    assert again[0].key and not again[1].key
    items = list(rec.recorder.ring.items)
    assert [("gap" if it.data is None else "own" if not it.rgb else "key" if it.key else "diff") for it in items] == [
        "key", "diff", "diff", "diff", "own", "key", "diff", "diff", "gap", "key", "diff"]
    got = [px for _, px in rolling.decode(items)]
    want = [bgr(n) for n in range(1, 5)] + [bgra(1)[..., :3]] + [bgr(n) for n in range(5, 10)]
    assert len(got) == 10 and all(np.array_equal(a, b) for a, b in zip(got, want))
    # A frame shown before the newest moment (the recorder's own came first) isn't put after it.
    assert rec.encode(items[-1].t - 0.01, frame_n(11), 11, 1) is None and rec.late == 1
    rec.close()


def _st(seq, bad=None, late=False):
    return SimpleNamespace(seq=seq, bad=bad, late=late)


def shown(n: int, t: float, over: str | None = "") -> Frame:
    """The frame loop's frame n shown at `t`, asked whether a window was over the game (ScreenFrames.cover_check)."""
    return Frame(frame_n(n), t, True, over=over)


def test_the_frame_loop_hands_over_every_other_frame_and_never_waits_dropping_past_the_queue(tmp_path):
    rec = made(tmp_path)
    assert rec.take()  # (its threads not started: nothing takes the queue)
    t0, took = time.monotonic(), []
    for k in range(60):  # 2 s at 30 a second
        began = time.perf_counter()
        rec.saw(_st(k + 1), shown(k, t0 + k / 30), 3)
        took.append(time.perf_counter() - began)
    # The bucket full at first (BURST: the first three), then every other one; 8 kept (the queue), the rest dropped.
    assert len(rec._q) == recording.QUEUE and rec.dropped == 31 - recording.QUEUE
    assert [e[2] for e in rec._q] == [1, 2, 3, 5, 7, 9, 11, 13] and max(took) < 0.01
    # Each handed over with the moment it was shown, on the recorder's clock.
    assert abs((rec._q[4][0] - rec._q[3][0]) - 2 / 30) < 0.01
    rec.close()
    # A frame loop slower than 15 a second (a crowd), its frames' times jittering: every one taken; and after a stall,
    # no catching up (two, the bucket's BURST, then every other at 30 again).
    slow = made(tmp_path, 3)
    assert slow.take()
    times = [k / 14.6 + (0.006 if k % 2 else -0.006) for k in range(20)]
    times += [times[-1] + 1.0 + k / 30 for k in range(8)]
    for k, t in enumerate(times):
        slow.saw(_st(k + 1), shown(k, t0 + t), 1)
    taken = [e[2] for e in slow._q]
    assert len(taken) + slow.dropped == 20 + 5 and taken[:8] == list(range(1, 9))
    slow.close()
    # Irregular, 13.7 a second on the whole (a crowd's: 40 and 106 ms in turn): every one taken.
    odd = made(tmp_path, 4)
    assert odd.take()
    t = t0
    for k in range(30):
        t += 0.040 if k % 2 else 0.106
        odd.saw(_st(k + 1), shown(k, t), 1)
    assert len(odd._q) + odd.dropped == 30
    odd.close()
    # At the service's own rate (localization.rate.HZ, 15 a second since 2026-10-30: recording.FPS), its frames' times
    # jittering a little: every one kept.
    from gamelab.localization import rate
    assert recording.FPS == rate.HZ
    every = made(tmp_path, 5)
    assert every.take()
    for k in range(30):
        every.saw(_st(k + 1), shown(k, t0 + k / rate.HZ + (0.004 if k % 2 else -0.004)), 1)
    assert len(every._q) + every.dropped == 30
    every.close()


def test_only_frames_of_the_game_are_kept(tmp_path):
    rec = made(tmp_path)
    assert rec.take()
    t0 = time.monotonic()
    one = shown(1, t0)
    rec.saw(_st(1), Frame.unfit(t0, Problem("window", "the game isn't in front")), 1)  # (not visible: no picture)
    rec.saw(_st(2, bad=Problem("offset", "the capture doesn't show the game where its window is")), one, 1)
    rec.saw(_st(5), shown(1, t0, over=None), 1)  # (not asked whether a window was over it: never kept)
    assert not rec._q and rec._offered < 0
    rec.saw(_st(4, bad=Problem("hud", "the HUD's border isn't on row 615")), one, 1)  # the game's own picture
    assert [e[2] for e in rec._q] == [4]
    rec.close()


def test_a_frame_with_another_window_over_the_game_is_a_gap_and_breaks_the_chain(tmp_path):
    """window.over's catch, which the grid's points miss (a notification between them, a window clicks pass through):
    nothing of the frame kept, nor held, a gap ("covered") instead, as the recorder's own check makes it, said with
    the window in the log; the next frame whole."""
    rec = made(tmp_path)
    assert rec.take()
    t0, items = time.monotonic(), []
    for k in range(6):
        over = "a window of class 'Toast' at 900,600 300x80" if k in (2, 3) else ""
        rec.saw(_st(k + 1), shown(k, t0 + k / 15, over=over), 1)
        while rec._q:
            items.append(rec.encode(*rec._q.popleft()))
    assert [it.why if it.data is None else ("key" if it.key else "diff") for it in items] == [
        "key", "diff", recording.COVERED, recording.COVERED, "key", "diff"]
    assert rec.covered == 2 and all(it.data is None for it in items[2:4])
    assert any("not recording: covered (by a window of class 'Toast'" in m for m in rec.said)
    got = [px for _, px in rolling.decode(list(rec.recorder.ring.items))]
    assert len(got) == 4 and all(np.array_equal(a, bgr(n)) for a, n in zip(got, (0, 1, 4, 5)))
    rec.close()


def test_with_the_frame_loop_under_10_a_second_the_recording_grabs_its_own_until_it_comes_back(tmp_path, monkeypatch):
    """The floor: the old recorder's 10 a second, kept whatever the frame loop does (a crowd without the native pieces:
    7-10 a second live). Its own moments meanwhile (the recorder's tick: a grab, looked at before and after), the
    loop's frames only counted; back to the loop's once it gives 12 a second for 2 s."""
    rec = made(tmp_path)
    assert rec.take()
    clock = {"now": time.monotonic(), "t": 0.0}
    monkeypatch.setattr(recording, "time", SimpleNamespace(  # (the module's clock alone: the test's)
        monotonic=lambda: clock["t"], perf_counter=time.perf_counter, thread_time=time.thread_time,
        strftime=time.strftime, time=time.time))
    seq = [0]

    def loop(rate: float, seconds: float):
        """The frame loop at `rate` and the ticks at 15 a second, on the test's own clock."""
        start = clock["now"]
        events = [(start + k / rate, "f") for k in range(int(round(seconds * rate)))]
        events += [(start + (k + 0.5) / 15, "t") for k in range(int(round(seconds * 15)))]
        for t, kind in sorted(events):
            clock["t"] = t
            if kind == "f":
                seq[0] += 1
                rec.saw(_st(seq[0]), shown(seq[0], t), 1)
                rec._q.clear()  # (the encoder's: not looked at here)
            else:
                rec.tick()
        clock["now"] = start + seconds

    loop(13.0, 2.0)  # over the floor: none of its own
    assert not rec.floor and rec.floor_own == 0 and rec.recorder.grab.n == 0
    loop(8.0, 1.5)  # under it: its own grabs, 15 a second (the loop's frames only counted)
    assert rec.floor and rec.floor_times == 1 and rec.recorder.grab.n >= 7
    queued = rec.taken + rec.dropped
    loop(11.0, 2.0)  # 11 a second: still under the way back (12)
    assert rec.floor and rec.recorder.grab.n >= 7 + 29 and rec.taken + rec.dropped == queued
    loop(14.0, 3.0)  # back over it: the loop's frames again
    assert not rec.floor and rec.floor_times == 1
    grabs = rec.recorder.grab.n
    loop(14.0, 1.0)
    assert rec.recorder.grab.n == grabs and rec.floor_own == grabs
    counts = rec.counts()
    assert counts["floor_moments"] == grabs and counts["floor_times"] == 1 and not counts["floor"]
    assert any("the frame loop gave" in m and "grabs its own" in m for m in rec.said)
    monkeypatch.undo()
    rec.close()


def test_while_the_frame_loop_has_no_frame_the_recording_takes_its_moments_as_the_recorder_did(tmp_path):
    rec = made(tmp_path)
    assert rec.take()
    rec.tick()  # nothing handed over yet: a grab of its own (the game shows)
    rec.recorder.game.why = "minimized"
    rec.tick()
    assert [(it.why, it.rgb) for it in rec.recorder.ring.items] == [("", False), ("minimized", False)]
    assert rec.own == 2 and rec.recorder.grab.n == 1
    rec.recorder.game.why = ""
    rec.saw(_st(1), shown(1, time.monotonic()), 1)  # a frame from the frame loop: none of its own
    rec.tick()
    assert rec.own == 2 and rec.recorder.grab.n == 1
    rec._offered -= recording.OWN_AFTER_S + 0.1  # ... until none has come for OWN_AFTER_S
    rec.tick()
    assert rec.own == 3 and rec.recorder.grab.n == 2
    rec.close()


def test_an_alarm_saves_the_minutes_before_it_into_before_and_its_own_clip(tmp_path, alarm_file):
    rec = made(tmp_path, folder_wait=0.3)
    assert rec.take()
    feed(rec, range(1, 601), poll=True)  # 40 s
    seen = rec.clock()
    alarm_file.write_text(json.dumps({"reason": "lie_detector", "time": "2026-10-26 01:01:45", "form": "countdown"}),
                          encoding="utf-8")
    os.utime(alarm_file, (seen, seen))
    feed(rec, range(601, 601 + 920), poll=True)  # 61.3 s on: the clip's last moment past
    assert rec.recorder.flush(60)
    assert rec.recorder.apart and not any("written apart failed" in m for m in rec.said)  # (in a process of its own)
    (clip,) = (tmp_path / "alarm_clips").iterdir()
    s = json.loads((clip / SUMMARY).read_text(encoding="utf-8"))
    assert s["kind"] == "the alarm's clip" and s["format"] == "png" and s["fps"] == 15.0
    assert s["frames"] == s["ticks"] and 1045 <= s["frames"] <= 1070  # 70 s at 15 a second, from a whole frame
    assert s["alarm_t"] == pytest.approx(10.0, abs=1.1) and s["sources"] == {"perception service": s["frames"]}
    index = [json.loads(x) for x in (clip / INDEX).read_text(encoding="utf-8").splitlines()]
    for r in index[:3] + index[-3:]:  # lossless, each the frame its number says
        assert np.array_equal(cv2.imread(str(clip / r["file"])), bgr(r["seq"]))
    (before,) = (tmp_path / "recent").iterdir()  # (no recording of the player's began: in recent/)
    b = json.loads((before / SUMMARY).read_text(encoding="utf-8"))
    assert b["kind"] == "before the Lie Detector" and b["format"] == "jpeg" and b["frames"] == b["ticks"]
    assert b["alarm_t"] == pytest.approx(40.0, abs=0.2) and b["frames"] == pytest.approx(675, abs=3)  # 40 s + 5 s
    rec.close()


@pytest.fixture
def alarm_file(tmp_path, monkeypatch):
    from gamelab.control import alarm
    path = tmp_path / "ALARM.json"
    monkeypatch.setattr(alarm, "ALARM", path)
    return path


def test_save_recent_writes_the_services_frames_as_the_recorder_wrote_its_own(tmp_path):
    rec = made(tmp_path, clock=Clock(time.time() - 30))
    assert rec.take()
    feed(rec, range(1, 151))  # 10 s
    rec.recorder.game.why = "not in front"
    for _ in range(15):  # 1 s of the recorder's own moments: gaps
        rec.recorder.tick()
        rec.clock.step(1 / rec.fps)
    feed(rec, range(151, 181))  # 2 s more
    rec.recorder.write_status()
    status = rolling.read_status(tmp_path / "recorder")
    assert status["source"] == recording.SOURCE and status["fps_asked"] == 15.0 and status["taken"] == 180
    got = {}
    asking = threading.Thread(target=lambda: got.update(rolling.ask(5, "check", box=tmp_path / "recorder")))
    asking.start()
    while asking.is_alive():
        rec.recorder.poll()
        time.sleep(0.01)
    assert "folder" in got, got
    assert not any("written apart failed" in m for m in rec.said)  # (in a process of its own: write_part_apart)
    folder = Path(got["folder"])
    assert got["gaps"] == {"not in front": 1.0} and got["window"] == [W, H]
    assert 2 * 15 + 30 <= got["frames"] <= 2 * 15 + 45  # 5 s asked: from the whole frame before them
    index = [json.loads(x) for x in (folder / INDEX).read_text(encoding="utf-8").splitlines()]
    assert index[-1]["seq"] == 180 and [r["gap"] for r in index if "gap" in r] == ["not in front"]
    back = cv2.imread(str(folder / index[-1]["file"]))  # (JPEG, as the recorder's: the frame as shown, encoded)
    assert np.array_equal(back, cv2.imdecode(np.frombuffer(rolling.encode(bgr(180)), np.uint8), cv2.IMREAD_COLOR))
    lines = rolling.status_lines(tmp_path / "recorder")
    assert any(recording.SOURCE in line and "15 a second" in line for line in lines)
    rec.close()


def test_one_recorder_at_a_time_the_service_waits_for_the_recorders_mutex_and_says_so(tmp_path):
    rec = made(tmp_path, 1)
    held = rolling.only_one(rec.mutex)  # `gamelab recorder` running
    assert held is not None
    assert not rec.take() and rec.recorder is None
    waiting = json.loads((tmp_path / "recorder" / recording.WAITING).read_text(encoding="utf-8"))
    assert waiting["pid"] == os.getpid() and any("`gamelab recorder` is running" in m for m in rec.said)
    rec.saw(_st(1), shown(1, time.monotonic()), 1)  # not recording: nothing taken, nothing waits
    assert not rec._q and "not recording" in rec.line()
    rolling.let_go(held)  # (the babysitter stopped it)
    assert rec.take() and rec.recorder is not None
    assert not (tmp_path / "recorder" / recording.WAITING).exists()
    assert rolling.only_one(rec.mutex) is None  # held by the service now: `gamelab recorder` says one runs
    rec.close()
    again = rolling.only_one(rec.mutex)
    assert again is not None
    rolling.let_go(again)
    assert json.loads((tmp_path / "recorder" / "status.json").read_text(encoding="utf-8"))["stopped"]


def test_the_service_hands_its_frames_to_the_recording_with_their_numbers(tmp_path):
    """The service on drawn frames at 30 a second, its recording's threads running: 15 frames a second kept, each with
    the number of the state the frame loop made of it, and the status file saying where they come from."""
    from gamelab.localization.service import Service
    from gamelab.screen import STRETCHED

    from test_feed import drawn_runner

    rec = made(tmp_path, 2, clock=time.time)
    address = rf"\\.\pipe\gamelab-perceive-test-rec-{os.getpid()}-{time.monotonic_ns()}"
    runner = drawn_runner()
    runner.frames.cover_check = False  # (as ScreenFrames has it: the service turns it on for its recording)
    svc = Service(lambda map_id, geo: runner, 1, lambda: (STRETCHED, None), address=address,
                  log=lambda m: None, recording=rec)
    svc.serve(until=time.monotonic() + 3.0)  # (its close closes the recording: the status says it stopped)
    assert runner.frames.cover_check is True
    items = list(rec.recorder.ring.items)
    frames = [it for it in items if it.data is not None and it.rgb]
    assert 25 <= len(frames) <= 46 and rec.dropped == 0  # about 15 a second once the run is made
    seqs, ts = [it.seq for it in frames], np.array([it.t for it in frames])
    assert seqs == sorted(set(seqs)) and np.diff(ts).mean() >= 1 / 15 - 0.005  # 15 a second at most, in order
    assert {it.run for it in frames} == {1}
    status = json.loads((tmp_path / "recorder" / "status.json").read_text(encoding="utf-8"))
    assert status["stopped"] and status["source"] == recording.SOURCE and status["pid"] == os.getpid()


def test_a_part_written_apart_comes_out_as_written_here_and_if_its_process_fails_it_is_written_here(tmp_path,
                                                                                                    monkeypatch):
    """The perception service's saves go to a process of their own (rolling.write_part_apart: no Python lock shared
    with the frame loop): the same files as write_part makes. With no process to be had (its python missing), the part
    is written here all the same, said in the log."""
    rec = made(tmp_path)
    assert rec.take()
    items = feed(rec, range(1, 40)) + [Item(rec.clock(), None, "not in front")]
    here = rolling.write_part(tmp_path / "here", items, {"kind": "x"}, rec.fps, png=True, encoders=2)
    said = []
    apart = rolling.write_part_apart(tmp_path / "apart", items, {"kind": "x"}, rec.fps, png=True, encoders=2,
                                     log=said.append)
    assert apart == here and not said
    for name in sorted(p.name for p in (tmp_path / "here").iterdir()):
        assert (tmp_path / "apart" / name).read_bytes() == (tmp_path / "here" / name).read_bytes()
    monkeypatch.setattr(rolling.sys, "executable", str(tmp_path / "no-python.exe"))
    again = rolling.write_part_apart(tmp_path / "again", items, {"kind": "x"}, rec.fps, png=True, encoders=2,
                                     log=said.append)
    assert again == here and len(said) == 1 and "written apart failed" in said[0]
    assert (tmp_path / "again" / rolling.SUMMARY).exists()
    rec.close()
