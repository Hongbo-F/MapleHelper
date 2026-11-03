"""The Lie Detector mini-game's target followed live, on the perception service's own frames (the user, 2026-11-02:
"Let's wire this and remove the old yolo-based live detection", and "I thought the perception process is already
screenshotting at 15fps"). It replaces the player's YOLO shadow (perception.lie_detector_yolo, removed that day), which
found the board's printed patterns 4 times a second and recorded them: this follows the target itself
(perception.lie_target) through the round, moves the cursor to it and records where it is.

- While the Lie Detector's alarm is up (its file, paths.ALARM, looked at once in ALARM_POLL_S at most), the newest frame
  the service has taken of the game is kept (`saw`, from Service._emit: its picture by reference, its number and run;
  the frame loop never waits). A thread of the service's (`run`) looks at it for the mini-game's board every
  BOARD_CHECK_S (perception.lie_detector's `over_play`, 5 ms): from the board's first sight the round is on, until the
  board has been gone for BOARD_GONE_S, the alarm has gone, or ROUND_MAX_S.
- The tracker runs in a process of its own (`python -m gamelab.control.lie_tracking`, below normal priority, OpenCV on
  CHILD_THREADS threads), started when the alarm goes up and stopped CHILD_IDLE_S after it has gone: in the service,
  whose OpenCV runs on one thread (localization.service.settle), a frame took 78 ms against 52-64 on 6-12, over the 66
  of 15 a second, and it would share the frame loop's Python lock (control.rolling.write_part_apart's measure: the
  loop's steps waited 18-21 ms with a save written there). Each round's board pictures go down its stdin, one when it
  is ready for the next: behind, it takes the newest (the frames between counted `skipped`), as the 10 fps clip showed
  it can (the tracker goes by each frame's own time).
- Each answer is the target's centre in three coordinates: `board` (px on the 750 x 500 board), `client` (px in the
  game's own 1366 x 768 client area: `board` + its corner there, lie_target.BOARD_AT_NATIVE) and `desktop` (px on the
  screen: `client` + the client area's corner on the screen, window.client_rect, looked at once in ORIGIN_POLL_S at
  most; None without the window). A cursor thread takes the newest desktop answer and hands its client coordinate to
  the elevated player (client.move_lie_cursor), whose window.move_cursor generates the trail: the administrator game
  blocks the ordinary service's input. Only the cursor moves inside the board, never a button or key; the round's end
  or the service's close cancels its plan.
- Each round is a file, paths.LIE_TARGETS/<YYYYmmdd-HHMMSS>.jsonl (the board's first sight), written by that process: a
  first line saying what it is, then a line an answer (`t`: monotonic s, `at`: the clock, `seq` and `run`: the
  service's frame number and run, which the alarm clip's index has too, so the clip's frames join; `board`, `client`,
  `desktop`; `angle`: rad, `score`, `white`: the white blob's own centre, `ms`: the step's time), none before the target
  has been seen white; the last line the round's `end`: why, frames stepped, answers, skipped, the step's median and
  worst ms. The same answers go into the service's telemetry (its persisted log, lab/telemetry/<time>-perceive.jsonl):
  a `lie_target` record each (the round's name added), `lie_target_cursor` records at queue/move/failure, and a
  `lie_target_round` record at its begin and end.
"""

import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .. import paths, telemetry
from ..localization import rate

ALARM_POLL_S = 0.5
BOARD_CHECK_S = 0.25   # how often the board is looked for (its window at its place: over_play)
BOARD_GONE_S = 3.0     # the board not seen this long: the round is over
ROUND_MAX_S = 60.0     # a round this long is over whatever (the longest seen: 10.4 s from START)
CHILD_IDLE_S = 60.0    # the tracker's process stopped this long after the alarm has gone
CHILD_THREADS = 6      # OpenCV's threads in it, and the tracker's (lie_target._pool): 64 ms a frame on 6, 71 on 4, 52
#                        on 12 (the machine busy, before the tracker's own threads: 43 -> 30 ms on the clips)
ORIGIN_POLL_S = 1.0    # how often the client area's corner on the screen is looked at again
FPS = rate.HZ          # the frames' nominal rate: the service's (the tracker's look-back: lie_target.BACK_S)
NO_WINDOW, BELOW_NORMAL = 0x08000000, 0x00004000
NAME = "%Y%m%d-%H%M%S"


class LieTracking:
    """The service's side (see the top): `saw` from the frame loop, `start` and `close` with the service, `line` for its
    status. `child` (for tests) stands in for starting the tracker's process: a callable returning an object with
    `send(msg)`, `replies()` (an iterator of its replies, read on a thread of their own) and `stop()`. `mover` stands
    in for window.move_cursor, with its same (x, y, wait=...) call."""

    def __init__(self, folder: Path | None = None, alarm: Path | None = None, spotter=None, log=None, child=None,
                 fps: float = FPS, title: str | None = None, origin=None, mover=None):
        self.folder = Path(folder) if folder is not None else paths.LIE_TARGETS
        self.title = title                  # the game window's (for the client area's corner on the screen)
        self.find_origin = origin or self._client_corner  # () -> (left, top) on the screen, or None
        self.mover = mover or self._move_cursor
        self._origin: tuple | None = None
        self._origin_at = -1e9
        self._round_name: str | None = None  # the round the tracker's answers are of (for the telemetry)
        self.alarm = alarm
        self.spotter = spotter
        self.log = log or (lambda msg: None)
        self.fps = fps
        self.make_child = child or (lambda: Child(self.folder, fps))
        self._cond = threading.Condition()
        self._newest: tuple | None = None   # (t, full, seq, run, clock): the newest frame kept, not taken yet
        self._closing = False
        self._thread: threading.Thread | None = None
        self._cursor_cond = threading.Condition()
        self._cursor_target: tuple[str, int, int] | None = None  # the newest answer not yet taken by the cursor thread
        self._cursor_round: str | None = None   # its trail may run only while this round is on
        self._cursor_thread: threading.Thread | None = None
        self.cursor_why_off = ""             # a mover failure stops cursor movement, not tracking or its records
        self._polled: float | None = None
        self._up = False
        self.child = None
        self.ready = threading.Event()      # the child is ready for a frame
        self.round: dict | None = None      # the round in progress: its name, start, last board sight, counts
        self.rounds = 0
        self.why_off = ""                   # what turned it off (its board look raised, say)
        self._gone_at: float | None = None  # since when the alarm has been down, the child up
        self._checked = -1e9

    # --- the frame loop's side ------------------------------------------------------------------------------------

    def saw(self, st, frame, run) -> None:
        """The frame loop's state of a frame and the frame (Service._emit): kept as the newest while the alarm is up or
        a round is on. Never waits."""
        if self.why_off or frame is None or getattr(frame, "full", None) is None or not getattr(frame, "visible", True):
            return
        if self.round is None and not self.alarm_up(time.monotonic()):
            return
        item = (frame.t, frame.full, getattr(st, "seq", None), run, time.time() - (time.monotonic() - frame.t))
        with self._cond:
            if self._newest is not None and self.round is not None:
                self.round["skipped"] += 1
            self._newest = item
            self._cond.notify()

    def alarm_up(self, now: float) -> bool:
        """Whether the alarm's file is there, looked at again ALARM_POLL_S after the last look (one that can't be looked
        at counts as there, as control.player.alarm_up's)."""
        if self._polled is None or not 0.0 <= now - self._polled < ALARM_POLL_S:
            self._polled = now
            try:
                os.stat(paths.ALARM if self.alarm is None else self.alarm)
                self._up = True
            except FileNotFoundError:
                self._up = False
            except OSError:
                self._up = True
        return self._up

    # --- its thread -------------------------------------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="lie-target", daemon=True)
        self._thread.start()

    def close(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        with self._cursor_cond:
            self._cursor_round = self._cursor_target = None
            self._cursor_cond.notify_all()
        if self._thread is not None:
            self._thread.join(max(0.0, deadline - time.monotonic()))
        if self._cursor_thread is not None:
            self._cursor_thread.join(max(0.0, deadline - time.monotonic()))

    def line(self) -> str:
        """For the service's status line: the round in progress, if any."""
        r = self.round
        if r is None:
            return ""
        return (f"; the Lie Detector's target followed for {time.monotonic() - r['started']:.0f} s: "
                f"{r['sent']} frames, {r['skipped']} skipped")

    def run(self) -> None:
        try:
            while True:
                with self._cond:
                    if self._newest is None and not self._closing:
                        self._cond.wait(BOARD_CHECK_S)
                    if self._closing:
                        break
                    item, self._newest = self._newest, None
                self.step(time.monotonic(), item)
        except Exception as e:  # noqa: BLE001 (the service goes on: only this is off)
            self.why_off = f"{type(e).__name__}: {e}"
            self._say(f"the Lie Detector's target tracking is off: {self.why_off}")
        if self.round is not None:
            self._end("stopped")
        self._stop_child()

    def step(self, now: float, item: tuple | None) -> None:
        """One turn of its thread at `now`: the child started or stopped with the alarm, the board looked for, the round
        begun or ended, the newest frame (`item`, or None) sent if the child is ready for it."""
        up = self.alarm_up(now)
        if up:
            self._gone_at = None
            if self.child is None:
                self._start_child()
        elif self.child is not None and self.round is None:
            self._gone_at = now if self._gone_at is None else self._gone_at
            if now - self._gone_at >= CHILD_IDLE_S:
                self._stop_child()
        if item is None:
            if self.round is not None and (now - self.round["board_at"] > BOARD_GONE_S or not up):
                self._end("board gone" if up else "alarm gone")
            return
        t, full, seq, run, clock = item
        if full.shape[:2] != (768, 1366):  # (the board's place is known at the game's own size only)
            return
        if now - self._checked >= BOARD_CHECK_S:
            self._checked = now
            board = self._board_up(full)
            if board and self.round is None and up:
                self._begin(now)
            if board and self.round is not None:
                self.round["board_at"] = now
        r = self.round
        if r is None:
            return
        if now - r["board_at"] > BOARD_GONE_S or now - r["started"] > ROUND_MAX_S or not up:
            self._end("board gone" if now - r["board_at"] > BOARD_GONE_S else "too long" if up else "alarm gone")
            return
        if self.child is None or not self.ready.is_set():
            r["skipped"] += 1
            return
        from ..perception.lie_target import BOARD_AT_NATIVE
        x, y, w, h = BOARD_AT_NATIVE
        self.ready.clear()
        if self._send(("frame", t, seq, run, round(clock, 3), np.ascontiguousarray(full[y:y + h, x:x + w]),
                       self.origin(now))):
            r["sent"] += 1

    def origin(self, now: float) -> tuple | None:
        """The game's client area's corner on the screen, looked at again ORIGIN_POLL_S after the last look."""
        if not 0.0 <= now - self._origin_at < ORIGIN_POLL_S:
            self._origin_at = now
            try:
                self._origin = self.find_origin()
            except Exception:  # noqa: BLE001 (no window: no desktop place)
                self._origin = None
        return self._origin

    def _client_corner(self) -> tuple | None:
        if not self.title:
            return None
        from .. import window
        area = window.client_rect(window.find_game(self.title))
        return area["left"], area["top"]

    def _board_up(self, full: np.ndarray) -> bool:
        if self.spotter is None:
            from ..perception.lie_detector import LieDetectorSpotter
            self.spotter = LieDetectorSpotter()
        return self.spotter.over_play(full) == "mini-game"

    def _begin(self, now: float) -> None:
        if self.child is None:
            self._start_child()
        name = time.strftime(NAME)
        self.round = {"name": name, "started": now, "board_at": now, "sent": 0, "skipped": 0}
        self.rounds += 1
        self._round_name = name
        with self._cursor_cond:
            self._cursor_round = name
            self._cursor_target = None
        self._send(("begin", name, {"fps": self.fps}))
        telemetry.emit("lie_target_round", event="begin", round=name, file=str(self.folder / f"{name}.jsonl"))
        self._say(f"the Lie Detector's board seen: its target followed, written to {self.folder / (name + '.jsonl')}")

    def _end(self, why: str) -> None:
        r, self.round = self.round, None
        with self._cursor_cond:
            self._cursor_round = self._cursor_target = None
            self._cursor_cond.notify_all()
        self._send(("end", why, r["skipped"]))
        self._say(f"the Lie Detector's target followed {time.monotonic() - r['started']:.0f} s ({why}): {r['sent']} "
                  f"frames, {r['skipped']} skipped")

    def _send(self, msg: tuple) -> bool:
        """A message to the tracker's process; False (and the process given up: started again at the next turn with the
        alarm up) if there's none or its pipe broke."""
        child = self.child
        if child is None:
            return False
        try:
            child.send(msg)
            return True
        except Exception as e:  # noqa: BLE001
            self._say(f"the Lie Detector's target tracker couldn't be reached ({type(e).__name__}: {e}): started again")
            self._stop_child()
            return False

    def _start_child(self) -> None:
        self.ready.clear()
        self.child = self.make_child()
        threading.Thread(target=self._replies, args=(self.child,), name="lie-target-replies", daemon=True).start()

    def _replies(self, child) -> None:
        try:
            for msg in child.replies():
                if msg[0] == "ready":
                    if len(msg) > 1 and msg[1] is not None:
                        answer = msg[1]
                        telemetry.emit("lie_target", round=self._round_name, **answer)
                        self._queue_cursor(answer.get("desktop"), seq=answer.get("seq"))
                    self.ready.set()
                elif msg[0] == "ended":
                    telemetry.emit("lie_target_round", event="end", round=self._round_name, **msg[1])
                elif msg[0] == "error":
                    self._say(f"the Lie Detector's target tracker failed: {msg[1]}")
        except Exception as e:  # noqa: BLE001 (its pipe broke: started again at the next alarm)
            self._say(f"the Lie Detector's target tracker's replies stopped: {type(e).__name__}: {e}")
        if self.child is child:
            self.child = None
            self.ready.clear()

    def _move_cursor(self, x: int, y: int, wait):
        """Move through the elevated player, whose window.move_cursor generates the trail. The ordinary perception
        process's SendInput is blocked by the administrator game; its old direct call also hid that failure."""
        from . import client
        corner = self.find_origin()
        if corner is None:
            raise RuntimeError("the game window's desktop position isn't known")
        return client.move_lie_cursor(round(x - corner[0]), round(y - corner[1]),
                                      active=lambda: wait(time.perf_counter()) is True)

    def _queue_cursor(self, desktop, seq=None) -> None:
        """Give the cursor thread only the newest unanswered target; never make the tracker's reply thread wait for a trail."""
        if desktop is None or self.cursor_why_off:
            return
        with self._cursor_cond:
            if self._cursor_round is None or self._closing:
                return
            self._cursor_target = (self._cursor_round, round(desktop[0]), round(desktop[1]), seq)
            if self._cursor_thread is None:
                self._cursor_thread = threading.Thread(target=self._cursor_run, name="lie-target-cursor", daemon=True)
                self._cursor_thread.start()
                telemetry.emit("lie_target_cursor", event="thread started", round=self._cursor_round)
            self._cursor_cond.notify_all()
        telemetry.emit("lie_target_cursor", event="queued", round=self._cursor_round,
                       x=round(desktop[0]), y=round(desktop[1]), seq=seq)

    def _cursor_run(self) -> None:
        while True:
            with self._cursor_cond:
                while self._cursor_target is None and not self._closing:
                    self._cursor_cond.wait()
                if self._closing:
                    return
                name, x, y, seq = self._cursor_target
                self._cursor_target = None
            if self._cursor_round != name:  # its round ended before this target was taken
                continue
            began = time.perf_counter()
            telemetry.emit("lie_target_cursor", event="move started", round=name, x=x, y=y, seq=seq)
            try:
                result = self.mover(x, y, wait=lambda due, name=name: self._cursor_wait(name, due))
            except Exception as e:  # noqa: BLE001 (tracking and its records carry on; only cursor movement stops)
                self.cursor_why_off = f"{type(e).__name__}: {e}"
                self._say(f"the Lie Detector's cursor movement is off: {self.cursor_why_off}")
                telemetry.emit("lie_target_cursor", event="failed", round=name, x=x, y=y, seq=seq,
                               error=self.cursor_why_off)
                with self._cursor_cond:
                    self._cursor_target = None
            else:
                telemetry.emit("lie_target_cursor", event="move finished", round=name, x=x, y=y, seq=seq,
                               result=result if isinstance(result, (bool, str)) else str(result),
                               ms=round(1000 * (time.perf_counter() - began), 1))

    def _cursor_wait(self, name: str, due: float) -> bool:
        """Wait for a trail sample, ending the trail early when its round ends or the service closes."""
        with self._cursor_cond:
            while not self._closing and self._cursor_round == name:
                left = due - time.perf_counter()
                if left <= 0:
                    return True
                self._cursor_cond.wait(left)
            return False

    def _stop_child(self) -> None:
        child, self.child = self.child, None
        self.ready.clear()
        if child is not None:
            try:
                child.stop()
            except Exception:  # noqa: BLE001
                pass

    def _say(self, msg: str) -> None:
        try:
            self.log(msg)
        except Exception:  # noqa: BLE001 (a log that can't be written stops nothing)
            pass


class Child:
    """The tracker's process (`python -m gamelab.control.lie_tracking`): messages pickled down its stdin, replies up its
    stdout."""

    def __init__(self, folder: Path, fps: float = FPS):
        flags = NO_WINDOW | BELOW_NORMAL if sys.platform == "win32" else 0
        self.proc = subprocess.Popen([sys.executable, "-m", "gamelab.control.lie_tracking", str(folder), str(fps)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     cwd=str(paths.REPO), creationflags=flags)
        self._lock = threading.Lock()

    def send(self, msg: tuple) -> None:
        with self._lock:
            pickle.dump(msg, self.proc.stdin, protocol=5)
            self.proc.stdin.flush()

    def replies(self):
        while True:
            try:
                yield pickle.load(self.proc.stdout)
            except EOFError:
                return

    def stop(self) -> None:
        try:
            self.send(("quit",))
            self.proc.stdin.close()
            self.proc.wait(5)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def serve(inp, out, folder: Path, fps: float = FPS) -> None:
    """The tracker's process: its loop over the messages on `inp` (pickled: ("begin", name, info), ("frame", t, seq,
    run, clock, board RGB, the client area's corner on the screen or None), ("end", why, skipped), ("quit",)), each
    round's file written in `folder`; on `out` ("ready",) at the start, ("ready", the answer's line or None) after each
    frame and ("ended", the round's end line) at its end."""
    import cv2

    from ..perception import lie_target
    bx, by = lie_target.BOARD_AT_NATIVE[:2]
    reply = lambda *msg: (pickle.dump(msg, out, protocol=5), out.flush())  # noqa: E731
    tracker = fh = None
    counts: dict = {}
    reply("ready")
    while True:
        try:
            msg = pickle.load(inp)
        except EOFError:
            break
        kind = msg[0]
        try:
            if kind == "begin":
                _, name, info = msg
                folder.mkdir(parents=True, exist_ok=True)
                fh = open(folder / f"{name}.jsonl", "a", encoding="utf-8", buffering=1)
                fh.write(json.dumps({"kind": "the Lie Detector's target (perception.lie_target)", "round": name,
                                     "board": list(lie_target.BOARD_AT_NATIVE), **info}) + "\n")
                tracker = lie_target.TargetTracker(fps=info.get("fps", fps))
                counts = {"frames": 0, "answers": 0, "ms": []}
            elif kind == "frame":
                _, t, seq, run, clock, board, corner = msg
                line = None
                if tracker is not None:
                    began = time.perf_counter()
                    a = tracker.step(cv2.cvtColor(board, cv2.COLOR_RGB2BGR), t)
                    ms = 1000 * (time.perf_counter() - began)
                    counts["frames"] += 1
                    counts["ms"].append(ms)
                    if a is not None:
                        counts["answers"] += 1
                        cx, cy = round(a["x"] + bx, 1), round(a["y"] + by, 1)
                        line = {"t": round(t, 4), "at": time.strftime("%H:%M:%S", time.localtime(clock))
                                + f".{int(clock * 1000) % 1000:03d}", "seq": seq, "run": run,
                                "board": [round(a["x"], 1), round(a["y"], 1)], "client": [cx, cy],
                                "desktop": None if corner is None else [round(corner[0] + cx, 1),
                                                                         round(corner[1] + cy, 1)],
                                "angle": round(a["angle"], 4), "score": round(a["score"], 3), "white": a["white"],
                                "ms": round(ms, 1)}
                        fh.write(json.dumps(line) + "\n")
                reply("ready", line)
            elif kind == "end":
                _, why, skipped = msg
                if fh is not None:
                    ms = counts.get("ms") or [0.0]
                    end = {"end": why, "frames": counts.get("frames", 0), "answers": counts.get("answers", 0),
                           "skipped": skipped, "ms_median": round(float(np.median(ms)), 1),
                           "ms_max": round(float(max(ms)), 1)}
                    fh.write(json.dumps(end) + "\n")
                    fh.close()
                    reply("ended", end)
                tracker = fh = None
            elif kind == "quit":
                break
        except Exception as e:  # noqa: BLE001 (said, and the next message taken)
            reply("error", f"{type(e).__name__}: {e}")
            if kind == "frame":
                reply("ready", None)
    if fh is not None:
        fh.close()


if __name__ == "__main__":
    import cv2
    cv2.setNumThreads(CHILD_THREADS)
    serve(sys.stdin.buffer, sys.stdout.buffer, Path(sys.argv[1]), float(sys.argv[2]))
