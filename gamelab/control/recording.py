"""The rolling recording kept by the perception service (from 2026-10-26): its own frames as it takes them, the last 5
minutes in memory, saved as `gamelab recorder` saved them (control.rolling: a Lie Detector alarm's part before it and
its clip, `save-recent`, the status file), where that recorder grabbed the screen itself with mss, 10 times a second.
The user, that morning: "The recording should probably just take frames from within the same perception process to
have a rolling window copy of things in memory, match fps or fps/2." So:

- FPS frames a second, every other one of the service's 30 (service.HZ / 2): a frame shown 1/FPS after the last one
  was due, SLACK_S short of it at most, is handed over from the frame loop (`saw`, from Service._emit: its picture by
  reference, its number and run) into a queue of QUEUE, never waited on (past it a frame is dropped, counted); with the
  frame loop slower than FPS (a crowd: 14.6 a second live), each of its frames. The pictures are the
  frames' own, which perception.frames.Pictures never writes over while they're held, never the shared memory's slots
  (encoding from one crashed the service, an access violation, as a new run unmapped them: D:/MapleHelper-scratch/
  pconst/hunt-recorder).
- An encoder thread keeps each as its difference from the frame before it (rolling.pack_diff), a whole frame every
  KEY_EVERY_S, after a gap, at another size, or where another moment came between (rolling.Recorder.add's `after`): a
  chain, decoded whole when it's saved (rolling.decode). Lossless, as the recorder's: on 150 consecutive frames of
  2026-10-26 01:02 (D:/MapleHelper-scratch/pconst/hunt-recorder/rec_bench.py), 332 KB a frame and 7.6 ms against 482
  KB and 8.9 ms kept whole: 5 minutes at 15 a second about 1.4 GB, 0.11 of a core.
- Only frames of the game: those the frame loop found showing it wholly, in front, uncovered and where its window is
  (ScreenFrames' checks, at the recorder's grid), less those whose capture is offset from the window (framecheck's
  "offset": GAP_KINDS). While the frame loop has none to give for OWN_AFTER_S (the game not on screen, another client
  size, the picture drawn off the client, the capture stalled or failing, the screen unchanged), the recording takes its
  moments itself, as `gamelab recorder` does (rolling.Recorder.tick: the window looked at, grabbed with mss, else a gap
  and why), FPS a second: what that recorder kept, at any client size, this keeps too.
- Its ticks, on a thread of their own FPS times a second: those moments, and rolling.Recorder.poll (the inbox, the
  alarm, the clips, the status file). A save is written on WRITERS threads at the lowest priority (WRITER_PRIORITY),
  WRITERS_HUNTING while a command that hunts is attached.
- One recorder at a time on the PC (rolling.MUTEX): while `gamelab recorder` holds it the service doesn't record, tries
  again every MUTEX_RETRY_S and says it's waiting (WAITING in lab/recorder: the babysitter stops the old recorder when
  it sees it).

What it can't do that the old recorder did: outlive the service. A service that ends (its watchdog's EXIT_STUCK, a
crash, a restart) loses the 5 minutes it held; the next starts empty.
"""

import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from .. import paths
from . import rolling
from .rolling import Item

FPS = 15.0          # frames a second kept: every other frame of the service's 30 (the user: "match fps or fps/2")
# A frame shown this much before it's due is taken (the frames' jitter): a quarter of 1/FPS. Simulated against frame
# loops at 14-30 a second, each period +-15% (D:/MapleHelper-scratch/pconst/build1/cadence.py): every frame of a loop
# at 14 a second, 14.4 a second at the least from loops of 15-30, every other frame at exactly 30; with 4 ms, 12.9 a
# second from a loop at 18.
SLACK_S = 0.016
# A whole frame at least this often: a save starts at one (a chain's frames before what it asks for come too, so a
# save of N s holds up to N + KEY_EVERY_S), and a frame lost to a chain lost loses at most this much. At 15 a second a
# whole frame is 482 KB against a difference's 332 (rec_bench.py): every second is 2% more than every 3.
KEY_EVERY_S = 1.0
QUEUE = 8           # frames waiting for the encoder at most (about half a second): past it the newest are dropped
# s without a frame from the frame loop after which the recording takes its moments itself (rolling.Recorder.tick). A
# frame comes every 1/FPS s while the game shows; the frame loop's slowest steps in a crowd took 0.13-0.15 s (p99,
# D:/MapleHelper-scratch/pconst/plan, the replay's base runs).
OWN_AFTER_S = 0.5
MUTEX_RETRY_S = 10.0  # s between tries for the recorder's mutex while `gamelab recorder` holds it
WRITERS = 4           # threads a save is encoded on (rolling.ENCODERS) ...
WRITERS_HUNTING = 2   # ... while a command that hunts is attached
# The saver's and the writers' thread priority: THREAD_PRIORITY_LOWEST, two below the frame loop's (in the service at
# normal priority, still above the background jobs at Idle). A save of 120 s during a hunt replayed (the crowd's
# frames, the fake tour, D:/MapleHelper-scratch/pconst/build1/replay, verify_save.py): below normal (-1), the frame
# loop's step median 91 ms during it against 76 outside, 10.5 frames a second against 12.4 (and on a 300 s save, the
# machine 73% busy, p99 232 ms against 183, 7.3 a second against 9.7); at the lowest (-2), 72 against 86 and 13.5
# against 11.3, as outside within the machine's swing (70-98% busy), written in 11.7 s (1 writer: 18.9 s).
WRITER_PRIORITY = -2
GAP_KINDS = ("offset",)  # a frame the frame loop read but whose picture may not be the window's: not kept
WAITING = "waiting.json"  # in lab/recorder: the service waiting for the recorder's mutex (pid, since, updated)
SOURCE = "the perception service's frames"


class Recording:
    """The rolling recording in the perception service (see the top): `saw` from the frame loop, `start` its threads
    (the mutex taken first), `close` at the service's end. `title`: the game window's (for its own moments); `game`,
    `grab`: what those look and grab with (rolling.GameWindow, rolling.ScreenGrab; tests pass fakes); `hunting`: whether
    a command that hunts is attached (the service's `fast`); `mutex`: the recorders' (rolling.MUTEX by default); the
    rest goes to the rolling.Recorder it makes (`box`, `lie_root`, `recent`, `clips`, `clock`: tests' own)."""

    def __init__(self, title: str = "", *, game=None, grab=None, fps: float = FPS,
                 hunting: Callable[[], bool] = lambda: False, mutex: str | None = None,
                 log: Callable[[str], None] = print, **recorder_kw):
        self.title, self.fps, self.hunting, self.log = title, fps, hunting, log
        self.mutex = mutex or rolling.MUTEX
        self._game, self._grab, self._kw = game, grab, recorder_kw
        self.clock: Callable[[], float] = recorder_kw.get("clock", time.time)
        self.box = Path(recorder_kw.get("box") or paths.RECORDER)
        self.recorder: rolling.Recorder | None = None
        self.handle = None                 # the mutex, once held
        self.dropped = self.late = self.own = self.taken = 0  # frames dropped (queue full), late, own moments, taken
        self._q: deque[tuple[float, object, int, int]] = deque()
        self._cond = threading.Condition()
        self._offered = -1e18              # (time.monotonic) when the frame loop last handed a frame over
        self._due = -1e18                  # when the next frame is due (the frame loop's clock)
        self._before = None                # the last frame encoded (its picture), and its moment, for a difference
        self._last: Item | None = None
        self._key_t = -1e18                # the last whole frame's time (time.time's)
        self._cpu = 0.0                    # s of CPU of the encoder's and the ticks' threads
        self._closing = False
        self._said_waiting = False
        self._threads: list[threading.Thread] = []

    # --- the frame loop's side ------------------------------------------------------------------------------------

    def saw(self, st, frame, run: int) -> None:
        """The frame loop's state of a frame and the frame (Service._emit; not a late state's): taken if it shows the
        game (see the top) FPS times a second at most, handed to the encoder. Never waits."""
        if self.recorder is None or frame is None or st.late or not frame.visible or frame.full is None:
            return
        if st.bad is not None and st.bad.kind in GAP_KINDS:
            return
        period = 1.0 / self.fps
        if frame.t < self._due - SLACK_S:
            return
        # Due a period after the last one was, not after the last frame taken: one a jitter brought late doesn't put the
        # next off (at 15 frames a second from the loop, all of them taken), and after a stall no catching up.
        self._due = self._due + period if self._due >= frame.t - period / 2 else frame.t + 0.75 * period
        now = time.monotonic()
        t = self.clock() - (now - frame.t)  # (the moment it was shown, on the recorder's clock: time.time's)
        with self._cond:
            self._offered = now
            if len(self._q) >= QUEUE:
                self.dropped += 1
                return
            self._q.append((t, frame.full, st.seq, run))
            self._cond.notify_all()

    # --- the encoder ------------------------------------------------------------------------------------------------

    def encode(self, t: float, full, seq: int, run: int) -> Item | None:
        """One frame into the ring: as its difference from the one before (a whole frame every KEY_EVERY_S, or where
        the chain broke); None if a moment of the recorder's own came after its time (late)."""
        rec = self.recorder
        began, cpu = time.perf_counter(), time.thread_time()
        newest = rec.ring.newest()
        if newest is not None and t <= newest.t:
            self.late += 1
            return None
        size = (int(full.shape[1]), int(full.shape[0]))
        before, last = self._before, self._last
        chained = (before is not None and before.shape == full.shape and last is not None and newest is last
                   and t - self._key_t < KEY_EVERY_S - 1e-6)
        item = Item(t, rolling.pack_diff(full, before) if chained else rolling.pack_whole(full), size=size,
                    key=not chained, rgb=True, seq=seq, run=run)
        if not rec.add(item, after=last if chained else None, cost=time.perf_counter() - began):
            chained = False  # (a moment of the recorder's own came between: a whole frame instead)
            item = Item(t, rolling.pack_whole(full), size=size, rgb=True, seq=seq, run=run)
            rec.add(item, cost=time.perf_counter() - began)
        if not chained:
            self._key_t = t
        self._before, self._last = full, item
        self.taken += 1
        self._cpu += time.thread_time() - cpu
        return item

    def _encoder(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._q or self._closing)
                if self._closing:
                    return
                t, full, seq, run = self._q.popleft()
            try:
                self.encode(t, full, seq, run)
            except Exception as e:  # (nothing that goes wrong in a frame stops the recording)
                self.recorder._say(f"recorder: a frame wasn't kept ({type(e).__name__}: {e})")
            finally:
                del full

    # --- the ticks ----------------------------------------------------------------------------------------------------

    def tick(self) -> None:
        """A tick: a moment of its own if the frame loop has handed none over for OWN_AFTER_S, then the recorder's
        poll (the inbox, the alarm, the clips, the status)."""
        cpu = time.thread_time()
        try:
            if time.monotonic() - self._offered > OWN_AFTER_S:
                self.recorder.tick()
                self.own += 1
            self.recorder.poll()
        finally:
            self._cpu += time.thread_time() - cpu

    def _ticks(self) -> None:
        while not self._closing and not self.take():
            with self._cond:
                self._cond.wait_for(lambda: self._closing, MUTEX_RETRY_S)
        if self._closing:
            return
        period, next_at, errors = 1.0 / self.fps, time.monotonic(), 0
        while not self._closing:
            try:
                self.tick()
            except Exception as e:  # (as rolling.Recorder.run: a moment that fails stops nothing)
                errors += 1
                if errors == 1 or errors % 1000 == 0:
                    self.recorder._say(f"recorder: a moment failed ({type(e).__name__}: {e}; {errors} so far)")
            next_at += period
            wait = next_at - time.monotonic()
            if wait < -period:
                next_at, wait = time.monotonic(), 0.0
            with self._cond:
                self._cond.wait_for(lambda: self._closing, max(0.0, wait))

    def take(self) -> bool:
        """The recorders' mutex taken, and the recorder made (True); else (`gamelab recorder` holds it) WAITING said in
        lab/recorder, for the babysitter."""
        handle = rolling.only_one(self.mutex)
        if handle is None:
            self._waiting()
            return False
        self.handle = handle
        try:
            (self.box / WAITING).unlink(missing_ok=True)
        except OSError:
            pass
        kw = dict(self._kw)
        kw.setdefault("ring", rolling.Ring(max_items=int(2 * rolling.SECONDS * self.fps)))
        self.recorder = rolling.Recorder(
            self._game or rolling.GameWindow(self.title), self._grab or rolling.ScreenGrab(), fps=self.fps,
            log=self.log, encoders=self.writers, cpu=lambda: self._cpu, source=SOURCE, extra=self.counts,
            writer_priority=WRITER_PRIORITY, **kw)
        self.recorder._say(f"recorder: in the perception service (pid {os.getpid()}): its frames {self.fps:g} times a "
                           f"second, the last {self.recorder.ring.seconds / 60:g} minutes kept in memory; `python -m "
                           f"gamelab save-recent` saves them; status: {self.recorder.status_path}")
        return True

    def _waiting(self) -> None:
        now = self.clock()
        info = {"pid": os.getpid(), "since": getattr(self, "_waiting_since", now), "updated": now}
        self._waiting_since = info["since"]
        try:
            self.box.mkdir(parents=True, exist_ok=True)
            tmp = self.box / (WAITING + ".tmp")
            tmp.write_text(json.dumps(info), encoding="utf-8")
            os.replace(tmp, self.box / WAITING)
        except OSError:
            pass
        if not self._said_waiting:
            self._said_waiting = True
            self.log(f"{time.strftime('%H:%M:%S')} recorder: `gamelab recorder` is running (its mutex held): the "
                     f"service records once it's stopped (the babysitter stops it), trying every {MUTEX_RETRY_S:g} s")

    def writers(self) -> int:
        return WRITERS_HUNTING if self.hunting() else WRITERS

    def counts(self) -> dict:
        """For the status file: the frames taken, dropped (the queue full), late, the moments of its own."""
        return {"taken": self.taken, "dropped": self.dropped, "late": self.late, "own_moments": self.own,
                "queue": len(self._q)}

    # --- starting, stopping -------------------------------------------------------------------------------------------

    def start(self, hunting: Callable[[], bool] | None = None) -> None:
        """Its threads: the ticks (which take the mutex first) and the encoder."""
        if hunting is not None:
            self.hunting = hunting
        for target, name in ((self._ticks, "perceive-recording"), (self._encoder, "perceive-recording-encoder")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._threads.append(thread)
            thread.start()

    def line(self) -> str:
        """For the service's console line: what it keeps."""
        rec = self.recorder
        if rec is None:
            return "; not recording (`gamelab recorder` runs)" if self.handle is None and self._said_waiting else ""
        return (f"; recording: {rec.ring.span():.0f} s kept ({rec.ring.bytes / 1024 ** 3:.2f} GB)"
                + (f", {self.dropped} frames dropped" if self.dropped else ""))

    def close(self, timeout: float = 120.0) -> None:
        """Its threads stopped, then the recorder closed (the saves queued written, the clips cut short, its status
        saying it stopped), the mutex let go."""
        with self._cond:
            self._closing = True
            self._q.clear()
            self._cond.notify_all()
        for thread in self._threads:
            thread.join(5.0)
        if self.recorder is not None:
            self.recorder.close(timeout)
        if self.handle is not None:
            rolling.let_go(self.handle)
            self.handle = None
