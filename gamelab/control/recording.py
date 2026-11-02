"""The rolling recording kept by the perception service (from 2026-10-26): its own frames as it takes them, the last 5
minutes in memory, saved as `gamelab recorder` saved them (control.rolling: a Lie Detector alarm's part before it and
its clip, `save-recent`, the status file), where that recorder grabbed the screen itself with mss, 10 times a second.
The user, that morning: "The recording should probably just take frames from within the same perception process to
have a rolling window copy of things in memory, match fps or fps/2." So:

- FPS frames a second, every one of the service's (localization.rate.HZ: 15 since 2026-10-30; every other one of its
  30 until then), picked on their own times by a token bucket (FPS tokens a second, BURST at most, a frame taken with one): handed over from the frame loop (`saw`, from
  Service._emit: its picture by reference, its number and run) into a queue of QUEUE, never waited on (past it a frame
  is dropped, counted). With the frame loop slower than FPS, as many as it makes: on the replays' real frame times
  (D:/MapleHelper-scratch/pconst/finish/cadence_rules.py), a loop at 12-14 a second gave 11.6-13.2 kept, at 25-30 a
  second 15.0 (the loop's own swing from second to second is what's lost: no second gets more than FPS). The pictures
  are the frames' own, which perception.frames.Pictures never writes over while they're held, never the shared
  memory's slots (encoding from one crashed the service, an access violation, as a new run unmapped them:
  D:/MapleHelper-scratch/pconst/hunt-recorder).
- Never fewer than the old recorder's 10 a second (FLOOR_FPS): while the frame loop gives fewer (a crowd, without the
  native pieces of work/perceive-native: 11.7% of the hunts' 10 s spans live on 2026-10-25 and 26 ran it at 7-10 a
  second), the recording takes its moments itself, as `gamelab recorder` does (rolling.Recorder.tick: the window looked
  at, grabbed with mss, looked at again, else a gap and why), FPS a second, until the loop gives FLOOR_BACK_FPS again;
  the loop's frames meanwhile only counted (a frame of its own between two of the loop's would make the second late).
- An encoder thread keeps each as its difference from the frame before it (rolling.pack_diff), a whole frame every
  KEY_EVERY_S, after a gap, at another size, or where another moment came between (rolling.Recorder.add's `after`): a
  chain, decoded whole when it's saved (rolling.decode). Lossless, as the recorder's. On the lossless frames of the alarm
  clip of 2026-10-26 01:02 (10 a second; D:/MapleHelper-scratch/pconst/finish/diff_rule.py) 180-375 KB a frame (the
  hunting before the alarm 265-375, the Lie Detector's dialog 122), 4-7 ms of CPU: 5 minutes at 15 a second 0.5-1.6 GB.
  A difference is bigger than the whole frame on a camera pan (86% of the pairs of one 10 s stretch), but keeping
  those whole saved only 3-5% (356 KB a frame against 375 on that stretch) for 65% more CPU (10.9 ms against 6.6: both
  packed), so it isn't done.
- Only frames of the game: those the frame loop found showing it wholly, in front, uncovered and where its window is
  (ScreenFrames' checks, at the recorder's grid), less those whose capture is offset from the window (framecheck's
  "offset": GAP_KINDS), and with no other program's window over any of it (Frame.over: window.over before the shot
  and after it, as the old recorder asked around its grabs; a frame with one is a gap, "covered"; one not asked, None,
  isn't taken). While the frame loop has none to give for OWN_AFTER_S (the game not on screen, another client size,
  the picture drawn off the client, the capture stalled or failing, the screen unchanged), the recording takes its
  moments itself (rolling.Recorder.tick, as for the floor): what that recorder kept, at any client size, this keeps too.
- Its ticks, on a thread of their own FPS times a second: those moments, and rolling.Recorder.poll (the inbox, the
  alarm, the clips, the status file). A save is written in a process of its own (rolling.write_part_apart, `apart`:
  written in the service, it shared the frame loop's Python lock, whatever the writers' priority: see WRITER_PRIORITY)
  on WRITERS threads there, WRITERS_HUNTING while a command that hunts is attached; in the service itself only if that
  process fails.
- One recorder at a time on the PC (rolling.MUTEX): while `gamelab recorder` holds it the service doesn't record, tries
  again every MUTEX_RETRY_S and says it's waiting (WAITING in lab/recorder: the babysitter stops the old recorder when
  it sees it, once it has no Lie Detector's parts or save to write).

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
from ..localization import rate
from . import rolling
from .rolling import Item

FPS = rate.HZ       # frames a second kept: every frame of the service's (the user: "match fps or fps/2")
# The token bucket's most (frames' worth): 2 lets a frame through after a short interval that follows a long one, so a
# loop slower than FPS has all its frames kept, and after a stall one frame more at most (no catching up). On the
# replays' frame times (cadence_rules.py, 77 runs): 0.1-0.2 a second more than the due grid first built (a frame due
# every 1/FPS, 16 ms early at most), 3 another 0.05-0.1, 1.5 none.
BURST = 2.0
# A whole frame at least this often: a save starts at one (a chain's frames before what it asks for come too, so a
# save of N s holds up to N + KEY_EVERY_S), and a frame lost to a chain lost loses at most this much. At 15 a second a
# whole frame is 482 KB against a difference's 332 (D:/MapleHelper-scratch/pconst/hunt-recorder/rec_bench.py): every
# second is 2% more than every 3.
KEY_EVERY_S = 1.0
QUEUE = 8           # frames waiting for the encoder at most (about half a second): past it the newest are dropped
# s without a frame from the frame loop after which the recording takes its moments itself (rolling.Recorder.tick). A
# frame comes every 1/FPS s while the game shows; the frame loop's slowest steps in a crowd took 0.13-0.15 s (p99,
# D:/MapleHelper-scratch/pconst/plan, the replay's base runs), the slowest of all 0.2-0.24 s (the review's runs of the
# branch). 0.5 s as first built: a stall of the loop's left the recording a gap of half a second before its own came.
OWN_AFTER_S = 0.25
# The floor: fewer than FLOOR_FPS frames from the frame loop in the last FLOOR_WINDOW_S (`gamelab recorder`'s 10 a
# second, which it kept: a median of 10 and never fewer than 9 in a second in the hunts before the Lie Detectors of
# 2026-10-26 00:25 and 01:01, the review of the branch), and the recording takes its own moments until the loop gives
# FLOOR_BACK_FPS a second over FLOOR_BACK_S. On the replays' crowd frames, the loop on a busy machine, 10-50% of the
# seconds held fewer than 10 of its frames (cadence_rules.py); live, 11.7% of the hunts' 10 s status lines said 7-10.
FLOOR_FPS = 10.0
FLOOR_WINDOW_S = 1.0
FLOOR_BACK_FPS = 12.0
FLOOR_BACK_S = 2.0
FLOOR_SAY_S = 600.0   # s between the log's lines on the floor taken (the count is in the status file)
MUTEX_RETRY_S = 10.0  # s between tries for the recorder's mutex while `gamelab recorder` holds it
WRITERS = 4           # threads a save is encoded on (rolling.ENCODERS) ...
WRITERS_HUNTING = 2   # ... while a command that hunts is attached
# The saver's thread priority (and the writers', a save written in the service: `apart` off, or its process failed):
# THREAD_PRIORITY_LOWEST, two below the frame loop's. It isn't what a save in the service cost the frame loop: on a
# quiet machine (16-32% busy) a 120 s save during a hunt replayed made its steps wait 18-20 ms (median) against 1-2
# outside it, 9.0-9.7 frames a second against 13.0-14.3, at the lowest (-2) and at normal (0) alike, 2 runs each
# alternated (D:/MapleHelper-scratch/pconst/finish/replay: s_p2*, s_p0*): Python's lock, shared with the writers.
WRITER_PRIORITY = -2
GAP_KINDS = ("offset",)  # a frame the frame loop read but whose picture may not be the window's: not kept
COVERED = "covered"      # a frame with another program's window over the game (Frame.over): a gap, as the recorder's
WAITING = "waiting.json"  # in lab/recorder: the service waiting for the recorder's mutex (pid, since, updated)
SOURCE = "the perception service's frames"


class Recording:
    """The rolling recording in the perception service (see the top): `saw` from the frame loop, `start` its threads
    (the mutex taken first), `close` at the service's end. `title`: the game window's (for its own moments); `game`,
    `grab`: what those look and grab with (rolling.GameWindow, rolling.ScreenGrab; tests pass fakes); `hunting`: whether
    a command that hunts is attached (the service's `fast`); `mutex`: the recorders' (rolling.MUTEX by default);
    `apart`: its saves written in a process of their own (rolling.write_part_apart; False: in this one, as first built);
    the rest goes to the rolling.Recorder it makes (`box`, `lie_root`, `recent`, `clips`, `clock`: tests' own)."""

    def __init__(self, title: str = "", *, game=None, grab=None, fps: float = FPS,
                 hunting: Callable[[], bool] = lambda: False, mutex: str | None = None,
                 log: Callable[[str], None] = print, apart: bool = True, **recorder_kw):
        self.title, self.fps, self.hunting, self.log = title, fps, hunting, log
        self.apart = apart  # its saves written in a process of their own (rolling.write_part_apart)
        self.mutex = mutex or rolling.MUTEX
        self._game, self._grab, self._kw = game, grab, recorder_kw
        self.clock: Callable[[], float] = recorder_kw.get("clock", time.time)
        self.box = Path(recorder_kw.get("box") or paths.RECORDER)
        self.recorder: rolling.Recorder | None = None
        self.handle = None                 # the mutex, once held
        self.dropped = self.late = self.own = self.taken = 0  # frames dropped (queue full), late, own moments, taken
        self.covered = 0                   # the frame loop's frames with another window over the game (gaps)
        self.floor = False                 # its own moments for the floor (the frame loop under FLOOR_FPS)
        self.floor_own = self.floor_times = 0  # own moments for the floor, times it was taken
        self._q: deque[tuple[float, object, int | None, int | None, str]] = deque()
        self._cond = threading.Condition()
        self._offered = -1e18              # (time.monotonic) when the frame loop last handed a frame over
        self._offering_since = -1e18       # ... since when it has without a pause of OWN_AFTER_S
        self._offers: deque[float] = deque()  # when it handed them over, the last FLOOR_BACK_S (the floor's count)
        self._floor_since = -1e18
        self._floor_said = -1e18
        self._tokens = BURST               # the bucket (frames' worth) ...
        self._bucket_t: float | None = None  # ... as of this frame's time (the frame loop's clock)
        self._before = None                # the last frame encoded (its picture), and its moment, for a difference
        self._last: Item | None = None
        self._key_t = -1e18                # the last whole frame's time (time.time's)
        self._cpu = 0.0                    # s of CPU of the encoder's and the ticks' threads
        self._closing = False
        self._said_waiting = False
        self._threads: list[threading.Thread] = []

    # --- the frame loop's side ------------------------------------------------------------------------------------

    def saw(self, st, frame, run: int) -> None:
        """The frame loop's state of a frame and the frame (Service._emit): taken if it shows the
        game (see the top) FPS times a second at most, handed to the encoder (a frame with a window over the game as a
        gap). Never waits. While the floor is taken, only counted."""
        if self.recorder is None or frame is None or not frame.visible or frame.full is None:
            return
        if (st.bad is not None and st.bad.kind in GAP_KINDS) or frame.over is None:
            return
        now = time.monotonic()
        if now - self._offered > OWN_AFTER_S:
            self._offering_since = now
        self._offered = now
        self._offers.append(now)
        if self.floor:
            return
        if self._bucket_t is not None:
            self._tokens = min(BURST, self._tokens + max(0.0, frame.t - self._bucket_t) * self.fps)
        self._bucket_t = frame.t
        if self._tokens < 1.0 - 1e-9:
            return
        self._tokens -= 1.0
        t = self.clock() - (now - frame.t)  # (the moment it was shown, on the recorder's clock: time.time's)
        full, covered = frame.full, 0.0
        if frame.over:
            # Another program's window over part of the game: the frame kept as it is, the boxes it covered noted
            # (since 2026-11-01); over all of it, or over it where the boxes aren't known (the walk failed), a gap
            # (nothing of its picture kept, nor held), as every such frame was until then.
            size = (int(full.shape[1]), int(full.shape[0]))
            covered = rolling.share(size, frame.over_boxes) if frame.over_boxes else 1.0
            if covered >= 1.0:
                full = None
        with self._cond:
            if len(self._q) >= QUEUE:
                self.dropped += 1
                return
            if full is None:
                self._q.append((t, None, None, None, frame.over, 0.0, ()))
            else:
                self._q.append((t, full, st.seq, run, frame.over or "", covered, frame.over_boxes if covered else ()))
            self._cond.notify_all()

    # --- the encoder ------------------------------------------------------------------------------------------------

    def encode(self, t: float, full, seq: int | None, run: int | None, over: str = "",
               covered: float = 0.0, boxes: tuple = ()) -> Item | None:
        """One frame into the ring: as its difference from the one before (a whole frame every KEY_EVERY_S, or where
        the chain broke), or, `full` None, a gap (COVERED: `over` says by what); None if a moment of the recorder's own
        came after its time (late). `covered`, `boxes`: what of it other programs' windows were over (Item.over)."""
        rec = self.recorder
        began, cpu = time.perf_counter(), time.thread_time()
        newest = rec.ring.newest()
        if newest is not None and t <= newest.t:
            self.late += 1
            return None
        if full is None:
            item = Item(t, None, COVERED)
            rec.add(item, detail=over)
            self._before, self._last = None, item  # (a gap breaks the chain)
            self.covered += 1
            self._cpu += time.thread_time() - cpu
            return item
        size = (int(full.shape[1]), int(full.shape[0]))
        before, last = self._before, self._last
        chained = (before is not None and before.shape == full.shape and last is not None and newest is last
                   and t - self._key_t < KEY_EVERY_S - 1e-6)
        item = Item(t, rolling.pack_diff(full, before) if chained else rolling.pack_whole(full), size=size,
                    key=not chained, rgb=True, seq=seq, run=run, covered=covered, over=tuple(boxes))
        detail = over if covered else ""  # (the window over part of it, for the log)
        if not rec.add(item, after=last if chained else None, cost=time.perf_counter() - began, detail=detail):
            chained = False  # (a moment of the recorder's own came between: a whole frame instead)
            item = Item(t, rolling.pack_whole(full), size=size, rgb=True, seq=seq, run=run, covered=covered,
                        over=tuple(boxes))
            rec.add(item, cost=time.perf_counter() - began, detail=detail)
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
                t, full, seq, run, over, covered, boxes = self._q.popleft()
            try:
                self.encode(t, full, seq, run, over, covered, boxes)
            except Exception as e:  # (nothing that goes wrong in a frame stops the recording)
                self.recorder._say(f"recorder: a frame wasn't kept ({type(e).__name__}: {e})")
            finally:
                del full

    # --- the ticks ----------------------------------------------------------------------------------------------------

    def tick(self) -> None:
        """A tick: a moment of its own if the frame loop has handed none over for OWN_AFTER_S, or while the floor is
        taken (`floored`), then the recorder's poll (the inbox, the alarm, the clips, the status)."""
        cpu = time.thread_time()
        try:
            now = time.monotonic()
            if now - self._offered > OWN_AFTER_S:
                self.recorder.tick()
                self.own += 1
            elif self.floored(now):
                self.recorder.tick()
                self.own += 1
                self.floor_own += 1
            self.recorder.poll()
        finally:
            self._cpu += time.thread_time() - cpu

    def floored(self, now: float) -> bool:
        """Whether the recording takes its own moments for the floor now: the frame loop has handed over fewer than
        FLOOR_FPS frames in the last FLOOR_WINDOW_S (once it's handed them over that long), until it gives FLOOR_BACK_FPS
        a second over FLOOR_BACK_S (once the floor's been taken that long)."""
        offers = self._offers
        while offers and offers[0] < now - FLOOR_BACK_S:
            offers.popleft()
        recent = list(offers)  # (a copy: the frame loop appends meanwhile)
        if self.floor:
            if now - self._floor_since >= FLOOR_BACK_S and len(recent) >= FLOOR_BACK_FPS * FLOOR_BACK_S:
                self.floor = False
            return self.floor
        if now - self._offering_since < FLOOR_WINDOW_S:
            return False
        got = sum(1 for t in recent if t >= now - FLOOR_WINDOW_S)
        if got >= FLOOR_FPS:
            return False
        self.floor, self._floor_since = True, now
        self.floor_times += 1
        if now - self._floor_said >= FLOOR_SAY_S:
            self._floor_said = now
            self.recorder._say(f"recorder: the frame loop gave {got} frames in the last {FLOOR_WINDOW_S:g} s: the "
                               f"recording grabs its own, {self.fps:g} a second, until it gives {FLOOR_BACK_FPS:g} a "
                               f"second again ({self.floor_times} times so far; said every {FLOOR_SAY_S / 60:g} "
                               f"minutes at most)")
        return True

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
        from ..perception import frames  # (its walk up the z-order with Python's lock kept: frames._overs)
        self.recorder = rolling.Recorder(
            self._game or rolling.GameWindow(self.title, overs=frames._overs), self._grab or rolling.ScreenGrab(),
            fps=self.fps,
            log=self.log, encoders=self.writers, cpu=lambda: self._cpu, source=SOURCE, extra=self.counts,
            writer_priority=WRITER_PRIORITY, apart=self.apart, **kw)
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
        """For the status file: the frames taken, dropped (the queue full), late, covered, the moments of its own (and
        those for the floor, the times it was taken, whether it is now)."""
        return {"taken": self.taken, "dropped": self.dropped, "late": self.late, "covered": self.covered,
                "own_moments": self.own, "floor_moments": self.floor_own, "floor_times": self.floor_times,
                "floor": self.floor, "queue": len(self._q)}

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
                + (f", {self.dropped} frames dropped" if self.dropped else "")
                + (f", its own grabs now (the frame loop under {FLOOR_FPS:g} a second)" if self.floor else "")
                + (f", {self.floor_own} own grabs for the floor so far" if self.floor_own else ""))

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
