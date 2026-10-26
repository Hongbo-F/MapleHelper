"""The rolling recording: the game's client area 10 times a second, the last 5 minutes kept in memory losslessly (its
pixels zstd-compressed: `pack`; JPEG until 2026-10-25), so what led up to a moment can be saved after it. The user asked for it on 2026-10-24 (21:50): something had popped up in the
game wanting an answer within 10 s, and nothing recorded it (the player's Lie Detector check knew only the countdown,
the mouse mini-game and the Truth Booth, and records only once it has seen one). It was the image quiz: since
2026-10-25 the player knows it and its warning, and the mini-game's notice, so their alarms are saved as any.

`python -m gamelab recorder` runs it (README: The rolling recording), at below-normal priority, kept running by the
babysitter, hunts or not. What it keeps is saved:
  - on a Lie Detector alarm (control.alarm's ALARM.json, written by the player's watchdog, whatever form it was seen
    as: its `form`): the 5 minutes before it, and on until the player's own recording of it has begun, go into that
    recording's folder as `before/`
    (control.lie_recorder), so the whole cycle is kept with what led up to it; as one is written, the Lie Detector's
    recordings are pruned to their 10 GB counting those parts (`_prune_lie`), whatever the player's age; and, on its
    own whatever becomes of those, the alarm's clip, from 10 s before it to 60 s after, to
    paths.ALARM_CLIPS/<YYYYmmdd-HHMMSS>-lie-detector[-<form>] (CLIP_BEFORE_S, CLIP_AFTER_S), lossless (PNG);
  - on demand (`python -m gamelab save-recent`, `ask` here): the last N s, to paths.RECENT/<YYYYmmdd-HHMMSS>[-name],
    asked through a file inbox (paths.RECORDER: inbox/<id>.json, taken as <id>.taken, answered in outbox/<id>.json).

A saved part (`write_part`), one folder:
  00000.jpg ...  the frames, as grabbed, encoded as they're written: JPEG (JPEG_PARAMS), or PNG (PNG_PARAMS: the
                 alarm's clip);
  index.jsonl    in time order, a line a frame: `i`, `file`, `t` (s from the part's first moment) and `at` (the clock
                 time); and a line a gap, the moments in a row with no frame for the same reason: `gap` (why: "no
                 window", "minimized", "no client area", "off the screen", "not in front", "covered", "moved", "capture
                 failed", "not encoded"), `t` and `until` (its first and last moment), `ticks`;
  summary.json   what it is (`kind`: "save-recent" or "before the Lie Detector", with the alarm and the moment it was
                 seen, `alarm_t`), `from` and `to`, `seconds`, `ticks`, `frames`, their `bytes`, the `gaps` (s by why),
                 `fps`, the `format` ("jpeg" or "png"), `jpeg_quality` (a JPEG part's) and the `window`'s size (the
                 last frame's).

A frame is kept only when the screen shows the game: its window there and not minimized, wholly on the monitors, in
front, and no other program's window over any of it (window.shows at perception.framecheck's COVER_GRID, then
window.over: every window above it in the z-order, as a window between the grid's points or one clicks pass through
isn't seen at them), asked before the grab and again after it, the window where it was. Otherwise the moment is a gap:
nothing of another window is ever kept. It reads the screen only: no keys, no clicks, and it never brings the game
forward.

Captured with mss, not DXGI desktop duplication: duplications of a monitor are limited (on 2026-10-24, with `watch` and
`pet` running, two more opened and a third failed: DuplicateOutput 0x887A0022, DXGI_ERROR_NOT_CURRENTLY_AVAILABLE). One
held all the time would leave a hunt's looks and its feed one fewer, and one that failed to open falls back to mss for
capture.RETRY_S (17 ms looks). mss takes none; its grab costs 2-3 ms more CPU (README: The rolling recording).

Imports: numpy, OpenCV, the shared modules, perception.framecheck (COVER_GRID), and control.alarm and
control.lie_recorder (the JPEG settings, the Lie Detector's folders, and pruning a folder of recordings: `recent/`, and
the Lie Detector's when a part goes into one, `_prune_lie`); zstd from the standard library (Python 3.14's
compression.zstd; zlib before it).
"""

import ctypes
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import capture, paths, window
from ..perception import framecheck
from . import alarm, lie_recorder

try:
    from compression import zstd  # (Python 3.14)

    def _compress(raw: bytes) -> bytes:
        return zstd.compress(raw, ZSTD_LEVEL)
    _decompress = zstd.decompress
except ImportError:  # (an older Python: zlib, lossless too, slower)
    import zlib

    def _compress(raw: bytes) -> bytes:
        return zlib.compress(raw, 1)
    _decompress = zlib.decompress

FPS = 10.0         # moments a second: a frame or a gap each
SECONDS = 300.0    # the last 5 minutes are kept (3,000 moments at FPS)
# Frames are kept losslessly since 2026-10-25 (the user asked): their BGR pixels zstd-compressed at level 1 (`pack`).
# On six 1366x768 frames saved losslessly (three scenes, a calibration shot, two Lie Detector boards; D:/MapleHelper-
# scratch's lossless.py): 757 KB a frame (381-1163) and 8.4 ms to pack, 2.2 GB for 5 minutes, 3.4 at the busiest.
# PNG at its fastest (level 1) was as big and took 55 ms, too slow for 10 a second; WebP lossless 502 KB in 667 ms;
# zstd level 3, 666 KB in 12.8 ms. They were JPEG (quality 75, full-resolution colour) until then: 253 KB in 4.6 ms on
# the same frames, 33 dB PSNR. A part is encoded as it's written: JPEG (`before/` and `save-recent`'s, as before) or
# PNG (the alarm's clip: lossless on disk too).
ZSTD_LEVEL = 1
JPEG_PARAMS = lie_recorder.JPEG_PARAMS  # the Lie Detector recordings' settings: a frame of either reads alike
# (813 KB and 55 ms a frame on the six, on ENCODERS threads as it's written.) Without PNG's row filters: on 80 frames
# of the two alarm clips of 2026-10-26, 31.0 ms of CPU a frame against 54.6, 581 KB against 538, the same pixels read
# back: a clip's 700 frames 17 s of CPU less, in the burst after an alarm (D:/MapleHelper-scratch/cpu/apply/
# pngbench.py).
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1, cv2.IMWRITE_PNG_FILTER, cv2.IMWRITE_PNG_FILTER_NONE]
ENCODERS = 4       # threads encoding a part's frames as it's written (OpenCV and zstd let go of the GIL)
# The frames kept take at most this much memory: past it the oldest go, younger than SECONDS or not (counted as
# `trimmed` in the status). 5 minutes at 1366x768 always fit (3.4 GB at the busiest); at 1920x1080, twice the pixels,
# about 4 (1 GB, 5 minutes of JPEG, until 2026-10-25). The PC has 32 GB.
MAX_BYTES = 4 * 1024 ** 3
MAX_ITEMS = int(2 * SECONDS * FPS)  # moments kept at most whatever the clock does (it can be set back)
RECENT_KEEP_BYTES = 5 * 1024 ** 3  # save-recent's saves: the oldest go, when one is made, past this (about 8 of 5 min)
ALARM_POLL_S = 0.5    # how often it looks for a new Lie Detector alarm
STATUS_S = 5.0        # how often it writes its status file (what `recorder --status` and `save-recent` read)
STATUS_STALE_S = 30.0  # a status file older than this: no recorder is running
STATUS_TRIES, STATUS_RETRY_S = 5, 0.02  # a status file held open by a reader: tried again this often, this far apart
LOG_EVERY_S = 600.0   # a line in its log this often: what it keeps, its rate and its cost
# On an alarm, how long it waits for the player's recording of it to begin (its folder in paths.LIE_RECORDINGS): the
# player writes the alarm, pages (about 1.2 s of beeps) and then starts it. Without one by then (the player not
# running, or `watch --lie-detector` raised it: it records nothing), the part goes to paths.RECENT, named for it.
FOLDER_WAIT_S = 20.0
FOLDER_SLACK_S = 2.0  # ... a recording counts as the alarm's if its folder is named for no more than this before it
# Each alarm also gets a clip of its own, whatever becomes of the player's recording and the part before it (the user,
# 2026-10-25: "a 2nd line of defense"): from CLIP_BEFORE_S before the alarm to CLIP_AFTER_S after, 10 a second, in
# paths.ALARM_CLIPS, written once its last moment is past (or, cut short, when the recorder stops). Its moments are
# held by the clip itself as they come, so a ring that loses them first (its cap) loses none of the clip's. Written as
# PNG, lossless: 70 s at 1366x768 is about 560 MB (813 KB a frame on the six frames of PNG_PARAMS' note), in about 14 s
# on ENCODERS threads; the oldest go past CLIPS_KEEP_BYTES, about 55 of them (D: had 27 GB free on 2026-10-25; the
# user is making room).
CLIP_BEFORE_S = 10.0
CLIP_AFTER_S = 60.0
CLIPS_KEEP_BYTES = 30 * 1024 ** 3  # (5 GB until the user asked for 30, 2026-10-25)
# The part before the Lie Detector ends this long after the alarm: the player's own frames begin about 1.5 s after it
# (it pages first). It's written with the clip, once CLIP_AFTER_S are past: until 2026-10-25 it was written at once, as
# the player's recording began, and its 3,000 frames' encoding ran beside the grabs while the Lie Detector was up.
BEFORE_TAIL_S = 5.0
SAVES_WAITING = 4     # saves waiting to be written at most; a request past it is answered "busy"
ASK_TAKEN_S = 3.0     # save-recent: the recorder takes a request within this (it looks every tick), or isn't running
ASK_WAIT_S = 180.0    # ... and answers within this (writing 5 minutes of frames takes seconds)
BEFORE = "before"     # the part in a Lie Detector recording's folder
INDEX, SUMMARY = "index.jsonl", "summary.json"
STAMP = lie_recorder.STAMP
MUTEX = "Local\\gamelab-rolling-recorder"  # one recorder at a time on this PC (`only_one`)
BELOW_NORMAL = 0x4000  # Windows' BELOW_NORMAL_PRIORITY_CLASS
ERROR_ALREADY_EXISTS = 183
NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class Item:
    """One moment: a frame (its pixels packed losslessly: `pack`; its size) or a gap (why there's no frame)."""
    t: float                     # when (time.time's clock): the frame's grab, the gap's look at the window
    data: bytes | None = None    # the frame's BGR pixels, compressed (`pack`); None: a gap
    why: str = ""                # a gap's reason
    size: tuple[int, int] | None = None  # a frame's width, height

    def pixels(self) -> np.ndarray:
        """The frame's BGR pixels, as grabbed."""
        w, h = self.size
        return np.frombuffer(_decompress(self.data), np.uint8).reshape(h, w, 3)


@dataclass(slots=True)
class Clip:
    """An alarm's saves being gathered: the alarm (ALARM.json's), when it was seen (time.time's clock), its clip's
    moments held so far, from CLIP_BEFORE_S before it (Recorder.check_alarm, then each tick's), and what the ring held
    as the alarm was seen, for the part before it."""
    info: dict
    seen: float
    items: list[Item]
    before: list[Item]


class Ring:
    """The last `seconds` of moments, oldest first, at most `max_bytes` of frames: past it the oldest go first, younger
    than `seconds` or not (`trimmed` counts those frames). Thread-safe: the recorder adds, the saver reads."""

    def __init__(self, seconds: float = SECONDS, max_bytes: int = MAX_BYTES, max_items: int = MAX_ITEMS):
        self.seconds, self.max_bytes, self.max_items = seconds, max_bytes, max_items
        self.items: deque[Item] = deque()
        self.bytes = self.frames = self.trimmed = 0
        self.lock = threading.Lock()

    def add(self, item: Item) -> None:
        with self.lock:
            self.items.append(item)
            if item.data is not None:
                self.bytes += len(item.data)
                self.frames += 1
            since = item.t - self.seconds + 1e-6  # (a moment exactly `seconds` before the newest is out)
            trimming = False  # frames went for the cap: so do the gaps before the oldest left
            while self.items:
                old = self.items[0]
                if not (old.t <= since or self.bytes > self.max_bytes or len(self.items) > self.max_items
                        or (trimming and old.data is None)):
                    break
                self.items.popleft()
                if old.data is not None:
                    self.bytes -= len(old.data)
                    self.frames -= 1
                    if old.t > since:
                        self.trimmed += 1
                        trimming = True

    def last(self, seconds: float) -> list[Item]:
        """The moments of the last `seconds` up to the newest, oldest first (a list of its own)."""
        with self.lock:
            if not self.items:
                return []
            since = self.items[-1].t - seconds + 1e-6
            return [it for it in self.items if it.t > since]

    def after(self, t: float) -> list[Item]:
        """The moments after `t`, oldest first."""
        with self.lock:
            return [it for it in self.items if it.t > t]

    def span(self) -> float:
        """How long the moments kept cover (s): from the oldest to the newest, and the newest's tick."""
        with self.lock:
            return 0.0 if not self.items else self.items[-1].t - self.items[0].t + 1 / FPS

    def gaps(self) -> Counter:
        """The gaps kept: moments by why."""
        with self.lock:
            return Counter(it.why for it in self.items if it.data is None)


# --- what it looks at and grabs ------------------------------------------------------------------------------------

class GameWindow:
    """The game window as the recorder asks after it (`check`): its client area (screen px, an mss box) when the screen
    shows all of it, else None and why not, in a few words. `detail` names the window over it, when one was."""

    def __init__(self, title: str):
        self.title = title
        self.covering: int | None = None  # the other program's window over the game at the last check (window.over)

    def check(self) -> tuple[dict | None, str]:
        try:
            hwnd = window.find_game(self.title)
        except RuntimeError:
            return None, "no window"
        if window.minimized(hwnd):  # (its client area is then 432x54 somewhere, not 0x0: window.minimized's note)
            return None, "minimized"
        area = window.client_rect(hwnd)
        if area["width"] <= 0 or area["height"] <= 0:
            return None, "no client area"
        if not window.on_screen(area):
            return None, "off the screen"
        if window.foreground_title() != self.title:
            return None, "not in front"
        self.covering = None
        if not window.shows(hwnd, area, grid=framecheck.COVER_GRID):
            return None, "covered"
        self.covering = window.over(hwnd, area)
        if self.covering is not None:
            return None, "covered"
        return area, ""

    def detail(self) -> str:
        """The window over the game at the last check, as the log names it ("" if none was found by window.over)."""
        if self.covering is None:
            return ""
        try:
            return "by " + window.describe(self.covering)
        except Exception:  # (gone meanwhile)
            return ""


class ScreenGrab:
    """A box of the screen as it shows (BGRA), through mss (capture.Grabber "mss": no duplication of the monitor is
    held; see the module's notes)."""

    def __init__(self):
        self.grabber = capture.Grabber("mss")

    def __call__(self, area: dict) -> np.ndarray:
        return self.grabber.shot(area).crop(0, 0, area["width"], area["height"])

    def close(self) -> None:
        self.grabber.close()


_packing = threading.local()  # pack's BGR picture, a thread's, kept from frame to frame


def pack(frame: np.ndarray) -> bytes:
    """A frame (BGRA as grabbed, or BGR) kept losslessly: its BGR pixels compressed (the alpha is dropped by OpenCV,
    1.7 ms; numpy's slice made contiguous took 6.6 ms); Item.pixels gives them back. 9.5 ms on the six frames. The BGR
    picture is the thread's own, made again only when the size changes: a new one a frame was 3 MB of fresh memory, 770
    page faults that Windows' kernel zeroes (its System process: about a core for 100,000 a second, D:/MapleHelper-
    scratch/cpu/review-numbers/zerotest.py), 7,700 a second at FPS; compressed, nothing of it is kept."""
    if frame.shape[2] != 4:
        return _compress(memoryview(np.ascontiguousarray(frame)).cast("B"))
    bgr = getattr(_packing, "bgr", None)
    if bgr is None or bgr.shape[:2] != frame.shape[:2]:
        bgr = _packing.bgr = np.empty((*frame.shape[:2], 3), np.uint8)
    return _compress(memoryview(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR, dst=bgr)).cast("B"))


def encode(frame: np.ndarray, png: bool = False) -> bytes:
    """A frame's pixels (BGR) as a JPEG (JPEG_PARAMS), or a PNG (PNG_PARAMS), to write."""
    ok, buf = cv2.imencode(".png" if png else ".jpg", frame, PNG_PARAMS if png else JPEG_PARAMS)
    if not ok:
        raise ValueError("the encoder refused it")
    return buf.tobytes()


# --- the recorder ----------------------------------------------------------------------------------------------------

class Recorder:
    """The rolling recording: `tick` takes a moment (a frame, or a gap and why), `poll` looks at the inbox, the alarm
    and the status between ticks, `run` does both FPS times a second. Saves are written on a thread of their own, one
    after the other (`flush` waits for them). `game` says whether the screen shows the game (GameWindow), `grab`
    grabs a box of the screen (ScreenGrab); tests pass fakes, and their own folders: `box` (the inbox, the outbox and
    the status file: paths.RECORDER), `lie_root` (paths.LIE_RECORDINGS), `recent` (paths.RECENT) and `clips`
    (paths.ALARM_CLIPS)."""

    def __init__(self, game, grab: Callable[[dict], np.ndarray], ring: Ring | None = None, *,
                 box: Path | None = None, lie_root: Path | None = None, recent: Path | None = None,
                 clips: Path | None = None, clock: Callable[[], float] = time.time,
                 pack_frame: Callable[[np.ndarray], bytes] = pack, fps: float = FPS,
                 log: Callable[[str], None] = print, folder_wait: float = FOLDER_WAIT_S,
                 recent_keep: int = RECENT_KEEP_BYTES, lie_keep: int = lie_recorder.KEEP_BYTES,
                 clips_keep: int = CLIPS_KEEP_BYTES):
        self.game, self.grab, self.ring = game, grab, ring or Ring()
        box = Path(box or paths.RECORDER)
        self.inbox, self.outbox, self.status_path = box / "inbox", box / "outbox", box / "status.json"
        self.lie_root = Path(lie_root or paths.LIE_RECORDINGS)
        self.recent = Path(recent or paths.RECENT)
        self.clips = Path(clips or paths.ALARM_CLIPS)
        self.clock, self.pack, self.fps, self.log = clock, pack_frame, fps, log
        self.folder_wait, self.recent_keep, self.lie_keep = folder_wait, recent_keep, lie_keep
        self.clips_keep = clips_keep
        self.pending_clips: list[Clip] = []  # alarms whose clip's last moment isn't past yet
        self.started = self.clock()
        self.state: str | None = None  # the last moment's gap reason ("": a frame), as the log last said it
        self.counts: Counter = Counter()  # moments by what: "frame", or a gap's why
        self.failures = 0
        self.saves, self.last_save = 0, None
        self.alarm_handled = (alarm.alarm_info() or {}).get("time")  # an alarm up as it starts isn't saved
        self._next_alarm = self._next_status = self.started
        self._next_log = self.started + LOG_EVERY_S
        self._costs: list[float] = []  # s a frame (look, grab, look, encode) since the last status
        self._cpu, self._wall = time.process_time(), time.monotonic()
        self._rate = {"fps": None, "ms_a_frame": None, "cpu_ms_a_frame": None, "core_share": None}
        self.jobs: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        for folder in (self.inbox, self.outbox):
            folder.mkdir(parents=True, exist_ok=True)
        for stale in self.inbox.glob("*.taken"):  # taken by a recorder that stopped before it answered
            stale.unlink(missing_ok=True)
        self._saver = threading.Thread(target=self._save_loop, name="rolling-saves", daemon=True)
        self._saver.start()

    # --- the moments ---

    def tick(self) -> Item:
        """One moment, added to the ring: the frame grabbed when the screen shows the game before the grab and after
        it, the window where it was; else a gap and why."""
        started = time.perf_counter()
        area, why = self.game.check()
        frame = None
        if area is not None:
            try:
                frame = self.grab(area)
            except Exception as e:  # mss failed (the desktop locked, a UAC prompt's secure desktop, ...)
                why = "capture failed"
                self.failures += 1
                if self.failures == 1 or self.failures % 1000 == 0:
                    self._say(f"recorder: a grab failed ({type(e).__name__}: {e}; {self.failures} so far)")
            else:
                after, why = self.game.check()
                if not why and after != area:
                    why = "moved"
        t = self.clock()
        item = Item(t, None, why)
        if not why:
            try:
                item = Item(t, self.pack(frame), size=(int(frame.shape[1]), int(frame.shape[0])))
                self._costs.append(time.perf_counter() - started)
            except Exception as e:
                item = Item(t, None, "not encoded")
                self._say(f"recorder: a frame wasn't encoded ({type(e).__name__}: {e})")
        self.ring.add(item)
        for clip in self.pending_clips:  # (held there too: the ring may lose them, its cap, before the clip's end)
            clip.items.append(item)
        self.counts[item.why or "frame"] += 1
        self._said(item.why)
        return item

    def _said(self, why: str) -> None:
        """The log says when the game starts or stops showing (a moment of the window moving or a grab failing
        changes nothing: those come and go)."""
        if why in ("moved", "capture failed", "not encoded") or why == self.state:
            return
        self.state = why
        detail = self.game.detail() if why == "covered" and hasattr(self.game, "detail") else ""
        self._say("recorder: the game is on screen: recording" if not why else
                  f"recorder: not recording: {why}" + (f" ({detail})" if detail else ""))

    # --- between moments ---

    def poll(self) -> None:
        """Between ticks: a request in the inbox (every tick), a new Lie Detector alarm (every ALARM_POLL_S), an alarm's
        clip whose last moment is past (every tick), the status file (every STATUS_S) and the log's line (every
        LOG_EVERY_S)."""
        self.read_inbox()
        now = self.clock()
        if now >= self._next_alarm:
            self._next_alarm = now + ALARM_POLL_S
            self.check_alarm()
        self.check_clips(now)
        if now >= self._next_status:
            self._next_status = now + STATUS_S
            self.write_status()
        if now >= self._next_log:
            self._next_log = now + LOG_EVERY_S
            self._say("recorder: " + self.summary())

    def run(self, seconds: float | None = None) -> None:
        """Tick and poll FPS times a second, for `seconds` or until Ctrl+C; then `close`. Behind (a save, the PC
        busy), it goes on from now: no moments are made up."""
        period = 1.0 / self.fps
        start = next_at = time.monotonic()
        self._say(f"recorder: the game's window {self.fps:g} times a second, the last {self.ring.seconds / 60:g} "
                  f"minutes kept in memory (at most {self.ring.max_bytes / 1024 ** 3:g} GB); `python -m gamelab "
                  f"save-recent` saves them; status: {self.status_path}")
        errors = 0
        try:
            while seconds is None or time.monotonic() - start < seconds:
                try:
                    self.tick()
                    self.poll()
                except Exception as e:  # (nothing that goes wrong in a moment stops the recording: its ring would go)
                    errors += 1
                    if errors == 1 or errors % 1000 == 0:
                        self._say(f"recorder: a moment failed ({type(e).__name__}: {e}; {errors} so far)")
                next_at += period
                wait = next_at - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                elif wait < -period:
                    next_at = time.monotonic()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def check_alarm(self) -> None:
        """A Lie Detector alarm not seen before (by its time): what the ring holds now, and what comes in its first
        BEFORE_TAIL_S, goes to the player's recording of it as `before/`, and its clip to `clips`; both are written once
        the clip's last moment is past (check_clips), so nothing heavy runs beside the grabs while the Lie Detector is
        up (the user, 2026-10-25)."""
        info = alarm.alarm_info()
        key = None if info is None else info.get("time")
        if key is None or key == self.alarm_handled:  # (none, or one read as it was written: looked at again soon)
            return
        self.alarm_handled = key
        try:
            seen = alarm.ALARM.stat().st_mtime
        except OSError:
            seen = self.clock()
        snapshot = self.ring.last(self.ring.seconds)
        seen_as = f", {info['form']}" if info.get("form") else ""
        self._say(f"recorder: Lie Detector alarm ({info.get('reason', '?')}{seen_as}, {key}): saving the "
                  f"{len(snapshot) / self.fps:.0f} s before it with the player's recording of it, and its own clip "
                  f"from {CLIP_BEFORE_S:g} s before it to {CLIP_AFTER_S:g} s after in {self.clips}, once those "
                  f"{CLIP_AFTER_S:g} s are past")
        # The moments are held from now on (`tick` adds each to the clip), the ring's from CLIP_BEFORE_S before the
        # alarm first, and all it holds now for the part before: the ring may lose them, its cap, before the clip's end.
        self.pending_clips.append(Clip(info, seen, [it for it in snapshot if it.t > seen - CLIP_BEFORE_S], snapshot))

    def check_clips(self, now: float, cut_short: bool = False) -> None:
        """Each alarm whose clip's last moment is past (`cut_short`: all, as the recorder stops): its clip, the moments
        up to CLIP_AFTER_S after the alarm, then its part before, queued to be written."""
        for clip in [c for c in self.pending_clips if cut_short or now >= c.seen + CLIP_AFTER_S]:
            self.pending_clips.remove(clip)
            items = [it for it in clip.items if it.t <= clip.seen + CLIP_AFTER_S]
            self.jobs.put(lambda clip=clip, items=items: self._save_clip(clip, items, cut_short))
            self.jobs.put(lambda clip=clip: self._save_before(clip))

    def read_inbox(self) -> None:
        """Take each request in the inbox (renamed <id>.taken) and queue its save: the last `seconds` as the ring holds
        them now."""
        try:
            asked = sorted(self.inbox.glob("*.json"))
        except OSError:
            return
        for path in asked:
            taken = path.with_suffix(".taken")
            try:
                os.replace(path, taken)
            except OSError:  # (withdrawn meanwhile)
                continue
            try:
                ask = json.loads(taken.read_text(encoding="utf-8"))
                seconds = min(max(float(ask.get("seconds", SECONDS)), 1 / self.fps), self.ring.seconds)
                name = safe_name(ask.get("name"))
            except (OSError, ValueError, TypeError, AttributeError) as e:
                self._answer(path.stem, taken, {"error": f"the request couldn't be read ({type(e).__name__}: {e})"})
                continue
            if self.jobs.unfinished_tasks >= SAVES_WAITING:
                self._answer(path.stem, taken, {"error": f"busy: {SAVES_WAITING} saves are waiting to be written"})
                continue
            snapshot = self.ring.last(seconds)
            self.jobs.put(lambda rid=path.stem, taken=taken, snapshot=snapshot, seconds=seconds, name=name:
                          self._save_recent(rid, taken, snapshot, seconds, name))

    # --- saving (the saver's thread) ---

    def _save_loop(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                job()
            except Exception as e:
                self._say(f"recorder: a save failed ({type(e).__name__}: {e})")
            finally:
                self.jobs.task_done()

    def _save_recent(self, rid: str, taken: Path, snapshot: list[Item], seconds: float, name: str) -> None:
        try:
            gone = lie_recorder.prune(self.recent, self.recent_keep)
            if gone:
                self._say(f"recorder: deleted the {len(gone)} oldest saves to keep the rest under "
                          f"{self.recent_keep / 1024 ** 3:g} GB")
            folder = new_folder(self.recent, name)
            summary = write_part(folder, snapshot, {"kind": "save-recent", "name": name or None,
                                                    "asked_s": seconds}, self.fps)
            answer = {"folder": str(folder),
                      **{k: summary[k] for k in ("seconds", "frames", "bytes", "gaps", "window")}}
            self._saved(folder, summary)
        except Exception as e:
            answer = {"error": f"{type(e).__name__}: {e}"}
            self._say(f"recorder: save-recent failed ({answer['error']})")
        self._answer(rid, taken, answer)

    def _save_before(self, clip: Clip) -> None:
        info, seen, snapshot = clip.info, clip.seen, clip.before
        folder = self.lie_folder(seen)
        since = snapshot[-1].t if snapshot else seen - self.ring.seconds
        items = snapshot + [it for it in clip.items if since < it.t <= seen + BEFORE_TAIL_S]
        if folder is not None:
            dest = folder / BEFORE
            self._prune_lie(folder)
        else:
            lie_recorder.prune(self.recent, self.recent_keep)
            dest = new_folder(self.recent, safe_name(f"lie-detector {info.get('form') or ''}".strip()))
        summary = write_part(dest, items, {
            "kind": "before the Lie Detector", "alarm": info, "alarm_at": day_time(seen),
            "alarm_t": None if not items else round(seen - items[0].t, 3),
            "player_recording": None if folder is None else str(folder)}, self.fps)
        self._saved(dest, summary)
        if folder is None:
            self._say(f"recorder: no recording of the player's began within {self.folder_wait:g} s of the alarm "
                      f"(the player not running?): the part before it is in {dest}")

    def _save_clip(self, clip: Clip, items: list[Item], cut_short: bool) -> None:
        gone = lie_recorder.prune(self.clips, self.clips_keep)
        if gone:
            self._say(f"recorder: deleted the {len(gone)} oldest alarm clips to keep the rest under "
                      f"{self.clips_keep / 1024 ** 3:g} GB")
        dest = new_folder(self.clips, safe_name(f"lie-detector {clip.info.get('form') or ''}".strip()))
        summary = write_part(dest, items, {
            "kind": "the alarm's clip", "alarm": clip.info, "alarm_at": day_time(clip.seen),
            "alarm_t": None if not items else round(clip.seen - items[0].t, 3),
            "asked": [-CLIP_BEFORE_S, CLIP_AFTER_S], "cut_short": cut_short}, self.fps, png=True)
        self._saved(dest, summary)

    def _prune_lie(self, spare: Path) -> None:
        """The Lie Detector's recordings kept to `lie_keep` (lie_recorder.KEEP_BYTES), their parts before counted, the
        one under way (`spare`) left out, before a part goes into it. The player prunes them too, but only as a
        recording starts, and a player from before 2026-10-24 counts no part before (its size_of counts a folder's
        files, not before/): the 10 GB would then hold some 58 cycles of 0.8 GB, 37 GB (32 GB were free on D: that
        evening). The player runs elevated and is restarted only with the user's click, so it can run old code for days;
        pruned here, the 10 GB hold whichever player wrote the cycles."""
        try:
            gone = lie_recorder.prune(self.lie_root, self.lie_keep, spare=spare)
        except Exception as e:
            self._say(f"recorder: pruning the Lie Detector's recordings failed ({type(e).__name__}: {e})")
            return
        if gone:
            self._say(f"recorder: deleted the {len(gone)} oldest Lie Detector recordings "
                      f"({sum(b for _, b in gone) / 1024 ** 2:.0f} MB) to keep the rest under "
                      f"{self.lie_keep / 1024 ** 3:g} GB")

    def lie_folder(self, seen: float) -> Path | None:
        """The player's recording of the Lie Detector seen at `seen` (time.time's clock): the newest folder in
        lie_root named for no more than FOLDER_SLACK_S before it, with no part before it yet; waited for until
        `folder_wait` s after the alarm (written a minute after it, the part waits no more). None if none comes."""
        deadline = time.monotonic() + max(0.0, seen + self.folder_wait - self.clock())
        while True:
            for folder in reversed(lie_recorder.recordings(self.lie_root)):
                made = stamp_time(folder.name)
                if made is None:
                    continue
                if made < seen - FOLDER_SLACK_S:
                    break
                if not (folder / BEFORE).exists():
                    return folder
            if time.monotonic() >= deadline or self._stop.is_set():
                return None
            time.sleep(0.2)

    def _saved(self, folder: Path, summary: dict) -> None:
        self.saves += 1
        self.last_save = str(folder)
        gaps = ", ".join(f"{why} {s:g} s" for why, s in summary["gaps"].items())
        self._say(f"recorder: saved {summary['seconds']:g} s ({summary['kind']}): {summary['frames']} frames "
                  f"({summary['bytes'] / 1024 ** 2:.1f} MB)" + (f", gaps: {gaps}" if gaps else "") + f": {folder}")

    def _answer(self, rid: str, taken: Path, answer: dict) -> None:
        try:
            tmp = self.outbox / f"{rid}.tmp"
            tmp.write_text(json.dumps(answer, indent=1), encoding="utf-8")
            os.replace(tmp, self.outbox / f"{rid}.json")
        except OSError as e:
            self._say(f"recorder: couldn't answer request {rid} ({e})")
        taken.unlink(missing_ok=True)

    def flush(self, timeout: float = 60.0) -> bool:
        """Wait for the saves queued to be written; whether they are."""
        deadline = time.monotonic() + timeout
        while self.jobs.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def close(self, timeout: float = 120.0) -> None:
        """Write what's queued (a save under way finishes) and the alarms' clips not over yet, cut short; say it
        stopped in the status file, let the grabber go."""
        self.check_clips(self.clock(), cut_short=True)
        self.flush(timeout)
        self._stop.set()
        self.jobs.put(None)
        self._saver.join(timeout=5)
        self.write_status(stopped=True)
        if hasattr(self.grab, "close"):
            self.grab.close()
        self._say("recorder: stopped; " + self.summary())

    # --- what it says ---

    def status(self, stopped: bool = False) -> dict:
        """What the status file says: whether the game shows, what the ring holds, the rate and cost since the last
        status (frames a second; ms a frame from the first look to the JPEG; the process's CPU a frame, and its share
        of a core, everything counted), the saves."""
        now = self.clock()
        cpu, wall = time.process_time(), time.monotonic()
        if wall - self._wall >= 1.0 or (stopped and wall > self._wall):  # (else the last's rate)
            frames, d_cpu, d_wall = len(self._costs), cpu - self._cpu, wall - self._wall
            self._rate = {"fps": round(frames / d_wall, 2),
                          "ms_a_frame": round(1000 * sum(self._costs) / frames, 2) if frames else None,
                          "cpu_ms_a_frame": round(1000 * d_cpu / frames, 2) if frames else None,
                          "core_share": round(d_cpu / d_wall, 4)}
            self._costs, self._cpu, self._wall = [], cpu, wall
        return {"pid": os.getpid(), "started": day_time(self.started), "updated": now, "updated_at": day_time(now),
                "stopped": stopped, "showing": self.state == "", "why": self.state,
                "kept_s": round(self.ring.span(), 1), "frames": self.ring.frames, "bytes": self.ring.bytes,
                "trimmed": self.ring.trimmed,
                "gaps_s": {k: round(v / self.fps, 1) for k, v in self.ring.gaps().items()},
                **self._rate, "saves": self.saves, "last_save": self.last_save,
                "max_bytes": self.ring.max_bytes, "seconds": self.ring.seconds, "fps_asked": self.fps}

    def write_status(self, stopped: bool = False) -> None:
        """The status file, put in place whole. Windows refuses to replace a file another program has open, as
        `recorder --status` has it for a moment (seen on 2026-10-24: the status saying it stopped was lost so): tried
        again STATUS_TRIES times STATUS_RETRY_S apart."""
        tmp = self.status_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self.status(stopped), indent=1), encoding="utf-8")
            for n in range(STATUS_TRIES):
                try:
                    os.replace(tmp, self.status_path)
                    return
                except PermissionError:
                    if n == STATUS_TRIES - 1:
                        raise
                    time.sleep(STATUS_RETRY_S)
        except OSError as e:  # (the disk full: the recording goes on)
            self._say(f"recorder: couldn't write its status ({e})")

    def summary(self) -> str:
        r = self._rate
        rate = "" if r["fps"] is None else (
            f"; {r['fps']:.1f} frames a second" + (f", {r['ms_a_frame']:.1f} ms a frame, CPU {r['cpu_ms_a_frame']:.1f} "
                                                   f"ms a frame" if r["ms_a_frame"] else "")
            + f", {r['core_share']:.3f} of a core")
        gaps = ", ".join(f"{k} {v / self.fps:.0f} s" for k, v in self.ring.gaps().items())
        return (f"kept {self.ring.span():.0f} s: {self.ring.frames} frames ({self.ring.bytes / 1024 ** 2:.0f} MB)"
                + (f", gaps: {gaps}" if gaps else "") + rate + f"; saves {self.saves}")

    def _say(self, msg: str) -> None:
        try:
            self.log(f"{time.strftime('%H:%M:%S')} {msg}")
        except Exception:
            pass  # a log that can't be written stops nothing


# --- a saved part ---------------------------------------------------------------------------------------------------

def write_part(folder: Path, items: list[Item], info: dict, fps: float = FPS, png: bool = False) -> dict:
    """Write moments into `folder` (made here; raises if it's there already): the frames, as JPEG (or `png`), on
    ENCODERS threads a batch at a time, index.jsonl and summary.json (`info`'s fields first, then the counts: the
    module's notes). Returns the summary."""
    folder.mkdir(parents=True)
    t0 = items[0].t if items else 0.0
    lines, gaps, run = [], Counter(), None
    shots = []  # the frames, in order
    shape = None
    for it in items:
        t = round(it.t - t0, 3)
        if it.data is None:
            gaps[it.why] += 1
            if run is not None and run["gap"] == it.why:
                run["until"], run["ticks"] = t, run["ticks"] + 1
            else:
                run = {"gap": it.why, "t": t, "until": t, "ticks": 1}
                lines.append(run)
            continue
        run = None
        name = f"{len(shots):05d}.{'png' if png else 'jpg'}"
        lines.append({"i": len(shots), "file": name, "t": t, "at": clock_time(it.t)})
        shots.append((name, it))
        shape = it.size
    size = 0
    with ThreadPoolExecutor(ENCODERS) as pool:
        for k in range(0, len(shots), 4 * ENCODERS):  # (a batch at a time: the frames' pixels aren't all out at once)
            batch = shots[k:k + 4 * ENCODERS]
            for (name, _), data in zip(batch, pool.map(lambda shot: encode(shot[1].pixels(), png), batch)):
                (folder / name).write_bytes(data)
                size += len(data)
    (folder / INDEX).write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    summary = {**info, "from": day_time(t0) if items else None, "to": day_time(items[-1].t) if items else None,
               "seconds": round(items[-1].t - t0 + 1 / fps, 1) if items else 0.0, "ticks": len(items),
               "frames": len(shots), "bytes": size, "gaps": {why: round(n / fps, 1) for why, n in gaps.items()},
               "fps": fps, "format": "png" if png else "jpeg", "jpeg_quality": None if png else lie_recorder.JPEG_QUALITY,
               "window": None if shape is None else list(shape)}
    (folder / SUMMARY).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def new_folder(root: Path, name: str = "") -> Path:
    """A folder in `root` that isn't there yet, named for now, then `name` (a number after it if one is named so
    already). write_part makes it (the saves are written one at a time)."""
    base = time.strftime(STAMP) + (f"-{name}" if name else "")
    for n in range(1, 100):
        folder = root / (base if n == 1 else f"{base}-{n}")
        if not folder.exists():
            return folder
    raise OSError(f"no free folder name for {base} in {root}")


def safe_name(name) -> str:
    """A request's name as a folder name's end: letters, digits, - and _ (others become -), 40 at most."""
    return "" if not name else NAME_CHARS.sub("-", str(name)).strip("-")[:40]


def stamp_time(name: str) -> float | None:
    """When a folder named STAMP[-...] was named (time.time's clock), or None."""
    try:
        return time.mktime(time.strptime(name[:15], STAMP))
    except ValueError:
        return None


def clock_time(t: float) -> str:
    """HH:MM:SS.mmm."""
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"


def day_time(t: float) -> str:
    """YYYY-mm-dd HH:MM:SS.mmm."""
    return time.strftime("%Y-%m-%d ", time.localtime(t)) + clock_time(t)


# --- the command line's side: asking the running recorder ------------------------------------------------------------

class NotRunning(RuntimeError):
    """No recorder is running, or it isn't reading its inbox."""


def read_status(box: Path | None = None) -> dict | None:
    """The recorder's status file, or None if there's none (or it can't be read)."""
    try:
        return json.loads((Path(box or paths.RECORDER) / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def running(box: Path | None = None, now: float | None = None) -> dict | None:
    """The running recorder's status: None if its file is missing, says it stopped, or is older than STATUS_STALE_S."""
    st = read_status(box)
    now = time.time() if now is None else now
    if st is None or st.get("stopped") or now - st.get("updated", 0) > STATUS_STALE_S:
        return None
    return st


def ask(seconds: float = SECONDS, name: str | None = None, box: Path | None = None,
        taken_s: float = ASK_TAKEN_S, wait_s: float = ASK_WAIT_S) -> dict:
    """Ask the running recorder to save its last `seconds` (the folder named for now and `name`), and wait for its
    answer: the `folder`, its `seconds`, `frames`, `bytes`, `gaps` and `window`; or `error`. Raises NotRunning if no
    recorder runs or none takes the request within `taken_s` (the request withdrawn)."""
    box = Path(box or paths.RECORDER)
    if running(box) is None:
        raise NotRunning(f"no recorder is running (its status file, {box / 'status.json'}, is missing, says it stopped "
                         f"or is older than {STATUS_STALE_S:g} s): start it with `python -m gamelab recorder` from a "
                         "checkout whose gamelab/lab is the player's (the babysitter keeps it running)")
    inbox, outbox = box / "inbox", box / "outbox"
    inbox.mkdir(parents=True, exist_ok=True)
    rid = f"{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
    request, answer = inbox / f"{rid}.json", outbox / f"{rid}.json"
    tmp = inbox / f"{rid}.tmp"
    tmp.write_text(json.dumps({"seconds": seconds, "name": name, "asked": time.time()}), encoding="utf-8")
    os.replace(tmp, request)
    start = time.monotonic()
    while True:
        if answer.exists():
            try:
                out = json.loads(answer.read_text(encoding="utf-8"))
            except (OSError, ValueError):  # (written as it was read: it's put there whole, by a rename)
                time.sleep(0.05)
                continue
            answer.unlink(missing_ok=True)
            return out
        waited = time.monotonic() - start
        if waited > taken_s and request.exists():
            try:
                request.unlink()
            except FileNotFoundError:  # taken just now
                continue
            raise NotRunning(f"the recorder didn't take the request within {taken_s:g} s (stopped, or stuck?): see "
                             f"`python -m gamelab recorder --status`")
        if waited > wait_s:
            raise RuntimeError(f"the recorder took the request but hasn't answered within {wait_s:g} s: its log says "
                               "why (the save may still be written)")
        time.sleep(0.05)


def status_lines(box: Path | None = None, now: float | None = None) -> list[str]:
    """What `recorder --status` prints."""
    st = read_status(box)
    if st is None:
        return [f"no recorder status in {Path(box or paths.RECORDER)}: it hasn't run here"]
    now = time.time() if now is None else now
    age = now - st.get("updated", 0)
    head = ("stopped" if st.get("stopped") else f"not running (its status is {age:.0f} s old)"
            if age > STATUS_STALE_S else "running")
    gaps = ", ".join(f"{k} {v:g} s" for k, v in st.get("gaps_s", {}).items()) or "none"
    lines = [f"recorder {head}: pid {st.get('pid')}, started {st.get('started')}, status of {st.get('updated_at')}",
             f"  the game {'on screen: recording' if st.get('showing') else 'not recording: ' + str(st.get('why'))}",
             f"  kept {st.get('kept_s', 0):g} s of the last {st.get('seconds', SECONDS):g}: {st.get('frames', 0)} "
             f"frames ({st.get('bytes', 0) / 1024 ** 2:.0f} MB of {st.get('max_bytes', MAX_BYTES) / 1024 ** 2:.0f}); "
             f"gaps: {gaps}" + (f"; {st['trimmed']} frames gone early for the memory cap" if st.get("trimmed") else "")]
    if st.get("fps") is not None:
        lines.append(f"  {st['fps']:g} frames a second; " + (
            f"{st['ms_a_frame']:g} ms a frame (look, grab, look, JPEG), CPU {st['cpu_ms_a_frame']:g} ms a frame, "
            if st.get("ms_a_frame") else "") + f"{st['core_share']:.3f} of a core")
    lines.append(f"  saves: {st.get('saves', 0)}" + (f", the last {st['last_save']}" if st.get("last_save") else ""))
    return lines


# --- running it -----------------------------------------------------------------------------------------------------

def only_one(name: str = MUTEX):
    """A named mutex held for as long as this process runs, so only one recorder runs on the PC: its handle, or None if
    another recorder holds it."""
    if sys.platform != "win32":
        return True
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = wintypes.HANDLE
    k32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = k32.CreateMutexW(None, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        k32.CloseHandle(handle)
        return None
    return handle


def quiet() -> None:
    """This process at below-normal priority (the game, the hunt and its feed come first) and OpenCV on one thread, as
    the feed's child runs (localization.feed.quiet)."""
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        k.SetPriorityClass(k.GetCurrentProcess(), BELOW_NORMAL)
    cv2.setNumThreads(1)
