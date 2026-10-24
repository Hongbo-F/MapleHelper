"""Recording the Lie Detector's whole cycle, for whoever answers it: from the moment the player's watchdog first sees
it until it has been off the screen for END_S, every picture the watchdog grabs (4 a second) is saved, with an index
and, at the end, a summary. The user asked for it on 2026-10-24: the player used to shut down at the alarm, so
nothing it saved showed them answering the Lie Detector.

One folder a cycle, paths.LIE_RECORDINGS/<YYYYmmdd-HHMMSS> (when it was first seen):
  00000.jpg ...  every frame grabbed: the game's client area as the watchdog grabbed it, as JPEG (JPEG_PARAMS);
  index.jsonl    a line a grab: `t` (s since the first sight) and either the frame's number `i`, its `file`, the
                 spotter's per-form `scores` (match / threshold, 1 = seen) and whether the Lie Detector was `seen`,
                 or `game`: why there is no frame ("no window", "minimized"; or "covered": another window over the
                 game, with the `scores` and `seen` of what was on top, its picture not saved);
  summary.json   at the end (`Recording.summary`): when it was first and last seen, how long the recording ran, its
                 frames, the forms seen and when each first and last showed, whether the Truth Booth did (the
                 outcome as far as the picture tells), how it ended, the alarm's screenshot and the window's size.

It writes only pictures of the game's window that the watchdog grabbed anyway: nothing of the mouse or the keyboard.
It never touches the game, and nothing here can answer the Lie Detector.

The writing is done on a thread of its own, behind a queue: a slow or full disk never holds up the watchdog, and a
frame that can't be written is logged, counted and left out while the recording goes on. When a recording starts,
the oldest others are deleted until they take at most KEEP_BYTES (`prune`, on a thread of its own; never the one
being written).

The elevated key player loads this module: it imports nothing else of gamelab but paths.
"""

import json
import os
import queue
import shutil
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from ..paths import LIE_RECORDINGS

# The recording ends once the game has shown no Lie Detector for this long (frames the game window was gone or
# minimized don't count: it may have been up behind it), and the player takes plans again only after as long
# (control.player.Watchdog). The spotter sees the mouse mini-game's notice only: once it closes, the game's "5 ...
# START" count and the figure tracking itself (moving the pointer over a fading shape) show nothing it knows, and
# they are what the person answering does. So it has to outlast them, as well as the moment between the countdown
# window ("N sec until the Lie Detector mini-game starts") and the Violettas' window (both have the yellow title) and
# a frame or two missed in a flicker (a window redrawn, the pointer over the title). How long the tracking lasts
# isn't measured (no recording of it yet: the first one's index will tell, the notice's last `seen` and the frames
# after it); a minute is several times the count's 5 s and a tracking of tens of seconds, at 240 frames of tail.
END_S = 60.0
# ... and at the latest this long after the first sight, seen or not: the longest cycle the game makes, as far as
# it has been seen, is a test left unanswered (its windows run a minute or two) and the Truth Booth it sends her to
# (4:44 left on its timer in the screenshot taken as she arrived, 2026-10-18), then END_S: about 8 minutes. Ten
# leave two to spare; a cycle that runs on (a window the spotter takes for the Lie Detector, the game minimized
# for good) stops recording there, at 2,400 frames (about 700 MB), and the player stays locked as before.
CAP_S = 10 * 60.0
# JPEG at quality 75 with full-resolution colour (4:4:4). Measured on the three real 1366x768 frames in
# tests/fixtures/native and the six Lie Detector screens in tests/fixtures/lie_detector shrunk to 1366x768
# (D:/MapleHelper-scratch/ld/jpeg_measure2.py): 297 KB a game frame (197 KB a Lie Detector screen), 4.9 ms to
# encode. With the usual half-resolution colour (4:2:0) the countdown's thin yellow title lost its colour: the
# spotter scored it 0.36-0.41 of its threshold on the decoded frame at every quality from 60 to 90; at 4:4:4 it
# keeps 1.11 (1.22 on the PNG); every other form scored as on the PNG. The mini-game notice's small text read as
# well as on the PNG, the HUD's 1366x768 text too (30.5 dB PSNR in the notice's box). Quality 85 is 434 KB a frame
# for 2.4 dB more; 65 was as legible at 238 KB, but its ringing showed round the letters. A PNG is 1,307 KB.
JPEG_QUALITY = 75
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
               cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
# What's kept: the newest recordings, 10 GB of them at most besides the one being written (the oldest deleted first
# when a recording starts). Telemetry keeps 2 GB, 4 days of `watch`; these are kept longer, being what the user
# asked for. A cycle answered in about 2.5 minutes (the test's windows, then the minute's tail) is some 600 frames,
# about 175 MB at 290 KB a frame, so 10 GB holds about 58 of those; 18 that went to the Truth Booth (8 minutes,
# 560 MB), 14 at the cap. The Lie Detector came about an hour apart at the most often (00:47, 01:58 and 02:54 on
# 2026-10-19: tests/fixtures/lie_detector), so 58 cycles are two days of play without a break, weeks of evenings.
# D: had 98 GB free on 2026-10-24.
KEEP_BYTES = 10 * 1024 ** 3
QUEUE_ITEMS = 16  # grabs waiting to be written (4 s of them, 3 MB a 1366x768 frame); past it a frame is dropped
LIVE_S = 5.0      # a recording whose index was written this recently, and not summed up, is still being written
STAMP = "%Y%m%d-%H%M%S"
INDEX, SUMMARY = "index.jsonl", "summary.json"
TRUTH_BOOTH = "truth booth"  # the spotter's name for the Truth Booth's form (perception.lie_detector.HALF_SIZE)


class Recording:
    """One cycle's recording. The watchdog adds every grab (`add`) and ends it (`finish`); a thread of its own writes
    the frames and the index as they come and the summary at the end. Making one makes its folder (OSError if it
    can't) and deletes the oldest recordings past `keep_bytes`."""

    def __init__(self, root: Path | None = None, alarm_info: dict | None = None, log=print,
                 keep_bytes: int = KEEP_BYTES, end_s: float = END_S, cap_s: float = CAP_S):
        self.root = _root(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.folder = _new_folder(self.root)
        self.started = time.strftime("%Y-%m-%d %H:%M:%S")
        self.screenshot = (alarm_info or {}).get("screenshot") if isinstance(alarm_info, dict) else None
        self.log, self.end_s, self.cap_s = log, end_s, cap_s
        # The watchdog's side (only it adds), read by the writer once the recording is ended.
        self.grabs = self.numbered = self.dropped = self._dropped_since = 0
        self.missing: Counter = Counter()  # grabs with no frame, by why
        self.forms: dict[str, dict] = {}   # form: first, last, frames
        self.first_seen = self.last_seen = None
        self.last_t = 0.0
        self.window = None
        self.end = None
        # The writer's side.
        self.frames = self.bytes = self.errors = self.index_errors = 0
        self.ended = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_ITEMS)
        self._writer = threading.Thread(target=self._write, name="lie-detector-recording", daemon=True)
        self._writer.start()
        threading.Thread(target=self._prune, args=(keep_bytes,), name="lie-detector-prune", daemon=True).start()

    def add(self, t: float, frame: np.ndarray | None, scores: dict[str, float] | None = None,
            why: str | None = None) -> None:
        """One grab, `t` s after the first sight: a frame (RGB) with the spotter's scores of it, or None and why
        there is none (the game window gone or minimized; or covered by another window, with the scores of what was
        on top, which aren't saved as a picture). Never blocks: with the queue full the frame is dropped (counted,
        and the next line in the index says how many went before it)."""
        record: dict = {"t": round(t, 3)}
        if frame is None:
            record["game"] = why or "no window"
            self.missing[record["game"]] += 1
        else:
            scores = scores or {}
            record["i"] = self.numbered
            self.numbered += 1
            if self.window is None:
                self.window = [int(frame.shape[1]), int(frame.shape[0])]
        if scores is not None:
            seen = bool(scores) and max(scores.values()) >= 1
            record.update(seen=seen, scores={k: round(float(v), 3) for k, v in scores.items()})
            for name, v in scores.items():
                if v >= 1:
                    form = self.forms.setdefault(name, {"first": record["t"], "last": record["t"], "frames": 0})
                    form["last"] = record["t"]
                    form["frames"] += 1
            if seen:
                self.first_seen = record["t"] if self.first_seen is None else self.first_seen
                self.last_seen = record["t"]
        self.grabs += 1
        self.last_t = t
        if self._dropped_since:
            record["dropped_before"] = self._dropped_since
        try:
            self._queue.put_nowait((record, frame))
            self._dropped_since = 0
        except queue.Full:  # the disk is behind: this grab goes, the watchdog doesn't wait
            self.dropped += 1
            self._dropped_since += 1
            if self.dropped == 1:
                self._say(f"Lie Detector recording: the disk is behind, a frame dropped at {t:.1f} s (counted)")

    def finish(self, end: str) -> None:
        """End it ('gone': off the screen for END_S, 'cap': CAP_S after the first sight, 'stopped': the player
        stopped): the writer writes what's waiting, then the summary, and logs the end. Returns at once."""
        self.end = end
        self.ended.set()

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the writer to be done (after `finish`); whether it is."""
        self._writer.join(timeout)
        return not self._writer.is_alive()

    def summary(self) -> dict:
        """What summary.json says. The outcome is as far as the picture tells: "sent to the Truth Booth" if it
        showed, "answered" if the Lie Detector went away without it, else "unknown" (still up when the recording
        ended at the cap or as the player stopped)."""
        booth = TRUTH_BOOTH in self.forms
        outcome = ("sent to the Truth Booth" if booth else "answered" if self.end == "gone" else
                   "unknown: still up when the recording ended")
        seen_s = None if self.first_seen is None else round(self.last_seen - self.first_seen, 3)
        return {"folder": str(self.folder), "started": self.started, "first_seen": self.first_seen,
                "last_seen": self.last_seen, "seen_s": seen_s, "duration_s": round(self.last_t, 3),
                "frames": self.frames, "frame_bytes": self.bytes, "grabs": self.grabs,
                "not_showing": dict(self.missing), "dropped": self.dropped, "write_errors": self.errors,
                "forms": self.forms, "truth_booth": booth, "outcome": outcome, "end": self.end,
                "end_stretch_s": self.end_s, "cap_s": self.cap_s, "alarm_screenshot": self.screenshot,
                "window": self.window, "jpeg_quality": JPEG_QUALITY}

    # --- the writer's thread ---

    def _write(self) -> None:
        while True:
            # Whether it was ended, read before the queue is: the watchdog adds a grab and then ends it in the same
            # look, so a grab added between an empty queue and a look at `ended` was left unwritten and uncounted.
            # Read first, an end seen means every grab is in the queue already.
            done = self.ended.is_set()
            try:
                record, frame = self._queue.get(timeout=0.05)
            except queue.Empty:
                if done:  # nothing is added once it's ended: all of it is written
                    break
                continue
            self._write_one(record, frame)
        summary = self.summary()
        try:
            (self.folder / SUMMARY).write_text(json.dumps(summary, indent=1), encoding="utf-8")
        except OSError as e:
            self._say(f"Lie Detector recording: couldn't write its summary ({e})")
        forms = ", ".join(self.forms) or "none"
        self._say(f"Lie Detector recording ended ({self.end}): {self.frames} frames over {self.last_t:.0f} s, seen "
                  f"{summary['seen_s'] or 0:.0f} s ({forms}), {summary['outcome']}"
                  + (f", {self.errors} frames not written" if self.errors else "")
                  + (f", {self.dropped} dropped" if self.dropped else "") + f": {self.folder}")

    def _write_one(self, record: dict, frame: np.ndarray | None) -> None:
        if frame is not None:
            path = self.folder / f"{record['i']:05d}.jpg"
            try:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), JPEG_PARAMS)
                if not ok:
                    raise ValueError("the JPEG encoder refused it")
                path.write_bytes(buf.tobytes())
                record["file"] = path.name
                self.frames += 1
                self.bytes += len(buf)
            except Exception as e:  # disk full, no permission, ...: logged once, counted, and on to the next
                self.errors += 1
                record["error"] = f"{type(e).__name__}: {e}"
                if self.errors == 1:
                    self._say(f"Lie Detector recording: couldn't write frame {record['i']} ({record['error']}); "
                              "going on (the summary counts them)")
                try:
                    path.unlink(missing_ok=True)  # no half-written picture
                except OSError:
                    pass
        try:
            with open(self.folder / INDEX, "a", encoding="utf-8") as f:  # closed each time: its last write shows
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            self.index_errors += 1
            if self.index_errors == 1:
                self._say(f"Lie Detector recording: couldn't write its index ({e}); going on")

    def _prune(self, keep_bytes: int) -> None:
        try:
            gone = prune(self.root, keep_bytes, spare=self.folder)
        except Exception as e:
            self._say(f"Lie Detector recordings: pruning failed ({e})")
            return
        if gone:
            self._say(f"Lie Detector recordings: deleted the {len(gone)} oldest "
                      f"({sum(b for _, b in gone) / 1024 ** 2:.0f} MB) to keep the rest under "
                      f"{keep_bytes / 1024 ** 3:g} GB")

    def _say(self, msg: str) -> None:
        try:
            self.log(msg)
        except Exception:
            pass  # a log that can't be written (the disk full) stops nothing


def _new_folder(root: Path) -> Path:
    """A new folder in `root` named for now (a suffix if one is named for this second already)."""
    stamp = time.strftime(STAMP)
    for n in range(1, 100):
        folder = root / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            continue
    raise OSError(f"no free folder name for {stamp} in {root}")


def _root(root: Path | None) -> Path:
    """`root`, or by default LIE_RECORDINGS (looked up when called)."""
    return LIE_RECORDINGS if root is None else Path(root)


def recordings(root: Path | None = None) -> list[Path]:
    """The recordings in `root` (LIE_RECORDINGS), oldest first (by name: the time each was first seen)."""
    try:
        return sorted(p for p in _root(root).iterdir() if p.is_dir())
    except OSError:  # no folder yet
        return []


def size_of(folder: Path) -> int:
    """The bytes of the files in a recording's folder."""
    total = 0
    try:
        with os.scandir(folder) as it:
            for entry in it:
                try:
                    total += entry.stat().st_size if entry.is_file() else 0
                except OSError:
                    pass
    except OSError:
        pass
    return total


def prune(root: Path | None = None, keep_bytes: int = KEEP_BYTES, spare: Path | None = None,
          dry_run: bool = False) -> list[tuple[Path, int]]:
    """Delete the oldest recordings in `root` (LIE_RECORDINGS) until the rest (`spare`, the one being written, left
    out of the sum) take at most `keep_bytes`; never `spare`. Returns those deleted (folder, bytes), oldest first;
    with `dry_run`, those it would delete. One that can't be deleted wholly (a frame open in a viewer: Windows
    won't delete an open file) is left, what of it went taken off the sum, and the rest of it goes next time."""
    spare = None if spare is None else Path(spare).resolve()
    folders = [(f, size_of(f)) for f in recordings(root) if spare is None or f.resolve() != spare]
    total, gone = sum(b for _, b in folders), []
    for folder, size in folders:
        if total <= keep_bytes:
            break
        if not dry_run:
            shutil.rmtree(folder, ignore_errors=True)
            if folder.exists():
                # Counted whole, what was left of it had newer recordings deleted in its place: one 3,000-byte
                # folder with a 10-byte frame held open took two newer ones of 285 bytes under a 1,000-byte cap.
                total -= size - size_of(folder)
                continue
        total -= size
        gone.append((folder, size))
    return gone


def summary_of(folder: Path) -> dict | None:
    """A recording's summary, or None if it has none (still being written, or the player stopped mid-way)."""
    try:
        return json.loads((Path(folder) / SUMMARY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def going(root: Path | None = None, now: float | None = None) -> Path | None:
    """The recording still being written in `root` (LIE_RECORDINGS), if any: the newest, with no summary yet and its
    index written in the last LIVE_S (the watchdog adds a line every grab, 4 a second, even with the game minimized).
    One the player stopped in the middle of has no summary either, but its index is old."""
    folders = recordings(root)
    if not folders or (folders[-1] / SUMMARY).exists():
        return None
    try:
        written = os.stat(folders[-1] / INDEX).st_mtime
    except OSError:
        try:
            written = os.stat(folders[-1]).st_mtime  # just made: no line yet
        except OSError:  # gone meanwhile
            return None
    return folders[-1] if (time.time() if now is None else now) - written <= LIVE_S else None
