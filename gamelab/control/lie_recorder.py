"""Recording the Lie Detector's whole cycle, for whoever answers it: from the moment the player's watchdog first sees
it until it has been off the screen for END_S, or its test (any form but the Truth Booth) for TEST_TAIL_S, every look
the watchdog takes (4 a second) goes into an index and, at the end, a summary. The user asked for it on 2026-10-24:
the player used to shut down at the alarm, so nothing it saved showed them answering the Lie Detector. Its pictures
(JPEG, 4 a second) were saved too until 2026-11-01 (the user: "stop keeping the 4fps recording and only use the
unified recorder's 15fps video saving"): the cycle's video is the rolling recorder's alarm clip, 15 a second, lossless,
written once this recording's summary says its test is over (control.rolling: Recorder.cycle_end).

One folder a cycle, paths.LIE_RECORDINGS/<YYYYmmdd-HHMMSS> (when it was first seen):
  index.jsonl    a line a look: `t` (s since the first sight) and either the look's frame number `i` (a frame of the
                 game itself), the spotter's per-form `scores` (match / threshold, 1 = seen) and whether the Lie
                 Detector was `seen`, or `game`: why there was no frame of the game ("no window", "minimized"; or
                 "covered": another window over the game, with the `scores` and `seen` of what was on top);
  summary.json   at the end (`Recording.summary`): when it was first seen (`first_seen_at`, the clock's) and last
                 seen, its test last (`test_last`), how long the recording ran, its looks, the forms seen and when
                 each first and last showed, whether the Truth Booth did (the outcome as far as the picture tells),
                 its `cause` (UNKNOWN_CAUSE: the Truth Booth with no test seen before it; else None), when the
                 Truth Booth stay ends by its clock (`booth_until`, the clock's; None: not seen), how it ended, the
                 alarm's screenshot and the window's size;
  cycle.live     (LIVE) while the cycle goes on after a recording ended at its test's end (the Truth Booth up): the
                 watchdog touches it every look and deletes it at the cycle's end (`going`);
  before/        written by the rolling recorder when it runs (control.rolling): the game 15 times a second for the 5
                 minutes before the alarm and on into this recording's first seconds, frames, index and summary of
                 its own.

Nothing of the mouse or the keyboard is recorded. It never touches the game, and nothing here can answer the Lie
Detector.

The writing is done on a thread of its own, behind a queue: a slow or full disk never holds up the watchdog. When a
recording starts, the oldest others are deleted until they take at most KEEP_BYTES (`prune`, on a thread of its own;
never the one being written).

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

import numpy as np

from ..paths import LIE_RECORDINGS

# The recording ends once the game has shown no Lie Detector for this long (frames the game window was gone,
# minimized or covered don't count: it may have been up behind), and the player, its alarm cleared, stops only after
# as long (control.player.Watchdog). The spotter sees the mouse mini-game's notice only: once it closes, the game's
# "5 ... START" count and the figure tracking itself (moving the pointer over a fading shape) show nothing it knows, and
# they are what the person answering does. So it has to outlast them, as well as the moment between the countdown
# window ("N sec until the Lie Detector mini-game starts") and the Violettas' window (both have the yellow title) and
# a frame or two missed in a flicker (a window redrawn, the pointer over the title). How long the tracking lasts
# isn't measured (no recording of it yet: the first one's index will tell, the notice's last `seen` and the frames
# after it); a minute was several times the count's 5 s and a tracking of tens of seconds, at 240 frames of tail.
# 10 s since the user's "Change lie detector timeout from 60s to 10s" (2026-10-29 ~01:26: a person answers it and
# clears the alarm, and the minute kept the player locked and the grind waiting after the answer; a tracking stretch
# that shows nothing for longer may end the cycle early: then the recording stops there and `clear-alarm` may stop the
# player while it's still up, which only a person starting it again undoes).
END_S = 10.0
# The recording ends this long after its test's last sight (the board, the quiz, the countdown and the boxes before
# them: every form but the Truth Booth) even if the cycle goes on (the user, 2026-11-03: "Only include frames of up to
# 5s once the board disappears", the recording of 00:01 had run 68 s into the Truth Booth); the cycle itself still ends
# END_S after the last sight of any form (control.player.Watchdog), the Truth Booth's too, and its recording's folder
# says it's going till then (LIVE). A test that sends her to the Truth Booth does so at once (0.4 s after the board,
# 2026-11-03 00:02:19): the recording sees it; one seen only later is put in the summary at the cycle's end.
TEST_TAIL_S = 5.0
# ... and at the latest this long after the first sight, seen or not: the longest cycle the game makes, as far as
# it has been seen, is a test left unanswered (its windows run a minute or two) and the Truth Booth it sends her to
# (4:44 left on its timer in the screenshot taken as she arrived, 2026-10-18), then END_S: about 8 minutes. Ten
# leave two to spare; a cycle that runs on (a window the spotter takes for the Lie Detector, the game minimized
# for good) stops recording there, at 2,400 looks, and the player stays up, locked, as before.
CAP_S = 10 * 60.0
# What's kept: the newest recordings, 10 GB of them at most besides the one being written (the oldest deleted first
# when a recording starts). A recording's own index is small (about 0.2 MB a cycle since its frames went, 2026-11-01);
# what counts is the rolling recording's 5 minutes before each alarm in it (`before/`, about 650 MB at 1366x768:
# control.rolling), so about 15 cycles in the 10 GB. The Lie Detector came about an hour apart at the most often (00:47,
# 01:58 and 02:54 on 2026-10-19: tests/fixtures/lie_detector).
KEEP_BYTES = 10 * 1024 ** 3
QUEUE_ITEMS = 16  # looks waiting to be written to the index (4 s of them); past it one is dropped
LIVE_S = 5.0      # a recording whose index was written this recently, and not summed up, is still being written
STAMP = "%Y%m%d-%H%M%S"
INDEX, SUMMARY, LIVE = "index.jsonl", "summary.json", "cycle.live"
TRUTH_BOOTH = "truth booth"  # the spotter's name for the Truth Booth's form (perception.lie_detector.HALF_SIZE)
# A Truth Booth first seen with no test seen before it in its cycle, and not in a stay seen before (the user,
# 2026-11-03: "if we entered a truth booth without triggering any of the mini-game/challenge, that probably means we
# missed some kind of check, we should immediately save a 5min-clip and mark them unknown reason that led to truth booth
# for later analysis"): its alarm says so (control.alarm's `cause`, UNKNOWN_CAUSE), as does its recording's summary,
# and the rolling recorder saves the 5 minutes before it at once (control.rolling, paths.BOOTH_UNKNOWN). A test sends
# her to the Truth Booth at once (0.4 s after the board, 2026-11-03 00:02:19), in its cycle. A stay's end is read off
# the booth's clock as it's seen (perception.lie_detector's `clock_left`; the user: "When you enter the truth booth you
# can see how much count down it has, and block the re-sighting"), the watchdog's `booth_until`, in the summary too
# (`booth_until`, the player started again in the booth: `last_booth_until`): a sighting before it (BOOTH_SLACK_S on)
# is that stay's (2026-10-29: seen 09:26:37 to 09:27:31 and again from 09:29:53, a cycle of its own after 2 minutes off
# the screen, 7:04 left then: her second stay that day, 10 minutes; the first stay, 5). A stay whose clock was never
# read ends BOOTH_UNREAD_S after its first sight, the longest stay seen.
UNKNOWN_CAUSE = "unknown"
BOOTH_UNREAD_S = 10 * 60.0
BOOTH_SLACK_S = 5.0


class Recording:
    """One cycle's recording. The watchdog adds every look (`add`) and ends it (`finish`); a thread of its own writes
    the index as it comes and the summary at the end. Making one makes its folder (OSError if it
    can't) and deletes the oldest recordings past `keep_bytes`."""

    def __init__(self, root: Path | None = None, alarm_info: dict | None = None, log=print,
                 keep_bytes: int = KEEP_BYTES, end_s: float = END_S, cap_s: float = CAP_S, t0: float | None = None):
        self.root = _root(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.folder = _new_folder(self.root)
        self.started = time.strftime("%Y-%m-%d %H:%M:%S")
        self.t0 = time.time() if t0 is None else t0  # the first sight (t 0), the clock's: the alarm clip's end by it
        self.screenshot = (alarm_info or {}).get("screenshot") if isinstance(alarm_info, dict) else None
        self.cause = alarm_info.get("cause") if isinstance(alarm_info, dict) else None  # UNKNOWN_CAUSE: no test seen
        self.booth_until: float | None = None  # the Truth Booth stay's end (time.time's clock), set by the watchdog
        self.log, self.end_s, self.cap_s = log, end_s, cap_s
        # The watchdog's side (only it adds), read by the writer once the recording is ended.
        self.grabs = self.numbered = self.dropped = self._dropped_since = 0
        self.missing: Counter = Counter()  # grabs with no frame, by why
        self.forms: dict[str, dict] = {}   # form: first, last, frames
        self.first_seen = self.last_seen = self.test_last = None
        self.last_t = 0.0
        self.window = None
        self.end = None
        # The writer's side.
        self.index_errors = 0
        self.ended = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_ITEMS)
        self._writer = threading.Thread(target=self._write, name="lie-detector-recording", daemon=True)
        self._writer.start()
        threading.Thread(target=self._prune, args=(keep_bytes,), name="lie-detector-prune", daemon=True).start()

    def add(self, t: float, frame: np.ndarray | None, scores: dict[str, float] | None = None,
            why: str | None = None) -> int | None:
        """One look, `t` s after the first sight: a frame of the game (RGB: only its size is kept) with the spotter's
        scores of it, or None and why there is none (the game window gone or minimized; or covered by another window,
        with the scores of what was on top). Returns the frame's number (None: no frame). Never blocks: with the queue full the look's line is dropped (counted,
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
                    if name != TRUTH_BOOTH:
                        self.test_last = record["t"]
            if seen:
                self._seen_at(record["t"])
        self.grabs += 1
        self.last_t = t
        if self._dropped_since:
            record["dropped_before"] = self._dropped_since
        try:
            self._queue.put_nowait(record)
            self._dropped_since = 0
        except queue.Full:  # the disk is behind: this look's line goes, the watchdog doesn't wait
            self.dropped += 1
            self._dropped_since += 1
            if self.dropped == 1:
                self._say(f"Lie Detector recording: the disk is behind, a look dropped at {t:.1f} s (counted)")
        return record.get("i")

    def _seen_at(self, t: float) -> None:
        """The Lie Detector seen at `t` (s after the first sight): its first and last sight."""
        self.first_seen = t if self.first_seen is None else min(self.first_seen, t)
        self.last_seen = t if self.last_seen is None else max(self.last_seen, t)

    def finish(self, end: str) -> None:
        """End it ('gone': off the screen for END_S, 'test over': its test for TEST_TAIL_S, 'cap': CAP_S after the
        first sight, 'stopped': the player stopped): the writer writes what's waiting, then the summary, and logs the
        end. Returns at once."""
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
        outcome = ("sent to the Truth Booth" if booth else "answered" if self.end in ("gone", "test over") else
                   "unknown: still up when the recording ended")
        seen_s = None if self.first_seen is None else round(self.last_seen - self.first_seen, 3)
        return {"folder": str(self.folder), "started": self.started, "first_seen_at": round(self.t0, 3),
                "first_seen": self.first_seen, "last_seen": self.last_seen, "test_last": self.test_last,
                "seen_s": seen_s, "duration_s": round(self.last_t, 3),
                "grabs": self.grabs, "not_showing": dict(self.missing), "dropped": self.dropped,
                "forms": self.forms, "truth_booth": booth, "outcome": outcome, "cause": self.cause,
                "booth_until": None if self.booth_until is None else round(self.booth_until, 3), "end": self.end,
                "end_stretch_s": self.end_s, "cap_s": self.cap_s, "alarm_screenshot": self.screenshot,
                "window": self.window}

    # --- the writer's thread ---

    def _write(self) -> None:
        while True:
            # Whether it was ended, read before the queue is: the watchdog adds a grab and then ends it in the same
            # look, so a grab added between an empty queue and a look at `ended` was left unwritten and uncounted.
            # Read first, an end seen means every grab is in the queue already.
            done = self.ended.is_set()
            try:
                record = self._queue.get(timeout=0.05)
            except queue.Empty:
                if done:  # nothing is added once it's ended: all of it is written
                    break
                continue
            self._write_one(record)
        summary = self.summary()
        try:
            (self.folder / SUMMARY).write_text(json.dumps(summary, indent=1), encoding="utf-8")
        except OSError as e:
            self._say(f"Lie Detector recording: couldn't write its summary ({e})")
        forms = ", ".join(self.forms) or "none"
        self._say(f"Lie Detector recording ended ({self.end}): {self.grabs} looks over {self.last_t:.0f} s, seen "
                  f"{summary['seen_s'] or 0:.0f} s ({forms}), {summary['outcome']}"
                  + (f", {self.dropped} dropped" if self.dropped else "") + f": {self.folder}")

    def _write_one(self, record: dict) -> None:
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
    """The bytes of the files in a recording's folder and the folders in it (the rolling recording's part before the
    Lie Detector, `before/`: control.rolling)."""
    total = 0
    try:
        with os.scandir(folder) as it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        total += size_of(Path(entry.path))
                    elif entry.is_file():
                        total += entry.stat().st_size
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


def last_booth_until(root: Path | None = None) -> float | None:
    """When the Truth Booth stay of the newest recording in `root` (LIE_RECORDINGS) whose summary has one ends
    (`booth_until`, time.time's clock); None if none has."""
    for folder in reversed(recordings(root)):
        until = (summary_of(folder) or {}).get("booth_until")
        if isinstance(until, (int, float)):
            return float(until)
    return None


def summary_of(folder: Path) -> dict | None:
    """A recording's summary, or None if it has none (still being written, or the player stopped mid-way)."""
    try:
        return json.loads((Path(folder) / SUMMARY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def going(root: Path | None = None, now: float | None = None) -> Path | None:
    """The recording still being written in `root` (LIE_RECORDINGS), if any, or whose cycle is still going: the
    newest, with no summary yet and its index written in the last LIVE_S (the watchdog adds a line every grab, 4 a
    second, even with the game minimized), or summed up at its test's end with its LIVE file touched in the last LIVE_S
    (the cycle going on: the Truth Booth). One the player stopped in the middle of has an old index or LIVE file."""
    folders = recordings(root)
    if not folders:
        return None
    newest = folders[-1]
    try:
        if (newest / SUMMARY).exists():
            written = os.stat(newest / LIVE).st_mtime
        else:
            try:
                written = os.stat(newest / INDEX).st_mtime
            except OSError:
                written = os.stat(newest).st_mtime  # just made: no line yet
    except OSError:  # (no LIVE file: its cycle over; or gone meanwhile)
        return None
    return newest if (time.time() if now is None else now) - written <= LIVE_S else None


def cycle_going(folder: Path) -> None:
    """The cycle of a recording ended at its test's end still going: its LIVE file touched (made if need be)."""
    try:
        (Path(folder) / LIVE).touch()
    except OSError:
        pass  # (`going` then says it's over a little early: `clear-alarm` raises it again if it's still up)


def cycle_over(folder: Path, truth_booth: bool, booth_until: float | None = None) -> None:
    """The cycle of a recording ended at its test's end over: its LIVE file gone, and its summary told if the cycle
    showed the Truth Booth after the recording ended, and when its stay ends (`booth_until`, the watchdog's)."""
    folder = Path(folder)
    summary = summary_of(folder)
    new_end = booth_until is not None and summary is not None and summary.get("booth_until") != round(booth_until, 3)
    if truth_booth and summary is not None and (not summary.get("truth_booth") or new_end):
        if not summary.get("truth_booth"):
            summary.update(truth_booth=True, outcome="sent to the Truth Booth", truth_booth_after=True)
        if booth_until is not None:
            summary["booth_until"] = round(booth_until, 3)
        try:
            (folder / SUMMARY).write_text(json.dumps(summary, indent=1), encoding="utf-8")
        except OSError:
            pass
    try:
        (folder / LIVE).unlink(missing_ok=True)
    except OSError:
        pass
