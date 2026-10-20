"""Telemetry: what the stack sees, thinks and does, as topics in a JSON-lines file. The mini-map window
(gamelab.ui) follows the file live or replays it.

Producers call `emit(topic, **data)` with plain JSON data. Nothing is written until a recorder is
installed (`record_to`), so tests and library use pay nothing. Each line is
{"t": seconds, "topic": topic, **data}. `t` is wall-clock time (the simulator's fake clock in a
simulated run), so a replay keeps the original pace.

Files are named for the time a run started and what it is (`<%Y%m%d-%H%M%S>-<name>.jsonl`), so the
newest sorts last. A run goes on in a new part past MAX_BYTES: a file of its own, named for the time the
part starts (`...-<name>-part2.jsonl`, always later than the part before), that starts with the run's
session record marked `continues` (the part before's file name) and `part`. When a run starts, and each
time one goes on in a new part, the oldest files are deleted: those last written over KEEP_DAYS ago, then
more until the rest take at most KEEP_BYTES (`prune`; never the part being written). So a run left going
for days (`watch`) keeps the folder bounded too.

Topics (map units; producers build the records, ui.state folds them):

  session   map, command, mode (live/sim); continues, part     the command line, the simulator; a new part
  scene     her {x, y, platform, how, seen, facing}, hp, mp,   every look (behavior.hunter, after Senses)
            monsters [{kind, x, y, platform, bar}], loot [{kind, x, y, platform}]
  drops     pending [{x, y, platform, mesos}], inventory_full  every look (behavior.hunter)
  glance    x, y, hp                                           control.navigator, while a chain runs
  route     dst, moves [{kind, src, dst, x, land_x}]           control.navigator, each plan
  chain     why, text, legs [{kind, src, dst, spans, end}]     control.navigator, each key plan sent
  monsters  list [{platform, x, vx, heading, reach}]           control.navigator (avoiding monsters)
  trip      dst, result                                        control.navigator, at every trip's end
  stance    act, text, aimed [{kind, x, y}]                    behavior.fight, each look of a fight
  tour      order, safe, grounds / stop, spot                  behavior.tour (stop, spot: also behavior.hunt)
  hunt      mode, rule, objective, ground, piles [{x, platform,   behavior.hunt, when the rule changes
            status, hers}]
  event     kind (kill, hit, potion, rest), text               behavior
  log       text                                               every log line
"""

import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .paths import TELEMETRY

FOLDER = TELEMETRY  # one file per run, newest last (git-ignored, like all of lab/)
STAMP = "%Y%m%d-%H%M%S"  # file names start with the time the run (or the part) started
# A run's file is continued in a new part past this. Hunting wrote 4.2 MB an hour (72 KB a minute, from
# the live tours), so a part is hours of play; a continuous feed of 10 looks a second would write 10-15 MB
# an hour. Parts let the retention below delete a long run a piece at a time, and keep any one file small
# enough for a replay to load whole.
MAX_BYTES = 64 * 1024 * 1024
# What's kept when a run starts or goes on in a new part: files last written in the last 14 days, and of those
# at most 2 GB, the newest (38 files and 2.9 MB from the first two days of live runs; `watch` writes about
# 400 MB a day).
KEEP_DAYS = 14
KEEP_BYTES = 2 * 1024 ** 3

_sink: Callable[[dict], None] | None = None


def emit(topic: str, **data) -> None:
    """Publish one record on `topic` (a no-op unless a recorder is installed). Telemetry never stops a
    run: if a record can't be written (a full disk, a value JSON can't hold), recording stops with one
    warning and the run goes on."""
    sink = _sink
    if sink is None:
        return
    try:
        sink({"t": round(time.time(), 3), "topic": topic, **data})
    except Exception as e:
        stop()
        print(f"telemetry stopped: can't record {topic!r} ({e})", file=sys.stderr)


def recording() -> bool:
    """Whether a recorder is installed (callers skip building records nobody would see)."""
    return _sink is not None


class Recorder:
    """Appends records to a JSON-lines file, one flushed line each, so a reader can follow along. With
    `max_bytes`, a record that would take the file past it goes into a new part instead (see the top):
    `path` is then the part being written, and `part` its number."""

    def __init__(self, path: Path, max_bytes: int | None = None):
        self.path, self.max_bytes, self.part = path, max_bytes, 1
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        self._size = path.stat().st_size  # bytes in the file (lines are ASCII: json.dumps escapes the rest)
        self._lock = threading.Lock()
        self._session: dict | None = None  # the run's session record, to start each new part with
        named = re.fullmatch(r"(\d{8}-\d{6})-(.+)", path.stem)
        self._name = named.group(2) if named else path.stem
        self._started = int(time.mktime(time.strptime(named.group(1), STAMP))) if named else int(time.time())

    def __call__(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":"), default=_plain) + "\n"
        size = len(line) + len(os.linesep) - 1  # written in text mode: "\n" is os.linesep on disk
        with self._lock:
            if self._file.closed:
                return
            if record.get("topic") == "session" and "continues" not in record:
                self._session = record
            if self.max_bytes is not None and self._size and self._size + size > self.max_bytes:
                self._next_part(record.get("t"))
            self._file.write(line)
            self._file.flush()
            self._size += size

    def _next_part(self, t: float | None) -> None:
        """Go on in a new file, named for now but always after the part before (so it sorts after it,
        even within the same second), headed by the run's session record marked as its continuation."""
        started = max(int(time.time()), self._started + 1)
        part = self.part + 1
        path = self.path.with_name(f"{time.strftime(STAMP, time.localtime(started))}-{self._name}-part{part}.jsonl")
        new = path.open("a", encoding="utf-8")  # if it can't be made, emit stops recording (one warning)
        self._file.close()
        before, self._file = self.path, new
        self.path, self.part, self._started = path, part, started
        session = {k: v for k, v in (self._session or {"topic": "session"}).items() if k != "t"}
        line = json.dumps({"t": t, **session, "continues": before.name, "part": part},
                          separators=(",", ":"), default=_plain) + "\n"
        self._file.write(line)
        self._size = len(line) + len(os.linesep) - 1
        _prune(self.path.parent, spare=self.path)  # a run going for days deletes the oldest as it goes

    def close(self) -> None:
        with self._lock:
            self._file.close()


def record_to(name: str = "run", folder: Path = FOLDER, max_bytes: int | None = MAX_BYTES) -> Path | None:
    """Start recording to a new file in `folder` (named by the time and `name`), in parts of `max_bytes`,
    after deleting the oldest files there (`prune`); returns its path, or None (with a warning) if it
    can't be written: the run goes on without telemetry."""
    global _sink
    stop()
    _prune(folder)
    path = folder / f"{time.strftime(STAMP)}-{name}.jsonl"
    try:
        _sink = Recorder(path, max_bytes)
    except OSError as e:
        print(f"telemetry off: can't write {path} ({e})", file=sys.stderr)
        return None
    return path


def prune(folder: Path = FOLDER, keep_days: float | None = None, keep_bytes: int | None = None,
          now: float | None = None, dry_run: bool = False, spare: Path | None = None) -> list[tuple[Path, int]]:
    """Delete the telemetry files (*.jsonl) in `folder` last written more than `keep_days` (KEEP_DAYS)
    ago, then the oldest of the rest until they take at most `keep_bytes` (KEEP_BYTES). Returns those
    deleted (path, bytes), oldest first; with `dry_run`, those it would delete. A file that can't be
    deleted (a run still writing it) is left, and the next oldest goes instead; `spare` (the part being
    written) is never deleted."""
    keep_days = KEEP_DAYS if keep_days is None else keep_days
    keep_bytes = KEEP_BYTES if keep_bytes is None else keep_bytes
    now = time.time() if now is None else now
    files = sorted((st.st_mtime, path.name, st.st_size, path) for path, st in files_in(folder))  # oldest first
    total, gone = sum(size for _, _, size, _ in files), []
    for mtime, _, size, path in files:
        if now - mtime <= keep_days * 86400 and total <= keep_bytes:
            break  # the rest are newer still
        if spare is not None and path.name == spare.name:
            continue
        if not dry_run:
            try:
                path.unlink()
            except OSError:
                continue
        total -= size
        gone.append((path, size))
    return gone


def _prune(folder: Path, spare: Path | None = None) -> None:
    """`prune`, saying so when it deletes anything."""
    if gone := prune(folder, spare=spare):
        print(f"telemetry: deleted the {len(gone)} oldest files in {folder} (older than {KEEP_DAYS} days, "
              f"or past {KEEP_BYTES / 1024 ** 3:.0f} GB in all)", file=sys.stderr)


def stop() -> None:
    """Stop recording (closes the file)."""
    global _sink
    sink, _sink = _sink, None
    if isinstance(sink, Recorder):
        sink.close()


@contextmanager
def captured() -> Iterator[list[dict]]:
    """Collect records in a list instead of a file (for tests)."""
    global _sink
    before, records = _sink, []
    _sink = records.append
    try:
        yield records
    finally:
        _sink = before


def read(path: Path) -> list[dict]:
    """Every record in a telemetry file (a half-written last line is skipped)."""
    with path.open(encoding="utf-8") as f:
        return _records(f)


class Follower:
    """Reads the records appended to a telemetry file since the last call (for a live view)."""

    def __init__(self, path: Path):
        self.path = path
        self._pos = 0
        self._partial = ""

    def poll(self) -> list[dict]:
        try:
            with self.path.open(encoding="utf-8") as f:
                f.seek(self._pos)
                chunk = f.read()
                self._pos = f.tell()
        except OSError:
            return []
        text, self._partial = self._partial + chunk, ""
        lines = text.split("\n")
        if not text.endswith("\n"):
            self._partial = lines.pop()  # still being written
        return _records(lines)


def newest(folder: Path = FOLDER) -> Path | None:
    """The telemetry file written to last (the run in progress, while there is one), if any.

    The window and the web page look for a newer run every second. On Windows os.scandir has each file's
    last write without a call per file: 0.10 ms with 40 files and 2.9 ms with 1000, where Path.glob and a
    stat of each took 1.2 ms and 44 ms. Comparing the names `record_to` gives was no faster (3.3 ms with
    1000) and is wrong once a year: they're in local time, so in the hour that repeats when the clocks go
    back, a finished run's name sorts after the one in progress."""
    try:
        with os.scandir(folder) as entries:
            files = [(e.stat().st_mtime, e.name) for e in entries if e.name.endswith(".jsonl")]
    except OSError:  # no folder yet (or a file gone while looking: the next look finds the rest)
        return None
    return folder / max(files)[1] if files else None


def files_in(folder: Path = FOLDER) -> Iterator[tuple[Path, os.stat_result]]:
    """Each telemetry file in `folder` and its stat (none if there's no folder; one deleted meanwhile is
    left out)."""
    try:
        with os.scandir(folder) as entries:
            for e in entries:
                if e.name.endswith(".jsonl"):
                    try:
                        if e.is_file():
                            yield Path(e.path), e.stat()
                    except OSError:
                        continue
    except OSError:
        return


def _records(lines) -> list[dict]:
    """The records in lines of a telemetry file; blank and broken lines are skipped."""
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _plain(value):
    """JSON fallback: numpy numbers and other odd values as plain numbers or text."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
