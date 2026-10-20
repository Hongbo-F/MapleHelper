"""Telemetry: what the stack sees, thinks and does, as topics in a JSON-lines file. The mini-map window
(gamelab.ui) follows the file live or replays it.

Producers call `emit(topic, **data)` with plain JSON data. Nothing is written until a recorder is
installed (`record_to`), so tests and library use pay nothing. Each line is
{"t": seconds, "topic": topic, **data}. `t` is wall-clock time (the simulator's fake clock in a
simulated run), so a replay keeps the original pace.

Topics (map units; producers build the records, ui.state folds them):

  session   map, command, mode (live/sim)                      the command line, the simulator
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
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .paths import TELEMETRY

FOLDER = TELEMETRY  # one file per run, newest last (git-ignored, like all of lab/)

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
    """Appends records to a JSON-lines file, one flushed line each, so a reader can follow along."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def __call__(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":"), default=_plain)
        with self._lock:
            if not self._file.closed:
                self._file.write(line + "\n")
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


def record_to(name: str = "run", folder: Path = FOLDER) -> Path | None:
    """Start recording to a new file in `folder` (named by the time and `name`); returns its path, or
    None (with a warning) if it can't be written: the run goes on without telemetry."""
    global _sink
    stop()
    path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.jsonl"
    try:
        _sink = Recorder(path)
    except OSError as e:
        print(f"telemetry off: can't write {path} ({e})", file=sys.stderr)
        return None
    return path


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
    """The most recently started telemetry file, if any.

    `record_to` names each file by the second it started, so the names are compared, not the files: the
    window and the web page look for a newer run every second, and reading names only takes 0.09 ms with
    38 files and 2.9 ms with 1000, where stat-ing every file took 1.0 ms and 36 ms. Runs started in the
    same second, and files not named by a time, go by when they were last written."""
    try:
        with os.scandir(folder) as entries:
            names = [e.name for e in entries if e.name.endswith(".jsonl")]
    except OSError:  # no folder yet
        return None
    timed = [n for n in names if _stamped(n)]
    candidates = [n for n in names if not _stamped(n)]  # untimed: by when they were last written
    if timed:
        last = max(timed)[:15]
        candidates += [n for n in timed if n.startswith(last)]
    if len(candidates) <= 1:
        return folder / candidates[0] if candidates else None

    def when(name: str) -> tuple[float, float]:
        try:
            written = (folder / name).stat().st_mtime
        except OSError:
            written = 0.0
        started = _started(name) if _stamped(name) else None
        return (written if started is None else started, written)
    return folder / max(candidates, key=when)


def _stamped(name: str) -> bool:
    """Whether a file is named as `record_to` names them: by the time it started (`%Y%m%d-%H%M%S-...`)."""
    return len(name) >= 15 and name[:8].isdigit() and name[8] == "-" and name[9:15].isdigit()


def _started(name: str) -> float | None:
    """When the run in a file `record_to` named started (None: not a real time)."""
    try:
        return time.mktime(time.strptime(name[:15], "%Y%m%d-%H%M%S"))
    except (ValueError, OverflowError):
        return None


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
