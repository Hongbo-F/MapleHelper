"""Telemetry: what the stack sees, thinks and does, as topics in a JSON-lines file. The mini-map window
(gamelab.ui) follows the file live or replays it.

Producers call `emit(topic, **data)` with plain JSON data. Nothing is written until a recorder is
installed (`record_to`), so tests and library use pay nothing. Each line is
{"t": seconds, "topic": topic, **data}. `t` is wall-clock time (the simulator's fake clock in a
simulated run), so a replay keeps the original pace.

A record is made into its line where it's emitted, and written by a thread of the recorder's own (`Recorder`), in
batches FLUSH_S apart: a write held up by the disk never holds up the frame loop or a hunt (live, on 2026-10-25 and 26,
the hunts and the perception service stood still together for 0.5-2 s now and then: plausibly the SSD's synchronous
writes, not proven; D:/MapleHelper-scratch/pconst/plan). A reader following the file sees a record FLUSH_S after it at
the latest (`flush` waits for those emitted so far). Past QUEUE_LINES waiting, records are dropped and counted
(`Recorder.dropped`). The perception service's console goes the same way (`console_in_background`).

Files are named for the time a run started and what it is (`<%Y%m%d-%H%M%S>-<name>.jsonl`); the newest
run is the file written to last (`newest`). A run goes on in a new part past MAX_BYTES: a file of its own,
named for the time the part starts (`...-<name>-part2.jsonl`, always later than the part before), that
starts with the run's session record marked `continues` (the part before's file name) and `part`; being
written to, it's the newest. When a run starts, and each time one goes on in a new part, the oldest files
are deleted until the rest take at most KEEP_BYTES (`prune`; never the part being written), and, if
KEEP_DAYS is set, those last written longer ago than that first (by default none goes for its age). So a
run left going for days (`watch`) keeps the folder bounded too.

Topics (map units; producers build the records, ui.state folds them):

  session   map, command, mode (live/sim); continues, part;    the command line, the simulator; a new part;
            physics {character, class, speed, jump, bonus,     her physics as the run plans her moves (from
            walk, climb}                                       2026-11-02: motion.physics.record; bonus the
                                                               Speed the client adds, motion.body.SPEED_BONUS)
  motion    old [walk, climb], new [walk, climb], speed, was,  behavior.hunter: her planner's walk and climb
            why (panel, kept, profile)                         changed (units/s; from 2026-11-02), the client's
                                                               formula at SPEED `speed` (`was` before): her
                                                               stat window's read, as her stats file kept it,
                                                               or her profile's
  scene     her {x, y, platform, how, seen, facing, grounded}, every look (behavior.hunter, after Senses;
            hp, mp, monsters [{kind, x, y, platform, bar,      `watch` 10 a second, her facing not known;
            facing}], loot [{kind, x, y, platform}], searched, from 2026-10-21's evening also what it
            wide, looked [[platform, x0, x1]], view [x0, y0,   searched, for analysis.respawn; players only
            x1, y1], players [{id, kind, x, y, platform}];     when there are any: localization.players; seq
            seq, age                                           the feed's frame number, age ms since the frame
                                                               showed: from 2026-11-02)
  tracks    list [{id, kind, state, x, y, vx, facing,          `watch` and the perception service, twice a
            pose, platform, bar, hits, seen, cause, gain,      second (localization.tracking; gain: the frame
            share}], frames                                    time of her gain the track is bound to, share:
                                                               a share's; from 2026-10-27; vx the game data's
                                                               speed the way it heads, no measure, and pose,
                                                               from 2026-10-30: its walk is x over time)
  exp       exp, level, gain, kinds, total, per_minute         each EXP gain (behavior.hunter; `watch`), her
                                                               number as read (localization.gains)
  drops     pending [{x, y, platform, mesos}], inventory_full  every look (behavior.hunter)
  glance    x, y, hp                                           control.navigator, while a chain runs
  route     dst, moves [{kind, src, dst, x, land_x}], from     control.navigator, each plan (from 2026-10-21 also
            {platform, x}, keep_off, avoid; ways [{via, cost,  where it was planned from and what it kept off; a
            chance}], chose                                    trip's first, the ways it chose among: route.ROUTES)
  chain     why, text, legs [{kind, src, dst, spans, end}]     control.navigator, each key plan sent
  moved     moves [{kind, src, dst, x}], planned, landed, hit  control.navigator, after a chain or a move (from
            (touched on the way: from 2026-10-28 any touch,    2026-10-21: the coverage report's)
            MISS too), touch (its kind)
  monsters  list [{platform, x, vx, heading, reach}]           control.navigator (avoiding monsters)
  trip      dst, result; on, x, walk [[units, hold ms]]       control.navigator, at every trip's end (from
                                                               2026-11-02 where she was as it ended, and its
                                                               arrival walk: navigator._trip_end)
  attack    key, why, kind, left, dead, hp, mp, platform,      behavior.attacks, each throw's attack and why
            x; track, side                                     (Attacks._say; from 2026-11-02 its target's
                                                               track, the perception service's, and the side:
                                                               localization.tracking counts her aim against her
                                                               kill's binding, the replay kit's feed reads both)
  no_cast   why, side, by, kind, platform, x, track; mp,       behavior.attacks: a cast not sent (every one her
            cost                                               MP refuses from 2026-11-02, with her MP reckoned;
                                                               its target's fields an attack record's)
  sweep     dest, candidates, order, extra, value, direct,     behavior.hunt, its itinerary when it changes (from
            budget, out; to                                    2026-11-02 why its way goes on to dest: Hunt._sweep_to)
  grind     what (loop, next, state, end), ...; state and     behavior.grind (from 2026-11-02 the loop's promise
            end: loop_s, loop_exp, loop_rate, promised,        against what it gave, the s by hunt rule:
            delivered, rule_s                                  Grind._promise)
  summary   seconds, standby_s, kills, exp_gains, exp,         behavior.hunt, once at the run's end (from
            exp_read, throws, casts, touches, misses, drinks,  2026-11-02: HuntReport.record)
            lowest_hp, max_hp, a_minute {...}, rules, rule_s
  stance    act, text, aimed [{kind, x, y}]                    behavior.fight, each look of a fight
  tour      order, safe, grounds / stop, spot                  behavior.tour (stop, spot: also behavior.hunt)
  hunt      mode, rule, objective, ground, piles [{x, platform,   behavior.hunt, when the rule changes
            status, hers}]
  event     kind, text; track (died, exp_gain, hp_drop,        behavior: kill, touch (from 2026-10-28, where it
            victim, let_go, touch, touch_fix); touch's touch   was 'hit', her HP dropping: MISS or damage, when
            (MISS, damage, ?), at, how (popup, hp), number,    its popup showed), touch_fix (a MISS that took HP
            crowded; touch_fix's was (MISS); immune's planned, after all), touch_unfit (her HP number's drop that
            left; died's data cause, at, hurt,                 fits no kind), immune (a chain not sent: her
                                                               invulnerability spent planning), potion, safety_stop,
                                                               exp (what a gain's size said), player (one
            by, share, seen; victim's kind, x, platform,       coming or going), died (the feed's deaths
            gain, share; let_go's kind, gain, why; retold's    not hers), backstop, follow, probe, retold (a
            key, side; backstop's (a pass regardless, from     throw told again the way the frame showed her
            2026-11-02) move, blocker, waited, blocked_s,      facing), landing (a trip's last landing priced:
            shots, why_not, episode, nth; kill_moved's         control.navigator), kill_moved (her kill put on
            gained, old, old_platform, old_x, new,             another monster by the tracker, from 2026-11-02:
            platform, x; rope_hang's step, x, under, s         behavior.ledger), rope_hang (a hang under a rope's
                                                               top and its end, from 2026-11-02:
                                                               control.navigator); `watch` and the perception
                                                               service:
                                                               died, exp_gain, hp_drop, touch, touch_fix, victim (a gain bound to
                                                               the monster her attack took), let_go (not it:
                                                               why 'seen alive', 'retold' or 'another seen
                                                               dying'; from 2026-10-27; seen: its death seen)
  log       text                                               every log line
  gc        collections [gen 0, 1, 2], pause_ms, longest_ms,   once a minute (the perception service, the commands
            frozen, seconds                                    that record telemetry: gcstats)
  perf      s, fps, cores, pass_ms [median, p90, p99,      the perception service, with its 10 s console line
            max], look_ms [median, p90], wide_ms [median,      (service.Perf: each frame's pass, her look and the
            p90, max], detections_a_frame, tracks_a_frame,     whole screen's search in it, the monsters found and
            strips_a_frame, age_ms [median, p90],              followed and the strips of drops a frame, a state's
            tracks_a_min, flickers_a_min                       age as it went out; from 2026-10-30. From 2026-10-28
                                                               until then wide_a_s, wide_asked_a_s, wide_ms,
                                                               wide_age_ms, wide_age_drops_ms, busy, band_alone_*:
                                                               the searches' thread's)
"""

import atexit
import json
import os
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .paths import TELEMETRY

# The writer's thread writes what's waiting this often (and flushes it to the file): a reader following the file (the
# mini-map, the web page) sees it this late at most. The service writes about 6 KB a second (a scene 10 times a second
# at ~400 bytes, the tracks twice at ~850), so about 1 KB a batch.
FLUSH_S = 0.2
# Records waiting to be written at most, past which the newest are dropped (counted: Recorder.dropped). 50,000 are
# about 2 hours of the service's records and 25 MB: only a disk that stopped taking writes fills it.
QUEUE_LINES = 50_000
BACKGROUND = True  # record_to's recorders write on their own thread (False: in the caller, as until 2026-10-26)

FOLDER = TELEMETRY  # one file per run, newest last (git-ignored, like all of lab/)
STAMP = "%Y%m%d-%H%M%S"  # file names start with the time the run (or the part) started
# A run's file is continued in a new part past this. Hunting wrote 4.2 MB an hour (72 KB a minute, from
# the live tours), so a part is hours of play; a continuous feed of 10 looks a second would write 10-15 MB
# an hour. Parts let the retention below delete a long run a piece at a time, and keep any one file small
# enough for a replay to load whole.
MAX_BYTES = 64 * 1024 * 1024
# What's kept when a run starts or goes on in a new part: the newest files, 2 GB of them at most (`watch` writes
# about 400 MB a day). None goes for its age alone (KEEP_DAYS None): live telemetry is evidence that later fits
# go back to (the terrain and geometry fits of 2026-10-20 were made from the live runs of 2026-10-19 and 20),
# and small old files cost next to no disk (the first two days of live runs were 38 files and 2.9 MB). The
# size cap alone bounds the disk. A number of days here deletes files last written longer ago than that too.
KEEP_DAYS: float | None = None
KEEP_BYTES = 2 * 1024 ** 3
RECENT_S = 7 * 86400  # s: files the directory says were written this recently are asked their last write (newest)
_DEFAULT = object()  # prune's keep_days when not given: KEEP_DAYS

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
    except _Stopped:  # (its writer couldn't write: it said why)
        stop()
    except Exception as e:
        stop()
        print(f"telemetry stopped: can't record {topic!r} ({e})", file=sys.stderr)


def recording() -> bool:
    """Whether a recorder is installed (callers skip building records nobody would see)."""
    return _sink is not None


def to_file() -> bool:
    """Whether records go into a file now (a Recorder's: not a test's `captured`)."""
    return isinstance(_sink, Recorder)


class _Stopped(Exception):
    """A recorder whose writer couldn't write (a full disk): it said why; the records go nowhere now."""


class Recorder:
    """Appends records to a JSON-lines file, so a reader can follow along: each made into its line where it's emitted
    (a value JSON can't hold raises there), and written with those before it by a thread of the recorder's own every
    FLUSH_S (`background`; else at once, one flushed line each, as until 2026-10-26). With `max_bytes`, a record that
    would take the file past it goes into a new part instead (see the top): `path` is then the part being written, and
    `part` its number. Past `keep` records waiting the newest are dropped (`dropped` counts them); a write that fails
    (a full disk) stops it, saying why once."""

    def __init__(self, path: Path, max_bytes: int | None = None, background: bool = True, keep: int = QUEUE_LINES):
        self.path, self.max_bytes, self.part, self.keep = path, max_bytes, 1, keep
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        self._size = path.stat().st_size  # bytes in the file (lines are ASCII: json.dumps escapes the rest)
        self._lock = threading.Lock()     # the file's (the writer's, or the caller's without one)
        self._session: dict | None = None  # the run's session record, to start each new part with
        named = re.fullmatch(r"(\d{8}-\d{6})-(.+)", path.stem)
        self._name = named.group(2) if named else path.stem
        self._started = int(time.mktime(time.strptime(named.group(1), STAMP))) if named else int(time.time())
        self.dropped = 0
        self.error: str | None = None
        self._q: deque[tuple[str, float | None]] = deque()  # (a line, its record's time) waiting to be written
        self._cond = threading.Condition()
        self._queued = self._done = 0  # lines taken, and written (or given up) since
        self._closing = self._hurry = False
        self._thread = None
        if background:
            self._thread = threading.Thread(target=self._run, name="telemetry-writer", daemon=True)
            self._thread.start()

    def __call__(self, record: dict) -> None:
        line = json.dumps(record, separators=(",", ":"), default=_plain) + "\n"
        if record.get("topic") == "session" and "continues" not in record:
            self._session = record
        if self.error is not None:
            raise _Stopped(self.error)
        if self._thread is None:
            with self._lock:
                if not self._file.closed:
                    self._write([(line, record.get("t"))])
            return
        with self._cond:
            if self._closing:
                return
            if len(self._q) >= self.keep:
                self.dropped += 1
                return
            self._q.append((line, record.get("t")))
            self._queued += 1

    def _run(self) -> None:
        """The writer's thread: what's waiting, every FLUSH_S, until closed and nothing waits."""
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._closing or self._hurry, FLUSH_S)
                batch, closing, self._hurry = list(self._q), self._closing, False
                self._q.clear()
            if batch and self.error is None:
                try:
                    with self._lock:
                        if not self._file.closed:
                            self._write(batch)
                except Exception as e:  # (a full disk, a part that couldn't be made: as emit's warning, once)
                    self.error = f"{type(e).__name__}: {e}"
                    print(f"telemetry stopped: can't write {self.path} ({self.error})", file=sys.stderr)
            with self._cond:
                self._done += len(batch)
                self._cond.notify_all()
                if closing and not self._q:
                    return

    def _write(self, batch: list[tuple[str, float | None]]) -> None:
        """Lines into the file, a new part where one would take it past max_bytes, then flushed (the lock held)."""
        for line, t in batch:
            size = len(line) + len(os.linesep) - 1  # written in text mode: "\n" is os.linesep on disk
            if self.max_bytes is not None and self._size and self._size + size > self.max_bytes:
                self._next_part(t)
            self._file.write(line)
            self._size += size
        self._file.flush()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait (`timeout` s at most) until what was emitted so far is written: whether it is."""
        if self._thread is None:
            return True
        with self._cond:
            upto, self._hurry = self._queued, True
            self._cond.notify_all()
            return self._cond.wait_for(lambda: self._done >= upto or not self._thread.is_alive(), timeout)

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

    def close(self, timeout: float = 5.0) -> None:
        """What's waiting written (`timeout` s at most: a disk that takes no writes doesn't hold the run's end up for
        longer), then the file closed. On a full disk the close's own flush fails too (the lines the failed write left
        in the file's buffer): the file is closed all the same and the error kept, not raised (it was said where the
        write failed): `stop` is how emit lets a failed recorder go, so a raise here stopped the run (the perception
        service on 2026-10-28 11:41-11:42, D: full: telemetry.emit -> stop -> close, OSError out of the frame loop)."""
        if self._thread is not None:
            with self._cond:
                self._closing = True
                self._cond.notify_all()
            self._thread.join(timeout)
        with self._lock:
            try:
                self._file.close()
            except OSError as e:
                if self.error is None:
                    self.error = f"{type(e).__name__}: {e}"


def record_to(name: str = "run", folder: Path = FOLDER, max_bytes: int | None = MAX_BYTES,
              background: bool | None = None) -> Path | None:
    """Start recording to a new file in `folder` (named by the time and `name`), in parts of `max_bytes`,
    after deleting the oldest files there (`prune`); returns its path, or None (with a warning) if it
    can't be written: the run goes on without telemetry. `background`: written by a thread of its own (BACKGROUND by
    default; see the top)."""
    global _sink
    stop()
    _prune(folder)
    path = folder / f"{time.strftime(STAMP)}-{name}.jsonl"
    try:
        _sink = Recorder(path, max_bytes, BACKGROUND if background is None else background)
    except OSError as e:
        print(f"telemetry off: can't write {path} ({e})", file=sys.stderr)
        return None
    return path


def flush(timeout: float = 5.0) -> bool:
    """Wait until what was emitted so far is in the file (`timeout` s at most): whether it is (True without a
    recorder)."""
    sink = _sink
    return sink.flush(timeout) if isinstance(sink, Recorder) else True


atexit.register(flush, 2.0)  # (a process ending without `stop`: what its writer holds goes in first)


def prune(folder: Path = FOLDER, keep_days=_DEFAULT, keep_bytes: int | None = None, now: float | None = None,
          dry_run: bool = False, spare: Path | None = None) -> list[tuple[Path, int]]:
    """Delete the oldest telemetry files (*.jsonl) in `folder` until the rest take at most `keep_bytes`
    (KEEP_BYTES), and with `keep_days` (KEEP_DAYS; None: no age rule, the default) those last written more
    than that many days ago first. Returns those deleted (path, bytes), oldest first; with `dry_run`, those
    it would delete. A file that can't be deleted (a run still writing it) is left, and the next oldest goes
    instead; `spare` (the part being written) is never deleted."""
    keep_days = KEEP_DAYS if keep_days is _DEFAULT else keep_days
    keep_bytes = KEEP_BYTES if keep_bytes is None else keep_bytes
    now = time.time() if now is None else now
    files = sorted((st.st_mtime, path.name, st.st_size, path) for path, st in files_in(folder))  # oldest first
    total, gone = sum(size for _, _, size, _ in files), []
    for mtime, _, size, path in files:
        aged = keep_days is not None and now - mtime > keep_days * 86400
        if not aged and total <= keep_bytes:
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


def kept() -> str:
    """What `prune` keeps, in words: 'the newest 2 GB', and the days if KEEP_DAYS is set."""
    days = "" if KEEP_DAYS is None else f" of those written in the last {KEEP_DAYS:g} days"
    return f"the newest {KEEP_BYTES / 1024 ** 3:g} GB{days}"


def _prune(folder: Path, spare: Path | None = None) -> None:
    """`prune`, saying so when it deletes anything."""
    if gone := prune(folder, spare=spare):
        print(f"telemetry: deleted the {len(gone)} oldest files in {folder} (it keeps {kept()})", file=sys.stderr)


def stop() -> None:
    """Stop recording (what's waiting written, the file closed)."""
    global _sink
    sink, _sink = _sink, None
    if isinstance(sink, Recorder):
        sink.close()


# --- the console, written by a thread of its own ---------------------------------------------------------------

CONSOLE_KEEP = 20_000  # pieces of text waiting for the console at most (the newest dropped past it, counted)


class Console:
    """Text for the process's console streams (sys.stdout, sys.stderr: the service's go to lab/perceive.log or the
    babysitter's file), written in order by a thread of its own every FLUSH_S and flushed: a `print(..., flush=True)`
    returns at once, whatever the disk does. `drain` writes what's waiting now (the process's end, the watchdog's)."""

    def __init__(self, keep: int = CONSOLE_KEEP):
        self.keep, self.dropped = keep, 0
        self._q: deque[tuple[object, str]] = deque()
        self._cond = threading.Condition()
        self._queued = self._done = 0
        self._hurry = False
        self._thread = threading.Thread(target=self._run, name="console-writer", daemon=True)
        self._thread.start()

    def put(self, stream, text: str) -> None:
        with self._cond:
            if len(self._q) >= self.keep:
                self.dropped += 1
                return
            self._q.append((stream, text))
            self._queued += 1

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._hurry, FLUSH_S)
                batch, self._hurry = list(self._q), False
                self._q.clear()
            touched = []
            for stream, text in batch:
                try:
                    stream.write(text)
                    if stream not in touched:
                        touched.append(stream)
                except (OSError, ValueError):  # (a console gone: nothing to say it on)
                    pass
            for stream in touched:
                try:
                    stream.flush()
                except (OSError, ValueError):
                    pass
            with self._cond:
                self._done += len(batch)
                self._cond.notify_all()

    def drain(self, timeout: float = 2.0) -> bool:
        """Wait (`timeout` s at most) until what was put so far is written."""
        with self._cond:
            upto, self._hurry = self._queued, True
            self._cond.notify_all()
            return self._cond.wait_for(lambda: self._done >= upto, timeout)


class _Queued:
    """A console stream whose writes go to a Console (its `flush` does nothing: the Console's thread flushes); the
    rest is the stream's own (encoding, fileno, isatty)."""

    def __init__(self, console: Console, stream):
        self._console, self._stream = console, stream

    def write(self, text: str) -> int:
        self._console.put(self._stream, text)
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


_CONSOLE: list[Console] = []


def console_in_background() -> Console:
    """The process's sys.stdout and sys.stderr written by a Console's thread from now on (once; the process's end
    drains it): the perception service's, whose log lines came from the frame loop."""
    if not _CONSOLE:
        console = Console()
        sys.stdout, sys.stderr = _Queued(console, sys.stdout), _Queued(console, sys.stderr)
        _CONSOLE.append(console)
        atexit.register(console.drain)
    return _CONSOLE[0]


def drain(timeout: float = 2.0) -> None:
    """The telemetry emitted and the console's text so far written, `timeout` s at most each (before an os._exit)."""
    flush(timeout)
    for console in _CONSOLE:
        console.drain(timeout)


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


def newest(folder: Path = FOLDER, now: float | None = None, match: Callable[[str], bool] | None = None,
           recent_s: float = RECENT_S) -> Path | None:
    """The telemetry file written to last (the run in progress, while there is one), if any; with `match`, of the files
    whose names it takes (ui.sources: the perception service's apart from the commands').

    The window and the web page look for a newer run every second. On Windows os.scandir has each file's
    last write without a call per file: 0.10 ms with 40 files and 2.9 ms with 1000, where Path.glob and a
    stat of each took 1.2 ms and 44 ms. But on NTFS what it has is the directory's copy, brought up to date
    only when the file is opened or closed: a part a run is still writing, that nothing else opens, shows
    as last written when it was made, 0 bytes long. A run that finished after it began then stayed the
    newest until that part closed (hours, for `watch`), and the page froze on it. So the files the
    directory says were written in the last RECENT_S are asked with os.stat (it opens them), the rest go
    by the directory: a part still open was made in that time (a part is MAX_BYTES at most: hunting writes
    that in 15 hours, `watch` in 4). 1.0 ms with 40 recent files, 24 ms with 1000. `recent_s` (RECENT_S by default)
    says how recent: the web page asks of the commands' files the last day's (ui.sources.COMMANDS_RECENT_S). Comparing
    the names `record_to` gives was no faster (3.3 ms with 1000) and is wrong once a year: they're in local time, so
    in the hour that repeats when the clocks go back, a finished run's name sorts after the one in
    progress."""
    now = time.time() if now is None else now
    try:
        with os.scandir(folder) as entries:
            listed = [(e.stat().st_mtime, e.name) for e in entries if e.name.endswith(".jsonl")
                      and (match is None or match(e.name))]
    except OSError:  # no folder yet (or a file gone while looking: the next look finds the rest)
        return None
    files = []
    for mtime, name in listed:
        if now - mtime < recent_s:
            try:
                mtime = os.stat(folder / name).st_mtime
            except OSError:  # gone meanwhile
                continue
        files.append((mtime, name))
    return folder / max(files)[1] if files else None


def files_in(folder: Path = FOLDER) -> Iterator[tuple[Path, os.stat_result]]:
    """Each telemetry file in `folder` and its os.stat (none if there's no folder; one deleted meanwhile is
    left out). Not the directory's copy that os.scandir has: on NTFS a file still being written shows
    there as it was when last opened, 0 bytes long if never, and `prune` would count it so."""
    try:
        with os.scandir(folder) as entries:
            for e in entries:
                if e.name.endswith(".jsonl"):
                    try:
                        if e.is_file():
                            yield Path(e.path), os.stat(e.path)
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
