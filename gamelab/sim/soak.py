"""Soak runs: the stack kept going for minutes on end, offline, to see whether what it holds in memory keeps
growing. Not part of pytest (they run for minutes):

    python -m gamelab.sim.soak frames [--minutes 10]   looks at the saved frames, as fast as they go: the real
                                                       perception and localization, the hunter's bookkeeping
                                                       after every look (ledger, drop store), the monster
                                                       tracker, telemetry
    python -m gamelab.sim.soak sim [--minutes 10]      the hunt loop in the simulator (preset game, seed 0),
                                                       for as many simulated hours as the minutes allow
    python -m gamelab.sim.soak feed [--minutes 10]     the continuous feed's child process on the saved frames,
                                                       the stack's side asking of it as a fight does; the
                                                       child's memory is followed too, and its frames a
                                                       second, its frame loop's step, its searches and its
                                                       CPU

Python's own allocations are followed with tracemalloc, and the process's private bytes too (what OpenCV and
numpy take outside it). Every SAMPLE_S it notes both; after the warm-up (WARMUP of the run: templates
loaded, threads started, caches filled) it fits a straight line to the samples, which gives the growth an
hour, and compares the last snapshot with the warm-up's, line by line and by part of the code (the stack,
the simulator, the rest); `--no-trace` follows the private bytes alone, at full speed. Telemetry is written
in parts of PART_BYTES to a temp folder, so rotation runs too. Nothing is written under gamelab/lab, no key is
sent and no window is touched.
"""

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

from .. import paths, telemetry

SAMPLE_S = 10.0           # s between samples
WARMUP = 0.2              # share of the run before growth is counted
PART_BYTES = 1024 * 1024  # telemetry parts this big (a run's are 64 MB): a soak goes through several
HOLD = 30                 # frames: looks in a row at each saved frame (her EXP and HP change between them)
TOP = 12                  # lines of code shown, by growth

_clock = time.perf_counter  # the real clock: the simulator puts its own in `time` while it runs


class Meter:
    """Samples of memory against wall time over a run of `seconds`, and snapshots at the warm-up's end
    and at the run's."""

    def __init__(self, seconds: float):
        self.seconds, self.start = seconds, _clock()
        self.samples: list[tuple[float, int, int | None]] = []  # (s from the start, traced, private bytes)
        self.warm: tuple[float, tracemalloc.Snapshot] | None = None
        self.end: tracemalloc.Snapshot | None = None
        self._next = 0.0

    def elapsed(self) -> float:
        return _clock() - self.start

    def due(self) -> bool:
        """Whether the run is over (then the end snapshot is taken, once)."""
        if self.elapsed() < self.seconds:
            return False
        if self.end is None:
            self.sample(force=True)
            self.end = tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None
        return True

    def sample(self, force: bool = False) -> None:
        t = self.elapsed()
        if t < self._next and not force:
            return
        self._next = t + SAMPLE_S
        if self.warm is None and t >= WARMUP * self.seconds:
            # First: a snapshot isn't traced, but its tens of MB are in the private bytes from now on.
            self.warm = (t, tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None)
        self.samples.append((t, tracemalloc.get_traced_memory()[0], private_bytes()))


class _Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
        (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                             "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                             "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]


def private_bytes(pid: int | None = None) -> int | None:
    """A process's private bytes (Windows' commit charge for it; this one's by default), or None elsewhere
    (or if it can't be read)."""
    if sys.platform != "win32":
        return None
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
    handle = k32.GetCurrentProcess() if pid is None else k32.OpenProcess(0x1000, False, pid)  # query limited info
    if not handle:
        return None
    try:
        c = _Counters(cb=ctypes.sizeof(_Counters))
        return c.PagefileUsage if psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb) else None
    finally:
        if pid is not None:
            k32.CloseHandle(handle)


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]


def cpu_seconds(pid: int | None = None) -> float | None:
    """The CPU time a process has used, all its threads, kernel and user (this one's by default), or None
    elsewhere (or if it can't be read)."""
    if sys.platform != "win32":
        return None
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    k32.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(_FileTime)] * 4
    handle = k32.GetCurrentProcess() if pid is None else k32.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        times = [_FileTime() for _ in range(4)]  # created, exited, kernel, user (100 ns ticks)
        if not k32.GetProcessTimes(handle, *(ctypes.byref(f) for f in times)):
            return None
        return sum((f.high << 32 | f.low) for f in times[2:]) / 1e7
    finally:
        if pid is not None:
            k32.CloseHandle(handle)


def slope(points: list[tuple[float, float]]) -> float:
    """The least-squares slope of (t, y) points (per unit of t)."""
    n = len(points)
    if n < 2:
        return 0.0
    mt, my = sum(t for t, _ in points) / n, sum(y for _, y in points) / n
    var = sum((t - mt) ** 2 for t, _ in points)
    return sum((t - mt) * (y - my) for t, y in points) / var if var else 0.0


def part_of(filename: str) -> str:
    """Which part of the code a traced allocation was made in: the stack, the simulator, or the rest
    (Python itself, numpy, OpenCV's wrappers, generated code)."""
    path = os.path.normcase(os.path.abspath(filename)) if not filename.startswith("<") else filename
    if path.startswith(os.path.normcase(str(paths.ROOT / "sim"))):
        return "simulator"
    return "stack" if path.startswith(os.path.normcase(str(paths.ROOT))) else "other"


def _trend(points: list[tuple[float, int]]) -> str:
    """Memory over the samples since the warm-up: the first and the last, the lowest and highest, the means
    of the first and the last third (steadier than two samples: big looks come and go), and the line."""
    mb = 1024 * 1024
    ys = [y for _, y in points]
    third = max(1, len(ys) // 3)
    return (f"{ys[0] / mb:.1f} MB after the warm-up, {ys[-1] / mb:.1f} at the end ({min(ys) / mb:.1f}-"
            f"{max(ys) / mb:.1f}); the first third's mean {sum(ys[:third]) / third / mb:.1f}, the last's "
            f"{sum(ys[-third:]) / third / mb:.1f}; the line {slope(points) * 3600 / mb:+.2f} MB an hour")


def report(meter: Meter, what: str, held: dict) -> list[str]:
    """What the run shows: the samples' growth an hour, the growth since the warm-up by part and by line,
    and how many things the stack holds at the end (`held`)."""
    warm_t, warm = meter.warm
    after = [s for s in meter.samples if s[0] >= warm_t]
    lines = [f"soak {what}: {meter.elapsed() / 60:.1f} min, {len(meter.samples)} samples; warm-up {warm_t:.0f} s"]
    for name, i in (("traced (tracemalloc)", 1), ("private bytes", 2)):
        if any(s[i] is None for s in after) or (i == 1 and warm is None):
            continue
        lines.append(f"{name}: " + _trend([(s[0], s[i]) for s in after]))
    if warm is None:  # not traced (--no-trace)
        return lines + ["held at the end: " + ", ".join(f"{k} {v}" for k, v in held.items())]
    grown: dict[str, int] = {}
    for stat in meter.end.compare_to(warm, "filename"):
        part = part_of(stat.traceback[0].filename)
        grown[part] = grown.get(part, 0) + stat.size_diff
    lines.append("growth since the warm-up by part: " + ", ".join(
        f"{part} {grown.get(part, 0) / 1024:+.0f} KB" for part in ("stack", "simulator", "other")))
    lines.append(f"the {TOP} lines that grew most since the warm-up:")
    for stat in meter.end.compare_to(warm, "lineno")[:TOP]:
        frame = stat.traceback[0]
        name = os.path.relpath(frame.filename, paths.REPO) if part_of(frame.filename) != "other" else frame.filename
        lines.append(f"  {stat.size_diff / 1024:+9.1f} KB {stat.count_diff:+7d} blocks  {name}:{frame.lineno}")
    lines.append("held at the end: " + ", ".join(f"{k} {v}" for k, v in held.items()))
    return lines


def soak_frames(meter: Meter) -> dict:
    """Looks at the saved frames for the meter's time: each frame HOLD looks in a row, every 5th look with
    the drops on her platform, every 30th the whole screen; after each, the hunter's bookkeeping (its own
    listeners), the monster tracker and a glance; a throw every 10th look (for the ledger)."""
    from maplehelper.engine import RIGHT

    from ..analysis import bench, calibration
    from ..behavior.hunter import Hunter
    from ..control.timeline import her_setup
    from ..localization.monsters import Tracker, tracks_record
    from ..perception.sprites import detector_for
    from . import scenarios
    model = scenarios.real_map()
    frames = bench.load(bench.saved_frames())
    if model is None or not frames:
        raise SystemExit("needs the map's game data and the saved frames in gamelab/lab")
    eyes = bench.SavedEyes(calibration.load().template, detector_for(model))
    hunter, tracker = Hunter(eyes, her_setup()[1], model, log=lambda message: None), Tracker(model)
    hit_log = paths.HIT_LOG
    with tempfile.TemporaryDirectory(prefix="gamelab-soak-") as tmp:
        paths.HIT_LOG = Path(tmp) / "hits.jsonl"  # her HP changes between frames: "hits"
        folder = Path(tmp) / "telemetry"
        telemetry.record_to("soak-frames", folder=folder, max_bytes=PART_BYTES)
        telemetry.emit("session", map=model.id, command="soak", mode="soak")
        try:
            n = 0
            while not meter.due():
                eyes.show(frames[(n // HOLD) % len(frames)][1])
                s = hunter.look(loot=n % 5 == 0, wide=n % 30 == 0)
                tracks = tracker.update(time.monotonic(), s.monsters)
                if tracks:
                    telemetry.emit("monsters", **tracks_record(tracks, model))
                if n % 10 == 0:
                    hunter.ledger.on_throw(time.monotonic(), RIGHT)
                eyes.glance()
                n += 1
                meter.sample()
        finally:
            telemetry.stop()
            paths.HIT_LOG = hit_log
        parts = list(telemetry.files_in(folder))
        return {"looks": n, "kills in the ledger": len(hunter.ledger.kills), "piles": len(hunter.drops.piles),
                "EXP gains kept": len(hunter.exp_events),
                "speeds kept": sum(len(v) for v in tracker.speeds.values()),
                "telemetry parts": len(parts), "telemetry MB": round(sum(st.st_size for _, st in parts) / 2 ** 20, 1)}


def soak_feed(meter: Meter) -> dict:
    """The continuous feed's child process (localization.feed) on the saved frames for the meter's time, each
    frame HOLD frames in a row as the screen, 30 a second. The stack's side asks of it as a fight does: a
    look with her band every 0.1 s (with the drops on her platform every 5th, the whole screen every 30th),
    a glance between, its events drained, a throw every 10th look. The child's private bytes are sampled
    with the meter's (the report's last lines); after the warm-up, its frames a second, its frame loop's
    step (each state's work["s"]), its searches (each late state's) and its CPU (cores: CPU seconds a
    second, all its threads)."""
    import functools

    from maplehelper.engine import RIGHT

    from ..analysis import bench, calibration
    from ..attack import reach
    from ..localization import feed
    from . import scenarios
    model, files = scenarios.real_map(), [str(p) for p in bench.saved_frames()]
    if model is None or not files:
        raise SystemExit("needs the map's game data and the saved frames in gamelab/lab")
    client = feed.FeedClient(functools.partial(feed.saved_world, files, model.id, calibration.load().template,
                                               (reach.LOOK_UP, reach.LOOK_DOWN), HOLD))
    if not client.alive:
        raise SystemExit(f"the feed didn't start: {client.why}")
    child, n, missed, events, glanced, waits = [], 0, 0, 0, 0, {"band": [], "loot": [], "wide": []}
    took: dict[str, list[float]] = {"step": [], "band": [], "wide": [], "drops": []}
    warm: list = []  # (time, the child's CPU seconds, its newest frame) once warmed up

    def seen(st) -> None:  # (the client's reader thread)
        if not warm or not st.visible:
            return
        w = st.work
        kind = ("wide" if w.get("wide") else "band" if w.get("band") else "drops") if st.late else "step"
        took[kind].append(w.get("s", 0.0))
    client.on_state = seen
    try:
        while not meter.due() and client.alive:
            now = time.monotonic()
            st = client.latest(feed.wanted(now, True, n % 30 == 0, n % 5 == 0, False))
            waits["wide" if n % 30 == 0 else "loot" if n % 5 == 0 else "band"].append(time.monotonic() - now)
            missed += st is None
            if st is not None:
                client.frame_of(st)
            glanced += client.glance(now) is not None
            events += len(client.drain_events())
            if n % 10 == 0:
                client.throw(now, RIGHT)
            n += 1
            before = len(meter.samples)
            meter.sample()
            if len(meter.samples) > before:
                child.append((meter.samples[-1][0], private_bytes(client.proc.pid)))
            if not warm and meter.warm is not None:
                warm.append((time.perf_counter(), cpu_seconds(client.proc.pid), newest_seq(client)))
            time.sleep(max(0.0, now + 0.1 - time.monotonic()))
        end = (time.perf_counter(), cpu_seconds(client.proc.pid), newest_seq(client))
    finally:
        client.close()
    since = meter.warm[0] if meter.warm else 0.0
    points = [(t, b) for t, b in child if t >= since and b is not None]
    rate = cpu = "not read"
    if warm:
        (t0, c0, f0), (t1, c1, f1) = warm[0], end
        rate = f"{(f1 - f0) / (t1 - t0):.1f}"
        cpu = f"{(c1 - c0) / (t1 - t0):.2f}" if c0 is not None and c1 is not None else "not read"
    return {"looks asked": n, "the wait for one (median, p90 ms)": _ms(waits),
            "not given in time": missed, "glances": glanced, "events drained": events,
            "dropped events": client.dropped, "child's frames": (client.bye or {}).get("frames"),
            "child's frames a second after the warm-up": rate, "child's CPU (cores)": cpu,
            "child's frame step and searches (median, p90 ms; how many)": _ms(took, counts=True),
            "child's private bytes": _trend(points) if len(points) >= 2 else "not read"}


def newest_seq(client) -> int:
    """The newest frame number the feed's client has had a state of."""
    return max((st.seq for st in list(client.states)), default=0)


def _ms(samples: dict[str, list[float]], counts: bool = False) -> str:
    """Each list's median and 90th percentile (in s) in ms, and with `counts` how many; empty ones left out."""
    out = []
    for k, v in samples.items():
        if v:
            v = sorted(v)
            out.append(f"{k} {1000 * v[len(v) // 2]:.1f}, {1000 * v[int(0.9 * (len(v) - 1))]:.1f}"
                       + (f" ({len(v)})" if counts else ""))
    return ", ".join(out)


def soak_sim(meter: Meter, seed: int = 0) -> dict:
    """The hunt loop in the simulator (sim.harness's `loop`, preset game) until the meter's time is up: one
    long run, then the next seed's if it ends first (she may die, as in the harness's runs). The meter is
    read at every look, whatever she's doing."""
    from ..behavior import hunt
    from ..localization.senses import Senses
    from ..motion import physics
    from . import harness, scenarios
    from .fakes import SimTimeout
    hold, book, look = {}, hunt.Hunt._book, Senses.look

    def booked(self, s):
        hold["hunt"] = self
        return book(self, s)

    def looked(self, *a, **k):
        meter.sample()
        if meter.due():
            raise SimTimeout("the soak's time is up")
        return look(self, *a, **k)

    hunt.Hunt._book, Senses.look = booked, looked
    runs = []
    with tempfile.TemporaryDirectory(prefix="gamelab-soak-") as tmp:
        folder = Path(tmp) / "telemetry"
        telemetry.record_to("soak-sim", folder=folder, max_bytes=PART_BYTES)
        try:
            while not meter.due():
                trace, sim = scenarios.run_sim(harness.scenario("loop", seed, "game", minutes=100_000))
                runs.append(f"seed {seed} {trace['seconds'] / 3600:.2f} h ({str(trace['result'])[:60]})")
                seed += 1
        finally:
            hunt.Hunt._book, Senses.look = book, look
            telemetry.stop()
        parts = list(telemetry.files_in(folder))
    h = hold["hunt"].h
    return {"runs": "; ".join(runs), "kills": len(sim.world.kill_log), "kills in the ledger": len(h.ledger.kills),
            "piles": len(h.drops.piles), "piles known": len(h.drops.known), "fly cache": len(physics._FLY_CACHE),
            "simulator's plans kept": len(trace["plans"]), "simulator's log lines": len(trace["log"]),
            "telemetry parts": len(parts), "telemetry MB": round(sum(st.st_size for _, st in parts) / 2 ** 20, 1)}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from ..motion.physics import FIXED_STATS
    # As the harness's runs: the same set order in every run, and her stats as measured, not config.toml's.
    if os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get(FIXED_STATS) != "1":
        env = dict(os.environ, PYTHONHASHSEED="0", **{FIXED_STATS: "1"})
        return subprocess.call([sys.executable, "-X", "utf8", "-m", "gamelab.sim.soak", *argv], env=env,
                               cwd=str(paths.REPO))
    p = argparse.ArgumentParser(prog="python -m gamelab.sim.soak", description=__doc__.splitlines()[0])
    p.add_argument("what", choices=("frames", "sim", "feed"))
    p.add_argument("--minutes", type=float, default=10.0, help="how long it runs (wall clock)")
    p.add_argument("--seed", type=int, default=0, help="sim: the world seed")
    p.add_argument("--no-trace", action="store_true", help="the private bytes only, without tracemalloc's "
                                                           "cost (16 looks a second here instead of 10)")
    args = p.parse_args(argv)
    if not args.no_trace:
        tracemalloc.start()
    meter = Meter(args.minutes * 60)
    held = {"frames": soak_frames, "feed": soak_feed}.get(args.what, lambda m: soak_sim(m, args.seed))(meter)
    print("\n".join(report(meter, args.what, held)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
