"""Python's garbage collections, timed: how many of each generation, their pauses in all and the longest, since the
last time each reader asked (gc.callbacks; `install` once a process). The perception service says them on its console
line every 10 s and records them once a minute (telemetry topic `gc`), as the commands that record telemetry do
(`report_every`), so a lag can be told from a collection live.

Measured before (D:/MapleHelper-scratch/pconst/plan, the user asking on 2026-10-26 whether the hunts' lags were GC's):
the service's frame loop replayed on the crowd's saved frames made 207 collections in 1,622 s, 0.0037% of the time, the
longest 4.2 ms; a hunt in the simulator 0.016-0.018% of its time, no full collection in 40 minutes. So collections
don't lag a hunt now; these say so live, and would show one that grows (a run's objects piling up). `freeze` moves
everything there is into the permanent generation (gc.freeze): the service does it as each run is made, so a full
collection no longer goes through the map's templates, the models and the modules.

Imports: the standard library.
"""

import gc
import threading
import time
from collections.abc import Callable

REPORT_S = 60.0  # s between `gc` records (report_every)


class _Window:
    """Collections since a reader last asked."""
    __slots__ = ("counts", "pause", "longest", "since")

    def __init__(self):
        self.counts, self.pause, self.longest, self.since = [0, 0, 0], 0.0, 0.0, time.monotonic()


class GcStats:
    """The collections' counts and pauses (perf_counter), for each reader by name (`take`)."""

    def __init__(self):
        self.total = [0, 0, 0]
        self.pause = 0.0
        self.longest = 0.0
        self._began: float | None = None
        self._windows: dict[str, _Window] = {}
        self._lock = threading.Lock()

    def callback(self, phase: str, info: dict) -> None:
        """gc.callbacks': a collection began or ended (they run one at a time, holding Python's lock). It takes no lock
        of its own: a collection can start in any thread at any allocation, `take`'s too, and one that waited there
        for a lock its own thread held would hang the process (the review of 2026-10-26 hung `take` so with the
        collector's threshold at 1 and 50). The readers' windows are copied in one step (list() of a dict's values,
        which no other thread interrupts) and each changed in place."""
        if phase == "start":
            self._began = time.perf_counter()
            return
        if self._began is None:
            return
        took, self._began = time.perf_counter() - self._began, None
        gen = min(2, max(0, int(info.get("generation", 0))))
        self.total[gen] += 1
        self.pause += took
        self.longest = max(self.longest, took)
        for w in list(self._windows.values()):
            w.counts[gen] += 1
            w.pause += took
            w.longest = max(w.longest, took)

    def take(self, reader: str) -> dict:
        """What came since `reader` last asked (since it first did, the first time): `collections` by generation,
        `pause_ms` in all, `longest_ms`, `seconds` the window; `frozen`, the objects in the permanent generation. The
        new window is made before the lock is taken (a collection may start as it's made: `callback`)."""
        now = time.monotonic()
        fresh = _Window()
        with self._lock:  # (between readers only)
            w = self._windows.get(reader)
            self._windows[reader] = fresh
        if w is None:
            w = fresh
        return {"collections": list(w.counts), "pause_ms": round(1000 * w.pause, 2),
                "longest_ms": round(1000 * w.longest, 2), "seconds": round(now - w.since, 1),
                "frozen": gc.get_freeze_count()}

    def line(self, reader: str) -> str:
        """`take` in words, for a console line: "GC 4 (gen 0, 1, 2: 3, 1, 0), 0.6 ms, the longest 0.3"."""
        s = self.take(reader)
        n = s["collections"]
        if not sum(n):
            return "GC none"
        return (f"GC {sum(n)} (gen 0, 1, 2: {n[0]}, {n[1]}, {n[2]}), {s['pause_ms']:.1f} ms, the longest "
                f"{s['longest_ms']:.1f}")


STATS = GcStats()
_INSTALLED: list[bool] = []
_REPORTING: list[threading.Thread] = []


def install() -> GcStats:
    """STATS timing every collection of this process from now on (once)."""
    if not _INSTALLED:
        gc.callbacks.append(STATS.callback)
        _INSTALLED.append(True)
    return STATS


def freeze() -> tuple[int, float]:
    """The young garbage collected, then every object there is moved into the permanent generation, which collections
    leave alone (gc.freeze; those frozen before let go first, so a run made before is collected): (objects frozen, s
    taken)."""
    began = time.perf_counter()
    gc.unfreeze()
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count(), time.perf_counter() - began


def report_every(emit: Callable[..., None], seconds: float = REPORT_S, reader: str = "telemetry",
                 when: Callable[[], bool] = lambda: True) -> None:
    """A `gc` record (`take`'s) every `seconds` through `emit` (telemetry.emit) while `when()` says so
    (telemetry.to_file: never into a test's records), on a daemon thread of its own (once a process)."""
    install()
    if _REPORTING:
        return

    def run():
        STATS.take(reader)
        while True:
            time.sleep(seconds)
            try:
                stats = STATS.take(reader)
                if when():
                    emit("gc", **stats)
            except Exception:  # (telemetry never stops a run)
                pass
    thread = threading.Thread(target=run, name="gc-report", daemon=True)
    _REPORTING.append(thread)
    thread.start()
