"""What the perception service and the hunts write, written off their loops (from 2026-10-26): telemetry by its
recorder's thread (a write held up by the disk holds up no emit; past the queue's bound records are dropped and counted;
nothing lost or reordered otherwise; a full disk stops it, said once), the service's console by a thread of its own, and
Python's garbage collections timed (gcstats). Temp folders only."""

import gc
import sys
import threading
import time

from gamelab import gcstats, telemetry


class Held:
    """A file whose writes wait for `gate` (a disk that stalls), as the recorder's file."""

    def __init__(self, f, gate: threading.Event):
        self.f, self.gate, self.writes = f, gate, 0

    def write(self, text):
        self.gate.wait(10.0)
        self.writes += 1
        return self.f.write(text)

    def __getattr__(self, name):
        return getattr(self.f, name)


def test_a_write_held_up_by_the_disk_holds_up_no_emit_and_nothing_is_lost_or_reordered(tmp_path):
    path = telemetry.record_to("perceive", folder=tmp_path)
    gate = threading.Event()
    sink = telemetry._sink
    sink._file = Held(sink._file, gate)
    try:
        took = []
        for i in range(2000):  # the service's records for 3 minutes
            began = time.perf_counter()
            telemetry.emit("scene", i=i, monsters=[{"kind": "pig", "x": i}])
            took.append(time.perf_counter() - began)
        assert max(took) < 0.05 and sum(took) < 0.5  # (made into lines here, written there)
        assert not telemetry.flush(0.3)  # the disk holds it up: not written yet
        gate.set()
        assert telemetry.flush(10.0)
    finally:
        telemetry.stop()
    assert [r["i"] for r in telemetry.read(path) if r["topic"] == "scene"] == list(range(2000))
    assert sink.dropped == 0


def test_past_its_bound_the_newest_records_are_dropped_and_counted(tmp_path):
    path = telemetry.record_to("perceive", folder=tmp_path)
    gate = threading.Event()
    sink = telemetry._sink
    sink.keep = 50
    sink._file = Held(sink._file, gate)
    try:
        for i in range(300):
            telemetry.emit("log", text=f"line {i}")
        gate.set()
        assert telemetry.flush(10.0)
    finally:
        telemetry.stop()
    lines = [r["text"] for r in telemetry.read(path)]
    assert sink.dropped >= 300 - 50 - 50 and lines == [f"line {i}" for i in range(len(lines))]  # the oldest kept


def test_a_write_that_fails_stops_the_recording_saying_so_once(tmp_path, capsys):
    telemetry.record_to("hunt", folder=tmp_path)
    sink = telemetry._sink

    class Full:
        closed = False

        def write(self, text):
            raise OSError(28, "No space left on device")

        def flush(self):
            pass

        def close(self):
            pass
    sink._file = Full()
    telemetry.emit("log", text="one")
    assert not telemetry.flush(1.0) or sink.error is not None
    deadline = time.monotonic() + 5
    while sink.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    telemetry.emit("log", text="two")  # (the recorder stopped: emit lets it go, quietly)
    assert not telemetry.recording() and sink.error and "No space left" in sink.error
    assert capsys.readouterr().err.count("telemetry stopped") == 1


def test_the_consoles_text_is_written_by_its_thread_in_order_and_a_stalled_console_holds_up_no_print():
    out, gate = [], threading.Event()

    class Stream:
        encoding = "utf-8"

        def write(self, text):
            gate.wait(10.0)
            out.append(text)

        def flush(self):
            pass
    console = telemetry.Console()
    stream = telemetry._Queued(console, Stream())
    began = time.perf_counter()
    for i in range(500):
        print(f"line {i}", file=stream, flush=True)
    assert time.perf_counter() - began < 0.5 and not console.drain(0.3)
    gate.set()
    assert console.drain(5.0)
    assert "".join(out) == "".join(f"line {i}\n" for i in range(500))
    assert stream.encoding == "utf-8" and stream.write("x") == 1  # (the rest is the stream's own)


def test_the_collections_are_counted_and_timed_for_each_reader():
    stats = gcstats.install()
    stats.take("test")
    garbage = []
    for _ in range(3):
        a, b = [], []
        a.append(b)
        b.append(a)  # (a cycle: only a collection frees it)
        garbage.append(a)
    del garbage, a, b
    gc.collect()
    gc.collect(0)
    got = stats.take("test")
    assert got["collections"][2] >= 1 and sum(got["collections"]) >= 2 and got["pause_ms"] > 0
    assert got["longest_ms"] <= got["pause_ms"] and got["seconds"] >= 0
    assert stats.take("test")["collections"] in ([0, 0, 0], [1, 0, 0], [0, 1, 0])  # (since: one may have come)
    assert stats.line("test").startswith("GC ")
    frozen, took = gcstats.freeze()
    try:
        assert frozen > 0 and gc.get_freeze_count() > 0 and took >= 0  # (frozen ones freed since are let go)
    finally:
        gc.unfreeze()
    assert gcstats.STATS.callback in gc.callbacks and gc.callbacks.count(gcstats.STATS.callback) == 1
    gcstats.install()
    assert gc.callbacks.count(gcstats.STATS.callback) == 1 and sys.modules["gamelab.gcstats"] is gcstats


GC_ANYWHERE = """
import gc, sys, threading
sys.path.insert(0, sys.argv[1])
from gamelab import gcstats
gcstats.install()
junk, done = [], [0]
def work():
    for i in range(20000):
        junk.append([i])
        if len(junk) > 2000:
            junk.clear()
        gcstats.STATS.take("reader")
        done[0] += 1
gc.set_threshold(1)
t = threading.Thread(target=work, daemon=True)
t.start()
t.join(30)
gc.set_threshold(2000, 10, 10)
print("hung" if t.is_alive() else f"done {done[0]} {sum(gcstats.STATS.total)}", flush=True)
sys.stdout.flush()
import os
os._exit(0)
"""


def test_a_collection_starting_inside_a_take_never_hangs_the_process():
    """A collection can start at any allocation, `take`'s own too: with the collector's threshold at 1, take() in a loop
    beside the collections (the review of 2026-10-26 hung the first take so, the callback waiting for the lock its own
    thread held). In a process of its own: a hang would freeze every collection of this one."""
    import subprocess
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, "-c", GC_ANYWHERE, root], capture_output=True, text=True, timeout=120)
    words = out.stdout.split()
    assert words[:2] == ["done", "20000"] and int(words[2]) > 1000, (out.stdout, out.stderr[-2000:])
