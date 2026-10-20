"""The soak tool's own workings, briefly: the runs themselves take minutes (`python -m gamelab.sim.soak`)
and aren't part of pytest."""

import tracemalloc

import pytest

from gamelab import paths
from gamelab.sim import soak


def test_the_growth_an_hour_is_the_samples_line():
    assert soak.slope([(0, 10.0), (10, 30.0), (20, 50.0)]) == pytest.approx(2.0)
    assert soak.slope([(5, 1.0)]) == 0.0


def test_allocations_are_put_down_to_the_stack_the_simulator_or_the_rest():
    assert soak.part_of(str(paths.ROOT / "behavior" / "ledger.py")) == "stack"
    assert soak.part_of(str(paths.ROOT / "sim" / "fakes.py")) == "simulator"
    assert soak.part_of("<frozen importlib._bootstrap>") == "other"


def test_a_short_run_reports_what_grew(monkeypatch):
    monkeypatch.setattr(soak, "SAMPLE_S", 0.0)
    kept = []
    tracemalloc.start()
    try:
        meter = soak.Meter(0.3)
        while not meter.due():
            kept.append(bytearray(1000))  # what "leaks": 1 KB a step, from this file
            meter.sample()
        lines = soak.report(meter, "test", {"steps": len(kept)})
    finally:
        tracemalloc.stop()
    assert meter.warm is not None and meter.end is not None and len(meter.samples) >= 3
    assert any("test_soak.py" in line for line in lines[:-1]) and lines[-1] == f"held at the end: steps {len(kept)}"
