"""analysis.walkfit over the records of 2026-11-02 (tests/fixtures/records/hour-cut): two stretches of the hour's tour
and perception records cut from that day's telemetry (00:37:50-00:38:15, the jumps 8->7 ending on 9 after walks of
106 and 126; 00:40:55-00:41:50, the arrival walk off 10 at 00:41:10.99), their records cut to the fields walkfit
reads."""

import json
import shutil
from pathlib import Path

import pytest

from gamelab.analysis import walkfit

FIX = Path(__file__).parent / "fixtures" / "records"


def test_walkfit_on_the_hours_two_stretches():
    """The fixture's stretches: her steady walks at 146-151 against the 125 planned (her profile's SPEED 100 then),
    the jumps 8->7 of 00:37:57.8 and 00:38:06.6 taking off a median 20 late after walks of 106 and 126 (+1.5 against
    the engine at her walk: the lateness is her walk's), her steps
    1.19 times what was asked, and of 7 'arrived' trips 2 on another platform with no touch (00:41:10.99: off 10's end
    onto 12). The same measures over the whole hour: 148.2 (81 walks), +7.5 / +24.9 late (108 legs; +0.1 / +3.0 against
    the engine at 148.2), steps 1.19 (68), 18 of 210 'arrived' elsewhere, 15 untouched (README: "Records to measure
    by")."""
    (f,) = walkfit.fit_runs([FIX / "hour-cut"], planned=125.0)
    assert f["planned"] == 125.0 and f["planned_from"] == "--planned" and f["by"] == "perception"
    w = f["walk"]
    assert w["n"] == 5 and w["median"] == pytest.approx(146.4, abs=0.05) and 146.0 < w["q3"] < 151.0
    assert w["ratio"] == pytest.approx(146.4 / 125, abs=0.002) and w["speed"] == pytest.approx(117.1, abs=0.1)
    t = f["takeoff"]
    assert t["n"] == 4 and t["long_n"] == 2 and t["long"] == pytest.approx(20.2, abs=0.05)
    assert t["engine_long"] == pytest.approx(1.5, abs=0.05)
    assert f["steps"]["n"] == 4 and f["steps"]["ratio"] == pytest.approx(1.19, abs=0.005)
    assert f["arrived"] == {"n": 7, "elsewhere": 2, "untouched": 2}
    assert "20261102-001748-tour" in walkfit.table([f])


def test_walkfit_reads_the_planned_walk_off_the_runs_session_and_each_file_once(tmp_path):
    """A run from 2026-11-02 on says the walk it planned at (the session's physics); a file met twice under one name
    is read once, its largest copy."""
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(FIX / "hour-cut", a)
    b.mkdir()
    tour = a / "20261102-001748-tour.jsonl"
    lines = tour.read_text(encoding="utf-8").splitlines()
    session = json.loads(lines[0])
    assert session["topic"] == "session"
    session["physics"] = {"character": "magician", "class": "magician", "speed": 119.2, "jump": 100.0, "walk": 149.0,
                          "climb": 117.0}
    tour.write_text("\n".join([json.dumps(session, separators=(",", ":"))] + lines[1:]) + "\n", encoding="utf-8")
    (b / tour.name).write_text(lines[0] + "\n" + lines[1] + "\n", encoding="utf-8")  # (a shorter copy: not read)
    (f,) = walkfit.fit_runs([b, a])
    assert f["planned"] == 149.0 and f["planned_from"].startswith("the run's records (magician)")
    assert f["walk"]["ratio"] == pytest.approx(146.4 / 149.0, abs=0.002) and f["arrived"]["n"] == 7


def test_walkfit_cuts_a_recordings_folder_to_its_frames(tmp_path):
    clip = tmp_path / "clip"
    shutil.copytree(FIX / "hour-cut", clip / "telemetry")
    frames = clip / "20261102-004202-clip"
    frames.mkdir()
    (frames / "index.jsonl").write_text(json.dumps({"i": 0, "at": "00:40:55.000", "seq": 1}) + "\n"
                                        + json.dumps({"i": 1, "at": "00:41:50.000", "seq": 2}) + "\n", encoding="utf-8")
    (f,) = walkfit.fit_runs([clip], planned=125.0, frames=True)
    assert f["run"] == "clip" and 0.8 < f["minutes"] <= 55 / 60  # (its records within the frames)
    assert f["arrived"]["elsewhere"] == 2 and f["takeoff"]["long_n"] == 0  # (the second stretch alone)


