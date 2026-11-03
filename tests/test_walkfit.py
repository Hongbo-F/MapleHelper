"""analysis.walkfit over the records of 2026-11-02 (tests/fixtures/records/hour-cut): two stretches of the hour's tour
and perception records cut from that day's telemetry (00:37:50-00:38:15, the jumps 8->7 ending on 9 after walks of
106 and 126; 00:40:55-00:41:50, the arrival walk off 10 at 00:41:10.99), their records cut to the fields walkfit
reads."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from gamelab.analysis import walkfit

FIX = Path(__file__).parent / "fixtures" / "records"


def test_walkfit_on_the_hours_two_stretches():
    """The fixture's stretches: her steady walks at 146-151 against the 125 planned (her profile's SPEED 100 then),
    the jumps 8->7 of 00:37:57.8 and 00:38:06.6 taking off a median 20 late after walks of 106 and 126 (+0.9 against
    the engine at her walk: the lateness is her walk's), her steps
    1.19 times what was asked, and of 7 'arrived' trips 2 on another platform with no touch (00:41:10.99: off 10's end
    onto 12). Her walk by motion.steady's estimator (the braking tails left out, Tree's slopes taken out by the v83
    rule) 147.2 over the 5, her x speed 146.5; walkfit's own until 2026-11-02 (`raw`: tails in, no slopes) 146.4. The
    same measures over the whole hour: 149.4 (66 walks; raw 148.2 over 81), +7.5 / +24.9 late (108 legs), steps 1.19 (52
    walked), 18 of 210 'arrived' elsewhere, 15 untouched (README: "Records to measure by"). Her stat window's SPEED
    given (`--speed 100`, as that night's read): the client's formula's 150 and 120 against her 147.2 (SPEED 97.7 by it,
    -2.3) and 119.2 (99.2, -0.8): the check of the formula her planner plans by."""
    (f,) = walkfit.fit_runs([FIX / "hour-cut"], planned=125.0, speed=100.0)
    assert f["planned"] == 125.0 and f["planned_from"] == "--planned" and f["by"] == "perception"
    assert (f["panel"], f["panel_from"]) == (100.0, "--speed")
    w = f["walk"]
    assert w["n"] == 5 and w["median"] == pytest.approx(147.2, abs=0.05) and w["v"] == pytest.approx(146.5, abs=0.05)
    assert 146.0 < w["q1"] < w["q3"] < 151.0 and f["slopes"] == "v83's"
    assert w["ratio"] == pytest.approx(147.2 / 125, abs=0.002)
    assert w["speed"] == pytest.approx(97.7, abs=0.1)  # (the SPEED that walks 147.2: 147.2 / 1.25 - 20, the client's)
    assert w["formula"] == 150.0 and w["off"] == pytest.approx(-2.3, abs=0.05)
    t = f["takeoff"]
    assert t["n"] == 4 and t["long_n"] == 2 and t["long"] == pytest.approx(20.2, abs=0.05)
    assert t["engine_long"] == pytest.approx(0.9, abs=0.05)
    c = f["climb"]  # (her climbs there: 2 up, 1 down, 119.2 by the median; SPEED + 20 at her 100 is 120)
    assert c["n"] == 3 and c["median"] == pytest.approx(119.2, abs=0.05) and c["up"]["n"] == 2
    assert c["formula"] == 120.0 and c["speed"] == pytest.approx(99.2, abs=0.05) and c["off"] == pytest.approx(-0.8,
                                                                                                             abs=0.05)
    assert walkfit.formula_words(f) == (
        "the client's formula at her stat window's SPEED 100 walks her 150 and climbs her 120; measured her walk 147.2 "
        "over 5 (SPEED 97.7, -2.3), her climb 119.2 over 3 (SPEED 99.2, -0.8; up 118.6 over 2, down 119.6 over 1)")
    (raw,) = walkfit.fit_runs([FIX / "hour-cut"], planned=125.0, raw=True)
    assert raw["walk"]["n"] == 5 and raw["walk"]["median"] == pytest.approx(146.4, abs=0.05)
    assert raw["walk"]["speed"] == pytest.approx(97.1, abs=0.1) and raw["takeoff"]["engine_long"] == pytest.approx(1.5,
                                                                                                                 abs=0.05)
    assert f["steps"]["n"] == 4 and f["steps"]["ratio"] == pytest.approx(1.19, abs=0.005)
    assert f["arrived"] == {"n": 7, "elsewhere": 2, "untouched": 2}
    assert "20261102-001748-tour: the client's formula at her stat window's SPEED 100" in walkfit.table([f])


def test_walkfit_reads_the_planned_walk_off_the_runs_session_and_each_file_once(tmp_path):
    """A run from 2026-11-02 on says the walk it planned at (the session's physics) and each change of it (its
    'motion' records: her stat window's SPEED, behavior.hunter), the walk against the formula at that SPEED; a file met
    twice under one name is read once, its largest copy."""
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(FIX / "hour-cut", a)
    b.mkdir()
    tour = a / "20261102-001748-tour.jsonl"
    lines = tour.read_text(encoding="utf-8").splitlines()
    session = json.loads(lines[0])
    assert session["topic"] == "session"
    session["physics"] = {"character": "magician", "class": "magician", "speed": 100.0, "jump": 100.0, "bonus": 20.0,
                          "walk": 150.0, "climb": 120.0}
    tour.write_text("\n".join([json.dumps(session, separators=(",", ":"))] + lines[1:]) + "\n", encoding="utf-8")
    (b / tour.name).write_text(lines[0] + "\n" + lines[1] + "\n", encoding="utf-8")  # (a shorter copy: not read)
    (f,) = walkfit.fit_runs([b, a])
    assert f["planned"] == 150.0 and f["planned_from"].startswith("the run's records (magician)")
    assert f["walk"]["ratio"] == pytest.approx(147.2 / 150.0, abs=0.002) and f["arrived"]["n"] == 7
    assert (f["panel"], f["walk"]["formula"], f["climb"]["formula"]) == (100.0, 150.0, 120.0)  # (the session's SPEED)
    t = json.loads(lines[1])["t"]  # her stat window read 104 at the run's first record (a 'motion' record of it)
    motion = {"t": t, "topic": "motion", "old": [150.0, 120.0], "new": [155.0, 124.0], "speed": 104.0, "was": 100.0,
              "why": "panel"}
    tour.write_text("\n".join([json.dumps(session, separators=(",", ":")), json.dumps(motion, separators=(",", ":"))]
                              + lines[1:]) + "\n", encoding="utf-8")
    (f,) = walkfit.fit_runs([a])
    assert (f["panel"], f["panel_from"], f["walk"]["formula"]) == (104.0, "the runs' records", 155.0)
    run = walkfit.Run(tour, [])
    assert run.walk_at(t - 1.0, 150.0) == 150.0 and run.walk_at(t + 1.0, 150.0) == 155.0


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




def _scenes(legs: list[tuple[float, float, float]], end: float, dt: float = 0.133) -> list[tuple]:
    """Her scenes every `dt` s (the perception service's every other frame) on platform 7 until `end`, standing at x 0
    but for `legs`: (start, how long, units) walks at a steady pace."""
    out, t, x0 = [], 0.0, 0.0
    while t <= end:
        x = x0
        for a, length, units in legs:
            if t >= a:
                x += units * min(1.0, (t - a) / length)
        out.append((round(t, 3), x, 7, "ground", "tag"))
        t += dt
    return out


def test_a_step_said_again_before_she_stood_still_or_undone_isnt_one():
    """The stance says a step look after look while a knock holds it back (the hour's 'left 120' three times at
    00:50:13.8-14.0, then 'right 67' twice, the second walked): only the step she walked is counted, and one that left
    her the other way isn't (00:22:52.97's 'left 135', 9 units right); two steps walked within STEP_S of each other
    both are (00:38:25.5 and 00:38:26.6)."""
    said = [(10.0, "left 120"), (10.13, "left 120"), (10.26, "left 120"), (10.39, "right 67"), (10.52, "right 67")]
    stances = [{"t": t, "act": "back", "text": f"stepping away {x}"} for t, x in said]
    stances += [{"t": 11.6, "act": "throw", "text": "throwing right"},
                {"t": 20.0, "act": "back", "text": "stepping away left 41"},
                {"t": 21.1, "act": "back", "text": "stepping away left 30"},
                {"t": 22.0, "act": "throw", "text": "throwing left"},
                {"t": 30.0, "act": "back", "text": "stepping away left 135"},
                {"t": 31.0, "act": "wait", "text": "nothing in reach"}]
    scenes = _scenes([(10.7, 0.55, 80.0), (20.15, 0.35, -50.0), (21.25, 0.25, -36.0), (30.1, 0.1, 9.0)], 32.0)
    run = SimpleNamespace(stances=stances, scenes=scenes, st=[q[0] for q in scenes], touched=lambda a, b: False)
    got = walkfit.steps(run)
    assert [(n, round(g)) for n, g in got] == [(67.0, 80), (41.0, 50), (30.0, 36)]
