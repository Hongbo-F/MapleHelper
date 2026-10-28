"""golden's runs on a pool of processes (`--jobs`): the same traces to the byte and the same output as in one process."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gamelab.sim import golden, scenarios


def test_golden_takes_jobs_and_a_folder():
    assert golden._options(["check", "a", "--jobs", "3", "b", "--folder=x"]) == (["check", "a", "b"], 3,
                                                                                 Path("x").resolve())
    assert golden._options(["record"]) == (["record"], 1, None)
    for bad in (["check", "--jobs", "0"], ["check", "--jobs"], ["check", "--jobs=two"]):
        with pytest.raises(ValueError):
            golden._options(bad)


SMALL = ("move_walk_off", "hunt_kill_unreachable", "move_jump")  # three of the shortest recorded scenarios


def _golden(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "gamelab.sim.golden", *args], capture_output=True, text=True,
                          cwd=str(Path(golden.__file__).resolve().parents[2]), encoding="utf-8")


def test_golden_on_a_pool_records_and_checks_as_one_process_does(tmp_path):
    if not scenarios.available() or any(n not in scenarios.SCENARIOS for n in SMALL):
        pytest.skip("needs the first map's game data and its scenarios")
    one, two = tmp_path / "one", tmp_path / "two"
    a = _golden("record", *SMALL, "--folder", str(one))
    b = _golden("record", *SMALL, "--folder", str(two), "--jobs", "2")
    assert a.returncode == b.returncode == 0, a.stdout + a.stderr + b.stdout + b.stderr
    for name in SMALL:
        assert (one / f"{name}.json").read_bytes() == (two / f"{name}.json").read_bytes()
    assert [line.split(":")[0] for line in b.stdout.splitlines()] == [f"recorded {n}" for n in SMALL]  # in order
    trace = json.loads((two / f"{SMALL[1]}.json").read_text(encoding="utf-8"))
    trace["log"].append("a line it never logged")
    (two / f"{SMALL[1]}.json").write_text(json.dumps(trace), encoding="utf-8")
    c1 = _golden("check", *SMALL, "--folder", str(two))
    c2 = _golden("check", *SMALL, "--folder", str(two), "--jobs", "3")
    assert c1.returncode == c2.returncode == 1 and c1.stdout == c2.stdout
    assert f"DIFFERS {SMALL[1]}:" in c2.stdout and f"ok      {SMALL[0]}" in c2.stdout
    assert "2 of 3 scenario(s) match" in c2.stdout
