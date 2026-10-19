"""Golden-master traces of the simulator's scenarios: record what gamelab does today, check it still does
exactly the same after a change (same key plans, same log, same results).

    python -m gamelab.sim.golden record [names]   write gamelab/lab/golden/<scenario>.json
    python -m gamelab.sim.golden check [names]    run again and compare; exit code 1 on any difference
    python -m gamelab.sim.golden list             the scenarios, and which have a recorded trace

`check` exits 0 only when every scenario asked for ran and matched: 1 when one differs (or has no
recorded trace), 2 when nothing could be run (no game data for the map on this computer).

The traces hold map-derived data, so they stay in gamelab/lab (git-ignored), like all game data.
Runs are made with PYTHONHASHSEED=0 (the command restarts itself with it), so nothing depends on the
order Python happens to keep a set of strings in.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..paths import LAB, REPO

FOLDER = LAB / "golden"
IGNORED = ("doc",)   # a scenario's description may change without the behaviour changing


def path_of(name: str) -> Path:
    return FOLDER / f"{name}.json"


def record(names: list[str]) -> int:
    """Record each scenario's trace. Exit code 1 if one no longer shows what it's there for (its
    `expects`): the trace is still written, but the scenario needs another look."""
    from . import scenarios
    FOLDER.mkdir(parents=True, exist_ok=True)
    lacking = 0
    for name in names:
        t = time.perf_counter()
        trace = scenarios.run(name)
        path_of(name).write_text(scenarios.as_json(trace) + "\n", encoding="utf-8")
        print(f"recorded {name}: {len(trace['plans'])} plans, {len(trace['log'])} log lines, "
              f"{trace['seconds']:.0f} s simulated ({time.perf_counter() - t:.1f} s) -> {path_of(name)}")
        if unmet := scenarios.unmet(trace):
            lacking += 1
            print(f"   ... but it no longer shows what it's there for: {'; '.join(unmet)}")
    return 1 if lacking else 0


def check(names: list[str]) -> int:
    from . import scenarios
    failed = 0
    for name in names:
        path = path_of(name)
        if not path.exists():
            print(f"MISSING {name}: no recorded trace ({path}); run `record` first")
            failed += 1
            continue
        want = json.loads(path.read_text(encoding="utf-8"))
        got = json.loads(scenarios.as_json(scenarios.run(name)))  # through JSON, as recorded
        problems = diff(want, got)
        if problems:
            failed += 1
            print(f"DIFFERS {name}:")
            for line in problems:
                print("   " + line)
        else:
            print(f"ok      {name}")
    print(f"{len(names) - failed} of {len(names)} scenario(s) match their recorded traces")
    if not names:
        return 2  # a check that compared nothing never passes
    return 1 if failed else 0


def diff(want: dict, got: dict) -> list[str]:
    """What differs between two traces, readably: per part, the first item that differs."""
    out = []
    for key in sorted(set(want) | set(got)):
        if key in IGNORED or want.get(key) == got.get(key):
            continue
        a, b = want.get(key), got.get(key)
        if isinstance(a, list) and isinstance(b, list):
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            label = {"plans": "plan", "log": "log line", "cancels": "cancel", "world_log": "world event"}.get(key, key)
            out.append(f"{key}: {len(a)} recorded, {len(b)} now; first difference at {label} #{i}:")
            out.append(f"   recorded: {_short(a[i]) if i < len(a) else '(none)'}")
            out.append(f"   now:      {_short(b[i]) if i < len(b) else '(none)'}")
            if key == "plans" and i < len(a) and i < len(b):
                out += _plan_diff(a[i], b[i])
        elif isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b), key=str):
                if a.get(k) != b.get(k):
                    out.append(f"{key}.{k}: recorded {_short(a.get(k))}, now {_short(b.get(k))}")
        else:
            out.append(f"{key}: recorded {_short(a)}, now {_short(b)}")
    return out


def _plan_diff(a: list, b: list) -> list[str]:
    """Where two plans ([t, name, events, end_ms]) part: the field, and for events the first one."""
    out = []
    for label, x, y in zip(("sent at", "name", "events", "end_ms"), a, b):
        if x == y:
            continue
        if label == "events" and isinstance(x, list) and isinstance(y, list):
            i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            out.append(f"   events differ from #{i}: recorded {x[i] if i < len(x) else '(none)'}, "
                       f"now {y[i] if i < len(y) else '(none)'}")
        else:
            out.append(f"   {label}: recorded {x}, now {y}")
    return out


def _short(value, limit: int = 300) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + " ..."


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if os.environ.get("PYTHONHASHSEED") != "0":  # the same set order in every run
        env = dict(os.environ, PYTHONHASHSEED="0")
        return subprocess.call([sys.executable, "-X", "utf8", "-m", "gamelab.sim.golden", *argv], env=env,
                               cwd=str(REPO))
    if not argv or argv[0] not in ("record", "check", "list"):
        print(__doc__)
        return 2
    from . import scenarios
    if not scenarios.available():
        print(f"map {scenarios.MAP_ID} isn't available on this computer (gamedata/data): nothing to run")
        return 2
    scenarios.real_map()  # data that's there but fails to load is an error, raised here
    names = argv[1:] or list(scenarios.SCENARIOS)
    unknown = [n for n in names if n not in scenarios.SCENARIOS]
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}; see `list`")
        return 2
    if argv[0] == "list":
        for name, sc in scenarios.SCENARIOS.items():
            print(f"{'*' if path_of(name).exists() else ' '} {name:28s} {sc.doc.splitlines()[0] if sc.doc else ''}")
        return 0
    return record(names) if argv[0] == "record" else check(names)


if __name__ == "__main__":
    raise SystemExit(main())
