"""Golden-master traces of the simulator's scenarios: record what gamelab does today, check it still does
exactly the same after a change (same key plans, same log, same results).

    python -m gamelab.sim.golden record [names] [--jobs N] [--folder DIR]   write gamelab/lab/golden/<scenario>.json
    python -m gamelab.sim.golden check [names] [--jobs N] [--folder DIR]    run again and compare; exit code 1 on any
                                                                            difference
    python -m gamelab.sim.golden list                                       the scenarios, and which have a trace

`--jobs N` runs the scenarios on a pool of N processes, each worker made ready as the command makes itself (the same
environment, imports and maps loaded) and each scenario run there exactly as here, its trace sent back as the JSON
`record` writes; the longest first, by their recorded traces' size. What's printed is in the order the names were
given, as with one, and so are the exit codes: `check`'s output is the same to the byte whatever N. Default 1, in this
process, as it always was. `--folder DIR`: the traces written to (record) or read from (check) DIR, not
gamelab/lab/golden (a scratch folder, to compare two recordings byte for byte).

`check` exits 0 only when every scenario asked for ran and matched: 1 when one differs (or has no
recorded trace), 2 when nothing could be run (no game data for the map on this computer). Scenarios on a map
whose data isn't here (West Rocky Mountain II's, say) are named and left out.

The traces hold map-derived data, so they stay in gamelab/lab (git-ignored), like all game data.
Runs are made with PYTHONHASHSEED=0 (the command restarts itself with it), so nothing depends on the
order Python happens to keep a set of strings in, and with her stats as measured (physics.FIXED_STATS), so
nothing depends on what config.toml's [character] says.
"""

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..paths import LAB, REPO

FOLDER = LAB / "golden"
IGNORED = ("doc",)   # a scenario's description may change without the behaviour changing


def path_of(name: str, folder: Path | None = None) -> Path:
    return (folder or FOLDER) / f"{name}.json"


@dataclass
class Ran:
    """One scenario's run as `record` and `check` use it: its trace as `record` writes it (JSON), what of what it's
    there for it doesn't show (scenarios.unmet, on the trace as run), its counts for `record`'s line, and how long it
    took (s)."""
    text: str
    unmet: list[str]
    plans: int
    log: int
    seconds: float
    took: float
    out: str = ""  # what the run printed, a worker's (with --jobs: printed here in its place)
    err: str = ""


def run_one(name: str) -> Ran:
    """Run scenario `name` in this process (the command's own)."""
    from . import scenarios
    t = time.perf_counter()
    trace = scenarios.run(name)
    return Ran(scenarios.as_json(trace) + "\n", scenarios.unmet(trace), len(trace["plans"]), len(trace["log"]),
               trace["seconds"], time.perf_counter() - t)


def _run_in_worker(name: str) -> Ran:
    """run_one in a worker, what it prints kept for the command to print in the scenario's place."""
    import contextlib
    import io
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        r = run_one(name)
    r.out, r.err = out.getvalue(), err.getvalue()
    return r


def _ready(map_ids: tuple[int, ...]) -> None:
    """A worker made ready as main() makes the command: the scenarios imported and every map they're on loaded."""
    from . import scenarios
    scenarios.real_map()
    for map_id in map_ids:
        scenarios.real_map(map_id)


def ran(names: list[str], jobs: int = 1, folder: Path | None = None) -> Iterator[tuple[str, Ran]]:
    """Each scenario's run, in the order of `names`: here one after another (`jobs` 1), or on a pool of `jobs`
    processes, the longest first (by their recorded traces' size, in `folder` or else gamelab/lab/golden: the slow
    ones don't come last), each given back in order once it and those before it are done."""
    if jobs <= 1 or len(names) <= 1:
        for name in names:
            yield name, run_one(name)
        return
    from concurrent.futures import ProcessPoolExecutor
    from . import scenarios
    from gamelab.sim import golden  # by its module name, so the workers can find it (not __main__)
    maps = tuple(sorted({scenarios.SCENARIOS[n].map_id for n in names}))
    size = {n: next((p.stat().st_size for p in (path_of(n, folder), path_of(n)) if p.exists()), 0) for n in names}
    pool = ProcessPoolExecutor(max_workers=min(jobs, len(names)), initializer=golden._ready, initargs=(maps,))
    try:
        futures = {n: pool.submit(golden._run_in_worker, n) for n in sorted(names, key=lambda n: -size[n])}
        for name in names:
            r = futures[name].result()
            sys.stdout.write(r.out)
            sys.stderr.write(r.err)
            yield name, r
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def record(names: list[str], jobs: int = 1, folder: Path | None = None) -> int:
    """Record each scenario's trace (in `folder`, gamelab/lab/golden by default). Exit code 1 if one no longer shows
    what it's there for (its `expects`): the trace is still written, but the scenario needs another look."""
    (folder or FOLDER).mkdir(parents=True, exist_ok=True)
    lacking = 0
    for name, r in ran(names, jobs, folder):
        path = path_of(name, folder)
        path.write_text(r.text, encoding="utf-8")
        print(f"recorded {name}: {r.plans} plans, {r.log} log lines, "
              f"{r.seconds:.0f} s simulated ({r.took:.1f} s) -> {path}", flush=True)
        if r.unmet:
            lacking += 1
            print(f"   ... but it no longer shows what it's there for: {'; '.join(r.unmet)}", flush=True)
    return 1 if lacking else 0


def check(names: list[str], jobs: int = 1, folder: Path | None = None) -> int:
    failed = 0
    missing = {n for n in names if not path_of(n, folder).exists()}
    runs = ran([n for n in names if n not in missing], jobs, folder)
    for name in names:
        path = path_of(name, folder)
        if name in missing:
            print(f"MISSING {name}: no recorded trace ({path}); run `record` first")
            failed += 1
            continue
        want = json.loads(path.read_text(encoding="utf-8"))
        got = json.loads(next(runs)[1].text)  # through JSON, as recorded
        problems = diff(want, got)
        if problems:
            failed += 1
            print(f"DIFFERS {name}:")
            for line in problems:
                print("   " + line)
            sys.stdout.flush()
        else:
            print(f"ok      {name}", flush=True)
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


def _thief_only() -> str | None:
    """The scenarios are the thief's (a magician's come with the simulator's magic, the design's S6): a GAMELAB_CHARACTER
    naming another, inherited by the run that restarts below, would play them with her stats (her class's fixed copy),
    and `record` write that over the thief's traces. Why not, or None."""
    from ..character import DEFAULT, ENV
    who = os.environ.get(ENV)
    if who and who != DEFAULT:
        return (f"{ENV} names {who}: the simulator's scenarios are the thief's until hers come (the design's S6); "
                f"run it with {ENV} unset")
    return None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if refused := _thief_only():
        print(refused)
        return 2
    from ..motion.physics import FIXED_STATS
    # The same set order in every run, and her stats as measured, not as config.toml has them.
    if os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get(FIXED_STATS) != "1":
        env = dict(os.environ, PYTHONHASHSEED="0", **{FIXED_STATS: "1"})
        return subprocess.call([sys.executable, "-X", "utf8", "-m", "gamelab.sim.golden", *argv], env=env,
                               cwd=str(REPO))
    if not argv or argv[0] not in ("record", "check", "list"):
        print(__doc__)
        return 2
    try:
        argv, jobs, folder = _options(argv)
    except ValueError as e:
        print(e)
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
    away = [n for n in names if not scenarios.available(scenarios.SCENARIOS[n].map_id)]
    if away and argv[0] != "list":
        print(f"left out, their map's game data isn't on this computer: {', '.join(away)}")
        names = [n for n in names if n not in away]
    for map_id in {scenarios.SCENARIOS[n].map_id for n in names}:
        scenarios.real_map(map_id)  # (each map's: one that fails to load raises here)
    if argv[0] == "list":
        for name, sc in scenarios.SCENARIOS.items():
            print(f"{'*' if path_of(name).exists() else ' '} {name:28s} {sc.doc.splitlines()[0] if sc.doc else ''}")
        return 0
    return record(names, jobs, folder) if argv[0] == "record" else check(names, jobs, folder)


def _options(argv: list[str]) -> tuple[list[str], int, Path | None]:
    """argv without `--jobs N` and `--folder DIR` (or `--jobs=N`, `--folder=DIR`), and their values (1, None)."""
    rest, jobs, folder = [], 1, None
    it = iter(argv)
    for arg in it:
        name, eq, value = arg.partition("=")
        if name not in ("--jobs", "--folder"):
            rest.append(arg)
            continue
        if not eq:
            value = next(it, None)
            if value is None:
                raise ValueError(f"{name} needs a value")
        if name == "--folder":
            folder = Path(value).resolve()
        elif value.isdigit() and int(value) >= 1:
            jobs = int(value)
        else:
            raise ValueError(f"--jobs takes a number of processes, 1 or more (not {value!r})")
    return rest, jobs, folder


if __name__ == "__main__":
    raise SystemExit(main())
