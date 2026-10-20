"""How long the stack computes between two of its calls to the game, in simulated hunts: what a look's answer
waits for before the next key plan goes out, and the like. In game that time comes on top of the looks' own.

    python -m gamelab.sim.latency loop|tour [--seeds 5] [--minutes 10] [--preset game] [--profile FILE]

The simulator's clock is told of every call to the game (a look, a glance, a key plan sent or waited for,
a sleep: fakes.Clock.charge and .mark, which `--think` uses to charge the stack's computing to the clock).
Here they time each stretch of the stack's own computing in between, with the real clock, by the call that
ends it, without charging it: the runs are the same as without (the harness's), and their results are
printed with the times. `--profile` also writes cProfile's statistics of the whole runs (pstats).

Times are this computer's, with whatever else is running; the simulator's own work isn't counted (its world
moves only inside the calls), nor the looks' pixel work (FakeEyes; bench-look times that).
"""

import argparse
import os
import statistics
import subprocess
import sys
import time

from ..paths import REPO

_clock = time.perf_counter  # the real clock: the simulator puts its own in `time` while it runs
WHAT = {"_now": "a look or glance", "send": "a key plan", "wait": "waiting for a plan", "sleep": "a sleep or a search"}


class Stretches:
    """The stack's computing between two calls to the game, timed by what ended it."""

    def __init__(self):
        self.times: dict[str, list[float]] = {}
        self._since: float | None = None

    def mark(self) -> None:
        self._since = _clock()

    def charge(self, caller: str) -> None:
        if self._since is not None:
            self.times.setdefault(caller, []).append(_clock() - self._since)
            self._since = None

    def table(self) -> list[str]:
        """One line per kind of call: how many, the median, 90th and 99th percentile and longest (ms), the
        total (s); and all of them."""
        lines = [f"{'before':22s} {'n':>7s} {'median':>8s} {'p90':>8s} {'p99':>8s} {'max':>8s} {'total s':>8s}"]
        everything = []
        for name, xs in sorted(self.times.items(), key=lambda kv: -sum(kv[1])):
            lines.append(_row(WHAT.get(name, name), xs))
            everything += xs
        return lines + [_row("any call", everything)] if everything else lines


def _row(name: str, xs: list[float]) -> str:
    xs = sorted(xs)
    at = lambda q: 1000 * xs[min(len(xs) - 1, int(q * (len(xs) - 1)))]
    return (f"{name:22s} {len(xs):7d} {1000 * statistics.median(xs):8.2f} {at(0.9):8.2f} {at(0.99):8.2f} "
            f"{1000 * xs[-1]:8.2f} {sum(xs):8.2f}")


def measure(policy: str, seeds: list[int], minutes: float, preset: str = "game", profile=None) -> tuple:
    """Run the harness's hunts (sim.harness.run_one) with the stack's computing timed: (Stretches, outcomes)."""
    from . import fakes, harness
    found, charge, mark = Stretches(), fakes.Clock.charge, fakes.Clock.mark

    def charged(clock) -> None:
        name = sys._getframe(1).f_code.co_name
        found.charge(name if name in WHAT else "sleep")
        if clock.think:
            charge(clock)

    def marked(clock) -> None:
        found.mark()
        if clock.think:
            mark(clock)

    fakes.Clock.charge, fakes.Clock.mark = charged, marked
    try:
        results = []
        for seed in seeds:
            if profile is not None:
                profile.enable()
            try:
                results.append(harness.run_one(policy, seed, preset, minutes))
            finally:
                if profile is not None:
                    profile.disable()
    finally:
        fakes.Clock.charge, fakes.Clock.mark = charge, mark
    return found, results


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if os.environ.get("PYTHONHASHSEED") != "0":  # the same set order as the harness's runs
        env = dict(os.environ, PYTHONHASHSEED="0")
        return subprocess.call([sys.executable, "-X", "utf8", "-m", "gamelab.sim.latency", *argv], env=env,
                               cwd=str(REPO))
    from .harness import POLICIES, PRESETS
    p = argparse.ArgumentParser(prog="python -m gamelab.sim.latency", description=__doc__.splitlines()[0])
    p.add_argument("policy", choices=sorted(POLICIES))
    p.add_argument("--seeds", type=int, default=5, help="world seeds 1 to this")
    p.add_argument("--minutes", type=float, default=10.0, help="simulated minutes a seed")
    p.add_argument("--preset", choices=sorted(PRESETS), default="game")
    p.add_argument("--profile", help="also write cProfile's statistics of the runs here")
    args = p.parse_args(argv)
    from . import scenarios
    if not scenarios.available():
        print(f"map {scenarios.MAP_ID} isn't available on this computer (gamedata/data): nothing to run")
        return 2
    import cProfile
    profile = cProfile.Profile() if args.profile else None
    began = _clock()
    found, results = measure(args.policy, list(range(1, args.seeds + 1)), args.minutes, args.preset, profile)
    print(f"{args.policy}, preset {args.preset}, seeds 1-{args.seeds}, {args.minutes:g} simulated minutes each "
          f"({_clock() - began:.0f} s); the stack's computing (ms), by the call to the game that came next:")
    print("\n".join(found.table()))
    for r in results:
        print(f"  seed {r['seed']}: {r['exp_per_min']} EXP a minute, {r['kills']} kills, {r['looks']} looks")
    if profile is not None:
        profile.dump_stats(args.profile)
        print(f"profile: {args.profile} (python -m pstats {args.profile})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
