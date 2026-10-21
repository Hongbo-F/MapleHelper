"""Results of simulated hunts, and A/B comparisons of hunting policies on the same seeds.

    python -m gamelab.sim.harness run POLICY [--seeds 10] [--minutes 20] [--preset game] [--jobs 4] [--save NAME]
                                             [--feed] [--costs coded|measured]
    python -m gamelab.sim.harness compare BASELINE CANDIDATE [--bar rest]   two saved runs, seed by seed, against the bar
    python -m gamelab.sim.harness check                         the bookkeeping agrees with every scenario's world
    python -m gamelab.sim.harness rests [--trips 40]            trips to rest at low HP, transit vs --rest-avoid

A run hunts for a fixed time (the horizon) from one of START_AT, on a world seed, with the policy's own
random choices fixed (POLICY_SEED), so two policies meet the same monsters until their kills differ.
What a run gave is read from the simulated world, not from the bot's own reports (those miss kills):

  exp_per_min      EXP she got, per minute of the horizon (a death stops the count)
  kills, rival     her kills, and another player's (Realism.rival_every)
  hp_lost, hp_low  HP lost to touches (each as the world took it: by the monster's touch damage), and
                   her lowest HP; potions; died_at
  fell_out_at      when she fell below the map's lowest floor (world.SimFellOut): the run ends there, and
                   counts against the bar as a death does
  touches, hurt    monsters touching her, and those of the touches she didn't evade
  drops_share      of her drops whose fate is known by the end (picked up, or lain their whole life),
                   the share she picked up
  longest_idle     the longest stretch without a kill or a pick-up, and stuck_s: the time beyond 60 s
                   in such stretches
  rest_hp_lost     HP lost to touches on the trips to rest (from 'rest_trip' to 'rest' events), and
                   rest_trips: how many (the bar for `--rest-avoid`: compare --bar rest)
  backstops        how often each backstop cap fired (telemetry events kind 'backstop')

Saved runs go to gamelab/lab/harness/<name>.json (git-ignored, like all game data). Runs are made with
PYTHONHASHSEED=0 and her stats as measured (physics.FIXED_STATS), as golden traces are.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from ..paths import LAB, REPO
from .world import DROP_LIFE, Noise, Realism

FOLDER = LAB / "harness"
POLICY_SEED = 7   # the policy's own random choices (route weights, sweep order): the same in every run
START_AT = [(36, 300.0), (32, 600.0), (16, 585.0), (14, 760.0), (29, 420.0), (12, -300.0), (21, -300.0),
            (24, 120.0), (4, 450.0), (1, 1200.0)]  # where seed k starts: START_AT[k % len]
# Worlds to compare in. "plain" is the simulator's own (every spawn filled, 8 s respawns, nobody else,
# perfect eyes). "game" adds the game's errors and a less crowded, contested map; its values are guesses
# (nothing about respawns, other players or potions is measured), chosen to be harder than "plain".
PRESETS = {
    "plain": (Noise(), Realism()),
    "game": (Noise(miss=0.275, walk_jitter=6.0, lag_sd=1.0, tap_lost=0.6, dot_px=True, hit_hidden=(1.0, 3.0),
                   detect_miss=0.1),
             Realism(spawn_share=0.6, rival_every=25.0, potions=60)),
}
IDLE_GRACE = 60.0  # s without a kill or a pick-up that still counts as hunting
# What the simulator's looks take (sim.fakes), to compare against: "coded", the fakes' own (a look 100 ms, as
# the README had them before the speedups of 2026-10-20); "measured", today's, as bench-look and bench-capture
# measured them offline (17 saved frames x 5, medians, after the profiling pass): a look placing her 12.0 ms of
# computing plus the wait for a frame (5.6 ms median with the screen changing) and its copy off the card and
# conversion (2.0), her band 23.8 ms more, the whole screen 83.8, a strip of drops about 8 (her platform's
# 15.6 ms in 1.9 strips). Before the profiling pass (as step 5 measured them): 0.024, 0.006, 0.032, 0.122 and
# 0.008. Glances keep their pace (Eyes.GLANCE_EVERY_S). With `--feed`, her looks come from the feed, whose
# own costs are localization.feed.COSTS.
COSTS = {
    "coded": {},
    "measured": {"LOOK_S": 0.020, "LOOK_LAG": 0.006, "DETECT_S": 0.024, "DETECT_WIDE_S": 0.084, "LOOT_S": 0.008},
}


# --- policies ------------------------------------------------------------------------------------------

def _tour(sim) -> None:
    """Today's tour, as `python -m gamelab tour` runs it, started again whenever a sweep ends."""
    h = sim.hunter(low_hp=0.3, potion_hp=0.45)
    tour = sim.tour(h, sim.navigator(h, fight=True, seed=POLICY_SEED), seed=POLICY_SEED)
    while True:
        tour.run()


def _loop(sim, rest_avoid: bool = False) -> None:
    """The hunt loop (`python -m gamelab tour --loop`), on the same hunter, navigator and map knowledge."""
    from ..behavior.hunt import Hunt
    h = sim.hunter(low_hp=0.3, potion_hp=0.45)
    tour = sim.tour(h, sim.navigator(h, fight=True, seed=POLICY_SEED), seed=POLICY_SEED)
    tour.rest_avoid = rest_avoid
    while True:
        Hunt(tour, log=tour.log).run(minutes=600)


def _loop_avoid(sim) -> None:
    """The hunt loop with `--rest-avoid`: the trips to rest keep clear of monsters (navigator.TRANSIT_AVOID)."""
    _loop(sim, rest_avoid=True)


POLICIES = {"tour": _tour, "loop": _loop, "loop-avoid": _loop_avoid}


# --- one run -------------------------------------------------------------------------------------------

def scenario(policy: str, seed: int, preset: str = "game", minutes: float = 20.0, think: float = 0.0,
             feed: bool = False):
    """The Scenario for one run: the policy hunting from START_AT[seed] in world `seed` until the horizon.
    `think`: charge the stack's own computing to the fake clock (fakes.Clock; the runs then vary a little);
    `feed`: her looks and glances from a continuous feed (localization.feed.SyncFeed)."""
    from . import scenarios
    noise, realism = PRESETS[preset]
    fn = POLICIES[policy]
    return scenarios.Scenario(f"{policy}:{preset}:{seed}", fn, START_AT[seed % len(START_AT)], monsters="all",
                              seed=seed, limit=minutes * 60, noise=noise, realism=realism, may_die=True,
                              think=think, feed=feed)


def run_one(policy: str, seed: int, preset: str = "game", minutes: float = 20.0, think: float = 0.0,
            feed: bool = False, costs: str = "coded") -> dict:
    """Run one hunt and return what it gave (outcomes). `costs`: what the simulator's looks take (COSTS)."""
    from .. import telemetry
    from . import fakes, scenarios
    before = {name: getattr(fakes, name) for name in COSTS[costs]}
    try:
        for name, value in COSTS[costs].items():
            setattr(fakes, name, value)
        with telemetry.captured() as records:
            trace, sim = scenarios.run_sim(scenario(policy, seed, preset, minutes, think, feed))
    finally:
        for name, value in before.items():
            setattr(fakes, name, value)
    out = outcomes(sim.world, records, minutes * 60)
    out.update(policy=policy, seed=seed, preset=preset, minutes=minutes, think=think, feed=feed, costs=costs,
               result=str(trace["result"])[:120])
    if feed:
        f = trace["feed"]
        out["feed_record"] = {"frames": f["frames"], "asked": f["asked"], "busy_s": f["busy_s"],
                              "senses": f["senses"], "tracker": f["tracker"]}
    out["looks"] = sim.eyes.counts["looks"]
    return out


def outcomes(world, records: list[dict], horizon: float) -> dict:
    """What a run gave, from the simulated world (and the telemetry it recorded)."""
    end = min(horizon, world.died_at) if world.died_at is not None else horizon
    if world.fell_out_at is not None:
        end = min(end, world.fell_out_at)
    mine = [k for k in world.kill_log if k[3] == "her"]
    drops = [d for d in world.drops if d.owner == "her" and d.since <= horizon]
    known = [d for d in drops if d.taken_by == "her" or d.since + DROP_LIFE <= horizon]
    picked = [d for d in known if d.taken_by == "her"]
    times = sorted([0.0] + [k[0] for k in mine] + [d.taken_at for d in world.drops if d.taken_by == "her"] + [end])
    gaps = [b - a for a, b in zip(times, times[1:])]
    events = [r for r in records if r.get("topic") == "event"]
    trips = _rest_trips(records)
    return {
        "exp": world.exp,
        "exp_per_min": round(world.exp / (horizon / 60), 2),
        "kills": len(mine),
        "rival": sum(1 for k in world.kill_log if k[3] == "rival"),
        "hp_lost": round(sum(hp for _, hp in world.touch_log), 3),
        "touches": world.touches,
        "hurt": world.touches - world.evaded,
        "hp_low": round(world.hp_low, 3),
        "potions": world.potions_used,
        "died_at": world.died_at,
        "fell_out_at": world.fell_out_at,
        "drops": len(drops),
        "drops_share": round(len(picked) / len(known), 3) if known else None,
        "longest_idle": round(max(gaps, default=0.0), 1),
        "stuck_s": round(sum(max(0.0, g - IDLE_GRACE) for g in gaps), 1),
        "rests": sum(1 for r in events if r.get("kind") == "rest"),
        "rest_trips": len(trips),
        "rest_hp_lost": round(sum(hp for t, hp in world.touch_log
                                  if t <= end and any(a <= t <= b for a, b in trips)), 3),
        "backstops": dict(Counter(r.get("text", "").split(":")[0] for r in events if r.get("kind") == "backstop")),
    }


def _rest_trips(records: list[dict]) -> list[tuple[float, float]]:
    """(start, end) of each trip to rest, in world time (s from the run's start): from a 'rest_trip' event
    to the next 'rest' event (or the end)."""
    start = next((r["t"] for r in records if r.get("topic") == "session"), None)
    if start is None:
        return []
    out, began = [], None
    for r in records:
        if r.get("topic") != "event":
            continue
        if r.get("kind") == "rest_trip":
            began = r["t"] - start
        elif r.get("kind") == "rest" and began is not None:
            out.append((began, r["t"] - start))
            began = None
    if began is not None:
        out.append((began, float("inf")))
    return out


# --- many runs -----------------------------------------------------------------------------------------

def run_many(policy: str, seeds: int, preset: str, minutes: float, jobs: int = 1, think: float = 0.0,
             feed: bool = False, costs: str = "coded") -> list[dict]:
    args = [(policy, k, preset, minutes, think, feed, costs) for k in range(1, seeds + 1)]
    if jobs <= 1:
        return [run_one(*a) for a in args]
    from gamelab.sim import harness  # by its module name, so worker processes can find it (not __main__)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(harness._run_one_args, args))


def _run_one_args(a: tuple) -> dict:
    return run_one(*a)


COLUMNS = ("seed", "exp_per_min", "kills", "rival", "hp_lost", "hp_low", "potions", "died_at", "fell_out_at",
           "drops_share", "longest_idle", "stuck_s", "rests", "rest_hp_lost")


def table(results: list[dict]) -> str:
    rows = [" ".join(f"{c:>12}" for c in COLUMNS)]
    for r in results:
        rows.append(" ".join(f"{_fmt(r.get(c)):>12}" for c in COLUMNS))
    rows.append(" ".join(f"{_fmt(v):>12}" for v in ["mean"] + [_mean(results, c) for c in COLUMNS[1:]]))
    caps = Counter()
    for r in results:
        caps.update(r["backstops"])
    rows.append(f"backstops fired: {dict(caps) or 'none'}")
    return "\n".join(rows)


def _mean(results: list[dict], key: str):
    vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
    if key in ("died_at", "fell_out_at"):
        return sum(1 for r in results if r.get(key) is not None)  # deaths, falls out of the map
    return round(statistics.fmean(vals), 3) if vals else None


def _fmt(v) -> str:
    return "-" if v is None else str(v)


# --- the bar -------------------------------------------------------------------------------------------

MIN_PAIRS = 10  # the bars are for 10 seeds or more, in both runs


def _paired(base: list[dict], cand: list[dict]) -> tuple[list[dict], list[dict]]:
    """The runs of the seeds both have, in the same order (means over different worlds compare nothing)."""
    by_seed = {r["seed"]: r for r in base}
    pairs = [(by_seed[r["seed"]], r) for r in cand if r["seed"] in by_seed]
    return [a for a, _ in pairs], [b for _, b in pairs]


def compare(base: list[dict], cand: list[dict]) -> tuple[bool, list[str]]:
    """Whether `cand` passes the bar against `base` (seed by seed, over the seeds both have: MIN_PAIRS at
    least), and why: better EXP a minute on at least 8 of 10 seeds and no seed more than 10% worse; at
    least the drops share on average; no more HP lost on average; no death, and no fall out of the map."""
    every = cand  # deaths count on every seed, compared or not: "no death" isn't a comparison
    base, cand = _paired(base, cand)
    if len(cand) < MIN_PAIRS:
        return False, [f"{len(cand)} seed(s) in both runs: the bar needs {MIN_PAIRS}", "does NOT pass the bar"]
    pairs = list(zip(base, cand))
    lines, ok = [], True
    if len(every) > len(cand):
        lines.append(f"{len(every) - len(cand)} candidate seed(s) without a baseline: not compared")
    better = sum(1 for a, b in pairs if b["exp_per_min"] > a["exp_per_min"])
    worst = min((b["exp_per_min"] / a["exp_per_min"] - 1 for a, b in pairs if a["exp_per_min"]), default=0.0)
    need = -(-len(pairs) * 8 // 10)
    lines.append(f"EXP a minute better on {better} of {len(pairs)} seeds (need {need}); worst seed {worst:+.0%}")
    ok &= better >= need and worst >= -0.10
    for key, higher in (("drops_share", True), ("hp_lost", False)):
        a, b = _mean(base, key), _mean(cand, key)
        good = a is not None and b is not None and (b >= a if higher else b <= a)
        lines.append(f"{key}: {a} -> {b} ({'ok' if good else 'worse' if a is not None and b is not None else 'missing'})")
        ok &= good
    deaths = sum(1 for r in every if r.get("died_at") is not None)
    lines.append(f"deaths: {deaths}")
    ok &= deaths == 0
    fell = [r["seed"] for r in every if r.get("fell_out_at") is not None]
    if fell:
        lines.append(f"fell out of the map: seed(s) {', '.join(map(str, fell))}")
    ok &= not fell
    lines.append("PASSES the bar" if ok else "does NOT pass the bar")
    return ok, lines


def compare_rest(base: list[dict], cand: list[dict]) -> tuple[bool, list[str]]:
    """Step 7's bar (`--rest-avoid`) on 20-minute runs: no more HP lost on the trips to rest on average
    (over the seeds both have, and with trips to rest on both sides), and no death. EXP a minute and the
    drops share are shown, not judged. (`rests` measures the trips themselves, with far more of them.)"""
    every = cand
    base, cand = _paired(base, cand)
    if len(cand) < MIN_PAIRS:
        return False, [f"{len(cand)} seed(s) in both runs: the bar needs {MIN_PAIRS}", "does NOT pass the rest bar"]
    lines, ok = [], True
    a, b = _mean(base, "rest_hp_lost"), _mean(cand, "rest_hp_lost")
    trips = (sum(r.get("rest_trips") or 0 for r in base), sum(r.get("rest_trips") or 0 for r in cand))
    a = a if trips[0] else None  # no trip to rest measured on a side: nothing to judge
    b = b if trips[1] else None
    good = a is not None and b is not None and b <= a  # runs saved before rest_hp_lost existed don't count
    lines.append(f"rest_hp_lost: {a} -> {b} ({'ok' if good else 'worse' if a is not None and b is not None else 'missing'}); rest trips "
                 f"{_mean(base, 'rest_trips')} -> {_mean(cand, 'rest_trips')}")
    ok &= good
    deaths = sum(1 for r in every if r.get("died_at") is not None)
    lines.append(f"deaths: {deaths}")
    ok &= deaths == 0
    for key in ("exp_per_min", "drops_share", "hp_lost"):
        lines.append(f"{key}: {_mean(base, key)} -> {_mean(cand, key)} (shown only)")
    lines.append("PASSES the rest bar" if ok else "does NOT pass the rest bar")
    return ok, lines


def save(name: str, results: list[dict]) -> None:
    FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / f"{name}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


def load(name: str) -> list[dict]:
    return json.loads((FOLDER / f"{name}.json").read_text(encoding="utf-8"))


# --- trips to rest (step 7) ------------------------------------------------------------------------------

REST_FROM = [(36, 300.0), (36, 800.0), (17, 1100.0), (4, 450.0), (12, -300.0), (2, 0.0), (18, 300.0),
             (21, -300.0), (1, 1200.0), (34, -300.0)]  # hunting grounds, where a stop on "HP low" leaves her
REST_HP = 0.28  # under the hunter's 30%: the trip a stop on low HP starts


def rest_trip(avoid: bool, k: int) -> dict:
    """One trip to rest from REST_FROM[k], at REST_HP, in world 100 + k of the game preset: the touches that
    hurt her on the way (from the 'rest_trip' event to the 'rest' one: not while she sits) and the HP they
    took, how long the trip took (None if it never ended), and whether she died."""
    from .. import telemetry
    from . import scenarios
    noise, realism = PRESETS["game"]

    def fn(sim):
        tour = sim.tour_stack()
        tour.rest_avoid = avoid
        tour.rest(max_seconds=0.1)

    sc = scenarios.Scenario("rest", fn, REST_FROM[k % len(REST_FROM)], monsters="all", seed=100 + k, hp=REST_HP,
                            limit=300, noise=noise, realism=realism, may_die=True)
    with telemetry.captured() as records:
        _, sim = scenarios.run_sim(sc)
    w = sim.world
    a, b = (_rest_trips(records) or [(0.0, float("inf"))])[0]
    return {"avoid": avoid, "k": k, "hurt": sum(1 for t, hp in w.touch_log if hp and a <= t <= b),
            "hp_lost": sum(hp for t, hp in w.touch_log if a <= t <= b),
            "dead": w.dead, "seconds": round(b - a, 1) if b != float("inf") else None,
            "kills": sum(1 for t, _, _, who in w.kill_log if who == "her" and a <= t <= b)}


def _rest_trip_args(a: tuple) -> dict:
    return rest_trip(*a)


def rest_trials(trips: int, jobs: int = 1) -> tuple[bool, list[str]]:
    """`trips` trips to rest each way (transit, and transit with avoidance: `--rest-avoid`), paired by start
    and world. Step 7's bar: less HP lost on the way on average, and no death."""
    args = [(a, k) for a in (False, True) for k in range(trips)]
    if jobs <= 1:
        res = [rest_trip(*a) for a in args]
    else:
        from gamelab.sim import harness
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            res = list(pool.map(harness._rest_trip_args, args))
    lines, means = [], {}
    for avoid in (False, True):
        rs = [r for r in res if r["avoid"] == avoid]
        done = [r["seconds"] for r in rs if r["seconds"] is not None]
        means[avoid] = statistics.fmean(r["hp_lost"] for r in rs)
        lines.append(f"{'avoid' if avoid else 'transit'}: HP lost {means[avoid]:.1%} a trip, "
                     f"{statistics.fmean(done) if done else float('nan'):.1f} s, deaths {sum(r['dead'] for r in rs)}, "
                     f"kills on the way {sum(r['kills'] for r in rs)}, not there in time {len(rs) - len(done)}")
    ok = means[True] <= means[False] and not any(r["dead"] for r in res if r["avoid"])
    lines.append("--rest-avoid PASSES the rest bar" if ok else "--rest-avoid does NOT pass the rest bar")
    return ok, lines


# --- checking the bookkeeping ----------------------------------------------------------------------------

def check() -> int:
    """Run every scenario and check that the world's own logs, which outcomes() reads, agree with its
    summary: her kills, pick-ups and drops."""
    from . import scenarios
    bad = 0
    for name, sc in scenarios.SCENARIOS.items():
        trace, sim = scenarios.run_sim(sc)
        w, summary = sim.world, trace["world"]
        kills = sum(1 for k in w.kill_log if k[3] == "her")
        picked = sum(1 for d in w.drops if d.taken_by == "her")
        problems = []
        if kills != sum(summary["kills"].values()):
            problems.append(f"kills {kills} vs {summary['kills']}")
        if picked != sum(summary["picked"].values()):
            problems.append(f"picked {picked} vs {summary['picked']}")
        print(f"{'ok' if not problems else 'DIFFERS'} {name}" + (f": {'; '.join(problems)}" if problems else ""))
        bad += bool(problems)
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from ..motion.physics import FIXED_STATS
    # The same set order in every run, and her stats as measured, not as config.toml has them.
    if os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get(FIXED_STATS) != "1":
        env = dict(os.environ, PYTHONHASHSEED="0", **{FIXED_STATS: "1"})
        return subprocess.call([sys.executable, "-X", "utf8", "-m", "gamelab.sim.harness", *argv], env=env,
                               cwd=str(REPO))
    p = argparse.ArgumentParser(prog="python -m gamelab.sim.harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("policy", choices=sorted(POLICIES))
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--minutes", type=float, default=20.0)
    r.add_argument("--preset", choices=sorted(PRESETS), default="game")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--think", type=float, default=0.0,
                   help="charge the stack's computing to the fake clock, times this (1: this computer's speed)")
    r.add_argument("--save", help="name to save the results under (default: POLICY-PRESET, then -feed and -COSTS)")
    r.add_argument("--feed", action="store_true", help="her looks and glances from a continuous feed (SyncFeed)")
    r.add_argument("--costs", choices=sorted(COSTS), default="coded",
                   help="what the simulator's looks take: coded (the fakes' own), measured (today's)")
    c = sub.add_parser("compare")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--bar", choices=("hunt", "rest"), default="hunt",
                   help="hunt: the loop's bar (default); rest: --rest-avoid's (HP lost on the trips to rest)")
    sub.add_parser("check")
    t = sub.add_parser("rests")
    t.add_argument("--trips", type=int, default=40)
    t.add_argument("--jobs", type=int, default=1)
    args = p.parse_args(argv)
    from . import scenarios
    if not scenarios.available():
        print(f"map {scenarios.MAP_ID} isn't available on this computer (gamedata/data): nothing to run")
        return 2
    if args.cmd == "check":
        return check()
    if args.cmd == "rests":
        ok, lines = rest_trials(args.trips, args.jobs)
        print("\n".join(lines))
        return 0 if ok else 1
    if args.cmd == "compare":
        ok, lines = (compare_rest if args.bar == "rest" else compare)(load(args.baseline), load(args.candidate))
        print("\n".join(lines))
        return 0 if ok else 1
    results = run_many(args.policy, args.seeds, args.preset, args.minutes, args.jobs, args.think, args.feed, args.costs)
    print(table(results))
    name = args.save or (f"{args.policy}-{args.preset}" + ("-feed" if args.feed else "")
                         + (f"-{args.costs}" if args.costs != "coded" else ""))
    save(name, results)
    print(f"saved {FOLDER / (name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

