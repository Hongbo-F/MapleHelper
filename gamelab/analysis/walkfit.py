"""Her walk as a run's telemetry shows it, against the walk the run planned her moves at: offline, from any run's
records, read only (ROADMAP item 2 of the recordings of 2026-11-02, motion's M5a).

Every timed hold she makes runs open loop on the planned walk (motion.physics.WALK_SPEED: Physics.img's walkSpeed 125
x (her SPEED + 20)/100 from 2026-11-02, motion.body.SPEED_BONUS; 125 x SPEED/100 before), so a planned walk that isn't
hers sends every timed walk too far or too short. On 2026-11-02 RiceBaby was planned at 125 units/s (her stat window's
SPEED 100 by the v83 formula) and walked 148: her takeoffs came late by about a sixth of the
walk before them, her steps went 1.19 times what was asked, and 15 of the hour's 210 'arrived' trips had her off the end
of the platform the trip was going to (P01, P02: D:/MapleHelper-scratch/rec-analysis/plans). It went unnoticed from
10-29 to 11-02, and the analysis' scripts that found it were hard-wired to that day's files and to 125. This reads the
planned walk from the run's own records (the 'session' record's `physics`, motion.physics.record, from 2026-11-02;
older runs: `--planned`, else the profile of the character as the code has it now, said as assumed) and gives, per run
and per map:

- walk: her steady walk on the ground, units/s (median, quartiles, the median's standard error, how many): stretches
  of the perception service's scenes (every other frame, about 7.5 a second; the run's own looks if there are none,
  sparser) placing her by her name tag on one platform, moving one way at WALK_BAND units/s between scenes, for
  STEADY_S or more, their first SETTLE_S (her run-up) left out, a line fitted to the rest with no point FIT_UNITS off it.
  Against the planned: the ratio, and the profile's SPEED (%) that would plan it (walk / 1.25 - 20: the client's
  formula, motion.body.SPEED_BONUS).
- takeoff: where she left the ground on a chain whose first leg was a walk then a jump (the tour's 'chain' records: one
  span of TAKEOFF_LEAST or more), against where the plan had her take off (+: past it, the way she walked), for chains
  with no touch before the takeoff (the perception service's 'touch' events, at the touch's own moment: their time less
  `late`): the median as planned and for walks of LONG_WALK or more; the engine's own takeoff at her measured walk
  (motion.body: run_time at the planned walk to the plan's takeoff, run_x there at the measured), the residual it
  leaves; and the line `late = slope x walk + b`.
- steps: the stance's steps away ('stepping away left/right N' in the 'stance' records, act 'back'), touch-free, on one
  platform: where she first stood still after moving against the N asked, for N of STEP_LEAST or more (median ratio,
  quartiles, how many).
- arrived: the trips that ended 'arrived' (the 'trip' records), and those whose last look had her on another platform
  than the trip's (`on`, from 2026-11-02; before, the run's last 'scene' record before the trip's), with and without a
  touch in the TOUCH_BEFORE_S before.

The walk `--planned` or `--character` gives must be the one the run's keys were timed at: the engine's columns take it
as the plan (`--character magician` on a run keyed at 125 says only how her walk compares with today's 150; what a plan
at 150 would have made of its keys: D:/MapleHelper-scratch/recfix/walk/scripts/estimate.py, README "Her walk speed").
On R1-R3 of 2026-11-02 (`--planned 125`: they have no physics record, and her profile then planned 125) it
reproduces the analysis to the digit: the hour's steady walk 148.2 over 81 walks (quartiles 146.3-150.2, se 0.4), its
'arrived' trips 18 of 210 on another platform, 15 with no touch, its takeoffs +7.5 units late at the median and +24.9
after walks of 100 or more over 108 legs, +0.1 and +3.0 against the engine at 148.2, its steps 1.19 (quartiles
1.14-1.22) over the 68 of 30 units or more (the analysis' "71 steps" counted the 3 shorter ones too; its ratio was
over the same 68). The slope of lateness on walk is 0.219 over these 108 legs; the analysis' 0.164 was over 133, touched
legs and walks of 5-20 among them (README: "Records to measure by").

    python -m gamelab.analysis.walkfit RUN [RUN ...] [--planned UNITS] [--character ID] [--frames] [--json]

A RUN is a telemetry folder (its *.jsonl, and those of a telemetry/ folder in it: a golden clip's) or files: the hunt's
(a tour or hunt command's) and the perception service's (*-perceive.jsonl). A file met twice under one name (a clip's
copy of the service's day-long file, and a later clip's longer copy) is read once, its largest copy. `--frames`: each
RUN folder that holds a recording (a folder with an index.jsonl, control.rolling's) is cut to its frames' span, so
three clips of one hour come out as three runs. Nothing here sends a key or looks at the game.
"""

import argparse
import json
import re
import statistics
import sys
from bisect import bisect_left
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..motion import body

# The steady walk (profile_checks.py of the analysis): scenes this close in time (s) are one stretch (the perception
# service's, about 7.5 a second; the run's own looks, 2-4 a second, at GAP_LOOKS_S) ...
GAP_S = 0.2
GAP_LOOKS_S = 0.6
WALK_BAND = (90.0, 220.0)  # ... moving one way at this many units/s from scene to scene (any SPEED's walk: 125-200)
STEADY_S = 0.8             # ... for this long at least, over MIN_SCENES scenes
MIN_SCENES = 6
SETTLE_S = 0.25            # her run-up (full speed 0.106 s after the game sees the arrow, motion.body), left out
FIT_UNITS = 6.0            # units: no scene further than this off the fitted line (a step, a knock, a misplaced tag)
# The takeoffs (engine_fit.py): a first leg's walk this long at least, and a long one; a chain's takeoff looked for
# up to TAKEOFF_S after it (and before the next chain)
TAKEOFF_LEAST = 20.0
LONG_WALK = 100.0
TAKEOFF_S = 4.0
# The steps (r23-verify-movement/steps.py): touch-free from STEP_TOUCH_S before to STEP_S after, the ratio over steps
# asked STEP_LEAST or more, standing still = under STILL_UNITS from one scene to the next once moving
STEP_TOUCH_S = (0.3, 1.8)
STEP_S = 1.8
STEP_LEAST = 30.0
STILL_UNITS = 2.0
STEP_RE = re.compile(r"stepping away (left|right) (\d+)")
TOUCH_BEFORE_S = 3.0   # an 'arrived' trip with a touch this soon before its end was knocked, not walked, off


# --- reading the runs ----------------------------------------------------------------------------------------------

def _command(path: Path) -> str | None:
    """The command a telemetry file's session record names (its first few lines), or None."""
    try:
        with open(path, encoding="utf-8") as f:
            for _, line in zip(range(5), f):
                r = json.loads(line)
                if r.get("topic") == "session":
                    return r.get("command")
    except (OSError, ValueError):
        pass
    return None


def files_of(args: list[Path]) -> dict[str, Path]:
    """Every telemetry file the RUN arguments name, by file name: the largest copy of each (a day-long file copied
    into several clips is one file)."""
    found: dict[str, Path] = {}
    for a in args:
        if a.is_dir():
            paths = [*a.glob("*.jsonl"), *(a / "telemetry").glob("*.jsonl")]
        else:
            paths = [a]
        for p in paths:
            if p.name not in found or p.stat().st_size > found[p.name].stat().st_size:
                found[p.name] = p
    return found


def _records(path: Path, topics: tuple[str, ...], kinds: tuple[str, ...] = ()) -> list[dict]:
    """A file's records on `topics` (and 'event' records of `kinds`), its other lines skipped unparsed (by their text:
    telemetry's lines are compact JSON, `"topic":"scene"`)."""
    keys = [f'"topic":"{t}"' for t in topics]
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if any(k in line for k in keys) or (kinds and '"topic":"event"' in line
                                                 and any(f'"kind":"{k}"' in line for k in kinds)):
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def _her(r: dict) -> tuple | None:
    """A scene record's her: (t, x, platform, mode, how), None if it didn't place her."""
    h = r.get("her")
    if not h or h.get("x") is None:
        return None
    return r["t"], h["x"], h.get("platform"), h.get("mode"), h.get("how")


def recording_span(folder: Path) -> tuple[float, float] | None:
    """The span (epoch s) of the frames of a recording in `folder` (a folder in it with an index.jsonl: its first and
    last frame's local `at`, on the day its folder's name starts with), or None."""
    for index in sorted(folder.glob("*/index.jsonl")):
        named = re.match(r"(\d{8})-", index.parent.name)
        if not named:
            continue
        day = datetime.strptime(named.group(1), "%Y%m%d")
        lines = index.read_text(encoding="utf-8").splitlines()
        ats = [json.loads(x)["at"] for x in (lines[0], lines[-1]) if x.strip()]

        def epoch(at: str) -> float:
            t = datetime.strptime(at, "%H:%M:%S.%f")
            return (day + timedelta(hours=t.hour, minutes=t.minute, seconds=t.second,
                                    microseconds=t.microsecond)).timestamp()
        a, b = epoch(ats[0]), epoch(ats[-1])
        return a, (b if b >= a else b + 86400)  # (past midnight: the last frame's on the next day)
    return None


class Run:
    """One hunt's records (a tour or hunt command's file), over [t0, t1], and the perception service's over it."""

    def __init__(self, path: Path, perceives: list[Path], span: tuple[float, float] | None = None,
                 name: str | None = None):
        self.path, self.name = path, name or path.stem
        rows = _records(path, ("session", "scene", "chain", "trip", "stance"), ("touch",))
        session = next((r for r in rows if r.get("topic") == "session"), {})
        times = [r["t"] for r in rows if "t" in r and r.get("topic") != "session"]
        self.t0, self.t1 = (min(times), max(times)) if times else (0.0, 0.0)
        if span is not None:
            self.t0, self.t1 = max(self.t0, span[0]), min(self.t1, span[1])
        rows = [r for r in rows if self.t0 <= r.get("t", 0.0) <= self.t1 or r.get("topic") == "session"]
        self.map = session.get("map")
        self.physics = session.get("physics")
        self.chains = [r for r in rows if r.get("topic") == "chain" and r.get("legs")]
        self.stances = [r for r in rows if r.get("topic") == "stance"]
        self.looks = [r for r in rows if r.get("topic") == "scene"]  # (the hunt's own looks)
        self.trips: list[tuple[dict, dict | None]] = []  # each trip record, with the hunt's last look before it
        last = None
        for r in rows:
            if r.get("topic") == "scene":
                last = r
            elif r.get("topic") == "trip":
                self.trips.append((r, last))
        own_touches = sorted(r["t"] for r in rows if r.get("topic") == "event" and r.get("kind") == "touch")
        # The perception service's scenes and touches over the run (a few s either side: a chain's takeoff after it)
        self.scenes: list[tuple] = []
        touches: list[float] = []
        for p in perceives:
            for r in _records(p, ("scene",), ("touch",)):
                if not self.t0 - 5.0 <= r.get("t", 0.0) <= self.t1 + 5.0:
                    continue
                if r["topic"] == "scene":
                    if (h := _her(r)) is not None:
                        self.scenes.append(h)
                elif r.get("kind") == "touch":  # (when it came: told `late` after it)
                    touches.append(r["t"] - (r.get("late") or 0.0))
        self.by_service = bool(self.scenes)
        if not self.scenes:  # (no perception service's records: the run's own looks)
            self.scenes = [h for r in self.looks if (h := _her(r)) is not None]
        self.scenes.sort()
        self.touches = sorted(touches) if touches else own_touches
        self.st = [s[0] for s in self.scenes]

    @property
    def minutes(self) -> float:
        return (self.t1 - self.t0) / 60.0

    def touched(self, a: float, b: float) -> bool:
        i = bisect_left(self.touches, a)
        return i < len(self.touches) and self.touches[i] <= b


# --- the measures --------------------------------------------------------------------------------------------------

def steady_walks(scenes: list[tuple], t0: float, t1: float, gap: float = GAP_S) -> list[float]:
    """Her steady walks' speeds (units/s) in `scenes` (t, x, platform, mode, how) from t0 to t1 (see the top)."""
    s = [q for q in scenes if t0 <= q[0] <= t1]
    out: list[float] = []

    def flush(seg: list[tuple]) -> None:
        if len(seg) < MIN_SCENES or seg[-1][0] - seg[0][0] < STEADY_S:
            return
        t = np.array([q[0] for q in seg])
        x = np.array([q[1] for q in seg])
        m = t >= t[0] + SETTLE_S
        if m.sum() < 4:
            return
        k = np.polyfit(t[m], x[m], 1)
        if np.abs(x[m] - np.polyval(k, t[m])).max() < FIT_UNITS:
            out.append(abs(float(k[0])))

    def ok(q: tuple) -> bool:
        return q[3] == "ground" and q[4] == "tag"

    runs, cur = [], []
    for q in s:  # stretches on one platform, by her tag, the scenes GAP_S apart at most
        if ok(q) and (not cur or (q[0] - cur[-1][0] < gap and q[2] == cur[-1][2])):
            cur.append(q)
        else:
            if len(cur) >= 5:
                runs.append(cur)
            cur = [q] if ok(q) else []
    if len(cur) >= 5:
        runs.append(cur)
    for c in runs:  # ... cut into the stretches moving one way at a walk
        seg = [c[0]]
        for a, b in zip(c, c[1:]):
            dx, dt = b[1] - a[1], b[0] - a[0]
            v = dx / dt if dt > 0 else 0.0
            if WALK_BAND[0] < abs(v) < WALK_BAND[1] and (len(seg) < 2 or (seg[-1][1] - seg[-2][1]) * dx > 0):
                seg.append(b)
            else:
                flush(seg)
                seg = [b]
        flush(seg)
    return out


def takeoffs(run: Run) -> list[tuple[float, float, int]]:
    """The chains whose first leg is a walk (TAKEOFF_LEAST or more) then a jump, touch-free to the takeoff: (the walk,
    where she took off past the plan's takeoff, the way she walked: + late, the walk's direction)."""
    rows = []
    for i, c in enumerate(run.chains):
        leg = c["legs"][0]
        if leg.get("kind") != "jump" or len(leg.get("spans", ())) != 1:
            continue
        x0, x1 = leg["spans"][0][1], leg["spans"][0][2]
        walk, d = abs(x1 - x0), 1 if x1 > x0 else -1
        if walk < TAKEOFF_LEAST:
            continue
        nxt = run.chains[i + 1]["t"] if i + 1 < len(run.chains) else c["t"] + TAKEOFF_S
        a, b = bisect_left(run.st, c["t"]), bisect_left(run.st, min(nxt, c["t"] + TAKEOFF_S))
        seg = run.scenes[a:b]
        take = next(((u, v) for u, v in zip(seg, seg[1:]) if u[3] == "ground" and v[3] == "air"), None)
        if take is None or run.touched(c["t"], take[1][0]):
            continue
        rows.append((walk, ((take[0][1] + take[1][1]) / 2 - x1) * d, d))
    return rows


def _shoe(walk: float, jump: float = 100.0) -> body.Shoe:
    """The engine's shoe that walks `walk` units/s, whatever SPEED gives it (Physics.img's otherwise)."""
    return replace(body.Shoe.from_physics(body.PHYSICS_IMG, 100.0, jump), vmax=float(walk))


def engine_residuals(rows: list[tuple], planned: float, measured: float, jump: float = 100.0) -> list[float]:
    """Each takeoff's residual against the engine walking `measured` (motion.body): the plan's keys put her jump at the
    time the planned walk reaches its takeoff; where the measured walk has her then, against where she took off."""
    sp, sm = _shoe(planned, jump), _shoe(measured, jump)
    out = []
    for walk, seen, d in rows:
        tau = body.run_time(walk, 0.0, d, sp)
        if tau is None:
            continue
        out.append(seen - (body.run_x(tau, 0.0, d, sm) * d - walk))
    return out


def steps(run: Run) -> list[tuple[float, float]]:
    """The stance's steps away, touch-free, on one platform: (units asked, units gone to where she first stood still)."""
    out = []
    for r in run.stances:
        m = STEP_RE.search(r.get("text") or "")
        if r.get("act") != "back" or not m:
            continue
        d, n, t = (-1 if m.group(1) == "left" else 1), float(m.group(2)), r["t"]
        if run.touched(t - STEP_TOUCH_S[0], t + STEP_TOUCH_S[1]):
            continue
        i = bisect_left(run.st, t)
        if i >= len(run.scenes):
            continue
        seg = [q for q in run.scenes[i:i + 20] if q[0] <= t + STEP_S]
        if not seg or any(q[3] != "ground" or q[2] != run.scenes[i][2] for q in seg):
            continue
        xs, moved, end = [q[1] for q in seg], False, None
        for u, v in zip(xs, xs[1:]):
            if abs(v - u) > STILL_UNITS:
                moved = True
            elif moved:
                end = v
                break
        if end is not None:
            out.append((n, (end - xs[0]) * d))
    return out


def arrivals(run: Run) -> list[dict]:
    """The trips that ended 'arrived': the platform it was going to (`dst`), the one her last look had her on (`on`:
    the trip record's own, else the run's last look before it), whether a touch came TOUCH_BEFORE_S before."""
    out = []
    for r, last in run.trips:
        if r.get("result") != "arrived":
            continue
        on = r["on"] if "on" in r else (None if last is None or not last.get("her") else last["her"].get("platform"))
        out.append({"t": r["t"], "dst": r.get("dst"), "on": on, "touched": run.touched(r["t"] - TOUCH_BEFORE_S, r["t"]),
                    "walk": r.get("walk")})
    return out


# --- the fit -------------------------------------------------------------------------------------------------------

def planned_walk(run: Run, planned: float | None = None, character: str | None = None) -> tuple[float, float, str]:
    """The walk (units/s) and JUMP (%) run `run` planned her moves at, and where that's from: its records' (the
    'session' record's physics), else `planned` (--planned), else her profile as the code has it now (`character`,
    else the one the lab plays: an assumption, said so)."""
    ph = run.physics or {}
    if ph.get("walk"):
        return float(ph["walk"]), float(ph.get("jump", 100.0)), f"the run's records ({ph.get('character', '?')})"
    if planned is not None:
        return float(planned), 100.0, "--planned"
    from .. import character as profiles
    which = character or profiles.active_id()
    stats = profiles.load(which).stats
    walk = body.Shoe.from_physics(body.PHYSICS_IMG, stats.speed, stats.jump, bonus=body.SPEED_BONUS).vmax
    return walk, stats.jump, f"assumed: {which}'s profile in the code now (SPEED {stats.speed:g}: {walk:g})"


def _quartiles(v: list[float]) -> tuple[float, float]:
    return float(np.percentile(v, 25)), float(np.percentile(v, 75))


def fit(runs: list[Run], planned: float | None = None, character: str | None = None, name: str | None = None) -> dict:
    """The measures over `runs` (one run, or a map's), against the walk planned (the first run's: see planned_walk)."""
    plans = [planned_walk(r, planned, character) for r in runs]  # (each run's own: a map's may mix them)
    one = len({round(p[0], 1) for p in plans}) == 1
    walk_p, source = (plans[0][0], plans[0][2]) if one else (None, "mixed: " + "; ".join(sorted({p[2] for p in plans})))
    walks = sorted(w for r in runs
                   for w in steady_walks(r.scenes, r.t0, r.t1, GAP_S if r.by_service else GAP_LOOKS_S))
    out: dict = {"run": name or runs[0].name, "map": runs[0].map, "minutes": round(sum(r.minutes for r in runs), 2),
                 "by": "perception" if all(r.by_service for r in runs) else "looks",
                 "planned": None if walk_p is None else round(walk_p, 1), "planned_from": source}
    w: dict = {"n": len(walks)}
    if walks:
        med = statistics.median(walks)
        q1, q3 = _quartiles(walks)
        w.update(median=round(med, 1), q1=round(q1, 1), q3=round(q3, 1),
                 se=round(statistics.stdev(walks) / len(walks) ** 0.5, 2) if len(walks) > 1 else None,
                 ratio=None if walk_p is None else round(med / walk_p, 3),
                 speed=round(med / body.PHYSICS_IMG["walkSpeed"] * 100.0 - body.SPEED_BONUS, 1))
    out["walk"] = w
    legs = [(takeoffs(r), p) for r, p in zip(runs, plans)]
    rows = [x for got, _ in legs for x in got]
    t: dict = {"n": len(rows)}
    if rows:
        late = [x[1] for x in rows]
        long_ = [x[1] for x in rows if x[0] >= LONG_WALK]
        t.update(planned=round(statistics.median(late), 1), long_n=len(long_),
                 long=round(statistics.median(long_), 1) if long_ else None)
        if len(rows) >= 3:
            k, b = np.polyfit([x[0] for x in rows], late, 1)
            t.update(slope=round(float(k), 3), intercept=round(float(b), 1))
        if walks:  # (each run's legs against the walk it planned)
            res = [v for got, p in legs for v in engine_residuals(got, p[0], w["median"], p[1])]
            res_long = [v for got, p in legs for v in engine_residuals([x for x in got if x[0] >= LONG_WALK], p[0],
                                                                         w["median"], p[1])]
            t.update(engine=round(statistics.median(res), 1) if res else None,
                     engine_long=round(statistics.median(res_long), 1) if res_long else None)
    out["takeoff"] = t
    st = [x for r in runs for x in steps(r)]
    ratios = sorted(went / asked for asked, went in st if asked >= STEP_LEAST)
    s: dict = {"n": len(ratios)}
    if ratios:
        q1, q3 = _quartiles(ratios)
        s.update(ratio=round(statistics.median(ratios), 2), q1=round(q1, 2), q3=round(q3, 2))
    out["steps"] = s
    arr = [x for r in runs for x in arrivals(r)]
    wrong = [x for x in arr if x["on"] is not None and x["on"] != x["dst"]]
    out["arrived"] = {"n": len(arr), "elsewhere": len(wrong), "untouched": sum(1 for x in wrong if not x["touched"])}
    return out


def fit_runs(args: list[Path], planned: float | None = None, character: str | None = None,
             frames: bool = False) -> list[dict]:
    """walkfit over the RUN arguments: a fit per run, then one per map over the runs on it (when there are several)."""
    found = files_of(args)
    perceives = [p for n, p in sorted(found.items()) if "-perceive" in n or _command(p) == "perceive"]
    hunts = {n: p for n, p in sorted(found.items()) if p not in perceives
             and (_command(p) or "").split("-")[0] in ("tour", "hunt", "nav")}
    runs: list[Run] = []
    if frames:
        for a in args:
            span = recording_span(a) if a.is_dir() else None
            if span is None:
                continue
            for n, p in hunts.items():
                run = Run(p, perceives, span, name=a.name)
                if run.t1 > run.t0:
                    runs.append(run)
    else:
        runs = [Run(p, perceives) for p in hunts.values()]
        runs = [r for r in runs if r.t1 > r.t0]
    out = [fit([r], planned, character) for r in runs]
    by_map = defaultdict(list)
    for r in runs:
        by_map[r.map].append(r)
    for m, rs in by_map.items():
        if len(rs) > 1:
            out.append(fit(rs, planned, character, name=f"map {m}: all {len(rs)}"))
    return out


def table(fits: list[dict]) -> str:
    """The fits as a table, a line each."""
    head = (f"{'run':40s} {'min':>5s} {'plan':>5s} | {'walk':>5s} {'IQR':>11s} {'se':>4s} {'n':>3s} {'x':>5s} "
            f"{'SPEED':>5s} | {'late':>5s} {'>=100':>5s} {'n':>4s} {'eng':>5s} {'eng100':>6s} {'slope':>5s} | "
            f"{'step':>4s} {'IQR':>9s} {'n':>3s} | {'arrived':>7s} {'off':>3s} {'clean':>5s}")
    lines = [head]

    def f(v, w, p=1):
        return f"{'-':>{w}s}" if v is None else f"{v:{w}.{p}f}"
    for x in fits:
        w, t, s, a = x["walk"], x["takeoff"], x["steps"], x["arrived"]
        iqr = "-" if "q1" not in w else f"{w['q1']:.1f}-{w['q3']:.1f}"
        siqr = "-" if "q1" not in s else f"{s['q1']:.2f}-{s['q3']:.2f}"
        lines.append(f"{x['run'][:40]:40s} {x['minutes']:5.1f} {f(x['planned'], 5)} | {f(w.get('median'), 5)} "
                     f"{iqr:>11s} {f(w.get('se'), 4)} {w['n']:3d} {f(w.get('ratio'), 5, 3)} {f(w.get('speed'), 5)} | "
                     f"{f(t.get('planned'), 5)} {f(t.get('long'), 5)} {t['n']:4d} {f(t.get('engine'), 5)} "
                     f"{f(t.get('engine_long'), 6)} {f(t.get('slope'), 5, 3)} | {f(s.get('ratio'), 4, 2)} {siqr:>9s} "
                     f"{s['n']:3d} | {a['n']:7d} {a['elsewhere']:3d} {a['untouched']:5d}")
    sources = sorted({(x["run"], x["planned_from"], x["by"]) for x in fits})
    lines.append("")
    lines += [f"  {r}: planned walk from {src}; positions by the {by}'s scenes" for r, src, by in sources]
    lines.append("  walk: units/s (x: against the planned; SPEED: the profile's % that plans it); late: units past the "
                 "planned takeoff (>=100: after walks of 100+; eng: against the engine at her measured walk); step: "
                 "gone / asked; arrived: 'arrived' trips, off: on another platform, clean: of those with no touch "
                 f"{TOUCH_BEFORE_S:g} s before")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gamelab.analysis.walkfit", description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="telemetry folders or files")
    ap.add_argument("--planned", type=float, help="the walk (units/s) a run without a physics record planned at")
    ap.add_argument("--character", help="... else this character's profile as the code has it now")
    ap.add_argument("--frames", action="store_true", help="each recording's folder cut to its frames' span")
    ap.add_argument("--json", action="store_true", help="the fits as JSON")
    a = ap.parse_args(argv)
    fits = fit_runs(a.runs, a.planned, a.character, a.frames)
    if not fits:
        print("no hunt's telemetry found (a tour or hunt command's *.jsonl)")
        return 1
    print(json.dumps(fits, indent=1) if a.json else table(fits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
