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

- walk: her steady walk on flat ground, units/s (median, quartiles, the median's standard error, how many), by the
  estimator her live calibration has (motion.calibrate.Walks: from 2026-11-02, one estimator for both): stretches of
  the perception service's scenes (every other frame, about 7.5 a second; the run's own looks if there are none,
  sparser) placing her by her name tag or the camera on one platform, moving one way between scenes at a walk, their
  run-up and braking tail left out, a line fitted to the rest, Tree Dungeon's slopes taken out by the v83 rule (the
  map's platforms, where its model loads; `--raw`: her x speed with each walk's braking tail in it, as before
  2026-11-02's calibration); `v` her x speed's median. Against the planned: the ratio, and the SPEED (%) it implies
  (walk / 1.25 - 20: the client's formula, motion.body.SPEED_BONUS) against her stat window's (`panel`: the run's
  'calibration' records', else `--speed`, else her profile's in the session's physics), within motion.calibrate's
  SPEED_TOL or not; and what her calibration would plan from them (`calibrated`: MIN_N and MAX_SE, else her prior).
- climb: her climbs up and down a rope or ladder (motion.calibrate.Climbs), the same way: the SPEED they imply
  (climb - 20) against her stat window's.
- takeoff: where she left the ground on a chain whose first leg was a walk then a jump (the tour's 'chain' records: one
  span of TAKEOFF_LEAST or more), against where the plan had her take off (+: past it, the way she walked), for chains
  with no touch before the takeoff (the perception service's 'touch' events, at the touch's own moment: their time less
  `late`): the median as planned and for walks of LONG_WALK or more; the engine's own takeoff at her measured walk
  (motion.body: run_time at the planned walk to the plan's takeoff, run_x there at the measured), the residual it
  leaves; and the line `late = slope x walk + b`. The planned walk a chain's as of its time: the session's, then each
  'calibration' record's (motion.calibrate: her planner's walk changed mid-run).
- steps: the stance's steps away ('stepping away left/right N' in the 'stance' records, act 'back'), touch-free, on one
  platform, the ones she walked (no stance record before she stood still, gone the way asked: `steps`): where she
  first stood still after moving against the N asked, for N of STEP_LEAST or more (median ratio, quartiles, how many).
- arrived: the trips that ended 'arrived' (the 'trip' records), and those whose last look had her on another platform
  than the trip's (`on`, from 2026-11-02; before, the run's last 'scene' record before the trip's), with and without a
  touch in the TOUCH_BEFORE_S before.

The walk `--planned` or `--character` gives must be the one the run's keys were timed at: the engine's columns take it
as the plan (`--character magician` on a run keyed at 125 says only how her walk compares with today's 150; what a plan
at 150 would have made of its keys: D:/MapleHelper-scratch/recfix/walk/scripts/estimate.py, README "Her walk speed").
On R1-R3 of 2026-11-02 (`--planned 125`: they have no physics record, and her profile then planned 125) it
reproduces the analysis to the digit: the hour's steady walk 148.2 over 81 walks (quartiles 146.3-150.2, se 0.4), its
'arrived' trips 18 of 210 on another platform, 15 with no touch, its takeoffs +7.5 units late at the median and +24.9
after walks of 100 or more over 108 legs, +0.1 and +3.0 against the engine at 148.2. Its steps: 1.19 (quartiles
1.16-1.23) over the 52 of 30 units or more she walked; the analysis' steps.py had 1.19 (1.14-1.22) over 68, each
'stepping away' record a step (its "71 steps" the 3 shorter ones too), 15 of them said again before she stood still and
one undone (`steps`). The slope of lateness on walk is 0.219 over these 108 legs; the analysis' 0.164 was over 133,
touched legs and walks of 5-20 among them (README: "Records to measure by").

    python -m gamelab.analysis.walkfit RUN [RUN ...] [--planned UNITS] [--character ID] [--speed SPEED] [--frames]
                                      [--raw] [--json] [--seed ID [--to FILE]]

`--seed ID` writes character ID's calibration file (motion.calibrate's: paths.of(ID).motion, her lab's; `--to` another)
from the runs' stretches, each with its map, when and her stat window's SPEED then (`--speed`, else the runs' records',
else her profile's), merged with what the file has: so a calibration starts from past runs' measures (R1-R3's for the
magician at `--speed 100`, the thief's Tree runs of 10-25 to 30 at `--speed 101`). It writes nothing else.

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

from ..motion import body, calibrate, physics

# The steady walk and climb: motion.calibrate's estimator (Walks, Climbs), her live calibration's, the run's own looks
# taken where it has no perception service's scenes (2-4 a second: GAP_LOOKS_S between two of a stretch at most)
GAP_LOOKS_S = 0.6
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
        rows = _records(path, ("session", "scene", "chain", "trip", "stance", "calibration"), ("touch",))
        session = next((r for r in rows if r.get("topic") == "session"), {})
        times = [r["t"] for r in rows if "t" in r and r.get("topic") != "session"]
        self.t0, self.t1 = (min(times), max(times)) if times else (0.0, 0.0)
        if span is not None:
            self.t0, self.t1 = max(self.t0, span[0]), min(self.t1, span[1])
        rows = [r for r in rows if self.t0 <= r.get("t", 0.0) <= self.t1 or r.get("topic") == "session"]
        self.map = session.get("map")
        self.physics = session.get("physics")
        self.model = _model(self.map)
        # her planner's walk and climb changed mid-run (motion.calibrate's 'calibration' records: t, what, new, panel)
        self.calibrations = [r for r in rows if r.get("topic") == "calibration"]
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
        self.places: list[calibrate.Place] = []  # (the same, for motion.calibrate's estimator)
        touches: list[float] = []
        for p in perceives:
            for r in _records(p, ("scene",), ("touch",)):
                if not self.t0 - 5.0 <= r.get("t", 0.0) <= self.t1 + 5.0:
                    continue
                if r["topic"] == "scene":
                    if (h := _her(r)) is not None:
                        self.scenes.append(h)
                    if (q := calibrate.place_of(r["t"], r.get("her"), self.model)) is not None:
                        self.places.append(q)
                elif r.get("kind") == "touch":  # (when it came: told `late` after it)
                    touches.append(r["t"] - (r.get("late") or 0.0))
        self.by_service = bool(self.scenes)
        if not self.scenes:  # (no perception service's records: the run's own looks)
            self.scenes = [h for r in self.looks if (h := _her(r)) is not None]
            self.places = [q for r in self.looks if (q := calibrate.place_of(r["t"], r.get("her"), self.model))
                           is not None]
        self.scenes.sort()
        self.places.sort()
        self.touches = sorted(touches) if touches else own_touches
        self.st = [s[0] for s in self.scenes]

    @property
    def minutes(self) -> float:
        return (self.t1 - self.t0) / 60.0

    def touched(self, a: float, b: float) -> bool:
        i = bisect_left(self.touches, a)
        return i < len(self.touches) and self.touches[i] <= b

    def walk_at(self, t: float, planned: float) -> float:
        """Her planned walk at t: `planned` (the run's start's), or the last 'calibration' record's of her walk before t."""
        for r in self.calibrations:
            if r.get("what") == "walk" and r.get("new") is not None and r["t"] <= t:
                planned = float(r["new"])
        return planned

    def panel(self) -> float | None:
        """Her stat window's SPEED as the run's 'calibration' records last had it, None if none did."""
        speeds = [r["panel"] for r in self.calibrations if isinstance(r.get("panel"), (int, float))]
        return float(speeds[-1]) if speeds else None


_model = calibrate.map_model  # (the map's model: its slopes, and its ropes for records before her mode was written)


# --- the measures --------------------------------------------------------------------------------------------------

def stretches(run: Run, raw: bool = False) -> tuple[list[calibrate.Walk], list[calibrate.Ascent]]:
    """Her steady walks and climbs over the run (motion.calibrate.measure: her live calibration's estimator), stretches
    of the run's own looks GAP_LOOKS_S apart at most where it has no perception service's scenes; `raw`: her x speed
    with each walk's braking tail in it, no slopes taken out (walkfit's walk until 2026-11-02's calibration)."""
    places = [q for q in run.places if run.t0 <= q.t <= run.t1]
    gap = calibrate.GAP_S
    try:
        if not run.by_service:
            calibrate.GAP_S = GAP_LOOKS_S
        return calibrate.measure(places, run.model, slopes=not raw, tail=0.0 if raw else calibrate.TAIL_S)
    finally:
        calibrate.GAP_S = gap


def takeoffs(run: Run) -> list[tuple[float, float, int, float]]:
    """The chains whose first leg is a walk (TAKEOFF_LEAST or more) then a jump, touch-free to the takeoff: (the walk,
    where she took off past the plan's takeoff, the way she walked: + late, the walk's direction, the chain's time)."""
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
        rows.append((walk, ((take[0][1] + take[1][1]) / 2 - x1) * d, d, c["t"]))
    return rows


def _shoe(walk: float, jump: float = 100.0) -> body.Shoe:
    """The engine's shoe that walks `walk` units/s, whatever SPEED gives it (Physics.img's otherwise)."""
    return replace(body.Shoe.from_physics(body.PHYSICS_IMG, 100.0, jump), vmax=float(walk))


def engine_residuals(rows: list[tuple], planned: float, measured: float, jump: float = 100.0,
                     walk_at=None) -> list[float]:
    """Each takeoff's residual against the engine walking `measured` (motion.body): the plan's keys put her jump at the
    time the planned walk reaches its takeoff; where the measured walk has her then, against where she took off.
    `walk_at(t, planned)`: the planned walk as of a chain's time (Run.walk_at: her calibration's changes), else
    `planned` throughout."""
    sm = _shoe(measured, jump)
    out = []
    for walk, seen, d, *at in rows:
        sp = _shoe(planned if walk_at is None or not at else walk_at(at[0], planned), jump)
        tau = body.run_time(walk, 0.0, d, sp)
        if tau is None:
            continue
        out.append(seen - (body.run_x(tau, 0.0, d, sm) * d - walk))
    return out


def steps(run: Run) -> list[tuple[float, float]]:
    """The stance's steps away, touch-free, on one platform: (units asked, units gone to where she first stood still).
    Only the steps she walked: the hunt loop walks a step as it decides it and looks again once the hold is over
    (Hunt._shoot), so a step whose next stance record (any act) came before she stood still wasn't walked (the
    stance said it again look after look, 0.11-0.15 s apart, while a knock held the step back, fight.knocked_move,
    or her next look decided something else), and one that left her where she started or the other way was undone by
    something else; neither is counted. (Until the review of 2026-11-02 each record was a step: the hour's 'left 120'
    said three times at 00:50:13.8-14.0, held back, then 'right 67' twice, counted as three steps gone -80 and two of
    80. Of the 73 so counted over R1's grind and the hour, 15 weren't walked, their next record 0.31-1.18 s before she
    stood still, the walked ones' 0.08 s or more after; and 00:22:52.97's 'left 135' left her 9 units right.)"""
    out = []
    for k, r in enumerate(run.stances):
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
        moved, end = False, None
        for u, v in zip(seg, seg[1:]):
            if abs(v[1] - u[1]) > STILL_UNITS:
                moved = True
            elif moved:
                end = (u[0], v[1])  # (standing still from u's scene on, at v's place)
                break
        if end is None:
            continue
        nxt = run.stances[k + 1]["t"] if k + 1 < len(run.stances) else None
        gone = (end[1] - seg[0][1]) * d
        if (nxt is None or nxt >= end[0]) and gone > 0:
            out.append((n, gone))
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
    walk = physics.walk_of(stats.speed)
    return walk, stats.jump, f"assumed: {which}'s profile in the code now (SPEED {stats.speed:g}: {walk:g})"


def _quartiles(v: list[float]) -> tuple[float, float]:
    return float(np.percentile(v, 25)), float(np.percentile(v, 75))


def fit(runs: list[Run], planned: float | None = None, character: str | None = None, name: str | None = None,
        speed: float | None = None, raw: bool = False) -> dict:
    """The measures over `runs` (one run, or a map's), against the walk planned (the first run's: see planned_walk);
    her stat window's SPEED the runs' records', else `speed`, else the session's physics' (her profile's then)."""
    plans = [planned_walk(r, planned, character) for r in runs]  # (each run's own: a map's may mix them)
    one = len({round(p[0], 1) for p in plans}) == 1
    walk_p, source = (plans[0][0], plans[0][2]) if one else (None, "mixed: " + "; ".join(sorted({p[2] for p in plans})))
    got = [stretches(r, raw) for r in runs]
    walks = [w for ws, _ in got for w in ws]
    climbs = [c for _, cs in got for c in cs]
    panel, panel_from = next(((v, "the runs' records") for r in runs if (v := r.panel()) is not None), (None, ""))
    if panel is None and speed is not None:
        panel, panel_from = float(speed), "--speed"
    if panel is None and (ph := runs[0].physics or {}).get("speed") is not None:
        panel, panel_from = float(ph["speed"]), "her profile's in the session's physics (assumed her stat window's)"
    out: dict = {"run": name or runs[0].name, "map": runs[0].map, "minutes": round(sum(r.minutes for r in runs), 2),
                 "by": "perception" if all(r.by_service for r in runs) else "looks",
                 "planned": None if walk_p is None else round(walk_p, 1), "planned_from": source,
                 "slopes": "none (--raw)" if raw else "v83's" if all(r.model is not None for r in runs)
                 else "none (no map model)", "panel": panel, "panel_from": panel_from}
    e = calibrate.estimate([x.limit for x in walks])
    w: dict = {"n": len(walks)}
    if e is not None:
        w.update(median=round(e.value, 1), q1=round(e.q1, 1), q3=round(e.q3, 1), se=e.se,
                 v=round(statistics.median(x.v for x in walks), 1),
                 ratio=None if walk_p is None else round(e.value / walk_p, 3),
                 speed=round(physics.speed_of_walk(e.value), 1))
        if panel is not None:
            w["off"] = round(physics.speed_of_walk(e.value) - panel, 1)
    out["walk"] = w
    c: dict = {"n": len(climbs)}
    for way, label in ((calibrate.UP, "up"), (calibrate.DOWN, "down")):
        ec = calibrate.estimate([x.v for x in climbs if x.way == way])
        if ec is not None:
            c[label] = {"median": round(ec.value, 1), "n": ec.n, "se": ec.se}
    ec = calibrate.estimate([x.v for x in climbs])
    if ec is not None:
        c.update(median=round(ec.value, 1), se=ec.se, speed=round(physics.speed_of_climb(ec.value), 1))
        if panel is not None:
            c["off"] = round(physics.speed_of_climb(ec.value) - panel, 1)
    out["climb"] = c
    # What her calibration would plan from these stretches (motion.calibrate: MIN_N and MAX_SE, else her prior at the
    # stat window's SPEED), and whether that's within SPEED_TOL of it
    if panel is None and character is not None:
        from .. import character as profiles
        prior_speed = profiles.load(character).stats.speed
    else:
        prior_speed = panel if panel is not None else physics.CHARACTER.speed
    cal = calibrate.Calibration(None, prior_speed, panel=None if panel is None else calibrate.Panel(panel),
                                model=runs[0].model, log=lambda _: None)
    cal.rows = {"walk": [[x.t, runs[0].map, x.platform, x.way, x.span, x.v, x.ground, x.limit, panel] for x in walks],
                "climb": [[x.t, runs[0].map, x.way, x.span, x.v, panel] for x in climbs]}
    out["calibrated"] = {kind: {"value": round(v, 2), "why": why} for kind in ("walk", "climb")
                         for v, why, _ in [cal.target(kind)]}
    out["calibrated"]["panel"] = cal.correlation(force=True) if panel is not None else ""
    legs = [(takeoffs(r), p, r) for r, p in zip(runs, plans)]
    rows = [x for got_, _, _ in legs for x in got_]
    t: dict = {"n": len(rows)}
    if rows:
        late = [x[1] for x in rows]
        long_ = [x[1] for x in rows if x[0] >= LONG_WALK]
        t.update(planned=round(statistics.median(late), 1), long_n=len(long_),
                 long=round(statistics.median(long_), 1) if long_ else None)
        if len(rows) >= 3:
            k, b = np.polyfit([x[0] for x in rows], late, 1)
            t.update(slope=round(float(k), 3), intercept=round(float(b), 1))
        if walks:  # (each run's legs against the walk it planned, as of each chain: Run.walk_at)
            res = [v for got_, p, r in legs for v in engine_residuals(got_, p[0], w["median"], p[1], r.walk_at)]
            res_long = [v for got_, p, r in legs for v in engine_residuals([x for x in got_ if x[0] >= LONG_WALK], p[0],
                                                                            w["median"], p[1], r.walk_at)]
            t.update(engine=round(statistics.median(res), 1) if res else None,
                     engine_long=round(statistics.median(res_long), 1) if res_long else None)
    out["takeoff"] = t
    st = [x for r in runs for x in steps(r)]
    ratios = sorted(went / asked for asked, went in st if asked >= STEP_LEAST)
    s_: dict = {"n": len(ratios)}
    if ratios:
        q1, q3 = _quartiles(ratios)
        s_.update(ratio=round(statistics.median(ratios), 2), q1=round(q1, 2), q3=round(q3, 2))
    out["steps"] = s_
    arr = [x for r in runs for x in arrivals(r)]
    wrong = [x for x in arr if x["on"] is not None and x["on"] != x["dst"]]
    out["arrived"] = {"n": len(arr), "elsewhere": len(wrong), "untouched": sum(1 for x in wrong if not x["touched"])}
    return out


def fit_runs(args: list[Path], planned: float | None = None, character: str | None = None,
             frames: bool = False, speed: float | None = None, raw: bool = False) -> list[dict]:
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
    out = [fit([r], planned, character, speed=speed, raw=raw) for r in runs]
    by_map = defaultdict(list)
    for r in runs:
        by_map[r.map].append(r)
    for m, rs in by_map.items():
        if len(rs) > 1:
            out.append(fit(rs, planned, character, name=f"map {m}: all {len(rs)}", speed=speed, raw=raw))
    return out


def seed(args: list[Path], who: str, to: Path | None = None, speed: float | None = None, map_id: int | None = None,
         log=print) -> tuple[Path, dict]:
    """Character `who`'s calibration file (`to`, else her lab's: paths.of(who).motion) seeded from the RUN arguments'
    perception records (every scene of each perception service's file, by the map its session records say; a run's own
    looks where none; `map_id` where a file's records name none: the user's own play's clip), each stretch with its map,
    when (the record's time) and her stat window's SPEED then (`speed`,
    else her profile's, said as assumed), merged with the rows the file has (the same stretch once), the newest
    motion.calibrate.KEEP a kind kept. The file and what was added."""
    from .. import character as profiles
    from .. import paths
    c = profiles.load(who)
    target = to if to is not None else paths.of(c).motion
    at_speed = float(speed) if speed is not None else float(c.stats.speed)
    found = files_of(args)
    perceives = [p for n, p in sorted(found.items()) if "-perceive" in n or _command(p) == "perceive"
                 or n == "telemetry.jsonl"]
    added = {"walk": 0, "climb": 0}
    store = calibrate.Store(target, log)
    rows = {"walk": store.walks(), "climb": store.climbs()}
    have = {kind: {(round(r[0], 1), r[-2]) for r in rows[kind]} for kind in rows}
    for path in perceives:
        places: dict = defaultdict(list)
        on = map_id
        with open(path, encoding="utf-8") as f:
            for line in f:
                if '"topic":"session"' in line:
                    on = json.loads(line).get("map", on)
                elif '"topic":"scene"' in line and on is not None:
                    r = json.loads(line)
                    if (q := calibrate.place_of(r["t"], r.get("her"), _model(on))) is not None:
                        places[on].append(q)
        for map_id, qs in places.items():
            ws, cs = calibrate.measure(qs, _model(map_id))
            for x in ws:
                row = [round(x.t, 1), map_id, x.platform, x.way, x.span, x.v, x.ground, x.limit, at_speed]
                if (row[0], row[-2]) not in have["walk"]:
                    rows["walk"].append(row)
                    have["walk"].add((row[0], row[-2]))
                    added["walk"] += 1
            for x in cs:
                row = [round(x.t, 1), map_id, x.way, x.span, x.v, at_speed]
                if (row[0], row[-2]) not in have["climb"]:
                    rows["climb"].append(row)
                    have["climb"].add((row[0], row[-2]))
                    added["climb"] += 1
    cal = calibrate.Calibration(None, at_speed, panel=calibrate.Panel(at_speed), model=None, log=lambda _: None)
    cal.store = store
    cal.rows = {kind: sorted(v)[-calibrate.KEEP:] for kind, v in rows.items()}
    data = cal.record()
    for kind in ("walk", "climb"):  # (what her planner would take from it: not what any planned yet)
        data[kind]["planned"] = cal.target(kind)[0]
    data["seeded"] = {"from": sorted(str(p) for p in perceives), "speed": at_speed,
                      "speed_from": "--speed" if speed is not None else f"assumed: {who}'s profile's SPEED"}
    store.save(data)
    return target, {"added": added, "walk": data["walk"], "climb": data["climb"]}


def table(fits: list[dict]) -> str:
    """The fits as a table, a line each."""
    head = (f"{'run':40s} {'min':>5s} {'plan':>5s} | {'walk':>5s} {'IQR':>11s} {'se':>4s} {'n':>3s} {'x':>5s} "
            f"{'SPEED':>5s} {'off':>4s} | {'climb':>5s} {'n':>3s} | {'late':>5s} {'>=100':>5s} {'n':>4s} {'eng':>5s} "
            f"{'eng100':>6s} {'slope':>5s} | {'step':>4s} {'IQR':>9s} {'n':>3s} | {'arrived':>7s} {'off':>3s} "
            f"{'clean':>5s}")
    lines = [head]

    def f(v, w, p=1):
        return f"{'-':>{w}s}" if v is None else f"{v:{w}.{p}f}"
    for x in fits:
        w, c, t, s, a = x["walk"], x["climb"], x["takeoff"], x["steps"], x["arrived"]
        iqr = "-" if "q1" not in w else f"{w['q1']:.1f}-{w['q3']:.1f}"
        siqr = "-" if "q1" not in s else f"{s['q1']:.2f}-{s['q3']:.2f}"
        lines.append(f"{x['run'][:40]:40s} {x['minutes']:5.1f} {f(x['planned'], 5)} | {f(w.get('median'), 5)} "
                     f"{iqr:>11s} {f(w.get('se'), 4, 2)} {w['n']:3d} {f(w.get('ratio'), 5, 3)} {f(w.get('speed'), 5)} "
                     f"{f(w.get('off'), 4)} | {f(c.get('median'), 5)} {c['n']:3d} | "
                     f"{f(t.get('planned'), 5)} {f(t.get('long'), 5)} {t['n']:4d} {f(t.get('engine'), 5)} "
                     f"{f(t.get('engine_long'), 6)} {f(t.get('slope'), 5, 3)} | {f(s.get('ratio'), 4, 2)} {siqr:>9s} "
                     f"{s['n']:3d} | {a['n']:7d} {a['elsewhere']:3d} {a['untouched']:5d}")
    sources = sorted({(x["run"], x["planned_from"], x["by"], x["slopes"], x["panel"], x["panel_from"]) for x in fits},
                     key=str)
    lines.append("")
    lines += [f"  {r}: planned walk from {src}; positions by the {by}'s scenes; slopes taken out: {sl}; her stat window's "
              f"SPEED {'-' if pn is None else f'{pn:g}'}{f' ({pf})' if pf else ''}"
              for r, src, by, sl, pn, pf in sources]
    for x in fits:
        cal = x["calibrated"]
        lines.append(f"  {x['run']}: her calibration would plan walk {cal['walk']['value']:g} ({cal['walk']['why']}), "
                     f"climb {cal['climb']['value']:g} ({cal['climb']['why']})" + (f"; {cal['panel']}" if cal['panel']
                                                                                    else ""))
    lines.append("  walk: units/s on flat ground (x: against the planned; SPEED: the % it implies by the client's formula; "
                 "off: that less her stat window's); climb: units/s up and down; late: units past the "
                 "planned takeoff (>=100: after walks of 100+; eng: against the engine at her measured walk); step: "
                 "gone / asked; arrived: 'arrived' trips, off: on another platform, clean: of those with no touch "
                 f"{TOUCH_BEFORE_S:g} s before")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gamelab.analysis.walkfit", description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path, help="telemetry folders or files")
    ap.add_argument("--planned", type=float, help="the walk (units/s) a run without a physics record planned at")
    ap.add_argument("--character", help="... else this character's profile as the code has it now")
    ap.add_argument("--speed", type=float, help="her stat window's SPEED over the runs, where their records don't say")
    ap.add_argument("--frames", action="store_true", help="each recording's folder cut to its frames' span")
    ap.add_argument("--raw", action="store_true", help="her x speed with each walk's braking tail in it, no slopes out")
    ap.add_argument("--json", action="store_true", help="the fits as JSON")
    ap.add_argument("--seed", metavar="ID", help="write character ID's calibration file from the runs (see the top)")
    ap.add_argument("--to", type=Path, help="... to this file instead of her lab's")
    ap.add_argument("--map", type=int, help="... the map where a file's records name none")
    a = ap.parse_args(argv)
    if a.seed:
        path, got = seed(a.runs, a.seed, a.to, a.speed, a.map)
        print(f"{path}: {got['added']['walk']} walks and {got['added']['climb']} climbs added; walk "
              f"{got['walk'].get('value', '-')} over {got['walk'].get('n', 0)} ({got['walk']['why']}), climb "
              f"{got['climb'].get('value', '-')} over {got['climb'].get('n', 0)} ({got['climb']['why']})")
        return 0
    fits = fit_runs(a.runs, a.planned, a.character, a.frames, a.speed, a.raw)
    if not fits:
        print("no hunt's telemetry found (a tour or hunt command's *.jsonl)")
        return 1
    print(json.dumps(fits, indent=1) if a.json else table(fits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
