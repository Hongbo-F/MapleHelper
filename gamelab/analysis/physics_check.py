"""Her movement in the recordings against the physics models, offline: her track (her name tag to a
fraction of a pixel, the camera taken out) against what each model makes of the same keys, as RMS error
in map units, x and y.

The models, each with fixed lags:
- before: what the planner and the simulator modelled before motion.body (physics.py's constants then):
  standing, she runs at full speed RUN_LAG after the arrow; let go, she runs on, then brakes evenly;
  in the air her sideways speed never changes;
- steps: that model with physics.py's constants now (the simulator's, until it runs on the engine);
- engine: motion.body.replay, with the median lags (body.Lags). `--fit` adds the engine with its lags
  fitted to each recording, which shows what is left once the lags' spread is taken out.

  python -m gamelab.analysis.physics_check                 every recording in CASES
  python -m gamelab.analysis.physics_check jb_k165 --fit   one, with the fitted lags too
  python -m gamelab.analysis.physics_check --fixture motion walk
                                                           save their tracks under tests/fixtures/tracks

A recording is tracked from lab/recordings (git-ignored) when it's there, else read from its fixture.
Its keys are read with her key bindings (config.toml [keys]).
"""

import argparse
import json
import sys
from dataclasses import dataclass, replace

import numpy as np

from ..motion import body, physics
from ..paths import RECORDINGS, REPO, TAG_TEMPLATE
from ..screen import UNITS_PER_HALF_PX
from . import recordings, scroll, track

FIXTURES = REPO / "tests" / "fixtures" / "tracks"
MIN_SCORE = 0.45   # frames where her name tag matched worse are left out (another name tag over hers, a blur)

# The tracked recordings: how long into each one she is compared (s after the first key), and the floor
# she lands on when it isn't the one she started on ((from s, y): y down, so -60 is 60 units up).
# Static camera at 60 fps: motion, held_jump, the jump-backs (jb_*, jump_back_land). Moving camera:
# walkjump, jumpA, jumpup, chain, do_ledge2. The jump-backs end before they land: an attack is playing
# then, which stops her dead (slides of 1.4-2.9 units where braking makes 7.6-11.7).
CASES = {
    "motion": (1.9, ()), "walk": (0.95, ()), "walk300": (0.95, ()), "walk600": (1.25, ()),
    "walk1000": (1.6, ()), "do_walk": (1.2, ()), "do_walk2": (1.1, ()), "walkjump": (1.6, ()),
    "chain": (1.2, ((0.6, -60.4),)), "jumpA": (1.5, ((0.4, -60.8),)), "jb_k165": (0.69, ()),
    "jb_k190": (0.66, ()), "jb_k290": (0.69, ()), "jb_k450": (0.69, ()), "jump_back_land": (0.69, ()),
    "held_jump": (1.2, ()), "jumpup": (1.05, ()), "do_ledge2": (1.35, ()),
}
NOTES = {"chain": "another player's name tag covers hers at the end", "do_ledge2": "moving camera, 30 fps"}


@dataclass
class Track:
    """Her path through a recording, in map units from where she stood at its first frame (x right, y
    down), and the keys that went out (t s, action, down)."""
    name: str
    events: list[tuple[float, str, bool]]
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    score: np.ndarray

    def used(self, until: float) -> np.ndarray:
        return (self.score > MIN_SCORE) & (self.t <= until)


def _actions(sent, key_names: dict[str, str]) -> list[tuple[float, str, bool]]:
    """The keys that went out ([ms, key, 'down'|'up']) as (s, action, down)."""
    return [(t / 1000, key_names.get(k, k), how == "down") for t, k, how in sent]


def her_key_names() -> dict[str, str]:
    """Her key bindings as key name -> action (config.toml [keys])."""
    from ..control.timeline import her_setup
    _, keys, _ = her_setup()
    return {getattr(keys, a).name: a for a in ("left", "right", "jump", "attack") if getattr(keys, a)}


def tracked(name: str, key_names: dict[str, str] | None = None) -> Track:
    """Her track through a saved recording: her name tag in every frame (track.fine_match, with the
    lab's name tag template), plus the camera's movement (scroll.camera), in map units."""
    rec = recordings.load(name)
    template = np.load(TAG_TEMPLATE)
    rows = np.array([track.fine_match(f, template) for f in rec.frames])
    cam = scroll.camera(rec.frames, [(r[0] + template.shape[1] / 2, r[1]) for r in rows])
    x = (rows[:, 0] + cam[:, 0] - rows[0, 0]) * UNITS_PER_HALF_PX
    y = (rows[:, 1] + cam[:, 1] - rows[0, 1]) * UNITS_PER_HALF_PX
    events = _actions(rec.sent, key_names if key_names is not None else her_key_names())
    return Track(name, events, np.asarray(rec.times, float) / 1000, x, y, rows[:, 2])


def save_fixture(tr: Track) -> None:
    """Write a track as tests/fixtures/tracks/<name>.json: our own measurement of the recording (not game
    data), small enough to keep in git."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    until, floors = CASES[tr.name]
    data = {"name": tr.name, "made_by": f"python -m gamelab.analysis.physics_check --fixture {tr.name}",
            "until": until, "floors": [list(f) for f in floors],
            "events": [[round(t * 1000, 1), a, "down" if down else "up"] for t, a, down in tr.events],
            "t": [round(float(v), 4) for v in tr.t], "x": [round(float(v), 2) for v in tr.x],
            "y": [round(float(v), 2) for v in tr.y], "score": [round(float(v), 3) for v in tr.score]}
    (FIXTURES / f"{tr.name}.json").write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")


def fixture(name: str) -> Track:
    """A track saved by save_fixture."""
    d = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return Track(d["name"], _actions(d["events"], {}), np.array(d["t"]), np.array(d["x"]), np.array(d["y"]),
                 np.array(d["score"]))


def load(name: str, key_names: dict[str, str] | None = None) -> Track:
    """The recording's track if it's in lab/recordings, else its fixture."""
    if (RECORDINGS / f"{name}.npz").exists():
        return tracked(name, key_names)
    return fixture(name)


# --- the models ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Steps:
    """The constants of the step model (the simulator's walk before the engine)."""
    walk: float        # units/s
    run_lag: float     # s from the arrow to running at full speed, standing
    jump_lag: float    # s from the jump key to lift-off
    stop_run: float    # units she goes on after letting go at full speed ...
    land_slide: float  # ... of which this much braking (the rest is the release's lag)


BEFORE = Steps(walk=149.0, run_lag=0.12, jump_lag=0.055, stop_run=27.0, land_slide=13.0)
NOW = Steps(walk=physics.WALK_SPEED, run_lag=physics.RUN_LAG, jump_lag=physics.JUMP_LAG,
            stop_run=physics.STOP_RUN, land_slide=physics.LAND_SLIDE)


def steps(events, until: float, m: Steps, *, start: float = -0.1, dt: float = 0.001, floors=()) -> np.ndarray:
    """The step model's (t, x, y) under a key timeline on flat ground, as the simulator moved her (sim/world:
    _ground, _land, the arc with no air control)."""
    decel = m.walk ** 2 / (2 * m.land_slide)
    release_lag = (m.stop_run - m.land_slide) / m.walk
    jumps = sorted(t + m.jump_lag for t, a, down in events if a == "jump" and down)
    arrows = sorted((t, a, down) for t, a, down in events if a in ("left", "right"))
    x = y = vx = vy = 0.0
    ground, floor = True, 0.0
    held: dict[str, float] = {}
    go, go_t, release = 0, -1e9, -1e9
    out, t, i, j = [], start, 0, 0
    while t <= until:
        while i < len(arrows) and arrows[i][0] <= t:
            ta, a, down = arrows[i]
            i += 1
            if down:
                held[a], go, go_t = ta, (1 if a == "right" else -1), ta
            else:
                held.pop(a, None)
                if not held:
                    release = ta
        d = 0 if not held else 1 if max(held, key=held.get) == "right" else -1
        for since, level in floors:
            if t >= since:
                floor = level
        if j < len(jumps) and t >= jumps[j]:
            if ground:
                ground, vy = False, -physics.JUMP_SPEED
            j += 1
        if ground:
            if d:
                if vx * d > 0:
                    vx = d * m.walk
                elif vx == 0:
                    if t >= go_t + m.run_lag:
                        vx = d * m.walk
                else:
                    vx = _brake(vx, decel * dt)
            elif vx == 0 and go and go_t + m.run_lag <= t < release + release_lag:
                vx = go * m.walk  # a short tap: let go before she started, she still moves
            elif vx and t >= release + release_lag:
                vx = _brake(vx, decel * dt)
            x += vx * dt
        else:
            vy = min(vy + physics.GRAVITY * dt, physics.FALL_SPEED)
            x, y = x + vx * dt, y + vy * dt
            if vy > 0 and y >= floor:
                y, vy, ground = floor, 0.0, True
                if d and vx * d > 0:
                    vx = d * m.walk
                else:
                    go, release = d, -1e9  # no arrow: she brakes at once
        t += dt
        out.append((t, x, y))
    return np.array(out)


def _brake(v: float, by: float) -> float:
    return 0.0 if abs(v) <= by else v - by if v > 0 else v + by


def engine(events, until: float, lags: body.Lags, floors=(), shoe: body.Shoe | None = None) -> np.ndarray:
    """The engine's (t, x, y) under a key timeline on flat ground."""
    r = body.replay(events, until + 0.02, shoe or physics.SHOE, lags, floors=floors)
    return np.array(r.points)


def errors(tr: Track, path: np.ndarray, until: float) -> tuple[float, float, float]:
    """RMS error in x and y of a model's path against the track, and the model's x at the last frame used."""
    ok = tr.used(until)
    t = tr.t[ok]
    xm, ym = np.interp(t, path[:, 0], path[:, 1]), np.interp(t, path[:, 0], path[:, 2])
    rms = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    return rms(tr.x[ok], xm), rms(tr.y[ok], ym), float(xm[-1])


def fit_lags(tr: Track, until: float, floors=(), rounds: int = 3) -> body.Lags:
    """The engine's lags fitted to one recording: each of on, off and jump in turn over its grid (5 ms,
    10 for off), the others held, `rounds` times, from the medians, for the least x² + y² error."""
    grids = {"on": np.arange(0.02, 0.1001, 0.005), "jump": np.arange(0.02, 0.1051, 0.005),
             "off": np.arange(0.02, 0.1201, 0.01)}
    lags = body.Lags()

    def cost(lg):
        ex, ey, _ = errors(tr, engine(tr.events, until, lg, floors), until)
        return ex * ex + ey * ey

    best = cost(lags)
    for _ in range(rounds):
        for name, grid in grids.items():
            for v in grid:
                trial = replace(lags, **{name: float(v)})
                c = cost(trial)
                if c < best - 1e-12:
                    best, lags = c, trial
    return lags


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m gamelab.analysis.physics_check", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help=f"recordings (default: all of {len(CASES)})")
    p.add_argument("--fit", action="store_true", help="also the engine with its lags fitted per recording")
    p.add_argument("--fixture", action="store_true", help="save the recordings' tracks as test fixtures")
    args = p.parse_args(argv)
    names = args.names or list(CASES)
    if unknown := [n for n in names if n not in CASES]:
        print(f"not in CASES: {unknown}")
        return 2
    keys = her_key_names()
    lags = body.Lags()
    head = "| Recording | before x / y | steps x / y | engine x / y |" + (" fitted x / y (on off jump ms) |" if args.fit else "")
    print(head + " end x seen | before | steps | engine |")
    print("|---" * (head.count("|") - 1 + 4) + "|")
    rows = []
    for name in names:
        until, floors = CASES[name]
        if args.fixture:
            tr = tracked(name, keys)
            save_fixture(tr)
        else:
            tr = load(name, keys)
        ok = tr.used(until)
        if ok.sum() < 5:
            print(f"| {name} | too few frames tracked |")
            continue
        b = errors(tr, steps(tr.events, until, BEFORE, floors=floors), until)
        s = errors(tr, steps(tr.events, until, NOW, floors=floors), until)
        e = errors(tr, engine(tr.events, until, lags, floors), until)
        line = f"| {name} | {b[0]:.2f} / {b[1]:.2f} | {s[0]:.2f} / {s[1]:.2f} | {e[0]:.2f} / {e[1]:.2f} |"
        row = [b[0], b[1], s[0], s[1], e[0], e[1]]
        if args.fit:
            fl = fit_lags(tr, until, floors)
            f = errors(tr, engine(tr.events, until, fl, floors), until)
            line += f" {f[0]:.2f} / {f[1]:.2f} ({fl.on * 1000:.0f} {fl.off * 1000:.0f} {fl.jump * 1000:.0f}) |"
            row += [f[0], f[1]]
        line += f" {tr.x[ok][-1]:.1f} | {b[2]:.1f} | {s[2]:.1f} | {e[2]:.1f} |"
        if name in NOTES:
            line += f" ({NOTES[name]})"
        print(line)
        rows.append(row)
    if rows:
        a = np.array(rows)
        mean, median = a.mean(axis=0), np.median(a, axis=0)
        cells = " | ".join(f"{mean[i]:.2f} ({median[i]:.2f}) / {mean[i + 1]:.2f} ({median[i + 1]:.2f})"
                           for i in range(0, a.shape[1], 2))
        print(f"| **mean (median)** | {cells} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
