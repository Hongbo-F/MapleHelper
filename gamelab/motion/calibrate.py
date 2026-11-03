"""Her walk and climb as perception's place track measures them: the one estimator her planner is calibrated by as she
hunts (`Calibration`, the hunter's: behavior.hunter) and analysis.walkfit measures a run by, kept in her lab between
runs. The user, 2026-11-02, asked whether the thief's planned SPEED should be his stat window's 101 or the 99.2 that kept
the simulator's results comparable: "I feel this should just be measured and updated dynamically and persisted,
constantly calibrated. Also correlate that with the stats panel value."; asked the same of the climb (the formula's
SPEED + 20 or the placeholder 117): "Same comment as above".

The estimator (Walks, Climbs: fed her places one at a time, each frame's state as the hunt sees it, or each scene
record of a run's telemetry):
- her walk: stretches of her places on one platform, on the ground, placed by her name tag or the camera (PLACED_BY),
  GAP_S apart at most, moving one way at WALK_BAND units/s from place to place, STEADY_S and MIN_PLACES or more; a
  line fitted from SETTLE_S after the stretch's start (her run-up: full speed 0.106 s after the game sees the arrow) to
  TAIL_S before its end (her braking: the release's lag and her slide are still faster than WALK_BAND's floor), no
  place FIT_UNITS off it; its x speed, with Tree Dungeon's slopes taken out by the v83 rule (`ground_factor`: RustMS
  Physics.cpp move_shoe, the rule of three that fits both characters' walks, D:/MapleHelper-scratch/recfix/
  walk-research/slope_models*.txt): her walk on flat ground, the shoe's vmax. Her hour of 2026-11-02 by it: 66 walks,
  the median 149.44 (bootstrap 90% 149.1-150.0); the thief's Tree runs of 10-25 to 30: 243 walks, 151.19 (150.7-
  151.75), against the client's 150 and 151.25 by their stat windows' SPEED 100 and 101 (README "Her walk and climb,
  calibrated"); raw, with each walk's braking tail in it and her x speed as it is, 148.2 (81 walks: walkfit's
  `--tail 0 --flat`).
- her climb: stretches of her places on one rope or ladder (mode 'rope', her x within ROPE_DX of the stretch's first),
  placed by PLACED_BY, GAP_S apart at most, moving one way at CLIMB_BAND, CLIMB_STEADY_S and CLIMB_PLACES or more; a
  line fitted from CLIMB_SETTLE_S after the start to CLIMB_TAIL_S before the end, no place CLIMB_FIT off it: up and
  down apart. Her hour of 2026-11-02 by it: 42 climbs up, the median 120.1, 5 down, 119.6 (SPEED + 20: 120). The
  thief's of 10-25 to 27 were placed by the minimap's dot alone (27 climbs up, 106-146 units/s, its 4-unit steps):
  none is taken, his climb his prior's (121) until he's measured.
Robust: a knock or a fall puts her in the air (her stretch ends there; a knock's slide on the ground is faster than
she walks), a misplaced look is off the line (the stretch isn't taken) or off her platform (it ends there), a stretch
is one value however long, and the planner goes by the median of the newest KEEP (`estimate`: its spread, and the
median's standard error from the median absolute deviation).

The calibration (Calibration), each run: her walk and climb from her prior (the client's formula, physics.walk_of and
climb_of, at her stat window's SPEED as last read, else her profile's) until the evidence is enough (MIN_N stretches,
the median's standard error MAX_SE at most; or GROSS_N with GROSS_SE where the prior is GROSS off: a profile far from
her), then the median, changed whenever it moves DEADBAND from what's planned; each change proposed to the planner
(physics.propose), made where no chain of hers runs (physics.settle), a 'calibration' record and a line in her log
when it is. Kept in her lab (paths.of().motion: lab/her_motion.json the thief's, lab/characters/<hers>/her_motion.json
another's; Store: written whole each time, a new file put in place, by this one writer), each stretch with the map,
when, and her stat window's SPEED then; loaded at the start, so each run starts calibrated. Her stat window's SPEED
(behavior.statcheck, the hunter's read) is correlated with it: the SPEED her walk implies by the formula against the
window's, SPEED_TOL apart noted, more warned of in her log and the 'calibration' record (a buff, gear, a map's
physics, the formula wrong), never stopping her; a stretch measured at another SPEED counts at hers now by the formula
(scaled by (SPEED + 20)/(its SPEED + 20)) until those at hers are enough alone. The simulator's world walks her at her
prior (sim.world.her_shoe), her planner calibrating as live: the run's own lab, never the real one (sim.fakes.patched).
"""

import json
import math
import os
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from .. import telemetry
from ..localization.mapdata import MapModel, Platform
from . import body, physics

# --- the estimator: her walk -----------------------------------------------------------------------------------------
PLACED_BY = ("tag", "camera")  # her places a stretch is made of: by her name tag or the camera (not the minimap's dot)
GAP_S = 0.2                    # s between two places of a stretch at most (the perception service's scene records come
                               # every other frame, about 7.5 a second; the hunt's states 15)
WALK_BAND = (90.0, 220.0)      # units/s from place to place of a walk (any SPEED's: 125-200)
STEADY_S = 0.8                 # a walk this long at least ...
MIN_PLACES = 6                 # ... over this many places
SETTLE_S = 0.25                # s of its start left out: her run-up
TAIL_S = 0.15                  # s of its end left out: her braking, faster than WALK_BAND's floor (stopping from full
                               # speed: on 0.068 s, then braking at 800 units/s² for 0.19 s, body.stop_time)
FIT_UNITS = 6.0                # no place further than this off the line fitted (a step, a knock, a misplaced look)
FIT_LEAST = 4                  # places the line is fitted to at least
# ... her climb
ROPE_DX = 4.0                  # units: her x on one rope (the stretch's first place's) give or take
CLIMB_BAND = (60.0, 200.0)     # units/s from place to place of a climb (any SPEED's: 100-160)
CLIMB_STEADY_S = 0.4
CLIMB_PLACES = 5
CLIMB_SETTLE_S = 0.1
CLIMB_TAIL_S = 0.1
CLIMB_FIT = 4.0
UP, DOWN = -1, 1               # a climb's way (y down)


class Place(NamedTuple):
    """Her place at a moment: t (s), map x and y, her platform's id (None off one), her mode ('ground', 'air', 'rope'),
    how she was placed ('tag', 'camera', 'dot', None)."""
    t: float
    x: float
    y: float
    platform: int | None
    mode: str | None
    how: str | None


def place_of(t: float, her: dict | None, model: MapModel | None = None) -> Place | None:
    """A scene record's her (telemetry: the perception service's or the hunt's) at `t`, None if it didn't place her.
    Records before the mode was written (the thief's of 10-25 to 30) had `grounded`: on the ground, or off it, where with
    `model` she's on a rope if one of its ropes is within ROPE_DX of her x and covers her y (a climb's stretch is its line
    all the same: Climbs; a jump past a rope isn't one, nor a fall at 670)."""
    if not her or her.get("x") is None or her.get("y") is None:
        return None
    x, y = float(her["x"]), float(her["y"])
    mode = her.get("mode")
    if mode is None:
        mode = "ground" if her.get("grounded") else "air"
        if mode == "air" and model is not None and her.get("platform") is None and any(
                abs(r.x - x) <= ROPE_DX and r.top - ROPE_DX <= y <= r.bottom + ROPE_DX for r in model.ropes):
            mode = "rope"
    return Place(float(t), x, y, her.get("platform"), mode, her.get("how"))


def place_of_state(st) -> Place | None:
    """A frame's state's her (localization.worldstate.WorldState: `where`, `mode`, `how`), None if it didn't place her."""
    w = getattr(st, "where", None)
    if w is None or w[0] is None or w[1] is None:
        return None
    return Place(float(st.t), float(w[0]), float(w[1]), w[2], getattr(st, "mode", None), getattr(st, "how", None))


def _fit(ts: list[float], vs: list[float], start: float, end: float, most: float) -> tuple[float, int] | None:
    """The slope of the line through (ts, vs) from `start` to `end` (s), and how many places, if FIT_LEAST or more and
    none `most` off it; else None."""
    pts = [(t, v) for t, v in zip(ts, vs) if start <= t <= end]
    if len(pts) < FIT_LEAST:
        return None
    n = len(pts)
    mt = sum(p[0] for p in pts) / n
    mv = sum(p[1] for p in pts) / n
    stt = sum((p[0] - mt) ** 2 for p in pts)
    if stt <= 0:
        return None
    k = sum((p[0] - mt) * (p[1] - mv) for p in pts) / stt
    b = mv - k * mt
    if max(abs(v - (k * t + b)) for t, v in pts) >= most:
        return None
    return k, n


def ground_factor(platform: Platform | None, x0: float, x1: float, way: int) -> float:
    """Her walk on flat ground over her x speed on `platform` from x0 to x1 going `way` (+1 right): the mean over her
    path (every unit of x) of 1/g, g her x speed's share of her walk on the foothold there by the v83 rule (RustMS
    Physics.cpp move_shoe): uphill she walks her walk along it, downhill (1 + s²) of it, s the slope's sine, her x speed
    that times the cosine. 1 on flat ground, or with no platform."""
    if platform is None or x1 == x0:
        return 1.0
    lo, hi = sorted((x0, x1))
    out, n = 0.0, 0
    x = lo
    while x < hi:
        m = 0.0
        for s in platform.segments:
            if s.x1 <= x <= s.x2 and s.x2 > s.x1:
                m = (s.y2 - s.y1) / (s.x2 - s.x1)
                break
        cos = 1.0 / math.sqrt(1.0 + m * m)
        s2 = m * m / (1.0 + m * m)
        out += 1.0 / ((1.0 + s2) * cos if way * m > 0 else cos)  # (y down: going `way`, m > 0 takes her down)
        n += 1
        x += 1.0
    return out / n if n else 1.0


@dataclass(frozen=True)
class Walk:
    """One steady walk: when it started (s), its platform, its way (+1 right), how far it went (units, the line's
    fit), her x speed over it (`v`, units/s), its ground's factor (ground_factor) and her walk on flat ground (`limit`
    = v x ground), from how many places."""
    t: float
    platform: int | None
    way: int
    span: float
    v: float
    ground: float
    limit: float
    n: int


@dataclass(frozen=True)
class Ascent:
    """One steady climb: when it started (s), its way (UP or DOWN), how far it went (units), her speed (units/s), from
    how many places."""
    t: float
    way: int
    span: float
    v: float
    n: int


class Walks:
    """Her steady walks as her places come (`add`, one at a time, in time order; `end` when they stop coming). `model`:
    the map, whose platforms' slopes are taken out (`slopes`; else her x speed is her walk, as the simulator's world walks
    her: sim.world). `tail`: s of each walk's end left out (TAIL_S)."""

    def __init__(self, model: MapModel | None = None, slopes: bool = True, tail: float = TAIL_S):
        self.model, self.slopes, self.tail = model, slopes, tail
        self._seg: list[Place] = []

    @staticmethod
    def ok(q: Place) -> bool:
        return q.mode == "ground" and q.how in PLACED_BY and q.platform is not None

    def add(self, q: Place) -> Walk | None:
        """Place `q` taken in; the walk it ended, if one was taken."""
        seg = self._seg
        if not self.ok(q):
            out = self._flush()
            self._seg = []
            return out
        if seg and q.t - seg[-1].t < GAP_S and q.platform == seg[-1].platform:
            a = seg[-1]
            dt = q.t - a.t
            v = (q.x - a.x) / dt if dt > 0 else 0.0
            if WALK_BAND[0] < abs(v) < WALK_BAND[1] and (len(seg) < 2 or (a.x - seg[-2].x) * (q.x - a.x) > 0):
                seg.append(q)
                return None
        out = self._flush()
        self._seg = [q]
        return out

    def end(self) -> Walk | None:
        """The places stopped coming: the walk they end, if one is taken."""
        out = self._flush()
        self._seg = []
        return out

    def _flush(self) -> Walk | None:
        seg = self._seg
        if len(seg) < MIN_PLACES or seg[-1].t - seg[0].t < STEADY_S:
            return None
        ts, xs = [q.t for q in seg], [q.x for q in seg]
        got = _fit(ts, xs, ts[0] + SETTLE_S, ts[-1] - self.tail, FIT_UNITS)
        if got is None:
            return None
        k, n = got
        used = [q.x for q in seg if ts[0] + SETTLE_S <= q.t <= ts[-1] - self.tail]
        way = 1 if k > 0 else -1
        platform = None
        if self.slopes and self.model is not None and seg[0].platform is not None:
            try:
                platform = self.model.platform(seg[0].platform)
            except (IndexError, KeyError, TypeError):
                platform = None
        g = ground_factor(platform, used[0], used[-1], way)
        v = abs(k)
        return Walk(round(seg[0].t, 3), seg[0].platform, way, round(abs(used[-1] - used[0]), 1), round(v, 2),
                    round(g, 4), round(v * g, 2), n)


class Climbs:
    """Her steady climbs as her places come (`add`, `end`: Walks')."""

    def __init__(self):
        self._seg: list[Place] = []

    def add(self, q: Place) -> Ascent | None:
        seg = self._seg
        if q.mode != "rope" or q.how not in PLACED_BY:
            out = self._flush()
            self._seg = []
            return out
        if seg and q.t - seg[-1].t < GAP_S and abs(q.x - seg[0].x) <= ROPE_DX:
            a = seg[-1]
            dt = q.t - a.t
            v = (q.y - a.y) / dt if dt > 0 else 0.0
            if CLIMB_BAND[0] < abs(v) < CLIMB_BAND[1] and (len(seg) < 2 or (a.y - seg[-2].y) * (q.y - a.y) > 0):
                seg.append(q)
                return None
        out = self._flush()
        self._seg = [q]
        return out

    def end(self) -> Ascent | None:
        out = self._flush()
        self._seg = []
        return out

    def _flush(self) -> Ascent | None:
        seg = self._seg
        if len(seg) < CLIMB_PLACES or seg[-1].t - seg[0].t < CLIMB_STEADY_S:
            return None
        ts, ys = [q.t for q in seg], [q.y for q in seg]
        got = _fit(ts, ys, ts[0] + CLIMB_SETTLE_S, ts[-1] - CLIMB_TAIL_S, CLIMB_FIT)
        if got is None:
            return None
        k, n = got
        return Ascent(round(seg[0].t, 3), UP if k < 0 else DOWN, round(abs(seg[-1].y - seg[0].y), 1),
                      round(abs(k), 2), n)


def map_model(map_id: int | None) -> MapModel | None:
    """Map `map_id`'s model (its platforms' slopes and its ropes: localization.mapdata.load, cached), None if it won't
    load (no game data here, or not that map's): analysis.walkfit's runs."""
    if map_id not in _MODELS:
        try:
            from ..localization import mapdata
            _MODELS[map_id] = None if map_id is None else mapdata.load(int(map_id))
        except Exception:  # noqa: BLE001 (no game data, a map not cached: its slopes not taken out, said)
            _MODELS[map_id] = None
    return _MODELS[map_id]


_MODELS: dict = {}


def measure(places: Iterable[Place], model: MapModel | None = None, slopes: bool = True,
            tail: float = TAIL_S) -> tuple[list[Walk], list[Ascent]]:
    """Every steady walk and climb in `places` (in time order): Walks' and Climbs', as a run's records give them
    (analysis.walkfit; the seed of a calibration from past runs: seed)."""
    walks, climbs = Walks(model, slopes, tail), Climbs()
    ws, cs = [], []
    for q in places:
        if (w := walks.add(q)) is not None:
            ws.append(w)
        if (c := climbs.add(q)) is not None:
            cs.append(c)
    if (w := walks.end()) is not None:
        ws.append(w)
    if (c := climbs.end()) is not None:
        cs.append(c)
    return ws, cs


@dataclass(frozen=True)
class Estimate:
    """A measure over stretches: the median, its quartiles, the median's standard error (1.2533 x the spread, 1.4826 x
    the median absolute deviation, over √n; None with one), how many."""
    value: float
    q1: float
    q3: float
    se: float | None
    n: int

    def words(self, unit: str = "units/s") -> str:
        se = "" if self.se is None else f" ± {self.se:.2f}"
        return f"{self.value:.1f}{se} {unit} ({self.n}, quartiles {self.q1:.1f}-{self.q3:.1f})"


def estimate(values: list[float]) -> Estimate | None:
    """The median of `values` with its spread (Estimate), None with none."""
    if not values:
        return None
    v = sorted(values)
    med = statistics.median(v)
    n = len(v)
    q1, q3 = (v[0], v[0]) if n == 1 else (statistics.quantiles(v, n=4, method="inclusive")[0],
                                         statistics.quantiles(v, n=4, method="inclusive")[2])
    mad = statistics.median(abs(x - med) for x in v)
    se = None if n < 2 else 1.2533 * 1.4826 * mad / math.sqrt(n)
    return Estimate(round(med, 2), round(q1, 2), round(q3, 2), None if se is None else round(se, 3), n)


# --- the calibration ------------------------------------------------------------------------------------------------
MIN_N = {"walk": 10, "climb": 6}        # stretches before her measure is planned with ...
MAX_SE = {"walk": 0.75, "climb": 0.75}  # ... and its median's standard error at most (units/s): her hour's 66 walks have
                                        # 0.45 by it, spread 2.0 (the median absolute deviation), so 10 of them about 1.2,
                                        # 15 about 0.95, 25 0.75; her climbs spread 0.7, 6 of them 0.5
GROSS = 0.05                            # ... or, where her prior is this share or more off her measure (a profile far
GROSS_N = {"walk": 4, "climb": 3}       # from her: the simulator's v83 prior of 125 for her 150), from this many with
GROSS_SE = {"walk": 2.0, "climb": 2.0}  # the median's error this at most
DEADBAND = 0.4                          # units/s her measure moves from what's planned before the planner follows it
ROUND = 0.1                             # units/s the planner takes her walk and climb to
KEEP = 200                              # stretches kept a kind, the newest (an hour's walks of hers is 66)
SPEED_TOL = 2.0                         # SPEED points between her stat window's and what her measure implies by the
                                        # client's formula: within, noted; beyond, warned (her 149.44 is 99.55, the
                                        # thief's 151.19 100.95)
SAME_SPEED = 0.5                        # a stretch at her stat window's SPEED within this counts as measured at hers
VERSION = 1                             # her motion file's (Store): another's is started anew


def _rows_ok(rows, width: int) -> list[list]:
    out = []
    for r in rows if isinstance(rows, list) else ():
        if isinstance(r, list) and len(r) == width and all(isinstance(v, (int, float)) or v is None for v in r) \
                and isinstance(r[-2], (int, float)):
            out.append(r)
    return out


# Her motion file's rows: a walk's [at (epoch s), map, platform, way, span, v, ground, limit, SPEED], a climb's [at, map,
# way, span, v, SPEED]: SPEED her stat window's when it was measured (None: not known, counted at hers now).
WALK_ROW = ("at", "map", "platform", "way", "span", "v", "ground", "limit", "speed")
CLIMB_ROW = ("at", "map", "way", "span", "v", "speed")


def _text(data: dict) -> str:
    """Her motion file's text: its summary indented, its rows one a line."""
    head = {k: v for k, v in data.items() if k not in ("walks", "climbs")}
    text = json.dumps(head, indent=1)[:-2]
    for key in ("walks", "climbs"):
        rows = data.get(key) or []
        text += f',\n "{key}": [' + ",".join("\n  " + json.dumps(r, separators=(",", ":")) for r in rows) + "]"
    return text + "\n}\n"


class Store:
    """Her motion file (paths.of().motion, or a seed's): its rows and what was planned, read once (a missing file, one
    that doesn't parse, or of another VERSION: none, said), written whole (`save`: a new file put in its place, never
    half written). `path` None: nothing read or written."""

    NONE = "none kept yet"

    def __init__(self, path: Path | None, log: Callable[[str], None] = print):
        self.path, self.log = path, log
        self.data: dict = {}
        self.why = ""  # why nothing was read, if it wasn't
        if path is None:
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.why = self.NONE
            return
        except (OSError, ValueError) as e:
            self.why = f"her motion file doesn't read ({type(e).__name__}): started anew"
            return
        if not isinstance(data, dict) or data.get("version") != VERSION:
            self.why = (f"her motion file is of another version ({data.get('version') if isinstance(data, dict) else '?'}"
                        f", not {VERSION}): started anew")
            return
        self.data = data

    def walks(self) -> list[list]:
        return _rows_ok(self.data.get("walks"), len(WALK_ROW))

    def climbs(self) -> list[list]:
        return _rows_ok(self.data.get("climbs"), len(CLIMB_ROW))

    def save(self, data: dict) -> bool:
        """`data` written to `path`; whether it was (a failure said, never raised: the hunt goes on)."""
        if self.path is None:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(_text(data), encoding="utf-8")
            os.replace(tmp, self.path)
            self.data = data
            return True
        except OSError as e:
            self.log(f"    her motion file not saved: {e}")
            return False


@dataclass
class Panel:
    """Her stat window's SPEED and JUMP (%) as last read, and when (epoch s; None: not known)."""
    speed: float
    jump: float | None = None
    at: float | None = None


@dataclass
class Calibration:
    """Her walk and climb calibrated as she hunts (see the top): her rows (Store's, and those measured since), her
    estimators, her stat window's read (`panel`), her profile's SPEED (`speed`: the prior where the window hasn't been
    read), the map (`model`: its slopes, and each row's map), `prior` the walk and climb to start from instead (the
    simulator's wrong priors: sim.harness `--world prior-v83`), `log`."""
    path: Path | None
    speed: float
    panel: Panel | None = None
    model: MapModel | None = None
    prior: tuple[float, float] | None = None
    log: Callable[[str], None] = print
    slopes: bool = True
    store: Store = field(init=False)
    rows: dict = field(init=False)
    walks: Walks = field(init=False)
    climbs: Climbs = field(init=False)
    seq: int = field(init=False, default=-1)    # the last frame's state taken in (feed_states), of the perception
    epoch: int | None = field(init=False, default=None)  # ... service's run (its frames counted anew: WorldState.epoch)
    taken: dict = field(init=False)             # stretches measured this run, by kind
    said: str = field(init=False, default="")   # her stat window's correlation as last said (`correlation`)

    def __post_init__(self):
        self.store = Store(self.path, self.log)
        self.rows = {"walk": self.store.walks(), "climb": self.store.climbs()}
        self.walks, self.climbs = Walks(self.model, self.slopes), Climbs()
        self.taken = {"walk": 0, "climb": 0}
        if self.panel is None and isinstance(p := self.store.data.get("panel"), dict) \
                and isinstance(p.get("speed"), (int, float)):
            self.panel = Panel(float(p["speed"]), p.get("jump"), p.get("at"))

    # --- what she's planned at ---
    @property
    def map_id(self) -> int | None:
        return getattr(self.model, "id", None)

    def speed_now(self) -> float:
        """Her SPEED as her stat window last read it, else her profile's."""
        return self.panel.speed if self.panel is not None else self.speed

    def prior_of(self, kind: str) -> float:
        """Her `kind`'s ('walk', 'climb') prior: the client's formula at speed_now (physics.walk_of, climb_of), or the
        one given (`prior`)."""
        if self.prior is not None:
            return float(self.prior[0 if kind == "walk" else 1])
        s = self.speed_now()
        return physics.walk_of(s) if kind == "walk" else physics.climb_of(s)

    def values(self, kind: str) -> list[float]:
        """Her `kind`'s stretches as measured at her SPEED now (speed_now): those at it alone if MIN_N, else all, each
        at another SPEED scaled by the formula ((SPEED + 20)/(its SPEED + 20): the walk's and the climb's alike)."""
        s = self.speed_now()
        col = -2  # (a row's value: a walk's limit, a climb's v)
        rows = self.rows[kind][-KEEP:]
        same = [r[col] for r in rows if r[-1] is None or abs(r[-1] - s) < SAME_SPEED]
        if len(same) >= MIN_N[kind] or len(same) == len(rows):
            return same
        bonus = body.SPEED_BONUS
        return [r[col] * ((s + bonus) / (r[-1] + bonus) if r[-1] is not None else 1.0) for r in rows]

    def measured(self, kind: str) -> Estimate | None:
        return estimate(self.values(kind))

    def target(self, kind: str) -> tuple[float, str, Estimate | None]:
        """What her planner should have for `kind`: (units/s, why: 'measured' or 'prior', the measure)."""
        est = self.measured(kind)
        prior = self.prior_of(kind)
        if est is not None and est.se is not None and (
                (est.n >= MIN_N[kind] and est.se <= MAX_SE[kind])
                or (est.n >= GROSS_N[kind] and est.se <= GROSS_SE[kind] and abs(est.value - prior) >= GROSS * prior)):
            return round(round(est.value / ROUND) * ROUND, 2), "measured", est
        return prior, "prior", est

    def planned(self) -> tuple[float, float]:
        """Her walk and climb as her planner has them, or will at the next settle (physics.pending)."""
        return physics.pending() or (physics.WALK_SPEED, physics.CLIMB_SPEED)

    # --- the run ---
    def start(self) -> None:
        """At a hunter's start (no chain of hers runs yet): her planner's walk and climb made as her rows say (or her
        prior), at once (physics.set_motion), said where anything is kept or changes (a hunt with nothing kept, at her
        prior, says nothing: the tests' and the simulator's but for a prior given), or her file couldn't be read."""
        old = (physics.WALK_SPEED, physics.CLIMB_SPEED)
        (walk, why_w, ew), (climb, why_c, ec) = self.target("walk"), self.target("climb")
        changed = physics.set_motion(walk, climb)
        if changed:
            self._made(old, (physics.WALK_SPEED, physics.CLIMB_SPEED), why="loaded" if "measured" in (why_w, why_c)
                       else "prior")
        unread = self.path is not None and self.store.why and self.store.why != Store.NONE
        if changed or self.rows["walk"] or self.rows["climb"] or unread:
            self.log(f"her walk {physics.WALK_SPEED:g} and climb {physics.CLIMB_SPEED:g} units/s: {self.words()}"
                     + (f" ({self.store.why})" if unread else ""))

    def words(self) -> str:
        """Where her walk and climb are from, in a few words."""
        out = []
        for kind in ("walk", "climb"):
            value, why, est = self.target(kind)
            if why == "measured":
                out.append(f"her {kind} measured, {est.words()}")
            else:
                n = 0 if est is None else est.n
                src = ("given" if self.prior is not None else f"her stat window's SPEED {self.panel.speed:g}"
                       if self.panel is not None else f"her profile's SPEED {self.speed:g}")
                out.append(f"her {kind} {value:g} by {src} (the client's formula; {n} of {MIN_N[kind]} stretches "
                           "measured)")
        return "; ".join(out)

    def feed_states(self, states) -> None:
        """The frames' states not taken in yet (by `seq`), in order: her places to the estimators."""
        for st in states:
            seq, epoch = getattr(st, "seq", None), getattr(st, "epoch", 0)
            if epoch != self.epoch:  # (another run of the service's: its frames counted from 1 again)
                self.epoch, self.seq = epoch, -1
                self.walks.end()
                self.climbs.end()
            if seq is None or seq <= self.seq:
                continue
            self.seq = seq
            if (q := place_of_state(st)) is not None:
                self.feed(q)

    def feed(self, q: Place) -> None:
        """One place of hers: a stretch it ends kept (and her file saved), her planner's walk or climb proposed again if
        its measure moved enough (`judge`)."""
        changed = False
        if (w := self.walks.add(q)) is not None:
            self._keep("walk", [round(time.time(), 1), self.map_id, w.platform, w.way, w.span, w.v, w.ground, w.limit,
                                self.panel.speed if self.panel is not None else None])
            changed = True
        if (c := self.climbs.add(q)) is not None:
            self._keep("climb", [round(time.time(), 1), self.map_id, c.way, c.span, c.v,
                                 self.panel.speed if self.panel is not None else None])
            changed = True
        if changed:
            self.judge()

    def _keep(self, kind: str, row: list) -> None:
        self.rows[kind] = (self.rows[kind] + [row])[-KEEP:]
        self.taken[kind] += 1
        self.save()

    def judge(self) -> bool:
        """Her planner's walk and climb proposed again (physics.propose) where her measure (or prior) has moved DEADBAND
        from what's planned; whether it was."""
        walk_now, climb_now = self.planned()
        walk, _, _ = self.target("walk")
        climb, _, _ = self.target("climb")
        walk = walk if abs(walk - walk_now) >= DEADBAND else walk_now
        climb = climb if abs(climb - climb_now) >= DEADBAND else climb_now
        if (walk, climb) == (walk_now, climb_now):
            return False
        physics.propose(walk, climb, then=lambda old, new: self._made(old, new, why=None))
        return True

    def panel_seen(self, speed: float | None, jump: float | None = None) -> list[str]:
        """Her stat window read SPEED `speed` (and JUMP): her prior from it, her rows at another SPEED counted at it by
        the formula, her planner's walk and climb proposed again (judge); what to say of it (her stat-window note:
        behavior.hunter._stats_read): a change it makes, and her measure's correlation with it (`correlation`)."""
        if speed is None:
            return []
        old = self.panel
        self.panel = Panel(float(speed), None if jump is None else float(jump), round(time.time(), 1))
        moved = old is not None and abs(old.speed - self.panel.speed) >= SAME_SPEED
        notes = []
        before = self.planned()
        if self.judge():
            after = self.planned()
            whys = {self.target(kind)[1] for kind in ("walk", "climb")}
            notes.append(f"her SPEED {speed:g} by her stat window" + (f" ({old.speed:g} before)" if moved else "")
                         + f": her walk {before[0]:g} -> {after[0]:g} and climb {before[1]:g} -> {after[1]:g} units/s "
                         "from her next move (" + ("the client's formula" if whys == {"prior"} else
                                                   "as measured, what was measured at another SPEED counted at this "
                                                   "one by the client's formula until measured again") + ")")
        elif moved:
            notes.append(f"her SPEED {speed:g} by her stat window, {old.speed:g} before: her walk and climb measured at "
                         f"{old.speed:g} counted at {speed:g} by the client's formula until measured again")
        self.save()
        notes.append(self.correlation(force=True))
        return [n for n in notes if n]

    def correlation(self, force: bool = False) -> str:
        """Her stat window's SPEED against what her measured walk and climb imply by the client's formula (SPEED_TOL),
        in words; "" where nothing's known, or (unless `force`) as said last."""
        if self.panel is None:
            return ""
        parts, warn = [], []
        for kind, implied_of in (("walk", physics.speed_of_walk), ("climb", physics.speed_of_climb)):
            value, why, est = self.target(kind)
            if why != "measured":
                continue
            implied = implied_of(est.value)
            off = implied - self.panel.speed
            parts.append(f"her {kind} {est.value:.1f} implies SPEED {implied:.1f}")
            if abs(off) > SPEED_TOL:
                warn.append(kind)
        if not parts:
            return ""
        text = (f"{'; '.join(parts)}, her stat window's {self.panel.speed:g}: "
                + (f"off by more than {SPEED_TOL:g} (a buff, gear, the map's physics or the formula wrong?): her moves "
                   "planned at her measure all the same" if warn else f"within {SPEED_TOL:g}"))
        if not force and text == self.said:
            return ""
        self.said = text
        if warn:
            telemetry.emit("calibration", what=",".join(warn), warn=text, panel=self.panel.speed,
                           implied=round(physics.speed_of_walk(self.target("walk")[2].value), 2)
                           if "walk" in warn else round(physics.speed_of_climb(self.target("climb")[2].value), 2),
                           map=self.map_id)
        return text

    def _made(self, old: tuple[float, float], new: tuple[float, float], why: str | None) -> None:
        """Her planner's walk or climb changed (old, new: (walk, climb)): a 'calibration' record each, a line in her
        log, her file saved, her stat window's correlation said if it changed."""
        for i, kind in enumerate(("walk", "climb")):
            if old[i] == new[i]:
                continue
            value, w, est = self.target(kind)
            implied = (physics.speed_of_walk if kind == "walk" else physics.speed_of_climb)(new[i])
            telemetry.emit("calibration", what=kind, old=round(old[i], 2), new=round(new[i], 2),
                           why=why or w, n=0 if est is None else est.n, se=None if est is None else est.se,
                           prior=round(self.prior_of(kind), 2), panel=None if self.panel is None else self.panel.speed,
                           implied=round(implied, 2), map=self.map_id)
            self.log(f"her {kind} {old[i]:g} -> {new[i]:g} units/s ({why or w}"
                     + ("" if est is None else f": {est.words()}") + f"; SPEED {implied:.1f} by the formula"
                     + ("" if self.panel is None else f", her stat window's {self.panel.speed:g}") + ")")
        self.save()
        if note := self.correlation():
            self.log(f"    {note}")

    def record(self) -> dict:
        """Her file's contents now (Store)."""
        out = {"version": VERSION, "about": "her walk and climb as perception's place track measured them "
                                            "(gamelab.motion.calibrate)", "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
               "map": self.map_id, "panel": None if self.panel is None else
               {"speed": self.panel.speed, "jump": self.panel.jump, "at": self.panel.at}}
        for kind in ("walk", "climb"):
            value, why, est = self.target(kind)
            entry = {"planned": round((physics.WALK_SPEED, physics.CLIMB_SPEED)[kind == "climb"], 2), "why": why,
                     "prior": round(self.prior_of(kind), 2)}
            if est is not None:
                entry.update(value=est.value, n=est.n, se=est.se, q1=est.q1, q3=est.q3, implied=round(
                    (physics.speed_of_walk if kind == "walk" else physics.speed_of_climb)(est.value), 2))
            if kind == "climb":
                for way, name in ((UP, "up"), (DOWN, "down")):
                    e = estimate([r[-2] for r in self.rows["climb"][-KEEP:] if r[2] == way])
                    if e is not None:
                        entry[name] = {"value": e.value, "n": e.n, "se": e.se}
            out[kind] = entry
        out["walks"], out["climbs"] = self.rows["walk"], self.rows["climb"]
        return out

    def save(self) -> bool:
        return self.store.save(self.record())
