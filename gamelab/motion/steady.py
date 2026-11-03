"""Her steady walks and climbs in a run's places, as perception's place track has them: what analysis.walkfit measures
her walk and climb by, offline, against the client's formula at her stat window's SPEED (motion.physics.walk_of,
climb_of). Her planner doesn't measure: her walk and climb are her stat window's SPEED's by that formula (the user,
2026-11-02, shown that the formula and the measures agree and that following a measure online mostly chases its
noise: "Just use panel, no need to measure."; README "Her walk and climb from her stat window's SPEED").

The estimator (Walks, Climbs, `measure`: fed her places one at a time, each scene record of a run's telemetry):
- her walk: stretches of her places on one platform, on the ground, placed by her name tag or the camera (PLACED_BY),
  GAP_S apart at most, moving one way at WALK_BAND units/s from place to place, STEADY_S and MIN_PLACES or more; a
  line fitted from SETTLE_S after the stretch's start (her run-up: full speed 0.106 s after the game sees the arrow) to
  TAIL_S before its end (her braking: the release's lag and her slide are still faster than WALK_BAND's floor), no
  place FIT_UNITS off it; its x speed, with Tree Dungeon's slopes taken out by the v83 rule (`ground_factor`: RustMS
  Physics.cpp move_shoe, the rule of three that fits both characters' walks, D:/MapleHelper-scratch/recfix/
  walk-research/slope_models*.txt): her walk on flat ground, the shoe's vmax. Her hour of 2026-11-02 by it: 66 walks,
  the median 149.44 (bootstrap 90% 149.1-150.0); the thief's Tree runs of 10-25 to 30: 243 walks, 151.19 (150.7-
  151.75), against the client's 150 and 151.25 by their stat windows' SPEED 100 and 101; raw, with each walk's braking
  tail in it and her x speed as it is, 148.2 (81 walks: walkfit's `--raw`).
- her climb: stretches of her places on one rope or ladder (mode 'rope', her x within ROPE_DX of the stretch's first),
  placed by PLACED_BY, GAP_S apart at most, moving one way at CLIMB_BAND, CLIMB_STEADY_S and CLIMB_PLACES or more; a
  line fitted from CLIMB_SETTLE_S after the start to CLIMB_TAIL_S before the end, no place CLIMB_FIT off it: up and
  down apart. Her hour of 2026-11-02 by it: 42 climbs up, the median 120.1, 5 down, 119.6 (SPEED + 20: 120). The
  thief's of 10-25 to 27 were placed by the minimap's dot alone (27 climbs up, 106-146 units/s, its 4-unit steps):
  none is taken.
Robust: a knock or a fall puts her in the air (her stretch ends there; a knock's slide on the ground is faster than
she walks), a misplaced look is off the line (the stretch isn't taken) or off her platform (it ends there), a stretch
is one value however long, and a run's measure is the median (`estimate`: its spread, and the median's standard error
from the median absolute deviation).
"""

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from ..localization.mapdata import MapModel, Platform

# --- the estimator: her walk -----------------------------------------------------------------------------------------
PLACED_BY = ("tag", "camera")  # her places a stretch is made of: by her name tag or the camera (not the minimap's dot)
GAP_S = 0.2                    # s between two places of a stretch at most (the perception service's scene records come
                               # every other frame, about 7.5 a second)
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
    her: sim.world). `tail`: s of each walk's end left out (TAIL_S); `gap`: s between two places of a walk at most
    (GAP_S)."""

    def __init__(self, model: MapModel | None = None, slopes: bool = True, tail: float = TAIL_S, gap: float = GAP_S):
        self.model, self.slopes, self.tail, self.gap = model, slopes, tail, gap
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
        if seg and q.t - seg[-1].t < self.gap and q.platform == seg[-1].platform:
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
    """Her steady climbs as her places come (`add`, `end`, `gap`: Walks')."""

    def __init__(self, gap: float = GAP_S):
        self.gap = gap
        self._seg: list[Place] = []

    def add(self, q: Place) -> Ascent | None:
        seg = self._seg
        if q.mode != "rope" or q.how not in PLACED_BY:
            out = self._flush()
            self._seg = []
            return out
        if seg and q.t - seg[-1].t < self.gap and abs(q.x - seg[0].x) <= ROPE_DX:
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


def measure(places: Iterable[Place], model: MapModel | None = None, slopes: bool = True, tail: float = TAIL_S,
            gap: float = GAP_S) -> tuple[list[Walk], list[Ascent]]:
    """Every steady walk and climb in `places` (in time order): Walks' and Climbs', as a run's records give them
    (analysis.walkfit)."""
    walks, climbs = Walks(model, slopes, tail, gap), Climbs(gap)
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
