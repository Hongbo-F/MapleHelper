"""Her place from the scene, frame by frame (`Locator.see`): the camera's view registered against the map's own art
(localization.camreg on mapart's picture, drawn from the game data), her feet on screen by the cheapest cue that finds
her, and a filter carrying her where none does. Her map place is the view plus her place on screen (a map unit a px at
the game's own 1366x768). The localization prototype of 2026-10-29 (D:/MapleHelper-scratch/loc2/combined/snap/c3_full,
loc.py's Combined: within 3 px on every held-out frame of the morning's clips, where the service had 71% of the easy
ones and 48% of the hard), here with its camera from the map's art (not a picture stitched from the morning's frames,
which reached a third of the map) and its plate and body cues in C (perception.plate, perception.poses).

The cues, cheapest first, each tried only when the one before didn't place her:
  1. tag     her tag's full gray template (NameTag's) in a window about where the filter puts her (`windows`: from its
             uncertainty), then NEAR about it, its best SURE or more; the whole play area (rows to ROWS_MAX, the HUD's
             top: NameTag stops at 691) when she's lost, and every FULL_EVERY-th frame she isn't surely found, its sure
             peaks taken where her minimap dot agrees;
  2. plate   her plate's exact colours counted about the prediction (Plate.near: whatever covers part of it only takes
             pixels away), some of her name shown whole (another wearer of her plate item is refused), her dot agreeing;
             lost, the gray peaks anywhere checked by it (Plate.verify);
  3. body    her body against her pose bank (BodyFinder.find: her glasses first, then her colours against the map's art
             behind her) in the box the filter's uncertainty gives (poses.tracked_box: tight across, a cast's twin origin
             25 px away left out); lost, across where her dot is at its height (at most every WIDE_EVERY-th frame, not
             after a tracked search that took WIDE_BUDGET_MS already);
  4. carried nothing placed her: the filter's prediction stands (`how` None), its confidence going down.
The filter: her x and its speed (a Kalman filter), y on her platform's line, in a flight under gravity, or on a rope;
a take-off from a rise off the line, a landing on the first line crossed, a walk off a platform's end, a wall stopping
her; a rope told from a flight by her last three measured places (constant speed against gravity); her minimap dot a box
she must be in, and an alarm when it disagrees for long (then she's put where it is). Her feet from her tag's centre are
her plate model's (PlateModel.feet: (+1.5, -16) for her orange-ribbon plate, measured at calibration, where
eyes.tag_below_feet's 11 put her 5 px low).

Counts of frames are the prototype's at its 15 frames a second (the clips): the service's 30 count `k` = 2 of them each
(`_rate`: by the frames' gaps), and her vertical motion is read from every k-th measured place, so what they mean in time
is the same.

Where it's used (Senses: the eyes' `locator`): at the game's own 1366x768 on a map whose art can be drawn (the game data
copy here), for a character whose plate model is in her lab folder (calibration's: localization.her_cues), with ON
(config.toml's [perception] scene, true by default): Eyes.look_at places her by it, the Scene says which cue placed her
(`how`: "tag", "plate", "body", "carried") and how sure (`confidence`), and its screen <-> map conversions go by the
view (Scene.view). A frame whose view was never registered (the first frames of a map that isn't on screen) is placed as
before (NameTag, her dot, pose.Camera). Elsewhere (1920x1080, the simulator, another map's missing data) nothing
changes.
"""

import math
import time
import tomllib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..paths import CONFIG
from ..perception import plate as plates
from ..perception import poses
from ..screen import NATIVE, Geometry
from .mapdata import MapModel, Platform
from .pose import Camera, Localizer

ON = True                  # her place from the scene where it can be (config.toml's [perception] scene says otherwise)

# --- the cues (the prototype's) ---
SURE = 0.85                # her tag's gray match this good is her tag
ROWS_MAX = 700             # her tag is searched to the HUD's top (NameTag stops at 691)
NEAR = (110, 100)          # the tag's second window, px about the prediction
WIN_MIN = (24, 36)         # the first window's least half-size (across, down)
LOST_N = 4                 # frames placed by nothing before she's lost (the whole play area searched)
FULL_EVERY = 30            # ... and every this many while her tag isn't surely found (NameTag's FULL_EVERY)
P_GROW_MAX = 4             # her plate's window grows this many frames' worth at most (Plate.GROW_MAX)
PLATE_DOT = True           # a plate answer must agree with her minimap dot too
SPRITE_OK = 0.60           # a pose agreeing this much places her
SPRITE_SURE = 0.80         # ... this much, surely (its sd SD['body'], else SD['body-weak'])
SPRITE_WIDE_AFTER = 2      # frames placed by nothing before her dot's column is searched for her body
WIDE_EVERY = 8             # ... at most this often
WIDE_DOT_Y = 60            # ... its rows her dot's height +- this
WIDE_BUDGET_MS = 12.0      # ... not in a frame whose tracked body search took this long already
SD = {"tag": 0.5, "plate": 0.5, "body": 1.0, "body-weak": 2.0}   # a cue's measurement sd (px)
# --- the filter (the prototype's, tuned on part1 outside the held-out clips) ---
DOT_OFF = (-0.88, 5.52)    # her dot's map place less her feet's
DOT_SD = (7.0, 6.0)
DOT_LAG_S = 0.066          # her dot shows where she was this long before
DOT_BOX = (12.0, 12.0)     # the dot's box: she's in it
DOT_ALARM, DOT_ALARM_N = 45, 3   # the dot this far (map units) for this many frames: the alarm (lost; then to it)
ACC_SD_GROUND, ACC_SD_AIR, VY_SD_ROPE = 900.0, 150.0, 200.0
GRAVITY, FALL_SPEED, JUMP_SPEED = 2000.0, 670.0, 555.0
WALK_MAX = 140.0           # her walking speed's most (map units a second)
TAKEOFF, LAND_SNAP = 2.5, 3.0
ROPE_ACC, FALL_ACC = 700.0, 1200.0   # her vertical acceleration under ROPE_ACC: a rope; over FALL_ACC: a fall
ROPE_VY = 220.0            # ... and her speed under this (climbing: 120)
ROPE_END = 5               # her feet reach a rope's bottom + 1 while on it
STILL_N, REANCHOR_REACH = 4, 30   # a view not registered: 'in the air' but still this many measured frames -> the line
                           # within this (map units)
FEET_MARGIN = 13           # her feet across: the map's platforms' ends (in its VR box) and this much more
PROTO_DT = 1 / 15          # the prototype's frames' gap (its counts are of these)
TIMES_KEPT = 600           # frames' times kept for the service's line (20 s at 30 a second)
GAP_S = 0.5                # s without a frame (the game not on screen, frames refused) after which she's placed afresh


def configured(path: Path = CONFIG) -> bool:
    """config.toml's `[perception] scene`: whether her place comes from the scene where it can (ON by default)."""
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return ON
    on = data.get("perception", {}).get("scene", ON)
    if not isinstance(on, bool):
        raise ValueError(f"{path}: [perception] scene must be true or false, got {on!r}")
    return on


class KF1:
    """[p, v] at a constant velocity, white acceleration noise (the prototype's)."""

    def __init__(self, p: float, v: float = 0.0, pp: float = 100.0, pv: float = 1e4):
        self.p, self.v = p, v
        self.P = np.array([[pp, 0.0], [0.0, pv]])

    def predict(self, dt: float, acc_sd: float, a: float = 0.0, vmax: float | None = None) -> None:
        self.p += self.v * dt + 0.5 * a * dt * dt
        self.v += a * dt
        if vmax is not None and self.v > vmax:
            self.v = vmax
        F = np.array([[1, dt], [0, 1]])
        q = acc_sd ** 2
        Q = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        self.P = F @ self.P @ F.T + Q

    def update(self, z: float, sd: float) -> float:
        S = self.P[0, 0] + sd * sd
        K = self.P[:, 0] / S
        r = z - self.p
        self.p += K[0] * r
        self.v += K[1] * r
        self.P = self.P - np.outer(K, self.P[0, :])
        return r

    @property
    def sd(self) -> float:
        return math.sqrt(max(self.P[0, 0], 0.0))


@dataclass
class Located:
    """Her place in one frame, by the scene. Screen px are a 1366x768 look frame's."""
    view: tuple[int, int]             # the map point at the frame's top-left (screen = map - view)
    view_sure: bool                   # registered on this frame (else carried from the last sure one)
    feet: tuple[float, float] | None  # her feet on screen (x her mirror axis, y her shoes' bottom); None: not found yet
    tag: tuple[int, int] | None       # her tag's centre on screen: as matched (tag, plate), else where her feet put it
    x: float | None = None            # her feet on the map
    y: float | None = None
    platform: Platform | None = None  # the platform she stands on (None: in the air, on a rope)
    mode: str | None = None           # "ground", "air" or "rope"
    how: str | None = None            # the cue that placed her in this frame: "tag", "plate", "body"; None: carried
    confidence: float = 0.0
    unseen: int = 0                   # frames in a row nothing placed her
    box: list | None = None           # her body's box (the body cue's)
    facing: str | None = None         # ... and which way it faces
    ms: dict = field(default_factory=dict)   # what each part took (ms)
    info: dict = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        return self.mode == "ground"

    @property
    def on_rope(self) -> bool:
        return self.mode == "rope"


class Locator:
    """Her place, frame after frame, on one map (the top). `reg`: the map's camera registration (camreg.Registration);
    `template`: her tag's full gray template, or what gives it (NameTag's, cut on the first look); `plate`: her plate's
    matcher (perception.plate.Plate); `body`: her body's search (perception.poses.BodyFinder; None: no body cue)."""

    def __init__(self, model: MapModel, reg, template, plate: plates.Plate, body: poses.BodyFinder | None = None,
                 geo: Geometry = NATIVE):
        self.model, self.reg, self.plate, self.body, self.geo = model, reg, plate, body, geo
        self._template = template
        self.feet_dx, self.feet_dy = plate.feet           # her feet from her tag's centre (the plate model's)
        self.letters = max(1, plate.sizes["letters"])
        self.frame_px = max(1, plate.sizes["frame"])
        self.dots_reader = Localizer(model, geo=geo)       # her dot -> the map
        self.camera = Camera(model.bounds)                 # (a cold registration's hint where the camera moves across)
        self.walls = [(w.x1, min(w.y1, w.y2), max(w.y1, w.y2)) for w in model.walls if w.x1 == w.x2]
        left, top, right, bottom = model.bounds
        inside = [p for p in model.places if p.right >= left and p.left <= right
                  and top <= p.y_at(p.clamp((left + right) / 2)) <= bottom] or list(model.places)
        self.feet_map_x = (min(p.left for p in inside) - FEET_MARGIN, max(p.right for p in inside) + FEET_MARGIN)
        self.counts = {"frames": 0, "tag": 0, "plate": 0, "body": 0, "carried": 0, "none": 0, "view carried": 0}
        self._said = dict(self.counts)               # (`line`'s: the counts at the last line)
        self.times: deque[float] = deque(maxlen=TIMES_KEPT)   # each frame's time (ms), the newest
        self.reset()

    def template(self) -> np.ndarray:
        t = self._template() if callable(self._template) else self._template
        if t is None:
            raise RuntimeError("no tag template")
        return t

    def reset(self) -> None:
        """Nothing known of her (the camera's registration keeps its own)."""
        self.v = self.vx = None      # the view (map y, x at the frame's top-left) the filter works with
        self.view_sure = False
        self.v_corr = 0.0            # a view not registered: what her feet on a line moved it by
        self.v_anchor: list[float] = []
        self.kx = self.ky = None
        self.mode, self.plat, self.rope = None, None, None
        self.t = None
        self.unseen = 0
        self.unsure = 0              # frames since her tag was last surely found
        self.dot_bad = 0
        self.dots: list = []
        self.ignore_plat = None
        self.last_x = self.last_y = None
        self.ground = None
        self.since_wide = 99
        self.yhist: list = []        # (t, map y) of her measured places
        self.still = 0
        self.k = 1                   # frames of the service a frame of the prototype's (`_rate`)
        self._gap = None
        self.timer: dict = {}

    def lose(self) -> None:
        """A frame failed its checks (Eyes.recover): she's searched for everywhere next."""
        self.unseen = max(self.unseen, LOST_N * self.k)

    def forget(self) -> None:
        """No frame for GAP_S: where the filter had her and how she moved are forgotten (the view's registration keeps
        its own: it searches as far as it must), the next cue to find her places her afresh."""
        k, gap = self.k, self._gap
        self.reset()
        self.k, self._gap = k, gap

    # --- helpers ---------------------------------------------------------------------------------------------------
    def _rate(self, dt: float) -> None:
        """`k`: how many of these frames make one of the prototype's (its counts are of frames 1/15 s apart)."""
        self._gap = dt if self._gap is None else 0.8 * self._gap + 0.2 * dt
        self.k = max(1, int(round(PROTO_DT / max(self._gap, 1e-3))))

    def feet_screen(self) -> tuple[float, float] | None:
        if self.kx is None or self.v is None:
            return None
        return (self.kx.p - self.vx, self.y() - self.v)

    def y(self) -> float:
        if self.mode == "ground" and self.plat is not None:
            return self.plat.y_at(self.kx.p)
        return self.ky.p

    def platform_near(self, x: float, y: float, up: float = LAND_SNAP + 1, down: float = LAND_SNAP + 1):
        best = None
        for p in self.model.places:
            if p.covers(x, 1):
                d = y - p.y_at(x)
                if -up <= d <= down and (best is None or abs(d) < abs(best[0])):
                    best = (d, p)
        return best

    def land_crossing(self, x: float, y0: float, y1: float):
        best = None
        for p in self.model.places:
            if not p.covers(x, 0.5):
                continue
            if self.ignore_plat and p.id == self.ignore_plat[0]:
                continue
            ly = p.y_at(x)
            if y0 <= ly + 0.5 and y1 >= ly:
                if best is None or ly < best[0]:
                    best = (ly, p)
        return best

    def _t(self, name: str, t0: float) -> float:
        t1 = time.perf_counter()
        self.timer[name] = self.timer.get(name, 0.0) + (t1 - t0) * 1e3
        return t1

    def dot_map(self, dot: tuple[float, float]) -> tuple[float, float]:
        """Her feet's map place by her minimap dot (the eyes' dot, full-size px, the minimap's shift taken off)."""
        x, y = self.dots_reader.to_map(dot)
        return x - DOT_OFF[0], y - DOT_OFF[1]

    # --- a frame ---------------------------------------------------------------------------------------------------
    def see(self, frame: np.ndarray, t: float, dot: tuple[float, float] | None = None) -> Located | None:
        """Her place in a 1366x768 RGB frame shown at t (s), her minimap dot `dot` (full-size px, shifted as the eyes
        give it; None: not read). None if the camera's view has never been registered (the old camera places her)."""
        self.timer = {}
        T0 = time.perf_counter()
        self.counts["frames"] += 1
        dt = PROTO_DT if self.t is None else max(0.01, t - self.t)
        if self.t is not None and dt > GAP_S:   # (the game off the screen a while, frames refused): start afresh
            self.forget()
            dt = PROTO_DT
        elif self.t is not None:
            self._rate(dt)
        self.t = t
        info: dict = {}
        dmap = None if dot is None else self.dot_map(dot)
        # 1. the camera
        hint = None
        if dmap is not None and self.reg.sure_at is None:
            hint = self.camera.corner(*dmap)
        v = self.reg.see(frame, t, hint)
        t1 = self._t("cam", T0)
        if v.how == "none":
            self.counts["none"] += 1
            return None
        self.view_sure = v.sure
        if v.sure:
            self.v_corr = 0.0
        else:
            self.counts["view carried"] += 1
        self.vx, self.v = float(v.x), float(v.y) + self.v_corr
        info["view"] = (v.how, v.cost, v.margin)
        # 2. her dot
        self.dots = (self.dots + [dmap])[-4:]
        # 3. predict
        if self.kx is not None:
            self.predict(dt)
        t2 = self._t("predict", t1)
        # 4. measure
        meas = self.measure(frame, info, dmap)
        t3 = self._t("measure", t2)
        # 5. update
        self.update(meas, dmap, info)
        if self.mode == "ground" and self.plat is not None and self.kx is not None:
            self.ground = (t, self.plat.y_at(self.kx.p))
        self._t("update", t3)
        self.timer["total"] = (time.perf_counter() - T0) * 1e3
        self.times.append(self.timer["total"])
        return self.output(info, meas, v)

    def line(self) -> str:
        """Since the last line (the perception service's console line, every 10 s): the share of frames each cue placed
        her in, carried, and with the view registered, and the frames' time (median, the most)."""
        now, before = dict(self.counts), self._said
        self._said = now
        d = {k: now[k] - before.get(k, 0) for k in now}
        n = d["frames"]
        if n <= 0:
            return ""
        cues = ", ".join(f"{k} {100 * d[k] / n:.1f}%" for k in ("tag", "plate", "body", "carried") if d[k])
        sure = 100 * (1 - (d["view carried"] + d["none"]) / n)
        times = sorted(self.times)
        ms = f", {times[len(times) // 2]:.1f} ms a frame (the most {times[-1]:.1f})" if times else ""
        self.times.clear()
        return f"her place by the scene: {cues or 'none'}; views registered {sure:.1f}%{ms}"

    def wall_clamp(self, x0: float) -> None:
        """Her x after a predict, stopped by a wall (a vertical foothold covering her height) between x0 and it."""
        x1 = self.kx.p
        if x1 == x0:
            return
        y = self.y() if self.mode == "ground" else self.ky.p
        lo, hi = min(x0, x1), max(x0, x1)
        for w in self.walls:
            if lo - 0.5 <= w[0] <= hi + 0.5 and w[1] <= y - 5 <= w[2]:
                self.kx.p = w[0] - 0.5 if x1 > x0 else w[0] + 0.5
                self.kx.v = 0.0
                return

    def predict(self, dt: float) -> None:
        x0 = self.kx.p
        self._predict(dt)
        self.wall_clamp(x0)

    def _predict(self, dt: float) -> None:
        if self.mode == "ground":
            self.kx.predict(dt, ACC_SD_GROUND)
            if abs(self.kx.v) > WALK_MAX:
                self.kx.v = math.copysign(WALK_MAX, self.kx.v)
            p = self.plat
            if p is not None and not p.covers(self.kx.p, 2):   # walked off its end: falling
                self.mode = "air"
                self.ky = KF1(p.y_at(p.clamp(self.kx.p)), 0.0, 1.0, 100.0)
                self.plat = None
        if self.mode == "air":
            self.kx.predict(dt, ACC_SD_AIR)
            y0 = self.ky.p
            self.ky.predict(dt, 300.0, a=GRAVITY, vmax=FALL_SPEED)
            if self.ignore_plat and self.ky.p > self.ignore_plat[1]:
                self.ignore_plat = None
            if self.ky.v > 0:
                hit = self.land_crossing(self.kx.p, y0, self.ky.p)
                if hit is not None:
                    self.mode, self.plat = "ground", hit[1]
                    self.ky = None
                    self.kx.v *= 0.5
        elif self.mode == "rope":
            self.kx.predict(dt, 20.0)
            self.ky.predict(dt, VY_SD_ROPE)

    def windows(self) -> tuple[int, int]:
        sx = self.kx.sd
        sy = (self.ky.sd if self.mode != "ground" and self.ky is not None else 0.5) + 1.0
        return int(np.clip(3 * sx + 14, WIN_MIN[0], 90)), int(np.clip(3 * sy + 12, WIN_MIN[1], 90))

    def dot_agrees(self, fx: float, fy: float, tol: tuple[float, float] = (30, 40)) -> bool:
        d = next((x for x in reversed(self.dots[-2:]) if x is not None), None)
        if d is None or self.v is None:
            return True
        return abs(fx + self.vx - d[0]) <= tol[0] and abs(fy + self.v - d[1]) <= tol[1]

    def _box(self, frame, x0, y0, x1, y1):
        return plates.gray_best(frame, self.template(), (x0, y0, x1, y1), ROWS_MAX)

    def measure(self, frame: np.ndarray, info: dict, dmap) -> tuple | None:
        """The cues, cheapest first (the top): (how, feet x, feet y, sd, extra) or None."""
        t0 = time.perf_counter()
        k = self.k
        tpl = self.template()
        th, tw = tpl.shape
        f = self.feet_screen()
        pt = None if f is None else (f[0] - self.feet_dx, f[1] - self.feet_dy)   # her tag's centre predicted
        lost = pt is None or self.unseen >= LOST_N * k or self.dot_bad >= DOT_ALARM_N * k
        gray = None
        # 1. her tag
        sure = []
        if pt is not None:
            wx, wy = self.windows()
            m = self._box(frame, pt[0] - tw // 2 - wx, pt[1] - th // 2 - wy, pt[0] + tw // 2 + wx + 1,
                          pt[1] + th // 2 + wy + 1)
            if m is not None:
                info["tag"] = round(m[0], 3)
                if m[0] >= SURE:
                    sure.append(m)
            if not sure:
                m = self._box(frame, pt[0] - tw // 2 - NEAR[0], pt[1] - th // 2 - NEAR[1], pt[0] + tw // 2 + NEAR[0],
                              pt[1] + th // 2 + NEAR[1])
                if m is not None:
                    info["near"] = round(m[0], 3)
                    if m[0] >= SURE:
                        sure.append(m)
        peaks = None
        if not sure and (lost or self.unsure % (FULL_EVERY * k) == FULL_EVERY * k - 1):
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            peaks = plates.gray_peaks(frame, tpl, k=4, rows=ROWS_MAX, gray=gray)
            info["whole"] = [(round(p[0], 3), p[1], p[2]) for p in peaks[:2]]
            for p in peaks:
                if p[0] >= SURE and (pt is None or self.dot_agrees(p[1] + self.feet_dx, p[2] + self.feet_dy)):
                    sure.append(p)
        t0 = self._t("tag", t0)
        if sure:
            s, cx, cy = (min(sure, key=lambda m: math.hypot(m[1] - pt[0], m[2] - pt[1])) if pt is not None
                         else max(sure))
            self.unsure = 0
            return ("tag", cx + self.feet_dx, cy + self.feet_dy, SD["tag"], {"score": round(s, 3), "centre": (cx, cy)})
        self.unsure += 1
        # 2. her plate
        best = None
        if pt is not None:
            miss = min(-(-self.unseen // k), P_GROW_MAX)
            best = self.plate.near(frame, pt, miss)
            if best is not None and PLATE_DOT and not self.dot_agrees(best.x + self.feet_dx, best.y + self.feet_dy):
                info["plate_dot_refused"] = [best.x, best.y]
                best = None
        if best is None and (lost or peaks is not None):
            if peaks is None:
                import cv2
                peaks = plates.gray_peaks(frame, tpl, k=4, rows=ROWS_MAX, gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
            b3 = self.plate.verify(frame, peaks)
            if b3 is not None and (pt is None or self.dot_agrees(b3.x + self.feet_dx, b3.y + self.feet_dy)):
                best = b3
        t0 = self._t("plate", t0)
        if best is not None:
            c = best.candidate
            conf = min(1.0, 0.5 + 0.5 * max(0, c.letters - 0.5 * c.bad) / self.letters + 0.2 * c.frame / self.frame_px)
            info["plate"] = {"frame": c.frame, "letters": c.letters, "bad": c.bad, "clean": c.clean}
            return ("plate", best.x + self.feet_dx, best.y + self.feet_dy, SD["plate"],
                    {"score": round(conf, 3), "centre": (best.x, best.y)})
        # 3. her body
        if self.body is None:
            return None
        res = None
        speed = 0.0 if self.kx is None else abs(self.kx.v)
        body = self.body
        body.feet_x = (int(self.feet_map_x[0] - self.vx), int(self.feet_map_x[1] - self.vx))
        view = (int(self.vx), int(round(self.v)))
        if f is not None:
            sy = (self.ky.sd if self.mode != "ground" and self.ky is not None else 0.5) + 1.0
            box, slack = poses.tracked_box(f, self.kx.sd, sy, self.mode == "ground", speed)
            res = body.find(frame, box, view, f, slack)
            info["body_path"] = " ".join(body.paths)
            if res is not None:
                info["body"] = round(res.agree, 3)
        self.since_wide += 1
        spent = (time.perf_counter() - t0) * 1e3
        if ((res is None or res.agree < SPRITE_OK) and (self.unseen >= SPRITE_WIDE_AFTER * k or lost or f is None)
                and self.since_wide >= WIDE_EVERY * k and dmap is not None
                and (WIDE_BUDGET_MS is None or f is None or spent < WIDE_BUDGET_MS)):
            # lost: across where her dot puts her, at her dot's height
            self.since_wide = 0
            dx, dy = dmap[0] - self.vx, dmap[1] - self.v
            r2 = body.find(frame, (dx - 40, dx + 40, dy - WIDE_DOT_Y, dy + WIDE_DOT_Y), view, None)
            info["body_wide"] = None if r2 is None else round(r2.agree, 3)
            if r2 is not None and (res is None or r2.agree > res.agree + 0.05):
                res = r2
        self._t("body", t0)
        if res is not None and res.agree >= SPRITE_OK:
            return ("body", res.fx, res.fy, SD["body"] if res.agree >= SPRITE_SURE else SD["body-weak"],
                    {"score": round(res.agree, 3), "pose": res.pose.id, "facing": res.pose.facing, "box": res.box})
        return None

    # --- the state from a first measurement --------------------------------------------------------------------------
    def init_from(self, fx: float, fy: float) -> None:
        mx = fx + self.vx
        self.kx = KF1(mx, 0.0, 1.0, 400.0)
        my = fy + self.v
        near = self.platform_near(mx, my)
        if near is not None:
            self.mode, self.plat, self.ky = "ground", near[1], None
        else:
            self.mode, self.plat = "air", None
            self.ky = KF1(my, 0.0, 4.0, 1e4)

    def update(self, meas: tuple | None, dmap, info: dict) -> None:
        k = self.k
        if meas is not None:
            how, fx, fy, sd, extra = meas
            info["how"] = how
            info["meas"] = [round(fx, 2), round(fy, 2)]
            if self.kx is None:
                self.init_from(fx, fy)
                self.unseen = 0
                return
            self.unseen = 0
            mx = fx + self.vx
            pred = self.feet_screen()
            jump = 0.0 if pred is None else math.hypot(fx - pred[0], fy - pred[1])
            if how in ("tag", "plate") and jump > 60:
                info["reset"] = round(jump, 1)
                self.kx = KF1(mx, 0.0, 1.0, 400.0)
                my = fy + self.v
                near = self.platform_near(mx, my)
                if near is not None:
                    self.mode, self.plat, self.ky = "ground", near[1], None
                else:
                    self.mode, self.plat, self.ky = "air", None, KF1(my, 0.0, 4.0, 1e4)
            elif (abs(mx - self.kx.p) > 3 * self.kx.sd + 4 and self.last_x is not None
                  and self.t - self.last_x[1] <= 0.25):
                v = (mx - self.last_x[0]) / max(self.t - self.last_x[1], 0.02)
                self.kx = KF1(mx, float(np.clip(v, -400, 400)), sd * sd, 60.0 ** 2)
                info["surprise_x"] = round(v, 1)
            else:
                self.kx.update(mx, sd)
            self.last_x = (mx, self.t)
            my = fy + self.v
            if "reset" in info:
                self.yhist = []
            self.yhist = (self.yhist + [(self.t, my)])[-(2 * k + 2):]
            self.update_y(my, sd, how, info)
            self.last_y = (my, self.t)
        else:
            self.unseen += 1
            info["how"] = None
        # her dot: coarse, always (a box she's in; an alarm when it's far for long)
        if dmap is not None and self.kx is not None:
            vy = self.ky.v if (self.ky is not None and self.mode != "ground") else 0.0
            ex = dmap[0] - (self.kx.p - self.kx.v * DOT_LAG_S)
            ey = dmap[1] - (self.y() - vy * DOT_LAG_S)
            box = (DOT_BOX[0] + 0.5 * abs(self.kx.v) * DOT_LAG_S, DOT_BOX[1] + 0.5 * abs(vy) * DOT_LAG_S)
            far = abs(ex) > DOT_ALARM or abs(ey) > DOT_ALARM
            self.dot_bad = self.dot_bad + 1 if far else 0
            info["dot_err"] = (round(ex, 1), round(ey, 1))
            if meas is None:
                if box[0] < abs(ex) <= 3 * DOT_SD[0] + 3 * self.kx.sd + box[0]:
                    self.kx.update(self.kx.p + ex - math.copysign(box[0], ex), 3.0)
                    info["dot_pull"] = round(ex, 1)
                if self.mode in ("air", "rope") and box[1] < abs(ey) <= 60:
                    self.ky.update(self.ky.p + ey - math.copysign(box[1], ey), 3.0)
                if far and self.dot_bad >= 2 * DOT_ALARM_N * k:
                    info["to_dot"] = True
                    self.kx = KF1(dmap[0], 0.0, DOT_SD[0] ** 2, 1e4)
                    near = self.model.platform_under(dmap[0], dmap[1] - 25, reach=50)
                    if near is not None:
                        self.mode, self.plat, self.ky = "ground", near, None
                    else:
                        self.mode, self.plat, self.ky = "air", None, KF1(dmap[1], 0.0, DOT_SD[1] ** 2, 1e4)

    @staticmethod
    def takeoff_v(rise: float) -> float:
        if rise >= JUMP_SPEED ** 2 / (2 * GRAVITY):
            return 0.0
        tau = (JUMP_SPEED - math.sqrt(JUMP_SPEED ** 2 - 2 * GRAVITY * rise)) / GRAVITY
        return -JUMP_SPEED + GRAVITY * tau

    def update_y(self, my: float, sd: float, how: str, info: dict) -> None:
        if self.mode == "ground":
            p = self.plat
            line = p.y_at(self.kx.p)
            dd = my - line
            info["off_line"] = round(dd, 1)
            if how in ("tag", "plate") and abs(dd) <= 3 and not self.view_sure:
                # the view not registered: her feet on the line anchor it (the median of the last few)
                self.v_anchor = (self.v_anchor + [-dd])[-9:]
                if len(self.v_anchor) >= 3:
                    corr = float(np.median(self.v_anchor[-3:]))
                    if abs(corr) >= 0.75:
                        self.v += corr
                        self.v_corr += corr
                        self.v_anchor = []
                        info["v_anchor"] = round(corr, 2)
            elif dd < -TAKEOFF:
                self.mode, self.plat = "air", None
                self.ky = KF1(my, self.takeoff_v(-dd), sd * sd, 150.0 ** 2)
                info["event"] = "takeoff"
            elif dd > TAKEOFF + 1:
                near = self.platform_near(self.kx.p, my)
                r = self.rope_at_top(self.kx.p, line)
                if near is not None and near[1] is not p:
                    self.plat = near[1]
                elif r is not None:
                    # down at a rope's top: she climbs down it at a constant speed (a fall shows gravity, and the
                    # kinematic check takes her off it then)
                    gap = self.t - self.last_y[1] if self.last_y is not None else PROTO_DT
                    self.mode, self.plat, self.rope = "rope", None, r
                    self.ky = KF1(my, float(np.clip(dd / max(gap, 0.02), 0.0, ROPE_VY)), sd * sd, 80.0 ** 2)
                    info["event"] = "rope (down from its top)"
                else:
                    self.ignore_plat = (p.id, line + 12)
                    self.mode, self.plat = "air", None
                    self.ky = KF1(my, 100.0, sd * sd, 200.0 ** 2)
                    info["event"] = "drop"
        elif self.mode in ("air", "rope"):
            if abs(my - self.ky.p) > 3 * self.ky.sd + 4 and self.last_y is not None and self.t - self.last_y[1] <= 0.25:
                dt = max(self.t - self.last_y[1], 0.02)
                v = (my - self.last_y[0]) / dt + (0.5 * GRAVITY * dt if self.mode == "air" else 0.0)
                if self.ground is not None and self.t - self.ground[0] <= 0.15 and my < self.ground[1] - TAKEOFF:
                    v = self.takeoff_v(self.ground[1] - my)
                self.ky = KF1(my, float(np.clip(v, -700, FALL_SPEED)), sd * sd, 80.0 ** 2)
                info["surprise_y"] = round(v, 1)
            else:
                self.ky.update(my, sd)
            near = self.platform_near(self.kx.p, self.ky.p, up=1.5, down=2.5)
            if near is not None and self.ky.v >= -50 and not (self.ignore_plat and near[1].id == self.ignore_plat[0]):
                self.mode, self.plat, self.ky = "ground", near[1], None
                info["event"] = "landed"
                return
            kin = self.kin()
            if self.mode == "air" and not self.view_sure and how in ("tag", "plate") and self.reanchor(my, kin, info):
                return
            if self.mode == "air":
                on_line = self.platform_near(self.kx.p, self.ky.p, up=6, down=6) is not None
                for r in self.model.ropes:
                    # on a rope she moves at a constant speed (climbing, or still); in the air gravity shows: her
                    # last three measured places tell them apart. Or a jump onto it: the filter's rise slowed under
                    # the climbing speed, its sd small.
                    if (abs(self.kx.p - r.x) <= 4 and r.top - 5 <= self.ky.p <= r.bottom + ROPE_END
                            and abs(self.kx.v) < 60 and not on_line):
                        steady = kin is not None and abs(kin[1]) <= ROPE_ACC and abs(kin[0]) <= ROPE_VY
                        grab = abs(self.ky.v) < 160 and self.ky.sd < 3
                        if steady or grab:
                            self.mode, self.rope = "rope", r
                            if steady:
                                self.ky.v = kin[0]
                            info["event"] = "rope" + ("" if steady else " (grabbed)")
                            break
            elif self.mode == "rope":
                r = self.rope
                top = self.platform_near(self.kx.p, self.ky.p, up=3, down=3)
                if abs(self.kx.p - r.x) > 8 or self.ky.p < r.top - 30:
                    self.mode, self.rope = "air", None
                    info["event"] = "off rope"
                elif self.ky.p <= r.top + 3 and top is not None:
                    self.mode, self.plat, self.ky, self.rope = "ground", top[1], None, None
                    info["event"] = "landed (rope top)"
                elif self.ky.p > r.bottom + ROPE_END or (kin is not None and kin[1] >= FALL_ACC and kin[0] > 30):
                    self.mode, self.rope = "air", None
                    info["event"] = "off rope (falling)"

    def kin(self) -> tuple[float, float] | None:
        """Her vertical speed and acceleration (map units/s, /s^2) from three of her last measured places (every k-th:
        the prototype's frames apart), or None."""
        k = self.k
        h = self.yhist[::-1][:2 * k + 1:k][::-1]
        if len(h) < 3 or h[2][0] - h[0][0] > 0.3 * k:
            return None
        (t1, y1), (t2, y2), (t3, y3) = h
        if t2 - t1 < 0.01 or t3 - t2 < 0.01:
            return None
        v12, v23 = (y2 - y1) / (t2 - t1), (y3 - y2) / (t3 - t2)
        return v23, (v23 - v12) / (0.5 * (t3 - t1))

    def reanchor(self, my: float, kin, info: dict) -> bool:
        """The view not registered and she's 'in the air' but has stood still (vertically) for STILL_N measured frames,
        not at a rope: she's on a platform the view put her off. The nearest line within REANCHOR_REACH her dot agrees
        with: the view moved to put her feet on it."""
        near_rope = any(abs(self.kx.p - r.x) <= 6 and r.top - 10 <= my <= r.bottom + 10 for r in self.model.ropes)
        if kin is None or abs(kin[0]) > 40 or abs(kin[1]) > 400 or near_rope:
            self.still = 0
            return False
        self.still += 1
        if self.still < STILL_N * self.k:
            return False
        d = next((x for x in reversed(self.dots[-2:]) if x is not None), None)
        best = None
        for p in self.model.places:
            if p.covers(self.kx.p, 1):
                ly = p.y_at(self.kx.p)
                if abs(ly - my) <= REANCHOR_REACH and (d is None or abs(ly - d[1]) <= 30) and (
                        best is None or abs(ly - my) < abs(best[0] - my)):
                    best = (ly, p)
        self.still = 0
        if best is None:
            return False
        corr = best[0] - my
        self.v += corr
        self.v_corr += corr
        self.mode, self.plat, self.ky = "ground", best[1], None
        self.v_anchor = []
        info["v_reanchor"] = round(corr, 2)
        info["event"] = "landed (re-anchored)"
        return True

    def rope_at_top(self, x: float, line: float):
        for r in self.model.ropes:
            if abs(x - r.x) <= 4 and abs(line - r.top) <= 12:
                return r
        return None

    def output(self, info: dict, meas: tuple | None, view) -> Located:
        f = self.feet_screen()
        how = info.get("how")
        self.counts[how or "carried"] += 1
        conf = 0.0
        tag = None
        extra = meas[4] if meas is not None else {}
        if f is not None:
            conf = extra.get("score", 0.5) if how else 0.5 * 0.8 ** (self.unseen / self.k)
            if how in ("tag", "plate"):
                tag = tuple(int(v) for v in extra["centre"])
            else:
                tag = (int(round(f[0] - self.feet_dx)), int(round(f[1] - self.feet_dy)))
        if f is not None:   # (plain floats: the state goes down the service's pipe)
            f = (float(f[0]), float(f[1]))
        return Located(
            view=(int(view.x), int(round(self.v))), view_sure=bool(view.sure), feet=f, tag=tag,
            x=None if f is None else float(self.kx.p), y=None if f is None else float(self.y()),
            platform=self.plat if (self.mode == "ground" and f is not None) else None, mode=self.mode, how=how,
            confidence=round(float(conf), 3), unseen=self.unseen, box=extra.get("box"), facing=extra.get("facing"),
            ms={k: round(v, 3) for k, v in self.timer.items()}, info=info)


def build(model: MapModel, template, lab: Path, geo: Geometry = NATIVE, log=None) -> Locator | None:
    """A Locator for map `model` at geometry `geo` with her models from her lab folder (`lab`: plate_1366x768.npz,
    poses_1366x768.npz), her tag's gray template (or what gives it); None where it can't be (the top: not 1366x768, no
    plate model, the map's art can't be drawn, or turned off: `configured`)."""
    from . import camreg, mapart
    if geo != NATIVE or not configured():
        return None
    path = plates.model_path(lab, geo)
    if not Path(path).exists():
        return None
    try:
        art = mapart.load(model.id)
    except (OSError, KeyError, ValueError) as e:
        if log is not None:
            log(f"  her place from the dot and her tag: the map's art can't be drawn ({e!r})")
        return None
    reg = camreg.Registration(art)
    body = None
    bank = poses.bank_path(lab, geo)
    if Path(bank).exists():
        body = poses.BodyFinder(poses.PoseBank.load(bank), poses.Background(art.rgb, (art.x0, art.y0), art.static))
    loc = Locator(model, reg, template, plates.Plate(plates.PlateModel.load(path)), body, geo)
    if log is not None:
        from .. import native
        c = "in C" if reg.lib is not None and native.locate_lib() is not None else \
            f"partly in numpy (camreg {camreg.why or 'C'}, locate {native.locate_why or 'C'})"
        log(f"  her place by the scene (localization.locate): the camera on the map's art ({art.rgb.shape[1]}x"
            f"{art.rgb.shape[0]}), her plate's model, " + (f"her pose bank ({len(body.poses)} poses)" if body is not None
                                                            else "no pose bank (no body cue)") + f"; {c}")
    return loc
