"""Human-like mouse movement trails, sampled the way a mouse reports them.

Kinematics: the sigma-lognormal model (Plamondon's kinematic theory). A move is
a primary stroke that lands near the target, plus corrective strokes that
overlap it; each stroke has a lognormal speed profile (an asymmetric bell:
quick acceleration, long braking) and bends along a circular arc.
Noise on top: speed-dependent motor noise (an Ornstein-Uhlenbeck drift whose
strength scales with speed, absorbed by the last correction) and 8-12 Hz
physiological tremor while the hand moves.
Sampling: the cursor is a whole pixel; the mouse is polled at `polling_hz` and
reports only when the pixel changed.

    generate_trail(x1, y1, x2, y2, polling_hz=125.0, duration=None, seed=None)
    -> np.ndarray of rows (t_ms, x, y); the first row is (0, x1, y1), the last
       (t_end, x2, y2). With `duration` (s), the last pixel change is re-timed to
       land on it, so t_end == duration * 1000 (in ~1% of trails, where the
       end is a slow 1-px flicker, up to a few ticks earlier).

CLI: python mouse_trail.py 100 200 800 450 [--hz 125] [--duration 0.6] [--seed 1]
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

FITTS_A = 0.10        # s, Fitts' law intercept
FITTS_B = 0.12        # s/bit, Fitts' law slope
MAX_CORRECTIONS = 3
FINE_DT = 0.0005      # s, grid for finding the last pixel change
NOISE_DT = 0.001      # s, grid of the noise processes
SPAN_Z = 3.5          # a stroke runs over its lognormal's mu -/+ SPAN_Z sigma


def natural_duration(distance: float, target_width: float = 20.0) -> float:
    """Fitts' law movement time (s) for a target `target_width` px wide."""
    return FITTS_A + FITTS_B * math.log2(distance / target_width + 1.0)


def _erf(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


_SPAN_HI = float(_erf(np.array(SPAN_Z / math.sqrt(2))))
_SPAN_LO = -_SPAN_HI


@dataclass
class Stroke:
    t0: float      # s, lognormal time origin
    mu: float      # ln s
    sigma: float
    arc: float     # px, path length
    th_s: float    # rad, start direction
    th_e: float    # rad, end direction

    @classmethod
    def between(cls, p, q, start, length, sigma, bend):
        """A stroke from p to q, moving from `start` for about `length` s
        (the lognormal's mu - 3 sigma .. mu + 3 sigma), bent by `bend` rad."""
        c = math.hypot(q[0] - p[0], q[1] - p[1])
        alpha = math.atan2(q[1] - p[1], q[0] - p[0])
        arc = c * bend / (2 * math.sin(bend / 2)) if abs(bend) > 1e-6 else c
        mu = math.log(length / (math.exp(SPAN_Z * sigma) - math.exp(-SPAN_Z * sigma)))
        return cls(start - math.exp(mu - SPAN_Z * sigma), mu, sigma, arc,
                   alpha - bend / 2, alpha + bend / 2)

    @property
    def end(self) -> float:
        return self.t0 + math.exp(self.mu + SPAN_Z * self.sigma)

    def scaled(self, k: float) -> "Stroke":
        return Stroke(self.t0 * k, self.mu + math.log(k), self.sigma,
                      self.arc, self.th_s, self.th_e)

    def _z(self, t):
        u = t - self.t0
        z = np.full_like(t, -SPAN_Z)
        m = u > 0
        z[m] = (np.log(u[m]) - self.mu) / self.sigma
        return u, np.clip(z, -SPAN_Z, SPAN_Z)

    def progress(self, t: np.ndarray) -> np.ndarray:
        """0 -> 1 over mu -/+ SPAN_Z sigma: the lognormal's CDF, cut to that span."""
        _, z = self._z(t)
        return (_erf(z / math.sqrt(2)) - _SPAN_LO) / (_SPAN_HI - _SPAN_LO)

    def speed(self, t: np.ndarray) -> np.ndarray:
        u, z = self._z(t)
        out = np.zeros_like(t)
        m = np.abs(z) < SPAN_Z
        k = self.arc / ((_SPAN_HI - _SPAN_LO) / 2)
        out[m] = k * np.exp(-0.5 * z[m] ** 2) / (self.sigma * math.sqrt(2 * math.pi) * u[m])
        return out

    def displacement(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = self.progress(t)
        bend = self.th_e - self.th_s
        if abs(bend) < 1e-6:
            return self.arc * f * math.cos(self.th_s), self.arc * f * math.sin(self.th_s)
        phi = self.th_s + bend * f
        r = self.arc / bend
        return r * (np.sin(phi) - math.sin(self.th_s)), r * (math.cos(self.th_s) - np.cos(phi))


def _plan(rng, p, q, target_width):
    """Strokes in natural time: a primary that misses a little, then corrections
    until one lands exactly on q."""
    d = math.hypot(q[0] - p[0], q[1] - p[1])
    ux, uy = (q[0] - p[0]) / d, (q[1] - p[1]) / d
    nx, ny = -uy, ux
    alpha = math.atan2(uy, ux)

    mt = natural_duration(d, target_width) * rng.lognormal(0.0, 0.10)
    along = rng.normal(-0.04, 0.05) * d          # primaries tend to undershoot
    across = rng.normal(0.0, 0.02) * d
    aim = (q[0] + ux * along + nx * across, q[1] + uy * along + ny * across)
    if math.hypot(aim[0] - q[0], aim[1] - q[1]) < 0.5:
        aim = q
    bend = rng.normal(0.18 * math.sin(alpha + 0.6), 0.08)   # arm/wrist arcs
    strokes = [Stroke.between(p, aim, 0.0, 0.85 * mt, rng.uniform(0.22, 0.32), bend)]

    k = 0
    while aim != q:
        k += 1
        r = math.hypot(q[0] - aim[0], q[1] - aim[1])
        if r <= target_width / 2 or k == MAX_CORRECTIONS:
            nxt = q
        else:
            e = rng.normal(0.0, 0.15 * r, 2)
            nxt = (q[0] + e[0], q[1] + e[1])
        prev = strokes[-1]
        start = prev.t0 + math.exp(prev.mu) + rng.uniform(0.0, 0.25) * (prev.end - prev.t0)
        length = (0.10 + 0.08 * math.log2(r / target_width + 1.0)) * rng.lognormal(0.0, 0.15)
        strokes.append(Stroke.between(aim, nxt, start, length, rng.uniform(0.25, 0.40),
                                      rng.normal(0.0, 0.25)))
        aim = nxt
    return strokes, mt


class _Noise:
    """Noise processes on an absolute-time grid, so a re-timed plan keeps them."""

    def __init__(self, rng, horizon, motor_noise, tremor_px, tau=0.04):
        n = int(horizon / NOISE_DT) + 2
        self.t = np.arange(n) * NOISE_DT
        a = math.exp(-NOISE_DT / tau)
        w = rng.normal(0.0, math.sqrt(1 - a * a), (n, 2))
        ou = np.empty((n, 2))
        ou[0] = rng.normal(0.0, 1.0, 2)
        for i in range(1, n):
            ou[i] = a * ou[i - 1] + w[i]
        self.ou = ou * motor_noise
        tr = np.zeros((n, 2))
        for _ in range(3):
            f = rng.uniform(8.0, 12.0)
            amp = rng.uniform(0.5, 1.0, 2)
            ph = rng.uniform(0.0, 2 * math.pi, 2)
            tr += amp * np.sin(2 * math.pi * f * self.t[:, None] + ph)
        rms = np.sqrt((tr ** 2).mean()) or 1.0
        self.tremor = tr * (tremor_px / rms)

    def offsets(self, strokes, t_eval):
        g = self.t
        speed = sum(s.speed(g) for s in strokes)
        peak = speed.max() or 1.0
        drift = np.cumsum(self.ou * speed[:, None] * NOISE_DT, axis=0)
        drift *= (1.0 - strokes[-1].progress(g))[:, None]     # the last correction absorbs it
        env = np.clip(speed / (0.2 * peak), 0.0, 1.0)[:, None]
        total = drift + self.tremor * env
        return (np.interp(t_eval, g, total[:, 0]), np.interp(t_eval, g, total[:, 1]))


def _positions(p, strokes, noise, t):
    x = np.full_like(t, float(p[0]))
    y = np.full_like(t, float(p[1]))
    for s in strokes:
        dx, dy = s.displacement(t)
        x += dx
        y += dy
    ox, oy = noise.offsets(strokes, t)
    return x + ox, y + oy


def _last_change(p, q, strokes, noise):
    """Time (s) after which the rounded cursor stays on q."""
    t_end = max(s.end for s in strokes) + 0.01
    t = np.arange(0.0, t_end, FINE_DT)
    x, y = _positions(p, strokes, noise, t)
    off = (np.rint(x) != q[0]) | (np.rint(y) != q[1])
    idx = np.nonzero(off)[0]
    return t[min(idx[-1] + 1, len(t) - 1)] if idx.size else 0.0


def generate_trail(x1, y1, x2, y2, polling_hz: float = 125.0,
                   duration: float | None = None, seed=None, *,
                   target_width: float = 20.0, motor_noise: float = 0.04,
                   tremor_px: float = 0.35) -> np.ndarray:
    """Rows (t_ms, x, y) of mouse events from (x1, y1) to (x2, y2).

    polling_hz: reports per second; an event is emitted only when the pixel changes.
    duration:   seconds from the first row to the last; None picks one from the
                distance with Fitts' law (a `target_width`-px target, corrections
                included, +-10% person-to-moment spread), ending on the first
                poll after the cursor settles.
    seed:       anything numpy.random.default_rng accepts, or a Generator.
    motor_noise / tremor_px: 0 for a noise-free (but still randomly planned) move.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    p = (round(x1), round(y1))
    q = (round(x2), round(y2))
    if p == q:
        return np.array([[0.0, p[0], p[1]]])
    period = 1.0 / polling_hz

    strokes, mt = _plan(rng, p, q, target_width)
    goal = mt if duration is None else duration
    horizon = 2.0 * max(strokes[-1].end, goal) + 0.5
    noise = _Noise(rng, horizon, motor_noise, tremor_px)

    target = goal - FINE_DT
    for _ in range(8):            # re-time the plan so the last pixel change lands at `goal`
        t_last = _last_change(p, q, strokes, noise)
        if t_last <= 0 or abs(t_last - target) <= FINE_DT:
            break
        strokes = [s.scaled(target / t_last) for s in strokes]
    if duration is None:          # the mouse's polling isn't synced to the hand
        phase = rng.uniform(0.0, period)
        t_final = phase + max(math.ceil((t_last - phase) / period - 1e-9), 0) * period
    else:
        t_final = duration
        phase = duration % period or period

    ticks = np.arange(phase, t_final + 1e-9, period)
    ticks = ticks[ticks > 0]
    x, y = _positions(p, strokes, noise, ticks)
    xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
    xi[-1], yi[-1] = q
    rows = [(0.0, p[0], p[1])]
    for t, a, b in zip(ticks, xi, yi):
        if (a, b) != rows[-1][1:]:
            rows.append((t * 1000.0, a, b))
    return np.array(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("x1", "y1", "x2", "y2"):
        ap.add_argument(name, type=float)
    ap.add_argument("--hz", type=float, default=125.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    print("t_ms,x,y")
    for t, x, y in generate_trail(a.x1, a.y1, a.x2, a.y2, a.hz, a.duration, a.seed):
        print(f"{t:.3f},{int(x)},{int(y)}")


if __name__ == "__main__":
    main()
