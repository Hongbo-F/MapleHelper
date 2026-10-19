"""What happened in a recording: which way each jump went, which way each throw went,
whether keys went out on time, and whether the game kept a steady frame rate."""

from dataclasses import dataclass

import numpy as np

from maplehelper.engine import RIGHT

from .camera import world_x
from .client import Recording
from .track import airborne, swings, track

UP_PX = 12  # a jump moving less than this (scale-2 px, world terms) went straight up; normal is ~45-65


@dataclass
class Jump:
    takeoff: float
    landing: float | None
    dx: float          # world movement, + = right
    direction: str     # 'back' (away from the monster), 'forward' or 'up'


@dataclass
class Throw:
    t: float
    side: str          # 'L' or 'R'
    toward_front: bool


@dataclass
class ComboResult:
    start: int
    jumps: list[Jump]
    throws: list[Throw]


def combos(rec: Recording, starts: list[int], end_ms: int, facing: str, template: np.ndarray) -> list[ComboResult]:
    """Split a recording of back-to-back combos (starting at `starts`) into per-combo results.

    `facing` is the way she faced when the first combo started: 'back' means away from that.
    """
    rows = track(rec.frames, rec.times, template)
    xs = world_x(rec.frames, rows)
    times = np.asarray(rec.times)

    def x_at(t: float) -> float:
        return xs[int(np.argmin(np.abs(times - t)))]

    back_sign = -1 if facing == RIGHT else 1
    front_side = "R" if facing == RIGHT else "L"
    all_swings = swings(rec.frames, rows, rec.times, template.shape[1])
    bounds = list(starts) + [end_ms]
    out = []
    for i, start in enumerate(starts):
        end = bounds[i + 1]
        jumps = []
        for takeoff, landing in airborne(rows):
            if start <= takeoff < end:
                dx = round(x_at(landing if landing is not None else times[-1]) - x_at(takeoff - 20), 1)
                direction = "up" if abs(dx) < UP_PX else ("back" if np.sign(dx) == back_sign else "forward")
                jumps.append(Jump(takeoff, landing, dx, direction))
        # A throw's swing shows ~0.5 s after its key, so windows are shifted by 300 ms.
        throws = [Throw(s["t"], s["side"], s["side"] == front_side)
                  for s in all_swings if start + 300 <= s["t"] < end + 300]
        out.append(ComboResult(start, jumps, throws))
    return out


def send_lag(rec: Recording) -> dict:
    """How late keys went out compared with the plan (should be a few ms at most)."""
    lags = [s[0] - p[0] for p, s in zip(rec.planned, rec.sent)]
    return {"sent": len(rec.sent), "planned": len(rec.planned),
            "max_ms": max(lags, default=0.0), "mean_ms": sum(lags) / len(lags) if lags else 0.0}


def frame_pacing(rec: Recording) -> dict:
    """Gaps between new game frames (frames where the picture changed)."""
    f = rec.frames.astype(np.int16)
    changed = [0] + [i for i in range(1, len(f)) if np.abs(f[i] - f[i - 1]).mean() > 0.05]
    updates = np.asarray(rec.times)[changed]
    gaps = np.diff(updates)
    if not len(gaps):
        return {"frames": len(changed), "median_ms": None, "p95_ms": None, "max_ms": None, "long_gaps": []}
    return {"frames": len(changed), "median_ms": float(np.median(gaps)), "p95_ms": float(np.percentile(gaps, 95)),
            "max_ms": float(gaps.max()),
            "long_gaps": [(float(updates[i]), float(g)) for i, g in enumerate(gaps) if g > 45]}
