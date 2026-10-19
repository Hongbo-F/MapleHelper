"""What happened in a recording: which way each jump went, which way each throw went,
whether keys went out on time, and whether the game kept a steady frame rate."""

from dataclasses import dataclass

import numpy as np

from maplehelper.engine import RIGHT

from .. import directions
from .recordings import Recording
from .scroll import world_x
from .track import airborne, swings, track

UP_PX = 12  # a jump moving less than this (scale-2 px, world terms) went straight up; normal is ~45-65
TAKEOFF_LEAD_MS = 20  # a jump's start x is read this long before its takeoff was seen
# A throw's swing shows ~0.5 s after its key, so a combo's throws are looked for this much after its start
# and end.
SWING_SHIFT_MS = 300
NEW_FRAME_DIFF = 0.05  # mean pixel difference above which a frame is a new game frame
LONG_GAP_MS = 45       # gaps between new game frames longer than this are listed


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

    back_sign = -directions.sign(facing)
    front_side = "R" if facing == RIGHT else "L"
    all_swings = swings(rec.frames, rows, template.shape[1])
    spans = airborne(rows)
    bounds = list(starts) + [end_ms]
    out = []
    for start, end in zip(bounds, bounds[1:]):
        jumps = []
        for takeoff, landing in spans:
            if start <= takeoff < end:
                dx = round(x_at(landing if landing is not None else times[-1]) - x_at(takeoff - TAKEOFF_LEAD_MS), 1)
                direction = "up" if abs(dx) < UP_PX else ("back" if np.sign(dx) == back_sign else "forward")
                jumps.append(Jump(takeoff, landing, dx, direction))
        throws = [Throw(s["t"], s["side"], s["side"] == front_side)
                  for s in all_swings if start + SWING_SHIFT_MS <= s["t"] < end + SWING_SHIFT_MS]
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
    changed = [0] + [i for i in range(1, len(f)) if np.abs(f[i] - f[i - 1]).mean() > NEW_FRAME_DIFF]
    updates = np.asarray(rec.times)[changed]
    gaps = np.diff(updates)
    if not len(gaps):
        return {"frames": len(changed), "median_ms": None, "p95_ms": None, "max_ms": None, "long_gaps": []}
    return {"frames": len(changed), "median_ms": float(np.median(gaps)), "p95_ms": float(np.percentile(gaps, 95)),
            "max_ms": float(gaps.max()),
            "long_gaps": [(float(updates[i]), float(g)) for i, g in enumerate(gaps) if g > LONG_GAP_MS]}
