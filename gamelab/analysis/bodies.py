"""Where the monster detector puts a monster, checked offline (`python -m gamelab bodies`): no game, no keys.

- **Feet, on saved screenshots:** each monster placed on the map (the caller's Senses look), its feet
  against the surface of the platform it was put on: by its template's bottom (as before the templates
  knew where the feet are in them) and by the template's origin. Looks where she isn't standing on the
  ground are left out, as her own height then shifts every monster seen.
- **x from frame to frame, on recordings** (half-size frames, the camera still: she stands): each monster
  followed from frame to frame, how far its x strays from a straight line through its neighbours' (within
  ±0.3 s), by the template's centre and by its origin; and how often the way it faces is the way it moves.

Saved recordings at 30-60 frames a second are the evidence here: a monster's x jumped up to 10 units
between frames as the winning template changed from a sprite to a cut-out whose picture is off-centre.
"""

import math
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..perception.mobs import Mob, MobDetector
from ..screen import UNITS_PER_HALF_PX

GROUNDED_PX = 6       # half-size px of ground under her feet that show she's standing (navigator.MIN_GROUND_PX)
# Saved screenshots where she's off the ground though the look can't tell (the basin's dirt walls behind her
# read as ground): in closeup.png her name tag is 9 px above where she stands, and every monster reads 5-15
# units low.
AIRBORNE = ("closeup.png",)
LINK_PX = (12, 10)    # a monster in the next frame is the same one within this far across and down (half px) ...
LINK_GAP = 3          # ... and this many frames later at most
FIT_S = 0.3           # s either side of a detection whose others its straight line goes through ...
FIT_MIN = 7           # ... and how many it needs
MOVING = 12.0         # units/s: slower than this, a monster isn't walking (for its facing)
MIN_TRACK = 15        # detections a monster needs to count on a recording
FACING = {1: "right", -1: "left", 0: "?"}


def grounded(scene) -> bool:
    """Whether she stands on the ground in a placed look (Scene): her tag seen, on a platform, ground under her."""
    g = scene.ground
    return scene.tag_seen and scene.platform is not None and g is not None and g[1] - g[0] >= GROUNDED_PX


def is_cutout(mob: Mob) -> bool:
    """A screenshot cut-out matched (its name is <kind>_<number>), not one of the game's sprites."""
    return mob.template.split("_")[-1].isdigit()


def feet_rows(scene, model) -> list[dict]:
    """The monsters of a placed look (Scene, with `monsters`) with a platform: their feet's map y minus that
    platform's surface (+: below it), by the template's bottom and (None if unknown) by its origin."""
    out = []
    for m in scene.monsters:
        if m.platform is None:
            continue
        surface = model.platform(m.platform).y_at(m.x)
        mob = m.mob
        out.append({"kind": m.kind, "template": mob.template, "cutout": is_cutout(mob), "pose": mob.pose,
                    "facing": mob.facing, "platform": m.platform,
                    "bottom": scene.map_y(mob.y + mob.h // 2) - surface,
                    "origin": None if mob.feet is None else scene.map_y(mob.feet) - surface})
    return out


def feet_table(rows: list[dict]) -> list[str]:
    """dy (map units) per kind of template: the game's sprites together, each cut-out on its own."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["template"] if r["cutout"] else "game sprites", []).append(r)
    lines = [f"{'template':16s} {'n':>3s}   {'bottom: mean':>12s} {'sd':>5s} {'min':>6s} {'max':>6s}"
             f"   {'origin: mean':>12s} {'sd':>5s} {'min':>6s} {'max':>6s}"]
    for name in sorted(groups, key=lambda g: (g != "game sprites", g)):
        rs = groups[name]
        cells = []
        for key in ("bottom", "origin"):
            xs = [r[key] for r in rs if r[key] is not None]
            cells.append(_stats(xs))
        lines.append(f"{name:16s} {len(rs):3d}   {cells[0]}   {cells[1]}")
    every = [r["origin"] for r in rows if r["origin"] is not None]
    if every:
        lines.append(f"all {len(every)} with an origin: {min(every):+.1f} to {max(every):+.1f} units "
                     f"(by their bottoms {min(r['bottom'] for r in rows):+.1f} to {max(r['bottom'] for r in rows):+.1f})")
    return lines


def _stats(xs: list[float]) -> str:
    if not xs:
        return f"{'-':>12s} {'':>5s} {'':>6s} {'':>6s}"
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{statistics.mean(xs):+12.1f} {sd:5.2f} {min(xs):+6.1f} {max(xs):+6.1f}"


def link(frames: list[list[Mob]]) -> list[list[tuple[int, Mob]]]:
    """Detections followed from frame to frame: each the nearest of its kind within LINK_PX of the last
    one, at most LINK_GAP frames later. Each track is [(frame index, Mob)]."""
    done, live = [], []
    for i, found in enumerate(frames):
        used, kept = set(), []
        for track in live:
            j, last = track[-1]
            if i - j > LINK_GAP:
                done.append(track)
                continue
            near = [(abs(m.x - last.x) + abs(m.y - last.y), k) for k, m in enumerate(found)
                    if k not in used and m.kind == last.kind and abs(m.x - last.x) <= LINK_PX[0]
                    and abs(m.y - last.y) <= LINK_PX[1]]
            if near:
                k = min(near)[1]
                used.add(k)
                track.append((i, found[k]))
            kept.append(track)
        live = kept + [[(i, m)] for k, m in enumerate(found) if k not in used]
    return done + live


def _slope(ts: np.ndarray, xs: np.ndarray) -> float:
    """A straight line's slope through (ts, xs), least squares, by hand (numpy.linalg isn't safe here)."""
    dt = ts - ts.mean()
    return float((dt * (xs - xs.mean())).sum() / (dt * dt).sum())


def residuals(ts: np.ndarray, xs: np.ndarray) -> list[float]:
    """How far each x is from the straight line through the others within FIT_S s of it (FIT_MIN at least)."""
    out = []
    for k in range(len(xs)):
        near = (ts >= ts[k] - FIT_S) & (ts <= ts[k] + FIT_S)
        if near.sum() < FIT_MIN:
            continue
        t, x = ts[near], xs[near]
        out.append(float(xs[k] - (x.mean() + _slope(t, x) * (ts[k] - t.mean()))))
    return out


def recording(frames: np.ndarray, times: np.ndarray, detector: MobDetector) -> dict:
    """On one recording (half-size frames, times in ms): x's stray (map units) by the template's centre and by
    where the feet are, and facing against motion, over the monsters seen in MIN_TRACK frames or more."""
    found = [detector.detect(np.ascontiguousarray(f)) for f in frames]
    ts_all = np.asarray(times, float) / 1000
    centre, feet, agree, moving, n = [], [], 0, 0, 0
    for track in link(found):
        if len(track) < MIN_TRACK:
            continue
        n += len(track)
        ts = np.array([ts_all[i] for i, _ in track])
        centre += residuals(ts, np.array([m.x for _, m in track], float))
        feet += residuals(ts, np.array([m.foot_x for _, m in track], float))
        xs = np.array([m.foot_x for _, m in track], float)
        for k, (_, m) in enumerate(track):
            near = (ts >= ts[k] - FIT_S) & (ts <= ts[k] + FIT_S)
            if near.sum() < FIT_MIN or not m.facing:
                continue
            v = _slope(ts[near], xs[near]) * UNITS_PER_HALF_PX
            if abs(v) > MOVING:
                moving += 1
                agree += (v > 0) == (m.facing > 0)
    return {"detections": sum(len(f) for f in found), "followed": n, "centre": _spread(centre),
            "origin": _spread(feet), "moving": moving, "facing_agrees": agree}


def _spread(res: list[float]) -> dict:
    """rms, 95th percentile and largest of residuals in half px, as map units."""
    if not res:
        return {"n": 0}
    e = np.abs(np.array(res)) * UNITS_PER_HALF_PX
    return {"n": len(res), "rms": round(math.sqrt(float((e ** 2).mean())), 2),
            "p95": round(float(np.percentile(e, 95)), 1), "max": round(float(e.max()), 1)}


def run(scenes: list[tuple[str, Callable]], model, recordings: list[Path], detector: MobDetector) -> list[str]:
    """The report: feet on each of `scenes` (name, a look giving a placed Scene) where she stands, then x and
    facing on each of `recordings` (.npz)."""
    rows, skipped = [], []
    for name, look in scenes:
        s = look()
        if name in AIRBORNE or not grounded(s):
            skipped.append(name)
            continue
        rows += [dict(r, image=name) for r in feet_rows(s, model)]
    lines = [f"feet against the platform each monster was put on (map units, + below its surface), "
             f"{len(scenes) - len(skipped)} screenshots where she stands"
             + (f" (left out, her not standing: {', '.join(skipped)})" if skipped else "")]
    lines += feet_table(rows)
    poses = Counter(f"{r['pose'] or '?'} {FACING[r['facing']]}" for r in rows)
    if poses:
        lines.append("pose and facing: " + ", ".join(f"{k} {v}" for k, v in sorted(poses.items())))
    for path in recordings:
        with np.load(path) as d:
            r = recording(d["frames"], d["times"], detector)
        c, o = r["centre"], r["origin"]
        lines.append(f"{path.stem}: {r['detections']} detections, {r['followed']} followed; x off a straight "
                     f"line, by the template's centre: rms {c.get('rms', '-')}, p95 {c.get('p95', '-')}, max "
                     f"{c.get('max', '-')}; by its feet: rms {o.get('rms', '-')}, p95 {o.get('p95', '-')}, max "
                     f"{o.get('max', '-')} (units); facing the way it moved {r['facing_agrees']} of {r['moving']}")
    return lines
