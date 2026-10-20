"""Timing a look, stage by stage (`python -m gamelab bench-look`): on saved screenshots (offline: no game,
no keys), or on the game as it is on screen (`--live`). Live, it only reads the screen: it never sends a
key and never brings the game to the front, and it refuses to run unless the game is in front already.

Saved frames go through the same capture path as the game's: each is turned back into the BGRA picture a
screen capture gives, and Eyes converts that as it would a capture. What offline numbers leave out is the
grab itself: mss waits for the compositor's next frame, 16.7 ms whatever the size of the box; DXGI desktop
duplication waits for the next frame the screen shows, and copying the game's 1920x1080 off the graphics
card takes about 2 ms.

A frame is timed `reps` times in a row (after one look at it that isn't timed), as a run of looks at a
scene that barely changes from one to the next.
"""

import statistics
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .. import capture, window
from ..paths import LAB
from ..perception.eyes import EXP_TEXT, TAG_BELOW_FEET, Eyes
from ..perception.hud import EXP_NUMBER, bars
from ..perception.lie_detector import LieDetectorSpotter
from ..perception.minimap import her_dot
from ..perception.terrain import find_ropes, ground_extent
from ..screen import GAME_SIZE, SCALE

# Saved screenshots of her on Thicket Around the Beach II (gamelab/lab), with monsters, ropes and drops in
# view: the default frames.
FRAMES = ("z2_shoot1.png", "z2_shoot2.png", "z2_shoot3.png", "z2_shoot4.png", "z2_shoot5.png", "z2_shoot6.png",
          "z1_after.png", "z1_hop.png", "z1_range.png", "z2_down.png", "z2_loot1.png", "z2_loot2.png",
          "stand.png", "loot.png", "sit.png", "after_rest.png", "closeup.png")


def saved_frames() -> list[Path]:
    """The default frames that are in gamelab/lab."""
    return [LAB / name for name in FRAMES if (LAB / name).exists()]


def load(paths: list[Path]) -> list[tuple[str, np.ndarray]]:
    """Saved screenshots (at the game's size), each as the BGRA picture a screen capture gives."""
    out = []
    for path in paths:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        if (rgb.shape[1], rgb.shape[0]) != GAME_SIZE:
            raise RuntimeError(f"{path} is {rgb.shape[1]}x{rgb.shape[0]}; a look needs {GAME_SIZE[0]}x{GAME_SIZE[1]}")
        out.append((Path(path).name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)))
    return out


class SavedScreen:
    """Stands in for the screen (capture.Grabber): shots of the frame on show (`bgra`), at once."""

    def __init__(self):
        self.bgra: np.ndarray | None = None

    def shot(self, area: capture.Area, after: float | None = None, at: float | None = None) -> capture.Shot:
        return SavedShot(self.bgra, area)


class SavedShot(capture.Shot):
    """Boxes of a saved frame, each a picture of its own like a capture's."""

    def __init__(self, bgra: np.ndarray, area: capture.Area):
        super().__init__(area, time.monotonic())
        self.bgra = bgra

    def crops(self, boxes, convert=None):
        x0, y0 = self.area["left"], self.area["top"]
        views = [self.bgra[y0 + y:y0 + y + h, x0 + x:x0 + x + w] for x, y, w, h in boxes]
        return [np.ascontiguousarray(v) if convert is None else convert(v) for v in views]


class SavedEyes(Eyes):
    """Eyes whose captures come from saved frames (`show`), never from the screen: no window is touched."""

    def __init__(self, tag_template: np.ndarray, detector=None):
        super().__init__("saved frames", tag_template, detector, screen=SavedScreen())

    def show(self, bgra: np.ndarray) -> None:
        self.screen.bgra = bgra

    def _to_front(self) -> None:
        pass

    def _client_area(self) -> dict[str, int]:
        h, w = self.screen.bgra.shape[:2]
        return {"left": 0, "top": 0, "width": w, "height": h}


class FrontEyes(Eyes):
    """Eyes on the game as it is: a capture refuses unless the game is in front already. It never brings
    it forward, so it can run while someone plays."""

    def _to_front(self) -> None:
        if window.foreground_title() != self.title:
            raise RuntimeError("the game isn't in front; bring it forward yourself (bench-look never does)")


class Timer:
    """Named stages, timed: each `time` call adds a sample. Also counts of things seen (`count`)."""

    def __init__(self):
        self.samples: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}

    def count(self, name: str, n: int) -> None:
        self.counts[name] = self.counts.get(name, 0) + n

    def add(self, name: str, seconds: float) -> None:
        self.samples.setdefault(name, []).append(seconds)

    def time(self, name: str, fn: Callable):
        t0 = time.perf_counter()
        out = fn()
        self.add(name, time.perf_counter() - t0)
        return out

    def table(self) -> list[str]:
        """One line per stage: how many samples, and the median, 90th percentile and fastest, in ms."""
        lines = [f"{'stage':40s} {'n':>5s} {'median':>8s} {'p90':>8s} {'min':>8s}  ms"]
        for name, xs in self.samples.items():
            xs = sorted(xs)
            lines.append(f"{name:40s} {len(xs):5d} {1000 * statistics.median(xs):8.2f} "
                         f"{1000 * xs[int(0.9 * (len(xs) - 1))]:8.2f} {1000 * xs[0]:8.2f}")
        return lines + [f"{name}: {n}" for name, n in self.counts.items()]


def stages(eyes: Eyes, timer: Timer, band: tuple[float, float], spotter: LieDetectorSpotter) -> None:
    """Each part of a look once, on what `eyes` captures now; then Eyes.look and the questions about it
    (`band`: map units above and below her feet searched for monsters)."""
    full = timer.time("capture (frame as RGB)", eyes._capture)
    if not isinstance(eyes, SavedEyes):  # live: how long ago the screen showed it
        timer.add("capture: the frame's age once RGB", time.monotonic() - eyes.seen_at)
    timer.time("HP and MP bars", lambda: bars(full))
    half = timer.time("half size", lambda: cv2.resize(full, (full.shape[1] // SCALE, full.shape[0] // SCALE),
                                                      interpolation=cv2.INTER_AREA))
    tag = timer.time("name tag", lambda: eyes.name_tag.find(full))
    timer.time("name tag: whole play area", lambda: eyes.name_tag.search(full))
    if tag is not None:
        timer.time("ground under her", lambda: ground_extent(half, tag[0], tag[1] - TAG_BELOW_FEET))
    dot = timer.time("minimap dot", lambda: her_dot(full))
    whole = timer.time("ropes: whole screen", lambda: find_ropes(full))
    if eyes.rope_columns is not None:  # Senses gives where the map's ropes can show
        columns = eyes.rope_columns(dot, tag) or []
        near = timer.time("ropes: where the map's can show", lambda: find_ropes(full, columns))
        inside = [r for r in whole if any(x0 <= r.x < x1 for x0, x1 in columns)]
        timer.count("ropes seen where the map's can show", len(inside))
        timer.count("... of them missed by the search there", sum(not any(abs(r.x - n.x) <= 2 for n in near)
                                                                  for r in inside))
    x0, y0, x1, y1 = EXP_NUMBER
    timer.time("EXP number", lambda: full[y0:y1, x0:x1].max(axis=2) > EXP_TEXT)
    timer.time("Lie Detector check (the player's)", lambda: spotter.seen(full))
    timer.time("glance (her dot and HP)", eyes.glance)
    obs = timer.time("Eyes.look", eyes.look)
    if obs.tag is None:
        return
    mobs = timer.time("monsters: her band", lambda: eyes.monsters(obs, *band))
    timer.time("monsters: whole screen", lambda: eyes.monsters_wide(obs))
    for mob in mobs:
        timer.time("monster HP bar (each)", lambda: eyes.bar(obs, mob))


def run(eyes: Eyes, band: tuple[float, float], looks: dict[str, Callable], frames: list | None = None,
        reps: int = 5, n: int = 100) -> Timer:
    """Time every stage, and each of `looks` (name -> a whole look, e.g. through Senses): on each of
    `frames` (name, BGRA) `reps` times, `eyes` being SavedEyes; or, with no frames, on `n` live captures."""
    timer, spotter = Timer(), LieDetectorSpotter()

    def once(t: Timer) -> None:
        tag = eyes.name_tag
        finds, missed = tag.looks, tag.missed
        stages(eyes, t, band, spotter)
        for name, look in looks.items():
            t.time(name, look)
        t.count("name tag looks", tag.looks - finds)
        t.count("... of them not near where she was last", tag.missed - missed)

    if frames is None:
        once(Timer())  # the first look loads templates and starts threads: not timed
        for _ in range(n):
            once(timer)
        return timer
    for _, bgra in frames:
        eyes.show(bgra)
        once(Timer())
        for _ in range(reps):
            once(timer)
    return timer
