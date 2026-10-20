"""analysis/bodies.py: following monsters from frame to frame, and how far their x strays, offline."""

from types import SimpleNamespace

import numpy as np
import pytest

from gamelab.analysis import bodies
from gamelab.perception.mobs import Mob


def test_monsters_are_followed_from_frame_to_frame_by_kind_and_distance():
    frames = [[Mob("pig", 100 + 2 * i, 50, 30, 20, 0.9), Mob("snail", 300 - i, 80, 20, 10, 0.9)] for i in range(10)]
    frames[4] = frames[4][:1]                           # the snail missed for a frame ...
    frames[8].append(Mob("pig", 140, 50, 30, 20, 0.9))  # ... and another pig shows up farther on
    tracks = bodies.link(frames)
    lengths = sorted((t[0][1].kind, len(t)) for t in tracks)
    assert lengths == [("pig", 1), ("pig", 10), ("snail", 9)]


def test_x_off_a_straight_line():
    ts = np.arange(0, 1, 1 / 30)
    xs = 100 + 40 * ts
    assert max(abs(r) for r in bodies.residuals(ts, xs)) < 1e-9
    xs[15] += 3  # one frame's x 3 px off
    res = bodies.residuals(ts, xs)
    assert max(res) == pytest.approx(3 - 3 / 19, abs=0.2)  # a 19-frame window takes a little of it in


def test_feet_against_the_platform_they_were_put_on():
    platform = SimpleNamespace(y_at=lambda x: 500.0)
    model = SimpleNamespace(platform=lambda pid: platform)
    mob = Mob("pig", 200, 90, 46, 30, 0.9, ox=203.0, feet=106.0, facing=-1, pose="move", template="pig_2")
    monster = SimpleNamespace(mob=mob, kind="pig", platform=3, x=0.0)
    scene = SimpleNamespace(monsters=[monster], map_y=lambda y: 500.0 + (y - 105) * 1.4)
    row, = bodies.feet_rows(scene, model)
    assert row["cutout"] and row["bottom"] == pytest.approx(0.0) and row["origin"] == pytest.approx(1.4)


def test_a_recording_gives_how_fast_each_kind_walked(monkeypatch):
    """A pig walking right at 125 units/s, then standing: its speed where it walked, and its facing."""
    from gamelab.screen import UNITS_PER_HALF_PX
    times = np.arange(60) * 1000 / 30  # 2 s at 30 frames a second
    xs = [100 + min(t, 1000) / 1000 * 125 / UNITS_PER_HALF_PX for t in times]
    frames = [[Mob("pig", round(x), 50, 30, 20, 0.9, ox=x, feet=60.0, facing=1, pose="move")] for x in xs]
    detector = SimpleNamespace(detect=lambda f: frames[int(f[0, 0, 0])])
    pictures = np.zeros((60, 2, 2, 3), np.uint8)
    pictures[:, 0, 0, 0] = np.arange(60)
    r = bodies.recording(pictures, times, detector)
    n, median, _ = r["speeds"]["pig"]
    assert median == pytest.approx(125, abs=1) and r["facing_agrees"] == r["moving"] == n > 15
