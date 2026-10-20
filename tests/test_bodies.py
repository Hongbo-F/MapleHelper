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


def test_a_recording_tells_how_often_the_pose_was_what_the_monster_did():
    """A pig walking 1 s then standing 1 s, its pose wrong for the last 10 frames of each."""
    from gamelab.screen import UNITS_PER_HALF_PX
    times = np.arange(60) * 1000 / 30
    xs = [100 + min(t, 1000) / 1000 * 125 / UNITS_PER_HALF_PX for t in times]
    pose = ["move"] * 20 + ["stand"] * 10 + ["stand"] * 20 + ["move"] * 10
    frames = [[Mob("pig", round(x), 50, 30, 20, 0.9, ox=x, feet=60.0, facing=1, pose=p, template="pig_move1")]
              for x, p in zip(xs, pose)]
    detector = SimpleNamespace(detect=lambda f: frames[int(f[0, 0, 0])])
    pictures = np.zeros((60, 2, 2, 3), np.uint8)
    pictures[:, 0, 0, 0] = np.arange(60)
    poses = bodies.recording(pictures, times, detector)["poses"]
    walking = (poses.get(("sprites", "walking", True), 0), poses.get(("sprites", "walking", False), 0))
    standing = (poses.get(("sprites", "standing", True), 0), poses.get(("sprites", "standing", False), 0))
    assert walking[0] >= 15 and 5 <= walking[1] <= 15 and standing[0] >= 10 and 5 <= standing[1] <= 15
    lines = bodies.pose_lines(poses)
    assert lines[0].startswith("   walking (x moving over 30 units/s), the pose said move: sprites")



def test_watch_tracks_velocities_against_how_their_x_moved(tmp_path, capsys):
    from gamelab import telemetry
    from gamelab.__main__ import main
    records = []
    for k in range(6):  # twice a second: one pig standing (said still), one walking 120 units/s (said 125)
        t = 100.0 + 0.5 * k
        records.append({"t": t, "topic": "tracks", "list": [
            {"id": 1, "state": "live", "x": 500.0, "vx": 0.0},
            {"id": 2, "state": "live", "x": 800.0 + 60.0 * k, "vx": 125.0 if k < 4 else -125.0}]})
    lines = bodies.tracks_lines(records)
    assert "in 5 of 5 (100%)" in lines[0] and "in 3 of 5 (60%)" in lines[1]  # the last two said it walked back
    path = tmp_path / "20261020-120000-watch.jsonl"
    path.write_text("\n".join(__import__("json").dumps(r) for r in records) + "\n", encoding="utf-8")
    assert main(["bodies", "--tracks", str(path)]) == 0
    assert "tracks that stood" in capsys.readouterr().out
    assert telemetry.read(path) == records
