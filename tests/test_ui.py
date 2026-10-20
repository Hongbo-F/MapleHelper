"""Telemetry and the mini-map, offline: records in, pictures out (no window, no game)."""

import numpy as np

from gamelab.localization import mapdata
from gamelab import telemetry
from gamelab.ui.render import HER, MapView
from gamelab.ui.state import State
from gamelab.ui.sources import ReplaySource
from gamelab.ui.app import ppm

from test_navigation import by_y, tree


def model():
    return mapdata.parse(1, tree(), "Test Map", {})


def test_nothing_is_recorded_without_a_recorder():
    telemetry.emit("scene", hp=1.0)  # no error, no file
    with telemetry.captured() as records:
        telemetry.emit("scene", hp=0.5)
    assert [(r["topic"], r["hp"]) for r in records] == [("scene", 0.5)] and "t" in records[0]
    assert not telemetry.recording()


def test_a_recording_can_be_followed_while_it_grows(tmp_path):
    path = telemetry.record_to("test", folder=tmp_path)
    try:
        follow = telemetry.Follower(path)
        telemetry.emit("log", text="one")
        assert [r["text"] for r in follow.poll()] == ["one"]
        assert follow.poll() == []
        telemetry.emit("log", text="two")
        telemetry.emit("log", text="three")
        assert [r["text"] for r in follow.poll()] == ["two", "three"]
    finally:
        telemetry.stop()
    assert telemetry.newest(tmp_path) == path
    assert [r["text"] for r in telemetry.read(path)] == ["one", "two", "three"]


def test_a_half_written_line_waits_for_the_rest(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"topic": "log", "text": "a"}\n{"topic": "lo', encoding="utf-8")
    follow = telemetry.Follower(path)
    assert [r["text"] for r in follow.poll()] == ["a"]
    with path.open("a", encoding="utf-8") as f:
        f.write('g", "text": "b"}\n')
    assert [r["text"] for r in follow.poll()] == ["b"]


def test_the_state_keeps_the_latest_of_each_topic_and_her_trail():
    s = State()
    s.apply({"topic": "session", "map": 1, "t": 10.0})
    s.apply({"topic": "scene", "t": 11.0, "her": {"x": 100, "y": 500, "platform": 2}, "hp": 0.9, "monsters": []})
    s.apply({"topic": "glance", "t": 11.1, "x": 120, "y": 500})
    s.apply({"topic": "route", "t": 11.2, "moves": [{"kind": "climb", "src": 2, "dst": 3, "x": 700, "land_x": 700}]})
    assert s.map == 1 and s.scene["hp"] == 0.9 and s.her()["x"] == 120 and len(s.trail) == 2
    s.apply({"topic": "trip", "t": 12.0, "result": "arrived"})
    assert s.route is None


def test_the_mini_map_draws_her_where_the_map_puts_her():
    m = model()
    view = MapView(m, width=400)
    s = State()
    c = by_y(m, 500, 400)
    s.apply({"topic": "scene", "t": 1.0, "her": {"x": 400, "y": 500, "platform": c.id, "facing": "right"},
             "hp": 0.5, "mp": 1.0, "monsters": [{"kind": "pig", "x": 600, "y": 500, "platform": c.id}],
             "loot": [{"kind": "mesos", "x": 300, "y": 500, "platform": c.id}]})
    img = view.draw(s)
    assert img.shape == (view.size[1], view.size[0], 3) and img.dtype == np.uint8
    x, y = view.pt(400, 500 - 8)
    assert tuple(img[y, x - 3]) == HER  # her dot (the facing arrow runs to the right of its centre)
    empty = view.draw(State())
    assert not np.array_equal(img, empty)


def test_the_hunt_loop_s_piles_and_rates_are_drawn():
    m = model()
    view = MapView(m, width=400)
    c = by_y(m, 500, 400)
    s = State()
    s.apply({"topic": "hunt", "t": 1.0, "mode": "hunt", "rule": "urgent pile", "objective": f"pile on {c.id}",
             "ground": c.id, "rate_here": 0.42, "rate_to_beat": [0.7, 3],
             "piles": [{"x": 300, "y": 500, "platform": c.id, "status": "seen", "hers": True, "urgent": True}]})
    img = view.draw(s)
    x, y = view.pt(300, 500 - 4)
    assert tuple(img[y - 5, x]) == (230, 60, 60)  # the urgent pile's box (DANGER), its top edge
    assert not np.array_equal(img, view.draw(State()))


def test_a_replay_plays_records_at_their_own_pace_and_can_seek():
    now = [0.0]
    records = [{"topic": "log", "t": 100.0 + i, "text": str(i)} for i in range(5)]
    src = ReplaySource(records, speed=2.0, clock=lambda: now[0])
    assert [r["text"] for r in src.poll()[1]] == ["0"]
    now[0] = 1.0  # 2 s of recording at double speed
    assert [r["text"] for r in src.poll()[1]] == ["1", "2"]
    src.paused = True
    now[0] = 5.0
    assert src.poll()[1] == []
    assert [r["text"] for r in src.seek(-10)] == ["0"]


def test_pictures_go_to_tk_as_binary_ppm():
    data = ppm(np.zeros((2, 3, 3), np.uint8))
    assert data.startswith(b"P6 3 2 255\n") and len(data) == len(b"P6 3 2 255\n") + 18


def test_a_route_can_be_previewed_offline():
    from gamelab.ui.sources import planned
    m = model()
    c, d = by_y(m, 500, 400), by_y(m, 300, 700)
    records = planned(m, c.id, 400, d.id)
    s = State()
    for r in records:
        s.apply(r)
    assert [mv["kind"] for mv in s.route["moves"]] == ["climb"]
    assert s.chain and s.chain["legs"][-1]["end"]["platform"] == d.id
    MapView(m, width=300).draw(s)


def test_an_old_sweep_track_replays_as_her_positions(tmp_path):
    import json
    from gamelab.ui.sources import load
    path = tmp_path / "sweep_track.json"
    path.write_text(json.dumps({"track": [[5.0, 100, 500, 2], [5.5, 110, 500, 2]]}), encoding="utf-8")
    s = State()
    for r in load(path, map_id=1):
        s.apply(r)
    assert s.map == 1 and s.her()["x"] == 110 and len(s.trail) == 2
