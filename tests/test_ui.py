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


def test_the_newest_run_is_the_latest_started(tmp_path):
    import os
    assert telemetry.newest(tmp_path / "none yet") is None and telemetry.newest(tmp_path) is None
    older, later = tmp_path / "20261020-013922-tour.jsonl", tmp_path / "20261020-014546-tour.jsonl"
    for path in (later, older):
        path.write_text("", encoding="utf-8")
    os.utime(later, (1_792_000_000, 1_792_000_000))
    os.utime(older, (1_793_000_000, 1_793_000_000))  # the older run written to last: its name still loses
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    assert telemetry.newest(tmp_path) == later
    same = tmp_path / "20261020-014546-nav.jsonl"  # started in the same second: the one written last
    same.write_text("", encoding="utf-8")
    os.utime(same, (1_791_000_000, 1_791_000_000))
    assert telemetry.newest(tmp_path) == later
    untimed = tmp_path / "bench.jsonl"  # not named by a time: by when it was last written
    untimed.write_text("", encoding="utf-8")
    os.utime(untimed, (1_900_000_000, 1_900_000_000))
    assert telemetry.newest(tmp_path) == untimed
    os.utime(untimed, (1_700_000_000, 1_700_000_000))
    assert telemetry.newest(tmp_path) == later


def test_live_source_looks_for_a_newer_run_once_a_second(tmp_path):
    from gamelab.ui.sources import LiveSource
    first = tmp_path / "20261020-010000-tour.jsonl"
    first.write_text('{"topic": "log", "t": 1.0, "text": "a"}\n', encoding="utf-8")
    now = [100.0]
    src = LiveSource(tmp_path, clock=lambda: now[0])
    fresh, records = src.poll()
    assert fresh and [r["text"] for r in records] == ["a"]
    (tmp_path / "20261020-020000-tour.jsonl").write_text('{"topic": "log", "t": 2.0, "text": "b"}\n',
                                                          encoding="utf-8")
    with first.open("a", encoding="utf-8") as f:
        f.write('{"topic": "log", "t": 1.5, "text": "a2"}\n')
    now[0] = 100.5  # the file it follows is still read on every poll; the folder isn't looked through
    fresh, records = src.poll()
    assert not fresh and [r["text"] for r in records] == ["a2"]
    now[0] = 101.1
    fresh, records = src.poll()
    assert fresh and src.path.name == "20261020-020000-tour.jsonl" and [r["text"] for r in records] == ["b"]


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


def test_the_web_page_serves_the_map_and_the_text_with_its_key_only():
    import json
    import threading
    import urllib.error
    import urllib.request

    from gamelab.ui.sources import ReplaySource
    from gamelab.ui.web import Feed, WebServer
    m = model()
    c = by_y(m, 500, 400)
    records = [{"topic": "session", "t": 0.0, "map": 1, "command": "tour", "mode": "live"},
               {"topic": "scene", "t": 0.5, "her": {"x": 400, "y": 500, "platform": c.id, "how": "tag"}, "hp": 0.8,
                "mp": 1.0, "monsters": [{"kind": "pig", "x": 600, "y": 500, "platform": c.id}], "loot": []},
               {"topic": "hunt", "t": 0.6, "rule": "shoot", "objective": None, "ground": c.id, "rate_here": 0.5},
               {"topic": "log", "t": 0.7, "text": "pig down"},
               {"topic": "log", "t": 30.0, "text": "much later"}]
    now = [0.0]
    feed = Feed(lambda map_id: MapView(m, width=300), ReplaySource(records, clock=lambda: now[0]), live=False)
    now[0] = 1.0
    feed.step()
    server = WebServer(feed, host="127.0.0.1", port=0, key="secret")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        try:
            urllib.request.urlopen(base + "/state.json")
            raise AssertionError("served without its key")
        except urllib.error.HTTPError as e:
            assert e.code == 403
        page = urllib.request.urlopen(base + "/?k=secret")
        assert b"gamelab mini-map" in page.read() and "gk=secret" in page.headers["Set-Cookie"]
        s = json.load(urllib.request.urlopen(base + "/state.json?k=secret"))
        assert s["hp"] == 0.8 and s["hunt"]["rule"] == "shoot" and s["logs"] == ["pig down"]
        assert s["her"]["platform"] == c.id and s["monsters"] == 1 and s["replay"]["length"] == 30.0
        jpeg = urllib.request.urlopen(base + "/frame.jpg?k=secret").read()
        assert jpeg[:2] == bytes([0xFF, 0xD8])  # a JPEG
        req = urllib.request.Request(base + "/control?k=secret&action=on", method="POST")
        assert urllib.request.urlopen(req).status == 200  # 10 s on: still before "much later"
        req = urllib.request.Request(base + "/control?k=secret&action=pause", method="POST")
        urllib.request.urlopen(req)
        s2 = json.load(urllib.request.urlopen(base + "/state.json?k=secret"))
        assert s2["replay"]["paused"] and s2["replay"]["pos"] == 11.0 and s2["version"] > s["version"]
    finally:
        server.shutdown()
        server.server_close()


def test_the_minimap_starts_the_window_and_the_web_page_or_just_one(monkeypatch):
    import gamelab.__main__ as cli
    from gamelab.ui import app, web
    started = []

    class Window:
        def __init__(self, view_for, source, live, position=None):
            started.append("window")

        def run(self):
            pass

    def start(feed, **kw):
        started.append("web")
        return "server"
    monkeypatch.setattr(app, "MapWindow", Window)
    monkeypatch.setattr(web, "start", start)
    monkeypatch.setattr(web, "stop", lambda server: started.append("web stopped"))
    monkeypatch.setattr(web, "Feed", lambda view_for, source, live: None)
    monkeypatch.setattr(cli, "_map_views", lambda args, width: lambda map_id: SimpleView())
    monkeypatch.setattr(cli, "_map_source", lambda args: (None, True))
    monkeypatch.setattr(cli, "_beside_game", lambda w, h: None)
    parser = cli.build_parser()

    def run(*argv):
        started.clear()
        args = parser.parse_args(["minimap", *argv])
        assert args.func(args) in (0, 1)
        return list(started)
    assert run() == ["web", "window", "web stopped"]  # both; closing the window stops the page too
    assert run("--window-only") == ["window"]
    monkeypatch.setattr(cli.time, "sleep", lambda s: (_ for _ in ()).throw(KeyboardInterrupt))  # Ctrl+C at once
    assert run("--web-only") == ["web", "web stopped"]

    def taken(feed, **kw):
        raise OSError("address already in use")
    monkeypatch.setattr(web, "start", taken)
    assert run() == ["window"]  # the port taken: the window still opens


class SimpleView:
    size = (760, 600)


def test_the_web_page_can_be_served_in_the_background_and_stopped():
    import urllib.request

    from gamelab.ui.sources import ReplaySource
    from gamelab.ui.web import Feed, start, stop
    m = model()
    feed = Feed(lambda map_id: MapView(m, width=300), ReplaySource([]), live=False)
    lines = []
    server = start(feed, host="127.0.0.1", port=0, key="k", announce=lines.append)
    try:
        assert urllib.request.urlopen(f"http://127.0.0.1:{server.port}/state.json?k=k").status == 200
        assert lines[1] == f"  http://127.0.0.1:{server.port}/?k=k" and len(lines) == 2  # this computer only
    finally:
        stop(server)
