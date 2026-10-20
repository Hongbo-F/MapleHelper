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


def test_the_newest_run_is_the_one_written_to_last(tmp_path):
    import os
    assert telemetry.newest(tmp_path / "none yet") is None and telemetry.newest(tmp_path) is None
    # When the clocks go back (here 2026-11-01, 02:00 PDT -> 01:00 PST), a run started at 01:10 PST is
    # named before the one started at 01:50 PDT twenty minutes earlier: the one in progress still wins.
    finished, running = tmp_path / "20261101-015000-tour.jsonl", tmp_path / "20261101-011000-tour.jsonl"
    for path in (finished, running):
        path.write_text("", encoding="utf-8")
    os.utime(finished, (1_793_523_300, 1_793_523_300))  # last written at 01:55 PDT
    os.utime(running, (1_793_524_210, 1_793_524_210))  # 01:10 PST, still writing
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    os.utime(tmp_path / "notes.txt", (1_900_000_000, 1_900_000_000))  # not telemetry
    assert telemetry.newest(tmp_path) == running
    bench = tmp_path / "bench.jsonl"  # named any way at all: by when it was written to
    bench.write_text("", encoding="utf-8")
    os.utime(bench, (1_793_524_220, 1_793_524_220))
    assert telemetry.newest(tmp_path) == bench
    os.utime(bench, (1_700_000_000, 1_700_000_000))
    assert telemetry.newest(tmp_path) == running


def test_live_source_looks_for_a_newer_run_once_a_second(tmp_path):
    from gamelab.ui.sources import LiveSource
    first = tmp_path / "20261020-010000-tour.jsonl"
    first.write_text('{"topic": "log", "t": 1.0, "text": "a"}\n', encoding="utf-8")
    now = [100.0]
    src = LiveSource(tmp_path, clock=lambda: now[0])
    fresh, records = src.poll()
    assert fresh and [r["text"] for r in records] == ["a"]
    with first.open("a", encoding="utf-8") as f:  # its last line, then a newer run starts
        f.write('{"topic": "log", "t": 1.5, "text": "a2"}\n')
    (tmp_path / "20261020-020000-tour.jsonl").write_text('{"topic": "log", "t": 2.0, "text": "b"}\n',
                                                          encoding="utf-8")
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


class OldView(MapView):
    """MapView.draw as it was before it was made faster: the tour's platforms drawn every time, and the
    reach bands blended over the whole picture."""

    def draw(self, state):
        img = self._base.copy()
        on_map = img[:self.map_h]
        tour = state.tour or {}
        self._platform_roles(on_map, tour)
        self._tour(on_map, tour)
        self._monsters(on_map, state.monsters)
        self._route(on_map, state.route)
        self._chain(on_map, state.chain)
        self._drops(on_map, state.drops)
        self._piles(on_map, state.hunt)
        scene = state.scene or {}
        her = state.her()
        if her is not None:
            self._old_reach(on_map, her)
        self._monsters_seen(on_map, scene.get("monsters") or [], state.stance)
        self._loot(on_map, scene.get("loot") or [])
        self._trail(on_map, state.trail)
        if her is not None:
            self._her(on_map, her)
        self._panel(img, state)
        return img

    def _old_reach(self, img, her):
        import cv2
        from gamelab import directions
        from gamelab.attack.reach import SHOT_DOWN, SHOT_UP, STAR_REACH, min_dx
        from gamelab.ui.render import REACH_BEHIND, REACH_FACING
        x, y = her["x"], her["y"]
        overlay = img.copy()
        for sign in (-1, 1):
            near, far = x + sign * min_dx(0), x + sign * STAR_REACH
            a, b = self.pt(min(near, far), y - SHOT_UP), self.pt(max(near, far), y + SHOT_DOWN)
            facing = her.get("facing") == directions.toward(sign)
            cv2.rectangle(overlay, a, b, REACH_FACING if facing else REACH_BEHIND, -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)


def busy_state(m, x: float, y: float, facing: str, safe=None) -> State:
    """A state with something of every kind on the map, her at (x, y)."""
    a, b, c, d = by_y(m, 300, 100), by_y(m, 360, 300), by_y(m, 500, 400), by_y(m, 300, 700)
    here = min((a, b, c, d), key=lambda p: abs(p.y_at(x) - y) if p.covers(x) else 1e9)
    s = State()
    s.apply({"topic": "session", "t": 0.0, "map": 1, "mode": "live"})
    s.apply({"topic": "tour", "t": 0.1, "order": [d.id, b.id, c.id], "safe": [a.id] if safe is None else safe,
             "grounds": [str(d.id), b.id]})
    s.apply({"topic": "tour", "t": 0.2, "stop": b.id, "spot": {"platform": c.id, "x": 300}})
    s.apply({"topic": "monsters", "t": 0.3, "list": [{"platform": b.id, "x": 300, "vx": 20, "reach": [260, 340]}]})
    s.apply({"topic": "route", "t": 0.4, "moves": [{"kind": "climb", "src": c.id, "dst": d.id, "x": 700, "land_x": 700},
                                                    {"kind": "jump", "src": d.id, "dst": d.id, "x": 650, "land_x": 760}]})
    s.apply({"topic": "chain", "t": 0.5, "text": "run, jump", "legs": [
        {"kind": "walk", "src": c.id, "dst": c.id, "spans": [[c.id, x, 700]], "end": {"platform": c.id, "x": 700}}]})
    s.apply({"topic": "drops", "t": 0.6, "pending": [{"x": 150, "y": 300, "platform": a.id, "mesos": True},
                                                      {"x": 250, "y": 500, "platform": c.id}]})
    s.apply({"topic": "hunt", "t": 0.7, "rule": "shoot", "piles": [
        {"x": 330, "y": 360, "platform": b.id, "status": "expected", "hers": True},
        {"x": 500, "y": 500, "platform": c.id, "status": "seen", "urgent": True}]})
    s.apply({"topic": "stance", "t": 0.8, "act": "throw", "aimed": [{"kind": "pig", "x": 600, "y": 500}]})
    for i in range(6):
        s.apply({"topic": "glance", "t": 0.9 + i / 10, "x": x - 60 + 10 * i, "y": y})
    s.apply({"topic": "scene", "t": 1.5, "her": {"x": x, "y": y, "platform": here.id, "facing": facing, "seen": False},
             "hp": 0.7, "mp": 0.4, "monsters": [{"kind": "pig", "x": 600, "y": 500, "platform": c.id, "bar": 0.5},
                                                {"kind": "snail", "x": 120, "y": 300, "platform": a.id}],
             "loot": [{"kind": "mesos", "x": 450, "y": 500, "platform": c.id}]})
    s.apply({"topic": "log", "t": 1.6, "text": "pig down"})
    return s


def test_the_faster_drawing_is_pixel_identical():
    m = model()
    states = [busy_state(m, 400, 500, "right"),               # on the lowest platform: the bands reach its foot
              busy_state(m, 100, 300, "left"),                # the bands cut by the picture's left edge
              busy_state(m, 700, 300, "right", safe=[]),      # other tour platforms
              busy_state(m, 760, 300, "right"), State()]
    for width in (300, 760, 900, 2250):
        old, new = OldView(m, width=width), MapView(m, width=width)
        for s in states + states:  # again: the tour's platforms from the kept picture
            assert np.array_equal(old.draw(s), new.draw(s))


def test_the_part_around_her_is_drawn_in_place():
    m = model()
    s = busy_state(m, 400, 500, "right")
    zoomed = MapView(m, width=900)
    left, top, w, h = 300, 150, 360, 200  # her in it, the bands to its foot
    whole = zoomed.draw(s)[top:top + h, left:left + w]
    part = zoomed.draw(s, crop=(left, top, w, h))
    assert part.shape == whole.shape
    same = (part == whole).all(axis=2)
    assert same.mean() > 0.999  # only lines cut by the part's edges are antialiased differently
    x, y = zoomed.pt(400, 500 - 8)
    assert tuple(part[y - top, x - left - 3]) == HER == tuple(whole[y - top, x - left - 3])


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


# --- the page's stream (Server-Sent Events) -----------------------------------------------------------

def run_records(m) -> list[dict]:
    c = by_y(m, 500, 400)
    return [{"topic": "session", "t": 100.0, "map": 1, "command": "tour", "mode": "live"},
            {"topic": "scene", "t": 100.5, "her": {"x": 400, "y": 500, "platform": c.id, "how": "tag"}, "hp": 0.8,
             "mp": 1.0, "monsters": [], "loot": []},
            {"topic": "hunt", "t": 100.6, "rule": "shoot", "ground": c.id},
            {"topic": "log", "t": 100.7, "text": "pig down"},
            {"topic": "log", "t": 102.0, "text": "later"},
            {"topic": "glance", "t": 102.1, "x": 420, "y": 500},
            {"topic": "log", "t": 130.0, "text": "much later"}]


def replay_feed(m, now: list[float]):
    from gamelab.ui.web import Feed
    feed = Feed(lambda map_id: MapView(m, width=300), ReplaySource(run_records(m), clock=lambda: now[0]), live=False)
    now[0] = 1.0  # one second in: the first four records
    feed.step()
    return feed


class Served:
    """A feed's page served on 127.0.0.1 (any free port) while in a `with`."""

    def __init__(self, feed, key: str = "secret"):
        import threading
        from gamelab.ui.web import WebServer
        self.server = WebServer(feed, host="127.0.0.1", port=0, key=key)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def __enter__(self):
        return self.server

    def __exit__(self, *exc):
        from gamelab.ui.web import stop
        stop(self.server)


def open_stream(server, path: str = "/events?k=secret", headers: dict | None = None):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    conn.request("GET", path, headers=headers or {})
    return conn, conn.getresponse()


def read_event(resp) -> tuple[str, dict, str | None]:
    """One Server-Sent Event of a stream: (its name, its data, its id)."""
    import json
    kind, data, ident = "message", [], None
    while True:
        line = resp.readline()
        if not line:
            raise EOFError("the stream ended")
        line = line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                return kind, json.loads("\n".join(data)), ident
            continue
        name, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if name == "event":
            kind = value
        elif name == "data":
            data.append(value)
        elif name == "id":
            ident = value


def test_the_page_streams_a_snapshot_then_the_new_records():
    m = model()
    now = [0.0]
    feed = replay_feed(m, now)
    with Served(feed) as server:
        conn, resp = open_stream(server)
        assert resp.status == 200 and resp.getheader("Content-Type") == "text/event-stream"
        kind, snap, _ = read_event(resp)
        assert kind == "snapshot" and snap["map"] == 1 and snap["now"] == 101.0 and snap["rate"] == 1.0
        assert {r["topic"] for r in snap["records"]} == {"session", "scene", "hunt"}
        assert snap["trail"] == [[400, 500]] and snap["logs"] == ["pig down"]
        assert snap["status"]["hp"] == 0.8 and snap["status"]["her"]["platform"] == by_y(m, 500, 400).id
        now[0] = 2.5
        feed.step()
        kind, update, ident = read_event(resp)
        assert kind == "message" and [r["topic"] for r in update["records"]] == ["log", "glance"]
        assert ident == str(update["seq"]) == str(feed.seq) and update["now"] == 102.5
        while "status" not in update:  # what changed in the text follows within STATUS_EVERY_S
            kind, update, _ = read_event(resp)
        assert update["status"]["her"]["x"] == 420 and "hp" not in update["status"]  # HP didn't change
        conn.close()


def test_the_stream_and_the_map_need_the_key():
    m = model()
    feed = replay_feed(m, [0.0])
    with Served(feed) as server:
        for path in ("/events", "/map.json", "/clock.json"):
            conn, resp = open_stream(server, path)
            assert resp.status == 403
            conn.close()
            conn, resp = open_stream(server, path, {"Cookie": "gk=secret"})
            assert resp.status == 200
            conn.close()


def test_a_seek_or_a_new_run_restarts_the_stream_with_a_snapshot(tmp_path):
    import urllib.request
    from gamelab.ui.sources import LiveSource
    from gamelab.ui.web import Feed
    m = model()
    feed = replay_feed(m, [0.0])
    with Served(feed) as server:
        conn, resp = open_stream(server)
        assert read_event(resp)[0] == "snapshot"
        urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{server.port}/control?k=secret&action=on", method="POST"))
        kind, snap, _ = read_event(resp)
        assert kind == "snapshot" and snap["logs"] == ["pig down", "later"]  # 11 s in
        conn.close()
    first = tmp_path / "20261020-010000-tour.jsonl"
    first.write_text('{"topic": "log", "t": 1.0, "text": "a"}\n', encoding="utf-8")
    now = [0.0]
    feed = Feed(lambda map_id: MapView(m, width=300), LiveSource(tmp_path, clock=lambda: now[0]), live=True)
    feed.step()
    with Served(feed) as server:
        conn, resp = open_stream(server)
        kind, snap, _ = read_event(resp)
        assert kind == "snapshot" and snap["logs"] == ["a"]
        (tmp_path / "20261020-020000-tour.jsonl").write_text('{"topic": "log", "t": 2.0, "text": "b"}\n',
                                                              encoding="utf-8")
        now[0] = 1.5
        feed.step()
        kind, snap, _ = read_event(resp)
        assert kind == "snapshot" and snap["logs"] == ["b"]
        conn.close()


def test_a_client_that_fell_behind_gets_a_snapshot(monkeypatch):
    from gamelab.ui import web
    monkeypatch.setattr(web, "RING", 8)
    m = model()
    now = [0.0]
    records = [{"topic": "log", "t": float(i), "text": str(i)} for i in range(30)]
    feed = web.Feed(lambda map_id: MapView(m, width=300), ReplaySource(records, clock=lambda: now[0]), live=False)
    behind, keeping_up = web.Cursor(), web.Cursor()
    assert feed.updates(behind, 0)[0] == "snapshot" and feed.updates(keeping_up, 0)[0] == "snapshot"
    assert feed.updates(keeping_up, 0)[0] == "ping"  # nothing new
    sent = -1
    for t in range(1, 21):
        now[0] = float(t)
        feed.step()
        if t % 4 == 0:  # 4 or 5 records since its last: all still kept
            kind, update = feed.updates(keeping_up, 0)
            assert kind == "update" and [r["text"] for r in update["records"]] == [str(i) for i in range(sent + 1, t + 1)]
            sent = t
    assert feed.updates(behind, 0)[0] == "snapshot"  # 20 records on: they're no longer all kept


def test_a_backlog_keeps_every_position_but_only_the_latest_plan():
    from gamelab.ui.web import _coalesce
    records = [{"topic": "route", "t": 1.0, "dst": 3, "moves": []},
               {"topic": "glance", "t": 1.1, "x": 10, "y": 500},
               {"topic": "scene", "t": 1.2, "her": {"x": 12, "y": 500, "platform": 2}, "hp": 0.9},
               {"topic": "chain", "t": 1.25, "text": "first"},
               {"topic": "log", "t": 1.3, "text": "off"},
               {"topic": "glance", "t": 1.4, "x": 20, "y": 500},
               {"topic": "route", "t": 1.5, "dst": 4, "moves": []},
               {"topic": "trip", "t": 1.55, "result": "arrived"},
               {"topic": "chain", "t": 1.6, "text": "second"},
               {"topic": "scene", "t": 1.7, "her": {"x": 30, "y": 500, "platform": 4}, "hp": 0.8},
               {"topic": "glance", "t": 1.8, "x": 35, "y": 500}]
    kept = _coalesce(records)
    assert [r["topic"] for r in kept] == ["glance", "scene", "log", "glance", "route", "trip", "chain", "scene",
                                         "glance"]
    whole, merged = State(), State()
    for r in records:
        whole.apply(r)
    for r in kept:
        merged.apply(r)
    assert merged.her() == whole.her() and list(merged.trail) == list(whole.trail) and len(whole.trail) == 5
    assert merged.route == whole.route is None and merged.chain == whole.chain and merged.chain["text"] == "second"


def test_stopping_the_server_ends_its_streams():
    import threading
    import time
    from gamelab.ui.web import HEARTBEAT_S, Feed, start, stop
    m = model()
    feed = Feed(lambda map_id: MapView(m, width=300), ReplaySource(run_records(m)), live=False)
    before = set(threading.enumerate())
    server = start(feed, host="127.0.0.1", port=0, key="secret", announce=lambda line: None)
    conn, resp = open_stream(server)
    assert read_event(resp)[0] == "snapshot"
    began = time.monotonic()
    stop(server)
    try:
        while True:
            read_event(resp)
    except EOFError:
        pass
    assert time.monotonic() - began < HEARTBEAT_S + 1
    conn.close()
    for _ in range(40):  # its handler thread is gone (and the feed's)
        left = [t for t in threading.enumerate() if t not in before and t.is_alive()]
        if not left:
            break
        time.sleep(0.05)
    assert not left, left


def test_the_oldest_stream_past_the_most_is_told_it_was_crowded_out(monkeypatch):
    from gamelab.ui import web
    monkeypatch.setattr(web, "MAX_STREAMS", 1)
    m = model()
    feed = replay_feed(m, [0.0])
    with Served(feed) as server:
        old_conn, old = open_stream(server)
        assert read_event(old)[0] == "snapshot"
        new_conn, new = open_stream(server)
        assert read_event(new)[0] == "snapshot"
        kind, data, _ = read_event(old)  # so the page doesn't come back by itself and crowd out the next
        assert (kind, data) == ("crowded", {"streams": 1})
        try:
            read_event(old)
            assert False, "the stream went on"
        except EOFError:
            pass
        assert len(server.streams) == 1 and server.streams[0].ended is None
        old_conn.close()
        new_conn.close()


def test_a_connection_dropped_between_requests_prints_nothing(capsys):
    import socket
    import struct
    import threading
    import time
    feed = replay_feed(model(), [0.0])
    with Served(feed) as server:
        before = set(threading.enumerate())
        s = socket.create_connection(("127.0.0.1", server.port), timeout=5)
        s.sendall(b"GET /clock.json?k=secret HTTP/1.1\r\nHost: here\r\n\r\n")
        assert s.recv(65536).startswith(b"HTTP/1.1 200")
        # A reset while the server waits for the next request on the connection (a phone gone)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        s.close()
        for _ in range(100):  # its handler thread has seen the reset
            if not [t for t in threading.enumerate() if t not in before and t.is_alive()]:
                break
            time.sleep(0.02)
        try:
            raise ValueError("a real error")
        except ValueError:
            server.handle_error(None, ("127.0.0.1", 1))
    err = capsys.readouterr().err
    assert "ValueError: a real error" in err and "ConnectionResetError" not in err and err.count("Traceback") == 1


def test_the_page_s_map_matches_the_window_s():
    import http.client
    import json
    from gamelab.attack.reach import STAR_REACH
    m = model()
    view = MapView(m, width=300)
    feed = replay_feed(m, [0.0])
    with Served(feed) as server:
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        conn.request("GET", "/map.json?k=secret")
        payload = json.loads(conn.getresponse().read())
        conn.request("GET", "/state.json?k=secret")  # the same connection (HTTP/1.1 keep-alive)
        assert json.loads(conn.getresponse().read())["hp"] == 0.8
        conn.close()
    assert (payload["id"], payload["x0"], payload["x1"], payload["y0"], payload["y1"], payload["margin"]) == \
        (1, view.x0, view.x1, view.y0, view.y1, view.margin)
    assert payload["platforms"] == [[[s.x1, s.y1, s.x2, s.y2] for s in p.segments] for p in m.platforms]
    assert len(payload["walls"]) == len(m.walls) and payload["ropes"] == [[r.x, r.top, r.bottom] for r in m.ropes]
    assert payload["style"]["her"] == "#%02x%02x%02x" % HER and payload["kind"]["pig"] == "#ff96aa"
    assert payload["reach"]["star"] == STAR_REACH and payload["zoom"] == 2.5


def test_the_page_folds_records_as_the_state_does():
    import json
    import re
    import urllib.request
    from gamelab.ui import state
    from gamelab.ui.web import LOG_KEEP
    m = model()
    with Served(replay_feed(m, [0.0])) as server:
        page = urllib.request.urlopen(f"http://127.0.0.1:{server.port}/?k=secret").read().decode("utf-8")
    assert f"const LATEST = {json.dumps(list(state.LATEST))}, TRAIL = {state.TRAIL}, LOG_KEEP = {LOG_KEEP}" in page
    assert 'const KEY = "secret"' in page and not re.search(r"__[A-Z_]+__", page)  # every blank filled in


def test_the_page_s_clock_follows_the_run(tmp_path):
    import time
    from gamelab.ui.sources import LiveSource
    from gamelab.ui.web import Feed
    m = model()
    now = [0.0]
    feed = replay_feed(m, now)
    assert feed.clock() == (101.0, 1.0)
    feed.control("faster")
    assert feed.clock() == (101.0, 2.0)
    feed.control("pause")
    assert feed.clock() == (101.0, 0.0)
    path = tmp_path / "20261020-010000-sim-tour_run.jsonl"
    path.write_text('{"topic": "log", "t": 1800000012.5, "text": "a"}\n', encoding="utf-8")  # the simulator's clock
    live = Feed(lambda map_id: MapView(m, width=300), LiveSource(tmp_path), live=True)
    assert abs(live.clock()[0] - time.time()) < 1  # nothing read yet: this computer's clock
    live.step()
    t, rate = live.clock()
    assert 1800000012.5 <= t < 1800000013.5 and rate == 1.0
    path.write_text(f'{{"topic": "log", "t": {time.time():.3f}, "text": "a"}}\n', encoding="utf-8")  # a run here
    live = Feed(lambda map_id: MapView(m, width=300), LiveSource(tmp_path), live=True)
    live.step()
    assert abs(live.clock()[0] - time.time()) < 0.1
