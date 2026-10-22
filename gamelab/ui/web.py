"""The mini-map as a web page, to watch a run from a phone on the same network: `python -m gamelab web`.

A small HTTP server (standard library only) follows the run's telemetry as the window does (a run live,
or a recorded one replayed) and pushes it to the page as it comes; the page draws the map itself, on a
canvas, layer for layer as render.MapView draws it for the window:

    /             the page: the map, and what she's doing as text (HP and MP, the hunt loop's rule, the
                  fight, the moves, the latest log lines)
    /events       the stream (Server-Sent Events): a snapshot of the state first, then the records as
                  they're read (every 25 ms), what changed in the text (at most 5 times a second), and a
                  ping after 5 s of quiet; past 16 streams, the oldest is told it was crowded out, and ends
    /map.json     the map: its platforms, walls, ropes and portals, with the colours and the reach the
                  window draws them with (from render.py, so the two can't drift apart)
    /control      replays only (POST action=pause|faster|slower|back|on): play, speed, jump 10 s
    /frame.jpg    the map drawn here (?view=her: zoomed in around her), /state.json the text, and
    /clock.json   the run's clock: the page doesn't need them (tests, a quick look, no JavaScript, and
                  the page's ?debug=1 times its round trips with the last)

Why push records and draw on the phone: the page used to fetch the text every 400 ms, then a 40 KB JPEG
of the map. That showed 1.6 pictures a second, 280 ms old on average on this computer (540 at worst) and
43% of her positions, for 45-70 KB/s. Pushed, every record reaches the page in about 20 ms, for about
2 KB/s, and the phone draws her gliding between her positions at its screen's frame rate. EventSource and
the canvas both work in iOS Safari. A stream that has gone quiet (Safari can leave one dead behind a
locked screen) is opened again, and every stream starts with a snapshot, so nothing needs resuming.
(WebRTC would need packages the standard library doesn't have, for nothing a LAN needs.)

It only reads telemetry: it sends no keys and never touches the game. Every request needs the key printed
at the start (?k=..., then a cookie), so nobody else on the network can watch.
"""

import json
import secrets
import socket
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from urllib.parse import parse_qs, urlparse

import cv2

from ..attack import reach
from ..attack.reach import SHOT_DOWN, SHOT_H, SHOT_UP, STAR_REACH, shot_dx
from .render import JUMP_ARC, KIND, PLAYERS, REACH_ALPHA, STYLE, MapView
from .state import LATEST, TRAIL, State

POLL_S = 0.025          # how often the telemetry is read: on average a record waits half of it (at 0.1 s,
                        # records reached the page 58 ms after they were written; at 0.025 s, 21 ms)
LOG_KEEP = 40           # log lines the page shows
SEEK_S = 10.0           # a replay's jump back or on
JPEG_QUALITY = 80
ZOOM = 2.5              # "around her": the map drawn this much bigger, and the part around her cut out ...
AROUND = 0.8            # ... as tall as this share of its width
RING = 4096             # records kept for the streams: a stream further behind gets a snapshot instead
HEARTBEAT_S = 5.0       # a stream with nothing to send pings this often (the page reopens one 12.5 s quiet)
MAX_STREAMS = 16        # past this many, the oldest stream ends (a phone gone from the Wi-Fi leaves one)
STATUS_EVERY_S = 0.2    # what changed in the text goes to a stream at most this often
HANDLER_TIMEOUT_S = 30  # a connection quiet this long (a phone gone mid-request, a stream stuck) is closed
OWN_CLOCK_S = 5.0       # records stamped within this of this computer's clock come from a run on it


class Cursor:
    """How far one stream has got: the last record it was sent (`seq`, of `epoch`), and the text it was
    last sent, of which version and when (monotonic s). Setting `ended` ends the stream: "stopping" (the
    server stops) or "crowded" (MAX_STREAMS newer ones are open)."""

    def __init__(self):
        self.seq, self.epoch = 0, -1
        self.version, self.status_at, self.text = -1, float("-inf"), {}
        self.ended: str | None = None


class Feed:
    """The run's state, kept up to date from `source` (sources.LiveSource or ReplaySource) by a thread,
    and what the page needs of it: the records for its streams, the text, the map, and the map drawn here
    with the view `view_for(map id)` gives (None: the map to start with)."""

    def __init__(self, view_for: Callable[[int | None], MapView], source, live: bool):
        self.view_for, self.source, self.live = view_for, source, live
        self.state = State()
        self.view = view_for(None)
        self.logs: deque[str] = deque(maxlen=LOG_KEEP)
        self.version = 0                 # goes up with every change (the text is made again then)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)   # the streams wait on it
        self.ring: deque[tuple[int, dict]] = deque(maxlen=RING)  # (seq, record): the records for the streams
        self.seq = 0                     # the number of the last record read
        self.epoch = 0                   # goes up when the state starts over (a newer run, a replay's jump)
        self._text: tuple[int, dict] = (-1, {})          # (version, the text without the log lines)
        self._maps: dict[int, bytes] = {}                 # map id -> map.json
        self._jpeg: dict[str, tuple[int, bytes]] = {}     # view -> (version, picture)
        self._zoomed: dict[int, MapView] = {}             # map id -> the view "around her" is cut from
        self._read: tuple[float, float] | None = None     # live: (the newest record's t, monotonic s when read)
        self._own_clock = True                            # ... stamped with this computer's clock
        self._pace, self._pace_from = 1.0, None           # ... the run's seconds a second, measured since
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="telemetry", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.step()
            self._stop.wait(POLL_S)

    def step(self) -> None:
        """Read what the source has, and apply it."""
        fresh, records = self.source.poll()
        if not (fresh or records):
            return
        with self.lock:
            if fresh:  # a newer run started: the page starts over too
                self._restart(records)
            else:
                self._apply(records)

    def _apply(self, records: list[dict]) -> None:
        """Fold new records into the state, and hand them to the streams (the lock held)."""
        self._fold(records)
        for r in records:
            self.seq += 1
            self.ring.append((self.seq, r))
        self.version += 1
        self.cond.notify_all()

    def _restart(self, records: list[dict]) -> None:
        """Start the state over from `records` (the lock held): the streams send a snapshot."""
        self.state, self.logs = State(), deque(maxlen=LOG_KEEP)
        self.ring.clear()
        self.epoch += 1
        self._read, self._own_clock, self._pace, self._pace_from = None, True, 1.0, None
        self._fold(records)
        self.version += 1
        self.cond.notify_all()

    def _fold(self, records: list[dict]) -> None:
        for r in records:
            self.state.apply(r)
            if r.get("topic") in ("log", "event"):
                self.logs.append(r.get("text") or r.get("kind", ""))
        if self.state.map is not None and self.state.map != getattr(self.view.model, "id", None):
            self.view = self.view_for(self.state.map)
        if self.live:
            self._timed(records)

    def _timed(self, records: list[dict]) -> None:
        """Note when the newest record was read, for clock()."""
        t = next((r["t"] for r in reversed(records) if isinstance(r.get("t"), (int, float))), None)
        if t is None:
            return
        now = time.monotonic()
        self._own_clock = abs(t - time.time()) < OWN_CLOCK_S
        if self._pace_from is None:
            self._pace_from = (t, now)
        elif now - self._pace_from[1] >= 2.0:
            t0, at = self._pace_from
            self._pace = min(max((t - t0) / (now - at), 0.0), 64.0)
            self._pace_from = (t, now)
        self._read = (t, now)

    def clock(self) -> tuple[float, float]:
        """(the run's time now, how fast it goes in its seconds a second): the page draws her by it,
        between her positions. A replay: where it is, at its speed (0 while paused). Live, a run on this
        computer: this computer's clock. A simulated run's records carry the simulator's own clock (a
        fixed date), so there it's the newest record's time, moved on at the pace the records came at."""
        if not self.live:
            src = self.source
            return src.t0 + src.pos, (0.0 if src.paused else float(src.speed))
        if self._read is None or self._own_clock:
            return time.time(), 1.0
        t, at = self._read
        return t + (time.monotonic() - at) * self._pace, self._pace

    # --- the page's stream -------------------------------------------------------------------------

    def updates(self, cursor: Cursor, timeout: float) -> tuple[str, dict] | None:
        """What a stream sends next, waiting up to `timeout` s for something to send:
        ("snapshot", ...) first, when the state started over (a newer run, a replay's jump), and when the
        stream fell behind the ring; ("update", {seq, now, rate, records, status?}): the records since
        its last, those a later one of the same topic replaces left out, and what changed in the text
        (at most every STATUS_EVERY_S); else ("ping", {now, rate}). None once the cursor has ended."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while not cursor.ended:
                if cursor.epoch != self.epoch or (self.ring and self.ring[0][0] > cursor.seq + 1):
                    return "snapshot", self._snapshot(cursor)
                now = time.monotonic()
                new = self.seq - cursor.seq
                text_due = cursor.version != self.version and now >= cursor.status_at + STATUS_EVERY_S
                if new or text_due:
                    clock, rate = self.clock()
                    out = {"seq": self.seq, "now": clock, "rate": rate,
                           "records": _coalesce([r for _, r in islice(reversed(self.ring), new)][::-1])}
                    cursor.seq = self.seq
                    if text_due:
                        text = self._status()
                        changed = {k: v for k, v in text.items() if cursor.text.get(k, ...) != v}
                        if changed:
                            out["status"] = changed
                        cursor.version, cursor.status_at, cursor.text = self.version, now, text
                    return "update", out
                if now >= deadline:
                    clock, rate = self.clock()
                    return "ping", {"now": clock, "rate": rate}
                wait = deadline - now
                if cursor.version != self.version:  # the text, once it may go
                    wait = min(wait, cursor.status_at + STATUS_EVERY_S - now)
                self.cond.wait(wait)
        return None

    def _snapshot(self, cursor: Cursor) -> dict:
        """Everything a page starts from (the lock held): the records the state keeps (the session, the
        tour so far, the latest of each other topic), her trail, the log lines and the text."""
        s = self.state
        records = [r for r in (s.session, s.tour, *(getattr(s, topic) for topic in LATEST), s.scene) if r]
        clock, rate = self.clock()
        text = self._status()
        cursor.seq, cursor.epoch = self.seq, self.epoch
        cursor.version, cursor.status_at, cursor.text = self.version, time.monotonic(), text
        return {"epoch": self.epoch, "seq": self.seq, "now": clock, "rate": rate, "map": s.map, "t": s.t,
                "records": records, "trail": list(s.trail), "logs": list(self.logs), "status": text}

    # --- the text, the map, the picture ------------------------------------------------------------

    def status(self) -> dict:
        """What the page shows as text (/state.json)."""
        with self.lock:
            return {**self._status(), "logs": list(self.logs)[::-1]}  # the newest first

    def _status(self) -> dict:
        """The text without the log lines (the lock held; made once per version), with a replay's place."""
        if self._text[0] != self.version:
            s = self.state
            scene = s.scene or {}
            her = s.her()
            hunt = s.hunt or {}
            self._text = (self.version, {
                "version": self.version, "live": self.live, "t": s.t or None, "map": s.map,
                "run": s.session.get("scenario") or s.session.get("command"), "mode": s.session.get("mode"),
                "her": None if her is None else {k: her.get(k) for k in ("x", "y", "platform", "how")},
                "hp": scene.get("hp"), "mp": scene.get("mp"),
                "monsters": len(scene.get("monsters") or []), "loot": len(scene.get("loot") or []),
                "players": len(scene.get("players") or []),
                "fight": (s.stance or {}).get("text") or (s.stance or {}).get("act"),
                "moving": (s.chain or {}).get("text"),
                "hunt": None if not hunt else {k: hunt.get(k) for k in
                                               ("rule", "objective", "ground", "rate_here", "rate_to_beat")},
                "exp": None if not s.exp else {k: s.exp.get(k) for k in ("exp", "level", "per_minute")},
                "stop": (s.tour or {}).get("stop"),
            })
        text = self._text[1]
        if self.live:
            return text
        src = self.source
        end = src.records[-1].get("t", 0.0) - src.t0 if src.records else 0.0
        return {**text, "replay": {"paused": src.paused, "speed": src.speed, "pos": round(src.pos, 1),
                                   "length": round(end, 1), "start": src.t0}}

    def map_json(self) -> bytes:
        """/map.json: the map the page draws now (made once per map)."""
        with self.lock:
            view = self.view
        key = view.model.id
        if key not in self._maps:
            self._maps[key] = _json(map_payload(view))
        return self._maps[key]

    def frame(self, view: str = "all") -> bytes:
        """The map (without the window's text panel: the page has it as text), as a JPEG: all of it, or
        (`view` "her") the part around her, ZOOM times bigger."""
        with self.lock:
            cached = self._jpeg.get(view)
            if cached is not None and cached[0] == self.version:
                return cached[1]
            version = self.version
            rgb = self._around_her() if view == "her" else None
            if rgb is None:
                rgb = self.view.draw(self.state)[:self.view.map_h]
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        self._jpeg[view] = (version, buf.tobytes())
        return self._jpeg[view][1]

    def _around_her(self):
        """The part of the zoomed map around her (None while she hasn't been seen), drawn only there."""
        her = self.state.her()
        if her is None:
            return None
        model = self.view.model
        zoomed = self._zoomed.get(model.id)
        if zoomed is None:
            zoomed = self._zoomed[model.id] = MapView(model, width=int(self.view.size[0] * ZOOM))
        w = self.view.size[0]
        h = min(int(w * AROUND), zoomed.map_h)
        x, y = zoomed.pt(her["x"], her["y"])
        left = min(max(x - w // 2, 0), zoomed.size[0] - w)
        top = min(max(y - h // 2, 0), zoomed.map_h - h)
        return zoomed.draw(self.state, crop=(left, top, w, h))

    def control(self, action: str) -> bool:
        """A replay's controls. False if there's no such action (or this isn't a replay)."""
        if self.live:
            return False
        src = self.source
        with self.lock:
            if action == "pause":
                src.paused = not src.paused
            elif action == "faster":
                src.speed = min(src.speed * 2, 64.0)
            elif action == "slower":
                src.speed = max(src.speed / 2, 0.125)
            elif action in ("back", "on"):
                self._restart(src.seek(SEEK_S if action == "on" else -SEEK_S))
            else:
                return False
            self.version += 1
            self.cond.notify_all()  # the clock's rate changed: the streams say so at once
        return True


def _coalesce(records: list[dict]) -> list[dict]:
    """Records for a stream, in order, without those a later one replaces: of the topics the state keeps
    only the latest of (state.LATEST), the last one. Every position (her trail), log line and the rest
    stays, so the page's state comes out the same."""
    last = {r.get("topic"): i for i, r in enumerate(records) if r.get("topic") in LATEST}
    keep = set(last.values())
    return [r for i, r in enumerate(records) if r.get("topic") not in LATEST or i in keep]


def map_payload(view: MapView) -> dict:
    """What the page draws the map with: the map (platform ids are their index), the view's bounds and
    margin (MapView: map units, px), and the window's colours and reach."""
    m = view.model
    return {
        "id": m.id, "name": m.name,
        "x0": view.x0, "x1": view.x1, "y0": view.y0, "y1": view.y1, "margin": view.margin,
        "platforms": [[[s.x1, s.y1, s.x2, s.y2] for s in p.segments] for p in m.platforms],
        "walls": [[w.x1, w.y1, w.x2, w.y2] for w in m.walls],
        "ropes": [[r.x, r.top, r.bottom] for r in m.ropes],
        "portals": [[p.x, p.y] for p in m.portals],
        "style": {name: _hex(c) for name, c in STYLE.items()},
        "kind": {name: _hex(c) for name, c in KIND.items()},
        "players": {name: _hex(c) for name, c in PLAYERS.items()},
        "reach": {"star": STAR_REACH, "up": SHOT_UP, "down": SHOT_DOWN, "min": shot_dx(0), "alpha": REACH_ALPHA,
                  "shot_h": SHOT_H},
        # per platform, the walls of its foothold group that stop a star (attack.reach.clear_span): x, top,
        # bottom, and the way its outer face looks (+1 right, -1 left: a star stops going in there only)
        "stops": [[[w.x1, min(w.y1, w.y2), max(w.y1, w.y2), reach.faces(w)] for w in p.walls]
                  if reach.WALLS_BLOCK else [] for p in m.platforms],
        "jump_arc": JUMP_ARC, "zoom": ZOOM, "around": AROUND,
    }


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _json(value) -> bytes:
    """Compact JSON (on one line, as an event's data must be); numpy numbers as plain ones."""
    return json.dumps(value, separators=(",", ":"), default=float).encode("utf-8")


def _event(kind: str, data: dict) -> bytes:
    """One Server-Sent Event: an update is the page's `message` (with its last record's number as id),
    the others are named."""
    head = b"id: %d\n" % data["seq"] if kind == "update" else b"event: %s\n" % kind.encode()
    return head + b"data: " + _json(data) + b"\n\n"


class WebServer(ThreadingHTTPServer):
    """The HTTP server: `feed`'s page, stream, map and text, for requests with `key`."""

    daemon_threads = True

    def __init__(self, feed: Feed, host: str = "0.0.0.0", port: int = 8765, key: str | None = None):
        self.feed = feed
        self.key = key or secrets.token_urlsafe(8)
        self.stopping = threading.Event()
        self.streams: list[Cursor] = []  # the open streams, oldest first (changed under feed.lock)
        super().__init__((host, port), _Handler)

    @property
    def port(self) -> int:
        return self.server_address[1]

    def urls(self) -> list[str]:
        """Where to open the page: on this computer, and (when serving the network) from a phone."""
        local = f"http://127.0.0.1:{self.port}/?k={self.key}"
        host = self.server_address[0]
        if host in ("127.0.0.1", "localhost"):
            return [local]
        return [local] + [f"http://{ip}:{self.port}/?k={self.key}" for ip in lan_addresses()]

    def open_stream(self) -> Cursor:
        """A new stream's cursor. Past MAX_STREAMS the oldest ends: a phone gone from the Wi-Fi leaves
        its stream open until a write to it fails, and a new page always gets in. The page so ended is
        told (`crowded`) and doesn't come back by itself, else it would end the next oldest a second
        later, and so on round every page open."""
        cursor = Cursor()
        with self.feed.cond:
            cursor.ended = "stopping" if self.stopping.is_set() else None
            self.streams.append(cursor)
            while len(self.streams) > MAX_STREAMS:
                self.streams.pop(0).ended = "crowded"
            self.feed.cond.notify_all()
        return cursor

    def close_stream(self, cursor: Cursor) -> None:
        with self.feed.cond:
            if cursor in self.streams:
                self.streams.remove(cursor)

    def end_streams(self) -> None:
        """End every stream, now and from now on (their threads leave at once: stopping the server
        doesn't stop them)."""
        with self.feed.cond:
            self.stopping.set()
            for cursor in self.streams:
                cursor.ended = "stopping"
            self.feed.cond.notify_all()

    def handle_error(self, request, client_address) -> None:
        """A connection dropped by the other end is no error: a phone leaves the Wi-Fi or iOS suspends
        the page, often between two requests on a kept-alive connection, where the next request is being
        read (with HTTP/1.0 nothing was read after a reply). Anything else is printed, as socketserver
        does."""
        if not isinstance(sys.exception(), ConnectionError):
            super().handle_error(request, client_address)


def lan_addresses() -> list[str]:
    """This computer's address on its network (where a phone on the same Wi-Fi reaches it)."""
    out = []
    try:  # the address of the interface that routes out (nothing is sent: UDP connect only picks a route)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            out.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in out and not ip.startswith("127."):
                out.append(ip)
    except OSError:
        pass
    return out


class _Handler(BaseHTTPRequestHandler):
    server: WebServer
    protocol_version = "HTTP/1.1"   # keep-alive: the page's few requests share a connection (each used to
                                    # open its own); every reply says its length, the stream closes at its end
    timeout = HANDLER_TIMEOUT_S     # a connection gone quiet frees its thread
    disable_nagle_algorithm = True  # small events go out at once, not held back for the phone's ACK

    def log_message(self, format, *args) -> None:  # quiet: the console shows the run's URLs only
        pass

    def _allowed(self, query: dict) -> bool:
        given = query.get("k", [""])[0]
        if not given:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            given = cookie["gk"].value if "gk" in cookie else ""
        return secrets.compare_digest(given, self.server.key)

    def _send(self, code: int, body: bytes, kind: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if not self._allowed(query):
            self._send(403, b"This page needs its key: open the address `python -m gamelab web` printed.",
                       "text/plain; charset=utf-8")
            return
        feed = self.server.feed
        if url.path == "/":
            cookie = f"gk={self.server.key}; Path=/; SameSite=Strict; Max-Age=2592000"
            self._send(200, page(self.server.key, self.server.urls()), "text/html; charset=utf-8",
                       {"Set-Cookie": cookie})
        elif url.path == "/events":
            self._stream()
        elif url.path == "/map.json":
            self._send(200, feed.map_json(), "application/json")
        elif url.path == "/frame.jpg":
            self._send(200, feed.frame(query.get("view", ["all"])[0]), "image/jpeg")
        elif url.path == "/state.json":
            self._send(200, _json(feed.status()), "application/json")
        elif url.path == "/clock.json":
            with feed.lock:
                now, rate = feed.clock()
            self._send(200, _json({"now": now, "rate": rate}), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        try:  # a body left unread would be taken for the next request on the connection
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if 0 < length <= 65536:
            self.rfile.read(length)
        elif length:
            self.close_connection = True
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if not self._allowed(query):
            self._send(403, b"forbidden", "text/plain")
            return
        if url.path != "/control":
            self._send(404, b"not found", "text/plain")
            return
        ok = self.server.feed.control(query.get("action", [""])[0])
        self._send(200 if ok else 400, b"ok" if ok else b"no such control", "text/plain")

    def _stream(self) -> None:
        """/events: a snapshot, then what comes, each event in one write (with Nagle off, it goes out at
        once), until the page goes, the server stops or too many streams are open. Its own thread waits
        for each, so a slow phone holds up nothing else."""
        server = self.server
        cursor = server.open_stream()
        self.close_connection = True
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            first = b"retry: 1000\n"  # how soon the browser tries again by itself when the stream drops
            while True:
                got = server.feed.updates(cursor, HEARTBEAT_S)
                if got is None:
                    if cursor.ended == "crowded":  # the page stops rather than come back (open_stream)
                        self.wfile.write(first + _event("crowded", {"streams": MAX_STREAMS}))
                    return
                self.wfile.write(first + _event(*got))
                first = b""
        except OSError:  # the page went
            pass
        finally:
            server.close_stream(cursor)


def start(feed: Feed, host: str = "0.0.0.0", port: int = 8765, key: str | None = None,
          announce: Callable[[str], None] = print, ends: str = "Ctrl+C stops it") -> WebServer:
    """Serve the page in the background (threads of this process); returns the server, for stop().
    Raises OSError if the port can't be had (another one already serving it)."""
    server = WebServer(feed, host, port, key)
    feed.start()
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.25}, name="web", daemon=True).start()
    announce(f"the mini-map, in a browser ({ends}):")
    for u in server.urls():
        announce(f"  {u}")
    if host not in ("127.0.0.1", "localhost"):
        announce("on a phone: the same Wi-Fi, and the second address (Windows may ask to let Python through "
                 "its firewall: allow it on private networks)")
    return server


def stop(server: WebServer) -> None:
    server.end_streams()
    server.feed.stop()
    server.shutdown()
    server.server_close()


def serve(feed: Feed, host: str = "0.0.0.0", port: int = 8765, key: str | None = None,
          announce: Callable[[str], None] = print) -> None:
    """Serve the page until Ctrl+C."""
    server = start(feed, host, port, key, announce)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop(server)


def page(key: str, urls: list[str]) -> bytes:
    """The page, with what it needs from here written in: the key and the addresses, and the state's
    topics and lengths, so its copy of State.apply keeps what this one keeps."""
    out = PAGE
    for name, value in (("__KEY__", key), ("__URLS__", urls), ("__LATEST__", list(LATEST)), ("__TRAIL__", TRAIL),
                        ("__LOG_KEEP__", LOG_KEEP), ("__HEARTBEAT_S__", HEARTBEAT_S)):
        out = out.replace(name, json.dumps(value).replace("<", "\\u003c"))
    return out.encode("utf-8")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#10141e">
<title>gamelab mini-map</title>
<style>
  :root { --bg: #10141e; --card: #191f2c; --line: #2a3242; --text: #d2d7e1; --dim: #8a90a0;
          --hp: #e64646; --mp: #5082ff; --ok: #5ad25a; --warn: #ffb03a; --accent: #50dcff; }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: var(--bg); color: var(--text);
               font: 15px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { position: sticky; top: 0; z-index: 2; display: flex; align-items: center; gap: 8px;
           padding: 10px 16px; background: rgba(16, 20, 30, .94); border-bottom: 1px solid var(--line); }
  header .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--dim); flex: none; }
  header .dot.live { background: var(--ok); }
  header .dot.replay { background: var(--accent); }
  header .dot.lost { background: var(--hp); }
  header .title { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  header .time { margin-left: auto; color: var(--dim); font-variant-numeric: tabular-nums; }
  main { max-width: 980px; margin: 0 auto; padding: 12px 16px 32px; }
  .views { display: flex; gap: 0; margin-bottom: 8px; border: 1px solid var(--line); border-radius: 8px;
           overflow: hidden; width: max-content; }
  .views button { min-height: 36px; padding: 0 14px; border: 0; background: var(--card); color: var(--dim);
                  font-size: 14px; }
  .views button.on { background: var(--line); color: var(--text); }
  .map { position: relative; border-radius: 10px; overflow: hidden; background: #000; min-height: 120px; }
  .map canvas { display: block; width: 100%; height: 0; }
  .map img { display: block; width: 100%; height: auto; }
  .map .empty { position: absolute; inset: 0; display: grid; place-items: center; padding: 0 16px;
                text-align: center; color: var(--dim); }
  .map .empty.over { background: rgba(16, 20, 30, .85); color: var(--text); }
  .map .debug { position: absolute; left: 6px; top: 6px; max-width: calc(100% - 12px); margin: 0; padding: 4px 6px;
                border-radius: 4px; background: rgba(0, 0, 0, .7); color: var(--text); pointer-events: none;
                font: 11px/1.35 ui-monospace, Consolas, monospace; white-space: pre-wrap; }
  .bars { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0; }
  .bar { background: var(--card); border-radius: 8px; padding: 8px 10px; }
  .bar .label { display: flex; justify-content: space-between; font-size: 13px; color: var(--dim); }
  .bar .track { height: 8px; margin-top: 6px; border-radius: 4px; background: var(--line); overflow: hidden; }
  .bar .fill { height: 100%; width: 0; transition: width .3s; }
  .cards { display: grid; gap: 8px; }
  .card { background: var(--card); border-radius: 8px; padding: 9px 12px; }
  .card .k { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--dim); }
  .card .v { margin-top: 2px; overflow-wrap: anywhere; }
  .card .v.big { font-size: 17px; font-weight: 600; }
  .controls { display: none; gap: 8px; margin: 12px 0; flex-wrap: wrap; }
  .controls button { flex: 1 1 auto; min-width: 56px; min-height: 44px; border: 1px solid var(--line);
                     border-radius: 8px; background: var(--card); color: var(--text); font-size: 16px; }
  .controls button:active { background: var(--line); }
  .controls .pos { flex-basis: 100%; color: var(--dim); font-size: 13px; text-align: center; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--dim); margin: 18px 0 6px; }
  ol.log { list-style: none; margin: 0; padding: 0; font: 13px/1.45 ui-monospace, Consolas, monospace; }
  ol.log li { padding: 4px 0; border-bottom: 1px solid var(--line); overflow-wrap: anywhere; color: var(--dim); }
  ol.log li:first-child { color: var(--text); }
  .phone { display: none; margin-top: 18px; padding: 12px; background: var(--card); border-radius: 8px; }
  .phone #qr { background: #fff; padding: 8px; width: max-content; margin-top: 8px; border-radius: 6px; }
  .phone code { overflow-wrap: anywhere; color: var(--accent); }
  @media (min-width: 900px) {
    .layout { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 16px; align-items: start; }
    .bars { margin-top: 0; }
  }
</style>
<noscript><style>.map .empty { display: none; }</style></noscript>
</head>
<body>
<header>
  <span class="dot" id="dot"></span>
  <span class="title" id="title">gamelab mini-map</span>
  <span class="time" id="time">--:--:--</span>
</header>
<main>
  <div class="layout">
    <div>
      <div class="views" role="group" aria-label="map view">
        <button data-v="her" id="v-her">Around her</button><button data-v="all" id="v-all">Whole map</button>
      </div>
      <div class="map"><canvas id="map" role="img" aria-label="the map"></canvas>
        <div class="empty" id="empty">waiting for a run...</div><pre class="debug" id="debug" hidden></pre>
        <noscript><img src="frame.jpg" alt="the map when the page was opened (JavaScript is off: reload for a newer one)"></noscript>
      </div>
      <div class="controls" id="controls">
        <button data-a="back" aria-label="back 10 s">&laquo; 10 s</button>
        <button data-a="pause" id="play">Pause</button>
        <button data-a="on" aria-label="on 10 s">10 s &raquo;</button>
        <button data-a="slower" aria-label="slower">&times;&frac12;</button>
        <button data-a="faster" aria-label="faster">&times;2</button>
        <div class="pos" id="pos"></div>
      </div>
    </div>
    <div>
      <div class="bars">
        <div class="bar"><div class="label"><span>HP</span><span id="hpv">-</span></div>
          <div class="track"><div class="fill" id="hp" style="background: var(--hp)"></div></div></div>
        <div class="bar"><div class="label"><span>MP</span><span id="mpv">-</span></div>
          <div class="track"><div class="fill" id="mp" style="background: var(--mp)"></div></div></div>
      </div>
      <div class="cards">
        <div class="card"><div class="k">Hunt</div><div class="v big" id="hunt">-</div><div class="v" id="rates"></div></div>
        <div class="card"><div class="k">Fight</div><div class="v" id="fight">-</div></div>
        <div class="card"><div class="k">Moving</div><div class="v" id="moving">-</div></div>
        <div class="card"><div class="k">Where</div><div class="v" id="where">-</div></div>
      </div>
      <div class="phone" id="phone"><div class="k">On your phone (same Wi-Fi)</div>
        <div id="qr"></div><div><code id="lan"></code></div></div>
    </div>
  </div>
  <h2>Log</h2>
  <ol class="log" id="log"></ol>
</main>
<script>
"use strict";
const KEY = __KEY__, URLS = __URLS__;
// From gamelab/ui/state.py and web.py, so this copy of State.apply keeps what the server's keeps.
const LATEST = __LATEST__, TRAIL = __TRAIL__, LOG_KEEP = __LOG_KEEP__, HEARTBEAT_S = __HEARTBEAT_S__;
const params = new URLSearchParams(location.search);
const DEBUG = params.get("debug") === "1";
// Her position is drawn this long (s of the run) behind its clock, so there's a next position to glide
// to: they come every 67 ms (84 at p75) while she moves. ?delay=0 draws the newest one.
const DELAY = params.has("delay") ? Math.max(0, Number(params.get("delay")) || 0) : 0.1;
const GAP = 0.6;  // her positions further apart than this (s) aren't glided between: she's held, then moved
const $ = id => document.getElementById(id);
const withKey = path => path + (path.includes("?") ? "&" : "?") + "k=" + encodeURIComponent(KEY);

// --- the state: the records folded as the server folds them -----------------------------------------

function blank() {
  return {t: 0, map: null, session: {}, scene: null, route: null, chain: null, monsters: null, stance: null,
          drops: null, tour: null, hunt: null, trail: [], logs: []};
}
let S = blank(), status = {};
let rev = 0;                                  // goes up with every change to the map's state: drawn again
let textChanged = true, logsChanged = true;

function keep(list, item, max) { list.push(item); if (list.length > max) list.splice(0, list.length - max); }
function moved(t, x, y) {   // her trail: [x, y, t], in time order
  const trail = S.trail;
  let i = trail.length;
  while (i > 0 && trail[i - 1][2] > t) i--;
  trail.splice(i, 0, [x, y, t]);
  if (trail.length > TRAIL) trail.splice(0, trail.length - TRAIL);
}
// State.apply (gamelab/ui/state.py), and the Feed's log lines (web.py)
function apply(r) {
  const topic = r.topic;
  if ("t" in r) S.t = r.t;
  if (LATEST.includes(topic)) S[topic] = r;
  else if (topic === "session") { S.session = r; if ("map" in r) S.map = r.map; }
  else if (topic === "scene") { S.scene = r; const her = r.her || {}; if (her.x != null) moved(r.t, her.x, her.y); }
  else if (topic === "glance") { if (r.x != null) moved(r.t, r.x, r.y); }
  else if (topic === "tour") S.tour = Object.assign({}, S.tour || {}, r);
  else if (topic === "trip") S.route = S.chain = null;
  if (topic === "log" || topic === "event") { keep(S.logs, r.text || r.kind || "", LOG_KEEP); logsChanged = true; }
}
function herState() {   // State.her: the last look's, moved on by any glance since
  const her = (S.scene || {}).her;
  if (!her || her.x == null) return null;
  if (!S.trail.length) return her;
  const last = S.trail[S.trail.length - 1];
  return Object.assign({}, her, {x: last[0], y: last[1]});
}
function herAt(t) {     // where she was at the run's time t: glided between the positions either side
  const trail = S.trail;
  let i = trail.length - 1;
  if (i < 0) return null;
  if (t == null || !(t < trail[i][2])) return trail[i];
  while (i > 0 && trail[i - 1][2] > t) i--;
  if (i === 0) return trail[0];
  const a = trail[i - 1], b = trail[i];
  if (b[2] - a[2] > GAP) return a;
  const f = (t - a[2]) / (b[2] - a[2]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, t];
}

// The run's clock: its time at the last message, moved on at its rate (1 live, a replay's speed, 0 while
// paused). The message that took the least time to come wins; the others only pull it back slowly.
let clock = null;
function setClock(m) {
  if (m.now == null) return;
  const rate = m.rate == null ? 1 : m.rate, base = m.now - rate * performance.now() / 1000;
  if (!clock || clock.rate !== rate || Math.abs(base - clock.base) > 1) clock = {rate, base};
  else clock.base = Math.max(base, clock.base - 0.005);
}
function runNow() { return clock ? clock.base + clock.rate * performance.now() / 1000 : null; }

function snapshot(s) {
  S = blank();
  for (const r of s.records) apply(r);
  S.trail = s.trail.map(p => [p[0], p[1], -Infinity]);
  S.logs = s.logs.slice();
  S.t = s.t;
  status = s.status || {};
  clock = null;
  setClock(s);
  camera = null;
  rev++; textChanged = logsChanged = true;
  wantMap();
}
function update(u) {
  setClock(u);
  if (DEBUG) late(u.records);
  for (const r of u.records) apply(r);
  if (u.records.length) rev++;
  if (u.status) { Object.assign(status, u.status); textChanged = true; }
  wantMap();
}

// --- the stream, kept open ---------------------------------------------------------------------------

const net = {msgs: 0, bytes: 0, opens: 0, restarts: 0, rtt: null};
// How long the stream has been quiet goes by Date.now(): on an iPhone performance.now() can stop while
// the phone sleeps, so after the screen was locked it would tell of less quiet than there was.
let es = null, lastMsg = Date.now();
let crowded = 0;   // the server ended this page's stream for this many newer ones: it waits to be shown again
function heard(e) { lastMsg = Date.now(); net.msgs++; net.bytes += e.data.length + 20; }
function quiet() { return Date.now() - lastMsg; }
function lost() { return !es || es.readyState !== 1 || quiet() > 2 * HEARTBEAT_S * 1000; }
function connect(byPage) {
  if (es) es.close();
  if (byPage) net.restarts++;
  crowded = 0;
  lastMsg = Date.now();
  es = new EventSource(withKey("events"));
  es.onopen = () => { net.opens++; textChanged = true; };
  es.onerror = () => { textChanged = true; };   // the browser tries again by itself (retry: 1000)
  es.addEventListener("snapshot", e => { heard(e); snapshot(JSON.parse(e.data)); });
  es.addEventListener("ping", e => { heard(e); setClock(JSON.parse(e.data)); });
  es.addEventListener("crowded", e => { es.close(); crowded = JSON.parse(e.data).streams || 1; textChanged = true; });
  es.onmessage = e => { heard(e); update(JSON.parse(e.data)); };
}
// iOS Safari can leave a stream dead behind a locked screen or a page in the background, still "open"
// and with no error: one quiet for 2.5 pings' time is opened again, and so is one quiet for over a
// ping's time when the page is shown again. A page crowded out comes back only when it's shown again
// (or reloaded): coming back by itself, it would crowd out the next oldest page.
setInterval(() => { if (!crowded && quiet() > 2.5 * HEARTBEAT_S * 1000) connect(true); wantMap(); }, 1000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && (crowded || quiet() > (HEARTBEAT_S + 1) * 1000)) connect(true);
});
window.addEventListener("pageshow", e => { if (e.persisted) connect(true); });

let M = null, C = null, mapAsked = -1e9, mapLoading = false;
function wantMap() {   // the map once, and again when the run moves to another
  if (mapLoading || (M && (S.map == null || S.map === M.id)) || performance.now() - mapAsked < 2000) return;
  mapLoading = true;
  mapAsked = performance.now();
  fetch(withKey("map.json"), {cache: "no-store"})
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(m => { M = m; C = m.style; rev++; })
    .catch(() => {})
    .finally(() => { mapLoading = false; });
}

// --- drawing: layer for layer as gamelab/ui/render.py's MapView.draw ---------------------------------

const cv = $("map"), ctx = cv.getContext("2d");
let view = "all", camera = null, lastKey = "", lastFrame = 0;
const perf = {frames: 0, drawn: 0, ms: []};
try { view = localStorage.getItem("gamelab-view") || (window.innerWidth < 700 ? "her" : "all"); }
catch (e) { view = window.innerWidth < 700 ? "her" : "all"; }

function yAt(pid, x) {   // Platform.y_at
  const segs = M.platforms[pid];
  for (const [x1, y1, x2, y2] of segs) if (x1 <= x && x <= x2) return x2 === x1 ? y1 : y1 + (x - x1) / (x2 - x1) * (y2 - y1);
  return x < segs[0][0] ? segs[0][1] : segs[segs.length - 1][3];
}
function known(pid) { return Number.isInteger(pid) && pid >= 0 && pid < M.platforms.length; }

function frame(now) {
  requestAnimationFrame(frame);
  perf.frames++;
  const dt = lastFrame ? Math.min((now - lastFrame) / 1000, 0.25) : 0;
  lastFrame = now;
  const W = cv.clientWidth;
  if (!M || !C || !W) return;
  const dpr = window.devicePixelRatio || 1;
  const u = Math.min(1, Math.max(0.6, W / 760));   // one px of the window's 760-px picture, in css px
  const margin = M.margin * u, fit = (W - 2 * margin) / (M.x1 - M.x0);
  const her = herState(), t = runNow();
  const when = t == null ? null : t - DELAY * Math.max(clock.rate, 1);
  const at = her ? (herAt(when) || [her.x, her.y]) : null;
  let k = fit, H = Math.round((M.y1 - M.y0) * k + 2 * margin), left = 0, top = 0;
  if (view === "her" && at) {   // ZOOM times bigger, the part around her (MapView's crop), followed smoothly
    k = fit * M.zoom;
    const mapW = (M.x1 - M.x0) * k + 2 * margin, mapH = (M.y1 - M.y0) * k + 2 * margin;
    H = Math.round(Math.min(W * M.around, mapH));
    const x = (at[0] - M.x0) * k + margin, y = (at[1] - M.y0) * k + margin;
    const gx = Math.min(Math.max(x - W / 2, 0), mapW - W), gy = Math.min(Math.max(y - H / 2, 0), mapH - H);
    if (!camera || camera.k !== k || Math.hypot(gx - camera.x, gy - camera.y) > W) camera = {k, x: gx, y: gy};
    else {
      const f = 1 - Math.exp(-dt / 0.08);
      camera.x += (gx - camera.x) * f; camera.y += (gy - camera.y) * f;
    }
    left = camera.x; top = camera.y;
  } else camera = null;
  const key = [rev, W, H, dpr, view, at && at[0].toFixed(2), at && at[1].toFixed(2), left.toFixed(1), top.toFixed(1)].join();
  if (key === lastKey) return;   // nothing moved: nothing to draw
  lastKey = key;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  }
  cv.style.height = H + "px";
  const t0 = performance.now();
  draw({k, ox: margin - M.x0 * k - left, oy: margin - M.y0 * k - top, u, W, H, dpr}, her, at, when);
  perf.drawn++;
  keep(perf.ms, performance.now() - t0, 240);
  if (!crowded) $("empty").style.display = "none";
}

function draw(v, her, at, when) {
  const {k, ox, oy, u} = v;
  const X = x => x * k + ox, Y = y => y * k + oy, P = (x, y) => [X(x), Y(y)];
  const font = Math.max(7, 11 * u) + "px system-ui, sans-serif";
  ctx.setTransform(v.dpr, 0, 0, v.dpr, 0, 0);
  ctx.globalAlpha = 1;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.fillStyle = C.background;
  ctx.fillRect(0, 0, v.W, v.H);
  function stroke(colour, width, path) {
    ctx.strokeStyle = colour; ctx.lineWidth = width * u; ctx.beginPath(); path(); ctx.stroke();
  }
  function seg(x1, y1, x2, y2) { ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x2), Y(y2)); }
  function platform(pid, colour, width) { stroke(colour, width, () => { for (const s of M.platforms[pid]) seg(...s); }); }
  function circle(p, r, colour, width) {   // width < 0: filled (as OpenCV's thickness)
    ctx.beginPath(); ctx.arc(p[0], p[1], r * u, 0, 2 * Math.PI);
    if (width < 0) { ctx.fillStyle = colour; ctx.fill(); }
    else { ctx.strokeStyle = colour; ctx.lineWidth = width * u; ctx.stroke(); }
  }
  function box(p, r, colour, width) {      // cv2.rectangle from p - r to p + r
    if (width < 0) { ctx.fillStyle = colour; ctx.fillRect(p[0] - (r + .5) * u, p[1] - (r + .5) * u, (2 * r + 1) * u, (2 * r + 1) * u); }
    else { ctx.strokeStyle = colour; ctx.lineWidth = width * u; ctx.strokeRect(p[0] - r * u, p[1] - r * u, 2 * r * u, 2 * r * u); }
  }
  function arrow(a, b, colour, width, tip) {   // cv2.arrowedLine
    const size = Math.hypot(b[0] - a[0], b[1] - a[1]) * tip, angle = Math.atan2(a[1] - b[1], a[0] - b[0]);
    stroke(colour, width, () => {
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
      for (const turn of [Math.PI / 4, -Math.PI / 4]) {
        ctx.moveTo(b[0] + size * Math.cos(angle + turn), b[1] + size * Math.sin(angle + turn)); ctx.lineTo(b[0], b[1]);
      }
    });
  }
  function label(text, p, colour) {
    ctx.fillStyle = colour; ctx.font = font; ctx.fillText(text, p[0], p[1]);
  }

  // the map (MapView._draw_base)
  stroke(C.wall, 1, () => { for (const w of M.walls) seg(...w); });
  stroke(C.rope, 1, () => { for (const [x, t, b] of M.ropes) seg(x, t, x, b); });
  stroke(C.platform, 2, () => { for (const p of M.platforms) for (const s of p) seg(...s); });
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  M.platforms.forEach((segs, pid) => { const p = P(segs[0][0], segs[0][1]); label(String(pid), [p[0], p[1] - 4 * u], C.dim); });
  for (const [x, y] of M.portals) circle(P(x, y), 4, C.portal, 1);
  // the tour's platforms (_platform_roles)
  const tour = S.tour || {};
  for (const pid of tour.safe || []) if (known(pid)) platform(pid, C.safe, 2);
  for (const pid of tour.grounds || []) if (known(Math.trunc(Number(pid)))) platform(Math.trunc(Number(pid)), C.ground, 3);
  // the tour's stops and spot (_tour)
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  (tour.order || []).forEach((pid, i) => {
    if (!known(pid)) return;
    const segs = M.platforms[pid], mid = (segs[0][0] + segs[segs.length - 1][2]) / 2, p = P(mid, yAt(pid, mid));
    const current = pid === tour.stop, c = [p[0], p[1] + 12 * u];
    circle(c, 8, current ? C.her : C.ground, current ? -1 : 1);
    label(String(i + 1), c, current ? C.background : C.ground);
  });
  const spot = tour.spot;
  if (spot && known(spot.platform == null ? -1 : spot.platform)) {
    const p = P(spot.x, yAt(spot.platform, spot.x)), y = p[1] - 6 * u, r = 6 * u;
    stroke(C.her, 2, () => { ctx.moveTo(p[0], y - r); ctx.lineTo(p[0] + r, y); ctx.lineTo(p[0], y + r); ctx.lineTo(p[0] - r, y); ctx.closePath(); });
  }
  // monsters as the planner predicts them (_monsters)
  stroke(C.danger, 2, () => {
    for (const mo of (S.monsters || {}).list || []) {
      const pid = mo.platform, reach = mo.reach;
      if (reach && reach.length && pid != null && known(pid)) {
        const y = yAt(pid, (reach[0] + reach[1]) / 2) - 3; seg(reach[0], y, reach[1], y);
      }
    }
  });
  // the route (_route, _arc)
  ((S.route || {}).moves || []).forEach((mv, i) => {
    if (!(known(mv.src) && known(mv.dst))) return;
    const a = P(mv.x, yAt(mv.src, mv.x)), b = P(mv.land_x, yAt(mv.dst, mv.land_x));
    if (mv.kind === "jump" || mv.kind === "walk_off" || mv.kind === "hop") {
      const lift = Math.max(8 * u, M.jump_arc * k);
      stroke(C.route, 1, () => { ctx.moveTo(a[0], a[1]); ctx.quadraticCurveTo((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 2 * lift, b[0], b[1]); });
      circle(b, 2, C.route, -1);
    } else arrow(a, b, C.route, 1, 0.15);
    if (i === 0) circle(a, 3, C.route, -1);
  });
  // the chain in the running key plan (_chain)
  for (const leg of (S.chain || {}).legs || []) {
    stroke(C.chain, 3, () => { for (const [pid, x0, x1] of leg.spans || []) if (known(pid)) seg(x0, yAt(pid, x0) - 6, x1, yAt(pid, x1) - 6); });
    const end = leg.end;
    if (end && known(end.platform == null ? -1 : end.platform)) circle(P(end.x, yAt(end.platform, end.x) - 6), 3, C.chain, -1);
  }
  // drops remembered (_drops), the hunt loop's piles (_piles)
  for (const d of (S.drops || {}).pending || []) if (d.platform != null) box(P(d.x, d.y - 4), 4, d.mesos ? C.loot : C.drops, 1);
  for (const d of (S.hunt || {}).piles || []) {
    if (d.platform == null || d.y == null) continue;
    box(P(d.x, d.y - 4), d.status === "expected" ? 3 : 5, d.urgent ? C.danger : d.hers ? C.drops : C.dim, 1);
  }
  // where a star from her reaches (_reach), from where she's drawn
  if (her) {
    const R = M.reach, pid = her.platform;
    let lo = -Infinity, hi = Infinity;  // cut short by a wall of her platform's foothold group facing her (clear_span)
    if (known(pid) && M.stops) {
      const segs = M.platforms[pid], line = at[1] - R.shot_h;
      const a = Math.min(Math.max(at[0], segs[0][0] + 1), segs[segs.length - 1][2] - 1);
      for (const [wx, top, bottom, faces] of M.stops[pid] || []) {
        if (!(top <= line && line < bottom)) continue;
        if (wx <= a && faces > 0) lo = Math.max(lo, wx);
        if (wx >= a && faces < 0) hi = Math.min(hi, wx);
      }
    }
    ctx.globalAlpha = R.alpha;
    for (const sign of [-1, 1]) {
      const near = at[0] + sign * R.min;
      const far = sign < 0 ? Math.max(at[0] - R.star, lo) : Math.min(at[0] + R.star, hi);
      if ((far - near) * sign <= 0) continue;
      const a = P(Math.min(near, far), at[1] - R.up), b = P(Math.max(near, far), at[1] + R.down);
      ctx.fillStyle = her.facing === (sign > 0 ? "right" : "left") ? C.reach_facing : C.reach_behind;
      ctx.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
    }
    ctx.globalAlpha = 1;
  }
  // the monsters in the last look, those aimed at ringed (_monsters_seen), and the drops seen (_loot)
  const aimed = new Set(((S.stance || {}).aimed || []).map(a => Math.round(a.x) + "," + Math.round(a.y)));
  const scene = S.scene || {};
  for (const mo of scene.monsters || []) {
    const p = P(mo.x, mo.y - 12);
    circle(p, 4, M.kind[mo.kind] || C.text, -1);
    if (aimed.has(Math.round(mo.x) + "," + Math.round(mo.y))) circle(p, 8, C.danger, 1);
    if (mo.bar != null) stroke(C.mob_hp, 2, () => { ctx.moveTo(p[0] - 6 * u, p[1] - 7 * u); ctx.lineTo(p[0] + (Math.trunc(12 * mo.bar) - 6) * u, p[1] - 7 * u); });
  }
  for (const d of scene.loot || []) box(P(d.x, d.y - 4), 2, C.loot, -1);
  // other players, by their minimap dots (_players)
  for (const pl of scene.players || []) {
    const p = P(pl.x, pl.y - 8);
    circle(p, 5, M.players[pl.kind] || C.text, -1); circle(p, 6, C.background, 1);
  }
  // her trail, up to where she's drawn (_trail), and her (_her)
  const trail = [];
  for (const p of S.trail) if (when == null || !(p[2] > when)) trail.push(p);
  if (her) trail.push(at);
  if (trail.length >= 2) stroke(C.her_trail, 1, () => { trail.forEach((p, i) => { i ? ctx.lineTo(X(p[0]), Y(p[1] - 4)) : ctx.moveTo(X(p[0]), Y(p[1] - 4)); }); });
  if (her) {
    const p = P(at[0], at[1] - 8);
    circle(p, 6, C.her, -1);
    if (her.facing === "left" || her.facing === "right") arrow(p, [p[0] + 14 * u * (her.facing === "right" ? 1 : -1), p[1]], C.background, 2, 0.5);
    if (her.seen === false) circle(p, 9, C.unseen, 1);
  }
}

// --- the text ----------------------------------------------------------------------------------------

function pct(v) { return v == null ? "-" : Math.round(v * 100) + "%"; }
function clockText(t) { return t ? new Date(t * 1000).toLocaleTimeString([], {hour12: false}) : "--:--:--"; }
function text(id, v) { $(id).textContent = v == null || v === "" ? "-" : v; }

function show(s) {
  text("title", (s.run ? s.run : "gamelab") + (s.map != null ? " · map " + s.map : "") + (s.replay ? " · replay" : ""));
  $("hp").style.width = s.hp == null ? "0" : Math.round(s.hp * 100) + "%";
  $("mp").style.width = s.mp == null ? "0" : Math.round(s.mp * 100) + "%";
  text("hpv", pct(s.hp)); text("mpv", pct(s.mp));
  const h = s.hunt;
  text("hunt", h ? (h.rule || "-") + (h.objective ? " — " + h.objective : h.ground != null ? " — platform " + h.ground : "") :
               (s.stop != null ? "tour stop: platform " + s.stop : null));
  let rates = "";
  if (h && h.rate_here != null) {
    rates = h.rate_here.toFixed(2) + " EXP/s here";
    if (h.rate_to_beat) rates += ", " + h.rate_to_beat[0].toFixed(2) + " at platform " + h.rate_to_beat[1];
  }
  if (s.exp && s.exp.exp != null) {  // her EXP as read off its number
    rates += (rates ? " · " : "") + "EXP " + s.exp.exp.toLocaleString("en-US") +
             (s.exp.per_minute != null ? " (" + Math.round(s.exp.per_minute) + " a minute)" : "");
  }
  $("rates").textContent = rates;
  text("fight", s.fight); text("moving", s.moving);
  text("where", s.her ? "x " + Math.round(s.her.x) + " · platform " + (s.her.platform == null ? "?" : s.her.platform) +
                        " (" + (s.her.how || "?") + ") · " + s.monsters + " monsters, " + s.loot + " drops in view" +
                        (s.players ? " · " + s.players + (s.players > 1 ? " other players" : " other player") : "") : "lost");
  if (s.replay) {
    $("controls").style.display = "flex";
    $("play").textContent = s.replay.paused ? "Play" : "Pause";
  }
}

function tick() {   // 4 times a second: the text, the log, the clock
  if (textChanged) { show(status); textChanged = false; }
  if (logsChanged) {
    const log = $("log"); log.replaceChildren();
    for (let i = S.logs.length - 1; i >= 0; i--) { const li = document.createElement("li"); li.textContent = S.logs[i]; log.appendChild(li); }
    logsChanged = false;
  }
  $("dot").className = "dot " + (lost() ? "lost" : status.replay ? "replay" : "live");
  $("time").textContent = clockText(S.t);
  const r = status.replay;
  if (r) {
    const t = runNow(), pos = t == null || r.start == null ? r.pos : Math.min(Math.max(t - r.start, 0), r.length);
    $("pos").textContent = (r.paused ? "paused · " : "") + "×" + r.speed + " · " + Math.round(pos) + " / " + Math.round(r.length) + " s";
  }
  const empty = $("empty");
  empty.classList.toggle("over", crowded > 0);
  if (crowded) {
    empty.textContent = "Stopped: " + crowded + (crowded === 1 ? " newer page is" : " newer pages are") + " watching. Reload to watch here.";
    empty.style.display = "";
  } else if (!M) empty.textContent = lost() && quiet() > 3000 ? "can't reach the mini-map: trying again..." : "waiting for a run...";
}

// ?debug=1: how fast it draws, what comes over the network, and how late it comes
const lates = [];   // [when it came (performance.now ms), how late its newest record came (ms)]
function late(records) {
  // Against the run's clock as the server sent it, set by the message that came quickest: how much
  // later than that this one came, and how long its record waited to be sent. Wi-Fi adds about half
  // the round trip (shown next to it). Not how old the newest record is: between moves she's looked
  // at only a few times a second, so that's often several hundred ms with nothing slow.
  let t = null;
  for (const r of records) if (typeof r.t === "number" && (t == null || r.t > t)) t = r.t;
  if (t != null && clock) keep(lates, [performance.now(), (runNow() - t) / (clock.rate || 1) * 1000], 2000);
}
let was = {t: performance.now(), frames: 0, drawn: 0, msgs: 0, bytes: 0};
function debug() {
  const now = performance.now(), dt = (now - was.t) / 1000;
  const ms = perf.ms.slice().sort((a, b) => a - b), q = f => ms.length ? ms[Math.floor(f * (ms.length - 1))].toFixed(2) : "-";
  const recent = lates.filter(l => now - l[0] < 5000).map(l => l[1]).sort((a, b) => a - b);
  const lq = f => recent.length ? Math.round(recent[Math.floor(f * (recent.length - 1))]) : "-";
  const t = runNow(), age = t == null || !S.t ? "-" : Math.round((t - S.t) * 1000);
  $("debug").textContent =
    "fps " + ((perf.frames - was.frames) / dt).toFixed(0) + ", drawn " + ((perf.drawn - was.drawn) / dt).toFixed(0) + "/s\n" +
    "draw ms p50 " + q(.5) + " p90 " + q(.9) + " max " + (ms.length ? ms[ms.length - 1].toFixed(2) : "-") + "\n" +
    "msgs " + ((net.msgs - was.msgs) / dt).toFixed(1) + "/s, " + ((net.bytes - was.bytes) / dt / 1024).toFixed(2) +
    " KiB/s, " + (net.bytes / 1024).toFixed(0) + " KiB\n" +
    "reconnects " + Math.max(0, net.opens - 1) + " (" + net.restarts + " by the page)\n" +
    "late p50 " + lq(.5) + " p90 " + lq(.9) + " ms (last 5 s) + rtt/2 " + (net.rtt == null ? "-" : Math.round(net.rtt / 2)) + " ms\n" +
    "newest record " + age + " ms ago\n" +
    "her " + Math.round(DELAY * 1000) + " ms behind, canvas " + cv.width + "x" + cv.height + " at " +
    +(window.devicePixelRatio || 1).toFixed(2) + "x";
  was = {t: now, frames: perf.frames, drawn: perf.drawn, msgs: net.msgs, bytes: net.bytes};
  const sent = performance.now();
  fetch(withKey("clock.json"), {cache: "no-store"}).then(r => r.arrayBuffer()).then(() => { net.rtt = performance.now() - sent; }).catch(() => {});
}

function setView(v) {
  view = v;
  try { localStorage.setItem("gamelab-view", v); } catch (e) {}
  $("v-her").classList.toggle("on", v === "her");
  $("v-all").classList.toggle("on", v === "all");
  camera = null; lastKey = "";
}
for (const b of document.querySelectorAll(".views button")) b.addEventListener("click", () => setView(b.dataset.v));
setView(view);
for (const b of document.querySelectorAll("#controls button")) {
  b.addEventListener("click", () => fetch(withKey("control?action=" + b.dataset.a), {method: "POST"}));
}

// On a computer, show how to open it on a phone: the network address, and a QR code to scan.
const lan = URLS.find(u => !u.includes("127.0.0.1"));
if (lan && window.matchMedia("(min-width: 900px)").matches && ["127.0.0.1", "localhost"].includes(location.hostname)) {
  $("phone").style.display = "block";
  $("lan").textContent = lan;
  const s = document.createElement("script");
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js";
  s.onload = () => new QRCode($("qr"), {text: lan, width: 168, height: 168});
  s.onerror = () => { $("qr").style.display = "none"; };
  document.head.appendChild(s);
}
if (DEBUG) { $("debug").hidden = false; setInterval(debug, 1000); }
wantMap();
connect(false);
setInterval(tick, 250);
tick();
requestAnimationFrame(frame);
</script>
</body>
</html>
"""
