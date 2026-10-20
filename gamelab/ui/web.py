"""The mini-map as a web page, to watch a run from a phone on the same network: `python -m gamelab web`.

A small HTTP server (standard library only) follows the run's telemetry as the window does (a run live,
or a recorded one replayed), draws the map with render.MapView, and serves:

    /             the page: the map, and what she's doing as text (HP and MP, the hunt loop's rule, the
                  fight, the moves, the latest log lines), refreshed a few times a second
    /frame.jpg    the map as of the latest record (?view=her: zoomed in around her, for a phone's screen)
    /state.json   the text
    /control      replays only (POST action=pause|faster|slower|back|on): play, speed, jump 10 s

It only reads telemetry: it sends no keys and never touches the game. Every request needs the key printed
at the start (?k=..., then a cookie), so nobody else on the network can watch.
"""

import json
import secrets
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

from .render import MapView
from .state import State

POLL_S = 0.1     # how often the telemetry is read
LOG_KEEP = 40    # log lines the page shows
SEEK_S = 10.0    # a replay's jump back or on
JPEG_QUALITY = 80
ZOOM = 2.5       # "around her": the map drawn this much bigger, and the part around her cut out ...
AROUND = 0.8     # ... as tall as this share of its width


class Feed:
    """The run's state, kept up to date from `source` (sources.LiveSource or ReplaySource) by a thread,
    and the map drawn from it with the view `view_for(map id)` gives (None: the map to start with)."""

    def __init__(self, view_for: Callable[[int | None], MapView], source, live: bool):
        self.view_for, self.source, self.live = view_for, source, live
        self.state = State()
        self.view = view_for(None)
        self.logs: deque[str] = deque(maxlen=LOG_KEEP)
        self.version = 0                 # goes up with every change: the page reloads the map then
        self.lock = threading.Lock()
        self._jpeg: dict[str, tuple[int, bytes]] = {}  # view -> (version, picture)
        self._zoomed: dict[int, MapView] = {}           # map id -> the view "around her" is cut from
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="telemetry", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.step()
            time.sleep(POLL_S)

    def step(self) -> None:
        """Read what the source has, and apply it."""
        fresh, records = self.source.poll()
        if not (fresh or records):
            return
        with self.lock:
            if fresh:  # a newer run started
                self.state, self.logs = State(), deque(maxlen=LOG_KEEP)
            self._apply(records)

    def _apply(self, records: list[dict]) -> None:
        for r in records:
            self.state.apply(r)
            if r.get("topic") in ("log", "event"):
                self.logs.append(r.get("text") or r.get("kind", ""))
        if self.state.map is not None and self.state.map != getattr(self.view.model, "id", None):
            self.view = self.view_for(self.state.map)
        self.version += 1

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

    def status(self) -> dict:
        """What the page shows as text."""
        with self.lock:
            s = self.state
            scene = s.scene or {}
            her = s.her()
            hunt = s.hunt or {}
            out = {
                "version": self.version, "live": self.live, "t": s.t or None, "map": s.map,
                "run": s.session.get("scenario") or s.session.get("command"), "mode": s.session.get("mode"),
                "her": None if her is None else {k: her.get(k) for k in ("x", "y", "platform", "how")},
                "hp": scene.get("hp"), "mp": scene.get("mp"),
                "monsters": len(scene.get("monsters") or []), "loot": len(scene.get("loot") or []),
                "fight": (s.stance or {}).get("text") or (s.stance or {}).get("act"),
                "moving": (s.chain or {}).get("text"),
                "hunt": None if not hunt else {k: hunt.get(k) for k in
                                               ("rule", "objective", "ground", "rate_here", "rate_to_beat")},
                "stop": (s.tour or {}).get("stop"),
                "logs": list(self.logs)[::-1],  # the newest first
            }
            if not self.live:
                src = self.source
                end = src.records[-1].get("t", 0.0) - src.t0 if src.records else 0.0
                out["replay"] = {"paused": src.paused, "speed": src.speed, "pos": round(src.pos, 1),
                                 "length": round(end, 1)}
            return out

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
                records = src.seek(SEEK_S if action == "on" else -SEEK_S)
                self.state, self.logs = State(), deque(maxlen=LOG_KEEP)
                self._apply(records)
            else:
                return False
            self.version += 1
        return True


class WebServer(ThreadingHTTPServer):
    """The HTTP server: `feed`'s page, map and text, for requests with `key`."""

    daemon_threads = True

    def __init__(self, feed: Feed, host: str = "0.0.0.0", port: int = 8765, key: str | None = None):
        self.feed = feed
        self.key = key or secrets.token_urlsafe(8)
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
            page = PAGE.replace("__KEY__", json.dumps(self.server.key)).replace(
                "__URLS__", json.dumps(self.server.urls()))
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8", {"Set-Cookie": cookie})
        elif url.path == "/frame.jpg":
            self._send(200, feed.frame(query.get("view", ["all"])[0]), "image/jpeg")
        elif url.path == "/state.json":
            self._send(200, json.dumps(feed.status()).encode("utf-8"), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
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
  .map { position: relative; border-radius: 10px; overflow: hidden; background: #000; }
  .map img { display: block; width: 100%; height: auto; }
  .map .empty { position: absolute; inset: 0; display: grid; place-items: center; color: var(--dim); }
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
      <div class="map"><img id="map" alt="the map"><div class="empty" id="empty">waiting for a run...</div></div>
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
const KEY = __KEY__, URLS = __URLS__;
const $ = id => document.getElementById(id);
let version = -1, loading = false, wanted = -1, failures = 0;
let view = "all";
try { view = localStorage.getItem("gamelab-view") || (window.innerWidth < 700 ? "her" : "all"); }
catch (e) { view = window.innerWidth < 700 ? "her" : "all"; }

function setView(v) {
  view = v;
  try { localStorage.setItem("gamelab-view", v); } catch (e) {}
  $("v-her").classList.toggle("on", v === "her");
  $("v-all").classList.toggle("on", v === "all");
  version = -1; loadFrame();
}

function pct(v) { return v == null ? "-" : Math.round(v * 100) + "%"; }
function clock(t) { return t ? new Date(t * 1000).toLocaleTimeString([], {hour12: false}) : "--:--:--"; }
function text(id, v) { $(id).textContent = v == null || v === "" ? "-" : v; }

function loadFrame() {
  if (loading || wanted === version || wanted < 0) return;
  loading = true;
  const v = wanted;
  const img = new Image();
  img.onload = () => { $("map").src = img.src; $("empty").style.display = "none"; version = v; loading = false; loadFrame(); };
  img.onerror = () => { loading = false; };
  img.src = "frame.jpg?k=" + encodeURIComponent(KEY) + "&view=" + view + "&v=" + v;
}

function show(s) {
  const dot = $("dot");
  dot.className = "dot " + (s.live ? "live" : "replay");
  text("title", (s.run ? s.run : "gamelab") + (s.map != null ? " · map " + s.map : "") + (s.live ? "" : " · replay"));
  $("time").textContent = clock(s.t);
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
  $("rates").textContent = rates;
  text("fight", s.fight); text("moving", s.moving);
  text("where", s.her ? "x " + Math.round(s.her.x) + " · platform " + (s.her.platform ?? "?") + " (" + (s.her.how || "?") + ")"
                     + " · " + s.monsters + " monsters, " + s.loot + " drops in view" : "lost");
  const log = $("log"); log.replaceChildren();
  for (const line of s.logs) { const li = document.createElement("li"); li.textContent = line; log.appendChild(li); }
  if (s.replay) {
    $("controls").style.display = "flex";
    $("play").textContent = s.replay.paused ? "Play" : "Pause";
    $("pos").textContent = (s.replay.paused ? "paused · " : "") + "×" + s.replay.speed + " · " +
                           Math.round(s.replay.pos) + " / " + Math.round(s.replay.length) + " s";
  }
  if (s.version !== wanted) { wanted = s.version; loadFrame(); }
}

async function tick() {
  try {
    const r = await fetch("state.json?k=" + encodeURIComponent(KEY), {cache: "no-store"});
    if (!r.ok) throw new Error(r.status);
    show(await r.json());
    failures = 0;
  } catch (e) {
    if (++failures > 2) $("dot").className = "dot lost";
  }
  setTimeout(tick, document.hidden ? 2000 : 400);
}

for (const b of document.querySelectorAll(".views button")) b.addEventListener("click", () => setView(b.dataset.v));
setView(view);
for (const b of document.querySelectorAll("#controls button")) {
  b.addEventListener("click", () => fetch("control?k=" + encodeURIComponent(KEY) + "&action=" + b.dataset.a, {method: "POST"}));
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
tick();
</script>
</body>
</html>
"""
