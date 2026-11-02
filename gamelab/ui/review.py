"""Overlaid recordings in the web UI, for a phone (the user, 2026-10-29: "Can you serve this on webui? I need to look at
th on my phone"): perception's outcome drawn on each frame of a recording (D:/MapleHelper-scratch/golden-tree/
overlay-quick-2min, the overlay tool's output), stepped through one frame at a time.

A set is a folder under ROOT (GAMELAB_REVIEW_ROOT, else the scratch folder's golden-tree beside the repo) with a
`frames/` folder of 00000.jpg ...; it may have legend.png, overlay.mp4 (served with byte ranges: a phone's video player
asks for them), README.md, LEGEND.md and problems.json ([{"frames": [a, b], "what": "..."}], the frames worth a look).
A group is a folder under ROOT with no frames/ of its own and sets in it (the user, 2026-11-01: "put all of them under
the same review folder? e.g. `/review/ld-movers-kf/xxxx`"); one level only.
    /review                       the sets and the groups
    /review/<group>/              a group's sets
    /review/<set>/                the viewer (in a group, /review/<group>/<set>/ and so on below)
    /review/<set>/meta.json       its frame count, problems and frame rate (the viewer plays at it: the user,
                                  2026-11-02, "default to the recorded fps instead of 5fps")
    /review/<set>/frames/N.jpg    a frame; legend.png, overlay.mp4, README.md, LEGEND.md
    /review/<set>/frames/N.jpg?w=W&q=Q   the frame made smaller: at most W px wide, JPEG quality Q (the viewer asks
                                  this way unless its data saver is off: the user, 2026-11-02, "can the server send
                                  compressed image to reduce the network transfer? ... hurting my mobile bandwidth";
                                  a 750 px frame of 178 KB is about 40 KB at 540 px, a 1696 px one of 377 KB 17 KB)
Names are checked (letters, digits, - and _; frames by number), so nothing outside a set is served."""
from __future__ import annotations

import json
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .. import paths

ROOT = Path(os.environ.get("GAMELAB_REVIEW_ROOT") or paths.REPO.parent / "MapleHelper-scratch" / "golden-tree")
NAME = re.compile(r"^[A-Za-z0-9_-]+$")
FRAME = re.compile(r"^\d{5}\.jpg$")
RATE = re.compile(r"(\d+)fps(?:-|$)")  # a set named for its frame rate: 20261029-162605-15fps, ...-15fps-0-compare
FPS = 15  # a set's frame rate when nothing tells it (perception's)
FILES = {"legend.png": "image/png", "overlay.mp4": "video/mp4", "README.md": "text/plain; charset=utf-8",
         "LEGEND.md": "text/plain; charset=utf-8"}
CHUNK = 1 << 20  # bytes a write, sending a big file
SMALL_Q = (30, 95)  # the qualities a smaller frame may ask for (outside: clamped)
SMALL_W = (64, 4096)  # and widths (never wider than the frame)
SMALL_CACHE = 96 << 20  # bytes of smaller frames kept, the least recently sent going first
_small, _small_bytes, _small_lock = OrderedDict(), 0, threading.Lock()


def sets(root: Path | None = None) -> list[str]:
    """The sets under `root` (ROOT): folders with a frames/ folder, by name."""
    root = root or ROOT
    try:
        return sorted(d.name for d in root.iterdir() if d.is_dir() and NAME.match(d.name) and (d / "frames").is_dir())
    except OSError:
        return []


def groups(root: Path | None = None) -> list[str]:
    """The groups under `root` (ROOT): folders with no frames/ folder and sets in them, by name."""
    root = root or ROOT
    try:
        return sorted(d.name for d in root.iterdir()
                      if d.is_dir() and NAME.match(d.name) and not (d / "frames").is_dir() and sets(d))
    except OSError:
        return []


def _listing(handler, title: str, links: list[tuple[str, str]]) -> None:
    body = "".join(f'<li><a href="{href}">{text}</a></li>' for href, text in links) or "<li>none</li>"
    handler._send(200, (f"<!doctype html><meta name=viewport content='width=device-width'><title>{title}</title>"
                        f"<body style='font:16px sans-serif;background:#111;color:#eee'><h3>{title}</h3>"
                        f"<ul>{body}</ul>").encode(), "text/html; charset=utf-8")


def meta(folder: Path) -> dict:
    frames = sorted(p.name for p in (folder / "frames").iterdir() if FRAME.match(p.name))
    try:
        problems = json.loads((folder / "problems.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        problems = []
    return {"name": folder.name, "frames": len(frames), "last": int(frames[-1][:5]) if frames else -1,
            "problems": problems, "files": [f for f in FILES if (folder / f).is_file()], "fps": fps(folder)}


def fps(folder: Path) -> float:
    """The set's frame rate: its overlay.mp4's, else the one its name gives (...-15fps), else FPS."""
    if (folder / "overlay.mp4").is_file():
        video = cv2.VideoCapture(str(folder / "overlay.mp4"))
        rate = video.get(cv2.CAP_PROP_FPS)
        video.release()
        if 0 < rate <= 120:
            return round(rate, 2)
    m = RATE.search(folder.name)
    return float(m.group(1)) if m and 0 < int(m.group(1)) <= 120 else float(FPS)


def _number(query: dict, key: str, lo: int, hi: int) -> int | None:
    try:
        return min(hi, max(lo, int(query[key][0])))
    except (KeyError, IndexError, ValueError):
        return None


def smaller(file: Path, width: int | None, quality: int | None) -> bytes | None:
    """The frame at most `width` px wide (its own width when None or wider), as a JPEG of `quality` (85 when None);
    None if it can't be read or decoded. Kept in memory (SMALL_CACHE), keyed by the file's time and size too."""
    global _small_bytes
    try:
        st = file.stat()
    except OSError:
        return None
    key = (str(file), st.st_mtime_ns, st.st_size, width, quality)
    with _small_lock:
        if key in _small:
            _small.move_to_end(key)
            return _small[key]
    try:
        img = cv2.imdecode(np.fromfile(str(file), np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        img = None
    if img is None:
        return None
    h, w = img.shape[:2]
    if width is not None and width < w:
        img = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85 if quality is None else quality])
    if not ok:
        return None
    body = enc.tobytes()
    with _small_lock:
        _small[key] = body
        _small_bytes += len(body)
        while _small_bytes > SMALL_CACHE and _small:
            _small_bytes -= len(_small.popitem(last=False)[1])
    return body


def _send_cached(handler, body: bytes, kind: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", kind)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "max-age=86400")
    handler.end_headers()
    handler.wfile.write(body)


def handle(handler, path: str, root: Path | None = None, query: dict | None = None) -> bool:
    """Answer a GET for `path` (its parsed query string `query`) if it's the review's (/review...): True if
    answered."""
    if path != "/review" and not path.startswith("/review/"):
        return False
    root = root or ROOT
    parts = [p for p in path.split("/")[2:] if p]
    if not parts:
        _listing(handler, "Recordings", [(f"/review/{g}/", f"{g}/") for g in groups(root)]
                 + [(f"/review/{n}/", n) for n in sets(root)])
        return True
    if NAME.match(parts[0]) and parts[0] in groups(root):
        group = parts[0]
        if len(parts) == 1:
            _listing(handler, group, [(f"/review/{group}/{n}/", n) for n in sets(root / group)])
            return True
        root, parts, label = root / group, parts[1:], f"{group}/"
    else:
        label = ""
    name = parts[0]
    folder = root / name
    if not NAME.match(name) or not (folder / "frames").is_dir():
        handler._send(404, b"no such recording", "text/plain")
        return True
    rest = parts[1:]
    if not rest:
        handler._send(200, viewer(label + name), "text/html; charset=utf-8")
    elif rest == ["meta.json"]:
        handler._send(200, json.dumps(meta(folder)).encode(), "application/json")
    elif len(rest) == 2 and rest[0] == "frames" and FRAME.match(rest[1]):
        q = query or {}
        width, quality = _number(q, "w", *SMALL_W), _number(q, "q", *SMALL_Q)
        body = smaller(folder / "frames" / rest[1], width, quality) if width or quality else None
        if body is not None:
            _send_cached(handler, body, "image/jpeg")
        else:  # as it is: asked so, or it couldn't be made smaller
            _file(handler, folder / "frames" / rest[1], "image/jpeg", cache=True)
    elif len(rest) == 1 and rest[0] in FILES:
        _file(handler, folder / rest[0], FILES[rest[0]], cache=True)
    else:
        handler._send(404, b"not found", "text/plain")
    return True


def _file(handler, file: Path, kind: str, cache: bool = False) -> None:
    """A file, whole or the byte range asked for (Range: bytes=a-b, as a phone's video player asks)."""
    try:
        size = file.stat().st_size
    except OSError:
        handler._send(404, b"not found", "text/plain")
        return
    start, end, code = 0, size - 1, 200
    got = re.match(r"bytes=(\d*)-(\d*)$", handler.headers.get("Range") or "")
    if got and (got.group(1) or got.group(2)):
        if got.group(1):
            start = int(got.group(1))
            end = min(int(got.group(2)), size - 1) if got.group(2) else size - 1
        else:  # the last N bytes
            start = max(0, size - int(got.group(2)))
        if start > end or start >= size:
            handler.send_response(416)
            handler.send_header("Content-Range", f"bytes */{size}")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        code = 206
    handler.send_response(code)
    handler.send_header("Content-Type", kind)
    handler.send_header("Content-Length", str(end - start + 1))
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
    if code == 206:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.end_headers()
    with open(file, "rb") as f:
        f.seek(start)
        left = end - start + 1
        while left > 0:
            chunk = f.read(min(CHUNK, left))
            if not chunk:
                break
            handler.wfile.write(chunk)
            left -= len(chunk)


def viewer(name: str) -> bytes:
    return (_VIEWER.replace("__NAME__", name)).encode()


_VIEWER = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__NAME__</title>
<style>
:root { --bg:#101214; --fg:#e8e8e8; --dim:#9aa0a6; --btn:#23272b; --acc:#4fc3f7; }
/* No double-tap zoom anywhere (two quick taps on a button stepped and zoomed the page, the user, 2026-10-29: "Fix
   the buttons, double tap on phone zooms in"); pinch zoom and scrolling stay. */
html, body, button, select, input, #imgbox, #img { touch-action: manipulation; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px system-ui, sans-serif;
  -webkit-tap-highlight-color: transparent; }
button { user-select:none; -webkit-user-select:none; }
header { position:sticky; top:0; background:var(--bg); padding:6px 8px; z-index:2; border-bottom:1px solid #222; }
#title { display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
#title b { font-size:15px; } #pos { color:var(--acc); font-variant-numeric:tabular-nums; }
#load { color:var(--dim); font-size:13px; font-variant-numeric:tabular-nums; margin-left:auto; }
#imgbox { overflow:auto; -webkit-overflow-scrolling:touch; background:#000; }
#img { display:block; width:100%; height:auto; }
#imgbox.native #img { width:auto; max-width:none; }
.row { display:flex; gap:6px; margin-top:6px; align-items:center; }
button, select, input { background:var(--btn); color:var(--fg); border:1px solid #333; border-radius:8px;
  font:15px system-ui; padding:9px 0; }
button { flex:1; min-width:0; } button:active { background:#2f353a; }
#go { width:76px; padding:9px 6px; } #rate { flex:0 0 86px; }
input[type=range] { flex:1; padding:0; }
#problems { padding:8px; } #problems h4 { margin:10px 0 6px; }
.p { display:block; width:100%; text-align:left; padding:9px 10px; margin:5px 0; }
.p span { color:var(--acc); margin-right:6px; font-variant-numeric:tabular-nums; }
a { color:var(--acc); } .links { padding:0 8px 16px; color:var(--dim); }
</style></head><body>
<header>
  <div id="title"><b>__NAME__</b><span id="load"></span><span id="pos">-</span></div>
  <div class="row">
    <button id="b15">&laquo; 15</button><button id="b1">&lsaquo; 1</button><button id="play">Play</button>
    <button id="f1">1 &rsaquo;</button><button id="f15">15 &raquo;</button>
  </div>
  <div class="row">
    <input type="range" id="slider" min="0" max="0" value="0">
    <input id="go" type="number" inputmode="numeric" placeholder="frame">
    <select id="rate"><option value="2">2 fps</option><option value="5">5 fps</option>
      <option value="15">15 fps</option></select>
  </div>
</header>
<div id="imgbox"><img id="img" alt="frame"></div>
<div class="row" style="padding:0 8px"><button id="zoom">Full size (scroll)</button>
  <button id="saver">Data saver: on</button></div>
<div id="problems"></div>
<div class="links" id="links"></div>
<script>
const base = location.pathname.replace(/\/?$/, "/");
let M = {frames: 0, last: -1, problems: []}, i = 0, timer = null;
const $ = id => document.getElementById(id), img = $("img"), cache = {};
const pad = n => String(n).padStart(5, "0");
// Data saver (on unless turned off; remembered): frames as wide as the picture shows them (1.5 device px a CSS px at
// most, in steps of 60 px so they cache), JPEG quality 60; full size keeps the frames as they are.
let saver = true, suffix = "";
try { saver = localStorage.getItem("review-saver") !== "off"; } catch (e) {}
const native = () => $("imgbox").classList.contains("native");
function sized() {  // the frames' query for the picture's width now
  const w = Math.min(4096, Math.ceil($("imgbox").clientWidth * Math.min(window.devicePixelRatio || 1, 1.5) / 60) * 60);
  suffix = !saver || native() ? "" : "?w=" + w + "&q=60";
}
const src = n => base + "frames/" + pad(n) + ".jpg" + suffix;
// Preloading (the user, 2026-11-02: "can ui preload the clip in browser, not wait until play button"): from the page's
// opening, the frames nearest ahead of the one shown, then just behind it, PARALLEL at a time, up to AHEAD frames ahead
// and BEHIND behind (a whole Lie Detector clip; the longest sets in windows that follow the frame shown).
const AHEAD = 600, BEHIND = 60, PARALLEL = 8;  // (through the tunnel a frame takes 0.5-1 s to arrive, 60 ms locally)
let active = 0, loaded = 0;
function nextToLoad() {
  for (let k = i; k <= Math.min(M.last, i + AHEAD); k++) if (!cache[src(k)]) return k;
  for (let k = i - 1; k >= Math.max(0, i - BEHIND); k--) if (!cache[src(k)]) return k;
  return -1;
}
function progress() {
  const want = Math.min(M.last, i + AHEAD) - Math.max(0, i - BEHIND) + 1;
  let ok = 0;
  for (let k = Math.max(0, i - BEHIND); k <= Math.min(M.last, i + AHEAD); k++) { const c = cache[src(k)]; if (c && c.done) ok++; }
  $("load").textContent = M.last < 0 || ok >= want ? "" : "loading " + ok + "/" + want;
}
function pump() {
  while (active < PARALLEL && M.last >= 0) {
    const n = nextToLoad();
    if (n < 0) break;
    const im = new Image();
    cache[src(n)] = im;
    active++;
    im.onload = im.onerror = () => { im.done = true; active--; progress(); pump(); };
    im.src = src(n);
  }
  progress();
}
function show(n) {
  i = Math.max(0, Math.min(M.last, n | 0)); img.src = src(i); $("slider").value = i;
  $("pos").textContent = i + " / " + M.last; history.replaceState(null, "", "#" + i); pump();
}
window.addEventListener("resize", () => { const before = suffix; sized(); if (suffix !== before) show(i); });
function step(d) { stop(); show(i + d); }
function stop() { if (timer) { clearInterval(timer); timer = null; $("play").textContent = "Play"; } }
function play() {
  if (timer) return stop();
  $("play").textContent = "Pause";
  timer = setInterval(() => { if (i >= M.last) return stop(); show(i + 1); }, 1000 / +$("rate").value);
}
$("b15").onclick = () => step(-15); $("b1").onclick = () => step(-1);
$("f1").onclick = () => step(1); $("f15").onclick = () => step(15); $("play").onclick = play;
$("rate").onchange = () => { if (timer) { stop(); play(); } };
$("slider").oninput = e => { stop(); show(+e.target.value); };
$("go").onchange = e => { stop(); show(+e.target.value); e.target.blur(); };
$("zoom").onclick = () => { const b = $("imgbox"); b.classList.toggle("native");
  $("zoom").textContent = b.classList.contains("native") ? "Fit to width" : "Full size (scroll)"; sized(); show(i); };
$("saver").textContent = "Data saver: " + (saver ? "on" : "off");
$("saver").onclick = () => { saver = !saver; $("saver").textContent = "Data saver: " + (saver ? "on" : "off");
  try { localStorage.setItem("review-saver", saver ? "on" : "off"); } catch (e) {} sized(); show(i); };
let sx = null;  // swipe on the picture: left/right a frame
img.addEventListener("touchstart", e => { if (e.touches.length === 1) sx = e.touches[0].clientX; }, {passive: true});
img.addEventListener("touchend", e => { if (sx === null) return; const dx = e.changedTouches[0].clientX - sx; sx = null;
  if (Math.abs(dx) > 50 && !$("imgbox").classList.contains("native")) step(dx < 0 ? 1 : -1); });
document.addEventListener("keydown", e => { if (e.target.tagName === "INPUT") return;
  if (e.key === "ArrowRight") step(1); else if (e.key === "ArrowLeft") step(-1); else if (e.key === " ") { e.preventDefault(); play(); } });
fetch(base + "meta.json").then(r => r.json()).then(m => {
  M = m; $("slider").max = M.last; sized();
  const rate = $("rate"), fps = String(M.fps || 15);  // the set's own frame rate, first (added when not listed)
  if (![...rate.options].some(o => o.value === fps)) {
    const o = new Option(fps + " fps", fps), after = [...rate.options].find(o => +o.value > +fps);
    rate.insertBefore(o, after || null);
  }
  rate.value = fps; [...rate.options].forEach(o => { o.textContent = o.value + " fps" + (o.value === fps ? " (recorded)" : ""); });
  const box = $("problems");
  if (M.problems.length) {
    box.innerHTML = "<h4>Worth a look</h4>";
    M.problems.forEach(p => { const b = document.createElement("button"); b.className = "p";
      const [a, z] = p.frames; b.innerHTML = "<span>" + a + (z && z !== a ? "-" + z : "") + "</span>" + p.what;
      b.onclick = () => { stop(); show(a); window.scrollTo({top: 0, behavior: "smooth"}); }; box.appendChild(b); });
  }
  $("links").innerHTML = M.files.map(f => '<a href="' + base + f + '">' + f + "</a>").join(" &middot; ");
  show(+(location.hash.slice(1) || 0));
});
</script></body></html>
"""
