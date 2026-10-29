"""Overlaid recordings in the web UI, for a phone (the user, 2026-10-29: "Can you serve this on webui? I need to look at
th on my phone"): perception's outcome drawn on each frame of a recording (D:/MapleHelper-scratch/golden-tree/
overlay-quick-2min, the overlay tool's output), stepped through one frame at a time.

A set is a folder under ROOT (GAMELAB_REVIEW_ROOT, else the scratch folder's golden-tree beside the repo) with a
`frames/` folder of 00000.jpg ...; it may have legend.png, overlay.mp4 (served with byte ranges: a phone's video player
asks for them), README.md, LEGEND.md and problems.json ([{"frames": [a, b], "what": "..."}], the frames worth a look).
    /review                       the sets
    /review/<set>/                the viewer
    /review/<set>/meta.json       its frame count and problems
    /review/<set>/frames/N.jpg    a frame; legend.png, overlay.mp4, README.md, LEGEND.md
Names are checked (letters, digits, - and _; frames by number), so nothing outside a set is served."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .. import paths

ROOT = Path(os.environ.get("GAMELAB_REVIEW_ROOT") or paths.REPO.parent / "MapleHelper-scratch" / "golden-tree")
NAME = re.compile(r"^[A-Za-z0-9_-]+$")
FRAME = re.compile(r"^\d{5}\.jpg$")
FILES = {"legend.png": "image/png", "overlay.mp4": "video/mp4", "README.md": "text/plain; charset=utf-8",
         "LEGEND.md": "text/plain; charset=utf-8"}
CHUNK = 1 << 20  # bytes a write, sending a big file


def sets(root: Path | None = None) -> list[str]:
    """The sets under `root` (ROOT): folders with a frames/ folder, by name."""
    root = root or ROOT
    try:
        return sorted(d.name for d in root.iterdir() if d.is_dir() and NAME.match(d.name) and (d / "frames").is_dir())
    except OSError:
        return []


def meta(folder: Path) -> dict:
    frames = sorted(p.name for p in (folder / "frames").iterdir() if FRAME.match(p.name))
    try:
        problems = json.loads((folder / "problems.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        problems = []
    return {"name": folder.name, "frames": len(frames), "last": int(frames[-1][:5]) if frames else -1,
            "problems": problems, "files": [f for f in FILES if (folder / f).is_file()]}


def handle(handler, path: str, root: Path | None = None) -> bool:
    """Answer a GET for `path` if it's the review's (/review...): True if answered."""
    if path != "/review" and not path.startswith("/review/"):
        return False
    root = root or ROOT
    parts = [p for p in path.split("/")[2:] if p]
    if not parts:
        body = "".join(f'<li><a href="/review/{n}/">{n}</a></li>' for n in sets(root)) or "<li>none</li>"
        handler._send(200, (f"<!doctype html><meta name=viewport content='width=device-width'><title>Review</title>"
                            f"<body style='font:16px sans-serif;background:#111;color:#eee'><h3>Recordings</h3>"
                            f"<ul>{body}</ul>").encode(), "text/html; charset=utf-8")
        return True
    name = parts[0]
    folder = root / name
    if not NAME.match(name) or not (folder / "frames").is_dir():
        handler._send(404, b"no such recording", "text/plain")
        return True
    rest = parts[1:]
    if not rest:
        handler._send(200, viewer(name), "text/html; charset=utf-8")
    elif rest == ["meta.json"]:
        handler._send(200, json.dumps(meta(folder)).encode(), "application/json")
    elif len(rest) == 2 and rest[0] == "frames" and FRAME.match(rest[1]):
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
body { margin:0; background:var(--bg); color:var(--fg); font:15px system-ui, sans-serif; }
header { position:sticky; top:0; background:var(--bg); padding:6px 8px; z-index:2; border-bottom:1px solid #222; }
#title { display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
#title b { font-size:15px; } #pos { color:var(--acc); font-variant-numeric:tabular-nums; }
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
  <div id="title"><b>__NAME__</b><span id="pos">-</span></div>
  <div class="row">
    <button id="b15">&laquo; 15</button><button id="b1">&lsaquo; 1</button><button id="play">Play</button>
    <button id="f1">1 &rsaquo;</button><button id="f15">15 &raquo;</button>
  </div>
  <div class="row">
    <input type="range" id="slider" min="0" max="0" value="0">
    <input id="go" type="number" inputmode="numeric" placeholder="frame">
    <select id="rate"><option value="2">2 fps</option><option value="5" selected>5 fps</option>
      <option value="15">15 fps</option></select>
  </div>
</header>
<div id="imgbox"><img id="img" alt="frame"></div>
<div class="row" style="padding:0 8px"><button id="zoom">Full size (scroll)</button></div>
<div id="problems"></div>
<div class="links" id="links"></div>
<script>
const base = location.pathname.replace(/\/?$/, "/");
let M = {frames: 0, last: -1, problems: []}, i = 0, timer = null;
const $ = id => document.getElementById(id), img = $("img"), cache = {};
const pad = n => String(n).padStart(5, "0");
const src = n => base + "frames/" + pad(n) + ".jpg";
function preload(n) { for (let k = 1; k <= 4; k++) { const m = n + k; if (m <= M.last && !cache[m]) {
  const im = new Image(); im.src = src(m); cache[m] = im; } } }
function show(n) {
  i = Math.max(0, Math.min(M.last, n | 0)); img.src = src(i); $("slider").value = i;
  $("pos").textContent = i + " / " + M.last; history.replaceState(null, "", "#" + i); preload(i);
}
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
  $("zoom").textContent = b.classList.contains("native") ? "Fit to width" : "Full size (scroll)"; };
let sx = null;  // swipe on the picture: left/right a frame
img.addEventListener("touchstart", e => { if (e.touches.length === 1) sx = e.touches[0].clientX; }, {passive: true});
img.addEventListener("touchend", e => { if (sx === null) return; const dx = e.changedTouches[0].clientX - sx; sx = null;
  if (Math.abs(dx) > 50 && !$("imgbox").classList.contains("native")) step(dx < 0 ? 1 : -1); });
document.addEventListener("keydown", e => { if (e.target.tagName === "INPUT") return;
  if (e.key === "ArrowRight") step(1); else if (e.key === "ArrowLeft") step(-1); else if (e.key === " ") { e.preventDefault(); play(); } });
fetch(base + "meta.json").then(r => r.json()).then(m => {
  M = m; $("slider").max = M.last;
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
