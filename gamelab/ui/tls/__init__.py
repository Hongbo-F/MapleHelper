"""The recorder's captures with their TLS decrypted, on the mini-map's web page (python -m gamelab web): /tls/.

    /tls/                       the page (static/: index.html, app.js, app.css): the captures, a list of the requests
                                (HTTP/1.x, HTTP/2, WebSocket messages) or of the TLS connections, and one in full
    /tls/api/captures           the .pcap files under the capture roots (ROOTS), keyed ones and the newest first
    /tls/api/open?cap=&local=&reload=   decrypt one (in the background) / how far it is, and whether its files have
                                changed since (`stale`: a recording still going)
    /tls/api/summary?cap=       its connections and items (ui.tls.view)
    /tls/api/item?id=, conn?n=  one item / connection in full (its records, its HTTP/2 frames)
    /tls/api/body?id=&part=req|resp|msg&form=decoded|wire[&dl=1]   a body, as bytes (application/octet-stream)
    /tls/api/record?n=&dir=&i=, stream?n=&dir=&off=&len=   a record's plaintext, a decrypted stream
    /tls/api/search?q=          the items whose headers or bodies hold q
    /tls/api/csauth?cap=        its x-csauth calls as a timeline per computer, each event the diff against the
                                computer's state (ui.tls.csauth)

`local` is 1 / 0 / missing: frames over 1500 B trusted (captured on this PC) / not / by the capture's folder. Captures are
looked for under ROOTS: the repo's .analysis/captures (or GAMELAB_TLS_ROOT) AND the native recorder's dump dir D:\\data\\runs
(+ %USERPROFILE%\\data-runs), or exactly GAMELAB_TLS_ROOTS if set. Both recorders' file names are read (frames.pcap/stream.raw
and capture.pcap/tzsp.raw). Who may read it: ui.tls.access
(Cloudflare Access for what comes through the tunnel). The decryption is ui.tls.decrypt and ui.tls.ngs, imported on the
first request (they need `cryptography`), so the mini-map runs without them.

Captured bytes reach the page only as data: every body goes out as application/octet-stream with nosniff, and the page
puts them in as text, never as HTML."""
from __future__ import annotations

import json
import os
import re
import threading
import traceback
from pathlib import Path

from ... import paths

def _roots():
    """Where captures are looked for, newest-recorder location included.  Override the whole list with
    GAMELAB_TLS_ROOTS (os.pathsep-separated); else the repo's .analysis/captures (or GAMELAB_TLS_ROOT) PLUS the
    hardened native recorder's default dump dir D:\\data\\runs and its %USERPROFILE%\\data-runs fallback.  Dirs
    that don't exist are skipped at discovery time, so listing them all is harmless."""
    env = os.environ.get("GAMELAB_TLS_ROOTS")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p]
    out, seen = [], set()
    cands = [Path(os.environ.get("GAMELAB_TLS_ROOT") or paths.REPO / ".analysis" / "captures"),
             Path(r"D:\data\runs")]
    prof = os.environ.get("USERPROFILE")
    if prof:
        cands.append(Path(prof) / "data-runs")
    for p in cands:
        k = os.path.normcase(str(p))
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


ROOTS = _roots()
ROOT = ROOTS[0]       # the primary location (back-compat; the captures list spans all of ROOTS)
STATIC = Path(__file__).with_name("static")
FILES = {"": ("index.html", "text/html; charset=utf-8"), "index.html": ("index.html", "text/html; charset=utf-8"),
         "app.js": ("app.js", "text/javascript; charset=utf-8"), "app.css": ("app.css", "text/css; charset=utf-8")}
SAFE = {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
        "Content-Security-Policy": "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
                                   "frame-ancestors 'none'"}
STREAM_MAX = 4 << 20     # bytes of a decrypted stream one request gets
_library = None
_library_lock = threading.Lock()


def library():
    global _library
    with _library_lock:
        if _library is None:
            from .view import Library
            _library = Library([str(r) for r in ROOTS])
        return _library


def handle(handler, path: str, query: dict | None = None) -> bool:
    """Answer a GET for `path` if it's /tls...: True if answered."""
    if path != "/tls" and not path.startswith("/tls/"):
        return False
    from . import access

    def send(code, body, kind="application/json", extra=None):
        handler._send(code, body.encode() if isinstance(body, str) else body, kind, {**SAFE, **(extra or {})})

    why = access.refused(handler.headers)
    if why:
        send(403, why, "text/plain; charset=utf-8")
        return True
    if path == "/tls":
        send(301, b"", "text/plain", {"Location": "/tls/"})
        return True
    rest = path[len("/tls/"):]
    q = {k: v[-1] for k, v in (query or {}).items()}
    try:
        if rest in FILES:
            name, kind = FILES[rest]
            send(200, (STATIC / name).read_bytes(), kind)
        elif rest.startswith("api/"):
            _api(send, rest[4:], q)
        else:
            send(404, b"not found", "text/plain")
    except Exception as e:  # noqa: BLE001 (said to the page, and on the console)
        traceback.print_exc()
        send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))
    return True


def _json(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), default=str).encode()


def _api(send, name: str, q: dict) -> None:
    lib = library()
    fail = lambda code, msg: send(code, _json({"error": msg}))  # noqa: E731
    if name == "captures":
        infos = [e.info() for e in lib.rescan().values()]
        send(200, _json(sorted(infos, key=lambda i: (not i["keylogs"], -i["mtime"]))))
        return
    local = {"1": True, "0": False}.get(q.get("local", ""))
    cap = lib.open(q.get("cap", ""), local, reload=name == "open" and q.get("reload") == "1")
    if cap is None:
        return fail(404, "unknown capture")
    if name == "open":
        return send(200, _json({"state": cap.state, "progress": cap.progress, "error": cap.error, "local": cap.local,
                                "stale": cap.stale()}))
    if cap.state != "ready":
        return fail(409, f"capture not ready: {cap.error or cap.progress}")
    if name == "summary":
        send(200, _json({"summary": cap.summary, "conns": cap.conns, "items": [cap.item_summary(it) for it in cap.items]}))
    elif name == "item":
        it = cap.find_item(q.get("id", ""))
        send(200, _json(cap.item_detail(it))) if it else fail(404, "no such item")
    elif name == "conn":
        d = cap.conn_detail(_int(q.get("n")))
        send(200, _json(d)) if d else fail(404, "no such connection")
    elif name == "body":
        it = cap.find_item(q.get("id", ""))
        b = cap.body(it, q.get("part", "req"), q.get("form", "decoded")) if it else None
        if b is None:
            return fail(404, "no such body")
        extra = {}
        if q.get("dl"):
            fn = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{cap.entry.id}_{it['id']}_{q.get('part', 'msg')}.{q.get('ext', 'bin')}")
            extra["Content-Disposition"] = f'attachment; filename="{fn}"'
        send(200, bytes(b), "application/octet-stream", extra)
    elif name == "record":
        try:
            b = cap.record_pt(_int(q.get("n")), q.get("dir"), _int(q.get("i")))
        except (KeyError, IndexError, TypeError):
            return fail(404, "no such record")
        send(200, bytes(b or b""), "application/octet-stream")
    elif name == "stream":
        try:
            app = cap.data[_int(q.get("n"))]["r"][q.get("dir")]["app"]
        except (KeyError, TypeError):
            return fail(404, "no such stream")
        off = max(_int(q.get("off"), 0), 0)
        ln = min(max(_int(q.get("len"), 65536), 0), STREAM_MAX)
        send(200, bytes(app[off:off + ln]), "application/octet-stream", {"X-Total-Length": str(len(app))})
    elif name == "search":
        send(200, _json({"ids": cap.search(q.get("q", ""))}))
    elif name == "csauth":
        if getattr(cap, "csauth", None) is None:
            from .csauth import timeline
            cap.csauth = _json(timeline(cap))
        send(200, cap.csauth)
    else:
        fail(404, "not found")


def _int(s, default=None):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default
