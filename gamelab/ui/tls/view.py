"""The TLS connections of the recorder's captures, decrypted with their key logs, made ready for the /tls/ page
(ui.tls): every connection (decrypted or why not, its certificates, its TLS records and each record's plaintext, the
decrypted streams), every HTTP/1.x exchange, HTTP/2 stream (HPACK decoded here) and WebSocket message, with the bodies
decoded (gzip / deflate / zstd, the NGS servers' `CE: G` responses, the NGS `AE: G` request bodies (ui.tls.ngs),
brotli if the `brotli` module is installed). Bytes the capture lost (the router's zero-filled frames) are flagged, as
ui.tls.decrypt does.

A capture is a .pcap under the root (ROOT in ui.tls), with the keys*.log beside it (merged); it's decrypted in memory
when the page opens it (a few seconds for 90 MB), nothing written. Frames over 1500 B: decrypt.frame_sources' rules
(tzsp.raw / summary.json beside the pcap), except that a capture in a folder named `local` is taken as captured on
this PC (decrypt.py's --local) unless the page says otherwise.

Built in D:/log decoding as .tools/tls_web.py, a server of its own (the user, 2026-11-13: "build a webui for me to
view the captured and decrypted tls packets"), and moved here, onto the mini-map's server and its tunnel ("this should
probably go to the maplehelper project as we already have a webui setup there with cloudflare tunnels")."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import traceback
import zlib
from types import SimpleNamespace

from . import decrypt as td

try:
    from . import ngs
except Exception:                                         # pragma: no cover
    ngs = None
try:
    import brotli                                         # optional
except Exception:
    brotli = None
try:
    from cryptography import x509
except Exception:                                         # pragma: no cover
    x509 = None

CT_NAMES = {20: "ChangeCipherSpec", 21: "Alert", 22: "Handshake", 23: "ApplicationData", 24: "Heartbeat"}
HS_NAMES = {0: "HelloRequest", 1: "ClientHello", 2: "ServerHello", 4: "NewSessionTicket", 5: "EndOfEarlyData",
            8: "EncryptedExtensions", 11: "Certificate", 12: "ServerKeyExchange", 13: "CertificateRequest",
            14: "ServerHelloDone", 15: "CertificateVerify", 16: "ClientKeyExchange", 20: "Finished", 24: "KeyUpdate",
            254: "MessageHash"}
WS_OPS = {0: "cont", 1: "text", 2: "binary", 8: "close", 9: "ping", 10: "pong"}
H2_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
H2_ERRORS = {0: "NO_ERROR", 1: "PROTOCOL_ERROR", 2: "INTERNAL_ERROR", 3: "FLOW_CONTROL_ERROR", 4: "SETTINGS_TIMEOUT",
             5: "STREAM_CLOSED", 6: "FRAME_SIZE_ERROR", 7: "REFUSED_STREAM", 8: "CANCEL", 9: "COMPRESSION_ERROR",
             10: "CONNECT_ERROR", 11: "ENHANCE_YOUR_CALM", 12: "INADEQUATE_SECURITY", 13: "HTTP_1_1_REQUIRED"}
H2_SETTINGS = {1: "HEADER_TABLE_SIZE", 2: "ENABLE_PUSH", 3: "MAX_CONCURRENT_STREAMS", 4: "INITIAL_WINDOW_SIZE",
               5: "MAX_FRAME_SIZE", 6: "MAX_HEADER_LIST_SIZE", 8: "ENABLE_CONNECT_PROTOCOL"}
TLS_NAMES = {"303": "1.2", "302": "1.1", "301": "1.0"}
HTTP1_START = re.compile(rb"^[A-Z][A-Z-]{1,15} \S+ HTTP/1\.[01]\r\n")


# ------------------------------------------------------------------------------------------------------------ HPACK

HP_STATIC = [(":authority", ""), (":method", "GET"), (":method", "POST"), (":path", "/"), (":path", "/index.html"),
             (":scheme", "http"), (":scheme", "https"), (":status", "200"), (":status", "204"), (":status", "206"),
             (":status", "304"), (":status", "400"), (":status", "404"), (":status", "500"), ("accept-charset", ""),
             ("accept-encoding", "gzip, deflate"), ("accept-language", ""), ("accept-ranges", ""), ("accept", ""),
             ("access-control-allow-origin", ""), ("age", ""), ("allow", ""), ("authorization", ""),
             ("cache-control", ""), ("content-disposition", ""), ("content-encoding", ""), ("content-language", ""),
             ("content-length", ""), ("content-location", ""), ("content-range", ""), ("content-type", ""),
             ("cookie", ""), ("date", ""), ("etag", ""), ("expect", ""), ("expires", ""), ("from", ""), ("host", ""),
             ("if-match", ""), ("if-modified-since", ""), ("if-none-match", ""), ("if-range", ""),
             ("if-unmodified-since", ""), ("last-modified", ""), ("link", ""), ("location", ""), ("max-forwards", ""),
             ("proxy-authenticate", ""), ("proxy-authorization", ""), ("range", ""), ("referer", ""), ("refresh", ""),
             ("retry-after", ""), ("server", ""), ("set-cookie", ""), ("strict-transport-security", ""),
             ("transfer-encoding", ""), ("user-agent", ""), ("vary", ""), ("via", ""), ("www-authenticate", "")]

# RFC 7541 appendix B is a canonical Huffman code: the code length of every symbol fixes it.
HUFF_LENS = {
    5: b"012aceiost", 6: b" %-./3456789=A_bdfghlmnpru", 7: b":BCDEFGHIJKLMNOPQRSTUVWYjkqvwxyz", 8: b"&*,;XZ",
    10: b'!"()?', 11: b"'+|", 12: b"#>", 13: b"\x00$@[]~", 14: b"^}", 15: b"<`{",
    19: [92, 195, 208], 20: [128, 130, 131, 162, 184, 194, 224, 226],
    21: [153, 161, 167, 172, 176, 177, 179, 209, 216, 217, 227, 229, 230],
    22: [129, 132, 133, 134, 136, 146, 154, 156, 160, 163, 164, 169, 170, 173, 178, 181, 185, 186, 187, 189, 190, 196,
         198, 228, 232, 233],
    23: [1, 135, 137, 138, 139, 140, 141, 143, 147, 149, 150, 151, 152, 155, 157, 158, 165, 166, 168, 174, 175, 180, 182,
         183, 188, 191, 197, 231, 239],
    24: [9, 142, 144, 145, 148, 159, 171, 206, 215, 225, 236, 237], 25: [199, 207, 234, 235],
    26: [192, 193, 200, 201, 202, 205, 210, 213, 218, 219, 238, 240, 242, 243, 255],
    27: [203, 204, 211, 212, 214, 221, 222, 223, 241, 244, 245, 246, 247, 248, 250, 251, 252, 253, 254],
    28: [2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 127, 220, 249],
    30: [10, 13, 22, 256]}


def _huff_decoder():
    syms = sorted((ln, s) for ln, ss in HUFF_LENS.items() for s in ss)
    assert sorted(s for _l, s in syms) == list(range(257)), "HPACK Huffman table: every symbol exactly once"
    assert sum(2.0 ** -ln for ln, _s in syms) == 1.0, "HPACK Huffman table: complete code"
    table, code, prev = {}, 0, syms[0][0]
    for ln, s in syms:
        code <<= ln - prev
        prev = ln
        table[(ln, code)] = s
        code += 1
    return table


HUFF = _huff_decoder()


def huff_decode(data):
    out, code, ln = bytearray(), 0, 0
    for byte in data:
        for i in range(7, -1, -1):
            code = (code << 1) | ((byte >> i) & 1)
            ln += 1
            s = HUFF.get((ln, code))
            if s is not None:
                if s == 256:
                    raise ValueError("EOS inside a Huffman string")
                out.append(s)
                code = ln = 0
            elif ln > 30:
                raise ValueError("bad Huffman code")
    if ln > 7 or code != (1 << ln) - 1:
        raise ValueError("bad Huffman padding")
    return bytes(out)


def hp_int(b, o, n):
    mask = (1 << n) - 1
    v = b[o] & mask
    o += 1
    if v < mask:
        return v, o
    m = 0
    while True:
        c = b[o]
        o += 1
        v += (c & 127) << m
        m += 7
        if not c & 128:
            return v, o


def hp_str(b, o):
    huff = b[o] & 0x80
    ln, o = hp_int(b, o, 7)
    s = bytes(b[o:o + ln])
    if len(s) < ln:
        raise ValueError("truncated string")
    return (huff_decode(s) if huff else s).decode("latin-1"), o + ln


class Hpack:
    """HPACK decoder of one direction.  Names and values are kept as latin-1 strings (one char per octet: the table
    sizes are octet counts); show() turns them into display text."""

    def __init__(self):
        self.dyn, self.size, self.max = [], 0, 4096

    def _get(self, i):
        if 1 <= i <= len(HP_STATIC):
            return HP_STATIC[i - 1]
        j = i - len(HP_STATIC) - 1
        if 0 <= j < len(self.dyn):
            return self.dyn[j]
        raise ValueError("HPACK index %d not in the tables" % i)

    def _evict(self):
        while self.size > self.max and self.dyn:
            n, v = self.dyn.pop()
            self.size -= len(n) + len(v) + 32

    def decode(self, b):
        out, o = [], 0
        while o < len(b):
            c = b[o]
            if c & 0x80:                                         # indexed
                i, o = hp_int(b, o, 7)
                out.append(self._get(i))
                continue
            if c & 0x20 and not c & 0x40:                        # dynamic table size update
                self.max, o = hp_int(b, o, 5)
                self._evict()
                continue
            index = c & 0x40                                     # literal with incremental indexing (else without / never)
            i, o = hp_int(b, o, 6 if index else 4)
            if i:
                n = self._get(i)[0]
            else:
                n, o = hp_str(b, o)
            v, o = hp_str(b, o)
            out.append((n, v))
            if index:
                self.dyn.insert(0, (n, v))
                self.size += len(n) + len(v) + 32
                self._evict()
        return [(show(n), show(v)) for n, v in out]


def show(s):
    try:
        return s.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return s


# ------------------------------------------------------------------------------------------------------------ helpers

def hget(headers, name):
    name = name.lower()
    for n, v in headers:
        if n.lower() == name:
            return v
    return ""


def lost_in(ranges, a, b):
    return sum(y - x for x, y in td.overlap(ranges, a, b))


def display_kind(ctype, body):
    if not body:
        return "empty"
    ct = ctype.lower()
    if ct.startswith("image/") and "svg" not in ct:
        return "image"
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return "binary"
    if sum(ch.isprintable() or ch in "\r\n\t\0\x1e" for ch in text) < 0.97 * len(text):
        return "binary"
    try:
        json.loads(text.strip().rstrip("\x1e"))
        return "json"
    except ValueError:
        return "text"


def decode_content(headers, wire):
    """(decoded body, how) for an HTTP body (Content-Encoding / NGS CE: G)."""
    if not wire:
        return wire, None
    if hget(headers, "content-encoding").lower().strip() == "br":
        if not brotli:
            return wire, "br (not decoded: the brotli module is not installed)"
        try:
            return brotli.decompress(wire), "br"
        except Exception as e:
            return wire, "br (decode failed: %s)" % e
    return td.decode_body(headers, wire)


B64_BODY = re.compile(rb"[A-Za-z0-9+/=\r\n]{8,}")


def decode_ngs_request(headers, wire, host):
    """The NGS AE: G request cipher (x-*.ngs.nexon.com): (plaintext, how) or None."""
    if ngs is None or not wire or not (host.endswith("ngs.nexon.com") or hget(headers, "ae")):
        return None
    w = wire.strip()
    if not B64_BODY.fullmatch(w):
        return None
    try:
        res = ngs.decode_body(w)
    except Exception:
        return None
    if not res.is_json:
        return None
    return res.plaintext, "NGS AE:G request cipher, seed k0=0x%04x (%s)" % (res.seed, res.method)


class Part:
    """One HTTP message (or one WebSocket message)."""

    def __init__(self, start, headers, wire, ts, lost=0, complete=True, host="", request=False, body=None, how=None):
        self.start, self.headers, self.wire, self.ts = start, headers, wire, ts
        self.lost, self.complete = lost, complete
        if body is None:
            body, how = decode_content(headers, wire)
            if request:
                r = decode_ngs_request(headers, wire, host)
                if r:
                    body, how = r
        self.body, self.how = body, how
        self.ctype = hget(headers, "content-type")
        self.display = display_kind(self.ctype, body)

    def summary(self):
        return {"size": len(self.body), "wire": len(self.wire), "ctype": self.ctype, "how": self.how, "lost": self.lost,
                "complete": self.complete, "display": self.display, "ts": self.ts}

    def detail(self):
        d = self.summary()
        d.update(start=self.start, headers=[[n, v] for n, v in self.headers])
        return d


# ------------------------------------------------------------------------------------------------------------ HTTP/1, WS

def ws_extension(headers):
    ext = hget(headers, "sec-websocket-extensions").lower()
    if "permessage-deflate" not in ext:
        return None
    return {"c2s_reset": "client_no_context_takeover" in ext, "s2c_reset": "server_no_context_takeover" in ext}


def ws_messages(buf, start, marks, lost, dname, deflate, host, path, notes):
    """WebSocket messages of one direction from `start`."""
    out, o, frag = [], start, None
    inflater = zlib.decompressobj(-15)
    while o + 2 <= len(buf):
        b0, b1 = buf[o], buf[o + 1]
        fin, rsv1, op = b0 >> 7, (b0 >> 6) & 1, b0 & 15
        ln, p = b1 & 127, o + 2
        if ln == 126:
            ln, p = int.from_bytes(buf[p:p + 2], "big"), p + 2
        elif ln == 127:
            ln, p = int.from_bytes(buf[p:p + 8], "big"), p + 8
        key = None
        if b1 & 0x80:
            key, p = buf[p:p + 4], p + 4
        payload = buf[p:p + ln]
        complete = p + ln <= len(buf)
        if key and len(key) == 4:
            payload = bytes(x ^ key[i & 3] for i, x in enumerate(payload))
        if op == 0 and frag is not None:
            frag["data"] += payload
            frag["end"] = p + ln
        elif op >= 8:
            out.append({"op": op, "data": payload, "off": o, "end": p + ln, "rsv1": False, "complete": complete})
        else:
            frag = {"op": op, "data": bytearray(payload), "off": o, "end": p + ln, "rsv1": bool(rsv1),
                    "complete": complete}
            if op == 0:
                notes.append("%s WebSocket continuation frame without a start at %d" % (dname, o))
        if fin and frag is not None and op < 8:
            out.append(frag)
            frag = None
        o = p + ln
    if frag is not None:
        frag["complete"] = False
        out.append(frag)
    msgs = []
    for m in out:
        data, how = bytes(m["data"]), None
        if m["rsv1"] and deflate:
            try:
                if deflate["%s_reset" % dname]:
                    inflater = zlib.decompressobj(-15)
                data, how = inflater.decompress(data + b"\x00\x00\xff\xff"), "permessage-deflate"
            except zlib.error as e:
                how = "permessage-deflate (failed: %s)" % e
        ctype = "text/plain" if m["op"] == 1 else ""
        part = Part("WebSocket %s" % WS_OPS.get(m["op"], m["op"]), [], bytes(m["data"]), td.ts_at(marks, m["off"]),
                    lost=lost_in(lost, m["off"], m["end"]), complete=m["complete"], body=data, how=how)
        part.ctype = ctype
        part.display = display_kind(ctype, data) if m["op"] in (1, 2) else ("text" if data else "empty")
        if m["op"] == 8 and len(data) >= 2:
            part.body = ("close %d %s" % (int.from_bytes(data[:2], "big"), data[2:].decode("utf-8", "replace"))).encode()
            part.display = "text"
        msgs.append((m, part))
    return msgs


def http1_items(conn, d, host):
    capp, cmarks, clost = d["c2s"]["app"], d["c2s"]["marks"], d["c2s"]["lost"]
    sapp, smarks, slost = d["s2c"]["app"], d["s2c"]["marks"], d["s2c"]["lost"]
    reqs = td.parse_http1(capp, True)
    ws_c = ws_s = None
    for i, q in enumerate(reqs):
        if hget(q["headers"], "upgrade").lower() == "websocket":
            reqs, ws_c = reqs[:i + 1], q["end"]
            break
    resps = td.parse_http1(sapp, False, [q["start"].split(" ", 1)[0] for q in reqs])
    for i, s in enumerate(resps):
        if s["status"] == 101:
            resps, ws_s = resps[:i + 1], s["end"]
            break
    final = [s for s in resps if not (s["status"] and 100 <= s["status"] < 200 and s["status"] != 101)]
    items = []
    for m, q in enumerate(reqs):
        s = final[m] if m < len(final) else None
        method, _, rest = q["start"].partition(" ")
        path = rest.rsplit(" ", 1)[0] if rest.endswith(("HTTP/1.1", "HTTP/1.0")) else rest
        qhost = hget(q["headers"], "host") or host
        req = Part(q["start"], q["headers"], q["body"], td.ts_at(cmarks, q["off"]), lost_in(clost, q["off"], q["end"]),
                   q["complete"], qhost, request=True)
        resp = None
        if s is not None:
            resp = Part(s["start"], s["headers"], s["body"], td.ts_at(smarks, s["off"]),
                        lost_in(slost, s["off"], s["end"]), s["complete"])
            resp.ts_end = td.ts_at(smarks, max(s["end"] - 1, s["off"]))
        items.append({"type": "http", "proto": "http/1.1", "method": method if q["complete"] or rest else "?",
                      "path": path, "host": qhost, "status": s["status"] if s else None,
                      "reason": s["start"].split(" ", 2)[2] if s and s["start"].count(" ") >= 2 else "",
                      "req": req, "resp": resp})
    if ws_c is not None and ws_s is not None:
        up = items[-1]
        deflate = ws_extension(up["resp"].headers) if up.get("resp") else None
        notes = conn["notes"]
        msgs = [("c2s", m, p) for m, p in ws_messages(capp, ws_c, cmarks, clost, "c2s", deflate, host, up["path"], notes)]
        msgs += [("s2c", m, p) for m, p in ws_messages(sapp, ws_s, smarks, slost, "s2c", deflate, host, up["path"], notes)]
        msgs.sort(key=lambda x: (x[2].ts, x[0]))
        for dname, m, p in msgs:
            items.append({"type": "ws", "proto": "websocket", "dir": dname, "op": WS_OPS.get(m["op"], str(m["op"])),
                          "method": "WS " + ("\u2191" if dname == "c2s" else "\u2193"), "path": up["path"],
                          "host": up["host"], "status": None, "reason": "", "msg": p})
    return items


# ------------------------------------------------------------------------------------------------------------ HTTP/2

def h2_items(conn, d, host, frames_out):
    streams, notes = {}, conn["notes"]

    def stream(sid):
        return streams.setdefault(sid, {"sid": sid, "req": None, "resp": None, "interim": [], "trailers": {},
                                        "data": {"c2s": bytearray(), "s2c": bytearray()}, "ts": {}, "end_ts": {},
                                        "lost": {"c2s": 0, "s2c": 0}, "rst": None, "pushed": False})

    for dname in ("c2s", "s2c"):
        buf, marks, lost = d[dname]["app"], d[dname]["marks"], d[dname]["lost"]
        hp, broken = Hpack(), False
        o = len(H2_PREFACE) if buf.startswith(H2_PREFACE) else 0
        pending = None
        while o + 9 <= len(buf):
            ln = int.from_bytes(buf[o:o + 3], "big")
            t, fl, sid = buf[o + 3], buf[o + 4], int.from_bytes(buf[o + 5:o + 9], "big") & 0x7FFFFFFF
            p = buf[o + 9:o + 9 + ln]
            ts = td.ts_at(marks, o)
            fr = {"dir": dname, "off": o, "ts": ts, "type": td.H2_TYPES.get(t, str(t)), "flags": fl, "sid": sid, "len": ln,
                  "info": "", "lost": lost_in(lost, o, o + 9 + ln)}
            frames_out.append(fr)
            if o + 9 + ln > len(buf):
                fr["info"] = "truncated (%d of %d bytes in the capture)" % (len(buf) - o - 9, ln)
            if t == 0:                                               # DATA
                data = p
                if fl & 8 and data:
                    data = data[1:len(data) - data[0]]
                s = stream(sid)
                s["data"][dname] += data
                s["lost"][dname] += fr["lost"]
                s["end_ts"][dname] = ts
                fr["info"] = "%d bytes%s" % (len(data), ", END_STREAM" if fl & 1 else "")
            elif t in (1, 5):                                        # HEADERS / PUSH_PROMISE
                q, pad, promised = 0, 0, None
                if fl & 8 and p:
                    pad, q = p[0], 1
                if t == 1 and fl & 0x20:
                    q += 5
                if t == 5:
                    promised, q = int.from_bytes(p[q:q + 4], "big") & 0x7FFFFFFF, q + 4
                pending = [sid, bytearray(p[q:len(p) - pad]), t, fl, ts, promised, fr]
            elif t == 9 and pending is not None:                     # CONTINUATION
                pending[1] += p
                pending[3] |= fl & 4
                pending[6] = fr
            elif t == 3:                                             # RST_STREAM
                code = int.from_bytes(p[:4], "big")
                stream(sid)["rst"] = H2_ERRORS.get(code, str(code))
                fr["info"] = H2_ERRORS.get(code, str(code))
            elif t == 7:                                             # GOAWAY
                code = int.from_bytes(p[4:8], "big")
                fr["info"] = "last stream %d, %s%s" % (int.from_bytes(p[:4], "big") & 0x7FFFFFFF,
                                                       H2_ERRORS.get(code, str(code)),
                                                       (" " + p[8:].decode("latin-1")) if len(p) > 8 else "")
            elif t == 4:                                             # SETTINGS
                fr["info"] = "ACK" if fl & 1 else ", ".join(
                    "%s=%d" % (H2_SETTINGS.get(int.from_bytes(p[i:i + 2], "big"), int.from_bytes(p[i:i + 2], "big")),
                               int.from_bytes(p[i + 2:i + 6], "big")) for i in range(0, len(p) - 5, 6))
            elif t == 8:                                             # WINDOW_UPDATE
                fr["info"] = "+%d" % (int.from_bytes(p[:4], "big") & 0x7FFFFFFF)
            elif t == 6:                                             # PING
                fr["info"] = ("ACK " if fl & 1 else "") + p.hex()
            if pending is not None and (t in (1, 5, 9)) and pending[3] & 4:
                psid, block, pt, pfl, pts, promised, pfr = pending
                pending = None
                if broken:
                    pfr["info"] = "header block not decoded (an earlier HPACK error)"
                else:
                    try:
                        hdrs = hp.decode(bytes(block))
                    except Exception as e:
                        broken = True
                        notes.append("%s HPACK decode failed at %d: %s (later header blocks of this direction are not "
                                     "decoded)" % (dname, o, e))
                        pfr["info"] = "HPACK error: %s" % e
                        hdrs = None
                    if hdrs is not None:
                        pfr["info"] = " ".join(v for n, v in hdrs if n in (":method", ":path", ":status"))
                        if pt == 5:
                            ps = stream(promised)
                            ps["req"], ps["pushed"], ps["ts"]["c2s"] = hdrs, True, pts
                            pfr["info"] = "promised stream %d: %s" % (promised, pfr["info"])
                        else:
                            s = stream(psid)
                            status = hget(hdrs, ":status")
                            if dname == "c2s":
                                if s["req"] is None:
                                    s["req"], s["ts"]["c2s"] = hdrs, pts
                                else:
                                    s["trailers"]["c2s"] = hdrs
                            elif status.startswith("1"):
                                s["interim"].append(hdrs)
                            elif s["resp"] is None:
                                s["resp"], s["ts"]["s2c"] = hdrs, pts
                            else:
                                s["trailers"]["s2c"] = hdrs
                            s["end_ts"].setdefault(dname, pts)
            o += 9 + ln
    items = []
    for sid in sorted(streams):
        s = streams[sid]
        if sid == 0 or (s["req"] is None and s["resp"] is None):
            continue
        rq = s["req"] or []
        method, path = hget(rq, ":method") or "?", hget(rq, ":path")
        shost = hget(rq, ":authority") or host
        start = "%s %s HTTP/2  (stream %d%s)" % (method, path, sid, ", pushed" if s["pushed"] else "")
        req = Part(start, rq + s["trailers"].get("c2s", []), bytes(s["data"]["c2s"]),
                   s["ts"].get("c2s", s["end_ts"].get("c2s", 0.0)), s["lost"]["c2s"], s["req"] is not None, shost,
                   request=True)
        resp = None
        if s["resp"] is not None:
            status = hget(s["resp"], ":status")
            resp = Part("HTTP/2 %s" % status, s["resp"] + s["trailers"].get("s2c", []), bytes(s["data"]["s2c"]),
                        s["ts"]["s2c"], s["lost"]["s2c"], s["rst"] is None)
            resp.ts_end = s["end_ts"].get("s2c", s["ts"]["s2c"])
        status = hget(s["resp"] or [], ":status")
        items.append({"type": "http", "proto": "h2", "method": method, "path": path, "host": shost,
                      "status": int(status) if status.isdigit() else None,
                      "reason": ("RST_STREAM " + s["rst"]) if s["rst"] else "", "req": req, "resp": resp, "sid": sid})
    items.sort(key=lambda it: it["req"].ts)
    return items


# ------------------------------------------------------------------------------------------------------------ TLS details

def parse_certs(body, tls13):
    if x509 is None:
        return []
    o = 1 + body[0] if tls13 else 0
    end = o + 3 + int.from_bytes(body[o:o + 3], "big")
    o += 3
    out = []
    while o + 3 <= min(end, len(body)):
        ln = int.from_bytes(body[o:o + 3], "big")
        der, o = body[o + 3:o + 3 + ln], o + 3 + ln
        if tls13:
            o += 2 + int.from_bytes(body[o:o + 2], "big")
        try:
            c = x509.load_der_x509_certificate(der)
            try:
                san = c.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(
                    x509.DNSName)
            except Exception:
                san = []
            out.append({"subject": c.subject.rfc4514_string(), "issuer": c.issuer.rfc4514_string(),
                        "not_before": c.not_valid_before_utc.strftime("%Y-%m-%d"),
                        "not_after": c.not_valid_after_utc.strftime("%Y-%m-%d"), "serial": "%x" % c.serial_number,
                        "san": san[:30] + (["... %d more" % (len(san) - 30)] if len(san) > 30 else []),
                        "sha256": hashlib.sha256(der).hexdigest()})
        except Exception as e:
            out.append({"subject": "(certificate not parsed: %s)" % e, "issuer": "", "san": []})
    return out


def label_records(recs, tls13, certs):
    """Fill rec["info"] (handshake message names, alerts) of one direction; collect the certificate chain."""
    hsbuf = bytearray()
    for rec in recs:
        ct = rec["inner"] if rec["status"] in ("ok", "unauth") else rec["ct"]
        rec["info"] = ""
        if rec["status"] in ("failed", "nokey"):
            rec["info"] = "not decrypted"
            continue
        pt = rec["pt"]
        if ct == 22 and pt is not None:
            cont = bool(hsbuf)
            hsbuf += pt
            names = []
            for t, body in td.hs_messages(hsbuf):
                names.append(HS_NAMES.get(t, "type %d" % t))
                if t == 11 and not certs:
                    try:
                        certs.extend(parse_certs(body, tls13))
                    except Exception as e:
                        certs.append({"subject": "(certificate list not parsed: %s)" % e, "issuer": "", "san": []})
            rec["info"] = ("(cont.) " if cont else "") + ", ".join(names) + (" (continues)" if hsbuf else "")
        elif ct == 21 and pt is not None and len(pt) >= 2:
            rec["info"] = "%s %s" % ({1: "warning", 2: "fatal"}.get(pt[0], pt[0]), td.ALERTS.get(pt[1], pt[1]))
        elif ct == 20:
            rec["info"] = "ChangeCipherSpec"


# ------------------------------------------------------------------------------------------------------------ captures

class Entry:
    def __init__(self, cid, name, pcap, keylogs, local_default):
        self.id, self.name, self.pcap, self.keylogs, self.local_default = cid, name, pcap, keylogs, local_default

    def stamp(self):
        return tuple((p, os.path.getsize(p), os.path.getmtime(p)) for p in [self.pcap] + self.keylogs if os.path.exists(p))

    def info(self):
        st = os.stat(self.pcap)
        return {"id": self.id, "name": self.name, "pcap": self.pcap, "size": st.st_size, "mtime": st.st_mtime,
                "keylogs": [os.path.basename(k) for k in self.keylogs], "local_default": self.local_default}


def discover(paths, extra_keylogs):
    """{id: Entry} of the .pcap files in `paths` (folders, searched 4 levels down past the recorder's own output
    folders, or files); an id is the pcap's path hashed, so it stays the same across restarts."""
    found = {}
    for root in paths:
        root = os.path.abspath(root)
        if os.path.isfile(root):
            files, base = [root], os.path.dirname(os.path.dirname(root))
        else:
            files, base = [], root
            for dp, dns, fns in os.walk(root):
                if dp[len(root):].count(os.sep) >= 4:
                    dns[:] = []
                dns[:] = [x for x in dns if x not in ("tls", "streams", "bodies", "replay")]
                files += [os.path.join(dp, f) for f in fns if f.lower().endswith(".pcap")]
        for f in files:
            d = os.path.dirname(f)
            kl = sorted(os.path.join(d, x) for x in os.listdir(d)
                        if re.match(r"(ssl)?keys.*\.log$", x, re.I) or x.lower().endswith(".keylog"))
            kl += [os.path.abspath(k) for k in extra_keylogs if os.path.abspath(k) not in kl]
            cid = hashlib.sha1(os.path.normcase(f).encode()).hexdigest()[:10]
            name = os.path.relpath(f, base).replace(os.sep, "/")
            local = os.path.basename(os.path.dirname(d)).lower() == "local"
            found[cid] = Entry(cid, name, f, kl, local)
    return found


class Capture:
    """A decrypted capture, held in memory."""

    def __init__(self, entry, local, extra_keylogs=()):
        self.entry, self.local, self.extra_keylogs = entry, local, list(extra_keylogs)
        self.state, self.progress, self.error = "loading", "starting", None
        self.stamp = entry.stamp()
        self.summary, self.conns, self.items, self.data = None, [], [], {}

    def stale(self):
        """True once its pcap or key logs changed, or a key log came, since it was decrypted (a recording still going,
        a key log copied in at its end): the page offers to decrypt it again."""
        if self.state != "ready":
            return False
        now = discover([self.entry.pcap], self.extra_keylogs).get(self.entry.id)
        return now is None or now.keylogs != self.entry.keylogs or self.stamp != self.entry.stamp()

    def run(self):
        try:
            self._run()
            self.state = "ready"
        except Exception as e:
            traceback.print_exc()
            self.state, self.error = "error", "%s: %s" % (type(e).__name__, e)

    def _run(self):
        t0 = time.time()
        e = self.entry
        self.progress = "reading %s" % os.path.basename(e.pcap)
        conns, sources = td.frame_sources(SimpleNamespace(pcap=e.pcap, local=self.local, tzsp_raw=None))
        keys = td.load_keylogs(e.keylogs)
        used, non_tls, hosts = set(), 0, {}
        for i, c in enumerate(conns):
            if i % 25 == 0:
                self.progress = "decrypting connection %d of %d" % (i + 1, len(conns))
            r = td.decrypt_conn(c, keys, with_records=True)
            if r is None:
                non_tls += 1
                continue
            if r["keyed"]:
                used.add(r["ch"]["random"])
            self._conn(c, r, hosts)
        self.items.sort(key=lambda it: (it["ts"], it["conn"]))
        conns_json = self.conns
        dec = [x for x in conns_json if x["decrypted"]]
        ts = [x["start"] for x in conns_json] + [x["end"] for x in conns_json]
        self.summary = {
            "capture": e.info(), "local": self.local, "sources": sources,
            "keylogs": e.keylogs, "key_randoms": len(keys), "key_used": len(used),
            "tcp_conns": len(conns), "non_tls": non_tls, "tls_conns": len(conns_json), "decrypted": len(dec),
            "records": sum(x["records"][0] + x["records"][1] for x in dec), "failed": sum(x["failed"] for x in dec),
            "unauth": sum(x["unauth"] for x in dec), "lost": sum(x["lost"][0] + x["lost"][1] for x in dec),
            "app_bytes": sum(x["app"][0] + x["app"][1] for x in dec),
            "http": sum(1 for it in self.items if it["type"] == "http"),
            "ws": sum(1 for it in self.items if it["type"] == "ws"),
            "first": min(ts) if ts else 0, "last": max(ts) if ts else 0,
            "hosts": sorted(([h, n] for h, n in hosts.items()), key=lambda x: (-x[1], x[0])),
            "seconds": round(time.time() - t0, 2), "ngs_decoder": ngs is not None, "brotli": brotli is not None}

    def _conn(self, c, r, hosts):
        ch, sh = r["ch"], r["sh"]
        n = c.n
        decrypted = r["c2s"] is not None
        sni = ch["sni"] or ""
        conn = {"n": n, "start": c.first_ts, "end": c.last_ts, "client": "%s:%d" % c.client,
                "server": "%s:%d" % c.server, "sni": sni, "tls": TLS_NAMES.get(r["tls"], r["tls"]), "suite": r["suite"],
                "alpn_offered": ch["alpn"], "alpn": r["alpn"] or "", "key": r["key"], "keyed": r["keyed"],
                "decrypted": decrypted, "notes": r["notes"], "records": [0, 0], "failed": 0, "unauth": 0,
                "app": [0, 0], "lost": [0, 0], "proto": "", "items": 0, "alerts": [], "certs": []}
        if not sh:
            conn["status"] = "no ServerHello"
        elif not r["keyed"]:
            conn["status"] = "no key"
        elif not decrypted:
            conn["status"] = "unsupported suite"
        data = {"r": r, "frames": [], "items": []}
        if decrypted:
            ci, si = r["c2s"]["info"], r["s2c"]["info"]
            conn["records"] = [ci["records"], si["records"]]
            conn["failed"] = ci["failed"] + si["failed"]
            conn["unauth"] = ci["unauth"] + si["unauth"]
            conn["app"] = [len(r["c2s"]["app"]), len(r["s2c"]["app"])]
            conn["lost"] = [sum(b - a for a, b in r["c2s"]["lost"]), sum(b - a for a, b in r["s2c"]["lost"])]
            conn["alerts"] = ["c2s " + x for x in ci["alerts"]] + ["s2c " + x for x in si["alerts"]]
            certs = []
            label_records(r["c2s"]["records"], r["tls"] == "1.3", [])
            label_records(r["s2c"]["records"], r["tls"] == "1.3", certs)
            conn["certs"] = certs
            conn["status"] = "decrypted" if not (conn["failed"] or sum(conn["lost"])) else "damaged"
            capp = r["c2s"]["app"]
            host = sni or c.server[0]
            items = []
            try:
                if capp.startswith(H2_PREFACE) or r["alpn"] == "h2":
                    conn["proto"] = "h2"
                    items = h2_items(conn, r, host, data["frames"])
                    data["frames"].sort(key=lambda f: (f["ts"], f["dir"]))
                elif HTTP1_START.match(capp):
                    conn["proto"] = "http/1.1"
                    items = http1_items(conn, r, host)
                    if any(it["type"] == "ws" for it in items):
                        conn["proto"] = "http/1.1 + websocket"
                elif capp or r["s2c"]["app"]:
                    conn["proto"] = "raw"
            except Exception as e:
                traceback.print_exc()
                conn["notes"].append("HTTP parse failed: %s: %s" % (type(e).__name__, e))
            for k, it in enumerate(items):
                it["id"] = "%d.%d" % (n, k)
                it["conn"] = n
                p0 = it.get("req") or it.get("msg")
                it["ts"] = p0.ts
                hosts[it["host"]] = hosts.get(it["host"], 0) + 1
            data["items"] = items
            conn["items"] = len(items)
            self.items += items
        self.conns.append(conn)
        self.data[n] = data

    # --------------------------------------------------------------- views

    def item_summary(self, it):
        d = {k: it.get(k) for k in ("id", "conn", "type", "proto", "method", "path", "host", "status", "reason", "ts",
                                     "dir", "op", "sid")}
        if it["type"] == "http":
            d["req"] = it["req"].summary()
            d["resp"] = it["resp"].summary() if it["resp"] else None
            if it["resp"] is not None:
                d["te"] = getattr(it["resp"], "ts_end", it["resp"].ts)
        else:
            m = it["msg"]
            d["msg"] = m.summary()
            if m.display in ("text", "json") and m.body:
                d["preview"] = m.body[:160].decode("utf-8", "replace")
        return d

    def find_item(self, iid):
        try:
            n, k = (int(x) for x in iid.split("."))
            return self.data[n]["items"][k]
        except (ValueError, KeyError, IndexError):
            return None

    def item_detail(self, it):
        d = self.item_summary(it)
        if it["type"] == "http":
            d["req"] = it["req"].detail()
            d["resp"] = it["resp"].detail() if it["resp"] else None
        else:
            d["msg"] = it["msg"].detail()
        return d

    def conn_detail(self, n):
        conn = next((c for c in self.conns if c["n"] == n), None)
        if conn is None:
            return None
        data = self.data[n]
        r = data["r"]
        recs = []
        for dname in ("c2s", "s2c"):
            if r[dname] is None:
                continue
            for i, rec in enumerate(r[dname]["records"]):
                recs.append({"dir": dname, "i": i, "off": rec["off"], "ts": rec["ts"], "ct": CT_NAMES.get(rec["ct"], rec["ct"]),
                             "len": rec["len"], "status": rec["status"],
                             "inner": CT_NAMES.get(rec["inner"], rec["inner"]) if rec["inner"] is not None else None,
                             "ptlen": rec["ptlen"], "lost": rec["lost"], "app_off": rec["app_off"],
                             "info": rec.get("info", ""),
                             "has_pt": rec["pt"] is not None or rec["app_off"] is not None})
        recs.sort(key=lambda x: (x["ts"], x["dir"], x["off"]))
        frames = [{k: f[k] for k in ("dir", "off", "ts", "type", "flags", "sid", "len", "info", "lost")}
                  for f in data["frames"]]
        return {"conn": conn, "records": recs, "frames": frames,
                "items": [self.item_summary(it) for it in data["items"]],
                "ch": {"random": r["ch"]["random"].hex(), "sni": r["ch"]["sni"], "alpn": r["ch"]["alpn"]}}

    def body(self, it, part, form):
        if it["type"] == "ws":
            p = it["msg"]
        else:
            p = it.get(part)
        if p is None:
            return None
        return p.wire if form == "wire" else p.body

    def record_pt(self, n, dname, i):
        r = self.data[n]["r"]
        rec = r[dname]["records"][i]
        if rec["app_off"] is not None:
            return r[dname]["app"][rec["app_off"]:rec["app_off"] + rec["ptlen"]]
        return rec["pt"]

    def search(self, q):
        q = q.lower()
        qb = q.encode("utf-8")
        out = []
        for it in self.items:
            parts = [it.get("req"), it.get("resp"), it.get("msg")]
            hay = [(it.get("host") or "") + " " + (it.get("path") or "") + " " + (it.get("method") or "")]
            hit = q in hay[0].lower()
            for p in parts:
                if hit:
                    break
                if p is None:
                    continue
                if q in (p.start + "\n" + "\n".join("%s: %s" % hv for hv in p.headers)).lower():
                    hit = True
                elif qb in p.body.lower() or qb in p.wire.lower():
                    hit = True
            if hit:
                out.append(it["id"])
        return out



# ------------------------------------------------------------------------------------------------------------ library

class Library:
    """The captures under `roots` and the decrypted ones in memory (the 4 most recently used)."""

    def __init__(self, roots, keylogs=()):
        self.roots, self.keylogs = list(roots), list(keylogs)
        self.lock = threading.Lock()
        self.entries = {}
        self.caps = {}                                  # (id, local) -> Capture
        self.order = []

    def rescan(self):
        found = discover([r for r in self.roots if os.path.exists(r)], self.keylogs)
        with self.lock:
            self.entries = found
        return found

    def open(self, cid, local, reload=False):
        """The Capture of (cid, local), decrypting it if needed. One whose files changed since (a recording in
        progress, a key log copied in at its end) is kept until `reload`: Capture.stale() says so meanwhile."""
        e = (self.rescan() if reload else self.entries).get(cid) or self.rescan().get(cid)
        if e is None:
            return None
        if local is None:
            local = e.local_default
        key = (cid, local)
        with self.lock:
            cap = self.caps.get(key)
            if cap is not None and reload and cap.state != "loading":
                cap = None
            if cap is None:
                cap = Capture(e, local, self.keylogs)
                self.caps[key] = cap
                threading.Thread(target=cap.run, name="tls-decrypt", daemon=True).start()
            if key in self.order:
                self.order.remove(key)
            self.order.append(key)
            while len(self.order) > 4:
                self.caps.pop(self.order.pop(0), None)
        return cap
