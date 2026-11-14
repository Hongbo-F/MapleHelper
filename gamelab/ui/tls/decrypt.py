#!/usr/bin/env python3
# A copy of D:/log decoding .tools/tls_decrypt.py as of 2026-11-13 (uncommitted changes
# included), for the /tls/ page (ui.tls). That repo's copy is where it was made; a fix to one belongs in the other.
"""Decrypt the TLS connections of a pcap with an SSLKEYLOGFILE key log and write HTTP transcripts.

    python .tools/tls_decrypt.py <capture.pcap> <keys.log> [--out DIR]

Needs the `cryptography` package (the `maplehelper` conda env has it).  Wireshark 2.4 on this PC cannot do RFC 8446 TLS 1.3,
so this does it directly:
    TLS 1.3   CLIENT/SERVER_HANDSHAKE_TRAFFIC_SECRET + CLIENT/SERVER_TRAFFIC_SECRET_0 (KeyUpdate followed), AES-GCM / ChaCha20
    TLS 1.2   CLIENT_RANDOM <master secret>, AEAD suites (AES-GCM / ChaCha20)
HTTP/1.x is parsed (chunked, gzip/deflate/zstd); HTTP/2 is detected and its frames listed (no HPACK here).

Damaged frames: a MikroTik `/tool sniffer` TZSP stream carries GRO-merged frames (IP length > 1500) at full length but with
everything after the first part zero-filled.  Those bytes are lost.  A TLS record that overlaps them cannot be authenticated;
it is decrypted WITHOUT authentication (GCM / ChaCha20 are stream ciphers, so the intact bytes come out right) and the lost
bytes are written as NUL and reported (`lost` columns, `[N bytes lost in the capture]` in the transcripts).

A frame > 1500 B captured ON this PC (Npcap) is the opposite: a Large Send Offload super-frame, intact.  capture.pcap does not
say where a frame came from, so the source of every frame is worked out (first rule that applies):
    --local                        every frame is this PC's: all big frames are intact
    tzsp.raw beside the pcap       (or --tzsp-raw FILE) the recorder's copy of every datagram that arrived from the router
                                   (UDP 37008): a frame found in it came from the router (big = damaged), any other frame
                                   came from Npcap (big = intact).  This is what makes --both recordings right without flags.
    summary.json beside the pcap   a --local recording (a local capture, no router stream): all frames are this PC's
    otherwise                      every big frame is treated as a router frame (damaged)

Writes to DIR (default: <pcap folder>/tls):
    index.tsv                     one row per TLS connection in the pcap (decrypted or why not)
    NN_<sni>_<port>.txt           HTTP transcript of a decrypted connection (timestamps from the capture)
    NN_<sni>_<port>.c2s.bin/.s2c.bin   the decrypted application data stream of each direction
    bodies/NN_mM_{req,resp}.<ext> every non-empty HTTP body (decoded)
"""
from __future__ import annotations

import argparse
import base64
import bisect
import gzip
import hashlib
import hmac
import json
import os
import re
import struct
import sys
import time
import zlib

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

try:
    from compression import zstd as _zstd          # Python 3.14+
except Exception:                                 # pragma: no cover
    _zstd = None

HRR_RANDOM = bytes.fromhex("CF21AD74E59A6111BE1D8C021E65B891C2A211167ABB8C5E079E09E2C8A8339C")
SUITES13 = {0x1301: ("TLS_AES_128_GCM_SHA256", AESGCM, 16, hashlib.sha256),
            0x1302: ("TLS_AES_256_GCM_SHA384", AESGCM, 32, hashlib.sha384),
            0x1303: ("TLS_CHACHA20_POLY1305_SHA256", ChaCha20Poly1305, 32, hashlib.sha256)}
SUITES12 = {0xC02F: ("ECDHE_RSA_AES_128_GCM_SHA256", "gcm", 16, 4, hashlib.sha256),
            0xC02B: ("ECDHE_ECDSA_AES_128_GCM_SHA256", "gcm", 16, 4, hashlib.sha256),
            0xC030: ("ECDHE_RSA_AES_256_GCM_SHA384", "gcm", 32, 4, hashlib.sha384),
            0xC02C: ("ECDHE_ECDSA_AES_256_GCM_SHA384", "gcm", 32, 4, hashlib.sha384),
            0x009C: ("RSA_AES_128_GCM_SHA256", "gcm", 16, 4, hashlib.sha256),
            0x009D: ("RSA_AES_256_GCM_SHA384", "gcm", 32, 4, hashlib.sha384),
            0xCCA8: ("ECDHE_RSA_CHACHA20_POLY1305", "chacha", 32, 12, hashlib.sha256),
            0xCCA9: ("ECDHE_ECDSA_CHACHA20_POLY1305", "chacha", 32, 12, hashlib.sha256)}
ALERTS = {0: "close_notify", 10: "unexpected_message", 20: "bad_record_mac", 40: "handshake_failure", 42: "bad_certificate",
          46: "certificate_unknown", 48: "unknown_ca", 50: "decode_error", 51: "decrypt_error", 70: "protocol_version",
          80: "internal_error", 90: "user_canceled", 109: "missing_extension", 112: "unrecognized_name", 116: "certificate_required",
          120: "no_application_protocol"}


# ------------------------------------------------------------------------------------------------------------ pcap + TCP

def read_pcap(path):
    data = open(path, "rb").read()
    magic = data[:4]
    fmt = {b"\xd4\xc3\xb2\xa1": ("<", 1e6), b"\xa1\xb2\xc3\xd4": (">", 1e6),
           b"\x4d\x3c\xb2\xa1": ("<", 1e9), b"\xa1\xb2\x3c\x4d": (">", 1e9)}.get(magic)
    if not fmt:
        raise SystemExit("%s: not a classic pcap (pcapng is not supported)" % path)
    e, div = fmt
    linktype = struct.unpack_from(e + "I", data, 20)[0]
    off = 24
    while off + 16 <= len(data):
        s, frac, incl, _orig = struct.unpack_from(e + "IIII", data, off)
        off += 16
        yield s + frac / div, linktype, data[off:off + incl]
        off += incl


def ip4(b):
    return "%d.%d.%d.%d" % tuple(b)


def parse_tcp(linktype, pkt):
    if linktype == 1:
        if len(pkt) < 14:
            return None
        et, l3 = struct.unpack_from(">H", pkt, 12)[0], 14
        while et in (0x8100, 0x88A8) and len(pkt) >= l3 + 4:
            et, l3 = struct.unpack_from(">H", pkt, l3 + 2)[0], l3 + 4
    elif linktype in (101, 228, 12):
        et, l3 = (0x0800 if pkt and pkt[0] >> 4 == 4 else 0x86DD), 0
    else:
        return None
    if et == 0x0800:
        if len(pkt) < l3 + 20:
            return None
        ihl = (pkt[l3] & 15) * 4
        tot = struct.unpack_from(">H", pkt, l3 + 2)[0]
        frag = struct.unpack_from(">H", pkt, l3 + 6)[0]
        if pkt[l3 + 9] != 6 or frag & 0x3FFF:
            return None
        src, dst = ip4(pkt[l3 + 12:l3 + 16]), ip4(pkt[l3 + 16:l3 + 20])
        if tot == 0:                       # LSO/TSO super-frame captured on the sending host: Windows leaves total length 0
            tot = len(pkt) - l3
        l4, end = l3 + ihl, min(len(pkt), l3 + tot)
        iplen = tot
    elif et == 0x86DD:
        if len(pkt) < l3 + 40 or pkt[l3 + 6] != 6:
            return None
        plen = struct.unpack_from(">H", pkt, l3 + 4)[0]
        src, dst = pkt[l3 + 8:l3 + 24].hex(), pkt[l3 + 24:l3 + 40].hex()
        l4, end = l3 + 40, min(len(pkt), l3 + 40 + plen)
        iplen = 40 + plen
    else:
        return None
    if end < l4 + 20:
        return None
    sport, dport, seq, _ack = struct.unpack_from(">HHII", pkt, l4)
    doff = (pkt[l4 + 12] >> 4) * 4
    flags = pkt[l4 + 13]
    return src, sport, dst, dport, seq, flags, bytes(pkt[l4 + doff:end]), iplen


class Conn:
    def __init__(self, n, client, server, ts):
        self.n, self.client, self.server, self.first_ts, self.last_ts = n, client, server, ts, ts
        self.segs = {"c2s": [], "s2c": []}
        self.isn = {"c2s": None, "s2c": None}


def tzsp_frame(d):
    """The Ethernet frame inside one TZSP v1 datagram (received / transmit, Ethernet encapsulation), or None."""
    if len(d) < 5 or d[0] != 1 or d[1] not in (0, 1) or (d[2] << 8 | d[3]) != 1:
        return None
    pos, n = 4, len(d)
    while pos < n:                                       # tags: 0 = padding, 1 = end, else [tag][len][value]
        tag = d[pos]
        pos += 1
        if tag == 1:
            return d[pos:] or None
        if tag == 0:
            continue
        if pos >= n or pos + 1 + d[pos] > n:
            return None
        pos += 1 + d[pos]
    return None


def frame_digest(frame):
    return hashlib.blake2b(frame, digest_size=16).digest()


def load_router_frames(path):
    """(set of digests of the frames inside the datagrams of a recorder's tzsp.raw, datagram count).  tzsp.raw =
    [u32 little-endian length][datagram] ...; a truncated last record (a crash mid-write) ends it."""
    frames, n = set(), 0
    with open(path, "rb", buffering=1 << 20) as f:
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            (ln,) = struct.unpack("<I", head)
            if ln > 65535:
                break
            d = f.read(ln)
            if len(d) < ln:
                break
            n += 1
            fr = tzsp_frame(d)
            if fr:
                frames.add(frame_digest(fr))
    return frames, n


def build_conns(pcap, trust_all=False, router=None):
    """TCP connections of the pcap.  A frame > 1500 B is marked damaged unless it is trusted: `trust_all` (every frame is
    this PC's), or `router` (digests of the frames that came from the router) is given and the frame is not in it.
    Returns (conns, frame source counts)."""
    conns, current = [], {}
    src_count = {"router": 0, "local": 0, "router big": 0, "local big": 0}
    for ts, lt, pkt in read_pcap(pcap):
        if router is not None:
            from_router = frame_digest(pkt) in router
            src_count["router" if from_router else "local"] += 1
        else:
            from_router = not trust_all
        t = parse_tcp(lt, pkt)
        if not t:
            continue
        src, sport, dst, dport, seq, flags, payload, iplen = t
        if iplen > 1500 and router is not None:
            src_count["router big" if from_router else "local big"] += 1
        a, b = (src, sport), (dst, dport)
        key = (a, b) if a < b else (b, a)
        syn, ack = flags & 0x02, flags & 0x10
        c = current.get(key)
        if syn and not ack and (c is None or (c.isn["c2s"] is not None and c.isn["c2s"] != seq)):
            c = None                                     # a new connection on this 4-tuple
        if c is None:
            if syn and not ack:
                client, server = a, b
            elif syn and ack:
                client, server = b, a
            else:                                        # first seen mid-stream: the lower port is the server
                client, server = (a, b) if a[1] > b[1] else (b, a)
            c = Conn(len(conns) + 1, client, server, ts)
            conns.append(c)
            current[key] = c
        d = "c2s" if a == c.client else "s2c"
        if syn:
            c.isn[d] = seq
        c.last_ts = ts
        if payload:
            c.segs[d].append((ts, seq, payload, iplen > 1500 and from_router))
    return conns, src_count


def zero_tail(p, run=16):
    """Offset where the zero-filled part of a damaged (GRO-merged) frame starts, or None."""
    i = p.find(b"\0" * run)
    return i if i >= 0 else None


def merge(ranges):
    out = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def subtract(ranges, cover):
    out = []
    for a, b in ranges:
        for c, d in cover:
            if d <= a or c >= b:
                continue
            if c > a:
                out.append((a, c))
            a = max(a, d)
            if a >= b:
                break
        if a < b:
            out.append((a, b))
    return out


def assemble(segs, isn):
    """Bytes of one direction from offset 0 up to the first hole, (offset, ts) marks, the hole list and the damaged
    (zero-filled) ranges that no intact copy covers."""
    if not segs:
        return b"", [], [], []
    base = (isn + 1) & 0xFFFFFFFF if isn is not None else min(segs, key=lambda s: s[0])[1]
    # A merged frame keeps only its first segment: past the MSS come some garbage bytes, then zeros (checked on 2026-11-11
    # against base64 response bodies).  The MSS is the largest normal segment of this direction (1368 when there is none).
    mss = min(max([len(p) for _t, _s, p, big in segs if not big] + [0]) or 1368, 1460)
    items = []
    for ts, seq, p, big in segs:
        rel = (seq - base) & 0xFFFFFFFF
        if rel >= 0x80000000:
            continue
        z = None
        if big:
            zt = zero_tail(p)
            z = min(mss, len(p) if zt is None else zt)
        items.append((rel, ts, p, z))
    items.sort(key=lambda x: (x[0], x[1]))
    buf, marks, holes, pos = bytearray(), [], [], 0
    good, bad = [], []
    for rel, ts, p, z in items:
        if rel > pos:
            holes.append((pos, rel))
            break
        if z is None:
            good.append((rel, rel + len(p)))
        else:
            good.append((rel, rel + z))
            bad.append((rel + z, rel + len(p)))
        end = rel + len(p)
        if end > pos:
            marks.append((pos, ts))
            buf += p[pos - rel:]
            pos = end
    # An intact copy (retransmission) repairs the damaged ranges and nothing else: the first copy of a good byte stands.  A
    # later copy must not overwrite it - a Windows TCP keep-alive probe re-sends the last byte at SND.NXT-1 as 0x00, which
    # used to replace the last byte of the record before every idle period (its GCM tag -> bad_record_mac on the record).
    badm = merge(bad)
    for rel, ts, p, z in items:
        if z is None and len(p) > 1 and rel < pos:
            for a, b in overlap(badm, rel, min(rel + len(p), pos)):
                buf[rel + a:rel + b] = p[a:b]
    damage = subtract(badm, merge(good))
    damage = [(a, min(b, pos)) for a, b in damage if a < pos]
    return bytes(buf), marks, holes, damage


def overlap(ranges, a, b):
    """Parts of `ranges` inside [a, b), relative to a."""
    return [(max(x, a) - a, min(y, b) - a) for x, y in ranges if x < b and y > a]


def ts_at(marks, off):
    i = bisect.bisect_right([m[0] for m in marks], off) - 1
    return marks[max(i, 0)][1] if marks else 0.0


# ------------------------------------------------------------------------------------------------------------ TLS

def records(buf):
    o = 0
    while o + 5 <= len(buf):
        ct, ver, ln = struct.unpack_from(">BHH", buf, o)
        if ct not in (20, 21, 22, 23, 24) or ln > 18432 + 256:
            return o, "not a TLS record at offset %d" % o
        if o + 5 + ln > len(buf):
            return o, "partial record at offset %d (%d of %d bytes)" % (o, len(buf) - o - 5, ln)
        yield o, ct, buf[o:o + 5], buf[o + 5:o + 5 + ln]
        o += 5 + ln
    return o, None


def iter_records(buf, notes):
    gen = records(buf)
    while True:
        try:
            yield next(gen)
        except StopIteration as stop:
            if stop.value and stop.value[1]:
                notes.append(stop.value[1])
            return


def hs_messages(hsbuf):
    """Pop complete handshake messages from a bytearray."""
    out = []
    while len(hsbuf) >= 4:
        ln = int.from_bytes(hsbuf[1:4], "big")
        if len(hsbuf) < 4 + ln:
            break
        out.append((hsbuf[0], bytes(hsbuf[4:4 + ln])))
        del hsbuf[:4 + ln]
    return out


def parse_exts(b, o):
    exts = {}
    if o + 2 > len(b):
        return exts
    end = o + 2 + struct.unpack_from(">H", b, o)[0]
    o += 2
    while o + 4 <= end:
        t, ln = struct.unpack_from(">HH", b, o)
        exts[t] = b[o + 4:o + 4 + ln]
        o += 4 + ln
    return exts


def alpn_list(ext):
    if not ext:
        return []
    out, o = [], 2
    while o < len(ext):
        ln = ext[o]
        out.append(ext[o + 1:o + 1 + ln].decode("latin-1"))
        o += 1 + ln
    return out


def parse_client_hello(body):
    rnd = body[2:34]
    o = 34
    o += 1 + body[o]
    o += 2 + struct.unpack_from(">H", body, o)[0]
    o += 1 + body[o]
    exts = parse_exts(body, o)
    sni = None
    if 0 in exts and len(exts[0]) > 5:
        e = exts[0]
        sni = e[5:5 + struct.unpack_from(">H", e, 3)[0]].decode("latin-1")
    return {"random": rnd, "sni": sni, "alpn": alpn_list(exts.get(16))}


def parse_server_hello(body):
    ver = struct.unpack_from(">H", body, 0)[0]
    rnd = body[2:34]
    o = 34
    o += 1 + body[o]
    suite = struct.unpack_from(">H", body, o)[0]
    exts = parse_exts(body, o + 3)
    if 43 in exts and len(exts[43]) == 2:
        ver = struct.unpack_from(">H", exts[43])[0]
    alpn = alpn_list(exts.get(16))
    return {"version": ver, "random": rnd, "suite": suite, "alpn": alpn[0] if alpn else None, "hrr": rnd == HRR_RANDOM}


def first_hello(buf, want):
    hsbuf = bytearray()
    for _o, ct, _h, frag in iter_records(buf, []):
        if ct == 23:
            break
        if ct != 22:
            continue
        hsbuf += frag
        for t, body in hs_messages(hsbuf):
            if want == 1 and t == 1:
                return parse_client_hello(body)
            if want == 2 and t == 2:
                sh = parse_server_hello(body)
                if not sh["hrr"]:
                    return sh
    return None


def hkdf_expand_label(secret, label, ctx, length, h):
    full = b"tls13 " + label
    info = struct.pack(">H", length) + bytes([len(full)]) + full + bytes([len(ctx)]) + ctx
    out, t, i = b"", b"", 1
    while len(out) < length:
        t = hmac.new(secret, t + info + bytes([i]), h).digest()
        out += t
        i += 1
    return out[:length]


def prf12(secret, label, seed, length, h):
    seed = label + seed
    a, out = seed, b""
    while len(out) < length:
        a = hmac.new(secret, a, h).digest()
        out += hmac.new(secret, a + seed, h).digest()
    return out[:length]


def xor_nonce(iv, seq):
    s = b"\0" * (len(iv) - 8) + struct.pack(">Q", seq)
    return bytes(x ^ y for x, y in zip(iv, s))


def stream_decrypt(chacha, key, nonce, data):
    """GCM / ChaCha20-Poly1305 payload decryption WITHOUT the tag check (for records with lost bytes)."""
    if chacha:
        c = Cipher(algorithms.ChaCha20(key, struct.pack("<I", 1) + nonce), mode=None)
    else:
        c = Cipher(algorithms.AES(key), modes.CTR(nonce + b"\0\0\0\2"))
    return c.decryptor().update(data)


class Dir13:
    """One direction of TLS 1.3: a list of traffic secrets (handshake, application, key updates)."""

    def __init__(self, suite, secrets):
        _name, self.aead, self.klen, self.h = SUITES13[suite]
        self.chacha = suite == 0x1303
        self.hlen = self.h().digest_size
        self.secrets = [s for s in secrets if s]
        self.stage, self.seq = 0, 0
        self._keys = {}

    def keys(self, i):
        if i not in self._keys:
            s = self.secrets[i]
            key = hkdf_expand_label(s, b"key", b"", self.klen, self.h)
            self._keys[i] = (self.aead(key), key, hkdf_expand_label(s, b"iv", b"", 12, self.h))
        return self._keys[i]

    def decrypt(self, hdr, frag, damaged):
        """-> (content type, plaintext, plaintext lost ranges, authenticated) or None."""
        if not damaged:
            for i in range(self.stage, len(self.secrets)):
                aead, _k, iv = self.keys(i)
                seq = self.seq if i == self.stage else 0
                try:
                    pt = aead.decrypt(xor_nonce(iv, seq), frag, hdr)
                except InvalidTag:
                    continue
                self.stage, self.seq = i, seq + 1
                pt = pt.rstrip(b"\0")
                return (pt[-1], pt[:-1], [], True) if pt else None
            self.seq += 1                                    # count it, so the next records still line up
            return None
        _a, key, iv = self.keys(self.stage)
        pt = bytearray(stream_decrypt(self.chacha, key, xor_nonce(iv, self.seq), frag[:-16]))
        self.seq += 1
        lost = [(a, min(b, len(pt))) for a, b in damaged if a < len(pt)]
        for a, b in lost:
            pt[a:b] = bytes(b - a)
        if lost and lost[-1][1] >= len(pt):
            return 23, bytes(pt), lost, False                # type byte and padding lost: assume application data
        pt = bytes(pt).rstrip(b"\0")
        return (pt[-1], pt[:-1], [(a, min(b, len(pt) - 1)) for a, b in lost], False) if pt else None

    def finished(self):
        if self.stage == 0 and len(self.secrets) > 1:
            self.stage, self.seq = 1, 0

    def key_update(self):
        new = hkdf_expand_label(self.secrets[self.stage], b"traffic upd", b"", self.hlen, self.h)
        self.secrets = self.secrets[:self.stage + 1] + [new]
        self.stage, self.seq = self.stage + 1, 0


class Dir12:
    def __init__(self, suite, key, iv):
        _name, self.mode, _kl, _il, _h = SUITES12[suite]
        self.key = key
        self.aead = AESGCM(key) if self.mode == "gcm" else ChaCha20Poly1305(key)
        self.iv, self.seq, self.active = iv, 0, False

    def decrypt(self, hdr, frag, damaged):
        ct, ver = hdr[0], hdr[1:3]
        if self.mode == "gcm":
            nonce, body, shift = self.iv + frag[:8], frag[8:], 8
        else:
            nonce, body, shift = xor_nonce(self.iv, self.seq), frag, 0
        aad = struct.pack(">Q", self.seq) + bytes([ct]) + ver + struct.pack(">H", len(body) - 16)
        self.seq += 1
        if not damaged:
            try:
                return ct, self.aead.decrypt(nonce, body, aad), [], True
            except InvalidTag:
                return None
        pt = bytearray(stream_decrypt(self.mode == "chacha", self.key, nonce, body[:-16]))
        lost = [(max(a - shift, 0), min(b - shift, len(pt))) for a, b in damaged if b > shift and a - shift < len(pt)]
        for a, b in lost:
            pt[a:b] = bytes(b - a)
        return ct, bytes(pt), lost, False


def run_direction(buf, marks, damage, tls13, dec, notes, recs=None):
    """Decrypt one direction.  Returns (app bytes, [(app offset, ts)], app lost ranges, info dict).  With a list as
    `recs`, one dict per TLS record is appended to it: off (stream offset), ts, ct (outer content type), len (fragment),
    status (plain / ok / unauth / failed / nokey), inner (inner content type), ptlen, lost (plaintext bytes lost),
    app_off (offset in the app stream, application data), pt (plaintext of a handshake / alert record)."""
    app, app_marks, app_lost = bytearray(), [], []
    info = {"records": 0, "failed": 0, "unauth": 0, "alerts": [], "alpn": None, "hs": []}
    hsbuf = bytearray()
    for off, ct, hdr, frag in iter_records(buf, notes):
        rec = None
        if recs is not None:
            rec = {"off": off, "ts": ts_at(marks, off), "ct": ct, "len": len(frag), "status": "plain", "inner": None,
                   "ptlen": None, "lost": 0, "app_off": None, "pt": None}
            recs.append(rec)
        if ct == 20:
            if not tls13 and dec:
                dec.active = True
            continue
        encrypted = (tls13 and ct == 23) or (not tls13 and dec is not None and dec.active)
        if not encrypted:
            if ct == 21 and len(frag) >= 2:
                info["alerts"].append("plain %s" % ALERTS.get(frag[1], frag[1]))
            if rec is not None and ct in (21, 22):
                rec["pt"] = frag
            continue
        if dec is None:
            info["failed"] += 1
            if rec is not None:
                rec["status"] = "nokey"
            continue
        if overlap(damage, off, off + 5):
            notes.append("record header at %d lost" % off)
        r = dec.decrypt(hdr, frag, overlap(damage, off + 5, off + 5 + len(frag)))
        if r is None:
            info["failed"] += 1
            if rec is not None:
                rec["status"] = "failed"
            continue
        ict, pt, lost, auth = r
        info["records"] += 1
        if not auth:
            info["unauth"] += 1
        if rec is not None:
            rec.update(status="ok" if auth else "unauth", inner=ict, ptlen=len(pt), lost=sum(b - a for a, b in lost))
            if ict == 23:
                rec["app_off"] = len(app)
            else:
                rec["pt"] = pt
        if ict == 22:
            hsbuf += pt
            for t, body in hs_messages(hsbuf):
                info["hs"].append(t)
                if t == 8:                                       # EncryptedExtensions
                    alpn = alpn_list(parse_exts(body, 0).get(16))
                    if alpn:
                        info["alpn"] = alpn[0]
                elif t == 20 and tls13:                          # Finished: the application keys follow
                    dec.finished()
                elif t == 24 and tls13:                          # KeyUpdate
                    dec.key_update()
        elif ict == 23:
            app_marks.append((len(app), ts_at(marks, off)))
            app_lost += [(len(app) + a, len(app) + b) for a, b in lost if b > a]
            app += pt
        elif ict == 21 and len(pt) >= 2:
            info["alerts"].append(ALERTS.get(pt[1], str(pt[1])))
    return bytes(app), app_marks, app_lost, info


# ------------------------------------------------------------------------------------------------------------ HTTP

def parse_http1(buf, is_request, req_methods=()):
    msgs, o, k = [], 0, 0
    while o < len(buf):
        i = buf.find(b"\r\n\r\n", o)
        if i < 0:
            msgs.append({"off": o, "end": len(buf), "start": "<incomplete head: %d bytes>" % (len(buf) - o), "headers": [],
                         "body": buf[o:], "complete": False, "status": None})
            break
        lines = buf[o:i].decode("latin-1").split("\r\n")
        start, headers = lines[0], []
        for line in lines[1:]:
            name, _, val = line.partition(":")
            headers.append((name.strip(), val.strip()))
        hd = {n.lower(): v for n, v in headers}
        p, complete = i + 4, True
        status = None
        if not is_request:
            try:
                status = int(start.split(" ")[1])
            except Exception:
                status = 0
        if "chunked" in hd.get("transfer-encoding", "").lower():
            body = bytearray()
            while True:
                j = buf.find(b"\r\n", p)
                if j < 0:
                    complete = False
                    p = len(buf)
                    break
                try:
                    size = int(buf[p:j].split(b";")[0].strip(), 16)
                except ValueError:
                    complete = False
                    p = len(buf)
                    break
                p = j + 2
                if size == 0:
                    if buf[p:p + 2] == b"\r\n":
                        p += 2
                    else:
                        e = buf.find(b"\r\n\r\n", p)
                        p = e + 4 if e >= 0 else len(buf)
                    break
                body += buf[p:p + size]
                if p + size > len(buf):
                    complete = False
                p += size + 2
            end = p
        elif "content-length" in hd:
            n = int(hd["content-length"] or 0)
            body, end = buf[p:p + n], p + n
            complete = end <= len(buf)
        elif is_request or (status and (status < 200 or status in (204, 304))) or \
                (k < len(req_methods) and req_methods[k] == "HEAD"):
            body, end = b"", p
        else:
            body, end = buf[p:], len(buf)
        msgs.append({"off": o, "end": min(end, len(buf)), "start": start, "headers": headers, "body": bytes(body),
                     "complete": complete, "status": status})
        if not is_request and status and 100 <= status < 200:
            pass                                                   # interim response: same request
        else:
            k += 1
        o = end
    return msgs


def decode_g(body):
    """NGS `CE: G` bodies (x-*.ngs.nexon.com): base64 of a gzip-shaped blob XORed with one byte (= blob[3], the first mtime
    byte, which is 0 in the clear); raw deflate from offset 10.  A body cut by lost capture bytes is decoded up to the cut."""
    cut = body.find(b"\0")
    txt = re.sub(rb"[^A-Za-z0-9+/=]", b"", body if cut < 0 else body[:cut])
    txt = txt[:len(txt) // 4 * 4]                        # a body cut short (lost bytes, lost records) ends mid-quantum
    raw = base64.b64decode(txt)
    if len(raw) < 11:
        return body, "G (too short to decode)"
    k = raw[3]
    x = bytes(c ^ k for c in raw)
    d = zlib.decompressobj(-15)
    out = d.decompress(x[10:])
    return out, "G: base64, XOR 0x%02x, deflate%s" % (k, "" if d.eof else ", PARTIAL (the capture lost the rest)")


def decode_body(headers, body):
    hd = {n.lower(): v for n, v in headers}
    enc = hd.get("content-encoding", "").lower().strip()
    if hd.get("ce", "").strip() == "G":                  # the NGS servers' own header
        enc = "g"
    if not body or not enc or enc == "identity":
        return body, None
    try:
        if enc == "gzip":
            return gzip.decompress(body), "gzip"
        if enc == "deflate":
            try:
                return zlib.decompress(body), "deflate"
            except zlib.error:
                return zlib.decompress(body, -15), "deflate(raw)"
        if enc == "g":
            return decode_g(body)
        if enc == "zstd" and _zstd:
            return _zstd.decompress(body), "zstd"
    except Exception as e:
        return body, "%s (decode failed: %s)" % (enc, e)
    return body, "%s (not decoded: no decoder here)" % enc


def guess_ext(headers, body):
    ctype = {n.lower(): v for n, v in headers}.get("content-type", "").lower()
    for k, ext in (("json", "json"), ("html", "html"), ("xml", "xml"), ("text", "txt"), ("x-www-form-urlencoded", "form"),
                   ("octet-stream", "bin"), ("protobuf", "pb"), ("image/png", "png"), ("image/jpeg", "jpg")):
        if k in ctype:
            return ext
    if body[:1] in (b"{", b"["):
        return "json"
    return "bin"


def render_body(body, ext):
    if not body:
        return []
    try:
        text = body.decode("utf-8")
        printable = sum(ch.isprintable() or ch in "\r\n\t\0" for ch in text) >= 0.97 * len(text)
    except UnicodeDecodeError:
        printable = False
    if printable:
        text = re.sub("\0{8,}", lambda m: "[%d bytes lost in the capture]" % len(m.group(0)), text)
        if ext == "json":
            try:
                return json.dumps(json.loads(text), indent=2, ensure_ascii=False).splitlines()
            except ValueError:
                pass
        return text.splitlines()
    out = []
    for i in range(0, min(len(body), 1024), 16):
        chunk = body[i:i + 16]
        out.append("%06x  %-48s %s" % (i, " ".join("%02x" % x for x in chunk),
                                       "".join(chr(x) if 32 <= x < 127 else "." for x in chunk)))
    if len(body) > 1024:
        out.append("... (%d bytes in all)" % len(body))
    return out


H2_TYPES = {0: "DATA", 1: "HEADERS", 2: "PRIORITY", 3: "RST_STREAM", 4: "SETTINGS", 5: "PUSH_PROMISE", 6: "PING", 7: "GOAWAY",
            8: "WINDOW_UPDATE", 9: "CONTINUATION"}


def h2_frames(buf):
    o = 24 if buf.startswith(b"PRI * HTTP/2.0") else 0
    out = []
    while o + 9 <= len(buf):
        ln = int.from_bytes(buf[o:o + 3], "big")
        t, fl, sid = buf[o + 3], buf[o + 4], struct.unpack_from(">I", buf, o + 5)[0] & 0x7FFFFFFF
        out.append((o, H2_TYPES.get(t, str(t)), fl, sid, ln))
        o += 9 + ln
    return out


# ------------------------------------------------------------------------------------------------------------ main

def load_keylog(path):
    keys = {}
    for line in open(path, encoding="latin-1"):
        parts = line.strip().split()
        if len(parts) != 3 or parts[0].startswith("#"):
            continue
        label, cr, secret = parts
        try:
            keys.setdefault(bytes.fromhex(cr), {})[label] = bytes.fromhex(secret)
        except ValueError:
            continue
    return keys


def fmt_ts(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) + ".%03d" % int((ts % 1) * 1000)


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:60]


TLS13_SECRETS = ("CLIENT_HANDSHAKE_TRAFFIC_SECRET", "SERVER_HANDSHAKE_TRAFFIC_SECRET", "CLIENT_TRAFFIC_SECRET_0",
                 "SERVER_TRAFFIC_SECRET_0")


def decrypt_conn(c, keys, with_records=False):
    """Decrypt one TCP connection with the key log `keys` (load_keylog).  None if it is not TLS (or its ClientHello was
    not captured), else a dict:
        ch, sh          parsed ClientHello / ServerHello (sh None: no ServerHello in the capture)
        notes           list of notes (holes, missing pieces, record parse problems)
        tls, suite      "1.3" / hex version and suite name ("" when there is no ServerHello)
        key             "no" / "yes" / "partial (missing ...)"
        keyed           True if the client random is in the key log
        alpn            ALPN chosen (decrypted connections)
        c2s, s2c        None unless decrypted, else {"app": bytes, "marks": [(app offset, ts)], "lost": app lost ranges,
                        "info": run_direction info, "records": list of record dicts (with_records) or None}"""
    cbuf, cmarks, choles, cdmg = assemble(c.segs["c2s"], c.isn["c2s"])
    sbuf, smarks, sholes, sdmg = assemble(c.segs["s2c"], c.isn["s2c"])
    ch = first_hello(cbuf, 1) if cbuf[:1] == b"\x16" else None
    if not ch:
        return None
    sh = first_hello(sbuf, 2)
    notes = []
    if choles:
        notes.append("c2s hole at %d" % choles[0][0])
    if sholes:
        notes.append("s2c hole at %d" % sholes[0][0])
    r = {"ch": ch, "sh": sh, "notes": notes, "tls": "", "suite": "", "key": "no", "keyed": False, "alpn": None,
         "c2s": None, "s2c": None}
    if not sh:
        notes.append("no ServerHello in the capture")
        return r
    tls13 = sh["version"] == 0x0304
    r["tls"] = "1.3" if tls13 else "%x" % sh["version"]
    suites = SUITES13 if tls13 else SUITES12
    r["suite"] = suites.get(sh["suite"], ("0x%04x (unsupported)" % sh["suite"],))[0]
    k = keys.get(ch["random"])
    if not k:
        notes.append("client random not in the key log")
        return r
    r["keyed"] = True
    if sh["suite"] not in suites:
        r["key"] = "yes"
        notes.append("cipher suite not supported here")
        return r
    if tls13:
        need = TLS13_SECRETS
        missing = [x for x in need if x not in k]
        cdec = Dir13(sh["suite"], [k.get(need[0]), k.get(need[2])])
        sdec = Dir13(sh["suite"], [k.get(need[1]), k.get(need[3])])
    else:
        missing = [] if "CLIENT_RANDOM" in k else ["CLIENT_RANDOM"]
        cdec = sdec = None
        if not missing:
            _n, _m, kl, il, h = SUITES12[sh["suite"]]
            kb = prf12(k["CLIENT_RANDOM"], b"key expansion", sh["random"] + ch["random"], 2 * kl + 2 * il, h)
            cdec = Dir12(sh["suite"], kb[:kl], kb[2 * kl:2 * kl + il])
            sdec = Dir12(sh["suite"], kb[kl:2 * kl], kb[2 * kl + il:2 * kl + 2 * il])
    r["key"] = "yes" if not missing else "partial (missing %s)" % ",".join(missing)
    for d, buf, marks, dmg, dec in (("c2s", cbuf, cmarks, cdmg, cdec), ("s2c", sbuf, smarks, sdmg, sdec)):
        recs = [] if with_records else None
        app, am, lost, info = run_direction(buf, marks, dmg, tls13, dec, notes, recs)
        r[d] = {"app": app, "marks": am, "lost": lost, "info": info, "records": recs}
    r["alpn"] = sh["alpn"] or r["s2c"]["info"]["alpn"]
    return r


def load_keylogs(paths):
    """load_keylog of several files, merged (the first file that has a secret wins)."""
    keys = {}
    for p in paths:
        for cr, d in load_keylog(p).items():
            e = keys.setdefault(cr, {})
            for label, secret in d.items():
                e.setdefault(label, secret)
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pcap")
    ap.add_argument("keylog")
    ap.add_argument("--out")
    ap.add_argument("--local", action="store_true",
                    help="every frame was captured on the endpoint itself (npcap/dumpcap): frames over 1500 bytes are LSO/RSC "
                         "super-frames with intact data, not router-damaged GRO frames (overrides the detection)")
    ap.add_argument("--tzsp-raw", metavar="FILE",
                    help="the recorder's tzsp.raw of this capture (default: tzsp.raw beside the pcap): frames found in it came "
                         "from the router, all others from this PC")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.pcap)), "tls")
    os.makedirs(os.path.join(out, "bodies"), exist_ok=True)
    keys = load_keylog(a.keylog)
    conns, sources = frame_sources(a)
    used = set()
    rows = [("n", "start", "client", "server", "sni", "tls", "suite", "alpn", "key", "records c2s", "records s2c",
             "failed", "app bytes c2s", "app bytes s2c", "http", "requests", "notes", "lost c2s", "lost s2c",
             "unauthenticated")]
    for c in conns:
        r = decrypt_conn(c, keys)
        if r is None:
            continue                                                   # not TLS (or ClientHello not captured)
        ch, notes = r["ch"], r["notes"]
        name = "%02d_%s_%d" % (c.n, safe(ch["sni"] or c.server[0]), c.client[1])
        row = [c.n, fmt_ts(c.first_ts), "%s:%d" % c.client, "%s:%d" % c.server, ch["sni"] or "", r["tls"], r["suite"], "",
               r["key"], 0, 0, 0, 0, 0, "", "", "", 0, 0, 0]
        if r["keyed"]:
            used.add(ch["random"])
        if r["c2s"] is None:
            row[16] = "; ".join(notes)
            rows.append(row)
            continue
        capp, cam, clost, cinfo = (r["c2s"][x] for x in ("app", "marks", "lost", "info"))
        sapp, sam, slost, sinfo = (r["s2c"][x] for x in ("app", "marks", "lost", "info"))
        row[17], row[18] = sum(b - a for a, b in clost), sum(b - a for a, b in slost)
        row[19] = cinfo["unauth"] + sinfo["unauth"]
        alpn = r["alpn"]
        row[7] = alpn or ""
        row[9], row[10], row[11] = cinfo["records"], sinfo["records"], cinfo["failed"] + sinfo["failed"]
        row[12], row[13] = len(capp), len(sapp)
        alerts = ["c2s alert " + x for x in cinfo["alerts"]] + ["s2c alert " + x for x in sinfo["alerts"]]
        open(os.path.join(out, name + ".c2s.bin"), "wb").write(capp)
        open(os.path.join(out, name + ".s2c.bin"), "wb").write(sapp)
        lines = ["connection %d  %s:%d -> %s:%d  SNI %s" % (c.n, c.client[0], c.client[1], c.server[0], c.server[1], ch["sni"]),
                 "TLS %s %s  ALPN offered %s, chosen %s" % (row[5], row[6], ",".join(ch["alpn"]) or "-", alpn or "-"),
                 "capture %s .. %s   decrypted records c2s %d, s2c %d, failed %d, unauthenticated (lost bytes) %d   %s" % (
                     fmt_ts(c.first_ts), fmt_ts(c.last_ts), row[9], row[10], row[11], row[19], "; ".join(notes + alerts)),
                 "bytes lost in the capture (router zero-filled merged frames): c2s %d of %d, s2c %d of %d" % (
                     row[17], len(capp), row[18], len(sapp)), ""]
        if capp.startswith(b"PRI * HTTP/2.0") or alpn == "h2":
            row[14] = "h2"
            for label, buf, marks in (("C->S", capp, cam), ("S->C", sapp, sam)):
                for o, t, fl, sid, ln in h2_frames(buf):
                    lines.append("%s %s  %-13s stream %-3d flags %02x len %d" % (fmt_ts(ts_at(marks, o)), label, t, sid, fl, ln))
            row[15] = "(HTTP/2: headers are HPACK-coded, not decoded here)"
        elif capp:
            row[14] = "http/1.x"
            reqs = parse_http1(capp, True)
            resps = parse_http1(sapp, False, [r["start"].split(" ", 1)[0] for r in reqs])
            final = [r for r in resps if not (r["status"] and 100 <= r["status"] < 200)]
            row[15] = " | ".join(r["start"][:100] for r in reqs)
            for m, req in enumerate(reqs):
                resp = final[m] if m < len(final) else None
                for kind, msg, marks, lostr in (("req", req, cam, clost), ("resp", resp, sam, slost)):
                    if msg is None:
                        lines += ["<<< no response in the capture", ""]
                        continue
                    arrow = ">>>" if kind == "req" else "<<<"
                    nlost = sum(b - a for a, b in overlap(lostr, msg["off"], msg["end"]))
                    lines.append("%s %s  %s%s%s" % (arrow, fmt_ts(ts_at(marks, msg["off"])), msg["start"],
                                                    "" if msg["complete"] else "   [INCOMPLETE]",
                                                    "   [%d bytes of this message lost in the capture]" % nlost if nlost else ""))
                    lines += ["    %s: %s" % hv for hv in msg["headers"]]
                    body, how = decode_body(msg["headers"], msg["body"])
                    if body:
                        ext = guess_ext(msg["headers"], body)
                        if how and how.startswith("G:") and body[:1] in (b"{", b"["):
                            ext = "json"
                        fn = "%02d_m%d_%s.%s" % (c.n, m + 1, kind, ext)
                        open(os.path.join(out, "bodies", fn), "wb").write(body)
                        lines.append("    -- body %d bytes%s%s -> bodies/%s" % (
                            len(body), " (wire %d, %s)" % (len(msg["body"]), how) if how else "", "", fn))
                        lines += ["    " + x for x in render_body(body, ext)]
                    lines.append("")
            extra = len(final) - len(reqs)
            if extra > 0:
                lines.append("(%d response(s) beyond the requests)" % extra)
        row[16] = "; ".join(notes + alerts)
        open(os.path.join(out, name + ".txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        rows.append(row)
    with open(os.path.join(out, "index.tsv"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    tls_rows = rows[1:]
    dec = [r for r in tls_rows if r[8] == "yes"]
    print("TLS connections in the capture: %d; with keys: %d; decrypted records %d (%d without authentication: lost bytes), "
          "failed %d; plaintext bytes lost %d of %d" % (
              len(tls_rows), len(dec), sum(r[9] + r[10] for r in dec), sum(r[19] for r in dec), sum(r[11] for r in dec),
              sum(r[17] + r[18] for r in dec), sum(r[12] + r[13] for r in dec)))
    print("key log: %d client randoms, %d of them used by this capture" % (len(keys), len(used)))
    print("frame sources:", sources)
    print("written to", out)


def frame_sources(a):
    """Build the connections with every frame's source decided by the rules of the module docstring.
    Returns (conns, a line saying how the sources were decided)."""
    if a.local:
        conns, _s = build_conns(a.pcap, trust_all=True)
        return conns, "--local: every frame is this PC's (big frames intact)"
    here = os.path.dirname(os.path.abspath(a.pcap))
    raw = a.tzsp_raw or os.path.join(here, "tzsp.raw")
    note = ""
    if os.path.isfile(raw):
        router, n = load_router_frames(raw)
        conns, s = build_conns(a.pcap, router=router)
        if s["router"] or not router:
            return conns, ("%s: %d frames from the router (UDP 37008; %d of them > 1500 B, treated as damaged), %d from this "
                           "PC (Npcap; %d > 1500 B, intact)" % (os.path.basename(raw), s["router"], s["router big"], s["local"],
                                                                s["local big"]))
        if a.tzsp_raw:
            raise SystemExit("--tzsp-raw %s (%d datagrams) matches no frame of the pcap: not this capture's" % (raw, n))
        note = "%s (%d datagrams) matches no frame of the pcap, ignored; " % (raw, n)
    summary = os.path.join(here, "summary.json")
    if os.path.isfile(summary):
        try:
            s = json.load(open(summary, encoding="utf-8"))
        except (OSError, ValueError):
            s = {}
        if s.get("local_capture") and not s.get("capture"):
            conns, _s = build_conns(a.pcap, trust_all=True)
            return conns, note + "summary.json: a --local recording, every frame is this PC's (big frames intact)"
    conns, _s = build_conns(a.pcap)
    return conns, note + ("no usable tzsp.raw: every frame > 1500 B is treated as a router frame (damaged); pass --local if "
                          "the capture was taken on the endpoint")


if __name__ == "__main__":
    main()
