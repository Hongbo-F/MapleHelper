"""The web UI's decrypted captures (ui.tls): HPACK as RFC 7541 decodes it, who may read /tls/ (Cloudflare Access for what
comes through the tunnel, ui.tls.access), and a TLS 1.3 capture made here decrypted end to end through the page's API:
its connection, records, an HTTP/1.1 exchange with an NGS `AE: G` request body decoded."""
import base64
import json
import struct
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import hashes  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import padding, rsa  # noqa: E402
from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: E402

from gamelab.ui import tls, web  # noqa: E402
from gamelab.ui.tls import access, decrypt, view  # noqa: E402


# ------------------------------------------------------------------------------------------------ HPACK

def test_hpack_decodes_rfc_7541s_huffman_coded_requests_with_the_dynamic_table_carried_between_them():
    """RFC 7541 C.4.1-C.4.3: three requests on one connection, the later ones indexing what the first added."""
    hp = view.Hpack()
    assert hp.decode(bytes.fromhex("828684418cf1e3c2e5f23a6ba0ab90f4ff")) == [
        (":method", "GET"), (":scheme", "http"), (":path", "/"), (":authority", "www.example.com")]
    assert hp.decode(bytes.fromhex("828684be5886a8eb10649cbf")) == [
        (":method", "GET"), (":scheme", "http"), (":path", "/"), (":authority", "www.example.com"),
        ("cache-control", "no-cache")]
    assert hp.decode(bytes.fromhex("828785bf408825a849e95ba97d7f8925a849e95bb8e8b4bf")) == [
        (":method", "GET"), (":scheme", "https"), (":path", "/index.html"), (":authority", "www.example.com"),
        ("custom-key", "custom-value")]
    assert hp.size == 164  # the table as C.4.3 leaves it: three entries, 164 octets


def test_a_huffman_string_with_bad_padding_or_an_index_past_the_tables_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        view.huff_decode(b"\x00")          # '0' then 3 zero bits of padding (padding is all ones)
    with pytest.raises(ValueError):
        view.Hpack().decode(b"\xbe")       # index 62: the dynamic table is empty


# ------------------------------------------------------------------------------------------------ access

TEAM, AUD = "https://myteam.cloudflareaccess.com", "aud-tag-1"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def jwt(claims=None, key=KEY, alg="RS256", kid="k1"):
    now = time.time()
    c = {"aud": [AUD], "iss": TEAM, "exp": now + 600, "nbf": now - 10, "email": "me@example.com", **(claims or {})}
    h = b64(json.dumps({"alg": alg, "kid": kid}).encode()) + "." + b64(json.dumps(c).encode())
    return h + "." + b64(key.sign(h.encode(), padding.PKCS1v15(), hashes.SHA256()))


@pytest.fixture
def access_set_up(monkeypatch, tmp_path):
    cfg = tmp_path / "cloudflare_access.json"
    cfg.write_text(json.dumps({"team": "myteam", "aud": AUD}), encoding="utf-8")
    monkeypatch.setattr(access, "CONFIG", cfg)
    monkeypatch.setattr(access, "team_key", lambda team, kid: KEY.public_key() if (team, kid) == (TEAM, "k1") else None)


CF = {"Cf-Ray": "8a1b2c3d4e5f-NRT", "Cf-Connecting-IP": "203.0.113.9", "Cdn-Loop": "cloudflare"}


@pytest.mark.parametrize("headers,ok", [
    ({}, True),                                                          # straight to the server: this PC, the LAN
    (CF, False),                                                         # through Cloudflare without a token
    ({**CF, "Cf-Access-Jwt-Assertion": jwt()}, True),
    ({**CF, "Cookie": "a=1; CF_Authorization=" + jwt()}, True),          # Access's cookie works too
    ({"Cdn-Loop": "cloudflare", "Cf-Access-Jwt-Assertion": jwt(key=OTHER)}, False),   # signed by someone else
    ({**CF, "Cf-Access-Jwt-Assertion": jwt({"aud": ["another-app"]})}, False),
    ({**CF, "Cf-Access-Jwt-Assertion": jwt({"iss": "https://evil.cloudflareaccess.com"})}, False),
    ({**CF, "Cf-Access-Jwt-Assertion": jwt({"exp": time.time() - 3600})}, False),
    ({**CF, "Cf-Access-Jwt-Assertion": jwt(kid="unknown")}, False),
    ({**CF, "Cf-Access-Jwt-Assertion": jwt().rsplit(".", 1)[0] + "."}, False),       # no signature
    ({**CF, "Cf-Access-Jwt-Assertion": "e30.e30.x"}, False),                           # alg missing
    ({"Cf-Ray": "x", "Cf-Access-Jwt-Assertion": "not a jwt"}, False),
], ids=["direct", "no-token", "header", "cookie", "other-key", "other-aud", "other-iss", "expired", "unknown-kid",
        "unsigned", "no-alg", "garbage"])
def test_through_cloudflare_only_an_access_token_signed_by_the_team_for_this_application_gets_in(access_set_up, headers,
                                                                                                 ok):
    assert (access.refused(headers) is None) is ok, access.refused(headers)


def test_without_its_configuration_everything_through_cloudflare_is_refused_and_direct_requests_still_get_in(
        monkeypatch, tmp_path):
    monkeypatch.setattr(access, "CONFIG", tmp_path / "none.json")
    assert "isn't set up" in access.refused({**CF, "Cf-Access-Jwt-Assertion": jwt()})
    assert access.refused({}) is None


def test_the_teams_name_alone_its_hostname_or_its_url_all_name_the_same_issuer(tmp_path):
    for team in ("myteam", "myteam.cloudflareaccess.com", "https://myteam.cloudflareaccess.com/"):
        (tmp_path / "c.json").write_text(json.dumps({"team": team, "aud": AUD}), encoding="utf-8")
        assert access.config(tmp_path / "c.json") == {"team": TEAM, "aud": AUD}


# ------------------------------------------------------------------------------------------------ a capture end to end

CR = bytes(range(32))                      # the client random
SECRETS = {name: bytes([i + 1]) * 32 for i, name in enumerate(decrypt.TLS13_SECRETS)}
PLAIN = b'{"lifetoken":"","gid":2867,"log_code":"0xE11F1201","result":4}'
ANSWER = b'{"ok":true}'


def ngs_body(pt: bytes, seed=0x1234, iv=b"\x07\x01\x02\x03\x04\x05\x06\x07\x08") -> bytes:
    """`pt` as the NGS client sends an AE: G body: base64 of [n][n+2 IV bytes][ciphertext] (ui.tls.ngs, inverted)."""
    ct, k = bytearray(), seed
    for j in range(0, len(pt) - 1, 2):
        w = (pt[j] | pt[j + 1] << 8) ^ k
        ct += bytes([w & 0xFF, w >> 8])
        k = (k + w) & 0xFFFF
    if len(pt) % 2:
        ct.append(pt[-1] ^ (k & 0xFF))
    return base64.b64encode(bytes([len(iv) - 2]) + iv + bytes(ct))


class Sealer:
    """One direction's TLS 1.3 records (AES-128-GCM), sealed with its traffic secrets in turn."""

    def __init__(self, *secrets):
        self.secrets, self.stage, self.seq = secrets, 0, 0

    def next_stage(self):
        self.stage, self.seq = self.stage + 1, 0

    def record(self, inner_type: int, data: bytes) -> bytes:
        s = self.secrets[self.stage]
        key = decrypt.hkdf_expand_label(s, b"key", b"", 16, decrypt.hashlib.sha256)
        iv = decrypt.hkdf_expand_label(s, b"iv", b"", 12, decrypt.hashlib.sha256)
        pt = data + bytes([inner_type])
        hdr = struct.pack(">BHH", 23, 0x0303, len(pt) + 16)
        ct = AESGCM(key).encrypt(decrypt.xor_nonce(iv, self.seq), pt, hdr)
        self.seq += 1
        return hdr + ct


def hs(t: int, body: bytes) -> bytes:
    return bytes([t]) + len(body).to_bytes(3, "big") + body


def plain(ct: int, data: bytes) -> bytes:
    return struct.pack(">BHH", ct, 0x0301, len(data)) + data


def client_hello() -> bytes:
    sni = b"x-phaethon.ngs.nexon.com"
    ext_sni = struct.pack(">HHHBH", 0, len(sni) + 5, len(sni) + 3, 0, len(sni)) + sni
    ext_alpn = struct.pack(">HHH", 16, 11, 9) + b"\x08http/1.1"
    exts = ext_sni + ext_alpn
    body = b"\x03\x03" + CR + b"\x00" + b"\x00\x02\x13\x01" + b"\x01\x00" + struct.pack(">H", len(exts)) + exts
    return plain(22, hs(1, body))


def server_hello() -> bytes:
    exts = struct.pack(">HHH", 43, 2, 0x0304)
    body = b"\x03\x03" + bytes(range(100, 132)) + b"\x00" + b"\x13\x01" + b"\x00" + struct.pack(">H", len(exts)) + exts
    return plain(22, hs(2, body))


def pcap(frames) -> bytes:
    """A classic pcap of Ethernet frames: (t, src, sport, dst, dport, seq, flags, payload)."""
    out = [struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)]
    for t, src, sport, dst, dport, seq, flags, payload in frames:
        tcp = struct.pack(">HHIIBBHHH", sport, dport, seq, 0, 5 << 4, flags, 65535, 0, 0) + payload
        ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(tcp), 0, 0, 64, 6, 0, bytes(src), bytes(dst))
        eth = b"\x02" * 6 + b"\x04" * 6 + b"\x08\x00" + ip + tcp
        out.append(struct.pack("<IIII", int(t), int(t % 1 * 1e6), len(eth), len(eth)) + eth)
    return b"".join(out)


@pytest.fixture
def capture(tmp_path):
    """A recording folder with one TLS 1.3 connection to x-phaethon: a POST of an AE: G body and its answer."""
    c2s = Sealer(SECRETS["CLIENT_HANDSHAKE_TRAFFIC_SECRET"], SECRETS["CLIENT_TRAFFIC_SECRET_0"])
    s2c = Sealer(SECRETS["SERVER_HANDSHAKE_TRAFFIC_SECRET"], SECRETS["SERVER_TRAFFIC_SECRET_0"])
    body = ngs_body(PLAIN)
    request = (b"POST /v1 HTTP/1.1\r\nHost: x-phaethon.ngs.nexon.com\r\nAE: G\r\nContent-Type: application/json\r\n"
               b"Content-Length: %d\r\n\r\n" % len(body)) + body
    response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(ANSWER), ANSWER)
    server_flight = server_hello() + s2c.record(22, hs(8, b"\x00\x00") + hs(20, b"\xaa" * 32))
    s2c.next_stage()
    client_flight = c2s.record(22, hs(20, b"\xbb" * 32))
    c2s.next_stage()
    cli, srv = ((192, 168, 1, 2), 50000), ((13, 158, 1, 1), 443)
    t0, ci, si = 1_794_600_000.0, 1000, 5000
    cseq, sseq, frames = ci + 1, si + 1, []

    def c(t, data):
        nonlocal cseq
        frames.append((t, *cli, *srv, cseq, 0x18, data))
        cseq += len(data)

    def s(t, data):
        nonlocal sseq
        frames.append((t, *srv, *cli, sseq, 0x18, data))
        sseq += len(data)

    frames.append((t0, *cli, *srv, ci, 0x02, b""))
    frames.append((t0 + 0.01, *srv, *cli, si, 0x12, b""))
    c(t0 + 0.02, client_hello())
    s(t0 + 0.05, server_flight)
    c(t0 + 0.06, client_flight + c2s.record(23, request))
    s(t0 + 0.10, s2c.record(23, response))
    rec = tmp_path / "captures" / "live" / "20261113-120000"
    rec.mkdir(parents=True)
    (rec / "capture.pcap").write_bytes(pcap(frames))
    (rec / "keys.log").write_text("".join(f"{n} {CR.hex()} {v.hex()}\n" for n, v in SECRETS.items()), encoding="utf-8")
    return tmp_path / "captures"


@pytest.fixture
def served(capture, monkeypatch, tmp_path):
    monkeypatch.setattr(tls, "_library", view.Library([str(capture)]))
    monkeypatch.setattr(access, "CONFIG", tmp_path / "no-access.json")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web._Handler)
    srv.feed = None  # (only /tls/ is asked)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url, headers=None):
    r = urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}))
    return r.status, r.headers, r.read()


def api(base, name, **q):
    return json.loads(get(f"{base}/tls/api/{name}?" + "&".join(f"{k}={v}" for k, v in q.items()))[2])


def opened(base, cap):
    for _ in range(100):
        st = api(base, "open", cap=cap)
        if st["state"] != "loading":
            return st
        time.sleep(0.05)
    raise AssertionError("still decrypting")


def test_a_capture_decrypts_through_the_page_its_request_body_decoded_from_ngs_ae_g(served):
    status, headers, page = get(served + "/tls/")
    assert status == 200 and b"app.js" in page and headers["X-Content-Type-Options"] == "nosniff"
    [cap] = api(served, "captures")
    assert (cap["name"], cap["keylogs"], cap["local_default"]) == ("live/20261113-120000/capture.pcap", ["keys.log"], False)
    assert opened(served, cap["id"])["state"] == "ready"
    d = api(served, "summary", cap=cap["id"])
    assert (d["summary"]["tls_conns"], d["summary"]["decrypted"], d["summary"]["failed"], d["summary"]["lost"]) == (1, 1, 0, 0)
    [conn] = d["conns"]
    assert (conn["sni"], conn["tls"], conn["suite"], conn["proto"], conn["status"]) == (
        "x-phaethon.ngs.nexon.com", "1.3", "TLS_AES_128_GCM_SHA256", "http/1.1", "decrypted")
    [item] = d["items"]
    assert (item["method"], item["path"], item["host"], item["status"]) == ("POST", "/v1", "x-phaethon.ngs.nexon.com", 200)
    assert item["req"]["how"].startswith("NGS AE:G") and item["req"]["display"] == "json"
    body = f"{served}/tls/api/body?cap={cap['id']}&id={item['id']}"
    assert get(body + "&part=req")[2] == PLAIN
    assert get(body + "&part=req&form=wire")[2] == ngs_body(PLAIN)
    status, headers, answer = get(body + "&part=resp")
    assert (answer, headers["Content-Type"]) == (ANSWER, "application/octet-stream")
    records = api(served, "conn", cap=cap["id"], n=conn["n"])["records"]
    assert [(r["dir"], r["inner"] or r["ct"]) for r in records] == [
        ("c2s", "Handshake"), ("s2c", "Handshake"), ("s2c", "Handshake"), ("c2s", "Handshake"),
        ("c2s", "ApplicationData"), ("s2c", "ApplicationData")]
    assert [r["info"] for r in records[:4]] == ["ClientHello", "ServerHello", "EncryptedExtensions, Finished", "Finished"]
    assert api(served, "search", cap=cap["id"], q="E11F1201")["ids"] == [item["id"]]   # found in the decoded body


def test_through_the_tunnel_without_access_the_page_and_its_api_are_refused(served):
    for path in ("/tls/", "/tls/api/captures", "/tls/app.js"):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(served + path, CF)
        assert e.value.code == 403
    assert get(served + "/tls/api/captures")[0] == 200


def test_nothing_but_the_pages_own_files_and_the_capture_s_items_are_served(served):
    [cap] = api(served, "captures")
    opened(served, cap["id"])
    for path in ("/tls/../web.py", "/tls/static/app.js", "/tls/view.py", "/tls/api/nothing",
                 f"/tls/api/body?cap={cap['id']}&id=../../x", f"/tls/api/record?cap={cap['id']}&n=1&dir=zz&i=0",
                 "/tls/api/summary?cap=nope"):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(served + path)
        assert e.value.code == 404, path


# ------------------------------------------------------------------------------------------------ the x-csauth timeline

from types import SimpleNamespace  # noqa: E402
import zlib  # noqa: E402

from gamelab.ui.tls import csauth  # noqa: E402

HW, IP = "20000-5DF338-3F5283-CAFBE5-A35FDE-168AB5D-0183", "192.168.195.98"
DECOY_ENTRY = {"guid": "1", "npacode": "G", "detail": {"reason": "4b3a0bb7"}, "log_id": 5, "log_code": "0x221FFFFF",
               "log_timestamp": 9, "result": 4, "signature_id": 0}
MARKER_ENTRY = {"guid": "1", "npacode": "G", "detail": {}, "log_id": 5, "log_code": "0x00000000", "log_timestamp": 9,
                "result": 0, "signature_id": 0}
INVENTORY = {"guid": "1", "npacode": "G", "detail": {"device_obj": {"cpu": "Ryzen", "gpu": "RTX"},
             "system_obj": {"os": "Windows 10 VB", "edition": "ServerRdsh "}}, "log_id": 7, "log_code": "0xC210F070",
             "log_timestamp": 11, "result": 4, "signature_id": 0}


def token(answer_set, salt=0):
    """A 19-hex token whose crc32 picks `answer_set` (csauth.answer_set)."""
    n = salt * 1000
    while zlib.crc32(("%019x" % n).encode()) % 5 != answer_set:
        n += 1
    return "%019x" % n


def beat(tok, crc, seq, collected, detected, sig=1):
    return {"lifetoken": "L1", "hwid": HW, "guid": "1", "npacode": "G", "process_id": 7, "csauthtoken": tok,
            "app_crc": {"MapleStory.exe": crc}, "module_crc": {"CORE": crc + 1}, "signature_crc": {"0x06061": sig},
            "sequence": seq, "collected_info": collected, "detected_info": detected}


def fake_capture(calls):
    """A ready Capture stand-in: x-csauth items (path, ts, request body, response body) on connection 1."""
    def part(body):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        return SimpleNamespace(body=raw, display="json" if not isinstance(body, bytes) else "text")
    items = [{"type": "http", "host": csauth.HOST, "path": path, "ts": ts, "conn": 1, "id": f"1.{k}", "status": 200,
              "proto": "http/1.1", "req": part(req), "resp": part(resp) if resp is not None else None}
             for k, (path, ts, req, resp) in enumerate(calls)]
    conns = [{"n": 1, "client": IP + ":5000", "sni": csauth.HOST, "decrypted": True, "start": 0, "end": 1},
             {"n": 2, "client": IP + ":5001", "sni": csauth.HOST, "decrypted": False, "start": 50, "end": 60,
              "status": "no key"}]
    return SimpleNamespace(items=items, conns=conns)


def test_the_timeline_shows_each_call_against_the_computers_previous_ones_without_decoys_or_noise():
    t0 = 1_794_600_000.0
    decoys = [DECOY_ENTRY] * 9
    calls = [
        ("/v1/issue", t0, {"lifetoken": "L1", "hwid": HW, "guid": "1", "npacode": "", "process_id": 7}, {"csauthtoken": "x"}),
        ("/v1/issue", t0 + 10, {"lifetoken": "L1", "hwid": HW, "guid": "1", "npacode": "G", "process_id": 7}, {"csauthtoken": "y"}),
        ("/v2", t0 + 70, beat(token(2), 100, {"0x1000": [1, 50]}, [INVENTORY] + decoys, [MARKER_ENTRY] + decoys),
         {"csauthtoken": "a", "config_changed": 1, "auth_cycle": 60, "detected": []}),
        # ticks advance, the token's answer set and its answers differ, one id takes a value it never had: all noise
        ("/v2", t0 + 130, beat(token(3), 900, {"0x1000": [2, 51]}, decoys + [DECOY_ENTRY], [MARKER_ENTRY] + decoys),
         {"csauthtoken": "b", "config_changed": 0, "auth_cycle": 60, "detected": []}),
        # set 2 again but with other answers (a file changed), a rule push, and the inventory again
        ("/v2", t0 + 190, beat(token(2, 1), 101, {"0x1000": [3, 50], "0x2040": [3, 9]}, [INVENTORY] + decoys,
                               [MARKER_ENTRY] + decoys, sig=2),
         {"csauthtoken": "c", "config_changed": 0, "auth_cycle": 60, "detected": []}),
        ("/v2", t0 + 500, beat(token(3, 1), 900, {"0x1000": [4, 50], "0x2040": [4, 9]}, [INVENTORY] + decoys,
                               [MARKER_ENTRY] + decoys, sig=2),
         {"csauthtoken": "d", "config_changed": 0, "auth_cycle": 60, "detected": [{"x": 1}]}),
        ("/v1/close", t0 + 560, {"lifetoken": "L1", "hwid": HW, "guid": "", "npacode": "", "process_id": 7}, b"Lig="),
    ]
    t = csauth.timeline(fake_capture(calls))
    json.dumps(t)
    [c] = t["computers"]
    assert (c["hwid"], c["ips"], c["accounts"], c["beats"]) == (HW, [IP], ["G"], 4)
    assert c["name"] == "Ryzen · RTX · Windows 10 VB · ServerRdsh"
    assert [(u["client"], u["connections"]) for u in c["unkeyed"]] == [(IP, 1)] and t["unkeyed"] == []
    issue1, issue2, b1, b2, b3, b4, close = c["events"]
    assert issue1["initial"] and "body" in issue1 and not issue1["quiet"]
    assert [(x["path"], x["old"], x["new"]) for x in issue2["changes"]] == [("npacode", "", "G")]
    # the first heartbeat: its new fields folded away, its real report, decoys and the marker only counted
    assert set(b1["introduced"]) == {"app_crc", "module_crc", "signature_crc"} and b1["changes"] == []
    assert [r["code"] for r in b1["reports"]] == ["0xC210F070"] and b1["hidden"] == {"decoys": 18, "markers": 1}
    assert b1["response"]["flags"] == {"config_changed": 1} and not b1["quiet"]
    # all noise: quiet, the one minor change kept for the curious
    assert b2["quiet"] and b2["reports"] == [] and [x["path"] for x in b2["changes"]] == ["sequence.0x1000[1]"]
    assert b2["changes"][0]["minor"] and b2["answer_set_first"]
    # the same answer set as b1 with another answer, the rule push, a module joining; the inventory came again
    paths = {x["path"]: x for x in b3["changes"]}
    assert paths["app_crc.MapleStory.exe"]["old"] == 100 and paths["app_crc.MapleStory.exe"]["new"] == 101
    assert "set 2" in paths["app_crc.MapleStory.exe"]["note"] and "module_crc.CORE" in paths
    assert paths["signature_crc.0x06061"]["new"] == 2 and paths["sequence.0x2040"]["op"] == "added"
    assert [r["code"] for r in b3["reports"]] == ["0xC210F070"] and b3["repeated"] == 0   # b2 had none: new again
    # late: a gap; the server's verdict flagged; the inventory repeated from b3
    assert b4["gap"] == 310.0 and b4["response"]["flags"] == {"detected": [{"x": 1}]} and b4["repeated"] == 1
    assert [(x["path"], x["new"]) for x in close["changes"]] == [("guid", ""), ("npacode", "")]
    assert close["response"]["text"] == "Lig="


def test_a_body_the_capture_damaged_is_listed_not_diffed():
    t = csauth.timeline(fake_capture([("/v2", 1.0, b'{"hwid": "2000\x00\x00\x00', None)]))
    [c] = t["computers"]
    assert c["hwid"] == "?" and c["events"][0]["damaged"]


def test_the_page_serves_the_timeline(served):
    [cap] = api(served, "captures")
    opened(served, cap["id"])
    assert api(served, "csauth", cap=cap["id"]) == {"computers": [], "unkeyed": []}   # (x-phaethon only)
