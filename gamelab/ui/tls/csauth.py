"""A capture's x-csauth stream (the NGS anti-cheat's session calls: /v1/issue, the /v2 heartbeat every ~60 s, /v1/close)
as a timeline per computer, each event showing only what changed since the computer's events before it (the user,
2026-11-14: "add a view on x-csauth stream, for each of the captured computer, along time, and remove the useless
0x221FFFFF log_code items in collected info and detected info? Each new entry on the timeline should just show the diff
against last item, so I can focus on real important things").

A computer is the request's `hwid` (the NGS machine id: constant per machine, D:/log decoding X_CSAUTH.md 7.1), named
from its inventory report (0xC210F070) when the capture has one. What an event shows, against that computer's state:

- `changes`: the fields whose value differs from the last value the computer sent (/v1/issue, /v2 and /v1/close share
  the identity fields: a new session shows its lifetoken and pid, a close its blanked account), and `introduced`: the
  fields it sends for the first time (the first /v2 after an issue brings two dozen, folded away).
- Not shown, being noise in every heartbeat: `csauthtoken` (the server's, chained call to call); `app_crc` and
  `module_crc` take one of 5 answer sets, the one `crc32(token) % 5` picks (X_CSAUTH.md 4.1), so each is compared with
  the last heartbeat that answered the same set: a change there is a file or module that changed; `sequence` (module
  check-ins): its ticks advance every heartbeat and a few ids flip between values, so only ids joining or leaving
  and a value an id never had before are shown, the latter as `minor` (a handful of ids keep taking new ones: on the
  2026-11-12 night 0x2040 did 5 times): a heartbeat with only minor changes is still quiet.
- `reports`: detected_info / collected_info entries other than the decoys (log_code 0x221FFFFF, constant reason hash,
  padding the lists to 10) and the one 0x00000000 heartbeat marker, which are only counted (`hidden`); an entry already
  in the computer's previous heartbeat (the same but for log_id / log_timestamp) is counted as `repeated`.
- `response`: its fields that changed (but the token), and config_changed / emergency_stop when set, detected when not
  empty: what the server asked of the client.
- `gap`: seconds since the computer's previous heartbeat when over twice the heartbeat period.
- `quiet`: nothing of the above: the page folds runs of them.

Bodies that aren't JSON (bytes lost in the capture) are listed as `damaged` and left out of the state. x-csauth TLS
connections that couldn't be decrypted (no key: the laptop's second, h2 client) are counted per client address."""
from __future__ import annotations

import json
import zlib

DECOY, MARKER = "0x221FFFFF", "0x00000000"
REPORT_LISTS = ("detected_info", "collected_info")
ANSWER_SETS = ("app_crc", "module_crc")
STAMPS = ("log_id", "log_timestamp")
HOST = "x-csauth.ngs.nexon.com"
GAP_FACTOR = 2.0            # a heartbeat later than this many periods after the last one: a gap
PERIOD_S = 60.0             # the heartbeat period when no response said its auth_cycle
LABELS = {"0xC210F070": "system inventory", "0xC210F060": "drivers", "0xC210F040": "process",
          "0xC210F044": "process (lower-cased path)", "0xC210F04A": "process (signed vendor)",
          "0xC210F055": "memory page", "0xC210F080": "USB device", "0xC210F081": "USB device",
          "0xE210D001": "VPN suspicion", "0xE210F011": "window check failed"}


def answer_set(token) -> int | None:
    """Which of the 5 app_crc / module_crc answer sets a heartbeat presenting `token` answers (X_CSAUTH.md 4.1)."""
    return zlib.crc32(token.encode()) % 5 if isinstance(token, str) and token else None


def diff(old, new, path=""):
    """[{path, op: changed|added|removed, old, new}] between two JSON values: dicts key by key, anything else whole."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in list(old) + [k for k in new if k not in old]:
            p = f"{path}.{k}" if path else str(k)
            if k not in new:
                out.append({"path": p, "op": "removed", "old": old[k], "new": None})
            elif k not in old:
                out.append({"path": p, "op": "added", "old": None, "new": new[k]})
            else:
                out += diff(old[k], new[k], p)
        return out
    return [] if old == new else [{"path": path, "op": "changed", "old": old, "new": new}]


def _key(entry) -> str:
    return json.dumps({k: v for k, v in entry.items() if k not in STAMPS}, sort_keys=True) if isinstance(entry, dict) \
        else json.dumps(entry)


class Computer:
    def __init__(self, hwid):
        self.hwid = hwid
        self.ips, self.accounts, self.inventory = [], [], None
        self.events = []
        self.state = {}                      # top-level field -> last value sent
        self.answers = {}                    # answer set -> {"app_crc": ..., "module_crc": ...}
        self.seq_seen = {}                   # sequence id -> the second values it has had
        self.last_reports = {name: set() for name in REPORT_LISTS}
        self.resp_state = {}
        self.last_beat = None                # ts of the last /v2
        self.period = PERIOD_S
        self.session = None
        self.beat = 0

    def info(self):
        inv = self.inventory or {}
        dev, sys_ = inv.get("device_obj") or {}, inv.get("system_obj") or {}
        name = " · ".join(x for x in (dev.get("cpu"), dev.get("gpu"), (sys_.get("os") or "").strip(),
                                      (sys_.get("edition") or "").strip()) if x)
        return {"hwid": self.hwid, "name": name or self.state.get("os_version") or "", "ips": self.ips,
                "accounts": self.accounts, "events": len(self.events),
                "beats": sum(1 for e in self.events if e["kind"] == "v2")}


def _sequence(c: Computer, seq) -> list:
    """The sequence's news: ids joining or leaving, and a second value an id never had (ticks ignored)."""
    out = []
    old = c.state.get("sequence")
    if isinstance(old, dict) and isinstance(seq, dict):
        for k in seq:
            if k not in old:
                out.append({"path": f"sequence.{k}", "op": "added", "old": None, "new": seq[k]})
        for k in old:
            if k not in seq:
                out.append({"path": f"sequence.{k}", "op": "removed", "old": old[k], "new": None})
    if isinstance(seq, dict):
        for k, v in seq.items():
            if not (isinstance(v, list) and len(v) == 2):
                continue
            seen = c.seq_seen.setdefault(k, set())
            if seen and v[1] not in seen and isinstance(old, dict) and k in old:
                out.append({"path": f"sequence.{k}[1]", "op": "changed", "old": old[k][1], "new": v[1], "minor": True,
                            "note": "a value this module never had before (seen: %s)" % sorted(seen)})
            seen.add(v[1])
    return out


def _request(c: Computer, ev: dict, body: dict) -> None:
    changes, introduced = [], {}
    for k, v in body.items():
        if k == "csauthtoken" or k in REPORT_LISTS:
            continue
        if k in ANSWER_SETS:
            continue
        if k == "sequence":
            changes += _sequence(c, v)
            c.state[k] = v
            continue
        if k not in c.state:
            introduced[k] = v
        else:
            changes += diff(c.state[k], v, k)
        c.state[k] = v
    s = answer_set(body.get("csauthtoken"))
    if s is not None and any(k in body for k in ANSWER_SETS):
        ev["answer_set"] = s
        prev = c.answers.get(s)
        now = {k: body[k] for k in ANSWER_SETS if k in body}
        if prev is None:
            ev["answer_set_first"] = True
            if "app_crc" not in c.state:
                introduced.update(now)
        else:
            for ch in diff(prev, now):
                ch["note"] = "against the last heartbeat answering set %d" % s
                changes.append(ch)
        c.answers[s] = now
        for k in ANSWER_SETS:
            c.state.setdefault(k, body.get(k))
    reports, hidden, repeated = [], {"decoys": 0, "markers": 0}, 0
    for name in REPORT_LISTS:
        if name not in body:
            continue
        keys = set()
        for entry in body[name] if isinstance(body[name], list) else []:
            code = entry.get("log_code") if isinstance(entry, dict) else None
            if code == DECOY:
                hidden["decoys"] += 1
                continue
            if code == MARKER:
                hidden["markers"] += 1
                continue
            k = _key(entry)
            keys.add(k)
            if k in c.last_reports[name]:
                repeated += 1
                continue
            reports.append({"list": name, "code": code, "label": LABELS.get(code, ""), "entry": entry})
            if code == "0xC210F070" and isinstance(entry.get("detail"), dict):
                c.inventory = entry["detail"]
        c.last_reports[name] = keys
    ev.update(changes=changes, introduced=introduced, reports=reports, repeated=repeated, hidden=hidden)


def _response(c: Computer, ev: dict, body) -> None:
    out = {"changes": [], "flags": {}}
    if isinstance(body, dict):
        for k, v in body.items():
            if k == "csauthtoken":
                continue
            if k in ("config_changed", "emergency_stop"):
                if v:
                    out["flags"][k] = v
                continue
            if k == "detected":
                if v:
                    out["flags"][k] = v
                continue
            if k in c.resp_state:
                out["changes"] += diff(c.resp_state[k], v, k)
            c.resp_state[k] = v
        if isinstance(body.get("auth_cycle"), (int, float)) and body["auth_cycle"] > 0:
            c.period = float(body["auth_cycle"])
    ev["response"] = out


def _json(part):
    if part is None or not part.body:
        return None
    try:
        v = json.loads(part.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return v if isinstance(v, dict) else None


def timeline(cap) -> dict:
    """{"computers": [Computer.info() + {"events": [...]}], "unkeyed": [{client, connections, first, last}]} of a
    ready Capture (view.Capture)."""
    conns = {c["n"]: c for c in cap.conns}
    computers: dict[str, Computer] = {}
    for it in cap.items:
        if it["type"] != "http" or it["host"] != HOST:
            continue
        conn = conns.get(it["conn"], {})
        ip = conn.get("client", "").rsplit(":", 1)[0]
        req, resp = it["req"], it["resp"]
        body = _json(req)
        kind = {"/v1/issue": "issue", "/v2": "v2", "/v1/close": "close"}.get(it["path"].split("?")[0], it["path"])
        ev = {"id": it["id"], "conn": it["conn"], "ts": it["ts"], "kind": kind, "path": it["path"],
              "status": it["status"], "client": ip, "proto": it["proto"]}
        if body is None or not body.get("hwid"):
            ev.update(damaged="the request body isn't JSON (bytes lost in the capture?)" if req.body
                      else "no request body", changes=[], reports=[], quiet=False)
            c = next((x for x in computers.values() if ip in x.ips), None)
            (c or computers.setdefault("?", Computer("?"))).events.append(ev)
            continue
        c = computers.get(body["hwid"]) or computers.setdefault(body["hwid"], Computer(body["hwid"]))
        if ip and ip not in c.ips:
            c.ips.append(ip)
        if body.get("npacode") and body["npacode"] not in c.accounts:
            c.accounts.append(body["npacode"])
        ev["initial"] = not c.events
        if kind == "issue" or body.get("lifetoken") != c.session:
            c.session, c.beat = body.get("lifetoken"), 0
            if kind == "issue":
                c.last_beat = None           # a new session: the time since the last one's heartbeat is no gap
        if kind == "v2":
            c.beat += 1
            ev["beat"] = c.beat
            if c.last_beat is not None and ev["ts"] - c.last_beat > GAP_FACTOR * c.period:
                ev["gap"] = round(ev["ts"] - c.last_beat, 1)
            c.last_beat = ev["ts"]
        ev["session"] = (body.get("lifetoken") or "")[:8]
        _request(c, ev, body)
        _response(c, ev, _json(resp))
        if resp is None:
            ev["response"]["flags"]["no response"] = True
        elif resp.display != "json" and resp.body:
            ev["response"]["text"] = resp.body[:64].decode("utf-8", "replace")
        if ev["initial"]:
            ev["introduced"] = {}
            ev["changes"] = []
            ev["body"] = _clean(body)
        major = [ch for ch in ev["changes"] if not ch.get("minor")]
        ev["quiet"] = not (ev["initial"] or major or ev["introduced"] or ev["reports"] or ev.get("gap")
                           or ev["response"]["changes"] or ev["response"]["flags"] or kind != "v2")
        c.events.append(ev)
    unkeyed = {}
    for cn in cap.conns:
        if cn["sni"] == HOST and not cn["decrypted"]:
            ip = cn["client"].rsplit(":", 1)[0]
            u = unkeyed.setdefault(ip, {"client": ip, "connections": 0, "first": cn["start"], "last": cn["end"],
                                        "why": cn.get("status", "")})
            u["connections"] += 1
            u["first"], u["last"] = min(u["first"], cn["start"]), max(u["last"], cn["end"])
    out = []
    for c in sorted(computers.values(), key=lambda c: c.events[0]["ts"] if c.events else 0):
        d = c.info()
        d["events"] = c.events
        d["unkeyed"] = [u for ip, u in unkeyed.items() if ip in c.ips]
        out.append(d)
    known = {ip for c in computers.values() for ip in c.ips}
    return {"computers": out, "unkeyed": [u for ip, u in unkeyed.items() if ip not in known]}


def _clean(body: dict) -> dict:
    """A request body as the first event shows it: the decoys and the heartbeat marker out of its report lists."""
    out = dict(body)
    for name in REPORT_LISTS:
        if isinstance(out.get(name), list):
            out[name] = [e for e in out[name] if not (isinstance(e, dict) and e.get("log_code") in (DECOY, MARKER))]
    return out
