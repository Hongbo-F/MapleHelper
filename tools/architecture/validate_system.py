"""Check the runtime-system data (tools/architecture/data/system/*.json): its shape, and that every piece of evidence
names a real file and a line where its symbol is.

    python tools/architecture/validate_system.py tools/architecture/data/system/input.json
    python tools/architecture/validate_system.py --all

Each file (see SYSTEM.md) is {"id", "summary_md", ...lists}; every item that makes a claim about the code carries
"evidence": [{"path": "gamelab/control/player.py", "line": 1023, "symbol": "Player._press"}] (symbol may be "" for a
line that's a statement). Process ids must be among those named in entrypoints.json (or the fixed outside ones).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate  # noqa: E402  (its file and line checks)

DIR = Path(__file__).resolve().parent / "data" / "system"
OUTSIDE = {"game", "user", "windows", "babysitter", "phone", "cloudflare", "claude"}
LISTS = {
    "entrypoints": {"processes": ["id", "name", "kind", "command", "started_by", "how_started", "summary"],
                    "spawns": ["parent", "child", "how", "when"]},
    "channels": {"channels": ["id", "from", "to", "transport", "name", "carries"]},
    "readers": {"readers": ["process", "what", "how", "purpose", "when"]},
    "input": {"senders": ["process", "api", "what", "conditions"],
              "plan_sources": ["process", "module", "what"],
              "path_steps": ["process", "text"]},
    "sequences": {"sequences": ["id", "title", "summary", "steps"]},
    "components": {"processes": ["process", "is_os_process", "plain_md", "components"]},
}


def process_ids() -> set[str]:
    try:
        d = json.loads((DIR / "entrypoints.json").read_text(encoding="utf-8"))
        return {p["id"] for p in d.get("processes", [])} | OUTSIDE
    except Exception:  # noqa: BLE001
        return set()


def check_evidence(r, where, ev):
    if not isinstance(ev, list):
        r.err(where, "'evidence' should be a list")
        return
    for i, e in enumerate(ev):
        w = f"{where}.evidence[{i}]"
        if not isinstance(e, dict) or "path" not in e:
            r.err(w, "should be {path, line, symbol}")
            continue
        if validate.check_path(r, w, e["path"]) and e.get("line") is not None:
            sym = e.get("symbol") or ""
            if sym:
                validate.check_symbol_line(r, w, e["path"], sym, e["line"])
            else:
                lines = validate.file_lines(e["path"])
                if lines is not None and (not isinstance(e["line"], int) or not 1 <= e["line"] <= len(lines)):
                    r.warn(w, f"line {e['line']} is not in {e['path']} ({len(lines)} lines)")


def check(path: Path, procs: set[str]) -> validate.Report:
    r = validate.Report(path.name)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        r.err("file", f"not valid JSON: {e}")
        return r
    if d.get("id") != path.stem or path.stem not in LISTS:
        r.err("file", f"id should be the file name, one of {sorted(LISTS)}")
        return r
    validate.need(r, d, "file", "summary_md", str, True)
    for key, fields in LISTS[path.stem].items():
        items = validate.need(r, d, "file", key, list, True) or []
        for i, it in enumerate(items):
            w = f"{key}[{i}]"
            if not isinstance(it, dict):
                r.err(w, "should be an object")
                continue
            for f in fields:
                if f not in it or it[f] in (None, ""):
                    r.err(w, f"missing '{f}'")
            if "evidence" in it:
                check_evidence(r, w, it["evidence"])
            elif key not in ("sequences",):
                r.warn(w, "no evidence")
            for pk in ("process", "from", "to", "parent", "child"):
                v = it.get(pk)
                for one in (v if isinstance(v, list) else [v]) if v else []:
                    if procs and one not in procs:
                        r.warn(w, f"{pk} {one!r} is not a process id in entrypoints.json")
            if key == "processes":
                for s in it.get("started_by", []) if isinstance(it.get("started_by"), list) else [it.get("started_by")]:
                    if procs and s not in procs:
                        r.warn(w, f"started_by {s!r} is not a process id")
            if key == "processes" and path.stem == "components":
                for j, c in enumerate(it.get("components", []) or []):
                    cw = f"{w}.components[{j}]"
                    for f in ("id", "name", "runs_as", "started_by", "does"):
                        if not c.get(f):
                            r.err(cw, f"missing '{f}'")
                    if "evidence" in c:
                        check_evidence(r, cw, c["evidence"])
                    else:
                        r.warn(cw, "no evidence")
            if key == "sequences":
                for j, st in enumerate(it.get("steps", []) or []):
                    sw = f"{w}.steps[{j}]"
                    for f in ("from", "to", "what"):
                        if not st.get(f):
                            r.err(sw, f"missing '{f}'")
                    for pk in ("from", "to"):
                        if procs and st.get(pk) and st[pk] not in procs:
                            r.warn(sw, f"{pk} {st[pk]!r} is not a process id")
                    if "evidence" in st:
                        check_evidence(r, sw, st["evidence"])
    return r


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    files = sorted(DIR.glob("*.json")) if argv == ["--all"] else [Path(a) for a in argv]
    procs, bad = process_ids(), 0
    for f in files:
        r = check(f, procs)
        print(f"{f.name}: {'OK' if not r.errors else 'FAILED'} ({len(r.errors)} errors, {len(r.warnings)} warnings)")
        for e in r.errors:
            print(f"  ERROR {e}")
        for w in r.warnings:
            print(f"  warn  {w}")
        bad += bool(r.errors)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
