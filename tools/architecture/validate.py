"""Check an architecture map data file (tools/architecture/data/*.json) against SCHEMA.md and the code.

    python tools/architecture/validate.py tools/architecture/data/perception-eyes.json [more files...]
    python tools/architecture/validate.py --all          # every file in data/, plus cross-file checks

Errors (wrong shape, a path that doesn't exist) fail it; warnings (a name not found near its line, a line past the
file's end) are listed for fixing. Prints OK per file when there are no errors.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
LAYERS = {"foundation", "perception", "localization", "motion", "attack", "control", "behavior", "sim", "ui",
          "analysis", "shared", "runtime", "tests"}
KINDS = {"class", "function", "method", "constant", "dataclass", "enum", "protocol"}
MODES = {"read", "write", "read-write"}
NEAR = 4  # a key item's name must appear within this many lines of its line

_lines_cache: dict[str, list[str]] = {}


def file_lines(rel: str) -> list[str] | None:
    if rel not in _lines_cache:
        p = ROOT / rel
        if not p.is_file():
            return None
        _lines_cache[rel] = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return _lines_cache[rel]


class Report:
    def __init__(self, name):
        self.name, self.errors, self.warnings = name, [], []

    def err(self, where, msg):
        self.errors.append(f"{where}: {msg}")

    def warn(self, where, msg):
        self.warnings.append(f"{where}: {msg}")


def need(r, obj, where, key, typ, nonempty=False):
    if key not in obj:
        r.err(where, f"missing '{key}'")
        return None
    v = obj[key]
    if not isinstance(v, typ):
        r.err(where, f"'{key}' should be {typ.__name__ if isinstance(typ, type) else typ}, is {type(v).__name__}")
        return None
    if nonempty and not v:
        r.err(where, f"'{key}' is empty")
    return v


def check_path(r, where, rel, *, warn_only=False):
    if not isinstance(rel, str) or "\\" in rel or rel.startswith("/"):
        r.err(where, f"bad path {rel!r} (repo-relative, forward slashes)")
        return False
    base = rel.split("::")[0]
    if not (ROOT / base).exists():
        (r.warn if warn_only else r.err)(where, f"no such file {base!r}")
        return False
    return True


def name_near(lines, line, name):
    token = name.split(".")[-1].split("(")[0].strip()
    if not token:
        return True
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])")
    lo, hi = max(0, line - 1 - NEAR), min(len(lines), line + NEAR)
    return any(pat.search(l) for l in lines[lo:hi])


def check_symbol_line(r, where, path, name, line):
    lines = file_lines(path)
    if lines is None:
        return
    if not isinstance(line, int) or line < 1:
        r.err(where, f"line {line!r} should be a positive int")
        return
    if line > len(lines):
        r.warn(where, f"line {line} is past the end of {path} ({len(lines)} lines)")
        return
    if not name_near(lines, line, name):
        token = name.split(".")[-1]
        hits = [i + 1 for i, l in enumerate(lines)
                if re.search(r"(def|class)\s+" + re.escape(token) + r"\b|^\s*" + re.escape(token) + r"\s*[:=]", l)]
        hint = f" (found at {hits[:5]})" if hits else ""
        r.warn(where, f"'{name}' not found near line {line} of {path}{hint}")


def check_flow(r, where, f):
    need(r, f, where, "id", str, True)
    need(r, f, where, "title", str, True)
    need(r, f, where, "summary", str)
    steps = need(r, f, where, "steps", list, True) or []
    for i, s in enumerate(steps):
        w = f"{where}.steps[{i}]"
        if not isinstance(s, dict):
            r.err(w, "should be an object")
            continue
        m = need(r, s, w, "module", str, True)
        need(r, s, w, "text", str, True)
        if m:
            check_path(r, w, m)


def check_module(r, where, m):
    path = need(r, m, where, "path", str, True)
    if path and check_path(r, where, path):
        lines = file_lines(path)
        n = m.get("lines")
        if lines is not None and isinstance(n, int) and abs(n - len(lines)) > 2:
            r.warn(where, f"lines {n} but {path} has {len(lines)}")
    for k in ("title", "tagline", "summary"):
        need(r, m, where, k, str, True)
    need(r, m, where, "details_md", str)
    items = need(r, m, where, "key_items", list) or []
    for i, it in enumerate(items):
        w = f"{where}.key_items[{i}]"
        if not isinstance(it, dict):
            r.err(w, "should be an object")
            continue
        nm = need(r, it, w, "name", str, True)
        kind = need(r, it, w, "kind", str, True)
        if kind and kind not in KINDS:
            r.warn(w, f"kind {kind!r} not one of {sorted(KINDS)}")
        need(r, it, w, "summary", str, True)
        if nm and path and "line" in it:
            check_symbol_line(r, f"{w} {nm}", path, nm, it["line"])
        elif "line" not in it:
            r.warn(w, "no line")
    for k in ("inputs", "outputs", "uses", "gotchas", "tests"):
        v = m.get(k, [])
        if not isinstance(v, list):
            r.err(where, f"'{k}' should be a list")
    for u in m.get("uses", []) or []:
        check_path(r, f"{where}.uses", u)
    for t in m.get("tests", []) or []:
        check_path(r, f"{where}.tests", t, warn_only=True)
    for i, f in enumerate(m.get("files_io", []) or []):
        if not isinstance(f, dict) or "path" not in f:
            r.err(f"{where}.files_io[{i}]", "should be {path, mode, what}")
        elif f.get("mode") not in MODES:
            r.warn(f"{where}.files_io[{i}]", f"mode {f.get('mode')!r} not one of {sorted(MODES)}")
    for i, t in enumerate(m.get("tunables", []) or []):
        if not isinstance(t, dict) or not {"name", "value", "meaning"} <= set(t):
            r.err(f"{where}.tunables[{i}]", "should be {name, value, meaning}")
    for i, d in enumerate(m.get("decisions", []) or []):
        if not isinstance(d, dict) or not {"what"} <= set(d):
            r.err(f"{where}.decisions[{i}]", "should be {date, what, why}")


def check_slice(r, d):
    need(r, d, "file", "id", str, True)
    layer = need(r, d, "file", "layer", str, True)
    if layer and layer not in LAYERS:
        r.err("file", f"layer {layer!r} not one of {sorted(LAYERS)}")
    for k in ("title", "tagline", "summary", "details_md"):
        need(r, d, "file", k, str, True)
    mods = need(r, d, "file", "modules", list) or []
    paths = []
    for i, m in enumerate(mods):
        if not isinstance(m, dict):
            r.err(f"modules[{i}]", "should be an object")
            continue
        check_module(r, f"modules[{i}] {m.get('path', '?')}", m)
        paths.append(m.get("path"))
    dup = {p for p in paths if paths.count(p) > 1}
    if dup:
        r.err("modules", f"duplicate paths {sorted(dup)}")
    groups = need(r, d, "file", "groups", list) or []
    grouped = []
    for i, g in enumerate(groups):
        w = f"groups[{i}]"
        if not isinstance(g, dict):
            r.err(w, "should be an object")
            continue
        need(r, g, w, "id", str, True)
        need(r, g, w, "title", str, True)
        need(r, g, w, "summary", str)
        for p in need(r, g, w, "modules", list) or []:
            grouped.append(p)
            if p not in paths:
                r.err(w, f"module {p!r} is not in this file's modules")
    if mods and groups:
        missing = [p for p in paths if p not in grouped]
        if missing:
            r.warn("groups", f"modules in no group: {missing}")
        twice = {p for p in grouped if grouped.count(p) > 1}
        if twice:
            r.warn("groups", f"modules in several groups: {sorted(twice)}")
    for i, t in enumerate(d.get("data_types", []) or []):
        w = f"data_types[{i}]"
        if not isinstance(t, dict):
            r.err(w, "should be an object")
            continue
        nm = need(r, t, w, "name", str, True)
        mp = need(r, t, w, "module", str, True)
        need(r, t, w, "summary", str, True)
        if mp and check_path(r, w, mp) and nm and "line" in t:
            check_symbol_line(r, f"{w} {nm}", mp, nm, t["line"])
    for i, f in enumerate(d.get("flows", []) or []):
        check_flow(r, f"flows[{i}]", f)
    if not isinstance(d.get("not_done", []), list):
        r.err("file", "'not_done' should be a list")


def check_runtime_extras(r, d):
    for i, p in enumerate(d.get("processes", []) or []):
        w = f"processes[{i}]"
        for k in ("id", "name", "summary"):
            need(r, p, w, k, str, True)
        for m in p.get("modules", []) or []:
            check_path(r, w, m)
    for i, f in enumerate(d.get("lab_files", []) or []):
        w = f"lab_files[{i}]"
        need(r, f, w, "path", str, True)
        need(r, f, w, "purpose", str, True)
        for m in (f.get("written_by", []) or []) + (f.get("read_by", []) or []):
            check_path(r, w, m)
    for i, c in enumerate(d.get("cli", []) or []):
        w = f"cli[{i}] {c.get('command', '?')}"
        need(r, c, w, "command", str, True)
        need(r, c, w, "summary", str, True)
        e = c.get("entry")
        if e and check_path(r, w, e) and c.get("symbol") and "line" in c:
            check_symbol_line(r, w, e, c["symbol"], c["line"])


def check_file(path: Path) -> Report:
    r = Report(path.name)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        r.err("file", f"not valid JSON: {e}")
        return r
    if not isinstance(d, dict):
        r.err("file", "top level should be an object")
        return r
    if d.get("id") != path.stem:
        r.err("file", f"id {d.get('id')!r} should be the file name {path.stem!r}")
    if path.stem == "flows":
        for i, f in enumerate(need(r, d, "file", "flows", list, True) or []):
            check_flow(r, f"flows[{i}]", f)
        return r
    if path.stem == "glossary":
        for i, g in enumerate(need(r, d, "file", "glossary", list, True) or []):
            need(r, g, f"glossary[{i}]", "term", str, True)
            need(r, g, f"glossary[{i}]", "meaning", str, True)
            for p in g.get("see", []) or []:
                check_path(r, f"glossary[{i}].see", p, warn_only=True)
        for i, t in enumerate(d.get("timeline", []) or []):
            need(r, t, f"timeline[{i}]", "date", str, True)
            need(r, t, f"timeline[{i}]", "what", str, True)
        return r
    check_slice(r, d)
    if d.get("layer") == "runtime":
        check_runtime_extras(r, d)
    if d.get("layer") == "tests":
        for i, t in enumerate(d.get("test_index", []) or []):
            check_path(r, f"test_index[{i}]", t.get("path", ""), warn_only=True)
    return r


def cross_checks(files: list[Path]) -> list[str]:
    owners: dict[str, list[str]] = {}
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for m in d.get("modules", []) or []:
            owners.setdefault(m.get("path"), []).append(p.stem)
    return [f"{path} is in several files: {who}" for path, who in owners.items() if len(who) > 1]


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    files = sorted(DATA.glob("*.json")) if argv == ["--all"] else [Path(a) for a in argv]
    bad = 0
    for f in files:
        r = check_file(f)
        status = "OK" if not r.errors else "FAILED"
        print(f"{f.name}: {status} ({len(r.errors)} errors, {len(r.warnings)} warnings)")
        for e in r.errors:
            print(f"  ERROR {e}")
        for w in r.warnings:
            print(f"  warn  {w}")
        bad += bool(r.errors)
    if argv == ["--all"]:
        for c in cross_checks(files):
            print(f"  cross {c}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
