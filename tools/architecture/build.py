"""Build the architecture map: tools/architecture/data/*.json (written by hand or by agents, see SCHEMA.md) plus facts
read straight from the code (each module's real imports and importers, line counts, the layer rules of
tests/test_architecture.py) -> one self-contained HTML page.

    python tools/architecture/build.py                       # gamelab/lab/architecture/index.html (+ artifact.html)
    python tools/architecture/build.py --out some/dir

index.html is a whole document (for a browser or the web UI); artifact.html is the same page without the document
skeleton, for publishing as a claude.ai Artifact. The built pages stay out of git (gamelab/lab is ignored).
"""

import argparse
import ast
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
TEMPLATE = HERE / "page.html"
OVERVIEW = HERE / "overview.md"
DEFAULT_OUT = ROOT / "gamelab" / "lab" / "architecture"
CODE_ROOTS = ("gamelab", "maplehelper", "gamedata", "tools/review_pages")

# The layers in the order the page shows them, with the README's one-line description of each.
LAYERS = [
    ("runtime", "Runtime", "The command line and the processes it starts: who runs, who's elevated, how they talk"),
    ("behavior", "Behavior", "What to do next: the hunter (the agent), fights, looting, tours, safety"),
    ("attack", "Attack", "Which monster, from where, and how: pure decisions"),
    ("control", "Control", "Decisions turned into keys: actions, one move, a route; the elevated key player"),
    ("motion", "Motion", "Getting from A to B: physics, routes, chained key timelines (pure planning)"),
    ("localization", "Localization", "Where she, the monsters and the drops are on the map"),
    ("perception", "Perception", "Pixels to what's on screen, with no map knowledge"),
    ("sim", "Simulator", "An offline game: the whole stack runs against it with no game and no keys"),
    ("analysis", "Analysis", "Measuring combos, movement and the hunt from recordings and telemetry"),
    ("ui", "UI", "Watching a run: the mini-map window, the web page, review pages"),
    ("tests", "Tests", "The checks: layer rules, the simulator's goldens, the test data"),
    ("shared", "Shared", "What every layer may use: telemetry, the game window, capture, native C, her profile"),
    ("foundation", "Foundation", "MapleHelper, the combo helper gamelab grew from, and the game data reader"),
]


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def layer_rules():
    """ALLOWED and SHARED from tests/test_architecture.py, so the page can't drift from the test."""
    tree = ast.parse((ROOT / "tests" / "test_architecture.py").read_text(encoding="utf-8"))
    allowed, shared = {}, []
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name == "ALLOWED" and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                allowed[k.value] = sorted(e.value for e in v.elts) if isinstance(v, ast.Set) else []
        elif name == "SHARED" and isinstance(node.value, ast.Set):
            shared = sorted(e.value for e in node.value.elts)
    return allowed, shared


def layer_of(path: str) -> str:
    parts = path.split("/")
    if parts[0] in ("maplehelper", "gamedata"):
        return "foundation"
    if parts[0] == "tests":
        return "tests"
    if parts[0] == "tools":
        return "ui"
    if parts[0] == "gamelab":
        if len(parts) == 2:
            return "runtime" if parts[1] == "__main__.py" else "shared"
        if parts[1] == "classes":
            return "shared"
        return parts[1]
    return "shared"


def resolve(dotted: list[str], files: set[str]) -> str | None:
    base = "/".join(dotted)
    for cand in (f"{base}.py", f"{base}/__init__.py"):
        if cand in files:
            return cand
    return None


def imports_of(path: str, files: set[str]) -> list[dict]:
    """The repo's own modules a file imports: [{path, lazy}] (lazy: only inside a function)."""
    src = (ROOT / path).read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    package = path.split("/")[:-1]
    found: dict[str, bool] = {}

    def add(target, lazy):
        if target and target != path:
            found[target] = found.get(target, True) and lazy

    def visit(node, lazy):
        for child in ast.iter_child_nodes(node):
            child_lazy = lazy or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.ImportFrom):
                if child.level:
                    base = package[:len(package) - (child.level - 1)]
                else:
                    base = []
                mod = base + (child.module.split(".") if child.module else [])
                if not mod or mod[0] not in ("gamelab", "maplehelper", "gamedata", "tools"):
                    continue
                for a in child.names:
                    sub = resolve(mod + [a.name], files)  # `from . import review`: a submodule
                    add(sub or resolve(mod, files), child_lazy)
            elif isinstance(child, ast.Import):
                for a in child.names:
                    parts = a.name.split(".")
                    if parts[0] in ("gamelab", "maplehelper", "gamedata"):
                        while parts and not resolve(parts, files):
                            parts = parts[:-1]
                        if parts:
                            add(resolve(parts, files), child_lazy)
            visit(child, child_lazy)

    visit(tree, False)
    return [{"path": p, "lazy": lazy} for p, lazy in sorted(found.items())]


def code_facts():
    tracked = [p for p in git("ls-files").splitlines()
               if p.startswith(CODE_ROOTS) and not p.startswith("gamelab/assets/")]
    files = set(tracked)
    facts = {}
    for p in tracked:
        if not p.endswith((".py", ".c", ".hlsl", ".js", ".css", ".html", ".ps1")):
            continue
        try:
            n = len((ROOT / p).read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            continue
        facts[p] = {"lines": n, "layer": layer_of(p), "imports": [], "imported_by": []}
    for p in facts:
        if p.endswith(".py"):
            facts[p]["imports"] = imports_of(p, files)
    for p, f in facts.items():
        for imp in f["imports"]:
            if imp["path"] in facts:
                facts[imp["path"]]["imported_by"].append({"path": p, "lazy": imp["lazy"]})
    edges = {}
    for p, f in facts.items():
        for imp in f["imports"]:
            a, b = f["layer"], facts.get(imp["path"], {}).get("layer")
            if b and a != b:
                edges[(a, b)] = edges.get((a, b), 0) + 1
    layer_edges = [{"from": a, "to": b, "count": n} for (a, b), n in sorted(edges.items())]
    # The tests: counted for the layer, and each module's test files read off their imports.
    test_files = [p for p in git("ls-files", "tests").splitlines() if p.endswith(".py")]
    tests = {"files": len(test_files), "lines": 0}
    for t in test_files:
        tests["lines"] += len((ROOT / t).read_text(encoding="utf-8", errors="replace").splitlines())
        for imp in imports_of(t, files):
            if imp["path"] in facts:
                facts[imp["path"]].setdefault("tested_by", []).append(t)
    return facts, layer_edges, tests


def load(name):
    p = DATA / f"{name}.json"  # DATA may be swapped by --data
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build_data():
    allowed, shared = layer_rules()
    facts, layer_edges, test_stats = code_facts()
    slices, modules = [], {}
    special = {"flows", "glossary"}
    for p in sorted(DATA.glob("*.json")):
        if p.stem in special:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        mods = d.get("modules", []) or []
        for m in mods:
            f = facts.get(m["path"], {})
            m["slice"] = d["id"]
            m["layer"] = d.get("layer") if d.get("layer") != "runtime" else "runtime"
            m["code"] = {"lines": f.get("lines", m.get("lines")), "imports": f.get("imports", []),
                         "imported_by": f.get("imported_by", []), "tested_by": f.get("tested_by", [])}
            modules[m["path"]] = m
        d["module_paths"] = [m["path"] for m in mods]
        d.pop("modules", None)
        slices.append(d)
    # Code the agents didn't describe still shows, with what the code says.
    unmapped = sorted(p for p in facts if p not in modules and not p.startswith("gamelab/ui/tls/static/x"))
    flows = (load("flows") or {}).get("flows", [])
    gl = load("glossary") or {}
    runtime = next((s for s in slices if s.get("layer") == "runtime"), {})
    tests = next((s for s in slices if s.get("layer") == "tests"), {})
    layer_stats = {}
    for p, f in facts.items():
        s = layer_stats.setdefault(f["layer"], {"files": 0, "lines": 0})
        s["files"] += 1
        s["lines"] += f["lines"]
    layer_stats["tests"] = test_stats
    system = {}
    for p in sorted((DATA / "system").glob("*.json")):
        system[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    parts = sorted((DATA / "system" / "parts").glob("*/components.json"))
    if parts and "components" not in system:
        system["components"] = {"id": "components", "summary_md": "", "processes": [
            proc for part in parts for proc in json.loads(part.read_text(encoding="utf-8")).get("processes", [])]}
    head = git("log", "-1", "--format=%h %ad", "--date=short").split()
    return {
        "generated": dt.date.today().isoformat(),
        "commit": head[0] if head else "",
        "commit_date": head[1] if len(head) > 1 else "",
        "overview_md": OVERVIEW.read_text(encoding="utf-8") if OVERVIEW.exists() else "",
        "layers": [{"id": i, "title": t, "blurb": b, "allowed": allowed.get(i), "stats": layer_stats.get(i, {})}
                   for i, t, b in LAYERS],
        "shared_modules": shared,
        "layer_edges": layer_edges,
        "slices": slices,
        "modules": modules,
        "unmapped": [{"path": p, **facts[p]} for p in unmapped],
        "flows": flows,
        "glossary": gl.get("glossary", []),
        "timeline": gl.get("timeline", []),
        "processes": runtime.get("processes", []),
        "lab_files": runtime.get("lab_files", []),
        "cli": runtime.get("cli", []),
        "test_index": tests.get("test_index", []),
        "system": system,
        "system_intro_md": (HERE / "system_intro.md").read_text(encoding="utf-8") if (HERE / "system_intro.md").exists() else "",
        "totals": {"files": len(facts), "lines": sum(f["lines"] for f in facts.values()),
                   "described": len(modules)},
    }


def main(argv=None):
    global DATA
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--data", type=Path, default=None, help="the data folder (default: tools/architecture/data)")
    args = ap.parse_args(argv)
    if args.data:
        DATA = args.data
    data = build_data()
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text(encoding="utf-8").replace("__ARCH_DATA__", blob)
    head, _, body = page.partition("<!--BODY-->")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "artifact.html").write_text(head + body, encoding="utf-8")
    (args.out / "index.html").write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + head + "</head>\n<body>\n" + body + "</body>\n</html>\n", encoding="utf-8")
    t = data["totals"]
    print(f"{args.out / 'index.html'}: {len(data['slices'])} slices, {t['described']} modules described of "
          f"{t['files']} files ({t['lines']:,} lines), {len(data['flows'])} flows, {len(data['glossary'])} terms, "
          f"system: {sorted(data['system'])}, "
          f"{len(blob) / 1e6:.2f} MB of data")
    if data["unmapped"]:
        print(f"  not described yet: {len(data['unmapped'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
