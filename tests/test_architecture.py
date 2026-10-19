"""gamelab's layers only import downwards (see gamelab/README.md): perception knows nothing of the map,
motion planning nothing of fighting, and so on. Checked by reading every module's imports. And the key
player, which runs as administrator, loads as little as it did before the layers."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "gamelab"
SHARED = {"paths", "screen", "telemetry", "window"}
# Each layer and the gamelab layers it may import (besides itself and SHARED).
ALLOWED = {
    "perception": set(),
    "localization": {"perception"},
    "motion": {"localization"},
    "attack": {"perception", "localization", "motion"},
    "control": {"perception", "localization", "motion"},
    "behavior": {"perception", "localization", "motion", "attack", "control"},
    "ui": {"localization", "motion", "attack"},
    "analysis": {"perception", "control"},
    "sim": {"perception", "localization", "motion", "attack", "control", "behavior", "analysis"},
}


def gamelab_imports(path: Path) -> set[str]:
    """The gamelab packages and top-level modules a file imports, relative or absolute."""
    package = ["gamelab", *path.relative_to(PKG).parent.parts]  # the file's own package
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - (node.level - 1)]
            elif node.module and node.module.split(".")[0] == "gamelab":
                base = []
            else:
                continue
            module = base + (node.module.split(".") if node.module else [])
            for full in ([module] if node.module else [module + [a.name] for a in node.names]):
                if len(full) >= 2 and full[0] == "gamelab":
                    out.add(full[1])
        elif isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] == "gamelab" and len(parts) >= 2:
                    out.add(parts[1])
    return out


LAYER_FILES = [(layer, f) for layer in ALLOWED for f in sorted((PKG / layer).rglob("*.py"))]


@pytest.mark.parametrize("layer,path", LAYER_FILES, ids=[f"{l}/{p.name}" for l, p in LAYER_FILES])
def test_layers_only_import_downwards(layer, path):
    used = gamelab_imports(path) - SHARED - {layer}
    assert used <= ALLOWED[layer], f"{path.relative_to(PKG)} imports {sorted(used - ALLOWED[layer])}"


def test_every_module_belongs_to_a_layer():
    top = {p.stem for p in PKG.glob("*.py")} - {"__init__", "__main__"}
    assert top <= SHARED, f"modules outside the layers: {sorted(top - SHARED)}"


def test_the_elevated_player_loads_only_what_it_needs():
    """`python -m gamelab player` runs as administrator and loads its code once: it goes through the
    command line (gamelab.__main__), then the player, and nothing of the stack (fights, navigation,
    planning, the simulator, the mini-map) or the game data may come along."""
    code = ("import sys, json; import gamelab.__main__ as m; m.build_parser(); "
            "import gamelab.control.player, gamelab.control.alarm, gamelab.perception.lie_detector; "
            "print(json.dumps(sorted(m for m in sys.modules if m.startswith(('gamelab', 'gamedata', 'tkinter')))))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=PKG.parent).stdout
    loaded = set(json.loads(out))
    stack = ("gamelab.behavior", "gamelab.motion", "gamelab.localization", "gamelab.attack", "gamelab.ui",
             "gamelab.sim", "gamedata", "tkinter")
    assert not [m for m in loaded if m.startswith(stack)], sorted(loaded)
