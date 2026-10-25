"""Screen geometry (gamelab.screen): every screen-space measure follows the game window's client size. At 1920x1080
everything is exactly as it was before the geometry was carried (the saved screenshots, the simulator and the golden
traces depend on it, to the bit); at the game's own 1366x768 the game is drawn pixel for pixel."""

import ast
from pathlib import Path

import numpy as np
import pytest

from gamelab import screen
from gamelab.screen import NATIVE, STRETCHED, Geometry, geometry, of_frame

PKG = Path(__file__).resolve().parents[1] / "gamelab"


def test_the_1920x1080_window_is_what_the_constants_were():
    assert STRETCHED == Geometry((1920, 1080), 2)
    assert screen.GAME_SIZE == (1920, 1080)
    assert screen.SCREEN_SCALE == 1080 / 768 == 1.40625
    assert screen.SCALE == 2
    assert screen.UNITS_PER_HALF_PX == 2 / (1080 / 768)
    assert STRETCHED.stretch_x == 1920 / 1366  # exp_number's SX
    assert STRETCHED.frame_size == (960, 540) and STRETCHED.suffix == "" and not STRETCHED.native


@pytest.mark.parametrize("n", [0, 1, 2.5, 6, 7, 12, 16, 22, 40, 52, 120, 300, 1080, -24, 1.5, 0.703125, 1e-3])
def test_converting_1920x1080_px_to_1920x1080_changes_nothing_to_the_bit(n):
    assert STRETCHED.px(n) == n and STRETCHED.fpx(n) == n
    assert STRETCHED.ipx(n) == round(n) and STRETCHED.ifpx(n) == round(n)
    assert STRETCHED.from_game(STRETCHED.to_game(n)) == pytest.approx(n, abs=1e-12)


def test_the_games_own_size_is_drawn_pixel_for_pixel():
    assert NATIVE == geometry(1366, 768) and NATIVE.native and NATIVE.suffix == "_1366x768"
    assert NATIVE.stretch == NATIVE.stretch_x == 1.0
    assert NATIVE.look_scale == screen.LOOK_SCALES[(1366, 768)]
    assert NATIVE.frame_scale == 1 / NATIVE.look_scale and NATIVE.units_per_px == NATIVE.look_scale
    assert NATIVE.px(1.40625) == 1.0 and NATIVE.px(300) == pytest.approx(213.33, abs=0.01)
    assert NATIVE.to_game(10.0) == NATIVE.from_game(10.0) == 10.0
    half = geometry(1366, 768, 2)
    assert half.frame_size == (683, 384) and half.units_per_px == 2.0 and half.fpx(8) == pytest.approx(5.69, abs=0.01)
    full = geometry(1366, 768, 1)
    assert full.frame_size == (1366, 768) and full.fpx(8) == pytest.approx(11.38, abs=0.01)


def test_a_game_px_is_where_the_stretch_puts_its_middle():
    # At 1920x1080 game px g's middle is at (g + 0.5) * 1.40625 - 0.5 on screen (bilinear filtering): DOT_ORIGIN
    # (65.1, 91.3) is game px (46.15, 64.78).
    assert STRETCHED.to_game(65.1) == pytest.approx(46.149, abs=1e-3)
    assert NATIVE.from_game(STRETCHED.to_game(91.3)) == pytest.approx(64.78, abs=1e-3)


def test_a_frame_says_its_geometry():
    assert of_frame(np.zeros((1080, 1920, 3), np.uint8)) == STRETCHED
    assert of_frame(np.zeros((768, 1366, 3), np.uint8)) == NATIVE
    assert of_frame(np.zeros((768, 1366, 3), np.uint8), 2).frame_size == (683, 384)
    assert geometry(1600, 900).look_scale == 2  # a size the stack doesn't know: half size, stretched 1.17


def test_the_exp_number_box_at_each_size():
    from gamelab.perception import exp_number
    assert exp_number.READ_BOX == exp_number.read_box(STRETCHED) == (1052, 1038, 1191, 1054)
    assert exp_number.read_box(NATIVE) == (748, 738, 847, 750)
    assert exp_number.read(np.zeros((3, 3, 3), np.uint8)).why == "no EXP number box"


def test_the_minimaps_sizes_at_1920x1080_are_its_constants():
    from gamelab.perception import minimap as m
    px = m.sizes(STRETCHED)
    assert px.box == m.MINIMAP_BOX and px.dot_area == m.DOT_AREA and px.dot_size == m.DOT_SIZE
    assert px.dot_full == m.DOT_FULL and px.dot_half == (m.DOT_FULL - 0.5) / 2
    assert px.marker_area == m.MARKER_AREA and px.marker_side == m.MARKER_SIDE and px.marker_fill == 0
    assert px.ring is m._RING and px.other_shift == m.OTHER_SHIFT
    native = m.sizes(NATIVE)
    assert native.box == (0, 68, 213, 213) and native.dot_full == 6 and native.dot_half == 2.5
    assert native.other_shift == 0.5 and native.marker_fill == m.MARKER_FILL


def test_rope_kernels_at_1920x1080_are_as_they_were():
    from gamelab.perception import terrain
    close, open_ = terrain._kernels(STRETCHED)
    assert close.shape == (9, 3) and open_.shape == (31, 1)
    close, open_ = terrain._kernels(NATIVE)
    assert close.shape == (6, 2) and open_.shape == (22, 1)


def test_templates_shrink_by_whole_steps_as_before():
    from gamelab.perception.mobs import frame_size
    assert frame_size(61, 37, 2.0) == (30, 18)  # half size: w // 2, as before
    assert frame_size(61, 37, 1.0) == (61, 37)
    assert frame_size(61, 37, 1.40625) == (43, 26)  # a 1920x1080 cut-out in a 1366x768 frame
    assert frame_size(1, 1, 2.0) == (1, 1)


def test_the_localizer_moves_its_origin_to_the_same_game_px():
    from gamelab.localization import mapdata
    from gamelab.localization.pose import DOT_ORIGIN, Localizer
    tree = {"foothold": {}, "miniMap": {"width": 1948, "height": 2152, "centerX": 479, "centerY": 1387, "mag": 4}}
    model = mapdata.parse(1, tree, "Test", {})
    loc = Localizer(model)
    assert loc.origin == DOT_ORIGIN and loc.per_px == 16 / 1.40625
    at_1920 = loc.to_map((120.0, 200.0))
    loc.use(NATIVE)
    assert loc.per_px == 16.0 and loc.origin == pytest.approx((46.149, 64.78), abs=1e-3)
    # The same game px of the dot at both sizes is the same place on the map
    dot = (STRETCHED.to_game(120.0), STRETCHED.to_game(200.0))
    assert loc.to_map(dot) == pytest.approx(at_1920, abs=1e-9)
    loc.use(STRETCHED)
    assert loc.origin == pytest.approx(DOT_ORIGIN, abs=1e-12)


def test_the_camera_puts_map_units_on_screen_at_the_windows_stretch():
    from gamelab.localization.pose import Camera
    cam = Camera((-1000, -1000, 1000, 1000))
    at = (0.0, 0.0)
    x1920 = cam.to_screen(100.0, 50.0, at)
    xn = cam.to_screen(100.0, 50.0, at, NATIVE)
    assert x1920 == pytest.approx((xn[0] * 1.40625, xn[1] * 1.40625))


# Modules that read the game go by the geometry their frames carry: the 1920x1080 window's numbers (screen.GAME_SIZE,
# SCREEN_SCALE, SCALE, UNITS_PER_HALF_PX) are for the simulator, which draws its world at that size.
ALIASES = {"GAME_SIZE", "SCREEN_SCALE", "SCALE", "UNITS_PER_HALF_PX"}


def _alias_users() -> list[str]:
    out = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG)
        if rel.parts[0] == "sim" or rel.name == "screen.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("screen"):
                used = {a.name for a in node.names} & ALIASES
                if used:
                    out.append(f"{rel}: {sorted(used)}")
    return out


def test_nothing_but_the_simulator_assumes_the_1920x1080_window():
    assert _alias_users() == []


# That alone let three things slip (the review of 2026-10-24): `map` placing her dot with Localizer(model), its
# geometry left at 1920x1080's by default; bodies making its own UNITS_PER_HALF_PX from STRETCHED; and looting comparing
# her ground with MIN_GROUND_PX as it is. So three more checks, over gamelab outside the simulator, each with the
# places that mean 1920x1080 on purpose and why: calls that leave a `geo=STRETCHED` default (resolved through each
# file's imports; a method called on an object isn't), module-level names made from STRETCHED's numbers, and *_PX
# constants (px of the 1920x1080 window) used other than through a geometry's conversion.
DEFAULT_ON_PURPOSE = {
    ("__main__.py", "load"): "_calibration: at 1920x1080, the calibration's files as they always were",
    ("localization/worldstate.py", "WorldState"): "a frame not visible: nothing was read off it",
    ("localization/service.py", "WorldState"): "no game window seen yet: nothing was read, and no geometry is known",
    ("perception/exp_number.py", "read_box"): "READ_BOX: 1920x1080's box, named so",
}
STRETCHED_ON_PURPOSE = {
    ("analysis/pasted.py", "STRETCH_TO"): "the scene stretched as the 1920x1080 window showed the game",
    ("analysis/respawn.py", "VIEW"): "scene records of before 2026-10-21's evening, all of the 1920x1080 window",
    ("analysis/respawn.py", "MINIMAP"): "the same",
    ("localization/pose.py", "VIEW"): "map units on screen: the game's view, the same at any size",
    ("perception/exp_number.py", "SX"): "1920x1080's stretch, for BAR's rows measured there (bar_rows)",
    ("perception/exp_number.py", "SY"): "the same",
    ("perception/minimap.py", "OTHER_SHIFT"): "1920x1080's (minimap.sizes(geo).other_shift at any size)",
    ("perception/mobs.py", "SEARCH_SCALE"): "step 1's scale at any size, where its numbers were measured",
}
PX_ON_PURPOSE = {
    ("localization/senses.py", "ROPE_MATCH_PX"): "ROPE_PAD, a sum of px converted where it's used",
    ("localization/tracking.py", "EDGE_PX"): "View.edge's default, 1920x1080's; worldstate sets it per look",
}


def _modules():
    """gamelab's modules outside the simulator: (dotted name, whether it's a package, path from gamelab, tree)."""
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG)
        if rel.parts[0] == "sim":
            continue
        parts = ("gamelab",) + rel.with_suffix("").parts
        pkg = parts[-1] == "__init__"
        yield ".".join(parts[:-1] if pkg else parts), pkg, rel.as_posix(), ast.parse(path.read_text(encoding="utf-8"))


def _imports(module: str, pkg: bool, tree: ast.Module, names: set[str]) -> dict:
    """A file's local names from its imports (anywhere in it): ("module", dotted) or ("symbol", (module, name))."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            src = node.module or ""
            if node.level:
                base = module.split(".") if pkg else module.split(".")[:-1]
                src = ".".join(base[:len(base) - node.level + 1] + ([node.module] if node.module else []))
            for a in node.names:
                full = f"{src}.{a.name}"
                out[a.asname or a.name] = ("module", full) if full in names else ("symbol", (src, a.name))
        elif isinstance(node, ast.Import):
            out.update({a.asname: ("module", a.name) for a in node.names if a.asname})
    return out


def _stretched_default(value) -> bool:
    """Whether a default is STRETCHED (or a dataclass field(default=STRETCHED))."""
    if isinstance(value, ast.Call) and getattr(value.func, "id", "") == "field":
        value = next((k.value for k in value.keywords if k.arg == "default"), None)
    return isinstance(value, ast.Name) and value.id == "STRETCHED"


def _geo_param(fn: ast.FunctionDef, method: bool):
    """A function's `geo` defaulting to STRETCHED: its index among the positional parameters (None: keyword only),
    or False if it hasn't one."""
    a = fn.args
    pos = [p.arg for p in a.posonlyargs + a.args][1 if method else 0:]
    if "geo" in pos:
        i = pos.index("geo")
        k = i - (len(pos) - len(a.defaults))
        return i if k >= 0 and _stretched_default(a.defaults[k]) else False
    kw = [p.arg for p in a.kwonlyargs]
    return None if "geo" in kw and _stretched_default(a.kw_defaults[kw.index("geo")]) else False


def _geo_takers(mods) -> dict:
    """(module, name) of every top-level function and class taking a geo that defaults to STRETCHED -> geo's
    positional index (None: by keyword only), dataclasses' fields and their bases' included."""
    out = {}
    for module, _, _, tree in mods:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and (i := _geo_param(node, False)) is not False:
                out[(module, node.name)] = i
            elif isinstance(node, ast.ClassDef):
                init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
                if init is not None:
                    if (i := _geo_param(init, True)) is not False:
                        out[(module, node.name)] = i
                    continue
                fields = [n for n in node.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
                kw_only = any(isinstance(d, ast.Call) and any(k.arg == "kw_only" for k in d.keywords)
                              for d in node.decorator_list)
                for i, f in enumerate(fields):
                    if f.target.id == "geo" and _stretched_default(f.value):
                        out[(module, node.name)] = None if kw_only else i
    names = {m for m, _, _, _ in mods}
    for module, pkg, _, tree in mods:  # (a dataclass whose geo is its base's: Scene, Observation's)
        imports = _imports(module, pkg, tree, names)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and (module, node.name) not in out:
                bases = [(module, b.id) if (module, b.id) in out else imports.get(b.id, (None, None))[1]
                         for b in node.bases if isinstance(b, ast.Name)]
                if any(b in out for b in bases if isinstance(b, tuple)):
                    out[(module, node.name)] = None
    return out


def _calls_leaving_geo() -> set[tuple[str, str]]:
    mods = list(_modules())
    takers, names, out = _geo_takers(mods), {m for m, _, _, _ in mods}, set()
    for module, pkg, rel, tree in mods:
        imports = _imports(module, pkg, tree, names)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f, target = node.func, None
            if isinstance(f, ast.Name):
                kind, what = imports.get(f.id, (None, None))
                target = what if kind == "symbol" else (module, f.id)
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                kind, what = imports.get(f.value.id, (None, None))
                target = (what, f.attr) if kind == "module" else None
            if target not in takers or any(k.arg in ("geo", None) for k in node.keywords) \
                    or any(isinstance(a, ast.Starred) for a in node.args):
                continue
            if takers[target] is None or len(node.args) <= takers[target]:
                out.add((rel, target[1]))
    return out


def _stretched_numbers() -> set[tuple[str, str]]:
    numbers = {"stretch", "stretch_x", "units_per_px", "frame_scale", "client", "look_scale", "frame_size"}
    out = set()
    for _, _, rel, tree in _modules():
        if rel == "screen.py":  # (where they're made, for the simulator)
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
            else:
                continue
            if any(isinstance(n, ast.Attribute) and getattr(n.value, "id", "") == "STRETCHED" and n.attr in numbers
                   for n in ast.walk(node.value)):
                out |= {(rel, n.id) for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return out


def _px_unconverted() -> set[tuple[str, str]]:
    def converted(node, parents) -> bool:  # inside geo.px(...) and the like, in the same function or class
        while node in parents and not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Lambda)):
            node = parents[node]
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in ("px", "fpx", "ipx", "ifpx", "box"):
                return True
        return False
    out = set()
    for _, _, rel, tree in _modules():
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        out |= {(rel, n.id) for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id.endswith("_PX")
                and isinstance(n.ctx, ast.Load) and not converted(n, parents)}
    return out


def test_nothing_but_the_simulator_leaves_a_geometry_at_1920x1080s_by_default():
    assert _calls_leaving_geo() == set(DEFAULT_ON_PURPOSE)


def test_nothing_but_the_simulator_makes_its_own_numbers_from_the_1920x1080_window():
    assert _stretched_numbers() == set(STRETCHED_ON_PURPOSE)


def test_px_of_the_1920x1080_window_are_converted_where_theyre_used():
    assert _px_unconverted() == set(PX_ON_PURPOSE)


def test_the_checks_find_what_they_look_for(tmp_path, monkeypatch):
    # The three slips the review found, written into a module of their own: each check names its own.
    pkg = tmp_path / "gamelab"
    (pkg / "localization").mkdir(parents=True)
    (pkg / "localization" / "__init__.py").write_text("")
    (pkg / "localization" / "pose.py").write_text(
        "from ..screen import STRETCHED\n\n\nclass Localizer:\n"
        "    def __init__(self, model, origin=0, geo=STRETCHED):\n        pass\n")
    (pkg / "slips.py").write_text(
        "from .localization.pose import Localizer\nfrom .screen import STRETCHED\n\nUNITS = STRETCHED.units_per_px\n"
        "MIN_GROUND_PX = 6\n\n\ndef f(model, s):\n    return Localizer(model), s.ground >= MIN_GROUND_PX, "
        "s.geo.fpx(MIN_GROUND_PX)\n")
    monkeypatch.setitem(globals(), "PKG", pkg)
    assert _calls_leaving_geo() == {("slips.py", "Localizer")}
    assert _stretched_numbers() == {("slips.py", "UNITS")}
    assert _px_unconverted() == {("slips.py", "MIN_GROUND_PX")}


def test_the_trackers_screen_edge_and_the_players_reach_go_by_the_look():
    from gamelab.localization import mapdata
    from gamelab.localization.players import MATCH_PX, Players
    from gamelab.localization.pose import Localizer
    from gamelab.localization.tracking import EDGE_PX, View
    assert View(0, 1, 0, 1, 960, 475).edge == EDGE_PX == STRETCHED.fpx(EDGE_PX)
    assert NATIVE.fpx(EDGE_PX) == pytest.approx(34.13, abs=0.01)
    tree = {"foothold": {}, "miniMap": {"width": 1948, "height": 2152, "centerX": 479, "centerY": 1387, "mag": 4}}
    loc = Localizer(mapdata.parse(1, tree, "Test", {}))
    from gamelab.perception.minimap import Marker
    for geo, step, same in ((STRETCHED, MATCH_PX, True), (STRETCHED, MATCH_PX + 0.5, False),
                            (NATIVE, NATIVE.px(MATCH_PX) - 0.01, True), (NATIVE, NATIVE.px(MATCH_PX) + 0.5, False)):
        loc.use(geo)
        players = Players(loc)
        players.update(0.0, [Marker("stranger", 100.0, 150.0)])
        (p,) = players.update(0.0, [Marker("stranger", 100.0, 150.0)])
        (q,) = players.update(0.0, [Marker("stranger", 100.0 + step, 150.0)])  # the same player, or still the old
        assert q.id == p.id and (q.dot[0] == 100.0 + step) == same, (geo, step)  # one, missed (a new dot waits)


# --- the commands take the game window's size as it is: a minimized one once it's back ----------------------------

def _game_window(monkeypatch, game: dict):
    """window.* for a game window that's minimized while game["iconic"] (its client area then 432x54, as GetClientRect
    gave on 2026-10-24), else 1366x768."""
    from gamelab import __main__ as cli
    monkeypatch.setattr(cli.window, "find_game", lambda title: 7)
    monkeypatch.setattr(cli.window, "minimized", lambda hwnd: game["iconic"])
    monkeypatch.setattr(cli.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 432, "height": 54}
                        if game["iconic"] else {"left": 0, "top": 0, "width": 1366, "height": 768})
    return cli


def test_a_command_that_brings_the_game_forward_restores_it_before_reading_its_size(monkeypatch):
    game = {"iconic": True, "brought": 0}
    cli = _game_window(monkeypatch, game)

    def bring(title, wait=0.0):
        game["brought"] += 1
        game["iconic"] = False
        return True
    monkeypatch.setattr(cli.window, "bring_to_front", bring)
    assert cli._game_geometry("MapleStory", restore=True) == NATIVE and game["brought"] == 1
    assert cli._game_geometry("MapleStory", restore=True) == NATIVE and game["brought"] == 1  # shown: as it is
    game["iconic"] = True
    monkeypatch.setattr(cli.window, "bring_to_front", lambda title, wait=0.0: False)  # Windows wouldn't
    with pytest.raises(RuntimeError, match="minimized and didn't come back"):
        cli._game_geometry("MapleStory", restore=True)


def test_the_hunting_commands_eyes_are_made_for_the_restored_window(monkeypatch):
    from types import SimpleNamespace

    from gamelab.perception import eyes, sprites
    game = {"iconic": True}
    cli = _game_window(monkeypatch, game)
    monkeypatch.setattr(cli.window, "bring_to_front", lambda title, wait=0.0: game.update(iconic=False) or True)
    monkeypatch.setattr(cli, "_calibration", lambda geo: SimpleNamespace(template=np.zeros((15, 55), np.uint8)))
    monkeypatch.setattr(sprites, "detector_for", lambda model, geo: ("detector", geo))
    monkeypatch.setattr(eyes, "Eyes",
                        lambda title, template, detector, geo, minimap_picture=None: SimpleNamespace(detector=detector, geo=geo))
    made = cli._eyes(SimpleNamespace(id=10004021), "MapleStory")
    assert made.geo == NATIVE and made.detector == ("detector", NATIVE)


def test_watch_waits_for_a_minimized_game_and_never_restores_it(monkeypatch, capsys):
    game = {"iconic": True}
    cli = _game_window(monkeypatch, game)
    monkeypatch.setattr(cli.window, "bring_to_front", lambda *a, **k: pytest.fail("brought the game forward"))
    waits = []

    def sleep(s):
        waits.append(s)
        game["iconic"] = len(waits) < 3  # restored by whoever is at the PC after three looks
    monkeypatch.setattr(cli.time, "sleep", sleep)
    assert cli._game_geometry("MapleStory") == NATIVE and waits == [cli.MINIMIZED_POLL_S] * 3
    assert "the game is minimized: waiting" in capsys.readouterr().out


def test_the_three_standing_tests_go_by_the_looks_geometry():
    # Ground under her feet that shows she stands: 6 half-size px of the 1920x1080 window, 8.5 px at 1366x768 full
    # size. looting.standing had kept the 1920x1080 number (6 px), where navigator.standing and Scene.grounded convert.
    from gamelab.behavior import looting
    from gamelab.control import navigator
    from gamelab.localization.pose import Where
    from gamelab.localization.scene import Scene
    from types import SimpleNamespace
    p = SimpleNamespace(id=1)
    for geo, width, stands in ((STRETCHED, 6, True), (STRETCHED, 5, False), (NATIVE, 9, True), (NATIVE, 8, False),
                               (NATIVE, 6, False)):
        s = Scene(t=0.0, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0, geo=geo,
                  where=Where(0, 0, p, (0, 0)), ground=(100, 100 + width))
        assert looting.standing(s) == navigator.standing(s) == s.grounded == stands, (geo.client, width)


def test_a_recordings_analyses_go_by_its_geometry(monkeypatch):
    # combo, lock and track read recordings: at 1366x768 their frames are full size, 1.42 times the px of the
    # 1920x1080 window's half size, where the jump's lift (6 px), a jump straight up (analyze.UP_PX, 12) and the
    # claw swing's box beside her tag (track.SWING_SIDE, 70) were measured.
    from types import SimpleNamespace

    from maplehelper.engine import RIGHT

    from gamelab.analysis import analyze, track
    ys = [100] * 5 + [93] * 8 + [100] * 5  # her tag 7 px up: a jump at half size, not at full size (8.5)
    rows = [(i * 30.0, 50, y, 1.0) for i, y in enumerate(ys)]
    assert track.airborne(rows) == [(150.0, 390.0)] and track.airborne(rows, geo=NATIVE) == []
    frame = np.zeros((200, 400, 3), np.uint8)
    frame[60:98, 10:25] = 255  # bright 75-90 px left of her tag: outside the box at half size, inside at full size
    assert track.swing_side(frame, 100, 100, 40) == (0, 0) and track.swing_side(frame, 100, 100, 40, NATIVE)[0] > 0
    rows = [(i * 30.0, 50, y, 1.0) for i, y in enumerate([100] * 5 + [70] * 8 + [100] * 5)]
    xs = [0.0] * 5 + [15.0] * 13  # 15 px across over the jump
    monkeypatch.setattr(analyze, "track", lambda frames, times, template: rows)
    monkeypatch.setattr(analyze, "world_x", lambda frames, rows, geo: xs)
    monkeypatch.setattr(analyze, "swings", lambda frames, rows, width, geo: [])
    for geo, direction in ((STRETCHED, "forward"), (NATIVE, "up")):
        rec = SimpleNamespace(frames=None, times=[r[0] for r in rows], geo=geo)
        (combo,) = analyze.combos(rec, [0], 600, RIGHT, np.zeros((15, 55), np.uint8))
        assert [(j.dx, j.direction) for j in combo.jumps] == [(15.0, direction)], geo
