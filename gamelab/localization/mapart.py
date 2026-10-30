"""A map's art drawn from the game data (gamedata): its tiles, objects and backgrounds where the game draws them, as one
picture in map units (`load`), to register the camera against what's on screen (localization.camreg).

What's drawn, in the game's order (as HeavenClient's MapBackgrounds and MapTilesObjs draw them; checked against the
frames: at the true view of 634 of Tree Dungeon's frames of 2026-10-29 the best offset was (0, 0) on every one, 96% of
the static play area within 2 levels, 99.98% of what nothing's drawn over: gamelab README, Localization): the
backgrounds behind (`back`, front 0) in their order, then layers 0 to 7, each its objects (`obj`, by z, then as listed)
and then its tiles (`tile`, by their picture's z or, where that's 0, the tile's zM, then as listed), then the
backgrounds in front (front 1). A piece's picture is drawn with its origin on its (x, y); an object or background
flipped (f 1) is mirrored about its origin. A background repeats every cx (cy) map units across (down) by its type (1
and 3-7 across, 2-3 and 5-7 down; cx or cy 0: its picture's size), its copies covering the whole picture.

Kept apart (drawn, but `static` false there, so registration leaves those pixels out):
- animated objects (more than one frame, or a frame whose alpha changes: a0 != a1): drawn with their first frame;
- objects whose every frame is invisible (a0 = a1 = 0) aren't drawn at all (footholds' carriers);
- backgrounds that don't move with the map 1:1: `back_moves` by the client's formula (shift = rx (W/2 - view) / 100
  + W/2, the same down: rx = ry = -100 moves exactly with the map, 0 stays on screen) and its own motion (types 4-7);
  and animated ones (ani 1);
- pictures gamedata can't read yet (`_rgba`): opaque, of unknown colour.
A pixel is `static` where what isn't the map's fixed art weighs under 1/255 of its colour: an opaque static piece over
anything, or static pieces down to the clear (black, as the game clears the screen).

Not drawn: what the game draws over the layers that isn't the map's (monsters, players, NPCs, portals, reactors, drops,
names, effects, the HUD): registration's truncated score takes care of it.

The picture spans every view the camera can take (`MapArt.extent`): across the VR box, or, on a map narrower than the
view (Tree Dungeon: 770 wide), the view centred on it (camera x fixed); the same down. Cached per map under
lab/mapart (git-ignored, like all extracted game data), with the code's VERSION and the copy's files' sizes and times in
its key: a changed game data copy draws it again.

    python -m gamelab.localization.mapart MAP_ID [OUT.png]    draws it (and its static mask) for a look
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gamedata.paths import COPY

from ..paths import LAB

CACHE = LAB / "mapart"
VERSION = 1          # what's drawn and how: a cache of another version is drawn again
VIEW = (1366, 768)   # the view the game lays out (screen.GAME_VIEW): one map unit to a game px
UNSURE = 1.0 / 255   # a pixel is static where what isn't the fixed art weighs less than this in its colour
# Background types (Map.img back/type): which repeat across and down, and which move by themselves.
TILE_X = {1, 3, 4, 6, 7}
TILE_Y = {2, 3, 5, 6, 7}
MOVING = {4, 5, 6, 7}


@dataclass
class Piece:
    """One picture as the map draws it. `x`, `y`: map units of its picture's top-left (flipped as drawn); `every`: a
    background's repeat across and down (0: none); `static` false with `why`: it doesn't stay put on the map."""
    kind: str              # "back", "obj" or "tile"
    name: str              # its picture's path in the game data
    layer: int             # 0-7; -1 a background behind, 8 one in front
    x: int
    y: int
    rgba: np.ndarray = field(repr=False)
    static: bool = True
    why: str = ""
    every: tuple[int, int] = (0, 0)
    opacity: float = 1.0   # a background's a / 255


@dataclass
class MapArt:
    """The map's art (`rgb`, H x W x 3 uint8, RGB) in map units from (x0, y0): pixel [r, c] is map point (x0 + c,
    y0 + r); `static` (H x W bool) where it's the map's fixed art alone (see the top); `left_out`: the pieces not
    static, by why (counts); `fixed`: the camera's view's top-left along x (y) where it can't move that way (the map
    no wider (taller) than the view), else None."""
    map_id: int
    x0: int
    y0: int
    rgb: np.ndarray = field(repr=False)
    static: np.ndarray = field(repr=False)
    fixed: tuple[int | None, int | None] = (None, None)
    views: tuple[int, int, int, int] = (0, 0, 0, 0)   # the views' top-lefts it can take: x from..to, y from..to
    left_out: dict = field(default_factory=dict)

    @property
    def extent(self) -> tuple[int, int, int, int]:
        """Map units it covers: x0, y0, x1, y1 (exclusive)."""
        h, w = self.static.shape
        return self.x0, self.y0, self.x0 + w, self.y0 + h

    def window(self, vx: int, vy: int, size: tuple[int, int] = VIEW) -> tuple[np.ndarray, np.ndarray]:
        """The art and its static mask that the view with top-left (vx, vy) shows (outside the picture: black, not
        static)."""
        w, h = size
        rgb = np.zeros((h, w, 3), np.uint8)
        ok = np.zeros((h, w), bool)
        x0, y0, x1, y1 = self.extent
        ax0, ay0, ax1, ay1 = max(vx, x0), max(vy, y0), min(vx + w, x1), min(vy + h, y1)
        if ax0 < ax1 and ay0 < ay1:
            rgb[ay0 - vy:ay1 - vy, ax0 - vx:ax1 - vx] = self.rgb[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
            ok[ay0 - vy:ay1 - vy, ax0 - vx:ax1 - vx] = self.static[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
        return rgb, ok


def views(info: dict, view: tuple[int, int] = VIEW) -> tuple[tuple[int, int], tuple[int, int]]:
    """The camera's view top-left's range across and down (from, to) by the map's VR box: between its edges, or
    centred on a map no wider (taller) than the view (localization.pose.Camera.corner's rule)."""
    out = []
    for lo, hi, size in ((info.get("VRLeft"), info.get("VRRight"), view[0]),
                         (info.get("VRTop"), info.get("VRBottom"), view[1])):
        if lo is None or hi is None:
            raise ValueError("the map has no VR box")
        if hi - lo <= size:
            v = int(round((lo + hi) / 2 - size / 2))
            out.append((v, v))
        else:
            out.append((int(lo), int(hi - size)))
    return out[0], out[1]


def back_moves(rx: float, ry: float, kind: int) -> bool:
    """Whether a background moves exactly with the map (stays put on it): by the client's formula (HeavenClient
    Background::draw: x = bg.x + rx (W/2 - view) / 100 + W/2, view the map's offset on screen), rx = ry = -100; and not
    one of the types that move by themselves (4-7)."""
    return rx == -100 and ry == -100 and kind not in MOVING


def _num(node, key, default=0):
    v = node.get(key, default) if isinstance(node, dict) else default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _frames(node) -> list:
    """An object's frames (canvases "0", "1", ...) in order."""
    from gamedata.wz import Canvas
    if not isinstance(node, dict):
        return [node] if node is not None else []
    return [v for k, v in sorted(((k, v) for k, v in node.items() if k.isdigit()), key=lambda kv: int(kv[0]))
            if isinstance(v, Canvas)]


def _origin(canvas) -> tuple[int, int]:
    o = canvas.props.get("origin")
    return (o.x, o.y) if o is not None else (0, 0)


def _rgba(canvas, tree: dict, copy: Path) -> tuple[np.ndarray, bool]:
    """A canvas's picture (RGBA), and whether it could be read: one of a format gamedata can't read yet is opaque and
    of unknown colour (False), so what it covers isn't static."""
    from gamedata.canvas import pixels, source
    picture = source(canvas, tree, copy)
    try:
        return pixels(picture), True
    except ValueError:
        rgba = np.zeros((max(1, picture.height), max(1, picture.width), 4), np.uint8)
        rgba[..., 3] = 255
        return rgba, False


def _placed(canvas, rgba: np.ndarray, x: int, y: int, flip: bool) -> tuple[int, int, np.ndarray]:
    """Where a picture's top-left goes with its origin on (x, y), and the picture as drawn (mirrored about its origin
    when flipped)."""
    ox, oy = _origin(canvas)
    h, w = rgba.shape[:2]
    if flip:
        return x - (w - ox), y - oy, rgba[:, ::-1]
    return x - ox, y - oy, rgba


def pieces(map_id: int, copy: Path = COPY) -> list[Piece]:
    """Every picture the map draws of its own, in the game's order (see the top)."""
    from gamedata.canvas import _image as image_of
    from gamedata.wz import Package

    from .mapdata import _image
    tree = _image(map_id)
    out: list[Piece] = []
    backs = _backs(tree, copy, image_of)
    out += [p for p in backs if p.layer < 0]
    obj_pkg = str(copy / "Map" / "Obj" / "Obj_000.wz")
    tile_pkg = str(copy / "Map" / "Tile" / "Tile_000.wz")
    for layer in range(8):
        node = tree.get(str(layer)) or {}
        objs = sorted(((int(k), v) for k, v in (node.get("obj") or {}).items() if k.isdigit()),
                      key=lambda kv: (_num(kv[1], "z"), kv[0]))
        for i, o in objs:
            name = f"Map/Obj/{o['oS']}.img/{o['l0']}/{o['l1']}/{o['l2']}"
            try:
                img = image_of(obj_pkg, f"{o['oS']}.img")
                at = img[str(o["l0"])][str(o["l1"])][str(o["l2"])]
            except KeyError:
                continue
            frames = _frames(at)
            if not frames:
                continue
            alphas = [(_num(f.props, "a0", 255), _num(f.props, "a1", 255)) for f in frames]
            if all(a == (0, 0) for a in alphas):
                continue  # never seen (a foothold's carrier)
            animated = len(frames) > 1 or any(a0 != a1 for a0, a1 in alphas)
            first = frames[0]
            rgba, read = _rgba(first, img, copy)
            x, y, drawn = _placed(first, rgba, _num(o, "x"), _num(o, "y"), bool(_num(o, "f")))
            if alphas[0][0] != 255 and not animated and read:
                drawn = drawn.copy()
                drawn[..., 3] = (drawn[..., 3].astype(np.uint16) * alphas[0][0] // 255).astype(np.uint8)
            why = "animated" if animated else ("unread" if not read else "")
            out.append(Piece("obj", name, layer, x, y, drawn, not why, why))
        ts = (node.get("info") or {}).get("tS")
        tiles = []
        for k, t in (node.get("tile") or {}).items():
            if not k.isdigit() or not ts:
                continue
            try:
                img = image_of(tile_pkg, f"{ts}.img")
                canvas = img[str(t["u"])][str(t["no"])]
            except KeyError:
                continue
            z = _num(canvas.props, "z")
            tiles.append(((z if z != 0 else _num(t, "zM")), int(k), t, canvas, img))
        for z, k, t, canvas, img in sorted(tiles, key=lambda e: (e[0], e[1])):
            rgba, read = _rgba(canvas, img, copy)
            x, y, drawn = _placed(canvas, rgba, _num(t, "x"), _num(t, "y"), False)
            out.append(Piece("tile", f"Map/Tile/{ts}.img/{t['u']}/{t['no']}", layer, x, y, drawn, read,
                             "" if read else "unread"))
    out += [p for p in backs if p.layer > 7]
    return out


def _backs(tree: dict, copy: Path, image_of) -> list[Piece]:
    out = []
    pkg = str(copy / "Map" / "Back" / "Back_000.wz")
    for k, b in sorted(((int(k), v) for k, v in (tree.get("back") or {}).items() if k.isdigit())):
        if not b.get("bS"):
            continue
        ani = bool(_num(b, "ani"))
        name = f"Map/Back/{b['bS']}.img/{'ani' if ani else 'back'}/{_num(b, 'no')}"
        try:
            img = image_of(pkg, f"{b['bS']}.img")
            at = img["ani" if ani else "back"][str(_num(b, "no"))]
        except KeyError:
            continue
        frames = _frames(at)
        if not frames:
            continue
        first = frames[0]
        rgba, read = _rgba(first, img, copy)
        kind = _num(b, "type")
        x, y, drawn = _placed(first, rgba, _num(b, "x"), _num(b, "y"), bool(_num(b, "f")))
        h, w = drawn.shape[:2]
        cx, cy = _num(b, "cx") or w, _num(b, "cy") or h
        every = (cx if kind in TILE_X else 0, cy if kind in TILE_Y else 0)
        moves = back_moves(_num(b, "rx"), _num(b, "ry"), kind)
        why = "" if moves and not (ani and len(frames) > 1) else ("animated" if moves else "parallax")
        why = why or ("" if read else "unread")
        out.append(Piece("back", name, 8 if _num(b, "front") else -1, x, y, drawn, not why, why, every,
                         _num(b, "a", 255) / 255))
    return out


def draw(map_id: int, copy: Path = COPY) -> MapArt:
    """The map's art drawn from the game data (no cache: `load`)."""
    from .mapdata import _image
    info = _image(map_id).get("info") or {}
    (vx0, vx1), (vy0, vy1) = views(info)
    x0, y0, x1, y1 = vx0, vy0, vx1 + VIEW[0], vy1 + VIEW[1]
    h, w = y1 - y0, x1 - x0
    rgb = np.zeros((h, w, 3), np.float32)
    unsure = np.zeros((h, w), np.float32)   # how much of each pixel's colour isn't the fixed art
    left_out: dict[str, int] = {}
    for p in pieces(map_id, copy):
        if not p.static:
            left_out[p.why] = left_out.get(p.why, 0) + 1
        for px, py in _copies(p, x0, y0, x1, y1):
            _blend(rgb, unsure, p, px - x0, py - y0)
    fixed = (vx0 if vx0 == vx1 else None, vy0 if vy0 == vy1 else None)
    return MapArt(map_id, x0, y0, np.clip(np.rint(rgb), 0, 255).astype(np.uint8), unsure < UNSURE, fixed,
                  (vx0, vx1, vy0, vy1), left_out)


def _copies(p: Piece, x0: int, y0: int, x1: int, y1: int):
    """Where each copy of a piece's picture goes (its top-left), those that reach into x0..x1, y0..y1."""
    h, w = p.rgba.shape[:2]
    ex, ey = p.every
    xs = [p.x] if not ex else range(p.x + ((x0 - w - p.x) // ex) * ex, x1, ex)
    ys = [p.y] if not ey else range(p.y + ((y0 - h - p.y) // ey) * ey, y1, ey)
    for py in ys:
        for px in xs:
            if px < x1 and py < y1 and px + w > x0 and py + h > y0:
                yield px, py


def _blend(rgb: np.ndarray, unsure: np.ndarray, p: Piece, c: int, r: int) -> None:
    """Draws a piece's picture with its top-left at column c, row r: over what's there by its alpha (and opacity)."""
    H, W = unsure.shape
    h, w = p.rgba.shape[:2]
    a0, b0, a1, b1 = max(0, r), max(0, c), min(H, r + h), min(W, c + w)
    if a0 >= a1 or b0 >= b1:
        return
    src = p.rgba[a0 - r:a1 - r, b0 - c:b1 - c]
    a = src[..., 3:4].astype(np.float32) / 255 * p.opacity
    rgb[a0:a1, b0:b1] = rgb[a0:a1, b0:b1] * (1 - a) + src[..., :3].astype(np.float32) * a
    u = unsure[a0:a1, b0:b1]
    if p.static:
        u *= 1 - a[..., 0]
    else:
        u[:] = u * (1 - a[..., 0]) + a[..., 0]


def _key(map_id: int, copy: Path) -> dict:
    """What a cache was drawn from: the code's VERSION and the game data's files' sizes and times."""
    files = [copy / "Map" / d / f"{d}_000.wz" for d in ("Obj", "Tile", "Back")]
    files += [copy / "Map" / "Obj" / "_Canvas" / "_Canvas_000.wz", copy / "Map" / "Back" / "_Canvas" / "_Canvas_000.wz",
              copy / "Map" / "Tile" / "_Canvas" / "_Canvas_000.wz"]
    files += sorted((copy / "Map" / "Map").glob("Map*/Map*_000.wz"))
    stamp = {f.relative_to(copy).as_posix(): [f.stat().st_size, int(f.stat().st_mtime)] for f in files if f.exists()}
    return {"version": VERSION, "map": map_id, "files": stamp}


_LOADED: dict[int, MapArt] = {}


def load(map_id: int, copy: Path = COPY, cache: Path | None = CACHE) -> MapArt:
    """The map's art: from the cache (lab/mapart/<map>.npz) if it was drawn from this game data by this code, else
    drawn (draw) and cached. Raises what reading the game data raises (KeyError, OSError, ValueError) if it can't."""
    if map_id in _LOADED and cache is CACHE:
        return _LOADED[map_id]
    key = _key(map_id, copy)
    path = None if cache is None else Path(cache) / f"{map_id}.npz"
    if path is not None and path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                if json.loads(str(z["key"])) == key:
                    meta = json.loads(str(z["meta"]))
                    art = MapArt(map_id, meta["x0"], meta["y0"], z["rgb"], z["static"],
                                 tuple(meta["fixed"]), tuple(meta["views"]), meta["left_out"])
                    _LOADED[map_id] = art
                    return art
        except (OSError, ValueError, KeyError):
            pass
    art = draw(map_id, copy)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"x0": art.x0, "y0": art.y0, "fixed": list(art.fixed), "views": list(art.views),
                "left_out": art.left_out}
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, rgb=art.rgb, static=art.static, key=json.dumps(key), meta=json.dumps(meta))
        tmp.replace(path)
        if cache is CACHE:
            _LOADED[map_id] = art
    return art


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[-1].strip())
        return 2
    from PIL import Image
    art = load(int(argv[0]))
    out = Path(argv[1]) if len(argv) > 1 else LAB / "shots" / f"mapart_{argv[0]}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(art.rgb).save(out)
    Image.fromarray((art.static * 255).astype(np.uint8)).save(out.with_name(out.stem + "_static.png"))
    x0, y0, x1, y1 = art.extent
    print(f"map {art.map_id}: x {x0}..{x1}, y {y0}..{y1}; views x {art.views[0]}..{art.views[1]}, "
          f"y {art.views[2]}..{art.views[3]}; static {art.static.mean():.1%}; left out {art.left_out or 'nothing'}; "
          f"{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
