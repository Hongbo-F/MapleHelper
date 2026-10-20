"""A map's layout, read from the game's own data (gamedata): platforms, walls, ropes, portals, monster
spawns (and each monster's EXP and body) and the minimap's scale, in the game's map units ("world" units:
x to the right, y downward).
Also which platforms monsters can reach (`MapModel.safe`), and a picture of the map (`render`).

The screen shows the world 1080/768 = 1.40625x larger at a 1920x1080 window (the game lays out a
1366x768 view), so world units are converted with screen.SCREEN_SCALE wherever screen pixels are involved.

Game data stays on this computer (gamedata/data, git-ignored); nothing here is committed.
"""

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from gamedata.paths import COPY, OUT
from gamedata.wz import Package, jsonable


MONSTER_REACH = 55          # map units a monster's body reaches above its platform (pigs, mushrooms)
CACHE = OUT / "maps"         # parsed maps, as JSON (git-ignored, like all extracted game data)

FEET_BELOW = 8               # map units her feet may sink below a platform's surface and still stand on it
MIN_FALL = 2                 # a platform must be this far below her to be one she falls onto
SPAWN_REACH = 60             # how far above its platform a spawn point may be placed


@dataclass(frozen=True)
class Segment:
    """A straight foothold (or wall) from (x1, y1) to (x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float

    def y_at(self, x: float) -> float:
        if self.x2 == self.x1:
            return self.y1
        t = (x - self.x1) / (self.x2 - self.x1)
        return self.y1 + t * (self.y2 - self.y1)


@dataclass
class Platform:
    """A walkable stretch of ground: footholds joined end to end, left to right, with no wall between."""
    id: int
    segments: list[Segment]

    @property
    def left(self) -> float:
        return self.segments[0].x1

    @property
    def right(self) -> float:
        return self.segments[-1].x2

    @property
    def top(self) -> float:
        return min(min(s.y1, s.y2) for s in self.segments)

    def y_at(self, x: float) -> float:
        for s in self.segments:
            if s.x1 <= x <= s.x2:
                return s.y_at(x)
        return self.segments[0].y1 if x < self.left else self.segments[-1].y2

    def covers(self, x: float, margin: float = 0) -> bool:
        return self.left - margin <= x <= self.right + margin

    def clamp(self, x: float) -> float:
        """x, moved onto the platform if it's past either end."""
        return min(max(x, self.left), self.right)


def runs_under(other: Platform, p: Platform, reach: float = MONSTER_REACH) -> bool:
    """Whether platform `other` runs under `p` within `reach`, judged at the middle of where they overlap:
    monsters walking on `other` can touch her standing on `p` (their bodies stick up that far)."""
    lo, hi = max(p.left, other.left), min(p.right, other.right)  # where they overlap
    mid = (lo + hi) / 2
    return lo < hi and 0 < other.y_at(mid) - p.y_at(mid) <= reach


@dataclass(frozen=True)
class Rope:
    """A rope or ladder."""
    x: float
    top: float       # y1: the upper end (where it hangs from)
    bottom: float    # y2
    ladder: bool     # "l": a ladder, not a rope
    from_platform: bool  # "uf": she can climb onto the platform at its top


@dataclass(frozen=True)
class Portal:
    name: str
    kind: int
    x: float
    y: float
    to_map: int | None


@dataclass(frozen=True)
class MobBody:
    """A monster's body, from the boxes (lt/rb, map units from its feet) of its frames in the game data
    (Mob/<id>.img): how tall it is at least while it stands or walks (the lowest top of its stand and move
    frames), how tall while it flinches after a hit (hit1), and half its width (its widest stand or move
    frame)."""
    top: float
    flinch_top: float
    half: float


@dataclass(frozen=True)
class Spawn:
    mob_id: int
    x: float
    y: float
    left: float      # "rx0", "rx1": the x range the game data gives the spawn
    right: float
    name: str = ""


@dataclass
class MapModel:
    """One map, as the game data describes it. Platform ids are their index in `platforms`."""
    id: int
    name: str
    bounds: tuple[float, float, float, float]      # VR left, top, right, bottom
    minimap: dict                                    # width, height, centerX, centerY, mag
    platforms: list[Platform]
    walls: list[Segment]
    ropes: list[Rope]
    portals: list[Portal]
    spawns: list[Spawn] = field(default_factory=list)
    physics: dict = field(default_factory=dict)      # the game's Physics.img (kept, not read: motion.physics
                                                     # has the values it needs as constants)
    mob_exp: dict[str, int] = field(default_factory=dict)  # monster name -> EXP per kill (game data)
    mob_bodies: dict[str, MobBody] = field(default_factory=dict)  # monster name -> its body (game data)

    # --- where things are -------------------------------------------------------------------

    def platform_under(self, x: float, y: float, reach: float = 40, slack: float = 15) -> Platform | None:
        """The platform her feet are on (or just above): the closest one at or below y, within reach.
        `slack` allows for x being a little off (the minimap dot is good to ~11 map units)."""
        best = None
        for p in self.platforms:
            if not p.covers(x, slack):
                continue
            gap = p.y_at(x) - y
            if -FEET_BELOW <= gap <= reach and (best is None or gap < best[0]):
                best = (gap, p)
        return best[1] if best else None

    def platform_below(self, x: float, y: float) -> Platform | None:
        """The first platform she'd land on falling from (x, y)."""
        best = None
        for p in self.platforms:
            if p.covers(x):
                gap = p.y_at(x) - y
                if gap > MIN_FALL and (best is None or gap < best[0]):
                    best = (gap, p)
        return best[1] if best else None

    def platform(self, pid: int) -> Platform:
        return self.platforms[pid]

    # --- monsters -----------------------------------------------------------------------------

    def safe(self, pid: int, reach: float = MONSTER_REACH) -> bool:
        """No monster can touch her standing on this platform: none spawn on it (monsters never leave
        their platform), and no platform they spawn on runs underneath within `reach` (their bodies
        stick up that far: pigs on platform 1 hit her on platform 0, 42 units above them)."""
        if pid in self.spawn_platforms:
            return False
        p = self.platform(pid)
        return not any(runs_under(self.platform(q), p, reach) for q in self.spawn_platforms)

    def portal_near(self, p: Platform, x: float, dx: float = 60, dy: float = 40) -> bool:
        """Whether a portal stands on platform `p` near map x (within `dx` along it and `dy` of its
        surface): holding up there would take it."""
        return any(abs(pt.x - x) < dx and abs(pt.y - p.y_at(pt.x)) < dy for pt in self.portals)

    @cached_property
    def spawn_platforms(self) -> dict[int, int]:
        """Platform id -> number of monster spawn points on it."""
        return {pid: len(names) for pid, names in self.monsters_by_platform.items()}

    @cached_property
    def monsters_by_platform(self) -> dict[int, list[str]]:
        """Platform id -> the names of the monsters spawning on it (one per spawn point)."""
        out: dict[int, list[str]] = {}
        for s in self.spawns:
            p = self.platform_under(s.x, s.y, reach=SPAWN_REACH)
            if p:
                out.setdefault(p.id, []).append(s.name or str(s.mob_id))
        return out


# --- reading the game data ------------------------------------------------------------------------

def _footholds(tree: dict) -> list[dict]:
    """Every foothold in a map image, flattened: its layer, group and id, its ends, and its neighbours'
    ids ("0": none)."""
    out = []
    for layer, groups in (tree.get("foothold") or {}).items():
        for group, segs in groups.items():
            for sid, s in segs.items():
                out.append({"layer": layer, "group": group, "id": sid, **{k: s[k] for k in ("x1", "y1", "x2", "y2")},
                            "prev": str(s.get("prev", 0)), "next": str(s.get("next", 0))})
    return out


def _platforms(footholds: list[dict]) -> tuple[list[Platform], list[Segment]]:
    """Join walkable footholds (going right) into platforms; vertical ones are walls. Platforms are
    numbered by the height of their left end (top first), then left to right."""
    by_key = {(f["layer"], f["group"], f["id"]): f for f in footholds}
    walls = [Segment(f["x1"], f["y1"], f["x2"], f["y2"]) for f in footholds if f["x1"] == f["x2"]]
    walkable = {k for k, f in by_key.items() if f["x2"] > f["x1"]}
    seen, chains = set(), []
    # Taken left to right, a foothold always comes after the one joined before it (which ends where
    # it starts), so each chain is met at its first foothold and walked forward to its end.
    for key in sorted(walkable, key=lambda k: (by_key[k]["x1"], by_key[k]["y1"])):
        if key in seen:
            continue
        chain, cur = [], key
        while cur in walkable and cur not in seen:
            seen.add(cur)
            f = by_key[cur]
            chain.append(Segment(f["x1"], f["y1"], f["x2"], f["y2"]))
            nxt = (f["layer"], f["group"], f["next"])
            if nxt in walkable and by_key[nxt]["x1"] == f["x2"]:
                cur = nxt
            else:
                break
        chains.append(chain)
    chains.sort(key=lambda c: (c[0].y1, c[0].x1))
    return [Platform(i, c) for i, c in enumerate(chains)], walls


def _monster_lives(tree: dict) -> list[dict]:
    """A map image's monster spawn points (its 'life' entries of type 'm')."""
    return [s for s in (tree.get("life") or {}).values() if isinstance(s, dict) and s.get("type") == "m"]


def _mob_id(life: dict) -> str:
    """A spawn point's monster id, as the name and EXP tables key it (no leading zeros)."""
    return str(int(life["id"]))


def parse(map_id: int, tree: dict, name: str, physics: dict, mob_names: dict | None = None,
          mob_exp: dict | None = None, mob_bodies: dict | None = None) -> MapModel:
    """A map's model from its image in the game data (`tree`, as JSON). `mob_names`, `mob_exp` and
    `mob_bodies` (as _mob_bodies gives them) are keyed by monster id (_mob_id); the model keys EXP and
    bodies by name."""
    info = tree.get("info", {})
    platforms, walls = _platforms(_footholds(tree))
    ropes = [Rope(r["x"], r["y1"], r["y2"], bool(r.get("l")), bool(r.get("uf")))
             for r in (tree.get("ladderRope") or {}).values()]
    portals = [Portal(p.get("pn", ""), p.get("pt", 0), p["x"], p["y"],
                      None if p.get("tm") in (None, 999999999) else p["tm"])  # 999999999: leads nowhere
               for p in (tree.get("portal") or {}).values()]
    mob_names = mob_names or {}
    spawns = [Spawn(int(s["id"]), s["x"], s.get("cy", s["y"]), s.get("rx0", s["x"]), s.get("rx1", s["x"]),
                    mob_names.get(_mob_id(s), ""))
              for s in _monster_lives(tree)]
    mm = tree.get("miniMap") or {}
    return MapModel(
        id=map_id, name=name,
        bounds=(info.get("VRLeft"), info.get("VRTop"), info.get("VRRight"), info.get("VRBottom")),
        minimap={k: mm.get(k) for k in ("width", "height", "centerX", "centerY", "mag")},
        platforms=platforms, walls=walls, ropes=ropes, portals=portals, spawns=spawns, physics=physics,
        mob_exp={mob_names.get(i, i): e for i, e in (mob_exp or {}).items()},
        mob_bodies={mob_names.get(i, i): MobBody(**b) for i, b in (mob_bodies or {}).items()})


def _image(map_id: int) -> dict:
    name = f"{map_id:09d}.img"
    for part in sorted((COPY / "Map" / "Map").glob("Map*/Map*_000.wz")):
        pkg = Package(part)
        if name in pkg.images():
            return pkg.image(name)
    raise FileNotFoundError(f"map {map_id} isn't in {COPY}; run `python -m gamedata copy --install <game folder>`")


def _map_name(map_id: int) -> str:
    string = Package(COPY / "String" / "String_000.wz").image("Map.img")
    for region in string.values():
        entry = region.get(f"{map_id:09d}") if isinstance(region, dict) else None
        if isinstance(entry, dict):
            return f"{entry.get('streetName')}: {entry.get('mapName')}"
    return str(map_id)


def load(map_id: int) -> MapModel:
    """The map's model, parsed from the game data (cached as JSON next to the other extracts)."""
    cache = CACHE / f"{map_id}.json"
    raw = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
    if raw is None or "mob_bodies" not in raw:  # not cached yet, or cached before EXP or bodies were kept
        physics = jsonable(Package(COPY / "Map" / "Map_000.wz").image("Physics.img"))
        tree = jsonable(_image(map_id))
        ids = {_mob_id(s) for s in _monster_lives(tree)}
        names = _mob_names(ids)
        raw = {"id": map_id, "name": _map_name(map_id), "tree": tree, "physics": physics, "mob_names": names,
               "mob_exp": _mob_exp(ids), "mob_bodies": _mob_bodies(ids)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw), encoding="utf-8")
    return parse(raw["id"], raw["tree"], raw["name"], raw["physics"], raw["mob_names"], raw["mob_exp"],
                 raw["mob_bodies"])


def _mob_names(ids: set[str]) -> dict[str, str]:
    """Monster id -> name, from the game's strings."""
    strings = Package(COPY / "String" / "String_000.wz").image("Mob.img")
    return {i: (strings.get(i.zfill(7)) or strings.get(i) or {}).get("name", "") for i in ids}


def _mob_exp(ids: set[str]) -> dict[str, int]:
    """Monster id -> EXP per kill, from each monster's info in the game data."""
    from gamedata.canvas import _mob_pack
    pack = _mob_pack()
    return {i: int((pack.image(f"Mob/{int(i):07d}.img").get("info") or {}).get("exp") or 0) for i in ids}


BODY_ACTIONS = ("stand", "move")  # a monster's frames while it stands or walks ...
FLINCH_ACTION = "hit1"            # ... and while it flinches after a hit


def _mob_bodies(ids: set[str]) -> dict[str, dict]:
    """Monster id -> its body (MobBody's fields), from the boxes (lt/rb: from its feet, y up negative) of
    its frames in the game data. A monster whose frames have no boxes is left out (localization.monsters'
    tables stand in for it)."""
    from gamedata.canvas import _mob_pack
    from gamedata.wz import Canvas
    pack = _mob_pack()
    out = {}
    for i in ids:
        tree = pack.image(f"Mob/{int(i):07d}.img")
        link = (tree.get("info") or {}).get("link")
        if link and not tree.get("stand"):  # its frames are another monster's
            tree = pack.image(f"Mob/{int(link):07d}.img")

        def boxes(actions: tuple[str, ...]) -> list:
            return [(f.props["lt"], f.props["rb"]) for a in actions for f in (tree.get(a) or {}).values()
                    if isinstance(f, Canvas) and f.props.get("lt") is not None and f.props.get("rb") is not None]

        body, flinch = boxes(BODY_ACTIONS), boxes((FLINCH_ACTION,))
        if not body:
            continue
        top = float(min(-lt.y for lt, _ in body))
        out[i] = {"top": top, "flinch_top": float(max((-lt.y for lt, _ in flinch), default=top)),
                  "half": max(rb.x - lt.x for lt, rb in body) / 2}
    return out


# --- a picture of the map -------------------------------------------------------------------------

def render(model: MapModel, path: Path, scale: float = 0.5, marks: list[tuple[float, float, str]] = ()) -> None:
    """Draw the map's platforms (green), walls (grey), ropes (orange), portals (blue) and spawns (red),
    and `marks` (map x, y, label) in yellow."""
    from PIL import Image, ImageDraw
    left, top, right, bottom = model.bounds
    pad = 10  # px around the map
    W, H = int((right - left) * scale) + 2 * pad, int((bottom - top) * scale) + 2 * pad

    def pt(x, y):
        return (x - left) * scale + pad, (y - top) * scale + pad

    img = Image.new("RGB", (W, H), (25, 30, 45))
    d = ImageDraw.Draw(img)
    for w in model.walls:
        d.line([pt(w.x1, w.y1), pt(w.x2, w.y2)], fill=(110, 110, 110), width=2)
    for p in model.platforms:
        for s in p.segments:
            d.line([pt(s.x1, s.y1), pt(s.x2, s.y2)], fill=(90, 220, 90), width=3)
        d.text(pt(p.left, p.top - 22), str(p.id), fill=(200, 255, 200))
    for r in model.ropes:
        d.line([pt(r.x, r.top), pt(r.x, r.bottom)], fill=(240, 170, 60), width=3)
    for p in model.portals:
        x, y = pt(p.x, p.y)
        d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(80, 160, 255), width=2)
    for s in model.spawns:
        x, y = pt(s.x, s.y)
        d.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(255, 80, 80))
    for x, y, label in marks:
        px, py = pt(x, y)
        d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(255, 230, 0))
        d.text((px + 8, py - 6), label, fill=(255, 230, 0))
    img.save(path)
