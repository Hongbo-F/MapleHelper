"""Tables pulled out of the game data: item names, monsters, maps, skills, movement physics and body animations.

Each table is built by a function of parsed trees (testable without the game's files); `build` loads them from
a copy of the game's Data folder.
"""

import re
from pathlib import Path

from .ms import MsPack
from .wz import Package, Vector, jsonable

# Equip categories in String/Eqp.img that are looks, not items.
NOT_ITEMS = {"Face", "Hair", "Skin"}


def _named(tree: dict) -> dict[int, str]:
    return {int(k): v["name"] for k, v in tree.items() if k.isdigit() and isinstance(v, dict) and "name" in v}


def items(etc: dict, consume: dict, eqp: dict) -> dict:
    """Item names by ID from String/Etc.img, Consume.img and Eqp.img: {"Etc": {id: name}, "Use": {...},
    "Equip": {category: {id: name}}}. Equip leaves out faces, hairs and skins."""
    equip = {}
    for group in eqp.values():  # one group, named after the world ("ClassicWorld")
        for category, entries in group.items():
            if category not in NOT_ITEMS and isinstance(entries, dict):
                equip[category] = _named(entries)
    return {"Etc": _named(etc.get("Etc", etc)), "Use": _named(consume), "Equip": equip}


def mob_names(string_mob: dict) -> dict[int, str]:
    return _named(string_mob)


MOB_FIELDS = ("level", "maxHP", "maxMP", "exp", "speed", "PADamage", "PDDamage", "MADamage", "MDDamage", "acc",
              "eva", "boss", "undead", "link")


def mobs(trees: dict[str, dict], names: dict[int, str]) -> list[dict]:
    """One row per monster image ("Mob/0000002.img"), with its name and stats, by level. `exp` is per kill."""
    rows = []
    for path, tree in trees.items():
        stem = Path(path).stem
        if not stem.isdigit():
            continue
        info = tree.get("info", {})
        row = {"id": int(stem), "name": names.get(int(stem))}
        row.update({k: info[k] for k in MOB_FIELDS if k in info})
        rows.append(row)
    return sorted(rows, key=lambda r: (r.get("level") or 0, r["id"]))


def maps(trees: dict[str, dict], string_map: dict, mob_name: dict[int, str]) -> list[dict]:
    """One row per map image ("Map0/000000040.img"): its names, whether it's a town, where you return to, its
    bounds, the monsters it spawns (name -> how many spawn points), its NPC count and the maps its portals lead to."""
    names = {}
    for region in string_map.values():
        for map_id, entry in region.items():
            if map_id.isdigit() and isinstance(entry, dict):
                names[int(map_id)] = (entry.get("streetName"), entry.get("mapName"))
    rows = []
    for path, tree in trees.items():
        stem = Path(path).stem
        if not stem.isdigit():
            continue
        map_id = int(stem)
        info = tree.get("info", {})
        life = [v for v in (tree.get("life") or {}).values() if isinstance(v, dict)]
        spawns: dict[str, int] = {}
        for spot in life:
            if spot.get("type") == "m":
                mob = int(spot["id"])
                key = mob_name.get(mob) or str(mob)
                spawns[key] = spawns.get(key, 0) + 1
        portals = [v for v in (tree.get("portal") or {}).values() if isinstance(v, dict)]
        street, name = names.get(map_id, (None, None))
        rows.append({
            "id": map_id, "street": street, "name": name, "town": bool(info.get("town")),
            "returnMap": info.get("returnMap"),
            "bounds": [info.get(k) for k in ("VRLeft", "VRTop", "VRRight", "VRBottom")],
            "mobs": spawns, "npcs": sum(1 for s in life if s.get("type") == "n"),
            "exits": sorted({p["tm"] for p in portals if p.get("tm") not in (None, 999999999, map_id)}),
        })
    return sorted(rows, key=lambda r: r["id"])


def skills(job_trees: dict[str, dict], string_skill: dict) -> dict[str, dict]:
    """Skills by ID from the job images ("Skill/400.img"): name, animation(s), required weapon type and each
    level's numbers (MP cost, damage, duration...)."""
    out = {}
    for tree in job_trees.values():
        for skill_id, skill in (tree.get("skill") or {}).items():
            levels = {}
            for level, stats in (skill.get("level") or {}).items():
                levels[int(level)] = {k: jsonable(v) for k, v in stats.items()
                                      if k != "hs" and isinstance(v, (int, float, str, Vector))}
            action = skill.get("action")
            out[skill_id] = {
                "name": (string_skill.get(skill_id) or {}).get("name"),
                "action": list(action.values()) if isinstance(action, dict) else action,
                "weapon": skill.get("weapon"),
                "levels": dict(sorted(levels.items())),
            }
    return dict(sorted(out.items()))


def actions(body: dict) -> dict[str, list[int]]:
    """Each body animation's frame delays in ms (from Character/00002000.img), as stored. Some skill animations
    have negative delays; what the sign means isn't known here."""
    out = {}
    for name, frames in body.items():
        if name == "info" or not isinstance(frames, dict):
            continue
        out[name] = [f["delay"] for k, f in sorted(frames.items(), key=lambda kv: int(kv[0]))
                     if k.isdigit() and isinstance(f, dict) and "delay" in f]
    return out


def _job(path: str) -> bool:
    # Job images are named by job number ("Skill/000.img", "Skill/410.img"); the rest is shared data.
    return re.fullmatch(r"Skill/\d{3,4}\.img", path) is not None


def build(data: Path) -> dict[str, object]:
    """Every table from a copy of the game's Data folder."""
    data = Path(data)
    string = Package(data / "String" / "String_000.wz")
    s = {name: string.image(name) for name in ("Etc.img", "Consume.img", "Eqp.img", "Mob.img", "Map.img", "Skill.img")}
    mob_pack = MsPack(data / "Packs" / "Mob_00000.ms")
    names = mob_names(s["Mob.img"])
    map_trees = {}
    for part in sorted((data / "Map" / "Map").glob("Map*/Map*_000.wz")):
        pkg = Package(part)
        map_trees.update({f"{part.parent.name}/{img}": pkg.image(img) for img in pkg.images()})
    skill_pack = MsPack(data / "Packs" / "Skill_00000.ms")
    return {
        "items": items(s["Etc.img"], s["Consume.img"], s["Eqp.img"]),
        "mobs": mobs({p: mob_pack.image(p) for p in mob_pack.images() if p.endswith(".img")}, names),
        "maps": maps(map_trees, s["Map.img"], names),
        "skills": skills({p: skill_pack.image(p) for p in skill_pack.images() if _job(p)}, s["Skill.img"]),
        "physics": jsonable(Package(data / "Map" / "Map_000.wz").image("Physics.img")),
        "actions": actions(Package(data / "Character" / "Character_000.wz").image("00002000.img")),
    }
