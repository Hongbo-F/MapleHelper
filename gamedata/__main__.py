"""gamedata command line. Run `python -m gamedata -h`.

  python -m gamedata copy --install D:\\maplestorycw   # copy the game's data (all but Sound) into gamedata/data/copy
  python -m gamedata list                             # every package and pack in the copy
  python -m gamedata list String/String_000.wz        # the images in one
  python -m gamedata dump Packs/Skill_00000.ms Skill/400.img
  python -m gamedata export String/String_000.wz      # images as JSON, into gamedata/data/out
  python -m gamedata tables                           # items, mobs, maps, skills, physics, animations
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import tables
from .ms import MsPack
from .paths import COPY, OUT, TABLES
from .wz import Canvas, Package, jsonable


def copy_data(source: Path, dest: Path, sound: bool = False) -> tuple[int, int, int]:
    """Copies the game's Data folder, skipping files whose size and time already match (and Sound, unless
    `sound`). Returns (files copied, files already there, bytes copied)."""
    copied = kept = size = 0
    for src in sorted(source.rglob("*")):
        rel = src.relative_to(source)
        if not src.is_file() or (rel.parts[0] == "Sound" and not sound):
            continue
        dst = dest / rel
        st = src.stat()
        if dst.exists() and dst.stat().st_size == st.st_size and dst.stat().st_mtime_ns == st.st_mtime_ns:
            kept += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        size += st.st_size
    return copied, kept, size


def open_pack(rel: str):
    path = COPY / rel
    if not path.is_file():
        raise SystemExit(f"{rel} isn't in the copy ({COPY}); run `python -m gamedata copy` or check the path")
    return MsPack(path) if path.suffix.lower() == ".ms" else Package(path)


def dump(tree, depth: int, indent: int = 0, limit: int = 30) -> None:
    items = list(tree.items())
    for i, (key, value) in enumerate(items):
        if i == limit:
            print(" " * indent + f"... {len(items) - limit} more")
            break
        if isinstance(value, dict):
            print(" " * indent + f"{key}/ ({len(value)})")
            if depth > 1:
                dump(value, depth - 1, indent + 2, limit)
        elif isinstance(value, Canvas):
            print(" " * indent + f"{key} = <picture {value.width}x{value.height}>")
        else:
            print(" " * indent + f"{key} = {value!r}")


def cmd_copy(args) -> int:
    source = Path(args.install) / "appdata" / "Data"
    if not source.is_dir():
        print(f"no game data at {source}: --install should be the game's folder (the one holding appdata)")
        return 1
    copied, kept, size = copy_data(source, COPY, args.sound)
    print(f"copied {copied} files ({size / 1e6:.0f} MB), {kept} already up to date -> {COPY}")
    return 0


def cmd_list(args) -> int:
    if args.package:
        pack = open_pack(args.package)
        for image in pack.images():
            print(image)
        return 0
    paths = sorted(COPY.rglob("*.wz")) + sorted(COPY.rglob("*.ms"))
    if not paths:
        print(f"nothing in {COPY}; run `python -m gamedata copy --install <game folder>` first")
        return 1
    for path in paths:
        rel = path.relative_to(COPY).as_posix()
        try:
            count = len(open_pack(rel).images())
        except ValueError as e:
            print(f"{rel:<48} unreadable: {e}")
            continue
        if count or args.all:
            print(f"{rel:<48} {count:>5} images")
    return 0


def cmd_dump(args) -> int:
    pack = open_pack(args.package)
    for image in args.images:
        print(f"== {image}")
        tree = pack.image(image)
        if isinstance(tree, dict):
            dump(tree, args.depth)
        else:
            print(tree)
    return 0


def cmd_export(args) -> int:
    for rel in args.packages:
        pack = open_pack(rel)
        folder = OUT / Path(rel).with_suffix("")
        for image in pack.images():
            path = folder / f"{image}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(jsonable(pack.image(image)), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{rel}: {len(pack.images())} images -> {folder}")
    return 0


def cmd_tables(args) -> int:
    built = tables.build(COPY)
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in built.items():
        path = TABLES / f"{name}.json"
        path.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
        count = (len(table["Etc"]) + len(table["Use"]) + sum(map(len, table["Equip"].values()))
                 if name == "items" else len(table))
        print(f"{name:<8} {count:>5} entries -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m gamedata", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("copy", help="copy the game's data folder into gamedata/data/copy")
    c.add_argument("--install", required=True, help="the game's folder (the one holding appdata)")
    c.add_argument("--sound", action="store_true", help="copy Sound too (about 170 MB of audio)")
    c.set_defaults(run=cmd_copy)
    c = sub.add_parser("list", help="list the packages in the copy, or one package's images")
    c.add_argument("package", nargs="?", help="a package or pack, relative to the copy (String/String_000.wz)")
    c.add_argument("--all", action="store_true", help="also list packages without images")
    c.set_defaults(run=cmd_list)
    c = sub.add_parser("dump", help="print images' property trees")
    c.add_argument("package")
    c.add_argument("images", nargs="+")
    c.add_argument("--depth", type=int, default=3, help="levels to print (default 3)")
    c.set_defaults(run=cmd_dump)
    c = sub.add_parser("export", help="write every image of the given packages as JSON")
    c.add_argument("packages", nargs="+")
    c.set_defaults(run=cmd_export)
    c = sub.add_parser("tables", help="build the summary tables")
    c.set_defaults(run=cmd_tables)
    args = p.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
