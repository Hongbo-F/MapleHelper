"""Try the readers: python -m maplehelper.ocr {hud,loot} [options]. Ctrl+C stops.

  python -m maplehelper.ocr hud                       # the game's bottom bar, once a second
  python -m maplehelper.ocr loot --device cuda        # pickup messages, on the GPU
  python -m maplehelper.ocr loot --image shot.png     # a screenshot of the whole game window
  python -m maplehelper.ocr loot --image samples/loot_test.png --as-is --once
"""

import argparse
import logging
import sys
import time

import numpy as np
from PIL import Image

from .engine import DEVICES, EngineSettings, shared
from .parse import parse_hud, parse_loot
from .regions import PRESETS, content_box


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m maplehelper.ocr", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=sorted(PRESETS), help="which part of the screen to read")
    p.add_argument("--title", default="MapleStory", help="the game window's title, or part of it (default MapleStory)")
    p.add_argument("--image", help="read this screenshot of the game window instead of the screen")
    p.add_argument("--as-is", action="store_true", help="with --image: the image is already just the region")
    p.add_argument("--device", choices=DEVICES, default="cpu", help="where the OCR models run (default cpu)")
    p.add_argument("--threads", type=int, default=1, help="CPU threads for OCR, 0 = all cores (default 1)")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between reads (default 1)")
    p.add_argument("--once", action="store_true", help="read once and stop")
    p.add_argument("--rows", action="store_true", help="also print the raw OCR lines")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    preset = PRESETS[args.what]
    engine = shared(EngineSettings(device=args.device, threads=args.threads))
    if args.image:
        picture = np.asarray(Image.open(args.image).convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        if not args.as_is:
            x, y, w, h = content_box(picture)
            picture = preset.region.crop(picture[y:y + h, x:x + w])
        capture = None
    else:
        from .capture import GameCapture
        capture = GameCapture(args.title)

    print(f"Reading {preset.name} ({args.device}, {args.threads or 'all'} threads). The first read loads the models.")
    try:
        while True:
            t0 = time.perf_counter()
            crop = picture if capture is None else capture.grab(preset.region)
            t1, c1 = time.perf_counter(), time.process_time()
            if crop is None:
                print(f"{time.strftime('%H:%M:%S')}  no picture: game window {capture.problem}")
            else:
                rows = engine.read(crop, min_height=preset.min_height)
                t2, c2 = time.perf_counter(), time.process_time()
                timing = f"grab {(t1 - t0) * 1000:5.1f}  ocr {(t2 - t1) * 1000:6.1f}  cpu {(c2 - c1) * 1000:6.1f} ms"
                if rows is None:
                    print("OCR isn't available (see the warning above)")
                    return 1
                _report(preset.name, rows, timing, args.rows)
            if args.once:
                return 0
            time.sleep(max(0.0, args.interval - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        return 0
    finally:
        if capture is not None:
            capture.close()


def _report(what: str, rows, timing: str, show_rows: bool) -> None:
    stamp = time.strftime("%H:%M:%S")
    if what == "hud":
        r = parse_hud(rows)
        if r is None:
            print(f"{stamp}  {timing}  (no HP/EXP found)")
        else:
            print(f"{stamp}  {timing}  HP {r.hp}/{r.hp_max}  MP {r.mp}/{r.mp_max}  "
                  f"EXP {r.exp} ({r.exp_percent}%)  LV {r.level} [{r.level_source}]")
    else:
        lines = [(row, parse_loot(row.text)) for row in rows]
        found = [(row, line) for row, line in lines if line]
        print(f"{stamp}  {timing}  {len(found)} message(s) on screen")
        for row, line in found:
            what_ = f"{line.item} x{line.quantity}" if line.kind == "item" else f"+{line.amount}"
            print(f"    {line.kind:<5} {what_:<28} {'sure' if line.sure else 'unsure'}   \"{row.text}\"")
    if show_rows:
        for row in rows:
            print(f"    row y={row.y:6.1f} h={row.height:4.1f} score={row.score:.2f}  {row.text}")


if __name__ == "__main__":
    sys.exit(main())
