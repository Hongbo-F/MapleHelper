"""Try the readers: python -m maplehelper.ocr {hud,loot} [options]. Ctrl+C stops.

  python -m maplehelper.ocr hud                       # the game's bottom bar, once a second
  python -m maplehelper.ocr loot --device cuda        # pickup messages, on the GPU
  python -m maplehelper.ocr loot --image shot.png     # a screenshot of the whole game window
  python -m maplehelper.ocr loot --image samples/loot_test.png --as-is --once
"""

import argparse
import logging
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .engine import DEVICES, EngineSettings, Row, shared
from .loot import LootRead, LootReader, same_line, similar_text
from .parse import parse_hud
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
    p.add_argument("--save", metavar="DIR", help="save each crop whose text changed (PNG + its OCR lines) into DIR")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    preset = PRESETS[args.what]
    engine = shared(EngineSettings(device=args.device, threads=args.threads))
    # Loaded before the loop: onnxruntime prints its load messages straight to the console, which would
    # otherwise land in the middle of a status line being rewritten.
    print(f"Loading the OCR models ({args.device}, {args.threads or 'all'} threads)...", flush=True)
    if not engine.load():
        print("OCR isn't available (see the warning above)")
        return 1
    if args.image:
        picture = np.asarray(Image.open(args.image).convert("RGB"))[:, :, ::-1]  # RGB -> BGR
        if not args.as_is:
            x, y, w, h = content_box(picture)
            picture = preset.region.crop(picture[y:y + h, x:x + w])
        capture = None
    else:
        from .capture import GameCapture
        capture = GameCapture(args.title)

    print(f"Reading {preset.name}. Ctrl+C stops.", flush=True)
    console = Console()
    if preset.name == "hud":
        report = HudReport(console, args.rows)
        read = lambda crop: engine.read(crop, min_height=preset.min_height)  # noqa: E731
    else:
        report = LootReport(console, args.rows)
        read = LootReader(engine, preset).read
    saver = Saver(args.save, preset.name) if args.save else None
    problem_since: dict[str, str] = {}
    try:
        while True:
            t0 = time.perf_counter()
            crop = picture if capture is None else capture.grab(preset.region)
            t1, c1 = time.perf_counter(), time.process_time()
            if crop is None:
                since = problem_since.setdefault(capture.problem, time.strftime("%H:%M:%S"))
                console.update(("problem", capture.problem),
                               f"{time.strftime('%H:%M:%S')}  no picture: game window {capture.problem} (since {since})")
            else:
                problem_since.clear()
                result = read(crop)
                t2, c2 = time.perf_counter(), time.process_time()
                if result is None:
                    console.line("OCR isn't available (see the warning above)")
                    return 1
                rows = result if preset.name == "hud" else result.rows
                if saver and (saved := saver(crop, rows)):
                    console.line(f"saved {saved}")
                report(result, f"grab {(t1 - t0) * 1000:.0f} / ocr {(t2 - t1) * 1000:.0f} / cpu {(c2 - c1) * 1000:.0f} ms")
            if args.once:
                return 0
            time.sleep(max(0.0, args.interval - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        return 0
    finally:
        console.end()
        if capture is not None:
            capture.close()


class Console:
    """A log that only scrolls when something changes: news that repeats what the last line said (still no game
    window, the same HP as before) rewrites that line in place. When output goes to a file, repeats are skipped."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty()
        self.key = None   # what the last line reports; a repeat of it rewrites the line
        self._open = 0    # length of the line being rewritten (tty only), 0 when none is open

    def update(self, key, text: str) -> None:
        """Show `text`. If it reports the same `key` as the last line, rewrite that line instead of adding one."""
        if self.tty:
            text = text[:shutil.get_terminal_size().columns - 1]  # a wrapped line can't be rewritten
        if key == self.key:
            if self.tty:
                self.stream.write("\r" + text.ljust(self._open))
                self._open = max(len(text), 1)
                self.stream.flush()
            return
        self.end()
        self.stream.write(text if self.tty else text + "\n")
        self.stream.flush()
        self.key, self._open = key, max(len(text), 1) if self.tty else 0

    def line(self, text: str) -> None:
        """A line that stays, whatever comes next."""
        self.end()
        self.stream.write(text + "\n")
        self.stream.flush()
        self.key = None

    def end(self) -> None:
        if self._open:
            self.stream.write("\n")
            self.stream.flush()
        self._open = 0


class HudReport:
    """A new line when HP, MP, EXP or level change; otherwise the last line just refreshes its time and timings."""

    def __init__(self, console: Console, show_rows: bool):
        self.console, self.show_rows = console, show_rows

    def __call__(self, rows, timing: str) -> None:
        r = parse_hud(rows)
        key = ("hud", None) if r is None else ("hud", r.hp, r.hp_max, r.mp, r.mp_max, r.exp, r.exp_percent, r.level)
        if self.show_rows and key != self.console.key:
            for row in rows:
                self.console.line(f"    row y={row.y:6.1f} h={row.height:4.1f} score={row.score:.2f}  {row.text}")
        stamp = time.strftime("%H:%M:%S")
        if r is None:
            self.console.update(key, f"{stamp}  no HP/EXP found   ({timing})")
        else:
            level = f"{r.level} ({r.level_source})" if r.level else "?"
            self.console.update(key, f"{stamp}  HP {r.hp}/{r.hp_max}  MP {r.mp}/{r.mp_max}  "
                                     f"EXP {r.exp} ({r.exp_percent}%)  LV {level}   ({timing})")


class RowMemory:
    """Remembers the lines seen recently, so a line that stays on screen counts as new only once although OCR
    reads it a little differently each time ("PlayerOne", "PlayerOn", "* PlayerOne"): same height on screen and
    similar text is the same line. A line not seen for FORGET_S is forgotten."""

    FORGET_S = 10.0

    def __init__(self):
        self._lines: list[tuple[Row, float]] = []  # (latest read of the line, when)

    def is_new(self, row: Row, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        self._lines = [(r, t) for r, t in self._lines if now - t <= self.FORGET_S]
        for i, (known, _) in enumerate(self._lines):
            if abs(row.y - known.y) <= 0.5 * max(row.height, known.height) and similar_text(row.text, known.text):
                self._lines[i] = (row if len(row.text) > len(known.text) else known, now)
                return False
        self._lines.append((row, now))
        return True


class LootReport:
    """Each message once when it shows up (a message stays on screen for several reads), plus a status line.

    "Shows up" compares with the previous read only, so a line that fades in and out of legibility can be listed
    twice; counting each pickup exactly once needs a tracker that follows the message stack between reads."""

    def __init__(self, console: Console, show_rows: bool):
        self.console, self.show_rows = console, show_rows
        self._seen: Counter = Counter()
        self._rows = RowMemory()
        self._interface: list[Row] = []

    def __call__(self, read: LootRead, timing: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        if self.show_rows:
            # Each line OCR sees in the box, once. Lines that aren't pickup messages (party member names, chat,
            # icons read as characters) are only shown here, never counted.
            for row in read.rows:
                if self._rows.is_new(row):
                    tag = "ignored" if row in read.ignored else "pickup "
                    self.console.line(f"    {tag} y={row.y:6.1f} h={row.height:4.1f} score={row.score:.2f}  {row.text}")
        for row in read.interface:
            if not any(same_line(row, known) for known in self._interface):
                self.console.line(f"{stamp}  skipping \"{row.text}\" from now on: it stays put, so it's interface")
        self._interface = read.interface
        found = read.messages
        now = Counter(_loot_key(line) for _, line in found)
        new = now - self._seen
        stamp = time.strftime("%H:%M:%S")
        for row, line in found:
            if new[_loot_key(line)] > 0:
                new[_loot_key(line)] -= 1
                what = f"{line.item} x{line.quantity}" if line.kind == "item" else f"+{line.amount}"
                self.console.line(f"{stamp}  {line.kind:<5} {what:<28} {'sure' if line.sure else 'unsure':<6}  \"{row.text}\"")
        self._seen = now
        skipped = f", {len(read.interface)} interface line(s) skipped" if read.interface else ""
        self.console.update(("loot",), f"{stamp}  {len(found)} message(s) on screen{skipped}   ({timing})")


class Saver:
    """Saves a crop and its OCR lines whenever a line shows up that wasn't there before (at most MAX_FILES).
    OCR reading the same lines a little differently doesn't count as new (see RowMemory)."""

    MAX_FILES = 200

    def __init__(self, folder: str, name: str):
        self.folder, self.name = Path(folder), name
        self.folder.mkdir(parents=True, exist_ok=True)
        self._rows = RowMemory()
        self._count = 0

    def __call__(self, crop: np.ndarray, rows) -> Path | None:
        new = [self._rows.is_new(r) for r in rows]  # every row, so each one is remembered
        if not any(new) or self._count >= self.MAX_FILES:
            return None
        self._count += 1
        stem = self.folder / f"{self.name}_{time.strftime('%Y%m%d-%H%M%S')}_{self._count:03d}"
        Image.fromarray(np.ascontiguousarray(crop[:, :, ::-1])).save(stem.with_suffix(".png"))  # BGR -> RGB
        stem.with_suffix(".txt").write_text(
            "".join(f"y={r.y:.1f} h={r.height:.1f} x={r.left:.0f}-{r.right:.0f} score={r.score:.2f}  {r.text}\n"
                    for r in rows), encoding="utf-8")
        return stem.with_suffix(".png")


def _loot_key(line) -> tuple:
    return line.kind, line.amount, line.item, line.quantity


if __name__ == "__main__":
    sys.exit(main())
