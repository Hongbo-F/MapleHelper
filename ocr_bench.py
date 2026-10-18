"""Benchmark RapidOCR on a screen region (or a saved image) and print timings.

Read-only by design: it captures the screen with mss, like any screenshot tool. It never opens
the game's process, reads its memory or sends it input.

Setup: the project's `maplehelper` env (environment.yml) has everything, including the CUDA build of
onnxruntime from conda-forge (PyPI has none for Python 3.15). To bring an existing env up to date:
  conda env update -n maplehelper -f environment.yml

Examples (python = C:\\Users\\Admin\\.conda\\envs\\maplehelper\\python.exe):
  python ocr_bench.py --list-windows
  python ocr_bench.py --window "MapleStory" --frac 0.78,0.57,1,0.86 --scale 2 --runs 20
  python ocr_bench.py --window "MapleStory" --frac 0,0.93,1,1 --device cuda --save hud.png
  python ocr_bench.py --region 0,1000,1280,60
  python ocr_bench.py --image samples/loot_test.png --device cuda --runs 20
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import platform
import statistics
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

# conda keeps the env's CUDA/cuDNN DLLs in Library\bin and puts that on PATH only when the env is
# activated. onnxruntime loads its CUDA provider through PATH, so add it here too: then running the env's
# python.exe directly (no `conda activate`) still finds the GPU.
_conda_bin = Path(sys.prefix, "Library", "bin")
if sys.platform == "win32" and _conda_bin.is_dir() and str(_conda_bin).lower() not in os.environ.get("PATH", "").lower():
    os.environ["PATH"] = f"{_conda_bin}{os.pathsep}{os.environ.get('PATH', '')}"

import cv2
import mss
import mss.exception
import numpy as np
import onnxruntime
from rapidocr import RapidOCR

Box = tuple[int, int, int, int]  # left, top, width, height in physical screen pixels


def four(kind):
    """argparse type for 'a,b,c,d'."""

    def parse(text: str):
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError("expected four comma-separated numbers")
        try:
            return tuple(kind(p) for p in parts)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e)) from None

    return parse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("what to read")
    src.add_argument("--window", metavar="TITLE", help="capture this window's client area (case-insensitive title substring)")
    src.add_argument("--region", type=four(int), metavar="X,Y,W,H", help="pixel box; relative to the window with --window, else the screen")
    src.add_argument("--frac", type=four(float), metavar="X0,Y0,X1,Y1", help="box as fractions of the window (or primary monitor), e.g. 0.78,0.57,1,0.86")
    src.add_argument("--image", type=Path, help="read a saved image instead of the screen")
    src.add_argument("--list-windows", action="store_true", help="list visible window titles and exit")

    ocr = p.add_argument_group("OCR settings")
    ocr.add_argument("--scale", type=float, default=1.0, help="resize the crop before OCR (small game text often reads better at 2)")
    ocr.add_argument("--det-side", type=int, default=960, help="detector size limit (Det.limit_side_len, default 960)")
    ocr.add_argument("--det-type", choices=("max", "min"), default="max", help="'max' caps the long side, 'min' grows the short side (Det.limit_type)")
    ocr.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = its default, all cores)")
    ocr.add_argument("--no-cls", action="store_true", help="skip the text-direction classifier (game text is never upside down)")
    ocr.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="run the models on the CPU or an NVIDIA GPU (CUDA)")
    ocr.add_argument(
        "--cudnn-search",
        choices=("HEURISTIC", "EXHAUSTIVE", "DEFAULT"),
        default="HEURISTIC",
        help="how cuDNN picks conv kernels for each new input shape; EXHAUSTIVE is RapidOCR's default but re-tunes "
        "on every new text-line width (default HEURISTIC)",
    )

    run = p.add_argument_group("benchmark")
    run.add_argument("--runs", type=int, default=10, help="timed runs (default 10)")
    run.add_argument("--warmup", type=int, default=2, help="untimed runs first: the first inference is slower (default 2)")
    run.add_argument("--interval", type=float, default=0.0, help="seconds to wait between runs, to mimic a polling loop")
    run.add_argument("--save", type=Path, help="save the last image fed to OCR, to check the crop")
    run.add_argument("--no-text", action="store_true", help="don't print what was read (timings only)")
    args = p.parse_args()
    if args.region and args.frac:
        p.error("use --region or --frac, not both")
    if args.image and (args.window or args.region or args.frac):
        p.error("--image reads a file; --window/--region/--frac pick a screen area")
    if args.runs < 1 or args.warmup < 0 or args.scale <= 0:
        p.error("--runs must be >= 1, --warmup >= 0 and --scale > 0")
    return args


# --- Windows: find the game window (by title only, no process access) -----------------------


def set_dpi_aware() -> None:
    """Window rects and mss then both use physical pixels, so crops line up on scaled displays."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def visible_windows() -> list[tuple[int, str]]:
    user32 = ctypes.windll.user32
    found: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def on_window(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(on_window, 0)
    return found


def find_window(title: str) -> tuple[int, str]:
    """First visible window whose title contains `title`, skipping this script's own console."""
    console = ctypes.windll.kernel32.GetConsoleWindow()
    needle = title.lower()
    matches = [
        (hwnd, t)
        for hwnd, t in visible_windows()
        if needle in t.lower() and hwnd != console and "ocr_bench" not in t.lower()
    ]
    if not matches:
        raise SystemExit(f"no visible window title contains {title!r} (try --list-windows)")
    if len(matches) > 1:
        others = ", ".join(repr(t) for _, t in matches[1:])
        print(f"note: several windows match; using {matches[0][1]!r} (also: {others})")
    return matches[0]


def client_box(hwnd: int) -> Box:
    """The window's client area on screen: the game's picture, without title bar or borders."""
    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    if not user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
        raise SystemExit("the window is gone")
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(origin))
    if rect.right <= 0 or rect.bottom <= 0:
        raise SystemExit("the window is minimized; restore it and run again")
    return origin.x, origin.y, rect.right, rect.bottom


def sub_box(base: Box, args: argparse.Namespace) -> Box:
    """The area to read inside `base` (the window or the primary monitor)."""
    bx, by, bw, bh = base
    if args.region:
        x, y, w, h = args.region
        return (bx + x, by + y, w, h) if args.window else (x, y, w, h)
    if args.frac:
        x0, y0, x1, y1 = args.frac
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise SystemExit("--frac needs 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
        left, top = bx + round(x0 * bw), by + round(y0 * bh)
        return left, top, max(1, round(x1 * bw) - round(x0 * bw)), max(1, round(y1 * bh) - round(y0 * bh))
    return base


# --- capture and OCR --------------------------------------------------------------------------


def grab(sct: mss.base.MSSBase, box: Box) -> np.ndarray:
    left, top, width, height = box
    shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
    return np.ascontiguousarray(np.asarray(shot)[:, :, :3])  # BGRA -> BGR, what RapidOCR expects


def make_engine(args: argparse.Namespace) -> tuple[RapidOCR, float]:
    params: dict[str, object] = {
        "Global.log_level": "warning",
        "Det.limit_side_len": args.det_side,
        "Det.limit_type": args.det_type,
    }
    if args.threads:
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = args.threads
        params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
    if args.no_cls:
        params["Global.use_cls"] = False
    if args.device == "cuda":
        params["EngineConfig.onnxruntime.use_cuda"] = True
        params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = args.cudnn_search
    t = time.perf_counter()
    engine = RapidOCR(params=params)
    return engine, (time.perf_counter() - t) * 1000


def session_providers(engine: RapidOCR) -> dict[str, str]:
    """Which execution provider each loaded model really runs on. RapidOCR loads the models on first use and
    falls back to the CPU with only a warning when CUDA won't load, so check after a run."""
    found = {}
    for name, attr in (("det", "text_det"), ("cls", "text_cls"), ("rec", "text_rec")):
        session = getattr(getattr(getattr(engine, attr, None), "session", None), "session", None)
        if session is not None:
            found[name] = session.get_providers()[0].removesuffix("ExecutionProvider")
    return found


def stage_ms(out) -> dict[str, float]:
    """RapidOCR's own per-stage times (det, cls, rec), in ms; stages that didn't run are left out."""
    stages = {}
    for name, sec in zip(("det", "cls", "rec"), getattr(out, "elapse_list", None) or ()):
        if isinstance(sec, (int, float)):
            stages[name] = sec * 1000
    return stages


@dataclass
class Run:
    capture: float
    prep: float
    ocr: float
    stages: dict[str, float]

    @property
    def total(self) -> float:
        return self.capture + self.prep + self.ocr


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown CPU"


def stats_line(name: str, values: list[float]) -> str:
    v = sorted(values)
    p95 = v[max(0, math.ceil(0.95 * len(v)) - 1)]
    return (
        f"  {name:<8} min {v[0]:7.1f}   median {statistics.median(v):7.1f}   mean {statistics.fmean(v):7.1f}"
        f"   p95 {p95:7.1f}   max {v[-1]:7.1f}  ms"
    )


def main() -> int:
    args = parse_args()
    set_dpi_aware()

    if args.list_windows:
        for hwnd, title in visible_windows():
            print(f"{hwnd:#010x}  {title}")
        return 0

    sct = None
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"can't read image {args.image}")
        where = f"image {args.image} ({image.shape[1]}x{image.shape[0]})"

        def source() -> tuple[np.ndarray, float]:
            return image, 0.0

    else:
        sct = mss.mss()
        hwnd = None
        if args.window:
            hwnd, title = find_window(args.window)
            where = f"window {title!r}"
        else:
            mon = sct.monitors[1]  # primary monitor
            where = "screen"

        def area() -> Box:
            base = client_box(hwnd) if hwnd else (mon["left"], mon["top"], mon["width"], mon["height"])
            return sub_box(base, args)

        def source() -> tuple[np.ndarray, float]:
            box = area()  # re-read each run, so a moved window is followed
            t = time.perf_counter()
            img = grab(sct, box)
            return img, (time.perf_counter() - t) * 1000

        x, y, w, h = area()
        where += f", box {w}x{h} at ({x},{y})"

    engine, load_ms = make_engine(args)
    print(f"CPU      {cpu_name()} ({os.cpu_count()} logical cores)")
    print(f"Runtime  onnxruntime {onnxruntime.__version__}, providers {', '.join(onnxruntime.get_available_providers())}")
    device = f"cuda (cuDNN search {args.cudnn_search})" if args.device == "cuda" else "cpu"
    print(
        f"OCR      device {device}, det {args.det_type} {args.det_side}, "
        f"scale x{args.scale:g}, threads {args.threads or 'default'}, cls {'off' if args.no_cls else 'on'}"
    )
    print(f"Reading  {where}")
    print(f"Engine init {load_ms:.0f} ms (the models load during the first run)\n")

    runs: list[Run] = []
    out = img = None
    try:
        for i in range(args.warmup + args.runs):
            img, cap_ms = source()
            t = time.perf_counter()
            if args.scale != 1:
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
            prep_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            out = engine(img)
            ocr_ms = (time.perf_counter() - t) * 1000
            run = Run(cap_ms, prep_ms, ocr_ms, stage_ms(out))

            label = f"warmup {i + 1}" if i < args.warmup else f"run {i - args.warmup + 1:>3}"
            parts = ", ".join(f"{k} {v:.1f}" for k, v in run.stages.items())
            print(
                f"{label:<10} capture {cap_ms:6.1f}  ocr {ocr_ms:7.1f}"
                + (f" ({parts})" if parts else "")
                + f"  total {run.total:7.1f} ms  lines {len(out.txts or ())}"
            )
            if i >= args.warmup:
                runs.append(run)
            if args.interval and i < args.warmup + args.runs - 1:
                time.sleep(args.interval)
    except mss.exception.ScreenShotError as e:
        raise SystemExit(f"screen capture failed (is the box on screen?): {e}") from None
    finally:
        if sct is not None:
            sct.close()

    providers = session_providers(engine)
    print(f"\nModels ran on: {', '.join(f'{k} {v}' for k, v in providers.items()) or 'unknown'}")
    if args.device == "cuda" and any(v != "CUDA" for v in providers.values()):
        print("  warning: CUDA was asked for but some models fell back to the CPU (see RapidOCR's log above)")

    print(f"\nSummary over {len(runs)} runs (input {img.shape[1]}x{img.shape[0]} px)")
    if not args.image:
        print(stats_line("capture", [r.capture for r in runs]))
    if args.scale != 1:
        print(stats_line("resize", [r.prep for r in runs]))
    print(stats_line("ocr", [r.ocr for r in runs]))
    for stage in ("det", "cls", "rec"):
        values = [r.stages[stage] for r in runs if stage in r.stages]
        if len(values) == len(runs):
            print(stats_line(f" {stage}", values))
    total = [r.total for r in runs]
    print(stats_line("total", total))
    print(f"  => about {1000 / statistics.median(total):.1f} reads per second at the median")

    if args.save:
        cv2.imwrite(str(args.save), img)
        print(f"\nSaved the last OCR input to {args.save}")

    if not args.no_text:
        print("\nText read on the last run (score):")
        for text, score in zip(out.txts or (), out.scores or ()):
            print(f"  {score:.2f}  {text}")
        if not out.txts:
            print("  (nothing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
