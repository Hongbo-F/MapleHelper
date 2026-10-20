"""gamelab's pieces in C (native.c), where Python itself was the cost: the monster search's step-1 bookkeeping,
which numpy did holding Python's lock (perception.mobs), and her flight, stepped 5 ms at a time by the movement
engine (motion.body.fly, as motion.physics.flight flies her).
Called through ctypes, which lets go of the lock for the call. Each does what the Python it stands in for does,
to the bit, and each caller does it in Python when the DLL isn't there (tests compare the two).

The DLL is built from native.c with MSVC 2022 (the Build Tools) the first time it's needed, into build/ next to
this file (git-ignored), named for the source's hash: an edited source gets a DLL of its own, and a DLL another
process has loaded is never written over. Without a compiler, or with USE off, `lib` is None.

    python -m gamelab.native    builds it now, and says where it is or why it couldn't
"""

import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

USE = True  # False: Python only, as before
SOURCE = Path(__file__).with_name("native.c")
BUILD = Path(__file__).with_name("build")
VCVARS = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
    "Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
# /fp:precise: IEEE arithmetic in the source's order, no fused multiply-adds (the answers must be Python's to the
# bit); /MT: the C runtime built in, so the DLL needs nothing beside it.
FLAGS = ["/nologo", "/O2", "/fp:precise", "/MT", "/LD"]

_lib: ctypes.CDLL | None = None
_tried = False
_lock = threading.Lock()  # the monster search's threads all ask at once
why = ""  # why the DLL isn't in use, if it isn't

_P, _I, _D, _F = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_float
SIGNATURES = {  # each function's (return type, argument types)
    "silhouette_peaks": (_I, [_P, _I, _I, _P, _I, _I, _D, _D, _F, _P, _P, _P, _I]),
    "fly": (_I, [_D, _D, _D, _D, _I, _D, _D, _P, _I, _P, _P, _P, _I, _I, _P, _P, _P, _P, _P, _P, _P]),
}


def dll_path() -> Path:
    """Where the DLL built from the current source is."""
    digest = hashlib.sha1(SOURCE.read_bytes() + " ".join(FLAGS).encode()).hexdigest()[:12]
    return BUILD / f"native-{digest}.dll"


def build() -> Path:
    """Build the DLL from native.c (MSVC 2022 Build Tools), unless it's built already; its path. Raises
    RuntimeError with the compiler's output if it can't."""
    out = dll_path()
    if out.exists():
        return out
    if sys.platform != "win32":
        raise RuntimeError("the native pieces are built with MSVC, on Windows")
    if not VCVARS.exists():
        raise RuntimeError(f"no MSVC 2022 Build Tools ({VCVARS} isn't there)")
    BUILD.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-", dir=BUILD) as tmp:
        dll = Path(tmp) / out.name
        command = f'call "{VCVARS}" >nul && cl {" ".join(FLAGS)} "{SOURCE}" /Fe:"{dll}" /Fo:native.obj'
        # One string, as cmd reads it (/s: the outer quotes go, the rest as written); a list would escape the
        # quotes in it the C runtime's way, which cmd doesn't read.
        done = subprocess.run(f'cmd /d /s /c "{command}"', capture_output=True, text=True, cwd=tmp,
                              timeout=300, stdin=subprocess.DEVNULL)
        if done.returncode != 0 or not dll.exists():
            raise RuntimeError(f"cl failed ({done.returncode}):\n{done.stdout}{done.stderr}")
        try:
            os.replace(dll, out)
        except OSError:  # another process built and loaded it meanwhile
            if not out.exists():
                raise
    for old in BUILD.glob("native-*.dll"):  # earlier sources' builds (about 100 KB each); a loaded one stays
        if old != out:
            try:
                old.unlink()
            except OSError:
                pass
    return out


def lib() -> ctypes.CDLL | None:
    """The DLL, built and loaded on first use; None if it can't be (`why` says why) or USE is off."""
    global _lib, _tried, why
    if not USE:
        why = "turned off (native.USE)"
        return None
    if _tried:
        return _lib
    with _lock:
        if _tried:
            return _lib
        try:
            dll = ctypes.CDLL(str(build()))
            for name, (restype, argtypes) in SIGNATURES.items():
                f = getattr(dll, name)
                f.restype, f.argtypes = restype, argtypes
            _lib, why = dll, ""
        except (OSError, RuntimeError, subprocess.SubprocessError, AttributeError) as e:
            why = str(e).splitlines()[0] if str(e) else type(e).__name__
        _tried = True
    return _lib


def peaks(res: np.ndarray, integral: np.ndarray, th: int, tw: int, cover: float,
          thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """perception.mobs._spots' step after its match: `res` (the match's float32 scores, changed in place)
    zeroed where the window's colour, from `integral` (float64, the kind's colour summed), over th * tw is
    below `cover`; then the peaks scoring at least `thr` (as float32) and no less than their 8 neighbours:
    their rows and columns (int64, as np.nonzero gives them) and scores, row by row. None without the DLL."""
    f = lib()
    if f is None:
        return None
    rh, rw = res.shape
    if not (res.flags.c_contiguous and res.dtype == np.float32 and integral.flags.c_contiguous
            and integral.dtype == np.float64 and integral.shape == (rh + th, rw + tw)):
        raise ValueError("peaks needs a contiguous float32 result and its float64 integral image")
    cap = rh * rw
    ys, xs, vs = np.empty(cap, np.int64), np.empty(cap, np.int64), np.empty(cap, np.float32)
    n = f.silhouette_peaks(res.ctypes.data, rh, rw, integral.ctypes.data, th, tw, float(th * tw), cover,
                           thr, ys.ctypes.data, xs.ctypes.data, vs.ctypes.data, cap)
    return ys[:n], xs[:n], vs[:n]


class Map:
    """A map's platforms laid out for `fly`: each one's ends, the y of its lower end and its segments (x1, y1, x2,
    y2), in the map's order. Arrays of its own, kept with it while it's in use (motion.physics keeps one per map)."""

    def __init__(self, platforms: list):
        self.n = len(platforms)
        counts = [len(p.segments) for p in platforms]
        self.arrays = [np.array([p.left for p in platforms], np.float64),
                       np.array([p.right for p in platforms], np.float64),
                       np.array([max(p.y_at(p.left), p.y_at(p.right)) for p in platforms], np.float64),
                       np.array(np.cumsum([0] + counts[:-1]), np.int32), np.array(counts, np.int32),
                       np.array([[s.x1, s.y1, s.x2, s.y2] for p in platforms for s in p.segments],
                                np.float64).reshape(-1)]
        self.pointers = [a.ctypes.data for a in self.arrays]


class Walls:
    """Walls laid out for `fly`, as body.vertical_walls gives them: their xs (sorted), and each one's top - 1 and
    bottom + 1 (motion.physics keeps one per map and foothold group)."""

    def __init__(self, walls: tuple[list[float], list[tuple[float, float]]]):
        xs, ys = walls
        self.n = len(xs)
        lo_hi = np.array(ys, np.float64).reshape(-1, 2)
        self.arrays = [np.array(xs, np.float64).reshape(-1), np.ascontiguousarray(lo_hi[:, 0]),
                       np.ascontiguousarray(lo_hi[:, 1])]
        self.pointers = [a.ctypes.data for a in self.arrays]


def shoe_of(shoe) -> ctypes.Array:
    """What `fly` takes of her shoe (motion.body.Shoe): fall, air_drag_fall, gravity, air_acc, air_speed, air_drag."""
    return (ctypes.c_double * 6)(shoe.fall, shoe.air_drag_fall, shoe.gravity, shoe.air_acc, shoe.air_speed,
                                 shoe.air_drag)


def fly(m: Map, w: Walls, x: float, y: float, vx: float, vy: float, arrow: int, dt: float, limit: float,
        shoe: ctypes.Array, skip: int) -> tuple[int, float, float, float, bool, bool] | None:
    """motion.body.fly's flight in C, as motion.physics.flight asks for it (not airborne, no map sides, no path):
    (index of the platform landed on, or -1; x; t; her sideways speed then; whether that x, and that speed, are
    the start's own, which no step changed). None without the DLL."""
    f = lib()
    if f is None:
        return None
    left, right, low, seg_at, seg_n, segs = m.pointers
    wall_xs, wall_lo, wall_hi = w.pointers
    out = (ctypes.c_double * 4)()
    q = f.fly(x, y, vx, vy, arrow, dt, limit, ctypes.addressof(shoe), skip, wall_xs, wall_lo, wall_hi, w.n, m.n,
              left, right, low, seg_at, seg_n, segs, ctypes.addressof(out))
    if q < -1:
        return None
    same = int(out[3])
    return q, out[0], out[1], out[2], bool(same & 1), bool(same & 2)


def main() -> int:
    try:
        print(f"built: {build()}")
    except RuntimeError as e:
        print(f"couldn't build gamelab's native pieces: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
