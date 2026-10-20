"""Small pieces of perception in C (kernels.c), for what numpy can't do without holding Python's lock: called
through ctypes, which lets go of it for the call, so the monster search's threads run them at once.

The DLL is built from kernels.c with MSVC 2022 (the Build Tools) the first time it's needed, into build/
next to this file (git-ignored), named for the source's hash, so an edited source gets a DLL of its own and a
DLL another process has loaded is never written over. Without a compiler, or with USE off, `peaks` answers
None and the callers do the same in numpy: the answers are the same either way (tests check it), only slower.

    python -m gamelab.perception.kernels    builds it now, and says where it is or why it couldn't
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

USE = True  # False: numpy only, as before the kernels
SOURCE = Path(__file__).with_name("kernels.c")
BUILD = Path(__file__).with_name("build")
VCVARS = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
    "Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
# /fp:precise: IEEE arithmetic in the source's order, no fused multiply-adds (the answers must be numpy's to the
# bit); /MT: the C runtime built in, so the DLL needs nothing beside it.
FLAGS = ["/nologo", "/O2", "/fp:precise", "/MT", "/LD"]

_lib: ctypes.CDLL | None = None
_tried = False
_lock = threading.Lock()  # the monster search's threads all ask at once
why = ""  # why the kernels aren't in use, if they aren't


def dll_path() -> Path:
    """Where the DLL built from the current source is."""
    digest = hashlib.sha1(SOURCE.read_bytes() + " ".join(FLAGS).encode()).hexdigest()[:12]
    return BUILD / f"kernels-{digest}.dll"


def build() -> Path:
    """Build the DLL from kernels.c (MSVC 2022 Build Tools), unless it's built already; its path. Raises
    RuntimeError with the compiler's output if it can't."""
    out = dll_path()
    if out.exists():
        return out
    if sys.platform != "win32":
        raise RuntimeError("the kernels are built with MSVC, on Windows")
    if not VCVARS.exists():
        raise RuntimeError(f"no MSVC 2022 Build Tools ({VCVARS} isn't there)")
    BUILD.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kernels-", dir=BUILD) as tmp:
        dll = Path(tmp) / out.name
        command = f'call "{VCVARS}" >nul && cl {" ".join(FLAGS)} "{SOURCE}" /Fe:"{dll}" /Fo:kernels.obj'
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
    return out


def lib() -> ctypes.CDLL | None:
    """The kernels, built and loaded on first use; None if they can't be (`why` says why) or USE is off."""
    global _lib, _tried, why
    if not USE:
        why = "turned off (kernels.USE)"
        return None
    if _tried:
        return _lib
    with _lock:
        if _tried:
            return _lib
        try:
            dll = ctypes.CDLL(str(build()))
            f = dll.silhouette_peaks
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                          ctypes.c_double, ctypes.c_double, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.c_int]
            _lib, why = dll, ""
        except (OSError, RuntimeError, subprocess.SubprocessError, AttributeError) as e:
            why = str(e).splitlines()[0] if str(e) else type(e).__name__
        _tried = True
    return _lib


def peaks(res: np.ndarray, integral: np.ndarray, th: int, tw: int, cover: float,
          thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """mobs._spots' step after its match, in C: `res` (the match's float32 scores, changed in place) zeroed
    where the window's colour, from `integral` (float64, the kind's colour summed), over th * tw is below
    `cover`; then the peaks scoring at least `thr` (as float32) and no less than their 8 neighbours: their
    rows and columns (int64, as np.nonzero gives them) and scores, row by row. None if the kernels aren't
    there."""
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


def main() -> int:
    try:
        print(f"built: {build()}")
    except RuntimeError as e:
        print(f"couldn't build the kernels: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
