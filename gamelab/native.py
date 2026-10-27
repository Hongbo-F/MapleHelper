"""gamelab's pieces in C (native.c), where Python itself was the cost: the monster search's step-1 bookkeeping,
which numpy did holding Python's lock (perception.mobs), and her flight, stepped 5 ms at a time by the movement
engine (motion.body.fly, as motion.physics.flight flies her); and where OpenCV was: the masked colour match
(perception.match: the monsters' re-finds and step-2 checks, the drops), OpenCV's score worked out exactly, a
re-find's window's dot products all at once by number-theoretic transform (native.c's ntt_*: exact, the same scores);
and a frame's re-finds, all their masked matches in one call on threads of the C's own (`Refinds`: rf_*, its contract
in native.c), where each re-find was a Python thread's, a call a picture list. Called through ctypes, which lets go of
the lock for the call. Each but the masked match does what the Python it
stands in for does, to the bit, and each caller does it in Python (or OpenCV) when the DLL isn't there (tests compare
the two; OpenCV's masked scores were within 1e-6 of the C's in colour and 3e-5 on brightness on a crowd's frames, a
little more where a window is about flat: native.c). The masked match is the C's only where the CPU has AVX2: without
it, its plain C took 3.4-3.7 times OpenCV's time over the frame loop's masked matches (`prepare_masked`).

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

_P, _I, _D, _F, _L = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_float, ctypes.c_longlong
SIGNATURES = {  # each function's (return type, argument types)
    "silhouette_peaks": (_I, [_P, _I, _I, _P, _I, _I, _D, _D, _F, _P, _P, _P, _I]),
    "fly": (_I, [_D, _D, _D, _D, _I, _D, _D, _P, _I, _P, _P, _P, _I, _I, _P, _P, _P, _P, _P, _P, _P]),
    "mm_simd": (_I, [_I]),
    "mm_prepare_size": (_L, [_I, _I, _I]),
    "mm_prepare": (_I, [_P, _L, _P, _L, _I, _I, _I, _I, _P, _L]),
    "mm_match": (_I, [_P, _I, _I, _L, _I, _P, _P]),
    "mm_best": (_I, [_P, _I, _I, _L, _I, _P, _I, _D, _P, _P, _P]),
    "mm_ntt": (_L, [_L]),
    "mm_transformed": (_I, [_P]),
    "rf_open": (_P, [_P, _I, _I]),
    "rf_run": (_I, [_P, _P, _I, _I, _L, _I, _P, _I, _P, _D, _D, _D, _I, _P]),
    "rf_start": (_I, [_P, _P, _I, _I, _L, _I, _P, _I, _P, _D, _D, _D, _I, _P]),
    "rf_next": (_I, [_P]),
    "rf_end": (None, [_P]),
    "rf_close": (None, [_P]),
    "rf_rounded": (_D, [_F]),
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


def masked_in_c() -> bool:
    """Whether the masked match is the C's: the DLL in use, on a CPU with AVX2 (prepare_masked)."""
    f = lib()
    return f is not None and bool(f.mm_simd(-1))


def state() -> str:
    """Whether the C is in use, and the masked match with it (for the perception service's console: a DLL that
    couldn't be built leaves everything to Python and OpenCV, slower, and nothing else would show it)."""
    if lib() is None:
        return f"off ({why or 'not built'}): Python and OpenCV instead, slower"
    return "in use, " + ("the masked match on AVX2" if masked_in_c() else
                         "but not the masked match (no AVX2 on this CPU: OpenCV's, faster than the plain C)")


class Masked:
    """A template and its mask made ready for the masked match in C (native.c, mm_prepare: its sums, its mask's runs,
    its rows as int16), in bytes of its own; `shape` is the template's height, width and channels."""
    __slots__ = ("buf", "address", "shape")

    def __init__(self, buf: np.ndarray, shape: tuple[int, int, int]):
        self.buf, self.address, self.shape = buf, buf.ctypes.data, shape


def prepare_masked(template: np.ndarray, mask: np.ndarray) -> Masked | None:
    """A template (uint8, one channel or three) and its mask (uint8, nonzero where it's matched: one channel, or one per
    channel alike, as perception.match's mask3) made ready for `masked_match` and `best_matches`. None without the
    DLL, on a CPU without AVX2 (mm_simd), or for one it doesn't take (another dtype or channel count, a mask whose
    channels differ, a template row too wide for its sums): OpenCV matches those.
    Without AVX2 the C's dot products are plain C, the same scores but slower than OpenCV's: over the frame loop's
    masked matches of 60 frames of a crowd (01:01 [700:760], 1,293 of them, and 1,601 with the drops'; one thread, best
    of 3, the machine idle), 5.4 s against OpenCV's 1.6 and 5.2 against 1.4, a re-find's window 7.3 ms against 1.8
    (its step-2 checks 0.31 against 0.53, the one kind it was faster at): D:/MapleHelper-scratch/pconst/pnative-finish/
    simd_bench.py. (The tests compare the plain C with AVX2 on templates made ready while it was on.)
    A template that fits the transform's plane (128 x 128) and whose dot products stay under its prime (255 x its sum)
    is made ready with its transform too, for the windows with many places (a re-find's): 64 KB a channel, 192 KB a
    colour template, three transforms of a plane at making (about 0.1 ms; native.c, mm_prepare)."""
    f = lib()
    if not masked_in_c() or template.dtype != np.uint8 or mask.dtype != np.uint8:
        return None
    cn = 1 if template.ndim == 2 else template.shape[2] if template.ndim == 3 else 0
    if cn not in (1, 3) or mask.shape[:2] != template.shape[:2] or mask.ndim not in (2, 3):
        return None
    if mask.ndim == 3:
        if mask.shape[2] not in (1, cn) or (mask.shape[2] > 1 and not (mask == mask[..., :1]).all()):
            return None
        mask = mask[..., 0]
    t, m = np.ascontiguousarray(template), np.ascontiguousarray(mask)
    th, tw = t.shape[:2]
    size = f.mm_prepare_size(th, tw, cn)
    buf = np.empty(size, np.uint8)
    if f.mm_prepare(t.ctypes.data, t.strides[0], m.ctypes.data, m.strides[0], m.strides[1], th, tw, cn,
                    buf.ctypes.data, size) != 0:
        return None
    return Masked(buf, (th, tw, cn))


def _takes(image: np.ndarray, cn: int) -> bool:
    """Whether the C reads `image` as it is: uint8, `cn` channels, each row's pixels side by side (a view of a frame's
    box is; a flipped one isn't)."""
    if image.dtype != np.uint8 or image.strides[0] <= 0:
        return False
    if image.ndim == 2:
        return cn == 1 and image.strides[1] == 1
    return image.ndim == 3 and image.shape[2] == cn and image.strides[1] == cn and image.strides[2] == 1


def masked_match(image: np.ndarray, template: Masked) -> np.ndarray | None:
    """The masked, normalised match (OpenCV's TM_CCOEFF_NORMED with a mask, worked out exactly: native.c) of a
    prepared template at every place in `image`: float32 scores, -1 where it's undefined (a flat window). None if the
    image isn't one the C takes (`_takes`), or is smaller than the template."""
    f = lib()
    th, tw, cn = template.shape
    if f is None or not _takes(image, cn) or image.shape[0] < th or image.shape[1] < tw:
        return None
    ih, iw = image.shape[:2]
    out = np.empty((ih - th + 1, iw - tw + 1), np.float32)
    if f.mm_match(image.ctypes.data, ih, iw, image.strides[0], cn, template.address, out.ctypes.data) != 0:
        return None
    return out


def best_matches(image: np.ndarray, templates: list[Masked],
                 stop: float = float("inf")) -> list[tuple[float, int, int]] | None:
    """Each prepared template's best score over `image` and where (x, y), as cv2.minMaxLoc gives them of
    `masked_match` (the first highest, row by row), in turn: up to the first whose best is `stop` or more. All in one
    call, the image loaded once. None as for `masked_match`."""
    f = lib()
    n = len(templates)
    if f is None or not n:
        return None
    cn = templates[0].shape[2]
    if not _takes(image, cn):
        return None
    ih, iw = image.shape[:2]
    pointers = (ctypes.c_void_p * n)(*[t.address for t in templates])
    scores, xs, ys = (ctypes.c_float * n)(), (ctypes.c_int * n)(), (ctypes.c_int * n)()
    k = f.mm_best(image.ctypes.data, ih, iw, image.strides[0], cn, pointers, n, stop, scores, xs, ys)
    if k < 0:
        return None
    return [(scores[i], xs[i], ys[i]) for i in range(k)]


def takes(image: np.ndarray, cn: int) -> bool:
    """Whether the masked match in C reads `image` as it is (`_takes`)."""
    return _takes(image, cn)


class RfJob(ctypes.Structure):
    """A monster to re-find in rf_run (native.c's RfJob: its contract there)."""
    _fields_ = [("id", ctypes.c_longlong), ("key", ctypes.c_longlong), ("wnear", ctypes.c_int * 4),
                ("wfull", ctypes.c_int * 4), ("first_at", ctypes.c_int), ("first_n", ctypes.c_int),
                ("next_at", ctypes.c_int), ("next_n", ctypes.c_int)]


class RfOut(ctypes.Structure):
    """What rf_run found of one (native.c's RfOut)."""
    _fields_ = [("tpl", ctypes.c_int * 2), ("score", ctypes.c_float * 2), ("x", ctypes.c_int * 2),
                ("y", ctypes.c_int * 2), ("looked", ctypes.c_int), ("found", ctypes.c_int), ("matches", ctypes.c_int),
                ("unused", ctypes.c_int), ("work", ctypes.c_longlong), ("us", ctypes.c_double)]


class Refinds:
    """A frame's re-finds in one call in C (native.c's rf_open, rf_start / rf_next / rf_end, rf_close: the contract
    there): a detector's prepared pictures (`Masked`, kept with it), its threads (the caller's among them, a pool made
    once), the memos of the windows it looked at. One call at a time (a lock, held from `start` to `end`)."""

    def __init__(self, pictures: list[Masked], threads: int):
        self._lib = lib()
        self.pictures = list(pictures)  # (their bytes stay while the C holds their addresses)
        self.threads = threads
        self._lock = threading.Lock()
        self._held = None  # what the C reads during a call
        self.ctx = None
        if self._lib is not None and self.pictures:
            n = len(self.pictures)
            self.ctx = self._lib.rf_open((ctypes.c_void_p * n)(*[p.address for p in self.pictures]), n, threads) or None

    def start(self, frame: np.ndarray, jobs: ctypes.Array, lists: np.ndarray, thr: float, good: float,
              keep: bool = False, near_good: float | None = None) -> ctypes.Array | None:
        """rf_start over `frame` (uint8, its rows' pixels side by side: `takes`) for `jobs` (an RfJob array, each id
        once), `lists` the jobs' picture indices (int32, contiguous): an RfOut each, filled in as `next` hands each job
        over (until -1); `end` after it, always. `near_good`: the bar of a near window's answer (rf_start's `ngood`;
        None: `good`'s alone). None if the C does nothing (no `end` then)."""
        if self.ctx is None or not _takes(frame, self.pictures[0].shape[2]) or lists.dtype != np.int32 \
                or not lists.flags.c_contiguous:
            return None
        n = len(jobs)
        outs = (RfOut * n)()
        h, w = frame.shape[:2]
        self._lock.acquire()
        if self._lib.rf_start(self.ctx, frame.ctypes.data, h, w, frame.strides[0], self.pictures[0].shape[2], jobs, n,
                              lists.ctypes.data, thr, good, good if near_good is None else near_good, int(keep),
                              outs) != 0:
            self._lock.release()
            return None
        self._held = (frame, jobs, lists, outs)
        return outs

    def next(self) -> int:
        """The next job done (its index; -1: none left), Python's lock let go while it waits or does one itself."""
        return self._lib.rf_next(self.ctx)

    def end(self) -> None:
        try:
            self._lib.rf_end(self.ctx)
        finally:
            self._held = None
            self._lock.release()

    def run(self, frame: np.ndarray, jobs: ctypes.Array, lists: np.ndarray, thr: float, good: float,
            keep: bool = False, near_good: float | None = None) -> ctypes.Array | None:
        """All of `start`, `next` and `end` at once: an RfOut each, or None if the C did nothing."""
        outs = self.start(frame, jobs, lists, thr, good, keep, near_good)
        if outs is not None:
            try:
                while self.next() >= 0:
                    pass
            finally:
                self.end()
        return outs

    def close(self) -> None:
        with self._lock:
            if self.ctx is not None:
                self._lib.rf_close(self.ctx)
                self.ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # (at the interpreter's exit)
            pass


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
    print(f"gamelab's C pieces: {state()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
