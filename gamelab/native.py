"""gamelab's pieces in C (native.c), where Python itself was the cost: the monster search's step-1 bookkeeping,
which numpy did holding Python's lock (perception.mobs), and her flight, stepped 5 ms at a time by the movement
engine (motion.body.fly, as motion.physics.flight flies her); and where OpenCV was: the masked colour match
(perception.match: the monsters' re-finds and step-2 checks, the drops), OpenCV's score worked out exactly, a
re-find's window's dot products all at once by number-theoretic transform (native.c's ntt_*: exact, the same scores);
and a frame's re-finds, all their masked matches in one call on threads of the C's own (`Refinds`: rf_*, its contract
in native.c), where each re-find was a Python thread's, a call a picture list; or on the GPU (gpumatch.c and
gpumatch.hlsl, Direct3D 11, in the same DLL: `GPU`, off by default), the same answers. Called through ctypes, which lets
go of the lock for the call. Each but the masked match does what the Python it
stands in for does, to the bit, and each caller does it in Python (or OpenCV) when the DLL isn't there (tests compare
the two; OpenCV's masked scores were within 1e-6 of the C's in colour and 3e-5 on brightness on a crowd's frames, a
little more where a window is about flat: native.c). The masked match is the C's only where the CPU has AVX2: without
it, its plain C took 3.4-3.7 times OpenCV's time over the frame loop's masked matches (`prepare_masked`).

The monster search's step 1 is a DLL of its own (search.c, `search_lib`: the band's shrink, its colour tests, the
silhouettes' scores where the cover test keeps a window, the peaks and the places, in a few calls a band), built the
same way, loaded only where the CPU has AVX2 (its intrinsics); without it the search's step 1 is numpy's and OpenCV's
as before (perception.mobs).

The DLLs are built from their sources (native.c's with gpumatch.c and gpumatch.hlsl in it) with MSVC 2022 (the Build
Tools) the first time they're needed, into build/ next to this file (git-ignored), each named for its sources' hash: an
edited source gets a DLL of its own, and a DLL another process has loaded is never written over. Without a compiler, or
with USE off, `lib` and `search_lib` are None.

    python -m gamelab.native    builds them now, and says where they are or why they couldn't be
"""

import ctypes
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import weakref
from pathlib import Path

import numpy as np

USE = True  # False: Python only, as before


def gpu_switch(environ=os.environ, path: Path | None = None) -> bool:
    """Whether the GPU backend is asked for: GAMELAB_GPU's 1 or 0, else whether `path` (paths.GPU, lab/GPU) is there.
    The file is the switch a perceive started through WMI sees (babysit3's, service.ensure's: a WMI process gets none
    of its starter's environment, the correctness review's wmi_env.txt)."""
    v = environ.get("GAMELAB_GPU", "")
    if v in ("0", "1"):
        return v == "1"
    if path is None:
        from .paths import GPU as path
    return path.exists()


# The re-finds' matches on the GPU (gpumatch.c, gpumatch.hlsl: Direct3D 11, behind rf_*; the answers the CPU's to the
# bit), for the contexts `Refinds` opens while it's on: off unless switched on (`gpu_switch`), as on this machine it
# doesn't pay (README, The re-finds' matches on the GPU).
GPU = gpu_switch()
SOURCE = Path(__file__).with_name("native.c")
SEARCH_SOURCE = Path(__file__).with_name("search.c")  # the monster search's step 1 (`search_lib`)
GPU_SOURCES = (Path(__file__).with_name("gpumatch.c"), Path(__file__).with_name("gpumatch.hlsl"))  # (in the DLL)
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
_search: ctypes.CDLL | None = None
_search_tried = False
search_why = ""  # why the search's DLL isn't in use, if it isn't
_open: "weakref.WeakSet[Refinds]" = weakref.WeakSet()  # the Refinds with a context (gpu_line)
_open_lock = threading.Lock()

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
    "gm_on": (_I, [_I]),
    "rf_gpu": (_I, [_P, _P, _I]),
    "rf_gpu_fault": (None, [_P, _I]),
    "gm_bests": (_I, [_P, _I, _P, _P, _P, _P, _I, _I, _P, _P, _P, _P, _P, _P]),
    "gm_approx": (_I, [_P, _P, _I, _I, _L, _I, _I, _P]),
    "gm_waits": (None, [_L, _L, _P]),
}


def dll_path(source: Path | None = None) -> Path:
    """Where the DLL built from the current source (native.c, with GPU_SOURCES in it; or `source`) is."""
    source = source or SOURCE
    extra = b"".join(p.read_bytes() for p in GPU_SOURCES) if source == SOURCE else b""
    digest = hashlib.sha1(source.read_bytes() + extra + " ".join(FLAGS).encode()).hexdigest()[:12]
    return BUILD / f"{source.stem}-{digest}.dll"


def build(source: Path | None = None) -> Path:
    """Build the DLL from native.c (or `source`; MSVC 2022 Build Tools), unless it's built already; its path. Raises
    RuntimeError with the compiler's output if it can't."""
    source = source or SOURCE
    out = dll_path(source)
    if out.exists():
        return out
    if sys.platform != "win32":
        raise RuntimeError("the native pieces are built with MSVC, on Windows")
    if not VCVARS.exists():
        raise RuntimeError(f"no MSVC 2022 Build Tools ({VCVARS} isn't there)")
    BUILD.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-", dir=BUILD) as tmp:
        dll = Path(tmp) / out.name
        # gpumatch.hlsl as a C string (gpumatch.c's GM_HLSL), compiled at run time by d3dcompiler_47; native.c's only.
        (Path(tmp) / "gpumatch_hlsl.h").write_text(_c_string("GM_HLSL", GPU_SOURCES[1].read_text(encoding="utf-8")),
                                                   encoding="utf-8")
        command = (f'call "{VCVARS}" >nul && cl {" ".join(FLAGS)} /I"{tmp}" "{source}" /Fe:"{dll}" '
                   f'/Fo:{source.stem}.obj')
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
    for old in BUILD.glob(f"{source.stem}-*.dll"):  # earlier sources' builds (about 100 KB each); a loaded one stays
        if old != out:
            try:
                old.unlink()
            except OSError:
                pass
    return out


def _c_string(name: str, text: str) -> str:
    """`text` as a C header defining `name`, a string literal (a line a piece)."""
    lines = [line.replace("\\", "\\\\").replace('"', '\\"') for line in text.splitlines()]
    return f"#define {name} \\\n" + "".join(f'    "{line}\\n" \\\n' for line in lines) + '    ""\n'


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
            dll.gm_on(1 if GPU else 0)
            _lib, why = dll, ""
        except (OSError, RuntimeError, subprocess.SubprocessError, AttributeError) as e:
            why = str(e).splitlines()[0] if str(e) else type(e).__name__
        _tried = True
    return _lib


SEARCH_SIGNATURES = {  # search.c's functions: (return type, argument types)
    "sr_ready": (_I, []),
    "sr_shrink": (_I, [_P, _I, _I, _L, _P, _I, _I, _L]),
    "sr_colours": (_I, [_P, _I, _I, _L, _P, _P, _I, _P, _P]),
    "sr_step1": (_I, [_P, _I, _I, _L, _P, _P, _I, _P, _I, _P, _F, _D, _D, _P, _P, _P, _P, _P, _P, _I, _P, _P]),
    "sr_places": (_I, [_P, _P, _P, _P, _P, _P, _P, _P, _I, _L, _I, _I, _P, _I]),
}


def search_lib() -> ctypes.CDLL | None:
    """The monster search's DLL (search.c), built and loaded on first use; None if it can't be (`search_why` says why),
    on a CPU without AVX2 (sr_ready: asked before anything else of it runs), or with USE off."""
    global _search, _search_tried, search_why
    if not USE:
        search_why = "turned off (native.USE)"
        return None
    if _search_tried:
        return _search
    with _lock:
        if _search_tried:
            return _search
        try:
            dll = ctypes.CDLL(str(build(SEARCH_SOURCE)))
            for name, (restype, argtypes) in SEARCH_SIGNATURES.items():
                f = getattr(dll, name)
                f.restype, f.argtypes = restype, argtypes
            if dll.sr_ready():
                _search, search_why = dll, ""
            else:
                search_why = "no AVX2 on this CPU"
        except (OSError, RuntimeError, subprocess.SubprocessError, AttributeError) as e:
            search_why = str(e).splitlines()[0] if str(e) else type(e).__name__
        _search_tried = True
    return _search


class SrUnit(ctypes.Structure):
    """A unit of the search (mobs.MobDetector.units, as a band's search filters them) for sr_step1 (search.c's)."""
    _fields_ = [("test", ctypes.c_int), ("bits", ctypes.c_uint), ("least", ctypes.c_double), ("first", ctypes.c_int),
                ("n", ctypes.c_int), ("gfirst", ctypes.c_int), ("gn", ctypes.c_int)]


class SrShape(ctypes.Structure):
    """A shape's silhouette made ready for sr_step1 (search.c's SrShape; its arrays kept by the caller)."""
    _fields_ = [("th", ctypes.c_int), ("tw", ctypes.c_int), ("tw4", ctypes.c_int), ("kmin", ctypes.c_int),
                ("st", ctypes.c_longlong), ("A", ctypes.c_double), ("scale", ctypes.c_double),
                ("rows", ctypes.c_void_p), ("ub", ctypes.c_void_p), ("order", ctypes.c_void_p),
                ("check", ctypes.c_int), ("unused", ctypes.c_int), ("rtt", ctypes.c_double)]


SR_GROUP = 16  # a group's shapes at most (search.c's GL)


class SrGroup(ctypes.Structure):
    """A unit's shapes of one size, their bounds' tables side by side (search.c's SrGroup)."""
    _fields_ = [("ng", ctypes.c_int), ("idx", ctypes.c_int * SR_GROUP), ("ug", ctypes.c_void_p),
                ("vg", ctypes.c_void_p), ("su", ctypes.c_double * SR_GROUP), ("sv", ctypes.c_double * SR_GROUP)]


def shrink_area(frame: np.ndarray, dw: int, dh: int) -> np.ndarray | None:
    """cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA) of a uint8 RGB picture whose pixels lie side by side
    (a band of a look frame), in C (search.c's sr_shrink: the same bytes); None where the C declines (not built, another
    picture, a scale it doesn't take: 1.75 or more, or none, either way)."""
    f = search_lib()
    if f is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or frame.strides[1] != 3 \
            or frame.strides[2] != 1 or frame.strides[0] <= 0:
        return None
    h, w = frame.shape[:2]
    out = np.empty((dh, dw, 3), np.uint8)
    if f.sr_shrink(frame.ctypes.data, h, w, frame.strides[0], out.ctypes.data, dh, dw, out.strides[0]) != 0:
        return None
    return out


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
    on = ("; the re-finds' matches switched to the GPU (its device made at a detector's first re-finds: whether it's"
          " on, the status line's \"the re-finds' GPU\")") if gpu() else ""
    return "in use, " + ("the masked match on AVX2" if masked_in_c() else
                         "but not the masked match (no AVX2 on this CPU: OpenCV's, faster than the plain C)") + on


def gpu(on: bool | None = None) -> bool:
    """Whether the `Refinds` made from now on have the GPU backend (native.c's gm_on; GPU at the start); `on` sets
    it. False without the DLL."""
    f = lib()
    if f is None:
        return False
    return bool(f.gm_on(-1 if on is None else int(bool(on))))


GPU_STATS = ("calls", "answered", "late", "busy", "full", "pairs", "cpu_pairs", "wait_us", "wait_max_us", "tasks",
             "places", "why", "gpu_us", "gpu_max_us", "many", "none", "prep_us", "match_us", "choose_us",
             "copy_us", "memo", "woke")  # gpumatch.c's GS_* (rf_gpu's stats)
GPU_WHY = {1: "no hardware adapter", 2: "no device", 3: "no fence", 4: "its shader didn't compile",
           5: "its buffers couldn't be made", 6: "no Direct3D 11 here, or the DLL built without its headers",
           7: "the device removed: a driver reset?", 8: "an error sending a call"}  # gpumatch.c's gm_off


def gpu_line() -> str:
    """The perception service's status line's part on the GPU backend (localization.service.Publisher.status, every
    STATUS_S), "" while no `Refinds` has one: each one's state and what it did since it was made (calls answered in
    time of those asked, late, the CPU's meanwhile, windows left to the memo, its wait a call)."""
    with _open_lock:
        open_ = list(_open)
    parts = []
    for rf in open_:
        state, st = rf.gpu_state()
        if state == "none":
            continue
        sent = st["answered"] + st["late"]
        did = (f"{st['answered']:,} of {st['calls']:,} calls answered, {st['late']} late, {st['busy']:,} the CPU's after"
               f" a late one, {st['memo']:,} windows the memo's" + (
                   f", its wait {st['wait_us'] / sent / 1000:.1f} ms a call ({st['wait_max_us'] / 1000:.1f} at most)"
                   if sent else ""))
        if state == "off":
            parts.append(f"the re-finds' GPU off ({GPU_WHY.get(st['why'], st['why'])}): {did}")
        elif state == "starting":
            parts.append("the re-finds' GPU starting (its device being made)")
        else:
            parts.append(f"the re-finds' GPU on: {did}")
    return "; ".join(parts)


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
        self._ctx_lock = threading.Lock()  # (the context's life: `close`, and `gpu_state` from another thread)
        self._held = None  # what the C reads during a call
        self.ctx = None
        if self._lib is not None and self.pictures:
            n = len(self.pictures)
            self.ctx = self._lib.rf_open((ctypes.c_void_p * n)(*[p.address for p in self.pictures]), n, threads) or None
        if self.ctx is not None:
            with _open_lock:
                _open.add(self)

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

    def gpu_state(self) -> tuple[str, dict[str, int]]:
        """Its GPU backend's state ("none": made with the GPU off; "starting", "on", "off": couldn't be made, or lost)
        and what it did (GPU_STATS: calls asked of it, answered in time, late, ...)."""
        stats = (ctypes.c_longlong * len(GPU_STATS))()
        with self._ctx_lock:  # (not the call's lock: read from the status line's thread while a call runs)
            if self.ctx is None:
                return "none", {}
            state = self._lib.rf_gpu(self.ctx, stats, len(stats))
        return ("none", "starting", "on", "off")[state], dict(zip(GPU_STATS, stats))

    def close(self) -> None:
        with self._lock, self._ctx_lock:
            if self.ctx is not None:
                self._lib.rf_close(self.ctx)  # (never waits for the GPU's device: gpumatch.c's gm_close)
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


def search_state() -> str:
    """Whether the monster search's step 1 is in C (search.c), for the consoles."""
    if search_lib() is not None:
        return "in use"
    return f"off ({search_why or 'not built'}): numpy and OpenCV instead, slower"


def main() -> int:
    try:
        print(f"built: {build()}")
        print(f"built: {build(SEARCH_SOURCE)}")
    except RuntimeError as e:
        print(f"couldn't build gamelab's native pieces: {e}")
        return 1
    print(f"gamelab's C pieces: {state()}; the monster search's step 1: {search_state()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
