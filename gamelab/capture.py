"""Capturing the screen: boxes of it as BGRA pictures (as mss gives them), through DXGI desktop duplication
or mss. Eyes captures the game with this (`[capture] method` in config.toml); the elevated key player
doesn't: it keeps mss and loads as little as it can.

mss copies the screen with GDI, which waits for the compositor's next frame: 16.7 ms a grab whatever the
box's size, 17-46 ms after a pause, so a glance (two grabs) took 33 ms. Desktop duplication is handed each
frame of a monitor as it's shown, with when it was shown (QueryPerformanceCounter's ticks, the clock
time.monotonic reads); a frame stays on the graphics card until a box of it is copied out. A frame comes
only when the screen changes, so the last one is the screen as it is until another comes.

What it takes care of: the first frame after duplicating starts is black, with no present time, so frames
without one (also those where only the mouse pointer moved) don't count. Duplicating is lost on a UAC
prompt, a mode change or a full-screen switch (DXGI_ERROR_ACCESS_LOST), and then it starts again. Anything
else that fails falls back to mss, and DXGI is tried again RETRY_S later. Boxes are in desktop pixels, the
physical ones: like mss, the grabber makes the process DPI-aware, so where the game's window is
(gamelab.window) and the monitor's pixels agree. Only one duplication of a monitor can be open in a
process at a time, so the process shares one grabber (`shared`).

Imports: the standard library, numpy, mss and maplehelper.win32 (no other gamelab layer).
"""

import ctypes
import threading
import time
import tomllib
import uuid
import weakref
from collections.abc import Callable
from ctypes import POINTER, byref, c_int64, c_uint, c_ulong, c_void_p, wintypes

import mss
import numpy as np

from maplehelper.win32 import enable_dpi_awareness

from .paths import CONFIG

METHODS = ("dxgi", "mss")
# A look waits this long at most for a frame shown after it began, as mss waited for the compositor's next
# frame: a frame of the game's (one every 16.7 ms) and the ~2 ms a frame comes after the time it was shown
# (present times were about 2 ms older than the frame's arrival, on the desktop), and a little more. When
# none comes by then, nothing on screen has changed since the frame held, which is then the screen as it
# is. 15 ms was shorter than a frame: against a duplication presenting every 16.7 ms with frames arriving
# 0-2 ms late, 6-18% of looks were handed the frame shown 1-3 ms before they began (tests/test_capture.py).
# AcquireNextFrame's waits end on Windows' 15.6 ms timer ticks: with nothing changing, a timeout of 1-15 ms
# took 16 ms, 17-20 ms took 31 and 50 ms took 63; a frame that comes ends the wait at once. So a look at a
# screen where nothing moves takes 31 ms, not 16; the game's screen always moves.
FRESH_WAIT_S = 0.020
# After duplicating starts: how long to wait for the first real frame (on the desktop it came 12 ms after
# the black one, even with nothing on screen changing).
FIRST_FRAME_S = 0.25
RETRY_S = 10.0  # after DXGI failed: mss for this long, then DXGI again

# HRESULTs (as unsigned 32-bit numbers)
S_OK = 0
DXGI_ERROR_INVALID_CALL = 0x887A0001
DXGI_ERROR_NOT_FOUND = 0x887A0002
DXGI_ERROR_ACCESS_LOST = 0x887A0026
DXGI_ERROR_WAIT_TIMEOUT = 0x887A0027
LOST = {DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_INVALID_CALL}  # duplicating again mends these

Area = dict[str, int]          # a box of the desktop as mss takes it: left, top, width, height
Box = tuple[int, int, int, int]  # x0, y0, x1, y1 (x1, y1 not included)


class CaptureError(RuntimeError):
    """Desktop duplication can't give a frame now (the grabber falls back to mss)."""


def configured_method(path=CONFIG) -> str:
    """config.toml's `[capture] method`: "dxgi" (the default) or "mss"."""
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return "dxgi"
    method = data.get("capture", {}).get("method", "dxgi")
    if method not in METHODS:
        raise ValueError(f"{path}: [capture] method must be one of {', '.join(map(repr, METHODS))}, got {method!r}")
    return method


# --- shots: one moment of the screen -----------------------------------------------------------------

Convert = Callable[[np.ndarray], np.ndarray]  # a BGRA box -> a picture of its own (e.g. RGB)


class Shot:
    """One moment of the screen, to copy boxes of: `crop(x, y, width, height)` in px of the shot's `area`,
    BGRA, each a picture of its own; or what `convert` makes of each (it's given a view it mustn't keep:
    converted straight from the graphics card's copy, a whole-area RGB frame took 2.0 ms instead of 3.8).
    `t`: when the screen looked like that (time.monotonic's clock)."""

    def __init__(self, area: Area, t: float | None = None):
        self.area, self.t = area, t

    def crop(self, x: int, y: int, width: int, height: int, convert: Convert | None = None) -> np.ndarray:
        return self.crops([(x, y, width, height)], convert)[0]

    def crops(self, boxes: list[tuple[int, int, int, int]], convert: Convert | None = None) -> list[np.ndarray]:
        """Several boxes (x, y, width, height) of the same moment."""
        raise NotImplementedError


class MssShot(Shot):
    """mss keeps no frames: each box is a grab of its own, which waits for the screen's next frame (`t`:
    when the last one came)."""

    def __init__(self, grabber: "Grabber", area: Area, t: float | None = None):
        super().__init__(area, t)
        self.grabber = grabber

    def crops(self, boxes, convert=None):
        out = []
        for x, y, w, h in boxes:
            grab = np.asarray(self.grabber.mss_screen().grab(
                {"left": self.area["left"] + x, "top": self.area["top"] + y, "width": w, "height": h}))
            self.t = self.grabber.clock()
            out.append(grab if convert is None else convert(grab))
        return out


class DxgiShot(Shot):
    """The frame the duplicator holds: boxes of it are copied off the graphics card. Valid until the
    grabber's next shot."""

    def __init__(self, grabber: "Grabber", area: Area, t: float, seq: int):
        super().__init__(area, t)
        self.grabber, self.seq = grabber, seq

    def crops(self, boxes, convert=None):
        g = self.grabber
        with g.lock:
            if self.seq != g.seq:
                raise RuntimeError("a shot's boxes are copied before the grabber's next shot")
            if g.dup is not None:
                x0, y0 = self.area["left"] - g.dup.desktop[0], self.area["top"] - g.dup.desktop[1]
                try:
                    return g.dup.read([(x0 + x, y0 + y, x0 + x + w, y0 + y + h) for x, y, w, h in boxes], convert)
                except CaptureError as e:
                    g.failed(e)
        return MssShot(g, self.area, self.t).crops(boxes, convert)


# --- the grabber -------------------------------------------------------------------------------------

def _open_mss():
    return mss.MSS()


class Grabber:
    """Shots of the screen: through desktop duplication (`method` "dxgi", the default, falling back to mss
    by itself when it can't be used), or mss. `opener` opens a monitor's duplication (Dxgi; tests pass
    fakes), `clock` and `sleep` are time.monotonic's and time.sleep's stand-ins."""

    def __init__(self, method: str | None = None, opener: Callable | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.method = method or configured_method()
        if self.method not in METHODS:
            raise ValueError(f"capture method must be one of {METHODS}, got {self.method!r}")
        self.opener, self.clock, self.sleep = opener, clock, sleep
        self.dup: Duplicator | None = None
        self._mss = None
        self.retry_at = 0.0
        self.counts = {"dxgi": 0, "mss": 0, "failed": 0, "duplicated again": 0}  # shots by method, and troubles
        self.error: str | None = None  # why DXGI failed last
        self.seq = 0                   # shots taken: a DxgiShot is only good until the next
        self.lock = threading.RLock()
        enable_dpi_awareness()  # window places in the monitor's real pixels (mss does the same)

    def shot(self, area: Area, after: float | None = None, at: float | None = None) -> Shot:
        """The screen at a moment, for boxes of `area` (desktop px): the newest frame; with `after`
        (time.monotonic's clock), a frame shown then or later (mss's grabs wait for the next frame anyway);
        with `at`, the newest frame once it's that time (it waits till then)."""
        if at is not None and (wait := at - self.clock()) > 0:
            self.sleep(wait)
        with self.lock:
            self.seq += 1
            if self.method == "dxgi" and self.clock() >= self.retry_at:
                try:
                    dup = self._duplicator(area)
                    if dup is not None:
                        again = dup.again
                        try:
                            t = dup.advance(after)
                        finally:
                            self.counts["duplicated again"] += dup.again - again
                        self.counts["dxgi"] += 1
                        return DxgiShot(self, area, t, self.seq)
                except CaptureError as e:
                    self.failed(e)
            self.counts["mss"] += 1
            return MssShot(self, area)

    def _duplicator(self, area: Area) -> "Duplicator | None":
        """The duplication of the monitor `area` is on (opened, or moved to another monitor, as needed);
        None if the area spans monitors (mss then)."""
        box = (area["left"], area["top"], area["left"] + area["width"], area["top"] + area["height"])
        centre = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        if self.dup is None or not self.dup.shows(centre):
            self.close_dxgi()
            self.dup = Duplicator(centre, self.opener, self.clock)
        return self.dup if self.dup.covers(box) else None

    def failed(self, e: CaptureError) -> None:
        """DXGI failed: mss until RETRY_S from now."""
        self.close_dxgi()
        self.counts["failed"] += 1
        self.error = str(e)
        self.retry_at = self.clock() + RETRY_S

    def mss_screen(self):
        if self._mss is None:
            self._mss = _open_mss()
        return self._mss

    def close_dxgi(self) -> None:
        if self.dup is not None:
            self.dup.close()
            self.dup = None

    def close(self) -> None:
        with self.lock:
            self.close_dxgi()
            if self._mss is not None:
                self._mss.close()
                self._mss = None

    def summary(self) -> str:
        """Shots by method and what went wrong, for a report."""
        text = f"capture {self.method}: " + ", ".join(f"{k} {v}" for k, v in self.counts.items())
        return text + (f"; DXGI failed last with: {self.error}" if self.error else "")


_SHARED: Grabber | None = None
_SHARED_LOCK = threading.Lock()


def shared() -> Grabber:
    """The process's grabber, as config.toml says (one duplication of a monitor can be open at a time)."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = Grabber()
        return _SHARED


# --- duplicating a monitor: the logic ------------------------------------------------------------------

class Duplicator:
    """Frames of the monitor that shows desktop pixel `point`, from its duplication (`opener`: Dxgi).
    Holds the newest frame (on the graphics card) and when the screen showed it (`t`)."""

    def __init__(self, point: tuple[int, int], opener: Callable | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.point, self.opener, self.clock = point, opener or Dxgi, clock
        self.com = self.opener(point)
        self.t: float | None = None  # None: no frame yet since duplicating started
        self.frames = 0              # frames of the screen taken
        self.again = 0               # times duplicating was lost and started again

    @property
    def desktop(self) -> Box:
        """Where the monitor is on the desktop."""
        return self.com.desktop

    def shows(self, point: tuple[int, int]) -> bool:
        x0, y0, x1, y1 = self.desktop
        return x0 <= point[0] < x1 and y0 <= point[1] < y1

    def covers(self, box: Box) -> bool:
        x0, y0, x1, y1 = self.desktop
        return x0 <= box[0] and y0 <= box[1] and box[2] <= x1 and box[3] <= y1

    def advance(self, after: float | None = None) -> float:
        """Take the newest frame, and give when the screen showed it. With `after`, a frame shown then or
        later, waited for (FRESH_WAIT_S at most); if none comes, the screen hasn't changed since the one
        held. Raises CaptureError if there's no frame at all."""
        start, timeout_ms, lost = self.clock(), 0, False
        while True:
            hr, present = self.com.acquire(timeout_ms)
            if hr == S_OK and present:  # 0: the black first frame, or only the pointer moved
                self.t = self.com.to_monotonic(present)
                self.frames += 1
            elif hr in LOST and not lost:
                lost = True
                self._again()
                start, timeout_ms = self.clock(), 0
                continue
            elif hr not in (S_OK, DXGI_ERROR_WAIT_TIMEOUT):
                raise CaptureError(f"AcquireNextFrame failed: 0x{hr:08X}" + (" (duplicated again)" if lost else ""))
            if self.t is not None and (after is None or self.t >= after):
                return self.t
            limit = FIRST_FRAME_S if self.t is None else FRESH_WAIT_S if after is not None else 0.0
            left = start + limit - self.clock()
            if left <= 0:
                if self.t is None:
                    raise CaptureError(f"no frame of the screen within {limit} s of duplicating it")
                return self.t
            timeout_ms = max(1, round(1000 * left))

    def _again(self) -> None:
        """Duplicating was lost (a UAC prompt, a mode change): start it again. Raises CaptureError if it
        can't be now."""
        self.com.close()
        self.t = None
        self.com = self.opener(self.point)
        self.again += 1

    def read(self, boxes: list[Box], convert: Convert | None = None) -> list[np.ndarray]:
        """Boxes (monitor px) of the frame held, BGRA (or as `convert` makes them)."""
        return self.com.read(boxes, convert)

    def close(self) -> None:
        self.com.close()


# --- duplicating a monitor: the COM calls (ctypes, no packages) ------------------------------------------

HRESULT = ctypes.c_long
MASK = 0xFFFFFFFF


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8)]


def _guid(text: str) -> _GUID:
    u = uuid.UUID(text)
    return _GUID(u.fields[0], u.fields[1], u.fields[2], (ctypes.c_ubyte * 8)(*u.bytes[8:]))


IID_IDXGIFactory1 = _guid("770aae78-f26f-4dba-a829-253c83d1b387")
IID_IDXGIOutput1 = _guid("00cddea8-939b-4b83-a340-a685226666cc")
IID_ID3D11Texture2D = _guid("6f15aaf2-d208-4e89-9ab4-489535d34f9c")

# Each method's place in its interface's table (declaration order, IUnknown's three first).
QUERY_INTERFACE, RELEASE = 0, 2
FACTORY1_ENUM_ADAPTERS1 = 12
ADAPTER_ENUM_OUTPUTS = 7
OUTPUT_GET_DESC = 7
OUTPUT1_DUPLICATE_OUTPUT = 22
DUPL_GET_DESC, DUPL_ACQUIRE_NEXT_FRAME, DUPL_RELEASE_FRAME = 7, 8, 14
DEVICE_CREATE_TEXTURE2D = 5
CONTEXT_MAP, CONTEXT_UNMAP, CONTEXT_COPY_SUBRESOURCE_REGION, CONTEXT_COPY_RESOURCE = 14, 15, 46, 47
TEXTURE2D_GET_DESC = 10

D3D_DRIVER_TYPE_UNKNOWN = 0         # the device is made on the adapter given
D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
D3D11_SDK_VERSION = 7
DXGI_FORMAT_B8G8R8A8_UNORM = 87     # what DuplicateOutput's frames always are
USAGE_DEFAULT, USAGE_STAGING = 0, 3
CPU_ACCESS_READ = 0x20000
MAP_READ = 1
ROTATION_IDENTITY = 1


class _OutputDesc(ctypes.Structure):
    _fields_ = [("DeviceName", ctypes.c_wchar * 32), ("DesktopCoordinates", wintypes.RECT),
                ("AttachedToDesktop", wintypes.BOOL), ("Rotation", c_uint), ("Monitor", c_void_p)]


class _DuplDesc(ctypes.Structure):  # DXGI_OUTDUPL_DESC, its DXGI_MODE_DESC flattened
    _fields_ = [("Width", c_uint), ("Height", c_uint), ("RefreshNumerator", c_uint), ("RefreshDenominator", c_uint),
                ("Format", c_uint), ("ScanlineOrdering", c_uint), ("Scaling", c_uint), ("Rotation", c_uint),
                ("DesktopImageInSystemMemory", wintypes.BOOL)]


class _TextureDesc(ctypes.Structure):
    _fields_ = [(name, c_uint) for name in ("Width", "Height", "MipLevels", "ArraySize", "Format", "SampleCount",
                                            "SampleQuality", "Usage", "BindFlags", "CPUAccessFlags", "MiscFlags")]


class _D3DBox(ctypes.Structure):
    _fields_ = [(name, c_uint) for name in ("left", "top", "front", "right", "bottom", "back")]


class _Mapped(ctypes.Structure):
    _fields_ = [("pData", c_void_p), ("RowPitch", c_uint), ("DepthPitch", c_uint)]


class _FrameInfo(ctypes.Structure):  # DXGI_OUTDUPL_FRAME_INFO
    _fields_ = [("LastPresentTime", c_int64), ("LastMouseUpdateTime", c_int64), ("AccumulatedFrames", c_uint),
                ("RectsCoalesced", wintypes.BOOL), ("ProtectedContentMaskedOut", wintypes.BOOL),
                ("PointerX", ctypes.c_long), ("PointerY", ctypes.c_long), ("PointerVisible", wintypes.BOOL),
                ("TotalMetadataBufferSize", c_uint), ("PointerShapeBufferSize", c_uint)]


_PROTOTYPES: dict = {}


def _call(obj: c_void_p, index: int, *args, argtypes: tuple = (), restype=HRESULT):
    """Method `index` of COM interface `obj`, called with `args` (ctypes lets go of Python's lock meanwhile)."""
    proto = _PROTOTYPES.get((restype, argtypes))
    if proto is None:
        proto = _PROTOTYPES[restype, argtypes] = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)
    table = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
    return proto(table[index])(obj, *args)


def _check(hr: int, what: str) -> None:
    if hr & MASK:
        raise CaptureError(f"{what} failed: 0x{hr & MASK:08X}")


def _release(obj: c_void_p) -> None:
    if obj.value:
        _call(obj, RELEASE, restype=c_ulong)
        obj.value = None


def _release_all(held: list[c_void_p]) -> None:
    while held:
        _release(held.pop())


def qpc() -> int:
    """QueryPerformanceCounter's ticks now."""
    ticks = c_int64()
    ctypes.windll.kernel32.QueryPerformanceCounter(byref(ticks))
    return ticks.value


def qpc_frequency() -> int:
    freq = c_int64()
    ctypes.windll.kernel32.QueryPerformanceFrequency(byref(freq))
    return freq.value


def qpc_offset(freq: int) -> float:
    """What to add to QueryPerformanceCounter's seconds to get time.monotonic's (0 where monotonic reads
    the same counter, as it does on Windows; measured, not assumed)."""
    before = time.monotonic()
    ticks = qpc()
    return (before + time.monotonic()) / 2 - ticks / freq


class Dxgi:
    """One monitor's desktop duplication: the COM objects (Duplicator decides what to do with them).
    `point`: a desktop pixel the monitor shows. The newest frame is copied into a texture of its own on
    the graphics card as it's taken (DXGI's own is handed back at once), and boxes of it are copied off
    the card through a staging texture. Raises CaptureError if the monitor can't be duplicated."""

    def __init__(self, point: tuple[int, int]):
        self._held: list[c_void_p] = []  # every COM pointer this holds, released in reverse
        self._finalizer = weakref.finalize(self, _release_all, self._held)
        self._format_checked = False
        try:
            self._open(point)
        except BaseException:
            self.close()
            raise

    def _keep(self, obj: c_void_p) -> c_void_p:
        self._held.append(obj)
        return obj

    def _open(self, point: tuple[int, int]) -> None:
        factory = self._keep(c_void_p())
        _check(ctypes.WinDLL("dxgi").CreateDXGIFactory1(byref(IID_IDXGIFactory1), byref(factory)),
               "CreateDXGIFactory1")
        adapter, output = self._find(factory, point)
        device, context, level = self._keep(c_void_p()), self._keep(c_void_p()), c_uint()
        _check(ctypes.WinDLL("d3d11").D3D11CreateDevice(
            adapter, D3D_DRIVER_TYPE_UNKNOWN, None, D3D11_CREATE_DEVICE_BGRA_SUPPORT, None, 0, D3D11_SDK_VERSION,
            byref(device), byref(level), byref(context)), "D3D11CreateDevice")
        output1 = self._keep(c_void_p())
        _check(_call(output, QUERY_INTERFACE, byref(IID_IDXGIOutput1), byref(output1), argtypes=(c_void_p, c_void_p)),
               "IDXGIOutput1")
        dupl = self._keep(c_void_p())
        _check(_call(output1, OUTPUT1_DUPLICATE_OUTPUT, device, byref(dupl), argtypes=(c_void_p, c_void_p)),
               "DuplicateOutput")
        desc = _DuplDesc()
        _call(dupl, DUPL_GET_DESC, byref(desc), argtypes=(c_void_p,), restype=None)
        if desc.Rotation not in (0, ROTATION_IDENTITY):
            raise CaptureError("the monitor is rotated")  # its frames come unrotated
        w, h = desc.Width, desc.Height
        if (w, h) != (self.desktop[2] - self.desktop[0], self.desktop[3] - self.desktop[1]):
            raise CaptureError(f"the monitor's frames are {w}x{h}, its place on the desktop {self.desktop}")
        self.size = (w, h)
        self._device, self._context, self._dupl = device, context, dupl
        self._last = self._texture(USAGE_DEFAULT, 0)
        self._staging = self._texture(USAGE_STAGING, CPU_ACCESS_READ)
        self._freq = qpc_frequency()
        self._offset = qpc_offset(self._freq)

    def _find(self, factory: c_void_p, point: tuple[int, int]) -> tuple[c_void_p, c_void_p]:
        """The adapter and output (monitor) showing `point`."""
        i = 0
        while True:
            adapter = c_void_p()
            hr = _call(factory, FACTORY1_ENUM_ADAPTERS1, i, byref(adapter), argtypes=(c_uint, c_void_p)) & MASK
            if hr == DXGI_ERROR_NOT_FOUND:
                raise CaptureError(f"no monitor shows desktop pixel {point}")
            _check(hr, "EnumAdapters1")
            try:
                j = 0
                while True:
                    output = c_void_p()
                    hr = _call(adapter, ADAPTER_ENUM_OUTPUTS, j, byref(output), argtypes=(c_uint, c_void_p)) & MASK
                    if hr == DXGI_ERROR_NOT_FOUND:
                        break
                    _check(hr, "EnumOutputs")
                    try:
                        desc = _OutputDesc()
                        hr = _call(output, OUTPUT_GET_DESC, byref(desc), argtypes=(c_void_p,))
                        r = desc.DesktopCoordinates
                        if (not hr & MASK and desc.AttachedToDesktop
                                and r.left <= point[0] < r.right and r.top <= point[1] < r.bottom):
                            self.desktop = (r.left, r.top, r.right, r.bottom)
                            self._keep(c_void_p(adapter.value))  # held from here on (and released at close)
                            self._keep(c_void_p(output.value))
                            adapter.value = output.value = None
                            return self._held[-2], self._held[-1]
                    finally:
                        _release(output)
                    j += 1
            finally:
                _release(adapter)
            i += 1

    def _texture(self, usage: int, cpu_access: int) -> c_void_p:
        """A texture the monitor's size, as its frames are."""
        desc = _TextureDesc(self.size[0], self.size[1], 1, 1, DXGI_FORMAT_B8G8R8A8_UNORM, 1, 0, usage, 0, cpu_access, 0)
        texture = self._keep(c_void_p())
        _check(_call(self._device, DEVICE_CREATE_TEXTURE2D, byref(desc), None, byref(texture),
                     argtypes=(c_void_p, c_void_p, c_void_p)), "CreateTexture2D")
        return texture

    def to_monotonic(self, ticks: int) -> float:
        """A present time (QueryPerformanceCounter's ticks) on time.monotonic's clock."""
        return ticks / self._freq + self._offset

    def acquire(self, timeout_ms: int) -> tuple[int, int]:
        """Wait up to `timeout_ms` for the monitor's next frame, and keep a copy of it on the card if the
        picture changed: (the HRESULT, its present time in QueryPerformanceCounter's ticks; 0 if the
        picture didn't change: the black first frame, or only the pointer moved)."""
        info, resource = _FrameInfo(), c_void_p()
        hr = _call(self._dupl, DUPL_ACQUIRE_NEXT_FRAME, timeout_ms, byref(info), byref(resource),
                   argtypes=(c_uint, c_void_p, c_void_p)) & MASK
        if hr != S_OK:
            return hr, 0
        try:
            if info.LastPresentTime:
                texture = c_void_p()
                _check(_call(resource, QUERY_INTERFACE, byref(IID_ID3D11Texture2D), byref(texture),
                             argtypes=(c_void_p, c_void_p)), "a frame's texture")
                try:
                    if not self._format_checked:
                        desc = _TextureDesc()
                        _call(texture, TEXTURE2D_GET_DESC, byref(desc), argtypes=(c_void_p,), restype=None)
                        if (desc.Width, desc.Height, desc.Format) != (*self.size, DXGI_FORMAT_B8G8R8A8_UNORM):
                            raise CaptureError(f"frames are {desc.Width}x{desc.Height} in format {desc.Format}")
                        self._format_checked = True
                    _call(self._context, CONTEXT_COPY_RESOURCE, self._last, texture, argtypes=(c_void_p, c_void_p),
                          restype=None)
                finally:
                    _release(texture)
        finally:
            _release(resource)
            _call(self._dupl, DUPL_RELEASE_FRAME)  # a failure here shows in the next AcquireNextFrame
        return S_OK, info.LastPresentTime

    def read(self, boxes: list[Box], convert: Convert | None = None) -> list[np.ndarray]:
        """Boxes (monitor px) of the frame kept, BGRA, each a picture of its own (or what `convert` makes of
        each, given a view of the mapped texture)."""
        w, h = self.size
        for x0, y0, x1, y1 in boxes:
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError(f"box {(x0, y0, x1, y1)} isn't on the {w}x{h} monitor")
            _call(self._context, CONTEXT_COPY_SUBRESOURCE_REGION, self._staging, 0, x0, y0, 0, self._last, 0,
                  byref(_D3DBox(x0, y0, 0, x1, y1, 1)),
                  argtypes=(c_void_p, c_uint, c_uint, c_uint, c_uint, c_void_p, c_uint, c_void_p), restype=None)
        mapped = _Mapped()
        _check(_call(self._context, CONTEXT_MAP, self._staging, 0, MAP_READ, 0, byref(mapped),
                     argtypes=(c_void_p, c_uint, c_uint, c_uint, c_void_p)), "Map")
        try:
            rows = np.ctypeslib.as_array(ctypes.cast(mapped.pData, POINTER(ctypes.c_ubyte)), shape=(h, mapped.RowPitch))
            views = [rows[y0:y1, 4 * x0:4 * x1].reshape(y1 - y0, x1 - x0, 4) for x0, y0, x1, y1 in boxes]
            return [np.array(v) if convert is None else convert(v) for v in views]
        finally:
            _call(self._context, CONTEXT_UNMAP, self._staging, 0, argtypes=(c_void_p, c_uint), restype=None)

    def close(self) -> None:
        self._finalizer()
