"""Finding the game window, its client area, whether it's full screen, and bringing it to the front; moving the mouse
pointer along a human-like trail.

The elevated key player imports this: keep its imports to the standard library and maplehelper's input helpers.
"""

import ctypes
import time
from ctypes import wintypes

from maplehelper.mouse_trail import generate_trail
from maplehelper.win32 import foreground_title, kernel32, send_mouse_move, user32  # noqa: F401  (foreground_title re-exported)

SW_RESTORE = 9  # ShowWindow: un-minimize
FRONT_POLL_S = 0.05  # bring_to_front: how often it looks whether the game came to the front
# Windows doesn't always let a program bring another window to the front (its foreground lock), and then
# SetForegroundWindow does nothing, quietly: on 2026-10-21 at 01:01:30 a tour's looks saw the window in front of
# the game for 2.5 s and the player refused its first plan ("game not in front (foreground: 'Claude')"; the
# player's own try had failed too; its log has two more such refusals on nights before). If it hasn't come within
# half the wait, bring_to_front also switches to it as Alt+Tab does (SwitchToThisWindow: no keys, but documented
# as not for general use, and not tried in game yet). On though untried, as it can only help: it runs only once
# SetForegroundWindow has been seen to fail (the game still not in front half way through the wait), it sends no
# keys and clicks nothing, and without it the command stops there anyway (__main__.FRONT_WAIT_S), the game behind.
# Worst case it does nothing, as SetForegroundWindow did. False: SetForegroundWindow alone (if in game it's seen
# to do anything but bring the game forward, such as flash its taskbar button or restore a window it shouldn't).
SWITCH_FALLBACK = True

user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
user32.FindWindowW.restype = wintypes.HWND
user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
user32.SwitchToThisWindow.argtypes = (wintypes.HWND, wintypes.BOOL)
user32.SwitchToThisWindow.restype = None
user32.BringWindowToTop.argtypes = (wintypes.HWND,)
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.c_void_p)
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
user32.IsIconic.argtypes = (wintypes.HWND,)
user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
user32.GetCursorPos.restype = wintypes.BOOL
user32.WindowFromPoint.argtypes = (wintypes.POINT,)
user32.MonitorFromPoint.argtypes = (wintypes.POINT, wintypes.DWORD)
user32.MonitorFromPoint.restype = wintypes.HMONITOR
MONITOR_DEFAULTTONULL = 0  # MonitorFromPoint: no monitor (None) for a point on none
user32.WindowFromPoint.restype = wintypes.HWND
user32.MonitorFromWindow.argtypes = (wintypes.HWND, wintypes.DWORD)
user32.MonitorFromWindow.restype = wintypes.HMONITOR
MONITOR_DEFAULTTONEAREST = 2  # MonitorFromWindow: the monitor nearest a window on none


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD)]


user32.GetMonitorInfoW.argtypes = (wintypes.HMONITOR, ctypes.POINTER(MONITORINFO))
user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
user32.GetAncestor.restype = wintypes.HWND
GA_ROOT = 2  # GetAncestor: the top-level window
user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
user32.AdjustWindowRectEx.argtypes = (ctypes.POINTER(wintypes.RECT), wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
user32.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                wintypes.UINT)
user32.GetWindow.argtypes = (wintypes.HWND, wintypes.UINT)
user32.GetWindow.restype = wintypes.HWND
GW_HWNDPREV = 3  # GetWindow: the window just above in the z-order
user32.IsWindowVisible.argtypes = (wintypes.HWND,)
user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
dwmapi = ctypes.WinDLL("dwmapi")
dwmapi.DwmGetWindowAttribute.argtypes = (wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD)
dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long  # an HRESULT: 0 is S_OK
DWMWA_EXTENDED_FRAME_BOUNDS, DWMWA_CLOAKED = 9, 14  # a window's box as drawn; whether it's hidden by the compositor
OVER_MAX = 2000  # `over`: windows above the game looked at at most (the walk can go round if windows change meanwhile)
# Programs whose windows `over` counts as the game's own, by their executable's name: the Nexon launcher, which starts
# the game, keeps an overlay over it the whole time (2026-10-29: class Chrome_WidgetWin_1, 2400x900 at 720,570, layered,
# click-through, topmost, drawing nothing the user sees), so every frame of the night was "covered": the Lie Detector's
# clips and save-recent's saves had none, and the user saw nothing over the game ("It's not, what are you talking
# about?"). What it draws is the game's launcher's, not another program's.
GAME_FAMILY = ("nexon_client.exe",)
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                ctypes.POINTER(wintypes.DWORD))
kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
# The game's client sizes the player may set (resize_client): the game lays out a 1366x768 view and stretches it to
# its window (screen.Geometry.stretch), so a 1366x768 client shows it pixel for pixel; 1920x1080 is how it was set until
# 2026-10-23. A normal program can't resize the game's window (it runs as administrator: "access denied", tried
# 2026-10-23), so only the elevated player does it.
CLIENT_SIZES = ((1366, 768), (1920, 1080))


def move_cursor(x: int, y: int, wait=None):
    """Move the pointer to screen pixel x, y along a generated trail (no click), one whole-pixel mouse event at a time.

    `wait` is the player's plan-aware wait when one is running: it takes the due perf_counter time and returns True to
    carry on, or an early result (False for a stop, "splice" for a new plan) which this returns without another event.
    With none, this sleeps until each sample is due. The last SetCursorPos removes SendInput's sub-pixel rounding.
    """
    here = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(here)):
        raise ctypes.WinError(ctypes.get_last_error())
    trail = generate_trail(here.x, here.y, x, y)
    started = time.perf_counter()
    for ms, px, py in trail[1:]:
        due = started + ms / 1000.0
        if wait is None:
            left = due - time.perf_counter()
            if left > 0:
                time.sleep(left)
        else:
            got = wait(due)
            if got is not True:
                return got
        try:
            send_mouse_move(round(px), round(py))
        except OSError:
            pass
    user32.SetCursorPos(x, y)
    return True


def find_game(title: str) -> int:
    """The game window's handle (raises if no window has that title)."""
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        raise RuntimeError(f"no window titled {title!r}; is the game running?")
    return hwnd


def client_rect(hwnd: int) -> dict:
    """The game's drawing area in screen pixels, as an mss region."""
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return {"left": origin.x, "top": origin.y, "width": r.right, "height": r.bottom}


def minimized(hwnd: int) -> bool:
    return bool(user32.IsIconic(hwnd))


def window_rect(hwnd: int) -> dict:
    """The game window's box as Windows keeps it (GetWindowRect: its title bar and frame, the invisible borders Windows
    10 puts round it to grab included), in screen pixels, as client_rect gives a box. On 2026-10-25 at 23:40 the game
    drew its picture at this box's corner, not its client area's (perception.framecheck.drawn_offset)."""
    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return {"left": r.left, "top": r.top, "width": r.right - r.left, "height": r.bottom - r.top}


def inset(hwnd: int, area: dict) -> tuple[int, int]:
    """Where a client area's corner is from its window's (window_rect): (18, 85) at 300% on 2026-10-25."""
    box = window_rect(hwnd)
    return area["left"] - box["left"], area["top"] - box["top"]


def monitor_of(hwnd: int) -> dict | None:
    """The monitor a window is on (the one most of it is on), in desktop px, as a client area is: left, top, width,
    height; None if Windows can't say."""
    info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
    monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    if not monitor or not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        return None
    r = info.rcMonitor
    return {"left": r.left, "top": r.top, "width": r.right - r.left, "height": r.bottom - r.top}


def full_screen(hwnd: int, area: dict) -> bool:
    """Whether a client area (client_rect's) is the whole of the monitor its window is on: the game full screen,
    whether its own mode set the monitor to its size or it covers the monitor as it is (2026-10-25 23:41: 3840x2160 at
    0,0). Windowed, its title bar is on the monitor too, so the client can't be all of it."""
    return area == monitor_of(hwnd)


def dpi_awareness() -> int | None:
    """This thread's DPI awareness: 0 unaware, 1 system aware, 2 per monitor (window places and captures in the
    monitor's real pixels); None if Windows can't say. On 2026-10-24 python.exe was per monitor (2) before any call
    (its compatibility flag HIGHDPIAWARE), at 300% scaling."""
    try:
        context = user32.GetThreadDpiAwarenessContext()
        user32.GetAwarenessFromDpiAwarenessContext.argtypes = (ctypes.c_void_p,)
        return int(user32.GetAwarenessFromDpiAwarenessContext(context))
    except (AttributeError, OSError):
        return None


def shows(hwnd: int, area: dict, inset: int = 8, grid: tuple[int, int] | None = None) -> bool:
    """Whether the screen shows window `hwnd` at its client area's middle and near each corner (`inset` px
    in): no other window is over those points. Screen captures see whatever is on top. With `grid` (spaces across and
    down), at a grid of points from one of those corners to the other instead."""
    x0, y0 = area["left"] + inset, area["top"] + inset
    x1, y1 = area["left"] + area["width"] - 1 - inset, area["top"] + area["height"] - 1 - inset
    if grid is None:
        points = [((x0 + x1) // 2, (y0 + y1) // 2), (x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    else:
        points = [(x0 + (x1 - x0) * i // grid[0], y0 + (y1 - y0) * j // grid[1])
                  for i in range(grid[0] + 1) for j in range(grid[1] + 1)]
    return all(user32.GetAncestor(user32.WindowFromPoint(wintypes.POINT(x, y)), GA_ROOT) == hwnd for x, y in points)


def over(hwnd: int, area: dict) -> int | None:
    """A window of another program over any of window `hwnd`'s client area `area` (screen px), or None: the windows
    above it in the z-order, one by one to the top, that are shown (visible, not minimized, not hidden by the
    compositor: DWMWA_CLOAKED, as a window on another virtual desktop or a closed Start menu is) and whose box as drawn
    (DWMWA_EXTENDED_FRAME_BOUNDS: without the invisible borders Windows 10 puts round a window to grab, else
    GetWindowRect) meets the area. `shows` looks at points only: a window between them (a notification, a small box
    of 200x150 at COVER_GRID's points 225x188 px apart, the client's left 8 px) or one clicks pass through (a layered
    overlay: WindowFromPoint skips it) passes it. The game's own windows are left out (its picture), and those of
    GAME_FAMILY's programs (its launcher's overlay). 39 windows above the game, none shown over it: 0.1 ms (median of
    50; 0.66 ms in the review's run), 2026-10-24."""
    return next((above for above, _ in _walk(hwnd, area)), None)


def overs(hwnd: int, area: dict) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Every window of another program over window `hwnd`'s client area `area`, as `over` finds the first: (its hwnd,
    its box as drawn: left, top, right, bottom, screen px), the nearest above the game first. The rolling recording
    notes them with a frame rather than dropping it (control.rolling.GameWindow)."""
    return list(_walk(hwnd, area))


def client_boxes(area: dict, boxes) -> list[tuple[int, int, int, int]]:
    """Screen boxes (left, top, right, bottom), as `overs` gives them, as boxes of the client area `area` (its own px:
    x0, y0, x1, y1), cut to the area; those that miss it left out. What the rolling recording notes a frame was covered
    by (control.rolling: Item.over)."""
    w, h = area["width"], area["height"]
    out = []
    for left, top, right, bottom in boxes:
        x0, y0 = max(0, left - area["left"]), max(0, top - area["top"])
        x1, y1 = min(w, right - area["left"]), min(h, bottom - area["top"])
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1, y1))
    return out


def _walk(hwnd: int, area: dict):
    """`over`'s and `overs`' walk up the z-order: each window of another program shown over `area`, with its box."""
    x0, y0 = area["left"], area["top"]
    x1, y1 = x0 + area["width"], y0 + area["height"]
    game = _process(hwnd)
    above = user32.GetWindow(hwnd, GW_HWNDPREV)
    for _ in range(OVER_MAX):
        if not above:
            return
        if user32.IsWindowVisible(above) and not user32.IsIconic(above) and not _cloaked(above) \
                and (pid := _process(above)) != game and not _family(pid):
            left, top, right, bottom = _drawn(above)
            if left < x1 and right > x0 and top < y1 and bottom > y0 and right > left and bottom > top:
                yield above, (left, top, right, bottom)
        above = user32.GetWindow(above, GW_HWNDPREV)


def describe(hwnd: int) -> str:
    """A window as a log names it: its class and box (not its title, which can say what's in it)."""
    name = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, name, len(name))
    left, top, right, bottom = _drawn(hwnd)
    return f"a window of class {name.value!r} at {left},{top} {right - left}x{bottom - top}"


def _process(hwnd: int) -> int:
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


_images: dict[int, str] = {}  # pid -> its executable's name, lower case ("" if it can't be read)


def _image(pid: int) -> str:
    """The executable's file name of process `pid`, lower case ("" if Windows won't say), kept once read."""
    if pid not in _images:
        name = ""
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            try:
                buf, size = ctypes.create_unicode_buffer(1024), wintypes.DWORD(1024)
                if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    name = buf.value.rsplit("\\", 1)[-1].lower()
            finally:
                kernel32.CloseHandle(handle)
        if len(_images) > 512:
            _images.clear()
        _images[pid] = name
    return _images[pid]


def _family(pid: int) -> bool:
    """Whether process `pid` is one of GAME_FAMILY's (the game's launcher): its windows are the game's own."""
    return _image(pid) in GAME_FAMILY


def _cloaked(hwnd: int) -> bool:
    cloaked = wintypes.DWORD(0)
    return dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked)) == 0 \
        and bool(cloaked.value)


def _drawn(hwnd: int) -> tuple[int, int, int, int]:
    """A window's box as drawn (left, top, right, bottom; screen px)."""
    r = wintypes.RECT()
    if dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(r), ctypes.sizeof(r)) != 0:
        user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def on_screen(area: dict) -> bool:
    """Whether every corner of a client area is on a monitor (none of it hangs off the desktop, as a window can after
    the desktop was resized: Chrome Remote Desktop fits it to the device it's used from). Off the desktop
    WindowFromPoint finds no window at all, and `shows` would take that for another window over the game."""
    x0, y0 = area["left"], area["top"]
    x1, y1 = x0 + area["width"] - 1, y0 + area["height"] - 1
    return all(user32.MonitorFromPoint(wintypes.POINT(x, y), MONITOR_DEFAULTTONULL)
               for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)))


def resize_client(hwnd: int, width: int, height: int) -> tuple[int, int]:
    """Resize the game window so its client area is `width` x `height` (one of CLIENT_SIZES), keeping its top-left
    corner where it is; returns the client size it then has. Raises ValueError for any other size, OSError if
    Windows refuses."""
    if (width, height) not in CLIENT_SIZES:
        raise ValueError(f"client size {width}x{height} isn't one of {CLIENT_SIZES}")
    style, ex = user32.GetWindowLongW(hwnd, -16) & 0xFFFFFFFF, user32.GetWindowLongW(hwnd, -20) & 0xFFFFFFFF
    want = wintypes.RECT(0, 0, width, height)
    if not user32.AdjustWindowRectEx(ctypes.byref(want), style, False, ex):
        raise ctypes.WinError(ctypes.get_last_error())
    swp_nomove, swp_nozorder, swp_noactivate = 0x2, 0x4, 0x10
    if not user32.SetWindowPos(hwnd, None, 0, 0, want.right - want.left, want.bottom - want.top,
                               swp_nomove | swp_nozorder | swp_noactivate):
        raise ctypes.WinError(ctypes.get_last_error())
    got = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(got))
    return got.right, got.bottom


def bring_to_front(title: str, wait: float = 0.0) -> bool:
    """Make the game the foreground window without sending it any keys, and say whether it is: waiting up to
    `wait` s for it to come (with SWITCH_FALLBACK, switching to it from half way).

    Windows only lets the process that owns the foreground window hand focus away, so this
    briefly attaches to that window's input thread first.
    """
    game = find_game(title)
    if foreground_title() == title:
        return True
    if user32.IsIconic(game):
        user32.ShowWindow(game, SW_RESTORE)
    me = kernel32.GetCurrentThreadId()
    other = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
    user32.AttachThreadInput(me, other, True)
    try:
        user32.BringWindowToTop(game)
        user32.SetForegroundWindow(game)
    finally:
        user32.AttachThreadInput(me, other, False)
    start, switched = time.monotonic(), False
    while foreground_title() != title:
        waited = time.monotonic() - start
        if waited >= wait:
            return False
        if SWITCH_FALLBACK and not switched and waited >= wait / 2:
            user32.SwitchToThisWindow(game, True)
            switched = True
        time.sleep(FRONT_POLL_S)
    return True
