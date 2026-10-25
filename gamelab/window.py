"""Finding the game window, its client area, and bringing it to the front; moving the mouse pointer.

The elevated key player imports this: keep its imports to the standard library and maplehelper.win32.
"""

import ctypes
import time
from ctypes import wintypes

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
user32.WindowFromPoint.argtypes = (wintypes.POINT,)
user32.WindowFromPoint.restype = wintypes.HWND
user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
user32.GetAncestor.restype = wintypes.HWND
GA_ROOT = 2  # GetAncestor: the top-level window
user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
user32.AdjustWindowRectEx.argtypes = (ctypes.POINTER(wintypes.RECT), wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
user32.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                wintypes.UINT)
# The game's client sizes the player may set (resize_client): the game lays out a 1366x768 view and stretches it to
# its window (screen.Geometry.stretch), so a 1366x768 client shows it pixel for pixel; 1920x1080 is how it was set until
# 2026-10-23. A normal program can't resize the game's window (it runs as administrator: "access denied", tried
# 2026-10-23), so only the elevated player does it.
CLIENT_SIZES = ((1366, 768), (1920, 1080))


def move_cursor(x: int, y: int) -> None:
    """Move the mouse pointer to screen pixel x, y (no click): as a mouse would (win32.send_mouse_move: an item dragged
    in the game follows it), then exactly there (SetCursorPos: the absolute coordinates are 1/65536 of the desktop)."""
    try:
        send_mouse_move(x, y)
    except OSError:
        pass
    user32.SetCursorPos(x, y)


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


def shows(hwnd: int, area: dict, inset: int = 8) -> bool:
    """Whether the screen shows window `hwnd` at its client area's middle and near each corner (`inset` px
    in): no other window is over those points. Screen captures see whatever is on top."""
    x0, y0 = area["left"] + inset, area["top"] + inset
    x1, y1 = area["left"] + area["width"] - 1 - inset, area["top"] + area["height"] - 1 - inset
    points = [((x0 + x1) // 2, (y0 + y1) // 2), (x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    return all(user32.GetAncestor(user32.WindowFromPoint(wintypes.POINT(x, y)), GA_ROOT) == hwnd for x, y in points)


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
