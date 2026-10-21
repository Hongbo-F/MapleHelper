"""Finding the game window, its client area, and bringing it to the front; moving the mouse pointer.

The elevated key player imports this: keep its imports to the standard library and maplehelper.win32.
"""

import ctypes
import time
from ctypes import wintypes

from maplehelper.win32 import foreground_title, kernel32, user32  # noqa: F401  (foreground_title re-exported)

SW_RESTORE = 9  # ShowWindow: un-minimize
FRONT_POLL_S = 0.05  # bring_to_front: how often it looks whether the game came to the front
# Windows doesn't always let a program bring another window to the front (its foreground lock), and then
# SetForegroundWindow does nothing, quietly: on 2026-10-21 at 01:01:30 a tour's looks saw the window in front of
# the game for 2.5 s and the player refused its first plan ("game not in front (foreground: 'Claude')"; the
# player's own try had failed too; its log has two more such refusals on nights before). If it hasn't come within
# half the wait, bring_to_front also switches to it as Alt+Tab does (SwitchToThisWindow: no keys, but documented
# as not for general use, and not tried in game yet). False: SetForegroundWindow alone.
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


def move_cursor(x: int, y: int) -> None:
    """Move the mouse pointer to screen pixel x, y (no click)."""
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


def shows(hwnd: int, area: dict, inset: int = 8) -> bool:
    """Whether the screen shows window `hwnd` at its client area's middle and near each corner (`inset` px
    in): no other window is over those points. Screen captures see whatever is on top."""
    x0, y0 = area["left"] + inset, area["top"] + inset
    x1, y1 = area["left"] + area["width"] - 1 - inset, area["top"] + area["height"] - 1 - inset
    points = [((x0 + x1) // 2, (y0 + y1) // 2), (x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    return all(user32.GetAncestor(user32.WindowFromPoint(wintypes.POINT(x, y)), GA_ROOT) == hwnd for x, y in points)


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
