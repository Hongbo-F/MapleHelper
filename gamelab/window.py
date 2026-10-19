"""Finding the game window, its client area, and bringing it to the front."""

import ctypes
from ctypes import wintypes

from maplehelper.win32 import foreground_title, kernel32, user32  # noqa: F401  (foreground_title re-exported)

user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
user32.FindWindowW.restype = wintypes.HWND
user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
user32.BringWindowToTop.argtypes = (wintypes.HWND,)
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.c_void_p)
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = (wintypes.DWORD, wintypes.DWORD, wintypes.BOOL)
user32.IsIconic.argtypes = (wintypes.HWND,)
user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)


user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)


def move_cursor(x: int, y: int) -> None:
    """Move the mouse pointer to screen pixel x, y (no click)."""
    user32.SetCursorPos(x, y)


def find_game(title: str) -> int:
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


def bring_to_front(title: str) -> bool:
    """Make the game the foreground window without sending it any keys.

    Windows only lets the process that owns the foreground window hand focus away, so this
    briefly attaches to that window's input thread first.
    """
    game = find_game(title)
    if foreground_title() == title:
        return True
    if user32.IsIconic(game):
        user32.ShowWindow(game, 9)  # SW_RESTORE
    me = kernel32.GetCurrentThreadId()
    other = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
    user32.AttachThreadInput(me, other, True)
    try:
        user32.BringWindowToTop(game)
        user32.SetForegroundWindow(game)
    finally:
        user32.AttachThreadInput(me, other, False)
    return True
