"""Finding the game window and grabbing regions of its picture (Windows).

Read-only, like any screenshot tool: the window is found by its title, pixels come from a screen grab (mss),
and the game's process is never opened, read or sent input.
"""

import ctypes
import os
import time
from ctypes import wintypes

import numpy as np

from .regions import Box, Region, content_box

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

GA_ROOT = 2
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)

# Windows whose title may mention the game but aren't it: browsers, terminals, Explorer.
NOT_GAME_CLASSES = {
    "Chrome_WidgetWin_0", "Chrome_WidgetWin_1", "MozillaWindowClass", "ApplicationFrameWindow",
    "ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS", "CabinetWClass", "Notepad", "Notepad++",
}

EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.EnumWindows.argtypes = (EnumWindowsProc, wintypes.LPARAM)
user32.IsWindow.argtypes = user32.IsWindowVisible.argtypes = user32.IsIconic.argtypes = (wintypes.HWND,)
user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
user32.GetClientRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
user32.ClientToScreen.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.POINT))
user32.WindowFromPoint.argtypes = (wintypes.POINT,)
user32.WindowFromPoint.restype = wintypes.HWND
user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
user32.GetAncestor.restype = wintypes.HWND
user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
user32.GetWindowLongW.restype = wintypes.LONG
user32.SetProcessDpiAwarenessContext.argtypes = (ctypes.c_void_p,)
kernel32.GetConsoleWindow.restype = wintypes.HWND


def set_dpi_aware() -> None:
    """Window rects in physical pixels, the same as mss grabs, so crops line up on scaled displays."""
    if not user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except OSError:
            pass  # already set for this process, or an old Windows


def window_title(hwnd) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _class_name(hwnd) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _pid(hwnd) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def find_window(title: str) -> int | None:
    """The game's top-level window: a visible window whose title is `title`, else one that contains it
    (case-insensitive). Browsers, terminals and this process's own windows are skipped."""
    needle, own, console = title.lower(), os.getpid(), kernel32.GetConsoleWindow()
    exact, partial = [], []

    @EnumWindowsProc
    def on_window(hwnd, _):
        if user32.IsWindowVisible(hwnd) and hwnd != console and _pid(hwnd) != own:
            name = window_title(hwnd).strip().lower()
            if needle in name and _class_name(hwnd) not in NOT_GAME_CLASSES:
                (exact if name == needle else partial).append(hwnd)
        return True

    user32.EnumWindows(on_window, 0)
    found = exact or partial
    return found[0] if found else None


def client_box(hwnd) -> Box | None:
    """The window's client area (the game's picture, without title bar and borders) in screen pixels;
    None when it's gone or minimized."""
    rect = wintypes.RECT()
    if not user32.IsWindow(hwnd) or user32.IsIconic(hwnd) or not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    if rect.right <= 0 or rect.bottom <= 0:
        return None
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    return origin.x, origin.y, rect.right, rect.bottom


def covered(hwnd, box: Box) -> bool:
    """Whether another app's window is over `box` (a browser in front, a chat window), at its centre or near
    its corners. A screen grab takes whatever is on top, so a covered region must not be read. Click-through
    overlays and this process's own windows don't count."""
    x, y, w, h = box
    own = os.getpid()
    for fx, fy in ((0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)):
        top = user32.GetAncestor(user32.WindowFromPoint(wintypes.POINT(int(x + fx * w), int(y + fy * h))), GA_ROOT)
        if not top or top == hwnd or _pid(top) == own:
            continue
        if user32.GetWindowLongW(top, GWL_EXSTYLE) & WS_EX_TRANSPARENT:
            continue
        return True
    return False


def visible_part(box: Box, monitors: list[dict], slack: int) -> Box | None:
    """The part of `box` on the monitor that shows most of it, or None when more than `slack` px of its width or
    height is off that monitor (a grab gives black there). `monitors` as mss lists them."""
    x, y, w, h = box
    best, best_area = None, 0
    for m in monitors:
        left, top = max(x, m["left"]), max(y, m["top"])
        right, bottom = min(x + w, m["left"] + m["width"]), min(y + h, m["top"] + m["height"])
        if right > left and bottom > top and (right - left) * (bottom - top) > best_area:
            best, best_area = (left, top, right - left, bottom - top), (right - left) * (bottom - top)
    if best is None or w - best[2] > slack or h - best[3] > slack:
        return None
    return best


class GameCapture:
    """Grabs regions of the game's picture. Re-finds the window when it closes, follows it when it moves, and
    re-measures the black bars when its size changes (and every BARS_EVERY_S). Use it from one thread.

    `problem` says why the last grab gave nothing: "not found", "minimized", "covered" or "off screen".
    """

    BARS_EVERY_S = 30.0
    OFF_SCREEN_SLACK = 10  # px of the window that may hang off the screen; that strip is just left out

    def __init__(self, title: str = "MapleStory"):
        import mss  # imported here so regions/parse stay usable without it

        set_dpi_aware()
        self.title = title
        self.problem: str | None = None
        self._sct = mss.MSS()
        self._screenshot_error = mss.exception.ScreenShotError
        self._hwnd = None
        self._bars: tuple[tuple[int, int], Box] | None = None  # (client size, picture box inside the client)
        self._bars_at = 0.0

    def play_area(self) -> Box | None:
        """The game's picture on screen (x, y, w, h), black bars left out; None with `problem` set."""
        hwnd = self._window()
        if hwnd is None:
            self.problem = "not found"
            return None
        client = client_box(hwnd)
        if client is None:
            self.problem = "minimized"
            return None
        size, now = client[2:], time.monotonic()
        stale = self._bars is None or self._bars[0] != size or now - self._bars_at > self.BARS_EVERY_S
        # Bars are measured on the whole picture only: an edge cut off by the screen would look like a bar.
        if stale and self._visible(client, slack=0) == client and not covered(hwnd, client):
            picture = self._shot(client)
            if picture is not None:
                self._bars, self._bars_at = (size, content_box(picture)), now
        if self._bars is None or self._bars[0] != size:
            return client
        bx, by, bw, bh = self._bars[1]
        return client[0] + bx, client[1] + by, bw, bh

    def grab(self, region: Region | None = None) -> np.ndarray | None:
        """The pixels (BGR) of `region` of the game's picture, or of the whole picture; None with `problem`
        set when the game isn't there to read."""
        area = self.play_area()
        if area is None:
            return None
        box = self._visible(region.box(area) if region else area)
        if box is None:
            self.problem = "off screen"
            return None
        if covered(self._hwnd, box):
            self.problem = "covered"
            return None
        image = self._shot(box)
        self.problem = None if image is not None else "off screen"
        return image

    def close(self) -> None:
        self._sct.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _window(self):
        if self._hwnd is None or not user32.IsWindow(self._hwnd):
            self._hwnd, self._bars = find_window(self.title), None
        return self._hwnd

    def _visible(self, box: Box, slack: int | None = None) -> Box | None:
        return visible_part(box, self._sct.monitors[1:], self.OFF_SCREEN_SLACK if slack is None else slack)

    def _shot(self, box: Box) -> np.ndarray | None:
        x, y, w, h = box
        try:
            shot = self._sct.grab({"left": x, "top": y, "width": w, "height": h})
        except self._screenshot_error:
            return None
        return np.ascontiguousarray(np.asarray(shot)[:, :, :3])  # BGRA -> BGR
