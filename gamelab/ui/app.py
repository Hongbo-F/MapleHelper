"""The mini-map window: follows a run's telemetry live, or replays a recorded run.

Live, it must never get in the game's way: the player only presses keys while the game is in front, and
screen captures see whatever window is on top. So the window is made hidden and shown without being
activated, clicks don't activate it (WS_EX_NOACTIVATE), focus is handed straight back should it ever
take it, and the command line places it beside the game, never over it. A replay has no game to protect: it can take focus, and keys control it (space pauses,
+ and - change the speed, Left and Right jump 10 s).
"""

import ctypes
import sys
import tkinter as tk
from collections.abc import Callable

import numpy as np

from .render import MapView
from .state import State

POLL_MS = 100
GWL_EXSTYLE, WS_EX_NOACTIVATE = -20, 0x08000000  # Windows: a window's extended style; "never activate"
SW_SHOWNOACTIVATE = 4
HWND_TOPMOST, SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = -1, 0x1, 0x2, 0x10


def ppm(rgb: np.ndarray) -> bytes:
    """An RGB picture as a binary PPM (what Tk's PhotoImage reads without extra packages)."""
    h, w = rgb.shape[:2]
    return b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(rgb, np.uint8).tobytes()


class MapWindow:
    """The window: polls `source` (sources.LiveSource or ReplaySource) and draws the state with the map
    view `view_for(map id)` gives (None: the map to start with)."""

    def __init__(self, view_for: Callable[[int | None], MapView], source, live: bool,
                 position: tuple[int, int] | None = None, title: str = "gamelab mini-map"):
        self.view_for, self.source = view_for, source
        self.state = State()
        self.view = view_for(None)
        self.root = tk.Tk()
        self.root.title(title)
        if position is not None:
            self.root.geometry(f"+{position[0]}+{position[1]}")
        before = _foreground()
        if live:
            self.root.withdraw()  # made hidden: showing it the usual way would activate it
        self.label = tk.Label(self.root, bd=0)
        self.label.pack()
        self._show()
        if live:
            self.root.update_idletasks()
            frame = int(self.root.wm_frame(), 16)
            _no_activate(frame)
            _show_inactive(frame, on_top=position is not None)
            self.root.update()
            if before and _foreground() != before:  # it shouldn't have, but if it took the focus: back
                _hand_focus_back(before)
        else:
            self.root.update()
            self.root.bind("<space>", lambda e: setattr(source, "paused", not source.paused))
            self.root.bind("<plus>", lambda e: setattr(source, "speed", source.speed * 2))
            self.root.bind("<minus>", lambda e: setattr(source, "speed", source.speed / 2))
            self.root.bind("<Right>", lambda e: self._seek(10))
            self.root.bind("<Left>", lambda e: self._seek(-10))

    def run(self) -> None:
        self.root.after(POLL_MS, self._tick)
        self.root.mainloop()

    def _tick(self) -> None:
        fresh, records = self.source.poll()
        if fresh:
            self.state = State()
        for r in records:
            self.state.apply(r)
        if self.state.map is not None and self.state.map != getattr(self.view.model, "id", None):
            self.view = self.view_for(self.state.map)
        if records or fresh:
            self._show()
        self.root.after(POLL_MS, self._tick)

    def _seek(self, seconds: float) -> None:
        self.state = State()
        for r in self.source.seek(seconds):
            self.state.apply(r)
        self._show()

    def _show(self) -> None:
        # Kept on self: Tk shows a blank picture once its PhotoImage is garbage collected.
        self._photo = tk.PhotoImage(data=ppm(self.view.draw(self.state)), format="PPM")
        self.label.configure(image=self._photo)


# --- Windows: never take the game's focus ---------------------------------------------------------

def _foreground() -> int | None:
    if sys.platform != "win32":
        return None
    return ctypes.windll.user32.GetForegroundWindow() or None


def _no_activate(hwnd: int) -> None:
    """Clicks on the window no longer activate it (the game keeps the focus)."""
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, user32.GetWindowLongW(hwnd, GWL_EXSTYLE) | WS_EX_NOACTIVATE)


def _show_inactive(hwnd: int, on_top: bool) -> None:
    """Show the window without activating it, and (`on_top`: it was placed beside the game, never over
    it) keep it above other windows, still never active."""
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
    if on_top:  # (handles are pointer-sized: HWND_TOPMOST is (HWND)-1)
        user32.SetWindowPos(ctypes.c_void_p(hwnd), ctypes.c_void_p(HWND_TOPMOST), 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)


def _hand_focus_back(hwnd: int) -> None:
    """Creating the window made it the foreground one: give the focus back to where it was (the
    foreground window's owner may hand it on)."""
    ctypes.windll.user32.SetForegroundWindow(hwnd)
