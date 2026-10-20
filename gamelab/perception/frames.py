"""Frames of the game for the continuous feed (localization.feed, `python -m gamelab watch`): the game window
as the screen shows it, captured without ever bringing it to the front (`ScreenFrames`), or saved frames
played back (`ListFrames`).

A capture sees whatever is on top. So a frame the game isn't wholly showing (another window in front of it or
over part of it, the game minimized or not 1920x1080, not running) is marked not visible, and nothing is read
from it: the feed waits for the game to come back rather than bring it forward (Eyes._capture does that for a
command that sends keys; nothing here ever does).
"""

import time
from dataclasses import dataclass

import numpy as np

from .. import capture, window
from ..screen import GAME_SIZE
from .eyes import rgb_of


@dataclass
class Frame:
    """One moment of the game window."""
    full: np.ndarray | None  # full size, RGB (None: not visible)
    t: float                 # when the screen showed it (time.monotonic's clock), at the latest known
    visible: bool            # the game wholly on screen, in front, at the size the templates were made at
    new: bool = True         # False: the same picture as the frame before (nothing on screen changed)
    why: str = ""            # why it isn't visible


class ScreenFrames:
    """The game window as the screen shows it, through the process's grabber (DXGI desktop duplication or
    mss, capture.shared). Reads the screen only: no keys, and the game is never brought to the front."""

    def __init__(self, title: str, grabber: capture.Grabber | None = None):
        self.title = title
        self.grabber = grabber or capture.shared()
        self._last: float | None = None  # the last frame's present time

    def next(self, after: float) -> Frame:
        """The game as the screen shows it: a frame shown at `after` or later if one comes (the grabber waits
        for it as long as a look would), else the one held, the screen not having changed since."""
        now = time.monotonic()
        try:
            hwnd = window.find_game(self.title)
        except RuntimeError:
            return Frame(None, now, False, why="the game isn't running")
        if window.minimized(hwnd):
            return Frame(None, now, False, why="the game is minimized")
        area = window.client_rect(hwnd)
        if (area["width"], area["height"]) != GAME_SIZE:
            return Frame(None, now, False, why=f"the game window is {area['width']}x{area['height']}, "
                                               f"not {GAME_SIZE[0]}x{GAME_SIZE[1]}")
        if window.foreground_title() != self.title or not window.shows(hwnd, area):
            return Frame(None, now, False, why="the game isn't in front")
        shot = self.grabber.shot(area, after=after)
        full = shot.crop(0, 0, area["width"], area["height"], rgb_of)
        shown = shot.t if shot.t is not None else time.monotonic()
        new = shown != self._last
        self._last = shown
        # A frame shown before `after` is the screen as it still was when the grabber gave up waiting.
        return Frame(full, shown if shown >= after else time.monotonic(), True, new)


class ListFrames:
    """Saved frames (full size, RGB) handed over in turn, round and round (`loop`), each stamped when it's
    handed over, as if the screen showed it then: for tests, and for soaking the feed offline."""

    def __init__(self, frames: list[np.ndarray], loop: bool = True):
        self.frames, self.loop, self.i = frames, loop, 0

    def next(self, after: float) -> Frame | None:
        """The next frame, shown at `after` at the soonest (None once the list is done, unless it loops)."""
        if self.i >= len(self.frames):
            if not self.loop or not self.frames:
                return None
            self.i = 0
        if (wait := after - time.monotonic()) > 0:
            time.sleep(wait)
        full = self.frames[self.i]
        self.i += 1
        return Frame(full, time.monotonic(), True)
