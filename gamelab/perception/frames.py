"""Frames of the game for the continuous feed (localization.feed, `python -m gamelab watch`): the game window
as the screen shows it, captured without ever bringing it to the front (`ScreenFrames`), or saved frames
played back (`ListFrames`).

A capture sees whatever is on top. So a frame the game isn't wholly showing (another window in front of it or
over part of it, the game minimized or not the client size the feed's eyes are for, partly off the screen, not
running) is marked not visible, and nothing is read from it: the feed waits for the game to come back rather than
bring it forward (nothing here ever does). Nor is a frame the window moved under while it was captured, or one the
capture held for STALE_S or more: their pictures aren't where the window is (Eyes' own checks once, now the frames':
perception.framecheck); nor one the game drew off its client area (framecheck.drawn_problem, with the window's own
frame known here: 2026-10-25 23:40), nor a capture that failed outright (kind "capture": a full-screen switch can make
the screen's copy fail for a moment; the frame loop goes on, where it ended the service). Each says why in words and as
a framecheck.Problem (`problem`), which the feed's state carries.
"""

import time
from dataclasses import dataclass

import numpy as np

from .. import capture, telemetry, window
from ..screen import STRETCHED, Geometry
from . import framecheck
from .eyes import rgb_of
from .framecheck import Problem


@dataclass
class Frame:
    """One moment of the game window."""
    full: np.ndarray | None  # full size, RGB (None: not visible)
    t: float                 # when the screen showed it (time.monotonic's clock), at the latest known
    visible: bool            # the game wholly on screen, in front, at the size the templates were made for
    new: bool = True         # False: the same picture as the frame before (nothing on screen changed)
    why: str = ""            # why it isn't visible ...
    problem: Problem | None = None  # ... as a framecheck.Problem (its kind decides what's done about it)

    @staticmethod
    def unfit(t: float, problem: Problem) -> "Frame":
        """A frame not to be read, for `problem`."""
        return Frame(None, t, False, why=problem.why, problem=problem)


class ScreenFrames:
    """The game window as the screen shows it, through the process's grabber (DXGI desktop duplication or
    mss, capture.shared). Reads the screen only: no keys, and the game is never brought to the front. `geo`: the
    client size the eyes reading the frames are for (screen.Geometry); at any other, not visible. `log`: where a move
    of the window is said."""

    def __init__(self, title: str, grabber: capture.Grabber | None = None, geo: Geometry = STRETCHED, log=print):
        self.title = title
        self.geo = geo
        self.grabber = grabber or capture.shared()
        self.log = log
        self.area: dict[str, int] | None = None  # where the client area was at the last frame read
        self._last: float | None = None  # the last frame's present time

    def next(self, after: float) -> Frame:
        """The game as the screen shows it: a frame shown at `after` or later if one comes (the grabber waits
        for it as long as a look would), else the one held, the screen not having changed since. Not visible, saying
        why, while the window isn't fit to read (framecheck.window_problem: gone, minimized, another size, partly off
        the screen, not in front, covered), and for a frame the window moved under or the capture held too long
        (framecheck.moved_problem, stale_problem)."""
        now = time.monotonic()
        try:
            hwnd = window.find_game(self.title)
            area = window.client_rect(hwnd)
            wrong = framecheck.window_problem(hwnd, area, self.title, self.geo)
        except RuntimeError as e:
            return Frame.unfit(now, Problem("gone", f"the game isn't running ({e})"))
        if wrong is not None:
            return Frame.unfit(now, wrong)
        try:
            shot = self.grabber.shot(area, after=after)
            full = shot.crop(0, 0, area["width"], area["height"], rgb_of)
        except Exception as e:  # (mss's GetDIBits refused, a duplication that couldn't be made: not the loop's end)
            return Frame.unfit(time.monotonic(), Problem("capture", f"the screen couldn't be captured "
                                                                    f"({type(e).__name__}: {e})"))
        try:
            moved = framecheck.moved_problem(area, window.client_rect(hwnd))  # (a window dragged moves between the two)
        except RuntimeError as e:
            return Frame.unfit(time.monotonic(), Problem("gone", f"the game isn't running ({e})"))
        if bad := moved or framecheck.stale_problem(shot.t, time.monotonic()) or framecheck.drawn_problem(
                full, self.geo, lambda: window.inset(hwnd, area)):
            return Frame.unfit(time.monotonic(), bad)
        if self.area is not None and area != self.area:
            text = f"the game window moved: ({self.area['left']}, {self.area['top']}) -> ({area['left']}, {area['top']})"
            self.log(f"  {text}")
            telemetry.emit("event", kind="window", text=text)
        self.area = area
        shown = shot.t if shot.t is not None else time.monotonic()
        new = shown != self._last
        self._last = shown
        # A frame shown before `after` is the screen as it still was when the grabber gave up waiting.
        return Frame(full, shown if shown >= after else time.monotonic(), True, new)


def glimpse(title: str, grabber: capture.Grabber | None = None) -> np.ndarray | None:
    """The game's client as the screen shows it now, whatever its size (RGB), unchecked: 1366x768 and 1920x1080 as they
    are, any other size brought to 1920x1080 (shrunk by area, stretched with Lanczos): the size the Lie Detector's
    templates are of (control.window_size.lie_detector_seen), and the HUD's buttons are found at it once shrunk
    (framecheck.hud_seen). None with no window, the game minimized, or no shot. The perception service's answer to a
    `full` ask for any size while its frames aren't fit to read for a fault Alt+Enter mends (framecheck.FIX_KINDS): the
    size fix's look for the Lie Detector before an Alt+Enter, the pet's for the HUD. Never brings the game forward. The
    process's grabber (capture.shared), as ScreenFrames'. A 3840x2160 picture takes 12 ms to convert and shrink
    (offline), its copy off the graphics card besides; 6.2 MB pickled down the pipe."""
    import cv2

    from .lie_detector import NATIVE_FRAME, TEMPLATE_FRAME
    try:
        hwnd = window.find_game(title)
        if window.minimized(hwnd):
            return None
        area = window.client_rect(hwnd)
        full = (grabber or capture.shared()).shot(area).crop(0, 0, area["width"], area["height"], rgb_of)
    except Exception:  # (no window, no shot: a capture refused mid-switch)
        return None
    size = (full.shape[1], full.shape[0])
    if size in (NATIVE_FRAME, TEMPLATE_FRAME):
        return full
    shrink = size[0] >= TEMPLATE_FRAME[0] and size[1] >= TEMPLATE_FRAME[1]
    return cv2.resize(full, TEMPLATE_FRAME, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LANCZOS4)


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
