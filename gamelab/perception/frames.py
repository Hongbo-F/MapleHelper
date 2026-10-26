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

A frame the capture still holds from more than HELD_S before the look asked keeps its own time (until 2026-10-26 it was
stamped as now, the screen as it still was): the game draws many times a second, so it's a stall, and a look waiting
for a frame after its keys mustn't take that picture for one. Those held shorter, from HELD_COUNT_S, are stamped as now
still and counted (`holds`: the service's console line says them), to tell live how often they come. Each frame's
picture is one of a few kept from frame to frame (`Pictures`), never one somebody still holds.

With `cover_check` on (the perception service's, while its rolling recording runs: control.recording) a frame also
says whether another program's window was over any of the game (`Frame.over`: window.over, every window above it in
the z-order, asked just before the shot and again just after it, as `gamelab recorder` asks around its grabs): the
grid's points (window.shows, in framecheck.window_problem) miss a window between them (a notification, a small box) and
one clicks pass through. What perception reads is as before (such a frame is visible all the same); the recording
keeps none of it.
"""

import ctypes
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass

import cv2
import numpy as np

from .. import capture, telemetry, window
from ..screen import STRETCHED, Geometry
from . import framecheck
from .eyes import rgb_of
from .framecheck import Problem

# A frame the duplication holds from this long before the look asked (its `after`) is a held one: given with its own
# time, not now's, and said once (a `held` event) until frames come again. DXGI waits FRESH_WAIT_S (20 ms) for a new
# frame and the game draws one every 16.7 ms in front, so a frame this old is a stall: live on 2026-10-26 01:21, a hunt
# stood still on "the capture's newest frame is 2.1 s old" (framecheck.STALE_S), and before that a frame up to 2 s old
# was stamped as now and acted on (D:/MapleHelper-scratch/pconst/plan: 0.3% of the hunts' waits on perception). Short
# of STALE_S, where such a frame becomes a problem (the duplication made afresh).
HELD_S = 0.5
# ... and one held from this long before it was asked, short of HELD_S, is stamped as now still but counted (`holds`):
# three frames' time at 60 a second, where DXGI's 20 ms wait and a frame drawn every 16.7 ms would have given a fresh
# one. How often they come isn't known (the review of 2026-10-26: the `held` event says only those past HELD_S); if
# often, HELD_S comes down to them.
HELD_COUNT_S = 0.05
# Pictures a ScreenFrames keeps to convert frames into (`Pictures`): the frame loop's, its search's, the recording's
# queue (8) and the one before, and a few held by the clients' answers. Past it a picture is made new (counted).
PICTURES = 24


@dataclass
class Frame:
    """One moment of the game window."""
    full: np.ndarray | None  # full size, RGB (None: not visible)
    t: float                 # when the screen showed it (time.monotonic's clock), at the latest known
    visible: bool            # the game wholly on screen, in front, at the size the templates were made for
    new: bool = True         # False: the same picture as the frame before (nothing on screen changed)
    why: str = ""            # why it isn't visible ...
    problem: Problem | None = None  # ... as a framecheck.Problem (its kind decides what's done about it)
    # Another program's window over any of the game at its shot (ScreenFrames.cover_check: window.over, before and
    # after it): "" none, else that window in a few words; None not asked (the rolling recording keeps no such frame).
    over: str | None = None

    @staticmethod
    def unfit(t: float, problem: Problem) -> "Frame":
        """A frame not to be read, for `problem`."""
        return Frame(None, t, False, why=problem.why, problem=problem)


class Pictures:
    """RGB pictures kept to convert the screen's frames into (`convert`, a Shot's convert: cv2.cvtColor into one), so
    that a frame isn't fresh memory: a new 1366x768 picture a frame was 3 MB, 770 page faults that Windows' kernel
    zeroes (its System process: about a core for 100,000 a second, D:/MapleHelper-scratch/cpu/review-numbers/
    zerotest.py), 23,000 a second at 30 frames. A picture is converted into again only once nothing else holds it (its
    reference count is the list's alone, as measured on one kept so: `_refs`), so a frame handed out (to the step, a
    search, the recording, a client's answer) is never written over while it's held; with all `keep` held a new one is
    made and not kept (`unkept`). Pictures of another size go when one of a new size is asked for."""

    def __init__(self, keep: int = PICTURES):
        self.keep = keep
        self._kept: list[np.ndarray] = [np.empty((1, 1, 3), np.uint8)]  # (no other reference to it: the free count)
        self._free_refs = self._refs(0)
        self._kept.clear()
        self.made = self.reused = self.unkept = 0

    def _refs(self, i: int) -> int:
        """Picture i's reference count, as read here (the same expression for the free count: what the interpreter
        adds for the call itself is the same, whatever the Python)."""
        return sys.getrefcount(self._kept[i])

    def convert(self, bgra: np.ndarray) -> np.ndarray:
        shape = (bgra.shape[0], bgra.shape[1], 3)
        if self._kept and self._kept[0].shape != shape:
            self._kept = [p for p in self._kept if p.shape == shape]
        for i in range(len(self._kept)):
            if self._refs(i) <= self._free_refs:
                self.reused += 1
                out = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB, dst=self._kept[i])
                return out
        picture = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        if len(self._kept) < self.keep:
            self._kept.append(picture)
            self.made += 1
        else:
            self.unkept += 1
        return picture


class ScreenFrames:
    """The game window as the screen shows it, through the process's grabber (DXGI desktop duplication or
    mss, capture.shared). Reads the screen only: no keys, and the game is never brought to the front. `geo`: the
    client size the eyes reading the frames are for (screen.Geometry); at any other, not visible. `log`: where a move
    of the window is said, and a held frame. Each frame's picture is one of `pictures`' (Pictures)."""

    def __init__(self, title: str, grabber: capture.Grabber | None = None, geo: Geometry = STRETCHED, log=print):
        self.title = title
        self.geo = geo
        self.grabber = grabber or capture.shared()
        self.log = log
        self.area: dict[str, int] | None = None  # where the client area was at the last frame read
        self._last: float | None = None  # the last frame's present time
        self.pictures = Pictures()
        self.held = 0          # frames given with their own time, held by the capture from HELD_S before they were asked
        self._holding = False  # (one said, until a fresh frame comes)
        self._holds = [0, 0.0]  # frames stamped as now though held from HELD_COUNT_S before, the longest (`holds`)
        self.cover_check = False  # window.over asked before and after each shot (Frame.over): the service's recording

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
        over = covering(hwnd, area) if self.cover_check else None  # (before the shot ...)
        try:
            shot = self.grabber.shot(area, after=after)
            full = shot.crop(0, 0, area["width"], area["height"], self.pictures.convert)
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
        if over == "":
            over = covering(hwnd, area)  # (... and after it: a window that came meanwhile)
        if self.area is not None and area != self.area:
            text = f"the game window moved: ({self.area['left']}, {self.area['top']}) -> ({area['left']}, {area['top']})"
            self.log(f"  {text}")
            telemetry.emit("event", kind="window", text=text)
        self.area = area
        shown = shot.t if shot.t is not None else time.monotonic()
        new = shown != self._last
        self._last = shown
        # A frame shown before `after` is the screen as it still was when the grabber gave up waiting; one shown more
        # than HELD_S before it, a stall: its own time (HELD_S).
        if after - shown <= HELD_S:
            self._holding = False
            if after - shown >= HELD_COUNT_S:
                self._holds[0] += 1
                self._holds[1] = max(self._holds[1], after - shown)
            return Frame(full, shown if shown >= after else time.monotonic(), True, new, over=over)
        self.held += 1
        if not self._holding:
            self._holding = True
            text = (f"the capture holds a frame shown {after - shown:.2f} s before it was asked for: given with its own "
                    f"time until a new one comes")
            self.log(f"  {text}")
            telemetry.emit("event", kind="held", text=text)
        return Frame(full, shown, True, new, over=over)

    def holds(self) -> tuple[int, float]:
        """Frames stamped as now though the capture held them from HELD_COUNT_S (short of HELD_S) before they were
        asked for, since the last call, and the longest of those holds (s)."""
        n, longest = self._holds
        self._holds = [0, 0.0]
        return n, longest


# window.over's walk up the z-order, here with Python's lock kept through the calls made for every window above the game
# (GetWindow, IsWindowVisible; IsIconic and GetWindowThreadProcessId for the shown ones): user32 reads them from the
# desktop's own memory, sending no message, so none can wait on another program. Through window.over (ctypes letting
# the lock go at each call, 2 a window, most of them hidden) the two walks of a frame took 1.75 ms of the perception
# service's frame interval, its search threads taking the lock at each call; with it kept, 1.2 ms, most of it the walk
# itself (277 windows above: the lowest shown of this PC's 278, the worst; D:/MapleHelper-scratch/pconst/finish/replay,
# b_nf* and a_fin* against the branch as reviewed: the interval less its measured parts, medians 1.9 and 1.3 ms against
# 0.1). Alone, 0.47 ms a walk against window.over's 0.60; with 3 threads of Python churning beside it, 0.35 s against
# 9.7 (over_same.py, the same answer on all 84 windows and boxes tried). Live, the game in front has only the topmost
# windows above it: 44 above the window in front on this PC, 0.077 ms a walk (above_front.py). The compositor's
# (window._cloaked, window._drawn) only for a shown window of another program, as window.over asks them, letting it go.
_held = ctypes.PyDLL("user32") if sys.platform == "win32" else None
if _held is not None:
    _held.GetWindow.argtypes = (wintypes.HWND, wintypes.UINT)
    _held.GetWindow.restype = wintypes.HWND
    for _name in ("IsWindowVisible", "IsIconic"):
        getattr(_held, _name).argtypes = (wintypes.HWND,)
        getattr(_held, _name).restype = wintypes.BOOL
    _held.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.c_void_p)
    _held.GetWindowThreadProcessId.restype = wintypes.DWORD


def _over(hwnd: int, area: dict) -> int | None:
    """window.over, its walk with Python's lock kept (above): another program's window shown over any of `area`."""
    if _held is None:
        return window.over(hwnd, area)
    x0, y0 = area["left"], area["top"]
    x1, y1 = x0 + area["width"], y0 + area["height"]
    pid = wintypes.DWORD(0)
    _held.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    game = pid.value
    above = _held.GetWindow(hwnd, window.GW_HWNDPREV)
    for _ in range(window.OVER_MAX):
        if not above:
            return None
        if _held.IsWindowVisible(above) and not _held.IsIconic(above):
            _held.GetWindowThreadProcessId(above, ctypes.byref(pid))
            if pid.value != game and not window._cloaked(above):
                left, top, right, bottom = window._drawn(above)
                if left < x1 and right > x0 and top < y1 and bottom > y0 and right > left and bottom > top:
                    return above
        above = _held.GetWindow(above, window.GW_HWNDPREV)
    return None


def covering(hwnd: int, area: dict) -> str:
    """Another program's window over any of the game's client area `area`, in a few words (window.over's test, `_over`;
    named by window.describe: its class and box, never its title), or "" if none. 0.077 ms with the 44 windows above
    the window in front on this PC (2026-10-26, the median of 500: D:/MapleHelper-scratch/pconst/finish/above_front.py),
    0.47 with all 277 (over_same.py)."""
    try:
        above = _over(hwnd, area)
    except Exception as e:  # (the window gone meanwhile: nothing of the frame is kept either)
        return f"not known ({type(e).__name__})"
    if above is None:
        return ""
    try:
        return window.describe(above)
    except Exception:  # (gone meanwhile)
        return "another window"


def glimpse(title: str, grabber: capture.Grabber | None = None) -> np.ndarray | None:
    """The game's client as the screen shows it now, whatever its size (RGB), unchecked: 1366x768 and 1920x1080 as they
    are, any other size brought to 1920x1080 (shrunk by area, stretched with Lanczos): the size the Lie Detector's
    templates are of (control.window_size.lie_detector_seen), and the HUD's buttons are found at it once shrunk
    (framecheck.hud_seen). None with no window, the game minimized, or no shot. The perception service's answer to a
    `full` ask for any size while its frames aren't fit to read for a fault Alt+Enter mends (framecheck.FIX_KINDS): the
    size fix's look for the Lie Detector before an Alt+Enter, the pet's for the HUD. Never brings the game forward. The
    process's grabber (capture.shared), as ScreenFrames'. A 3840x2160 picture takes 12 ms to convert and shrink
    (offline), its copy off the graphics card besides; 6.2 MB pickled down the pipe."""
    from .lie_detector import to_template
    try:
        hwnd = window.find_game(title)
        if window.minimized(hwnd):
            return None
        area = window.client_rect(hwnd)
        full = (grabber or capture.shared()).shot(area).crop(0, 0, area["width"], area["height"], rgb_of)
    except Exception:  # (no window, no shot: a capture refused mid-switch)
        return None
    return to_template(full)


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
        return Frame(full, time.monotonic(), True, over="")  # (a saved frame: no window over it)
