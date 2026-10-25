"""Whether a capture is of the game as the eyes expect it: the game window where it was read to be, in front, the
client size its templates were made for, and the frame lined up with its client area. A look that fails is never
acted on (localization.senses: it's taken again, and after a while given up, LookLost); what's wrong is said in words
(`Problem`), and Eyes.recover does what it can about it.

Why (2026-10-24): at 16:46:50 a hunt read her place off the map from its first look (x 950 on a map 360..410 wide)
and sent a plan off a rope every 5 s for two minutes, as nothing asked whether the look made sense (the map's minimap
picture wasn't in use in that process; README, Capture checks). At 16:48:45 the user was moving the game window: two
saved frames were offset from the client area by +26,+26 (the window's own title bar in the frame's top rows) and
-5,+6. At 16:55:33 a look captured another program's window over the game, which a look then never asked about.

Two anchors, which every frame of the game at its own 1366x768 shows at the same pixels whatever the map: the minimap
window's title strip ("MINI MAP") at the top left, and the HUD's border (row 700 dark, row 701 white, from x 283 to
1080; screen.HUD_TOP_GAME). Measured on the 82 distinct saved 1366x768 frames (both maps of 2026-10-23 and 24, the
hunts' hp0_* shots among them; D:/MapleHelper-scratch/capture/build/measure.py) and on shifted copies of a good one
(hp0_135315): see the constants. At 1920x1080 (the stretched window, as until 2026-10-23, and the simulator's frames)
they aren't measured, and nothing is checked there.

Imports: numpy, OpenCV, and the shared layer (screen, window, paths).
"""

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .. import paths, window
from ..screen import GAME_VIEW, HUD_TOP_GAME, Geometry

# The minimap window's title strip, at the game's own size: x0, y0, x1, y1 (the minimap window at the top left, as
# MINIMAP_BOX has it). Its picture is assets/minimap_title.png, cut from hp0_135315.png (Tree Dungeon, 2026-10-24
# 13:53); on West Rocky Mountain II's frames it's the same to the pixel.
TITLE_BOX = (0, 2, 80, 20)
# The mean difference (0-255, over its pixels and channels) a frame's strip may have from that picture. On the 79
# good frames 0.0-1.1; shifted by 1 px 24-33 (across 1 and 2: 27 and 22, down 1: 33), 40-82 by 2-26 px; the two
# offset frames of 16:48, 43 and 73; another window over the game, 191.
TITLE_DIFF = 10.0
# The HUD's border, every HUD_STEP-th px of x HUD_X: row HUD_TOP_GAME darker than HUD_DARK in every channel, the row
# under it brighter than HUD_WHITE. On the good frames all of them but on one, where the game's menu (opened by
# hand) covers part of the border: 0.883. Shifted down or up by a px or more: none (0.000); across by 1-10 px it
# stays (0.99-1.00: the title strip tells those).
HUD_X = (290, 1075)
HUD_STEP = 4
HUD_DARK = 40
HUD_WHITE = 200
HUD_SHARE = 0.5   # ... the least share of those that must be so
# The border's ends, in the HUD's own frame: row 701 is white from x HUD_ENDS[0] to HUD_ENDS[1] and dark (8) just
# outside them, on every good frame of both maps (whatever the background beside the HUD). A frame shifted across by
# a px or more has them wrong; something over an end of the border (the game's menu) leaves them alone or not, so an
# end wrong by itself fails nothing: it tells a frame whose title strip is off from a minimap moved by hand.
HUD_ENDS = (285, 1080)
# Searched for the border on a frame that fails (to say how far it's off): this many px either side of where it
# should be, and at least HUD_FOUND px of a row with it. The frames of 16:48 come out +26,+26 and -5,+6, as the
# minimap window and the quest box matched against hp0_135315 had them; every shift of the good frame tried (up to
# 40 px each way) comes out as it was made.
OFFSET_REACH = 48
HUD_FOUND = 600
# A frame shown this long before the look asked for it is stale: the game draws a new frame many times a second when
# it's in front (its monsters, its background), so the duplication has stopped handing over frames. DXGI otherwise
# waits 20 ms at most for one (capture.FRESH_WAIT_S); not seen in game.
STALE_S = 2.0

# A window over the game is looked for at a grid of points over its client area (window.shows), COVER_GRID[0] spaces
# across and COVER_GRID[1] down from one corner (8 px in) to the other: 7 x 5 points, 225 x 188 px apart at 1366x768,
# so a window that size anywhere over it is seen (the five points of before, its middle and corners, missed a box over
# x 300-660, y 60-360). 35 calls of WindowFromPoint: 0.23 ms (2026-10-24, the PC busy with simulator runs), a
# look 20-50 ms, a glance's 63 ms apart.
COVER_GRID = (6, 4)

# What a Problem is about, which decides what Eyes.recover does about it:
#   size     the client isn't the size the eyes are for: waited out, never resized here
#   minimized  the game minimized (a look restores it, as it always did, unless it's waiting for the user already)
#   gone     no game window
#   offscreen  part of the client area isn't on any monitor (the desktop resized?): waited out, never moved here
#   window   the game not in front, or covered by another window
#   moved    the game window moved while the look was taken
#   offset   the frame doesn't line up with the client area (both anchors off)
#   minimap  the minimap window's title strip isn't where it should be (closed, moved or covered)
#   hud      the HUD's border isn't where it should be (covered, or something over it)
#   stale    the capture's newest frame is old
#   picture  the map's minimap picture isn't found in the minimap window (another map?)
#   tag      her name tag's full-size template couldn't be cut (she isn't on screen)
#   place    her minimap dot puts her off the map (localization.placecheck)
CAPTURE_KINDS = ("offset", "stale")  # the capture itself may be at fault: the duplication is reopened
# The user sets these right (or the game is restarted): until they are, a look asks the window alone, and neither
# brings the game to the front nor captures it (Eyes.window_wait), so a hunt standing still never pulls the game over
# what the user is doing meanwhile.
WAIT_KINDS = ("size", "minimized", "gone", "offscreen")


@dataclass(frozen=True)
class Problem:
    """What's wrong with a look: `kind` (above) and `why`, in words for the log."""
    kind: str
    why: str

    def __str__(self) -> str:
        return self.why


class BadLook(RuntimeError):
    """A look that isn't of the game as the eyes expect it (Eyes.look): not to be acted on."""

    def __init__(self, problem: Problem):
        super().__init__(problem.why)
        self.problem = problem

    @property
    def why(self) -> str:
        return self.problem.why

    @property
    def kind(self) -> str:
        return self.problem.kind


class LookLost(BadLook):
    """Looks kept failing their checks for longer than localization.senses.LOST_S: nothing may act until one passes
    (a hunt stands still and looks again: Senses.regain); the command stops with this if nothing waits."""


def checked(geo: Geometry) -> bool:
    """Whether frames of this geometry are checked: the anchors are measured at the game's own 1366x768 only."""
    return geo.client == GAME_VIEW


@lru_cache(maxsize=1)
def title_picture() -> np.ndarray:
    """The minimap window's title strip as it should be (RGB, int16)."""
    bgr = cv2.imread(str(paths.ASSETS / "minimap_title.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(paths.ASSETS / "minimap_title.png")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.int16)


def title_box() -> tuple[int, int, int, int]:
    """TITLE_BOX as a shot's box: x, y, width, height."""
    x0, y0, x1, y1 = TITLE_BOX
    return x0, y0, x1 - x0, y1 - y0


def title_diff(strip: np.ndarray) -> float:
    """How far a capture of TITLE_BOX (RGB) is from the title strip's picture (mean difference, 0-255)."""
    want = title_picture()
    if strip.shape[:2] != want.shape[:2]:
        return 255.0
    return float(np.abs(strip[..., :3].astype(np.int16) - want).mean())


def hud_share(full: np.ndarray) -> float:
    """The share of the HUD border's sampled px that are as the border is (a full-size 1366x768 frame, RGB)."""
    top = full[HUD_TOP_GAME, HUD_X[0]:HUD_X[1]:HUD_STEP].astype(np.int16)
    under = full[HUD_TOP_GAME + 1, HUD_X[0]:HUD_X[1]:HUD_STEP].astype(np.int16)
    return float(((top.max(axis=1) < HUD_DARK) & (under.min(axis=1) > HUD_WHITE)).mean())


def ends_right(full: np.ndarray) -> bool:
    """Whether the HUD border's ends are where they should be (HUD_ENDS)."""
    row = full[HUD_TOP_GAME + 1]
    (a, b) = HUD_ENDS
    return bool(row[a - 1].max() < HUD_DARK and row[a].min() > HUD_WHITE
                and row[b].min() > HUD_WHITE and row[b + 1].max() < HUD_DARK)


def find_offset(full: np.ndarray) -> tuple[int, int] | None:
    """How far a frame is off its client area (dx, dy: the game's picture moved right and down), by where the HUD's
    border is in it: None if it isn't within OFFSET_REACH of where it should be (not the game, or covered). Across, by
    whichever of its ends is nearer where it should be: something over one end (a game window over x 1080) only
    shortens the border found, and a frame really off has both ends off alike. About 4 ms; only asked of a frame that
    failed."""
    h, w = full.shape[:2]
    y0, y1 = max(0, HUD_TOP_GAME - OFFSET_REACH), min(h - 1, HUD_TOP_GAME + OFFSET_REACH + 1)
    x0, x1 = max(0, HUD_ENDS[0] - OFFSET_REACH), min(w, HUD_ENDS[1] + OFFSET_REACH + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    rows = full[y0:y1 + 1, x0:x1]
    pair = (rows[:-1].max(axis=2) < HUD_DARK) & (rows[1:].min(axis=2) > HUD_WHITE)  # dark here, white under
    counts = pair.sum(axis=1)
    best = int(np.argmax(counts))
    if counts[best] < HUD_FOUND:
        return None
    xs = np.flatnonzero(pair[best])
    left, right = x0 + int(xs[0]) - HUD_ENDS[0], x0 + int(xs[-1]) - HUD_ENDS[1]
    return left if abs(left) <= abs(right) else right, y0 + best - HUD_TOP_GAME


def frame_problem(full: np.ndarray, geo: Geometry) -> Problem | None:
    """What's wrong with a full-size frame of the client area (RGB), by the two anchors; None if nothing is (or it
    isn't checked at this size). 0.04 ms on a good frame."""
    if not checked(geo) or (full.shape[1], full.shape[0]) != geo.client:
        return None
    x0, y0, x1, y1 = TITLE_BOX
    minimap_off = title_diff(full[y0:y1, x0:x1]) > TITLE_DIFF
    hud_off = hud_share(full) < HUD_SHARE
    if minimap_off and (hud_off or not ends_right(full)):
        off = find_offset(full)
        if off is None:
            return Problem("offset", "the capture doesn't show the game where its window is: neither the minimap "
                                     "window nor the HUD is in its place (another window, or not the game)")
        if off != (0, 0):
            return Problem("offset", f"the capture is offset by ({off[0]:+d}, {off[1]:+d}) px from the game's client "
                                     f"area (the window moving while it was captured?)")
        hud_off = False  # the border is where it should be, an end of it covered (a game window over x 1080)
    if minimap_off:
        return Problem("minimap", "the minimap window's title isn't at the top left (the minimap closed, moved or "
                                  "covered)")
    if hud_off:
        return Problem("hud", f"the HUD's border isn't on row {HUD_TOP_GAME} (something over it, or the chat box "
                              f"resized)")
    return None


def window_problem(hwnd: int, area: dict, title: str, geo: Geometry) -> Problem | None:
    """What's wrong with the game window for a capture: minimized, not the client size the eyes are for, partly off the
    screen, not in front, or with another window over it (window.shows, at COVER_GRID's points); None if nothing is.
    Those the user sets right (WAIT_KINDS) are asked first. About 0.25 ms (COVER_GRID's points the most of it)."""
    if window.minimized(hwnd):
        return Problem("minimized", "the game is minimized: waiting for it to be restored")
    size = (area["width"], area["height"])
    if size != geo.client:
        return size_problem(size, geo)
    if not window.on_screen(area):
        return Problem("offscreen", f"the game's client at ({area['left']}, {area['top']}) is partly off the screen "
                                    f"(the desktop resized?): waiting for the user to move it back (it isn't moved "
                                    f"here)")
    if window.foreground_title() != title:
        return Problem("window", "the game isn't in front")
    if not window.shows(hwnd, area, grid=COVER_GRID):
        return Problem("window", "another window is over the game")
    return None


def size_problem(size: tuple[int, int], geo: Geometry) -> Problem:
    """The client's size isn't the eyes': said so, and waited out (the user sets it back; it's never resized here)."""
    return Problem("size", f"the game's client is {size[0]}x{size[1]}, not {geo.client[0]}x{geo.client[1]}: "
                           f"waiting for the user to set it back (it isn't resized here)")


def moved_problem(before: dict, after: dict) -> Problem | None:
    """The game window moved (or changed size) between reading where it is and capturing it."""
    if before == after:
        return None
    return Problem("moved", f"the game window moved while it was captured: ({before['left']}, {before['top']}) -> "
                            f"({after['left']}, {after['top']})" + (
                                "" if (before["width"], before["height"]) == (after["width"], after["height"])
                                else f", {before['width']}x{before['height']} -> {after['width']}x{after['height']}"))


def stale_problem(shown: float | None, now: float) -> Problem | None:
    """The frame captured was shown STALE_S or more before `now`."""
    if shown is None or now - shown < STALE_S:
        return None
    return Problem("stale", f"the capture's newest frame is {now - shown:.1f} s old (the duplication stopped handing "
                            f"over frames?)")
