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
# A server notice's dark band over the top of the client halves the strip exactly: 0.46 from the picture at half
# brightness (2026-11-05 18:31:42, which went on 120 s with nothing sent, the strip reading 108 from the picture itself);
# the same frame shifted by 1-10 px reads 9.8-24.4, hence a limit of its own.
TITLE_DIMMED_DIFF = 5.0
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
# The game drawing its picture off its own client area (2026-10-25 23:40, Tree Dungeon): the client the right size,
# 1366x768 at (2454, 107) desktop px (DPI 300%; the window 2436,22..3838,893), the picture drawn at the window's
# corner, 18 px left of the client's and 85 up (the HUD's border on row 615, x 267-1062; the minimap's title strip
# above the client, cut off; under the HUD the desktop or black), on every look until Alt+Enter twice set it right.
# Neither anchor is in its place and the border is past OFFSET_REACH, so every look said "the capture doesn't show
# the game where its window is" and a hunt stood still for ever. The border is looked for whole (its sampled px and
# both its ends at a shift: `whole_at`) up to DRAWN_REACH px up and DRAWN_ACROSS across, or nearer at the window's
# own frame (DRAWN_SLACK px either way): such a frame is the game's drawing, not the capture's (kind "drawn":
# control.window_size mends it). On the saved good frames none is (README, Capture checks).
DRAWN_REACH = 160   # guess: the window's title bar and frame at 300% are 85 px; this reaches to about 500%
DRAWN_ACROSS = 64   # guess: the frame's side at 300% is 18 px
DRAWN_SLACK = 2     # guess: px a shift may be off the window's frame and still be it
# The HUD's three buttons (CASH SHOP, MENU, SHORT CUT: x0, y0, x1, y1 at the game's own size), cut from tree_135315.png
# into assets/hud_buttons.png: what says the game is in play on a frame whose 1-px border is resampled away (a client
# of another size brought back to 1366x768: the border's rows read 41-92 dark and 178-212 white, against 40 and 200).
# Matched (grey, TM_CCOEFF_NORMED; `hud_seen`) on the good frames at 1366x768 and as clients of 3840x2160, 2560x1440,
# 1600x900, 1396x1952, 1350x700 and 1280x720 show them (69 frames: both maps' and the Lie Detector screens', the
# notice over the buttons the lowest): 0.774 or more; on the drawn frame of 23:40 1.000 at (-18, -85); on the desktop
# and the game's play area moved over its HUD, 0.52 at most (D:/MapleHelper-scratch/winfix/build/hud_template.py).
BUTTONS_BOX = (858, 734, 1076, 766)
HUD_SEEN = 0.7
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
#   size     the client isn't the size the eyes are for: waited out, never resized here (the commands that act set
#            it right with Alt+Enter twice: control.window_size)
#   drawn    the client the right size, the game's picture drawn off it (drawn_offset): Alt+Enter twice, as `size`
#   minimized  the game minimized (a look restores it, as it always did, unless it's waiting for the user already)
#   gone     no game window
#   offscreen  part of the client area isn't on any monitor (the desktop resized?): waited out, never moved here
#   window   the game not in front, or covered by another window
#   moved    the game window moved while the look was taken
#   offset   the frame doesn't line up with the client area (both anchors off)
#   minimap  the minimap window's title strip isn't where it should be (closed, moved or covered)
#   hud      the HUD's border isn't where it should be (covered, or something over it)
#   stale    the capture's newest frame is old
#   capture  the screen couldn't be copied at all (perception.frames)
#   picture  the map's minimap picture isn't found in the minimap window (another map?)
#   tag      her name tag's full-size template couldn't be cut (she isn't on screen)
#   place    her minimap dot puts her off the map (localization.placecheck)
CAPTURE_KINDS = ("offset", "stale", "capture")  # the capture itself may be at fault: the duplication is reopened
# The user sets these right (or the game is restarted): until they are, a look asks the window alone, and neither
# brings the game to the front nor captures it (Eyes.window_wait), so a hunt standing still never pulls the game over
# what the user is doing meanwhile.
WAIT_KINDS = ("size", "minimized", "gone", "offscreen")
# What Alt+Enter twice (full screen and back, as the user does it) sets right: control.window_size, through the player,
# by a command that acts (Senses.window_fix) or, bounded, the pet's babysitter (behavior.pet.PetWait).
FIX_KINDS = ("size", "drawn")


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
    """How far a capture of TITLE_BOX (RGB) is from the title strip's picture (mean difference, 0-255), under a server
    notice's band from the picture dimmed by it."""
    want = title_picture()
    if strip.shape[:2] != want.shape[:2]:
        return 255.0
    got = strip[..., :3].astype(np.int16)
    plain = float(np.abs(got - want).mean())
    if plain <= TITLE_DIFF:
        return plain
    dimmed = float(np.abs(got - want / 2).mean())
    return dimmed if dimmed <= TITLE_DIMMED_DIFF else plain


def hud_share(full: np.ndarray, dx: int = 0, dy: int = 0) -> float:
    """The share of the HUD border's sampled px that are as the border is (a full-size 1366x768 frame, RGB), with the
    game's picture moved `dx` right and `dy` down (0 where the border would be off the frame)."""
    y, x0, x1 = HUD_TOP_GAME + dy, HUD_X[0] + dx, HUD_X[1] + dx
    if y < 0 or y + 1 >= full.shape[0] or x0 < 0 or x1 > full.shape[1]:
        return 0.0
    top = full[y, x0:x1:HUD_STEP].astype(np.int16)
    under = full[y + 1, x0:x1:HUD_STEP].astype(np.int16)
    return float(((top.max(axis=1) < HUD_DARK) & (under.min(axis=1) > HUD_WHITE)).mean())


def ends_right(full: np.ndarray, dx: int = 0, dy: int = 0) -> bool:
    """Whether the HUD border's ends are where they should be (HUD_ENDS), with the picture moved `dx`, `dy`."""
    y, a, b = HUD_TOP_GAME + 1 + dy, HUD_ENDS[0] + dx, HUD_ENDS[1] + dx
    if not (0 <= y < full.shape[0] and 1 <= a and b + 1 < full.shape[1]):
        return False
    row = full[y]
    return bool(row[a - 1].max() < HUD_DARK and row[a].min() > HUD_WHITE
                and row[b].min() > HUD_WHITE and row[b + 1].max() < HUD_DARK)


def whole_at(full: np.ndarray, dx: int, dy: int) -> bool:
    """Whether the HUD's border is whole with the game's picture moved `dx` right and `dy` down: HUD_SHARE of its
    sampled px as the border is, and both its ends where they'd be (a picture moved, not something dark over white)."""
    return hud_share(full, dx, dy) >= HUD_SHARE and ends_right(full, dx, dy)


def find_offset(full: np.ndarray, up: int = OFFSET_REACH, down: int = OFFSET_REACH, across: int = OFFSET_REACH,
                whole: bool = False) -> tuple[int, int] | None:
    """How far a frame is off its client area (dx, dy: the game's picture moved right and down), by where the HUD's
    border is in it: None if it isn't within `up` px above where it should be to `down` px below (negative: above
    too) and `across` px either side (by default OFFSET_REACH each way: not the game, or covered). Across, by
    whichever of its ends is nearer where it should be: something over one end (a game window over x 1080) only
    shortens the border found, and a frame really off has both ends off alike. With `whole`, the best row whose
    shift has the border whole there (`whole_at`), not merely the row with the most of it. About 4 ms (the near
    search; the drawn one's, DRAWN_REACH up, about 6 more); only asked of a frame that failed."""
    h, w = full.shape[:2]
    y0, y1 = max(0, HUD_TOP_GAME - up), min(h - 1, HUD_TOP_GAME + down + 1)
    x0, x1 = max(0, HUD_ENDS[0] - across), min(w, HUD_ENDS[1] + across + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    rows = full[y0:y1 + 1, x0:x1]
    pair = (rows[:-1].max(axis=2) < HUD_DARK) & (rows[1:].min(axis=2) > HUD_WHITE)  # dark here, white under
    counts = pair.sum(axis=1)
    for best in np.argsort(-counts, kind="stable")[:None if whole else 1]:
        if counts[best] < HUD_FOUND:
            return None
        xs = np.flatnonzero(pair[best])
        left, right = x0 + int(xs[0]) - HUD_ENDS[0], x0 + int(xs[-1]) - HUD_ENDS[1]
        off = left if abs(left) <= abs(right) else right, y0 + int(best) - HUD_TOP_GAME
        if not whole or whole_at(full, *off):
            return off
    return None


def anchors_off(full: np.ndarray) -> bool:
    """Whether neither anchor is in its place (the minimap's title strip off, and the HUD's border off or its ends
    wrong): a frame frame_problem looks for the game's picture in."""
    x0, y0, x1, y1 = TITLE_BOX
    return title_diff(full[y0:y1, x0:x1]) > TITLE_DIFF and (hud_share(full) < HUD_SHARE or not ends_right(full))


def drawn_offset(full: np.ndarray, inset: tuple[int, int] | None = None) -> tuple[int, int] | None:
    """How far the game drew its picture off its client area (dx, dy), if it did: its HUD's border whole (`whole_at`)
    further up than OFFSET_REACH (to DRAWN_REACH up, DRAWN_ACROSS across), or nearer at the window's own frame
    (`inset`: the client area's corner from the window's, desktop px; the picture drawn at the window's corner, as at
    23:40, is -inset, DRAWN_SLACK px either way). None if it didn't (or the frame is in its place). A capture lagging a
    window dragged is off within OFFSET_REACH, and at the frame's shift only by chance (and only while it's dragged)."""
    near = find_offset(full)
    if near is not None:
        at_frame = inset is not None and near != (0, 0) and all(abs(o + i) <= DRAWN_SLACK for o, i in zip(near, inset))
        return near if at_frame and abs(near[1]) >= DRAWN_SLACK and whole_at(full, *near) else None
    return _far(full)


def _far(full: np.ndarray) -> tuple[int, int] | None:
    """The HUD's border whole further up than OFFSET_REACH (DRAWN_REACH up at most, DRAWN_ACROSS across): about 5 ms."""
    return find_offset(full, up=DRAWN_REACH, down=-OFFSET_REACH - 1, across=DRAWN_ACROSS, whole=True)


def drawn_problem(full: np.ndarray, geo: Geometry, inset=None) -> Problem | None:
    """A frame the game drew off its client area (drawn_offset), said so; None if it isn't one (or isn't checked at
    this size). `inset`: the window's frame (as drawn_offset's), or what says it (called only for a frame that fails
    both anchors: the window read then alone)."""
    if not checked(geo) or (full.shape[1], full.shape[0]) != geo.client or not anchors_off(full):
        return None
    off = drawn_offset(full, inset() if callable(inset) else inset)
    return None if off is None else _drawn(off)


@lru_cache(maxsize=1)
def buttons_picture() -> np.ndarray:
    """The HUD's three buttons as they should be (grey, uint8: assets/hud_buttons.png, BUTTONS_BOX of tree_135315)."""
    bgr = cv2.imread(str(paths.ASSETS / "hud_buttons.png"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(paths.ASSETS / "hud_buttons.png")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def hud_seen(full: np.ndarray) -> tuple[float, tuple[int, int]]:
    """How well the HUD's buttons (CASH SHOP, MENU, SHORT CUT) match where they'd be in a frame of the client (RGB, any
    size: brought to 1366x768 first), DRAWN_REACH up to OFFSET_REACH down and DRAWN_ACROSS across of their place, and
    at what shift: whether the game is in play (its HUD on screen), on a frame the border's 1-px rows don't survive (a
    client of another size, its picture resampled twice). HUD_SEEN or more: seen. About 3 ms."""
    if (full.shape[1], full.shape[0]) != GAME_VIEW:
        full = cv2.resize(np.ascontiguousarray(full), GAME_VIEW, interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(np.ascontiguousarray(full[..., :3]), cv2.COLOR_RGB2GRAY)
    want = buttons_picture()
    bx, by = BUTTONS_BOX[:2]
    x0, y0 = max(0, bx - DRAWN_ACROSS), max(0, by - DRAWN_REACH)
    x1 = min(GAME_VIEW[0], BUTTONS_BOX[2] + DRAWN_ACROSS)
    y1 = min(GAME_VIEW[1], BUTTONS_BOX[3] + OFFSET_REACH)
    _, best, _, at = cv2.minMaxLoc(cv2.matchTemplate(grey[y0:y1, x0:x1], want, cv2.TM_CCOEFF_NORMED))
    return float(best), (at[0] + x0 - bx, at[1] + y0 - by)


def _drawn(off: tuple[int, int]) -> Problem:
    dx, dy = off
    return Problem("drawn", f"the game draws its picture ({dx:+d}, {dy:+d}) px off its client area (the HUD's border on "
                            f"row {HUD_TOP_GAME + dy}, not {HUD_TOP_GAME}; the client the right size): Alt+Enter twice, "
                            f"full screen and back, sets it right")


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
            if (drawn := _far(full)) is not None:  # (the window's frame isn't known here: the service's frames know
                return _drawn(drawn)                 # it, perception.frames, and catch one drawn nearer at it)
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
    """The client's size isn't the eyes': said so, and waited out (the user sets it back, or a command that acts with
    Alt+Enter twice: control.window_size; it's never resized here)."""
    return Problem("size", f"the game's client is {size[0]}x{size[1]}, not {geo.client[0]}x{geo.client[1]}: "
                           f"waiting for it to be set back (it isn't resized here)")


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
