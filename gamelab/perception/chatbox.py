"""The chat box at the bottom left, expanded or collapsed (`read`), off a whole frame of the 1366x768 client, by its
drawn parts alone (never its text): its +/- button and its bottom row.

Expanded (the user's 2026-10-28 "if the chat window is expended, you should press ' key to collapse it"), it shows four
lines of chat over the play area's bottom left (x 287-833, y 633-700, translucent, a scrollbar at x 835-848) and below
them its input row: the 'All' tab (x 287-366, navy (0x3_, 0x5_, 0x7_) with a dark border) and a white field to x 811,
then its button with a minus. Collapsed, the row shows the newest chat line on gray (0x77) with the button's plus and
two small arrows beside it. The button (x 818-829, y 709-720) is a 12 x 12 square of the game's UI: black border, blue
inside (b over r by 128-190), its glyph white: the minus two rows (716-717) x 821-826, the plus two rows (714-715) and
two columns (823-824, rows 712-717). The glyph says what clicking it does, so it tells the state whatever the box's
height, its lines' colours (a megaphone's teal or pink backing, a notice's dark one) or what's behind it or over its
upper part (the party invite at x 708-870, y 633-700, translucent over the scrollbar). On the 10-30 08:17:50 recording
of BugBaby (3,615 frames at 15 a second, D:/MapleHelper-scratch/loc2build/bugbaby/frames; the box expanded until the
user collapsed it 34.3 s in, frame 515) and the 10-30 recording of her played by hand (4,507 frames, collapsed
throughout: D:/MapleHelper-scratch/golden-clips/20261030-bugbaby-human-play), every expanded frame's button matched the
minus's 100 px of 100, its border 40 of 40, its blue 88 of 88, its field white on 100% of its px, its tab navy on 96%;
every collapsed frame's the plus's 100 of 100 but one (frame 592: the mouse pointer over the button, unknown), its field
white on 0.1% at most, its tab navy on 16% at most (D:/MapleHelper-scratch/chat: features.py; read, scan.py, scan.txt:
515 of 515 expanded, 7,606 of 7,607 collapsed). No false expanded on the 107 shots of lab/shots and every 20th frame of
the 21 Lie Detector recordings (JPEG, 3,077: 7 expanded, each so by eye; scan_more.py).

The game's notices (the system lines flashing at the middle right) and the effects of the play area aren't over the
button's place (the UI is drawn over the scene); what is (the mouse pointer, a window dragged there) leaves it unknown.

Its input line may have the keyboard (`typing`): the field showing text or a caret (TYPED_LEAST dark px or more in its
white; an expanded box's field was empty on every frame of the recording, her keys going to the game all the while: her
MP spent by her Lucky Seven), or white with the box collapsed (the input line open without the box: not seen, taken
so). A key pressed then would be typed into it.

At any other client size the box isn't read: unknown (the HUD is drawn scaled there; no frame of it kept). Cheap: the
button's 144 px, and the field's and the tab's 8,500 only where its glyph is one of the two (0.04-0.08 ms a frame)."""

from dataclasses import dataclass

import numpy as np

FRAME = (1366, 768)        # the client size it's read at (else unknown)
EXPANDED, COLLAPSED, UNKNOWN = "expanded", "collapsed", "unknown"
BUTTON = (818, 709)        # its +/- button's top-left (client px): 12 x 12, the border its outer px (corners left out)
SIDE = 12
# Its glyph's white px in the button's 10 x 10 inside (rows, columns from (819, 710))
MINUS = np.zeros((SIDE - 2, SIDE - 2), bool)
MINUS[6:8, 2:8] = True
PLUS = np.zeros((SIDE - 2, SIDE - 2), bool)
PLUS[4:6, 2:8] = True
PLUS[2:8, 4:6] = True
GLYPH_WHITE = 192          # a glyph px: every channel this or over (its 0xd_-0xf_)
GLYPH_OFF = 3              # px of the inside's 100 off the glyph's at most (0 on every frame of the recordings)
BORDER_DARK = 48           # a border px: no channel over this (0x00 on the recordings) ...
BORDER_LEAST = 36          # ... this many of its 40 at least
BLUE_SHARE = 0.9           # the inside's px not of the glyph: this share blue at least (b 128 or over, over r by 64)
FIELD = (372, 707, 808, 724)  # the input row's field (x0, y0, x1, y1): white expanded, the newest line's gray collapsed
FIELD_WHITE = 224          # a field px white: every channel this or over ...
FIELD_SHARE = 0.75         # ... over this share of it: the input row (1.0 expanded, 0.001 collapsed at most)
TYPED_DARK = 128           # a px of text or a caret in the white field: no channel at this or over ...
TYPED_LEAST = 4            # ... this many of them: its input line has text or a caret (0 on the recording)
TAB = (289, 708, 365, 722)  # the 'All' tab's fill (x0, y0, x1, y1) ...
TAB_SHARE = 0.6            # ... navy over this share (0.961 expanded, 0.164 collapsed at most)


@dataclass(frozen=True)
class ChatBox:
    """What a frame shows of the chat box: `state` EXPANDED, COLLAPSED or UNKNOWN (`why`); `typing`: its input line has
    text or a caret, or is open with the box collapsed (it may have the keyboard: a key would be typed into it)."""
    state: str
    typing: bool = False
    why: str = ""

    @property
    def told(self) -> bool:
        return self.state != UNKNOWN

    def words(self) -> str:
        if self.state == UNKNOWN:
            return f"not told ({self.why})" if self.why else "not told"
        return self.state + (", its input line with text or a caret" if self.typing else "")


def _low(px: np.ndarray) -> np.ndarray:
    """Each px's lowest channel (np.minimum: a reduction over the channels' axis takes 10 times as long)."""
    return np.minimum(np.minimum(px[..., 0], px[..., 1]), px[..., 2])


def _high(px: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(px[..., 0], px[..., 1]), px[..., 2])


def glyph(frame: np.ndarray) -> str | None:
    """Which glyph the button shows on a 1366x768 frame (RGB): "minus" (the box expanded), "plus" (collapsed), or None
    (neither: covered, not drawn, another screen)."""
    x, y = BUTTON
    b = frame[y:y + SIDE, x:x + SIDE, :3].astype(np.int16)
    if b.shape[:2] != (SIDE, SIDE):
        return None
    border = np.concatenate([b[1:-1, 0], b[1:-1, -1], b[0, 1:-1], b[-1, 1:-1]])
    if np.count_nonzero(_high(border) <= BORDER_DARK) < BORDER_LEAST:
        return None
    inside = b[1:-1, 1:-1]
    white = _low(inside) >= GLYPH_WHITE
    blue = (inside[..., 2] >= 128) & (inside[..., 2] - inside[..., 0] >= 64)
    for name, mask in (("minus", MINUS), ("plus", PLUS)):
        if np.count_nonzero(white != mask) <= GLYPH_OFF:
            rest = ~mask
            return name if np.count_nonzero(blue & rest) >= BLUE_SHARE * np.count_nonzero(rest) else None
    return None


def _field(frame: np.ndarray) -> tuple[float, int]:
    """The input row's field: the share of its px white, and its dark px (text, a caret) where it's white."""
    x0, y0, x1, y1 = FIELD
    f = frame[y0:y1, x0:x1]
    white = np.count_nonzero(_low(f) >= FIELD_WHITE) / ((y1 - y0) * (x1 - x0))
    return white, np.count_nonzero(_high(f) < TYPED_DARK) if white >= FIELD_SHARE else 0


def _navy(frame: np.ndarray) -> float:
    """The share of the 'All' tab's fill that's its navy."""
    x0, y0, x1, y1 = TAB
    t = frame[y0:y1, x0:x1].astype(np.int16)
    r, b = t[..., 0], t[..., 2]
    return np.count_nonzero((b - r >= 40) & (b >= 90) & (b <= 150) & (_high(t) <= 160)) / ((y1 - y0) * (x1 - x0))


def read(frame: np.ndarray | None) -> ChatBox:
    """The chat box on a whole frame of the client (RGB, full size; see the module's doc)."""
    if frame is None:
        return ChatBox(UNKNOWN, why="no frame")
    h, w = frame.shape[:2]
    if (w, h) != FRAME:
        return ChatBox(UNKNOWN, why=f"the client is {w}x{h}, not {FRAME[0]}x{FRAME[1]}")
    g = glyph(frame)
    if g is None:
        return ChatBox(UNKNOWN, why="its button not seen")
    white, typed = _field(frame)
    if g == "plus":
        return ChatBox(COLLAPSED, typing=bool(white >= FIELD_SHARE))
    if white < FIELD_SHARE or _navy(frame) < TAB_SHARE:
        return ChatBox(UNKNOWN, why=f"its button's minus, its input row not seen ({white:.0%} of its field white)")
    return ChatBox(EXPANDED, typing=bool(typed >= TYPED_LEAST))
