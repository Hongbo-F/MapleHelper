"""The game's own words on screen: the chat box's one line above the HUD (the message bar: `ChatReader`), where the game
says why it refused a key, and which of the pickup log's OCR rows are its lines (`log_rows`: not the party window's).

The message bar. Live on 2026-10-26 at 22:58:08-31 she hung at the top of the rope at x 111 on Tree Dungeon, Forest Up
North II (her feet about 30 below platform 18, which her dot snapped her onto), and every Lucky Seven press was refused:
the bar read "That cannot be used while you're on a rope or ladder." 0.19 s after the first (its key at 08.949, frame
822 at 09.142 of lab/recent/20261026-225844-out-of-darts), her MP stayed put, and after 20 presses out of darts
(behavior.darts) stopped the hunt with her stars in her bag, three times that night (22:58:32, 23:11:49, 23:12:42). The
bar is at game px x 290-830, y 703-727 at 1366x768 (CHAT, fractions of the window: the HUD scales with it; at 1920x1080
computed only, no full-size frame of it kept). It shows the last chat line from anyone, the game's own lines, the
server's notices, other players' megaphones and chat, and a line doesn't fade: it stays until the next replaces it (a
megaphone after 8.0 s at 22:58:17.163, the refusal back 0.15 s after the next press; 4 times in 23 s, up 0.6-8.0 s).
The game's own lines are pink: median RGB 244-248, 174-176, 185-188 on all 177 read on the saved frames (25,000+ frames
of tonight's 4 recent recordings, the 5 Lie Detector "before" sets, the 10-24/25 recordings; 676 lines read only when
the line changed: D:/MapleHelper-scratch/rope/diag/msgbar_scan.py), megaphones about (113, 6, 56), notices yellow, chat
and the rest white. So the bar's pink px (`game_text`) tell a game line in under a millisecond; only a new one is read
(maplehelper.ocr's shared RapidOCR, 285 ms wall / 281 ms CPU median over 11 reads at idle priority with the game
running, confidence 0.99-1.00 as it is: enlarged x2, 388 ms). The game's lines read: "That cannot be used while you're
on a rope or ladder." (28 reads; 10-25 00:28:07 too), "You don't have enough MP to use this skill." (36, 10-25 00:53),
"You are lacking the MP Potion that your pet is supposed to use." (95), "Your pet is running low on the allotted MP
potions." (14), "You cannot move to the requested channel at this time. ...", "You have played MapleStory for 2
hour(s).", "When a hit from an enemy, equipment swap, or use of certain skills, trigger...". None tells of stars or
darts: what the game says when she has none isn't known yet (perception.loot.says_no_darts).

A line redrawn is the same px as before: the same refusal twice in a row can't be told from one left up (`ChatLine.same`
says whether two reads are the one line, by their text px).

The pickup log's rows: its lines are drawn right-aligned against the window's right edge (their OCR rows ending at x
1359-1361 at 1366x768 on 6 saved frames), stacking up 14 px a line from y 606-616 (the newest) behind the party window,
whose names end at x 1266-1287 (BugBaby, RiceBaby: the leader's star read as "7", a snowflake as a CJK sign). At
22:58:23 the pickup log read for out of darts said "7BugBaby / RideBaby" (and the sign), no pickup in the last 4 s
(each line lasts about 3.8 s): rows not reaching LOG_RIGHT of the right edge are left out (`log_rows`), wherever the
party window is and however big (it moves and grows with the party)."""

import re
from dataclasses import dataclass

import numpy as np

# The message bar: game px x 290-830, y 703-727 of the 1366x768 window (the HUD's border at row 700), as fractions of
# the window (left, top, right, bottom)
CHAT = (290 / 1366, 703 / 768, 830 / 1366, 727 / 768)
# The game's own lines' pink (RGB 244-248, 174-176, 185-188 by their median): red over 200, and over green by 40, over
# blue by 30, blue not under green (the rope refusal's 673 px of text on frame 1065 by these, 697 by red-green over 40
# alone; megaphones' (113, 6, 56), notices' yellow and white chat none)
PINK_RED, PINK_OVER_GREEN, PINK_OVER_BLUE = 200, 40, 30
MIN_TEXT = 40       # px of pink in the bar for a game line (a line's text: 500-930)
SAME_SHARE = 0.9    # two reads whose pink px overlap this much (of their union) are the one line (`ChatLine.same`)
# The pickup log's rows: those ending within this share of the window's width of its right edge (10 game px at 1366x768:
# the log's rows end 5-7 px from it, the party window's names 79-100)
LOG_RIGHT = 10 / 1366

# What the game says when it refuses a key: on a rope or a ladder (no attack, no skill there), or for want of MP
ON_ROPE = re.compile(r"on\W*a\W*rope|rope\W*or\W*ladder|on\W*a\W*ladder", re.I)
REFUSED = re.compile(r"(?:can\W?not|can\W?t|unable\W*to)\W*be\W*used|(?:do\W*not|don\W?t)\W*have\W*enough\W*MP"
                     r"|not\W*enough\W*MP\W*to\W*use", re.I)


def refusal(text: str) -> str | None:
    """Why a line of the game's says it refused her key: 'rope' (she's on a rope or a ladder), 'refused' (anything
    else: "cannot be used", too little MP), or None (it doesn't). "You are lacking the MP Potion that your pet is
    supposed to use." is none (her pet's)."""
    if not text:
        return None
    if ON_ROPE.search(text):
        return "rope"
    return "refused" if REFUSED.search(text) else None


def game_text(bar: np.ndarray) -> np.ndarray:
    """The message bar's px (RGB) that are the game's own pink text."""
    px = bar[..., :3].astype(np.int16)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    return (r >= PINK_RED) & (r - g >= PINK_OVER_GREEN) & (r - b >= PINK_OVER_BLUE) & (b >= g - 5)


def chat_box(width: int, height: int) -> tuple[int, int, int, int]:
    """The message bar in a window of this size: x, y, w, h (px)."""
    left, top, right, bottom = CHAT
    x0, y0, x1, y1 = round(left * width), round(top * height), round(right * width), round(bottom * height)
    return x0, y0, x1 - x0, y1 - y0


@dataclass(eq=False)
class ChatLine:
    """What the message bar showed: the game's own line's text ("" if it showed none, or it couldn't be read), its pink
    px (`mask`; None: no game line), and when (`t`, the frame's time)."""
    text: str = ""
    mask: np.ndarray | None = None
    t: float = 0.0

    @property
    def refusal(self) -> str | None:
        """Why it says the game refused her key, if it does (`refusal`)."""
        return refusal(self.text)

    def same(self, other: "ChatLine | None") -> bool:
        """Whether `other` showed this very line (its pink px the same, SAME_SHARE of them; with no px, as the
        simulator's lines, its text): a line redrawn, or one left up, can't be told apart."""
        if other is None:
            return False
        if self.mask is None and other.mask is None:
            return bool(self.text) and self.text == other.text
        if self.mask is None or other.mask is None or self.mask.shape != other.mask.shape:
            return False
        union = np.count_nonzero(self.mask | other.mask)
        return union > 0 and np.count_nonzero(self.mask & other.mask) >= SAME_SHARE * union


class ChatReader:
    """The message bar off whole frames (`read`): its pink px every time, its text by OCR only when they show a game
    line not read before (the last read's text again for the same px). `ocr`: what reads a crop (BGR) to its rows'
    texts (by default maplehelper.ocr's shared engine; "" without the OCR packages)."""

    def __init__(self, ocr=None):
        self.ocr = ocr or _shared_ocr
        self.last: ChatLine | None = None  # the last game line read

    def read(self, frame: np.ndarray, t: float = 0.0) -> ChatLine:
        """The message bar of `frame` (the whole window, RGB), shown at `t`."""
        x, y, w, h = chat_box(frame.shape[1], frame.shape[0])
        bar = frame[y:y + h, x:x + w]
        mask = game_text(bar)
        if np.count_nonzero(mask) < MIN_TEXT:
            return ChatLine("", None, t)
        line = ChatLine("", mask, t)
        if line.same(self.last):
            line.text = self.last.text
        else:
            line.text = self.ocr(np.ascontiguousarray(bar[:, :, 2::-1]))  # RGB -> BGR
            self.last = line
        return line


def _shared_ocr(bgr: np.ndarray) -> str:
    """maplehelper.ocr's shared engine's read of a crop, its rows joined; "" without the OCR packages."""
    try:
        from maplehelper.ocr import shared
    except ImportError:
        return ""
    return " ".join(r.text for r in shared().read(bgr, min_height=0) or [])


def log_rows(rows, width: int, x0: int = 0) -> list:
    """Of the OCR rows (maplehelper.ocr.Row) read off a crop starting at window x `x0`, in a window `width` px wide,
    the pickup log's: those reaching within LOG_RIGHT of the window's right edge (the log's lines are right-aligned
    there; the party window's names stop short of its HP bars)."""
    edge = width - LOG_RIGHT * width
    return [r for r in rows if x0 + r.right >= edge]
