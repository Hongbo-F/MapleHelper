"""Her EXP number in the bottom bar, read: "13,549[67.02%]", the EXP she has into her level and its share of what
the level needs, glyph by glyph as the game draws it. A wrong value must never come out; no value is fine.

How the game draws it (on calibrate.png, drawn so, every pixel of the box read came out within 2 levels of the
screen, 0.3 on average): at its own 1366x768, with the status bar's number pictures (gamedata.canvas.status_numbers:
the digits, ',', '.' and '%' white, '[' and ']' green, 5x7 game px for a digit), each glyph its picture's width plus
one blank column after the one before, the first with its top left at game px (749, 740): the bar's left (279) plus
StatusBar.img's posNumberExp (470). The screen shows the whole picture stretched to 1920x1080 with bilinear
filtering, game px (gx, gy)'s middle at screen ((gx + 0.5) * SX - 0.5, (gy + 0.5) * SY - 0.5): SX is 1920/1366, not
the 1080/768 = 1.40625 down (with that across, the number came out 11 levels off on average, half a pixel to the
right; at the minimap, on the screen's left, the two differ by 0.1 px at most).

How it's read (`read`): over the box READ_BOX, every glyph the number can hold is drawn as the game would draw it
at every game x from the first one on, and so is a blank column. As glyphs never share a column (each has a blank
one after it), the screen's columns split into cells, one per glyph, each depending on that glyph alone (a screen
column goes to the cell of the game column its sample falls before), so a reading is a path through the box, cell
after cell, and the best path (the least squared difference from the screen) is found exactly (dynamic
programming). Each cell of it must then be clearly what it says:
- its pixels within FIT of the glyph as the game draws it (root mean square over the cell, in levels);
- every other reading of that cell (the best path through any other glyph or blank covering its first column)
  worse by LEAD at least (as a root mean square over the cell);
- the glyphs must spell "<EXP, commas every 3 digits>[<percent with 2 decimals>%]" and nothing but blank cells
  after it (a digit-like cell right after the number means one was missed: no reading);
- the percent must be what the EXP is of some level's requirement (maplehelper.ocr.exp_table: Classic World's
  table, meowdb.com's; the game rounds to the nearest hundredth: 2,613 at level 19 shows 15.67%, 15.6655).
The background is the bar's own, dark and the same along each row: taken per row from the box's darkest columns.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from maplehelper.ocr.exp_table import EXP_TO_NEXT

SX, SY = 1920 / 1366, 1080 / 768  # screen px per game px across (the game's 1366 columns stretched to 1920) and down
NUMBER_AT = (749, 740)  # game px where the number's first glyph has its top left
NUMBER_END = 848        # game px: the number's cells end before this (the bar's right edge, screen column 1193,
                        # samples game x 848.6)
# The full-size screen box read (x0, y0, x1, y1): from the first screen column whose sample falls past game px 748
# (left of it the "EXP." label's dark outline reaches 1050) to the last before NUMBER_END - 1, and two rows of plain
# bar above the glyphs' and three below (row 1054 is the gauge's border).
READ_BOX = (1052, 1038, 1191, 1054)
GLYPHS = "0123456789,.[]%"
BLANK = " "             # a blank game column: the cells after the number (and anything else's, if it doesn't fit)
FIT = 4.0    # levels (root mean square over a cell): a glyph drawn as the game draws it. On the 32 saved 1920x1080
             # frames every cell was within 0.68 of it; the cursor, a window or anything else over the bar, much more
LEAD = 10.0  # levels (as FIT): how much worse every other reading of a cell must be. On the saved frames the least was
             # 14.1, a '[' against a ']' (16.9 a ',' against a '.', 21.6 an '8' against a '6' or a '9')
BG_SHARE = 10  # percent: each row's background is its darkest columns' (this share of them: text only brightens it,
               # and covers 59% of a row's columns at most in "13,549[67.02%]", 75% in "888,888[88.88%]")
PERCENT_SLACK = 0.51  # hundredths of a percent the percent may be off what the EXP is of the requirement: its rounding
NUMBER = re.compile(r"(0|[1-9]\d{0,2}(?:,\d{3})*)\[(0|[1-9]\d?|100)\.(\d\d)%\] *")
# The bar's colours behind the number, a row of READ_BOX each: the same on every saved frame (to a level). The simulator
# draws its EXP number over them.
BAR = ((78, 86, 90), (73, 81, 86), (70, 78, 84), (68, 76, 82), (66, 74, 80), (64, 72, 77), (62, 70, 75), (60, 68, 73),
       (58, 66, 72), (56, 64, 70), (54, 63, 68), (52, 61, 66), (51, 60, 64), (50, 58, 63), (49, 57, 62), (49, 57, 62))


@dataclass(frozen=True)
class ExpReading:
    """Her EXP number as read: `exp` (None: not read, `why` says why), the percent shown, and the levels whose
    requirement the two fit (one, or several where the percent is too small to tell them apart). `text`: the best
    path's glyphs, read or not; `fit` and `lead`: its worst cell's (levels)."""
    exp: int | None
    percent: float | None = None
    levels: tuple[int, ...] = ()
    text: str = ""
    fit: float = 0.0
    lead: float = 0.0
    why: str = ""

    @property
    def level(self) -> int | None:
        """Her level, when only one fits."""
        return self.levels[0] if len(self.levels) == 1 else None

    @property
    def need(self) -> int | None:
        """The EXP her level needs, when only one level fits."""
        return EXP_TO_NEXT.get(self.level) if self.level is not None else None


def sample_at(i: np.ndarray | int, scale: float, start: int) -> np.ndarray:
    """Where screen px `start` + i samples the game's picture (game px; px g's middle is at g)."""
    return (start + np.asarray(i, float) + 0.5) / scale - 0.5


def weights(samples: np.ndarray, first: int, n: int) -> np.ndarray:
    """Bilinear weights (len(samples) x n) of game px first .. first + n - 1 at each sample."""
    return np.maximum(0.0, 1.0 - np.abs(samples[:, None] - (first + np.arange(n))[None, :]))


@dataclass(frozen=True)
class _Cells:
    """Every glyph (and the blank) drawn at every game x it can start at, one cell each: where it starts and how far
    the next one does, its cell's screen columns of READ_BOX (c0 to c1), and the pixels where it shows over the
    background (flat indices into the box's pixels, `pixel` and `row`, of cell `cell`), with what it adds there:
    alpha `a`, premultiplied colour `c` (levels)."""
    char: np.ndarray     # (n,) the glyph's character
    x: np.ndarray        # (n,) game x
    advance: np.ndarray  # (n,)
    c0: np.ndarray       # (n,)
    c1: np.ndarray       # (n,)
    cell: np.ndarray     # (m,)
    pixel: np.ndarray    # (m,)
    row: np.ndarray      # (m,)
    a: np.ndarray        # (m,)
    c: np.ndarray        # (m, 3)
    ending: tuple        # per game x from NUMBER_AT[0]: the cells that end there ...
    starting: tuple      # ... and those that start there
    index: dict          # (character, game x) -> its cell


@lru_cache(maxsize=1)
def _cells() -> _Cells:
    """The cells (from the game data's pictures: 30 ms, and 150 to draw them all, once)."""
    from gamedata.canvas import status_numbers
    pictures = status_numbers()
    x0, y0, x1, y1 = READ_BOX
    width = x1 - x0
    across = sample_at(np.arange(width), SX, x0)
    down = sample_at(np.arange(y1 - y0), SY, y0)
    chars, xs, advances, c0, c1, cell, pixel, row, a, c = [], [], [], [], [], [], [], [], [], []
    for char in GLYPHS + BLANK:
        rgba = pictures[char].astype(float) / 255 if char != BLANK else np.zeros((7, 0, 4))
        h, w = rgba.shape[:2]
        alpha, colour = rgba[..., 3], rgba[..., :3] * 255 * rgba[..., 3:]  # colour in levels, premultiplied
        wy = weights(down, NUMBER_AT[1], h)
        for x in range(NUMBER_AT[0], NUMBER_END - w):
            mine = np.nonzero((across >= x - 1) & (across < x + w))[0]  # sampling game columns x - 1 .. x + w
            if w:
                wx = weights(across[mine], x, w)
                # (einsum, not @: numpy's float products go through its BLAS, whose DLL fails to load here)
                ai = np.einsum("yr,rc,xc->yx", wy, alpha, wx)
                ci = np.einsum("yr,rcz,xc->yxz", wy, colour, wx)
                ys, js = np.nonzero(ai > 0)
                cell.append(np.full(len(ys), len(xs))), pixel.append(ys * width + mine[js]), row.append(ys)
                a.append(ai[ys, js]), c.append(ci[ys, js])
            chars.append(char), xs.append(x), advances.append(w + 1), c0.append(mine[0]), c1.append(mine[-1] + 1)
    xs, advances = np.array(xs), np.array(advances)
    nodes = range(NUMBER_AT[0], NUMBER_END + 1)
    return _Cells(np.array(chars), xs, advances, np.array(c0), np.array(c1), np.concatenate(cell),
                  np.concatenate(pixel), np.concatenate(row), np.concatenate(a).astype(np.float32),
                  np.concatenate(c).astype(np.float32), tuple(np.nonzero(xs + advances == n)[0] for n in nodes),
                  tuple(np.nonzero(xs == n)[0] for n in nodes),
                  {(ch, int(x)): i for i, (ch, x) in enumerate(zip(chars, xs))})


def read(box: np.ndarray) -> ExpReading:
    """Her EXP number in `box`, the screen's READ_BOX (full size, RGB, as Eyes.look_at cuts it)."""
    if box is None or box.shape[:2] != (READ_BOX[3] - READ_BOX[1], READ_BOX[2] - READ_BOX[0]):
        return ExpReading(None, why="no EXP number box")
    cells = _cells()
    s = box.astype(np.float32)
    rows = s.shape[0]
    bg = np.percentile(s, BG_SHARE, axis=1).astype(np.float32)  # (rows, 3)
    over = s - bg[:, None, :]  # what's on the background
    # Each cell's squared difference from the screen: as if blank over its columns, then where its glyph shows.
    upto = np.concatenate([[0.0], np.cumsum((over ** 2).sum(axis=(0, 2)))])
    flat = over.reshape(-1, 3)[cells.pixel]
    off = flat - cells.c + bg[cells.row] * cells.a[:, None]
    cost = upto[cells.c1] - upto[cells.c0] + np.bincount(cells.cell, ((off ** 2) - (flat ** 2)).sum(axis=1),
                                                         minlength=len(cells.x))
    values = (cells.c1 - cells.c0) * rows * 3
    # The best path from NUMBER_AT to NUMBER_END (cost so far, and the cell into each game x), and the best from
    # each game x to the end.
    n = NUMBER_END - NUMBER_AT[0]
    so_far, into = np.full(n + 1, np.inf), np.full(n + 1, -1)
    so_far[0] = 0.0
    for i in range(1, n + 1):
        e = cells.ending[i]
        if len(e):
            via = so_far[cells.x[e] - NUMBER_AT[0]] + cost[e]
            j = int(np.argmin(via))
            so_far[i], into[i] = via[j], e[j]
    after = np.full(n + 1, np.inf)
    after[n] = 0.0
    for i in range(n - 1, -1, -1):
        e = cells.starting[i]
        if len(e):
            after[i] = (cost[e] + after[cells.x[e] + cells.advance[e] - NUMBER_AT[0]]).min()
    path, i = [], n
    while i > 0:
        path.append(int(into[i]))
        i = cells.x[path[-1]] - NUMBER_AT[0]
    path.reverse()
    text = "".join(cells.char[path])
    # Each cell of it: how well its glyph fits, and how much worse the best path through any other cell covering its
    # first column is (every path covers each game column with one cell).
    total = so_far[n]
    through = so_far[cells.x - NUMBER_AT[0]] + cost + after[cells.x + cells.advance - NUMBER_AT[0]]
    fit, lead = 0.0, np.inf
    for e in path:
        covers = (cells.x <= cells.x[e]) & (cells.x[e] < cells.x + cells.advance)
        covers[e] = False
        fit = max(fit, float(np.sqrt(max(0.0, cost[e]) / values[e])))
        lead = min(lead, float(np.sqrt(max(0.0, through[covers].min() - total) / values[e])))
    if fit > FIT:
        return ExpReading(None, text=text, fit=fit, lead=lead, why=f"a cell unlike any glyph ({fit:.1f} levels)")
    if lead < LEAD:
        return ExpReading(None, text=text, fit=fit, lead=lead, why=f"a cell that could be another ({lead:.1f} levels)")
    m = NUMBER.fullmatch(text)
    if m is None:
        return ExpReading(None, text=text, fit=fit, lead=lead, why=f"not an EXP number: {text.strip()!r}")
    exp, percent = int(m.group(1).replace(",", "")), float(f"{m.group(2)}.{m.group(3)}")
    hundredths = round(percent * 100)
    levels = tuple(lv for lv, need in sorted(EXP_TO_NEXT.items())
                   if exp < need and abs(exp * 10000 / need - hundredths) <= PERCENT_SLACK)
    if not levels:
        return ExpReading(None, percent, text=text, fit=fit, lead=lead,
                          why=f"{exp} at {percent:.2f}% fits no level's requirement")
    return ExpReading(exp, percent, levels, text, fit, lead)


def shown(exp: int, level: int) -> str:
    """What the bar shows with `exp` into `level`: "13,549[67.02%]" (the percent to the nearest hundredth, as the
    game rounds it)."""
    return f"{exp:,}[{exp * 100 / EXP_TO_NEXT[level]:.2f}%]"


def draw(text: str, under: np.ndarray | None = None) -> np.ndarray:
    """`text` drawn over READ_BOX as the game draws its EXP number (the reader's own model of it: the simulator's
    number), over the box `under` (RGB; by default the bar's own, BAR). Glyphs past the bar's end are left out."""
    cells = _cells()
    x0, y0, x1, y1 = READ_BOX
    out = (np.broadcast_to(np.array(BAR, np.float32)[:, None, :], (y1 - y0, x1 - x0, 3)) if under is None
           else under.astype(np.float32)).reshape(-1, 3).copy()
    x = NUMBER_AT[0]
    for ch in text:
        e = cells.index.get((ch, x))
        if e is None:
            break
        mine = slice(*np.searchsorted(cells.cell, [e, e + 1]))  # (a cell's pixels are together, in order)
        px, a = cells.pixel[mine], cells.a[mine][:, None]
        out[px] = out[px] * (1 - a) + cells.c[mine]
        x += int(cells.advance[e])
    return np.clip(out.reshape(y1 - y0, x1 - x0, 3) + 0.5, 0, 255).astype(np.uint8)
