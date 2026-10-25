"""Bars on screen: her HP and MP in the bottom bar, her EXP number (it changes only when she
gains EXP; what it says is read by exp_number, only then), the purple text when a monster touches her, and a
monster's own HP bar.

The px below were measured on the 1920x1080 window and are converted to the frame's own geometry (screen.Geometry):
at the game's own 1366x768 the HP bar's red is on row 756 from x 503 to 607, the MP bar's blue from 611 to 715, and
her EXP number's text on rows 740-746 from x 749 (n28_000.png, West Rocky Mountain II). A monster's HP bar was measured
at 1366x768, in game px (each a look frame's `frame_scale` px).
"""

from functools import lru_cache

import cv2
import numpy as np

from ..screen import STRETCHED, Geometry


# Her EXP number in the bottom bar ("1,385 [8.30%]"), full-size px: it only changes when she gains
# EXP, i.e. when a monster she hit dies. It sits on the bar's plain dark background, so its text pixels
# are identical from look to look otherwise. (The "You received EXP" message at the bottom right was
# tried first, but it's faint and was missed over a bright background.)
EXP_NUMBER = (1060, 1030, 1200, 1056)  # x0, y0, x1, y1 (at 1920x1080: exp_number_box)
# HP and MP bars in the bottom bar, as fractions of the game window (measured at 1920x1080; they hold at 1366x768).
HP_BAR = (707 / 1920, 854 / 1920)
MP_BAR = (859 / 1920, 1006 / 1920)
BAR_ROW = 1064 / 1080
EXP_CHANGED = 3   # pixels of her EXP number that must differ from the last look's for an EXP gain (at 1366x768 a
                  # digit's own change is 5 or more of its 35)


def exp_number_box(geo: Geometry = STRETCHED) -> tuple[int, int, int, int]:
    """EXP_NUMBER in a client area of this geometry's (screen px)."""
    return geo.box(EXP_NUMBER)


def bars(full: np.ndarray) -> tuple[float, float]:
    """HP and MP as fractions (0-1), read from the bar colours in a full-size game frame."""
    h, w = full.shape[:2]
    row = full[int(BAR_ROW * h)].astype(np.int16)

    def span(bar: tuple[float, float]) -> np.ndarray:
        return row[int(bar[0] * w):int(bar[1] * w)]

    return hp_fill(span(HP_BAR)), mp_fill(span(MP_BAR))


# A bar's fill is its colour well above the other two channels (BAR_TINT), whatever its brightness: when her HP
# changes the game flashes the bar pink (224, 150, 147) or dark red (77, 3, 0) for a moment, where it's (231, 9, 0)
# otherwise, and the old test (red over 150, green and blue under 90) read 0% on every flash: five times in 30 s of
# hunting on 2026-10-24, each a trip to rest. The empty part is grey (64-214 on all three).
BAR_TINT = 40


def hp_fill(row: np.ndarray) -> float:
    """The share of a row of the HP bar (RGB, int16) that's red: her HP left."""
    on = row[:, 0] - np.maximum(row[:, 1], row[:, 2]) > BAR_TINT
    return float(on.sum()) / max(1, len(row))


def mp_fill(row: np.ndarray) -> float:
    """The share of a row of the MP bar (RGB, int16) that's blue: her MP left."""
    on = row[:, 2] - np.maximum(row[:, 0], row[:, 1]) > BAR_TINT
    return float(on.sum()) / max(1, len(row))


def exp_changed(before: np.ndarray | None, now: np.ndarray | None) -> bool:
    """Whether her EXP number (its text pixels, Observation.exp) changed between two looks: it went up."""
    return (before is not None and now is not None and now.shape == before.shape
            and int((now != before).sum()) >= EXP_CHANGED)


def touch_popup(frame: np.ndarray, tag: tuple[int, int], geo: Geometry = STRETCHED) -> int:
    """Purple pixels above her head (a look's frame, of geometry `geo`; `tag`: her name tag's centre). The "MISS" a
    thief's evasion shows when a monster touches her without damage, and the numbers of damage she
    takes, are both purple, and nothing else on screen is. A count of 40+ means a monster touched her (in a half-size
    frame of the 1920x1080 window). (Used to count touches when hops over monsters were timed; nothing calls it during
    a hunt.)"""
    x, y = tag
    up, low, side = geo.ifpx(120), geo.ifpx(15), geo.ifpx(70)
    box = frame[max(0, y - up):max(0, y - low), max(0, x - side):x + side].astype(np.int16)
    if box.size == 0:
        return 0
    r, g, b = box[..., 0], box[..., 1], box[..., 2]
    return int(((r > 120) & (b > 150) & (g < 110) & (b - g > 80) & (r - g > 50)).sum())


# A monster's HP bar as the game draws it, in game px (measured at 1366x768, where it's drawn pixel for pixel: the saved
# frames and the rolling recorder's clips of 2026-10-24 and 25, 567 bars over Green and Horny Mushrooms). A white frame
# (BAR_FRAME, (255, 255, 255)) between two black ones, one outside and one inside it, and inside those the fill
# (BAR_FILL): three rows (BAR_GREEN) green (0, 255, 0) for the HP left, over a fourth red (170, 0, 0); after it the
# latest damage, red (about 220 in the clips) fading to dark red ((136, 0, 0), over (97, 0, 0)), and the HP lost black.
# A party member's bar over a player is the same frame filled red, and a monster's as it dies all red and black: no
# green. At 1920x1080 it's that picture stretched, and a look's half size blends each 1-px line into its neighbours.
# Until 2026-10-25 the bar was read as the longest run of pure green, black and dark red in a row above the monster,
# meant to be the green row: at 1366x768 the black frame outside is 52 px of pure black, longer than the fill's 48, and
# of the 503 bars that read it was the run read on 375 (the black rows inside on 53, the red row under the green on 49),
# so 488 of 567 read empty and 64 none; the scenery's 1-px black outlines read as empty bars over 267 of 2207 monsters
# with none. At half size of 1920x1080 the black rows blend away or not by where the bar falls: 111 of 161 bars on the
# first map's recordings read empty (and live there on 2026-10-21, a third of the bars read 0).
BAR_FRAME = (50, 8)  # the light frame's width and height
BAR_FILL = (2, 46)   # the fill's left in it, and its width
BAR_GREEN = (2, 5)   # the fill's rows from the frame's top that are green for the HP left (the next is red under it)
BAR_AT = (-25, -29)  # the light frame's top-left from a monster's centre and top (its picture's): -25, -29 on every
                     # one of the 574 bars over the mushrooms' poses at 1366x768 (the 1920x1080 window's pigs 17.8-20.8
                     # half-size px above, its slimes, whose pictures change height as they bounce, 0-21)
BAR_SLACK = (14, 12, 31)  # game px it's looked for either side of there, above, and below
BAR_MATCH = 0.75     # the frame's least colour against the bar's (TM_CCOEFF_NORMED) that makes one: 0.75-1.0 on the
                     # labelled bars (under 0.84 on 1% of them: behind sparkles, a crescent of light), at most 0.73
                     # around a monster with none
BAR_TIE = 0.1        # ... and how much less than the best match another place may match to be a bar of its own
BAR_GREENISH = 60    # a fill px's green over its red and blue that makes it the HP left (255 for (0, 255, 0); the JPEG
                     # clips' 160 and up, their lost px' 40 at most)
BAR_LOST = 80        # green and blue under this in a fill px: the latest damage or the HP lost (red or black)


@lru_cache(maxsize=4)
def _bar_frames(scale: float) -> tuple[tuple[float, float, np.ndarray], ...]:
    """The bar's frame (BAR_FRAME, the black frames outside and inside it, and the fill all dark: its least colour)
    as a look frame of `scale` frame px a game px shows it, 1 light and 0 dark, for each place a game px's edge can be
    within a frame px (in quarters across, halves down; at 1 only on one): (x, y of the light frame's top-left in it,
    frame px, and the picture). Each frame px is the share of it each game px covers; those the bar covers only partly
    (with whatever is around it) are left out."""
    game = _bar_picture(None)[..., 0] / 255
    across = (0.0,) if scale == 1 else (0.0, 0.25, 0.5, 0.75)  # (where the fill starts, which its share goes by)
    down = (0.0,) if scale == 1 else (0.0, 0.5)
    out = []
    for py in down:
        ay = _cover(game.shape[0], scale, py)
        rows = np.nonzero(ay.sum(axis=1) > 0.999)[0]
        for px in across:
            ax = _cover(game.shape[1], scale, px)
            cols = np.nonzero(ax.sum(axis=1) > 0.999)[0]
            picture = (ay[rows] @ game @ ax[cols].T).astype(np.float32)
            out.append((px + scale - cols[0], py + scale - rows[0], picture))
    return tuple(out)


def _cover(n: int, scale: float, phase: float) -> np.ndarray:
    """Frame px x game px: how much of frame px k game px g covers, the game px's edges at phase + g * scale."""
    k = np.arange(int(np.ceil(n * scale + phase)) + 1)[:, None]
    g = np.arange(n)[None, :]
    return np.clip(np.minimum(k + 1, (g + 1) * scale + phase) - np.maximum(k, g * scale + phase), 0, None)


def _bar_picture(left: float | None) -> np.ndarray:
    """The bar in game px (RGB, float), its black frame outside included: `left` the share of the fill green (rounded
    to the game's px), or None for its frames alone, every colour but the light frame's black."""
    w, h = BAR_FRAME[0] + 2, BAR_FRAME[1] + 2
    bar = np.zeros((h, w, 3), np.float32)
    bar[1, 1:w - 1] = bar[h - 2, 1:w - 1] = bar[1:h - 1, 1] = bar[1:h - 1, w - 2] = 255
    if left is not None:
        x0, n = 1 + BAR_FILL[0], int(round(BAR_FILL[1] * min(1.0, max(0.0, left))))
        bar[1 + BAR_GREEN[0]:1 + BAR_GREEN[1], x0:x0 + n] = (0, 255, 0)
        bar[1 + BAR_GREEN[1], x0:x0 + n] = (170, 0, 0)
    return bar


def draw_hp_bar(frame: np.ndarray, x: float, top: float, left: float, scale: float) -> None:
    """Draw a monster's HP bar into `frame` (an RGB look frame of `scale` px a game px) as the game shows it over a
    monster whose centre is at `x` and top at `top` (BAR_AT), `left` of its fill green: for the simulator's frames and
    tests. Each frame px takes the share of it each game px of the bar covers (as a stretched window blends them), the
    rest of it what was there."""
    bar = _bar_picture(left)
    ox, oy = x + (BAR_AT[0] - 1) * scale, top + (BAR_AT[1] - 1) * scale  # the black frame outside's top-left
    c0, r0 = int(np.floor(ox)), int(np.floor(oy))
    ax, ay = _cover(bar.shape[1], scale, ox - c0), _cover(bar.shape[0], scale, oy - r0)
    cols, rows = np.arange(c0, c0 + ax.shape[0]), np.arange(r0, r0 + ay.shape[0])
    cx, ry = (cols >= 0) & (cols < frame.shape[1]), (rows >= 0) & (rows < frame.shape[0])
    if not cx.any() or not ry.any():
        return
    ax, ay = ax[cx], ay[ry]
    picture = np.einsum("ig,ghc,jh->ijc", ay, bar, ax)
    covered = ay.sum(axis=1)[:, None, None] * ax.sum(axis=1)[None, :, None]
    region = frame[rows[ry][0]:rows[ry][-1] + 1, cols[cx][0]:cols[cx][-1] + 1]
    region[...] = np.clip(np.rint(region * (1 - covered) + picture), 0, 255).astype(frame.dtype)


def hp_bar(frame: np.ndarray, x: int, top: int, geo: Geometry = STRETCHED) -> float | None:
    """A monster's HP bar, shown above it once it has been hit: the share of its fill still green (the HP left), or
    None if there's no bar (or none with any green: a party member's over a player, a monster's as it dies). `frame` is
    a look's RGB frame (of geometry `geo`); (x, top) the monster's centre and top.

    The bar is found by its frame (BAR_FRAME: a white one between black ones), its least colour matched against the
    bar's around BAR_AT from the monster (BAR_SLACK): of the places it matches (BAR_MATCH) about as well as the best
    (BAR_TIE), the nearest there. Its fill is read off its green rows: the HP left is green from the left (BAR_GREENISH)
    to the last green column, whatever covers some of it (a damage number over its top rows, sparkles, the latest
    damage's red: the end is still the last green), its blended edge counted by how green it is. After it come the
    latest damage and the HP lost (BAR_LOST), unless something covers the end (a damage number, a sprite, a vine):
    then the end is read as halfway along what covers it, to the first red or black after it (on the clips, 0.45, 0.73
    and 0.47 where the next frame without it read 0.46, 0.72 and 0.50)."""
    s = geo.frame_scale
    (ax, ay), (sx, up, down) = BAR_AT, BAR_SLACK
    fw, fh = BAR_FRAME[0] * s, BAR_FRAME[1] * s
    x0, y0 = max(0, int(np.floor(x + (ax - sx) * s - 2))), max(0, int(np.floor(top + (ay - up) * s - 2)))
    x1 = min(frame.shape[1], int(np.ceil(x + (ax + sx) * s + fw + 2)))
    y1 = min(frame.shape[0], int(np.ceil(top + (ay + down) * s + fh + 2)))
    if x1 - x0 < fw or y1 - y0 < fh:
        return None
    box = frame[y0:y1, x0:x1]
    least = box.min(axis=2).astype(np.float32) / 255
    found = []  # (how well, the light frame's top-left x, y in the box) where it matches best around
    for px, py, picture in _bar_frames(s):
        th, tw = picture.shape
        if least.shape[0] < th or least.shape[1] < tw:
            continue
        match = np.nan_to_num(cv2.matchTemplate(least, picture, cv2.TM_CCOEFF_NORMED), nan=0.0, posinf=0.0, neginf=0.0)
        peaks = (match >= BAR_MATCH) & (match >= cv2.dilate(match, np.ones((5, 5), np.uint8)))
        found += [(float(match[yy, xx]), xx + px, yy + py) for yy, xx in zip(*np.nonzero(peaks))]
    # Each bar once, where it matches best: none within half a bar of a better one (the frame slid along its own long
    # rows matches too, less well: slid 3.5 half-size px, 0.90 against 0.99; slid 7, 0.81, and nearer where a slime's is
    # expected, it read 0.97 for 0.74). Of those matching about as well as the best (BAR_TIE), the nearest to where it's
    # expected first.
    found.sort(reverse=True)
    places = []
    for score, fx, fy in found:
        if score >= found[0][0] - BAR_TIE and all(abs(fx - qx) >= fw / 2 or abs(fy - qy) >= fh / 2
                                                  for _, _, qx, qy in places):
            places.append((abs(fx + x0 - (x + ax * s)) + abs(fy + y0 - (top + ay * s)), -score, fx, fy))
    for _, _, fx, fy in sorted(places):
        left = _bar_fill(box, fx, fy, s)
        if left is not None:
            return left
    return None


def _rows_in(lo: float, hi: float) -> list[int]:
    """The frame rows wholly within [lo, hi) (frame px), or if none is (a stretch blending them all with the rows
    around), the one most in it."""
    return list(range(int(np.ceil(lo)), int(np.floor(hi)))) or [int(np.floor((lo + hi) / 2))]


def _bar_fill(box: np.ndarray, fx: float, fy: float, s: float) -> float | None:
    """The share of a bar's fill that's green (see hp_bar), its light frame's top-left at (fx, fy) in `box` (frame px,
    `s` of them a game px); None if none of it is."""
    lo, width = fx + BAR_FILL[0] * s, BAR_FILL[1] * s
    rows = _rows_in(fy + BAR_GREEN[0] * s, fy + BAR_GREEN[1] * s)
    under = _rows_in(fy + BAR_GREEN[0] * s, fy + (BAR_GREEN[1] + 1) * s)  # and the red row under them
    c0, c1 = int(np.floor(lo)), int(np.ceil(lo + width))
    if min(rows + under) < 0 or max(rows + under) >= box.shape[0] or c0 < 0 or c1 > box.shape[1]:
        return None
    fill = box[:, c0:c1].astype(np.int16)
    r, g, b = fill[..., 0], fill[..., 1], fill[..., 2]
    # Green over red and blue, with blue under half of it: the HP left (a glow tinting it yellow still is; a vine's mint
    # green over the bar isn't, nor sparkles' white tinting it), each column as green as its greenest row (what covers
    # some of the rows leaves the others).
    green = np.where(2 * b < g, g - np.maximum(r, b), 0)[rows].max(axis=0)
    on = np.nonzero(green > BAR_GREENISH)[0]
    if on.size == 0:
        return None
    full = max(1, int(np.median(green[on])))  # the fill's own green (255; less in the JPEG clips)
    last = int(np.nonzero(green >= full / 2)[0][-1])
    end = last + 1.0
    if last + 1 < green.size:  # the next column partly green (a stretch blending green with what's after it)
        end += float(np.clip(green[last + 1] / full, 0, 1))
    # What's after the green is the latest damage and the HP lost (red or black: little green or blue in all four
    # rows), unless something covers it (a damage number, a sprite drawn over the bar, a vine): then the green's end is
    # somewhere under it, read as halfway along it.
    lost = ((g[under] < BAR_LOST) & (b[under] < BAR_LOST)).mean(axis=0) >= 0.5
    after = int(np.ceil(end))
    if after < green.size and not lost[after]:
        seen = np.nonzero(lost[after:])[0]
        end = (end + (after + seen[0] if seen.size else green.size - (c1 - lo - width))) / 2
    return float(np.clip((c0 + end - lo) / width, 0.0, 1.0))
