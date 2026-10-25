"""The minimap (top left): her yellow dot, and other players' dots, in full-size game-window pixels.

Every dot is one of the game's own markers (Map.wz, MapHelper.img/minimap): `user` (hers: a yellow core 6 game px
across), `another` (anyone else: red), `party` and `partymaster` (her party: orange, the leader's with a yellow
crown above), `guild` and `guildmaster` (lavender), `friend` (a buddy: cyan), each a core 5 game px across in a dark
grey ring and a white halo, 11-12 px in all; `npc` (green, taller), `portal` (cyan rings, 14 px) and others. The
game draws them at its own 1366x768, each at a whole game pixel, and the whole picture is shown 1.40625x larger
with bilinear filtering: a stranger's marker pasted so onto a saved frame came out within half a level of the one
the game drew there (`hover_D.png`, on average over a 30 px square). So a core is 7 px across on screen.

At the game's own 1366x768 (since 2026-10-23) every marker is its picture, pixel for pixel: her yellow core 6 px
across (24 px of yellow), another's red core 5 across (21 px), on West Rocky Mountain II's 31 saved frames of that
size. The px below were measured on the 1920x1080 window and go by the frame's geometry (`sizes`: screen.Geometry).
"""

import time
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from ..screen import STRETCHED, Geometry, of_frame

# The minimap window sits at the top left; its map picture is inside this box (full-size px at 1920x1080; at
# 1366x768 (0, 68, 213, 213)).
MINIMAP_BOX = (0, 95, 300, 300)  # x0, y0, x1, y1
DOT_AREA = (12, 200)  # px a yellow blob covers to be her dot (smallest, largest) ...
DOT_SIZE = 16         # ... and its widest and tallest
# Her dot's yellow uncovered: 7-8 px wide and tall (37-41 px of it on the 31 saved 1920x1080 frames of this map, 38-42
# for her marker pasted at 32 places across the minimap). A dot narrower or shorter than this has something drawn
# over part of it: another player's marker, if one touches it (DOT_UNCOVER).
DOT_FULL = 7
DOT_CORE = 6  # game px: her marker's yellow core, which no window size shows narrower (1366x768 shows it 6 px wide)
# Whether her dot, partly under another player's marker, is taken from its side that shows (as wide as DOT_FULL). Drawn
# over hers 4-5 game px off (46-57 map units), another's marker pulled her yellow's middle 1.3-1.7 px its way on
# average (15-19 units), up to 2.2; 6 px off 0.6; taken so, 0.4 (up to 1.2). (The game's markers pasted at 600 places
# around her on the saved frames.) Which the game draws on top isn't known: under hers, theirs covers nothing of her
# dot, and this changes nothing. Over hers within 4 px, or a leader's (15 px tall with its crown) just below her, a
# marker can hide her dot whole.
DOT_UNCOVER = True


@dataclass(frozen=True)
class Marker:
    """Another player's dot on the minimap: what kind of player, and where its core's middle is (full-size px)."""
    kind: str   # "stranger", "party", "guild" or "buddy"
    x: float
    y: float


# Each kind's core colours, as OpenCV's HSV (hue 0-180) ranges, wide enough for the core's own shades and its edges
# blended with the dark ring and the halo: `another` dd0000-ff0000 (hue 0); `party` ff7700-ffaa00 and dd9900 (14-21);
# `friend` 00ccee-44ddff and 44bbdd (94-97); `guild` 6666ff-ccccff and aaaadd (120, washed out: saturation 51-153).
# Her yellow (ffdd44, ffee22: hue 25-28), the party leader's crown (ffcc00: 24), the NPCs' green (60) and the
# minimap's dark blue (18334c: hue 102, value 76) are none of them. Portals' cyan is a buddy's, but drawn as two rings
# with dark between them, never a solid core of a buddy's size. The buddy's range stops 2 short of the other cyan
# markers' in the data: `eventNerota`'s 44ffff (90) and the `zeronpc` NPCs' 11aaff (101), which with 88-102 were each
# found as a buddy wherever they were drawn.
MARKER_COLOURS = {
    "stranger": (((0, 170, 160), (8, 255, 255)), ((172, 170, 160), (180, 255, 255))),
    "party": (((10, 190, 190), (22, 255, 255)),),
    "buddy": (((92, 120, 190), (99, 255, 255)),),
    "guild": (((114, 40, 190), (126, 175, 255)),),
}
# A core on screen: 32-40 px (the 3 saved frames with other players' markers, 11 of them), 6-7 px wide and tall; 31-40
# and 6-7 for the game's markers pasted at 34 places; 24-44 and 5-8 for the 105 found on the saved frames, half-size
# ones enlarged back among them. A core partly under another marker is smaller; under 14 px it isn't counted. Bigger
# than these, it's not a player's: `compassActive`'s cyan star (its middle 10 px across, 60 px) passed as a buddy's
# with up to 11 px and 70.
MARKER_AREA = (14, 52)
MARKER_SIDE = (4, 9)
# ... and solid: at least this share of its box (at 1366x768, where nothing blurs the markers: a core is 21 of its
# 5 x 5, and a portal's inner ring, 4 x 4 with a hole, 8, passed as a buddy's core at both portals on all 31 saved
# frames of that size). At 1920x1080 the filtering blurs a portal's rings into its dark, and no part of one was ever a
# core's size: not asked there (Sizes.marker_fill 0), as before.
MARKER_FILL = 0.7
# Around a core, its dark grey ring (333333-555555, 3 game px from its middle: 4.2 on screen): at most RING_RADII
# from the core's middle, at least RING_DARK of RING_ANGLES directions must reach a dark grey pixel. 15-16 of 16 do
# on the saved frames; another marker drawn over a side covers a few.
RING_RADII = np.arange(3.0, 5.6, 0.5)
RING_ANGLES = 16
RING_DARK = 0.6
_ANGLES = np.linspace(0, 2 * np.pi, RING_ANGLES, endpoint=False)
_RING = np.stack([np.outer(np.cos(_ANGLES), RING_RADII), np.outer(np.sin(_ANGLES), RING_RADII)], axis=-1)
# Her marker is 12 game px with its core's middle on its origin, the others' 11 px with theirs half a pixel right of
# and below theirs (origins (6, 6) and (5, 5) in the game data): another player's core is drawn 0.5 game px (0.70
# screen px) right of and below where hers would be at the same place. Measured on the saved frames: hers 0.66 and
# 0.50 px left of and above the game pixel she's drawn at, a stranger's 0.16 and 0.17 right of and below its.
OTHER_SHIFT = 0.5 * STRETCHED.stretch  # at 1920x1080 (Sizes.other_shift)


@dataclass(frozen=True)
class Sizes:
    """The px above in a frame of one geometry: the 1920x1080 window's own, else scaled with the game's stretch
    (areas with its square), but for a full dot (DOT_CORE at least: no size shows the core narrower, and 1366x768
    shows it exactly) and its middle (at 1366x768 the core's own: 2.5 px from its edges)."""
    box: tuple[int, int, int, int]
    dot_area: tuple[float, float]
    dot_size: int
    dot_full: int
    dot_half: float
    marker_area: tuple[float, float]
    marker_side: tuple[float, float]
    marker_fill: float
    ring: np.ndarray
    other_shift: float


@lru_cache(maxsize=8)
def sizes(geo: Geometry = STRETCHED) -> Sizes:
    """The minimap's px in a frame of geometry `geo`."""
    area = lambda n: geo.px(geo.px(n))  # px squared (exactly n at 1920x1080)
    full = max(DOT_CORE, geo.ipx(DOT_FULL))
    # A full dot's middle from its first and last px: 7-8 px blurred at 1920x1080 (3-3.5); the core itself at 1366x768
    half = (DOT_CORE - 1) / 2 if geo.native else (full - 0.5) / 2
    ring = _RING if geo == STRETCHED else _RING * geo.px(1.0)
    return Sizes(geo.box(MINIMAP_BOX), (area(DOT_AREA[0]), area(DOT_AREA[1])), geo.ipx(DOT_SIZE), full, half,
                 (area(MARKER_AREA[0]), area(MARKER_AREA[1])), (geo.px(MARKER_SIDE[0]), geo.px(MARKER_SIDE[1])),
                 0.0 if geo == STRETCHED else MARKER_FILL, ring, 0.5 * geo.stretch)


def minimap_box(geo: Geometry = STRETCHED) -> tuple[int, int, int, int]:
    """MINIMAP_BOX in a frame of geometry `geo` (full-size px)."""
    return sizes(geo).box


def her_dot(full: np.ndarray, geo: Geometry | None = None) -> tuple[float, float] | None:
    """Her yellow dot on the minimap, in game-window pixels (full size), or None if hidden. `geo`: the frame's
    geometry (by default from its size)."""
    geo = geo or of_frame(full)
    x0, y0, x1, y1 = minimap_box(geo)
    return dot_in_box(full[y0:y1, x0:x1], geo)


def dot_in_box(box: np.ndarray, geo: Geometry = STRETCHED) -> tuple[float, float] | None:
    """Her dot in a capture of just `minimap_box(geo)` (RGB), in game-window pixels: the biggest yellow blob of
    a dot's size. Partly under another player's marker, from the side of it that shows (DOT_UNCOVER)."""
    px = sizes(geo)
    x0, y0 = px.box[:2]
    area = box.astype(np.int16)
    r, g, b = area[..., 0], area[..., 1], area[..., 2]
    yellow = ((r > 220) & (g > 190) & (b < 90)).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(yellow)
    blobs = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, count)
             if px.dot_area[0] <= stats[i, cv2.CC_STAT_AREA] <= px.dot_area[1]
             and stats[i, cv2.CC_STAT_WIDTH] <= px.dot_size and stats[i, cv2.CC_STAT_HEIGHT] <= px.dot_size]
    if not blobs:
        return None
    _, i = max(blobs, key=lambda blob: blob[0])
    cx, cy = float(centroids[i][0]), float(centroids[i][1])
    if DOT_UNCOVER and (stats[i, cv2.CC_STAT_WIDTH] < px.dot_full or stats[i, cv2.CC_STAT_HEIGHT] < px.dot_full):
        cx, cy = _uncovered(box, stats[i], cx, cy, geo)
    return cx + x0, cy + y0


def _uncovered(box: np.ndarray, st: np.ndarray, cx: float, cy: float,
               geo: Geometry = STRETCHED) -> tuple[float, float]:
    """Her dot's middle when its yellow blob (stats `st`, middle (cx, cy), box px) is cut short across or up and down:
    with another player's marker touching it on one side, from its edge on the other, as if it were DOT_FULL px. A
    yellow cut short with no marker by it (the minimap's edge) stays as it is."""
    px = sizes(geo)
    left, top, w, h = (int(v) for v in st[:4])
    pad = px.dot_size
    wx, wy = max(0, left - pad), max(0, top - pad)
    near = markers_in_box(box[wy:top + h + pad, wx:left + w + pad], geo)
    touching = [(m.x + wx, m.y + wy) for m in near
                if abs(m.x + wx - cx) <= 2 * px.dot_full and abs(m.y + wy - cy) <= 2 * px.dot_full]
    if not touching:
        return cx, cy
    mx, my = min(touching, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    half = px.dot_half
    if w < px.dot_full and abs(mx - cx) >= abs(my - cy) / 2:
        cx = left + half if mx > cx else left + w - 1 - half
    if h < px.dot_full and abs(my - cy) >= abs(mx - cx) / 2:
        cy = top + half if my > cy else top + h - 1 - half
    return cx, cy


def other_dots(full: np.ndarray, geo: Geometry | None = None) -> list[Marker]:
    """Other players' dots on the minimap (full-size px), in a capture of the whole game window (of geometry `geo`,
    by default from its size)."""
    geo = geo or of_frame(full)
    x0, y0, x1, y1 = minimap_box(geo)
    return [Marker(m.kind, m.x + x0, m.y + y0) for m in markers_in_box(full[y0:y1, x0:x1], geo)]


def markers_in_box(box: np.ndarray, geo: Geometry = STRETCHED) -> list[Marker]:
    """Other players' markers in a picture of (part of) the minimap (RGB), in its px: a core of a kind's colour
    (MARKER_COLOURS), of a core's size (MARKER_AREA, MARKER_SIDE), in a dark grey ring (RING_DARK). Two markers
    5 game px apart or more (57 map units) are two; 4 px apart, a third of the time; nearer, the one drawn last hides
    most of the other's core, and they're one. `geo`: the geometry of the frame it's from."""
    if box.size == 0:
        return []
    px = sizes(geo)
    hsv = cv2.cvtColor(np.ascontiguousarray(box), cv2.COLOR_RGB2HSV)
    out = []
    for kind, ranges in MARKER_COLOURS.items():
        mask = cv2.inRange(hsv, *ranges[0])
        for lo, hi in ranges[1:]:
            mask |= cv2.inRange(hsv, lo, hi)
        # Only where the kind's colour is (most looks: none of it, or the portals' cyan): 0.44 ms a look in all (median
        # on the 31 saved frames, at most 0.56), against 1.9 with each kind's blobs counted over the whole box.
        x0, y0, w0, h0 = cv2.boundingRect(mask)
        if not w0:
            continue
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask[y0:y0 + h0, x0:x0 + w0], connectivity=8)
        for i in range(1, count):
            a, w, h = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if not (px.marker_area[0] <= a <= px.marker_area[1] and px.marker_side[0] <= w <= px.marker_side[1]
                    and px.marker_side[0] <= h <= px.marker_side[1] and a >= px.marker_fill * w * h):
                continue
            cx, cy = float(centroids[i][0]) + x0, float(centroids[i][1]) + y0
            if _ringed(box, cx, cy, px.ring):
                out.append(Marker(kind, cx, cy))
    return sorted(out, key=lambda m: (m.y, m.x))


def _ringed(box: np.ndarray, cx: float, cy: float, ring: np.ndarray = _RING) -> bool:
    """Whether a core at (cx, cy) has a marker's dark grey ring around it (RING_DARK of the directions; `ring`: the
    offsets looked at, Sizes.ring)."""
    h, w = box.shape[:2]
    xs = np.clip(np.rint(cx + ring[..., 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(cy + ring[..., 1]).astype(int), 0, h - 1)
    px = box[ys, xs].astype(np.int16)  # directions x radii x RGB
    hi, lo = px.max(axis=2), px.min(axis=2)
    dark = (hi < 130) & (hi - lo < 50)
    return dark.any(axis=1).mean() >= RING_DARK


# Where the map's own minimap picture is in the window, by matching it (the miniMap canvas from the game data,
# mapdata.minimap_picture): the window centres the picture under the map's name, as wide as that name needs, and on
# a map taller than the window shows (Tree Dungeon, Forest Up North II: 67 x 183 game px in a window showing 112 rows)
# it scrolls the picture to keep her dot in view, so where map point (-centerX, -centerY) is on screen moves as she
# climbs. Her dot read with the first map's origin there put her x at 902 on a map 360..410 wide (2026-10-24).
# Matched as bands of the picture (each PICTURE_BAND of the box's height, or all of it if shorter), every band's place
# in the window telling where the picture's top-left corner is: on that map's 20 saved 1366x768 frames the best band
# scored 0.65-0.74, the best at any other place 0.20-0.26, and her dot then fell on the platform the screen showed
# (28 at (259, -47); 19 at (307, -639)).
PICTURE_BASE = (47, 72)  # game px: the picture's top-left corner on the first map, where DOT_ORIGIN was fitted
PICTURE_MATCH = 0.45     # the least match (TM_CCOEFF_NORMED) a band's place is taken at
PICTURE_BAND = 0.55      # a band's height, as a share of the box's
PICTURE_NEAR = 16        # px around the last corner a look searches first (a scroll is one dot's move at most)
PICTURE_RETRY_S = 1.0    # s after a whole search found nothing before the next (the minimap closed: 5-10 ms a look)


class MinimapPicture:
    """Finds the map's minimap picture in captures of `minimap_box(geo)`, keeping where it last was (`corner`)."""

    def __init__(self, picture: np.ndarray, geo: Geometry = STRETCHED):
        k = geo.stretch
        h, w = picture.shape[:2]
        if k != 1:  # shown 1.40625x larger at 1920x1080, with bilinear filtering
            picture = cv2.resize(picture, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_LINEAR)
        self.picture = np.ascontiguousarray(picture[..., :3]).astype(np.float32)
        self.geo = geo
        self.corner: tuple[float, float] | None = None  # full-size px in the window, as last found
        self.score: float | None = None  # the best match of the last search (None: none made, the box too small)
        self._retry_at = 0.0  # time.monotonic() before which no whole search is made again (PICTURE_RETRY_S)

    def lose(self) -> None:
        """Forget where the picture was, and search the whole box at the next look (after a look that failed its
        checks: perception.framecheck; a corner found in a frame offset from the window would be off by as much)."""
        self.corner, self._retry_at = None, 0.0

    def find(self, box: np.ndarray) -> tuple[float, float] | None:
        """The picture's top-left corner (full-size px in the game window) in a capture of `minimap_box` (RGB), or
        None if no band of it matches there (the minimap closed, covered or another map's); `corner` keeps the
        last one found."""
        x0, y0 = minimap_box(self.geo)[:2]
        box = box.astype(np.float32)
        ph, pw = self.picture.shape[:2]
        bh = min(ph, max(8, int(box.shape[0] * PICTURE_BAND)))
        if box.shape[0] < bh or box.shape[1] < pw:
            self.score = None
            return None
        best = None
        if self.corner is not None:  # first where it was: the band across the box's middle, around its last place
            cx, cy = self.corner[0] - x0, self.corner[1] - y0
            top = int(round(min(max(0, box.shape[0] / 2 - bh / 2 - cy), ph - bh)))
            ax, ay = int(round(cx)) - PICTURE_NEAR, int(round(cy + top)) - PICTURE_NEAR
            sub = box[max(0, ay):ay + bh + 2 * PICTURE_NEAR, max(0, ax):ax + pw + 2 * PICTURE_NEAR]
            if sub.shape[0] >= bh and sub.shape[1] >= pw:
                best = self._best(sub, top, bh, max(0, ax), max(0, ay))
        if (best is None or best[0] < PICTURE_MATCH) and time.monotonic() >= self._retry_at:
            for top in range(0, ph - bh + 1, max(1, bh // 4)):  # every quarter band: one is in view
                found = self._best(box, top, bh, 0, 0)
                if best is None or found[0] > best[0]:
                    best = found
        self.score = None if best is None else best[0]
        if best is None or best[0] < PICTURE_MATCH:
            self._retry_at = time.monotonic() + PICTURE_RETRY_S
            return None
        self.corner = (x0 + best[1], y0 + best[2])
        return self.corner

    def _best(self, area: np.ndarray, top: int, bh: int, ox: int, oy: int) -> tuple[float, int, int]:
        """(score, x, y): where the band of the picture from row `top` matches best in `area`, as the corner's px in
        the box (`area` starts at (ox, oy) in it)."""
        r = cv2.matchTemplate(area, self.picture[top:top + bh], cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(r)
        return float(score), ox + x, oy + y - top

    def shift(self, corner: tuple[float, float] | None = None) -> tuple[float, float]:
        """How far (full-size px) the picture is from where it was on the first map (PICTURE_BASE), by whole game px
        (it's drawn at whole game px): what to take off a dot to read it with that map's origin. Nothing until it's
        been found."""
        corner = corner or self.corner
        if corner is None:
            return 0.0, 0.0
        g = self.geo
        gx, gy = round(g.to_game(corner[0])), round(g.to_game(corner[1]))
        return (gx - PICTURE_BASE[0]) * g.stretch, (gy - PICTURE_BASE[1]) * g.stretch  # a distance: no half px
