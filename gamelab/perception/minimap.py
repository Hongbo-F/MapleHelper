"""The minimap (top left): her yellow dot, and other players' dots, in full-size game-window pixels.

Every dot is one of the game's own markers (Map.wz, MapHelper.img/minimap): `user` (hers: a yellow core 6 game px
across), `another` (anyone else: red), `party` and `partymaster` (her party: orange, the leader's with a yellow
crown above), `guild` and `guildmaster` (lavender), `friend` (a buddy: cyan), each a core 5 game px across in a dark
grey ring and a white halo, 11-12 px in all; `npc` (green, taller), `portal` (cyan rings, 14 px) and others. The
game draws them at its own 1366x768, each at a whole game pixel, and the whole picture is shown 1.40625x larger
with bilinear filtering: a stranger's marker pasted so onto a saved frame came out within half a level of the one
the game drew there (`hover_D.png`, on average over a 30 px square). So a core is 7 px across on screen.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from ..screen import SCREEN_SCALE

# The minimap window sits at the top left; its map picture is inside this box (full-size px).
MINIMAP_BOX = (0, 95, 300, 300)  # x0, y0, x1, y1
DOT_AREA = (12, 200)  # px a yellow blob covers to be her dot (smallest, largest) ...
DOT_SIZE = 16         # ... and its widest and tallest
# Her dot's yellow uncovered: 7-8 px wide and tall (37-41 px of it on the 31 saved 1920x1080 frames of this map, 38-42
# for her marker pasted at 32 places across the minimap). A dot narrower or shorter than this has something drawn
# over part of it: another player's marker, if one touches it (DOT_UNCOVER).
DOT_FULL = 7
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
OTHER_SHIFT = 0.5 * SCREEN_SCALE


def her_dot(full: np.ndarray) -> tuple[float, float] | None:
    """Her yellow dot on the minimap, in game-window pixels (full size), or None if hidden."""
    x0, y0, x1, y1 = MINIMAP_BOX
    return dot_in_box(full[y0:y1, x0:x1])


def dot_in_box(box: np.ndarray) -> tuple[float, float] | None:
    """Her dot in a capture of just MINIMAP_BOX (RGB), in game-window pixels: the biggest yellow blob of
    a dot's size. Partly under another player's marker, from the side of it that shows (DOT_UNCOVER)."""
    x0, y0 = MINIMAP_BOX[:2]
    area = box.astype(np.int16)
    r, g, b = area[..., 0], area[..., 1], area[..., 2]
    yellow = ((r > 220) & (g > 190) & (b < 90)).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(yellow)
    blobs = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, count)
             if DOT_AREA[0] <= stats[i, cv2.CC_STAT_AREA] <= DOT_AREA[1]
             and stats[i, cv2.CC_STAT_WIDTH] <= DOT_SIZE and stats[i, cv2.CC_STAT_HEIGHT] <= DOT_SIZE]
    if not blobs:
        return None
    _, i = max(blobs, key=lambda blob: blob[0])
    cx, cy = float(centroids[i][0]), float(centroids[i][1])
    if DOT_UNCOVER and (stats[i, cv2.CC_STAT_WIDTH] < DOT_FULL or stats[i, cv2.CC_STAT_HEIGHT] < DOT_FULL):
        cx, cy = _uncovered(box, stats[i], cx, cy)
    return cx + x0, cy + y0


def _uncovered(box: np.ndarray, st: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """Her dot's middle when its yellow blob (stats `st`, middle (cx, cy), box px) is cut short across or up and down:
    with another player's marker touching it on one side, from its edge on the other, as if it were DOT_FULL px. A
    yellow cut short with no marker by it (the minimap's edge) stays as it is."""
    left, top, w, h = (int(v) for v in st[:4])
    pad = DOT_SIZE
    wx, wy = max(0, left - pad), max(0, top - pad)
    near = markers_in_box(box[wy:top + h + pad, wx:left + w + pad])
    touching = [(m.x + wx, m.y + wy) for m in near if abs(m.x + wx - cx) <= 2 * DOT_FULL and abs(m.y + wy - cy) <= 2 * DOT_FULL]
    if not touching:
        return cx, cy
    mx, my = min(touching, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    half = (DOT_FULL - 0.5) / 2  # a full dot's middle from its first and last px (7-8 px: 3-3.5)
    if w < DOT_FULL and abs(mx - cx) >= abs(my - cy) / 2:
        cx = left + half if mx > cx else left + w - 1 - half
    if h < DOT_FULL and abs(my - cy) >= abs(mx - cx) / 2:
        cy = top + half if my > cy else top + h - 1 - half
    return cx, cy


def other_dots(full: np.ndarray) -> list[Marker]:
    """Other players' dots on the minimap (full-size px), in a capture of the whole game window."""
    x0, y0, x1, y1 = MINIMAP_BOX
    return [Marker(m.kind, m.x + x0, m.y + y0) for m in markers_in_box(full[y0:y1, x0:x1])]


def markers_in_box(box: np.ndarray) -> list[Marker]:
    """Other players' markers in a picture of (part of) the minimap (RGB), in its px: a core of a kind's colour
    (MARKER_COLOURS), of a core's size (MARKER_AREA, MARKER_SIDE), in a dark grey ring (RING_DARK). Two markers
    5 game px apart or more (57 map units) are two; 4 px apart, a third of the time; nearer, the one drawn last hides
    most of the other's core, and they're one."""
    if box.size == 0:
        return []
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
            if not (MARKER_AREA[0] <= a <= MARKER_AREA[1] and MARKER_SIDE[0] <= w <= MARKER_SIDE[1]
                    and MARKER_SIDE[0] <= h <= MARKER_SIDE[1]):
                continue
            cx, cy = float(centroids[i][0]) + x0, float(centroids[i][1]) + y0
            if _ringed(box, cx, cy):
                out.append(Marker(kind, cx, cy))
    return sorted(out, key=lambda m: (m.y, m.x))


def _ringed(box: np.ndarray, cx: float, cy: float) -> bool:
    """Whether a core at (cx, cy) has a marker's dark grey ring around it (RING_DARK of the directions)."""
    h, w = box.shape[:2]
    xs = np.clip(np.rint(cx + _RING[..., 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(cy + _RING[..., 1]).astype(int), 0, h - 1)
    px = box[ys, xs].astype(np.int16)  # directions x radii x RGB
    hi, lo = px.max(axis=2), px.min(axis=2)
    dark = (hi < 130) & (hi - lo < 50)
    return dark.any(axis=1).mean() >= RING_DARK
