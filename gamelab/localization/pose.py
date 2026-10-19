"""Where she is in map coordinates, from her minimap dot (`Localizer`), refined by what's on screen
(`refine`: a rope in view or the visible end of her platform).

The minimap is the map drawn at 1/2^mag scale (mag 4 = 1/16), then shown 1.40625x larger like the
rest of the game at 1920x1080, so one minimap pixel is 16 / 1.40625 = 11.4 map units. The dot sits
above her feet, so the origin below is fitted to where her feet are: measured at map 10000012 by her
dot on the basin floor (y 575) and right after climbing the rope at x 734 onto platform 30 (y 275),
which agree within a pixel. `Localizer.anchor` refits it from any spot whose map position is known.

`Camera` places her on screen from her map position, for when her name tag is hidden.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..screen import GAME_SIZE, SCALE, SCREEN_SCALE
from .mapdata import MapModel, Platform

if TYPE_CHECKING:
    from .scene import Scene


# Screen px (full size) where map point (-centerX, -centerY) would put her dot. x refined against
# ropes in view (the dot read ~8 units right of the rope-based position on three platforms).
DOT_ORIGIN = (65.1, 91.3)
# The dot snaps to a platform up to ~25 units above or below it: the closest one at or below a point
# DOT_SNAP_LIFT above the dot's map position, within DOT_SNAP_REACH.
DOT_SNAP_LIFT = 25
DOT_SNAP_REACH = 50

VIEW = (GAME_SIZE[0] / SCREEN_SCALE, GAME_SIZE[1] / SCREEN_SCALE)  # map units on screen: 1365 x 768
CAMERA_OFFSET = (674.0, 489.0)  # where the camera keeps her: map units from the screen's top-left corner
CAMERA_LEARN_RATE = 0.2         # how far each look nudges that towards where she was seen
CAMERA_STOPPED = 0.5            # map units the camera may lag where it would keep her before it counts as
                                # stopped at an edge

# Screen px the drawn ground sticks out past a platform's real end. Negative: measured against ropes
# on platforms 32, 30 and 14, the real ends are 21-27 px beyond where the grass seems to end.
GRASS_OVERHANG = -24
ROPE_MATCH_PX = 40   # full-size px a rope on screen may be from where the dot puts a map rope and be that one
SCREEN_EDGE_PX = 12  # full-size px: a platform end this close to the screen's side may be cut off, not its end
DOT_AGREE = 45       # map units a screen-based x may be from the dot's and still be believed


@dataclass
class Where:
    """Her position on the map."""
    x: float                    # map units, her feet
    y: float
    platform: Platform | None   # the platform she's standing on (None: in the air, on a rope, unsure)
    dot: tuple[float, float]    # the minimap dot it came from (full-size px)


class Localizer:
    """Her map position from her minimap dot, and back."""

    def __init__(self, model: MapModel, origin: tuple[float, float] = DOT_ORIGIN):
        self.model = model
        self.origin = origin
        mm = model.minimap
        self.per_px = (2 ** mm["mag"]) / SCREEN_SCALE   # map units per minimap pixel
        self.centre = (mm["centerX"], mm["centerY"])

    def to_map(self, dot: tuple[float, float]) -> tuple[float, float]:
        return ((dot[0] - self.origin[0]) * self.per_px - self.centre[0],
                (dot[1] - self.origin[1]) * self.per_px - self.centre[1])

    def to_dot(self, x: float, y: float) -> tuple[float, float]:
        return (self.origin[0] + (x + self.centre[0]) / self.per_px,
                self.origin[1] + (y + self.centre[1]) / self.per_px)

    def where_dot(self, dot: tuple[float, float]) -> Where:
        x, y = self.to_map(dot)
        # The dot is only good to ~1 px (11 map units); snap to the platform she's most likely on.
        platform = self.model.platform_under(x, y - DOT_SNAP_LIFT, reach=DOT_SNAP_REACH)
        if platform is not None:
            x = platform.clamp(x)
            y = platform.y_at(x)
        return Where(x, y, platform, dot)

    def anchor(self, dot: tuple[float, float], x: float, y: float) -> None:
        """Refit the origin from a dot seen while her feet were at a known map point."""
        self.origin = (dot[0] - (x + self.centre[0]) / self.per_px, dot[1] - (y + self.centre[1]) / self.per_px)


class Camera:
    """Which part of the map is on screen, so she can be placed on screen from her map position when
    her name tag can't be seen (e.g. behind the party HP box at the map's edge).

    The camera keeps her at a fixed spot on screen, except that it never shows past the map's edges
    (its VR bounds): near an edge it stops, and she walks towards the side of the screen. `offset` is
    where she stands on screen (map units from its top-left corner), learned from looks where both her
    name tag and her map position are known.
    """

    def __init__(self, bounds: tuple[float, float, float, float], offset: tuple[float, float] = CAMERA_OFFSET):
        self.bounds = bounds
        self.offset = offset

    def corner(self, x: float, y: float) -> tuple[float, float]:
        """Map point at the screen's top-left corner while her feet are at (x, y)."""
        left, top, right, bottom = self.bounds
        return (min(max(x - self.offset[0], left), right - VIEW[0]),
                min(max(y - self.offset[1], top), bottom - VIEW[1]))

    def clamped(self, x: float, y: float) -> tuple[bool, bool]:
        """Whether the camera is stopped at an edge, across and up/down, while her feet are at (x, y)."""
        cx, cy = self.corner(x, y)
        return abs(cx - (x - self.offset[0])) > CAMERA_STOPPED, abs(cy - (y - self.offset[1])) > CAMERA_STOPPED

    def to_screen(self, x: float, y: float, at: tuple[float, float]) -> tuple[float, float]:
        """Full-size screen px of map point (x, y) while her feet are at `at`."""
        cx, cy = self.corner(*at)
        return (x - cx) * SCREEN_SCALE, (y - cy) * SCREEN_SCALE

    def learn(self, at: tuple[float, float], screen: tuple[float, float], rate: float = CAMERA_LEARN_RATE) -> None:
        """Her feet at map `at` were seen at full-size screen px `screen`: nudge the offset towards it
        (in each direction the camera isn't stopped at an edge)."""
        cx, cy = self.clamped(*at)
        ox, oy = self.offset
        if not cx:
            ox += rate * (screen[0] / SCREEN_SCALE - ox)
        if not cy:
            oy += rate * (screen[1] / SCREEN_SCALE - oy)
        self.offset = (ox, oy)


def rope_x(where: Where, s: "Scene", model: MapModel) -> float | None:
    """Her x from ropes in view: each is drawn exactly at its map x, and detected to ~2 px."""
    if s.tag is None or not s.ropes:
        return None
    her_x = SCALE * s.tag[0]
    estimates = []
    for seen in s.ropes:
        for rope in model.ropes:
            expected = her_x + (rope.x - where.x) * SCREEN_SCALE  # where the dot says it should be
            if abs(seen.x - expected) <= ROPE_MATCH_PX:
                estimates.append(rope.x - (seen.x - her_x) / SCREEN_SCALE)
    if not estimates:
        return None
    estimates.sort()
    return estimates[len(estimates) // 2]


def edge_x(where: Where, s: "Scene") -> float | None:
    """Her x from the visible end(s) of her platform (the average of those that agree with the dot)."""
    p = where.platform
    if p is None or s.tag is None or s.ground is None:
        return None
    her_x = SCALE * s.tag[0]
    left, right = SCALE * s.ground[0] + GRASS_OVERHANG, SCALE * s.ground[1] - GRASS_OVERHANG
    width = s.frame.shape[1] * SCALE
    candidates = []
    if right < width - SCREEN_EDGE_PX - GRASS_OVERHANG:  # her platform's right end is on screen
        candidates.append(p.right - (right - her_x) / SCREEN_SCALE)
    if left > SCREEN_EDGE_PX + GRASS_OVERHANG:
        candidates.append(p.left + (her_x - left) / SCREEN_SCALE)
    good = [x for x in candidates if abs(x - where.x) <= DOT_AGREE]  # must agree with the dot roughly
    return sum(good) / len(good) if good else None


def refine(where: Where, s: "Scene", model: MapModel) -> tuple[Where, str]:
    """Her x along her platform, best source first: a rope in view (ropes are drawn exactly at their
    map x), the visible end of her platform, else the minimap dot (only good to ~11 map units).
    Both screen sources need her name tag."""
    p = where.platform
    if p is None or s.tag is None:
        return where, "dot"
    for how, x in (("rope", rope_x(where, s, model)), ("edge", edge_x(where, s))):
        if x is not None and abs(x - where.x) <= DOT_AGREE:
            x = p.clamp(x)
            return Where(x, p.y_at(x), p, where.dot), how
    return where, "dot"
