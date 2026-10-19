"""Where she is in map coordinates, from her minimap dot.

The minimap is the map drawn at 1/2^mag scale (mag 4 = 1/16), then shown 1.40625x larger like the
rest of the game at 1920x1080, so one minimap pixel is 16 / 1.40625 = 11.4 map units. The dot sits
above her feet, so the origin below is fitted to where her feet are: measured at map 10000012 by her
dot on the basin floor (y 575) and right after climbing the rope at x 734 onto platform 30 (y 275),
which agree within a pixel. `Localizer.anchor` refits it from any spot whose map position is known.

`Camera` places her on screen from her map position, for when her name tag is hidden.
"""

from dataclasses import dataclass

import numpy as np

from .mapdata import SCREEN_SCALE, MapModel, Platform
from .terrain import her_dot

# Screen px (full size) where map point (-centerX, -centerY) would put her dot. x refined against
# ropes in view (the dot read ~8 units right of the rope-based position on three platforms).
DOT_ORIGIN = (65.1, 91.3)


@dataclass
class Where:
    x: float                    # map units, her feet
    y: float
    platform: Platform | None   # the platform she's standing on (None: in the air, on a rope, unsure)
    dot: tuple[float, float]


class Localizer:
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

    def where(self, full: np.ndarray) -> Where | None:
        dot = her_dot(full)
        return None if dot is None else self.where_dot(dot)

    def where_dot(self, dot: tuple[float, float]) -> Where:
        x, y = self.to_map(dot)
        # The dot is only good to ~1 px (11 map units); snap to the platform she's most likely on.
        platform = self.model.platform_under(x, y - 25, reach=50)
        if platform is not None:
            x = min(max(x, platform.left), platform.right)
            y = platform.y_at(x)
        return Where(x, y, platform, dot)

    def anchor(self, dot: tuple[float, float], x: float, y: float) -> None:
        """Refit the origin from a dot seen while her feet were at a known map point."""
        self.origin = (dot[0] - (x + self.centre[0]) / self.per_px, dot[1] - (y + self.centre[1]) / self.per_px)


VIEW = (1920 / SCREEN_SCALE, 1080 / SCREEN_SCALE)  # map units on screen: 1365 x 768


class Camera:
    """Which part of the map is on screen, so she can be placed on screen from her map position when
    her name tag can't be seen (e.g. behind the party HP box at the map's edge).

    The camera keeps her at a fixed spot on screen, except that it never shows past the map's edges
    (its VR bounds): near an edge it stops, and she walks towards the side of the screen. `offset` is
    where she stands on screen (map units from its top-left corner), learned from looks where both her
    name tag and her map position are known.
    """

    def __init__(self, bounds: tuple[float, float, float, float], offset: tuple[float, float] = (674.0, 489.0)):
        self.bounds = bounds
        self.offset = offset

    def corner(self, x: float, y: float) -> tuple[float, float]:
        """Map point at the screen's top-left corner while her feet are at (x, y)."""
        left, top, right, bottom = self.bounds
        return (min(max(x - self.offset[0], left), right - VIEW[0]),
                min(max(y - self.offset[1], top), bottom - VIEW[1]))

    def clamped(self, x: float, y: float) -> tuple[bool, bool]:
        cx, cy = self.corner(x, y)
        return abs(cx - (x - self.offset[0])) > 0.5, abs(cy - (y - self.offset[1])) > 0.5

    def to_screen(self, x: float, y: float, at: tuple[float, float]) -> tuple[float, float]:
        """Full-size screen px of map point (x, y) while her feet are at `at`."""
        cx, cy = self.corner(*at)
        return (x - cx) * SCREEN_SCALE, (y - cy) * SCREEN_SCALE

    def learn(self, at: tuple[float, float], screen: tuple[float, float], rate: float = 0.2) -> None:
        """Her feet at map `at` were seen at full-size screen px `screen`: nudge the offset towards it
        (in each direction the camera isn't stopped at an edge)."""
        cx, cy = self.clamped(*at)
        ox, oy = self.offset
        if not cx:
            ox += rate * (screen[0] / SCREEN_SCALE - ox)
        if not cy:
            oy += rate * (screen[1] / SCREEN_SCALE - oy)
        self.offset = (ox, oy)
