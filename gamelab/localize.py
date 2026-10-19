"""Where she is in map coordinates, from her minimap dot.

The minimap is the map drawn at 1/2^mag scale (mag 4 = 1/16), then shown 1.40625x larger like the
rest of the game at 1920x1080, so one minimap pixel is 16 / 1.40625 = 11.4 map units. The dot sits
above her feet, so the origin below is fitted to where her feet are: measured at map 10000012 by her
dot on the basin floor (y 575) and right after climbing the rope at x 734 onto platform 30 (y 275),
which agree within a pixel. `Localizer.anchor` refits it from any spot whose map position is known.
"""

from dataclasses import dataclass

import numpy as np

from .mapdata import SCREEN_SCALE, MapModel, Platform
from .terrain import her_dot

DOT_ORIGIN = (64.4, 91.3)  # screen px (full size) where map point (-centerX, -centerY) would put her dot


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
