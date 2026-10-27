"""Where she is in map coordinates, from her minimap dot (`Localizer`), refined by what's on screen
(`refine`: a rope in view or the visible end of her platform).

The minimap is the map drawn at 1/2^mag scale (mag 4 = 1/16), then shown 1.40625x larger like the
rest of the game at 1920x1080, so one minimap pixel is 16 / 1.40625 = 11.4 map units. The dot sits
above her feet, so the origin below is fitted to where her feet are: measured at map 10000012 by her
dot on the basin floor (y 575) and right after climbing the rope at x 734 onto platform 30 (y 275),
which agree within a pixel. `Localizer.anchor` refits it from any spot whose map position is known.

`Camera` places her on screen from her map position, for when her name tag is hidden.

Screen px go by the look's geometry (screen.Geometry): at the game's own 1366x768 the minimap is drawn pixel for
pixel, one minimap pixel 16 map units, and DOT_ORIGIN (fitted at 1920x1080) is where the same game px is at that
size (`Localizer.use`).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..screen import STRETCHED, Geometry
from .mapdata import MapModel, Platform

if TYPE_CHECKING:
    from .scene import Scene


# Screen px (full size, 1920x1080) where map point (-centerX, -centerY) would put her dot. x refined against
# ropes in view (the dot read ~8 units right of the rope-based position on three platforms).
DOT_ORIGIN = (65.1, 91.3)
# The dot snaps to a platform up to ~25 units above or below it: the closest one at or below a point
# DOT_SNAP_LIFT above the dot's map position, within DOT_SNAP_REACH.
DOT_SNAP_LIFT = 25
DOT_SNAP_REACH = 50
# ... but not by a rope that reaches up to that platform (hanging from it, ROPE_TOP_DY: motion.route's): there her dot
# more than ROPE_TOP_GAP under its surface, within ROPE_TOP_X of the rope, is her on the rope (no platform), unless it's
# where she'd stand on another platform. Snapped, the top 30-31 units of every rope hanging from a platform on the three
# maps read her standing on it (D:/MapleHelper-scratch/rope/diag/snap_zones.py). Live on 2026-10-26 (Tree Dungeon,
# Forest Up North II, 1366x768) a touch near the top of the rope at x 111 (19 -> 18) cut her climb and let go of up;
# the game keeps her on a rope when touched (11 of 11 hits there), so she hung about 30 under 18's surface, her dot at
# (101.3, -861.1), 31 under it, and was placed on 18 at x 113.3: her Lucky Seven presses refused there ("That cannot be
# used while you're on a rope or ladder.") ended the hunt for out of darts three times that night (22:58:32, 23:11:49,
# 23:12:42; behavior.darts). Over 2,046 chain starts of the tour's telemetry (10-24 to 10-26) her dot was a median 4.7
# and at most 14.0 under the platform she stood on (8 between 12.5 and 14.0), never more by a rope's top; at a rope's
# top she hung at 14.8 (22:19, 9 chain starts), 19.3, 22.2 and 30.8: so this catches 5 of the 6 hangs
# (control.navigator's check of a move her dot never moved on is for the rest). Her dot's x read 101.6 hanging at x
# 111 (one minimap px is 16 units at 1366x768).
ROPE_TOP_X = 16      # map units
ROPE_TOP_GAP = 17    # map units
ROPE_TOP_DY = 20     # map units (motion.route.ROPE_TOP_DY)
STAND_ABOVE = 8      # a platform this far under her dot (mapdata.FEET_BELOW) or ROPE_TOP_GAP over it she stands on

# Map units on screen: the game lays out the same view at any window size (1366 x 768 at 1366x768; by the height's
# stretch, which the camera's numbers were fitted with, 1365.3 x 768 at 1920x1080).
VIEW = (STRETCHED.client[0] / STRETCHED.stretch, STRETCHED.client[1] / STRETCHED.stretch)
CAMERA_OFFSET = (674.0, 489.0)  # where the camera keeps her: map units from the screen's top-left corner
CAMERA_LEARN_RATE = 0.2         # how far each look nudges that towards where she was seen
CAMERA_STOPPED = 0.5            # map units the camera may lag where it would keep her before it counts as
                                # stopped at an edge

# Screen px (1920x1080) the drawn ground sticks out past a platform's real end. Negative: measured against ropes
# on platforms 32, 30 and 14, the real ends are 21-27 px beyond where the grass seems to end.
GRASS_OVERHANG = -24
ROPE_MATCH_PX = 40   # full-size px (1920x1080) a rope on screen may be from where the dot puts a map rope and be it
SCREEN_EDGE_PX = 12  # full-size px (1920x1080): a platform end this close to the screen's side may be cut off
DOT_AGREE = 45       # map units a screen-based x may be from the dot's and still be believed


@dataclass
class Where:
    """Her position on the map."""
    x: float                    # map units, her feet
    y: float
    platform: Platform | None   # the platform she's standing on (None: in the air, on a rope, unsure)
    dot: tuple[float, float]    # the minimap dot it came from (full-size px)


class Localizer:
    """Her map position from her minimap dot, and back, in a game window of geometry `geo` (`use` changes it)."""

    def __init__(self, model: MapModel, origin: tuple[float, float] = DOT_ORIGIN, geo: Geometry = STRETCHED):
        self.model = model
        self.origin = origin   # screen px of `geo`'s (DOT_ORIGIN is 1920x1080's)
        self.geo = STRETCHED
        mm = model.minimap
        self.mag = 2 ** mm["mag"]
        self.centre = (mm["centerX"], mm["centerY"])
        self.use(geo)

    @property
    def per_px(self) -> float:
        """Map units per minimap pixel (11.4 at 1920x1080, 16 at 1366x768 for a minimap at 1/16)."""
        return self.mag / self.geo.stretch

    def use(self, geo: Geometry) -> None:
        """Dots from now on are in a game window of geometry `geo`: the origin moves to the same game px there."""
        if geo != self.geo:
            self.origin = tuple(geo.from_game(self.geo.to_game(v)) for v in self.origin)
            self.geo = geo

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
        if platform is not None and self.on_rope_under(platform, x, y):
            return Where(x, y, None, dot)  # (on the rope: as anywhere off the platforms, the dot's own place)
        if platform is not None:
            x = platform.clamp(x)
            y = platform.y_at(x)
        return Where(x, y, platform, dot)

    def on_rope_under(self, p: Platform, x: float, y: float) -> bool:
        """Whether her dot at map (x, y), which snaps onto platform `p`, has her on a rope under it instead: more than
        ROPE_TOP_GAP under its surface, within ROPE_TOP_X of a rope reaching up to it (its top at most ROPE_TOP_DY under
        the surface) and down past her, and not where she'd stand on another platform (STAND_ABOVE)."""
        if y - p.y_at(p.clamp(x)) <= ROPE_TOP_GAP:
            return False
        if not any(abs(r.x - x) <= ROPE_TOP_X and p.covers(r.x, 4) and r.top <= p.y_at(r.x) + ROPE_TOP_DY
                   and y <= r.bottom for r in self.model.ropes):
            return False
        return not any(q is not p and q.covers(x, 15) and -ROPE_TOP_GAP <= q.y_at(x) - y <= STAND_ABOVE
                       for q in self.model.places)

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
        """Map point at the screen's top-left corner while her feet are at (x, y). Across (or up and down) a map no
        wider (taller) than the view, the camera doesn't move: the map is centred on screen (Tree Dungeon, Forest Up
        North II, 770 wide: its leaves' ends on screen put the corner at -658, the middle of -360..410 less half the
        view's 1366; stopped at the right edge, as before, it would have been -955)."""
        left, top, right, bottom = self.bounds
        return (_axis(x - self.offset[0], left, right, VIEW[0]), _axis(y - self.offset[1], top, bottom, VIEW[1]))

    def fixed(self) -> tuple[bool, bool]:
        """Whether the camera can't move at all across, and up and down: the map is no wider (taller) than the view.
        Then her name tag on screen places her on the map exactly (camera_x)."""
        left, top, right, bottom = self.bounds
        return right - left <= VIEW[0], bottom - top <= VIEW[1]

    def clamped(self, x: float, y: float) -> tuple[bool, bool]:
        """Whether the camera is stopped at an edge, across and up/down, while her feet are at (x, y)."""
        cx, cy = self.corner(x, y)
        return abs(cx - (x - self.offset[0])) > CAMERA_STOPPED, abs(cy - (y - self.offset[1])) > CAMERA_STOPPED

    def to_screen(self, x: float, y: float, at: tuple[float, float],
                  geo: Geometry = STRETCHED) -> tuple[float, float]:
        """Full-size screen px (of a game window of geometry `geo`) of map point (x, y) while her feet are at `at`."""
        cx, cy = self.corner(*at)
        return (x - cx) * geo.stretch, (y - cy) * geo.stretch

    def learn(self, at: tuple[float, float], screen: tuple[float, float], rate: float = CAMERA_LEARN_RATE,
              geo: Geometry = STRETCHED) -> None:
        """Her feet at map `at` were seen at full-size screen px `screen` (of a game window of geometry `geo`): nudge
        the offset towards it (in each direction the camera isn't stopped at an edge)."""
        cx, cy = self.clamped(*at)
        ox, oy = self.offset
        if not cx:
            ox += rate * (screen[0] / geo.stretch - ox)
        if not cy:
            oy += rate * (screen[1] / geo.stretch - oy)
        self.offset = (ox, oy)


def _axis(follow: float, lo: float, hi: float, view: float) -> float:
    """The camera's corner along one axis: following her (`follow`) between the map's edges, or the map centred if
    it's no bigger than the view."""
    if hi - lo <= view:
        return (lo + hi) / 2 - view / 2
    return min(max(follow, lo), hi - view)


def camera_x(where: Where, s: "Scene", camera: "Camera | None") -> float | None:
    """Her x from her name tag on screen, where the camera can't move across (Camera.fixed): the map's x at her tag's
    middle. Exact to a px (1 map unit at 1366x768), where the platform's visible end drifts with what's drawn over it:
    standing still in the exit portal's light on Tree Dungeon, Forest Up North II (2026-10-24), the end read her x
    anywhere from 265 to 293 about every second, her tag 264.3 on all 20 saved frames (her dot: 262)."""
    if camera is None or s.tag is None or not camera.fixed()[0]:
        return None
    geo = s.geo
    return camera.corner(where.x, where.y)[0] + geo.to_game(geo.look_scale * s.tag[0])


def rope_x(where: Where, s: "Scene", model: MapModel) -> float | None:
    """Her x from ropes in view: each is drawn exactly at its map x, and detected to ~2 px."""
    if s.tag is None or not s.ropes:
        return None
    geo = s.geo
    her_x, match = geo.look_scale * s.tag[0], geo.px(ROPE_MATCH_PX)
    estimates = []
    for seen in s.ropes:
        for rope in model.ropes:
            expected = her_x + (rope.x - where.x) * geo.stretch  # where the dot says it should be
            if abs(seen.x - expected) <= match:
                estimates.append(rope.x - (seen.x - her_x) / geo.stretch)
    if not estimates:
        return None
    estimates.sort()
    return estimates[len(estimates) // 2]


def edge_x(where: Where, s: "Scene") -> float | None:
    """Her x from the visible end(s) of her platform (the average of those that agree with the dot)."""
    p = where.platform
    if p is None or s.tag is None or s.ground is None:
        return None
    geo = s.geo
    k, grass, edge = geo.look_scale, geo.px(GRASS_OVERHANG), geo.px(SCREEN_EDGE_PX)
    her_x = k * s.tag[0]
    left, right = k * s.ground[0] + grass, k * s.ground[1] - grass
    width = s.frame.shape[1] * k
    candidates = []
    if right < width - edge - grass:  # her platform's right end is on screen
        candidates.append(p.right - (right - her_x) / geo.stretch)
    if left > edge + grass:
        candidates.append(p.left + (her_x - left) / geo.stretch)
    good = [x for x in candidates if abs(x - where.x) <= DOT_AGREE]  # must agree with the dot roughly
    return sum(good) / len(good) if good else None


def refine(where: Where, s: "Scene", model: MapModel, camera: "Camera | None" = None) -> tuple[Where, str]:
    """Her x along her platform, best source first: her name tag where the camera can't move across (camera_x), a
    rope in view (ropes are drawn exactly at their map x), the visible end of her platform, else the minimap dot (only
    good to ~11 map units). The screen sources need her name tag."""
    p = where.platform
    if p is None or s.tag is None:
        return where, "dot"
    for how, x in (("camera", camera_x(where, s, camera)), ("rope", rope_x(where, s, model)),
                   ("edge", edge_x(where, s))):
        if x is not None and abs(x - where.x) <= DOT_AGREE:
            x = p.clamp(x)
            return Where(x, p.y_at(x), p, where.dot), how
    return where, "dot"
