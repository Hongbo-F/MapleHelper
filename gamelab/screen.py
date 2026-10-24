"""Screen geometry every layer shares, from the game window's client size: how the game's own view shows in it,
the frames looks work on, and how map units show on screen.

The game lays out a 1366x768 view (GAME_VIEW) whatever its window's size, one game px to a map unit, and stretches
it to the window's client area. Until 2026-10-23 that was 1920x1080 on this PC's 4K screen, the view stretched
1.40625 times, and every template, calibration and pixel constant was made on that stretched picture. Since then
the game runs at its own 1366x768 (Options, Graphics, Resolution, windowed): drawn pixel for pixel. A `Geometry`
says which, from the client size of the frame in hand (`of_frame`), and goes with what is read from that frame
(perception.eyes.Observation.geo), so nothing assumes a size: px measured on the stretched window are converted
with `px` and `fpx`, and are exactly what they were at 1920x1080.

STRETCHED is the 1920x1080 window. The simulator draws its world at that size, and the saved screenshots in
gamelab/lab are of it; GAME_SIZE, SCREEN_SCALE, SCALE and UNITS_PER_HALF_PX are its numbers, for those alone
(tests/test_geometry.py checks that nothing that reads the game uses them, leaves a `geo` at STRETCHED by default,
makes numbers of its own from STRETCHED's, or uses a *_PX constant unconverted, but where it says why).
"""

from dataclasses import dataclass

GAME_VIEW = (1366, 768)      # the view the game lays out, in game px; one game px is one map unit
STRETCHED_CLIENT = (1920, 1080)  # the client size the templates and constants were made at until 2026-10-23
# How much a look shrinks the client area's picture, by client size (any other size: 2).
# - At 1920x1080, half size (960x540, 0.70 px per map unit): plenty for the monsters and drops, a quarter of the
#   pixels (the name tag is matched at full size).
# - At the game's own 1366x768, full size: 1366x768 is about twice 960x540's pixels at 1.42 times the detail, every
#   pixel the game's own, and half size (683x384) is half 960x540's pixels at 0.71 of its detail. Measured on the
#   saved frames (README, Screen size).
LOOK_SCALES = {STRETCHED_CLIENT: 2, GAME_VIEW: 1}
HUD_TOP = 0.88               # the 1920x1080 window's bottom bar starts this far down: monsters and drops are looked
                             # for above it (Geometry.hud_top; as it always was there)
# Where the HUD (the chat box and the bars under it) starts at the game's own size, in game px from the top: its border
# is row 700 (dark, 8 a channel, from x 283 to 1080; row 701 white) on every 1366x768 frame of both maps of 2026-10-23
# and 24, and at 1920x1080 at screen row 984-985 (700.2 game px). HUD_TOP's 0.88 is game row 675.8: at 1366x768 the
# 25 rows above the chat box, where Tree Dungeon's platforms 31 and 32 show, were never searched.
HUD_TOP_GAME = 700


@dataclass(frozen=True)
class Geometry:
    """A game window's client area and the look frames taken of it. Screen px are the client area's own (full
    size); frame px are a look's (the client area shrunk `look_scale` times)."""
    client: tuple[int, int]   # the client area, screen px (width, height)
    look_scale: int = 2       # a look's frame is the client area shrunk this much

    @property
    def stretch(self) -> float:
        """Screen px per game px (per map unit): the client's height over the view's (1.40625 at 1920x1080).
        Across it's `stretch_x`, the same to 0.05%."""
        return self.client[1] / GAME_VIEW[1]

    @property
    def stretch_x(self) -> float:
        """Screen px per game px across (1920/1366 = 1.4056 at 1920x1080: perception.exp_number goes by it)."""
        return self.client[0] / GAME_VIEW[0]

    @property
    def frame_scale(self) -> float:
        """A look frame's px per map unit (0.703 at 1920x1080 half size; 1 at 1366x768 full size)."""
        return self.stretch / self.look_scale

    @property
    def units_per_px(self) -> float:
        """Map units per look-frame px."""
        return self.look_scale / self.stretch

    @property
    def frame_size(self) -> tuple[int, int]:
        """A look frame's (width, height)."""
        return self.client[0] // self.look_scale, self.client[1] // self.look_scale

    @property
    def hud_top(self) -> float:
        """Where the HUD starts, as a share of the window's (and a look frame's) height: monsters and drops are
        looked for above it. At 1920x1080 HUD_TOP, as it always was there (the simulator draws its world at that
        size, and the scenes it records go by it); at any other size the chat box's top, HUD_TOP_GAME (700/768)."""
        return HUD_TOP if self.client == STRETCHED_CLIENT else HUD_TOP_GAME / GAME_VIEW[1]

    @property
    def native(self) -> bool:
        """Whether the game is drawn pixel for pixel (its own 1366x768)."""
        return self.client == GAME_VIEW

    @property
    def suffix(self) -> str:
        """What the files made for this client size are told apart by: "" at 1920x1080 (the files as they always
        were), else "_<width>x<height>"."""
        return "" if self.client == STRETCHED_CLIENT else f"_{self.client[0]}x{self.client[1]}"

    def px(self, n: float) -> float:
        """`n` screen px of the 1920x1080 window, in this one's screen px (exactly `n` at 1920x1080)."""
        return n * (self.stretch / _STRETCH_1920)

    def fpx(self, n: float) -> float:
        """`n` px of a half-size look frame of the 1920x1080 window, in this geometry's look frames (exactly `n` at
        1920x1080 half size)."""
        return n * (self.frame_scale / _FRAME_1920)

    def ipx(self, n: float) -> int:
        """`px`, rounded to a whole px."""
        return int(round(self.px(n)))

    def ifpx(self, n: float) -> int:
        """`fpx`, rounded to a whole px."""
        return int(round(self.fpx(n)))

    def box(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """A box (x0, y0, x1, y1) of the 1920x1080 window's screen px, in this one's (each edge rounded)."""
        return tuple(self.ipx(v) for v in box)

    def to_game(self, p: float) -> float:
        """A place on screen (screen px, a px's middle at its index) as a place in the game's own picture (game px):
        the stretch maps a game px's middle (g + 0.5) * stretch - 0.5 on screen (bilinear filtering)."""
        return (p + 0.5) / self.stretch - 0.5

    def from_game(self, g: float) -> float:
        """A place in the game's own picture (game px) on screen (to_game the other way)."""
        return (g + 0.5) * self.stretch - 0.5


_STRETCH_1920 = STRETCHED_CLIENT[1] / GAME_VIEW[1]
_FRAME_1920 = _STRETCH_1920 / 2


def geometry(width: int, height: int, look_scale: int | None = None) -> Geometry:
    """The geometry of a client area this size, its looks shrunk `look_scale` times (by default LOOK_SCALES')."""
    size = (int(width), int(height))
    return Geometry(size, LOOK_SCALES.get(size, 2) if look_scale is None else look_scale)


def of_frame(full, look_scale: int | None = None) -> Geometry:
    """The geometry of a full-size frame (a picture of the whole client area: height x width x ...)."""
    return geometry(full.shape[1], full.shape[0], look_scale)


STRETCHED = geometry(*STRETCHED_CLIENT)
NATIVE = geometry(*GAME_VIEW)

# The 1920x1080 window's numbers (STRETCHED's), for the simulator, which draws its world at that size, and for the
# saved screenshots of it.
GAME_SIZE = STRETCHED.client
SCREEN_SCALE = STRETCHED.stretch
SCALE = STRETCHED.look_scale
UNITS_PER_HALF_PX = STRETCHED.units_per_px
