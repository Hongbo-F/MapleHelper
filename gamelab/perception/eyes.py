"""Looking at the game: a capture of the game window turned into what's on screen, in look-frame pixels,
without the player (screen capture needs no admin rights).

Eyes looks at a game window of one client size (screen.Geometry: 1920x1080, its looks at half size, or the game's
own 1366x768, at full size), which its templates were made for, and every Observation says which (`geo`): what
reads it converts its px with it.

Eyes is perception's only door to pixels. `look` gives an Observation (her name tag, HP and MP, the
ground under her, her minimap dot and other players', ropes, her EXP number and the box it's read from);
localization then asks about that look's frame through Eyes (`monsters`, `items`, `bar`) instead of reading pixels
itself, so the simulator (gamelab.sim) can stand in for the game by answering the same questions.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .. import capture, paths, window
from ..screen import HUD_TOP, STRETCHED, Geometry
from .exp_number import read_box
from .hud import BAR_ROW, HP_BAR, bars, exp_number_box, hp_bar, hp_fill
from .loot import Item, LootDetector
from .minimap import Marker, MinimapPicture, dot_in_box, her_dot, minimap_box, other_dots
from .mobs import Mob, MobDetector
from .nametag import NameTag
from .terrain import ScreenRope, find_ropes, ground_extent


TAG_BELOW_FEET = 8  # half-size px (1920x1080) from the ground line she stands on down to her name tag's centre
EXP_TEXT = 150      # her EXP number's text pixels are brighter than this (in any channel)
# Glances start this far apart at least. With mss a glance was two grabs that each waited for the
# compositor's next frame (33 ms), so with the navigator's 30 ms between glances her dot was seen about
# every 63 ms, and what the navigator makes of glances in a row was set at that pace (her dot standing
# still, so cutting a plan off its path and `recover`); it's judged in time now (navigator.STILL_S,
# OFF_PATH_S), so it means the same at any pace. From the latest frame a glance takes 0.2 ms; 0: as often
# as asked (about every 30 ms in the navigator). Glances keep this pace without a feed: the simulator
# charges a glance mss's 35 ms (sim.fakes.GLANCE_S), so a faster one isn't tried there. Not tried in game.
GLANCE_EVERY_S = 0.063


def tag_below_feet(geo: Geometry = STRETCHED) -> int:
    """TAG_BELOW_FEET in a look frame of geometry `geo` (8 at 1920x1080 half size, 11 at 1366x768 full size)."""
    return geo.ifpx(TAG_BELOW_FEET)


def band_rows(obs: Observation, up: float, down: float) -> tuple[int, int]:
    """The rows of a look's frame from `up` map units above her feet to `down` below, above the bottom bar."""
    feet, bottom, per_px = obs.floor_y, int(obs.frame.shape[0] * HUD_TOP), obs.geo.units_per_px
    return max(0, int(feet - up / per_px)), min(bottom, int(feet + down / per_px))


def rgb_of(bgra: np.ndarray) -> np.ndarray:
    """A screen capture (BGRA, as mss gives it) as an RGB picture of its own. Viewed as RGB in place
    ([:, :, 2::-1]) it cost nothing here, but every OpenCV call on the view then copied it first: 17-20 ms
    a look in all, against about 1 ms for this."""
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)


@dataclass(kw_only=True)
class Observation:
    """What one look at the game shows (look-frame px unless noted)."""
    t: float                                 # when (time.monotonic)
    frame: np.ndarray                        # the game window shrunk geo.look_scale times (half size at 1920x1080), RGB
    tag: tuple[int, int] | None              # her name tag's centre; her feet are just above it
    hp: float                                # HP and MP left (0-1)
    mp: float
    ground: tuple[int, int] | None = None    # screen x range of the ground under her
    dot: tuple[float, float] | None = None   # her minimap dot (full-size px)
    others: list[Marker] = field(default_factory=list)  # other players' minimap dots (full-size px)
    ropes: list[ScreenRope] = field(default_factory=list)  # ropes and ladders on screen (full-size px)
    exp: np.ndarray | None = None            # the text pixels of her EXP number (EXP_NUMBER): changed, she gained EXP
    exp_box: np.ndarray | None = None        # the box her EXP number is read from (exp_number.read_box, full size,
                                             # RGB): read (exp_number.read) only when `exp` changes
    geo: Geometry = STRETCHED                # the game window's client size and the look's scale (the simulator's
                                             # frames are the 1920x1080 window's)

    @property
    def floor_y(self) -> int | None:
        """The ground line she stands on (her name tag hangs just below it)."""
        return None if self.tag is None else self.tag[1] - tag_below_feet(self.geo)


class Eyes:
    """The game window, looked at. `title`: the game window's title; `tag_template`: her name tag as calibration
    keeps it (calibration.load(geo).template). The monster detector defaults to the screenshot cut-outs; the loot
    detector is made on first use. `geo`: the game window's geometry, which the templates and the detectors given
    are for (screen.Geometry; by default 1920x1080's): a capture of another size is refused."""
    seen_at: float | None = None  # when the screen last captured looked like that (time.monotonic)
    geo: Geometry = STRETCHED     # the game window's geometry (a subclass that doesn't call __init__ has 1920x1080's)

    def __init__(self, title: str, tag_template: np.ndarray, detector: MobDetector | None = None,
                 loot: LootDetector | None = None, screen=None, geo: Geometry = STRETCHED,
                 minimap_picture: np.ndarray | None = None):
        """`screen`: what takes shots of the screen (by default the process's capture.Grabber: DXGI or mss,
        as config.toml says; analysis.bench passes saved frames). `minimap_picture`: the map's own minimap picture
        (mapdata.minimap_picture): with it, dots are read as if the picture were where it was on the first map
        (minimap.MinimapPicture), wherever the window puts or scrolls it."""
        self.title = title
        self.geo = geo
        self.detector = detector or MobDetector(frame_scale=geo.frame_scale)
        self._loot = loot
        self.screen = screen or capture.shared()
        self._glanced = -GLANCE_EVERY_S    # when the last glance began
        self.name_tag = NameTag(tag_template, paths.calibrated(paths.TAG_FULL, geo.suffix), geo)
        self.want_ropes = False  # Senses turns this on
        self.minimap = MinimapPicture(minimap_picture, geo) if minimap_picture is not None else None
        # Where ropes are looked for, from her minimap dot and her name tag's centre: ranges of x on screen
        # (full-size px; None: the whole screen). Senses works them out from the map's ropes.
        self.rope_columns: Callable[[tuple[float, float] | None, tuple[int, int] | None],
                                    list[tuple[int, int]] | None] | None = None

    def look(self) -> Observation:
        """HP/MP, her name tag, the ground under her, her minimap dot and other players'; no monsters (`monsters`)."""
        return self.look_at(self._capture())

    def look_at(self, full: np.ndarray, t: float | None = None, ropes: bool | None = None) -> Observation:
        """What a capture of the game window shows (full size, RGB), as `look` reads it: `t` is when the
        screen showed it (by default now, the end of the look); `ropes` False leaves the rope search out
        (by default `want_ropes`). The continuous feed (localization.feed) looks at frames it captured
        itself this way. The capture must be of the client size the eyes are for (`geo`)."""
        geo = self.geo
        if (full.shape[1], full.shape[0]) != geo.client:
            raise ValueError(f"a frame of {full.shape[1]}x{full.shape[0]}; these eyes are for "
                             f"{geo.client[0]}x{geo.client[1]}")
        hp, mp = bars(full)
        k = geo.look_scale
        frame = full if k == 1 else cv2.resize(full, (full.shape[1] // k, full.shape[0] // k),
                                               interpolation=cv2.INTER_AREA)
        tag = self.name_tag.find(full)
        ground = ground_extent(frame, tag[0], tag[1] - tag_below_feet(geo), geo) if tag else None
        dot = her_dot(full, geo)
        others = other_dots(full, geo)  # 0.4-0.6 ms (the 31 saved frames of this map)
        if self.minimap is not None:
            mx0, my0, mx1, my1 = minimap_box(geo)
            self.minimap.find(full[my0:my1, mx0:mx1])
            sx, sy = self.minimap.shift()
            dot, others = self._shifted(dot), [replace(m, x=m.x - sx, y=m.y - sy) for m in others]
        found = []
        if self.want_ropes if ropes is None else ropes:
            found = find_ropes(full, self.rope_columns(dot, tag) if self.rope_columns else None, geo)
        x0, y0, x1, y1 = exp_number_box(geo)
        bx0, by0, bx1, by1 = read_box(geo)
        return Observation(t=time.monotonic() if t is None else t, frame=frame, tag=tag, hp=hp, mp=mp,
                           ground=ground, dot=dot, others=others, ropes=found,
                           exp=full[y0:y1, x0:x1].max(axis=2) > EXP_TEXT, exp_box=full[by0:by1, bx0:bx1].copy(),
                           geo=geo)

    def glance(self) -> tuple[tuple[float, float] | None, float]:
        """Just her minimap dot and HP, from the newest frame (with DXGI 0.2 ms, with mss two grabs of 16.7
        ms each), GLANCE_EVERY_S after the last glance began at the soonest: for watching a plan run."""
        begun = time.monotonic()
        area = self._client_area()
        x0, y0, x1, y1 = minimap_box(self.geo)
        w, h = area["width"], area["height"]
        a, b = int(HP_BAR[0] * w), int(HP_BAR[1] * w)
        shot = self.screen.shot(area, at=self._glanced + GLANCE_EVERY_S)
        self._glanced = max(begun, self._glanced + GLANCE_EVERY_S)
        box, row = shot.crops([(x0, y0, x1 - x0, y1 - y0), (a, int(BAR_ROW * h), b - a, 1)], rgb_of)
        self.seen_at = shot.t
        if self.minimap is not None:
            self.minimap.find(box)
        return self._shifted(dot_in_box(box, self.geo)), hp_fill(row[0].astype(np.int16))

    def _shifted(self, dot: tuple[float, float] | None) -> tuple[float, float] | None:
        """A dot as it would be with the minimap's picture where it was on the first map (minimap.MinimapPicture)."""
        if dot is None or self.minimap is None:
            return dot
        sx, sy = self.minimap.shift()
        return dot[0] - sx, dot[1] - sy

    def messages(self) -> str:
        """The pickup log at the bottom right, read by OCR (~0.4 s): "Leather x1 earned. (Etc)", and what
        the game says when a pick-up fails. Empty if the OCR packages (environment.yml) aren't installed."""
        try:
            from maplehelper.ocr import LOOT, shared
        except ImportError:
            return ""
        begun = time.monotonic()
        area = self._client_area()
        x, y, w, h = LOOT.region.box((0, 0, area["width"], area["height"]))
        shot = self.screen.shot(area, after=begun).crop(x, y, w, h)
        rows = shared().read(np.ascontiguousarray(shot[:, :, :3]), min_height=LOOT.min_height) or []  # BGR
        return "\n".join(r.text for r in rows)

    # --- capturing the game window ------------------------------------------------------------------

    def _client_area(self) -> dict[str, int]:
        """The game window's client area on screen: left, top, width, height."""
        return window.client_rect(window.find_game(self.title))

    def _to_front(self) -> None:
        """Bring the game to the front unless it's there already: a capture sees whatever window is on top."""
        if window.foreground_title() != self.title:
            window.bring_to_front(self.title)
            time.sleep(0.3)

    def _capture(self) -> np.ndarray:
        """The whole client area, full size, RGB, with the game in front, as the screen showed it after
        this began (a look sees what came of the keys before it). It must be the size the templates were
        made for (`geo`)."""
        self._to_front()
        begun = time.monotonic()
        area = self._client_area()
        if (area["width"], area["height"]) != self.geo.client:
            raise RuntimeError(f"the game window is {area['width']}x{area['height']}; the monster templates and "
                               f"her name tag here are for {self.geo.client[0]}x{self.geo.client[1]} (it was that "
                               f"size when this began): run it again")
        shot = self.screen.shot(area, after=begun)
        full = shot.crop(0, 0, area["width"], area["height"], rgb_of)
        self.seen_at = shot.t
        return full

    # --- questions about a look's frame -----------------------------------------------------------

    def monsters(self, obs: Observation, up: float, down: float) -> list[Mob]:
        """Monsters from `up` map units above her feet to `down` below (her screen position must be known)."""
        return self.detector.detect(obs.frame, rows=band_rows(obs, up, down))

    def monsters_wide(self, obs: Observation) -> list[Mob]:
        """Every monster on screen (above the bottom bar)."""
        return self.detector.detect_wide(obs.frame, (0, int(obs.frame.shape[0] * HUD_TOP)))

    def refind(self, obs: Observation, mob: Mob, box: tuple[int, int, int, int], fallback: bool = True) -> Mob | None:
        """Monster `mob`, seen in an earlier frame, looked for again where its picture should be now (`box`:
        x0, y0, x1, y1), above the bottom bar (MobDetector.refind; without `fallback`, only with the template
        that matched it last)."""
        h, w = obs.frame.shape[:2]
        return self.detector.refind(obs.frame, box, mob.kind, mob.template, mob.facing, (0, 0, w, int(h * HUD_TOP)),
                                    fallback)

    def bar(self, obs: Observation, mob: Mob) -> float | None:
        """A monster's HP bar (the fraction left), or None if it hasn't been hit."""
        return hp_bar(obs.frame, mob.x, mob.y - mob.h // 2, obs.geo)

    def items(self, obs: Observation, box: tuple[int, int, int, int], mesos_only: bool = False) -> list[Item]:
        """Drops lying in a box (x0, y0, x1, y1) of the frame."""
        return self.loot_detector().detect(obs.frame, box, mesos_only=mesos_only)

    def items_in(self, obs: Observation, boxes: list[tuple[int, int, int, int]],
                 mesos_only: bool = False) -> list[list[Item]]:
        """`items` in each of several boxes, searched together."""
        return self.loot_detector().detect_each(obs.frame, boxes, mesos_only=mesos_only)

    @property
    def item_height(self) -> int:
        """The tallest drop picture (look-frame px): a box shorter than this can't hold one."""
        return self.loot_detector().height

    @property
    def kinds(self) -> set[str]:
        """The monster kinds the detector knows."""
        return set(self.detector.templates)

    def loot_detector(self) -> LootDetector:
        """The loot detector, made (from the game's item pictures, for frames of `geo`) on first use."""
        if self._loot is None:
            self._loot = LootDetector(geo=self.geo)
        return self._loot
