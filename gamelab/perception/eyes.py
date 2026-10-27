"""Looking at the game: a capture of the game window turned into what's on screen, in look-frame pixels,
without the player (screen capture needs no admin rights).

Eyes looks at a game window of one client size (screen.Geometry: 1920x1080, its looks at half size, or the game's
own 1366x768, at full size), which its templates were made for, and every Observation says which (`geo`): what
reads it converts its px with it.

Eyes is perception's only door to pixels. `look` gives an Observation (her name tag, HP and MP, the
ground under her, her minimap dot and other players', ropes, her EXP number and the box it's read from);
localization then asks about that look's frame through Eyes (`monsters`, `items`, `bar`, and which way she faces:
`heading`) instead of reading pixels itself, so the simulator (gamelab.sim) can stand in for the game by answering the
same questions.

A look is checked before anything is read from it (perception.framecheck): the game window where it was read to be,
in front, uncovered, on the screen and the size the eyes are for, its frame fresh and lined up with the client area,
and the map's minimap picture found. One that fails raises BadLook, saying why; `recover` then does what it can
(localization.senses takes it again, and gives up after a while). While the window is unfit in a way only the user sets
right (framecheck.WAIT_KINDS: another size, partly off the screen, gone), a look asks the window alone: the game isn't
brought to the front or captured. A glance that fails has no dot and keeps the last HP read.

A command's eyes take no shots of their own (`NoScreen`: one raises): the game is looked at all the time by the
perception service (localization.service), whose states every look reads, and what a command's eyes read off a whole
frame (the pet's panels, the pick-up log, her EXP number for `hunt look`) is its frame at full size (`full_frames`). The
service's own eyes, and the offline tools' on saved frames, have a screen to shoot.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .. import capture, paths, telemetry, window
from ..screen import STRETCHED, Geometry
from . import framecheck
from .exp_number import read_box
from .framecheck import BadLook, Problem
from .heading import Heading, Reading, her_box
from .hud import BAR_ROW, HP_BAR, bars, exp_number_box, hp_bar, hp_fill
from .loot import Item, LootDetector
from .messages import ChatLine, ChatReader, log_rows
from .minimap import PICTURE_MATCH, Marker, MinimapPicture, dot_in_box, her_dot, minimap_box, other_dots
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
# `recover`: while frames keep failing as the capture's own fault (framecheck.CAPTURE_KINDS), the duplication is opened
# afresh at the first and mss takes over at the second, then the two again in turn this often (mss lasts
# capture.RETRY_S, 10 s, a time). Before, only the first two failures of a run of any kind did anything, so a capture
# still lagging after a drag that had first failed as "moved" was never tried again (the reviewer's repro_attempts.py).
CAPTURE_AGAIN_S = 5.0


class NoScreen:
    """A command's screen: none. Every shot raises, so nothing in a command captures the game behind the perception
    service's back (its one DXGI duplication: a third failed with 0x887A0022 when watch, the pet's babysitter and a hunt
    each held their own)."""

    def shot(self, area, after: float | None = None, at: float | None = None):
        raise RuntimeError("a command's eyes take no shots: its frames are the perception service's "
                           "(localization.service; Eyes.full_frames)")


def tag_below_feet(geo: Geometry = STRETCHED) -> int:
    """TAG_BELOW_FEET in a look frame of geometry `geo` (8 at 1920x1080 half size, 11 at 1366x768 full size)."""
    return geo.ifpx(TAG_BELOW_FEET)


def band_rows(obs: Observation, up: float, down: float) -> tuple[int, int]:
    """The rows of a look's frame from `up` map units above her feet to `down` below, above the bottom bar."""
    feet, bottom, per_px = obs.floor_y, int(obs.frame.shape[0] * obs.geo.hud_top), obs.geo.units_per_px
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
    bad: Problem | None = None               # why this look isn't of the game as expected (framecheck; Senses: her
                                             # place off the map), nothing read from it then; None: it is (or
                                             # wasn't checked)
    picture: tuple[float, float, float] | None = None  # the map's minimap picture: its corner (full-size px) and how
                                                       # well it matched this look (None: no picture, or no search)

    @property
    def floor_y(self) -> int | None:
        """The ground line she stands on (her name tag hangs just below it)."""
        return None if self.tag is None else self.tag[1] - tag_below_feet(self.geo)


class Eyes:
    """The game window, looked at. `title`: the game window's title; `tag_template`: her name tag as calibration
    keeps it (calibration.load(geo).template). The monster detector defaults to the screenshot cut-outs; the loot
    detector is made on first use, for the map's drops (`drop_names`). `geo`: the game window's geometry, which the
    templates and the detectors given are for (screen.Geometry; by default 1920x1080's): a capture of another size is
    refused."""
    seen_at: float | None = None  # when the screen last captured looked like that (time.monotonic)
    geo: Geometry = STRETCHED     # the game window's geometry (a subclass that doesn't call __init__ has 1920x1080's)
    # The drops the loot detector looks for, by name (perception.drops.items_for: Senses tells it its map's); None: the
    # first map's (loot.ITEMS)
    drop_names: tuple[str, ...] | None = None
    check_frames = True           # whether frames are checked (framecheck.frame_problem): not frames resampled from
                                  # another size (bench-look --shrink), whose 1-px anchors are blurred
    _hp, _mp = 1.0, 1.0           # the HP and MP the last frame that passed its checks read (a refused one's)

    def __init__(self, title: str, tag_template: np.ndarray, detector: MobDetector | None = None,
                 loot: LootDetector | None = None, screen=None, geo: Geometry = STRETCHED,
                 minimap_picture: np.ndarray | None = None):
        """`screen`: what takes shots of the screen (the perception service's capture.Grabber, DXGI or mss as
        config.toml says; analysis.bench's saved frames; by default none, NoScreen: a command's, whose frames come from
        the service, `full_frames`). `minimap_picture`: the map's own minimap picture
        (mapdata.minimap_picture): with it, dots are read as if the picture were where it was on the first map
        (minimap.MinimapPicture), wherever the window puts or scrolls it."""
        self.title = title
        self.geo = geo
        self.detector = detector or MobDetector(frame_scale=geo.frame_scale)
        self._loot = loot
        self.screen = screen if screen is not None else NoScreen()
        # A command's whole frames: the game window at full size (RGB) shown at a time or later (the perception
        # service's client's full_frame), raising BadLook if there's none; None: shot here (`screen`).
        self.full_frames: Callable[[float], np.ndarray] | None = None
        self._glanced = -GLANCE_EVERY_S    # when the last glance began
        self.area: dict[str, int] | None = None  # where the client area was at the last look that passed
        self._waited: Problem | None = None  # what the last look waited out without touching the window (WAIT_KINDS)
        self._fixes, self._fixed_at = 0, -1e18  # what `recover` did about the capture since a frame last passed, when
        self.log: Callable[[str], None] = print  # what the eyes say (the window moved; the capture reopened)
        self.glances_failed = 0            # glances that failed their checks (no dot from them)
        self.name_tag = NameTag(tag_template, paths.calibrated(paths.of().tag_full, geo.suffix), geo)
        self._heading: Heading | None = None  # her pictures (heading_bank: made on first use)
        self.want_ropes = False  # Senses turns this on
        self.minimap = MinimapPicture(minimap_picture, geo) if minimap_picture is not None else None
        # Where ropes are looked for, from her minimap dot and her name tag's centre: ranges of x on screen
        # (full-size px; None: the whole screen). Senses works them out from the map's ropes.
        self.rope_columns: Callable[[tuple[float, float] | None, tuple[int, int] | None],
                                    list[tuple[int, int]] | None] | None = None

    def look(self) -> Observation:
        """HP/MP, her name tag, the ground under her, her minimap dot and other players'; no monsters (`monsters`).
        Raises BadLook if it isn't of the game as expected (see the module's docstring)."""
        obs = self.look_at(self._capture())
        if obs.bad is not None:
            raise BadLook(obs.bad)
        return obs

    def look_at(self, full: np.ndarray, t: float | None = None, ropes: bool | None = None) -> Observation:
        """What a capture of the game window shows (full size, RGB), as `look` reads it: `t` is when the
        screen showed it (by default now, the end of the look); `ropes` False leaves the rope search out
        (by default `want_ropes`). The continuous feed (localization.feed) looks at frames it captured
        itself this way. The capture must be of the client size the eyes are for (`geo`). A frame that fails its
        checks (framecheck) gives an Observation with `bad` saying why, and nothing read from it but the bars."""
        geo = self.geo
        if (full.shape[1], full.shape[0]) != geo.client:
            raise ValueError(f"a frame of {full.shape[1]}x{full.shape[0]}; these eyes are for "
                             f"{geo.client[0]}x{geo.client[1]}")
        hp, mp = bars(full)
        k = geo.look_scale
        frame = full if k == 1 else cv2.resize(full, (full.shape[1] // k, full.shape[0] // k),
                                               interpolation=cv2.INTER_AREA)
        if self.check_frames and (bad := framecheck.frame_problem(full, geo)) is not None:  # before her name tag,
            return self._refused(t, frame, bad)                         # whose template is cut from the first look
        self._fixes = 0  # the capture is fine
        picture = None
        if self.minimap is not None:
            mx0, my0, mx1, my1 = minimap_box(geo)
            self.minimap.find(full[my0:my1, mx0:mx1])
            if self.minimap.corner is None:  # never found: her dot can't be read (its origin isn't known)
                score = "no search" if self.minimap.score is None else f"best match {self.minimap.score:.2f}"
                return self._refused(t, frame, Problem(
                    "picture", f"the minimap doesn't show this map's picture ({score}, {PICTURE_MATCH} needed): "
                               f"another map, or the minimap covered"))
            picture = (*self.minimap.corner, -1.0 if self.minimap.score is None else round(self.minimap.score, 3))
        try:
            tag = self.name_tag.find(full)
        except RuntimeError as e:  # her name tag's full-size template can't be cut: she isn't on screen
            return self._refused(t, frame, Problem("tag", str(e)))
        ground = ground_extent(frame, tag[0], tag[1] - tag_below_feet(geo), geo) if tag else None
        dot = her_dot(full, geo)
        others = other_dots(full, geo)  # 0.4-0.6 ms (the 31 saved frames of this map)
        if self.minimap is not None:
            sx, sy = self.minimap.shift()
            dot, others = self._shifted(dot), [replace(m, x=m.x - sx, y=m.y - sy) for m in others]
        found = []
        if self.want_ropes if ropes is None else ropes:
            found = find_ropes(full, self.rope_columns(dot, tag) if self.rope_columns else None, geo)
        x0, y0, x1, y1 = exp_number_box(geo)
        bx0, by0, bx1, by1 = read_box(geo)
        self._hp, self._mp = hp, mp
        return Observation(t=time.monotonic() if t is None else t, frame=frame, tag=tag, hp=hp, mp=mp,
                           ground=ground, dot=dot, others=others, ropes=found,
                           exp=full[y0:y1, x0:x1].max(axis=2) > EXP_TEXT, exp_box=full[by0:by1, bx0:bx1].copy(),
                           geo=geo, picture=picture)

    def _refused(self, t: float | None, frame: np.ndarray, bad: Problem) -> Observation:
        """A look that failed its checks: nothing read from it, its HP and MP the last good frame's (the bars of a frame
        off the client area are read off something else: 0% on the 16:48:45 frame, a hit to the feed's world step)."""
        return Observation(t=time.monotonic() if t is None else t, frame=frame, tag=None, hp=self._hp, mp=self._mp,
                           geo=self.geo, bad=bad)

    def glance(self) -> tuple[tuple[float, float] | None, float]:
        """Just her minimap dot and HP, from the newest frame (with DXGI 0.2 ms, with mss two grabs of 16.7
        ms each), GLANCE_EVERY_S after the last glance began at the soonest: for watching a plan run. One that
        fails its checks (the game window gone, minimized, another size, not in front or covered; the minimap
        window's title strip not where it should be: framecheck) has no dot, and the last HP read."""
        begun = time.monotonic()
        try:
            area = self._client_area()
            wrong = self._window_problem(area)
        except RuntimeError:  # the game window is gone
            area, wrong = None, True
        if wrong:
            self._glanced = max(begun, self._glanced + GLANCE_EVERY_S)
            self.glances_failed += 1
            return None, self._hp
        x0, y0, x1, y1 = minimap_box(self.geo)
        w, h = area["width"], area["height"]
        a, b = int(HP_BAR[0] * w), int(HP_BAR[1] * w)
        boxes = [(x0, y0, x1 - x0, y1 - y0), (a, int(BAR_ROW * h), b - a, 1)]
        if framecheck.checked(self.geo):
            boxes.append(framecheck.title_box())
        shot = self.screen.shot(area, at=self._glanced + GLANCE_EVERY_S)
        self._glanced = max(begun, self._glanced + GLANCE_EVERY_S)
        box, row, *title = shot.crops(boxes, rgb_of)
        self.seen_at = shot.t
        if title and framecheck.title_diff(title[0]) > framecheck.TITLE_DIFF:
            self.glances_failed += 1
            return None, self._hp
        if self.minimap is not None:
            self.minimap.find(box)
        self._hp = hp_fill(row[0].astype(np.int16))
        return self._shifted(dot_in_box(box, self.geo)), self._hp

    def _shifted(self, dot: tuple[float, float] | None) -> tuple[float, float] | None:
        """A dot as it would be with the minimap's picture where it was on the first map (minimap.MinimapPicture)."""
        if dot is None or self.minimap is None:
            return dot
        sx, sy = self.minimap.shift()
        return dot[0] - sx, dot[1] - sy

    def messages(self) -> str:
        """The pickup log at the bottom right, read by OCR (~0.4 s): "Leather x1 earned. (Etc)", and what
        the game says when a pick-up fails. Empty if the OCR packages (environment.yml) aren't installed. Its rows
        only (perception.messages.log_rows: right-aligned against the window's edge), not the party window's over
        it: at 22:58:23 on 2026-10-26 it read her party's names ("7BugBaby", the leader's star as a 7, and "RideBaby"),
        and no pickup line."""
        try:
            from maplehelper.ocr import LOOT, shared
        except ImportError:
            return ""
        begun = time.monotonic()
        if getattr(self, "full_frames", None) is not None:  # (a command's: the perception service's frame)
            full = self.full_frames(begun)
            width = full.shape[1]
            x, y, w, h = LOOT.region.box((0, 0, width, full.shape[0]))
            shot = full[y:y + h, x:x + w, ::-1]  # RGB -> BGR
        else:
            area = self._client_area()
            width = area["width"]
            x, y, w, h = LOOT.region.box((0, 0, width, area["height"]))
            shot = self.screen.shot(area, after=begun).crop(x, y, w, h)
        rows = shared().read(np.ascontiguousarray(shot[:, :, :3]), min_height=LOOT.min_height) or []  # BGR
        return "\n".join(r.text for r in log_rows(rows, width, x))

    def chat_line(self) -> ChatLine:
        """The message bar (the chat's one line above the HUD) as shown now: the game's own line on it, read by OCR
        when it's one not read before (perception.messages.ChatReader: its pink px tell it in under a millisecond, the
        OCR ~0.28 s), where the game says why it refused a key ("That cannot be used while you're on a rope or
        ladder."). No game line (or no OCR packages): its text is ""."""
        begun = time.monotonic()
        if getattr(self, "full_frames", None) is not None:  # (a command's: the perception service's frame)
            full = self.full_frames(begun)
        else:
            area = self._client_area()
            full = self.screen.shot(area, after=begun).crop(0, 0, area["width"], area["height"], rgb_of)
        if getattr(self, "chat", None) is None:
            self.chat = ChatReader()
        return self.chat.read(full, begun)

    # --- capturing the game window ------------------------------------------------------------------

    def _client_area(self) -> dict[str, int]:
        """The game window's client area on screen: left, top, width, height."""
        return window.client_rect(window.find_game(self.title))

    def _window_problem(self, area: dict[str, int]) -> Problem | None:
        """What's wrong with the game window for a capture of `area` (framecheck.window_problem), or None."""
        return framecheck.window_problem(window.find_game(self.title), area, self.title, self.geo)

    def window_wait(self) -> Problem | None:
        """What's wrong with the game window that only the user sets right (framecheck.WAIT_KINDS: gone, minimized,
        another size, partly off the screen), asked of the window alone: no capture, and the game not brought to the
        front; None if nothing is."""
        try:
            wrong = self._window_problem(self._client_area())
        except RuntimeError as e:  # no game window
            return Problem("gone", str(e))
        return wrong if wrong is not None and wrong.kind in framecheck.WAIT_KINDS else None

    def _to_front(self) -> None:
        """Bring the game to the front unless it's there already: a capture sees whatever window is on top."""
        if window.foreground_title() != self.title:
            window.bring_to_front(self.title)
            time.sleep(0.3)

    def _capture(self) -> np.ndarray:
        """The whole client area, full size, RGB, with the game in front, as the screen showed it after
        this began (a look sees what came of the keys before it). Raises BadLook if the game window isn't fit to
        capture (gone, minimized, not the size the templates were made for, partly off the screen, not in front,
        covered), moved while it was captured, or the frame is stale (framecheck). While it's unfit in a way the user
        sets right (window_wait), the game isn't brought to the front: a hunt standing still on a resized window
        looked twice a second, and would have pulled the game over whatever the user was doing every time (the
        reviewer's focus.py: 136 times in 63 s). A minimized game is restored, as it always was, unless the last look
        was waiting for the user already. A command's (`full_frames`): the perception service's frame shown after now,
        its checks the service's (BadLook if it had none)."""
        if getattr(self, "full_frames", None) is not None:
            return self.full_frames(time.monotonic())
        wait = self.window_wait()
        if wait is not None and (wait.kind != "minimized" or getattr(self, "_waited", None) is not None):
            self._waited = wait
            raise BadLook(wait)
        self._waited = None
        try:
            self._to_front()
            begun = time.monotonic()
            area = self._client_area()
            wrong = self._window_problem(area)
        except BadLook:
            raise
        except RuntimeError as e:  # the game window gone meanwhile
            raise BadLook(Problem("gone", str(e))) from None
        if wrong is not None:
            raise BadLook(wrong)
        shot = self.screen.shot(area, after=begun)
        full = shot.crop(0, 0, area["width"], area["height"], rgb_of)
        self.seen_at = shot.t
        try:
            after = self._client_area()  # read again: a window being dragged moves between the two (16:48:45)
        except RuntimeError as e:
            raise BadLook(Problem("gone", str(e))) from None
        if wrong := framecheck.moved_problem(area, after) or framecheck.stale_problem(shot.t, time.monotonic()):
            raise BadLook(wrong)
        if self.area is not None and area != self.area:
            text = f"the game window moved: ({self.area['left']}, {self.area['top']}) -> ({area['left']}, {area['top']})"
            self.log(f"  {text}")
            telemetry.emit("event", kind="window", text=text)
        self.area = area
        return full

    def recover(self, problem: Problem) -> None:
        """After a look that failed its checks (`problem`), before the next: the minimap's picture and her name tag are
        searched for afresh; if the capture itself may be at fault (framecheck.CAPTURE_KINDS: the frame offset from the
        window, or old), the duplication is opened afresh at the first such failure since a frame last passed, mss
        takes over for capture.RETRY_S at the second, and then the two in turn every CAPTURE_AGAIN_S while they last
        (the window moving is waited out: its next look reads where it is). The game window is never moved or resized
        here. (The simulator's eyes have none of these, and nothing is done.)"""
        if getattr(self, "minimap", None) is not None:
            self.minimap.lose()
        if getattr(self, "name_tag", None) is not None:
            self.name_tag.lose()
        screen, log = getattr(self, "screen", None), getattr(self, "log", print)
        if problem.kind not in framecheck.CAPTURE_KINDS:
            return
        fixes, now = getattr(self, "_fixes", 0), time.monotonic()
        if fixes >= 2 and now - getattr(self, "_fixed_at", -1e18) < CAPTURE_AGAIN_S:
            return
        if fixes % 2 == 0 and hasattr(screen, "reopen"):
            screen.reopen()
            log("  the capture is duplicated afresh")
        elif fixes % 2 == 1 and hasattr(screen, "fall_back"):
            screen.fall_back(problem.why)
            log(f"  the capture falls back to mss for {capture.RETRY_S:.0f} s")
        self._fixes, self._fixed_at = fixes + 1, now

    # --- questions about a look's frame -----------------------------------------------------------

    def monsters(self, obs: Observation, up: float, down: float) -> list[Mob]:
        """Monsters from `up` map units above her feet to `down` below (her screen position must be known)."""
        return self.detector.detect(obs.frame, rows=band_rows(obs, up, down))

    def monsters_wide(self, obs: Observation) -> list[Mob]:
        """Every monster on screen (above the bottom bar)."""
        return self.detector.detect_wide(obs.frame, (0, int(obs.frame.shape[0] * obs.geo.hud_top)))

    def refind(self, obs: Observation, mob: Mob, box: tuple[int, int, int, int], fallback: bool = True,
               near: int | None = None, memo: dict | None = None) -> Mob | None:
        """Monster `mob`, seen in an earlier frame, looked for again where its picture should be now (`box`:
        x0, y0, x1, y1), above the bottom bar (MobDetector.refind; without `fallback`, only with the template
        that matched it last and its animation's next frames, not its kind's whole set; `near` and `memo` as
        there, given only to a detector that takes them: the simulator's fakes don't)."""
        h, w = obs.frame.shape[:2]
        more = {} if near is None and memo is None else {"near": near, "memo": memo}
        return self.detector.refind(obs.frame, box, mob.kind, mob.template, mob.facing,
                                    (0, 0, w, int(h * obs.geo.hud_top)), fallback, **more)

    def refind_all(self, obs: Observation, asks: list[tuple], threads: int = 1, pool=None) -> list[Mob | None] | None:
        """`refind` of each monster asked, (mob, box, fallback, near, memo, id: the caller's for it from frame to frame),
        all in one call in C on `threads` threads (MobDetector.refind_all; `pool`: an Executor for the fallbacks); None
        where that can't be (no C, or a detector without it: the simulator's fakes), and the caller asks `refind` of
        each."""
        many = getattr(self.detector, "refind_all", None)
        if many is None:
            return None
        h, w = obs.frame.shape[:2]
        return many(obs.frame, [(box, mob.kind, mob.template, mob.facing, fallback, near, memo, tid)
                                for mob, box, fallback, near, memo, tid in asks],
                    (0, 0, w, int(h * obs.geo.hud_top)), threads, pool)

    def heading(self, obs: Observation) -> Reading:
        """Which way she faces in a look's frame (perception.heading), by her picture above her name tag: not told
        where the tag wasn't seen (`tag_seen` False: where the camera puts her), nor before she has pictures."""
        tag = obs.tag if getattr(obs, "tag_seen", True) else None
        if obs.frame is None:
            return Reading(None, why="no frame")
        return self.heading_bank().read(obs.frame, tag)

    def learn_heading(self, obs: Observation, side: str, reading: Reading | None = None) -> bool:
        """A look whose keys say beyond doubt that she faces `side` teaches her pictures (Heading.learn; `reading`:
        what they read on it). True if they learnt it."""
        if obs.frame is None or obs.tag is None or not getattr(obs, "tag_seen", True):
            return False
        return self.heading_bank().learn(obs.frame, obs.tag, side, obs.t, reading)

    def keep_heading(self, obs: Observation, path) -> bool:
        """Save her box from a look's frame, three times its size around her (PNG at `path`), for a look by eye at why
        the frame and her keys disagreed. True if it was saved."""
        if obs.frame is None or obs.tag is None:
            return False
        x0, y0, x1, y1 = her_box(obs.tag, obs.geo)
        w, h = x1 - x0, y1 - y0
        crop = obs.frame[max(0, y0 - h):y1 + h, max(0, x0 - w):x1 + w]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return bool(cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2BGR)))
        except (OSError, cv2.error):
            return False

    def heading_bank(self) -> Heading:
        """Her pictures for these eyes' window size (perception.heading: kept in lab, the seed in assets until then, or
        where lab's can't be read, which the eyes' log says)."""
        if getattr(self, "_heading", None) is None:
            mine = paths.of()  # (hers)
            self._heading = Heading(self.geo, paths.calibrated(mine.heading, self.geo.suffix),
                                    paths.calibrated(mine.heading_seed, self.geo.suffix),
                                    log=lambda text: getattr(self, "log", print)(text))
        return self._heading

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
        """The loot detector, made (from the game's item pictures of `drop_names`, for frames of `geo`) on first use."""
        if self._loot is None:
            self._loot = LootDetector(geo=self.geo, items=self.drop_names)
        return self._loot
