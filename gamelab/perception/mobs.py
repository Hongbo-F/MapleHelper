"""Finding monsters on screen: colour silhouettes first, then masked template matching.

Each monster type has a few templates (sprites cut out with a mask, gamelab/assets/mobs/<kind>_*.png,
RGBA with the alpha as mask) and a colour test for its body. Step 1 matches each template's body-colour
silhouette against the frame's body-colour mask (one fast single-channel pass over the frame). Step 2
checks only the best few spots with the full masked colour template, which rejects scenery that merely
shares the colour (leaves, clouds). Frames are look frames (the game window shrunk as screen.Geometry says: half
size at 1920x1080, full size at the game's own 1366x768), and the templates are brought to their scale; the px below
were measured in half-size frames of the 1920x1080 window and go by it too.

Coarse to fine: step 1 always works at that scale (SEARCH_SCALE, 0.70 px per game px), where every number it goes by
was measured. A finer look frame (the game's own 1366x768 at full size, 1 px per game px) is shrunk to it for step 1,
and step 2 checks the places it found at the frame's own size, with the game's own pictures as they are (sprites.py):
only the pixels the game draws as they are (opaque ones: a slime is see-through, and its half-opaque pixels on screen
are partly what's behind it), in small windows around the spots. There a place passes a gate at step 1's scale first,
and then several of its templates are checked, not only its best (FINE_TEMPLATES). At 1920x1080 half size the look
frame is at that scale already, and all of it is as it was.

The matches run on a few threads (OpenCV lets go of Python's lock while it matches). Which spots step 2
checks doesn't depend on what the checks find (at a finer frame, on what the gate's checks find, all made
first), so they're all picked first, and the answers are the same as one thread's, in the same order.

A monster found says which template matched it, and so what that template's folder knows of it (INDEX,
CUTOUTS): where its feet are in the picture, which way it faces (the game's sprites face left; a flipped
template faces right), and its frame and pose.

The first map's four kinds have colour tests written by hand (PALETTES). Any other kind is looked for with a gate
learned from the map's background colours (perception.gates, `background`), and a few things go otherwise for those
kinds only, so the first map's searches are as they were: whole-screen bands overlap by the tallest such picture; a
place's gate tries its first GATE_TRIES templates, not only its first; at 1920x1080 half size step 2 checks up to
PER_PLACE of a place's templates, as at the game's own size; kinds whose pictures are the same shape (Horny and Green
Mushroom, Stump and Dark Stump) are searched as one (`units`), each kind's picture checked side by side at a place; a
monster found as a kind that shares its body with one searched apart (`twins`: a Stump, a Dark Stump and an Axe Stump)
is given the kind whose picture it is (`_pick_kind`); and at the game's own size a monster re-found as a kind that
shares its body with others isn't re-found where another's picture is the monster's (`refind`).
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

from .. import native, paths
from ..screen import STRETCHED
from .gates import FOG_ALPHAS, Gates, kind_histograms, learn
from .match import masked_match, suppress_overlaps


ASSETS = paths.ASSETS / "mobs"
THRESHOLD = 0.6              # masked colour match a spot needs to count as a monster
SILHOUETTE_THRESHOLD = 0.35  # step 1: a spot worth checking
MAX_CHECKS = 12              # step 2: best spots checked per monster type
MIN_COVER = 0.25             # step 1: a spot's window needs this share of the template's own body colour
MIN_BODY_PIXELS = 20         # a monster type is looked for only where this many pixels have its colour
COARSE = 2                   # step 1 works at 1/COARSE of its frame's size (SEARCH_SCALE)
# Step 1's frame, in px per game px: the 1920x1080 window's half size (0.703), where step 1's numbers (its
# thresholds, MIN_COVER, MIN_BODY_PIXELS, MAX_CHECKS, the colour tests) were all measured. A finer look frame is
# shrunk to it (area averaged) for step 1 only.
SEARCH_SCALE = STRETCHED.frame_scale
CHECK_SLACK = 6              # step 2 looks this far (px) around a spot, so spots closer than this are one
# Step 2 on a look frame finer than step 1's (the game's own 1366x768 at full size). There a monster's frames differ
# crisply: drawn as the game draws them on frames of the game at that size (analysis.pasted: the first map's sprites,
# each frame both ways twice, on West Rocky Mountain II's 31 frames), a snail or slime checked with the frame whose
# silhouette matched best scored 0.43-0.55 where its own frame, a few spots down at the same place, scored 1.000,
# and 16 of 120 were missed. So each place's best FINE_TEMPLATES templates (4: 112 found) are checked, each at its
# own spot, where the place's best scores FINE_GATE at step 1's scale first, as step 2 went at 1920x1080 (CHECK_SLACK
# around it): 114 of the 120 were found. Of 337 drawn, 331 scored 0.55 there (333 at 0.5, 329 at 0.6), and this
# map's rock 2.8 places a frame (10.2 at 0.5): with six drawn on each frame, the whole screen took 90 ms (97 at 0.5;
# 83 at 1920x1080, the same scenes stretched). Each template is looked for FINE_SLACK look-frame px around its spot:
# of the 189 matches scoring 0.85 or more, all were within 3 px of it (in windows of 11 px templates slid onto the
# monster beside theirs, and an Orange Mushroom's frame scored 0.61 on this map's rock).
FINE_TEMPLATES = 6
FINE_GATE = 0.55
FINE_SLACK = 4
# A learned kind's place (only theirs: the first map's go as they were) is gated with its first GATE_TRIES templates in
# turn, until one scores FINE_GATE. A stump's frames' silhouettes match a blob of its colour about alike, its flinch
# frame's (the largest) most often best: on West Rocky Mountain II's 31 labelled frames at 1366x768 (2026-10-24) the
# places of Stumps and Axe Stumps in plain view were gated with it at 0.29-0.46, where their own frame, the place's
# second or third, scored 0.67-0.77 and then 0.94-1.00 at full size. Gated with the first alone 53 of the 58 in plain
# view were found (on its 42 labelled 1920x1080 frames shrunk to 1366x768, 143 of 203), a whole screen taking 91 ms;
# with the first 2, 56 (167), in 106.
GATE_TRIES = 2
OPAQUE = 128                 # a game picture's pixel is drawn from this alpha up (sprites.build's threshold)
WIDE_BAND, WIDE_STEP = 120, 90  # detect_wide: rows per band, and rows from one band's top to the next
THREADS = 6                  # matches run at once: the band 70 -> 32 ms, the whole screen 250 -> 117 (8 are no faster).
# (Her band's on 3, tried on 2026-10-26: in the service at 30 frames a second it came 6-7 ms later, median 29 ms against
# 22-23, for 0.02-0.03 cores: D:/MapleHelper-scratch/cpu/fix/out/hunt-*.txt.)
# A monster seen in the last frame is looked for again this far (px) around where it should be (`refind`):
# at 30 frames a second a pig walking 125 units/s moves 3 px a frame, and where her minimap dot puts her
# (which places the monsters on the map) can step by a minimap pixel, 11.4 units or 8 px, as she walks.
# The research's prototype re-found 84 of 84 monsters in saved frames with 16 px, 78 with 8.
REFIND_PAD = 16
# On a look frame finer than step 1's (the game's own 1366x768 at full size), a monster's frames differ crisply, and
# the one that matched it last seldom finds it once its animation steps on. Drawn as the game draws them
# (analysis.pasted.refinds: the first map's standing and walking sprites, both ways, found on West Rocky Mountain II's
# 31 frames, then drawn where they stood as a frame they may show next), it alone found its pose's next frame 23 times
# in 50, a step from standing to walking or back 13 in 52 and a flinch 9 in 52 (at 1920x1080 half size 39, 31 and
# 27): a live track's re-find went on to the kind's whole set the rest of the time (5.1-8.8 ms a re-find, at 90% 20),
# a hidden one's stayed lost. So there, where it scores under REFIND_GOOD, the frames it most likely went to
# (`_next_frames`, REFIND_NEXT of them) are matched in the same window in turn until one scores REFIND_GOOD, the
# best kept: every one found, none going on to the whole set, 46-52 of each with the feet within a unit, in 3.2 ms
# (its pose's next frame), 7.8 (standing to walking or back) and 6.3 (a flinch); unchanged, one match as before
# (1.7). Each frame followed by the next on two frames (MobDetector.refind as here, 200 times): within a unit 182 at
# 0.9, 200 at 0.95, 168 at 0.8; at 0.95 a monster with her or another over part of it would take REFIND_NEXT matches
# every frame.
REFIND_GOOD = 0.9
REFIND_NEXT = 4
# What a folder of templates may say about them, by file name: where the monster's feet are in each picture
# ("origin", in the picture's own px, of its size "w" x "h"), which way it faces ("faces_right"; the game's
# sprites face left), its frame in the game data ("frame": "move/2") and that frame's body box ("lt", "rb").
INDEX = "index.json"      # the game's sprites (sprites.build)
CUTOUTS = "cutouts.json"  # the screenshot cut-outs, matched against the sprites (sprites.cutout_origins)
# A folder's pictures are drawn this many px per game px ({"px_per_game_px": 1.0}: sprites.build at the game's own
# size); without the file, as the 1920x1080 window shows them (1.40625: the cut-outs, and the sprites made for it).
SCALE_FILE = "scale.json"
# For kinds with a learned gate (perception.gates) only. At 1920x1080 half size, where step 1's frame is the look
# frame, each place's best PER_PLACE templates are checked, each at its own spot, as FINE_TEMPLATES are at a finer
# frame: on 42 labelled 1920x1080 frames of West Rocky Mountain II (2026-10-23) a red snail's five frames' silhouettes
# matched about alike where the gate takes only its shell, and the one checked was often the wrong frame (a stand frame
# at 0.92, its flinch checked at 0.55): checking the best 3 found 171 of 203 monsters in plain view instead of 152.
PER_PLACE = 3
# Two learned kinds share a body where their pictures of a frame (same name, same facing, set on their feet) have this
# share at least of the larger's pixels (step 2's) in common, on the frames they both have (median): Horny and Green
# Mushroom are one sprite with two caps (0.95 at either size), Stump and Dark Stump one in two colours (1.00), an Axe
# Stump a Stump with an axe (0.85; 0.81 at the least); a Red Snail fits inside a Stump (0.32-0.38). A monster found as
# one is checked as those searched apart from it (`units`), at its feet (`_pick_kind`).
TWIN_COVER = 0.8
# Of two such pictures, one has parts of its own where this share of its pixels lies outside the other's (by a px): an
# Axe Stump's axe, 15% of its picture (a Horny or Green Mushroom's own edge pixels are 5%). Then it's the monster's
# only where those parts alone match this well (masked, normalised). On the 42 labelled 1920x1080 frames of West Rocky
# Mountain II (shrunk to 1366x768, and at 1920x1080 half size) an Axe Stump's axe matched 0.83-0.97 on 39 of the 40 in
# plain view (one at 0.25), where on a Stump or Dark Stump the axe's place scored -0.6 to 0.70 (one at 0.70, the next
# 0.52); by the score alone, 10 of the 40 Axe Stumps were taken for Stumps (the Stump's picture matches the stump that
# is all of it but the axe as well).
OWN_PARTS = 0.1
OWN_MATCH = 0.6
# A re-found monster's kind is checked with its twins' pictures set on its feet where they lie on its own picture's
# (no search around: `MobDetector.aligned`): the shift, this many px at most, at which their masks have the most in
# common. The game's own frames of two kinds can be a px apart: an Axe Stump's move/1 and move/3 lie 1 px across from
# a Stump's (set on their origins), and checked there, 27 of the 40 Axe Stumps in plain view on West Rocky Mountain
# II's 1366x768 frames failed their axe's test (a Stump's track there went on on them).
ALIGN = 2
# Pictures of a frame (same name, facing and size) of two such kinds are the same shape where they have this share of
# the larger's pixels in common: Horny and Green Mushroom (0.95), Stump and Dark Stump (1.00); not a Stump and an Axe
# Stump (0.85). Step 1 matches one silhouette for them (`_shapes`).
SAME_SHAPE = 0.9


def picture_scale(folder: Path) -> float:
    """How many px per game px a folder's pictures are drawn at (SCALE_FILE)."""
    path = folder / SCALE_FILE
    if not path.is_file():
        return STRETCHED.stretch
    return float(json.loads(path.read_text(encoding="utf-8"))["px_per_game_px"])


def frame_size(w: int, h: int, shrink: float) -> tuple[int, int]:
    """A w x h picture's size in frames it's `shrink` times as big as (its px per frame px): a whole shrink cuts as
    before (half size: w // 2), any other rounds."""
    k = round(shrink)
    if abs(shrink - k) < 1e-9 and k >= 1:
        return max(1, w // k), max(1, h // k)
    return max(1, round(w / shrink)), max(1, round(h / shrink))


@dataclass(frozen=True)
class Mob:
    """A monster seen on screen, in frame pixels (half size): the template that matched it, and what that
    template knows of it. The template's centre and bottom aren't the monster's feet: a sprite stands on its
    origin, up to 8.5 map units off its picture's centre, and a slime's hop frames end 10 units above its
    feet; the screenshot cut-outs were cropped 9.5 units above a pig's feet and 14 below a slime's. Where the
    template knows its origin (`ox`, `feet`), that is where the monster stands."""
    kind: str
    x: int        # the template's centre
    y: int
    w: int
    h: int
    score: float
    ox: float | None = None    # where its feet are: x ... (None: the template doesn't know)
    feet: float | None = None  # ... and y
    facing: int = 0            # which way it faces: 1 right, -1 left (0: not known)
    pose: str = ""             # what it's doing: "stand", "move" (walking) or "hit1" (flinching); "": not known
    frame: str = ""            # the frame of its animation that matched ("move/2"), if known
    template: str = ""         # the picture that matched (its file's name: "pig_move2", "pig_1")

    @property
    def foot_x(self) -> float:
        """The x it stands at: its origin's, else the template's centre."""
        return self.x if self.ox is None else self.ox

    @property
    def feet_y(self) -> float:
        """The y it stands at: its origin's, else the template's bottom."""
        return self.y + self.h // 2 if self.feet is None else self.feet


class MobTemplate(NamedTuple):
    """One monster picture (one pose, facing one way), at frame size, and what its folder says of it (INDEX,
    CUTOUTS)."""
    rgb: np.ndarray         # its colours
    mask3: np.ndarray       # where the picture is (uint8, one copy per colour channel): step 2 matches these pixels
    silhouette: np.ndarray  # its body colour, at 1/COARSE of SEARCH_SCALE (step 1)
    origin: tuple[float, float] | None = None  # where the monster's feet are, from its top-left (frame px)
    facing: int = 0         # which way the monster faces in it: 1 right, -1 left, 0 not known
    frame: str = ""         # its frame in the game data ("move/2")
    pose: str = ""          # that frame's action: "stand", "move" or "hit1"
    box: tuple[float, float, float, float] | None = None  # its body (left, top, right, bottom), map units
                                                          # from its feet, y up negative, facing applied
    name: str = ""          # its file's name, without ".png"
    step1: tuple[np.ndarray, np.ndarray] | None = None  # its colours and mask at step 1's scale, where that's not the
                                                         # frame's (FINE_GATE's check)


def split_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A picture's red, green and blue as int16 arrays, so differences between them can go negative."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return r, g, b


def _pig(r, g, b):
    return (r > 190) & (g > 120) & (b > 110) & (r - g > 18) & (r - b > 15)


def _slime(r, g, b):
    return (g > 170) & (g - r > 30) & (g - b > 90)


def _snail(r, g, b):
    teal = (r < 90) & (g > 80) & (b > 80) & (g < 170) & (np.abs(g - b) < 40)
    lavender = (r > 140) & (b > 190) & (g > 130) & (b - g > 15)
    return teal | lavender


def _orange_mushroom(r, g, b):
    """The orange cap (daffodils are yellower: higher green)."""
    return (r > 215) & (g > 100) & (g < 175) & (b < 90) & (r - g > 55)


# Each monster kind's body colour, as a test on a picture's r, g and b (int16, from split_channels). A kind
# needs one before its templates are used.
PALETTES = {"pig": _pig, "slime": _slime, "snail": _snail, "orange-mushroom": _orange_mushroom}
KINDS_WITH_PALETTE = tuple(PALETTES)


def palette(kind: str, rgb: np.ndarray, channels=None, gates: Gates | None = None) -> np.ndarray:
    """Pixels that have the colour of this monster's body. `channels` = `split_channels(rgb)`, if already
    split (saves time when testing several kinds on one frame). A kind without a hand-made test (PALETTES) is tested
    with its learned gate, from `gates`."""
    r, g, b = channels or split_channels(rgb)
    if kind in PALETTES:
        return PALETTES[kind](r, g, b)
    if gates is None or kind not in gates:
        raise ValueError(f"unknown monster kind {kind!r}")
    return gates.test(kind, r, g, b)


def read_meta(folders: list[Path]) -> dict[str, dict]:
    """What these folders say about their templates (INDEX, CUTOUTS), by file name."""
    out: dict[str, dict] = {}
    for folder in folders:
        for name in (INDEX, CUTOUTS):
            if (folder / name).is_file():
                out.update(json.loads((folder / name).read_text(encoding="utf-8")))
    return out


def _picture(rgba: np.ndarray, shrink: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A template's picture (RGBA) in frames `shrink` times smaller than it is drawn: its colours, the pixels step 2
    matches and where it's drawn (step 1's silhouette), both uint8. At its own size (the game's pictures at the game's
    own 1366x768) it's as it is, and step 2 matches only its opaque pixels: those the game draws as they are (a
    slime is see-through: 11,464 of its 17,927 pixels drawn are half opaque, alpha 128-254, and on screen partly
    what's behind it; drawn as the game draws it on frames of the game at 1366x768, 396 times, its frames scored
    0.96 or more with them, median 0.98, and 1.000 without; on the same frames without it, the best anywhere near
    0.46 with them and 0.40 without). Shrunk (area averaged), both are where it's drawn, as before."""
    alpha = rgba[..., 3]
    if abs(shrink - 1) < 1e-9:
        return np.ascontiguousarray(rgba[..., :3]), (alpha == 255).astype(np.uint8), (alpha >= OPAQUE).astype(np.uint8)
    h, w = rgba.shape[:2]
    size = frame_size(w, h, shrink)
    rgb = cv2.resize(np.ascontiguousarray(rgba[..., :3]), size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize((alpha >= OPAQUE).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    return rgb, mask, mask


def _known(entry: dict | None, size: tuple[int, int], flip: bool) -> tuple:
    """What a template's entry (INDEX, CUTOUTS) says of it at frame size `size` (height, width), as drawn or
    flipped: MobTemplate's origin, facing, frame, pose and box. Without an origin, nothing is known."""
    if not entry or entry.get("origin") is None:
        return None, 0, "", "", None
    th, tw = size
    ox, oy = entry["origin"][0] * tw / entry["w"], entry["origin"][1] * th / entry["h"]
    right = bool(entry.get("faces_right")) != flip
    box = None
    if entry.get("lt") and entry.get("rb"):  # the game's boxes are for a monster facing left
        (left, top), (r, bottom) = entry["lt"], entry["rb"]
        box = (-r, top, -left, bottom) if right else (left, top, r, bottom)
    frame = entry.get("frame", "")
    return (tw - ox if flip else ox, oy), 1 if right else -1, frame, frame.split("/")[0], box


class MobDetector:
    """Templates come from `folder` (or a list of folders): PNGs named <kind>_<anything>.png, RGBA at full
    screen size. gamelab/assets/mobs has cut-outs from screenshots; sprites.py makes them from the game data.
    What the folders say about their templates (INDEX, CUTOUTS) comes along, and `meta` (by file name) on top.
    `threads`: how many matches run at once (1: all on the caller's thread). `frame_scale`: the frames' px per game px
    (screen.Geometry.frame_scale; by default the 1920x1080 window's at half size), which each folder's pictures are
    brought to from theirs (`picture_scale`): a folder drawn at the frames' own scale is used as it is.
    `search_scale`: step 1's frame, px per game px (SEARCH_SCALE; a frame no finer than it is searched as it is).
    `kinds`: only these kinds' templates (by default every kind with a colour test). Given (a map's own:
    sprites.detector_for at the game's own size), there may be none, a map none of whose monsters has a colour test
    yet, and the detector then finds nothing; without it, no templates at all is a mistake (a folder with none).
    `background`: a map's background colours (gates.load_background), against which the kinds in `learned` (by
    default those in `kinds`) that have no hand-made colour test get a gate learned (gates.learn) from their own
    pictures as step 1 sees them, blended toward a fog's white by `fog_alphas`; without it, they aren't looked for."""

    def __init__(self, folder: Path | list[Path] = ASSETS, threads: int = THREADS, meta: dict | None = None,
                 frame_scale: float = STRETCHED.frame_scale, search_scale: float = SEARCH_SCALE, kinds=None,
                 background: np.ndarray | None = None, learned=None, fog_alphas=FOG_ALPHAS):
        self.templates: dict[str, list[MobTemplate]] = {}
        folders = folder if isinstance(folder, list) else [folder]
        meta = {**read_meta(folders), **(meta or {})}
        shrinks = {f: picture_scale(f) / frame_scale for f in folders}
        self.px = frame_scale / STRETCHED.frame_scale  # this frame's px per px of a half-size 1920x1080 frame
        # Step 1's frame: the look frame shrunk `search` times (1: the look frame itself), its px `spx` of a half-size
        # 1920x1080 frame's (1 at either size; 0.71 for a half-size look at 1366x768, searched as it is).
        self.search = max(1.0, frame_scale / search_scale)
        self.spx = self.px / self.search
        to_learn = set() if background is None else set((kinds if learned is None else learned) or ()) - set(PALETTES)
        pictures = []  # (path, kind, its colours, step 2's mask, where it's drawn, and those two at step 1's scale)
        for path in sorted(p for f in folders for p in f.glob("*.png")):
            kind = path.stem.split("_")[0]
            if kind not in PALETTES and kind not in to_learn or kinds is not None and kind not in kinds:
                continue
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            rgb, mask, drawn = _picture(rgba, shrinks[path.parent])
            small, small_drawn = rgb, drawn
            if self.search != 1:  # step 1's picture, from the picture as it is
                small, _, small_drawn = _picture(rgba, shrinks[path.parent] * self.search)
            pictures.append((path, kind, rgb, mask, drawn, small, small_drawn))
        # Each learned kind's gate, from its pictures' pixels as step 1 sees them, against the map's background.
        luts = {}
        for k in sorted(to_learn):
            px = [s[d > 0] for _, kind, _, _, _, s, d in pictures if kind == k]
            if px:
                luts[k] = learn(kind_histograms(np.concatenate(px), fog_alphas), background)
        self.gates = Gates(luts) if luts else None
        self.learned = frozenset(luts)  # the kinds looked for with a learned gate
        for path, kind, rgb, mask, drawn, small, small_drawn in pictures:
            for flip in (False, True):  # monsters face either way
                t, m = (rgb[:, ::-1], mask[:, ::-1]) if flip else (rgb, mask)
                t = np.ascontiguousarray(t)
                step1 = None
                if self.search == 1:
                    s, d = t, (drawn[:, ::-1] if flip else drawn)
                else:
                    s, d = (small[:, ::-1], small_drawn[:, ::-1]) if flip else (small, small_drawn)
                    s = np.ascontiguousarray(s)
                    step1 = (s, np.dstack([d] * 3))
                silhouette = (palette(kind, s, gates=self.gates) & (d > 0)).astype(np.float32)
                coarse = cv2.resize(silhouette, (max(1, s.shape[1] // COARSE), max(1, s.shape[0] // COARSE)),
                                    interpolation=cv2.INTER_AREA)
                self.templates.setdefault(kind, []).append(MobTemplate(
                    t, np.dstack([m] * 3), coarse, *_known(meta.get(path.name), t.shape[:2], flip), path.stem,
                    step1))
        if not self.templates and kinds is None:
            raise RuntimeError(f"no monster templates in {folder}")
        # Whole-screen bands overlap by the tallest learned kind's picture and step 2's slack, so one standing across a
        # band's edge is whole in the next (an Axe Stump's flinch is 93 game px tall, a band's overlap 43 at the game's
        # own size); without learned kinds, as they were.
        tallest = max((t.rgb.shape[0] for k in self.learned for t in self.templates.get(k, ())), default=0)
        self.overlap = tallest + round(CHECK_SLACK * self.px) if tallest else 0
        self.twins = _twins({k: self.templates[k] for k in self.learned if k in self.templates})
        same = _twins({k: self.templates[k] for k in self.learned if k in self.templates}, SAME_SHAPE)
        # What step 1 searches as one (`units`): each kind alone, as always, but learned kinds whose pictures are the
        # same shape (Horny and Green Mushroom, Stump and Dark Stump: SAME_SHAPE), searched together: by their gates
        # together (their silhouettes made by them too), one silhouette a frame for them (`_shapes`), one set of
        # places, and at each place step 2 checks each kind's picture, side by side, the better kept. Apart, each made
        # a place at the same monster, gated and checked there. Kinds that share a body with kinds searched apart
        # (`apart`: an Axe Stump with a Stump and a Dark Stump) are told apart where they're found (`_picked`).
        self.units: list[tuple[tuple[str, ...], list[tuple[str, MobTemplate]], list[list[int]]]] = []
        for kind in self.templates:
            if any(kind in kinds for kinds, _, _ in self.units):
                continue
            kinds = (kind,) if kind not in same else tuple(sorted(_together(kind, same)))
            if len(kinds) > 1:
                for k in kinds:
                    self.templates[k] = [t._replace(silhouette=self._silhouette(kinds, *t.step1) if t.step1 else
                                                    self._silhouette(kinds, t.rgb, t.mask3)) for t in self.templates[k]]
            members = [(k, t) for k in kinds for t in self.templates[k]]
            self.units.append((kinds, members, _shapes(members)))
        unit = {k: kinds for kinds, _, _ in self.units for k in kinds}
        # (a picture's name and facing, a twin's picture's name) -> where the twin's lies on it (ALIGN, `refind`)
        self.aligned = {(a.name, a.facing, b.name): _aligned(a, b) for k, others in self.twins.items()
                        for a in self.templates[k] for o in others for b in self.templates[o]
                        if a.frame and a.frame == b.frame and a.facing == b.facing and a.origin and b.origin}
        self.apart = {k: others for k, v in self.twins.items() if (others := tuple(t for t in v if t not in unit[k]))}
        self._pool = ThreadPoolExecutor(threads, "monster-search", _one_opencv_thread) if threads > 1 else None
        self.fallbacks = 0  # re-finds that went on to the kind's whole set (refind)
        native.lib()  # built (the first time, a few seconds) and loaded now, not in the first look

    def detect(self, frame: np.ndarray, rows: tuple[int, int] | None = None) -> list[Mob]:
        """Monsters in `frame` (RGB, a look frame at `frame_scale`). `rows` limits the search to a band."""
        return self._picked(frame, rows, self._detect(frame, rows, self._map))

    def detect_wide(self, frame: np.ndarray, rows: tuple[int, int]) -> list[Mob]:
        """Monsters over a tall stretch of the frame, band by band. Over the whole screen, green
        foliage took all of the slime candidate spots and three slimes in plain view were missed;
        each band gets its own best spots. The bands are searched at once, each on one thread."""
        # A band starts every WIDE_STEP rows until one reaches rows[1] (it's cut short there); always one.
        band, step = round(WIDE_BAND * self.px), round(WIDE_STEP * self.px)
        band = step + max(band - step, self.overlap)
        starts = range(rows[0], max(rows[0] + 1, rows[1] - band + step), step)
        bands = self._map(lambda y0: self._detect(frame, (y0, min(rows[1], y0 + band)), _serial), starts)
        return self._picked(frame, rows, suppress_overlaps([hit for band in bands for hit in band]))

    def refind(self, frame: np.ndarray, box: tuple[int, int, int, int], kind: str, template: str = "",
               facing: int = 0, bounds: tuple[int, int, int, int] | None = None, fallback: bool = True) -> Mob | None:
        """A monster seen before, looked for again around where it should be: `box` (x0, y0, x1, y1, frame
        px) is where its picture should be, searched REFIND_PAD px around (within `bounds`, by default the
        frame). First the picture that matched it last (`template`, `facing`: Mob.template and Mob.facing; a
        cut-out that doesn't know its facing, both ways), one masked match over the window; on a frame finer than
        step 1's, where that scores under REFIND_GOOD, the frames its animation most likely went to, in turn
        (`_next_frames`); if none reaches THRESHOLD (and `fallback`), every template of its kind there, as `detect`
        does. On a frame finer than step 1's, a learned kind that shares its body with others (`twins`: Horny and
        Green Mushroom; a Stump, a Dark Stump and an Axe Stump) is re-found only where its picture is the monster's,
        told apart from theirs side by side at its feet as a search tells them (`_its_kind`): the other kind's picture
        of a frame scores 0.79-0.92 on a mushroom and up to 1.00 on a stump, so a track would otherwise go on on
        another kind's monster walking past or standing where it died. The one found nearest the box's centre, or
        None. On the caller's thread: a window is small, and the pool's hand-offs cost more."""
        h, w = frame.shape[:2]
        bx0, by0, bx1, by1 = bounds or (0, 0, w, h)
        pad = round(REFIND_PAD * self.px)
        x0, y0 = max(bx0, int(box[0]) - pad), max(by0, int(box[1]) - pad)
        x1, y1 = min(bx1, int(box[2]) + pad), min(by1, int(box[3]) + pad)
        if x1 <= x0 or y1 <= y0 or kind not in self.templates:
            return None
        window = frame[y0:y1, x0:x1]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best, last = None, None
        for t in self.templates[kind]:
            if not template or t.name != template or (facing and t.facing and t.facing != facing):
                continue
            last = last or t
            th, tw = t.rgb.shape[:2]
            if window.shape[0] < th or window.shape[1] < tw:
                continue
            _, score, _, (x, y) = cv2.minMaxLoc(masked_match(window, t.rgb, t.mask3))
            if score >= THRESHOLD and (best is None or score > best.score):
                best = _seen(kind, t, x0 + x + tw // 2, y0 + y + th // 2, tw, th, round(float(score), 3))
        # A hidden track's (without `fallback`) too. Tried with its last template only (2026-10-26: 6.4 ms a frame less
        # on the saved frames of 01:01), a crowd's tracks went otherwise: on 600 frames of 00:25, 142 frames' states
        # differed and 2 kills more were hers, given to tracks left hidden 10-20 s at one x (D:/MapleHelper-scratch/
        # cpu/review-safety/perc; README, CPU).
        if self.search != 1 and last is not None and (best is None or best.score < REFIND_GOOD):
            for t in _next_frames(self.templates[kind], last):  # (finer than step 1's: its frames differ)
                th, tw = t.rgb.shape[:2]
                if window.shape[0] < th or window.shape[1] < tw:
                    continue
                _, score, _, (x, y) = cv2.minMaxLoc(masked_match(window, t.rgb, t.mask3))
                if score >= THRESHOLD and (best is None or score > best.score):
                    best = _seen(kind, t, x0 + x + tw // 2, y0 + y + th // 2, tw, th, round(float(score), 3))
                    if score >= REFIND_GOOD:
                        break
        if best is not None and not self._its_kind(frame[:by1, :bx1], best):
            best = None  # another kind's monster: not this one
        if best is not None or not fallback:
            return best
        self.fallbacks += 1
        found = [_moved(m, x0, y0) for m in self._detect(window, None, _serial, (kind,))]
        found = [m for m in found if m.kind == kind and self._its_kind(frame[:by1, :bx1], m)]
        return min(found, key=lambda m: (m.x - cx) ** 2 + (m.y - cy) ** 2) if found else None

    def _its_kind(self, frame: np.ndarray, m: Mob) -> bool:
        """Whether monster `m`, re-found as its kind (a learned one that shares its body with others: `twins`), is
        that kind: its picture, just matched where it is, checked side by side with theirs set on its feet there, each
        where it lies on its picture (`_pick_kind` with no slack, `aligned`; matched 4 px around, as a search does, a
        twin's picture took 1.2-1.5 ms, against 0.1). Only on a look frame finer than step 1's (the game's own
        pixels): at 1920x1080 half size, where the game's pictures are resampled, an Axe Stump's axe failed its test
        on 3 of the 81 drawn on West Rocky Mountain II's frames and re-found as their next frame (`bench-paste`),
        which were then lost, and the game isn't run at that size any more. Any other kind is its kind."""
        if m.kind not in self.twins or self.search == 1:
            return True
        return _pick_kind(frame, m, self.twins[m.kind], self.templates, 0, self.aligned).kind == m.kind

    def _silhouette(self, kinds: tuple[str, ...], picture: np.ndarray, mask3: np.ndarray) -> np.ndarray:
        """A learned kind's template's silhouette at 1/COARSE of step 1's scale (MobTemplate.silhouette), by the gates
        of `kinds` together, from its picture at step 1's scale and where it's drawn there (`mask3`)."""
        silhouette = (self.gates.test(kinds, *split_channels(picture)) & (mask3[..., 0] > 0)).astype(np.float32)
        return cv2.resize(silhouette, (max(1, picture.shape[1] // COARSE), max(1, picture.shape[0] // COARSE)),
                          interpolation=cv2.INTER_AREA)

    def _picked(self, frame: np.ndarray, rows: tuple[int, int] | None, hits: list[Mob]) -> list[Mob]:
        """`hits`, each of a learned kind that shares its body with kinds searched apart from it (`twins` outside its
        unit: an Axe Stump's with a Stump's) given the kind whose picture it is (`_pick_kind`, searched step 2's slack
        around), within the rows searched. Those searched with it were checked at its place already, side by side."""
        if not self.apart or not any(m.kind in self.apart for m in hits):
            return hits
        if rows:
            frame = frame[:max(0, rows[1])]
        slack = FINE_SLACK if self.search != 1 else round(CHECK_SLACK * self.spx)
        out = [_pick_kind(frame, m, self.apart[m.kind], self.templates, slack) if m.kind in self.apart else m
               for m in hits]
        return suppress_overlaps(out) if any(a is not b for a, b in zip(out, hits)) else out

    def _map(self, fn, items) -> list:
        """fn of each item, in order: on the pool, or here with one thread."""
        return list(self._pool.map(fn, items)) if self._pool else [fn(item) for item in items]

    def _detect(self, frame: np.ndarray, rows: tuple[int, int] | None, run, only=None) -> list[Mob]:
        """detect, with `run(fn, items)` doing the matches (on the pool, or one after another); `only`: these
        kinds alone (default: every kind)."""
        y_off = 0
        if rows:
            y_off = max(0, rows[0])
            frame = frame[y_off:rows[1]]
        if not frame.shape[0] or not frame.shape[1]:  # a band with no rows (her band off screen: Eyes.band_rows),
            return []  # as at 1920x1080 half size, where no colour then passed; shrunk for step 1, OpenCV raised
        small = step1_frame(frame, self.search)  # step 1's frame: SEARCH_SCALE, unless the look frame is no finer
        if self.search != 1:  # the look frame's px per step 1's
            sx, sy = frame.shape[1] / small.shape[1], frame.shape[0] / small.shape[0]
        # A size that's a multiple of COARSE, so step 1's mask is an exact shrink. An odd one (the fight's
        # band was 151 rows) smeared faint colour into blank windows, which then scored a perfect 1.0.
        small = small[:small.shape[0] // COARSE * COARSE, :small.shape[1] // COARSE * COARSE]
        if self.search == 1:
            frame = small
        channels = split_channels(small)
        coarse_size = (small.shape[1] // COARSE, small.shape[0] // COARSE)
        looked = None  # every pixel's learned kinds (gates.Gates.lookup), worked out once for them all
        units = []  # (its kinds, their templates as (kind, template), shapes, their colour at 1/COARSE size, its cover)
        for kinds, members, shapes in self.units:
            if only is not None and any(k not in only for k in kinds):
                members = [(k, t) for k, t in members if k in only]
                shapes = [[i] for i in range(len(members))]
                if not members:
                    continue
            if kinds[0] in self.learned:
                looked = self.gates.lookup(*channels) if looked is None else looked
                colour = self.gates.of(kinds, looked)
            else:
                colour = palette(kinds[0], small, channels)
            if colour.sum() < MIN_BODY_PIXELS * self.spx * self.spx:
                continue
            coarse = cv2.resize(colour.astype(np.float32), coarse_size, interpolation=cv2.INTER_AREA)
            units.append((kinds, members, shapes, coarse, Cover(coarse)))
        # Step 1, every template of every kind (one a shape: `units`); then the spots each kind checks (kinds that
        # share a body, together); then step 2, all at once, every template of each shape picked.
        jobs = [(coarse, cover, members[shape[0]][1], i) for _, members, shapes, coarse, cover in units
                for i, shape in enumerate(shapes)]
        found = run(lambda job: _spots(*job), jobs)
        checks, places, start, slack = [], [], 0, round(CHECK_SLACK * self.spx)
        for kinds, members, shapes, _, _ in units:
            spots = sorted((s for f in found[start:start + len(shapes)] for s in f), reverse=True)
            start += len(shapes)
            learned = kinds[0] in self.learned
            if self.search == 1:
                picked = (_to_check(spots, slack) if not learned
                          else [spot for place in _places(spots, slack, PER_PLACE) for spot in place])
                checks += [(*members[j], cx, cy) for cx, cy, i in picked for j in shapes[i]]
            else:
                places += [(learned, [(*members[shapes[i][0]], cx, cy) for cx, cy, i in place],
                            [(*members[j], cx, cy) for cx, cy, i in place for j in shapes[i]])
                           for place in _places(spots, slack)]
        if self.search == 1:
            results = run(lambda c: _check(frame, c[1], c[2], c[3], slack), checks)
        else:  # the gate at step 1's scale, then each place's templates at the look frame's, around their own spots
            gate = run(lambda p: _gate(small, p[1], slack, GATE_TRIES if p[0] else 1), places)
            checks = [(kind, t, *_finer(t, cx, cy, sx, sy)) for (_, _, every), g in zip(places, gate)
                      if g is not None and g[0] >= FINE_GATE for kind, t, cx, cy in every]
            results = run(lambda c: _check(frame, c[1], c[2], c[3], FINE_SLACK), checks)
        hits = []
        for (kind, template, *_), result in zip(checks, results):
            if result and result[0] >= THRESHOLD:
                score, x, y, w, h = result
                hits.append(_seen(kind, template, x, y_off + y, w, h, round(score, 3)))
        return suppress_overlaps(hits)


def step1_frame(frame: np.ndarray, search: float) -> np.ndarray:
    """A look frame (or a band of it) as step 1 sees it, `search` times smaller (MobDetector.search): shrunk to
    SEARCH_SCALE (area averaged), unless it's no finer (as it is)."""
    if search == 1:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (max(1, round(w / search)), max(1, round(h / search))), interpolation=cv2.INTER_AREA)


def _seen(kind: str, t: MobTemplate, x: int, y: int, w: int, h: int, score: float) -> Mob:
    """The monster template `t` matched, centred at (x, y): where it stands, if the template knows."""
    ox = feet = None
    if t.origin is not None:  # the window's top-left (the centre is that plus half the size, rounded down)
        ox, feet = round(x - w // 2 + t.origin[0], 2), round(y - h // 2 + t.origin[1], 2)
    return Mob(kind, x, y, w, h, score, ox, feet, t.facing, t.pose, t.frame, t.name)


def _twins(templates: dict[str, list[MobTemplate]], least: float = TWIN_COVER) -> dict[str, tuple[str, ...]]:
    """The kinds each of these shares its body with (TWIN_COVER), by kind: those whose pictures of the frames both
    have (same frame, same facing, set on their feet) have that much in common, the larger's share (median over those
    frames)."""
    out: dict[str, list[str]] = {}
    kinds = sorted(templates)
    for i, a in enumerate(kinds):
        for b in kinds[i + 1:]:
            covers = [_cover(ta, tb) for ta in templates[a] for tb in templates[b]
                      if ta.frame and ta.frame == tb.frame and ta.facing == tb.facing and ta.origin and tb.origin]
            if covers and float(np.median(covers)) >= least:
                out.setdefault(a, []).append(b)
                out.setdefault(b, []).append(a)
    return {k: tuple(v) for k, v in out.items()}


def _shapes(members: list[tuple[str, MobTemplate]]) -> list[list[int]]:
    """Which of a unit's templates (kind, template) step 1 matches as one (indices, the first's silhouette for them
    all): those of other kinds with the same frame, facing and size whose pictures, set on their feet, have SAME_SHAPE
    of the larger's pixels in common; every other template alone."""
    shapes: list[list[int]] = []
    for i, (kind, t) in enumerate(members):
        for shape in shapes:
            t0 = members[shape[0]][1]
            if (kind not in {members[j][0] for j in shape} and t.frame and t.frame == t0.frame
                    and t.facing == t0.facing and t.rgb.shape == t0.rgb.shape
                    and t.silhouette.shape == t0.silhouette.shape and t.origin and t0.origin
                    and _cover(t, t0) >= SAME_SHAPE):
                shape.append(i)
                break
        else:
            shapes.append([i])
    return shapes


def _together(kind: str, twins: dict[str, tuple[str, ...]]) -> set[str]:
    """`kind` and every kind it shares a body with, through others too (`twins`)."""
    out, todo = {kind}, [kind]
    while todo:
        for k in twins.get(todo.pop(), ()):
            if k not in out:
                out.add(k)
                todo.append(k)
    return out


def _cover(a: MobTemplate, b: MobTemplate, shift: tuple[int, int] = (0, 0)) -> float:
    """How much two pictures set on their feet have in common (step 2's masks), the larger's share; with `shift`, b
    moved that many px across and down from there."""
    ma, mb = a.mask3[..., 0] > 0, b.mask3[..., 0] > 0
    dx = int(round(a.origin[0] - b.origin[0])) + shift[0]  # b's top-left in a's px
    dy = int(round(a.origin[1] - b.origin[1])) + shift[1]
    h, w = ma.shape
    ys, xs = np.nonzero(mb)
    ys, xs = ys + dy, xs + dx
    inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    both = int(ma[ys[inside], xs[inside]].sum())
    return both / max(1, int(ma.sum()), int(mb.sum()))


def _aligned(a: MobTemplate, b: MobTemplate) -> tuple[int, int]:
    """Where picture b lies on picture a (of one frame, facing one way): the shift from where their feet set it, ALIGN
    px at most, at which their masks have the most in common (`_cover`), the smallest of equals."""
    shifts = [(dx, dy) for dy in range(-ALIGN, ALIGN + 1) for dx in range(-ALIGN, ALIGN + 1)]
    return max(shifts, key=lambda d: (_cover(a, b, d), -abs(d[0]) - abs(d[1])))


def _pick_kind(frame: np.ndarray, m: Mob, twins: tuple[str, ...], templates: dict[str, list[MobTemplate]],
               slack: int, aligned: dict | None = None) -> Mob:
    """Monster `m` as the kind, of its own and `twins`, whose picture of its frame (facing its way, set on its feet,
    matched `slack` px around; with no slack, there alone, shifted to where it lies on `m`'s picture by `aligned`:
    MobDetector.aligned) it is. A picture with parts of its own (OWN_PARTS: an Axe Stump's axe) is the monster's
    only where those parts match too (OWN_MATCH), and then it is, over the pictures it holds (a Stump's); otherwise
    the one that scores best (Horny or Green Mushroom, Stump or Dark Stump: their colours)."""
    if not m.frame or m.ox is None:
        return m
    own = next((t for t in templates.get(m.kind, ()) if t.name == m.template and t.facing == m.facing), None)
    if own is None:
        return m
    th, tw = own.rgb.shape[:2]
    cands = [(m, own, int(m.x) - tw // 2, int(m.y) - th // 2)]  # (as that kind, its picture, its top-left)
    for kind in twins:
        for t in templates.get(kind, ()):
            if t.frame != m.frame or t.facing != m.facing or t.origin is None:
                continue
            th, tw = t.rgb.shape[:2]
            cx, cy = int(round(m.ox - t.origin[0])) + tw // 2, int(round(m.feet - t.origin[1])) + th // 2
            if slack:
                found = _match_at(frame, t.rgb, t.mask3, cx, cy, slack)
            else:
                dx, dy = (aligned or {}).get((own.name, own.facing, t.name), (0, 0))
                found = _match_here(frame, t.rgb, t.mask3[..., 0] > 0, cx + dx, cy + dy)
            if found is not None:
                score, x, y, w, h = found
                cands.append((_seen(kind, t, x, y, w, h, round(score, 3)), t, x - w // 2, y - h // 2))
    if len(cands) == 1:
        return m
    kept, bigger = [], []
    for i, (seen, t, x0, y0) in enumerate(cands):
        mask = t.mask3[..., 0] > 0
        others = np.zeros(mask.shape, np.uint8)
        for j, (_, o, ox0, oy0) in enumerate(cands):
            if j != i:
                _stamp(others, o.mask3[..., 0] > 0, ox0 - x0, oy0 - y0)
        parts = mask & (cv2.dilate(others, np.ones((3, 3), np.uint8)) == 0)
        if parts.sum() < OWN_PARTS * mask.sum():
            kept.append(seen)
            continue
        patch = frame[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
        if patch.shape[:2] == mask.shape and _score(patch, t.rgb, parts) >= OWN_MATCH:
            bigger.append(seen)
    return max(bigger or kept or [m], key=lambda c: c.score)


def _match_here(frame: np.ndarray, tpl: np.ndarray, mask: np.ndarray, cx: int, cy: int):
    """`_match_at` with no slack: a picture's match centred at (cx, cy) alone (`_score`; `mask`: where it's drawn,
    one channel), or None where it isn't wholly in the frame."""
    th, tw = tpl.shape[:2]
    x0, y0 = cx - tw // 2, cy - th // 2
    if x0 < 0 or y0 < 0 or y0 + th > frame.shape[0] or x0 + tw > frame.shape[1]:
        return None
    return _score(frame[y0:y0 + th, x0:x0 + tw], tpl, mask), cx, cy, tw, th


def _score(patch: np.ndarray, tpl: np.ndarray, mask: np.ndarray) -> float:
    """The masked, normalised match of a picture on a patch of its size, at that one place: masked_match's score
    (OpenCV's TM_CCOEFF_NORMED with a mask: each channel's mean over the mask taken off, the sums over every
    channel), worked out in numpy: the same to 1e-9, in a seventh of the time OpenCV takes for one place (0.1 ms
    against 0.73, a 58x56 picture). -1 where it's undefined (a flat patch), as there."""
    i, t = patch[mask].astype(np.float64), tpl[mask].astype(np.float64)
    i -= i.mean(0)
    t -= t.mean(0)
    den = float(np.sqrt(float((t * t).sum()) * float((i * i).sum())))
    return float((t * i).sum()) / den if den > 0 else -1.0


def _stamp(canvas: np.ndarray, mask: np.ndarray, dx: int, dy: int) -> None:
    """Set `canvas` (uint8) to 1 where `mask` is, its top-left at (dx, dy) of the canvas."""
    h, w = canvas.shape
    mh, mw = mask.shape
    y0, x0, y1, x1 = max(0, dy), max(0, dx), min(h, dy + mh), min(w, dx + mw)
    if y1 > y0 and x1 > x0:
        canvas[y0:y1, x0:x1] |= mask[y0 - dy:y1 - dy, x0 - dx:x1 - dx]


def _moved(m: Mob, dx: int, dy: int) -> Mob:
    """A monster found in a window of the frame, in the frame's px (the window's top-left at dx, dy)."""
    return replace(m, x=m.x + dx, y=m.y + dy, ox=None if m.ox is None else round(m.ox + dx, 2),
                   feet=None if m.feet is None else round(m.feet + dy, 2))


def _next_frames(templates: list[MobTemplate], last: MobTemplate, most: int | None = None) -> list[MobTemplate]:
    """The frames a monster that `last` matched most likely shows now, facing the same way, at most `most` (by default
    REFIND_NEXT): the next frame of its pose and the one after (an animation loops), the other poses' first frames
    (standing to walking and back, a flinch: hit1), then the rest of its pose. None for a template that doesn't know
    its frame (a cut-out)."""
    most = REFIND_NEXT if most is None else most
    pose, _, i = last.frame.partition("/")
    if not pose or not i.isdigit():
        return []
    poses: dict[str, dict[int, MobTemplate]] = {}
    for t in templates:
        p, _, j = t.frame.partition("/")
        if p and j.isdigit() and t.facing == last.facing:
            poses.setdefault(p, {})[int(j)] = t
    own, i = poses.get(pose, {}), int(i)
    n = max(own, default=i) + 1
    order = [own.get((i + 1) % n), own.get((i + 2) % n)] + [poses[p].get(0) for p in sorted(poses) if p != pose]
    order += [own.get((i + k) % n) for k in range(3, n)]
    out: list[MobTemplate] = []
    for t in order:
        if t is not None and t.name != last.name and all(t.name != o.name for o in out):
            out.append(t)
    return out[:most]


def _one_opencv_thread() -> None:
    """Each of the pool's threads runs its matches by itself. With OpenCV's own threads as well (24 here),
    the whole screen took 141 ms instead of 117 (and without the pool, 307 instead of 250). OpenCV's thread
    count is the process's, not a thread's, so this sets it for the whole look: the half-size frame takes
    0.8 ms instead of 0.4, and the look without monsters is no slower (bench-look)."""
    cv2.setNumThreads(1)


def _serial(fn, items) -> list:
    return [fn(item) for item in items]


class Cover:
    """How much of each window of the frame has a kind's body colour, from the colour's integral image:
    one pass over the frame, then four lookups a window (a TM_CCORR with a template of ones did the
    same sums, one pass per template size)."""

    def __init__(self, coarse: np.ndarray):
        self.integral = cv2.integral(coarse, sdepth=cv2.CV_64F)
        self._sums: dict[tuple[int, int], np.ndarray] = {}

    def sums(self, h: int, w: int) -> np.ndarray:
        """The body colour in each h x w window, by its top-left corner (the shape of a match's result)."""
        out = self._sums.get((h, w))
        if out is None:  # two threads may both work it out: the same numbers either way
            ii = self.integral
            out = self._sums[(h, w)] = ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]
        return out


def _spots(coarse: np.ndarray, cover: Cover, template: MobTemplate, i: int) -> list[tuple[float, int, int, int]]:
    """Step 1 for one template (index `i`): where its body-colour silhouette matches the frame's body
    colour (`coarse`, at 1/COARSE size): (score, centre x and y in the frame, i) of each spot."""
    silhouette = template.silhouette
    if coarse.shape[0] < silhouette.shape[0] or coarse.shape[1] < silhouette.shape[1]:
        return []
    res = cv2.matchTemplate(coarse, silhouette, cv2.TM_CCOEFF_NORMED)
    # A window with next to no body colour has next to no variance, and its normalised score
    # is noise that can reach 1.0. In the basin, such windows took all MAX_CHECKS checks,
    # so snails and pigs at her level were never checked.
    least = MIN_COVER * float(silhouette.mean())
    # In C if it's built (native.peaks: the same, but without holding Python's lock), else in numpy.
    found = native.peaks(res, cover.integral, *silhouette.shape, least, SILHOUETTE_THRESHOLD)
    if found is None:
        res[cover.sums(*silhouette.shape) / silhouette.size < least] = 0
        ys, xs = np.nonzero((res >= SILHOUETTE_THRESHOLD) & (res == cv2.dilate(res, np.ones((3, 3), np.uint8))))
        found = ys, xs, res[ys, xs]
    return [(float(v), COARSE * x + COARSE * silhouette.shape[1] // 2,
             COARSE * y + COARSE * silhouette.shape[0] // 2, i) for y, x, v in zip(*found)]


def _to_check(spots: list[tuple[float, int, int, int]], slack: int = CHECK_SLACK) -> list[tuple[int, int, int]]:
    """Step 2's spots (centre x, y, template), best first: at most MAX_CHECKS, none within `slack` (CHECK_SLACK in
    the frame's px) of one already picked."""
    picked = []
    for _, cx, cy, i in spots:
        if len(picked) >= MAX_CHECKS:
            break
        if any(abs(cx - x) < slack and abs(cy - y) < slack for x, y, _ in picked):
            continue
        picked.append((cx, cy, i))
    return picked


def _places(spots: list[tuple[float, int, int, int]], slack: int,
            most: int | None = None) -> list[list[tuple[int, int, int]]]:
    """`_to_check`'s places, each with the templates whose spots are there, best first, at most `most` of them (by
    default FINE_TEMPLATES): for each place, its spots (centre x, y, template), the first `_to_check`'s. A spot is a
    place's when it's within `slack` of the place's first, the first such place of those `_to_check` picked (as they
    were picked, a spot near none of those already picked made one); spots further down than `_to_check` looks only
    join places. (Picked spot by spot in Python, every spot against every place, it took 4.4 ms a band.)"""
    most = FINE_TEMPLATES if most is None else most
    picked = _to_check(spots, slack)
    places = [[p] for p in picked]
    if not spots or not picked:
        return places
    a = np.array([s[1:] for s in spots])  # (x, y, template) of each spot, best first
    p = np.array(picked)
    near = (np.abs(a[:, None, 0] - p[None, :, 0]) < slack) & (np.abs(a[:, None, 1] - p[None, :, 1]) < slack)
    has, which = near.any(axis=1), near.argmax(axis=1)
    for k, place in enumerate(places):
        for cx, cy, i in a[has & (which == k)].tolist():
            if len(place) >= most:
                break
            if all(i != j for _, _, j in place):
                place.append((cx, cy, i))
    return places


def _gate(small: np.ndarray, place: list[tuple[str, MobTemplate, int, int]], slack: int, tries: int = 1):
    """The fine gate's match at a place (`_match_at` at step 1's scale, `slack` around): its first template's; with
    `tries` (a learned kind's GATE_TRIES), the first of its first `tries` templates that scores FINE_GATE, else the
    last one's."""
    found = None
    for _, t, cx, cy in place[:tries]:
        found = _match_at(small, *t.step1, cx, cy, slack)
        if found is not None and found[0] >= FINE_GATE:
            break
    return found


def _finer(template: MobTemplate, cx: int, cy: int, sx: float, sy: float) -> tuple[int, int]:
    """A step 1 spot (its centre in step 1's frame, `_spots`) in a look frame `sx` x `sy` times finer: its window's
    top-left corner scaled, plus half the template's size there."""
    sh, sw = template.silhouette.shape
    th, tw = template.rgb.shape[:2]
    left, top = cx - COARSE * sw // 2, cy - COARSE * sh // 2
    return int(round(left * sx)) + tw // 2, int(round(top * sy)) + th // 2


def _check(frame: np.ndarray, template: MobTemplate, cx: int, cy: int, slack: int = CHECK_SLACK):
    """Masked colour match of one template centred within `slack` (CHECK_SLACK in the frame's px) of (cx, cy):
    (score, x, y, w, h), or None if it doesn't fit in the frame there."""
    return _match_at(frame, template.rgb, template.mask3, cx, cy, slack)


def _match_at(frame: np.ndarray, tpl: np.ndarray, mask: np.ndarray, cx: int, cy: int, slack: int):
    """`_check` with a picture and its mask (uint8, one per colour channel)."""
    th, tw = tpl.shape[:2]
    x0, y0 = max(0, cx - tw // 2 - slack), max(0, cy - th // 2 - slack)
    window = frame[y0:y0 + th + 2 * slack, x0:x0 + tw + 2 * slack]
    if window.shape[0] < th or window.shape[1] < tw:
        return None
    _, score, _, (x, y) = cv2.minMaxLoc(masked_match(window, tpl, mask))
    return float(score), x0 + x + tw // 2, y0 + y + th // 2, tw, th
