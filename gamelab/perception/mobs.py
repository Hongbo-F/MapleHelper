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

A monster followed is re-found first only as far around where it should be as it can have got since it was last seen
(`near_pad`), and a window whose pixels are the very same as at its last re-find gives the last answer (`refind`'s
`memo`): fewer matches a frame, master's answers on every frame compared (the memo exactly; near first wherever no
place outside its window outscores one at NEAR_GOOD inside it: README, Fewer re-find matches). A frame's re-finds go to the C
together (`refind_all`: every window's matches in one call on threads of its own, the rest here as each comes back),
and a twin kind's check side by side keeps what its pictures alone give (`_pick_here`): the same answers, in less time
(README, One call for the re-finds).

A search's step 1 is in C where it's built (gamelab/search.c, SEARCH_C: `_InC`): the band shrunk, the colour tests, each
kind's coarse picture and integrals, each silhouette scored only where the cover test keeps a window and OpenCV's score
could reach SILHOUETTE_THRESHOLD, exactly (integer sums, where OpenCV's float32 differ by up to 1.1e-6 on the saved
frames), the peaks and the places, in two calls a band; the fine gate's and step 2's matches in a few calls to native.c's
re-finds (`_matches`). Where OpenCV's noise could make a unit's peaks or their order other than the exact scores' (two
neighbours' scores, a score and the threshold, or two of its spots within NOISE_K's bound of each other), those shapes
are matched as master matches them (`_InC._master`): the answers are master's by construction, not only on the frames
compared (README, A cheaper whole screen).
"""

import ctypes
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

from .. import native, paths
from ..screen import STRETCHED
from . import match
from .gates import FOG_ALPHAS, Gates, kind_histograms, learn
from .match import best_match, best_matches, suppress_overlaps


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
# Step 1 of a band in C (gamelab/search.c, native.search_lib: the band shrunk, every kind's colour test, its coarse
# picture and integrals, each silhouette scored only where the cover test keeps a window and OpenCV's score could reach
# SILHOUETTE_THRESHOLD, worked out exactly, the peaks, the places), and the fine gate's and step 2's masked matches of
# a band in a few calls to native.c's re-finds (rf_run, one picture over its own window a job, its best whatever it
# scores: `_matches`), where each was a call through Python: `_InC`. Where OpenCV's float32 noise could decide master's
# peaks or their order (NOISE_K), those shapes are matched as master matches them (`_InC._master`): master's answers
# (README, A cheaper whole screen). False: numpy, OpenCV and Python as before.
SEARCH_C = True
# How far OpenCV's step 1 score (cv2.matchTemplate's TM_CCOEFF_NORMED of the float32 coarse picture) can be from the
# exact one at a window: NOISE_K n sqrt(S_II S_TT) / sqrt(A B) + NOISE_ABS (search.c's bound: the window's colour and
# the silhouette's energy over the score's normalisation, n its cells). Measured (2026-10-28, D:/MapleHelper-scratch/
# sfast/finish/probe/noise2.py: every window OpenCV puts at 0.30 or more and its 8 neighbours, the exact scores from
# the integer sums; 5.40 M on 354 saved frames of 01:01's crowd and 00:25's and 23:15's before/, the whole screen's
# bands and her band, and 0.56 M on the review's inputs of West Rocky Mountain II's detector over them, its sprites
# blended in): |OpenCV - exact| at most 1.10e-6 on the saved frames and 1.05e-5 on those; NOISE_ABS and 4.0 x 2^-24 of
# the energy term cover every one on the saved frames, 7.5 x 2^-24 on those (S_IT for the energy took 5.2 and 11.4: no
# one constant). So the bound is 4x the worst seen. Where a peak by master's rule, or the order of two of a unit's
# spots, depends on which side of its bound OpenCV's score falls, master's answer is OpenCV's noise's, and those shapes
# are worked out as master does (`_InC._master`); MobDetector.noise_over counts, there, any peak where OpenCV's lay
# outside. A crowd's whole screen 22.6 ms of CPU on one thread without the check, 24.4 with it, 26.4 with twice the
# bound (sfast/finish/runs/cost_crowd_s.json, 200 frames, Idle); on the frame loop, hunting in a crowd, 0.01-0.03 of a
# core more, a whole screen out 1.7-3.5 ms later and her band's search 1.5-1.9 (sfast/finish/runs/report_replay.md).
NOISE_K = 32 * 2.0 ** -24
NOISE_ABS = 2.5e-7
# With step 1 in C a whole screen's bands go on this many threads at once (`detect_wide`: a pool of their own, beside
# the THREADS one master's way uses; each makes its threads only as it needs them), her band on the caller's (two
# C calls and a few rf_run calls, nothing for a pool). A crowd's whole screen (2026-10-27, D:/MapleHelper-scratch/sfast/
# adopt/runs/timing: 300 frames of 01:01's crowd, medians, CPUs 12-23 56-63% busy with other work): on 1 thread 20.0 ms
# of CPU, on 2 12.8 ms wall (23.2 of CPU), on 3 10.1 (23.7), on 6 8.5 (24.6); master's on its 6 42.8 (157), on 1 115.
SEARCH_THREADS = 3
CHECK_ROWS = 4    # search.c: a window's exact products this many rows at a time between the early tests
STEP1_PEAKS = 1 << 15  # search.c: a band's peaks at most (a crowd's whole screen has 329 in all its bands)
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
# A monster followed is looked for first only as far around where it should be as it can have got since it was last
# seen (`near_pad`: turned at once at its kind's fastest walk, twice that walk over the time, and NEAR_MARGIN px more),
# and in the REFIND_PAD window only where that finds no picture at REFIND_GOOD, or another kind's monster: there, the
# same pictures in the same order, it's the whole window's answer as before (README, Fewer re-find matches). Not
# THRESHOLD: a picture cut by the near window's edge, the monster further off, can score 0.6-0.9. Where master's
# re-finds found them on 01:01's before/ (2026-10-26, D:/MapleHelper-scratch/pfast/pfr: ident.py --stats, ana.py), a
# monster's feet were 4 px from where they should be (median), 16-17 at 90% and 22 at 99%, its frames 0.1 s apart as
# saved or each held 3 times (15 and 22 at 1/30 s: the motion of 0.1 s comes on every third): the tracker walks it on
# as it was, these mushrooms hop and turn, and her minimap dot's steps move them all on screen. At 0.1 s since its last
# sighting the near window is no smaller than REFIND_PAD's (4 + 24 px against 23 at 1366x768: nothing changes there);
# at 1/30 s it's 12 px against 23, 625 places against 2,209. Of 0-10 px tried on the held frames, a margin of 4 found
# 65% of the live re-finds there (57-78%), its windows' places about 0.66 of today's (0.59-0.84). Where a track's last
# re-find found it only outside its near window (or not at all), the next goes to the whole window at once (`_far`): on
# the service's frame loop replayed on the saved frames stamped 1/30 s apart, their motion 0.1 s's (3 times live's a
# frame), near first without it took 2.28 cores for 28.9 frames a second against master's 2.08 for 29.8; with it 1.89
# for 30.2 against 1.95 for 30.2 (pfast/tools/replay.py, the crowd's [700:1300] of 01:01; README).
NEAR_MARGIN = 4
# A near window's answer is taken only at NEAR_GOOD or more (as rounded), else the REFIND_PAD window as before. Over
# REFIND_GOOD alone, a monster further off than the near window could be answered otherwise than by the whole window:
# there its last picture scored over REFIND_GOOD outside the near window, where in it the near window went on to a
# frame it may have gone to. Drawn on a frame of 01:01 (2026-10-27, the review of perception-fast: D:/MapleHelper-
# scratch/pfast/review-correct/near_edge2.py, Tree Dungeon's pictures and the frames they most likely went to set on
# their feet from 2 px inside to 5 px past the near window, 8 ways), 37 scenes of 11,040 were answered otherwise, all
# Horny Mushrooms 10 px or more off up or down: in 25 the near window took another frame at 0.902-0.919 where the
# whole window found 0.977-1.0; in 12 the near window's was the better, 0.977-1.0 where the whole window's first
# picture over REFIND_GOOD scored 0.906-0.919. At 0.95, 12 of the 11,040 (the 12 better ones): the 25 are the whole
# window's answers. On the saved frames 0.18-0.34 near answers a frame scored under 0.95 (01:01's crowd stamped 1/30 s
# apart, and each held 3 times; 00:25 stamped 1/30 s: 0.23) and go on to the whole window too; the states master's
# on every one of both Lie Detectors' before/ either way (pfast/fix-reviews; README, Perception, faster re-finds).
NEAR_GOOD = 0.95
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
    `threads`: how many matches run at once (1: all on the caller's thread; with step 1 in C, SEARCH_THREADS at most:
    a whole screen's bands, one a thread, on a pool of their own). `frame_scale`: the frames' px per game px
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
        # Master's matches on `threads`; with step 1 in C a whole screen's bands on SEARCH_THREADS of their own (each
        # pool makes its threads as its jobs come: the one not in use has none). Which one a search takes is asked each
        # time (`detect_wide`), so SEARCH_C turned off, or a plan search.c doesn't take, has master's threads.
        self._pool = ThreadPoolExecutor(threads, "monster-search", _one_opencv_thread) if threads > 1 else None
        bands = min(threads, SEARCH_THREADS)
        self._bands = ThreadPoolExecutor(bands, "monster-search-bands", _one_opencv_thread) if bands > 1 else None
        self._c: dict = {}         # `only` -> its step 1 in C (_InC), None where the C can't
        self._c_tables: dict = {}  # id(template) -> (it, its silhouette made ready for the C, or None: `_tables`)
        self._c_local = threading.local()  # each thread's re-find contexts for the gate's and checks' matches
        self._c_making = threading.Lock()
        self.fallbacks = 0  # re-finds that went on to the kind's whole set (refind) ...
        self.recalled = 0   # ... and windows whose answer was the last one's (refind's `memo`) ...
        self.redone = 0     # ... step 1's shapes worked out as master does, its noise able to decide (_InC._master) ...
        self.noise_over = 0  # ... of their peaks, those where OpenCV's score lay further than NOISE_K's bound: none yet
        self._counting = threading.Lock()  # ... counted by re-finds on several threads at once (worldstate's pool)
        native.lib()  # built (the first time, a few seconds) and loaded now, not in the first look
        for t in (t for ts in self.templates.values() for t in ts):  # made ready for the C's masked match now too
            match.prepared(t.rgb, t.mask3)
            if t.step1 is not None:
                match.prepared(*t.step1)
        # The re-finds in one call in C (`refind_all`): every picture by its index here, the pictures of each re-find
        # (`_lists_of`: kind, template, facing) as runs of indices in one array, the C's context by thread count.
        self._rf_table = [t for ts in self.templates.values() for t in ts]
        self._rf_at = {id(t): i for i, t in enumerate(self._rf_table)}
        self._rf_index: dict[tuple, tuple[int, int, int, int, int]] = {}
        self._rf_flat: list[int] = []
        self._rf_lists = np.zeros(1, np.int32)
        self._rf: dict[int, native.Refinds | None] = {}
        self._rf_making = threading.Lock()  # (the C's contexts and lists made by one re-find step at a time)
        self._picking: dict = {}  # what `_its_kind` works out of the pictures alone, kept (`_pick_here`)

    def detect(self, frame: np.ndarray, rows: tuple[int, int] | None = None) -> list[Mob]:
        """Monsters in `frame` (RGB, a look frame at `frame_scale`). `rows` limits the search to a band."""
        return self._picked(frame, rows, self._detect(frame, rows, self._map))

    def detect_wide(self, frame: np.ndarray, rows: tuple[int, int]) -> list[Mob]:
        """Monsters over a tall stretch of the frame, band by band. Over the whole screen, green
        foliage took all of the slime candidate spots and three slimes in plain view were missed;
        each band gets its own best spots. The bands are searched at once, each on one thread (THREADS of them, or
        SEARCH_THREADS with step 1 in C)."""
        # A band starts every WIDE_STEP rows until one reaches rows[1] (it's cut short there); always one.
        band, step = round(WIDE_BAND * self.px), round(WIDE_STEP * self.px)
        band = step + max(band - step, self.overlap)
        starts = range(rows[0], max(rows[0] + 1, rows[1] - band + step), step)
        in_c = SEARCH_C and native.search_lib() is not None and self._step1_in_c(None) is not None
        pool = self._bands if in_c and self._pool is not None else self._pool

        def one(y0):
            return self._detect(frame, (y0, min(rows[1], y0 + band)), _serial)

        bands = list(pool.map(one, starts)) if pool is not None else [one(y0) for y0 in starts]
        return self._picked(frame, rows, suppress_overlaps([hit for band in bands for hit in band]))

    def refind(self, frame: np.ndarray, box: tuple[int, int, int, int], kind: str, template: str = "",
               facing: int = 0, bounds: tuple[int, int, int, int] | None = None, fallback: bool = True,
               near: int | None = None, memo: dict | None = None) -> Mob | None:
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
        None. On the caller's thread: a window is small, and the pool's hand-offs cost more (the feed's step runs a
        frame's re-finds at once, each on a thread of its own pool: localization.worldstate, REFIND_THREADS).
        `near` (px): looked for first only that far around the box (`near_pad`: as far as it can have got since it was
        last seen), the same way; the REFIND_PAD window as above only where that finds no picture at NEAR_GOOD, or
        another kind's monster. `memo`: a dict the caller keeps for this monster from one re-find to the next (one a
        track): a window at the same place, looked at with the same pictures, whose pixels are the very same as last
        time gives what it gave then (`_recall`) without matching again; and where the last re-find found it only
        outside its near window, or not at all, this one goes to the whole window at once (`_far`)."""
        h, w = frame.shape[:2]
        bounds = bounds or (0, 0, w, h)
        bx1, by1 = bounds[2], bounds[3]
        pad = round(REFIND_PAD * self.px)
        full = _window(box, pad, bounds)
        if full is None or kind not in self.templates:
            return None
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        seen = (kind, template, facing)  # what the windows are looked at with (`_recall`)
        nearer = near is not None and near < pad
        if nearer and not _far(memo):  # first only as far around as it can have got (`near_pad`)
            small = _window(box, near, bounds)
            if small is not None:
                best = self._looked(frame, small, seen, memo, "near")
                # (over REFIND_GOOD as rounded: a score that rounds to 0.9 may be under it, and the matches went on to
                # the next pictures. Taken at 0.9, a held frame of 01:01 kept one at 0.900 at the near window's edge,
                # where its next picture, a px taller, scored 0.99 in the whole window. And NEAR_GOOD or more.)
                if best is not None and best.score > REFIND_GOOD and best.score >= NEAR_GOOD \
                        and self._its_kind(frame[:by1, :bx1], best):
                    return best
        best = self._looked(frame, full, seen, memo, "full")
        if best is not None and not self._its_kind(frame[:by1, :bx1], best):
            best = None  # another kind's monster: not this one
        if nearer:
            _note_far(memo, best is None or not _inside(best, box, near))
        if best is not None or not fallback:
            return best
        with self._counting:
            self.fallbacks += 1
        return self._fallback(frame, full, kind, box, bounds, memo)

    def _fallback(self, frame: np.ndarray, full: tuple[int, int, int, int], kind: str, box: tuple[int, int, int, int],
                  bounds: tuple[int, int, int, int], memo: dict | None) -> Mob | None:
        """A re-find's last resort (`refind`): every template of its kind over the whole window `full`, as `detect`
        does; of those found that are its kind, the one nearest the box's centre, or None."""
        x0, y0, x1, y1 = full
        bx1, by1 = bounds[2], bounds[3]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        known, found = _recall(memo, "whole", full, kind, frame)
        if known:
            self._recalled()
        else:
            found = [_moved(m, x0, y0) for m in self._detect(frame[y0:y1, x0:x1], None, _serial, (kind,))]
            _keep(memo, "whole", full, kind, frame, found)
        found = [m for m in found if m.kind == kind and self._its_kind(frame[:by1, :bx1], m)]
        return min(found, key=lambda m: (m.x - cx) ** 2 + (m.y - cy) ** 2) if found else None

    def refind_ahead(self, frame: np.ndarray, asks: list[tuple], bounds: tuple[int, int, int, int], threads: int,
                     token: int, more: tuple[int, int]) -> bool:
        """The re-finds of `refind_all` sent to the GPU ahead of it (native.Refinds.ahead), before the frame has placed
        her: each monster asked (box, kind, template, facing, id) where its picture should be by the frame before's
        camera, its full window `more` px wider each side (x) and taller (y) than `refind_all`'s, so that it holds the
        one `refind_all` will look in where the camera moved no more. `token`: the frame's, for `refind_all`'s
        `ahead`. Whether they were sent: only with the GPU in use; the answers are the same either way."""
        rf = self._in_c(threads)
        if rf is None or not asks or not rf.gpu_ready():
            return False
        pad = round(REFIND_PAD * self.px)
        jobs = (native.RfJob * len(asks))()
        n = 0
        for box, kind, template, facing, tid in asks:
            if kind not in self.templates:
                continue
            x0, y0 = max(bounds[0], int(box[0]) - pad - more[0]), max(bounds[1], int(box[1]) - pad - more[1])
            x1, y1 = min(bounds[2], int(box[2]) + pad + more[0]), min(bounds[3], int(box[3]) + pad + more[1])
            if x1 <= x0 or y1 <= y0:
                continue
            job = jobs[n]
            job.id = tid
            job.key, job.first_at, job.first_n, job.next_at, job.next_n = self._lists_of(kind, template, facing)
            job.wfull[:] = (x0, y0, x1, y1)
            n += 1
        return n > 0 and rf.ahead(frame, (native.RfJob * n).from_buffer(jobs), self._rf_lists, token)

    def refind_all(self, frame: np.ndarray, asks: list[tuple], bounds: tuple[int, int, int, int] | None = None,
                   threads: int = 1, pool=None, ahead: int | None = None) -> list[Mob | None] | None:
        """`refind` of each monster asked, (box, kind, template, facing, fallback, near, memo, id) as there (`id`: the
        caller's for it, from frame to frame: its windows' memos, one a monster), with the same answers: every window's
        matches (`_matched`'s, near and full) in one call in C on `threads` threads (native.Refinds, rf_start: Python's
        lock let go; the contract in native.c), the rest here as in `refind`, each monster's as the C hands it over,
        while it goes on with the others (`_its_kind`; `_note_far`; the fallback, on `pool`, an Executor, if one is
        given; the near window's answer passing NEAR_GOOD but not its kind, its whole window in a second call after).
        `ahead`: the frame's token, where its re-finds were sent ahead (`refind_ahead`).
        None where the C can't (not built, a picture it doesn't take, a frame it doesn't read as it is): the caller
        re-finds each with `refind`."""
        rf = self._in_c(threads)
        if rf is None:
            return None
        h, w = frame.shape[:2]
        bounds = bounds or (0, 0, w, h)
        bx1, by1 = bounds[2], bounds[3]
        pad = round(REFIND_PAD * self.px)
        n = len(asks)
        jobs = (native.RfJob * n)()
        plans: list = [None] * n  # (its full window, whether looked for nearer first) of those with one
        for i, (box, kind, template, facing, fallback, near, memo, tid) in enumerate(asks):
            job = jobs[i]
            job.id = tid
            full = _window(box, pad, bounds)
            if full is None or kind not in self.templates:
                continue  # (nothing: its memo kept)
            job.key, job.first_at, job.first_n, job.next_at, job.next_n = self._lists_of(kind, template, facing)
            job.wfull[:] = full
            nearer = near is not None and near < pad
            if nearer and not _far(memo):
                small = _window(box, near, bounds)
                if small is not None:
                    job.wnear[:] = small
            plans[i] = (full, nearer)
        outs = rf.start(frame, jobs, self._rf_lists, THRESHOLD, REFIND_GOOD, near_good=NEAR_GOOD, ahead=ahead)
        if outs is None:
            return None
        found: list[Mob | None] = [None] * n
        again, alone, last = [], [], []  # (to look at in their whole windows yet; the C couldn't; to the fallback)
        futures = {}
        recalled = 0

        def whole(i: int, o) -> None:  # (its whole window's answer, as `refind` takes it)
            box, kind, _, _, fallback, near, memo, _ = asks[i]
            best = self._rf_mob(kind, o, 1) if o.tpl[1] >= 0 else None
            if best is not None and not self._its_kind(frame[:by1, :bx1], best):
                best = None  # another kind's monster: not this one
            if plans[i][1]:
                _note_far(memo, best is None or not _inside(best, box, near))
            if best is not None or not fallback:
                found[i] = best
            elif pool is not None:
                futures[i] = pool.submit(self._fallback, frame, plans[i][0], kind, box, bounds, memo)
            else:
                last.append(i)

        try:
            while (i := rf.next()) >= 0:
                o = outs[i]
                if plans[i] is None:
                    continue
                recalled += bool(o.looked & 2) + bool(o.looked & 8)
                if o.looked & 16:
                    alone.append(i)
                elif o.found == 1:
                    best = self._rf_mob(asks[i][1], o, 0)
                    if self._its_kind(frame[:by1, :bx1], best):
                        found[i] = best
                    else:
                        again.append(i)  # (another kind's monster near: its whole window, as `refind` goes on to)
                else:
                    whole(i, o)
        finally:
            rf.end()
        if again:
            more = (native.RfJob * len(again))()
            for k, i in enumerate(again):
                ctypes.memmove(ctypes.byref(more[k]), ctypes.byref(jobs[i]), ctypes.sizeof(native.RfJob))
                more[k].wnear[:] = (0, 0, 0, 0)
            outs2 = rf.run(frame, more, self._rf_lists, THRESHOLD, REFIND_GOOD, keep=True, near_good=NEAR_GOOD)
            for k, i in enumerate(again):
                if outs2 is None or outs2[k].looked & 16:
                    alone.append(i)
                else:
                    recalled += bool(outs2[k].looked & 8)
                    whole(i, outs2[k])
        with self._counting:
            self.recalled += recalled
            self.fallbacks += len(last) + len(futures)
        for i in alone:  # (the C couldn't: these as before; never so far)
            box, kind, template, facing, fallback, near, memo, _ = asks[i]
            found[i] = self.refind(frame, box, kind, template, facing, bounds, fallback, near, memo)
        for i in last:
            box, kind, _, _, _, _, memo, _ = asks[i]
            found[i] = self._fallback(frame, plans[i][0], kind, box, bounds, memo)
        for i, f in futures.items():
            found[i] = f.result()
        return found

    def _in_c(self, threads: int) -> "native.Refinds | None":
        """The C's re-finds on `threads` threads (`refind_all`), made the first time (each picture's index: its place
        in `_rf_table`); None where it can't be: the DLL or AVX2 not there, a picture the C doesn't take."""
        rf = self._rf.get(threads)
        if rf is None and threads not in self._rf:
            with self._rf_making:
                if threads not in self._rf:
                    ready = [match.prepared(t.rgb, t.mask3) for t in self._rf_table]
                    rf = native.Refinds(ready, threads) if ready and all(r is not None for r in ready) else None
                    if rf is not None and rf.ctx is None:
                        rf = None
                    self._rf[threads] = rf
                rf = self._rf[threads]
        return rf

    def _lists_of(self, kind: str, template: str, facing: int) -> tuple[int, int, int, int, int]:
        """A re-find's pictures for the C (`refind_all`; `_matched`'s, as indices into `_rf_table`, in `_rf_lists`):
        (their number, the first pictures' start and count, the next frames' start and count). Made once each (one
        step at a time: `_rf_making`)."""
        key = (kind, template, facing)
        got = self._rf_index.get(key)
        if got is not None:
            return got
        with self._rf_making:
            return self._listed(key)

    def _listed(self, key: tuple) -> tuple[int, int, int, int, int]:
        """`_lists_of`'s, made under `_rf_making` (the array before the index: a step that finds its key finds its
        pictures in `_rf_lists`)."""
        kind, template, facing = key
        got = self._rf_index.get(key)
        if got is None:
            first, last = [], None
            for t in self.templates[kind]:
                if not template or t.name != template or (facing and t.facing and t.facing != facing):
                    continue
                last = last or t
                first.append(self._rf_at[id(t)])
            after = [] if self.search == 1 or last is None else [
                self._rf_at[id(t)] for t in _next_frames(self.templates[kind], last)]
            at = len(self._rf_flat)
            self._rf_flat += first + after
            got = (len(self._rf_index), at, len(first), at + len(first), len(after))
            self._rf_lists = np.array(self._rf_flat or [0], np.int32)
            self._rf_index[key] = got
        return got

    def _rf_mob(self, kind: str, o, side: int) -> Mob:
        """The monster a window's best picture in the C's answer `o` (native.RfOut; `side` 0 near, 1 full) is, as
        `_matched` makes it."""
        t = self._rf_table[o.tpl[side]]
        th, tw = t.rgb.shape[:2]
        return _seen(kind, t, o.x[side] + tw // 2, o.y[side] + th // 2, tw, th, round(float(o.score[side]), 3))

    def _looked(self, frame: np.ndarray, win: tuple[int, int, int, int], seen: tuple, memo: dict | None,
                slot: str) -> Mob | None:
        """`_matched` in window `win`, or what it gave the last time (`memo[slot]`: the same window, the same
        pictures, the very same pixels)."""
        known, best = _recall(memo, slot, win, seen, frame)
        if known:
            self._recalled()
            return best
        best = self._matched(frame, win, *seen)
        _keep(memo, slot, win, seen, frame, best)
        return best

    def _recalled(self) -> None:
        with self._counting:
            self.recalled += 1

    def _matched(self, frame: np.ndarray, win: tuple[int, int, int, int], kind: str, template: str,
                 facing: int) -> Mob | None:
        """A re-find's matches in window `win` (x0, y0, x1, y1) of `frame` (`refind`): the picture that matched it
        last (`template`, `facing`; both ways a cut-out that doesn't know its facing), and on a frame finer than step
        1's, where that scores under REFIND_GOOD, the frames its animation most likely went to, in turn until one
        scores REFIND_GOOD (`_next_frames`). The best that scores THRESHOLD, or None."""
        x0, y0, x1, y1 = win
        window = frame[y0:y1, x0:x1]

        def fits(t: MobTemplate) -> bool:
            return window.shape[0] >= t.rgb.shape[0] and window.shape[1] >= t.rgb.shape[1]

        best, last = None, None
        first = []  # (its last template, or both ways a cut-out that doesn't know its facing)
        for t in self.templates[kind]:
            if not template or t.name != template or (facing and t.facing and t.facing != facing):
                continue
            last = last or t
            if fits(t):
                first.append(t)
        # Each list's matches in one call where the masked match is in C (match.best_matches: the window loaded once),
        # their answers taken in turn as before.
        for t, (score, (x, y)) in zip(first, best_matches(window, [(t.rgb, t.mask3) for t in first])):
            if score >= THRESHOLD and (best is None or score > best.score):
                th, tw = t.rgb.shape[:2]
                best = _seen(kind, t, x0 + x + tw // 2, y0 + y + th // 2, tw, th, round(float(score), 3))
        # A hidden track's (without `fallback`) too. Tried with its last template only (2026-10-26: 6.4 ms a frame less
        # on the saved frames of 01:01), a crowd's tracks went otherwise: on 600 frames of 00:25, 142 frames' states
        # differed and 2 kills more were hers, given to tracks left hidden 10-20 s at one x (D:/MapleHelper-scratch/
        # cpu/review-safety/perc; README, CPU).
        if self.search != 1 and last is not None and (best is None or best.score < REFIND_GOOD):
            # (finer than step 1's: its frames differ)
            after = [t for t in _next_frames(self.templates[kind], last) if fits(t)]
            pictures = [(t.rgb, t.mask3) for t in after]
            for t, (score, (x, y)) in zip(after, best_matches(window, pictures, stop=REFIND_GOOD)):
                if score >= THRESHOLD and (best is None or score > best.score):
                    th, tw = t.rgb.shape[:2]
                    best = _seen(kind, t, x0 + x + tw // 2, y0 + y + th // 2, tw, th, round(float(score), 3))
                    if score >= REFIND_GOOD:
                        break
        return best

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
        return _pick_kind(frame, m, self.twins[m.kind], self.templates, 0, self.aligned, self._picking).kind == m.kind

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
        kinds alone (default: every kind). Step 1 and the matches in C where it can (SEARCH_C: `_InC`, `_matches`),
        else in numpy and OpenCV (`_step1`, `run`)."""
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
        slack = round(CHECK_SLACK * self.spx)
        c = self._step1_in_c(only) if SEARCH_C and native.search_lib() is not None else None
        # Step 1, every template of every kind (one a shape: `units`); then the spots each kind checks (kinds that
        # share a body, together); then step 2, all at once, every template of each shape picked. In C the places
        # come from it (sr_places), or its units where master's noise could decide one of them (`_InC.step1`).
        got = c.step1(small, slack if self.search != 1 else None) if c is not None else None
        places, units = got if got is not None else (None, None)
        checks = []
        if places is None:
            places = []
            for kinds, members, shapes, found in (units if units is not None else self._step1(small, run, only)):
                spots = sorted((s for f in found for s in f), reverse=True)
                learned = kinds[0] in self.learned
                if self.search == 1:
                    picked = (_to_check(spots, slack) if not learned
                              else [spot for place in _places(spots, slack, PER_PLACE) for spot in place])
                    checks += [(*members[j], cx, cy) for cx, cy, i in picked for j in shapes[i]]
                else:
                    places += _unit_places(learned, members, shapes, spots, slack)
        rf = self._contexts() if SEARCH_C and native.lib() is not None else None  # (the gate's, the checks')
        at = self._rf_at
        if self.search == 1:
            results = None if rf is None else _matches(rf[1], rf[2], frame, [
                (at[id(t)], *t.rgb.shape[:2], cx, cy, slack) for _, t, cx, cy in checks])
            if results is None:
                results = run(lambda c: _check(frame, c[1], c[2], c[3], slack), checks)
        else:  # the gate at step 1's scale, then each place's templates at the look frame's, around their own spots
            passed = None if rf is None else _gated(rf[0], rf[2], at, small, places, slack)
            if passed is None:
                gate = run(lambda p: _gate(small, p[1], slack, GATE_TRIES if p[0] else 1), places)
                passed = [g is not None and g[0] >= FINE_GATE for g in gate]
            checks = [(kind, t, *_finer(t, cx, cy, sx, sy)) for (_, _, every), ok in zip(places, passed)
                      if ok for kind, t, cx, cy in every]
            results = None if rf is None else _matches(rf[1], rf[2], frame, [
                (at[id(t)], *t.rgb.shape[:2], cx, cy, FINE_SLACK) for _, t, cx, cy in checks])
            if results is None:
                results = run(lambda c: _check(frame, c[1], c[2], c[3], FINE_SLACK), checks)
        hits = []
        for (kind, template, *_), result in zip(checks, results):
            if result and result[0] >= THRESHOLD:
                score, x, y, w, h = result
                hits.append(_seen(kind, template, x, y_off + y, w, h, round(score, 3)))
        return suppress_overlaps(hits)

    def _units(self, only=None):
        """The units a search looks for (`units`), as (kinds, members, shapes): with `only`, the members of those
        kinds alone, each its own shape (a unit with none of them, not at all)."""
        for kinds, members, shapes in self.units:
            if only is not None and any(k not in only for k in kinds):
                members = [(k, t) for k, t in members if k in only]
                shapes = [[i] for i in range(len(members))]
                if not members:
                    continue
            yield kinds, members, shapes

    def _step1(self, small: np.ndarray, run, only=None) -> list[tuple]:
        """Step 1 in numpy and OpenCV, on step 1's frame (its size a multiple of COARSE): each unit looked for (its
        colour's pixels at least MIN_BODY_PIXELS), as (kinds, members, shapes, each shape's spots: `_spots`)."""
        channels = split_channels(small)
        coarse_size = (small.shape[1] // COARSE, small.shape[0] // COARSE)
        looked = None  # every pixel's learned kinds (gates.Gates.lookup), worked out once for them all
        units = []  # (its kinds, their templates as (kind, template), shapes, their colour at 1/COARSE size, its cover)
        for kinds, members, shapes in self._units(only):
            if kinds[0] in self.learned:
                looked = self.gates.lookup(*channels) if looked is None else looked
                colour = self.gates.of(kinds, looked)
            else:
                colour = palette(kinds[0], small, channels)
            if colour.sum() < MIN_BODY_PIXELS * self.spx * self.spx:
                continue
            coarse = cv2.resize(colour.astype(np.float32), coarse_size, interpolation=cv2.INTER_AREA)
            units.append((kinds, members, shapes, coarse, Cover(coarse)))
        jobs = [(coarse, cover, members[shape[0]][1], i) for _, members, shapes, coarse, cover in units
                for i, shape in enumerate(shapes)]
        found = run(lambda job: _spots(*job), jobs)
        out, start = [], 0
        for kinds, members, shapes, _, _ in units:
            out.append((kinds, members, shapes, found[start:start + len(shapes)]))
            start += len(shapes)
        return out

    def _step1_in_c(self, only=None) -> "_InC | None":
        """Step 1 in C for `only` (`_InC`), made the first time; None where the C can't (not built, no AVX2, a kind
        without a colour test it knows, a silhouette it doesn't take)."""
        got = self._c.get(only, False)
        if got is False:
            with self._c_making:
                got = self._c.get(only, False)
                if got is False:
                    got = self._c[only] = _InC.made(self, only)
        return got

    def _contexts(self):
        """This thread's re-find contexts for the gate's and checks' matches (`_matches`): (step 1's pictures' (None at
        step 1's own scale), the look frame's, every picture's index), each picture by its place in `_rf_table`; made
        at its first search, and gone with the thread. None where the C can't (not built, no AVX2, a picture it doesn't
        take)."""
        got = getattr(self._c_local, "rf", False)
        if got is False:
            got = None
            table = self._rf_table
            full = [match.prepared(t.rgb, t.mask3) for t in table]
            small = [match.prepared(*t.step1) for t in table] if self.search != 1 else []
            if table and all(p is not None for p in full) and all(p is not None for p in small):
                cf = native.Refinds(full, 1)
                cs = native.Refinds(small, 1) if small else None
                if cf.ctx is not None and (cs is None or cs.ctx is not None):
                    got = (cs, cf, np.arange(len(table), dtype=np.int32))
            self._c_local.rf = got
        return got


# Step 1 in C (search.c's colour_of and colour16: mobs.PALETTES' hand-made tests, by kind; tests compare them on every
# colour). A kind with a hand-made test not here is searched as before.
_HAND_TESTS = {"slime": 1, "pig": 2, "snail": 3, "orange-mushroom": 4}
_PLACES_ROOM = 1 << 16  # sr_places' answer, ints


class _Tables(NamedTuple):
    """A shape's silhouette made ready for search.c (`_tables`): native.SrShape's fields and the arrays they point at,
    and the bounds' tables a group of its size takes (`_InC`)."""
    th: int
    tw: int
    kmin: int          # the least colour (quarters of a cell) a window needs to pass the cover test
    st: int            # S_T, in T_int (T x 2^shift)
    A: float           # n S_TT - S_T^2 in T_int^2, rounded once
    rtt: float         # sqrt(S_TT), T_int units (the noise bound's: NOISE_K)
    shift: int
    rows: np.ndarray   # T, th rows of tw4 doubles (zeros past tw)
    ub: np.ndarray     # [th, 4 tw + 1] int64: a row's most S_I4T with s4 quarters of colour in it
    vb: np.ndarray     # [tw, 4 th + 1] int64: a column's
    order: np.ndarray  # the rows, most of T's mass first (int32)


def _tables(silhouette: np.ndarray) -> _Tables | None:
    """A silhouette (float32, at 1/COARSE of step 1's scale: MobTemplate.silhouette) made ready for search.c: its
    values as integers T_int = T x 2^shift (each float32 a multiple of 2^-shift: exact), its sums, the cover test's
    least colour (`_spots`' and native.peaks' test, worked out the same way for every count of quarters), each row's
    and each column's bound (the most its dot product with I4 can be for each sum of I4 in it: 4s on its largest values,
    the rest on the next one), its rows by mass. None where search.c doesn't take it: larger than 64 x 256, its exact
    sums over 2^52 (or its products' sums over 2^61), or about flat (OpenCV's score is 1 at every window then, where
    its variance is under DBL_EPSILON: under 1e-12 here)."""
    th, tw = silhouette.shape
    n = th * tw
    if not n or th > 64 or tw > 256:
        return None
    values = silhouette.astype(np.float64)
    if not np.isfinite(values).all() or values.min() < 0:
        return None
    shift = max(float(v).as_integer_ratio()[1].bit_length() - 1 for v in np.unique(values))
    scaled = values * 2.0 ** shift  # (a power of two: exact)
    ti = scaled.astype(np.int64)
    if not np.array_equal(ti.astype(np.float64), scaled):
        return None
    top = int(ti.max())
    if 4 * n * top >= 1 << 52 or 4 * n * n * top >= 1 << 61:
        return None
    flat = [int(a) for a in ti.ravel()]
    st = sum(flat)
    A = n * sum(a * a for a in flat) - st * st
    if A <= 0 or A / (n * n) / 4.0 ** shift < 1e-12:
        return None
    least = MIN_COVER * float(silhouette.mean())  # (as _spots: its cover test, s / size < least, s = k / 4 exactly)
    ok = ~((np.arange(4 * n + 1, dtype=np.float64) / 4) / float(n) < least)
    kmin = int(np.argmax(ok)) if ok.any() else 4 * n + 1
    rows = np.zeros((th, (tw + 3) & ~3), np.float64)
    rows[:, :tw] = values

    def bounds(m: np.ndarray) -> np.ndarray:  # each row of m: its most sum(I4 x m) for each sum of I4 (0..4 each)
        k = m.shape[1]
        desc = -np.sort(-m, axis=1)
        pre = np.zeros((m.shape[0], k + 1), np.int64)
        pre[:, 1:] = np.cumsum(desc, axis=1)
        s4 = np.arange(4 * k + 1)
        q, rem = s4 // 4, s4 % 4
        padded = np.concatenate([desc, np.zeros((m.shape[0], 1), np.int64)], axis=1)
        return np.ascontiguousarray(4 * pre[:, q] + rem * padded[:, q])

    order = np.lexsort((np.arange(th), -ti.sum(axis=1))).astype(np.int32)
    rtt = math.sqrt(sum(a * a for a in flat))
    return _Tables(th, tw, kmin, st, float(A), rtt, shift, rows, bounds(ti), bounds(ti.T), order)


def _ceil_scaled(v: np.ndarray, shift: int, k: int) -> np.ndarray:
    """ceil(v x 2^(k - shift)) of non-negative int64s, exactly."""
    return v << (k - shift) if k >= shift else -((-v) >> (shift - k))


def _group_table(bounds: list[np.ndarray], shifts: list[int]) -> tuple[np.ndarray, int]:
    """A group's shapes' bounds (each [rows, s4] in its T_int units) side by side as int16 lanes ([rows, s4, GL]),
    each rounded up to units of 2^-k of T for one k, the largest whose values fit int16: (the table, k)."""
    top = max(float(b.max()) / 2.0 ** s for b, s in zip(bounds, shifts))
    k = int(math.floor(math.log2(32767 / max(top, 1e-9))))
    while True:
        scaled = [_ceil_scaled(b, s, k) for b, s in zip(bounds, shifts)]
        if max(int(b.max()) for b in scaled) <= 32767:
            break
        k -= 1
    table = np.zeros(bounds[0].shape + (native.SR_GROUP,), np.int16)
    for q, b in enumerate(scaled):
        table[:, :, q] = b
    return table, k


class _InC:
    """A detector's step 1 for one `only` made ready for search.c (`MobDetector._step1_in_c`): its units as `_detect`
    looks for them (`MobDetector._units`), each shape's silhouette's tables (`_tables`, kept with the detector), a
    unit's shapes of one size in groups of up to native.SR_GROUP (their bounds side by side), the learned gates'
    table."""

    @classmethod
    def made(cls, det: "MobDetector", only) -> "_InC | None":
        if native.search_lib() is None:
            return None
        units = list(det._units(only))
        if len(units) > 16 or any(kinds[0] not in det.learned and kinds[0] not in _HAND_TESTS for kinds, _, _ in units):
            return None
        lut = det.gates.lut if det.gates is not None else np.zeros(1 << 15, np.uint8)
        if int(lut.max(initial=0)) >= 1 << 15:
            return None  # (more learned kinds than search.c's int16 lanes take)
        tables = []
        for _, members, shapes in units:
            for shape in shapes:
                t = members[shape[0]][1]
                got = det._c_tables.get(id(t))
                if got is None or got[0] is not t:
                    got = det._c_tables[id(t)] = (t, _tables(t.silhouette))
                if got[1] is None:
                    return None
                tables.append(got[1])
        return cls(det, units, tables, np.ascontiguousarray(lut.astype(np.uint32)))

    def __init__(self, det: "MobDetector", units: list[tuple], tables: list[_Tables], lut: np.ndarray):
        self.det, self.units, self.tables, self.lut = det, units, tables, lut
        self.n = len(tables)
        self.shapes = (native.SrShape * max(1, self.n))()
        for k, tb in enumerate(tables):
            self.shapes[k] = native.SrShape(tb.th, tb.tw, tb.rows.shape[1], tb.kmin, tb.st, tb.A, 2.0 ** tb.shift,
                                            tb.rows.ctypes.data, tb.ub.ctypes.data, tb.order.ctypes.data, CHECK_ROWS, 0,
                                            tb.rtt)
        groups, cunits, self.held, first = [], [], [], 0
        self.first = []  # each unit's first shape
        for kinds, members, shapes in units:
            gfirst = len(groups)
            by_size: dict[tuple[int, int], list[int]] = {}  # (in order of first appearance)
            for k in range(first, first + len(shapes)):
                by_size.setdefault((tables[k].th, tables[k].tw), []).append(k)
            for ks in by_size.values():
                for c0 in range(0, len(ks), native.SR_GROUP):
                    chunk = ks[c0:c0 + native.SR_GROUP]
                    shifts = [tables[k].shift for k in chunk]
                    ug, ku = _group_table([tables[k].ub for k in chunk], shifts)
                    vg, kv = _group_table([tables[k].vb for k in chunk], shifts)
                    g = native.SrGroup()
                    g.ng = len(chunk)
                    for q, k in enumerate(chunk):
                        g.idx[q] = k
                        g.su[q] = 2.0 ** (shifts[q] - ku)
                        g.sv[q] = 2.0 ** (shifts[q] - kv)
                    g.ug, g.vg = ug.ctypes.data, vg.ctypes.data
                    groups.append(g)
                    self.held.append((ug, vg))
            learned = kinds[0] in det.learned
            test, bits = (0, det.gates.bits(kinds)) if learned else (_HAND_TESTS[kinds[0]], 0)
            cunits.append(native.SrUnit(test, bits, MIN_BODY_PIXELS * det.spx * det.spx, first, len(shapes), gfirst,
                                        len(groups) - gfirst))
            self.first.append(first)
            first += len(shapes)
        self.groups = (native.SrGroup * max(1, len(groups)))(*groups)
        self.cunits = (native.SrUnit * max(1, len(cunits)))(*cunits)
        self.nu = len(cunits)
        self.sh = np.array([tb.th for tb in tables] or [0], np.int32)
        self.sw = np.array([tb.tw for tb in tables] or [0], np.int32)
        self.ufirst = np.array(self.first or [0], np.int32)
        self.un = np.array([len(shapes) for _, _, shapes in units] or [0], np.int32)

    def peaks(self, small: np.ndarray) -> tuple | None:
        """sr_step1 over step 1's frame (its size a multiple of COARSE): (each shape's peaks' count, -1 for a unit not
        looked for; each shape's flag (search.c's SR_UNSURE: master's peaks there OpenCV's noise's);
        their rows, columns, scores and noise bounds, shape after shape; each unit's I4, (h / COARSE, w / COARSE) a
        unit, None for a band without pixels), or None where the C declines."""
        h, w = small.shape[:2]
        out_n = np.zeros(max(1, self.n), np.int32)
        flags = np.zeros(max(1, self.n), np.int32)
        ys, xs = np.empty(STEP1_PEAKS, np.int64), np.empty(STEP1_PEAKS, np.int64)
        vs, es = np.empty(STEP1_PEAKS, np.float32), np.empty(STEP1_PEAKS, np.float32)
        if not h or not w:  # (no pixels: none of any colour, every unit under MIN_BODY_PIXELS)
            out_n[:] = -1
            return out_n, flags, ys[:0], xs[:0], vs[:0], es[:0], None
        if small.dtype != np.uint8 or small.ndim != 3 or small.shape[2] != 3:
            return None
        if small.strides[1] != 3 or small.strides[2] != 1 or small.strides[0] <= 0:
            small = np.ascontiguousarray(small)
        i4 = np.empty((max(1, self.nu), h // COARSE, w // COARSE), np.uint8)
        total = native.search_lib().sr_step1(
            small.ctypes.data, h, w, small.strides[0], self.lut.ctypes.data, self.cunits, self.nu, self.shapes, self.n,
            self.groups, SILHOUETTE_THRESHOLD, NOISE_K, NOISE_ABS, out_n.ctypes.data, flags.ctypes.data,
            ys.ctypes.data, xs.ctypes.data, vs.ctypes.data, es.ctypes.data, STEP1_PEAKS, i4.ctypes.data, None)
        if total < 0:
            return None
        return out_n, flags, ys[:total], xs[:total], vs[:total], es[:total], i4

    def step1(self, small: np.ndarray, slack: int | None) -> tuple | None:
        """Step 1 of a band from the C's peaks, master's answer. With `slack` (a frame finer than step 1's): (its places
        as `_detect` makes them, each `_places` of a unit looked for as (learned, the gate's (kind, template, cx, cy),
        every template's), None), by sr_places, but for a unit whose spots master's float32 noise could make other than
        the exact scores' (`_unsure`): its spots worked out as master does (`_master`), placed as master places them.
        Without: (None, its units as `MobDetector._step1` gives them, (kinds, members, shapes, each shape's spots as
        `_spots` makes them, to the type), an unsure one's as master makes them). None where the C declines."""
        got = self.peaks(small)
        if got is None:
            return None
        out_n, flags, ys, xs, vs, es, i4 = got
        looked, unsure = self._unsure(out_n, flags, vs, es, ys, xs)
        if slack is not None:
            by_unit = self._sr_places(out_n, ys, xs, vs, slack)
            if by_unit is not None:
                for u, these in unsure.items():
                    kinds, members, shapes = self.units[u]
                    spots = self._master(u, self._spots_of(u, looked[u], out_n, ys, xs, vs, es), these, i4)
                    by_unit[u] = _unit_places(kinds[0] in self.det.learned, members, shapes,
                                              sorted((s for f in spots for s in f), reverse=True), slack)
                return [p for u in looked for p in by_unit[u]], None
        units = []
        for u, at in looked.items():
            kinds, members, shapes = self.units[u]
            spots = self._spots_of(u, at, out_n, ys, xs, vs, es)
            units.append((kinds, members, shapes, self._master(u, spots, unsure[u], i4) if u in unsure
                          else [[s[:4] for s in f] for f in spots]))
        return None, units

    def spots(self, small: np.ndarray) -> list[tuple] | None:
        """`MobDetector._step1`'s answer (`step1`'s units); None where the C declines."""
        got = self.step1(small, None)
        return None if got is None else got[1]

    def places(self, small: np.ndarray, slack: int) -> list[tuple] | None:
        """`_detect`'s places on a frame finer than step 1's (`step1`'s); None where the C declines."""
        got = self.step1(small, slack)
        if got is None:
            return None
        places, units = got
        if places is None:  # (sr_places declined: its answer's room)
            places = [p for kinds, members, shapes, found in units for p in _unit_places(
                kinds[0] in self.det.learned, members, shapes, sorted((s for f in found for s in f), reverse=True),
                slack)]
        return places

    def _unsure(self, out_n, flags, vs, es, ys, xs) -> tuple[dict, dict]:
        """({each unit looked for: its first peak's index}, {each unit whose spots master's float32 noise could make
        other than the exact scores': those shapes}): search.c's flags (a shape's peaks), and the two shapes of any two
        of a unit's spots, as `_places` sorts them, whose scores lie within their bounds of each other (their order)."""
        looked, unsure, at = {}, {}, 0
        for u, ((kinds, members, shapes), first) in enumerate(zip(self.units, self.first)):
            if out_n[first] < 0:
                continue
            looked[u] = at
            k = len(shapes)
            counts = out_n[first:first + k]
            these = set(np.flatnonzero(flags[first:first + k]).tolist())
            total = int(counts.sum())
            if total > 1:
                i = np.repeat(np.arange(k), counts)
                cx = COARSE * xs[at:at + total] + COARSE * self.sw[first + i] // 2
                cy = COARSE * ys[at:at + total] + COARSE * self.sh[first + i] // 2
                v, e = vs[at:at + total].astype(np.float64), es[at:at + total].astype(np.float64)
                o = np.lexsort((i, cy, cx, v))[::-1]  # (sorted(reverse=True) of the (score, cx, cy, i) tuples)
                close = v[o[:-1]] - v[o[1:]] <= e[o[:-1]] + e[o[1:]]
                if close.any():
                    these.update(i[o[:-1][close]].tolist())
                    these.update(i[o[1:][close]].tolist())
            if these:
                unsure[u] = these
            at += total
        return looked, unsure

    def _spots_of(self, u: int, at: int, out_n, ys, xs, vs, es) -> list[list[tuple]]:
        """A unit's shapes' spots as `_spots` makes them (to the type), each with its noise bound fifth; `at`: its first
        peak's index."""
        kinds, members, shapes = self.units[u]
        first, spots = self.first[u], []
        for i, shape in enumerate(shapes):
            c = int(out_n[first + i])
            silhouette = members[shape[0]][1].silhouette
            spots.append([(float(v), COARSE * x + COARSE * silhouette.shape[1] // 2,
                           COARSE * y + COARSE * silhouette.shape[0] // 2, i, float(e))
                          for y, x, v, e in zip(ys[at:at + c], xs[at:at + c], vs[at:at + c], es[at:at + c])])
            at += c
        return spots

    def _master(self, u: int, spots: list[list[tuple]], unsure: set, i4: np.ndarray) -> list[list[tuple]]:
        """A unit's spots where master's float32 noise could make some other (`_unsure`'s shapes): those shapes matched
        as master's step 1 matches them (`_spots`, on the unit's coarse picture: I4 / COARSE^2, what cv2.resize's
        INTER_AREA makes of its colour, to the bit); then any other shape with a spot whose exact score lies within its
        bound of one of theirs (master's now) as `_places` sorts them, until none: every two spots are then in master's
        order, each shape's peaks master's. Counted (MobDetector.redone), and where master's score at a peak both have
        lies further from the exact one than its bound (MobDetector.noise_over: the bound's check on live frames)."""
        kinds, members, shapes = self.units[u]
        coarse = i4[u].astype(np.float32) / np.float32(COARSE * COARSE)
        cover = Cover(coarse)
        done: dict[int, list] = {}
        todo, over = set(unsure), 0
        while todo:
            for i in todo:
                done[i] = _spots(coarse, cover, members[shapes[i][0]][1], i)
                exact = {(s[1], s[2]): s for s in spots[i]}
                for s in done[i]:
                    c = exact.get((s[1], s[2]))
                    over += c is not None and abs(s[0] - c[0]) > c[4]
            every = sorted([s + (0.0,) for i in done for s in done[i]] +
                           [s for i, f in enumerate(spots) if i not in done for s in f], reverse=True)
            todo = set()
            for p, q in zip(every, every[1:]):
                if (p[4] or q[4]) and p[0] - q[0] <= p[4] + q[4]:
                    todo.update(i for i in (p[3], q[3]) if i not in done)
        with self.det._counting:
            self.det.redone += len(done)
            self.det.noise_over += over
        return [done[i] if i in done else [s[:4] for s in f] for i, f in enumerate(spots)]

    def _sr_places(self, out_n, ys, xs, vs, slack: int) -> dict | None:
        """Each unit looked for's places by sr_places (as `_places` makes them from the C's spots), by the unit's index;
        None where it declines."""
        ufirst = self.ufirst.copy()
        for u in range(self.nu):
            if out_n[ufirst[u]] < 0:
                ufirst[u] = -1
        out = np.empty(_PLACES_ROOM, np.int32)
        n = native.search_lib().sr_places(ys.ctypes.data, xs.ctypes.data, vs.ctypes.data, out_n.ctypes.data,
                                          self.sh.ctypes.data, self.sw.ctypes.data, ufirst.ctypes.data,
                                          self.un.ctypes.data, self.nu, slack, MAX_CHECKS, FINE_TEMPLATES,
                                          out.ctypes.data, _PLACES_ROOM)
        if n < 0:
            return None
        o, k, by_unit = out[:n].tolist(), 0, {}
        for u, (kinds, members, shapes) in enumerate(self.units):
            if ufirst[u] < 0:
                continue
            learned = kinds[0] in self.det.learned
            count, k = o[k], k + 1
            places = by_unit[u] = []
            for _ in range(count):
                ne, k = o[k], k + 1
                place = [(o[k + 3 * e], o[k + 3 * e + 1], o[k + 3 * e + 2]) for e in range(ne)]
                k += 3 * ne
                places.append((learned, [(*members[shapes[i][0]], cx, cy) for cx, cy, i in place],
                               [(*members[j], cx, cy) for cx, cy, i in place for j in shapes[i]]))
        return by_unit


_RF_JOBS = 1024  # native.c's RF_MAX_JOBS: a call's jobs at most


def _matches(rf: "native.Refinds", lists: np.ndarray, image: np.ndarray, asks: list[tuple]) -> list | None:
    """Each ask (its picture's index in `rf`'s, the picture's height and width, cx, cy, slack) matched as `_match_at`
    matches it over `image` (its window `slack` around, cut at the image's edges; a picture that doesn't fit it, None):
    (score, x, y, w, h), x and y of `_match_at`'s types. All in one rf_run call, each a job of one picture over its
    window, its best whatever it scores (threshold -2: mm_best's, the first highest score row by row). None where the C
    does nothing (the caller matches them its own way)."""
    n = len(asks)
    if not n:
        return []
    if n > _RF_JOBS:
        return None
    out: list = [None] * n
    jobs = (native.RfJob * n)()
    ih, iw = image.shape[:2]
    which = []
    for k, (idx, th, tw, cx, cy, slack) in enumerate(asks):
        x0, y0 = max(0, cx - tw // 2 - slack), max(0, cy - th // 2 - slack)  # (`_match_at`'s, to the type)
        x1, y1 = min(iw, x0 + tw + 2 * slack), min(ih, y0 + th + 2 * slack)
        if y1 - y0 < th or x1 - x0 < tw:
            continue
        j = jobs[len(which)]
        j.id, j.key = len(which), idx
        j.wfull[:] = (int(x0), int(y0), int(x1), int(y1))
        j.first_at, j.first_n = idx, 1
        which.append((k, x0, y0, th, tw))
    if not which:
        return out
    if len(which) < n:
        jobs = (native.RfJob * len(which)).from_buffer(jobs)
    outs = rf.run(image, jobs, lists, -2.0, 2.0)
    if outs is None:
        return None
    for j, (k, x0, y0, th, tw) in enumerate(which):
        o = outs[j]
        if o.tpl[1] < 0 or o.looked & 16:
            return None  # (no best: a score that isn't a number; never so far)
        out[k] = (float(o.score[1]), x0 + (int(o.x[1]) - int(x0)) + tw // 2, y0 + (int(o.y[1]) - int(y0)) + th // 2,
                  tw, th)
    return out


def _gated(rf: "native.Refinds", lists: np.ndarray, at: dict, small: np.ndarray, places: list[tuple],
           slack: int) -> list[bool] | None:
    """Whether each place passes the fine gate (`_gate`'s answer at FINE_GATE: any of its first GATE_TRIES templates,
    for a learned kind's place, else its first, scoring FINE_GATE on step 1's frame), in a call for each try
    (`_matches`); None where the C does nothing."""
    passed = [False] * len(places)
    todo = [(p, 0) for p in range(len(places))]
    while todo:
        asks = []
        for p, k in todo:
            _, t, cx, cy = places[p][1][k]
            asks.append((at[id(t)], *t.step1[0].shape[:2], cx, cy, slack))
        got = _matches(rf, lists, small, asks)
        if got is None:
            return None
        nxt = []
        for (p, k), g in zip(todo, got):
            if g is not None and g[0] >= FINE_GATE:
                passed[p] = True
            elif k + 1 < min(len(places[p][1]), GATE_TRIES if places[p][0] else 1):
                nxt.append((p, k + 1))
        todo = nxt
    return passed


def step1_frame(frame: np.ndarray, search: float) -> np.ndarray:
    """A look frame (or a band of it) as step 1 sees it, `search` times smaller (MobDetector.search): shrunk to
    SEARCH_SCALE (area averaged: in C where it can, search.c's sr_shrink, the same bytes as cv2.resize), unless it's
    no finer (as it is)."""
    if search == 1:
        return frame
    h, w = frame.shape[:2]
    size = (max(1, round(w / search)), max(1, round(h / search)))
    out = native.shrink_area(frame, *size) if SEARCH_C else None
    return out if out is not None else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


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
               slack: int, aligned: dict | None = None, cache: dict | None = None) -> Mob:
    """Monster `m` as the kind, of its own and `twins`, whose picture of its frame (facing its way, set on its feet,
    matched `slack` px around; with no slack, there alone, shifted to where it lies on `m`'s picture by `aligned`:
    MobDetector.aligned) it is. A picture with parts of its own (OWN_PARTS: an Axe Stump's axe) is the monster's
    only where those parts match too (OWN_MATCH), and then it is, over the pictures it holds (a Stump's); otherwise
    the one that scores best (Horny or Green Mushroom, Stump or Dark Stump: their colours). With no slack and a
    `cache` (a dict kept with the templates), the same answer by `_pick_here`."""
    if not m.frame or m.ox is None:
        return m
    if not slack and cache is not None:
        return _pick_here(frame, m, twins, templates, aligned, cache)
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


PICKING_MOST = 20000  # entries `_pick_here` keeps (a few hundred on a map's pictures; past this, it starts again)


def _pick_here(frame: np.ndarray, m: Mob, twins: tuple[str, ...], templates: dict[str, list[MobTemplate]],
               aligned: dict | None, cache: dict) -> Mob:
    """`_pick_kind` with no slack (`_its_kind`'s, every re-find of a twin kind's), the same answer to the bit: what
    depends on the pictures alone worked out once and kept in `cache` (the pictures to set beside `m`'s, by its kind,
    picture and facing; each picture's mask and its side of `_score`, its colours there less their mean and their sum
    of squares; for each layout of the pictures found, relative to each other, each one's own parts and its side of
    `_score` over them), the frame's side of each score worked out as there."""
    if len(cache) > PICKING_MOST:
        cache.clear()
    key = ("plan", m.kind, m.template, m.facing, m.frame, twins)
    plan = cache.get(key)
    if plan is None:
        own = next((t for t in templates.get(m.kind, ()) if t.name == m.template and t.facing == m.facing), None)
        beside = []  # (kind, picture, its shift: `aligned`)
        if own is not None:
            for kind in twins:
                for t in templates.get(kind, ()):
                    if t.frame != m.frame or t.facing != m.facing or t.origin is None:
                        continue
                    beside.append((kind, t, *(aligned or {}).get((own.name, own.facing, t.name), (0, 0))))
        plan = cache[key] = (own, beside)
    own, beside = plan
    if own is None:
        return m
    th, tw = own.rgb.shape[:2]
    cands = [(m, own, int(m.x) - tw // 2, int(m.y) - th // 2)]  # (as that kind, its picture, its top-left)
    for kind, t, dx, dy in beside:
        th, tw = t.rgb.shape[:2]
        cx, cy = int(round(m.ox - t.origin[0])) + tw // 2 + dx, int(round(m.feet - t.origin[1])) + th // 2 + dy
        x0, y0 = cx - tw // 2, cy - th // 2
        if x0 < 0 or y0 < 0 or y0 + th > frame.shape[0] or x0 + tw > frame.shape[1]:
            continue  # (`_match_here`: not wholly in the frame)
        mask, side = _picture_side(t, cache)
        score = _score_with(frame[y0:y0 + th, x0:x0 + tw], mask, side)
        cands.append((_seen(kind, t, cx, cy, tw, th, round(score, 3)), t, x0, y0))
    if len(cands) == 1:
        return m
    layout = ("layout",) + tuple((id(t), x0 - cands[0][2], y0 - cands[0][3]) for _, t, x0, y0 in cands)
    parts_of = cache.get(layout)
    if parts_of is None:
        parts_of = []
        for i, (_, t, x0, y0) in enumerate(cands):
            mask = _picture_side(t, cache)[0]
            others = np.zeros(mask.shape, np.uint8)
            for j, (_, o, ox0, oy0) in enumerate(cands):
                if j != i:
                    _stamp(others, _picture_side(o, cache)[0], ox0 - x0, oy0 - y0)
            parts = mask & (cv2.dilate(others, np.ones((3, 3), np.uint8)) == 0)
            parts_of.append(None if parts.sum() < OWN_PARTS * mask.sum() else (parts, _side(t.rgb, parts)))
        cache[layout] = parts_of
    kept, bigger = [], []
    for (seen, t, x0, y0), own_parts in zip(cands, parts_of):
        if own_parts is None:
            kept.append(seen)
            continue
        parts, side = own_parts
        patch = frame[y0:y0 + parts.shape[0], x0:x0 + parts.shape[1]]
        if patch.shape[:2] == parts.shape and _score_with(patch, parts, side) >= OWN_MATCH:
            bigger.append(seen)
    return max(bigger or kept or [m], key=lambda c: c.score)


def _picture_side(t: MobTemplate, cache: dict) -> tuple[np.ndarray, tuple[np.ndarray, float]]:
    """Picture t's mask (one channel, bool) and its side of `_score` over it (`_side`), kept in `cache`."""
    key = ("picture", id(t))
    got = cache.get(key)
    if got is None or got[0] is not t:
        mask = t.mask3[..., 0] > 0
        got = cache[key] = (t, mask, _side(t.rgb, mask))
    return got[1], got[2]


def _side(tpl: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """A picture's side of `_score` over `mask`: its colours there less their mean (float64), their sum of squares."""
    t = tpl[mask].astype(np.float64)
    t -= t.mean(0)
    return t, float((t * t).sum())


def _score_with(patch: np.ndarray, mask: np.ndarray, side: tuple[np.ndarray, float]) -> float:
    """`_score` of a picture on `patch` over `mask`, its side worked out already (`_side`): the same, to the bit."""
    t, tt = side
    i = patch[mask].astype(np.float64)
    i -= i.mean(0)
    den = float(np.sqrt(tt * float((i * i).sum())))
    return float((t * i).sum()) / den if den > 0 else -1.0


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


def near_pad(speed: float, dt: float) -> int:
    """How far (look-frame px) around where its picture should be a monster followed is looked for first (`refind`'s
    `near`): as far as it can have gone the wrong way since it was last seen `dt` s ago, walking at most `speed`
    (look-frame px/s; the tracker's prediction walks it on as it was: turned at once, it's twice that far off), and
    NEAR_MARGIN px more."""
    return NEAR_MARGIN + int(math.ceil(2 * speed * max(0.0, dt)))


def _window(box: tuple[int, int, int, int], pad: int, bounds: tuple[int, int, int, int]):
    """`box` (x0, y0, x1, y1) `pad` px larger each way, within `bounds`: (x0, y0, x1, y1), or None if that's empty."""
    x0, y0 = max(bounds[0], int(box[0]) - pad), max(bounds[1], int(box[1]) - pad)
    x1, y1 = min(bounds[2], int(box[2]) + pad), min(bounds[3], int(box[3]) + pad)
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)


def _inside(m: Mob, box: tuple[int, int, int, int], pad: int) -> bool:
    """Whether monster `m`'s picture lies inside `box` `pad` px larger each way (where a window that size finds it)."""
    x0, y0 = m.x - m.w // 2, m.y - m.h // 2
    return (x0 >= int(box[0]) - pad and y0 >= int(box[1]) - pad and x0 + m.w <= int(box[2]) + pad
            and y0 + m.h <= int(box[3]) + pad)


def _far(memo: dict | None) -> bool:
    """Whether the last re-find (`memo`) went on to the whole window and found the monster outside the near one, or not
    at all: then the next goes to the whole window at once (`_note_far`)."""
    return memo is not None and memo.get("far", False)


def _note_far(memo: dict | None, far: bool) -> None:
    if memo is not None:
        memo["far"] = far


def _recall(memo: dict | None, slot: str, win: tuple[int, int, int, int], key, frame: np.ndarray) -> tuple[bool, object]:
    """What `memo[slot]` kept (`_keep`) if it was of the same window `win` of a frame, looked at with the same `key`,
    and `frame` there has the very same pixels: (True, it); else (False, None)."""
    if memo is None:
        return False, None
    kept = memo.get(slot)
    if kept is None or kept[0] != win or kept[1] != key:
        return False, None
    x0, y0, x1, y1 = win
    if not np.array_equal(kept[2], frame[y0:y1, x0:x1]):
        return False, None
    return True, kept[3]


def _keep(memo: dict | None, slot: str, win: tuple[int, int, int, int], key, frame: np.ndarray, answer) -> None:
    """What window `win` of `frame`, looked at with `key`, gave (`answer`), kept in `memo[slot]` with its pixels."""
    if memo is not None:
        x0, y0, x1, y1 = win
        memo[slot] = (win, key, frame[y0:y1, x0:x1].copy(), answer)


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


def _unit_places(learned: bool, members: list, shapes: list, spots: list, slack: int) -> list[tuple]:
    """A unit's places as `_detect` checks them on a frame finer than step 1's: each of `_places` of its spots (sorted,
    best first) as (learned, the gate's (kind, template, cx, cy), every template's)."""
    return [(learned, [(*members[shapes[i][0]], cx, cy) for cx, cy, i in place],
             [(*members[j], cx, cy) for cx, cy, i in place for j in shapes[i]]) for place in _places(spots, slack)]


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
    score, (x, y) = best_match(window, tpl, mask)
    return float(score), x0 + x + tw // 2, y0 + y + th // 2, tw, th
