"""Her attack's MISS over a monster, off a frame: the popup the game draws over a monster her star didn't hurt (the
user, 2026-10-28, asked whether it should be read: "Yes, read it"). behavior.keycheck counts a regular press followed by
a MISS over its target as one that took effect (the key works: it missed), not as one that did nothing.

The picture is the game's own (the copy's Effect.wz BasicEff.img NoRed0/Miss, 98x38, its origin at the bottom centre:
assets/misses, cut by `cut_assets`): MISS in the fill of her damage numbers (red at the top through orange to yellow,
OpenCV hue 0-27, saturated), a white rim inside a black one. Her damage numbers are the same fill (the same set's
digits), and so are the orange undersides of Tree Dungeon's platforms and its maple leaves: what tells MISS is its
shape. The violet popups over her (NoViolet: a monster's touch, MISS or damage) are another set and another reader's
(perception.touches, the touch watch), not this one's.

How (`MissReader.find`, in boxes of a look's frame, the hunt's: above her presses' targets): the fill's px (FILL: hue,
saturation and value, the JPEG's or a fade's slack), their overlap with the picture's fill matched over the box
(cv2.matchTemplate's correlation of the two masks: the count of px both have, at every place; the frame's own count
there by a box sum), the best place's IoU (both / either); MISS_OK at least is a MISS there. Its place: the popup's
bottom centre in the box (the picture's origin).

Measured on the saved frames of 2026-10-24 to 28 at 1366x768 (the Lie Detectors' `before` folders and the rolling
recorder's saves of 10-25 and 10-26: 26,968 frames, about 37 minutes of her hunting at 10-15 a second, JPEG 75; every blob
of the fill 30-160 px wide a box, 350,459 boxes: D:/MapleHelper-scratch/akey2/miss): 5 MISS popups in 4 places (10-24
23:48:52 and 23:50:30, two at once there, Lucky Seven's two stars; 10-25 00:52:23; 10-26 23:10:08), each read in 2-8
frames at 0.73-0.96; of the 24 frames showing one at full opacity 23 read (the other 0.59), of the 40 showing one at all,
fading too, 25. Nothing else read: the best of the rest 0.55 (a hand of the Lie Detector's picture), two damage numbers
overlapping 0.52, her numbers and the platforms' undersides 0.45 at the most. Drawn into a frame at 100%, 80% and 60%
opacity and saved as JPEG the picture reads 0.94, 0.80, 0.24: it's a popup's first half second that reads. Of the
four, one was her Lucky Seven's at a Horny Mushroom (telemetry 20261026-230001-tour), one over a monster beside another
player with no attack of hers near (another player's popup is the same picture: behavior.keycheck counts one only over
her press's target, in its time), one with no telemetry. Her hits rarely miss there at her levels.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..paths import ASSETS

FOLDER = ASSETS / "misses"
PICTURE = "NoRed0_Miss.png"
FILL = (27, 175, 110, 190)  # the fill's px: OpenCV hue at most 27 or at least 175 (red wraps), saturation and value at
                            # least these (a fresh popup: 120 and 238 and up; the slack for JPEG and the fade's first steps)
MISS_OK = 0.6               # IoU of the fill against the picture's at least: MISS (live 0.87-0.96, nothing else over 0.55)
SLACK = 6                   # px a box is padded by (the popup drifts up about 30 px a second, the camera eases)


@dataclass(frozen=True)
class Popup:
    """A MISS seen: its bottom centre (look-frame px: the picture's origin) and its IoU."""
    x: float
    y: float
    score: float


def fill_mask(rgb: np.ndarray) -> np.ndarray:
    """The px of `rgb` (RGB) in her damage numbers' fill (FILL)."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[:, :, :3]), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue_hi, hue_lo, sat, val = FILL
    return ((h <= hue_hi) | (h >= hue_lo)) & (s >= sat) & (v >= val)


class MissReader:
    """Finds the MISS picture in boxes of a frame, at `scale` look-frame px a game px (1 at 1366x768; the stretched
    1920x1080's half-size looks 0.703)."""

    def __init__(self, folder: Path = FOLDER, scale: float = 1.0):
        pic = cv2.imread(str(Path(folder) / PICTURE), cv2.IMREAD_UNCHANGED)
        if pic is None:
            raise FileNotFoundError(Path(folder) / PICTURE)
        rgba = np.ascontiguousarray(pic[:, :, [2, 1, 0, 3]])
        if scale != 1.0:
            rgba = cv2.resize(rgba, (max(1, round(rgba.shape[1] * scale)), max(1, round(rgba.shape[0] * scale))),
                              interpolation=cv2.INTER_AREA)
        h, w = rgba.shape[:2]
        mask = (rgba[:, :, 3] > 200) & fill_mask(rgba)
        hsv = cv2.cvtColor(np.ascontiguousarray(rgba[:, :, :3]), cv2.COLOR_RGB2HSV)
        mask &= hsv[:, :, 1] >= 120  # (the picture's own fill: its saturated px)
        ys, xs = np.nonzero(mask)
        self.top, self.left = int(ys.min()), int(xs.min())
        self.fill = np.ascontiguousarray(mask[self.top:ys.max() + 1, self.left:xs.max() + 1].astype(np.float32))
        self.count = float(self.fill.sum())
        self.size = (w, h)  # the whole picture's, its origin at the bottom centre (49 of 98, 37 of 38)
        self.origin = (w / 2.0, float(h))

    def find(self, frame: np.ndarray, boxes) -> list[Popup]:
        """The MISS popups in `boxes` of `frame` (RGB; boxes (x0, y0, x1, y1) in its px, each padded by SLACK), one a
        box at the most (its best place), MISS_OK and up."""
        out = []
        fh, fw = self.fill.shape
        for x0, y0, x1, y1 in boxes:
            x0, y0 = max(0, int(x0) - SLACK), max(0, int(y0) - SLACK)
            x1, y1 = min(frame.shape[1], int(x1) + SLACK), min(frame.shape[0], int(y1) + SLACK)
            if x1 - x0 < fw or y1 - y0 < fh:
                continue
            got = self.best(frame[y0:y1, x0:x1])
            if got is not None and got[2] >= MISS_OK:
                px, py, score = got
                out.append(Popup(x0 + px, y0 + py, score))
        return out

    def best(self, rgb: np.ndarray) -> tuple[float, float, float] | None:
        """The best place of the picture's fill in `rgb` (RGB, at least the fill's size): its bottom centre (px in
        `rgb`) and the IoU there; None if it's smaller."""
        f = fill_mask(rgb).astype(np.float32)
        fh, fw = self.fill.shape
        if f.shape[0] < fh or f.shape[1] < fw or f.sum() < self.count * MISS_OK:
            return None
        both = cv2.matchTemplate(f, self.fill, cv2.TM_CCORR)  # px of both, at each place
        ones = cv2.boxFilter(f, -1, (fw, fh), normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        ones = ones[:both.shape[0], :both.shape[1]]  # px of the frame's fill under the picture's box there
        iou = both / np.maximum(1.0, self.count + ones - both)
        _, score, _, (bx, by) = cv2.minMaxLoc(iou)
        x = bx - self.left + self.origin[0]
        y = by - self.top + self.origin[1]
        return float(x), float(y), float(score)


def cut_assets(dest: Path = FOLDER, copy: Path | None = None) -> Path:
    """The MISS picture (BasicEff.img NoRed0/Miss, its pixels where its link points) into `dest`, from the copy."""
    from PIL import Image

    from gamedata.canvas import pixels, source
    from gamedata.paths import COPY
    from gamedata.wz import Package
    copy = COPY if copy is None else Path(copy)
    tree = Package(copy / "Effect" / "Effect_000.wz").image("BasicEff.img")
    Path(dest).mkdir(parents=True, exist_ok=True)
    out = Path(dest) / PICTURE
    Image.fromarray(pixels(source(tree["NoRed0"]["Miss"], tree, copy)), "RGBA").save(out)
    return out
