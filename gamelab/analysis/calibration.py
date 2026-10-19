"""Where she stands on screen and what her name tag looks like: the crop recordings are taken of, and the
name tag template that finds her in them. Redo after moving to another map, moving the game window, or
changing resolution."""

import json
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..control import client
from ..paths import CALIBRATION, LAB, TAG_TEMPLATE
from . import recordings, track

CROP_W, CROP_H = 800, 400   # client pixels recorded around her
TAG_IN_CROP = (400, 300)     # where her name tag sits in that crop (room above for jumps)
SCALE = 2                    # recordings are downscaled 2x: plenty for tracking, 4x less data
TAG_MATCH_MIN = 0.7          # a saved template matching this well has found her


@dataclass
class Calibration:
    crop: list[int]          # x, y, width, height in client pixels
    scale: int
    template: np.ndarray     # her name tag, grayscale, at `scale`


def calibrate(tag: tuple[int, int] | None = None, fresh: bool = False) -> Calibration:
    """Find her (or use `tag`, client pixels), pick the recording crop around her, and save it.

    Her name tag looks the same on every map, so once a template exists it is reused to find her;
    `fresh` rebuilds it (needed after switching characters, or if a rope crossed the tag last time).
    """
    LAB.mkdir(parents=True, exist_ok=True)
    full = client.submit(fps=10, duration_ms=200, scale=1, lead_ms=0, name="calibrate")
    frame = full["frames"][-1]
    Image.fromarray(frame).save(LAB / "calibrate.png")
    template = None if fresh or not TAG_TEMPLATE.exists() else np.load(TAG_TEMPLATE)
    if tag is None:
        tag = _find_tag(frame, template)
    width, height = full["client"]["width"], full["client"]["height"]
    x = min(max(0, tag[0] - TAG_IN_CROP[0]), width - CROP_W)
    y = min(max(0, tag[1] - TAG_IN_CROP[1]), height - CROP_H)
    crop = [int(x), int(y), CROP_W, CROP_H]

    small = recordings.screenshot("calibrate", crop=crop, scale=SCALE)
    Image.fromarray(small).save(LAB / "calibrate_crop.png")
    if template is None:
        template = track.make_template(small, ((tag[0] - x) // SCALE, (tag[1] - y) // SCALE), scale=SCALE)
        np.save(TAG_TEMPLATE, template)
    CALIBRATION.write_text(json.dumps({"crop": crop, "scale": SCALE, "tag": list(map(int, tag))}))
    return Calibration(crop, SCALE, template)


def _find_tag(frame: np.ndarray, template: np.ndarray | None) -> tuple[int, int]:
    """Her name tag's centre in a full-size frame: where the saved template matches (in a half-size
    copy), or else the name tag nearest the middle."""
    if template is not None:
        (x, y), score = track.best_match(frame[::SCALE, ::SCALE], template)
        if score >= TAG_MATCH_MIN:
            return (x + template.shape[1] // 2) * SCALE, (y + template.shape[0] // 2) * SCALE
    tag = track.find_name_tag(frame)
    if tag is None:
        raise RuntimeError(f"couldn't find her name tag; look at {LAB / 'calibrate.png'} and pass --tag X,Y")
    return tag


def load() -> Calibration:
    """The saved calibration (raises if there is none yet)."""
    if not CALIBRATION.exists():
        raise RuntimeError("not calibrated yet: run `python -m gamelab calibrate`")
    data = json.loads(CALIBRATION.read_text())
    return Calibration(data["crop"], data["scale"], np.load(TAG_TEMPLATE))
