"""Where she stands on screen and what her name tag looks like. Redo after moving to another map,
moving the game window, or changing resolution."""

import json
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from . import client, track
from .paths import CALIBRATION, LAB, TAG_TEMPLATE

CROP_W, CROP_H = 800, 400   # client pixels recorded around her
TAG_IN_CROP = (400, 300)     # where her name tag sits in that crop (room above for jumps)
SCALE = 2                    # recordings are downscaled 2x: plenty for tracking, 4x less data


@dataclass
class Calibration:
    crop: list[int]
    scale: int
    template: np.ndarray

    @property
    def tag_width(self) -> int:
        return self.template.shape[1]


def _locate(frame_small: np.ndarray, template: np.ndarray) -> tuple[tuple[int, int], float]:
    """Centre of the best template match in a scale-2 frame, and its match score."""
    gray = cv2.cvtColor(np.ascontiguousarray(frame_small), cv2.COLOR_RGB2GRAY)
    _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED))
    return (x + template.shape[1] // 2, y + template.shape[0] // 2), score


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
    if tag is None and template is not None:
        (sx, sy), score = _locate(frame[::SCALE, ::SCALE], template)
        if score >= 0.7:
            tag = (sx * SCALE, sy * SCALE)
    if tag is None:
        tag = track.find_name_tag(frame)
        if tag is None:
            raise RuntimeError(f"couldn't find her name tag; look at {LAB / 'calibrate.png'} and pass --tag X,Y")
    width, height = full["client"]["width"], full["client"]["height"]
    x = min(max(0, tag[0] - TAG_IN_CROP[0]), width - CROP_W)
    y = min(max(0, tag[1] - TAG_IN_CROP[1]), height - CROP_H)
    crop = [int(x), int(y), CROP_W, CROP_H]

    small = client.submit(fps=10, duration_ms=200, crop=crop, scale=SCALE, lead_ms=0, name="calibrate")["frames"][-1]
    Image.fromarray(small).save(LAB / "calibrate_crop.png")
    if template is None:
        template = track.make_template(small, ((tag[0] - x) // SCALE, (tag[1] - y) // SCALE), scale=SCALE)
        np.save(TAG_TEMPLATE, template)
    CALIBRATION.write_text(json.dumps({"crop": crop, "scale": SCALE, "tag": list(map(int, tag))}))
    return Calibration(crop, SCALE, template)


def load() -> Calibration:
    if not CALIBRATION.exists():
        raise RuntimeError("not calibrated yet: run `python -m gamelab calibrate`")
    data = json.loads(CALIBRATION.read_text())
    return Calibration(data["crop"], data["scale"], np.load(TAG_TEMPLATE))
