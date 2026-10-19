"""Where she stands on screen and what her name tag looks like. Redo after moving to another map,
moving the game window, or changing resolution."""

import json
from dataclasses import dataclass

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


def calibrate(tag: tuple[int, int] | None = None) -> Calibration:
    """Find her name tag (or use `tag`, client pixels), pick the recording crop and save a template."""
    full = client.submit(fps=10, duration_ms=200, scale=1, lead_ms=0, name="calibrate")
    frame = full["frames"][-1]
    LAB.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(LAB / "calibrate.png")
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
    template = track.make_template(small, ((tag[0] - x) // SCALE, (tag[1] - y) // SCALE))
    np.save(TAG_TEMPLATE, template)
    CALIBRATION.write_text(json.dumps({"crop": crop, "scale": SCALE, "tag": list(map(int, tag))}))
    return Calibration(crop, SCALE, template)


def load() -> Calibration:
    if not CALIBRATION.exists():
        raise RuntimeError("not calibrated yet: run `python -m gamelab calibrate`")
    data = json.loads(CALIBRATION.read_text())
    return Calibration(data["crop"], data["scale"], np.load(TAG_TEMPLATE))
