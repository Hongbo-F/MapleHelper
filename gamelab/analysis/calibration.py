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
TAG_SLACK = 40               # client px: her name tag this close to TAG_IN_CROP in the stored crop, it's kept (the
                             # runs of a sweep that end where they started then share one crop)


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
    x, y, _, _ = crop = crop_around(tag, full["client"]["width"], full["client"]["height"])

    small = recordings.screenshot("calibrate", crop=crop, scale=SCALE)
    Image.fromarray(small).save(LAB / "calibrate_crop.png")
    if template is None:
        template = track.make_template(small, ((tag[0] - x) // SCALE, (tag[1] - y) // SCALE), scale=SCALE)
        np.save(TAG_TEMPLATE, template)
    CALIBRATION.write_text(json.dumps({"crop": crop, "scale": SCALE, "tag": list(map(int, tag))}))
    return Calibration(crop, SCALE, template)


def crop_around(tag: tuple[int, int], width: int, height: int) -> list[int]:
    """The recording crop (client px: x, y, width, height) with her name tag's centre `tag` at TAG_IN_CROP, kept
    inside a client `width` x `height`."""
    x = min(max(0, tag[0] - TAG_IN_CROP[0]), width - CROP_W)
    y = min(max(0, tag[1] - TAG_IN_CROP[1]), height - CROP_H)
    return [int(x), int(y), CROP_W, CROP_H]


def match_tag(frame: np.ndarray, template: np.ndarray) -> tuple[tuple[int, int] | None, float]:
    """Her name tag's centre in a full-size frame, where the saved template matches it (in a half-size copy)
    TAG_MATCH_MIN or better (else None), and how well the best match did."""
    (x, y), score = track.best_match(frame[::SCALE, ::SCALE], template)
    if score < TAG_MATCH_MIN:
        return None, score
    return ((x + template.shape[1] // 2) * SCALE, (y + template.shape[0] // 2) * SCALE), score


def crop_on_her(cal: Calibration, frame: np.ndarray, width: int, height: int) -> tuple[list[int], str]:
    """Where to record her: the crop around her name tag in `frame` (full size, of a client `width` x `height`,
    as the game shows it now), where the saved template finds it, unless the stored crop has it within TAG_SLACK of
    where calibrate puts it; where it isn't found, the stored crop. And what to tell the user: nothing when the
    stored crop is kept; else where she is and whether the stored one had her, or that her tag wasn't found. The
    stored crop is calibrate's, around wherever she stood then: in game on 2026-10-21 the 16 `act --record` runs
    of a sweep never had her in their crop, from a calibration she had walked away from (its outcomes were read
    off the screenshots after each run instead)."""
    tag, score = match_tag(frame, cal.template)
    if tag is None:
        return list(cal.crop), (f"her name tag wasn't found on screen (the best match {score:.2f}, under "
                                f"{TAG_MATCH_MIN}): recording the stored crop {list(cal.crop)}, which may not have "
                                f"her (run calibrate where she stands)")
    x, y, w, h = cal.crop
    if abs(tag[0] - x - TAG_IN_CROP[0]) <= TAG_SLACK and abs(tag[1] - y - TAG_IN_CROP[1]) <= TAG_SLACK:
        return list(cal.crop), ""
    crop = crop_around(tag, width, height)
    had = "had her too" if x <= tag[0] < x + w and y <= tag[1] < y + h else "hasn't her"
    return crop, f"her name tag is at {tag[0]},{tag[1]}: recording {crop} around it (the stored {list(cal.crop)} {had})"


def crop_now(cal: Calibration) -> tuple[list[int], str]:
    """crop_on_her on a full-size frame of the game as it is now: one short capture that sends no keys."""
    full = client.submit(fps=10, duration_ms=100, scale=1, lead_ms=0, name="find_her")
    return crop_on_her(cal, full["frames"][-1], full["client"]["width"], full["client"]["height"])


def _find_tag(frame: np.ndarray, template: np.ndarray | None) -> tuple[int, int]:
    """Her name tag's centre in a full-size frame: where the saved template matches (in a half-size
    copy), or else the name tag nearest the middle."""
    if template is not None and (tag := match_tag(frame, template)[0]) is not None:
        return tag
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
