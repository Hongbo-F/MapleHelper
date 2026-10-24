"""Where she stands on screen and what her name tag looks like: the crop recordings are taken of, and the
name tag template that finds her in them. Redo after moving to another map, moving the game window, or
changing resolution.

Her name tag is drawn at the game's size, so each client size (screen.Geometry) has a calibration of its own
(paths.calibrated: calibration_1366x768.json and so on; 1920x1080's are the files as they always were), its crop
and template at that size: recordings are shrunk as looks are (half size at 1920x1080, full size at the game's own
1366x768) and cover the same part of the game either way (800 x 400 client px at 1920x1080, 569 x 284 at 1366x768).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .. import paths
from ..control import client
from ..paths import LAB
from ..screen import STRETCHED, Geometry, geometry
from . import recordings, track

CROP_W, CROP_H = 800, 400   # client pixels recorded around her (at 1920x1080: Calibration.geo converts them)
TAG_IN_CROP = (400, 300)     # where her name tag sits in that crop (room above for jumps)
SCALE = 2                    # recordings are downscaled 2x at 1920x1080 (the look's scale: Geometry.look_scale):
                             # plenty for tracking, 4x less data
TAG_MATCH_MIN = 0.7          # a saved template matching this well has found her
TAG_SLACK = 40               # client px: her name tag this close to TAG_IN_CROP in the stored crop, it's kept (the
                             # runs of a sweep that end where they started then share one crop)


@dataclass
class Calibration:
    crop: list[int]          # x, y, width, height in client pixels
    scale: int
    template: np.ndarray     # her name tag, grayscale, at `scale`
    geo: Geometry = field(default=STRETCHED)  # the client size it's for


def files(geo: Geometry = STRETCHED) -> tuple:
    """The calibration's files for a client of this geometry: its crop (json), her name tag at the recordings'
    scale, and her name tag at full size (perception.nametag cuts that one on its first look)."""
    return (paths.calibrated(paths.CALIBRATION, geo.suffix), paths.calibrated(paths.TAG_TEMPLATE, geo.suffix),
            paths.calibrated(paths.TAG_FULL, geo.suffix))


def pictures(geo: Geometry = STRETCHED) -> tuple[Path, Path]:
    """What `calibrate` saves to be checked by eye, for a client of this geometry: the frame it looked at and the
    recording crop it picked (her name tag near its middle). lab/calibrate.png and calibrate_crop.png at 1920x1080,
    calibrate_1366x768.png and calibrate_crop_1366x768.png at the game's own size."""
    return LAB / f"calibrate{geo.suffix}.png", LAB / f"calibrate_crop{geo.suffix}.png"


def calibrate(tag: tuple[int, int] | None = None, fresh: bool = False) -> Calibration:
    """Find her (or use `tag`, client pixels), pick the recording crop around her, and save it, for the game window
    at the size it is now.

    Her name tag looks the same on every map, so once a template exists it is reused to find her;
    `fresh` rebuilds it (needed after switching characters, or if a rope crossed the tag last time), and the full-size
    one the next look cuts from it.
    """
    LAB.mkdir(parents=True, exist_ok=True)
    full = client.submit(fps=10, duration_ms=200, scale=1, lead_ms=0, name="calibrate")
    frame = full["frames"][-1]
    geo = geometry(full["client"]["width"], full["client"]["height"])
    scale = geo.look_scale
    cal_file, small_file, full_file = files(geo)
    looked, cropped = pictures(geo)
    Image.fromarray(frame).save(looked)
    template = None if fresh or not small_file.exists() else np.load(small_file)
    if tag is None:
        tag = _find_tag(frame, template, scale, looked)
    x, y, _, _ = crop = crop_around(tag, full["client"]["width"], full["client"]["height"], geo)

    small = recordings.screenshot("calibrate", crop=crop, scale=scale)
    Image.fromarray(small).save(cropped)
    if template is None:
        template = track.make_template(small, ((tag[0] - x) // scale, (tag[1] - y) // scale), scale=scale, geo=geo)
        np.save(small_file, template)
        full_file.unlink(missing_ok=True)  # cut again from the new one, on the next look
    cal_file.write_text(json.dumps({"crop": crop, "scale": scale, "tag": list(map(int, tag)),
                                    "client": list(geo.client)}))
    return Calibration(crop, scale, template, geo)


def crop_around(tag: tuple[int, int], width: int, height: int, geo: Geometry = STRETCHED) -> list[int]:
    """The recording crop (client px: x, y, width, height) with her name tag's centre `tag` at TAG_IN_CROP, kept
    inside a client `width` x `height` (of geometry `geo`: CROP_W and the rest are 1920x1080's)."""
    w, h = geo.ipx(CROP_W), geo.ipx(CROP_H)
    x = min(max(0, tag[0] - geo.ipx(TAG_IN_CROP[0])), width - w)
    y = min(max(0, tag[1] - geo.ipx(TAG_IN_CROP[1])), height - h)
    return [int(x), int(y), w, h]


def match_tag(frame: np.ndarray, template: np.ndarray, scale: int = SCALE) -> tuple[tuple[int, int] | None, float]:
    """Her name tag's centre in a full-size frame, where the saved template matches it (in a copy shrunk `scale`
    times, the template's) TAG_MATCH_MIN or better (else None), and how well the best match did."""
    (x, y), score = track.best_match(frame[::scale, ::scale], template)
    if score < TAG_MATCH_MIN:
        return None, score
    return ((x + template.shape[1] // 2) * scale, (y + template.shape[0] // 2) * scale), score


def crop_on_her(cal: Calibration, frame: np.ndarray, width: int, height: int) -> tuple[list[int], str]:
    """Where to record her: the crop around her name tag in `frame` (full size, of a client `width` x `height`,
    as the game shows it now), where the saved template finds it, unless the stored crop has it within TAG_SLACK of
    where calibrate puts it; where it isn't found, the stored crop. And what to tell the user: nothing when the
    stored crop is kept; else where she is and whether the stored one had her, or that her tag wasn't found. The
    stored crop is calibrate's, around wherever she stood then: in game on 2026-10-21 the 16 `act --record` runs
    of a sweep never had her in their crop, from a calibration she had walked away from (its outcomes were read
    off the screenshots after each run instead)."""
    tag, score = match_tag(frame, cal.template, cal.scale)
    if tag is None:
        return list(cal.crop), (f"her name tag wasn't found on screen (the best match {score:.2f}, under "
                                f"{TAG_MATCH_MIN}): recording the stored crop {list(cal.crop)}, which may not have "
                                f"her (run calibrate where she stands)")
    x, y, w, h = cal.crop
    geo = cal.geo
    at, slack = (geo.ipx(TAG_IN_CROP[0]), geo.ipx(TAG_IN_CROP[1])), geo.px(TAG_SLACK)
    if abs(tag[0] - x - at[0]) <= slack and abs(tag[1] - y - at[1]) <= slack:
        return list(cal.crop), ""
    crop = crop_around(tag, width, height, geo)
    had = "had her too" if x <= tag[0] < x + w and y <= tag[1] < y + h else "hasn't her"
    return crop, f"her name tag is at {tag[0]},{tag[1]}: recording {crop} around it (the stored {list(cal.crop)} {had})"


def crop_now(cal: Calibration) -> tuple[list[int], str]:
    """crop_on_her on a full-size frame of the game as it is now: one short capture that sends no keys."""
    full = client.submit(fps=10, duration_ms=100, scale=1, lead_ms=0, name="find_her")
    return crop_on_her(cal, full["frames"][-1], full["client"]["width"], full["client"]["height"])


def _find_tag(frame: np.ndarray, template: np.ndarray | None, scale: int = SCALE,
              looked: Path = LAB / "calibrate.png") -> tuple[int, int]:
    """Her name tag's centre in a full-size frame: where the saved template matches (in a copy shrunk `scale`
    times), or else the name tag nearest the middle. `looked`: where the frame was saved, said if it isn't found."""
    if template is not None and (tag := match_tag(frame, template, scale)[0]) is not None:
        return tag
    tag = track.find_name_tag(frame)
    if tag is None:
        raise RuntimeError(f"couldn't find her name tag; look at {looked} and pass --tag X,Y")
    return tag


def load(geo: Geometry = STRETCHED) -> Calibration:
    """The saved calibration for a game window of geometry `geo` (raises if there is none yet)."""
    cal_file, small_file, _ = files(geo)
    if not cal_file.exists():
        size = f"{geo.client[0]}x{geo.client[1]}"
        raise RuntimeError(f"not calibrated at {size} yet: run `python -m gamelab calibrate` with the game at {size}")
    data = json.loads(cal_file.read_text())
    return Calibration(data["crop"], data["scale"], np.load(small_file), geo)
