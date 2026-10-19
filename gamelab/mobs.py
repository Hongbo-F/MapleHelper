"""Finding monsters on screen: colour silhouettes first, then masked template matching.

Each monster type has a few templates (sprites cut out with a mask, gamelab/assets/mobs/<kind>_*.png,
RGBA with the alpha as mask) and a colour test for its body. Step 1 matches each template's body-colour
silhouette against the frame's body-colour mask (one fast single-channel pass over the frame). Step 2
checks only the best few spots with the full masked colour template, which rejects scenery that merely
shares the colour (leaves, clouds). Frames are game-window pictures downscaled by SCALE.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parent / "assets" / "mobs"
SCALE = 2
THRESHOLD = 0.6            # masked colour match a spot needs to count as a monster
SILHOUETTE_THRESHOLD = 0.35  # step 1: a spot worth checking
MAX_CHECKS = 12            # step 2: best spots checked per monster type


@dataclass(frozen=True)
class Mob:
    kind: str
    x: int        # centre, in frame pixels
    y: int
    w: int
    h: int
    score: float

    @property
    def feet_y(self) -> int:
        return self.y + self.h // 2


def palette(kind: str, rgb: np.ndarray, channels=None) -> np.ndarray:
    """Pixels that have the colour of this monster's body. `channels` = (r, g, b) as int16, if
    already split (saves time when testing several kinds on one frame)."""
    r, g, b = channels or (rgb[..., i].astype(np.int16) for i in range(3))
    if kind == "pig":
        return (r > 190) & (g > 120) & (b > 110) & (r - g > 18) & (r - b > 15)
    if kind == "slime":
        return (g > 170) & (g - r > 30) & (g - b > 90)
    if kind == "snail":
        teal = (r < 90) & (g > 80) & (b > 80) & (g < 170) & (np.abs(g - b) < 40)
        lavender = (r > 140) & (b > 190) & (g > 130) & (b - g > 15)
        return teal | lavender
    if kind == "orange-mushroom":  # the orange cap (daffodils are yellower: higher green)
        return (r > 215) & (g > 100) & (g < 175) & (b < 90) & (r - g > 55)
    raise ValueError(f"unknown monster kind {kind!r}")


KINDS_WITH_PALETTE = ("pig", "slime", "snail", "orange-mushroom")


class MobDetector:
    """Templates come from `folders`: PNGs named <kind>_<anything>.png, RGBA at full screen size.
    gamelab/assets/mobs has cut-outs from screenshots; gamelab.sprites makes them from the game data."""

    def __init__(self, folder: Path | list[Path] = ASSETS, scale: int = SCALE, threshold: float = THRESHOLD):
        self.threshold = threshold
        self.templates: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        folders = folder if isinstance(folder, list) else [folder]
        for path in sorted(p for f in folders for p in f.glob("*.png")):
            if path.stem.split("_")[0] not in KINDS_WITH_PALETTE:
                continue
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            h, w = rgba.shape[:2]
            size = (max(1, w // scale), max(1, h // scale))
            rgb = cv2.resize(np.ascontiguousarray(rgba[..., :3]), size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize((rgba[..., 3] > 0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            kind = path.stem.split("_")[0]
            for flip in (False, True):  # monsters face either way
                t, m = (rgb[:, ::-1], mask[:, ::-1]) if flip else (rgb, mask)
                t = np.ascontiguousarray(t)
                silhouette = (palette(kind, t) & (m > 0)).astype(np.float32)
                coarse = cv2.resize(silhouette, (max(1, t.shape[1] // 2), max(1, t.shape[0] // 2)),
                                    interpolation=cv2.INTER_AREA)
                self.templates.setdefault(kind, []).append((t, np.dstack([m] * 3), coarse))
        if not self.templates:
            raise RuntimeError(f"no monster templates in {folder}")

    def detect(self, frame: np.ndarray, rows: tuple[int, int] | None = None) -> list[Mob]:
        """Monsters in `frame` (RGB, game window / SCALE). `rows` limits the search to a band."""
        y_off = 0
        if rows:
            y_off = max(0, rows[0])
            frame = frame[y_off:rows[1]]
        channels = tuple(frame[..., i].astype(np.int16) for i in range(3))
        coarse_size = (frame.shape[1] // 2, frame.shape[0] // 2)
        hits = []
        for kind, templates in self.templates.items():
            colour = palette(kind, frame, channels)
            if colour.sum() < 20:
                continue
            # Step 1: body-colour silhouettes, at half this frame's size.
            coarse = cv2.resize(colour.astype(np.float32), coarse_size, interpolation=cv2.INTER_AREA)
            spots = []
            for i, (_, _, silhouette) in enumerate(templates):
                if coarse.shape[0] < silhouette.shape[0] or coarse.shape[1] < silhouette.shape[1]:
                    continue
                res = cv2.matchTemplate(coarse, silhouette, cv2.TM_CCOEFF_NORMED)
                peaks = (res >= SILHOUETTE_THRESHOLD) & (res == cv2.dilate(res, np.ones((3, 3), np.uint8)))
                for y, x in zip(*np.nonzero(peaks)):
                    spots.append((float(res[y, x]), 2 * x + silhouette.shape[1], 2 * y + silhouette.shape[0], i))
            spots.sort(reverse=True)
            # Step 2: full masked colour check of the template that found each of the best spots.
            checked: list[tuple[int, int]] = []
            for _, cx, cy, i in spots:
                if len(checked) >= MAX_CHECKS:
                    break
                if any(abs(cx - x) < 6 and abs(cy - y) < 6 for x, y in checked):
                    continue
                checked.append((cx, cy))
                found = self._check(frame, templates[i], cx, cy)
                if found and found[0] >= self.threshold:
                    score, x, y, w, h = found
                    hits.append(Mob(kind, x, y_off + y, w, h, round(score, 3)))
        return _suppress(hits)

    @staticmethod
    def _check(frame, template, cx: int, cy: int, slack: int = 6):
        """Masked colour match of one template centred near (cx, cy): (score, x, y, w, h)."""
        tpl, mask, _ = template
        th, tw = tpl.shape[:2]
        x0, y0 = max(0, cx - tw // 2 - slack), max(0, cy - th // 2 - slack)
        window = frame[y0:y0 + th + 2 * slack, x0:x0 + tw + 2 * slack]
        if window.shape[0] < th or window.shape[1] < tw:
            return None
        res = np.nan_to_num(cv2.matchTemplate(window, tpl, cv2.TM_CCOEFF_NORMED, mask=mask), nan=-1.0,
                            posinf=-1.0, neginf=-1.0)
        _, score, _, (x, y) = cv2.minMaxLoc(res)
        return float(score), x0 + x + tw // 2, y0 + y + th // 2, tw, th


def _suppress(hits: list[Mob]) -> list[Mob]:
    kept: list[Mob] = []
    for hit in sorted(hits, key=lambda m: -m.score):
        if all(abs(hit.x - k.x) > max(hit.w, k.w) // 2 or abs(hit.y - k.y) > max(hit.h, k.h) // 2 for k in kept):
            kept.append(hit)
    return kept
