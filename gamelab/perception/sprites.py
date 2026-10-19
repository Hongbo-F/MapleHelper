"""Monster templates made from the game's own sprites (gamedata.canvas), drawn the size they appear on
screen. They're extracted game data, so they're written next to the other extracts (git-ignored).

A monster's kind is its name in lower case with dashes ("orange-mushroom"; perception.kinds).
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gamedata.paths import OUT

from ..screen import SCREEN_SCALE
from .kinds import kind_of
from .mobs import ASSETS, KINDS_WITH_PALETTE, MobDetector


SPRITES = OUT / "sprites"
OPAQUE = 128  # a game picture's pixel is opaque from this alpha up


def screen_size(picture: np.ndarray) -> tuple[int, int]:
    """(width, height) of a game picture as it shows on a 1920x1080 screen."""
    h, w = picture.shape[:2]
    return max(1, round(w * SCREEN_SCALE)), max(1, round(h * SCREEN_SCALE))


def build(names_by_id: dict[int, str], folder: Path = SPRITES) -> list[Path]:
    """Write `<kind>_<action><index>.png` (RGBA, screen size at 1920x1080) for each monster id -> name:
    every frame of gamedata.canvas.mob_frames' poses (stand, move and hit)."""
    from gamedata.canvas import mob_frames
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for mob_id, name in names_by_id.items():
        for frame in mob_frames(mob_id):
            rgba = frame["rgba"]
            size = screen_size(rgba)
            rgb = cv2.resize(np.ascontiguousarray(rgba[..., :3]), size, interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(np.ascontiguousarray(rgba[..., 3]), size, interpolation=cv2.INTER_LINEAR)
            alpha = np.where(alpha >= OPAQUE, 255, 0).astype(np.uint8)
            path = folder / f"{kind_of(name)}_{frame['action']}{frame['index']}.png"
            Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path)
            written.append(path)
    return written


def ensure(names_by_id: dict[int, str], folder: Path = SPRITES) -> Path:
    """Build templates for any of these monsters (id -> name) that don't have them yet; returns the folder.
    (loot.ensure, unlike this, rebuilds every template when any is missing.)"""
    missing = {i: n for i, n in names_by_id.items() if not any(folder.glob(f"{kind_of(n)}_*.png"))}
    if missing:
        build(missing, folder)
    return folder


def detector_for(model) -> MobDetector:
    """A MobDetector for a map: screenshot cut-outs (good at monsters half hidden in plants) plus the
    game's sprites of every monster that spawns there (every pose, any monster with a colour gate)."""
    names_by_id = {s.mob_id: s.name for s in model.spawns if s.name and kind_of(s.name) in KINDS_WITH_PALETTE}
    return MobDetector([ASSETS, ensure(names_by_id)])
