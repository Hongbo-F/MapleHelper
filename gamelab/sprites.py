"""Monster templates made from the game's own sprites (gamedata.canvas), drawn the size they appear on
screen. They're extracted game data, so they're written next to the other extracts (git-ignored).

A monster's kind is its name in lower case with dashes ("orange-mushroom"); `kind_of` maps names.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gamedata.paths import OUT

from .mapdata import SCREEN_SCALE

SPRITES = OUT / "sprites"


def kind_of(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


def build(mobs: dict[int, str], folder: Path = SPRITES, actions=("stand", "move", "hit1")) -> list[Path]:
    """Write `<kind>_<action><index>.png` (RGBA, screen size at 1920x1080) for each mob id -> name."""
    from gamedata.canvas import mob_frames
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for mob_id, name in mobs.items():
        for frame in mob_frames(mob_id, actions):
            rgba = frame["rgba"]
            h, w = rgba.shape[:2]
            size = (max(1, round(w * SCREEN_SCALE)), max(1, round(h * SCREEN_SCALE)))
            rgb = cv2.resize(np.ascontiguousarray(rgba[..., :3]), size, interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(np.ascontiguousarray(rgba[..., 3]), size, interpolation=cv2.INTER_LINEAR)
            alpha = np.where(alpha >= 128, 255, 0).astype(np.uint8)
            path = folder / f"{kind_of(name)}_{frame['action']}{frame['index']}.png"
            Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path)
            written.append(path)
    return written


def ensure(mobs: dict[int, str], folder: Path = SPRITES) -> Path:
    """Build templates for any of these monsters that don't have them yet; returns the folder."""
    missing = {i: n for i, n in mobs.items() if not any(folder.glob(f"{kind_of(n)}_*.png"))}
    if missing:
        build(missing, folder)
    return folder


def detector_for(model):
    """A MobDetector for a map: screenshot cut-outs (good at monsters half hidden in plants) plus the
    game's sprites of every monster that spawns there (every pose, any monster with a colour gate)."""
    from .mobs import ASSETS, KINDS_WITH_PALETTE, MobDetector
    mobs = {s.mob_id: s.name for s in model.spawns if s.name and kind_of(s.name) in KINDS_WITH_PALETTE}
    return MobDetector([ASSETS, ensure(mobs)])
