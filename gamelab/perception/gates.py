"""Colour gates learned from data, for the monster kinds without a hand-made one (mobs.PALETTES): which colours are a
kind's, by how much more often its own pictures have them than the map's background does.

A gate is step 1's colour test (mobs.py): where a kind's colour is, its templates' silhouettes are matched. The first
map's four kinds have tests written by hand; any other kind gets a gate learned, so a map onboards from its game data
and a few saved frames of it. A kind's colours are its templates' pixels as step 1 sees them (the game's own sprites
shrunk to step 1's scale, as MobDetector makes them, at either window size), as drawn and blended toward a fog's white
(FOG_ALPHAS); the background's are the play area (above the HUD) of saved frames of the map at step 1's scale, with the
monsters on them cut out (`background`). A colour, binned BITS bits a channel, is the kind's where its share of any one
blend's pixels is RATIO times its share of the background's or more. The background's counts are kept per map in
assets/backgrounds/<map id>.npz (in git; the frames stay out of it), and MobDetector learns its kinds' gates from them
as it's made (its `background`).

All of a detector's learned gates are one lookup a frame: each colour bin's byte holds a bit a kind (`Gates`), so step 1
works out each pixel's bin once and looks it up once, whatever the number of kinds.
"""

from pathlib import Path

import numpy as np

from .. import paths

BITS = 5                # bits a channel a colour is binned to: 32,768 colours. On West Rocky Mountain II's stumps at
                        # 1920x1080 half size (2026-10-23), at 4 the gates took 5.8% of the background for 69% of
                        # their colours; at 6, 0.2% for 38% (bins of 4 are finer than the screen's own blur); at 5, 1.8%
                        # for 61%
RATIO = 5.0             # a kind's colour is this many times commoner in its pictures than in the background (README:
                        # Perception, learned gates, for what each ratio found)
FOG = (245, 245, 245)   # the fog drifting over West Rocky Mountain II: fitted as frame = a * sprite + (1 - a) * fog on
                        # 23 labelled monsters under it (a 0.24-0.6), the fog came out 220-265 a channel
FOG_ALPHAS = (1.0, 0.7, 0.4)  # a kind's colours as drawn and blended toward FOG this much: the masked match itself
                        # doesn't mind fog (a blend with one colour keeps its normalised score), only the gate did: at
                        # 1920x1080 half size red snails under it went from 48 to 53 of 57 found
BACKGROUNDS = paths.ASSETS / "backgrounds"  # <map id>.npz: a map's background's counts (`learn-colours`)


def bins(rgb: np.ndarray) -> np.ndarray:
    """Each pixel's colour bin (RGB, uint8, any shape ending in 3): r, g and b's top BITS bits, red highest."""
    return index(*(rgb[..., i].astype(np.int16) for i in range(3)))


def index(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The colour bins of a picture split into int16 channels (mobs.split_channels)."""
    s = 8 - BITS
    return ((r >> s) << (2 * BITS)) | ((g >> s) << BITS) | (b >> s)


def histogram(pixels: np.ndarray) -> np.ndarray:
    """How many of these pixels (n x 3 RGB, uint8) fall in each colour bin."""
    return np.bincount(bins(np.asarray(pixels, np.uint8).reshape(-1, 3)).ravel(), minlength=1 << (3 * BITS))


def kind_histograms(pixels: np.ndarray, alphas=FOG_ALPHAS, fog=FOG) -> list[np.ndarray]:
    """A kind's colours (its templates' pixels, n x 3 RGB), as drawn (alpha 1) and blended toward `fog` by each
    alpha."""
    px = np.asarray(pixels, np.float64).reshape(-1, 3)
    white = np.asarray(fog, np.float64)
    return [histogram(np.clip(np.round(a * px + (1 - a) * white), 0, 255).astype(np.uint8)) for a in alphas]


def background(pictures: list[np.ndarray], cut: list[list[tuple[int, int, int, int]]] | None = None) -> np.ndarray:
    """The colours of these pictures (RGB: the play area of frames at step 1's scale), but in each one's boxes (x0, y0,
    x1, y1) in `cut`: where monsters were found."""
    counts = np.zeros(1 << (3 * BITS), np.int64)
    for i, picture in enumerate(pictures):
        keep = np.ones(picture.shape[:2], bool)
        for x0, y0, x1, y1 in (cut[i] if cut else ()):
            keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
        counts += histogram(picture[keep])
    return counts


def learn(kind: list[np.ndarray], back: np.ndarray, ratio: float | None = None) -> np.ndarray:
    """A kind's gate, as a table of the colour bins that are its: those whose share of one of `kind`'s histograms
    (kind_histograms) is `ratio` (by default RATIO) times their share of the background's or more. Half a pixel is
    added to every bin, so a colour the background never shows is the kind's if its pictures have it at all, and one
    neither has isn't."""
    ratio = RATIO if ratio is None else ratio
    pb = (back + 0.5) / (back.sum() + 0.5 * back.size)
    lut = np.zeros(back.size, bool)
    for h in kind:
        pk = (h + 0.5) / (h.sum() + 0.5 * h.size)
        lut |= (pk >= ratio * pb) & (h > 0)
    return lut


class Gates:
    """The learned colour tests of a detector's kinds, one lookup for them all: each colour bin's entry has a bit for
    each kind whose colour it is."""

    def __init__(self, luts: dict[str, np.ndarray]):
        if len(luts) > 32:
            raise ValueError(f"at most 32 kinds a detector learns gates for, not {len(luts)}")
        dtype = np.uint8 if len(luts) <= 8 else np.uint16 if len(luts) <= 16 else np.uint32
        self.bit = {kind: dtype(1 << i) for i, kind in enumerate(luts)}
        self.lut = np.zeros(1 << (3 * BITS), dtype)
        for kind, lut in luts.items():
            if lut.shape != self.lut.shape:
                raise ValueError(f"a gate has {self.lut.size} colours, not {lut.shape}")
            self.lut[lut.astype(bool)] |= self.bit[kind]

    def __contains__(self, kind: str) -> bool:
        return kind in self.bit

    def bits(self, kinds: str | tuple[str, ...]) -> int:
        """The bits of a kind, or of several (any of them)."""
        out = 0
        for kind in (kinds,) if isinstance(kinds, str) else kinds:
            out |= int(self.bit[kind])
        return out

    def lookup(self, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Every pixel's kinds (its bits) from a picture's int16 channels (mobs.split_channels)."""
        return self.lut.take(index(r, g, b))

    def of(self, kinds: str | tuple[str, ...], looked_up: np.ndarray) -> np.ndarray:
        """Where a kind's colour is (of several: any of theirs), from `lookup`'s answer."""
        return (looked_up & self.lut.dtype.type(self.bits(kinds))) != 0

    def test(self, kinds: str | tuple[str, ...], r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """A kind's colour test (of several: any of theirs), called like a hand-made one (mobs.PALETTES) on a
        picture's int16 channels."""
        return self.of(kinds, self.lookup(r, g, b))

    def share(self, kinds: str | tuple[str, ...], counts: np.ndarray) -> float:
        """The share of the pixels counted by colour bin (`counts`: a background's) a kind's gate takes."""
        return float(counts[(self.lut & self.lut.dtype.type(self.bits(kinds))) != 0].sum() / max(1, counts.sum()))


def background_path(map_id: int, folder: Path = BACKGROUNDS) -> Path:
    return folder / f"{map_id}.npz"


def save_background(map_id: int, counts: np.ndarray, frames: int, cut: int, folder: Path = BACKGROUNDS) -> Path:
    """Keep a map's background colours (`counts`, from `frames` saved frames with `cut` monsters cut out)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = background_path(map_id, folder)
    np.savez_compressed(path, counts=counts.astype(np.uint32), bits=BITS, frames=frames, cut=cut)
    return path


def load_background(map_id: int | None, folder: Path = BACKGROUNDS) -> np.ndarray | None:
    """A map's background colours, or None if they haven't been learned (or were binned otherwise than BITS)."""
    path = background_path(map_id, folder) if map_id is not None else None
    if path is None or not path.is_file():
        return None
    with np.load(path) as data:
        if int(data["bits"]) != BITS:
            return None
        return data["counts"].astype(np.int64)
