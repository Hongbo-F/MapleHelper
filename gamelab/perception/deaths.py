"""A monster's death seen (DeathWatch): its death animation, the game data's die1 frames, matched where a monster
followed was last seen once its own pictures there have stopped matching well.

The game never says which monster died: her EXP number goes up 0.2 s after the killing key (the server's), and the
monster is still drawn alive, walking with its bar up, until her star lands, about 0.85-0.9 s after the key; then it
stops where it got to and plays die1 (a Horny or Green Mushroom: 4 frames of 180, 180, 180 and 300 ms, the last fading
out; frame 0 its standing picture with its eyes shut, 1 sinking into its cap, 2 the cap with a sliver of face, 3 the
cap alone), and is gone about 0.7 s later. The monster detector has no die1 pictures (sprites.build takes stand, move
and hit1), so a dying monster's track is lost and kept hidden for seconds (live, the service judged it dead a median
1.40 s after its die1 began), and its HP bar simply goes with the hit (no "empty" bar: 2 of 24,722 track rows).

Measured offline on the Lie Detector's and the rolling recording's frames of 2026-10-26 (16.8 minutes, 166 deaths
labelled by eye on contact sheets, 142 of them on a track of the service's frame step replayed;
D:/MapleHelper-scratch/kplace/diag-signs, detect.py and evaluate.py): around the feet of a track re-found under
REFOUND_UNDER, or lost within RECENT_S, die1/1-3 scored DIE_GOOD or more on 110 of the 142 deaths (77%), and with
die1/0 at DIE0_GOOD where no live picture is (a track not re-found in the frame) 118 (83%), 7 fires on no labelled
death in 16.8 minutes, every one of them a real death on inspection; a median 0.14 s after die1 began (p10 0.04, p90
0.42), 1.03 s after the killing key. Every EVERY-th frame a track (the frame loop's 30 a second: 15 looks a track a
second at most), PAD around where its feet would be, both ways it can face (8 masked matches in one window, in C:
match.best_matches). Built so and replayed in the service's frame step on the three recordings with her kills labelled
(D:/MapleHelper-scratch/kplace/build: replay.py, eval_deaths.py), it saw 83 deaths, 73 of the 111 labelled there and
the other 10 every one a real death by eye, 0.13 s after die1 began at the median (p10 0.04, p90 0.44); 0.83 ms a look
on one thread, 0.86 looks a frame (0.72 ms a frame, the worst 7.3), 0.5 ms a frame on the re-finds' pool (the worst
2.8). The die1 frames come from the game data as the watch is made (gamedata.canvas.mob_frames, its die1 action: in
memory, nothing written; 0.5 s), drawn as the look frame shows them.

It comes too late for her next throw after a kill (decided about 0.45 s after the killing key, keyed at 0.85): that
goes by her EXP gain and the monster her attack took then (localization.tracking). What it's for: which monster really
died, about a second after the key, so a kill put on the wrong one is put right and the throws after it stop, and
another player's kill is dead at once, not 3.5 s later (tracking.HIDE_S).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .kinds import kind_of
from .match import best_matches
from .mobs import OPAQUE

DIE_GOOD = 0.7      # die1/1-3 matched this well around a track's feet: it's dying
DIE0_GOOD = 0.6     # die1/0 (the standing picture, eyes shut) this well, on a track not re-found in the frame
REFOUND_UNDER = 0.9  # a track re-found this well (mobs.REFIND_GOOD) is alive: not looked at
RECENT_S = 0.6      # a track lost this recently is looked at
PAD = 16         # look-frame px at 1366x768 (a game px each) around where its feet would be: 8 lost 8 of 106
EVERY = 2           # a track is looked at every this many frames (by its id: half the tracks a frame)


@dataclass(frozen=True)
class DiePicture:
    """One die1 frame as the look frame shows it, facing one way: its colours, its mask (the pixels drawn opaque), and
    where its feet are in it (`ox`, `oy`)."""
    index: int
    rgb: np.ndarray
    mask3: np.ndarray
    ox: int
    oy: int

    @property
    def w(self) -> int:
        return self.rgb.shape[1]

    @property
    def h(self) -> int:
        return self.rgb.shape[0]


def pictures_of(frames: list[dict], scale: float = 1.0) -> list[DiePicture]:
    """The die1 frames (gamedata.canvas.mob_frames' dicts: rgba facing left, origin) as a look frame `scale` times the
    game's size shows them, both ways: at 1 the game's own pixels, its alpha-255 ones drawn (as the detector's are
    matched at 1366x768); else resized linearly, from OPAQUE (sprites.build's)."""
    out = []
    for f in frames:
        rgba = np.ascontiguousarray(f["rgba"])
        ox, oy = f["origin"]
        if scale != 1.0:
            size = (max(1, round(rgba.shape[1] * scale)), max(1, round(rgba.shape[0] * scale)))
            rgb = cv2.resize(np.ascontiguousarray(rgba[..., :3]), size, interpolation=cv2.INTER_LINEAR)
            alpha = cv2.resize(np.ascontiguousarray(rgba[..., 3]), size, interpolation=cv2.INTER_LINEAR)
            mask = (alpha >= OPAQUE).astype(np.uint8)
            ox, oy = round(ox * scale), round(oy * scale)
        else:
            rgb, mask = rgba[..., :3], (rgba[..., 3] == 255).astype(np.uint8)
        for flip in (False, True):
            c = rgb[:, ::-1] if flip else rgb
            m = mask[:, ::-1] if flip else mask
            out.append(DiePicture(f["index"], np.ascontiguousarray(c), np.ascontiguousarray(np.dstack([m, m, m])),
                                  c.shape[1] - ox if flip else ox, oy))
    return out


class DeathWatch:
    """The die1 pictures of a map's monsters (`pictures`: kind -> DiePicture list) and the look at a track's place
    (`dying`). `scale`: look-frame px per game px (Geometry.frame_scale)."""

    def __init__(self, pictures: dict[str, list[DiePicture]], scale: float = 1.0):
        self.pictures = {k: v for k, v in pictures.items() if v}
        self.scale = scale
        self.pad = max(4, round(PAD * scale))

    @classmethod
    def of(cls, names_by_id: dict[int, str], scale: float = 1.0) -> "DeathWatch":
        """The watch for these monsters (id -> name), their die1 frames read from the game data (none for a monster
        the data doesn't have)."""
        from gamedata.canvas import mob_frames
        pictures = {}
        for mob_id, name in names_by_id.items():
            try:
                frames = mob_frames(mob_id, actions=("die1",))
            except (FileNotFoundError, KeyError, OSError):
                frames = []
            pictures[kind_of(name)] = pictures_of(frames, scale)
        return cls(pictures, scale)

    def kinds(self) -> set[str]:
        return set(self.pictures)

    def best(self, frame: np.ndarray, kind: str, fx: float, fy: float) -> tuple[float, int] | None:
        """The best die1 frame's match around where a `kind`'s feet would be at (fx, fy) (look-frame px), and its
        index; None where its pictures don't fit the frame there."""
        pics = self.pictures.get(kind)
        if not pics:
            return None
        h, w = frame.shape[:2]
        x0 = int(max(0, min(fx - p.ox for p in pics) - self.pad))
        y0 = int(max(0, min(fy - p.oy for p in pics) - self.pad))
        x1 = int(min(w, max(fx - p.ox + p.w for p in pics) + self.pad))
        y1 = int(min(h, max(fy - p.oy + p.h for p in pics) + self.pad))
        if x1 <= x0 or y1 <= y0:
            return None
        window = np.ascontiguousarray(frame[y0:y1, x0:x1])
        fits = [p for p in pics if window.shape[0] >= p.h and window.shape[1] >= p.w]
        if not fits:
            return None
        best = None
        for p, (score, _) in zip(fits, best_matches(window, [(p.rgb, p.mask3) for p in fits])):
            s = float(score) if np.isfinite(score) else -1.0
            if best is None or s > best[0]:
                best = (s, p.index)
        return best

    def dying(self, look, kind: str, fx: float, fy: float, refound: float | None) -> bool:
        """Whether a monster of `kind` whose feet would be at (fx, fy) (look-frame px) in `look` (its `frame`: an
        Observation, a Scene) is seen dying there: its die1/1-3 at DIE_GOOD or more, or, not re-found in this frame
        (`refound`: its re-find's score, None if it wasn't), die1/0 at DIE0_GOOD."""
        got = self.best(look.frame, kind, fx, fy)
        if got is None:
            return False
        score, index = got
        return score >= DIE_GOOD if index > 0 else (refound is None and score >= DIE0_GOOD)


def wants(score: float | None, since: float) -> bool:
    """Whether a track is looked at for its death: re-found in this frame under REFOUND_UNDER (`score`; None: not
    re-found), or lost RECENT_S or less ago (`since`: s since it was last seen)."""
    return (score is not None and score < REFOUND_UNDER) or (score is None and 0.0 < since <= RECENT_S)
