"""Recordings of the game: frames and the keys sent, saved by name, and contact sheets to look at
them. Also single screenshots, and the recordings kept as JPEG frames with an index (control.rolling's, the Lie
Detector's: `Folder`), as a contact sheet or a video.

A recording keeps the game window's client size and how much its frames were shrunk (`geo`: screen.Geometry), so
its px are read at that size; one saved before 2026-10-23 says neither, and was of the 1920x1080 window at half size.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..control import client
from ..paths import RECORDINGS
from ..screen import STRETCHED_CLIENT, Geometry, geometry


@dataclass
class Recording:
    name: str
    frames: np.ndarray            # (n, h, w, 3) RGB
    times: np.ndarray             # ms relative to the plan's first key event
    planned: list = field(default_factory=list)  # [ms, key, 'down'|'up'] as planned
    sent: list = field(default_factory=list)     # same, with the times the keys actually went out
    client: tuple = STRETCHED_CLIENT             # the game window's client size when it was recorded ...
    scale: int = 2                               # ... and how much its frames were shrunk

    @property
    def geo(self) -> Geometry:
        """The geometry its frames are of (their scale the look's)."""
        return geometry(*self.client, self.scale)


def record(events, *, name, duration_ms, crop, scale, fps=60, lead_ms=100) -> Recording:
    """Run a key timeline while recording, save it under `name`, and return it."""
    fps = min(fps, int(900_000 / (duration_ms + lead_ms)))  # the player caps a plan at 900 frames
    result = client.submit(events, fps=fps, duration_ms=duration_ms, crop=crop, scale=scale, lead_ms=lead_ms, name=name)
    area = result.get("client") or {}
    size = (area["width"], area["height"]) if "width" in area else STRETCHED_CLIENT
    rec = Recording(name, result["frames"], result["times"], [list(e) for e in events], result["sent"], size, scale)
    if result["problems"]:
        print(f"  {name}: {'; '.join(result['problems'])}")
    save(rec)
    return rec


def screenshot(name: str, *, scale: int, duration_ms: int = 200, crop=None) -> np.ndarray:
    """One picture of the game (of `crop`, client pixels, if given): the last frame of a short capture
    that sends no keys."""
    return client.submit(fps=10, duration_ms=duration_ms, crop=crop, scale=scale, lead_ms=0, name=name)["frames"][-1]


def save(rec: Recording) -> None:
    """Save a recording under its name: frames and times (.npz), and its key events (.events.json)."""
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RECORDINGS / f"{rec.name}.npz", frames=rec.frames, times=rec.times)
    (RECORDINGS / f"{rec.name}.events.json").write_text(json.dumps({"planned": rec.planned, "sent": rec.sent,
                                                                    "client": list(rec.client), "scale": rec.scale}))


def load(name: str) -> Recording:
    """A saved recording (with no key events if they weren't saved)."""
    with np.load(RECORDINGS / f"{name}.npz") as data:
        frames, times = data["frames"], data["times"]
    events_path = RECORDINGS / f"{name}.events.json"
    ev = json.loads(events_path.read_text()) if events_path.exists() else {"planned": [], "sent": []}
    return Recording(name, frames, times, ev["planned"], ev["sent"], tuple(ev.get("client", STRETCHED_CLIENT)),
                     ev.get("scale", 2))


def file_geometry(path: Path) -> Geometry:
    """The geometry of a saved recording's frames (its .npz anywhere), from its key events beside it (`save`): one
    saved before 2026-10-23, or without them, was of the 1920x1080 window at half size."""
    events = path.with_name(f"{path.stem}.events.json")
    ev = json.loads(events.read_text()) if events.exists() else {}
    return geometry(*ev.get("client", STRETCHED_CLIENT), ev.get("scale", 2))


def contact_sheet(frames, times, path, *, cols=8, every=3, zoom=1, labels=None) -> None:
    """Tile frames (labelled with their time in ms, or with `labels`) into one image so they can be looked at
    together."""
    picked = list(range(0, len(frames), every))
    h, w = frames[0].shape[:2]
    w, h = w * zoom, h * zoom
    sheet = Image.new("RGB", (cols * w, ((len(picked) + cols - 1) // cols) * h), "black")
    draw = ImageDraw.Draw(sheet)
    for n, i in enumerate(picked):
        x, y = (n % cols) * w, (n // cols) * h
        img = Image.fromarray(frames[i])
        if zoom > 1:
            img = img.resize((w, h), Image.NEAREST)
        sheet.paste(img, (x, y))
        label = f"{times[i]:.0f}ms" if labels is None else labels[i]
        draw.rectangle([x, y, x + max(64, 6 * len(label) + 6), y + 14], fill="black")
        draw.text((x + 3, y + 1), label, fill="yellow")
    sheet.save(path)


# --- recordings kept as JPEG frames with an index ------------------------------------------------------------------

SHEET_EVERY = 10   # a folder's contact sheet: a frame a second of the rolling recording's 10 ...
SHEET_SHRINK = 4   # ... each a quarter the size (341x192 at 1366x768): 300 of them for 5 minutes, 2,736x7,296 px
VIDEO_FPS = 10.0   # a folder's video plays in real time at this rate (the rolling recording's)
GAP_GREY = 40      # a moment with no frame in the video: a card this grey, saying why


@dataclass
class Folder:
    """A recording kept as JPEG frames with an index.jsonl: control.rolling's (save-recent's, the part before a Lie
    Detector) or control.lie_recorder's. `frames`: (t in s from its start, file, label: the clock time if the index
    says it) in time order; `gaps`: (t, until, why), the moments with no frame (a Lie Detector recording's: a grab
    each)."""
    folder: Path
    frames: list[tuple[float, Path, str]]
    gaps: list[tuple[float, float, str]]


def load_folder(folder) -> Folder:
    """A recording kept as JPEG frames, from its index (a line that can't be read, half written as its writer stopped,
    is left out)."""
    folder = Path(folder)
    frames, gaps = [], []
    for line in (folder / "index.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "file" in r:
            t = float(r["t"])
            frames.append((t, folder / r["file"], r["at"][:10] if "at" in r else f"{t:.2f}s"))
        elif "gap" in r or "game" in r:
            gaps.append((float(r["t"]), float(r.get("until", r["t"])), r.get("gap") or r["game"]))
    return Folder(folder, frames, gaps)


def folder_sheet(rec: Folder, path, *, start: float = -1e9, end: float = 1e9, every: int = SHEET_EVERY,
                 shrink: int = SHEET_SHRINK, cols: int = 8) -> int:
    """A contact sheet of a frame every `every` from `start` to `end` (s), each `shrink` times smaller and labelled
    with its clock time; returns how many. Only those frames are read (5 minutes of frames are 9 GB as pictures)."""
    picked = [f for f in rec.frames if start <= f[0] <= end][::max(1, every)]
    if not picked:
        raise ValueError(f"no frames in {rec.folder} from {start:g} to {end:g} s")
    tiles, size = [], None
    for _, file, _ in picked:
        img = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"{file} can't be read")
        size = size or (max(1, img.shape[1] // shrink), max(1, img.shape[0] // shrink))
        tiles.append(cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    contact_sheet(tiles, None, path, cols=cols, every=1, labels=[label for _, _, label in picked])
    return len(picked)


def folder_video(rec: Folder, path, *, start: float = -1e9, end: float = 1e9, fps: float = VIDEO_FPS) -> int:
    """The recording from `start` to `end` (s) as an mp4 playing in real time at `fps`: each moment shows the newest
    frame or gap at or before it (a gap as a grey card saying why), the clock time at the top left, every frame at the
    first's size. MPEG-4 Part 2 (OpenCV's own 'mp4v'; its build has no H.264): about 30 dB, the HUD's text legible.
    Returns the video's frames."""
    moments = sorted([(t, 0, file, label) for t, file, label in rec.frames]
                     + [(t, 1, why, f"{t:.2f}s") for t, _, why in rec.gaps], key=lambda m: (m[0], m[1]))
    moments = [m for m in moments if start <= m[0] <= end]
    first = next((m for m in moments if m[1] == 0), None)
    if first is None:
        raise ValueError(f"no frames in {rec.folder} from {start:g} to {end:g} s")
    h, w = cv2.imread(str(first[2]), cv2.IMREAD_COLOR).shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV can't write an mp4 to {path}")
    n, i, shown, picture = round((moments[-1][0] - moments[0][0]) * fps) + 1, 0, None, None
    try:
        for k in range(n):
            at = moments[0][0] + k / fps
            while i + 1 < len(moments) and moments[i + 1][0] <= at + 1e-6:
                i += 1
            if moments[i] is not shown:
                shown = moments[i]
                picture = _video_picture(shown, w, h)
            writer.write(picture)
    finally:
        writer.release()
    return n


def _video_picture(moment, w: int, h: int) -> np.ndarray:
    """A moment of a folder's video: its frame (BGR, at w x h) or a gap's card, with its label."""
    _, gap, what, label = moment
    if gap:
        out = np.full((h, w, 3), GAP_GREY, np.uint8)
        text = f"no frame: {what}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(out, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    else:
        out = cv2.imread(str(what), cv2.IMREAD_COLOR)
        if out is None:
            out = np.zeros((h, w, 3), np.uint8)
        elif out.shape[:2] != (h, w):  # (the window resized during the recording)
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(out, (0, 0), (8 + 11 * len(label), 22), (0, 0, 0), -1)
    cv2.putText(out, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out
