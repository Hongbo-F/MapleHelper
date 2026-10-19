"""Recordings of the game: frames and the keys sent, saved by name, and contact sheets to look at
them. Also single screenshots.
"""

import json
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from ..control import client
from ..paths import RECORDINGS


@dataclass
class Recording:
    name: str
    frames: np.ndarray            # (n, h, w, 3) RGB
    times: np.ndarray             # ms relative to the plan's first key event
    planned: list = field(default_factory=list)  # [ms, key, 'down'|'up'] as planned
    sent: list = field(default_factory=list)     # same, with the times the keys actually went out


def record(events, *, name, duration_ms, crop, scale, fps=60, lead_ms=100) -> Recording:
    """Run a key timeline while recording, save it under `name`, and return it."""
    fps = min(fps, int(900_000 / (duration_ms + lead_ms)))  # the player caps a plan at 900 frames
    result = client.submit(events, fps=fps, duration_ms=duration_ms, crop=crop, scale=scale, lead_ms=lead_ms, name=name)
    rec = Recording(name, result["frames"], result["times"], [list(e) for e in events], result["sent"])
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
    (RECORDINGS / f"{rec.name}.events.json").write_text(json.dumps({"planned": rec.planned, "sent": rec.sent}))


def load(name: str) -> Recording:
    """A saved recording (with no key events if they weren't saved)."""
    with np.load(RECORDINGS / f"{name}.npz") as data:
        frames, times = data["frames"], data["times"]
    events_path = RECORDINGS / f"{name}.events.json"
    ev = json.loads(events_path.read_text()) if events_path.exists() else {"planned": [], "sent": []}
    return Recording(name, frames, times, ev["planned"], ev["sent"])


def contact_sheet(frames, times, path, *, cols=8, every=3, zoom=1) -> None:
    """Tile frames (labelled with their time) into one image so they can be looked at together."""
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
        draw.rectangle([x, y, x + 64, y + 14], fill="black")
        draw.text((x + 3, y + 1), f"{times[i]:.0f}ms", fill="yellow")
    sheet.save(path)
