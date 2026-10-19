"""Talking to the player (submit a plan, wait for its result) and saving/loading recordings."""

import json
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from .paths import ALARM, INBOX, OUTBOX, PLAYER_LOG, RECORDINGS


class PlayerError(RuntimeError):
    pass


class LieDetectorAlarm(PlayerError):
    """The game's Lie Detector was seen; gamelab won't send input until the alarm is cleared."""


def check_alarm() -> None:
    from .alarm import alarm_info
    if info := alarm_info():
        raise LieDetectorAlarm(f"Lie Detector seen at {info.get('time', '?')}")


@dataclass
class Recording:
    name: str
    frames: np.ndarray            # (n, h, w, 3) RGB
    times: np.ndarray             # ms relative to the plan's first key event
    planned: list = field(default_factory=list)  # [ms, key, 'down'|'up'] as planned
    sent: list = field(default_factory=list)     # same, with the times the keys actually went out


def submit(events=(), *, fps=0, duration_ms=0, crop=None, scale=1, focus=True, lead_ms=100,
           name="plan", timeout=120) -> dict:
    """Hand a plan to the running player and wait for the result (frames included if recorded)."""
    plan_id = send(events, fps=fps, duration_ms=duration_ms, crop=crop, scale=scale, focus=focus,
                   lead_ms=lead_ms, name=name)
    return wait(plan_id, timeout)


def send(events=(), *, fps=0, duration_ms=0, crop=None, scale=1, focus=True, lead_ms=100, name="plan",
         end_ms: int | None = None) -> str:
    """Hand a plan to the running player without waiting; returns its id (for wait / cancel).
    `end_ms`: the plan lasts this long even after its last key (the next plan starts then)."""
    check_alarm()
    INBOX.mkdir(parents=True, exist_ok=True)
    plan_id = f"{time.strftime('%H%M%S')}_{name}_{uuid.uuid4().hex[:4]}"
    capture = {"fps": fps, "duration_ms": duration_ms, "scale": scale, "lead_ms": lead_ms}
    if crop:
        capture["crop"] = [int(v) for v in crop]
    plan = {"events": [list(e) for e in events], "capture": capture, "focus": focus}
    if end_ms is not None:
        plan["end_ms"] = end_ms
    tmp = INBOX / f"{plan_id}.tmp"
    tmp.write_text(json.dumps(plan), encoding="utf-8")
    tmp.replace(INBOX / f"{plan_id}.json")
    return plan_id


def done(plan_id: str) -> bool:
    return (OUTBOX / f"{plan_id}.json").exists()


def cancel(plan_id: str) -> bool:
    """Cut a plan short: if the player hasn't picked it up it's withdrawn (returns True: no result
    will come), else the player stops pressing and lets go of every key it holds (wait for its
    result as usual)."""
    try:
        (INBOX / f"{plan_id}.json").unlink()  # the player renames plans it takes, so it's not started
        return True
    except FileNotFoundError:
        (INBOX / f"{plan_id}.cancel").touch()
        return False


def wait(plan_id: str, timeout: float = 120) -> dict:
    """The result of a plan sent with `send` (frames included if recorded)."""
    result_path = OUTBOX / f"{plan_id}.json"
    end = time.monotonic() + timeout
    while not result_path.exists():
        if (INBOX / f"{plan_id}.json").exists() and ALARM.exists():  # player shut down before taking it
            (INBOX / f"{plan_id}.json").unlink(missing_ok=True)
            check_alarm()
        if time.monotonic() > end:
            (INBOX / f"{plan_id}.json").unlink(missing_ok=True)
            raise PlayerError(f"no result after {timeout}s; is the player running? "
                              f"Start it with `python -m gamelab player --elevate` (log: {PLAYER_LOG})")
        time.sleep(0.01)
    result = _read_result(result_path)
    result_path.unlink(missing_ok=True)
    if "error" in result:
        (OUTBOX / f"{plan_id}.npz").unlink(missing_ok=True)
        check_alarm()
        raise PlayerError(result["error"])
    frames_path = OUTBOX / f"{plan_id}.npz"
    if frames_path.exists():
        with np.load(frames_path) as data:
            result["frames"], result["times"] = data["frames"], data["times"]
        frames_path.unlink()
    return result


def _read_result(path, tries: int = 50) -> dict:
    """A result file, retried for up to ~1 s: just after the player moves it into place, Windows can
    still hold it locked (a sweep crashed on PermissionError reading one)."""
    for i in range(tries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, ValueError):
            if i == tries - 1:
                raise
            time.sleep(0.02)


def record(events, *, name, duration_ms, crop, scale, fps=60, lead_ms=100) -> Recording:
    """Run a key timeline while recording, save it under `name`, and return it."""
    fps = min(fps, int(900_000 / (duration_ms + lead_ms)))  # the player caps a plan at 900 frames
    result = submit(events, fps=fps, duration_ms=duration_ms, crop=crop, scale=scale, lead_ms=lead_ms, name=name)
    rec = Recording(name, result["frames"], result["times"], [list(e) for e in events], result["sent"])
    if result["problems"]:
        print(f"  {name}: {'; '.join(result['problems'])}")
    save(rec)
    return rec


def save(rec: Recording) -> None:
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RECORDINGS / f"{rec.name}.npz", frames=rec.frames, times=rec.times)
    (RECORDINGS / f"{rec.name}.events.json").write_text(json.dumps({"planned": rec.planned, "sent": rec.sent}))


def load(name: str) -> Recording:
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
