"""Talking to the elevated key player (control.player) through files: a plan goes into the inbox
(paths.INBOX) as JSON, and the player answers in the outbox (paths.OUTBOX), with any frames it
recorded beside the answer. Submit a plan and wait for its result, or send it, watch for it to be
done and cut it short. No plan goes out while the Lie Detector alarm is up (control.alarm).
"""

import json
import time
import uuid

import numpy as np

from ..paths import ALARM, INBOX, OUTBOX, PLAYER_LOG

POLL_S = 0.01  # how often `wait` looks for the result


class PlayerError(RuntimeError):
    """The player refused or failed a plan, or never answered it."""


class LieDetectorAlarm(PlayerError):
    """The game's Lie Detector was seen; gamelab won't send input until the alarm is cleared."""


def check_alarm() -> None:
    """Raise LieDetectorAlarm while the Lie Detector alarm is up."""
    from .alarm import alarm_info
    if info := alarm_info():
        raise LieDetectorAlarm(f"Lie Detector seen at {info.get('time', '?')}")


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
    """Whether the player has answered the plan (its result is waiting)."""
    return (OUTBOX / f"{plan_id}.json").exists()


def cancel(plan_id: str) -> bool:
    """Cut a plan short: if the player hasn't picked it up it's withdrawn (returns True: no result
    will come), else the player stops pressing and lets go of every key it holds (wait for its
    result as usual). A plan that is over already is left as it is: no cancel file is made for it."""
    try:
        (INBOX / f"{plan_id}.json").unlink()  # the player renames plans it takes, so it's not started
        return True
    except FileNotFoundError:
        # It renames a plan to .running when it takes it, and deletes that only once its keys are let go
        # of: without it, the plan is over (a cancel made then was never read, and stayed in the inbox).
        if (INBOX / f"{plan_id}.running").exists():
            (INBOX / f"{plan_id}.cancel").touch()
        return False


def wait(plan_id: str, timeout: float = 120) -> dict:
    """The result of a plan sent with `send` (frames included if recorded)."""
    plan_path = INBOX / f"{plan_id}.json"
    result_path = OUTBOX / f"{plan_id}.json"
    frames_path = OUTBOX / f"{plan_id}.npz"
    end = time.monotonic() + timeout
    while not result_path.exists():
        if plan_path.exists() and ALARM.exists():  # the player shut down before taking it
            plan_path.unlink(missing_ok=True)
            check_alarm()
        if time.monotonic() > end:
            plan_path.unlink(missing_ok=True)
            raise PlayerError(f"no result after {timeout}s; is the player running? "
                              f"Start it with `python -m gamelab player --elevate` (log: {PLAYER_LOG})")
        time.sleep(POLL_S)
    result = _read_result(result_path)
    result_path.unlink(missing_ok=True)
    # The plan is over: a cancel the player didn't use up (it came as the plan ended) would stay for good.
    (INBOX / f"{plan_id}.cancel").unlink(missing_ok=True)
    if "error" in result:
        frames_path.unlink(missing_ok=True)
        check_alarm()
        raise PlayerError(result["error"])
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
