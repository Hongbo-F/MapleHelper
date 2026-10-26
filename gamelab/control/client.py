"""Talking to the elevated key player (control.player) through files: a plan goes into the inbox
(paths.INBOX) as JSON, and the player answers in the outbox (paths.OUTBOX), with any frames it
recorded beside the answer. Submit a plan and wait for its result, or send it, watch for it to be
done and cut it short. No plan goes out while the Lie Detector alarm is up (control.alarm), but the two kinds the
player still takes then (alarm.PLAN_KINDS: her pet's food, her retreat's moves), sent with their `kind`.
"""

import json
import os
import time
import uuid

import numpy as np

from ..paths import ALARM, INBOX, OUTBOX, PLAYER_LOG
from .alarm import LIE_DETECTOR_ERROR, PLAN_KINDS

POLL_S = 0.01  # how often `wait` looks for the result
_KINDS: set[str] = set()  # plans sent with a kind, not answered yet: the alarm doesn't withdraw them (`wait`)


class PlayerError(RuntimeError):
    """The player refused or failed a plan, or never answered it."""


class LieDetectorAlarm(PlayerError):
    """The game's Lie Detector was seen; gamelab won't send input until the alarm is cleared."""


def check_alarm() -> None:
    """Raise LieDetectorAlarm while the Lie Detector alarm is up."""
    from .alarm import alarm_info
    if info := alarm_info():
        raise LieDetectorAlarm(f"Lie Detector seen at {info.get('time', '?')}")


def alarm_up() -> bool:
    """Whether the Lie Detector alarm is up (its file there: control.alarm)."""
    return ALARM.exists()


def alarm_age() -> float | None:
    """Seconds since the player saw the Lie Detector, by the alarm's `at` (alarm.raise_alarm; an alarm without one, by
    when its file was last written); None with no alarm up."""
    from .alarm import alarm_info
    try:
        written = os.stat(ALARM).st_mtime
    except OSError:
        return None
    at = (alarm_info() or {}).get("at")
    return max(0.0, time.time() - (at if isinstance(at, (int, float)) else written))


def submit(events=(), *, fps=0, duration_ms=0, crop=None, scale=1, focus=True, lead_ms=100,
           name="plan", timeout=120, window_client=None, kind: str | None = None) -> dict:
    """Hand a plan to the running player and wait for the result (frames included if recorded). `window_client`:
    (width, height) to resize the game's client area to first (window.CLIENT_SIZES). `kind`: as `send`'s."""
    plan_id = send(events, fps=fps, duration_ms=duration_ms, crop=crop, scale=scale, focus=focus,
                   lead_ms=lead_ms, name=name, window_client=window_client, kind=kind)
    return wait(plan_id, timeout)


def send(events=(), *, fps=0, duration_ms=0, crop=None, scale=1, focus=True, lead_ms=100, name="plan",
         end_ms: int | None = None, window_client=None, kind: str | None = None) -> str:
    """Hand a plan to the running player without waiting; returns its id (for wait / cancel).
    `end_ms`: the plan lasts this long even after its last key (the next plan starts then). `kind`: one of
    alarm.PLAN_KINDS (her pet's food, her retreat's moves), which goes out with the Lie Detector alarm up too, for the
    player to take or refuse by what the kind allows (player.alarm_refusal); any other plan is refused here then."""
    if kind is None:
        check_alarm()
    elif kind not in PLAN_KINDS:
        raise ValueError(f"no plan kind {kind!r}; the kinds are {', '.join(PLAN_KINDS)}")
    INBOX.mkdir(parents=True, exist_ok=True)
    # Only what the player keeps of a plan's file name (player._answer): a name with a space in it ("pet food")
    # was answered as "petfood", and its result never found.
    name = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "plan"
    plan_id = f"{time.strftime('%H%M%S')}_{name}_{uuid.uuid4().hex[:4]}"[:64]
    capture = {"fps": fps, "duration_ms": duration_ms, "scale": scale, "lead_ms": lead_ms}
    if crop:
        capture["crop"] = [int(v) for v in crop]
    plan = {"events": [list(e) for e in events], "capture": capture, "focus": focus}
    if end_ms is not None:
        plan["end_ms"] = end_ms
    if window_client is not None:
        plan["window_client"] = [int(v) for v in window_client]
    if kind is not None:
        plan["kind"] = kind
        _KINDS.add(plan_id)
    tmp = INBOX / f"{plan_id}.tmp"
    tmp.write_text(json.dumps(plan), encoding="utf-8")
    tmp.replace(INBOX / f"{plan_id}.json")
    return plan_id


def busy() -> bool:
    """Whether a plan is waiting for the player or being played (its file in the inbox, from this process or any
    other): keys of a plan sent now would come after it, or in the middle of what it does (control.window_size)."""
    return any(INBOX.glob("*.json")) or any(INBOX.glob("*.running"))


def done(plan_id: str) -> bool:
    """Whether the player has answered the plan (its result is waiting)."""
    return (OUTBOX / f"{plan_id}.json").exists()


def cancel(plan_id: str) -> bool:
    """Cut a plan short: if the player hasn't picked it up it's withdrawn (returns True: no result
    will come), else the player stops pressing and lets go of every key it holds (wait for its
    result as usual). A plan that is over already is left as it is: no cancel file is made for it."""
    try:
        (INBOX / f"{plan_id}.json").unlink()  # the player renames plans it takes, so it's not started
        _KINDS.discard(plan_id)
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
    kind = plan_id in _KINDS  # sent with a kind: the player decides, the alarm up or not
    _KINDS.discard(plan_id)
    end = time.monotonic() + timeout
    while not result_path.exists():
        if plan_path.exists() and ALARM.exists() and not kind:  # not taken yet, and it would only be refused (or
            plan_path.unlink(missing_ok=True)                    # no player)
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
        error = str(result["error"])
        if not kind:
            check_alarm()
        if error.startswith(LIE_DETECTOR_ERROR):  # refused for it: its file not written yet, or a kind's refusal
            raise LieDetectorAlarm(error)
        raise PlayerError(error)
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
