"""Talking to the elevated key player (control.player) through files: a plan goes into the inbox
(paths.INBOX) as JSON, and the player answers in the outbox (paths.OUTBOX), with any frames it
recorded beside the answer. Submit a plan and wait for its result, or send it, watch for it to be
done and cut it short, or (a player that takes them: can_splice) hand it a plan that takes over from the running one at
a moment of its own, the keys held then kept held (`splice`: the navigator's plans replanned as she goes,
control.navigator.REPLAN). No plan goes out while the Lie Detector alarm is up (control.alarm), but the two kinds the
player still takes then (alarm.PLAN_KINDS: her pet's food, her retreat's moves), sent with their `kind`.
"""

import json
import os
import time
import uuid

import numpy as np

from ..paths import ALARM, INBOX, OUTBOX, PLAYER_INFO, PLAYER_LOG
from .alarm import LIE_DETECTOR_ERROR, PLAN_KINDS

POLL_S = 0.01  # how often `wait` looks for the result
_KINDS: set[str] = set()  # plans sent with a kind, not answered yet: the alarm doesn't withdraw them (`wait`)
SPLICED = "spliced"  # a plan's result's problem when another took over from it (`splice`)
_SPLICES: dict[str, str] = {}  # a splice's plan id -> the plan it takes over from, until it's known taken or not (`taken`)
_NEVER: set[str] = set()       # splices the player never took (withdrawn: `cancel`, `taken`)
_INFO: dict = {"stamp": None, "can": False}  # can_splice's last read of paths.PLAYER_INFO, by its stamp


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
    plan_id = _plan_id(name)
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


def _plan_id(name: str) -> str:
    """A new plan's id: the time, its name (only what the player keeps of a file's name: player._answer) and a random
    tail."""
    name = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "plan"
    return f"{time.strftime('%H%M%S')}_{name}_{uuid.uuid4().hex[:4]}"[:64]


def can_splice() -> bool:
    """Whether the running player takes splices (`splice`): it says so in paths.PLAYER_INFO as it starts (control.
    player.serve; it removes the file as it stops), and its process is still there. A player started before that code
    (its code is loaded once, and starting it again takes the user's UAC click) writes none: the navigator falls back
    to cutting the running plan and sending the next at once. Read again whenever the file changes (its stamp)."""
    try:
        st = os.stat(PLAYER_INFO)
    except OSError:
        _INFO.update(stamp=None, can=False)
        return False
    stamp = (st.st_mtime_ns, st.st_size)
    if _INFO["stamp"] != stamp:
        try:
            info = json.loads(PLAYER_INFO.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False  # (being written: asked again next time)
        _INFO.update(stamp=stamp, can=bool(info.get("splice")) and _alive(info.get("pid")))
    return _INFO["can"]


def _alive(pid) -> bool:
    """Whether process `pid` is still running (Windows: OpenProcess and its exit code, which an elevated process's
    gives too; elsewhere os.kill's probe; never Windows' os.kill, which ends a process)."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


def splice(running_id: str, events=(), at_ms: int = 0, held=(), *, end_ms: int | None = None, name: str = "plan",
           kind: str | None = None) -> str:
    """Hand the player a plan that takes over from the running plan `running_id` at `at_ms` of that plan's own clock
    (its first key's time 0; at once if that's gone by when the player sees it): the running plan's keys from then on
    don't come, the keys it holds then that `held` (key names) doesn't have are let go and those `held` has that it
    doesn't are pressed, the new plan's events play from then on (a 'down' of a key held then is left out), and its
    `end_ms` is as `send`'s. Returns its id (for done / wait / cancel / splice). It goes into the inbox as
    `{running_id}.splice` (not a `*.json`, so the player never takes it as a plan of its own), replacing one not taken
    yet. The player answers the running plan with the problem SPLICED (and `spliced`: the new id, at_ms, how late it
    took it) and the new one as any. A splice the player never took (the running plan was over first: its result has
    no SPLICED) is withdrawn with `unsplice`. Refused with the Lie Detector alarm up as `send` is (`kind` as its)."""
    if kind is None:
        check_alarm()
    elif kind not in PLAN_KINDS:
        raise ValueError(f"no plan kind {kind!r}; the kinds are {', '.join(PLAN_KINDS)}")
    INBOX.mkdir(parents=True, exist_ok=True)
    plan_id = _plan_id(name)
    plan = {"id": plan_id, "events": [list(e) for e in events], "at_ms": int(at_ms), "held": [str(k) for k in held],
            "capture": {"fps": 0, "duration_ms": 0, "scale": 1, "lead_ms": 0}, "focus": False}
    if end_ms is not None:
        plan["end_ms"] = end_ms
    if kind is not None:
        plan["kind"] = kind
        _KINDS.add(plan_id)
    tmp = INBOX / f"{running_id}.splice.tmp"
    tmp.write_text(json.dumps(plan), encoding="utf-8")
    tmp.replace(INBOX / f"{running_id}.splice")
    for child, parent in list(_SPLICES.items()):  # (one into the same plan not taken yet: replaced, never taken)
        if parent == running_id:
            del _SPLICES[child]
            _NEVER.add(child)
    _SPLICES[plan_id] = running_id
    return plan_id


def taken(plan_id: str) -> bool | None:
    """For a plan handed over with `splice`: whether the player took it (True), never will (False: the plan it was to
    take over from was over first, its result without SPLICED, and the splice is withdrawn; or it was withdrawn), or
    None while that isn't known yet (that plan still runs). The plan it takes over from has its result read here, once
    it's there (`wait`); a plan not sent by `splice` was taken."""
    parent = _SPLICES.get(plan_id)
    if parent is None:
        return plan_id not in _NEVER
    if not done(parent):
        return None
    del _SPLICES[plan_id]
    try:
        result = wait(parent, timeout=5)
    except PlayerError:
        result = {}
    if SPLICED in result.get("problems", ()) and result.get("spliced", {}).get("to") == plan_id:
        return True
    unsplice(parent)
    _NEVER.add(plan_id)
    return False


def unsplice(running_id: str) -> bool:
    """Withdraw the splice into plan `running_id` if the player hasn't taken it (True: it never will; False: it took
    it, or there was none)."""
    try:
        (INBOX / f"{running_id}.splice").unlink()
        return True
    except FileNotFoundError:
        return False


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
    result as usual). A plan that is over already is left as it is: no cancel file is made for it. A splice the player
    hasn't taken yet is withdrawn and the plan it was to take over from, still running, is cut short (its result read by
    `taken`)."""
    if plan_id in _NEVER:
        return True
    parent = _SPLICES.get(plan_id)
    if parent is not None and unsplice(parent):
        _NEVER.add(plan_id)  # (`taken` reads the result of the plan it was to take over from, cut short now)
        cancel(parent)
        return True
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
