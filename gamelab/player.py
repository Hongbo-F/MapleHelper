"""Admin-side key player and frame recorder for in-game tests.

MapleStory runs as administrator, and Windows blocks key presses from normal programs into admin
windows. So this one process runs elevated (one UAC prompt per test session) and everything else
runs normally and talks to it through files. To keep that safe it only ever:

- reads plans from gamelab/lab/inbox (plain JSON data, never code),
- presses keys that config.toml binds to her in-game actions plus a few exploring keys (EXPLORE_KEYS),
  never Enter, and only while the game is in front,
- moves the mouse pointer inside the game window (never clicks),
- watches the game for the Lie Detector (the game's anti-bot check) the whole time it runs; if it
  appears, input stops at once, the player pages whoever is at the PC and shuts down, and it won't
  start again until `python -m gamelab clear-alarm` (it never tries to answer it),
- records the game's client area while a plan runs,
- writes results to gamelab/lab/outbox.

Its code and key list are loaded once at startup. It exits when gamelab/lab/STOP appears or after
3 idle hours. Start it with `python -m gamelab player --elevate`; stop it with `python -m gamelab stop`.
"""

import json
import threading
import time
import traceback
from dataclasses import fields

import mss
import numpy as np

from maplehelper import config, win32
from maplehelper.combos import COMBOS
from maplehelper.config import GameKeys
from maplehelper.keys import Key
from maplehelper.keys import parse as parse_key

from . import window
from .paths import ALARM, CONFIG, INBOX, LAB, OUTBOX, PLAYER_LOG, STOP

IDLE_LIMIT_S = 3 * 60 * 60
MAX_EVENTS = 400
MAX_PLAN_MS = 20_000
MAX_FRAMES = 900


# Beyond the actions bound in config.toml [keys] (moving, climbing, attacking, looting, chair,
# potions), exploring needs the key bindings window (\) and closing windows (esc).
EXPLORE_KEYS = ("\\", "esc")
# Never pressed, whatever the config says: Enter sends whatever is typed in the chat box.
NEVER_KEYS = {"enter"}


def allowed_keys(keys: GameKeys, extra=EXPLORE_KEYS) -> dict[str, Key]:
    """Her bound in-game actions plus the exploring keys, minus anything in NEVER_KEYS."""
    bound = [getattr(keys, f.name) for f in fields(GameKeys)] + [parse_key(name) for name in extra]
    return {k.name: k for k in bound if k and k.name not in NEVER_KEYS}


def validate(plan: dict, allowed: dict[str, Key]) -> None:
    """Raise ValueError unless the plan is well-formed and only uses allowed keys.

    Events are [ms, key, 'down'|'up'] or [ms, 'mouse', x, y] (move the cursor to client pixel x, y;
    there is deliberately no way to click).
    """
    events = plan.get("events", [])
    if not isinstance(events, list) or len(events) > MAX_EVENTS:
        raise ValueError(f"events must be a list of at most {MAX_EVENTS}")
    last = 0
    for event in events:
        if isinstance(event, list) and len(event) == 4 and event[1] == "mouse":
            t, _, x, y = event
            if not (isinstance(x, int) and isinstance(y, int) and x >= 0 and y >= 0):
                raise ValueError(f"bad mouse position in {event!r}")
        elif isinstance(event, list) and len(event) == 3:
            t, name, kind = event
            if name not in allowed:
                raise ValueError(f"key {name!r} is not allowed; allowed: {sorted(allowed)}")
            if kind not in ("down", "up"):
                raise ValueError(f"bad event kind {kind!r}")
        else:
            raise ValueError(f"bad event {event!r}; expected [ms, key, 'down'|'up'] or [ms, 'mouse', x, y]")
        if not (isinstance(t, (int, float)) and last <= t <= MAX_PLAN_MS):
            raise ValueError(f"event times must be sorted and within 0-{MAX_PLAN_MS} ms")
        last = t
    end = plan.get("end_ms")
    if end is not None and not (isinstance(end, (int, float)) and last <= end <= MAX_PLAN_MS):
        raise ValueError(f"end_ms must be after the last event and within {MAX_PLAN_MS} ms")
    cap = plan.get("capture", {})
    fps, duration = cap.get("fps", 0), cap.get("duration_ms", 0)
    if not (0 <= fps <= 60 and 0 <= duration <= MAX_PLAN_MS):
        raise ValueError(f"capture needs fps 0-60 and duration 0-{MAX_PLAN_MS} ms")
    if fps * (duration + cap.get("lead_ms", 100)) / 1000 > MAX_FRAMES:
        raise ValueError(f"capture would exceed {MAX_FRAMES} frames")
    if cap.get("scale", 1) not in (1, 2, 3, 4):
        raise ValueError("scale must be 1-4")
    crop = cap.get("crop")
    if crop is not None and not (len(crop) == 4 and all(isinstance(v, int) and v >= 0 for v in crop)):
        raise ValueError("crop must be [x, y, width, height] in client pixels")


LIE_DETECTOR_ERROR = "Lie Detector on screen: all input stopped"


def _sleep_until(t: float, stop: threading.Event | None = None) -> bool:
    """Sleep until perf_counter() reaches t. Returns False early if `stop` gets set."""
    while (left := t - time.perf_counter()) > 0:
        if stop is not None and stop.is_set():
            return False
        time.sleep(min(left - 0.002, 0.01) if left > 0.004 else 0)
    return stop is None or not stop.is_set()


class Watchdog(threading.Thread):
    """Looks at the game every `interval` seconds, whether or not a plan is running. When the
    Lie Detector shows up it sets `alarm` first (stopping any keys within ~10 ms, mid-plan too),
    then records it and pages via on_alarm; the player then shuts down."""

    def __init__(self, game_title: str, alarm: threading.Event, spotter=None, grab=None,
                 interval: float = 0.25, on_alarm=None):
        super().__init__(name="lie-detector-watch", daemon=True)
        from .alarm import LieDetectorSpotter, raise_alarm
        self.title = game_title
        self.alarm = alarm
        self.spotter = spotter or LieDetectorSpotter()
        self.grab = grab
        self.interval = interval
        self.on_alarm = on_alarm or raise_alarm
        self.stopped = threading.Event()

    def run(self) -> None:
        with mss.MSS() as sct:
            while not self.stopped.is_set():
                frame = self.grab() if self.grab else self._grab(sct)
                if frame is not None and self.spotter.seen(frame):
                    self.alarm.set()
                    self.on_alarm(frame)
                    return
                self.stopped.wait(self.interval)

    def _grab(self, sct):
        try:
            client = window.client_rect(window.find_game(self.title))
        except RuntimeError:
            return None
        if client["width"] <= 0 or client["height"] <= 0:  # minimized
            return None
        return np.asarray(sct.grab(client))[:, :, 2::-1]  # BGRA -> RGB, full size


class Player:
    def __init__(self, allowed: dict[str, Key], game_title: str, alarm: threading.Event | None = None):
        self.allowed = allowed
        self.title = game_title
        self.alarm = alarm or threading.Event()

    def run(self, plan: dict) -> dict:
        if self.alarm.is_set():
            return {"error": LIE_DETECTOR_ERROR}
        validate(plan, self.allowed)
        hwnd = window.find_game(self.title)
        if plan.get("focus") and win32.foreground_title() != self.title:
            window.bring_to_front(self.title)
            time.sleep(0.5)
        if win32.foreground_title() != self.title:
            return {"error": f"game not in front (foreground: {win32.foreground_title()!r})"}

        client = window.client_rect(hwnd)
        cap = plan.get("capture", {})
        x, y, w, h = cap.get("crop") or [0, 0, client["width"], client["height"]]
        region = {"left": client["left"] + x, "top": client["top"] + y, "width": w, "height": h}
        scale, lead_ms = cap.get("scale", 1), cap.get("lead_ms", 100)
        fps, duration = cap.get("fps", 0), cap.get("duration_ms", 0)

        t0 = time.perf_counter() + lead_ms / 1000
        sent, held, problems = [], set(), []

        cancel = INBOX / f"{plan['id']}.cancel"  # the client cut the plan short: stop pressing

        def wait(due: float) -> bool:
            """Until `due`, unless the plan is cancelled or the alarm goes off."""
            while True:
                if cancel.exists():
                    problems.append("cancelled")
                    return False
                if not _sleep_until(min(due, time.perf_counter() + 0.015), self.alarm):
                    problems.append(LIE_DETECTOR_ERROR)
                    return False
                if time.perf_counter() >= due:
                    return True

        def press_keys():
            for event in plan.get("events", []):
                if not wait(t0 + event[0] / 1000):
                    return
                if win32.foreground_title() != self.title:
                    problems.append("game left the foreground; stopped pressing keys")
                    return
                if event[1] == "mouse":
                    x, y = event[2], event[3]
                    if x >= client["width"] or y >= client["height"]:
                        problems.append(f"mouse {x},{y} is outside the game; skipped")
                        continue
                    window.move_cursor(client["left"] + x, client["top"] + y)
                else:
                    _, name, kind = event
                    win32.send_key(self.allowed[name], kind == "down")
                    (held.add if kind == "down" else held.discard)(name)
                sent.append([round((time.perf_counter() - t0) * 1000, 1), *event[1:]])
            if plan.get("end_ms") is not None:  # the plan lasts this long (the next one follows on)
                wait(t0 + plan["end_ms"] / 1000)

        presser = threading.Thread(target=press_keys, name="key-player")
        frames, times = [], []
        presser.start()
        try:
            if fps and duration:
                with mss.MSS() as sct:
                    start = t0 - lead_ms / 1000
                    for i in range(int(fps * (duration + lead_ms) / 1000)):
                        if not _sleep_until(start + i / fps, self.alarm):
                            break
                        shot = np.asarray(sct.grab(region))[::scale, ::scale, 2::-1]  # BGRA -> RGB
                        times.append(round((time.perf_counter() - t0) * 1000, 1))
                        frames.append(shot.copy())
        finally:
            presser.join()
            for name in held:
                win32.send_key(self.allowed[name], False)
            cancel.unlink(missing_ok=True)
        result = {"sent": sent, "problems": problems, "region": region, "scale": scale, "client": client}
        if self.alarm.is_set():
            result["error"] = LIE_DETECTOR_ERROR
        if frames:
            np.savez_compressed(OUTBOX / f"{plan['id']}.npz", frames=np.stack(frames), times=np.array(times))
            result["frames"] = len(frames)
        return result


def _log(msg: str) -> None:
    with open(PLAYER_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def _answer(path, result: dict) -> str:
    # Output names come from the sanitized file name, never from the plan's contents.
    plan_id = "".join(ch for ch in path.stem if ch.isalnum() or ch in "-_")[:64] or "plan"
    path.unlink(missing_ok=True)
    tmp = OUTBOX / f"{plan_id}.tmp"
    tmp.write_text(json.dumps(result), encoding="utf-8")
    tmp.replace(OUTBOX / f"{plan_id}.json")
    return plan_id


def serve() -> None:
    for d in (LAB, INBOX, OUTBOX):
        d.mkdir(parents=True, exist_ok=True)
    STOP.unlink(missing_ok=True)
    if ALARM.exists():
        _log("refusing to start: a Lie Detector alarm hasn't been cleared (python -m gamelab clear-alarm)")
        return
    for stale in [*INBOX.glob("*.json"), *INBOX.glob("*.cancel"), *INBOX.glob("*.running")]:  # left over
        stale.unlink()
    cfg = config.load(CONFIG, (c.name for c in COMBOS))
    alarm = threading.Event()
    player = Player(allowed_keys(cfg.keys), cfg.game_window_title, alarm)
    win32.enable_dpi_awareness()
    watchdog = Watchdog(cfg.game_window_title, alarm)
    watchdog.start()
    _log(f"player started; allowed keys {sorted(player.allowed)}; watching for the Lie Detector")
    idle_since = time.monotonic()
    with win32.precise_timers():
        while not STOP.exists() and not alarm.is_set() and time.monotonic() - idle_since < IDLE_LIMIT_S:
            plans = sorted(INBOX.glob("*.json"))
            if not plans:
                time.sleep(0.01)
                continue
            path = plans[0]
            running = path.with_suffix(".running")
            try:
                path.replace(running)  # claimed: from now on the client can't withdraw it, only cancel it
            except OSError:
                continue  # withdrawn just now
            try:
                plan = json.loads(running.read_text(encoding="utf-8"))
                plan["id"] = "".join(ch for ch in path.stem if ch.isalnum() or ch in "-_")[:64] or "plan"
                result = player.run(plan)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
            try:
                plan_id = _answer(running, result)
                _log(f"ran {plan_id}: {result.get('error') or 'ok'}")
            except OSError as e:
                _log(f"couldn't answer {path.stem}: {e}")
                running.unlink(missing_ok=True)
            idle_since = time.monotonic()
    watchdog.stopped.set()
    if alarm.is_set():
        for path in INBOX.glob("*.json"):
            _answer(path, {"error": LIE_DETECTOR_ERROR})
        _log("LIE DETECTOR seen: input stopped and the player shut down")
    _log("player stopped")
