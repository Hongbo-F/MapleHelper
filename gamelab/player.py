"""Admin-side key player and frame recorder for in-game tests.

MapleStory runs as administrator, and Windows blocks key presses from normal programs into admin
windows. So this one process runs elevated (one UAC prompt per test session) and everything else
runs normally and talks to it through files. To keep that safe it only ever:

- reads plans from gamelab/lab/inbox (plain JSON data, never code),
- presses keys that config.toml binds to her in-game actions plus a few exploring keys (EXPLORE_KEYS),
  never Enter, and only while the game is in front,
- moves the mouse pointer inside the game window (never clicks),
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
from .paths import CONFIG, INBOX, LAB, OUTBOX, PLAYER_LOG, STOP

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


def _sleep_until(t: float) -> None:
    while (left := t - time.perf_counter()) > 0:
        time.sleep(left - 0.002 if left > 0.004 else 0)


class Player:
    def __init__(self, allowed: dict[str, Key], game_title: str):
        self.allowed = allowed
        self.title = game_title

    def run(self, plan: dict) -> dict:
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

        def press_keys():
            for event in plan.get("events", []):
                _sleep_until(t0 + event[0] / 1000)
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

        presser = threading.Thread(target=press_keys, name="key-player")
        frames, times = [], []
        presser.start()
        try:
            if fps and duration:
                with mss.MSS() as sct:
                    start = t0 - lead_ms / 1000
                    for i in range(int(fps * (duration + lead_ms) / 1000)):
                        _sleep_until(start + i / fps)
                        shot = np.asarray(sct.grab(region))[::scale, ::scale, 2::-1]  # BGRA -> RGB
                        times.append(round((time.perf_counter() - t0) * 1000, 1))
                        frames.append(shot.copy())
        finally:
            presser.join()
            for name in held:
                win32.send_key(self.allowed[name], False)
        result = {"sent": sent, "problems": problems, "region": region, "scale": scale, "client": client}
        if frames:
            np.savez_compressed(OUTBOX / f"{plan['id']}.npz", frames=np.stack(frames), times=np.array(times))
            result["frames"] = len(frames)
        return result


def _log(msg: str) -> None:
    with open(PLAYER_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def serve() -> None:
    for d in (LAB, INBOX, OUTBOX):
        d.mkdir(parents=True, exist_ok=True)
    STOP.unlink(missing_ok=True)
    cfg = config.load(CONFIG, (c.name for c in COMBOS))
    player = Player(allowed_keys(cfg.keys), cfg.game_window_title)
    win32.enable_dpi_awareness()
    _log(f"player started; allowed keys {sorted(player.allowed)}")
    idle_since = time.monotonic()
    with win32.precise_timers():
        while not STOP.exists() and time.monotonic() - idle_since < IDLE_LIMIT_S:
            plans = sorted(INBOX.glob("*.json"))
            if not plans:
                time.sleep(0.05)
                continue
            path = plans[0]
            # Output names come from the sanitized file name, never from the plan's contents.
            plan_id = "".join(ch for ch in path.stem if ch.isalnum() or ch in "-_")[:64] or "plan"
            try:
                plan = json.loads(path.read_text(encoding="utf-8"))
                plan["id"] = plan_id
                result = player.run(plan)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
            path.unlink(missing_ok=True)
            tmp = OUTBOX / f"{plan_id}.tmp"
            tmp.write_text(json.dumps(result), encoding="utf-8")
            tmp.replace(OUTBOX / f"{plan_id}.json")
            _log(f"ran {plan_id}: {result.get('error') or 'ok'}")
            idle_since = time.monotonic()
    _log("player stopped")
