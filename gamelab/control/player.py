"""Admin-side key player and frame recorder for in-game tests.

MapleStory runs as administrator, and Windows blocks key presses from normal programs into admin
windows. So this one process runs elevated (one UAC prompt per test session) and everything else
runs normally and talks to it through files. To keep that safe it only ever:

- reads plans from gamelab/lab/inbox (plain JSON data, never code),
- presses keys (any key config.toml can name: ALL_KEYS) and the mouse buttons, only while the game is in
  front,
- moves the mouse pointer and clicks only inside the game window,
- resizes the game window to one of window.CLIENT_SIZES when a plan asks (`window_client`; a normal program can't),
- watches the game for the Lie Detector (the game's anti-bot check) the whole time it runs; if it
  appears, input stops at once (all that goes out after is letting go of the keys and buttons a plan
  held), the player pages whoever is at the PC, records the Lie Detector's whole cycle
  (control.lie_recorder) while refusing every plan, and stops once a person has run `python -m gamelab
  clear-alarm` and the Lie Detector has been off the screen for a minute; it won't start again until
  the alarm is cleared (it never tries to answer it),
- records the game's client area while a plan runs,
- writes results to gamelab/lab/outbox.

Its code and key list are loaded once at startup. It exits when gamelab/lab/STOP appears, after 24 idle
hours (not while it records a Lie Detector) or after a Lie Detector, as above: plans are taken again
only by a player started again, which takes the user's UAC click. Start it with `python -m gamelab
player --elevate`; stop it with `python -m gamelab stop`.

Of gamelab, this module itself imports only paths and window (and control.alarm, control.lie_recorder and
perception.lie_detector when a Watchdog is created). Plans come from control.client.
"""

import json
import os
import threading
import time
import traceback
from dataclasses import fields

import cv2
import mss
import numpy as np

from maplehelper import config, win32
from maplehelper.combos import COMBOS
from maplehelper.config import GameKeys
from maplehelper.keys import KEY_CODES, Key
from maplehelper.keys import parse as parse_key

from .. import window
from ..paths import ALARM, CONFIG, INBOX, LAB, LIE_RECORDINGS, OUTBOX, PLAYER_LOG, STOP

# It used to stop after 3 idle hours; on 2026-10-23 the user asked that it not be exited proactively (starting it
# again needs their UAC click), and it had stopped by itself overnight: a day, so it outlasts a night without plans.
IDLE_LIMIT_S = 24 * 60 * 60
MAX_EVENTS = 400
MAX_PLAN_MS = 20_000
MAX_FRAMES = 900
LIE_DETECTOR_ERROR = "Lie Detector on screen: all input stopped"
# Past this, the player's log becomes player.log.1 (the one before goes) when the player starts. It writes
# about 40 bytes a plan, some 150 KB an hour of hunting, and was never cut: 5 MB is about 30 hours.
LOG_MAX_BYTES = 5 * 1024 * 1024

# Every key config.toml can name, and both mouse buttons: since 2026-10-23 the player may press any of
# them (the key bindings window, \, needs clicks to save a change). The Lie Detector still stops all of it.
ALL_KEYS = tuple(KEY_CODES)
MOUSE_BUTTONS = ("lbutton", "rbutton")


def allowed_keys(keys: GameKeys | None = None, extra=ALL_KEYS) -> dict[str, Key]:
    """Her bound in-game actions and every other key (`extra`), by name."""
    bound = ([getattr(keys, f.name) for f in fields(GameKeys)] if keys else []) + [parse_key(n) for n in extra]
    return {k.name: k for k in bound if k}


def validate(plan: dict, allowed: dict[str, Key]) -> None:
    """Raise ValueError unless the plan is well-formed and only uses allowed keys.

    Events are [ms, key, 'down'|'up'] (a key, or a mouse button: 'lbutton', 'rbutton', where the pointer
    is) or [ms, 'mouse', x, y] (move the pointer to client pixel x, y).
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
            if name not in allowed and name not in MOUSE_BUTTONS:
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
    size = plan.get("window_client")
    if size is not None and (not isinstance(size, list) or tuple(size) not in window.CLIENT_SIZES):
        raise ValueError(f"window_client must be one of {window.CLIENT_SIZES}")
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


def _sleep_until(t: float, stop: threading.Event | None = None) -> bool:
    """Sleep until perf_counter() reaches t. Returns False early if `stop` gets set."""
    while (left := t - time.perf_counter()) > 0:
        if stop is not None and stop.is_set():
            return False
        # Sleep in steps of at most 10 ms (so `stop` is seen) that wake 2 ms early; spin under 4 ms.
        time.sleep(min(left - 0.002, 0.01) if left > 0.004 else 0)
    return stop is None or not stop.is_set()


def alarm_up() -> bool:
    """Whether the alarm file (paths.ALARM) is there; when that can't be told, it is (the lock stays). Asked of
    os.stat: Path.exists answers False for any error (Python 3.15 here), so an alarm file it couldn't look at had
    read as cleared."""
    try:
        os.stat(ALARM)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


class Watchdog(threading.Thread):
    """Looks at the game every `interval` seconds, whether or not a plan is running, and records each Lie Detector's
    whole cycle.

    A cycle starts when the Lie Detector is first seen: `alarm` is set first (any keys stop within ~10 ms, mid-plan
    too), then on_alarm writes the alarm and its screenshot and pages (control.alarm.raise_alarm), as before; only
    then does the recording start (control.lie_recorder), with that first frame. From there every grab goes into it
    until the game has shown no Lie Detector for `end_s` (lie_recorder.END_S): the cycle's end; or, at the latest,
    `cap_s` after the first sight, when the recording stops and the cycle goes on unrecorded. A grab with the game
    window gone, minimized or covered by another window (checked for the Lie Detector all the same) is recorded as
    such, with no frame, and doesn't count as the Lie Detector gone.

    `alarm` is never cleared: the player takes plans again only once a person has started it again (`player
    --elevate`: the user's UAC click), as when it shut down at the alarm. `cleared` is set, and the player stops
    (serve), once both a person has cleared the alarm (the alarm file, paths.ALARM, gone: `python -m gamelab
    clear-alarm`) and the game has shown no Lie Detector for `end_s` since the cycle; until then it watches, and
    records a new cycle as any. Taking plans again by itself, a file any program can delete (`clear-alarm` is an
    ordinary command) would be all that stood between the keys and a Lie Detector the spotter can't see (the figure
    tracking: lie_recorder.END_S). Seen again with the alarm file gone (cleared while it's still up), it's raised
    again, paged too, so gamelab's commands go on refusing. An alarm file that never got written (the disk full)
    can't be cleared: then the player stays up, locked, until it's stopped.

    `grab` (for tests) stands in for grabbing the game: it returns a frame (RGB), (a frame, "covered"), or why
    there's none ("no window", "minimized"; None is "no window"). Nothing that goes wrong in a look, the recording or
    the log stops the watching: it's logged and the next look comes."""

    def __init__(self, game_title: str, alarm: threading.Event, spotter=None, grab=None,
                 interval: float = 0.25, on_alarm=None, folder=None, log=None, end_s=None, cap_s=None):
        super().__init__(name="lie-detector-watch", daemon=True)
        from ..perception.lie_detector import LieDetectorSpotter
        from . import lie_recorder
        from .alarm import raise_alarm
        self.title = game_title
        self.alarm = alarm
        self.spotter = spotter or LieDetectorSpotter()
        self.grab = grab
        self.interval = interval
        self.on_alarm = on_alarm or raise_alarm
        self.folder = folder or LIE_RECORDINGS
        self.log = log or _log
        self.end_s = lie_recorder.END_S if end_s is None else end_s
        self.cap_s = lie_recorder.CAP_S if cap_s is None else cap_s
        self.Recording = lie_recorder.Recording
        self.stopped = threading.Event()
        self.cleared = threading.Event()      # the alarm cleared by a person and the screen clear since: the player
                                              # stops (serve)
        self.cycle_t0: float | None = None    # when the cycle in progress was first seen (monotonic s), else None
        self.gone_since: float | None = None  # since when the game has shown no Lie Detector (None: seen, or the game
                                              # not showing, since the last look)
        self.recording = None                 # the cycle's recording while it's written
        self.ended: list = []                 # recordings finished, their writers maybe still at it
        self.troubles: dict[str, int] = {}
        self.alarm_written = alarm.is_set()   # whether the alarm file was there since `alarm` was set: its going
                                              # is a person clearing it

    def run(self) -> None:
        with mss.MSS() as sct:
            due = time.monotonic()
            while not self.stopped.is_set():
                try:
                    self.tick(time.monotonic(), sct)
                except Exception as e:  # nothing stops the watching
                    self._trouble("a look failed", e)
                # 4 a second from the start of each look (it waited `interval` after each, about 3 a second with the
                # look's own 70 ms: 3.10 a second on a 1366x768 frame, D:/MapleHelper-scratch/ld/rate_measure.py);
                # one that ran long (the paging's beeps) is followed at once, not caught up.
                due = max(due + self.interval, time.monotonic())
                self.stopped.wait(due - time.monotonic())
        if self.recording is not None:
            self._end_recording("stopped")
        for rec in self.ended:
            rec.join(timeout=20)  # its last frames and summary written before the player exits

    def tick(self, now: float, sct=None) -> None:
        """One look at the game (`now`: monotonic s), and what follows from it."""
        frame, why = self._look(sct)
        scores = None
        if frame is not None:
            try:
                scores = self.spotter.scores(frame)
            except Exception as e:  # (a frame it can't take, such as one smaller than its templates)
                self._trouble("the Lie Detector check failed", e)
                frame, why = None, "check failed"
        seen = scores is not None and max(scores.values()) >= 1
        shown = frame is not None and why is None  # the game itself, nothing over it: only then is the frame saved
        if seen or not shown:
            self.gone_since = None  # a game not showing (gone, minimized, covered) may have it up behind
        elif self.gone_since is None:
            self.gone_since = now
        if seen and self.cycle_t0 is None:
            self.alarm.set()  # first: keys stop within ~10 ms
            info = self._raise(frame)  # the alarm, its screenshot, paging: before anything of the recording
            self.cycle_t0 = now
            self.alarm_written = alarm_up()
            self._say("LIE DETECTOR seen: input stopped; the player stays up refusing every plan, records it, and "
                      f"stops once `clear-alarm` has been run and none has been on screen for {self.end_s:g} s "
                      "(it takes plans again only once it's started again)"
                      + ("" if self.alarm_written else " (the alarm file couldn't be written: locked until it's "
                                                       "stopped)"))
            self._start_recording(info)
        elif seen and self.alarm_written and not alarm_up():
            # Cleared while it's still up (the cycle not over): raised again, so gamelab's commands go on refusing
            # (exit code 3) and the player doesn't stop with it on screen; paged again, as nobody may be at it.
            self._raise(frame)
            self.alarm_written = alarm_up()
            self._say("the Lie Detector alarm was cleared while the Lie Detector is still on screen: raised again "
                      f"(clear it once it has been off the screen for {self.end_s:g} s)")
        if self.recording is not None:
            self.recording.add(now - self.cycle_t0, frame if shown else None, scores, why)
            if now - self.cycle_t0 >= self.cap_s:
                self._end_recording("cap")
        if self.cycle_t0 is not None and self.gone_for(now):
            if self.recording is not None:
                self._end_recording("gone")
            self.cycle_t0 = None
            self._say(f"the Lie Detector has been off the screen for {self.end_s:g} s: its cycle is over"
                      + (" (waiting for `clear-alarm`, then the player stops)" if alarm_up() else ""))
        if self.alarm.is_set() and not self.cleared.is_set():
            if alarm_up():
                self.alarm_written = True
            elif self.alarm_written and self.cycle_t0 is None and self.gone_for(now):
                self.cleared.set()  # a person cleared it, and none on screen for end_s: the player stops
                self._say(f"the Lie Detector alarm was cleared and none has been on screen for {self.end_s:g} s: "
                          "the player stops (it takes plans again once it's started again: `python -m gamelab "
                          "player --elevate`)")

    def gone_for(self, now: float) -> bool:
        """Whether the game has shown no Lie Detector for end_s, every look of it showing the game."""
        return self.gone_since is not None and now - self.gone_since >= self.end_s

    def _look(self, sct):
        """(the frame, None); (the frame, "covered"): another window is over some of the game, the frame being
        whatever is on top; or (None, why there's none)."""
        try:
            got = self.grab() if self.grab else self._grab(sct)
        except Exception as e:
            self._trouble("grabbing the game failed", e)
            return None, "grab failed"
        if isinstance(got, tuple):
            return got
        if got is None or isinstance(got, str):
            return None, got or "no window"
        return got, None

    def _grab(self, sct):
        """The game's client area (RGB); with another window over some of it, (that, "covered"); or why there's none:
        "no window" or "minimized"."""
        try:
            hwnd = window.find_game(self.title)
        except RuntimeError:
            return "no window"
        # Minimized, its client area isn't 0 x 0 (GetClientRect gave 432x54 on 2026-10-24): ask Windows.
        if window.minimized(hwnd):
            return "minimized"
        client_area = window.client_rect(hwnd)
        if client_area["width"] <= 0 or client_area["height"] <= 0:
            return "minimized"
        # A screen capture sees whatever is on top (window.shows: the middle and near each corner). Still checked for
        # the Lie Detector, as it always was (it may show round the window over it), but not saved, nor counted as
        # the Lie Detector gone: a terminal opened over the game had gone to disk as frames of it, and after a
        # minute of it ended the cycle as answered with the Lie Detector maybe up behind.
        covered = not window.shows(hwnd, client_area)
        # RGB, full size, a picture of its own: viewed as RGB in place, every OpenCV call on it copied it first
        frame = cv2.cvtColor(np.asarray(sct.grab(client_area)), cv2.COLOR_BGRA2RGB)
        return (frame, "covered") if covered else frame

    def _raise(self, frame) -> dict | None:
        try:
            info = self.on_alarm(frame)
        except Exception as e:  # (the alarm Event is up already: nothing goes out)
            self._trouble("raising the alarm failed", e)
            return None
        return info if isinstance(info, dict) else None

    def _start_recording(self, info) -> None:
        try:
            self.recording = self.Recording(self.folder, info, log=self._say, end_s=self.end_s, cap_s=self.cap_s)
        except Exception as e:  # (no folder: the cycle goes on unrecorded)
            self._trouble("starting the Lie Detector recording failed", e)
            return
        self._say(f"recording the Lie Detector, 4 frames a second until it's been off the screen {self.end_s:g} s: "
                  f"{self.recording.folder}")

    def _end_recording(self, end: str) -> None:
        rec, self.recording = self.recording, None
        try:
            rec.finish(end)  # its writer logs the end once it has written the rest and the summary
        except Exception as e:
            self._trouble("ending the Lie Detector recording failed", e)
        self.ended = [r for r in self.ended if not r.join(0)] + [rec]  # (those still writing, and this one)

    def _trouble(self, what: str, e: Exception) -> None:
        """Logged the first time, then every 1000th (4 a second would fill the log)."""
        n = self.troubles[what] = self.troubles.get(what, 0) + 1
        if n == 1 or n % 1000 == 0:
            self._say(f"{what} ({type(e).__name__}: {e}); {n} time{'s' if n > 1 else ''} so far, watching on")

    def _say(self, msg: str) -> None:
        try:
            self.log(msg)
        except Exception:
            pass  # a log that can't be written stops nothing


class Player:
    """Plays one plan at a time: presses its keys on time (on a thread of their own) while recording
    the game's client area, and stops at once when the alarm goes off or the plan is cancelled."""

    def __init__(self, allowed: dict[str, Key], game_title: str, alarm: threading.Event | None = None):
        self.allowed = allowed
        self.title = game_title
        self.alarm = alarm or threading.Event()

    def run(self, plan: dict) -> dict:
        """Play a plan (checked by `validate` first) and return its result: the keys sent and when,
        any problems, and the frames (saved beside the result) if it records."""
        if self.alarm.is_set():
            return {"error": LIE_DETECTOR_ERROR}
        validate(plan, self.allowed)
        hwnd = window.find_game(self.title)
        if plan.get("window_client"):  # before anything else: the capture's area is the new client area
            window.resize_client(hwnd, *plan["window_client"])
            time.sleep(0.5)  # let the game lay itself out again
        if plan.get("focus") and win32.foreground_title() != self.title:
            window.bring_to_front(self.title)
            time.sleep(0.5)  # let the window come to the front
        if win32.foreground_title() != self.title:
            return {"error": f"game not in front (foreground: {win32.foreground_title()!r})"}

        client_area = window.client_rect(hwnd)
        cap = plan.get("capture", {})
        x, y, w, h = cap.get("crop") or [0, 0, client_area["width"], client_area["height"]]
        region = {"left": client_area["left"] + x, "top": client_area["top"] + y, "width": w, "height": h}
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
                if not _sleep_until(min(due, time.perf_counter() + 0.015), self.alarm):  # 15 ms steps
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
                    if x >= client_area["width"] or y >= client_area["height"]:
                        problems.append(f"mouse {x},{y} is outside the game; skipped")
                        continue
                    window.move_cursor(client_area["left"] + x, client_area["top"] + y)
                else:
                    _, name, kind = event
                    _press(self.allowed, name, kind == "down")
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
                _press(self.allowed, name, False)
            cancel.unlink(missing_ok=True)
        result = {"sent": sent, "problems": problems, "region": region, "scale": scale, "client": client_area}
        if self.alarm.is_set():
            result["error"] = LIE_DETECTOR_ERROR
        if frames:
            np.savez_compressed(OUTBOX / f"{plan['id']}.npz", frames=np.stack(frames), times=np.array(times))
            result["frames"] = len(frames)
        return result


def _press(allowed: dict[str, Key], name: str, down: bool) -> None:
    """Press or release a key or a mouse button by name."""
    if name in MOUSE_BUTTONS:
        win32.send_mouse_button(name, down)
    else:
        win32.send_key(allowed[name], down)


def _log(msg: str) -> None:
    """A line in the player's log (paths.PLAYER_LOG)."""
    with open(PLAYER_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def _rotate_log(limit: int = LOG_MAX_BYTES) -> None:
    """Past `limit` bytes the log is moved to player.log.1, replacing the one before, and a new one starts
    (only when the player starts, so a log is never moved while it's written)."""
    try:
        if PLAYER_LOG.stat().st_size > limit:
            PLAYER_LOG.replace(PLAYER_LOG.with_name(PLAYER_LOG.name + ".1"))
    except OSError:
        pass  # no log yet, or it can't be moved now: it goes on


def _answer(path, result: dict) -> str:
    """Write a plan's result to the outbox (atomically) in place of the plan file at `path`; returns
    the plan id."""
    # Output names come from the sanitized file name, never from the plan's contents.
    plan_id = "".join(ch for ch in path.stem if ch.isalnum() or ch in "-_")[:64] or "plan"
    path.unlink(missing_ok=True)
    tmp = OUTBOX / f"{plan_id}.tmp"
    tmp.write_text(json.dumps(result), encoding="utf-8")
    tmp.replace(OUTBOX / f"{plan_id}.json")
    return plan_id


def serve() -> None:
    """The player: take plans from the inbox in name order and answer each, until STOP, 24 idle hours (not counting
    a Lie Detector's recording, at most lie_recorder.CAP_S) or a Lie Detector dealt with. From a Lie Detector's
    sight it refuses every plan (Player.run: the alarm Event) while the watchdog records it, and it stops once a
    person has cleared the alarm and the screen has been clear since (Watchdog.cleared): plans are taken again only
    by a player started again, which takes the user's UAC click, as when it shut down at the alarm. It won't start
    with the alarm up."""
    for d in (LAB, INBOX, OUTBOX):
        d.mkdir(parents=True, exist_ok=True)
    _rotate_log()
    STOP.unlink(missing_ok=True)
    if alarm_up():
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
    _log(f"player started; {len(player.allowed)} keys and the mouse buttons allowed; watching for the Lie Detector")
    idle_since = time.monotonic()
    with win32.precise_timers():
        while not STOP.exists() and not watchdog.cleared.is_set() and time.monotonic() - idle_since < IDLE_LIMIT_S:
            if watchdog.recording is not None:  # a Lie Detector's recording goes on to its end, idle or not
                idle_since = time.monotonic()
            plans = sorted(INBOX.glob("*.json"))
            if not plans:
                time.sleep(0.01)  # idle: look for plans every 10 ms
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
    # Let the watchdog finish writing the alarm, paging and its recording: it's a background thread, and exiting at
    # once killed it mid-write once (no ALARM.json, an empty screenshot).
    watchdog.join(timeout=30)
    if alarm.is_set():
        for path in INBOX.glob("*.json"):
            _answer(path, {"error": LIE_DETECTOR_ERROR})
        _log("stopping after the Lie Detector, its alarm cleared: start the player again to take plans "
             "(python -m gamelab player --elevate)" if watchdog.cleared.is_set() else
             "stopping with the Lie Detector alarm up (locked)")
    _log("player stopped")
