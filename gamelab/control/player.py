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
  (control.lie_recorder) while refusing every plan but two kinds (alarm_refusal: a tap of her pet food key,
  a few a minute; her retreat's moves, arrows, jump, up and down, for alarm.RETREAT_WINDOW_S after the
  sighting; neither with the mouse; those two held to it with the alarm file up too, raised by another program; her
  keys for them config.toml's, or where the running commands follow them to, paths.ALARM_KEYS: alarm.AlarmKeys), and
  stops once a person has run `python -m gamelab clear-alarm` and the Lie Detector has been off the screen for a
  minute; it won't start again until the alarm is cleared (it never tries to answer it, and never moves her by
  itself: her retreat is the running hunt's, behavior.retreat, which knows the map and where she is; with no hunt
  running, nothing moves her),
- records the game's client area while a plan runs,
- writes results to gamelab/lab/outbox.

Its code and key list are loaded once at startup. It exits when gamelab/lab/STOP appears, after 24 idle
hours (not while it records a Lie Detector) or after a Lie Detector, as above: plans are taken again
only by a player started again, which takes the user's UAC click. Start it with `python -m gamelab
player --elevate`; stop it with `python -m gamelab stop`.

Of gamelab, this module itself imports only paths, window and control.alarm (its constants: what goes with the alarm
up), and control.lie_recorder, perception.lie_detector and perception.framecheck (its capture anchors; it imports
paths, screen and window) when a Watchdog is created. Plans come from control.client.
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
from ..paths import ALARM, ALARM_KEYS, CONFIG, INBOX, LAB, LIE_RECORDINGS, OUTBOX, PLAYER_LOG, STOP
from .alarm import (ALARM_UNSAFE, LIE_DETECTOR_ERROR, PET_FOOD, PET_FOOD_PER_MIN, PLAN_KINDS, RETREAT, RETREAT_WINDOW_S,
                    AlarmKeys)

# It used to stop after 3 idle hours; on 2026-10-23 the user asked that it not be exited proactively (starting it
# again needs their UAC click), and it had stopped by itself overnight: a day, so it outlasts a night without plans.
IDLE_LIMIT_S = 24 * 60 * 60
MAX_EVENTS = 400
MAX_PLAN_MS = 20_000
MAX_FRAMES = 900
TAP_MAX_MS = 200   # with the alarm up, a pet food plan is over within this of its start (a tap: 40 ms), and a plan
                   # let through starts within it (its lead: 0 from Actions, the client's default 100)
RETREAT_OVER = "the retreat's time after the Lie Detector is over: stopped pressing"
# A Lie Detector cycle's end (a minute without it on screen) waits this long at most for the YOLO shadow's check of a
# frame of that minute (ShadowWorker: it may yet see a pattern board there); past it, a check that hangs (a GPU driver
# stuck) no longer holds the end, nor the player's stop after `clear-alarm`.
SHADOW_WAIT_S = 30.0
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


def alarm_refusal(plan: dict, keys: GameKeys | None, since: float | None, fed: list[float], now: float) -> str | None:
    """Why the player won't run `plan` with the Lie Detector alarm up (None: it will), `since` s after the sighting
    (None: when isn't known), the pet food plans it ran with the alarm up having started at `fed` (time.monotonic(), as
    `now`). Two kinds go (alarm.PLAN_KINDS), and only as their kind says:
    - `pet_food`: one tap of her pet food key (config.toml [keys] pet_food) and nothing else, the whole plan over within
      TAP_MAX_MS of its start (its key down and up, its end: it holds the player no longer); PET_FOOD_PER_MIN in any
      minute at most;
    - `retreat`: her moves alone (her arrows, jump, up and down: walking, jumping, climbing, dropping), every key it
      presses let go by its end, until RETREAT_WINDOW_S after the sighting.
    Neither moves the mouse, clicks, resizes the window or records, and either starts within TAP_MAX_MS (its lead);
    every other plan is refused, as before. `plan` is well-formed already (`validate`)."""
    kind = plan.get("kind")
    if kind not in PLAN_KINDS:
        return "only her pet's food and her retreat go with the alarm up"
    events = plan.get("events", [])
    cap = plan.get("capture", {})
    if plan.get("window_client") is not None or cap.get("fps", 0):
        return "no window resize or recording with the alarm up"
    if any(len(e) != 3 or e[1] in MOUSE_BUTTONS for e in events):
        return "no mouse with the alarm up"
    # Its start bounded: `validate` bounds a plan's lead only by the frames it records, and with none a plan let
    # through could wait any time before its first key, holding the player (a retreat's moves, the stop once the alarm
    # is cleared) while it did.
    lead = cap.get("lead_ms", 100)
    if not (isinstance(lead, (int, float)) and 0 <= lead <= TAP_MAX_MS):
        return f"a plan with the alarm up starts within {TAP_MAX_MS} ms (its lead_ms: {lead!r})"
    if kind == PET_FOOD:
        key = keys.pet_food.name if keys is not None and keys.pet_food is not None else None
        if key is None:
            return "no pet food key is bound"
        end = plan.get("end_ms")
        if [e[1:] for e in events] != [[key, "down"], [key, "up"]] or events[1][0] > TAP_MAX_MS \
                or end is not None and end > TAP_MAX_MS:
            return f"a pet food plan is one tap of {key!r} within {TAP_MAX_MS} ms and nothing else"
        if sum(1 for t in fed if now - t < 60) >= PET_FOOD_PER_MIN:
            return f"{PET_FOOD_PER_MIN} pet foods a minute at most with the alarm up"
        return None
    if since is None:
        return "when the Lie Detector was seen isn't known"
    if since > RETREAT_WINDOW_S:
        return f"her retreat's {RETREAT_WINDOW_S:g} s after the Lie Detector are over"
    moves = {k.name for k in (keys.left, keys.right, keys.jump, keys.up, keys.down) if k} if keys is not None else set()
    held = set()
    for _, name, how in events:
        if name not in moves:
            return f"{name!r} isn't one of her moves' keys (her arrows, jump, up, down)"
        (held.add if how == "down" else held.discard)(name)
    if held:
        return f"a retreat lets go of every key it presses ({', '.join(sorted(held))} held at its end)"
    return None


# The key thread waits for a key's time in 15 ms steps (the cancel file and a retreat's window looked at between), each
# slept on the alarm (`halt.wait`: it ends at once when the alarm goes off), and the last FINAL_S before the key as ever
# (_sleep_until: time.sleep's precise steps, the last 4 ms spun). Every step was _sleep_until's, spun at its end: 0.12
# cores while plans ran back to back, about 0.02 so (D:/MapleHelper-scratch/cpu/apply/keywait.py). FINAL_S is more
# than an alarm wait's lateness (1-2 ms with the 1 ms timer serve asks for, up to a tick of 15.6 ms without).
WAIT_STEP_S = 0.015
FINAL_S = 0.02


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


def _alarm_age() -> float | None:
    """Seconds since the Lie Detector was seen by the alarm file's `at` (alarm.raise_alarm: time.time()), or None if it
    can't be told (no file, no `at`, an `at` to come)."""
    try:
        at = json.loads(ALARM.read_text(encoding="utf-8")).get("at")
    except (OSError, ValueError, AttributeError):
        return None
    now = time.time()
    return now - at if isinstance(at, (int, float)) and at <= now else None


class Alarm(threading.Event):
    """The Lie Detector seen: set by the Watchdog, never cleared; `at`, when it was first set (time.monotonic()), from
    which the player takes her retreat's moves for alarm.RETREAT_WINDOW_S (alarm_refusal)."""

    def __init__(self):
        super().__init__()
        self.at: float | None = None

    def set(self) -> None:
        if self.at is None:
            self.at = time.monotonic()
        super().set()


class ShadowWorker(threading.Thread):
    """The YOLO shadow's checks (perception.lie_detector_yolo: pattern boards, recorded, never acted on), on a thread of
    their own: the watchdog hands it a frame of a cycle (`offer`) and grabs on at once; when it's free it checks the
    newest frame handed it (those handed it meanwhile go unchecked), and the result waits, however late, for the
    watchdog to take it (`take`) onto the frame it was of. First it warms the shadow up (its `warm`, if it has one:
    the model loaded and a first check), when the player starts: measured 5.0 s to import torch and ultralytics and 3.9 s
    to load the model and check once on an idle PC.

    On the watchdog's thread (f404c55) the first sighting's check loaded all that there, and the 21:02 recording of
    2026-10-24 has no frame from its first, at 0 s, to 16.1 s, by when the Truth Booth showed: the mouse mini-game was
    lost (and the alarm itself, set only after the check, waited as long). A check that fails is handed back as such
    (logged by the watchdog); a warm-up that fails turns the shadow off for good (each check would load it again)."""

    def __init__(self, shadow, log=None):
        super().__init__(name="lie-detector-shadow", daemon=True)
        self.shadow = shadow
        self.log = log or (lambda msg: None)
        self.stopped = threading.Event()
        self.ready = threading.Event()  # warmed up (or that failed): checks are made from now on
        self.off = False                # its warm-up failed: no checks
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._next: tuple | None = None   # the newest frame handed it, not taken yet: (t, frame, tag)
        self._busy: float | None = None   # the t of the frame being checked
        self._done: list[tuple] = []      # checks back, not taken yet: (t, frame, tag, score, suggestions, error)

    def offer(self, t: float, frame: np.ndarray, tag=None) -> None:
        """A frame grabbed at `t` (the watchdog's clock), to check when it's free (in place of any handed it before
        and not taken yet); `tag` comes back with its result."""
        if self.off:
            return
        with self._lock:
            self._next = (t, frame, tag)
        self._wake.set()

    def take(self) -> list[tuple]:
        """The checks back since the last take, oldest first: (t, frame, tag, score, suggestions, error), score and
        suggestions None if it failed (error: the exception)."""
        with self._lock:
            out, self._done = self._done, []
        return out

    def out(self) -> list[float]:
        """The grab times of the frames whose checks aren't taken yet: being checked, waiting, or back."""
        with self._lock:
            return ([t for t in (self._busy, self._next and self._next[0]) if t is not None]
                    + [r[0] for r in self._done])

    def idle(self) -> bool:
        """Whether nothing is being checked or waiting to be (what's back may be waiting to be taken)."""
        with self._lock:
            return self._busy is None and self._next is None

    def run(self) -> None:
        self._warm_up()
        self.ready.set()
        while not self.stopped.is_set():
            if not self._wake.wait(0.2):
                continue
            with self._lock:
                self._wake.clear()
                job, self._next = self._next, None
                if job is None or self.off:
                    continue
                self._busy = job[0]
            t, frame, tag = job
            try:
                score, suggestions = self.shadow.check(frame)
                result = (t, frame, tag, float(score), suggestions, None)
            except Exception as e:  # handed back: the watchdog logs it
                result = (t, frame, tag, None, None, e)
            with self._lock:
                self._done.append(result)
                self._busy = None

    def _warm_up(self) -> None:
        warm = getattr(self.shadow, "warm", None)
        if warm is None:
            return
        started = time.monotonic()
        try:
            warm()
        except Exception as e:
            self.off = True
            self._say(f"the YOLO shadow couldn't be loaded ({type(e).__name__}: {e}): it's off, the Lie Detector "
                      "watched and recorded as ever")
            return
        # The device it checks on, said (lie_detector_yolo's `device`: 0, the GPU, or "cpu"): on the CPU a cycle's
        # checks took 6.4-6.7 cores against 0.09-0.10 on the GPU (D:/MapleHelper-scratch/cpu/ldwatch/yolo_bench.py).
        device = getattr(self.shadow, "device", None)
        on = "" if device is None else " on the CPU" if device == "cpu" else f" on GPU {device}"
        self._say(f"the YOLO shadow is ready{on}: loaded and a first check in {time.monotonic() - started:.1f} s, off "
                  "the watch")

    def _say(self, msg: str) -> None:
        try:
            self.log(msg)
        except Exception:
            pass  # a log that can't be written stops nothing


class Watchdog(threading.Thread):
    """Looks at the game every `interval` seconds, whether or not a plan is running, and records each Lie Detector's
    whole cycle.

    A cycle starts when the Lie Detector is first seen, in any of its forms (`form`: the one it was seen as, the best
    match; since 2026-10-25 also the mini-game's notice and the image quiz's warning, ahead of their tests): `alarm` is
    set first (any keys stop within ~10 ms, mid-plan too), then on_alarm writes the alarm, named for that form, and its
    screenshot and pages (control.alarm.raise_alarm), as before; only
    then does the recording start (control.lie_recorder), with that first frame. From there every grab goes into it
    until the game has shown no Lie Detector for `end_s` (lie_recorder.END_S): the cycle's end; or, at the latest,
    `cap_s` after the first sight, when the recording stops and the cycle goes on unrecorded. A grab with the game
    window gone or minimized is recorded as such, with no frame, and doesn't count as the Lie Detector gone; one with
    another window over some of the game (checked for the Lie Detector all the same: the screen as it is there) is
    recorded with no frame too, but counts as gone when the check sees none (the user, 2026-10-29: "count seconds when
    game window is covered": this app's window over a corner of the game had kept the player locked after the answer).

    `shadow` (True: the YOLO checkpoint, perception.lie_detector_yolo; or one made for tests; None or False: none)
    checks the game's frames of a cycle for the mouse mini-game's pattern board on a thread of its own (ShadowWorker),
    warmed up from the start: a look never waits for it, so the frames keep coming 4 a second whatever it does. Its
    checks come back late, each recorded with the frame it was of, and a board seen keeps the cycle going as a sight of
    the Lie Detector at that frame would (`_shadow_back`; the end waits for its check of the cycle's last minute,
    SHADOW_WAIT_S at most); it never starts a cycle, nor sends anything.

    `alarm` is never cleared: the player takes plans again only once a person has started it again (`player
    --elevate`: the user's UAC click), as when it shut down at the alarm. `cleared` is set, and the player stops
    (serve), once both a person has cleared the alarm (the alarm file, paths.ALARM, gone: `python -m gamelab
    clear-alarm`) and the game has shown no Lie Detector for `end_s` since the cycle; until then it watches, and
    records a new cycle as any. Taking plans again by itself, a file any program can delete (`clear-alarm` is an
    ordinary command) would be all that stood between the keys and a Lie Detector the spotter can't see (the figure
    tracking: lie_recorder.END_S). Seen again with the alarm file gone (cleared while it's still up), it's raised
    again, paged too, so gamelab's commands go on refusing. An alarm file that never got written (the disk full)
    can't be cleared: then the player stays up, locked, until it's stopped.

    A look matches each form only around the places it has always shown (the spotter's `near`) on a frame of the
    game's own size, 1366x768, whose picture is where it belongs: either capture anchor in its place (the minimap's
    title strip, the HUD's border: perception.framecheck.anchors_off, 0.04 ms). Every lie_detector.WHOLE_EVERY-th look
    (the first too) is of the whole frame (`scores`): a form shown anywhere else on such a frame is seen within that
    many looks, 2 s. The whole frame is looked at every time while a cycle is in progress (what it records is as it
    ever was), on a frame of any other size (a client of 1920x1080, or of another size brought to it: no anchors are
    measured there), and with neither anchor in its place: the game drawing its picture off its client (2026-10-25
    23:40 and 2026-10-26 00:29, 18 px left and 85 up) moves every form off its places too, and `near` saw none of the
    2,230 such frames the whole frame sees (D:/MapleHelper-scratch/cpu/review-safety/ld). Of the 16,050 saved frames
    at 1366x768 (the recordings, before/, the rolling saves, the clips, the fixtures), the 3,809 that show a form all
    have an anchor in place, and every one of the 16,050 drawn off so has neither (D:/MapleHelper-scratch/cpu/fix/
    anchors_scan.py). A check that fails says the whole frame. At 4 looks a second on saved hunt frames, 0.07-0.12
    cores against 0.55-0.58 with the whole frame every look (D:/MapleHelper-scratch/cpu/fix/out/watch.txt). A spotter
    without `near` (the tests' own) is asked for `scores` every time.

    `grab` (for tests) stands in for grabbing the game: it returns a frame (RGB), (a frame, "covered"), or why
    there's none ("no window", "minimized"; None is "no window"). Nothing that goes wrong in a look, the recording or
    the log stops the watching: it's logged and the next look comes."""

    def __init__(self, game_title: str, alarm: threading.Event, spotter=None, grab=None,
                 interval: float = 0.25, on_alarm=None, folder=None, log=None, end_s=None, cap_s=None,
                 shadow=False):
        super().__init__(name="lie-detector-watch", daemon=True)
        from ..perception.framecheck import anchors_off
        from ..perception.lie_detector import NATIVE_FRAME, WHOLE_EVERY, LieDetectorSpotter
        from ..perception.lie_detector_yolo import FORM, YoloPatternShadow
        from . import lie_recorder
        from .alarm import raise_alarm
        self.title = game_title
        self.alarm = alarm
        self.spotter = spotter or LieDetectorSpotter()
        self.whole_every = WHOLE_EVERY
        self.looks = 0                        # looks with a frame so far (every whole_every-th of the whole frame)
        self.anchors_off, self.native = anchors_off, NATIVE_FRAME
        self.shadow = (YoloPatternShadow() if shadow is True else shadow) or None
        self.shadow_form = FORM
        self.grab = grab
        self.interval = interval
        self.form: str | None = None  # the form the cycle in progress was first seen as (the spotter's name)
        self.on_alarm = on_alarm or (lambda frame: raise_alarm(frame, form=self.form))
        self.folder = folder or LIE_RECORDINGS
        self.log = log or _log
        self.end_s = lie_recorder.END_S if end_s is None else end_s
        self.cap_s = lie_recorder.CAP_S if cap_s is None else cap_s
        self.Recording = lie_recorder.Recording
        self.stopped = threading.Event()
        self.looked = threading.Event()       # its first look is done: the player takes no plan before (serve)
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
        self.filed_at: float | None = None    # the last look that found the alarm file there, `alarm` set
        # The YOLO shadow's checks, on a thread of their own, warming up from now (the player's start): never in the
        # looks, which grab and record on meanwhile.
        self.shadow_worker = ShadowWorker(self.shadow, self._say) if self.shadow is not None else None
        if self.shadow_worker is not None:
            self.shadow_worker.start()

    def run(self) -> None:
        # OpenCV on this thread alone (its 24 threads cost 0.27 cores more at 4 looks a second, for no less time a look:
        # the player had it only from ultralytics' import, the shadow's, setting it to 0, seconds after the start)
        cv2.setNumThreads(1)
        with mss.MSS() as sct:
            due = time.monotonic()
            while not self.stopped.is_set():
                try:
                    self.tick(time.monotonic(), sct)
                except Exception as e:  # nothing stops the watching
                    self._trouble("a look failed", e)
                self.looked.set()
                # 4 a second from the start of each look (it waited `interval` after each, about 3 a second with the
                # look's own 70 ms: 3.10 a second on a 1366x768 frame, D:/MapleHelper-scratch/ld/rate_measure.py);
                # one that ran long (the paging's beeps) is followed at once, not caught up.
                due = max(due + self.interval, time.monotonic())
                self.stopped.wait(due - time.monotonic())
        if self.recording is not None:
            self._end_recording("stopped")
        if self.shadow_worker is not None:
            self.shadow_worker.stopped.set()  # (not waited for: a check that hangs holds up nothing)
        for rec in self.ended:
            rec.join(timeout=20)  # its last frames and summary written before the player exits

    def tick(self, now: float, sct=None) -> None:
        """One look at the game (`now`: monotonic s), and what follows from it: first the YOLO shadow's checks come
        back since the last (`_shadow_back`), then the look."""
        self._shadow_back(now)
        frame, why = self._look(sct)
        scores = None
        if frame is not None:
            try:
                scores = self._scores(frame)
            except Exception as e:  # (a frame it can't take, such as one smaller than its templates)
                self._trouble("the Lie Detector check failed", e)
                frame, why = None, "check failed"
        seen = scores is not None and max(scores.values()) >= 1
        shown = frame is not None and why is None  # the game itself, nothing over it: only then is the frame saved
        if seen or frame is None:
            self.gone_since = None  # a game not showing (gone, minimized) may have it up behind; a covered one counts
        elif self.gone_since is None:
            self.gone_since = now
        if seen and self.cycle_t0 is None:
            self.alarm.set()  # first: keys stop within ~10 ms
            self.form = max(scores, key=scores.get)
            info = self._raise(frame)  # the alarm, its screenshot, paging: before anything of the recording
            self.cycle_t0 = now
            self.alarm_written = alarm_up()
            self._say(f"LIE DETECTOR seen ({self.form}): input stopped; the player stays up refusing every plan (but a "
                      f"tap of her pet food, and her retreat's moves for {RETREAT_WINDOW_S:g} s), records it, and "
                      f"stops once `clear-alarm` has been run and none has been on screen for {self.end_s:g} s "
                      "(it takes plans again only once it's started again)"
                      + ("" if self.alarm_written else " (the alarm file couldn't be written: locked until it's "
                                                       "stopped)"))
            self._start_recording(info)
        elif seen and self.alarm_written and not alarm_up():
            self._raise_again(frame)
        number = None
        if self.recording is not None:
            number = self.recording.add(now - self.cycle_t0, frame if shown else None, scores, why)
            if now - self.cycle_t0 >= self.cap_s:
                self._end_recording("cap")
        # The shadow checks the game's own frames of a cycle (a covered one counts as the Lie Detector up anyway, and
        # isn't saved), on its thread: it can keep the cycle going through the pattern play, but neither start one nor
        # send anything.
        if self.shadow_worker is not None and self.cycle_t0 is not None and shown:
            self.shadow_worker.offer(now, frame, (self.recording, number))
        if self.cycle_t0 is not None and self.gone_for(now) and not self._shadow_out(now):
            if self.recording is not None:
                self._end_recording("gone")
            self.cycle_t0 = None
            self._say(f"the Lie Detector has been off the screen for {self.end_s:g} s: its cycle is over"
                      + (" (waiting for `clear-alarm`, then the player stops)" if alarm_up() else ""))
        if self.alarm.is_set() and not self.cleared.is_set():
            if alarm_up():
                self.alarm_written, self.filed_at = True, now
            elif self.alarm_written and self.cycle_t0 is None and self.gone_for(now):
                self.cleared.set()  # a person cleared it, and none on screen for end_s: the player stops
                self._say(f"the Lie Detector alarm was cleared and none has been on screen for {self.end_s:g} s: "
                          "the player stops (it takes plans again once it's started again: `python -m gamelab "
                          "player --elevate`)")

    def _scores(self, frame) -> dict[str, float]:
        """The spotter's scores of a look's frame: near the forms' places (`near`), or of the whole frame (`scores`) on
        every whole_every-th look, the first too, while a cycle is in progress, and on a frame whose places aren't
        known: of another size than the game's own, or its picture off its place (`in_place`; see the top)."""
        self.looks += 1
        near = getattr(self.spotter, "near", None)
        if (near is None or self.looks % self.whole_every == 1 % self.whole_every or self.cycle_t0 is not None
                or not self.in_place(frame)):
            return self.spotter.scores(frame)
        return near(frame)

    def in_place(self, frame) -> bool:
        """Whether a frame is of the game's own size (lie_detector.NATIVE_FRAME) with its picture where it belongs:
        either capture anchor in its place (framecheck.anchors_off). False if that can't be told (the minimap title's
        picture unreadable, say): the whole frame is looked at then."""
        try:
            return (frame.shape[1], frame.shape[0]) == self.native and not self.anchors_off(frame)
        except Exception as e:
            self._trouble("the capture anchors' check failed", e)
            return False

    def gone_for(self, now: float) -> bool:
        """Whether the game has shown no Lie Detector for end_s, every look of it having a frame (the game, or the
        screen over it where another window covers some of it)."""
        return self.gone_since is not None and now - self.gone_since >= self.end_s

    def _raise_again(self, frame) -> None:
        """Cleared while it's still up (the cycle not over): raised again, so gamelab's commands go on refusing (exit
        code 3) and the player doesn't stop with it on screen; paged again, as nobody may be at it."""
        self._raise(frame)
        self.alarm_written = alarm_up()
        self._say("the Lie Detector alarm was cleared while the Lie Detector is still on screen: raised again "
                  f"(clear it once it has been off the screen for {self.end_s:g} s)")

    def _shadow_back(self, now: float) -> None:
        """The YOLO shadow's checks back since the last look (ShadowWorker), however late: each onto the frame it was
        of in the cycle's recording (Recording.add_shadow), if that's still being written; and a pattern board seen
        (a score of 1 or more) is the Lie Detector seen then, in the cycle going on, as the spotter's sight would be: the
        stretch without it starts after that frame (here: at the next look without it, as late as the check), and an
        alarm cleared since that frame's look is raised again. A check of a cycle over is left: the shadow can't start
        one."""
        if self.shadow_worker is None:
            return
        for t, frame, (rec, number), score, suggestions, error in self.shadow_worker.take():
            if error is not None:  # the template alarm and frame recording still work without YOLO
                self._trouble("the YOLO shadow check failed", error)
                continue
            if rec is not None and rec is self.recording:
                rec.add_shadow(t - self.cycle_t0, number, self.shadow_form, score, suggestions, late=now - t)
            if score < 1 or self.cycle_t0 is None or t < self.cycle_t0:
                continue
            if self.gone_since is not None and self.gone_since <= t:
                self.gone_since = None
            if self.alarm_written and (self.filed_at is None or t > self.filed_at) and not alarm_up():
                self._raise_again(frame)

    def _shadow_out(self, now: float) -> bool:
        """Whether the cycle's end waits for the YOLO shadow: its check of a frame of the stretch without the Lie
        Detector (grabbed from gone_since until the end came due) isn't back yet, SHADOW_WAIT_S past that due at most (a
        check that hangs holds the end, and the player's stop, no longer)."""
        if self.shadow_worker is None or self.gone_since is None:
            return False
        due = self.gone_since + self.end_s
        return now - due < SHADOW_WAIT_S and any(self.gone_since <= t < due for t in self.shadow_worker.out())

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
        # RGB, full size, a picture of its own: viewed as RGB in place, every OpenCV call on it copied it first. A
        # client of another size than the game's own or 1920x1080 (full screen at 3840x2160, resized) is brought to
        # 1920x1080, the templates' size (lie_detector.to_template): as it was, no saved Lie Detector screen was seen
        # at such a size (0.53-0.78 of the thresholds), and Alt+Enter twice puts the game full screen on purpose.
        from ..perception.lie_detector import to_template
        frame = to_template(cv2.cvtColor(np.asarray(sct.grab(client_area)), cv2.COLOR_BGRA2RGB))
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
    the game's client area, and stops at once when the alarm goes off or the plan is cancelled. With the alarm up it
    plays only what alarm_refusal lets through, by `keys` (her bindings: config.toml's at its start), or with
    `keys_file` by those keys with the file's over her moves and pet food (alarm.AlarmKeys, read again whenever it
    changes, none of `unsafe`: where the running commands follow them to, control.keymap), and says so in its log."""

    def __init__(self, allowed: dict[str, Key], game_title: str, alarm: threading.Event | None = None,
                 keys: GameKeys | None = None, log=None, keys_file=None, unsafe=ALARM_UNSAFE):
        self.allowed = allowed
        self.title = game_title
        self.alarm = alarm if alarm is not None else Alarm()
        self.keys = keys
        self.log = log or _log
        self.alarm_keys = None if keys_file is None else AlarmKeys(keys, keys_file, unsafe, log=self.log)
        self.fed: list[float] = []  # when the pet food plans it ran with the alarm up started (time.monotonic())

    def allowance(self) -> GameKeys | None:
        """Her keys as the alarm's allowance takes them now: `keys`, the file's over them (`keys_file`)."""
        return self.keys if self.alarm_keys is None else self.alarm_keys.current()

    def run(self, plan: dict) -> dict:
        """Play a plan (checked by `validate` first) and return its result: the keys sent and when,
        any problems, and the frames (saved beside the result) if it records. With the alarm up already, only a
        plan alarm_refusal lets through: never brought to the front, and played to its end (a retreat's to its
        window's), not stopped by the alarm it came in. A plan of a kind that goes with the alarm up (alarm.PLAN_KINDS)
        is held to the same with the alarm file up though this player hasn't seen the Lie Detector (`watch
        --lie-detector` raised it), its sighting the file's `at`: such a plan gets past the client's check of the file
        (client.send), and had played as any, the game brought to the front while the test was being answered."""
        during = self.alarm.is_set()
        kind = plan.get("kind")
        if during and kind not in PLAN_KINDS:
            return {"error": LIE_DETECTOR_ERROR}
        filed = not during and kind in PLAN_KINDS and alarm_up()
        validate(plan, self.allowed)
        cutoff = None  # (perf_counter) when a retreat's window ends: its keys stop there
        if during or filed:
            at = getattr(self.alarm, "at", None)  # (a plain Event: when isn't known, and no retreat goes)
            since = _alarm_age() if filed else None if at is None else time.monotonic() - at
            if (refused := self._with_alarm(plan, since, filed)) is not None:
                return refused
            if kind == RETREAT:
                cutoff = time.perf_counter() + (RETREAT_WINDOW_S - since)
            during = True  # (filed: as if it came in with this player's alarm)
        hwnd = window.find_game(self.title)
        if plan.get("window_client"):  # before anything else: the capture's area is the new client area
            window.resize_client(hwnd, *plan["window_client"])
            time.sleep(0.5)  # let the game lay itself out again
        # The alarm looked at again after each wait: a Lie Detector seen in one had the game brought to the front all
        # the same (the keys' own waits stop at it, but bringing it forward isn't one of them). With it up already,
        # never: whoever is at the PC is answering it.
        if plan.get("focus") and not during and win32.foreground_title() != self.title and not self.alarm.is_set():
            window.bring_to_front(self.title)
            time.sleep(0.5)  # let the window come to the front
        if self.alarm.is_set() and not during:
            return {"error": LIE_DETECTOR_ERROR}
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
        halt = threading.Event() if during else self.alarm  # (let through with the alarm up: it plays on)

        def wait(due: float) -> bool:
            """Until `due`, unless the plan is cancelled, the alarm goes off, or a retreat's window ends."""
            while True:
                if cancel.exists():
                    problems.append("cancelled")
                    return False
                if cutoff is not None and time.perf_counter() >= cutoff:
                    problems.append(RETREAT_OVER)
                    return False
                end = min(due, cutoff) if cutoff is not None else due
                left = end - time.perf_counter()
                if left > FINAL_S:  # slept (no spin), WAIT_STEP_S at most: the checks above between
                    stopped = halt.wait(min(WAIT_STEP_S, left - FINAL_S))
                else:               # the last FINAL_S: to the key's time as ever
                    stopped = not _sleep_until(end, halt)
                if stopped:
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
        if self.alarm.is_set() and not during:
            result["error"] = LIE_DETECTOR_ERROR
        if frames:
            np.savez_compressed(OUTBOX / f"{plan['id']}.npz", frames=np.stack(frames), times=np.array(times))
            result["frames"] = len(frames)
        return result

    def _with_alarm(self, plan: dict, since: float | None, filed: bool = False) -> dict | None:
        """A plan of a kind that goes with the alarm up, `since` s after the sighting (None: when isn't known): its
        refusal (the Lie Detector error, and why), or None, logged either way (a pet food counted). `filed`: the alarm
        is the alarm file's, this player not having seen the Lie Detector."""
        now = time.monotonic()
        self.fed = [t for t in self.fed if now - t < 60]
        kind, name = plan["kind"], plan.get("id", "?")
        up = "Lie Detector alarm up" + (" (its file's; this player hasn't seen it)" if filed else "")
        if why := alarm_refusal(plan, self.allowance(), since, self.fed, now):
            self.log(f"{up}: refused {name} ({kind}): {why}")
            return {"error": f"{LIE_DETECTOR_ERROR}: {why}"}
        if kind == PET_FOOD:
            self.fed.append(now)
            self.log(f"{up}: her pet's food ({name}), {len(self.fed)} in the last minute")
        else:
            self.log(f"{up}: her retreat's plan {name}, {since:.1f} s after the sighting "
                     f"({sum(1 for e in plan['events'] if e[2] == 'down')} keys; it's cut at {RETREAT_WINDOW_S:g} s)")
        return None


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
    sight it refuses every plan (Player.run: the alarm Event) but the two kinds alarm_refusal lets through (her pet's
    food; her retreat's moves, for alarm.RETREAT_WINDOW_S) while the watchdog records it, and it stops once a
    person has cleared the alarm and the screen has been clear since (Watchdog.cleared): plans are taken again only
    by a player started again, which takes the user's UAC click, as when it shut down at the alarm. It won't start
    with the alarm up, takes no plan before the watchdog's first look at the game, and stops if the watchdog does."""
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
    alarm = Alarm()
    # (never onto maplehelper's own keys either: config.toml's [hotkeys] and [control], hidden from the game while it runs)
    unsafe = (*ALARM_UNSAFE, *(k.name for k in getattr(cfg, "hotkeys", {}).values()),
              *(k.name for k in (getattr(cfg, "abort", None), getattr(cfg, "slow_mode", None)) if k is not None))
    player = Player(allowed_keys(cfg.keys), cfg.game_window_title, alarm, keys=cfg.keys, keys_file=ALARM_KEYS,
                    unsafe=unsafe)
    win32.enable_dpi_awareness()
    watchdog = Watchdog(cfg.game_window_title, alarm, shadow=True)
    watchdog.start()
    now_keys = player.allowance()
    pet = now_keys.pet_food.name if now_keys is not None and now_keys.pet_food is not None else None
    _log(f"player started; {len(player.allowed)} keys and the mouse buttons allowed; watching for the Lie Detector "
         f"(with it seen: {f'a tap of {pet!r} for her pet, {PET_FOOD_PER_MIN} a minute, and ' if pet else ''}her "
         f"retreat's moves for {RETREAT_WINDOW_S:g} s; nothing else; her keys for them "
         + ("config.toml's" if player.alarm_keys.used is None else f"{ALARM_KEYS.name}'s, as followed") + ", and "
         f"{ALARM_KEYS.name}'s whenever it changes)")
    idle_since = time.monotonic()
    with win32.precise_timers():
        while not STOP.exists() and not watchdog.cleared.is_set() and time.monotonic() - idle_since < IDLE_LIMIT_S:
            if watchdog.recording is not None:  # a Lie Detector's recording goes on to its end, idle or not
                idle_since = time.monotonic()
            if not watchdog.is_alive():  # (it can't take a picture: mss refused) nothing would see a Lie Detector
                _log("the Lie Detector watch has stopped: stopping, as nothing would see one")
                break
            plans = sorted(INBOX.glob("*.json"))
            # No plan before the watchdog's first look: a Lie Detector on screen as the player starts (started again
            # with the Truth Booth up) is seen before any key, not up to a look and its 70 ms later.
            if not plans or not watchdog.looked.is_set():
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
