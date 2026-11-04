"""The Lie Detector's Violetta game followed live, on the perception service's own frames (the user, 2026-11-03: "wire
in the code to run this live when the violetta mini game shows up in game", "at the end of the success, click the
ok button of 'you have successfully passed'", and "let's actually click it"). It says where the made-up Violetta is
after each round and clicks that answer's button.

- While the Lie Detector's alarm is up (its file, paths.ALARM, looked at once in ALARM_POLL_S at most) and for
  AFTER_S after, the service's frames go to a thread of its own (`saw`: by reference, the frame loop never waits; at
  most QUEUE frames waiting, the oldest dropped and counted). The model (perception.violetta: YOLO on the GPU) is loaded
  as the alarm goes up, before the game's window shows (its box comes 10 s and its window 15 s more before the
  shuffles).
- The game: from its window's first sight (perception.lie_detector's over_play "countdown": its yellow title at its
  place, looked for every WINDOW_CHECK_S) until it's been gone WINDOW_GONE_S, or GAME_MAX_S. Every frame its boxes
  (perception.violetta.Detector); at each answer window (the buttons lit: buttons_lit, its first frame) the round
  solved from the frames since the one before (perception.violetta.solve_round), the made-up one's slot chained from
  its start (start_slot: the boxes of its face before the first window, at the slots measured then: slots_of), and
  said: the service's log, a `violetta` telemetry record, and a push to the phone (control.notify's Bark: lab/bark.json)
  "Violetta round N: slot S"; and that slot's button clicked (perception.violetta.BUTTON_AT) through the elevated
  player (client.click_lie_answer: alarm.LIE_CURSOR's click form, one move and one left click), cancelled once the
  buttons go dark.
  The round is solved knowing its window's time (solve_round's `window`), 0.45 s or so.
- The pass's dialog ("You have successfully passed the Lie Detector Test", perception.violetta.OkDialog), looked for
  every OK_CHECK_S while the alarm is up or AFTER_S after it went: seen on OK_SEEN looks in a row, its OK button is
  clicked through the elevated player (client.click_lie_ok: one click, inside the button's box, the game the window
  under the pointer: alarm.LIE_OK, player.alarm_refusal), at most OK_TRIES times OK_RETRY_S apart while it stays.
Each game's frames, boxes and answers: the `violetta` telemetry records (event "begin", "round", "answer", "end", "ok").
"""

import os
import threading
import time
from collections import deque

from .. import paths, telemetry

ALARM_POLL_S = 0.5
AFTER_S = 90.0         # frames looked at this long after the alarm has gone (the dialog shows once the game's over)
QUEUE = 45             # frames waiting for the thread at most (3 s at 15 a second)
WINDOW_CHECK_S = 0.25  # the game's window looked for this often before the game
WINDOW_ROUND_S = 1.0   # ... and during it
WINDOW_GONE_S = 3.0    # the window unseen this long: the game is over
GAME_MAX_S = 150.0     # a game this long is over whatever (4 rounds: about 60 s)
OK_CHECK_S = 0.5
OK_SEEN = 2            # looks in a row seeing the dialog before its OK is clicked
OK_RETRY_S = 3.0
OK_TRIES = 3


class VioletaLive:
    """The service's side (see the top): `saw` from the frame loop, `start` and `close` with the service, `line` for its
    status. For tests: `detector` (perception.violetta.Detector's `detect`), `spotter` (its `over_play`), `ok` (the
    dialog's finder: frame -> (x, y) or None), `clicker` (the pass's OK click), `answer` (x, y, active -> bool: the
    answer click through the player), `push` (title, text) and `alarm` (the alarm file's path)."""

    def __init__(self, alarm=None, spotter=None, log=None, detector=None, ok=None, clicker=None, push=None,
                 answer=None):
        self.alarm = alarm
        self.spotter = spotter
        self.log = log or (lambda msg: None)
        self.detector = detector
        self.ok = ok
        self.clicker = clicker or self._click
        self.answer = answer or self._click_answer
        self.answering: threading.Thread | None = None  # the click on the answer's button under way
        self.push = push or self._push
        self._cond = threading.Condition()
        self._queue: deque = deque()
        self.dropped = 0
        self._closing = False
        self._thread: threading.Thread | None = None
        self._polled: float | None = None
        self._up = False
        self._gone_at: float | None = None   # since when the alarm has been down (monotonic)
        self._loading: threading.Thread | None = None
        self.game: dict | None = None        # the game in progress
        self.games = 0
        self.why_off = ""
        self._checked = -1e9
        self._ok_checked = -1e9
        self._ok_seen = 0
        self._ok_clicks: list[float] = []

    # --- the frame loop's side ------------------------------------------------------------------------------------

    def saw(self, st, frame, run) -> None:
        """The frame loop's frame (Service._emit): queued while the alarm is up, a game is on, or AFTER_S after."""
        if self.why_off or frame is None or getattr(frame, "full", None) is None or not getattr(frame, "visible", True):
            return
        now = time.monotonic()
        if self.game is None and not self.alarm_up(now) and (self._gone_at is None or now - self._gone_at > AFTER_S):
            return
        with self._cond:
            if len(self._queue) >= QUEUE:
                self._queue.popleft()
                self.dropped += 1
            self._queue.append((frame.t, frame.full, getattr(st, "seq", None), run))
            self._cond.notify()

    def alarm_up(self, now: float) -> bool:
        """Whether the alarm's file is there (looked at again ALARM_POLL_S after the last look; one that can't be looked
        at counts as there); when it last went down kept (`_gone_at`; None while it's up or was never up)."""
        if self._polled is None or not 0.0 <= now - self._polled < ALARM_POLL_S:
            self._polled = now
            was = self._up
            try:
                os.stat(paths.ALARM if self.alarm is None else self.alarm)
                self._up = True
            except FileNotFoundError:
                self._up = False
            except OSError:
                self._up = True
            if self._up:
                self._gone_at = None
            elif was:  # (gone just now: AFTER_S of frames from here)
                self._gone_at = now
        return self._up

    # --- its thread -------------------------------------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="violetta", daemon=True)
        self._thread.start()

    def close(self, timeout: float = 10.0) -> None:
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)

    def line(self) -> str:
        g = self.game
        if g is None:
            return ""
        return (f"; the Violetta game followed for {time.monotonic() - g['started']:.0f} s: "
                f"round {len(g['answers']) + 1}")

    def run(self) -> None:
        try:
            while True:
                with self._cond:
                    if not self._queue and not self._closing:
                        self._cond.wait(WINDOW_CHECK_S)
                    if self._closing:
                        break
                    item = self._queue.popleft() if self._queue else None
                self.step(time.monotonic(), item)
        except Exception as e:  # noqa: BLE001 (the service goes on: only this is off)
            self.why_off = f"{type(e).__name__}: {e}"
            self._say(f"the Violetta game's following is off: {self.why_off}")
        if self.game is not None:
            self._end("stopped")

    def step(self, now: float, item) -> None:
        """One turn of its thread: the model loaded as the alarm goes up; the game begun, followed, ended; its rounds
        said at their answer windows; the pass's dialog clicked."""
        up = self.alarm_up(now)
        if up and self.detector is None and self._loading is None:
            self._loading = threading.Thread(target=self._load, name="violetta-load", daemon=True)
            self._loading.start()
        if item is None:
            if self.game is not None and now - self.game["window_at"] > WINDOW_GONE_S:
                self._end("window gone")
            return
        t, full, seq, run = item
        if full.shape[:2] != (768, 1366):  # (the window's and the stage's places are known at the game's own size)
            return
        self._look_for_ok(now, full)
        g = self.game
        if g is None:
            if not up or now - self._checked < WINDOW_CHECK_S:
                return
            self._checked = now
            if self._window_up(full):
                self._begin(now)
            return
        if now - g["looked"] >= WINDOW_ROUND_S:
            g["looked"] = now
            if self._window_up(full):
                g["window_at"] = now
        if now - g["window_at"] > WINDOW_GONE_S or now - g["started"] > GAME_MAX_S:
            self._end("window gone" if now - g["window_at"] > WINDOW_GONE_S else "too long")
            return
        if self.detector is None:  # (the model still loading: the frame goes uncounted)
            g["unseen"] += 1
            return
        from ..perception import violetta
        boxes = self.detector([full])[0]
        g["ts"].append(t)
        g["boxes"].append(boxes)
        lit = violetta.buttons_lit(full)
        opened = lit and not g["lit"]
        g["lit"] = lit  # (before the round's pointer looks at it)
        if opened:
            self._round(len(g["ts"]) - 1)

    def _load(self) -> None:
        try:
            from ..perception import violetta
            d = violetta.Detector()
            d.load()
            self.detector = d.detect
            self._say("the Violetta game's model is loaded (the Lie Detector's alarm is up)")
        except Exception as e:  # noqa: BLE001
            self.why_off = f"the model didn't load ({type(e).__name__}: {e})"
            self._say(f"the Violetta game's following is off: {self.why_off}")

    def _window_up(self, full) -> bool:
        if self.spotter is None:
            from ..perception.lie_detector import LieDetectorSpotter
            self.spotter = LieDetectorSpotter()
        return self.spotter.over_play(full) == "countdown"

    def _begin(self, now: float) -> None:
        self.game = {"started": now, "window_at": now, "looked": now, "ts": [], "boxes": [], "lit": False,
                     "windows": [], "answers": [], "slots": None, "start": None, "unseen": 0,
                     "name": time.strftime("%Y%m%d-%H%M%S")}
        self.games += 1
        telemetry.emit("violetta", event="begin", game=self.game["name"])
        self._say("the Violetta game's window seen: its rounds followed")

    def _round(self, k: int) -> None:
        """An answer window at frame k: the round since the last one solved, the made-up one's slot said."""
        from ..perception import violetta
        g = self.game
        if not g["windows"]:  # (the first: the slots and the start measured on the frames before it)
            g["slots"] = violetta.slots_of(g["boxes"][:k])
            g["start"] = violetta.start_slot(g["boxes"][:k], g["slots"])
            g["where"] = g["start"]
        prev = g["windows"][-1] if g["windows"] else 0
        g["windows"].append(k)
        perm, kind, info = violetta.solve_round(g["ts"][prev:k + 1], g["boxes"][prev:k + 1], g["slots"],
                                                window=g["ts"][k])
        n = len(g["answers"]) + 1
        if g["where"] is not None:
            g["where"] = perm[g["where"]]
        slot = None if g["where"] is None else g["where"] + 1
        g["answers"].append(slot)
        one, two = info["one"], info["two"]
        telemetry.emit("violetta", event="round", game=g["name"], round=n, slot=slot, kind=kind,
                       perm=[p + 1 for p in perm], start=None if g["start"] is None else g["start"] + 1,
                       slots=[round(s, 1) for s in g["slots"]], frames=k + 1 - prev,
                       one=[round(one[0], 1), [p + 1 for p in one[1]], one[2], one[3]],
                       two=None if two[1] is None else [round(two[0], 1), [p + 1 for p in two[1]],
                                                        [p + 1 for p in two[2]], two[3], two[4]])
        if slot is None:
            self._say(f"the Violetta game, round {n}: the made-up one's start wasn't seen (no answer)")
            return
        self._say(f"the Violetta game, round {n}: the made-up one is at slot {slot} ({kind} hop(s), "
                  f"{[p + 1 for p in perm]})")
        self._answer(g, slot)
        try:
            self.push(f"Violetta round {n}: slot {slot}", f"the made-up Violetta is at slot {slot} (round {n} of 4)")
        except Exception as e:  # noqa: BLE001 (the push is a convenience)
            self._say(f"the Violetta push failed: {type(e).__name__}: {e}")

    def _answer(self, g: dict, slot: int) -> None:
        """Click slot's answer button on a thread of its own, cancelled once the buttons go dark or the game ends."""
        from ..perception import violetta
        x, y = violetta.BUTTON_AT[slot - 1]
        active = lambda: self.game is g and g["lit"]  # noqa: E731

        def go():
            try:
                done = self.answer(x, y, active)
            except Exception as e:  # noqa: BLE001 (no player, a refusal: the answer's said all the same)
                done = False
                self._say(f"the click on slot {slot}'s button failed: {type(e).__name__}: {e}")
            telemetry.emit("violetta", event="answer", slot=slot, x=x, y=y, clicked=bool(done))

        self.answering = threading.Thread(target=go, name="violetta-answer", daemon=True)
        self.answering.start()

    def _click_answer(self, x: int, y: int, active) -> bool:
        from . import client
        return client.click_lie_answer(int(x), int(y), active)

    def _end(self, why: str) -> None:
        g, self.game = self.game, None
        telemetry.emit("violetta", event="end", game=g["name"], why=why, answers=g["answers"], frames=len(g["ts"]),
                       unseen=g["unseen"], dropped=self.dropped)
        self._say(f"the Violetta game over ({why}): answers {g['answers']}, {len(g['ts'])} frames")

    def _look_for_ok(self, now: float, full) -> None:
        """The pass's dialog: its OK clicked once seen OK_SEEN looks in a row, OK_TRIES times at most."""
        if now - self._ok_checked < OK_CHECK_S:
            return
        self._ok_checked = now
        if self.ok is None:
            from ..perception import violetta
            self.ok = violetta.OkDialog()
        at = self.ok(full)
        if at is None:
            self._ok_seen = 0
            return
        self._ok_seen += 1
        self._ok_clicks = [c for c in self._ok_clicks if now - c < 60]
        if self._ok_seen < OK_SEEN or len(self._ok_clicks) >= OK_TRIES or \
                (self._ok_clicks and now - self._ok_clicks[-1] < OK_RETRY_S):
            return
        self._ok_clicks.append(now)
        x, y = at
        try:
            done = self.clicker(x, y)
        except Exception as e:  # noqa: BLE001 (no player, the alarm's refusal: said, tried again while it stays)
            done = False
            self._say(f"the pass's OK click failed: {type(e).__name__}: {e}")
        telemetry.emit("violetta", event="ok", x=x, y=y, clicked=bool(done))
        self._say(f"the pass's dialog seen: its OK {'clicked' if done else 'not clicked'} at {x}, {y}")

    def _click(self, x: int, y: int) -> bool:
        from . import client
        return client.click_lie_ok(int(x), int(y))

    def _push(self, title: str, text: str) -> None:
        from . import notify
        notify.bark(notify.load_bark(), title, text, level="timeSensitive", log=self._say)

    def _say(self, msg: str) -> None:
        try:
            self.log(msg)
        except Exception:  # noqa: BLE001
            pass


class Together:
    """Several of the service's Lie Detector watchers as one (its `lie_tracking`): each started, fed, said, closed."""

    def __init__(self, *parts):
        self.parts = parts

    def start(self) -> None:
        for p in self.parts:
            p.start()

    def saw(self, st, frame, run) -> None:
        for p in self.parts:
            p.saw(st, frame, run)

    def line(self) -> str:
        return "".join(p.line() for p in self.parts)

    def close(self, timeout: float = 10.0) -> None:
        for p in self.parts:
            p.close(timeout)
