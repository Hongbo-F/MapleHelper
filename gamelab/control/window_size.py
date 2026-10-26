"""Setting the game's client right again as the user does it: Alt+Enter, the game goes full screen; Alt+Enter
again, it's windowed at its own size, drawn where its window is (from 2026-10-24; on master's perception service
from 2026-10-26).

The eyes are for a 1366x768 client (perception.framecheck): a look at another size, or at a picture the game drew off
its client area (kind "drawn"), has nothing to act on, and a hunt stands still until it's set right. On 2026-10-24 at
about 19:55 the client was another size, from a cause nobody knows (once before, Chrome Remote Desktop had resized the
desktop and the client was 1396x1952), and the user asked for the way they set it back by hand: "use alt + enter to
enter full screen mode, then use alt + enter again to return to window-ed mode". On 2026-10-25 at 23:21 the game was
full screen at 3840x2160 for a minute; at 23:40 the client was 1366x768 but the game drew its picture 18 px left and 85
up (at the window's corner, framecheck.drawn_offset), and Alt+Enter twice set that right too. A normal program can't
press keys in the game (it runs as administrator): the keys go through the player, which plays them only with the
game in front and checks that again before every key. The perception service sends no keys: a command that acts
makes the try (Senses.window_fix: SizeFix), and the pet's babysitter a bounded one (behavior.pet.PetWait).

`refit` is one try, which `window --recover` makes by hand: Alt+Enter, a wait for the window to change and settle
(full screen: its client the whole monitor, window.full_screen), no second before FULL_HOLD_S after the first (at 23:41
a second 3.5 s after the first wasn't taken; one about 8 s after it was), Alt+Enter again (from full screen, up to
BACK_PRESSES of them while one changes nothing), the same wait, and the client's size read (window.client_rect, in the
monitor's real px: gamelab's commands are DPI-aware). A game full screen already gets only those from full screen; the
second goes out only once the first has put the game full screen (one that left it windowed, or changed nothing, would
be put full screen by a second and left so). `SizeFix` is what a command's Senses hands every look with nothing to act
on for a fault Alt+Enter mends (framecheck.FIX_KINDS): it tries once the same fault has stood STEADY_S, once more
AGAIN_S later, and then waits for the user, as before; HOURLY tries an hour at most, by any process. A hunt standing
still (Senses.regain) makes its tries there; any other command's looks wait for its first try (`pending`).

No Alt+Enter goes out while the Lie Detector alarm is up (said, and the command goes on as the alarm has it: a hunt's
next plan or look raises it and she retreats, the pet goes on feeding) or the Lie Detector is on the game's screen:
it's looked for before each Alt+Enter on the perception service's picture of the client at whatever size
(`lie_detector_sighting`), and seen there the alarm is raised as the player's watchdog raises it (control.alarm: the
alarm file, its screenshot, the page), for every command to refuse and a hunt to retreat. That look is the guard at
another size: the player's watchdog saw the Lie Detector only at 1366x768 and 1920x1080 (the saved screens stretched to
3840x2160 scored 0.67-0.74 of the threshold there: D:/MapleHelper-scratch/winfix/review-safety/watchdog_sizes.out)
until a player started since 2026-10-26 brings a client of any other size to 1920x1080 first, as the service's picture
is (perception.lie_detector.to_template). No picture to look at, no Alt+Enter: none to begin with, and after one put the
game full screen the look is asked again for LOOK_FOR_S, then the try stops with the game left full screen (a game left
so is set right by the next try or the user; an Enter typed into the Lie Detector isn't undone). Nor while a plan is
waiting for the player or being played (client.busy; before a press back from full screen it's waited for, BUSY_WAIT_S
at most); while another process is trying (the named mutex LOCK: let go of when a process ends, even killed); with the
game minimized, at the try's start or in it (the user switched away from the game full screen: it isn't brought back
over what they're doing); nor unless the game is in front once brought forward (Alt+Enter in another window does
something else there). Alt is her jump key too (config.toml): she jumps where she stands at each one.

A try is counted (`on_try`: SizeFix's TRIES, WINDOW_TRIES, the pet's TRY_EVERY_S) as its first Alt+Enter goes to the
player, so a try cut short (the process killed in the hold: the babysitter stops the pet as a hunt starts) is one too;
one that sent none isn't (`Refit.counted`).

The waits are guesses but for FULL_HOLD_S until the live check (README: Capture checks) measures them: `window
--recover` says how long the window took to settle after each Alt+Enter.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .. import telemetry, window
from ..paths import WINDOW_TRIES
from ..perception.framecheck import FIX_KINDS, HUD_SEEN, Problem
from ..screen import GAME_VIEW
from . import client

ALT_LEAD_MS = 50  # guess: Alt is held this long before Enter goes down, and after it comes up
ENTER_MS = 40     # Enter's tap (actions.TAP_MS)
# The plan, as the explore steps `alt+ 50 enter 50 alt-` would make it (the user's manual fix through `act` at 23:41 on
# 2026-10-25): Alt pressed, Enter tapped, Alt let go. Each Alt+Enter is a plan of its own: in one plan with a second
# 3.5 s on, the second wasn't taken.
ALT_ENTER = ((0, "alt", "down"), (ALT_LEAD_MS, "enter", "down"), (ALT_LEAD_MS + ENTER_MS, "enter", "up"),
             (2 * ALT_LEAD_MS + ENTER_MS, "alt", "up"))
PLAN_TIMEOUT_S = 10.0  # the player's answer to it (140 ms of keys; the player waits 0.5 s for the game to come forward)
FRONT_WAIT_S = 3.0     # s the game is waited for to come to the front before an Alt+Enter (as __main__.FRONT_WAIT_S)
POLL_S = 0.1           # how often the window is read after an Alt+Enter
# Guess: the window must change within this of an Alt+Enter, or the key did nothing (at 23:41 one wasn't taken: the
# window stayed full screen). A monitor takes a second or three to change its mode; the game's window changes as soon
# as the game asks for it.
CHANGE_S = 8.0
# Guess: ... then stay as it is this long before the next Alt+Enter or before its size counts (a game resetting its
# drawing for the new mode may still move its window) ...
SETTLE_S = 1.5
SETTLE_BY_S = 20.0     # ... and settle within this of the key, or the try stops there (a window that keeps changing)
# Measured (2026-10-25 23:41, `act alt+ 50 enter 50 alt- 3500 alt+ 50 enter 50 alt-` through the player): the first
# Alt+Enter put the game full screen (3840x2160 at 0,0); the second, 3.5 s after it, wasn't taken; one more about 8 s
# after the first set it back to 1366x768 at (2454, 107), drawn right. So from full screen no Alt+Enter goes out sooner
# than this after the player answered the one that put it there (its keys played by then; counted from before that
# plan was made ready, a first that waited 2.5 s for the game to come to the front had the way back 5.5 s after it),
# whenever its window looked settled ...
FULL_HOLD_S = 8.0
# ... and one that changes nothing within CHANGE_S is pressed again, this many from full screen at most in a try (a
# game left full screen is worse than another key: the next try, AGAIN_S on, starts from full screen).
BACK_PRESSES = 3
# Guess: after an Alt+Enter, a look for the Lie Detector with no picture (the duplication made afresh after the switch,
# the service's frame loop reading the new size) is asked again every LOOK_EVERY_S for LOOK_FOR_S at most; then no more
# Alt+Enter in the try.
LOOK_EVERY_S = 0.5
LOOK_FOR_S = CHANGE_S
BUSY_WAIT_S = PLAN_TIMEOUT_S  # guess: a plan waiting or played (another process's) waited for before a press back
LOCK = "Local\\gamelab-window-fix"  # the named mutex a try holds (one process's try at a time)
UNCHANGED = "the window didn't change"  # (why an Alt+Enter wasn't taken, as `_settle` says it)

# SizeFix: when a look has nothing to act on for a fault Alt+Enter mends.
STEADY_S = 5.0     # guess: the same fault, the window where it was, stood this long before a try (not in the middle of
                   # the user's own drag)
# Guess: a command that stops on looks failing waits for the first try this much longer than STEADY_S at most
# (`pending`): a plan, or another process's try, holding it off (a chain's plan lasts a few seconds).
HOLD_S = 10.0
TRIES = 2          # tries each time it goes wrong, AGAIN_S apart; then the user is waited for, as before
AGAIN_S = 30.0     # guess
HOURLY = 4         # tries in any hour at most, by any process (WINDOW_TRIES): never a loop of full-screen switches
HOUR_S = 3600.0
UNPRESSED = 4      # guess: tries that brought the game forward but sent no Alt+Enter (no picture), each time at most


def size_of(area: dict | None) -> tuple[int, int] | None:
    return None if area is None else (area["width"], area["height"])


def _text(size: tuple[int, int] | None) -> str:
    return "no window" if size is None else f"{size[0]}x{size[1]}"


def _say(log: Callable[[str], None], text: str) -> None:
    """A line in the command's log, and the telemetry's `window` event."""
    log(f"  {text}")
    telemetry.emit("event", kind="window", text=text)


@lru_cache(maxsize=1)
def _spotter():
    from ..perception.lie_detector import LieDetectorSpotter
    return LieDetectorSpotter()


@dataclass
class Sighting:
    """The Lie Detector seen on a picture of the client: in which form (perception.lie_detector's name), and the
    picture (for the alarm's screenshot)."""
    form: str
    frame: np.ndarray


def lie_detector_sighting(frame: np.ndarray) -> Sighting | None:
    """The Lie Detector on a picture of the game's client (RGB), or None: the perception service's at whatever size the
    client is (perception.frames.glimpse: 1366x768 as it is, any other size brought to 1920x1080, the spotter's
    templates' size). The six saved screens and those of 2026-10-24 and 25 as the game's own view stretched to a client
    of 3840x2160, 2560x1440, 1600x900, 1396x1952, 1350x700 or 1280x720 and brought so to 1920x1080: every form is seen
    at every size (1.08-1.43 of the threshold) but the countdown, seen at 3840x2160, 1600x900 and 1396x1952 (1.01-1.02)
    and not at 2560x1440, 1350x700 or 1280x720 (0.76-0.86: its thin yellow letters resampled); brought to 1366x768
    instead, the countdown was seen at none (0.70-0.71): D:/MapleHelper-scratch/winfix/build/ld_sizes2.py,
    ld_sizes.py."""
    scores = _spotter().scores(frame)
    form = max(scores, key=scores.get)
    return Sighting(form, frame) if scores[form] >= 1 else None


def lie_detector_seen(frame: np.ndarray) -> bool:
    """Whether the Lie Detector is on a picture of the game's client (lie_detector_sighting)."""
    return lie_detector_sighting(frame) is not None


def _alarm(seen, log: Callable[[str], None]) -> str:
    """The Lie Detector seen by a try's look (`seen`: a Sighting, or True from a look that gives no picture): the alarm
    raised as the player's watchdog raises it (control.alarm.raise_alarm: its file, the picture as its screenshot, the
    page), unless it's up already; what to say of it."""
    from .alarm import raise_alarm
    form = getattr(seen, "form", None)
    what = "the Lie Detector is on screen" + (f" ({form})" if form else "")
    if client.alarm_up():
        return what
    try:
        raise_alarm(getattr(seen, "frame", None), form=form)
    except Exception as e:  # (the disk full: paged all the same, by raise_alarm; or the page failed, the file there)
        if not client.alarm_up():
            _say(log, f"{what}: the alarm couldn't be written ({type(e).__name__}: {e})")
            return what
    _say(log, f"{what}: the Lie Detector alarm raised (the player's watchdog may not see it at this size)")
    return what + ": the alarm raised"


@dataclass
class Refit:
    """What one try came to."""
    before: tuple[int, int] | None          # the client's size before it ...
    after: tuple[int, int] | None = None    # ... and after (None: no window)
    full: bool = False                      # full screen after it
    ok: bool = False                        # the size wanted, windowed
    pressed: int = 0                        # Alt+Enters the player played
    took: list[float] = field(default_factory=list)  # s from each to the window's last change before it settled
    why: str = ""                           # why it stopped short
    held: bool = False                      # not tried, nothing done to the game: minimized, the alarm up, a plan or
                                            # another process's try going on
    counted: bool = False                   # an Alt+Enter went to the player: one of the tries (`on_try` as it went)
    pressed_at: float | None = None         # when the player answered the last Alt+Enter it played (time.monotonic)
    settled: float | None = None            # when the window last settled (time.monotonic): no frame before it counts

    def text(self, want: tuple[int, int]) -> str:
        times = ", ".join(f"{t:.1f}" for t in self.took)
        how = f" (Alt+Enter {self.pressed} time{'s' if self.pressed != 1 else ''}" + (
            f"; the window settled {times} s after" if times else "") + ")"
        if self.ok:
            return f"the client is {_text(want)} again" + how
        now = ("the game window is gone" if self.after is None else
               f"the client is {_text(self.after)}{' (full screen)' if self.full else ''}, not {_text(want)}")
        return now + how + (f": {self.why}" if self.why else "")


def refit(title: str, want: tuple[int, int] = GAME_VIEW, log: Callable[[str], None] = print,
          spot: Callable[[], object] | None = None, on_try: Callable[[], None] | None = None) -> Refit:
    """One try at setting the game's client to `want` with Alt+Enter twice (from full screen, only the way back), as
    the module's docstring says; what came of it is said and returned. Not tried with the game minimized, the Lie
    Detector alarm up or something going on (`held`: a plan, another process's try), which the caller says (SizeFix,
    `window --recover`). `spot`: the Lie Detector on screen (a Sighting, or anything true; falsy: not seen; it raises
    if the screen can't be looked at: then no Alt+Enter; None, never looked at: none either). `on_try`: called as the
    first Alt+Enter goes to the player (`counted`). Raises RuntimeError with no game window."""
    hwnd = window.find_game(title)
    area = window.client_rect(hwnd)
    out = Refit(before=size_of(area))
    if window.minimized(hwnd):
        out.why, out.held = "the game is minimized", True
        return out
    if client.alarm_up():
        out.why, out.held = "the Lie Detector alarm is up", True
        return out
    if client.busy():
        out.why, out.held = "a plan is waiting for the player or being played", True
        return out
    lock = _lock()
    if lock is None:
        out.why, out.held = "another process is setting it right", True
        return out
    try:
        full = window.full_screen(hwnd, area)
        _say(log, f"the client is {_text(out.before)}, full screen: Alt+Enter to set it back" if full else
             f"the client is {_text(out.before)}" + (f", not {_text(want)}" if out.before != want else "")
             + ": Alt+Enter twice")
        try:
            _presses(title, area, full, want, out, spot, on_try, log)
        except RuntimeError as e:  # the game window gone meanwhile
            out.why = str(e)
    finally:
        _unlock(lock)
    try:
        hwnd = window.find_game(title)
        area = window.client_rect(hwnd)
        out.after, out.full = size_of(area), window.full_screen(hwnd, area)
    except RuntimeError:
        out.after, out.full = None, False
    out.ok = out.after == want and not out.full
    _say(log, out.text(want))
    return out


def _presses(title: str, area: dict, full: bool, want: tuple[int, int], out: Refit,
             spot: Callable[[], object] | None, on_try: Callable[[], None] | None, log: Callable[[str], None]) -> None:
    """The Alt+Enters of one try (see refit): to full screen if it's windowed, held there FULL_HOLD_S after the player
    played that one, then back, pressed again while one changes nothing (BACK_PRESSES at most); stopped short
    (out.why) at what stands in the way, a window that doesn't change or settle, the size set right by the first, or a
    first that didn't put it full screen."""
    def press(area: dict) -> tuple[dict, bool]:
        return _press(title, area, out, spot, on_try, log)

    if not full:
        area, changed = press(area)
        if not changed:
            return
        full = window.full_screen(window.find_game(title), area)
        if size_of(area) == want and not full:
            return  # set right already (a second would put it full screen)
        if not full:  # (a game full screen at first that wasn't told so, say)
            out.why = "the first Alt+Enter didn't put it full screen (a second would)"
            return
        if (wait := out.pressed_at + FULL_HOLD_S - time.monotonic()) > 0:
            time.sleep(wait)  # (at 23:41 one 3.5 s after the first wasn't taken)
    for k in range(BACK_PRESSES):
        area, changed = press(area)
        if changed or not out.why.startswith(UNCHANGED):
            return
        hwnd = window.find_game(title)
        if not window.full_screen(hwnd, window.client_rect(hwnd)):
            out.why = ""
            return  # (changed after all, past CHANGE_S: no more)
        if k + 1 < BACK_PRESSES:
            out.why = ""  # not taken, as at 23:41: pressed again


def _press(title: str, area: dict, out: Refit, spot: Callable[[], object] | None, on_try: Callable[[], None] | None,
           log: Callable[[str], None]) -> tuple[dict, bool]:
    """One Alt+Enter once nothing stands in its way, and the window waited for to change and settle: (the client area
    then, whether it changed); out.why says what stopped it (out.held: before anything was done to the game)."""
    def stop(why: str) -> tuple[dict, bool]:
        out.why, out.held = why, not out.pressed and not fronted
        return area, False

    fronted = False
    if window.minimized(window.find_game(title)):  # (switched away from full screen, the game minimizes)
        return stop("the game was minimized meanwhile: it isn't brought back over what's in front" if out.pressed
                    else "the game is minimized")
    if client.alarm_up():
        return stop("the Lie Detector alarm is up" + (" (the game may be left full screen)" if out.pressed else ""))
    if out.pressed:  # (the way back: a plan of another process's, the pet's food say, waited for)
        until = time.monotonic() + BUSY_WAIT_S
        while client.busy() and time.monotonic() < until:
            time.sleep(POLL_S)
        if client.alarm_up():
            return stop("the Lie Detector alarm is up (the game may be left full screen)")
    if client.busy():
        return stop("a plan is waiting for the player or being played"
                    + (f" (waited for {BUSY_WAIT_S:g} s)" if out.pressed else ""))
    fronted = True
    if not window.bring_to_front(title, wait=FRONT_WAIT_S):
        return stop(f"the game didn't come to the front ({window.foreground_title()!r} is in front)")
    seen, why = _look(spot, again=out.pressed > 0)
    if why is not None:  # (no picture: the perception service down, a capture refused mid-switch)
        return stop(f"the screen couldn't be looked at for the Lie Detector ({why})"
                    + (f" in {LOOK_FOR_S:g} s: no Alt+Enter without that look" if out.pressed else ""))
    if seen:
        return stop(_alarm(seen, log))
    if not out.counted:
        out.counted = True  # (counted as it goes: a try cut short in its hold is one too)
        if on_try is not None:
            on_try()
    try:
        played = client.submit([list(e) for e in ALT_ENTER], focus=True, lead_ms=0, name="alt_enter",
                               timeout=PLAN_TIMEOUT_S)
    except client.LieDetectorAlarm:
        out.why = "the Lie Detector alarm is up" + (" (the game may be left full screen)" if out.pressed else "")
        return area, False
    except client.PlayerError as e:
        out.why = f"the player didn't play Alt+Enter: {e}"
        return area, False
    if not any(len(e) > 2 and e[1] == "enter" and e[2] == "down" for e in played.get("sent", [])):
        problems = "; ".join(played.get("problems", [])) or "nothing sent"
        out.why = f"the player didn't press Enter ({problems})"
        return area, False
    out.pressed += 1
    out.pressed_at = time.monotonic()  # (the player answers once the plan's keys are played)
    after, took, why = _settle(title, area)
    if took is None:
        problems = "; ".join(played.get("problems", []))
        out.why = why + (f" (the player: {problems})" if problems else "")
        return after if after is not None else area, False
    out.took.append(round(took, 2))
    out.settled = time.monotonic()
    return after, True


def _look(spot: Callable[[], object] | None, again: bool) -> tuple[object, str | None]:
    """The look for the Lie Detector before an Alt+Enter: (what `spot` saw, None), or (None, why there's no picture).
    With `again` (an Alt+Enter has gone: the duplication may be made afresh, the service reading the new size), asked
    again every LOOK_EVERY_S for LOOK_FOR_S at most."""
    if spot is None:
        return None, "nothing to look with"
    until = time.monotonic() + (LOOK_FOR_S if again else 0.0)
    while True:
        try:
            return spot(), None
        except Exception as e:  # (BadLook from the service; anything else from its pipe)
            why = getattr(e, "why", None) or str(e)
        if time.monotonic() + LOOK_EVERY_S > until:
            return None, why
        time.sleep(LOOK_EVERY_S)


def _area(title: str) -> dict | None:
    """The game's client area, or None while there's no window (a game may make its window afresh as it switches)."""
    try:
        return window.client_rect(window.find_game(title))
    except RuntimeError:
        return None


def _settle(title: str, before: dict | None) -> tuple[dict | None, float | None, str]:
    """After an Alt+Enter: the client area once it has changed from `before` and stayed as it is SETTLE_S, and the s
    from the key to its last change; (the area, None, why) if it didn't change within CHANGE_S or settle within
    SETTLE_BY_S."""
    start = time.monotonic()
    last, changed = before, None
    while True:
        time.sleep(POLL_S)
        now = time.monotonic()
        area = _area(title)
        if area != last:
            last, changed = area, now
        if changed is None and now - start >= CHANGE_S:
            return area, None, f"{UNCHANGED} within {CHANGE_S:g} s of Alt+Enter"
        if changed is not None and area is not None and now - changed >= SETTLE_S:
            return area, changed - start, ""
        if now - start >= SETTLE_BY_S:
            return area, None, f"the window didn't settle within {SETTLE_BY_S:g} s of Alt+Enter"


def _lock():
    """The named mutex LOCK, held for a try (a handle), or None while another process's try holds it. Windows lets go
    of it when a process ends, however it ends (the babysitter stops the pet's process when a hunt starts)."""
    from ..localization.service import only_one
    try:
        return only_one(LOCK)
    except OSError:
        return None


def trying() -> bool:
    """Whether a process holds LOCK now: a try going on (the pet waits meanwhile: __main__._pet_unfit)."""
    from ..localization.service import running
    return running(LOCK)


def _unlock(handle) -> None:
    import ctypes
    import sys
    if sys.platform == "win32" and handle not in (None, True):
        ctypes.WinDLL("kernel32").CloseHandle(handle)


def tries_within(path, now: float) -> list[float]:
    """The times (time.time) Alt+Enter was tried within the hour before `now`, by any process."""
    try:
        tries = json.loads(path.read_text(encoding="utf-8")).get("tries", [])
    except (OSError, ValueError, AttributeError):
        return []
    return [t for t in tries if isinstance(t, (int, float)) and now - HOUR_S < t <= now]


def book(path, now: float) -> None:
    """A try at `now`, kept with those of the hour before it."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"tries": tries_within(path, now) + [now]}), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass  # (the hour's count then misses it: TRIES still bounds this process's)


class SizeFix:
    """What a command's Senses hands each look with nothing to act on (Senses.window_fix): for a fault Alt+Enter mends
    (framecheck.FIX_KINDS: the client another size, or the game drawing off it), `refit` once the same fault, the
    window where it was, has stood STEADY_S, once more AGAIN_S later if the looks after didn't pass, and then nothing
    until a look has passed (`passed`: set right, by it or the user) and it goes wrong again; AGAIN_S between any two
    tries, and HOURLY tries an hour at most (by any process: WINDOW_TRIES). A try is counted as its first Alt+Enter
    goes out (refit's `on_try`). One that something held off before it did anything isn't, nor is a game minimized or
    gone tried; one that brought the game forward but sent no Alt+Enter (no picture to look for the Lie Detector on,
    the game not in front) isn't either, AGAIN_S to the next, UNPRESSED of them at most. Why a try wasn't made is said
    once each time it goes wrong (a look every 0.2 s asks). `spot`: refit's (the Lie Detector looked for on the
    perception service's picture). `in_play`: how well the HUD's buttons match on the client's picture
    (framecheck.hud_seen's score; it raises with no picture), looked at before each try: below HUD_SEEN the game isn't
    in play (its login or character screen, where Enter types into a field or picks), and no try is made, AGAIN_S to
    the next look; None: not looked at."""

    def __init__(self, title: str, want: tuple[int, int] = GAME_VIEW, log: Callable[[str], None] = print,
                 spot: Callable[[], object] | None = None, in_play: Callable[[], float] | None = None):
        self.title, self.want, self.log, self.spot, self.in_play = title, want, log, spot, in_play
        self.tries = 0                   # tries since it went wrong (an Alt+Enter sent)
        self.unpressed = 0               # ... and those that sent none, the game brought forward
        self.gave_up: str | None = None  # why it waits for the user (said once)
        self.settled: float | None = None  # when the window last settled after a try (time.monotonic)
        self._since: tuple[tuple, float] | None = None  # the fault (its kind, the client area), since when it's been so
        self._wrong: float | None = None  # since when it has been wrong (any fault)
        self._last = -1e18               # when the last try ended, or the last look found the game not in play
        self._said: set[str] = set()     # why tries weren't made, said since it went wrong

    def passed(self) -> None:
        """A look passed (the client is right again): its going wrong again starts afresh (AGAIN_S still between two
        tries)."""
        self._since, self._wrong, self.tries, self.unpressed = None, None, 0, 0
        self.gave_up, self._said = None, set()

    def pending(self) -> bool:
        """Whether its first try at the fault is still to come: none made since it went wrong, not given up, and wrong
        less than STEADY_S + HOLD_S (a plan, or another process's try, holding it off is waited for that long at most).
        A command that stops on looks failing waits for it (Senses._failed)."""
        return (self.gave_up is None and self.tries == 0 and self._wrong is not None
                and time.monotonic() - self._wrong < STEADY_S + HOLD_S)

    def __call__(self, problem: Problem, wait: bool = True) -> bool:
        """After a look with nothing to act on for `problem` (only FIX_KINDS are anything to it): whether a try set the
        client right now. `wait` False: tried without the STEADY_S wait (a command that finds it so as it starts)."""
        if problem.kind not in FIX_KINDS or self.gave_up is not None:
            return False
        try:
            hwnd = window.find_game(self.title)
            if window.minimized(hwnd):
                return self._not_now("the game is minimized")
            area = window.client_rect(hwnd)
        except RuntimeError as e:  # no game window
            return self._not_now(str(e))
        if problem.kind == "size" and size_of(area) == self.want and not window.full_screen(hwnd, area):
            return False  # (set right meanwhile: the next look tells)
        now = time.monotonic()
        self._wrong = now if self._wrong is None else self._wrong
        fault = (problem.kind, tuple(sorted(area.items())))
        if self._since is None or self._since[0] != fault:
            self._since = (fault, now)
        if (wait and now - self._since[1] < STEADY_S) or now - self._last < AGAIN_S:
            return False
        if self.tries >= TRIES:  # (the looks after its tries didn't pass: the game drawing off its client still)
            return self._give_up(f"Alt+Enter didn't set it right in {TRIES} tries")
        if self.unpressed >= UNPRESSED:
            return self._give_up(f"no Alt+Enter could be sent in {UNPRESSED} tries")
        if client.alarm_up():
            return self._not_now("the Lie Detector alarm is up")
        recent = tries_within(WINDOW_TRIES, time.time())
        if len(recent) >= HOURLY:
            return self._give_up(f"Alt+Enter was tried {len(recent)} times in the last hour ({HOURLY} at most)")
        if self.in_play is not None and (why := self._not_in_play()) is not None:
            self._last = now  # (looked at again AGAIN_S on: not a picture down the pipe at every look)
            return self._not_now(why)
        try:
            out = refit(self.title, self.want, self.log, self.spot, on_try=self._tried)
        except RuntimeError as e:  # the game window gone meanwhile
            return self._not_now(str(e))
        if out.held:
            return self._not_now(out.why)
        self._last = time.monotonic()
        if not out.counted:  # (brought forward, no Alt+Enter sent: not one of the tries)
            self.unpressed += 1
            return self._not_now(out.why)
        if out.ok:
            self.settled = out.settled if out.settled is not None else self._last
            return True
        if self.tries >= TRIES:
            self._give_up(f"Alt+Enter didn't set it right in {TRIES} tries")
        return False

    def _tried(self) -> None:
        """refit's `on_try`, as a try's first Alt+Enter goes to the player: counted, here and in the hour's."""
        self.tries += 1
        book(WINDOW_TRIES, time.time())

    def _not_in_play(self) -> str | None:
        """Why the game isn't seen in play on the client's picture (`in_play`), or None if it is."""
        try:
            score = self.in_play()
        except RuntimeError as e:  # (BadLook: no picture from the perception service)
            return f"no picture of the client to look for the game in play on ({getattr(e, 'why', e)})"
        if score >= HUD_SEEN:
            return None
        return (f"the game isn't seen in play (its HUD's buttons match {score:.2f}, {HUD_SEEN} needed: its login or "
                f"character screen?)")

    def _not_now(self, why: str) -> bool:
        """No try now, for `why`: said once while it stays wrong."""
        if why not in self._said:
            self._said.add(why)
            _say(self.log, f"no Alt+Enter: {why}")
        return False

    def _give_up(self, why: str) -> bool:
        self.gave_up = f"{why}: waiting for the user to set the client back to {_text(self.want)}"
        _say(self.log, self.gave_up)
        return False
