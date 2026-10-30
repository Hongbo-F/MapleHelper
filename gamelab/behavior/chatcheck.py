"""Her chat box collapsed at a hunt's start if it's expanded (the user, 2026-10-30 ~12:45: "At the start of grind/hunt,
you should collapse the chat window if it's open."). Expanded, the box covers the play area's bottom left (x 287-850, y
633-700 at 1366x768: four lines of chat over the bottom platforms), hiding the monsters there and her plate when she's
under it, where her place by the scene does worse than her dot alone (localization.locate); hunts started with it so on
2026-10-29 at 09:26 and 15:58 and on 10-30 at 08:13 (lab/shots/statwindow_*.png: her stat window's frames at their
starts), and on 10-30 the user collapsed it by hand 34 s into the grind's recording of 08:17:50.

**What tells it** (perception.chatbox: the box's +/- button and its input row, at 1366x768): every frame the
perception service reads (Observation.chat, WorldState.chat), as each of the hunt's looks has it (Scene.chat). None
(the simulator's eyes): nothing is done, nothing said; None from a perception service (a service started from the code
before this): said once ("restart it"), nothing sent.

**When** (`ChatCheck.run`): at a hunt's start (`hunt`, `tour`, `probe run`: gamelab.__main__'s _hunter,
Hunter.collapse_chat, before her stat window's read: the box hides monsters its quiet moment looks for; `loop`: whether
a hunt loop follows, `tour --loop`), once a run:
1. looks until one tells, QUIET_S at most. Collapsed: said, nothing more. At another client size (or no frame): said at
   once, nothing sent. Not told on a look of the client's size (its button hidden: the mouse pointer over it, the UI
   greyed while a map loads, a window over it) or its input line marked (ChatBox.typing: text or a caret, it may have
   the keyboard and the key would be typed into it; the pointer over the field reads so too): the mouse moved off the
   box once a run (MOUSE_AT, as the key check's and her stat window's; no click), the looks from MOUSE_SETTLE_S after
   it read again. Still marked: said, nothing sent; no Esc (with nothing to close, Esc opens the game's MENU) and no
   Enter (it sends what's typed). Not told on any look within QUIET_S: put off (below), never pressed (her chat key
   toggles the box: pressed with it collapsed it would expand it).
2. expanded: her chat key at a quiet look (keycheck.unquiet: placed on a platform, not on a rope nor at its top,
   untouched for KNOCK_S, no monster within PLACE_CLEAR of her) after the box was seen expanded with its field
   unmarked on CLEAR_LOOKS looks in a row or more, over CLEAR_S (a caret blinking off on one look shows on another; in
   the hunt loop every look and glance of the hunt's counts: `watch`); a monster within PLACE_CLEAR puts it off at
   once (nothing moves her for it, and standing there it would come to her); no such look in QUIET_S, put off. Put off
   (or not told, or no look at all: LookLost) in the hunt loop (`tour --loop`, `--grind`), it's raised again (ChatDue,
   from Hunter.health_check) at the first quiet look DEFER_GAP_S or more later that tells the box, that look its
   first, DEFERS_MAX times at the most, then left so this run; elsewhere (`hunt kill`, a plain `tour`, `probe run`:
   `loop` False) left so at once, said. Every look asks the safety stop (her HP watch's due), as her stat window's do.
3. her chat key (`chat_key`: CHAT+'s in her remembered key bindings, else ', CHAT+'s in her key bindings of
   2026-10-28; never one of her own keys, Esc, the backslash or maplehelper's hotkeys) tapped once, through the player
   as every key of hers (control.player: every key config.toml can name, ' among them; with the Lie Detector alarm up
   it refuses the plan, client.LieDetectorAlarm, and the alarm is asked before the tap too). Then looks, SETTLE_S after
   the key and on, until one tells, VERIFY_S at most: collapsed, "the chat box expanded: collapsed (')"; its field
   marked more than before the key (the key typed into it: a lone apostrophe is 1 x 3 px), left so; still expanded on a
   look STILL_S after the key (a game slow to act on it shows it expanded a while: never a second key on a frame it
   may not have redrawn yet), the key again as in 2 (PRESSES_MAX, 2, in all), then "the chat box still expanded after
   ' twice: left so"; not told within VERIFY_S, left so (never pressed on a look that can't tell).
Only her chat key and a mouse move go out: nothing that moves or attacks her, no click. A telemetry event 'chat_box' at
the end (and at each put-off)."""

import time
from collections.abc import Callable
from dataclasses import dataclass

from .. import telemetry
from ..control import client
from ..perception.chatbox import COLLAPSED, EXPANDED, UNKNOWN
from ..perception.framecheck import LookLost
from .keycheck import MOUSE_AT, QUIET_LOOK_S, QUIET_S, near_her, unquiet
from .statcheck import bound_key, session_of

FUNCTION = "CHAT+"     # the key bindings' name of what toggles the box (perception.keybindings.FUNCTIONS[16])
KEY = "'"              # ... its key in her key bindings of 2026-10-28 (both characters' remembered bindings since)
PRESSES_MAX = 2        # her chat key at most this many times a run: once, and once more if it's still expanded
SETTLE_S = 0.3         # s after her chat key before a look tells what it did (the box redrawn the next frame: 515) ...
STILL_S = 1.0          # ... before one still expanded says it didn't take (the review of 2026-10-30: a game slow over
                       # 0.4 s to act on it had shown it expanded on both looks that judged it, and a second key
                       # would have expanded it again) ...
VERIFY_S = 2.0         # ... and this long at most for a look that tells
CLEAR_S = 0.6          # its field unmarked on the looks over this long before the key (a caret blinking: off a while,
CLEAR_LOOKS = 3        # ... about half a second) and on this many in a row at least (none marked between)
MOUSE_SETTLE_S = 0.2   # s after the mouse moved off the box before a look counts (a frame with the pointer gone)
MOUSE_MOVES_MAX = 1    # the mouse moved off the box at most this many times a run
DEFER_GAP_S = 2.0      # put off (not quiet): again at a quiet look in the hunt loop this long after at the soonest
DEFERS_MAX = 10        # ... this many times, then left so this run


class ChatDue(Exception):
    """Raised by Hunter.health_check in the hunt loop at a quiet look when the chat box's collapse put off is due again:
    the hunt loop collapses it (Hunter.collapse_chat, that look its first: `look`) and goes on. Not a HuntStop: the hunt
    doesn't end."""

    def __init__(self, look=None):
        super().__init__("the chat box's collapse due")
        self.look = look


@dataclass
class Outcome:
    """What a check did: `what` "collapsed" (her chat key, and the box seen collapsed), "was collapsed" (nothing to do),
    "left" (expanded, left so: `text` says why), "not told" (no look told it: nothing sent), "not read" (no state of it:
    the simulator's eyes, a perception service of the code before), "put off" (not quiet, not told or no look: in the
    hunt loop tried again later), "skipped" (done this run already); what was sent (key names, "mouse") and how long it
    took."""
    what: str
    text: str = ""
    sent: tuple = ()
    seconds: float = 0.0


def chat_key(keys, character: str | None = None) -> tuple[str | None, str]:
    """Her chat key and where it's from, or (None, why not): CHAT+'s in her remembered key bindings, else KEY
    (statcheck.bound_key: refused where it's one of her own keys, Esc, the backslash or maplehelper's hotkeys, or her
    remembered bindings have CHAT+ on no key)."""
    return bound_key(FUNCTION, KEY, "', CHAT+'s in her key bindings of 2026-10-28 (none remembered since)", keys,
                     character)


class ChatCheck:
    """Her chat box collapsed for hunter `h` if it's expanded (see the module's doc): its looks, actions and log.
    `presses`: her chat key's taps this run; `moves`: the mouse's moves off the box; `done`: nothing more this run;
    `loop`: whether a hunt loop tries a put-off again (None: as the hunter's `doubts_raise` says, or a start's)."""

    def __init__(self, h, log: Callable[[str], None] | None = None):
        self.h = h
        self.log = log or h.log
        self.presses = 0
        self.moves = 0
        self.done = False
        self.pending = False      # put off (not quiet, not told), tried again in the hunt loop (`again`)
        self.pending_at = 0.0
        self.defers = 0
        self.loop: bool | None = None
        self.sent: list[str] = []
        self.key: str | None = None
        self.last: Outcome | None = None
        self._began = 0.0
        self._counts_from = -1e18  # looks of frames shown before this don't count (the mouse just moved off the box)
        self._clear: tuple[float, float, int] | None = None  # the looks in a row showing it expanded, unmarked: the
        # first's and the last's times, how many

    def again(self, t: float, s=None) -> bool:
        """Whether a collapse put off is due again at `t` (a quiet look: Hunter.health_check), DEFER_GAP_S after the
        last try at the soonest, at a look `s` that tells the box (one that doesn't: the mouse still to move off it)."""
        if not (self.pending and not self.done and t - self.pending_at >= DEFER_GAP_S):
            return False
        told = getattr(s, "chat", None) if s is not None else None
        return s is None or told is not None and (told.state != UNKNOWN
                                                  or told.readable and self.moves < MOUSE_MOVES_MAX)

    def watch(self, s) -> None:
        """A look's or a glance's state of the hunt's while a collapse is put off (Hunter._chat_seen): the box expanded
        with its field unmarked on it, the stretch of such looks goes on (`_clear_seen`: enough lets the key go at once
        at ChatDue's look); else it's over."""
        told, t = getattr(s, "chat", None), getattr(s, "t", None)
        if told is not None and t is not None and told.state == EXPANDED and not told.typing and t >= self._counts_from:
            self._clear_seen(t)
        else:
            self._clear = None

    def _clear_seen(self, t: float) -> bool:
        """A look at `t` showing the box expanded, its field unmarked: whether such looks have gone on in a row long
        enough for the key (CLEAR_LOOKS of them at least, over CLEAR_S). The same frame twice (a look and a glance of
        it) counts once; one older than the stretch's first starts it again."""
        if self._clear is None or t < self._clear[0]:
            self._clear = (t, t, 1)
        elif t > self._clear[1]:
            self._clear = (self._clear[0], t, self._clear[2] + 1)
        first, last, n = self._clear
        return n >= CLEAR_LOOKS and last - first >= CLEAR_S

    def run(self, at_start: bool = False, look=None, loop: bool | None = None) -> Outcome:
        """The check (the module's doc), unless it's done this run. `at_start`: a hunt's start; `loop`: whether a hunt
        loop follows (put off, it's tried again there; kept for the run); `look`: its first look (ChatDue's, the quiet
        one it was raised at: a look taken again a moment later had a monster come within PLACE_CLEAR, in the
        simulator's crowd, and it was put off again). Raises SafetyStop (the HP watch, at every look) and
        client.LieDetectorAlarm (the alarm up: at a look, before the key or the mouse, or the player refusing them)."""
        if loop is not None:
            self.loop = loop
        if self.done:
            return Outcome("skipped", "the chat box: done this run")
        self._began = time.monotonic()
        self.pending = False
        end = self._began + QUIET_S
        why, expanded = None, False
        while True:
            if look is not None:
                s, look = look, None
            else:
                try:
                    s = self._look()
                except LookLost as lost:  # (no look for LOST_S: before the hunt loop's window fix, or in it)
                    return self._defer(None, at_start, f"not told (no look: {lost.why})")
            told = getattr(s, "chat", None)
            if told is None:  # (eyes that don't read it: the simulator's, a service of the code before)
                return self._not_read()
            t = getattr(s, "t", None)
            t = time.monotonic() if t is None else t
            if t < self._counts_from:
                pass  # (a frame from before the mouse moved off the box)
            elif told.state == COLLAPSED:
                self._clear = None
                if self.presses:  # (a key of an earlier try's, its look lost)
                    return self._end("collapsed", f"the chat box expanded: collapsed ({self.key})")
                return self._end("was collapsed", "the chat box: collapsed")
            elif not told.readable:  # (another client size, no frame: nothing over the box hides it)
                return self._end("not told", f"the chat box {told.words()}: {self._unsent()}")
            elif told.state == UNKNOWN or told.typing:
                self._clear = None
                if self._mouse_off(told):
                    continue
                if told.typing:
                    return self._end("left", "the chat box expanded, its input line with text or a caret (it may "
                                             "have the keyboard: a key would be typed into it): left so, "
                                             + self._unsent())
            else:  # expanded, its field unmarked
                expanded = True
                if self.presses >= PRESSES_MAX:
                    return self._end("left", f"the chat box still expanded after {self._key_words()} twice: left so")
                clear = self._clear_seen(t)
                why = unquiet(self.h, s)
                if why is None and clear:
                    if (out := self._press(told, at_start)) is not None:
                        return out
                    end, why = time.monotonic() + QUIET_S, None  # (still expanded after it: again at a quiet look)
                    continue
                if why is not None and near_her(s, getattr(self.h, "model", None)):
                    return self._defer(why, at_start)
            if time.monotonic() >= end:
                if expanded:
                    return self._defer(why or f"its input line not seen unmarked for {CLEAR_S:g} s", at_start)
                return self._defer(None, at_start, told.words())
            time.sleep(QUIET_LOOK_S)

    def _key_words(self) -> str:
        return self.key or KEY

    def _unsent(self) -> str:
        """Her chat key not sent, in words: "nothing sent", or with the mouse moved off the box, that it wasn't."""
        return "nothing sent" if not self.sent else f"{self._key_words()} not sent"

    def _look(self):
        """A look of the check's: the safety stop asked of it as Hunter.health_check would (her HP watch's due)."""
        s = self.h.look()
        hp = getattr(self.h, "hp_watch", None)
        if hp is not None and (stop := hp.due()) is not None:
            raise stop
        return s

    def _loops(self, at_start: bool) -> bool:
        """Whether a put-off is tried again in a hunt loop: `loop` as the start gave it, else the hunter's doubts_raise
        (the hunt loop's) or a start's."""
        return self.loop if self.loop is not None else bool(getattr(self.h, "doubts_raise", False)) or at_start

    def _mouse_off(self, told) -> bool:
        """The mouse moved off the box (MOUSE_AT), once a run (MOUSE_MOVES_MAX), at a look of the client's size that
        doesn't tell it or shows its input line marked: the pointer over its button hides the glyph, over its field it
        marks it (the review of 2026-10-30: 215 of 216 places over the field read typing). Whether it moved (the looks
        from MOUSE_SETTLE_S after it count)."""
        if self.moves >= MOUSE_MOVES_MAX:
            return False
        client.check_alarm()  # (the player refuses a mouse move with the alarm up too)
        self.h.actions.play([[0, "mouse", *MOUSE_AT]], "chat_box_mouse")
        self.moves += 1
        self.sent.append("mouse")
        self._counts_from = time.monotonic() + MOUSE_SETTLE_S
        self.log(f"  the chat box {told.words()}: the mouse moved off it (the pointer may be what's over it), "
                 "looked at again")
        return True

    def _press(self, before, at_start: bool) -> Outcome | None:
        """Her chat key tapped (the box as `before` showed it), then looks SETTLE_S after it and on until one tells
        (VERIFY_S at most): the outcome, or None if it's still expanded STILL_S after it (the key again at a quiet look:
        `run`)."""
        if self.key is None:
            key, source = chat_key(getattr(self.h.actions, "keys", None))
            if key is None:
                return self._end("left", f"the chat box expanded: {source}: left so")
            self.key = key
        client.check_alarm()  # (the player refuses it too: as every plan of hers)
        self.h.actions.play([[0, self.key, "down"], [40, self.key, "up"]], "chat_collapse")
        self.sent.append(self.key)
        self.presses += 1
        self._clear = None
        sent = time.monotonic()
        told = None
        while True:
            try:
                s = self._look()
            except LookLost as lost:
                return self._defer(None, at_start, f"not told (no look: {lost.why})")
            told = getattr(s, "chat", None)
            t = getattr(s, "t", None)
            t = time.monotonic() if t is None else t
            if told is not None and t >= sent + SETTLE_S:  # (a frame from after the key has had its time)
                if told.state == COLLAPSED:
                    return self._end("collapsed", f"the chat box expanded: collapsed ({self.key})")
                if told.state == EXPANDED and (told.typing or told.marks > before.marks):
                    return self._end("left", f"the chat box still expanded after {self.key}, its input line marked "
                                             "now (the key typed into it?): left so")
                if told.state == EXPANDED and t >= sent + STILL_S:
                    if self.presses >= PRESSES_MAX:
                        return self._end("left", f"the chat box still expanded after {self.key} twice: left so")
                    self.log(f"  the chat box still expanded after {self.key}: again")
                    return None
            if time.monotonic() - sent >= VERIFY_S:
                words = "no look" if told is None else told.words()
                return self._end("left", f"the chat box not seen collapsed after {self.key} ({words} "
                                         f"{VERIFY_S:g} s on): left so")
            time.sleep(QUIET_LOOK_S)

    def _not_read(self) -> Outcome:
        """No state of the box on a look: said (once: the check is done) where the looks are a perception service's,
        which reads it on every frame since this, so it's one started from the code before; else (the simulator's
        eyes, a test's) nothing said."""
        if session_of(self.h) is None:
            return self._end("not read", "", say=False)
        return self._end("not read", "the chat box not read (the perception service's looks carry none: one started "
                                     "from the code before it? restart it to have it collapsed): nothing sent")

    def _defer(self, why: str | None, at_start: bool, told: str | None = None) -> Outcome:
        """Put off: expanded and not quiet (`why`), or not told (`told`: the box's words, or no look): in the hunt loop
        tried again at a quiet look that tells it (`again`), DEFERS_MAX times; else left so."""
        self.defers += 1
        self._clear = None
        what = "expanded" if told is None else told
        if self.defers > DEFERS_MAX or not self._loops(at_start):
            many = f"{DEFERS_MAX} times put off" if self.defers > DEFERS_MAX else ""
            if told is None:
                return self._end("left", f"the chat box expanded: no quiet moment to collapse it ({why}"
                                         + (f"; {many}" if many else "") + "): left so this run")
            after = f"after {self._key_words()}: left so" if self.presses else self._unsent()
            return self._end("left" if self.presses else "not told",
                             f"the chat box {told}" + (f" ({many})" if many else "") + f": {after}")
        self.pending, self.pending_at = True, time.monotonic()
        text = (f"the chat box {what}: its collapse put off" + (f" ({why})" if why else "")
                + ": again at a quiet moment in the hunt"
                + (f" ({self.defers} of {DEFERS_MAX})" if self.defers > 1 else ""))
        self.log(f"  {text}")
        telemetry.emit("event", kind="chat_box", text=text, what="put off", sent=list(self.sent))
        out = Outcome("put off", text, tuple(self.sent), time.monotonic() - self._began)
        self.last = out
        return out

    def _end(self, what: str, text: str, say: bool = True) -> Outcome:
        """Done this run: `text` said (`say`), a telemetry event 'chat_box'."""
        self.done, self.pending = True, False
        out = Outcome(what, text, tuple(self.sent), time.monotonic() - self._began)
        if say:
            self.log(f"  {text}")
            telemetry.emit("event", kind="chat_box", text=text, what=what, sent=list(self.sent),
                           seconds=round(out.seconds, 2))
        self.last = out
        return out
