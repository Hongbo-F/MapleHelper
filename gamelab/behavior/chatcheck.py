"""Her chat box collapsed at a hunt's start if it's expanded (the user, 2026-10-30 ~12:45: "At the start of grind/hunt,
you should collapse the chat window if it's open."). Expanded, the box covers the play area's bottom left (x 287-850, y
633-700 at 1366x768: four lines of chat over the bottom platforms), hiding the monsters there and her plate when she's
under it, where her place by the scene does worse than her dot alone (localization.locate); hunts started with it so on
2026-10-29 at 09:26 and 15:58 and on 10-30 at 08:13 (lab/shots/statwindow_*.png: her stat window's frames at their
starts), and on 10-30 the user collapsed it by hand 34 s into the grind's recording of 08:17:50.

**What tells it** (perception.chatbox: the box's +/- button and its input row, at 1366x768): every frame the
perception service reads (Observation.chat, WorldState.chat), as each of the hunt's looks has it (Scene.chat). None
(the simulator's eyes, a perception service started from the code before this): nothing is done, nothing said.

**When** (`ChatCheck.run`): at a hunt's start (`hunt`, `tour`, `probe run`: gamelab.__main__'s _hunter,
Hunter.collapse_chat, before her stat window's read: the box hides monsters its quiet moment looks for), once a run:
1. looks until one tells, QUIET_S at most. Collapsed: said, nothing more. Not told on any (unknown: its button covered,
   the UI greyed while a map loads, another client size): said, nothing sent (her chat key toggles the box: pressed with
   it collapsed it would expand it). Its input line holding text or a caret (ChatBox.typing: it may have the keyboard,
   and the key would be typed into it): said, nothing sent; no Esc first (with nothing to close, Esc opens the game's
   MENU) and no Enter (it sends what's typed).
2. expanded: a quiet moment (keycheck.unquiet: placed on a platform, not on a rope nor at its top, untouched for
   KNOCK_S, no monster within PLACE_CLEAR of her) within QUIET_S; a monster within PLACE_CLEAR puts it off at once
   (nothing moves her for it, and standing there it would come to her); none, put off. In the hunt loop (`tour --loop`,
   `--grind`) it's raised again (ChatDue, from Hunter.health_check) at the first quiet look DEFER_GAP_S or more later,
   that look its first, DEFERS_MAX times at the most, then left so this run; elsewhere (`hunt kill`, a plain `tour`,
   `probe run`) not this run. Every look asks the safety stop (her HP watch's due), as her stat window's do.
3. her chat key (`chat_key`: CHAT+'s in her remembered key bindings, else ', CHAT+'s in her key bindings of
   2026-10-28; never one of her own keys, Esc, the backslash or maplehelper's hotkeys) tapped once, through the player
   as every key of hers (control.player: every key config.toml can name, ' among them; with the Lie Detector alarm up
   it refuses the plan, client.LieDetectorAlarm, and the alarm is asked before the tap too). Then looks, SETTLE_S after
   the key and on, until one tells, VERIFY_S at most: collapsed, "the chat box expanded: collapsed (')"; still
   expanded, the key again at a quiet look (PRESSES_MAX, 2, in all), then "the chat box still expanded after ' twice:
   left so"; its input line holding text after it (the key typed into it), left so; not told within VERIFY_S, left so
   (never pressed on a look that can't tell).
Only her chat key goes out: nothing that moves or attacks her, no mouse. A telemetry event 'chat_box' at the end."""

import time
from collections.abc import Callable
from dataclasses import dataclass

from .. import telemetry
from ..control import client
from ..perception.chatbox import COLLAPSED, EXPANDED
from .keycheck import QUIET_LOOK_S, QUIET_S, near_her, unquiet
from .statcheck import bound_key

FUNCTION = "CHAT+"     # the key bindings' name of what toggles the box (perception.keybindings.FUNCTIONS[16])
KEY = "'"              # ... its key in her key bindings of 2026-10-28 (both characters' remembered bindings since)
PRESSES_MAX = 2        # her chat key at most this many times a run: once, and once more if it's still expanded
SETTLE_S = 0.3         # s after her chat key before a look tells what it did (the box redrawn the next frame: 515)
VERIFY_S = 1.5         # ... and this long at most for a look that tells
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
    the simulator's eyes), "put off" (not quiet: in the hunt loop tried again later), "skipped" (done this run already);
    what was sent (key names) and how long it took."""
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
    `presses`: her chat key's taps this run; `done`: nothing more this run."""

    def __init__(self, h, log: Callable[[str], None] | None = None):
        self.h = h
        self.log = log or h.log
        self.presses = 0
        self.done = False
        self.pending = False      # put off (not quiet), tried again in the hunt loop (`again`)
        self.pending_at = 0.0
        self.defers = 0
        self.sent: list[str] = []
        self.key: str | None = None
        self.last: Outcome | None = None
        self._began = 0.0

    def again(self, t: float) -> bool:
        """Whether a collapse put off is due again at `t` (a quiet look: Hunter.health_check), DEFER_GAP_S after the
        last try at the soonest."""
        return self.pending and not self.done and t - self.pending_at >= DEFER_GAP_S

    def run(self, at_start: bool = False, look=None) -> Outcome:
        """The check (the module's doc), unless it's done this run. `at_start`: a hunt's start (put off there, it's
        tried again only in the hunt loop); `look`: its first look (ChatDue's, the quiet one it was raised at: a look
        taken again a moment later had a monster come within PLACE_CLEAR, in the simulator's crowd, and it was put off
        again). Raises SafetyStop (the HP watch, at every look) and client.LieDetectorAlarm (the alarm up: at a look,
        before the key, or the player refusing it)."""
        if self.done:
            return Outcome("skipped", "the chat box: done this run")
        self._began = time.monotonic()
        self.pending = False
        end = self._began + QUIET_S
        why = "not looked"
        while True:
            s, look = (self._look() if look is None else look), None
            told = getattr(s, "chat", None)
            if told is None:  # (eyes that don't read it: the simulator's, a service of the code before)
                return self._end("not read", "", say=False)
            if told.state == COLLAPSED:
                return self._end("was collapsed", "the chat box: collapsed")
            if told.state == EXPANDED and told.typing:
                return self._end("left", "the chat box expanded, its input line with text or a caret (it may have "
                                         "the keyboard: a key would be typed into it): left so, nothing sent")
            if told.state == EXPANDED:
                if self.presses >= PRESSES_MAX:
                    return self._end("left", f"the chat box still expanded after {self._key_words()} twice: left so")
                why = unquiet(self.h, s)
                if why is None:
                    if (out := self._press()) is not None:
                        return out
                    end = time.monotonic() + QUIET_S  # (still expanded after it: the key again at a quiet look)
                    continue
                if near_her(s, getattr(self.h, "model", None)):
                    return self._defer(why, at_start)
            if time.monotonic() >= end:
                if told.state == EXPANDED:
                    return self._defer(why, at_start)
                return self._end("not told", f"the chat box {told.words()}: nothing sent")
            time.sleep(QUIET_LOOK_S)

    def _key_words(self) -> str:
        return self.key or KEY

    def _look(self):
        """A look of the check's: the safety stop asked of it as Hunter.health_check would (her HP watch's due)."""
        s = self.h.look()
        hp = getattr(self.h, "hp_watch", None)
        if hp is not None and (stop := hp.due()) is not None:
            raise stop
        return s

    def _press(self) -> Outcome | None:
        """Her chat key tapped, then looks SETTLE_S after it and on until one tells (VERIFY_S at most): the outcome, or
        None if it's still expanded (the key again at a quiet look: `run`)."""
        if self.key is None:
            key, source = chat_key(getattr(self.h.actions, "keys", None))
            if key is None:
                return self._end("left", f"the chat box expanded: {source}: left so")
            self.key = key
        client.check_alarm()  # (the player refuses it too: as every plan of hers)
        self.h.actions.play([[0, self.key, "down"], [40, self.key, "up"]], "chat_collapse")
        self.sent.append(self.key)
        self.presses += 1
        sent = time.monotonic()
        while True:
            s = self._look()
            told = getattr(s, "chat", None)
            late = getattr(s, "t", sent + SETTLE_S) >= sent + SETTLE_S  # (a frame from after the key has had its time)
            if told is not None and late and told.state == COLLAPSED:
                return self._end("collapsed", f"the chat box expanded: collapsed ({self.key})")
            if told is not None and late and told.state == EXPANDED:
                if told.typing:
                    return self._end("left", f"the chat box still expanded after {self.key}, its input line with text "
                                             "or a caret now (the key typed into it?): left so")
                if self.presses >= PRESSES_MAX:
                    return self._end("left", f"the chat box still expanded after {self.key} twice: left so")
                self.log(f"  the chat box still expanded after {self.key}: again")
                return None
            if time.monotonic() - sent >= VERIFY_S:
                words = "no look" if told is None else told.words()
                return self._end("left", f"the chat box not seen collapsed after {self.key} ({words} "
                                         f"{VERIFY_S:g} s on): left so")
            time.sleep(QUIET_LOOK_S)

    def _defer(self, why: str, at_start: bool) -> Outcome:
        """Put off (`why`): in the hunt loop tried again at a quiet look (`again`), DEFERS_MAX times; else left so."""
        self.defers += 1
        loop = bool(getattr(self.h, "doubts_raise", False)) or at_start
        if self.defers > DEFERS_MAX or not loop:
            return self._end("left", f"the chat box expanded: no quiet moment to collapse it ({why}"
                                     + (f"; {DEFERS_MAX} times put off" if self.defers > DEFERS_MAX else "")
                                     + "): left so this run")
        self.pending, self.pending_at = True, time.monotonic()
        text = (f"the chat box expanded: its collapse put off ({why}): again at a quiet moment in the hunt"
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
