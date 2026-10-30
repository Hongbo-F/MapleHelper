"""Her stat window opened, read and closed: her W.DEF for the safety stop's reserve and the positioning model, from the
window rather than a prior until her touches fit it. The user, 2026-10-28 ~18:20, on a first run's DEF not known (the
reserve at DEF 0, 453 HP of 840 on Tree Dungeon, until 8 touches are kept): "You can open character stats panel to
check the defense".

**When** (`StatCheck.due`): at a hunt's start (`hunt`, `tour`, `probe run`: gamelab.__main__'s _hunter, Hunter.read_stats,
after her max HP is read), if her DEF isn't known from the window this session (bootstrap.HerStats.window_due: none read,
read at another level, in another session of the perception service or over SESSION_S ago, her newest touches since
not fitting it), and only
- with the reader calibrated on this client (perception.stat_window.Calibration, lab/stat_window.json: a live capture,
  `gamelab stats --read --record`, found the window and where it opens; till then nothing is sent, said once),
- with her stats key known and safe (`stats_key`: ABILITY's key in her remembered key bindings, lab/keybindings-<her
  id>.json, where there is one; else S, ABILITY's in her key bindings of 2026-10-28; never one of her own keys, Esc, the
  backslash or maplehelper's hotkeys: refused, said),
- READS_MAX times a run at the most.

**How** (`StatCheck.run`, each step said in the log, a telemetry event 'stat_window' at the end):
1. a quiet moment (keycheck.unquiet: on a platform, not on a rope nor at its top, untouched for KNOCK_S, no monster
   within PLACE_CLEAR of her; her still over two looks QUIET_LOOK_S apart) within QUIET_S; none, or a monster within
   PLACE_CLEAR (at once: nothing moves her for it, and standing there it would come to her), put off. In the hunt
   loop (`tour --loop`, `--grind`) it's raised again (StatsDue, from Hunter.health_check) at the first quiet look
   DEFER_GAP_S or more later, DEFERS_MAX times at the most, then not read this run; elsewhere not read this run. Every
   look of it asks the safety stop (her HP watch's due), as the key check's do. No retreat: nothing moves her for it.
2. a frame first: the client not at 1366x768, nothing is sent (the window is read at that size); open already, read as
   it is (then closed).
3. the mouse off where the window opens (MOUSE_AT, or another place off the calibrated panels: a pointer over a value
   draws its tooltip), her stats key tapped, frames SETTLE_S, AGAIN_S and LATE_S after one another until it's seen open
   (a panel found by its picture: perception.stat_window.find). Not seen: the frame against the one before the key
   where the calibration has the Main panel (CHANGED_SHARE of its px changed by CHANGE_LEVEL), unchanged it didn't
   open (nothing more sent), else it may be open.
4. the DETAIL panel (her W.DEF's) not open with it: a click on its DETAIL button (its centre, the Main panel as found:
   the only click sent), the mouse off again, frames DETAIL_S and DETAIL_AGAIN_S until it's seen; not seen, the Main
   panel's values are read without her W.DEF (said).
5. read (perception.stat_window.read: both panels' value columns by the OCR, one frame; a second frame if W.DEF read no
   number), the frame kept (lab/shots/statwindow_<time>.png, READS_MAX a run).
6. closed: her stats key again (it toggles the window), frames CLOSE_S and CLOSE_AGAIN_S later; still open on both, Esc,
   and so once more; still open it may be open. Esc is never sent with no panel seen (with nothing open Esc opens the
   game's MENU).
It **may be open** (not seen closed after her stats key, or no frame to tell by): StatWindowStop (a HuntStop: her retreat
in a hunt, the page "close it with S or Esc", exit code 7; at a hunt's start nothing is hunted). With the Lie Detector
alarm up the player refuses its keys (client.LieDetectorAlarm, as every plan's; a window that may be open then is said
and paged). Only her stats key, one click and Esc go out: nothing that attacks or moves her.

**What then** (Hunter._stats_read), each checked before it's taken (the review of 2026-10-28: a misread was taken as
read): its level or max HP not her HUD's where both are read (`disagreement`), none of it. Else her W.DEF hers
(bootstrap.HerStats.see_window: the reserve and the positioning model take it, her touches only check it; not one
outside 0 to DEF_TOP, nor one her touches don't bear out: HerStats.window_doubt), kept with her level and the session;
her max HP if her HUD's number hasn't read, where her HUD's as her stats file last kept it or her HP bar bears it out
(`borne`); her level where nothing has read it; her max MP her casts are reckoned at (Hunter.max_mp, Attacks.max_mp:
config.toml [character] mp until then, said), where her MP bar bears it out or it's within MP_SPAN of theirs; ATTACK,
ACCURACY, EVASION, MAGIC, MAGIC DEF., her four stats, SPEED and JUMP kept in her stats file (not modelled from yet:
said where SPEED and JUMP differ from config.toml's, which plan her moves)."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np

from .. import paths, telemetry
from ..control import alarm, client
from ..perception import stat_window as sw
from ..perception.framecheck import BadLook
from .bootstrap import session_of_feed
from .keycheck import NOT_ATTACK, QUIET_LOOK_S, QUIET_S, _still, near_her, unquiet
from .safety import HuntStop

READS_MAX = 2          # reads a run at the most: the start's, and one after a level-up in the hunt loop (`level_up`)
SETTLE_S = 0.6         # s after her stats key before the window's frame (the key check's) ...
AGAIN_S = 0.4          # ... one more this long after, if it isn't open yet ...
LATE_S = 1.0           # ... and one more this long after that (2 s after the key: its opening isn't measured live)
DETAIL_S = 0.4         # s after the click on DETAIL before the frame that shows its panel ...
DETAIL_AGAIN_S = 0.6   # ... and one more this long after
CLOSE_S = 0.6          # s after her stats key (closing it) or Esc before the frame that says it closed ...
CLOSE_AGAIN_S = 0.4    # ... and one more this long after before Esc
CHANGE_LEVEL = 48      # a px changed (any channel) by more than this ...
CHANGED_SHARE = 0.85   # ... over this share of where the Main panel opens: something opened there (keycheck's)
MOUSE_AT = (1100, 300)  # client px at 1366x768 the pointer goes to first (the key check's), or ...
MOUSE_ELSE = ((1300, 420), (120, 420), (683, 60))  # ... the first of these off every calibrated panel
CLICK_HOLD_MS = 60     # the click's button held this long
DEFER_GAP_S = 5.0      # a read put off is tried again at a quiet look this long after at the soonest ...
DEFERS_MAX = 6         # ... this many times, then not this run
EXIT = 7               # the command's exit code after StatWindowStop (the key check's 6)
SHOTS_MAX = READS_MAX  # the window's frames kept a run
# What a read is checked by before it's taken (the review of 2026-10-28: a misread W.DEF, max HP or max MP was taken as
# read): its level and max HP her HUD's where both are read (`disagreement`: else none of it is taken); its max HP, taken
# only while her HUD's number hasn't read, her HUD's as her stats file last kept it or borne out by her HP bar (`borne`);
# its max MP borne out by her MP bar, or within MP_SPAN of what her casts are reckoned at
BAR_TOL = 0.05         # her bar's share against the window's cur / max: this far apart at the most
MP_SPAN = 1.25         # the window's max MP within this factor (either way) of her casts' max MP
VALUE_TOP = 30000      # the game's cap on max HP and MP: a number over it is a misread


class StatWindowStop(HuntStop):
    """Her stat window may be open in the game (not seen closed after her stats key and Esc): the hunt ends, or at its
    start nothing is hunted (raised by StatCheck.run). As every HuntStop: her retreat in a hunt, said in capitals,
    paged, the command's own exit code."""
    label = "STAT WINDOW"
    title = "Stat window"
    event = "stat_window_stop"
    exit_code = EXIT
    ends = "to a safe place (no more throws)"
    advice = "Close her stat window in the game (her stats key, S, or Esc) before hunting again."

    def __init__(self, text: str, at: float = 0.0, at_start: bool = False):
        self.at = at
        if at_start:  # (raised before the hunt: nothing hunted, nobody else to page)
            self.after = "Nothing was hunted."
        super().__init__(text)

    def page(self) -> tuple[str, str]:
        return ("MapleStory: her stat window", f"gamelab stopped: {self}.")


class StatsDue(Exception):
    """Raised by Hunter.health_check in the hunt loop at a quiet look when a read put off is due again: the hunt loop
    reads her stat window (Hunter.read_stats) and goes on. Not a HuntStop: the hunt doesn't end."""


@dataclass
class Outcome:
    """What a read did: `what` "read" (her W.DEF read), "partly" (the window read, not her W.DEF: the DETAIL panel not
    seen, no number in its cell), "not opened" (nothing sent, or her stats key and nothing opened), "put off" (not
    quiet: in the hunt loop tried again later), "skipped" (not due, not calibrated, no key: `text` says why); the
    reading, what was sent (`sent`: key names), how long it took."""
    what: str
    text: str
    reading: sw.StatReading | None = None
    sent: tuple = ()
    seconds: float = 0.0


def _remembered(character: str | None = None) -> tuple[dict | None, str]:
    """Her remembered key bindings' map of what each key does (control.keymap's lab/keybindings-<her id>.json,
    'functions'), and its file in words; (None, "") without one."""
    try:
        from .. import character as profiles
        who = character or profiles.active().id
    except Exception:
        who = character or "thief"
    name = getattr(paths, "KEYBINDINGS", "keybindings-{}.json").format(who)
    path = paths.LAB / name
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, ""
    fn = data.get("functions") if isinstance(data, dict) else None
    if not isinstance(fn, dict) or not fn:
        return None, ""
    return fn, f"{path.name}, read {data.get('read_at', '?')}"


def _canon(key: str) -> str:
    return {"rctrl": "ctrl", "rshift": "shift", "ralt": "alt"}.get(key, key)


def stats_key(keys, character: str | None = None) -> tuple[str | None, str]:
    """Her stats key and where it's from, or (None, why not): ABILITY's key in her remembered key bindings (`_remembered`),
    else KEY (S: ABILITY in her key bindings of 2026-10-28, lab/shots/keyconfig.png); refused where it's one of her own
    keys (config.toml [keys] as she plays them: `keys`, a GameKeys), Esc, the backslash or maplehelper's hotkeys
    (keycheck.NOT_ATTACK), or her remembered bindings have ABILITY on no key."""
    return bound_key(sw.FUNCTION, sw.KEY, "S, ABILITY's in her key bindings of 2026-10-28 (none remembered since)", keys,
                     character)


def bound_key(function: str, default: str, default_source: str, keys,
              character: str | None = None) -> tuple[str | None, str]:
    """The key of the game's `function` (its name in the key bindings window: perception.keybindings.FUNCTIONS) and
    where it's from, or (None, why not): its key in her remembered key bindings (`_remembered`), else `default` (said
    as `default_source`); refused as `stats_key` says (behavior.chatcheck's CHAT+ too)."""
    fn, where = _remembered(character)
    if fn is not None:
        on = sorted(k for k, f in fn.items() if f == function)
        if not on:
            return None, f"{function} on no key in her remembered key bindings ({where})"
        key, source = on[0], f"{function}'s in her remembered key bindings ({where})"
    else:
        key, source = default, default_source
    if key in NOT_ATTACK:
        return None, f"{key.upper()} ({source}) isn't a key to press for it"
    mine = {}
    for f in fields(keys) if keys is not None else ():
        k = getattr(keys, f.name, None)
        if k is not None and getattr(k, "name", None):
            mine[_canon(k.name)] = f.name
    if _canon(key) in mine:
        return None, f"{key.upper()} ({source}) is her {mine[_canon(key)].replace('_', ' ')} key too: not pressed for it"
    return key, source


def disagreement(reading: sw.StatReading, stats) -> str | None:
    """Where her stat window's read disagrees with her HUD's as `stats` (bootstrap.HerStats) has them: its level and her
    level, its max HP and her HUD's HP number's, each where both are read; in words, None if nowhere. The two show the
    same numbers in game, so a disagreement says the read is off (the panel's cells, the OCR): none of it is taken."""
    off = []
    level, most = getattr(stats, "level", None), getattr(stats, "max_hp", None)
    if reading.level is not None and level is not None and int(reading.level) != int(level):
        off.append(f"level {reading.level}, {getattr(stats, 'level_from', '') or 'her level as read'} {level}")
    if reading.max_hp is not None and most is not None and int(reading.max_hp) != int(most):
        off.append(f"max HP {reading.max_hp}, {getattr(stats, 'HP_FROM', 'her HUD')} {most}")
    return " and ".join(off) or None


def borne(pair, share: float | None, kept: float | None = None, span: float | None = None) -> str | None:
    """Why the window's cur / max `pair` (HP or MP) isn't taken (None: it is): not a cur / max (both read, 0 <= cur <=
    max <= VALUE_TOP); else borne out by `kept` (the same max as known otherwise: her HUD's as last kept, with `span`
    None) or within `span` of it (either way), or by her bar's `share` (cur under max, the window's share within BAR_TOL
    of it: at full her bar can't tell a misread max); else not borne out."""
    if not isinstance(pair, (tuple, list)) or len(pair) != 2 or None in pair:
        return f"{pair} isn't a cur / max read"
    cur, most = float(pair[0]), float(pair[1])
    if not 0 <= cur <= most or not 0 < most <= VALUE_TOP:
        return f"{pair[0]} / {pair[1]} isn't a cur / max (0 <= cur <= max <= {VALUE_TOP})"
    if kept is not None and kept > 0:
        if (most == float(kept)) if span is None else (1 / span <= most / float(kept) <= span):
            return None
    if share is not None and cur < most and abs(cur / most - float(share)) <= BAR_TOL:
        return None
    bar = ("her bar not read" if share is None else
           "at full her bar can't tell a misread max" if cur >= most else
           f"her bar at {float(share):.0%} against its {cur / most:.0%}")
    known = ("" if kept is None else f"not {kept:g} as known" if span is None
             else f"not within x{span:g} of {kept:g} as known")
    return ", ".join(x for x in (known, bar) if x) + ": not borne out"


def session_of(h) -> str | None:
    """The perception service's run hunter `h`'s looks come from (its pid: a session of the game's, as near as it's
    known), None without one (the simulator, a test's hunter)."""
    return session_of_feed(getattr(getattr(h, "senses", None), "feed", None))


class StatCheck:
    """Her stat window read for hunter `h` (see the module's doc): its senses, actions, stats and log. `reader`: what
    reads a frame (perception.stat_window.read with the eyes' OCR: the simulator's eyes have their own, `stat_ocr`).
    `window_state` after `_window`: "not opened", "closed", "may be open"; `sent`: the keys and click sent."""

    def __init__(self, h, log: Callable[[str], None] | None = None, cal: sw.Calibration | None = None):
        self.h = h
        self.log = log or h.log
        self._cal = cal
        self.reads = 0
        self.frames_kept = 0
        self.pending = False      # a read put off (not quiet), tried again in the hunt loop (`again`)
        self.pending_at = 0.0
        self.defers = 0
        self.said_skip = ""       # why it isn't read, as last said (said once)
        self.window_state = "not opened"
        self.sent: list[str] = []
        self.last: Outcome | None = None
        self.scene = None         # the read's last look (its bars: what the window's HP and MP are checked by)
        self._last_frame = None
        self._near: dict = {}

    @property
    def cal(self) -> sw.Calibration | None:
        if self._cal is None:
            self._cal = sw.Calibration.load()
        return self._cal

    def _session(self) -> str | None:
        return session_of(self.h)

    def due(self, force: bool = False) -> str | None:
        """Why the window isn't read now (None: it's due): her DEF known from it this session (not with `force`),
        the reader not calibrated, no safe stats key, READS_MAX reached."""
        h = self.h
        why = None if force else h.stats.window_due(h.stats.level, self._session(), h.touches.rows)
        if why is None and not force:
            return f"her {h.stats.defence.words()}, this session's"
        if self.reads >= READS_MAX:
            return f"read {self.reads} times this run already"
        if self.cal is None:
            return ("its reader not calibrated on this client yet: run `python -m gamelab stats --read --record` once, "
                    "the game up and her somewhere quiet")
        if (key := stats_key(h.actions.keys)[0]) is None:
            return stats_key(h.actions.keys)[1]
        self.key = key
        return None

    def level_up(self, level: int) -> None:
        """Her level read as `level` in a hunt (Hunter._level_seen): her W.DEF read off the window at another, the read
        again at the hunt loop's next quiet look (pending, due at once), READS_MAX a run; said."""
        read = self.h.stats.window_read.get("level") if self.h.stats.window is not None else None
        if (read is None or int(read) == int(level) or self.reads >= READS_MAX or self.pending
                or self.cal is None):
            return
        self.pending, self.pending_at, self.defers = True, time.monotonic() - DEFER_GAP_S, 0
        self.log(f"  her level {level} now, her W.DEF read at level {read}: her stat window again at a quiet moment in "
                 "the hunt")

    def again(self, t: float) -> bool:
        """Whether a read put off is due again at `t` (a quiet look: Hunter.health_check), DEFER_GAP_S after the last
        try at the soonest."""
        return self.pending and t - self.pending_at >= DEFER_GAP_S

    def run(self, at_start: bool = False, quiet: bool = True, force: bool = False) -> Outcome:
        """A read (the module's doc), if it's due (else "skipped", said once). `at_start`: a hunt's start (put off
        there, it's tried again only in the hunt loop); `quiet` False: no quiet moment waited for (`gamelab stats
        --read`, whoever is at the PC watching); `force`: read whatever her stats file says. Raises StatWindowStop
        (the window may be open), SafetyStop (the HP watch, at every look) and client.LieDetectorAlarm (the player
        refuses its keys then)."""
        start = time.monotonic()
        if (why := self.due(force)) is not None:
            self.pending = False
            out = Outcome("skipped", f"her stat window not read: {why}")
            if why != self.said_skip:
                self.said_skip = why
                self.log(f"  {out.text}")
            return out
        if quiet and (why := self._quiet()) is not None:
            return self._defer(why, at_start)
        self.pending = False
        self.sent = []
        reading = self._window()
        if self.window_state == "may be open":
            text = ("her stat window may be open in the game (her stats key sent, it wasn't seen closed): close it with "
                    f"{self.key.upper()} or Esc" + ("" if reading is None or not reading.open
                                                     else f"; it read {reading.text()}"))
            telemetry.emit("event", kind="stat_window", text=text, what="stopped", sent=list(self.sent),
                           seconds=round(time.monotonic() - start, 2),
                           **({"reading": reading.record()} if reading is not None else {}))
            stop = StatWindowStop(text, at=time.monotonic(), at_start=at_start)
            if at_start:  # (in a hunt her retreat pages it: behavior.retreat.on_alarm)
                self.log(f"{stop.label}: {stop}")
                alarm.page(*stop.page())
            raise stop
        out = self._outcome(reading)
        out.seconds = time.monotonic() - start
        out.sent = tuple(self.sent)
        self.log(f"  {out.text} ({out.seconds:.1f} s; sent {', '.join(out.sent) or 'nothing'})")
        if out.what in ("read", "partly"):
            self.reads += 1
            self.h._stats_read(reading, out)  # (her stats, her reserve made again: what changed said after it)
        telemetry.emit("event", kind="stat_window", text=out.text, what=out.what, sent=list(out.sent),
                       seconds=round(out.seconds, 2), **({"reading": reading.record()} if reading is not None else {}))
        self.last = out
        return out

    def _outcome(self, reading: sw.StatReading | None) -> Outcome:
        if reading is None:
            return Outcome("not opened", "her stat window not read: no frame of the game")
        if not reading.open:
            return Outcome("not opened", f"her stat window not read: {reading.why or 'it did not open'}", reading)
        if reading.wdef is None:
            return Outcome("partly", f"her stat window read without her W.DEF ({reading.why}): {reading.text()}",
                           reading)
        return Outcome("read", f"her stat window: {reading.text()}", reading)

    # --- 1. a quiet moment -------------------------------------------------------------------------------------------

    def _look(self):
        """A look of the read's: the safety stop asked of it as Hunter.health_check would (her HP watch's due)."""
        s = self.h.look()
        self.scene = s  # (her bars as of the read's last look: `borne`, Hunter._stats_read)
        hp = getattr(self.h, "hp_watch", None)
        if hp is not None and (stop := hp.due()) is not None:
            raise stop
        return s

    def _quiet(self) -> str | None:
        """Her still over two looks QUIET_LOOK_S apart with nothing near (keycheck.unquiet) within QUIET_S; a monster
        within PLACE_CLEAR of her puts it off at once (nothing moves her for it: standing there waiting, it would come
        to her). Why it's put off, or None."""
        last, why = None, "not looked"
        end = time.monotonic() + QUIET_S
        while True:
            s = self._look()
            why = unquiet(self.h, s)
            if why is None and last is not None and _still(last, s):
                return None
            if why is not None and near_her(s, self.h.model):
                return why
            if time.monotonic() >= end:
                return why or "not standing still"
            last = s if why is None else None
            time.sleep(QUIET_LOOK_S)

    def _defer(self, why: str, at_start: bool) -> Outcome:
        """Put off (`why`): in the hunt loop tried again at a quiet look (`again`), DEFERS_MAX times; else not this run."""
        self.defers += 1
        loop = bool(getattr(self.h, "doubts_raise", False)) or at_start
        if self.defers > DEFERS_MAX or not loop:
            self.pending = False
            text = (f"her stat window not read: no quiet moment for it ({why}"
                    + (f"; {DEFERS_MAX} times put off" if self.defers > DEFERS_MAX else "") + "): not this run")
        else:
            self.pending, self.pending_at = True, time.monotonic()
            text = (f"her stat window put off ({why}): again at a quiet moment in the hunt"
                    + (f" ({self.defers} of {DEFERS_MAX})" if self.defers > 1 else ""))
        self.log(f"  {text}")
        telemetry.emit("event", kind="stat_window", text=text, what="put off")
        out = Outcome("put off", text)
        self.last = out
        return out

    # --- 2-6. the window ---------------------------------------------------------------------------------------------

    def _frame(self):
        """The game window's whole client area now (RGB: Eyes._capture), or None (said)."""
        try:
            frame = self.h.senses.eyes._capture()
        except (BadLook, RuntimeError) as e:
            self.log(f"  her stat window: no frame ({getattr(e, 'why', e)})")
            return None
        self._last_frame = frame
        return frame

    def _ocr(self):
        return getattr(self.h.senses.eyes, "stat_ocr", None)

    def _read(self, frame=None) -> sw.StatReading | None:
        """A frame now (or `frame`), read (None: no frame); where its panels are kept for the next frame's find."""
        frame = self._frame() if frame is None else frame
        if frame is None:
            return None
        reading = sw.read(frame, self._ocr(), self.cal, self._near)
        self._near.update({k: v for k, v in reading.at.items() if v is not None})
        return reading

    def _seen(self, frame) -> dict:
        """Where each panel is on `frame` (no OCR): name -> (x, y) or None."""
        found = sw.find_all(frame, self.cal, self._near)
        self._near.update({k: v[0] for k, v in found.items() if v[0] is not None})
        return {k: v[0] for k, v in found.items()}

    def _mouse_at(self) -> tuple[int, int]:
        """MOUSE_AT, or the first of MOUSE_ELSE off every panel as calibrated and as seen (16 px clear)."""
        boxes = []
        for name, t in sw.templates(self.cal).items():
            at = self._near.get(name) or (self.cal.place(name) if self.cal is not None else None)
            if at is not None:
                w, h = t.size
                boxes.append((at[0] - 16, at[1] - 16, at[0] + w + 16, at[1] + h + 16))
        for x, y in (MOUSE_AT, *MOUSE_ELSE):
            if not any(a <= x < c and b <= y < d for a, b, c, d in boxes):
                return x, y
        return MOUSE_AT

    def _tap(self, key: str, name: str) -> None:
        self.h.actions.play([[0, key, "down"], [40, key, "up"]], name)
        self.sent.append(key)

    def _changed(self, before, after) -> bool:
        """Whether `after` differs from `before` where the calibration has the Main panel open: CHANGED_SHARE of its px
        by more than CHANGE_LEVEL in a channel. Not told (a frame missing, another size, no place): changed."""
        at = self.cal.place("main") if self.cal is not None else None
        if before is None or after is None or before.shape != after.shape or at is None:
            return True
        t = sw.templates(self.cal).get("main")
        w, h = t.size if t is not None else (185, 281)
        x, y = at
        a = before[y:y + h, x:x + w, :3].astype(np.int16)
        b = after[y:y + h, x:x + w, :3].astype(np.int16)
        return a.size > 0 and float((np.abs(a - b).max(axis=2) > CHANGE_LEVEL).mean()) >= CHANGED_SHARE

    def _window(self) -> sw.StatReading | None:
        """Her stat window opened, read and closed (the module's doc, 2-6): its reading (None: no frame);
        `window_state` and `sent` say how it was left."""
        client.check_alarm()
        act = self.h.actions
        self.window_state = "not opened"
        try:
            before = self._frame()
            if before is None:
                self.log("  her stat window: no frame of the game to read it on: not opened")
                return None
            fh, fw = before.shape[:2]
            if (fw, fh) != sw.FRAME:
                self.log(f"  her stat window: the client is {fw}x{fh}, not {sw.FRAME[0]}x{sw.FRAME[1]}: not opened "
                         "(it's read at that size)")
                return sw.StatReading(why=f"the frame is {fw}x{fh}, not {sw.FRAME[0]}x{sw.FRAME[1]}")
            act.play([[0, "mouse", *self._mouse_at()]], "stat_window_mouse")
            self.sent.append("mouse")
            shown = self._frame()
            if shown is None:
                self.log("  her stat window: no frame after the mouse moved: not opened")
                return None
            seen = self._seen(shown)
            already = any(v is not None for v in seen.values())
            if already:
                self.window_state = "may be open"
            else:
                behind = shown
                # (before the plan: one the Lie Detector stops halfway may have sent her stats key, and the player's
                # refusal raises out of it: the review of 2026-10-28, a window open through the test with nobody told)
                self.window_state = "may be open"
                self._tap(self.key, "stat_window")
                for wait in (SETTLE_S, AGAIN_S, LATE_S):
                    time.sleep(wait)
                    shown = self._frame()
                    seen = self._seen(shown) if shown is not None else {}
                    if any(v is not None for v in seen.values()):
                        break
                if shown is None or not any(v is not None for v in seen.values()):
                    if shown is not None and not self._changed(behind, shown):
                        self.window_state = "not opened"
                        self.log(f"  her stat window: it didn't open after {self.key.upper()} (nothing changed where "
                                 "it opens): not read, nothing more sent")
                        return sw.StatReading(why=f"it didn't open after {self.key.upper()}")
                    self.log(f"  her stat window: NOT SEEN after {self.key.upper()}, "
                             + ("no frame to tell by" if shown is None else "though the frame changed where it opens")
                             + ": it may be open")
                    return None if shown is None else sw.StatReading(why="not seen, the frame changed where it opens")
            self.log("  her stat window" + (" (open already)" if already else "") + ": "
                     + ", ".join(f"{n} at {tuple(a)}" for n, a in seen.items() if a is not None))
            if seen.get("main") is not None and sw.wants_detail(seen, self.cal):
                if sw.can_click(self.cal):
                    shown = self._detail(seen["main"], shown)
                else:
                    self.log("  her stat window: the DETAIL panel isn't open and its button's place isn't known "
                             "(a live calibration that read none): her W.DEF isn't read")
            reading = self._read(shown)
            if reading is not None and reading.open and reading.wdef is None and reading.detail:
                again = self._read()  # (a second frame: the OCR read no number in W.DEF's cell)
                if again is not None and again.wdef is not None:
                    reading = again
            self._keep(self._last_frame)
            if self._close():
                self.window_state = "closed"
            return reading
        except BaseException:
            if self.window_state == "may be open":  # (the Lie Detector halfway, ...: said before it goes on)
                text = (f"her stat window may still be open in the game: close it with {self.key.upper()} or Esc")
                self.log(f"STAT WINDOW: {text}")
                alarm.page("MapleStory: her stat window", text)
            raise

    def _detail(self, main_at: tuple[int, int], shown):
        """The DETAIL panel opened by a click on its button (the Main panel at `main_at`), the mouse off again after;
        frames DETAIL_S and DETAIL_AGAIN_S until it's seen. The frame to read (the last taken)."""
        x, y = sw.detail_button(main_at, self.cal)
        off = self._mouse_at()
        self.h.actions.play([[0, "mouse", x, y], [80, "lbutton", "down"], [80 + CLICK_HOLD_MS, "lbutton", "up"],
                             [80 + CLICK_HOLD_MS + 60, "mouse", *off]], "stat_window_detail")
        self.sent.append("click DETAIL")
        for wait in (DETAIL_S, DETAIL_AGAIN_S):
            time.sleep(wait)
            frame = self._frame()
            if frame is None:
                return shown
            shown = frame
            if self._seen(frame).get("detail") is not None:
                self.log("  her stat window: DETAIL open (a click on its button)")
                return frame
        self.log("  her stat window: the DETAIL panel not seen after a click on its button (her W.DEF is on it)")
        return shown

    def _close(self) -> bool:
        """Her stats key (the window seen open: it toggles it), frames CLOSE_S and CLOSE_AGAIN_S later; Esc only if
        both still show it open, and so once more. Whether it's seen closed (said)."""
        for go, key in ((1, self.key), (2, "esc"), (3, "esc")):
            self._tap(key, "stat_window_close")
            for wait in (CLOSE_S, CLOSE_AGAIN_S):
                time.sleep(wait)
                frame = self._frame()
                if frame is None:
                    self.log(f"  her stat window: no frame after {key.upper()}: not known whether it closed")
                    return False
                if not any(v is not None for v in self._seen(frame).values()):
                    self.log(f"  her stat window: closed ({key.upper()}"
                             + (", the key before didn't close it" if go > 1 else "") + ")")
                    return True
        self.log(f"  her stat window: STILL OPEN after {self.key.upper()} and Esc twice")
        return False

    def _keep(self, frame) -> None:
        """The window's frame saved (lab/shots/statwindow_<time>.png), SHOTS_MAX a run at the most."""
        if frame is None or self.frames_kept >= SHOTS_MAX:
            return
        try:
            from PIL import Image
            paths.SHOTS.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(frame)).save(paths.SHOTS / f"statwindow_{time.strftime('%Y%m%d-%H%M%S')}.png")
            self.frames_kept += 1
        except Exception:
            pass
