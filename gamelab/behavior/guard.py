"""While no hunt runs she isn't left to die, and a hunt can be stopped so she parks first. 2026-11-05: a grind killed at
11:15:35 ("Stop now, i need to work") left her among 7-9 monsters; 133 touches, her pet's potions out at 11:26, dead at
11:27:48 and nothing said for 5 h 40 min. The user, on being offered a stop that parks and a guard that parks her
between hunts: "you should build both, but the platform picking should be dynamically chosen based on the current map".

`python -m gamelab stop` (`stop`): the way to stop a hunt. It asks the running hunt to stop (paths.STOP_HUNT, read at
its next look or glance: Hunter._stop_now, behavior.safety.StopAsked, so no plan after it but her moves), cuts the plan
playing (client.cut_running: a throw's key not pressed yet isn't), sets the grind
loop's STOP too, and waits to say what came of it: she goes to the chosen place (behavior.retreat.park_place) as on any
stop, with no page. The game not in front: the request stands, nothing brings it forward, and she goes when it's back;
after behavior.hunter.STOP_WAIT_S the hunt ends with a note (paths.PARK_NOTE) for the guard, so the babysitter's pet feeder comes back.
With no hunt running it parks her (`nav --park`, for the babysitter's regex to hold the pet feeder meanwhile). A hunt that
ends on its own time parks too (`tour --loop`); one killed or crashed leaves its entry behind (paths.HUNT, `over` removes
a clean end's) and the guard's next pass takes that for a note.

The guard (`Guard`, in `pet --every`): the babysitter keeps that running exactly while no hunt runs, it's attached to the
perception service and it knows the map, so no new process, and nothing but the service's newest state each GUARD_S (no
frame work, no keys of its own). What sets it off, from perception alone (never from where she seems to be): she's touched
TOUCHES times in TOUCH_WITHIN_S (her popup: perception.touches; not one `crowded` by another player on her), her HP under the user's stop rule on
LOW_PASSES passes (behavior.safety's: her profile's hits of the map's largest touch) unless she's known parked (a park or a
stop's retreat told she's there, none hunting or playing her since: `Pass._parked`), or a note (a stop's, or a hunt's
that died). Then, in order:
- dead (behavior.vitals: one critical push, none repeated): nothing, until she's alive;
- the Lie Detector alarm up: no moves, as ever;
- played by hand (Pass._hands): she's left alone with nothing said; under the stop rule one alert, still no moves;
- the game in front and the player up: `nav --park --tell`, detached, one push saying so, and again at most every
  PARK_AGAIN_S and PARK_MOST times an episode: it's that command that says where she ended up;
- else one push saying why not, and she parks at the game's return (it never brings it forward, nor starts the player),
  however long that takes: an episode with a park still wanted doesn't end. The game just back she waits BACK_S twice over.
A push is critical but for a stop's note alone (the user asked for that one). An episode ends after EPISODE_S with nothing
setting it off. The state is in a lab file (paths.GUARD): the babysitter stops the pet feeder as soon as the park starts.

Blind: with the game behind another window or minimized, or the service looking at another map (the babysitter's `pet
--map`, `service._hello`), or none, there's no usable state: touches, her HP and her potions can't be seen. After BLIND_S
of that, one regular push if she isn't known parked; it re-arms when states are back.

The switch: paths.GUARD_MODE, read each pass (`python -m gamelab guard off`, `alert`, `on`): off, nothing; alert only, the
push and no moves. Her potions and her death are told in any mode but off (behavior.vitals).

Input: Windows' last input (window.last_input_age) moves with the player's injected keys too (measured live: at 0.09 s
at the median, 500 updates a minute, in a hunt on 2026-11-05), so what counts is input later than the player's last key
(its log's last line, `player_last_key`), looked at each pass and remembered for HAND_S (the pet's food tap moves it too,
and would hide a key pressed before it), with the game in front, and more than BACK_S after the game came back (the click
or Alt+Tab that brought it isn't playing her). Chrome Remote Desktop's host injects its keys the same way from the console
session (remoting_desktop.exe there, session 1, as the game and the player; read off the process list, not tried: the
README's live check does)."""

import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .. import paths, telemetry, window
from ..control import client
from .vitals import Vitals

GUARD_S = 1.0        # s between passes
STATE_AGE_S = 3.0    # a state older than this says nothing
LOW_PASSES = 2       # passes in a row under the stop rule (a bar misread for a frame isn't a stop rule)
TOUCHES = 2          # touches within TOUCH_WITHIN_S: she's being touched (one on her touches every 1.5 s or so; a lone popup
TOUCH_WITHIN_S = 30.0  # is mostly another player's: 4 MISSes in 5.6 h on platform 27, where nothing walks)
HAND_S = 60.0        # s since real input that has her played by hand
BACK_S = 5.0         # s after the game comes back in which input is what brought it (a park's held twice this, for whoever
                     # has come back to play to show it)
BLIND_S = 300.0      # s with no usable state before it's said
STALE_S = 60.0       # s after which a stop's request is left over from another hunt (a hunt's start: `claim`)
HUNTS = ("hunt", "probe", "nav", "tour")  # the commands `_record` says hunt (`claim`)
KEY_SLACK_S = 1.5    # s past the player's log's last line (its stamps are whole s, written as a plan ends) that's not its keys
LOG_TAIL = 2048      # bytes of the player's log read for it: only once something's set the guard off
PARK_AGAIN_S = 120.0 # s between parks an episode, and ...
PARK_MOST = 3        # ... parks at most: a park that fails doesn't loop
EPISODE_S = 300.0    # s with nothing setting it off that end an episode: what comes next is told afresh
STOP_SAY_S = 300.0   # s `stop` waits to say what came of it
OFF, ALERT, ON = "off", "alert", "on"
RAN = re.compile(rb"^(\d\d):(\d\d):(\d\d) (?:ran|spliced) ", re.M)
LOOP_STOP = Path("D:/MapleHelper-scratch/mageq/empty/STOP")  # the grind loops' (grind_loop.py, grind_loop_thief.py): read
                                                             # between grinds, so the loop starts no other


def read_mode() -> str:
    """The switch as it is now (paths.GUARD_MODE): on unless it says otherwise."""
    try:
        mode = paths.GUARD_MODE.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return ON
    return mode if mode in (OFF, ALERT) else ON


def player_last_key(now: float, log: Path | None = None) -> float | None:
    """When the player last played a plan (wall clock), by its log's last 'ran' or 'spliced' line (time-only stamps: the
    newest such time not past `now`); None if its tail has none."""
    try:
        with open(log or paths.PLAYER_LOG, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - LOG_TAIL))
            found = RAN.findall(f.read())
    except OSError:
        return None
    if not found:
        return None
    h, m, s = (int(v) for v in found[-1])
    at = datetime.fromtimestamp(now).replace(hour=h, minute=m, second=s, microsecond=0)
    return (at if at.timestamp() <= now + 1 else at - timedelta(days=1)).timestamp()


@dataclass
class Seen:
    """What one pass has to go by."""
    touched: bool = False
    low: bool = False    # her HP under the stop rule
    note: str = ""       # a note: she was to park, why
    urgent: bool = False  # ... nobody having asked for it (a hunt that died)
    dead: bool = False
    alarm: bool = False
    front: bool = True   # the game in front
    held: bool = False   # ... only just back (Pass._hands)
    player: bool = True  # the player up
    by_hand: bool = False
    hp: str = ""         # her HP in words


class Guard:
    """The guard's decisions (`act`). `push(title, text, critical)` for the phone, `start_park()` for `nav --park`."""

    def __init__(self, push: Callable[[str, str, bool], None], start_park: Callable[[], None],
                 log: Callable[[str], None] = print, clock=None):
        self.push, self.start_park, self.log, self.clock = push, start_park, log, clock or time.time
        try:
            self.state = dict(json.loads(paths.GUARD.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            self.state = {}

    def _keep(self) -> None:
        try:
            paths.GUARD.write_text(json.dumps(self.state), encoding="utf-8")
        except OSError:
            pass

    def _over(self) -> None:
        if self.state.get("episode") is not None:
            self.state = {"park_at": self.state.get("park_at", 0.0)}
            self._keep()

    def say(self, title: str, text: str, critical: bool) -> None:
        """A push, logged."""
        self.log(f"guard: {title}: {text}")
        telemetry.emit("event", kind="guard", text=f"{title}: {text}")
        self.push(title, text, critical)

    def _alert(self, key: str, title: str, text: str, critical: bool = True) -> bool:
        """A push, once an episode by `key`; whether it went."""
        if key in self.state.get("told", ()):
            return False
        self.state.setdefault("told", []).append(key)
        self._keep()
        self.say(title, text, critical)
        return True

    def act(self, seen: Seen, mode: str = ON) -> str | None:
        """One pass's decision (see the top): 'park' if it started one, 'alert' on alert only's push, else None."""
        now, st = self.clock(), self.state
        if mode == OFF:
            return None
        if seen.dead:
            paths.PARK_NOTE.unlink(missing_ok=True)
        if seen.dead or (st.get("episode") is not None and not st.get("pending") and now - st["last"] >= EPISODE_S):
            self._over()
            st = self.state
            if seen.dead:
                return None
        if seen.touched or seen.low or seen.note:
            if st.get("episode") is None:
                st.update(episode=now, told=[], parks=0)
            was = st.get("pending")
            st.update(pending=True, last=now)
            if not was:
                self._keep()
        elif st.get("episode") is None:
            return None
        if not st.get("pending") or seen.alarm:
            return None
        why = "her HP is under the stop rule" if seen.low else seen.note or "she's being touched"
        why = f"{why}{f' ({seen.hp})' if seen.hp else ''}, no hunt running"
        critical = seen.touched or seen.low or seen.urgent or not seen.note  # (a stop's note alone: the user asked)
        if seen.by_hand:
            if seen.low:
                self._alert("low", "HP low", f"{why}; you're playing her: no moves")
            st["pending"] = False  # (left alone: the next touch, once nobody's playing, sets it off again)
            paths.PARK_NOTE.unlink(missing_ok=True)  # (she's theirs now)
            return None
        if mode == ALERT:
            st["pending"] = False
            return "alert" if self._alert("alert", "She needs you", f"{why}; the guard is on alert only: no moves") else None
        if not seen.front or not seen.player:
            whose = "the game isn't in front" if not seen.front else "the player isn't running"
            self._alert("wait", "She can't park yet", f"{why}; {whose}: she parks once it's put right", critical)
            return None
        if seen.held or now - st.get("park_at", 0.0) < PARK_AGAIN_S:
            return None
        if st.get("parks", 0) >= PARK_MOST:
            self._alert("failed", "She can't park", f"{why}; {PARK_MOST} parks haven't ended it: she's yours to see to")
            st["pending"] = False
            paths.PARK_NOTE.unlink(missing_ok=True)
            return None
        st.update(park_at=now, parks=st.get("parks", 0) + 1, pending=False)
        self._alert("parking", "Parking her", f"{why}; she's going to a safe place", critical)
        self._keep()
        self.start_park()
        return "park"


class Pass:
    """A pass of the guard in `pet --every` (`tick`): `latest()` the perception service's newest state to act on (None:
    none), `low(st)` whether it's under the stop rule, `title` the game's window."""

    def __init__(self, guard: Guard, latest: Callable, low: Callable, title: str, vitals: Vitals):
        self.guard, self.latest, self.low, self.title, self.vitals = guard, latest, low, title, vitals
        self.told: float | None = None  # the newest touch as of the last pass: none set it off from before the guard
        self.touches: deque[float] = deque(maxlen=TOUCHES)
        self.lows = 0
        self.front: bool | None = None  # the game in front at the last pass ...
        self.front_at = 0.0             # ... since when, if it's come back since the pass began
        self.input_at = 0.0             # Windows' last input as last seen ...
        self.played_at = 0.0            # ... and the last that was someone playing her
        self.blind_at: float | None = None  # no usable state since then
        self.blind_told = False

    def _hands(self, now: float) -> None:
        """Each pass, whatever else is going on: when the game came back, and the last input that was someone playing her
        (see the top)."""
        front = window.foreground_title() == self.title
        if front and self.front is False:
            self.front_at = now
        self.front = front
        age = window.last_input_age()
        if age is None or now - age <= self.input_at + 0.05:  # (GetTickCount's 16 ms steps: no new input)
            return
        self.input_at = now - age
        if front and self.input_at >= self.front_at + BACK_S:
            key = player_last_key(now)
            if key is None or self.input_at > key + KEY_SLACK_S:
                self.played_at = self.input_at

    def _parked(self) -> bool:
        """A park or a stop's retreat told she's there (paths.PARKED, gone at a hunt's start: `claim`), and nobody's played
        her since."""
        done = parked_said(self.played_at)
        return done is not None and done[0]

    def _blind(self, now: float) -> None:
        """No usable state (see the top)."""
        self.blind_at = now if self.blind_at is None else self.blind_at
        if self.blind_told or now - self.blind_at < BLIND_S or client.alarm_up() or running_hunt() is not None \
                or self._parked():
            return
        self.blind_told = True
        away = "the game isn't in front" if not self.front else "no usable state from the perception service"
        self.guard.say("The guard can't see her", f"{away} for {BLIND_S / 60:g} minutes: nothing tells if she's touched, "
                                                  "and she isn't known to be on a safe place", False)

    def tick(self) -> str | None:
        mode = read_mode()
        if mode == OFF:
            return None
        now = self.guard.clock()
        self._hands(now)
        left_behind()
        note = read_note()
        seen, st = Seen(note=note.get("why", ""), urgent=bool(note.get("urgent"))), self.latest()
        if st is None:
            self._blind(now)
        else:
            self.blind_at, self.blind_told = None, False
            self.vitals.see(st.hp, st.mp, st.hp_of)
            touch = st.touch
            if touch is not None and not touch.get("crowded"):
                if self.told is not None and touch["told"] > self.told:
                    self.touches.append(touch["told"])
                    seen.touched = len(self.touches) == TOUCHES and self.touches[-1] - self.touches[0] <= TOUCH_WITHIN_S
                self.told = touch["told"]
            elif self.told is None:
                self.told = 0.0
            self.lows = self.lows + 1 if self.low(st) else 0
            seen.low = self.lows >= LOW_PASSES and not self._parked()
            seen.hp = f"HP {st.hp_of[0]} of {st.hp_of[1]}" if st.hp_of is not None else f"HP {st.hp:.0%}"
        seen.dead = self.vitals.zero  # (from the pass that reads HP 0: told once it's lasted)
        if seen.touched or seen.low or seen.note or self.guard.state.get("pending"):
            if running_hunt() is not None:  # (the babysitter's 5 s to stop the pet feeder, or no babysitter)
                return None
            seen.alarm, seen.player = client.alarm_up(), client.player_up()
            seen.front, seen.held = self.front, self.front and now < self.front_at + 2 * BACK_S
            seen.by_hand = self.front and now - self.played_at <= HAND_S
        return self.guard.act(seen, mode)


def _entry() -> dict | None:
    """The command hunting now, or last, as it said at its start (paths.HUNT: {'pid', 'started', 'command'}: `claim`)."""
    try:
        hunt = json.loads(paths.HUNT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return hunt if isinstance(hunt, dict) else None


def _running(hunt: dict) -> bool:
    """Whether that command's process is still running: its pid, and the same process (a pid gets given again)."""
    pid, started = hunt.get("pid"), hunt.get("started")
    return client._alive(pid) and (started is None or started == client.started(pid))


def running_hunt() -> dict | None:
    """The command hunting now, if one is."""
    hunt = _entry()
    return hunt if hunt is not None and _running(hunt) else None


def claim(command: str) -> None:
    """`command`, this process, is hunting (gamelab.__main__._record, if it's one of HUNTS), for `stop` and the guard. One
    already running keeps its entry, but for a park: a hunt ends that (retreat._taken_over). What a guard and a stop
    left is over; but a stop's request just made stands, for this hunt's first look: the one it asked may have ended
    just then, and the grind loop's next begun."""
    now = running_hunt()
    if now is not None and not now.get("command", "").endswith("park"):
        return
    pid = os.getpid()
    paths.HUNT.write_text(json.dumps({"pid": pid, "started": client.started(pid), "command": command}), encoding="utf-8")
    try:
        fresh = time.time() - json.loads(paths.STOP_HUNT.read_text(encoding="utf-8"))["t"] <= STALE_S
    except (OSError, ValueError, KeyError, TypeError):
        fresh = False
    if not fresh:
        paths.STOP_HUNT.unlink(missing_ok=True)
    if not command.endswith("park"):
        for path in (paths.PARK_NOTE, paths.PARKED, paths.GUARD):
            path.unlink(missing_ok=True)


def over() -> None:
    """This command ended as it should (gamelab.__main__.main): a hunt's entry stays only if it was killed or crashed."""
    hunt = _entry()
    if hunt is not None and hunt.get("pid") == os.getpid():
        paths.HUNT.unlink(missing_ok=True)


def left_behind() -> None:
    """A hunt's entry with its process gone (`over`): nothing parked her, so a note, as a stop's, and at once: with the game
    behind there's no touch to see (the 11:15 of the top)."""
    hunt = _entry()
    if hunt is not None and not _running(hunt):
        paths.HUNT.unlink(missing_ok=True)
        paths.PARK_NOTE.write_text(json.dumps({"t": time.time(), "urgent": True, "why": f"`{hunt.get('command')}` ended "
                                               "without parking her"}), encoding="utf-8")


def read_note() -> dict:
    """The note (paths.PARK_NOTE: {'t', 'why', 'urgent'}): `retreat.told`'s, `stop`'s, `left_behind`'s; {} if none."""
    try:
        note = json.loads(paths.PARK_NOTE.read_text(encoding="utf-8"))
    except OSError:
        return {}
    except ValueError:
        note = {}
    return {"why": "a stop left a note", **note} if isinstance(note, dict) else {"why": "a stop left a note"}


def later() -> str:
    """What comes of her once the game's back, as `stop` and `nav --park` say it."""
    return "the guard is off: nothing parks her then" if read_mode() == OFF else "she parks once it's back (the guard, `pet --every`)"


def parked_said(since: float) -> tuple[bool, str] | None:
    """How the last park or stop's retreat came out since wall time `since` (paths.PARKED: behavior.retreat.told), as
    (she's there, in words); None if none has."""
    try:
        row = json.loads(paths.PARKED.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return (row["there"], row["said"]) if row.get("t", 0.0) >= since else None


def stop(title: str, say: Callable[[str], None] = print, park: Callable[[], int] | None = None, sleep=time.sleep) -> int:
    """`python -m gamelab stop` (see the top). 0 once the stop is done or standing, said; 1 if it can't tell."""
    start = time.time()
    if LOOP_STOP.parent.exists():
        LOOP_STOP.touch()
    hunt = running_hunt()
    if hunt is None:
        if client.alarm_up():
            say("no hunt is running, and with the Lie Detector alarm up nothing moves her")
            return 0
        if window.foreground_title() != title:
            paths.PARK_NOTE.write_text(json.dumps({"t": start, "why": "`stop` with no hunt"}), encoding="utf-8")
            say(f"no hunt is running; the game isn't in front: {later()}")
            return 0
        (park or _park_now)()
        hunt = running_hunt()  # (a grind the loop began meanwhile: it reads STOP only between grinds)
    if hunt is not None:
        if not hunt.get("command", "").endswith("park"):
            paths.STOP_HUNT.write_text(json.dumps({"t": start}), encoding="utf-8")
            if not client.alarm_up():  # (with it up the plans there are her retreat's, and the Lie Detector's cursor and OK)
                client.cut_running()
        say(f"asked `{hunt.get('command')}` (pid {hunt.get('pid')}) to stop")
        while _running(hunt) and time.time() - start < STOP_SAY_S:
            sleep(0.2)
    done = parked_said(start)
    if done is not None:
        say(done[1])
        return 0 if done[0] else 1
    if hunt is not None and _running(hunt):
        say(f"`{hunt.get('command')}` hasn't stopped in {STOP_SAY_S:g} s (the game not in front?): the request stands")
        return 1
    if hunt is not None:
        paths.STOP_HUNT.unlink(missing_ok=True)
    paths.PARK_NOTE.write_text(json.dumps({"t": start, "why": "`stop`: nothing said of a park"}), encoding="utf-8")
    say(f"it ended without saying where she is: {later()}")
    return 1


def _park_now() -> int:
    """`nav --park`, its own process for the babysitter's regex to see, waited for."""
    return subprocess.run([sys.executable, "-X", "utf8", "-m", "gamelab", "nav", "--park"], cwd=paths.REPO).returncode
