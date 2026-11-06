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
With no hunt running it parks her (`nav --park`, for the babysitter's regex to hold the pet feeder meanwhile).

The guard (`Guard`, in `pet --every`): the babysitter keeps that running exactly while no hunt runs, it's attached to the
perception service and it knows the map, so no new process, and nothing but the service's newest state each GUARD_S (no
frame work, no keys of its own). What sets it off, from perception alone (never from where she seems to be): she's touched
TOUCHES times in TOUCH_WITHIN_S (her popup: perception.touches; not one `crowded` by another player on her), her HP under the user's stop rule on
LOW_PASSES passes (behavior.safety's: her profile's hits of the map's largest touch), or a stop's note. Then, in order:
- dead (behavior.vitals: one critical push, none repeated): nothing, until she's alive;
- the Lie Detector alarm up: no moves, as ever;
- played by hand (`played_by_hand`: the game in front and input within HAND_S that isn't the player's own keys): she's
  left alone with nothing said; under the stop rule one alert, still no moves;
- the game in front and the player up: `nav --park --tell`, detached, one critical push saying so, and again at most
  every PARK_AGAIN_S and PARK_MOST times an episode: it's that command that says where she ended up;
- else one critical push saying why not, and she parks at the game's return (it never brings it forward, nor starts the
  player).
An episode ends after EPISODE_S with nothing setting it off. The state is in a lab file (paths.GUARD): the babysitter stops
the pet feeder as soon as the park starts.

The switch: paths.GUARD_MODE, read each pass (`python -m gamelab guard off`, `alert`, `on`): off, nothing; alert only, the
push and no moves. Her potions and her death are told in any mode but off (behavior.vitals).

Input: Windows' last input (window.last_input_age) moves with the player's injected keys too (measured live: at 0.09 s
at the median, 500 updates a minute, in a hunt on 2026-11-05), so what counts is input later than the player's last key
(its log's last line, `player_last_key`). Chrome Remote Desktop's host injects its keys the same way from the console
session (remoting_desktop.exe there, session 1, as the game and the player; read off the process list, not tried: the
README's live check does)."""

import json
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


def played_by_hand(now: float, front: bool, input_age: float | None, last_key: float | None) -> bool:
    """Whether someone's playing her by hand: the game in front, and input within HAND_S that's later than the player's
    last key (KEY_SLACK_S past its stamp). Input not known: no."""
    if not front or input_age is None or input_age > HAND_S:
        return False
    return last_key is None or now - input_age > last_key + KEY_SLACK_S


@dataclass
class Seen:
    """What one pass has to go by."""
    touched: bool = False
    low: bool = False    # her HP under the stop rule
    note: bool = False   # a stop's note: she was to park
    dead: bool = False
    alarm: bool = False
    front: bool = True   # the game in front
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

    def _alert(self, key: str, title: str, text: str) -> bool:
        """A critical push, once an episode by `key`; whether it went."""
        if key in self.state.get("told", ()):
            return False
        self.state.setdefault("told", []).append(key)
        self._keep()
        self.log(f"guard: {title}: {text}")
        telemetry.emit("event", kind="guard", text=f"{title}: {text}")
        self.push(title, text, True)
        return True

    def act(self, seen: Seen, mode: str = ON) -> str | None:
        """One pass's decision (see the top): 'park' if it started one, 'alert' on alert only's push, else None."""
        now, st = self.clock(), self.state
        if mode == OFF:
            return None
        if seen.dead or (st.get("episode") is not None and now - st["last"] >= EPISODE_S):
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
        why = "her HP is under the stop rule" if seen.low else "a stop asked her to park" if seen.note else "she's being touched"
        why = f"{why}{f' ({seen.hp})' if seen.hp else ''}, no hunt running"
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
            self._alert("wait", "She can't park yet", f"{why}; {whose}: she parks once it's put right")
            return None
        if st.get("parks", 0) >= PARK_MOST:
            self._alert("failed", "She can't park", f"{why}; {PARK_MOST} parks haven't ended it: she's yours to see to")
            st["pending"] = False
            paths.PARK_NOTE.unlink(missing_ok=True)
            return None
        if now - st.get("park_at", 0.0) < PARK_AGAIN_S:
            return None
        st.update(park_at=now, parks=st.get("parks", 0) + 1, pending=False)
        self._alert("parking", "Parking her", f"{why}; she's going to a safe place")
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

    def tick(self) -> str | None:
        mode = read_mode()
        if mode == OFF:
            return None
        seen, st = Seen(note=paths.PARK_NOTE.exists()), self.latest()
        if st is not None:
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
            seen.low = self.lows >= LOW_PASSES
            seen.hp = f"HP {st.hp_of[0]} of {st.hp_of[1]}" if st.hp_of is not None else f"HP {st.hp:.0%}"
        seen.dead = self.vitals.zero  # (from the pass that reads HP 0: told once it's lasted)
        if seen.touched or seen.low or seen.note or self.guard.state.get("pending"):
            if running_hunt() is not None:  # (the babysitter's 5 s to stop the pet feeder, or no babysitter)
                return None
            now = self.guard.clock()
            seen.alarm, seen.player = client.alarm_up(), client.player_up()
            seen.front = window.foreground_title() == self.title
            seen.by_hand = played_by_hand(now, seen.front, window.last_input_age(), player_last_key(now))
        return self.guard.act(seen, mode)


def running_hunt() -> dict | None:
    """The command hunting now, as it said at its start (paths.HUNT: {'pid', 'command'}), if it's still running."""
    try:
        hunt = json.loads(paths.HUNT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return hunt if isinstance(hunt, dict) and client._alive(hunt.get("pid")) else None


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
            say("no hunt is running; the game isn't in front: she parks once it's back (the guard, `pet --every`)")
            return 0
        (park or _park_now)()
    else:
        if not hunt.get("command", "").endswith("park"):
            paths.STOP_HUNT.write_text(json.dumps({"t": start}), encoding="utf-8")
            client.cut_running()
        say(f"asked `{hunt.get('command')}` (pid {hunt.get('pid')}) to stop")
        while client._alive(hunt.get("pid")) and time.time() - start < STOP_SAY_S:
            sleep(0.2)
    done = parked_said(start)
    if done is not None:
        say(done[1])
        return 0 if done[0] else 1
    if hunt is not None and client._alive(hunt.get("pid")):
        say(f"`{hunt.get('command')}` hasn't stopped in {STOP_SAY_S:g} s (the game not in front?): the request stands")
        return 1
    if hunt is not None:
        paths.STOP_HUNT.unlink(missing_ok=True)
    paths.PARK_NOTE.write_text(json.dumps({"t": start, "why": "`stop`: nothing said of a park"}), encoding="utf-8")
    say("it ended without saying where she is: the guard parks her if the game's there")
    return 1


def _park_now() -> int:
    """`nav --park`, its own process for the babysitter's regex to see, waited for."""
    return subprocess.run([sys.executable, "-X", "utf8", "-m", "gamelab", "nav", "--park"], cwd=paths.REPO).returncode
