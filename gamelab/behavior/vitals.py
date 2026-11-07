"""Her potions running out and her death, on the phone (control.notify.phone), in a hunt (Hunter._vitals) and between
hunts (behavior.guard): the user, 2026-11-05, after she stood among monsters for 12 minutes and lay dead for 5 h 40 min
with nothing said. Each once: a push when her pet's HP or MP potions stop working (a regular push, as her backpack
full) or she's dead (a critical one), and none again until they've worked, or she's alive, again; the state in a lab file
(paths.VITALS), for the babysitter stops and starts the pet feeder around every hunt.

Told by her bars, not the game's words. The game does say it, "You are lacking the MP Potion that your pet is supposed
to use." on the message bar (perception.messages; frame D:/MapleHelper-scratch/nohit/raw/20261105-081701-no-hit/00104.jpg;
the HP line is only supposed from that one), but any chat line replaces it within seconds and a read is an OCR's 285 ms,
where the bars are in every state, and can be checked over all the live telemetry: her pet drinks within 0.2 s of a touch
(behavior.safety) and a potion is a jump of a bar (Meat +150 of her 1244, an MP potion about +19%; natural regeneration
about 1 HP a second). So:
- HP: HP_DROPS touches' drops under HP_LOW of her bar with no such jump between;
- MP: under MP_LOW for MP_OUT_S with no such jump (a cast's worth regained and spent again keeps her there: 2,926 of 3,138
  states of 2026-11-05 08:11-08:18 at 1-3%, her potions out; with them her MP never stays there: the 0.0 before each
  potion is one frame's);
- dead: HP 0 by her number, or her bar HP_ZERO or less, for DEAD_S (a look's misread 0 lasts a frame: Hunter._doubt_hp).
AGAIN_S keeps a run-out that comes and goes (a potion picked up and drunk at once) from flooding the phone. Replayed on
the 1.22 million live states of 10-28 to 11-05 (D:/MapleHelper-scratch/parkguard, vitals_replay.py): 13 pushes, MP 5, HP 7,
dead 1 (README, A stop that parks)."""

import json
import time
from collections import deque
from collections.abc import Callable

from .. import paths

HP_LOW = 0.6       # of her bar: where her pet drinks (live on 2026-11-05: 949, 902, 855, 739 of 1244, then 899 0.13 s on)
HP_DROPS = 2       # touches' drops under it (each of DROP or more, TOUCH_APART_S apart) with no jump between
DROP = 0.015       # of her bar: a Green Mushroom's 48 of her 1244 is 3.9%
TOUCH_APART_S = 1.0
JUMP = 0.05        # of a bar, over the last WINDOW_S: a potion (its bar moves over a few frames, its number at once)
WINDOW_S = 1.5
MP_LOW = 0.05      # of her bar
MP_OUT_S = 60.0
HP_ZERO = 0.02     # as behavior.hunter's
DEAD_S = 3.0
AGAIN_S = 30 * 60.0  # s before the same potion is told of again, whatever came between


def load() -> dict:
    """What's been told (paths.VITALS: 'hp', 'mp' the time told, None once they've worked again; 'told_hp', 'told_mp' the
    last time told; 'dead'); {} if none's kept."""
    try:
        state = json.loads(paths.VITALS.read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


class Vitals:
    """A state's bars at a time (`see`), the pushes as they fall due. `push(title, text, critical)`; None: nothing goes.
    `clock`: the wall clock, what's kept across processes goes by it; a replay gives its own `t`."""

    def __init__(self, push: Callable[[str, str, bool], None] | None = None, log: Callable[[str], None] | None = None,
                 clock=None):
        self.push, self.log, self.clock = push, log, clock or time.time
        self.state = load()
        self.last: dict[str, deque] = {"hp": deque(), "mp": deque()}  # the last WINDOW_S of each bar: (t, share)
        self.drops, self.drop_at = 0, 0.0
        self.mp_since: float | None = None
        self.zero_since: float | None = None

    def _tell(self, key: str, title: str, text: str, critical: bool, t: float) -> None:
        self.state[key] = t
        self._keep()
        if self.log is not None:
            self.log(f"{title}: {text}")
        if self.push is not None:
            self.push(title, text, critical)

    def _keep(self) -> None:
        try:
            paths.VITALS.write_text(json.dumps(self.state), encoding="utf-8")
        except OSError:
            pass

    def _worked(self, key: str) -> None:
        """Her `key` potions worked: the next run-out is told (AGAIN_S after the last)."""
        if self.state.get(key) is not None:
            self.state[f"told_{key}"], self.state[key] = self.state[key], None
            self._keep()

    def _moved(self, key: str, t: float, share: float) -> tuple[bool, float]:
        """Whether bar `key` jumped over WINDOW_S (a potion: its window starts afresh), and how far it fell in it."""
        last = self.last[key]
        while last and t - last[0][0] > WINDOW_S:
            last.popleft()
        jumped = bool(last) and share - min(v for _, v in last) >= JUMP
        fell = max((v for _, v in last), default=share) - share
        if jumped:
            last.clear()
        last.append((t, share))
        return jumped, fell

    def _may_tell(self, key: str, t: float) -> bool:
        told = self.state.get(f"told_{key}")
        return self.state.get(key) is None and (told is None or t - told >= AGAIN_S)

    def see(self, hp: float | None, mp: float | None, number: tuple | None = None, t: float | None = None) -> None:
        """A state's HP bar (a share; None: not read) and MP bar, her HP number (cur, max; None: not read) at `t`."""
        t = self.clock() if t is None else t
        if hp is not None or number is not None:
            share = number[0] / number[1] if number is not None and number[1] else hp
            self._see_hp(share, number is not None and number[0] == 0 or share <= HP_ZERO, t)
        if mp is not None:
            self._see_mp(mp, t)

    def _see_hp(self, share: float, zero: bool, t: float) -> None:
        if zero:
            self.zero_since = t if self.zero_since is None else self.zero_since
            self.last["hp"].clear()
            self.drops = 0
            if t - self.zero_since >= DEAD_S and self.state.get("dead") is None:
                self._tell("dead", "She died", f"her HP is 0 at {time.strftime('%H:%M:%S', time.localtime(t))}: "
                                               "nothing is sent until she's revived", True, t)
            return
        self.zero_since = None
        if self.state.get("dead") is not None:
            self.state["dead"] = None
            self._keep()
        jumped, fell = self._moved("hp", t, share)
        if share >= HP_LOW:
            self.drops = 0
        if jumped:
            self.drops = 0
            self._worked("hp")
        elif fell >= DROP and share < HP_LOW and t - self.drop_at >= TOUCH_APART_S:
            self.drops, self.drop_at = self.drops + 1, t
            self.last["hp"].clear()  # (the next from here: this one's fall stays in the window for WINDOW_S)
            self.last["hp"].append((t, share))
        if self.drops >= HP_DROPS and share < HP_LOW and self._may_tell("hp", t):
            self._tell("hp", "HP potions not working", f"{self.drops} touches with no potion, HP {share:.0%}: her pet's "
                                                       "HP potions out, or it's hungry", False, t)

    def _see_mp(self, share: float, t: float) -> None:
        if self._moved("mp", t, share)[0]:
            self.mp_since = None
            self._worked("mp")
        if share >= MP_LOW:
            self.mp_since = None
            return
        self.mp_since = t if self.mp_since is None else self.mp_since
        if t - self.mp_since >= MP_OUT_S and self._may_tell("mp", t):
            self._tell("mp", "MP potions not working", f"MP under {MP_LOW:.0%} for {t - self.mp_since:.0f} s with no "
                                                       "potion: her pet's MP potions out, or it's hungry", False, t)

    @property
    def zero(self) -> bool:
        """Her HP was 0 at the last state."""
        return self.zero_since is not None
