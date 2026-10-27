"""Out of darts: her Lucky Seven presses taking no MP and killing nothing, 20 in a row on Tree Dungeon (twice and more
the most the live hunts with her darts had), end the hunt as the safety stop does (behavior.safety.HuntStop: she goes to
a safe place, the log says why in capitals, whoever is at the PC is paged, the command exits with its own code), and
the rolling recording's last minutes are saved. The user, 2026-10-26 01:28, during the grind of 01:04: "stop, you run
out of darts"; the bot kept pressing for nothing.

What the game showed then (the grind's log, D:/MapleHelper-scratch/live/grind-0104.log, its telemetry and the perception
service's): her last cast that took MP was keyed at 01:28:16.80 (her bar 37 -> 35 px of 104 at 17.11); every press after
it took none, 23 of them over 20 s on platform 24 at a Horny Mushroom 100-150 off whose bar read 0.26 throughout, until
she went for her drops at 01:28:38 and the user stopped the hunt. No frame of it was kept (the rolling recording is
saved only when asked), so what the game says, if anything, isn't known; nor was her dart count ever on screen (it's in
her inventory's Use tab: reading it means opening the window, which no hunt does).

Her presses take no MP with her darts too: 10-15% of them live (the A+B analysis), mostly one or two in a row, and 5-9
in a row (counted as here) 14 times in the 79 minutes of the five live hunts of 2026-10-25 23:43 to 2026-10-26 01:28,
never more (D:/MapleHelper-scratch/norest/darts/count.py). Two of those runs are on the frames the Lie Detector
recordings kept (lab/lie_detector/20261026-002509/before, 00:23:19-28, and 20261026-010145/before, 00:58:19-26): a Green
Mushroom (and a Snail) standing on her at platform 24's right end, which the looks didn't find, and every press a swipe
of her claw at it, 18-24 damage or MISS over it, her MP 172-175/623 on the HUD unchanged for 10 s, the green dead after
9 presses on it. So up close the Lucky Seven key swipes as the attack key does, and takes no MP (sim.world still has it
cast there). Of the 14, 9 ended in such a kill (7 greens, 233 HP: 8-9 presses; 2 Horny Mushrooms, wounded before), the
others in her casts taking MP again (the monster gone, or she stepped away) or her MP read under her pet's warning. A
Horny Mushroom standing on her whole would take longer: 403 HP, defence 10.

So a stretch: her Lucky Seven presses (Attacks.on_throw's keys) since her bar last showed a cast's MP (FALL_STEPS under
its highest read since), a drink of her pet's (DRINK_STEPS over it), a read too low to tell by (LOW_STEPS) or a kill of
hers (her EXP told of it: Hunter._gained; a party share names no kill and doesn't count); of its presses, those her bar
has had MP_SHOWS_S to show count, and not those keyed in a gap in the reads (GAP_S: looks failing, the game perhaps not
in front and her keys not reaching it; a trip between fights, no press in it, goes on with the stretch). Every look's
bar feeds it (Attacks.on_look and the glance before a key), whoever took the look. It stops the hunt
(Hunter.health_check, raising OutOfDarts) once a stretch has `presses_for` the map: PRESSES on Tree Dungeon, twice and
more the 9 the hunts with her darts had at the most (11 with the perception service's 10 reads a second merged in, the
review's count); at 01:28 the 20th came at 01:28:36.4, 19.6 s after her last cast's key and 10 s before the user stopped
the hunt (tests/test_darts.py replays the five hunts: no stop with her darts, at 20 nor at 12). Until the review it
was 1.5 times a Horny Mushroom standing on her whole (32), which the 79 minutes never had, and at 01:28 it would not
have come before the user's stop (22 presses when she stopped pressing). A whole Horny Mushroom standing on her
unseen, from its full 403 HP (21.1 presses of her swipes, `swipes`), would now stop a hunt with her darts: never seen
live (the two the stretches had were wounded). On a map whose toughest monster takes more swipes than a Horny
Mushroom, PRESSES that many times more (West Rocky Mountain II's Axe Stump: 22). Out of darts in the simulator
(loop_tree_out_of_darts) it stops 18.7 s after her last star (on seeds 2 and 3, 19 and 27 s).

Not watched: the regular attack's presses (`--attack regular`, and `probe run`, which throws the regular attack
always): they take no MP with her darts too, so nothing tells. With the attack choice its regular presses (to finish a
monster whose bar or her throws make low, or her MP under a cast) aren't counted: a stretch stands still while she
throws them. And a read under LOW_STEPS starts it again at every look: out of darts with her MP that low and her pet's
MP potions gone too, it waits for her regen to pass LOW_STEPS (a step every 18 s or so: up to 90 s).

The game's own word, if it has one: a stretch READ_EVERY presses long (and each as many more) has the pickup log read
(Eyes.messages: OCR of where the game says what she got, and that her inventory is full), and if it says she has no
stars left (perception.loot.says_no_darts: the usual wordings, as the game's isn't known), the stop comes then. On the
pickup log of the live frames kept (1,413 frames a second apart) the wordings matched nothing. The stop's log line has
what the log said then, and the recording's last KEEP_S s are saved (control.alarm.keep_recent:
lab/recent/<time>-out-of-darts), so the next time what the game shows can be seen."""

import math
from dataclasses import dataclass, field

from ..attack.choice import MP_STEPS
from ..perception.loot import says_no_darts
from .safety import HuntStop

EXIT = 5               # the command's exit code after it (the safety stop's is 4, the Lie Detector's 3)
MP_FIRST_S = 0.21      # s after a cast's key that its MP may first show on her bar (behavior.attacks.MP_FIRST_S) ...
MP_SHOWS_S = 0.55      # ... and by when it has (attacks.MP_SHOWS_S)
FALL_STEPS = 2         # her bar this many steps (of MP_STEPS) under its highest read since a stretch began: a cast's
                       # MP (14 MP, 2.3 steps: live 2-3, 1 twice in 76)
DRINK_STEPS = 4        # ... this many over it: a drink (her pet's MP potion: 16-25 steps), which could hide casts
                       # (her regen: a step every 18 s or so)
LOW_STEPS = 5          # a read under this (her pet's MP warning, 5%: two casts) can't tell: the stretch ends
GAP_S = 2.0            # no read for this long (looks failing): the presses keyed in it aren't counted
PRESSES = 20           # a stretch this long stops the hunt on Tree Dungeon: twice and more the 9 in a row the live
                       # hunts with her darts had at the most (11 with the perception service's reads in), and the
                       # 20th came 19.6 s after her last cast's key at 01:28, 10 s before the user stopped the hunt
PRESS_DAMAGE = 21.0    # a press's damage at a monster standing on her, her claw's swipes, misses in: a Green
                       # Mushroom's 233 HP (defence 0) in 9-11 presses live (2026-10-26 00:12:19, 00:23:19, 01:28:04)
HORNY = (403.0, 10.0)  # Tree Dungeon's toughest monster, a Horny Mushroom (HP, PDDamage): 21.1 presses of her swipes;
                       # a map whose toughest takes more has PRESSES that many times more
READ_EVERY = 12        # a stretch this many presses long, and each as many more: the pickup log read
KEEP_S = 90.0          # s of the rolling recording saved after the stop (control.alarm.keep_recent)
REPORT_FROM = 4        # a hunt's longest stretch this long or longer is said at its end (DartWatch.summary)


def swipes(hp: float, pdd: float = 0.0) -> float:
    """The presses of her claw's swipes a monster of this HP and PDDamage standing on her takes to die (PRESS_DAMAGE a
    press, less its defence as the game takes it off, x 100 / (PDDamage + 100))."""
    return hp / (PRESS_DAMAGE * 100.0 / (pdd + 100.0))


def presses_for(mob_hp: dict[str, float], mob_pdd: dict[str, float]) -> int:
    """The presses of a stretch that end the hunt on a map with these monsters (their HP and PDDamage by name):
    PRESSES, as on Tree Dungeon, and as many times more as the toughest of them standing on her takes more of her
    claw's swipes than a Horny Mushroom does (West Rocky Mountain II's Axe Stump: 23.0, 22 presses)."""
    most = max((swipes(hp, mob_pdd.get(name, 0.0)) for name, hp in mob_hp.items() if hp > 0), default=0.0)
    return max(PRESSES, math.ceil(PRESSES * most / swipes(*HORNY) - 1e-9))


class OutOfDarts(HuntStop):
    """Her Lucky Seven presses took no MP and killed nothing over a stretch long enough (DartWatch): out of darts.
    `presses`, `seconds`: the stretch's; `mp`: her bar's share then; `said`: the pickup log's text then (None: not
    read), and `told`: whether it said so (says_no_darts); `at`: the stop's time (time.monotonic)."""
    label = "OUT OF DARTS"
    title = "Out of darts"
    event = "out_of_darts"
    exit_code = EXIT
    keep_s = KEEP_S
    ends = "to a safe place (no more throws)"
    advice = "Give her throwing stars (her Use tab) before hunting again."

    def __init__(self, presses: int, seconds: float, mp: float, said: str | None = None, told: bool = False,
                 at: float = 0.0):
        self.presses, self.seconds, self.mp, self.said, self.told, self.at = presses, seconds, mp, said, told, at
        super().__init__(self._text())

    def _text(self) -> str:
        why = (f"{self.presses} Lucky Seven presses over {self.seconds:.0f} s took no MP and killed nothing "
               f"(her MP {self.mp:.0%})")
        if self.told:
            return f"the game says she's out of throwing stars ({self.lines()}); {why}"
        return f"{why}: she's out of throwing stars" + (f"; the pickup log then: {self.lines()}" if self.said else "")

    def lines(self) -> str:
        """The pickup log's text as one line."""
        return " / ".join(line for line in (self.said or "").splitlines() if line.strip()) or "nothing"

    def heard(self, said: str) -> None:
        """The pickup log as read at the stop (what the game said then), into its words."""
        self.said = said
        self.args = (self._text(),)

    def page(self) -> tuple[str, str]:
        return ("MapleStory: out of darts",
                f"gamelab stopped the hunt: {self.presses} Lucky Seven presses took no MP and killed nothing. "
                "She went to a safe place. Give her throwing stars.")


@dataclass
class DartWatch:
    """Her Lucky Seven presses and her MP bar, a look at a time, for out of darts (`due`): see the module's doc. The
    current stretch's presses (`keys`), those of them its last read had shown (`shown`, over `span` s), the longest
    shown (`longest`, for the run's report)."""
    presses: int = PRESSES          # the presses of a stretch that stop the hunt (presses_for)
    keys: list[float] = field(default_factory=list)  # the stretch's presses (their keys' times)
    top: int | None = None          # her bar's highest read since the stretch began (steps of MP_STEPS)
    read_at: float | None = None    # the last read's time ...
    share: float = 0.0              # ... and its share
    shown: int = 0                  # the stretch's presses the last read had shown (MP_SHOWS_S after their keys) ...
    span: float = 0.0               # ... from the first to the last of them, s
    read: int = 0                   # its presses shown when the pickup log was last read for it (`asks`)
    told: str | None = None         # the pickup log's text that said she's out (says_no_darts), this stretch
    longest: tuple[int, float] = (0, 0.0)  # the longest stretch shown (presses, s)
    fired: bool = False

    def cast(self, t: float) -> None:
        """A Lucky Seven press, its key at `t`."""
        self.keys.append(t)

    def kill(self, t: float) -> None:
        """A kill of hers (her EXP told of it) at `t`: her presses before it did something."""
        self._restart(t)

    def see(self, t: float, share: float | None) -> None:
        """Her MP bar's `share` off a frame shown at `t` (None: not read)."""
        if share is None:
            return
        px = round(share * MP_STEPS)
        if self.read_at is not None and t - self.read_at > GAP_S:  # (presses no look saw the game take aren't counted)
            self.keys = [k for k in self.keys if k <= self.read_at]
        self.read_at, self.share = t, share
        if px < LOW_STEPS:
            self._restart(t - MP_FIRST_S)
            self.top = None
            return
        if self.top is None or px <= self.top - FALL_STEPS or px >= self.top + DRINK_STEPS:
            # (the first read, a cast's MP shown, a drink: a stretch from here; presses it can't show yet go into it)
            self._restart(t - MP_FIRST_S)
            self.top = px
            return
        self.top = max(self.top, px)
        shown = [k for k in self.keys if k <= t - MP_SHOWS_S]
        self.shown, self.span = len(shown), (shown[-1] - shown[0] if shown else 0.0)
        if (self.shown, self.span) > self.longest:
            self.longest = (self.shown, self.span)

    def asks(self) -> bool:
        """Whether the pickup log is to be read now (`hear` what it says): a stretch READ_EVERY presses long, and each
        as many more, until the stop."""
        return not self.fired and self.shown >= self.read + READ_EVERY

    def hear(self, said: str) -> None:
        """The pickup log as read now (`asks`): if it says she's out of throwing stars, the stop is due."""
        self.read = self.shown
        if says_no_darts(said):
            self.told = said

    def due(self, now: float = 0.0) -> OutOfDarts | None:
        """The stop, the first time a stretch has `presses` presses shown, or READ_EVERY and the pickup log said she's
        out; else None."""
        if self.fired or not (self.shown >= self.presses or (self.told is not None and self.shown >= READ_EVERY)):
            return None
        self.fired = True
        return OutOfDarts(self.shown, self.span, self.share, said=self.told, told=self.told is not None, at=now)

    def summary(self) -> str:
        """The longest stretch, for the end of a run's log ("" under REPORT_FROM presses)."""
        n, s = self.longest
        return (f"her Lucky Seven presses: {n} in a row over {s:.1f} s took no MP at the most (out of darts stops the "
                f"hunt at {self.presses})") if n >= REPORT_FROM else ""

    def _restart(self, after: float) -> None:
        """A new stretch: the presses keyed after `after` (their MP not shown yet) go on into it."""
        self.keys = [k for k in self.keys if k > after]
        self.shown, self.span, self.read, self.told = 0, 0.0, 0, None
