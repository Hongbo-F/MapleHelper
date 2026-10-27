"""The safety stop: her HP under SAFETY_HP for SAFETY_S (SAFETY_LOOKS looks of it at least) ends the hunt. She goes to a
safe place as she does on the Lie Detector (behavior.retreat: `on_alarm` around every hunting command), the log says why
in capitals, and whoever is at the PC is paged as the Lie Detector pages them (control.alarm.page: alert sounds and a
Windows notification that doesn't take focus from the game). No sitting and no waiting for HP: the user, 2026-10-26,
after a touch stood her up from the chair and she waited out the rest's 120 s with her HP flat: "Just remove the sit
recover hp functionality, it's no longer needed". Her pet drinks her potions.

Why 40% for 2 s. Her pet drinks an HP potion (an Orange Potion, 250 HP of her 774) when a touch leaves her under its
warning, 70%: live on 2026-10-25 every reading under 0.70 was followed by a rise within 0.12-0.16 s (the perception
service's 10 readings a second; sim.world.PET_HP). Over the five live hunts of 2026-10-25 23:43 to 2026-10-26 01:28
(79 minutes, 10,548 looks; their telemetry's 'scene' records, D:/MapleHelper-scratch/norest/hp_live.py) her HP never
read under 53.8%: under 60% on two single looks (0.1 s in all), under 70% 7.3 s in all, in 19 stretches of 1.1 s (two
looks) at the longest. A Horny Mushroom's touch takes 111-131 HP, 15-17% of hers (meowdb's formula, sim.world): with the
pet drinking, getting under 40% takes two touches inside its 0.2 s between potions, and the next potion is a few tenths
of a second away. Under 40% for 2 s means it isn't drinking: out of potions, hungry, not out. From 40% (310 HP) two more
Horny touches (262 at most) still leave her alive on her way to a safe place; from 30% one does (the Tree Dungeon study:
at 30% two Horny touches kill her). Three looks over 2 s: one look's misreading (her HP bar read 0 for a look:
Hunter._doubt_hp) can't stop the hunt, and the hunt's looks (a few a second) see it within half a second of the 2 s.

It's watched on every look (Hunter._vitals: whoever took it) and raised where every fight, trip and pass already asks
after her health, every look (Hunter.health_check), once: what follows it (her way to the safe place, the stand-by) goes
on. With `--low-hp` the threshold is another (`hunt`, `probe run`, `tour`).

Out of darts (behavior.darts) ends a hunt the same way: both are a HuntStop, which retreat.on_alarm, the command line
and the simulator take as one."""

from dataclasses import dataclass

SAFETY_HP = 0.4     # HP under this ...
SAFETY_S = 2.0      # ... for this long ...
SAFETY_LOOKS = 3    # ... over this many looks at least, stops the hunt
EXIT = 4            # the command's exit code after it (the Lie Detector's is 3)


class HuntStop(Exception):
    """What ends a hunt for whoever is at the PC to put right, raised by Hunter.health_check: the safety stop, and out
    of darts (behavior.darts.OutOfDarts). Around every hunting command (behavior.retreat.on_alarm) she goes to a safe
    place as on the Lie Detector, the log says why in capitals (`label`: "<label>: <the stop>. The hunt ends: <ends>"),
    a telemetry event says so (`event`), whoever is at the PC is paged (`page`: a title and a line), the rolling
    recording's last `keep_s` s are saved (0: none), and the command exits with `exit_code`, printing `advice`
    (gamelab.__main__). `title`: its retreat's telemetry events' word (behavior.retreat.Why)."""
    label = "STOP"
    title = "Stop"
    event = "stop"
    exit_code = 1
    keep_s = 0.0
    ends = "to a safe place"
    advice = ""
    at = 0.0

    def page(self) -> tuple[str, str]:
        """What's paged (control.alarm.page): a title and a line."""
        return ("MapleStory: the hunt stopped", str(self))


class SafetyStop(HuntStop):
    """Her HP stayed under the safety stop's threshold: the hunt ends (raised by Hunter.health_check). `hp`: her HP at
    the look that ended it; `seconds`: how long it had been under; `looks`: the looks of it; `threshold`; `at`: that
    look's time (time.monotonic)."""
    label = "SAFETY STOP"
    title = "Safety stop"
    event = "safety_stop"
    exit_code = EXIT
    ends = "to a safe place (no sitting, no waiting for HP)"
    advice = "Check her HP potions and her pet (fed, out, its potions) in the game before hunting again."

    def __init__(self, hp: float, seconds: float, looks: int, threshold: float, at: float = 0.0):
        self.hp, self.seconds, self.looks, self.threshold, self.at = hp, seconds, looks, threshold, at
        super().__init__(f"HP {hp:.0%}, under {threshold:.0%} for {seconds:.1f} s ({looks} looks): her pet's potions "
                         "aren't keeping her up (none left? the pet hungry or not out?)"
                         + ("; she may be dead" if hp <= 0.02 else ""))

    def page(self) -> tuple[str, str]:
        """What's paged (control.alarm.page): a title and a line."""
        return ("MapleStory: her HP stayed low",
                f"gamelab stopped the hunt: HP {self.hp:.0%}, under {self.threshold:.0%} for {self.seconds:.0f} s. "
                "She went to a safe place. Check her potions and her pet.")


@dataclass
class HpWatch:
    """The safety stop's watch over her HP, a look at a time (`see`); `due` once it's to stop, then never again."""
    threshold: float = SAFETY_HP
    since: float | None = None  # the first look of the stretch under the threshold (its time)
    looks: int = 0              # ... the looks of it
    hp: float = 1.0             # the last look's
    at: float = 0.0             # ... and its time
    fired: bool = False

    def see(self, t: float, hp: float | None) -> None:
        """A look at time `t` read her HP `hp` (None: not read, which neither starts nor ends a stretch)."""
        if hp is None:
            return
        self.hp, self.at = hp, t
        if hp >= self.threshold:
            self.since, self.looks = None, 0
            return
        if self.since is None:
            self.since = t
        self.looks += 1

    def due(self) -> SafetyStop | None:
        """The stop, the first time her HP has been under the threshold SAFETY_S over SAFETY_LOOKS looks; else None."""
        if self.fired or self.since is None or self.looks < SAFETY_LOOKS or self.at - self.since < SAFETY_S:
            return None
        self.fired = True
        return SafetyStop(self.hp, self.at - self.since, self.looks, self.threshold, self.at)
