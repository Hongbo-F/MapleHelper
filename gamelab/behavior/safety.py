"""The safety stop: her HP under SAFETY_HP for SAFETY_S (SAFETY_LOOKS looks of it at least), or under FLOOR_HP
FLOOR_LOOKS looks in a row, ends the hunt. She goes to a safe place as she does on the Lie Detector (behavior.retreat:
`on_alarm` around every hunting command), the log says why in capitals, and whoever is at the PC is paged as the Lie
Detector pages them (control.alarm.page: alert sounds and a Windows notification that doesn't take focus from the game).
No sitting and no waiting for HP: the user, 2026-10-26, after a touch stood her up from the chair and she waited out the
rest's 120 s with her HP flat: "Just remove the sit recover hp functionality, it's no longer needed". Her pet drinks her
potions.

Why 65% for 2 s. Her pet drinks an HP potion (an Orange Potion, 250 HP of her 774) when a touch leaves her under its
warning, 70%: live on 2026-10-25 every reading under 0.70 was followed by a rise within 0.12-0.16 s (the perception
service's 10 readings a second; sim.world.PET_HP). Over the five live hunts of 2026-10-25 23:43 to 2026-10-26 01:28
(79 minutes, 10,548 looks; their telemetry's 'scene' records, D:/MapleHelper-scratch/norest/hp_live.py) her HP never
read under 53.8%; under 65% on 8 single looks, the next look over it 0.86 s later at the most; under 70% in 19
stretches, three looks over 0.17 s at the most. A Horny Mushroom's touch takes 111-131 HP, 15-17% of hers (meowdb's
formula, sim.world), and after a touch she can't be touched for a moment: with the pet drinking she's back over 65%
within tenths of a second. Under 65% for 2 s, three looks of it, means it isn't drinking: out of potions, hungry, not
out. Five points under its warning, so a bar read at 69.x% isn't taken for it; 2 s rather than 1, so a lag of the
game's isn't. First built at 40% (two Horny touches of margin, the 30% first thought of one): the review found it late
once the potions ran out, in the simulator 5 deaths in 14 such runs against 2 at 65% (master, resting, 9): her looks
pause 2-3.4 s on trips and loot runs, and on the tour's seed 2 she was at 40-47% for 10 s with no potion, three touches
in a gap of 3.4 s, her first look under 40% read 11%, and she died before 2 s of looks under it. The floor, FLOOR_HP
FLOOR_LOOKS looks in a row, is for such a gap: under it the 2 s aren't waited (an HP read 0 after a higher one isn't
believed until the next look reads it too, Hunter._doubt_hp: it takes three such misreads in a row). With a potion key
of her own drinking under `--potion-hp` (none bound since 2026-10-23; the first map's simulator binds one, no pet),
five points under that if it's lower (`threshold`): between the two is where her key keeps her HP.

It's watched on every look (Hunter._vitals: whoever took it) and raised where every fight, trip and pass already asks
after her health, every look (Hunter.health_check), once: what follows it (her way to the safe place, the stand-by) goes
on. With `--low-hp` the threshold is another (`hunt`, `probe run`, `tour`).

Out of darts (behavior.darts) ends a hunt the same way: both are a HuntStop, which retreat.on_alarm, the command line
and the simulator take as one."""

from dataclasses import dataclass

SAFETY_HP = 0.65    # HP under this (UNDER under her pet's warning, 70%) ...
SAFETY_S = 2.0      # ... for this long ...
SAFETY_LOOKS = 3    # ... over this many looks at least, stops the hunt
FLOOR_HP = 0.25     # ... and HP under this ...
FLOOR_LOOKS = 2     # ... on this many looks in a row stops it then (a gap in her looks can't hide a fall)
UNDER = 0.05        # with a potion key of her own (`--potion-hp`), this under its threshold if that's lower
EXIT = 4            # the command's exit code after it (the Lie Detector's is 3)


def threshold(potion_hp: float | None = None) -> float:
    """The safety stop's threshold: SAFETY_HP, under her pet's warning; with a potion key of her own drinking under
    `potion_hp` (Hunter.drink_if_low; none bound since 2026-10-23), UNDER under that if it's lower: between the two her
    HP is where her key leaves it."""
    return SAFETY_HP if potion_hp is None else min(SAFETY_HP, round(potion_hp - UNDER, 3))


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
    look's time (time.monotonic); `floor`: the floor if it was under that which stopped it (FLOOR_LOOKS looks in a
    row), else None."""
    label = "SAFETY STOP"
    title = "Safety stop"
    event = "safety_stop"
    exit_code = EXIT
    ends = "to a safe place (no sitting, no waiting for HP)"
    advice = "Check her HP potions and her pet (fed, out, its potions) in the game before hunting again."

    def __init__(self, hp: float, seconds: float, looks: int, threshold: float, at: float = 0.0,
                 floor: float | None = None):
        self.hp, self.seconds, self.looks, self.threshold, self.at = hp, seconds, looks, threshold, at
        self.floor = floor
        under = f"under {threshold:.0%} for {seconds:.1f} s ({looks} looks)"
        if floor is not None:
            under += f", under {floor:.0%} on {FLOOR_LOOKS} looks in a row"
        super().__init__(f"HP {hp:.0%}, {under}: her pet's potions aren't keeping her up (none left? the pet hungry "
                         "or not out?)" + ("; she may be dead" if hp <= 0.02 else ""))

    def page(self) -> tuple[str, str]:
        """What's paged (control.alarm.page): a title and a line."""
        return ("MapleStory: her HP stayed low",
                f"gamelab stopped the hunt: HP {self.hp:.0%}, under {self.threshold:.0%} for {self.seconds:.1f} s. "
                "She went to a safe place. Check her potions and her pet.")


@dataclass
class HpWatch:
    """The safety stop's watch over her HP, a look at a time (`see`); `due` once it's to stop, then never again."""
    threshold: float = SAFETY_HP
    since: float | None = None  # the first look of the stretch under the threshold (its time)
    looks: int = 0              # ... the looks of it
    low: int = 0                # ... the looks in a row under the floor (FLOOR_HP, or the threshold if that's lower)
    hp: float = 1.0             # the last look's
    at: float = 0.0             # ... and its time
    fired: bool = False

    @property
    def floor(self) -> float:
        return min(FLOOR_HP, self.threshold)

    def see(self, t: float, hp: float | None) -> None:
        """A look at time `t` read her HP `hp` (None: not read, which neither starts nor ends a stretch)."""
        if hp is None:
            return
        self.hp, self.at = hp, t
        self.low = self.low + 1 if hp < self.floor else 0
        if hp >= self.threshold:
            self.since, self.looks = None, 0
            return
        if self.since is None:
            self.since = t
        self.looks += 1

    def due(self) -> SafetyStop | None:
        """The stop, the first time her HP has been under the threshold SAFETY_S over SAFETY_LOOKS looks, or under the
        floor FLOOR_LOOKS looks in a row; else None."""
        if self.fired or self.since is None:
            return None
        floor = self.low >= FLOOR_LOOKS
        if not floor and (self.looks < SAFETY_LOOKS or self.at - self.since < SAFETY_S):
            return None
        self.fired = True
        return SafetyStop(self.hp, self.at - self.since, self.looks, self.threshold, self.at,
                          self.floor if floor else None)
