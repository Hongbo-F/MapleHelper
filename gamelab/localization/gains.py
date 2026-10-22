"""Her EXP gains, from her EXP number as read (perception.exp_number): how much each one was, the kills it was, and
her EXP a minute from the number itself.

The number is read when it changes (perception.hud.exp_changed), so reading costs nothing in between. A gain's size
names the kind killed: each kind's EXP per kill is the game data's (MapModel.mob_exp; on Thicket Around the Beach II
a snail 2, a slime 12, a pig 13, an Orange Mushroom 15), and in the saved frames her gains were exactly those (453 ->
466 -> 478 -> 490 -> 503 -> 516: a pig, two slimes and two pigs; 451 -> 453, a snail). A gain that is several kills
(two between two looks) is split into them, the fewest kills that add up to it, their kinds named only when one way
does. A gain no kills add up to (a party member's share, a quest's EXP) is flagged and isn't a kill.

- A gain worth more than JUMP_KILLS of the map's best kills, or the number going down (a death costs EXP), waits for
  a second reading that agrees with it (the next look reads the number again, `wants_reading`); one that doesn't
  agree is dropped as a misreading.
- A level-up: the number back near 0 at the next level (each level's requirement is the EXP table's) is a gain of
  what was left of the last level and what's into the new one.
- A change the number can't be read at (the cursor or a window over it) waits for the next reading to say what it
  was: the cursor over the number is a change as it comes and one as it goes, and the reading after it says whether
  any EXP came meanwhile (none: no kill) and how much. The changes it couldn't read come in bursts (each ALONE_S from
  the next: a cursor moving is many changes, a kill one), the first of them the cover coming; the kills the reading's
  size says go at the times of the latest bursts. A burst of one change with none after it that no reading has
  explained UNREAD_WAIT_S later is one kill of a kind not known (a cursor left over the number, or a level whose
  requirement the table has wrong: then it's never read), as every change was before the number was read; the
  first burst never is.
- Its sizes are checked: of the last RATE_GAINS gains each from one change, fewer than RATE_OFF one kill of a kind
  (at EXP_RATE) says the sizes aren't the game data's (an EXP event or coupon, a share of a party's or another
  player's kill); then every change is one kill of a kind not known, as before, nothing is split or flagged, and
  it's logged (`notes`), until RATE_ON of them are again. Until RATE_ON of the first RATE_GAINS have been, a gain is
  named when it's one kill but not split (at rate 2 every kill would be two).
"""

from collections import deque
from dataclasses import dataclass, field
from itertools import combinations_with_replacement

from maplehelper.ocr.exp_table import EXP_TO_NEXT

from ..perception.exp_number import ExpReading
from ..perception.kinds import kind_of

JUMP_KILLS = 4    # a gain worth more than this many of the map's best kills waits for a second reading ...
SPLIT_KILLS = 4   # ... and one is split into this many kills at most (a run of changes it couldn't read: as many as
RUN_KILLS = 12    # its bursts, up to this)
RETRY_S = 0.5     # s between readings while the number has no value yet (the first look, or after a change it couldn't
RETRY_MAX_S = 8.0  # read), doubled after each that fails up to this: a cursor left over the number, or a percent no
                   # level's requirement fits, costs a reading (8.6 ms) every 8 s, not every 0.5 (17-20 ms a second)
JUMP_LEAST = 30   # EXP: the least a jump is, whatever the map's monsters give
# s between two changes it couldn't read for them to be two bursts: a cursor moving over the number changes it at every
# look (the feed's 30 a second, the hunter's 3-10), her kills come seconds apart (one every 6 s in the simulator's hunt
# loop, 130 EXP a minute at about 13 a kill).
ALONE_S = 0.5
# s a burst of one change it couldn't read waits for a reading to say what it was before it's one kill of a kind not
# known: a cursor passing over the number is gone within it (4 retries).
UNREAD_WAIT_S = 1.5
# The server's EXP rate: each kind's EXP per kill is the game data's times this, rounded. The saved frames' gains were
# the data's own (rate 1); an event or a coupon would make it 1.5 or 2.
EXP_RATE = 1.0
# The rate check: of the last RATE_GAINS gains each from one change, fewer than RATE_OFF one kill of a kind turns the
# sizes off (each change one kill of a kind not known), RATE_ON or more back on. At rate 2 every kill reads as two
# of its kind (a pig's 26 two pigs), at 1.5 a slime's 18 or a pig's 19 as 3 or 4 kills and a snail's 3 or an Orange
# Mushroom's 22 as none: without the check 6 kills counted 12-14 (6 at rate 1). In the simulator (rate 1) every gain of the harness's runs (preset
# game, tour and loop, 10 seeds of 20 minutes: 3,719 gains) was one kill, and the sizes never went off; in game a
# gain of two kills between two looks, or a party's share, isn't one kill either, and it takes 5 of 8 in a row.
RATE_GAINS = 8
RATE_OFF = 4
RATE_ON = 6
RATES = (0.5, 1.5, 2.0, 2.5, 3.0)  # the rates `notes` says the gains fit, when the sizes go off
# Whether a gain's size says what it was at all: the kinds it was named, two kills split, one no kill gives flagged.
# Off, every change is one kill of a kind not known, as before the number was read (its EXP a minute still counts).
NAME_KINDS = True


@dataclass(frozen=True)
class Gain:
    """Her EXP number changed at `t`: by `exp` (None: it couldn't be read, or the levels read don't follow), the kills
    that was by kind (None where the kind isn't known; empty: no kills add up to it, or the number went down), and
    when each was (`times`; empty: all at t)."""
    t: float
    exp: int | None
    kinds: tuple[str | None, ...]
    times: tuple[float, ...] = ()

    @property
    def kills(self) -> int:
        return len(self.kinds)

    @property
    def odd(self) -> bool:
        """A gain no kill of the map's explains (a party share, a quest), a fall, or levels that don't follow: not a
        kill."""
        return not self.kinds

    def when(self, i: int) -> float:
        """When its i-th kill was."""
        return self.times[i] if i < len(self.times) else self.t


@dataclass
class ExpMeter:
    """Her EXP as the readings of her number go (`update` at every look that read it or saw it change, `due` at every
    look), and what each change was. `exp_per_kill`: monster name -> EXP per kill (MapModel.mob_exp)."""
    exp_per_kill: dict[str, int]
    at: tuple[int | None, int] | None = None   # her level (None: not known yet) and EXP, as last read
    total: int = 0                             # EXP gained since the first reading
    since: float | None = None                 # ... when that was
    held: tuple[float, ExpReading, bool] | None = None  # a jump or a fall, waiting for a second reading (and whether
                                                         # it was a change)
    # The bursts of changes since the last reading that it couldn't read: [first change's time, last's, changes,
    # counted as a kill (due)].
    run: list[list] = field(default_factory=list)
    tried: float = -1e9                        # when a reading was last tried without a change (wants_reading) ...
    retry: float = RETRY_S                     # ... and how long until the next
    sized: bool | None = None                  # the rate check: whether the gains' sizes are the game data's (None:
                                               # not seen yet)
    recent: deque = field(default_factory=lambda: deque(maxlen=RATE_GAINS))  # the last gains from one change each
    notes: list[str] = field(default_factory=list)  # what's to be logged (the rate check changing its mind)
    counts: dict = field(default_factory=lambda: {"read": 0, "unread": 0, "kills": 0, "split": 0, "odd": 0,
                                                  "held": 0, "dropped": 0, "fell": 0, "levels": 0, "covered": 0,
                                                  "late": 0, "over": 0, "unsized": 0})

    def __post_init__(self):
        self.per_kill = {kind_of(name): max(1, round(e * EXP_RATE)) for name, e in self.exp_per_kill.items() if e > 0}
        self.best = max(self.per_kill.values(), default=0)
        self.jump = max(JUMP_LEAST, JUMP_KILLS * self.best)

    @property
    def unread(self) -> int:
        """Changes since the last reading that it couldn't read."""
        return sum(b[2] for b in self.run)

    def wants_reading(self, t: float) -> bool:
        """Whether a look with the number unchanged should read it anyway: to start from (none read yet, or since
        the last change it couldn't read), or to confirm a jump held; `retry` apart at most (RETRY_S, doubled after
        each that fails, to RETRY_MAX_S)."""
        if (self.at is None or self.run or self.held is not None) and t - self.tried >= self.retry:
            self.tried = t
            return True
        return False

    def per_minute(self, now: float) -> float | None:
        """Her EXP a minute since the first reading, from the number itself (None in the first 10 s)."""
        if self.since is None or now - self.since < 10.0:
            return None
        return self.total * 60.0 / (now - self.since)

    def update(self, t: float, reading: ExpReading | None, changed: bool) -> list[Gain]:
        """A look at t that read the number (`reading`; None or without a value: it couldn't be read), or that saw it
        change (`changed`). Returns what the change was: no gain, or one (of one kill or several, or none)."""
        if reading is None or reading.exp is None:
            if not changed:
                self.retry = min(RETRY_MAX_S, self.retry * 2)
                return []
            self.counts["unread"] += 1
            self.retry = RETRY_S  # (read again soon: whatever is over the number may be going)
            if self.at is None:  # nothing read yet to tell it by: a kill, as every change was before the number was read
                self.counts["kills"] += 1
                return [Gain(t, None, (None,))]
            if self.run and t - self.run[-1][1] < ALONE_S:
                self.run[-1][1] = t
                self.run[-1][2] += 1
            else:
                self.run.append([t, t, 1, False])
            return []
        self.counts["read"] += 1
        self.retry = RETRY_S
        level = self._level(reading)
        if self.at is None:  # the first reading: where she starts from
            self._accept(t, reading.exp, level)
            self.counts["kills"] += changed
            return [Gain(t, None, (None,))] if changed else []
        size = self._since(level, reading.exp)
        if self.held is not None:
            t0, first, was = self.held
            self.held = None
            if first.exp == reading.exp and set(first.levels) & set(reading.levels):
                return self._take(t0, size, reading.exp, level, was)
            self.counts["dropped"] += 1  # the one held was misread, or passing: this reading is weighed on its own
        if size == 0:  # no EXP came: something passed over the number (no kill), or a reading after a change that read
            if self.run:  # the same
                self._end_run(t, 0)
            return []
        kills = self._expected(changed)
        limit = self.jump if not self.run else max(self.jump, self.best * (kills + 1))
        if size is not None and 0 < size <= limit:
            return self._take(t, size, reading.exp, level, changed)
        self.held = (t, reading, changed)  # a jump, a fall, or a level that doesn't follow: a second reading must agree
        self.counts["held"] += 1
        return []

    def due(self, t: float) -> list[Gain]:
        """At every look: the bursts of changes it couldn't read, of one change each with none after it, that no
        reading has said what they were UNREAD_WAIT_S later (not the first: the cover coming), each one kill of a
        kind not known, at its time."""
        out = []
        for i, b in enumerate(self.run):
            if i > 0 and not b[3] and b[2] == 1 and t - b[1] >= UNREAD_WAIT_S:  # (by then no change can join it)
                b[3] = True
                self.counts["late"] += 1
                self.counts["kills"] += 1
                out.append(Gain(b[0], None, (None,)))
        return out

    # --- the parts ---------------------------------------------------------------------------------

    def split(self, size: int, expect: int = 1, most: int = SPLIT_KILLS) -> tuple[str | None, ...]:
        """The kills a gain of `size` was: of the numbers of kills (`most` at most) whose EXP can add up to it, the one
        nearest `expect` (on a tie the fewer), by kind if only one set of kinds of that many does, else as many kills
        of kinds not known; () if none does."""
        kinds = sorted(self.per_kill)
        found = {}
        for k in range(1, most + 1):
            if k * min(self.per_kill.values(), default=1) > size:
                break
            ways = [c for c in combinations_with_replacement(kinds, k) if sum(self.per_kill[x] for x in c) == size]
            if ways:
                found[k] = ways
                if k >= expect:
                    break
        if not found:
            return ()
        k = min(found, key=lambda n: (abs(n - expect), n))
        return found[k][0] if len(found[k]) == 1 else (None,) * k

    def _expected(self, changed: bool) -> int:
        """How many kills a reading after the run of changes it couldn't read can have been: one a burst, less the
        first's if this reading is a change (the cover coming, then going); one for a reading's own change."""
        if not self.run:
            return 1
        return max(1, len(self.run) - (1 if changed else 0))

    def _level(self, reading: ExpReading) -> int | None:
        """Her level: the reading's if only one fits it, else the one that follows from the last (the same, or the next
        after a level-up)."""
        if reading.level is not None or self.at is None or self.at[0] is None:
            return reading.level
        l0, e0 = self.at
        for lv in ((l0, l0 + 1) if reading.exp >= e0 else (l0 + 1, l0)):
            if lv in reading.levels:
                return lv
        return None

    def _since(self, level: int | None, exp: int) -> int | None:
        """EXP gained from the last reading to (`level`, `exp`): negative if it went down, None if the levels don't
        follow (more than one level-up, or down)."""
        l0, e0 = self.at
        if level == l0 or level is None or l0 is None:
            return exp - e0
        if level == l0 + 1 and l0 in EXP_TO_NEXT:
            return EXP_TO_NEXT[l0] - e0 + exp
        return None

    def _accept(self, t: float, exp: int, level: int | None) -> None:
        self.at = (level, exp)
        if self.since is None:
            self.since = t

    def _end_run(self, t: float, kills: int | None = None, changed: bool = True) -> list[float]:
        """The run of changes it couldn't read is over at a reading at t (`changed`: a change itself, the cover
        going), and was `kills` kills (None: as many as were counted, due). Returns when the kills not counted yet
        were: at the latest bursts not counted but the first (the cover coming; when this reading is no change, there
        was no cover, and it's a kill's too), and any more at t. Counted ones beyond `kills` can't be taken back
        (`over`)."""
        run, self.run = self.run, []
        counted = sum(1 for b in run if b[3])
        kills = counted if kills is None else kills
        if kills == 0:
            self.counts["covered"] += 1
        self.counts["over"] += max(0, counted - kills)
        free = [b[0] for b in run[1:] if not b[3]][::-1] + [b[0] for b in run[:1] if not b[3] and not changed]
        new = max(0, kills - counted)
        return sorted(free[:new] + [t] * max(0, new - len(free)))

    def _take(self, t: float, size: int | None, exp: int, level: int | None, changed: bool) -> list[Gain]:
        """A change confirmed (at t, when it was first seen): a gain of `size`, a fall, or levels that don't follow;
        it ends the run of changes it couldn't read before it, if any."""
        run = bool(self.run)
        counted = sum(1 for b in self.run if b[3])
        expect = self._expected(changed)
        self._accept(t, exp, level)
        if size is None or size < 0:
            self.counts["levels" if size is None else "fell"] += 1
            self._end_run(t)
            return [Gain(t, size, ())]
        self.total += size
        if not run:
            self._check(size)
        if not (NAME_KINDS and self.sized is not False):  # the sizes off: each change one kill of a kind not known
            self.counts["unsized"] += 1
            kinds = (None,) * (expect if run else 1)
        elif not run:
            kinds = self.split(size)
        else:
            kinds = self.split(size, expect, min(RUN_KILLS, max(SPLIT_KILLS, expect + 1)))
        if self.sized is None and len(kinds) > 1:  # not seen to be the data's yet: not split (at rate 2 each kill would
            kinds = (None,) if not run else (None,) * len(kinds)  # be)
        if not kinds:
            self.counts["odd"] += 1
            self._end_run(t)
            return [Gain(t, size, ())]
        if not run:
            self.counts["kills"] += len(kinds)
            self.counts["split"] += len(kinds) > 1
            return [Gain(t, size, kinds)]
        times = self._end_run(t, len(kinds), changed)  # the kills not counted yet (the run's last, or beyond them)
        if not times:
            return []
        # Named when none was counted before and all are one kind: which burst was which kind isn't known.
        named = kinds if counted == 0 and len(set(kinds)) == 1 else (None,) * len(times)
        self.counts["kills"] += len(times)
        self.counts["split"] += len(times) > 1
        return [Gain(times[0], size, tuple(named), tuple(times))]

    def _check(self, size: int) -> None:
        """The rate check (see the top), on a gain from one change of `size`."""
        self.recent.append(size)
        if len(self.recent) < RATE_GAINS:
            return
        def ones(rate: float) -> int:  # how many of the recent gains were one kill at `rate`
            singles = {max(1, round(e * rate)) for e in self.exp_per_kill.values() if e > 0}
            return sum(s in singles for s in self.recent)
        one = ones(EXP_RATE)
        if self.sized is None and one >= RATE_ON:
            self.sized = True
        elif self.sized is not False and one < RATE_OFF:
            self.sized = False
            fits = [r for r in RATES if ones(r) >= RATE_ON]
            self.notes.append(f"her EXP gains aren't the game data's EXP a kill at rate {EXP_RATE:g} ({one} of the last "
                              f"{RATE_GAINS} were one kill: {', '.join(map(str, self.recent))}): each change counted "
                              f"as one kill, kinds not named" + (f"; they fit rate {fits[0]:g} (gains.EXP_RATE)"
                                                                 if fits else ""))
        elif self.sized is False and one >= RATE_ON:
            self.sized = True
            self.notes.append(f"her EXP gains are the game data's again ({one} of the last {RATE_GAINS} one kill): "
                              f"kinds named")
