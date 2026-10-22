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
- A change the number can't be read at (the cursor or a window over it) is one kill of a kind not known, as every
  change was before the number was read; the EXP it was still counts toward her EXP a minute once a reading comes.
"""

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

from maplehelper.ocr.exp_table import EXP_TO_NEXT

from ..perception.exp_number import ExpReading
from ..perception.kinds import kind_of

JUMP_KILLS = 4    # a gain worth more than this many of the map's best kills waits for a second reading ...
SPLIT_KILLS = 4   # ... and one is split into this many kills at most
RETRY_S = 0.5     # s between readings while the number has no value yet (the first look, or after one it couldn't read)
JUMP_LEAST = 30   # EXP: the least a jump is, whatever the map's monsters give
# Whether a gain's size names the kinds it was (else each of its kills is of a kind not known, as before the number
# was read). In the saved frames every gain was the game data's EXP for a kind, the server's rate 1; a party's share,
# an event's rate or the EXP shared with another player who hit the monster too would make sizes the data's don't.
NAME_KINDS = True


@dataclass(frozen=True)
class Gain:
    """Her EXP number changed at `t`: by `exp` (None: it couldn't be read), the kills that was by kind (None where
    the kind isn't known; empty: no kills add up to it, or the number went down)."""
    t: float
    exp: int | None
    kinds: tuple[str | None, ...]

    @property
    def kills(self) -> int:
        return len(self.kinds)

    @property
    def odd(self) -> bool:
        """A gain no kill of the map's explains (a party share, a quest), or a fall: not a kill."""
        return not self.kinds


@dataclass
class ExpMeter:
    """Her EXP as the readings of her number go (`update` at every look that read it), and what each change was.
    `exp_per_kill`: monster name -> EXP per kill (MapModel.mob_exp)."""
    exp_per_kill: dict[str, int]
    at: tuple[int | None, int] | None = None   # her level (None: not known yet) and EXP, as last read
    total: int = 0                             # EXP gained since the first reading
    since: float | None = None                 # ... when that was
    held: tuple[float, ExpReading] | None = None  # a jump or a fall, waiting for a second reading
    unread: int = 0                            # changes since the last reading that couldn't be read
    tried: float = -1e9                        # when a reading was last tried without a change (wants_reading)
    counts: dict = field(default_factory=lambda: {"read": 0, "unread": 0, "kills": 0, "split": 0, "odd": 0,
                                                  "held": 0, "dropped": 0, "fell": 0})

    def __post_init__(self):
        self.per_kill = {kind_of(name): e for name, e in self.exp_per_kill.items() if e > 0}
        self.jump = max(JUMP_LEAST, JUMP_KILLS * max(self.per_kill.values(), default=0))

    def wants_reading(self, t: float) -> bool:
        """Whether a look with the number unchanged should read it anyway: to start from (none read yet, or since
        the last change it couldn't read), or to confirm a jump held; RETRY_S apart at most."""
        if (self.at is None or self.unread or self.held is not None) and t - self.tried >= RETRY_S:
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
                return []
            self.counts["unread"] += 1
            if self.at is not None:
                self.unread += 1
            return [Gain(t, None, (None,))]
        self.counts["read"] += 1
        level = self._level(reading)
        if self.at is None:  # the first reading: where she starts from
            self._accept(t, reading.exp, level)
            return [Gain(t, None, (None,))] if changed else []
        size = self._since(level, reading.exp)
        if self.unread:  # after changes it couldn't read: their kills are counted, and their EXP from here
            if size is not None and 0 <= size <= self.jump * (self.unread + 1):
                self.total += size
            self.unread, self.held = 0, None
            self._accept(t, reading.exp, level)
            return [Gain(t, None, (None,))] if changed else []
        if self.held is not None:
            t0, first = self.held
            if first.exp == reading.exp and set(first.levels) & set(reading.levels):
                self.held = None
                return self._take(t0, size, reading.exp, level)
            self.held = None  # the one held was misread, or passing: this reading is weighed on its own
            self.counts["dropped"] += 1
        if size == 0:
            return []
        if size is not None and 0 < size <= self.jump:
            return self._take(t, size, reading.exp, level)
        self.held = (t, reading)  # a jump, a fall, or a level that doesn't follow: a second reading must agree
        self.counts["held"] += 1
        return []

    # --- the parts ---------------------------------------------------------------------------------

    def split(self, size: int) -> tuple[str | None, ...]:
        """The kills a gain of `size` was: the fewest of the map's kinds whose EXP adds up to it (SPLIT_KILLS at most),
        by kind if only one set does, else as many kills of kinds not known; () if none does."""
        kinds = sorted(self.per_kill)
        for k in range(1, SPLIT_KILLS + 1):
            ways = [c for c in combinations_with_replacement(kinds, k) if sum(self.per_kill[x] for x in c) == size]
            if ways:
                return ways[0] if len(ways) == 1 and NAME_KINDS else (None,) * k
        return ()

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

    def _take(self, t: float, size: int | None, exp: int, level: int | None) -> list[Gain]:
        """A change confirmed (at t, when it was first seen): a gain of `size`, or a fall."""
        self._accept(t, exp, level)
        if size is None or size < 0:
            self.counts["fell"] += 1
            return [Gain(t, size, ())]
        if size == 0:
            return []
        self.total += size
        kinds = self.split(size)
        self.counts["kills"] += len(kinds)
        self.counts["split"] += len(kinds) > 1
        self.counts["odd"] += not kinds
        return [Gain(t, size, kinds)]
