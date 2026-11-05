"""The loot chain (`on`, GAMELAB_LOOT_CHAIN: '0' off, the whole screen alone to the digit; on by default on this branch). The user,
2026-11-04: "You should also apply similar concept for loot pickup, don't just go to the first target, pickup, then wait, plan for
next target, move, then pickup. You should instead plan for a few loot targets in advance so you don't need to have this
'pauses'". The sweep (behavior.sweep) still says which piles and in what order; this says how she goes through them: MOST_PILES at
once, as one chain of the navigator's (control.navigator's `through`: the route's moves and 'walk' legs, motion.flow.Flow._walk), to
each bunch of drops as a look last saw them, not to the pile's middle, pick-up pressed all the way, with no stand, no settle and no
look between them. She stands only for the presses a run past a bunch wouldn't give (RUN_PRESSES), and where a pile she has been
through still shows drops (`Stretch.again`: as a stand does, a press for each and one more). What's left once she's passed it a look
says, and it's a stop of the next plan (Grind._candidate: none of it, and the pile's no stop; 100% is the aim, what's left isn't
left). Pure: Grind has the looks, the drop store and the navigator; here's the plan's shape. README: "The loot chain".
"""

import os
from dataclasses import dataclass, field

from ..control.actions import PILE_EVERY_MS
from .ledger import Pile
from .looting import PILE_GAP, STOOD_ON

MOST_PILES = 4      # piles a chain goes through at most ("a few": the rest are the next plan's, from a newer look)
# Presses a run through a bunch of drops gives it for sure: she passes its PICK_REACH (15 units either side) in 30 units, 0.24 s at
# 125 a second, 0.2 at her 150: 2 at a press every flow.WALK_PICK_EVERY (0.12 s), the magician's too but by a hair. What's left
# after that a look says. Read from the game's data and the simulator's reach (live her reach isn't measured apart from STOOD_ON's
# "within 15"), not tried live.
RUN_PRESSES = 2
PRESS_S = PILE_EVERY_MS / 1000.0
CHAIN_OFF_S = 0.5   # guess: s she goes the old way (a trip, a stand) after a monster stood in a chain's first walk
END_ROOM = 10.0     # units a drop by a platform's end is walked to short of it (under her reach: no slide off for it; a guess)


def on() -> bool:
    """Whether her drops go by the chain (GAMELAB_LOOT_CHAIN: '0' off; on by default on this branch; with the whole screen alone)."""
    return os.environ.get("GAMELAB_LOOT_CHAIN", "1") != "0"


def bunches(xs) -> list[tuple[float, int]]:
    """Drops' x made bunches, those closer together than PILE_GAP one (picked up from one place): (their middle, how many), left to right."""
    out: list[list[float]] = []
    for x in sorted(xs):
        if out and x - out[-1][-1] < PILE_GAP:
            out[-1].append(x)
        else:
            out.append([x])
    return [(sum(b) / len(b), len(b)) for b in out]


def stand_s(n: int, again: bool = False) -> float:
    """The s she stands at a bunch of `n` drops: the presses a run through it wouldn't give; `again` (she's been through it and
    drops still show): a press for each and one more, as a stand has always pressed."""
    return max(0, (n + 1) if again else (n - RUN_PRESSES)) * PRESS_S


@dataclass
class Stretch:
    """One pile of a chain: its bunches of drops as seen (left to right), the kinds of the items among them, and those she has run
    over (`ran`: their index), and whether she has been through it before."""
    pile: Pile
    bunches: list[tuple[float, int]]
    kinds: tuple[str, ...] = ()
    again: bool = False
    seen: bool = True   # (False: drops no look has seen yet, or not landed: a chain ends there, and a look says)
    ran: set[int] = field(default_factory=set)
    booked: bool = False

    @property
    def platform(self) -> int:
        return self.pile.platform

    @property
    def done(self) -> bool:
        return len(self.ran) >= len(self.bunches)


@dataclass
class LootChain:
    """A chain's plan: its stretches in the sweep's order, where she goes on to after them (`dest`: her target's place, the next
    stop's; None: nowhere), the look it was planned on and its time by the sweep's reckoning (`est_s`)."""
    stretches: list[Stretch]
    dest: tuple[int, float] | None = None
    planned_on: object = None
    est_s: float = 0.0
    started: float = 0.0

    def way(self, platform: int, x: float, enter) -> list[tuple[int, float, float]]:
        """What's left of the way from (platform, x), for the navigator's `through`: (platform, x, s standing there), each stretch from
        its nearer end as she'll come onto it (`enter(platform, x, onto, lo, hi)`: the x she comes onto that stretch at, or None: no
        way), a bunch she'll only run past merged into the walk that goes on past it. A stand where she already is: a walk of nothing has
        no keys."""
        out, at, here = [], x, platform
        for st in self.stretches:
            left = [b for i, b in enumerate(st.bunches) if i not in st.ran]
            if not left:
                continue
            lo, hi = left[0][0], left[-1][0]
            came = at if st.platform == here else enter(here, at, st.platform, lo, hi)
            if came is None:
                break
            for bx, n in (left if abs(came - lo) <= abs(came - hi) else left[::-1]):
                out.append((st.platform, bx, stand_s(n, st.again)))
                at, here = bx, st.platform
        if self.dest is not None:
            out.append((self.dest[0], self.dest[1], 0.0))
        out = _run_through(out, platform, x)
        if out and out[0][0] == platform and abs(out[0][1] - x) < 1.0 and out[0][2] <= 0.0:
            out[0] = (platform, out[0][1], PRESS_S)
        return out

    def book(self, ran: list[tuple[int, float, float]], place=lambda p: p) -> None:
        """Her chain ran her over these stretches (platform, x from, x to; `place`: a twin's the platform it lies on, as her moves have
        it: MapModel.place_of): the bunches within STOOD_ON of them are behind her."""
        for st in self.stretches:
            on = place(st.platform)
            for i, (bx, _) in enumerate(st.bunches):
                if i not in st.ran and any(p == on and min(a, b) - STOOD_ON <= bx <= max(a, b) + STOOD_ON for p, a, b in ran):
                    st.ran.add(i)

    @property
    def done(self) -> bool:
        return all(st.done for st in self.stretches)

    def text(self) -> str:
        """The log's: '3 stops, x -33 on 9, x 100 on 8, x 148 on 8: 2.9 s', and where she goes on to."""
        n = len(self.stretches)
        return (f"{n} stop{'' if n == 1 else 's'}, "
                + ", ".join(f"x {st.pile.x:.0f} on {st.platform}" for st in self.stretches) + f": {self.est_s:.1f} s"
                + ("" if self.dest is None else f", then on to x {self.dest[1]:.0f} on {self.dest[0]}"))

    def record(self) -> dict:
        """The 'loot_chain' record's: each stretch's platform, bunches (x, drops), whether she ran over all of it, and whether she'd been
        through it before (`again`: what a run through drops left behind, the first live runs' count of RUN_PRESSES)."""
        return {"stops": [[st.platform, [[round(x), n] for x, n in st.bunches], st.done, st.again] for st in self.stretches],
                "planned": len(self.stretches), "done": sum(st.done for st in self.stretches),
                "est_s": round(self.est_s, 2), "dest": None if self.dest is None else [self.dest[0], round(self.dest[1])]}


def _run_through(way: list[tuple[int, float, float]], platform: int, x: float) -> list[tuple[int, float, float]]:
    """`way` from (platform, x) with a goal she only runs past (no stand there) dropped where the walk on from it goes on the same way."""
    out: list[tuple[int, float, float]] = []
    for p, gx, stand in way:
        if out and out[-1][0] == p and out[-1][2] <= 0.0:
            before = out[-2][1] if len(out) >= 2 and out[-2][0] == p else x if p == platform and len(out) == 1 else None
            if before is not None and (out[-1][1] - before) * (gx - out[-1][1]) >= 0.0:
                out[-1] = (p, gx, stand)
                continue
        out.append((p, gx, stand))
    return out
