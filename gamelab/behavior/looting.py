"""Looting: picking up what her kills drop, and remembering drops she can't pick up yet (the Looter).

Drops land about 2 s after a kill (the death animation, then they pop up and fall), spread about
LOOT_REACH around where the monster died, and monsters are drawn in front of them. So after a fight
she sweeps where each kill died, then looks again and goes back for what's left (`mop_up`). Drops on
other platforms are remembered (`pending`), for the tour to collect. A drop she has stood on twice,
pressing pick-up, that's still there is someone else's (drops belong to the killer for a while), or her
inventory is full: the game says so in the pickup log, and then she only takes mesos.
"""

import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from maplehelper.engine import LEFT, RIGHT

from ..attack.reach import EDGE_MARGIN
from ..attack.stance import TOO_CLOSE
from ..localization.monsters import Monster
from ..localization.scene import Loot, Scene
from ..perception.loot import says_full

if TYPE_CHECKING:
    from .hunter import Hunter

LOOT_REACH = 60    # drops spread about this far around where the monster died
PILE_GAP = 15      # drops closer together than this are one pile: picked up standing in one place
STOOD_ON = 15      # she stood on a drop when she stopped within this of it (walks stop ~10 units off)
MOP_UP = 3         # looks after a sweep for drops still on her platform, going back for them
SETTLE = 3.0       # s a kill's drops take to land and show: recorded at 2 fps, they lay on the ground about
                   # 2 s after her EXP went up (the death animation, then they pop up and fall)
EMPTY_LOOKS = 1.5  # s between two looks that must both see a remembered drop spot empty before it's
                   # forgotten: monsters are drawn over drops, and a pig walking over coins hid them
UNPICKABLE_FOR = 90  # s drops she failed to pick up are left alone (they vanish after a while anyway)
FAILED_PICKS = 3   # items in a row that stayed under her with no item picked up: her inventory is full
DROPS_LAST = 90    # s remembered drops are worth going back for (they vanish, and others may take them)
MIN_WALK = 10      # a sweep's stops nearer to her than this aren't walked to: she's there


@dataclass
class DropSpot:
    """Where drops lie or will land, to be picked up: where a kill of hers died, or drops seen on the
    ground."""
    x: float
    y: float
    platform: int | None
    since: float
    mesos: bool = False      # mesos were seen there (still worth a trip with her inventory full)
    on_ground: bool = False  # seen on the ground (else: where a kill of hers died)
    empty_since: float | None = None  # when a look first saw the spot empty (see EMPTY_LOOKS)


class Looter:
    """What she picks up, what she remembers for later (`pending`), and what she can't pick up."""

    def __init__(self, hunter: "Hunter"):
        self.h = hunter
        self.pending: list[DropSpot] = []  # drops off her platform, left for later (the tour collects them)
        self.inventory_full = False  # pick-up failed for want of room: items are left, mesos still taken
        self.unpickable: list[tuple[float, int, float]] = []  # (x, platform, when) of drops she couldn't take
        self._failed_picks = 0  # items in a row that stayed under her (see FAILED_PICKS)

    def wanted(self, loot: list[Loot]) -> list[Loot]:
        """The drops worth going for: not ones she failed to pick up, and only mesos with her inventory full."""
        now = time.monotonic()
        self.unpickable = [u for u in self.unpickable if now - u[2] < UNPICKABLE_FOR]
        return [d for d in loot if (d.mesos or not self.inventory_full)
                and not any(p == d.platform and abs(x - d.x) <= STOOD_ON for x, p, _ in self.unpickable)]

    def remember(self, s: Scene) -> None:
        """Note what a look (with `loot`) saw, for the tour to collect: drops seen are remembered (a pile per
        LOOT_REACH). A remembered spot is forgotten (taken by someone, or by her on the way) once its drops
        have had time to land (SETTLE) and two looks EMPTY_LOOKS apart saw nothing there: one look can
        miss drops a monster stands in front of."""
        now = time.monotonic()
        seen = self.wanted(s.loot)
        kept = []
        for d in self.pending:
            if now - d.since < SETTLE or not s.in_view(d.platform, d.x) or any(_in_pile(d, loot) for loot in seen):
                d.empty_since = None  # still landing, out of view, or drops seen there
            elif d.empty_since is None:
                d.empty_since = now   # seen empty once
            elif now - d.empty_since >= EMPTY_LOOKS:
                continue              # seen empty twice: forgotten
            kept.append(d)
        if gone := len(self.pending) - len(kept):
            self.h.log(f"  {gone} remembered drop spot(s) seen empty twice: forgotten")
        self.pending = kept
        self.note(seen)

    def note(self, seen: list[Loot]) -> None:
        """Remember drops seen, a pile per LOOT_REACH, for later (the tour collects them)."""
        for loot in seen:
            pile = next((d for d in self.pending if _in_pile(d, loot)), None)
            if pile is None:
                self.pending.append(DropSpot(loot.x, loot.y, loot.platform, time.monotonic(), loot.mesos,
                                             on_ground=True))
            else:
                pile.mesos = pile.mesos or loot.mesos

    def collect(self, drops: DropSpot) -> bool:
        """Pick up the drops now if they're on her platform and no monster stands by them; otherwise
        remember them (pending) for later. Drops seen around the spot are walked to directly. Where
        a kill died she sweeps even when nothing shows: drops can still be falling, or hidden behind a
        monster, and the trip there cost far more. Drops seen earlier that are gone are done with.
        Returns True when done with them."""
        s = self.h.look(loot=True)
        if s.where is None or s.platform is None or drops.platform != s.platform.id:
            self.pending.append(drops)
            return False
        if _guarded(drops.x, drops.platform, s.monsters):
            self.h.log("  drops guarded by a monster: later")
            self.pending.append(drops)
            return False
        seen = [d for d in self.wanted(s.loot) if d.platform == drops.platform
                and abs(d.x - drops.x) <= 2 * LOOT_REACH]
        if seen:
            self.sweep_through(s, [DropSpot(d.x, d.y, d.platform, s.t) for d in seen], margin=0)
        elif drops.on_ground and time.monotonic() - drops.since >= SETTLE and s.in_view(drops.platform, drops.x):
            self.h.log(f"  the drops seen at x {drops.x:.0f} are gone")
        else:
            self.sweep_through(s, [drops])
        return True

    def loot_pending(self, s: Scene | None = None) -> None:
        """Sweep remembered drops that are on her platform (loot() re-queues any still guarded); the
        rest stay remembered. `s` is a recent look, to tell which ones are on her platform."""
        s = s or self.h.look(mobs=False)
        here = None if s.platform is None else s.platform.id
        fresh = [d for d in self.pending if time.monotonic() - d.since < DROPS_LAST]  # drops vanish
        self.pending = [d for d in fresh if d.platform != here]
        for d in fresh:
            if d.platform == here:
                self.collect(d)

    def sweep_after(self, spots: list[DropSpot]) -> None:
        """Sweep up a fight's drops once it's over. The fight often ends mid-jump (a jump-throw), where
        a look finds her on no platform, and the sweep used to be skipped: those drops were left. Looks
        again until she has landed; drops on another platform than hers are left to the tour."""
        for _ in range(8):
            s = self.h.look(mobs=False)
            if s.where is not None and s.platform is not None:
                self.pending += [d for d in spots if d.platform != s.platform.id]
                self.sweep_through(s, spots)
                return
            time.sleep(0.15)
        self.pending += spots

    def sweep_through(self, s: Scene, spots: list[DropSpot], margin: float = LOOT_REACH) -> None:
        """Pick up the drops on her platform (`walk_piles`), then look again for any still there and go
        back for them (`mop_up`)."""
        self.mop_up(self.walk_piles(s, spots, margin))

    def walk_piles(self, s: Scene, spots: list[DropSpot], margin: float) -> list[float]:
        """Walk over the drops on her platform: from the nearer end of them to the farther one (and
        `margin` on past each end: a kill's drops spread around where it died), pressing pick-up all
        the way, and standing on each pile for a few presses. A pick-up press takes one item, a pig
        drops several, and walking past at full speed got one or two. Returns where she stood."""
        p = s.platform

        def clamp(v: float) -> float:  # never past her platform's ends
            return min(max(v, p.left + EDGE_MARGIN), p.right - EDGE_MARGIN)

        piles = _piles([clamp(d.x) for d in spots if d.platform == p.id])
        if not piles:
            return []
        x = s.where.x
        if abs(x - piles[-1]) < abs(x - piles[0]):
            piles.reverse()
        first, last = piles[0], piles[-1]
        sign = 1 if last >= first else -1
        stops = ([(clamp(first - sign * margin), False)] + [(g, True) for g in piles]
                 + [(clamp(last + sign * margin), False)])  # (where, a pile to stand on)
        for goal, pile in stops:
            dx = goal - x
            if abs(dx) >= MIN_WALK:
                self.h.actions.walk(RIGHT if dx > 0 else LEFT, abs(dx))
                x = goal
            if pile:
                self.h.actions.pick_here()
        self.h.report.loot_runs += 1
        return piles

    def mop_up(self, stood: list[float] | None = None) -> None:
        """Look again after a sweep for drops still on her platform, and go back for them: walking past
        at full speed, or stopping a little off a pile, left some. A drop she has stood on twice that's
        still there can't be picked up (`cant_pick`); drops with a monster by them wait for later.
        `stood`: where the sweep just had her stand picking up."""
        stood, here, tried = list(stood or []), None, []  # tried: items she went back for last round
        for _ in range(MOP_UP):
            time.sleep(0.3)  # a picked-up drop flies to her for a moment
            s = self.h.look(loot=True)
            if s.where is None or s.platform is None or (here is not None and s.platform.id != here):
                return  # in the air, or knocked off: whatever is left, the tour sees
            here = s.platform.id
            seen = [d for d in s.loot if d.platform == here]
            if any(_picked_up(t, seen) for t in tried):
                self._failed_picks = 0  # an item she went back for is gone: she picked it up
            left = self.wanted(seen)
            stuck = [d for d in left if sum(abs(x - d.x) <= STOOD_ON for x in stood) >= 2]
            if stuck:
                self.cant_pick(stuck)
                left = [d for d in left if d not in stuck]
            guarded = [d for d in left if _guarded(d.x, here, s.monsters)]
            if guarded:
                self.h.log(f"  {len(guarded)} drop(s) by a monster: later")
                self.note(guarded)
                left = [d for d in left if d not in guarded]
            if not left:
                return
            self.h.log(f"  {_summary(left)} still on platform {here}: back for them")
            tried = [d for d in left if not d.mesos]
            stood += self.walk_piles(s, [DropSpot(d.x, d.y, here, s.t) for d in left], margin=0)

    def cant_pick(self, stuck: list[Loot]) -> None:
        """Drops she stood on twice, pressing pick-up, that are still there: left alone for a while. Mesos
        always fit, so those are someone else's (drops belong to the killer for a while). Items may be
        too, unless the pickup log says her inventory is full, or items keep staying under her
        (FAILED_PICKS times in a row, with none picked up in between): then items are left, mesos taken."""
        now = time.monotonic()
        self.unpickable += [(d.x, d.platform, now) for d in stuck]
        self.h.log(f"  can't pick up {_summary(stuck)} at x {', '.join(f'{d.x:.0f}' for d in stuck)}: left")
        if self.inventory_full or all(d.mesos for d in stuck):
            return
        self._failed_picks += 1
        text = self.h.senses.eyes.messages()
        if says_full(text):
            why = f"the game says so: {' / '.join(line for line in text.splitlines() if says_full(line))}"
        elif self._failed_picks >= FAILED_PICKS:
            why = f"items stayed under her {self._failed_picks} times in a row"
        else:
            return
        self.inventory_full = True
        self.h.log(f"  her inventory is full ({why}): leaving items, picking up mesos only")


def _in_pile(spot: DropSpot, loot: Loot) -> bool:
    """Whether drops seen (`loot`) are in a remembered spot's pile: on its platform, within LOOT_REACH."""
    return spot.platform == loot.platform and abs(spot.x - loot.x) <= LOOT_REACH


def _guarded(x: float, platform: int, monsters: list[Monster]) -> bool:
    """Whether a monster stands by drops at map x on `platform` (nearer than TOO_CLOSE): they wait."""
    return any(m.platform == platform and abs(m.x - x) < TOO_CLOSE for m in monsters)


def _picked_up(item: Loot, seen: list[Loot]) -> bool:
    """Whether an item she went back for is gone: no item is seen where it lay any more."""
    return not any(abs(d.x - item.x) <= STOOD_ON for d in seen if not d.mesos)


def _piles(xs: list[float]) -> list[float]:
    """Drop spots closer together than PILE_GAP made one pile (their middle), in order."""
    piles: list[list[float]] = []
    for x in sorted(xs):
        if piles and x - piles[-1][-1] < PILE_GAP:
            piles[-1].append(x)
        else:
            piles.append([x])
    return [round(sum(p) / len(p)) for p in piles]


def _summary(loot: list[Loot]) -> str:
    """Drops by kind: "2 mesos, 1 leather"."""
    return ", ".join(f"{n} {k}" for k, n in Counter(d.item.kind for d in loot).items())
