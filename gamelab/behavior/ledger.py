"""The hunt's bookkeeping, kept up from every look, whoever takes it (Senses.listeners), and from every
throw (Actions.throw_listeners): the kill ledger and the drop store.

- **The ledger** (`Ledger`): every EXP gain the hunter sees (at most one a look: her EXP number only says
  it changed) is a kill. It's placed (kind, platform, where) when a star of hers explains it: a monster
  that can have taken the attack of a throw made in the last STAR_WINDOW s (or, with none, that long before
  the look before, as the gain came after it; any of them: which of the claw's stances an attack takes is
  drawn, and the client's order within a band of the star's search can't be seen), preferring one that's
  gone from view, the newest throw's and the likeliest first.
  A gain no throw explains (party EXP, or a kill seen only from a look without monsters) still counts,
  credited to where she is, but has no place and so no drops.
- **The drop store** (`DropStore`): every pile she knows of, with when its kill happened (not when it was
  first seen), when it was last seen, whether a kill of hers explains it, and its status. A pile is gone
  as the Looter forgets drops: never before SETTLE after the kill, and only after two empty looks
  EMPTY_LOOKS apart, because monsters are drawn over drops. Gone piles are dropped from the store
  GONE_KEEP_S later; what the hunt loop counts of them is kept in counters (`gone_tried`).

Step 4 of gamelab/docs/hunt-modes.md: these are observers. Nothing reads them to decide yet; the
simulator checks them against its world (tests/test_ledger.py), and the hunt loop will read them.
"""

from dataclasses import dataclass, field

from maplehelper.engine import RIGHT

from ..attack import reach
from ..localization.monsters import Monster, same
from ..localization.scene import Loot, Scene
from . import looting
from .looting import EMPTY_LOOKS, LOOT_REACH, SETTLE, standing

STAR_WINDOW = 2.5  # s after a throw's attack key that an EXP gain can still be its kill: 520 ms to the
                   # release, up to 0.65 s of flight (434 units at 690 a second), and up to a look later
DROP_LANDS = 2.0   # s after a kill that its drops lie on the ground (recorded at 2 fps)
# s a pile seen gone stays in the store before it's dropped. Nothing reads a gone pile but the hunt loop's
# count of the ones she went for (kept in DropStore.gone_tried), so this only bounds the store: kept
# forever, every kill added a pile, and every look went through all of them.
GONE_KEEP_S = 300.0


@dataclass
class Kill:
    """One EXP gain of hers, and what it was if a throw explains it."""
    t: float                  # the look that saw her EXP go up
    kind: str | None          # None: no throw of hers explains it (no place, no drops)
    platform: int | None      # where the monster died ...
    x: float
    y: float
    ground: int | None        # ... and the platform the kill is credited to (hers, when unplaced)

    @property
    def placed(self) -> bool:
        return self.kind is not None


@dataclass
class _Throw:
    t: float                  # when its attack key went down
    side: str
    path: list[Monster]       # what can have taken its attack then, the likeliest first (from the last look
                              # with monsters)


class Ledger:
    """Her kills for the whole hunt: every EXP gain, placed where a throw of hers explains it."""

    def __init__(self, hunter, store: "DropStore | None" = None):
        self.h = hunter
        self.store = store
        self.kills: list[Kill] = []
        self._accounted = hunter.exp_gains  # gains already in the ledger
        self._throws: list[_Throw] = []
        self._last: Scene | None = None     # the last look that searched for monsters
        self._looked: float | None = None   # when the last look of all was

    @property
    def placed(self) -> list[Kill]:
        return [k for k in self.kills if k.placed]

    def on_throw(self, when: float, side: str) -> None:
        """A throw (Actions.throw_listeners): what's in its path, from the last look with monsters."""
        s = self._last
        path = [] if s is None or s.where is None or s.platform is None else in_path(s, side, s.monsters)
        self._throws.append(_Throw(when, side, path))

    def on_look(self, s: Scene) -> None:
        """After every look (after the hunter's own count): each new EXP gain becomes a kill. Throws are kept
        STAR_WINDOW s, or that long before the look before if it's older: a gain seen now came after it."""
        since = s.t if self._looked is None else min(s.t, self._looked)
        self._looked = s.t
        self._throws = [th for th in self._throws if since - th.t <= STAR_WINDOW]
        new, self._accounted = self.h.exp_gains - self._accounted, self.h.exp_gains
        for _ in range(new):
            self.kills.append(self._place(s))
        if s.searched:
            self._last = s

    def _place(self, s: Scene) -> Kill:
        """The kill a gain seen in look `s` was: a monster that can have taken the attack of a throw made in
        the last STAR_WINDOW s (its path), preferring one not seen any more (the newest throw first, the
        likeliest of its path first); with none, of a throw made up to STAR_WINDOW s before the look before,
        which the gain came after; else unplaced. A fight's last star often lands after its last look, and the
        walks and pick-ups of the sweep after it take no look for up to 7 s: in the simulator, 2 of
        tour_hunt_basin's 61 kills went unplaced that way, and 9 of 58 once its monsters walked as fast as in
        the game. Going back that far for every gain placed more kills on the wrong monster (458 of 612 right
        in 10 scenarios, against 437 of 628)."""
        recent = [th for th in self._throws if s.t - th.t <= STAR_WINDOW and th.path]
        firsts = [(th, m) for th in reversed(recent or self._throws) for m in th.path]
        if s.searched:
            gone = [(th, m) for th, m in firsts if not any(same(o, m) for o in s.monsters)]
            firsts = gone or firsts
        here = None if s.platform is None else s.platform.id
        if not firsts:
            x, y = (s.where.x, s.where.y) if s.where is not None else (0.0, 0.0)
            return Kill(s.t, None, None, x, y, here)
        th, m = firsts[0]
        self._throws.remove(th)  # one star, one kill
        kill = Kill(s.t, m.kind, m.platform, m.x, m.y, m.platform if m.platform is not None else here)
        if self.store is not None:
            self.store.on_kill(kill)
        return kill


def in_path(s: Scene, side: str, monsters: list[Monster]) -> list[Monster]:
    """The monsters an attack from the ground that way from where she stands in look `s` can take, the
    likeliest first (attack.reach.take_chances, by each kind's body: the nearest the claw's stance reaches,
    else the star's search's first band, any monster in it), and none behind a wall of her foothold group
    (a kill after a throw from 4 isn't put on a monster on 2)."""
    x, y = s.where.x, s.where.y
    ahead = [m for m in monsters if (m.x > x) == (side == RIGHT) and reach.wall_to(s.platform, x, y, m) is None]
    chances = reach.take_chances([(abs(m.x - x), m.y - y, *reach.body_of(m.kind)) for m in ahead])
    return [m for c, m in sorted(zip(chances, ahead), key=lambda cm: -cm[0]) if c > 0]


@dataclass
class Pile:
    """Drops lying together (or about to land) somewhere on the map."""
    x: float
    y: float
    platform: int
    since: float              # when its kill happened (hers), or when it was first seen
    hers: bool                # a kill of hers explains it
    mesos: bool = False       # mesos were seen in it
    last_seen: float | None = None
    empty_since: float | None = None  # when a look first saw it empty (see EMPTY_LOOKS)
    status: str = "expected"  # expected (a kill's, not seen yet) | seen | gone (seen empty twice)
    tries: int = 0            # times the hunt loop went for it (DropStore.tried)
    gone_at: float | None = None  # when it was seen gone


@dataclass
class DropStore:
    """Every pile she knows of. Fed by placed kills (Ledger) and by every look that searched for loot."""
    piles: list[Pile] = field(default_factory=list)  # the piles not gone, and those gone in the last GONE_KEEP_S
    gone_tried: int = 0  # piles the hunt loop went for that looks have seen gone, ever (dropped ones too)

    @property
    def known(self) -> list[Pile]:
        """The piles not gone."""
        return [p for p in self.piles if p.status != "gone"]

    def tried(self, p: Pile) -> None:
        """The hunt loop went for pile `p` (or gave up on it): one more try. A pile can be tried after a
        look saw it gone (a trip to it then ends), so it's counted in `gone_tried` from its first try."""
        if p.status == "gone" and not p.tries:
            self.gone_tried += 1
        p.tries += 1

    def on_kill(self, kill: Kill) -> None:
        if kill.platform is not None:
            self.piles.append(Pile(kill.x, kill.y, kill.platform, kill.t, hers=True))

    def on_look(self, s: Scene) -> None:
        """A look that searched for loot: drops seen confirm a pile (or start one); a pile in view that two
        looks EMPTY_LOOKS apart see empty, SETTLE after its kill, is gone. With looting.STANDING_ONLY, only looks
        that show her standing see a pile empty (looting.standing: from mid-jump a look's strips miss the drops on
        the ground)."""
        if not s.looked:
            return
        self.piles = [p for p in self.piles if p.gone_at is None or s.t - p.gone_at <= GONE_KEEP_S]
        seen, ground = list(s.loot), not looting.STANDING_ONLY or standing(s)
        for p in self.known:
            if any(_in_pile(p, d) for d in seen):
                p.status, p.last_seen, p.empty_since = "seen", s.t, None
                p.mesos = p.mesos or any(d.mesos for d in seen if _in_pile(p, d))
            elif ground and s.t - p.since >= SETTLE and s.in_view(p.platform, p.x):
                if p.empty_since is None:
                    p.empty_since = s.t
                elif s.t - p.empty_since >= EMPTY_LOOKS:
                    p.status, p.gone_at = "gone", s.t
                    self.gone_tried += p.tries > 0
        for d in seen:
            if not any(_in_pile(p, d) for p in self.known):
                self.piles.append(Pile(d.x, d.y, d.platform, s.t, hers=False, mesos=d.mesos, last_seen=s.t,
                                       status="seen"))


def _in_pile(p: Pile, d: Loot) -> bool:
    return p.platform == d.platform and abs(p.x - d.x) <= LOOT_REACH

