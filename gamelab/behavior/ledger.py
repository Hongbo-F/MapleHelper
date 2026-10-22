"""The hunt's bookkeeping, kept up from every look, whoever takes it (Senses.listeners), and from every
throw (Actions.throw_listeners): the kill ledger and the drop store.

- **The ledger** (`Ledger`): every kill the hunter's EXP gains tell of is a kill (a gain's size names its kind, or
  says it was two kills, or none: localization.gains). It's placed (kind, platform, where) when a star of hers
  explains it: a monster of the kind named (when one was) that can have taken the attack of a throw made in the
  last STAR_WINDOW s (or, with none, that long before the look before, as the gain came after it; any of them:
  which of the claw's stances an attack takes is drawn, and the client's order within a band of the star's search
  can't be seen), preferring one that's gone from view (and of those, one with another player by it last:
  PLAYER_KILLS), the newest throw's and the likeliest first; where that monster was seen last (a throw's path is
  followed from look to look). A kill no throw explains (a kill seen only from a look without monsters) still
  counts, credited to where she is, by the kind its gain named, but has no place and so no drops.
- **The drop store** (`DropStore`): every pile she knows of, with when its kill happened (not when it was
  first seen), when it was last seen, whether a kill of hers explains it, what the last look that saw it saw
  in it and what that's worth (looting.worth: the game data's prices), and its status. A pile is gone
  as the Looter forgets drops: never before SETTLE after the kill, and only after two empty looks
  EMPTY_LOOKS apart, because monsters are drawn over drops. Gone piles are dropped from the store
  GONE_KEEP_S later; what the hunt loop counts of them is kept in counters (`gone_tried`).
- **Her wounds** (`Wounds`): the monsters she has hurt and not killed. A monster in the path of a throw of
  hers (as the ledger has it) whose HP bar shows up or changes in a look within STAR_WINDOW s of the throw is
  hers, and followed look to look until a kill of hers is placed on it, it's out of sight for WOUND_LOST_S,
  or WOUND_S have gone by since it was first hurt (chased too long). The fights and the hunt loop finish
  these before they move on (behavior.fight.owed): she used to leave monsters half hunted.

Step 4 of gamelab/docs/hunt-modes.md: these are observers, kept up whoever looks; the simulator checks the
ledger against its world (tests/test_ledger.py), and the hunt loop reads the drop store and the wounds.
"""

from dataclasses import dataclass, field

from maplehelper.engine import RIGHT

from ..attack import reach
from ..localization.monsters import TRACK_SLACK, Monster, same, seen_again
from ..localization.players import near_rival
from ..localization.scene import Loot, Scene
from . import looting
from .looting import COINS, EMPTY_LOOKS, LOOT_REACH, SETTLE, standing, worth

STAR_WINDOW = 2.5  # s after a throw's attack key that an EXP gain can still be its kill: 520 ms to the
                   # release, up to 0.65 s of flight (434 units at 690 a second), and up to a look later
DROP_LANDS = 2.0   # s after a kill that its drops lie on the ground (recorded at 2 fps)
# s a pile seen gone stays in the store before it's dropped. Nothing reads a gone pile but the hunt loop's
# count of the ones she went for (kept in DropStore.gone_tried), so this only bounds the store: kept
# forever, every kill added a pile, and every look went through all of them.
GONE_KEEP_S = 300.0
# What a pile no look has seen yet is worth (Pile.value), in mesos: a kill's pile is coins most often (63-80% of
# each kind's piles seen where her kills died on 2026-10-21; looting.COINS), and its item is worth less than they are
# but for the Slime Bubble (looting.PRICES).
PILE_WORTH = COINS["bronze"]
# Whether a gain goes on a monster gone from view by another player (localization.players.near_rival) only when no
# other a throw of hers explains has gone: one gone by them, they likelier killed. In the simulator's hunt loop (preset
# game, 20 seeds of 20 minutes) her throws' paths rarely held the other player's kills: of about 4,200 kills, 9 were
# placed where its kill was without this, 8 with it, and as many placed right (67%); EXP a minute 129.93 -> 129.76
# (2 seeds' runs changed), the tour's runs the same. On for the game, where another player may hunt beside her.
PLAYER_KILLS = True
# Whether a kill whose gain named a kind (localization.gains) that none of her recent throws' paths holds goes on the
# likeliest monster of any kind (True), or unplaced (False: no pile expected; its kind still counted). Over the three
# basin scenarios' seeds 1-10 (30 runs, 1,111 kills): with the kind placed only on one of that kind, 90% of the kills
# were placed and 96% of those right (kind, platform, within 2.5 s of the world's kill), 86% of all; with any kind as
# well, 99.5% placed and 86% of those right, 85% of all; without the kind (before her EXP number was read), 99.9%
# and 70%.
KIND_ANY = False
# Her wounds (Wounds). A bar new, or this much other than at the throw, is a hit (fight.HURT). A wound unseen this long
# is gone: a hit monster hides under its damage numbers for 1-3 s (fight.HIDDEN_FOR is how long she keeps
# throwing at one unseen; seen again within this, it's still hers), and seen without its bar it's still kept
# (in the live looks of 2026-10-21 a bar seen was gone at the next look of the monster 70 times, 21 of them
# with no kill after). Chased this long from the look that first saw it hurt, it's no longer owed: in the
# simulator's game-like world (20 minutes, 30 seeds) she killed half of the monsters she hurt within 5.4 s of
# the first hit and 84% within 20 s.
WOUND_HURT = 0.02
WOUND_LOST_S = 3.0
WOUND_S = 15.0


@dataclass
class Kill:
    """One kill her EXP told of, and what it was if a throw explains it."""
    t: float                  # the look that saw her EXP go up
    kind: str | None          # None: no throw of hers explains it (no place, no drops)
    platform: int | None      # where the monster died ...
    x: float
    y: float
    ground: int | None        # ... and the platform the kill is credited to (hers, when unplaced)
    named: str | None = None  # the kind her EXP gain's size named (None: not known)

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
        for kind in self.h.kinds_told(new):
            self.kills.append(self._place(s, kind))
        if s.searched:
            self._last = s
            # What can have taken a throw's attack is followed from look to look (as Wounds follows them), so a
            # kill is placed, and its pile expected, where its monster was seen last, not where it was at the
            # throw: a slime shot from 16 walked 90 units on 17 before it died, and the drops seen where it fell,
            # past the pile's LOOT_REACH, were taken for someone else's (the hunt loop waits for those).
            for th in self._throws:
                th.path = [seen_again(m, s.monsters) or m for m in th.path]

    def _place(self, s: Scene, kind: str | None = None) -> Kill:
        """The kill a gain seen in look `s` was (of `kind`, when its size named one): a monster of that kind that
        can have taken the attack of a throw made in the last STAR_WINDOW s (its path), preferring one not seen any
        more (the newest throw first, the likeliest of its path first); with none, of a throw made up to STAR_WINDOW
        s before the look before, which the gain came after; else unplaced (or, with KIND_ANY, of any kind). A
        fight's last star often lands after its last look, and the walks and pick-ups of the sweep after it take no
        look for up to 7 s: in the simulator, 2 of tour_hunt_basin's 61 kills went unplaced that way, and 9 of 58 once
        its monsters walked as fast as in the game. Going back that far for every gain placed more kills on the
        wrong monster (458 of 612 right in 10 scenarios, against 437 of 628)."""
        firsts = self._candidates(s.t, kind)
        if not firsts and kind is not None and KIND_ANY:
            firsts = self._candidates(s.t, None)
        if s.searched:
            gone = [(th, m) for th, m in firsts if not any(same(o, m) for o in s.monsters)]
            if PLAYER_KILLS and s.players:  # (a stable sort: the newest throw's and the likeliest first as before)
                gone.sort(key=lambda tm: near_rival(s.players, tm[1].x, tm[1].y) is not None)
            firsts = gone or firsts
        here = None if s.platform is None else s.platform.id
        if not firsts:
            x, y = (s.where.x, s.where.y) if s.where is not None else (0.0, 0.0)
            return Kill(s.t, None, None, x, y, here, kind)
        th, m = firsts[0]
        self._throws.remove(th)  # one star, one kill
        kill = Kill(s.t, m.kind, m.platform, m.x, m.y, m.platform if m.platform is not None else here, kind)
        if self.store is not None:
            self.store.on_kill(kill)
        return kill

    def _candidates(self, t: float, kind: str | None) -> list[tuple[_Throw, Monster]]:
        """What a gain seen at t can have been (of `kind`, if not None): the monsters in the paths of the throws of
        the last STAR_WINDOW s, else of those kept (see _place), the newest throw's first, each path's likeliest
        first."""
        ok = (lambda m: True) if kind is None else (lambda m: m.kind == kind)
        recent = [th for th in self._throws if t - th.t <= STAR_WINDOW and any(ok(m) for m in th.path)]
        return [(th, m) for th in reversed(recent or self._throws) for m in th.path if ok(m)]


@dataclass
class Wound:
    """A monster she has hurt and not seen die: as a look last saw it (`m`), when a look first saw it hurt by
    her (`since`), and when one last saw it (`seen`)."""
    m: Monster
    since: float
    seen: float


class Wounds:
    """The monsters she has hurt and not killed (see the module's docstring), kept up from every look that
    searched for monsters and every throw. `ledger`: its kills close the wounds they're placed on."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.open: list[Wound] = []
        self._paths: list[tuple[float, list[Monster]]] = []  # recent throws: when, and their path, followed
        self._last: Scene | None = None   # the last look that searched for monsters
        self._kills = len(ledger.kills)   # the ledger's kills already gone through

    def on_throw(self, when: float, side: str) -> None:
        """A throw (Actions.throw_listeners): what's in its path, from the last look with monsters (in_path)."""
        s = self._last
        if s is not None and s.where is not None and s.platform is not None:
            self._paths.append((when, in_path(s, side, s.monsters)))

    def on_look(self, s: Scene) -> None:
        """After every look (after the ledger's): a monster in a recent throw's path with its bar new or changed
        is hers; her wounds are followed; a kill placed on one closes it, and so do WOUND_LOST_S out of sight
        and WOUND_S since it was first hurt."""
        for k in self.ledger.kills[self._kills:]:
            if k.placed:
                self.open = [w for w in self.open if not (w.m.kind == k.kind and w.m.platform == k.platform
                                                          and abs(w.m.x - k.x) < TRACK_SLACK)]
        self._kills = len(self.ledger.kills)
        if not s.searched or s.where is None:
            return
        self._last = s
        self._paths = [(t, path) for t, path in self._paths if s.t - t <= STAR_WINDOW]
        for _, path in self._paths:
            for i, m in enumerate(path):
                if (again := seen_again(m, s.monsters)) is None:
                    continue
                # Its bar new, or other than it was (two of a kind near each other can be taken for each other
                # from look to look: a higher one is still one in her star's path that she has hurt)
                if again.bar is not None and (m.bar is None or abs(again.bar - m.bar) > WOUND_HURT) \
                        and not any(seen_again(w.m, [again]) is again for w in self.open):
                    self.open.append(Wound(again, s.t, s.t))
                path[i] = again
        hurt = [o for o in s.monsters if o.bar is not None]  # a hurt one shows its bar: those first
        for w in self.open:
            if (again := seen_again(w.m, hurt) or seen_again(w.m, s.monsters)) is not None:
                w.m, w.seen = again, s.t
        self.open = [w for w in self.open if s.t - w.seen < WOUND_LOST_S and s.t - w.since < WOUND_S]

    def seen_within(self, now: float, seconds: float, hurt_by: float | None = None) -> list[Monster]:
        """Her wounds as last seen, those seen within `seconds` of `now` (and first hurt by `hurt_by`, if given)."""
        return [w.m for w in self.open if now - w.seen <= seconds and (hurt_by is None or w.since <= hurt_by)]


def in_path(s: Scene, side: str, monsters: list[Monster]) -> list[Monster]:
    """The monsters an attack from the ground that way from where she stands in look `s` can take, the
    likeliest first (attack.reach.take_chances, by each kind's body: the nearest the claw's stance reaches,
    else the star's search's first band, any monster in it, whatever the terrain), and of those none whose
    star terrain stops (reach.wall_to: a kill after a throw from 4 isn't put on a monster on 2). The client
    checks the line to its pick alone, and stopped, no other monster takes the star: so with the first band's
    only monster on 2, there's none (one the claw reaches is swiped whatever the terrain)."""
    x, y = s.where.x, s.where.y
    ahead = [m for m in monsters if (m.x > x) == (side == RIGHT)]
    bodies = [(abs(m.x - x), m.y - y, *reach.body_of(m.kind)) for m in ahead]
    chances = reach.take_chances(bodies)
    return [m for c, m, b in sorted(zip(chances, ahead, bodies), key=lambda cmb: -cmb[0])
            if c > 0 and (reach.claw_chance(*b) > 0 or reach.wall_to(s.platform, x, y, m) is None)]


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
    rushed: int = 0           # ... of those, at once for what it's worth (hunt's rule 6a)
    gone_at: float | None = None  # when it was seen gone
    items: tuple[str, ...] = ()   # what the latest look that saw it saw in it (one kind per drop) ...
    worth: float | None = None    # ... what those are worth together (looting.worth); None until a look sees it ...
    dearest: float | None = None  # ... and the dearest of them on its own

    @property
    def value(self) -> float:
        """What it's worth, in mesos: what was seen in it last, or PILE_WORTH before it's been seen."""
        return PILE_WORTH if self.worth is None else self.worth


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
            if inside := [d for d in seen if _in_pile(p, d)]:
                p.status, p.last_seen, p.empty_since = "seen", s.t, None
                p.mesos = p.mesos or any(d.mesos for d in inside)
                p.items, p.worth, p.dearest = _contents(inside)
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
                new = self.piles[-1]
                new.items, new.worth, new.dearest = _contents([e for e in seen if _in_pile(new, e)])


def _in_pile(p: Pile, d: Loot) -> bool:
    return p.platform == d.platform and abs(p.x - d.x) <= LOOT_REACH


def _contents(seen: list[Loot]) -> tuple[tuple[str, ...], float, float]:
    """The drops seen in a pile: their kinds, what they're worth together (looting.worth), and the dearest's worth."""
    each = [worth(d.item.kind, d.item.coin) for d in seen]
    return tuple(d.item.kind for d in seen), float(sum(each)), float(max(each, default=0.0))

