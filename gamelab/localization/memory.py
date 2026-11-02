"""A map-wide memory of the monsters she has seen (Memory): every look's monsters, wherever the look searched for them
(her band, a slice of the sweep, the whole screen: Scene.rows), kept from look to look, so what a look didn't search
isn't forgotten. One look's monsters are only what that look searched: in the live hunts of 2026-10-24 on Tree Dungeon,
Forest Up North II, 90% of her looks searched only the band around her platform, and monsters on other platforms were
seen only on the whole-screen looks, a median 4.9 s apart (up to 22.8 s).

Each monster remembered (Remembered) has its kind, its platform and x, when it was seen last, how many looks have
covered its place since without it, and whether another player is by it. A sighting is the same monster as one
remembered when it's the same kind on the same platform, within MATCH_SLACK of where it was and what it can have walked
since (MATCH_SPEED x its speed x the time): the nearest first. One no entry explains is new.

A look covers a monster's place (Cover.holds) when the monster's whole body there was inside one stretch of rows the
look searched, inside the screen's sides (EDGE) and not under the minimap: the detector finds only bodies wholly inside
what it searches, so a body cut by the view's top edge, the bottom bar or a band's edge is never missed by a look, only
not found. How tall a body is, for that: as tall as it walks (monsters.MONSTER_HEIGHT; a learned kind, the lowest top
of its stand and move frames: MobBody.top) and BODY_SLACK. Taller, as tall as it flinches with a margin (62 units for
a Horny Mushroom), platform 19 on Tree Dungeon, 60 units under the view's top from where she hunts, was never covered:
on the 10 Hz watch run its remembered monsters were never forgotten by looks, only by age, and every respawn there
added one (5.0 at a time on average, where the looks saw 0.93). Across, where it can be is where it can have walked on
its platform since it was seen; COVER_SHARE of that must be covered (it's likelier there than not). All of it, or
three quarters, a monster on platform 30 (its left third 1 unit behind the bottom bar, seen from 27) was never forgotten
by looks once it could have walked there, and was kept up to MEMORY_S (0.84 remembered there on average, where the looks
saw 0.22). Looks while she isn't standing don't count (in the air every monster's y is off by her height).

How sure a remembered monster is still there (`p`, Remembered.p) goes down two ways:
- by evidence: once looks covering its place have missed it, by the time from its last sighting to the latest such
  look, as exp(-d / MISS_S). Measured on the whole-screen looks of `gamelab watch` at 10 a second on 2026-10-24
  (lab/telemetry/20261024-144419-watch.jsonl, 8750 looks, a crowd of other players hunting 19-24 and 29-30): of the
  monsters missing that long with their place covered, 0.84 were seen again after 0.3 s, 0.69 after 1 s, 0.51 after 2 s,
  0.35 after 3 s, 0.27 after 4 s and 0.20 after 5 s; exp(-d / 3) is 0.90, 0.72, 0.51, 0.37, 0.26 and 0.19. A hit
  monster hides under its damage numbers for 1-3 s (the README), and other players' hits do that too. On her hunts'
  looks of that day (band looks, whole-screen ones now and then) the table by covered misses and time was the same
  shape, lower at the start (0.6-0.7 within a second): her own kills were in it, which the memory takes from the
  ledger instead. Covered looks without it don't count while a throw of hers went its way in the last HIDE_S
  (tracking.HIDE_S, from HIT_AFTER_S after its key) and it was seen in that time: it may be hiding under her damage
  numbers. Not seen in that time, they count: in loop_shoots_21_from_22 she kept throwing toward 21, and a monster
  remembered there that wasn't (below) was never missed by a look, and kept her off the drops by it.
- by age: exp(-a / STAY_S), a the time since it was seen. The live telemetry of 2026-10-24 (persistence of a monster
  seen at t seen again there at t + 10 s, normalised by the same at 2-4 s): 0.75-0.86, so 35-66 s; 45 s.
It's forgotten (state "gone") under P_FORGET or MEMORY_S after it was seen, and marked "killed" when the ledger places
a kill of hers on it (`kills`, a callable giving the ledger's kills: t, kind, platform, x). A monster killed fades out
where it died for about a second: a sighting of its kind within FADE_S of the kill where it can be (as far from where it
was seen last as a sighting of it would be matched, or FADE_DX from where the ledger placed the kill) is that one, not
a new one. Only within FADE_DX of the kill's place, its body seen fading 77 units from there (where it had been knocked
to) was taken for a new one in loop_shoots_21_from_22, which nobody had killed after. Ended entries are kept
ENDED_KEEP_S (for the mini-map's crosses), then dropped.

The memory is passive: it takes every look (`on_look`, a Senses listener) and changes nothing she does by itself.
`update` takes plain data, so the same memory runs on a recorded run's 'scene' records (`replay`): what it would have
held at each look. Its telemetry is the topic 'memory' (`record`), every RECORD_EVERY_S at most.
"""

import math
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .. import telemetry
from ..character import THIEF_SHAPE, Shape
from ..perception.mobs import Mob
from ..perception.minimap import MINIMAP_BOX, minimap_box
from ..screen import STRETCHED
from .mapdata import MapModel
from .monsters import DEFAULT_SPEED, MONSTER_HEIGHT, Monster, bodies, speeds
from .players import Player, near_rival
from .scene import Scene
from .tracking import HIDE_S, HIT_AFTER_S, THROW_DY, THROW_REACH

MATCH_SLACK = 40.0  # units a monster seen again may be from where it was (her dot's minimap pixel is 11.4 units,
                    # knockback 18-25) ...
MATCH_SPEED = 1.3   # ... plus this times what it can have walked since at its kind's speed (tracking's own margin)
MATCH_WALK_S = 30.0  # ... over this long at most (by then it can be anywhere on its platform)
EDGE = 30.0         # units inside the screen's sides a body must be all of to count as in view (half a body: the
                    # detector finds only whole ones)
COVER_SHARE = 0.5   # a look covers where a monster can be when it searched this share of it at least (see the top)
COVER_POINTS = 9    # ... judged at this many places along it
BODY_SLACK = 4.0    # units above a body's top (as tall as it walks) that the rows must reach too
DEFAULT_TOP = 60.0  # a kind with no body known: as tall as this (an Orange Mushroom's 55 and a little)
MISS_S = 3.0        # s: by evidence, p = exp(-d / MISS_S) (see the top)
STAY_S = 45.0       # s: by age, p = exp(-a / STAY_S) (see the top)
P_FORGET = 0.15     # forgotten under this (about 5.7 s after it was last seen, with looks covering its place)
MEMORY_S = 60.0     # ... or this long after it was seen, whatever p says
FADE_S = 1.5        # s a killed monster may still be seen fading where it died (tracking.FADE_S) ...
FADE_DX = 30.0      # ... this close to where its kill was placed (or where it can be from its last sighting)
KILL_DX = 150.0     # a kill of hers marks the nearest monster remembered on its platform this close to where the
                    # ledger placed it
ENDED_KEEP_S = 2.0  # s a killed or forgotten monster is kept, ended, for the mini-map
RECORD_EVERY_S = 0.5  # s between two 'memory' records at the least
EVENTS = 400        # the latest new, killed and gone monsters kept (Memory.events), for whoever learns from them
COVERS = 256        # the latest looks' covers kept (Memory.covers), for whoever judges places by them: a trip's looks
                    # come 5-10 a second, and its passes, which read them, can be 20 s apart
SURE = 0.5          # p at which a remembered monster counts as there (Memory.there): about 2 s of covered misses


@dataclass
class Remembered:
    """A monster she has seen, as the memory keeps it."""
    id: int
    kind: str
    platform: int
    x: float              # where it was seen last (map units, its feet)
    y: float
    seen: float           # when (time.monotonic, the look's)
    first: float          # when it was first seen
    monster: Monster      # its last sighting, as placed (for whoever judges reach and guards by it)
    sightings: int = 1
    misses: int = 0       # covered looks without it since it was seen last
    missed: float | None = None  # the latest of those looks
    covered: float | None = None  # the latest look that covered its place, with it seen or not
    rival: bool = False   # another player was by it in the latest look that showed the other players
    state: str = "seen"   # "seen" (remembered), "killed" (by her: the ledger), "gone" (forgotten)
    ended: float | None = None  # when it was killed or forgotten
    died_x: float | None = None  # where the ledger placed its kill (killed)
    moved_from: tuple | None = None  # her kill moved onto it (Memory._take_moves): where it was put first (platform,
                                     # kind, x) and when (event 'moved')

    @property
    def live(self) -> bool:
        return self.state == "seen"

    def p(self, t: float) -> float:
        """How sure it's still there at time t (see the top)."""
        if not self.live:
            return 0.0
        p = math.exp(-max(0.0, t - self.seen) / STAY_S)
        if self.missed is not None:
            p *= math.exp(-max(0.0, self.missed - self.seen) / MISS_S)
        return p

    def age(self, t: float) -> float:
        return t - self.seen

    def record(self, t: float) -> dict:
        return {"id": self.id, "kind": self.kind, "platform": self.platform, "x": round(self.x, 1),
                "y": round(self.y, 1), "age": round(t - self.seen, 2), "p": round(self.p(t), 2),
                "misses": self.misses, "state": self.state,
                **({"covered_age": round(t - self.covered, 2)} if self.covered is not None else {}),
                **({"rival": True} if self.rival else {})}


@dataclass(frozen=True)
class Cover:
    """What one look searched for monsters, in map units: stretches of rows (y from, to; down is +), the screen across
    (x from, to), and the minimap's box (x from, y from, x to, y to: what's under it doesn't show)."""
    t: float
    rows: tuple[tuple[float, float], ...]
    x0: float
    x1: float
    minimap: tuple[float, float, float, float] | None = None

    def holds(self, x: float, feet: float, half: float, top: float) -> bool:
        """Whether a body standing at (x, feet), `half` wide either side and `top` tall, was wholly searched."""
        if x - half < self.x0 + EDGE or x + half > self.x1 - EDGE:
            return False
        head = feet - top - BODY_SLACK
        if not any(lo <= head and feet <= hi for lo, hi in self.rows):
            return False
        mm = self.minimap
        return mm is None or not (x - half < mm[2] and x + half > mm[0] and head < mm[3] and feet > mm[1])


def cover_of(s: Scene) -> Cover | None:
    """What look `s` searched for monsters (Scene.rows), or None: it searched none, or didn't place her standing (in the
    air or on a rope every monster's y is off by her height; Scene.grounded)."""
    if not s.searched or not s.rows or s.where is None or s.tag is None or not s.grounded:
        return None
    w = s.frame.shape[1]
    box = minimap_box(s.geo)
    k = s.geo.look_scale
    mm = (s.map_x(box[0] / k), s.map_y(box[1] / k), s.map_x(box[2] / k), s.map_y(box[3] / k))
    return Cover(s.t, tuple(s.rows), s.map_x(0), s.map_x(w), mm)


class Memory:
    """The monsters she has seen, kept across looks (see the top). `kills`: the ledger's kills so far (behavior.ledger:
    each with t, kind, platform and x, and its track if the feed's tracker said which), read at every look for the new
    ones; None: none told. `dead`: the feed's tracks taken for dead (behavior.ledger.Ledger.dead), whose monsters are
    killed, and `theirs` those of them another player killed (Ledger.theirs): 'taken', not 'killed'. `moved`: her kills
    the tracker put on another monster after the ledger had them (Ledger.moved: `_take_moves`). `shape`: what
    her attacks are to it (gamelab.character.Shape; the thief's: tracking's HIDE_S, HIT_AFTER_S, THROW_REACH,
    THROW_DY)."""

    def __init__(self, model: MapModel, kills: Callable[[], list] | None = None,
                 dead: Callable[[], object] | None = None, theirs: Callable[[], object] | None = None,
                 moved: Callable[[], list] | None = None, shape: Shape = THIEF_SHAPE):
        self.model = model
        self.shape = shape
        self.kills = kills
        self.dead = dead
        self.theirs = theirs
        self.moved = moved
        self.entries: list[Remembered] = []
        self.events: deque[tuple[str, float, Remembered]] = deque(maxlen=EVENTS)  # (what, when, the monster)
        self.covered: dict[int, float] = {}  # platform -> the latest look that covered all of it
        self.covers: deque[Cover] = deque(maxlen=COVERS)  # the latest looks' covers (behavior.patrol's points)
        self.counts = {"looks": 0, "covered": 0, "new": 0, "killed": 0, "taken": 0, "gone": 0, "aged": 0}
        # (every event appended is counted under new, killed, taken, gone or aged: behavior.patrol.Belief reads the
        # events by that sum)
        self._bodies = bodies(model)
        self._speeds = speeds(model)
        self._next_id = 1
        self._kills_seen = 0
        self._moved_seen = 0
        self._throws: deque[tuple[float, int, float, float]] = deque(maxlen=16)  # (t, side +1/-1, her x, her y)
        self._her: tuple[float, float] | None = None
        self._recorded = -1e18
        # Each platform's tallest monster body of the kinds that spawn on it, for whether a look covered it whole.
        self._tallest = {pid: max((self._top(k) for k in kinds), default=DEFAULT_TOP)
                         for pid, kinds in _kinds_by_platform(model).items()}

    # --- what she has seen -----------------------------------------------------------------------------

    def alive(self, t: float, min_p: float = 0.0) -> list[Remembered]:
        """The monsters remembered (not killed or forgotten) at time t, at least `min_p` sure."""
        return [e for e in self.entries if e.live and e.p(t) >= min_p]

    def there(self, t: float, platform: int | None = None, kinds=None, min_p: float = SURE) -> list[Monster]:
        """The monsters (their last sightings, as placed) remembered at time t at least `min_p` sure (SURE: about 2 s of
        looks covering their place without them), on `platform` (None: anywhere), of `kinds` (None: any)."""
        return [e.monster for e in self.entries if e.live and (platform is None or e.platform == platform)
                and (kinds is None or e.kind in kinds) and e.p(t) >= min_p]

    def covered_at(self, platform: int) -> float | None:
        """When a look last covered all of `platform` (any monster that spawns on it wholly searched at every one of
        COVER_POINTS places along it), if one has."""
        return self.covered.get(platform)

    def holds_body(self, cover: Cover, platform: int, x: float, kind: str) -> bool:
        """Whether `cover` wholly searched a monster of `kind` standing on `platform` at x (a spawn point's monster,
        for one)."""
        p = self.model.platform(platform)
        return cover.holds(x, p.y_at(x), self._half(kind), self._top(kind))

    # --- keeping it up ---------------------------------------------------------------------------------

    def on_look(self, s: Scene) -> None:
        """A Senses listener: the look's monsters (band, and the sweep's or the feed's off it: Scene.monsters and
        Scene.swept) into the memory, and its 'memory' record sent (telemetry) every RECORD_EVERY_S at most."""
        if s.where is not None:
            self._her = (s.where.x, s.where.y)
        if not s.searched or s.where is None:
            self._end_old(s.t)
            return
        cover = cover_of(s)
        players = s.players if s.dot is not None else None  # a look without her dot says nothing of theirs
        self.update(s.t, [*s.monsters, *s.swept], cover, players)
        if telemetry.recording() and s.t - self._recorded >= RECORD_EVERY_S:
            self._recorded = s.t
            telemetry.emit("memory", **self.record(s.t))

    def on_throw(self, when: float, side: str, attack: str | None = None) -> None:
        """An Actions throw listener: where she threw from and which way, for the monsters that may be hiding under her
        damage numbers (HIDE_S)."""
        if self._her is not None:
            self._throws.append((when, 1 if side == "right" else -1, *self._her))

    def update(self, t: float, sightings: Iterable[Monster], cover: Cover | None,
               players: list[Player] | None = None) -> None:
        """One look at time t: its monsters (placed; those on no platform are left out), what it covered (None: nothing
        it can be judged by), the other players (None: not known this look)."""
        self.counts["looks"] += 1
        unplaced = self._take_kills(t)
        seen = [m for m in sightings if m.platform is not None]
        live = [e for e in self.entries if e.live]
        fading = [e for e in self.entries if e.state == "killed" and t - e.ended < FADE_S]
        pairs = []
        for j, m in enumerate(seen):
            for e in live:
                if e.platform == m.platform and e.kind == m.kind:
                    d = abs(m.x - e.x)
                    if d <= self._reach(e, t):
                        pairs.append((d, e.id, j, e))
        pairs.sort(key=lambda q: (q[0], q[1], q[2]))
        matched: set[int] = set()
        used: set[int] = set()
        for _, eid, j, e in pairs:
            if eid in matched or j in used:
                continue
            matched.add(eid)
            used.add(j)
            m = seen[j]
            e.x, e.y, e.seen, e.monster = m.x, m.y, t, m
            e.sightings += 1
            e.misses, e.missed = 0, None
            if cover is not None:
                e.covered = t
        for j, m in enumerate(seen):
            if j in used or any(self._fading(f, m) for f in fading):
                continue
            e = Remembered(self._next_id, m.kind, m.platform, m.x, m.y, t, t, m)
            if cover is not None:
                e.covered = t
            self._next_id += 1
            self.entries.append(e)
            self.events.append(("new", t, e))
            self.counts["new"] += 1
        for k in unplaced:  # a kill no throw explains: the likeliest she threw at (not seen now first; nearest her)
            near = [e for e in live if e.live and self._thrown_at(e, t) and getattr(k, "named", None) in (None, e.kind)]
            if near and self._her is not None:
                self._killed(min(near, key=lambda q: (q.id in matched, abs(q.x - self._her[0]))), t, None)
        if cover is not None:
            self.counts["covered"] += 1
            self.covers.append(cover)
            for e in live:
                if e.id not in matched and self._holds(cover, e, t):
                    e.covered = t
                    if not self._thrown_at(e, t):
                        e.misses += 1
                        e.missed = t
            for pid in self._tallest:
                if self._covers_platform(cover, pid):
                    self.covered[pid] = t
        if players is not None:
            for e in self.entries:
                if e.live:
                    e.rival = near_rival(players, e.x, e.y) is not None
        for e in self.entries:
            if e.live and (e.p(t) < P_FORGET or t - e.seen > MEMORY_S):
                aged = e.missed is None or t - e.seen > MEMORY_S
                e.state, e.ended = "gone", t
                self.events.append(("gone", t, e))
                self.counts["aged" if aged else "gone"] += 1
        self._end_old(t)

    def _take_kills(self, t: float) -> list:
        """Her new kills (the ledger's): each one placed marks the monster remembered of its track (its last sighting's:
        Monster.track) killed, else (no track known) the nearest remembered on its platform, of its kind, within KILL_DX
        of where it was placed; and the feed's tracks taken for dead (`dead`) mark theirs: killed, or taken where
        another player killed it (`theirs`: event 'taken', which behavior.patrol.Belief learns `take` from, as from a
        monster by another player that looks forgot). Counted as 'dead' alone (2026-10-27), each such event pushed one
        of her kills before it out of what Belief read, and another player's kills were hers there: in loop_tree_grind
        the share Belief learnt another player takes of Green Mushrooms went from master's 0.8 to the prior 0.5 (review
        of 2026-10-27, review-behaviour: memcount.py, runsc_fix.py). The ones no throw of hers explains (unplaced) are
        given back: `update` marks the likeliest monster she threw at (of the kind her EXP gain named, if it named
        one), one this look didn't see first (one it saw may be fading out where it died), nearest her. Left alone, a
        monster she killed so stayed remembered for the 4.7 s that throws her way keep its looks from counting, and then
        till looks forgot it: on platform 12 in loop_lull_loot the ground wasn't empty in the lulls, and lull loot (rule
        8) never came."""
        if self.kills is None:
            return []
        kills = self.kills()
        new, self._kills_seen = kills[self._kills_seen:], len(kills)
        unplaced = []
        for k in new:
            if getattr(k, "platform", None) is None or getattr(k, "kind", None) is None:
                unplaced.append(k)
                continue
            if (track := getattr(k, "track", None)) is not None:
                for e in [e for e in self.entries if e.live and getattr(e.monster, "track", None) == track][:1]:
                    self._killed(e, t, k.x)
                continue
            near = [e for e in self.entries if e.live and e.platform == k.platform and e.kind == k.kind
                    and abs(e.x - k.x) <= KILL_DX]
            if near:
                self._killed(min(near, key=lambda q: abs(q.x - k.x)), t, k.x)
        dead = self.dead() if self.dead is not None else ()
        theirs = self.theirs() if self.theirs is not None else ()
        self._take_moves(t, dead)
        for e in [e for e in self.entries if e.live and getattr(e.monster, "track", None) is not None
                  and e.monster.track in dead]:
            if e.monster.track in theirs:  # (another player's kill: fading where it died, as one of hers)
                e.state, e.ended, e.died_x = "killed", t, e.x
                self.events.append(("taken", t, e))
                self.counts["taken"] += 1
            else:  # (hers, with no kill of hers on it yet)
                self._killed(e, t, e.x)
        return unplaced

    def _take_moves(self, t: float, dead) -> None:
        """Her kills the tracker put on another monster since (`moved`: behavior.ledger.Ledger.moved, the magician's,
        whose kills go by her search): the monster it was on, killed here by it (its track's), alive again where its
        track isn't dead now (the tracker let it go: seen alive), back as it was; the one it's on now (its track's),
        killed. Where one was alive again, that's the one kill moved, not another: event 'moved', not 'killed'
        (patrol.Belief empties the new one's point for it at the kill's time, in place of the old one's, and keeps its
        kill time once); else (none was killed here by it) 'killed'. Until 2026-11-02 each kill was taken once, as
        first placed: on that night's recordings master's tracker first put 28 of her 149 labelled kills on another
        monster (the replay kit's labels-v2), the real one remembered alive while the live one was killed here (a
        target no more, a pile's guard a ghost: P14, P34)."""
        if self.moved is None:
            return
        moved = self.moved()
        new, self._moved_seen = moved[self._moved_seen:], len(moved)
        for _, k, old, _, _ in new:
            was = None
            for e in self.entries:
                if e.state == "killed" and old is not None and getattr(e.monster, "track", None) == old \
                        and old not in dead:
                    was = was or (e.platform, e.kind, e.x if e.died_x is None else e.died_x, e.ended)
                    e.state, e.ended, e.died_x = "seen", None, None
            track = getattr(k, "track", None)
            if track:
                for e in [e for e in self.entries if e.live and getattr(e.monster, "track", None) == track][:1]:
                    if was is None:
                        self._killed(e, t, k.x)
                        continue
                    e.state, e.ended, e.died_x, e.moved_from = "killed", t, k.x, was
                    self.events.append(("moved", t, e))
                    self.counts["moved"] = self.counts.get("moved", 0) + 1

    def _killed(self, e: Remembered, t: float, x: float | None) -> None:
        e.state, e.ended, e.died_x = "killed", t, x
        self.events.append(("killed", t, e))
        self.counts["killed"] += 1

    def _end_old(self, t: float) -> None:
        self.entries = [e for e in self.entries if e.live or t - e.ended < ENDED_KEEP_S]

    def _reach(self, e: Remembered, t: float) -> float:
        """How far from where it was seen a sighting can be that monster (MATCH_SLACK, MATCH_SPEED)."""
        walk = self._speeds.get(e.kind, DEFAULT_SPEED) * min(max(0.0, t - e.seen), MATCH_WALK_S)
        return MATCH_SLACK + MATCH_SPEED * walk

    def _top(self, kind: str) -> float:
        """How tall a monster of `kind` is for a look to have searched all of it (see the top)."""
        b = self._bodies.get(kind)
        return DEFAULT_TOP if b is None else max(b.top, MONSTER_HEIGHT.get(kind, 0.0))

    def _half(self, kind: str) -> float:
        b = self._bodies.get(kind)
        return 30.0 if b is None else b.half

    def _holds(self, cover: Cover, e: Remembered, t: float) -> bool:
        """Whether `cover` searched where monster `e` can be (COVER_SHARE of it): its platform from where it was seen,
        as far as it can have walked since either way."""
        p = self.model.platform(e.platform)
        walk = self._speeds.get(e.kind, DEFAULT_SPEED) * max(0.0, t - e.seen)
        return _share(cover, p, p.clamp(e.x - walk), p.clamp(e.x + walk), self._half(e.kind), self._top(e.kind))

    def _covers_platform(self, cover: Cover, pid: int) -> bool:
        """Whether `cover` searched all of platform `pid` (every one of COVER_POINTS places along it) for its tallest
        body: a platform half of it searched isn't seen empty."""
        p = self.model.platform(pid)
        return _share(cover, p, p.left, p.right, 0.0, self._tallest[pid], need=1.0)

    def _fading(self, f: Remembered, m: Monster) -> bool:
        """Whether sighting `m` can be killed monster `f` fading out where it died (see the top)."""
        return f.platform == m.platform and f.kind == m.kind and (
            abs(m.x - f.x) <= self._reach(f, f.ended) or (f.died_x is not None and abs(m.x - f.died_x) <= FADE_DX))

    def _thrown_at(self, e: Remembered, t: float) -> bool:
        """Whether a throw of hers in the last HIDE_S (her star leaving up to HIT_AFTER_S after its key) went monster
        `e`'s way, within a star's reach of her (tracking.THROW_REACH, THROW_DY), and it was seen since: it may be
        hiding under her damage numbers."""
        s = self.shape
        if t - e.seen >= s.hide_s + s.hit_after_s:
            return False
        for when, side, x, y in self._throws:
            if (t - (when + s.hit_after_s) < s.hide_s and max(0.0, s.near) <= (e.x - x) * side <= s.ahead
                    and -s.up <= e.y - y <= s.down):  # (ahead of her only: the thief's 0, not her claw's 10 behind)
                return True
        return False

    # --- telemetry ------------------------------------------------------------------------------------

    def record(self, t: float) -> dict:
        """The memory as telemetry data ('memory'): every monster kept (Remembered.record: its age and p, ended ones
        too, for a moment), and how long ago each platform was last covered all of it ('covered': platform, s)."""
        return {"list": [e.record(t) for e in self.entries],
                "covered": [[pid, round(t - at, 1)] for pid, at in sorted(self.covered.items())]}


def _share(cover: Cover, p, lo: float, hi: float, half: float, top: float, need: float = COVER_SHARE) -> bool:
    """Whether `cover` searched `need` (COVER_SHARE) at least of platform `p` from x `lo` to `hi` (COVER_POINTS along
    it) for a body `half` wide either side and `top` tall."""
    xs = [lo + (hi - lo) * i / (COVER_POINTS - 1) for i in range(COVER_POINTS)] if hi > lo else [lo]
    return sum(cover.holds(x, p.y_at(x), half, top) for x in xs) >= need * len(xs)


def _kinds_by_platform(model: MapModel) -> dict[int, set[str]]:
    """The kinds that spawn on each platform (perception.kinds' names), every platform with monsters."""
    from ..perception.kinds import kind_of
    return {pid: {kind_of(n) for n in names} for pid, names in model.monsters_by_platform.items()}


# --- a recorded run, again ---------------------------------------------------------------------------------

def replay(records: Iterable[dict], model: MapModel, band: tuple[float, float],
           every: float = RECORD_EVERY_S, memory: "Memory | None" = None) -> list[dict]:
    """The memory run again on a recorded run's records (telemetry, in order): every 'scene' record's monsters (and
    'swept') into it, with what the look covered (cover_of_record), and her kills from the 'kill' events (her throws
    aren't in the records: a monster she threw at is missed as any other). The records with a 'memory' record after a
    'scene' one every `every` s at most (RECORD_EVERY_S, as live) and after the last, any 'memory' records they had
    left out. `memory`: the memory to run (a new one by default), its counts kept for the caller."""
    import re
    kill_re = re.compile(r"^(?P<kind>\S+) down on platform (?P<platform>-?\d+)")  # behavior.fight's 'kill' events
    kills: list = []
    if memory is None:
        memory = Memory(model)
    memory.kills = lambda: kills
    out, last, pending = [], -1e18, -1e18
    for r in records:
        if r.get("topic") == "memory":
            continue
        out.append(r)
        if r.get("topic") == "event" and r.get("kind") == "kill" and (m := kill_re.match(r.get("text", ""))):
            kills.append(_Kill(r["t"], m["kind"], int(m["platform"]), float(r.get("x") or 0.0)))
            continue
        if r.get("topic") != "scene":
            continue
        t, her = r["t"], r.get("her") or {}
        if her.get("x") is not None:
            memory._her = (her["x"], her["y"])
        if not r.get("searched") or her.get("x") is None:
            continue
        monsters = [_monster(m) for m in [*r.get("monsters", []), *r.get("swept", [])]]
        players = [Player(p["id"], p["kind"], p["x"], p["y"], p.get("platform"), (0.0, 0.0), t, t)
                   for p in r.get("players", [])]
        memory.update(t, monsters, cover_of_record(r, band), players)
        if t - last >= every:
            last = t
            out.append({"t": t, "topic": "memory", **memory.record(t)})
        else:
            pending = t
    if last < pending:
        out.append({"t": pending, "topic": "memory", **memory.record(pending)})
    return out


def cover_of_record(r: dict, band: tuple[float, float]) -> Cover | None:
    """What a recorded look ('scene' record) searched for monsters: its 'rows', else (recorded before they were) its
    band around her feet or, for a whole-screen look, its 'view'; the minimap at the view's top left, as big in map
    units at any window size (MINIMAP_BOX at 1920x1080 over its stretch). None when it didn't place her standing."""
    her, view = r.get("her") or {}, r.get("view")
    if not r.get("searched") or view is None or her.get("x") is None or not her.get("grounded", True):
        return None
    x0, y0, x1, y1 = view
    rows = r.get("rows")
    if rows is None:
        rows = [[y0, y1]] if r.get("wide") else [[max(y0, her["y"] - band[0]), min(y1, her["y"] + band[1])]]
    a, b, c, d = (v / STRETCHED.stretch for v in MINIMAP_BOX)
    return Cover(r["t"], tuple(tuple(v) for v in rows), x0, x1, (x0 + a, y0 + b, x0 + c, y0 + d))


@dataclass(frozen=True)
class _Kill:
    t: float
    kind: str
    platform: int
    x: float


def _monster(m: dict) -> Monster:
    """A monster as a scene record has it, as a Monster again (its picture not known)."""
    mob = Mob(m["kind"], 0, 0, 0, 0, 1.0, facing=m.get("facing", 0))  # (screen px not recorded)
    return Monster(mob, m["x"], m["y"], m.get("platform"), m.get("bar"), m.get("facing", 0))
