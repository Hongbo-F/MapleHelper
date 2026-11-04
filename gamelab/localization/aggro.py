"""Her monsters' aggro (`Aggro`): every monster she has hurt that may still chase her, kept for as long as its chase
lasts, through the looks that don't see it and the tracker's changes of its id, with where it likeliest is now; and
which monster likeliest touched her at each touch (`touches`). A belief, passive like the memory (localization.memory):
it takes every look (`on_look`, a Senses listener) and her throws (`throw`, an Actions throw listener) and changes
nothing she does by itself; the planner asks it (`chasers_at`, `near`; `Guess.track` as the move planner's
monsters.MonsterTrack), as the spawn clock is asked (localization.spawnclock).

The rule it follows (the user's, 2026-10-31; sim.world.AGGRO_S, the v83 client's Mob::chase as HeavenClient has it):
an unhurt monster wanders; a hit of hers draws its chase, CHASE_S (3 to 4 minutes) from the hit, each hit anew. Chasing,
every THINK_S it decides which way to walk: at her, or on the way it chose while she is within CHASE_PASS across (so it
walks past her and back), never standing, at its kind's walk speed (the game data's: monsters.speeds), along its own
platform wherever she is, turning back at its walk edges (monsters.walk_edges) and, where its way still points there,
walking to that end again. Hit, it stands its flinch out first (monsters.FLINCH_S).

Why it's needed (the arrival diagnosis of 2026-11-03, D:/MapleHelper-scratch/arrival/REPORT.md, the reviewers' arr_7):
at the -80 rope's arrivals onto Tree Dungeon's 10 the look held none of the monsters within 60 of her in about 30% of
the cases, tracks had no heading right after the climb, the navigator's memory of a trip forgets after 2 s
(navigator.TRIP_MEMORY_S), its `_chasing` is a bar seen in the newest state, and the hunt's wounds (behavior.ledger.
Wounds) go WOUND_LOST_S out of sight and WOUND_S after the first hit: nothing kept who she had hurt for the 3-4 minutes
they chase her.

What makes a chaser (`_bar_hit`), from the feed's events and tracks (Scene.events, Scene.tracks: localization.tracking):
- a 'bar_hit' event "hers": a track's HP bar down within her Shape's window after a throw of hers that had it in its
  path (tracking.MonsterTracker._by);
- a 'bar_hit' that isn't (the tracker's window past: a monster hidden under her damage numbers or her claw's effect for
  1-3 s is seen again with its bar down later) on a track a throw of hers had in its path up to LATE_S before (`throw`:
  her throws, the paths from the last look's tracks as the tracker takes them, tracking.MonsterTracker._in_path).
A hit on a track no chaser has, first seen after a chaser of its kind there was last seen (the tracker found a monster
again under a new id while the old one waits hidden, or let it go), within that one's reach, is that chaser's
(`_lost_near`), not a new one.

What keeps it (`_associate`): each look, every chaser goes to the track that's it, one each, the nearest its prediction
first (`_pairs`): its own track as long as that shows a bar (Track.bar_at, read within BAR_FRESH_S of its sighting) or
did BAR_GONE_S ago at most, OWN_BONUS nearer than another; another chaser's track never, while that one has it, seen or
not; another only first seen since this one was last (found again under a new id), or, once it has none, any showing a
bar within its reach (REJOIN_SLACK more). Its track gone from the tracker, it's lost and its chase walked on; its track
seen BAR_GONE_S with no bar on it (the track went over to an unhurt monster, or the bar it was made by was a neighbour's
read over it: perception.hud.hp_bar's BAR_SLACK, 5% of the simulator's bars read over unhurt monsters), it's doubted
(DOUBT_P) until a track with a bar is it, else dropped DOUBT_S later ('phantom'). What ends it: its death (a 'died'
event, or its track a kill's: tracking.Track.killed; one taken for another's kill with nobody by it is kept, lost,
OTHER_DEATH_P as likely, as the tracker takes a hurt monster hidden past HIDE_S for dead), CHASE_S after her last hit,
FORGET_S unseen (`_end`).

Where it is (`_guess`, `_chase_x`): from its last sighting, its chase walked on against her x as the looks had her (her
path), its first decision half a decision on if it was seen walking (at once after a hit's flinch), then every THINK_S,
for DET_S unseen (or till it can have reached her, and DET_S more), its reach SPREAD_0 and SPREAD_RATE of its walk a
second unseen either side; past that it's about her: her x, CHASE_PASS and a decision's walk either side (it walks past
her that far before it turns), within its walk edges. How sure (`p`): the chase still on (1 until CHASE_S[0] after her
last hit, falling to 0 at CHASE_S[1]), times its being about at all (1 while seen within SEEN_S, then
exp(-(unseen - SEEN_S) / LOST_S)), times OTHER_DEATH_P or DOUBT_P where those hold; 0 while its track is a kill's.

Her touches (`touches`, `toucher`): each 'touch' event's monster at its popup's onset, against her place then (her
path's look before it, HER_AT_S old at most, else the nearest): of the chasers' guesses and the other tracks' last
sightings walked on at their heading (0.5 s at most), those whose body can reach hers from where they can be, the one
whose reach puts it on her likeliest (the narrowest first); the side it came from, where it was against her, put right
by a damage touch's knock as her next looks have her (KNOCK_DX: away from it); its kind and track.

Pure: nothing here looks or presses a key; times are the looks' and the feed's events' clock (Scene.t, Event.t).
`update` takes plain data, so the same belief runs on a recorded run's telemetry (`replay`: the perception service's
'tracks' and 'event' records, her throws from the hunt's 'attack' records)."""

import math
from bisect import bisect_right
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from .mapdata import MapModel
from .monsters import DEFAULT_SPEED, FLINCH_S, MonsterTrack, bodies, speeds, touch_box, walk_edges

CHASE_S = (180.0, 240.0)  # s a hit of hers draws its chase (sim.world.AGGRO_S: v83's, drawn uniformly in between)
CHASE_PASS = 100.0        # units: within this of her across, a chaser keeps the way it chose (sim.world.AGGRO_PASS)
THINK_S = 0.945           # s between a chaser's decisions (sim.world.CHASE_THINK_MS: 0.72-1.17 s, the middle)
LATE_S = 4.0              # s after a throw of hers that a bar seen down on a monster in its path is her hit
PATH_SLACK = 20.0         # units past her Shape's reach a monster of the last look may be and still be in a throw's path
SEEN_S = 10.0             # s unseen before a chaser's being about is doubted ...
LOST_S = 60.0             # ... then exp(-(unseen - SEEN_S) / this)
FORGET_S = 120.0          # s unseen after which it's dropped (p by then exp(-110 / 60): 0.16)
OTHER_DEATH_P = 0.5       # a chaser the tracker took for another's kill, nobody by it: kept this likely
REJOIN_SLACK = 40.0       # units past its reach a new track may be and still be a lost chaser again
BAR_GONE_S = 1.5          # s a chaser's track is seen with no bar read on it (Track.bar_at) before it isn't the chaser
STALE_S = 0.3             # s unseen before a chaser's own track may be another's new sighting (tracking.MISSING_S)
OWN_BONUS = 30.0          # units: a chaser's own track is taken for it this much farther off than another
UNHURT_COST = 40.0        # units: a track showing no bar is taken for a chaser this much farther off
BAR_FRESH_S = 0.5         # s: a track whose bar was read this soon before its last sighting shows one now
DOUBT_P = 0.3             # a chaser whose bar left its track, not found again: this likely ...
DOUBT_S = 8.0             # ... for this long, then dropped (a bar read off a neighbour made it: 'phantom')
DET_S = 1.0               # s unseen its chase is walked on as last seen (its way till its next decision or so) ...
SPREAD_0 = 15.0           # units: its reach either side of its prediction just seen ...
SPREAD_RATE = 0.35        # ... and this much of its walk speed a second unseen on top
PATH_S = 30.0             # s of her path kept (her x for the chases' predictions)
TOUCH_REACH = 22.0        # units past a monster's half width and hers that it may be and still be the one that touched
HER_HALF = 20.0           # her half width for touches (motion.bodies' HER_HALF is 18-22 by her pose)
HER_AT_S = 0.5            # s: her path's look before a touch this recent is where she was at it
KNOCK_DX = 15.0           # units her place moved from before a damage touch to the next look: knocked that way
TOUCH_KEEP = 64           # touches kept


@dataclass
class Chaser:
    """A monster she has hurt, as the belief has it."""
    key: int                  # the belief's own id (kept across the tracker's ids)
    kind: str
    platform: int | None
    x: float                  # its last sighting (map x, its feet) ...
    y: float
    seen: float               # ... and when (the looks' clock)
    heading: int = 0          # +1 right, -1 left, 0 not known: the way it walked when last seen
    track: int | None = None  # the tracker's track it is now (None: lost)
    tracks: list[int] = field(default_factory=list)  # every track it has been
    bar: float | None = None  # HP left as its bar last read (None: never read)
    first: float = 0.0        # when her first hit on it showed ...
    hit: float = 0.0          # ... and her last
    hits: int = 0
    how: str = "hit"          # how it became a chaser ('hit': a bar_hit hers; 'late': her throw's path, seen later)
    lost: float | None = None  # when its track was let go (None: it has one)
    dying: bool = False       # its track a kill's (tracking.Track.killed), not yet dead
    other: bool = False       # taken for another's kill with nobody by it (OTHER_DEATH_P)
    doubt: float | None = None  # when its bar left its track (BAR_GONE_S), till it's found again with one

    def record(self, t: float) -> dict:
        return {"key": self.key, "kind": self.kind, "platform": self.platform, "x": round(self.x, 1),
                "seen": round(t - self.seen, 2), "heading": self.heading, "track": self.track,
                "bar": None if self.bar is None else round(self.bar, 2), "hit": round(t - self.hit, 1),
                "hits": self.hits, **({"lost": round(t - self.lost, 1)} if self.lost is not None else {}),
                **({"other": True} if self.other else {}),
                **({"doubt": round(t - self.doubt, 1)} if self.doubt is not None else {})}


@dataclass(frozen=True)
class Guess:
    """A chaser where the belief has it at a moment: its likeliest x, where it can be (`lo`, `hi`), the way its chase
    walks then, and how likely it's there chasing her (`p`)."""
    chaser: Chaser
    t: float
    x: float
    lo: float
    hi: float
    way: int
    p: float

    @property
    def kind(self) -> str:
        return self.chaser.kind

    @property
    def platform(self) -> int | None:
        return self.chaser.platform

    def track(self, speed: float, seen_at: float = 0.0, half: float = 28.0, height: float = 56.0) -> MonsterTrack:
        """As the move planner takes a monster (monsters.MonsterTrack) at plan time `seen_at` (this guess's moment):
        where it likeliest is, walking its chase's way (`vx`) as long as that's a prediction (`trust`: DET_S from its
        sighting), else (about her) its heading not known for as long as its reach is wide (`unsure`: half its reach
        over its walk)."""
        if self.way:
            trust = max(0.0, DET_S - max(0.0, self.t - self.chaser.seen))
            return MonsterTrack(self.platform if self.platform is not None else -1, self.x, speed, self.way, half,
                                self.way * speed, seen_at, height, self.way, self.kind, None, 0.0, self.chaser.track,
                                trust, None)
        unsure = max(0.0, (self.hi - self.lo) / 2.0) / max(speed, 1.0)
        return MonsterTrack(self.platform if self.platform is not None else -1, self.x, speed, 0, half, None, seen_at,
                            height, 0, self.kind, None, 0.0, self.chaser.track, None, unsure)


@dataclass(frozen=True)
class Touched:
    """A touch of hers and the monster that likeliest made it."""
    t: float                  # the touch (its popup's onset: the event's `at`, else the event's frame)
    told: float               # the look that told it
    damage: bool              # it took HP (a MISS knocks her nowhere)
    her: float | None         # her x then (her path)
    kind: str | None          # the monster's kind (None: none near enough)
    side: int                 # +1 it was on her right, -1 her left, 0 not known
    x: float | None           # where it was then
    track: int | None         # its track, if one
    key: int | None           # the chaser it was, if one
    dx: float | None          # how far from her (middle to middle)
    by: str                   # how it was told: 'chaser', 'track', 'none'
    knock: int = 0            # the way her next look had her moved (+1 right, -1 left, 0 not)

    def record(self) -> dict:
        return {"t": round(self.t, 3), "damage": self.damage, "kind": self.kind, "side": self.side,
                "x": None if self.x is None else round(self.x, 1), "track": self.track, "key": self.key,
                "dx": None if self.dx is None else round(self.dx, 1), "by": self.by, "knock": self.knock}


@dataclass
class _Throw:
    t: float
    side: int                 # +1 right, -1 left
    path: set[int]            # the tracks its star could meet (as of the last look)


class Aggro:
    """The monsters she has hurt that chase her (see the top)."""

    def __init__(self, model: MapModel | None = None, reach: tuple[float, float, float, float] | None = None):
        self.model = model
        self.walking = speeds(model)
        self.bodies = bodies(model)
        # where a throw of hers can take a monster: ahead of her middle from `near` to `ahead`, `up` above her feet to
        # `down` below (her Shape's: tracking.MonsterTracker._in_path; the thief's by default)
        self.reach = reach or (-10.0, 474.0, 70.0, 70.0)
        self.chasers: dict[int, Chaser] = {}
        self.ended: deque[tuple[float, Chaser, str]] = deque(maxlen=64)  # (when, the chaser, why): dead, over, gone
        self.touches: deque[Touched] = deque(maxlen=TOUCH_KEEP)
        self.path: list[tuple[float, float, float, int | None]] = []  # her place by look: (t, x, y, platform)
        self.throws: deque[_Throw] = deque()
        self._next = 1
        self._tracks: dict[int, object] = {}  # the last look's tracks by id
        self._t: float | None = None           # the last look's time
        self.counts = {"hits": 0, "late hits": 0, "rejoined": 0, "lost": 0, "dead": 0, "over": 0, "gone": 0,
                       "touches": 0, "other deaths": 0, "bar gone": 0, "phantom": 0}

    # --- what it's fed ------------------------------------------------------------------------------------------------

    def on_look(self, s) -> None:
        """A look (Senses.listeners): its time, her place, the feed's events since the last look and its tracks. A look
        made up from another (Scene.what_if) isn't one."""
        if getattr(s, "what_if", False):
            return
        her = None
        if s.where is not None:
            p = s.where.platform
            her = (s.where.x, s.where.y, None if p is None else p.id)
        self.update(s.t, her, getattr(s, "tracks", None) or (), getattr(s, "events", None) or ())

    def throw(self, when: float, side: str, aim: int | None = None) -> None:
        """A throw of hers (Actions.throw_listeners: its key's time, 'left' or 'right'; `aim`, the track it was aimed at,
        if told): the tracks of the last look its star could meet, for a bar seen down on one of them later (LATE_S)."""
        sign = 1 if side == "right" else -1
        path = self._in_path(when, sign)
        if aim is not None:
            path.add(aim)
        self.throws.append(_Throw(when, sign, path))

    def update(self, t: float, her: tuple[float, float, int | None] | None, tracks: Iterable,
               events: Iterable) -> None:
        """A look at t: her place (x, y, platform; None: not placed), the feed's tracks as of it (tracking.Track, or
        anything with the same fields: `replay`'s) and the events since the last look (tracking.Event)."""
        if her is not None and (not self.path or t >= self.path[-1][0]):
            self.path.append((t, her[0], her[1], her[2]))
        if self.path and t - self.path[0][0] > 2 * PATH_S:  # (trimmed now and then: kept PATH_S to 2 PATH_S)
            del self.path[:bisect_right(self.path, t - PATH_S, key=lambda p: p[0])]
        while self.throws and t - self.throws[0].t > LATE_S:
            self.throws.popleft()
        by_id = {tr.id: tr for tr in tracks}
        touches = []
        for e in sorted(events, key=lambda e: e.t):
            if e.kind == "bar_hit" and e.track is not None:
                self._bar_hit(e, by_id)
            elif e.kind == "died" and e.track is not None:
                self._died(e)
            elif e.kind == "touch":
                touches.append(e)
        self._follow(t, by_id)
        self._associate(t, by_id)
        self._end(t)
        self._tracks, self._t = by_id, t
        for e in touches:
            self._touched(e, t, by_id)
        self._knocks(t)

    # --- inside -------------------------------------------------------------------------------------------------------

    def _bar_hit(self, e, by_id: dict) -> None:
        """A track's bar down (a 'bar_hit' event): her hit when the tracker says so, or a throw of hers had it in its
        path within LATE_S (`throws`)."""
        by = (e.data or {}).get("by")
        late = by != "hers" and any(0.0 <= e.t - th.t <= LATE_S and e.track in th.path for th in self.throws)
        if by != "hers" and not late:
            return
        bar = (e.data or {}).get("bar")
        c = self._of_track(e.track)
        if c is None:
            tr = by_id.get(e.track) or self._tracks.get(e.track)
            if tr is None:
                return
            c = self._lost_near(tr, e.t)
            if c is None:
                c = self._new(tr, e.t, "late" if late else "hit")
            else:  # (a lost chaser hit again: the track it's on now)
                c.track, c.lost = tr.id, None
                if tr.id not in c.tracks:
                    c.tracks.append(tr.id)
                self.counts["rejoined"] += 1
        c.hit, c.hits = max(c.hit, e.t), c.hits + 1
        if bar is not None:
            c.bar = bar
        c.other, c.doubt = False, None
        self.counts["late hits" if late else "hits"] += 1

    def _new(self, tr, t: float, how: str) -> Chaser:
        c = Chaser(self._next, tr.kind, tr.platform, tr.x, tr.y, self._seen_of(tr, t), self._heading(tr), tr.id,
                   [tr.id], getattr(tr, "bar", None), t, t, 0, how)
        self._next += 1
        self.chasers[c.key] = c
        return c

    def _lost_near(self, tr, t: float) -> Chaser | None:
        """A chaser of track `tr`'s kind and platform not seen for STALE_S (no track, or its own out of sight: the
        tracker may follow a monster found again under a new id while the old one waits hidden) whose reach at t holds
        it (REJOIN_SLACK more): the nearest its prediction."""
        best = None
        for c in self.chasers.values():
            if c.dying or c.kind != tr.kind or (c.platform is not None and tr.platform is not None
                                                and c.platform != tr.platform):
                continue
            if c.track is not None and t - c.seen <= STALE_S:  # (its own track seen just now: not this one)
                continue
            first = getattr(tr, "first_seen", None)
            if first is not None and first <= c.seen:  # (a monster followed since before it was last seen: another)
                continue
            g = self._guess(c, t)
            score = (c.doubt is not None, abs(tr.x - g.x))  # (a doubted one last: it may be a bar misread)
            if g.lo - REJOIN_SLACK <= tr.x <= g.hi + REJOIN_SLACK and (best is None or score < best[0]):
                best = (score, c)
        return None if best is None else best[1]

    def _of_track(self, tid: int) -> Chaser | None:
        return next((c for c in self.chasers.values() if c.track == tid), None)

    def _died(self, e) -> None:
        """A track's death: its chaser's, if its cause is hers or someone was by it; else (another's kill with nobody
        by it: the tracker takes a monster hidden past HIDE_S for dead) its chaser is kept, lost, less likely."""
        c = self._of_track(e.track)
        if c is None:
            return
        data = e.data or {}
        if data.get("cause") == "other" and data.get("by") is None:
            c.track, c.lost, c.other = None, e.t, True
            self.counts["other deaths"] += 1
            return
        self._drop(c, e.t, "dead")

    def _drop(self, c: Chaser, t: float, why: str) -> None:
        self.chasers.pop(c.key, None)
        self.ended.append((t, c, why))
        self.counts[why] += 1

    def _follow(self, t: float, by_id: dict) -> None:
        """Each chaser's own track as of this look: a kill's (dying; dead, it's dropped); gone from the tracker (let go:
        out of sight too long, lost in a crowd), the chaser is lost."""
        for c in list(self.chasers.values()):
            if c.track is None:
                continue
            tr = by_id.get(c.track)
            if tr is None:
                if c.dying:
                    self._drop(c, t, "dead")
                else:
                    c.track, c.lost = None, t
                    self.counts["lost"] += 1
                continue
            if getattr(tr, "state", "live") == "dead":
                self._drop(c, t, "dead")
                continue
            c.dying = bool(getattr(tr, "killed", False))

    def _associate(self, t: float, by_id: dict) -> None:
        """Every chaser (not dying) to the track that's it this look, one each, the nearest pairs first (`_pairs`):
        where it's seen, its heading and bar. One its own track has no longer (its bar gone from it BAR_GONE_S: the
        track went over to an unhurt monster, or the bar it was taken by was a neighbour's read over it, hud.hp_bar's
        BAR_SLACK) is doubted until it's found again with one (DOUBT_S, DOUBT_P); one whose track went to another is
        lost."""
        chasers = [c for c in self.chasers.values() if not c.dying]
        taken = {c.track for c in self.chasers.values() if c.dying and c.track is not None}
        got: dict[int, object] = {}
        for _, key, tid in self._pairs(t, chasers, by_id, taken):
            if key in got or tid in taken:
                continue
            got[key] = by_id[tid]
            taken.add(tid)
        for c in chasers:
            tr = got.get(c.key)
            if tr is None:
                own = by_id.get(c.track) if c.track is not None else None
                if own is not None and own.id in taken:  # (its track is another's now)
                    c.track, c.lost = None, t
                    self.counts["lost"] += 1
                elif own is not None and self._bar_gone(own, t):
                    c.track, c.lost, c.doubt = None, t, t
                    self.counts["bar gone"] += 1
                continue
            if c.track != tr.id:
                if c.track is None:
                    self.counts["rejoined"] += 1
                c.track = tr.id
                if tr.id not in c.tracks:
                    c.tracks.append(tr.id)
            seen = self._seen_of(tr, t)
            if seen is not None and seen >= c.seen:
                c.x, c.y, c.seen = tr.x, tr.y, seen
                if tr.platform is not None:
                    c.platform = tr.platform
                c.heading = self._heading(tr) or c.heading
            if getattr(tr, "bar", None) is not None:
                c.bar = tr.bar
            c.lost = None
            if self._shows_bar(tr):
                c.doubt, c.other = None, False

    def _pairs(self, t: float, chasers: list[Chaser], by_id: dict, taken: set) -> list[tuple[float, int, int]]:
        """The (cost, chaser, track) pairs a look can make: a track of the chaser's kind on its platform, not a kill's,
        seen since the last look, no sooner than the chaser was last, another chaser's track never (its own while it has
        it, seen or not), and with a track of its own only one first seen since it was last seen (a monster found again
        under a new id); showing a bar (`_shows_bar`), or its own whose bar was read BAR_GONE_S before its sighting at
        most, or (not doubted) one new since it was lost; within the chaser's reach at the track's sighting
        (REJOIN_SLACK more); the cost how far from its prediction then, its own track OWN_BONUS nearer, one with no bar
        UNHURT_COST farther."""
        out = []
        owned = {c.track: c.key for c in self.chasers.values() if c.track is not None}
        for tr in by_id.values():
            if tr.id in taken or getattr(tr, "killed", False) or getattr(tr, "state", "live") in ("dead", "offscreen"):
                continue
            seen = self._seen_of(tr, t)
            if seen is None:
                continue
            hurt = self._shows_bar(tr)
            first = getattr(tr, "first_seen", None)
            fresh = self._t is None or seen > self._t + 1e-6  # (seen since the last look)
            for c in chasers:
                if tr.kind != c.kind or (c.platform is not None and tr.platform is not None
                                         and tr.platform != c.platform):
                    continue
                own = tr.id == c.track
                if not fresh or (not own and seen < c.seen - 1e-6) or (not own and tr.id in owned):
                    continue  # (another chaser's track is that one's while it has it, seen or not)
                if not own and c.track is not None and (first is None or first <= c.seen):
                    continue  # (with a track of its own, only one found since it was last seen can be it)
                if not hurt:
                    if own:
                        if self._bar_gone(tr, t):
                            continue
                    elif c.doubt is not None or c.lost is None or first is None or first < c.lost - 0.5:
                        continue
                g = self._guess(c, seen)
                if not g.lo - REJOIN_SLACK <= tr.x <= g.hi + REJOIN_SLACK:
                    continue
                out.append((abs(tr.x - g.x) - (OWN_BONUS if own else 0.0) + (0.0 if hurt else UNHURT_COST),
                            c.key, tr.id))
        return sorted(out)

    def _bar_gone(self, tr, t: float) -> bool:
        """Whether track `tr` has been seen BAR_GONE_S with no bar read on it (Track.bar_at; a record without it:
        never)."""
        bar_at, seen = getattr(tr, "bar_at", None), self._seen_of(tr, t)
        return bar_at is not None and seen is not None and seen - bar_at > BAR_GONE_S

    def _end(self, t: float) -> None:
        for c in list(self.chasers.values()):
            if t - c.hit >= CHASE_S[1]:
                self._drop(c, t, "over")
            elif t - c.seen >= FORGET_S:
                self._drop(c, t, "gone")
            elif c.doubt is not None and t - c.doubt >= DOUBT_S:
                self._drop(c, t, "phantom")

    def _shows_bar(self, tr) -> bool:
        """Whether track `tr` shows an HP bar as of its last sighting: one read within BAR_FRESH_S of it (Track.bar_at;
        a record without it: its bar as kept)."""
        if getattr(tr, "bar", None) is None:
            return False
        bar_at, seen = getattr(tr, "bar_at", None), getattr(tr, "last_seen", None)
        return bar_at is None or seen is None or seen - bar_at <= BAR_FRESH_S

    @staticmethod
    def _seen_of(tr, t: float) -> float | None:
        """When track `tr` was last seen (Track.last_seen; a record without it: now if it's live, else not known)."""
        seen = getattr(tr, "last_seen", None)
        if seen is not None:
            return seen
        return t if getattr(tr, "state", "live") == "live" else None

    @staticmethod
    def _heading(tr) -> int:
        vx = getattr(tr, "vx", 0.0) or 0.0
        return 1 if vx > 0 else -1 if vx < 0 else 0

    def _in_path(self, t: float, sign: int) -> set[int]:
        """The last look's tracks a throw of hers keyed at t that way could meet (her place as of her path's last), as
        far as each can have walked since that look (PATH_SLACK, and its walk at DEFAULT_SPEED for that long, 1 s at
        most) beyond her Shape's reach either way."""
        if not self.path or not self._tracks:
            return set()
        _, hx, hy, _ = self.path[-1]
        near, ahead, up, down = self.reach
        slack = PATH_SLACK + DEFAULT_SPEED * min(1.0, max(0.0, t - (self._t if self._t is not None else t)))
        return {tr.id for tr in self._tracks.values()
                if getattr(tr, "state", "live") not in ("dead", "offscreen")
                and near - slack <= sign * (tr.x - hx) <= ahead + slack and -up <= tr.y - hy <= down}

    # --- where they are -----------------------------------------------------------------------------------------------

    def her_x(self, t: float) -> float | None:
        """Her x at t as her path has it: the last look's at or before t (the first's before it), None with none."""
        if not self.path:
            return None
        i = bisect_right(self.path, t, key=lambda p: p[0])
        return self.path[max(0, i - 1)][1]

    def _her_at(self, t: float) -> tuple | None:
        """Her place by her path at t: the last look's before it if within HER_AT_S of it (a look after a touch has her
        knocked away already), else the look nearest it; None with none."""
        if not self.path:
            return None
        i = bisect_right(self.path, t + 1e-6, key=lambda p: p[0])
        if i and t - self.path[i - 1][0] <= HER_AT_S:
            return self.path[i - 1]
        return min(self.path[max(0, i - 1):i + 1], key=lambda p: abs(p[0] - t))

    def _edges(self, c: Chaser) -> tuple[float, float] | None:
        if self.model is None or c.platform is None or not 0 <= c.platform < len(self.model.platforms):
            return None
        return walk_edges(self.model.platform(c.platform))

    def _chase_x(self, c: Chaser, t: float) -> tuple[float, int]:
        """Where chaser `c` likeliest is at t, and the way it walks then: from its last sighting, its flinch stood out
        if her hit came after it, then a decision every THINK_S, at her (her path's x then) or on the way it chose while
        she's within CHASE_PASS across, walked at its kind's speed, folding back at its walk edges (as sim.world's
        SimMonster._fold: the way it chose kept)."""
        edges = self._edges(c)
        speed = self.walking.get(c.kind, DEFAULT_SPEED)
        x, way, tt = c.x, c.heading, c.seen
        if edges is not None:
            x = min(max(x, edges[0]), edges[1])
        if c.hit > tt:  # (hit since its sighting: it stood its flinch out where it was, then chases at her anew)
            tt, way = min(t, max(tt, c.hit + FLINCH_S)), 0
        moving, first = way, True
        while tt < t - 1e-9:
            hx = self.her_x(tt)
            if hx is not None and (way == 0 or abs(hx - x) >= CHASE_PASS) and not (first and way):
                way = 1 if hx >= x else -1
            # (seen walking, its decision then is half through, as likely as not: it walks on THINK_S / 2 first)
            nxt = min(t, tt + (THINK_S / 2 if first and way else THINK_S))
            first = False
            x, moving = _fold(x, way, speed * (nxt - tt), edges)
            tt = nxt
        return x, moving

    def _guess(self, c: Chaser, t: float) -> Guess:
        """Chaser `c` at t: its chase walked on from its last sighting (`_chase_x`) while that's a prediction (DET_S
        unseen, or till it can have reached her and DET_S more), its reach SPREAD_0 and SPREAD_RATE of its walk a
        second unseen either side; past that, about her: it walks past her by CHASE_PASS and up to a decision's walk
        more before it turns (CHASE_PASS + THINK_S of its walk either side of her x, its likeliest place her x), within
        its walk edges."""
        speed = self.walking.get(c.kind, DEFAULT_SPEED)
        unseen = max(0.0, t - c.seen)
        edges = self._edges(c)
        hx = self.her_x(t)
        if hx is not None and unseen > DET_S + max(0.0, abs(c.x - hx) - CHASE_PASS) / max(speed, 1.0):
            half = CHASE_PASS + speed * THINK_S
            x, lo, hi, way = hx, hx - half, hx + half, 0
            if edges is not None:
                x = min(max(x, edges[0]), edges[1])
        else:
            x, way = self._chase_x(c, t)
            spread = SPREAD_0 + SPREAD_RATE * speed * unseen
            lo, hi = x - spread, x + spread
        if edges is not None:
            lo, hi = max(lo, edges[0]), min(hi, edges[1])
        return Guess(c, t, x, lo, hi, way, self.p(c, t))

    def p(self, c: Chaser, t: float) -> float:
        """How likely chaser `c` is about and chasing her at t (see the top)."""
        since = t - c.hit
        chase = 1.0 if since <= CHASE_S[0] else max(0.0, (CHASE_S[1] - since) / (CHASE_S[1] - CHASE_S[0]))
        unseen = t - c.seen
        about = 1.0 if unseen <= SEEN_S else math.exp(-(unseen - SEEN_S) / LOST_S)
        return chase * about * (OTHER_DEATH_P if c.other else 1.0) * (DOUBT_P if c.doubt is not None else 1.0)             * (0.0 if c.dying else 1.0)

    # --- what it answers ----------------------------------------------------------------------------------------------

    def chasers_at(self, t: float, platform: int | None = None, min_p: float = 0.0) -> list[Guess]:
        """The chasers at t (on `platform`, if given), where each likeliest is, as likely as `min_p` at least; the
        likeliest first."""
        out = [self._guess(c, t) for c in self.chasers.values() if platform is None or c.platform == platform]
        return sorted((g for g in out if g.p > 0.0 and g.p >= min_p), key=lambda g: -g.p)

    def near(self, t: float, x: float, within: float, platform: int | None = None,
             min_p: float = 0.0) -> list[Guess]:
        """The chasers at t whose reach comes within `within` of x (on `platform`, if given), the nearest first."""
        return sorted((g for g in self.chasers_at(t, platform, min_p) if g.lo - within <= x <= g.hi + within),
                      key=lambda g: abs(g.x - x))

    def toucher(self, after: float | None = None) -> Touched | None:
        """Her latest touch's likeliest monster (one at or after `after`, if given), or None."""
        if not self.touches or (after is not None and self.touches[-1].t < after):
            return None
        return self.touches[-1]

    def record(self, t: float) -> dict:
        """As telemetry data: every chaser, its guess now, and the counts."""
        return {"chasers": [{**c.record(t), "at": round(g.x, 1), "p": round(g.p, 2)}
                            for c in self.chasers.values() for g in (self._guess(c, t),)],
                "counts": dict(self.counts)}

    # --- her touches --------------------------------------------------------------------------------------------------

    def _touched(self, e, t: float, by_id: dict) -> None:
        """A 'touch' event told in the look at t: the likeliest monster (see the top)."""
        data = e.data or {}
        at = data.get("at")
        at = e.t if at is None else min(at, e.t)
        damage = data.get("kind") != "miss"
        her = self._her_at(at)
        self.counts["touches"] += 1
        if her is None:
            self.touches.append(Touched(at, t, damage, None, None, 0, None, None, None, None, "none"))
            return
        hx, hy = her[1], her[2]
        cands = []  # (x, where it can be: lo, hi, kind, track, key, by, sure of its side)
        for c in self.chasers.values():
            if c.dying or abs(c.y - hy) > 120:
                continue
            g = self._guess(c, at)
            cands.append((g.x, g.lo, g.hi, c.kind, c.track, c.key, "chaser", g.way != 0 or t - c.seen <= DET_S))
        mine = {c.track for c in self.chasers.values() if c.track is not None}
        for tr in by_id.values():
            if tr.id in mine or getattr(tr, "killed", False) or getattr(tr, "state", "live") in ("dead", "offscreen"):
                continue
            if abs(tr.y - hy) > 120:
                continue
            seen = self._seen_of(tr, t)
            dt = 0.0 if seen is None else max(-0.5, min(0.5, at - seen))
            x = tr.x + (getattr(tr, "vx", 0.0) or 0.0) * dt
            spread = SPREAD_0 + SPREAD_RATE * self.walking.get(tr.kind, DEFAULT_SPEED) * abs(dt)
            cands.append((x, x - spread, x + spread, tr.kind, tr.id, None, "track", True))
        best = None
        for x, lo, hi, kind, track, key, by, sure in cands:
            reach = touch_box(kind, self.bodies.get(kind))[0] + HER_HALF + TOUCH_REACH
            if not lo - reach <= hx <= hi + reach:
                continue
            # (how likely its body was on hers: the share of where it can be that reaches her, the narrowest first)
            like = min(hi + reach, hx + reach) - max(lo - reach, hx - reach)
            like /= max(hi - lo + 2 * reach, 1.0)
            score = (like, -abs(x - hx))
            if best is None or score > best[0]:
                best = (score, x, kind, track, key, by, sure)
        if best is None:
            self.touches.append(Touched(at, t, damage, hx, None, 0, None, None, None, None, "none"))
            return
        _, x, kind, track, key, by, sure = best
        side = (1 if x > hx else -1 if x < hx else 0) if sure else 0
        self.touches.append(Touched(at, t, damage, hx, kind, side, x, track, key, abs(x - hx), by))

    def _knocks(self, t: float) -> None:
        """Her damage touches of the last looks with no knock judged: the way her place moved from her path's last before
        the touch to this look (KNOCK_DX), the side the monster was taken to be on put right by it (away from it)."""
        if not self.path:
            return
        now = self.path[-1]
        for i in range(len(self.touches) - 1, -1, -1):
            k = self.touches[i]
            if k.knock or not k.damage or k.her is None or now[0] <= k.t or now[0] - k.t > 1.0:
                if now[0] - k.t > 1.0:
                    break
                continue
            moved = now[1] - k.her
            if abs(moved) < KNOCK_DX:
                continue
            way = 1 if moved > 0 else -1
            self.touches[i] = Touched(k.t, k.told, k.damage, k.her, k.kind, -way, k.x, k.track, k.key, k.dx, k.by, way)


def _fold(x: float, way: int, go: float, edges: tuple[float, float] | None) -> tuple[float, int]:
    """Walked `go` units from x the way `way` (+1 right, -1 left, 0 standing), folding back at `edges` (None: none):
    where it is, and the way it's walking there."""
    if not way or go <= 0.0:
        return x, way
    if edges is None:
        return x + way * go, way
    lo, hi = edges
    span = hi - lo
    if span <= 0:
        return lo, way
    u = (x - lo) + way * go
    m = u % (2 * span)
    return (lo + m, way) if m <= span else (lo + 2 * span - m, -way)


@dataclass
class _Rec:
    """A track as the perception service's 'tracks' telemetry records it (`replay`): its fields as tracking.Track's
    that the belief reads; when it was last seen worked out from its frame count (`seen`) going up, its bar's read
    time not recorded (None: its bar as kept)."""
    id: int
    kind: str
    platform: int | None
    x: float
    y: float
    vx: float
    facing: int
    state: str
    bar: float | None
    last_seen: float
    first_seen: float
    killed: bool
    bar_at: float | None = None


def replay(records: Iterable[dict], model: MapModel | None = None,
           reach: tuple[float, float, float, float] | None = None, aggro: "Aggro | None" = None):
    """The belief run again on recorded telemetry (in time order: the perception service's records merged with the
    hunt's): her place from each 'scene' record (into her path, every frame's), her throws from the hunt's 'attack' records (their `side`, else the
    side of their target's x; their `track`, the one aimed at, where recorded), and a look
    at every 'tracks' record: its tracks (`_Rec`), with what happened since (the records' 'died' and 'touch' events; a
    track's `hits` going up a 'bar_hit', whose "by" isn't recorded: hers by her throws' paths, LATE_S). Yields (t, the
    belief) after each look. `aggro`: the belief to run (a new one by default)."""
    import re

    from .tracking import Event
    a = aggro if aggro is not None else Aggro(model, reach)
    her = None
    prev: dict[int, dict] = {}
    seen_at: dict[int, float] = {}
    first: dict[int, float] = {}
    events: list = []
    cause_re, by_re = re.compile(r"cause (\w+)"), re.compile(r"\bby (\d+)")
    for r in records:
        topic, t = r.get("topic"), r.get("t")
        if topic == "scene":
            h = r.get("her") or {}
            if h.get("x") is not None:
                her = (h["x"], h["y"], h.get("platform"))
                if not a.path or t > a.path[-1][0]:
                    a.path.append((t, *her))  # (her path at every frame recorded: the looks are fewer)
        elif topic == "attack":
            side = r.get("side")
            if side not in ("left", "right") and r.get("x") is not None and her is not None:
                side = "right" if r["x"] >= her[0] else "left"  # (records before 2026-11-02 have no side: the target's)
            if side in ("left", "right"):
                a.throw(t, side, r.get("track"))
        elif topic == "event" and r.get("kind") == "died" and r.get("track") is not None:
            text = r.get("text", "")
            cause, by = cause_re.search(text), by_re.search(text)
            events.append(Event("died", t, r["track"], {"cause": cause.group(1) if cause else None,
                                                        "by": int(by.group(1)) if by else None}))
        elif topic == "event" and r.get("kind") == "touch":
            events.append(Event("touch", t, None, {"kind": r.get("touch"), "at": t - (r.get("late") or 0.0)}))
        elif topic == "tracks":
            tracks = []
            for d in r.get("list", ()):
                tid = d["id"]
                p = prev.get(tid)
                first.setdefault(tid, t)
                if d.get("state") == "live" and (p is None or d.get("seen", 0) > p.get("seen", 0)):
                    seen_at[tid] = t
                if p is not None and d.get("hits", 0) > p.get("hits", 0):
                    events.append(Event("bar_hit", t, tid, {"bar": d.get("bar"), "by": None}))
                prev[tid] = d
                tracks.append(_Rec(tid, d.get("kind", ""), d.get("platform"), d["x"], d["y"], d.get("vx") or 0.0,
                                   d.get("facing") or 0, d.get("state", "live"), d.get("bar"),
                                   seen_at.get(tid, first[tid]), first[tid],
                                   d.get("state") == "dead" or (d.get("state") == "dying" and "gain" in d)))
            a.update(t, her, tracks, events)
            events = []
            yield t, a
