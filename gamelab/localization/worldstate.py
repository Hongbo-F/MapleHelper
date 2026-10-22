"""The continuous feed's step: one frame of the game in, the state of what's on screen out (WorldState), for
the feed (localization.feed) to hand to whoever asks and for `python -m gamelab watch` to publish. (Not
the simulator's world, gamelab.sim.world: this is what the stack sees of a world.)

Every frame: her place (the same look as Senses.look takes: her name tag, HP and MP, the ground, her minimap
dot, ropes), her EXP gains and HP drops (events, each at the frame it showed), and each monster tracked
re-found around where it should be (Eyes.refind: the template that matched it last, else its kind's).

The searches run off that frame loop (`Searches`: in the feed, a thread of its own, `SearchThread`), one at a
time, on the newest frame placed when one is due: monsters not tracked yet in her band every BAND_EVERY_S,
the whole screen every WIDE_EVERY_S; the drops only when asked (or, for `watch`, on a schedule of their own).
A consumer can ask for any of those on the next frame (`Asks`); asked while a search is out, they're owed and
handed over as soon as it's back, on the newest frame. What a search found comes back to a mailbox the
frame loop empties (`came_back`, and in each step after placing her, before the re-finds, so what it found
anew is re-found in that very frame): it's merged into the tracker as of its own frame (tracking.merge), and
that frame's state is given again (`late`), with what the searches found in it, as the frame loop alone would
have had it with the searches done inline. The tracker (localization.tracking) keeps each monster's id,
velocity, facing, bar and state, and tells what happened to them.

The state holds plain data (no map objects, no frame unless kept in-process), so it can go to another
process; `to_scene` turns it back into the Scene a look of the stack's would have been.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from ..perception.eyes import Observation, band_rows
from ..perception.hud import exp_changed
from ..perception.minimap import MINIMAP_BOX
from ..screen import HUD_TOP, SCALE
from .mapdata import MapModel
from .monsters import Monster
from .pose import Where
from .scene import Loot, Scene, find_loot
from .senses import Senses
from .tracking import DYING, HIDDEN, LIVE, Event, MonsterTracker, Track, View, overlap, same, snapshot

# Ropes are searched every frame: placed by a rope in one frame and by her dot alone in the next, her x (and
# with it every monster's) would step by up to a minimap pixel (11.4 units) from frame to frame, which the
# tracker would take for walking. In the columns where the map's ropes can show it's 8.6 ms of a look's 16.
ROPES_EVERY = 1
# s between searches of her band for monsters not followed yet, 5 a second. By time, not frames: counted every 6th
# frame, a feed slower than 30 frames a second (on the saved frames, 18 with the searches) searched it less often
# than a look's freshness needs (feed.MOBS_FRESH_S), so every look asked for one, and half the frames searched it.
BAND_EVERY_S = 0.2
WIDE_EVERY_S = 1.0    # s between searches of the whole screen (122 ms each, bench-look)
LOOT_EVERY_S = 0.5    # `watch`: s between searches for the drops on her platform ...
LOOT_WIDE_EVERY_S = 2.0  # ... and on every platform in view
HP_DROP = 0.004       # HP lost from one frame to the next that counts as a hit (control.navigator.HIT)
HIT_MERGE_S = 0.2     # a drop this soon after the last is the same hit, if the bar takes a few frames to go down


@dataclass(frozen=True)
class Schedule:
    """How often the step searches without being asked."""
    ropes_every: int = ROPES_EVERY       # frames
    band_s: float | None = BAND_EVERY_S  # s (None: only when asked)
    wide_s: float | None = WIDE_EVERY_S  # s (None: only when asked)
    loot_s: float | None = None          # s (None: only when asked)
    loot_wide_s: float | None = None


WATCH = Schedule(loot_s=LOOT_EVERY_S, loot_wide_s=LOOT_WIDE_EVERY_S)  # `watch`: the drops too, for the mini-map


@dataclass
class Asks:
    """What a consumer asks of the next frame shown at `after` or later."""
    after: float = 0.0
    band: bool = False        # her band searched for monsters
    wide: bool = False        # the whole screen
    loot: bool = False        # the drops on her platform ...
    loot_wide: bool = False   # ... or on every platform in view
    mesos_only: bool = False  # (only mesos: her inventory is full)

    def merge(self, other: "Asks | None") -> "Asks":
        """This ask and a newer one (`other`) in one: everything either wants, from the sooner time, and the
        newer's mesos_only (her inventory may have filled or freed since)."""
        if other is None:
            return self
        return Asks(min(self.after, other.after), self.band or other.band, self.wide or other.wide,
                    self.loot or other.loot, self.loot_wide or other.loot_wide, other.mesos_only)


@dataclass
class WorldState:
    """What one frame showed, placed on the map, and the tracks and events it brought. Plain data."""
    seq: int                       # the step's frame number
    t: float                       # when the screen showed it (time.monotonic's clock)
    visible: bool                  # False: the game wasn't wholly on screen; nothing was read (see `why`)
    why: str = ""
    tag: tuple[int, int] | None = None   # Observation's and Scene's fields
    tag_seen: bool = True
    hp: float = 0.0
    mp: float = 0.0
    ground: tuple[int, int] | None = None
    dot: tuple[float, float] | None = None
    ropes: list = field(default_factory=list)
    exp: bytes = b""               # her EXP number's text pixels, packed (np.packbits) ...
    exp_shape: tuple = ()          # ... and their shape
    where: tuple | None = None     # her map position: x, y, platform id (None: none), dot
    how: str = "dot"
    monsters: list[Monster] = field(default_factory=list)  # every monster seen in this frame, placed
    band_at: float | None = None   # the frame time of the last search of her band (or the whole screen) come back ...
    wide_at: float | None = None   # ... and of the whole screen
    loot: list[Loot] | None = None  # the drops seen, if this frame was searched for them (None: it wasn't) ...
    looked: list = field(default_factory=list)  # ... where (Scene.looked)
    loot_wide: bool = False        # ... on every platform in view (else on hers)
    mesos_only: bool = False
    tracks: list[Track] = field(default_factory=list)  # the monsters followed, alive (seen or not) or just dead
    events: list[Event] = field(default_factory=list)  # what happened, as of this frame
    work: dict = field(default_factory=dict)  # what this frame took: re-finds, and `s` on the real clock; a
                                              # late state, its searches (Found.work)
    frame: np.ndarray | None = None  # the half-size frame (in-process; the feed's child puts it in its ring)
    slot: int | None = None          # ... the ring's slot it's in
    late: bool = False             # this frame's state given again, now with what its searches found (seq, t and
                                   # slot are the frame's; given after the states of frames since)

    @property
    def placed(self) -> bool:
        return self.visible and self.tag is not None and self.where is not None

    def exp_pixels(self) -> np.ndarray | None:
        if not self.exp_shape:
            return None
        n = int(np.prod(self.exp_shape))
        return np.unpackbits(np.frombuffer(self.exp, np.uint8), count=n).reshape(self.exp_shape).astype(bool)


# --- the searches, off the frame loop -------------------------------------------------------------------

@dataclass
class Job:
    """One frame's searches, handed to the worker as soon as she's placed in it: that frame's look placed (its
    Scene: all the searches need) and what it showed of the map; then, once the frame loop is done with the
    frame (before the answer can be merged), the tracks re-found in it and its state as the frame loop gave it
    (the late state is built on it)."""
    seq: int
    t: float
    scene: Scene
    view: View
    known: dict[int, Monster] = field(default_factory=dict)  # track id -> where it was re-found in this frame (the
    state: WorldState | None = None                          # re-finds the tracker took)
    band: bool = False             # her band searched for monsters ...
    wide: bool = False             # ... or the whole screen
    loot: bool = False             # the drops on her platform ...
    loot_wide: bool = False        # ... or on every platform in view
    mesos_only: bool = False


@dataclass
class Found:
    """What a frame's searches found: the worker's answer, posted to the mailbox."""
    job: Job
    monsters: list[Monster] | None = None  # None: no monsters were searched for
    loot: list[Loot] | None = None         # None: no drops were
    looked: list = field(default_factory=list)  # where the drops were looked for (Scene.looked)
    took: float = 0.0                      # s it took (the real clock)

    @property
    def work(self) -> dict:
        """What the searches did, and (`s`) took on the real clock."""
        j = self.job
        return {"band": int(j.band and not j.wide), "wide": int(j.wide), "loot strips": len(self.looked),
                "s": self.took}


def search(senses: Senses, job: Job) -> Found:
    """A frame's searches (Senses.search, find_loot), as the worker runs them."""
    began = time.perf_counter()
    s, found = job.scene, Found(job)
    if job.band or job.wide:
        found.monsters = senses.search(s, wide=job.wide)
    if job.loot or job.loot_wide:
        before = len(s.looked)
        found.loot = find_loot(s, senses.model, senses.eyes, job.loot_wide, job.mesos_only)
        found.looked = list(s.looked[before:])
    found.took = time.perf_counter() - began
    return found


class Searches:
    """Where a frame's searches run while the frame loop goes on: one job at a time, its answer left in a mailbox
    that holds one. The frame loop hands over the next only once it has taken the last (`free`), so an answer is
    never overwritten unread; the frames in between are the frame loop's alone. `wake`, if set, is called once an
    answer is in (the frame loop may be waiting for asks or for its next frame)."""
    wake: Callable[[], None] | None = None

    def free(self) -> bool:
        raise NotImplementedError

    def submit(self, job: Job) -> None:
        raise NotImplementedError

    def take(self, by: float | None = None) -> Found | None:
        """The answer, if it's in (`by`: in by then, on a clock of the caller's; the real ones ignore it).
        A search that failed raises its error here, in the frame loop."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class InlineSearches(Searches):
    """On the caller's thread, at once: the answer is in as soon as the job is handed over (tests, bench-look)."""

    def __init__(self, senses: Senses):
        self.senses, self._box = senses, None

    def free(self) -> bool:
        return self._box is None

    def submit(self, job: Job) -> None:
        self._box = search(self.senses, job)

    def take(self, by: float | None = None) -> Found | None:
        out, self._box = self._box, None
        return out


class Worker(Searches):
    """`fn(job)` on a thread of its own, one job at a time, its answer (or the error it raised) left for `take`.
    Python's lock: the monster search lets go of it while OpenCV matches and its C peaks run (perception.mobs),
    so the frame loop mostly goes on meanwhile (see SearchThread)."""

    def __init__(self, fn: Callable, name: str = "feed-worker"):
        self.fn = fn
        self._cond = threading.Condition()
        self._job = None        # handed over, not started
        self._busy = False      # started, not done
        self._box = None        # done, not taken
        self._stop = False
        self.done = 0           # jobs done
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def free(self) -> bool:
        with self._cond:
            return self._job is None and not self._busy and self._box is None

    def submit(self, job) -> None:
        with self._cond:
            if self._stop:
                return
            self._job = job
            self._cond.notify_all()

    def take(self, by: float | None = None):
        with self._cond:
            out, self._box = self._box, None
        if isinstance(out, BaseException):
            raise out
        return out

    def close(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._job is not None or self._stop)
                if self._stop:
                    return
                job, self._job, self._busy = self._job, None, True
            try:
                out = self.fn(job)
            except Exception as e:  # a bug: raised in the frame loop by `take`
                out = e
            with self._cond:
                self._box, self._busy = out, False
                self.done += 1
            wake = self.wake
            if wake is not None:
                wake()


class SearchThread(Worker):
    """The searches on a thread of the feed's process, beside its frame loop. Measured on the 17 saved frames
    (the frame loop's step back to back, the searches 5 a second in her band and once a second over the whole
    screen; README, Continuous perception): the frame loop's step took 21.4 ms on average beside them (p90 35.7)
    against 19.4 (24.8) alone, and the searches 30.9 and 124 ms (median) against 26.0 and 93 alone. In a process
    of their own (their own detector and threads, the frame from the ring) the step took 18.9 ms (26.7) and the
    searches 30.6 and 103, for a second set of templates and threads and the frame copied over: the thread is
    enough for 30 frames a second."""

    def __init__(self, senses: Senses):
        super().__init__(lambda job: search(senses, job), "feed-searches")


# --- the step ---------------------------------------------------------------------------------------------

class WorldStep:
    """Frame after frame of the game, turned into WorldStates. `senses`: placing her and searching for monsters
    (its eyes look at the frames, its map places what they see); `throws`: whether her throws will be told
    (`throw`, for whose kills are whose: localization.tracking); `searches`: where the searches run (by default
    at once on the caller's thread, InlineSearches; the feed's, a SearchThread)."""

    def __init__(self, senses: Senses, throws: bool = True, schedule: Schedule = Schedule(),
                 searches: Searches | None = None):
        self.senses, self.eyes, self.model = senses, senses.eyes, senses.model
        self.tracker = MonsterTracker(self.model, throws)
        self.schedule = schedule
        self.searches = searches if searches is not None else InlineSearches(senses)
        self.seq = 0
        self.band_at: float | None = None
        self.wide_at: float | None = None
        self.loot_at: float | None = None
        self.loot_wide_at: float | None = None
        self.mesos_only = False
        self.owed: Asks | None = None    # searches asked for and not handed over yet (one was out)
        self._last: Job | None = None    # the newest frame placed, as a job for the worker (its searches not set)
        self._newest: tuple[View, float] | None = None  # what the newest frame placed showed of the map, and when
        self._late: list[WorldState] = []  # late states made in a step, for came_back
        self._exp: np.ndarray | None = None
        self._hp: float | None = None
        self._hit_at = -1e9
        self.counts = {"frames": 0, "not visible": 0, "placed": 0, "re-finds": 0, "re-found": 0, "fallbacks": 0,
                       "band": 0, "wide": 0, "loot": 0, "loot wide": 0, "exp gains": 0, "hits": 0, "searches": 0}

    def throw(self, t: float, side: str) -> None:
        """She threw toward `side` at t."""
        self.tracker.throw(t, side)

    def step(self, look: Callable[[bool], Observation], t: float, visible: bool = True, asks: Asks | None = None,
             why: str = "", by: float | None = None) -> WorldState:
        """The frame shown at t. `look(ropes)` reads it (Eyes.look_at, or the simulator's FakeEyes.observe),
        searching for ropes or not; with `visible` False nothing is read. `asks`: what was asked of it (searched
        on it, or as soon as the search out is back). Once she's placed, what the searches found since the last
        frame is merged (`by`: what's in by then, on the simulator's clock), the searches due are handed over on
        this frame, and then the monsters followed are re-found; the late states the merge makes, and those of
        searches back by the time the caller next asks, come from `came_back`."""
        began = time.perf_counter()
        self.seq += 1
        self.counts["frames"] += 1
        if not visible:
            self.counts["not visible"] += 1
            self.owed = None  # whoever asked takes the look itself
            return WorldState(self.seq, t, False, why=why)
        if asks is not None and (asks.band or asks.wide or asks.loot or asks.loot_wide):
            self.owed = asks if self.owed is None else self.owed.merge(asks)
        sch = self.schedule
        obs = look(self.seq % sch.ropes_every == 0)
        s = self.senses.place(obs)
        events = self._vitals(s, t)
        placed = s.tag is not None and s.where is not None
        view = self._view(s) if placed else None
        if placed:
            self._newest = (view, t)
        self._merge(self.searches.take(by))  # before the re-finds: what they found anew is re-found in this frame
        job = None
        if placed:  # the searches due handed over now, beside the re-finds (they need only her place)
            job = self._last = Job(self.seq, t, s, view)
            self._hand_over()
        work = {"re-finds": 0, "fallbacks": 0}
        fallbacks = getattr(self.eyes.detector, "fallbacks", 0)
        refound: dict[int, Monster] = {}
        if placed:
            self.counts["placed"] += 1
            for tr in self.tracker.live():
                if tr.state not in (LIVE, HIDDEN, DYING):
                    continue
                box = self._box(s, tr, t)
                if box is None:
                    continue
                work["re-finds"] += 1
                m = self.eyes.refind(s, tr.mob, box, fallback=tr.state != HIDDEN)
                if m is not None:
                    refound[tr.id] = s.place(m, self.model, self.eyes.bar(s, m))
            work["fallbacks"] = getattr(self.eyes.detector, "fallbacks", 0) - fallbacks
            self.counts["re-finds"] += work["re-finds"]
            self.counts["re-found"] += len(refound)
            self.counts["fallbacks"] += work["fallbacks"]
        her = (s.where.x, s.where.y) if placed else None
        events += self.tracker.update(t, refound, [], view, her)
        seen: list[Monster] = []  # re-found (two tracks may have found one)
        for m in refound.values():
            if not any(same(m.mob, k.mob) for k in seen):
                seen.append(m)
        exp = np.zeros((0,), bool) if s.exp is None else np.asarray(s.exp, bool)
        where = None if s.where is None else (
            s.where.x, s.where.y, None if s.where.platform is None else s.where.platform.id, s.where.dot)
        work["s"] = time.perf_counter() - began  # (the real clock: what the frame loop's step took)
        st = WorldState(
            self.seq, t, True, tag=s.tag, tag_seen=s.tag_seen, hp=s.hp, mp=s.mp, ground=s.ground, dot=s.dot,
            ropes=list(s.ropes), exp=np.packbits(exp).tobytes() if s.exp is not None else b"",
            exp_shape=exp.shape if s.exp is not None else (), where=where, how=s.how, monsters=seen,
            band_at=self.band_at, wide_at=self.wide_at, mesos_only=self.mesos_only,
            tracks=snapshot(list(self.tracker.tracks.values())), events=events, work=work, frame=s.frame)
        if job is not None:  # (its answer is merged only after this) the re-finds the tracker took, as inline
            job.known.update({tid: m for tid, m in refound.items() if tid in self.tracker.refound_seen})
            job.state = st
        return st

    def came_back(self, by: float | None = None) -> list[WorldState]:
        """The late states of frames whose searches are back: merged in the last step, and (now) any back since
        it (`by`, as for `step`). Then the searches due are handed over, on the newest frame placed."""
        found = self.searches.take(by)
        if found is not None:
            self._merge(found)
            self._hand_over()
        out, self._late = self._late, []
        return out

    def close(self) -> None:
        self.searches.close()

    def _hand_over(self) -> None:
        """The searches due (asked, or by the schedule), handed to the worker on the newest frame placed, if it's
        free: what's asked only on a frame shown at the time asked for or later."""
        job = self._last
        if job is None or not self.searches.free():
            return
        t, sch = job.t, self.schedule
        owed = self.owed if self.owed is not None and t >= self.owed.after - 1e-9 else None
        asked = owed or Asks()
        wide = asked.wide or (sch.wide_s is not None and (self.wide_at is None or t - self.wide_at >= sch.wide_s))
        band = asked.band or (sch.band_s is not None and (
            self.band_at is None or t - self.band_at >= sch.band_s - 1e-6))
        loot_wide = asked.loot_wide or (sch.loot_wide_s is not None and (
            self.loot_wide_at is None or t - self.loot_wide_at >= sch.loot_wide_s))
        loot = loot_wide or asked.loot or (sch.loot_s is not None and (
            self.loot_at is None or t - self.loot_at >= sch.loot_s))
        if not (wide or band or loot):
            return
        if owed is not None:
            if owed.loot or owed.loot_wide:
                self.mesos_only = owed.mesos_only
            self.owed = None
        self._last = None  # a frame is searched once
        job.band, job.wide, job.loot, job.loot_wide = band or wide, wide, loot, loot_wide
        job.mesos_only = self.mesos_only
        self.tracker.searching = job.t
        self.searches.submit(job)

    def _merge(self, found: Found | None) -> None:
        """What a frame's searches found, merged into the tracker as of that frame (tracking.merge, with the
        tracks as they are now and what the newest frame placed shows), and that frame's late state made."""
        if found is None:
            return
        job = found.job
        self.tracker.searching = None
        now, t_now = self._newest or (None, None)
        events: list[Event] = []
        seen = list(job.state.monsters)
        band_at, wide_at = job.state.band_at, job.state.wide_at
        if found.monsters is not None:
            known = list(job.known.values())
            fresh = [m for m in found.monsters if not any(m.kind == k.kind and overlap(m.mob, k.mob) for k in known)]
            events = self.tracker.merge(job.t, fresh, set(job.known), job.view, now, t_now)
            again = list(seen)
            seen += [m for m in fresh if not any(m.kind == k.kind and overlap(m.mob, k.mob) for k in again)]
            self.band_at = band_at = job.t if self.band_at is None else max(self.band_at, job.t)
            self.counts["wide" if job.wide else "band"] += 1
            if job.wide:
                self.wide_at = wide_at = job.t if self.wide_at is None else max(self.wide_at, job.t)
        if found.loot is not None:
            self.loot_at = job.t
            self.counts["loot"] += 1
            if job.loot_wide:
                self.loot_wide_at = job.t
                self.counts["loot wide"] += 1
        self.counts["searches"] += 1
        self._late.append(replace(
            job.state, monsters=seen, band_at=band_at, wide_at=wide_at, loot=found.loot, looked=found.looked,
            loot_wide=job.loot_wide and found.loot is not None, mesos_only=job.mesos_only,
            tracks=snapshot(list(self.tracker.tracks.values())), events=events, work=found.work, late=True))

    def _vitals(self, s: Scene, t: float) -> list[Event]:
        """Her EXP number changed since the last frame (a kill of hers: the tracker puts it on a monster), or
        her HP dropped (a hit)."""
        out = []
        if exp_changed(self._exp, s.exp):
            out.append(Event("exp_gain", t))
            self.tracker.gain(t)
            self.counts["exp gains"] += 1
        if s.exp is not None:
            self._exp = s.exp
        if self._hp is not None and s.hp < self._hp - HP_DROP:
            new = t - self._hit_at > HIT_MERGE_S
            out.append(Event("hp_drop", t, data={"lost": round(self._hp - s.hp, 4), "hp": round(s.hp, 3), "new": new}))
            self._hit_at = t
            self.counts["hits"] += new
        self._hp = s.hp
        return out

    def _box(self, s: Scene, tr: Track, t: float) -> tuple[int, int, int, int] | None:
        """Where track `tr`'s picture should be in this frame (half-size px), from where it should be on the
        map; None if that's off the play area."""
        m = tr.mob
        cx = s.screen_x(self.tracker.predicted(tr, t)) + (m.x - m.foot_x)
        cy = s.screen_y(tr.y) + (m.y - m.feet_y)
        h, w = s.frame.shape[:2]
        box = (round(cx - m.w / 2), round(cy - m.h / 2), round(cx + m.w / 2), round(cy + m.h / 2))
        if box[2] <= 0 or box[0] >= w or box[3] <= 0 or box[1] >= int(h * HUD_TOP):
            return None
        return box

    @staticmethod
    def _view(s: Scene) -> View:
        h, w = s.frame.shape[:2]
        bottom = int(h * HUD_TOP)
        mx, my = MINIMAP_BOX[2] / SCALE, MINIMAP_BOX[3] / SCALE
        return View(s.map_x(0), s.map_x(w), s.map_y(0), s.map_y(bottom), w, bottom,
                    (s.map_x(0), s.map_y(0), s.map_x(mx), s.map_y(my)))


def to_scene(st: WorldState, model: MapModel, band: tuple[float, float], mobs: bool = True, wide: bool = False,
             loot: bool = False, mesos_only: bool = False, frame: np.ndarray | None = None) -> Scene:
    """The Scene a look (Senses.look with these arguments) would have given in the frame of state `st`: her
    place; with `mobs`, the monsters seen in her band (or, if `wide`, all of them); with `loot` (and the
    state searched for drops), the drops on her platform (or, if `wide`, every one in view)."""
    p = None if st.where is None or st.where[2] is None else model.platform(st.where[2])
    where = None if st.where is None else Where(st.where[0], st.where[1], p, st.where[3])
    s = Scene(t=st.t, frame=st.frame if frame is None else frame, tag=st.tag, hp=st.hp, mp=st.mp, ground=st.ground,
              dot=st.dot, ropes=list(st.ropes), exp=st.exp_pixels(), where=where, how=st.how, tag_seen=st.tag_seen)
    if mobs and s.tag is not None and s.where is not None:
        if wide:
            s.monsters = list(st.monsters)
        else:
            top, bottom = band_rows(s, *band)
            s.monsters = [m for m in st.monsters if top <= m.mob.y - m.mob.h // 2 and m.mob.y + m.mob.h // 2 <= bottom]
        s.searched, s.wide = True, wide
    if loot and st.loot is not None:
        mine = None if p is None else p.id
        keep = (lambda pid: True) if wide else (lambda pid: pid == mine)
        s.loot = [d for d in st.loot if keep(d.platform) and (d.mesos or not mesos_only)]
        s.looked = [lk for lk in st.looked if keep(lk[0])]
    return s
