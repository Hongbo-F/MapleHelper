"""The continuous feed's step: one frame of the game in, the state of what's on screen out (WorldState), for
the feed (localization.feed) to hand to whoever asks and for `python -m gamelab watch` to publish. (Not
the simulator's world, gamelab.sim.world: this is what the stack sees of a world.)

Every frame: her place (the same look as Senses.look takes: her name tag, HP and MP, the ground, her minimap
dot, ropes, the other players' dots), her EXP gains and HP drops (events, each at the frame it showed: a gain with
her EXP number as read then, perception.exp_number, the kills its size was going to the tracker by kind,
localization.gains), and each monster tracked re-found around where it should be (Eyes.refind: the template that
matched it last, else its kind's; first only as far around as it can have got since it was last seen, and a window
the very same as at its last re-find answered as then: NEAR_FIRST, `_memos`; all of a frame's re-finds' matches in
one call in C, ONE_CALL: Eyes.refind_all), each sighting with its track's id (Monster.track); and those re-found under
or lost lately looked at for their death (DEATHS: perception.deaths, the game data's die1 frames where each was).

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
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from .. import native
from ..character import THIEF_SHAPE
from ..perception import deaths, exp_number, hp_number
from ..perception.eyes import Observation, band_rows
from ..perception.exp_number import ExpReading
from ..perception.framecheck import Problem
from ..perception.hud import exp_changed
from ..perception.minimap import minimap_box
from ..perception.mobs import Mob, MobDetector, _one_opencv_thread, near_pad
from ..screen import STRETCHED, Geometry
from .gains import ExpMeter, Gain
from .mapdata import MapModel
from .monsters import DEFAULT_SPEED, Monster
from .pose import Where
from .scene import Loot, Scene, find_loot
from .senses import Senses
from .tracking import (DYING, EDGE_PX, HIDDEN, LIVE, SPEED_CLAMP, Event, MonsterTracker, Track, View,
                       overlap, same, snapshot)

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
# A monster followed has its HP bar read (perception.hud.hp_bar, 1.1 ms each) in a frame while it shows one (it did
# within BAR_SHOWN_S: a frame or two) or a throw of hers went its way in the last tracking.STAR_WINDOW; else every
# BAR_EVERY-th frame, its track's bar as last read meanwhile (a monster nobody hits shows none). Read in every frame, the
# bars were 74% of what the simulator's feed cost (loop_tree_lie_detector: 37.5 s of its 41 in SyncFeed.tick), and live
# pays for them as much. One seen again after BAR_UNSEEN_S or more unseen (hidden under her damage numbers, 1-3 s in
# game) has it read in the frame it's back in: its bar as last read is from before what hid it, and carried over, the
# probes read a star that hit as a miss (test_probe's hidden target: 2 of 30 so, with every star a hit).
BAR_EVERY = 5
BAR_SHOWN_S = 0.25
BAR_UNSEEN_S = 0.15
HIT_MERGE_S = 0.2     # a drop this soon after the last is the same hit, if the bar takes a few frames to go down
# Her HP number's `before` (an `hp_read` event's: the HP it said at the change before, a drop from it a touch, exactly) is
# known only over frames all read: a frame not read (not visible, failing its checks) or a stretch over HP_GAP_S between
# two frames read has the number forgotten (hp_number.Follow.see(None)), the next read's `before` None. Two touches can't
# both come within it (live none within 1.25 s of the one before: sim.world.INVULNERABLE_LIVE); a potion between makes
# a drop smaller, never larger. Without it a drop over 60 unread frames was kept as one touch of their two (the review of
# 2026-10-28: D:/MapleHelper-scratch/safety3/review-evidence/demo/gap_demo.py).
HP_GAP_S = 1.0
# A frame's re-finds run at once on a pool of this many threads (the real monster detector's: its masked matches, in C
# or OpenCV, let go of Python's lock; a lone re-find on the frame loop's thread, the hand-off costing more), each with
# its HP bar's read if that's due, and the frame loop places them in turn: the states are one thread's. In a crowd
# re-finds were most of the frame loop's step, 6 ms a monster followed (the CPU review, 2026-10-26: 42 ms a frame with
# one followed, 110 with 12), and the loop made 15 frames a second of the 30 asked. With the masked match in C, on the
# service's frame loop replayed on 600 frames of 2026-10-26 01:01's crowd (D:/MapleHelper-scratch/pconst/pnative, 30 a
# second asked, the machine 70-80% busy): a frame's 11 re-finds 13.4-14.8 ms on 4 threads (3 runs), 30 frames a second,
# the step 19.8-21.5 ms (median; p90 30.6-33.0); on 3 threads the step 23.0 (34.2), on 6 19.2 (29.2) for more CPU.
REFIND_THREADS = 4
# Each track is looked for first only as far around where it should be as it can have got since it was last seen
# (mobs.near_pad, MobDetector.refind's `near`), and its windows' answers are kept from frame to frame (`_memos`, refind's
# `memo`): on both Lie Detectors' saved frames, each held 3 times (1/30 s apart), the masked matches' work a frame 39-40%
# less, and as saved (0.1 s apart: the near window is REFIND_PAD's) 2-3%; every state and every re-find's answer as
# before, 30,013 states (D:/MapleHelper-scratch/pfast/pfr, 2026-10-26; README, Fewer re-find matches).
NEAR_FIRST = True
# A frame's re-finds, all of their masked matches, in one call in C on REFIND_THREADS threads of its own (Eyes.refind_all,
# native.c's rf_*), Python's lock let go, each one's Python done as the C hands it back, where each was a task of the
# pool's, a C call a picture list, numpy's slices and copies between: the same answers (0 of 30,480 states differ on
# both Lie Detectors' before/, as saved and held 3 times); on the crowd's [700:1300] of 01:01 held 3 times, the re-finds
# 5.1-5.6 ms a frame against 6.5-7.2, and the frame loop replayed 0.11-0.14 cores less at 30 frames a second
# (D:/MapleHelper-scratch/pfast/pfr3, 2026-10-26; README, One call for the re-finds). Without the C, as before.
ONE_CALL = True
# A frame's re-finds sent to the GPU ahead of its look (native.AHEAD: the GPU switch's `ahead`;
# MobDetector.refind_ahead): at the frame's start each followed monster's window where the frame before's camera puts
# it, AHEAD_MORE px (x, y) wider each side than the re-find's own, so that the GPU's passes go on while the look places
# her; the frame's call answers each window lying inside the one sent from it, the rest on the C's threads (native.c,
# gpumatch.c's rf_ahead). The camera moved no more than that from frame to frame on 83-84% of the jobs of the two Lie
# Detectors' saved frames (2 px across, 6 up or down; 4 and 4: 82%; D:/MapleHelper-scratch/gsmooth/hide/cover), for
# 1.36 times the GPU's work on the windows themselves.
AHEAD_MORE = (2, 6)
# Whether the frame step looks for the deaths of the monsters it follows (perception.deaths: every EVERY-th frame a
# track re-found under REFOUND_UNDER or lost RECENT_S or less ago, its die1 frames where it was), and tells the tracker
# of each seen (tracking.seen_dying: her kill's monster put right, another player's dead at once). With real eyes, the
# die1 frames from the game data, read as the step is made (0.5 s); the simulator's eyes have their own (`deaths`,
# sim.fakes.FakeDeaths), or none. 0.5 ms a frame on the re-finds' pool, replayed on the 23:10 recording (0.72 ms on one
# thread; perception.deaths).
DEATHS = True


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
    # Why nothing may be acted on from this frame (perception.framecheck): the game not wholly on screen (with `visible`
    # False: its window's problem), or a frame that failed its checks (the minimap window or the HUD not where they
    # should be, the capture offset, the map's minimap picture or her name tag not found), or her place off the map
    # (localization.placecheck). Nothing but the bars is read from it; None: it's of the game as expected.
    bad: Problem | None = None
    tag: tuple[int, int] | None = None   # Observation's and Scene's fields
    tag_seen: bool = True
    hp: float = 0.0
    mp: float = 0.0
    ground: tuple[int, int] | None = None
    dot: tuple[float, float] | None = None
    others: list = field(default_factory=list)   # other players' minimap dots (perception.minimap.Marker)
    players: list = field(default_factory=list)  # ... the players they are (localization.players.Player)
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
    frame: np.ndarray | None = None  # the look's frame (in-process; the feed's child puts it in its ring)
    slot: int | None = None          # ... the ring's slot it's in
    late: bool = False             # this frame's state given again, now with what its searches found (seq, t and
                                   # slot are the frame's; given after the states of frames since)
    geo: Geometry = STRETCHED      # the frame's geometry (Observation.geo)

    epoch: int = 0                 # the perception service's run it's of (localization.service): its ring's frames
    hp_of: tuple | None = None     # her HP number as read at its last change, if it read: (HP, max HP); None: not
                                   # known (perception.hp_number: read at every change of its pixels, an `hp_read`
                                   # event each)

    @property
    def placed(self) -> bool:
        return self.visible and self.bad is None and self.tag is not None and self.where is not None

    @property
    def usable(self) -> bool:
        """Whether the stack may act on it: of the game as expected (placed or not: her dot may be hidden)."""
        return self.visible and self.bad is None

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
        self.since: float | None = None  # when the job being done began (time.monotonic; None: none): the service's
                                         # watchdog's (localization.service: a search that never ends)
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
                self.since = time.monotonic()
            try:
                out = self.fn(job)
            except Exception as e:  # a bug: raised in the frame loop by `take`
                out = e
            with self._cond:
                self._box, self._busy, self.since = out, False, None
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
    at once on the caller's thread, InlineSearches; the feed's, a SearchThread); `refind_threads`: how many re-finds run
    at once (by default REFIND_THREADS with a real monster detector; the simulator's fakes re-find on the caller's
    thread, in Python)."""

    def __init__(self, senses: Senses, throws: bool = True, schedule: Schedule = Schedule(),
                 searches: Searches | None = None, refind_threads: int | None = None):
        self.senses, self.eyes, self.model = senses, senses.eyes, senses.model
        if refind_threads is None:
            refind_threads = REFIND_THREADS if isinstance(getattr(self.eyes, "detector", None), MobDetector) else 1
        self._refinds = (ThreadPoolExecutor(refind_threads, "refinds", _one_opencv_thread) if refind_threads > 1
                         else None)
        self._refind_threads = refind_threads
        # Each track's re-find windows and what they gave (MobDetector.refind's `memo`), by its id: with a real monster
        # detector (the simulator's fakes re-find as always).
        real = isinstance(getattr(self.eyes, "detector", None), MobDetector)
        self._memos: dict[int, dict] | None = {} if real else None
        self.tracker = MonsterTracker(self.model, throws, getattr(senses, "shape", THIEF_SHAPE))  # (her Shape)
        self._real = real
        self._watch_of: tuple[float, deaths.DeathWatch | None] | None = None  # (its look frame's scale, the watch)
        if real and DEATHS and (geo := getattr(self.eyes, "geo", None)) is not None:
            self._watch_of = (geo.frame_scale, self._death_watch(geo.frame_scale))  # (made now: 0.5 s, not mid-hunt)
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
        self.meter = ExpMeter(self.model.mob_exp)  # what her EXP gains were, for the tracker
        self.gains: deque[Gain] = deque(maxlen=64)  # ... the newest of them, as told (for watch's 'exp' records)
        self._hp: float | None = None
        self._hit_at = -1e9
        self.hp_number = hp_number.Follow()  # her HP number, read whenever its pixels change (_vitals) ...
        self._hp_number_t: float | None = None  # ... and the last frame it was seen on (HP_GAP_S)
        if real and (geo := getattr(self.eyes, "geo", None)) is not None:
            hp_number.warm(geo)  # (its cells made now: 190 ms, not at the first frame)
        self._cam: tuple | None = None   # the last frame placed's camera (her tag x, feet y, map x, y): AHEAD's
        self._ahead: int | None = None   # this frame's token, if its re-finds were sent ahead
        self.counts = {"frames": 0, "not visible": 0, "placed": 0, "re-finds": 0, "re-found": 0, "fallbacks": 0,
                       "band": 0, "wide": 0, "loot": 0, "loot wide": 0, "exp gains": 0, "exp read": 0, "hits": 0,
                       "searches": 0}

    def throw(self, t: float, side: str, attack: str | None = None) -> None:
        """She threw toward `side` at t (with `attack`: its id, if told)."""
        self.tracker.throw(t, side, attack)

    def step(self, look: Callable[[bool], Observation], t: float, visible: bool = True, asks: Asks | None = None,
             why: str = "", by: float | None = None, bad: Problem | None = None) -> WorldState:
        """The frame shown at t. `look(ropes)` reads it (Eyes.look_at, or the simulator's FakeEyes.observe),
        searching for ropes or not; with `visible` False nothing is read (`why`, and `bad`, the window's problem, say
        why). A frame that fails its checks has its Problem in the state (`bad`). `asks`: what was asked of it (searched
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
            self.hp_number.see(None)  # (what her HP number did meanwhile isn't known: HP_GAP_S)
            return WorldState(self.seq, t, False, why=why, bad=bad or Problem("window", why or "not visible"))
        if asks is not None and (asks.band or asks.wide or asks.loot or asks.loot_wide):
            self.owed = asks if self.owed is None else self.owed.merge(asks)
        sch = self.schedule
        self._ahead = self._send_ahead(look, t)
        obs = look(self.seq % sch.ropes_every == 0)
        s = self.senses.place(obs)
        events = self._vitals(s, t) if s.bad is None else []  # (a frame that failed its checks: its EXP and HP aren't
                                                               # the game's, and the next good frame's are compared)
        if s.bad is not None:
            self.hp_number.see(None)  # (her HP number's isn't known across it: HP_GAP_S)
        placed = s.tag is not None and s.where is not None
        view = self._view(s) if placed else None
        if placed:
            self._newest = (view, t)
            self._cam = (s.tag[0], s.floor_y, s.where.x, s.where.y)
        self._merge(self.searches.take(by))  # before the re-finds: what they found anew is re-found in this frame
        job = None
        if placed:  # the searches due handed over now, beside the re-finds (they need only her place)
            job = self._last = Job(self.seq, t, s, view)
            self._hand_over()
        work = {"re-finds": 0, "fallbacks": 0}
        fallbacks = getattr(getattr(self.eyes, "detector", None), "fallbacks", 0)
        refound: dict[int, Monster] = {}
        unread: set[int] = set()  # the tracks re-found whose bars weren't read (_bar_due)
        if placed:
            self.counts["placed"] += 1
            todo = []  # (the track, where its picture should be)
            for tr in self.tracker.live():
                if tr.state not in (LIVE, HIDDEN, DYING):
                    continue
                box = self._box(s, tr, t)
                if box is None:
                    continue
                work["re-finds"] += 1
                todo.append((tr, box))
            if self._memos is not None:  # (made here, on this thread: the pool's threads only use them)
                self._memos = {tr.id: self._memos.get(tr.id, {}) for tr, _ in todo}
            for (tr, _), (m, read, bar) in zip(todo, self._refind_all(s, todo, t)):
                if m is not None:
                    refound[tr.id] = replace(s.place(m, self.model, bar), track=tr.id)
                    if not read:
                        unread.add(tr.id)
            work["fallbacks"] = getattr(getattr(self.eyes, "detector", None), "fallbacks", 0) - fallbacks
            self.counts["re-finds"] += work["re-finds"]
            self.counts["re-found"] += len(refound)
            self.counts["fallbacks"] += work["fallbacks"]
            if DEATHS:
                self._deaths(s, t, refound, work)
        her = (s.where.x, s.where.y) if placed else None
        events += self.tracker.update(t, refound, [], view, her, s.players)
        seen: list[Monster] = []  # re-found (two tracks may have found one)
        for tid, m in refound.items():
            if tid in unread and (tr := self.tracker.tracks.get(tid)) is not None:
                m = replace(m, bar=tr.bar)  # (its bar as last read)
            if not any(same(m.mob, k.mob) for k in seen):
                seen.append(m)
        exp = np.zeros((0,), bool) if s.exp is None else np.asarray(s.exp, bool)
        where = None if s.where is None else (
            s.where.x, s.where.y, None if s.where.platform is None else s.where.platform.id, s.where.dot)
        work["s"] = time.perf_counter() - began  # (the real clock: what the frame loop's step took)
        st = WorldState(
            self.seq, t, True, tag=s.tag, tag_seen=s.tag_seen, hp=s.hp, mp=s.mp, ground=s.ground, dot=s.dot,
            others=list(s.others), players=list(s.players), ropes=list(s.ropes),
            exp=np.packbits(exp).tobytes() if s.exp is not None else b"",
            exp_shape=exp.shape if s.exp is not None else (), where=where, how=s.how, monsters=seen,
            band_at=self.band_at, wide_at=self.wide_at, mesos_only=self.mesos_only,
            tracks=snapshot(list(self.tracker.tracks.values())), events=events, work=work, frame=s.frame, geo=s.geo,
            bad=s.bad, hp_of=self.hp_number.of if s.bad is None else None)
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
        if self._refinds is not None:
            self._refinds.shutdown(wait=False)

    def _refind_all(self, s: Scene, todo: list[tuple[Track, tuple[int, int, int, int]]],
                    t: float) -> list[tuple[Mob | None, bool, float | None]]:
        """Each track of `todo` re-found in the frame of look `s` (shown at t) where its picture should be, with its HP
        bar if that's due (`_bar_due`): (the monster found or None, whether its bar was read, the bar), in `todo`'s
        order. Their matches all in one call in C on REFIND_THREADS threads (ONE_CALL: Eyes.refind_all; the fallbacks on
        the pool), else more than one at once on the pool; meanwhile nothing else touches the tracker, as the frame loop
        waits for them."""
        many = getattr(self.eyes, "refind_all", None) if self._memos is not None and ONE_CALL and todo else None
        if many is not None:  # (all in one call in C)
            asks = [(tr.mob, box, tr.state != HIDDEN, self._near(s, tr, t), self._memos[tr.id], tr.id) for tr, box in todo]
            found = many(s, asks, self._refind_threads, self._refinds, self._ahead)
            if found is not None:
                return [self._bar_of(s, tr, m, t) for (tr, _), m in zip(todo, found)]
        if self._refinds is None or len(todo) < 2:
            return [self._refind(s, tr, box, t) for tr, box in todo]
        return list(self._refinds.map(lambda job: self._refind(s, job[0], job[1], t), todo))

    def _send_ahead(self, look, t: float) -> int | None:
        """The frame's re-finds sent to the GPU ahead of its look (AHEAD: native.AHEAD; Eyes.refind_ahead), from the
        frame before's camera: its token (the step's seq), or None if they weren't (off, no GPU in use, no camera yet,
        nothing followed, the look without its picture: the simulator's, a saved look's)."""
        if not native.AHEAD or self._cam is None or self._memos is None or not ONE_CALL:
            return None
        full, many = getattr(look, "full", None), getattr(self.eyes, "refind_ahead", None)
        if full is None or many is None:
            return None
        tag_x, feet_y, where_x, where_y = self._cam
        geo = self.eyes.geo
        upp, h, w = geo.units_per_px, full.shape[0], full.shape[1]
        asks = []
        for tr in self.tracker.live():
            if tr.state not in (LIVE, HIDDEN, DYING):
                continue
            m = tr.mob  # (_box's, the camera the frame before's)
            cx = tag_x + (self.tracker.predicted(tr, t) - where_x) / upp + (m.x - m.foot_x)
            cy = feet_y + (tr.y - where_y) / upp + (m.y - m.feet_y)
            box = (round(cx - m.w / 2), round(cy - m.h / 2), round(cx + m.w / 2), round(cy + m.h / 2))
            if box[2] <= 0 or box[0] >= w or box[3] <= 0 or box[1] >= int(h * geo.hud_top):
                continue
            asks.append((box, m.kind, m.template, m.facing, tr.id))
        if not asks or not many(full, asks, self._refind_threads, self.seq, AHEAD_MORE):
            return None
        self.counts["sent ahead"] = self.counts.get("sent ahead", 0) + 1
        return self.seq

    def _refind(self, s: Scene, tr: Track, box: tuple[int, int, int, int],
                t: float) -> tuple[Mob | None, bool, float | None]:
        if self._memos is None:  # (the simulator's fakes: REFIND_PAD around, as always)
            m = self.eyes.refind(s, tr.mob, box, fallback=tr.state != HIDDEN)
        else:
            m = self.eyes.refind(s, tr.mob, box, fallback=tr.state != HIDDEN, near=self._near(s, tr, t),
                                 memo=self._memos[tr.id])
        return self._bar_of(s, tr, m, t)

    def _bar_of(self, s: Scene, tr: Track, m: Mob | None, t: float) -> tuple[Mob | None, bool, float | None]:
        """Track `tr` re-found as `m` (or not: None), with its HP bar if that's due (`_bar_due`)."""
        if m is None:
            return None, False, None
        read = self._bar_due(tr, t)
        return m, read, self.eyes.bar(s, m) if read else None

    def _near(self, s: Scene, tr: Track, t: float) -> int | None:
        """How far (look-frame px) around its box track `tr` is looked for first (mobs.near_pad): its kind's fastest
        walk as the tracker has it (SPEED_CLAMP x its speed) over the time since it was last seen; None: not first
        nearer (NEAR_FIRST off)."""
        if not NEAR_FIRST:
            return None
        speed = SPEED_CLAMP * self.tracker.walking.get(tr.kind, DEFAULT_SPEED) / s.geo.units_per_px
        return near_pad(speed, t - tr.last_seen)

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
        if job.band:  # (a search for the drops alone can't see a monster: hidden tracks don't wait for it)
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
            fresh = [m if tid is None else replace(m, track=tid) for m, tid in zip(fresh, self.tracker.merged)]
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
        """Her EXP number changed since the last frame (`exp_gain`, with the number as read: the kills its size was go
        to the tracker, which puts each on a monster), or was read unchanged (`exp_read`: to start from, or to confirm a
        jump: ExpMeter.wants_reading); her HP dropped (a hit); her HP number changed and read (`hp_read`: her HP and
        max HP, and the HP it said at the change before if that read too, `before`: a drop between the two is a touch,
        exactly; None: not known)."""
        out = []
        changed = exp_changed(self._exp, s.exp)
        if changed or self.meter.wants_reading(t):
            reading = exp_number.read(s.exp_box) if s.exp_box is not None else None
            if changed or (reading is not None and reading.exp is not None):
                out.append(Event("exp_gain" if changed else "exp_read", t, data=reading_data(reading)))
            self.counts["exp gains"] += changed
            self.counts["exp read"] += reading is not None and reading.exp is not None
            self._told(self.meter.update(t, reading, changed))
        self._told(self.meter.due(t))  # changes it couldn't read that no reading has explained: a kill each
        if s.exp is not None:
            self._exp = s.exp
        if self._hp is not None and s.hp < self._hp - HP_DROP:
            new = t - self._hit_at > HIT_MERGE_S
            out.append(Event("hp_drop", t, data={"lost": round(self._hp - s.hp, 4), "hp": round(s.hp, 3), "new": new}))
            self._hit_at = t
            self.counts["hits"] += new
        self._hp = s.hp
        if self._hp_number_t is not None and t - self._hp_number_t > HP_GAP_S:
            self.hp_number.see(None)  # (a stretch with no frame: what it did meanwhile isn't known)
        self._hp_number_t = t
        seen = self.hp_number.see(getattr(s, "hp_box", None), s.geo)
        if seen is not None and seen[0].cur is not None:
            out.append(Event("hp_read", t, data={"cur": seen[0].cur, "max": seen[0].max, "before": seen[1]}))
        return out

    def _told(self, gains: list[Gain]) -> None:
        """Her kills these gains were, told to the tracker by kind, each at its time (and the gains kept: `gains`)."""
        for g in gains:
            self.gains.append(g)
            for i, kind in enumerate(g.kinds):
                self.tracker.gain(g.when(i), kind)
            if g.odd and g.exp is not None and g.exp > 0:  # a share of a kill: another player's, a monster she hurt
                self.tracker.share(g.t)

    def _deaths(self, s: Scene, t: float, refound: dict[int, Monster], work: dict) -> None:
        """The tracks that may be dying looked at for it in the frame of look `s` (shown at t; perception.deaths:
        re-found under REFOUND_UNDER, `refound`, or lost RECENT_S or less ago, each every EVERY-th frame, its die1
        frames where its feet are, or were when last seen), on the re-finds' pool; those seen dying told to the tracker.
        `work` gets how many were looked at and how long it took (real clock)."""
        watch = self._watch(s)
        if watch is None:
            return
        began = time.perf_counter()
        asks = []
        for tr in self.tracker.live():
            if tr.state not in (LIVE, HIDDEN, DYING) or tr.died_seen is not None or tr.kind not in watch.pictures \
                    or (self.seq + tr.id) % deaths.EVERY:
                continue
            m = refound.get(tr.id)
            score = None if m is None else m.mob.score
            if not deaths.wants(score, t - tr.last_seen):
                continue
            if m is not None:
                fx, fy = m.mob.foot_x, m.mob.feet_y
            else:  # where its feet were, the camera's moves since taken out
                fx, fy = s.screen_x(tr.seen_x if tr.seen_x is not None else tr.x), s.screen_y(tr.y)
            asks.append((tr.id, tr.kind, fx, fy, score))
        if not asks:
            return

        def look(a):
            return watch.dying(s, a[1], a[2], a[3], a[4])
        found = list(self._refinds.map(look, asks)) if self._refinds is not None and len(asks) > 1 else \
            [look(a) for a in asks]
        for a, dead in zip(asks, found):
            if dead:
                self.tracker.seen_dying(a[0], t)
        work["deaths"], work["deaths_s"] = len(asks), time.perf_counter() - began
        self.counts["death looks"] = self.counts.get("death looks", 0) + len(asks)

    def _watch(self, s: Scene):
        """The death watch for look `s`'s frames: with real eyes, the map's monsters' die1 frames drawn at its frame's
        scale (made once, perception.deaths.DeathWatch.of); else the eyes' own (`deaths`), if they have one."""
        if not self._real:
            return getattr(self.eyes, "deaths", None)
        scale = s.geo.frame_scale
        if self._watch_of is None or self._watch_of[0] != scale:
            self._watch_of = (scale, self._death_watch(scale))
        return self._watch_of[1]

    def _death_watch(self, scale: float):
        """The map's monsters' die1 frames drawn at `scale` (perception.deaths.DeathWatch.of); None without the game
        data. About 0.5 s, once."""
        names = {sp.mob_id: sp.name for sp in getattr(self.model, "spawns", ()) if sp.name}
        try:
            return deaths.DeathWatch.of(names, scale)
        except (ImportError, FileNotFoundError, OSError):  # (no game data here)
            return None

    def _bar_due(self, tr: Track, t: float) -> bool:
        """Whether track `tr`'s HP bar is read in the frame shown at t (BAR_EVERY, BAR_SHOWN_S, BAR_UNSEEN_S)."""
        if tr.bar_at is not None and t - tr.bar_at <= BAR_SHOWN_S:
            return True
        if tr.state == HIDDEN or t - tr.last_seen >= BAR_UNSEEN_S:  # (seen again)
            return True
        if any(0 <= t - th.t <= self.tracker.shape.window_s and tr.id in th.aimed for th in self.tracker.throws):
            return True
        return (self.seq + tr.id) % BAR_EVERY == 0

    def _box(self, s: Scene, tr: Track, t: float) -> tuple[int, int, int, int] | None:
        """Where track `tr`'s picture should be in this frame (half-size px), from where it should be on the
        map; None if that's off the play area."""
        m = tr.mob
        cx = s.screen_x(self.tracker.predicted(tr, t)) + (m.x - m.foot_x)
        cy = s.screen_y(tr.y) + (m.y - m.feet_y)
        h, w = s.frame.shape[:2]
        box = (round(cx - m.w / 2), round(cy - m.h / 2), round(cx + m.w / 2), round(cy + m.h / 2))
        if box[2] <= 0 or box[0] >= w or box[3] <= 0 or box[1] >= int(h * s.geo.hud_top):
            return None
        return box

    @staticmethod
    def _view(s: Scene) -> View:
        h, w = s.frame.shape[:2]
        bottom = int(h * s.geo.hud_top)
        box = minimap_box(s.geo)
        mx, my = box[2] / s.geo.look_scale, box[3] / s.geo.look_scale
        return View(s.map_x(0), s.map_x(w), s.map_y(0), s.map_y(bottom), w, bottom,
                    (s.map_x(0), s.map_y(0), s.map_x(mx), s.map_y(my)), s.geo.fpx(EDGE_PX))


def to_scene(st: WorldState, model: MapModel, band: tuple[float, float], mobs: bool = True, wide: bool = False,
             loot: bool = False, mesos_only: bool = False, frame: np.ndarray | None = None) -> Scene:
    """The Scene a look (Senses.look with these arguments) would have given in the frame of state `st`: her
    place; with `mobs`, the monsters seen in her band (or, if `wide`, all of them); with `loot` (and the
    state searched for drops), the drops on her platform (or, if `wide`, every one in view)."""
    p = None if st.where is None or st.where[2] is None else model.platform(st.where[2])
    where = None if st.where is None else Where(st.where[0], st.where[1], p, st.where[3])
    s = Scene(t=st.t, frame=st.frame if frame is None else frame, tag=st.tag, hp=st.hp, mp=st.mp, ground=st.ground,
              dot=st.dot, others=list(st.others), ropes=list(st.ropes), exp=st.exp_pixels(), where=where, how=st.how,
              tag_seen=st.tag_seen, players=list(st.players), geo=st.geo, tracks=list(st.tracks),
              hp_of=getattr(st, "hp_of", None))
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


def reading_data(reading: ExpReading | None) -> dict:
    """An EXP reading as an event's data (plain, for the pipe): its EXP, percent and the levels it fits; {} unread."""
    if reading is None or reading.exp is None:
        return {}
    return {"exp": reading.exp, "percent": reading.percent, "levels": list(reading.levels)}


def reading_of(data: dict) -> ExpReading | None:
    """The EXP reading an event's data holds (reading_data), or None."""
    return ExpReading(data["exp"], data.get("percent"), tuple(data.get("levels", ()))) if "exp" in data else None
