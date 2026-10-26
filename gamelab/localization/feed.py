"""The continuous feed: the game looked at every frame and the latest of what it saw handed to the stack as it
asks (Senses: every look and glance). Live, it runs in the perception service, a process of its own that runs at all
times (localization.service: `run` there, its states down a pipe to each command, its frames in a `Ring` in shared
memory); in the simulator `SyncFeed`, in-process on the fake clock; on saved frames and in tests `InlineFeed`, the step
run on the eyes' own frame when a look asks.

Why a process: the stack's planning holds Python's lock, and a perception thread in the same process got
16 of its 60 looks in 2 s while the planner worked (the research's gil.py; Python 3.15 here has the lock on).
Each frame it runs the world step (localization.worldstate): her place, the monsters tracked, her EXP gains and hits;
the searches (her band, the whole screen, the drops) run on a thread beside that frame loop
(worldstate.SearchThread), and each frame's state searched is given again once they're back (`late`). The look frames
(half size at 1920x1080, full size at the game's own 1366x768: screen.Geometry) go into a ring in shared memory
(`Ring`: the oldest overwritten, never more).

What the stack asks of a feed (`Feed`):
- `latest(want)`: the newest state of a frame shown at `want.after` or later that has what's wanted fresh (`fresh`):
  her place and the monsters followed are that frame's (re-found every frame), monsters new in her band of a search
  within MOBS_FRESH_S of it, the whole screen within WIDE_FRESH_S, drops searched on that very frame (a pile just
  picked up must not be seen again), and nothing that failed its checks (WorldState.bad). If none has, it asks the next
  frame for what the newest state wouldn't have fresh by then (to_ask: so a look reads the feed's world state and waits
  for a frame, not for a search, ASK_ONLY_STALE), and waits, FEED_WAIT_S at most; then (or at once while the newest
  state isn't of the game as expected) None, and Senses asks again, says why (`problem`) and stands still after a while.
- `wait_frame(timeout)`: until a new frame's state comes; `drain_events(upto)`: every event since the last
  drain of a frame up to the one a look shows, each once; `throw`: her throw, for the tracker.

`SyncFeed` is the same thing in-process for the simulator: its clock calls `tick` at every frame the game
shows, the world step runs on the fake eyes there, and each state is ready its measured cost (COSTS) after
its frame; a search handed over at the end of a frame's step comes back its own cost later (`TimedSearches`),
merged by the first frame placing her after that, or as it comes if the frame loop is idle, as in the service.
Nothing it does is charged to the stack's time.
"""

import ctypes
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from ..perception.eyes import Eyes, Observation, band_rows
from ..perception.framecheck import BadLook, Problem
from ..perception.frames import Frame
from ..screen import STRETCHED, Geometry
from .senses import Senses
from .tracking import Event
from .worldstate import Asks, Found, Job, Schedule, SearchThread, Searches, WorldState, WorldStep, search, to_scene

HZ = 30.0            # frames a second the feed looks at (the game shows 60; a frame's step takes 15-20 ms)
RING_SLOTS = 4       # look frames a Ring keeps by default (the service's: service.RING_SLOTS)
FRAME_SHAPE = (540, 960, 3)  # ... each of this shape at 1920x1080 (frame_shape)
MOBS_FRESH_S = 0.3   # a look's monsters: her band searched at most this long before its frame. It's searched every
                     # 0.2 s (worldstate.BAND_EVERY_S), and the frame after that and the search itself take about 0.1 s
                     # more: with 0.25, on the saved frames a third of the frames searched it, every look asking.
                     # (Without ASK_ONLY_STALE a look asks for it on a frame of its own.)
WIDE_FRESH_S = 0.5   # ... the whole screen (it's searched every second; tracked monsters are re-found every frame)
FEED_WAIT_S = 0.6    # s a look waits for the feed's state before it asks again (the whole screen with every drop in
                     # view takes ~0.25 s; Senses.look stands still after senses.LOST_S of none)
GLANCE_WAIT_S = 0.25  # s a glance waits for a frame from the feed (none: no dot, Senses.glance)
NOT_VISIBLE_S = 0.25  # s between checks while the game isn't wholly on screen
RECOVER_EVERY_S = 0.2  # s at least between two of the eyes' recoveries while frames fail their checks (Eyes.recover:
                       # the minimap's picture and her name tag searched afresh, the capture duplicated afresh or mss),
                       # as a look taken again (senses.RETRY_GAP_S)
KEEP = 8             # states the stack's side keeps
EVENTS_KEPT = 10_000  # events kept for the next drain at most (the oldest go: a stack that never drains)
BELOW_NORMAL = 0x4000  # Windows' BELOW_NORMAL_PRIORITY_CLASS (quiet: the harness's and soak's workers)
# True: a look reads the feed's world state, and asks the next frame only for what the newest state wouldn't have fresh
# by then (to_ask): her place and the monsters it follows are of the look's own frame (re-found in every frame), the
# monsters new in her band of a search within MOBS_FRESH_S, the whole screen within WIDE_FRESH_S; the drops it asks for
# are searched on that frame. False: it asks for everything it wants, as a blocking look searches it all on its own
# frame. On 2026-10-21 (preset game on the first map, the searches still in the frame loop, 20 seeds x 20 min, --costs
# measured) asking for everything made the hunt loop 7.1% +- 1.8% better and the tour 9.0% +- 2.7%; asking only for
# what's stale, the feed's own loop on the saved frames had searched her band 4.9 times a second instead of 7.0 and
# looked at 21.7 frames a second instead of 18.3. With the searches on a thread of their own and the feed the default
# (2026-10-24; Tree Dungeon's crowd, zero-walls, the regular attack, 8 seeds x 10 min, --costs measured, her band looks
# with what the feed follows: TRACKED), reading made the hunt loop's EXP a minute 67.3 against asking's 68.4 and 73.1
# without a feed (-3.3% seed by seed against that, standard error 9.0%; asking +1.9%, 18.0%): level within the spread,
# and a look waits for its frame, not for a search of its own: 43 ms (median) against 73 for the hunt loop's look in the
# simulator (README, Continuous perception: The feed by default).
ASK_ONLY_STALE = True


# --- frames in shared memory --------------------------------------------------------------------------

def frame_shape(geo: Geometry = STRETCHED) -> tuple[int, int, int]:
    """The shape of a look frame of geometry `geo` (RGB): (540, 960, 3) at 1920x1080, (768, 1366, 3) at 1366x768."""
    w, h = geo.frame_size
    return h, w, 3


class Ring:
    """Look frames in shared memory, `slots` of them, each with its frame number and time: the perception service
    writes them in turn (the oldest overwritten), a command copies one out by its number. Made by the service (`name`
    None) and opened by name in each command."""

    def __init__(self, name: str | None = None, slots: int = RING_SLOTS, shape: tuple = FRAME_SHAPE):
        self.slots, self.shape = slots, shape
        size = slots * (16 + int(np.prod(shape)))
        self.shm = SharedMemory(name=name, create=name is None, size=size if name is None else 0)
        self.name = self.shm.name
        self._heads = np.ndarray((slots, 2), np.float64, self.shm.buf)  # frame number (-1: being written), time
        self._frames = np.ndarray((slots, *shape), np.uint8, self.shm.buf, offset=slots * 16)
        if name is None:
            self._heads[:] = -1

    def put(self, seq: int, t: float, frame: np.ndarray) -> int | None:
        """Write frame number `seq` into its slot (seq mod slots); the slot, or None if it isn't the ring's shape."""
        if frame.shape != self.shape:
            return None
        slot = seq % self.slots
        self._heads[slot, 0] = -1  # a reader copying it now sees it change and gives it up
        self._frames[slot] = frame
        self._heads[slot, 1] = t
        self._heads[slot, 0] = seq
        return slot

    def get(self, slot: int, seq: int) -> np.ndarray | None:
        """A copy of frame number `seq`, or None if it has been overwritten (or is being)."""
        if not 0 <= slot < self.slots or self._heads[slot, 0] != seq:
            return None
        out = self._frames[slot].copy()
        return out if self._heads[slot, 0] == seq else None

    def held(self) -> list[int]:
        """The frame numbers the ring holds now."""
        return sorted(int(s) for s in self._heads[:, 0] if s >= 0)

    def slot_of(self, seq: int) -> int | None:
        """The slot holding frame number `seq`, or None if it doesn't (never put, or overwritten since)."""
        slot = seq % self.slots
        return slot if self._heads[slot, 0] == seq else None

    def close(self, unlink: bool = False) -> None:
        del self._heads, self._frames  # the views first: shared memory with views on it can't close
        self.shm.close()
        if unlink:
            self.shm.unlink()


# --- what the child runs ---------------------------------------------------------------------------------

@dataclass
class Runner:
    """What the feed looks with: its frames (perception.frames: the screen, or a list), the eyes that read
    them (Eyes.look_at), and the world step."""
    frames: object                 # .next(after) -> Frame | None
    eyes: Eyes
    step: WorldStep
    title: str = ""
    recovered: float = -1e18       # when the eyes last recovered (`recover`)

    def look(self, frame: Frame) -> Callable[[bool], Observation]:
        return lambda ropes: self.eyes.look_at(frame.full, frame.t, ropes)

    def recover(self, problem: Problem) -> None:
        """A frame that failed its checks (or the game not wholly on screen): the eyes do what they can before the next
        (Eyes.recover), RECOVER_EVERY_S apart at most."""
        now = time.monotonic()
        if now - self.recovered >= RECOVER_EVERY_S and hasattr(self.eyes, "recover"):
            self.recovered = now
            self.eyes.recover(problem)


def saved_world(files: list[str], map_id: int, template: np.ndarray, band: tuple[float, float], hold: int = 30,
                schedule: Schedule = Schedule()) -> Runner:
    """Saved screenshots (full-size game frames, all of one size: its geometry's) for a screen, each `hold` frames in
    a row, round and round: the perception service run offline (`python -m gamelab perceive --frames`,
    `python -m gamelab.sim.soak feed`)."""
    from PIL import Image

    from ..perception.frames import ListFrames
    from ..perception.sprites import detector_for
    from ..screen import of_frame
    from . import mapdata
    model = mapdata.load(map_id)
    frames = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    geo = of_frame(frames[0])
    eyes = Eyes("saved frames", template, detector_for(model, geo), screen=object(), geo=geo,  # handed its frames
                minimap_picture=mapdata.minimap_picture(model.id))
    senses = Senses(eyes, model, band)
    return Runner(ListFrames([f for f in frames for _ in range(hold)]), eyes,
                  WorldStep(senses, schedule=schedule, searches=SearchThread(senses)))


def run(runner: Runner, hz: float | Callable[[], float], poll: Callable[[float], tuple[Asks | None, bool]],
        emit: Callable[[WorldState, Frame | None], None], until: float | None = None,
        shown: Callable[[Frame], None] | None = None) -> None:
    """Frame after frame, `hz` a second (or as often as `hz()` says, asked at each frame: the perception service's
    rate follows who's attached): `poll(wait)` waits up to `wait` s for what's asked (asks, stop), and
    what's asked is done on the first frame shown at the time asked for, taken at once then (its searches on
    it, or as soon as the one out is back). Each frame's state goes to `emit`, and so do the late states of the
    frames searched, as their searches come back (with no frame: theirs is in the past). The poll should end
    early when the searches' answer is in (the step's `searches.wake`). `shown(frame)`, if given, sees each
    frame showing the game as soon as it's taken, before the step. Ends on stop, at `until`
    (time.monotonic), or when the frames run out."""
    due, pending = time.monotonic(), None
    while True:
        period = 1.0 / (hz() if callable(hz) else hz)
        for st in runner.step.came_back():  # searches back while it waited
            emit(st, None)
        now = time.monotonic()
        wake = due if pending is None else min(due, max(now, pending.after))
        asks, stop = poll(max(0.0, wake - now))
        if stop or (until is not None and time.monotonic() >= until):
            return
        if asks is not None:
            pending = asks if pending is None else pending.merge(asks)
        now = time.monotonic()
        asked = pending is not None and now >= pending.after
        if not asked and now < due:
            continue
        frame = runner.frames.next(pending.after if asked else due)
        if frame is None:
            return
        if not frame.visible:
            st = runner.step.step(None, frame.t, False, why=frame.why, bad=frame.problem)
            emit(st, frame)
            runner.recover(st.bad)
            due, pending = time.monotonic() + NOT_VISIBLE_S, None  # whoever asked takes the look itself
            continue
        use = pending if asked and frame.t >= pending.after else None
        if shown is not None:
            shown(frame)
        if frame.new or use is not None:  # an unchanged screen nobody asked about: nothing new to say
            st = runner.step.step(runner.look(frame), frame.t, True, use)
            late = runner.step.came_back()  # merged in the step, or back since (this frame's own, if that quick)
            for done in late:
                if done.seq != st.seq:
                    emit(done, None)
            emit(st, frame)
            for done in late:
                if done.seq == st.seq:
                    emit(done, None)
            if st.bad is not None:
                runner.recover(st.bad)
        if use is not None:
            pending = None
        if time.monotonic() >= due:
            due = max(due + period, time.monotonic())


def quiet() -> None:
    """This process at below-normal priority (the game and the perception service keep their pace: the harness's,
    soak's and the offline measurements' workers), and OpenCV on one thread: the monster search's own threads do the
    work, and OpenCV's 24 on top only cost (a 1920x1080 conversion took 25 ms of CPU across them)."""
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        k.SetPriorityClass(k.GetCurrentProcess(), BELOW_NORMAL)
    import cv2
    cv2.setNumThreads(1)


# --- the stack's side ------------------------------------------------------------------------------------

def wanted(after: float, mobs: bool, wide: bool, loot: bool, mesos_only: bool) -> Asks:
    """What a look (Senses.look's arguments) wants of a frame."""
    return Asks(after, band=mobs and not wide, wide=mobs and wide, loot=loot and not wide, loot_wide=loot and wide,
                mesos_only=mesos_only)


def to_ask(want: Asks, newest: WorldState | None, frame_s: float = 1.0 / HZ) -> Asks:
    """What to ask of the next frame for a look that wants `want`: all of it, or (ASK_ONLY_STALE) only what the
    next frame wouldn't have fresh anyway, judging by the newest state. The next frame: a frame after `want.after`
    or after the newest's, whichever is later (a look's `after` is often before the newest frame: Senses.fresh_after,
    up to MAX_AGE_S before now; judged by `after` alone, a search the next frame would have stale went unasked, and the
    look waited for the feed's own schedule, or FEED_WAIT_S)."""
    if not ASK_ONLY_STALE:
        return want
    soon = max(want.after, newest.t if newest is not None else -1e18) + frame_s
    stale = lambda at, fresh_s: newest is None or at is None or soon - at > fresh_s
    return Asks(want.after, band=want.band and stale(newest and newest.band_at, MOBS_FRESH_S),
                wide=want.wide and stale(newest and newest.wide_at, WIDE_FRESH_S), loot=want.loot,
                loot_wide=want.loot_wide, mesos_only=want.mesos_only)


def adds(ask: Asks, asked: Asks) -> bool:
    """Whether `ask` wants a search that `asked` (what a look has asked for so far) didn't."""
    return ((ask.band and not asked.band) or (ask.wide and not asked.wide) or (ask.loot and not asked.loot)
            or (ask.loot_wide and not asked.loot_wide))


def fresh(st: WorldState, want: Asks) -> bool:
    """Whether state `st` is good for a look that wants `want`: shown at `want.after` or later, and fresh as
    the look needs (see the top): the monsters it wants searched within MOBS_FRESH_S (WIDE_FRESH_S) of the state's
    frame (with ASK_ONLY_STALE, the default, any search that recent will do: it asked only for what was stale;
    without, the search it asked for, on a frame shown at `want.after` or later, as a blocking look searches its own
    frame). One where she isn't placed is as good as a look would be; one that failed its checks (`bad`) is never
    fresh."""
    if not st.usable or st.t < want.after:
        return False
    if not st.placed:
        return True
    if want.band and not _searched(st.band_at, st.t, MOBS_FRESH_S, want.after):
        return False
    if want.wide and not _searched(st.wide_at, st.t, WIDE_FRESH_S, want.after):
        return False
    if want.loot or want.loot_wide:
        if st.loot is None or (want.loot_wide and not st.loot_wide) or (st.mesos_only and not want.mesos_only):
            return False
    return True


def _searched(at: float | None, t: float, within: float, after: float) -> bool:
    return at is not None and t - at <= within and (ASK_ONLY_STALE or at >= after - 1e-9)


def newest_fresh(states, want: Asks) -> WorldState | None:
    """Of `states` (in the order they came), the one of the newest frame that's fresh for `want`; of two of the
    same frame, the later (a late state has its searches)."""
    best = None
    for st in states:
        if fresh(st, want) and (best is None or st.t >= best.t):
            best = st
    return best


def newest_frame(states) -> WorldState | None:
    """The state of the newest frame given (late states left out: theirs is in the past)."""
    return next((st for st in reversed(states) if not st.late), None)


class Feed:
    """What the stack asks of a feed (service.ServiceClient, SyncFeed, InlineFeed)."""
    alive = True
    why = ""  # why it's down, if it is

    def latest(self, want: Asks, timeout: float = FEED_WAIT_S) -> WorldState | None:
        raise NotImplementedError

    def problem(self) -> Problem | None:
        """Why there's no state to act on now (the newest one's, or perception down); None: the newest is usable."""
        newest = newest_frame(list(self._states()))
        if newest is None:
            return Problem("perception", self.why or "no state from perception yet")
        return newest.bad

    def _states(self):
        return ()

    def bring_forward(self, problem: Problem | None) -> None:
        """What a command that sends keys does while there's no state to act on for `problem` (the service's client
        brings the game to the front when it isn't); by default nothing."""

    def full_frame(self, after: float) -> np.ndarray:
        """The game window at full size (RGB), shown at `after` or later; raises BadLook if there's none."""
        raise BadLook(Problem("perception", "this feed has no full frames"))

    def frame_of(self, st: WorldState) -> np.ndarray:
        """The state's half-size frame."""
        raise NotImplementedError

    def wait_frame(self, timeout: float) -> bool:
        """Wait until a new frame's state comes (True), `timeout` s at most."""
        raise NotImplementedError

    def drain_events(self, upto: float | None = None) -> list[Event]:
        """Every event not drained yet of a frame shown at `upto` or before (None: all), each once; the later
        ones wait for the next drain."""
        raise NotImplementedError

    def throw(self, t: float, side: str) -> None:
        """She threw toward `side` at t (Actions.throw_listeners)."""

    def close(self) -> None:
        pass

    def look(self, senses: Senses, mobs: bool, wide: bool, loot: bool, mesos_only: bool, after: float | None = None):
        """The Scene of a look with these arguments from the feed, or None (Senses then asks again). A look at her
        band also has the monsters the feed follows off it, re-found in that frame (`Scene.swept`: the fights go by the
        band's, her memory takes them all), and says where it searched (`Scene.rows`): her band, and
        the whole screen while the feed's last search of it is within WIDE_FRESH_S of the frame (everything on screen
        then was found, and followed since). `after`: of a frame shown then or later (None: now)."""
        st = self.latest(wanted(time.monotonic() if after is None else after, mobs, wide, loot, mesos_only))
        if st is None:
            return None
        s = to_scene(st, senses.model, senses.band, mobs, wide, loot, mesos_only, self.frame_of(st))
        if s.searched and not s.wide:
            s.swept = [m for m in st.monsters if not any(m is k for k in s.monsters)]
            s.rows = [senses._map_rows(s, band_rows(s, *senses.band))]
            if st.wide_at is not None and st.t - st.wide_at <= WIDE_FRESH_S:
                s.rows.append(senses._play_rows(s))
        return s

    def glance(self, after: float, timeout: float = GLANCE_WAIT_S) -> WorldState | None:
        """The newest visible state of a frame shown at `after` or later (waited for), or None."""
        return self.latest(Asks(after), timeout)


def _drain(events: deque[Event], upto: float | None) -> list[Event]:
    """Take the events of frames shown at `upto` or before (all if None) off the front of `events` (in the order
    their states came: a frame's after the one before's, but a late state's, of its own older frame, after the
    frames since, so they wait for the next drain past those)."""
    out = []
    while events and (upto is None or events[0].t <= upto + 1e-9):
        out.append(events.popleft())
    return out


# --- in-process, for the simulator ---------------------------------------------------------------------

@dataclass(frozen=True)
class Costs:
    """What each part of a frame takes the feed (s), as measured offline. The frame loop's (bench-look medians on
    the 17 saved frames after the profiling pass; the re-finds on them, refind_bench in the step-5 report):
    capture, look, refind, fallback, and `beside`, what a frame's step takes more while a search is out. The
    searches', on their thread beside the frame loop (worldstate.SearchThread; on the saved frames, 30 frames a
    second and her band asked for every 0.1 s: README, Continuous perception), slower than inline by Python's
    lock and the frame loop's work: band, wide, loot_strip. Before the profiling pass: her place 0.016, her
    band 0.032, the whole screen 0.122."""
    capture: float = 0.002     # the frame copied off the graphics card and converted (bench-capture: 2.0 ms)
    look: float = 0.012        # her place, ropes included (Senses.look without monsters: 13.3 ms, 1.3 of it the
                               # saved frame's conversion, which `capture` counts)
    refind: float = 0.0007     # a monster re-found with the template that matched it last (0.69 ms)
    fallback: float = 0.0034   # ... and with its kind's whole set, when that missed (3.4 ms)
    beside: float = 0.002      # (a step with a search out took 19.1 ms on average, with none 16.8)
    band: float = 0.026        # her band searched (on the thread 26.5 ms, median; inline in the frame loop it had
                               # taken 22: the feed step with it 34.2, without 12.9)
    wide: float = 0.100        # the whole screen (on the thread 100.4; inline 86, 98.6 - 12.9)
    loot_strip: float = 0.008  # a strip searched for drops (inline 7: her platform's 15.6 ms in 1.9 strips, every
                               # platform in view 71 ms in 28.6; on the thread, taken as much slower as her band)

    def of(self, st: WorldState) -> float:
        """What a state took: a frame's, the frame loop's part (her place and the re-finds); a late state's, its
        searches'."""
        w = st.work
        if not st.visible:
            return self.capture
        if st.late:
            return self.search(w)
        return self.capture + self.look + self.refind * w.get("re-finds", 0) + self.fallback * w.get("fallbacks", 0)

    def search(self, work: dict) -> float:
        """What a frame's searches take on their thread (`work`: Found.work)."""
        return (self.band * work.get("band", 0) + self.wide * work.get("wide", 0)
                + self.loot_strip * work.get("loot strips", 0))


COSTS = Costs()
GAME_HZ = 60.0  # frames a second the game shows


class TimedSearches(Searches):
    """The searches in-process for the simulator: a job is searched at once, on its own frame (the fake frames
    keep what they showed), but its answer is in only its cost (Costs.search) after it began (`start`: once the
    frame's step that handed it over had placed her, or as the one before came back)."""

    def __init__(self, senses: Senses, costs: Costs = COSTS):
        self.senses, self.costs = senses, costs
        self._found: Found | None = None
        self.began: float | None = None  # when the job out began (None: not yet, or none out)

    @property
    def ready_at(self) -> float | None:
        """When the answer out is in (None: none out, or not begun)."""
        if self._found is None or self.began is None:
            return None
        return self.began + self.costs.search(self._found.work)

    def free(self) -> bool:
        return self._found is None

    def submit(self, job: Job) -> None:
        self._found, self.began = search(self.senses, job), None

    def start(self, at: float) -> None:
        """A job handed over and not begun begins at `at`."""
        if self._found is not None and self.began is None:
            self.began = at

    def take(self, by: float | None = None) -> Found | None:
        ready = self.ready_at
        if ready is None or (by is not None and ready > by + 1e-9):
            return None
        out, self._found, self.began = self._found, None, None
        return out


class SyncFeed(Feed):
    """The feed in-process, stepped by whoever keeps time: `tick(t)` at every frame the game shows (the
    simulator's clock calls it, GAME_HZ), `look(ropes)` reading the screen then (FakeEyes.observe). A frame is
    taken `hz` times a second, or at once when something was asked; its state is ready its cost later
    (`costs`), the next frame not before. Its searches run as on the child's thread (TimedSearches): handed
    over once a frame's step has placed her (`capture` and `look` into it), back their cost later; merged by
    the first frame's step placing her after that, else as soon as the frame loop is done with the frame it's
    on, or at once if it's idle; each such frame's late state is ready then. Waiting goes through time.sleep,
    the simulator's own."""

    def __init__(self, step: WorldStep, look: Callable[[bool], Observation], costs: Costs = COSTS, hz: float = HZ,
                 guard: Callable[[], None] | None = None):
        """`guard`: called at every ask; it raises when the fakes aren't in place (sim.scenarios.Sim.guard)."""
        self.step, self._look, self.costs, self.hz, self.guard = step, look, costs, hz, guard
        self.searches = step.searches = TimedSearches(step.senses, costs)
        self.states: deque[tuple[float, WorldState]] = deque(maxlen=KEEP)  # (ready at, state)
        self.events: deque[Event] = deque()
        self.asks: Asks | None = None
        self.busy_until = -1e18
        self.next_due = -1e18
        self._seen_seq = 0
        self.counts = {"frames": 0, "asked": 0, "busy": 0.0, "searches": 0, "searching": 0.0}

    def tick(self, t: float) -> None:
        """The game shows a frame at t."""
        self._settle(t)
        if t < self.busy_until - 1e-9:
            return
        asked = self.asks is not None and t >= self.asks.after - 1e-9
        if not asked and t < self.next_due - 1e-9:
            return
        asks, self.asks = (self.asks, None) if asked else (None, self.asks)
        placed = t + self.costs.capture + self.costs.look  # (her place is known: merged, the next handed over)
        out = not self.searches.free()  # a search out as the frame begins ...
        st = self.step.step(self._look, t, True, asks, by=placed)
        self.searches.start(placed)  # what it handed over begins beside the re-finds
        cost = self.costs.of(st) + (self.costs.beside if out or not self.searches.free() else 0.0)  # ... or then
        end = t + cost
        late = self.step.came_back(by=end)  # merged in the step, or back by its end
        self.searches.start(end)  # what that handed over begins once the frame is done
        self.busy_until, self.next_due = end, t + 1.0 / self.hz
        self.counts["frames"] += 1
        self.counts["asked"] += asked
        self.counts["busy"] += cost
        self._add(end, late + [st])

    def _settle(self, now: float) -> None:
        """The searches back by `now` while the frame loop wasn't on a frame placing her: merged as they come
        (or as the frame it was on is done), and the next handed over then."""
        while (ready := self.searches.ready_at) is not None:
            at = max(ready, self.busy_until)
            if at > now + 1e-9:
                return
            late = self.step.came_back(by=at)
            self.searches.start(at)
            self._add(at, late)

    def _add(self, ready: float, states: list[WorldState]) -> None:
        for st in states:
            self.states.append((ready, st))
            self.events.extend(st.events)
            if st.late:
                self.counts["searches"] += 1
                self.counts["searching"] += self.costs.of(st)

    def latest(self, want: Asks, timeout: float = FEED_WAIT_S) -> WorldState | None:
        if self.guard is not None:
            self.guard()
        deadline, asked = time.monotonic() + timeout, None
        while True:
            now = time.monotonic()
            self._settle(now)
            ready = [st for at, st in self.states if at <= now + 1e-9]
            st = newest_fresh(ready, want)
            if st is not None:
                self._seen_seq = max(self._seen_seq, st.seq)
                return st
            newest = newest_frame(ready)
            if newest is not None and newest.t >= want.after and not newest.usable:
                return None  # (the game not on screen or not as expected: Senses says why, and asks again)
            if (asked is None or ASK_ONLY_STALE) and not any(fresh(st, want) for _, st in self.states):
                ask = to_ask(want, newest_frame([st for _, st in self.states]), 1.0 / self.hz)
                if asked is None or adds(ask, asked):  # (as FeedClient.latest: once, and a search gone stale since)
                    self.asks = ask if self.asks is None else self.asks.merge(ask)
                    asked = ask if asked is None else asked.merge(ask)
            if now >= deadline:
                return None
            wakes = [ready for ready, st in self.states if ready > now and fresh(st, want)]
            if (back := self.searches.ready_at) is not None:
                wakes.append(max(back, self.busy_until))
            time.sleep(max(1e-4, min(wakes + [now + 1.0 / GAME_HZ, deadline]) - now))

    def frame_of(self, st: WorldState) -> np.ndarray:
        return st.frame

    def _states(self):
        now = time.monotonic()
        return [st for ready, st in self.states if ready <= now + 1e-9]

    def wait_frame(self, timeout: float) -> bool:
        self._settle(time.monotonic())
        seen = max((st.seq for ready, st in self.states if ready <= time.monotonic()), default=0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline - 1e-9:
            now = time.monotonic()
            self._settle(now)
            if any(st.seq > seen and ready <= now for ready, st in self.states):
                return True
            later = [ready for ready, st in self.states if st.seq > seen and ready > now]
            time.sleep(max(1e-4, min(later + [now + 1.0 / GAME_HZ, deadline]) - now))
        return False

    def drain_events(self, upto: float | None = None) -> list[Event]:
        self._settle(time.monotonic())
        return _drain(self.events, upto)

    def throw(self, t: float, side: str) -> None:
        self._settle(time.monotonic())
        self.step.throw(t, side)


# --- on saved frames and in tests -------------------------------------------------------------------------

class InlineFeed(Feed):
    """The world step run when a look or glance asks, on the eyes' own frame then (Eyes.look on their capture: saved
    frames, bench-look's and bodies'; the tests' made-up frames; the simulator's FakeEyes charging a look's time): a
    command's looks come from the perception service instead (localization.service), whose frames it looks at all the
    time, and the simulator's from SyncFeed. What's asked is searched on that frame at once (InlineSearches): each look
    is a frame of its own, taken when it asks (its `after` aside: the eyes' clock may be another's, a test's)."""

    def __init__(self, eyes: Eyes, model, band: tuple[float, float]):
        self.eyes = eyes
        self.step = WorldStep(Senses(eyes, model, band), schedule=Schedule(band_s=None, wide_s=None))
        self.events: deque[Event] = deque(maxlen=EVENTS_KEPT)
        self.newest: WorldState | None = None

    def latest(self, want: Asks, timeout: float = FEED_WAIT_S) -> WorldState | None:
        st = self._frame(want)
        return st if st.usable else None

    def _frame(self, asks: Asks | None) -> WorldState:
        """A frame stepped now (its late state, with its searches, if it was searched)."""
        t = time.monotonic()
        try:
            obs = self.eyes.look()
        except BadLook as e:  # (eyes that check their own frames: the game window, offline)
            st = self.step.step(None, t, False, why=e.why, bad=e.problem)
        else:  # (what's asked, of this very frame: the eyes' clock may not be the asker's)
            st = self.step.step(lambda ropes: obs, obs.t, True, None if asks is None else replace(asks, after=-1e18))
        if st.bad is not None and hasattr(self.eyes, "recover"):  # before the next look, as the service's frame loop
            self.eyes.recover(st.bad)
        late = self.step.came_back()
        for done in [st] + late:
            self.events.extend(done.events)
        self.newest = late[-1] if late else st
        return self.newest

    def _states(self):
        return () if self.newest is None else (self.newest,)

    def frame_of(self, st: WorldState) -> np.ndarray:
        return st.frame

    def wait_frame(self, timeout: float) -> bool:
        return True  # (each look takes a frame of its own)

    def drain_events(self, upto: float | None = None) -> list[Event]:
        return _drain(self.events, upto)

    def throw(self, t: float, side: str) -> None:
        self.step.throw(t, side)
