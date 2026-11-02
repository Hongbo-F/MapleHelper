"""The continuous feed: the game looked at every frame and the latest of what it saw handed to the stack as it
asks (Senses: every look and glance). Live, it runs in the perception service, a process of its own that runs at all
times (localization.service: `run` there, its states down a pipe to each command, its frames in a `Ring` in shared
memory); in the simulator `SyncFeed`, in-process on the fake clock; on saved frames and in tests `InlineFeed`, the step
run on the eyes' own frame when a look asks.

Why a process: the stack's planning holds Python's lock, and a perception thread in the same process got
16 of its 60 looks in 2 s while the planner worked (the research's gil.py; Python 3.15 here has the lock on).
Each frame, HZ a second (localization.rate: 15, everything once a frame since 2026-10-30), it runs one pass of the world
step (localization.worldstate): her look and the whole screen's search for monsters at once, her place, her EXP gains,
HP and touches, the monsters matched to their tracks, their bars and deaths, the drops asked for and a few more by
rotation. The look frames (half size at 1920x1080, full size at the game's own 1366x768: screen.Geometry) go into a
ring in shared memory (`Ring`: the oldest overwritten, never more).

What the stack asks of a feed (`Feed`):
- `latest(want)`: the newest state of a frame shown at `want.after` or later that has what's wanted fresh (`fresh`):
  her place, the HUD and every monster on screen are that frame's (every state has them), the drops searched on that
  very frame (a pile just picked up must not be seen again), and nothing that failed its checks (WorldState.bad). If
  none has, it asks for a frame shown then and for the drops it wants on it (to_ask), and waits, FEED_WAIT_S at most;
  then (or at once while the newest state isn't of the game as expected) None, and Senses asks again, says why
  (`problem`) and stands still after a while. A glance asks for nothing (the next frame comes anyway).
- `wait_frame(timeout)`: until a new frame's state comes; `drain_events(upto)`: every event since the last
  drain of a frame up to the one a look shows, each once; `throw`: her throw, for the tracker.

Frames are HZ a second whatever is asked (`run`, SyncFeed.tick: the user, 2026-10-30, "I meant everything at 15 fps"):
what a look asks for (a frame shown at its time or later, the drops on it) goes into the next frame due, and the look
waits for it. Until the review of 2026-10-30 a frame asked for was taken at once and the next re-phased a period after
it, so a look between two frames had one of its own: 15.7-15.9 frames a second in the simulator's standard grind
(560-643 asked in 5 minutes), every look that wanted drops a pass of its own.

`SyncFeed` is the same thing in-process for the simulator: its clock calls `tick` at every frame the game
shows, the pass runs on the fake eyes there, and each state is ready its measured cost (COSTS) after its frame, the
next frame not before. Nothing it does is charged to the stack's time.
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
from . import worldstate
from .rate import HZ
from .senses import Senses
from .tracking import Event
from .worldstate import Asks, Schedule, WorldState, WorldStep, to_scene

RING_SLOTS = 4       # look frames a Ring keeps by default (the service's: service.RING_SLOTS)
FRAME_SHAPE = (540, 960, 3)  # ... each of this shape at 1920x1080 (frame_shape)
FEED_WAIT_S = 0.6    # s a look waits for the feed's state before it asks again (a frame with every drop in view searched
                     # takes 46-63 ms more; Senses.look stands still after senses.LOST_S of none)
GLANCE_WAIT_S = 0.25  # s a glance waits for a frame from the feed (none: no dot, Senses.glance)
NOT_VISIBLE_S = 0.25  # s between checks while the game isn't wholly on screen
RECOVER_EVERY_S = 0.2  # s at least between two of the eyes' recoveries while frames fail their checks (Eyes.recover:
                       # the minimap's picture and her name tag searched afresh, the capture duplicated afresh or mss),
                       # as a look taken again (senses.RETRY_GAP_S)
KEEP = 8             # states the stack's side keeps
EVENTS_KEPT = 10_000  # events kept for the next drain at most (the oldest go: a stack that never drains)
BELOW_NORMAL = 0x4000  # Windows' BELOW_NORMAL_PRIORITY_CLASS (quiet: the harness's and soak's workers)
IDLE_CLASS = 0x40      # ... and IDLE_PRIORITY_CLASS (a process started at it stays there)


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

class Look:
    """A frame's look, as the world step takes it (`look(ropes)`: Eyes.look_at), with the frame's picture (`full`)
    for the whole screen's search beside it (worldstate.HALVES)."""
    __slots__ = ("eyes", "frame")

    def __init__(self, eyes: Eyes, frame: Frame):
        self.eyes, self.frame = eyes, frame

    @property
    def full(self) -> np.ndarray:
        return self.frame.full

    def __call__(self, ropes: bool) -> Observation:
        return self.eyes.look_at(self.frame.full, self.frame.t, ropes)


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
        return Look(self.eyes, frame)

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
    return Runner(ListFrames([f for f in frames for _ in range(hold)]), eyes, WorldStep(senses, schedule=schedule))


def run(runner: Runner, hz: float | Callable[[], float], poll: Callable[[float], tuple[Asks | None, bool]],
        emit: Callable[[WorldState, Frame | None], None], until: float | None = None,
        shown: Callable[[Frame], None] | None = None) -> None:
    """Frame after frame, `hz` a second (or as often as `hz()` says, asked at each frame; the perception service's is
    one rate, whoever is attached, since 2026-10-26: rate.HZ), one pass each: `poll(wait)` waits up to `wait` s for
    what's asked (asks, stop), and what's asked goes into the next frame due shown at the time asked for or later (no
    frame of its own: HZ a second whatever is asked). Each frame's state goes to `emit`. `shown(frame)`, if given, sees each frame showing the game as soon as it's taken, before the
    pass. Ends on stop, at `until` (time.monotonic), or when the frames run out. A rate gone up takes from the next frame
    (its due brought forward to a period after the last frame's: the service's went from 10 to 30 as a command that
    hunts attached, until 2026-10-26)."""
    due, pending, taken = time.monotonic(), None, None
    while True:
        period = 1.0 / (hz() if callable(hz) else hz)
        if taken is not None:
            due = min(due, taken + period)
        now = time.monotonic()
        asks, stop = poll(max(0.0, due - now))
        if stop or (until is not None and time.monotonic() >= until):
            return
        if asks is not None:
            pending = asks if pending is None else pending.merge(asks)
        if time.monotonic() < due:
            continue  # (an ask waits for the frame due)
        taken = time.monotonic()
        frame = runner.frames.next(due)
        if frame is None:
            return
        if not frame.visible:
            st = runner.step.step(None, frame.t, False, why=frame.why, bad=frame.problem)
            emit(st, frame)
            runner.recover(st.bad)
            due, pending, taken = time.monotonic() + NOT_VISIBLE_S, None, None  # whoever asked takes the look itself
            continue
        use = pending if pending is not None and frame.t >= pending.after - 1e-9 else None
        if shown is not None:
            shown(frame)
        if frame.new or use is not None:  # an unchanged screen nobody asked about: nothing new to say
            st = runner.step.step(runner.look(frame), frame.t, True, use)
            emit(st, frame)
            if st.bad is not None:
                runner.recover(st.bad)
        if use is not None:
            pending = None
        due = max(due + period, time.monotonic())  # (on the grid, or now if it's behind)


def quiet() -> None:
    """This process at below-normal priority (the game and the perception service keep their pace: the harness's,
    soak's and the offline measurements' workers), or at idle if it was started there (a background job's, with all its
    children: its workers had gone up to below normal), and OpenCV on one thread: the monster search's own threads do
    the work, and OpenCV's 24 on top only cost (a 1920x1080 conversion took 25 ms of CPU across them)."""
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.GetPriorityClass.argtypes = (ctypes.c_void_p,)
        k.GetPriorityClass.restype = ctypes.c_uint
        k.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        if k.GetPriorityClass(k.GetCurrentProcess()) != IDLE_CLASS:
            k.SetPriorityClass(k.GetCurrentProcess(), BELOW_NORMAL)
    import cv2
    cv2.setNumThreads(1)


# --- the stack's side ------------------------------------------------------------------------------------

def wanted(after: float, mobs: bool, wide: bool, loot: bool, mesos_only: bool) -> Asks:
    """What a look (Senses.look's arguments) wants of a frame."""
    return Asks(after, band=mobs and not wide, wide=mobs and wide, loot=loot and not wide, loot_wide=loot and wide,
                mesos_only=mesos_only)


def to_ask(want: Asks) -> Asks:
    """What to ask of the next frame for a look that wants `want`: a frame shown at its `after` or later, and the drops
    it wants searched on it. Her band and the whole screen aren't asked for: every frame's state has them (since
    2026-10-30; until then a look asked for them when the newest state's searches would be stale by the next frame)."""
    return Asks(want.after, loot=want.loot, loot_wide=want.loot_wide, mesos_only=want.mesos_only)


def adds(ask: Asks, asked: Asks) -> bool:
    """Whether `ask` wants a search that `asked` (what a look has asked for so far) didn't."""
    return (ask.loot and not asked.loot) or (ask.loot_wide and not asked.loot_wide)


def fresh(st: WorldState, want: Asks) -> bool:
    """Whether state `st` is good for a look that wants `want`: shown at `want.after` or later, and with the drops it
    wants searched on that very frame (her place, the HUD and every monster on screen are every state's). One where she
    isn't placed is as good as a look would be; one that failed its checks (`bad`) is never fresh."""
    # (within 1e-9 s, as WorldStep hands a frame what was asked of it: shown at 5.5 s, a frame took a look's ask for
    # 5.50000001 s, after her last key's effect, and wasn't fresh for it, and the look waited out FEED_WAIT_S, nothing
    # asked again: simulator, 2026-10-26)
    if not st.usable or st.t < want.after - 1e-9:
        return False
    if not st.placed:
        return True
    if want.loot or want.loot_wide:
        if st.loot is None or (want.loot_wide and not st.loot_wide) or (st.mesos_only and not want.mesos_only):
            return False
    return True


def newest_fresh(states, want: Asks) -> WorldState | None:
    """Of `states` (in the order they came), the one of the newest frame that's fresh for `want`."""
    best = None
    for st in states:
        if fresh(st, want) and (best is None or st.t >= best.t):
            best = st
    return best


def newest_frame(states) -> WorldState | None:
    """The state of the newest frame given."""
    return next((st for st in reversed(states) if not st.late), None)


class Feed:
    """What the stack asks of a feed (service.ServiceClient, SyncFeed, InlineFeed)."""
    alive = True
    why = ""  # why it's down, if it is

    def latest(self, want: Asks, timeout: float = FEED_WAIT_S, ask: bool = True) -> WorldState | None:
        """The newest state fresh for `want` (`fresh`), waited for `timeout` s at most; with `ask`, a frame (and the
        drops) asked for when none has them (to_ask); without, the frames as they come (a glance's)."""
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

    def pending_events(self, upto: float | None = None) -> list[Event]:
        """The events not drained yet of frames shown at `upto` or before (None: all), left for the next drain: what
        has happened since the last look, for a glance between looks (behavior.hunter: a throw's last look before its
        key)."""
        return [e for e in list(getattr(self, "events", ())) if upto is None or e.t <= upto + 1e-9]

    def newest(self, after: float) -> WorldState | None:
        """The newest state there is, if it's usable and of a frame shown at `after` or later; else None. Nothing is
        asked of the next frame, nor waited for: a glance at the world as perception has it now (behavior.hunter's last
        look before a key), which changes nothing of what the feed looks at and when."""
        st = newest_frame(list(self._states()))
        return st if st is not None and st.usable and st.t >= after else None

    def throw(self, t: float, side: str, attack: str | None = None, aim: tuple | None = None) -> None:
        """She threw toward `side` at t (Actions.throw_listeners), with `attack` (its id: gamelab.character) and `aim`
        (the monster her attack choice aimed it at: tracking.MonsterTracker.throw), if told."""

    def close(self) -> None:
        pass

    def look(self, senses: Senses, mobs: bool, wide: bool, loot: bool, mesos_only: bool, after: float | None = None):
        """The Scene of a look with these arguments from the feed, or None (Senses then asks again). A look at her
        band also has the monsters on the rest of the screen (`Scene.swept`: the fights go by the band's, her memory
        takes them all), and says where it searched (`Scene.rows`): her band and the whole screen, every frame's.
        `after`: of a frame shown then or later (None: now)."""
        st = self.latest(wanted(time.monotonic() if after is None else after, mobs, wide, loot, mesos_only))
        if st is None:
            return None
        s = to_scene(st, senses.model, senses.band, mobs, wide, loot, mesos_only, self.frame_of(st))
        if s.searched and not s.wide:
            s.swept = [m for m in st.monsters if not any(m is k for k in s.monsters)]
            s.rows = [senses._map_rows(s, band_rows(s, *senses.band)), senses._play_rows(s)]
        return s

    def glance(self, after: float, timeout: float = GLANCE_WAIT_S) -> WorldState | None:
        """The newest visible state of a frame shown at `after` or later (waited for, nothing asked: the frames come
        HZ a second anyway), or None."""
        return self.latest(Asks(after), timeout, ask=False)


def _drain(events: deque[Event], upto: float | None) -> list[Event]:
    """Take the events of frames shown at `upto` or before (all if None) off the front of `events` (in the order
    their states came)."""
    out = []
    while events and (upto is None or events[0].t <= upto + 1e-9):
        out.append(events.popleft())
    return out


# --- in-process, for the simulator ---------------------------------------------------------------------

@dataclass(frozen=True)
class Costs:
    """What each part of a frame's pass takes the feed (s), as measured offline at the game's own 1366x768 (the live
    size), for the simulator's feed: `capture`, the frame copied off the graphics card and converted; `look`, her
    look (her place, ropes, the HUD); `wide`, the whole screen's search, at once with her look (worldstate.HALVES:
    the longer of the two and `beside` more, else both); `bar`, a monster's HP bar read; `death`, a track looked at for
    its death; `loot_strip`, a strip of drops searched."""
    capture: float = 0.002     # (bench-capture: 2.0 ms)
    look: float = 0.0085       # her look alone, the thief's (her dot's, ropes included: 8.5 ms; the magician's by the
                               # scene 4.5)
    wide: float = 0.0084       # the whole screen at 1366x768: 8.39 ms (median; p90 12.6) on the user's own play's clip
    beside: float = 0.0018     # the halves at once took 10.3 ms (median; p90 15.9) where the longer took 8.5
    bar: float = 0.0011        # an HP bar (perception.hud.hp_bar: 1.1 ms)
    death: float = 0.0008      # a death look (perception.deaths: 0.83 ms on one thread)
    loot_strip: float = 0.0038  # a strip of drops (every platform in view, 12 strips, 46.5 ms: drops_cost.py)

    def of(self, st: WorldState) -> float:
        """What a frame's pass took (its state's `work`: whether the whole screen was searched, the bars read, the
        death looks, the strips of drops)."""
        w = st.work
        if not st.visible:
            return self.capture
        both = (max(self.look, self.wide) + self.beside if worldstate.HALVES else self.look + self.wide) \
            if "wide s" in w else self.look
        return (self.capture + both + self.bar * w.get("bars", 0) + self.death * w.get("deaths", 0)
                + self.loot_strip * w.get("strips", 0))


# The pass's costs at the live size, 1366x768 (D:/MapleHelper-scratch/replan/p15: halves.py, cpu15.py, drops_cost.py,
# clip_mobs.py; Idle priority on CPUs 12-23, another workflow's replay beside it): her look and the whole screen at once
# 10.3 ms (median), a strip of drops 3.84. Until 2026-10-30 COSTS were 1920x1080's (her place 12 ms, the whole screen 100
# on its thread) and COSTS_1366 the service's at 1366x768 (the harness's `--costs measured-1366`); a whole screen of 100
# ms in every frame would hold the simulator's feed to 9 frames a second.
COSTS = Costs()
COSTS_1366 = COSTS  # (the harness's `--costs measured-1366`: the same since 2026-10-30)
GAME_HZ = 60.0  # frames a second the game shows


class SyncFeed(Feed):
    """The feed in-process, stepped by whoever keeps time: `tick(t)` at every frame the game shows (the
    simulator's clock calls it, GAME_HZ), `look(ropes)` reading the screen then (FakeEyes.observe). A frame is
    taken `hz` times a second, what's asked going into the next one due (as `run`: no frame of its own); its state is
    ready its cost later (`costs`: Costs.of), the next frame not before. Waiting goes through time.sleep, the
    simulator's own."""

    def __init__(self, step: WorldStep, look: Callable[[bool], Observation], costs: Costs | None = None,
                 hz: float = HZ, guard: Callable[[], None] | None = None):
        """`costs`: by default COSTS as it is when the feed is made (the harness's `--costs` may set it for a run);
        `guard`: called at every ask; it raises when the fakes aren't in place (sim.scenarios.Sim.guard)."""
        costs = COSTS if costs is None else costs
        self.step, self._look, self.costs, self.hz, self.guard = step, look, costs, hz, guard
        self.states: deque[tuple[float, WorldState]] = deque(maxlen=KEEP)  # (ready at, state)
        self.events: deque[Event] = deque()
        self.asks: Asks | None = None
        self.busy_until = -1e18
        self.next_due = -1e18
        self._seen_seq = 0
        self.counts = {"frames": 0, "asked": 0, "busy": 0.0}

    def tick(self, t: float) -> None:
        """The game shows a frame at t."""
        if t < self.busy_until - 1e-9 or t < self.next_due - 1e-9:
            return  # (not due: what's asked waits for the frame due)
        asked = self.asks is not None and t >= self.asks.after - 1e-9
        asks, self.asks = (self.asks, None) if asked else (None, self.asks)
        st = self.step.step(self._look, t, True, asks)
        cost = self.costs.of(st)
        end = t + cost
        self.busy_until, self.next_due = end, t + 1.0 / self.hz
        self.counts["frames"] += 1
        self.counts["asked"] += asked
        self.counts["busy"] += cost
        self.states.append((end, st))
        self.events.extend(st.events)

    def latest(self, want: Asks, timeout: float = FEED_WAIT_S, ask: bool = True) -> WorldState | None:
        if self.guard is not None:
            self.guard()
        deadline, asked = time.monotonic() + timeout, False
        while True:
            now = time.monotonic()
            ready = [st for at, st in self.states if at <= now + 1e-9]
            st = newest_fresh(ready, want)
            if st is not None:
                self._seen_seq = max(self._seen_seq, st.seq)
                return st
            newest = newest_frame(ready)
            if newest is not None and newest.t >= want.after and not newest.usable:
                return None  # (the game not on screen or not as expected: Senses says why, and asks again)
            if ask and not asked and not any(fresh(st, want) for _, st in self.states):
                self.asks = to_ask(want) if self.asks is None else self.asks.merge(to_ask(want))
                asked = True  # (once: a frame asked for comes, or the look asks again)
            if now >= deadline:
                return None
            wakes = [ready for ready, st in self.states if ready > now and fresh(st, want)]
            time.sleep(max(1e-4, min(wakes + [now + 1.0 / GAME_HZ, deadline]) - now))

    def frame_of(self, st: WorldState) -> np.ndarray:
        return st.frame

    def _states(self):
        now = time.monotonic()
        return [st for ready, st in self.states if ready <= now + 1e-9]

    def wait_frame(self, timeout: float) -> bool:
        seen = max((st.seq for ready, st in self.states if ready <= time.monotonic()), default=0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline - 1e-9:
            now = time.monotonic()
            if any(st.seq > seen and ready <= now for ready, st in self.states):
                return True
            later = [ready for ready, st in self.states if st.seq > seen and ready > now]
            time.sleep(max(1e-4, min(later + [now + 1.0 / GAME_HZ, deadline]) - now))
        return False

    def drain_events(self, upto: float | None = None) -> list[Event]:
        return _drain(self.events, upto)

    def throw(self, t: float, side: str, attack: str | None = None, aim: tuple | None = None) -> None:
        self.step.throw(t, side, attack, aim)


# --- on saved frames and in tests -------------------------------------------------------------------------

class InlineFeed(Feed):
    """The world step run when a look or glance asks, on the eyes' own frame then (Eyes.look on their capture: saved
    frames, bench-look's and bodies'; the tests' made-up frames; the simulator's FakeEyes charging a look's time): a
    command's looks come from the perception service instead (localization.service), whose frames it looks at all the
    time, and the simulator's from SyncFeed. Each look is a frame of its own, taken when it asks (its `after` aside: the
    eyes' clock may be another's, a test's), its pass as the service's (no drops by rotation: the drops asked for)."""

    def __init__(self, eyes: Eyes, model, band: tuple[float, float]):
        self.eyes = eyes
        self.step = WorldStep(Senses(eyes, model, band), schedule=Schedule(drops_strips=0))
        self.events: deque[Event] = deque(maxlen=EVENTS_KEPT)
        self.state: WorldState | None = None  # the last frame's state

    def latest(self, want: Asks, timeout: float = FEED_WAIT_S, ask: bool = True) -> WorldState | None:
        st = self._frame(want)
        return st if st.usable else None

    def _frame(self, asks: Asks | None) -> WorldState:
        """A frame stepped now."""
        t = time.monotonic()
        try:
            obs = self.eyes.look()
        except BadLook as e:  # (eyes that check their own frames: the game window, offline)
            st = self.step.step(None, t, False, why=e.why, bad=e.problem)
        else:  # (what's asked, of this very frame: the eyes' clock may not be the asker's)
            st = self.step.step(lambda ropes: obs, obs.t, True, None if asks is None else replace(asks, after=-1e18))
        if st.bad is not None and hasattr(self.eyes, "recover"):  # before the next look, as the service's frame loop
            self.eyes.recover(st.bad)
        self.events.extend(st.events)
        self.state = st
        return st

    def _states(self):
        return () if self.state is None else (self.state,)

    def newest(self, after: float) -> WorldState | None:
        """The state of the last frame stepped, if usable: no frame of its own (each look takes one)."""
        return self.state if self.state is not None and self.state.usable else None

    def frame_of(self, st: WorldState) -> np.ndarray:
        return st.frame

    def wait_frame(self, timeout: float) -> bool:
        return True  # (each look takes a frame of its own)

    def drain_events(self, upto: float | None = None) -> list[Event]:
        return _drain(self.events, upto)

    def throw(self, t: float, side: str, attack: str | None = None, aim: tuple | None = None) -> None:
        self.step.throw(t, side, attack, aim)
