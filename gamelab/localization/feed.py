"""The continuous feed: the game looked at every frame, in a process of its own, and the latest of what it saw
handed to the stack as it asks (Senses(feed=...)).

Why a process: the stack's planning holds Python's lock, and a perception thread in the same process got
16 of its 60 looks in 2 s while the planner worked (the research's gil.py; Python 3.15 here has the lock on).
So the feed is a child process (`serve`, started with spawn, at below-normal priority so the game, mostly
on one core, keeps its frame pacing). Each frame it runs the world step (localization.worldstate): her
place, the monsters tracked, her EXP gains and hits. The half-size frames go into a ring of RING_SLOTS in
shared memory (`Ring`: the oldest overwritten, never more), and each frame's state (plain data, a few KB)
down a pipe; the stack's requests (what it wants searched on the next frame, her throws) go up another.

The stack's side (`FeedClient`) keeps the last few states, read off the pipe by a thread of its own:
- `latest(after, want)`: the newest state of a frame shown at `after` or later (a look sees what came of
  the keys sent before it) that has what's wanted fresh: monsters in her band searched within MOBS_FRESH_S of
  its frame (tracked ones are re-found every frame), the whole screen within WIDE_FRESH_S, drops searched on
  that very frame (a pile just picked up must not be seen again). If none has, it asks for it on the next
  frame and waits, FEED_WAIT_S at most; then (or once the game isn't in front, or the child has died) None,
  and Senses takes the look itself as before.
- `wait_frame(timeout)`: until a new frame's state comes; `drain_events(upto)`: every event since the last
  drain of a frame up to the one a look shows, each once.

`SyncFeed` is the same thing in-process for the simulator: its clock calls `tick` at every frame the game
shows, the world step runs on the fake eyes there, and each state is ready its measured cost (COSTS) after
its frame. Nothing it does is charged to the stack's time.
"""

import ctypes
import multiprocessing
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from ..perception.eyes import Eyes, Observation
from ..perception.frames import Frame
from .senses import Senses
from .tracking import Event
from .worldstate import Asks, Schedule, WorldState, WorldStep, to_scene

HZ = 30.0            # frames a second the feed looks at (the game shows 60; a look takes 16 ms, a band search 32)
RING_SLOTS = 4       # half-size frames kept in shared memory: at 30 Hz the newest 133 ms
FRAME_SHAPE = (540, 960, 3)
MOBS_FRESH_S = 0.3   # a look's monsters: her band searched at most this long before its frame. It's searched every
                     # 0.2 s (worldstate.BAND_EVERY_S), and the frame after that and the search itself take about 0.1 s
                     # more: with 0.25, on the saved frames a third of the frames searched it, every look asking.
WIDE_FRESH_S = 0.5   # ... the whole screen (it's searched every second; tracked monsters are re-found every frame)
FEED_WAIT_S = 0.6    # s a look waits for the feed before it takes the look itself (the whole screen with every
                     # drop in view takes ~0.25 s)
GLANCE_WAIT_S = 0.25  # s a glance waits for a frame from the feed before it takes one itself
NOT_VISIBLE_S = 0.25  # s between checks while the game isn't wholly on screen
KEEP = 8             # states the stack's side keeps
EVENTS_KEPT = 10_000  # events kept for the next drain at most (the oldest go: a stack that never drains)
BELOW_NORMAL = 0x4000  # Windows' BELOW_NORMAL_PRIORITY_CLASS
# True: a look asks the next frame only for what the newest state wouldn't have fresh by then (to_ask); False:
# for everything it wants, as a blocking look searches it all on its own frame. Asking only for what's stale,
# the feed's own loop on the saved frames (a look every 0.1 s, every 5th with the drops) searched her band 4.9
# times a second instead of 7.0 and looked at 21.7 frames a second instead of 18.3. But a look's monsters were
# then up to 0.3 s older than its frame, and in the simulator (harness, preset game, 20 seeds x 20 min, --costs
# measured, --feed; the feed's frames charged their measured costs, so its slower frame rate counts) asking for
# everything made the hunt loop 7.1% +- 1.8% better (15 of 19 seeds; one fell out of the map at the basin's
# end), rests 3.32 -> 2.63 a run, and the tour 9.0% +- 2.7% (15 of 20). Asking for her band always and the
# whole screen only when stale was no better (-0.5% and -1.6%). Searching her band every 0.1 s instead of 0.2
# gained 3.5% in both.
ASK_ONLY_STALE = False


# --- frames in shared memory --------------------------------------------------------------------------

class Ring:
    """Half-size frames in shared memory, `slots` of them, each with its frame number and time: the child
    writes them in turn (the oldest overwritten), the stack copies one out by its number. Made by the stack's
    side (`name` None) and opened by name in the child."""

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

    def look(self, frame: Frame) -> Callable[[bool], Observation]:
        return lambda ropes: self.eyes.look_at(frame.full, frame.t, ropes)


def screen_world(map_id: int, title: str, template: np.ndarray, band: tuple[float, float], throws: bool = True,
                 schedule: Schedule = Schedule()) -> Runner:
    """The game on screen (perception.frames.ScreenFrames: never brought to the front), with monster
    templates for map `map_id` and her name tag (`template`, calibration's)."""
    from ..perception.frames import ScreenFrames
    from ..perception.sprites import detector_for
    from . import mapdata
    model = mapdata.load(map_id)
    frames = ScreenFrames(title)
    eyes = Eyes(title, template, detector_for(model), screen=frames.grabber)
    return Runner(frames, eyes, WorldStep(Senses(eyes, model, band), throws, schedule), title)


def saved_world(files: list[str], map_id: int, template: np.ndarray, band: tuple[float, float], hold: int = 30,
                schedule: Schedule = Schedule()) -> Runner:
    """Saved screenshots (full-size game frames) for a screen, each `hold` frames in a row, round and round:
    the feed run offline (`python -m gamelab.sim.soak feed`)."""
    from PIL import Image

    from ..perception.frames import ListFrames
    from ..perception.sprites import detector_for
    from . import mapdata
    model = mapdata.load(map_id)
    frames = [np.asarray(Image.open(f).convert("RGB")) for f in files]
    eyes = Eyes("saved frames", template, detector_for(model), screen=object())  # it's handed its frames
    return Runner(ListFrames([f for f in frames for _ in range(hold)]), eyes, WorldStep(Senses(eyes, model, band),
                                                                                        schedule=schedule))


def run(runner: Runner, hz: float, poll: Callable[[float], tuple[Asks | None, bool]],
        emit: Callable[[WorldState, Frame], None], until: float | None = None) -> None:
    """Frame after frame, `hz` a second: `poll(wait)` waits up to `wait` s for what's asked (asks, stop), and
    what's asked is done on the first frame shown at the time asked for, taken at once then. Each frame's
    state goes to `emit`. Ends on stop, at `until` (time.monotonic), or when the frames run out."""
    period = 1.0 / hz
    due, pending = time.monotonic(), None
    while True:
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
            emit(runner.step.step(None, frame.t, False, why=frame.why), frame)
            due, pending = time.monotonic() + NOT_VISIBLE_S, None  # whoever asked takes the look itself
            continue
        use = pending if asked and frame.t >= pending.after else None
        if frame.new or use is not None:  # an unchanged screen nobody asked about: nothing new to say
            emit(runner.step.step(runner.look(frame), frame.t, True, use), frame)
        if use is not None:
            pending = None
        if time.monotonic() >= due:
            due = max(due + period, time.monotonic())


def serve(requests, states, ring_name: str, make: Callable[[], Runner], hz: float = HZ) -> None:
    """The child process: builds its runner with `make()`, then runs it, frames into the ring and states down
    `states`, asks and throws read from `requests`, until asked to stop or the stack's end of a pipe closes."""
    quiet()
    try:
        runner = make()
        ring = Ring(ring_name)
    except Exception as e:
        states.send(("error", f"the feed couldn't start: {type(e).__name__}: {e}"))
        return
    states.send(("hello", {"priority": _priority(), "pid": multiprocessing.current_process().pid}))

    def poll(wait: float) -> tuple[Asks | None, bool]:
        asks = None
        deadline = time.monotonic() + wait
        while True:
            left = deadline - time.monotonic()
            if not requests.poll(max(0.0, left) if asks is None else 0):
                return asks, False
            msg = requests.recv()
            if msg[0] == "stop":
                return asks, True
            if msg[0] == "ask":
                asks = msg[1] if asks is None else asks.merge(msg[1])  # the older first: the newer's mesos_only
            elif msg[0] == "throw":
                runner.step.throw(*msg[1:])

    def emit(st: WorldState, frame: Frame) -> None:
        if st.frame is not None:
            st.slot = ring.put(st.seq, st.t, st.frame)
        states.send(("state", replace(st, frame=None)))

    try:
        run(runner, hz, poll, emit)
        states.send(("bye", runner.step.counts))
    except (EOFError, BrokenPipeError, OSError):
        pass  # the stack is gone
    except Exception as e:  # a bug: the stack is told why before it looks for itself
        try:
            states.send(("error", f"the feed stopped: {type(e).__name__}: {e}"))
        except OSError:
            pass
        raise
    finally:
        ring.close()


def quiet() -> None:
    """This process at below-normal priority (the game keeps its frame pacing), and OpenCV on one thread: the
    monster search's own threads do the work, and OpenCV's 24 on top only cost (a 1920x1080 conversion took
    25 ms of CPU across them)."""
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        k.SetPriorityClass(k.GetCurrentProcess(), BELOW_NORMAL)
    import cv2
    cv2.setNumThreads(1)


def _priority() -> int | None:
    if sys.platform != "win32":
        return None
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetPriorityClass.argtypes = (ctypes.c_void_p,)
    return k.GetPriorityClass(k.GetCurrentProcess())


# --- the stack's side ------------------------------------------------------------------------------------

def wanted(after: float, mobs: bool, wide: bool, loot: bool, mesos_only: bool) -> Asks:
    """What a look (Senses.look's arguments) wants of a frame."""
    return Asks(after, band=mobs and not wide, wide=mobs and wide, loot=loot and not wide, loot_wide=loot and wide,
                mesos_only=mesos_only)


def to_ask(want: Asks, newest: WorldState | None, frame_s: float = 1.0 / HZ) -> Asks:
    """What to ask of the next frame for a look that wants `want`: all of it, or (ASK_ONLY_STALE) only what the
    next frame wouldn't have fresh anyway, judging by the newest state."""
    if not ASK_ONLY_STALE:
        return want
    soon = want.after + frame_s
    stale = lambda at, fresh_s: newest is None or at is None or soon - at > fresh_s
    return Asks(want.after, band=want.band and stale(newest and newest.band_at, MOBS_FRESH_S),
                wide=want.wide and stale(newest and newest.wide_at, WIDE_FRESH_S), loot=want.loot,
                loot_wide=want.loot_wide, mesos_only=want.mesos_only)


def fresh(st: WorldState, want: Asks) -> bool:
    """Whether state `st` is good for a look that wants `want`: shown at `want.after` or later, and fresh as
    the look needs (see the top). One where she isn't placed is as good as a look would be."""
    if not st.visible or st.t < want.after:
        return False
    if not st.placed:
        return True
    if want.band and (st.band_at is None or st.t - st.band_at > MOBS_FRESH_S):
        return False
    if want.wide and (st.wide_at is None or st.t - st.wide_at > WIDE_FRESH_S):
        return False
    if want.loot or want.loot_wide:
        if st.loot is None or (want.loot_wide and not st.loot_wide) or (st.mesos_only and not want.mesos_only):
            return False
    return True


class Feed:
    """What the stack asks of a feed (FeedClient, SyncFeed)."""
    alive = True
    why = ""  # why it's down, if it is

    def latest(self, want: Asks, timeout: float = FEED_WAIT_S) -> WorldState | None:
        raise NotImplementedError

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

    def look(self, senses: Senses, mobs: bool, wide: bool, loot: bool, mesos_only: bool):
        """The Scene of a look with these arguments from the feed, or None (Senses then looks itself)."""
        st = self.latest(wanted(time.monotonic(), mobs, wide, loot, mesos_only))
        if st is None:
            return None
        return to_scene(st, senses.model, senses.band, mobs, wide, loot, mesos_only, self.frame_of(st))

    def glance(self, after: float, timeout: float = GLANCE_WAIT_S) -> WorldState | None:
        """The newest visible state of a frame shown at `after` or later (waited for), or None."""
        return self.latest(Asks(after), timeout)


class FeedClient(Feed):
    """The feed's child process, from the stack's side. `make`: a picklable callable building the child's
    Runner (screen_world, with its arguments, for the game)."""

    def __init__(self, make: Callable[[], Runner], hz: float = HZ, start: bool = True):
        self.make, self.hz = make, hz
        self.states: deque[WorldState] = deque(maxlen=KEEP)
        self.events: deque[Event] = deque(maxlen=EVENTS_KEPT)
        self.dropped = 0          # events lost because nothing drained them
        self.served = self.refused = 0  # states handed out by latest(), and times it had none to give
        self.hello: dict | None = None
        self.bye: dict | None = None
        self.alive, self.why = False, "not started"
        self._cond = threading.Condition()
        self._send_lock = threading.Lock()
        self._seen_seq = 0
        self.proc = None
        if start:
            self.start()

    def start(self, timeout: float = 30.0) -> bool:
        """Start the child and wait (`timeout` s at most) for it to be ready; False (with `why`) if it isn't."""
        ctx = multiprocessing.get_context("spawn")
        self.ring = Ring()
        child_requests, self._requests = ctx.Pipe(duplex=False)  # each (receiving end, sending end)
        self._states, child_states = ctx.Pipe(duplex=False)
        self.proc = ctx.Process(target=serve, args=(child_requests, child_states, self.ring.name, self.make, self.hz),
                                name="gamelab-feed", daemon=True)
        self.proc.start()
        child_requests.close()
        child_states.close()
        self.alive, self.why = True, ""
        self._reader = threading.Thread(target=self._read, name="feed-reader", daemon=True)
        self._reader.start()
        with self._cond:
            self._cond.wait_for(lambda: self.hello is not None or not self.alive, timeout)
        if self.hello is None and self.alive:
            self.why = f"the feed didn't start within {timeout:.0f} s"
            self.close()
        return self.alive

    def _read(self) -> None:
        """The pipe from the child, read as it comes (a thread of its own, so the child never waits on it)."""
        try:
            while True:
                msg = self._states.recv()
                with self._cond:
                    if msg[0] == "state":
                        st = msg[1]
                        self.states.append(st)
                        room = EVENTS_KEPT - len(self.events)
                        self.dropped += max(0, len(st.events) - room)
                        self.events.extend(st.events)
                    elif msg[0] == "hello":
                        self.hello = msg[1]
                    elif msg[0] == "error":
                        self.why, self.alive = msg[1], False
                    elif msg[0] == "bye":
                        self.bye = msg[1]
                    self._cond.notify_all()
        except (EOFError, OSError):
            pass
        with self._cond:
            if self.alive:
                self.alive, self.why = False, self.why or "the feed's process ended"
            self._cond.notify_all()

    def _send(self, *msg) -> None:
        try:
            with self._send_lock:
                self._requests.send(msg)
        except (OSError, ValueError):
            with self._cond:
                self.alive, self.why = False, self.why or "the feed's process ended"

    def latest(self, want: Asks, timeout: float = FEED_WAIT_S) -> WorldState | None:
        deadline, asked = time.monotonic() + timeout, None
        with self._cond:
            while True:
                if not self.alive:
                    self.refused += 1
                    return None
                for st in reversed(self.states):
                    if fresh(st, want):
                        self._seen_seq = max(self._seen_seq, st.seq)
                        self.served += 1
                        return st
                newest = self.states[-1] if self.states else None
                left = deadline - time.monotonic()
                if (newest is not None and newest.t >= want.after and not newest.visible) or left <= 0:
                    self.refused += 1  # the game isn't on screen (the look brings it forward itself), or too slow
                    return None
                seq = newest.seq if newest is not None else 0
                if seq != asked:  # nothing fresh as of the newest state: asked (again) of the next frame
                    self._send("ask", to_ask(want, newest))
                    asked = seq
                self._cond.wait(left)

    def frame_of(self, st: WorldState) -> np.ndarray:
        frame = None if st.slot is None else self.ring.get(st.slot, st.seq)
        return frame if frame is not None else np.zeros(FRAME_SHAPE, np.uint8)

    def wait_frame(self, timeout: float) -> bool:
        with self._cond:
            seen = self.states[-1].seq if self.states else 0
            return self._cond.wait_for(lambda: not self.alive or (self.states and self.states[-1].seq > seen),
                                       timeout) and self.alive

    def drain_events(self, upto: float | None = None) -> list[Event]:
        with self._cond:
            return _drain(self.events, upto)

    def throw(self, t: float, side: str) -> None:
        if self.alive:
            self._send("throw", t, side)

    def close(self) -> None:
        """Stop the child (asked, then made to) and free the ring."""
        if self.proc is not None:
            self._send("stop")
            self.proc.join(2.0)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(2.0)
        with self._cond:
            self.alive = False
            self.why = self.why or "closed"
            self._cond.notify_all()
        for end in (getattr(self, "_requests", None), getattr(self, "_states", None)):
            if end is not None:
                end.close()
        if getattr(self, "ring", None) is not None:
            self.ring.close(unlink=True)
            self.ring = None


def _drain(events: deque[Event], upto: float | None) -> list[Event]:
    """Take the events of frames shown at `upto` or before (all if None) off the front of `events` (in frame
    order: each state's come after the one before's)."""
    out = []
    while events and (upto is None or events[0].t <= upto + 1e-9):
        out.append(events.popleft())
    return out


# --- in-process, for the simulator ---------------------------------------------------------------------

@dataclass(frozen=True)
class Costs:
    """What each part of a frame takes the feed (s), as measured offline (bench-look medians on the 17 saved
    frames after the profiling pass; the re-finds on them, refind_bench in the step-5 report). Before that
    pass: her place 0.016, her band 0.032, the whole screen 0.122."""
    capture: float = 0.002     # the frame copied off the graphics card and converted (bench-capture: 2.0 ms)
    look: float = 0.012        # her place, ropes included (Senses.look without monsters: 13.3 ms, 1.3 of it the
                               # saved frame's conversion, which `capture` counts)
    refind: float = 0.0007     # a monster re-found with the template that matched it last (0.69 ms)
    fallback: float = 0.0034   # ... and with its kind's whole set, when that missed (3.4 ms)
    band: float = 0.022        # her band searched (the feed step with it 34.2 ms, without 12.9)
    wide: float = 0.086        # the whole screen (98.6 - 12.9 ms)
    loot_strip: float = 0.007  # a strip searched for drops (her platform's 15.6 ms in 1.9 strips; every platform
                               # in view, 71 ms in 28.6)

    def of(self, st: WorldState) -> float:
        w = st.work
        if not st.visible:
            return self.capture
        return (self.capture + self.look + self.refind * w.get("re-finds", 0) + self.fallback * w.get("fallbacks", 0)
                + self.band * w.get("band", 0) + self.wide * w.get("wide", 0)
                + self.loot_strip * w.get("loot strips", 0))


COSTS = Costs()
GAME_HZ = 60.0  # frames a second the game shows


class SyncFeed(Feed):
    """The feed in-process, stepped by whoever keeps time: `tick(t)` at every frame the game shows (the
    simulator's clock calls it, GAME_HZ), `look(ropes)` reading the screen then (FakeEyes.observe). A frame is
    taken `hz` times a second, or at once when something was asked; its state is ready its cost later
    (`costs`), the next frame not before. Waiting goes through time.sleep, the simulator's own."""

    def __init__(self, step: WorldStep, look: Callable[[bool], Observation], costs: Costs = COSTS, hz: float = HZ):
        self.step, self._look, self.costs, self.hz = step, look, costs, hz
        self.states: deque[tuple[float, WorldState]] = deque(maxlen=KEEP)  # (ready at, state)
        self.events: deque[Event] = deque()
        self.asks: Asks | None = None
        self.busy_until = -1e18
        self.next_due = -1e18
        self._seen_seq = 0
        self.counts = {"frames": 0, "asked": 0, "busy": 0.0}

    def tick(self, t: float) -> None:
        """The game shows a frame at t."""
        if t < self.busy_until - 1e-9:
            return
        asked = self.asks is not None and t >= self.asks.after - 1e-9
        if not asked and t < self.next_due - 1e-9:
            return
        asks, self.asks = (self.asks, None) if asked else (None, self.asks)
        st = self.step.step(self._look, t, True, asks)
        cost = self.costs.of(st)
        self.busy_until, self.next_due = t + cost, t + 1.0 / self.hz
        self.counts["frames"] += 1
        self.counts["asked"] += asked
        self.counts["busy"] += cost
        self.states.append((t + cost, st))
        self.events.extend(st.events)

    def latest(self, want: Asks, timeout: float = FEED_WAIT_S) -> WorldState | None:
        deadline = time.monotonic() + timeout
        while True:
            now = time.monotonic()
            for ready, st in reversed(self.states):
                if ready <= now + 1e-9 and fresh(st, want):
                    self._seen_seq = max(self._seen_seq, st.seq)
                    return st
            if not any(fresh(st, want) for _, st in self.states):
                newest = self.states[-1][1] if self.states else None
                ask = to_ask(want, newest, 1.0 / self.hz)
                self.asks = ask if self.asks is None else self.asks.merge(ask)
            if now >= deadline:
                return None
            pending = [ready for ready, st in self.states if ready > now and fresh(st, want)]
            time.sleep(max(1e-4, min(pending + [now + 1.0 / GAME_HZ, deadline]) - now))

    def frame_of(self, st: WorldState) -> np.ndarray:
        return st.frame

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

    def throw(self, t: float, side: str) -> None:
        self.step.throw(t, side)
