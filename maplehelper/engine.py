"""Runs combos: presses keys with timing, narrates each step, and can be stopped mid-way."""

import logging
import math
import queue
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from .config import GameKeys, Timing
from .keys import Key

log = logging.getLogger(__name__)

LEFT, RIGHT = "left", "right"

SendKey = Callable[[Key, bool], None]


class Display(Protocol):
    def step(self, title: str, text: str) -> None: ...
    def done(self, title: str, text: str, recipe: list[str], linger_ms: int) -> None: ...


class Aborted(Exception):
    pass


@dataclass(frozen=True)
class Combo:
    name: str                          # used in config.toml [hotkeys]
    title: str                         # shown on the coach overlay
    script: Callable[["Ctx"], None]


class Ctx:
    """What a combo script uses to act.

    `front` is the arrow pointing at the monsters (the way she faced when she pressed the
    button) and `back` points away from them; both stay fixed for the whole combo.
    """

    def __init__(self, send: SendKey, keys: GameKeys, timing: Timing, facing: str,
                 slow: bool, abort: threading.Event, display: Display, title: str):
        self.keys = keys
        self.t = timing
        self.front = keys.left if facing == LEFT else keys.right
        self.back = keys.right if facing == LEFT else keys.left
        self.facing = facing  # updated as the combo turns her around
        self.turned_at = 0    # perf_counter_ns of the combo's last arrow press; 0 if it never turned her
        self.recipe: list[str] = []
        self._send = send
        self._slow = slow
        self._abort = abort
        self._display = display
        self._title = title
        self._held: set[Key] = set()

    def say(self, text: str, pause: bool = True) -> None:
        """Start a new step: show it on the coach. In slow mode, pause so she can read it,
        except for steps that happen mid-move (pause=False), such as while she's in the air."""
        self.recipe.append(text)
        self._display.step(self._title, text)
        if self._slow and pause:
            self.wait(self.t.slow_pause)

    def wait(self, ms: int) -> None:
        if self._abort.wait(ms / 1000):
            raise Aborted

    def down(self, key: Key) -> None:
        if self._abort.is_set():
            raise Aborted
        self._send(key, True)
        self._held.add(key)
        if key in (self.keys.left, self.keys.right):
            self.facing = LEFT if key == self.keys.left else RIGHT
            self.turned_at = time.perf_counter_ns()

    def up(self, key: Key) -> None:
        self._send(key, False)
        self._held.discard(key)

    def tap(self, key: Key, ms: int | None = None) -> None:
        """Press and release a key; `ms` is how long to hold it (default: a quick tap)."""
        self.down(key)
        self.wait(self.t.key_press if ms is None else ms)
        self.up(key)

    def release_all(self) -> None:
        for key in self._held:
            self._send(key, False)
        self._held.clear()

    @property
    def throw_label(self) -> str:
        k = self.keys
        if k.lucky_seven:
            return f"{k.lucky_seven.label} (Lucky Seven)"
        return f"{k.jump.label} + {k.attack.label} in mid-air"

    @property
    def air_throw_label(self) -> str:
        k = self.keys
        return f"{k.lucky_seven.label} (Lucky Seven)" if k.lucky_seven else k.attack.label

    def air_throw(self) -> None:
        """While airborne, Lucky Seven or a plain attack both throw a star."""
        self.tap(self.keys.lucky_seven or self.keys.attack)

    def jump_attack(self) -> None:
        """Attacking while airborne throws a star instead of swiping with the claw."""
        self.tap(self.keys.jump)
        self.wait(self.t.air_delay)
        self.tap(self.keys.attack)

    def throw(self, times: int = 1) -> None:
        for _ in range(times):
            if self.keys.lucky_seven:
                self.tap(self.keys.lucky_seven)
            else:
                self.jump_attack()
            self.wait(self.t.throw_gap)


class _Silent:
    def step(self, title: str, text: str) -> None:
        pass

    def done(self, title: str, text: str, recipe: list[str], linger_ms: int) -> None:
        pass


class _DryRun(Ctx):
    """Plays a combo script without pressing keys or sleeping, adding up its waits to time it."""

    def __init__(self, keys: GameKeys, timing: Timing, facing: str, slow: bool):
        super().__init__(lambda key, down: None, keys, timing, facing, slow, threading.Event(), _Silent(), "")
        self.elapsed_ms = 0

    def wait(self, ms: int) -> None:
        self.elapsed_ms += ms


# What Runner.trigger did with a press.
STARTED, QUEUED, DROPPED = "started", "queued", "dropped"


class Runner:
    """Runs one combo at a time on a worker thread, so the keyboard hook never waits."""

    def __init__(self, send: SendKey, keys: GameKeys, timing: Timing, display: Display, combos: Iterable[Combo]):
        self.facing = RIGHT
        self.slow = False
        self.combos = {c.name: c for c in combos}
        self._send = send
        self._keys = keys
        self._timing = timing
        self._display = display
        self._abort = threading.Event()
        self._queue: queue.SimpleQueue[Combo | None] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._busy = False
        self._buffered: Combo | None = None  # pressed near the end of the running combo; runs right after it
        self._ends_at = math.inf             # perf_counter time the running combo is expected to end
        self._her_turn_at = 0                # perf_counter_ns of her last ← / → press
        self._thread = threading.Thread(target=self._work, name="combo-runner", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.abort()
        self._queue.put(None)
        self._thread.join(timeout=2)

    def trigger(self, name: str) -> str:
        """Start a combo without blocking, and return what happened to the press.

        While another combo runs, a press in its last `buffer_window` ms is QUEUED to start the
        moment it ends (a newer press replaces an older one); an earlier press is DROPPED.
        """
        combo = self.combos[name]
        with self._lock:
            if self._busy:
                if self._ends_at - time.perf_counter() > self._timing.buffer_window / 1000:
                    return DROPPED
                self._buffered = combo
                return QUEUED
            self._busy = True
            self._ends_at = math.inf  # until the combo starts and its length is known
            self._abort.clear()
            self._queue.put(combo)
        return STARTED

    def abort(self) -> None:
        """Stop the running combo and forget any press waiting to run after it."""
        with self._lock:
            self._buffered = None
            self._abort.set()

    def note_facing(self, facing: str) -> None:
        """She pressed ← or → herself."""
        with self._lock:
            self.facing = facing
            self._her_turn_at = time.perf_counter_ns()

    def toggle_slow(self) -> bool:
        self.slow = not self.slow
        self._display.done("MapleHelper", f"Slow mode {'ON' if self.slow else 'OFF'}", [], 1500)
        return self.slow

    def expected_ms(self, combo: Combo) -> int:
        """How long the combo takes with the current timings and slow mode."""
        dry = _DryRun(self._keys, self._timing, self.facing, self.slow)
        combo.script(dry)
        return dry.elapsed_ms

    def run(self, combo: Combo) -> None:
        """Run one combo to the end (or until aborted) on the calling thread."""
        self._abort.clear()
        self._run(combo)

    def _run(self, combo: Combo) -> None:
        for key in (self._keys.left, self._keys.right):
            self._send(key, False)  # start clean even if she's holding an arrow
        ctx = Ctx(self._send, self._keys, self._timing, self.facing, self.slow, self._abort, self._display, combo.title)
        ending = "Error, see console"
        try:
            ends_at = time.perf_counter() + self.expected_ms(combo) / 1000
            with self._lock:
                self._ends_at = ends_at
            combo.script(ctx)
            ending = "Done ✓"
        except Aborted:
            ending = "Stopped"
        finally:
            ctx.release_all()
            with self._lock:
                # Whoever turned her last decides her facing: the combo, or her own arrow press during it.
                if ctx.turned_at > self._her_turn_at:
                    self.facing = ctx.facing
            self._display.done(combo.title, ending, ctx.recipe, self._timing.recap)

    def _work(self) -> None:
        while (combo := self._queue.get()) is not None:
            try:
                self._run(combo)
            except Exception:
                log.exception("combo %s failed", combo.name)
            with self._lock:
                if self._buffered:
                    # Clear the stop flag here, under the lock, so an Esc that arrives from now on
                    # still stops the buffered combo.
                    self._abort.clear()
                    self._ends_at = math.inf
                    self._queue.put(self._buffered)
                    self._buffered = None
                else:
                    self._busy = False
