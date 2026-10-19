"""Turn MapleHelper's own combo code into exact key timelines, so tests play what the helper plays."""

import dataclasses
import threading

from maplehelper import config
from maplehelper.combos import COMBOS
from maplehelper.config import GameKeys, Timing
from maplehelper.engine import Ctx, _Silent

from .paths import CONFIG


def her_setup(regular_attack: bool = False) -> tuple[config.Config, GameKeys, Timing]:
    """config.toml, optionally with Lucky Seven off so throws use the plain attack key (no MP)."""
    cfg = config.load(CONFIG, (c.name for c in COMBOS))
    keys = dataclasses.replace(cfg.keys, lucky_seven=None) if regular_attack else cfg.keys
    return cfg, keys, cfg.timing


class _Clock(Ctx):
    """Runs a combo on a virtual clock, recording the key events it sends."""

    def __init__(self, keys: GameKeys, timing: Timing, facing: str, start_ms: int):
        self.events: list[list] = []
        self.now = start_ms
        super().__init__(self._record, keys, timing, facing, False, threading.Event(), _Silent(), "")

    def _record(self, key, down):
        self.events.append([round(self.now), key.name, "down" if down else "up"])

    def wait(self, ms: int) -> None:
        self.now += ms


def combo_events(name: str, facing: str, start_ms: int, keys: GameKeys, timing: Timing) -> tuple[list, int]:
    """Key events for one combo starting at start_ms, and the time it ends."""
    clock = _Clock(keys, timing, facing, start_ms)
    clock._send(keys.left, False)  # the runner's clean start
    clock._send(keys.right, False)
    next(c for c in COMBOS if c.name == name).script(clock)
    clock.release_all()
    return clock.events, clock.now


def action_events(fn, args, facing: str, keys: GameKeys, timing: Timing) -> tuple[list, int]:
    """Key events for one maplehelper.actions move (fn(ctx, *args)) starting at 0, and its end time."""
    clock = _Clock(keys, timing, facing, 0)
    fn(clock, *args)
    clock.release_all()
    return clock.events, clock.now


def chain(name: str, facing: str, count: int, keys: GameKeys, timing: Timing) -> tuple[list, list[int], int]:
    """`count` copies of a combo back to back, the way the helper runs a queued press."""
    events, starts, t = [], [], 0
    for _ in range(count):
        starts.append(t)
        ev, t = combo_events(name, facing, t, keys, timing)
        events += ev
    return events, starts, t
