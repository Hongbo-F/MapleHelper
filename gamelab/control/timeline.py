"""Turn MapleHelper's own combo code into exact key timelines, so tests play what the helper plays.
Also loads her setup (config.toml: keys and timing) for the commands and the simulator.
"""

import dataclasses
import threading

from maplehelper import config
from maplehelper.combos import COMBOS
from maplehelper.config import GameKeys, Timing
from maplehelper.engine import Ctx, _Silent
from maplehelper.keys import parse as parse_key

from ..paths import CONFIG


# The keys the simulator plays with: her bindings up to 2026-10-22 (attack f, Lucky Seven g, potions r and t, no
# pet), kept fixed so that a change to her live bindings in config.toml doesn't change what the simulator records
# (the golden traces name the keys). They're names only there: the simulator acts on what each is bound to.
SIM_KEYS = GameKeys(left=parse_key("left"), right=parse_key("right"), jump=parse_key("alt"), attack=parse_key("f"),
                    lucky_seven=parse_key("g"), up=parse_key("up"), down=parse_key("down"), pick_up=parse_key("z"),
                    chair=parse_key("d"), sit=parse_key("x"), hp_potion=parse_key("r"), mp_potion=parse_key("t"))


def sim_setup() -> tuple[GameKeys, Timing]:
    """The simulator's keys (SIM_KEYS) and config.toml's timing."""
    return SIM_KEYS, her_setup()[2]


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
        super().__init__(self._record, keys, timing, facing, slow=False, abort=threading.Event(),
                         display=_Silent(), title="")

    def _record(self, key, down):
        self.events.append([round(self.now), key.name, "down" if down else "up"])

    def wait(self, ms: int) -> None:
        self.now += ms


def combo_events(name: str, facing: str, start_ms: int, keys: GameKeys, timing: Timing) -> tuple[list, int]:
    """Key events for one combo starting at start_ms, and the time it ends."""
    clock = _Clock(keys, timing, facing, start_ms)
    clock._record(keys.left, False)  # the runner's clean start: both arrows released
    clock._record(keys.right, False)
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
