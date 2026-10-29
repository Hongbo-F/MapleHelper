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
# ... and her bindings from 2026-10-23, on West Rocky Mountain II (config.toml then): attack ctrl, Lucky Seven shift,
# no potion keys (her pet drinks them), pet food u. The simulator plays her with these there (sim.scenarios.HER).
SIM_KEYS_PET = dataclasses.replace(SIM_KEYS, attack=parse_key("ctrl"), lucky_seven=parse_key("shift"), hp_potion=None,
                                   mp_potion=None, pet_food=parse_key("u"))


def sim_setup() -> tuple[GameKeys, Timing]:
    """The simulator's keys (SIM_KEYS) and config.toml's timing."""
    return SIM_KEYS, her_setup()[2]


def her_setup(regular_attack: bool = False, remembered: bool = True) -> tuple[config.Config, GameKeys, Timing]:
    """config.toml, her keys the binding last read off the game's KEY BINDINGS window where there's one
    (`remembered`: control.keymap.at_start; config.toml's [keys] otherwise, and `cfg.keys` always), optionally with
    Lucky Seven off so throws use the plain attack key (no MP). Another character than the thief (gamelab.character:
    the one this process plays) has her profile's keys for config.toml's [keys] (`cfg.keys` hers too): the magician's
    Magic Claw on her own key."""
    cfg = config.load(CONFIG, (c.name for c in COMBOS))
    if (mine := _profile_keys()) is not None:
        cfg = dataclasses.replace(cfg, keys=mine)
    keys = cfg.keys
    if remembered:
        from . import keymap
        keys = keymap.at_start(cfg.keys).keys
    keys = dataclasses.replace(keys, lucky_seven=None, magic_claw=None) if regular_attack else keys
    return cfg, keys, cfg.timing


def _profile_keys() -> GameKeys | None:
    """The keys of the character this process plays where she isn't the thief (her profile's: gamelab.character), else
    None (the thief's are config.toml's flat [keys], as before)."""
    from .. import character
    c = character.active()
    return None if c.klass == config.THIEF or c.keys is None else c.keys


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
