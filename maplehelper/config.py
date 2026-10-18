"""Loads config.toml: her in-game keys, the buttons that trigger combos, and timings."""

import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

from . import keys
from .keys import Key


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class GameKeys:
    """What each action is bound to in her in-game Key Config."""
    left: Key
    right: Key
    jump: Key
    attack: Key
    lucky_seven: Key | None = None  # None: throw stars with jump + attack instead


@dataclass(frozen=True)
class Timing:
    """All values in milliseconds."""
    key_press: int = 40     # how long a single tap holds the key down
    air_delay: int = 120    # after jumping, wait this long so the attack happens mid-air
    throw_gap: int = 650    # time for one throw animation to finish before the next
    turn_tap: int = 50      # tap length that turns her around without walking far
    walk_away: int = 500    # how long "Back Off" walks away
    hop_hold: int = 350     # how long the away arrow stays held during "Hop Back"
    slow_pause: int = 900   # slow mode: pause before each step so she can read it
    recap: int = 3000       # how long the full recipe stays on screen after a combo


@dataclass(frozen=True)
class Config:
    game_window_title: str
    keys: GameKeys
    hotkeys: dict[str, Key]  # combo name -> button she presses for it
    abort: Key
    slow_mode: Key
    timing: Timing


def load(path: str | Path, combo_names: Iterable[str]) -> Config:
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None
    return parse(data, combo_names)


def parse(data: dict, combo_names: Iterable[str]) -> Config:
    combo_names = set(combo_names)
    title = data.get("game_window_title", "MapleStory")
    if not isinstance(title, str) or not title:
        raise ConfigError("game_window_title must be non-empty text")

    game_keys = _game_keys(_table(data, "keys"))

    hotkeys = {name: _key(f"hotkeys.{name}", value) for name, value in _table(data, "hotkeys").items()}
    if unknown := set(hotkeys) - combo_names:
        raise ConfigError(f"[hotkeys] has unknown combo(s) {sorted(unknown)}; available: {sorted(combo_names)}")

    control = _table(data, "control")
    abort = _key("control.abort", control.get("abort", "esc"))
    slow_mode = _key("control.slow_mode", control.get("slow_mode", "f12"))

    _check_collisions(game_keys, hotkeys, abort, slow_mode)
    return Config(title, game_keys, hotkeys, abort, slow_mode, _timing(_table(data, "timing")))


def _table(data: dict, name: str) -> dict:
    table = data.get(name, {})
    if not isinstance(table, dict):
        raise ConfigError(f"[{name}] must be a table")
    return table


def _key(where: str, value) -> Key:
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a key name in quotes, got {value!r}")
    try:
        return keys.parse(value)
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from None


def _game_keys(table: dict) -> GameKeys:
    known = {f.name for f in fields(GameKeys)}
    if unknown := set(table) - known:
        raise ConfigError(f"[keys] has unknown action(s) {sorted(unknown)}; known: {sorted(known)}")
    required = ("left", "right", "jump", "attack")
    if missing := [name for name in required if not table.get(name)]:
        raise ConfigError(f"[keys] is missing {missing}")
    # An empty string means "not bound / not learned yet".
    return GameKeys(**{name: _key(f"keys.{name}", value) for name, value in table.items() if value != ""})


def _timing(table: dict) -> Timing:
    known = {f.name for f in fields(Timing)}
    if unknown := set(table) - known:
        raise ConfigError(f"[timing] has unknown setting(s) {sorted(unknown)}; known: {sorted(known)}")
    for name, value in table.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"timing.{name} must be a whole number of milliseconds >= 0, got {value!r}")
    return Timing(**table)


def _check_collisions(game_keys: GameKeys, hotkeys: dict[str, Key], abort: Key, slow_mode: Key) -> None:
    # Combo buttons and the slow-mode button are hidden from the game, so they must not be keys she plays with.
    swallowed = {f"hotkeys.{name}": key for name, key in hotkeys.items()}
    swallowed["control.slow_mode"] = slow_mode
    game = {f"keys.{f.name}": getattr(game_keys, f.name) for f in fields(GameKeys)}
    taken = {key.name: where for where, key in game.items() if key}
    taken.setdefault(abort.name, "control.abort")
    for where, key in swallowed.items():
        if key.name in taken:
            raise ConfigError(f"{where} = {key.name!r} is already used by {taken[key.name]}")
        taken[key.name] = where
