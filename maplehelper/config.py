"""Loads config.toml: her in-game keys, the buttons that trigger combos, timings, and her stats. The flat tables
([keys], [hotkeys], [timing], [character]) are the thief's; another character of hers (the magician, from 2026-10-27)
has tables of her own under [characters] (`characters`; gamelab.character makes her profile of them)."""

import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field, fields, replace
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
    # Used by maplehelper.actions (climbing, looting, resting); None = not bound.
    up: Key | None = None
    down: Key | None = None
    pick_up: Key | None = None
    chair: Key | None = None
    sit: Key | None = None
    hp_potion: Key | None = None
    mp_potion: Key | None = None
    pet_food: Key | None = None  # feeds her pet (which drinks her potions for her)
    # A magician's skills ([characters.<id>.keys], class "magician"; never the thief's: CLASS_SKILLS). Nothing presses
    # them yet: the attack layer takes them with her attacks (gamelab README: Characters).
    magic_claw: Key | None = None
    magic_guard: Key | None = None
    magic_armor: Key | None = None
    energy_bolt: Key | None = None


# Each class's skill keys (GameKeys' fields only one class binds): the flat [keys] are the thief's, so a magician's
# skill there is an error, and Lucky Seven under a magician's [characters.<id>.keys].
CLASS_SKILLS = {"thief": ("lucky_seven",), "magician": ("magic_claw", "magic_guard", "magic_armor", "energy_bolt")}
# Each class's skill levels a [characters.<id>.skills] table may give (the thief's Lucky Seven is [character]'s).
CLASS_SKILL_LEVELS = {"thief": (), "magician": ("magic_claw", "energy_bolt", "magic_guard", "magic_armor",
                                                "mp_recovery", "max_mp")}
CLASS_BUFFS = {"thief": (), "magician": ("magic_guard", "magic_armor")}
PET_SETTINGS = ("hp_warning", "mp_warning", "hp_heal", "mp_heal")  # her pet's auto-potion: shares of HP/MP, HP/MP a potion
THIEF = "thief"  # the character the flat tables are


@dataclass(frozen=True)
class Timing:
    """All values in milliseconds. Defaults were measured in game with her claw's regular attack:
    a jump is airborne ~530 ms, an attack throws its star ~530 ms after the key, and she can't
    jump or attack again until ~730 ms after the attack key (presses before that are ignored)."""
    key_press: int = 40     # how long a single tap holds the key down
    run_up: int = 100       # Jump-Back Shot: hold the away arrow this long before jumping; shorter
                            # sometimes gives a straight-up jump
    air_delay: int = 60     # after the jump key: keep holding the away arrow this long (Jump-Back Shot),
                            # or wait this long before the mid-air attack (Jump-Throw)
    turn_lead: int = 50     # Jump-Back Shot: hold the turn arrow this long before attacking, and keep it
                            # held through the attack; a quick tap right after takeoff is sometimes ignored
    throw_gap: int = 750    # after releasing the attack key, wait this long before the next action
                            # (key_press + throw_gap must clear the ~730 ms attack lock)
    turn_tap: int = 50      # tap length that turns her around on the ground without walking far
    walk_away: int = 500    # how long "Back Off" walks away
    buffer_window: int = 300  # a combo button pressed this close to the end of a running combo is
                              # queued to run right after it; pressed earlier, it's ignored
    slow_pause: int = 900   # slow mode: pause before each step so she can read it
    recap: int = 3000       # how long the full recipe stays on screen after a combo


@dataclass(frozen=True)
class Character:
    """Her stats that set how the game moves her, in % as the stat window shows them. The game walks her
    at 125 units/s x SPEED/100 and jumps her at 555 units/s x JUMP/100 (its Physics.img), counting
    SPEED from 80 to 140 and JUMP from 80 to 123. Gear and Haste raise them. gamelab plans her moves
    with them (gamelab.motion.physics). And her MP and her Lucky Seven's skill level, which what a cast costs
    and how many she has left come from (gamelab.attack.choice: Lucky Seven or the regular attack).

    Not here: her HP, her level and her DEF (W.DEF), what a monster's touch takes is measured against. They change
    as she plays (a level, new gear), and a number written here went stale (config.toml's hp = 774 at level 25 while
    the HUD said 840 at level 28, 2026-10-28), so perception reads them (gamelab.behavior.bootstrap: her max HP off the
    HUD's HP number, her level off its "LV." digits, her DEF off her stat window when it's read, else fitted to her
    touches). The user, 2026-10-28: "I thought hp and level will come from perception, please remove them completely
    from config.toml". A table that gives them is refused (PERCEIVED), so none lingers there unread."""
    speed: float = 119.2    # measured: she walks 148-151 units/s (149 = 125 x 1.192)
    jump: float = 100.0     # measured: her jumps rise 77-78 units (77 at 100%)
    mp: float = 623.0       # her MP when full (her MP bar's number, level 25, 2026-10-25; 601 at level 24)
    lucky_seven: float = 16.0  # her Lucky Seven's skill level: 14 MP a cast in the game data (at 16 and 17); her MP
                               # bar fell 2.345 px of 104 a cast (+- 0.03) over 364 one-cast stretches of the live hunts
                               # of 2026-10-25 23:43 to 01:03, 14.05 MP at 623 (13 at 14 and 15 would be 2.17 px, 15 at
                               # 18 and 19 2.50: D:/MapleHelper-scratch/hunt-ab/a/scripts/castcost_all.py); 11 at level
                               # 10, 11.4 measured on 2026-10-24
    # What her bars are reckoned with, as the stack has them for her (nothing reads these three yet: each module has
    # its own copy, the thief's, until her profile is read there: gamelab README, Characters).
    avoid: float = 27.0        # sim.world.EVADE_ACC: fit to 13 touches hopping over monsters (not her stat window's)
    climb_speed: float = 117.0  # motion.physics.CLIMB_SPEED: units/s on a rope, measured
    mp_regen: float = 0.32     # sim.world.MP_REGEN: MP a second, idle


@dataclass(frozen=True)
class CharacterConfig:
    """One of [characters]' tables: another character than the thief, her class ("magician"), the lines her bottom bar
    shows (`job` above `name`), and what of her the file gives: her Key Config, timing, stats (a partial table: the
    rest is her class's, gamelab.classes), skill levels, her pet's auto-potion settings and the buffs kept up. The
    thief's table ([characters.thief]) gives only her lines: her keys, timing and stats are the flat tables."""
    id: str
    klass: str
    name: str | None = None
    job: str | None = None
    keys: GameKeys | None = None      # None: not given
    timing: Timing | None = None
    stats: dict[str, float] = field(default_factory=dict)   # Character's fields she gives (her stat window)
    skills: dict[str, int] = field(default_factory=dict)    # skill -> level (CLASS_SKILL_LEVELS)
    pet: dict[str, float] = field(default_factory=dict)     # PET_SETTINGS
    buffs: dict[str, bool] = field(default_factory=dict)    # buff -> kept up (CLASS_BUFFS)
    broken: str | None = None  # why her table doesn't parse (kept apart: the thief, the player and every other
                               # character's commands go on without her; gamelab.character.load raises it for her)


@dataclass(frozen=True)
class Config:
    game_window_title: str
    keys: GameKeys
    hotkeys: dict[str, Key]  # combo name -> button she presses for it
    abort: Key
    slow_mode: Key
    timing: Timing
    character: Character = field(default_factory=Character)
    characters: dict[str, CharacterConfig] = field(default_factory=dict)  # [characters.<id>] (the thief's too, if given)
    default_character: str = THIEF  # [characters] default: who plays when nothing names her (gamelab.character)


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
    default, others = _characters(_table(data, "characters"))
    for c in list(others.values()):  # each other character's keys against the control keys (the hotkeys are the
        if c.keys is not None and c.broken is None:  # thief's): a clash is hers, kept apart as any of her mistakes
            try:
                _check_collisions(c.keys, {}, abort, slow_mode, f"characters.{c.id}.keys")
            except ConfigError as e:
                others[c.id] = replace(c, broken=str(e))
    return Config(title, game_keys, hotkeys, abort, slow_mode, _timing(_table(data, "timing")),
                  _character(_table(data, "character")), others, default)


def character(path: str | Path) -> Character:
    """Just her stats ([character]), for gamelab's physics, which needs them before anything else is
    loaded: the defaults if the file isn't there."""
    data = _read(path)
    return Character() if data is None else _character(_table(data, "character"))


def characters(path: str | Path) -> tuple[str, dict[str, CharacterConfig]]:
    """Just [characters] (its `default`, and each character's table), for gamelab.character, which needs it before
    anything else is loaded, as physics does [character]: ("thief", {}) if the file or the table isn't there."""
    data = _read(path)
    return (THIEF, {}) if data is None else _characters(_table(data, "characters"))


def _read(path: str | Path) -> dict | None:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None


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


def _game_keys(table: dict, klass: str = THIEF, where: str = "keys") -> GameKeys:
    """A Key Config (the flat [keys]: the thief's; or a [characters.<id>.keys] of `klass`): the actions any class
    binds and the class's own skills (CLASS_SKILLS), not another class's."""
    others = {k for c, skills in CLASS_SKILLS.items() if c != klass for k in skills} - set(CLASS_SKILLS[klass])
    known = {f.name for f in fields(GameKeys)} - others
    if unknown := set(table) - known - others:
        raise ConfigError(f"[{where}] has unknown action(s) {sorted(unknown)}; known: {sorted(known)}")
    if theirs := sorted(set(table) & others):
        mine = "a magician's go in [characters.<id>.keys]" if klass == THIEF else f"her skills: {list(CLASS_SKILLS[klass])}"
        raise ConfigError(f"[{where}] has {theirs}, not a {klass}'s skill(s) ({mine})")
    required = ("left", "right", "jump", "attack")
    if missing := [name for name in required if not table.get(name)]:
        raise ConfigError(f"[{where}] is missing {missing}")
    # An empty string means "not bound / not learned yet".
    return GameKeys(**{name: _key(f"{where}.{name}", value) for name, value in table.items() if value != ""})


ID = re.compile(r"[a-z][a-z0-9_-]{0,31}")  # a character's id: what --character and `gamelab character use` name
LINES = ("name", "job")                    # her bottom bar's lines, given as text
SUBTABLES = ("keys", "timing", "stats", "skills", "pet", "buffs")


def _characters(table: dict) -> tuple[str, dict[str, CharacterConfig]]:
    """[characters]: `default` (who plays when nothing names her: "thief" unless given) and each character's table.
    A table's class is its `class`, or its id when that's a class's name ([characters.magician]). A mistake in another
    character's table is kept with her (CharacterConfig.broken), not raised: the thief's commands, and the player that
    every command's keys go through, never stop for another character's table, as they didn't before there were any;
    she can't play until it's put right (gamelab.character.load says why). The thief's own table and `default` are
    everyone's: a mistake there is raised."""
    default = table.get("default", THIEF)
    out = {}
    for cid, sub in table.items():
        if cid == "default":
            continue
        try:
            out[cid] = _character_table(cid, sub)
        except ConfigError as e:
            if cid == THIEF:
                raise
            klass = sub.get("class", cid if cid in CLASS_SKILLS else None) if isinstance(sub, dict) else None
            out[cid] = CharacterConfig(cid, klass if isinstance(klass, str) else "?", broken=str(e))
    if not isinstance(default, str) or default not in {THIEF, *out, *CLASS_SKILLS}:
        if default == "auto":  # (the design's: the profile whose name her bottom bar shows; nothing reads it yet)
            raise ConfigError("[characters] default = \"auto\" needs her bottom bar read, which isn't built yet: name "
                              "the character")
        raise ConfigError(f"[characters] default must name a character ({sorted({THIEF, *out})}), got {default!r}")
    return default, out


def _character_table(cid: str, sub) -> CharacterConfig:
    """One of [characters]' tables, [characters.<cid>], as config.toml has it."""
    where = f"characters.{cid}"
    if not isinstance(sub, dict):
        raise ConfigError(f"[characters] {cid} must be a table ([{where}]), or `default`")
    if not ID.fullmatch(cid):
        raise ConfigError(f"[{where}]: a character's id is lower-case letters, digits, - and _ (got {cid!r})")
    klass = sub.get("class", cid if cid in CLASS_SKILLS else None)
    if klass not in CLASS_SKILLS:
        raise ConfigError(f"[{where}] class must be one of {sorted(CLASS_SKILLS)}, got {klass!r}")
    if klass == THIEF and cid != THIEF:
        raise ConfigError(f"[{where}]: the thief is [characters.thief] and the flat tables, one of her")
    if cid == THIEF and klass != THIEF:
        raise ConfigError(f"[{where}] class must be \"thief\" (the flat tables are hers), got {klass!r}: another "
                          f"character's table goes under an id of her own ([characters.magician])")
    if unknown := set(sub) - {"class", *LINES, *SUBTABLES}:
        raise ConfigError(f"[{where}] has unknown setting(s) {sorted(unknown)}; known: "
                          f"{sorted({'class', *LINES, *SUBTABLES})}")
    for line in LINES:
        if line in sub and (not isinstance(sub[line], str) or not sub[line].strip()):
            raise ConfigError(f"{where}.{line} must be text, as her bottom bar shows it, got {sub[line]!r}")
    if cid == THIEF:
        if given := sorted(set(sub) & set(SUBTABLES)):
            raise ConfigError(f"[{where}] has {given}: the thief's keys, timing and stats are the flat [keys], "
                              f"[timing] and [character]")
        return CharacterConfig(cid, klass, sub.get("name"), sub.get("job"))
    tables = {name: _table(sub, name) for name in SUBTABLES}
    return CharacterConfig(
        cid, klass, sub.get("name"), sub.get("job"),
        keys=_game_keys(tables["keys"], klass, f"{where}.keys") if tables["keys"] else None,
        timing=_timing(tables["timing"]) if tables["timing"] else None,
        stats=_stats(tables["stats"], f"{where}.stats"),
        skills=_levels(tables["skills"], CLASS_SKILL_LEVELS[klass], f"{where}.skills"),
        pet=_pet(tables["pet"], f"{where}.pet"),
        buffs=_buffs(tables["buffs"], CLASS_BUFFS[klass], f"{where}.buffs"))


# Her stats perception reads, never config.toml's (Character's docstring): a table giving one is refused, saying so.
PERCEIVED = ("hp", "level", "defence")


def _perceived(table: dict, where: str) -> None:
    if given := [name for name in PERCEIVED if name in table]:
        raise ConfigError(f"[{where}] has {given}: her HP, level and DEF aren't read from config.toml (perception reads "
                          "her max HP and level off her HUD, her DEF off her stat window or her touches: "
                          "gamelab.behavior.bootstrap); delete the line(s)")


def _stats(table: dict, where: str) -> dict[str, float]:
    _perceived(table, where)
    known = {f.name for f in fields(Character)} - {"lucky_seven"}  # (a skill level: the thief's, in [character])
    if unknown := set(table) - known:
        raise ConfigError(f"[{where}] has unknown stat(s) {sorted(unknown)}; known: {sorted(known)}")
    for name, value in table.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(f"{where}.{name} must be a number above 0, got {value!r}")
    return {name: float(value) for name, value in table.items()}


def _levels(table: dict, known: tuple[str, ...], where: str) -> dict[str, int]:
    if unknown := set(table) - set(known):
        raise ConfigError(f"[{where}] has unknown skill(s) {sorted(unknown)}; known: {sorted(known)}")
    for name, value in table.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 30:
            raise ConfigError(f"{where}.{name} must be her skill's level (a whole number, 0-30), got {value!r}")
    return dict(table)


def _pet(table: dict, where: str) -> dict[str, float]:
    if unknown := set(table) - set(PET_SETTINGS):
        raise ConfigError(f"[{where}] has unknown setting(s) {sorted(unknown)}; known: {list(PET_SETTINGS)}")
    for name, value in table.items():
        share = name.endswith("_warning")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or (share and value >= 1):
            what = "a share of her bar (0.70 for 70%)" if share else "the HP or MP a potion gives"
            raise ConfigError(f"{where}.{name} must be {what}, got {value!r}")
    return {name: float(value) for name, value in table.items()}


def _buffs(table: dict, known: tuple[str, ...], where: str) -> dict[str, bool]:
    if unknown := set(table) - set(known):
        raise ConfigError(f"[{where}] has unknown buff(s) {sorted(unknown)}; known: {list(known)}")
    if bad := [n for n, v in table.items() if not isinstance(v, bool)]:
        raise ConfigError(f"{where}.{bad[0]} must be true or false (kept up or not)")
    return dict(table)


def _timing(table: dict) -> Timing:
    known = {f.name for f in fields(Timing)}
    if unknown := set(table) - known:
        raise ConfigError(f"[timing] has unknown setting(s) {sorted(unknown)}; known: {sorted(known)}")
    for name, value in table.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"timing.{name} must be a whole number of milliseconds >= 0, got {value!r}")
    return Timing(**table)


def _character(table: dict) -> Character:
    _perceived(table, "character")
    known = {f.name for f in fields(Character)}
    if unknown := set(table) - known:
        raise ConfigError(f"[character] has unknown stat(s) {sorted(unknown)}; known: {sorted(known)}")
    for name, value in table.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            what = f"her {name.upper()} stat in %" if name in ("speed", "jump") else f"her {name.upper()}"
            raise ConfigError(f"character.{name} must be {what} (a number above 0), got {value!r}")
    return Character(**{name: float(value) for name, value in table.items()})


def _check_collisions(game_keys: GameKeys, hotkeys: dict[str, Key], abort: Key, slow_mode: Key,
                      where: str = "keys") -> None:
    # Combo buttons and the slow-mode button are hidden from the game, so they must not be keys she plays with.
    swallowed = {f"hotkeys.{name}": key for name, key in hotkeys.items()}
    swallowed["control.slow_mode"] = slow_mode
    game = {f"{where}.{f.name}": getattr(game_keys, f.name) for f in fields(GameKeys)}
    taken = {key.name: where for where, key in game.items() if key}
    taken.setdefault(abort.name, "control.abort")
    for where, key in swallowed.items():
        if key.name in taken:
            raise ConfigError(f"{where} = {key.name!r} is already used by {taken[key.name]}")
        taken[key.name] = where
