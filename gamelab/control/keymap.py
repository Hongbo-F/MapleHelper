"""Her keys as the game has them: followed off the game's KEY BINDINGS window, remembered per character, and handed to
the elevated player for the Lie Detector's allowance.

The user, 2026-10-28 (~16:00), asked whether her moves' keys moved in the game should stop the hunt or be followed:
"Follow them"; and whether the binding read off the window should be kept: "Remember it". Until then the hunt pressed
config.toml's [keys] whatever the game had, and the key check (behavior.keycheck) followed her regular attack alone,
stopping the hunt for a moved jump or arrow.

- `follow`: every key of hers that's bound (config.toml's names: GameKeys) to where the window read shows its function
  (FUNCTION_OF: ATTACK, JUMP, the MOVE arrows, PICK UP, SIT; her Lucky Seven's, pet food's and chair's pictures,
  perception.keybindings.PICTURES): on her key still, kept; on another she can press, that key; where the window can't
  tell (her key's cell illegible or not drawn, a picture there the reader doesn't know), kept, said. None she can press
  (on no key; only on keys never pressed for it, NEVER, or for her moves and pet food the alarm's ALARM_UNSAFE):
  a move of hers, her Lucky Seven, her pet food or her Magic Claw stops the hunt (`stop`, NEEDED: she can't hunt, get
  to safety or keep her pet fed without them); her regular attack is off for the run (behavior.keycheck, as before);
  the rest (her pick-up, chair, sit, the magician's buffs) are off, said. The game binds one function a key, so two of
  her actions can't land on one key; one kept where the window can't tell that another lands on is off, said.
- `remember` / `at_start`: the binding read, with when and off what frame, saved per character
  (lab/keybindings-<character>.json, paths.KEYBINDINGS) at every read of the window, and a command's keys at its start
  (control.timeline.her_setup) are those, not config.toml's [keys] (which stay the first default only; a remembered
  file that doesn't hold keys as they can be is left out, said; `gamelab keys --forget` drops it); the log says which.
- `publish`: her moves' and pet food's keys in use written for the player (paths.ALARM_KEYS, control.alarm.AlarmKeys):
  at a command's start and whenever the hunt follows them, so what goes with the Lie Detector's alarm up is her moves
  and her pet food by the keys they're on now, without the player started again (its UAC click).
"""

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field, fields

from maplehelper.config import GameKeys
from maplehelper.keys import KEY_CODES
from maplehelper.keys import parse as parse_key

from .. import paths
from ..perception import keybindings as kb
from .alarm import ALARM_FIELDS, ALARM_UNSAFE, alarm_keys_problem

# What the window shows on each key of hers (perception.keybindings.FUNCTIONS' names, PICTURES' for her pictures)
FUNCTION_OF = {"attack": "ATTACK", "jump": "JUMP", "left": "MOVE LEFT", "right": "MOVE RIGHT", "up": "MOVE UP",
               "down": "MOVE DOWN", "pick_up": "PICK UP", "sit": "SIT",
               **{f: name for f, (name, _) in kb.PICTURES.items()}}
MOVES = ("left", "right", "jump", "up", "down")
# On no key she can press, the hunt stops: her moves (her trips, her retreat from the Lie Detector), her Lucky Seven
# (the attack choice's skill: her casts would go out on a key that does something else, or nothing), her pet food (her
# pet drinks her potions: unfed it stops), her Magic Claw (the magician's attack)
NEEDED = (*MOVES, "lucky_seven", "pet_food", "magic_claw")
# Keys none of hers is followed onto: Esc (maplehelper's abort, and the game's MENU), the backslash (SET KEY: what opens
# the window) and maplehelper's hotkeys (config.toml [hotkeys] and [control]: hidden from the game while it runs); `never`
# adds config.toml's own
NEVER = ("esc", "\\", "f1", "f2", "f3", "f4", "f12")
SOURCE_CONFIG = "config.toml [keys]"


def never() -> tuple[str, ...]:
    """NEVER and config.toml's [hotkeys] and [control] keys (as far as it can be read)."""
    out = list(NEVER)
    try:
        from maplehelper import config
        from maplehelper.combos import COMBOS
        cfg = config.load(paths.CONFIG, (c.name for c in COMBOS))
        out += [k.name for k in cfg.hotkeys.values()] + [cfg.abort.name, cfg.slow_mode.name]
    except Exception:
        pass
    return tuple(dict.fromkeys(out))


def names(keys: GameKeys) -> dict[str, str | None]:
    """Her keys by field: each key's config.toml name, None where it isn't bound."""
    return {f.name: getattr(getattr(keys, f.name), "name", None) for f in fields(GameKeys)}


def keys_of(table: dict, base: GameKeys) -> GameKeys:
    """`base` with `table`'s fields (config.toml names, None: not bound) in place of its own."""
    return dataclasses.replace(base, **{f: None if k is None else parse_key(k)
                                        for f, k in table.items() if f in {x.name for x in fields(GameKeys)}})


def changes(a: GameKeys, b: GameKeys) -> list[tuple[str, str | None, str | None]]:
    """Her keys that differ from `a` to `b`: (field, a's name, b's name)."""
    na, nb = names(a), names(b)
    return [(f, na[f], nb[f]) for f in na if na[f] != nb[f]]


def changes_text(diff, fields_only: tuple[str, ...] | None = None) -> str:
    """Changes as "JUMP ALT -> C, PET FOOD U -> I" (a field to no key: "-> NONE")."""
    return ", ".join(f"{f.replace('_', ' ').upper()} {(was or 'none').upper()} -> {(now or 'none').upper()}"
                     for f, was, now in diff if fields_only is None or f in fields_only)


@dataclass
class Followed:
    """What `follow` made of a window read: her keys as it has them (`keys`), those moved (`moved`: field -> (was, now)),
    those on no key she can press (`off`: field -> where the window has it, in words), why the hunt stops (`stop`: a
    NEEDED one off; None), those the window doesn't tell (`unknown`: field -> why; kept as they were), and lines to say."""
    keys: GameKeys
    moved: dict[str, tuple[str, str]] = field(default_factory=dict)
    off: dict[str, str] = field(default_factory=dict)
    stop: str | None = None
    unknown: dict[str, str] = field(default_factory=dict)
    said: list[str] = field(default_factory=list)

    def text(self) -> str:
        """Her keys moved, in words ("JUMP ALT -> C, ...")."""
        return changes_text([(f, a, b) for f, (a, b) in self.moved.items()])


def _canon(key: str | None) -> str | None:
    return None if key is None else kb.SAME_KEY.get(key, key)


def follow(keys: GameKeys, fn: dict[str, str | None], illegible=(), never_keys: tuple[str, ...] | None = None,
           unsafe: tuple[str, ...] = ALARM_UNSAFE) -> Followed:
    """Her keys `keys` followed to where window map `fn` (perception.keybindings.Reader.functions: key -> its function's
    name, a picture's name, kb.PICTURE for one not known, None for nothing) has their functions (see the module's doc).
    `illegible`: the read's keys whose cells it couldn't read; `never_keys` (NEVER and config.toml's, by default) and,
    for her moves and pet food, `unsafe` (the alarm's): keys none is followed onto."""
    never_keys = never() if never_keys is None else never_keys
    out = Followed(keys)
    table = names(keys)
    new = dict(table)
    gone = set()  # keys a function of hers moved onto: one kept on them, where the window can't tell, does that now
    for f, key in table.items():
        if key is None:
            continue
        name = FUNCTION_OF.get(f)
        mine = _canon(key)
        shown = fn.get(mine, "?") if mine in kb.KEYS else "?"
        if name is None:  # (her potions: their pictures aren't known here)
            out.unknown[f] = "not read off the window"
            continue
        on = [k for k in kb.KEYS if fn.get(k) == name]
        if mine in on:
            continue
        bad = set(never_keys) | (set(unsafe) if f in ALARM_FIELDS else set())
        ok = [k for k in on if k not in bad]
        if ok:
            new[f] = ok[0]
            gone.add(ok[0])
            out.moved[f] = (key, ok[0])
            continue
        if not on and (mine in illegible or shown == "?" or (f in kb.PICTURES and shown == kb.PICTURE)):
            out.unknown[f] = ("its key not legible" if mine in illegible else "its key not drawn in the window"
                              if shown == "?" else "a picture on its key the window's reader doesn't know")
            continue
        where = (f"on {', '.join(k.upper() for k in on)} (not a key to press for it)" if on else "on no key")
        out.off[f] = where
        if f in NEEDED and out.stop is None:
            out.stop = (f"her {f.replace('_', ' ')} key {key.upper()} shows "
                        + ("nothing" if shown is None else shown if shown not in ("?", kb.PICTURE) else "a picture")
                        + f" in the game's Key Bindings, and {name} is {where}")
        if f != "attack":  # (GameKeys has an attack key always: keycheck turns the regular attack off for the run)
            new[f] = None
    for f in list(out.unknown):
        if table[f] is not None and _canon(table[f]) in gone and f not in out.moved:
            out.off[f] = f"its key {table[f].upper()} now another of hers"
            out.said.append(f"{f.replace('_', ' ')} {table[f].upper()} not pressed any more: "
                            f"{_moved_onto(out, table[f])} is on it now")
            del out.unknown[f]
            new[f] = None
    for f, (was, now) in out.moved.items():
        out.said.append(f"{f.replace('_', ' ')} on {now.upper()}, not {was.upper()}")
    for f, where in out.off.items():
        if f not in NEEDED and f != "attack" and "another of hers" not in where:
            out.said.append(f"{f.replace('_', ' ')} {where}: not pressed any more")
    out.keys = keys_of(new, keys)
    return out


def _moved_onto(out: Followed, key: str) -> str:
    return next((f.replace("_", " ") for f, (_, now) in out.moved.items() if _canon(now) == _canon(key)), "another")


# --- remembered -----------------------------------------------------------------------------------------------------

def path_of(character: str | None = None):
    """Character `character`'s remembered binding (None: the one this process plays)."""
    return paths.LAB / paths.KEYBINDINGS.format(character or _who())


def _who() -> str:
    try:
        from .. import character
        return character.active().id
    except Exception:
        return "thief"


def remember(keys: GameKeys, character: str | None = None, *, config_keys: GameKeys | None = None, frame=None,
             preset: int | None = None, functions: dict | None = None, attack_off: str | None = None,
             why: str = "") -> object:
    """Her keys as the window was read (`keys`, followed), saved for her next runs: when, the frame it was read off
    (`frame`: its saved file), the preset, the whole map (`functions`), config.toml's keys then, and whether the game
    had Attack on no key (`attack_off`: why; the next run starts with the regular attack off). Written whole (a file
    beside it, then replaced). Returns its path; None if it couldn't be written (said by the caller's log)."""
    path = path_of(character)
    data = {"character": character or _who(), "read_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frame": None if frame is None else _in_lab(frame), "preset": preset, "why": why,
            "keys": names(keys), "attack_off": attack_off,
            "config": None if config_keys is None else names(config_keys), "functions": functions or {}}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        return None
    return path


def _in_lab(frame) -> str:
    """A frame's file as the remembered binding names it: under lab/ by its path there ("shots/keybindings_....png"),
    else as it is."""
    from pathlib import Path
    try:
        return Path(frame).resolve().relative_to(Path(paths.LAB).resolve()).as_posix()
    except (ValueError, OSError):
        return str(frame)


def problem(table: dict, never_keys: tuple[str, ...] | None = None) -> str | None:
    """Why a remembered binding's keys (`table`) aren't keys as they can be (None: they are): a field GameKeys hasn't,
    a name config.toml can't name, her arrows or jump on none, two of hers on one key, one of NEVER, or her moves' and
    pet food's not as the alarm's allowance takes them (alarm.alarm_keys_problem)."""
    if not isinstance(table, dict):
        return "no keys"
    known = {f.name for f in fields(GameKeys)}
    if extra := set(table) - known:
        return f"unknown fields {sorted(extra)}"
    for f, k in table.items():
        if k is not None and (not isinstance(k, str) or k not in KEY_CODES):
            return f"{f} {k!r}: not a key config.toml can name"
    if any(table.get(f) is None for f in ("left", "right", "jump", "attack")):
        return "her arrows, jump or attack on no key"
    bound = [_canon(k) for k in table.values() if k is not None]
    if len(set(bound)) != len(bound):
        return "two of hers on one key"
    if bad := sorted(k for k in bound if k in (never() if never_keys is None else never_keys)):
        return f"on {', '.join(bad)}: not a key to press for her"
    return alarm_keys_problem({f: table.get(f) for f in ALARM_FIELDS})


@dataclass
class Start:
    """A command's keys at its start (`at_start`): them, where they're from (in words, for the log), whether the
    remembered binding has the game's Attack on no key (why; None), and the remembered file's data (None: none used)."""
    keys: GameKeys
    source: str
    attack_off: str | None = None
    data: dict | None = None


def at_start(config_keys: GameKeys, character: str | None = None) -> Start:
    """Her keys for a command's start: the remembered binding's (`remember`) over config.toml's, else config.toml's
    [keys] (no file, or one left out: said in `source`)."""
    path = path_of(character)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Start(config_keys, SOURCE_CONFIG)
    except (OSError, ValueError) as e:
        return Start(config_keys, f"{SOURCE_CONFIG} ({path.name} not read: {type(e).__name__})")
    table = data.get("keys") if isinstance(data, dict) else None
    if (why := problem(table)) is not None:
        return Start(config_keys, f"{SOURCE_CONFIG} ({path.name} left out: {why})")
    keys = keys_of(table, config_keys)
    source = (f"the key bindings read off the game's window at {data.get('read_at', '?')} ({path.name}"
              + (f", from {os.path.basename(str(data['frame']))}" if data.get("frame") else "") + ")")
    diff = changes(config_keys, keys)
    if diff:
        source += f"; not config.toml's: {changes_text(diff)}"
    if isinstance(data.get("config"), dict) and data["config"] != names(config_keys):
        source += ("; config.toml [keys] changed since it was read (its keys are the first default only: "
                   "`gamelab keys --forget` goes back to them)")
    off = data.get("attack_off")
    return Start(keys, source, off if isinstance(off, str) and off else None, data)


def forget(character: str | None = None) -> bool:
    """The remembered binding dropped (`gamelab keys --forget`): config.toml's [keys] from the next start. Whether
    there was one."""
    path = path_of(character)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def publish(keys: GameKeys, why: str = "", character: str | None = None) -> bool:
    """Her moves' and pet food's keys in use written for the player (paths.ALARM_KEYS: control.alarm.AlarmKeys), whole
    (a file beside it, then replaced). Whether it was written (not with keys the allowance can't take)."""
    table = {f: n for f, n in names(keys).items() if f in ALARM_FIELDS}
    if alarm_keys_problem(table) is not None:
        return False
    data = {"keys": table, "character": character or _who(), "at": time.strftime("%Y-%m-%d %H:%M:%S"), "why": why}
    try:
        paths.ALARM_KEYS.parent.mkdir(parents=True, exist_ok=True)
        tmp = paths.ALARM_KEYS.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, paths.ALARM_KEYS)
    except OSError:
        return False
    return True
