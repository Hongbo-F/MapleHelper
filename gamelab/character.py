"""Who plays: the character a process plays as, and her profile (what the stack takes of her).

Until 2026-10-27 gamelab hunted with one character, the thief (BugBaby, a Rogue: Lucky Seven and her claw's stars).
That morning the user: "I have another character magician who will use magic claw to grind on this map ... I will
switch back and forth between the magician and the thief, so the code needs to be generalized". Her attack is
another kind (Magic Claw: the client's stepped search for an aimed spell, from 50 ahead of her to 300, nothing stopping
it, no star flying, from the ground only: D:/MapleHelper-scratch/mage/design.md and range/magic-claw-range.md), and so
are her stats, keys, ammunition (her MP), files and what perception reads of her.

One character per process (`active_id`, first match): `--character <id>` on `python -m gamelab` (it sets ENV before
any command loads gamelab.motion.physics, which works her walk and jumps out once, at import; child processes inherit
it), ENV, lab/character.json (`gamelab character use <id>`), config.toml's [characters] default, then the thief. With
physics' FIXED_STATS set (the tests, the golden traces, the harness) only ENV counts, the thief by default, and each
character's stats are her class's fixed copy (`fixed_stats`: the thief's config.Character(), as before). It's resolved
once a process (`active`, which physics' stats are), so what lets a command run, her physics and her files are one
character's. Until the game's bottom bar is read (the design's S0g), a command that looks at or acts in the game plays
another character only when ENV names her (`named`: gamelab.__main__); the two files' choice is the offline commands'.

The profile is data only: this module and gamelab.classes import nothing of gamelab but paths, so every layer may read
it (localization takes its `Shape`). The thief's values are today's module constants, kept where they are and read
there by her code as before; here they describe her (tests/test_character.py checks each against the constant it
names). The magician's, but for Magic Claw's search (the client's) and the game data's tables, are placeholders, each
marked, until the user's answers and the live measurements (gamelab README: Characters).
"""

import contextlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

from maplehelper import config

from .paths import CONFIG, LAB

ENV = "GAMELAB_CHARACTER"            # the character a process plays (--character sets it; child processes inherit it)
FIXED_STATS = "GAMELAB_FIXED_STATS"  # (motion.physics.FIXED_STATS: tests, goldens and the harness set it)
CHOSEN = LAB / "character.json"      # `gamelab character use <id>`: who plays when neither names one
DEFAULT = config.THIEF               # ... nor config.toml: the thief, as before


@dataclass(frozen=True)
class Box:
    """Where an attack picks, from her feet as she faces right (x ahead, y up; mirrored facing left)."""
    near: float  # the nearest x it covers (0: her middle; negative: behind her)
    far: float   # the farthest
    down: float  # how far under her feet
    up: float    # how far over them


@dataclass(frozen=True)
class Search:
    """A forward target search as the client runs it (RustMS ShotPath.h:311-339, search_steps and in_step; Combat.cpp:
    1843-1875, find_shot_target; meowdb's {aimHeight 28, divisor 4, step 20, starts: magic 50, ranged 65}): bands
    `step` wide from `start` ahead of her out to `range`, band k's window reaching (start + step k) // spread above and
    below `row` over her feet, mirrored facing left, overlap strict; the first band any body overlaps picks. Nothing
    behind her, nor anything whose far side is within `start` of her (D:/MapleHelper-scratch/mage/range/
    magic-claw-range.md, 1.2-1.3). Data: the attack layer picks by it (the design's S5; the thief's is attack.reach's
    own, which this describes)."""
    start: int       # the first band's near edge, ahead of her
    range: int       # the last band's far edge (a body's near side must be short of it)
    step: int = 20   # ShotPath.h:29 STEP
    spread: int = 4  # ShotPath.h:26 SPREAD_DIVISOR
    row: int = 28    # ShotPath.h:20 HEIGHT (attack.reach.SHOT_H)

    @property
    def bands(self) -> int:
        return math.ceil((self.range - self.start) / self.step)

    def window(self, k: int) -> tuple[int, int]:
        """Band k's window, (floor, roof) above her feet (a floor under 0 is under them)."""
        s = (self.start + self.step * k) // self.spread
        return self.row - s, self.row + s

    @property
    def floor(self) -> int:
        """The lowest any band's window reaches, above her feet: the last band's."""
        return self.window(self.bands - 1)[0]

    @property
    def roof(self) -> int:
        """... and the highest."""
        return self.window(self.bands - 1)[1]


@dataclass(frozen=True)
class Geometry:
    """What an attack picks and what stops it (the attack layer makes the functions of it: S1's aim). "star_search"
    is the thief's: attack.reach exactly (its bands, its line, the claw up close, its air search; `search` describes
    its bands). "search": the client's search alone (`search`: an aimed spell's, Magic Claw's). "melee": a box (a
    wand's swing)."""
    kind: str                       # "star_search" | "search" (Magic Claw) | "melee" (a wand's swing)
    search: Search | None = None    # star_search, search: the bands it picks in
    box: Box | None = None          # melee: the pick's area
    air: str | None = None          # from a jump: "search" (reach's), "same" (as on the ground, from her pose then),
                                    # None (never keyed in the air)
    line: str | None = None         # what stops it: "v83" (reach.LINE: walls and floors), None (nothing)
    point_blank: str | None = None  # "claw": reach.CLAW swipes up close instead (the thief's regular attack)
    order: str = "band"             # "band" (the first band any body overlaps, any in it alike) | "nearest" (feet to
                                    # feet)
    mob_count: int = 1


@dataclass(frozen=True)
class AttackSpec:
    """One of her attacks, as data (S3's attack.arsenal scales lo, hi and mp by her skill level off skills.json)."""
    id: str                         # "lucky_seven", "star", "magic_claw", "wand"
    key: str                        # its GameKeys field
    name: str                       # in logs and summaries
    verb: str                       # "throwing", "casting"
    skill_id: int | None            # None: the regular attack
    level: int | None
    hits: int                       # hits on its one monster: Lucky Seven's 2 stars, Magic Claw's attackCount 2
    lo: float | None                # damage a hit, before defence (None: not known)
    hi: float | None
    defence: str                    # "pdd" | "mdd": the monster's defence against it
    mp: float                       # MP a use
    ammo: int                       # stars a use
    lock_ms: int                    # key -> her next jump or attack is taken
    lock_extra_ms: tuple[int, int]  # margin past the lock before the next attack (on the ground, after a jump's)
    arrow_lock_ms: int              # key -> an arrow moves her
    release_ms: int                 # key -> the star leaves (the claw swings)
    damage_ms: int                  # key -> its damage counts (the server's: EXP, the bar)
    lands_ms: int                   # key -> the hit shows on the monster (a star's leaving, before its flight)
    speed: float | None             # a projectile's units/s (None: no flight)
    cycle_s: float                  # s between uses: the lock and its margin
    miss: float                     # the share that hits nothing
    hidden_s: float                 # s a monster hit stays under its hit's effect, unseen
    air: bool                       # usable from a jump
    geometry: Geometry


@dataclass(frozen=True)
class Shape:
    """What localization takes of her attacks (their union): which monsters a use of one went at, and how long
    after its key what it did can show. Passed into the trackers (MonsterTracker, Memory, Senses); THIEF_SHAPE is
    their constants (tracking, monsters, senses). The nearest it picks is signed (the thief's 10 behind her; Magic
    Claw's search never behind her, nor in its first 50 ahead), and up and down go apart (its windows widen upward:
    100 over her feet to 44 under them at its far end)."""
    near: float                         # units ahead of her middle a monster's middle it picks can be, at the least
                                        # (negative: behind her) ...
    ahead: float                        # ... and at the most
    up: float                           # ... its feet over hers, at the most
    down: float                         # ... and under them
    hit_after_s: float                  # s after the key it may hit at the latest
    window_s: float                     # s after the key a kill or a hit can be its
    hide_s: float                       # s a monster may stay out of sight under its hit and still be the same one
    gain_after_key: tuple[float, float]  # s after the key its kill's EXP gain shows
    pick_s: float                       # s after the key it picks its monster, from where she is then
    order: str                          # "line": the first in a star's way | "band": the first band, any in it
    look_band: tuple[float, float]      # how far above and below her feet a look searches (attack.reach.LOOK_*)
    shows_wait_s: float                 # a look for the next attack waits for a kill to show this long at most


@dataclass(frozen=True)
class Perception:
    """What perception reads of her: her files (paths.of: "" the thief's, today's names), her bottom bar's lines,
    her heading's box and its sureness while she attacks, and her attack's hit effect over a monster."""
    files: str
    job: str | None
    name: str | None
    heading_box: tuple[int, int, int]    # perception.heading's UP, DOWN, HALF
    attack_score: float                  # ... ATTACK_SCORE, ATTACK_MARGIN
    attack_margin: float
    hit_effect: tuple[float, float, float | None]  # (w, h game px, ms)


@dataclass(frozen=True)
class Policy:
    default_mode: str          # the attack mix: "choice" (Lucky Seven or the regular attack) | "skill"
    finish_with_regular: bool  # the regular attack finishes a low monster (the user's rule for the thief)
    regular_needs_box: bool    # the regular attack only at a monster in its own reach (a wand's swing)
    unseen: str                # at nothing in sight: "regular" (hunt.UNSEEN_ATTACK: no MP thrown away) | "none"
    close_quarters: str        # "claw" (stance's claw-edge rules) | "touch" (danger's escape only)
    mp_floor: float            # MP kept back (0: a cast whenever it's paid for)


@dataclass(frozen=True)
class Pet:
    """Her pet's auto-potion: under these shares of her bars it drinks, and what a potion gives."""
    hp_warning: float
    mp_warning: float
    hp_heal: float
    mp_heal: float


@dataclass(frozen=True)
class Buff:
    id: str
    key: str                  # its GameKeys field
    skill_id: int
    level: int | None
    mp: float | None          # MP a cast (skills.json's mpCon at her level: S3)
    duration_s: float | None  # None: a toggle (on until pressed again, a death or a re-login)
    action_ms: int            # its cast's frames (alert2)


@dataclass(frozen=True)
class Priors:
    """What the planner was tuned or measured with (learned values override them)."""
    kill_s_per_exp: float  # behavior.patrol.KILL_S_PER_EXP
    danger_hp_s: float     # patrol.DANGER_HP_S
    hp_price: float        # patrol.HP_PRICE
    ground_danger: float   # behavior.hunt.GROUND_DANGER
    hp_exp: float          # hunt.HP_EXP
    seen_share: float      # behavior.tour.SEEN_SHARE
    threat_s: float        # attack.danger.THREAT_S
    keep_danger: float     # danger.KEEP_DANGER
    meet_s: float          # control.navigator.MEET_S


@dataclass(frozen=True)
class Character:
    """One character of hers, as the stack takes her (data: `load` makes it of her class and config.toml)."""
    id: str
    klass: str
    stats: config.Character           # speed, jump, HP, MP, level, ... (physics' when she plays)
    fixed_stats: config.Character     # ... and the copy FIXED_STATS pins (her class's: no file changes it)
    skills: dict[str, int]            # skill -> level
    keys: config.GameKeys | None      # None: the flat [keys] (the thief's), or not given yet
    timing: config.Timing | None      # None: the flat [timing]
    ammo: str | None                  # "throwing stars" | None
    stops: tuple[str, ...]            # the hunt's stops of her own: ("out_of_darts",) | ("out_of_mp",)
    buffs: tuple[Buff, ...]           # kept up
    magic_guard: float                # share of a touch to her MP while Magic Guard is on
    attacks: dict[str, AttackSpec]    # "skill", "regular"
    pet: Pet
    policy: Policy
    priors: Priors
    shape: Shape
    perception: Perception
    placeholders: tuple[str, ...] = ()  # what of her profile is a placeholder still (gamelab.classes), said by
                                        # `gamelab character`
    safety_hits: int = 3              # the safety stop's reserve: this many of the map's largest touches (behavior.safety;
                                      # the user, 2026-10-28: "for thief with high evasion this should be 2 hits, 3 for
                                      # magician"; the thief's 3 for now: gamelab.classes.thief.SAFETY_HITS)

    @property
    def name(self) -> str | None:
        return self.perception.name

    @property
    def job(self) -> str | None:
        return self.perception.job

    @property
    def files(self) -> str:
        return self.perception.files


def _classes() -> dict:
    from .classes import magician, thief
    return {"thief": thief, "magician": magician}


def known(path=CONFIG) -> list[str]:
    """Every character a process can play: each class by its name, and config.toml's [characters] tables (one whose
    table doesn't parse too: `load` says why she can't)."""
    return sorted({*config.CLASS_SKILLS, *config.characters(path)[1]})


def named(environ=os.environ) -> bool:
    """Whether ENV names who plays (--character, or set by hand), not `character use`'s file or [characters] default."""
    return bool(environ.get(ENV))


def active_id(environ=os.environ, path=CONFIG, chosen: Path | None = None) -> str:
    """The character this process plays (see the top): ENV, else (unless FIXED_STATS) `gamelab character use`'s
    choice and config.toml's [characters] default, else the thief."""
    if environ.get(ENV):
        return environ[ENV]
    if environ.get(FIXED_STATS):
        return DEFAULT
    chosen = CHOSEN if chosen is None else chosen
    try:
        which = json.loads(chosen.read_text(encoding="utf-8")).get("character")
    except (OSError, ValueError, AttributeError):  # none chosen (or a file that can't be read: as none)
        which = None
    if isinstance(which, str) and which:
        return which
    return config.characters(path)[0]


def how_chosen(environ=os.environ, path=CONFIG, chosen: Path | None = None) -> str:
    """Why `active_id` names the one it does, in words (`gamelab character`)."""
    if environ.get(ENV):
        return f"{ENV} (--character)"
    if environ.get(FIXED_STATS):
        return f"{FIXED_STATS} (the thief unless {ENV} names another)"
    chosen = CHOSEN if chosen is None else chosen
    try:
        if json.loads(chosen.read_text(encoding="utf-8")).get("character"):
            return f"{chosen} (`gamelab character use`)"
    except (OSError, ValueError, AttributeError):
        pass
    return "config.toml [characters] default" if config.characters(path)[0] != DEFAULT else "the default"


def load(which: str | None = None, path=CONFIG, environ=os.environ) -> Character:
    """Character `which` (None: the one this process plays) as her class has her, with what config.toml gives of her:
    the thief's stats are its flat [character], another's her [characters.<id>] tables over her class's. Under
    FIXED_STATS (the tests, goldens and harness) she is her class's alone, config.toml not read, as physics' stats
    weren't: `which` names a class then."""
    which = active_id(environ, path) if which is None else which
    fixed = bool(environ.get(FIXED_STATS))
    table = None if fixed else config.characters(path)[1].get(which)
    if table is not None and table.broken:
        raise config.ConfigError(f"{which} can't play: {table.broken}")
    klass = table.klass if table is not None else which
    classes = _classes()
    if klass not in classes:
        known_ = sorted(classes) if fixed else known(path)
        raise config.ConfigError(f"no character {which!r}: known {known_}" + (
            f" (under {FIXED_STATS} a character is her class)" if fixed else " (config.toml [characters.<id>] names "
                                                                              "more)"))
    c = classes[klass].profile(which)
    if klass == config.THIEF:  # (her stats and Lucky Seven's level: [character], as the stack has them)
        stats = config.Character() if fixed else config.character(path)
        c = classes[klass].leveled(replace(c, stats=stats), stats.lucky_seven)
    elif table is not None:
        c = replace(c, stats=replace(c.fixed_stats, **table.stats))
    if table is not None:
        c = classes[klass].configured(c, table)
    return c


def fixed_stats(which: str) -> config.Character:
    """Character `which`'s stats as FIXED_STATS pins them: her class's copy (the thief's config.Character()), whatever
    config.toml says (not read)."""
    classes = _classes()
    if which not in classes:
        raise config.ConfigError(f"under {FIXED_STATS} {ENV} names a class ({sorted(classes)}), got {which!r}")
    return classes[which].FIXED_STATS


def stats(path=CONFIG, environ=os.environ) -> config.Character:
    """The stats of the character this process plays, as motion.physics takes them: under FIXED_STATS her fixed copy;
    the thief's config.toml [character] (exactly as before); another's her own tables over her class's."""
    which = active_id(environ, path)
    if environ.get(FIXED_STATS):
        return fixed_stats(which)
    if which == DEFAULT:
        return config.character(path)
    return load(which, path, environ).stats


_ACTIVE: list[Character] = []


def active() -> Character:
    """The character this process plays, loaded once: physics' stats are hers (motion.physics.CHARACTER), and so are
    her files (paths.of) and what lets a command run (gamelab.__main__), whatever names another later."""
    if not _ACTIVE:
        _ACTIVE.append(load())
    return _ACTIVE[0]


def resolved() -> Character | None:
    """The character this process plays if `active` has loaded her already (her physics among others), else None."""
    return _ACTIVE[0] if _ACTIVE else None


@contextlib.contextmanager
def using(c: Character):
    """Tests: `c` as the character this process plays, for the block."""
    saved = list(_ACTIVE)
    _ACTIVE[:] = [c]
    try:
        yield c
    finally:
        _ACTIVE[:] = saved


def choose(which: str, chosen: Path | None = None, path=CONFIG) -> Path:
    """`gamelab character use <id>`: write who plays when nothing else names her (lab/character.json)."""
    if which not in known(path):
        raise config.ConfigError(f"no character {which!r}: known {known(path)}")
    chosen = CHOSEN if chosen is None else chosen
    chosen.parent.mkdir(parents=True, exist_ok=True)
    chosen.write_text(json.dumps({"character": which}) + "\n", encoding="utf-8")
    return chosen


def describe(c: Character) -> list[str]:
    """A character's profile in a few lines (`gamelab character`)."""
    s = c.stats
    lines = [f"{c.id} ({c.klass}): {c.job or '?'} {c.name or '(name not given)'}; MP {s.mp:g}, SPEED {s.speed:g}, "
             f"JUMP {s.jump:g} (her max HP, level and DEF: perception's, behavior.bootstrap)"]
    for slot, a in c.attacks.items():
        g, s = a.geometry, a.geometry.search
        if g.kind == "star_search":
            where = f"attack.reach's star search, {s.start}..{s.range} ahead" if s else "attack.reach's star search"
        elif s is not None:
            where = (f"the client's search, {s.start}..{s.range} ahead in bands {s.step} wide, {-s.floor} under her "
                     f"feet to {s.roof} over them at its far end; " + ("nothing stops it" if g.line is None else
                                                                      f"line {g.line}")
                     + ("" if a.air else "; from the ground only"))
        else:
            where = f"box {g.box.near:g}..{g.box.far:g} ahead, {g.box.down:g} down to {g.box.up:g} up"
        lines.append(f"  {slot}: {a.name} ({a.key}; {where}; {a.hits} hit(s), {a.mp:g} MP, lock {a.lock_ms} ms)")
    lines.append(f"  safety stop: under {c.safety_hits} of the map's largest touches (behavior.safety)")
    files = "lab/, as before" if not c.files else f"lab/characters/{c.files}/"
    keys = ("the flat [keys]" if c.klass == config.THIEF else "not given yet" if c.keys is None
            else f"[characters.{c.id}.keys]")
    lines.append(f"  files: {files}; keys: {keys}; stops: {', '.join(c.stops) or 'none'}")
    if c.placeholders:
        lines.append(f"  placeholders still: {', '.join(c.placeholders)}")
    return lines


def __getattr__(name: str):
    """THIEF_SHAPE: gamelab.classes.thief.SHAPE, localization's constants (looked up when first asked: the classes
    import this module)."""
    if name == "THIEF_SHAPE":
        return _classes()["thief"].SHAPE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
