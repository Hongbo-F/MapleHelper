"""gamelab.character and gamelab.classes: the character a process plays, her profile as data, config.toml's
[characters] tables (maplehelper.config), and the thief's profile being today's constants (README: Characters)."""

import json
from pathlib import Path

import pytest

from maplehelper import config
from maplehelper.combos import COMBOS

from gamelab import character
from gamelab.classes import magician

NAMES = [c.name for c in COMBOS]

MAGE = {"class": "magician", "name": "MageBaby", "job": "Magician",
        "keys": {"left": "left", "right": "right", "jump": "alt", "attack": "ctrl", "magic_claw": "shift",
                 "magic_guard": "a", "pick_up": "z", "pet_food": "u"},
        "stats": {"mp": 610, "speed": 105},
        "skills": {"magic_claw": 16, "energy_bolt": 1, "magic_guard": 5},
        "pet": {"mp_warning": 0.3},
        "buffs": {"magic_guard": True, "magic_armor": True}}


def _toml(tmp_path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


# --- config.toml's [characters] ---------------------------------------------------------------------------------

def test_without_characters_the_flat_tables_are_the_thiefs_as_before(config_data):
    cfg = config.parse(config_data, NAMES)
    assert cfg.characters == {} and cfg.default_character == "thief"
    assert cfg.keys.magic_claw is None and cfg.keys.lucky_seven.name == "shift"
    shipped = config.load(Path(__file__).resolve().parents[1] / "config.toml", NAMES)
    assert shipped.default_character == "thief"


def test_a_magicians_tables(config_data):
    config_data["characters"] = {"default": "magician", "magician": MAGE, "thief": {"name": "BugBaby", "job": "Rogue"}}
    cfg = config.parse(config_data, NAMES)
    m = cfg.characters["magician"]
    assert cfg.default_character == "magician" and m.klass == "magician" and (m.name, m.job) == ("MageBaby", "Magician")
    assert m.keys.magic_claw.name == "shift" and m.keys.lucky_seven is None and m.keys.magic_guard.name == "a"
    assert m.stats == {"mp": 610.0, "speed": 105.0} and m.skills["magic_claw"] == 16
    assert m.pet == {"mp_warning": 0.3} and m.buffs == {"magic_guard": True, "magic_armor": True}
    assert cfg.characters["thief"].name == "BugBaby" and cfg.characters["thief"].keys is None
    assert cfg.keys.lucky_seven.name == "shift"  # the flat [keys] are the thief's still


def test_a_class_named_table_needs_no_class(config_data):
    config_data["characters"] = {"magician": {"name": "MageBaby"}}
    assert config.parse(config_data, NAMES).characters["magician"].klass == "magician"


# Mistakes everyone's (the flat tables, the thief's own, `default`): raised
@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["keys"].update(magic_claw="q"), "['magic_claw'], not a thief's skill"),
    (lambda d: d["characters"].update(thief={"keys": {"left": "left"}}), "the thief's keys, timing and stats"),
    (lambda d: d["characters"].update(thief={"class": "magician"}), "class must be \"thief\" (the flat tables"),
    (lambda d: d["characters"].update(thief={"name": ""}), "must be text"),
    (lambda d: d["characters"].update(default="bishop"), "default must name a character"),
    (lambda d: d["characters"].update(default="auto"), "needs her bottom bar read"),
])
def test_bad_characters_are_explained(config_data, mutate, message):
    config_data["characters"] = {"magician": json.loads(json.dumps(MAGE))}
    mutate(config_data)
    with pytest.raises(config.ConfigError) as err:
        config.parse(config_data, NAMES)
    assert message in str(err.value)


# Mistakes in another character's table: hers, kept apart (CharacterConfig.broken), the thief's config as ever
@pytest.mark.parametrize("mutate, who, message", [
    (lambda d: d["characters"]["magician"]["keys"].update(lucky_seven="g"), "magician", "['lucky_seven'], not a magician's"),
    (lambda d: d["characters"]["magician"]["keys"].update(haste="h"), "magician", "unknown action(s) ['haste']"),
    (lambda d: d["characters"]["magician"]["keys"].pop("attack"), "magician", "[characters.magician.keys] is missing ['attack']"),
    (lambda d: d["characters"]["magician"]["keys"].update(magic_armor="f12"), "magician", "already used by characters.magician"),
    (lambda d: d["characters"]["magician"].update({"class": "bishop"}), "magician", "class must be one of"),
    (lambda d: d["characters"].update(mage2={"name": "X"}), "mage2", "class must be one of"),
    (lambda d: d["characters"].update(bug={"class": "thief"}), "bug", "the thief is [characters.thief]"),
    (lambda d: d["characters"].update({"Mage": {"class": "magician"}}), "Mage", "lower-case letters"),
    (lambda d: d["characters"].update(mage3=4), "mage3", "must be a table"),
    (lambda d: d["characters"]["magician"].update(level=3), "magician", "unknown setting(s) ['level']"),
    (lambda d: d["characters"]["magician"]["stats"].update(luk=4), "magician", "unknown stat(s) ['luk']"),
    (lambda d: d["characters"]["magician"]["stats"].update(lucky_seven=4), "magician", "unknown stat(s) ['lucky_seven']"),
    (lambda d: d["characters"]["magician"]["stats"].update(mp=0), "magician", "characters.magician.stats.mp must be"),
    # Her HP, level and DEF are perception's (behavior.bootstrap): a table giving one is refused, saying so.
    (lambda d: d["characters"]["magician"]["stats"].update(hp=380), "magician",
     "has ['hp']: her HP, level and DEF aren't read from config.toml"),
    (lambda d: d["characters"]["magician"]["stats"].update(level=26, defence=60), "magician",
     "has ['level', 'defence']: her HP, level and DEF aren't read from config.toml"),
    (lambda d: d["characters"]["magician"]["skills"].update(magic_claw=31), "magician", "must be her skill's level"),
    (lambda d: d["characters"]["magician"]["skills"].update(lucky_seven=3), "magician", "unknown skill(s) ['lucky_seven']"),
    (lambda d: d["characters"]["magician"]["pet"].update(hp_warning=70), "magician", "a share of her bar"),
    (lambda d: d["characters"]["magician"]["buffs"].update(magic_guard="yes"), "magician", "must be true or false"),
    (lambda d: d["characters"]["magician"].update(name=""), "magician", "must be text"),
])
def test_another_characters_mistakes_are_hers(config_data, mutate, who, message):
    config_data["characters"] = {"magician": json.loads(json.dumps(MAGE))}
    mutate(config_data)
    cfg = config.parse(config_data, NAMES)  # (the player's and the thief's commands' load: it goes on)
    assert message in cfg.characters[who].broken and cfg.keys.lucky_seven.name == "shift"
    assert cfg.default_character == "thief"
    others = {cid: c for cid, c in cfg.characters.items() if cid != who}
    assert all(c.broken is None for c in others.values())


def test_a_broken_table_never_stops_the_thief(tmp_path):
    """The review's: a mistake in the magician's table (Lucky Seven's key under hers) stopped every thief command, the
    player too (config.parse, physics' stats at import): now it's hers alone."""
    path = _toml(tmp_path, '[character]\nspeed = 125\n[characters.magician.keys]\nleft = "left"\nright = "right"\n'
                           'jump = "alt"\nattack = "ctrl"\nlucky_seven = "shift"\n')
    chosen = tmp_path / "character.json"
    assert character.active_id({}, path, chosen) == "thief"
    assert character.stats(path, {}) == config.Character(speed=125)
    assert character.load("thief", path, {}).stats.speed == 125.0
    assert "magician" in character.known(path)
    with pytest.raises(config.ConfigError, match=r"magician can't play: .*\['lucky_seven'\], not a magician's"):
        character.load("magician", path, {})
    with pytest.raises(config.ConfigError, match="can't play"):
        character.stats(path, {character.ENV: "magician"})  # (named: she can't, and says why)


def test_characters_alone_for_before_anything_loads(tmp_path):
    assert config.characters(tmp_path / "nope.toml") == ("thief", {})
    path = _toml(tmp_path, '[characters]\ndefault = "magician"\n[characters.magician]\nname = "MageBaby"\n')
    default, tables = config.characters(path)
    assert default == "magician" and tables["magician"].name == "MageBaby"


# --- which character a process plays -------------------------------------------------------------------------------

def test_who_plays_first_match(tmp_path):
    path = _toml(tmp_path, '[characters]\ndefault = "magician"\n')
    chosen = tmp_path / "character.json"
    assert character.active_id({}, path, chosen) == "magician"            # config.toml's default
    chosen.write_text(json.dumps({"character": "thief"}))
    assert character.active_id({}, path, chosen) == "thief"               # `gamelab character use` over it
    assert character.active_id({character.ENV: "magician"}, path, chosen) == "magician"  # --character / the env
    assert character.active_id({}, tmp_path / "nope.toml", tmp_path / "none.json") == "thief"  # the default
    chosen.write_text("not json")
    assert character.active_id({}, path, chosen) == "magician"            # a file that can't be read: as none


def test_under_fixed_stats_only_the_environment_counts(tmp_path):
    path = _toml(tmp_path, '[characters]\ndefault = "magician"\n')
    chosen = tmp_path / "character.json"
    chosen.write_text(json.dumps({"character": "magician"}))
    fixed = {character.FIXED_STATS: "1"}
    assert character.active_id(fixed, path, chosen) == "thief"
    assert character.active_id({**fixed, character.ENV: "magician"}, path, chosen) == "magician"
    from gamelab.motion import physics
    assert character.FIXED_STATS == physics.FIXED_STATS


def test_the_tests_play_the_thief():
    assert character.active_id() == "thief" and character.active().klass == "thief"


def test_choose_writes_the_lab_file(tmp_path):
    chosen = tmp_path / "lab" / "character.json"
    character.choose("magician", chosen, tmp_path / "nope.toml")
    assert json.loads(chosen.read_text()) == {"character": "magician"}
    with pytest.raises(config.ConfigError, match="no character 'bishop'"):
        character.choose("bishop", chosen, tmp_path / "nope.toml")


def test_known_characters(tmp_path):
    path = _toml(tmp_path, '[characters.mage]\nclass = "magician"\n')
    assert character.known(path) == ["mage", "magician", "thief"]
    assert character.load("mage", path, {}).files == "mage"  # her own files' folder (paths.of)
    with pytest.raises(config.ConfigError, match="no character"):
        character.load("bishop", path, {})
    # Under FIXED_STATS (these tests'), a character is her class's, config.toml not read
    with pytest.raises(config.ConfigError, match="a character is her class"):
        character.load("mage", path)
    with pytest.raises(config.ConfigError, match="names a class"):
        character.stats(path, {character.ENV: "mage", character.FIXED_STATS: "1"})


def test_using_puts_a_profile_in_place():
    mage = character.load("magician")
    with character.using(mage):
        assert character.active() is mage
    assert character.active().id == "thief"


# --- the profiles -------------------------------------------------------------------------------------------------

def test_the_thiefs_lucky_seven_at_her_level(tmp_path):
    """The review's: her profile kept Lucky Seven at level 16 and 14 MP whatever [character] lucky_seven said, where the
    stack casts at hers (Attacks(level=c.lucky_seven): choice.lucky_seven)."""
    from gamelab.attack import choice
    path = _toml(tmp_path, "[character]\nlucky_seven = 18\n")
    l7 = character.load("thief", path, {}).attacks["skill"]
    want = choice.lucky_seven(18)
    assert (l7.level, l7.mp, l7.lo, l7.hi) == (18, 15.0, want.lo, want.hi) and l7.mp == choice.cast_mp(18)
    assert character.load("thief", path, {}).skills == {"lucky_seven": 18}
    assert "Lucky Seven (lucky_seven;" in character.describe(character.load("thief", path, {}))[1]
    assert "15 MP" in character.describe(character.load("thief", path, {}))[1]


def test_the_thiefs_stats_are_physics_as_before(tmp_path):
    path = _toml(tmp_path, "[character]\nspeed = 130\njump = 110\n")
    assert character.stats(path, {}) == config.Character(130, 110) == config.character(path)
    assert character.stats(path, {character.FIXED_STATS: "1"}) == config.Character()
    assert character.load("thief", path, {}).stats == config.Character(130, 110)
    assert character.load("thief", path, {}).skills == {"lucky_seven": 16}


def test_under_fixed_stats_config_toml_isnt_read(tmp_path):
    broken = _toml(tmp_path, "[characters = not toml")
    fixed = {character.FIXED_STATS: "1"}
    assert character.stats(broken, fixed) == config.Character()  # (physics', as before: the goldens' and the tests')
    assert character.load(None, broken, fixed).id == "thief"
    assert character.load("magician", broken, {**fixed, character.ENV: "magician"}).stats == magician.FIXED_STATS
    with pytest.raises(config.ConfigError):
        character.stats(broken, {})


def test_a_magicians_stats_hers_over_her_class(tmp_path):
    path = _toml(tmp_path, "[characters.magician]\n[characters.magician.stats]\nmp = 380\nspeed = 105\n")
    s = character.stats(path, {character.ENV: "magician"})
    assert (s.mp, s.speed, s.jump) == (380.0, 105.0, 100.0)
    # FIXED_STATS: her class's copy, whatever the file says (her goldens don't hang on it)
    assert character.stats(path, {character.ENV: "magician", character.FIXED_STATS: "1"}) == magician.FIXED_STATS
    assert character.stats(tmp_path / "nope.toml", {character.ENV: "magician"}) == magician.FIXED_STATS


def test_a_magicians_profile_from_her_tables(tmp_path, config_data):
    config_data["characters"] = {"magician": MAGE}
    table = config.parse(config_data, NAMES).characters["magician"]
    m = magician.configured(magician.profile(), table)
    assert (m.name, m.job, m.keys.magic_claw.name) == ("MageBaby", "Magician", "shift")
    claw = m.attacks["skill"]
    assert (claw.level, claw.mp, claw.hits, claw.defence, claw.speed) == (16, 16.0, 2, "mdd", None)
    assert [b.id for b in m.buffs] == ["magic_guard"]  # Magic Armor isn't learned (her skills): not kept up
    assert m.magic_guard == pytest.approx(0.42) and m.buffs[0].duration_s is None and m.buffs[0].mp == 8.0
    assert m.pet.mp_warning == 0.3 and m.pet.hp_warning == magician.PET.hp_warning
    assert m.stops == ("out_of_mp",) and m.ammo is None and m.files == "magician"


def test_magic_claw_is_the_clients_search():
    """Magic Claw as the client picks (range/magic-claw-range.md 1.2-1.3, which supersedes design.md 4.1's box: the
    review's): the aimed spell's search, 50 to 300 ahead in bands 20 wide, band k's window 28 -/+ (50 + 20 k) // 4 over
    her feet, the first band picking; nothing stops it and it's cast from the ground only (the user's, settled)."""
    from gamelab.localization import monsters
    m = character.load("magician")
    g = m.attacks["skill"].geometry
    assert (g.kind, g.line, g.air, g.point_blank, g.order, g.mob_count) == ("search", None, None, None, "band", 1)
    s = g.search
    assert (s.start, s.range, s.step, s.spread, s.row, s.bands) == (50, 300, 20, 4, 28, 13)
    assert [s.window(k) for k in range(s.bands)] == [  # the range doc's table 1.3
        (16, 40), (11, 45), (6, 50), (1, 55), (-4, 60), (-9, 65), (-14, 70), (-19, 75), (-24, 80), (-29, 85),
        (-34, 90), (-39, 95), (-44, 100)]
    assert (s.floor, s.roof) == (-44, 100) and m.attacks["skill"].air is False
    # Her Shape from it: from 6 ahead (never behind her) to 374, feet 110 over hers to 119 under
    sh = m.shape
    assert (sh.near, sh.ahead, sh.up, sh.down, sh.order) == (6.0, 374.0, 110.0, 119.0, "band")
    assert sh.look_band == (180, 124) and sh.hit_after_s == pytest.approx(0.715) and sh.pick_s == 0.065
    assert magician.WIDEST_HALF == max(monsters.MONSTER_HALF.values())
    assert magician.TALLEST_TOP == max(monsters.MONSTER_FLINCH_TOP.values())
    assert m.attacks["skill"].lock_ms == 730 and m.attacks["skill"].cycle_s == 0.79
    assert m.policy.default_mode == "skill" and m.policy.unseen == "none" and m.policy.close_quarters == "touch"
    assert "lock and delays" in m.placeholders and not {"air", "walls", "the box's edges"} & set(m.placeholders)
    assert "from the ground only" in "\n".join(character.describe(m)) and "nothing stops it" in "\n".join(
        character.describe(m))


def test_the_thiefs_profile_is_todays_constants():
    """The thief's profile describes the stack's constants (the code reads them there): equal, so they can't drift."""
    from maplehelper.game import COMBAT
    from gamelab.attack import choice, danger, reach
    from gamelab.behavior import attacks, fight, hunt, patrol, tour
    from gamelab.control import actions, navigator
    from gamelab.localization import monsters, senses, tracking
    from gamelab.perception import heading
    from gamelab.sim import world
    t = character.load("thief")
    l7, star = t.attacks["skill"], t.attacks["regular"]
    assert (l7.level, l7.hits, (l7.lo, l7.hi), l7.mp) == (choice.SKILL_LEVEL, choice.SKILL_STARS, choice.SKILL_STAR,
                                                         choice.cast_mp(choice.SKILL_LEVEL))
    assert choice.lucky_seven(l7.level) == choice.Attack("lucky_seven", l7.hits, l7.lo, l7.hi, l7.mp)
    assert (star.hits, (star.lo, star.hi)) == (choice.REGULAR.stars, choice.STAR) and star.key == choice.REGULAR.key
    assert (l7.name, star.name) == (choice.lucky_seven(16).name, choice.REGULAR.name)
    for a in (l7, star):
        assert (a.lock_ms, a.release_ms) == (COMBAT.attack_lock_ms, COMBAT.attack_release_ms)
        assert a.lock_extra_ms == (actions.THROW_LOCK_EXTRA_MS, actions.JUMP_THROW_LOCK_EXTRA_MS)
        assert (a.arrow_lock_ms, a.damage_ms, a.lands_ms) == (actions.ARROW_LOCK_MS, actions.DAMAGE_SHOWS_MS,
                                                              actions.STAR_DELAY_MS)
        assert (a.speed, a.cycle_s, a.miss) == (reach.STAR_SPEED, choice.THROW_S, choice.MISS)
        assert a.cycle_s == patrol.THROW_S and a.hidden_s == fight.HIDDEN_FOR == attacks.HIDDEN_FOR
    assert star.geometry.point_blank == "claw" and l7.geometry.point_blank is None
    s = t.shape
    assert (s.near, s.ahead, s.up, s.down, s.hit_after_s, s.window_s, s.hide_s) == (
        -10, tracking.THROW_REACH, tracking.THROW_DY, tracking.THROW_DY, tracking.HIT_AFTER_S, tracking.STAR_WINDOW,
        tracking.HIDE_S)
    from gamelab.motion import physics
    assert s.pick_s == physics.JUMP_LAG
    g = l7.geometry.search
    assert (g.start, g.range, g.step, g.spread, g.row) == (reach.SEARCH_START, reach.STAR_RANGE, reach.SEARCH_STEP,
                                                           reach.SEARCH_SPREAD, reach.SHOT_H)
    assert star.geometry.search == g and l7.geometry.kind == "star_search"
    from gamelab.classes import thief
    assert (thief.LUCKY_SEVEN_MP, thief.LUCKY_SEVEN_DAMAGE) == (choice.LUCKY_SEVEN_MP, choice.LUCKY_SEVEN_DAMAGE)
    for level in range(0, 23):  # (past the tables' ends too: their nearest)
        a, b = thief.lucky_seven(level), choice.lucky_seven(level)
        assert (a.hits, a.lo, a.hi, a.mp) == (b.stars, b.lo, b.hi, b.mp), level
    assert thief.lucky_seven(choice.SKILL_LEVEL) == l7
    assert s.gain_after_key == monsters.GAIN_AFTER_KEY and s.shows_wait_s == senses.SHOWS_WAIT_S
    assert s.look_band == (reach.LOOK_UP, reach.LOOK_DOWN) and s is character.THIEF_SHAPE
    p = t.perception
    assert p.heading_box == (heading.UP, heading.DOWN, heading.HALF) and p.files == ""
    assert (p.attack_score, p.attack_margin) == (heading.ATTACK_SCORE, heading.ATTACK_MARGIN)
    assert t.policy.unseen == ("regular" if hunt.UNSEEN_ATTACK else "none")
    assert (t.pet.hp_warning, t.pet.mp_warning, t.pet.hp_heal, t.pet.mp_heal) == (
        world.PET_HP, world.PET_MP, world.HP_POTION, world.LEVEL_25.mp_potion)
    r = t.priors
    assert (r.kill_s_per_exp, r.danger_hp_s, r.hp_price) == (patrol.KILL_S_PER_EXP, patrol.DANGER_HP_S, patrol.HP_PRICE)
    assert (r.ground_danger, r.seen_share) == (hunt.GROUND_DANGER, tour.SEEN_SHARE)
    # A bar of her HP at the user's reference prices (2 HP a meso, at MESO_EXP): the simulator's her, 774 HP.
    assert patrol.HP_PRICE == hunt.bar_exp(world.LEVEL_25.max_hp) == 193.5
    assert (r.threat_s, r.keep_danger, r.meet_s) == (danger.THREAT_S, danger.KEEP_DANGER, navigator.MEET_S)
    c = config.Character()
    assert (c.avoid, c.mp_regen) == (world.EVADE_ACC, world.MP_REGEN)
    # Her HP, level and DEF aren't the profile's (nor config.toml's): perception's (behavior.bootstrap).
    assert not {"hp", "level", "defence"} & {f.name for f in __import__("dataclasses").fields(c)}
    from gamelab.motion import physics
    assert physics.climb_of(c.speed) == physics.CLIMB_SPEED == 121.0 and t.fixed_stats == c  # (SPEED + 20: his profile's)


def test_the_magicians_skill_tables_are_the_game_datas():
    from gamedata.paths import TABLES
    path = TABLES / "skills.json"
    if not path.is_file():
        pytest.skip("no game data tables (python -m gamedata tables)")
    skills = json.loads(path.read_text(encoding="utf-8"))

    def column(sid, name):
        levels = skills[sid]["levels"]
        return tuple(levels[str(k)][name] for k in range(1, len(levels) + 1))
    assert magician.MAGIC_CLAW_MP == column("2001003", "mpCon")
    assert (magician.MAGIC_GUARD_X, magician.MAGIC_GUARD_MP) == (column("2001000", "x"), column("2001000", "mpCon"))
    assert (magician.MAGIC_ARMOR_S, magician.MAGIC_ARMOR_MP) == (column("2001001", "time"), column("2001001", "mpCon"))
    assert set(column("2001003", "attackCount")) == {magician.MAGIC_CLAW.hits}
    assert set(column("2001003", "mobCount")) == {magician.CAST.mob_count}


def test_the_safety_stops_hits_are_hers_the_thief_3_for_now_the_magician_2_for_now():
    """The user, 2026-10-28: "Well for thief with high evasion this should be 2 hits, 3 for magician" (behavior.safety:
    her reserve is that many of the map's largest touch), and at ~15:30 "3 hits for now" for the thief, until her retreat
    dodges (gamelab.classes.thief.SAFETY_HITS says why and when it goes back to 2); for the magician, 2026-10-29 ~00:01,
    her pet drinking only at ~55%: "Stop at 2 touches for now", under her pet's warning (TooStrong lets the map run);
    `gamelab character` says it."""
    from gamelab.behavior import safety
    from gamelab.classes import thief
    assert (character.load("thief").safety_hits, character.load("magician").safety_hits) == (3, 2) == (
        thief.SAFETY_HITS, magician.SAFETY_HITS) and safety.HITS == 3
    import inspect
    said = inspect.getsource(thief).split("SAFETY_HITS = 3")[0].rsplit("\n\n", 1)[-1]
    assert "3 hits for now" in said and "Back to 2 once her retreat dodges" in said
    said = inspect.getsource(magician).split("SAFETY_HITS = 2")[0].rsplit("\n\n", 1)[-1]
    assert "Stop at 2 touches for now" in said
    assert magician.PET.hp_warning == 0.55 and 2 * 137 / 628 < magician.PET.hp_warning  # (her 2 Horny touches live)
    assert "  safety stop: under 2 of the map's largest touches (behavior.safety)" in character.describe(
        character.load("magician"))



# --- the command line and physics (S0b) -------------------------------------------------------------------------

def test_physics_takes_the_stats_of_who_plays(tmp_path):
    from gamelab.motion import physics
    path = _toml(tmp_path, "[character]\nspeed = 125\njump = 110\n[characters.magician.stats]\nmp = 380\n")
    assert physics.CHARACTER == config.Character()  # the tests: FIXED_STATS, the thief
    assert physics.stats(path, environ={}) == config.Character(speed=125, jump=110)  # the thief's [character], as before
    assert physics.stats(path, environ={physics.FIXED_STATS: "1"}) == config.Character()
    mage = physics.stats(path, environ={character.ENV: "magician"})
    assert (mage.mp, mage.speed) == (380.0, magician.FIXED_STATS.speed)
    assert physics.stats(path, environ={character.ENV: "magician", physics.FIXED_STATS: "1"}) == magician.FIXED_STATS


def test_the_character_flag_plays_her_in_the_process_and_its_children():
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in __import__("os").environ.items() if k != character.ENV}
    out = subprocess.run([sys.executable, "-m", "gamelab", "--character", "magician", "character"], cwd=root, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.startswith("plays: magician (by GAMELAB_CHARACTER"), out.stdout
    code = ("import os, sys; from gamelab.__main__ import main; main(['--character', 'magician', 'character']); "
            "from gamelab.motion import physics; print(os.environ['GAMELAB_CHARACTER'], physics.CHARACTER.mp)")
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.splitlines()[-1] == f"magician {magician.FIXED_STATS.mp}", out.stdout + out.stderr


def test_the_flag_checks_who_and_when(monkeypatch):
    from gamelab import __main__ as cli
    from gamelab.motion import physics  # noqa: F401 (loaded: who plays is resolved, the thief)
    monkeypatch.setenv(character.ENV, "thief")  # (restored after the test)
    assert "no character 'bishop'" in cli._play_as("bishop")
    assert "comes too late" in cli._play_as("magician")  # her physics is the thief's here already
    assert cli._play_as("thief") is None
    assert cli.main(["--character", "bishop", "character"]) == 2


def test_the_character_command(monkeypatch, capsys, tmp_path):
    from gamelab import __main__ as cli
    monkeypatch.delenv(character.ENV, raising=False)
    assert cli.main(["character"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("plays: thief") and "* thief (thief)" in out and "magician (magician)" in out
    assert "placeholders still" in out
    assert cli.main(["character", "use", "magician"]) == 0
    assert json.loads(character.CHOSEN.read_text()) == {"character": "magician"}  # (the test's own: conftest)
    out = capsys.readouterr().out  # (the review's: what the choice doesn't do yet, and what keeps running)
    assert "plays her only when named" in out and "perceive and pet processes running now keep" in out
    assert "the perception service and the pet feeder play her once they start again" in out
    monkeypatch.delenv(character.FIXED_STATS)  # (as a command has it: the file read; restored after the test)
    assert cli.main(["character"]) == 0 and "any other command only with --character magician" in (
        capsys.readouterr().out)
    real = character.load
    monkeypatch.setattr(character, "load", lambda cid, *a, **k: (_ for _ in ()).throw(config.ConfigError(
        "magician can't play: [characters.magician.keys] is missing ['attack']")) if cid == "magician" else real(cid))
    monkeypatch.setenv(character.FIXED_STATS, "1")
    monkeypatch.setenv(character.ENV, "thief")
    assert cli.main(["character"]) == 0  # (her table's mistake said; the thief's shown as ever)
    out = capsys.readouterr().out
    assert "magician: magician can't play" in out and "* thief (thief)" in out
    monkeypatch.delenv(character.ENV)
    assert cli.main(["character", "use", "bishop"]) == 2 and cli.main(["character", "use"]) == 2
    assert cli.main(["character", "check"]) == 2  # nothing is read yet (the game isn't looked at: conftest refuses)
    assert "nothing was looked at" in capsys.readouterr().out


def test_the_magician_hunts_but_runs_none_of_the_thiefs_probes_locks_or_combos(monkeypatch):
    """From 2026-10-29 the magician hunts (`tour`, `hunt`, `nav`: her Magic Claw, gamelab.attack.magic); `lock`,
    `combo` and `probe run` stay the thief's (__main__.HERS_NOT_YET), refused saying why and what chose her."""
    from gamelab import __main__ as cli
    parser = cli.build_parser()
    monkeypatch.delenv(character.ENV, raising=False)
    monkeypatch.setattr(character, "_ACTIVE", [])  # (who plays resolved again, as a command's start does)
    for argv in (["hunt", "kill"], ["tour"], ["lock"], ["probe", "run", "claw-edge"], ["perceive"]):
        try:
            args = parser.parse_args(argv)
        except SystemExit:  # (a command's own arguments aside: only its name counts)
            continue
        assert cli._refused_as(args) is None
    monkeypatch.setenv(character.ENV, "magician")
    monkeypatch.setattr(character, "_ACTIVE", [])
    assert cli._refused_as(parser.parse_args(["tour"])) is None  # (hers: Magic Claw)
    refused = cli._refused_as(parser.parse_args(["probe", "run", "claw-edge"]))
    assert "only the thief runs it so far" in refused and "by GAMELAB_CHARACTER (--character)" in refused
    assert "star and claw probes" in refused
    assert cli._refused_as(parser.parse_args(["probe", "list"])) is None  # offline
    assert cli._refused_as(parser.parse_args(["character"])) is None
    assert cli._refused_as(parser.parse_args(["calibrate"])) is None  # (her own: L0's first step)
    assert cli._refused_as(parser.parse_args(["perceive"])) is None  # (the service is whoever plays)
    assert cli.main(["probe", "run", "claw-edge"]) == 2


def test_the_perception_service_is_whoever_plays_and_says_so(monkeypatch):
    """From 2026-10-29 the service perceives the character it plays (her name tag, pictures and Shape), named or
    chosen by `character use` (the babysitter starts it with no --character), and says who in its welcome: a command
    that hunts refuses one perceiving another (__main__._perceives_her). The review's worry (a service built as the
    magician serving the thief's hunts after a switch) is that refusal now. The service a command starts is that
    command's character's (GAMELAB_CHARACTER kept; the thief's command line as before)."""
    import sys
    from gamelab import __main__ as cli
    from gamelab.localization import service
    monkeypatch.setenv(character.ENV, "magician")
    monkeypatch.setattr(character, "_ACTIVE", [])
    parser = cli.build_parser()
    assert cli._refused_as(parser.parse_args(["perceive"])) is None
    assert cli._refused_as(parser.parse_args(["perceive", "--frames", "x"])) is None  # (saved frames: its own pipe)
    started = []
    monkeypatch.setattr(service, "_start_detached", lambda command, folder: started.append(command))
    monkeypatch.setattr(service, "running", lambda *a: False)
    monkeypatch.setattr(service, "saved", lambda *a: False)
    master = (f'cmd.exe /c set PYTHONUNBUFFERED=1&& set OPENBLAS_NUM_THREADS=1&& "{sys.executable}" -u -X utf8 -m '
              f'gamelab perceive --map 10002073 >> "{service.log_path()}" 2>&1')
    for who, line in (("magician", "cmd.exe /c set GAMELAB_CHARACTER=magician&& " + master[len("cmd.exe /c "):]),
                      ("thief", master), ("", master)):
        monkeypatch.setenv(character.ENV, who)
        monkeypatch.setattr(service, "_ENSURED", [-1e18])
        assert service.ensure(10002073, log=lambda m: None) is True
        assert started[-1] == line, who


def test_a_choice_by_file_alone_plays_the_thief_in_the_game_but_for_the_service_the_pet_and_the_hunts(monkeypatch,
                                                                                                         capsys):
    """The review's: `character use magician` left behind switched the babysat pet feeder and perception service to
    her files without a word. Until her bottom bar is read (S0g), a command in the game plays her only when named:
    under the file (or [characters] default) alone it plays the thief and says so. From 2026-10-29 the perception
    service and the pet feeder (__main__.FOLLOW_CHOICE: the babysitter starts them with no --character) and the
    commands that hunt play the one chosen, and a hunt refuses a service perceiving another (_perceives_her)."""
    from gamelab import __main__ as cli
    parser = cli.build_parser()
    character.choose("magician")  # (the test's own file: conftest)
    monkeypatch.delenv(character.FIXED_STATS)  # (as a command has it: the file read; restored after the test)
    monkeypatch.setenv(character.ENV, "")  # (none named; restored after the test)
    monkeypatch.setattr(character, "_ACTIVE", [])
    assert cli._unnamed(parser.parse_args(["pet", "--every", "1"])) is None  # (hers: the babysitter's pet feeder)
    assert cli._unnamed(parser.parse_args(["perceive"])) is None and not character.named()
    said = cli._unnamed(parser.parse_args(["watch"]))
    assert said.startswith("watch plays the thief: ") and "names magician" in said and "--character magician" in said
    assert character.named() and character.active_id() == "thief"  # (set for the command and its children)
    monkeypatch.setenv(character.ENV, "")
    assert cli._unnamed(parser.parse_args(["character"])) is None  # offline: the file's choice stands
    assert cli._unnamed(parser.parse_args(["perceive", "--frames", "x"])) is None  # (saved frames: offline)
    assert character.active_id() == "magician"
    assert cli._unnamed(parser.parse_args(["tour"])) is None  # it hunts: hers
    assert cli._refused_as(parser.parse_args(["tour"])) is None
    refused = cli._refused_as(parser.parse_args(["probe", "run", "claw-edge"]))
    assert "only the thief runs it so far" in refused and "`gamelab character use`" in refused
    monkeypatch.setenv(character.ENV, "magician")
    assert cli._unnamed(parser.parse_args(["pet", "--every", "1"])) is None  # named: hers (L0's `pet --read`)
    monkeypatch.setenv(character.ENV, "")
    character.choose("thief")
    assert cli._unnamed(parser.parse_args(["pet", "--every", "1"])) is None and not character.named()


def test_the_goldens_play_the_thief_only_and_the_harness_the_magician_too(monkeypatch, capsys):
    """The review's: golden and the harness restart themselves with the environment, GAMELAB_CHARACTER and all, so
    `golden record` would have written the magician's placeholder physics over the thief's traces. The goldens are still
    the thief's alone; from 2026-11-02 the harness plays the magician too (`--character magician`, or GAMELAB_CHARACTER's:
    her world, sim.scenarios.HERS; tests/test_sim_magic.py), and refuses any other."""
    from gamelab.sim import golden, harness
    monkeypatch.setenv(character.ENV, "magician")
    assert golden.main(["record"]) == 2
    assert capsys.readouterr().out.count("the simulator's scenarios are the thief's") == 1
    assert harness._who(["run", "loop"]) == ("magician", "")
    monkeypatch.setenv(character.ENV, "warrior")
    assert harness.main(["run", "loop"]) == 2 and "not warrior" in capsys.readouterr().out
    monkeypatch.setenv(character.ENV, "thief")
    assert golden._thief_only() is None and harness._who(["run", "loop"]) == ("thief", "")


def test_who_plays_is_resolved_once():
    """The review's: what let a command run, her physics and her files each resolved who plays again; now physics'
    stats are `active`'s (the process's: the thief's in the tests), which the refusal and paths.of read too."""
    from gamelab import paths
    from gamelab.motion import physics
    assert character.resolved() is not None and physics.CHARACTER == character.resolved().stats
    assert paths.of().pet == paths.PET


# --- her files (S0c) ------------------------------------------------------------------------------------------

def test_the_thiefs_files_are_paths_names_as_they_are(monkeypatch, tmp_path):
    from gamelab import paths
    mine = paths.of(character.load("thief"))
    for attr, name in paths.MINE.items():
        assert getattr(mine, attr) == getattr(paths, name)
    monkeypatch.setattr(paths, "PET", tmp_path / "pet.json")  # a test's own, as before: read when asked
    assert paths.of().pet == tmp_path / "pet.json"
    with pytest.raises(AttributeError):
        mine.telemetry  # (not hers: everyone's)


def test_a_magicians_files_are_her_own(monkeypatch, tmp_path):
    from gamelab import paths
    mage = character.load("magician")
    mine = paths.of(mage)
    assert mine.pet == paths.LAB / "characters" / "magician" / "pet.json"
    assert mine.heading_shots == paths.LAB / "characters" / "magician" / "heading"
    assert mine.heading_seed == paths.ASSETS / "heading_magician.npz"
    assert paths.calibrated(mine.tag_template, "_1366x768") == (
        paths.LAB / "characters" / "magician" / "tag_template_1366x768.npy")
    assert paths.calibrated(mine.heading_seed, "_1366x768") == paths.ASSETS / "heading_magician_1366x768.npz"
    monkeypatch.setattr(paths, "LAB", tmp_path)
    assert paths.of(mage).grounds == tmp_path / "characters" / "magician" / "grounds.json"


def test_the_stack_opens_her_files(monkeypatch, tmp_path):
    from gamelab import paths
    from gamelab.analysis import calibration
    from gamelab.behavior import pet, tour
    from gamelab.screen import geometry
    monkeypatch.setattr(paths, "LAB", tmp_path)
    monkeypatch.setattr(paths, "GROUNDS", tmp_path / "grounds.json")  # (the thief's: a test's own)
    mage = character.load("magician")
    with character.using(mage):
        cal, small, full = calibration.files(geometry(1366, 768))
        assert cal == tmp_path / "characters" / "magician" / "calibration_1366x768.json"
        assert (small.name, full.name) == ("tag_template_1366x768.npy", "tag_template_full_1366x768.npy")
        pet._save({"level": 50.0, "at": 1.0})
        assert json.loads((tmp_path / "characters" / "magician" / "pet.json").read_text())["level"] == 50.0
        assert pet._state()["level"] == 50.0
        tour._save_grounds(10002073, {"23": {"visits": 1, "exp": 40}})
        assert tour._load_grounds(10002073) == {"23": {"visits": 1, "exp": 40}}
        assert (tmp_path / "characters" / "magician" / "grounds.json").is_file()
    assert tour._load_grounds(10002073) == {} and not paths.GROUNDS.exists()  # the thief's: not hers
    assert calibration.files(geometry(1366, 768))[0] == paths.calibrated(paths.CALIBRATION, "_1366x768")


def test_calibrate_saves_hers_in_her_folder(monkeypatch, tmp_path):
    """The review's: `--character magician calibrate --fresh` (L0's first step) captured the game and then failed
    (FileNotFoundError: nothing made lab/characters/magician/), and her pictures to check went over the thief's."""
    import numpy as np
    from gamelab import paths
    from gamelab.analysis import calibration
    frame = np.zeros((768, 1366, 3), np.uint8)
    monkeypatch.setattr(paths, "LAB", tmp_path)
    monkeypatch.setattr(calibration, "LAB", tmp_path)
    monkeypatch.setattr(calibration.client, "submit",
                        lambda **kw: {"frames": [frame], "client": {"width": 1366, "height": 768}})
    monkeypatch.setattr(calibration.recordings, "screenshot",
                        lambda name, crop, scale: frame[crop[1]:crop[1] + crop[3], crop[0]:crop[0] + crop[2]])
    monkeypatch.setattr(calibration.track, "make_template", lambda small, at, scale=1, geo=None: np.ones((8, 30)))
    with character.using(character.load("magician")):
        calibration.calibrate(tag=(683, 400), fresh=True)
    mine = tmp_path / "characters" / "magician"
    assert sorted(p.name for p in mine.iterdir()) == ["calibrate_1366x768.png", "calibrate_crop_1366x768.png",
                                                      "calibration_1366x768.json", "tag_template_1366x768.npy"]
    assert not (tmp_path / "calibrate_1366x768.png").exists()  # (the thief's: lab/, as before)
    assert calibration.pictures(calibration.geometry(1366, 768))[0] == tmp_path / "calibrate_1366x768.png"


# --- the game data her attacks come from (S0d, S0e) --------------------------------------------------------------

def test_magic_claws_row_in_the_game_data():
    """skills.json's rows as `python -m gamedata tables` now builds them (tables.skills), from the copy of the game's
    data: what the magician's priors rest on [data] (gamelab.classes.magician)."""
    from gamedata import tables
    from gamedata.ms import MsPack
    from gamedata.paths import COPY
    from gamedata.wz import Package
    if not (COPY / "Packs" / "Skill_00000.ms").is_file():
        pytest.skip("no copy of the game's data (python -m gamedata copy)")
    pack = MsPack(COPY / "Packs" / "Skill_00000.ms")
    strings = Package(COPY / "String" / "String_000.wz").image("Skill.img")
    rows = tables.skills({p: pack.image(p) for p in ("Skill/200.img", "Skill/400.img")}, strings)
    claw, bolt, l7 = rows["2001003"], rows["2001002"], rows["4001003"]
    assert claw["info"] == {"type": 1, "magicDamage": 1} and claw["req"] == {"2001002": 1}
    assert claw["desc_range"] == magician.SEARCH.range == 300 and "ball" not in claw and "range" not in claw
    assert (bolt["range"], bolt["desc_range"], bolt["ball"]) == (350, 350, True)
    assert l7["desc_range"] == 200  # Lucky Seven's "within 200" (the claw's; Keen Eyes adds the rest: reach.STAR_RANGE)
    assert tuple(claw["levels"][k]["mpCon"] for k in range(1, 21)) == magician.MAGIC_CLAW_MP


def test_a_maps_monsters_defence_against_magic_avoid_and_level():
    """MapModel's mob_mdd, mob_eva and mob_level (mapdata._mob_hp: MDDamage, eva, level), what a magician's Magic Claw
    goes by; a map cached before them is read for them from the game data, its cache left as it is."""
    import os
    from gamelab.localization import mapdata
    model = mapdata.parse(1, {"info": {}}, "a map", {}, {"100": "Horny Mushroom"}, mob_hp={"100": 403}, mob_pdd={"100": 10},
                          mob_mdd={"100": 0}, mob_eva={"100": 13}, mob_level={"100": 22})
    assert (model.mob_mdd, model.mob_eva, model.mob_level) == ({"Horny Mushroom": 0.0}, {"Horny Mushroom": 13.0},
                                                               {"Horny Mushroom": 22.0})
    assert mapdata.parse(1, {"info": {}}, "a map", {}).mob_mdd == {}
    cache = mapdata.CACHE / "10002073.json"
    if not cache.is_file() or not (mapdata.COPY / "Packs" / "Mob_00000.ms").is_file():
        pytest.skip("no Tree Dungeon cache or game data copy here")
    before = os.stat(cache).st_mtime_ns
    tree = mapdata.load(10002073)
    assert tree.mob_mdd == {"Horny Mushroom": 0.0, "Green Mushroom": 30.0, "Slime": 20.0}
    assert tree.mob_eva == {"Horny Mushroom": 13.0, "Green Mushroom": 8.0, "Slime": 0.0}
    assert tree.mob_level == {"Horny Mushroom": 22.0, "Green Mushroom": 14.0, "Slime": 6.0}
    assert tree.mob_pdd == {"Horny Mushroom": 10.0, "Green Mushroom": 0.0, "Slime": 10.0}  # as cached
    assert os.stat(cache).st_mtime_ns == before  # (one cache, several checkouts: not written)


# --- what localization takes of her attacks (S0f) ----------------------------------------------------------------

def _tracker_with_one(shape=None, x=400.0, dy=0.0):
    """A tracker with her at x 0 (y 0) and one live track standing `x` ahead of her, `dy` above."""
    from types import SimpleNamespace
    from gamelab.localization import tracking
    tr = tracking.MonsterTracker(None) if shape is None else tracking.MonsterTracker(None, shape=shape)
    tr.her = (0.0, 0.0)
    tr.tracks = {1: SimpleNamespace(id=1, state=tracking.LIVE, x=x, y=-dy)}
    tr.predicted = lambda track, at: track.x  # (standing where it is)
    return tr


def test_the_trackers_take_the_thiefs_shape_by_default():
    from types import SimpleNamespace
    from gamelab.localization import memory, senses, tracking, worldstate
    from gamelab.localization.monsters import GAIN_AFTER_KEY, dead_one
    assert tracking.MonsterTracker(None).shape is character.THIEF_SHAPE
    assert memory.Memory.__init__.__defaults__[-1] is character.THIEF_SHAPE
    assert senses.Senses.__init__.__defaults__[-1] is character.THIEF_SHAPE
    assert dead_one.__defaults__[-1] == GAIN_AFTER_KEY
    mage = character.load("magician").shape
    model = SimpleNamespace(mob_exp={}, spawns=[], mob_speeds={})
    assert worldstate.WorldStep(SimpleNamespace(eyes=None, model=model, shape=mage)).tracker.shape is mage
    assert worldstate.WorldStep(SimpleNamespace(eyes=None, model=model)).tracker.shape is character.THIEF_SHAPE


def test_which_monsters_a_throw_went_at_by_her_shape():
    mage = character.load("magician").shape
    for x, dy, thief, magician in ((400.0, 0.0, True, False),     # the star's 474 ahead; her search's 374
                                   (-5.0, 0.0, True, False),      # 10 behind the thief; never behind her (from 6 ahead)
                                   (100.0, 65.0, True, True),     # 70 above for the thief; 110 for her
                                   (100.0, 100.0, False, True),   # her windows widen up to 100 over her feet
                                   (100.0, -90.0, False, True),   # ... and down to 44 under them, a body's top in it
                                   (100.0, -125.0, False, False)):
        tr = _tracker_with_one(x=x, dy=dy)
        tr.throw(1.0, "right")
        assert (1 in tr.throws[-1].aimed) is thief and tr.throws[-1].attack is None, (x, dy)
        tr = _tracker_with_one(mage, x=x, dy=dy)
        tr.throw(1.0, "right", "magic_claw")
        assert (1 in tr.throws[-1].aimed) is magician and tr.throws[-1].attack == "magic_claw", (x, dy)
    tr = _tracker_with_one(mage, x=100.0)
    tr.throw(2.0, "right", "magic_claw")
    assert tr._last_hit(tr.tracks[1]) == pytest.approx(2.0 + mage.hit_after_s)  # no star's flight: 0.715 s
    tr = _tracker_with_one(x=100.0)
    tr.throw(2.0, "right")
    assert tr._last_hit(tr.tracks[1]) == pytest.approx(2.0 + 1.2)  # tracking.HIT_AFTER_S


def test_the_attack_id_goes_through_the_feed_and_the_service():
    from types import SimpleNamespace
    from gamelab.localization import service
    from gamelab.localization.feed import InlineFeed, SyncFeed
    told = []
    step = SimpleNamespace(throw=lambda *a: told.append(a))
    InlineFeed.throw(SimpleNamespace(step=step), 1.0, "left")
    SyncFeed.throw(SimpleNamespace(step=step, _settle=lambda now: None), 2.0, "right", "magic_claw")
    InlineFeed.throw(SimpleNamespace(step=step), 3.0, "left", None, (7, "horny-mushroom", -40.0, 10))
    assert told == [(1.0, "left", None, None), (2.0, "right", "magic_claw", None),
                    (3.0, "left", None, (7, "horny-mushroom", -40.0, 10))]
    sent = []
    client = SimpleNamespace(hello=SimpleNamespace(throws=True), _send=lambda *a: sent.append(a))
    service.ServiceClient.throw(client, 1.0, "left")
    service.ServiceClient.throw(client, 2.0, "right", "magic_claw")
    service.ServiceClient.throw(client, 3.0, "left", None, (7, "horny-mushroom", -40.0, 10))  # (her aim, told)
    assert sent == [("throw", 1.0, "left"), ("throw", 2.0, "right", "magic_claw"),  # the thief's message as before
                    ("throw", 3.0, "left", None, (7, "horny-mushroom", -40.0, 10))]
    told.clear()
    me = SimpleNamespace(runner=SimpleNamespace(step=step))
    for msg in sent:
        service.Service._take(me, None, msg, None)
    assert told == [(1.0, "left", None, None), (2.0, "right", "magic_claw", None),
                    (3.0, "left", None, (7, "horny-mushroom", -40.0, 10))]
