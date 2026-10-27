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
        "stats": {"hp": 380, "mp": 610, "level": 26, "speed": 105},
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
    assert m.stats == {"hp": 380.0, "mp": 610.0, "level": 26.0, "speed": 105.0} and m.skills["magic_claw"] == 16
    assert m.pet == {"mp_warning": 0.3} and m.buffs == {"magic_guard": True, "magic_armor": True}
    assert cfg.characters["thief"].name == "BugBaby" and cfg.characters["thief"].keys is None
    assert cfg.keys.lucky_seven.name == "shift"  # the flat [keys] are the thief's still


def test_a_class_named_table_needs_no_class(config_data):
    config_data["characters"] = {"magician": {"name": "MageBaby"}}
    assert config.parse(config_data, NAMES).characters["magician"].klass == "magician"


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["keys"].update(magic_claw="q"), "['magic_claw'], not a thief's skill"),
    (lambda d: d["characters"]["magician"]["keys"].update(lucky_seven="g"), "['lucky_seven'], not a magician's"),
    (lambda d: d["characters"]["magician"]["keys"].update(haste="h"), "unknown action(s) ['haste']"),
    (lambda d: d["characters"]["magician"]["keys"].pop("attack"), "[characters.magician.keys] is missing ['attack']"),
    (lambda d: d["characters"]["magician"]["keys"].update(magic_armor="f12"), "already used by characters.magician"),
    (lambda d: d["characters"]["magician"].update({"class": "bishop"}), "class must be one of"),
    (lambda d: d["characters"].update(mage2={"name": "X"}), "class must be one of"),
    (lambda d: d["characters"].update(bug={"class": "thief"}), "the thief is [characters.thief]"),
    (lambda d: d["characters"].update(thief={"keys": {"left": "left"}}), "the thief's keys, timing and stats"),
    (lambda d: d["characters"].update(default="bishop"), "default must name a character"),
    (lambda d: d["characters"].update(default="auto"), "needs her bottom bar read"),
    (lambda d: d["characters"].update({"Mage": {"class": "magician"}}), "lower-case letters"),
    (lambda d: d["characters"]["magician"].update(level=3), "unknown setting(s) ['level']"),
    (lambda d: d["characters"]["magician"]["stats"].update(luk=4), "unknown stat(s) ['luk']"),
    (lambda d: d["characters"]["magician"]["stats"].update(lucky_seven=4), "unknown stat(s) ['lucky_seven']"),
    (lambda d: d["characters"]["magician"]["stats"].update(hp=0), "characters.magician.stats.hp must be"),
    (lambda d: d["characters"]["magician"]["skills"].update(magic_claw=31), "must be her skill's level"),
    (lambda d: d["characters"]["magician"]["skills"].update(lucky_seven=3), "unknown skill(s) ['lucky_seven']"),
    (lambda d: d["characters"]["magician"]["pet"].update(hp_warning=70), "a share of her bar"),
    (lambda d: d["characters"]["magician"]["buffs"].update(magic_guard="yes"), "must be true or false"),
    (lambda d: d["characters"]["magician"].update(name=""), "must be text"),
])
def test_bad_characters_are_explained(config_data, mutate, message):
    config_data["characters"] = {"magician": json.loads(json.dumps(MAGE))}
    mutate(config_data)
    with pytest.raises(config.ConfigError) as err:
        config.parse(config_data, NAMES)
    assert message in str(err.value)


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
    assert character.load("mage", path).files == "mage"  # her own files' folder (paths.of)
    with pytest.raises(config.ConfigError, match="no character"):
        character.load("bishop", path)


def test_using_puts_a_profile_in_place():
    mage = character.load("magician")
    with character.using(mage):
        assert character.active() is mage
    assert character.active().id == "thief"


# --- the profiles -------------------------------------------------------------------------------------------------

def test_the_thiefs_stats_are_physics_as_before(tmp_path):
    path = _toml(tmp_path, "[character]\nspeed = 130\njump = 110\n")
    assert character.stats(path, {}) == config.Character(130, 110) == config.character(path)
    assert character.stats(path, {character.FIXED_STATS: "1"}) == config.Character()
    assert character.load("thief", path, {}).stats == config.Character(130, 110)
    assert character.load("thief", path, {}).skills == {"lucky_seven": 16}


def test_a_magicians_stats_hers_over_her_class(tmp_path):
    path = _toml(tmp_path, "[characters.magician]\n[characters.magician.stats]\nhp = 380\nspeed = 105\n")
    s = character.stats(path, {character.ENV: "magician"})
    assert (s.hp, s.speed, s.jump, s.mp) == (380.0, 105.0, 100.0, magician.FIXED_STATS.mp)
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


def test_the_magicians_placeholders_are_the_designs():
    m = character.load("magician")
    box = m.attacks["skill"].geometry.box
    assert (box.near, box.far, box.down, box.up) == (-10, 300, 50, 50)  # design.md 4.1
    g = m.attacks["skill"].geometry
    assert (g.kind, g.line, g.air, g.order, g.mob_count) == ("box", None, None, "nearest", 1)
    assert m.attacks["skill"].lock_ms == 730 and m.attacks["skill"].cycle_s == 0.79
    s = m.shape
    assert (s.ahead, s.behind, s.dy, s.look_band, s.order) == (374.0, 44.0, 60.0, (130, 65), "nearest")
    assert m.policy.default_mode == "skill" and m.policy.unseen == "none" and m.policy.close_quarters == "touch"
    assert m.placeholders and "the box's edges" in m.placeholders


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
    assert (s.ahead, s.dy, s.hit_after_s, s.window_s, s.hide_s) == (
        tracking.THROW_REACH, tracking.THROW_DY, tracking.HIT_AFTER_S, tracking.STAR_WINDOW, tracking.HIDE_S)
    assert s.gain_after_key == monsters.GAIN_AFTER_KEY and s.shows_wait_s == senses.SHOWS_WAIT_S
    assert s.look_band == (reach.LOOK_UP, reach.LOOK_DOWN) and s is character.THIEF_SHAPE
    p = t.perception
    assert p.heading_box == (heading.UP, heading.DOWN, heading.HALF) and p.files == ""
    assert (p.attack_score, p.attack_margin) == (heading.ATTACK_SCORE, heading.ATTACK_MARGIN)
    assert t.policy.unseen == ("regular" if hunt.UNSEEN_ATTACK else "none")
    assert (t.pet.hp_warning, t.pet.mp_warning, t.pet.hp_heal, t.pet.mp_heal) == (
        world.PET_HP, world.PET_MP, world.HP_POTION, world.MP_POTION)
    r = t.priors
    assert (r.kill_s_per_exp, r.danger_hp_s, r.hp_price) == (patrol.KILL_S_PER_EXP, patrol.DANGER_HP_S, patrol.HP_PRICE)
    assert (r.ground_danger, r.hp_exp, r.seen_share) == (hunt.GROUND_DANGER, hunt.HP_EXP, tour.SEEN_SHARE)
    assert (r.threat_s, r.keep_danger, r.meet_s) == (danger.THREAT_S, danger.KEEP_DANGER, navigator.MEET_S)
    c = config.Character()
    assert (c.defence, c.avoid, c.mp_regen) == (danger.DEFENCE, world.EVADE_ACC, world.MP_REGEN)
    from gamelab.motion import physics
    assert c.climb_speed == physics.CLIMB_SPEED and t.fixed_stats == c


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



# --- the command line and physics (S0b) -------------------------------------------------------------------------

def test_physics_takes_the_stats_of_who_plays(tmp_path):
    from gamelab.motion import physics
    path = _toml(tmp_path, "[character]\nspeed = 125\njump = 110\n[characters.magician.stats]\nhp = 380\n")
    assert physics.CHARACTER == config.Character()  # the tests: FIXED_STATS, the thief
    assert physics.stats(path, environ={}) == config.Character(speed=125, jump=110)  # the thief's [character], as before
    assert physics.stats(path, environ={physics.FIXED_STATS: "1"}) == config.Character()
    mage = physics.stats(path, environ={character.ENV: "magician"})
    assert (mage.hp, mage.speed) == (380.0, magician.FIXED_STATS.speed)
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
            "from gamelab.motion import physics; print(os.environ['GAMELAB_CHARACTER'], physics.CHARACTER.hp)")
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.splitlines()[-1] == f"magician {magician.FIXED_STATS.hp}", out.stdout + out.stderr


def test_the_flag_checks_who_and_when(monkeypatch):
    from gamelab import __main__ as cli
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
    assert cli.main(["character", "use", "bishop"]) == 2 and cli.main(["character", "use"]) == 2
    assert cli.main(["character", "check"]) == 2  # nothing is read yet (the game isn't looked at: conftest refuses)
    assert "nothing was looked at" in capsys.readouterr().out


def test_only_the_thief_attacks_in_the_game(monkeypatch):
    from gamelab import __main__ as cli
    parser = cli.build_parser()
    monkeypatch.delenv(character.ENV, raising=False)
    for argv in (["hunt", "kill"], ["tour"], ["lock"], ["probe", "run", "claw-edge"]):
        try:
            args = parser.parse_args(argv)
        except SystemExit:  # (a command's own arguments aside: only its name counts)
            continue
        assert cli._refused_as(args) is None
    monkeypatch.setenv(character.ENV, "magician")
    assert "only the thief hunts so far" in cli._refused_as(parser.parse_args(["tour"]))
    assert cli._refused_as(parser.parse_args(["probe", "list"])) is None  # offline
    assert cli._refused_as(parser.parse_args(["character"])) is None
    assert cli.main(["tour"]) == 2


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
    assert claw["desc_range"] == magician.BOX.far == 300 and "ball" not in claw and "range" not in claw
    assert (bolt["range"], bolt["desc_range"], bolt["ball"]) == (350, 350, True)
    assert l7["desc_range"] == 200  # Lucky Seven's "within 200" (the claw's; Keen Eyes adds the rest: reach.STAR_RANGE)
    assert tuple(claw["levels"][k]["mpCon"] for k in range(1, 21)) == magician.MAGIC_CLAW_MP
