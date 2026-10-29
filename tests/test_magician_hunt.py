"""The magician hunting with her profile, the quick way (2026-10-29: the user, "I need it ASAP to start grinding on
magician"): her keys (Magic Claw on Shift, her chair '-'), Magic Claw's search as the attack layer's constants
(gamelab.attack.magic: 50..300 ahead, no claw up close, nothing stopping it; the thief's back after), never an attack
keyed in the air, never her regular attack (no wand swing at range, none to finish, none for MP), out of MP as her
stop, and who may run what (`_refused_as`, the perception service's character)."""

import dataclasses
import time

import pytest
from maplehelper.engine import RIGHT

from gamelab import character
from gamelab.attack import magic, reach
from gamelab.attack.choice import Attack
from gamelab.behavior import darts
from gamelab.behavior.attacks import Attacks
from gamelab.classes import magician
from gamelab.control.actions import Actions

from test_choice import MODEL, _look

MAGE = magician.profile()


def test_her_keys_are_her_profiles_magic_claw_on_shift_her_chair_on_minus():
    from gamelab.control import keymap, timeline
    with character.using(MAGE):
        cfg, keys, _ = timeline.her_setup(remembered=False)
        assert keymap.path_of().name == "keybindings-magician.json"  # (her own remembered binding: keys --read)
    assert keys == cfg.keys == MAGE.keys
    assert (keys.magic_claw.name, keys.lucky_seven, keys.chair.name, keys.attack.name) == ("shift", None, "-", "ctrl")
    assert (keys.pet_food.name, keys.jump.name, keys.energy_bolt.name) == ("u", "alt", "a")
    thief = timeline.her_setup(remembered=False)[1]  # (the thief's: config.toml's flat [keys], as before)
    assert thief.magic_claw is None and thief.lucky_seven is not None
    # Her keys followed off the game's window: Magic Claw's picture is one it knows, and on no key she stops
    assert keymap.FUNCTION_OF["magic_claw"] == "MAGIC CLAW" and "magic_claw" in keymap.NEEDED


def test_her_skill_key_is_magic_claw_s_and_her_retreat_has_none_of_her_skills():
    sent = []
    a = Actions(MAGE.keys, skill=True)
    a.play = lambda events, name, **kw: sent.append((name, events))
    assert a._attack_key(RIGHT, ground=True) == "shift"
    a.choose = lambda side: "shift"
    assert a._attack_key(RIGHT) == "shift"
    moves = a.moves_only().keys
    assert moves.magic_claw is None and moves.energy_bolt is None and moves.jump.name == "alt"


def test_from_the_ground_only_no_attack_key_goes_in_a_jump():
    """Magic Claw isn't cast mid-jump (the user, 2026-10-27): with `ground_only` a jump throw is a throw from the
    ground (no jump key), a jump back the jump away alone (no attack key)."""
    sent = []
    a = Actions(MAGE.keys, skill=True)
    a.play = lambda events, name, **kw: sent.append((name, events))
    a.ground_only = True
    assert a.jump_throw(RIGHT) and a.jump_back(RIGHT)
    (n1, e1), (n2, e2) = sent
    assert n1 == "throw" and "alt" not in {e[1] for e in e1} and "shift" in {e[1] for e in e1}
    assert n2 == "jump_back" and "alt" in {e[1] for e in e2} and "shift" not in {e[1] for e in e2}
    # The thief's as before: a jump throw jumps and attacks
    sent.clear()
    t = Actions(dataclasses.replace(MAGE.keys, magic_claw=None, lucky_seven=MAGE.keys.magic_claw), skill=True)
    t.play = lambda events, name, **kw: sent.append((name, events))
    assert t.jump_throw(RIGHT)
    assert sent[0][0] == "jump_throw" and {"alt", "shift"} <= {e[1] for e in sent[0][1]}


def test_magic_claws_search_in_the_attack_layer_and_the_thiefs_back_after():
    """Her search box: from 50 ahead (a body's far side past it) to 300 (its near side short of it), nearest band
    first, never behind her; no claw up close; nothing stops it; no jump shots planned. Restored: the thief's."""
    from gamelab.attack import spots, stance
    from gamelab.behavior import fight
    thief = (reach.STAR_RANGE, reach.SEARCH_START, reach.SHOT_RANGE, reach.WALLS_BLOCK, reach.CLAW, spots.SHOT_RANGE,
             stance.PASSING_JUMPS, stance.ESCAPE_UP, reach.can_hit.__defaults__, reach.STAR_SURE)
    assert stance.ESCAPE_UP  # (the thief's escape up onto a platform overhead, as on master)
    top, half = reach.body_of("green-mushroom")
    claw_near = reach.claw_chance(40, 0, top, half)
    assert claw_near > 0 and reach.can_shoot(380, 0, reach=434, kind="green-mushroom")  # (the thief's: star to 400)
    with magic.swapped(MAGE) as done:
        assert done and magic.applied()
        assert (reach.STAR_RANGE, reach.SEARCH_START, reach.BANDS, reach.SHOT_RANGE) == (300, 50, 13, 310)
        assert [reach.window(k) for k in (0, 12)] == [(16, 40), (-44, 100)]  # the range doc's first and last windows
        assert reach.claw_chance(40, 0, top, half) == 0 and reach.claw_chance(5, 0, top, half) == 0
        assert reach.can_shoot(200, 0, kind="green-mushroom") and reach.can_shoot(300, 0, kind="green-mushroom")
        assert not reach.can_shoot(300 + half + 1, 0, reach=999, kind="green-mushroom")  # past her search
        assert not reach.can_shoot(50 - half - 1, 0, kind="green-mushroom")  # in her search's first 50
        assert reach.min_dx(0, top, half) == 50 - half  # nearest band first: its far side just past 50
        assert reach.can_shoot(250, -90, kind="green-mushroom") and not reach.can_shoot(100, -90, kind="green-mushroom")
        s = _look([("green-mushroom", 200.0, None)])
        assert not reach.WALLS_BLOCK and reach.wall_to(s.platform, 0.0, 0.0, s.monsters[0]) is None  # (nothing stops it)
        assert spots.SHOT_RANGE == 310 and not stance.PASSING_JUMPS
        assert not stance.ESCAPE_UP  # (her escape never a jump up: it would be a cast from the ground beside it)
        with character.using(MAGE):  # (no jump cover for an attack she doesn't make from a jump)
            assert not fight._air() and fight.jump_cover(None, _look([]), [], []) is None
        assert fight._air()  # (the thief's)
        assert reach.claw_edge(half) == 50 + half + reach.CLAW_MARGIN  # (her decisions keep one that far off)
    assert not magic.applied()
    assert thief == (reach.STAR_RANGE, reach.SEARCH_START, reach.SHOT_RANGE, reach.WALLS_BLOCK, reach.CLAW,
                     spots.SHOT_RANGE, stance.PASSING_JUMPS, stance.ESCAPE_UP, reach.can_hit.__defaults__,
                     reach.STAR_SURE)
    assert reach.claw_chance(40, 0, top, half) == claw_near
    assert not magic.apply(character.load("thief"))  # (nothing for the thief)


def test_her_escape_is_never_the_jump_up_onto_a_platform_overhead():
    """test_danger's scene (a Horny Mushroom 60 off on her platform, another platform 60 over her): the thief escapes up
    onto it throwing back as she rises; she can't cast in a jump, so her escape is anything but that jump (as a cast
    from the ground it would leave her beside the monster her close quarters ruled too near)."""
    from gamelab.attack import stance
    from test_danger import HER, Danger, build, seen
    model = build([(0, 500, 1000, 500), (0, 440, 400, 440)], [("19", 300, 500)])
    her, over = model.platform_under(700, 500), model.platform_under(200, 440)
    mine = seen("horny-mushroom", 260, 500, her.id)

    def act():
        return stance.stance(200, 500, her, [mine], around=[mine], danger=Danger(model, HER),
                             can_jump=lambda a, b: stance.open_above(model, her, a, b),
                             up=lambda x: stance.overhead(model, her, x), edges=stance.drop_limits(model, her))
    thief = act()
    assert thief[0] == "jump_throw" and stance.plan_of(thief).why == f"up onto {over.id}"  # (the thief's, as on master)
    with magic.swapped(MAGE):
        hers = act()
        plan = stance.plan_of(hers)
        assert not (hers[0] == "jump_throw" and plan is not None and plan.why.startswith("up onto")), hers


def _her_attacks():
    said = []
    claw = MAGE.attacks["skill"]
    a = Attacks(MODEL, MAGE.keys, "skill", max_mp=MAGE.stats.mp, level=0, log=said.append,
                skill=Attack("magic_claw", claw.hits, claw.lo, claw.hi, claw.mp), magic=True, no_regular=True)
    return a, said


def test_she_casts_magic_claw_whenever_her_mp_pays_and_never_her_regular_attack():
    a, said = _her_attacks()
    assert a.skill.name == "Magic Claw" and a.skill.mp == 20.0
    a.on_look(_look([("green-mushroom", 200.0, None)]))
    assert a.key("right") == "shift"  # Magic Claw
    a.on_look(_look([("green-mushroom", 200.0, 0.1)]))  # (low: the thief would finish it with a regular star)
    assert a.key("right") == "shift"
    a.on_look(_look([("green-mushroom", 200.0, 0.5)], mp=15 / 1007))  # under a cast: no wand swing for 'MP'
    assert a.key("right") is None
    assert a.counts["attack"] == 0 and a.counts["magic_claw"] == 2 and a.counts["no cast: MP"] == 1
    assert any("no cast (MP)" in line for line in said)
    assert a.summary().startswith("Magic Claw 2, the regular attack 0 (no cast: MP 1)")


def test_out_of_mp_when_her_mp_stays_under_two_casts():
    w = darts.MpWatch(20.0, 1007.0)
    t = 1000.0
    for k in range(13):  # 12 s of reads under 40 MP ...
        w.see(t + k, 30 / 1007)
        if k == 6:
            w.see(t + k + 0.5, 600 / 1007)  # ... but her pet drank half way: none
    assert w.due(t + 13) is None
    for k in range(13):
        w.see(t + 20 + k, 0.02)  # 20 MP for 12 s: out of MP potions
    stop = w.due(t + 33)
    assert isinstance(stop, darts.OutOfMp) and stop.exit_code == darts.OUT_OF_MP_EXIT and "OUT OF MP" == stop.label
    assert w.due(t + 40) is None  # (once)


def test_she_hunts_tours_and_travels_but_not_the_thiefs_probes_locks_or_combos(monkeypatch):
    from gamelab import __main__ as cli
    parser = cli.build_parser()
    monkeypatch.setenv(character.ENV, "magician")
    monkeypatch.setattr(character, "_ACTIVE", [])
    for argv in (["tour"], ["hunt", "kill"], ["nav", "--to", "1"], ["perceive"], ["pet", "--every", "1"]):
        try:
            args = parser.parse_args(argv)
        except SystemExit:
            continue
        assert cli._refused_as(args) is None, argv
    for argv in (["probe", "run", "claw-edge"], ["lock", "attack"]):
        try:
            args = parser.parse_args(argv)
        except SystemExit:
            continue
        refused = cli._refused_as(args)
        assert refused is not None and "only the thief runs it so far" in refused, argv
    # --character after the command's name (the babysitter's process match sees `gamelab tour`)
    assert parser.parse_args(["tour", "--character", "magician"]).character == "magician"
    assert parser.parse_args(["--character", "thief", "tour"]).character == "thief"


def test_a_hunt_refuses_a_service_perceiving_another_character():
    from gamelab import __main__ as cli
    with character.using(MAGE):
        cli._perceives_her({"character": "magician"}, hunts=True)
        with pytest.raises(RuntimeError, match="perceives thief, not magician.*character use magician"):
            cli._perceives_her({}, hunts=True)  # (a service of the code before: the thief's)
        cli._perceives_her({"character": "thief"}, hunts=False)  # (the pet's feeder: said, goes on)
    cli._perceives_her({}, hunts=True)  # the thief's with the thief's: as before


def test_her_hunter_casts_from_the_ground_watches_her_mp_and_takes_her_shape(monkeypatch, tmp_path):
    """Hunter for the magician (her profile active): 'skill' whatever was asked, ground only, Magic Claw her skill,
    no regular attack, out of darts never, her MP watched, her Shape in her senses."""
    from gamelab import paths
    monkeypatch.setattr(paths, "LAB", tmp_path)  # (her own files: lab/characters/magician/)
    from gamelab.behavior.hunter import Hunter
    from gamelab.localization import mapdata
    from test_fight import EMPTY_MAP, _Eyes
    said = []
    with character.using(MAGE):
        h = Hunter(_Eyes([]), MAGE.keys, mapdata.parse(1, EMPTY_MAP, "Test", {}), log=said.append, attack="choice")
    assert h._attack_mode == "skill" and h.actions.ground_only and h.actions.skill
    assert h.attacks.skill.key == "magic_claw" and h.attacks.skill.mp == 20.0 and h.attacks.no_regular
    assert h.mp_watch is not None and h.darts.presses == 10 ** 9 and h.mp_watch.max_mp == h.max_mp
    assert h.senses.shape is MAGE.shape
    assert any("her attack: skill (Magic Claw whenever her MP pays for a cast), not choice" in line for line in said)
    # The thief's as before: her mode as asked, jump throws, out of darts, no MP watch, her Shape
    t = Hunter(_Eyes([]), dataclasses.replace(MAGE.keys, magic_claw=None, lucky_seven=MAGE.keys.magic_claw),
               mapdata.parse(1, EMPTY_MAP, "Test", {}), log=lambda line: None, attack="choice")
    assert t._attack_mode == "choice" and not t.actions.ground_only and t.mp_watch is None
    assert t.attacks.skill.key == "lucky_seven" and not t.attacks.no_regular and t.darts.presses < 10 ** 9
    assert t.senses.shape is character.THIEF_SHAPE
