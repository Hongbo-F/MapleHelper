"""The magician hunting with her profile, the quick way (2026-10-29: the user, "I need it ASAP to start grinding on
magician"): her keys (Magic Claw on Shift, her chair '-'), Magic Claw's search as the attack layer's constants
(gamelab.attack.magic: 50..300 ahead, no claw up close, nothing stopping it; the thief's back after), never an attack
keyed in the air, never her regular attack (no wand swing at range, none to finish, none for MP), out of MP as her
stop, and who may run what (`_refused_as`, the perception service's character); and (2026-10-29: the user, "Why you
were standing on platform 18 and cast magic claw for nothing, wasting mp?") her casts only where they can do something:
at a body seen that her search picks, not dead, not again where one took no effect, and up close no point-blank answer
that goes out as a cast picking nothing (behavior.attacks.Attacks.picks, attack.stance.POINT_BLANK)."""

import dataclasses
import time
from types import SimpleNamespace

import pytest
from maplehelper.engine import LEFT, RIGHT

from gamelab import character
from gamelab.attack import magic, reach
from gamelab.attack.choice import Attack
from gamelab.behavior import darts
from gamelab.behavior.attacks import Attacks
from gamelab.classes import magician
from gamelab.control.actions import Actions
from gamelab.localization.mapdata import MobBody

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
             stance.PASSING_JUMPS, stance.ESCAPE_UP, reach.can_hit.__defaults__, reach.STAR_SURE, stance.POINT_BLANK)
    assert stance.ESCAPE_UP and stance.POINT_BLANK  # (the thief's escape up, and her point-blank answers, as on master)
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
        assert not stance.POINT_BLANK  # (nor a jump throw up close: Policy.close_quarters 'touch')
        with character.using(MAGE):  # (no jump cover for an attack she doesn't make from a jump)
            assert not fight._air() and fight.jump_cover(None, _look([]), [], []) is None
        assert fight._air()  # (the thief's)
        assert reach.claw_edge(half) == 50 + half + reach.CLAW_MARGIN  # (her decisions keep one that far off)
    assert not magic.applied()
    assert thief == (reach.STAR_RANGE, reach.SEARCH_START, reach.SHOT_RANGE, reach.WALLS_BLOCK, reach.CLAW,
                     spots.SHOT_RANGE, stance.PASSING_JUMPS, stance.ESCAPE_UP, reach.can_hit.__defaults__,
                     reach.STAR_SURE, stance.POINT_BLANK)
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



# --- her casts only where they can do something (2026-10-29: the user, "Why you were standing on platform 18 and cast
# magic claw for nothing, wasting mp?"; behavior.attacks.Attacks.picks, attack.stance.POINT_BLANK) -------------------

TREE = SimpleNamespace(mob_hp={"Green Mushroom": 233.0, "Horny Mushroom": 403.0},
                       mob_pdd={"Green Mushroom": 0.0, "Horny Mushroom": 10.0},
                       mob_mdd={"Green Mushroom": 30.0, "Horny Mushroom": 0.0},  # (Tree Dungeon's game data)
                       mob_bodies={"Green Mushroom": MobBody(51.0, 56.0, 28.0, 0.84),
                                   "Horny Mushroom": MobBody(51.0, 56.0, 28.0, 0.84)})


def _picking(picks: bool = True, kills: list | None = None):
    """Her Attacks as her Hunter makes them (Policy.unseen 'none': `picks`), on Tree Dungeon's kinds; `picks` False: as
    on master."""
    said = []
    claw = MAGE.attacks["skill"]
    a = Attacks(TREE, MAGE.keys, "skill", max_mp=MAGE.stats.mp, level=0, log=said.append,
                kills=(lambda: kills) if kills is not None else None,
                skill=Attack("magic_claw", claw.hits, claw.lo, claw.hi, claw.mp), magic=True, no_regular=True,
                picks=picks)
    return a, said


def _cast(a: Attacks, at: float | None = None, side: str = RIGHT) -> bool:
    """A throw that way: its key asked (Actions' choose), and if it's a cast, its key told down at `at` (on_throw; the
    last look's time by default)."""
    if a.key(side) is None:
        return False
    a.on_throw(a._last.t if at is None else at, side)
    return True


def test_her_hunter_casts_only_where_her_search_picks_a_body_the_thiefs_as_before(monkeypatch, tmp_path):
    from gamelab import paths
    from gamelab.behavior.hunter import Hunter
    from gamelab.localization import mapdata
    from test_fight import EMPTY_MAP, _Eyes
    monkeypatch.setattr(paths, "LAB", tmp_path)
    assert MAGE.policy.unseen == "none" and character.load("thief").policy.unseen == "regular"
    with character.using(MAGE):
        h = Hunter(_Eyes([]), MAGE.keys, mapdata.parse(1, EMPTY_MAP, "Test", {}), log=lambda line: None)
    assert h.attacks.picks
    t = Hunter(_Eyes([]), dataclasses.replace(MAGE.keys, magic_claw=None, lucky_seven=MAGE.keys.magic_claw),
               mapdata.parse(1, EMPTY_MAP, "Test", {}), log=lambda line: None, attack="choice")
    assert not t.attacks.picks


def test_no_cast_at_nothing_nor_at_one_her_looks_dont_see():
    """Nothing is cast at nothing (Policy.unseen 'none': the thief's 'skill' mode cast whenever the MP paid, 'nothing
    seen in its way'), nor at a monster the look didn't see: live at 01:55:45 a Green Mushroom 110 ahead at 22% (~51
    HP) took a cast, and at 01:55:46-47 the looks didn't see it ('+94 hidden': most likely dead, its death unseen) and
    two more went at where her memory had it, its reckoning 51 HP each time; at 01:56:02-04 three at one '+28 hidden'
    whose kill had been placed 1.5 s before."""
    with magic.swapped(MAGE):
        a, said = _picking()
        t = time.monotonic()
        a.on_look(_look([], t=t))
        assert a.key(RIGHT) is None and a.counts["no cast: nothing her search picks"] == 1
        assert any("no cast right: nothing her search picks" in line for line in said)
        a.on_look(_look([("green-mushroom", 110.0, 0.22)], t=t + 0.5))
        assert _cast(a)
        ghost = _look([("green-mushroom", 94.0, 0.22)]).monsters[0]
        a.unseen = lambda s: [ghost]  # (the hunt loop's: the feed's hidden tracks, her memory's)
        a.on_look(_look([], t=t + 1.4))
        assert a.key(RIGHT) is None and a.key(RIGHT) is None and a.counts["magic_claw"] == 1
        assert a.target(RIGHT) is None
        # As on master (picks off): the one she last cast at that way within HIDDEN_FOR, and those about, take casts
        b, _ = _picking(picks=False)
        b.on_look(_look([("green-mushroom", 110.0, 0.22)], t=t + 0.5))
        assert _cast(b)
        b.unseen = lambda s: [ghost]
        b.on_look(_look([], t=t + 1.4))
        assert b.key(RIGHT) == "shift" and b.key(RIGHT) == "shift"


def test_no_cast_inside_her_searchs_first_50_or_behind_her_by_the_bodies_game_data():
    """Her search picks a body whose far side is past 50 ahead (the client's: a Green Mushroom 28 either side, the game
    data, from 22 ahead; reach's table takes it for a pig's 34), never one behind her."""
    with magic.swapped(MAGE):
        a, _ = _picking()
        a.on_look(_look([("green-mushroom", 20.0, None)]))
        assert a.key(RIGHT) is None and a.counts["no cast: nothing her search picks"] == 1
        a.on_look(_look([("green-mushroom", -120.0, None)]))
        assert a.key(RIGHT) is None and a.counts["no cast: nothing her search picks"] == 2  # (behind her)
        a.on_look(_look([("green-mushroom", 25.0, None)]))
        assert a.key(RIGHT) == "shift"  # (its far side 53 ahead)
        b, _ = _picking(picks=False)
        b.on_look(_look([("green-mushroom", 20.0, None)]))
        assert b.key(RIGHT) == "shift"  # (master's: reach's 34 either side, 54)


def test_after_a_cast_that_should_have_killed_it_none_until_her_exp_settles_it():
    """A Green Mushroom at ~51 HP takes a cast (2 hits of 76-267 against its 30 MDD: dead 72.5% of the time, a miss the
    rest): until her EXP says it didn't die (KILL_SHOWN_S with no gain of its kind) it's likely dead and nothing is cast
    at it; settled alive and seen with its bar as before, the cast took no effect (rule 3, below)."""
    with magic.swapped(MAGE):
        a, said = _picking()
        t = time.monotonic()
        a.on_look(_look([("green-mushroom", 110.0, 0.22)], t=t))
        assert _cast(a)
        a.on_look(_look([("green-mushroom", 108.0, 0.22)], t=t + 0.3))  # (before its EXP shows)
        assert a.key(RIGHT) is None and a.counts["no cast: likely dead"] == 1
        a.heard(t + 0.75, [])  # (her EXP read up to then: no gain)
        a.on_look(_look([("green-mushroom", 106.0, 0.22)], t=t + 0.75))
        assert a.key(RIGHT) is None and a.counts["no cast: no effect"] == 1 and a.counts["held: no effect"] == 1


def test_a_cast_that_took_no_effect_isnt_repeated_until_she_or_it_moves_or_its_bar_changes():
    """Live on platform 18 at 01:55:17-41: a Green Mushroom 106-150 ahead, its bar never up, 26 casts in 24 s. Judged at
    the first look LANDS_S after the key that sees it: still no bar up, the mark is held; a look that has it HOLD_MOVED
    nearer or farther, her HOLD_MOVED along, or its bar up lifts it."""
    with magic.swapped(MAGE):
        from gamelab.behavior import attacks as attacks_mod
        a, said = _picking()
        t = time.monotonic()
        a.on_look(_look([("green-mushroom", 132.0, None)], t=t))
        assert _cast(a)
        a.heard(t + 0.7, [])
        a.on_look(_look([("green-mushroom", 128.0, None)], t=t + 0.7))
        assert a.key(RIGHT) is None and a.counts["held: no effect"] == 1
        assert any("took no effect" in line for line in said)
        a.on_look(_look([("green-mushroom", 120.0, None)], t=t + 1.5))  # (8 nearer)
        assert a.key(RIGHT) is None and a.counts["no cast: no effect"] == 2
        # ... and the fights wait (behavior.fight.cast_or_wait: no throw her cast would send nothing with), the thief's
        # throw as the stance says
        from gamelab.behavior import fight
        assert fight.cast_or_wait(SimpleNamespace(attacks=a), ("throw", RIGHT)) == ("wait",)
        assert fight.cast_or_wait(SimpleNamespace(attacks=a), ("back", LEFT, 60.0)) == ("back", LEFT, 60.0)
        assert fight.cast_or_wait(SimpleNamespace(attacks=_picking(picks=False)[0]), ("throw", RIGHT)) == ("throw",
                                                                                                          RIGHT)
        a.on_look(_look([("green-mushroom", 128.0 + attacks_mod.HOLD_MOVED, None)], t=t + 2.4))
        assert a.counts["held: lifted"] == 1 and _cast(a)
        # Judged again: none; its bar up later (another's damage, or hers shown late) lifts it
        a.heard(t + 3.1, [])
        a.on_look(_look([("green-mushroom", 150.0, None)], t=t + 3.1))
        assert a.key(RIGHT) is None and a.counts["held: no effect"] == 2
        a.on_look(_look([("green-mushroom", 150.0, 0.6)], t=t + 3.8))
        assert a.counts["held: lifted"] == 2 and a.key(RIGHT) == "shift"
        summary = a.summary()
        assert "no cast: no effect 3" in summary and "her casts that took no effect 2 (holds lifted 2)" in summary
        # An effect is no hold: its bar lower after the key
        b, _ = _picking()
        b.on_look(_look([("green-mushroom", 200.0, 0.8)], t=t))
        assert _cast(b)
        b.heard(t + 0.8, [])
        b.on_look(_look([("green-mushroom", 196.0, 0.3)], t=t + 0.8))
        assert b.counts["held: no effect"] == 0 and b.key(RIGHT) == "shift"
        # Her attack's MISS read over it (keycheck's, Hunter._miss_points: her casts' targets among its boxes): her cast
        # picked it and missed, no hold
        m, _ = _picking()
        m.on_look(_look([("green-mushroom", 132.0, None)], t=t))
        assert _cast(m) and [o.x for o in m.boxes(t + 0.7)] == [132.0] and m.boxes(t + 0.2) == []
        m.heard(t + 0.7, [])
        missed = _look([("green-mushroom", 128.0, None)], t=t + 0.7)
        missed.misses = [(130.0, -40.0)]
        m.on_look(missed)
        assert m.counts["held: no effect"] == 0 and m.key(RIGHT) == "shift"
        # Her moved HOLD_MOVED along: lifted too
        c, _ = _picking()
        c.on_look(_look([("green-mushroom", 132.0, None)], t=t))
        assert _cast(c)
        c.heard(t + 0.7, [])
        c.on_look(_look([("green-mushroom", 128.0, None)], t=t + 0.7))
        assert c.key(RIGHT) is None
        moved = _look([("green-mushroom", 128.0, None)], t=t + 1.5)
        moved.where.x = -attacks_mod.HOLD_MOVED  # (she stepped back a band: it's 148 ahead of her)
        c.on_look(moved)
        assert c.counts["held: lifted"] == 1 and c.key(RIGHT) == "shift"


def test_a_kills_body_takes_no_cast_while_it_fades():
    """A kill of hers placed on the mark she cast at: its body, drawn dying (0.84 s for a mushroom) and fading, takes no
    cast for FADE_S (ledger.dying's window ends with the animation, from the look that placed it), a live one beside
    it does."""
    with magic.swapped(MAGE):
        from gamelab.behavior.ledger import Kill
        kills = []
        a, _ = _picking(kills=kills)
        t = time.monotonic()
        a.on_look(_look([("green-mushroom", 100.0, 0.3)], t=t))
        assert _cast(a)
        kills.append(Kill(t - 1.0, "green-mushroom", 1, 100.0, 0.0, 1))  # (placed: ledger.dying's window over at once)
        a.heard(t + 0.6, [(t + 0.35, frozenset({"green-mushroom"}))])
        a.on_look(_look([("green-mushroom", 102.0, None)], t=t + 0.6))
        assert a.key(RIGHT) is None and a.counts["no cast: nothing her search picks"] == 1
        a.on_look(_look([("green-mushroom", 102.0, None), ("green-mushroom", 180.0, None)], t=t + 0.9))
        assert a.key(RIGHT) == "shift" and a._target.x == 180.0
        b, _ = _picking(picks=False, kills=[])
        b.on_look(_look([("green-mushroom", 102.0, None)], t=t + 0.9))
        assert b.key(RIGHT) == "shift"  # (master's: no kill near it within its window)


def _close_scene(her_x: float, monsters: list[tuple[str, float]], wall_left: bool = False):
    """A long platform (0..1000 at y 500) and Tree Dungeon's kinds at these x on it (test_danger's map); `wall_left`: a
    wall rising from its left end (no room behind her there)."""
    from test_danger import build, seen
    footholds = [(0, 500, 1000, 500)] + ([(0, 400, 0, 500)] if wall_left else [])
    model = build(footholds, [("13", 900, 500), ("19", 950, 500)])
    p = model.platform_under(500, 500)
    return model, p, [seen(kind, x, 500, p.id) for kind, x in monsters]


def _her_stance(model, p, x, mine, facing=RIGHT, castable=None):
    from gamelab.attack import stance
    from gamelab.localization.monsters import bodies
    tops = {k: b.top for k, b in bodies(model).items()}
    halves = {k: b.half for k, b in bodies(model).items()}
    return stance.stance(x, 500, p, mine, True, facing, can_jump=lambda a, b: stance.open_above(model, p, a, b),
                         tops=tops, halves=halves, around=mine, castable=castable)


def test_up_close_she_steps_or_waits_never_a_jump_throw_that_goes_out_as_a_cast_picking_nothing():
    """Live at 01:56:02 'jump-throwing right at 40 ms, tap first (green-mushroom +28 hidden)' and 01:55:51 at one +43:
    the thief's point-blank answers went out for her as casts from the ground (Actions.ground_only). Hers (POINT_BLANK
    off): a cast from the ground where her search surely picks it when the cast starts, else a step to where it will,
    the jump away, a cast at another her search picks, or a wait."""
    from gamelab.attack import stance
    model, p, mine = _close_scene(300, [("green-mushroom", 315.0)])
    thief = _her_stance(model, p, 300, mine)
    with magic.swapped(MAGE):
        assert not stance.POINT_BLANK
        act = _her_stance(model, p, 300, mine)
        assert act[0] == "back" and act[1] == LEFT, act  # (a step away: from 93 off a cast from the ground takes it)
        model, p, mine = _close_scene(40, [("green-mushroom", 55.0)], wall_left=True)
        act = _her_stance(model, p, 40, mine)
        assert act[0] not in ("jump_throw", "throw"), act  # (cornered: no cast at it, inside her search's first 50)
        model, p, mine = _close_scene(40, [("green-mushroom", 55.0), ("green-mushroom", 200.0)], wall_left=True)
        assert _her_stance(model, p, 40, mine) == ("throw", RIGHT)  # (one her search picks that way: cast at it)
        model, p, mine = _close_scene(300, [("green-mushroom", 360.0)])
        assert _her_stance(model, p, 300, mine) == ("throw", RIGHT)  # (60 ahead: her search surely picks it)
    assert stance.POINT_BLANK
    model, p, mine = _close_scene(300, [("green-mushroom", 315.0)])
    assert _her_stance(model, p, 300, mine) == thief  # (the thief's, as on master)


def test_the_stance_throws_at_none_her_cast_wont_go_at_but_still_keeps_off_it():
    """`castable` (behavior.fight.castable_of: Attacks.refused): a held or unseen one in reach is no throw ('wait' with
    nothing else), and up close she still steps away from it (her escapes stay)."""
    with magic.swapped(MAGE):
        model, p, mine = _close_scene(300, [("green-mushroom", 450.0)])
        assert _her_stance(model, p, 300, mine)[0] == "throw"
        assert _her_stance(model, p, 300, mine, castable=lambda m: False) == ("wait",)
        model, p, mine = _close_scene(300, [("green-mushroom", 330.0)])
        assert _her_stance(model, p, 300, mine, castable=lambda m: False)[0] == "back"
