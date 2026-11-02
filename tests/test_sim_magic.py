"""The magician in the simulator (from 2026-11-02, the roadmap's item 8): Magic Claw as the game casts it (world.SimMagic,
Her.magic), and her as the day's recordings measured her (world.MAGE_32), on Tree Dungeon's real platforms and monsters.
A fact a test: the client's search by its bands from 50 to 300 ahead, nothing stopping it, from the ground only, 2 casts
a Horny Mushroom and 1 a Green, 20 MP a cast and the game's refusal under it, its lock, when its hit shows, her pet's
potions and her MP's own rise, her touches. The thief's world is the goldens' to check (none of this is his)."""

import pytest

from gamelab.control.timeline import SIM_KEYS_MAGE, SIM_KEYS_PET
from gamelab.sim import scenarios
from gamelab.sim import world as world_mod
from gamelab.sim.world import (CLAW_HIT, CLAW_LOCK_S, CLAW_SHOWS_S, MAGE_32, MEAT_HP, MP_REFUSAL, STAR_MEAN, Noise,
                               World, evade_chance, touch_damage)

TREE = 10002073
HORNY, GREEN = "Horny Mushroom", "Green Mushroom"


def tree_map():
    m = scenarios.real_map(TREE)
    if m is None:
        pytest.skip("needs Tree Dungeon, Forest Up North II's game data (gamedata/data)")
    return m


def her_world(monsters, start=(33, 0.0), facing="right", **kw):
    """Her (MAGE_32) on Tree Dungeon at `start`, with `monsters` [(name, platform, x)] and nothing else."""
    return World(tree_map(), SIM_KEYS_MAGE, start, monsters=monsters, character=kw.pop("character", MAGE_32),
                 facing=facing, **kw)


def cast(w, t, key="shift"):
    w.key(t, key, True)
    w.key(t + 0.05, key, False)


def test_her_profile_is_the_recordings():
    # Her stat window (level 32: HP 858, MP 1549, W.DEF 75, AVOID 16), her walk (148.2), her pet's Meat (160) under 75%,
    # its MP potion (187) under 6%, her MP's rise (2.4 a second), Magic Claw's 20 MP: world.MAGE_32's comment says where
    h = MAGE_32
    assert (h.max_hp, h.level, h.mp, h.defence, h.avoid) == (858.0, 32, 1549.0, 75.0, 16.0)
    assert (h.walk_speed, h.potion, h.pet, h.mp_potion, h.pet_mp, h.mp_regen, h.cast_mp) == \
        (148.2, MEAT_HP, 0.75, 187.0, 0.06, 2.4, 20.0)
    assert MEAT_HP == 160.0 and h.magic.hit == CLAW_HIT and h.magic.hits == 2 and h.magic.lock_s <= 0.80
    assert h.magic.search.start == 50 and h.magic.search.range == 300 and h.magic.search.bands == 13


def test_her_touches_all_hurt_as_much_as_live():
    # At AVOID 16 the game's hit check lets no touch MISS (the hour's 170 touches, 0 MISS); at DEF 75 a Horny Mushroom
    # takes 110-130 HP and a Green 48-58 (live 110-130 and 50-58)
    w = her_world([(HORNY, 33, 100.0), (GREEN, 33, -100.0)])
    for m, (lo, hi) in zip(w.monsters, ((110, 130), (48, 58))):
        assert 1.0 - evade_chance(m.info, MAGE_32, w._mob_levels[m.kind]) == pytest.approx(1.0)
        taken = [round(touch_damage(m.info, u / 100, MAGE_32) * 858) for u in range(100)]
        assert min(taken) >= lo - 2 and max(taken) <= hi + 2


@pytest.mark.parametrize("ahead, picked", [(-100.0, False), (10.0, False), (21.0, False), (23.0, True),
                                           (200.0, True), (320.0, True), (329.0, False)])
def test_her_search_picks_from_its_first_50_to_300_ahead(ahead, picked):
    # Never behind her, nor a body whose far side is within 50 of her (a Horny is 28 either side: its middle 22 ahead or
    # nearer); out to a body whose near side is short of 300 (its middle 328 or nearer), the overlap strict
    w = her_world([(HORNY, 33, -60.0 + ahead)], start=(33, -60.0))  # (all inside 33's walk: -200 to 290)
    assert (w.magic_pick("right") is w.monsters[0]) is picked


def test_the_nearest_band_any_body_overlaps_picks():
    w = her_world([(GREEN, 33, 250.0), (HORNY, 33, 120.0)])
    assert w.magic_pick("right").kind == "horny-mushroom"
    w = her_world([(GREEN, 33, -150.0), (HORNY, 33, 120.0)], facing="left")
    assert w.magic_pick("left").kind == "green-mushroom"


def test_nothing_stops_it_where_the_stars_line_is():
    # Her on 9 at x -50, a Green Mushroom on 10 at x 200, 58 below her: the star's line (v83) is stopped by 9's own floor,
    # Magic Claw's isn't (no line at all: the user's, settled) and kills it
    w = her_world([(GREEN, 10, 200.0)], start=(9, -50.0))
    g = w.monsters[0]
    assert w.sight_stop(g) is not None and w.magic_pick("right") is g
    cast(w, 0.0)
    w.advance(1.0)
    assert w.kills == {"green-mushroom": 1}


def test_a_press_in_the_air_does_nothing():
    w = her_world([(HORNY, 33, 150.0)])
    w.key(0.0, "alt", True)
    w.key(0.05, "alt", False)
    cast(w, 0.2)  # in the air (lifted off 65 ms after the jump key)
    w.advance(1.5)
    assert w.mode == "ground" and w.air_presses == 1 and w.casts == 0 and w.attacks == 0 and w.mp == 1.0
    assert w.monsters[0].hp == w.monsters[0].hp_max
    cast(w, 1.5)  # on the ground again: cast
    w.advance(2.0)
    assert w.casts == 1 and w.monsters[0].hp < w.monsters[0].hp_max


def test_two_casts_a_horny_mushroom_and_one_a_green():
    # R1-R3's kills: a Horny (403 HP, MDDamage 0) takes 2 casts, at most 362 a cast; a Green (233, MDDamage 30) 1, at
    # least 314 x 100 / 130 = 241
    for seed in range(1, 25):
        w = her_world([(HORNY, 33, 150.0)], seed=seed)
        cast(w, 0.0)
        w.advance(0.85)
        m = w.monsters[0]
        assert m.alive and 403 - 2 * CLAW_HIT[1] <= m.hp * STAR_MEAN <= 403 - 2 * CLAW_HIT[0]
        cast(w, 0.9)
        w.advance(1.5)
        assert not m.alive and w.kills == {"horny-mushroom": 1} and w.casts == 2
        w = her_world([(GREEN, 33, 150.0)], seed=seed)
        cast(w, 0.0)
        w.advance(0.5)
        assert w.kills == {"green-mushroom": 1} and w.casts == 1


def test_the_lock_takes_a_cast_from_0_80_s_on_and_her_stacks_pace():
    # A second cast keyed under CLAW_LOCK_S after the first is ignored (counted), one at it or later taken; the live
    # pace, 0.846 s back to back, never refused
    for gap, taken in ((0.75, False), (0.79, False), (0.80, True), (0.846, True)):
        w = her_world([(HORNY, 33, 150.0)])
        cast(w, 0.0)
        cast(w, gap)
        w.advance(2.0)
        assert w.casts == (2 if taken else 1) and w.attack_locked == (0 if taken else 1)
    assert CLAW_LOCK_S <= 0.80


def test_20_mp_a_cast_and_the_games_refusal_under_it():
    w = her_world([(HORNY, 33, 150.0)])
    cast(w, 0.0)
    w.advance(0.1)
    assert w.mp * 1549 == pytest.approx(1549 - 20 + 2.4 * 0.1, abs=0.01)
    w.mp = 15 / 1549  # (under a cast)
    cast(w, 1.0)
    w.advance(1.2)
    assert w.no_mp == 1 and w.casts == 1 and w.chat[1] == MP_REFUSAL and w.attack_lock_ms < w.ms
    cast(w, 1.25)  # refused: no lock, so the next press isn't held back by one (and is refused again)
    w.advance(1.4)
    assert w.no_mp == 2


def test_her_pet_drinks_an_mp_potion_under_6_percent_and_her_mp_rises_2_4_a_second():
    w = her_world([(HORNY, 33, 150.0)])
    w.mp = 110 / 1549
    cast(w, 0.0)  # 90 left, under 6% of 1549 (93): her pet drinks 187
    w.advance(0.1)
    assert w.mp_potions == 1 and w.mp * 1549 == pytest.approx(110 - 20 + 187 + 0.24, abs=0.01)
    w = her_world([])
    w.mp = 0.5
    w.advance(10.0)
    assert w.mp * 1549 == pytest.approx(1549 * 0.5 + 24.0, abs=0.01)


def test_her_pet_drinks_meat_under_75_percent():
    # A touch that leaves her under 75% of 858: her pet drinks Meat, 160 HP, at once (a touch's knock aside)
    w = her_world([(HORNY, 33, 0.0)], hp=0.80)
    w.advance(0.3)
    assert w.touches >= 1 and w.pet_potions == 1
    taken = 0.80 - (w.hp - 160 / 858)
    assert 110 / 858 - 0.003 <= taken <= 130 / 858 + 0.003


def test_her_hit_shows_0_53_s_after_the_key_and_a_killed_one_is_drawn_alive_till_then():
    w = her_world([(GREEN, 33, 150.0)])
    cast(w, 0.0)
    w.advance(0.4)
    m = w.monsters[0]
    assert not m.alive and m.drawn_alive == pytest.approx(CLAW_SHOWS_S, abs=0.01)
    assert [v for v in w.visible_monsters() if v[0] is m][0][3]  # (drawn alive)
    w.advance(0.6)
    assert not [v for v in w.visible_monsters() if v[0] is m][0][3]  # (dying)
    w = her_world([(HORNY, 33, 150.0)])
    cast(w, 0.0)
    w.advance(0.5)
    assert w.monsters[0].hit_at is None and w.monsters[0].flinch_at is None
    w.advance(0.6)
    assert w.monsters[0].hit_at == pytest.approx(CLAW_SHOWS_S, abs=0.01) and w.monsters[0].flinch_at is not None


def test_a_monster_her_cast_hurt_is_lost_0_4_to_1_s_under_her_hit():
    # The looks lose it under her hit's effect (SimMagic.hidden) where the presets' star hides one 1-3 s
    from gamelab.sim.fakes import Clock, FakeEyes
    noise = Noise(hit_hidden=(1.0, 3.0))
    hidden = []
    for seed in range(1, 21):
        w = her_world([(HORNY, 33, 150.0)], seed=seed, noise=noise)
        eyes = FakeEyes(w, Clock(), w.model)
        cast(w, 0.0)
        t, lost = 0.6, None
        while t < 3.0:
            w.advance(t)
            if lost is None and eyes._unseen(w.monsters[0], True):
                lost = t
            elif lost is not None and not eyes._unseen(w.monsters[0], True):
                hidden.append(t - CLAW_SHOWS_S)
                break
            t += 0.05
    assert len(hidden) == 20 and 0.4 - 0.05 <= min(hidden) and max(hidden) <= 1.0 + 0.05


def test_her_wand_swings_at_nothing_and_her_keys_are_the_simulators():
    # Her regular attack (Ctrl) does nothing (her stack never keys it: Attacks.no_regular), counted; Shift is Magic Claw
    w = her_world([(HORNY, 33, 150.0)])
    cast(w, 0.0, "ctrl")
    w.advance(1.0)
    assert w.swings == 1 and w.casts == 0 and w.monsters[0].hp == w.monsters[0].hp_max
    assert SIM_KEYS_MAGE.magic_claw.name == "shift" and SIM_KEYS_MAGE.lucky_seven is None
    assert world_mod._actions(SIM_KEYS_MAGE)["shift"] == "skill"
    assert "magic_claw" not in {f for f, _ in world_mod.ACTIONS if getattr(SIM_KEYS_PET, f, None) is not None}


# --- the harness plays her (sim.harness --character magician) ----------------------------------------------------------

def test_her_on_tree_is_the_magician_where_the_process_plays_her():
    from gamelab.sim.world import LEVEL_25
    assert scenarios.her_on(TREE, "magician") == (SIM_KEYS_MAGE, MAGE_32)
    assert scenarios.her_on(TREE, "thief") == (SIM_KEYS_PET, LEVEL_25) == scenarios.her_on(TREE)  # (the tests: thief)
    assert scenarios.her_on(1, "magician") == scenarios.her_on(1)  # (hers on Tree Dungeon alone)


def test_the_harness_plays_whom_it_is_told():
    from gamelab.sim import harness
    assert harness._who(["run", "loop-grind", "--character", "magician"], {}) == ("magician", "")
    assert harness._who(["run", "loop-grind"], {"GAMELAB_CHARACTER": "magician"}) == ("magician", "")  # (`ab`'s filler)
    assert harness._who(["run", "loop-grind"], {}) == ("thief", "")
    assert harness._who(["run", "x", "--character", "magician"], {"GAMELAB_CHARACTER": "thief"})[0] is None
    assert harness._who(["run", "x", "--character", "warrior"], {})[0] is None
    sc = harness.scenario("loop-grind", 1, "crowd", map_id=TREE, world=("zero-walls",), attack="choice")
    assert sc.her.magic is None and sc.her.skill == world_mod.LUCKY_SEVEN and sc.book is None  # (the thief's, as ever)


def test_her_damage_book_is_her_stacks_as_the_hour_began():
    # Her stack's book when the hour of 2026-11-02 began: the fit its file had (her level-29 and 30 reads: 125-150 a hit),
    # her level-31 and 32 reads too few for another (3 exact of 15), so a Green Mushroom (303 before its defence) is
    # reckoned to outlive a cast as live (behavior.damage)
    from gamelab.behavior import damage
    book = damage.Book(scenarios.HER_BOOK, log=lambda line: None, level=lambda: 32)
    book.path = None  # (nothing saved)
    assert book.attack(damage.Attack("magic_claw", 2, 76, 267, 20.0)).lo == 125
    assert book.attack(damage.Attack("magic_claw", 2, 76, 267, 20.0)).hi == 150
    assert max(r.t for r in book.reads("magic_claw")) <= 1793607468.0  # (00:17:48, UTC-8)
    assert damage.fit(book.reads("magic_claw"), 2) is None


@pytest.mark.slow
def test_a_simulated_grind_of_hers_casts_from_the_ground_inside_her_lock_never():
    # The harness as her (a process of her own): her stack casts Magic Claw in her world, never in the air, never inside
    # the world's lock (0.80 s), never her wand; its results say whom they are of
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    tree_map()
    from gamelab.paths import REPO
    with tempfile.TemporaryDirectory() as tmp:
        name = Path(tmp) / "her.json"
        done = subprocess.run([sys.executable, "-m", "gamelab.sim.harness", "run", "loop-grind", "--character",
                               "magician", "--map", str(TREE), "--preset", "crowd", "--world", "zero-walls", "--seeds",
                               "1", "--minutes", "0.75", "--save", str(name.with_suffix(""))], cwd=REPO,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
        r = json.loads(Path(name).read_text(encoding="utf-8"))[0]
    assert r["character"] == "magician" and r["casts"] >= 5 and r["kills"] >= 2
    assert r["air_presses"] == 0 and r["attack_locked"] == 0 and r["swings"] == 0 and r["misses"] == 0


def test_her_walk_as_planned_is_a_world_of_its_own():
    # `--world walk-planned`: her world's walk the planner's (her profile's 125), what the planner's error costs her
    import dataclasses
    from gamelab.motion import physics
    from gamelab.sim import harness
    assert harness.WORLDS["walk-planned"] == {"walk_speed": None} and "walk_speed" in harness.HER_FIELDS
    assert world_mod.her_shoe(dataclasses.replace(MAGE_32, walk_speed=None)) is physics.SHOE
    assert world_mod.her_shoe(MAGE_32).vmax == pytest.approx(148.2)
