"""The simulator on a second map, West Rocky Mountain II (10004021): her as she plays there (level 23, HP 730, her
keys from 2026-10-23, her pet drinking her potions), a touch's damage by her defence, the pet's potions, and the
map's monsters (Stump, Dark Stump, Axe Stump, Red Snail: their stars to kill, speeds and bodies from the game data).
The first map's world is as it was: her at level 19, a touch a share of PADamage, HITS. Synthetic maps need no game
data; the real-map tests are skipped without it."""

import dataclasses
from types import SimpleNamespace

import pytest

from gamelab.behavior.hunter import Hunter
from gamelab.control.timeline import SIM_KEYS, SIM_KEYS_PET
from gamelab.localization import mapdata
from gamelab.perception import exp_number
from gamelab.sim import fakes, golden, scenarios
from gamelab.sim import world as world_mod
from gamelab.sim.world import LEVEL_19, LEVEL_23, Her, MobInfo, Realism, World
from maplehelper.keys import parse as K

from test_sim import plat, still, tap, tree

ROCKY = scenarios.ROCKY
KEYS = dataclasses.replace(SIM_KEYS_PET)


@pytest.fixture(scope="module")
def model():
    return mapdata.parse(1, tree(), "Test Map", {}, {"2": "Snail", "8": "Pig"}, {"2": 2})


def rocky() -> mapdata.MapModel:
    m = scenarios.real_map(ROCKY)
    if m is None:
        pytest.skip("needs West Rocky Mountain II's game data (gamedata/data)")
    return m


# --- her on each map -----------------------------------------------------------------------------------------

def test_she_is_level_19_on_the_first_map_and_level_23_with_her_pet_on_west_rocky_mountain_ii():
    assert scenarios.her_on(scenarios.MAP_ID) == (SIM_KEYS, LEVEL_19)
    assert scenarios.her_on(ROCKY) == (SIM_KEYS_PET, LEVEL_23)
    assert scenarios.her_on(1) == (SIM_KEYS, LEVEL_19)  # any other map (a synthetic one): the first map's
    assert (LEVEL_19.max_hp, LEVEL_19.level, LEVEL_19.exp, LEVEL_19.defence, LEVEL_19.pet) == (642, 19, 421, None, None)
    assert (LEVEL_23.max_hp, LEVEL_23.level, LEVEL_23.exp, LEVEL_23.pet) == (730, 23, 31_736, 0.45)
    # Her bindings from 2026-10-23: attack ctrl, Lucky Seven shift, no potion keys (her pet drinks them), pet food u
    assert SIM_KEYS_PET == dataclasses.replace(SIM_KEYS, attack=K("ctrl"), lucky_seven=K("shift"), hp_potion=None,
                                               mp_potion=None, pet_food=K("u"))
    assert world_mod._actions(SIM_KEYS_PET) == {"left": "left", "right": "right", "up": "up", "down": "down",
                                                "alt": "jump", "ctrl": "attack", "shift": "attack", "z": "pick",
                                                "d": "chair", "u": "pet_food"}


def test_a_scenario_runs_on_its_own_map_with_her_as_she_is_there():
    sc = scenarios.Scenario("rocky", lambda sim: {"keys": sim.keys.attack.name, "her": sim.world.character.level,
                                                  "map": sim.model.id}, (22, -700.0), map_id=ROCKY)
    trace = scenarios.run_scenario(sc, rocky())
    assert trace["result"] == {"keys": "ctrl", "her": 23, "map": ROCKY}
    assert scenarios.run_scenario(dataclasses.replace(sc, her=LEVEL_19), rocky())["result"]["her"] == 19


# --- a touch by her defence --------------------------------------------------------------------------------------

def test_with_her_defence_known_a_touch_takes_hp_by_meowdbs_formula():
    axe, red_snail, snail = MobInfo(75.0, 72.0), MobInfo(9.0, 45.0), MobInfo(3.0, 33.0)
    # raw = 75 x 1.3 = 97.5; 97.5 x (1 - 75 / (75 + 300 + 195)) - (75 / 25 + 23 / 10) = 79.4: 79 HP of her 730
    assert world_mod.touch_damage(axe, 0.5, LEVEL_23) == pytest.approx(79 / 730)
    # raw 10.8: 10.8 x (1 - 75 / 396.6) - 5.3 = 3.46: 3 HP
    assert world_mod.touch_damage(red_snail, 0.0, LEVEL_23) == pytest.approx(3 / 730)
    assert world_mod.touch_damage(snail, 0.99, LEVEL_23) == pytest.approx(1 / 730)  # a weak hit takes 1 HP at least
    assert world_mod.touch_damage(None, 0.5, LEVEL_23) == world_mod.TOUCH_DAMAGE
    # The first map's her: a share of PADamage, as fitted (test_sim)
    assert world_mod.touch_damage(MobInfo(15.0, 54.0), 0.5) == world_mod.touch_damage(MobInfo(15.0, 54.0), 0.5,
                                                                                         LEVEL_19) == 15 * 0.68 / 642
    # DEFENCE is what makes the formula give the logged hits' 9.56 HP at level 19 (pigs and slimes half each)
    at_19 = Her(defence=world_mod.DEFENCE)
    mean = sum(world_mod.touch_damage(MobInfo(pad, 0.0), (i + 0.5) / 400, at_19) * 642
               for pad in (15.0, 13.0) for i in range(400)) / 800
    assert mean == pytest.approx(9.56, abs=0.1)


# --- her pet -----------------------------------------------------------------------------------------------------

def _touched(model, monkeypatch, hp, character, realism=None):
    """A world where a pig touches her at once for 5% of her HP (never evaded)."""
    monkeypatch.setattr(world_mod, "evade_chance", lambda info: 0.0)
    monkeypatch.setattr(world_mod, "touch_damage", lambda info, u, her=LEVEL_19: 0.05)
    c = plat(model, 500, 400)
    w = still(World(model, KEYS, (c.id, 400.0), monsters=[("Pig", c.id, 420.0)], hp=hp, character=character,
                    realism=realism))
    w.advance(0.05)
    assert w.touches == 1
    return w


def test_her_pet_drinks_a_potion_right_after_a_touch_leaves_her_under_its_warning(model, monkeypatch):
    w = _touched(model, monkeypatch, 0.48, Her(pet=0.45))  # 48% -> 43%: under the warning
    assert w.hp == pytest.approx(0.43 + world_mod.POTION_HP) and w.potions_used == w.pet_potions == 1
    assert any("her pet drinks a potion for her" in line for line in w.log)
    assert w.summary()["pet_potions"] == 1
    w = _touched(model, monkeypatch, 0.6, Her(pet=0.45))  # 55% after it: no potion
    assert w.hp == pytest.approx(0.55) and w.potions_used == 0 and "pet_potions" not in w.summary()
    w = _touched(model, monkeypatch, 0.48, LEVEL_19)  # no pet
    assert w.hp == pytest.approx(0.43) and w.potions_used == 0


def test_her_pet_drinks_from_her_stock_and_with_none_left_nothing(model, monkeypatch):
    w = _touched(model, monkeypatch, 0.48, Her(pet=0.45), Realism(potions=1))
    assert w.potions_left == 0 and w.pet_potions == 1
    w = _touched(model, monkeypatch, 0.48, Her(pet=0.45), Realism(potions=0))
    assert w.hp == pytest.approx(0.43) and w.pet_potions == 0
    assert any("her pet has no potion left to drink" in line for line in w.log)


def test_the_pet_food_key_feeds_her_pet(model):
    c = plat(model, 500, 400)
    w = World(model, KEYS, (c.id, 400.0), monsters=None, character=LEVEL_23)
    tap(w, 0.1, "u", 0.04)
    w.advance(0.5)
    assert w.pet_fed == 1 and any("feeds her pet" in line for line in w.log) and w.hp == 1.0


def test_with_no_potion_key_the_hunter_neither_drinks_nor_says_it_does():
    said, drunk = [], []
    h = Hunter.__new__(Hunter)  # only what drink_if_low uses
    h.potion_hp, h._last_potion, h.log, h.report = 0.45, -1e9, said.append, SimpleNamespace(potions=0)
    h.feed_pet_if_due = lambda s=None: False
    h.actions = SimpleNamespace(keys=SimpleNamespace(hp_potion=None), drink=lambda: drunk.append(1) or True)
    h.drink_if_low(SimpleNamespace(hp=0.2))
    assert said == [] and drunk == []
    h.actions.keys.hp_potion = K("r")
    h.drink_if_low(SimpleNamespace(hp=0.2))
    assert said == ["HP 20%: drinking a potion"] and drunk == [1] and h.report.potions == 1


# --- the map's monsters -------------------------------------------------------------------------------------------

def test_a_kind_the_first_map_hasnt_takes_as_many_stars_as_its_hp_and_defence_need():
    assert world_mod.stars_to_kill(45) == pytest.approx(1 + 15 / 41)  # a star of 30-70 kills a snail 26 times in 41
    assert [round(world_mod.stars_to_kill(hp, pdd), 2) for hp, pdd in ((68, 0), (133, 30), (236, 30), (371, 30))] \
        == [1.95, 3.99, 6.66, 10.18]
    hits = world_mod.hits_of
    assert [hits(k, MobInfo(0, 0, hp=hp, pdd=pdd)) for k, hp, pdd in (
        ("red-snail", 68, 0), ("stump", 133, 30), ("dark-stump", 236, 30), ("axe-stump", 371, 30))] == [2, 4, 7, 10]
    assert hits("slime", MobInfo(0, 0, hp=115, pdd=10)) == 2  # HITS' kinds as fitted (by the formula 3)
    assert hits("stump", None) == hits("stump", MobInfo(9.0, 45.0)) == world_mod.HITS_UNKNOWN


def test_west_rocky_mountain_ii_s_monsters_are_as_the_game_data_has_them():
    m = rocky()
    w = World(m, SIM_KEYS_PET, (22, -700.0), monsters="all", character=LEVEL_23)
    if any(mo.info is None for mo in w.monsters):
        pytest.skip("the monsters' game data isn't on this computer")
    kinds = {}
    for mo in w.monsters:
        kinds.setdefault(mo.kind, set()).add((mo.hp_max, round(mo.speed, 2), mo.half, mo.height, mo.info.pad,
                                              mo.info.acc))
    assert kinds == {"stump": {(4, 37.5, 33.0, 51.0, 11, 45)}, "dark-stump": {(7, 37.5, 33.0, 51.0, 34, 61)},
                     "axe-stump": {(10, 62.5, 33.0, 51.0, 75, 72)}, "red-snail": {(2, 62.5, 20.5, 34.0, 9, 45)}}
    count = {k: sum(mo.kind == k for mo in w.monsters) for k in kinds}
    assert count == {"stump": 20, "red-snail": 12, "dark-stump": 10, "axe-stump": 7}  # the map's 49 spawn points
    assert not any(mo.platform.id in m.twins for mo in w.monsters)  # a twin's spawns go with the platform it lies on


def test_the_fake_detector_knows_the_maps_kinds_and_her_number_starts_at_her_level():
    assert fakes.detector_kinds(scenarios.real_map() or pytest.skip("needs the first map")) == fakes.KINDS
    m = rocky()
    assert fakes.detector_kinds(m) == fakes.KINDS + ("axe-stump", "dark-stump", "red-snail", "stump")

    def fn(sim):
        box = sim.eyes._exp_box()
        reading = exp_number.read(box)
        return {"exp": reading.exp, "level": 23 in reading.levels, "kinds": sorted(sim.eyes.kinds)}
    try:
        got = scenarios.run_scenario(scenarios.Scenario("number", fn, (22, -700.0), map_id=ROCKY), m)["result"]
    except (OSError, ValueError) as e:  # the bar's glyphs come from the game data copy
        pytest.skip(f"the game data copy isn't here ({e})")
    assert got["exp"] == 31_736 and got["level"]
    assert {"stump", "dark-stump", "axe-stump", "red-snail"} <= set(got["kinds"])


def test_golden_leaves_out_the_scenarios_of_a_map_whose_data_isnt_here(monkeypatch, capsys):
    rocky_names = [n for n, sc in scenarios.SCENARIOS.items() if sc.map_id == ROCKY]
    if not rocky_names:
        pytest.skip("no scenario on West Rocky Mountain II")
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # golden.main runs here, not in a fresh process
    real = scenarios.available
    monkeypatch.setattr(scenarios, "available", lambda map_id=scenarios.MAP_ID: map_id != ROCKY and real(map_id))
    if not real():
        pytest.skip("needs the first map's game data")
    assert golden.main(["check", rocky_names[0]]) == 2  # nothing could be compared: never a pass
    assert "left out" in capsys.readouterr().out


def test_her_pet_goes_hungry_in_time_a_pet_food_fills_it_and_too_hungry_it_drinks_nothing(model, monkeypatch):
    from gamelab.behavior import pet
    c = plat(model, 500, 400)
    w = World(model, KEYS, (c.id, 400.0), monsters=None, character=LEVEL_23)
    assert w.pet_fullness() == world_mod.PET_FULLNESS == 60
    w.advance(pet.TICK_S * 3 + 1)
    assert w.pet_fullness() == 60 - 3 * pet.HUNGRY
    tap(w, w.t + 0.1, "u", 0.04)
    w.advance(w.t + 0.5)
    assert w.pet_fullness() == 57 + pet.FOOD
    w = _touched(model, monkeypatch, 0.48, Her(pet=0.45))
    assert w.pet_potions == 1
    monkeypatch.setattr(world_mod, "PET_FULLNESS", float(pet.PACKED_AT))
    w = _touched(model, monkeypatch, 0.48, Her(pet=0.45))
    assert w.pet_potions == 0 and any("too hungry" in line for line in w.log)


def test_in_the_simulator_her_pet_care_reads_the_worlds_pet_and_feeds_it_once_at_the_start():
    sc = scenarios.Scenario("pet", lambda sim: sim.hunter().feed_pet_if_due(), (22, -700.0), map_id=ROCKY)
    trace = scenarios.run_scenario(sc, rocky())
    assert trace["result"] is True  # its fullness known
    assert [p[1] for p in trace["plans"]] == ["pet_food"]  # no click, no Esc: the panel read off the world
    assert any("pet: fullness 60, fed 1 -> 90" in line for line in trace["log"])
    assert trace["world_log"] and "feeds her pet" in trace["world_log"][0]
