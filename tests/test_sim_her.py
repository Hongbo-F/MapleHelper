"""The simulator's her: as she plays on each map (sim.scenarios.HER: her keys from 2026-10-23, her pet drinking her
potions), a touch's damage by her defence, the pet's potions and food, a kind's stars to kill by its HP and defence;
and a scenario, golden, the harness and the coverage report on a map of their own. Synthetic maps need no game data;
the real-map tests (Tree Dungeon's) are skipped without it. (West Rocky Mountain II's, this file's first subject, and
the first map's went on 2026-10-31.)"""

import dataclasses
from types import SimpleNamespace

import pytest

from gamelab.behavior.hunter import Hunter
from gamelab.control.timeline import SIM_KEYS, SIM_KEYS_PET
from gamelab.localization import mapdata
from gamelab.sim import golden, harness, scenarios
from gamelab.sim import world as world_mod
from gamelab.sim.world import LEVEL_19, LEVEL_23, Her, MobInfo, Realism, World
from maplehelper.keys import parse as K

from test_sim import plat, still, tap, tree

TREE = scenarios.TREE
KEYS = dataclasses.replace(SIM_KEYS_PET)


@pytest.fixture(scope="module")
def model():
    return mapdata.parse(1, tree(), "Test Map", {}, {"2": "Snail", "8": "Pig"}, {"2": 2})


def tree_map() -> mapdata.MapModel:
    m = scenarios.real_map(TREE)
    if m is None:
        pytest.skip("needs Tree Dungeon's game data (gamedata/data)")
    return m


# --- her on each map -----------------------------------------------------------------------------------------

def test_on_a_made_up_map_she_is_level_19_and_with_her_pet_her_keys_are_her_bindings_of_2026_10_23():
    assert scenarios.her_on(1) == (SIM_KEYS, LEVEL_19)  # any map but hers (a synthetic one): the first map's
    assert (LEVEL_19.max_hp, LEVEL_19.level, LEVEL_19.exp, LEVEL_19.defence, LEVEL_19.pet) == (642, 19, 421, None, None)
    assert scenarios.her_on(TREE)[0] == SIM_KEYS_PET
    # Her bindings from 2026-10-23: attack ctrl, Lucky Seven shift (its own: world.Her.skill), no potion keys (her pet
    # drinks them), pet food u
    assert SIM_KEYS_PET == dataclasses.replace(SIM_KEYS, attack=K("ctrl"), lucky_seven=K("shift"), hp_potion=None,
                                               mp_potion=None, pet_food=K("u"))
    assert world_mod._actions(SIM_KEYS_PET) == {"left": "left", "right": "right", "up": "up", "down": "down",
                                                "alt": "jump", "ctrl": "attack", "shift": "skill", "z": "pick",
                                                "u": "pet_food"}  # (the chair key does nothing: no rests)


def test_a_scenario_runs_on_its_own_map_with_her_as_she_is_there():
    sc = scenarios.Scenario("tree", lambda sim: {"keys": sim.keys.attack.name, "her": sim.world.character.level,
                                                 "map": sim.model.id}, (28, 247.0), map_id=TREE)
    trace = scenarios.run_scenario(sc, tree_map())
    assert trace["result"] == {"keys": "ctrl", "her": 25, "map": TREE}
    assert scenarios.run_scenario(dataclasses.replace(sc, her=LEVEL_19), tree_map())["result"]["her"] == 19


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
    monkeypatch.setattr(world_mod, "evade_chance", lambda info, her=None, mob_level=None: 0.0)
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


def test_a_potion_heals_what_it_heals_where_that_is_known_and_a_share_of_her_hp_where_not(model, monkeypatch):
    w = _touched(model, monkeypatch, 0.48, Her(max_hp=752.0, pet=0.45, potion=100.0))  # a Red Potion's 100 HP
    assert w.hp == pytest.approx(0.43 + 100.0 / 752.0) and w.pet_potions == 1
    w = _touched(model, monkeypatch, 0.48, Her(max_hp=752.0, pet=0.45))
    assert w.hp == pytest.approx(0.43 + world_mod.POTION_HP)


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


def test_golden_leaves_out_the_scenarios_of_a_map_whose_data_isnt_here(monkeypatch, capsys):
    tree_names = [n for n, sc in scenarios.SCENARIOS.items() if sc.map_id == TREE]
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # golden.main runs here, not in a fresh process
    real = scenarios.available
    monkeypatch.setattr(scenarios, "available", lambda map_id=scenarios.MAP_ID: map_id != TREE and real(map_id))
    assert golden.main(["check", tree_names[0]]) == 2  # nothing could be compared: never a pass
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
    sc = scenarios.Scenario("pet", lambda sim: sim.hunter().feed_pet_if_due(), (28, 247.0), map_id=TREE)
    trace = scenarios.run_scenario(sc, tree_map())
    assert trace["result"] is True  # its fullness known
    assert [p[1] for p in trace["plans"]] == ["pet_food"]  # no click, no Esc: the panel read off the world
    assert any("pet: fullness 60, fed 1 -> 90" in line for line in trace["log"])
    assert trace["world_log"] and "feeds her pet" in trace["world_log"][0]


# --- the harness on a map of its own -------------------------------------------------------------------------------

def _run(k: int, exp: float, **kw) -> dict:
    return dict(seed=k, exp_per_min=exp, drops_share=0.5, hp_lost=0.5, died_at=None, fell_out_at=None, **kw)


def test_the_harness_runs_on_a_map_of_its_own_and_compares_runs_of_one_map_only():
    sc = harness.scenario("loop", 3, map_id=TREE)
    assert (sc.map_id, sc.start, sc.name) == (TREE, harness.TREE_START_AT[3], "loop:game:3:10002073")
    base, better = [_run(k, 50.0) for k in range(1, 11)], [_run(k, 60.0) for k in range(1, 11)]
    assert harness.compare(base, better)[0]  # runs saved before the map was kept are the first map's
    there = [dict(r, map=TREE) for r in better]
    ok, lines = harness.compare(base, there)
    assert not ok and lines[0] == "the runs are on different maps ([10000012] and [10002073])"
    assert harness.compare([dict(r, map=TREE) for r in base], there)[0]


def test_the_coverage_report_asks_the_first_maps_questions_on_the_first_map_only():
    from gamelab.analysis import coverage
    graph = coverage.graph_for(tree_map())
    lines = coverage.report(graph, [("none", coverage.merge([], graph))])
    assert not any(line.startswith("== Platform") or line.startswith("== 23 -> 17") for line in lines)
    assert coverage.FOCUS == {scenarios.MAP_ID: (12,)}
