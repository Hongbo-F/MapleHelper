"""The no-hit scenes (sim.no_hit_scenes): moments of the failsafe's recordings in the world, whether each shows its
fault on this code and whether what solved means for it holds, with her jump aside and her hop up on. On master every scene's fault
shows and none is solved; `FAULT` and `OPEN`, those still so: a scene comes out of them as its fix lands, its tests flipping. Those
are the recorded phase's verdicts: `nh.SHIFTS` move most of them (README). With --slow: 8 seeds of 11 scenes, about 2 minutes on 2
workers."""

import functools
import os
from unittest import mock

import pytest

from gamelab.attack import danger
from gamelab.sim import harness, no_hit_scenes as nh, scenarios
from gamelab.sim.world import Realism, SimJumper, World

REAL = pytest.mark.skipif(not scenarios.available(scenarios.TREE), reason="needs Tree Dungeon's game data (gamedata/data)")
SEEDS = range(1, 9)
# The scenes whose fault this code still has: 111448's a03 and a04 the search says take a star (of those 83-89% hit: no jump for them);
# 080939's platform has no room for a jump either way; own_kill_235621's is perception's binding and stranger_dying_080929's kill
# comes after the last frame before her key, which no drop of a throw reaches.
FAULT = {"too_close_111448", "walked_past_080939", "stranger_dying_080929", "own_kill_235621"}
# ... and those whose fault's gone but what solved means isn't all there: 081701's first throw goes at the look, the search saying a
# star, so her first hit is 2.1 s in against 2 and 1.2 at nothing against 1; 080032's first throw from her landing, a tap later, loses
# its walker on 2 of 8, her first hit 2.1 s against 2, and one stands 0.85 s after a miss; in 15 s her hops' scenes end below master's EXP,
# 140 of 160 and 175 of 187, a pile's touches and, on 233732, 2.5 stabs (README).
OPEN = FAULT | {"too_close_081701", "walked_past_080032", "from_below_233732", "from_below_234441"}


@functools.lru_cache(maxsize=None)
def _runs(name: str) -> tuple:
    with mock.patch.dict(os.environ, GAMELAB_JUMP_ASIDE="1", GAMELAB_HOP_UP="1"):
        return tuple(nh.run_scene(name, seed)[2] for seed in SEEDS)


def _world(*mobs) -> World:
    model = scenarios.real_map(scenarios.TREE)
    keys, her = scenarios.her_on(scenarios.TREE)
    w = World(model, keys, (33, 100.0), monsters=[(m.name, m.platform, m.x) for m in mobs], realism=Realism(chase=False),
              character=her)
    for m, mob in zip(w.monsters, mobs):
        nh._place(m, mob, w)
    return w


def test_her_at_level_35_is_a_world_of_the_harness_and_her_keep_off_is_lives():
    assert harness.WORLDS["her-35"] is nh.HER_35
    assert all(k in harness.HER_FIELDS for k in nh.HER_35)
    her = harness.scenario("loop-grind", 1, "crowd", 1.0, 0.0, scenarios.TREE, ("her-35",), (), "choice").her
    assert (her.max_hp, her.level, her.defence, her.mp, her.cast_mp) == (1244.0, 35, 117.0, 1048.0, 16.0)


@REAL
def test_her_keep_off_of_a_horny_at_level_35_is_about_72():
    d = danger.of_map(scenarios.real_map(scenarios.TREE), danger.Her(nh.HER_35["max_hp"], nh.HER_35["level"], nh.HER_35["defence"]))
    assert d.keep_off("horny-mushroom") == pytest.approx(72.0, abs=10.0)


@REAL
def test_a_walker_walks_its_rails_until_she_hits_it():
    w = _world(nh.Mob(nh.G, 33, 100.0, ((0.5, 1), (0.25, 0), (0.5, -1))), nh.Mob(nh.G, 33, 200.0, ((1.0, 1),)))
    a, b = w.monsters
    speed = a.speed
    assert [round(a.at(t), 1) for t in (0.25, 0.75, 1.25)] == [round(100.0 + speed * 0.25, 1), round(100.0 + speed * 0.5, 1), 100.0]
    assert a.faces(1.0)[0] == -1
    assert b.at(0.5) == pytest.approx(200.0 + speed * 0.5)
    b.rails, b.hit_at = ((9.0, -1),), 0.5  # (hit: its own walk at its next decision, whatever the rails have left)
    b.at(1.1)
    assert b.rails == ((9.0, -1),)


@REAL
def test_a_jumper_on_rails_goes_past_its_reach_and_stays_on_the_ground():
    w = _world(nh.Mob(nh.S, 33, 280.0, ((0.25, 1), (0.25, 0), (0.25, -1))))
    m = w.monsters[0]
    assert isinstance(m, SimJumper)
    speed = m.speed
    assert m.at(0.1) == pytest.approx(280.0 + speed * 0.1)
    assert m.at(0.4) == pytest.approx(280.0 + speed * 0.25)
    assert m.at(0.7) == pytest.approx(280.0 + speed * 0.05)
    assert not m.in_air(0.7)


@REAL
def test_a_scene_runs_from_its_moment_and_stops_at_its_length():
    trace, sim, out = nh.run_scene("walked_past_080939", 1)
    assert sim.clock.elapsed == pytest.approx(nh.SCENES["walked_past_080939"].seconds, abs=0.5)
    assert out["attacks"] >= 1
    assert "SimTimeout" in str(trace["result"])  # (its length is the clock's limit: no run ends before)


def test_a_shift_moves_her_or_the_rails_and_nothing_else():
    sc = nh.SCENES["walked_past_080939"]
    assert nh.shifted(sc, "her+6").start == (16, 114.5) and nh.shifted(sc, "her+6").mobs == sc.mobs
    late, early = nh.shifted(sc, "late").mobs[0], nh.shifted(sc, "early").mobs[0]
    assert (late.x, late.rails[:2]) == (59.0, ((0.1, 0), (1.11, 1)))
    assert early.x == pytest.approx(69.0) and early.rails[0] == pytest.approx((1.01, 1)) and early.rails[1:] == sc.mobs[0].rails[1:]


@REAL
@pytest.mark.slow
def test_the_wall_foots_verdicts_hold_at_every_shift():
    def runs(shift):
        with mock.patch.dict(os.environ, GAMELAB_JUMP_ASIDE="1", GAMELAB_HOP_UP="1"):
            return [nh.run_scene("wall_foot_231613", seed, shift)[2] for seed in SEEDS]
    assert nh.over_shifts("wall_foot_231613", runs) == {"fault": 0, "solved": 4}


@REAL
@pytest.mark.slow
@pytest.mark.parametrize("name", list(nh.SCENES))
def test_the_recordings_fault_shows_where_it_isnt_fixed(name):
    assert nh.fault_shows(name, list(_runs(name))) == (name in FAULT)


@REAL
@pytest.mark.slow
@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.xfail(n in OPEN, strict=True, reason="this code's fault: flip it as its fix lands"))
                                  for n in nh.SCENES])
def test_what_solved_means(name):
    assert not nh.unmet(name, list(_runs(name)))
