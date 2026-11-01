"""The simulator's monsters as the v83 client's AI moves them (the user, 2026-10-31: "Yes, align them"), as RustMS's
copy of HeavenClient has it (Mob::step_body, think, think_jump, chase; FootholdTree::get_walk_edge, limit_jumper,
get_wall_ahead; Physics::jump): its decisions in whole AI steps (sim.world.AI_STEP, IDLE_THINK_MS, CHASE_THINK_MS), a
walker's turn WALK_EDGE from its platform's ends, its chase keeping the way it chose near her, a hit a chase anew, and
the monsters that jump (SimJumper: Tree Dungeon's Slimes) jumping, at walls and off their platform too."""

import random
from collections import Counter

import pytest

from gamelab.localization.mapdata import Platform, Segment
from gamelab.sim import scenarios
from gamelab.sim import world as W
from tests.test_sim import KEYS

TREE = 10002073


def walker(seed: int = 3, p: Platform | None = None) -> W.SimMonster:
    m = W.SimMonster(0, "Pig", "pig", p or Platform(0, [Segment(0, 500, 400, 500)]), 200.0, random.Random(seed),
                     speed=125.0)
    m.start(0.0, 200.0)
    return m


def test_a_walkers_decisions_hold_whole_ai_steps_standing_left_or_right_a_third_each():
    m = walker()
    seen = []
    while len(seen) < 3000:
        m.at(m.t1 + 1e-9)
        seen.append((round((m.t1 - m.t0) / W.AI_STEP, 6), m.d))
    steps = Counter(k for k, _ in seen)
    assert all(k == int(k) for k in steps) and min(steps) == 11 and max(steps) == 33  # (1.0-3.0 s)
    ways = Counter(d for _, d in seen)
    assert all(abs(ways[d] / len(seen) - 1 / 3) < 0.03 for d in (-1, 0, 1)), ways


def test_a_walker_turns_20_along_its_end_footholds_from_its_platforms_ends():
    flat = Platform(0, [Segment(0, 500, 400, 500)])
    assert W.walk_edges(flat) == (20.0, 380.0)
    sloped = Platform(1, [Segment(0, 500, 30, 540), Segment(30, 540, 400, 540)])  # (its first foothold 50 long, 30 across)
    assert W.walk_edges(sloped) == (pytest.approx(12.0), 380.0)
    m = walker(p=flat)
    xs = [m.at(i * 0.05) for i in range(1, 8000)]
    assert 20.0 - 1e-9 <= min(xs) < 22.0 and 378.0 < max(xs) <= 380.0 + 1e-9


def test_the_stack_predicts_the_turns_where_the_simulator_and_the_game_make_them():
    """One rule for both (localization.monsters.walk_edges): the simulator's walkers turn by it, and the stack's
    predictions of where a monster's middle turns (Monster.at, the tracker, attack.danger, motion.flow, behavior.probe)
    go by it; a kind that jumps reaches its platform's ends (attack.danger's threats that hop)."""
    from gamelab.localization import monsters
    flat = Platform(0, [Segment(0, 500, 400, 500)])
    assert W.walk_edges is monsters.walk_edges and W.WALK_EDGE == monsters.WALK_EDGE == 20.0
    assert monsters.walk_edges(flat) == (20.0, 380.0) and monsters.walk_edges(flat, jumps=True) == (0.0, 400.0)
    seen = monsters.MonsterTrack(0, 370.0, vx=100.0, seen_at=0.0, half=34.0)  # (its half width 34: it turned at 366)
    assert seen.at(0.1, flat) == pytest.approx(380.0) and seen.at(0.2, flat) == pytest.approx(370.0)


def test_chasing_by_her_it_keeps_the_way_it_chose_back_to_the_end_that_turned_it():
    """Mob::think: within AGGRO_PASS of her it keeps the way it chose, so the end of its platform that turned it is
    walked back to at its next decision (where the simulator had it go on the way the turn sent it, past her)."""
    m = walker()
    m.chase, m.chase_rng = (lambda m, t: 370.0), random.Random(1)
    m.at(1.0)
    m.aggro(1.0)
    decisions = set()
    for i in range(1, 1200):
        m.at(1.0 + i * 0.05)
        decisions.add((m.t0, m.x0, m.d))
    near = [d for t0, x0, d in decisions if abs(370.0 - x0) < W.AGGRO_PASS]
    assert any(x0 > 370.0 for _, x0, _ in decisions) and near and all(d == 1 for d in near)
    longest = (sum(W.CHASE_THINK_MS) - 1) // 90 * W.AI_STEP
    assert all(d != 0 and t1 - t0 <= longest + 1e-9 for t0, t1, d in {(m.t0, m.t1, m.d)})


def test_a_hit_is_a_chase_anew_its_way_decided_again_at_her():
    m = walker()
    m.chase, m.chase_rng = (lambda m, t: 300.0), random.Random(1)
    m.at(1.0)
    m.aggro(1.0)
    m.chase_dir = -1  # (as if it kept going left by her)
    m.at(1.5)
    m.knock(1.5, -1)
    m.aggro(1.5)
    assert m.chase_dir == 0 and m.t1 == pytest.approx(1.5 + W.monster_data.FLINCH_S)
    m.at(m.t1 + 0.01)
    assert m.d == (1 if 300.0 >= m.x0 else -1)  # (at her, from the flinch's end)


@pytest.fixture(scope="module")
def tree():
    m = scenarios.real_map(TREE)
    if m is None:
        pytest.skip("needs Tree Dungeon, Forest Up North II's game data (gamedata/data)")
    return m


def slime(tree, seed: int = 1, x: float = 0.0) -> tuple[W.World, W.SimJumper]:
    w = W.World(tree, KEYS, (16, 100.0), monsters=[("Slime", 33, x)], seed=seed)
    return w, w.monsters[0]


def test_tree_dungeons_slimes_jump_and_its_mushrooms_walk(tree):
    w = W.World(tree, KEYS, (16, 100.0), monsters="all", seed=1)
    assert {m.kind for m in w.jumpers} == {"slime"} and len(w.jumpers) == 2
    assert all(not isinstance(m, W.SimJumper) for ms in w.by_platform.values() for m in ms)


def test_a_slime_jumps_77_high_hops_onto_32_over_33_and_comes_back(tree):
    """Platform 32 is 57-60 over 33, within its jump: its jumps under it land it there, and off 32's free ends it
    jumps back down (or onto 31)."""
    rises, visits = [], Counter()
    for seed in (1, 2, 3):
        w, m = slime(tree, seed, x=60.0)
        last = None
        for i in range(1, 6000):  # (10 minutes)
            t = i * 0.1
            m.at(t)
            visits[m.platform.id] += 1
            s = m.segs[-1]
            if s[0] == "air" and s is not last:
                last = s
                ys = [y for _, y in s[2].path]
                rises.append(ys[0] - min(ys))
    assert len(rises) > 100 and max(rises) == pytest.approx(555.0 ** 2 / 4000, abs=1.0)
    assert visits[32] > 0 and visits[33] > 0.5 * sum(visits.values())


def test_a_slime_by_33s_wall_jumps_at_it_and_turns_back(tree):
    """33's right end rises into the map's wall (x 310): within JUMP_AHEAD of it the slime jumps at it while its
    decision lasts; it never gets past it, and pressed against it it jumps again a tick after each landing (a jump asked
    for at its platform's end) until an AI step comes in that tick and turns it back (about 16 s here): the client's
    rules played out."""
    w, m = slime(tree, x=200.0)
    p = m.platform
    m.segs, m.ai_t, m.thinktimer = [("walk", 0.0, 200.0, 1, p)], 0.09, 25
    xs, jumped_from = [], []
    for i in range(1, 2400):  # (a minute)
        t = i * 0.025
        xs.append(m.at(t))
        if m.segs[-1][0] == "air" and not jumped_from:
            jumped_from.append(m.segs[-1][2].path[0][0])
    assert jumped_from and 310.0 - W.JUMP_AHEAD <= jumped_from[0] < 310.0
    assert max(xs) <= 310.0 + 1e-6 and any(s[0] == "walk" and s[3] == -1 for s in m.segs)


def test_hit_in_the_air_a_slime_is_stood_where_it_lands_and_pushed_on_from_there(tree):
    w, m = slime(tree, x=60.0)
    m.at(0.05)
    m._jump(0.05, 1, check=False)
    f = m.segs[-1][2]
    m.knock(0.2, 1)
    s = m.segs[-1]
    assert s[0] == "hit" and s[2] == pytest.approx(f.land_x) and s[4] is f.platform
    assert m.at(0.2 + W.KNOCK_MOB_S) == pytest.approx(f.land_x + W.KNOCK_MOB)
    assert m.hit_until == pytest.approx(0.2 + W.monster_data.FLINCH_S)
