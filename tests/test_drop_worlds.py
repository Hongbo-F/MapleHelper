"""The simulator's drops fitted to the live service's sightings (sim.world.DropMix, `--world map-drops`) and her drops
taken by others as long as they lay live (HER_DROPS_LAST, `--world drop-takers`): each against the numbers it was
fitted to (the loot design of 2026-10-30, D:/MapleHelper-scratch/replan/loot)."""

import random
from types import SimpleNamespace

import pytest

from gamelab.sim import harness
from gamelab.sim.world import HER_DROPS_LAST, TREE_DROP_MIX, DropMix, World, drop_life


def _platform():
    return SimpleNamespace(id=3, left=-500.0, right=500.0, y_at=lambda x: 100.0)


def _drops(name: str, n: int = 4000, seed: int = 1):
    """`n` kills of a monster named `name` (the game's name) at x 0, through World._mixed_drops."""
    world = SimpleNamespace(realism=SimpleNamespace(drop_mix=TREE_DROP_MIX), _mix_rng=random.Random(seed))
    m = SimpleNamespace(name=name, platform=_platform())
    return [World._mixed_drops(world, m, 0.0, 2.0) for _ in range(n)]


def test_map_drops_gives_the_live_drops_a_kill_their_mesos_share_and_pile_sizes():
    piles = _drops("Horny Mushroom") + _drops("Green Mushroom", seed=2)
    drops = [d for p in piles for d in p]
    assert len(drops) / len(piles) == pytest.approx(0.64, abs=0.03)
    assert sum(d.mesos for d in drops) / len(drops) == pytest.approx(0.57, abs=0.03)
    full = [p for p in piles if p]
    for n, share in ((1, 0.61), (2, 0.23), (3, 0.08)):
        assert sum(len(p) == n for p in full) / len(full) == pytest.approx(share, abs=0.03)
    assert all(d.since == 2.0 and d.platform == 3 and d.y == 100.0 for d in drops)


def test_map_drops_mesos_by_their_kinds_amounts_and_items_its_kinds():
    horny = [d for p in _drops("Horny Mushroom") for d in p]
    assert all(40 <= d.amount <= 53 for d in horny if d.mesos)
    assert {d.kind for d in horny if not d.mesos} <= {"horny-mushroom-cap", "orange-potion", "bronze-ore",
                                                       "blue-potion", "magic-potion", "topaz-ore", "sapphire-ore"}
    caps = sum(d.kind == "horny-mushroom-cap" for d in horny if not d.mesos)
    assert caps / sum(not d.mesos for d in horny) > 0.8  # (its cap's 40% against the others' 1% at the most)
    assert all(25 <= d.amount <= 30 for p in _drops("Green Mushroom") for d in p if d.mesos)
    assert all(10 <= d.amount <= 20 for p in _drops("Slime") for d in p if d.mesos)


def test_map_drops_lie_twenty_units_apart_on_alternating_sides():
    """Live, neighbouring drops lay 20-25 units apart (3,649 of 6,739 gaps under 70): the first where it died, then
    +20 and -20, +40 (or the other way round), each +-3."""
    for p in [p for p in _drops("Horny Mushroom", n=600) if p]:
        xs = sorted(d.x for d in p)
        assert all(14.0 <= b - a <= 26.0 for a, b in zip(xs, xs[1:])), xs
        assert min(abs(x) for x in xs) <= 3.0


def test_map_drops_are_kept_on_the_platform():
    world = SimpleNamespace(realism=SimpleNamespace(drop_mix=DropMix(per_kill=4.0, sizes=((4, 1.0),))),
                            _mix_rng=random.Random(3))
    m = SimpleNamespace(name="Green Mushroom", platform=_platform())
    for _ in range(50):
        assert all(-497.0 <= d.x <= 497.0 for d in World._mixed_drops(world, m, 495.0, 0.0))


def test_drop_takers_lives_follow_the_live_curve():
    rng = random.Random(5)
    lives = [max(drop_life(rng.random()), 15.0) for _ in range(20000)]
    for t, share in HER_DROPS_LAST:
        if t >= 15.0:  # (nobody else picks hers up before she owns them no more: rival_owns)
            assert sum(life > t for life in lives) / len(lives) == pytest.approx(share, abs=0.015), t
    assert drop_life(1.0) == 0.0 and drop_life(0.0) == 180.0 and drop_life(0.55) == pytest.approx(60.0)


def test_the_harness_worlds_switch_them_on():
    assert harness.WORLDS["map-drops"] == {"drop_mix": TREE_DROP_MIX}
    assert harness.WORLDS["drop-takers"] == {"drop_takers": True}
    assert harness.PILE_RULES[-3:] == ("sweep", "sweep: landing", "sweep: clear first")
