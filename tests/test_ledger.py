"""The hunt's bookkeeping (behavior.ledger): every EXP gain a kill, placed where a star of hers explains it,
and every pile she knows of, kept from every look and every throw, whoever makes them. Nothing decides
from them yet (step 4 of docs/hunt-modes.md), so they're checked against the simulated world."""

import numpy as np
import pytest

from gamelab.behavior.ledger import DropStore, Kill, Ledger
from gamelab.behavior.looting import EMPTY_LOOKS, SETTLE
from gamelab.localization import mapdata
from gamelab.localization.scene import Loot, Scene
from gamelab.perception.loot import Item
from gamelab.sim import scenarios

from test_sim import plat, tree


@pytest.fixture(scope="module")
def model():
    return mapdata.parse(1, tree(), "Test Map", {}, {"2": "Snail"}, {"2": 2})


def _shoot_a_snail(sim):
    """She throws at the snail on her platform until her EXP goes up, then looks for its drops."""
    h = sim.hunter()
    h.actions.facing = None
    for _ in range(4):
        h.look()
        h.actions.throw("right")
    for _ in range(3):
        h.look()
    h.look(loot=True)
    return {"kills": [(k.kind, k.platform) for k in h.ledger.kills], "piles": len(h.drops.known),
            "hers": [p.hers for p in h.drops.known], "status": [p.status for p in h.drops.known]}


def test_a_kill_is_placed_where_her_star_went_and_its_drops_are_expected_there(model):
    b = plat(model, 360, 300)  # snails spawn on B, so a snail seen there is placed on it
    sc = scenarios.Scenario("ledger", _shoot_a_snail, (b.id, 240.0), monsters=[("Snail", b.id, 380.0)], limit=60)
    tr = scenarios.run_scenario(sc, model)
    assert tr["world"]["kills"] == {"snail": 1}
    assert tr["result"]["kills"] == [["snail", b.id]]
    assert tr["result"]["hers"] == [True] and tr["result"]["status"] == ["seen"]  # its drops showed up


class _Hunter:
    exp_gains = 0


def _look(t, x=100.0, platform=None, loot=(), looked=()):
    s = Scene(t=t, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0)
    s.loot, s.looked = list(loot), list(looked)
    return s


def test_a_gain_no_throw_explains_counts_without_a_place_or_drops():
    h, store = _Hunter(), DropStore()
    ledger = Ledger(h, store)
    h.exp_gains = 1  # party EXP, say: she threw nothing
    ledger.on_look(_look(10.0))
    (k,) = ledger.kills
    assert not k.placed and ledger.placed == [] and store.piles == []


def test_a_pile_is_forgotten_only_after_two_empty_looks_apart_and_settled():
    store = DropStore()
    store.on_kill(Kill(0.0, "pig", 3, 500.0, 300.0, 3))
    view = [(3, 300.0, 700.0)]
    store.on_look(_look(1.0, looked=view))                     # still landing: not empty yet
    assert store.piles[0].empty_since is None
    store.on_look(_look(SETTLE + 0.1, looked=view))            # empty once
    store.on_look(_look(SETTLE + 0.1 + EMPTY_LOOKS / 2, looked=view))
    assert store.known                                         # not twice far enough apart
    store.on_look(_look(SETTLE + 0.2 + EMPTY_LOOKS, looked=view))
    assert store.known == []
    item = Item("mesos", 0, 0, 13, 13, 0.9)
    store.on_look(_look(20.0, loot=[Loot(item, 900.0, 300.0, 3)], looked=[(3, 800.0, 1000.0)]))
    (p,) = store.known
    assert not p.hers and p.mesos and p.status == "seen"       # drops no kill of hers explains


@pytest.mark.parametrize("name", ["hunt_clear_basin", "tour_hunt_basin", "nav_fight_basin"])
def test_the_ledger_counts_every_kill_and_places_most_of_them_right(name):
    if scenarios.real_map() is None:
        pytest.skip("needs the real map's data")
    made = []
    real = scenarios.Sim.hunter

    def hunter(self, **kw):
        made.append(real(self, **kw))
        return made[-1]
    scenarios.Sim.hunter = hunter
    try:
        trace, sim = scenarios.run_sim(scenarios.SCENARIOS[name])
    finally:
        scenarios.Sim.hunter = real
    world = [k for k in sim.world.kill_log if k[3] == "her"]
    kills = made[0].ledger.kills
    assert len(kills) == len(world)  # the simulator never shows two kills between looks
    placed = [k for k in kills if k.placed]
    used, right = set(), 0
    for k in placed:
        t = k.t - sim.clock.start
        i = next((i for i, (wt, kind, pid, _) in enumerate(world)
                  if i not in used and kind == k.kind and pid == k.platform and 0 <= t - wt <= 2.5), None)
        if i is not None:
            used.add(i)
            right += 1
    assert len(placed) >= 0.9 * len(kills) and right >= 0.65 * len(placed)
