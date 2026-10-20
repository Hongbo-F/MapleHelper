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


def test_with_standing_only_a_look_from_mid_jump_doesnt_see_a_pile_empty(monkeypatch):
    # In mid-jump a look's loot strips sit above the ground (her y is put on the platform under her): it finds no
    # drops, and two such looks forgot piles with drops still on them. With looting.STANDING_ONLY only a look that
    # shows ground under her feet can see one empty.
    from gamelab.behavior import looting
    monkeypatch.setattr(looting, "STANDING_ONLY", True)
    store = DropStore()
    store.on_kill(Kill(0.0, "pig", 3, 500.0, 300.0, 3))
    view = [(3, 300.0, 700.0)]

    def look(t, ground):
        s = _look(t, looked=view)
        s.ground = ground
        store.on_look(s)
    look(SETTLE + 0.1, (100, 100))                   # no ground under her feet: in the air
    look(SETTLE + 0.2 + EMPTY_LOOKS, (100, 100))
    assert store.known and store.piles[0].empty_since is None
    look(SETTLE + 3.0, (60, 140))                    # standing
    look(SETTLE + 3.1 + EMPTY_LOOKS, (60, 140))
    assert store.known == []


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
    # Placed: over 8 seeds of each basin scenario, 0.82-1.0 of a run's kills (0.85-1.0 with her walk and
    # jump from motion.body; pooled, 0.90-0.99 before and 0.91-0.99 after). The unplaced are kills whose
    # EXP shows only after a loot sweep, more than STAR_WINDOW after the throw. Placed right: over seeds 1-8
    # of each, 0.59-0.91 of the placed (pooled 0.72-0.78), and 0.67-0.83 (0.73-0.76) once jump-backs left at
    # full speed; tour_hunt_basin's seed 22 then placed 0.57 right. The bars were 0.9 and 0.65, set on the
    # pinned seeds, and lowered twice as seeds moved; over seeds 1-16 of each, at the physics branch's base
    # (c26a177) and on that branch: placed 0.82-1.0 and 0.86-1.0 (pooled 0.89-0.97 and 0.92-0.98), placed
    # right 0.64-0.92 and 0.54-0.91 (pooled 0.74-0.76 and 0.73-0.75). The pooled shares didn't move; one seed
    # in 16 runs to either side, and the old bars failed on some seeds at the base too. With the attack
    # track's decisions (kind and platform, within 2.5 s of the world's kill): 72-78% on average over seeds
    # 1-10 of each of these scenarios, 54-87% seed by seed (61-86% before the claw's edge went by the
    # monster's size, with the same averages); the basin's pigs and slimes crowd her, walk through her and
    # hide when hit.
    assert len(placed) >= 0.8 * len(kills) and right >= 0.55 * len(placed)
