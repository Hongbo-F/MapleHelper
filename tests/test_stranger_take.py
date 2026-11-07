"""The stranger's cut (patrol.TAKE_WEIGHT, Belief.share; README: "Where she hunts"): at the logged moments her grinds of
2026-11-04 and 05 switched to an all-low round, the belief frozen as the log has it (tests/fixtures/
stranger_sends_2026-11-05.json), the real Patrol.best and Patrol.rate no longer take her there."""

import json
import math
from pathlib import Path

import pytest

from gamelab.behavior import patrol
from gamelab.localization import mapdata
from gamelab.localization.memory import Memory
from gamelab.localization.players import Player
from gamelab.motion.route import Planner
from gamelab.perception.kinds import kind_of
from gamelab.sim import scenarios

SENDS = json.loads((Path(__file__).parent / "fixtures" / "stranger_sends_2026-11-05.json").read_text(encoding="utf-8"))
MOMENTS = [dict(zip(SENDS["columns"], m)) for m in SENDS["moments"]]
LOW = 28  # (the block she gets to by one rope, no spawn point on the platforms above it)


@pytest.fixture(scope="module")
def tree():
    if not scenarios.available(10002073):
        pytest.skip("needs Tree Dungeon's game data (gamedata/data)")
    model = mapdata.load(10002073)
    exp = {kind_of(n): e for n, e in model.mob_exp.items()}
    kinds = {k for k, e in exp.items() if e >= 0.25 * max(exp.values())}
    return patrol.Patrol(model, kinds, exp, travel=Planner(model).arrive)


def _frozen(p, m, t=1000.0):
    """Patrol `p`'s belief as moment `m` has it: the dots through Belief.update, the kill time, the others' refill and
    the takes as the 'state' record logged them."""
    b = p.belief
    b.rival_at = {}
    dots = [Player(i, "stranger", x, y, None, (0.0, 0.0), t, t) for i, (x, y) in enumerate(m["dots"])]
    b.update(t, Memory(p.model), dots)
    p.kill_s = m["kill_s"]
    b._others = b._rate = -math.log(1.0 - m["others_tick"]) / patrol.TICK_S
    b.take = lambda kind: m["takes"].get(kind, patrol.TAKE_PRIOR)
    return t


def _all_low(p, t):
    return all(p.platform_of(j) >= LOW for j in p.best(t).stops)


def test_the_frozen_moments_are_the_moments_she_switched(tree, monkeypatch):
    """With all of the cut, as live then: the belief frozen gives the numbers logged (the middle of the new round's
    within 2%, her round's within 10%), and the all-low round is best at 19 of the 22, as it was to take it up."""
    monkeypatch.setattr(patrol, "TAKE_WEIGHT", 1.0)
    new, own, low = [], [], 0
    for m in MOMENTS:
        t = _frozen(tree, m)
        new.append(abs(tree.rate([tree.stop_on(s) for s in m["new"]], t).rate * 60 / m["new_rate"] - 1))
        own.append(abs(tree.rate([tree.stop_on(s) for s in m["round"]], t).rate * 60 / m["round_rate"] - 1))
        low += _all_low(tree, t)
    assert sorted(new)[len(new) // 2] < 0.02 and sorted(own)[len(own) // 2] < 0.1
    assert low == 19


@pytest.mark.parametrize("m", MOMENTS, ids=[f"{m['tour'][-6:]}+{m['at']}" for m in MOMENTS])
def test_a_stranger_seen_no_longer_sends_her_to_an_all_low_round(tree, m):
    """patrol.TAKE_WEIGHT as it is: the best round isn't all low (those rounds made 322-347 where she had been making
    343: the README)."""
    assert not _all_low(tree, _frozen(tree, m))
