"""Chained moves (flow.py) on a small made-up map: timelines, variants, monsters, plan watching."""

import pytest

from gamelab import mapdata
from gamelab.flow import JUMP_LAG, RUN_LAG, Body, Flow, Keys, Monster
from gamelab.route import WALK_SPEED, Planner

from test_navigation import by_y, tree

KEYS = {"left": "left", "right": "right", "jump": "alt", "up": "up", "down": "down"}


@pytest.fixture
def model():
    return mapdata.parse(1, tree(), "Test Map", {})


def test_keys_carry_the_arrow_across_legs():
    k = Keys(KEYS)
    k.set_arrow(0.0, 1)
    k.set_arrow(0.5, 1)      # still running right: nothing new
    k.set_arrow(1.0, -1)     # turn: right up, left down
    k.set_arrow(1.5, 0)
    assert k.sorted() == [[0, "right", "down"], [1000, "right", "up"], [1001, "left", "down"], [1500, "left", "up"]]


def test_a_rope_is_grabbed_on_the_run(model):
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    path = Planner(model).plan(c.id, 400, d.id)
    legs, keys, why = Flow(model, KEYS).chain(path, Body(0.0, 400, c))
    assert why == "done" and [leg.move.kind for leg in legs] == ["climb"]
    ev = keys.sorted()
    jump = next(e[0] for e in ev if e[1] == "alt" and e[2] == "down")
    up = next(e[0] for e in ev if e[1] == "up" and e[2] == "down")
    assert ev[0] == [0, "right", "down"] and 0 <= up - jump <= 80  # runs at it, jumps, holds up as she lifts off
    # The jump leaves the ground where she's one grab lead (~41 units) short of the rope.
    lifts = (jump / 1000 + JUMP_LAG - RUN_LAG) * WALK_SPEED + 400
    assert 700 - lifts == pytest.approx(41.6, abs=3)
    assert legs[-1].end.platform is d and legs[-1].end.x == 700


def test_chains_stop_where_a_monster_could_get_in_the_way(model):
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    path = Planner(model).plan(c.id, 200, d.id)
    flow = Flow(model, KEYS)
    blocked, _, why = flow.chain(path, Body(0.0, 200, c), [Monster(c.id, 450, speed=0.0)])
    assert blocked == [] and why == "monster"
    # One on another platform can't reach her.
    clear, _, why = flow.chain(path, Body(0.0, 200, c), [Monster(by_y(model, 360, 300).id, 450, speed=0.0)])
    assert len(clear) == 1 and why == "done"
    # One walking away (heading left, seen far left of her path) only turns back so far.
    away, _, _ = flow.chain(path, Body(0.0, 200, c), [Monster(c.id, 60, speed=70.0, heading=-1)])
    assert len(away) == 1


def test_where_she_should_be_while_a_chain_runs(model):
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    path = Planner(model).plan(c.id, 400, d.id)
    legs, _, _ = Flow(model, KEYS).chain(path, Body(0.0, 400, c))
    assert Flow.expected(legs, 0.5) == c.id                  # running at the rope
    assert Flow.expected(legs, legs[0].end.t - 0.3) is None   # on the rope
    assert Flow.expected(legs, legs[0].end.t + 0.3) == d.id   # on top


def test_too_short_a_run_up_steps_back_first(model):
    a, b = by_y(model, 300, 100), by_y(model, 360, 300)
    jump = next(m for m in Planner(model).moves if m.src == a.id and m.dst == b.id and m.kind == "jump")
    flow = Flow(model, KEYS)
    k = Keys(KEYS)
    leg = flow.leg(jump, Body(0.0, a.right - 30, a), k, "run")  # 30 from the end: too close to run
    assert leg is not None and leg.end.platform is b
    ev = k.sorted()
    assert ev[0][1] == "left"                                # a step back first ...
    assert any(e[1] == "right" and e[2] == "down" for e in ev)  # ... then the run at the edge
