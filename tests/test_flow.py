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
    blocked, _, why = flow.chain(path, Body(0.0, 200, c), [Monster(c.id, 450, speed=0.0)], hops=False)
    assert blocked == [] and why == "monster"
    hopped, _, _ = flow.chain(path, Body(0.0, 200, c), [Monster(c.id, 450, speed=0.0)])
    assert [leg.move.kind for leg in hopped] == ["hop", "climb"]
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
    from gamelab.route import Move
    # (The planner walks off A to B, it being lower; a jump there still needs its run-up.)
    jump = Move("jump", a.id, b.id, a.right - 40, 280, "right")
    flow = Flow(model, KEYS)
    k = Keys(KEYS)
    leg = flow.leg(jump, Body(0.0, a.right - 30, a), k, "run")  # 30 from the end: too close to run
    assert leg is not None and leg.end.platform is b
    ev = k.sorted()
    assert ev[0][1] == "left"                                # a step back first ...
    assert any(e[1] == "right" and e[2] == "down" for e in ev)  # ... then the run at the edge


def test_overlapping_holds_of_a_key_merge():
    k = Keys(KEYS)
    k.hold(0.5, 3.0, "up")   # one climb's up, held a bit past the top
    k.hold(2.9, 4.0, "up")   # the next climb presses up before that ends
    k.tap(1.0, "jump")
    assert [e for e in k.sorted() if e[1] == "up"] == [[500, "up", "down"], [4000, "up", "up"]]
    assert [e for e in k.sorted() if e[1] == "alt"] == [[1000, "alt", "down"], [1040, "alt", "up"]]


def test_hops_over_a_monster_walking_at_her(model):
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    path = Planner(model).plan(c.id, 100, d.id)
    flow = Flow(model, KEYS)
    pig = Monster(c.id, 420, speed=70.0, heading=-1, half=34.0, vx=-70.0)
    legs, keys, why = flow.chain(path, Body(0.0, 100, c), [pig])
    assert [leg.move.kind for leg in legs] == ["hop", "climb"]
    hop = legs[0]
    assert hop.end.platform is c and hop.end.x > pig.at(hop.end.t, c)  # lands past it
    # Without hops the same chain stops before the run.
    assert flow.chain(path, Body(0.0, 100, c), [pig], hops=False)[2] == "monster"


def test_a_hop_leaves_the_ground_before_she_can_touch_it_on_the_way_up(model):
    from gamelab.flow import HER_HALF
    from gamelab.route import GRAVITY, JUMP_SPEED
    c = by_y(model, 500, 400)
    pig = Monster(c.id, 420, speed=70.0, half=34.0, vx=0.0, height=46.0)
    hop = Flow(model, KEYS).hop(Body(0.0, 100, c), pig, 1, Keys(KEYS))
    rise = (JUMP_SPEED - (JUMP_SPEED ** 2 - 2 * GRAVITY * 46) ** 0.5) / GRAVITY  # s to climb 46 units
    assert 420 - hop.move.x == pytest.approx(34 + HER_HALF + WALK_SPEED * rise + 3, abs=2)
    assert hop.end.x > 420  # comes down past its middle: any knockback pushes her on


def test_no_hop_over_a_monster_walking_away_from_her(model):
    c = by_y(model, 500, 400)
    away = Monster(c.id, 420, speed=70.0, half=34.0, vx=70.0)
    assert Flow(model, KEYS).hop(Body(0.0, 100, c), away, 1, Keys(KEYS)) is None
