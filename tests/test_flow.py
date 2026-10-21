"""Chained moves (flow.py) on a small made-up map: timelines, variants, monsters, plan watching."""

import pytest


from gamelab.localization.monsters import MonsterTrack
from gamelab.motion.flow import Body, Flow, JUMP_LAG, Keys, RUN_LAG
from gamelab.motion.physics import WALK_SPEED
from gamelab.motion.route import Planner

from test_navigation import by_y, tree
from gamelab.localization import mapdata

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


def test_a_drop_is_made_only_where_the_floor_lets_her_through():
    from gamelab.motion.route import Move
    t = tree()
    t["foothold"]["1"]["1"]["2"]["forbidFallDown"] = 1  # A's 100..200 forbids it, 0..100 doesn't
    m = mapdata.parse(1, t, "Test Map", {})
    a, c = by_y(m, 300, 100), by_y(m, 500, 400)
    move = Move("drop", a.id, c.id, 100, 100)
    leg = Flow(m, KEYS).leg(move, Body(0.0, 180, a), Keys(KEYS))
    assert leg is not None and leg.end.platform is c and leg.spans[-1].x1 < 100.5 and leg.end.x < 100.5
    plain = mapdata.parse(1, tree(), "Test Map", {})  # without the flag she drops near where she stands
    a, c = by_y(plain, 300, 100), by_y(plain, 500, 400)
    leg = Flow(plain, KEYS).leg(Move("drop", a.id, c.id, 100, 100), Body(0.0, 180, a), Keys(KEYS))
    assert leg.spans[-1].x1 > 150


def test_a_drop_isnt_made_a_taps_step_off_where_it_goes_through():
    # Standing within STEP_ACCEPT of where a move wants her does for other moves, not for a drop: on a
    # foothold that forbids it the drop does nothing, and by a rope's top down climbs onto the rope. She
    # steps in first, to where it goes through (route.drop_spots), and in the world she drops onto C.
    from gamelab.motion import route
    from gamelab.motion.route import Move
    from gamelab.sim.world import World
    from test_sim import KEYS as WORLD_KEYS
    t = tree()
    t["foothold"]["1"]["1"]["2"]["forbidFallDown"] = 1  # A's 100..200 forbids it, 0..100 doesn't
    m = mapdata.parse(1, t, "Test Map", {})
    a, c, d = by_y(m, 300, 100), by_y(m, 500, 400), by_y(m, 300, 700)  # D: a rope hangs from it at 700
    for here, top, floor in ((101.0, a, c), (104.0, a, c), (107.0, a, c), (110.0, a, c), (676.0, d, c), (724.0, d, c)):
        k = Keys(KEYS)
        leg = Flow(m, KEYS).leg(Move("drop", top.id, floor.id, here, here), Body(0.0, here, top), k)
        x = leg.spans[-1].x1
        assert leg.end.platform is floor and route.drop_clear(m, top, x) and abs(x - here) > 1, here
        w = World(m, WORLD_KEYS, (top.id, here), monsters=None)
        for ms, name, how in k.sorted():
            w.key(ms / 1000, name, how == "down")
        w.advance(leg.end.t + 1.0)
        assert w.plat is floor and w.mode == "ground", here
    # Standing where it goes through, she drops there.
    leg = Flow(m, KEYS).leg(Move("drop", a.id, c.id, 90, 90), Body(0.0, 90.0, a), Keys(KEYS))
    assert leg.spans[-1].x1 == 90


def test_chains_stop_where_a_monster_could_get_in_the_way(model):
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    path = Planner(model).plan(c.id, 200, d.id)
    flow = Flow(model, KEYS)
    blocked, _, why = flow.chain(path, Body(0.0, 200, c), [MonsterTrack(c.id, 450, speed=0.0)], hops=False)
    assert blocked == [] and why == "monster"
    hopped, _, _ = flow.chain(path, Body(0.0, 200, c), [MonsterTrack(c.id, 450, speed=0.0)])
    assert [leg.move.kind for leg in hopped] == ["hop", "climb"]
    # One on another platform can't reach her.
    clear, _, why = flow.chain(path, Body(0.0, 200, c), [MonsterTrack(by_y(model, 360, 300).id, 450, speed=0.0)])
    assert len(clear) == 1 and why == "done"
    # One walking away (heading left, seen far left of her path) only turns back so far.
    away, _, _ = flow.chain(path, Body(0.0, 200, c), [MonsterTrack(c.id, 60, speed=70.0, heading=-1)])
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
    from gamelab.motion.route import Move
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
    pig = MonsterTrack(c.id, 420, speed=70.0, heading=-1, half=34.0, vx=-70.0)
    legs, keys, why = flow.chain(path, Body(0.0, 100, c), [pig])
    assert [leg.move.kind for leg in legs] == ["hop", "climb"]
    hop = legs[0]
    assert hop.end.platform is c and hop.end.x > pig.at(hop.end.t, c)  # lands past it
    # Without hops the same chain stops before the run.
    assert flow.chain(path, Body(0.0, 100, c), [pig], hops=False)[2] == "monster"


def test_a_hop_leaves_the_ground_before_she_can_touch_it_on_the_way_up(model):
    from gamelab.motion.flow import HER_HALF
    from gamelab.motion.physics import GRAVITY, JUMP_SPEED
    c = by_y(model, 500, 400)
    pig = MonsterTrack(c.id, 420, speed=70.0, half=34.0, vx=0.0, height=46.0)
    hop = Flow(model, KEYS).hop(Body(0.0, 100, c), pig, 1, Keys(KEYS))
    rise = (JUMP_SPEED - (JUMP_SPEED ** 2 - 2 * GRAVITY * 46) ** 0.5) / GRAVITY  # s to climb 46 units
    assert 420 - hop.move.x == pytest.approx(34 + HER_HALF + WALK_SPEED * rise + 3, abs=2)
    assert hop.end.x > 420  # comes down past its middle: any knockback pushes her on


def test_after_a_hop_she_runs_on_from_the_games_next_step_as_after_any_landing(model):
    # Landing between two of the game's steps she stands until the next (body.LANDING_ENDS_STEP), so the leg
    # after a hop (often the move's jump: _hop_then) starts half a step after she lands, as after a jump's
    # landing (Flow._landing). Without it the next jump was keyed 15 ms early and took off about 2 units short.
    from gamelab.motion import body
    from gamelab.motion.physics import JUMP_SPEED, flight
    c = by_y(model, 500, 400)
    pig = MonsterTrack(c.id, 420, speed=70.0, half=34.0, vx=0.0, height=46.0)
    k = Keys(KEYS)
    hop = Flow(model, KEYS).hop(Body(0.0, 100, c), pig, 1, k)
    t_off = hop.spans[0].t1
    _, _, ta, _ = flight(model, hop.move.x, c.y_at(hop.move.x), WALK_SPEED, -JUMP_SPEED, group=c.group)
    assert body.landing_wait() == pytest.approx(0.015)
    assert hop.end.t == pytest.approx(t_off + ta + body.landing_wait(), abs=0.002) and hop.end.run == 1


def test_no_hop_over_a_monster_walking_away_from_her(model):
    c = by_y(model, 500, 400)
    away = MonsterTrack(c.id, 420, speed=70.0, half=34.0, vx=70.0)
    assert Flow(model, KEYS).hop(Body(0.0, 100, c), away, 1, Keys(KEYS)) is None


# --- her body as the engine moves it (motion.body's closed forms) ----------------------------------

def replayed(k: Keys, until: float, running: bool = False) -> float:
    """Where body.replay (the engine, step by step) has her at `until` under plan k's arrows, from x 0
    standing, or running right at full speed (the arrow held from a second before): the mean over the
    phases of the game's steps, which is what the closed forms stand for."""
    from gamelab.motion import body
    from gamelab.motion.physics import LAGS, SHOE
    events = [(-1.0, "right", True)] if running else []
    events += [(t / 1000, name, how == "down") for t, name, how in k.sorted() if name in ("left", "right")
               and not (running and t == 0 and how == "down")]
    out = []
    for phase in body.phases(dt=0.001):
        points = body.replay(events, until, SHOE, LAGS, start=-1.0 if running else 0.0, dt=0.001, phase=phase).points
        at_0 = next(x for t, x, _ in points if t >= -1e-9) if running else 0.0
        out.append(points[-1][1] - at_0)
    return sum(out) / len(out)


def test_she_stops_where_the_engine_says_from_standing_and_running(model):
    c = by_y(model, 500, 400)
    flow = Flow(model, KEYS)
    for running, (lo, hi) in ((False, (38, 42)), (False, (14, 16)), (True, (60, 70)), (True, (30, 31))):
        k = Keys(KEYS)
        t, x = flow._stop_within(Body(0.0, 0.0, c, run=int(running)), lo, hi, (lo + hi) / 2, k)
        assert lo - 0.5 <= x <= hi + 0.5
        assert replayed(k, t, running) == pytest.approx(x, abs=0.3)          # stopped there ...
        assert replayed(k, t + 0.5, running) == pytest.approx(x, abs=0.3)    # ... by then
        assert replayed(k, t - 0.02, running) < x - 0.01                     # (and not long before)


def test_a_step_shorter_than_a_tap_isnt_taken(model):
    from gamelab.motion.flow import STEP_ACCEPT
    c = by_y(model, 500, 400)
    flow, k = Flow(model, KEYS), Keys(KEYS)
    assert flow._stop_within(Body(0.0, 100.0, c), 100 + STEP_ACCEPT - 2, 150, 120, k) == (0.0, 100.0)
    assert k.sorted() == []  # a tap that short moves her 0 or 7 units in game
    t, x = flow._stop_within(Body(0.0, 100.0, c), 100 + STEP_ACCEPT + 4, 150, 112, k)
    assert x == pytest.approx(112, abs=0.3) and [e[1] for e in k.sorted()] == ["right", "right"]


def test_a_jump_from_standing_is_never_keyed_in_the_kicks_blind_spot(model):
    from gamelab.motion.physics import LAGS, SHOE, WALK_SPEED
    from gamelab.motion import body
    c = by_y(model, 500, 400)
    standing = Body(0.0, 100.0, c)
    # Meant to leave the ground 145 ms after the arrow (the key 80 ms after it): put off to 150 ms, after four
    # of the game's steps of her run but for 5% of jumps, so she leaves at full speed (a jump back used to be
    # keyed at 80 ms: three of five left at 121-127).
    press, t, x, vx = Flow.liftoff(standing, 1, 0.08 + JUMP_LAG)
    assert press == pytest.approx(0.150) and vx == WALK_SPEED and t == pytest.approx(0.150 + JUMP_LAG)
    assert x == pytest.approx(100 + body.run_x(t - LAGS.on, 0.0, 1, SHOE), abs=1e-6)
    # Keyed later, it's as planned; running already, any time (the arrow was seen long ago).
    assert Flow.liftoff(standing, 1, 0.2 + JUMP_LAG)[0] == pytest.approx(0.2)
    assert Flow.liftoff(Body(0.0, 100.0, c, run=1), 1, 0.05 + JUMP_LAG)[:2] == pytest.approx((0.05, 0.05 + JUMP_LAG))
    # Straight up (no arrow): she leaves where she stands, with no sideways speed.
    assert Flow.liftoff(standing, 0, 0.0)[2:] == (100.0, 0.0)


def test_a_jump_leaves_with_what_she_has_on_the_ground(model):
    c = by_y(model, 500, 400)
    # Landed at 120 units/s with the arrow held (a jump that left so): she speeds up again at 1400 units/s²,
    # and a jump at once leaves at what she has by then (no kick above 0.8 x 149).
    landed = Body(1.0, 100.0, c, run=1, landed=True, v=120.0)
    press, t, x, vx = Flow.liftoff(landed, 1, 1.0)
    assert press == pytest.approx(landed.ready_at) and vx == pytest.approx(min(149.0, 120 + 1400 * (t - 1.0)))
    full = 29 / 1400  # s to full speed
    assert x == pytest.approx(100 + (120 + 149) / 2 * full + 149 * (t - 1.0 - full))


def test_a_stop_landing_slides_from_her_landing_speed(model):
    from gamelab.motion.flow import ARROW_LAG
    from gamelab.motion.route import Move
    c = by_y(model, 500, 400)
    k = Keys(KEYS)
    k.set_arrow(0.0, 1)
    m = Move("jump", c.id, c.id, 100, 200, "right")
    # (she stands where she lands until the game's next step first: half a step on average)
    from gamelab.motion import body
    wait = body.landing_wait()
    leg = Flow(model, KEYS)._landing(m, Body(0.0, 100.0, c), k, "stop", 1.0, 300.0, c, 1, 127.0, [])
    assert leg.end.x == pytest.approx(300 + 127 ** 2 / 1600) and leg.end.t == pytest.approx(1.0 + wait + 127 / 800)
    assert [1000 - ARROW_LAG * 1000, "right", "up"] in k.sorted()  # let go so the game sees it in the air
    ran = Flow(model, KEYS)._landing(m, Body(0.0, 100.0, c), Keys(KEYS), "run", 1.0, 300.0, c, 1, 127.0, [])
    assert ran.end.run == 1 and ran.end.v == 127.0 and ran.end.landed
    assert ran.end.x == 300.0 and ran.end.t == pytest.approx(1.0 + wait) and wait == pytest.approx(0.015)


# --- on the real map ------------------------------------------------------------------------------------

from gamelab.sim import scenarios  # noqa: E402

REAL = pytest.mark.skipif(not scenarios.available(), reason="needs the map's game data (gamedata/data)")


def _real_moves(*wanted):
    """The real map and its route moves of each (kind, src, dst) wanted (the first of each)."""
    m = scenarios.real_map()
    moves = Planner(m).moves
    return m, [next(mv for mv in moves if (mv.kind, mv.src, mv.dst) == w) for w in wanted]


@REAL
def test_a_step_after_a_climb_goes_down_when_planned_and_a_run_before_the_top(old_route_takeoffs):
    """After the climb 36 -> 32 the jump onto 29 needs a step back (right, ~20 units) at the route's takeoffs of
    the night of 2026-10-21 (old_route_takeoffs: 33-45 units from 32's left end): its arrow goes down where the
    plan has her standing after the climb, not before the top (in game, pressed before the top it walked her
    8-30 units too far and the jumps after it came down in the basin). After the climb 6 -> 3, the run to walk
    off 3's left end goes down before the top (flow.PRE_ARROW), as before: she sets off as the climb-over ends."""
    from gamelab.motion.flow import CLIMB_OVER, PRE_ARROW, STEP_EARLY_MAX_MS, ms
    m, (climb, jump) = _real_moves(("climb", 36, 32), ("jump", 32, 29))
    legs, keys, why = Flow(m, KEYS).chain([climb, jump], Body(0.0, 470.0, m.platform(36)))
    assert why == "done" and [leg.move.kind for leg in legs] == ["climb", "jump"]
    ev = [e for e in keys.sorted() if e[0] > ms(legs[0].start.t) + 1000]  # after the run at the rope
    down, up = ev[0], next(e for e in ev if e[1] == "right" and e[2] == "up")
    assert down[1:] == ["right", "down"] and down[0] == ms(legs[0].end.t)  # where she stands after the climb
    assert 0 < up[0] - down[0] <= STEP_EARLY_MAX_MS
    m, (climb, walk) = _real_moves(("climb", 6, 3), ("walk_off", 3, 4))
    legs, keys, why = Flow(m, KEYS).chain([climb, walk], Body(0.0, 790.0, m.platform(6)))
    assert why == "done" and [leg.move.kind for leg in legs] == ["climb", "walk_off"]
    left = [e[0] for e in keys.sorted() if e[1:] == ["left", "down"]]
    assert left[-1] == ms(legs[0].end.t - CLIMB_OVER - PRE_ARROW)


@REAL
def test_with_the_placed_takeoffs_a_step_after_a_climb_still_goes_down_when_planned():
    """With the route's takeoffs placed for room (route._placed) the jump 32 -> 29 takes off 15 units from 32's
    left end (516), 23 left of the rope she climbs from 36: after the climb she runs at it, the arrow down before
    the top (PRE_ARROW), with no step back. The climb 7 -> 1 and the jump right off 1 onto 0 (taking off at 1170,
    18 left of the rope) has the step back now (left, about 30 units): its arrow goes down where the plan has her
    standing after the climb, and is let go within STEP_EARLY_MAX_MS."""
    from gamelab.motion.flow import CLIMB_OVER, PRE_ARROW, STEP_EARLY_MAX_MS, ms
    m, (climb, jump) = _real_moves(("climb", 36, 32), ("jump", 32, 29))
    assert jump.takeoff == 516
    legs, keys, why = Flow(m, KEYS).chain([climb, jump], Body(0.0, 470.0, m.platform(36)))
    assert why == "done" and [leg.move.kind for leg in legs] == ["climb", "jump"]
    down = [e for e in keys.sorted() if e[0] > ms(legs[0].start.t) + 1000][0]
    assert down[1:] == ["left", "down"] and down[0] == ms(legs[0].end.t - CLIMB_OVER - PRE_ARROW)
    m, (climb, jump) = _real_moves(("climb", 7, 1), ("jump", 1, 0))
    legs, keys, why = Flow(m, KEYS).chain([climb, jump], Body(0.0, 1188.0, m.platform(7)))
    assert why == "done" and [leg.move.kind for leg in legs] == ["climb", "jump"]
    ev = [e for e in keys.sorted() if e[0] > ms(legs[0].start.t) + 1000]
    down, up = ev[0], next(e for e in ev if e[1] == "left" and e[2] == "up")
    assert down[1:] == ["left", "down"] and down[0] == ms(legs[0].end.t)
    assert 0 < up[0] - down[0] <= STEP_EARLY_MAX_MS


def _route_jumps_land_with_room() -> int:
    """Each of the map's running jumps as chains do it, from where they pick (with rng), from standing at its
    spot and 20 units either side, and running in: from where she really leaves the ground it lands on its
    platform LANDING_ROOM in from its ends, and still does if she leaves 8 units early or 6% slower (the check
    the investigation of 2026-10-21's misses ran). How many route jumps there are."""
    import random

    from gamelab import directions
    from gamelab.motion.physics import JUMP_SPEED, flight, fly
    from gamelab.motion.route import JUMP_FROM, LANDING_ROOM
    m = scenarios.real_map()
    jumps = [mv for mv in Planner(m).moves if mv.kind == "jump"]
    flow = Flow(m, KEYS, rng=random.Random(1))
    for mv in jumps:
        a, q, d = m.platform(mv.src), m.platform(mv.dst), directions.sign(mv.direction)
        spot = mv.takeoff if mv.takeoff is not None else (a.right if d > 0 else a.left) - d * JUMP_FROM
        starts = [Body(0.0, mv.x + d * dx, a) for dx in (0, -20, 20) if a.covers(mv.x + d * dx, -6)]
        starts += [Body(0.0, spot - d * 60, a, run=d)] if a.covers(spot - d * 60, -6) else []
        for b in starts:
            for _ in range(3):
                leg = flow.leg(mv, b, Keys(KEYS))
                assert leg is not None, (mv, b.x)
                x = leg.spans[-1].x1
                land, lx, _, _ = flight(m, x, a.y_at(x), d * WALK_SPEED, -JUMP_SPEED, group=a.group)
                assert land is q and q.covers(lx - LANDING_ROOM) and q.covers(lx + LANDING_ROOM), (mv, x, lx)
                assert fly(m, x - d * 8, a.y_at(x), d * WALK_SPEED, -JUMP_SPEED, group=a.group)[0] is q, (mv, x)
                assert fly(m, x, a.y_at(x), d * WALK_SPEED * 0.94, -JUMP_SPEED, group=a.group)[0] is q, (mv, x)
    return len(jumps)


@REAL
def test_every_route_jump_takes_off_where_it_lands_with_room_to_spare(old_route_takeoffs):
    """_route_jumps_land_with_room at the route's takeoffs of the night of 2026-10-21 (old_route_takeoffs), its 81
    jumps. Before Flow.sure_takeoff, 47 of the 81 had takeoffs a chain could pick that failed it, landing as little
    as 2.6 units onto their platform."""
    assert _route_jumps_land_with_room() == 81


@REAL
def test_every_placed_route_jump_takes_off_where_it_lands_with_room_to_spare_and_flow_takes_its_takeoff():
    """_route_jumps_land_with_room at the route's takeoffs placed for room (route._placed: 55 jumps, a jump off an
    end and one along the platform that take off at the same spot being one), the least room 16.5 units, not 9.2.
    And the route and the chains agree: from standing at its spot and running in, Flow takes every route jump off
    from its own takeoff (Flow._jump_from passes sure_takeoff there), and from the 10 units before it that its rng
    may pick (_takeoffs) for all but the 4 jumps down from along a platform (10 -> 15, 13 -> 20, 18 -> 32, 19 ->
    24) where a slower takeoff from further back comes down on her own platform's end: there no takeoff lands
    with room from all of them (route._placed took the one with the most under the slop alone), and the chain
    takes off from the route's."""
    from gamelab import directions
    assert _route_jumps_land_with_room() == 55
    m = scenarios.real_map()
    flow, narrow = Flow(m, KEYS), set()
    for mv in (mv for mv in Planner(m).moves if mv.kind == "jump"):
        a, d = m.platform(mv.src), directions.sign(mv.direction)
        assert mv.takeoff is not None and (a.right - mv.takeoff if d > 0 else mv.takeoff - a.left) >= 15, mv
        for b in (Body(0.0, mv.x, a), Body(0.0, mv.takeoff - d * 60, a, run=d)):
            if not a.covers(b.x, -6):
                continue
            assert flow._jump_from(mv, b, Keys(KEYS), "run", mv.takeoff) is not None, (mv, b)
            if any(flow._jump_from(mv, b, Keys(KEYS), "run", mv.takeoff - d * v) is None for v in range(2, 11, 2)):
                narrow.add((mv.src, mv.dst))
    assert narrow == {(10, 15), (13, 20), (18, 32), (19, 24)}


@REAL
def test_a_climb_from_under_its_rope_has_her_at_the_top_when_the_world_does():
    """Standing under a rope 40-41 above her (6 at 723, 30 at 795, 36 at 539, 14 at 798), the plan's top of the
    climb is the world's to 15 ms, whatever the phase of the game's steps: she grabs the rope as her feet come
    within ROPE_ARMS of its bottom (0.045 s up), on her feet at its bottom. Timed 0.15 s after leaving the ground,
    it had her there 0.10 s late, and a run off the top to a jump 15 units from the end went off it first
    (test_sim: test_a_climb_from_under_its_rope_then_a_run_to_a_jump_lands_on_every_seed)."""
    from gamelab.motion.flow import CLIMB_OVER
    from gamelab.sim.world import World
    from test_sim import KEYS as WORLD_KEYS
    m = scenarios.real_map()
    moves = Planner(m).moves
    for src, x, dst in ((6, 723.0, 3), (30, 795.0, 25), (36, 539.0, 32), (14, 798.0, 6)):
        climb = next(mv for mv in moves if (mv.kind, mv.src, mv.dst) == ("climb", src, dst))
        assert abs(climb.rope_x - x) <= 4
        legs, keys, why = Flow(m, KEYS).chain([climb], Body(0.0, x, m.platform(src)))
        assert why == "done"
        for phase in (0, 10, 25):
            w = World(m, WORLD_KEYS, (src, x), monsters=None, step_phase=phase)
            for at, name, how in keys.sorted():
                w.key(at / 1000, name, how == "down")
            w.advance(legs[-1].end.t + 1.0)
            top = next(float(line.split()[0]) for line in w.log if f"climbs onto platform {dst}" in line)
            assert top == pytest.approx(legs[-1].end.t - CLIMB_OVER, abs=0.015), (src, dst, phase)
