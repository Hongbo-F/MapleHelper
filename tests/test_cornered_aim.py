"""Her aim up close by a wall (the cornered study's R6 and R8, 2026-11-03: D:/MapleHelper-scratch/cornered): a star's
line checked to the client's own hit point too (reach.CLIENT_HIT_POINT: a monster at the foot of a wall rising from its
platform's end, its body over the wall, takes no star from a jump in place close by, the line going into the wall), a
jump throw up close counted sure only with its own key's line clear (stance._from_here), and up close the throw at
whichever monster reaches her first, the sure one the other way in place of an unsure jump (stance.FIRST_TO_REACH)."""

import math

import pytest

from maplehelper.engine import LEFT, RIGHT

from gamelab.attack import reach
from gamelab.attack import stance as st
from gamelab.localization import mapdata
from gamelab.localization.mapdata import Platform, Segment
from gamelab.localization.monsters import Monster
from gamelab.motion import physics
from gamelab.perception.mobs import Mob

GREEN = "green-mushroom"
TOPS, HALVES = {GREEN: 51.0}, {GREEN: 28.0}
BODY = (TOPS[GREEN], HALVES[GREEN])
RISE = physics.jump_rise(physics.JUMP_ATTACK_MS + physics.JUMP_LAG * 1000)  # her feet when a jump throw's attack picks


def _fh(x1, y1, x2, y2, prev=0, nxt=0):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "prev": prev, "next": nxt}


def corner():
    """A floor (0..600, y 500) with a wall rising 200 from its left end (drawn downward: its face looks right, into the
    room), one foothold group: as Tree Dungeon's 7 and the trunk's wall at x -220."""
    tree = {"info": {"VRLeft": -200, "VRTop": 0, "VRRight": 800, "VRBottom": 800},
            "foothold": {"1": {"1": {"1": _fh(0, 300, 0, 500, 0, 2), "2": _fh(0, 500, 600, 500, 1, 0)}}}}
    model = mapdata.parse(1, tree, "Corner", {})
    (p,) = model.platforms
    return model, p


def mon(x, platform=0, y=500.0, bar=None):
    return Monster(Mob(GREEN, 0, 0, 30, 30, 0.9), x, y, platform, bar)


def stance(x, ms, p, facing=LEFT, pace=st.NO_PACE, **kw):
    return st.stance(x, p.y_at(x), p, ms, False, facing, tops=TOPS, halves=HALVES, pace=pace, **kw)


def test_the_clients_hit_point_past_a_wall_its_body_is_over_stops_the_star():
    _, p = corner()
    (wall,) = p.walls
    assert reach.faces(wall) == 1
    # A Green Mushroom 13 from the wall (as the Horny Mushroom at -207 by the trunk at -220), its body 28 either side:
    # over the wall by 15. From 59 off, up in a jump throw, the shot's row starts 65 ahead of her, inside its body: the
    # client's hit point is the middle of -15..7, behind the wall; taken inside its platform (x 1) it was clear.
    hx, _ = reach.hit_point(72, 500 - RISE - reach.SHOT_H, -1, 13, 500, *BODY)
    assert hx == pytest.approx(-4.0)
    assert reach.wall_between(p, 72, 500, 13, RISE, p, ty=500, body=BODY) is wall
    assert reach.wall_between(p, 72, 500, 13, RISE, p, ty=500, body=BODY, client=False) is None
    # Farther off the row starts past its middle's side of the wall: the two points agree, clear either way.
    for x in (113, 200):
        assert reach.wall_between(p, x, 500, 13, RISE, p, ty=500, body=BODY) is None
    # Its platform not known: the hit point as is (the client's), stopped as before.
    assert reach.wall_between(p, 72, 500, 13, RISE, ty=500, body=BODY) is wall


def test_her_decisions_keep_to_it_and_the_old_rule_is_one_switch_away(monkeypatch):
    _, p = corner()
    green = mon(13)
    assert reach.wall_to(p, 72, 500, green, RISE) is not None  # (a pig's body by default: over the wall by 21)
    monkeypatch.setattr(reach, "CLIENT_HIT_POINT", False)
    assert reach.wall_to(p, 72, 500, green, RISE) is None


def test_clear_length_answers_as_wall_between_by_the_clients_point_too():
    # spots' coverage tries every line at once (reach._sight_clear): the same answers as one at a time, here where the
    # client's point is behind the wall (from x 72, the first 20 units of the floor, a body over the wall).
    _, p = corner()
    lo, hi, step = 1.0, 60.0, 2.0
    n = max(1, math.ceil((hi - lo) / step))
    width = (hi - lo) / n
    xs = [reach._inside(lo + (i + 0.5) * width, p) for i in range(n)]
    one = width * sum(reach.wall_between(p, 72, 500, t, target=p, ty=500, body=BODY) is None for t in xs)
    assert reach.clear_length(p, 72, 500, p, lo, hi, BODY, step) == pytest.approx(one)
    assert one < hi - lo  # (some of it behind the wall)


def test_up_close_a_jump_throw_into_the_wall_isnt_sure_she_steps_away_first(monkeypatch):
    # Her at 72 facing the Green at the wall's foot 59 off: as before a jump throw in place, its attack sure to pick it
    # (reach.air_sure: 54 for a mushroom) and its line taken clear; the client's line goes into the wall: no sure throw
    # from here, so a step away to where it is (from 100 off its line clears).
    _, p = corner()
    green = mon(13)
    monkeypatch.setattr(reach, "CLIENT_HIT_POINT", False)
    old = stance(72, [green], p)
    assert old[0] == "jump_throw" and old[1] == LEFT and st.plan_of(old).target is green
    monkeypatch.setattr(reach, "CLIENT_HIT_POINT", True)
    here = st._from_here(72, 500, LEFT, green, LEFT, lambda a, b: True, *BODY, [green], HALVES, p, st.NO_PACE)
    assert not here.sure
    act = stance(72, [green], p)
    assert act[0] == "back" and act[1] == RIGHT
    # From where it sends her, the jump throw is sure and clear.
    x = 72 + act[2]
    there = st._from_here(x, 500, LEFT, green, LEFT, lambda a, b: True, *BODY, [green], HALVES, p, st.NO_PACE)
    assert there.sure and reach.wall_between(p, x, 500, 13, RISE, p, ty=500, body=BODY) is None


def test_hemmed_in_she_throws_at_the_one_she_surely_hits_not_into_the_wall(monkeypatch):
    # Episode A (22:50:25-51): the one by the wall 59 off on the side she faces, another 56 off the other way, no room
    # to step or jump back. The jump at the one by the wall is no longer sure, and with FIRST_TO_REACH the sure one the
    # other way comes in its place (turning: a tap first, its attack 40 ms after the jump key).
    _, p = corner()
    wall_side, other = mon(13), mon(128)
    monkeypatch.setattr(st, "FIRST_TO_REACH", False)
    act = stance(72, [wall_side, other], p)
    assert act[0] == "jump_throw" and act[1] == LEFT  # unsure, into the wall
    monkeypatch.setattr(st, "FIRST_TO_REACH", True)
    act = stance(72, [wall_side, other], p)
    assert act[0] == "jump_throw" and act[1] == RIGHT and st.plan_of(act).target is other
    # Facing the other one already: thrown at as before, no turn.
    act = stance(72, [wall_side, other], p, facing=RIGHT)
    assert act[0] == "jump_throw" and act[1] == RIGHT and st.plan_of(act).why == "close"


def test_with_both_sure_the_one_that_reaches_her_first(monkeypatch):
    # A flat floor with no walls: one Green 70 off on the side she faces standing, another 100 off behind her walking at
    # her at 95 units/s (on her in about 0.5 s; still 70 off when a jump throw tapping round first picks). Both throws
    # from where she stands are sure; the walker reaches her first: thrown at, where she'd have thrown the way she faces.
    p = Platform(0, [Segment(0, 500, 1000, 500)])
    ahead, walker = mon(570), mon(400)
    pace = st.Pace(0.0, lambda m: 95.0 if m is walker else 0.0, {GREEN: 100.0}, HALVES)
    for t in (ahead, walker):
        side = RIGHT if t.x > 500 else LEFT
        assert st._from_here(500, 500, side, t, RIGHT, lambda a, b: True, *BODY, [ahead, walker], HALVES, p,
                             pace).sure
    assert st._contact_s(walker, 500, pace, p, HALVES, 1.0) < 1.0 and st._contact_s(ahead, 500, pace, p, HALVES, 1.0) \
        == float("inf")
    monkeypatch.setattr(st, "FIRST_TO_REACH", False)
    assert stance(500, [ahead, walker], p, facing=RIGHT, pace=pace)[:2] == ("jump_throw", RIGHT)
    monkeypatch.setattr(st, "FIRST_TO_REACH", True)
    act = stance(500, [ahead, walker], p, facing=RIGHT, pace=pace)
    assert act[:2] == ("jump_throw", LEFT) and st.plan_of(act).target is walker
    # Neither coming at her: the more hurt first, else the side she faces, as before.
    still = st.Pace(0.0, lambda m: 0.0, {GREEN: 100.0}, HALVES)
    assert stance(500, [ahead, walker], p, facing=RIGHT, pace=still)[:2] == ("jump_throw", RIGHT)
    hurt = mon(400, bar=0.4)
    act = stance(500, [ahead, hurt], p, facing=RIGHT, pace=still)
    assert act[:2] == ("jump_throw", LEFT) and st.plan_of(act).target is hurt


def test_the_magicians_close_quarters_are_as_they_were(monkeypatch):
    # Without POINT_BLANK (her Magic Claw, cast from the ground only, no wall stops it) FIRST_TO_REACH changes nothing.
    p = Platform(0, [Segment(0, 500, 1000, 500)])
    ahead, walker = mon(570), mon(400)
    pace = st.Pace(0.0, lambda m: 95.0 if m is walker else 0.0, {GREEN: 100.0}, HALVES)
    monkeypatch.setattr(st, "POINT_BLANK", False)
    on = stance(500, [ahead, walker], p, facing=RIGHT, pace=pace)
    monkeypatch.setattr(st, "FIRST_TO_REACH", False)
    assert stance(500, [ahead, walker], p, facing=RIGHT, pace=pace) == on


def test_the_simulators_world_stops_such_a_star_whatever_her_decisions_take(monkeypatch):
    # The world's line (World.sight_stop) goes to the client's own hit point too, with her decisions' switch off as on;
    # `--world hit-inside` (Realism.hit_inside) puts the old world back.
    from test_sim import KEYS
    from gamelab.sim.world import Realism, World
    model, p = corner()
    for hit_inside, stopped in ((False, True), (True, False)):
        w = World(model, KEYS, (p.id, 72.0), monsters=[("Green Mushroom", p.id, 13.0)],
                  realism=Realism(hit_inside=hit_inside))
        w.y = 500 - RISE  # (up in a jump throw, as its attack picks)
        monkeypatch.setattr(reach, "CLIENT_HIT_POINT", False)
        assert (w.sight_stop(w.monsters[0]) is not None) == stopped
        monkeypatch.setattr(reach, "CLIENT_HIT_POINT", True)
        assert (w.sight_stop(w.monsters[0]) is not None) == stopped


def test_a_skills_row_from_her_x_as_the_port_has_it_puts_the_hit_point_inside(monkeypatch):
    # The row's start, 65 ahead of her, is the star search's (reach.SEARCH_START), assumed for the row (the review of
    # 2026-11-03): RustMS's port starts a skill's row (Lucky Seven's) at her x, so the hit point is the middle of the
    # body's part from -15 to 41, 13 by its feet, inside the platform: clear. `--world hit-row-0` (Realism.skill_row 0)
    # gives the world's skill stars so; its regular attack's row as before (the port's whack reach, up to 65).
    from test_sim import KEYS
    from gamelab.sim.world import Realism, World
    model, p = corner()
    hx, _ = reach.hit_point(72, 500 - RISE - reach.SHOT_H, -1, 13, 500, *BODY, start=0.0)
    assert hx == pytest.approx(13.0)
    assert reach.wall_between(p, 72, 500, 13, RISE, p, ty=500, body=BODY, row_start=0.0) is None
    w = World(model, KEYS, (p.id, 72.0), monsters=[("Green Mushroom", p.id, 13.0)], realism=Realism(skill_row=0.0))
    w.y = 500 - RISE
    assert w.sight_stop(w.monsters[0], skill=True) is None
    assert w.sight_stop(w.monsters[0]) is not None
    w = World(model, KEYS, (p.id, 72.0), monsters=[("Green Mushroom", p.id, 13.0)])
    w.y = 500 - RISE
    assert w.sight_stop(w.monsters[0], skill=True) is not None
