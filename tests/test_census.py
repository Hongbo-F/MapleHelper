"""The census of the map's monsters and the trips it leads (behavior.patrol, from 2026-11-03: the pile-up study,
D:/MapleHelper-scratch/pileup): every spawn point's monsters, a kind she doesn't hunt too, in view what she sees, out
of view what she last saw and her respawns she didn't see come back, the whole held to the map's cap; another player's
dot by a point; the user's gate (her view's spawn points' monsters against the map's average over the last OCC_S); her
kill rate and handling time; a trip to what it believes off her round; and with it off (patrol.CENSUS) the choice as it
was. On tests/test_grind's synthetic maps."""

import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gamelab.behavior import patrol
from gamelab.behavior.grind import Grind
from gamelab.behavior.hunt import Visit
from gamelab.localization.memory import Cover, Memory
from gamelab.localization.players import Player
from tests.test_grind import COLUMN, COLUMN_SPAWNS, GREEN, HORNY, TOUCH, _column, _map, _pid, _sight

BOTTOM = ((-150.0, 50.0),)  # a look's rows over the column's bottom platform (y 0) alone


def _look(m, b, t, sightings=(), players=None):
    """A look at time t covering the column's bottom platform (x -200..600), told to memory `m` and belief `b`."""
    m.update(t, list(sightings), Cover(t, BOTTOM, -200.0, 600.0), players)
    b.update(t, m, players)


def test_the_census_counts_every_spawn_point_hunted_or_not_and_holds_the_maps_cap():
    model = _map(COLUMN, COLUMN_SPAWNS[:4] + [(2150, 0, "g")])
    p = patrol.Patrol(model, {HORNY}, {HORNY: 41.0, GREEN: 26.0}, lambda a, x, b, bx: 2.0, touch=TOUCH)
    b = p.belief
    far = _pid(model, 2150, 0)
    # The Green Mushroom's point she doesn't hunt takes the cap like any: its census, never a stop's
    assert len(b.points) == 4 and [(q.platform, q.x, q.kind) for q in b.extra] == [(far, 2150.0, GREEN)]
    assert len(b.census) == 5 and sum(b.census.values()) == pytest.approx(b.cap)
    assert all(far != s.platform for s in p.stops)


def test_a_kill_whose_monster_she_doesnt_see_come_back_is_one_born_out_of_view():
    model, ids, far, p = _column()
    b, m = p.belief, Memory(model)
    q = next(q for q in b.points if q.platform == ids[0])
    _look(m, b, 0.0, [_sight(model, HORNY, 150, 0)])
    out = [i for i in b.census if i != q.id]
    assert b.in_view == {q.id} and b.census[q.id] == 1.0
    assert sum(b.census[i] for i in out) == pytest.approx(b.cap - 1)  # (the cap less what's in view)
    m._killed(m.entries[0], 1.0, 150.0)
    _look(m, b, 1.0)
    assert b.pending == [1.0] and b.census[q.id] == 0.0
    assert sum(b.census[i] for i in out) == pytest.approx(b.cap - 1)  # (its monster not back yet: the pass to come)
    # The map's pass after it: no newborn seen fading in by her point, so it came back out of view
    _look(m, b, 1.0 + patrol.TICK_S + 0.1)
    assert b.pending == [] and b.due_n == 1 and b.births_seen == 0 and b.assigned == 1.0
    assert b.census[q.id] == 0.0 and sum(b.census[i] for i in out) == pytest.approx(b.cap)
    # ... and one seen fading in at a point in view accounts for her next kill's: nothing more out of view
    _look(m, b, 20.0, [replace(_sight(model, HORNY, 160, 0), mob=replace(_sight(model, HORNY, 160, 0).mob, fading=0.3))])
    assert b.births_seen == 1 and b.census[q.id] == 1.0
    m._killed(m.entries[-1], 21.0, 160.0)
    _look(m, b, 21.0)
    _look(m, b, 21.0 + patrol.TICK_S + 0.1)
    assert b.due_n == 2 and b.assigned == 1.0


def test_unseen_births_go_where_points_are_likeliest_clear():
    model, ids, far, p = _column()
    b = p.belief
    qs = {q.platform: q.id for q in b.points}
    b.in_view = {qs[ids[0]]}
    for i in b.census:
        b.census[i] = 0.0
    b.census[qs[ids[1]]] = 3.0  # a platform believed full: a point there is seldom clear
    b._spread(1.0)
    got = {pid: b.census[i] for pid, i in qs.items()}
    assert got[ids[0]] == 0.0 and sum(got.values()) == pytest.approx(4.0)
    assert got[ids[1]] - 3.0 < got[ids[2]] == pytest.approx(got[ids[3]]) == pytest.approx(got[far])


def test_another_players_dot_by_a_point_out_of_view_takes_its_monsters():
    model, ids, far, p = _column()
    b, m = p.belief, Memory(model)
    q = next(q for q in b.points if q.platform == ids[2])
    _look(m, b, 0.0)
    assert b.census[q.id] > 0
    _look(m, b, 1.0, players=[Player(1, "stranger", 150.0, -400.0, ids[2], (0.0, 0.0), 1.0, 1.0)])
    assert b.census[q.id] == 0.0 and b.rival_t == 1.0


def _gated(p, t, monsters_in_view: float):
    """Her view the column's bottom point for the last OCC_S, holding this many monsters at every look; her kills a
    few over the last BAR_S."""
    b = p.belief
    q = next(q for q in b.points if q.platform == p.stops[0].platform)
    b.in_view, b.t0 = {q.id}, t - 200.0
    b.occ = [(t - k, 1.0, monsters_in_view) for k in range(int(patrol.OCC_S) - 1, -1, -1)]
    b.all_kills = [t - 150.0, t - 100.0, t - 50.0]


def test_the_gate_opens_when_her_views_points_hold_under_their_share_of_the_cap():
    model, ids, far, p = _column()
    b, t = p.belief, 300.0
    share = b.cap / (len(b.points) + len(b.extra))  # the map's average a point (2 of 5)
    _gated(p, t, 0.0)
    assert b.ratio(t) == 0.0 and p.bar(t) == pytest.approx(3 / patrol.BAR_S)
    _gated(p, t, patrol.OCC_MAX * share * 0.99)
    assert b.ratio(t) == pytest.approx(patrol.OCC_MAX * 0.99) and p.bar(t) is not None
    _gated(p, t, patrol.OCC_MAX * share * 1.01)  # her view holding its share: the mobs aren't piling elsewhere
    assert p.bar(t) is None
    b.t0 = t - patrol.OCC_S / 2 + 1.0  # too little seen yet
    assert b.ratio(t) is None and p.bar(t) is None


def test_a_trip_goes_where_the_census_believes_a_pile_off_her_round_and_beats_her_kill_rate():
    model, ids, far, p = _column()
    b, t = p.belief, 300.0
    p.replan(t, force=True)
    here, fq = p.stop_on(ids[0]), next(q.id for q in b.points if q.platform == far)
    assert p.stop_on(far) not in p.plan.stops
    _gated(p, t, 0.0)
    b.census[fq] = 3.0  # three believed on the far platform, 30 s off
    j, info = p.choose(t, ids[0], 150.0, current=here)
    assert p.platform_of(j) == far and info["why"].startswith("trip of 1 stops") and info["trip"] == [far]
    assert p.trip == [p.stop_on(far)] and p.trips == 1 and info["monsters"] == pytest.approx(3.0)
    assert info["monsters"] >= patrol.PULL_GAIN * info["bar"] / 60 * info["detour"]
    # On her way: the trip's stop again, its pile believed there yet
    j2, info2 = p.choose(t + 1, ids[0], 150.0, heading=j)
    assert j2 == j and info2["why"] == "trip (1 to go)"
    # Got there: the trip done, the loop goes on
    j3, info3 = p.choose(t + 40, far, 2150.0, current=j)
    assert p.trip is None and not info3["why"].startswith("trip")


def test_no_trip_with_the_gate_shut_nor_when_her_kill_rate_beats_it():
    model, ids, far, p = _column()
    b, t = p.belief, 300.0
    p.replan(t, force=True)
    here, fq = p.stop_on(ids[0]), next(q.id for q in b.points if q.platform == far)
    _gated(p, t, 1.0)  # her view holding more than its share
    b.census[fq] = 3.0
    assert not p.choose(t, ids[0], 150.0, current=here, peek=True)[1]["why"].startswith("trip")
    _gated(p, t, 0.0)
    b.all_kills = [t - k for k in range(0, 180, 3)]  # a kill every 3 s: three monsters over a 70 s detour don't pay
    b.census[fq] = 3.0
    assert not p.choose(t, ids[0], 150.0, current=here, peek=True)[1]["why"].startswith("trip")


def test_her_handling_time_a_kill_is_pooled_over_her_last_stops():
    model, ids, far, p = _column()
    assert p.handle_s() is None
    for _ in range(3):
        p.handled(10.0, 2)
    assert p.handle_s() is None  # (6 kills: under HANDLE_MIN, her kill time stands in)
    p.handled(4.0, 2)
    assert p.handle_s() == pytest.approx(34.0 / 8)
    for _ in range(patrol.HANDLE_N):
        p.handled(6.0, 3)
    assert len(p._handles) == patrol.HANDLE_N and p.handle_s() == pytest.approx(2.0)


def test_a_stop_she_leaves_tells_her_handling_to_the_patrol():
    got = []
    g = object.__new__(Grind)
    now = time.monotonic()
    g.visit = Visit(3, SimpleNamespace(), arrived=now - 10.0, kills=[now - 8.0, now - 3.0])
    g.patrol, g._left, g.stays = SimpleNamespace(handled=lambda s, k: got.append((s, k))), {}, []
    g._leave_ground()
    assert got == [(pytest.approx(7.0), 2)]
    g.visit = Visit(3, SimpleNamespace(), arrived=now - 10.0)  # (no kill there: nothing)
    g._leave_ground()
    assert len(got) == 1


def test_with_the_census_off_the_choice_is_as_before(monkeypatch):
    model, ids, far, p = _column()
    b, t = p.belief, 300.0
    p.replan(t, force=True)
    _gated(p, t, 0.0)
    b.census[next(q.id for q in b.points if q.platform == far)] = 3.0
    monkeypatch.setattr(patrol, "CENSUS", False)
    here = p.stop_on(ids[0])
    assert p.choose(t, ids[0], 150.0, current=here, peek=True) == p._choose(t, ids[0], 150.0, current=here, peek=True)
    assert not p.choose(t, ids[0], 150.0, current=here)[1]["why"].startswith("trip")
