"""The map's spawn pass learned from the births she sees (localization.spawnclock), the free spawn points by the
server's rule and their newborns' price (attack.danger: respawn_points, respawn_risk), and the spawn guard that holds
a trip to a pile for the pass (behavior.hunt: _respawn_guard, _go_for, _off_points; the cornered clip's R3,
2026-11-03)."""

import math
import time
from types import SimpleNamespace

import pytest

from gamelab.attack import danger as danger_mod
from gamelab.attack.danger import NEWBORN_SEEN, NEWBORN_TOUCH, RESPAWN_NEAR, SPAWN_CLEAR, Danger
from gamelab.behavior import hunt
from gamelab.behavior.ledger import Pile
from gamelab.localization.spawnclock import PERIOD_S, SOLID_WEIGHT, SpawnClock
from gamelab.perception.spawns import FADE_PER_S


def _event(kind, t, fading=None):
    return SimpleNamespace(kind=kind, t=t, data={"kind": "horny-mushroom", **({"fading": fading} if fading is not None
                                                                          else {})})


def test_the_clock_learns_the_pass_from_newborns_seen_fading_in():
    """Each newborn's birth is its sighting less its alpha over FADE_PER_S; two passes' births agreeing make it sure,
    and then the passes ahead are on its grid: the next after a moment, and how likely an interval meets a window about
    one (0 well between two, 1 across one)."""
    c = SpawnClock()
    assert not c.sure and c.next_pass(100.0) is None
    base = 1000.0
    c.see([_event("spawned", base + 0.2, fading=0.2 * FADE_PER_S), _event("entered", base + 0.5)])
    assert c.births == 1 and not c.sure
    c.see([_event("spawned", base + PERIOD_S + 0.1, fading=0.1 * FADE_PER_S)])
    assert c.sure and abs(c.next_pass(base + 3.0) - (base + PERIOD_S)) < 0.01
    assert abs(c.next_pass(base + PERIOD_S + 0.5) - (base + 2 * PERIOD_S)) < 0.02
    T = base + 2 * PERIOD_S
    assert c.chance(T - 3.0, T - 2.0, -0.15, 1.0) < 1e-6          # well before the next pass
    assert c.chance(T - 0.5, T + 0.5, -0.15, 1.0) > 0.999         # across it
    assert c.chance(T + 1.5, T + 2.0, -0.15, 1.0) < 1e-6          # after its window
    assert c.record()["passes"] == 2


def test_before_a_phase_the_window_is_as_likely_as_its_share_of_a_period():
    c = SpawnClock()
    assert c.chance(0.0, 1.0, -0.15, 1.0) == pytest.approx(2.15 / PERIOD_S)
    assert c.chance(0.0, 10.0, -0.15, 1.0) == 1.0


def test_births_of_one_pass_are_one_and_a_stray_found_solid_doesnt_move_the_phase():
    """Births within SAME_PASS_S are one pass's; a track re-found by a point (solid, SOLID_WEIGHT) off the grid is out
    of the fit (INLIER_S), the phase kept."""
    c = SpawnClock()
    for k in range(4):
        t = 500.0 + k * PERIOD_S
        c.birth(t, 1.0)
        c.birth(t + 0.05, 1.0)
    assert len(c.passes) == 4
    c.birth(500.0 + 4 * PERIOD_S + 3.0, SOLID_WEIGHT)  # (a stray, mid-period)
    nxt = c.next_pass(500.0 + 4 * PERIOD_S + 3.5)
    assert abs(nxt - (500.0 + 5 * PERIOD_S)) < 0.06


def test_the_period_is_fitted_once_the_passes_span_long_enough_within_its_slack():
    """The server's period live 7.1146 s (both hours of 2026-11-02); a run whose passes come 7.115 s apart (the
    simulator's TICK_S) over minutes is fitted to it, one at 7.3 isn't believed past PERIOD_SLACK."""
    c = SpawnClock()
    for k in range(40):
        c.birth(10.0 + k * 7.115, 1.0)
    assert c.period == pytest.approx(7.115, abs=0.002)
    d = SpawnClock()
    for k in range(40):
        d.birth(10.0 + k * 7.3, 1.0)
    assert d.period <= PERIOD_S + 0.05 + 1e-9


# --- the free points by the server's rule, and their newborns' price ----------------------------------------------

def _danger():
    from test_danger import HER, build
    m = build([(-220, 500, -6, 500), (22, 530, 210, 530)], [("19", -207, 500), ("19", -94, 500), ("13", 162, 530)])
    return m, Danger(m, HER)


def _mon(kind, pid, x, model):
    from gamelab.localization.monsters import Monster
    from gamelab.perception.mobs import Mob
    return Monster(Mob(kind, 0, 0, 30, 30, 0.9), x, model.platform(pid).y_at(x), pid)


def test_a_point_is_free_with_no_monster_of_any_kind_within_the_servers_box_and_counts_near_her_stretch():
    m, d = _danger()
    seven = m.platform_under(-150, 500).id
    eight = m.platform_under(100, 530).id
    assert d.respawn_points(seven, -160, -140, []) == [(-207, "horny-mushroom"), (-94, "horny-mushroom")]
    assert d.respawn_points(seven, -30, -30, []) == [(-94, "horny-mushroom")]        # (-207 is 177 off)
    assert d.respawn_points(seven, -94 + RESPAWN_NEAR + 1, -6, []) == []
    # any monster within SPAWN_CLEAR across and up or down keeps it, of any kind, on any platform
    near = _mon("green-mushroom", eight, 30.0, m)                                      # 124 from -94: not
    assert d.respawn_points(seven, -160, -140, [near]) == [(-207, "horny-mushroom"), (-94, "horny-mushroom")]
    held = _mon("slime", seven, -94 + SPAWN_CLEAR - 1, m)
    assert d.respawn_points(seven, -160, -140, [held]) == [(-207, "horny-mushroom")]
    # a kind under DANGER_MIN isn't one: platform 8's Green Mushroom (7.2% of her HP) counts, as at 752 HP
    assert d.respawn_points(eight, 150, 170, []) == [(162, "green-mushroom")]


def test_a_newborns_price_is_its_fill_the_touch_it_adds_and_its_kinds_share():
    m, d = _danger()
    pts = [(-207, "horny-mushroom"), (-94, "horny-mushroom")]
    got = d.respawn_risk(pts, 0.27, 1.0)
    assert got == pytest.approx(2 * 0.27 * (NEWBORN_TOUCH - NEWBORN_SEEN) * d.share("horny-mushroom"))
    assert d.respawn_risk(pts, 0.27, 0.0) == 0.0


# --- the guard: a trip to a pile held for the pass ------------------------------------------------------------------

def _guarded(monkeypatch, pass_in: float, here_x: float = -60.0, travel: float = 0.6, rate: float = 6.85,
             hurt: float = 0.7):
    """A hunt on platform 7 (two Horny Mushroom points, -207 and -94) by a pile at -157, its clock sure with the next
    pass `pass_in` s from now, her landing prices given."""
    m, d = _danger()
    seven = m.platform_under(-150, 500).id
    now = time.monotonic()
    clock = SpawnClock()
    for k in (1, 2, 3):
        clock.birth(now + pass_in - k * PERIOD_S, 1.0)
    h = object.__new__(hunt.Hunt)
    h.h = SimpleNamespace(spawn_clock=clock, danger=d, ledger=None, max_hp=752.0, report=SimpleNamespace(touches=0,
                                                                                                         misses=0))
    h.model, h._unseen = m, (lambda s: [])
    h._travel_to_pile = lambda s, p: travel
    h.log = lambda *a: None
    from gamelab.attack.stance import Prices
    monkeypatch.setattr(hunt.fight, "landing_prices",
                        lambda hh, dd: Prices(lambda kind: dd.taken(kind) * 752.0 / 4.0, rate, hurt))
    pile = Pile(-157.0, 500.0, seven, now - 3.0, True, status="seen", xs=(-157.0,), last_seen=now)
    s = SimpleNamespace(platform=m.platform(seven), where=SimpleNamespace(x=here_x, y=500.0), monsters=[], swept=[],
                        tracks=[], t=now, players=[])
    return h, s, pile


def test_a_trip_whose_stand_meets_the_pass_waits_for_it_when_the_newborns_cost_more_than_the_wait(monkeypatch):
    """Going, the pass 0.2 s off, has her by both points about it (-94 on her walk there, -207 by the pile), waiting
    by -94's alone (34 off it, and on 7 nowhere to step off): -207's newborn more (0.27 x 0.56 x 0.7 of a touch's 30.3
    EXP: 3.2) against 0.4 s of her 6.85 EXP a second (2.7): she waits. With the pass 2 s off (15.1 EXP of waiting) she
    goes; with it past and seen (its window over) she goes; with no clock sure she goes."""
    h, s, pile = _guarded(monkeypatch, 0.2)
    why = h._respawn_guard(s, pile, 0.6)
    assert why is not None and why.startswith("a horny-mushroom spawn point at x -207 on")
    assert "(its newborn 3.2 EXP more than waiting, the wait 2.7)" in why
    h, s, pile = _guarded(monkeypatch, 2.0)
    assert h._respawn_guard(s, pile, 0.6) is None
    h, s, pile = _guarded(monkeypatch, -0.5)  # (passed: its newborns, if any, the look's monsters)
    assert h._respawn_guard(s, pile, 0.6) is None
    h, s, pile = _guarded(monkeypatch, 0.2)
    h.h.spawn_clock = SpawnClock()
    assert h._respawn_guard(s, pile, 0.6) is None
    h, s, pile = _guarded(monkeypatch, 0.2)
    monkeypatch.setattr(hunt, "RESPAWN_GUARD", False)
    assert h._respawn_guard(s, pile, 0.6) is None


def test_a_stand_done_before_the_pass_goes_and_a_held_point_doesnt_count(monkeypatch):
    h, s, pile = _guarded(monkeypatch, 3.0, travel=0.3)  # (her stand over 1.4 s before the pass)
    assert h._respawn_guard(s, pile, 0.3) is None
    h, s, pile = _guarded(monkeypatch, 0.2)
    m = h.model
    s.monsters = [_mon("horny-mushroom", s.platform.id, -100.0, m), _mon("horny-mushroom", s.platform.id, -200.0, m)]
    assert h._respawn_guard(s, pile, 0.6) is None


def test_waiting_for_the_pass_by_a_free_point_she_steps_off_it_where_the_platform_has_room(monkeypatch):
    """On a platform with room past the points' RESPAWN_NEAR she waits there (the nearest, OFF_MARGIN past it); on 7,
    214 wide with its two points 113 apart, none at GROUND_MARGIN from its ends: she waits where she is."""
    h, s, pile = _guarded(monkeypatch, 0.2)
    h._in_the_way = lambda s, x, g: False
    assert h._off_points(s) is None  # (7: nowhere 75 off both points)
    from test_danger import HER, build
    wide = build([(-400, 500, 400, 500)], [("19", 0, 500)])
    h.h.danger, h.model = Danger(wide, HER), wide
    p = wide.platform_under(0, 500)
    s2 = SimpleNamespace(platform=p, where=SimpleNamespace(x=30.0, y=500.0), monsters=[], swept=[], tracks=[],
                         t=s.t, players=[])
    side, units, goal = h._off_points(s2)
    assert side == "right" and goal >= RESPAWN_NEAR + hunt.OFF_MARGIN and math.isclose(units, goal - 30.0)
