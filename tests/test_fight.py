"""The fight loop (behavior.fight.clear) on prepared looks and a fake clock: hidden monsters, kills
from her EXP, the kill count, far targets, the star's path; and the hunter counting EXP gains."""

import time
from types import SimpleNamespace

import numpy as np

from maplehelper.config import GameKeys
from maplehelper.engine import LEFT, RIGHT

from gamelab.behavior.fight import EMPTY_PATIENCE, HIDDEN_FOR
from gamelab.behavior.hunter import Hunter
from gamelab.behavior.looting import LOOT_REACH, DropSpot
from gamelab.localization import mapdata
from gamelab.perception.eyes import Observation

from test_attack import PLAT, target

EMPTY_MAP = {"info": {"VRLeft": 0, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
             "miniMap": {"centerX": 0, "centerY": 0, "mag": 4}}


class _Eyes:
    """Hands out prepared looks (no game needed)."""
    want_ropes = False

    def __init__(self, looks):
        self.looks = iter(looks)

    def look(self):
        return next(self.looks)


def hunter(eyes=None) -> Hunter:
    return Hunter(eyes or _Eyes([]), GameKeys.__new__(GameKeys), mapdata.parse(1, EMPTY_MAP, "Test", {}),
                  log=lambda m: None)


def test_a_kill_is_confirmed_by_her_exp_number_changing():
    before = np.zeros((26, 140), bool)
    after = before.copy()
    after[5:15, 100:104] = True  # a digit changed
    looks = [Observation(t=t, frame=np.zeros((10, 10, 3), np.uint8), tag=None, hp=1.0, mp=1.0, exp=e)
             for t, e in ((1.0, before), (1.5, before), (2.0, after), (2.5, after))]
    h = hunter(_Eyes(looks))
    for _ in looks:
        h.look(mobs=False)
    assert h.exp_events == [2.0] and h.report.exp_kills == 1
    assert not h.take_exp(since=2.5)   # gains before she started throwing aren't this target's
    assert h.take_exp(since=1.8) and h.exp_events == []


def _fighter(monkeypatch, looks, on_look=None):
    """A Hunter whose looks come from `looks` (lists of monsters), 0.3 s apart on a fake clock; her
    throws are recorded instead of pressed."""
    h = hunter()
    clock, seq, throws = [0.0], iter(looks), []

    def look(*a, **k):
        clock[0] += 0.3
        if on_look:
            on_look(h, clock[0])
        return SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT, hp=1.0, monsters=next(seq, []))

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", lambda s: None)
    h.look, h.health_check = look, lambda s: None
    h.actions.throw = h.actions.jump_throw = h.actions.jump_back = lambda side: throws.append((round(clock[0], 1), side))
    h.looter.sweep_through = lambda s, deaths: None
    h.looter.mop_up = lambda stood=None: None  # looking for loot is tested on its own (test_loot.py)
    return h, throws


def test_she_keeps_throwing_at_a_monster_hidden_after_a_hit(monkeypatch):
    h, throws = _fighter(monkeypatch, [[target(700)]])  # seen once, then lost under its damage numbers
    r = h.clear(patience=2.0)
    assert len(throws) == 1 + round(HIDDEN_FOR / 0.3 - 1) and {side for _, side in throws} == {RIGHT}
    assert r.stop == "nothing left in reach"


def test_her_exp_going_up_ends_the_throws_at_a_hidden_monster(monkeypatch):
    def kill_at(h, t):
        if abs(t - 0.9) < 1e-6:
            h.report.exp_kills += 1
    h, throws = _fighter(monkeypatch, [[target(700)]], on_look=kill_at)
    r = h.clear(patience=2.0)
    assert len(throws) == 2 and r.kills == {"pig": 1}


def test_with_no_monster_in_sight_she_moves_on_soon(monkeypatch):
    h, throws = _fighter(monkeypatch, [])
    h.clear(patience=2.0)
    assert time.monotonic() <= 0.3 + EMPTY_PATIENCE + 0.3 and not throws
    # One seen, just out of reach: she waits the full patience for it to walk in.
    far = [[target(980, y=200)]] * 20
    h, throws = _fighter(monkeypatch, far)
    h.clear(patience=2.0, approach=False)
    assert time.monotonic() >= 2.0 and not throws


def test_drops_are_swept_once_she_lands_or_left_to_the_tour(monkeypatch):
    h, _ = _fighter(monkeypatch, [])
    swept = []
    h.looter.sweep_through = lambda s, deaths: swept.append([d.x for d in deaths])
    air = SimpleNamespace(where=None, platform=None)
    ground = SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT)
    looks = iter([air, air, ground])
    h.look = lambda *a, **k: next(looks)
    deaths = [DropSpot(600, 500, 0, 0.0), DropSpot(700, 560, 1, 0.0)]  # one on her platform, one below
    h.looter.sweep_after(deaths)
    assert swept == [[600, 700]] and [d.x for d in h.looter.pending] == [700]
    h.looter.pending, swept[:] = [], []
    h.look = lambda *a, **k: air  # never lands (on a rope, say): the tour collects them later
    h.looter.sweep_after(deaths)
    assert not swept and len(h.looter.pending) == 2


def test_she_stands_on_each_pile_pressing_pick_up(monkeypatch):
    h, _ = _fighter(monkeypatch, [])
    del h.looter.sweep_through  # the real one
    did = []
    h.actions.walk = lambda direction, units: did.append(("walk", direction, round(units)))
    h.actions.pick_here = lambda presses=4: did.append(("pick",))
    s = SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT)
    h.looter.sweep_through(s, [DropSpot(600, 500, 0, 0.0), DropSpot(400, 500, 0, 0.0)])
    # Nearer end first (both 100 away: the first found), over each pile, stopping on it.
    walks = [d for d in did if d[0] == "walk"]
    assert did.count(("pick",)) == 2
    assert sum(u for _, d, u in walks if d == RIGHT) - sum(u for _, d, u in walks if d == LEFT) in (
        600 + LOOT_REACH - 500, 400 - LOOT_REACH - 500)


def test_a_kill_after_she_moved_on_to_the_next_target_still_gets_its_drops(monkeypatch):
    # She threw at a pig on the right, then turned to one on the left; the right one's death (EXP up)
    # comes in after that, when the last throw aimed only left.
    def kill_at(h, t):
        if abs(t - 1.2) < 1e-6:
            h.report.exp_kills += 1
    looks = [[target(700)], [target(450)], [target(450)], [target(450)], [target(450)]]
    h, throws = _fighter(monkeypatch, looks, on_look=kill_at)
    swept = []
    h.looter.sweep_after = lambda deaths: swept.extend(d.x for d in deaths)
    h.clear(max_kills=1, patience=2.0)
    assert [side for _, side in throws][:2] == [RIGHT, LEFT]  # then at the one on her left
    assert h.report.kills == {"pig": 1} and swept == [700]   # not the one still in plain sight


def test_after_the_kill_count_she_finishes_what_is_in_reach(monkeypatch):
    def kill_at(h, t):
        if abs(t - 0.6) < 1e-6:
            h.report.exp_kills += 1
    looks = [[target(700), target(250)]] * 2 + [[target(250)]] * 3 + [[]] * 5
    h, throws = _fighter(monkeypatch, looks, on_look=kill_at)
    h.looter.sweep_after = lambda deaths: None
    r = h.clear(max_kills=1, patience=2.0, finish=True)
    assert len(throws) >= 3 and LEFT in [side for _, side in throws]  # the one on the left too
    assert r.stop == "kill limit"
    h2, throws2 = _fighter(monkeypatch, looks, on_look=kill_at)
    h2.looter.sweep_after = lambda deaths: None
    h2.clear(max_kills=1, patience=2.0)  # without finish: stops at the kill
    assert len(throws2) < len(throws)


def test_with_far_on_she_shoots_monsters_on_any_platform_in_reach(monkeypatch):
    far_pig = target(700, y=560, platform=7)  # another platform, 60 below and 200 off: in reach
    h, throws = _fighter(monkeypatch, [[far_pig]] * 3)
    h.clear(ground={3}, patience=0.5, far=True)
    assert throws and throws[0][1] == RIGHT
    h, throws = _fighter(monkeypatch, [[far_pig]] * 3)
    h.clear(ground={3}, patience=0.5)  # without far: not one of hers
    assert not throws


def test_a_kill_is_matched_to_the_first_monster_in_the_star_path(monkeypatch):
    # A snail (not hunted) stands between her and the pig she throws at: the star hits the snail,
    # and the EXP that comes in is the snail's, not the pig's.
    def kill_at(h, t):
        if abs(t - 0.6) < 1e-6:
            h.report.exp_kills += 1
    snail, pig = target(650, kind="snail"), target(780)
    looks = [[snail, pig], [pig], [pig]] + [[]] * 6
    h, throws = _fighter(monkeypatch, looks, on_look=kill_at)
    h.looter.sweep_after = lambda deaths: None
    h.clear(kinds={"pig"}, max_kills=1, patience=0.5)
    assert h.report.kills == {"snail": 1}


def test_hidden_monsters_are_told_by_the_one_time_read_per_look(monkeypatch):
    # A clock that moves on with every read: a monster seen in this look must not count as hidden
    # (the fight reads the time once per look; a second read would make every monster a "hidden" ghost).
    h, throws = _fighter(monkeypatch, [[target(700)]] * 3 + [[]] * 10)
    ticks = [0.0]

    def monotonic():
        ticks[0] += 0.001
        return ticks[0]
    looks = iter([[target(700)]] * 3)
    h.look = lambda *a, **k: SimpleNamespace(where=SimpleNamespace(x=500, y=500), platform=PLAT, hp=1.0,
                                             monsters=next(looks, []))
    monkeypatch.setattr(time, "monotonic", monotonic)
    h.clear(patience=0.01, seconds=0.5)
    assert throws and all(side == RIGHT for _, side in throws)
