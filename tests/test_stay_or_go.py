"""Stay or go (behavior.targeting.stay_or_go, GAMELAB_STAY_OR_GO; README: "Where she hunts"): her long stays low of 2026-11-05 replayed on the looks she had
(tests/fixtures/hunt_stays_2026-11-05.json: the real Chooser, Targeting._worth and Grind._goes), the journeys that passed monsters put in the world
(hunt_journeys_2026-11-05.json: sim.hunt_scenes, the full stack), and the pieces: what's on the screen, the bar, the landing's question."""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gamelab.attack import danger as danger_mod
from gamelab.attack import stance
from gamelab.attack import target as tg
from gamelab.behavior import patrol, targeting
from gamelab.behavior.grind import DELIVERS, GO_GAIN, TO_TARGET, Grind
from gamelab.control import navigator
from gamelab.control.navigator import ASKED, COMING, Navigator
from gamelab.localization.monsters import Monster, bodies, speeds
from gamelab.motion.route import Planner
from gamelab.sim import hunt_scenes, scenarios

STAYS = json.loads((Path(__file__).parent / "fixtures" / "hunt_stays_2026-11-05.json").read_text(encoding="utf-8"))["cases"]
JOURNEYS = json.loads(hunt_scenes.JOURNEYS.read_text(encoding="utf-8"))["cases"]
TREE = scenarios.TREE
pytestmark = pytest.mark.skipif(not scenarios.available(TREE), reason="needs Tree Dungeon's game data (gamedata/data)")
EXP = {"horny-mushroom": 41.0, "green-mushroom": 26.0, "slime": 12.0}
KILL_S = {"horny-mushroom": 2.33, "green-mushroom": 1.26, "slime": 1.16}  # (a Lucky Seven's, behavior.attacks' prior)


@pytest.fixture(scope="module")
def tree():
    return (m := scenarios.real_map(TREE)), Planner(m), danger_mod.of_map(m, danger_mod.Her(1244.0, 35.0, 117.0))


def _targeting(weighs=True, landings=True):
    with mock.patch.dict(os.environ, GAMELAB_STAY_OR_GO="1" if weighs and landings else "go" if weighs else "land" if landings else "0"):
        return targeting.Targeting(SimpleNamespace(h=SimpleNamespace(attacks=SimpleNamespace(picks=False)), model=scenarios.real_map(TREE), kinds=None, danger=None))


def _grind(tp, rate=None, nearest=lambda here, x: 7):
    """A grind without __init__, her round at `rate` EXP a minute, its look at the round a no-op."""
    g = Grind.__new__(Grind)
    g.tp, g.patrol = tp, SimpleNamespace(plan=None if rate is None else SimpleNamespace(rate=rate / 60.0), nearest=nearest)
    g._look_at_round = lambda now: None
    return g


# --- the long stays --------------------------------------------------------------------------------------------------

def _terms(*cells):
    return tuple(tg.Terms(tg.Foe(None, exp, kill, p, None, False, 0.0, 0.0, False), tg.Way(way, "now"), 1.0) for exp, kill, p, way in cells)


def _look(model, planner, danger, look):
    t, pid, x, y, mp, mobs = look
    tops = bodies(model)
    halves = {k: b.half for k, b in tops.items()}
    here = model.platform(pid)
    ms = [Monster(SimpleNamespace(kind=k, fading=None), mx, my, p, bar, 0, "", i + 1) for i, (k, mx, my, p, bar) in enumerate(mobs)]
    lo, hi = stance._stand_limits(here, crowded=True)
    her = tg.Her(here, x, y, 0.05, 0.0, None, True, ms, None, halves, danger, stance.Pace(0.05, lambda m: 0.0, speeds(model), halves, None, 0.0), lo, hi, jump_hits=None,
                 can_jump=stance.open_above(model, here, x, x), routes=lambda src, sx, dst, a, b, off=frozenset(): planner.arrive_within(src, round(sx / 8.0) * 8.0, dst, a, b, off),
                 stands=tg.in_height(model.places))
    foes = [tg.Foe(m, EXP[m.kind], KILL_S[m.kind] * (1.0 if m.bar is None else max(0.2, m.bar)), 1.0, None, False, 0.0, 0.0, False) for m in ms]
    return t, here, x, her, foes


def _first_go(case, weighs=True):
    """The first moment of `case`'s stay `Grind._goes` sends her from, with what it read: (s, lately, bar), None: none."""
    tp = _targeting(weighs, False)
    g = _grind(tp)
    rates, clock = case["rates"], [0.0]
    her = SimpleNamespace(platform=SimpleNamespace(id=31), where=SimpleNamespace(x=0.0))
    with mock.patch.object(targeting.time, "monotonic", lambda: clock[0]):
        tp._worth_at = case["looks"][0][0]
        for clock[0], trip, top in case["looks"]:
            tp._worth(tg.Choice(top=_terms(*((exp, kill, 1.0, way) for exp, kill, way in top))))
            rate = next((r for at, r, _ in reversed(rates) if at <= clock[0]), None)  # (none yet: the tour's first ask to come)
            g.patrol.plan = None if rate is None else SimpleNamespace(rate=rate / 60.0)
            if clock[0] >= 0.0 and trip and g._goes(her) is not None:
                return clock[0], tp.lately * 60.0, g._round_bar() * 60.0
    return None


@pytest.mark.parametrize("case", STAYS, ids=[c["name"] for c in STAYS])
def test_her_long_stay_low_she_goes_when_the_first_pile_has_paid_the_second_of_them_at_once(case):
    """07:53, 08:10, 17:22 and 17:29 she goes after the first pile has paid, as the rate falls; 08:03 at once, 8 monsters on the screen, on the rate of a
    round that promises 680: that one may be wrong (README: what a live run can tell). The looks 0.5 s apart here, as near as that to the replay at every look."""
    got = _first_go(case)
    assert got is not None and got[0] == pytest.approx(case["every_look_goes_at"], abs=2.0), got
    assert got[2] > GO_GAIN * got[1]


@pytest.mark.parametrize("case", STAYS, ids=[c["name"] for c in STAYS])
def test_unweighed_she_stays(case):
    assert _first_go(case, weighs=False) is None


# --- the journeys, at their landings -------------------------------------------------------------------------------------

def _landing_scene(model, look):
    here = model.platform(look["platform"])
    return SimpleNamespace(t=look["s"], platform=here, where=SimpleNamespace(x=look["x"], y=look["y"], platform=here), what_if=False, landing=True, players=[], monsters=[
        Monster(SimpleNamespace(kind=k, fading=None), mx, my, p, bar, 0, "", i + 1) for i, (k, mx, my, p, bar) in enumerate(look["monsters"])])


def _answers(tree, case):
    """Each landing of the journey asked what the real question asks of its look: the look's choice through the real Targeting.peek (her there, its monsters, her
    kill times), then a target trip's `changed`, a loot chain's `_chain_stops`, a stop trip's `on_screen` at her round's bar. [(platform, what it says, got, bar, in reach)]."""
    model, planner, danger = tree
    tp = _targeting()
    g = _grind(tp, case["rate"])
    tp.g, tp.screen, tp.chain = g, True, True
    g._alive, g._rest_of_screen = (lambda s, ms, now: ms), (lambda s, now: [])
    g._may_stop_her, g._close_quarters, g._worthless, g._owed, g._valuable_pile = (lambda s: []), (lambda s, w: False), (lambda p: False), (lambda s, ms: None), (lambda s: None)

    def her_at(s, pace, visible, hidden):
        return _look(model, planner, danger, [s.t, s.platform.id, s.where.x, s.where.y, None, [[m.kind, m.x, m.y, m.platform, m.bar] for m in s.monsters]])[3]

    def foes_of(s, alive, visible, pace, more):
        return [tg.Foe(m, EXP[m.kind], KILL_S[m.kind] * (1.0 if m.bar is None else max(0.2, m.bar)), 1.0, None, False, 0.0, 0.0, False) for m in s.monsters]
    tp.her_at, tp.foes_of = her_at, foes_of
    tp._close_first = staticmethod(lambda her, foes: foes)
    out, target, chain = [], case["target"], SimpleNamespace(chain=SimpleNamespace(planned_on=None), pile=None)
    with mock.patch.object(targeting.fight, "pace_of", lambda h, s: stance.NO_PACE):
        for look in case["landings"]:
            s = _landing_scene(model, look)
            if target is not None:
                near = [m for m in s.monsters if m.kind == target["kind"] and m.platform == target["on"]]
                tp.chooser.primary = near[0] if near else Monster(SimpleNamespace(kind=target["kind"], fading=None), 1e4, 0.0, target["on"], track=999)
            says = (tp.changed(s) if case["kind"] == "target" else Grind._chain_stops(g, s, chain) if case["kind"] == "loot" else tp.on_screen(s, g._round_bar()))
            c = tp.peek(s)
            out.append((look["platform"], bool(says), round(tp.got * 60.0), round(g._round_bar() * 60.0), None if c.primary is None else c.primary.way.how, len(s.monsters)))
    return out


# The first landing of each that calls for her, the look's question at its first look there; None: none does (18:44's has a Green a trip away and no more). 18:26's is the touched
# journey: asked once her invulnerability is over (Navigator._landing_calls), its first landing past that here.
STOPS = {"074824": 11, "111310": 17, "173514": 14, "180940": 14, "181056": 14, "183210": 14, "184415": None, "181033": 13, "181600": 12, "183926": 15, "182627": 15}


@pytest.mark.parametrize("case", JOURNEYS, ids=[c["name"] for c in JOURNEYS])
def test_the_journey_stops_at_the_first_landing_that_calls_for_her(tree, case):
    """The journeys that passed monsters live, each landing's question on its look. 18:16 is a stop's trip, past 15 monsters for a stop 12.6 s off: there the bar decides,
    and what's in reach at P12 clears it (458 over 365 EXP a minute)."""
    rows = _answers(tree, case)
    assert next((r[0] for r in rows if r[1]), None) == STOPS[case["name"]], rows


# The journey, where the question stopped her in the world, on seed 1 (of seeds 1-3, D:/MapleHelper-scratch/hunts/built/js3.txt: 183210 on all, 173514 on 1 and 2, on 3 she stops on P14 unasked).
IN_THE_WORLD = {"183210": 14, "173514": 14, "182627": 17}


@pytest.mark.parametrize("name", IN_THE_WORLD)
def test_in_the_world_she_stops_at_a_landing_with_monsters_she_ran_past_unasked(monkeypatch, name):
    case, there = hunt_scenes.cases()[name], IN_THE_WORLD[name]
    monkeypatch.setenv("GAMELAB_STAY_OR_GO", "0")
    off = hunt_scenes.run_journey(case)
    monkeypatch.setenv("GAMELAB_STAY_OR_GO", "1")
    on = hunt_scenes.run_journey(case)
    assert off["asked"] == 0 and off["path"][1] != there and off["first_attack"] != there, off["path"]
    assert on["asked"] >= 1 and on["path"][1] == there and on["first_attack"] == there, on["path"]
    assert on["kills"] > off["kills"]


# --- the pieces --------------------------------------------------------------------------------------------------------

def test_what_the_screen_has_is_her_time_on_each_of_its_best_and_lately_lets_it_down_over_a_minute():
    tp = _targeting()
    clock = [100.0]
    with mock.patch.object(targeting.time, "monotonic", lambda: clock[0]):
        tp._worth_at = 100.0
        c = tg.Choice(top=_terms((41.0, 2.0, 1.0, 1.0), (26.0, 1.0, 0.5, 9.0)))
        tp._worth(c)
        want = (41.0 + 13.0) / (1.0 + 2.0 + targeting.KILL_GAP_S + 1.0 + targeting.KILL_GAP_S)  # (the way to the first alone)
        assert tp.got == pytest.approx(want) and tp.lately == pytest.approx(want)
        clock[0] += targeting.SCREEN_S / 2.0
        tp._worth(tg.Choice())
        assert tp.got == 0.0 and tp.lately == pytest.approx(want / 2.0)
        clock[0] += targeting.SCREEN_S * 2.0
        tp._worth(tg.Choice())
        assert tp.lately == 0.0
        tp._worth(c)
        assert tp.lately == pytest.approx(want)  # (up at once)


def test_she_goes_only_with_a_round_clearly_over_what_the_screen_has_had():
    tp = _targeting()
    s = SimpleNamespace(platform=SimpleNamespace(id=31), where=SimpleNamespace(x=0.0))
    tp.lately = 400.0 / 60.0
    assert _grind(tp, 400.0 * GO_GAIN / DELIVERS - 1.0)._goes(s) is None
    assert _grind(tp, 400.0 * GO_GAIN / DELIVERS + 1.0)._goes(s) == 7
    assert _grind(tp, None)._goes(s) is None
    assert _grind(tp, 9999.0, nearest=lambda here, x: None)._goes(s) is None
    off = _targeting(False, False)
    off.lately = 0.0
    assert _grind(off, 9999.0)._goes(s) is None


def test_a_trip_to_a_stop_ends_only_for_what_beats_her_round_and_a_trip_to_her_target_or_unweighed_as_it_was():
    tp = _targeting()
    tp.screen = True
    g = _grind(tp, 500.0)
    stop, target = SimpleNamespace(kind="spot", why=""), SimpleNamespace(kind="spot", why=TO_TARGET)
    bar = g._round_bar()
    assert bar == pytest.approx(DELIVERS * 500.0 / 60.0)
    tp.got = bar - 0.01
    assert not g._free_trip(stop) and g._free_trip(target)
    tp.got = bar + 0.01
    assert g._free_trip(stop)
    off = _targeting(False, False)
    off.screen = True
    off.got = 0.0
    g = _grind(off, 500.0)
    assert g._round_bar() == -1.0 and g._free_trip(stop)
    assert _grind(tp, None)._round_bar() == -1.0
    look = SimpleNamespace()
    tp.peek = lambda s: tg.Choice(primary=object())
    tp.got = 1.0
    assert not tp.on_screen(look, 2.0) and tp.on_screen(look, 0.5) and tp.on_screen(look)


def test_the_switch_and_her_casts(monkeypatch):
    for value, want in (("0", (False, False)), ("land", (True, False)), ("go", (False, True)), ("1", (True, True))):
        monkeypatch.setenv("GAMELAB_STAY_OR_GO", value)
        assert targeting.stay_or_go() == want
    monkeypatch.setenv("GAMELAB_STAY_OR_GO", "1")
    g = SimpleNamespace(h=SimpleNamespace(attacks=SimpleNamespace(picks=True)), model=scenarios.real_map(TREE), kinds=None, danger=None)
    tp = targeting.Targeting(g)
    assert (tp.landings, tp.weighs) == (False, False)  # (the magician's as they were)
    monkeypatch.setattr(targeting, "CASTS_TOO", True)
    tp = targeting.Targeting(g)
    assert (tp.landings, tp.weighs) == (True, True)


def test_the_round_s_nearest_stop_and_none_on_the_round():
    m = scenarios.real_map(TREE)
    planner = Planner(m)
    exp = {"horny-mushroom": 41.0, "green-mushroom": 26.0, "slime": 12.0}
    p = patrol.Patrol(m, set(exp), exp, travel=planner.arrive)
    top, upper, low = p.stop_on(7), p.stop_on(14), p.stops[next(j for j in range(len(p.stops)) if p.stops[j].platform >= 31)]
    p.plan = SimpleNamespace(stops=(top, upper))
    assert p.nearest(low.platform, low.x) == upper  # (the nearer)
    assert p.nearest(14, p.stops[upper].x) is None  # (on one of its stops: one stop to the next isn't going)
    covered = next(p.points[q].platform for q, c in p.stops[upper].cover if c > 0 and p.points[q].platform != 14)
    assert p.nearest(covered, p.stops[upper].x) is None  # (... nor on a platform one of them gets at)
    p.plan = SimpleNamespace(stops=(top,))
    p.travel = lambda a, x, b, bx: None
    assert p.nearest(low.platform, low.x) is None  # (no way there)


# --- the landing's question ---------------------------------------------------------------------------------------------

def _landed(until, platform=14, mode="ground", post_hit=False):
    """A navigator without __init__, a chain 10 -> 14 -> 19 running, her glance on `platform`."""
    nav = Navigator.__new__(Navigator)
    nav.landing_asks, nav._asked_on = True, None
    nav.senses = SimpleNamespace(glanced=SimpleNamespace(where=(0.0, 0.0, platform), mode=mode), band=None)
    nav.model = None
    st = SimpleNamespace(until=until, post_hit_until=time.monotonic() + (9.0 if post_hit else -9.0))
    leg = lambda a, b: SimpleNamespace(start=SimpleNamespace(platform=SimpleNamespace(id=a)), end=SimpleNamespace(platform=SimpleNamespace(id=b)))
    return nav, st, SimpleNamespace(legs=[leg(10, 14), leg(14, 19)], more=False)


@pytest.fixture
def looks(monkeypatch):
    monkeypatch.setattr(navigator, "to_scene", lambda state, model, band, mobs, wide: SimpleNamespace(where=SimpleNamespace(platform=object()), searched=True))


def test_a_landing_inside_a_chain_gets_the_question_once(looks):
    asked = []
    nav, st, running = _landed(lambda s: asked.append(s) or True)
    assert nav._landing_calls(st, running, None) and len(asked) == 1
    assert not nav._landing_calls(st, running, None) and len(asked) == 1  # (the same landing)
    nav.senses.glanced = SimpleNamespace(where=(0.0, 0.0, None), mode="air")  # (off it, and on it again: a landing)
    assert not nav._landing_calls(st, running, None)
    nav.senses.glanced = SimpleNamespace(where=(0.0, 0.0, 14), mode="ground")
    assert nav._landing_calls(st, running, None) and len(asked) == 2


def test_not_where_it_set_off_nor_its_end_nor_while_she_s_invulnerable_nor_in_the_air_nor_with_no_question(looks):
    asked = []
    until = lambda s: asked.append(s) or True
    for kw in (dict(platform=10), dict(platform=19), dict(post_hit=True), dict(mode="air")):
        nav, st, running = _landed(until, **kw)
        assert not nav._landing_calls(st, running, None)
    nav, st, running = _landed(None)
    assert not nav._landing_calls(st, running, None)
    assert not asked
    nav, st, running = _landed(until, platform=19)
    assert nav._landing_calls(st, running, SimpleNamespace())  # (its end, a chain queued after it: on past it unasked otherwise)
    nav, st, running = _landed(until, platform=19)
    running.more = True
    assert nav._landing_calls(st, running, None)


def test_a_cut_for_a_landing_is_no_miss():
    assert navigator.ASKED.startswith(ASKED) and not ASKED.startswith(COMING)


def test_a_pile_no_look_has_seen_off_its_platform_s_end_gets_no_chain():
    """(S6's seed 3 with the design's first form: "loot: 1 stop, x 329 on 32: -0.1 s" every 0.2 s for 150 s, P32 ending at 179: `quiet` takes any pile with monsters on
    the screen now.) Its try spent, and none planned; one by the platform's end, as ever."""
    m = scenarios.real_map(TREE)
    g = Grind.__new__(Grind)
    tried = []
    g.model, g.tp, g._tried = m, SimpleNamespace(choice=None), tried.append
    far = SimpleNamespace(x=329.0, platform=32, status="expected")
    near = SimpleNamespace(x=m.platform(32).right + 20.0, platform=32, status="expected")
    g._pile_of = {1: far, 2: near}

    def stop(key):
        return SimpleNamespace(spot=SimpleNamespace(key=key, guards=(), lands=0.0), arrive=9.0, leave=9.0)
    g._sweeper = SimpleNamespace(current=SimpleNamespace(stops=[stop(1)]), dest=None)
    look = SimpleNamespace(platform=m.platform(32), where=SimpleNamespace(x=100.0), t=0.0)
    assert g._loot_plan(look) is None and tried == [far]
    g._sweeper.current.stops = [stop(2)]
    chain = g._loot_plan(look)
    assert chain is not None and chain.stretches[0].pile is near and tried == [far]
