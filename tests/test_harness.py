"""The simulator's realism knobs (all off by default) and the results harness that compares hunting
policies on the same seeds (gamelab.sim.harness). Synthetic maps need no game data; the real-map test is
skipped without it."""

import pytest

from gamelab.behavior.hunter import Hunter
from gamelab.localization import mapdata
from gamelab.sim import fakes, harness, scenarios
from gamelab.sim import world as world_mod
from gamelab.sim.fakes import Clock, FakeEyes, SimTimeout
from gamelab.sim.world import DROP_LIFE, TOUCH_DAMAGE, Noise, Realism, World

from test_sim import KEYS, plat, still, tap, tree


@pytest.fixture(scope="module")
def model():
    return mapdata.parse(1, tree(), "Test Map", {}, {"2": "Snail"}, {"2": 2})


def world(model, pid, x, **kw):
    kw.setdefault("monsters", None)
    return World(model, KEYS, (pid, x), **kw)


def test_a_hit_monster_hides_for_a_while_and_a_look_can_miss_one(model):
    b = plat(model, 360, 300)
    clock = Clock()
    w = still(World(model, KEYS, (b.id, 240.0), monsters=[("Pig", b.id, 380.0)], origin=clock.now,
                    noise=Noise(hit_hidden=(1.0, 3.0))))
    eyes = FakeEyes(w, clock, model)
    h = Hunter(eyes, KEYS, model, log=lambda m: None)
    assert len(h.look().monsters) == 1
    w.monsters[0].hit_at = w.t  # a star just hit it
    assert h.look().monsters == []
    clock.advance_to(clock.now + 3.1)
    assert len(h.look().monsters) == 1  # back after 1-3 s
    w = World(model, KEYS, (b.id, 240.0), monsters=[("Pig", b.id, 380.0)], origin=clock.now,
              noise=Noise(detect_miss=1.0))
    assert Hunter(FakeEyes(w, clock, model), KEYS, model, log=lambda m: None).look().monsters == []


def test_another_player_kills_monsters_and_its_drops_are_not_hers_for_a_while(model):
    c = plat(model, 500, 400)
    w = world(model, c.id, 300.0, monsters=[("Snail", c.id, 310.0)],
              realism=Realism(rival_every=1.0, rival_owns=15.0, rival_takes=20.0))
    w.invulnerable_ms = 100_000
    still(w)
    w.advance(2.0)
    assert [k[3] for k in w.kill_log] == ["rival"] and w.exp == 0
    w.advance(4.0)
    (d,) = w.live_drops()
    assert d.owner == "rival"
    w.x = d.x
    tap(w, 4.0, "z", 0.04)
    w.advance(4.2)
    assert w.picked == {}  # not hers yet
    w.advance(30.0)
    assert d.gone and d.taken_by == "rival"  # the rival picked it up


def test_potions_run_out_and_arrows_can_be_locked_during_an_attack(model):
    c = plat(model, 500, 400)
    w = world(model, c.id, 100.0, hp=0.2, realism=Realism(potions=1))
    tap(w, 0.0, "r", 0.04)
    tap(w, 0.5, "r", 0.04)
    w.advance(1.0)
    assert w.hp == pytest.approx(0.5) and w.potions_used == 1 and any("no potion left" in line for line in w.log)
    for lock, moved in ((False, True), (True, False)):
        w = world(model, c.id, 100.0, realism=Realism(lock_arrows=lock))
        tap(w, 0.0, "f", 0.04)
        tap(w, 0.1, "right", 0.4)  # inside the 730 ms lock
        w.advance(0.7)
        assert (w.x > 110) == moved


def test_the_knobs_are_off_by_default_and_spawn_share_thins_the_monsters():
    assert Noise() == Noise(hit_hidden=(), detect_miss=0.0)
    assert Realism().respawn == world_mod.RESPAWN and Realism().rival_every == 0 and Realism().potions is None
    if not scenarios.available():
        pytest.skip("needs the real map's data")
    model = scenarios.real_map()
    full = World(model, KEYS, (36, 300.0), monsters="all")
    half = World(model, KEYS, (36, 300.0), monsters="all", realism=Realism(spawn_share=0.5))
    assert len(full.monsters) == 48 and 14 <= len(half.monsters) <= 34


def test_sim_timeout_isnt_caught_by_a_loop_that_recovers_from_errors():
    clock = Clock(limit=1.0)
    with pytest.raises(SimTimeout):
        try:
            clock.sleep(2.0)
        except Exception:  # what a loop that carries on after errors would do
            pass
    assert not issubclass(SimTimeout, Exception)


def test_thinking_can_be_charged_to_the_fake_clock(monkeypatch):
    real = iter([10.0, 10.5, 10.6])
    monkeypatch.setitem(fakes._REAL, "perf_counter", lambda: next(real))
    clock = Clock(think=2.0)
    clock.mark()        # a fake call ends at real 10.0 ...
    clock.sleep(0.1)    # ... and the next starts at 10.5: 0.5 s of thinking, charged twice over
    assert clock.elapsed == pytest.approx(1.1)
    assert Clock().think == 0.0


def _result(seed, epm, share=0.5, hp=0.1, died=None):
    return {"seed": seed, "exp_per_min": epm, "drops_share": share, "hp_lost": hp, "died_at": died}


def test_the_bar_needs_more_exp_on_most_seeds_and_nothing_worse():
    base = [_result(k, 50.0) for k in range(1, 11)]
    ok, lines = harness.compare(base, [_result(k, 60.0) for k in range(1, 11)])
    assert ok and lines[-1] == "PASSES the bar"
    assert not harness.compare(base, [_result(k, 60.0 if k > 3 else 49.0) for k in range(1, 11)])[0]  # 7 of 10
    assert not harness.compare(base, [_result(k, 60.0 if k > 1 else 40.0) for k in range(1, 11)])[0]  # -20%
    assert not harness.compare(base, [_result(k, 60.0, hp=0.2) for k in range(1, 11)])[0]
    assert not harness.compare(base, [_result(k, 60.0, died=100.0 if k == 5 else None) for k in range(1, 11)])[0]


def test_outcomes_are_read_from_the_world(model):
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0, monsters=[("Snail", c.id, 400.0)], facing="right"))
    tap(w, 0.0, "f", 0.04)
    w.advance(3.0)
    (d,) = w.live_drops()
    w.x = d.x
    tap(w, 3.0, "z", 0.04)
    w.advance(3.5)
    out = harness.outcomes(w, [{"topic": "event", "kind": "backstop", "text": "LOOT_SECONDS: 180 s"}], 60.0)
    assert out["kills"] == 1 and out["exp"] == 2 and out["exp_per_min"] == 2.0
    assert out["drops"] == 1 and out["drops_share"] == 1.0 and out["backstops"] == {"LOOT_SECONDS": 1}
    assert out["longest_idle"] == pytest.approx(60.0 - 3.0, abs=0.6)  # from her pick-up to the end
    w.drops[0].taken_by, w.drops[0].since = "", 0.0  # a drop left lying its whole life counts as missed
    assert harness.outcomes(w, [], DROP_LIFE + 1)["drops_share"] == 0.0


def test_hp_lost_on_the_trips_to_rest_is_counted_between_their_events(model):
    c = plat(model, 500, 400)
    w = still(world(model, c.id, 100.0))
    w.touch_log = [(5.0, True), (12.0, True), (13.0, False), (30.0, True)]
    t0 = 1000.0
    records = [{"topic": "session", "t": t0}, {"topic": "event", "kind": "rest_trip", "t": t0 + 10.0},
               {"topic": "event", "kind": "rest", "t": t0 + 20.0}, {"topic": "event", "kind": "rest_trip", "t": t0 + 25.0}]
    out = harness.outcomes(w, records, 60.0)
    assert out["rest_trips"] == 2  # the last one still going at the end
    assert out["rest_hp_lost"] == pytest.approx(2 * TOUCH_DAMAGE)  # at 12 s and 30 s; 13 s was evaded
    base = [dict(_result(k, 50.0), rest_hp_lost=0.05, rest_trips=2) for k in range(1, 11)]
    assert harness.compare_rest(base, [dict(r, rest_hp_lost=0.02) for r in base])[0]
    assert not harness.compare_rest(base, [dict(r, rest_hp_lost=0.08) for r in base])[0]
    # Runs saved before the metric existed, or with no seed in common, never pass.
    assert not harness.compare_rest([_result(k, 50.0) for k in range(1, 11)], base)[0]
    assert not harness.compare([], [])[0] and not harness.compare(base, [_result(k, 60.0) for k in range(11, 21)])[0]
    assert not harness.compare_rest(base, [dict(r, seed=r["seed"] + 10) for r in base])[0]
    # No trip to rest on a side: nothing measured, nothing passed.
    assert not harness.compare_rest(base, [dict(r, rest_hp_lost=0.0, rest_trips=0) for r in base])[0]
    # The bar is over the seeds both have: a 20-seed base against 10 seeds of a candidate compares those 10.
    harder = base + [dict(_result(k, 50.0, share=0.1, hp=1.0)) for k in range(11, 21)]
    assert not harness.compare(harder, [_result(k, 60.0, share=0.4, hp=0.2) for k in range(1, 11)])[0]
    # Fewer than 10 seeds in common never pass; a death counts on any seed, compared or not.
    better = [dict(r, exp_per_min=60.0, rest_hp_lost=0.01) for r in base]
    assert not harness.compare(base[:9], better[:9])[0] and not harness.compare_rest(base[:9], better[:9])[0]
    assert harness.compare(base, better)[0] and harness.compare_rest(base, better)[0]
    died = better + [dict(better[0], seed=15, died_at=600.0)]
    assert not harness.compare(base, died)[0] and not harness.compare_rest(base, died)[0]


def test_a_trip_to_rest_is_measured_both_ways():
    if not scenarios.available():
        pytest.skip("needs the real map's data")
    ok, lines = harness.rest_trials(1)
    assert lines[0].startswith("transit: HP lost") and lines[1].startswith("avoid: HP lost") and "rest bar" in lines[2]
    assert harness.rest_trip(True, 0)["kills"] == 0  # keeping clear, she shoots nothing


def test_a_short_harness_run_on_the_real_map():
    if not scenarios.available():
        pytest.skip("needs the real map's data")
    out = harness.run_one("tour", 1, "game", minutes=1.0)
    assert out["policy"] == "tour" and "SimTimeout" in out["result"] and out["exp_per_min"] >= 0
    assert set(harness.COLUMNS) - {"seed"} <= set(out)
