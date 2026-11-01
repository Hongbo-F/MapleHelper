"""The loot sweep (behavior.sweep): its exact solve against brute force, its budget, its price of her time, the survival
of drops, landings waited for, guards shot first, a stand's check against monsters by their heading and speed, and an
itinerary kept against small gains. Pure: route times are stand-ins (a straight line at her walk, platforms apart)."""

import itertools
import random
import time
from types import SimpleNamespace

import pytest

from gamelab.behavior import sweep
from gamelab.behavior.sweep import Guard, Spot, Sweeper

WALK = 149.0  # her walk, units a second (motion.physics.WALK_SPEED)


def line(a: int, ax: float, b: int, bx: float) -> float:
    """Route times on a line of platforms: a walk, and 1.5 s more for each platform between."""
    return abs(bx - ax) / WALK + 1.5 * abs(b - a)


def _spot(key, x, worth=30.0, platform=1, stand=0.3, lands=0.0, still=None, guards=()):
    return Spot(key, platform, x, worth, stand, lands, still or (lambda dt: 1.0), tuple(guards))


def _brute(spots, here, dest, rate, arrive, meso_exp, budget):
    """The best itinerary's value by trying every order of every subset (the objective of sweep.plan)."""
    direct = 0.0 if dest is None else arrive(*here, *dest)
    best = 0.0
    for k in range(1, len(spots) + 1):
        for order in itertools.permutations(spots, k):
            at, clock, gain = here, 0.0, 0.0
            for sp in order:
                start = max(clock + arrive(*at, sp.platform, sp.x), sp.lands) + sp.clear_s
                gain += sp.worth * sp.still(start) * meso_exp + sum(g.exp for g in sp.guards)
                at, clock = (sp.platform, sp.x), start + sp.stand
            extra = clock + (0.0 if dest is None else arrive(*at, *dest)) - direct
            if extra <= budget + 1e-9:
                best = max(best, gain - rate * max(extra, 0.0))
    return best


def test_the_exact_solve_is_brute_forces_on_random_instances(monkeypatch):
    monkeypatch.setattr(sweep, "CAND_S", 1e9)  # (every spot a candidate, as brute force has them)
    rng = random.Random(4)
    for trial in range(60):
        n = rng.randint(1, 6)
        spots = [_spot(i, rng.uniform(-400, 400), rng.uniform(5, 120), rng.randint(1, 3), rng.uniform(0.2, 0.8),
                       rng.choice([0.0, 0.0, rng.uniform(0, 2)])) for i in range(n)]
        here, dest = (rng.randint(1, 3), rng.uniform(-400, 400)), rng.choice([None, (rng.randint(1, 3), 300.0)])
        rate = rng.uniform(1, 12)
        it = sweep.plan(spots, here, dest, rate, line, 0.5, budget=1e9)
        assert it.value == pytest.approx(_brute(spots, here, dest, rate, line, 0.5, 1e9), abs=1e-6), trial


def test_with_survival_the_solve_is_within_a_hair_of_brute_force(monkeypatch):
    """Each set's quickest order is weighed by its drops' survival at its arrivals: exact in travel, and within 2% of
    the best order where the survival falls (the live curve's steepest: 0.6% a second, here each spot's its own)."""
    monkeypatch.setattr(sweep, "CAND_S", 1e9)
    rng = random.Random(9)
    for trial in range(40):
        n = rng.randint(2, 6)
        spots = [_spot(i, rng.uniform(-400, 400), rng.uniform(5, 120), rng.randint(1, 3), 0.4,
                       still=(lambda dt, a=rng.uniform(0, 0.006): max(0.0, 1.0 - a * dt))) for i in range(n)]
        here, dest = (2, 0.0), (rng.randint(1, 3), 250.0)
        it = sweep.plan(spots, here, dest, 4.0, line, 0.5, budget=1e9)
        best = _brute(spots, here, dest, 4.0, line, 0.5, 1e9)
        assert best * 0.98 <= it.value <= best + 1e-9, trial


def test_the_budget_is_kept_two_seconds_at_least_or_half_the_way():
    assert sweep.budget_for(1.0) == sweep.BUDGET_MIN and sweep.budget_for(10.0) == pytest.approx(5.0)
    # From x 0 to 1490 (10 s straight: a budget of 5 s), a pile 300 units back (4.3 s more) and one on the platform
    # above (3.3 s more): one of them, not both (7.6 s)
    it = sweep.plan([_spot("near", -300.0, worth=1000.0), _spot("up", 0.0, worth=1000.0, platform=2)], (1, 0.0),
                    (1, 1490.0), 1.0, line, 0.5)
    assert it.keys() == ("up",) and it.out == {"near": "cost"} and it.extra <= it.budget == pytest.approx(5.0)
    # ... and one 400 back (5.7 s more) isn't a candidate at all (CAND_S)
    it = sweep.plan([_spot("back", -400.0, worth=1000.0)], (1, 0.0), (1, 1490.0), 1.0, line, 0.5)
    assert not it.stops and it.out["back"] == f"cost {800 / WALK + 0.3:.1f} s"


def test_a_pile_worth_less_a_second_than_her_time_is_left():
    """Break-even: worth x MESO_EXP against rate x extra. A pile 75 units off her way (1 s there and back, 0.3 s on it):
    30 mesos at 0.5 is 15 EXP, which pays at 10 EXP a second (13 EXP), not at 12 (15.6)."""
    spot = _spot("p", 75.0, worth=30.0)
    assert sweep.plan([spot], (1, 0.0), (1, 0.0), 10.0, line, 0.5).keys() == ("p",)
    it = sweep.plan([spot], (1, 0.0), (1, 0.0), 12.0, line, 0.5)
    assert not it.stops and it.out["p"] == "cost"
    assert sweep.plan([spot], (1, 0.0), (1, 0.0), 12.0, line, 1.0).keys() == ("p",)  # (worth twice as much at 1.0)


def test_she_goes_back_for_a_fresh_pile_behind_her_not_an_old_one():
    """Going on to x 600, a pile 120 behind her: a fresh one of hers (still 0.99 by her getting there) pays for its 1.9
    s back and forth, the same worth a fifth as likely to lie there by then (another's 140 s old: ledger.SURVIVAL)
    doesn't."""
    fresh = _spot("fresh", -120.0, worth=40.0, still=lambda dt: 0.99)
    old = _spot("old", -120.0, worth=40.0, still=lambda dt: 0.2)
    assert sweep.plan([fresh], (1, 0.0), (1, 600.0), 6.0, line, 0.5).keys() == ("fresh",)
    assert sweep.plan([old], (1, 0.0), (1, 600.0), 6.0, line, 0.5).keys() == ()


def test_drops_not_landed_yet_are_waited_for_only_where_they_pay():
    """By her at x 20, her kill's drops land 1.5 s from now: worth 40 (20 EXP) they're waited for at 5 EXP a second
    (1.8 s: 9), not at 12 (21.6); and the wait counts in the budget."""
    landing = _spot("k", 20.0, worth=40.0, lands=1.5)
    it = sweep.plan([landing], (1, 0.0), (1, 0.0), 5.0, line, 0.5)
    assert it.keys() == ("k",) and it.stops[0].arrive == pytest.approx(1.5)
    assert sweep.plan([landing], (1, 0.0), (1, 0.0), 12.0, line, 0.5).keys() == ()
    late = _spot("late", 20.0, worth=400.0, lands=3.0)
    assert sweep.plan([late], (1, 0.0), (1, 0.0), 1.0, line, 0.5).out["late"] == "budget"


def test_a_guard_is_shot_first_its_exp_counted_and_its_kill_time_charged():
    """A pile a Horny Mushroom would meet her on: its 2 s kill comes before the stand, and its 41 EXP with the pile's."""
    horny = Guard(SimpleNamespace(kind="horny-mushroom"), 2.0, 41.0)
    guarded = _spot("g", 150.0, worth=20.0, guards=[horny])
    it = sweep.plan([guarded], (1, 0.0), (1, 0.0), 6.0, line, 0.5, budget=10.0)
    st = it.first
    assert st.spot.key == "g" and st.arrive == pytest.approx(150 / WALK + 2.0)
    assert it.gain == pytest.approx(20 * 0.5 + 41.0)
    assert it.extra == pytest.approx(300 / WALK + 2.0 + 0.3)


def test_the_order_is_the_quickest_two_piles_on_two_platforms():
    """From platform 1 at x 0 on to platform 3 at x 900: the piles on 1 (x 200) and 2 (x 100) taken in that order (1.9
    s more; the other way round 3.6)."""
    a, b = _spot("on 1", 200.0, platform=1), _spot("on 2", 100.0, platform=2)
    it = sweep.plan([b, a], (1, 0.0), (3, 900.0), 2.0, line, 0.5)
    assert it.keys() == ("on 1", "on 2")


def test_no_way_and_too_dear_on_its_own_are_left_out_and_said():
    def arrive(a, ax, b, bx):
        return None if b == 9 or a == 9 else line(a, ax, b, bx)
    it = sweep.plan([_spot("cut off", 0.0, platform=9), _spot("far", 2000.0)], (1, 0.0), (1, 0.0), 1.0, arrive, 0.5)
    assert it.out == {"cut off": "no way", "far": f"cost {2 * 2000 / WALK + 0.3:.1f} s"}


def test_an_itinerary_is_kept_unless_another_is_clearly_better():
    """Two piles each worth the other's detour (patrol's warning case, loop_tree_grind_jump_key): solved again at every
    look, she keeps the one she's going for; one clearly better (KEEP_SHARE and KEEP_S more) is taken."""
    sw = Sweeper()
    left, right = _spot("left", -50.0, worth=40.0), _spot("right", 50.0, worth=40.0)
    first = sw.solve([left, right], (1, 0.0), (1, 0.0), 2.0, line, 0.5).keys()
    assert len(first) == 2
    for x in (5.0, -5.0, 10.0, -10.0):  # where she is, a step either way: the same order on
        assert sw.solve([left, right], (1, x), (1, 0.0), 2.0, line, 0.5).keys() == first
    # One only, a little more worth on the other side: the one she's on kept
    sw2 = Sweeper()
    sw2.solve([_spot("a", -100.0, worth=40.0)], (1, 0.0), (1, 0.0), 8.0, line, 0.5)
    kept = sw2.solve([_spot("a", -100.0, worth=40.0), _spot("b", 100.0, worth=44.0)], (1, 0.0), (1, 0.0), 8.0, line,
                     0.5)
    assert "a" in kept.keys()
    better = sw2.solve([_spot("a", -100.0, worth=40.0), _spot("b", 100.0, worth=400.0)], (1, 0.0), (1, 0.0), 8.0, line,
                       0.5)
    assert better.keys()[0] == "b"


def test_a_stand_meets_a_monster_walking_in_not_one_walking_away():
    """Her stand from x 0 to 20 over 0.5-1.5 s: a Horny Mushroom 120 to the right walking left at the game data's 100
    units a second (28 half its width, 22 hers, the margin 10) is on her by then; walking away it never is; standing,
    neither; its heading not known, either way: it is."""
    def reach(m):
        return m.vx, 100.0, 28.0, -500.0, 500.0
    toward, away, still, unknown = (SimpleNamespace(x=120.0, vx=v) for v in (-100.0, 100.0, 0.0, None))
    assert sweep.meets([toward], 0.0, 20.0, 0.5, 1.5, reach, 22.0) is toward
    assert sweep.meets([away], 0.0, 20.0, 0.5, 1.5, reach, 22.0) is None
    assert sweep.meets([still], 0.0, 20.0, 0.5, 1.5, reach, 22.0) is None
    assert sweep.meets([unknown], 0.0, 20.0, 0.5, 1.5, reach, 22.0) is unknown
    # one walking away from its platform's end turns back there: from 150 to 200 by 0.5 s, back to 50 by 2.0
    turning = SimpleNamespace(x=150.0, vx=100.0)

    def short(m):
        return m.vx, 100.0, 28.0, -500.0, 200.0
    assert sweep.meets([turning], 0.0, 20.0, 0.5, 1.5, short, 22.0) is None
    assert sweep.meets([turning], 0.0, 20.0, 0.5, 2.0, short, 22.0) is turning


def test_bounce_turns_back_at_the_ends():
    assert sweep.bounce(0.0, 100.0, 1.0, -500.0, 500.0) == pytest.approx(100.0)
    assert sweep.bounce(450.0, 100.0, 1.0, -500.0, 500.0) == pytest.approx(450.0)
    assert sweep.bounce(450.0, 100.0, 11.0, -500.0, 500.0) == pytest.approx(-450.0)
    assert sweep.bounce(0.0, 0.0, 9.0, -1.0, 1.0) == 0.0


def test_eight_candidates_solve_within_a_few_ms():
    rng = random.Random(2)
    spots = [_spot(i, rng.uniform(-600, 600), rng.uniform(10, 80), rng.randint(1, 4)) for i in range(8)]
    times = []
    for _ in range(9):
        t0 = time.perf_counter()
        sweep.plan(spots, (2, 0.0), (3, 300.0), 4.0, line, 0.5, budget=1e9)
        times.append(time.perf_counter() - t0)
    assert sorted(times)[4] < 0.025  # (median; 8 candidates; the design's 5 ms on a quiet core, CI's leeway)


def test_the_record_says_the_order_its_numbers_and_why_the_rest_were_left():
    it = sweep.plan([_spot("p", 75.0, worth=30.0), _spot("q", 3000.0)], (1, 0.0), (1, 0.0), 2.0, line, 0.5)
    rec = it.record()
    assert rec["order"] == [[1, 75, pytest.approx(75 / WALK, abs=0.01), 30.0, 0]] and rec["out"] == {"q": "cost 40.6 s"}
    assert set(rec) == {"order", "extra", "value", "direct", "budget", "rate", "out"}
