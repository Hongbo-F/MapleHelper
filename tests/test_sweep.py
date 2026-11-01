"""The loot sweep (behavior.sweep): its exact solve against brute force, each pile within what it may add to her way and
an itinerary within the sum of its piles' (their allowances, Spot.allow: no price of her time, no meso's worth in EXP,
from 2026-11-01), the most mesos then the least time, the survival of drops, landings waited for, guards shot first and
their kill times not charged, a stand's check against monsters by their heading and speed, and an itinerary kept
against small gains. Pure: route times are stand-ins (a straight line at her walk, platforms apart)."""

import inspect
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


def _spot(key, x, worth=30.0, platform=1, stand=0.3, lands=0.0, still=None, guards=(), allow=3.0):
    return Spot(key, platform, x, worth, stand, lands, still or (lambda dt: 1.0), tuple(guards), allow)


def _brute(spots, here, dest, arrive, budget=None):
    """The best itinerary's (value, extra) by trying every order of every subset of the spots each within its allowance
    on its own (the objective of sweep.plan: the most mesos to GAIN_TIE, then the least time)."""
    direct = 0.0 if dest is None else arrive(*here, *dest)

    def on(p):
        return 0.0 if dest is None else arrive(*p, *dest)
    alone = [sp for sp in spots if max(arrive(*here, sp.platform, sp.x), sp.lands) + sp.stand
             + on((sp.platform, sp.x)) - direct <= sp.allow + 1e-9]
    best = None
    for k in range(1, len(alone) + 1):
        for order in itertools.permutations(alone, k):
            at, clock, gain = here, 0.0, 0.0
            for sp in order:
                start = max(clock + arrive(*at, sp.platform, sp.x), sp.lands) + sp.clear_s
                gain += sp.worth * sp.still(start)
                at, clock = (sp.platform, sp.x), start + sp.stand
            extra = clock + on(at) - direct
            if extra - sum(sp.clear_s for sp in order) > sum(sp.allow for sp in order) + 1e-9:
                continue
            if budget is not None and extra > budget + 1e-9:
                continue
            key = (round(gain / sweep.GAIN_TIE), -extra)
            if best is None or key > best[0]:
                best = (key, gain, extra)
    return (0.0, 0.0) if best is None else (best[1], best[2])


def test_no_price_of_her_time_nor_meso_exchange_in_the_sweep():
    """The user, 2026-11-01: "I'm not sure about this currency conversion. My idea is just picking up mesos and items as
    we kill mobs": the solve takes no rate and no meso's worth in EXP, its itineraries no time's price."""
    for fn in (sweep.plan, sweep.again, Sweeper.solve):
        assert not {"rate", "meso_exp"} & set(inspect.signature(fn).parameters)
    assert not hasattr(sweep.Itinerary(), "rate") and not hasattr(sweep, "CAND_S")


def test_the_exact_solve_is_brute_forces_on_random_instances():
    rng = random.Random(4)
    for trial in range(60):
        n = rng.randint(1, 6)
        spots = [_spot(i, rng.uniform(-400, 400), rng.uniform(5, 120), rng.randint(1, 3), rng.uniform(0.2, 0.8),
                       rng.choice([0.0, 0.0, rng.uniform(0, 2)]), allow=rng.uniform(0.5, 6.0)) for i in range(n)]
        here, dest = (rng.randint(1, 3), rng.uniform(-400, 400)), rng.choice([None, (rng.randint(1, 3), 300.0)])
        budget = rng.choice([None, None, rng.uniform(1.0, 8.0)])
        it = sweep.plan(spots, here, dest, line, budget)
        value, extra = _brute(spots, here, dest, line, budget)
        assert it.value == pytest.approx(value, abs=1e-6), trial
        assert it.extra == pytest.approx(extra, abs=1e-6), trial


def test_with_survival_the_solve_is_within_a_hair_of_brute_force():
    """Each set's quickest order is weighed by its drops' survival at its arrivals: exact in travel, and within 2% of
    the best order where the survival falls (the live curve's steepest: 0.6% a second, here each spot's its own)."""
    rng = random.Random(9)
    for trial in range(40):
        n = rng.randint(2, 6)
        spots = [_spot(i, rng.uniform(-400, 400), rng.uniform(5, 120), rng.randint(1, 3), 0.4, allow=8.0,
                       still=(lambda dt, a=rng.uniform(0, 0.006): max(0.0, 1.0 - a * dt))) for i in range(n)]
        here, dest = (2, 0.0), (rng.randint(1, 3), 250.0)
        it = sweep.plan(spots, here, dest, line)
        best, _ = _brute(spots, here, dest, line)
        assert best * 0.98 <= it.value <= best + 1e-9, trial


def test_a_pile_is_taken_within_its_allowance_and_left_over_it():
    """A pile 200 units off her way there and back (2.7 s, and its 0.3 s stand): taken with a fresh pile's 3 s, left with
    an old one's 1 s (behavior.hunt.ALLOW_FRESH, ALLOW_OLD: no late trip for an old far pile), said why; one 50 off (0.7
    s and its stand) taken with 1 s too: on her way."""
    far = 400 / WALK + 0.3
    assert sweep.plan([_spot("fresh", 200.0, allow=3.0)], (1, 0.0), (1, 0.0), line).keys() == ("fresh",)
    it = sweep.plan([_spot("old", 200.0, allow=1.0)], (1, 0.0), (1, 0.0), line)
    assert not it.stops and it.out == {"old": f"cost {far:.1f} s over 1.0"}
    assert sweep.plan([_spot("near", 50.0, allow=1.0)], (1, 0.0), (1, 0.0), line).keys() == ("near",)
    # A coin 160 units off her stop, a round trip of 2.4 s: fetched fresh, where the 2 s budget never fetched it (the
    # independent check of 2026-11-01, finding 2), whatever her time's worth
    assert sweep.plan([_spot("coin", 160.0, worth=15.0, allow=3.0)], (1, 0.0), (1, 0.0), line).keys() == ("coin",)


def test_an_itinerary_is_within_the_sum_of_its_piles_allowances():
    """Two fresh piles 150 either side of her (2.0 s there and back and its stand each, 3.0 s allowed each): both, 4.6 s,
    within their 6 s together, though over either's alone."""
    a, b = _spot("a", -150.0, allow=3.0), _spot("b", 150.0, allow=3.0)
    it = sweep.plan([a, b], (1, 0.0), (1, 0.0), line)
    assert set(it.keys()) == {"a", "b"} and 3.0 < it.extra <= it.allowed == 6.0
    # ... and within the budget, where one is given (the hunt loop's way out, its lull)
    assert len(sweep.plan([a, b], (1, 0.0), (1, 0.0), line, budget=3.0).keys()) == 1


def test_the_most_mesos_then_the_least_time():
    """Of the itineraries within their allowances the one picking up the most (40 + 10 over 40 alone), and of two as
    good the quicker: within a budget of 1.7 s, a coin 50 off her way rather than its twin 100 off."""
    big, small = _spot("big", 100.0, worth=40.0), _spot("small", -100.0, worth=10.0)
    assert set(sweep.plan([big, small], (1, 0.0), (1, 0.0), line).keys()) == {"big", "small"}
    near, far = _spot("near", 50.0, worth=15.0), _spot("far", -100.0, worth=15.0)
    it = sweep.plan([near, far], (1, 0.0), (1, 0.0), line, budget=1.7)
    assert it.keys() == ("near",) and it.value == pytest.approx(15.0) and it.out == {"far": "cost"}


def test_she_goes_back_for_a_fresh_pile_behind_her_not_an_old_one():
    """Going on to x 600, a pile 120 behind her (1.9 s back and forth): taken while it's fresh (3 s), not once it's old
    (1 s); the same worth a fifth as likely to lie there by then is still worth the trip, and counts a fifth."""
    fresh = _spot("fresh", -120.0, worth=40.0, still=lambda dt: 0.99)
    old = _spot("old", -120.0, worth=40.0, allow=1.0)
    assert sweep.plan([fresh], (1, 0.0), (1, 600.0), line).keys() == ("fresh",)
    assert sweep.plan([old], (1, 0.0), (1, 600.0), line).keys() == ()
    unlikely = sweep.plan([_spot("u", -120.0, worth=40.0, still=lambda dt: 0.2)], (1, 0.0), (1, 600.0), line)
    assert unlikely.keys() == ("u",) and unlikely.value == pytest.approx(8.0)


def test_drops_not_landed_yet_are_waited_for_within_their_allowance():
    """By her at x 20, her kill's drops land 1.5 s from now: waited for (1.8 s with the stand, within 3 s), and the wait
    counts in the allowance: landing in 3 s, they aren't (3.4 s)."""
    landing = _spot("k", 20.0, worth=40.0, lands=1.5)
    it = sweep.plan([landing], (1, 0.0), (1, 0.0), line)
    assert it.keys() == ("k",) and it.stops[0].arrive == pytest.approx(1.5)
    late = _spot("late", 20.0, worth=400.0, lands=3.0)
    assert sweep.plan([late], (1, 0.0), (1, 0.0), line).out["late"] == f"cost {3.0 + 0.3 + 20 / WALK:.1f} s over 3.0"


def test_a_guard_is_shot_first_its_kill_time_not_charged_to_the_allowance():
    """A pile a Horny Mushroom would meet her on: its 2 s kill comes before the stand (its EXP: "kill all the mobs that
    are in the way"), and isn't counted against the pile's 3 s: 2.3 s of walk and stand."""
    horny = Guard(SimpleNamespace(kind="horny-mushroom"), 2.0, 41.0)
    guarded = _spot("g", 150.0, worth=20.0, guards=[horny])
    it = sweep.plan([guarded], (1, 0.0), (1, 0.0), line)
    st = it.first
    assert st.spot.key == "g" and st.arrive == pytest.approx(150 / WALK + 2.0)
    assert it.extra == pytest.approx(300 / WALK + 2.0 + 0.3) and it.extra - it.clear <= it.allowed
    assert it.value == pytest.approx(20.0)


def test_the_order_is_the_quickest_two_piles_on_two_platforms():
    """From platform 1 at x 0 on to platform 3 at x 900: the piles on 1 (x 200) and 2 (x 100) taken in that order (1.9
    s more; the other way round 3.6)."""
    a, b = _spot("on 1", 200.0, platform=1), _spot("on 2", 100.0, platform=2)
    it = sweep.plan([b, a], (1, 0.0), (3, 900.0), line)
    assert it.keys() == ("on 1", "on 2")


def test_no_way_and_too_dear_on_its_own_are_left_out_and_said():
    def arrive(a, ax, b, bx):
        return None if b == 9 or a == 9 else line(a, ax, b, bx)
    it = sweep.plan([_spot("cut off", 0.0, platform=9), _spot("far", 2000.0)], (1, 0.0), (1, 0.0), arrive)
    assert it.out == {"cut off": "no way", "far": f"cost {2 * 2000 / WALK + 0.3:.1f} s over 3.0"}


def test_an_itinerary_is_kept_unless_another_is_clearly_better():
    """Two piles each worth the other's detour (patrol's warning case, loop_tree_grind_jump_key): solved again at every
    look, she keeps the one she's going for; one picking up clearly more (KEEP_SHARE) is taken."""
    sw = Sweeper()
    left, right = _spot("left", -50.0, worth=40.0), _spot("right", 50.0, worth=40.0)
    first = sw.solve([left, right], (1, 0.0), (1, 0.0), line).keys()
    assert len(first) == 2
    for x in (5.0, -5.0, 10.0, -10.0):  # where she is, a step either way: the same order on
        assert sw.solve([left, right], (1, x), (1, 0.0), line).keys() == first
    # One only within a budget, a little more worth on the other side: the one she's on kept
    sw2 = Sweeper()
    sw2.solve([_spot("a", -150.0, worth=40.0)], (1, 0.0), (1, 0.0), line, budget=3.0)
    pair = [_spot("a", -150.0, worth=40.0), _spot("b", 150.0, worth=44.0)]
    assert sw2.solve(pair, (1, 0.0), (1, 0.0), line, budget=3.0).keys() == ("a",)
    pair[1] = _spot("b", 150.0, worth=400.0)
    assert sw2.solve(pair, (1, 0.0), (1, 0.0), line, budget=3.0).keys() == ("b",)


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
    spots = [_spot(i, rng.uniform(-600, 600), rng.uniform(10, 80), rng.randint(1, 4), allow=1e9) for i in range(8)]
    times = []
    for _ in range(9):
        t0 = time.perf_counter()
        sweep.plan(spots, (2, 0.0), (3, 300.0), line)
        times.append(time.perf_counter() - t0)
    assert sorted(times)[4] < 0.025  # (median; 8 candidates; the design's 5 ms on a quiet core, CI's leeway)


def test_the_record_says_the_order_its_numbers_and_why_the_rest_were_left():
    it = sweep.plan([_spot("p", 75.0, worth=30.0), _spot("q", 3000.0)], (1, 0.0), (1, 0.0), line)
    rec = it.record()
    assert rec["order"] == [[1, 75, pytest.approx(75 / WALK, abs=0.01), 30.0, 0, 3.0]]
    assert rec["out"] == {"q": "cost 40.6 s over 3.0"}
    assert set(rec) == {"order", "extra", "value", "direct", "budget", "out"} and rec["budget"] is None
