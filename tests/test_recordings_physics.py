"""The engine (motion.body) against her tracked movement in the game: six recordings at 60 fps, a walk with a
jump (motion), a walk (walk), three jump-backs keyed 80 ms after the arrow (jb_k165, jb_k190, jb_k450) and
a jump straight up with the key held (held_jump); and the two jump sweeps of 2026-10-21 (34 jumps from
standing keyed 0-160 ms after the arrow, 18 of them tracked at 30 fps). The tracks are our own measurements
of the recordings, kept as small fixtures in tests/fixtures/tracks (`python -m gamelab.analysis.physics_check
--fixture NAME` remakes one from lab/recordings) and tests/fixtures/kick_sweeps_2026-10-21.json (its
`method` says how each number was taken). Errors are RMS, in map units."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gamelab.analysis import physics_check as pc
from gamelab.motion import body, physics

NAMES = ["motion", "walk", "jb_k165", "jb_k190", "jb_k450", "held_jump"]
LAGS = body.Lags()
SHOE = body.Shoe.from_physics(body.PHYSICS_IMG, speed=119.2, jump=100)
SWEEPS = json.loads((Path(__file__).parent / "fixtures" / "kick_sweeps_2026-10-21.json").read_text(encoding="utf-8"))
# The engine as it was before the sweeps: a 5 ms step, and the lags of the 60 fps recordings with their spread.
FIVE_MS = body.TICK
OLD_LAGS = body.Lags(on=0.060, off=0.072, jump=0.067, sd_on=0.010, sd_off=0.011, sd_jump=0.012)


def case(name):
    tr = pc.fixture(name)
    until, floors = pc.CASES[name]
    return tr, until, floors


def engine_errors(name, lags=LAGS):
    tr, until, floors = case(name)
    return pc.errors(tr, pc.engine(tr.events, until, lags, floors, SHOE), until)


def test_the_fixtures_are_her_tracks():
    for name in NAMES:
        tr, until, _ = case(name)
        ok = tr.used(until)
        assert ok.sum() >= 25 and tr.x[0] == tr.y[0] == 0  # from where she stood at the first frame
        assert any(a in ("left", "right") for _, a, _ in tr.events) or name == "held_jump"


@pytest.mark.parametrize("name, x_rms, y_rms", [
    ("motion", 2.3, 2.0),   # measured 2.10 / 1.81 (1.38 in x with her landing not waiting for the next step)
    ("walk", 1.5, 0.1),     # 1.35 / 0.01: 0.43 on the 5 ms engine (see below)
    ("jb_k190", 2.4, 1.4),  # 2.21 / 1.25: 1.74 on the 5 ms engine, where its jump at 80 ms was kicked by the median
                            # lags as she was; on the steps 58% of the phases leave at 126 (0.6 with its own: below)
    ("jb_k165", 1.0, 1.2),  # 0.88 / 1.05: the blind spot (see below), half the phases leaving at 126
    ("jb_k450", 1.9, 2.7),  # 1.73 / 2.51
    ("held_jump", 0.2, 4.4),  # 0.12 / 4.18: it lifted off about 84 ms after the key (36 jumps: 45-95)
])
def test_with_the_median_lags_the_engine_follows_her(name, x_rms, y_rms):
    """The engine's path is the mean over the phases its 30 ms steps can have against the keys (what the
    planner's closed forms expect). On the 5 ms engine with the lags then the jump-backs were off by 4.3-5.6
    units in x (a jump at 80 ms took her kicked or not by the lags alone), the walks by 0.6-1.0; y by 1.25-3.7.
    The walks are followed less well than they were: over the seven 60 fps walks (walk, walk300, walk600,
    walk1000, do_walk, do_walk2, motion) 1.64 units in x against 1.17, two better and five worse, and the long
    ones end 1.6-5.6 units short of where she stopped (walk1000 154.9 against 158.1). Most of that is the
    release's lag: fitted to the sweeps it's 2 ms after the press's (68 against 66), where the walks want it
    about 12 after (as on the 5 ms engine, 72 against 60); with 78 they're off by 1.52 (walk 0.91)."""
    ex, ey, _ = engine_errors(name)
    assert ex <= x_rms and ey <= y_rms


def test_the_engine_fits_the_recordings_better_than_the_model_it_replaces():
    before, engine = [], []
    for name in NAMES:
        tr, until, floors = case(name)
        before.append(pc.errors(tr, pc.steps(tr.events, until, pc.BEFORE, floors=floors), until)[:2])
        engine.append(engine_errors(name)[:2])
    before, engine = np.mean(before, axis=0), np.mean(engine, axis=0)
    assert engine[0] < before[0] - 1.2 and engine[1] < before[1] - 1.2  # 1.40 / 1.80 against 2.86 / 3.29


@pytest.mark.parametrize("name, kicked", [("jb_k165", False), ("jb_k190", True), ("jb_k450", False)])
def test_with_their_own_lags_the_jump_backs_take_off_as_seen(name, kicked):
    """Each jump-back's jump key came 80 ms after the arrow, where the game's steps decide her speed: three
    steps of the arrow before the jump's have her at 126, too fast for the kick (below 0.8 x 149), two at 84,
    kicked to full speed. In jb_k165 and jb_k450 she had run 4-6 units when she lifted off (three steps: 5.7
    by the engine): no kick, she left at 121-127. In jb_k190 the arrow was seen a step later, she had gone
    1.5 (two steps: 2.5), and the kick took her to 142. With lags and the steps' phase fitted to the
    recording, the engine lifts her off as far along, takes off the same way, and follows her to 0.6 units."""
    tr, until, floors = case(name)
    lags, phase = pc.fit_lags(tr, until, floors)
    ex, ey, _ = pc.errors(tr, pc.engine(tr.events, until, lags, floors, SHOE, phase=phase), until)
    assert ex <= 0.6 and ey <= 1.3
    r = body.replay(tr.events, until, SHOE, lags, phase=phase)
    (t_off, v_ground, vx), = r.takeoffs
    assert (abs(vx) > abs(v_ground) + 1) == kicked
    p = np.array(r.points)
    ran, seen = (np.interp(t_off - 0.01, t, x) for t, x in ((p[:, 0], p[:, 1]), (tr.t, tr.x)))  # just before
    assert ran == pytest.approx(seen, abs=0.6) and (abs(seen) < 3) == kicked


def test_held_jump_with_its_own_lag_rises_and_falls_as_the_game_does():
    tr, until, floors = case("held_jump")
    lags, phase = pc.fit_lags(tr, until, floors)
    ex, ey, _ = pc.errors(tr, pc.engine(tr.events, until, lags, floors, SHOE, phase=phase), until)
    (t_off, _, _), = body.replay(tr.events, until, SHOE, lags, phase=phase).takeoffs
    key = next(t for t, a, down in tr.events if a == "jump" and down)
    assert t_off - key == pytest.approx(0.084, abs=0.006) and ey <= 1.0  # 555 up, 2000 down: 0.93


# --- the jump sweeps of 2026-10-21 -----------------------------------------------------------------------

def _events(run):
    return [(t / 1000, action, how == "down") for t, action, how in run["keys"]]


def _own(run, step=body.STEP):
    """A tracked jump replayed with what its track shows: the game's steps on the step that lifted her off
    (its phase), and the lags from its keys to the start of her run and to the lift-off (each including the
    wait for its step), at 1 ms."""
    lags = body.Lags(on=run["on_lag_ms"] / 1000, jump=run["jump_lag_ms"] / 1000)
    phase = (round(run["liftoff_ms"]) % round(step * 1000)) / 1000
    return body.replay(_events(run), 1.6, SHOE, lags, dt=0.001, step=step, phase=phase)


def _landing_x(r, sign):
    t_off = r.takeoffs[0][0]
    return next(sign * x for t, x, y in r.points if t > t_off + 0.05 and y == 0)


def test_the_sweeps_fixture_is_our_measurements():
    s1, s2 = SWEEPS["sweep1"], SWEEPS["sweep2"]
    assert len(s1) == 16 and len(s2) == 18
    for run in s1 + s2:
        assert [a for _, a, _ in run["keys"]] == [("right" if run["dir"] == "r" else "left"), "jump", "jump",
                                                  ("right" if run["dir"] == "r" else "left")]
        assert abs(run["keys"][1][0] - run["keys"][0][0] - run["gap_ms"]) < 1
    for run in s2:
        tr = run["track"]
        assert len(tr["t"]) == len(tr["x"]) == len(tr["used"]) >= 55 and sum(tr["used"]) >= 50
        assert (run["outcome"] == "short") == (run["air_vx"] < 140)


def test_with_each_tracked_jumps_own_lags_the_engine_takes_off_and_lands_as_she_did():
    """The game moves her in steps of 30 ms: her ground run when she left sat on the steps' ladder (0.63, 2.52,
    5.67, 9.80, 14.27 units after one to five steps of 1400 units/s²), and the speed she left with on the kick's
    rule over it: after one or two steps kicked to full speed, after three at 126 (all five such: 124.7-126.9),
    after four or more at full speed. Each track's lags are read off its lift-off: the arrow's from how far she
    had run by then and when (on_lag_ms, the fixture's `method`), the steps' phase from when. So the engine
    lifting her off after as many steps and as far along follows from how they were read, and checks only that
    the reading holds together. What this tests is the speed she left with, the kick's rule over those steps
    (to 4 units/s in all 18), and so where she came down (within 2 units; the fixture's landing is the line of
    her air speed). The 5 ms engine with the same lags has three of the five that left at 126 leave at 133, 133
    and 149 (kicked from 119), landing 4-14 units off."""
    off = 0
    for run in SWEEPS["sweep2"]:
        sign = 1 if run["dir"] == "r" else -1
        r = _own(run, FIVE_MS)
        off += abs(abs(r.takeoffs[0][2]) - run["air_vx"]) > 4 and abs(_landing_x(r, sign) - run["landing_x"]) > 3
    assert off >= 3
    for run in SWEEPS["sweep2"]:
        sign = 1 if run["dir"] == "r" else -1
        r = _own(run)
        (t_off, v_ground, vx), = r.takeoffs
        steps = 4 if abs(v_ground) >= SHOE.vmax else round(abs(v_ground) / (SHOE.acc * body.STEP))
        assert min(steps, 4) == min(run["steps"], 4), run["name"]
        assert ("short" if abs(vx) < 140 else "full") == run["outcome"], run["name"]
        assert abs(vx) == pytest.approx(run["air_vx"], abs=4), run["name"]
        p = np.array(r.points)
        assert sign * np.interp(t_off, p[:, 0], p[:, 1]) == pytest.approx(run["ground_x_at_liftoff"], abs=1.0)
        assert _landing_x(r, sign) == pytest.approx(run["landing_x"], abs=3), run["name"]


def _called_right(runs, lags, step):
    """How many of `runs` the engine's chance of a short takeoff calls right: short where it's over a half."""
    return sum((body.short_takeoff_chance(run["gap_ms"], SHOE, lags, step=step) > 0.5) == (run["outcome"] == "short")
               for run in runs)


def test_the_engines_chance_of_a_short_takeoff_against_both_sweeps():
    """Called short where its chance is over a half, the engine with the median lags gets 25 of the 34 jumps
    right (24.5 expected by its own chances), the 5 ms engine with the lags then 23. No rule by the gap alone
    gets near 30 of 34: where the game's steps fall against the keys can't be seen, and the same gap goes
    either way (at 80 ms 2 of 4 left short), so even the sweeps' own majority at each gap gets 29. It calls
    wrong the two keyed 20 ms after the arrow (short, at 39%), 60_r (whose arrow the game acted on 37 ms after
    its key, the soonest of the 18), two of the four at 80 ms and one at 100 (at 58-62%), and three of the
    four at 120, which left short where it gives 16%. By the median lags it has too few short in all (8.8
    expected where 13 were; 5.5 from 80 to 120 ms, where 8 of 12 were): physics.BLIND_ENDS_MS is safe from
    that only as it takes the worst case over what the sweeps allow (SWEEP_LAG_GAP: up to 43% at 120 ms). By
    the likelihood of all 34 outcomes the step engine is e^7 likelier than the 5 ms one; what sets the two
    apart is the speeds the jumps left with (above)."""
    runs = SWEEPS["sweep1"] + SWEEPS["sweep2"]
    assert _called_right(runs, LAGS, body.STEP) == 25 and _called_right(runs, OLD_LAGS, FIVE_MS) == 23
    by_gap = {}
    for run in runs:
        by_gap.setdefault(round(run["gap_ms"] / 10), []).append(run["outcome"] == "short")
    assert sum(max(sum(v), len(v) - sum(v)) for v in by_gap.values()) == 29
    expected = sum(body.short_takeoff_chance(run["gap_ms"], SHOE, LAGS) for run in runs)
    assert expected == pytest.approx(8.8, abs=0.1) and sum(run["outcome"] == "short" for run in runs) == 13
    worst = max(body.short_takeoff_chance(120, SHOE, replace(LAGS, jump=LAGS.on + gap, sd_on=sd / math.sqrt(2),
                                                            sd_jump=sd / math.sqrt(2)))
                for gap in physics.SWEEP_LAG_GAP for sd in physics.SWEEP_LAG_GAP_SD)
    assert body.short_takeoff_chance(120, SHOE, LAGS) < 0.2 and worst > 0.4

    def log_likelihood(lags, step):
        p = [min(max(body.short_takeoff_chance(run["gap_ms"], SHOE, lags, step=step), 1e-4), 1 - 1e-4) for run in runs]
        return sum(math.log(q if run["outcome"] == "short" else 1 - q) for q, run in zip(p, runs))

    assert log_likelihood(LAGS, body.STEP) > log_likelihood(OLD_LAGS, FIVE_MS) + 6  # -15.8 against -23.2
