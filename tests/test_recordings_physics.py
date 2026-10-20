"""The engine (motion.body) against her tracked movement in six recordings of the game: a walk with a
jump (motion), a walk (walk), three jump-backs keyed 80 ms after the arrow (jb_k165, jb_k190, jb_k450)
and a jump straight up with the key held (held_jump). The tracks are our own measurements of the
recordings, kept as small fixtures in tests/fixtures/tracks (`python -m gamelab.analysis.physics_check
--fixture NAME` remakes one from lab/recordings). Errors are RMS, in map units."""

import numpy as np
import pytest

from gamelab.analysis import physics_check as pc
from gamelab.motion import body

NAMES = ["motion", "walk", "jb_k165", "jb_k190", "jb_k450", "held_jump"]
LAGS = body.Lags()
SHOE = body.Shoe.from_physics(body.PHYSICS_IMG, speed=119.2, jump=100)


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
    ("motion", 1.3, 2.5),  # measured 1.04 / 2.23
    ("walk", 1.3, 0.1),    # 0.61 / 0.01
    ("jb_k190", 2.0, 1.6),  # 1.74 / 1.44
    ("jb_k165", 4.6, 1.5),  # 4.33 / 1.25: the kick's blind spot (see below)
    ("jb_k450", 5.9, 2.1),  # 5.60 / 1.86
    ("held_jump", 0.2, 4.0),  # 0.12 / 3.71: it lifted off about 80 ms after the key (18 jumps: 45-95)
])
def test_with_the_median_lags_the_engine_follows_her(name, x_rms, y_rms):
    ex, ey, _ = engine_errors(name)
    assert ex <= x_rms and ey <= y_rms


def test_the_engine_fits_the_recordings_better_than_the_model_it_replaces():
    before, engine = [], []
    for name in NAMES:
        tr, until, floors = case(name)
        before.append(pc.errors(tr, pc.steps(tr.events, until, pc.BEFORE, floors=floors), until)[:2])
        engine.append(engine_errors(name)[:2])
    before, engine = np.mean(before, axis=0), np.mean(engine, axis=0)
    assert engine[0] < before[0] - 0.4 and engine[1] < before[1] - 1.0  # 2.24 / 1.75 against 2.86 / 3.29


@pytest.mark.parametrize("name, kicked", [("jb_k165", False), ("jb_k190", True), ("jb_k450", False)])
def test_with_their_own_lags_the_jump_backs_take_off_as_seen(name, kicked):
    """Each jump-back's jump key came 80 ms after the arrow, where the kick decides her speed: it comes
    only below 0.8 x 149 on the ground, less than 5.1 units into a standing start. In jb_k165 and jb_k450
    she had run 4-6 units when she lifted off: no kick, she left at 121-127. In jb_k190 the arrow was
    seen late, she had gone 1.5, and the kick took her to 142. With lags fitted to the recording, the
    engine lifts her off as far along, takes off the same way, and follows her to half a unit."""
    tr, until, floors = case(name)
    lags = pc.fit_lags(tr, until, floors)
    ex, ey, _ = pc.errors(tr, pc.engine(tr.events, until, lags, floors, SHOE), until)
    assert ex <= 0.6 and ey <= 2.0
    r = body.replay(tr.events, until, SHOE, lags)
    (t_off, v_ground, vx), = r.takeoffs
    assert (abs(vx) > abs(v_ground) + 1) == kicked
    p = np.array(r.points)
    ran, seen = (np.interp(t_off - 0.01, t, x) for t, x in ((p[:, 0], p[:, 1]), (tr.t, tr.x)))  # just before
    assert ran == pytest.approx(seen, abs=1.0) and (abs(seen) < 3) == kicked


def test_held_jump_with_its_own_lag_rises_and_falls_as_the_game_does():
    tr, until, floors = case("held_jump")
    lags = pc.fit_lags(tr, until, floors)
    ex, ey, _ = pc.errors(tr, pc.engine(tr.events, until, lags, floors, SHOE), until)
    assert lags.jump == pytest.approx(0.080, abs=0.006) and ey <= 1.0  # 555 up, 2000 down: 0.93
