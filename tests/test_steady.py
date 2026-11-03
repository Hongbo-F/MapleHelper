"""Her steady walks and climbs in a run's places (motion.steady: analysis.walkfit's estimator, offline; her planner plans
by her stat window's SPEED, unmeasured: tests/test_panel_speed.py), on her recordings (tests/fixtures/steady/
places_tree_2026.json: R1's grind whole, the hour's and the thief's Tree stretches, made by D:/MapleHelper-scratch/
recfix/calib/scripts/make_fixture.py): the walk research's numbers, knocks, falls and misplaced looks, a flat walk's
braking tail and a slope's v83 rule."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gamelab.motion import body, physics, steady
from gamelab.motion.steady import Place
from gamelab.sim import scenarios

FIX = json.loads((Path(__file__).parent / "fixtures" / "steady" / "places_tree_2026.json").read_text(encoding="utf-8"))
MODES = {0: "ground", 1: "air", 2: "rope"}
HOWS = {0: "tag", 1: "camera", 2: "dot", 3: None}
REAL = pytest.mark.skipif(not scenarios.available(scenarios.TREE), reason="needs Tree Dungeon's game data (gamedata/data)")


def _undo(col: list) -> list:
    out, v = [], 0
    for i, d in enumerate(col):
        v = d if i == 0 else v + d
        out.append(v)
    return out


def _runs(col: list) -> list:
    return [v for v, n in col for _ in range(n)]


def places(which: str) -> list[list[Place]]:
    """The fixture's tracks of set `which`, each her places in order."""
    out = []
    for tr in FIX["sets"][which]["tracks"]:
        t, x, y = _undo(tr["t"]), _undo(tr["x"]), _undo(tr["y"])
        pl, mode, how = _runs(tr["platform"]), _runs(tr["mode"]), _runs(tr["how"])
        out.append([Place(tr["t0"] + t[i] / 1000, x[i] / 10, y[i] / 10, None if pl[i] < 0 else pl[i], MODES[mode[i]],
                          HOWS[how[i]]) for i in range(len(t))])
    return out


def measured(which: str, **kw):
    """Every walk and climb of set `which`'s tracks (steady.measure each, on Tree Dungeon's model)."""
    model = scenarios.real_map(scenarios.TREE)
    ws, cs = [], []
    for track in places(which):
        w, c = steady.measure(track, model, **kw)
        ws += w
        cs += c
    return ws, cs


# --- the estimator --------------------------------------------------------------------------------------------------

@REAL
def test_her_walk_and_climb_on_the_hours_stretches_are_the_researchs():
    """The hour of 2026-11-02 (00:17:48-00:53:06): 66 steady walks, her walk on flat ground 149.44 by the median (the
    walk research's walk_fixture.py: tests/fixtures/walks_tree_2026.json, the same 66), her x speed 148.83; the
    client's 150 at her stat window's SPEED 100 within 0.5%. Her climbs: 42 up at 120.1, 5 down at 119.6 (SPEED + 20:
    120; profile_checks.py's 5-place windows had 119.9 and 119.6)."""
    ws, cs = measured("hour")
    e = steady.estimate([w.limit for w in ws])
    assert e.n == 66 and e.value == pytest.approx(149.44, abs=0.01) and e.se < 0.5
    assert steady.estimate([w.v for w in ws]).value == pytest.approx(148.83, abs=0.01)
    walks = json.loads((Path(__file__).parent / "fixtures" / "walks_tree_2026.json").read_text(encoding="utf-8"))
    assert sorted(w.limit for w in ws) == pytest.approx(sorted(r[-1] for r in walks["characters"]["magician"]["walks"]),
                                                        abs=0.25)  # (the fixture's places to 0.1 unit and 1 ms)
    assert e.value == pytest.approx(physics.walk_of(100), rel=0.005)
    up = steady.estimate([c.v for c in cs if c.way == steady.UP])
    down = steady.estimate([c.v for c in cs if c.way == steady.DOWN])
    assert (up.n, down.n) == (42, 5) and up.value == pytest.approx(120.1, abs=0.05)
    assert down.value == pytest.approx(119.6, abs=0.05) and abs(up.value - physics.climb_of(100)) < 0.5


@REAL
def test_the_thiefs_walk_on_his_tree_runs_and_no_climb_by_his_minimap_dot():
    """The thief on Tree Dungeon (the bot's hunt of 10-26 22:54 and the user's play of 10-30): 77 walks, 150.9 by the
    median (the research's per set 151.0 and 150.7), the client's 151.25 at his stat window's 101 within 0.5%; his
    climbs then were placed by the minimap's dot alone (rope places not by PLACED_BY): none taken."""
    ws, cs = measured("thief")
    e = steady.estimate([w.limit for w in ws])
    assert e.n == 77 and e.value == pytest.approx(physics.walk_of(101), rel=0.005) and cs == []


@REAL
def test_r1_whole_with_its_knocks_falls_and_looks_as_they_came():
    """R1's grind whole (00:09:36-00:14:57, every scene record): 9 steady walks at 149.8, 10 climbs at 119.5 up; the
    stretches its knocks, falls and jumps cut aren't taken. A look misplaced 20 units in a walk, or a knock's 0.27 s in
    the air in it, cuts it there: what's left is taken or not, never off by the misplaced look."""
    (track,) = places("r1")
    ws, cs = steady.measure(track, scenarios.real_map(scenarios.TREE))
    assert len(ws) == 9 and steady.estimate([w.limit for w in ws]).value == pytest.approx(149.8, abs=0.05)
    assert len(cs) == 10 and steady.estimate([c.v for c in cs if c.way == steady.UP]).value == pytest.approx(
        119.5, abs=0.05)
    assert all(steady.WALK_BAND[0] < w.limit < steady.WALK_BAND[1] for w in ws)
    w = max(ws, key=lambda w: w.span)
    i = next(i for i, q in enumerate(track) if q.t == w.t)
    j = i + len([q for q in track[i:i + 40] if q.platform == w.platform and q.mode == "ground"]) // 2
    for spoil in (lambda q: q._replace(x=q.x + 20.0), lambda q: q._replace(mode="air", platform=None)):
        bad = track[:j] + [spoil(q) for q in track[j:j + 2]] + track[j + 2:]
        got, _ = steady.measure(bad, scenarios.real_map(scenarios.TREE))
        near = [g for g in got if abs(g.t - w.t) < 3.0]
        assert all(abs(g.limit - w.limit) < 3.0 for g in near), (near, w)


def _walk(v: float, seconds: float, x0: float = 0.0, hz: float = 15.0, platform: int = 0,
          t0: float = 0.0) -> list[Place]:
    """Her places walking from standing at a steady `v` (her run-up at 1400 units/s², her braking at 800 after it as
    motion.body has them: the release's lag, then the slide), then standing a while."""
    shoe = dataclasses.replace(body.Shoe.from_physics(body.PHYSICS_IMG, 100.0, 100.0), vmax=v)
    out, t, x, u = [], 0.0, x0, 0.0
    while t < seconds + 1.0:
        out.append(Place(t0 + t, x, 0.0, platform, "ground", "tag"))
        d = 1 if t < seconds else 0
        steps = 10
        for _ in range(steps):
            u, dx = body.ground_step(u, d, shoe, 1 / hz / steps)
            x += dx
        t += 1 / hz
    return out


def test_a_walk_on_flat_ground_is_its_walk_its_braking_tail_left_out():
    """A walk of 1.5 s at 150 on flat ground, her run-up and braking as the engine has them: 150 within 0.3, its tail out
    (TAIL_S); with it in, her braking pulls it down."""
    ws, _ = steady.measure(_walk(150.0, 1.5))
    assert len(ws) == 1 and ws[0].limit == pytest.approx(150.0, abs=0.3) and ws[0].ground == 1.0
    raw, _ = steady.measure(_walk(150.0, 1.5), tail=0.0)
    assert raw[0].limit < ws[0].limit


def test_a_slopes_v83_rule_is_taken_out():
    """On a foothold of slope 0.4 her x speed is her walk times the cosine uphill, times (1 + sin²) cos downhill (the
    v83 rule); the estimator gives her walk back either way."""
    from gamelab.localization.mapdata import Platform, Segment
    p = Platform(0, [Segment(0.0, 0.0, 1000.0, -400.0)])  # (y down: going right is uphill)
    m = 0.4
    cos, s2 = 1 / (1 + m * m) ** 0.5, m * m / (1 + m * m)
    for way, share in ((1, cos), (-1, (1 + s2) * cos)):
        g = steady.ground_factor(p, 100.0, 900.0, way)
        assert g == pytest.approx(1 / share, rel=1e-6)
    model = SimpleNamespace(platform=lambda pid: p)
    track = [q._replace(x=100 + q.x) for q in _walk(150.0 * cos, 2.0)]
    ws, _ = steady.measure(track, model)
    assert ws[0].limit == pytest.approx(150.0, abs=0.4) and ws[0].v < 140.0
