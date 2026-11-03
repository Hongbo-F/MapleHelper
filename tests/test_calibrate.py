"""Her walk and climb calibrated from perception's place track (motion.calibrate; the user, 2026-11-02: "I feel this
should just be measured and updated dynamically and persisted, constantly calibrated. Also correlate that with the stats
panel value."): the estimator on her recordings (tests/fixtures/calibrate/places_tree_2026.json: R1's grind whole, the
hour's and the thief's Tree stretches, made by D:/MapleHelper-scratch/recfix/calib/scripts/make_fixture.py), her file
kept between runs, the prior and her stat window's SPEED, and her planner following mid-run (motion.physics.propose,
settle, set_motion)."""

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gamelab import telemetry
from gamelab.motion import body, calibrate, physics
from gamelab.motion.calibrate import Calibration, Panel, Place, Store
from gamelab.sim import scenarios

FIX = json.loads((Path(__file__).parent / "fixtures" / "calibrate" / "places_tree_2026.json").read_text(encoding="utf-8"))
MODES = {0: "ground", 1: "air", 2: "rope"}
HOWS = {0: "tag", 1: "camera", 2: "dot", 3: None}
REAL = pytest.mark.skipif(not scenarios.available(scenarios.TREE), reason="needs Tree Dungeon's game data (gamedata/data)")
PRIOR = (physics.WALK_SPEED, physics.CLIMB_SPEED)  # (the thief's, as the tests play him: 151.25 and 121)
AT_IMPORT = {n: getattr(physics, n) for n in physics.WALK_NAMES}


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
    """Every walk and climb of set `which`'s tracks (calibrate.measure each, on Tree Dungeon's model)."""
    model = scenarios.real_map(scenarios.TREE)
    ws, cs = [], []
    for track in places(which):
        w, c = calibrate.measure(track, model, **kw)
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
    e = calibrate.estimate([w.limit for w in ws])
    assert e.n == 66 and e.value == pytest.approx(149.44, abs=0.01) and e.se < 0.5
    assert calibrate.estimate([w.v for w in ws]).value == pytest.approx(148.83, abs=0.01)
    walks = json.loads((Path(__file__).parent / "fixtures" / "walks_tree_2026.json").read_text(encoding="utf-8"))
    assert sorted(w.limit for w in ws) == pytest.approx(sorted(r[-1] for r in walks["characters"]["magician"]["walks"]),
                                                        abs=0.25)  # (the fixture's places to 0.1 unit and 1 ms)
    assert e.value == pytest.approx(physics.walk_of(100), rel=0.005)
    up = calibrate.estimate([c.v for c in cs if c.way == calibrate.UP])
    down = calibrate.estimate([c.v for c in cs if c.way == calibrate.DOWN])
    assert (up.n, down.n) == (42, 5) and up.value == pytest.approx(120.1, abs=0.05)
    assert down.value == pytest.approx(119.6, abs=0.05) and abs(up.value - physics.climb_of(100)) < 0.5


@REAL
def test_the_thiefs_walk_on_his_tree_runs_and_no_climb_by_his_minimap_dot():
    """The thief on Tree Dungeon (the bot's hunt of 10-26 22:54 and the user's play of 10-30): 77 walks, 150.9 by the
    median (the research's per set 151.0 and 150.7), the client's 151.25 at his stat window's 101 within 0.5%; his
    climbs then were placed by the minimap's dot alone (rope places not by PLACED_BY): none taken."""
    ws, cs = measured("thief")
    e = calibrate.estimate([w.limit for w in ws])
    assert e.n == 77 and e.value == pytest.approx(physics.walk_of(101), rel=0.005) and cs == []


@REAL
def test_r1_whole_with_its_knocks_falls_and_looks_as_they_came():
    """R1's grind whole (00:09:36-00:14:57, every scene record): 9 steady walks at 149.8, 10 climbs at 119.5 up; the
    stretches its knocks, falls and jumps cut aren't taken. A look misplaced 20 units in a walk, or a knock's 0.27 s in
    the air in it, cuts it there: what's left is taken or not, never off by the misplaced look."""
    (track,) = places("r1")
    ws, cs = calibrate.measure(track, scenarios.real_map(scenarios.TREE))
    assert len(ws) == 9 and calibrate.estimate([w.limit for w in ws]).value == pytest.approx(149.8, abs=0.05)
    assert len(cs) == 10 and calibrate.estimate([c.v for c in cs if c.way == calibrate.UP]).value == pytest.approx(
        119.5, abs=0.05)
    assert all(calibrate.WALK_BAND[0] < w.limit < calibrate.WALK_BAND[1] for w in ws)
    w = max(ws, key=lambda w: w.span)
    i = next(i for i, q in enumerate(track) if q.t == w.t)
    j = i + len([q for q in track[i:i + 40] if q.platform == w.platform and q.mode == "ground"]) // 2
    for spoil in (lambda q: q._replace(x=q.x + 20.0), lambda q: q._replace(mode="air", platform=None)):
        bad = track[:j] + [spoil(q) for q in track[j:j + 2]] + track[j + 2:]
        got, _ = calibrate.measure(bad, scenarios.real_map(scenarios.TREE))
        near = [g for g in got if abs(g.t - w.t) < 3.0]
        assert all(abs(g.limit - w.limit) < 3.0 for g in near), (near, w)


def _walk(v: float, seconds: float, x0: float = 0.0, hz: float = 15.0, platform: int = 0, t0: float = 0.0,
          tail: bool = True) -> list[Place]:
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
    ws, _ = calibrate.measure(_walk(150.0, 1.5))
    assert len(ws) == 1 and ws[0].limit == pytest.approx(150.0, abs=0.3) and ws[0].ground == 1.0
    raw, _ = calibrate.measure(_walk(150.0, 1.5), tail=0.0)
    assert raw[0].limit < ws[0].limit


def test_a_slopes_v83_rule_is_taken_out():
    """On a foothold of slope 0.4 her x speed is her walk times the cosine uphill, times (1 + sin²) cos downhill (the
    v83 rule); the estimator gives her walk back either way."""
    from gamelab.localization.mapdata import Platform, Segment
    p = Platform(0, [Segment(0.0, 0.0, 1000.0, -400.0)])  # (y down: going right is uphill)
    m = 0.4
    cos, s2 = 1 / (1 + m * m) ** 0.5, m * m / (1 + m * m)
    for way, share in ((1, cos), (-1, (1 + s2) * cos)):
        g = calibrate.ground_factor(p, 100.0, 900.0, way)
        assert g == pytest.approx(1 / share, rel=1e-6)
    model = SimpleNamespace(platform=lambda pid: p)
    track = [q._replace(x=100 + q.x) for q in _walk(150.0 * cos, 2.0)]
    ws, _ = calibrate.measure(track, model)
    assert ws[0].limit == pytest.approx(150.0, abs=0.4) and ws[0].v < 140.0


# --- her file ---------------------------------------------------------------------------------------------------------

def test_her_file_missing_corrupt_old_or_with_bad_rows_is_tolerated(tmp_path):
    path = tmp_path / "her_motion.json"
    assert Store(path).why == Store.NONE and Store(path).walks() == []
    path.write_text("{not json", encoding="utf-8")
    assert "doesn't read" in Store(path).why and Store(path).data == {}
    path.write_text(json.dumps({"version": calibrate.VERSION + 1, "walks": [[1, 2, 3, 1, 50, 150, 1, 150, 100]]}),
                    encoding="utf-8")
    assert "another version" in Store(path).why and Store(path).walks() == []
    good = [10.0, 10002073, 23, 1, 80.0, 150.1, 1.0, 150.1, 100.0]
    path.write_text(json.dumps({"version": calibrate.VERSION, "walks": [good, [1, 2], "x", [None] * 9,
                                                                         good[:-1] + ["fast"]],
                                "climbs": {"no": "list"}}), encoding="utf-8")
    s = Store(path)
    assert s.walks() == [good] and s.climbs() == [] and s.why == ""


def test_her_file_is_written_whole_and_a_failure_said_not_raised(tmp_path):
    said = []
    c = Calibration(tmp_path / "lab" / "her_motion.json", 101.0, log=said.append)
    for k in range(3):
        c.feed_states([])  # (nothing)
    assert c.save() and not (tmp_path / "lab" / "her_motion.json.tmp").exists()
    data = json.loads((tmp_path / "lab" / "her_motion.json").read_text(encoding="utf-8"))
    assert data["version"] == calibrate.VERSION and data["walk"]["prior"] == 151.25 and data["walks"] == []
    (tmp_path / "dir").mkdir()
    bad = Calibration(tmp_path / "dir", 101.0, log=said.append)  # (a folder where her file would be)
    assert not bad.save() and any("not saved" in line for line in said)


# --- the prior, her stat window, and her planner following ---------------------------------------------------------

def _rows(n: int, value: float, speed: float | None = 100.0, kind: str = "walk") -> list[list]:
    if kind == "walk":
        return [[float(i), 10002073, 23, 1, 80.0, value, 1.0, value + 0.1 * (i % 3 - 1), speed] for i in range(n)]
    return [[float(i), 10002073, -1, 60.0, value + 0.1 * (i % 3 - 1), speed] for i in range(n)]


def test_the_prior_is_the_clients_formula_at_her_stat_windows_speed_else_her_profiles():
    c = Calibration(None, 101.0, log=lambda _: None)
    assert c.target("walk") == (151.25, "prior", None) and c.target("climb")[:2] == (121.0, "prior")
    c.panel = Panel(104.0)
    assert c.prior_of("walk") == 155.0 and c.prior_of("climb") == 124.0
    given = Calibration(None, 101.0, prior=(125.0, 117.0), log=lambda _: None)
    assert (given.prior_of("walk"), given.prior_of("climb")) == (125.0, 117.0)


def test_her_measure_is_planned_once_enough_and_steady_and_only_beyond_the_deadband():
    c = Calibration(None, 100.0, panel=Panel(100.0), log=lambda _: None)
    c.rows["walk"] = _rows(calibrate.MIN_N["walk"] - 1, 149.4)
    assert c.target("walk")[1] == "prior"  # (9 of 10: her prior, 150, the 4 of the gross rule are only 0.4% off it)
    c.rows["walk"] = _rows(calibrate.MIN_N["walk"], 149.4)
    value, why, est = c.target("walk")
    assert why == "measured" and value == 149.4 and est.n == 10
    assert c.judge() and physics.pending() == (149.4, 120.0)  # (her climb her prior at her stat window's 100)
    assert physics.settle() and (physics.WALK_SPEED, physics.CLIMB_SPEED) == (149.4, 120.0)
    c.rows["walk"] += _rows(5, 149.9)  # (the median moves, under DEADBAND: nothing proposed)
    assert 0 < c.measured("walk").value - 149.4 < calibrate.DEADBAND and not c.judge() and physics.pending() is None
    # A prior far off (v83's 125): 4 stretches are enough (GROSS)
    far = Calibration(None, 100.0, prior=(125.0, 117.0), log=lambda _: None)
    far.rows["walk"] = _rows(calibrate.GROSS_N["walk"], 150.0)
    assert far.target("walk")[1] == "measured" and far.target("walk")[0] == pytest.approx(150.0, abs=0.1)
    far.rows["walk"] = _rows(calibrate.GROSS_N["walk"] - 1, 150.0)
    assert far.target("walk")[:2] == (125.0, "prior")


def test_stretches_at_another_speed_count_by_the_formula_until_hers_are_enough():
    """Measured at SPEED 100 (150), her stat window now 104: counted at 150 x 124/120 = 155 until 10 are measured at
    104; then those alone."""
    c = Calibration(None, 100.0, panel=Panel(104.0), log=lambda _: None)
    c.rows["walk"] = _rows(20, 150.0, speed=100.0)
    assert c.measured("walk").value == pytest.approx(150.0 * 124 / 120, abs=0.2)
    c.rows["walk"] += _rows(calibrate.MIN_N["walk"], 158.0, speed=104.0)
    assert c.measured("walk").value == pytest.approx(158.0, abs=0.2) and c.measured("walk").n == 10


def test_her_stat_windows_speed_correlated_within_noted_beyond_warned():
    with telemetry.captured() as records:
        c = Calibration(None, 100.0, log=lambda _: None)
        c.rows["walk"] = _rows(15, 149.4)
        notes = c.panel_seen(100, 100)
        assert notes[-1].endswith("her stat window's 100: within 2") and "implies SPEED 99.5" in notes[-1]
        assert not [r for r in records if r.get("warn")]
        notes = c.panel_seen(103, 100)  # (a buff, gear: her measure at 100 counted at 103 by the formula, still off)
        assert "103 before" not in notes[0] and "her SPEED 103 by her stat window" in notes[0]
    c2 = Calibration(None, 100.0, panel=Panel(100.0), log=lambda _: None)
    c2.rows["walk"] = _rows(15, 156.0)  # (she walks 156: SPEED 104.8 implied)
    with telemetry.captured() as records:
        note = c2.correlation(force=True)
    assert "off by more than 2" in note and "planned at her measure all the same" in note
    (r,) = [r for r in records if r.get("topic") == "calibration"]
    assert r["what"] == "walk" and r["panel"] == 100.0 and r["implied"] == pytest.approx(104.8, abs=0.05)


def test_a_change_waits_for_settle_and_is_made_everywhere_her_walk_is():
    """physics.propose holds a change until settle (where no chain of hers runs), told then; set_motion makes every
    name of her walk again in every module that took one (`from .physics import WALK_SPEED`), the modules' own of them
    (follows), counts EPOCH on; the simulator's world walks her at her truth all the same."""
    from gamelab.control import actions, mover, navigator
    from gamelab.motion import flow, route
    from gamelab.sim import world
    told = []
    epoch = physics.EPOCH
    took = [(mod, n) for name, mod in list(sys.modules.items()) if name.startswith("gamelab.") and mod is not physics
            for n in physics.WALK_NAMES if vars(mod).get(n) is getattr(physics, n)]
    assert {(m.__name__, n) for m, n in took} >= {("gamelab.behavior.hunt", "WALK_SPEED"),
                                                  ("gamelab.control.mover", "CLIMB_SPEED"),
                                                  ("gamelab.motion.flow", "SHOE"), ("gamelab.motion.flow", "STOP_RUN")}
    physics.propose(125.0, 117.0, then=lambda old, new: told.append((old, new)))
    assert physics.WALK_SPEED == PRIOR[0] and physics.pending() == (125.0, 117.0) and physics.EPOCH == epoch
    assert physics.settle() and told == [(PRIOR, (125.0, 117.0))] and physics.pending() is None
    assert physics.EPOCH == epoch + 1 and not physics.settle()
    assert physics.SHOE.vmax == 125.0 and physics.SHOE.vjump == physics.PRIOR_SHOE.vjump
    for mod, n in took:  # (every module that took a name of her walk: its new value)
        assert vars(mod)[n] is getattr(physics, n), (mod.__name__, n)
    assert mover.WALK_SPEED == navigator.WALK_SPEED == flow.WALK_SPEED == 125.0 and route.CLIMB_SPEED == 117.0
    assert flow.ROPE_SPEED == navigator.ROPE_SPEED == 117.0 and navigator.KEY_LEAD["walk_off"] == physics.STOP_S
    assert flow.GRAB_LEAD == route.GRAB_LEAD == pytest.approx(125.0 * physics.JUMP_SPEED / physics.GRAVITY)
    assert route.ROPE_VX == pytest.approx(1.3 * 125.0) and route.ROPE_JUMP_SLIDE == body.slide(route.ROPE_VX,
                                                                                                physics.SHOE)
    assert actions.JB_RUN_MS == physics.BLIND_ENDS_MS == 60  # (the blind spot of three steps ends sooner at 125)
    assert physics.RUN_LAG == pytest.approx(physics.LAGS.on + 125.0 / 2800) and physics.calibrated()
    assert world.her_shoe(world.Her()) is physics.PRIOR_SHOE and world.her_climb(world.Her()) == PRIOR[1]
    physics.reset()
    assert {n: getattr(physics, n) for n in physics.WALK_NAMES} == AT_IMPORT and not physics.calibrated()
    assert flow.ROPE_SPEED == navigator.ROPE_SPEED == PRIOR[1] and actions.JB_RUN_MS == 150


@REAL
def test_a_navigator_makes_its_route_graph_again_at_its_next_pass_and_a_chain_planned_keeps_its_timings():
    """A change proposed mid-run is made at the navigator's next pass (Navigator._follow_walk, at Navigator.step's start),
    its route graph made again there (its climbs' costs at the new climb); the moves planned before keep their times."""
    from gamelab.control.navigator import Navigator
    from gamelab.motion.route import Planner
    model = scenarios.real_map(scenarios.TREE)
    nav = Navigator.__new__(Navigator)
    nav.model, nav.planner, nav._epoch, nav._siblings = model, Planner(model), physics.EPOCH, {"x": 1}
    before = nav.planner
    climb = next(m for m in before.moves if m.kind == "climb")
    seconds = climb.seconds
    nav._follow_walk()
    assert nav.planner is before and nav._siblings == {"x": 1}  # (nothing proposed: as it was)
    physics.propose(PRIOR[0], 110.0)
    assert nav.planner is before  # (not before the pass)
    nav._follow_walk()
    assert nav.planner is not before and nav._siblings is None and physics.CLIMB_SPEED == 110.0
    again = next(m for m in nav.planner.moves if (m.kind, m.src, m.dst, m.x) == (climb.kind, climb.src, climb.dst,
                                                                              climb.x))
    assert climb.seconds == seconds and again.seconds > seconds  # (the one planned before keeps its time)


def _states(places: list[Place], seq0: int = 1, epoch: int = 1):
    return [SimpleNamespace(seq=seq0 + i, epoch=epoch, t=q.t, where=(q.x, q.y, q.platform, None), mode=q.mode, how=q.how)
            for i, q in enumerate(places)]


def test_online_from_a_wrong_prior_her_planner_follows_her_walk_and_her_file_starts_the_next_run(tmp_path):
    """Her states as the feed holds them (Feed.since; each taken once, by seq): from v83's 125 and the old 117, four
    walks at 151.25 propose her walk (GROSS), made at settle: a 'calibration' record (old 125, new 151.2, n 4, SPEED
    implied 101.0) and a line said; her file has the walks; a hunter after it starts at her prior (4 of 10 stretches),
    at her measure with 12 (start: loaded)."""
    path = tmp_path / "her_motion.json"
    said = []
    with telemetry.captured() as records:
        c = Calibration(path, 101.0, prior=(125.0, 117.0), log=said.append)
        c.start()
        assert (physics.WALK_SPEED, physics.CLIMB_SPEED) == (125.0, 117.0)
        t = 0.0
        for k in range(calibrate.GROSS_N["walk"]):
            walk = _walk(151.25, 1.4, x0=100.0 * k, platform=23, t0=t)
            states = _states(walk, seq0=1 + 100 * k)
            c.feed_states(states)
            c.feed_states(states)  # (again: taken once)
            t = walk[-1].t + 1.0
            if k < calibrate.GROSS_N["walk"] - 1:
                assert physics.pending() is None
        walk = physics.pending()[0]
        assert c.taken["walk"] == 4 and walk == pytest.approx(151.25, abs=0.06) and physics.pending()[1] == 117.0
        assert physics.settle() and physics.WALK_SPEED == walk
    (rec,) = [r for r in records if r.get("topic") == "calibration" and r.get("what") == "walk" and r["old"] == 125.0]
    assert rec["new"] == walk and rec["n"] == 4 and rec["implied"] == pytest.approx(101.0, abs=0.05)
    assert any(f"her walk 125 -> {walk:g} units/s (measured" in line for line in said)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["walks"]) == 4 and data["walk"]["planned"] == walk
    physics.reset()
    said.clear()
    nxt = Calibration(path, 101.0, log=said.append)
    nxt.start()  # (no prior given: her file's 4 walks are within 0.1% of her prior 151.25, not enough alone: MIN_N)
    assert physics.WALK_SPEED == 151.25 and "4 of 10 stretches measured" in said[0]
    nxt.rows["walk"] = nxt.rows["walk"] * 3  # (12 of them: measured, planned as measured)
    nxt.start()
    assert abs(physics.WALK_SPEED - 151.25) <= 0.05 + 1e-9 and "her walk measured" in said[-1]


def test_a_new_run_of_the_perception_service_counts_its_frames_anew():
    c = Calibration(None, 101.0, log=lambda _: None)
    walk = _walk(151.25, 1.4, platform=23)
    c.feed_states(_states(walk[:10], seq0=500, epoch=1))
    c.feed_states(_states(walk[10:], seq0=1, epoch=2))  # (its frames from 1 again: taken, the stretch cut there)
    assert c.seq == len(walk) - 10 and c.epoch == 2


def test_seeding_her_file_from_past_runs_merges_and_writes_nothing_else(tmp_path):
    """walkfit --seed: her calibration file from a run's perception records, each stretch with its map, when and the
    SPEED given; seeded again, nothing twice."""
    from gamelab.analysis import walkfit
    to = tmp_path / "seed.json"
    path, got = walkfit.seed([Path(__file__).parent / "fixtures" / "records" / "hour-cut"], "magician", to, speed=100.0)
    assert path == to and got["added"]["walk"] == 5 and got["added"]["climb"] == 4  # (the scenes' whole file)
    data = json.loads(to.read_text(encoding="utf-8"))
    assert data["walks"][0][1] == 10002073 and data["walks"][0][-1] == 100.0 and data["seeded"]["speed"] == 100.0
    _, again = walkfit.seed([Path(__file__).parent / "fixtures" / "records" / "hour-cut"], "magician", to, speed=100.0)
    assert again["added"] == {"walk": 0, "climb": 0} and len(json.loads(to.read_text(encoding="utf-8"))["walks"]) == 5
    assert sorted(p.name for p in tmp_path.iterdir()) == ["seed.json"]
