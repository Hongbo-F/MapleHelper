"""Her walk and climb from her stat window's SPEED by the client's formula (motion.physics: walk_of, climb_of, set_speed,
propose, settle; behavior.hunter: _speed_start, _speed_read; behavior.statcheck.speed_doubt). The user, 2026-11-02, shown
that the formula and her measured walks and climbs agree: "Just use panel, no need to measure." So: her profile's SPEED
at import, her stat window's as her stats file kept it at a hunter's start, a read at a hunt's start at once, a read put
off into the hunt loop where no chain of hers runs; a misread not taken; one line and one 'motion' record a change."""

import dataclasses
import sys

import pytest

from gamelab import telemetry
from gamelab.behavior.bootstrap import HerStats
from gamelab.behavior.hunter import Hunter
from gamelab.behavior.statcheck import SPEED_STEP, speed_doubt
from gamelab.classes import magician, thief
from gamelab.motion import body, physics
from gamelab.sim import harness, scenarios
from gamelab.sim.world import LEVEL_25, Realism

REAL = pytest.mark.skipif(not scenarios.available(scenarios.TREE), reason="needs Tree Dungeon's game data (gamedata/data)")
PROFILE = (physics.WALK_SPEED, physics.CLIMB_SPEED)  # (the thief's, as the tests play him: 151.25 and 121)
AT_IMPORT = {n: getattr(physics, n) for n in physics.WALK_NAMES}


# --- the formula and her profiles --------------------------------------------------------------------------------

def test_her_walk_and_climb_are_the_clients_formula_at_her_speed():
    """125 x (SPEED + 20)/100 and SPEED + 20 (meowdb's guide and calculator: the 20 this client adds), SPEED within the
    client's 80-140 first; the thief's profile 101 (his stat window's) and the magician's 100 (hers)."""
    assert [physics.walk_of(s) for s in (100, 101, 104, 140, 150, 70)] == [150.0, 151.25, 155.0, 200.0, 200.0, 125.0]
    assert [physics.climb_of(s) for s in (100, 101, 104, 140, 150, 70)] == [120.0, 121.0, 124.0, 160.0, 160.0, 100.0]
    assert physics.speed_of_walk(151.25) == pytest.approx(101.0) and physics.speed_of_climb(119.9) == pytest.approx(99.9)
    assert (thief.FIXED_STATS.speed, magician.FIXED_STATS.speed) == (101.0, 100.0)
    assert PROFILE == (151.25, 121.0) == (physics.PROFILE_SHOE.vmax, physics.PROFILE_CLIMB) and physics.SPEED == 101.0
    assert physics.shoe_of(magician.FIXED_STATS).vmax == 150.0 and physics.climb_of(magician.FIXED_STATS.speed) == 120.0
    from gamelab.sim import world
    assert (world.MAGE_WALK, world.MAGE_CLIMB) == (150.0, 120.0)  # (the simulator's magician: her profile's formula)


# --- the planner's walk set, and settled where no chain runs -----------------------------------------------------

def test_set_speed_makes_every_name_of_her_walk_again_wherever_it_was_taken():
    """physics.set_speed makes every name of her walk again in every module that took one (`from .physics import
    WALK_SPEED`), the modules' own of them (follows), counts EPOCH on; the simulator's world walks her at her profile's
    all the same; reset puts every name back as imported (the expressions at each, _derive's)."""
    from gamelab.control import actions, mover, navigator
    from gamelab.motion import flow, route
    from gamelab.sim import world
    epoch = physics.EPOCH
    took = [(mod, n) for name, mod in list(sys.modules.items()) if name.startswith("gamelab.") and mod is not physics
            for n in physics.WALK_NAMES if vars(mod).get(n) is getattr(physics, n)]
    assert {(m.__name__, n) for m, n in took} >= {("gamelab.behavior.hunt", "WALK_SPEED"),
                                                  ("gamelab.control.mover", "CLIMB_SPEED"),
                                                  ("gamelab.motion.flow", "SHOE"), ("gamelab.motion.flow", "STOP_RUN")}
    assert physics.set_speed(80.0) and physics.EPOCH == epoch + 1
    assert (physics.WALK_SPEED, physics.CLIMB_SPEED, physics.SPEED) == (125.0, 100.0, 80.0)
    assert physics.SHOE.vmax == 125.0 and physics.SHOE.vjump == physics.PROFILE_SHOE.vjump
    for mod, n in took:  # (every module that took a name of her walk: its new value)
        assert vars(mod)[n] is getattr(physics, n), (mod.__name__, n)
    assert mover.WALK_SPEED == navigator.WALK_SPEED == flow.WALK_SPEED == 125.0 and route.CLIMB_SPEED == 100.0
    assert flow.ROPE_SPEED == navigator.ROPE_SPEED == 100.0 and navigator.KEY_LEAD["walk_off"] == physics.STOP_S
    assert flow.GRAB_LEAD == route.GRAB_LEAD == pytest.approx(125.0 * physics.JUMP_SPEED / physics.GRAVITY)
    assert route.ROPE_VX == pytest.approx(1.3 * 125.0) and route.ROPE_JUMP_SLIDE == body.slide(route.ROPE_VX,
                                                                                                physics.SHOE)
    assert actions.JB_RUN_MS == physics.BLIND_ENDS_MS == 60  # (the blind spot of three steps ends sooner at 125)
    assert physics.RUN_LAG == pytest.approx(physics.LAGS.on + 125.0 / 2800)
    assert world.her_shoe(world.Her()) is physics.PROFILE_SHOE and world.her_climb(world.Her()) == PROFILE[1]
    assert not physics.set_speed(80.0) and physics.EPOCH == epoch + 1  # (the same: nothing made again)
    assert physics.reset() and physics.SPEED == 101.0 and physics.SHOE is physics.PROFILE_SHOE
    assert {n: getattr(physics, n) for n in physics.WALK_NAMES} == AT_IMPORT
    assert flow.ROPE_SPEED == navigator.ROPE_SPEED == PROFILE[1] and actions.JB_RUN_MS == 150


def test_a_change_proposed_waits_for_settle_and_is_told_once_made():
    """physics.propose holds a SPEED until settle (where no chain of hers runs: a navigator's pass, the hunt loop's),
    told then (old, new: walk, climb, SPEED); one proposed that changes nothing is told nothing."""
    told = []
    physics.propose(104.0, then=lambda old, new: told.append((old, new)))
    assert (physics.WALK_SPEED, physics.SPEED) == (PROFILE[0], 101.0) and not told  # (not before settle)
    assert physics.settle() and told == [((151.25, 121.0, 101.0), (155.0, 124.0, 104.0))]
    assert not physics.settle()  # (nothing proposed)
    physics.propose(104.0, then=lambda old, new: told.append((old, new)))
    assert not physics.settle() and len(told) == 1


def test_the_simulators_planner_can_be_held_apart_from_her_stat_windows_speed():
    """physics.set_planned (the simulator's `--world planner-v83`, world.Her.planned): her planner at v83's walk 125 (the
    client's shoe at SPEED 80, v83's at 100 without the client's Speed, as master planned her) and the old climb 117
    whatever her SPEED, every name of her walk made again as set_speed makes it, SPEED kept; reset back to the formula's."""
    from gamelab.motion import flow
    epoch = physics.EPOCH
    assert physics.set_planned(125.0, 117.0) and physics.EPOCH == epoch + 1
    assert (physics.WALK_SPEED, physics.CLIMB_SPEED, physics.SPEED, flow.ROPE_SPEED) == (125.0, 117.0, 101.0, 117.0)
    assert physics.SHOE == body.Shoe.from_physics(body.PHYSICS_IMG, 100.0, physics.CHARACTER.jump)  # (v83's, no bonus)
    assert not physics.set_planned(125.0, 117.0)
    assert physics.reset() and (physics.WALK_SPEED, physics.CLIMB_SPEED) == PROFILE and physics.EPOCH == epoch + 2


@REAL
def test_the_harness_planner_v83_holds_her_planner_and_the_world_walks_her_at_her_own():
    """`--world planner-v83` (world.Her.planned): after her hunter's start her planner at 125 and 117 (Sim.hunter, said in
    her log), the world walking her at her own (her truth, the thief's 151.25 and 121); her profile's again at the run's
    end (Sim's exit)."""
    from gamelab.sim import world
    assert harness.WORLDS["planner-v83"] == {"planned": (125.0, 117.0)} and "planned" in harness.HER_FIELDS
    got = {}

    def fn(sim):
        sim.hunter()
        got["planned"] = (physics.WALK_SPEED, physics.CLIMB_SPEED)
        got["world"] = (world.her_shoe(sim.world.character).vmax, world.her_climb(sim.world.character))
        got["log"] = list(sim.trace.log)
        return {}
    sc = harness.scenario("loop", 1, "game", 0.05, map_id=harness.TREE, world=("planner-v83",))
    scenarios.run_sim(dataclasses.replace(sc, fn=fn))
    assert got["planned"] == (125.0, 117.0) and got["world"] == PROFILE
    assert any(line.endswith("her planner held at walk 125 and climb 117 units/s, the world's own 151.25 and 121 (the "
                             "world's Her.planned)") for line in got["log"])
    assert (physics.WALK_SPEED, physics.CLIMB_SPEED) == PROFILE  # (put back at the run's end)


@REAL
def test_a_navigator_makes_its_route_graph_again_at_its_next_pass_and_a_move_planned_keeps_its_time():
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
    physics.propose(90.0)  # (137.5 and 110)
    assert nav.planner is before  # (not before the pass)
    nav._follow_walk()
    assert nav.planner is not before and nav._siblings is None and physics.CLIMB_SPEED == 110.0
    again = next(m for m in nav.planner.moves if (m.kind, m.src, m.dst, m.x) == (climb.kind, climb.src, climb.dst,
                                                                              climb.x))
    assert climb.seconds == seconds and again.seconds > seconds  # (the one planned before keeps its time)


# --- a read checked, and her hunter's start ----------------------------------------------------------------------

def test_a_speed_outside_the_clients_range_or_a_jump_past_gear_is_no_read():
    """statcheck.speed_doubt: within 80-140, and SPEED_STEP (10) or less from what's planned or from her last read kept;
    so a misread (101 read 131) isn't taken, and a gear change past SPEED_STEP is once a second read agrees."""
    assert SPEED_STEP == 10.0 and body.SPEED_RANGE == (80.0, 140.0)
    assert speed_doubt(104.0, 101.0) is None and speed_doubt(111.0, 101.0) is None and speed_doubt(91.0, 101.0) is None
    assert speed_doubt(160.0, 101.0) == "outside the client's 80-140" and speed_doubt(70.0, 101.0) is not None
    assert speed_doubt(131.0, 101.0) == "+30 from the 101 planned: more than gear gives between two reads"
    assert speed_doubt(131.0, 101.0, 101.0) == speed_doubt(131.0, 101.0)
    assert speed_doubt(131.0, 101.0, 131.0) is None  # (read twice the same: taken)
    assert speed_doubt(125.0, 101.0, 131.0) is None and speed_doubt(116.0, 101.0, 131.0) == (
        "+15 from the 101 planned and -15 from her last read, 131: more than gear gives between two reads")


def _hunter(tmp_path, window: dict | None = None):
    """A hunter as far as her walk and climb go (Hunter's _speed_* on her stats file at `tmp_path`, which keeps her stat
    window's read `window`: {"read": {...}, "speed": ...}, None: none), its log; her start (_speed_start) made."""
    path = tmp_path / "her_stats.json"
    if window is not None:
        st = HerStats(path)
        st.see_window(111.0, level=30, read=window.get("read"), speed=window.get("speed"))
    h = Hunter.__new__(Hunter)
    h.stats, h.said = HerStats(path), []
    h.log = h.said.append
    h.speed_plan = (101.0, "profile")
    with telemetry.captured() as records:
        h._speed_start()
    h.records = [r for r in records if r.get("topic") == "motion"]
    return h


def test_her_start_is_her_stat_windows_speed_as_kept_else_her_profiles(tmp_path):
    """At her hunter's start, at once: the SPEED her stats file kept from her last read ('kept'), one line and one
    'motion' record; with none her profile's, nothing said (nothing changed)."""
    h = _hunter(tmp_path / "none")
    assert h.speed_plan == (101.0, "profile") and (physics.WALK_SPEED, physics.CLIMB_SPEED) == PROFILE
    assert h.said == [] and h.records == []
    h = _hunter(tmp_path / "kept", {"read": {"speed": 104, "jump": 100}, "speed": 104.0})
    assert h.speed_plan == (104.0, "kept") and (physics.WALK_SPEED, physics.CLIMB_SPEED, physics.SPEED) == (
        155.0, 124.0, 104.0)
    assert h.said == ["her walk 151.25 -> 155 and climb 121 -> 124 units/s: SPEED 104, her stat window's as her stats "
                      "file kept it, by the client's formula"]
    (rec,) = h.records
    assert (rec["old"], rec["new"], rec["speed"], rec["was"], rec["why"]) == ([151.25, 121.0], [155.0, 124.0], 104.0,
                                                                             101.0, "kept")
    physics.reset()
    h = _hunter(tmp_path / "broken", {"read": {"speed": 104}, "speed": 400.0})  # (a file edited by hand, say)
    assert h.speed_plan == (101.0, "profile") and h.said[0].startswith(
        "her stat window's SPEED 400 as her stats file kept it not taken (outside the client's 80-140)")


def test_a_stats_file_from_before_the_panel_rule_has_its_reads_speed_judged_as_a_read(tmp_path):
    """Her stats file of before 2026-11-02 keeps the read's SPEED alone (none taken): taken as a read is against her
    profile's (statcheck.speed_doubt), else her profile's, said."""
    h = _hunter(tmp_path / "old", {"read": {"speed": 104}})
    assert h.speed_plan == (104.0, "kept") and physics.WALK_SPEED == 155.0
    physics.reset()
    h = _hunter(tmp_path / "misread", {"read": {"speed": 131}})
    assert h.speed_plan == (101.0, "profile") and physics.WALK_SPEED == PROFILE[0]
    assert h.said == ["her stat window's SPEED 131 as her stats file kept it not taken (+30 from the 101 planned: more "
                      "than gear gives between two reads): her walk and climb her profile's, SPEED 101"]


def test_a_read_misread_isnt_taken_and_a_change_two_reads_agree_on_is(tmp_path):
    """Hunter._speed_read: a SPEED it takes from the next settle ('panel'), a misread said and not taken (what's planned
    kept, her stats file keeping the read as read and no SPEED taken from it), the same read again taken."""
    h = _hunter(tmp_path)
    assert h._speed_read(None) == [] and h._speed_read(101) == [] and not physics.settle()  # (the same: no change)
    note = h._speed_read(131)
    assert note == ["her stat window's SPEED 131 not taken (+30 from the 101 planned: more than gear gives between two "
                    "reads; a misread?): her walk and climb as they were, SPEED 101"]
    assert h.speed_plan == (101.0, "panel") and not physics.settle() and physics.WALK_SPEED == PROFILE[0]
    h.stats.see_window(111.0, read={"speed": 131})  # (as Hunter._stats_read keeps it: read as read)
    assert h._speed_read(131) == [] and h.speed_plan == (131.0, "panel")
    assert physics.WALK_SPEED == PROFILE[0] and physics.settle()  # (from the next settle)
    assert (physics.WALK_SPEED, physics.CLIMB_SPEED) == (188.75, 151.0)
    assert h.said[-1] == ("her walk 151.25 -> 188.75 and climb 121 -> 151 units/s: SPEED 131, her stat window's, by the "
                          "client's formula")
    assert h._speed_read(150)[0].startswith("her stat window's SPEED 150 not taken (outside the client's 80-140")


# --- in the simulator: a read at a hunt's start, and one put off into the hunt loop ------------------------------

def _shown(sim, speed: float) -> None:
    """Her stat window, as the simulator draws it, shows SPEED `speed` (her profile's otherwise: FakeEyes.stat_values)."""
    for eyes in (sim.eyes, sim.frames):
        eyes.stat_values = (lambda f: lambda: {**f(), "speed": speed})(eyes.stat_values)


@REAL
def test_a_read_at_a_hunts_start_sets_her_walk_and_climb_at_once_and_her_stats_file_keeps_it():
    from test_statcheck import _start
    got = _start(before=lambda sim, h: _shown(sim, 104))
    h = got["h"]
    assert got["out"].what == "read" and h.speed_plan == (104.0, "panel")
    assert (physics.WALK_SPEED, physics.CLIMB_SPEED, physics.SPEED) == (155.0, 124.0, 104.0)  # (at once)
    lines = [line for line in got["log"] if "her walk 151.25 -> 155" in line]
    assert len(lines) == 1 and lines[0].endswith("her walk 151.25 -> 155 and climb 121 -> 124 units/s: SPEED 104, her "
                                                 "stat window's, by the client's formula")
    assert h.stats.window_read["speed"] == 104.0 and h.stats.window_read["read"]["speed"] == 104


@REAL
def test_a_read_put_off_into_the_hunt_loop_changes_her_walk_where_no_chain_of_hers_runs():
    """Put off at the start (monsters near), read at the hunt loop's quiet look (StatsDue), her walk and climb changed
    at its next pass: between her key plans, never while one plays; her navigator's route graph made again."""
    got = {}

    def fn(sim):
        _shown(sim, 104)
        epoch = physics.EPOCH
        got["result"] = scenarios.stat_run(0.5)(sim)
        got["plans"] = [(t, max(max((e[0] for e in events), default=0), end_ms or 0) / 1000)  # (its keys' span)
                        for t, _, events, end_ms in sim.trace.plans]
        got["log"] = list(sim.trace.log)
        got["epoch"] = physics.EPOCH - epoch
        return {}
    scenarios.run_sim(scenarios.Scenario("panel_put_off", fn, (23, 73.0), monsters="all", seed=1, limit=200,
                                         map_id=scenarios.TREE, attack="choice",
                                         her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER),
                                         realism=Realism(zero_walls=True)))
    assert got["result"]["start"][0] == "put off" and got["epoch"] == 1
    t_read = next(float(line.split()[0]) for line in got["log"] if "her stat window: W.DEF" in line)
    (t_set,) = [float(line.split()[0]) for line in got["log"] if "her walk 151.25 -> 155 and climb 121 -> 124" in line]
    assert t_read <= t_set and (physics.WALK_SPEED, physics.CLIMB_SPEED) == (155.0, 124.0)
    playing = [(t, t + d) for t, d in got["plans"] if t < t_set - 0.01 < t + d]
    assert not playing, (t_set, playing)
