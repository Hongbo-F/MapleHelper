"""The in-game probes (behavior.probe, `python -m gamelab probe`), offline only: what a probe throws at, what
it reads from the looks after a throw, its scorer against simulated worlds whose answers are known, and that
it runs the way `hunt kill` does (the player, the Lie Detector alarm). Nothing here touches the game: the
runs are the simulator's (FakeEyes, FakePlayer), and tests/conftest.py refuses any key or window call."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from maplehelper.engine import RIGHT

from gamelab.attack import reach
from gamelab.behavior import probe
from gamelab.behavior.probe import Probe, choose, outcomes, shown
from gamelab.control import alarm, client
from gamelab.localization.monsters import bodies
from gamelab.sim import scenarios
from gamelab.sim import world as world_mod
from gamelab.sim.world import Noise

from test_attack import PLAT, step_map, target

REAL = pytest.mark.skipif(not scenarios.available(), reason="needs the map's game data (gamedata/data)")

# From x 500 on PLAT (test_attack's: y 500, 0 to 1000), throwing right at pigs on platform 1, 110-200 off.
TOY = Probe("toy", "a test", 0, 500.0, RIGHT, (1,), ("pig",), (110, 200))


def scene(*monsters, x=500.0, y=500.0, platform=PLAT):
    return SimpleNamespace(where=SimpleNamespace(x=x, y=y), platform=platform, monsters=list(monsters))


# --- what a probe throws at ----------------------------------------------------------------------------

def test_a_probe_throws_only_at_a_lone_monster_in_range():
    pig = target(650, 560, platform=1)  # 150 off, 60 below
    assert choose(scene(pig), TOY) is pig
    # Another in front within 420 (any platform, any kind): no throw, unless the crowd allows it.
    far = target(800, 560, platform=1)
    assert choose(scene(pig, far), TOY) is None
    assert choose(scene(pig, far), TOY, crowd=1) is pig
    assert choose(scene(pig, target(600, kind="snail")), TOY) is None
    # Not in range, not its kind, not on its platforms: no throw.
    assert choose(scene(target(750, 560, platform=1)), TOY) is None
    assert choose(scene(target(650, 560, platform=1, kind="slime")), TOY) is None
    assert choose(scene(target(650, 500, platform=0)), TOY) is None
    # Behind her, past FRONT, or far below or above her: not in front, so it doesn't count.
    for other in (target(400), target(500 + probe.FRONT + 10, 560, platform=1), target(700, 500 + 160, platform=3)):
        assert choose(scene(pig, other), TOY) is pig


def test_who_counts_as_in_front_keeps_to_the_wall_switch(monkeypatch):
    # The step map: she's on A below the step wall of her own group, the pig up on B behind it. While her
    # decisions keep to the rule, the pig can't take a star and isn't counted; with the rule off (the probes
    # that measure it), it's thrown at.
    model, a, b, _ = step_map()
    p = Probe("toy", "a test", a.id, 100.0, RIGHT, (b.id,), ("pig",), (150, 300))
    s = scene(target(320, 440, platform=b.id), x=100.0, y=500.0, platform=a)
    monkeypatch.setattr(reach, "WALLS_BLOCK", True)
    assert choose(s, p) is None
    monkeypatch.setattr(reach, "WALLS_BLOCK", False)
    assert choose(s, p) is s.monsters[0]


def test_a_probe_s_place_is_checked_before_it_runs():
    model, a, b, _ = step_map()
    ok = Probe("toy", "a test", a.id, 100.0, RIGHT, (b.id,), ("pig",), (150, 300))
    assert probe.problems(model, ok) == []
    assert "isn't on platform" in probe.problems(model, probe.adjust(ok, x=300.0))[0]
    # A jump back needs room behind her: she jumps away from the target.
    assert "a jump back needs" in probe.problems(model, probe.adjust(ok, x=50.0, style="jump_back"))[0]
    assert probe.problems(model, probe.adjust(ok, x=150.0, style="jump_back")) == []
    assert "no such way" in probe.problems(model, probe.adjust(ok, style="lucky_seven"))[0]


@REAL
def test_every_probe_can_run_from_its_place_on_the_real_map():
    model = scenarios.real_map()
    for p in probe.PROBES.values():
        assert probe.problems(model, p) == [], p.name
        here = model.platform(p.platform)
        assert here.left + reach.EDGE_MARGIN <= p.x <= here.right - reach.EDGE_MARGIN, p.name
        # Its targets' platforms are that way, and reach into its range sideways.
        for kind in p.kinds:
            lo, hi = p.range_of(kind)
            sign = 1 if p.side == RIGHT else -1
            assert any(lo <= (x - p.x) * sign <= hi for q in p.at for x in range(int(model.platform(q).left),
                                                                                  int(model.platform(q).right))), p.name
    # The research's other places and ways to throw.
    for name, changes in (("step-up", {"x": 260.0}), ("step-up", {"style": "jump_throw"}),
                          ("step-up", {"style": "jump_back"}), ("step-up", {"x": 260.0, "style": "jump_back"}),
                          ("step-down", {"x": 195.0, "dx": (160, 250)})):
        assert probe.problems(model, probe.adjust(probe.PROBES[name], **changes)) == [], (name, changes)


# --- what a look after a throw shows -------------------------------------------------------------------

def entry(before=None, after=(), unseen=(), gone=False, target=True):
    return {"kind": "pig", "x": 0.0, "y": 0.0, "platform": 1, "target": target, "at_pick": True,
            "bar_before": before, "after": [list(a) for a in after], "unseen": list(unseen), "gone": gone}


def test_a_hit_is_a_bar_new_or_lower_and_a_kill_her_exp_with_it_gone():
    assert shown(entry(after=[(0.1, None), (0.2, 0.67)]), 0, False) is True       # its bar shows up
    assert shown(entry(0.67, [(0.1, 0.67), (0.3, 0.33)]), 0, False) is True       # ... or drops
    assert shown(entry(0.67, [(0.1, 0.67), (0.4, 0.67)]), 0, False) is False      # as before, settled
    assert shown(entry(None, [(-0.1, None), (0.5, None)]), 0, False) is False
    assert shown(entry(None, [(0.1, None)], unseen=[0.4, 0.9]), 0, False) is None  # hidden: can't tell
    # Killed: her EXP up, and it gone (seen fading out first, without its bar).
    assert shown(entry(None, [(0.2, None), (0.5, None)], gone=True), 1, True) is True
    assert shown(entry(None, [(0.2, None), (0.5, None)], gone=True), 0, True) is False
    row = {"exp": 1, "evidence": [entry(None, [(0.5, None)], gone=True), entry(None, [(0.5, None)], gone=True,
                                                                                  target=False)]}
    assert [s for _, s in outcomes(row)] == [False, False]  # two gone: the kill can't be placed


# --- runs in the simulator, and the scorer ---------------------------------------------------------------

def _rows(trace: dict) -> list[dict]:
    return [json.loads(line) for line in trace["files"]["probe_geometry.jsonl"]]


@REAL
def test_the_scorer_finds_the_window_top_the_world_has(monkeypatch):
    # From 13 at a lone pig on 12, 60 above her, 110-200 off: with the window's top at 21 the star picks it
    # only from 191, with meowdb's 44 from 111. Each world's log fits its own top and rules out the other. With
    # no hit at all the fit would be flat down to 0, not bounding the top from below (the scenario expects one).
    model = scenarios.real_map()
    rows = _rows(scenarios.run("probe_top"))
    low = probe.fit_top0(rows, model)
    assert low.holds(21) and not low.holds(44) and low.fits[0] >= 16 and low.fits[1] <= 30
    assert "SEARCH_TOP0: fits" in probe.score(rows, model)
    monkeypatch.setattr(reach, "SEARCH_TOP0", 44)
    high = probe.fit_top0(_rows(scenarios.run("probe_top")), model)
    monkeypatch.undo()
    assert high.holds(44) and not high.holds(21) and high.fits[0] >= 35


@REAL
def test_the_scorer_finds_the_window_floor():
    # From 14's right end at an Orange Mushroom on 17, 60 below her, 90-150 off: with the window's floor 12 above
    # her feet in the first band, its body (55 tall) is picked from 113; with 7 from 93, with 17 from 133.
    def run(sim):
        h = sim.hunter()
        return str(probe.run(h, sim.navigator(h), probe.adjust(probe.PROBES["floor"], throws=20), seconds=300))
    sc = scenarios.Scenario("probe_floor", run, (14, 824.0), monsters=[("Orange Mushroom", 17, 940.0)], limit=400)
    fit = probe.fit_floor0(_rows(scenarios.run_scenario(sc)), scenarios.real_map())
    assert fit.holds(12) and not fit.holds(5) and not fit.holds(20)


@REAL
def test_the_scorer_checks_the_wall_rule_and_its_line(monkeypatch):
    # From 4 at an Orange Mushroom on 2 behind the step wall: the world's stars stop at the wall from the
    # ground (the line 28 up), not from jump backs (44 up when the attack starts: 72). The rule fits that
    # better than no walls, and the line's height fits from 16 (under it, jump backs would be stopped too) to
    # 60 (over it, the throws from the ground would get past).
    model = scenarios.real_map()
    rows = _rows(scenarios.run("probe_step_up"))
    lls = probe.terrain_lls(rows, model)
    rule, none = lls["her group's walls, a level line (the rule)"], lls["no wall stops a star"]
    assert rule > none + 5
    line = probe.fit_shot_h(rows, model)
    assert line.holds(28) and not line.holds(10) and not line.holds(65)
    # A world where no wall stops a star: the throws from the ground hit too (all 12 on seed 1; one whose Orange
    # Mushroom had walked to within 193 of her by the pick, where the search doesn't take it yet, may not), and no
    # walls fits better.
    open_world = SimpleNamespace(**{k: getattr(reach, k) for k in dir(reach) if not k.startswith("__")})
    open_world.wall_between = lambda *a, **k: None
    monkeypatch.setattr(world_mod, "reach", open_world)
    rows = _rows(scenarios.run("probe_step_up"))
    monkeypatch.undo()
    assert [s for r in rows for e, s in outcomes(r) if e["target"]].count(True) >= 10
    lls = probe.terrain_lls(rows, model)
    assert lls["no wall stops a star"] > lls["her group's walls, a level line (the rule)"] + 5


@REAL
def test_the_scorer_tells_the_area_of_an_attack_from_a_jump_from_the_search():
    # Jump throws at a snail about 40 units off: the world's attack picks by the area, and the snail dies to
    # every star; the search (from 65 ahead, as from the ground or from as high as she is) wouldn't pick it.
    # She takes her place facing the way she throws, so no jump throw turns her (a turn's tap walked her 8 units
    # on, to 32 off where the snail came back, and it touched her) and each goes straight up from x 30: the
    # snail is 40 off at all but the first, before which it walked on to 67 off (inside the probe's 35-85) while
    # she took her place.
    rows = _rows(scenarios.run("probe_air"))
    assert {r["x"] for r in rows} == {30.0} and all(35 <= abs(r["dx"]) <= 85 for r in rows)
    assert sum(35 <= abs(r["dx"]) <= 50 for r in rows) >= len(rows) - 1
    lls = probe.air_lls(rows, scenarios.real_map())
    assert lls["area"] > max(lls["search"], lls["ground"]) + 5


def _probe_run(name: str, start: tuple, monsters: list, throws: int = 20) -> list[dict]:
    """A simulated run of probe `name` (its log's rows), from `start` with `monsters` placed."""
    def run(sim):
        h = sim.hunter()
        return str(probe.run(h, sim.navigator(h), probe.adjust(probe.PROBES[name], throws=throws), seconds=400))
    return _rows(scenarios.run_scenario(scenarios.Scenario(f"probe_{name}", run, start, monsters=monsters, limit=500)))


@REAL
def test_the_scorer_tells_the_search_from_the_old_ramp_42_below_her(monkeypatch):
    # From 0 at a lone pig on 1, 42 below her, 70-160 off: the world's search picks it from 71 by its middle; the old
    # ramp (reach.RAMP_BELOW, what her decisions keep to there) from 126. Each world's log fits its own: the
    # search's by 19.9 (log-likelihood), the ramp's by 4.5 (9.3 with pigs walking 70 units/s, not the game data's
    # 125: they walk further between her look and the pick).
    model = scenarios.real_map()
    run = lambda: _probe_run("below-42", (0, 1222.0), [("Pig", 1, 1110.0)])  # noqa: E731
    rows = run()
    assert len(probe.below(rows)) == 20
    lls = probe.ramp_lls(rows, model)
    assert lls["the search"] > lls["the search, the old ramp below her"] + 5
    assert "the search, the old ramp below her" in probe.score(rows, model)
    ramp_world = SimpleNamespace(**{k: getattr(reach, k) for k in dir(reach) if not k.startswith("__")})

    def search_band(near, far, bottom, top, *a, **k):  # none below her closer by its middle than the ramp
        return None if (near + far) / 2 < reach.ramp_dx(-bottom) else reach.search_band(near, far, bottom, top, *a, **k)
    ramp_world.search_band = search_band
    monkeypatch.setattr(world_mod, "reach", ramp_world)
    rows = run()
    monkeypatch.undo()
    lls = probe.ramp_lls(rows, model)
    assert lls["the search, the old ramp below her"] > lls["the search"] + 4


@REAL
def test_the_scorer_tells_the_claw_s_swipes_from_stars_by_the_bar_they_take():
    # On 21 at a lone pig, then a lone slime, walking at her 95-160 off: the world's claw swipes a body that reaches
    # into its 90 units, a pig out to 124 by its middle and a slime to 118, for a fifth of a star's damage (a pig's
    # bar by 7%, a slime's by 10%); stars from there. Down at 4, 60 below her, the world's claw (on her level only)
    # swipes nothing.
    for kind, edge in (("Pig", 124), ("Slime", 118.5)):
        table = probe.claw_table(_probe_run("claw-edge", (21, -225.0), [(kind, 21, -90.0)]))
        claw = [lo for (_, _, dy, lo), g in table.items() if g["claw"]]
        star = [lo for (_, _, dy, lo), g in table.items() if g["star"]]  # a kill can be either
        assert claw and star and max(claw) < edge <= max(star) + 10 and min(star) >= edge - 10, (kind, table)
    rows = _probe_run("claw-down", (3, 716.0), [("Pig", 4, 620.0)])
    table = probe.claw_table(rows)
    assert sum(g["throws"] for g in table.values()) == 20 and not any(g["claw"] for g in table.values())
    assert all(dy == 60 for _, _, dy, _ in table) and "the claw's" in probe.score(rows, scenarios.real_map())


@REAL
def test_a_hidden_target_is_waited_for_and_never_read_as_a_miss(monkeypatch):
    # The game's hit monsters hide under their damage numbers for 1-3 s: the evidence waits up to
    # HIDDEN_MAX_S for the target to show again. With meowdb's top in the world, a star picks the pig from 111
    # units off: none of the throws its attack took may be read as a miss (seen again with its bar unchanged),
    # and most are read as hits (19 of the 30; 22 with pigs walking 70 units/s, not the game data's 125). Throws
    # at a pig that walked out of the search's window by the pick (one 112 off that walked to 100) the world's
    # star missed, and so they're read.
    monkeypatch.setattr(reach, "SEARCH_TOP0", 44)
    sc = dataclasses.replace(scenarios.SCENARIOS["probe_top"], noise=Noise(hit_hidden=(1.0, 3.0)))
    trace = scenarios.run_scenario(sc)
    model = scenarios.real_map()
    body = bodies(model)
    told = [(r, s) for r in _rows(trace) for e, s in outcomes(r) if e["target"]]
    assert len(told) == 30 and [s for _, s in told].count(True) > len(told) / 2
    assert all(probe.taken(r, model, body, top0=44) is None for r, s in told if s is False)
    assert any(r["looks"] > 12 for r in _rows(trace))  # looked on past EVIDENCE_S for a hidden one


@REAL
def test_a_probe_throws_with_the_regular_attack_and_puts_the_wall_switch_back():
    # Even for a hunter set to throw Lucky Seven (MP), a probe throws the regular attack: what every log
    # measured. The wall rule is the probe's for the run (off for step-up) and put back after it.
    keys = []

    def run(sim):
        h = sim.hunter(skill=True)
        probe.run(h, sim.navigator(h), probe.adjust(probe.PROBES["step-up"], x=260.0, throws=2), seconds=60)
        keys.append(reach.WALLS_BLOCK)
        return "done"
    sc = scenarios.Scenario("probe_keys", run, (4, 260.0), monsters=[("Orange Mushroom", 2, -50.0)], limit=200)
    trace = scenarios.run_scenario(sc)
    sim_keys = scenarios.her_setup()[1]
    attack_keys = {e[1] for p in trace["plans"] if p[1] == "throw" for e in p[2]} - {sim_keys.left.name}
    assert attack_keys == {sim_keys.attack.name} and keys == [True]


def test_a_probe_waiting_for_a_lone_monster_stops_once_the_alarm_is_up(alarm_file, monkeypatch):
    # Waiting for a lone monster in range she sends no key, only looks, and a look brings the game to the front:
    # once the player has seen the Lie Detector (the alarm raised), the next pass stops the run, the alarm passing
    # on to the command line (exit code 3), and the wall switch is put back.
    import time
    model, a, b, _ = step_map()
    p = Probe("toy", "a test", a.id, 100.0, RIGHT, (b.id,), ("pig",), (150, 300), walls=False)
    looks = []

    def look(**kw):
        looks.append(1)
        if len(looks) == 3:  # the player raises it between two looks
            alarm_file.write_text(json.dumps({"reason": "lie_detector", "time": "now"}), encoding="utf-8")
        return scene(x=100.0, platform=a)
    h = SimpleNamespace(model=model, actions=SimpleNamespace(skill=True, facing=RIGHT), log=lambda m: None, look=look,
                        health_check=lambda s: None, knocked=lambda s: False)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(reach, "WALLS_BLOCK", True)
    with pytest.raises(client.LieDetectorAlarm):
        probe.run(h, None, p)
    assert len(looks) == 3 and reach.WALLS_BLOCK is True and h.actions.skill is False


# --- the command line ----------------------------------------------------------------------------------

@pytest.fixture
def alarm_file(tmp_path, monkeypatch):
    path = tmp_path / "ALARM.json"
    monkeypatch.setattr(alarm, "ALARM", path)
    monkeypatch.setattr(client, "ALARM", path)
    return path


def test_probe_run_is_refused_while_the_alarm_is_up_and_list_and_score_are_not(alarm_file, tmp_path, capsys):
    from gamelab.__main__ import LIE_DETECTOR_EXIT, build_parser, main
    alarm_file.write_text(json.dumps({"reason": "lie_detector", "time": "now"}), encoding="utf-8")
    # Refused before anything else: building the hunter, or bringing the game to the front (which the tests'
    # guard would fail on).
    assert main(["probe", "run", "top-60"]) == LIE_DETECTOR_EXIT
    assert main(["probe", "list"]) == 0 and "top-60" in capsys.readouterr().out
    log = tmp_path / "probe_geometry.jsonl"
    assert main(["probe", "score", str(log)]) == 0 and "no probe throws" in capsys.readouterr().out
    args = build_parser().parse_args(["probe", "run", "step-up", "--dx", "110-200", "--style", "jump_back",
                                      "--walls", "on"])
    assert (args.dx, args.style, args.walls, args.game) == ((110.0, 200.0), "jump_back", "on", True)
    assert not getattr(build_parser().parse_args(["probe", "score"]), "game", False)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["probe", "run", "top-60", "--dx", "200-110"])


@REAL
def test_probe_score_reads_a_log(tmp_path, capsys):
    from gamelab.__main__ import main
    log = tmp_path / "probe_geometry.jsonl"
    rows = _rows(scenarios.run("probe_step_up"))
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert main(["probe", "score", str(log)]) == 0
    out = capsys.readouterr().out
    assert "12 probe throws" in out and "SHOT_H: fits 16 to 60" in out and "step-up jump_back x" in out
