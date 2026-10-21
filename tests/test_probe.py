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


def test_a_monster_behind_a_wall_still_counts_as_in_front(monkeypatch):
    # The client checks the line to its pick alone: a monster behind a wall of her group that the attack takes
    # first takes it, and the star stops at the wall. test_attack's shield map: from A at 25, a pig up on B at
    # 215 (band 5, past the step wall) comes before a pig on L at 190, 60 below (band 6): the one on L isn't alone
    # in front, and with a crowd it's still shielded. With the rule off, as before.
    from test_attack import shield_map
    model, a, b, low = shield_map()
    up, under = target(215, 440, platform=b.id), target(190, 560, platform=low.id)
    p = Probe("toy", "a test", a.id, 25.0, RIGHT, (low.id,), ("pig",), (150, 200))
    s = scene(up, under, x=25.0, y=500.0, platform=a)
    monkeypatch.setattr(reach, "WALLS_BLOCK", True)
    assert probe.in_front(s, RIGHT) == [up, under]
    assert choose(s, p) is None and choose(s, p, crowd=1) is None
    monkeypatch.setattr(reach, "WALLS_BLOCK", False)
    assert choose(s, p) is None and choose(s, p, crowd=1) is under


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


def test_one_odd_throw_never_decides_a_question_alone():
    # Throws both answers have the attack take, every one a star's hit (as in the simulator), and one that only
    # one of them has it take, where none showed (a misread bar, a monster walked off where the pick had it):
    # the rates are kept 5% from 0 and 1 (RATE_FLOOR), so it weighs under DECIDES however many hits come with
    # it (at 2%, 3.6 with 19). Nor is a question decided on fewer than DECIDE_THROWS throws they differ on.
    for agree in (10, 19, 50):
        rows = [{"exp": 0, "evidence": [entry(None, [(0.5, 0.6)])]} for _ in range(agree)]
        rows.append({"exp": 0, "evidence": [entry(None, [(0.5, None)])]})

        def took(last: float):
            return lambda r: [(e, 0.0, last if r is rows[-1] else 1.0) for e in r["evidence"]]
        (taken, _), (not_taken, _) = probe.likelihood(rows, took(1.0)), probe.likelihood(rows, took(0.0))
        assert 2.5 < not_taken - taken < probe.DECIDES, agree
    q = probe.QUESTIONS["LINE"]
    lls = {"the level line": 0.0, "the v83 line": -10.0}
    assert probe.Answers(q, lls, 20, probe.DECIDE_THROWS - 1).decided is None
    assert probe.Answers(q, lls, 20, probe.DECIDE_THROWS).decided == "the level line"
    assert "on only 4 throws" in probe.answers_text(probe.Answers(q, lls, 20, 4))[-1]


# --- runs in the simulator, and the scorer ---------------------------------------------------------------

def _rows(trace: dict) -> list[dict]:
    return [json.loads(line) for line in trace["files"]["probe_geometry.jsonl"]]


# Where each probe's simulated run starts, the monster it throws at, and what the run changes of the probe.
PLACES = {
    "top-60": ((13, -14.0), [("Pig", 12, -170.0)], {}),
    "top-120": ((15, 76.0), [("Pig", 12, -300.0)], {}),
    "air": ((22, 30.0), [("Snail", 22, 70.0)], {}),
    "other-wall": ((19, 104.0), [("Pig", 18, 330.0)], {}),
    "step-up": ((4, 260.0), [("Orange Mushroom", 2, -50.0)], {"x": 260.0}),
    "step-down": ((2, 50.0), [("Pig", 4, 280.0)], {}),
    "hanging-wall": ((14, 706.0), [("Pig", 17, 910.0)], {}),
    "below-42": ((0, 1222.0), [("Pig", 1, 1110.0)], {}),
    "claw-edge": ((21, -225.0), [("Pig", 21, -90.0)], {}),
    "claw-down": ((3, 716.0), [("Pig", 4, 620.0)], {"throws": 40}),
}


def _reach_with(**changes) -> SimpleNamespace:
    """attack.reach as the simulator's world sees it, with `changes`."""
    out = SimpleNamespace(**{k: getattr(reach, k) for k in dir(reach) if not k.startswith("__")})
    for k, v in changes.items():
        setattr(out, k, v)
    return out


def _search_by_the_ramp(near, far, bottom, top, *a, **k):  # none below her closer by its middle than the ramp
    return None if (near + far) / 2 < reach.ramp_dx(-bottom) else reach.search_band(near, far, bottom, top, *a, **k)


def _every_wall(platform, x, y, tx, rise=0.0, **k):  # any group's wall across the level line stops a star
    line, (lo, hi) = y - rise - reach.SHOT_H, sorted((x, tx))
    return next((w for w in scenarios.real_map().walls
                 if w.x1 == w.x2 and lo < w.x1 < hi and min(w.y1, w.y2) <= line < max(w.y1, w.y2)), None)


# Worlds with the other answer to a question: what's set for the run.
WORLDS = {
    "the logs' top": lambda mp: mp.setattr(reach, "SEARCH_TOP0", probe.LOGS_TOP0),
    "the old claw": lambda mp: mp.setattr(reach, "CLAW", probe.OLD_CLAW),
    "the swings": lambda mp: mp.setattr(reach, "CLAW", probe.SWINGS),
    "the ramp": lambda mp: mp.setattr(world_mod, "reach", _reach_with(search_band=_search_by_the_ramp)),
    "the v83 line": lambda mp: mp.setattr(reach, "LINE", "v83"),
    "no walls": lambda mp: mp.setattr(world_mod, "reach", _reach_with(wall_between=lambda *a, **k: None)),
    "every wall": lambda mp: mp.setattr(world_mod, "reach", _reach_with(wall_between=_every_wall)),
    "the v83 air": lambda mp: mp.setattr(world_mod, "AIR_PICK", "v83"),
}
_RUNS: dict = {}


def _probe_run(name: str, world: str | None = None, throws: int = 20, **changes) -> list[dict]:
    """A simulated run of probe `name` from its PLACES, in reach's world or one of WORLDS (its log's rows; each
    run once)."""
    start, monsters, place = PLACES[name]
    p = probe.adjust(probe.PROBES[name], **{"throws": throws, **place, **changes})
    key = (p, world)
    if key not in _RUNS:
        def run(sim):
            h = sim.hunter()
            return str(probe.run(h, sim.navigator(h), p, seconds=900))
        with pytest.MonkeyPatch.context() as mp:
            if world is not None:
                WORLDS[world](mp)
            _RUNS[key] = _rows(scenarios.run_scenario(
                scenarios.Scenario(f"probe_{name}", run, start, monsters=monsters, limit=1100)))
    return _RUNS[key]


# Each probe in reach's world and in one with the other answer to its question: the answer the scorer must
# decide on (DECIDES better than the other), its index in the question's answers (0: reach's).
RECOVERS = [
    ("top-60", None, "SEARCH_TOP0", 0), ("top-60", "the logs' top", "SEARCH_TOP0", 1),
    ("top-120", None, "SEARCH_TOP0", 0), ("top-120", "the logs' top", "SEARCH_TOP0", 1),
    ("claw-edge", None, "CLAW_FAR", 0), ("claw-edge", "the old claw", "CLAW_FAR", 1),
    ("claw-down", None, "claw_takes", 0), ("claw-down", "the swings", "claw_takes", 1),
    ("below-42", None, "RAMP_BELOW", 0), ("below-42", "the ramp", "RAMP_BELOW", 1),
    ("step-down", None, "LINE", 0), ("step-down", "the v83 line", "LINE", 1),
    ("hanging-wall", None, "LINE", 0), ("hanging-wall", "the v83 line", "LINE", 1),
    ("step-up", None, "WALLS", 0), ("step-up", "no walls", "WALLS", 1),
    ("other-wall", None, "OTHER_WALLS", 0), ("other-wall", "every wall", "OTHER_WALLS", 1),
    ("air", None, "in_area", 0), ("air", "the v83 air", "in_area", 1),
]


@REAL
@pytest.mark.parametrize("name,world,key,answer", RECOVERS,
                         ids=[f"{n}-{w or 'reach'}-{k}" for n, w, k, _ in RECOVERS])
def test_each_probe_s_log_decides_the_answer_its_world_has(name, world, key, answer):
    # Every probe, run in a world with reach's answer to its question and in one with the other: from its log
    # the scorer decides on the world's answer, on DECIDE_THROWS throws or more where the two have the attack
    # take something else (so on seeds 1-8 too, but for three runs with fewer such throws: below-42 on seed 7,
    # 2 and 1; hanging-wall on seed 6 in reach's world, 4; never on the other answer). below-42 also asks the
    # claw off her level, but its few throws at pigs within 99 decide it alone on 2 of those seeds in a world
    # with the swings: claw-down's 40 throws do on all 8.
    q, model = probe.QUESTIONS[key], scenarios.real_map()
    rows = _probe_run(name, world)
    assert rows and {r["probe"] for r in rows} == {name}
    scored = probe.answers(rows, model, q)
    assert scored.decided == q.answers[answer][0], (name, world, scored.lls, scored.differ)
    assert scored.differ >= probe.DECIDE_THROWS
    assert f"-> {q.answers[answer][0]}:" in "\n".join(probe.answers_text(scored))


@REAL
def test_the_step_up_s_throws_cant_tell_the_two_lines_apart():
    # From 4 at 2 behind the step wall: the v83 line to a walking body on 2 ends at its feet, 60 up, below the
    # step's top corner whatever the distance, so it meets the wall where the level line does (from the
    # ground and jump throws) or clears it with it (jump backs). So the lines are asked only of step-down and
    # hanging-wall.
    model = scenarios.real_map()
    rows = _probe_run("step-up") + _probe_run("step-up", style="jump_back")
    scored = probe.answers(rows, model, dataclasses.replace(probe.QUESTIONS["LINE"], probes=("step-up",)))
    assert scored.throws == len(rows) and scored.differ == 0 and scored.decided is None
    assert "can't tell" in probe.answers_text(scored)[-1]


@REAL
def test_the_scorer_finds_the_window_top_the_world_has(monkeypatch):
    # From 13 at a lone pig on 12, 60 above her, 110-200 off: with the client's top (44, reach's) the star picks
    # it from 111, with the logs' lower one (21) only from 191. Each world's log fits its own top and rules out
    # the other. With no hit at all the fit would be flat down to 0, not bounding the top from below (on seed 1
    # the lower top's world has one).
    model = scenarios.real_map()
    rows = _rows(scenarios.run("probe_top"))
    high = probe.fit_top0(rows, model)
    assert high.holds(44) and not high.holds(21) and high.fits[0] >= 35
    assert "SEARCH_TOP0: fits" in probe.score(rows, model) and "the logs' fit 21" in probe.score(rows, model)
    monkeypatch.setattr(reach, "SEARCH_TOP0", probe.LOGS_TOP0)
    low = probe.fit_top0(_rows(scenarios.run("probe_top")), model)
    monkeypatch.undo()
    assert low.holds(21) and not low.holds(44) and low.fits[0] >= 16 and low.fits[1] <= 30


@REAL
def test_the_scorer_finds_the_window_floor(monkeypatch):
    # From 14's right end at an Orange Mushroom on 17, 60 below her, 90-150 off: with the window's floor 12 above
    # her feet in the first band, its body (55 tall) is picked from 113; with 7 from 93, with 17 from 133. In a
    # world whose floor is 20 (the logs rule out 20 and up), from 153, and the fit finds that instead.
    def run(sim):
        h = sim.hunter()
        return str(probe.run(h, sim.navigator(h), probe.adjust(probe.PROBES["floor"], throws=20), seconds=300))
    sc = scenarios.Scenario("probe_floor", run, (14, 824.0), monsters=[("Orange Mushroom", 17, 940.0)], limit=400)
    fit = probe.fit_floor0(_rows(scenarios.run_scenario(sc)), scenarios.real_map())
    assert fit.holds(12) and not fit.holds(5) and not fit.holds(20)
    monkeypatch.setattr(reach, "SEARCH_FLOOR0", 20)
    rows = _rows(scenarios.run_scenario(sc))
    monkeypatch.undo()
    fit = probe.fit_floor0(rows, scenarios.real_map())
    assert fit.holds(20) and not fit.holds(12)


@REAL
def test_the_scorer_brackets_the_line_s_height_from_the_step_up():
    # From 4 at an Orange Mushroom on 2 behind the step wall: the world's stars stop at the wall from the
    # ground (the line 28 up), not from jump backs (44 up when the attack starts: 72). The line's height fits
    # from 16 (under it, jump backs would be stopped too) to 60 (over it, the throws from the ground would get
    # past); and her group's walls fit far better than none.
    model = scenarios.real_map()
    rows = _rows(scenarios.run("probe_step_up"))
    line = probe.fit_shot_h(rows, model)
    assert line.holds(28) and not line.holds(10) and not line.holds(65)
    scored = probe.answers(rows, model, probe.QUESTIONS["WALLS"])
    assert scored.decided == "they do" and scored.lead > 5


@REAL
def test_the_scorer_tells_the_area_of_an_attack_from_a_jump_from_the_search():
    # Jump throws at a snail about 40 units off: the world's attack picks by the area, and the snail dies to
    # every star; the v83 client's, as from the ground but from 21-26 up, wouldn't pick it. She takes her place
    # facing the way she throws, so no jump throw turns her (a turn's tap walked her 8 units on, to 32 off
    # where the snail came back, and it touched her) and each goes straight up from x 30: the snail is 40 off
    # at all but the first, before which it walked on to 72 off (inside the probe's 35-85) while she took her
    # place.
    rows = _rows(scenarios.run("probe_air"))
    assert {r["x"] for r in rows} == {30.0} and all(35 <= abs(r["dx"]) <= 85 for r in rows)
    assert sum(35 <= abs(r["dx"]) <= 50 for r in rows) >= len(rows) - 1
    scored = probe.answers(rows, scenarios.real_map(), probe.QUESTIONS["in_area"])
    assert scored.decided == "the area fixed to her feet (reach.in_area)" and scored.lead > 5


@REAL
def test_the_scorer_tells_the_claw_s_swipes_from_stars_by_the_bar_they_take():
    # On 21 at a lone pig, then a lone slime, walking at her 95-160 off: the world's claw swipes a body that reaches
    # into its stance's box (65 ahead of her), a pig out to 99 by its middle and a slime to 93.5, for a fifth of a
    # star's damage (a pig's bar by 7%, a slime's by 10%); stars from there. Down at 4, 60 below her, the world's
    # claw (its boxes 15-30 above her feet) swipes nothing; by the swings' boxes (17 below her feet) a third of
    # the throws at a pig within 99.
    def claw_run(kind: str) -> list[dict]:
        def run(sim):
            h = sim.hunter()
            return str(probe.run(h, sim.navigator(h), probe.adjust(probe.PROBES["claw-edge"], throws=20), seconds=400))
        return _rows(scenarios.run_scenario(scenarios.Scenario("probe_claw", run, (21, -225.0),
                                                               monsters=[(kind, 21, -90.0)], limit=500)))
    for kind, edge in (("Pig", 99.5), ("Slime", 93.5)):
        table = probe.claw_table(claw_run(kind))
        claw = [lo for (_, _, dy, lo), g in table.items() if g["claw"]]
        star = [lo for (_, _, dy, lo), g in table.items() if g["star"]]  # a kill can be either
        assert claw and star and max(claw) < edge <= max(star) + 10 and min(star) >= edge - 10, (kind, table)
    rows = _probe_run("claw-down")
    table = probe.claw_table(rows)
    assert sum(g["throws"] for g in table.values()) == 40 and not any(g["claw"] for g in table.values())
    assert all(dy == 60 for _, _, dy, _ in table) and "the claw's" in probe.score(rows, scenarios.real_map())
    table = probe.claw_table(_probe_run("claw-down", "the swings"))
    assert 8 <= sum(g["claw"] for g in table.values()) <= 20


@REAL
def test_a_hidden_target_is_waited_for_and_never_read_as_a_miss(monkeypatch):
    # The game's hit monsters hide under their damage numbers for 1-3 s: the evidence waits up to
    # HIDDEN_MAX_S for the target to show again. With the client's top in the world (reach's), a star picks the
    # pig from 111 units off: none of the throws its attack took may be read as a miss (seen again with its bar
    # unchanged), and most are read as hits (19 of the 30; 22 with pigs walking 70 units/s, not the game data's
    # 125). Throws at a pig that walked out of the search's window by the pick (two 128-129 off that walked to
    # 105) the world's star missed, and so they're read.
    sc = dataclasses.replace(scenarios.SCENARIOS["probe_top"], noise=Noise(hit_hidden=(1.0, 3.0)))
    trace = scenarios.run_scenario(sc)
    model = scenarios.real_map()
    body, client_top = bodies(model), dataclasses.replace(probe.reach_model(), top0=44)
    told = [(r, s) for r in _rows(trace) for e, s in outcomes(r) if e["target"]]
    assert len(told) == 30 and [s for _, s in told].count(True) > len(told) / 2
    for r, s in told:
        if s is False:
            assert all(swipe + star == 0 for e, swipe, star in probe.chances(r, model, body, client_top) if e["target"])
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
