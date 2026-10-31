"""The in-game probes (behavior.probe, `python -m gamelab probe`), offline only: what a probe throws at, what
it reads from the looks after a throw, its scorer's rules, and that it runs the way `hunt kill` does (the player,
the Lie Detector alarm), on made-up maps. Its simulated runs on the first map, where the probes are set up
(probe.PROBES), went with that map's tests on 2026-10-31. Nothing here touches the game: tests/conftest.py refuses
any key or window call."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from maplehelper.engine import RIGHT

from gamelab.attack import reach
from gamelab.behavior import probe
from gamelab.behavior.probe import Probe, choose, outcomes, shown
from gamelab.control import alarm, client
from gamelab.localization.monsters import Monster
from gamelab.perception.mobs import Mob

from test_attack import PLAT, step_map, target


# Only with --slow (tests/conftest.py): the probes' answers are in the code since 2026-10-21, so these matter after a
# change to behavior/probe.py, before probing again.
pytestmark = pytest.mark.slow

# From x 500 on PLAT (test_attack's: y 500, 0 to 1000), throwing right at pigs on platform 1, 110-200 off.
TOY = Probe("toy", "a test", 0, 500.0, RIGHT, (1,), ("pig",), (110, 200))


def scene(*monsters, x=500.0, y=500.0, platform=PLAT):
    return SimpleNamespace(where=SimpleNamespace(x=x, y=y), platform=platform, monsters=list(monsters))


# --- what a probe throws at ----------------------------------------------------------------------------

def test_a_probe_throw_dropped_at_its_last_look_is_no_probe_throw():
    """A throw her last look before the key drops (Actions.still: a kill of a kind in its way come in since the look it
    was decided on) sends no key: probe._throw has no star to gather evidence for (None) and counts no throw (until the
    review of 2026-10-26 it scored one all the same)."""
    pig = target(650, 560, platform=1)
    said = []
    for went, flying in ((False, None), (True, "flying")):
        h = SimpleNamespace(model=SimpleNamespace(id=0, platform=lambda pid: None),
                            actions=SimpleNamespace(throw_listeners=[], throw=lambda side: went),
                            report=SimpleNamespace(throws=0), exp_gains=0, log=said.append)
        got = probe._throw(h, scene(pig), TOY, pig)
        assert (got is None) == (flying is None) and h.report.throws == (0 if flying is None else 1)
        assert h.actions.throw_listeners == []
    assert "dropped at its last look" in said[0]


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
    for other in (target(300), target(500 + probe.FRONT + 10, 560, platform=1), target(700, 500 + 160, platform=3)):
        assert choose(scene(pig, other), TOY) is pig
    # But one on her platform in her way (in_her_way: here, of a kind it doesn't throw at, within 150) stops the
    # throw from either side, whatever the crowd: it's fought off first.
    for close in (target(400), target(600)):
        assert choose(scene(pig, close), TOY, crowd=3) is None


def test_no_probe_throw_while_one_of_her_platform_s_monsters_is_closer_than_its_range():
    # A probe that throws at her own platform's monsters (claw-edge, air): one of its kinds there, closer than its
    # range starts, takes the attack itself (the claw's or a star) before any in range, and tells nothing, so no
    # throw while it's there, whatever the crowd; once it's out of the way, the one in range alone is thrown at.
    p = Probe("toy", "a test", 0, 500.0, RIGHT, (0,), ("pig",), (110, 200), crowd=2)
    far, close, behind = target(650), target(560), target(440)
    assert choose(scene(far), p, crowd=2) is far
    assert choose(scene(far, close), p, crowd=2) is None and choose(scene(far, behind), p, crowd=2) is None
    assert choose(scene(far, target(300)), p, crowd=2) is far  # behind her past its range's start
    assert choose(scene(far, target(720)), p, crowd=2) is far  # in front, farther: a crowd of one


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


def test_where_a_monster_will_be_when_the_attack_picks(monkeypatch):
    # A pig seen walking goes on at its kind's speed (125 units/s) the way it faces, turning back at the ends of
    # its walk (its half width, 34, inside its platform's: B runs 200-400 on the step map); one standing, or with
    # its way or its pose not read, stays where the look had it. A probe with at_pick throws by that (claw-down).
    model, a, b, _ = step_map()

    def pig(x, facing=-1, pose="move"):
        return Monster(Mob("pig", 0, 0, 30, 30, 0.9), x, 440.0, b.id, None, facing, pose)
    assert probe.x_after(pig(300, 1), 0.1, model) == pytest.approx(312.5)
    assert probe.x_after(pig(360, 1), 0.1, model) == pytest.approx(359.5)   # to 366 and back
    assert probe.x_after(pig(240), 0.1, model) == pytest.approx(240.5)      # to 234 and back
    for still in (pig(360, 1, "stand"), pig(360, 1, "hit1"), pig(360, 0), pig(360, 1, "")):
        assert probe.x_after(still, 0.1, model) == 360
    # 220 off at the look, walking at her: 195 off 0.2 s later.
    p = Probe("toy", "a test", a.id, 100.0, RIGHT, (b.id,), ("pig",), (150, 200))
    s = scene(pig(320), x=100.0, y=500.0, platform=a)
    monkeypatch.setattr(reach, "WALLS_BLOCK", False)  # B is behind the step wall of her group
    assert choose(s, p, model=model, lead=0.2) is None
    assert choose(s, dataclasses.replace(p, at_pick=True), model=model, lead=0.2) is s.monsters[0]
    assert choose(s, dataclasses.replace(p, at_pick=True), model=model, lead=0.0) is None


def test_a_probe_s_place_is_checked_before_it_runs():
    model, a, b, _ = step_map()
    ok = Probe("toy", "a test", a.id, 100.0, RIGHT, (b.id,), ("pig",), (150, 300))
    assert probe.problems(model, ok) == []
    assert "isn't on platform" in probe.problems(model, probe.adjust(ok, x=300.0))[0]
    # A jump back needs room behind her: she jumps away from the target.
    assert "a jump back needs" in probe.problems(model, probe.adjust(ok, x=50.0, style="jump_back"))[0]
    assert probe.problems(model, probe.adjust(ok, x=150.0, style="jump_back")) == []
    assert "no such way" in probe.problems(model, probe.adjust(ok, style="lucky_seven"))[0]


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


def test_the_scorer_takes_a_walking_monster_as_anywhere_it_can_have_got_to_by_the_pick():
    # The look that logged where each monster was at the pick isn't taken at the pick: one it saw walking is as
    # likely anywhere it walks from PICK_SLACK_S before that look to as long after it (a pig 5 units either way),
    # one standing where it was. So at an answer's edge a throw weighs less, and none tips a question alone.
    walking = {**entry(), "x": 100.0, "facing": 1, "pose": "move"}
    standing = {**entry(target=False), "x": 50.0, "facing": -1, "pose": "stand"}
    row = {"pick": {"x": 0.0, "y": 0.0, "t": 0.1}, "evidence": [walking, standing]}
    placed = probe.placings(row)
    assert [p[0]["x"] for p in placed] == pytest.approx([95, 97.5, 100, 102.5, 105])
    assert {p[1]["x"] for p in placed} == {50.0}
    assert len(probe.placings({**row, "evidence": [standing, {**walking, "pose": "stand"}]})) == 1
    assert len(probe.placings({**row, "evidence": [entry()]})) == 1  # a log from before poses were kept


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


# --- who is in her way, and the alarm ----------------------------------------------------------------------

def test_who_is_in_her_way_on_her_platform():
    # One of a kind it throws at, on its platform (hers), closer than its kind's range starts; any other closer
    # than fight.IN_HER_WAY (150); those on other platforms never.
    p = Probe("toy", "a test", 0, 500.0, RIGHT, (0,), ("pig",), (110, 200), by_kind=(("slime", (60, 90)),))
    pig, far_pig, snail = target(600), target(390), target(640, kind="snail")
    below = target(520, 560, platform=1)
    s = scene(pig, far_pig, snail, below)
    assert probe.in_her_way(p, s) == [pig, snail]  # 100 off: under the pig's 110; the snail 140 off, under 150
    assert probe.in_her_way(p, s, x=400.0) == [far_pig]  # the pig 10 off; the snail 240 off, the other pig 200
    both, slime = dataclasses.replace(p, kinds=("pig", "slime")), target(550, kind="slime")
    assert probe.in_her_way(both, scene(slime)) == [slime]  # a slime by its own range, 60-90: 50 off
    assert probe.in_her_way(both, scene(target(570, kind="slime"))) == []


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
