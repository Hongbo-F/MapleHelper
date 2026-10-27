"""The safety stop (behavior.safety): her HP under SAFETY_HP for SAFETY_S over SAFETY_LOOKS looks ends the hunt; she
goes to a safe place as on the Lie Detector (behavior.retreat.on_alarm), the log says so in capitals, whoever is at the
PC is paged, and the command exits with its own code. No sitting and no waiting for HP (the rests went on 2026-10-26).
In the simulator, end to end: loop_tree_safety_stop (tests/test_sim.py runs every scenario)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gamelab.behavior import retreat
from gamelab.behavior.hunter import Hunter
from gamelab.behavior.safety import EXIT, SAFETY_HP, SAFETY_LOOKS, SAFETY_S, HpWatch, SafetyStop

FIXTURES = Path(__file__).parent / "fixtures"


def _watch(looks, threshold=SAFETY_HP):
    """An HpWatch fed `looks` ([t, hp]...); the stop if it came (and at which look), else None."""
    w = HpWatch(threshold)
    for i, (t, hp) in enumerate(looks):
        w.see(t, hp)
        if (stop := w.due()) is not None:
            return i, stop
    return None


def test_it_stops_once_her_hp_has_been_under_the_threshold_long_enough_over_enough_looks():
    assert (SAFETY_HP, SAFETY_S, SAFETY_LOOKS, EXIT) == (0.4, 2.0, 3, 4)
    under = [[t / 4, 0.3] for t in range(12)]  # a look every 0.25 s, all under 40%
    i, stop = _watch(under)
    assert under[i][0] == SAFETY_S and stop.seconds == SAFETY_S and stop.looks == 9 and stop.hp == 0.3
    assert "under 40% for 2.0 s (9 looks)" in str(stop) and "none left?" in str(stop)
    title, text = stop.page()
    assert title == "MapleStory: her HP stayed low" and "under 40% for 2 s" in text and "safe place" in text
    # Once: what follows it (her way to a safe place) goes on, under the threshold still.
    w = HpWatch()
    for t, hp in under:
        w.see(t, hp)
    assert w.due() is not None and w.due() is None
    w.see(10.0, 0.1)
    assert w.due() is None and w.fired


def test_a_dip_her_pet_drinks_back_a_misread_or_a_look_without_hp_doesnt_stop_it():
    # Back over the threshold in between: the stretch starts again.
    assert _watch([[0.0, 0.3], [1.0, 0.35], [1.9, 0.45], [2.5, 0.3], [3.5, 0.3]]) is None
    # 2 s under it, but over two looks only (looks failing meanwhile): not yet, the third look stops it.
    assert _watch([[0.0, 0.3], [2.5, 0.3]]) is None
    assert _watch([[0.0, 0.3], [2.5, 0.3], [2.6, 0.3]])[0] == 2
    # A look that didn't read her HP neither starts nor ends a stretch.
    assert _watch([[0.0, 0.3], [1.0, None], [1.5, 0.3], [2.0, 0.3]])[0] == 3
    # Dead (HP 0): it stops too, and says she may be.
    i, stop = _watch([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert i == 2 and "she may be dead" in str(stop)
    # Another threshold (`--low-hp`).
    assert _watch([[0.0, 0.45], [1.0, 0.45], [2.0, 0.45]]) is None
    assert _watch([[0.0, 0.45], [1.0, 0.45], [2.0, 0.45]], threshold=0.5) is not None


def test_on_the_live_hunts_of_2026_10_25_it_never_stops_her_hp_never_read_under_53_percent():
    """Her HP as the five live hunts read it, 79 minutes (tests/fixtures/hp_live_2026-10-25.json: every look under 0.75
    and its neighbours): never under 0.538, so never a stop; nor would there have been one at 70%, her pet's own warning
    (19 stretches under it, 1.1 s and two looks at the longest: its potion came)."""
    runs = json.loads((FIXTURES / "hp_live_2026-10-25.json").read_text(encoding="utf-8"))["runs"]
    assert sum(r["looks"] for r in runs.values()) == 10548 and round(sum(r["minutes"] for r in runs.values())) == 79
    assert min(hp for r in runs.values() for _, hp in r["low"]) == 0.538
    for name, r in runs.items():
        assert _watch(r["low"]) is None, name
        assert _watch(r["low"], threshold=0.7) is None, name


def test_her_health_check_drinks_and_raises_the_stop_once():
    h = object.__new__(Hunter)
    drunk = []
    h.drink_if_low = drunk.append
    h.hp_watch = HpWatch()
    looks = [SimpleNamespace(t=t / 2, hp=0.3) for t in range(6)]
    for s in looks[:4]:
        h.hp_watch.see(s.t, s.hp)
        assert h.health_check(s) is None  # 1.5 s under it
    h.hp_watch.see(looks[4].t, looks[4].hp)
    with pytest.raises(SafetyStop):
        h.health_check(looks[4])
    h.hp_watch.see(looks[5].t, looks[5].hp)
    assert h.health_check(looks[5]) is None  # once
    assert drunk == looks  # her potions every time, first


def test_at_the_stop_she_retreats_from_it_its_said_in_capitals_she_is_paged_and_it_goes_on(monkeypatch):
    said, paged, went = [], [], []
    h = SimpleNamespace(log=said.append)

    def away(hunter, seen, why, log=None, rng=None):
        went.append((hunter, why))
        return retreat.Retreat(0.0, result="arrived")
    monkeypatch.setattr(retreat, "away", away)
    monkeypatch.setattr(retreat.alarm, "page", lambda title, text: paged.append(title))
    stop = SafetyStop(0.31, 2.1, 7, 0.4)
    with pytest.raises(SafetyStop):
        with retreat.on_alarm(h) as watch:
            raise stop
    assert went == [(h, retreat.SAFETY)] and watch.retreat.result == "arrived"
    assert said[0].startswith("SAFETY STOP: HP 31%, under 40% for 2.1 s") and "no sitting" in said[0]
    assert paged == ["MapleStory: her HP stayed low"]
    # A retreat that fails is said, and she's paged all the same.

    def broken(*a, **k):
        raise RuntimeError("no dot")
    monkeypatch.setattr(retreat, "away", broken)
    with pytest.raises(SafetyStop):
        with retreat.on_alarm(h):
            raise stop
    assert said[-1] == "retreat: failed (RuntimeError: no dot)" and len(paged) == 2
    # The Lie Detector's retreat is as it was: its own words, from its sighting.
    assert (retreat.LIE_DETECTOR.label, retreat.LIE_DETECTOR.after) == ("LIE DETECTOR", "after it was seen")


def test_a_hunting_command_stopped_by_it_exits_with_its_own_code(monkeypatch, capsys):
    from gamelab import __main__ as cli

    def tour(args):
        raise SafetyStop(0.2, 2.0, 5, 0.4)
    monkeypatch.setattr(cli, "cmd_tour", tour)
    assert cli.main(["tour", "--loop"]) == EXIT
    out = capsys.readouterr().out
    assert out.startswith("SAFETY STOP: HP 20%, under 40% for 2.0 s") and "Check her HP potions" in out
    # `--low-hp` sets its threshold (None: SAFETY_HP), on every command that hunts; the rests' flags are gone.
    p = cli.build_parser()
    for argv in (["tour"], ["hunt", "clear"], ["probe", "run", "top-60"]):
        assert p.parse_args(argv).low_hp is None and p.parse_args(argv + ["--low-hp", "0.5"]).low_hp == 0.5
    for gone in ("--rest-below", "--rest-avoid"):
        with pytest.raises(SystemExit):
            p.parse_args(["tour", gone, "0.5"] if gone == "--rest-below" else ["tour", gone])
