"""The records a run leaves to be analysed by (from 2026-11-02: ROADMAP item 2 of the analysis of that day's three
recordings, D:/MapleHelper-scratch/rec-analysis/plans): the run summary and its telemetry 'summary' record, the trips'
end places, the passes regardless' why, the loop's promise against what it gave, the session's physics (analysis.walkfit
reads them: tests/test_walkfit.py). R1's perception service's EXP records over its grind (00:09:36-00:14:57) are a
fixture (tests/fixtures/records/r1-grind-exp.jsonl)."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gamelab import telemetry
from gamelab.behavior.fight import Report
from gamelab.behavior.grind import Grind
from gamelab.behavior.hunt import Hunt, HuntReport
from gamelab.behavior.hunter import Hunter, feed_reads
from gamelab.control.navigator import TripState, _trip_end
from gamelab.localization.gains import ExpMeter
from gamelab.localization.tracking import Event
from gamelab.motion import physics
from gamelab.sim import scenarios

FIX = Path(__file__).parent / "fixtures" / "records"
TREE_EXP = {"Green Mushroom": 26, "Horny Mushroom": 41}


def _r1_changes(before: bool) -> list[Event]:
    """R1's EXP changes over its grind as the perception service told them (its 'exp' records: the number after, its
    level and the gain), each with what it said before when `before` (worldstate's, from 2026-11-02)."""
    out = []
    for line in (FIX / "r1-grind-exp.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        data = {"exp": r["exp"], "percent": None, "levels": [r["level"]]}
        if before and r["gain"] is not None:
            data["before"] = [r["level"], r["exp"] - r["gain"]]
        out.append(Event("exp_gain", r["t"], data=data))
    return out


@pytest.mark.parametrize("before,total", [(True, 1798), (False, 1772)])
def test_r1s_exp_off_her_number_counts_the_first_kill_of_a_hunt_that_attached_after_the_service_read_it(before, total):
    """R1's 53 gains, +1,798 (the analysis' count off the service's records). A hunt's meter attaching after the service
    read the number hears only its changes: started from the first one's `before`, it reads them all; without (as until
    2026-11-02) the first change was its first reading, and that kill's 26 were lost (the summary's +1,731 lost the
    stand-by trip's 41 too: Hunt.run counts it now)."""
    meter = ExpMeter(TREE_EXP)
    changes = _r1_changes(before)
    assert len(changes) == 53
    for e in changes:  # (a change a look)
        for t, reading, changed in feed_reads([e], meter):
            meter.update(t, reading, changed)
    assert meter.total == total


def test_a_meter_that_has_read_the_number_takes_no_before():
    meter = ExpMeter(TREE_EXP)
    meter.update(1.0, *feed_reads([Event("exp_read", 1.0, data={"exp": 100, "levels": [31]})], meter)[0][1:])
    reads = feed_reads([Event("exp_gain", 2.0, data={"exp": 126, "levels": [31], "before": [31, 90]})], meter)
    assert [r[2] for r in reads] == [True]  # (its own reading stands: 100, the gain 26)


def test_the_run_summary_says_every_throw_the_stand_by_and_her_touches_potions_and_lowest_hp():
    """R1's figures as the summary now has them (the analysis' truth: 53 kills, +1,798, 85 casts)."""
    r = HuntReport(seconds=301.0, standby_s=13.0, grounds=[7, 8], kills={"green-mushroom": 25, "horny-mushroom": 28},
                   exp_gains=53, exp=1798.0, exp_read=1798, throws=80, casts=85, standby_kills=1, piles=65,
                   touches=29, misses=0, drinks=15, lowest_hp=527, max_hp=842, stop="time up")
    text = str(r)
    assert text.startswith("301 s and 13 s to stand by; grounds [7, 8]; kills: 25 green-mushroom, 28 horny-mushroom "
                           "(EXP gains 53, EXP 1798; off her EXP number +1798, 343.6 a minute; 1 on her way to stand "
                           "by)")
    assert "throws 85 (5 clearing her way); piles 65; touches 29 (5.5 a minute), 0 MISS; potions 15 (2.9 a minute), " \
           "lowest HP 527 of 842; ended: time up" in text
    rec = r.record()
    assert rec["casts"] == 85 and rec["a_minute"]["exp_read"] == pytest.approx(343.57, abs=0.01)
    assert rec["lowest_hp"] == 527 and rec["standby_s"] == 13.0 and json.dumps(rec)


def test_the_hunt_counts_the_stand_by_trips_kills_and_takes_the_hunters_figures():
    kills = [SimpleNamespace(kind="green-mushroom", named=None)] * 52 \
        + [SimpleNamespace(kind=None, named="horny-mushroom")]  # (the 53rd on her way to stand by: named by its gain)
    h = SimpleNamespace(report=Report(throws=85, touches=29, misses=0, drinks=15, lowest_hp=527.0), max_hp=842,
                        ledger=SimpleNamespace(kills=kills))
    hunt = object.__new__(Hunt)
    hunt.h, hunt.report, hunt._kills_seen = h, HuntReport(throws=80, exp_gains=52), 52
    hunt.tour, hunt._avg_exp = SimpleNamespace(exp={"horny-mushroom": 41.0}), lambda: 30.0
    assert len(hunt._book_after()) == 1 and hunt._kills_seen == 53
    assert (hunt.report.exp_gains, hunt.report.exp, hunt.report.standby_kills) == (53, 41.0, 1)
    with telemetry.captured() as records:
        hunt._summary(SimpleNamespace(since=0.0, total=1798), None)
    r = hunt.report
    assert (r.exp_read, r.casts, r.throws, r.touches, r.drinks, r.lowest_hp, r.max_hp) == (1798, 85, 80, 29, 15, 527.0,
                                                                                             842)
    assert [x["topic"] for x in records] == ["summary"] and records[0]["casts"] == 85


def test_a_kill_whose_exp_showed_before_time_up_is_the_hunts_not_the_stand_bys():
    """The kills booked after her trip to stand by: those whose EXP showed before the hunt loop was over are the hunt's
    own (counted, a grind's credit by the platform it died on), only the rest the stand-by trip's. The loop's last pass
    can end on one its looks hadn't had: loop_tree_grind's Green, its EXP seen come in by the glance before a throw's
    key at 180.3 s, the ledger's at the stand-by trip's first look (the review of 2026-11-02)."""
    green = SimpleNamespace(kind="green-mushroom", named=None, placed=True, platform=7, ground=7, t=181.2, gained=180.2)
    horny = SimpleNamespace(kind="horny-mushroom", named=None, placed=True, platform=9, ground=9, t=184.0, gained=183.9)
    h = SimpleNamespace(ledger=SimpleNamespace(kills=[green, horny]))
    g = object.__new__(Grind)
    g.h, g.report, g._kills_seen, g.credit = h, HuntReport(exp_gains=15, exp=400.0), 0, {}
    g._credited, g._moves_seen = {}, 0  # (the ledger's moves followed: Grind._book_kills; none here)
    g.tour, g._avg_exp = SimpleNamespace(exp={"green-mushroom": 26.0, "horny-mushroom": 41.0}), lambda: 30.0
    g.visit = g.obj = None
    g._over_at, g._s = 180.4, SimpleNamespace(t=180.4, platform=SimpleNamespace(id=7))  # (the loop over, her last look)
    assert g._book_after() == [horny] and g._kills_seen == 2
    r = g.report
    assert (r.exp_gains, r.exp, r.standby_kills, r.on_the_way) == (17, 467.0, 1, 2) and g.credit == {7: [1, 26.0]}


def test_her_potions_and_lowest_hp_are_her_hp_numbers():
    rep = Report()
    for hp in (700, 527, 800):
        rep.see_hp(hp)
    rep.see_hp(None)
    other = Report(drinks=2, lowest_hp=510.0)
    rep.add(other)
    assert rep.lowest_hp == 510.0 and rep.drinks == 2


def test_a_level_ups_refill_isnt_a_potion():
    """Her HP number rising POTION_RISE_HP or more from one read to the next is a potion, but not a rise to a max HP
    other than the read before's: a level's refill (the review of 2026-11-02; R1's 15 rises were +160 each)."""
    h = object.__new__(Hunter)
    h.report, h._read_max = Report(), None
    h._max_hp_seen, h._touched = (lambda most: None), (lambda *a: None)
    reads = [(1.0, 600, 842, None), (2.0, 760, 842, 600), (3.0, 700, 842, 760), (4.0, 870, 870, 700),
             (5.0, 700, 870, 870), (6.0, 860, 870, 700)]
    h._hp_number(SimpleNamespace(events=[Event("hp_read", t, data={"cur": c, "max": m, "before": b})
                                         for t, c, m, b in reads], hp_of=None, hp_box=None))
    assert h.report.drinks == 2 and h.report.lowest_hp == 600  # (+160 twice; the level's +170 to 870 isn't one)


def test_a_trip_record_says_where_she_was_and_its_arrival_walk():
    p10, p12 = SimpleNamespace(id=10), SimpleNamespace(id=12)
    st = TripState(10, x=210.0)
    st.last = SimpleNamespace(where=SimpleNamespace(platform=p10, x=-45.5))
    assert _trip_end(st) == {"on": 10, "x": -45.5}
    st.walked_to, st.walked = SimpleNamespace(where=SimpleNamespace(platform=p12, x=242.46)), [[255.5, 1940]]
    assert _trip_end(st) == {"on": 12, "x": 242.5, "walk": [[255.5, 1940]]}  # (off 10's end: 'arrived' on 12)


def test_the_loops_promise_against_what_it_gave():
    g = object.__new__(Grind)
    now = time.monotonic()
    g.patrol = SimpleNamespace(plan=SimpleNamespace(rate=8.5))
    g.report = HuntReport(exp=437.0)
    g.report.rule_s.update({"sweep": 21.0, "travel": 7.2})
    g._start, g._loop_at, g._loop_exp = now - 60.0, now - 30.0, 237.0
    g._promised, g._promised_s = 9.27 * 60.0, 60.0
    got = g._promise()
    assert got["loop_rate"] == 510.0 and got["promised"] == pytest.approx(556.2) and got["loop_exp"] == 200.0
    assert got["delivered"] == pytest.approx(437.0, abs=0.5) and got["loop_s"] == pytest.approx(30.0, abs=0.5)
    assert got["rule_s"] == {"sweep": 21.0, "travel": 7.2}


def test_the_session_says_her_physics_as_planned():
    rec = physics.record()  # (FIXED_STATS: the thief's)
    assert rec["character"] == "thief" and rec["speed"] == 101.0 and rec["walk"] == pytest.approx(151.25)
    assert rec["bonus"] == 20.0  # (the client's hidden Speed: 125 x (101 + 20)/100)
    assert rec["climb"] == physics.CLIMB_SPEED == 121.0 and json.dumps(rec)
    assert (rec["prior_walk"], rec["prior_climb"], rec["calibrated"]) == (151.25, 121.0, False)
    physics.set_motion(150.4, 120.2)  # (as her calibration has them: motion.calibrate)
    rec = physics.record()
    assert (rec["walk"], rec["climb"], rec["prior_walk"], rec["calibrated"]) == (150.4, 120.2, 151.25, True)


# --- a simulated grind's records ----------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not scenarios.available(scenarios.TREE), reason="needs Tree Dungeon's game data (gamedata/data)")
def test_a_simulated_grinds_records_add_up():
    """loop_tree_grind's: the summary's casts are every throw the simulator saw, its kills and EXP its EXP records';
    every trip says where it left her; every scene its frame and age; the grind's states its promise and delivery."""
    with telemetry.captured() as records:
        scenarios.run("loop_tree_grind")
    (s,) = [r for r in records if r["topic"] == "summary"]
    gains = [r for r in records if r["topic"] == "exp" and r.get("gain")]
    assert s["casts"] == sum(1 for r in records if r["topic"] == "sim_attack") and s["casts"] >= s["throws"]
    assert s["exp_gains"] == len(gains) and s["exp_read"] == gains[-1]["total"] and s["standby_s"] > 0
    assert s["touches"] == sum(1 for r in records if r["topic"] == "event" and r.get("kind") == "touch")
    assert s["lowest_hp"] is not None and sum(s["rule_s"].values()) == pytest.approx(s["seconds"], rel=0.05)
    trips = [r for r in records if r["topic"] == "trip"]
    assert trips and all("on" in r and "x" in r for r in trips)
    scenes = [r for r in records if r["topic"] == "scene"]
    assert all(isinstance(r.get("seq"), int) and r["age"] >= 0 for r in scenes)
    assert [r["seq"] for r in scenes] == sorted(r["seq"] for r in scenes)
    states = [r for r in records if r["topic"] == "grind" and r["what"] in ("state", "end")]
    assert states and all({"loop_rate", "promised", "delivered", "rule_s"} <= set(r) for r in states)
    passes = [r for r in records if r["topic"] == "event" and r.get("kind") == "backstop" and "move" in r]
    assert all({"episode", "nth", "waited", "shots"} <= set(r) for r in passes)
