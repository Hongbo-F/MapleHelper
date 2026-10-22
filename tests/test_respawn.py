"""The respawn and drop-life report (gamelab.analysis.respawn): its survival estimate, which looks see a platform
empty, how an episode after a kill ends, how a drop's life ends, what it infers from the older telemetry, what a
scene record now says, and that on the simulator, whose monsters come back 8 s after they die, it measures 8 s."""

import math

import pytest

from gamelab import telemetry
from gamelab.analysis import respawn
from gamelab.attack.reach import LOOK_DOWN, LOOK_UP
from gamelab.localization.pose import Camera
from gamelab.sim import scenarios
from gamelab.sim.world import RESPAWN

REAL = pytest.mark.skipif(not scenarios.available(), reason="needs the map's game data (gamedata/data)")
BAND = (LOOK_UP, LOOK_DOWN)


def test_kaplan_meier_by_hand():
    """Four times, the second censored: 3 of 4 left after 1 s, half of the 2 still followed after 3 s, none after 4."""
    curve = respawn.km([(1.0, True), (2.0, False), (3.0, True), (4.0, True)])
    assert [(t, round(s, 4)) for t, s, _ in curve] == [(1.0, 0.75), (3.0, 0.375), (4.0, 0.0)]
    assert math.isclose(curve[1][2], 1 / 12 + 1 / 2)  # Greenwood's sum: d/(n(n-d)) at 1 s and 3 s
    est, lo, hi = respawn.quantile(curve, 0.5)
    assert est == 3.0 and lo <= est and (hi is None or hi >= est)
    s, lo, hi = respawn.survival(curve, 2.5)
    assert s == 0.75 and 0 < lo < s < hi < 1


def test_a_median_past_the_last_event_is_not_reached():
    """All censored but one early: the median isn't reached, and the report says so instead of making one up."""
    obs = [(2.0, True)] + [(t, False) for t in (5.0, 9.0, 13.0, 20.0, 40.0, 60.0, 90.0)]
    s = respawn.summary(obs)
    assert s["median"] is None and s["events"] == 1 and s["n"] == 8
    assert [round(v, 3) for _, v, *_ in s["left"][:2]] == [0.875, 0.875]  # S(5 s), S(8 s)


# --- synthetic runs on the real map ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    return scenarios.real_map()


def _scene(model, t, pid, x, monsters=(), loot=(), view=None, searched=True, looked=None, **her):
    """A scene record as the stack writes it from 2026-10-21's evening: her standing on pid at x, having searched her
    band (the screen where the camera puts it)."""
    y = model.platform(pid).y_at(x)
    if view is None:
        cx, cy = Camera(model.bounds).corner(x, y)
        view = [round(cx), round(cy), round(cx + respawn.VIEW[0]), round(cy + respawn.VIEW[1] * 0.88)]
    return {"topic": "scene", "t": t, "her": {"x": x, "y": y, "platform": pid, "seen": True, "grounded": True, **her},
            "monsters": [{"kind": k, "x": mx, "y": model.platform(mp).y_at(mx), "platform": mp, "bar": bar}
                         for k, mp, mx, bar in monsters],
            "loot": [{"kind": k, "x": lx, "y": model.platform(lp).y_at(lx), "platform": lp} for k, lp, lx in loot],
            "searched": searched, "wide": False, "looked": looked if looked is not None else [], "view": view}


def _kill(t, kind, pid, x):
    return {"topic": "event", "t": t, "kind": "kill", "text": f"{kind} down on platform {pid}", "x": x, "y": 0.0}


def _episodes(model, records):
    seq = respawn.looks(records, model, Camera(model.bounds).corner, BAND)
    return respawn.respawns(seq, respawn.kills(records), model, BAND)


def _snail_on_22(model, back_at=9.25, bar=None, stop=None, every=0.25, other=False, gap=None):
    """Her on 22 by its one spawn point (a snail's): the snail seen, killed at 0.3 s, the platform seen empty every
    `every` s, then a snail seen back at `back_at` (with an HP bar, or none: None never), looks stopping at `stop`, and
    none in the span `gap` (from, to)."""
    records = [{"topic": "session", "t": 0.0, "map": model.id, "command": "tour", "mode": "live"},
               _scene(model, 0.0, 22, 60.0, [("snail", 22, 80.0, 0.3)]), _kill(0.3, "snail", 22, 80.0)]
    t = 0.5
    while t < (back_at or 20.0) - 1e-9 and (stop is None or t < stop):
        if gap is None or not gap[0] < t < gap[1]:
            records.append(_scene(model, t, 22, 60.0, [("snail", 22, 100.0, None)] if other else ()))
        t += every
    if back_at is not None:
        records.append(_scene(model, back_at, 22, 60.0, [("snail", 22, 53.0, bar)]))
    return records


@REAL
def test_a_platform_emptied_by_a_kill_and_seen_again_gives_the_respawn(model):
    """The snail on 22 killed at 0.3 s, 22 seen empty, a snail seen there at 9.25 s: back between the last empty look
    (9.0 s) and that one, the midpoint 8.83 s after the kill."""
    eps, dropped = _episodes(model, _snail_on_22(model))
    assert not dropped and len(eps) == 1
    e = eps[0]
    assert e.event and e.platform == 22 and math.isclose(e.time, (9.0 + 9.25) / 2 - 0.3) and math.isclose(e.width, 0.25)
    assert e.lone and e.near  # no other kill on 22 before it; she stood by its spawn point


@REAL
@pytest.mark.parametrize("case, why", [("bar", "a hurt one back"), ("other", "not emptied"),
                                       ("away", "away"), ("gap", "away"), ("end", "end"),
                                       ("unseen", "its monster not seen there")])
def test_how_an_episode_ends_or_makes_none(model, case, why):
    """A snail back with an HP bar is one she had hurt, hidden for a while (dropped); another snail there all along
    leaves it not emptied; looks stopping at 5 s and a snail seen at 9.25 s censor it at the last empty look (she was
    away); so do no looks from 5 to 8 s and 22 seen empty again after (a snail may have come and been killed by someone
    else meanwhile); none seen by the end censors it at the last look; a kill whose monster wasn't seen there makes
    none."""
    records = {"bar": lambda: _snail_on_22(model, bar=0.5), "other": lambda: _snail_on_22(model, other=True),
               "away": lambda: _snail_on_22(model, stop=5.0),
               "gap": lambda: _snail_on_22(model, back_at=None, gap=(5.0, 8.0)),
               "end": lambda: _snail_on_22(model, back_at=None),
               "unseen": lambda: [r for r in _snail_on_22(model) if r["t"] != 0.0 or r["topic"] != "scene"]}[case]()
    eps, dropped = _episodes(model, records)
    if why in ("away", "end"):
        assert len(eps) == 1 and not eps[0].event and eps[0].why == why
        last = max(r["t"] for r in records if r["topic"] == "scene" and not r["monsters"]
                   and (case != "gap" or r["t"] <= 5.0))  # (the gap's: the last before it)
        assert math.isclose(eps[0].time, last - 0.3)
    else:
        assert not eps and dropped == {why: 1}


@REAL
def test_a_look_that_sees_part_of_the_platform_sees_a_monster_back_but_not_the_platform_empty(model):
    """With the screen's left edge over 22's middle, a look doesn't see 22 empty (a monster could be on the rest), but
    a snail it does see there is one back."""
    p = model.platform(22)
    half = [round(p.left + 80), -400, round(p.left + 80 + respawn.VIEW[0]), 300]
    seen = _scene(model, 0.0, 22, 60.0, [("snail", 22, 120.0, None)], view=half)
    empty = _scene(model, 0.0, 22, 60.0, view=half)
    corner = Camera(model.bounds).corner
    a, b = respawn.looks([seen, empty], model, corner, BAND)
    assert not respawn.covers(b, p, BAND)
    records = _snail_on_22(model, back_at=None, stop=9.1) + [dict(seen, t=9.25)]
    eps, _ = _episodes(model, records)
    assert len(eps) == 1 and eps[0].event and math.isclose(eps[0].time, (9.0 + 9.25) / 2 - 0.3)


@REAL
def test_the_older_telemetry_is_read_by_what_followed_each_look(model):
    """Records from before the scene said what it searched: a look followed by the hunt loop's 'monsters' record
    searched (its empty list is a platform seen empty); one followed by nothing didn't (the navigator's look for her
    landing); one within AIR_S of a fight's jump had her in the air."""
    new = _snail_on_22(model)
    old = []
    for r in new:
        if r["topic"] == "scene":
            r = {k: v for k, v in r.items() if k not in ("searched", "wide", "looked", "view")}
            r["her"] = {k: v for k, v in r["her"].items() if k != "grounded"}
        old.append(r)
        if r["topic"] == "scene":
            old.append({"topic": "monsters", "t": r["t"], "list": []})
    eps_new, _ = _episodes(model, new)
    eps_old, _ = _episodes(model, old)
    assert [(e.time, e.event) for e in eps_old] == [(e.time, e.event) for e in eps_new]
    bare = [r for r in old if r["topic"] != "monsters"]
    eps, dropped = _episodes(model, bare)
    assert not eps and dropped == {"not emptied": 1}  # the empty looks said nothing: the snail may never have left
    corner = Camera(model.bounds).corner
    look = next(r for r in old if r["topic"] == "scene" and r["t"] == 0.5)
    jumped = [{"topic": "stance", "t": 1.0, "act": "jump_throw", "aimed": []}, dict(look, t=1.5), dict(look, t=2.4)]
    seq = respawn.looks(jumped, model, corner, BAND)
    assert [lk.standing for lk in seq] == [False, True]  # 0.5 s after the jump, then 1.4 s


def _pig_drop(model, end_x=None, kill_by=None, until=61.0):
    """A pig killed on 36 at x 500 at 0 s, its mesos seen at x 505 from 2.5 s to 60 s with her on 36 at x 300, then
    looks that look all along 36 and see none (her walking to `end_x` meanwhile); another kill at x 490 at `kill_by`."""
    look = [[36, 0, 1000]]
    records = [{"topic": "session", "t": -1.0, "map": model.id, "command": "tour", "mode": "live"},
               _scene(model, -0.5, 36, 300.0, [("pig", 36, 500.0, 0.4)]), _kill(0.0, "pig", 36, 500.0)]
    t = 2.5
    while t <= 60.0:
        records.append(_scene(model, t, 36, 300.0, loot=[("mesos", 36, 505.0)], looked=look))
        if kill_by is not None and t < kill_by <= t + 2.5:
            records.append(_kill(kill_by, "pig", 36, 490.0))
        t += 2.5
    x = 300.0 if end_x is None else end_x
    for t in (61.0, 62.0, 63.0)[:int(until - 58)]:
        records.append(_scene(model, t, 36, x, looked=look))
    return records


@REAL
@pytest.mark.parametrize("case", ["gone", "picked", "a kill by it", "unseen"])
def test_how_a_drop_s_life_ends(model, case):
    """Her pig's mesos, seen up to 60 s: gone without her by it, at the midpoint of 60 s and the first look that saw
    none (61 s) from the kill; with her walking over it meanwhile, hers (picked up: censored at 60 s); with another
    kill by it at 31 s, followed only until then (another drop could land where it lay); with no look after, never seen
    gone."""
    records = {"gone": lambda: _pig_drop(model), "picked": lambda: _pig_drop(model, end_x=600.0),
               "a kill by it": lambda: _pig_drop(model, kill_by=31.0),
               "unseen": lambda: _pig_drop(model, until=58.0)}[case]()
    seq = respawn.looks(records, model, Camera(model.bounds).corner, BAND)
    lives = respawn.drops(seq, respawn.kills(records))
    life = next(d for d in lives if d.kind == "mesos")
    assert life.hers and life.born == 0.0 and life.why == case
    want = {"gone": (60.0 + 61.0) / 2, "picked": 60.0, "a kill by it": 30.0, "unseen": 60.0}[case]
    assert math.isclose(life.time, want) and life.event == (case == "gone")


@REAL
def test_a_drop_with_another_over_it_is_not_seen_gone(model):
    """Another drop lying within OVERLAP_DX of it: drops drawn over one another are found as one, so not seeing it
    there says nothing."""
    records = _pig_drop(model)
    for r in records:
        if r["topic"] == "scene" and r["t"] > 60:
            r["loot"] = [{"kind": "leather", "x": 512.0, "y": 575.0, "platform": 36}]
    seq = respawn.looks(records, model, Camera(model.bounds).corner, BAND)
    life = next(d for d in respawn.drops(seq, respawn.kills(records)) if d.kind == "mesos")
    assert life.why == "unseen" and not life.event


# --- what the stack records, and the simulator --------------------------------------------------------------

@REAL
def test_on_the_simulator_it_measures_the_simulator_s_respawn():
    """The simulator brings a monster back at its spawn point RESPAWN (8 s) after it dies: a pig alone in the basin,
    shot dead, then 14 s of looks with her standing there. The report finds one episode, back 7-8.7 s after the kill's
    event (that comes at the look after the death, so a little short; the look that sees it back up to a look late):
    7.6 s. (Seed 3: on 5 of seeds 1-8 her sweep of the pig's drops, whose looks don't search for monsters, lasted
    until it was back, so the basin was never seen empty and there's no episode; on the other three, 7.6-7.8 s.)"""

    def fn(sim):
        h = sim.hunter()
        h.clear({"pig"}, ground={36}, max_kills=1, seconds=40, far=True)
        until = sim.clock.elapsed + 14.0
        while sim.clock.elapsed < until:
            h.look()
        return {}

    sc = scenarios.Scenario("respawn", fn, (36, 450.0), monsters=[("Pig", 36, 650.0)], seed=3)
    with telemetry.captured() as records:
        _, sim = scenarios.run_sim(sc)
    model = sim.model
    eps, dropped = _episodes(model, records)
    assert len(eps) == 1, (eps, dropped)
    e = eps[0]
    assert e.event and e.platform == 36 and RESPAWN - 1.0 <= e.time <= RESPAWN + 0.7, e


@REAL
def test_a_hunt_with_the_feed_records_the_deaths_of_others_kills():
    """With the continuous feed a hunt publishes the deaths the feed saw that weren't hers ('died', as `watch` does),
    which the report reads as others' kills; hers stay the 'kill' events alone (not counted twice). feed_loop_rival
    for 45 s: another player kills a monster every 20 s, and the feed sees one of those die."""
    import dataclasses
    sc = dataclasses.replace(scenarios.SCENARIOS["feed_loop_rival"], fn=scenarios._loop(0.75), expects=())
    with telemetry.captured() as records:
        trace = scenarios.run_scenario(sc)
    died = [r for r in records if r["topic"] == "event" and r["kind"] == "died"]
    assert died and len(died) == trace["feed"]["tracker"]["died other"]
    dead = respawn.kills(records)
    assert sum(not k.hers for k in dead) == len(died)
    assert sum(k.hers for k in dead) == sum(1 for r in records if r["topic"] == "event" and r["kind"] == "kill")


@REAL
def test_the_command_reads_telemetry_files(model, tmp_path, capsys):
    """`python -m gamelab respawn FILE`: the report of a telemetry file, offline."""
    import json

    from gamelab.__main__ import main
    path = tmp_path / "20261021-120000-tour.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _snail_on_22(model)), encoding="utf-8")
    assert main(["respawn", str(path)]) == 0
    out = capsys.readouterr().out
    assert "1 files" in out and "n 1 (1 seen, 0 censored)" in out and "createMobInterval none" in out
