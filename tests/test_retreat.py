"""On the Lie Detector, away to a safe place within 10 s (behavior.retreat; the user, 2026-10-24): where she goes
(`choose`), that the hunt stops and the retreat runs there (`on_alarm`), with her moves alone (Actions.moves_only), and
in the simulator the player seeing it (sim.fakes.FakePlayer.alarm_at) and the retreat on Tree Dungeon. The real-map
tests are skipped without the game data."""

import dataclasses
import math
from types import SimpleNamespace

import pytest

from gamelab.attack import danger
from gamelab.behavior import retreat
from gamelab.behavior.retreat import Place, choose, on_alarm, trip_s
from gamelab.control import client
from gamelab.control.actions import Actions
from gamelab.control.alarm import PET_FOOD, RETREAT, RETREAT_S
from gamelab.control.timeline import SIM_KEYS_PET
from gamelab.localization import mapdata
from gamelab.motion.route import Planner
from gamelab.sim import scenarios

from test_sim import plat, tree

TREE = scenarios.TREE


def tree_map() -> mapdata.MapModel:
    m = scenarios.real_map(TREE)
    if m is None:
        pytest.skip("needs Tree Dungeon, Forest Up North II's game data (gamedata/data)")
    return m


def _choose(m, pid, x, left=9.5, monsters=(), her=danger.Her(752, 24)):
    """`choose` from (platform pid, x), with a look that saw `monsters`, `left` s to get there."""
    h = SimpleNamespace(danger=danger.of_map(m, her))
    nav = SimpleNamespace(model=m, planner=Planner(m))
    s = SimpleNamespace(where=SimpleNamespace(platform=m.platform(pid), x=x), monsters=list(monsters))
    return choose(h, nav, s, left)


# --- where she goes ---------------------------------------------------------------------------------------------

def test_on_a_safe_platform_or_a_place_no_monster_reaches_she_stays():
    m = tree_map()
    assert _choose(m, 25, 150.0) == Place(25, 150.0, "here: a safe platform", 0.0)
    assert not m.safe(28)  # 28 isn't MapModel.safe, but no monster reaches it by the danger model
    stay = _choose(m, 28, 207.0)
    assert stay.why == "here: safe enough" and stay.seconds == 0 and stay.danger <= retreat.SAFE_DANGER


def test_she_goes_to_the_nearest_safe_platform_when_it_is_near():
    m = tree_map()
    p = _choose(m, 24, 231.0)  # shooting 23 from 24, as live: down onto 25, under her
    assert (p.platform, p.x, p.why) == (25, None, "a safe platform") and trip_s(p.seconds) <= retreat.NEAR_S


def test_from_the_floor_no_safe_platform_is_near_and_a_rope_as_safe_comes_first():
    """From Tree Dungeon's floor (33) the nearest place no monster reaches is 28, 4.3 s by route: past NEAR_S by
    `trip_s` (the crowd's stops made it 5-12 s in the simulator). The rope at 146 from 32, as safe at its safest height
    (attack.ropes) and quicker, comes first."""
    m = tree_map()
    p = _choose(m, 33, 0.0)
    assert p.rope is not None and (p.platform, p.rope.x) == (32, 146) and p.danger <= retreat.SAFE_DANGER
    assert p.why == "a rope as safe: no safe platform near"
    assert trip_s(Planner(m).arrive(33, 0.0, 28, 207.0)) > retreat.NEAR_S


def test_with_no_safe_place_in_time_she_goes_to_the_safest_she_gets_to_or_stays():
    m = tree_map()
    p = _choose(m, 33, 0.0, left=2.0)  # nothing safe gets her there in 2 s
    assert p.why.endswith("the safest within reach") and trip_s(p.seconds) <= 2.0 or p.seconds == 0
    here = danger.of_map(m, danger.Her(752, 24))
    assert p.danger <= max(here.at(33, 0.0, None), here.at(33, 0.0, []))
    assert _choose(m, 33, 0.0, left=0.1).seconds == 0  # no time for anything: she stays


def test_the_look_s_monsters_count_a_place_is_safe_only_if_they_and_every_spawn_point_leave_it_so():
    from gamelab.localization.monsters import Monster
    m = tree_map()
    assert _choose(m, 28, 207.0).why == "here: safe enough"
    horny = Monster(SimpleNamespace(kind="horny-mushroom"), 207.0, m.platform(28).y_at(207.0), 28)  # seen on 28
    assert _choose(m, 28, 207.0, monsters=[horny]).why != "here: safe enough"


def test_on_a_map_where_no_kind_counts_only_a_safe_platform_is_a_safe_place():
    """On the first map no kind's touch counts (attack.danger.DANGER_MIN): every place would be "safe enough". She goes
    to a safe platform (MapModel.safe) she gets to in time, or stays."""
    m = mapdata.parse(1, tree(), "Test Map", {}, {"2": "Snail", "8": "Pig"}, {"2": 2})
    d = danger.of_map(m, danger.Her(642, 19))
    assert not d.any
    c = plat(m, 500, 400)
    safe = [p.id for p in m.places if m.safe(p.id)]
    p = _choose(m, c.id, 400.0, her=danger.Her(642, 19))
    if m.safe(c.id):
        assert p.seconds == 0
    else:
        assert p.platform in safe or (p.seconds == 0 and p.why == "here: the safest within reach")
        assert _choose(m, c.id, 400.0, left=0.1, her=danger.Her(642, 19)).seconds == 0


# --- the hunt stops, and the retreat runs there --------------------------------------------------------------------

def test_on_the_alarm_the_retreat_runs_and_the_alarm_goes_on(monkeypatch):
    ran, said = [], []
    h = SimpleNamespace(log=said.append)
    monkeypatch.setattr(retreat, "retreat", lambda hunter, log=None, rng=None: ran.append(hunter) or "went")
    with on_alarm(h) as watch:
        pass  # no alarm: nothing
    assert ran == [] and watch.retreat is None
    with pytest.raises(client.LieDetectorAlarm):
        with on_alarm(h) as watch:
            raise client.LieDetectorAlarm("Lie Detector seen at 21:02:00")
    assert ran == [h] and watch.retreat == "went"
    with pytest.raises(client.PlayerError):  # any other error: no retreat
        with on_alarm(h):
            raise client.PlayerError("game not in front")
    assert ran == [h]

    def broken(hunter, log=None, rng=None):
        raise ValueError("rope out of reach")
    monkeypatch.setattr(retreat, "retreat", broken)
    with pytest.raises(client.LieDetectorAlarm):  # what goes wrong in the retreat doesn't hide the alarm
        with on_alarm(h):
            raise client.LieDetectorAlarm("x")
    assert said == ["retreat: failed (ValueError: rope out of reach)"]


def test_too_late_for_a_retreat_she_stays_and_nothing_is_sent(monkeypatch):
    said = []
    monkeypatch.setattr(client, "alarm_age", lambda: RETREAT_S + 1)
    out = retreat.retreat(SimpleNamespace(log=said.append, actions=None))
    assert out.result == "too late" and out.place is None and "she stays where she is" in said[0]


def test_her_retreat_s_actions_are_her_moves_alone_with_no_mouse_and_her_locks_kept(monkeypatch):
    sent = []
    monkeypatch.setattr(client, "submit", lambda events, **kw: sent.append((kw["name"], kw.get("kind"), events)))
    a = Actions(SIM_KEYS_PET)
    a.lock_until, a.arrow_until, a.facing = 123.0, 124.0, "left"
    moves = a.moves_only()
    assert moves.kind == RETREAT and (moves.lock_until, moves.arrow_until, moves.facing) == (123.0, 124.0, "left")
    k = moves.keys
    assert (k.pick_up, k.pet_food, k.lucky_seven, k.chair) == (None, None, None, None)
    assert (k.left, k.right, k.jump, k.up, k.down) == (SIM_KEYS_PET.left, SIM_KEYS_PET.right, SIM_KEYS_PET.jump,
                                                       SIM_KEYS_PET.up, SIM_KEYS_PET.down)
    assert moves.pickups(1000) == []  # (her walks press none)
    moves.nudge_mouse()  # whoever is at the PC may be answering the Lie Detector with the mouse
    moves.play([[0, "left", "down"], [40, "left", "up"]], "walk")
    a.nudge_mouse()
    assert [(name, kind) for name, kind, _ in sent] == [("walk", RETREAT), ("mouse", None)]


def test_a_hunt_that_only_looks_stops_at_the_alarm_and_its_pet_care_with_it(monkeypatch):
    from gamelab.behavior.hunter import Hunter

    def up():
        raise client.LieDetectorAlarm("Lie Detector seen at 21:02:00")
    monkeypatch.setattr(client, "check_alarm", up)
    h = Hunter.__new__(Hunter)  # (no eyes: neither gets as far as a look)
    with pytest.raises(client.LieDetectorAlarm):
        h.look()
    with pytest.raises(client.LieDetectorAlarm):
        h.feed_pet_if_due()


# --- the player seeing it in the simulator -------------------------------------------------------------------------

def test_in_the_simulator_the_player_stops_the_plan_running_at_the_sighting_then_takes_only_her_food_and_retreat():
    tree_map()

    def run(sim):
        keys, out = sim.keys, {}
        seen = sim.player.alarm_at = sim.clock.now + 0.2  # (the throw's arrow before, its attack key after)
        throw = [[0, keys.right.name, "down"], [400, keys.attack.name, "down"], [440, keys.attack.name, "up"],
                 [900, keys.right.name, "up"]]
        try:
            client.submit(throw, lead_ms=0, name="throw")
        except client.LieDetectorAlarm as e:
            out["throw"] = str(e)
        try:
            client.submit(throw, lead_ms=0, name="throw2")
        except client.LieDetectorAlarm as e:
            out["throw2"] = str(e)
        tap = [[0, keys.pet_food.name, "down"], [40, keys.pet_food.name, "up"]]
        out["food"] = "error" not in client.submit(tap, lead_ms=0, name="pet_food", kind=PET_FOOD)
        try:
            client.submit([[0, keys.left.name, "down"], [100, keys.attack.name, "down"], [140, keys.attack.name, "up"],
                           [300, keys.left.name, "up"]], lead_ms=0, name="flow", kind=RETREAT)
        except client.LieDetectorAlarm as e:
            out["retreat_attack"] = str(e)
        out["retreat"] = "error" not in client.submit([[0, keys.left.name, "down"], [300, keys.left.name, "up"]],
                                                      lead_ms=0, name="flow", kind=RETREAT)
        out["played"] = sorted([round(t - seen, 3), kind, *e] for kind, events in sim.player.played.values()
                               for t, *e in events)
        out["age"], out["up"] = round(client.alarm_age(), 3), client.alarm_up()
        return out

    out = scenarios.run_scenario(scenarios.Scenario("ld", run, (25, 150.0), map_id=TREE))["result"]
    assert out["throw"] == "Lie Detector on screen: all input stopped"  # cut at the sighting
    assert out["throw2"].startswith("Lie Detector seen at")  # refused as it was sent (client.send's check)
    assert "isn't one of her moves' keys" in out["retreat_attack"]
    assert out["food"] and out["retreat"] and out["up"] and out["age"] > 0
    after = [e for e in out["played"] if e[0] >= 0]
    assert [e[1:] for e in after] == [[PET_FOOD, "u", "down"], [PET_FOOD, "u", "up"],
                                      [RETREAT, "left", "down"], [RETREAT, "left", "up"]]
    assert all(e[2] == "right" for e in out["played"] if e[0] < 0)  # the throw's arrow went down; its attack never


def _hanging_at(y: float) -> dict:
    """Her retreat (retreat.retreat) as she hangs on Tree Dungeon's rope at 146 (between 30 above and 32 below), her feet
    at map y, the Lie Detector seen as it starts: how it went, what played from the sighting on, where she is 10 s after
    it (`then`), and the log."""
    def run(sim):
        tour = sim.tour_stack()
        seen = sim.player.alarm_at = sim.clock.now
        then = {}
        sim.world.call_at(seen + RETREAT_S, lambda w: then.update(mode=w.mode, x=round(w.x), y=round(w.y),
                                                                  safe=scenarios.safe_there(sim)))
        out = retreat.retreat(tour.h, log=sim.log)
        sim.clock.sleep(max(0.0, seen + RETREAT_S + 0.1 - sim.clock.now))
        sim.world.advance(sim.clock.now)
        played = sorted(e for kind, events in sim.player.played.values() for e in events)
        return {"retreat": out.record(), "played": played, "then": then}
    tr = scenarios.run_scenario(scenarios.Scenario("ld_rope", run, (32, 146.0), hanging=y, map_id=TREE))
    return dict(tr["result"], log="\n".join(tr["log"]))


def test_hanging_on_a_rope_as_safe_as_a_safe_place_she_stays_on_it():
    """Review, 2026-10-24: the retreat always got her off a rope first, the one the hunt waits on too (its rule 9); in the
    simulator, off the rope at -8 (its danger 0) and across the map, 9.2 s after the sighting. At y 62 on the rope at
    146 no monster reaches her (attack.danger, every spawn point's monster at its point): she stays, nothing pressed."""
    tree_map()
    out = _hanging_at(62.0)
    assert out["retreat"]["why"] == "here: a safe rope" and out["retreat"]["result"] == "stayed", out["log"]
    assert out["retreat"]["to"] == "the rope at x 146" and out["retreat"]["took"] < 1.5
    assert out["played"] == [] and "getting off" not in out["log"]
    assert out["then"]["mode"] == "rope" and out["then"]["safe"]


def test_hanging_on_a_rope_within_a_monsters_reach_she_gets_off_it_for_a_safe_place():
    """At y 22 on the same rope the Horny Mushrooms of 30, just above, reach her (danger 0.059): not a safe place. Off it
    at once, once her dot has shown her hanging (the navigator's look until she stands still, 2 s on a rope, skipped),
    onto 28 by 10 s."""
    m = tree_map()
    d = danger.of_map(m, danger.Her(752, 24))
    assert d.on_rope(next(r for r in m.ropes if r.x == 146), 22.0, None) > retreat.SAFE_DANGER
    out = _hanging_at(22.0)
    assert out["retreat"]["why"] != "here: a safe rope", out["log"]
    assert "on the rope at x 146 (danger 0.059 there): getting off" in out["log"]
    getting_off = next(line for line in out["log"].splitlines() if "getting off" in line)
    assert float(getting_off.split()[0]) < 1.0
    assert out["played"] and out["then"]["safe"], out["log"]


# (loop_tree_lie_detector, _28 and _23 are checked with every scenario: test_sim's
# test_real_map_scenarios_show_what_they_are_there_for.)

def _sweep(name: str, seeds, times, only=None, feed: bool = False) -> list[dict]:
    """The scenario `name` over `seeds` and sighting `times` (`feed`: her looks from a feed, as `tour --loop` takes them
    by default); each run's result (the runs that fell out of the map before the sighting, a simulator effect on Tree
    Dungeon without group 0's walls, left out)."""
    base, out = dataclasses.replace(scenarios.SCENARIOS[name], feed=feed), []
    for seed in seeds:
        for after in times:
            def fn(sim, a=after):
                run = scenarios.lie_detector(a)
                return (scenarios.harness._only(run, only) if only else run)(sim)
            tr = scenarios.run_scenario(dataclasses.replace(base, seed=seed, fn=fn, limit=600))
            if isinstance(tr["result"], dict):
                out.append(dict(tr["result"], unmet=scenarios.unmet(tr, base), seed=seed, after=after))
    return out


# Without the feed (`--no-feed`, and the simulator's default) the hunt loop sweeps the screen a slice a look (hunt.SWEEP):
# hunting the whole map so, she was on a safe place 10 s after the sighting on 15 of 32 (seeds 1-8; 37 of 62 over seeds
# 1-16), 28 of 29 with hunt.SWEEP off, and 27 of 32 at master's head before the rebase of work/integrate-3 (2026-10-25).
# The retreat is the same either way (the sweep is off by then: Hunt.run's `finally`); the hunt before it isn't: on the
# floor more often at the sighting (38 of 61 runs against 30 of 61), hit just before it more often (14 against 8), and
# from the same places her retreat fared worse (the floor and 32: 20 of 38 against 23 of 30). Not fixed: the feed, which
# serves `tour --loop`'s looks by default, leaves the sweep out of the looks it serves. Since her jumps' attacks are
# weighed and her jump throws turn in the air (2026-10-25), 26 of 32 without the feed (with it 28 of 32, none out of the
# map; shooting 23 from 24, 32 of 32): no longer an expected failure, the hunt before the sighting gone otherwise.


@pytest.mark.slow
@pytest.mark.parametrize("name,only,share,feed,seeds", [
    ("loop_tree_lie_detector", None, 0.7, True, range(1, 5)),
    ("loop_tree_lie_detector", None, 0.7, True, range(5, 9)),
    ("loop_tree_lie_detector", None, 0.7, False, range(1, 9)),
    ("loop_tree_lie_detector_23", (23,), 0.9, False, range(1, 9))])
def test_on_most_seeds_she_is_on_a_safe_place_10_s_after_the_sighting(name, only, share, feed, seeds):
    """The hunt loop on Tree Dungeon, the Lie Detector seen at 40, 75, 110 or 150 s, seeds 1-8: from the floor's crowd
    (the whole map hunted) 27 of 32 on a safe place 10 s after the sighting at master's head (2026-10-24: 70% asked, for
    room), the rest knocked back by the Green Mushrooms on 32 and 30 on the way; with her looks from the feed, as
    `tour --loop` takes them by default since then (each half of the seeds a test of its own: a run with the feed takes
    about 9 s), 24 of 29 (14 of 16 on seeds 1-4, 10 of 13 on 5-8, where three of seed 8's fell out of the map before the
    sighting); shooting 23 from 24, as live, 31 of 32 (47 of 48 over seeds 1-12). Since her jumps' attacks are weighed
    (2026-10-25): with the feed 28 of 32 (14 of 16 on each half), without it 26 of 32, shooting 23 from 24 32 of 32. On
    every run nothing but her moves after it (no attack, pick-up, pet food or mouse), and her retreat's moves refused
    past the player's 12 s."""
    tree_map()
    runs = _sweep(name, seeds, (40.0, 75.0, 110.0, 150.0), only, feed)
    # (the runs that fell out of the map before the sighting are left out, three of seed 8's with the feed: up to a
    # quarter of them and two more)
    assert len(runs) >= 3 * len(seeds) - 2, [(r["seed"], r["after"]) for r in runs]
    safe = sum(bool(r["then"].get("safe")) for r in runs)
    assert safe >= math.ceil(share * len(runs)), [(r["seed"], r["after"], r["retreat"]) for r in runs
                                                   if not r["then"].get("safe")]
    alone = "her moves alone after the sighting, no attack and no mouse"
    assert not [r for r in runs if alone in r["unmet"]]
    assert all("are over" in r["late_retreat"] for r in runs)
