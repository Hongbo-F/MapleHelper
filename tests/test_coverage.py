"""The route coverage report (gamelab.analysis.coverage): which move of a chain a miss goes on, from the navigator's
own records and from the older telemetry's chains and log lines; what the navigator records for it; why a move
lost; and the regression test that the hunt policies reach every platform monsters spawn on."""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from gamelab import telemetry
from gamelab.analysis import coverage
from gamelab.motion.route import route_record
from gamelab.sim import harness, scenarios

REAL = pytest.mark.skipif(not scenarios.available(), reason="needs the map's game data (gamedata/data)")


def test_a_chain_s_move_that_took_her_off_it_is_the_one_after_the_last_platform_she_stood_on():
    up = [36, 32, 29, 28]  # a climb, then two jumps up the basin's left side
    assert coverage.follow(up, [36, 32, 29, 28]) == (3, None)
    assert coverage.follow(up, [36, 32, 36]) == (1, 36)  # the jump off 32 came down in the basin
    assert coverage.follow(up, [32, 28]) == (3, None)    # seen standing now and then only: still all of it
    assert coverage.follow(up, [36, 28, 31]) == (3, None)  # what comes after its end is no part of it
    assert coverage.follow([36, 32], [36]) == (0, None)


@pytest.fixture(scope="module")
def graph():
    model = scenarios.real_map()
    return None if model is None else coverage.Graph(model)


def _standing(graph, pid: int, x: float, topic: str = "glance") -> list[dict]:
    """Two records of her standing on platform pid at x (one is a moment in the air)."""
    y = graph.model.platform(pid).y_at(x)
    if topic == "glance":
        return [{"topic": "glance", "t": 0.0, "x": x, "y": y}] * 2
    return [{"topic": "scene", "t": 0.0, "her": {"x": x, "y": y, "platform": pid}}] * 2


def _up_the_left_side(graph, older: bool = False) -> list[dict]:
    """A route from the basin up its left side and its chain sent: 36 -> 32 -> 29 -> 28."""
    path = [graph.by_key[k] for k in ("climb 36->32 @539", "jump 32->29 @536", "jump 29->28 @356")]
    route = route_record(28, path, start=None if older else (36, 520.0))
    legs = [{"kind": m.kind, "src": m.src, "dst": m.dst, "spans": [], "end": {}} for m in path]
    return _standing(graph, 36, 520, "scene")[:1] + [{"topic": "route", "t": 0.0, **route},
                                                     {"topic": "chain", "t": 0.0, "why": "done", "legs": legs}]


@REAL
@pytest.mark.parametrize("older", [False, True], ids=["moved record", "older telemetry"])
def test_a_miss_goes_on_the_jump_she_came_down_short_from(graph, older):
    """She climbed onto 32 and was seen standing there, then ended in the basin: the jump off 32 missed, the climb was
    made, the jump after it never set off. From the navigator's 'moved' record, and from the older telemetry's chain
    and its "ended on ... instead of" log line alike."""
    records = _up_the_left_side(graph, older) + _standing(graph, 36, 520) + _standing(graph, 32, 560)
    route = next(r for r in records if r["topic"] == "route")
    records += _standing(graph, 36, 470, "scene")
    if older:
        records.append({"topic": "log", "t": 0.0, "text": "    ended on platform 36 (129..951, y 575) instead of "
                                                         "platform 28; replanning"})
    else:
        records.append({"topic": "moved", "t": 0.0, "planned": 28, "landed": 36, "hit": False,
                        "moves": [{"kind": m["kind"], "src": m["src"], "dst": m["dst"], "x": m["x"]}
                                  for m in route["moves"]]})
    t = coverage.tally(records + [{"topic": "trip", "t": 0.0, "dst": 28, "result": "arrived"}], graph)
    assert t["moves"]["climb 36->32 @539"] == [1, 1, 0, 0]
    assert t["moves"]["jump 32->29 @536"] == [1, 1, 1, 0]
    assert t["moves"]["jump 29->28 @356"] == [1, 0, 0, 0]
    assert t["elsewhere"] == {"jump 32->29 @536": {"36": 1}} and t["arrived"] == {"32": 1, "36": 1}
    assert len(t["plans"]) == 1 and t["plans"][0][:3] == [36, 520, 28]  # (older: from the look before it)
    assert t["older"] == (1 if older else 0)


@REAL
def test_a_hit_is_not_a_miss_and_a_chain_that_ends_where_planned_made_every_move(graph):
    moved = {"topic": "moved", "t": 0.0, "planned": 28, "landed": 36, "hit": True,
             "moves": [{"kind": "climb", "src": 36, "dst": 32, "x": 539.0}, {"kind": "jump", "src": 32, "dst": 29,
                                                                             "x": 536.0}]}
    t = coverage.tally(_up_the_left_side(graph) + _standing(graph, 32, 560) + [moved], graph)
    assert t["moves"]["jump 32->29 @536"] == [1, 1, 0, 1] and not t["elsewhere"]
    made = dict(moved, landed=28, hit=False, moves=moved["moves"] + [{"kind": "jump", "src": 29, "dst": 28,
                                                                       "x": 356.0}])
    t = coverage.tally(_up_the_left_side(graph) + [made], graph)  # seen nowhere in between: the navigator's word
    assert [t["moves"][k][1:] for k in ("climb 36->32 @539", "jump 32->29 @536", "jump 29->28 @356")] == [[1, 0, 0]] * 3


@REAL
def test_her_time_goes_to_the_platform_each_look_put_her_on(graph):
    looks = [{"topic": "scene", "t": t, "her": {"x": 500.0, "y": 575.0, "platform": p}}
             for t, p in ((0.0, 36), (2.0, 36), (3.0, None), (4.0, 32), (20.0, 36))]
    t = coverage.tally(looks, graph)
    assert t["platforms"] == {"32": [1, coverage.GAP_S], "36": [2, 2.0 + 1.0]}  # a gap over GAP_S is a pause


@REAL
def test_the_navigator_s_records_put_each_move_made_on_its_move_of_the_graph():
    """nav_rope_past_end (23 -> 17 by the climb past 23's end, one chain), and the two rope hops instead, the climb
    costing 20 times as much as `nav 17 --avoid climb` has it (a move the mover makes alone): the route and 'moved'
    records (test_sim) put on their moves of the graph."""
    g = coverage.graph_for(scenarios.real_map())
    with telemetry.captured() as records:
        trace = scenarios.run("nav_rope_past_end")
    assert trace["result"]["result"] == "arrived"
    t = coverage.tally(records, g)
    assert t["moves"] == {"climb 23->17 @999": [1, 1, 0, 0]} and t["plans"][0][:3] == [23, 900, 17]

    def hops(sim):
        nav = sim.navigator(sim.hunter())
        weight = nav.weight
        nav.weight = lambda m: weight(m) * (20.0 if m.kind == "climb" else 1.0)
        return nav.go(17)
    with telemetry.captured() as records:
        trace = scenarios.run_scenario(scenarios.Scenario("hops", hops, (23, 900.0)))
    assert trace["result"]["result"] == "arrived"
    t = coverage.tally(records, g)
    assert t["moves"]["rope_jump 23->17 @959"][:3] == [1, 1, 0] and t["arrived"] == {"17": 1}


@REAL
def test_the_two_rope_hops_lose_to_the_climb_past_23_s_end_by_their_cost(graph, monkeypatch):
    """By the planner's times the hops are 1.1 s (29%) slower than the climb from where the map's quickest routes
    leave 23 (x 903), so they are the quickest way nowhere ('cost'). The weights (each move 1 to 1.35 times, a trip)
    make that up only when the climb's draw is 1.28 times the hops' or more: on 1.2% of the trips from there. With a
    choice of two ways (route.ROUTES 2, temperature 0.1) the hops are offered when the weights leave them within
    route.STRETCH of the climb, and taken then by the softmax: on about 10% of the trips."""
    from gamelab.motion import route
    hop, climb = graph.by_key["rope_jump 23->17 @959"], graph.by_key["climb 23->17 @999"]
    wide = coverage.map_wide(graph)
    assert abs(wide[coverage.key_of(climb)][0]) < 1e-9 and 1.0 < wide[coverage.key_of(hop)][0] < 1.2
    code, _ = coverage.why(hop, [0, 0, 0, 0], coverage.plan_gaps(graph, [], [hop])[coverage.key_of(hop)],
                           wide[coverage.key_of(hop)])
    assert code == "cost"
    chosen = graph.cost([climb], 903)
    plan = [[23, 903, 17, chosen, chosen, [], [], 1]]
    monkeypatch.setattr(route, "ROUTES", 1)
    expected, tried = coverage.drawn(graph, plan, [hop], draws=2000)
    assert tried == 1 and 0.005 < expected < 0.02
    monkeypatch.setattr(route, "ROUTES", 2)
    monkeypatch.setattr(route, "TEMPERATURE", 0.1)
    expected, tried = coverage.drawn(graph, plan, [hop], draws=300)
    assert tried == 1 and 0.07 < expected < 0.13


@REAL
def test_merged_runs_name_their_moves_as_today_s_graph_does(graph):
    """A tally keyed with other stats (a jump's start a unit or two elsewhere) lands on today's moves."""
    t = {"moves": {"jump 32->29 @537": [2, 1, 0, 0], "jump 32->29 @536": [1, 1, 0, 0]}, "platforms": {}, "plans": []}
    assert coverage.merge([t], graph)["moves"] == {"jump 32->29 @536": [3, 2, 0, 0]}


def test_the_picture_draws_taken_planned_and_unplanned_moves_apart():
    from gamelab.localization import mapdata
    from gamelab.motion.route import Planner
    from gamelab.ui.render import COVER_NEVER, COVER_PLANNED, COVER_TAKEN, coverage_picture
    from test_navigation import tree
    model = mapdata.parse(1, tree(), "Test Map", {})
    moves = Planner(model).moves
    rows = [(m, 3, 2) if i % 3 == 0 else (m, 1, 0) if i % 3 == 1 else (m, 0, 0) for i, m in enumerate(moves)]
    img = coverage_picture(model, rows, width=600, title="test")
    assert img.shape[1] == 600
    px = img.reshape(-1, 3).astype(int)
    for colour in (COVER_TAKEN, COVER_PLANNED, COVER_NEVER):
        assert (np.abs(px - colour).sum(axis=1) < 30).any()


# The regression test: in the simulator the hunt policies get her onto every platform monsters spawn on.
REACH_SEEDS = 5
REACH_MINUTES = 10


@REAL
def test_the_hunt_policies_reach_every_platform_monsters_spawn_on_within_five_seeds():
    """The tour and the hunt loop, each on seeds 1-5 for 10 minutes in the harness's world with 15% of the spawns
    filled (preset game-sparse): between them a move gets her onto every platform monsters spawn on (16 on this
    map; the tally's 'arrived'). On 2026-10-21 every 4 seeds in a row of the first 16 did (13 of 13; 3 in a row,
    13 of 14), and of these 10 runs at least 2 reach each platform (12: the tour on seed 2, the loop on seed 5; 35
    three). In the harness's own world (60% of the spawns) it's the fights in the basin that decide it, not the
    routes: the tour got onto 7 and 35 on 2 seeds of 30, and the hunt loop never onto 2, 7, 12, 20 or 21 in 20
    (README: Route coverage). The runs go in worker processes, as the harness's own do (a few seconds each)."""
    spawn = set(scenarios.real_map().spawn_platforms)
    args = [(p, k, "game-sparse", REACH_MINUTES, 0.0, False, "coded")
            for p in ("tour", "loop") for k in range(1, REACH_SEEDS + 1)]
    with ProcessPoolExecutor(max_workers=min(len(args) // 2, os.cpu_count() or 1)) as pool:
        runs = list(pool.map(harness._run_one_args, args))
    reached = {int(p) for r in runs for p in r["coverage"]["arrived"]}
    assert spawn <= reached, f"never got onto {sorted(spawn - reached)}"
