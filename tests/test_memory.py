"""Her memory of the monsters (localization.memory) and the sweep that keeps it up (localization.senses: a slice of
the screen a look), on made-up maps and frames drawn as data: no game, no simulator."""

import math
import time
from types import SimpleNamespace

import pytest

from gamelab.localization import mapdata, memory
from gamelab.localization.memory import Cover, Memory, P_FORGET, cover_of, replay
from gamelab.localization.monsters import Monster
from gamelab.localization.players import Player
from gamelab.localization.senses import Senses
from gamelab.perception.eyes import band_rows
from gamelab.perception.mobs import Mob
from gamelab.screen import UNITS_PER_HALF_PX as U

from test_worldstate import Drawn, pig

C, B, A = 0, 1, 2  # her platform A (y 500), B 200 above (y 300), C 410 above (y 90: cut by the screen's top); ids top
                   # down


def three_floors():
    tree = {
        "info": {"VRLeft": 0, "VRTop": 0, "VRRight": 3000, "VRBottom": 1000},
        "foothold": {"1": {
            "1": {"1": {"x1": 400, "y1": 500, "x2": 1600, "y2": 500, "prev": 0, "next": 0}},   # hers
            "2": {"2": {"x1": 450, "y1": 300, "x2": 1500, "y2": 300, "prev": 0, "next": 0}},   # 200 above
            "3": {"3": {"x1": 500, "y1": 90, "x2": 1500, "y2": 90, "prev": 0, "next": 0}},     # 410 above
        }},
        "miniMap": {"width": 3000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4},
    }
    m = mapdata.parse(1, tree, "Test", {})
    assert [round(m.platform(pid).y_at(1000)) for pid in (A, B, C)] == [500, 300, 90]
    return m


def mon(x: float, platform: int, kind: str = "pig", model=None, bar=None) -> Monster:
    y = {A: 500.0, B: 300.0, C: 90.0}[platform]
    return Monster(Mob(kind, 0, 0, 0, 0, 0.9), x, y, platform, bar)


def whole(t: float) -> Cover:
    """A whole-screen look with her at (1000, 500): the screen from y 73 to 175 below her feet, x 317 to 1683."""
    return Cover(t, ((500 - 300 * U, 500 + 175 * U),), 1000 - 480 * U, 1000 + 480 * U)


def band(t: float) -> Cover:
    """A look at her band alone (100 up and down)."""
    return Cover(t, ((400.0, 600.0),), 1000 - 480 * U, 1000 + 480 * U)


def test_a_monster_seen_once_is_kept_through_looks_that_dont_search_its_place():
    m = Memory(three_floors())
    m.update(0.0, [mon(1000, B)], whole(0.0))
    for t in (0.5, 1.0, 5.0):
        m.update(t, [], band(t))  # her band: platform B isn't in it
    (e,) = m.alive(5.0)
    assert e.platform == B and e.misses == 0 and e.missed is None
    assert e.p(5.0) == pytest.approx(math.exp(-5.0 / memory.STAY_S))  # by age alone
    assert [x.x for x in m.there(5.0, B)] == [1000] and m.there(5.0, A) == []


def test_looks_covering_its_place_forget_it_by_evidence_and_not_at_one_look():
    m = Memory(three_floors())
    m.update(0.0, [mon(1000, B)], whole(0.0))
    m.update(0.2, [], whole(0.2))  # one look missing it: still there, most likely (hidden under damage numbers)
    (e,) = m.entries
    assert e.live and e.misses == 1 and e.p(0.2) > 0.9
    m.update(0.4, [mon(1010, B)], whole(0.4))  # seen again: the same one, and no longer missed
    assert m.entries == [e] and e.misses == 0 and e.x == 1010 and e.sightings == 2
    t = 0.4
    while e.live:
        t = round(t + 0.2, 1)
        m.update(t, [], whole(t))
    # Gone once exp(-d / MISS_S - d / STAY_S) fell under P_FORGET: d about 5.3 s after it was last seen.
    d = -math.log(P_FORGET) / (1 / memory.MISS_S + 1 / memory.STAY_S)
    assert e.state == "gone" and t - 0.4 == pytest.approx(d, abs=0.25)
    assert m.counts["gone"] == 1 and m.events[-1][0] == "gone"
    assert m.there(t, B, min_p=0.0) == []
    m.update(t + memory.ENDED_KEEP_S, [], band(t + memory.ENDED_KEEP_S))
    assert m.entries == []  # kept a moment, ended, for the mini-map; then dropped


def test_a_body_cut_by_the_screen_s_top_is_never_missed_only_aged():
    m = Memory(three_floors())
    m.update(0.0, [mon(1000, C)], whole(0.0))  # its feet 410 above hers: its body's top out of the screen
    for t in range(1, 30):
        m.update(float(t), [], whole(float(t)))
    (e,) = m.entries
    assert e.live and e.misses == 0  # never wholly searched: no look says it's gone
    m.update(memory.MEMORY_S + 1, [], whole(memory.MEMORY_S + 1))
    assert e.state == "gone" and m.counts["aged"] == 1


def test_a_sighting_is_the_same_monster_when_it_can_have_walked_there():
    m = Memory(three_floors())
    m.update(0.0, [mon(1000, B)], whole(0.0))
    m.update(0.1, [mon(1150, B)], whole(0.1))  # 150 units in 0.1 s: another one (a pig walks 125 a second)
    assert len(m.alive(0.1)) == 2
    m = Memory(three_floors())
    m.update(0.0, [mon(1000, B)], whole(0.0))
    m.update(1.0, [mon(1150, B)], whole(1.0))  # in a second it can have: the same
    assert len(m.alive(1.0)) == 1 and m.alive(1.0)[0].x == 1150
    m.update(1.1, [mon(1150, B, kind="slime"), mon(1150, A)], whole(1.1))  # another kind, another platform: new
    assert sorted((e.kind, e.platform) for e in m.alive(1.1)) == sorted([("pig", A), ("pig", B), ("slime", B)])


def test_a_kill_of_hers_marks_the_one_it_was_and_its_fading_body_is_not_a_new_one():
    kills = []
    m = Memory(three_floors(), kills=lambda: kills)
    m.update(0.0, [mon(900, A), mon(1300, A)], band(0.0))
    kills.append(SimpleNamespace(t=0.5, kind="pig", platform=A, x=1290.0))
    m.update(0.5, [mon(900, A), mon(1290, A)], band(0.5))  # it fades out where it died, still seen for a moment
    assert [(e.x, e.state, e.died_x) for e in m.entries] == [(900, "seen", None), (1300, "killed", 1290.0)]
    m.update(0.7, [mon(900, A), mon(1360, A)], band(0.7))  # ... knocked on past where the ledger placed it
    assert m.counts["killed"] == 1 and m.counts["new"] == 2
    m.update(0.7, [mon(1360, A, kind="slime")], band(0.7))  # another kind there: another monster
    assert m.counts["new"] == 3
    kills.append(SimpleNamespace(t=0.6, kind=None, platform=None, x=0.0))  # one no throw explains: no place
    m.update(3.0, [mon(900, A)], band(3.0))
    assert [e.x for e in m.entries if e.kind == "pig"] == [900]


def test_a_monster_she_threw_at_is_not_missed_while_it_may_hide_under_her_damage_numbers():
    m = Memory(three_floors())
    m._her = (1000.0, 500.0)
    m.update(0.0, [mon(1200, A), mon(800, A)], band(0.0))
    m.on_throw(0.0, "right")
    for t in (0.5, 1.0, 2.0):
        m.update(t, [], band(t))
    right, left = sorted(m.entries, key=lambda e: -e.x)
    assert right.misses == 0 and left.misses == 3  # only the one her way can be hiding under her numbers
    m.on_throw(3.0, "right")  # throwing on that way: still, a while after it was seen last, looks without it count
    t = memory.HIT_AFTER_S + memory.HIDE_S + 0.1
    m.update(t, [], band(t))
    assert right.misses == 1 and right.p(t) < memory.SURE


def test_the_minimap_hides_what_is_under_it_and_other_players_are_noted():
    m = Memory(three_floors())
    m.update(0.0, [mon(470, B), mon(1000, B)], whole(0.0))
    covered = Cover(0.2, whole(0.2).rows, whole(0.2).x0, whole(0.2).x1, minimap=(300.0, 100.0, 530.0, 290.0))
    them = [Player(1, "stranger", 1050.0, 300.0, B, (0.0, 0.0), 0.0, 0.0)]
    m.update(0.2, [], covered, them)
    under, clear = sorted(m.entries, key=lambda e: e.x)
    assert under.misses == 0 and clear.misses == 1
    assert clear.rival and under.rival is False  # 580 across from them: out of their reach


def test_a_platform_is_covered_when_a_look_searched_all_of_it():
    m = Memory(three_floors())
    m.update(1.0, [], band(1.0))
    assert m.covered_at(A) is None  # no monsters spawn on these platforms: none is judged
    model = three_floors()
    model.monsters_by_platform.update({A: ["Pig"], B: ["Pig"], C: ["Pig"]})
    m = Memory(model)
    m.update(1.0, [], band(1.0))
    m.update(2.0, [], whole(2.0))
    assert m.covered_at(A) == 2.0 and m.covered_at(B) == 2.0 and m.covered_at(C) is None  # C's monsters: cut


def _senses(model, sweep: bool = False):
    eyes = Drawn(model)
    s = Senses(eyes, model, band=(100, 100), sweep=sweep)
    asked = []
    find = eyes.monsters_rows
    eyes.monsters_rows = lambda obs, rows: (asked.append(rows), find(obs, rows))[1]
    return eyes, s, asked


def test_the_sweep_searches_the_screen_outside_her_band_a_slice_a_look_from_the_top_down():
    model = three_floors()
    eyes, senses, asked = _senses(model, sweep=True)
    eyes.mobs = [pig(1000, 300), pig(1000)]  # one on B (out of her band), one by her
    looks = [senses.look() for _ in range(5)]
    bottom = int(540 * looks[0].geo.hud_top)
    b0, b1 = band_rows(looks[0], 100, 100)  # her band's rows
    body = math.ceil((max(b.top for b in memory.bodies(model).values()) + 8) / U)
    cycle = asked[:4]
    assert cycle[0][0] == 0 and cycle[-1][1] == bottom and asked[4] == asked[0]  # 4 looks, then from the top again
    for (_, end), (start, _) in zip(cycle, cycle[1:]):
        assert end - start >= body or start >= b1 - body  # each overlaps the last by a body (or starts under the band)
    assert cycle[2][1] <= b0 + body and cycle[3][0] >= b1 - body  # nothing searched twice in the band's middle
    assert all(len(s.monsters) == 1 and s.monsters[0].platform == A for s in looks)  # the fights' monsters: the band's
    assert [len(s.swept) for s in looks] == [0, 1, 0, 0, 0] and looks[1].swept[0].platform == B
    assert looks[1].rows == [(500 + (b0 - 300) * U, 500 + (b1 - 300) * U),
                             (500 + (cycle[1][0] - 300) * U, 500 + (cycle[1][1] - 300) * U)]
    assert senses.sweep_counts == {"slices": 5, "held": 0}


def test_the_sweep_holds_back_while_she_is_pressed_and_whole_screen_looks_have_none():
    model = three_floors()
    eyes, senses, asked = _senses(model, sweep=True)
    senses.sweep_hold = lambda s: True
    s = senses.look()
    assert asked == [] and senses.sweep_counts["held"] == 1 and len(s.rows) == 1
    senses.sweep_hold = None
    s = senses.look(wide=True)
    assert asked == [] and s.rows == [(500 - 300 * U, 500 + (int(540 * s.geo.hud_top) - 300) * U)]
    eyes2, off, asked2 = _senses(model)
    off.look()
    assert asked2 == []  # off by default


def test_the_memory_takes_every_look_and_replays_from_the_records_as_it_ran():
    model = three_floors()
    eyes, senses, _ = _senses(model, sweep=True)
    live = Memory(model)
    senses.listeners.append(live.on_look)
    records = []
    senses.listeners.append(lambda s: records.append({"t": s.t, "topic": "scene", **s.record()}))
    eyes.mobs = [pig(1000, 300), pig(900)]
    for i in range(6):
        if i == 3:
            eyes.mobs = [pig(900)]  # the one on B is gone
        senses.look(wide=i == 0)
        time.sleep(0.01)
    s = senses.look()
    assert cover_of(s) is not None and len(live.entries) == 2
    again = replay(records, model, (100, 100))
    last = [r for r in again if r["topic"] == "memory"][-1]
    assert last["t"] == s.t and last["list"] == live.record(s.t)["list"]
    assert [e["state"] for e in last["list"]] == ["seen", "seen"] and last["list"][0]["misses"] > 0  # B's: missed
