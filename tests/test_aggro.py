"""Her aggro belief (localization.aggro): who she hurt and chases her, kept through the looks that miss them and the
tracker's new ids, where each likeliest is, and who touched her; on a made-up map with tracks and events as data: no
game, no simulator (the simulator's measure is in the README: Her aggro belief)."""

import pytest

from gamelab.localization import aggro, mapdata
from gamelab.localization.aggro import (Aggro, BAR_GONE_S, CHASE_PASS, CHASE_S, DET_S, DOUBT_S, LATE_S, OTHER_DEATH_P,
                                        Chaser, _Rec, replay)
from gamelab.localization.monsters import MonsterTrack, walk_edges
from gamelab.localization.tracking import Event

P = 0  # the one platform: x 0 to 1000 at y 500
HORNY = "horny-mushroom"


def flat():
    tree = {
        "info": {"VRLeft": 0, "VRTop": 0, "VRRight": 1200, "VRBottom": 800},
        "foothold": {"1": {"1": {"1": {"x1": 0, "y1": 500, "x2": 1000, "y2": 500, "prev": 0, "next": 0}}}},
        "miniMap": {"width": 1200, "height": 800, "centerX": 0, "centerY": 0, "mag": 4},
    }
    return mapdata.parse(1, tree, "Test", {})


def track(tid, x, t, *, vx=0.0, bar=None, bar_at=None, state="live", seen=None, first=0.0, killed=False):
    return _Rec(tid, HORNY, P, float(x), 500.0, vx, 1 if vx > 0 else -1 if vx < 0 else 0, state, bar,
                t if seen is None else seen, first, killed, bar_at if bar_at is not None else (t if bar else None))


def hit(tid, t, bar=0.8, by="hers"):
    return Event("bar_hit", t, tid, {"bar": bar, "by": by})


def her(x=500.0):
    return (x, 500.0, P)


@pytest.fixture
def a():
    return Aggro(flat())


def test_her_hit_makes_a_chaser_and_another_s_does_not(a):
    a.update(10.0, her(), [track(1, 400, 10.0, bar=0.8), track(2, 600, 10.0, bar=0.7)],
             [hit(1, 10.0), hit(2, 10.0, by="other")])
    assert [c.track for c in a.chasers.values()] == [1]
    c = next(iter(a.chasers.values()))
    assert (c.kind, c.platform, c.x, c.bar, c.hits) == (HORNY, P, 400.0, 0.8, 1)


def test_a_bar_seen_down_late_on_a_monster_her_throw_had_in_its_path_is_hers(a):
    a.update(10.0, her(), [track(1, 700, 10.0)], [])
    a.throw(10.1, "right")  # (700 is 200 ahead of her: in a star's path)
    a.update(12.5, her(), [track(1, 690, 12.5, bar=0.6)], [hit(1, 12.5, by="other")])
    assert [(c.track, c.how) for c in a.chasers.values()] == [(1, "late")]
    a.update(12.6 + LATE_S, her(), [track(1, 690, 12.6 + LATE_S, bar=0.4)], [hit(1, 12.6 + LATE_S, 0.4, by="other")])
    assert next(iter(a.chasers.values())).hits == 1  # (no throw of hers that recent: another's)


def test_its_chase_lasts_3_to_4_minutes_from_her_last_hit(a):
    a.update(0.0, her(), [track(1, 400, 0.0, bar=0.8)], [hit(1, 0.0)])
    c = next(iter(a.chasers.values()))
    for t, p in ((CHASE_S[0] - 1, 1.0), ((CHASE_S[0] + CHASE_S[1]) / 2, 0.5)):
        a.update(t, her(), [track(1, 450, t, bar=0.8)], [])
        assert a.p(c, t) == pytest.approx(p)
    a.update(215.0, her(), [track(1, 450, 215.0, bar=0.6)], [hit(1, 215.0, 0.6)])  # (hit again: anew)
    a.update(CHASE_S[1] + 1, her(), [track(1, 450, CHASE_S[1] + 1, bar=0.6)], [])
    assert a.p(c, CHASE_S[1] + 1) == 1.0 and c.hits == 2
    a.update(215.0 + CHASE_S[1], her(), [track(1, 450, 215.0 + CHASE_S[1], bar=0.6)], [])
    assert not a.chasers and a.ended[-1][2] == "over"


def test_unseen_long_it_s_doubted_and_forgotten(a):
    a.update(0.0, her(), [track(1, 400, 0.0, bar=0.8)], [hit(1, 0.0)])
    c = next(iter(a.chasers.values()))
    a.update(1.0, her(), [], [])
    assert a.p(c, aggro.SEEN_S) == 1.0 and a.p(c, aggro.SEEN_S + aggro.LOST_S) == pytest.approx(0.3679, abs=1e-3)
    a.update(aggro.FORGET_S, her(), [], [])
    assert not a.chasers and a.ended[-1][2] == "gone"


def test_its_death_ends_it_and_another_s_kill_with_nobody_by_it_only_doubts_it(a):
    a.update(0.0, her(), [track(1, 400, 0.0, bar=0.8), track(2, 600, 0.0, bar=0.8)], [hit(1, 0.0), hit(2, 0.0)])
    a.update(1.0, her(), [track(1, 400, 1.0, bar=0.8, state="dead"), track(2, 600, 1.0, bar=0.8, state="dead")],
             [Event("died", 1.0, 1, {"cause": "hers"}), Event("died", 1.0, 2, {"cause": "other"})])
    assert len(a.chasers) == 1
    c = next(iter(a.chasers.values()))
    assert c.track is None and c.other and a.p(c, 1.0) == pytest.approx(OTHER_DEATH_P)
    a.update(2.0, her(), [track(3, 580, 2.0, bar=0.8, first=2.0)], [])  # (found again under a new id: about after all)
    assert c.track == 3 and not c.other
    a.update(3.0, her(), [track(3, 580, 3.0, bar=0.8, state="dead")], [Event("died", 3.0, 3, {"cause": "other", "by": 7})])
    assert not a.chasers  # (killed by another player by it)


def test_a_kill_s_track_dying_isn_t_about_and_dead_it_s_dropped(a):
    a.update(0.0, her(), [track(1, 400, 0.0, bar=0.3)], [hit(1, 0.0, 0.3)])
    a.update(0.5, her(), [track(1, 410, 0.5, bar=0.3, state="dying", killed=True)], [])
    assert not a.chasers_at(0.5)
    a.update(1.0, her(), [], [])
    assert not a.chasers and a.ended[-1][2] == "dead"


def test_unseen_it_walks_at_her_and_past_her_turning_at_its_walk_edges(a):
    a.update(0.0, her(800.0), [track(1, 100, 0.0, vx=-100.0, bar=0.8)], [hit(1, -5.0)])
    c = next(iter(a.chasers.values()))
    lo, hi = walk_edges(a.model.platform(P))
    g = a._guess(c, 0.9)  # (walking left on, till its next decision: at the edge it turns)
    assert lo <= g.x <= hi and g.x < 100
    g = a._guess(c, 3.0)  # (decided at her: right)
    assert g.way == 1 and g.x > 100
    far = a._guess(c, 20.0)  # (long since: about her, past her by CHASE_PASS and a decision's walk either way)
    assert far.x == pytest.approx(800.0) and far.way == 0
    assert far.lo == pytest.approx(800.0 - CHASE_PASS - a.walking.get(HORNY, aggro.DEFAULT_SPEED) * aggro.THINK_S)
    assert far.hi == pytest.approx(hi)


def test_lost_by_the_tracker_it_s_found_again_under_a_new_id(a):
    a.update(0.0, her(), [track(1, 400, 0.0, vx=100.0, bar=0.8)], [hit(1, 0.0)])
    key = next(iter(a.chasers))
    a.update(1.0, her(), [], [])  # (let go)
    assert a.chasers[key].track is None
    a.update(1.5, her(), [track(9, 470, 1.5, bar=0.8, first=1.5), track(8, 900, 1.5, bar=0.5, first=1.5)], [])
    assert a.chasers[key].track == 9 and a.chasers[key].tracks == [1, 9] and len(a.chasers) == 1


def test_a_new_hit_on_a_monster_found_again_is_the_lost_chaser_s(a):
    a.update(0.0, her(), [track(1, 400, 0.0, bar=0.8)], [hit(1, 0.0)])
    a.update(1.0, her(), [track(1, 400, 0.0, bar=0.8, state="hidden")], [])  # (hidden under her damage numbers)
    a.update(2.0, her(), [track(1, 400, 0.0, bar=0.8, state="hidden"), track(5, 450, 2.0, bar=0.5, first=2.0)],
             [hit(5, 2.0, 0.5)])
    assert len(a.chasers) == 1 and next(iter(a.chasers.values())).track == 5


def test_a_hit_on_another_monster_followed_all_along_is_another_chaser(a):
    a.update(0.0, her(), [track(1, 400, 0.0, bar=0.8), track(2, 430, 0.0)], [hit(1, 0.0)])
    a.update(1.0, her(), [track(1, 400, 0.0, bar=0.8, state="hidden"), track(2, 430, 1.0, bar=0.5)], [hit(2, 1.0, 0.5)])
    # (2 was followed beside it all along: another monster, not 1 found again)
    assert sorted(c.track for c in a.chasers.values()) == [1, 2]


def test_its_bar_gone_from_its_track_it_s_doubted_then_found_on_the_hurt_one_or_dropped(a):
    a.update(0.0, her(), [track(1, 400, 0.0, bar=0.8), track(2, 420, 0.0)], [hit(1, 0.0)])
    key = next(iter(a.chasers))
    t = BAR_GONE_S + 0.2  # (track 1 seen all along with no bar read: an unhurt monster under it now)
    a.update(t, her(), [track(1, 410, t, bar=0.8, bar_at=0.0), track(2, 430, t)], [])
    c = a.chasers[key]
    assert c.track is None and c.doubt == t and a.p(c, t) < 1.0
    a.update(t + 0.5, her(), [track(1, 410, t + 0.5, bar=0.8, bar_at=0.0), track(2, 430, t + 0.5, bar=0.8)], [])
    assert c.track == 2 and c.doubt is None  # (the hurt one, the bar on it)
    b = Aggro(flat())
    b.update(0.0, her(), [track(1, 400, 0.0, bar=0.8)], [hit(1, 0.0)])
    b.update(t, her(), [track(1, 410, t, bar=0.8, bar_at=0.0)], [])
    b.update(t + DOUBT_S + 0.1, her(), [track(1, 410, t + DOUBT_S + 0.1, bar=0.8, bar_at=0.0)], [])
    assert not b.chasers and b.ended[-1][2] == "phantom"


def test_near_answers_the_chasers_by_her_and_as_the_planner_s_tracks(a):
    a.update(0.0, her(), [track(1, 450, 0.0, vx=100.0, bar=0.8), track(2, 900, 0.0, bar=0.8)],
             [hit(1, 0.0), hit(2, 0.0)])
    near = a.near(0.0, 500.0, 60.0, platform=P)
    assert [g.chaser.track for g in near] == [1]
    mt = near[0].track(100.0)
    assert isinstance(mt, MonsterTrack) and mt.platform == P and mt.x == 450.0 and mt.vx == 100.0
    assert mt.track == 1 and mt.trust == DET_S
    far = a.chasers_at(30.0)[0].track(100.0)  # (unseen long: about her, its heading not known)
    assert far.vx is None and far.unsure > 1.0


def test_the_toucher_is_the_one_whose_body_could_reach_hers_its_side_from_her_knock(a):
    a.update(0.0, her(500.0), [track(1, 530, 0.0, bar=0.8), track(2, 700, 0.0)], [hit(1, 0.0)])
    a.update(0.3, her(500.0), [track(1, 530, 0.3, bar=0.8), track(2, 700, 0.3)],
             [Event("touch", 0.3, None, {"kind": "damage", "at": 0.25})])
    k = a.toucher()
    assert (k.kind, k.track, k.by, k.side, k.damage) == (HORNY, 1, "chaser", 1, True)
    a.update(0.6, her(430.0), [track(1, 530, 0.6, bar=0.8), track(2, 700, 0.6)], [])  # (knocked left: it was right)
    assert a.toucher().knock == -1 and a.toucher().side == 1
    a.update(5.0, her(100.0), [track(1, 530, 5.0, bar=0.8)], [Event("touch", 5.0, None, {"kind": "miss", "at": 4.9})])
    assert a.toucher().kind is None and a.toucher().by == "none"  # (nothing near her)


def test_replay_runs_it_on_telemetry_records():
    recs = [
        {"t": 0.0, "topic": "scene", "her": {"x": 500.0, "y": 500.0, "platform": P}},
        {"t": 0.0, "topic": "tracks", "list": [{"id": 1, "kind": HORNY, "state": "live", "x": 600.0, "y": 500.0,
                                                 "vx": 0.0, "facing": -1, "platform": P, "bar": None, "hits": 0,
                                                 "seen": 3}]},
        {"t": 0.1, "topic": "attack", "kind": HORNY, "x": 600.0, "platform": P},  # (no side: the target's)
        {"t": 0.6, "topic": "tracks", "list": [{"id": 1, "kind": HORNY, "state": "live", "x": 590.0, "y": 500.0,
                                                 "vx": -100.0, "facing": -1, "platform": P, "bar": 0.7, "hits": 1,
                                                 "seen": 8}]},
    ]
    looks = [(t, len(b.chasers)) for t, b in replay(recs, flat())]
    assert looks == [(0.0, 0), (0.6, 1)]


def test_a_look_made_up_from_another_is_not_one(a):
    from types import SimpleNamespace
    s = SimpleNamespace(what_if=True, t=0.0, where=None, tracks=[track(1, 400, 0.0, bar=0.8)], events=[hit(1, 0.0)])
    a.on_look(s)
    assert not a.chasers and not a.path


def test_its_record_is_plain_data(a):
    import json
    a.update(0.0, her(), [track(1, 400, 0.0, bar=0.8)], [hit(1, 0.0)])
    rec = a.record(1.0)
    assert json.loads(json.dumps(rec))["chasers"][0]["track"] == 1
    assert isinstance(next(iter(a.chasers.values())), Chaser)
    assert DET_S > 0
