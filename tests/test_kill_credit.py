"""Her kill's monster by Magic Claw's own search (localization.tracking's band order, her Shape's KillCredit: the
magician's, from 2026-11-02), and the hunt following the tracker's corrections (behavior.ledger's `moved`,
localization.memory, behavior.grind's credit): on made-up frames, as tests/test_tracking.py makes them."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from gamelab import character
from gamelab.localization import tracking
from gamelab.localization.tracking import DEAD, DYING, LIVE, MonsterTracker
from test_tracking import FPS, HER, THIRD, run, sighting, view

MAGE = character.load("magician").shape


def _seen(tracker, monsters, seconds=THIRD / FPS, start=100.0):
    run(tracker, [(monsters, view())] * round(seconds * FPS), start=start)


def _victims(events):
    return [e.track for e in events if e.kind == "victim"]


def test_her_search_picks_by_band_over_a_nearer_bystander_it_cant_reach():
    """A pig 70 below her and 60 ahead is the nearest in the thief's box (its 75 down), but no band of her search
    reaches it (its windows sink under her feet only from 250 ahead): the one on her level 120 ahead, in band 1, is her
    kill. The thief's stars, the same frames: the nearer. (R1 00:11:50.7: her kill went to a bystander on platform 9,
    about 50 below her and 70 across.)"""
    seen = [sighting(1120, pose="stand"), sighting(1060, pose="stand", y=HER[1] + 70, platform=2)]
    for shape, want in ((MAGE, [1]), (character.THIEF_SHAPE, [2])):
        tracker = MonsterTracker(shape=shape)
        _seen(tracker, seen)
        tracker.throw(100.4, "right")
        tracker.gain(100.6, "pig")
        events, _ = run(tracker, [(seen, view())], start=100.6)
        assert _victims(events) == want, shape.order


def test_one_straight_above_her_is_never_her_search_s():
    """A pig 100 above her, 10 ahead, is inside the thief's box (110 up) and nearest; her search never picks within its
    first 50 ahead: the one 200 ahead on her level is her kill (R1 00:14:42.4's kind of bystander)."""
    seen = [sighting(1010, pose="stand", y=HER[1] - 100, platform=2), sighting(1200, pose="stand")]
    tracker = MonsterTracker(shape=MAGE)
    _seen(tracker, seen)
    tracker.throw(100.4, "right")
    tracker.gain(100.6, "pig")
    events, _ = run(tracker, [(seen, view())], start=100.6)
    assert _victims(events) == [2]


@pytest.mark.parametrize("aim", [None, "track", "place"])
def test_in_her_first_band_the_oldest_comes_first_whatever_she_aimed_at(aim):
    """Two pigs both in her search's first band holding any (120 and 105 ahead: band 1): the older track (the client
    takes the one that entered the map first) is her kill, though the other is nearer and her attack choice aimed the
    cast at it (told by its track, or by where she saw it: a replay's aim). The aim is found all the same, and counted
    against the binding."""
    seen = [sighting(1120, pose="stand"), sighting(1105, pose="stand")]
    tracker = MonsterTracker(shape=MAGE)
    _seen(tracker, seen)
    told = {None: None, "track": (2, "pig", 1105.0, 1), "place": (None, "pig", 1109.0, 1)}[aim]
    tracker.throw(100.4, "right", "magic_claw", told)
    assert tracker.throws[-1].aim == (None if aim is None else 2)
    tracker.gain(100.6, "pig")
    events, _ = run(tracker, [(seen, view())], start=100.6)
    assert _victims(events) == [1]
    assert tracker.counts.get("aim elsewhere", 0) == (0 if aim is None else 1)
    assert "aim bound" not in tracker.counts


def test_her_first_band_beats_an_older_one_farther_on_but_in_the_tie_s_bands():
    """An older pig two bands past a newer one (band 3 against 1): the client's search picks the first band's (tie 0).
    With a tie of 2 bands (the tracker's x error), the two are one band's: the older one."""
    seen = [sighting(1150, pose="stand"), sighting(1105, pose="stand")]
    for tie, want in ((0, [2]), (2, [1])):
        tracker = MonsterTracker(shape=replace(MAGE, kill=replace(MAGE.kill, tie=tie)))
        _seen(tracker, seen)
        tracker.throw(100.4, "right", "magic_claw", (2, "pig", 1105.0, 1))
        tracker.gain(100.6, "pig")
        events, _ = run(tracker, [(seen, view())], start=100.6)
        assert _victims(events) == want, tie
        assert tracker.counts.get("aim bound" if want == [2] else "aim elsewhere") == 1, tie


def test_an_aim_told_by_a_track_that_s_dead_or_another_kind_is_none_not_one_by_place():
    """Her cast aimed at a track the tracker has dead (her own dead: 18 such casts in about 14 minutes of R1-R3) or of
    another kind: aimed at none, not at the nearest of that kind by place; a track no longer kept (P16's id churn) or
    none told: by place."""
    tracker = MonsterTracker(shape=MAGE)
    _seen(tracker, [sighting(1120, pose="stand"), sighting(1150, pose="stand")])
    tracker.tracks[1].state = DEAD
    assert tracker._aim_of((1, "pig", 1120.0, 1), 100.4) is None
    assert tracker._aim_of((2, "snail", 1150.0, 1), 100.4) is None
    assert tracker._aim_of((99, "pig", 1140.0, 1), 100.4) == 2
    assert tracker._aim_of((None, "pig", 1125.0, 1), 100.4) == 2
    assert tracker._aim_of((None, "pig", 1125.0 + tracking.AIM_DX + 30, 1), 100.4) is None


def test_the_thief_s_stars_ignore_the_aim():
    """The thief's kill goes by his stars' order whatever the aim told (his goldens unchanged)."""
    seen = [sighting(1105, pose="stand"), sighting(1120, pose="stand")]
    tracker = MonsterTracker()
    _seen(tracker, seen)
    tracker.throw(100.4, "right", None, (2, "pig", 1120.0, 1))
    tracker.gain(100.6, "pig")
    events, _ = run(tracker, [(seen, view())], start=100.6)
    assert _victims(events) == [1]
    assert not any(k.startswith("aim ") for k in tracker.counts)  # (his counts as before)


def test_the_hunt_s_throw_message_with_an_aim_reads_as_before_to_an_older_service():
    """The hunt tells a throw with her attack's id and her aim, the message's fifth item: a service from before
    (40ba190's Service._take, which reads msg[1..3]) takes the throw as ever, the aim aside; this one passes it on; a
    throw told with neither is the three items it was."""
    from gamelab.localization.service import Service, ServiceClient
    hunt = ServiceClient("tour", acts=True, throws=True, map_id=2, start=False, ensure=None)
    sent = []
    hunt._send = lambda *msg: sent.append(msg)
    hunt.throw(100.4, "right", "magic_claw", (7, "horny-mushroom", -60.0, 15))
    hunt.throw(101.4, "left")
    hunt.throw(102.4, "left", None, (None, "slime", 30.0, 9))
    assert [len(m) for m in sent] == [5, 3, 5]
    old, new = [], []

    def read_before(msg):  # (40ba190's line, as it was)
        old.append((msg[1], msg[2], msg[3] if len(msg) > 3 else None))
    svc = Service.__new__(Service)
    svc.runner = SimpleNamespace(step=SimpleNamespace(throw=lambda *a: new.append(a)))
    for msg in sent:
        read_before(msg)
        svc._take(None, msg, None)
    assert old == [(100.4, "right", "magic_claw"), (101.4, "left", None), (102.4, "left", None)]
    assert new == [(100.4, "right", "magic_claw", (7, "horny-mushroom", -60.0, 15)), (101.4, "left", None, None),
                   (102.4, "left", None, (None, "slime", 30.0, 9))]


def test_her_hidden_victim_in_an_earlier_band_beats_a_fresh_bystander_farther_on():
    """Her victim hidden under her claw (out of sight longer than FRESH_S at the gain) in band 1, a bystander seen in
    the gain's frame in band 8: the client picks band 1, and so does her binding; the thief's order (seen lately
    first) takes the bystander. (R1's four 'hidden victim' cases: the aimed, hurt victim in an earlier band than the
    fresh bystander that took the gain.)"""
    for shape, want in ((MAGE, [1]), (character.THIEF_SHAPE, [2])):
        tracker = MonsterTracker(shape=shape)
        _seen(tracker, [sighting(1120, pose="stand"), sighting(1250, pose="stand")])
        _seen(tracker, [sighting(1250, pose="stand")], 0.4, start=100.0 + THIRD / FPS)  # the near one hides
        tracker.throw(100.6, "right")
        tracker.gain(100.85, "pig")
        events, _ = run(tracker, [([sighting(1250, pose="stand")], view())], start=100.85)
        assert _victims(events) == want, shape.order


def test_her_victim_stops_walking_as_her_hit_lands_lands_s_after_the_key():
    """No star flies: the monster bound walks on until her hit shows on it, her KillCredit's lands_s after the key (the
    victims' walks stopped a median 0.49-0.50 s after it on the recordings), not the thief's 0.75 s and the flight."""
    seen = [sighting(1120, pose="stand")]
    star = 100.4 + tracking.STAR_LANDS_S + 120 / tracking.STAR_FLIGHT
    for shape, lands in ((MAGE, 100.4 + MAGE.kill.lands_s), (character.THIEF_SHAPE, star)):
        tracker = MonsterTracker(shape=shape)
        _seen(tracker, seen)
        tracker.throw(100.4, "right")
        tracker.gain(100.6, "pig")
        run(tracker, [(seen, view())], start=100.6)
        assert tracker.tracks[1].lands == pytest.approx(lands, abs=0.02), shape.order


def test_her_kill_let_go_goes_on_one_in_her_search_seen_dying_since():
    """Her gain bound to the pig 120 ahead, seen alive REBIND_S on: let go, and it goes on the one in her search whose
    death was seen since (booked another's then: it had no track among her picks, a new one after the bind), not left
    unexplained (BAND_LET_GO_DEAD; R3 00:49:59.9's kind: her victim's death seen and booked 'other')."""
    tracker = MonsterTracker(shape=MAGE)
    near = sighting(1120, pose="stand")
    _seen(tracker, [near])
    key, g = 100.4, 100.6
    tracker.throw(key, "right")
    tracker.gain(g, "pig")
    events, _ = run(tracker, [([near], view())], start=g)  # (bound: the one 120 ahead)
    events += run(tracker, [([near, sighting(1220, pose="stand")], view())] * 4, start=g + 1 / FPS)[0]  # a new one
    tracker.seen_dying(2, g + 5 / FPS + 0.3)
    events += run(tracker, [([near], view())], start=g + 5 / FPS + 0.3)[0]
    assert [(e.track, e.data["cause"]) for e in events if e.kind == "died"] == [(2, "other")]
    events += run(tracker, [([near], view())] * FPS, start=g + 0.8)[0]
    assert [(e.kind, e.track) for e in events if e.kind in ("victim", "let_go")] == [
        ("victim", 1), ("let_go", 1), ("victim", 2)]
    assert tracker.tracks[2].cause == "hers" and tracker.counts["died other"] == 0
    old = tracking.BAND_LET_GO_DEAD
    tracking.BAND_LET_GO_DEAD = False
    try:
        tracker = MonsterTracker(shape=MAGE)
        _seen(tracker, [near])
        tracker.throw(key, "right")
        tracker.gain(g, "pig")
        run(tracker, [([near], view())], start=g)
        run(tracker, [([near, sighting(1220, pose="stand")], view())] * 4, start=g + 1 / FPS)
        tracker.seen_dying(2, g + 5 / FPS + 0.3)
        run(tracker, [([near], view())], start=g + 5 / FPS + 0.3)
        run(tracker, [([near], view())] * FPS, start=g + 0.8)
        assert tracker.tracks[2].cause == "other" and tracker.counts["gains unexplained"] == 1
    finally:
        tracking.BAND_LET_GO_DEAD = old


def test_the_search_s_band_is_attack_reach_s_with_her_constants():
    """character.Search.band (localization's, which can't import the attack layer) is attack.reach.search_band with
    her constants set (attack.magic), on a grid of bodies around her."""
    from gamelab.attack import magic, reach
    mage = character.load("magician")
    s = mage.attacks["skill"].geometry.search
    with magic.swapped(mage):
        for ahead in range(-40, 360, 7):
            for dy in range(-140, 141, 9):
                for half, top in ((28.0, 51.0), (34.0, 38.0), (19.0, 24.0)):
                    assert s.band(ahead, dy, half, top) == reach.search_band(*reach.body_box(ahead, dy, top, half)), \
                        (ahead, dy, half, top)


# --- the hunt following the tracker's corrections ---------------------------------------------------------------

def _hunter(shape):
    return SimpleNamespace(exp_gains=0, senses=SimpleNamespace(shape=shape), model=None)


def _look(t, tracks, monsters=()):
    return SimpleNamespace(t=t, tracks=tracks, monsters=list(monsters), searched=True, where=None, platform=None,
                           players=[])


def test_a_kill_put_on_another_monster_is_kept_in_moved_for_her_alone():
    """The ledger follows a kill onto the track the tracker moved its gain to (as before, both characters); with her
    kill credit it also keeps the move (`moved`: the old track, platform and x), the thief's ledger not."""
    from gamelab.behavior.ledger import Ledger
    from gamelab.localization.tracking import Track
    from gamelab.perception.mobs import Mob

    mob = Mob("pig", 0, 0, 30, 30, 0.9)

    def track(tid, x, platform, state=LIVE, gain=None):
        return Track(tid, "pig", platform, x, 500.0, mob, 0.0, 10.0, state=state, gain=gain)

    for shape, follows in ((MAGE, True), (character.THIEF_SHAPE, False)):
        h = _hunter(shape)
        h.gains_told = lambda n: [(10.2, "pig")] * n
        ledger = Ledger(h)
        assert ledger.follows is follows
        h.exp_gains = 1
        ledger.on_look(_look(10.4, [track(5, 400.0, 3, DYING, 10.2), track(6, 420.0, 4)]))
        (k,) = ledger.kills
        assert (k.track, k.platform) == (5, 3)
        ledger.on_look(_look(11.6, [track(5, 470.0, 3), track(6, 420.0, 4, DYING, 10.2)]))
        assert (k.track, k.platform) == (6, 4)
        assert [(m[2], m[3]) for m in ledger.moved] == ([(5, 3)] if follows else [])


def test_her_memory_and_the_grind_s_credit_follow_a_moved_kill():
    """Her memory: the monster the kill was first put on, alive again (its track not dead now), the one it's on now
    killed, told as the one kill moved ('moved', not a second 'killed'); patrol.Belief: the kill's time once, the new
    one's point emptied at it, the old one's not; the grind's credit by platform: the kill (and its EXP) moved to the
    platform it's on now."""
    from gamelab.behavior import patrol
    from gamelab.behavior.grind import Grind
    from gamelab.localization.memory import Memory
    from test_memory import mon, three_floors, band, A, B

    kills, moved, dead = [], [], {}
    model = three_floors()
    m = Memory(model, kills=lambda: kills, dead=lambda: dead, moved=lambda: moved)
    belief = patrol.Belief(model, [patrol.Point(0, A, 1300.0, "pig", 10.0), patrol.Point(1, B, 1100.0, "pig", 10.0)])
    first, real = replace(mon(1300, A), track=5), replace(mon(1100, B), track=6)
    m.update(0.0, [first, real], band(0.0))
    k = SimpleNamespace(t=0.5, kind="pig", platform=A, x=1300.0, track=5)
    kills.append(k)
    dead[5] = 0.5
    m.update(0.5, [first, real], band(0.5))
    assert {e.monster.track: e.state for e in m.entries} == {5: "killed", 6: "seen"}
    belief._take_events(0.5, m)
    assert belief.kills == [0.5] and belief.killed == {0: 0.5} and belief.empty_since == {0: 0.5}
    moved.append((1.6, k, 5, A, 1300.0))
    k.track, k.platform, k.x = 6, B, 1100.0
    del dead[5]
    dead[6] = 1.6
    m.update(1.6, [first], band(1.6))
    assert {e.monster.track: e.state for e in m.entries} == {5: "seen", 6: "killed"}
    assert m.counts["killed"] == 1 and m.counts["moved"] == 1
    assert [(what, e.monster.track) for what, _, e in m.events if what != "new"] == [("killed", 5), ("moved", 6)]
    belief._take_events(1.6, m)
    assert belief.kills == [0.5] and belief.killed == {1: 0.5} and belief.empty_since == {1: 0.5}

    g = Grind.__new__(Grind)  # (its bookkeeping alone: the hunt loop's own is Hunt._book_kills's, left out)
    g.h, g.credit, g._credited, g._moves_seen = SimpleNamespace(ledger=SimpleNamespace(kills=[], moved=[])), {}, {}, 0
    g.tour, g._avg_exp = SimpleNamespace(exp={"pig": 10.0}), lambda: 5.0
    k2 = SimpleNamespace(t=10.0, kind="pig", named="pig", platform=3, ground=3, x=400.0, placed=True)
    g.h.ledger.kills.append(k2)
    book = Grind._book_kills  # (Hunt._book passes each look's new kills, none too; Hunt._book_after the loop's last)
    import gamelab.behavior.hunt as hunt
    saved = hunt.Hunt._book_kills
    hunt.Hunt._book_kills = lambda self, s, kills: None
    try:
        book(g, SimpleNamespace(t=10.1), [k2])
        assert g.credit == {3: [1, 10.0]}
        k2.platform = 4
        g.h.ledger.moved.append((10.9, k2, 5, 3, 400.0))
        book(g, SimpleNamespace(t=11.0), [])
        assert g.credit == {3: [0, 0.0], 4: [1, 10.0]}
        book(g, SimpleNamespace(t=11.5), [])  # (each move once)
        assert g.credit == {3: [0, 0.0], 4: [1, 10.0]}
    finally:
        hunt.Hunt._book_kills = saved
