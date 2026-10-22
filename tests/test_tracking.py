"""Monsters followed from frame to frame (localization.tracking), on made-up 30 Hz sequences of sightings: one
id across walks and turns, facing from the first frame, off screen against dead, a monster hidden under its
damage numbers kept, and who killed it."""

import pytest

from gamelab.localization import tracking
from gamelab.localization.monsters import Monster
from gamelab.localization.tracking import (DEAD, DYING, HIDDEN, LIVE, OFFSCREEN, MonsterTracker, View)
from gamelab.perception.mobs import Mob
from gamelab.screen import UNITS_PER_HALF_PX as U

FPS = 30
KINDS = ("pig", "slime", "snail", "orange-mushroom")
HER = (1000.0, 500.0)  # her feet on the map; on screen at column 480, row 300 (half size)


def view(her_x: float = HER[0]) -> View:
    """What the frame shows with her at map x `her_x` (the camera keeps her at column 480)."""
    return View(her_x - 480 * U, her_x + 480 * U, HER[1] - 300 * U, HER[1] + 175 * U, 960, 475)


def sighting(x: float, facing: int = 1, pose: str = "move", bar: float | None = None, kind: str = "pig",
             her_x: float = HER[0], y: float = HER[1], platform: int | None = 1) -> Monster:
    """A monster seen with its feet at map (x, y), as Scene.place puts it, its picture on screen."""
    sx, feet = round(480 + (x - her_x) / U), round(300 + (y - HER[1]) / U)
    mob = Mob(kind, sx, feet - 16, 48, 32, 0.9, float(sx), float(feet), facing, pose, "move/1", f"{kind}_move1")
    return Monster(mob, x, y, platform, bar, facing, pose)


def run(tracker: MonsterTracker, frames, refind: bool = True, start: float = 100.0):
    """Feed the tracker a frame at a time: each frame's sightings (None: none). With `refind`, each tracked
    monster is handed back as re-found, as the world step's windows find them; the rest as a search found
    them. Returns every event and the tracks' (id, state) after each frame."""
    events, states = [], []
    for k, (seen, v) in enumerate(frames):
        t = start + k / FPS
        refound, found = {}, []
        for m in seen or []:
            near = [tr for tr in tracker.live() if refind and tr.kind == m.kind
                    and abs(tracker.predicted(tr, t) - m.x) < 20]
            if near:
                refound[min(near, key=lambda tr: abs(tracker.predicted(tr, t) - m.x)).id] = m
            else:
                found.append(m)
        events += tracker.update(t, refound, found, v, HER)
        states.append({tr.id: tr.state for tr in tracker.tracks.values()})
    return events, states


def walk(x0: float, v: float, seconds: float, start: float = 0.0, **kw) -> list:
    """A pig walking at v units/s from x0, a frame at a time."""
    return [([sighting(x0 + v * (start + k / FPS), facing=1 if v > 0 else -1, **kw)], view())
            for k in range(round(seconds * FPS))]


@pytest.mark.parametrize("refind", [True, False])
def test_one_id_across_a_walk_and_a_turn_facing_known_from_the_first_frame(refind):
    tracker = MonsterTracker()
    there = walk(800, 125, 1.5)
    back = walk(800 + 125 * 1.5, -125, 1.5)
    events, states = run(tracker, there + back, refind=refind)
    assert [e.kind for e in events] == ["appeared"]
    assert all(list(s) == [1] for s in states)
    tr = tracker.tracks[1]
    assert tr.state == LIVE and tr.facing == -1 and tr.seen == 90
    first = MonsterTracker()
    run(first, there[:1])
    assert first.tracks[1].facing == 1  # from the template that matched, on its first frame


def test_its_speed_within_a_fifth_after_half_a_second_and_turning_with_it():
    tracker = MonsterTracker()
    vx = []
    for k, frame in enumerate(walk(800, 125, 1.5) + walk(800 + 125 * 1.5, -125, 0.5)):
        run(tracker, [frame], start=100.0 + k / FPS)
        vx.append(tracker.tracks[1].vx)
    assert vx[15] == pytest.approx(125, rel=0.2)
    assert vx[45] < -100  # the frame after it turned: the move frame faces the other way


def test_a_pose_sets_its_speed_only_while_its_sightings_agree(monkeypatch):
    # In game the pose is the matched template's, wrong for about a fifth of the sightings and for up to 1.4 s
    # on end: a pig standing in move frames isn't taken to walk, nor one walking in stand frames to stand.
    standing = [([sighting(900.0, facing=1, pose="move")], view()) for _ in range(30)]
    tracker = MonsterTracker()
    run(tracker, standing)
    tr = tracker.tracks[1]
    assert abs(tr.vx) < 5 and abs(tr.x - 900) < 1 and abs(tracker.predicted(tr, 101.5) - 900) < 5
    tracker = MonsterTracker()
    run(tracker, walk(800, 125, 0.5) + walk(800 + 125 * 0.5, 125, 0.5, pose="stand"))
    assert tracker.tracks[1].vx > 100  # it went on walking
    # One setting off in move frames walks its speed once it's moved FADE_MOVE (10 units: 0.08 s for a pig).
    tracker = MonsterTracker()
    vx = []
    for k, frame in enumerate(standing + walk(900, 125, 0.5, start=0.0)):
        run(tracker, [frame], start=100.0 + k / FPS)
        vx.append(tracker.tracks[1].vx)
    assert vx[29 + 4] == 125.0 and vx[29 + 1] < 60
    # Without the gate the pose alone decides, as before.
    monkeypatch.setattr(tracking, "POSE_GATE_S", None)
    tracker = MonsterTracker()
    run(tracker, standing)
    assert tracker.tracks[1].vx == 125.0


def test_two_walking_past_each_other_keep_their_ids():
    tracker = MonsterTracker()
    frames = [([sighting(700 + 125 * k / FPS), sighting(1000 - 125 * k / FPS, facing=-1)], view())
              for k in range(60)]
    run(tracker, frames, refind=False)
    a, b = tracker.tracks[1], tracker.tracks[2]
    assert (a.facing, b.facing) == (1, -1)
    assert a.x == pytest.approx(700 + 125 * 59 / FPS, abs=3) and b.x == pytest.approx(1000 - 125 * 59 / FPS, abs=3)


def test_one_that_walks_off_screen_is_offscreen_not_dead_and_back_with_its_id():
    tracker = MonsterTracker()
    out = walk(1550, 125, 1.0)                                         # walks off the right side
    gone = [(None, view())] * (6 * FPS)                                # 6 s off screen
    back = [([sighting(1660, facing=-1)], view())] * 10                # comes back in
    events, states = run(tracker, out + gone + back)
    kinds = [e.kind for e in events]
    assert kinds == ["appeared", "left", "back"]
    assert states[len(out) + 30][1] == OFFSCREEN and "died" not in kinds
    assert list(tracker.tracks) == [1] and tracker.tracks[1].state == LIVE


def test_the_camera_moving_away_leaves_it_offscreen():
    tracker = MonsterTracker()
    frames = [([sighting(500, pose="stand", her_x=1000 + 4 * k)], view(1000 + 4 * k)) for k in range(30)]
    frames += [(None, view(1120 + 10 * k)) for k in range(60)]  # she runs right: it drops off the left
    events, _ = run(tracker, frames)
    assert [e.kind for e in events] == ["appeared", "left"] and tracker.tracks[1].state == OFFSCREEN


@pytest.mark.parametrize("hidden_s", [1.0, 2.0, 3.0])
def test_hidden_under_its_damage_numbers_for_up_to_3_s_it_keeps_its_id(hidden_s):
    tracker = MonsterTracker()
    before = [([sighting(1200, pose="stand", bar=1.0 if k < 10 else 0.6)], view()) for k in range(20)]
    hidden = [(None, view())] * round(hidden_s * FPS)
    after = [([sighting(1205, pose="stand", bar=0.6)], view())] * 10
    events, states = run(tracker, before + hidden + after)
    assert states[len(before) + 15][1] == HIDDEN
    assert [e.kind for e in events if e.kind != "bar_hit"] == ["appeared"]
    assert list(tracker.tracks) == [1] and tracker.tracks[1].state == LIVE


def test_one_gone_in_plain_view_with_no_exp_of_hers_was_killed_by_someone_else():
    tracker = MonsterTracker()
    hurt = [([sighting(1200, pose="stand", bar=1.0 if k < 10 else 0.3)], view()) for k in range(30)]
    events, states = run(tracker, hurt + [(None, view())] * (5 * FPS))
    died = [e for e in events if e.kind == "died"]
    assert [(e.track, e.data["cause"], e.data["hurt"]) for e in died] == [(1, "other", True)]
    assert died[0].t - (100 + 29 / FPS) == pytest.approx(tracking.HIDE_S, abs=0.05)
    bar = [e for e in events if e.kind == "bar_hit"]
    assert [e.data["by"] for e in bar] == ["other"]  # her throws are known, and none went its way
    assert tracker.counts["died other"] == 1 and tracker.counts["died hers"] == 0


def test_one_she_threw_at_that_vanished_as_her_exp_went_up_was_hers():
    tracker = MonsterTracker()
    t0 = 100.0
    standing = [([sighting(1200, pose="stand", bar=None if k < 10 else 0.4)], view()) for k in range(30)]
    events, _ = run(tracker, standing[:5], start=t0)
    tracker.throw(t0 + 5 / FPS, "right")             # her star goes its way ...
    more, _ = run(tracker, standing[5:], start=t0 + 5 / FPS)
    tracker.throw(t0 + 1.0, "right")                 # ... and another one kills it:
    tracker.gain(t0 + 1.4)                            # her EXP goes up
    fading = [([sighting(1200, pose="", bar=0.4)], view())] * 24  # still seen fading for 0.8 s
    states = []
    for k, frame in enumerate(fading + [(None, view())] * 60):
        ev, st = run(tracker, [frame], start=t0 + 1.4 + k / FPS)
        more += ev
        states += st
    assert states[5][1] == DYING  # fading, while her gain waits for it to go
    died = [e for e in more if e.kind == "died"]
    assert [(e.data["cause"], e.data["at"]) for e in died] == [("hers", round(t0 + 1.4, 3))]
    assert [e.data["by"] for e in more if e.kind == "bar_hit"] == ["hers"]
    assert tracker.tracks[1].state == DEAD and tracker.tracks[1].cause == "hers"


def test_a_throw_aims_at_the_monsters_a_star_can_pick_and_none_beyond():
    # A star picks a body whose near side is within 400 (attack.reach.STAR_RANGE): a pig's middle up to 434
    # (STAR_REACH). The tracker allows 40 more for what moves before the pick (THROW_REACH, a number: localization
    # doesn't import attack); a pig 520 off isn't in a throw's path, and her EXP isn't put on it.
    from gamelab.attack import reach
    assert reach.STAR_REACH <= tracking.THROW_REACH <= reach.STAR_REACH + 60
    tracker = MonsterTracker()
    run(tracker, [([sighting(HER[0] + dx, pose="stand") for dx in (300, 460, 520)], view())] * 6)
    tracker.throw(100.0 + 6 / FPS, "right")
    assert sorted(round(tracker.tracks[i].x - HER[0]) for i in tracker.throws[-1].aimed) == [300, 460]


def test_her_kill_is_the_one_that_faded_not_the_one_hiding_under_its_damage_numbers():
    tracker = MonsterTracker()
    t0 = 100.0
    both = [([sighting(1150, pose="stand", bar=0.7), sighting(1300, pose="stand", bar=0.3)], view())] * 10
    run(tracker, both, start=t0)
    tracker.throw(t0 + 0.4, "right")
    # The near one is hit and hides under its damage numbers; the far one dies and fades for 0.8 s.
    hiding = [([sighting(1300, pose="", bar=0.3)], view())] * 24 + [(None, view())] * 30
    tracker.gain(t0 + 0.5)
    events, _ = run(tracker, hiding, start=t0 + 0.5)
    events += run(tracker, [([sighting(1152, pose="stand", bar=0.5)], view())] * 5, start=t0 + 2.3)[0]
    died = [e for e in events if e.kind == "died"]
    assert [(e.track, e.data["cause"]) for e in died] == [(2, "hers")]
    assert tracker.tracks[1].state == LIVE  # back from hiding, the same one


def test_played_by_hand_her_exp_puts_the_kill_on_one_in_reach_that_went():
    tracker = MonsterTracker(throws=False)
    t0 = 100.0
    run(tracker, [([sighting(1250, pose="stand")], view())] * 10, start=t0)
    tracker.gain(t0 + 0.4)
    events, _ = run(tracker, [(None, view())] * 60, start=t0 + 0.4)
    assert [(e.kind, e.data.get("cause")) for e in events] == [("died", "hers")]


def test_a_match_on_scenery_for_a_frame_or_two_is_no_death():
    tracker = MonsterTracker()
    events, _ = run(tracker, [([sighting(1200, pose="stand")], view())] * 2 + [(None, view())] * (5 * FPS))
    assert [e.kind for e in events] == ["appeared"] and not tracker.tracks
    assert tracker.counts["flickers"] == 1


def test_tracks_dont_pile_up():
    tracker = MonsterTracker()
    frames = []
    for n in range(40):  # a monster every 2 s (each kind every 8 s), killed by someone else after 1 s on screen
        frames += [([sighting(1000, pose="stand", kind=KINDS[n % 4])], view())] * FPS + [(None, view())] * FPS
    run(tracker, frames + [(None, view())] * (10 * FPS))
    assert tracker.counts["tracks"] == 40 and tracker.counts["died other"] == 40 and not tracker.tracks


def test_nothing_is_judged_while_she_isnt_placed():
    tracker = MonsterTracker()
    run(tracker, [([sighting(1200, pose="stand")], view())] * 10)
    events, states = run(tracker, [(None, None)] * (5 * FPS), start=101.0)
    assert events == [] and states[-1] == {1: LIVE}


# --- a search's sightings merged frames after their own (merge: the searches run off the frame loop) --------

def test_a_late_sighting_of_a_track_seen_since_is_that_track_and_tells_it_nothing():
    tracker = MonsterTracker()
    run(tracker, walk(900, 125, 0.5))  # followed, re-found every frame to t 100.47
    tr = tracker.tracks[1]
    before = (tr.x, tr.last_seen, tr.seen)
    events = tracker.merge(100.3, [sighting(900 + 125 * 0.3)], set(), view(), view(), 100.5)
    assert events == [] and list(tracker.tracks) == [1] and (tr.x, tr.last_seen, tr.seen) == before


def test_a_late_sighting_of_one_hidden_since_brings_it_back_as_of_its_frame():
    tracker = MonsterTracker()
    run(tracker, [([sighting(1200, pose="stand")], view())] * 10 + [(None, view())] * 15)  # hidden from 100.6
    tr = tracker.tracks[1]
    assert tr.state == HIDDEN
    events = tracker.merge(100.7, [sighting(1203, pose="stand")], set(), view(), view(), 100.8)
    assert events == [] and tr.state == LIVE and tr.last_seen == 100.7 and tr.x == pytest.approx(1203, abs=2)


def test_a_late_sighting_of_one_gone_off_screen_since_leaves_it_off_and_one_in_view_now_is_back():
    tracker = MonsterTracker()
    run(tracker, walk(1400, 125, 0.3))  # walking right
    gone = view(her_x=700)  # she has walked left since: where it should be now is off screen
    run(tracker, [(None, gone)], start=100.4)
    tr = tracker.tracks[1]
    assert tr.state == OFFSCREEN
    tracker.tracks[1].last_seen = 100.2  # its re-finds missed it from then on: a search of 100.3 saw it
    assert tracker.merge(100.3, [sighting(1480)], set(), view(), gone, 100.4) == [] and tr.state == OFFSCREEN
    tr.last_seen = 100.2
    events = tracker.merge(100.3, [sighting(1480)], set(), view(), view(), 100.4)  # in view now: back
    assert [e.kind for e in events] == ["back"] and tr.state == LIVE


def test_a_late_sighting_where_a_monster_was_found_dead_since_is_it_and_one_dead_before_is_a_new_one():
    tracker = MonsterTracker()
    run(tracker, [([sighting(1200, pose="stand")], view())] * 10)
    tr = tracker.tracks[1]
    tr.state, tr.died_at, tr.judged, tr.cause = DEAD, 100.4, 100.5, "other"
    assert tracker.merge(100.45, [sighting(1200, pose="stand")], set(), view(), view(), 100.6) == []
    assert list(tracker.tracks) == [1] and tr.state == DEAD and tracker.counts["late sightings of the dead"] == 1
    events = tracker.merge(100.55, [sighting(1200, pose="stand")], set(), view(), view(), 100.6)  # dead before it
    assert [(e.kind, e.track, e.t) for e in events] == [("appeared", 2, 100.55)]
    assert tracker.tracks[2].first_seen == 100.55


def test_a_spawn_a_late_search_saw_first_is_one_spawn_at_its_frame():
    from gamelab.localization import mapdata
    tree = {"info": {"VRLeft": 0, "VRTop": 0, "VRRight": 3000, "VRBottom": 1000},
            "foothold": {"1": {"1": {"1": {"x1": 400, "y1": 500, "x2": 1600, "y2": 500, "prev": 0, "next": 0}}}},
            "life": {"0": {"type": "m", "id": "1210100", "x": 1200, "y": 500, "fh": 1}},
            "miniMap": {"width": 3000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4}}
    model = mapdata.parse(1, tree, "Test", {}, {"1210100": "Pig"})
    tracker = MonsterTracker(model)
    pid = model.platform_under(1200, 500).id
    events = tracker.merge(100.3, [sighting(1205, platform=pid)], set(), view(), view(), 100.4)
    assert [(e.kind, e.t) for e in events] == [("spawned", 100.3)]
    run(tracker, [([sighting(1205, platform=pid)], view())] * 3, start=100.4)  # re-found since
    assert tracker.merge(100.45, [sighting(1205, platform=pid)], set(), view(), view(), 100.5) == []
    assert tracker.counts["tracks"] == 1
