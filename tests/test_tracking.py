"""Monsters followed from frame to frame (localization.tracking), on made-up sequences of the whole screen's
detections at the perception's rate (localization.rate.HZ: 15 a second): one id across walks and turns, facing and
the game data's speed from the first frame, a prediction that turns back at its platform's ends, two passing each
other kept apart, off screen against dead, a monster hidden under its damage numbers kept, and who killed it."""

import pytest

from gamelab.localization import tracking
from gamelab.localization.monsters import Monster
from gamelab.localization.rate import HZ
from gamelab.localization.tracking import (DEAD, DYING, HIDDEN, LIVE, OFFSCREEN, MonsterTracker, View)
from gamelab.perception.mobs import Mob
from gamelab.screen import UNITS_PER_HALF_PX as U

FPS = round(HZ)
THIRD = round(FPS / 3)  # frames in a third of a second (10 at 30 a second, 5 at 15)
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


def run(tracker: MonsterTracker, frames, start: float = 100.0):
    """Feed the tracker a frame at a time: each frame's detections (None: none), as the whole screen found them.
    Returns every event and the tracks' (id, state) after each frame."""
    events, states = [], []
    for k, (seen, v) in enumerate(frames):
        t = start + k / FPS
        events += tracker.update(t, list(seen or []), v, HER)
        states.append({tr.id: tr.state for tr in tracker.tracks.values()})
    return events, states


def walk(x0: float, v: float, seconds: float, start: float = 0.0, **kw) -> list:
    """A pig walking at v units/s from x0, a frame at a time."""
    return [([sighting(x0 + v * (start + k / FPS), facing=1 if v > 0 else -1, **kw)], view())
            for k in range(round(seconds * FPS))]


def test_one_id_across_a_walk_and_a_turn_facing_known_from_the_first_frame():
    tracker = MonsterTracker()
    there = walk(800, 125, 1.5)
    back = walk(800 + 125 * 1.5, -125, 1.5)
    events, states = run(tracker, there + back)
    assert [e.kind for e in events] == ["appeared"]
    assert all(list(s) == [1] for s in states)
    tr = tracker.tracks[1]
    assert tr.state == LIVE and tr.facing == -1 and tr.seen == len(there + back)
    first = MonsterTracker()
    run(first, there[:1])
    assert first.tracks[1].facing == 1  # from the template that matched, on its first frame


def test_its_speed_is_its_kinds_from_the_game_data_the_way_it_heads_from_its_first_frame():
    """monsters.walk_vx: a move frame walks its kind's speed (the game data's; a pig's 125, the tables') the way it
    faces, a stand or a flinch (hit1) frame stands, from the first frame (no filter to settle); one whose pose isn't
    known keeps what it had."""
    tracker = MonsterTracker()
    vx = []
    frames = walk(800, 125, 1.0) + walk(800 + 125, -125, 0.5) + [([sighting(800 + 125 - 62.5, pose="stand")], view())]
    frames += [([sighting(800 + 125 - 62.5, pose="")], view())] + [([sighting(800 + 50, pose="hit1")], view())]
    for k, frame in enumerate(frames):
        run(tracker, [frame], start=100.0 + k / FPS)
        vx.append(tracker.tracks[1].vx)
    assert vx[0] == 125.0 and all(v == 125.0 for v in vx[:FPS])
    assert vx[FPS] == -125.0  # (the frame it turned: the move frame faces the other way)
    assert vx[-3] == 0.0 and vx[-2] == 0.0 and vx[-1] == 0.0  # (stand; its pose unknown, as it was; flinching)
    slime = MonsterTracker()
    run(slime, [([sighting(900, kind="slime")], view())])
    assert slime.tracks[1].vx == 100.0  # (a slime's speed -20: 125 x 0.8)


def test_its_prediction_turns_back_at_its_platforms_ends():
    """Where a track should be between detections walks on its heading and turns back at the end its centre can reach
    (its platform's end less half its body: MonsterTrack.at's), for EXTRAPOLATE_S at most; clamped there until
    2026-10-30. 80% of the turns on the user's own play came at the ends."""
    from gamelab.localization import mapdata
    tree = {"info": {"VRLeft": 0, "VRTop": 0, "VRRight": 3000, "VRBottom": 1000},
            "foothold": {"1": {"1": {"1": {"x1": 400, "y1": 500, "x2": 1300, "y2": 500, "prev": 0, "next": 0}}}},
            "miniMap": {"width": 3000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4}}
    model = mapdata.parse(1, tree, "Test", {})
    pid = model.platform_under(1000, 500).id
    tracker = MonsterTracker(model)
    run(tracker, [([sighting(1250, platform=pid)], view())])
    tr = tracker.tracks[1]
    end = 1300 - tracking.MONSTER_HALF["pig"]  # (1266: where a pig's centre turns)
    assert tracker.predicted(tr, tr.last_seen + 0.1) == pytest.approx(1250 + 12.5)  # (short of it)
    assert tracker.predicted(tr, tr.last_seen + 0.2) == pytest.approx(end - (1250 + 25 - end))
    assert tracker.predicted(tr, tr.last_seen + 5.0) == pytest.approx(end - (1250 + 62.5 - end))  # (EXTRAPOLATE_S)
    standing = MonsterTracker(model)
    run(standing, [([sighting(1250, pose="stand", platform=pid)], view())])
    assert standing.predicted(standing.tracks[1], 101.0) == 1250


def test_it_keeps_its_id_across_a_few_frames_the_whole_screen_missed():
    """Missed in 1-4 frames (the whole screen's chains on the user's own play came back after 1-4 frames 172 times of
    483 ends): the same track, hidden once MISSING_S has gone by, its place predicted meanwhile."""
    for missed in (1, 2, 4, 6):
        tracker = MonsterTracker()
        frames = walk(800, 100, 1.0)
        frames[5:5 + missed] = [(None, view())] * missed
        events, states = run(tracker, frames)
        assert [e.kind for e in events] == ["appeared"] and all(set(st) == {1} for st in states)
        assert (tracking.HIDDEN in [st[1] for st in states]) == (missed / FPS >= tracking.MISSING_S)


def test_two_walking_past_each_other_keep_their_ids():
    """Two of a kind crossing: each predicted on its heading, the nearer pair taken first, and one facing against a
    track's heading away from its platform's ends FACING_COST farther (the gate at 15 a second, 43.7 units, holds
    both a frame on)."""
    tracker = MonsterTracker()
    frames = [([sighting(700 + 125 * k / FPS), sighting(1000 - 125 * k / FPS, facing=-1)], view())
              for k in range(2 * FPS)]
    run(tracker, frames)
    a, b = tracker.tracks[1], tracker.tracks[2]
    assert (a.facing, b.facing) == (1, -1) and len(tracker.tracks) == 2
    last = 2 * FPS - 1
    assert a.x == pytest.approx(700 + 125 * last / FPS, abs=3) and b.x == pytest.approx(1000 - 125 * last / FPS, abs=3)
    # Side by side the frame they meet, a gap of 2 either way: the heading's cost tells them apart.
    tracker = MonsterTracker()
    run(tracker, [([sighting(840), sighting(860, facing=-1)], view())])
    t = 100.0 + 1 / FPS
    there = [sighting(848, facing=-1), sighting(852)]  # (they've met: the left one's now the one facing left)
    tracker.update(t, there, view(), HER)
    assert tracker.matched[1].facing == 1 and tracker.matched[2].facing == -1


def test_one_that_walks_off_screen_is_offscreen_not_dead_and_back_with_its_id():
    tracker = MonsterTracker()
    out = walk(1550, 125, 1.0)                                         # walks off the right side
    gone = [(None, view())] * (6 * FPS)                                # 6 s off screen
    back = [([sighting(1660, facing=-1)], view())] * 10                # comes back in
    events, states = run(tracker, out + gone + back)
    kinds = [e.kind for e in events]
    assert kinds == ["appeared", "left", "back"]
    assert states[len(out) + FPS][1] == OFFSCREEN and "died" not in kinds
    assert list(tracker.tracks) == [1] and tracker.tracks[1].state == LIVE


def test_the_camera_moving_away_leaves_it_offscreen():
    tracker = MonsterTracker()
    frames = [([sighting(500, pose="stand", her_x=1000 + 8 * k)], view(1000 + 8 * k)) for k in range(15)]
    frames += [(None, view(1120 + 20 * k)) for k in range(30)]  # she runs right: it drops off the left
    events, _ = run(tracker, frames)
    assert [e.kind for e in events] == ["appeared", "left"] and tracker.tracks[1].state == OFFSCREEN


@pytest.mark.parametrize("hidden_s", [1.0, 2.0, 3.0])
def test_hidden_under_its_damage_numbers_for_up_to_3_s_it_keeps_its_id(hidden_s):
    tracker = MonsterTracker()
    before = [([sighting(1200, pose="stand", bar=1.0 if k < 5 else 0.6)], view()) for k in range(10)]
    hidden = [(None, view())] * round(hidden_s * FPS)
    after = [([sighting(1205, pose="stand", bar=0.6)], view())] * 5
    events, states = run(tracker, before + hidden + after)
    assert states[len(before) + 8][1] == HIDDEN
    assert [e.kind for e in events if e.kind != "bar_hit"] == ["appeared"]
    assert list(tracker.tracks) == [1] and tracker.tracks[1].state == LIVE


def test_one_gone_in_plain_view_with_no_exp_of_hers_was_killed_by_someone_else():
    tracker = MonsterTracker()
    hurt = [([sighting(1200, pose="stand", bar=1.0 if k < 5 else 0.3)], view()) for k in range(FPS)]
    events, states = run(tracker, hurt + [(None, view())] * (5 * FPS))
    died = [e for e in events if e.kind == "died"]
    assert [(e.track, e.data["cause"], e.data["hurt"]) for e in died] == [(1, "other", True)]
    assert died[0].t - (100 + (FPS - 1) / FPS) == pytest.approx(tracking.HIDE_S, abs=1 / FPS + 1e-9)
    bar = [e for e in events if e.kind == "bar_hit"]
    assert [e.data["by"] for e in bar] == ["other"]  # her throws are known, and none went its way
    assert tracker.counts["died other"] == 1 and tracker.counts["died hers"] == 0


def test_one_she_threw_at_that_vanished_as_her_exp_went_up_was_hers():
    tracker = MonsterTracker()
    t0 = 100.0
    standing = [([sighting(1200, pose="stand", bar=None if k < THIRD else 0.4)], view()) for k in range(FPS)]
    first = THIRD // 2
    events, _ = run(tracker, standing[:first], start=t0)
    tracker.throw(t0 + first / FPS, "right")             # her star goes its way ...
    more, _ = run(tracker, standing[first:], start=t0 + first / FPS)
    tracker.throw(t0 + 1.0, "right")                 # ... and another one kills it:
    tracker.gain(t0 + 1.4)                            # her EXP goes up
    fading = [([sighting(1200, pose="", bar=0.4)], view())] * round(0.8 * FPS)  # still seen fading for 0.8 s
    states = []
    for k, frame in enumerate(fading + [(None, view())] * (2 * FPS)):
        ev, st = run(tracker, [frame], start=t0 + 1.4 + k / FPS)
        more += ev
        states += st
    assert states[THIRD // 2][1] == DYING  # fading, while her gain waits for it to go
    died = [e for e in more if e.kind == "died"]
    assert [(e.data["cause"], e.data["at"]) for e in died] == [("hers", round(t0 + 1.4, 3))]
    assert [e.data["by"] for e in more if e.kind == "bar_hit"] == ["hers"]
    assert tracker.tracks[1].state == DEAD and tracker.tracks[1].cause == "hers"


def test_one_her_waiting_gain_would_go_on_out_of_sight_is_owed_not_about():
    """A gain no throw's key explains (none GAIN_AFTER_KEY before it: here the throw was 0.9 s before) waits for its
    monster to go (FADE_S + MISSING_S); meanwhile the likeliest of its monsters, out of sight already, is `owed`
    (localization.unseen doesn't take it for one about: review of 2026-10-26, a just-killed monster taken for one hidden
    under her body kept piles waiting and throws going at it); not while the likeliest is still seen fading (`dying`),
    nor once the gain has gone on it (dead)."""
    tracker = MonsterTracker()
    t0 = 100.0
    standing = [([sighting(1200, pose="stand", bar=0.4), sighting(1350, pose="stand")], view()) for k in range(FPS)]
    run(tracker, standing, start=t0)
    tracker.throw(t0 + 0.5, "right")
    tracker.gain(t0 + 1.4)
    fade = round(0.8 * FPS)
    fading = [([sighting(1200, pose="", bar=0.4), sighting(1350, pose="stand")], view())] * fade
    owed, states = [], []
    for k, frame in enumerate(fading + [([sighting(1350, pose="stand")], view())] * (2 * FPS)):
        _, st = run(tracker, [frame], start=t0 + 1.4 + k / FPS)
        states += st
        owed.append({tr.id for tr in tracker.tracks.values() if tr.owed})
    assert states[THIRD // 2][1] == DYING and owed[THIRD // 2] == set()  # fading: not owed
    out = fade + round(0.4 * FPS)  # out of sight 0.4 s, the gain waiting: owed
    assert states[out][1] == HIDDEN and owed[out] == {1}
    assert tracker.tracks[1].state == DEAD and owed[-1] == set() and 2 not in set().union(*owed)


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
    both = [([sighting(1150, pose="stand", bar=0.7), sighting(1300, pose="stand", bar=0.3)], view())] * THIRD
    run(tracker, both, start=t0)
    tracker.throw(t0 + 0.4, "right")
    # The near one is hit and hides under its damage numbers; the far one dies and fades for 0.8 s.
    hiding = [([sighting(1300, pose="", bar=0.3)], view())] * round(0.8 * FPS) + [(None, view())] * FPS
    tracker.gain(t0 + 0.5)
    events, _ = run(tracker, hiding, start=t0 + 0.5)
    events += run(tracker, [([sighting(1152, pose="stand", bar=0.5)], view())] * 5, start=t0 + 2.3)[0]
    died = [e for e in events if e.kind == "died"]
    assert [(e.track, e.data["cause"]) for e in died] == [(2, "hers")]
    assert tracker.tracks[1].state == LIVE  # back from hiding, the same one


def test_played_by_hand_her_exp_puts_the_kill_on_one_in_reach_that_went():
    tracker = MonsterTracker(throws=False)
    t0 = 100.0
    run(tracker, [([sighting(1250, pose="stand")], view())] * THIRD, start=t0)
    tracker.gain(t0 + 0.4)
    events, _ = run(tracker, [(None, view())] * (2 * FPS), start=t0 + 0.4)
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


def _player(x: float, kind: str = "stranger", pid: int = 7):
    from gamelab.localization.players import Player
    return Player(pid, kind, x, HER[1], 1, (0.0, 0.0), 0.0, 0.0, seen=2)


def test_one_gone_by_another_player_is_theirs_and_one_gone_with_nobody_by_it_is_nobodys():
    """With the other players on the minimap told (OTHERS_BY_PLAYER): a monster gone from sight in plain view is
    someone else's kill only with another player by it, and the death names them; with nobody else by it (her
    party doesn't count), it's let go without a death."""
    gone = [([sighting(1200, pose="stand")], view())] * FPS + [(None, view())] * (5 * FPS)
    for players, deaths in (([_player(1400)], [(1, "other", 7)]), ([_player(2000)], []),
                            ([_player(1250, kind="party")], []), ([], [])):
        tracker = MonsterTracker()
        tracker.update(99.0, [], view(), HER, players)
        events, _ = run(tracker, gone)
        assert [(e.track, e.data["cause"], e.data.get("by")) for e in events if e.kind == "died"] == deaths
        assert tracker.counts["nobody by it"] == (not deaths)


def test_her_gain_goes_on_one_gone_with_nobody_else_by_it():
    """Two monsters her throws went toward go at once, one by another player (350 units off it, 500 off the other), her
    gain no throw's kill by its time (the throw 0.9 s before it): her EXP gain is the other one, and the one by them is
    theirs (PLAYERS_LAST). With nobody else about, the nearer her is hers. (A throw keyed GAIN_AFTER_KEY before the gain
    puts it on what its attack took, whoever is by it: test_her_kill_is_what_her_attack_took_at_the_key.)"""
    t0 = 100.0
    for players, hers in (([], 1), ([_player(800, pid=3)], 2)):
        tracker = MonsterTracker()
        tracker.update(t0 - 1.0, [], view(), HER, players)
        run(tracker, [([sighting(1150, pose="stand"), sighting(1300, pose="stand")], view())] * THIRD, start=t0)
        tracker.throw(t0 - 0.5, "right")
        tracker.gain(t0 + 0.4)
        events, _ = run(tracker, [(None, view())] * (5 * FPS), start=t0 + 0.4)
        died = {e.track: (e.data["cause"], e.data.get("by")) for e in events if e.kind == "died"}
        assert died[hers] == ("hers", None) and died.get(3 - hers) == (("other", 3) if players else None)


# --- her kill bound to what her attack took (tracking._bind) -------------------------------------------------------

def _two(t0: float, near: float = 1150.0, far: float = 1300.0, **kw):
    """A tracker with two pigs standing ahead of her (right), seen for a third of a second."""
    tracker = MonsterTracker(**kw)
    run(tracker, [([sighting(near, pose="stand"), sighting(far, pose="stand")], view())] * THIRD, start=t0)
    return tracker


def test_her_kill_is_what_her_attack_took_at_the_key():
    """A gain GAIN_AFTER_KEY after a throw's key is the kill of the nearest of its kind ahead of her as its attack
    picked, at once (`victim`), whoever is by it, though both go out of sight: the one bound is dying from its gain's
    frame (Track.killed), dead once out of sight from CONFIRM_S on, its gain's frame its death's time; the other,
    hidden, is judged as before (someone else's, by another player)."""
    t0 = 100.0
    tracker = _two(t0)
    tracker.update(t0 + 0.3, [], view(), HER, [_player(1100, pid=3)])  # (by both: PLAYERS_LAST changes nothing)
    tracker.throw(t0 + 0.35, "right")
    tracker.gain(t0 + 0.6)
    events, states = run(tracker, [(None, view())] * (5 * FPS), start=t0 + 0.6)
    assert [(e.kind, e.track) for e in events if e.kind == "victim"] == [("victim", 1)]
    assert states[0][1] == DYING and states[0][2] in (LIVE, HIDDEN)  # (out of sight MISSING_S by then, or nearly)
    died = {e.track: (e.data["cause"], e.t - (t0 + 0.6)) for e in events if e.kind == "died"}
    assert died[1][0] == "hers" and tracking.CONFIRM_S <= died[1][1] < tracking.CONFIRM_S + 0.1
    assert died[2][0] == "other"
    assert [e.data["at"] for e in events if e.kind == "died" and e.track == 1] == [round(t0 + 0.6, 3)]
    assert tracker.counts["kills bound"] == 1


def test_her_kill_bound_walks_on_with_its_id_until_her_star_lands():
    """The monster bound walks on (it's drawn alive until her star lands, 0.6-0.9 s after the gain): its track follows
    it frame by frame, dying all along, and it's dead once it's gone."""
    t0 = 100.0
    tracker = _two(t0)
    tracker.throw(t0 + 0.3, "right")
    tracker.gain(t0 + 0.55)
    walking = [([sighting(1150 + 100 * k / FPS, bar=0.5), sighting(1300, pose="stand")], view())
               for k in range(2 * THIRD)]
    _, states = run(tracker, walking, start=t0 + 0.55)
    tr = tracker.tracks[1]
    assert all(st[1] == DYING for st in states) and tr.x > 1190 and tr.gain == t0 + 0.55 and tr.killed
    run(tracker, [([sighting(1300, pose="stand")], view())] * (2 * FPS), start=t0 + 0.55 + 2 * THIRD / FPS)
    assert tracker.tracks[1].state == DEAD and tracker.tracks[1].cause == "hers" and tracker.tracks[2].state == LIVE


def test_one_seen_lately_comes_before_one_out_of_sight_where_it_should_be():
    """Of the two of its kind ahead of her, one out of sight longer than FRESH_S at the gain comes after one seen in
    its frame, nearer or not (out of sight it's where it should be, and often isn't: hit and hidden, or gone)."""
    t0 = 100.0
    tracker = _two(t0)
    run(tracker, [([sighting(1300, pose="stand")], view())] * round(0.4 * FPS), start=t0 + THIRD / FPS)  # the near
    #                                                                                                  one hides
    tracker.throw(t0 + 0.4, "right")
    tracker.gain(t0 + 0.7)
    events, _ = run(tracker, [([sighting(1300, pose="stand")], view())], start=t0 + 0.7)
    assert [e.track for e in events if e.kind == "victim"] == [2]


def test_her_kill_is_one_ahead_of_her_within_her_attacks_reach_up_and_down():
    """Behind her, or more than VICTIM_UP above her feet (VICTIM_DOWN below), it isn't her attack's: the nearer ahead
    within them is."""
    t0 = 100.0
    tracker = MonsterTracker()
    seen = [sighting(960, pose="stand"), sighting(1100, pose="stand", y=HER[1] - 130, platform=2),
            sighting(1250, pose="stand", y=HER[1] - 90, platform=3)]
    run(tracker, [(seen, view())] * THIRD, start=t0)
    tracker.throw(t0 + 0.4, "right")
    tracker.gain(t0 + 0.7)
    events, _ = run(tracker, [(seen, view())], start=t0 + 0.7)
    assert [e.data["x"] for e in events if e.kind == "victim"] == [1250]


def test_one_bound_seen_again_rebind_s_on_was_another_and_the_gain_goes_on_the_next_gone():
    """The one bound re-found REBIND_S after its gain is alive: let go (`let_go`), and the gain goes on the next its
    attack took that's out of sight then (her star took that one), dead once it's been gone from CONFIRM_S on."""
    t0 = 100.0
    tracker = _two(t0)
    tracker.throw(t0 + 0.3, "right")
    tracker.gain(t0 + 0.55)
    frames = [([sighting(1150, pose="stand")], view())] * (2 * FPS)  # the near one stays; the far one is gone
    events, _ = run(tracker, frames, start=t0 + 0.55)
    assert [(e.kind, e.track) for e in events if e.kind in ("victim", "let_go")] == [
        ("victim", 1), ("let_go", 1), ("victim", 2)]
    assert tracker.tracks[1].state == LIVE and tracker.tracks[1].gain is None
    assert [(e.track, e.data["cause"]) for e in events if e.kind == "died"] == [(2, "hers")]


def test_a_death_seen_is_dead_at_once_and_puts_her_kill_right():
    """A death seen (`seen_dying`: perception.deaths) on the one bound makes it dead at once; on another her attack
    could have taken, the gain goes on that one (the one bound let go); on one no gain of hers can be, someone else's,
    at once (not HIDE_S later)."""
    t0 = 100.0
    for seen in (1, 2):
        tracker = _two(t0)
        tracker.throw(t0 + 0.3, "right")
        tracker.gain(t0 + 0.55)
        run(tracker, [([sighting(1150, pose="stand"), sighting(1300, pose="stand")], view())] * THIRD, start=t0 + 0.55)
        tracker.seen_dying(seen, t0 + 0.9)
        events, _ = run(tracker, [([sighting(1150 if seen == 2 else 1300, pose="stand")], view())], start=t0 + 0.9)
        assert [(e.track, e.data["cause"]) for e in events if e.kind == "died"] == [(seen, "hers")]
        assert tracker.tracks[3 - seen].state == LIVE and tracker.tracks[3 - seen].gain is None
    tracker = _two(t0)
    tracker.seen_dying(2, t0 + 0.4)
    events, _ = run(tracker, [([sighting(1150, pose="stand")], view())], start=t0 + 0.4)
    assert [(e.track, e.data["cause"], e.data["seen"]) for e in events if e.kind == "died"] == [(2, "other", True)]


def test_a_death_seen_sooner_than_her_star_can_land_isnt_her_kill():
    """Another player's kill of the far one seen dying in her gain's frame, or a frame after it (DEATH_AFTER_GAIN_S:
    her star hasn't landed by then), doesn't take her gain: it stays on the near one her attack took, and the far one is
    someone else's. (The gains were bound after the deaths seen, with no time asked: the far one took the gain, and
    the near one was a target again: review of 2026-10-27, review-behaviour/order.py.)"""
    t0 = 100.0
    both = [sighting(1150, pose="stand"), sighting(1300, pose="stand")]
    for after in (0.0, 1 / FPS):
        tracker = _two(t0)
        tracker.throw(t0 + 0.4, "right")
        run(tracker, [(both, view())] * 7, start=t0 + 10 / FPS)
        g = t0 + 0.65
        tracker.gain(g)
        if not after:
            tracker.seen_dying(2, g)
        events, _ = run(tracker, [(both, view())], start=g)
        if after:
            tracker.seen_dying(2, g + after)
            events += run(tracker, [([sighting(1150, pose="stand")], view())], start=g + after)[0]
        assert [(e.kind, e.track) for e in events if e.kind in ("victim", "let_go")] == [("victim", 1)]
        assert [(e.track, e.data["cause"]) for e in events if e.kind == "died"] == [(2, "other")]
        assert tracker.tracks[1].killed


def test_her_kill_bound_isnt_found_again_on_a_neighbour_once_her_star_has_landed():
    """Her kill walks on until her star lands (STAR_LANDS_S after the key, and her distance over STAR_FLIGHT), then
    stands there dying, and the detector loses it: lost (MISSING_S) after that (LANDS_SLACK_S), a sighting 70 units on,
    which the track's window (grown with the time out of sight) would take, is another monster, a new track at once (no
    search needed), and the one bound isn't let go for it but dead from CONFIRM_S on. One where it was lost is it still.
    (On the recordings' replays a bound track was re-found 45-97 units from its last sighting 0.7-1.0 s on, and let go,
    on 7 of 165 bindings: review of 2026-10-27, review-truth/binds.py.)"""
    t0 = 100.0
    tracker = MonsterTracker()
    run(tracker, [([sighting(1150, pose="stand")], view())] * THIRD, start=t0)
    tracker.throw(t0 + 0.3, "right")
    g = t0 + 0.55
    tracker.gain(g)
    run(tracker, [([sighting(1150, pose="stand")], view())] * round(0.6 * FPS), start=g)  # drawn alive till her star
    #                                                                                          lands
    lands = t0 + 0.3 + tracking.STAR_LANDS_S + 150 / tracking.STAR_FLIGHT
    assert tracker.tracks[1].lands == pytest.approx(lands)
    there = sighting(1170, pose="stand")
    assert not tracker._stray(tracker.tracks[1], there, g + 0.9)
    assert not tracker._stray(tracker.tracks[1], sighting(1220, pose="stand"), lands + tracking.LANDS_SLACK_S - 0.01)
    other = sighting(1220, pose="stand")
    events = tracker.update(g + 0.9, [other], view(), HER)  # (in the gate its time out of sight grew)
    assert tracker.tracks[1].last_seen < g + 0.9 and [e.kind for e in events] == ["appeared"]
    for k in range(1, 2 * FPS):
        events += tracker.update(g + 0.9 + k / FPS, [other], view(), HER)
    assert not [e for e in events if e.kind == "let_go"]
    assert tracker.tracks[1].state == DEAD and tracker.tracks[1].cause == "hers" and tracker.tracks[2].state == LIVE


def test_a_gain_of_no_kind_known_goes_on_the_nearest_ahead_her_throws_hurt_first():
    """Its kind not known (its size unread, or no kill's), the gain goes on the nearest ahead of her of any kind, one
    showing its bar (her throws hurt it) before one unhurt (UNKNOWN_HURT_FIRST); a kind named, the nearest of it."""
    t0 = 100.0
    seen = [sighting(1100, pose="stand", kind="slime"), sighting(1250, pose="stand", bar=0.4)]
    for kind, x in ((None, 1250), ("slime", 1100)):
        tracker = MonsterTracker()
        run(tracker, [(seen, view())] * THIRD, start=t0)
        tracker.throw(t0 + 0.4, "right")
        tracker.gain(t0 + 0.65, kind)
        events, _ = run(tracker, [(seen, view())], start=t0 + 0.65)
        assert [e.data["x"] for e in events if e.kind == "victim"] == [x]


def test_a_throw_told_again_the_other_way_binds_its_kill_on_that_side():
    """Her keys had her throw left (the stance's side), her look's frame showed her facing right (behavior.hunter:
    told again, at its key, RETOLD_S): the gain bound already to the pig on her left is let go and bound to the one on
    her right her attack took; told again before the gain, it's bound there at once. Told again the same way, or a
    throw at another key, it changes nothing. (On the recording of 2026-10-26 00:20 two of 15 kills were bound so to a
    live one on the side she didn't throw at: review of 2026-10-27, review-truth.)"""
    t0 = 100.0
    both = [sighting(850, pose="stand"), sighting(1150, pose="stand")]
    for before in (False, True):
        tracker = MonsterTracker()
        run(tracker, [(both, view())] * THIRD, start=t0)
        key = t0 + 0.4
        tracker.throw(key, "left")
        if before:
            tracker.throw(key, "right")
        g = t0 + 0.65
        tracker.gain(g, "pig")
        events, _ = run(tracker, [(both, view())], start=g)
        tracker.throw(key, "right")
        tracker.throw(key, "right")  # (once is enough)
        events += run(tracker, [(both, view())] * 3, start=g + 1 / FPS)[0]
        bound = [(e.kind, e.track, e.data.get("why")) for e in events if e.kind in ("victim", "let_go")]
        assert bound == ([("victim", 2, None)] if before else [("victim", 1, None), ("let_go", 1, "retold"),
                                                                ("victim", 2, None)])
        assert tracker.tracks[2].killed and not tracker.tracks[1].killed and len(tracker.throws) == 1


def test_told_again_after_her_victims_death_was_seen_it_is_hers_after_all():
    """The throw told again only after the death of the pig on her right was seen (someone else's then: her gain was
    bound to the one on her left), DEATH_AFTER_GAIN_S or more after the gain: the one on her left is let go, and the
    one on her right is her kill after all (`_taken_back`), not someone else's. (Live at 2026-10-26 00:20:13, her
    victim's death was seen 0.09 s before the frame that told the throw again.)"""
    t0 = 100.0
    both = [sighting(850, pose="stand"), sighting(1150, pose="stand")]
    tracker = MonsterTracker()
    run(tracker, [(both, view())] * THIRD, start=t0)
    key, g = t0 + 0.4, t0 + 0.65
    tracker.throw(key, "left")
    tracker.gain(g, "pig")
    events, _ = run(tracker, [(both, view())] * 12, start=g)
    tracker.seen_dying(2, g + 12 / FPS)
    events += run(tracker, [([both[0]], view())], start=g + 12 / FPS)[0]
    assert [(e.track, e.data["cause"]) for e in events if e.kind == "died"] == [(2, "other")]
    tracker.throw(key, "right")
    events = run(tracker, [([both[0]], view())], start=g + 13 / FPS)[0]
    assert [(e.kind, e.track) for e in events if e.kind in ("victim", "let_go", "died")] == [
        ("let_go", 1), ("victim", 2), ("died", 2)]
    assert tracker.tracks[2].cause == "hers" and tracker.tracks[2].gain == g and not tracker.tracks[1].killed
    assert tracker.counts["died other"] == 0 and tracker.counts["died hers"] == 1
