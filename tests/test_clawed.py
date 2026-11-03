"""R1 of the cornered study (2026-11-03; the user's clip of 2026-11-02 22:51, "cornered character making bad attack
move": D:/MapleHelper-scratch/cornered): the looks are blind at her body, and a press from the ground goes to the claw
on a monster there, whatever she aimed at (15-25 a swipe, no star). A touch of hers with no monster about her that
could have made it (behavior.fight.clawed: the navigator's R7 test, control.navigator.fresh_touch, TOUCHER_S,
TOUCHER_NEAR) has a throw at one past the claw's edge go from a jump in place where its attack surely picks it
(attack.stance.CLAWED_AIR: _from_air; _from_here's ground_ok in close quarters). End to end: loop_tree_touched_unseen
(tests/test_sim.py's scenarios)."""

from dataclasses import replace
from types import SimpleNamespace

from maplehelper.engine import LEFT, RIGHT

from gamelab.attack import stance as st
from gamelab.behavior import fight
from gamelab.control import navigator as nav
from gamelab.localization import unseen as unseen_mod
from gamelab.localization.mapdata import Platform, Segment
from gamelab.localization.monsters import bodies
from gamelab.localization.tracking import DYING, LIVE
from gamelab.localization.unseen import unseen

from test_clearance import FLAT, GREEN, HALVES, mon, stance
from test_unseen import _map, _scene, _seen, _track

OTHER = Platform(1, [Segment(0, 400, 1000, 400)])


def touched(plan_or_act) -> bool:
    plan = st.plan_of(plan_or_act)
    return plan is not None and plan.why.startswith("touched") and plan.sure


def test_a_touch_with_none_by_her_has_a_throw_past_the_claws_edge_go_from_a_jump():
    # A Green 150 off, standing: from the ground a star (past the claw's edge, 105). With a monster the looks miss on
    # her (`clawed`), that press would be the claw's on it: the jump in place keyed 40 ms after the jump key instead,
    # sure to pick the Green (reach.air_sure, CLEAR_MARGIN farther), its jitter's room 40-55 as her jumps up close have.
    green = mon(650)
    assert stance(500, [green]) == ("throw", RIGHT)
    act = stance(500, [green], clawed=True)
    assert act[:2] == ("jump_throw", RIGHT) and touched(act)
    plan = st.plan_of(act)
    assert (plan.key_ms, plan.way, plan.room, plan.target) == (40, 0, (40, 55), green)
    # Facing away: tapped round on the ground first, as up close (an air turn's attack, her feet 52-56 up, picks one on
    # her level only from 144).
    act = stance(500, [green], facing=LEFT, clawed=True)
    assert act[:2] == ("jump_throw", RIGHT) and st.plan_of(act).why == "touched, tap first"
    assert not st.plan_of(act).air_turn
    # Out to the star's reach on her level.
    assert touched(stance(500, [mon(880)], clawed=True))


def test_with_no_jump_sure_or_the_switch_off_it_goes_from_the_ground_as_before(monkeypatch):
    green = mon(650)
    # Under a platform overhead (a jump would land her on it): from the ground (the claw's swipe, if one's there).
    assert stance(500, [green], clawed=True, can_jump=lambda a, b: False) == ("throw", RIGHT)
    monkeypatch.setattr(st, "CLAWED_AIR", False)
    assert stance(500, [green], clawed=True) == ("throw", RIGHT)


def test_up_close_the_throw_from_here_is_no_star_from_the_ground_when_clawed():
    # _from_here (close quarters: one too close by its danger, past the claw's edge where its attack starts): 115 off,
    # from the ground sure; with `ground_ok` False (stance's `clawed`) the jump in place, as sure.
    args = (500, 500, RIGHT, mon(615), RIGHT, lambda a, b: True, 51.0, 28.0, [], HALVES, FLAT, st.NO_PACE)
    assert st._from_here(*args) == st.Here(("throw", RIGHT), True, 115)
    here = st._from_here(*args, ground_ok=False)
    assert here.sure and here.act[:2] == ("jump_throw", RIGHT)


class _Danger:
    any = True

    @staticmethod
    def beside(m, platform, x):
        return False


def _hunter(touch: dict | None, danger=_Danger()):
    return SimpleNamespace(danger=danger, senses=SimpleNamespace(touch=touch))


def _look(t: float, x: float = 500.0, monsters=()):
    where = SimpleNamespace(x=x, y=FLAT.y_at(x), platform=FLAT)
    return SimpleNamespace(t=t, where=where, platform=FLAT, monsters=list(monsters))


def test_clawed_is_the_navigators_r7_test_for_a_monster_on_her():
    # A touch 0.5 s before the look (its popup's time), nothing the look found within TOUCHER_NEAR of her on her
    # platform: a monster the looks miss is on her.
    s = _look(100.0)
    h = _hunter({"kind": "MISS", "at": 99.5})
    assert fight.clawed(h, s, [])
    past, by, other = mon(500 + nav.TOUCHER_NEAR + 1), mon(500 - nav.TOUCHER_NEAR), mon(500, 400, OTHER.id)
    seen = _look(100.0, monsters=[past, by, other])
    assert fight.clawed(h, seen, [past])  # (one past it can't have made the touch)
    assert not fight.clawed(h, seen, [by])  # (one the look found by her could: the stance has it)
    assert fight.clawed(h, seen, [other])  # (one on another platform, not beside her)
    # A stand-in by her of one the look didn't find (hidden, unseen) is a monster the looks miss: still clawed.
    assert fight.clawed(h, s, [by])
    # Not after TOUCHER_S, nor for a MISS that may be another player's on her (`crowded`), nor with none.
    assert not fight.clawed(_hunter({"kind": "MISS", "at": 100.0 - nav.TOUCHER_S - 0.01}), s, [])
    assert not fight.clawed(_hunter({"kind": "MISS", "at": 99.5, "crowded": True}), s, [])
    assert not fight.clawed(_hunter(None), s, [])
    # Where nothing counts as a danger (the first map), as it was.
    assert not fight.clawed(_hunter({"kind": "MISS", "at": 99.5}, danger=None), s, [])
    assert not fight.clawed(_hunter({"kind": "MISS", "at": 99.5}, danger=SimpleNamespace(any=False)), s, [])
    # The navigator's own test is the same one (Navigator._toucher: fresh_touch).
    assert nav.fresh_touch(h.senses, s) is h.senses.touch
    assert nav.fresh_touch(_hunter({"kind": "MISS", "at": 101.0}).senses, s) is None  # (after the look: not yet)
    # A touch told in the look's own frame: its popup's `at` rounded to the ms can read 0.5 ms after the frame's time
    # (fight.TOUCH_ROUND_S): the fights take it; R7's stand-in as it was.
    same = _hunter({"kind": "MISS", "at": 100.0005})
    assert fight.clawed(same, s, []) and nav.fresh_touch(same.senses, s) is None


# --- R1 extended (a): a LIVE track the look's frame missed is about (localization.unseen.UNSEEN_LIVE) -----------------

def test_a_live_track_missed_in_the_looks_frame_counts_as_about(monkeypatch):
    """unseen.hidden and unseen: a track the feed still has LIVE whose last sighting is before the look's frame (missed
    in it: under tracking.MISSING_S) is about, walked on as a hidden one is; one seen in the frame isn't (the look has
    it). One found of another track beside it doesn't explain it (the tracker matched the frame's sightings already:
    two of a kind side by side are two); one found with no track near enough does, as for a hidden one. Off: HIDDEN
    ones alone."""
    m = _map()
    her, t = 1, 100.0
    missed = _track(1, 520.0, her, t - 0.07, vx=100.0, state=LIVE)  # missed in this frame: on her, walking right
    now = _track(2, 300.0, her, t, state=LIVE)  # seen in the frame
    s = _scene(t, 500.0, her, tracks=[missed, now])
    assert [tr.id for tr in unseen_mod.hidden(s)] == [1]
    assert [round(g.x) for g in unseen(s, None, m)] == [527]  # (0.07 s at 100 units/s)
    beside = replace(_seen("horny-mushroom", 490.0, her, m), track=3)  # another track's, 30 off it
    s.monsters = [beside]
    assert [g.track for g in unseen(s, None, m)] == [1]
    s.monsters = [_seen("horny-mushroom", 490.0, her, m)]  # found with no track (no feed's): that one
    assert unseen(s, None, m) == []
    s.monsters = []
    monkeypatch.setattr(unseen_mod, "UNSEEN_LIVE", False)
    assert unseen_mod.hidden(s) == [] and unseen(s, None, m) == []


def test_a_live_track_missed_on_her_is_about_her_and_a_monster_the_looks_miss():
    """The LIVE track missed on her is among the monsters about her (Hunt._with_unseen's hidden, the stance's `around`:
    her danger, her jumps' arcs, too close), and with a touch it's the monster the looks miss on her, `clawed` (a press
    from the ground would swipe it), as with none at all; the same one found in the look explains the touch."""
    m = _map()
    her = m.platform(1)
    h = _hunter({"kind": "MISS", "at": 99.5})
    on_her = unseen(_scene(100.0, 500.0, her.id, tracks=[_track(1, 520.0, her.id, 99.93, state=LIVE)]), None, m)
    s = SimpleNamespace(t=100.0, where=SimpleNamespace(x=500.0, y=500.0, platform=her), platform=her, monsters=[])
    assert len(on_her) == 1 and fight.clawed(h, s, on_her)
    s.monsters = on_her
    assert not fight.clawed(h, s, on_her)


# --- R1 extended (b): the fight's hidden ones walk on along their track (behavior.fight.HIDDEN_WALKS) ----------------

def test_a_hidden_one_she_threw_at_walks_on_along_its_track(monkeypatch):
    """FightState.hidden with hidden_place: a monster she threw at that the look doesn't see, heading at her, is where
    its feed's track has walked it since its last sighting (its heading at its kind's speed, 100 for a Horny Mushroom from
    the game data), no nearer than where its body meets hers (HIDDEN_AT_HER), not where it was last seen; one heading
    away (hurt, it turns to chase her), standing (heading 0) or whose track is dying stays as last seen, and so do all of
    them with the switch off. A throw at its stand-in keeps its entry as it was."""
    m = _map()
    her, t = 1, 100.0
    last = replace(_seen("horny-mushroom", 300.0, her, m), track=7)
    tr = _track(7, 300.0, her, t - 1.0, vx=100.0)  # out of sight a second, walking right
    s = _scene(t, 500.0, her, tracks=[tr])
    h = SimpleNamespace(model=m)
    state = fight.FightState(thrown=[[last, t - 1.0]])
    got = state.hidden(t, fight.hidden_place(h, s))
    assert len(got) == 1 and round(got[0].x) == 400 and got[0].track == 7 and got[0].y == m.platform(her).y_at(400)
    # aimed at, it's its entry: kept as last seen then, not a new one seen now
    state.aimed = [got[0]]
    state.note_throw(t)
    assert len(state.thrown) == 1 and state.thrown[0][0] is last and state.thrown[0][1] == t - 1.0
    # out of sight 3 s: it would be past her, so it's where it meets her (its half and hers from her x, on its side)
    far = _track(7, 300.0, her, t - 3.0, vx=100.0)
    met = state.hidden(t, fight.hidden_place(h, _scene(t, 500.0, her, tracks=[far])))[0]
    assert met.x == 500.0 - (bodies(m)["horny-mushroom"].half + st.HER_HALF)
    # heading away, standing, dying, or switched off: as last seen
    for other in (_track(7, 300.0, her, t - 1.0, vx=-100.0), _track(7, 300.0, her, t - 1.0, vx=0.0),
                  _track(7, 300.0, her, t - 1.0, vx=100.0, state=DYING)):
        assert state.hidden(t, fight.hidden_place(h, _scene(t, 500.0, her, tracks=[other])))[0] is last
    monkeypatch.setattr(fight, "HIDDEN_WALKS", False)
    assert fight.hidden_place(h, s) is None and state.hidden(t, fight.hidden_place(h, s))[0] is last


# --- R1 extended: up close, past the one on her (attack.stance.CLAWED_CLOSE) ---------------------------------------------

def test_up_close_an_unsure_jump_in_place_becomes_a_sure_one_past_the_monster_on_her(monkeypatch):
    """With a monster the looks miss on her (`clawed`) and one too close in front of her that no throw from where she
    stands surely takes (a Green 40 off: inside her jump's reach, 54, and the claw's box), the last resort is a jump in
    place sure of nothing; a sure jump throw that way at one past it (a Green 150 off) goes instead (stance._from_air:
    its star takes the one on her if its body reaches the search's window, else the one past it). Not clawed, or off:
    as before."""
    on_her, past = mon(50), mon(160)  # (her at 10, her platform's left end behind her: no room to step or jump back)
    before = stance(10, [on_her, past])
    assert before == ("jump_throw", RIGHT) and st.plan_of(before) is None
    act = stance(10, [on_her, past], clawed=True)
    assert act[:2] == ("jump_throw", RIGHT) and touched(act) and st.plan_of(act).target is past
    monkeypatch.setattr(st, "CLAWED_CLOSE", False)
    assert stance(10, [on_her, past], clawed=True) == before


def test_up_close_facing_away_the_unsure_tap_first_jump_becomes_a_sure_one_past_the_monster_on_her(monkeypatch):
    """Facing away from the one on her, the last resort is the jump in place tapped round first, as unsure of its
    target (_from_here's `sure` False) though its plan says 'tap first' (JumpPlan.sure is its jitter room's, checked or
    not: the review of 2026-11-03 found CLAWED_CLOSE taking that for a sure jump and keeping it): clawed, the sure jump
    throw past it goes instead, tapped round too."""
    on_her, past = mon(50), mon(160)
    before = stance(10, [on_her, past], facing=LEFT)
    assert before[:2] == ("jump_throw", RIGHT) and st.plan_of(before).why == "tap first"
    act = stance(10, [on_her, past], facing=LEFT, clawed=True)
    assert act[:2] == ("jump_throw", RIGHT) and touched(act) and st.plan_of(act).target is past
    assert st.plan_of(act).why == "touched, tap first" and not st.plan_of(act).air_turn
    monkeypatch.setattr(st, "CLAWED_CLOSE", False)
    assert stance(10, [on_her, past], facing=LEFT, clawed=True) == before


def test_up_close_the_other_ways_throw_is_weighed_as_a_clawed_decision_throws(monkeypatch):
    # The three cornered fixes together (work/cornered-all): with FIRST_TO_REACH her close quarters weigh the nearest on
    # each side, and with `clawed` neither throw is from the ground (a monster the looks miss is on her: the claw would
    # take the press). A Green 45 off ahead (a jump in place picks one from 54: unsure) and another 120 off behind her
    # (past the claw's edge: from the ground sure, unclawed), which also hems her in (no step, no jump back): the sure
    # throw the other way in place of the unsure jump. Unclawed from the ground; clawed from a jump in place
    # ('touched'), not a ground press the claw on her would take (where R1 alone had the unsure jump ahead).
    ahead, behind = mon(545), mon(380)
    args = (500, [ahead, behind], [ahead, behind], RIGHT, FLAT.left, FLAT.right, lambda a, b: True, {GREEN: 51.0},
            HALVES)
    kw = dict(platform=FLAT, y=500.0)
    monkeypatch.setattr(st, "FIRST_TO_REACH", True)
    assert st._close_quarters(*args, **kw) == ("throw", LEFT)
    act = st._close_quarters(*args, clawed=True, **kw)
    assert act[:2] == ("jump_throw", LEFT) and touched(act) and st.plan_of(act).target is behind
    monkeypatch.setattr(st, "FIRST_TO_REACH", False)
    act = st._close_quarters(*args, clawed=True, **kw)
    assert act[:2] == ("jump_throw", RIGHT) and not touched(act)  # (the unsure jump in place at the one ahead)
