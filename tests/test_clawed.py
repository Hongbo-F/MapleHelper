"""R1 of the cornered study (2026-11-03; the user's clip of 2026-11-02 22:51, "cornered character making bad attack
move": D:/MapleHelper-scratch/cornered): the looks are blind at her body, and a press from the ground goes to the claw
on a monster there, whatever she aimed at (15-25 a swipe, no star). A touch of hers with no monster about her that
could have made it (behavior.fight.clawed: the navigator's R7 test, control.navigator.fresh_touch, TOUCHER_S,
TOUCHER_NEAR) has a throw at one past the claw's edge go from a jump in place where its attack surely picks it
(attack.stance.CLAWED_AIR: _from_air; _from_here's ground_ok in close quarters). End to end: loop_tree_touched_unseen
(tests/test_sim.py's scenarios)."""

from types import SimpleNamespace

from maplehelper.engine import LEFT, RIGHT

from gamelab.attack import stance as st
from gamelab.behavior import fight
from gamelab.control import navigator as nav
from gamelab.localization.mapdata import Platform, Segment

from test_clearance import FLAT, HALVES, mon, stance

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


def _look(t: float, x: float = 500.0):
    where = SimpleNamespace(x=x, y=FLAT.y_at(x), platform=FLAT)
    return SimpleNamespace(t=t, where=where, platform=FLAT)


def test_clawed_is_the_navigators_r7_test_for_a_monster_on_her():
    # A touch 0.5 s before the look (its popup's time), nothing of the look's or about it within TOUCHER_NEAR of her
    # on her platform: a monster the looks miss is on her.
    s = _look(100.0)
    h = _hunter({"kind": "MISS", "at": 99.5})
    assert fight.clawed(h, s, [])
    assert fight.clawed(h, s, [mon(500 + nav.TOUCHER_NEAR + 1)])  # (one past it can't have made the touch)
    assert not fight.clawed(h, s, [mon(500 - nav.TOUCHER_NEAR)])  # (one by her could: the stance has it)
    assert fight.clawed(h, s, [mon(500, 400, OTHER.id)])  # (one on another platform, not beside her)
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
