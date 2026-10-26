"""Her actions as key plans (control.actions), without the player."""

from types import SimpleNamespace

import pytest
from maplehelper.engine import RIGHT

from gamelab.control.actions import STAR_DELAY_MS, Actions
from gamelab.motion.flow import JUMP_LAG
from gamelab.motion.physics import JUMP_ATTACK_MS, LAGS, air_time


def actions(sent: list) -> Actions:
    key = lambda name: SimpleNamespace(name=name)
    a = Actions(SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), attack=key("f"),
                                lucky_seven=None, pick_up=key("z"), hp_potion=None, chair=None))
    a.play = lambda events, name: sent.append((name, events))
    return a


def test_jump_back_turns_and_throws_just_before_landing():
    sent = []
    actions(sent).jump_back(RIGHT)  # the monster is on her right: she jumps left
    (_, events), = sent
    assert [e[0] for e in events] == sorted(e[0] for e in events)  # the player refuses unsorted plans
    at = lambda k, how: next(t for t, name, h_ in events if name == k and h_ == how)
    lift = at("alt", "down") + 1000 * JUMP_LAG
    land = lift + 1000 * air_time(0)
    apex = lift + 1000 * air_time(0) / 2
    assert at("left", "down") < at("alt", "down")              # moving away when she jumps
    assert lift < at("right", "down") + 1000 * LAGS.on         # the game sees the turn once she's off the ground
    assert at("right", "down") < at("f", "down") < apex        # turned in the air, then the key
    leaves = at("f", "down") + STAR_DELAY_MS                    # ... the star leaves ~520 ms later
    # as she lands, give or take: the attack physics.AIR_TURN_LEAD_MS after the turn's arrow (until 2026-10-26 25 ms
    # after it, the star just before she landed, and both live jump backs of 2026-10-25 threw the wrong way)
    assert -40 <= land - leaves <= 40
    assert at("f", "down") - at("right", "down") >= 60
    assert at("left", "up") <= at("right", "down")             # no longer holding away when she lands


def test_the_attack_keys_come_as_long_after_the_jump_keys_as_attack_reach_reckons():
    # attack.reach works out what an attack from a jump hits from these gaps (motion.physics): the key
    # plans must keep to them.
    from gamelab.attack import reach
    from gamelab.motion.physics import AIR_TURN_ATTACK_MS, JUMP_ATTACK_MS
    assert reach.TURN_KEY_MS == AIR_TURN_ATTACK_MS  # (a jump back turns her in the air)
    for method, gap in (("jump_throw", JUMP_ATTACK_MS), ("jump_back", AIR_TURN_ATTACK_MS)):
        sent = []
        a = actions(sent)
        a.facing = RIGHT
        getattr(a, method)(RIGHT)
        (_, events), = sent
        at = lambda k: next(t for t, name, h_ in events if name == k and h_ == "down")
        assert at("f") - at("alt") == gap, method


def test_a_jump_backs_lock_counts_from_when_its_plan_is_done_as_a_jump_throws(monkeypatch):
    """play waits for the plan, whose last key is the attack's: her lock runs the attack's lock and a jump throw's
    margin from then, however late in the jump the attack is keyed (until the review of 2026-10-25 it counted the
    attack key's time into the plan again: 235 ms more by default, 630 keyed 480 ms after the jump key)."""
    from gamelab.control import actions as actions_mod
    from maplehelper.game import COMBAT
    now = [100.0]
    monkeypatch.setattr(actions_mod.time, "monotonic", lambda: now[0])
    want = (COMBAT.attack_lock_ms + actions_mod.JUMP_THROW_LOCK_EXTRA_MS) / 1000
    for key_ms in (None, 85, 480):
        a = actions([])
        a.jump_throw(RIGHT, key_ms)
        assert a.lock_until == pytest.approx(now[0] + want)
        a.lock_until = 0.0
        a.jump_back(RIGHT, key_ms)
        assert a.lock_until == pytest.approx(now[0] + want), key_ms


def test_a_throw_turns_her_only_when_she_faces_the_other_way():
    sent = []
    a = actions(sent)
    a.throw(RIGHT)
    a.lock_until = 0.0
    a.throw(RIGHT)
    turned, again = sent[0][1], sent[1][1]
    assert [e[1] for e in turned] == ["right", "right", "f", "f"] and [e[1] for e in again] == ["f", "f"]
    assert a.facing == RIGHT


def test_no_potion_key_means_no_potion():
    sent = []
    assert actions(sent).drink() is False and not sent


def test_a_plan_with_a_jump_waits_out_the_attack_lock_and_one_without_doesnt(monkeypatch):
    """A jump (or attack) pressed inside the 730 ms attack lock is ignored, so every plan that presses one
    is held back just enough: a jump 200 ms into the plan, with 0.5 s of lock left, waits 0.3 s."""
    from gamelab.control import actions as actions_mod
    now, slept = [100.0], []
    monkeypatch.setattr(actions_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(actions_mod.time, "sleep", lambda s: slept.append(round(s, 3)))
    a = actions([])
    a.lock_until = 100.5
    a.hold_back([[0, "right", "down"], [200, "alt", "down"], [240, "alt", "up"]])
    a.hold_back([[0, "right", "down"], [400, "right", "up"]])        # no jump: nothing to wait for
    a.hold_back([[0, "right", "down"], [600, "alt", "down"]])        # its jump comes after the lock anyway
    assert slept == [0.3]


def test_a_plans_first_arrow_waits_until_her_attack_stops_swallowing_arrows(monkeypatch):
    """An arrow pressed while her attack plays doesn't move her until about 0.79 s after its key (in game on
    2026-10-21: chains sent about 0.3 s after a throw's key stood still, and those held back to 0.6 s set off
    0.12 s late), so a plan's first arrow goes down ARROW_LOCK_MS (0.8 s) after the last attack key at the
    soonest: an arrow-first plan 0.3 s after a throw waits 0.5 s, one whose arrow comes 0.3 s in waits 0.2 s.
    Jumps and attacks still wait for the attack lock, as before."""
    from gamelab.control import actions as actions_mod
    from maplehelper.game import COMBAT
    now, slept = [100.0], []
    monkeypatch.setattr(actions_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(actions_mod.time, "sleep", lambda s: slept.append(round(s, 3)))
    a = actions([])
    a.facing = RIGHT
    a.throw(RIGHT)  # no turn: the attack key goes down as the plan starts, at 100.0
    assert a.arrow_until == pytest.approx(100.0 + actions_mod.ARROW_LOCK_MS / 1000) == pytest.approx(100.8)
    now[0] = 100.3
    a.hold_back([[0, "left", "down"], [500, "left", "up"]])                   # waits until 0.8 s after the key
    a.hold_back([[0, "z", "down"], [40, "z", "up"], [300, "left", "down"]])  # its arrow 0.3 s in: 0.2 s more
    a.hold_back([[0, "z", "down"], [40, "z", "up"], [500, "left", "down"]])  # 0.5 s in: 0.8 s after it anyway
    a.hold_back([[0, "alt", "down"], [40, "alt", "up"]])                      # a jump: the attack lock, as before
    lock = (COMBAT.attack_lock_ms + actions_mod.THROW_LOCK_EXTRA_MS) / 1000
    assert slept == [0.5, 0.2, round(100.0 + lock - 100.3, 3)]
    # A jump back's and a jump throw's attack keys come into the plan: the arrows wait from there.
    for method, key_ms in (("jump_back", actions_mod.JB_THROW_MS), ("jump_throw", JUMP_ATTACK_MS)):
        now[0] = 200.0
        getattr(a, method)(RIGHT)
        assert a.arrow_until == pytest.approx(200.0 + (key_ms + actions_mod.ARROW_LOCK_MS) / 1000), method


def test_the_movers_walks_press_pick_up_as_every_walk_does():
    """Walking to an x (after a trip, before a jump up or a drop) presses pick-up every 150 ms and on
    through the slide, like Actions.walk: anything she passes over is looted."""
    from gamelab.control.actions import WALK_PICK_AFTER_MS, WALK_PICK_EVERY_MS
    from gamelab.control.mover import Mover
    events, _ = Mover(None, actions([]), None).walk_events(150.0, tolerance=8)
    picks = [e[0] for e in events if e[1] == "z" and e[2] == "down"]
    release = next(e[0] for e in events if e[1] == "right" and e[2] == "up")
    assert picks[0] == 0 and picks[1] == WALK_PICK_EVERY_MS and release < picks[-1] < release + WALK_PICK_AFTER_MS


def test_no_jump_in_her_key_plans_is_keyed_in_the_kicks_blind_spot():
    """A jump keyed 60-149 ms after the arrow, from standing, can find her three of the game's steps into her
    run: no kick, and she leaves at 126 units/s (three of the five jump-backs recorded at 60 fps, keyed 80 ms
    after the arrow, left at 121-127; of the jumps of the sweeps keyed 80-120 ms after it, 8 of 12)."""
    from gamelab.control import actions as actions_mod
    from gamelab.control.mover import MIN_RUN
    from gamelab.motion import body
    from gamelab.motion.physics import LAGS, SHOE, run_ms, sure_jump_ms
    sent = []
    actions(sent).jump_back(RIGHT)
    (_, events), = sent
    at = lambda k, how: next(t for t, name, h_ in events if name == k and h_ == how)
    gap = at("alt", "down") - at("left", "down")
    assert gap == actions_mod.JB_RUN_MS and sure_jump_ms(gap) == gap and body.short_takeoff_chance(gap, SHOE, LAGS) <= 0.01
    # The mover's shortest run-up is keyed as its length says (not put off), outside the blind spot.
    assert run_ms(MIN_RUN) == round(MIN_RUN / 0.149 + 54) and body.jump_window_ok(run_ms(MIN_RUN), SHOE, LAGS)


def test_a_jump_throw_that_turns_her_turns_in_the_air_or_first_with_its_wait():
    # She jumps at once and taps the arrow in the air (AIR_TURN_MS after the jump key, once she's off the ground), the
    # attack AIR_TURN_LEAD_MS after that arrow at the soonest (AIR_TURN_ATTACK_MS: live on 2026-10-25, 25 ms after it,
    # 10 of 28 air turns went the wrong way), as the jump backs turn: by the engine she comes down 3.8 units on
    # (2.7-4.3), against 5.3 with the tap first and the jump JUMP_THROW_TURN_MS after it, the attack 160 ms sooner.
    from maplehelper.game import MOVEMENT
    from gamelab.control import actions as actions_mod
    from gamelab.control.actions import JUMP_THROW_TURN_MS
    from gamelab.motion import body
    from gamelab.motion.physics import AIR_TURN_ATTACK_MS, AIR_TURN_LEAD_MS, AIR_TURN_MS, SHOE
    sent = []
    a = actions(sent)
    a.facing = "left"
    a.jump_throw(RIGHT)
    (_, events), = sent
    at = lambda k, how: next(t for t, name, h_ in events if name == k and h_ == how)
    assert at("alt", "down") == 0 and at("right", "down") == AIR_TURN_MS and at("f", "down") == AIR_TURN_ATTACK_MS
    assert at("f", "down") - at("right", "down") == AIR_TURN_LEAD_MS >= 60
    assert a.facing == RIGHT
    # An attack that must come sooner (air_turn False), or with AIR_TURN off: the tap first, and the jump once its
    # walk has stopped.
    for kw in ({"air_turn": False}, {}):
        sent.clear()
        a.lock_until, a.facing = 0.0, "left"
        old = actions_mod.AIR_TURN
        actions_mod.AIR_TURN = bool(kw)
        try:
            a.jump_throw(RIGHT, **kw)
        finally:
            actions_mod.AIR_TURN = old
        (_, events), = sent
        assert at("right", "down") == 0 and at("right", "up") == MOVEMENT.tap_turn_ms
        lift = at("alt", "down") / 1000 + JUMP_LAG
        assert at("alt", "down") == JUMP_THROW_TURN_MS and lift > body.hold_time(MOVEMENT.tap_turn_ms, SHOE, LAGS) + 0.03
    # Facing the right way already, no tap, and the jump goes at once; its attack as late as asked.
    sent.clear()
    a.lock_until = 0.0
    a.jump_throw(RIGHT, 230)
    assert [e[1] for e in sent[0][1]] == ["alt", "alt", "f", "f"] and sent[0][1][0][0] == 0 and sent[0][1][2][0] == 230


def test_a_jump_forward_holds_the_arrow_through_its_kick_and_throws_that_way():
    from gamelab.control.actions import FORWARD_HOLD_MS
    from gamelab.motion.physics import FORWARD_JUMP_MS
    sent = []
    a = actions(sent)
    a.facing = "left"
    a.jump_throw(RIGHT, 60, way=1)
    (_, events), = sent
    at = lambda k, how: next(t for t, name, h_ in events if name == k and h_ == how)
    assert at("right", "down") == 0 and at("alt", "down") == FORWARD_JUMP_MS
    assert at("right", "up") == FORWARD_JUMP_MS + FORWARD_HOLD_MS and at("f", "down") == FORWARD_JUMP_MS + 60
    assert a.facing == RIGHT and [e[1] for e in events].count("right") == 2  # the held arrow, no tap besides


def test_the_movers_drop_steps_off_a_spot_where_it_wouldnt_go_through(monkeypatch):
    """Within 20 units of the move's spot does for the mover's drop, unless a drop there wouldn't go through
    (on a foothold that forbids it, or by a rope's top): then it walks to the nearest spot that would."""
    from gamelab.control import mover as mover_mod
    from gamelab.localization import mapdata
    from gamelab.motion.route import Move
    from test_navigation import by_y, tree
    monkeypatch.setattr(mover_mod.time, "sleep", lambda s: None)
    t = tree()
    t["foothold"]["1"]["1"]["2"]["forbidFallDown"] = 1  # A's 100..200
    m = mapdata.parse(1, t, "Test Map", {})
    a, c = by_y(m, 300, 100), by_y(m, 500, 400)
    sent = []
    acts = actions(sent)
    acts.keys.down = SimpleNamespace(name="down")
    mover = mover_mod.Mover(None, acts, m)
    for here, walks in ((104.0, {"left"}), (90.0, set())):
        sent.clear()
        mover._drop(Move("drop", a.id, c.id, 92.0, 92.0), SimpleNamespace(where=SimpleNamespace(x=here)))
        (_, events), = sent
        assert {name for _, name, how in events if name in ("left", "right") and how == "down"} == walks, here


def test_climbing_a_rope_by_a_portal_the_mover_holds_up_only_once_she_is_off_the_ground(monkeypatch):
    """Up on the ground at a portal takes it (whether the game takes one in the air too isn't known: mover.PORTAL_UP_MS):
    under a rope there, the jump's key goes first and up PORTAL_UP_MS after it, and running at one up goes
    PORTAL_UP_MS after the jump's key, not with it; anywhere else up goes first under it, and with the jump on a run."""
    from gamelab.control import mover as mover_mod
    from gamelab.localization import mapdata
    from gamelab.motion.route import Move
    from test_navigation import by_y, tree
    t = tree()
    m = mapdata.parse(1, t, "Test Map", {})
    c, d = by_y(m, 500, 400), by_y(m, 300, 700)
    rope = m.ropes[0]
    sent = []
    acts = actions(sent)
    acts.keys.up = SimpleNamespace(name="up")
    mover = mover_mod.Mover(None, acts, m)
    climb = Move("climb", c.id, d.id, rope.x, rope.x, rope_x=rope.x, rope_top=rope.top, rope_bottom=rope.bottom)
    here = SimpleNamespace(where=SimpleNamespace(x=rope.x))
    for portal, first in ((False, "up"), (True, "alt")):
        monkeypatch.setattr(m, "portal_near", lambda p, x, portal=portal: portal)
        sent.clear()
        mover._climb(climb, here)
        (_, events), = sent
        assert events[0][1] == first and events[0][2] == "down"
        up_at = next(e[0] for e in events if e[1] == "up" and e[2] == "down")
        assert up_at == (40 + mover_mod.PORTAL_UP_MS if portal else 0)
        sent.clear()
        mover._climb(climb, SimpleNamespace(where=SimpleNamespace(x=rope.x - 100)))  # a run at it, and the jump
        (_, events), = sent
        up_at = next(e[0] for e in events if e[1] == "up" and e[2] == "down")
        jump_at = next(e[0] for e in events if e[1] == "alt" and e[2] == "down")
        assert up_at == jump_at + (mover_mod.PORTAL_UP_MS if portal else 0)


def test_a_jump_that_turns_her_in_the_air_has_its_attack_60_ms_after_the_arrow_where_the_kind_allows():
    """reach.turn_key_ms: a Horny or Green Mushroom (51 tall) is picked by an attack AIR_TURN_ATTACK_MS (120 ms) after
    the jump key, the turn's arrow AIR_TURN_LEAD_MS (60) before it (25 ms, as until 2026-10-26: 10 of 28 such turns
    threw the wrong way live); a pig (38) or a slime (40) only sooner, its attack as late as it's still picked, a snail
    by none (JUMP_BACK_ATTACK_MS: none can turn and pick it). A jump back at a pig keys its attack so (fight._turn_key,
    from the nearest monster the way it throws)."""
    from gamelab.attack import reach
    from gamelab.behavior.fight import _turn_key
    from gamelab.motion.physics import AIR_TURN_ATTACK_MS, JUMP_BACK_ATTACK_MS
    assert reach.turn_key_ms(51) == AIR_TURN_ATTACK_MS == 120 and reach.air_hits(51, 145)
    assert JUMP_BACK_ATTACK_MS < reach.turn_key_ms(38) < AIR_TURN_ATTACK_MS and reach.air_hits(38, reach.turn_key_ms(38))
    assert not reach.air_hits(38, reach.turn_key_ms(38) + 5) and reach.turn_key_ms(24) == JUMP_BACK_ATTACK_MS
    from test_attack import PLAT, target
    h = SimpleNamespace(model=None)
    s = SimpleNamespace(where=SimpleNamespace(x=500.0, y=500.0), platform=PLAT, monsters=[target(560), target(300)])
    assert _turn_key(h, s, ("jump_back", RIGHT)) == reach.turn_key_ms(38)  # the pig on her right
    s.monsters = [target(560, kind="orange-mushroom")]
    assert _turn_key(h, s, ("jump_throw", RIGHT)) == AIR_TURN_ATTACK_MS and _turn_key(h, s, ("throw", RIGHT)) == 120


def test_a_jump_backs_path_and_a_planned_turning_shot_go_by_the_attack_key_as_pressed():
    """fight.attack_gap: the path a throw's kill is credited by (FightState.aimed: a gain soon after its key) is worked
    out where the attack picks; a jump back without a plan has its attack at the turn key (fight._turn_key: 120, a
    pig's 105), as Actions.jump_back keys it (until the review of 2026-10-26 at 85, 35 ms lower in the jump). A plan's
    shot planned turning (reach.Shot.turn) goes at its own key, which reach.shots worked out for its kind (a pig's 105):
    not pushed to 120."""
    from gamelab.attack import reach
    from gamelab.attack.stance import JumpPlan
    from gamelab.behavior.fight import _throw, attack_gap
    from gamelab.motion.physics import AIR_TURN_ATTACK_MS
    acts = SimpleNamespace(faces=lambda side: False)
    assert attack_gap(acts, ("jump_back", RIGHT)) == AIR_TURN_ATTACK_MS
    assert attack_gap(acts, ("jump_back", RIGHT), 105) == 105
    plan = JumpPlan(105, 0, True, "reach", reach.Shot(1, None, key_ms=105, turn=True))
    assert attack_gap(acts, ("jump_throw", RIGHT, plan), AIR_TURN_ATTACK_MS) == 105
    unplanned = JumpPlan(60, 0, True, "reach", reach.Shot(1, None, key_ms=60, turn=False))
    assert attack_gap(acts, ("jump_throw", RIGHT, unplanned), AIR_TURN_ATTACK_MS) == AIR_TURN_ATTACK_MS
    sent = []
    acts.jump_throw = lambda side, *a, **k: sent.append((a, k))
    _throw(acts, ("jump_throw", RIGHT, plan), plan, AIR_TURN_ATTACK_MS)
    _throw(acts, ("jump_throw", RIGHT, unplanned), unplanned, AIR_TURN_ATTACK_MS)
    assert sent == [((105, 0, True), {"turn_key": 105}), ((60, 0, True), {})]
