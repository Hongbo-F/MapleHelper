"""The evade work (2026-10-28; the user, watching a movement-only walk: "Looks like there was no attempt to avoid hitting
the mobs"): her navigator's evade step for a move a monster is in the way of (R1: a refuge along her platform or a move
off it, waiting there while it stays safe, else past it as a watched chain, never the mover), the mover's moves watched
(R2), the stand check to her invulnerability's end and the contact rule (R3), ways around (R4), the planner's monsters
without an escort (R5), the small fixes (R6: a run-up's step back in a leg's path, a hop timed for a monster walking at her,
a queued drop's landing and a rope's top checked against the newest tracks), and her retreat's dodge (R7). On the made-up
map of test_navigation (no game data needed); the simulator's walk on Tree Dungeon at the end (skipped without its data)."""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gamelab.control import navigator as nv
from gamelab.control.navigator import Navigator
from gamelab.localization import mapdata
from gamelab.localization.monsters import Monster, MonsterTrack
from gamelab.localization.pose import Where
from gamelab.localization.scene import Scene
from gamelab.motion.route import Planner

from test_navigation import by_y, tree, walker

NOW = 100.0


@pytest.fixture
def model():
    return mapdata.parse(1, tree(), "Test Map", {})


def _navigator(model, escort=None, **kw):
    key = lambda name: SimpleNamespace(name=name)
    keys = SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), up=key("up"), down=key("down"),
                           pick_up=None)
    return Navigator(None, SimpleNamespace(keys=keys, sent_at=0.0), model, escort=escort, **kw)


def _trip(model, monkeypatch, snails, x=300.0, escort=None, mode=None, kind="snail", on=None):
    """A trip from C at map x to D (along C to the rope at 700, up it), the look it plans from with snails standing on
    C (or `on`: (platform, x)s) at `snails` (one look: their walking not known), the clock stopped at NOW, no hop over
    them (flow.hop: the evade step's own cases). Returns the navigator, the trip, the look, the route, the chain's
    monsters and what was carried out (the chains: _carry_out_chain stubbed)."""
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    nav = _navigator(model, escort=escort)
    nav.flow.hop = lambda *a, **k: None
    monkeypatch.setattr(nv.time, "sleep", lambda s: None)
    monkeypatch.setattr(nv.time, "monotonic", lambda: NOW)
    st = nav.begin(d.id, mode=mode)
    st.moved_at = -1e9
    seen = [walker(sx, kind=kind, pose="stand", platform=c.id) for sx in snails]         + [walker(ox, kind=kind, pose="stand", platform=op) for op, ox in (on or ())]
    s = Scene(t=NOW, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0,
              where=Where(x, 500, c, (0, 0)), searched=True, wide=True, monsters=seen)
    path = Planner(model).plan(c.id, x, d.id)
    monsters, _ = nav._chain_monsters(st, s, path)
    carried = []

    def carry(st_, s_, path_, legs, keys, why, dst, mons=()):
        carried.append((legs, why))
        end = legs[-1].end
        after = replace(s_, where=Where(end.x, end.platform.y_at(end.x), end.platform, (0, 0)))
        return [leg.move for leg in legs], end.platform, after
    nav._carry_out_chain = carry
    return nav, st, s, path, monsters, carried


def _events(monkeypatch):
    got = []
    monkeypatch.setattr(nv.telemetry, "emit", lambda topic, **kw: got.append((topic, kw)))
    return got


# --- R1: the evade step --------------------------------------------------------------------------------------------

def test_a_move_a_monster_blocks_where_she_cant_wait_takes_her_to_the_nearest_safe_spot_along_her_platform(
        model, monkeypatch):
    # A snail 50 units on along C, standing, in the way of her run to the rope: within a second it can reach her, so she
    # can't wait (_safe_to_stand); the refuge is the nearest spot to the left that a step there and standing there
    # EVADE_HOLD_S keep clear of it (a step of ~90 units), gone to as a chain (one 'step' leg), said, an 'evade' event.
    events = _events(monkeypatch)
    nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [350.0])
    legs, keys, why = nav._chain(st, s, path, monsters, NOW)
    assert legs == [] and why == "monster"
    assert not nav._safe_to_stand(300.0, s.where.platform, path[0], monsters)
    assert nav._wait_for_monster(st, s, path, monsters, legs, why, NOW, NOW) is True
    assert len(carried) == 1 and [leg.move.kind for leg in carried[0][0]] == ["step"]
    end = carried[0][0][-1].end.x
    assert 190 <= end <= 260 and end < 300  # (away from it, to the left)
    assert st.refuge == (s.where.platform.id, end) and st.refuges == 1 and st.evade_since == NOW
    assert any(t == "event" and kw.get("kind") == "evade" and kw.get("step") == "spot" for t, kw in events)
    assert st.trip.moves == []  # (a step along her platform is no move of the route's: _note_landing's)


def test_at_a_refuge_that_stays_safe_she_waits_and_looks_again(model, monkeypatch):
    # At the refuge (st.refuge), her waits spent, the way still blocked by a snail too far to reach her soon: she waits
    # there WAIT_S and looks again (no chain), within the budget.
    d = by_y(model, 300, 700)
    nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [], on=[(d.id, 700.0)])  # (by the rope's top)
    legs, keys, why = nav._chain(st, s, path, monsters, NOW)
    assert legs == [] and why == "monster" and nav._safe_to_stand(300.0, s.where.platform, path[0], monsters)
    st.waited, st.refuge, st.evade_since = nv.WAIT_LOOKS, (s.where.platform.id, 300.0), NOW - 1.0
    waits = st.trip.waits
    assert nav._wait_for_monster(st, s, path, monsters, legs, why, NOW, NOW) is True
    assert carried == [] and st.trip.waits == waits + 1


def test_with_no_refuge_she_goes_past_the_monster_in_the_way_as_a_watched_chain_said(model, monkeypatch):
    # Snails 60 units either way on C: no spot she can step to clear of both (the step to one side passes the other),
    # no move off C but the rope beyond one: no refuge (NO_REFUGE). She goes past the one in the way (the chain planned
    # without it, the other still kept clear of): a chain, said, an 'evade' event 'pass' and a backstop.
    events = _events(monkeypatch)
    nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [360.0, 240.0])
    legs, keys, why = nav._chain(st, s, path, monsters, NOW)
    assert legs == [] and why == "monster"
    assert nav._refuge(st, s, monsters) is None
    assert nav._wait_for_monster(st, s, path, monsters, legs, why, NOW, NOW) is True
    assert len(carried) == 1 and [leg.move.kind for leg in carried[0][0]] == ["climb"]
    backstops = [kw["text"] for t, kw in events if t == "event" and kw.get("kind") == "backstop"]
    assert backstops and backstops[0].startswith("NO_REFUGE") and "past the snail" in backstops[0]
    assert any(kw.get("step") == "pass" for t, kw in events if t == "event")
    # (its record says why: the move, the snail in its way on her platform, the looks waited, the shots, its episode)
    rec = next(kw for t, kw in events if t == "event" and kw.get("kind") == "backstop")
    assert rec["move"][:3] == [path[0].kind, path[0].src, path[0].dst] and (rec["episode"], rec["nth"]) == (1, 1)
    assert rec["blocker"] == {"kind": "snail", "platform": s.where.platform.id, "x": pytest.approx(360.0, abs=1),
                              "rel": "same"} and rec["waited"] == 0 and rec["shots"] == 0
    assert st.refuge is None and st.passed == (s.where.platform.id, NOW)
    assert st.evade_key is None and st.evade_since is None  # (made: its budget gone with it, BLOCK_MEMORY's _made)


def test_her_budget_spent_she_goes_past_without_another_refuge(model, monkeypatch):
    # EVADE_S since the first refuge decision for this move: past it (EVADE_S), however safe a refuge is.
    events = _events(monkeypatch)
    nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [350.0])
    legs, keys, why = nav._chain(st, s, path, monsters, NOW)
    st.evade_since = NOW - nv.EVADE_S - 0.1
    assert nav._wait_for_monster(st, s, path, monsters, legs, why, NOW, NOW) is True
    assert [leg.move.kind for leg in carried[0][0]] == ["climb"]
    assert any(kw.get("text", "").startswith("EVADE_S") for t, kw in events if kw.get("kind") == "backstop")
    # ... and so with EVADE_REFUGES taken, or the trip's deadline near
    for setup, name in ((lambda st: setattr(st, "refuges", nv.EVADE_REFUGES), "EVADE_REFUGES"),
                        (lambda st: setattr(st, "deadline", NOW + 1.0), "DEADLINE")):
        nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [350.0])
        setup(st)
        assert nav._why_past(st).startswith(name) and nav._refuge(st, s, monsters) is not None
        assert nav._wait_for_monster(st, s, path, monsters, [], "monster", NOW, NOW) is True
        assert [leg.move.kind for leg in carried[0][0]] == ["climb"]


def test_a_pass_leaves_out_the_monsters_in_the_way_one_at_a_time_else_goes_blind_never_the_mover(model, monkeypatch):
    # Three snails on her way along C to the rope: the pass leaves out the first in the way, then the next (PASS_TRIES),
    # and with them all out a chain plans: never the mover for a move a monster is in the way of.
    nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [400.0, 500.0, 600.0], x=300.0)
    assert nav._go_past(st, s, path, monsters, NOW, "NO_REFUGE: test") is True
    assert [leg.move.kind for leg in carried[0][0]] == ["climb"]
    monkeypatch.setattr(nv, "PASS_TRIES", 1)  # one left out only: still blocked, so blind
    nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [400.0, 500.0, 600.0], x=300.0)
    said = []
    nav.log = said.append
    assert nav._go_past(st, s, path, monsters, NOW, "NO_REFUGE: test") is True
    assert any("(blind)" in line for line in said)


def test_the_evade_step_is_off_as_before(model, monkeypatch):
    # EVADE off: a move she can't wait for goes to the mover, as before (False: the caller's move alone).
    monkeypatch.setattr(nv, "EVADE", False)
    nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [350.0])
    assert nav._wait_for_monster(st, s, path, monsters, [], "monster", NOW, NOW) is False and carried == []


def test_a_refuge_move_off_her_platform_where_no_spot_along_it_is_safe(model, monkeypatch):
    # From D (600..800 at y 300) at 700, down to C, a slime standing at 770 on D: none of D's spots is clear of it for
    # EVADE_HOLD_S (it's 200 wide), so the refuge is a move off D that chains clear (the walk-off at its left end, onto
    # C), the quickest that lands where she can stand (refuge_s: math.inf, a retreat after a stop, here); and not back
    # onto a platform her last pass set off from within PASS_COMMIT_S.
    d, c = by_y(model, 300, 700), by_y(model, 500, 400)
    nav = _navigator(model)
    monkeypatch.setattr(nv.time, "sleep", lambda s: None)
    monkeypatch.setattr(nv.time, "monotonic", lambda: NOW)
    st = nav.begin(c.id, refuge_s=math.inf)
    st.moved_at = -1e9
    s = Scene(t=NOW, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0,
              where=Where(700, 300, d, (0, 0)), searched=True, wide=True,
              monsters=[walker(770.0, kind="slime", pose="stand", platform=d.id)])
    path = Planner(model).plan(d.id, 700, c.id)
    monsters, _ = nav._chain_monsters(st, s, path)
    ref = nav._refuge(st, s, monsters)
    assert ref is not None and ref[0] == "move" and ref[4][0] != d.id
    st.passed = (ref[4][0], NOW)  # (just passed from there: not back onto it, PASS_COMMIT_S)
    again = nav._refuge(st, s, monsters)
    assert again is None or again[4][0] != ref[4][0]


def test_with_an_escort_the_evade_step_comes_before_her_last_shot_and_the_pass_regardless(model, monkeypatch):
    # _meet, nothing to shoot (no star reaches), the way around tried, no hop, her waits spent: a refuge, before the
    # last shot (LAST_SHOT) and the pass regardless.
    shot = []
    escort = SimpleNamespace(threat=lambda kind: True, clear_for=lambda *a, **k: shot.append(a) or None)
    nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [350.0], escort=escort, mode="transit")
    nav.mover.locate = lambda mobs=False, wide=False: s
    st.waited, st.avoid[nv._key(path[0])] = nv.WAIT_LOOKS, NOW
    legs, keys, why = nav._chain(st, s, path, monsters, NOW)
    assert legs == [] and why == "monster"
    assert nav._meet(st, s, path, monsters, legs, why, NOW, NOW) is True
    assert carried and [leg.move.kind for leg in carried[0][0]] == ["step"] and st.refuge is not None


# --- R3: the stand check to her invulnerability's end, the contact rule ------------------------------------------------

def test_the_stand_check_behind_waiting_runs_to_her_invulnerabilitys_end_and_half_a_second_more(model, monkeypatch):
    # A snail 100 units off reaches her standing in about 1.4 s: with 1.2 s of her invulnerability left, a second's
    # stand counted from it (as before) is no check; to its end and STAND_PAST_S, it is.
    nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [400.0])
    c = s.where.platform
    nav.flow.immune = 0.0
    assert nav._safe_to_stand(300.0, c, path[0], monsters)
    nav.flow.immune = 1.2
    assert not nav._safe_to_stand(300.0, c, path[0], monsters)
    monkeypatch.setattr(nv, "STAND_WINDOW", False)
    assert nav._safe_to_stand(300.0, c, path[0], monsters)


def test_a_touch_she_cant_step_out_of_is_taken_as_given_and_she_goes_through_in_the_window_after_it(
        model, monkeypatch):
    # Snails on her either side, 45 off (their boxes on hers within CONTACT_S whatever she does): the contact rule takes
    # the touch as given and plans with the window after it (flow.immune), and the chain past them plans; not while she's
    # invulnerable, not again within CONTACT_AGAIN_S unless a touch was told since.
    events = _events(monkeypatch)
    nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [345.0, 255.0])
    hit = nav._contact(st, s, path[0], monsters)
    assert hit is not None and hit[1] <= nv.CONTACT_S
    assert nav._wait_for_monster(st, s, path, monsters, [], "monster", NOW, NOW) is True
    assert [leg.move.kind for leg in carried[0][0]] == ["climb"]
    assert any(kw.get("step") == "contact" for t, kw in events if t == "event")
    assert st.contact_at == NOW and nav._contact(st, s, path[0], monsters) is None  # (once in CONTACT_AGAIN_S)
    nav.flow.immune = 0.5
    st.contact_at = -1e18
    assert nav._contact(st, s, path[0], monsters) is None  # (invulnerable already)
    nav.flow.immune = 0.0
    far = [replace(t, seen_at=-1.0) for t in monsters]  # (sightings a second old: not seen live)
    assert nav._contact(st, s, path[0], far) is None


# --- R4: ways around ------------------------------------------------------------------------------------------------

def test_without_an_escort_a_way_around_costs_every_move_the_same_monsters_block(model, monkeypatch):
    # The first block with her spot safe: _around (as an escort's), not the first move's cost alone.
    nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [500.0])
    arounds = []
    nav._around = lambda st_, s_, move, mons, at: arounds.append(move) or st_.avoid.update({nv._key(move): at})
    assert nav._wait_for_monster(st, s, path, monsters, [], "monster", NOW, NOW) is True
    assert arounds == [path[0]]
    monkeypatch.setattr(nv, "AROUND_ALL", False)
    st.avoid.clear()
    assert nav._wait_for_monster(st, s, path, monsters, [], "monster", NOW, NOW) is True
    assert arounds == [path[0]] and nv._key(path[0]) in st.avoid


# --- R5: the planner's monsters without an escort ----------------------------------------------------------------------

def _track(tid, x, platform, last_seen, state="hidden", vx=0.0, kind="snail"):
    from gamelab.localization.tracking import Track
    from gamelab.perception.mobs import Mob
    return Track(tid, kind, platform, x, 500, Mob(kind, 0, 0, 30, 30, 0.9), last_seen - 2, last_seen, vx=vx,
                 state=state)


def test_without_an_escort_her_chains_keep_clear_of_the_feeds_hidden_monsters_walking_as_followed_too(
        model, monkeypatch):
    # The look's monster whose walking its tracker can't tell yet (one look) walks as the feed follows it (vx); the feed's
    # monster out of sight on her way is kept clear of too; a monster the look placed on no platform (in the air) is put
    # on the one under its feet; and a hidden track on her with no touch for GHOST_S while she can be touched is none.
    c = by_y(model, 500, 400)
    seen = walker(600.0, kind="snail", pose="stand", platform=c.id)
    seen = replace(seen, track=7)
    air = Monster(seen.mob, 450.0, 480.0, None, None, 0, "stand")
    tracks = [_track(7, 600.0, c.id, NOW, state="live", vx=-40.0), _track(8, 520.0, c.id, NOW - 0.5),
              _track(9, 302.0, c.id, NOW - 1.5)]
    nav = _navigator(model)
    nav.senses = SimpleNamespace(touch=None, invulnerable_until=lambda: -1e18)
    monkeypatch.setattr(nv.time, "monotonic", lambda: NOW)
    st = nav.begin(by_y(model, 300, 700).id)
    s = Scene(t=NOW, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0,
              where=Where(300, 500, c, (0, 0)), searched=True, wide=True, monsters=[seen, air], tracks=tracks)
    path = Planner(model).plan(c.id, 300, by_y(model, 300, 700).id)
    monsters, _ = nav._chain_monsters(st, s, path)
    xs = sorted((round(t.x), t.vx) for t in monsters)
    assert (600, -40.0) in xs  # (walking as followed)
    assert any(x == 450 for x, _ in xs)  # (placed on C, under its feet)
    assert any(x == 520 for x, _ in xs)  # (out of sight, on her way)
    assert not any(x == 302 for x, _ in xs)  # (the ghost on her)
    monkeypatch.setattr(nv, "UNSEEN_AVOID", False)
    st = nav.begin(by_y(model, 300, 700).id)
    monsters, _ = nav._chain_monsters(st, s, path)
    assert sorted((round(t.x), t.vx) for t in monsters) == [(600, None)]


# --- R2: the mover's moves watched -------------------------------------------------------------------------------------

def test_a_move_of_the_movers_is_watched_and_cut_short_on_a_touch(monkeypatch):
    # Mover._send with a watch: the plan sent without waiting, the watch asked between glances, cut (every key let go:
    # client.cancel) and MoveCut raised when it says why; unwatched, Actions.play as before.
    from gamelab.control import mover as mv
    sent, cancelled = [], []
    monkeypatch.setattr(mv.client, "send", lambda events, **kw: sent.append(events) or "p1")
    monkeypatch.setattr(mv.client, "done", lambda pid: False)
    monkeypatch.setattr(mv.client, "cancel", lambda pid: cancelled.append(pid) or True)
    played = []
    actions = SimpleNamespace(hold_back=lambda e: None, play=lambda e, name: played.append(name), facing=None,
                              ended=lambda: None, pressed=False, sent_at=0.0)
    m = mv.Mover(None, actions, None)
    m._send([[0, "right", "down"], [200, "right", "up"]], "walk")
    assert played == ["walk"] and sent == []
    asks = iter([None, None, "touched (MISS: HP 90% -> 90%)"])
    m.watch = lambda: next(asks)
    with pytest.raises(mv.MoveCut, match="MISS"):
        m._send([[0, "right", "down"], [200, "right", "up"]], "walk")
    assert len(sent) == 1 and cancelled == ["p1"] and actions.pressed


def test_a_run_ups_back_off_into_a_monster_climbs_from_under_the_rope_or_refuses_the_move(model, monkeypatch):
    # Mover._climb too close to the rope for a run at it: the back-off is checked (Mover.clear); into a monster, from
    # under the rope instead if that walk is clear, else MoveRefused (the navigator: a monster in its way, AVOID_SECONDS).
    from gamelab.control import mover as mv
    from gamelab.motion.route import Move
    c = by_y(model, 500, 400)
    rope = model.ropes[0]
    m = mv.Mover(None, SimpleNamespace(keys=SimpleNamespace(up=SimpleNamespace(name="up"),
                                                             jump=SimpleNamespace(name="alt")),
                                       arrow=lambda d: d), model, log=lambda *a: None)
    walked, climbed = [], []
    m.walk_to = lambda x, tolerance=0, s=None: walked.append(x) or s
    m._climb_from_under = lambda hold, portal=False: climbed.append(hold)
    m._send = lambda events, name: climbed.append(name)
    move = Move("climb", c.id, by_y(model, 300, 700).id, rope.x, rope.x, rope_x=rope.x, rope_top=rope.top,
                rope_bottom=rope.bottom)
    s = SimpleNamespace(where=Where(rope.x - 30, 500, c, (0, 0)))
    m.clear = lambda x0, x1: "a snail by it" if x1 < x0 else None  # (the back-off, leftwards, meets one)
    m._climb(move, s)
    assert walked == [rope.x] and climbed and climbed[0] != "climb"  # (from under it)
    m.clear = lambda x0, x1: "a snail by it"  # (and under it too)
    walked.clear()
    with pytest.raises(mv.MoveRefused):
        m._climb(move, s)
    assert walked == []


# --- R6: the small fixes -----------------------------------------------------------------------------------------------

def test_a_run_ups_step_back_is_in_the_legs_path(model, monkeypatch):
    # A running jump from standing 5 units from its takeoff steps back first (flow._runup): the leg's path has that
    # step back (RUNUP_SPANS), so a monster just behind her is met; as one straight span it went unseen.
    from test_flow import KEYS

    from gamelab.motion import flow as fl
    from gamelab.motion.flow import Body, Flow, Keys
    from gamelab.motion.route import Move
    c = by_y(model, 500, 400)
    f = Flow(model, KEYS)
    m = Move("jump", c.id, c.id, 600, 700, "right", takeoff=600.0)
    leg = f._jump_from(m, Body(0.0, 595.0, c), Keys(KEYS), "run", 600.0)
    if leg is None:
        pytest.skip("no such jump on the made-up map")
    assert len(leg.spans) == 2 and leg.spans[0].x1 < 595.0
    behind = [MonsterTrack(c.id, leg.spans[0].x1 - 30.0, speed=0.0, half=10.0, vx=0.0)]
    assert f.danger(leg, behind) is not None
    monkeypatch.setattr(fl, "RUNUP_SPANS", False)
    leg = f._jump_from(m, Body(0.0, 595.0, c), Keys(KEYS), "run", 600.0)
    assert len(leg.spans) == 1


def test_a_hop_over_a_monster_whose_walking_isnt_known_is_timed_as_if_it_walked_at_her(model, monkeypatch):
    # flow.hop: a monster seen once (vx None) closes on her at its speed and hers (HOP_AT_HER), so she takes off sooner
    # than for one standing (vx 0): as if it stood, it came on into her before her jump.
    from test_flow import KEYS

    from gamelab.motion import flow as fl
    from gamelab.motion.flow import Body, Flow, Keys
    c = by_y(model, 500, 400)
    f = Flow(model, KEYS)
    unknown = MonsterTrack(c.id, 500.0, speed=43.75, half=19.0, vx=None, height=26.0)
    standing = replace(unknown, vx=0.0)
    a, b = f.hop(Body(0.0, 200.0, c), unknown, 1, Keys(KEYS)), f.hop(Body(0.0, 200.0, c), standing, 1, Keys(KEYS))
    assert a is not None and b is not None and a.move.x < b.move.x  # (takes off sooner, further back)
    monkeypatch.setattr(fl, "HOP_AT_HER", False)
    assert f.hop(Body(0.0, 200.0, c), unknown, 1, Keys(KEYS)).move.x == pytest.approx(b.move.x)


def test_a_queued_drop_whose_landing_the_newest_tracks_have_a_monster_on_is_withdrawn(model, monkeypatch):
    # _queued_meets: the chain queued to start next, its first leg a drop from A onto C... (here the walk-off from A, not
    # checked) or a climb up the rope: with the feed's newest track by the rope's top on D, the climb meets it (its
    # flight and climb, not the run to it).
    from test_flow import KEYS

    from gamelab.motion.flow import Body, Flow, Keys
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    nav = _navigator(model)
    monkeypatch.setattr(nv.time, "monotonic", lambda: NOW)
    path = Planner(model).plan(c.id, 600, d.id)
    legs, _, why = nav.flow.chain(path, Body(0.0, 600.0, c), [])
    assert [leg.move.kind for leg in legs] == ["climb"]
    queued = nv._Running("q", legs, NOW + 0.05, False)
    state = SimpleNamespace(placed=True, t=NOW, tracks=[_track(3, 700.0, d.id, NOW, state="live", kind="snail")])
    nav.senses = SimpleNamespace(glanced=state, invulnerable_until=lambda: -1e18)
    assert nav._queued_meets(queued) is not None
    state.tracks = [_track(3, 300.0, c.id, NOW, state="live")]  # (behind her on C: the run's, not checked)
    assert nav._queued_meets(queued) is None


def test_touched_on_a_rope_she_looks_at_its_top_first_and_goes_down_again_if_a_monster_is_there(model, monkeypatch):
    # _climb_on after a touch on the rope at 700: with the feed's newest track on D by its top and her up there only
    # after her invulnerability is over, down the rope (to its bottom) instead of up into it; with none there, or up
    # within the window with STAND_PAST_S to spare (the touch left her one: up through it, the next pass gets her clear),
    # up as before. (Without ROPE_HANG: with it, a top taken has her down to HANG_UNDER only and hang there,
    # tests/test_rope_hang.py.)
    monkeypatch.setattr(nv, "ROPE_HANG", False)
    rope = model.ropes[0]
    d = by_y(model, 300, 700)
    nav = _navigator(model)
    played = []
    nav.actions.play = lambda events, name: played.append((name, events))
    monkeypatch.setattr(nv.time, "monotonic", lambda: NOW)
    nav.senses = SimpleNamespace(glanced=SimpleNamespace(t=NOW, tracks=[_track(4, 710.0, d.id, NOW, state="live")]))
    nav._climb_on(rope.top, rope.top + 40, rope.x)
    assert played[-1][0] == "climb_down" and played[-1][1][0][1] == "down"
    nav.senses.invulnerable_until = lambda: NOW + 1.45  # (just touched: up within the window)
    nav._climb_on(rope.top, rope.top + 40, rope.x)
    assert played[-1][0] == "climb_on" and played[-1][1][0][1] == "up"
    nav.senses.invulnerable_until = lambda: -1e18
    nav.senses.glanced.tracks = []
    nav._climb_on(rope.top, rope.top + 40, rope.x)
    assert played[-1][0] == "climb_on" and played[-1][1][0][1] == "up"


def test_a_jump_that_never_moved_her_dot_by_a_ropes_top_means_she_is_on_the_rope_however_near_its_top(model):
    # _rope_top: her dot 6 under D's surface by the rope at 700, never moved over the chain: standing there (STUCK_GAP),
    # as before, unless the chain had her jump (a jump moves her dot within 0.3 s): then she's on the rope, and off it
    # (the evade step's passes from a rope's top were sent again and again: the harness's walk, seed 34).
    d = by_y(model, 300, 700)
    nav = _navigator(model)
    nav.senses = SimpleNamespace(localizer=SimpleNamespace(to_map=lambda dot: (700.0, 306.0)))
    nav.mover.off_rope = lambda after, rope, at=None: "off"
    s = SimpleNamespace(where=Where(700, 300, d, (10.0, 20.0)))
    st = SimpleNamespace(fresh=None)
    assert nav._rope_top(st, s, s) is s
    assert nav._rope_top(st, s, s, jumped=True) == "off"
    # ROPE_CUT: a chain from there cut short for her dot standing where it runs her (0.12 s in, short of STUCK_S): the
    # rope too, the evade step on or off (on the rope under 30's top, 112 chains "ended on platform 30 instead": the
    # harness's walk, seed 9)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nv, "EVADE", False)
        nav.cut_still = False
        assert nav._rope_top(st, s, s, jumped=True) is s
        nav.cut_still = True
        assert nav._rope_top(st, s, s, jumped=True) == "off"


# --- R7: her retreat's dodge -------------------------------------------------------------------------------------------

def test_a_dodging_trip_keeps_clear_of_the_monster_that_touched_her_where_the_look_has_none_by_her(model, monkeypatch):
    # go's `toucher` (her retreat): a touch 0.5 s ago, no monster of the look within TOUCHER_NEAR of her: a stand-in at
    # her x, its walking not known, of the map's kind that touches hardest; none with one by her, or the touch old.
    c = by_y(model, 500, 400)
    m = mapdata.parse(1, tree(), "Test Map", {}, {"2": "Horny Mushroom"}, mob_damage={"Horny Mushroom": 110.0})
    nav = _navigator(m)
    monkeypatch.setattr(nv.time, "monotonic", lambda: NOW)
    nav.senses = SimpleNamespace(touch={"at": NOW - 0.5, "told": NOW - 0.45, "kind": "MISS"})
    s = Scene(t=NOW, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0,
              where=Where(300, 500, by_y(m, 500, 400), (0, 0)), searched=True, wide=True, monsters=[])
    t = nav._toucher(s, [])
    assert t is not None and t.x == 300 and t.vx is None and t.kind == "horny-mushroom" and t.seen_at == -0.5
    assert nav._toucher(s, [MonsterTrack(c.id, 320.0)]) is None
    nav.senses.touch = {"at": NOW - 3.0, "told": NOW - 3.0, "kind": "MISS"}
    assert nav._toucher(s, []) is None


def test_her_retreat_dodges_its_trips_budget_the_time_it_has_left_and_after_a_stop_refuges_to_the_end(monkeypatch):
    # retreat._away: its trips are made with the dodge (Navigator.go's toucher, evade_s, refuge_s, deadline_pass): after a
    # stop the budget is its STOP_S, refuge moves without a bound and no pass for the deadline; for the Lie Detector
    # EVADE_S at most, refuge moves bounded.
    from gamelab.behavior import retreat as rv
    gone = []

    class Nav:
        def __init__(self, *a, **kw):
            self.made = kw
            self.mover = SimpleNamespace(locate=lambda **kw: here)
            self.model = None

        def go(self, *a, **kw):
            gone.append(kw)
            return SimpleNamespace(result="arrived")
    here = SimpleNamespace(where=SimpleNamespace(platform=SimpleNamespace(id=1), x=0.0, y=0.0), monsters=[], t=NOW)
    monkeypatch.setattr(rv, "Navigator", Nav)
    monkeypatch.setattr(rv, "choose", lambda h, nav, s, left, hang: rv.Place(2, None, "a safe platform", 2.0))
    monkeypatch.setattr(rv.time, "monotonic", lambda: NOW)
    h = SimpleNamespace(actions=SimpleNamespace(moves_only=lambda: None), senses=None, model=None, log=lambda *a: None,
                        danger=None, memory=None)
    rv._away(h, NOW, rv.SAFETY)
    assert gone[0]["toucher"] and gone[0]["refuge_s"] == math.inf and not gone[0]["deadline_pass"]
    assert gone[0]["evade_s"] == pytest.approx(rv.STOP_S)
    gone.clear()
    rv._away(h, NOW, rv.LIE_DETECTOR)
    assert gone[0]["refuge_s"] == nv.EVADE_MOVE_S and gone[0]["deadline_pass"] and gone[0]["evade_s"] == nv.EVADE_S


def test_her_retreat_weighs_each_places_way_a_move_by_a_monster_of_the_look_dearer(model):
    # retreat._way_cost: the quickest way onto D from C at 200 is up the rope at 700; a monster of the look on C by the
    # climb's start makes it WAY_BLOCKED_S dearer; none there, none dearer.
    from gamelab.behavior import retreat as rv
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    nav = _navigator(model)
    seen = [walker(700.0, kind="snail", pose="stand", platform=c.id)]
    assert rv._way_cost(nav, c.id, 200.0, seen)(d.id) == pytest.approx(rv.WAY_BLOCKED_S)
    assert rv._way_cost(nav, c.id, 200.0, [])(d.id) == 0.0


# --- the simulator: the walk on Tree Dungeon ----------------------------------------------------------------------------

def test_in_the_simulators_walk_she_evades_and_no_move_a_monster_blocks_goes_to_the_mover():
    # harness `walk` (movement only, as the live QA walk) in the standard world (the crowd, walkers, near-misses): she
    # steps to refuges and goes past, said ('evade' events), and her touches on the mover's moves are none or next to it
    # (the mover only for a move no chain plans: 4 since chains are replanned as they run, 2026-10-30, navigator.REPLAN,
    # drops no chain could plan with a monster by them; 3 before). Seed 3 since work/replan's rebase onto the server's
    # respawn and the chase (2026-10-31): on seed 1 two touches on the mover's 2 moves; of seeds 1-6 two on 1, one on
    # 2, none on 3-6 (D:/MapleHelper-scratch/replan/pos/fix/out/walk.txt).
    from gamelab.sim import harness, scenarios
    if not scenarios.available(scenarios.TREE):
        pytest.skip("Tree Dungeon's game data isn't on this computer")
    out = harness.run_one("walk", 3, "crowd", minutes=1.0, costs="measured", map_id=scenarios.TREE,
                          world=("zero-walls", "near-misses", "walkers"), attack="choice")
    assert out["evades"].get("spot", 0) + out["evades"].get("move", 0) >= 1 and out["evades"].get("pass", 0) >= 1
    assert out["touches_by"]["move"] <= 1 and out["mover_moves"] <= 4


# --- R8: fights ------------------------------------------------------------------------------------------------------

def test_touched_in_a_fight_she_gets_clear_of_the_monster_on_her_before_her_invulnerability_ends(monkeypatch):
    # attack.stance's _close_quarters: a Green Mushroom on her (its body on hers, 50 off), which a fight lets come to the
    # claw's reach (its keep-off under it): a short step to where her next throw takes it (_clear_step), as before;
    # with her invulnerability on after a touch (Pace.window), the escape decides and staying put costs CLEAR_STAY more
    # (FIGHT_CLEAR): she jumps back, clear of it, throwing; one about to die (its bar at KILL_BAR or under) as before,
    # and so with the window over, or FIGHT_CLEAR off.
    from test_danger import HER, build, seen

    from gamelab.attack import stance
    from gamelab.attack.danger import Danger
    model = build([(0, 500, 1000, 500)], [("13", 550, 500)])
    d, p = Danger(model, HER), model.platform(0)
    green = seen("green-mushroom", 550, 500, p.id)
    calm = stance.stance(500, 500, p, [green], danger=d, facing="right")
    touched = stance.stance(500, 500, p, [green], danger=d, facing="right", pace=stance.Pace(window=1.0))
    assert calm[0] == "back" and calm[2] < 50 and touched == ("jump_back", "right"), (calm, touched)
    dying = replace(green, bar=0.2)
    assert stance.stance(500, 500, p, [dying], danger=d, facing="right", pace=stance.Pace(window=1.0)) == \
        stance.stance(500, 500, p, [dying], danger=d, facing="right")
    assert stance.stance(500, 500, p, [green], danger=d, facing="right", pace=stance.Pace(window=0.1)) == calm
    monkeypatch.setattr(stance, "FIGHT_CLEAR", False)
    assert stance.stance(500, 500, p, [green], danger=d, facing="right", pace=stance.Pace(window=1.0)) == calm


# --- the walk: the harness's policy, and `nav --walk` for the live check ------------------------------------------------

def test_a_walk_goes_on_random_trips_to_the_platforms_of_the_qa_walk_saving_the_recording_every_so_often(monkeypatch):
    # behavior.walk: trips to a platform drawn from walk.PLATFORMS on Tree Dungeon (elsewhere any 60 wide), an x 15 in
    # from its ends, TRIP_S each, until the minutes pass; save() every SAVE_S. `nav --walk` takes no platform.
    from gamelab.behavior import walk
    clock = {"t": 0.0}
    monkeypatch.setattr(walk.time, "monotonic", lambda: clock["t"])
    tree_model = SimpleNamespace(id=walk.TREE, places=[SimpleNamespace(id=i, left=0.0, right=200.0) for i in (5, 7, 33,
                                                                                                             34)])
    gone, saved = [], []

    def go(pid, x=None, seconds=0.0):
        gone.append((pid, x, seconds))
        clock["t"] += 100.0
        return SimpleNamespace(result="arrived", moves=[], hits=0, waits=0)
    trips = walk.walk(SimpleNamespace(go=go), tree_model, __import__("random").Random(1), minutes=10,
                      save=lambda: saved.append(clock["t"]))
    assert len(trips) == 6 and {pid for pid, _, _ in gone} <= {7, 33} and all(15 <= x <= 185 for _, x, _ in gone)
    assert all(s == walk.TRIP_S for _, _, s in gone) and saved == [300.0, 600.0]
    other = SimpleNamespace(id=1, places=[SimpleNamespace(id=1, left=0.0, right=50.0), SimpleNamespace(id=2, left=0.0,
                                                                                                        right=90.0)])
    assert [p.id for p in walk.places(other)] == [2]
    from gamelab import __main__ as cli
    assert cli.cmd_nav(SimpleNamespace(walk=None, platform=None, park=False)) == 2


# --- after the review of 2026-10-29 (D:/MapleHelper-scratch/evade/review) ---------------------------------------------

def test_the_walk_on_arrival_is_refused_only_where_standing_where_she_is_is_clear(model, monkeypatch):
    # ARRIVE_STAND, for a trip whose x is what it's for (her retreat's place: go's arrive_waits, TripState.hold_x): on the
    # trip's platform (C), sent on to x 520. A snail by x 560: the walk meets it, standing at 300 doesn't: she stays,
    # the trip short of its x (Trip.short), said. A snail on her at 310: the walk meets it at once, and so does
    # standing: she walks on (refused, she "arrived" where it went on touching her: the review's LD-28, seed 9). For
    # another trip (any x on the platform does: the walk's), and as before (ARRIVE_STAND off), refused either way.
    c = by_y(model, 500, 400)
    monkeypatch.setattr(nv.time, "sleep", lambda s: None)
    monkeypatch.setattr(nv.time, "monotonic", lambda: NOW)

    def arrival(snail_x, hold=True):
        nav = _navigator(model)
        st = nav.begin(c.id, x=520.0)
        st.hold_x = hold
        s = Scene(t=NOW, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0,
                  where=Where(300, 500, c, (0, 0)), searched=True, wide=True,
                  monsters=[walker(snail_x, kind="snail", pose="stand", platform=c.id)])
        return nav, st, s
    nav, st, s = arrival(560.0)
    said = []
    nav.log = said.append
    assert nav._arrival_clear(st, s) is False and "not walked on to x 520" in st.trip.short
    assert any("not walked on" in line for line in said)
    nav, st, s = arrival(310.0)
    assert nav._arrival_clear(st, s) is True and st.trip.short == ""
    nav, st, s = arrival(310.0, hold=False)
    assert nav._arrival_clear(st, s) is False
    monkeypatch.setattr(nv, "ARRIVE_STAND", False)
    nav, st, s = arrival(310.0)
    assert nav._arrival_clear(st, s) is False


def test_a_pass_leaves_the_monster_in_the_way_out_only_within_the_window_its_touch_opens(model, monkeypatch):
    # PASS_WINDOW (_within_window): a pass's chain that never meets the monster it left out is kept as it is; one clear
    # of it once the window its first touch opens is over (that touch's plan time, INVULNERABLE_S on less
    # IMMUNE_MARGIN_S) is kept too, said from when; one that stays in its box past that is planned again with the monster
    # counting from then (MonsterTrack.counts_from: flow's danger), its own first touch checked; nothing chains so: the
    # monster left out wholly, as before. (Left out for the whole chain, her retreat after a stop was planned lying on 15
    # for a drop, a Horny Mushroom walking at her, past each window: the review's tour seed 7.)
    from gamelab.motion.flow import Body, Leg, Span
    from gamelab.motion.route import Move
    c = by_y(model, 500, 400)
    nav = _navigator(model)
    w = nv.INVULNERABLE_S - nv.IMMUNE_MARGIN_S

    def stand(x, t0, t1):
        return Leg(Move("walk", c.id, c.id, x, x, ""), Body(t0, x, c), Body(t1, x, c), [Span(c.id, t0, t1, x, x)])
    mo = MonsterTrack(c.id, 300.0, speed=0.0, half=20.0, vx=0.0, kind="snail")
    assert nav.flow.danger(stand(300.0, 0.0, 3.0), [replace(mo, counts_from=1.0)])[1] >= 1.0  # (flow: from then on)
    assert nav._within_window(None, None, [], [stand(600.0, 0.0, 3.0)], [], mo, NOW, []) is None  # (never meets it)
    short = [stand(300.0, 0.0, 0.2), stand(600.0, 0.2, 1.0)]  # (in its box at first, then away: within its window)
    assert nav._within_window(None, None, [], short, [], mo, NOW, []) == (None, pytest.approx(w))
    planned = []

    def chain(st, s, path, monsters, looked, priced=None):
        planned.append(monsters)
        return [stand(300.0, 0.0, 0.5)], "keys", "done"
    nav._chain = chain
    long = [stand(300.0, 0.0, 3.0)]  # (in its box 3 s: past the window)
    got, until = nav._within_window(None, None, [], long, [], mo, NOW, [])
    assert got[0] == [stand(300.0, 0.0, 0.5)] and until == pytest.approx(w)
    assert planned[0][-1].counts_from == pytest.approx(w) and planned[0][-1].x == mo.x
    nav._chain = lambda *a, **k: ([], None, "monster")
    assert nav._within_window(None, None, [], long, [], mo, NOW, []) is None  # (nothing chains so: as before)


def test_a_blocked_moves_budget_is_kept_over_a_touched_pass_and_the_next_block_goes_past_at_once(model, monkeypatch):
    # BLOCK_MEMORY and PASS_AVOID (off as committed: measured, the walk's touches up or her trips' arrivals down; the
    # switches' comment): the way up blocked by a snail standing by the rope's top on D, her spot safe. Its budget spent
    # (EVADE_S since the move's first block): past it at once, no wait nor way around; the pass touched and her knocked
    # back onto C (the move not made): the move dearer (st.avoid), its budget kept, and the next block of it goes past at
    # once too (the review's walk 19: a refuge, 3.5 s waited and a pass into it 4 times in one trip). Once she makes the
    # move its budget goes; BLOCK_FORGET_S without a block of it, and it's forgotten; another move's budget is its own.
    d = by_y(model, 300, 700)
    monkeypatch.setattr(nv, "BLOCK_MEMORY", True)
    monkeypatch.setattr(nv, "PASS_AVOID", True)
    events = _events(monkeypatch)
    nav, st, s, path, monsters, carried = _trip(model, monkeypatch, [], on=[(d.id, 700.0)])
    legs, keys, why = nav._chain(st, s, path, monsters, NOW)
    assert legs == [] and why == "monster" and nav._safe_to_stand(300.0, s.where.platform, path[0], monsters)
    k = nv._key(path[0])
    st.evade_key, st.evade_since, st.blocked_at = k, NOW - nv.EVADE_S - 0.1, NOW - 0.2

    def knocked(st_, s_, path_, legs_, keys_, why_, dst, mons=()):  # (touched on the way, back where she was)
        carried.append((legs_, why_))
        return [leg.move for leg in legs_], d, s_
    nav._carry_out_chain = knocked
    nav._touch_since = lambda since: {"kind": "MISS", "at": NOW}
    said = []
    nav.log = said.append
    assert nav._wait_for_monster(st, s, path, monsters, legs, why, NOW, NOW) is True
    assert len(carried) == 1 and not any("waiting" in line or "way around" in line for line in said)
    assert st.avoid.get(k) == NOW and st.evade_key == k and st.evade_since == NOW - nv.EVADE_S - 0.1
    assert nav._wait_for_monster(st, s, path, monsters, legs, why, NOW, NOW) is True
    assert len(carried) == 2 and not any("waiting" in line for line in said)
    passes = [kw["text"] for t, kw in events if t == "event" and kw.get("kind") == "backstop"]
    assert len(passes) == 2 and all(p.startswith("EVADE_S") for p in passes)
    # (the two passes of the move within PASS_EPISODE_S: one episode, the second its 2nd; the snail on D, above her)
    recs = [kw for t, kw in events if t == "event" and kw.get("kind") == "backstop"]
    assert [(r["episode"], r["nth"]) for r in recs] == [(1, 1), (1, 2)] and recs[0]["blocker"]["rel"] == "above"
    nav._made(st, [path[0]], True)  # (made: its budget gone)
    assert st.evade_key is None and st.evade_since is None and k not in st.blocks
    other = replace(path[0], x=path[0].x + 1.0)  # (another move: its own budget, this one's kept by)
    st.evade_key, st.evade_since, st.refuges, st.blocked_at = k, NOW - 1.0, 2, NOW
    nav._evade_for(st, other)
    assert (st.evade_since, st.refuges) == (None, 0) and st.blocks[k][:2] == (NOW - 1.0, 2)
    nav._evade_for(st, path[0])
    assert (st.evade_since, st.refuges) == (NOW - 1.0, 2)
    monkeypatch.setattr(nv.time, "monotonic", lambda: NOW + nv.BLOCK_FORGET_S + 1.0)
    nav._evade_for(st, path[0])
    assert (st.evade_since, st.refuges) == (None, 0)  # (forgotten)


def test_with_toucher_all_every_trip_keeps_clear_of_the_monster_that_touched_her(model, monkeypatch):
    # TOUCHER_ALL (off: measured worse in the walk): a plain trip (no `toucher`: not a retreat) keeps clear of R7's
    # stand-in at her x after a touch where the look has none by her, standing there (TOUCHER_STAND); off (as
    # committed), her retreat's alone.
    m = mapdata.parse(1, tree(), "Test Map", {}, {"2": "Horny Mushroom"}, mob_damage={"Horny Mushroom": 110.0})
    c, d = by_y(m, 500, 400), by_y(m, 300, 700)
    nav = _navigator(m)
    monkeypatch.setattr(nv.time, "monotonic", lambda: NOW)
    monkeypatch.setattr(nv, "TOUCHER_ALL", True)
    nav.senses = SimpleNamespace(touch={"at": NOW - 0.5, "told": NOW - 0.45, "kind": "MISS"},
                                 invulnerable_until=lambda: NOW + 0.95)
    s = Scene(t=NOW, frame=np.zeros((1, 1, 3), np.uint8), tag=(0, 0), hp=1.0, mp=1.0,
              where=Where(300, 500, c, (0, 0)), searched=True, wide=True, monsters=[])
    path = Planner(m).plan(c.id, 300, d.id)
    monsters, _ = nav._chain_monsters(nav.begin(d.id), s, path)
    assert any(t.x == 300 and t.kind == "horny-mushroom" and t.vx == 0.0 for t in monsters)  # (standing: TOUCHER_STAND)
    monkeypatch.setattr(nv, "TOUCHER_ALL", False)
    monsters, _ = nav._chain_monsters(nav.begin(d.id), s, path)
    assert not any(t.x == 300 for t in monsters)


def test_a_safe_platform_a_monsters_platform_lies_over_within_her_height_is_weighed_by_its_places():
    # retreat.wholly_safe (OVERHEAD): P (0..400 at y 500) has no spawn and none under it (MapModel.safe), but Q (200..400
    # at y 450, a monster's) lies 50 over its right half, within her height: not wholly safe (Tree Dungeon's 25 under
    # 24). Off, as MapModel.safe.
    from test_navigation import seg

    from gamelab.behavior import retreat as rv
    data = {"info": {"VRLeft": -100, "VRTop": 0, "VRRight": 600, "VRBottom": 700},
            "foothold": {"1": {"1": dict([seg(1, 1, 1, 0, 500, 400, 500)]),
                               "2": dict([seg(1, 2, 2, 200, 450, 400, 450)]),
                               "3": dict([seg(1, 3, 3, 0, 300, 100, 300)])}},
            "ladderRope": {}, "portal": {},
            "life": {"0": {"type": "m", "id": "0000002", "x": 300, "y": 450, "cy": 450, "rx0": 200, "rx1": 400}},
            "miniMap": {"width": 700, "height": 700, "centerX": 100, "centerY": 0, "mag": 4}}
    m = mapdata.parse(2, data, "Over", {})
    p, top = by_y(m, 500, 100), by_y(m, 300, 50)
    assert m.safe(p.id) and not rv.wholly_safe(m, p.id) and rv.wholly_safe(m, top.id)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rv, "OVERHEAD", False)
        assert rv.wholly_safe(m, p.id)


def test_a_drop_with_no_spot_clear_of_a_ropes_top_goes_from_the_moves_own_spot(monkeypatch):
    # Mover._drop (DROP_AT_SPOT): Tree Dungeon's drop from 18 onto 19 at x 128, no spot over 19 clear of the rope's top
    # at 111 (route.drop_spots none): from 110, within 20 of it, she walked nowhere and dropped by the rope's top, onto
    # the rope (11 times in a minute: the review's loop seed 18); now to 128 first. Off: as before.
    from gamelab.control import mover as mv
    from gamelab.motion.route import drop_spots
    from gamelab.sim import scenarios
    m = scenarios.real_map(scenarios.TREE)
    if m is None:
        pytest.skip("needs Tree Dungeon's game data (gamedata/data)")
    drop = next(mm for mm in Planner(m).moves if mm.kind == "drop" and (mm.src, mm.dst) == (18, 19))
    assert drop_spots(m, m.platform(18), m.platform(19)) == []
    keys = SimpleNamespace(down=SimpleNamespace(name="down"), jump=SimpleNamespace(name="alt"))
    sent = []
    actions = SimpleNamespace(keys=keys, arrow=lambda d: d, pickups=lambda ms: [])
    mover = mv.Mover(None, actions, m, log=lambda *a: None)
    mover._send = lambda events, name: sent.append(events)
    monkeypatch.setattr(mv.time, "sleep", lambda s: None)
    at = SimpleNamespace(where=Where(110.0, -901, m.platform(18), (0, 0)))
    mover._drop(drop, at)
    assert sent[-1][0][1:] == ["right", "down"]  # (a walk right first)
    downs = [e for e in sent[-1] if e[1] == "down"]  # down let go 80 ms after the jump (as a chain's drop does)
    assert [e[2] for e in downs] == ["down", "up"] and downs[1][0] - downs[0][0] == mv.DROP_DOWN_UP_MS == 200
    monkeypatch.setattr(mv, "DROP_AT_SPOT", False)
    mover._drop(drop, at)
    assert sent[-1][0][1:] == ["down", "down"]  # (as before: down at once)


def test_a_drop_that_ended_on_a_rope_is_another_way_for_the_rest_of_the_trip(model, monkeypatch):
    # ROPE_DROP: a drop onto C that ended on no platform (a rope by its top: the hunt loop's seed 10, 28 onto 30 by the
    # rope at 146, 12 times in 57 s) costs CANNOT_S for the rest of the trip (st.cannot), said, and isn't a chain's other
    # way between the same platforms either (_pricing's alternatives); off, as before.
    from gamelab.motion.route import Move
    c, a = by_y(model, 500, 400), by_y(model, 300, 100)
    nav = _navigator(model)
    st = nav.begin(c.id)
    said = []
    nav.log = said.append
    drop = Move("drop", a.id, c.id, 100.0, 100.0)
    on_rope = SimpleNamespace(where=SimpleNamespace(platform=None))
    nav._rope_drop(st, [drop], c, on_rope)
    assert nv._key(drop) in st.cannot and any("ended on a rope" in line for line in said)
    landed = SimpleNamespace(where=SimpleNamespace(platform=c))
    st2 = nav.begin(c.id)
    nav._rope_drop(st2, [drop], c, landed)
    assert not st2.cannot
    nav._siblings = {(a.id, c.id): [drop, Move("walk_off", a.id, c.id, 200.0, 210.0)]}
    nav.escort = SimpleNamespace(landings=lambda monsters: (lambda leg, last=True: (0.0,)))
    monkeypatch.setattr(nv, "LANDING_CLEAR", True)
    monkeypatch.setattr(Navigator, "fight", property(lambda self: True))
    _, alts, _ = nav._pricing(st, [])
    assert [m.kind for m in alts(drop)] == ["walk_off"]
    monkeypatch.setattr(nv, "ROPE_DROP", False)
    st3 = nav.begin(c.id)
    nav._rope_drop(st3, [drop], c, on_rope)
    assert not st3.cannot
