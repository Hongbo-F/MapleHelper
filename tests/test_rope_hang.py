"""Hanging under a rope's top until the way up clears (control.navigator: ROPE_HANG; the roadmap's item 9, P08, from the
three recordings of RiceBaby grinding on Tree Dungeon on 2026-11-02): 19-23 of the hour's 170 touches came at the top of
the rope at x -80 from 11 to 10, a Horny Mushroom on 10 beside it, over 68 climbs up it. While she climbs, the way up is
checked from the newest state; where it isn't clear she stops on the rope HANG_UNDER under its top, where no monster's
body reaches her, and climbs the rest once it is, or after HANG_S. On Tree Dungeon's own geometry (its game data), with
the navigator's senses stood in for: the feed's state as tracks, her dot as the localizer places it."""

from types import SimpleNamespace

import pytest

from gamelab.control import navigator as nv
from gamelab.control.navigator import (CLIMB_ON_X, HANG_FLOOR, HANG_S, HANG_UNDER, HUNG, ROPE_SPEED, Navigator)
from gamelab.localization.pose import Localizer
from gamelab.localization.tracking import Track
from gamelab.motion import bodies
from gamelab.motion.physics import CLIMB_SPEED
from gamelab.motion.route import Move, grab_height
from gamelab.localization.rate import HZ
from gamelab.perception.mobs import Mob
from gamelab.screen import NATIVE
from gamelab.sim import scenarios

TREE = 10002073
NOW = 1000.0
HORNY = "horny-mushroom"


@pytest.fixture(scope="module")
def tree_map():
    m = scenarios.real_map(TREE)
    if m is None:
        pytest.skip("needs Tree Dungeon, Forest Up North II's game data (gamedata/data)")
    return m


def _rope(m, x=-80.0):
    return next(r for r in m.ropes if abs(r.x - x) < 1 and r.bottom - r.top > 150)


def _track(tid, x, platform, vx=0.0, kind=HORNY, last_seen=NOW, state="live"):
    return Track(tid, kind, platform, x, 0, Mob(kind, 0, 0, 30, 30, 0.9), last_seen - 2, last_seen, vx=vx,
                 state=state)


class _Clock:
    """time.monotonic for the navigator, moved on by its glances (one frame each)."""

    def __init__(self):
        self.t = NOW

    def __call__(self):
        return self.t


def _navigator(m, monkeypatch, tracks=(), y=None, x=-80.0, touched=None):
    """A navigator on Tree Dungeon, without an escort, its clock stopped at NOW (moved a frame on by each glance), its
    senses a stand-in: the feed's newest state with `tracks`, her dot at map (x, y) on the rope at -80, no touch (or
    `touched()`), her invulnerability over; its key plans recorded (`played`) and its telemetry (`events`)."""
    key = lambda name: SimpleNamespace(name=name)
    keys = SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), up=key("up"), down=key("down"),
                           pick_up=None)
    played = []
    actions = SimpleNamespace(keys=keys, sent_at=0.0, play=lambda events, name: played.append((name, events)))
    nav = Navigator(None, actions, m, log=lambda line: None)
    loc = Localizer(m, geo=NATIVE)
    clock = _Clock()
    monkeypatch.setattr(nv.time, "monotonic", clock)
    state = SimpleNamespace(t=NOW, placed=True, tracks=list(tracks))
    place = {"y": y, "x": x}

    def glance():
        clock.t += 1.0 / HZ
        state.t = clock.t
        nav.senses.glanced_at = clock.t
        for tr in state.tracks:
            tr.last_seen = clock.t
            tr.x += tr.vx / HZ
        return (loc.to_dot(place["x"], place["y"]) if place["y"] is not None else None), 1.0

    nav.senses = SimpleNamespace(localizer=loc, glanced=state, glanced_at=NOW, invulnerable_until=lambda: -1e18,
                                 wait_frame=lambda timeout: None, glance=glance,
                                 touched_since=touched or (lambda since: None))
    events = []
    monkeypatch.setattr(nv.telemetry, "emit", lambda topic, **kw: events.append((topic, kw)))
    nav.played, nav.events, nav.place, nav.state, nav.clock = played, events, place, state, clock
    return nav


def _due(nav, rope, under, age=0.05):
    """_hang_due with her dot `under` under the rope's top, its frame `age` s old."""
    grab = grab_height(nav.model.platform(11).y_at(rope.x), rope)
    dot = nav.senses.localizer.to_dot(rope.x, rope.top + under)
    return nav._hang_due(rope, grab, dot, nav.clock.t - age)


def test_the_hang_is_out_of_reach_of_a_monster_at_the_top_and_the_one_at_its_foot():
    """Her box (bodies: the client's, 65 tall) with her feet HANG_UNDER under the top of the rope at -80 can't overlap
    a Horny Mushroom's (28 either side, 56 tall) standing anywhere on 10 within reach of the rope, nor one on 11 under
    it; at 48 under (the live touch R2 #24) it does."""
    m = scenarios.real_map(TREE)
    if m is None:
        pytest.skip("needs the game data")
    rope = _rope(m)
    p10, p11 = m.platform(10), m.platform(11)
    for mx in range(int(rope.x) - 49, int(rope.x) + 50, 7):
        top = p10.y_at(mx)
        assert not bodies.touches(rope.x, rope.top + HANG_UNDER - 10, mx, mx, 28.0, top, 56.0), mx
        assert bodies.touches(rope.x, rope.top + 48, mx, mx, 28.0, top, 56.0), mx
        assert not bodies.touches(rope.x, rope.top + HANG_UNDER, mx, mx, 28.0, p11.y_at(mx), 56.0), mx


def test_a_climb_stops_about_hang_under_its_top_for_a_horny_standing_by_it(tree_map, monkeypatch):
    """_hang_due: a Horny standing on 10 20 units right of the rope's top. Low on the rope, the next frame decides; in
    the band, she stops where the cut would leave her, HANG_UNDER to a frame's climb more under the top; past
    HANG_FLOOR it's too late (on up, as before); in the jump at the rope (under where she grabs it), nothing."""
    rope = _rope(tree_map)
    nav = _navigator(tree_map, monkeypatch, [_track(1, rope.x + 20, 10)])
    lag = ROPE_SPEED * (0.05 + nv.HANG_LAG_S)
    assert _due(nav, rope, 150) is None  # (low: the next frame decides)
    under = HANG_UNDER + lag + 4
    got = _due(nav, rope, under)
    assert got is not None and got[0] is rope and "horny-mushroom on 10" in got[2]
    assert HANG_UNDER <= got[1] - rope.top <= HANG_UNDER + ROPE_SPEED / HZ
    assert _due(nav, rope, HANG_FLOOR - 1 + lag) is None  # (too late)
    assert _due(nav, rope, 200) is None  # (in the jump at it: under where she grabs it, 158 under)
    nav.state.tracks = []
    assert _due(nav, rope, under) is None  # (nothing there: on up)


def test_a_horny_walking_away_from_the_top_lets_her_climb_on_and_one_walking_to_it_stops_her(tree_map, monkeypatch):
    """The way up is the climb's rest, the climb-over and STAND_PAST_S at the top against the monster as it walks: one
    150 right walking away doesn't stop her, one 150 right walking at the rope does."""
    rope = _rope(tree_map)
    under = HANG_UNDER + ROPE_SPEED * (0.05 + nv.HANG_LAG_S) + 4
    nav = _navigator(tree_map, monkeypatch, [_track(1, rope.x + 150, 10, vx=60.0)])
    assert _due(nav, rope, under) is None
    nav = _navigator(tree_map, monkeypatch, [_track(1, rope.x + 150, 10, vx=-100.0)])
    assert _due(nav, rope, under) is not None


def test_a_walker_on_9_by_its_end_over_the_rope_s_top_stops_her_too(tree_map, monkeypatch):
    """9's left end (-64) is 40-60 over 10 there, under her height: a Horny on 9 by that end can touch her as she
    climbs over the rope's top at -80 (every platform's monsters are checked, not the top's alone)."""
    rope = _rope(tree_map)
    nav = _navigator(tree_map, monkeypatch, [_track(1, -50.0, 9)])
    under = HANG_UNDER + ROPE_SPEED * (0.05 + nv.HANG_LAG_S) + 4
    got = _due(nav, rope, under)
    assert got is not None and "on 9" in got[2]


def test_hanging_she_climbs_on_once_the_way_up_is_clear(tree_map, monkeypatch):
    """_hang: a Horny standing by the top walks off right after a second or so: she hangs, then up the rest, held
    from her height at flow.ROPE_SPEED and UP_EXTRA; 'rope_hang' events at the start and the end ('clear')."""
    rope = _rope(tree_map)
    horny = _track(1, rope.x + 10, 10)
    nav = _navigator(tree_map, monkeypatch, [horny], y=rope.top + 85)
    glance = nav.senses.glance

    def walking_off():
        if nav.clock.t - NOW > 1.0:
            horny.vx = 100.0
        return glance()
    nav.senses.glance = walking_off
    text = nav._hang(rope, rope.top + 85, "a horny-mushroom on 10 by its top")
    held = nav.clock.t - NOW
    assert text.startswith("the way up clear after") and 1.0 < held < HANG_S
    assert nav.played == [("hang_up", [[0, "up", "down"], [nv.ms(85 / ROPE_SPEED + nv.UP_EXTRA), "up", "up"]])]
    steps = [kw["step"] for topic, kw in nav.events if kw.get("kind") == "rope_hang"]
    assert steps == ["hang", "clear"]
    assert not any(kw.get("kind") == "backstop" for _, kw in nav.events)


def test_hanging_past_hang_s_she_goes_up_regardless_a_backstop(tree_map, monkeypatch):
    """A Horny standing at the top the whole time: after HANG_S up regardless, a backstop event saying so."""
    rope = _rope(tree_map)
    nav = _navigator(tree_map, monkeypatch, [_track(1, rope.x, 10)], y=rope.top + 85)
    text = nav._hang(rope, rope.top + 85, "a horny-mushroom on 10 by its top")
    assert text.startswith("up regardless after") and HANG_S <= nav.clock.t - NOW < HANG_S + 0.2
    assert [kw["text"].split(":")[0] for _, kw in nav.events if kw.get("kind") == "backstop"] == ["HANG_S"]
    assert nav.played[-1][0] == "hang_up"


def test_a_touch_while_she_hangs_ends_the_hang(tree_map, monkeypatch):
    """A touch told while she hangs (something reached her there after all): up at once, her invulnerability her way
    past it."""
    rope = _rope(tree_map)
    nav = _navigator(tree_map, monkeypatch, [_track(1, rope.x, 10)], y=rope.top + 85,
                     touched=lambda since: {"kind": "damage"})
    assert nav._hang(rope, rope.top + 85, "a horny").startswith("touched after")
    assert nav.played[-1][0] == "hang_up"


def test_touched_on_the_rope_with_the_top_taken_she_goes_down_to_hang_under_it_not_down_the_rope(tree_map,
                                                                                                    monkeypatch):
    """_climb_on (CLIMB_LOOK) touched 30 under the top with a Horny there: down to HANG_UNDER (50 at the climb's own
    speed down, physics.CLIMB_SPEED) and the hang, where it went down the whole rope; ROPE_HANG off, as before."""
    rope = _rope(tree_map)
    nav = _navigator(tree_map, monkeypatch, [_track(1, rope.x + 5, 10)], y=rope.top + HANG_UNDER)
    nav._climb_on(rope.top, rope.top + 30, rope.x)
    names = [name for name, _ in nav.played]
    assert names == ["hang_down", "hang_up"]
    assert nav.played[0][1] == [[0, "down", "down"], [nv.ms((HANG_UNDER - 30) / CLIMB_SPEED), "down", "up"]]
    monkeypatch.setattr(nv, "ROPE_HANG", False)
    nav = _navigator(tree_map, monkeypatch, [_track(1, rope.x + 5, 10)], y=rope.top + 30)
    nav._climb_on(rope.top, rope.top + 30, rope.x)
    assert [name for name, _ in nav.played] == ["climb_down"]


def test_the_mover_s_lone_climb_is_watched_for_the_top_too(tree_map, monkeypatch):
    """_move_watch for a climb move (the mover's: about 13 of the hour's climbs up the rope at -80): her dot in the
    band with a Horny by the top cuts it (HUNG), `_hang_at` saying where she hangs; a walk move isn't checked so."""
    rope = _rope(tree_map)
    under = HANG_UNDER + ROPE_SPEED * (1.0 / HZ + nv.HANG_LAG_S) - 2
    nav = _navigator(tree_map, monkeypatch, [_track(1, rope.x + 20, 10)], y=rope.top + under)
    s = SimpleNamespace(t=NOW - 1.0, hp=1.0)
    climb = Move("climb", 11, 10, rope.x, rope.x, rope_x=rope.x, rope_top=rope.top, rope_bottom=rope.bottom)
    watch = nav._move_watch(s, climb)
    assert watch() == HUNG and nav._hang_at[0] is rope
    assert HANG_FLOOR <= nav._hang_at[1] - rope.top <= HANG_UNDER + ROPE_SPEED / HZ
    walk = Move("jump", 11, 13, -100.0, -150.0, "left")
    watch = nav._move_watch(s, walk)
    assert watch() is None and nav._hang_at is None
    monkeypatch.setattr(nv, "ROPE_HANG", False)
    assert nav._move_watch(s, climb)() is None


def test_her_dot_off_the_rope_is_no_hang(tree_map, monkeypatch):
    """Her dot more than CLIMB_ON_X off the rope's x (knocked off, or not on it) decides nothing."""
    rope = _rope(tree_map)
    nav = _navigator(tree_map, monkeypatch, [_track(1, rope.x + 20, 10)])
    grab = grab_height(tree_map.platform(11).y_at(rope.x), rope)
    under = HANG_UNDER + ROPE_SPEED * (0.05 + nv.HANG_LAG_S) + 4
    dot = nav.senses.localizer.to_dot(rope.x + CLIMB_ON_X + 15, rope.top + under)
    assert nav._hang_due(rope, grab, dot, NOW - 0.05) is None
