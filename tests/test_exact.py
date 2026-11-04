"""The exact finder (perception.exact, exact.c): a monster's picture found by its exact pixels however much of it is covered;
what the frame step does with its finds (localization.worldstate) and the tracker with its misses (Track.blind). Pictures
made up here: no game data, no frames saved."""

from types import SimpleNamespace

import numpy as np
import pytest

from gamelab import native
from gamelab.localization.monsters import Monster
from gamelab.localization.tracking import HIDDEN, LIVE, MonsterTracker, Track, View
from gamelab.localization.worldstate import WorldStep
from gamelab.perception import deaths, exact, hud
from gamelab.perception.exact import MIN_EXACT, ExactFinder, Find, Picture
from gamelab.perception.mobs import Mob

needs_c = pytest.mark.skipif(native.exact_lib() is None, reason="no exact finder's DLL here")
RNG = np.random.default_rng(7)


def _picture(kind="horny-mushroom", w=56, h=60, die=False, facing=1, seed=1) -> Picture:
    r = np.random.default_rng(seed)
    rgb = r.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = np.ones((h, w), bool)
    mask[:6, :6] = False  # (a corner not drawn)
    return Picture(kind, f"{kind}_{'die' if die else 'move'}{seed}", facing, "die1/1" if die else "move/1",
                   "die1" if die else "move", die, rgb, mask, w / 2.0, float(h))


def _frame(h=768, w=1366) -> np.ndarray:
    return RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _draw(frame: np.ndarray, p: Picture, x0: int, y0: int) -> None:
    window = frame[y0:y0 + p.h, x0:x0 + p.w]
    window[p.mask] = p.rgb[p.mask]


@needs_c
def test_a_picture_is_found_where_it_lies_however_much_of_it_is_covered():
    p, other = _picture(seed=1), _picture("green-mushroom", seed=2)
    f = ExactFinder([p, other])
    frame = _frame()
    _draw(frame, p, 400, 300)
    got = f.find(frame, (0, 700))
    assert [(g.picture.name, g.x0, g.y0) for g in got] == [(p.name, 400, 300)]
    assert got[0].share == 1.0 and got[0].foot_x == 400 + p.ox and got[0].feet_y == 300 + p.oy
    # Covered by her, her pet, her plate: 90% of it drawn over
    frame[300:354, 400:456] = RNG.integers(0, 256, (54, 56, 3), dtype=np.uint8)
    got = f.find(frame, (0, 700))
    assert [g.picture.name for g in got] == [p.name] and got[0].exact >= MIN_EXACT and got[0].share < 0.2


@needs_c
def test_nothing_on_a_frame_not_of_the_games_pixels_and_little_of_a_picture_isnt_a_find():
    p = _picture()
    f = ExactFinder([p])
    frame = _frame()
    _draw(frame, p, 400, 300)
    # (JPEG, a shade: not one pixel its own)
    shaded = (frame.astype(np.int16) * 3 // 4).astype(np.uint8)
    assert f.find(shaded, (0, 700)) == []
    # Under MIN_EXACT showing
    clean = _frame()
    window = clean[300:360, 400:456]
    window[:3] = p.rgb[:3]
    assert f.find(clean, (0, 700)) == []


@needs_c
def test_its_frames_share_pixels_each_pixel_is_one_finds():
    """A picture and the next of its animation share most of their pixels: the one with more of them exact takes them."""
    p = _picture(seed=3)
    rgb = p.rgb.copy()
    rgb[40:] = np.random.default_rng(9).integers(0, 256, rgb[40:].shape, dtype=np.uint8)
    near = Picture(p.kind, "other frame", 1, "move/2", "move", False, rgb, p.mask, p.ox, p.oy)
    f = ExactFinder([near, p])
    frame = _frame()
    _draw(frame, p, 200, 100)
    assert [g.picture.name for g in f.find(frame, (0, 700))] == [p.name]


@needs_c
def test_a_picture_cut_by_the_frames_edge_is_found_and_the_rows_are_kept_to():
    p = _picture()
    f = ExactFinder([p])
    frame = _frame()
    _draw(frame, p, 0, 300)
    frame[300:360, 0:28] = RNG.integers(0, 256, (60, 28, 3), dtype=np.uint8)  # (as if its left half were past the edge)
    assert [g.x0 for g in f.find(frame, (0, 700))] == [0]
    assert f.find(frame, (400, 700)) == []  # (above the rows asked)


def _find(p, x0=100, y0=100, share=0.5):
    return Find(p, x0, y0, 1000, share)


def test_a_find_adds_a_detection_only_where_none_lies_and_never_a_dying_one():
    p, die = _picture(), _picture(die=True, seed=5)
    near = Mob(p.kind, 128, 130, 56, 60, 0.9, 100 + p.ox + 5, 100 + p.oy + 5, 1, "move")
    far = Mob(p.kind, 400, 130, 56, 60, 0.9, 400.0, 160.0, 1, "move")
    assert exact.fresh([_find(p)], [near]) == []
    got = exact.fresh([_find(p)], [far])
    # (score 1, whatever share of it showed: no death watch on a monster its exact pixels say is alive: deaths.wants)
    assert len(got) == 1 and got[0].score == 1.0 and got[0].foot_x == 100 + p.ox and got[0].kind == p.kind
    assert not deaths.wants(got[0].score, 0.0)
    assert got[0].pose == "move" and got[0].facing == 1 and got[0].fading is None
    assert exact.fresh([_find(die)], []) == []
    assert len(exact.fresh([_find(p), _find(p, 102, 101)], [])) == 1  # (one body, two of its pictures)


def test_whether_it_can_see_on_the_frames():
    p = _picture()
    m = Mob(p.kind, 128, 130, 56, 60, 0.9, 100.0, 160.0, 1, "move")
    assert exact.confirmed([_find(p)], [m])
    assert not exact.confirmed([], [m], True)  # (the search has them, it none: a shade, a fade, a frame not exact)
    assert exact.confirmed([], [], True) and not exact.confirmed([], [], False)  # (an empty screen: as before)


def test_anchors_keep_to_colours_sparse_on_the_map():
    p = _picture()
    every = exact.anchors_of(p, None)
    assert 0 < len(every) <= 2 * exact.MOST_ANCHORS
    dens = np.zeros(1 << 15)
    dens[:1 << 14] = 1e6  # (half the colour bins all over the map)
    kept = exact.anchors_of(p, dens)
    from gamelab.perception import gates
    assert kept and all(gates.bins(p.rgb[y, x]) >= 1 << 14 for x, y in kept)
    assert exact.density(None) is None
    assert exact.density(np.ones(1 << 15)).sum() == pytest.approx(1366 * 700)


# --- the tracker and the frame step -----------------------------------------------------------------------------------


def _mon(x, kind="horny-mushroom", mob=None):
    return Monster(mob or Mob(kind, int(x), 100, 56, 60, 0.9, x, 130.0, 1, "move"), x, 130.0, None)


def test_the_tracker_counts_the_frames_the_finder_missed_a_track():
    model = SimpleNamespace(mob_hp={}, mob_exp={}, platform=lambda i: None, mob_bodies={}, mob_speeds={}, spawns=[])
    tr = MonsterTracker(model, throws=False)
    view = View(0, 1366, 0, 700, 1366, 700, (0, 0, 0, 0), 24.0)
    tr.update(0.0, [_mon(500.0)], view, (100.0, 130.0))
    assert next(iter(tr.tracks.values())).blind == 0
    tr.update(0.07, [], view, (100.0, 130.0), exact=True)
    tr.update(0.13, [], view, (100.0, 130.0), exact=True)
    tr.update(0.2, [], view, (100.0, 130.0))  # (a frame it can't see on: no count)
    t = next(iter(tr.tracks.values()))
    assert t.blind == 2
    tr.update(0.27, [_mon(505.0)], view, (100.0, 130.0), exact=True)
    assert next(iter(tr.tracks.values())).blind == 0


def _step(tracks, finds):
    step = object.__new__(WorldStep)
    step.tracker = SimpleNamespace(live=lambda: tracks)
    step._finds = finds
    step._deaths_told = []
    step.counts = {"exact deaths": 0}
    return step


def _at(i, x, last, state=HIDDEN, kind="horny-mushroom"):
    mob = Mob(kind, 0, 0, 56, 60, 0.9, x, 130.0, 1, "move")
    return Track(i, kind, 1, x, 130.0, mob, 0.0, last, state=state, seen_x=x)


def test_a_die1_find_is_the_death_of_the_track_not_found_alive_there():
    s = SimpleNamespace(map_x=lambda x: x, map_y=lambda y: y)
    die = _picture(die=True, seed=5, h=30)
    find = Find(die, 472, 100, 800, 0.8)  # (its feet at 500, 130)
    t = 10.0
    near, beside, other = _at(1, 505.0, t - 0.1), _at(2, 480.0, t, LIVE), _at(3, 505.0, t - 0.1, kind="green-mushroom")
    matched = {2: _mon(480.0)}
    step = _step([near, beside, other], [find])
    assert step._died_seen(s, t, matched) == [1]
    # Too far from where it was last seen, for what it can have walked since
    assert _step([_at(1, 600.0, t - 0.1)], [find])._died_seen(s, t, {}) == []
    assert _step([_at(1, 560.0, t - 0.4)], [find])._died_seen(s, t, {}) == [1]
    # The search has it on that very place, the shut eyes of die1/0 as a stand: its track's
    on_it = {4: _mon(500.0)}
    assert _step([_at(4, 500.0, t, LIVE)], [find])._died_seen(s, t, on_it) == [4]
    # ... unless a picture of it alive is found there too: two bodies
    alive = Find(_picture(seed=6), 472, 70, 900, 0.5)
    assert _step([_at(4, 500.0, t, LIVE)], [find, alive])._died_seen(s, t, on_it) == []


def test_one_animation_tells_one_death():
    """A monster's die1 runs about 12 frames, found in each: the first's is its death, the rest are that animation, not the
    nearest of its kind that a frame missed (R2's frame 425: #10 marked dead, walking on as #12). Another's, farther off or
    later, tells its own."""
    s = SimpleNamespace(map_x=lambda x: x, map_y=lambda y: y)
    die = _picture(die=True, seed=5, h=30)
    find = Find(die, 472, 100, 800, 0.8)  # (its feet at 500, 130)
    t = 10.0
    dead, beside = _at(1, 505.0, t - 0.1), _at(2, 520.0, t - 0.1)
    step = _step([dead, beside], [find])
    assert step._died_seen(s, t, {}) == [1]
    dead.died_seen = t
    assert step._died_seen(s, t + 0.067, {}) == []  # (the next frame of it: `beside` is alive)
    assert step._died_seen(s, t + 0.8, {}) == []
    step.tracker = SimpleNamespace(live=lambda: [_at(2, 520.0, t + 1.0)])
    assert step._died_seen(s, t + 1.1, {}) == [2]  # (past DIE_ONE_S: another's)
    step = _step([beside], [find])
    step._deaths_told = [(t, "horny-mushroom", 500.0, 130.0)]  # (the death watch told it: its animation goes on)
    assert step._died_seen(s, t + 0.1, {}) == []
    step = _step([dead, beside], [find, Find(die, 572, 100, 800, 0.8)])  # (and one farther off, the same frame)
    dead.died_seen = None
    step.tracker = SimpleNamespace(live=lambda: [dead, _at(3, 600.0, t - 0.1)])
    assert sorted(step._died_seen(s, t, {})) == [1, 3]


def test_a_bar_two_monsters_could_share_is_left_as_last_read():
    """hud.bar_shared: another detection's bar would lie where this one's is looked for (BAR_SLACK): whose it is can't be
    told. Beside it by more than the slack, or well under it, its own."""
    geo = SimpleNamespace(frame_scale=1.0)
    a = Mob("horny-mushroom", 500, 100, 56, 60, 1.0, 500.0, 130.0, 1, "move")
    near = Mob("horny-mushroom", 515, 100, 56, 60, 1.0, 515.0, 130.0, 1, "move")  # (R2's frame 375: 15 px off, one bar on both)
    far = Mob("horny-mushroom", 520, 100, 56, 60, 1.0, 520.0, 130.0, 1, "move")
    under = Mob("horny-mushroom", 505, 200, 56, 60, 1.0, 505.0, 230.0, 1, "move")
    assert hud.bar_shared(a, [a, near], geo) and hud.bar_shared(near, [a, near], geo)
    assert not hud.bar_shared(a, [a], geo) and not hud.bar_shared(a, [a, far], geo) and not hud.bar_shared(a, [a, under], geo)
    # the reader with the finder on: none read for either; a track's stays as last read
    s = SimpleNamespace(geo=geo)
    step = object.__new__(WorldStep)
    step.finder, step.counts = object(), {"bars": 0}
    step._bar_due = lambda tr, t: True
    step._bar_read = {}
    step.eyes = SimpleNamespace(bar=lambda s, mob: 0.15)
    unread, work = set(), {}
    read = step._reader(s, 1.0, unread, work, [a, near])
    m = Monster(a, 500.0, 130.0, None)
    assert read(m, SimpleNamespace(id=3)) is m and unread == {3} and work["bars"] == 0 and work["bars shared"] == 1
    assert read(m, None).bar is None
    read = step._reader(s, 1.0, unread, work, [a])
    assert read(m, SimpleNamespace(id=4)).bar == 0.15
    # ... and with the search alone as ever (two that near it doesn't find)
    step.finder = None
    assert step._reader(s, 1.0, unread, work, [a, near])(m, SimpleNamespace(id=5)).bar == 0.15


def test_the_step_says_whether_the_finder_is_on():
    step = object.__new__(WorldStep)
    step.finder, step.finder_why = None, "its C: no compiler"
    assert step.finder_line() == "the exact finder off: its C: no compiler"
    step.finder = SimpleNamespace(pictures=[1, 2], anchors=30)
    assert step.finder_line() == "the exact finder on (2 pictures, 30 anchors)"
