"""Her touches (perception.touches; README, Touches): the violet popup born over her name tag, MISS or the damage she
took. On crops of the hunting recordings of 2026-10-24..28 (tests/fixtures/touches, each a few frames of a fixed screen
region around her tag: a MISS, a Horny Mushroom's 133, another player's 40 over her with her HP the same, a skill's
flash, her 124 under her own orange numbers), on the game's own sprites drawn (MISS and the digits read at the game's
size and at half of 1920x1080, her orange numbers and the criticals not violet at all); and what the stack makes of one:
the frame step's 'touch' event (an HP drop with no popup one too; a popup soon after a touch the same one), every
state's newest, Senses' invulnerability after it, the hunter's (a MISS knocks nothing), her trips' legs while she can't
be touched (flow.Flow.immune), and the simulator's (her invulnerability, her evasion by the game's hit check, her popups
told as the game shows them, the harness's counts)."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from gamelab.perception import touches
from gamelab.perception.touches import DAMAGE, INVULNERABLE_S, MISS, Touch, TouchWatch

FIXTURES = Path(__file__).parent / "fixtures" / "touches"


@pytest.fixture(scope="module")
def reader():
    r = touches.Reader.of(1.0)  # (the game's own 1366x768: a game px a look-frame px)
    if r is None:
        pytest.skip("needs the game data (gamedata/data): Effect/Effect_000.wz BasicEff.img")
    return r


def _case(name):
    return json.loads((FIXTURES / "index.json").read_text())[name]


def _watch_through(name, reader, after_s=0.85):
    """Case `name`'s frames through a TouchWatch, then one more update `after_s` after the last with no frame (her HP
    the same): the touches told, and the watch."""
    c = _case(name)
    w = TouchWatch(reader)
    told = []
    for f in c["frames"]:
        frame = np.asarray(Image.open(FIXTURES / f["file"]).convert("RGB"))
        told += w.update(frame, tuple(f["tag"]), f["seen"], f["hp"], f["t"])
    last = c["frames"][-1]
    told += w.update(None, None, True, last["hp"], last["t"] + after_s)
    return told, w


def test_her_miss_is_told_in_the_frame_it_shows_with_nothing_else(reader):
    # Her MISS over her tag (an orange 18 of hers on a monster above it): told at once, in its first frame, its HP the
    # same; the frame before shows none.
    told, w = _watch_through("miss", reader)
    assert [(t.kind, t.text, t.t, t.late, t.how) for t in told] == [(MISS, "MISS", 0.0, 0.0, touches.POPUP)]
    assert w.counts == {"touches": 1, "misses": 1, "others": 0, "read": 1}


def test_her_damage_is_told_with_its_number_once_her_hp_bar_moves(reader):
    # A Horny Mushroom's 133: born over her tag, read off the game's sprites, held until her bar moves (the next frame,
    # 0.101 s on: 1.0 -> 0.96), told as of its first frame.
    told, _ = _watch_through("damage", reader)
    assert [(t.kind, t.text, t.t, t.late) for t in told] == [(DAMAGE, "133", 0.0, 0.101)]
    assert told[0].data()["at"] == 0.0 and abs(told[0].dx) <= touches.BIRTH_DX


def test_another_players_number_over_her_and_a_skills_flash_are_not_hers(reader):
    # Another player's 40 born over her (they stood on her), her HP the same for HP_WAIT_S: not hers; a knight's skill
    # flashing violet for a frame, unread: not hers either. Both kept for the record.
    for name in ("other", "flash"):
        told, w = _watch_through(name, reader)
        assert told == [] and len(w.others) == 1 and w.counts["others"] == 1, name
    assert _watch_through("other", reader)[1].others[0].text == "40"


def test_her_number_under_her_own_orange_ones(reader):
    # Her 124 half under her own orange 17 and 140: the line runs 80 px wide and the reader can't read it. With
    # HOLD_UNREAD it waits for her bar, which moves (0.94 -> 0.84): damage; without, by its width alone, a MISS (the
    # measured detector's one kind wrong of 268 so, 2026-10-28).
    told, _ = _watch_through("orange", reader)
    assert len(told) == 1 and told[0].t == 0.0
    assert told[0].kind == (DAMAGE if touches.HOLD_UNREAD else MISS)


def test_without_the_game_data_a_popup_is_told_by_its_width():
    # No reader (no game data copy): a line MISS_W wide or more is MISS at once; a narrower one is a number, hers once
    # her HP bar moves, another's when it doesn't.
    told, _ = _watch_through("miss", None)
    assert [(t.kind, t.late) for t in told] == [(MISS, 0.0)]
    told, _ = _watch_through("damage", None)
    assert [(t.kind, t.text, t.late) for t in told] == [(DAMAGE, "", 0.101)]
    told, w = _watch_through("other", None)
    assert told == [] and len(w.others) == 1


def test_a_popup_seen_again_is_the_same_touch_and_one_after_her_invulnerability_a_new_one(reader):
    # Her MISS's frame again 0.5 s on (the popup found again after her own numbers hid it): the same touch
    # (REFRACT_S); at the same place 1.6 s on (her invulnerability over, another touch): a new one.
    c = _case("miss")
    f = c["frames"][1]
    frame = np.asarray(Image.open(FIXTURES / f["file"]).convert("RGB"))
    w = TouchWatch(reader)
    got = [w.update(frame, tuple(f["tag"]), True, f["hp"], t) for t in (0.0, 0.5, 1.6)]
    assert [[x.kind for x in g] for g in got] == [[MISS], [], [MISS]]


def _sprite_frame(parts, back=(40, 90, 40), scale=1.0):
    """The game's popup sprites (BasicEff.img paths) drawn left to right as the game lays a number out (each glyph over
    the last by about two fifths of it), their bottoms on one row, on `back`; shrunk to `scale` as a look frame shows
    it."""
    from gamedata.canvas import pixels, source
    from gamedata.paths import COPY
    from gamedata.wz import Package
    tree = Package(COPY / "Effect" / "Effect_000.wz").image("BasicEff.img")
    img = np.zeros((200, 300, 3), np.float64)
    img[:] = back
    x = 60
    for part in parts:
        folder, name = part.split("/")
        s = pixels(source(tree[folder][name], tree))
        h, w = s.shape[:2]
        a = s[..., 3:4] / 255.0
        img[110 - h:110, x:x + w] = img[110 - h:110, x:x + w] * (1 - a) + s[..., :3] * a
        x += int(w * 0.62)
    img = img.astype(np.uint8)
    if scale != 1.0:
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return img


@pytest.mark.parametrize("scale", [1.0, 0.703], ids=["1366x768", "half of 1920x1080"])
def test_the_games_own_popups_are_read_and_her_orange_numbers_and_criticals_arent_violet(reader, scale):
    r = reader if scale == 1.0 else touches.Reader.of(scale)
    for back in ((40, 90, 40), (60, 140, 230), (230, 230, 230)):  # (grass, sky, a light wall)
        for parts, said in ((["NoViolet0/Miss"], (MISS, MISS)), (["NoViolet1/5", "NoViolet0/4"], (DAMAGE, "54")),
                            (["NoViolet1/1", "NoViolet0/3", "NoViolet0/3"], (DAMAGE, "133"))):
            frame = _sprite_frame(parts, back, scale)
            (line,) = touches.lines(frame, scale=scale)
            assert r.read(frame, tuple(line[:4]))[:2] == said, (parts, back, scale)
        for parts in (["NoRed1/5", "NoRed0/4"], ["NoCri1/5", "NoCri0/4"], ["NoRed0/Miss"]):
            assert touches.lines(_sprite_frame(parts, back, scale), scale=scale) == [], (parts, back)


# --- the frame step ---------------------------------------------------------------------------------------------

class _Told:
    """A touch watch telling what it's scripted to (`at`: frame time -> Touches), as the simulator's FakeTouches."""

    def __init__(self, at):
        self.at = at

    def update(self, frame, tag, tag_seen, hp, t):
        return list(self.at.get(round(t, 4), ()))


def _frames(step, n, hp_at, start=100.0):
    from test_worldstate import FPS
    out = []
    for k in range(n):
        t = round(start + k / FPS, 4)
        step.eyes.hp = hp_at(k)
        out.append(step.step(lambda ropes, t=t: step.eyes.frame(t, ropes), t))
        step.came_back()
    return out


@pytest.fixture
def step():
    from test_worldstate import Drawn, small_map

    from gamelab.localization.senses import Senses
    from gamelab.localization.worldstate import Schedule, WorldStep
    model = small_map()
    return WorldStep(Senses(Drawn(model), model, band=(100, 100)), schedule=Schedule())


def test_a_popup_of_hers_is_a_touch_event_and_every_state_after_has_it(step):
    # A MISS told in frame 11 (k 10): its 'touch' event in that frame alone, the state's newest touch from then on;
    # damage 1.5 s later as her bar drops: its event beside the hp_drop, and the newest touch that one.
    t0, t1 = 100.0 + 10 / 30, 100.0 + 55 / 30
    step.eyes.touches = _Told({round(t0, 4): [Touch(t0, MISS, MISS)], round(t1, 4): [Touch(t1, DAMAGE, "54")]})
    states = _frames(step, 70, lambda k: 0.9 if k >= 55 else 1.0)
    events = [(s.seq, e.kind, e.data.get("kind")) for s in states for e in s.events]
    assert events == [(11, "touch", MISS), (56, "touch", DAMAGE), (56, "hp_drop", None)]
    assert [s.touch for s in states[:10]] == [None] * 10
    assert all(s.touch["kind"] == MISS and s.touch["told"] == round(t0, 3) for s in states[10:55])
    assert states[-1].touch["kind"] == DAMAGE and states[-1].touch["text"] == "54"
    assert (step.counts["touches"], step.counts["misses"], step.counts["touches by HP"]) == (2, 1, 0)


def test_an_hp_drop_no_popup_was_told_for_is_a_touch_and_a_popup_right_after_it_the_same(step):
    # Her bar drops in frame 21 with no popup of hers told (her tag hidden, a window over her): a touch by her HP bar
    # alone. Its popup found two frames later is that touch (TOUCH_SAME_S), not another.
    t = 100.0 + 22 / 30
    step.eyes.touches = _Told({round(t, 4): [Touch(t, DAMAGE, "54")]})
    states = _frames(step, 40, lambda k: 0.9 if k >= 20 else 1.0)
    touched = [(s.seq, e.data["how"]) for s in states for e in s.events if e.kind == "touch"]
    assert touched == [(21, touches.HP)]
    assert (step.counts["touches"], step.counts["touches by HP"]) == (1, 1)
    # ... and without a watch at all (eyes that read no popups), her HP drops alone are her touches
    step.eyes.touches = None
    more = _frames(step, 60, lambda k: 0.8 if k >= 30 else 0.9, start=110.0)
    assert [e.data["how"] for s in more for e in s.events if e.kind == "touch"] == [touches.HP]


def test_senses_keep_her_newest_touch_and_her_invulnerability_after_it():
    from gamelab.localization.senses import Senses
    s = Senses.__new__(Senses)  # (only its touch bookkeeping)
    s.touch, s.touch_listeners = None, []
    heard = []
    s.touch_listeners.append(heard.append)
    assert s.touched_since(0.0) is None and s.invulnerable_until() < -1e17
    miss = {**Touch(10.0, MISS, MISS).data(), "told": 10.0}
    s._saw_touch(miss)
    s._saw_touch(dict(miss))  # (the same, from a later state)
    assert heard == [miss] and s.touched_since(9.9) == miss and s.touched_since(10.0) is None
    assert s.invulnerable_until() == pytest.approx(10.0 + INVULNERABLE_S)
    hurt = {**Touch(12.0, DAMAGE, "54", late=0.1).data(), "told": 12.1}  # (told with her bar, after its popup)
    s._saw_touch(hurt)
    assert s.touched_since(12.05) == hurt and s.invulnerable_until() == pytest.approx(12.0 + INVULNERABLE_S)


# --- the stack ----------------------------------------------------------------------------------------------------

def test_a_miss_knocks_nothing_and_damage_does_and_both_are_her_last_touch():
    from gamelab.behavior.fight import Report
    from gamelab.behavior.hunter import KNOCK_S, Hunter
    h = Hunter.__new__(Hunter)  # (only what _touched uses)
    h.hit_at = h.hit_kind = h.knock_at = h.touch_at = None
    h.report, h.log, h._last_where = Report(), (lambda text: None), None
    h.actions = SimpleNamespace(pressed=True)
    look = SimpleNamespace(t=10.05, hp=0.9, where=None)
    h._touched(look, Touch(10.0, MISS, MISS).data())
    assert (h.hit_at, h.hit_kind, h.touch_at, h.knock_at) == (10.05, MISS, 10.0, None) and not h.knocked(look)
    h._touched(SimpleNamespace(t=11.6, hp=0.8, where=None), Touch(11.5, DAMAGE, "54").data())
    assert (h.hit_at, h.hit_kind, h.touch_at, h.knock_at) == (11.6, DAMAGE, 11.5, 11.6)
    assert h.knocked(SimpleNamespace(t=11.6 + KNOCK_S - 0.01))
    assert not h.knocked(SimpleNamespace(t=11.6 + KNOCK_S + 0.01))
    assert (h.report.touches, h.report.misses) == (2, 1) and "touches: 2 (1 MISS)" in str(h.report)


def test_with_a_feed_that_tells_no_touches_her_new_hp_drops_are_her_touches():
    # (a perception service started before 2026-10-28's, or with worldstate.TOUCHES off): a new hp_drop no touch came
    # with, within TOUCH_SAME_S either way or of her last, is a touch that took HP seen by her bar alone; one that came
    # with a touch, or a drop of the same hit (not `new`), isn't another.
    from gamelab.behavior.hunter import Hunter
    from gamelab.localization.tracking import Event
    h = Hunter.__new__(Hunter)
    h.touch_at = None
    drop = Event("hp_drop", 20.0, data={"lost": 0.1, "hp": 0.8, "new": True})
    look = SimpleNamespace(t=20.1, events=[drop, Event("hp_drop", 20.1, data={"lost": 0.02, "hp": 0.78, "new": False})])
    assert h._untold(look, []) == [{"kind": DAMAGE, "text": "", "at": 20.0, "how": touches.HP}]
    assert h._untold(look, [Touch(19.9, DAMAGE, "54").data()]) == []
    h.touch_at = 19.5  # (told in the look before)
    assert h._untold(look, []) == []


def test_her_trips_legs_pass_a_monster_while_she_cant_be_touched():
    # test_flow's run along C with a monster standing at x 450: the leg meets it, the chain stops short (`monster`).
    # With her invulnerability lasting past where it meets her (Flow.immune: plan time) the leg passes it and the chain
    # goes on; over before she gets there, the same as ever.
    from test_flow import KEYS
    from test_navigation import by_y, tree

    from gamelab.localization import mapdata
    from gamelab.localization.monsters import MonsterTrack
    from gamelab.motion.flow import Body, Flow, Keys
    from gamelab.motion.route import Planner
    model = mapdata.parse(1, tree(), "Test Map", {})
    c, d = by_y(model, 500, 400), by_y(model, 300, 700)
    path = Planner(model).plan(c.id, 200, d.id)
    flow = Flow(model, KEYS)
    mo = [MonsterTrack(c.id, 450, speed=0.0)]
    leg = flow.leg(path[0], Body(0.0, 200, c), Keys(KEYS), "run")
    (_, meets) = flow.danger(leg, mo)
    assert flow.chain(path, Body(0.0, 200, c), mo, hops=False)[2] == "monster"
    flow.immune = leg.end.t + 1.0
    assert flow.danger(leg, mo) is None and flow.chain(path, Body(0.0, 200, c), mo, hops=False)[2] == "done"
    flow.immune = meets / 2
    assert flow.danger(leg, mo)[1] == meets


# --- the simulator --------------------------------------------------------------------------------------------------

def test_the_worlds_invulnerability_is_the_frames_and_her_evasion_the_games_hit_check():
    from gamelab.sim import world as world_mod
    assert world_mod.INVULNERABLE == world_mod.INVULNERABLE_LIVE == INVULNERABLE_S == 1.45
    green, horny = world_mod.MobInfo(54.0, 79.0), world_mod.MobInfo(110.0, 94.0)  # (Tree Dungeon's: game data)
    her = world_mod.LEVEL_25
    assert her.avoid == world_mod.LEVEL_25_AVOID == 58.0
    # by the game's hit check at level 25 (Green Mushroom level 14, Horny 22): MISS 52%, 26% (the frames: 51%, 28%)
    assert world_mod.evade_chance(green, her, 14.0) == pytest.approx(0.523, abs=0.001)
    assert world_mod.evade_chance(horny, her, 22.0) == pytest.approx(0.261, abs=0.001)
    # at level 28 with the avoid its MISSes fit (70.6): 71% and 44% (the frames: 72%, 42%)
    l28 = dataclasses.replace(her, level=28, avoid=70.6)
    assert world_mod.evade_chance(green, l28, 14.0) == pytest.approx(0.709, abs=0.001)
    assert world_mod.evade_chance(horny, l28, 22.0) == pytest.approx(0.439, abs=0.001)
    # a MISS at most 90.2% (one that fails hits 8% of the time); no avoid, or no level known: EVADE_ACC as before
    assert 1 - world_mod.game_hit_chance(79.0, 14.0, 500.0, 28) == pytest.approx(0.902, abs=0.002)
    assert world_mod.evade_chance(green, her, None) == pytest.approx(27 / 79)
    assert world_mod.evade_chance(green, world_mod.LEVEL_19, 14.0) == pytest.approx(27 / 79)


def test_the_simulators_popups_are_told_as_the_game_shows_them_a_miss_too():
    # The world's touches (World.touch_log: a MISS takes nothing) drawn as popups over her for POPUP_S (Screen.popups),
    # each told once, in the first frame that shows it, MISS or the HP it took as its number.
    from gamelab.sim import world as world_mod
    from gamelab.sim.fakes import FakeEyes, FakeTouches
    w = SimpleNamespace(touch_log=[], t=0.0, character=world_mod.LEVEL_25)
    eyes = SimpleNamespace(world=w)
    watch = FakeTouches()
    touched = [(1.0, 0.0), (2.5, 54 / 774)]  # (a MISS at 1.0 s, 54 HP of her 774 at 2.5)

    def frame_at(t):
        w.t = t
        w.touch_log = [x for x in touched if x[0] <= t]
        return SimpleNamespace(screen=SimpleNamespace(popups=FakeEyes._popups(eyes)))
    told = [(t, [(x.kind, x.text) for x in watch.update(frame_at(t), None, True, 1.0, 100 + t)])
            for t in (1.02, 1.5, 2.1, 2.52, 3.0, 3.6)]
    assert told == [(1.02, [(MISS, MISS)]), (1.5, []), (2.1, []), (2.52, [(DAMAGE, "54")]), (3.0, []), (3.6, [])]
    assert watch.counts == {"touches": 2, "misses": 1}


def test_the_harness_counts_her_misses_and_the_touches_the_stack_was_told_of():
    # outcomes: her touches, those that hurt and her MISSes (the world's), the touches her stack was told of (its
    # 'touch' events) and the MISSes among them; navigation: of the plans cut for a hit, those for a touch, and for a
    # MISS.
    from test_sim import KEYS, plat, tree

    from gamelab.localization import mapdata
    from gamelab.sim import harness
    from gamelab.sim.world import World
    model = mapdata.parse(1, tree(), "Test Map", {}, {"2": "Snail"}, {"2": 2})
    c = plat(model, 500, 400)
    w = World(model, KEYS, (c.id, 385.0), monsters=[("Snail", c.id, 360.0)])
    w.monsters[0].speed = 0.0
    w.advance(6.0)
    events = [{"topic": "event", "kind": "touch", "touch": MISS}, {"topic": "event", "kind": "touch", "touch": DAMAGE},
              {"topic": "event", "kind": "potion"}]
    out = harness.outcomes(w, events, 60.0)
    assert out["touches"] == w.touches >= 4 and out["misses"] == w.evaded and out["hurt"] == w.touches - w.evaded
    assert (out["touches_told"], out["misses_told"]) == (2, 1)
    nav = harness.navigation(["    touched (MISS: HP 90% -> 90%): plan cut short; on platform 3 ...",
                              "    touched (damage 54: HP 90% -> 83%): plan cut short; on platform 3 ...",
                              "    a monster coming into its way (...): plan cut short; ..."], None)
    assert (nav["cut_hit"], nav["cut_touch"], nav["cut_miss"]) == (3, 2, 1)


def test_a_miss_on_a_trip_cuts_it_as_a_hit_does_and_she_replans_and_climbs_on(monkeypatch):
    # nav_tree_rope_hit's trip (from 19 up to 16 by the rope at x 111, a Green Mushroom walking 18 by its top) with
    # every touch a MISS: the MISS cuts the chain on the rope (nothing else would: her HP stays as it was), up goes down
    # again (climb_on), and she gets to 16; the hunter has it as her last touch, no knock.
    from gamelab.sim import scenarios
    from gamelab.sim import world as world_mod
    if not scenarios.available(scenarios.TREE):
        pytest.skip("Tree Dungeon's game data isn't on this computer")
    monkeypatch.setattr(world_mod, "evade_chance", lambda info, her=None, mob_level=None: 1.0)
    seen = {}

    def fn(sim):
        h = sim.hunter()
        trip = sim.navigator(h, fight=True).go(16)
        seen.update(kind=h.hit_kind, knock=h.knock_at, touches=h.report.touches)
        return trip
    sc = scenarios.SCENARIOS["nav_tree_rope_hit"]
    trace = scenarios.run_scenario(dataclasses.replace(sc, fn=fn, expects=()))
    log = "\n".join(trace["log"])
    assert trace["world"]["touches"] >= 1 and trace["world"]["evaded"] == trace["world"]["touches"]
    assert "touched (MISS" in log and "plan cut short" in log and "touched on the rope" in log
    assert trace["world"]["platform"] == 16
    assert seen["kind"] == MISS and seen["knock"] is None and seen["touches"] >= 1
