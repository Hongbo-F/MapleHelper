"""Which way she faces, seen on a frame (perception.heading): her picture above her name tag against her pictures and
their mirror images, on frames whose facing is known (tests/fixtures/heading: 20 frames at 1366x768 in her outfit of
2026-10-24, and 23 of the old recordings' at 1920x1080 half size, labelled by the keys sent); her pictures kept and
read back safely; her keys going by it (control.actions.Actions.see, faces); and the simulator, whose eyes see the
world's facing (or the wrong way: sim.world.Noise.heading_wrong) and whose world can turn her without a key
(Noise.turned, Realism.knock_turns)."""

import dataclasses
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from maplehelper.engine import LEFT, RIGHT

from gamelab import paths, telemetry
from gamelab.control import actions as actions_mod
from gamelab.control.actions import HEADING_SETTLE_S, LEARN_AFTER_S, Actions
from gamelab.perception import heading
from gamelab.perception.heading import Heading, Reading
from gamelab.perception.nametag import NameTag
from gamelab.screen import NATIVE, STRETCHED
from gamelab.sim import scenarios
from gamelab.sim.world import LEVEL_25, Noise, Realism

FIXTURES = Path(__file__).parent / "fixtures"
LABELS = json.loads((FIXTURES / "heading" / "labels.json").read_text(encoding="utf-8"))
GEOS = {"native": NATIVE, "old": STRETCHED}
TREE = 10002073


def crops(kind: str) -> list[tuple[np.ndarray, str, dict]]:
    """The fixture's crops of her (`kind`: "native" or "old"): each with the way she faces and its labels."""
    meta = LABELS[kind]
    w, h = meta["size"]
    strip = np.asarray(Image.open(FIXTURES / "heading" / f"{kind}.png").convert("RGB"))
    return [(strip[:h, i * w:(i + 1) * w], f["side"], f) for i, f in enumerate(meta["frames"])]


def placed(crop: np.ndarray, kind: str, mirror: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
    """A look frame of the crop's geometry, blank but for the crop, her name tag's centre in the middle of the play
    area (clear of the minimap); mirrored if asked (she then faces the other way), and where her tag is."""
    geo, (tx, ty) = GEOS[kind], LABELS[kind]["tag"]
    fw, fh = geo.frame_size
    frame = np.zeros((fh, fw, 3), np.uint8)
    x, y = fw // 2, int(fh * 0.6)
    frame[y - ty:y - ty + crop.shape[0], x - tx:x - tx + crop.shape[1]] = crop
    if mirror:
        return np.ascontiguousarray(frame[:, ::-1]), (fw - 1 - x, y)
    return frame, (x, y)


def bank_of(items, kind: str, path=None) -> Heading:
    """Pictures learnt from these crops (each as facing its way)."""
    bank = Heading(GEOS[kind], path, background=False)
    for crop, side, _ in items:
        assert bank.add(*placed(crop, kind), side)
    bank.rebuild()
    return bank


def other(side: str) -> str:
    return LEFT if side == RIGHT else RIGHT


def scene(name: str) -> str:
    """Which of the 2026-10-24 study's scenes a native frame is of: frames of one scene are near alike (her on the
    same spot, in the same pose), so one is never read with pictures of another of the same scene."""
    for prefix, place in (("n28", "West Rocky II"), ("first", "Tree Dungeon"), ("here", "Tree Dungeon"),
                          ("hp0_1353", "hit on her chair"), ("hp0_1648", "hit"), ("nm_", "by the portal")):
        if name.startswith(prefix):
            return place
    raise AssertionError(name)


# --- perception -----------------------------------------------------------------------------------------------------

def test_her_box_follows_the_windows_size():
    assert heading.box_px(STRETCHED) == (50, 8, 16, 3)
    assert heading.box_px(NATIVE) == (71, 11, 23, 4)


def test_each_scene_at_1366x768_is_read_with_pictures_of_the_other_scenes():
    """Leave one scene out over the 20 frames (5 scenes): none read the wrong way, one not told (hp0_135320, hit on
    her chair: neither way like her, 0.21), 15 of the 19 told strongly, as measured (README, Her heading). (A mirror
    image would only swap the two ways' scores: it tells nothing more.)"""
    items = crops("native")
    wrong, untold, strong = [], [], 0
    for place in sorted({scene(f["file"]) for _, _, f in items}):
        bank = bank_of([c for c in items if scene(c[2]["file"]) != place], "native")
        for crop, side, f in items:
            if scene(f["file"]) == place:
                r = bank.read(*placed(crop, "native"))
                if r.side is None:
                    untold.append((f["file"], r.why))
                elif r.side != side:
                    wrong.append(f["file"])
                strong += r.strong and r.side == side
    assert not wrong and untold == [("hp0_135320.png", "low")] and strong >= 15, (wrong, untold, strong)


def test_the_old_recordings_frames_are_read_with_pictures_of_other_recordings():
    """At 1920x1080 half size, pictures learnt from half the old recordings read the frames of the other half (and
    their mirror images): none the wrong way, nearly all told."""
    items = crops("old")
    names = sorted({f["recording"] for _, _, f in items})
    for learnt in (names[::2], names[1::2]):
        bank = bank_of([c for c in items if c[2]["recording"] in learnt], "old")
        reads = [(bank.read(*placed(crop, "old", mirror)), side if not mirror else other(side))
                 for crop, side, f in items if f["recording"] not in learnt for mirror in (False, True)]
        assert not [r for r, truth in reads if r.side not in (None, truth)]
        assert sum(r.side is not None for r, _ in reads) >= 0.9 * len(reads)


def test_the_seed_tells_her_facing_on_the_saved_frames_of_a_scene_it_wasnt_made_from():
    """The seed in gamelab/assets (16 frames at 1366x768, all the fixture's scenes but West Rocky Mountain II's) on
    the two full frames of tests/fixtures/native, which are of that scene, her tag found as a look finds it: she faces
    left, and right in the frames mirrored (her tag where the mirror puts it: the game doesn't mirror her name)."""
    seed = paths.calibrated(paths.HEADING_SEED, NATIVE.suffix)
    bank = Heading(NATIVE, None, seed)
    assert bank.ready and len(bank.refs) <= heading.REFERENCES
    made_of = {crop.tobytes() for crop in bank.pool}
    n28 = [heading.canonical(heading.cut(*placed(c, "native"), NATIVE), side) for c, side, f in crops("native")
           if scene(f["file"]) == "West Rocky II"]
    assert len(bank.pool) == 16 and n28 and not made_of & {c.tobytes() for c in n28}  # (not made of that scene)
    template = FIXTURES / "native" / "tag_template_1366x768.npy"
    for name in ("n28_000.png", "n28_008.png"):
        full = np.asarray(Image.open(FIXTURES / "native" / name).convert("RGB"))
        x, y = NameTag(np.load(template), template, NATIVE).find(full)
        mirrored = np.ascontiguousarray(full[:, ::-1])
        for frame, tag, truth in ((full, (x, y), LEFT), (mirrored, (full.shape[1] - 1 - x, y), RIGHT)):
            r = bank.read(frame, tag)
            assert r.side == truth and r.strong, (name, truth, r)


def test_what_cant_be_told():
    items = crops("native")
    assert Heading(NATIVE).read(*placed(items[0][0], "native")) == heading.UNKNOWN  # no pictures yet
    bank = bank_of(items, "native")
    frame, (x, y) = placed(items[0][0], "native")
    assert bank.read(frame, None).why == "no tag"
    assert bank.read(frame, (5, y)).why == "off the frame"
    assert bank.read(frame, (100, 250)).why == "minimap"  # her box under the minimap (0..213, 68..213 at 1366x768)
    blank = np.zeros_like(frame)
    assert bank.read(blank, (x, y)).side is None  # nobody there: both ways alike


def test_a_decisive_picture_is_the_only_one_matched(monkeypatch):
    """The picture that matched best last is tried first; one leading the other way by DECISIVE decides alone."""
    items = crops("native")
    bank = bank_of(items, "native")
    assert len(bank.refs) > 1
    bank.read(*placed(items[0][0], "native"))  # (its best picture now tried first)
    calls = []
    real = heading.cv2.matchTemplate
    monkeypatch.setattr(heading.cv2, "matchTemplate", lambda *a, **k: calls.append(1) or real(*a, **k))
    r = bank.read(*placed(items[0][0], "native"))
    assert r.margin >= heading.DECISIVE and len(calls) == 2  # as it is and mirrored, one picture


def test_her_pictures_learn_from_looks_whose_keys_leave_no_doubt_and_are_kept(tmp_path):
    items = crops("native")
    path = tmp_path / "heading_1366x768.npz"
    bank = Heading(NATIVE, path, background=False)
    t = 100.0
    for crop, side, _ in items[:heading.FIRST - 1]:
        assert bank.learn(*placed(crop, "native"), side, t)
        t += heading.LEARN_EVERY_S
    assert not bank.ready
    crop, side, _ = items[heading.FIRST - 1]
    assert not bank.learn(*placed(crop, "native"), side, t - heading.LEARN_EVERY_S / 2)  # one a second at most
    assert bank.learn(*placed(crop, "native"), side, t)
    assert bank.ready and path.exists() and len(bank.pool) == heading.FIRST
    again = Heading(NATIVE, path)
    assert len(again.refs) == len(bank.refs) and len(again.pool) == heading.FIRST
    assert not Heading(STRETCHED, path).ready  # pictures of another window size aren't hers there


def test_a_black_box_isnt_learnt():
    """A look whose frame had left the perception service's ring comes out black (ServiceClient.frame_of): its box
    isn't her, and isn't learnt, whatever the keys say."""
    items = crops("native")
    bank = Heading(NATIVE, None, background=False)
    frame, tag = placed(items[0][0], "native")
    assert not bank.learn(np.zeros_like(frame), tag, items[0][1], 100.0) and not bank.pool
    assert bank.learn(frame, tag, items[0][1], 100.0) and len(bank.pool) == 1


def test_a_file_of_her_pictures_cut_off_or_of_another_box_is_set_aside_for_the_seed_and_written_anew(tmp_path):
    """Her pictures in lab that can't be read (a file cut off) or are of another box (her box's size changed since)
    don't stop a look: the seed stands in, it's said, and the next save writes the file whole again."""
    items = crops("native")
    seed = tmp_path / "seed.npz"
    bank_of(items[:8], "native", seed)
    path = tmp_path / "heading_1366x768.npz"
    path.write_bytes(seed.read_bytes()[:len(seed.read_bytes()) // 2])  # cut off
    said = []
    bank = Heading(NATIVE, path, seed, background=False, log=said.append)
    assert bank.ready and len(bank.pool) == 8 and "can't be read" in said[0]
    assert bank.read(*placed(items[0][0], "native")).side == items[0][1]
    bank.save()
    assert len(Heading(NATIVE, path).pool) == 8 and not list(tmp_path.glob("*.part"))  # whole again, in its place
    np.savez_compressed(path, pool=np.zeros((3, 10, 10, 3), np.uint8), refs=np.zeros((0, 0, 0, 3), np.float32),
                        client=np.array(NATIVE.client))  # (no box kept, as before it was: its boxes say)
    said.clear()
    assert len(Heading(NATIVE, path, seed, log=said.append).pool) == 8 and "another box" in said[0]


def test_a_save_that_fails_leaves_the_last_whole_file(tmp_path, monkeypatch):
    items = crops("native")
    path = tmp_path / "h.npz"
    bank = bank_of(items[:8], "native", path)
    before = path.read_bytes()
    bank.add(*placed(items[9][0], "native"), items[9][1])

    def refused(*a):
        raise PermissionError("held by a reader")
    monkeypatch.setattr(heading.os, "replace", refused)
    bank.save()
    assert path.read_bytes() == before and not list(tmp_path.glob("*.part"))


def test_pictures_made_from_a_pool_dropped_meanwhile_are_dropped_too(tmp_path, monkeypatch):
    """forget while a thread makes pictures from the pool as it was: those pictures aren't used or kept."""
    items = crops("native")
    path = tmp_path / "h.npz"
    bank = Heading(NATIVE, path)
    for crop, side, _ in items[:12]:
        bank.add(*placed(crop, "native"), side)
    started, go = threading.Event(), threading.Event()
    real = heading.references

    def slow(pool, k=heading.REFERENCES):
        started.set()
        go.wait(5)
        return real(pool, k)
    monkeypatch.setattr(heading, "references", slow)
    bank.rebuild()
    assert started.wait(5)
    bank.forget()
    go.set()
    bank.wait()
    assert not bank.ready and not bank.pool
    assert len(Heading(NATIVE, path).pool) == 0  # the old pool isn't back on disk


def test_pictures_that_keep_reading_the_other_way_from_her_keys_are_dropped(tmp_path):
    """A look the pictures read the other way from keys that can't be wrong isn't learnt, and counts against them:
    OUTFIT[0] of the last OUTFIT[1] such looks, and they're no longer hers (a new outfit): dropped."""
    items = crops("native")
    bank = bank_of(items, "native", tmp_path / "h.npz")
    t = 0.0
    for i in range(heading.OUTFIT[0]):
        crop, side, _ = items[i % len(items)]
        frame, tag = placed(crop, "native")
        r = bank.read(frame, tag)
        assert r.side == side
        assert not bank.learn(frame, tag, other(side), t, r)
        t += heading.LEARN_EVERY_S
    assert bank.counts["forgot"] == 1 and not bank.ready and not bank.pool


# --- her keys going by it (control.actions) -----------------------------------------------------------------------

@pytest.fixture
def keys_at(monkeypatch):
    """Actions on a clock of the test's own (now[0]), its plans kept (`sent`), each through hold_back as play does."""
    now = [1000.0]
    monkeypatch.setattr(actions_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(actions_mod.time, "sleep", lambda s: now.__setitem__(0, now[0] + s))
    key = lambda name: SimpleNamespace(name=name)
    a = Actions(SimpleNamespace(left=key("left"), right=key("right"), jump=key("alt"), attack=key("f"),
                                lucky_seven=None, pick_up=key("z"), hp_potion=None, chair=None))
    a.sent = []

    def play(events, name):
        a.hold_back(events)
        a.sent.append((name, events))
    a.play = play
    return a, now


def taps(a: Actions) -> bool:
    """Whether the last throw tapped an arrow first."""
    return any(e[1] in ("left", "right") for e in a.sent[-1][1])


def test_after_a_move_a_strong_frame_says_which_way_she_faces_and_the_throw_doesnt_tap(keys_at):
    a, now = keys_at
    a.play([[0, "left", "down"], [300, "left", "up"]], "walk")  # a mover's move: the keys leave it unknown
    a.facing = None
    assert a.see(LEFT, now[0] + 0.1, strong=True) is None and a.heading is None and a.saw == "too soon after an arrow"
    assert a.see(LEFT, now[0] + 0.3 + HEADING_SETTLE_S, strong=True) is None
    assert a.heading == LEFT and a.facing_by == "frame" and a.saw == "told otherwise"
    assert a.facing is None  # the decisions still go by her keys', as before
    now[0] += 1.0
    a.throw(LEFT)  # (no arrow: the frame still tells, as one shown before the throw would)
    assert not taps(a) and a.heading == a.facing == LEFT and a.facing_by == "frame" and a.turns == 0
    a.lock_until = a.arrow_until = 0.0
    a.throw(RIGHT)  # the other way: the tap, counted
    assert taps(a) and a.heading == a.facing == RIGHT and a.turns == 1 and a.turn_ms == 80


def test_a_frame_that_isnt_strong_adds_a_tap_but_never_takes_one_away(keys_at):
    """Reading.strong or not: after a move a frame that isn't strong tells her heading, but the throw that way still
    taps (as with her keys' facing unknown); with her keys' facing that way it changes nothing; told the other way it
    makes the throw tap, the safe way, whichever it's right about."""
    a, now = keys_at
    a.facing = None
    later = now[0] + 1.0
    assert a.see(LEFT, later) is None and a.heading == LEFT and a.facing_by == "frame" and not a.faces(LEFT)
    now[0] = later + 0.5
    a.throw(LEFT)
    assert taps(a) and a.turns == 1
    a.lock_until = a.arrow_until = 0.0
    a.see(LEFT, now[0] + 1.0)  # agrees with her keys: nothing changes
    assert a.saw == "as it was" and a.facing_by == "keys" and a.faces(LEFT)
    now[0] += 1.5
    a.throw(LEFT)
    assert not taps(a)
    a.lock_until = a.arrow_until = 0.0
    a.see(RIGHT, now[0] + 1.0)  # the other way, not strongly: a tap either way
    assert a.heading == RIGHT and not a.faces(LEFT) and not a.faces(RIGHT)
    now[0] += 1.5
    a.throw(LEFT)
    assert taps(a)


def test_a_frame_shown_before_a_throw_without_a_tap_still_tells(keys_at):
    """A look that doesn't wait for her keys' effect (senses.KEYS_WAIT "moves" or "none") can be of a frame shown
    before a throw's key, right after it. A throw that needed no tap pressed no arrow and didn't turn her, so that
    frame, after her last arrow, still tells her heading: turned without a key, she's tapped back on the next throw.
    (Counted against the throw's keys, with KEYS_WAIT "moves" every such look found her heading by her keys again,
    and every star went the wrong way.)"""
    a, now = keys_at
    a.facing = RIGHT  # her keys' last arrow: right
    now[0] += 2.0
    a.throw(RIGHT)
    assert not taps(a) and a.facing_by == "keys"
    a.lock_until = a.arrow_until = 0.0
    assert a.see(LEFT, now[0] - 0.02, strong=True) == RIGHT  # shown 20 ms before the throw's key: turned since
    assert a.saw == "told otherwise" and a.heading == LEFT and a.facing_by == "frame"
    now[0] += 1.0
    a.throw(RIGHT)
    assert taps(a) and a.heading == a.facing == RIGHT and a.facing_by == "keys" and a.turns == 1
    assert a.see(LEFT, now[0] - 0.02, strong=True) is None and a.saw == "too soon after an arrow"  # before the tap
    assert a.heading == RIGHT


def test_every_throw_is_the_attack_choices_key_with_the_tap_her_heading_asks_for(keys_at):
    """The attack choice (behavior.attacks, through Actions.choose) keys each throw's attack, and her heading (`faces`)
    decides whether its arrow is tapped first: both on every throw, from the ground and jumping; a jump back presses
    its arrows whatever she faces, its key the choice's too."""
    a, now = keys_at
    a.keys.lucky_seven, a.skill = SimpleNamespace(name="q"), True
    chosen, asked = iter(["q", "f", "q", "f", "q"]), []
    a.choose = lambda side: asked.append(side) or next(chosen)

    def attack_key() -> str:
        return next(e[1] for e in a.sent[-1][1] if e[1] in ("f", "q"))

    def unlock(dt: float = 1.0) -> None:
        a.lock_until = a.arrow_until = 0.0
        now[0] += dt

    a.facing = None  # after a move: her keys can't tell
    a.see(LEFT, now[0] + 1.0, strong=True)
    unlock(1.5)
    a.throw(LEFT)  # a strong frame says left: no tap; Lucky Seven, as chosen
    assert not taps(a) and attack_key() == "q"
    unlock()
    a.see(RIGHT, now[0] - 0.5)  # told right, not strongly: the throw left taps first; the regular attack
    a.throw(LEFT)
    assert taps(a) and attack_key() == "f"
    unlock()
    a.jump_throw(LEFT)  # her keys left her facing left: no tap
    assert not taps(a) and attack_key() == "q"
    unlock()
    a.jump_throw(RIGHT)
    assert taps(a) and attack_key() == "f"
    unlock()
    a.jump_back(LEFT)
    assert attack_key() == "q" and asked == [LEFT, LEFT, LEFT, RIGHT, LEFT]


def test_a_frame_telling_the_other_way_from_her_last_arrow_goes_over_it_and_is_said_once(keys_at):
    a, now = keys_at
    a.facing = None
    a.throw(RIGHT)  # tapped right
    later = a.arrow_until + 0.5
    assert a.see(LEFT, later) == RIGHT and a.heading == LEFT  # turned without a key: said, and the frame goes
    assert a.see(LEFT, later + 0.1) is None                    # ... once
    now[0] = later + 1
    a.throw(RIGHT)
    assert taps(a)  # the throw turns her back first


def test_while_her_attack_plays_only_a_strong_frame_counts(keys_at):
    a, now = keys_at
    a.facing = RIGHT
    a.throw(RIGHT)  # no tap; her attack plays until arrow_until
    t = now[0] + 0.3
    assert t < a.arrow_until
    assert a.see(LEFT, t) is None and a.heading == RIGHT and a.saw == "her attack playing"
    assert a.see(LEFT, t, strong=True) == RIGHT and a.heading == LEFT


def test_a_frame_that_cant_tell_leaves_her_facing_but_after_a_hit_makes_it_unknown(keys_at):
    a, now = keys_at
    a.facing = RIGHT
    t = now[0] + 2.0
    assert a.see(None, t) is None and a.heading == RIGHT and a.saw == "not told"
    assert a.see(None, t + 0.1, hit_at=t + 0.05) is None and a.heading is None and a.saw == "unknown after a hit"
    now[0] = t + 1
    a.throw(RIGHT)
    assert taps(a)  # unknown: the safe tap


def test_no_frame_counts_while_a_navigator_chain_plays(keys_at):
    a, now = keys_at
    a.facing = RIGHT
    a.sent_at = now[0]  # the navigator's chains go out around play (client.send)
    assert a.see(LEFT, now[0] + 5) is None and a.heading == RIGHT and a.saw == "a chain playing"
    now[0] += 6
    a.facing = None  # the navigator: the chains are over
    assert a.see(LEFT, now[0] + 0.1) is None  # what they last pressed may not show yet
    assert a.see(LEFT, now[0] + HEADING_SETTLE_S + 0.01) is None and a.heading == LEFT


def test_her_keys_say_her_facing_beyond_doubt_only_well_after_her_arrow_with_no_hit_or_doubt(keys_at):
    a, now = keys_at
    a.walk(RIGHT, 50)
    end = a.arrows_at
    assert a.keys_say(end + 0.1) is None and a.keys_say(end + LEARN_AFTER_S) == RIGHT
    assert a.keys_say(end + 1, hit_at=end + 0.5) is None  # a hit since: a knock may have turned her
    a.see(LEFT, end + 1)
    assert a.keys_say(end + 2) is None  # a frame said otherwise
    a.facing = None
    assert a.keys_say(end + 3) is None  # a move: the keys don't know


# --- the simulator --------------------------------------------------------------------------------------------------

needs_tree = pytest.mark.skipif(not scenarios.available(TREE), reason="needs Tree Dungeon's game data (gamedata/data)")


def _kill_on_23(noise=None, realism=None, read=True, seed=1, start=-150.0, mushrooms=(120.0,), facing=RIGHT, her=None):
    """`hunt kill` of the rightmost Horny Mushroom on 23 (Tree Dungeon; her at `start`, mushrooms at `mushrooms`): her
    attacks counted, and those that went the other way from where the stack threw them (the world's count, by her
    facing as each started). `read` False: eyes that can't read her picture. What came of it, and the trace."""
    out = {}

    def fn(sim):
        if not read:
            sim.eyes.heading = lambda obs: Reading(None, why="no pictures")
        h = sim.hunter()
        try:
            s = h.look()
            out["ended"] = h.kill(max(s.monsters, key=lambda m: m.x), seconds=40)
        finally:  # (she may die: SimDied ends the run)
            out["counts"] = dict(h.heading_counts)
            out["throws"], out["wrong way"] = sim.world.attacks_meant, sim.world.wrong_way
            out["taps"] = h.actions.turns
        return {}
    sc = scenarios.Scenario("heading_kill_23", fn, (23, start), monsters=[("Horny Mushroom", 23, x) for x in mushrooms],
                            seed=seed, limit=60, noise=noise, realism=realism, map_id=TREE, facing=facing, her=her)
    trace = scenarios.run_scenario(sc)
    out["knocks"] = sum("knocked by" in line for line in trace["world_log"])
    return out, trace


@needs_tree
def test_turned_without_a_key_her_frame_tells_and_no_star_goes_the_wrong_way():
    """Turned away from the Horny Mushroom twice while she shoots it (Noise.turned: her player's own arrow, say): the
    next look's frame tells, so the throw after it taps back first, and only the two throws decided before each turn
    (it came while they waited out her attack lock) go the wrong way; the second turn, against her last arrow, is
    logged, and her scenes carry the heading, who it's by and the frame's reading. Those two throws, and only those, are
    told to the feed again the way she faced (Hunter._retell: after the last look that agreed with her keys; the
    tracker binds their kills on that side). With eyes that can't read her picture, her keys think she faces it: every
    star after the turn goes the wrong way (and the mushroom walks into her)."""
    turned = Noise(turned=((1.5, LEFT), (4.0, LEFT)))
    with telemetry.captured() as records:
        out, trace = _kill_on_23(turned)
    assert out["ended"] == "killed" and out["throws"] >= 8 and out["wrong way"] == 2, out
    assert out["counts"]["against"] == 1
    assert any("the frame shows her facing left, her last arrow right" in line for line in trace["log"])
    assert [line.split("told again: ")[1][:11] for line in trace["log"] if "told again" in line] == ["toward left"] * 2
    hers = [r["her"] for r in records if r["topic"] == "scene" and r.get("her")]
    assert hers and {"facing", "facing_by", "facing_keys", "facing_seen", "facing_strong"} <= set(hers[0])
    assert any(h["facing_by"] == "frame" and h["facing_seen"] == LEFT and h["facing_strong"] for h in hers)
    blind, _ = _kill_on_23(turned, read=False)
    assert blind["wrong way"] >= 4, blind


# (with her touches as the world had them when these were measured: a touch left her a second, and a Horny Mushroom's
# MISSed EVADE_ACC over its accuracy, 29%; the world's 1.45 s and her avoid since 2026-10-28 knock her less often, and
# none of seeds 1-32 keeps the single seed's case below: what's tested is her heading after a knock, not the knocks)
KNOCKED = dict(realism=Realism(knock_turns=True, zero_walls=True, invulnerable=1.0), start=0.0,
               mushrooms=(160.0, -150.0), her=dataclasses.replace(LEVEL_25, avoid=None))


@needs_tree
def test_in_a_world_where_a_knock_turns_her_the_frame_keeps_her_stars_going_the_right_way():
    """Realism.knock_turns (not the game's, by HeavenClient): shooting the mushroom on her right, the one on her left
    walks into her and the knock turns her to face it. The frame tells (logged, against her last arrow), and of her
    stars only one decided before a knock goes the wrong way; with eyes that can't read her picture, 2 do (seed 24,
    two knocks; on seed 3 4 did before her looks were perception's latest frames, Senses.fresh_after)."""
    out, _ = _kill_on_23(seed=24, **KNOCKED)
    assert out["knocks"] >= 1 and out["wrong way"] <= 1 and out["counts"]["against"] >= 1, out
    blind, _ = _kill_on_23(seed=24, read=False, **KNOCKED)
    assert blind["wrong way"] >= 2, blind


@needs_tree
def test_over_eight_seeds_of_knocks_that_turn_her_her_stars_go_the_wrong_way_only_when_decided_before_one(monkeypatch):
    """As above on seeds 1-8: with the frame, at most one wrong-way star a knock (3 for 12 knocks, as measured), where
    eyes that can't read her picture send 6 the wrong way (13 knocks). (Knocks that turned her; the stars by the
    world's count, World.wrong_way. Before her looks were perception's latest frames, Senses.fresh_after: 4 for 10
    knocks, and 16 for 19.) In the world as it picked from a jump until 2026-10-27 (sim.world.AIR_PICK 'widening'): by
    the star's search alone ('search') her jump throws' kills come otherwise and the seeds go otherwise from there (5
    for 13 knocks with the frame, 6 without: the same 8 seeds, chaos)."""
    from gamelab.sim import world as world_mod
    monkeypatch.setattr(world_mod, "AIR_PICK", "widening")
    seen = [_kill_on_23(seed=k, **KNOCKED)[0] for k in range(1, 9)]
    blind = [_kill_on_23(seed=k, read=False, **KNOCKED)[0] for k in range(1, 9)]
    wrong = sum(o["wrong way"] for o in seen)
    assert wrong <= sum(o["knocks"] for o in seen) and sum(o["wrong way"] for o in blind) >= 2 * wrong


@needs_tree
def test_frames_read_the_wrong_way_but_not_strongly_never_take_a_tap_away(monkeypatch):
    """Noise.heading_wrong on every frame, her facing away from the Horny Mushroom at the start: read as weakly as the
    frames read the wrong way on 2026-10-24 (0.31, 0.24 ahead), her first throw taps the arrow first (the safe way;
    the rest she throws by her keys, the frames coming while her attack plays) and none goes the wrong way, as with no
    frame read wrong. Were a frame that isn't strong let take a tap away (Actions.faces by `heading` alone), or were
    the wrong frames strong (which none read was), every star would go the wrong way."""
    away = dict(facing=LEFT, seed=1)
    right, _ = _kill_on_23(**away)
    weak, _ = _kill_on_23(Noise(heading_wrong=(1.0, 0.31, 0.24, False)), **away)
    assert weak["ended"] == right["ended"] == "killed" and weak["wrong way"] == right["wrong way"] == 0, (weak, right)
    # (said, as live: once an arrow, and her loot walk after the kill presses another, 2026-10-26)
    assert weak["taps"] == right["taps"] == 1 and weak["counts"]["against"] >= 1
    sure, _ = _kill_on_23(Noise(heading_wrong=(1.0, 0.6, 0.4, False)), **away)
    assert sure["taps"] == 0 and sure["wrong way"] == sure["throws"] >= 8, sure
    monkeypatch.setattr(Actions, "faces", lambda self, direction: self.heading == direction)
    ungated, _ = _kill_on_23(Noise(heading_wrong=(1.0, 0.31, 0.24, False)), **away)
    assert ungated["taps"] == 0 and ungated["wrong way"] == ungated["throws"] >= 8, ungated


@needs_tree
def test_the_fake_eyes_see_the_worlds_facing_but_not_on_a_rope_nor_on_frames_that_cant_tell():
    seen = {}

    def fn(sim):
        h = sim.hunter()
        s = h.look()
        seen["ground"] = (sim.eyes.heading(s), sim.world.facing)
        sim.world.turn(LEFT)
        seen["turned"] = sim.eyes.heading(h.look())
        return {}
    scenarios.run_scenario(scenarios.Scenario("heading_eyes", fn, (23, 73.0), map_id=TREE))
    (r, facing), turned = seen["ground"], seen["turned"]
    assert r.side == facing == RIGHT and r.strong and turned.side == LEFT
    hanging = {}

    def on_rope(sim):
        hanging["r"] = sim.eyes.heading(sim.hunter().look())
        return {}
    scenarios.run_scenario(scenarios.Scenario("heading_rope", on_rope, (29, -8.0), hanging=-150.0, map_id=TREE))
    assert hanging["r"].side is None and hanging["r"].why == "rope"
    unsure = {}

    def many(sim):
        h = sim.hunter()
        reads = []
        for _ in range(200):
            reads.append(sim.eyes.heading(h.look()))
            sim.clock.sleep(0.05)
        unsure["share"] = sum(r.side is None for r in reads) / len(reads)
        return {}
    scenarios.run_scenario(scenarios.Scenario("heading_unsure", many, (23, 73.0), map_id=TREE,
                                              noise=Noise(heading_unsure=0.2)))
    assert 0.1 <= unsure["share"] <= 0.3
    wrong = {}

    def runs(sim):
        h = sim.hunter()
        reads = []
        for _ in range(200):
            reads.append(sim.eyes.heading(h.look()))
            sim.clock.sleep(0.05)
        wrong["reads"] = [(r.side != sim.world.facing, r.score) for r in reads]
        return {}
    scenarios.run_scenario(scenarios.Scenario("heading_wrong", runs, (23, 73.0), map_id=TREE,
                                              noise=Noise(heading_wrong=(0.3, 0.31, 0.24, True))))
    flips = [w for w, _ in wrong["reads"]]
    assert 0.1 <= sum(flips) / len(flips) <= 0.5 and all(score == 0.31 for w, score in wrong["reads"] if w)
    assert sum(a != b for a, b in zip(flips, flips[1:])) < sum(flips) / 2  # in runs (a spell, 20 looks), not alone


# --- the live check's command (watch --heading) ---------------------------------------------------------------------

def test_watch_says_each_change_of_her_facing(monkeypatch):
    """`watch --heading` reads her facing off every frame the perception service shows it (Eyes.heading on the ring's
    frame), says each change (not every frame) and counts how many frames told it; it presses nothing, a viewer."""
    import time
    from collections import deque

    from gamelab import __main__ as cli
    from gamelab.localization.worldstate import WorldState
    logged = []
    monkeypatch.setattr(cli, "_log", logged.append)

    class Perception:  # the service's client: a new state at each wait, 30 a second
        def __init__(self):
            self.states, self.seq, self.run = deque(maxlen=8), 0, (1, 1)

        def wait_frame(self, timeout):
            time.sleep(1 / 30)
            self.seq += 1
            self.states.append(WorldState(self.seq, time.monotonic(), True, tag=(480, 308), tag_seen=True,
                                          where=(1000.0, 500.0, 0, (0.0, 0.0))))
            return True

        def frame_of(self, st):
            return np.zeros((540, 960, 3), np.uint8)

        def status(self):
            return {"clients": [{"name": "watch", "acts": False}]}

    class Turning:  # she faces right for the first 0.3 s of frames, then left (the arrow pressed by hand)
        first = None

        def heading_bank(self):
            return SimpleNamespace(refs=[1], pool=[1, 2], ready=True)

        def heading(self, obs):
            self.first = obs.t if self.first is None else self.first
            return Reading(RIGHT if obs.t - self.first < 0.3 else LEFT, 0.6, 0.4)
    monkeypatch.setattr(cli, "_perception", lambda model, title, name, acts: Perception())
    monkeypatch.setattr(cli, "_heading_eyes", lambda title: Turning())
    assert cli.main(["watch", "--minutes", "0.012", "--heading", "--map", "10000012"]) == 0
    said = [line.strip() for line in logged if "her frame" in line]
    assert said == ["her frame: faces right (matched 0.60, 0.40 ahead)",
                    "her frame: faces left (matched 0.60, 0.40 ahead)"]
