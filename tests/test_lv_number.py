"""Her level off the bottom bar's "LV." digits (perception.lv_number): read on the saved frames as the level her HP
number's max says, garbage refused, drawn levels read back at both client sizes, read only when its box changes; the
perception service's world step reads it (an `lv_read` event, WorldState.level_of), and the hunter takes it
(behavior.bootstrap: tests/test_bootstrap.py)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gamelab.perception import lv_number
from gamelab.screen import STRETCHED, geometry

FIXTURES = Path(__file__).parent / "fixtures" / "lv_number"
NATIVE = geometry(1366, 768)


def _boxes():
    want = json.loads((FIXTURES / "boxes.json").read_text(encoding="utf-8"))["frames"]
    with np.load(FIXTURES / "boxes.npz") as z:
        return {name: (z[name], want[name]) for name in z.files}


def test_where_it_is():
    """The status bar's panel (StatusBar.img canvas:backgrnd1) is drawn at the bar's top left (279, 695) less its
    origin; a two-digit level's first digit has its top left at (329, 744): the box from 317 to 363, rows 742-759."""
    assert lv_number.read_box(NATIVE) == (317, 742, 363, 760)
    x0, y0, x1, y1 = lv_number.read_box(STRETCHED)
    assert x1 - x0 > 46 and y1 - y0 > 18  # (the stretched window's: wider and taller)


def test_on_the_saved_frames_it_reads_the_level_her_hp_number_says():
    """The 40 lossless 1366x768 frames of lab/ and lab/shots: every one with the game's HUD reads its level exactly (each
    cell within a hundredth of a level of its glyph), the level her max HP says (22 x level + 224); the 3 hp0_* frames of
    garbage are refused (no wrong value ever)."""
    boxes = _boxes()
    assert len(boxes) == 40
    read = {}
    for name, (box, want) in boxes.items():
        r = lv_number.read(box, NATIVE)
        if want is None:
            assert r.level is None and r.why.startswith("a cell unlike any glyph"), name
            continue
        assert r.level == want["level_by_hp"] and r.fit < 0.1 and r.lead >= 19, (name, r)
        read[name] = r.level
    assert sorted(set(read.values())) == [24, 25, 26, 28] and len(read) == 37


@pytest.mark.parametrize("geo", [NATIVE, STRETCHED], ids=["1366x768", "1920x1080"])
def test_a_drawn_level_reads_back(geo):
    """The reader's own model of the bar (the simulator's drawing: draw) reads back at either size, one digit or two
    (three-digit levels aren't placed as the game would: none seen)."""
    for level in (1, 9, 10, 19, 25, 28, 99):
        assert lv_number.read(lv_number.draw(level, geo), geo).level == level
    blank = lv_number.draw_text(lv_number.LV, "", None, geo)
    assert lv_number.read(blank, geo).level is None and "not a level" in lv_number.read(blank, geo).why


def test_a_wrong_box_or_none_isnt_read():
    assert lv_number.read(None).why == "no level box"
    assert lv_number.read(np.zeros((5, 5, 3), np.uint8), NATIVE).why == "no level box"
    noise = np.random.default_rng(1).integers(0, 255, lv_number.draw(25, NATIVE).shape, dtype=np.uint8)
    assert lv_number.read(noise, NATIVE).level is None


def test_it_reads_only_when_its_box_changes():
    f = lv_number.Follow()
    box = lv_number.draw(28, NATIVE)
    assert f.see(box, NATIVE).level == 28 and f.level == 28 and f.reads == 1
    assert f.see(box.copy(), NATIVE) is None and f.reads == 1 and f.level == 28
    assert f.see(lv_number.draw(29, NATIVE), NATIVE).level == 29 and f.reads == 2
    garbage = np.full_like(box, 255)
    assert f.see(garbage, NATIVE).level is None and f.level is None and f.failed == 1
    assert f.see(None) is None and f.level is None


def test_the_world_step_reads_it_and_says_so():
    """WorldStep._vitals: an `lv_read` event at each change of the digits that reads, every state's `level_of` what they
    say now (None over a box that doesn't read); a look's Scene has it, and its record (to_scene)."""
    from test_worldstate import Drawn, small_map

    from gamelab.localization.senses import Senses
    from gamelab.localization.worldstate import Schedule, WorldStep, to_scene

    class WithLevel(Drawn):
        level = 25

        def frame(self, t, ropes):
            obs = super().frame(t, ropes)
            box = lv_number.draw(self.level, STRETCHED) if self.level else None
            obs.lv_box = box if self.level != -1 else np.full_like(lv_number.draw(1, STRETCHED), 255)
            return obs
    model = small_map()
    eyes = WithLevel(model)
    step = WorldStep(Senses(eyes, model, band=(100, 100)), schedule=Schedule())
    plan = {0: 25, 10: 26, 20: -1, 25: 26}
    states = []
    for k in range(30):
        eyes.level = plan.get(k, eyes.level)
        t = 100 + k / 30
        states.append(step.step(lambda ropes, t=t: eyes.frame(t, ropes), t))
    reads = [(s.seq, e.data) for s in states for e in s.events if e.kind == "lv_read"]
    assert reads == [(1, {"level": 25}), (11, {"level": 26}), (26, {"level": 26})]
    assert [s.level_of for s in states[9:12]] == [25, 26, 26] and states[22].level_of is None
    assert to_scene(states[29], model, (100, 100)).level_of == 26
    assert to_scene(states[29], model, (100, 100)).record()["level_of"] == 26
