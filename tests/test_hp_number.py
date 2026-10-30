"""Her HP number in the bottom bar (perception.hp_number): "[624/840]" read glyph by glyph as her EXP number is, on the
lossless frames saved in game (tests/fixtures/hp_number: its box cut from each of 19 of the 40 1366x768 frames of lab/
and lab/shots, every value among them), drawn numbers read back, garbage and impossible numbers refused, the bar
against it; read only when its pixels change (Follow), and by the feed's step at every change as an event `hp_read`."""

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from gamedata.paths import COPY
from gamelab.perception import hp_number
from gamelab.perception.hp_number import Follow, read, read_box, shown
from gamelab.screen import STRETCHED, geometry

pytestmark = pytest.mark.skipif(not (COPY / "UI" / "UI_000.wz").is_file(),
                                reason="needs the game data copy's UI (gamedata/data/copy)")
NATIVE = geometry(1366, 768)
FIXTURES = Path(__file__).parent / "fixtures" / "hp_number"
LAB = Path(__file__).resolve().parents[1] / "gamelab" / "lab"


def _fixtures():
    boxes = np.load(FIXTURES / "boxes.npz")
    want = json.loads((FIXTURES / "boxes.json").read_text(encoding="utf-8"))
    return {name: (boxes[name], want[name]) for name in want}


def test_its_box_at_1366x768_is_beside_the_hp_label_on_the_exp_numbers_rows():
    assert read_box(NATIVE) == (517, 738, 609, 750)
    from gamelab.perception import exp_number
    assert read_box(NATIVE)[1::2] == exp_number.read_box(NATIVE)[1::2]


def test_every_saved_frame_reads_as_the_game_showed_it_and_the_garbage_ones_not_at_all():
    """The 16 that read: 752 to 840 (her levels 24 to 28); every cell within 0.6 levels of its glyph and 22 ahead of
    any other. The 3 hp0_* frames of 2026-10-24's evening whose bottom bar is garbage (the HP read 0 then): no value."""
    got = {}
    for name, (box, want) in _fixtures().items():
        r = read(box)
        got[name] = r.of
        if want is None:
            assert r.cur is None and r.why.startswith("a cell"), name
        else:
            assert r.of == tuple(want[:2]) and r.fit < 1.0 and r.lead > 20, (name, r)
    assert sum(v is not None for v in got.values()) == 16
    assert {m for v in got.values() if v is not None for m in [v[1]]} == {752, 774, 796, 840}


def test_calibrate_1366x768_png_reads_752_of_752():
    path = LAB / "calibrate_1366x768.png"
    if not path.exists():
        pytest.skip("needs gamelab/lab/calibrate_1366x768.png")
    from PIL import Image
    full = np.asarray(Image.open(path).convert("RGB"))
    x0, y0, x1, y1 = read_box(NATIVE)
    assert read(full[y0:y1, x0:x1]).of == (752, 752)


@pytest.mark.parametrize("geo", [NATIVE, STRETCHED], ids=["1366x768", "1920x1080"])
def test_numbers_drawn_as_the_game_draws_them_read_back(geo):
    for cur, most in ((0, 840), (1, 1), (262, 840), (840, 840), (9999, 9999), (12345, 99999)):
        r = read(hp_number.draw(shown(cur, most), geo))
        assert r.of == (cur, most) and r.fit < 1.0, (cur, most, r)


def test_what_isnt_an_hp_number_isnt_read():
    for text, why in (("[900/840]", "900 over its max 840"), ("[840840]", "not an HP number"),
                      ("[0840/840]", "not an HP number"), ("", "not an HP number")):
        r = read(hp_number.draw(text, NATIVE))
        assert r.cur is None and r.why.startswith(why), (text, r)
    box = hp_number.draw("[600/840]", NATIVE)
    box[:, 30:50] = 255  # the cursor over it
    assert read(box).cur is None
    assert read(None).why == "no HP number box" and read(box[:, :-3]).why == "no HP number box"


def test_the_bar_is_never_over_the_number_by_a_px_and_on_a_still_frame_its_ceil_or_one_more():
    """The HP bar's red px (perception.hud.bars, of 104 at 1366x768) against the number on the same frames: on the
    frames of a still bar ceil(104 cur / max) or one more (the Lie Detectors', the menus', calibrate's), never more; on
    those saved as the bar moved (the hp0_* frames: saved because the bar was read wrong, and hunt_look's) under it.
    So the bar less a px (behavior.safety.BAR_SLACK) is under her HP, or over it by less than a px."""
    from gamelab.behavior.safety import BAR_SLACK
    still = moving = 0
    for name, (box, want) in _fixtures().items():
        if want is None:
            continue
        cur, most, px = want
        over = px - math.ceil(104 * cur / most)
        assert over <= 1, name
        assert (px / 104 - BAR_SLACK) * most < cur + most / 104, name
        if name.startswith(("hp0_", "hunt_look")):
            moving += 1
            assert over <= 0, name
        else:
            still += 1
            assert over in (0, 1), name
    assert (still, moving) == (10, 6)


def test_it_is_read_only_when_its_pixels_change_and_says_what_it_said_before():
    f = Follow()
    full = hp_number.draw(shown(840, 840), NATIVE)
    r, before = f.see(full, NATIVE)
    assert r.of == (840, 840) and before is None and f.of == (840, 840) and f.reads == 1
    assert f.see(full.copy(), NATIVE) is None and f.reads == 1  # the same pixels: not read again
    r, before = f.see(hp_number.draw(shown(723, 840), NATIVE), NATIVE)
    assert (r.cur, before) == (723, 840) and f.of == (723, 840)
    covered = hp_number.draw(shown(723, 840), NATIVE)
    covered[:, 30:50] = 255
    r, before = f.see(covered, NATIVE)  # covered: not read, nothing known
    assert r.cur is None and before == 723 and f.of is None and f.failed == 1
    r, before = f.see(hp_number.draw(shown(823, 840), NATIVE), NATIVE)
    assert r.cur == 823 and before is None  # what came between isn't known: no touch from it
    assert f.see(None) is None and f.of is None


def test_a_read_takes_a_few_ms():
    hp_number.warm(NATIVE)
    box = hp_number.draw(shown(624, 840), NATIVE)
    took = []
    for _ in range(5):
        t = time.perf_counter()
        read(box)
        took.append(time.perf_counter() - t)
    assert min(took) < 0.05  # 2.9 ms median on this PC (2026-10-28); a loose bound for a busy machine


def test_the_feeds_step_reads_it_at_every_change_and_says_so():
    """WorldStep._vitals: an `hp_read` event at each change of the number that read (her HP, max HP and what it said at
    the change before if that read: a drop between the two is a touch), and every state's `hp_of` what it says now; a
    look's Scene has it (to_scene)."""
    from test_worldstate import Drawn, small_map

    from gamelab.localization.senses import Senses
    from gamelab.localization.worldstate import Schedule, WorldStep, to_scene

    class WithNumber(Drawn):
        text = shown(840, 840)

        def frame(self, t, ropes):
            obs = super().frame(t, ropes)
            obs.hp_box = hp_number.draw(self.text, STRETCHED) if self.text is not None else None
            return obs
    model = small_map()
    eyes = WithNumber(model)
    step = WorldStep(Senses(eyes, model, band=(100, 100)), schedule=Schedule())
    plan = {0: shown(840, 840), 10: shown(723, 840), 12: shown(823, 840), 20: "[8", 25: shown(700, 840)}
    states = []
    for k in range(30):
        eyes.text = plan.get(k, eyes.text)
        t = 100 + k / 30
        states.append(step.step(lambda ropes, t=t: eyes.frame(t, ropes), t))
    reads = [(s.seq, e.data) for s in states for e in s.events if e.kind == "hp_read"]
    assert reads == [(1, {"cur": 840, "max": 840, "before": None}), (11, {"cur": 723, "max": 840, "before": 840}),
                     (13, {"cur": 823, "max": 840, "before": 723}), (26, {"cur": 700, "max": 840, "before": None})]
    assert [s.hp_of for s in states[9:13]] == [(840, 840), (723, 840), (723, 840), (823, 840)]
    assert states[22].hp_of is None and states[29].hp_of == (700, 840)
    assert to_scene(states[29], model, (100, 100)).hp_of == (700, 840)
    assert to_scene(states[29], model, (100, 100)).record()["hp_of"] == [700, 840]


def test_a_drop_over_frames_the_step_didnt_read_isnt_a_touch():
    """The review of 2026-10-28 (D:/MapleHelper-scratch/safety3/review-evidence/demo/gap_demo.py): 60 frames not read
    (the game not in front) while the number went 840 -> 723 -> 606 made one `hp_read` of 606 'before' 840, a touch of
    234 kept, which fits no kind and counted for every kind of the map, her reserve 59% of her HP. Now a frame not
    visible, one failing its checks, or a stretch of over HP_GAP_S with no frame has the number forgotten: the next
    read's `before` is None (not known), and only a drop over frames all read is a touch."""
    from test_worldstate import Drawn, small_map

    from gamelab.localization.senses import Senses
    from gamelab.localization.worldstate import HP_GAP_S, Schedule, WorldStep
    from gamelab.perception.framecheck import Problem

    class WithNumber(Drawn):
        text, bad = shown(840, 840), False

        def frame(self, t, ropes):
            obs = super().frame(t, ropes)
            obs.hp_box = hp_number.draw(self.text, STRETCHED)
            if self.bad:
                obs.bad = Problem("window", "the window moved")
            return obs
    model = small_map()
    eyes = WithNumber(model)
    step = WorldStep(Senses(eyes, model, band=(100, 100)), schedule=Schedule())
    clock, reads = [100.0], []

    def frame(cur, visible=True, bad=False, dt=1 / 30):
        clock[0] += dt
        t = clock[0]
        eyes.text, eyes.bad = shown(cur, 840), bad
        st = step.step(lambda ropes, t=t: eyes.frame(t, ropes), t, visible=visible,
                       why="" if visible else "the game isn't in front")
        reads.extend(e.data for e in st.events if e.kind == "hp_read")
        return st
    for _ in range(10):
        frame(840)
    for cur in [723] * 30 + [606] * 30:  # not visible
        frame(cur, visible=False)
    assert frame(606).hp_of == (606, 840) and reads[-1] == {"cur": 606, "max": 840, "before": None}
    for _ in range(5):
        frame(489, bad=True)  # failing their checks: none read, nor the HP of either
    st = frame(489)
    assert st.hp_of == (489, 840) and reads[-1] == {"cur": 489, "max": 840, "before": None}
    frame(372, dt=HP_GAP_S + 0.1)  # a stretch with no frame
    assert reads[-1] == {"cur": 372, "max": 840, "before": None}
    frame(255)  # every frame read: a touch
    assert reads[-1] == {"cur": 255, "max": 840, "before": 372}
    assert [r["before"] for r in reads] == [None, None, None, None, 372]
