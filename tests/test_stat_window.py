"""Her W.DEF off her stat window (perception.stat_window), prepared and not read live yet: no saved frame shows the
window, so these frames are made of the game data's own pictures (UI CharacterStat.img) pasted on a saved frame, the
value drawn in its cell. Where it's found, where W.DEF's cell is, its number taken from what the OCR reads (the real
OCR with --slow); and what it gives her stats (behavior.bootstrap: her stat window's over her touches)."""

from pathlib import Path

import numpy as np
import pytest

from gamelab.perception import stat_window as sw

LAB = Path(__file__).resolve().parents[1] / "gamelab" / "lab"


def _frame(at=(700, 200), wdef: str | None = "111", jpeg: bool = False):
    """A 1366x768 frame (a saved one if it's here, else grey) with the Detail panel pasted at `at` and `wdef` drawn in
    W.DEF's cell (a plain font: the game's own isn't to hand)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        pytest.skip("needs PIL")
    base = LAB / "calibrate_1366x768.png"
    frame = (np.asarray(Image.open(base).convert("RGB")).copy() if base.is_file()
             else np.full((768, 1366, 3), 90, np.uint8))
    rgba = sw.picture("Detail/backgrnd")
    h, w = rgba.shape[:2]
    x, y = at
    a = rgba[..., 3:] / 255.0
    frame[y:y + h, x:x + w] = (rgba[..., :3] * a + frame[y:y + h, x:x + w] * (1 - a)).astype(np.uint8)
    if wdef is not None:
        img = Image.fromarray(frame)
        try:
            font = ImageFont.truetype("arial.ttf", 11)
        except OSError:
            font = ImageFont.load_default()
        ImageDraw.Draw(img).text((x + sw.WDEF_AT[0], y + sw.WDEF_AT[1] + 1), wdef, fill=(0, 0, 0), font=font)
        frame = np.asarray(img).copy()
    if jpeg:
        import cv2
        ok, data = cv2.imencode(".jpg", frame[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)[..., ::-1].copy()
    return frame


def test_the_window_and_its_cell_from_the_game_data():
    """S opens it (ABILITY in her key bindings, lab/shots/keyconfig.png); its DETAIL panel holds WEAPON DEF., its value
    drawn from (78, 28) in the light cell 76-176 x 28-42 (Detail/vector:pdd; the picture's own px)."""
    assert sw.KEY == "s" and sw.WDEF_AT == (78, 28) and sw.CELL == (76, 28, 177, 43)
    detail = sw.picture("Detail/backgrnd")
    assert detail.shape == (223, 187, 4) and sw.picture("Main/backgrnd").shape == (281, 185, 4)
    x0, y0, x1, y1 = sw.CELL
    cell = detail[y0:y1, x0:x1, :3].reshape(-1, 3)
    assert (cell == (225, 223, 213)).all()  # one colour: the value is all that's drawn on it
    assert sw.cell((700, 200)) == (776, 228, 877, 243)


def test_it_is_found_where_it_is_and_not_where_it_isnt():
    at, fit = sw.find(_frame((640, 180), None))
    assert at == (640, 180) and fit == 0.0
    at, fit = sw.find(_frame((640, 180), "111"))  # (the value drawn over a cell: a few px off the picture)
    assert at == (640, 180) and fit < sw.FIND_FIT
    at, fit = sw.find(_frame((640, 180), "111", jpeg=True))
    assert at == (640, 180) and fit < sw.FIND_FIT
    plain = _frame((640, 180), None)
    plain[180:403, 640:827] = 90
    at, fit = sw.find(plain)
    assert at is None and fit > 10 * sw.FIND_FIT


def test_wdef_is_the_first_number_in_its_cell():
    """The cell cut, enlarged x2 as the message bar's is, read by the OCR (here a stand-in that says what it was given:
    the real one with --slow), its first whole number taken; none there, or no window: None, why said."""
    frame = _frame((700, 200), "111")
    given = []

    def ocr(bgr):
        given.append(bgr.shape)
        return "111"
    got = sw.read_wdef(frame, ocr)
    assert (got.wdef, got.at, got.why) == (111, (700, 200), "") and given == [(30, 202, 3)]
    assert sw.read_wdef(frame, lambda bgr: "W.DEF 111 (+5)").wdef == 111
    assert sw.read_wdef(frame, lambda bgr: "").why == "no number in W.DEF's cell ('')"
    blank = _frame((700, 200), None)
    blank[200:423, 700:887] = 90
    assert sw.read_wdef(blank, ocr).wdef is None and "no stat window's Detail panel" in sw.read_wdef(blank, ocr).why


@pytest.mark.slow
def test_the_real_ocr_reads_a_drawn_value():
    pytest.importorskip("rapidocr")
    got = sw.read_wdef(_frame((700, 200), "111"))
    assert got.wdef == 111, got


def test_what_it_reads_is_her_def_kept_over_her_touches(tmp_path):
    from gamelab.behavior.bootstrap import HerStats
    st = HerStats(tmp_path / "her_stats.json")
    got = sw.read_wdef(_frame((700, 200), "111"), lambda bgr: "111")
    assert st.see_window(got.wdef, at=0.0).startswith("her DEF 111 (her stat window's W.DEF, read ")
    assert HerStats(tmp_path / "her_stats.json").defence.source == "stat window"
