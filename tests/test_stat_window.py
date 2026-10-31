"""Her stat window read off a frame (perception.stat_window): not seen live yet (no saved frame shows it), so these frames
are made of the game data's own pictures (UI CharacterStat.img) pasted on a saved frame, the values drawn in their
cells (`compose`), and the OCR is a stand-in that says what was drawn where (the real one with --slow). Where it's
found, where each value's cell is, how a value's text is parsed; the calibration a live capture leaves (the game data's
pictures, or a window cut from the live frame where they don't match, its cells found by their labels); and what a read
gives her stats (behavior.bootstrap: her stat window's DEF over her touches)."""

import json
from pathlib import Path

import numpy as np
import pytest

from gamelab.perception import stat_window as sw

LAB = Path(__file__).resolve().parents[1] / "gamelab" / "lab"
VALUES = {"name": "BugBaby", "job": "Rogue", "level": 28, "hp": (840, 840), "mp": (138, 679), "str": 4, "dex": 25,
          "int": 4, "luk": 120, "attack": (60, 75), "wdef": 111, "magic": 10, "mdef": 50, "acc": 85, "eva": 60,
          "speed": 119, "jump": 100}


def _base():
    """A saved 1366x768 frame of the game if it's here, else a grey one."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("needs PIL")
    base = LAB / "calibrate_1366x768.png"
    return (np.asarray(Image.open(base).convert("RGB")).copy() if base.is_file()
            else np.full((768, 1366, 3), 90, np.uint8))


def _jpeg(frame):
    import cv2
    ok, data = cv2.imencode(".jpg", frame[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(data, cv2.IMREAD_COLOR)[..., ::-1].copy()


class Texts:
    """A stand-in OCR over a frame's drawn texts: ocr(crop, box) -> the rows of those whose centre is in `box`, in the
    crop's px (no pixel check: for frames drawn here by hand)."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.asked = []

    def __call__(self, crop, box):
        self.asked.append(tuple(box))
        x0, y0, x1, y1 = box
        out = []
        for (a, b, c, d), text in self.texts:
            cx, cy = (a + c) / 2, (b + d) / 2
            if x0 <= cx < x1 and y0 <= cy < y1:
                out.append(sw.Row(text, cx - x0, cy - y0, a - x0, c - x0, d - b))
        return out


def test_the_window_and_its_cells_from_the_game_data():
    """S opens it (ABILITY in her key bindings, lab/shots/keyconfig.png); CHARACTER STAT (185 x 281) and its DETAIL
    panel (187 x 223: backgrnd2 the same but for its left 4 columns), each value in a light cell around its vector:
    CELLS as measured off the pictures (HP, MP and the four stats cut short of their AP buttons), W.DEF's from (78,
    28), the DETAIL button where the game data puts it (its origin negated)."""
    assert sw.KEY == "s" and sw.FUNCTION == "ABILITY" and sw.WDEF_AT == (78, 28) and sw.CELL == (76, 28, 177, 43)
    assert sw.picture("Detail/backgrnd").shape == (223, 187, 4) and sw.picture("Main/backgrnd").shape == (281, 185, 4)
    two = sw.picture("Detail/backgrnd2").astype(int)[58:]
    one = sw.picture("Detail/backgrnd").astype(int)
    assert (one[:, sw.DETAIL_LEFT:] == two[:, sw.DETAIL_LEFT:]).all() and not (one == two).all()
    for panel in ("main", "detail"):
        measured = sw.cells_of(panel)
        for name, (x0, y0, x1, y1) in sw.CELLS[panel].items():
            m = measured[name]
            assert (x0, y0) == m[:2] and y1 == m[3] and x1 <= m[2], (panel, name, m)
            cell = sw.picture("Main/backgrnd" if panel == "main" else "Detail/backgrnd")[y0:y1, x0:x1, :3]
            assert (cell.reshape(-1, 3) == sw.LIGHT).mean() > 0.95  # (the light cell, a row of the job's two)
    assert sw.picture("Main/button:detail").shape[:2] == (sw.DETAIL_BUTTON[3], sw.DETAIL_BUTTON[2])  # (18 x 47)
    assert sw.detail_button((560, 150)) == (560 + 122 + 23, 150 + 256 + 9)
    assert sw.cell((700, 200)) == (776, 228, 877, 243)


def test_it_is_found_where_it_is_and_not_where_it_isnt():
    """Both panels found to the px where they're pasted, whatever their values say (their cells left out of the
    match), as a JPEG too; on the frame without them nowhere (20,000 and more at the best place)."""
    base = _base()
    drawn = sw.compose(VALUES, base=base, at={"main": (300, 120), "detail": (485, 120)})
    found = sw.find_all(drawn.frame)
    assert found["main"] == ((300, 120), 0.0) and found["detail"] == ((485, 120), 0.0)
    other = sw.compose({**VALUES, "wdef": 9999, "hp": (1, 99999), "name": "Somebody Else"}, base=base,
                       at={"main": (300, 120), "detail": (485, 120)})
    assert {k: v[1] for k, v in sw.find_all(other.frame).items()} == {"main": 0.0, "detail": 0.0}
    jpeg = sw.find_all(_jpeg(drawn.frame))
    assert jpeg["main"][0] == (300, 120) and jpeg["detail"][0] == (485, 120)
    assert max(v[1] for v in jpeg.values()) < sw.FIND_FIT
    nothing = sw.find_all(base)
    assert all(at is None and fit > 10 * sw.FIND_FIT for at, fit in nothing.values())
    assert not sw.is_open(base) and sw.is_open(drawn.frame)
    # Where it was last seen first (SLACK), else anywhere: moved 3 px, then 300
    assert sw.find(drawn.frame, "main", near=(303, 118))[0] == (300, 120)
    assert sw.find(drawn.frame, "main", near=(600, 400))[0] == (300, 120)
    alone = sw.compose(VALUES, base=base, detail=False)
    assert sw.find_all(alone.frame)["detail"][0] is None and sw.find_all(alone.frame)["main"][0] == sw.DRAW_AT["main"]


def test_shown_exactly_where_its_calibration_has_it_open():
    """`shown` (her place by the scene counted covered under it: localization.locate.Cover): the first panel found at
    its calibrated place alone, no search, whatever its values say, as a JPEG too; moved off it (the user dragged it),
    or with no calibration, nothing; a place off the frame skipped."""
    base = _base()
    cal = sw.Calibration(panels={"main": {"at": [300, 120]}, "detail": {"at": [485, 120]}}, detail_with_main=True)
    drawn = sw.compose(VALUES, base=base, at={"main": (300, 120), "detail": (485, 120)})
    assert sw.shown(drawn.frame, cal) == "main" and sw.shown(_jpeg(drawn.frame), cal) == "main"
    alone = sw.compose(VALUES, base=base, at={"main": (600, 120), "detail": (485, 120)})   # (its Main panel elsewhere)
    assert sw.shown(alone.frame, cal) == "detail"
    moved = sw.compose(VALUES, base=base, at={"main": (500, 300), "detail": (685, 300)})
    assert sw.shown(moved.frame, cal) is None and sw.shown(base, cal) is None and sw.shown(drawn.frame, None) is None
    off = sw.Calibration(panels={"main": {"at": [1300, 700]}, "detail": {"at": [485, 120]}})
    assert sw.shown(drawn.frame, off) == "detail" and sw.shown(drawn.frame, sw.Calibration(panels={})) is None


def test_shown_over_a_box_only_a_panel_over_some_of_it():
    """`over` (where the scene's filter has her feet and tag: the review of 2026-10-30, a loss of her away from the
    window had counted covered): only a panel over some of the box is looked at (Main x 300-485, y 120-401; Detail
    x 485-672, y 120-343)."""
    cal = sw.Calibration(panels={"main": {"at": [300, 120]}, "detail": {"at": [485, 120]}}, detail_with_main=True)
    drawn = sw.compose(VALUES, base=_base(), at={"main": (300, 120), "detail": (485, 120)}).frame
    assert sw.shown(drawn, cal, over=(350, 200, 400, 250)) == "main"
    assert sw.shown(drawn, cal, over=(500, 200, 550, 250)) == "detail"
    assert sw.shown(drawn, cal, over=(350, 395, 400, 450)) == "main"                  # (its bottom rows)
    assert sw.shown(drawn, cal, over=(350, 401, 400, 450)) is None                    # (just under it)
    assert sw.shown(drawn, cal, over=(700, 200, 750, 250)) is None                    # (beside it)
    assert sw.shown(_base(), cal, over=(350, 200, 400, 250)) is None                  # (over her, not open)


def test_both_panels_read_in_one_ocr_call_each():
    """Each panel's value column read by the OCR in one go, each row taken for the cell it falls in, and parsed: HP
    and MP (cur, max), ATTACK (lo, hi), the rest the first whole number; the DETAIL panel closed: no W.DEF, said."""
    drawn = sw.compose(VALUES, base=_base())
    asked = []

    def ocr(crop, box):
        asked.append(box)
        return drawn.ocr(crop, box)
    r = sw.read(drawn.frame, ocr)
    assert r.open and r.detail and r.why == "" and len(asked) == 2
    assert r.values == {k: v for k, v in VALUES.items()}
    assert (r.wdef, r.level, r.max_hp, r.max_mp) == (111, 28, 840, 679)
    assert r.text().startswith("W.DEF 111, level 28, HP 840/840, MP 138/679, ATTACK 60~75, ACC 85, AVOID 60")
    alone = sw.compose(VALUES, base=_base(), detail=False)
    r = sw.read(alone.frame, alone.ocr)
    assert r.open and not r.detail and r.wdef is None and r.level == 28
    assert r.why == "the DETAIL panel isn't open (her W.DEF is on it)"
    blank = sw.compose({**VALUES, "wdef": None}, base=_base())
    r = sw.read(blank.frame, blank.ocr)
    assert r.wdef is None and r.why == "no number in W.DEF's cell ('')"
    r = sw.read(_base(), drawn.ocr)
    assert not r.open and r.why.startswith("no panel of the stat window on the frame (the best fits ")
    # (a stand-in reading a crop of another frame reads nothing: the simulator's eyes' stat_ocr)
    assert drawn.ocr(np.zeros((10, 10, 3), np.uint8), (0, 0, 10, 10)) == []


@pytest.mark.parametrize("name, text, value", [
    ("hp", "840 / 840", (840, 840)), ("hp", "840/ 840", (840, 840)), ("mp", "138 /679", (138, 679)),
    ("hp", "840", (None, 840)), ("attack", "60~ 75", (60, 75)), ("attack", "60 - 75", (60, 75)),
    ("wdef", "111", 111), ("wdef", "111 (+5)", 111), ("speed", "119%", 119), ("str", "25 (20+5)", 25),
    ("level", "", None), ("wdef", "W.DEF", None), ("name", " BugBaby ", "BugBaby")])
def test_a_value_parsed_from_its_text(name, text, value):
    assert sw.parse(name, text) == value


def test_wdef_alone_off_a_saved_frame():
    """`gamelab stats --frame`'s read (read_wdef): W.DEF's cell alone, enlarged x2 as the message bar's is, its first
    whole number taken."""
    frame = sw.compose(VALUES, base=_base(), at={"main": (515, 200), "detail": (700, 200)}).frame
    given = []

    def ocr(bgr):
        given.append(bgr.shape)
        return "111"
    got = sw.read_wdef(frame, ocr)
    assert (got.wdef, got.at, got.why) == (111, (700, 200), "") and given == [(30, 202, 3)]
    assert sw.read_wdef(frame, lambda bgr: "W.DEF 111 (+5)").wdef == 111
    assert sw.read_wdef(frame, lambda bgr: "").why == "no number in W.DEF's cell ('')"
    assert "no stat window's Detail panel" in sw.read_wdef(_base(), ocr).why


def test_the_calibration_kept_and_read_back(tmp_path):
    cal = sw.Calibration("copy", {"main": {"at": [560, 150], "size": [185, 281]}}, detail_with_main=False, at=1000.0)
    path = cal.save(tmp_path / "stat_window.json")
    back = sw.Calibration.load(path)
    assert back == sw.Calibration("copy", {"main": {"at": [560, 150], "size": [185, 281]}}, at=1000.0)
    assert back.place("main") == (560, 150) and back.place("detail") is None
    assert "the game data's pictures; main at (560, 150); DETAIL by a click" in back.words()
    for bad in ("not json", json.dumps({"panels": {}}), json.dumps([1, 2])):
        (tmp_path / "bad.json").write_text(bad, encoding="utf-8")
        assert sw.Calibration.load(tmp_path / "bad.json") is None
    assert sw.Calibration.load(tmp_path / "none.json") is None


def test_calibrated_off_the_game_data_where_it_matches():
    drawn = sw.compose(VALUES, base=_base())
    cal, said = sw.calibrate(drawn.frame, _base())
    assert cal.source == "copy" and cal.place("main") == sw.DRAW_AT["main"] and cal.place("detail") == sw.DRAW_AT[
        "detail"]
    assert said.startswith("the game data's pictures match: main at (560, 150) (fit 0.0)")


def _newer_ui(base, at=(300, 200)):
    """A stat window the game data's pictures don't match (theirs recoloured, its labels where theirs are), its values
    drawn, and a stand-in OCR reading its labels and values: (frame, the OCR, its cells in frame px)."""
    frame = base.copy()
    rgba = sw.picture("Main/backgrnd")
    h, w = rgba.shape[:2]
    x, y = at
    a = rgba[..., 3:] / 255.0
    frame[y:y + h, x:x + w] = ((255 - rgba[..., :3]) * a + frame[y:y + h, x:x + w] * (1 - a)).astype(np.uint8)
    texts, cells = [], {}
    labels = {"level": "LEVEL", "hp": "HP", "mp": "MP", "str": "STR", "dex": "DEX", "int": "INT", "luk": "LUK"}
    for name, label in labels.items():
        x0, y0, x1, y1 = sw.CELLS["main"][name]
        texts.append(((x + 12, y + y0, x + 12 + 8 * len(label), y1 + y), label))
        texts.append(((x + x0 + 2, y + y0 + 1, x + x0 + 40, y + y1 - 1), sw.texts_of(VALUES)[name]))
        cells[name] = (x + x0, y + y0, x + x1, y + y1)
    rgba2 = sw.picture("Detail/backgrnd")
    dx, dy = x + w, y
    h2, w2 = rgba2.shape[:2]
    a2 = rgba2[..., 3:] / 255.0
    frame[dy:dy + h2, dx:dx + w2] = ((255 - rgba2[..., :3]) * a2 + frame[dy:dy + h2, dx:dx + w2] * (1 - a2)).astype(
        np.uint8)
    for name, label in {"wdef": "WEAPON DEF.", "acc": "ACCURACY", "eva": "EVASION", "attack": "ATTACK"}.items():
        x0, y0, x1, y1 = sw.CELLS["detail"][name]
        texts.append(((dx + 8, dy + y0, dx + 8 + 6 * len(label), dy + y1), label))
        texts.append(((dx + x0 + 2, dy + y0 + 1, dx + x0 + 40, dy + y1 - 1), sw.texts_of(VALUES)[name]))
    return frame, Texts(texts), (x, y, x + w, y + h), (dx, dy, dx + w2, dy + h2)


def test_calibrated_off_a_live_frame_where_the_game_data_doesnt_match(tmp_path, monkeypatch):
    """A window the game data's pictures don't match (a newer client's): each window that opened (the frame against
    the one before it) cut as a template of the live frame, its value cells found by their labels (the OCR: here a
    stand-in), masked out of it, saved; then read by them: found where it is, its values in their cells."""
    from gamelab import paths
    monkeypatch.setattr(paths, "STAT_TEMPLATES", tmp_path / "stat_window")
    base = _base()
    frame, ocr, main_box, detail_box = _newer_ui(base)
    assert all(at is None for at, _ in sw.find_all(frame).values())  # (the game data's: not on it)
    boxes = sw.changed_boxes(base, frame)
    assert len(boxes) >= 1 and all(b[0] >= main_box[0] - 1 and b[2] <= detail_box[2] + 1 for b in boxes)
    cal, said = sw.calibrate(frame, None, ocr, boxes=[main_box, detail_box])
    assert cal is not None and cal.source == "live", said
    assert set(cal.panels) == {"main", "detail"} and said.startswith("cut from the live frame: main at (300, 200)")
    assert set(cal.panels["detail"]["cells"]) == {"wdef", "acc", "eva", "attack"}
    assert (tmp_path / "stat_window" / "main.png").is_file() and (tmp_path / "stat_window" / "detail.png").is_file()
    # Read by the cut templates: found where it is, its values read in the cells its labels gave
    found = sw.find_all(frame, cal)
    assert found["main"][0] == main_box[:2] and found["detail"][0] == detail_box[:2]
    r = sw.read(frame, ocr, cal)
    assert (r.wdef, r.level, r.max_hp, r.values["acc"], r.values["eva"]) == (111, 28, 840, 85, 60)
    # ... and moved 40 px: found there (the whole frame searched)
    moved, ocr2, box2, _ = _newer_ui(base, at=(340, 220))
    assert sw.find_all(moved, cal)["main"][0] == box2[:2]
    assert sw.read(moved, ocr2, cal).level == 28
    # Nothing to cut: no window found, said
    none, why = sw.calibrate(base, base, ocr)
    assert none is None and "no window found to cut" in why


def test_a_label_and_its_value_on_one_row():
    """The OCR may read a label and its value as one row ("WEAPON DEF. 111"): the field taken, its value cell from where
    the value starts; "DETAIL" the button's place."""
    assert sw._label_of("WEAPON DEF. 111") == ("wdef", " 111")
    assert sw._label_of("Weapon Def") == ("wdef", "")
    assert sw._label_of("MAGIC DEF. 50")[0] == "mdef" and sw._label_of("MAGIC 10")[0] == "magic"
    assert sw._label_of("EVASION 60")[0] == "eva" and sw._label_of("AVOID 60")[0] == "eva"
    assert sw._label_of("nothing")[0] is None
    crop = np.full((60, 180, 3), 200, np.uint8)
    rows = [sw.Row("WEAPON DEF. 111", 70, 10, 5, 140, 12), sw.Row("DETAIL", 150, 50, 130, 170, 12)]
    cells, button = sw.layout(crop, lambda c, b: rows)
    assert button == (150, 50) and cells["wdef"][1:] == (2, 178, 18) and 90 <= cells["wdef"][0] <= 120


@pytest.mark.slow
def test_the_real_ocr_reads_a_drawn_window():
    pytest.importorskip("rapidocr")
    r = sw.read(sw.compose(VALUES, base=_base()).frame)
    assert (r.wdef, r.level, r.max_hp, r.max_mp, r.get("attack")) == (111, 28, 840, 679, (60, 75)), r.texts


@pytest.mark.slow
def test_the_real_ocr_finds_a_newer_windows_labels(tmp_path, monkeypatch):
    """The game data's panel recoloured, its labels as they're drawn on it, values in a plain font: the real OCR finds
    the labels and the cells, and reads W.DEF with them."""
    pytest.importorskip("rapidocr")
    from gamelab import paths
    monkeypatch.setattr(paths, "STAT_TEMPLATES", tmp_path / "stat_window")
    base = _base()
    drawn = sw.compose(VALUES, base=base, at={"main": (300, 200), "detail": (485, 200)})
    frame = drawn.frame.copy()
    for x0, y0, x1, y1 in ((300, 200, 485, 481), (485, 200, 672, 423)):
        region = frame[y0:y1, x0:x1].astype(int)
        frame[y0:y1, x0:x1] = np.dstack([255 - region[..., 2], region[..., 0], region[..., 1]]).astype(np.uint8)
    assert all(at is None for at, _ in sw.find_all(frame).values())
    cal, said = sw.calibrate(frame, base)  # (one window cut: the two panels side by side)
    assert cal is not None and cal.source == "live" and "wdef" in cal.panels["main"]["cells"], said
    r = sw.read(frame, None, cal)
    assert (r.wdef, r.level) == (111, 28), (said, r.texts)


def test_what_it_reads_is_her_def_kept_over_her_touches(tmp_path):
    from gamelab.behavior.bootstrap import HerStats
    st = HerStats(tmp_path / "her_stats.json")
    drawn = sw.compose(VALUES, base=_base())
    got = sw.read(drawn.frame, drawn.ocr)
    note = st.see_window(got.wdef, level=got.level, session="1234", read=got.values)
    assert note.startswith("her DEF 111 (her stat window's W.DEF, read ")
    again = HerStats(tmp_path / "her_stats.json", session="1234")  # (the next run, the same session: hers)
    assert again.defence.source == "stat window" and again.defence.low == 111
    assert again.window_read["level"] == 28 and again.window_read["session"] == "1234"
    assert again.window_read["read"]["hp"] == [840, 840] and again.window_read["read"]["acc"] == 85
    assert "name" not in again.window_read["read"]
