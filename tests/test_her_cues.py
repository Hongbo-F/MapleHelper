"""Her cues' models made at calibration (gamelab/localization/her_cues.py): her feet's offset down from her plate's
centre measured where she stands on a platform's line (16 for her orange-ribbon plate), none taken where too few frames
agree; a folder's frames in its index's order; from frames where her tag as she has it scores low on her plate, its
own sure levels, her new tag template cut from her plate and her models made about its centre, her feet across by the
plate's axis, all written to the folder asked for."""

import json
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from gamelab.localization import her_cues


@dataclass
class Line:
    left: float
    right: float
    y: float
    slope: float = 0.0

    def y_at(self, x: float) -> float:
        return self.y + self.slope * (x - self.left)


def test_her_feet_are_where_the_line_under_her_is_in_most_frames():
    lines = [Line(-200, 100, 50), Line(120, 300, -60, 0.1)]
    view = (-658, -579)
    seen = []
    for k in range(30):   # standing and walking on the first: the plate's centre 16 under the line
        mx = -150 + 8 * k
        seen.append(((mx - 1.5 - view[0], 50 - view[1] + 16), view))
    for k in range(10):   # on the sloped one
        mx = 140 + 15 * k
        seen.append(((mx - 1.5 - view[0], round(-60 + 0.1 * (mx - 120)) - view[1] + 16), view))
    for k in range(12):   # in the air over the first, 1-12 px over its line: the line at other offsets
        seen.append(((-100 - view[0], 50 - view[1] + 16 - (k + 1)), view))
    dy, agree, of = her_cues.feet_below(seen, lines, 1.5)
    assert dy == -16.0 and agree == 40 and of > 40
    dy, agree, of = her_cues.feet_below(seen[30:], lines, 1.5)   # mostly in the air: not enough agree
    assert dy is None or agree >= her_cues.FEET_SURE * of
    assert her_cues.feet_below([], lines, 1.5) == (None, 0, 0)


def test_a_folders_frames_come_in_its_index_order(tmp_path):
    for k, name in enumerate(("b.png", "a.png", "c.png")):
        cv2.imwrite(str(tmp_path / name), np.full((4, 4, 3), k * 50, np.uint8))
    (tmp_path / "index.jsonl").write_text("\n".join(json.dumps({"file": n, "t": t}) for n, t in
                                                    (("b.png", 0.0), ("a.png", 0.07), ("c.png", 0.13))),
                                          encoding="utf-8")
    frames = her_cues.frames_of(tmp_path)
    assert [p.name for p, _ in frames] == ["b.png", "a.png", "c.png"] and [t for _, t in frames] == [0.0, 0.07, 0.13]
    assert [p.name for p, _ in her_cues.frames_of(tmp_path, every=2)] == ["b.png", "c.png"]
    assert her_cues.rgb(frames[1][0])[0, 0].tolist() == [50, 50, 50]


def plate_frames(folder, m, n: int = 36, seed: int = 0) -> list:
    """n lossless frames of her plate (tests/test_plate.py's) at places spread over the play area, its translucent
    interior a little different in each (as the art behind it is), on a smooth background of its own; their index.
    -> the centres."""
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = m.box
    centres, rows = [], []
    ramp = np.add.outer(np.linspace(0, 90, 768), np.linspace(0, 60, 1366))
    for k in range(n):
        fr = np.clip(np.stack([ramp + 3 * k, ramp[::-1] + 20, ramp[:, ::-1] + 40 - k], axis=2), 0, 255).astype(np.uint8)
        cx, cy = 260 + 23 * k, 260 + 9 * (k % 17)
        pic = m.picture.astype(np.int16)
        unstable = m.share < 0.9
        pic[unstable] += rng.integers(-10, 11, (int(unstable.sum()), 3))
        fr[cy + y0:cy + y1, cx + x0:cx + x1] = np.clip(pic, 0, 255).astype(np.uint8)
        name = f"{k:05d}.png"
        cv2.imwrite(str(folder / name), np.ascontiguousarray(fr[..., ::-1]))
        rows.append({"i": k, "file": name, "t": k / 15})
        centres.append((cx, cy))
    (folder / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return centres


def test_her_tag_template_is_cut_from_her_plate_and_her_models_built_on_its_centre(tmp_path):
    """Her tag as she has it scores low on her plate (BugBaby's old text tag: 0.86-0.895), so its sure levels are its
    own (no frame reaches the prototype's 0.95/0.9); her new template is cut from her plate's model, the model made
    again about its centre, her feet across from the plate's own axis, all kept with the templates in the folder."""
    from gamelab.perception import plate as P
    from tests.test_plate import model
    m = model(6)
    frames = tmp_path / "frames"
    frames.mkdir()
    centres = plate_frames(frames, m)
    gray = cv2.cvtColor(np.ascontiguousarray(m.picture), cv2.COLOR_RGB2GRAY).astype(np.int16)
    x0, _, y0, _ = m.box
    old = gray[-7 - y0:-7 - y0 + 15, -32 - x0:-32 - x0 + 64]          # an old tag: 64 x 15 about the plate's centre ...
    old = np.clip(old + np.random.default_rng(1).normal(0, 45, old.shape), 0, 255).astype(np.uint8)   # ... unlike it
    out = tmp_path / "out"
    said = her_cues.build(frames, 10002073, out, old, bank=False, log=lambda s: None)
    typical, picks, bank = said["tag"]["levels"]
    assert typical < 0.95 and picks < P.PICKS_MOST and bank < picks              # its own levels, below the constants
    cut, (dx, dy) = P.tag_template(m)
    assert said["tag"]["centre_moved"] == [dx, dy] and said["tag"]["cut_again"] == [0, 0]
    assert said["tag"]["levels_cut"][0] > 0.99 and said["tag"]["levels_cut"][1:] == [P.PICKS_MOST, P.BANK_MOST]
    made = P.PlateModel.load(P.model_path(out))
    assert made.box == m.recentred(dx, dy).box and made.n == len(centres)
    assert made.feet[0] == made.axis() == m.axis() - dx and made.sure == tuple(said["tag"]["levels_cut"])
    files = her_cues.template_files(out)
    assert [f.name for f in files] == ["tag_template_1366x768.npy", "tag_template_full_1366x768.npy"]
    small, full = (np.load(f) for f in files)
    assert np.array_equal(small, full) and full.shape == tuple(said["tag"]["cut"]) == P.tag_template(made)[0].shape
    fr = cv2.imread(str(frames / "00003.png"))[..., ::-1]
    s, cx, cy = P.gray_peaks(np.ascontiguousarray(fr), full, k=1)[0]
    assert s > 0.99 and (cx, cy) == (centres[3][0] + dx, centres[3][1] + dy)
    hit = P.Plate(made).near(np.ascontiguousarray(fr), (cx + 2, cy - 1))
    assert (hit.x, hit.y) == (cx, cy) and hit.feet[0] == centres[3][0] + m.axis()   # her feet: the plate's axis


def test_her_cues_writes_to_a_folder_given_or_her_lab_folder_when_asked(tmp_path, capsys):
    with pytest.raises(SystemExit):
        her_cues.main([str(tmp_path)])                    # neither --out nor --lab: refused
    assert "--out" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        her_cues.main([str(tmp_path), "--out", str(tmp_path / "o"), "--lab"])
