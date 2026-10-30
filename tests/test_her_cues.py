"""Her cues' models made at calibration (gamelab/localization/her_cues.py): her feet's offset down from her plate's
centre measured where she stands on a platform's line (16 for her orange-ribbon plate), none taken where too few frames
agree; a folder's frames in its index's order."""

import json
from dataclasses import dataclass

import cv2
import numpy as np

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
