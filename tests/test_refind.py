"""A monster followed, re-found (MobDetector.refind): first only as far around where it should be as it can have got
since it was last seen (`near`, mobs.near_pad), the REFIND_PAD window only where that finds no picture at REFIND_GOOD;
a window at the same place, looked at with the same pictures, whose pixels are the very same as at the last re-find,
gives the last answer without matching again (`memo`), its kind's whole set's too. Made-up game pictures (a pig and its
next frame, test_pasted's) drawn on a frame of the game at its own 1366x768."""

import json

import numpy as np
import pytest

from gamelab import native
from gamelab.analysis import pasted
from gamelab.perception.mobs import INDEX, NEAR_MARGIN, REFIND_GOOD, REFIND_PAD, MobDetector, near_pad
from gamelab.screen import HUD_TOP, NATIVE

from test_pasted import frame, game_folder, pig

pytestmark = pytest.mark.skipif(native.lib() is None, reason=f"gamelab's C pieces aren't built: {native.why}")


def pictures() -> dict[str, np.ndarray]:
    right = pig().copy()
    nxt = right.copy()
    nxt[..., :3] = np.roll(right[..., :3], 4, axis=1)  # its next frame: its colours 4 px across in the same outline
    return {"pig_move0": right, "pig_move1": nxt}


@pytest.fixture(scope="module")
def detector(tmp_path_factory):
    folder = game_folder(tmp_path_factory.mktemp("pigs"), pictures())
    index = json.loads((folder / INDEX).read_text(encoding="utf-8"))
    index["pig_move1.png"]["frame"] = "move/1"
    (folder / INDEX).write_text(json.dumps(index), encoding="utf-8")
    return MobDetector(folder, frame_scale=NATIVE.frame_scale, threads=1)


def scene(name: str = "pig_move0", left: int = 600, top: int = 300) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A frame with the picture `name` drawn (facing left, as drawn) with its top-left at (left, top), and its box."""
    rgba = pictures()[name]
    out = frame()
    pasted.paste(out, rgba, left, top)
    return out, (left, top, left + rgba.shape[1], top + rgba.shape[0])


@pytest.fixture
def windows(monkeypatch):
    """The size (w, h) of every window the C is asked to match, in turn."""
    seen = []
    best = native.best_matches

    def spy(image, ready, stop=float("inf")):
        seen.append((image.shape[1], image.shape[0]))
        return best(image, ready, stop)
    monkeypatch.setattr(native, "best_matches", spy)
    return seen


BOUNDS = (0, 0, 1366, int(768 * HUD_TOP))


def test_how_far_around_it_is_looked_for_first():
    assert near_pad(120.0, 1 / 30) == NEAR_MARGIN + 8   # 30 frames a second: a pig turned at once is 8 px off
    assert near_pad(120.0, 0.1) == NEAR_MARGIN + 24     # 10 a second: more than REFIND_PAD's 23 at 1366x768 ...
    assert near_pad(120.0, -0.01) == NEAR_MARGIN        # (a late frame's time before its last sighting)


def test_near_first_finds_what_the_whole_window_finds_and_matches_less(detector, windows):
    full_pad = round(REFIND_PAD * detector.px)
    frame_, box = scene()
    for dx in (0, 5):  # where it should be, and 5 px off it (within the near window's 4 + 8)
        moved = (box[0] - dx, box[1], box[2] - dx, box[3])
        windows.clear()
        whole = detector.refind(frame_, moved, "pig", "pig_move0", -1, BOUNDS)
        assert windows == [(box[2] - box[0] + 2 * full_pad, box[3] - box[1] + 2 * full_pad)]
        windows.clear()
        near = detector.refind(frame_, moved, "pig", "pig_move0", -1, BOUNDS, near=NEAR_MARGIN + 8)
        assert near == whole and near.score >= REFIND_GOOD
        assert windows == [(box[2] - box[0] + 2 * (NEAR_MARGIN + 8), box[3] - box[1] + 2 * (NEAR_MARGIN + 8))]


def test_where_near_finds_nothing_good_the_whole_window_is_looked_at(detector, windows):
    frame_, box = scene()
    moved = (box[0] - 18, box[1], box[2] - 18, box[3])  # 18 px off: outside the near window, inside REFIND_PAD's
    whole = detector.refind(frame_, moved, "pig", "pig_move0", -1, BOUNDS)
    windows.clear()
    near = detector.refind(frame_, moved, "pig", "pig_move0", -1, BOUNDS, near=NEAR_MARGIN + 8)
    assert near == whole and near is not None and (near.x - near.w // 2, near.y - near.h // 2) == box[:2]
    assert len(windows) >= 2 and windows[0][0] < windows[-1][0]  # near first, then the whole window
    frame_, box = scene("pig_move1")  # its next frame drawn there: near, after its last picture, its next one
    whole = detector.refind(frame_, box, "pig", "pig_move0", -1, BOUNDS)
    windows.clear()
    near = detector.refind(frame_, box, "pig", "pig_move0", -1, BOUNDS, near=NEAR_MARGIN + 8)
    assert near.template == "pig_move1" and near == whole
    assert len(windows) == 2 and windows[0] == windows[1]  # (both in the near window: its last, then its next)


def test_a_window_the_very_same_as_last_time_gives_the_last_answer(detector, windows):
    frame_, box = scene()
    memo = {}
    first = detector.refind(frame_, box, "pig", "pig_move0", -1, BOUNDS, near=NEAR_MARGIN + 8, memo=memo)
    recalled = detector.recalled
    windows.clear()
    again = detector.refind(frame_.copy(), box, "pig", "pig_move0", -1, BOUNDS, near=NEAR_MARGIN + 8, memo=memo)
    assert again == first and windows == [] and detector.recalled == recalled + 1
    changed = frame_.copy()
    changed[box[1] + 10, box[0] + 10] ^= 1  # one pixel of it
    assert detector.refind(changed, box, "pig", "pig_move0", -1, BOUNDS, near=NEAR_MARGIN + 8, memo=memo) is not None
    assert len(windows) == 1
    windows.clear()  # moved a px: another window (its pixels may be the same, not its places)
    shifted = (box[0] + 1, box[1], box[2] + 1, box[3])
    assert detector.refind(changed, shifted, "pig", "pig_move0", -1, BOUNDS, near=NEAR_MARGIN + 8, memo=memo) == \
        detector.refind(changed, shifted, "pig", "pig_move0", -1, BOUNDS, near=NEAR_MARGIN + 8)
    assert len(windows) == 2
    windows.clear()  # looked at with another picture (its last sighting's): matched again
    detector.refind(changed, shifted, "pig", "pig_move1", -1, BOUNDS, near=NEAR_MARGIN + 8, memo=memo)
    assert windows


def test_the_kinds_whole_set_is_recalled_too_and_still_counted(detector, windows):
    frame_ = frame()
    box = (600, 300, 700, 380)  # nothing there
    memo = {}
    assert detector.refind(frame_, box, "pig", "pig_move0", -1, BOUNDS, memo=memo) is None
    fallbacks, recalled, n = detector.fallbacks, detector.recalled, len(windows)
    assert n >= 1
    assert detector.refind(frame_, box, "pig", "pig_move0", -1, BOUNDS, memo=memo) is None
    assert len(windows) == n and detector.fallbacks == fallbacks + 1 and detector.recalled == recalled + 2
    memo.clear()
    assert detector.refind(frame_, box, "pig", "pig_move0", -1, BOUNDS, memo=memo) is None and len(windows) > n


def test_found_outside_the_near_window_the_next_goes_to_the_whole_window_at_once(detector, windows):
    full_pad, near = round(REFIND_PAD * detector.px), NEAR_MARGIN + 8
    frame_, box = scene()
    off = (box[0] - 18, box[1], box[2] - 18, box[3])  # 18 px off: outside the near window, inside REFIND_PAD's
    memo = {}
    detector.refind(frame_, off, "pig", "pig_move0", -1, BOUNDS, near=near, memo=memo)
    assert windows[0][0] < windows[-1][0] and memo["far"]  # near (its last picture, its next), then the whole
    windows.clear()
    again = frame_.copy()
    again[box[1] + 10, box[0] + 10] ^= 1  # (not the very same window: matched again)
    assert detector.refind(again, off, "pig", "pig_move0", -1, BOUNDS, near=near, memo=memo) is not None
    assert windows == [(box[2] - box[0] + 2 * full_pad, box[3] - box[1] + 2 * full_pad)]  # the whole window alone
    windows.clear()  # where it should be again: the whole window finds it inside the near one, and the next goes near
    assert detector.refind(frame_, box, "pig", "pig_move0", -1, BOUNDS, near=near, memo=memo) is not None
    assert len(windows) == 1 and not memo["far"]
    windows.clear()
    assert detector.refind(again, box, "pig", "pig_move0", -1, BOUNDS, near=near, memo=memo) is not None
    assert windows == [(box[2] - box[0] + 2 * near, box[3] - box[1] + 2 * near)]
