"""The masked colour match in C (gamelab/native.c's mm_*, through perception.match) against OpenCV's: on the windows the
monster search, its re-finds and the drops' search match on frames of the game with the first map's pictures drawn as
the game draws them (analysis.pasted), at the game's own 1366x768 and stretched to 1920x1080 (half size): the best place
the same, the score within 3e-5 of OpenCV's wherever either scores 0.5 or more (more apart only in windows about flat,
where OpenCV's float32 sums lose digits); flat and constant pictures, ties, a template too big for the int32 sums at
once, AVX2 against plain C; the re-finds' answers the same with the C off, and the matches of a re-find handed over in
one call only as far as they're looked at."""

import dataclasses

import cv2
import numpy as np
import pytest

from gamelab import native
from gamelab.analysis import pasted
from gamelab.perception import loot, match, sprites
from gamelab.screen import NATIVE, STRETCHED

from test_native_search import first_map, needs_game_data  # noqa: F401 (a fixture)
from test_pasted import FRAMES, frame

pytestmark = pytest.mark.skipif(native.lib() is None, reason=f"gamelab's C pieces aren't built: {native.why}")


def opencv(image, template, mask):
    return np.nan_to_num(cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask), nan=-1.0, posinf=-1.0,
                         neginf=-1.0)


class Recorder:
    """Every masked match the C is asked for (native.masked_match, native.best_matches), with the pictures each
    prepared template was made from (native.prepare_masked)."""

    def __init__(self, monkeypatch):
        self.pictures, self.calls = {}, []
        prepare, one, best = native.prepare_masked, native.masked_match, native.best_matches

        def prepare_masked(template, mask):
            ready = prepare(template, mask)
            if ready is not None:
                self.pictures[id(ready)] = (ready, template.copy(), mask.copy())
            return ready

        def masked_match(image, ready):
            out = one(image, ready)
            self.calls.append(("all", image.copy(), [ready], out))
            return out

        def best_matches(image, ready, stop=float("inf")):
            out = best(image, ready, stop)
            self.calls.append(("best", image.copy(), list(ready), out))
            return out

        monkeypatch.setattr(native, "prepare_masked", prepare_masked)
        monkeypatch.setattr(native, "masked_match", masked_match)
        monkeypatch.setattr(native, "best_matches", best_matches)
        match._PREPARED.clear()  # (made ready again, through the recorder)

    def compare(self) -> dict:
        """Each call's answers against OpenCV's: the largest difference anywhere and where either scores 0.5 or more
        (the thresholds' range), by channels; and how many best places differ."""
        worst, near, places, n = {1: 0.0, 3: 0.0}, {1: 0.0, 3: 0.0}, 0, 0
        for kind, image, ready, out in self.calls:
            assert out is not None
            for r, found in zip(ready, out if kind == "best" else [out]):
                _, template, mask = self.pictures[id(r)]
                ref = opencv(image, template, mask)
                cn = 1 if image.ndim == 2 else 3
                n += 1
                _, best, _, at = cv2.minMaxLoc(ref)
                if kind == "all":
                    d = np.abs(found.astype(np.float64) - ref)
                    worst[cn] = max(worst[cn], float(d.max()))
                    high = (found >= 0.5) | (ref >= 0.5)
                    near[cn] = max(near[cn], float(d[high].max()) if high.any() else 0.0)
                    places += cv2.minMaxLoc(found)[3] != at
                else:
                    score, x, y = found
                    worst[cn] = max(worst[cn], abs(score - best))
                    if max(score, best) >= 0.5:
                        near[cn] = max(near[cn], abs(score - best))
                    places += (x, y) != at
        return {"worst": worst, "near": near, "places": places, "matches": n}


def tallies(result: dict) -> dict:
    return {name: {step: dataclasses.replace(r, times=[]) for step, r in steps.items()}
            for name, steps in result.items()}


def test_the_c_scores_what_a_search_and_its_refinds_match_as_opencv_does(first_map, monkeypatch):  # noqa: F811
    needs_game_data()
    rec = Recorder(monkeypatch)
    native_det = sprites.detector_for(first_map, NATIVE)
    stretched_det = sprites.detector_for(first_map, STRETCHED)
    backgrounds = [frame(n) for n in FRAMES]
    result = pasted.refinds(backgrounds, {"native": (native_det, NATIVE), "stretched": (stretched_det, STRETCHED)},
                            sprites.folder_for(1.0), (140, 75), seed=4, kinds={"pig"})
    got = rec.compare()
    # (9.5e-7 on a crowd's frames: native.c. Here up to 1.1e-5, in windows a few levels from flat, where OpenCV's
    # float32 sums lose digits: D:/MapleHelper-scratch/pconst/pnative/diff_where.py)
    assert got["matches"] > 3000 and got["places"] == 0 and got["near"][3] < 3e-5 and got["worst"][3] < 3e-5, got
    assert result["native"]["unchanged"].found == result["native"]["unchanged"].n > 10


def test_the_c_scores_the_drops_strips_on_brightness_as_opencv_does(first_map, monkeypatch):  # noqa: F811
    needs_game_data()
    rec = Recorder(monkeypatch)
    backgrounds = [frame(n) for n in FRAMES]
    result = pasted.drops(backgrounds, {"native": (loot.LootDetector(geo=NATIVE), NATIVE)}, loot.folder_for(1.0), 1, 1)
    got = rec.compare()
    # (up to 1.0e-4 at scores of 0.24-0.32, where a strip is about flat: its spread 3 grey levels)
    assert got["matches"] > 500 and got["places"] == 0 and got["near"][1] < 3e-5 and got["worst"][1] < 3e-4, got
    assert all(t.found == t.drawn for t in result["tally"]["native"].values())


def test_with_the_c_off_the_refinds_come_out_the_same(first_map, monkeypatch):  # noqa: F811
    needs_game_data()
    backgrounds = [frame(n) for n in FRAMES]

    def run():
        det = sprites.detector_for(first_map, NATIVE)
        return tallies(pasted.refinds(backgrounds, {"native": (det, NATIVE)}, sprites.folder_for(1.0), (140, 75),
                                      seed=5, kinds={"orange-mushroom"}))

    with_c = run()
    monkeypatch.setattr(native, "USE", False)
    assert native.lib() is None and match.prepared(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)) is None
    assert run() == with_c


# --- the score's edges ----------------------------------------------------------------------------------------------

def scene_and_picture(seed: int = 1, size=(60, 70), tsize=(30, 26), cn: int = 3):
    rng = np.random.default_rng(seed)
    shape = (size[0], size[1], cn) if cn == 3 else size
    image = rng.integers(0, 256, shape, np.uint8)
    template = image[11:11 + tsize[0], 17:17 + tsize[1]].copy()
    mask = (rng.random(tsize) > 0.3).astype(np.uint8)
    return image, template, np.dstack([mask] * 3) if cn == 3 else mask


@pytest.mark.parametrize("cn", [3, 1])
def test_a_picture_found_where_it_was_cut_scores_1_there_and_opencvs_scores_elsewhere(cn):
    image, template, mask = scene_and_picture(cn=cn)
    out, ref = match.masked_match(image, template, mask), opencv(image, template, mask)
    assert out.dtype == np.float32 and out.shape == ref.shape and out[11, 17] == 1.0
    assert np.abs(out.astype(np.float64) - ref).max() < 3e-5
    assert match.best_match(image, template, mask) == (1.0, (17, 11))


def test_a_flat_window_and_a_flat_picture_score_minus_1_as_opencvs_nan_did():
    image, template, mask = scene_and_picture()
    image[:40, :45] = (90, 120, 30)  # every channel flat: its variance 0
    out, ref = match.masked_match(image, template, mask), opencv(image, template, mask)
    assert (out[:40 - 30 + 1, :45 - 26 + 1] == -1).all() and (ref[:11, :20] == -1).all()
    flat = np.full_like(template, 77)
    assert (match.masked_match(image, flat, mask) == -1).all() and (opencv(image, flat, mask) == -1).all()
    none = np.zeros_like(mask)  # no pixel matched
    assert (match.masked_match(image, template, none) == -1).all()


def test_one_flat_channel_isnt_a_flat_window():
    image, template, mask = scene_and_picture(2)
    image[..., 1] = 140  # green flat everywhere, red and blue not
    out, ref = match.masked_match(image, template, mask), opencv(image, template, mask)
    assert (out > -1).all() and np.abs(out.astype(np.float64) - ref).max() < 3e-5


def test_of_equal_scores_the_first_row_by_row_is_the_best_as_minmaxloc_has_it():
    rng = np.random.default_rng(5)
    template = rng.integers(0, 256, (12, 14, 3), np.uint8)
    mask = np.ones((12, 14, 3), np.uint8)
    image = np.zeros((40, 60, 3), np.uint8)
    for x, y in ((40, 5), (8, 20), (30, 20)):  # the same picture three times: 1.0 at each
        image[y:y + 12, x:x + 14] = template
    out = match.masked_match(image, template, mask)
    assert (out[5, 40], out[20, 8], out[20, 30]) == (1.0, 1.0, 1.0)
    assert cv2.minMaxLoc(out)[3] == (40, 5) == match.best_match(image, template, mask)[1]
    flat = np.zeros_like(image)  # all -1: the first place
    _, top, _, at = cv2.minMaxLoc(opencv(flat, template, mask))
    assert match.best_match(flat, template, mask) == (-1.0, (0, 0)) == (top, at)


def test_a_template_too_big_for_the_int32_sums_at_once_is_summed_in_groups():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (150, 150, 3), np.uint8)
    template = image[10:130, 12:132].copy()  # 120 x 120 x 3: 23.4 million a row at most, 91 rows to an int32
    template[::7] = 255
    mask = np.dstack([(rng.random((120, 120)) > 0.05).astype(np.uint8)] * 3)
    ready = match.prepared(template, mask)
    assert ready is not None
    out, ref = match.masked_match(image, template, mask), opencv(image, template, mask)
    assert np.abs(out.astype(np.float64) - ref).max() < 1e-4 and cv2.minMaxLoc(out)[3] == cv2.minMaxLoc(ref)[3]


def test_avx2_and_plain_c_give_the_same_scores():
    f = native.lib()
    on = f.mm_simd(-1)
    try:
        for cn in (3, 1):
            image, template, mask = scene_and_picture(3, (70, 90), (33, 41), cn)
            f.mm_simd(1)
            a = match.masked_match(image, template, mask)
            f.mm_simd(0)
            b = match.masked_match(image, template, mask)
            assert np.array_equal(a, b)
    finally:
        f.mm_simd(on)


def test_a_window_a_view_of_the_frame_is_read_as_it_is_and_one_the_c_cant_read_goes_to_opencv():
    image, template, mask = scene_and_picture(4, (80, 90))
    view = image[5:70, 3:80]  # rows 270 bytes apart
    assert np.array_equal(match.masked_match(view, template, mask),
                          match.masked_match(np.ascontiguousarray(view), template, mask))
    flipped = image[:, ::-1]  # (negative strides: OpenCV's)
    assert native.masked_match(flipped, match.prepared(template, mask)) is None
    assert np.array_equal(match.masked_match(flipped, template, mask), opencv(flipped, template, mask))
    floats = template.astype(np.float32)  # not uint8: OpenCV's
    assert match.prepared(floats, mask) is None


def test_a_picture_is_made_ready_once_while_its_arrays_are_kept():
    image, template, mask = scene_and_picture(6)
    first = match.prepared(template, mask)
    assert match.prepared(template, mask) is first and match.prepared(template.copy(), mask) is not first
    key = (id(template), id(mask))
    del template
    assert key not in match._PREPARED  # gone with its array


def test_a_refinds_pictures_are_matched_in_one_call_as_far_as_theyre_looked_at(monkeypatch):
    image, template, mask = scene_and_picture(8)
    other = np.ascontiguousarray(template[:, ::-1])
    calls = []
    best = native.best_matches
    monkeypatch.setattr(native, "best_matches", lambda im, ready, stop=float("inf"): calls.append(len(ready)) or
                        best(im, ready, stop))
    pictures = [(other, mask), (template, mask), (other, mask), (template, mask)]
    each = [match.best_match(image, t, m) for t, m in pictures]
    calls.clear()
    found = match.best_matches(image, pictures, stop=0.9)
    assert [next(found), next(found)] == each[:2] and calls == [4]  # in C up to the first scoring 0.9 (1.0)
    assert list(found) == each[2:] and calls == [4, 2]               # the rest only if the caller goes on
    calls.clear()
    assert list(match.best_matches(image, pictures)) == each and calls == [4]
