"""The monster search's step 1 in C (gamelab/search.c, through perception.mobs._InC and native.search_lib): the band's
shrink the same bytes as cv2.resize; the colour tests mobs.PALETTES' and the learned gates' on every colour; each
silhouette's scores the exact ones (N / sqrt(A B) from integer sums, OpenCV's float32 within the noise bound of them,
mobs.NOISE_K) and the peaks master's rule on them, wherever the cover test keeps a window; where the bounds leave
master's peaks or their order to OpenCV's noise, those shapes worked out as master does (mobs._InC._master), so a unit's
spots come in master's order; the places mobs._places'; and a search's answers (every field, to the type) the same with
it as without it (mobs.SEARCH_C), with the fine gate's and step 2's matches in rf_run calls, on the made-up kinds, the
first map's cut-outs at 1920x1080 half size, Tree Dungeon's sprites, and the saved inputs where OpenCV's noise decided
(tests/fixtures/search)."""

import dataclasses
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gamelab import native
from gamelab.analysis import pasted
from gamelab.perception import gates, mobs
from gamelab.perception.mobs import COARSE, SILHOUETTE_THRESHOLD, MobDetector
from gamelab.screen import NATIVE, STRETCHED

import testdata
from test_gates import body, detector as learned_detector
from test_pasted import FRAMES, frame, game_folder, pig
from test_perception import typed, with_monsters

pytestmark = pytest.mark.skipif(native.search_lib() is None or native.lib() is None,
                                reason=f"the search's C isn't here: {native.search_why or native.why}")

SEARCH = NATIVE.frame_scale / mobs.SEARCH_SCALE  # a 1366x768 look frame's px per step 1's
NOISY = Path(__file__).parent / "fixtures" / "search"  # where OpenCV's noise decided (its README)


def both(monkeypatch, search) -> tuple:
    """search() with the C (mobs.SEARCH_C) and without: each answer's fields, to the type."""
    out = []
    for on in (True, False):
        monkeypatch.setattr(mobs, "SEARCH_C", on)
        out.append(typed(dataclasses.astuple(m) for m in search()))
    monkeypatch.setattr(mobs, "SEARCH_C", True)
    return tuple(out)


# --- the shrink --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("h, w", [(193, 1366), (150, 1366), (101, 1366), (77, 961), (90, 100), (17, 23), (700, 1366)])
def test_the_shrink_is_opencvs_to_the_byte(h, w):
    rng = np.random.default_rng(h * 7 + w)
    big = rng.integers(0, 256, (h + 9, w + 31, 3), dtype=np.uint8)
    smooth = cv2.GaussianBlur(big, (0, 0), 3)
    for picture in (big[:h, :w], smooth[5:5 + h, 20:20 + w], np.ascontiguousarray(smooth[:h, :w])):  # views too
        size = (max(1, round(w / SEARCH)), max(1, round(h / SEARCH)))
        got = native.shrink_area(picture, *size)
        assert got is not None and np.array_equal(got, cv2.resize(picture, size, interpolation=cv2.INTER_AREA))
        assert np.array_equal(mobs.step1_frame(picture, SEARCH), got)


def test_the_shrink_declines_what_opencv_does_otherwise_and_step_1_then_has_opencv_do_it():
    picture = np.random.default_rng(3).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    for size in ((30, 20), (60, 20), (30, 40), (80, 50), (34, 22)):  # 2x (resizeAreaFast), none one way, larger, 1.76x
        assert native.shrink_area(picture, *size) is None
    assert native.shrink_area(picture[:, ::-1], 42, 28) is None  # (a flipped view: not pixels side by side)
    assert native.shrink_area(picture.astype(np.float32), 42, 28) is None
    assert np.array_equal(mobs.step1_frame(picture[:2, :2], SEARCH), cv2.resize(picture[:2, :2], (1, 1),
                                                                                  interpolation=cv2.INTER_AREA))


# --- the colour tests -------------------------------------------------------------------------------------------

def colours_in_c(rgb: np.ndarray, units: list[tuple[int, int]], lut: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """sr_colours: each unit's (test, bits) I4 picture and count."""
    h, w = rgb.shape[:2]
    cu = (native.SrUnit * len(units))(*[native.SrUnit(t, b, 0.0, 0, 0, 0, 0) for t, b in units])
    out = np.zeros((len(units), h // 2, w // 2), np.uint8)
    count = (ctypes_ll * len(units))()
    assert native.search_lib().sr_colours(rgb.ctypes.data, h, w, rgb.strides[0], lut.ctypes.data, cu, len(units),
                                          out.ctypes.data, count) == 0
    return out, list(count)


ctypes_ll = __import__("ctypes").c_longlong


def cells(mask: np.ndarray) -> np.ndarray:
    """Each 2x2 cell's count of a mask's pixels (I4)."""
    m = mask.astype(np.uint8)
    return m[0::2, 0::2] + m[0::2, 1::2] + m[1::2, 0::2] + m[1::2, 1::2]


def test_the_hand_made_colour_tests_are_mobs_own_on_every_colour():
    tests = list(mobs._HAND_TESTS.items())
    assert {k for k, _ in tests} == set(mobs.PALETTES)  # (a kind with a test the C doesn't know is searched as before)
    lut = np.zeros(1 << 15, np.uint32)
    for strip in range(16):  # every colour once, top-left of a cell of its own, the other three black (no kind's)
        v = np.arange(strip << 20, (strip + 1) << 20, dtype=np.uint32).reshape(1024, 1024)
        rgb = np.zeros((2048, 2048, 3), np.uint8)
        rgb[0::2, 0::2] = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], -1).astype(np.uint8)
        got, count = colours_in_c(rgb, [(code, 0) for _, code in tests], lut)
        channels = mobs.split_channels(rgb)
        for (kind, _), g, c in zip(tests, got, count):
            want = cells(mobs.PALETTES[kind](*channels))
            assert np.array_equal(g, want) and c == int(want.sum()), (kind, strip)


def test_the_learned_gates_lookup_is_gates_own_and_a_ragged_row_end_too():
    rng = np.random.default_rng(11)
    luts = {k: rng.random(1 << 15) < p for k, p in (("a", 0.3), ("b", 0.05), ("c", 0.6))}
    g = gates.Gates(luts)
    lut = np.ascontiguousarray(g.lut.astype(np.uint32))
    units = [(0, g.bits("a")), (0, g.bits(("b", "c"))), (1, 0), (0, g.bits(("a", "b", "c")))]
    for h, w in ((64, 2048), (38, 2042), (6, 14)):  # (rows whose last pixels aren't 16 at a time)
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        got, count = colours_in_c(rgb, units, lut)
        r, gg, b = mobs.split_channels(rgb)
        for (test, bits), have, c in zip(units, got, count):
            kinds = [k for k in luts if int(g.bit[k]) & bits]
            want = cells(g.test(tuple(kinds), r, gg, b) if test == 0 else mobs._slime(r, gg, b))
            assert np.array_equal(have, want) and c == int(want.sum())


# --- the silhouettes' scores and the peaks --------------------------------------------------------------------------

def exact_scores(I4: np.ndarray, tables: "mobs._Tables", silhouette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every window's score as search.c works it out (N / sqrt(A B) from the exact integer sums, rounded to float32), 0
    where the cover test drops it or its colour is flat: numpy's, window by window in int64; and its noise bound
    (NOISE_K n sqrt(S_II S_TT) / sqrt(A B) + NOISE_ABS)."""
    th, tw = silhouette.shape
    n = th * tw
    ti = (silhouette.astype(np.float64) * 2.0 ** tables.shift).astype(np.int64)
    a = I4.astype(np.int64)
    rh, rw = a.shape[0] - th + 1, a.shape[1] - tw + 1
    si, sii, sit = (np.zeros((rh, rw), np.int64) for _ in range(3))
    for r in range(th):
        for c in range(tw):
            win = a[r:r + rh, c:c + tw - tw + rw]
            si += win
            sii += win * win
            sit += win * ti[r, c]
    N = n * sit - si * tables.st
    B = n * sii - si * si
    keep = (si >= tables.kmin) & (B > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.sqrt(tables.A * B.astype(np.float64))
        v = (N.astype(np.float64) / root).astype(np.float32)
        e = (mobs.NOISE_K * n * np.sqrt(sii.astype(np.float64)) * tables.rtt / root + mobs.NOISE_ABS).astype(
            np.float32)
    return np.where(keep, v, np.float32(0)), np.where(keep, e, np.float32(0))


def peaks_of(res: np.ndarray, thr: float) -> set[tuple[int, int, float]]:
    """Master's rule (native.c's silhouette_peaks): at least thr, no less than any of the 8 neighbours."""
    padded = np.pad(res, 1, constant_values=-np.inf)
    most = np.max([padded[1 + dy:1 + dy + res.shape[0], 1 + dx:1 + dx + res.shape[1]]
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
    ys, xs = np.nonzero((res >= np.float32(thr)) & (res >= most))
    return {(int(y), int(x), float(res[y, x])) for y, x in zip(ys, xs)}


def blobs(h: int, w: int, colour, seed: int) -> np.ndarray:
    """Step 1's frame of grey noise with blobs, bars and a solid block of `colour` (flat windows), some ragged."""
    rng = np.random.default_rng(seed)
    out = rng.integers(60, 120, (h, w, 3), dtype=np.uint8)
    for _ in range(25):
        cy, cx, ry, rx = rng.integers(0, h), rng.integers(0, w), rng.integers(3, 30), rng.integers(3, 40)
        yy, xx = np.ogrid[:h, :w]
        inside = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        inside &= rng.random((h, w)) < rng.choice([1.0, 0.8, 0.5])
        out[inside] = colour
    out[h // 3:h // 3 + 40, w // 2:w // 2 + 70] = colour  # (every window inside it flat)
    return out


def unit_order(units: list) -> list:
    """Each unit's spots as `_places` takes them (sorted, best first): their places and shapes, to the type."""
    return [(u[0], typed(s[1:] for s in sorted((s for f in u[3] for s in f), reverse=True))) for u in units]


def test_the_scores_are_the_exact_ones_and_the_peaks_master_rule_on_them(tmp_path):
    det = MobDetector(game_folder(tmp_path, {"pig_move0": pig()}), threads=1, frame_scale=NATIVE.frame_scale)
    c = det._step1_in_c(None)
    assert c is not None and c.n == 2  # (the picture, both ways)
    pig_colour = (230, 150, 120)
    compared, scored, opencv_peaks, same, flagged = 0, 0, 0, 0, 0
    for seed in range(6):
        small = blobs(136, 400, pig_colour, seed)
        out_n, flags, ys, xs, vs, es, i4 = c.peaks(small)
        colour = mobs.palette("pig", small)
        I4 = cells(colour)
        assert np.array_equal(i4[0], I4)  # (the unit's I4, for a shape worked out as master does)
        coarse = cv2.resize(colour.astype(np.float32), (200, 68), interpolation=cv2.INTER_AREA)
        assert np.array_equal(coarse * 4, I4.astype(np.float32))  # (the coarse picture is I4 / 4 to the bit)
        at = 0
        for k, (_, t) in enumerate(c.units[0][1]):
            at, here = at + out_n[k], slice(at, at + out_n[k])
            got = {(int(y), int(x), float(v)) for y, x, v in zip(ys[here], xs[here], vs[here])}
            exact, bound = exact_scores(I4, c.tables[k], t.silhouette)
            assert got == peaks_of(exact, SILHOUETTE_THRESHOLD), (seed, k)  # bit for bit
            assert {(int(y), int(x)): float(e) for y, x, e in zip(ys[here], xs[here], es[here])} == pytest.approx(
                {(y, x): float(bound[y, x]) for y, x, _ in got}, rel=1e-6)
            res = cv2.matchTemplate(coarse, t.silhouette, cv2.TM_CCOEFF_NORMED)
            kept = exact != 0
            assert np.all(np.abs(res[kept].astype(np.float64) - exact[kept]) <= bound[kept])  # OpenCV's float32 noise
            sums = cv2.integral(coarse, sdepth=cv2.CV_64F)
            th, tw = t.silhouette.shape
            window = sums[th:, tw:] - sums[:-th, tw:] - sums[th:, :-tw] + sums[:-th, :-tw]
            passes = window / t.silhouette.size >= mobs.MIN_COVER * float(t.silhouette.mean())
            assert np.all(res[passes & ~kept] == 0)  # OpenCV's flat windows score 0 as well
            master = mobs._spots(coarse, mobs.Cover(coarse), t, k)
            opencv_peaks += len(master)
            here_same = {(y, x) for y, x, _ in got} == {
                ((cy - COARSE * th // 2) // COARSE, (cx - COARSE * tw // 2) // COARSE) for _, cx, cy, _ in master}
            assert here_same or flags[k], (seed, k)  # (OpenCV's noise can decide master's peaks only where flagged)
            same += here_same
            flagged += bool(flags[k])
            compared += 1
            scored += len(got)
        # The units' spots as step 1 hands them over: master's, in master's order (the places are made from that)
        assert unit_order(c.spots(small)) == unit_order(det._step1(small, mobs._serial)), seed
    assert scored > 30 and opencv_peaks > 30 and same >= compared - 2 and flagged >= compared - same


def test_a_band_past_the_peaks_room_goes_on_without_the_c_and_leaves_it_clean(tmp_path, monkeypatch):
    det = MobDetector(game_folder(tmp_path, {"pig_move0": pig()}), threads=1, frame_scale=NATIVE.frame_scale)
    c = det._step1_in_c(None)
    small = blobs(136, 400, (230, 150, 120), 2)
    before = c.peaks(small)
    monkeypatch.setattr(mobs, "STEP1_PEAKS", 1)
    assert c.peaks(small) is None and c.places(small, 6) is None  # (past cap: the caller's own way)
    monkeypatch.setattr(mobs, "STEP1_PEAKS", 1 << 15)
    again = c.peaks(small)  # every score plane zeroed again after the overflow
    assert all(np.array_equal(a, b) for a, b in zip(before, again))
    scene = frame()
    pasted.paste(scene, pig(), 600, 300)
    monkeypatch.setattr(mobs, "STEP1_PEAKS", 1)
    assert both(monkeypatch, lambda: det.detect_wide(scene, (0, 700)))[0]  # (found, as without the C)
    a, b = both(monkeypatch, lambda: det.detect_wide(scene, (0, 700)))
    assert a == b


def test_tied_scores_are_left_to_opencvs_noise_as_master_leaves_them(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    stripes = np.zeros((48, 64, 4), np.uint8)  # a kind whose picture is stripes of its colour, the same along x
    stripes[..., 3] = 255
    stripes[..., :3] = 60
    for r in range(0, 48, 12):
        stripes[r:r + 6, :, :3] = (230, 150, 120)
    det = MobDetector(game_folder(tmp_path / "a", {"pig_move0": stripes}), threads=1, frame_scale=NATIVE.frame_scale)
    c = det._step1_in_c(None)
    small = np.full((136, 400, 3), 60, np.uint8)
    wide = cv2.resize(np.tile(stripes[..., :3], (2, 5, 1))[:, :300], None, fx=1 / SEARCH, fy=1 / SEARCH,
                      interpolation=cv2.INTER_NEAREST)
    small[20:20 + wide.shape[0], 40:40 + wide.shape[1]] = wide  # (a stretch of them: every window inside ties)
    out_n, flags, *_ = c.peaks(small)
    assert flags[:2].all() and out_n[:2].min() > 100  # (a run of tied neighbours, every one of them a peak by the rule)
    before = det.redone
    assert unit_order(c.spots(small)) == unit_order(det._step1(small, mobs._serial))  # (OpenCV's noise's peaks)
    assert det.redone > before and det.noise_over == 0
    scene = cv2.resize(small, None, fx=SEARCH, fy=SEARCH, interpolation=cv2.INTER_NEAREST)
    a, b = both(monkeypatch, lambda: det.detect(scene))
    assert a == b
    # The same picture twice on an even background: two spots of one exact score, in the order OpenCV's noise gives
    det = MobDetector(game_folder(tmp_path / "b", {"pig_move0": pig()}), threads=1, frame_scale=NATIVE.frame_scale)
    c = det._step1_in_c(None)
    small = np.full((136, 400, 3), 90, np.uint8)
    picture = cv2.resize(pig(), None, fx=1 / SEARCH, fy=1 / SEARCH, interpolation=cv2.INTER_AREA)
    for left in (20, 220):
        pasted.paste(small, picture, left, 40)
    assert not c.peaks(small)[1][:2].any()
    assert unit_order(c.spots(small)) == unit_order(det._step1(small, mobs._serial))
    assert det.redone == 1 and det.noise_over == 0  # (the shape of the two worked out as master does)


@pytest.mark.parametrize("name, map_id, geo, only", [
    ("thicket_plateau_a.png", 10004021, NATIVE, None), ("thicket_plateau_b.png", 10004021, NATIVE, None),
    ("thicket_window.png", 10004021, NATIVE, ("red-snail",)), ("thicket_half.png", 10004021, STRETCHED, None),
    ("tree_half_order.png", 10002073, STRETCHED, None)])
def test_where_opencvs_noise_decided_the_answer_is_masters(monkeypatch, name, map_id, geo, only):
    from gamedata.paths import COPY
    if not (COPY / "Packs" / "Mob_00000.ms").is_file():
        pytest.skip("needs the game data copy")
    from gamelab.localization import mapdata
    from gamelab.perception import sprites
    det = sprites.detector_for(mapdata.load(map_id), geo)
    assert det._step1_in_c(only) is not None
    scene = np.asarray(Image.open(NOISY / name).convert("RGB"))
    search = (lambda: det.detect(scene)) if only is None else (lambda: det._detect(scene, None, mobs._serial, only))
    a, b = both(monkeypatch, search)
    assert a == b and a
    # As first built (the exact scores' peaks and order taken as they are), the C answered otherwise here
    monkeypatch.setattr(mobs._InC, "_master", lambda self, u, spots, unsure, i4: [[s[:4] for s in f] for f in spots])
    assert typed(dataclasses.astuple(m) for m in search()) != b


def test_silhouettes_the_c_doesnt_take_are_searched_as_before(tmp_path, monkeypatch):
    sil = np.zeros((65, 10), np.float32)
    sil[5:20, 2:8] = 1
    assert mobs._tables(sil) is None  # (taller than 64 cells)
    assert mobs._tables(np.ones((8, 8), np.float32)) is None  # (flat: OpenCV's score is 1 everywhere)
    assert mobs._tables(np.float32(2.0 ** -40) * np.eye(8, dtype=np.float32)) is None  # (as flat to OpenCV: 1e-25)
    assert mobs._tables(np.float32(0.25) * np.eye(8, dtype=np.float32)) is not None
    tall = np.zeros((200, 60, 4), np.uint8)
    tall[:, 10:50] = (*pig()[30, 50, :3], 255)
    tall[::7, :, :3] = 40
    det = MobDetector(game_folder(tmp_path, {"pig_move0": pig(), "pig_move1": tall}), threads=1,
                      frame_scale=NATIVE.frame_scale)
    assert det._step1_in_c(None) is None  # (one of its silhouettes doesn't fit: master's way for the detector)
    scene = frame()
    pasted.paste(scene, pig(), 500, 300)
    a, b = both(monkeypatch, lambda: det.detect_wide(scene, (0, 700)))
    assert a == b and len(a) >= 1


def test_masters_way_has_masters_threads_whenever_it_runs(tmp_path, monkeypatch):
    det = MobDetector(game_folder(tmp_path, {"pig_move0": pig()}), frame_scale=NATIVE.frame_scale)
    assert det._pool._max_workers == mobs.THREADS and det._bands._max_workers == mobs.SEARCH_THREADS
    scene = frame()
    pasted.paste(scene, pig(), 500, 300)
    used = []

    def counted(pool, its_map):
        return lambda fn, items: used.append(pool) or its_map(fn, items)

    for pool in (det._pool, det._bands):
        monkeypatch.setattr(pool, "map", counted(pool, pool.map))
    det.detect_wide(scene, (0, 700))
    assert used == [det._bands]  # (step 1 in C: the bands on SEARCH_THREADS)
    monkeypatch.setattr(mobs, "SEARCH_C", False)  # (turned off with the detector made: master's way, on its THREADS)
    det.detect_wide(scene, (0, 700))
    assert used[1] is det._pool


# --- the places ---------------------------------------------------------------------------------------------------

def test_the_places_are_mobs_places_of_the_sorted_spots():
    rng = np.random.default_rng(5)
    for trial in range(300):
        nu = int(rng.integers(1, 4))
        out_n, ys, xs, vs, sh, sw, ufirst, un = [], [], [], [], [], [], [], []
        for _ in range(nu):
            looked = rng.random() > 0.15  # (a unit not looked for: -1s, no peaks; sr_step1's)
            ufirst.append(len(out_n) if looked else -1)
            un.append(int(rng.integers(1, 4)))
            for _ in range(un[-1]):
                sh.append(int(rng.integers(3, 30)))
                sw.append(int(rng.integers(3, 40)))
                n = int(rng.integers(0, 40)) if looked else -1
                out_n.append(n)
                if n > 0:
                    at = rng.choice(60 * 90, size=n, replace=False)  # (a shape's peaks apart)
                    values = rng.choice([0.35, 0.5, 0.75, 0.9], size=n) if trial % 3 == 0 else rng.random(n)
                    ys += (at // 90).tolist()
                    xs += (at % 90).tolist()
                    vs += np.float32(values).tolist()
        slack = int(rng.choice([4, 6, 12]))
        arrays = [np.array(ys or [0], np.int64), np.array(xs or [0], np.int64), np.array(vs or [0], np.float32)]
        arrays += [np.array(a, np.int32) for a in (out_n, sh, sw, ufirst, un)]
        room = 1 << 16
        out = np.empty(room, np.int32)
        got = native.search_lib().sr_places(*[a.ctypes.data for a in arrays], nu, slack, mobs.MAX_CHECKS,
                                            mobs.FINE_TEMPLATES, out.ctypes.data, room)
        assert got >= 0
        o, at, p = out[:got].tolist(), 0, 0
        for u in range(nu):
            if ufirst[u] < 0:
                continue
            spots = []
            for i in range(un[u]):
                shape = ufirst[u] + i
                for q in range(out_n[shape]):
                    spots.append((vs[at + q], COARSE * xs[at + q] + COARSE * sw[shape] // 2,
                                  COARSE * ys[at + q] + COARSE * sh[shape] // 2, i))
                at += out_n[shape]
            want = mobs._places(sorted(spots, reverse=True), slack)
            count, p = o[p], p + 1
            places = []
            for _ in range(count):
                ne, p = o[p], p + 1
                places.append([tuple(o[p + 3 * e:p + 3 * e + 3]) for e in range(ne)])
                p += 3 * ne
            assert places == [[tuple(e) for e in place] for place in want], trial
        assert p == got


# --- the answers ----------------------------------------------------------------------------------------------------

def test_a_search_finds_what_it_found_without_the_c_at_the_games_own_size(tmp_path, monkeypatch):
    det = MobDetector(game_folder(tmp_path, {"pig_move0": pig()}), threads=3, frame_scale=NATIVE.frame_scale)
    found = 0
    for name in FRAMES:
        scene = frame(name)
        for left, top, flip in ((300, 200, False), (700, 420, True), (1100, 560, False), (150, 600, True)):
            pasted.paste(scene, pig()[:, ::-1] if flip else pig(), left, top)
        for search in (lambda: det.detect_wide(scene, (0, 700)), lambda: det.detect(scene, (150, 400)),
                       lambda: det.detect(scene, (690, 760))):
            a, b = both(monkeypatch, search)
            assert a == b
            found += len(a)
        box = (300, 200, 300 + pig().shape[1], 200 + pig().shape[0])  # a re-find's fallback: its kind's whole set
        a, b = both(monkeypatch, lambda: [m for m in [det._fallback(scene, (280, 180, box[2] + 20, box[3] + 20), "pig",
                                                                    box, (0, 0, 1366, 700), None)] if m])
        assert a == b and a
    assert found >= 10


def test_a_learned_kinds_search_finds_what_it_found_without_the_c(tmp_path, monkeypatch):
    det = learned_detector(tmp_path, {"blob_move0": body(), "other_move0": body(cap=(40, 200, 220))})
    assert det.learned == {"blob", "other"}
    scene = frame()
    for left, top, rgba in ((400, 300, body()), (800, 500, body()[:, ::-1]), (620, 320, body(cap=(40, 200, 220)))):
        pasted.paste(scene, rgba, left, top)
    found = 0
    for search in (lambda: det.detect_wide(scene, (0, 700)), lambda: det.detect(scene, (250, 420))):
        a, b = both(monkeypatch, search)
        assert a == b
        found += len(a)
    assert found >= 3


def test_at_1920x1080_half_size_the_first_maps_monsters_are_found_as_without_the_c(monkeypatch):
    frames = [with_monsters(("pig_0", 100, 420, False), ("slime_4", 400, 300, True), ("snail_8", 700, 460, False),
                            ("pig_2", 800, 200, True), ("snail_10", 300, 470, True)), with_monsters()]
    det, found = MobDetector(), 0
    assert det.search == 1 and det._step1_in_c(None) is not None
    for scene in frames:
        for search in (lambda: det.detect(scene), lambda: det.detect(scene, (250, 401)),
                       lambda: det.detect_wide(scene, (0, 486))):
            a, b = both(monkeypatch, search)
            assert a == b
            found += len(a)
    assert found >= 10


def test_tree_dungeons_sprites_find_what_they_found_without_the_c(monkeypatch):
    from gamedata.paths import COPY
    if not (COPY / "Packs" / "Mob_00000.ms").is_file():
        pytest.skip("needs the game data copy")
    from gamelab.localization import mapdata
    from gamelab.perception import sprites
    _, tree = testdata.labelled(testdata.TREE_DUNGEON)
    det = sprites.detector_for(mapdata.load(10002073), NATIVE)
    assert det._step1_in_c(None) is not None
    found = 0
    # every other one of its 24 frames (all 8 platforms they were saved on), and one of West Rocky Mountain II's
    for path in sorted(tree.values())[::2] + [Path(__file__).parent / "fixtures" / "native" / FRAMES[0]]:
        scene = np.asarray(Image.open(path).convert("RGB"))
        for search in (lambda: det.detect_wide(scene, (0, 700)), lambda: det.detect(scene, (300, 520))):
            a, b = both(monkeypatch, search)
            assert a == b, path.name
            found += len(a)
    assert found >= 20
