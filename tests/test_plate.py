"""Her name plate's exact-colour match (gamelab/perception/plate.py; its counting in C, gamelab/locate.c's lc_plate): the
counts the exact number of the plate's stable pixels the frame has in their colour at each place (a brute-force count
checks them), the candidates and their evidence the same in C as in numpy on every window, her plate found under
what covers part of it, below the HUD's top never, another wearer's plate (her plate item, another name) refused by
her letters' bins, and the model built from frames where her tag is sure. (The prototype's answers on the evaluation
set's frames: README, Her place by her plate and her body.)"""

import numpy as np
import pytest

from gamelab import native
from gamelab.perception import plate as P
from gamelab.perception.plate import Plate, PlateModel

H, W = 768, 1366


def model(seed: int = 0, name_seed: int | None = None) -> PlateModel:
    """A plate like hers: a frame of a few wood colours with a black outline, a translucent interior (unstable), a
    name's letters white inside it (143 px of the interior's middle rows, as hers: `name_seed`'s, by default the
    plate's own)."""
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = P.BOX
    h, w = y1 - y0, x1 - x0
    pic = np.zeros((h, w, 3), np.uint8)
    wood = np.array([[17, 119, 221], [0, 85, 187], [0, 68, 170], [17, 34, 119], [0, 68, 238], [68, 34, 17]], np.uint8)
    pic[:] = wood[rng.integers(len(wood), size=(h, w))]
    pic[0, :] = pic[-1, :] = pic[:, 0] = pic[:, -1] = 0                   # the black outline
    share = np.ones((h, w))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ix0, ix1, iy0, iy1 = P.INTERIOR
    inside = (xx >= ix0) & (xx <= ix1) & (yy >= iy0) & (yy <= iy1)
    pic[inside] = (40, 40, 60)
    share[inside] = 0.5                                                  # translucent: the art shows through
    rows = inside & (yy >= -3) & (yy <= 3)
    name = np.random.default_rng(seed if name_seed is None else name_seed)
    letters = np.zeros(rows.shape, bool)
    letters[tuple(np.array(np.nonzero(rows))[:, name.choice(int(rows.sum()), 143, replace=False)])] = True
    pic[letters] = 255
    share[letters] = 1.0
    return PlateModel(pic, share)


def frame(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))


def draw(fr: np.ndarray, m: PlateModel, cx: int, cy: int, stable_only: bool = True) -> None:
    x0, x1, y0, y1 = m.box
    sl = (slice(cy + y0, cy + y1), slice(cx + x0, cx + x1))
    if stable_only:
        keep = m.share >= P.STABLE_SHARE
        fr[sl][keep] = m.picture[keep]
    else:
        fr[sl] = m.picture


def brute_counts(pl: Plate, fr: np.ndarray, box) -> np.ndarray:
    """The definition: at each place, the stable px whose colour the frame has there."""
    x0, y0, x1, y1 = box
    idx = pl._window_index(fr, x0, y0, x1, y1, P._hud(fr))
    out = np.zeros((y1 - y0 - pl.th + 1, x1 - x0 - pl.tw + 1), np.int32)
    ty, tx = np.nonzero(pl.tidx != P.NONE)
    for y, x, t in zip(ty, tx, pl.tidx[ty, tx]):
        out += idx[y:y + out.shape[0], x:x + out.shape[1]] == t
    return out


needs_c = pytest.mark.skipif(native.locate_lib() is None, reason=f"locate.c isn't here: {native.locate_why}")


@pytest.fixture
def numpy_only(monkeypatch):
    monkeypatch.setattr(P, "USE_C", False)


def both(pl: Plate, fn, monkeypatch):
    """fn(pl) in C and in numpy."""
    monkeypatch.setattr(P, "USE_C", True)
    a = fn(pl)
    monkeypatch.setattr(P, "USE_C", False)
    b = fn(pl)
    monkeypatch.setattr(P, "USE_C", True)
    return a, b


def test_the_model_tells_its_parts_apart():
    pl = Plate(model())
    s = pl.sizes
    assert s["letters"] == 143 and s["background"] == (P.INTERIOR[1] - P.INTERIOR[0] + 1) * 11 - 143
    assert s["frame"] + s["letters"] + 2 * (P.BOX[1] - P.BOX[0]) + 2 * (P.BOX[3] - P.BOX[2]) - 4 == s["stable"]
    assert pl.white != P.NONE and pl.pal[pl.white] == 0xFFFFFF


def test_the_palettes_hash_tells_every_colour_apart_and_no_other_colour_is_one():
    pl = Plate(model(3))
    rgb = np.random.default_rng(4).integers(0, 256, (200, 300, 3), dtype=np.uint8)
    codes = P.pack(rgb)
    idx = pl.index(rgb)
    assert ((idx == P.NONE) == ~np.isin(codes, pl.pal)).all()
    assert (pl.pal[idx[idx != P.NONE]] == codes[idx != P.NONE]).all()
    assert (pl.index(pl.model.picture)[pl.model.share >= P.STABLE_SHARE] != P.NONE).all()


@pytest.mark.parametrize("seed", range(4))
def test_the_counts_are_the_exact_count_in_c_and_in_numpy(seed, monkeypatch):
    m = model(seed)
    pl = Plate(m)
    fr = frame(seed + 10)
    rng = np.random.default_rng(seed)
    cx, cy = int(rng.integers(300, 1000)), int(rng.integers(260, 600))
    draw(fr, m, cx, cy)
    fr[cy - 4:cy + 9, cx - 20:cx + 15] = (0, 85, 187)    # a block of one of its colours over part of it
    box = pl.window(fr.shape, (cx + 3, cy - 2), (70, 45))
    want = brute_counts(pl, fr, box)
    c, n = both(pl, lambda p: p.counts(fr, box), monkeypatch)
    assert np.array_equal(n, want)
    if native.locate_lib() is not None:
        assert np.array_equal(c, want)


@needs_c
@pytest.mark.parametrize("seed", range(12))
def test_the_candidates_are_the_same_in_c_as_in_numpy(seed, monkeypatch):
    m = model(seed % 3)
    pl = Plate(m)
    rng = np.random.default_rng(seed)
    fr = frame(seed)
    cx, cy = int(rng.integers(60, 1300)), int(rng.integers(20, 760))
    draw(fr, m, int(np.clip(cx, 57, W - 59)), int(np.clip(cy, 17, H - 18)))
    for _ in range(int(rng.integers(0, 4))):  # things over it
        x, y = int(rng.integers(0, W - 40)), int(rng.integers(0, H - 30))
        fr[y:y + int(rng.integers(5, 30)), x:x + int(rng.integers(5, 40))] = rng.integers(0, 256, 3)
    pred = (cx + rng.uniform(-30, 30), cy + rng.uniform(-20, 20))
    for reach in ((60, 40), (85, 55), (4, 3), (160, 100)):
        c, n = both(pl, lambda p: p.candidates(fr, pred, reach), monkeypatch)
        assert c == n


def test_her_plate_is_found_where_it_is_partly_covered(monkeypatch):
    m = model(1)
    pl = Plate(m)
    fr = frame(2)
    draw(fr, m, 700, 400)
    fr[388:405, 650:690] = (90, 200, 90)          # a slime over its left half
    for use_c in (True, False):
        monkeypatch.setattr(P, "USE_C", use_c)
        hit = pl.near(fr, (706, 395))
        assert hit is not None and (hit.x, hit.y) == (700, 400)
        assert hit.feet == (700 + P.FEET[0], 400 + P.FEET[1])
        assert hit.candidate.frame < pl.sizes["frame"] and hit.candidate.clean >= P.DEFAULTS["clean_min"]


def test_nothing_of_it_counts_from_the_huds_top_down(monkeypatch):
    m = model(1)
    pl = Plate(m)
    for use_c in (True, False):
        monkeypatch.setattr(P, "USE_C", use_c)
        fr = frame(5)
        draw(fr, m, 600, 700 - P.BOX[2] + 1)            # its whole picture under the HUD's top
        assert pl.candidates(fr, (600, 718), (60, 40)) == []
        fr = frame(5)
        draw(fr, m, 600, 690)                          # half above it: what's above counts
        c = pl.candidates(fr, (600, 690), (60, 40))
        assert c and (c[0].x, c[0].y) == (600, 690)
        assert 0 < c[0].frame < pl.sizes["frame"]


def test_another_wearers_plate_is_refused_by_her_letters_bins(monkeypatch):
    hers = model(1)
    pl = Plate(hers)
    other = model(1, name_seed=99)                    # her plate item, another name
    fr = frame(6)
    draw(fr, other, 500, 300, stable_only=False)
    draw(fr, hers, 900, 300)
    for use_c in (True, False):
        monkeypatch.setattr(P, "USE_C", use_c)
        near_other = pl.candidates(fr, (500, 300), (60, 40))
        assert near_other and (near_other[0].x, near_other[0].y) == (500, 300)
        assert near_other[0].clean < P.DEFAULTS["clean_min"]
        assert pl.near(fr, (500, 300)) is None                     # not her, though her prediction is there
        assert pl.verify(fr, [(0.9, 500, 300), (0.8, 900, 300)]).x == 900   # the gray peaks: hers taken


def test_a_plate_in_the_minimap_is_no_candidate(numpy_only):
    m = model(2)
    pl = Plate(m)
    fr = frame(7)
    draw(fr, m, 120, 100)
    assert pl.candidates(fr, (120, 100), (60, 40)) == []


def test_the_model_is_built_from_sure_frames_and_kept(tmp_path):
    m = model(4)
    frames = []
    for k in range(5):
        fr = frame(20 + k)
        draw(fr, m, 400 + 9 * k, 300 + 7 * k, stable_only=False)
        frames.append((fr, (400 + 9 * k, 300 + 7 * k)))
    built = PlateModel.build(frames)
    assert built.n == 5 and np.array_equal(built.picture, m.picture) and (built.share == 1).all()
    path = P.model_path(tmp_path)
    built.save(path)
    back = PlateModel.load(path)
    assert path.name == "plate_1366x768.npz" and np.array_equal(back.picture, built.picture)
    assert back.box == P.BOX and back.feet == P.FEET and back.interior == P.INTERIOR and back.n == 5


def test_the_gray_template_finds_her_tag_near_and_anywhere():
    rng = np.random.default_rng(3)
    fr = np.ascontiguousarray(rng.integers(0, 60, (H, W, 3), dtype=np.uint8))
    tag = rng.integers(0, 256, (15, 64), dtype=np.uint8)
    fr[500:515, 600:664] = tag[..., None]
    s, cx, cy = P.gray_near(fr, tag, (640, 510), (24, 36))
    assert s > 0.99 and (cx, cy) == (632, 507)
    peaks = P.gray_peaks(fr, tag)
    assert peaks[0][0] > 0.99 and peaks[0][1:] == (632, 507)
    fr[690:705, 300:364] = tag[..., None]             # under the HUD's top: not searched
    assert all(p[2] < 690 for p in P.gray_peaks(fr, tag) if p[0] > 0.9)
