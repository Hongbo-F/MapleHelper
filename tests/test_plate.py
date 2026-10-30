"""Her name plate's exact-colour match (gamelab/perception/plate.py; its counting in C, gamelab/locate.c's lc_plate): the
counts the exact number of the plate's stable pixels the frame has in their colour at each place (a brute-force count
checks them), the candidates and their evidence the same in C as in numpy on every window, her plate found under
what covers part of it, below the HUD's top never, another wearer's plate (her plate item, another name) refused by
her letters' bins, by the white where her name has none, or where its frame matches better than where her letters
line up, and the model built from frames where her tag is sure, her tag's template cut from it and the model re-centred
on that, her mirror axis by the plate's own columns, her tag's sure levels from its scores; on real plates
(tests/fixtures/plates: crops of BugBaby's, RiceBaby's and AAABattery's, the same plate item) each character's model
takes her own and refuses the others'. (The prototype's answers on the evaluation set's frames: README, Her place by
her plate and her body.)"""

import json
from pathlib import Path

import cv2
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
    assert back.sure is None                          # (a model made before her tag's levels were kept: RiceBaby's)
    built.sure = (0.996, 0.95, 0.9)
    built.save(path)
    assert PlateModel.load(path).sure == (0.996, 0.95, 0.9)


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


OLD_RULES = dict(w_bad_far=0.5, align=0)   # the prototype's: what let another wearer's plate through


def test_white_where_her_name_has_none_weighs_as_much_as_her_letters_without_a_prediction(monkeypatch):
    """Another wearer's longer name (the review's AAABattery under BugBaby's model: 84 of her letters' px white, 75
    where hers has none, a bin or two of hers whole): at the gray peaks (no prediction) 84 - 0.5 x 75 passed, 84 - 75
    doesn't. Where she's predicted it weighs half, as what covers her own plate is white there too (a damage number)."""
    hers = model(1)
    pl, old = Plate(hers), Plate(hers, OLD_RULES)
    x0, x1, y0, y1 = hers.box
    yy, xx = np.mgrid[y0:y1, x0:x1]
    letters = hers.letters()
    ix0, ix1, iy0, iy1 = hers.interior
    inside = (xx >= ix0) & (xx <= ix1) & (yy >= iy0) & (yy <= iy1)
    pic = hers.picture.copy()
    first_bins = xx < ix0 + 2 * P.BIN_WIDTH                  # his name shows hers whole in her first two bins ...
    rng = np.random.default_rng(5)
    rest = np.flatnonzero(letters & ~first_bins)
    keep = 84 - int((letters & first_bins).sum())
    pic.reshape(-1, 3)[rng.choice(rest, len(rest) - keep, replace=False)] = (40, 40, 60)   # ... some of the rest
    extra = np.flatnonzero(inside & ~letters & ~first_bins)
    pic.reshape(-1, 3)[rng.choice(extra, 75, replace=False)] = 255                      # ... and white elsewhere
    other = PlateModel(pic, hers.share)
    for use_c in (True, False):
        monkeypatch.setattr(P, "USE_C", use_c)
        fr = frame(9)
        draw(fr, other, 600, 300, stable_only=False)
        ev = pl.evidence(fr, 600, 300)
        assert (ev["letters"], ev["bad"]) == (84, 75) and ev["clean"] >= P.DEFAULTS["clean_min"]
        assert old.verify(fr, [(0.6, 600, 300)]) is not None        # the prototype's rule took him
        assert pl.verify(fr, [(0.6, 600, 300)]) is None
        assert pl.near(fr, (601, 299)) is not None                  # (where she's predicted, with her dot agreeing)
        draw(fr, hers, 600, 300, stable_only=False)                # her own plate there: taken
        assert pl.verify(fr, [(0.6, 600, 300)]).candidate.bad == 0


def test_a_plate_whose_letters_line_up_with_hers_where_its_frame_doesnt_is_refused(monkeypatch):
    """Another wearer's plate whose name shares letters with hers ("Baby" of RiceBaby's under BugBaby's model): placed
    where her letters line up, 7 px off its own frame, the plate's px match more where its frame is (`aligned`)."""
    hers = model(3)
    pl, old = Plate(hers), Plate(hers, OLD_RULES)
    letters = hers.letters()
    x0, _, y0, _ = hers.box
    ys, xs = np.nonzero(letters)
    for use_c in (True, False):
        monkeypatch.setattr(P, "USE_C", use_c)
        fr = frame(10)
        draw(fr, hers, 640, 400, stable_only=False)
        fr[400 + y0 + ys, 640 + x0 + xs] = (40, 40, 60)              # its own name not hers ...
        fr[400 + y0 + ys, 647 + x0 + xs] = 255                       # ... her letters 7 px right of its frame's place
        cands = pl.candidates(fr, (647, 400), (3, 3))               # where her letters line up
        assert (cands[0].x, cands[0].y, cands[0].letters, cands[0].bad) == (647, 400, 143, 0)
        assert old.decide(cands, (647, 400), frame=fr) is not None   # the prototype's rule took it
        assert pl.aligned(fr, cands[0]) < P.DEFAULTS["align"]          # its frame matches better 7 px left
        assert pl.decide(cands, (647, 400), frame=fr) is None
        own = frame(10)
        draw(own, hers, 640, 400, stable_only=False)
        own[392:402, 600:640] = (90, 200, 90)                       # her own, partly covered: where it is
        hit = pl.near(own, (642, 401))
        assert hit is not None and (hit.x, hit.y) == (640, 400) and pl.aligned(own, hit.candidate) == 1.0


def test_her_tag_template_is_cut_from_her_plates_letters_and_the_model_recentred_on_it(monkeypatch):
    m = model(2)
    t, (dx, dy) = P.tag_template(m)
    ly, lx = np.nonzero(m.letters())
    x0, _, y0, _ = m.box
    m2 = P.TEMPLATE_MARGIN
    assert t.shape == (ly.max() - ly.min() + 1 + 2 * m2, lx.max() - lx.min() + 1 + 2 * m2) and t.dtype == np.uint8
    assert (dx, dy) == (lx.min() - m2 + x0 + t.shape[1] // 2, ly.min() - m2 + y0 + t.shape[0] // 2)
    r = m.recentred(dx, dy)
    assert r.box == (m.box[0] - dx, m.box[1] - dx, m.box[2] - dy, m.box[3] - dy) and r.n == m.n
    assert r.feet == (m.feet[0] - dx, m.feet[1] - dy) and np.array_equal(r.letters(), m.letters())
    fr = frame(11)
    draw(fr, m, 700, 420, stable_only=False)
    s, cx, cy = P.gray_near(fr, t, (700 + dx, 420 + dy), (20, 10))
    assert s > 0.99 and (cx, cy) == (700 + dx, 420 + dy)          # her tag's match: where the re-centred model's is
    for use_c in (True, False):
        monkeypatch.setattr(P, "USE_C", use_c)
        hit = Plate(r).near(fr, (cx + 3, cy - 2))
        assert (hit.x, hit.y) == (cx, cy) and hit.feet == (700 + m.feet[0], 420 + m.feet[1])   # her feet unmoved
    again, centre = P.tag_template(r)
    assert np.array_equal(again, t) and centre == (0, 0)
    blank = PlateModel(m.picture.copy(), m.share.copy())
    blank.picture[blank.letters()] = (40, 40, 60)
    with pytest.raises(ValueError):
        P.tag_template(blank)


def test_her_mirror_axis_is_the_middle_of_her_plates_stable_columns():
    m = model(4)
    assert m.axis() == (0 + (m.box[1] - m.box[0] - 1)) / 2 + m.box[0]        # the whole box stable: its middle
    share = m.share.copy()
    share[:, :6] = 0.0                                                     # its 6 left columns not hers ...
    share[3, 2] = 1.0                                                      # ... but a stray px
    assert PlateModel(m.picture, share, m.box).axis() == (6 + m.box[1] - m.box[0] - 1) / 2 + m.box[0]
    assert PlateModel(m.picture, np.zeros_like(share), m.box).axis() is None


def test_her_tags_sure_levels_come_from_its_own_scores():
    rng = np.random.default_rng(0)
    tail = list(rng.uniform(0.4, 0.9, 300))
    rice = list(np.clip(rng.normal(0.978, 0.005, 1600), 0, 0.995)) + tail     # RiceBaby's template on her frames
    typical, picks, bank = P.sure_levels(rice)
    assert abs(typical - 0.981) < 0.003 and (picks, bank) == (P.PICKS_MOST, P.BANK_MOST)   # the constants it had
    old = list(np.clip(rng.normal(0.879, 0.005, 2000), 0, 0.895)) + tail      # BugBaby's old text tag on her plate
    typical, picks, bank = P.sure_levels(old)
    assert 0.875 < typical < 0.888 and 0.84 < picks < 0.87 and 0.8 < bank < picks
    assert sum(v >= picks for v in old) > 1000                            # her plate's picks are there
    with pytest.raises(ValueError):
        P.sure_levels([])


def test_the_picks_from_peaks_found_once_are_sure_centres_own():
    m = model(5)
    t, (dx, dy) = P.tag_template(m)
    box = m.recentred(dx, dy).box
    frames = []
    for k in range(12):
        fr = frame(30 + k)
        draw(fr, m, 300 + 37 * k, 250 + 11 * k, stable_only=False)
        frames.append((k, fr))
    hud = P._hud(frames[0][1])
    want = P.sure_centres(frames, t, hud, box=box)
    got = P.pick_centres(((k, P.gray_peaks(f, t, k=4, rows=hud)) for k, f in frames), hud, box=box)
    assert got == want and len(got) == 12


# --- real plates: crops of the recordings (tests/fixtures/plates; README: Her place by her plate and her body) ------

PLATES = Path(__file__).parent / "fixtures" / "plates"


def real(name: str) -> tuple[np.ndarray, dict]:
    """A fixture crop put back where it was in a frame of the game's own size (mid gray around it), and its note."""
    notes = json.loads((PLATES / "plates.json").read_text(encoding="utf-8"))
    n = notes[name]
    crop = np.ascontiguousarray(cv2.imread(str(PLATES / n["file"]))[..., ::-1])
    fr = np.full((H, W, 3), 90, np.uint8)
    x0, y0 = n["at"]
    fr[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return fr, n


def real_model(who: str) -> PlateModel:
    return PlateModel.load(PLATES / f"{who}_plate_1366x768.npz")


@pytest.mark.parametrize("use_c", [True, False])
def test_on_real_plates_each_takes_her_own_and_refuses_the_others(use_c, monkeypatch):
    monkeypatch.setattr(P, "USE_C", use_c)
    bug, rice = Plate(real_model("bugbaby")), Plate(real_model("ricebaby"))
    # her own plates (BugBaby's training frames: whole, and partly covered; RiceBaby's)
    for name, pl in (("bugbaby_whole", bug), ("bugbaby_covered", bug), ("ricebaby", rice)):
        fr, n = real(name)
        c = tuple(n["centre"])
        hit = pl.near(fr, (c[0] + 3, c[1] - 2))
        assert hit is not None and (hit.x, hit.y) == c, name
        assert pl.verify(fr, [(0.9, *c)]) is not None, name
    # RiceBaby's plate under BugBaby's model: "Baby" lines up 6 px off its frame (the prototype's rule took it)
    fr, n = real("ricebaby")
    c = tuple(n["centre"])
    took = Plate(real_model("bugbaby"), OLD_RULES).near(fr, c)
    assert took is not None and abs(took.x - c[0]) == 6
    assert bug.aligned(fr, took.candidate) < P.DEFAULTS["align"] and bug.near(fr, c) is None
    # AAABattery's plate (the same item; RiceBaby's frame of 10-29) at the gray peaks: his white where hers has none
    fr, n = real("aaabattery")
    c = tuple(n["centre"])
    assert Plate(real_model("bugbaby"), OLD_RULES).verify(fr, [(0.6, *c)]) is not None
    assert bug.verify(fr, [(0.6, *c)]) is None and rice.verify(fr, [(0.6, *c)]) is None
    # BugBaby's plate under RiceBaby's model
    fr, n = real("bugbaby_whole")
    c = tuple(n["centre"])
    assert rice.near(fr, c) is None and rice.verify(fr, [(0.6, *c)]) is None


def test_bugbabys_tag_cut_from_her_real_plate_scores_on_hers_not_on_the_others():
    m = real_model("bugbaby")
    t, centre = P.tag_template(m)
    assert t.shape == (15, 52) and centre == (0, 0)                 # (her model is centred on it already)
    for name, most in (("bugbaby_whole", None), ("ricebaby", 0.85), ("aaabattery", 0.85)):
        fr, n = real(name)
        s, cx, cy = P.gray_near(fr, t, n["centre"], (40, 25))
        if most is None:
            assert s > 0.98 and (cx, cy) == tuple(n["centre"])
        else:
            assert s < most, (name, s)
