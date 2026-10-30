"""Her body's search by her poses (gamelab/perception/poses.py; its pixel checks in C, gamelab/locate.c's lc_pack and
lc_agree): every candidate's agreement the same share in C as in numpy at every offset (in the box and off it), her
pose found where it's drawn by her anchor or, the anchor covered, by her colours' mass, darkened by the blink too, the
prediction choosing between one picture at two origins, behind the HUD her pixels above its top alone counted, the bank
kept, and her mirror axis by her poses seen both ways. (The prototype's answers on the evaluation set's frames: README,
Her place by her plate and her body.)"""

import numpy as np
import pytest

from gamelab import native
from gamelab.perception import poses as PO
from gamelab.perception.poses import Anchor, Background, BodyFinder, Pose, PoseBank

H, W = 768, 1366
HER = np.array([[180, 40, 200], [150, 30, 170], [230, 190, 160], [60, 50, 90], [90, 70, 140], [200, 160, 120],
                [120, 100, 80], [40, 30, 60]], np.uint8)   # her palette (made up)
LENS = np.array([[60, 50, 200], [250, 250, 250], [20, 20, 20]], np.uint8)   # her glasses: purple lenses, white, black


def anchor_picture(seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pic = LENS[rng.integers(len(LENS), size=(8, 14))]
    return np.ascontiguousarray(pic), np.ones((8, 14), bool)


def bank(seed: int = 0, n: int = 3) -> PoseBank:
    """n poses of hers: an ellipse of her colours each, her anchor (her glasses) pasted at a place of its own."""
    rng = np.random.default_rng(seed)
    apic, amask = anchor_picture()
    poses = []
    for i in range(n):
        h, w = int(rng.integers(70, 90)), int(rng.integers(36, 52))
        yy, xx = np.mgrid[0:h, 0:w]
        mask = ((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2 <= 1.0
        pic = HER[rng.integers(len(HER), size=(h, w))]
        ay, ax = int(rng.integers(8, 16)), int(rng.integers(w // 2 - 10, w // 2 - 4))
        pic[ay:ay + 8, ax:ax + 14] = apic
        mask[ay:ay + 8, ax:ax + 14] = True
        pic[~mask] = 0
        dx0, dy0 = -w // 2 - 0.5 + int(rng.integers(-3, 4)), -h
        p = Pose(i, np.ascontiguousarray(pic), mask, float(dx0), int(dy0), n=5, places=5,
                 anchors=[(0, int(round(dx0 + 0.5)) + ax, dy0 + ay)])
        poses.append(p)
    return PoseBank(poses, [Anchor(apic, amask, "glasses")], {"made": "test"})


def frame(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))


def draw(fr: np.ndarray, p: Pose, fx: float, fy: float, gain: float = 1.0) -> None:
    col = p.colour if gain == 1.0 else np.round(p.colour * np.float32(gain))
    fr[int(fy) + p.ys, int(np.floor(fx)) + p.xs] = col.astype(np.uint8)


needs_c = pytest.mark.skipif(native.locate_lib() is None, reason=f"locate.c isn't here: {native.locate_why}")


def both(monkeypatch, fn):
    out = []
    for on in (True, False):
        monkeypatch.setattr(PO, "USE_C", on)
        out.append(fn())
    monkeypatch.setattr(PO, "USE_C", True)
    return out


def key(hit):
    return None if hit is None else (hit.agree, hit.pose.id, hit.fx, hit.fy)


@needs_c
@pytest.mark.parametrize("R", [0, 1, 3])
def test_every_candidates_agreement_is_the_same_in_c_as_in_numpy(R):
    b = bank(2, 4)
    finder = BodyFinder(b)
    fr = frame(3)
    for i, p in enumerate(b.poses):
        draw(fr, p, 300.5 + 150 * i, 400, gain=1.0 if i % 2 else 0.5)
    rng = np.random.default_rng(R)
    x0, y0, x1, y1 = 200, 280, 900, 470     # the box: some candidates' pixels fall off it
    sets = (finder.sub, finder.scan, finder.full)
    for sets_ in sets:
        n = 60
        which = rng.integers(0, len(b.poses), n).astype(np.int32)
        fx = rng.integers(-40, x1 - x0 + 40, n).astype(np.int32)
        fy = rng.integers(-30, y1 - y0 + 90, n).astype(np.int32)
        fx[:4] = [100 - x0, 250 - x0, 400 - x0, 550 - x0]
        fy[:4] = 400 - y0
        got = []
        for lib in (native.locate_lib(), None):
            pic = PO._Picture(fr, x0, y0, x1, y1, lib)
            assert (pic.lib is None) == (lib is None)
            got.append(pic.agree(sets_, which, fx, fy, R))
        assert got[0].shape == (n, (2 * R + 1) ** 2)
        assert np.array_equal(got[0], got[1])


@pytest.mark.parametrize("i", range(3))
def test_her_pose_is_found_by_her_anchor(i, monkeypatch):
    b = bank()
    finder = BodyFinder(b)
    fr = frame(4)
    draw(fr, b.poses[i], 700.5, 500)
    a, c = both(monkeypatch, lambda: finder.find(fr, (690, 710, 470, 520)))
    assert key(a) == key(c)
    assert a.pose.id == i and (a.fx, a.fy) == (700.5, 500.0) and a.agree == 1.0 and finder.paths == ["head"]
    assert a.box == [700.5 + b.poses[i].dx0, 500 + b.poses[i].dy0, 700.5 + b.poses[i].dx0 + b.poses[i].w,
                     500 + b.poses[i].dy0 + b.poses[i].h]


def test_the_blink_darkened_pose_is_hers_too(monkeypatch):
    b = bank()
    finder = BodyFinder(b)
    fr = frame(5)
    draw(fr, b.poses[1], 640.5, 450, gain=0.5)
    a, c = both(monkeypatch, lambda: finder.find(fr, (620, 660, 430, 470)))
    assert key(a) == key(c) and a.pose.id == 1 and (a.fx, a.fy) == (640.5, 450.0) and a.agree == 1.0


def test_with_her_anchor_covered_her_colours_mass_places_her(monkeypatch):
    b = bank()
    fr = frame(6)
    p = b.poses[2]
    draw(fr, p, 800.5, 520)
    k, ax, ay = p.anchors[0]
    fr[520 + ay - 2:520 + ay + 10, 800 + ax - 2:800 + ax + 16] = (0, 200, 0)    # something over her glasses
    art = frame(6)   # the map's art: the frame as it was without her
    background = Background(art, (-658, -579))
    for bg in (None, background):
        finder = BodyFinder(b, bg)
        a, c = both(monkeypatch, lambda: finder.find(fr, (780, 820, 490, 540), view=(-658, -579)))
        assert key(a) == key(c)
        assert a.pose.id == 2 and (a.fx, a.fy) == (800.5, 520.0) and finder.paths[-1].startswith("mass")
        assert a.agree > 0.9


def test_one_picture_at_two_origins_is_told_apart_by_her_motion(monkeypatch):
    b = bank()
    finder = BodyFinder(b)
    fr = frame(7)
    draw(fr, b.poses[0], 600.5, 500)
    draw(fr, b.poses[0], 660.5, 500)          # (a cast draws her body 25 px ahead: here apart, both whole)
    for pred, want in (((598, 501), 600.5), ((663, 499), 660.5)):
        a, c = both(monkeypatch, lambda: finder.find(fr, (590, 670, 480, 520), pred=pred, slack=6.0))
        assert key(a) == key(c) and a.fx == want


def test_nothing_is_found_outside_her_box_or_where_she_isnt(monkeypatch):
    b = bank()
    finder = BodyFinder(b, feet_x=(425, 981))
    fr = frame(8)
    draw(fr, b.poses[0], 300.5, 500)            # outside the feet's columns
    a, c = both(monkeypatch, lambda: finder.find(fr, (280, 320, 480, 520)))
    assert a is None and c is None
    a = finder.find(frame(9), (600, 640, 480, 520))
    assert a is None or a.agree < 0.5


def test_the_bank_is_kept_as_it_was(tmp_path):
    b = bank(3, 4)
    b.poses[1].mirrored, b.poses[1].n, b.poses[1].n_src, b.poses[1].facing = True, 0, 7, "left"
    b.poses[2].grounded = 0.25
    path = PO.bank_path(tmp_path)
    b.save(path)
    back = PoseBank.load(path)
    assert path.name == "poses_1366x768.npz" and back.meta == {"made": "test"}
    assert len(back.poses) == 4 and [a.name for a in back.anchors] == ["glasses"]
    for p, q in zip(b.poses, back.poses):
        assert (p.id, p.dx0, p.dy0, p.n, p.n_src, p.places, p.facing, p.mirrored, p.grounded, p.anchors) == \
               (q.id, q.dx0, q.dy0, q.n, q.n_src, q.places, q.facing, q.mirrored, q.grounded, q.anchors)
        assert np.array_equal(p.picture, q.picture) and np.array_equal(p.mask, q.mask)
        assert np.array_equal(p.subset, q.subset) and np.array_equal(p.subset, PO.subset_of(p.id, len(p.ys)))
    assert np.array_equal(back.anchors[0].picture, b.anchors[0].picture)


def test_the_tracked_box_is_tight_across_and_reaches_up_over_her_platform():
    box, slack = PO.tracked_box((700.5, 500.0), sx=1.0, sy=0.5, grounded=True, speed=100.0)
    assert box == (700.5 - 13, 700.5 + 13, 500 - 64, 500 + 24) and slack == pytest.approx(6 + 1.5 * 100 * 0.066)
    box, _ = PO.tracked_box((700.5, 500.0), sx=30.0, sy=40.0, grounded=False, speed=0.0)
    assert box == (700.5 - 90, 700.5 + 90, 500 - 102, 500 + 90)


@needs_c
def test_the_mass_picture_is_the_same_in_c_as_in_numpy():
    b = bank(1, 4)
    fr = frame(11)
    draw(fr, b.poses[0], 500.5, 400)
    draw(fr, b.poses[1], 560.5, 410, gain=0.5)
    art = frame(11)
    art[300:330, 420:470] = fr[300:330, 420:470] // 2       # some art differing a little, some a lot
    known = np.ones(art.shape[:2], bool)
    known[350:360] = False
    for finder, view in ((BodyFinder(b, Background(art, (-658, -579), known)), (-658, -579)),
                         (BodyFinder(b, Background(art, (-600, -579), known)), (-658, -579)),   # the art 58 px off
                         (BodyFinder(b), None)):
        for box in ((430, 300, 640, 470), (0, 0, 120, 90), (1250, 700, 1366, 768)):
            c = finder.mass_picture(fr, *box, view, native.locate_lib())
            n = finder.mass_picture(fr, *box, view, None)
            assert c.dtype == n.dtype == np.float32 and np.array_equal(c, n)
            assert set(np.unique(n).tolist()) <= {1.0, -0.5, 0.0, -0.25}


def test_her_bank_is_built_from_frames_where_her_tag_is_sure(tmp_path):
    """Three poses of hers drawn over the map's art at six places each: the bank has them (cut to her palette), their
    mirror images, her glasses as an anchor held by every pose, and finds her by it."""
    drawn = bank(4, 3)
    rng = np.random.default_rng(12)
    art = np.ascontiguousarray(rng.integers(0, 256, (900, 1500, 3), dtype=np.uint8))
    background = Background(art, (-700, -600))
    view = (-658, -579)                         # the screen's top-left on the map: the art from (42, 21) of its own
    samples = []
    for p in drawn.poses:
        for k in range(6):
            fx, fy = 400.5 + 90 * k + 7 * p.id, 300 + 40 * p.id + 11 * k
            fr = np.ascontiguousarray(art[21:21 + H, 42:42 + W])
            draw(fr, p, fx, fy)
            samples.append((fr, (fx, fy), view))
    built = PO.build_bank(samples, background, on_line=lambda mx, my: True)
    seen = [p for p in built.poses if not p.mirrored]
    assert len(seen) == 3 and len(built.poses) == 6 and all(p.n == 6 and p.places == 6 for p in seen)
    pixels = lambda p: sorted(zip(p.ys.tolist(), p.xs.tolist(), map(tuple, p.colour.astype(int).tolist())))
    for p, q in zip(sorted(seen, key=lambda p: p.ys.min()), sorted(drawn.poses, key=lambda p: p.ys.min())):
        assert pixels(p) == pixels(q) and p.grounded == 1.0      # her pixels, each where it is from her feet
    names = [a.name for a in built.anchors]
    assert names[:2] == ["glasses", "glasses mirrored"] and all(any(k in (0, 1) for k, _, _ in p.anchors)
                                                                    for p in built.poses)
    built.save(PO.bank_path(tmp_path))
    finder = BodyFinder(PoseBank.load(PO.bank_path(tmp_path)), background)
    fr = frame(13)
    draw(fr, seen[1], 650.5, 480)
    hit = finder.find(fr, (640, 660, 460, 500), view=view)
    assert hit.pose.id == seen[1].id and (hit.fx, hit.fy) == (650.5, 480.0) and finder.paths == ["head"]


HUD = 700   # the HUD's top at 1366x768 (screen.HUD_TOP_GAME)


def test_behind_the_huds_top_only_her_pixels_above_it_count(monkeypatch):
    """BugBaby falling behind the HUD (her held-out frames 2528-2531, her legs below row 700): her own pose, its pixels
    there counted as disagreeing, was under the 0.8 her head's answer needs, and a wrong pose 20 px right and 28 up
    agreed 0.67-0.81 by her colours' mass; counted of those above it, her own agrees whole."""
    b = bank()
    finder = BodyFinder(b)
    for i, p in enumerate(b.poses):
        fr = frame(12 + i)
        fy = HUD + 24                                   # her feet 24 px under the HUD's top ...
        draw(fr, p, 640.5, fy)
        fr[HUD:] = 0                                    # ... the HUD drawn over her legs
        hidden = int(((p.ys + fy) >= HUD).sum()) / len(p.ys)
        assert 0.2 < hidden < PO.SHOWN_LEAST
        a, c = both(monkeypatch, lambda: finder.find(fr, (630, 650, fy - 20, fy + 6)))
        assert key(a) == key(c)
        assert a.pose.id == i and (a.fx, a.fy) == (640.5, fy) and a.agree == 1.0 and finder.paths == ["head"]
        pic = PO._Picture(fr, 560, 560, 720, 768, None)  # (counted against her, as before: her hidden share lost)
        r = pic.agree(finder.full, np.array([i], np.int32), np.array([80]), np.array([fy - 560]), 0)
        assert abs(r[0, 0] - (1 - hidden)) < 1e-9


def test_a_pose_mostly_behind_the_hud_counts_its_shown_pixels_against_at_least_half():
    b = bank()
    finder = BodyFinder(b)
    p = b.poses[1]
    for lib in (None, PO.native.locate_lib()):
        for fy in (HUD + 10, HUD + 50):
            fr = frame(15)
            draw(fr, p, 700.5, fy)
            fr[HUD:] = 0
            pic = PO._Picture(fr, 600, 560, 800, 768, lib, HUD)
            r = pic.agree(finder.full, np.array([1], np.int32), np.array([100]), np.array([fy - 560]), 0)[0, 0]
            shown = int(((p.ys + fy) < HUD).sum())
            assert abs(r - shown / max(shown, PO.SHOWN_LEAST * len(p.ys))) < 1e-9
            assert (r == 1.0) == (shown >= PO.SHOWN_LEAST * len(p.ys))


def test_her_mirror_axis_by_her_poses_seen_both_ways():
    """Each pose seen facing right against the one seen facing left, mirrored about the bank's feet: where her feet's
    x is e off in the bank, the left one sits 2e across from the right one's mirror image."""
    b = bank(4, 3)
    for e in (0.0, 2.5, -3.5):
        poses = []
        for p in b.poses:
            w = p.picture.shape[1]
            q = Pose(100 + p.id, np.ascontiguousarray(p.picture[:, ::-1]), np.ascontiguousarray(p.mask[:, ::-1]),
                     -p.dx0 - w + 1 + 2 * e, p.dy0, n=5, places=5, facing="left")
            poses += [Pose(p.id, p.picture, p.mask, p.dx0, p.dy0, n=5, places=5, facing="right"), q]
        corr, votes = PO.mirror_axis(PoseBank(poses, b.anchors))
        assert corr == e and votes > 0
    assert PO.mirror_axis(PoseBank([b.poses[0]], b.anchors)) == (None, 0)
