"""Colour gates learned for the monster kinds without a hand-made one (perception.gates; mobs.MobDetector's
`background`), and kinds that share a body searched as one and told apart by their pictures (mobs._pick_kind). Made-up
kinds are drawn onto tests/fixtures/native (West Rocky Mountain II at 1366x768)."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gamelab.analysis import pasted
from gamelab.perception import gates, mobs
from gamelab.perception.mobs import INDEX, SCALE_FILE, MobDetector
from gamelab.screen import NATIVE, STRETCHED

from test_pasted import FRAMES, frame


# --- made-up kinds --------------------------------------------------------------------------------------------------

def body(cap=(200, 40, 200), stem=(230, 200, 150), axe=False) -> np.ndarray:
    """A made-up monster facing left (RGBA): a cap over a stem, with two dark marks, 44 x 40 (with an axe sticking up
    from the cap's left, 44 x 56); its feet at the bottom middle."""
    h = 56 if axe else 40
    rgba = np.zeros((h, 44, 4), np.uint8)
    top = h - 40
    rgba[top:top + 18, 2:42] = (*cap, 255)
    rgba[top + 18:h, 12:32] = (*stem, 255)
    rgba[top + 4:top + 7, 6:16] = (20, 20, 20, 255)  # an eye: something not flat in every picture
    rgba[top + 22:top + 26, 16:20] = (60, 30, 10, 255)
    if axe:
        rgba[0:top, 4:14] = (150, 150, 160, 255)
        rgba[2:6, 14:30] = (90, 60, 30, 255)
    return rgba


def folder(tmp_path: Path, pictures: dict[str, np.ndarray]) -> Path:
    """Made-up game pictures at the game's own size, each a frame "move/0" standing on its bottom middle."""
    for name, rgba in pictures.items():
        Image.fromarray(rgba, "RGBA").save(tmp_path / f"{name}.png")
    index = {f"{name}.png": {"origin": [rgba.shape[1] // 2, rgba.shape[0]], "w": rgba.shape[1], "h": rgba.shape[0],
                             "frame": "move/0", "lt": [-20, -38], "rb": [20, 0]} for name, rgba in pictures.items()}
    (tmp_path / INDEX).write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / SCALE_FILE).write_text(json.dumps({"px_per_game_px": 1.0}), encoding="utf-8")
    return tmp_path


def background(names=FRAMES) -> np.ndarray:
    """The fixture frames' colours above the HUD at step 1's scale, as learn_background counts them (no monster of
    a made-up kind on them)."""
    search = NATIVE.frame_scale / mobs.SEARCH_SCALE
    return gates.background([mobs.step1_frame(frame(n)[:int(768 * NATIVE.hud_top)], search) for n in names])


def detector(tmp_path, pictures, kinds=None, **kw) -> MobDetector:
    return MobDetector(folder(tmp_path, pictures), threads=1, frame_scale=NATIVE.frame_scale,
                       kinds=set(kinds or {n.split("_")[0] for n in pictures}), background=background(), **kw)


def drawn(rgba, left, top, flip=False, name="n28_000.png"):
    scene = frame(name)
    pasted.paste(scene, rgba[:, ::-1] if flip else rgba, left, top)
    return scene


# --- gates ----------------------------------------------------------------------------------------------------------

def test_a_kinds_gate_holds_the_colours_its_pictures_have_far_more_often_than_the_background():
    back = np.zeros(1 << (3 * gates.BITS), np.int64)
    common, rare, never = (gates.bins(np.array([c], np.uint8))[0] for c in ((100, 80, 60), (10, 200, 30), (250, 0, 250)))
    back[common], back[rare] = 1_000_000, 10
    kind = [gates.histogram(np.array([(100, 80, 60)] * 50 + [(10, 200, 30)] * 50 + [(250, 0, 250)] * 5, np.uint8))]
    lut = gates.learn(kind, back)
    assert not lut[common] and lut[rare] and lut[never]  # the background's own colour out, however common in the kind
    assert lut.sum() == 2  # and nothing the kind never shows
    strict = gates.learn(kind, back, ratio=500)  # 297 times commoner in the kind than in the background; 680
    assert not strict[rare] and strict[never]


def test_every_kinds_gate_is_one_lookup_a_pixel():
    luts = {k: np.zeros(1 << (3 * gates.BITS), bool) for k in ("a", "b", "c")}
    red, green = (gates.bins(np.array([c], np.uint8))[0] for c in ((250, 10, 10), (10, 250, 10)))
    luts["a"][red] = luts["b"][red] = luts["b"][green] = True
    g = gates.Gates(luts)
    picture = np.array([[(250, 10, 10), (10, 250, 10), (0, 0, 0)]], np.uint8)
    channels = mobs.split_channels(picture)
    looked = g.lookup(*channels)
    assert g.of("a", looked).tolist() == [[True, False, False]] and g.of("b", looked).tolist() == [[True, True, False]]
    assert g.of(("a", "c"), looked).tolist() == [[True, False, False]] == g.test("a", *channels).tolist()
    assert g.lut.dtype == np.uint8 and gates.Gates({str(i): luts["a"] for i in range(9)}).lut.dtype == np.uint16
    assert g.share("b", np.bincount([red, red, green, 0], minlength=g.lut.size)) == 0.75


def test_a_backgrounds_counts_leave_out_the_boxes_cut_and_are_kept_by_map(tmp_path):
    picture = np.zeros((10, 10, 3), np.uint8)
    picture[2:4, 2:4] = (200, 0, 0)
    counts = gates.background([picture], [[(2, 2, 4, 4)]])
    assert counts.sum() == 96 and counts[gates.bins(np.array([(200, 0, 0)], np.uint8))[0]] == 0
    path = gates.save_background(123, counts, frames=1, cut=1, folder=tmp_path)
    assert path == tmp_path / "123.npz" and np.array_equal(gates.load_background(123, tmp_path), counts)
    assert gates.load_background(124, tmp_path) is None and gates.load_background(None, tmp_path) is None
    np.savez_compressed(path, counts=counts, bits=gates.BITS + 1, frames=1, cut=1)
    assert gates.load_background(123, tmp_path) is None  # binned otherwise: learned again


# --- the search with learned gates --------------------------------------------------------------------------------

def test_a_kind_without_a_hand_made_test_is_looked_for_with_its_learned_gate(tmp_path):
    det = detector(tmp_path, {"blob_move0": body()})
    assert det.learned == {"blob"} and det.units == [(("blob",), det.units[0][1], [[0], [1]])]
    for left, top, flip in ((300, 250, False), (900, 120, True)):
        found = det.detect_wide(drawn(body(), left, top, flip), (0, 700))
        assert [(m.kind, m.foot_x, m.feet_y, m.facing, m.score) for m in found] == [
            ("blob", left + 22, top + 40, 1 if flip else -1, 1.0)]
    assert det.detect_wide(frame(), (0, 700)) == []  # the map's frames as they are: nothing
    plain = MobDetector(folder(tmp_path, {"blob_move0": body()}), threads=1, frame_scale=1.0, kinds={"blob"})
    assert plain.templates == {} and plain.learned == frozenset()  # without the map's background, not looked for


def test_the_first_maps_kinds_are_looked_for_as_they_were_whatever_background():
    det = MobDetector(threads=1, background=np.ones(1 << (3 * gates.BITS), np.int64))  # (no kind to learn)
    assert det.learned == frozenset() and det.gates is None and det.overlap == 0 and det.twins == {}
    assert all(len(kinds) == 1 and shapes == [[i] for i in range(len(members))] for kinds, members, shapes in det.units)


def test_whole_screen_bands_overlap_by_the_tallest_learned_picture(tmp_path):
    tall = cv2.resize(body(), (44, 120), interpolation=cv2.INTER_NEAREST)
    det = detector(tmp_path, {"pole_move0": tall})
    assert det.overlap == 120 + round(mobs.CHECK_SLACK * det.px)
    # its top 60 rows under the second band's: in bands of 171 rows (as without learned kinds), none holds it whole
    found = det.detect_wide(drawn(tall, 600, 188), (0, 700))
    assert [(m.kind, m.feet_y) for m in found] == [("pole", 188 + 120)]
    det.overlap = 0
    assert det.detect_wide(drawn(tall, 600, 188), (0, 700)) == []


# --- kinds that share a body ---------------------------------------------------------------------------------------

def test_kinds_of_one_shape_are_searched_as_one_and_told_apart_by_their_colours(tmp_path):
    # (caps about as far apart as a Horny Mushroom's and a Green's, to the masked match: a place is gated with the
    # first kind's picture)
    pictures = {"violet-cap_move0": body(cap=(170, 60, 190)), "pink-cap_move0": body(cap=(200, 60, 150))}
    det = detector(tmp_path, pictures)
    assert det.twins == {"pink-cap": ("violet-cap",), "violet-cap": ("pink-cap",)}
    (kinds, members, shapes), = det.units
    assert kinds == ("pink-cap", "violet-cap") and len(members) == 4 and len(shapes) == 2  # one silhouette a frame
    for kind, cap in (("violet-cap", (170, 60, 190)), ("pink-cap", (200, 60, 150))):
        for flip in (False, True):
            found = det.detect_wide(drawn(body(cap=cap), 640, 300, flip), (0, 700))
            assert [(m.kind, m.foot_x, m.feet_y) for m in found] == [(kind, 662, 340)], (kind, flip)
            band = det.detect(drawn(body(cap=cap), 640, 300, flip), (200, 420))
            assert [m.kind for m in band] == [kind]


def test_a_kind_with_parts_of_its_own_is_the_monsters_only_where_they_show(tmp_path):
    pictures = {"stub_move0": body(cap=(120, 80, 40)), "stub-axe_move0": body(cap=(120, 80, 40), axe=True)}
    det = detector(tmp_path, pictures)
    assert det.twins == {"stub": ("stub-axe",), "stub-axe": ("stub",)}
    assert sorted(kinds for kinds, _, _ in det.units) == [("stub",), ("stub-axe",)]  # not the same shape: apart
    found = det.detect_wide(drawn(body(cap=(120, 80, 40), axe=True), 400, 200), (0, 700))
    assert [(m.kind, m.feet_y) for m in found] == [("stub-axe", 256)]
    found = det.detect_wide(drawn(body(cap=(120, 80, 40)), 400, 216), (0, 700))
    assert [(m.kind, m.feet_y) for m in found] == [("stub", 256)]  # the stub matches the axe's own picture well too
    plain = mobs.Mob("stub-axe", 422, 228, 44, 56, 0.9, 422.0, 256.0, -1, "move", "move/0", "stub-axe_move0")
    picked = mobs._pick_kind(drawn(body(cap=(120, 80, 40)), 400, 216)[:700], plain, det.twins["stub-axe"],
                             det.templates, mobs.FINE_SLACK)
    assert picked.kind == "stub"
