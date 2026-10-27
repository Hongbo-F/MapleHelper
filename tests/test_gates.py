"""Colour gates learned for the monster kinds without a hand-made one (perception.gates; mobs.MobDetector's
`background`), kinds that share a body searched as one and told apart by their pictures (mobs._pick_kind) and re-found
as their own kind only (MobDetector.refind), and a map's background colours learned from saved frames
(sprites.learn_background, `learn-colours`). Made-up kinds are drawn onto
tests/fixtures/native (West Rocky Mountain II at 1366x768); the new maps' labelled frames stay in D:/MapleHelper-scratch
(the tests on them skip without them)."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from gamelab.__main__ import main
from gamelab.analysis import pasted
from gamelab.perception import gates, match, mobs, sprites
from gamelab.perception.mobs import INDEX, SCALE_FILE, MobDetector
from gamelab.screen import NATIVE, STRETCHED

from test_pasted import FRAMES, frame

LABELS = Path(__file__).parent / "fixtures"
TREE_DUNGEON = Path("D:/MapleHelper-scratch/map3/native")       # 10002073 at 1366x768, labelled
WEST_ROCKY = Path("D:/MapleHelper-scratch/newmap/frames")       # 10004021 at 1920x1080, 42 of them labelled
WEST_ROCKY_NATIVE = Path("D:/MapleHelper-scratch/newmap/native")  # ... and at 1366x768, labelled


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
    colours = ((100, 80, 60), (10, 200, 30), (250, 0, 250))
    common, rare, never = (gates.bins(np.array([c], np.uint8))[0] for c in colours)
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


def refound(det, scene, kind, left, top, fallback=True):
    """A track of `kind` (its picture facing left, last seen with its top-left at left, top) re-found in `scene`, as
    WorldStep re-finds (above the chat box): (its feet) or None."""
    t = next(t for t in det.templates[kind] if t.facing == -1)
    th, tw = t.rgb.shape[:2]
    m = det.refind(scene, (left, top, left + tw, top + th), kind, t.name, t.facing, (0, 0, 1366, 700), fallback)
    return None if m is None else (m.kind, m.foot_x, m.feet_y)


def test_a_track_is_re_found_on_a_monster_of_its_own_kind_alone(tmp_path):
    caps = {"violet-cap": (170, 60, 190), "pink-cap": (200, 60, 150)}
    det = detector(tmp_path, {f"{k}_move0": body(cap=c) for k, c in caps.items()})
    scene = drawn(body(cap=caps["pink-cap"]), 640, 300)
    for fallback in (True, False):
        assert refound(det, scene, "pink-cap", 640, 300, fallback) == ("pink-cap", 662, 340)
        assert refound(det, scene, "violet-cap", 640, 300, fallback) is None
    twins, det.twins = det.twins, {}
    assert refound(det, scene, "violet-cap", 640, 300) == ("violet-cap", 662, 340)  # its picture fits the other
    det.twins = twins
    pictures = {"stub_move0": body(cap=(120, 80, 40)), "stub-axe_move0": body(cap=(120, 80, 40), axe=True)}
    det = detector(tmp_path, pictures)
    with_axe, without = drawn(pictures["stub-axe_move0"], 400, 200), drawn(pictures["stub_move0"], 400, 216)
    assert refound(det, with_axe, "stub-axe", 400, 200) == ("stub-axe", 422, 256)
    assert refound(det, with_axe, "stub", 400, 216) is None  # the stub that is all of it but the axe: not a stub
    assert refound(det, without, "stub", 400, 216) == ("stub", 422, 256)
    assert refound(det, without, "stub-axe", 400, 200) is None


def test_told_apart_with_the_pictures_side_kept_the_same_as_worked_out_each_time(tmp_path):
    # (_pick_here: `_its_kind`'s _pick_kind with no slack, what the pictures alone give kept from call to call)
    caps = {"violet-cap": (170, 60, 190), "pink-cap": (200, 60, 150)}
    sets = [{"stub_move0": body(cap=(120, 80, 40)), "stub-axe_move0": body(cap=(120, 80, 40), axe=True)},
            {f"{k}_move0": body(cap=c) for k, c in caps.items()}]
    told = 0
    for i, pictures in enumerate(sets):
        (tmp_path / str(i)).mkdir()
        det = detector(tmp_path / str(i), pictures)
        cache = {}
        for pic in pictures.values():
            for left, top, flip in ((400, 200, False), (403, 197, True), (2, 0, False), (1320, 600, False)):
                scene = drawn(pic, left, top, flip)[:700]
                for kind in det.twins:
                    for t in det.templates[kind]:
                        th, tw = t.rgb.shape[:2]
                        for dx, dy in ((0, 0), (1, 0), (0, -1), (-3, 2)):
                            m = mobs._seen(kind, t, left + dx + tw // 2, top + dy + th // 2, tw, th, 0.912)
                            alone = mobs._pick_kind(scene, m, det.twins[kind], det.templates, 0, det.aligned)
                            for _ in range(2):  # (worked out, then kept)
                                kept = mobs._pick_kind(scene, m, det.twins[kind], det.templates, 0, det.aligned,
                                                       cache)
                                assert kept == alone, (kind, t.name, left, top, dx, dy)
                            told += alone.kind != kind
        assert any(k[0] == "layout" for k in cache)
    assert told  # (some told as the other kind)


def test_a_twins_picture_is_checked_where_it_lies_on_the_monsters():
    pic = np.zeros((20, 20, 3), np.uint8)
    pic[5:15, 4:12] = 1
    shifted = np.roll(pic, 1, axis=1)  # the same body a px across, on the same feet
    one, other = (SimpleNamespace(mask3=p, origin=(10.0, 20.0)) for p in (pic, shifted))  # (what _cover reads)
    assert mobs._aligned(one, other) == (-1, 0) and mobs._aligned(other, one) == (1, 0)
    assert mobs._aligned(one, one) == (0, 0)


def test_one_places_score_is_the_masked_matchs():
    rng = np.random.default_rng(3)
    tpl, patch = (rng.integers(0, 256, (40, 44, 3), np.uint8) for _ in range(2))
    mask = rng.random((40, 44)) > 0.4
    patch[mask] = np.clip(0.6 * tpl[mask] + 0.4 * patch[mask], 0, 255).astype(np.uint8)
    cv = float(match.masked_match(patch, tpl, np.dstack([mask.astype(np.uint8)] * 3))[0, 0])
    assert abs(mobs._score(patch, tpl, mask) - cv) < 1e-6 and cv > 0.5
    flat = np.full_like(patch, 90)
    assert mobs._score(flat, tpl, mask) == -1.0 == float(match.masked_match(flat, tpl, np.dstack([mask] * 3)
                                                                             .astype(np.uint8))[0, 0])


# --- the background, learned from saved frames --------------------------------------------------------------------

def test_a_maps_background_is_learned_with_its_monsters_cut_out(tmp_path):
    (tmp_path / "pics").mkdir()
    pics = folder(tmp_path / "pics", {"blob_move0": body()})
    frames = [drawn(body(), 300 + 100 * i, 250, name=n) for i, n in enumerate(FRAMES)]
    made = []

    def make(geo, back):
        made.append(back.copy())
        return MobDetector(pics, threads=1, frame_scale=geo.frame_scale, kinds={"blob"}, background=back)
    counts, cut = sprites.learn_background(frames, make)
    magenta = gates.bins(np.array([(200, 40, 200)], np.uint8))[0]
    assert cut == 3 and made[0][magenta] > 500  # first against the frames as they are, the blobs in them
    assert counts[magenta] == 0 and counts.sum() < made[0].sum()  # then without them


def test_learn_colours_keeps_a_maps_background_where_its_detector_finds_it(tmp_path, monkeypatch, capsys):
    from gamelab.localization import mapdata
    pics = folder(tmp_path, {"blob_move0": body()})
    model = SimpleNamespace(id=4242, spawns=[SimpleNamespace(mob_id=1, name="Blob")])
    monkeypatch.setattr(mapdata, "load", lambda map_id: model)
    monkeypatch.setattr(sprites, "ensure", lambda names, stretch: pics)
    shots = tmp_path / "shots"
    shots.mkdir()
    for i, n in enumerate(FRAMES):
        Image.fromarray(drawn(body(), 300 + 100 * i, 250, name=n)).save(shots / n)
    out = tmp_path / "backgrounds"
    assert main(["learn-colours", "--map", "4242", str(shots), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "3 frames (1366x768)" in printed and "3 monsters found" in printed and "blob's gate takes" in printed
    counts = gates.load_background(4242, out)
    det = sprites.detector_for(model, NATIVE, counts)
    assert det.learned == {"blob"}
    assert [m.kind for m in det.detect_wide(drawn(body(), 800, 300), (0, 700))] == ["blob"]
    assert sprites.detector_for(model, NATIVE).learned == frozenset()  # none kept for it in assets/backgrounds


# --- the new maps' labelled frames (D:/MapleHelper-scratch; skipped without them) ---------------------------------

def needs(path: Path, model_id: int):
    from gamedata.paths import COPY
    if not path.is_dir() or not (COPY / "Packs" / "Mob_00000.ms").is_file():
        pytest.skip(f"needs {path} and the game data copy")
    from gamelab.localization import mapdata
    return mapdata.load(model_id)


def _match(found, labels, tol):
    """Each label's nearest monster found within `tol` px of its feet, one each: {label index: monster}."""
    pairs = sorted((abs(m.foot_x - lab["x"]) + abs(m.feet_y - lab["y"]), i, j) for i, m in enumerate(found)
                   for j, lab in enumerate(labels)
                   if abs(m.foot_x - lab["x"]) <= tol and abs(m.feet_y - lab["y"]) <= tol)
    used, out = set(), {}
    for _, i, j in pairs:
        if i not in used and j not in out:
            used.add(i)
            out[j] = found[i]
    return out


def test_tree_dungeons_mushrooms_are_found_and_told_apart():
    model = needs(TREE_DUNGEON, 10002073)
    labels = json.loads((LABELS / "tree_dungeon_monsters.json").read_text(encoding="utf-8"))["frames"]
    det = sprites.detector_for(model, NATIVE)
    assert det.learned == {"horny-mushroom", "green-mushroom"} and "snail" not in det.templates
    plain, found_plain, wrong, extra = 0, 0, 0, []
    for name, e in labels.items():
        scene = np.asarray(Image.open(TREE_DUNGEON / name).convert("RGB"))
        found = det.detect_wide(scene, (0, 700))
        hit = _match(found, e["monsters"], 10)
        for j, m in enumerate(e["monsters"]):
            clean = m["state"] == "normal" and not m["hidden"] and (m["fit"] or 0) >= 0.99 and m["y"] <= 700
            plain += clean
            if j in hit:
                found_plain += clean
                wrong += hit[j].kind != m["kind"]
                if clean:
                    assert abs(hit[j].foot_x - m["x"]) <= 3 and abs(hit[j].feet_y - m["y"]) <= 3, (name, m)
                    assert hit[j].facing == m["facing"]
        extra += [(name, m.kind, m.foot_x, m.feet_y) for i, m in enumerate(found) if m not in hit.values()]
        her = e["her"][1]
        band = det.detect(scene, (her - 140, min(700, her + 75)))
        extra += [(name, m.kind) for m in band if m not in _match(band, e["monsters"], 10).values()]
    assert (plain, found_plain, wrong, extra) == (44, 44, 0, [])  # no player, pet or pet's tag taken for one


def test_horny_and_green_mushrooms_are_told_apart_wherever_the_sprite_fit_saw_one():
    model = needs(TREE_DUNGEON, 10002073)
    labels = json.loads((LABELS / "tree_dungeon_monsters.json").read_text(encoding="utf-8"))["frames"]
    det = sprites.detector_for(model, NATIVE)
    told = []
    for name, e in labels.items():
        scene = np.asarray(Image.open(TREE_DUNGEON / name).convert("RGB"))[:700]
        for m in e["monsters"]:
            if not m["kind"].endswith("mushroom") or not m["frame"].startswith(("stand", "move", "hit1")):
                continue
            for start in ("horny-mushroom", "green-mushroom"):  # found as either, the same monster's kind
                t = next(t for t in det.templates[start] if t.frame == m["frame"] and t.facing == m["facing"])
                th, tw = t.rgb.shape[:2]
                if m["y"] > 700:  # (behind the chat bar: not searched)
                    continue
                r = mobs._match_at(scene, t.rgb, t.mask3, int(m["x"] - t.origin[0]) + tw // 2,
                                   int(m["y"] - t.origin[1]) + th // 2, mobs.FINE_SLACK)
                hit = mobs._seen(start, t, r[1], r[2], r[3], r[4], round(r[0], 3))
                told.append(mobs._pick_kind(scene, hit, det.twins[start], det.templates, mobs.FINE_SLACK).kind
                            == m["kind"])
    assert len(told) == 2 * 55 and all(told)  # spawning, flinching and covered ones too


def test_a_mushrooms_or_a_stumps_track_re_finds_it_only_if_of_its_kind():
    # Each mushroom and stump whose frame the labels give, re-found where it stands as WorldStep re-finds (above the
    # chat box, the fallback on) by a track of its own kind and by one of each kind that shares its body, each with
    # the picture that matched it last (its frame, facing its way). Without the kind check, the other kinds' tracks
    # re-found 52 of the 55 mushrooms and 115 of the stumps' 116.
    twins = {"horny-mushroom": ("green-mushroom",), "green-mushroom": ("horny-mushroom",),
             "stump": ("dark-stump", "axe-stump"), "axe-stump": ("stump", "dark-stump")}
    out = {}
    for model_id, root, fixture in ((10002073, TREE_DUNGEON, "tree_dungeon_monsters.json"),
                                    (10004021, WEST_ROCKY_NATIVE, "west_rocky_native_monsters.json")):
        det = sprites.detector_for(needs(root, model_id), NATIVE)
        tally = {"own": [0, 0], "other": [0, 0]}  # (asked, re-found within 3 px)
        for name, e in json.loads((LABELS / fixture).read_text(encoding="utf-8"))["frames"].items():
            path = root / name if (root / name).is_file() else root.parent / name
            scene = np.asarray(Image.open(path).convert("RGB"))
            for m in e["monsters"] if isinstance(e, dict) else [m for m in e if not m["flags"]]:
                if m["kind"] not in twins or m["y"] > 700 or not m["frame"].startswith(("stand", "move", "hit1")):
                    continue
                for role, kinds in (("own", (m["kind"],)), ("other", twins[m["kind"]])):
                    for kind in kinds:
                        t = next(t for t in det.templates[kind] if t.frame == m["frame"] and t.facing == m["facing"])
                        x0, y0 = round(m["x"] - t.origin[0]), round(m["y"] - t.origin[1])
                        box = (x0, y0, x0 + t.rgb.shape[1], y0 + t.rgb.shape[0])
                        r = det.refind(scene, box, kind, t.name, t.facing, (0, 0, 1366, 700))
                        tally[role][0] += 1
                        tally[role][1] += r is not None and abs(r.foot_x - m["x"]) <= 3 and abs(r.feet_y - m["y"]) <= 3
        out[model_id] = tally
        if model_id == 10004021:  # an Axe Stump's move/1 and move/3 lie a px across from a Stump's on their feet
            assert {(a, b) for (a, _, b), d in det.aligned.items() if d != (0, 0)} >= {
                ("axe-stump_move1", "stump_move1"), ("stump_move3", "axe-stump_move3")}
            assert all(abs(d[0]) == 1 and d[1] == 0 for d in det.aligned.values() if d != (0, 0))
    assert out == {10002073: {"own": [55, 53], "other": [55, 0]}, 10004021: {"own": [58, 58], "other": [116, 0]}}


def test_west_rocky_mountains_stumps_and_snails_are_found_at_the_games_own_size():
    model = needs(WEST_ROCKY_NATIVE, 10004021)
    labels = json.loads((LABELS / "west_rocky_native_monsters.json").read_text(encoding="utf-8"))["frames"]
    det = sprites.detector_for(model, NATIVE)
    assert det.learned == {"stump", "dark-stump", "axe-stump", "red-snail"}
    plain, found_plain, wrong, extra = 0, 0, 0, []
    for name, ms in labels.items():
        path = WEST_ROCKY_NATIVE / name if name.startswith("n28") else WEST_ROCKY_NATIVE.parent / name
        found = det.detect_wide(np.asarray(Image.open(path).convert("RGB")), (0, 700))
        hit = _match(found, ms, 10)
        for j, m in enumerate(ms):
            plain += not m["flags"]
            if j in hit and not m["flags"]:
                found_plain += 1
                wrong += hit[j].kind != m["kind"]
                assert abs(hit[j].foot_x - m["x"]) <= 1 and abs(hit[j].feet_y - m["y"]) <= 1
        extra += [(name, m.kind, m.foot_x) for m in found if m not in hit.values()]
    assert plain == 58 and found_plain >= 56 and wrong == 0 and extra == []


def test_west_rocky_mountains_monsters_are_found_on_its_1920x1080_frames():
    model = needs(WEST_ROCKY, 10004021)
    labels = json.loads((LABELS / "west_rocky_monsters.json").read_text(encoding="utf-8"))["frames"]
    det = sprites.detector_for(model, STRETCHED)
    plain, found_plain, wrong, extra = 0, 0, 0, []
    for name, ms in labels.items():
        full = np.asarray(Image.open(WEST_ROCKY / name).convert("RGB"))
        half = cv2.resize(full, (960, 540), interpolation=cv2.INTER_AREA)  # as a look shrinks it
        found = [m for m in det.detect_wide(half, (0, 475)) if m.kind in det.learned]
        half_px = [dict(m, x=m["x"] / 2, y=m["y"] / 2) for m in ms]
        hit = _match(found, half_px, 7)
        for j, m in enumerate(ms):
            plain += not m["flags"]
            if j in hit and not m["flags"]:
                found_plain += 1
                wrong += hit[j].kind != m["kind"]
        extra += [(name, m.kind) for m in found if m not in hit.values()]
    assert plain == 203 and found_plain >= 165 and wrong <= 3 and extra == []
