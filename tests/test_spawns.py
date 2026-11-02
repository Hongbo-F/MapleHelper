"""Monsters fading in at their spawn points (perception.spawns): the blend fit's integer sums in C and in numpy the
same; a newborn drawn over the map's art found at its first faint frames, followed, and handed over to the whole screen;
what's drawn over a point that isn't a newborn left alone. On made-up pictures over a made-up map's art, and on crops of
Tree Dungeon's lossless frames (tests/fixtures/spawns: the golden clip of the user's own play and the Lie Detector's
alarm clips; index.json says what each is and what the watch finds there) pasted into the map's art at their view:
newborns at the alphas the prototype found them at (D:/MapleHelper-scratch/spawn-fade/proto), the nearest negatives to
the rule of every labelled point-frame, and her standing on a spawn point for 6.6 s before the newborn there. Then the
frame step's and the tracker's side: the newborns among the frame's detections until the whole screen has them, one
track each from their first sighting, their alpha and when they turn solid in it."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gamelab import native
from gamelab.perception import spawns
from gamelab.perception.mobs import Mob
from gamelab.perception.spawns import Art, Point, SpawnWatch

FIXTURES = Path(__file__).parent / "fixtures" / "spawns"
TREE = 10002073


class Flat:
    """A level floor at y from `left` to `right` (a platform's surface, as spawns.Floor takes one)."""

    def __init__(self, left: float, right: float, y: float):
        self.left, self.right, self.y = left, right, y

    def y_at(self, x: float) -> float:
        return self.y


def picture(seed: int, w: int = 44, h: int = 40, ox: int = 22, oy: int = 38) -> dict:
    """A made-up monster frame (gamedata.canvas.mob_frames' dict): a blob of its own colours, opaque, its edge half seen
    (alpha 128), facing left; its feet at (ox, oy)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    r = ((xx - w / 2) / (w / 2 - 2)) ** 2 + ((yy - h / 2) / (h / 2 - 2)) ** 2
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = rng.integers(0, 256, (h, w, 3))
    rgba[..., 0] = np.clip(rgba[..., 0].astype(int) + 60 * (xx < w // 3), 0, 255)  # (one side redder: a facing)
    rgba[..., 3] = np.where(r <= 0.8, 255, np.where(r <= 1.0, 128, 0))
    return {"rgba": rgba, "origin": (ox, oy)}


def frames(seed: int = 1) -> list[dict]:
    """A kind's stand/0, move/0..2 and hit1/0 frames (made up)."""
    names = [("stand", 0), ("move", 0), ("move", 1), ("move", 2), ("hit1", 0)]
    return [{"action": a, "index": i, **picture(seed * 10 + k)} for k, (a, i) in enumerate(names)]


SIZE = (400, 300)   # (a small frame: the watch's arithmetic is the same at any size)
ART = Art(np.random.default_rng(7).integers(0, 256, (600, 800, 3), dtype=np.uint8), (-400, -300))
VIEW = (-200, -200)
FLOOR = Flat(-300.0, 300.0, -20.0)


def watch(use_c=None, points=None) -> SpawnWatch:
    pts = points or [Point("blob", -60.0, FLOOR, 1), Point("blob", 70.0, FLOOR, 1)]
    return SpawnWatch(pts, {"blob": frames()}, ART, exclude=((0, 0, 40, 40),), bottom=280, size=SIZE, use_c=use_c)


def scene(view=VIEW) -> np.ndarray:
    """The map's art at `view`: a frame with nothing over it."""
    x0, y0 = ART.origin
    vx, vy = view
    return np.ascontiguousarray(ART.rgb[vy - y0:vy - y0 + SIZE[1], vx - x0:vx - x0 + SIZE[0]])


def draw(frame: np.ndarray, frame_dict: dict, x: float, y: float, a: float, right: bool = False,
         view=VIEW) -> np.ndarray:
    """Frame with the picture drawn as the game blends a monster fading in: its feet at map (x, y), opacity a (F = B + a
    A / 255 (S - B), rounded)."""
    rgba = frame_dict["rgba"][:, ::-1] if right else frame_dict["rgba"]
    ox, oy = frame_dict["origin"]
    ox = rgba.shape[1] - ox if right else ox
    left, top = int(round(x - ox)) - view[0], int(round(y - oy)) - view[1]
    h, w = rgba.shape[:2]
    out = frame.copy()
    t0, l0, t1, l1 = max(top, 0), max(left, 0), min(top + h, out.shape[0]), min(left + w, out.shape[1])
    if t0 >= t1 or l0 >= l1:
        return out   # (none of it on the frame)
    sub = rgba[t0 - top:t1 - top, l0 - left:l1 - left]   # (what's on the frame of it)
    patch = out[t0:t1, l0:l1].astype(float)
    k = a * sub[..., 3:4] / 255.0
    out[t0:t1, l0:l1] = np.rint(patch + k * (sub[..., :3] - patch)).astype(np.uint8)
    return out


def needs_c():
    if native.spawns_lib() is None:
        pytest.skip(f"spawns.c isn't built: {native.spawns_why}")


# --- the sums ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("masked", [False, True])
def test_the_sums_in_c_are_numpys_to_the_integer(masked):
    """sp_sums against sums_numpy on random windows (3 or 4 bytes a pixel, a frame's sub-window with its row stride),
    runs of 1-8 lanes, with and without a mask; sp_empty against empty_numpy."""
    needs_c()
    rng = np.random.default_rng(3)
    pics = spawns.sampled(frames(), actions=("stand", "move"))
    w_c, w_np = watch(), watch(use_c=False)
    for channels in (3, 4):
        big = rng.integers(0, 256, (120, 160, channels), dtype=np.uint8)
        window = big[10:90, 20:130]          # (not contiguous: the frame's rows)
        plate = np.ascontiguousarray(rng.integers(0, 256, (80, 110, 3), dtype=np.uint8))
        places = [(x, y) for x in range(30, 90) for y in (55.0, 56.0, 57.0)]
        hyps = spawns._hypotheses(pics, range(len(pics.list)), places, (0, 0, 110, 80))
        assert sorted(set(hyps.runs[:, 3].tolist())) == [1, 2, 3, 4, 5, 6, 7, 8] or 8 in hyps.runs[:, 3]
        keep = (rng.random((80, 110)) > 0.3).astype(np.uint8) if masked else None
        c = w_c._sums(window, plate, keep, hyps, pics)
        n = spawns.sums_numpy(window, plate, keep, hyps, pics)
        assert c.dtype == np.int64 and np.array_equal(c, n)
        assert w_c._empty(window, plate) == spawns.empty_numpy(window, plate) == w_np._empty(window, plate)


def test_runs_are_a_pictures_hypotheses_a_column_apart_eight_at_most():
    pics = spawns.sampled(frames(), names=("stand/0",))
    hyps = spawns._hypotheses(pics, [0, 1], [(float(x), 50.0) for x in range(20, 42)] + [(25.0, 50.0)],
                              (0, 0, 100, 80))   # (feet at 20 and 21: the picture past the window's left edge)
    assert hyps.runs.tolist() == [[0, 12, 0, 8], [0, 12, 8, 8], [0, 12, 16, 4], [1, 12, 0, 8], [1, 12, 8, 8],
                                  [1, 12, 16, 4]]
    assert len(hyps.X) == 40 and hyps.X[:8].tolist() == [22.0 + k for k in range(8)]  # (25 twice: once)


# --- the watch on made-up pictures --------------------------------------------------------------------------------

@pytest.mark.parametrize("a", [0.04, 0.065, 0.1, 0.3, 0.6, 0.78])
def test_a_newborn_is_found_at_its_point_as_faint_as_it_shows(a):
    """A newborn's move/0 at its point, at the alphas the prototype first found Tree Dungeon's at (0.065 the median),
    up to A_START: its alpha to 0.005, the fit near perfect (only the frame's rounding off it)."""
    w = watch()
    got = w.look(draw(scene(), frames()[1], -60.0, -20.0, a), VIEW, 0.0)
    assert len(got) == 1
    s = got[0]
    assert (s.point, s.kind, s.x, s.y, s.picture.name, s.onset) == (0, "blob", -60.0, -20.0, "move/0", True)
    assert s.a == pytest.approx(a, abs=0.005) and 1 - s.r2 < spawns.U_MAX and s.res < 0.5   # (the frame's rounding)
    assert w.newborns and w.newborns[0].since == 0.0


@pytest.mark.parametrize("a", [0.85, 0.9])
def test_a_newborn_isnt_first_found_past_a_start_but_one_followed_is(a):
    """Over A_START a fit isn't a newborn's start (a solid monster under a faint shade fits nearly as one: the whole
    screen's from there); one found fainter is followed on past it, up to A_HI."""
    assert watch().look(draw(scene(), frames()[1], -60.0, -20.0, a), VIEW, 0.0) == []
    w = watch()
    assert len(w.look(draw(scene(), frames()[1], -60.0, -20.0, 0.75), VIEW, 0.0)) == 1
    got = w.look(draw(scene(), frames()[1], -60.0, -20.0, a), VIEW, 1 / 15)
    assert len(got) == 1 and not got[0].onset and got[0].a == pytest.approx(a, abs=0.005)


def test_what_isnt_a_newborns_blend_is_left_alone():
    """Nothing over the points; a monster drawn solid there (over A_HI: the whole screen's); another picture blended
    there (a fit to the point's pictures can't explain it); noise over the art; the art one px off (a view off)."""
    w = watch()
    assert w.look(scene(), VIEW, 0.0) == []
    assert w.look(draw(scene(), frames()[1], -60.0, -20.0, 1.0), VIEW, 0.1) == []
    stranger = {"action": "move", "index": 0, **picture(99)}
    assert w.look(draw(scene(), stranger, -60.0, -20.0, 0.3), VIEW, 0.2) == []
    noisy = np.clip(scene().astype(int) + np.random.default_rng(5).integers(-12, 13, (SIZE[1], SIZE[0], 3)), 0, 255)
    assert w.look(noisy.astype(np.uint8), VIEW, 0.3) == []
    assert w.look(draw(scene((-199, -200)), frames()[1], -59.0, -20.0, 0.3, view=(-199, -200)), VIEW, 0.4) == []
    assert w.counts["onsets"] == 0


def test_a_newborn_is_followed_as_it_fades_in_and_walks_until_the_whole_screen_finds_it():
    """Alpha 0.082 a frame (1.224 a second at 15), 6 units a frame to the right in its move frames: followed each
    frame (onset False, its first frame's time kept), its point not looked at for another meanwhile; once the whole
    screen finds a monster within HANDOVER_DX of it, that frame's sighting is dropped and it isn't followed."""
    w = watch()
    walk = frames()
    seen = []
    for k in range(7):
        a, x, t = 0.05 + 0.082 * k, -60.0 + 6 * k, k / 15
        got = w.look(draw(scene(), walk[1 + k % 3], x, -20.0, a, right=True), VIEW, t)
        assert len(got) == 1 and got[0].onset == (k == 0) and got[0].since == 0.0
        assert got[0].x == x and got[0].picture.right and got[0].a == pytest.approx(a, abs=0.005)
        seen.append(got[0])
    assert w.counts["onsets"] == 1 and w.counts["followed"] == 6
    got = w.look(draw(scene(), walk[2], -60.0 + 42, -20.0, 0.62, right=True), VIEW, 7 / 15)
    assert w.handed(got, [(-60.0 + 42 + 20, -20.0, "blob")]) == got and w.newborns  # (20 units off: another monster)
    assert w.handed(got, [(-60.0 + 45, -21.0, "blob")]) == [] and not w.newborns and w.counts["handed"] == 1
    assert w.look(draw(scene(), walk[3], -60.0 + 48, -20.0, 0.7, right=True), VIEW, 8 / 15) == []  # (left to it)


def test_a_monster_of_another_kind_by_a_newborn_has_the_frames_place_and_the_newborn_is_still_followed():
    """The whole screen's monster of another kind within HANDOVER_DX of a newborn: the frame's sighting left out (it
    may be the newborn taken for another kind: not two monsters there), the newborn followed on; one of its kind there
    is it (handed over)."""
    w = watch()
    got = w.look(draw(scene(), frames()[1], -60.0, -20.0, 0.3), VIEW, 0.0)
    assert w.handed(got, [(-56.0, -20.0, "other")]) == [] and len(w.newborns) == 1
    assert w.counts["passed"] == 1 and w.counts["handed"] == 0
    got = w.look(draw(scene(), frames()[1], -60.0, -20.0, 0.38), VIEW, 1 / 15)
    assert len(got) == 1 and not got[0].onset and w.handed(got, []) == got
    assert w.handed(got, [(-56.0, -20.0, "blob"), (-56.0, -20.0, "other")]) == [] and not w.newborns


def test_a_picture_fading_out_is_followed_no_further_and_its_place_left_till_the_fade_is_over():
    """A newborn's alpha may seem to fall DROP_A under the most it's been, frame to frame or over several: a picture
    fading out in place is followed no further than that, and its place isn't looked at for a newborn again for a whole
    fade (1 / FADE_PER_S), where it was found anew every other frame (the client fades none out but its die1/3, which
    never fits); a newborn there after that is found."""
    w = watch()
    alphas = [0.5, 0.46, 0.43, 0.4, 0.37, 0.3, 0.2, 0.1]
    seen = [w.look(draw(scene(), frames()[0], -60.0, -20.0, a), VIEW, k / 15) for k, a in enumerate(alphas)]
    assert [len(g) for g in seen] == [1, 1, 1, 0, 0, 0, 0, 0] and w.counts["onsets"] == 1 and w.counts["out"] == 1
    later = 3 / 15 + 1 / spawns.FADE_PER_S + 0.01
    got = w.look(draw(scene(), frames()[1], -60.0, -20.0, 0.1), VIEW, later)
    assert len(got) == 1 and got[0].onset


def test_a_newborn_lost_for_lost_after_frames_is_let_go():
    w = watch()
    assert len(w.look(draw(scene(), frames()[1], -60.0, -20.0, 0.1), VIEW, 0.0)) == 1
    assert w.look(scene(), VIEW, 1 / 15) == [] and w.newborns   # (one frame unfound: still followed)
    w.missed()
    assert not w.newborns and w.counts["lost"] == 1


def test_her_box_and_other_monsters_are_left_out_of_the_fit():
    """Her box over the newborn (more than 1 - MIN_USED of its pixels): not tried; over a part of it: found, from the
    rest. Another monster's box at the point (the frame before's, ON_POINT off it) left out the same way; one found on
    the point itself isn't (it may be the newborn)."""
    frame = draw(scene(), frames()[1], -60.0, -20.0, 0.2)
    assert watch().look(frame, VIEW, 0.0, her=(-60.0, -20.0)) == []
    part = watch().look(frame, VIEW, 0.0, her=(-60.0 - 60, -20.0))
    assert len(part) == 1 and part[0].a == pytest.approx(0.2, abs=0.005)
    assert watch().look(frame, VIEW, 0.0, others=[(-60.0 + 20, -20.0)]) == []
    assert len(watch().look(frame, VIEW, 0.0, others=[(-60.0 + 5, -20.0)])) == 1


def test_a_point_partly_off_the_screen_is_looked_at_in_what_shows():
    """A point whose window is past the screen's left edge or top: looked at in the pixels on the screen (the rest left
    out of the fit, the art in their place), its newborn found as long as MIN_USED of its picture shows, and followed
    there. One reaching under the game's windows or the chat box isn't (ONSET_COVERED: newborns there fit badly, and a
    solid monster cut by the chat box came nearest the rule); one out of view isn't either."""
    for x, floor, cut in ((-180.0, FLOOR, "left"), (0.0, Flat(-300, 300, -178.0), "top")):
        w = watch(points=[Point("blob", x, floor, 1)])
        frame = draw(scene(), frames()[1], x, floor.y, 0.2)
        got = w.look(frame, VIEW, 0.0)
        assert w.counts["points"] == w.counts["partly"] == 1, cut
        assert len(got) == 1 and got[0].a == pytest.approx(0.2, abs=0.005) and got[0].x == x, cut
        again = w.look(draw(scene(), frames()[1], x, floor.y, 0.28), VIEW, 1 / 15)
        assert len(again) == 1 and not again[0].onset and again[0].a == pytest.approx(0.28, abs=0.005), cut
        assert w.look(scene(), VIEW, 2 / 15) == [] and watch(points=w.points).look(scene(), VIEW, 0.0) == [], cut
    w = watch(points=[Point("blob", -210.0, FLOOR, 1)])   # (too little of it shows: under MIN_USED)
    assert w.look(draw(scene(), frames()[1], -210.0, -20.0, 0.2), VIEW, 0.0) == [] and w.counts["partly"] == 1
    for x, floor in ((-150.0, Flat(-300, 300, -157.0)), (0.0, Flat(-300, 300, 92.0)), (-280.0, FLOOR)):
        w = watch(points=[Point("blob", x, floor, 1)])   # (under the made-up window at 0-40, the bottom; out of view)
        assert w.look(draw(scene(), frames()[1], x, floor.y, 0.2), VIEW, 0.0) == [] and w.counts["points"] == 0


def test_a_sighting_is_a_detection_as_the_whole_screens_are():
    w = watch()
    s = w.look(draw(scene(), frames()[1], -60.0, -20.0, 0.3, right=True), VIEW, 0.0)[0]
    m = s.mob(VIEW)
    p = s.picture
    assert (m.kind, m.foot_x, m.feet_y, m.facing, m.pose, m.frame) == ("blob", 140.0, 180.0, 1, "move", "move/0")
    assert (m.w, m.h) == (p.w, p.h) and m.x == 140 - p.ox + p.w // 2 and m.fading == pytest.approx(0.3, abs=0.005)
    assert m.score >= 0.99   # (its fit's share explained: not looked at for its death, perception.deaths)


def test_the_watch_works_in_numpy_without_its_c():
    frame = draw(scene(), frames()[1], -60.0, -20.0, 0.1)
    a = watch(use_c=False).look(frame, VIEW, 0.0)
    assert len(a) == 1 and a[0].a == pytest.approx(0.1, abs=0.005)
    if native.spawns_lib() is not None:
        b = watch().look(frame, VIEW, 0.0)
        assert [(s.x, s.y, s.a, s.r2, s.res) for s in a] == [(s.x, s.y, s.a, s.r2, s.res) for s in b]


# --- Tree Dungeon's frames ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tree():
    from gamedata.paths import COPY
    if not (COPY / "Map" / "Map_000.wz").is_file():
        pytest.skip("needs the game data copy (gamedata/data/copy)")
    needs_c()
    from gamelab.localization import mapart, mapdata
    from gamelab.localization.camreg import EXCLUDE, HUD_TOP
    from gamelab.perception.kinds import kind_of
    model = mapdata.load(TREE)
    art = mapart.load(TREE)
    ids = {kind_of(sp.name): sp.mob_id for sp in model.spawns if sp.name}
    points = [Point(kind_of(sp.name), sp.x, p, p.id) for sp in model.spawns
              if (p := model.platform_under(sp.x, sp.y, reach=60)) is not None]   # (as the tracker puts them)

    def make() -> SpawnWatch:
        return SpawnWatch.of(points, ids, Art(art.rgb, (art.x0, art.y0), art.static), TREE, exclude=EXCLUDE,
                             bottom=HUD_TOP)
    return SimpleNamespace(model=model, art=art, make=make, index=json.loads((FIXTURES / "index.json").read_text()))


def _frame(tree, e) -> tuple[np.ndarray, tuple[int, int]]:
    """Fixture `e`'s frame: its crop pasted into the map's art at its view (nothing else drawn: the art is what the
    frame shows where nothing is drawn over it)."""
    view = tuple(e["view"])
    frame, _ = tree.art.window(*view)
    x0, y0, x1, y1 = e["box"]
    frame[y0:y1, x0:x1] = np.asarray(Image.open(FIXTURES / e["file"]).convert("RGB"))
    return frame, view


def _looked(w, e, frame, view, masks=True):
    got = w.look(frame, view, e["t"], tuple(e["her"]) if masks and e["her"] else None,
                 [tuple(o) for o in e["others"]] if masks else ())
    return [{"point": s.point, "a": round(s.a, 4), "x": s.x, "y": round(s.y, 2), "picture": s.picture.name,
             "right": s.picture.right, "onset": s.onset} for s in got]


def test_the_points_and_pictures_are_tree_dungeons(tree):
    w = tree.make()
    assert len(w.points) == 31 and {p.kind for p in w.points} == set(spawns.MEASURED[TREE])
    assert {k: (v.k, len(v.list)) for k, v in w.onset.items()} == {
        "horny-mushroom": (657, 4), "green-mushroom": (626, 4), "slime": (538, 4)}
    assert {k: len(v.list) for k, v in w.follow.items()} == {"horny-mushroom": 14, "green-mushroom": 14, "slime": 22}


def test_the_watch_is_for_the_maps_measured_alone(tree):
    """On a map not measured (the first map: its art and monsters not replayed so) none of its points, its kinds Tree
    Dungeon's or not; the frame step makes no watch there, nor on the simulator's eyes (nor counts newborns: the
    goldens' traces are as before the watch)."""
    from gamelab.perception.kinds import kind_of
    ids = {kind_of(sp.name): sp.mob_id for sp in tree.model.spawns if sp.name}
    art = Art(tree.art.rgb, (tree.art.x0, tree.art.y0), tree.art.static)
    assert SpawnWatch.of(tree.make().points, ids, art, 10000012).points == []
    from gamelab.localization.worldstate import WorldStep
    from gamelab.screen import NATIVE
    reg = SimpleNamespace(size=tuple(NATIVE.client), art=SimpleNamespace(rgb=art.rgb, x0=art.origin[0],
                                                                         y0=art.origin[1], static=art.known))
    step = SimpleNamespace(eyes=SimpleNamespace(geo=NATIVE), senses=SimpleNamespace(locator=SimpleNamespace(reg=reg)),
                           model=tree.model)
    assert len(WorldStep._spawn_watch(step).points) == 31
    step.model = SimpleNamespace(id=10000012, spawns=tree.model.spawns, platform_under=tree.model.platform_under)
    assert WorldStep._spawn_watch(step) is None
    plain = WorldStep(SimpleNamespace(eyes=None, model=tree.model))   # (the simulator's eyes: no watch, no count)
    assert plain.spawns is None and "newborns" not in plain.counts   # (its traces as before the watch)


@pytest.mark.parametrize("name", ["horny_a04", "green_a06", "horny_a11", "slime_a06", "green_a34", "horny_a53"])
def test_tree_dungeons_newborns_are_found_at_the_prototypes_alphas(tree, name):
    """Each a newborn the prototype's phase 2 found at its point (index.json's `found`: the same alpha, feet and picture
    as there), masked or not."""
    e = next(e for e in tree.index["single"] if e["name"] == name)
    frame, view = _frame(tree, e)
    assert e["label"] == "spawn_fade" and len(e["found"]) == 1
    assert _looked(tree.make(), e, frame, view) == e["found"] == _looked(tree.make(), e, frame, view, masks=False)


def test_a_tree_dungeon_newborn_first_seen_past_a_start_is_the_whole_screens(tree):
    """The prototype's newborn at point 3 first found at alpha 0.81 (its start hidden): over A_START, not a start of
    the watch's (the whole screen finds 93% of newborns at 0.8-0.9), masked or not."""
    e = next(e for e in tree.index["single"] if e["name"] == "horny_a81")
    frame, view = _frame(tree, e)
    assert e["label"] == "spawn_fade" and [f["a"] for f in e["found"]] == [0.8076] > [spawns.A_START]
    assert _looked(tree.make(), e, frame, view) == [] == _looked(tree.make(), e, frame, view, masks=False)


@pytest.mark.parametrize("name", ["near_monster", "near_monster2", "her_passing", "dying", "undetected_monster",
                                  "damage", "lie_detector", "player", "pet"])
def test_tree_dungeons_hard_negatives_never_fire(tree, name):
    """The nearest negatives to the rule of the 153,474 labelled point-frames (phase 1: the solid, partly covered
    monsters 11.8 times off it at their nearest, her passing, a monster dying, damage numbers, the Lie Detector, another
    player, her pet): nothing, with the masks and without."""
    e = next(e for e in tree.index["single"] if e["name"] == name)
    frame, view = _frame(tree, e)
    assert e["label"] != "spawn_fade" and e["found"] == []
    assert _looked(tree.make(), e, frame, view) == [] and _looked(tree.make(), e, frame, view, masks=False) == []


def test_a_tree_dungeon_newborn_is_followed_from_its_first_visible_frame(tree):
    """The golden clip's newborn at point 22 (frames 42-51: the mechanics' example), alpha 0.039 to 0.749 while it walks
    52 units left: the onset, then nine follows, each as the prototype's phase 2 had it."""
    seq = next(s for s in tree.index["sequences"] if s["name"] == "follow22")
    w = tree.make()
    alphas = []
    for e in seq["frames"]:
        frame, view = _frame(tree, e)
        got = _looked(w, e, frame, view)
        assert got == e["found"] and len(got) == 1
        alphas.append(got[0]["a"])
    assert alphas[0] == pytest.approx(0.039, abs=0.001) and alphas[-1] == pytest.approx(0.749, abs=0.001)
    assert all(b > a for a, b in zip(alphas, alphas[1:]))
    assert [r["onset"] for r in (e["found"][0] for e in seq["frames"])] == [True] + [False] * 9


def test_her_standing_on_a_spawn_point_for_seconds_doesnt_fool_it(tree):
    """She stood on point 14 for 6.6 s (the golden clip, frames 2618-2716: her body and name tag over it, a frame every
    0.8 s here): nothing found, with her box left out or not; the background is the map's own art, which nothing she
    does changes. The newborn there 7 frames after she left (2723 at alpha 0.02, under the empty test) is found at
    2724, alpha 0.098, the same as with no frame of her before, and followed."""
    seq = next(s for s in tree.index["sequences"] if s["name"] == "her14")
    w, bare = tree.make(), tree.make()
    plates = [o.plate.copy() for o in w._onsets if o.hyps is not None]
    for e in seq["frames"]:
        frame, view = _frame(tree, e)
        got = _looked(w, e, frame, view)
        assert got == e["found"]
        if e["label"] in ("her", "her_tag"):
            assert got == [] and _looked(tree.make(), e, frame, view, masks=False) == []
    assert [e["found"] for e in seq["frames"][-3:]][0] == []
    assert seq["frames"][-2]["found"] == [{"point": 14, "a": 0.0979, "x": 28.0, "y": -568.18, "picture": "move/0",
                                           "right": True, "onset": True}]
    e = seq["frames"][-2]
    frame, view = _frame(tree, e)
    assert _looked(bare, e, frame, view) == e["found"]   # (the same with nothing before it)
    assert all(np.array_equal(a, o.plate) for a, o in zip(plates, [o for o in w._onsets if o.hyps is not None]))


# --- the frame step and the tracker ---------------------------------------------------------------------------------

def _mob(x, fading=None, kind="blob"):
    return Mob(kind, int(x), 160, 44, 40, 0.99 if fading is not None else 0.9, float(x), 180.0, 1, "move", "move/0", "",
               fading)


def test_the_frame_step_adds_the_newborns_the_whole_screen_hasnt_found():
    """WorldStep._handed: this frame's newborns as detections beside the whole screen's, those it found too (its monster
    within HANDOVER_DX across and HANDOVER_DY down) left to it, the watch following them no more; the count of
    newborns first found."""
    from gamelab.localization.worldstate import WorldStep
    w = watch()
    a = w.look(draw(draw(scene(), frames()[1], -60.0, -20.0, 0.2), frames()[1], 70.0, -20.0, 0.2), VIEW, 0.0)
    assert [s.point for s in a] == [0, 1]
    step = SimpleNamespace(spawns=w, counts={"newborns": 0})
    s = SimpleNamespace(view=VIEW, map_x=lambda x: x + VIEW[0], map_y=lambda y: y + VIEW[1])
    work = {}
    whole = [_mob(70 - VIEW[0] + 4)]   # (the whole screen's monster 4 units from the newborn at 70)
    got = WorldStep._handed(step, s, a, whole, work)
    assert [m.foot_x + VIEW[0] for m in got] == [-60.0] and got[0].fading == pytest.approx(0.2, abs=0.005)
    assert work == {"newborns": 1, "newborns found": 1} and step.counts["newborns"] == 1
    assert [nb.point for nb in w.newborns] == [0]


def test_the_frame_step_looks_only_where_the_view_is_registered_and_her_placed():
    from gamelab.localization.worldstate import WorldStep
    w = watch()
    assert w.look(draw(scene(), frames()[1], -60.0, -20.0, 0.1), VIEW, 0.0)
    tracker = SimpleNamespace(matched={})
    step = SimpleNamespace(spawns=w, tracker=tracker)
    frame = draw(scene(), frames()[1], -60.0, -20.0, 0.18)
    unsure = SimpleNamespace(bad=None, view_sure=False, view=VIEW, frame=frame, where=SimpleNamespace(x=0.0, y=0.0))
    assert WorldStep._newborns(step, unsure, 1 / 15, True, {}) == [] and w.newborns[0].misses == 1
    sure = SimpleNamespace(bad=None, view_sure=True, view=VIEW, frame=frame, where=SimpleNamespace(x=200.0, y=-20.0))
    work = {}
    got = WorldStep._newborns(step, sure, 2 / 15, True, work)
    assert len(got) == 1 and not got[0].onset and "spawns s" in work
    assert WorldStep._newborns(SimpleNamespace(spawns=None), sure, 3 / 15, True, {}) == []


def test_the_watch_is_made_only_where_it_can_run():
    """At the game's own 1366x768 with the scene's locator (its registration has the map's art) and the C; never on
    the simulator's eyes or at 1920x1080."""
    from gamelab.localization.worldstate import WorldStep
    from gamelab.screen import NATIVE, STRETCHED
    model = SimpleNamespace(spawns=())
    no_locator = SimpleNamespace(eyes=SimpleNamespace(geo=NATIVE), senses=SimpleNamespace(locator=None), model=model)
    assert WorldStep._spawn_watch(no_locator) is None
    stretched = SimpleNamespace(eyes=SimpleNamespace(geo=STRETCHED), model=model,
                                senses=SimpleNamespace(locator=SimpleNamespace(reg=SimpleNamespace(size=(1366, 768)))))
    assert WorldStep._spawn_watch(stretched) is None


def test_a_newborn_is_one_track_from_its_first_sighting_its_fade_in_it():
    """A fading sighting is a new track 'spawned' (its alpha in the event), its `fading` and `solid_at`; the whole
    screen's sighting of it later the same track, its alpha on its fade's line until then; lost a frame after its
    first sighting, it isn't dropped as a flicker (it's a monster, not scenery); and a fading sighting never takes the
    track of a monster seen before its fade began."""
    from gamelab.localization.monsters import Monster
    from gamelab.localization.tracking import HIDDEN, MonsterTracker, View
    view = View(-200.0, 200.0, -300.0, 100.0, 400, 300)
    tr = MonsterTracker()
    ev = tr.update(0.0, [Monster(_mob(140, 0.1), -60.0, -20.0, 1)], view, None)
    assert [(e.kind, e.data.get("fading")) for e in ev] == [("spawned", 0.1)]
    (t1,) = tr.tracks.values()
    assert t1.fading == 0.1 and t1.solid_at == pytest.approx((1 - 0.1) / spawns.FADE_PER_S)
    assert t1.record()["fading"] == 0.1 and t1.opacity(0.0) == pytest.approx(0.1)
    tr.update(1 / 15, [Monster(_mob(141, 0.18), -59.0, -20.0, 1)], view, None)
    tr.update(2 / 15, [Monster(_mob(147), -53.0, -20.0, 1)], view, None)   # (the whole screen's: solid as it knows)
    assert list(tr.tracks) == [t1.id] and t1.fading == pytest.approx(0.18 + spawns.FADE_PER_S / 15, abs=0.002)
    tr.update(1.0, [Monster(_mob(150), -50.0, -20.0, 1)], view, None)
    assert t1.fading is None and t1.opacity(1.0) == 1.0
    lone = MonsterTracker()
    lone.update(0.0, [Monster(_mob(140, 0.1), -60.0, -20.0, 1)], view, None)
    for k in range(1, 8):
        lone.update(k / 15, [], view, None)
    assert lone.counts["flickers"] == 0 and [t.state for t in lone.tracks.values()] == [HIDDEN]
    old = MonsterTracker()
    for k in range(4):   # (a monster standing there for 0.2 s)
        old.update(k / 15, [Monster(replace(_mob(140), pose="stand"), -60.0, -20.0, 1)], view, None)
    old.update(1.0, [Monster(_mob(141, 0.1), -59.0, -20.0, 1)], view, None)   # (a newborn where it stood)
    assert old.counts["tracks"] == 2 and [t.how for t in old.tracks.values()] == ["appeared", "spawned"]
    again = MonsterTracker()   # (the newborn before at its point, hidden since: not the next one, 3 s on)
    for k in range(4):
        again.update(k / 15, [Monster(_mob(140 + k, 0.1 + 0.082 * k), -60.0 + k, -20.0, 1)], view, None)
    for k in range(4, 30):
        again.update(k / 15, [], view, None)
    assert [t.state for t in again.tracks.values()] == [HIDDEN]
    again.update(3.0, [Monster(_mob(140, 0.07), -60.0, -20.0, 1)], view, None)
    assert again.counts["tracks"] == 2 and [t.how for t in again.tracks.values()] == ["spawned", "spawned"]


def test_a_monsters_fade_is_in_its_record():
    from gamelab.localization.monsters import Monster
    from gamelab.localization.scene import _monster_record
    m = Monster(_mob(140, 0.25), -60.0, -20.0, 1)
    assert m.fading == 0.25 and _monster_record(m)["fading"] == 0.25
    assert "fading" not in _monster_record(Monster(_mob(140), -60.0, -20.0, 1))


def test_the_services_line_says_the_spawn_watchs_time():
    """service.Perf: the spawn watch's time a frame where it ran ('spawns s': median, p90, the most) and the newborns
    it found a minute ('newborns found'), on the 10 s line after the strips of drops; nothing of it where it didn't
    run."""
    from gamelab.localization import service
    from gamelab.localization.worldstate import WorldState
    perf = service.Perf(100.0)
    for k in range(150):
        t = 100 + k / 15
        perf.state(WorldState(k, t, True, work={"s": 0.012, "look s": 0.008, "wide s": 0.007, "detections": 3,
                                                "spawns s": 0.004 if k % 5 == 0 else 0.001, "newborns": 1,
                                                "newborns found": 1 if k % 15 == 0 else 0}), t + 0.013)
    r = perf.record(110.0, {"tracks": 0, "flickers": 0}, fps=15.0)
    assert r["spawns_ms"] == [1.0, 4.0, 4.0] and r["newborns_a_min"] == 60.0
    assert "0.0 strips of drops; the spawn watch 1.0 ms (p90 4.0, max 4.0), 60 newborns found a minute; a state" \
        in service.perf_line(r)
    empty = perf.record(120.0, {"tracks": 0, "flickers": 0})
    assert "spawns_ms" not in empty and "spawn watch" not in service.perf_line(empty)


def test_the_frame_steps_spawn_points_are_the_trackers(tree):
    """worldstate.spawn_points: the map's spawns on the platforms under them, as the tracker has them (its `spawned`)."""
    from gamelab.localization.tracking import MonsterTracker
    from gamelab.localization.worldstate import spawn_points
    pts = spawn_points(tree.model)
    assert [(p.kind, p.x, p.platform) for p in pts] == [
        (kind, x, plat.id) for kind, plat, x in MonsterTracker(tree.model).spawns if plat is not None]
    assert [(p.kind, p.x, p.platform) for p in pts] == [(p.kind, p.x, p.platform) for p in tree.make().points]
