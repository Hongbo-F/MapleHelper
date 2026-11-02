"""Monsters fading in at their spawn points (perception.spawns): the blend fit's integer sums in C and in numpy the
same; a newborn drawn over the map's art found at its first faint frames, followed, and handed over to the whole screen;
what's drawn over a point that isn't a newborn left alone. On made-up pictures over a made-up map's art, and on crops of
Tree Dungeon's lossless frames (tests/fixtures/spawns: the golden clip of the user's own play and the Lie Detector's
alarm clips; index.json says what each is and what the watch finds there) pasted into the map's art at their view:
newborns at the alphas the prototype found them at (D:/MapleHelper-scratch/spawn-fade/proto), the nearest negatives to
the rule of every labelled point-frame, and her standing on a spawn point for 6.6 s before the newborn there."""

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
    patch = out[top:top + h, left:left + w].astype(float)
    k = a * rgba[..., 3:4] / 255.0
    out[top:top + h, left:left + w] = np.rint(patch + k * (rgba[..., :3] - patch)).astype(np.uint8)
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
    clear = spawns._hypotheses(pics, [0], [(float(x), 50.0) for x in range(20, 40)], (0, 0, 100, 80),
                               (0, 0, ((0, 0, 10, 100),)))   # (a window over the first 10 columns on screen)
    assert clear.runs[:, 2].min() >= 10


# --- the watch on made-up pictures --------------------------------------------------------------------------------

@pytest.mark.parametrize("a", [0.04, 0.065, 0.1, 0.3, 0.6, 0.9])
def test_a_newborn_is_found_at_its_point_as_faint_as_it_shows(a):
    """A newborn's move/0 at its point, at the alphas the prototype first found Tree Dungeon's at (0.065 the median),
    up to 0.9: its alpha to 0.005, the fit near perfect (only the frame's rounding off it)."""
    w = watch()
    got = w.look(draw(scene(), frames()[1], -60.0, -20.0, a), VIEW, 0.0)
    assert len(got) == 1
    s = got[0]
    assert (s.point, s.kind, s.x, s.y, s.picture.name, s.onset) == (0, "blob", -60.0, -20.0, "move/0", True)
    assert s.a == pytest.approx(a, abs=0.005) and 1 - s.r2 < spawns.U_MAX and s.res < 0.5   # (the frame's rounding)
    assert w.newborns and w.newborns[0].since == 0.0


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
    assert w.handed(got, [(-60.0 + 42 + 20, -20.0)]) == got and w.newborns   # (one 20 units off: another monster)
    assert w.handed(got, [(-60.0 + 45, -21.0)]) == [] and not w.newborns and w.counts["handed"] == 1
    assert w.look(draw(scene(), walk[3], -60.0 + 48, -20.0, 0.7, right=True), VIEW, 8 / 15) == []  # (left to it)


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


def test_a_point_not_wholly_in_the_play_area_or_under_the_games_windows_isnt_looked_at():
    w = watch(points=[Point("blob", -175.0, FLOOR, 1)])   # (its window past the screen's left edge)
    assert w.look(draw(scene(), frames()[1], -175.0, -20.0, 0.2), VIEW, 0.0) == [] and w.counts["points"] == 0
    under = watch(points=[Point("blob", -165.0, Flat(-300, 300, -150.0), 1)])   # (under the made-up window at 0-40)
    assert under.look(scene(), VIEW, 0.0) == [] and under.counts["points"] == 0
    low = watch(points=[Point("blob", 0.0, Flat(-300, 300, 90.0), 2)])   # (its feet under the play area's bottom)
    assert low.look(scene(), VIEW, 0.0) == [] and low.counts["points"] == 0


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
        return SpawnWatch.of(points, ids, Art(art.rgb, (art.x0, art.y0), art.static), exclude=EXCLUDE, bottom=HUD_TOP)
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
    assert len(w.points) == 31 and {p.kind for p in w.points} == set(spawns.MEASURED)
    assert {k: (v.k, len(v.list)) for k, v in w.onset.items()} == {
        "horny-mushroom": (657, 4), "green-mushroom": (626, 4), "slime": (538, 4)}
    assert {k: len(v.list) for k, v in w.follow.items()} == {"horny-mushroom": 14, "green-mushroom": 14, "slime": 22}


@pytest.mark.parametrize("name", ["horny_a04", "green_a06", "horny_a11", "slime_a06", "green_a34", "horny_a53",
                                  "horny_a81"])
def test_tree_dungeons_newborns_are_found_at_the_prototypes_alphas(tree, name):
    """Each a newborn the prototype's phase 2 found at its point (index.json's `found`: the same alpha, feet and picture
    as there), masked or not."""
    e = next(e for e in tree.index["single"] if e["name"] == name)
    frame, view = _frame(tree, e)
    assert e["label"] == "spawn_fade" and len(e["found"]) == 1
    assert _looked(tree.make(), e, frame, view) == e["found"] == _looked(tree.make(), e, frame, view, masks=False)


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
