"""The map's art drawn from the game data (localization.mapart) and the camera registered against it frame by frame
(localization.camreg, its costs in C: camreg.c). The frames: tests/fixtures/capture (Tree Dungeon, Forest Up North II,
1366x768: a good frame high on the map, and captures shifted off the client) and tests/fixtures/native (West Rocky
Mountain II, a map the camera moves across); the art needs the game data copy (gamedata/data/copy)."""

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamedata.canvas import pixels
from gamedata.wz import Canvas
from gamelab.localization import camreg, mapart
from gamelab.localization.camreg import Registration, costs_np, gray_of, shrink

FIXTURES = Path(__file__).with_name("fixtures")
TREE, ROCKY = 10002073, 10004021
GOOD = FIXTURES / "capture" / "tree_135315.png"   # her on platform 23, high up the map: view (-658, -948)
SHIFTED = [FIXTURES / "capture" / n for n in ("tree_164845_offset.png", "tree_234040_drawn.png",
                                              "tree_002902_drawn.png")]
ROCKY_FRAMES = [FIXTURES / "native" / n for n in ("n28_000.png", "n28_008.png", "native_first.png")]
ROCKY_VIEW = (-1477, 2167)  # its ropes at screen x 941 and 1174 are map x -536 and -303 (fixtures/native/README.md)


def needs_game_data():
    from gamedata.paths import COPY
    if not (COPY / "Map" / "Map_000.wz").is_file():
        pytest.skip("needs the game data copy (gamedata/data/copy)")


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    needs_game_data()
    return mapart.load(TREE, cache=tmp_path_factory.mktemp("mapart"))


@pytest.fixture(scope="module")
def rocky(tmp_path_factory):
    needs_game_data()
    return mapart.load(ROCKY, cache=tmp_path_factory.mktemp("mapart"))


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def window_frame(art, vx, vy, seed=0):
    """A frame of the art at view (vx, vy), with things drawn over it (bodies, names) and the HUD and minimap."""
    frame, _ = art.window(vx, vy)
    frame = frame.copy()
    rng = np.random.default_rng(seed)
    for _ in range(12):
        x, y = int(rng.integers(0, 1300)), int(rng.integers(0, 650))
        frame[y:y + 50, x:x + 40] = rng.integers(0, 256, 3, dtype=np.uint8)
    frame[700:] = 90
    frame[:200, :240] = 200
    return frame


# --- the art ---------------------------------------------------------------------------------------------------

def test_tree_dungeons_art_spans_every_view_the_camera_takes_and_is_all_fixed(tree):
    assert tree.extent == (-658, -2070, 708, 460)  # the view centred across 770 units, VR top to bottom
    assert tree.fixed == (-658, None) and tree.views == (-658, -658, -2070, -308)
    assert tree.left_out == {} and tree.static.all()  # its backgrounds move with it (rx = ry = -100)


def test_backgrounds_that_dont_move_with_the_map_are_drawn_but_left_out(rocky):
    assert mapart.back_moves(-100, -100, 2) and not mapart.back_moves(0, 0, 0)
    assert not mapart.back_moves(-100, -100, 4)  # (it moves by itself)
    assert rocky.left_out == {"parallax": 18} and rocky.fixed == (None, None)
    assert 0.5 < rocky.static.mean() < 0.7  # the sky behind its rocks


def test_the_art_is_the_frame_wherever_nothing_is_drawn_over_it(tree):
    frame = rgb(GOOD)
    art, ok = tree.window(-658, -948)
    ok[700:] = False
    ok[:210, :250] = False
    d = np.abs(frame.astype(np.int16) - art.astype(np.int16)).max(axis=2)[ok]
    assert (d <= 2).mean() > 0.95  # the rest: players, monsters, names, the party's HP box


def test_a_flipped_picture_is_mirrored_about_its_origin():
    canvas = Canvas(3, 1, 2, {"origin": type("V", (), {"x": 1, "y": 0})()})
    rgba = np.arange(12, dtype=np.uint8).reshape(1, 3, 4)
    assert mapart._placed(canvas, rgba, 10, 5, False)[:2] == (9, 5)
    x, y, drawn = mapart._placed(canvas, rgba, 10, 5, True)
    assert (x, y) == (8, 5) and drawn[0, 0, 0] == 8  # its origin's column stays on x 10: 8 + (3 - 1) - 1


def test_a_picture_gamedata_cant_read_counts_as_unknown(monkeypatch):
    def unreadable(canvas):
        raise ValueError("picture format 1026 isn't supported yet")
    monkeypatch.setattr("gamedata.canvas.pixels", unreadable)
    picture = Canvas(4, 2, 1026)
    rgba, read = mapart._rgba(picture, {}, Path("."))
    assert not read and rgba.shape == (2, 4, 4) and (rgba[..., 3] == 255).all()


def test_format_517_is_one_rgb565_value_a_16_by_16_block():
    values = np.array([[0xF800, 0x07E0]], "<u2")   # red, green
    packed = zlib.compress(values.tobytes())
    blob = bytes([0]) + bytes(4) + struct.pack("<i", len(packed) + 1) + bytes([0]) + packed
    rgba = pixels(Canvas(32, 16, 517, raw=(blob, 0, len(blob))))
    assert rgba.shape == (16, 32, 4) and (rgba[..., 3] == 255).all()
    assert rgba[0, 0, :3].tolist() == [255, 0, 0] and rgba[15, 15, :3].tolist() == [255, 0, 0]
    assert rgba[0, 16, :3].tolist() == [0, 255, 0] and rgba[15, 31, :3].tolist() == [0, 255, 0]


def test_the_art_is_cached_and_drawn_again_for_another_version(tmp_path, monkeypatch):
    needs_game_data()
    first = mapart.load(TREE, cache=tmp_path)
    monkeypatch.setattr(mapart, "draw", lambda *a: pytest.fail("drawn again"))
    again = mapart.load(TREE, cache=tmp_path)
    assert np.array_equal(again.rgb, first.rgb) and again.views == first.views
    monkeypatch.setattr(mapart, "VERSION", mapart.VERSION + 1)
    with pytest.raises(pytest.fail.Exception, match="drawn again"):
        mapart.load(TREE, cache=tmp_path)


# --- the costs in C and in numpy -----------------------------------------------------------------------------------

def test_the_c_costs_and_gray_are_numpys_to_the_integer():
    lib = camreg.lib()
    if lib is None:
        pytest.skip(f"camreg.c isn't built: {camreg.why}")
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, (90, 211, 4), dtype=np.uint8)
    rows = np.array([0, 3, 7, 50, 89], np.int32)
    art = rng.integers(0, 256, (140, 260), dtype=np.uint8)
    am = np.where(rng.random((140, 260)) < 0.8, 255, 0).astype(np.uint8)
    xs = np.array([-30, -5, 0, 7, 49, 100, 250, 300], np.int32)  # some views part or all off the art
    ys = np.array([-60, 0, 13, 50, 20, -100, 30, 0], np.int32)
    for use in (1, 0):
        assert lib.cr_simd(use) == (use and lib.cr_simd(1))
        lib.cr_simd(use)
        for cn in (3, 4):
            f = np.ascontiguousarray(frame[..., :cn])
            out = np.empty((len(rows), f.shape[1]), np.uint8)
            assert lib.cr_gray(f.ctypes.data, f.shape[0], f.shape[1], f.strides[0], cn, rows.ctypes.data, len(rows),
                               out.ctypes.data) == 0
            assert np.array_equal(out, gray_of(f[rows]))
        g = gray_of(frame[rows])
        gm = np.where(rng.random(g.shape) < 0.9, 255, 0).astype(np.uint8)
        sums, counts = np.zeros(len(xs), np.int64), np.zeros(len(xs), np.int64)
        assert lib.cr_costs(g.ctypes.data, gm.ctypes.data, len(rows), g.shape[1], rows.ctypes.data, art.ctypes.data,
                            am.ctypes.data, art.shape[0], art.shape[1], art.strides[0], -20, -40, xs.ctypes.data,
                            ys.ctypes.data, len(xs), camreg.TRUNC, sums.ctypes.data, counts.ctypes.data) == 0
        want = costs_np(g, gm, rows, art, am, -20, -40, xs, ys)
        assert np.array_equal(sums, want[0]) and np.array_equal(counts, want[1])
        assert counts[6] == counts[7] == 0 < counts[5] < counts[2]  # (off the art, part on it)
    lib.cr_simd(1)


def test_gray_is_the_same_for_rgb_and_bgr_and_blocks_are_rounded_means():
    p = np.array([[[10, 200, 31], [31, 200, 10], [0, 0, 255]]], np.uint8)
    assert gray_of(p).tolist() == [[111, 111, 64]]  # avg(avg(10, 31), 200) = avg(21, 200); avg(avg(0, 255), 0)
    g, m = shrink(np.array([[1, 2, 9], [3, 4, 9]], np.uint8), np.array([[255, 255, 0], [255, 0, 0]], np.uint8), 2)
    assert g.tolist() == [[3]] and m.tolist() == [[0]]  # (10 + 2) // 4; a block with a px out
    lib = camreg.lib()
    if lib is not None:  # (the C's the same bytes)
        big = np.random.default_rng(5).integers(0, 256, (703, 1366), dtype=np.uint8)
        assert np.array_equal(shrink(big, None, 4, lib)[0], shrink(big, None, 4)[0])
        assert shrink(big, None, 4, lib)[0].shape == (175, 341)


# --- registration ----------------------------------------------------------------------------------------------------

def test_a_good_frame_is_placed_exactly_and_shifted_captures_are_not(tree):
    reg = Registration(tree)
    view = reg.see(rgb(GOOD))
    assert (view.x, view.y, view.sure, view.how) == (-658, -948, True, "cold")
    assert type(view.x) is int and type(view.y) is int  # (plain numbers: the scene's telemetry is JSON)
    assert view.cost < 2 and view.margin > 2 and view.box == (-658, -948, 708, -180)
    for path in SHIFTED:  # the game drawn off the capture: no view fits, it's carried
        shifted = reg.see(rgb(path))
        assert not shifted.sure and shifted.how == "carried" and (shifted.x, shifted.y) == (-658, -948)
        assert shifted.margin < camreg.MARGIN_OK / 5
    assert reg.see(rgb(GOOD)).how == "near"  # (and back where it was at once)


def test_the_view_is_followed_frame_by_frame_and_found_again_after_frames_that_arent_the_map(tree):
    reg = Registration(tree)
    ys = [-700, -700, -706, -715, -727, -742, -760, None, -830, -830, None, -831]  # None: a black frame
    hows = []
    for i, y in enumerate(ys):
        frame = window_frame(tree, -658, y, seed=i) if y is not None else np.zeros((768, 1366, 3), np.uint8)
        view = reg.see(frame, t=i / 30)
        hows.append(view.how)
        if y is None:
            assert not view.sure and view.how == "carried"
        else:
            assert view.sure and (view.x, view.y) == (-658, y), i
    # 34 px up from where it was going (-796 by its 18 a frame), then 35 down from where that put it: wide
    assert hows == ["cold"] + ["near"] * 6 + ["carried", "wide", "wide", "carried", "near"]
    assert reg.counts == {"near": 7, "wide": 2, "cold": 1, "carried": 2}


def test_a_view_just_past_a_search_is_found_not_its_edge(tree):
    """The cheapest view of a window can be its edge, a px short of the true view: it's settled to a view no
    neighbour of which costs less (SETTLE steps), else the next search goes on."""
    for off, how in ((13, "near"), (camreg.NEAR + camreg.SETTLE + 1, "wide")):
        reg = Registration(tree)
        reg.sure_at = (-658, -1200, None)
        view = reg.see(window_frame(tree, -658, -1200 - off))
        assert (view.x, view.y, view.sure, view.how) == (-658, -1200 - off, True, how)


def test_numpy_alone_places_the_frame_as_the_c_does(tree):
    frame = window_frame(tree, -658, -1200)
    a, b = Registration(tree, use_c=False), Registration(tree)
    a.sure_at = b.sure_at = (-658, -1203, None)
    va, vb = a.see(frame), b.see(frame)
    assert (va.x, va.y, va.cost, va.margin) == (vb.x, vb.y, vb.cost, vb.margin) == (-658, -1200, vb.cost, vb.margin)


def test_on_a_map_the_camera_moves_across_the_view_is_found_both_ways(rocky):
    if camreg.lib() is None:
        pytest.skip(f"camreg.c isn't built: {camreg.why}")  # (numpy's cold search of the whole map takes seconds)
    reg = Registration(rocky)
    for path in ROCKY_FRAMES:  # the real frames: her, other players, the clouds the map draws over itself
        reg.sure_at = None
        view = reg.see(rgb(path), hint=(ROCKY_VIEW[0] + 40, ROCKY_VIEW[1] - 50))
        assert (view.x, view.y, view.sure) == (*ROCKY_VIEW, True)
    reg = Registration(rocky)
    path = [(-1400, 2000), (-1395, 2000), (-1388, 1996), (-1380, 1990), (-1371, 1990), (-1362, 1992)]
    for i, (x, y) in enumerate(path):
        view = reg.see(window_frame(rocky, x, y, seed=i), hint=(x - 30, y + 30) if i == 0 else None)
        assert (view.x, view.y, view.sure) == (x, y, True)


def test_a_frame_of_another_size_is_refused_and_none_placed_before_the_first_sure_one(tree, monkeypatch):
    with pytest.raises(ValueError, match="1366x768"):
        Registration(tree).see(np.zeros((540, 960, 3), np.uint8))
    first = Registration(tree).see(np.zeros((768, 1366, 3), np.uint8))
    assert (first.x, first.y, first.sure, first.how) == (0, 0, False, "none")
    assert camreg.for_map(TREE, (960, 540)) is None  # (1920x1080's half-size look frames)
    monkeypatch.setattr(mapart, "load", lambda map_id: tree)
    assert camreg.for_map(TREE).art is tree


def test_a_lost_view_searches_the_whole_of_a_map_the_camera_moves_both_ways_seldom(rocky, tree):
    """The whole of such a map unhinted is 80-900 ms (the frame loop's 33 a frame): with her dot's view as the hint
    each lost frame searches about it alone, the whole map every COLD_WHOLE_S (a misread dot's hint found nothing);
    without one at once, then as seldom; where the camera moves one way only (Tree Dungeon, 4-6 ms) every lost frame."""
    if camreg.lib() is None:
        pytest.skip(f"camreg.c isn't built: {camreg.why}")
    calls = []

    def spied(reg):
        cold = reg.cold
        reg.cold = lambda frame, g, hint=None: calls.append(hint is not None) or cold(frame, g, hint)
        return reg

    real = rgb(ROCKY_FRAMES[0])
    black = np.zeros_like(real)
    reg = spied(Registration(rocky))
    assert reg.see(real, t=0.0, hint=(ROCKY_VIEW[0] + 40, ROCKY_VIEW[1] - 50)).sure
    calls.clear()
    for i in range(1, 31):   # a second of frames that aren't the map, her dot read
        assert reg.see(black, t=i / 30, hint=ROCKY_VIEW).how == "carried"
    assert calls == [True] * 30
    calls.clear()
    for i in range(31, 91):   # two more seconds
        reg.see(black, t=i / 30, hint=ROCKY_VIEW)
    assert calls.count(True) == 60 and calls.count(False) == 1
    assert reg.see(real, t=91 / 30, hint=ROCKY_VIEW).how == "near"   # (and found again at once)
    calls.clear()
    for i in range(92, 122):   # her dot not read: the whole map at once, then not for COLD_WHOLE_S
        reg.see(black, t=i / 30)
    assert calls == [False]
    assert reg.see(real, t=122 / 30).how == "near"
    far = (ROCKY_VIEW[0] + 300, ROCKY_VIEW[1] - 300)   # a hint off the view (a misread dot)
    reg = spied(Registration(rocky))
    assert reg.see(real, t=0.0, hint=far).how == "none"
    assert reg.see(real, t=camreg.COLD_WHOLE_S, hint=far).how == "cold"   # (the whole map searched, found)
    reg = spied(Registration(tree))
    assert reg.see(window_frame(tree, -658, -1200), t=0.0).sure
    calls.clear()
    for i in range(1, 6):
        reg.see(np.zeros((768, 1366, 3), np.uint8), t=i / 30, hint=(-658, -1200))
    assert calls == [False] * 5


def test_the_latest_maps_art_alone_is_kept_in_memory(monkeypatch, tmp_path):
    """A service going map after map kept every one's art for its life (Tree Dungeon 14 MB, Henesys 33, Ellinia 56)."""
    def draw(map_id, copy=None):
        return mapart.MapArt(map_id, 0, 0, np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), bool))
    monkeypatch.setattr(mapart, "draw", draw)
    monkeypatch.setattr(mapart, "_key", lambda map_id, copy: {"map": map_id})
    monkeypatch.setattr(mapart, "CACHE", tmp_path)
    monkeypatch.setattr(mapart, "_LOADED", {})
    a = mapart.load(1, cache=tmp_path)
    assert mapart.load(1, cache=tmp_path) is a and list(mapart._LOADED) == [1]
    mapart.load(2, cache=tmp_path)
    assert list(mapart._LOADED) == [2]
    again = mapart.load(1, cache=tmp_path)   # (from its file)
    assert again is not a and again.map_id == 1 and list(mapart._LOADED) == [1]
    mapart.load(3, cache=tmp_path / "other")   # another cache's (a test's, a tool's): not kept
    assert list(mapart._LOADED) == [1]
    mapart.forget()
    assert mapart._LOADED == {}
