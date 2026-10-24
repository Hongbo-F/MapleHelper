"""Where the map's own minimap picture is in the window (minimap.MinimapPicture), and her dot read by it: on a map
taller than the window shows, the picture scrolls with her."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab.localization import mapdata
from gamelab.localization.pose import Localizer
from gamelab.perception import minimap
from gamelab.perception.minimap import MinimapPicture, minimap_box
from gamelab.screen import Geometry, STRETCHED

NATIVE = Geometry((1366, 768))
FRAMES = Path("D:/MapleHelper-scratch/map3/native")  # Tree Dungeon, Forest Up North II at 1366x768 (no keys)


def _picture(seed=1, size=(183, 67)):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (size[0] // 6 + 1, size[1] // 6 + 1, 3)).astype(np.uint8)
    return np.asarray(Image.fromarray(small).resize((size[1], size[0]), Image.NEAREST))


def test_a_scrolled_picture_is_found_where_it_starts_above_the_window():
    pic = _picture()
    x0, y0, x1, y1 = minimap_box(NATIVE)
    box = np.full((y1 - y0, x1 - x0, 3), 119, np.uint8)
    # the window shows rows 60..171 of the picture, its left edge 40 px in, from the box's row 4
    box[4:116, 40:107] = pic[60:172]
    found = MinimapPicture(pic, NATIVE).find(box)
    assert found == (x0 + 40, y0 + 4 - 60)


def test_the_next_look_finds_it_scrolled_on_and_a_covered_minimap_finds_nothing():
    pic = _picture(2)
    x0, y0, x1, y1 = minimap_box(NATIVE)
    finder = MinimapPicture(pic, NATIVE)
    for scroll in (20, 26, 31):
        box = np.full((y1 - y0, x1 - x0, 3), 119, np.uint8)
        box[4:116, 40:107] = pic[scroll:scroll + 112]
        assert finder.find(box) == (x0 + 40, y0 + 4 - scroll)
    assert finder.find(np.full((y1 - y0, x1 - x0, 3), 119, np.uint8)) is None
    assert finder.corner == (x0 + 40, y0 + 4 - 31)  # the last place it was seen stays


def test_the_shift_is_none_where_the_first_map_had_its_picture():
    finder = MinimapPicture(_picture(), NATIVE)
    finder.corner = (minimap.PICTURE_BASE[0], minimap.PICTURE_BASE[1])
    assert finder.shift() == (0.0, 0.0)
    stretched = MinimapPicture(_picture(), STRETCHED)
    stretched.corner = (STRETCHED.from_game(minimap.PICTURE_BASE[0]) + 0.4, STRETCHED.from_game(minimap.PICTURE_BASE[1]))
    assert stretched.shift() == (0.0, 0.0)  # by whole game px


@pytest.mark.skipif(mapdata.minimap_picture(10002073) is None or not FRAMES.is_dir(),
                    reason="needs the game data copy and the saved frames of Tree Dungeon, Forest Up North II")
def test_her_dot_falls_on_the_platform_she_stands_on_where_the_minimap_scrolls():
    model = mapdata.load(10002073)
    pic = mapdata.minimap_picture(10002073)
    x0, y0, x1, y1 = minimap_box(NATIVE)
    checked = 0
    for name, platform in (("here_000.png", 28), ("here_039.png", 28), ("first_000.png", 19)):
        path = FRAMES / name
        if not path.is_file():
            continue
        full = np.asarray(Image.open(path).convert("RGB"))
        finder = MinimapPicture(pic, NATIVE)
        assert finder.find(full[y0:y1, x0:x1]) is not None
        dot = minimap.her_dot(full, NATIVE)
        sx, sy = finder.shift()
        where = Localizer(model, geo=NATIVE).where_dot((dot[0] - sx, dot[1] - sy))
        assert where.platform is not None and where.platform.id == platform, (name, where)
        checked += 1
    assert checked


def test_on_a_map_narrower_than_the_view_the_camera_is_centred_and_her_tag_places_her():
    from types import SimpleNamespace
    from gamelab.localization.pose import VIEW, Camera, Where, camera_x, refine
    bounds = (-360, -2070, 410, 460)  # Tree Dungeon, Forest Up North II: 770 wide
    camera = Camera(bounds)
    assert camera.fixed() == (True, False)
    assert camera.corner(0, -1000)[0] == camera.corner(300, -50)[0] == pytest.approx(25 - VIEW[0] / 2)
    assert camera.corner(0, -1000)[1] != camera.corner(0, -50)[1]  # up and down it follows her
    assert not Camera((-479, -1387, 1469, 765)).fixed()[0]  # the first map: wider than the view
    model = mapdata.load(10002073) if mapdata.minimap_picture(10002073) is not None else None
    if model is None:
        pytest.skip("needs the game data copy")
    p = next(pl for pl in model.platforms if pl.id == 28)
    s = SimpleNamespace(tag=(461, 270), ground=(443, 475), geo=NATIVE, frame=np.zeros((768, 1366, 3), np.uint8),
                        ropes=[])
    where = Where(262.0, p.y_at(262.0), p, (93.5, 202.5))
    assert camera_x(where, s, camera) == pytest.approx(25 - VIEW[0] / 2 + NATIVE.to_game(922))
    refined, how = refine(where, s, model, camera)
    assert how == "camera" and refined.x == pytest.approx(264.3, abs=0.5)
