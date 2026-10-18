import numpy as np
import pytest

from maplehelper.ocr.engine import rows_from, upscale
from maplehelper.ocr.regions import HUD, LOOT, Region, content_box


def test_region_box_scales_with_the_picture():
    assert LOOT.region.box((0, 0, 1000, 1000)) == (780, 570, 220, 290)
    assert LOOT.region.box((100, 50, 2000, 1000)) == (100 + 1560, 50 + 570, 440, 290)
    assert HUD.region.box((0, 0, 1920, 1080)) == (0, 1015, 1920, 65)


def test_region_box_is_never_empty():
    assert Region(0.5, 0.5, 0.5001, 0.5001).box((0, 0, 100, 100))[2:] == (1, 1)


def test_region_rejects_bad_fractions():
    with pytest.raises(ValueError):
        Region(0.5, 0, 0.4, 1)
    with pytest.raises(ValueError):
        Region(0, 0, 1, 1.2)


def test_region_crop():
    picture = np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3)
    crop = Region(0.5, 0.5, 1, 1).crop(picture)
    assert crop.shape == (50, 100, 3)
    assert (crop[0, 0] == picture[50, 100]).all()


def scene(h, w):
    rng = np.random.default_rng(0)
    return rng.integers(40, 255, size=(h, w, 3), dtype=np.uint8)


def test_content_box_trims_pillarbox_and_letterbox():
    picture = np.zeros((100, 200, 3), np.uint8)
    picture[:, 30:170] = scene(100, 140)
    assert content_box(picture) == (30, 0, 140, 100)
    picture = np.zeros((120, 200, 3), np.uint8)
    picture[10:110] = scene(100, 200)
    assert content_box(picture) == (0, 10, 200, 100)


def test_content_box_leaves_a_dark_edge_alone():
    picture = scene(100, 200)
    picture[:, :40] = 0  # dark scenery on one side only
    assert content_box(picture) == (0, 0, 200, 100)
    picture = scene(100, 200)
    picture[:, :2], picture[:, -2:] = 0, 0  # too thin to be bars
    assert content_box(picture) == (0, 0, 200, 100)


def test_upscale_only_grows_short_images():
    img = np.zeros((90, 300, 3), np.uint8)
    big, factor = upscale(img, 450)
    assert big.shape == (450, 1500, 3) and factor == 5
    same, factor = upscale(img, 60)
    assert same is img and factor == 1
    assert upscale(img, 0)[1] == 1


def test_rows_from_undoes_the_upscale_and_sorts_top_down():
    boxes = [[[0, 100], [50, 100], [50, 120], [0, 120]], [[10, 0], [90, 0], [90, 30], [10, 30]]]
    rows = rows_from(boxes, ("second", " first "), (0.9, 0.8), factor=2.0)
    assert [r.text for r in rows] == ["first", "second"]
    first = rows[0]
    assert (first.x, first.y, first.height, first.left, first.right) == (25, 7.5, 15, 5, 45)
    assert rows_from(None, None, None) == []


def test_rows_on_one_line_read_left_to_right():
    def box(x, y, h):
        return [[x, y], [x + 40, y], [x + 40, y + h], [x, y + h]]

    boxes = [box(300, 11, 20), box(10, 9, 23), box(150, 10, 20), box(20, 60, 20)]
    rows = rows_from(boxes, ("MP", "LV", "HP", "next line"), (1, 1, 1, 1))
    assert [r.text for r in rows] == ["LV", "HP", "MP", "next line"]
