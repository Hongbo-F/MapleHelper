"""The second map, West Rocky Mountain II (10004021), from its game data: where her minimap dot places her, the
platforms that are one ground, and the routes over its stairs and floating steps. Each test needs the map's parsed
game data (gamedata/data/out/maps), as the first map's do."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab.localization import mapdata
from gamelab.localization.pose import Localizer
from gamelab.perception import minimap
from gamelab.perception.minimap import MinimapPicture
from gamelab.screen import NATIVE, of_frame

FIXTURES = Path(__file__).parent / "fixtures"
ROCKY = 10004021
FIRST = 10000012


def load(map_id: int = ROCKY) -> mapdata.MapModel:
    if not (mapdata.CACHE / f"{map_id}.json").is_file():
        pytest.skip("needs the map's parsed game data (gamedata/data/out/maps)")
    return mapdata.load(map_id)


def crop(name: str) -> np.ndarray:
    """A saved minimap (MINIMAP_BOX, RGB): tests/test_minimap.py's crop."""
    return np.asarray(Image.open(FIXTURES / f"minimap_{name}.png").convert("RGB")).copy()


# --- where her dot places her ----------------------------------------------------------------------------------

def picture(map_id: int = ROCKY) -> np.ndarray:
    pic = mapdata.minimap_picture(map_id)
    if pic is None:
        pytest.skip("needs the game data copy (gamedata/data/copy)")
    return pic


def test_her_dot_places_her_where_she_stood_once_the_minimap_picture_is_found():
    # On the floating step 24 (-1013..-877, both its ends on screen) the ropes at -1463 and -303 in view put her at
    # -938; on stair 29 (-923..-833) she stood near its left end. Read as if the picture were where the first map's is
    # (minimap.MinimapPicture: here it's 5 game px further left), her dot reads 80 units further right than with the
    # first map's origin alone, which put her at 24's left end (-1013, clamped) and 38 left of stair 29, on no platform.
    model = load()
    for name, pid, x in (("rocky", 24, -946), ("rocky_stairs", 29, -881)):
        box = crop(name)
        dot = minimap.dot_in_box(box)
        before = Localizer(model).where_dot(dot)
        finder = MinimapPicture(picture())
        assert finder.find(box) is not None
        sx, sy = finder.shift()
        where = Localizer(model).where_dot((dot[0] - sx, dot[1] - sy))
        assert where.platform is not None and where.platform.id == pid and where.x == pytest.approx(x, abs=1)
        assert Localizer(model).to_map((dot[0] - sx, dot[1]))[0] - Localizer(model).to_map(dot)[0] == pytest.approx(80)
        assert before.platform is None or before.platform.id != pid or before.x < where.x - 60


def test_at_1366x768_the_picture_is_found_and_her_dot_places_her_on_stair_28():
    # The game at its own size (tests/fixtures/native: she stands on stair 28, -833..-743, where the ropes in view at
    # screen x 941 and 1174 put her name tag at map x -792): the picture's corner at game px (42, 72), 5 left of where
    # the first map's is (minimap.PICTURE_BASE, 47 and 72). Her dot then reads -800 on 28; with the first map's origin
    # alone, -880, on no platform.
    model = load()
    for name in ("n28_000", "n28_008", "native_first"):
        full = np.asarray(Image.open(FIXTURES / "native" / f"{name}.png").convert("RGB"))
        geo = of_frame(full)
        x0, y0, x1, y1 = minimap.minimap_box(geo)
        finder = MinimapPicture(picture(), geo)
        assert geo == NATIVE and finder.find(full[y0:y1, x0:x1]) == (42, 72)
        assert MinimapPicture(picture(FIRST), geo).find(full[y0:y1, x0:x1]) is None  # the first map's picture
        dot = minimap.her_dot(full, geo)
        loc = Localizer(model, geo=geo)
        assert loc.where_dot(dot).platform is None
        sx, sy = finder.shift()
        where = loc.where_dot((dot[0] - sx, dot[1] - sy))
        assert where.platform.id == 28 and where.x == pytest.approx(-800, abs=1) and abs(where.x + 792) < 16


def test_on_the_first_map_the_picture_is_where_the_origin_was_fitted():
    load(FIRST)
    for name in ("stranger", "party"):
        finder = MinimapPicture(picture(FIRST))
        assert finder.find(crop(name)) is not None and finder.shift() == (0.0, 0.0)

