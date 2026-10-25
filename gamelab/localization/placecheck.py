"""Whether a place read off her minimap dot is on the map at all: a look that puts her off it isn't acted on
(Senses takes it again; perception.framecheck has the checks of the frame itself).

At 16:46:50 on 2026-10-24 every one of a hunt's 1,148 looks put her 92-956 units right of Tree Dungeon's right edge
(x 501.6-1365.6 on a map whose camera stops at 410), from its first look: her dot was read without the map's minimap
picture, with the first map's origin (perception.minimap.MinimapPicture), and the hunt sent a plan off a rope every
5 s for two minutes. On every look of that day's other tours there (ten runs from 13:49 to 16:54, 7,634 looks) she
was 100 units or more inside its sides and 212 or more inside its top and bottom (x -218.4 to 310.0, y -1677.5 to
247.8, against -360..410 and -2070..460).

The map's edges are its VR box (MapModel.bounds: where the camera stops, which the game keeps her inside of), or
where its platforms and ropes reach if it has none. A place read in the right map but wrong within it (the picture
found at a wrong place, say) isn't caught here: whether there's a platform or a rope under her was weighed and left
out, as in mid-air off a platform there can be nothing under her for hundreds of units, and a
look in mid-fall taken for a bad one would hold the hunt up.
"""

from .mapdata import MapModel
from .pose import Where

# Map units past the map's edges a place may be read at (her dot is good to about 11 units, a minimap pixel): across
# and down. The explorer's numbers (D:/MapleHelper-scratch/capture/explore/placecheck.py); every bad look of 16:46:50
# was further out than either.
OFF_X = 20
OFF_Y = 50


def edges(model: MapModel) -> tuple[float, float, float, float]:
    """The map's left, top, right and bottom (map units): its VR box, or its platforms' and ropes' reach."""
    if all(v is not None for v in model.bounds):
        return model.bounds
    xs = [p.left for p in model.platforms] + [p.right for p in model.platforms] + [r.x for r in model.ropes]
    ys = ([p.y_at(p.left) for p in model.platforms] + [p.y_at(p.right) for p in model.platforms]
          + [r.top for r in model.ropes] + [r.bottom for r in model.ropes])
    return min(xs), min(ys), max(xs), max(ys)


def off_map(model: MapModel, where: Where) -> str | None:
    """Why a place isn't on the map (in words for the log), or None if it is."""
    left, top, right, bottom = edges(model)
    if left - OFF_X <= where.x <= right + OFF_X and top - OFF_Y <= where.y <= bottom + OFF_Y:
        return None
    return (f"her minimap dot puts her at ({where.x:.0f}, {where.y:.0f}), off the map (x {left:.0f}..{right:.0f}, "
            f"y {top:.0f}..{bottom:.0f}): the minimap's picture not found where it is, or another map")
