"""Other players on the minimap: their markers found in pictures of it (perception.minimap), on real minimaps and
on the game's own marker pictures pasted onto them as the game draws them, never taking her dot, an NPC or a portal
for one, nor pulling her dot off."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gamedata.paths import COPY
from gamelab.perception import minimap
from gamelab.perception.minimap import MARKER_COLOURS, MINIMAP_BOX, OTHER_SHIFT, Marker, dot_in_box, markers_in_box
from gamelab.screen import SCREEN_SCALE

FIXTURES = Path(__file__).parent / "fixtures"
X0, Y0 = MINIMAP_BOX[:2]  # where the fixtures' crops start on the 1920x1080 screen


def crop(name: str) -> np.ndarray:
    """A saved minimap (the screen's MINIMAP_BOX, RGB): `stranger`, another player on this map (hover_D.png of
    2026-10-21); `party`, a leader of her party on the safe platform 32, her dot over a portal's (calibrate.png);
    `crowd`, nine strangers and an NPC in Kerning City Construction Site, whose minimap reaches further right
    (lie_detector_025439.png, 330 px wide)."""
    return np.asarray(Image.open(FIXTURES / f"minimap_{name}.png").convert("RGB")).copy()


# --- real minimaps ------------------------------------------------------------------------------------------

def test_another_player_on_this_map_is_a_stranger_and_her_dot_stays_hers():
    box = crop("stranger")
    (m,) = markers_in_box(box)
    assert m.kind == "stranger" and abs(m.x - 122.46) < 0.1 and abs(m.y + Y0 - 254.8) < 0.1
    assert dot_in_box(box) == pytest.approx((172.31, 237.15), abs=0.01)  # as before the markers were looked for


def test_a_leader_of_her_party_is_party_and_her_dot_over_a_portal_is_still_found():
    box = crop("party")
    assert markers_in_box(box) == [Marker("party", 160.5, 243.5 - Y0)]  # its crown (yellow) isn't her dot
    assert dot_in_box(box) == pytest.approx((89.5, 248.5), abs=0.01)


def test_a_crowd_of_strangers_side_by_side_are_each_found_and_the_npc_among_them_isnt():
    box = crop("crowd")
    found = markers_in_box(box)
    assert {m.kind for m in found} == {"stranger"}
    whole = [(277.1, 55.8), (291.1, 55.9), (303.9, 55.9), (270.1, 79.7), (294.0, 79.6), (306.7, 79.6),
             (114.0, 81.0), (277.0, 95.0), (291.0, 95.0)]  # 13-14 px apart, their halos touching
    for x, y in whole:
        assert any(abs(m.x - x) < 0.2 and abs(m.y - y) < 0.2 for m in found), (x, y)
    # and one cut off by the minimap's right edge (half its core); none where the green NPC stands (282, 79)
    assert len(found) == len(whole) + 1 and not any(abs(m.x - 282) < 6 and abs(m.y - 79) < 6 for m in found)


def _full_size(path: Path) -> bool:
    try:
        return Image.open(path).size == (1920, 1080)
    except OSError:  # one of them is cut short (written as the player shut down)
        return False


def test_saved_frames_of_this_map_show_no_other_player_but_the_two_seen():
    lab = Path(__file__).resolve().parents[1] / "gamelab" / "lab"
    frames = sorted(lab.glob("*.png")) if lab.is_dir() else []
    full = [p for p in frames if _full_size(p) and "lie_detector_025439" not in p.name]
    if len(full) < 20:
        pytest.skip("needs the saved 1920x1080 screenshots (gamelab/lab)")
    seen = {p.name: minimap.other_dots(np.asarray(Image.open(p).convert("RGB"))) for p in full}
    with_others = {name: [m.kind for m in ms] for name, ms in seen.items() if ms}
    # A stranger in the two frames of 2026-10-21 taken while the user hovered over the key bindings; the party
    # leader on platform 32 in two frames five minutes apart; nothing in the other 27.
    assert with_others == {"hover_D.png": ["stranger"], "hover_G.png": ["stranger"], "calibrate.png": ["party"],
                           "lie_detector_012504.png": ["party"]}


# --- the game's own markers pasted on them ------------------------------------------------------------------

@pytest.fixture(scope="module")
def markers():
    if not (COPY / "Map" / "_Canvas").is_dir():
        pytest.skip("needs the game data copy (gamedata/data/copy)")
    from gamedata.canvas import minimap_markers
    return minimap_markers()


# Where each kind's core's middle is in its picture: the data's origin where it has one (user, another, friend), else
# the same place as `another`'s (the party leader's and the guild master's crowns above): a party leader's on the
# saved frames sat 18 units above the safe platform's surface placed so (her own dot sat 7 off it).
ORIGINS = {"another": (5, 5), "friend": (5, 5), "party": (5, 5), "guild": (5, 5), "partymaster": (5, 9),
           "guildmaster": (5, 9), "user": (6, 6), "npc": (5, 6), "portal": (7, 7)}
KIND_OF = {"another": "stranger", "party": "party", "partymaster": "party", "guild": "guild",
           "guildmaster": "guild", "friend": "buddy"}


def paste(box: np.ndarray, rgba: np.ndarray, origin: tuple[int, int], gx: int, gy: int) -> np.ndarray:
    """A marker drawn on a minimap crop as the game draws it: at whole game px (gx, gy) of its 1366x768 view (its
    origin there), and shown 1.40625x larger, filtered bilinearly. (So pasted over the one the game drew in
    hover_D.png, at game px (87, 181), it came out within half a level of it on average.)"""
    a = rgba.astype(np.float32) / 255
    pre = np.dstack([a[..., :3] * a[..., 3:], a[..., 3:]])  # premultiplied, as the filter blends it
    s = SCREEN_SCALE
    tx, ty = (gx - origin[0] + 0.5) * s - 0.5 - X0, (gy - origin[1] + 0.5) * s - 0.5 - Y0
    h, w = box.shape[:2]
    over = cv2.warpAffine(pre, np.float32([[s, 0, tx], [0, s, ty]]), (w, h), flags=cv2.INTER_LINEAR)
    out = box.astype(np.float32) / 255 * (1 - over[..., 3:]) + over[..., :3]
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def drawn_at(gx: int, gy: int) -> tuple[float, float]:
    """Where a marker drawn at game px (gx, gy) has its core's middle, in a crop's px."""
    s = SCREEN_SCALE
    return (gx + 0.5) * s - 0.5 - X0, (gy + 0.5) * s - 0.5 - Y0


def her_game_px(dot: tuple[float, float]) -> tuple[int, int]:
    """The game px her dot (full-size px, as dot_in_box gives it) is drawn at: her core's middle is on it."""
    return round((dot[0] + 0.5) / SCREEN_SCALE), round((dot[1] + 0.5) / SCREEN_SCALE)


# Spots on this map's minimap picture (game px) clear of her dot, the portals and the markers already there, in
# both crops: over the sky, the cliffs, the trees, platforms and their grass.
SPOTS = [(55, 80), (70, 95), (92, 77), (120, 82), (150, 76), (160, 100), (52, 130), (75, 120), (100, 140),
         (130, 128), (150, 150), (58, 160), (95, 165), (140, 190), (160, 175), (63, 200), (110, 202)]


@pytest.mark.parametrize("name", ["stranger", "party"])
def test_each_kind_of_marker_is_found_where_it_is_drawn_and_nothing_else(markers, name):
    box = crop(name)
    before = markers_in_box(box)
    for sprite, kind in KIND_OF.items():
        for gx, gy in SPOTS:
            found = [m for m in markers_in_box(paste(box, markers[sprite][0], ORIGINS[sprite], gx, gy))
                     if m not in before]
            x, y = drawn_at(gx, gy)
            assert len(found) == 1 and found[0].kind == kind, (sprite, gx, gy, found)
            assert abs(found[0].x - x) < 0.5 and abs(found[0].y - y) < 0.5, (sprite, gx, gy, found)


@pytest.mark.parametrize("name", ["stranger", "party"])
def test_npcs_and_portals_are_never_players(markers, name):
    box = crop(name)
    before = markers_in_box(box)
    for sprite in ("npc", "portal", "enchantportal", "hiddenportal", "eventnpc", "startnpc", "shop", "trunk",
                   "transport", "rune", "usertrader", "anothertrader"):
        for gx, gy in SPOTS:
            drawn = paste(box, markers[sprite][0], ORIGINS.get(sprite, (5, 5)), gx, gy)
            assert markers_in_box(drawn) == before, (sprite, gx, gy)


@pytest.mark.parametrize("name", ["stranger", "party"])
def test_others_touching_her_dot_are_found_and_dont_pull_it_off(markers, name):
    """Her marker drawn over theirs or under, 5-12 game px off (57-137 map units) on either side, above or below.
    Drawn over hers (which the game may not do), one 5 game px off pulled her dot 1.3 px its way on average (up
    to 2.2) before her dot was taken from its side that shows (DOT_UNCOVER); now 0.4 (up to 0.8)."""
    box = crop(name)
    hx, hy = her_game_px(dot_in_box(box))
    alone = paste(box, markers["user"][0], ORIGINS["user"], hx, hy)  # her drawn again where she is
    her = dot_in_box(alone)
    for sprite, kind in KIND_OF.items():
        for dx, dy in [(d, e) for d in (-12, -9, -7, -5, 5, 7, 9, 12) for e in (-2, 0, 2)] + \
                      [(e, d) for d in (-12, -8, -6, 6, 8, 12) for e in (-1, 1)]:
            other = paste(box, markers[sprite][0], ORIGINS[sprite], hx + dx, hy + dy)
            under = paste(other, markers["user"][0], ORIGINS["user"], hx, hy)  # hers drawn last
            over = paste(alone, markers[sprite][0], ORIGINS[sprite], hx + dx, hy + dy)  # theirs drawn last
            # A leader's marker (15 px tall, its crown above) drawn over hers from below can hide her dot whole.
            for drawn, slack in ((under, 0.3),) + (((over, 0.9),) if markers[sprite][0].shape[0] == 11 else ()):
                dot = dot_in_box(drawn)
                assert dot is not None and np.hypot(dot[0] - her[0], dot[1] - her[1]) < slack, (sprite, dx, dy, dot)
            x, y = drawn_at(hx + dx, hy + dy)
            # Hers drawn over theirs hides some of their core: 7 game px off or more (80 map units) theirs is always
            # found, 6 px off missed 12% of the time, 5 px off 60% (every offset around her, both minimaps).
            for drawn in (over,) + ((under,) if np.hypot(dx, dy) >= 7 else ()):
                assert any(m.kind == kind and abs(m.x - x) < 1 and abs(m.y - y) < 1 for m in markers_in_box(drawn)), \
                    (sprite, dx, dy)


def test_beside_a_portal_another_player_is_found(markers):
    box = crop("stranger")
    for px, py in ((56, 175), (158, 175)):  # the two portals' markers, game px
        for dx, dy in ((-10, 0), (-7, -2), (7, 2), (10, 0), (0, -9), (6, -6)):
            found = [m for m in markers_in_box(paste(box, markers["another"][0], ORIGINS["another"], px + dx, py + dy))
                     if m.kind == "stranger" and abs(m.x - 122.46) > 1]
            x, y = drawn_at(px + dx, py + dy)
            assert len(found) == 1 and abs(found[0].x - x) < 0.5 and abs(found[0].y - y) < 0.5, (px, dx, dy, found)


def test_two_strangers_side_by_side_are_two_down_to_4_game_px_apart(markers):
    """Nearer than that, the one drawn last hides most of the other's core, and they're one."""
    box = crop("party")
    for gx, gy in SPOTS[:8]:
        for d in (5, 6, 8, 10, 12):
            two = paste(paste(box, markers["another"][0], (5, 5), gx, gy), markers["another"][0], (5, 5), gx + d, gy)
            assert sum(m.kind == "stranger" for m in markers_in_box(two)) == 2, (gx, gy, d)


def test_each_kind_has_its_markers_core_colours_and_hers_none(markers):
    """Every opaque, coloured pixel of a marker's core (its picture in the game data) is in its kind's colours
    (MARKER_COLOURS) and no other kind's; her marker's yellow, the party leader's crown, an NPC's green and a
    portal's cyan rings are in none but (the cyan) a buddy's."""
    def kinds_of(rgb):
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)
        return {k for k, ranges in MARKER_COLOURS.items() if any(cv2.inRange(hsv, lo, hi)[0, 0] for lo, hi in ranges)}

    for sprite, kind in KIND_OF.items():
        rgba = markers[sprite][0]
        core = [tuple(int(v) for v in px[:3]) for px in rgba.reshape(-1, 4)
                if px[3] == 255 and int(px[:3].max()) - int(px[:3].min()) > 60]
        crown = [c for c in core if c[0] > 220 and c[1] > 190]  # the leaders' yellow crowns
        assert core and all(kinds_of(c) == {kind} for c in core if c not in crown), sprite
        assert all(not kinds_of(c) for c in crown)
    for sprite in ("user", "npc"):
        rgba = markers[sprite][0]
        assert not any(kinds_of(tuple(int(v) for v in px[:3])) for px in rgba.reshape(-1, 4) if px[3] == 255), sprite


def test_marker_shift_is_the_pictures_half_pixel():
    """A stranger's core is drawn half a game px right of and below where her dot's would be (origins (5, 5) of
    11 px and (6, 6) of 12): 0.70 screen px."""
    assert OTHER_SHIFT == pytest.approx(0.703, abs=0.001)
