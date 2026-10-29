"""The game's KEY BINDINGS window read (perception.keybindings): on the live window of 2026-10-28 10:56 (its 636x445
crop at (177, 174) of the 1366x768 frame, tests/fixtures/keybindings/live_window.png, and what was behind it once Esc
closed it, live_behind.png), on three of 2026-10-24 (lab/shots/nm_keys, nm_drag, nm_pick: her pet food dragged onto
U, its tooltip over the keyboard) and on frames drawn from the game's own pictures (keybindings.compose): ATTACK on
Ctrl, on another key, on none, in the tray; the window dragged, partly off the frame, covered, the Controller tab,
dimmed, noisy, a pointer over a cell; another window open; the pictures the same as the copy's."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab.perception import keybindings as kb
from maplehelper.keys import KEY_CODES

FIXTURES = Path(__file__).parent / "fixtures"
AT = kb.OPEN_AT


def _png(name: str) -> np.ndarray:
    return np.asarray(Image.open(FIXTURES / "keybindings" / name).convert("RGB"))


def _frame(window: np.ndarray | None = None, at=AT) -> np.ndarray:
    """A 1366x768 frame: what was behind the window live, the window (a 636x445 crop) over it at `at`."""
    out = np.full((768, 1366, 3), (60, 110, 60), np.uint8)
    behind = _png("live_behind.png")
    out[AT[1]:AT[1] + behind.shape[0], AT[0]:AT[0] + behind.shape[1]] = behind
    if window is not None:
        h, w = window.shape[:2]
        x, y = at
        x1, y1 = min(1366, x + w), min(768, y + h)
        out[y:y1, x:x1] = window[:y1 - y, :x1 - x]
    return out


@pytest.fixture(scope="module")
def reader():
    return kb.Reader()


@pytest.fixture(scope="module")
def live():
    return _frame(_png("live_window.png"))


def test_the_live_window_reads_attack_on_ctrl_preset_2(reader, live):
    r = reader.read(live)
    assert r.read and r.open and r.at == AT and r.preset == 2
    assert r.attack == ("ctrl",) and not r.in_tray and not r.unsure
    assert r.title < 0.01 and r.gaps > 0.99
    # exact: both Ctrl cells 0.0; the best other key 57.6 (Esc's MENU)
    assert r.scores["ctrl"] == 0.0 and min(v for k, v in r.scores.items() if k != "ctrl") > 50


def test_the_live_window_s_whole_map_43_functions_and_her_pictures(reader, live):
    """Every function on its key, and her pictures told apart where they're known (kb.PICTURES: her Lucky Seven on both
    Shifts, her Pet Food on U, The Green Relaxer on D); her potions on C and V a picture not known."""
    fn = reader.functions(live, reader.read(live))
    functions = {k: v for k, v in fn.items() if v not in (None, kb.PICTURE)}
    assert len(functions) == 44  # (43 cells: both Ctrls one key, both Alts one; and her three pictures)
    assert (fn["ctrl"], fn["alt"], fn["z"], fn["x"], fn["\\"], fn["esc"], fn["space"]) == (
        "ATTACK", "JUMP", "PICK UP", "SIT", "SET KEY", "MENU", "NPC CHAT")
    assert (fn["shift"], fn["u"], fn["d"]) == ("LUCKY SEVEN", "PET FOOD", "THE GREEN RELAXER")
    assert {k for k, v in fn.items() if v == kb.PICTURE} == {"c", "v"}  # (her potions: pictures not known)
    assert fn["f8"] is None and fn["a"] is None
    assert kb.functions_text(fn, ("ctrl", "a")) == "CTRL ATTACK"


@pytest.mark.parametrize("name", ["nm_keys", "nm_drag"])
def test_the_window_of_2026_10_24_reads_too(reader, name):
    """Saved on 2026-10-24 (lab/shots/nm_keys.png, nm_drag.png: her pet food put on U by hand, the pointer dragging
    it over U in the second), the window then at (180, 174): ATTACK on Ctrl, preset 2, U a picture (the food's)."""
    r = reader.read(_frame(_png(f"{name}_window.png"), at=(180, 174)))
    assert r.read and r.at == (180, 174) and r.attack == ("ctrl",) and r.preset == 2
    f = _frame(_png(f"{name}_window.png"), at=(180, 174))
    fn = reader.functions(f, r)
    # her Pet Food on U; dragged over it (the pointer holding it there), a picture not known: not told
    assert fn["u"] == ("PET FOOD" if name == "nm_keys" else kb.PICTURE) and fn["shift"] == "LUCKY SEVEN"


def test_a_tooltip_over_the_keyboard_isn_t_read(reader):
    """The same day (lab/shots/nm_pick.png): the pointer over U, its Pet Food tooltip over a third of the keyboard: the
    window found, nothing read off it."""
    r = reader.read(_frame(_png("nm_pick_window.png"), at=(180, 174)))
    assert r.open and not r.read and r.why.startswith("covered") and r.attack == ()


def test_the_frame_after_esc_is_not_open(reader):
    r = reader.read(_frame())
    assert not r.open and not r.read and r.why == "not open" and r.title > 0.3


def test_a_frame_of_another_size_isn_t_read(reader, live):
    r = reader.read(np.zeros((1080, 1920, 3), np.uint8))
    assert not r.read and "1920x1080" in r.why


def test_the_window_drawn_from_its_pictures_is_the_live_one(reader, live):
    """compose's window with the live bindings matches the live crop on 96% of its px (the rest: her items' pictures,
    the buttons compose doesn't draw) and reads the same."""
    c = kb.compose(kb.LIVE, kb.LIVE_PRESET, frame=_frame())
    win = (slice(AT[1], AT[1] + 445), slice(AT[0], AT[0] + 636))
    same = (np.abs(c[win].astype(int) - live[win].astype(int)).max(axis=2) <= 8).mean()
    assert same > 0.95
    r = reader.read(c)
    assert r.read and r.attack == ("ctrl",) and r.preset == 2


@pytest.mark.parametrize("moved, want", [
    ({"ctrl": None, "a": 52}, ("a",)),               # moved to A (as at 00:30 on 2026-10-28, a key of the user's)
    ({"ctrl": None}, ()),                            # on no key
    ({"f8": 52}, ("f8", "ctrl")),                    # on two keys (the keyboard's order)
    ({"ctrl": None, "shift": 52}, ("shift",)),       # on a key she has bound otherwise
    ({"ctrl": None, "pagedown": 52}, ("pagedown",)),
])
def test_attack_wherever_it_is(reader, moved, want):
    r = reader.read(kb.compose({**kb.LIVE, **moved}, 1, frame=_frame()))
    assert r.read and r.attack == want and r.preset == 1


def test_attack_in_the_tray_is_on_no_key(reader):
    r = reader.read(kb.compose({**kb.LIVE, "ctrl": None}, 3, frame=_frame(), tray={(16, 1): 202, (3, 0): 52}))
    assert r.read and r.attack == () and r.in_tray and r.preset == 3


def test_the_window_dragged_elsewhere(reader):
    r = reader.read(kb.compose(kb.LIVE, 2, at=(477, 54), frame=_frame()))
    assert r.read and r.at == (477, 54) and r.attack == ("ctrl",)


def test_partly_off_the_frame_isn_t_read(reader, live):
    win = _png("live_window.png")
    r = reader.read(_frame(win, at=(27, 574)))
    assert r.open and not r.read and r.why == "partly off the frame" and r.attack == ()


def test_the_controller_tab_isn_t_read(reader):
    r = reader.read(kb.compose(kb.LIVE, 2, frame=_frame(), tab=1))
    assert r.open and not r.read and r.why == "the Controller tab"


def test_another_window_over_the_keyboard_isn_t_read(reader, live):
    """Part of the keyboard covered (another window over it): its px between the cells aren't the picture's, nothing
    is read off it."""
    f = live.copy()
    f[AT[1] + 140:AT[1] + 300, AT[0] + 40:AT[0] + 400] = (240, 230, 200)
    r = reader.read(f)
    assert r.open and not r.read and r.why.startswith("covered") and r.attack == ()


def test_another_window_open_isn_t_the_key_bindings(reader):
    """The Lie Detector's window (its real top) where the key bindings would be: not open."""
    other = np.asarray(Image.open(FIXTURES / "lie_detector_window.png").convert("RGB"))
    f = _frame()
    h, w = min(other.shape[0], 445), min(other.shape[1], 636)
    f[AT[1]:AT[1] + h, AT[0]:AT[0] + w] = other[:h, :w]
    assert not reader.read(f).open


def test_dimmed_or_noisy_still_reads(reader, live):
    """10 levels darker, 15% dimmer (the title strip's levels fitted and undone: Reader._levels), +-3 noise."""
    for f in (np.clip(live.astype(int) - 10, 0, 255), np.clip(live.astype(int) * 0.85, 0, 255)):
        r = reader.read(f.astype(np.uint8))
        assert r.read and r.attack == ("ctrl",) and r.preset == 2, r.text()
    rng = np.random.default_rng(1)
    r = reader.read(np.clip(live.astype(int) + rng.integers(-3, 4, live.shape), 0, 255).astype(np.uint8))
    assert r.read and r.attack == ("ctrl",) and r.preset == 2


def test_a_pointer_over_one_ctrl_the_other_still_reads(reader, live):
    """A pointer-sized blob over the left Ctrl's picture: that cell fails, the right one's still ATTACK."""
    import cv2
    f = live.copy()
    left = next(c for c in kb.BOUND if c.key == "ctrl")
    x, y = AT[0] + kb.KEYBOARD_AT[0] + left.x, AT[1] + kb.KEYBOARD_AT[1] + left.y
    cv2.fillPoly(f, [np.array([[x + 14, y + 8], [x + 14, y + 27], [x + 26, y + 21]])], (255, 255, 255))
    r = reader.read(f)
    assert r.read and r.attack == ("ctrl",)


def test_the_layout_s_keys_are_config_names_and_each_has_its_name_picture(reader):
    assert len(kb.BOUND) == 77 and len(kb.KEYS) == 74
    assert all(k in KEY_CODES for k in kb.KEYS)
    assert all(c.code in reader.labels for c in kb.BOUND)
    assert kb.same_key("rctrl", "ctrl") and kb.same_key("ctrl", "ctrl") and not kb.same_key("ctrl", "alt")
    assert set(reader.icons) == set(kb.FUNCTIONS)


def test_the_pictures_are_the_copy_s(tmp_path):
    """assets/keybindings as cut_assets writes them from the copy (UI/UI_000.wz KeyBindings.img), where it's here."""
    from gamedata.paths import COPY
    if not (COPY / "UI" / "UI_000.wz").is_file():
        pytest.skip("needs the game data's copy (gamedata/data/copy)")
    n = kb.cut_assets(tmp_path)
    mine = sorted(p.relative_to(kb.FOLDER).as_posix() for p in kb.FOLDER.rglob("*.png"))
    assert n == len(mine) == 140  # (133 of KeyBindings.img, 7 of her items' and skills', kb.PICTURES)
    assert (tmp_path / "picture" / "origins.json").read_text() == (kb.FOLDER / "picture" / "origins.json").read_text()
    for rel in mine:
        a = np.asarray(Image.open(kb.FOLDER / rel).convert("RGBA"))
        b = np.asarray(Image.open(tmp_path / rel).convert("RGBA"))
        assert a.shape == b.shape and (a == b).all(), rel


def test_attack_s_cells_covered_isn_t_on_no_key(reader, live):
    """The review's probe: a box over both Ctrl cells (the keyboard's px between the cells still 0.99 as drawn) read
    "ATTACK on no key". Their key names aren't drawn there either: covered, not read."""
    f = live.copy()
    for c in kb.BOUND:
        if c.key == "ctrl":
            x, y = AT[0] + kb.KEYBOARD_AT[0] + c.x, AT[1] + kb.KEYBOARD_AT[1] + c.y
            f[y - 2:y + 28, x:x + 40] = (240, 230, 200)
    r = reader.read(f)
    assert r.open and not r.read and r.why.startswith("covered (not legible: CTRL") and r.illegible == ("ctrl",)
    assert reader.read(live).illegible == ()


def test_the_window_dragged_past_the_frame_s_right_edge_is_open_not_read(reader):
    """The review's case: at x 900 its whole title isn't in the frame (it was "not open", so nothing closed it); its
    name, the title's left TITLE_PART px, is: open, partly off the frame, not read."""
    r = reader.read(kb.compose(kb.LIVE, 2, at=(900, 174), frame=_frame()))
    assert r.open and not r.read and r.why == "partly off the frame" and r.at == (900, 174)
    assert not reader.read(_frame()).open  # (the minimap's title, "MINI MAP", isn't its name)


def test_her_pictures_are_told_apart_on_the_live_window_and_nowhere_else(reader, live):
    """kb.PICTURES on the live window: each at its own cell 0.98-1.00 of its px drawn as they are (but under the key's
    name and an item's count), at any other cell 0.60 at the most (Energy Bolt's against the FACE pictures, which are
    functions, read first)."""
    r = reader.read(live)
    ox, oy = r.at
    win = reader._levels(live[oy:oy + kb.WINDOW[1], ox:ox + kb.WINDOW[0]])
    own = {"lucky_seven": "shift", "pet_food": "u", "chair": "d"}
    for c in kb.BOUND:
        for name, share in reader.picture_shares(win, c).items():
            if own.get(name) == c.key:
                assert share >= 0.98, (name, c.key, share)
            else:
                assert share <= 0.61, (name, c.key, share)


@pytest.mark.parametrize("bindings, want", [
    ({"shift": None, "a": "lucky_seven"}, {"a": "LUCKY SEVEN", "shift": None}),       # Lucky Seven moved to A
    ({"u": None, "i": "pet_food"}, {"i": "PET FOOD", "u": None}),                      # her pet food to I
    ({"shift": "pet_food", "u": "lucky_seven"}, {"shift": "PET FOOD", "u": "LUCKY SEVEN"}),
    ({"alt": None, "c": 53, "left": None, "q": 300}, {"c": "JUMP", "q": "MOVE LEFT", "alt": None, "left": None}),
])
def test_moved_functions_and_pictures_read_where_they_are(reader, bindings, want):
    """compose's window with her keys moved (the simulator draws it so: sim.world.World.binding_view) reads them where
    they are."""
    f = kb.compose({**kb.LIVE, **bindings}, 2, frame=_frame())
    fn = reader.functions(f, reader.read(f))
    assert {k: fn[k] for k in want} == want
