"""Her EXP number read (perception.exp_number): on every saved screenshot of the bar, each checked by eye against the
value written here; on the saved crops of it, enlarged as they were saved; and on the game's own glyphs drawn as the
game draws them, over the bar and over other backgrounds, faded, covered, shifted, or with something after them. A
wrong value must never come out; no value may."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gamedata.paths import COPY
from gamelab.perception import exp_number
from gamelab.perception.exp_number import NUMBER_AT, READ_BOX, SX, SY, read
from maplehelper.ocr.exp_table import EXP_TO_NEXT

LAB = Path(__file__).resolve().parents[1] / "gamelab" / "lab"

pytestmark = pytest.mark.skipif(not (COPY / "UI" / "UI_000.wz").is_file(),
                                reason="needs the game data copy (gamedata/data/copy): the bar's glyphs")

# Every saved 1920x1080 screenshot, and what its EXP number says, read by eye off a crop of it enlarged 4 times
# (and each fits her level's requirement: level 19 needs 16,680, 20 needs 20,216).
FRAMES = {
    "after_rest.png": (447, 2.68, 19), "calibrate.png": (13549, 67.02, 20), "closed.png": (421, 2.52, 19),
    "closeup.png": (447, 2.68, 19), "hover_D.png": (421, 2.52, 19), "hover_G.png": (421, 2.52, 19),
    "hover_R.png": (421, 2.52, 19), "hover_T.png": (421, 2.52, 19), "hover_U.png": (421, 2.52, 19),
    "kc_full.png": (421, 2.52, 19), "keyconfig.png": (421, 2.52, 19), "lie_detector_004720.png": (1878, 11.26, 19),
    "lie_detector_012504.png": (13547, 67.01, 20), "lie_detector_015828.png": (3162, 18.96, 19),
    "lie_detector_025439.png": (5225, 31.32, 19), "lie_detector_204759.png": (869, 5.21, 19),
    "lie_detector_214638.png": (7654, 45.89, 19), "loot.png": (435, 2.61, 19), "sit.png": (435, 2.61, 19),
    "stand.png": (435, 2.61, 19), "z1_after.png": (453, 2.72, 19), "z1_hop.png": (451, 2.70, 19),
    "z1_range.png": (453, 2.72, 19), "z2_down.png": (453, 2.72, 19), "z2_loot1.png": (503, 3.02, 19),
    "z2_loot2.png": (516, 3.09, 19), "z2_shoot1.png": (453, 2.72, 19), "z2_shoot2.png": (466, 2.79, 19),
    "z2_shoot3.png": (478, 2.87, 19), "z2_shoot4.png": (490, 2.94, 19), "z2_shoot5.png": (503, 3.02, 19),
    "z2_shoot6.png": (516, 3.09, 19),
}
# The saved crops of the bar (2026-10-18 and 19), by eye the same way: where each was cut from the screen and how many
# times it was enlarged (with PIL's bicubic: enlarging the screen's own pixels so gives them back to the level).
_EXP_CROPS = {"exp_before": 1878, "exp_after": 1930, "exp_before5": 1936, "exp_after5": 2028, "exp_before6": 2028,
              "exp_after6": 2036, "exp_before7": 2147, "exp_after7": 2560, "exp_before8": 2613, "exp_after8": 2967,
              "exp_before9": 2976, "exp_before10": 3174, "exp_before11": 6459, "exp_after11": 6856,
              "exp_before12": 6856, "exp_after12": 7247}
_PERCENT = {1878: 11.26, 1930: 11.57, 1936: 11.61, 2028: 12.16, 2036: 12.21, 2147: 12.87, 2560: 15.35, 2613: 15.67,
            2967: 17.79, 2976: 17.84, 3174: 19.03, 6459: 38.72, 6856: 41.10, 7247: 43.45}
CROPS = {**{f"{name}.png": ((1010, 1030, 3), (exp, _PERCENT[exp], 19)) for name, exp in _EXP_CROPS.items()},
         "hud_exp.png": ((690, 1030, 2), (597, 3.58, 19)), "hud_end.png": ((690, 1030, 1), (813, 4.87, 19)),
         "rest_hud.png": ((690, 1030, 1), (447, 2.68, 19))}


def _saved(name: str) -> np.ndarray:
    path = LAB / name
    if not path.is_file():
        pytest.skip(f"needs the saved screenshots (gamelab/lab/{name})")
    return np.asarray(Image.open(path).convert("RGB"))


def box_of(full: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = READ_BOX
    return full[y0:y1, x0:x1]


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_every_saved_screenshot_reads_as_checked_by_eye(name):
    r = read(box_of(_saved(name)))
    assert (r.exp, r.percent, r.level) == FRAMES[name], r
    assert r.fit < 1.0 and r.lead > 14.0  # every cell within a level of the game's glyph; the next best far off


def enlarging(n: int, k: int) -> np.ndarray:
    """PIL's bicubic enlargement of n samples k times, as a matrix (n*k x n): each sample's own, enlarged alone."""
    out = np.zeros((n * k, n), np.float32)
    for i in range(n):
        row = np.zeros((1, n), np.float32)
        row[0, i] = 1.0
        out[:, i] = np.asarray(Image.fromarray(row, "F").resize((n * k, 1), Image.BICUBIC))[0]
    return out


def unenlarged(img: np.ndarray, k: int) -> np.ndarray:
    """The screen's pixels a crop enlarged k times was made from: the enlargement undone by least squares (OpenCV's
    solver: numpy's linear algebra doesn't load here), down then across, to about a level."""
    if k == 1:
        return img.astype(np.float32)
    down, across = enlarging(img.shape[0] // k, k), enlarging(img.shape[1] // k, k)
    out = np.zeros((img.shape[0] // k, img.shape[1] // k, 3), np.float32)
    for ch in range(3):
        _, rows = cv2.solve(down, img[..., ch].astype(np.float32), flags=cv2.DECOMP_SVD)
        _, cols = cv2.solve(across, np.ascontiguousarray(rows.T), flags=cv2.DECOMP_SVD)
        out[..., ch] = cols.T
    return np.clip(out, 0, 255)


@pytest.mark.parametrize("name", sorted(CROPS))
def test_every_saved_crop_of_the_bar_reads_as_checked_by_eye(name):
    (x, y, k), want = CROPS[name]
    screen = unenlarged(_saved(name), k)
    x0, y0, x1, y1 = READ_BOX
    box = screen[y0 - y:y1 - y, x0 - x:x1 - x]
    if box.shape[1] < x1 - x0:  # rest_hud.png ends a column short, on the plain bar: that column again
        box = np.concatenate([box, box[:, -1:]], axis=1)
    r = read(box)
    assert (r.exp, r.percent, r.level) == want, r


# --- the game's glyphs, drawn as the game draws them ----------------------------------------------------------------

@pytest.fixture(scope="module")
def glyphs():
    from gamedata.canvas import status_numbers
    return status_numbers()


def drawn(text: str, under: np.ndarray, glyphs: dict, at: int = NUMBER_AT[0], opacity: float = 1.0) -> np.ndarray:
    """`text` drawn over the box `under` as the game draws its EXP number: its glyphs side by side at game px, each its
    picture's width and a blank column apart from the last, the first at game x `at`; the whole picture stretched to
    1920x1080 with bilinear filtering (OpenCV's, not the reader's own), and laid over `under` at `opacity`."""
    gx0, gy0 = at - 4, NUMBER_AT[1] - 4
    canvas = np.zeros((16, 130, 4), np.float32)
    x = 4
    for ch in text:
        if ch == " ":  # (not the game's: a gap a digit wide)
            x += 6
            continue
        g = glyphs[ch].astype(np.float32) / 255
        h, w = g.shape[:2]
        canvas[4:4 + h, x:x + w, :3] += g[..., :3] * g[..., 3:]
        canvas[4:4 + h, x:x + w, 3:] += g[..., 3:]
        x += w + 1
    x0, y0, x1, y1 = READ_BOX
    tx, ty = (gx0 + 0.5) * SX - 0.5 - x0, (gy0 + 0.5) * SY - 0.5 - y0
    over = cv2.warpAffine(canvas, np.float32([[SX, 0, tx], [0, SY, ty]]), (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    out = under.astype(np.float32) * (1 - opacity * over[..., 3:]) + opacity * over[..., :3] * 255
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def text_of(exp: int, level: int) -> tuple[str, float]:
    """What the bar shows with `exp` into `level`, and its percent (to the nearest hundredth, as the game rounds it)."""
    percent = round(exp * 100 / EXP_TO_NEXT[level], 2)
    return f"{exp:,}[{percent:.2f}%]", percent


@pytest.fixture(scope="module")
def bar():
    """The bar's own background under the number: calibrate.png's box, each row its darkest colour."""
    box = box_of(_saved("calibrate.png")).astype(np.float32)
    return np.broadcast_to(np.percentile(box, 10, axis=1)[:, None, :], box.shape).astype(np.uint8).copy()


def values(n: int, seed: int) -> list[tuple[int, int]]:
    """n (EXP, level) she could have: levels 1-60, EXP anywhere in them."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        level = int(rng.integers(1, 61))
        out.append((int(rng.integers(0, EXP_TO_NEXT[level])), level))
    return out


def test_drawn_as_the_game_draws_it_the_saved_frame_comes_back(glyphs, bar):
    # The drawing here is the game's own to a level or two: calibrate.png's number, drawn over its bar.
    frame = box_of(_saved("calibrate.png")).astype(int)
    again = drawn("13,549[67.02%]", bar, glyphs).astype(int)
    assert np.abs(again - frame).mean() < 0.6 and np.abs(again - frame).max() <= 6


def test_any_number_drawn_over_the_bar_reads_right(glyphs, bar):
    for exp, level in values(300, 1) + [(0, 1), (95_699, 30), (888_888, 59), (14, 1)]:
        text, percent = text_of(exp, level)
        r = read(drawn(text, bar, glyphs))
        assert r.exp == exp and r.percent == percent and level in r.levels, (text, r)


BACKGROUNDS = {
    "black": lambda rng, shape: np.zeros(shape),
    "white": lambda rng, shape: np.full(shape, 255.0),
    "grey": lambda rng, shape: np.full(shape, 128.0),
    "gauge green": lambda rng, shape: np.broadcast_to([170.0, 220.0, 20.0], shape),
    "sky": lambda rng, shape: np.broadcast_to([120.0, 190.0, 240.0], shape),
    "dark down": lambda rng, shape: np.broadcast_to(np.linspace(20, 90, shape[0])[:, None, None], shape),
    "light across": lambda rng, shape: np.broadcast_to(np.linspace(60, 230, shape[1])[None, :, None], shape),
    "noise": lambda rng, shape: rng.uniform(0, 255, shape),
    "dark noise": lambda rng, shape: rng.uniform(40, 90, shape),
    "stripes": lambda rng, shape: np.broadcast_to((np.arange(shape[1]) // 3 % 2 * 200.0)[None, :, None], shape),
    "blobs": lambda rng, shape: cv2.GaussianBlur(rng.uniform(0, 255, shape).astype(np.float32), (0, 0), 3) * 1.0,
}


@pytest.mark.parametrize("name", sorted(BACKGROUNDS))
@pytest.mark.parametrize("opacity", [1.0, 0.75, 0.5])
def test_over_any_background_and_faded_a_wrong_value_never_comes_out(glyphs, name, opacity):
    rng = np.random.default_rng(7)
    shape = (READ_BOX[3] - READ_BOX[1], READ_BOX[2] - READ_BOX[0], 3)
    for exp, level in values(25, sorted(BACKGROUNDS).index(name)):
        under = np.clip(BACKGROUNDS[name](rng, shape), 0, 255).astype(np.uint8)
        text, percent = text_of(exp, level)
        r = read(drawn(text, under, glyphs, opacity=opacity))
        assert r.exp is None or (r.exp, r.percent) == (exp, percent), (name, opacity, text, r)


@pytest.mark.parametrize("name", ["black", "dark down"])
def test_over_a_plain_background_it_still_reads(glyphs, name):
    # Each row the same colour, as the bar's is, and dark like it. (Over grey 128 the green brackets are too like each
    # other for LEAD: no reading.)
    rng = np.random.default_rng(3)
    shape = (READ_BOX[3] - READ_BOX[1], READ_BOX[2] - READ_BOX[0], 3)
    for exp, level in values(20, 5):
        text, percent = text_of(exp, level)
        r = read(drawn(text, np.clip(BACKGROUNDS[name](rng, shape), 0, 255).astype(np.uint8), glyphs))
        assert (r.exp, r.percent) == (exp, percent), (name, text, r)


def test_covered_anywhere_by_the_cursor_or_a_window_it_reads_right_or_not_at_all(glyphs, bar):
    rng = np.random.default_rng(11)
    shape = bar.shape
    misses = 0
    for exp, level in values(200, 9):
        text, percent = text_of(exp, level)
        box = drawn(text, bar, glyphs)
        x, y = int(rng.integers(-6, shape[1])), int(rng.integers(-8, shape[0]))
        w, h = int(rng.integers(3, 20)), int(rng.integers(3, 20))
        colour = rng.choice([0, 255, 128]) if rng.random() < 0.7 else rng.integers(0, 256, 3)
        box[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = colour
        r = read(box)
        assert r.exp is None or (r.exp, r.percent) == (exp, percent), (text, (x, y, w, h), r)
        misses += r.exp is None
    assert misses >= 150  # most are covered somewhere it shows (those that aren't still read)


def test_a_digit_right_after_the_number_or_further_along_and_it_isnt_read(glyphs, bar):
    assert read(drawn("453[2.72%]", bar, glyphs)).exp == 453
    for text in ("453[2.72%]7", "453[2.72%]  1", "453[2.72%]    .", "453[2.72%]0[", "453[2.72%][2.72"):
        r = read(drawn(text, bar, glyphs))
        assert r.exp is None and r.why.startswith("not an EXP number"), (text, r)
    r = read(drawn("453[2.72%][2.72%]", bar, glyphs))  # running past the bar's end
    assert r.exp is None, r


@pytest.mark.parametrize("text,why", [
    ("453[2.73%]", "fits no level"),       # the percent isn't what 453 is of any level's requirement
    ("16,680[100.00%]", "fits no level"),  # 16,680 is all level 19 needs: she would be 20
    ("0453[2.72%]", "not an EXP number"),  # a leading zero
    ("453[02.72%]", "not an EXP number"),
    ("4,53[2.72%]", "not an EXP number"),  # commas every 3 digits from the right
    ("4530[2.72%]", "not an EXP number"),
    ("453[2.72]", "not an EXP number"),
    ("453", "not an EXP number"),
])
def test_what_isnt_an_exp_number_of_her_level_isnt_read(glyphs, bar, text, why):
    r = read(drawn(text, bar, glyphs))
    assert r.exp is None and why in r.why, r


@pytest.mark.parametrize("shift", [-1, 1, 2])
def test_drawn_off_its_place_it_isnt_read(glyphs, bar, shift):
    # One game px left or right of where the game draws it (another client, another layout): no value, not a wrong one.
    for exp, level in values(30, 13):
        r = read(drawn(text_of(exp, level)[0], bar, glyphs, at=NUMBER_AT[0] + shift))
        assert r.exp is None, r


def test_a_small_percent_fits_several_levels_and_names_none():
    # 0.00% at 3 EXP fits every level needing more than 600; 2.52% at 421 only level 19's 16,680.
    box = box_of(_saved("closed.png"))
    assert read(box).levels == (19,)
    r = exp_number.ExpReading(3, 0.0, (40, 41))
    assert r.level is None and r.need is None and exp_number.ExpReading(421, 2.52, (19,)).need == 16_680


def test_reading_takes_a_few_milliseconds():
    # Only when the number changes (a kill): on the saved frames about 8 ms (once the glyphs are drawn, 0.2 s).
    import time
    box = box_of(_saved("calibrate.png"))
    read(box)
    t = time.perf_counter()
    for _ in range(20):
        read(box)
    assert (time.perf_counter() - t) / 20 < 0.05
