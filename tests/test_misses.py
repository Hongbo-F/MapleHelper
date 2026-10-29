"""Her attack's MISS over a monster (perception.misses; the user, 2026-10-28: "Yes, read it"): the reader on crops of
the saved hunting frames (tests/fixtures/misses: the MISS popups of 2026-10-24 23:48:52 and 23:50:30, 10-25 00:52:23
and 10-26 23:10:08, and the best scores of what isn't one: overlapping damage numbers, a hand, her numbers, MISSes
faded), on the picture drawn faded, its asset the copy's; the watch counting a press with a MISS over its target as
one that took effect (keycheck.RegularWatch: "missed"). End to end: loop_tree_attack_misses_seen (no doubt) and
loop_tree_attack_misses (the MISSes unseen: the doubt, as before)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gamelab.behavior import keycheck
from gamelab.behavior.keycheck import DOUBT_PRESSES, RegularWatch
from gamelab.localization.monsters import Monster
from gamelab.perception import misses

FIXTURES = Path(__file__).parent / "fixtures" / "misses"


@pytest.fixture(scope="module")
def reader():
    return misses.MissReader()


def _png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("miss_*.png")), ids=lambda p: p.stem)
def test_the_live_misses_are_read(reader, path):
    """Each a MISS popup over a monster (the last fading, 0.79), read in its crop: 0.79-0.96, MISS_OK 0.6."""
    frame = _png(path)
    found = reader.find(frame, [(0, 0, frame.shape[1], frame.shape[0])])
    assert len(found) == 1 and found[0].score >= 0.78
    assert 30 <= found[0].x <= 150 and 30 <= found[0].y <= frame.shape[0]  # (its bottom centre in the crop)


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("not_*.png")), ids=lambda p: p.stem)
def test_what_isn_t_a_fresh_miss_isn_t_read(reader, path):
    """The best scores of the 350,000 boxes of the fill that weren't a MISS (overlapping numbers 0.52, a hand 0.55),
    her numbers, and MISSes faded past half (the reader reads a popup's first half second, at full opacity)."""
    frame = _png(path)
    assert reader.find(frame, [(0, 0, frame.shape[1], frame.shape[0])]) == []
    got = reader.best(frame)
    assert got is None or got[2] < 0.56


def test_drawn_faded_it_reads_to_80_percent(reader):
    """The game's picture drawn over a hunting frame at 100% and 80% opacity reads (0.94, 0.80), at 60% it doesn't (it
    fades from about 0.4 s: perception.misses)."""
    back = _png(FIXTURES / "not_numbers_20261026-231501_129.png").copy()
    back[:] = back[:, :1].mean(axis=(0, 1)).astype(np.uint8)  # (a plain piece of the scene)
    pic = np.asarray(Image.open(misses.FOLDER / misses.PICTURE).convert("RGBA")).astype(np.float32)
    scores = {}
    for alpha in (1.0, 0.8, 0.6):
        f = back.astype(np.float32).copy()
        a = pic[:, :, 3:4] / 255.0 * alpha
        f[40:78, 40:138] = pic[:, :, :3] * a + f[40:78, 40:138] * (1 - a)
        got = reader.best(f.astype(np.uint8))
        scores[alpha] = 0.0 if got is None else got[2]
    assert scores[1.0] > 0.9 and scores[0.8] > 0.75 and scores[0.6] < misses.MISS_OK


def test_the_picture_is_the_copy_s(tmp_path):
    from gamedata.paths import COPY
    if not (COPY / "Effect" / "Effect_000.wz").is_file():
        pytest.skip("needs the game data's copy (gamedata/data/copy)")
    out = misses.cut_assets(tmp_path)
    a = np.asarray(Image.open(out).convert("RGBA"))
    b = np.asarray(Image.open(misses.FOLDER / misses.PICTURE).convert("RGBA"))
    assert a.shape == b.shape == (38, 98, 4) and (a == b).all()


# --- the watch: a press with a MISS over its target took effect ------------------------------------------------------

def mon(x=100.0, bar=0.5, y=-450.0, kind="horny-mushroom") -> Monster:
    return Monster(SimpleNamespace(kind=kind), x, y, 23, bar)


def _look(w: RegularWatch, t: float, m: Monster, miss=None) -> None:
    w.look(t, [m], SimpleNamespace(misses=[] if miss is None else [miss]))


def test_a_press_with_her_miss_over_its_target_resets_the_run():
    """Four presses without effect, then one whose look 0.5 s after its key has her MISS over its target (60 over its
    feet): 'missed', the run starts again; five more without effect: the doubt (counted from there)."""
    w = RegularWatch()
    t = 0.0
    for _ in range(DOUBT_PRESSES - 1):
        m = mon()
        w.pressed(t, m)
        _look(w, t + 1.0, mon())
        t += 1.5
    assert len(w.run) == DOUBT_PRESSES - 1
    w.pressed(t, mon())
    _look(w, t + 0.5, mon(), miss=(105.0, -510.0))
    assert w.counts["missed"] == 1 and w.run == [] and w.due() is None
    t += 1.5
    for _ in range(DOUBT_PRESSES):
        w.pressed(t, mon())
        _look(w, t + 1.0, mon())
        t += 1.5
    assert w.due() is not None


@pytest.mark.parametrize("miss, why", [
    ((300.0, -510.0), "another monster's: 200 across"),
    ((100.0, -700.0), "250 over its feet"),
    ((100.0, -430.0), "under its feet"),
])
def test_a_miss_elsewhere_isn_t_hers_at_it(miss, why):
    w = RegularWatch()
    w.pressed(0.0, mon())
    _look(w, 0.7, mon(), miss=miss)
    _look(w, 1.0, mon())
    assert w.counts["missed"] == 0 and w.counts["none"] == 1, why


def test_too_soon_or_late_or_a_cast_s_isn_t_the_press_s():
    """A MISS before MISS_FROM_S after the key, or past MISS_BY_S: not the press's; one a Lucky Seven cast at it keyed
    after the press could have made: passed over."""
    w = RegularWatch()
    w.pressed(0.0, mon())
    _look(w, 0.1, mon(), miss=(100.0, -510.0))
    _look(w, 1.0, mon())
    assert w.counts["none"] == 1
    w = RegularWatch()
    m = mon()
    w.pressed(0.0, m)
    w.cast(0.3, m)
    _look(w, 0.6, mon(), miss=(100.0, -510.0))
    assert w.counts["passed: a cast's MISS maybe"] == 1 and w.counts["missed"] == 0
    assert keycheck.MISS_FROM_S < 0.6 < keycheck.MISS_BY_S
