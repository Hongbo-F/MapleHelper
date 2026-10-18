"""Runs the real OCR models on the sample image (skipped when the OCR packages aren't installed)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rapidocr")
from PIL import Image  # noqa: E402

from maplehelper.ocr import EngineSettings, parse_hud, parse_loot, shared  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "loot_test.png"


@pytest.fixture(scope="module")
def sample():
    return np.asarray(Image.open(SAMPLE).convert("RGB"))[:, :, ::-1]  # BGR, as the screen grabs give it


@pytest.fixture(scope="module")
def engine():
    return shared(EngineSettings(device="cpu", threads=1))


def test_engine_is_shared():
    assert shared(EngineSettings(threads=1)) is shared(EngineSettings(threads=1))


def test_reads_sample_lines_in_original_pixels(engine, sample):
    rows = engine.read(sample, min_height=450)  # enlarged ~4.7x for OCR
    assert [r.text for r in rows][:3] == ["You received EXP (+21)", "You have gained mesos (+27)", "Leaf x1 earned. (Etc)"]
    ys = [r.y for r in rows]
    assert ys == sorted(ys)
    assert all(0 < r.y < sample.shape[0] and 0 < r.height < 30 for r in rows)  # back in the 96 px tall original


def test_sample_parses(engine, sample):
    rows = engine.read(sample, min_height=450)
    loot = [parse_loot(r.text) for r in rows]
    assert [(l.kind, l.amount or l.quantity) for l in loot if l] == [("exp", 21), ("mesos", 27), ("item", 1)]
    hud = parse_hud(rows)
    assert (hud.hp, hud.hp_max, hud.mp, hud.mp_max, hud.exp, hud.exp_percent) == (1234, 1500, 567, 890, 12345, 45.67)


def test_runs_on_the_cpu(engine, sample):
    engine.read(sample)
    assert set(engine.providers().values()) == {"CPU"}
