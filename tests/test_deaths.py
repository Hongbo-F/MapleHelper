"""A monster's death seen (perception.deaths): the game data's die1 frames matched where a track's feet are, on crops of
the Lie Detector's recording of 2026-10-26 23:10 (tests/fixtures/deaths: a Horny and a Green Mushroom dying, die1/3 and
die1/2, and each standing a moment before; their feet at the crop's (120, 150)); and the frame step asking it only of
the tracks that may be dying (deaths.wants), the tracker told of each seen."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab.perception import deaths
from gamelab.perception.deaths import DeathWatch

FIXTURES = Path(__file__).parent / "fixtures" / "deaths"
TREE = {19: "Horny Mushroom", 13: "Green Mushroom", 7: "Slime"}  # Tree Dungeon's monsters (mapdata: its spawns)


@pytest.fixture(scope="module")
def watch():
    try:
        w = DeathWatch.of(TREE, 1.0)  # (the game's own 1366x768: a game px a look-frame px)
    except (ImportError, FileNotFoundError, OSError):
        pytest.skip("needs the game data (gamedata/data)")
    if not w.pictures:
        pytest.skip("needs the game data's Mob pictures")
    return w


def _crop(name):
    e = json.loads((FIXTURES / "index.json").read_text())[name]
    return np.asarray(Image.open(FIXTURES / name).convert("RGB")), e["kind"], e["feet"]


class _Look:
    def __init__(self, frame):
        self.frame = frame


def test_the_die1_frames_are_the_game_datas_each_both_ways(watch):
    assert set(watch.pictures) == {"horny-mushroom", "green-mushroom", "slime"}
    assert sorted(p.index for p in watch.pictures["horny-mushroom"]) == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("name, index", [("horny_dying.png", 3), ("green_dying.png", 2)])
def test_a_dying_mushroom_is_seen_dying_where_its_feet_are(watch, name, index):
    frame, kind, (fx, fy) = _crop(name)
    score, got = watch.best(frame, kind, fx, fy)
    assert got == index and score >= 0.95
    assert watch.dying(_Look(frame), kind, fx, fy, None) and watch.dying(_Look(frame), kind, fx, fy, 0.7)


@pytest.mark.parametrize("name", ["horny_alive.png", "green_alive.png"])
def test_one_standing_isnt(watch, name):
    frame, kind, (fx, fy) = _crop(name)
    assert not watch.dying(_Look(frame), kind, fx, fy, None)


def test_die1_0_is_her_standing_picture_eyes_shut_so_it_counts_only_where_she_wasnt_re_found(watch, monkeypatch):
    """die1/0 is the standing picture with its eyes shut: matched DIE0_GOOD, it's a death only on a track not re-found
    in the frame (a live one standing there is re-found)."""
    frame, kind, (fx, fy) = _crop("horny_dying.png")
    monkeypatch.setattr(watch, "best", lambda *a: (0.8, 0))
    assert watch.dying(_Look(frame), kind, fx, fy, None) and not watch.dying(_Look(frame), kind, fx, fy, 0.85)


def test_only_tracks_re_found_under_their_good_score_or_lost_lately_are_looked_at():
    assert deaths.wants(0.8, 0.0) and not deaths.wants(0.95, 0.0)
    assert deaths.wants(None, 0.3) and not deaths.wants(None, deaths.RECENT_S + 0.1) and not deaths.wants(None, 0.0)


def test_the_frame_step_tells_the_tracker_of_the_deaths_its_watch_sees():
    """WorldStep._deaths: each track that may be dying, every EVERY-th frame (by its id), looked at where its feet are
    (re-found: as found; lost: as last seen, the camera's move taken out), and those seen dying told (seen_dying)."""
    from types import SimpleNamespace

    from gamelab.localization.monsters import Monster
    from gamelab.localization.tracking import LIVE, MonsterTracker, Track
    from gamelab.localization.worldstate import WorldStep
    from gamelab.perception.mobs import Mob

    asked = []

    class Watch:
        pictures = {"pig": ()}

        def dying(self, look, kind, fx, fy, refound):
            asked.append((kind, round(fx), round(fy), refound))
            return refound is None

    step = WorldStep.__new__(WorldStep)
    step.tracker, step.seq, step._refinds, step.counts, step._real = MonsterTracker(), 2, None, {}, False
    step.eyes = SimpleNamespace(deaths=Watch())
    mob = Mob("pig", 100, 90, 40, 30, 0.8, 100.0, 105.0, 1, "move")
    for tid, last in ((2, 9.95), (4, 9.6), (6, 9.0)):  # re-found now; lost 0.4 s ago; lost 1 s ago
        step.tracker.tracks[tid] = Track(tid, "pig", 1, 500.0 + tid, 300.0, mob, 0.0, last, state=LIVE, seen_x=500.0)
    s = SimpleNamespace(screen_x=lambda x: x - 400.0, screen_y=lambda y: y - 200.0, frame=None)
    work = {}
    step._deaths(s, 10.0, {2: Monster(mob, 502.0, 300.0, 1, track=2)}, work)
    assert asked == [("pig", 100, 105, 0.8), ("pig", 100, 100, None)] and work["deaths"] == 2
    assert step.tracker.tracks[4].died_seen == 10.0 and step.tracker.tracks[2].died_seen is None
