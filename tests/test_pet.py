"""Her pet kept fed (behavior.pet): its fullness read off her Character Info, fed while low, the window closed; and a
plan's name as the player keeps it."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from maplehelper import keys

from gamelab import paths
from gamelab.behavior import pet
from gamelab.behavior.pet import PetCare, read_fullness
from gamelab.control import client
from gamelab.screen import geometry

NATIVE = geometry(1366, 768)  # full-size looks at the game's own size


def _row(text, x, y, half=20, height=20):
    return SimpleNamespace(text=text, score=1.0, x=x, y=y, height=height, left=x - half, right=x + half)


def _rows(fullness, level=24, name="BugBaby", shown=True):
    """What the OCR read off the window (nm_pet3, nm_pet7, at the px it gave): its title, the name under its portrait,
    LEVEL; the pet panel with the fullness as given if `shown`, else its SHOW PET INFO button (read garbled, as the OCR
    read it)."""
    rows = [_row("CHARACTER INFO", 299, 190, 40), _row(name, 318, 314, 25),
            _row("LEVEL", 414, 238, 20), _row(str(level), 464, 238, 7)]
    if not shown or fullness is None:
        return rows + [_row("SHO U PET IHFO", 528, 344, 40)]
    return rows + [_row("TYPE", 414, 360, 15), _row("Snail", 470, 360, 15),
                   _row("CLOSENESS", 414, 397, 30), _row("8", 464, 397, 5),
                   _row("FULLNESS", 414, 418, 30), _row(str(fullness), 464, 418, 10)]


class _Game:
    """The window open or not (hers, or another player's: `theirs`), its fullness, and what the stack pressed."""

    def __init__(self, fullness, opens=True, theirs=False, shown=True):
        self.fullness, self.opens, self.theirs, self.shown = fullness, opens, theirs, shown
        self.open, self.plans = False, []

    def read(self, image):
        if image.shape[0] < 60:  # her bottom bar's name box: her job above her name
            return [_row("Rogue", 30, 8), _row("BugBaby", 36, 21)]
        if not self.open:
            return [_row("Quest Helper", 30, 1)]
        if self.theirs:
            return _rows(None, level=20, name="Lioryl", shown=False)
        return _rows(self.fullness, shown=self.shown)


class _Actions:
    def __init__(self, game, pet_food=True):
        self.game = game
        self.keys = SimpleNamespace(pet_food=keys.parse("u") if pet_food else None)

    def feed_pet(self):
        self.game.plans.append("pet_food")
        self.game.fullness = min(100, self.game.fullness + 30)
        return True

    def play(self, events, name):
        self.game.plans.append(name)
        if name == "pet_info" and self.game.opens:
            assert [e[1] for e in events].count("lbutton") == 4  # a double-click
            self.game.open = True
        if name == "pet_close":
            self.game.open = False
        if name == "pet_panel":
            self.game.shown = True


class _Eyes:
    def __init__(self, tag=(922, 540)):
        self.tag = tag

    def look(self):
        return SimpleNamespace(tag=self.tag, geo=NATIVE)

    def _capture(self):
        return np.zeros((768, 1366, 3), np.uint8)


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PET", tmp_path / "pet.json")
    monkeypatch.setattr(pet, "SETTLE_S", 0)
    monkeypatch.setattr(pet.telemetry, "emit", lambda *a, **k: None)
    return tmp_path / "pet.json"


def test_the_fullness_is_read_off_the_panel_and_nothing_without_it():
    game = _Game(39)
    game.open = True
    assert read_fullness(np.zeros((768, 1366, 3), np.uint8), game) == 39
    game.open = False
    assert read_fullness(np.zeros((768, 1366, 3), np.uint8), game) is None


def test_a_hungry_pet_is_fed_until_it_is_full_enough_and_the_window_closed(state):
    game = _Game(9)
    care = PetCare(_Actions(game), _Eyes(), ocr=game, log=lambda m: None, clock=lambda: 1000.0)
    assert care.check() == 99
    assert game.plans == ["pet_info", "pet_food", "pet_food", "pet_food", "pet_close"]
    assert not game.open
    saved = json.loads(state.read_text())
    assert saved == {"checked": 1000.0, "fed": 1000.0, "fullness": 99, "level": 99.0, "at": 1000.0, "read_at": 1000.0}


def test_a_fed_pet_gets_no_food_and_its_window_is_closed(state):
    game = _Game(75)
    care = PetCare(_Actions(game), _Eyes(), ocr=game, log=lambda m: None, clock=lambda: 1000.0)
    assert care.check() == 75
    assert game.plans == ["pet_info", "pet_close"]
    assert "fed" not in json.loads(state.read_text())


def test_esc_is_never_pressed_without_the_window_open_and_food_comes_only_after_a_long_while(state):
    game = _Game(50, opens=False)  # her tag found, but the window doesn't open (someone in the way)
    care = PetCare(_Actions(game), _Eyes(), ocr=game, log=lambda m: None, clock=lambda: 1000.0)
    state.write_text(json.dumps({"fed": 1000.0 - 60}))
    assert care.check() is None
    assert game.plans == ["pet_info"]  # no Esc: it would open the game menu
    care.clock = lambda: 1000.0 + pet.FALLBACK_S
    assert care.check() is None
    assert game.plans == ["pet_info", "pet_info", "pet_food"]


def test_another_player_s_window_is_closed_and_the_check_tried_again_soon(state):
    game = _Game(50, theirs=True)  # someone standing on her takes the double-click
    care = PetCare(_Actions(game), _Eyes(), ocr=game, log=lambda m: None, clock=lambda: 1000.0)
    assert care.check() is None
    assert game.plans == ["pet_info", "pet_close"] and not game.open  # closed, nothing fed
    assert json.loads(state.read_text())["retry_at"] == 1000.0 + pet.RETRY_S
    assert care.check() is None and game.plans == ["pet_info", "pet_close"]  # not again before then
    game.theirs = False
    care.clock = lambda: 1000.0 + pet.RETRY_S
    assert care.check() == 50 + 30 and game.plans[-3:] == ["pet_info", "pet_food", "pet_close"]


def test_her_window_s_collapsed_pet_panel_is_opened_before_it_is_read(state):
    game = _Game(60, shown=False)  # opened by a double-click on her: SHOW PET INFO
    care = PetCare(_Actions(game), _Eyes(), ocr=game, log=lambda m: None, clock=lambda: 1000.0)
    assert care.check() == 90
    assert game.plans == ["pet_info", "pet_panel", "pet_food", "pet_close"]


def test_a_check_is_due_every_so_often_and_never_without_a_pet_food_key(state):
    game = _Game(99)
    care = PetCare(_Actions(game), _Eyes(), ocr=game, log=lambda m: None, clock=lambda: 5000.0)
    assert care.due()
    state.write_text(json.dumps({"checked": 5000.0 - pet.CHECK_EVERY_S + 1}))
    assert not care.due()
    assert not PetCare(_Actions(game, pet_food=False), _Eyes(), ocr=game, clock=lambda: 5000.0).due()


def test_her_body_is_double_clicked_above_her_name_tag(state):
    game = _Game(99)
    seen = []
    actions = _Actions(game)
    actions.play = lambda events, name: seen.append((name, events)) or setattr(game, "open", name == "pet_info")
    PetCare(actions, _Eyes((922, 540)), ocr=game, log=lambda m: None).check()
    name, events = seen[0]
    assert name == "pet_info" and events[0] == [0, "mouse", 922, 540 - pet.BODY_ABOVE_TAG]


def test_a_plan_name_keeps_only_what_the_player_keeps(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "INBOX", tmp_path)
    plan_id = client.send([], name="pet food")
    assert "_petfood_" in plan_id and (tmp_path / f"{plan_id}.json").exists()


def test_read_once_then_followed_in_time_and_fed_when_it_falls_under_the_line(state):
    game = _Game(85)
    now = [10_000.0]
    care = PetCare(_Actions(game), _Eyes(), ocr=game, log=lambda m: None, clock=lambda: now[0])
    assert care.tend() == 85  # never read: read once
    assert game.plans == ["pet_info", "pet_close"]
    now[0] += 10 * pet.TICK_S  # ten ticks out: 75, nothing pressed
    assert care.tend() == pytest.approx(85 - 10 * pet.HUNGRY)
    assert game.plans == ["pet_info", "pet_close"]
    now[0] += 20 * pet.TICK_S  # 55: under the line, one food
    assert care.tend() == pytest.approx(55 + pet.FOOD)
    assert game.plans == ["pet_info", "pet_close", "pet_food"]
    saved = json.loads(state.read_text())
    assert saved["level"] == pytest.approx(85) and saved["fed"] == now[0] and saved["read_at"] == 10_000.0


def test_time_away_from_the_game_does_not_count(state):
    game = _Game(70)
    now = [0.0]
    care = PetCare(_Actions(game), _Eyes(), ocr=game, log=lambda m: None, clock=lambda: now[0])
    care.tend()
    now[0] += 100 * pet.TICK_S
    assert care.tend(present=False) == pytest.approx(70)  # the game closed meanwhile: the pet wasn't out
    now[0] += 5 * pet.TICK_S
    assert care.tend() == pytest.approx(65)


def test_a_hunt_reads_it_only_with_no_monster_near_her(state):
    game = _Game(90)
    care = PetCare(_Actions(game), _Eyes(), ocr=game, log=lambda m: None, clock=lambda: 0.0)
    assert care.tend(read_ok=False) is None and game.plans == []
