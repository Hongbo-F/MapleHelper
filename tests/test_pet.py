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


def _rows(fullness):
    """What the OCR read off the panel at the saved screenshots (nm_pet1-3), the fullness field as given."""
    rows = [SimpleNamespace(text="TYPE", score=1.0, x=36, y=11, height=22, left=19, right=53),
            SimpleNamespace(text="Snail", score=1.0, x=94, y=11, height=22, left=76, right=113),
            SimpleNamespace(text="CLOSENESS", score=1.0, x=35, y=47, height=22, left=1, right=70),
            SimpleNamespace(text="8", score=1.0, x=83, y=49, height=21, left=75, right=92),
            SimpleNamespace(text="FULLNESS", score=1.0, x=35, y=68, height=22, left=4, right=66)]
    if fullness is not None:
        rows.append(SimpleNamespace(text=str(fullness), score=0.99, x=87, y=69, height=20, left=76, right=98))
    return rows


class _Game:
    """The window open or not, its fullness, and what the stack pressed."""

    def __init__(self, fullness, opens=True):
        self.fullness, self.opens, self.open, self.plans = fullness, opens, False, []

    def read(self, image):
        return _rows(self.fullness) if self.open else [SimpleNamespace(text="Quest Helper", score=1.0, x=1, y=1,
                                                                       height=10, left=0, right=50)]


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
    assert saved == {"checked": 1000.0, "fed": 1000.0, "fullness": 99}


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
