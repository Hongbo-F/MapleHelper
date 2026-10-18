import pytest

from maplehelper import config, keys
from maplehelper.app import DEFAULT_CONFIG
from maplehelper.combos import COMBOS
from maplehelper.controller import Controller
from maplehelper.engine import LEFT, RIGHT

VK = {name: keys.parse(name).vk for name in ("f1", "f2", "f4", "f12", "esc", "left", "right", "a")}


class FakeRunner:
    def __init__(self):
        self.facing = RIGHT
        self.triggered: list[str] = []
        self.aborts = 0
        self.slow_toggles = 0

    def trigger(self, name):
        self.triggered.append(name)
        return True

    def abort(self):
        self.aborts += 1

    def toggle_slow(self):
        self.slow_toggles += 1
        return True


class Window:
    title = "MapleStory"


@pytest.fixture
def setup():
    cfg = config.load(DEFAULT_CONFIG, (c.name for c in COMBOS))
    runner, window = FakeRunner(), Window()
    return Controller(cfg, runner, lambda: window.title), runner, window


def test_hotkey_in_game_triggers_combo_and_is_hidden_from_game(setup):
    ctl, runner, _ = setup
    assert ctl.on_key(VK["f2"], True) is True
    assert ctl.on_key(VK["f2"], False) is True
    assert runner.triggered == ["back_off"]


def test_hotkey_outside_game_passes_through(setup):
    ctl, runner, window = setup
    window.title = "Notepad"
    assert ctl.on_key(VK["f2"], True) is False
    assert ctl.on_key(VK["f2"], False) is False
    assert runner.triggered == []


def test_holding_a_hotkey_triggers_once(setup):
    ctl, runner, _ = setup
    for _ in range(5):  # auto-repeat
        assert ctl.on_key(VK["f1"], True) is True
    ctl.on_key(VK["f1"], False)
    ctl.on_key(VK["f1"], True)
    assert runner.triggered == ["jump_throw", "jump_throw"]


def test_release_is_hidden_even_if_focus_left_the_game(setup):
    ctl, _, window = setup
    ctl.on_key(VK["f4"], True)
    window.title = "Discord"
    assert ctl.on_key(VK["f4"], False) is True


def test_arrows_pass_through_and_set_facing(setup):
    ctl, runner, _ = setup
    assert ctl.on_key(VK["left"], True) is False
    assert runner.facing == LEFT
    assert ctl.on_key(VK["right"], True) is False
    assert runner.facing == RIGHT


def test_abort_passes_through_and_stops_combo(setup):
    ctl, runner, _ = setup
    assert ctl.on_key(VK["esc"], True) is False
    assert runner.aborts == 1


def test_slow_mode_toggle_is_hidden_from_game(setup):
    ctl, runner, _ = setup
    assert ctl.on_key(VK["f12"], True) is True
    assert ctl.on_key(VK["f12"], False) is True
    assert runner.slow_toggles == 1


def test_other_keys_pass_through(setup):
    ctl, runner, _ = setup
    assert ctl.on_key(VK["a"], True) is False
    assert ctl.on_key(VK["a"], False) is False
    assert runner.triggered == []
