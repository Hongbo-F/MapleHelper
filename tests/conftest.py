import copy

import pytest

# A fixed config for tests, so editing config.toml for her keys never breaks them.
BASE_CONFIG = {
    "game_window_title": "MapleStory",
    "keys": {"left": "left", "right": "right", "jump": "alt", "attack": "ctrl",
             "lucky_seven": "shift"},
    "hotkeys": {"jump_throw": "f1", "back_off": "f2", "hop_back": "f3", "jump_back_shot": "f5"},
    "control": {"abort": "esc", "slow_mode": "f12"},
    "timing": {"throw_gap": 650},
}


@pytest.fixture
def config_data():
    return copy.deepcopy(BASE_CONFIG)
