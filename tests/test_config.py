import tomllib

import pytest

from maplehelper import config
from maplehelper.app import DEFAULT_CONFIG
from maplehelper.combos import COMBOS

NAMES = [c.name for c in COMBOS]


def default_data():
    with open(DEFAULT_CONFIG, "rb") as f:
        return tomllib.load(f)


def parse(data):
    return config.parse(data, NAMES)


def test_shipped_config_loads():
    cfg = config.load(DEFAULT_CONFIG, NAMES)
    assert cfg.keys.jump.name == "alt"
    assert cfg.keys.lucky_seven.name == "shift"
    assert set(cfg.hotkeys) == set(NAMES)
    assert cfg.timing.throw_gap == 650


def test_empty_skill_means_not_bound():
    data = default_data()
    data["keys"]["lucky_seven"] = ""
    assert parse(data).keys.lucky_seven is None


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["keys"].update(jump="altt"), "keys.jump: unknown key 'altt'"),
    (lambda d: d["keys"].pop("attack"), "missing ['attack']"),
    (lambda d: d["keys"].update(haste="h"), "unknown action(s) ['haste']"),
    (lambda d: d["hotkeys"].update(back_off="shift"), "already used by keys.lucky_seven"),
    (lambda d: d["hotkeys"].update(back_off="f1"), "already used by hotkeys.jump_throw"),
    (lambda d: d["hotkeys"].update(panic="esc"), "already used by control.abort"),
    (lambda d: d["hotkeys"].update(fly="f5"), "unknown combo(s) ['fly']"),
    (lambda d: d["hotkeys"].update(back_off=2), "must be a key name in quotes"),
    (lambda d: d["timing"].update(throw_gapp=600), "unknown setting(s) ['throw_gapp']"),
    (lambda d: d["timing"].update(throw_gap=-1), "timing.throw_gap must be"),
    (lambda d: d["timing"].update(throw_gap=True), "timing.throw_gap must be"),
    (lambda d: d.update(game_window_title=""), "game_window_title"),
])
def test_bad_config_is_explained(mutate, message):
    data = default_data()
    mutate(data)
    with pytest.raises(config.ConfigError) as err:
        parse(data)
    assert message in str(err.value)


def test_missing_file_is_explained(tmp_path):
    with pytest.raises(config.ConfigError, match="not found"):
        config.load(tmp_path / "nope.toml", NAMES)
