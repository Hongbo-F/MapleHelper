import pytest

from maplehelper import config
from maplehelper.app import DEFAULT_CONFIG
from maplehelper.combos import COMBOS

NAMES = [c.name for c in COMBOS]


def test_shipped_config_is_valid():
    cfg = config.load(DEFAULT_CONFIG, NAMES)
    assert set(cfg.hotkeys) <= set(NAMES)


def test_parses_keys_hotkeys_and_timing(config_data):
    cfg = config.parse(config_data, NAMES)
    assert cfg.keys.jump.name == "alt"
    assert cfg.keys.lucky_seven.name == "shift"
    assert cfg.hotkeys["jump_back_shot"].name == "f5"
    assert cfg.timing.throw_gap == 650
    assert cfg.timing.air_delay == config.Timing().air_delay  # default when not set


def test_empty_skill_means_not_bound(config_data):
    config_data["keys"]["lucky_seven"] = ""
    assert config.parse(config_data, NAMES).keys.lucky_seven is None


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["keys"].update(jump="altt"), "keys.jump: unknown key 'altt'"),
    (lambda d: d["keys"].pop("attack"), "missing ['attack']"),
    (lambda d: d["keys"].update(haste="h"), "unknown action(s) ['haste']"),
    (lambda d: d["hotkeys"].update(back_off="shift"), "already used by keys.lucky_seven"),
    (lambda d: d["hotkeys"].update(back_off="f1"), "already used by hotkeys.jump_throw"),
    (lambda d: d["hotkeys"].update(hop_back="esc"), "already used by control.abort"),
    (lambda d: d["keys"].update(dark_sight="insert"), "unknown action(s) ['dark_sight']"),
    (lambda d: d["hotkeys"].update(panic="f4"), "unknown combo(s) ['panic']"),
    (lambda d: d["hotkeys"].update(fly="f6"), "unknown combo(s) ['fly']"),
    (lambda d: d["hotkeys"].update(back_off=2), "must be a key name in quotes"),
    (lambda d: d["timing"].update(throw_gapp=600), "unknown setting(s) ['throw_gapp']"),
    (lambda d: d["timing"].update(throw_gap=-1), "timing.throw_gap must be"),
    (lambda d: d["timing"].update(throw_gap=True), "timing.throw_gap must be"),
    (lambda d: d.update(game_window_title=""), "game_window_title"),
])
def test_bad_config_is_explained(config_data, mutate, message):
    mutate(config_data)
    with pytest.raises(config.ConfigError) as err:
        config.parse(config_data, NAMES)
    assert message in str(err.value)


def test_missing_file_is_explained(tmp_path):
    with pytest.raises(config.ConfigError, match="not found"):
        config.load(tmp_path / "nope.toml", NAMES)
