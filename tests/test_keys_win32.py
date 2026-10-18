import ctypes

import pytest

from maplehelper import keys, win32


def test_parse_is_case_insensitive_and_labels_like_keycaps():
    assert keys.parse(" Alt ").name == "alt"
    assert keys.parse("left").label == "←"
    assert keys.parse("shift").label == "SHIFT"


def test_unknown_key_lists_valid_names():
    with pytest.raises(ValueError, match="valid keys: a, b"):
        keys.parse("hyper")


def test_arrows_and_nav_keys_are_extended():
    assert all(keys.parse(k).extended for k in ("left", "right", "up", "down", "insert", "delete", "home"))
    assert not keys.parse("alt").extended


def test_name_of_round_trips():
    assert keys.name_of(keys.parse("f2").vk) == "f2"
    assert keys.name_of(0xFF) == "vk_0xff"


@pytest.mark.skipif(ctypes.sizeof(ctypes.c_void_p) != 8, reason="sizes below are for 64-bit Python")
def test_struct_sizes_match_windows_64bit():
    assert ctypes.sizeof(win32.INPUT) == 40
    assert ctypes.sizeof(win32.KBDLLHOOKSTRUCT) == 24


@pytest.mark.parametrize("name", ["left", "right", "alt", "ctrl", "shift", "insert", "delete", "f1", "f12", "a", "1"])
def test_keys_have_scan_codes(name):
    assert win32.scan_code(keys.parse(name)) > 0
