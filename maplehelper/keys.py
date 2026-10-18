"""Key names used in config.toml, mapped to Windows virtual-key codes."""

from dataclasses import dataclass
import string

# name -> (virtual-key code, is an "extended" key on the keyboard)
# Left/right modifier codes are used because the low-level hook reports those, never the generic ones.
_SPECIAL = {
    "left": (0x25, True), "up": (0x26, True), "right": (0x27, True), "down": (0x28, True),
    "shift": (0xA0, False), "rshift": (0xA1, False),
    "ctrl": (0xA2, False), "rctrl": (0xA3, True),
    "alt": (0xA4, False), "ralt": (0xA5, True),
    "space": (0x20, False), "enter": (0x0D, False), "tab": (0x09, False),
    "esc": (0x1B, False), "backspace": (0x08, False), "capslock": (0x14, False),
    "insert": (0x2D, True), "delete": (0x2E, True),
    "home": (0x24, True), "end": (0x23, True),
    "pageup": (0x21, True), "pagedown": (0x22, True),
    "pause": (0x13, False), "scrolllock": (0x91, False),
    "`": (0xC0, False), "-": (0xBD, False), "=": (0xBB, False),
    "[": (0xDB, False), "]": (0xDD, False), "\\": (0xDC, False),
    ";": (0xBA, False), "'": (0xDE, False),
    ",": (0xBC, False), ".": (0xBE, False), "/": (0xBF, False),
}

_LABELS = {"left": "←", "right": "→", "up": "↑", "down": "↓"}

KEY_CODES: dict[str, tuple[int, bool]] = {
    **{c: (ord(c.upper()), False) for c in string.ascii_lowercase},
    **{d: (ord(d), False) for d in string.digits},
    **{f"f{n}": (0x6F + n, False) for n in range(1, 13)},
    **_SPECIAL,
}


@dataclass(frozen=True)
class Key:
    name: str
    vk: int
    extended: bool

    @property
    def label(self) -> str:
        """How the key is written on the coach overlay, like a keycap."""
        return _LABELS.get(self.name, self.name.upper())


_NAMES = {vk: name for name, (vk, _) in KEY_CODES.items()}


def name_of(vk: int) -> str:
    return _NAMES.get(vk, f"vk_{vk:#04x}")


def parse(name: str) -> Key:
    norm = name.strip().lower()
    if norm not in KEY_CODES:
        raise ValueError(f"unknown key {name!r}; valid keys: {', '.join(KEY_CODES)}")
    vk, extended = KEY_CODES[norm]
    return Key(norm, vk, extended)
