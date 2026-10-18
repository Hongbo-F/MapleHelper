"""Reading the game's text from the screen: the bottom bar (HP, MP, EXP, level) and pickup messages.

Read-only: screen grabs plus OCR on this computer. Needs the OCR packages in environment.yml (the combo
helper doesn't). Try it with `python -m maplehelper.ocr hud` while the game is open.
"""

from .engine import EngineSettings, OcrEngine, Row, shared
from .exp_table import EXP_TO_NEXT
from .loot import LootRead, LootReader
from .parse import HudReading, LootLine, level_from_exp, match_name, parse_hud, parse_loot
from .regions import HUD, LOOT, PRESETS, Preset, Region, content_box

__all__ = [
    "EngineSettings", "OcrEngine", "Row", "shared", "EXP_TO_NEXT", "LootRead", "LootReader",
    "HudReading", "LootLine", "level_from_exp", "match_name", "parse_hud", "parse_loot",
    "HUD", "LOOT", "PRESETS", "Preset", "Region", "content_box",
]
