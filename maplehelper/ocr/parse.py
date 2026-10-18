"""Turning OCR'd lines into game values, tolerant of the ways OCR misreads game text.

OCR on game fonts confuses a few characters often enough to plan for: "+" read as "t" or "4", "1" as
"l", "I", "i" or "|", "0" as "O" or "o", "m" as "rn". Numbers are folded back where a number is expected,
and a value read from a doubtful character is marked unsure, so a caller can wait for a clearer read.
"""

import difflib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .engine import Row
from .exp_table import EXP_TO_NEXT

# Characters OCR puts where a digit should be, mapped back. Only the common, unambiguous ones: folding
# letters like S or B too would turn a label stuck to a number into digits.
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "i": "1", "|": "1"})
_NUM = r"\d[\d,.OolIi|]*"  # starts with a real digit; folded by _number()


def _number(token: str) -> int | None:
    digits = token.translate(_DIGIT_FIX).replace(",", "").replace(".", "")
    return int(digits) if digits.isdigit() else None


# --- the bottom bar ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HudReading:
    hp: int | None = None
    hp_max: int | None = None
    mp: int | None = None
    mp_max: int | None = None
    exp: int | None = None          # EXP points into the current level
    exp_percent: float | None = None
    level: int | None = None
    level_source: str | None = None  # "exp": worked out from EXP and the EXP table; "read": the digits as read


_HP = re.compile(rf"\bH\s?P\W*({_NUM})\s*[/\\]\s*({_NUM})", re.I)
_MP = re.compile(rf"\bM\s?P\W*({_NUM})\s*[/\\]\s*({_NUM})", re.I)
_EXP = re.compile(rf"\bE\s?X\s?P\W*({_NUM})\s*[\[({{]?\s*(\d{{1,3}}(?:[.,]\d{{1,3}})?)\s*%", re.I)
_LEVEL = re.compile(r"\bL\s?V\s?\.?\s*(\d{1,3})\b", re.I)
_LONE_NUMBER = re.compile(r"(\d{1,3})")
_GAUGE = re.compile(r"\b(?:H\s?P|M\s?P|E\s?X\s?P)\b", re.I)

LEVEL_TOLERANCE = 0.02  # EXP-for-next-level worked out from points / percent must be this close to the table's
MIN_PERCENT = 1.0       # below this, the two decimals of the percentage are too coarse to pin a level
MAX_LEVEL = 200


def parse_hud(rows: Iterable[Row], exp_table: Mapping[int, int] | None = EXP_TO_NEXT) -> HudReading | None:
    """HP, MP, EXP and level from the bottom bar's lines; None when they don't look like the bar.

    `exp_table` maps level -> EXP needed to reach the next level (Classic World's by default; None for none).
    With it, the level comes from the EXP, because the game's stylized level digits misread often ("19" as
    "09"). A level read from the digits is kept only when the EXP fits it."""
    rows = list(rows)
    text = "  ".join(r.text for r in rows)
    hp, mp, exp = _pair(_HP, text), _pair(_MP, text), _EXP.search(text)
    if not hp and not exp:
        return None

    points = percent = None
    if exp:
        points, percent = _number(exp.group(1)), float(exp.group(2).replace(",", "."))
        if not 0 <= percent <= 100:
            points = percent = None

    level, source = None, None
    if exp_table and points is not None and percent is not None:
        level = level_from_exp(points, percent, exp_table)
        source = "exp" if level else None
    if level is None and (digits := _level_digits(rows)):
        for candidate in _level_candidates(digits):
            if not exp_table or points is None or _exp_fits(candidate, points, percent, exp_table):
                level, source = candidate, "read"
                break

    return HudReading(*(hp or (None, None)), *(mp or (None, None)), points, percent, level, source)


def _level_digits(rows: list[Row]) -> str | None:
    # The level as drawn: "LV. 19" in one row, or its digits alone in a row left of the HP/MP/EXP gauges.
    for row in rows:
        if m := _LEVEL.search(row.text):
            return m.group(1)
    gauge = min((r.left for r in rows if _GAUGE.search(r.text)), default=None)
    for row in rows:
        if (gauge is None or row.right <= gauge) and (m := _LONE_NUMBER.fullmatch(row.text.strip())):
            return m.group(1)
    return None


def _level_candidates(digits: str) -> list[int]:
    # A level never starts with 0: the stylized 1 reads as 0 ("19" as "09", "10" as "00"). Try the repaired
    # digits first, then the digits as read.
    tries = ["1" + digits[1:], digits.lstrip("0")] if digits.startswith("0") else [digits]
    return [int(d) for d in tries if d and 1 <= int(d) <= MAX_LEVEL]


def _exp_fits(level: int, points: int, percent: float | None, exp_table: Mapping[int, int]) -> bool:
    # A level is possible when its requirement holds the points, and (with a percentage) points / percent
    # lands near it. The percentage shows two decimals, so it can be off by 0.01: more slack the smaller it is.
    need = exp_table.get(level)
    if not need or points > need:
        return False
    if not percent:
        return True
    slack = LEVEL_TOLERANCE + 0.01 / percent
    return abs(points * 100 / percent - need) <= slack * need


def level_from_exp(points: int, percent: float, exp_table: Mapping[int, int]) -> int | None:
    """The one level whose EXP requirement matches points / percent, or None when no single level does."""
    if percent < MIN_PERCENT or points <= 0:
        return None
    needed = points * 100 / percent
    hits = [lv for lv, need in exp_table.items() if need and abs(need - needed) <= LEVEL_TOLERANCE * need]
    return hits[0] if len(hits) == 1 else None


def _pair(pattern: re.Pattern, text: str) -> tuple[int, int] | None:
    # "current / max", dropped when it can't be right (current above max). The bar writes "HP[642/642]",
    # and OCR reads the closing "]" as "1" often enough ("HP[642/6421"): with an opening "[" and no closing
    # one, a trailing 1-like character is that bracket.
    if not (m := pattern.search(text)):
        return None
    cur_raw, top_raw = m.group(1), m.group(2)
    opened = "[" in text[m.start():m.start(1)]
    closed = text[m.end(2):m.end(2) + 1] in ("]", ")")
    if opened and not closed and len(top_raw) > 1 and top_raw[-1] in "1lIi|":
        top_raw = top_raw[:-1]
    cur, top = _number(cur_raw), _number(top_raw)
    if cur is None or top is None or top == 0 or cur > top:
        return None
    return cur, top


# --- pickup messages --------------------------------------------------------------------------

@dataclass(frozen=True)
class LootLine:
    kind: str                     # "exp", "mesos" or "item"
    amount: int | None = None     # EXP or mesos gained
    item: str | None = None       # item name as read (or as matched, when names were given)
    quantity: int | None = None
    sure: bool = False            # read with its "+" sign / clean digits; False: a doubtful character was folded


_AMOUNT = re.compile(r"[(\[{]\s*([+t4]?)\s*(\d[\d,.]*)\s*[)\]}]")
_EARNED = r"e\s?[ae]?\s?r?\s?(?:rn|[nm]){1,2}\s?e?d\b"   # earned, eamed, earmed, eaned, earnd...
_ITEM = re.compile(rf"^(?P<name>.+?)\s*[xX×]\s*(?P<n>[\dlIi|]{{1,4}})\s*\.?\s*{_EARNED}", re.I)
_EXP_LINE = re.compile(r"\bE\s?X\s?P\b", re.I)
_RECEIVED = re.compile(r"(?:re)?cei?v|recei|rece|re\w?[ie]{1,2}v", re.I)  # also OCR's reeived, recieved
_MESOS = re.compile(r"\b(?:m|rn)[e3]s[o0]s?\b|\b(?:m|rn)es\b|meso", re.I)
_GAINED = re.compile(r"gain|have", re.I)


def parse_loot(text: str, item_names: Iterable[str] | None = None) -> LootLine | None:
    """One pickup message, e.g. "You received EXP (+21)", "You have gained mesos (+27)",
    "Leaf x1 earned. (Etc)"; None for anything else. With `item_names`, an item's name is matched to them."""
    text = " ".join(text.split())
    if m := _ITEM.search(text):
        name = m.group("name").strip(" .,:;-")
        qty_raw = m.group("n")
        qty = _number(qty_raw)
        if item_names is not None:
            name = match_name(name, item_names) or name
        return LootLine("item", item=name, quantity=qty, sure=qty_raw.isdigit())
    amount = _AMOUNT.search(text)
    if amount is None:
        return None
    sign, value = amount.group(1), _number(amount.group(2))
    sure = sign == "+"
    if _EXP_LINE.search(text) and _RECEIVED.search(text):
        return LootLine("exp", amount=value, sure=sure)
    if _MESOS.search(text) and (_GAINED.search(text) or "meso" in text.lower()):
        return LootLine("mesos", amount=value, sure=sure)
    return None


# --- names ------------------------------------------------------------------------------------

NAME_MIN = 0.8      # the best match must score at least this (difflib ratio)...
NAME_MARGIN = 0.07  # ...and beat the runner-up by this much, or the read is too ambiguous to name

_FOLD = str.maketrans({"0": "o", "1": "l", "|": "l", "5": "s"})


def _plain(text: str) -> str:
    return " ".join(text.lower().split())


def _folded(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _plain(text).translate(_FOLD).replace("rn", "m"))


def match_name(text: str, names: Iterable[str]) -> str | None:
    """The known name `text` was most likely read from; None when no name is clearly the one."""
    names = list(names)
    plain = _plain(text)
    for name in names:
        if _plain(name) == plain:
            return name
    folded = _folded(text)
    scored = sorted(
        ((max(difflib.SequenceMatcher(None, plain, _plain(n)).ratio(),
              difflib.SequenceMatcher(None, folded, _folded(n)).ratio()), n) for n in names),
        reverse=True,
    )
    if not scored or scored[0][0] < NAME_MIN:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < NAME_MARGIN:
        return None
    return scored[0][1]
