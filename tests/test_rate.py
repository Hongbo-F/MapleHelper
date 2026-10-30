"""Perception's one rate (localization.rate.HZ: 15 frames a second, everything once a frame; the user, 2026-10-30: "I
meant everything at 15 fps"): what counts frames takes it from there, and no other frames-a-second constant or 1/15,
1/30 of a second is written into localization/, perception/ or control/ (a rate changed in one place only)."""

import inspect
import io
import tokenize
from pathlib import Path

from gamelab.control import recording
from gamelab.localization import feed, rate, senses, service
from gamelab.localization.feed import SyncFeed

ROOT = Path(__file__).parent.parent / "gamelab"
# (file, name): the frames-a-second constants and gaps that aren't perception's rate
OTHER_RATES = {("localization/rate.py", "HZ"),             # the one
               ("localization/feed.py", "GAME_HZ"),        # the game's own 60
               ("control/recording.py", "FLOOR_FPS"),      # the old recorder's floor, which the recording keeps
               ("control/recording.py", "FLOOR_BACK_FPS"),
               ("control/rolling.py", "FPS"),              # `gamelab recorder`'s own grabs, 10 a second
               ("localization/locate.py", "PROTO_DT")}     # the prototype's frames' gap, its counts' unit


def test_everything_takes_the_one_rate():
    assert rate.HZ == 15.0
    assert feed.HZ is rate.HZ and service.HZ is rate.HZ and recording.FPS == rate.HZ
    assert inspect.signature(SyncFeed).parameters["hz"].default is rate.HZ
    assert inspect.signature(service.Service).parameters["hz"].default is rate.HZ
    assert senses.FEED_GLANCE_EVERY_S == 0.0  # (glances: every frame the feed looks at)


def _tokens(path: Path) -> list:
    """The file's tokens but comments and line ends (docstrings are STRING tokens: not looked in)."""
    return [t for t in tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline)
            if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)]


def test_no_other_rate_is_written_into_perception():
    found = []
    for folder in ("localization", "perception", "control"):
        for path in sorted((ROOT / folder).glob("*.py")):
            name, toks = f"{folder}/{path.name}", _tokens(path)
            for i, tok in enumerate(toks[:-2]):
                after = toks[i + 1].string, toks[i + 2]
                # a frames-a-second constant set to a number at the top level (NAME_HZ = 30.0, NAME_FPS = 15)
                if tok.type == tokenize.NAME and tok.string.endswith(("HZ", "FPS")) and tok.start[1] == 0 \
                        and after[0] == "=" and after[1].type == tokenize.NUMBER \
                        and (name, tok.string) not in OTHER_RATES:
                    found.append(f"{name}:{tok.start[0]}: {tok.string} = {after[1].string}")
                # 1 / 15 or 1 / 30 (of a second) in code, but `PROTO_DT = 1 / 15`
                if tok.type == tokenize.NUMBER and tok.string in ("1", "1.0") and after[0] == "/" \
                        and after[1].string in ("15", "15.0", "30", "30.0") \
                        and (name, toks[max(0, i - 2)].string) not in OTHER_RATES:
                    found.append(f"{name}:{tok.start[0]}: 1 / {after[1].string}")
    assert found == []
