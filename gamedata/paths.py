"""Where gamedata keeps the copy of the game's data and what it extracts (git-ignored)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COPY = DATA / "copy"     # a copy of the game's appdata/Data folder; everything is read from here
OUT = DATA / "out"       # JSON exports
TABLES = OUT / "tables"  # the tables `python -m gamedata tables` builds
