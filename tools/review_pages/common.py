"""Where the review pages' builders find things, for both groups' scripts (ld/, violetta/): nothing here or in them is
tied to one PC. Importing this puts the repo on sys.path, from this file's own place, so `gamelab` imports from any
folder (a worker, spawned on Windows, imports it again the same way)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gamelab import paths  # noqa: E402

PICTURES = paths.REPO / "gamedata/data/out/pictures/UI/AntiMacro.img/lieDetector"  # the game's own, for the sprites
WEIGHTS = paths.ASSETS / "violetta.pt"  # the Violetta detector, in git


def root() -> Path:
    """Where the pages go: GAMELAB_REVIEW_ROOT (build.py's --root sets it, so its workers and steps see it too), else
    the folder the web UI serves. Read when asked: a script takes it at its import, after build.py has set it."""
    from gamelab.ui import review  # (not at this file's import: only the page writers ask, not every worker)
    return Path(os.environ.get("GAMELAB_REVIEW_ROOT") or review.ROOT)


def work() -> Path:
    """Work data (frame caches, detections, solved games, unchecked references): GAMELAB_REVIEW_WORK, else under the
    lab, which git ignores."""
    return Path(os.environ.get("GAMELAB_REVIEW_WORK") or paths.LAB / "review_work")
