"""Where the tests' data too big for git lives (README: Tests): gamelab/lab/testdata, git-ignored like the rest of the
lab, one folder a set. Never a scratch folder: the cleanup of D:/MapleHelper-scratch on 2026-10-28 deleted the labelled
frames of Tree Dungeon and West Rocky Mountain II that tests read there, and those tests skipped without a word. Nor
gamelab/lab/recent, which the rolling recorder prunes to its newest 5 GB (rolling.RECENT_KEEP_BYTES), nor
lab/lie_detector (lie_recorder.prune): the clips the 1366x768 HP bars were labelled on went that way.

The sets (each one's labels in tests/fixtures, in git):
- `tree_dungeon_native`: 24 lossless frames of Tree Dungeon, Forest Up North II (10002073) at the game's own 1366x768,
  saved in game on 2026-10-28 19:17-19:19 on a crowded channel, 3 on each of platforms 13, 16, 18, 20, 21, 23, 25 and 30
  (td_p<platform>_<i>.png, the td_p24 ones on 25: her_platform; tree_dungeon_native_monsters.json: every monster, player
  and pet; no drop on any).
- `tree_dungeon_drops`: 7 frames of the hunts there of 2026-10-24 with drops lying (hp0_*.png, as saved in lab/shots;
  tree_dungeon_drops.json; three of them have labelled HP bars too, hp_bars/labelled.json's "shots/").
(The first map's `hp_bars_1920x1080` and West Rocky Mountain II's sets, `west_rocky_native` and `west_rocky`, are no
test's since 2026-10-31, when those maps' tests went.)
Lost, to be captured again (`lost`): `tree_dungeon_clips`, the rolling recorder's clips of her Tree Dungeon hunts of
2026-10-24 and 25 (609 frames, JPEG) with labelled HP bars at 1366x768, pruned from lab/recent.
What a new capture would be (then labelled offline, as tree_dungeon_native's were, into a new labels file):
- HP bars at 1366x768 (`tree_dungeon_clips`): frames of a hunt of hers on Tree Dungeon (a monster's bar shows only once
  she has hit it): `save-recent` during one, its folder copied here, the bars labelled as hp_bars/labelled.json's.

Where: GAMELAB_TESTDATA, a folder (or several, separated by os.pathsep), if set; else this checkout's
gamelab/lab/testdata, then, in a git worktree, the main checkout's (a worktree's lab is a copy without it). A test whose
set isn't there FAILS, saying which and where it looked; on a computer without the sets GAMELAB_NO_TESTDATA=1 skips
those tests instead, saying the same."""

import json
import os
from pathlib import Path

import pytest

ENV = "GAMELAB_TESTDATA"
OFF = "GAMELAB_NO_TESTDATA"
REPO = Path(__file__).resolve().parents[1]
LAB_TESTDATA = Path("gamelab") / "lab" / "testdata"

FIXTURES = Path(__file__).parent / "fixtures"

TREE_DUNGEON = "tree_dungeon_native"
TREE_DUNGEON_DROPS = "tree_dungeon_drops"
LABELS = {TREE_DUNGEON: "tree_dungeon_native_monsters.json", TREE_DUNGEON_DROPS: "tree_dungeon_drops.json"}
WHAT = {TREE_DUNGEON: "Tree Dungeon's labelled frames at 1366x768 of 2026-10-28",
        TREE_DUNGEON_DROPS: "Tree Dungeon's frames with drops of 2026-10-24 (lab/shots/hp0_*.png)"}


def main_checkout(repo: Path = REPO) -> Path | None:
    """The main checkout of the git worktree `repo` (its .git a file "gitdir: <main>/.git/worktrees/<name>"), or None
    where `repo` is the main checkout itself or not a worktree."""
    git = repo / ".git"
    if not git.is_file():
        return None
    try:
        text = git.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text[len("gitdir:"):].strip())
    if not gitdir.is_absolute():
        gitdir = (repo / gitdir).resolve()
    common = gitdir / "commondir"  # (git's own pointer from a worktree's folder to the shared .git)
    try:
        shared = (gitdir / common.read_text(encoding="utf-8").strip()).resolve() if common.is_file() else None
    except OSError:
        return None
    shared = shared or gitdir.parents[1]
    return shared.parent


def roots() -> list[Path]:
    """The folders looked in for a set, in order (the module's doc)."""
    if os.environ.get(ENV):
        return [Path(p) for p in os.environ[ENV].split(os.pathsep) if p]
    out = [REPO / LAB_TESTDATA]
    main = main_checkout()
    if main is not None and main.resolve() != REPO:
        out.append(main / LAB_TESTDATA)
    return out


def find(name: str, files=()) -> Path | None:
    """The set `name`'s folder: the first of roots() that has it with every one of `files` in it; else None."""
    for root in roots():
        path = root / name
        if path.is_dir() and all((path / f).is_file() for f in files):
            return path
    return None


def folder(name: str, files=(), what: str = "") -> Path:
    """The set `name`'s folder, with every one of `files` (names in it), for a test that reads them: where it isn't,
    the test fails, saying what it needs and where it looked (or, with GAMELAB_NO_TESTDATA=1, skips saying so)."""
    files = list(files)
    path = find(name, files)
    if path is not None:
        return path
    looked = []
    for root in roots():
        here = root / name
        lacking = [f for f in files if not (here / f).is_file()] if here.is_dir() else None
        looked.append(f"{here} (not there)" if lacking is None else
                      f"{here} (without {', '.join(lacking[:3])}{' ...' if len(lacking) > 3 else ''})")
    missing(f"{what or name}: the test data set {name!r} isn't here: looked in {'; '.join(looked)}")


def labelled(name: str) -> tuple[dict, dict]:
    """A set's labels by frame (its fixture's "frames", LABELS) and each frame's path, every frame they name in its
    folder (`folder`: the test fails without them)."""
    labels = json.loads((FIXTURES / LABELS[name]).read_text(encoding="utf-8"))["frames"]
    root = folder(name, labels, WHAT[name])
    return labels, {frame: root / frame for frame in labels}


def missing(why: str):
    """A test's data isn't here: it fails, saying why, unless GAMELAB_NO_TESTDATA=1, where it skips saying the same."""
    if os.environ.get(OFF, "") not in ("", "0"):
        pytest.skip(f"no test data ({OFF}={os.environ[OFF]}): {why}")
    pytest.fail(f"TEST DATA MISSING: {why}. Put it there, set {ENV} to the folder that holds it, or set {OFF}=1 on a "
                f"computer without it to skip the tests that need it (README: Tests; tests/testdata.py)", pytrace=False)


SCRATCH_CLEANUP = "deleted from D:/MapleHelper-scratch on 2026-10-28"


def lost(name: str, files=(), what: str = "", labels: str = "", gone: str = SCRATCH_CLEANUP) -> Path:
    """A set whose frames were lost (`gone`: how; most were deleted from D:/MapleHelper-scratch on 2026-10-28): its
    folder if it's back (the same frames, named as their labels name them), else the test skips, saying plainly it
    needs a new capture (and new labels: `labels` were made on the lost frames)."""
    files = list(files)
    path = find(name, files)
    if path is not None:
        return path
    pytest.skip(f"NEEDS A NEW CAPTURE: {what or name}: its frames were {gone} and exist nowhere else; capture the map "
                f"again into gamelab/lab/testdata/{name} and label the new frames as {labels or 'its labels'} were "
                f"labelled (the old labels fit only the lost frames)")
