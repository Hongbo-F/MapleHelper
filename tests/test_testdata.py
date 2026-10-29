"""tests/testdata.py: the test data sets found where the setting says, a test without its set failing with why (or
skipping with GAMELAB_NO_TESTDATA=1), and a lost set's tests saying they need a new capture."""

import ast
import re
from pathlib import Path

import pytest

import testdata


def test_a_worktrees_main_checkout_is_read_off_its_git_file(tmp_path):
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (main / ".git" / "worktrees" / "wt" / "commondir").write_text("../..\n", encoding="utf-8")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n", encoding="utf-8")
    assert testdata.main_checkout(wt) == main.resolve()
    assert testdata.main_checkout(main) is None  # (its .git a folder: the main checkout itself)
    (wt / ".git").write_text("not a worktree\n", encoding="utf-8")
    assert testdata.main_checkout(wt) is None


def test_the_sets_are_looked_for_where_the_setting_says_then_by_default_here_and_in_the_main_checkout(tmp_path,
                                                                                                       monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    (b / "frames").mkdir(parents=True)
    (b / "frames" / "one.png").write_bytes(b"")
    monkeypatch.setenv(testdata.ENV, f"{a}{testdata.os.pathsep}{b}")
    assert testdata.roots() == [a, b]
    assert testdata.find("frames", ["one.png"]) == b / "frames"
    assert testdata.find("frames", ["one.png", "two.png"]) is None
    monkeypatch.delenv(testdata.ENV)
    monkeypatch.setattr(testdata, "main_checkout", lambda repo=None: tmp_path / "main")
    assert testdata.roots() == [testdata.REPO / testdata.LAB_TESTDATA, tmp_path / "main" / testdata.LAB_TESTDATA]


def test_a_test_without_its_set_fails_saying_where_it_looked_unless_told_to_skip(tmp_path, monkeypatch):
    monkeypatch.setenv(testdata.ENV, str(tmp_path))
    monkeypatch.delenv(testdata.OFF, raising=False)
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "one.png").write_bytes(b"")
    assert testdata.folder("frames", ["one.png"]) == tmp_path / "frames"
    with pytest.raises(pytest.fail.Exception) as no_set:
        testdata.folder("elsewhere", what="some frames")
    with pytest.raises(pytest.fail.Exception) as short:
        testdata.folder("frames", ["one.png", "two.png"])
    assert "TEST DATA MISSING: some frames" in str(no_set.value) and "(not there)" in str(no_set.value)
    assert testdata.ENV in str(no_set.value) and testdata.OFF in str(no_set.value)
    assert "(without two.png)" in str(short.value)
    monkeypatch.setenv(testdata.OFF, "1")
    with pytest.raises(pytest.skip.Exception, match="no test data .*some frames"):
        testdata.folder("elsewhere", what="some frames")
    monkeypatch.setenv(testdata.OFF, "0")  # (0: as unset)
    with pytest.raises(pytest.fail.Exception):
        testdata.folder("elsewhere")


def test_a_lost_sets_tests_skip_saying_they_need_a_new_capture_unless_it_is_back(tmp_path, monkeypatch):
    monkeypatch.setenv(testdata.ENV, str(tmp_path))
    monkeypatch.delenv(testdata.OFF, raising=False)
    with pytest.raises(pytest.skip.Exception, match="NEEDS A NEW CAPTURE: the old frames: its frames were deleted "
                                                    "from D:/MapleHelper-scratch on 2026-10-28 and"):
        testdata.lost("old", ["a.png"], "the old frames", "tests/fixtures/old.json")
    with pytest.raises(pytest.skip.Exception, match="NEEDS A NEW CAPTURE: clips: its frames were pruned and"):
        testdata.lost("old", ["a.png"], "clips", gone="pruned")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "a.png").write_bytes(b"")
    assert testdata.lost("old", ["a.png"]) == tmp_path / "old"


SCRATCH = re.compile(r"maplehelper-scratch(?=[\\/]|$)", re.I)  # a place in a scratch folder (not words about one)
PRUNED = ("RECENT", "LIE_RECORDINGS", "ALARM_CLIPS")  # the lab's folders the recorders prune to their newest GBs
PRUNED_DIRS = ("recent", "lie_detector", "alarm_clips")  # ... by name under paths.LAB (`lab-usage` says how much)


def _places(path: Path) -> list[tuple[int, str]]:
    """Where the Python file `path` names a scratch folder (a string other than a docstring: any slash, any case, in an
    f-string too; comments and docstrings may) or a folder of the lab's that the recorders prune (paths.RECENT,
    LIE_RECORDINGS, ALARM_CLIPS, or LAB / "recent/..."): (line, what)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = {id(node.body[0].value) for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body
            and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs
                and SCRATCH.search(node.value)):
            out.append((node.lineno, node.value))
        elif (isinstance(node, ast.Attribute) and node.attr in PRUNED and isinstance(node.value, ast.Name)
              and node.value.id == "paths"):
            out.append((node.lineno, f"paths.{node.attr}"))
        elif (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant)
              and isinstance(node.right.value, str) and re.split(r"[\\/]", node.right.value)[0] in PRUNED_DIRS
              and (isinstance(node.left, ast.Attribute) and node.left.attr == "LAB"
                   or isinstance(node.left, ast.Name) and node.left.id == "LAB")):
            out.append((node.lineno, f"LAB / {node.right.value!r}"))
    return out


def test_the_check_for_frames_read_from_a_scratch_or_pruned_folder_sees_them_however_written(tmp_path):
    sample = tmp_path / "test_sample.py"
    sample.write_text("\n".join([
        '"""Frames once in D:/MapleHelper-scratch/map3 (a docstring may say so)."""',
        "from pathlib import Path",
        'A = Path(r"D:\\MapleHelper-scratch\\map3\\native")',
        'B = "d:/maplehelper-scratch/newmap/frames/x.png"',
        'C = Path("D:/MapleHelper-scratch") / "newmap"',
        'D = [f"D:/MapleHelper-scratch/{n}.png" for n in "ab"]',
        'E = paths.RECENT / "20261024-235133-heading-wrong"',
        'F = paths.LAB / "recent/20261024-235133-heading-wrong/00055.jpg"',
        'G = LAB / "lie_detector"',
        "def f():",
        '    """Read from D:/MapleHelper-scratch/x once."""  # (and a comment: D:/MapleHelper-scratch/y)',
        '    monkeypatch.setattr(paths, "RECENT", tmp_path)',
        '    return "deleted from D:/MapleHelper-scratch on 2026-10-28", paths.SHOTS / "hp0_135315.png"',
    ]), encoding="utf-8")
    assert "\\MapleHelper-scratch\\map3" in sample.read_text(encoding="utf-8")  # (A as a test file would have it)
    assert sorted(line for line, _ in _places(sample)) == [3, 4, 5, 6, 7, 8, 9]


def test_no_test_reads_frames_from_a_scratch_folder_or_one_the_recorders_prune():
    # (the cleanup of 2026-10-28 took the frames tests read from D:/MapleHelper-scratch, and the rolling recorder's
    # pruning of lab/recent the clips the HP bars were labelled on; their tests skipped silently or read what was left)
    places = {p.name: _places(p) for p in Path(__file__).parent.glob("*.py") if p.name != Path(__file__).name}
    assert {name: found for name, found in places.items() if found} == {}
