"""tests/testdata.py: the test data sets found where the setting says, a test without its set failing with why (or
skipping with GAMELAB_NO_TESTDATA=1), and a lost set's tests saying they need a new capture."""

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
    with pytest.raises(pytest.skip.Exception, match="NEEDS A NEW CAPTURE: the old frames"):
        testdata.lost("old", ["a.png"], "the old frames", "tests/fixtures/old.json")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "a.png").write_bytes(b"")
    assert testdata.lost("old", ["a.png"]) == tmp_path / "old"


def test_no_test_reads_frames_from_a_scratch_folder():
    # (the cleanup of 2026-10-28 took the frames tests read from D:/MapleHelper-scratch, and they skipped silently)
    tests = Path(__file__).parent
    readers = [p.name for p in tests.glob("*.py")
               if p.name != Path(__file__).name and 'Path("D:/MapleHelper-scratch' in p.read_text(encoding="utf-8")]
    assert readers == []
