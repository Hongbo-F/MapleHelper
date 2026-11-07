"""tools/review_pages, the review pages' builders: no clips, GPU or game data here. No path tied to one PC; clips.json;
a clip's range from a made-up round file and frames; what --list says and that a page is never written over; the
approved reference before the unchecked one."""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "review_pages"
sys.path[:0] = [str(TOOLS), str(TOOLS / "ld")]

import build  # noqa: E402
import clip_range  # noqa: E402
import common  # noqa: E402
import lib  # noqa: E402
import reference  # noqa: E402
from gamelab.perception import lie_detector  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "lie_detector" / "references"
YELLOW = cv2.cvtColor(np.full((1, 1, 3), (28, 200, 240), np.uint8), cv2.COLOR_HSV2BGR)[0, 0]  # the banner's
TIED = re.compile(r"(?<![a-z])[a-z]:[\\/]|appdata|maplehelper.scratch", re.I)  # a drive's path, or a place on this PC


def test_nothing_in_the_builders_is_tied_to_one_pc():
    files = [p for p in TOOLS.rglob("*") if p.suffix in (".py", ".md", ".json") and "__pycache__" not in p.parts]
    found = [(p.name, m.group()) for p in files for m in [TIED.search(p.read_text(encoding="utf-8"))] if m]
    assert not found


def test_the_web_ui_and_the_builders_look_for_the_pages_in_one_place(tmp_path, monkeypatch):
    from gamelab.ui import review
    monkeypatch.delenv("GAMELAB_REVIEW_ROOT", raising=False)
    monkeypatch.delenv("GAMELAB_REVIEW_WORK", raising=False)
    assert common.root() == review.ROOT
    assert common.work() == common.paths.LAB / "review_work"
    monkeypatch.setenv("GAMELAB_REVIEW_ROOT", str(tmp_path / "a"))
    monkeypatch.setenv("GAMELAB_REVIEW_WORK", str(tmp_path / "b"))
    assert common.root() == tmp_path / "a" and common.work() == tmp_path / "b"


def test_clips_json_holds_each_older_clips_range_and_each_has_an_approved_reference():
    assert len(lib.CLIPS) == 11
    for key, c in lib.CLIPS.items():
        assert re.fullmatch(r"\d{8}-\d{6}", key)
        assert set(c) == {"fps", "first", "last", "start", "end", "start_j", "end_j", "round_s"}
        assert c["first"] < c["start"] < c["end"] <= c["last"]
        assert c["start"] == c["first"] + round(clip_range.START_S * c["fps"]) == c["first"] + c["start_j"]
        assert c["end"] == c["first"] + c["end_j"]
        assert (lib.APPROVED / f"{key}.json").exists()


def test_the_approved_references_have_the_centres_the_user_approved():
    for key in lib.CLIPS:
        fixture = json.loads((FIXTURES / f"{key}.json").read_text(encoding="utf-8"))
        approved = json.loads((lib.APPROVED / f"{key}.json").read_text(encoding="utf-8"))
        frames = {f["i"]: f for f in approved["frames"].values()}
        assert fixture["start"] == lib.CLIPS[key]["start"] and fixture["end"] == lib.CLIPS[key]["end"]
        for i, (x, y) in fixture["centre"].items():
            assert np.hypot(frames[int(i)]["x"] - x, frames[int(i)]["y"] - y) <= 0.08  # (the fixture's 0.1 px rounding)


@pytest.fixture
def board_look(monkeypatch):
    """The live look for the board, on the made-up frames (too small for the real one): a frame's first pixel says."""
    class Look:
        def over_play(self, frame):
            return "mini-game" if frame[0, 0].any() else None
    monkeypatch.setattr(lie_detector, "LieDetectorSpotter", Look)


def made_up_clip(tmp_path, banner_from=120, flash=None, round_stamp="20261104-180324", answers_from=11, seq0=1000,
                 board_from=9):
    """A clip of 150 frames, 15 a second, recorded from 18:03:12 to 18:03:43, its frames small and only those looked at
    written; the board up from `board_from`; yellow on the board from `banner_from`, and on frame `flash` alone."""
    clip, rounds = tmp_path / "20261104-180343-lie-detector-mini-game-notice", tmp_path / "rounds"
    clip.mkdir()
    rounds.mkdir()
    (clip / "summary.json").write_text(json.dumps({"from": "2026-11-04 18:03:12.939", "to": "2026-11-04 18:03:43.152"}))
    rows = [{"i": i, "file": f"{i:05d}.png", "t": i / 15, "seq": seq0 + i, "run": 1} for i in range(150)]
    (clip / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows + [{"gap": "covered", "t": 10.0}]))
    plain, glowing = np.zeros((230, 400, 3), np.uint8), np.zeros((230, 400, 3), np.uint8)
    glowing[180:230, 308:400] = YELLOW
    up = plain.copy()
    up[0, 0] = 255
    for i in range(board_from - 1, answers_from):
        cv2.imwrite(str(clip / f"{i:05d}.png"), up if i >= board_from else plain)
    for i in range(70, 150):
        cv2.imwrite(str(clip / f"{i:05d}.png"), glowing if i >= banner_from or i == flash else plain)
    lines = [{"kind": "the target", "fps": 15.0}]
    lines += [{"seq": seq0 + i, "board": [1, 2]} for i in range(answers_from, 130)] + [{"end": "board gone"}]
    (rounds / f"{round_stamp}.jsonl").write_text("\n".join(json.dumps(r) for r in lines))
    return clip, rounds


def test_a_clips_range_comes_from_its_round_file_and_the_banner(tmp_path, board_look):
    clip, rounds = made_up_clip(tmp_path, banner_from=120, flash=100)  # (a frame's flash is not the banner)
    (rounds / "20261104-170000.jsonl").write_text("not this clip's")
    assert clip_range.clip_range(clip, rounds) == {
        "clip": clip.name, "round": "20261104-180324", "fps": 15, "first": 9, "start": 78, "end": 120, "last": 141}


@pytest.mark.parametrize("board_from", (8, 9, 10, 11))
def test_a_clips_first_frame_is_the_boards_however_late_the_first_answer(tmp_path, board_look, board_from):
    """The first answer is at frame 11, 0 to 3 frames after the board on the clips; START counts from the board."""
    clip, rounds = made_up_clip(tmp_path, board_from=board_from)
    found = clip_range.clip_range(clip, rounds)
    assert found["first"] == board_from and found["start"] == board_from + 69


def test_the_banner_is_found_from_its_fade_in(tmp_path, board_look):
    """20261104-185153's first two frames of it were 560 and 490 px, the third 3425: the sets stop before all three."""
    clip, rounds = made_up_clip(tmp_path, banner_from=122)
    for i, side in ((120, 24), (121, 22)):  # (576 and 484 px)
        frame = np.zeros((230, 400, 3), np.uint8)
        frame[180:180 + side, 308:308 + side] = YELLOW
        cv2.imwrite(str(clip / f"{i:05d}.png"), frame)
    assert clip_range.clip_range(clip, rounds)["end"] == 120


def test_a_frame_whose_file_is_gone_is_passed_over(tmp_path, board_look):
    clip, rounds = made_up_clip(tmp_path, banner_from=120)
    (clip / "00100.png").unlink()
    assert clip_range.clip_range(clip, rounds)["end"] == 120


def test_no_round_file_or_no_banner_says_so_and_gives_no_range(tmp_path, capsys, board_look):
    clip, rounds = made_up_clip(tmp_path, banner_from=999, round_stamp="20261104-170000")  # (from before the clip)
    assert clip_range.clip_range(clip, rounds) is None
    assert "no live round file" in capsys.readouterr().out
    (rounds / "20261104-170000.jsonl").rename(rounds / "20261104-180324.jsonl")
    assert clip_range.clip_range(clip, rounds) is None
    assert "no SUCCESS banner" in capsys.readouterr().out


@pytest.fixture
def clips_and_pages(tmp_path, monkeypatch):
    """Two mini-game clips and a Violetta one; the first clip has a page in ld-outline-pf3 only."""
    clips = tmp_path / "clips"
    for name in ("20261104-180343-lie-detector-mini-game-notice", "20261105-080220-lie-detector-mini-game-notice",
                 "20261104-004840-lie-detector-countdown", "20261104-015256-lie-detector-quiz-warning"):
        (clips / name).mkdir(parents=True)
    monkeypatch.setattr(build, "CLIPS", clips)
    monkeypatch.setenv("GAMELAB_REVIEW_ROOT", str(tmp_path))  # (put back after: main sets it for its workers)
    root = tmp_path / "pages"
    page = root / "ld-outline-pf3" / "20261104-180343-15fps"
    (page / "frames").mkdir(parents=True)
    (page / "README.md").write_text("as built")
    built = []
    monkeypatch.setattr(build, "build_ld", lambda todo: built.append(todo))
    monkeypatch.setattr(lib, "clip", lambda key: {"fps": 15})
    monkeypatch.setattr(lib, "APPROVED", tmp_path / "approved")
    return root, page, built


def test_list_says_which_clips_have_no_page_in_each_group_and_builds_nothing(clips_and_pages, capsys):
    root, page, built = clips_and_pages
    build.main(["ld", "--list", "--root", str(root)])
    out = capsys.readouterr().out
    assert "ld-reference: 2 with no page: 20261104-180343 20261105-080220" in out
    assert "ld-outline-pf3: 1 with no page: 20261105-080220" in out
    assert not built and page.exists()
    build.main(["violetta", "--list", "--root", str(root)])
    assert "violetta: 1 with no page: 20261104-004840\n" in capsys.readouterr().out


def test_a_page_is_never_written_over_unless_forced_for_that_clip(clips_and_pages, capsys):
    root, page, built = clips_and_pages
    build.main(["ld", "--missing", "--root", str(root)])
    assert built.pop() == {"20261104-180343": ["ld-reference", "ld-speed-compare"],
                           "20261105-080220": ["ld-reference", "ld-outline-pf3", "ld-speed-compare"]}
    assert "20261104-180343: ld-outline-pf3 skipped: its page is there" in capsys.readouterr().out
    build.main(["ld", "20261104-180343-lie-detector-mini-game-notice", "--root", str(root)])
    assert built.pop() == {"20261104-180343": ["ld-reference", "ld-speed-compare"]}
    assert (page / "README.md").read_text() == "as built"
    build.main(["ld", "--force", "20261104-180343", "--root", str(root)])
    assert built.pop() == {"20261104-180343": ["ld-reference", "ld-outline-pf3", "ld-speed-compare"]}
    assert not page.exists()  # (the old one gone; the build, left out here, makes the new one)
    with pytest.raises(SystemExit):
        build.main(["ld", "--missing", "--force", "--root", str(root)])  # (for the clips named only)


def test_an_approved_clips_reference_is_never_rebuilt(clips_and_pages, tmp_path):
    root, page, built = clips_and_pages
    (tmp_path / "approved").mkdir()
    (tmp_path / "approved" / "20261105-080220.json").write_text("{}")
    build.main(["ld", "--force", "20261105-080220", "--root", str(root)])
    assert built.pop() == {"20261105-080220": ["ld-outline-pf3", "ld-speed-compare"]}
    with pytest.raises(ValueError, match="never rebuilt"):
        reference.build("20261105-080220", {})


def test_the_approved_reference_is_found_before_the_unchecked_one(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "APPROVED", tmp_path / "approved")
    monkeypatch.setattr(lib, "UNCHECKED", tmp_path / "unchecked")
    (tmp_path / "approved").mkdir()
    (tmp_path / "unchecked").mkdir()
    (tmp_path / "unchecked" / "k.json").write_text("{}")
    assert lib.reference_of("k") == (tmp_path / "unchecked" / "k.json", False)
    (tmp_path / "approved" / "k.json").write_text("{}")
    assert lib.reference_of("k") == (tmp_path / "approved" / "k.json", True)
