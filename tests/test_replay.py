"""The replay kit (gamelab.sim.replay, replay_run, replay_score: README, Replaying the recordings): the labels in git, the
scores on small made-up dumps, the feed and the clock off a recording's own records, and a few frames run twice to the
byte. The recordings themselves (golden-clips, 6.6 GB) aren't in git: the tests that read them are skipped without
them."""

import gzip
import json
from pathlib import Path

import pytest

from gamelab.sim import replay, replay_score

CLIPS_HERE = all((c.root / "telemetry").is_dir() for c in replay.CLIPS.values())
needs_clips = pytest.mark.skipif(not CLIPS_HERE, reason="the recordings (golden-clips) aren't on this machine")


def labels(name: str) -> dict:
    return json.loads(replay.CLIPS[name].labels.read_text(encoding="utf-8"))


# --- the labels ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,onsets,births,found,touches,kills", [
    ("R1", 47, 41, 31, 21, 45), ("R2", 92, 63, 62, 26, 69), ("R3", 105, 77, 72, 24, 51)])
def test_the_labels_are_the_analyses(name, onsets, births, found, touches, kills):
    """Each recording's labels: small, versioned, every class there, the counts the analyses checked (R2+R3's 197 onsets,
    R1's 30 given out, the births in view and the watch's misses, the 71 touches, her full kills)."""
    path = replay.CLIPS[name].labels
    assert path.stat().st_size < 100_000
    lab = labels(name)
    assert lab["version"] == replay.LABELS_VERSION == "v1" and lab["clip"] == replay.CLIPS[name].folder
    cl = lab["classes"]
    assert set(cl) == {"onsets", "births", "touches", "under_her", "kills", "players"}
    assert len(cl["onsets"]["items"]) == onsets
    assert len(cl["births"]["items"]) == births and sum(b["found"] for b in cl["births"]["items"]) == found
    assert len(cl["touches"]["items"]) == touches and all(t["popup"] for t in cl["touches"]["items"])
    assert len(cl["kills"]["items"]) == kills
    assert {k["how"] for k in cl["kills"]["items"]} <= {"die1", "frames", "eye", "binding", "unsure"}
    if name == "R1":
        assert sum(o.get("given", False) for o in cl["onsets"]["items"]) == 30
    # every onset is on a birth, and a birth's first onset isn't a re-find
    ids = {b["id"] for b in cl["births"]["items"]}
    assert all(o["birth"] in ids for o in cl["onsets"]["items"])
    firsts = {}
    for o in cl["onsets"]["items"]:
        firsts.setdefault(o["birth"], o)
    assert not any(o["refind"] for o in firsts.values())


# --- the scores ------------------------------------------------------------------------------------------------------

def frame(i, t, **kw):
    return {"i": i, "t": t, "her": [0.0, 0.0, 10, "tag", "ground"], "v": [0, 0], "m": [], "tr": [], "ev": [], "pl": [],
            "sg": [], **kw}


def sg(point, x, a, onset, since):
    return [point, "H", x, -100.0, a, 0.99, 0.3, onset, since]


def test_onsets_scored_against_the_births():
    """An onset at a labelled one's point and frame is of record; one at a birth's point within REFIND_FRAMES after it
    a re-find; one at a point and time with no birth false."""
    frames = [frame(0, 0.0, sg=[sg(4, 10.0, 0.04, 1, 0.0)]), frame(1, 0.067, sg=[sg(4, 10.0, 0.12, 0, 0.0)]),
              frame(5, 0.333, sg=[sg(4, 10.0, 0.5, 1, 0.333)]),       # a re-find of the birth at f0
              frame(9, 0.6, sg=[sg(7, 50.0, 0.04, 1, 0.6)])]          # no birth at point 7: false
    births = [{"id": 0, "point": 4, "frame": 0, "found": True}, {"id": 1, "point": 5, "frame": 3, "found": False}]
    items = [{"point": 4, "frame": 0, "x": 10.0, "y": -100.0, "birth": 0, "refind": False},
             {"point": 4, "frame": 5, "x": 10.0, "y": -100.0, "birth": 0, "refind": True}]
    got = replay_score.score_onsets(frames, items, births)
    assert (got["of_record"], got["onsets"], got["false"], got["refinds"]) == (2, 3, 1, 1)
    assert got["false_at"] == [[9, 7]]
    b = replay_score.score_births(frames, births)
    assert b["seen_by_master"] == {"births": 1, "found": 1, "frames_late": 0, "alpha": 0.04}
    assert b["missed_by_master"]["found"] == 0


def test_touches_scored_by_time_how_and_number():
    t0 = 100.0
    frames = [frame(i, i / 15) for i in range(30)]
    frames[3]["ev"] = [["touch", None, 0.2, {"kind": "damage", "text": "118", "at": t0 + 2 / 15, "how": "popup"}]]
    frames[20]["ev"] = [["touch", None, 1.3, {"kind": "damage", "text": "", "at": t0 + 20 / 15, "how": "hp"}]]
    items = [{"frame": 2, "damage": 118}, {"frame": 10, "damage": 51}]
    got = replay_score.score_touches(frames, t0, items)
    assert (got["found"], got["by_popup"], got["number_right"], got["false"], got["frames_late"]) == (1, 1, 1, 1, 0)


def test_kills_scored_first_and_last_binding():
    """A gain first bound to a bystander, let go, then bound to the victim: its first binding wrong, its last right;
    an unsure label isn't scored."""
    t0 = 0.0
    frames = [frame(i, i / 15) for i in range(40)]
    frames[3]["tr"] = [[8, "dying", "H", -179.0, -1047.0, 15, None, "s", None, 0.2, None],
                       [5, "hidden", "H", -67.0, -1047.0, 15, None, "s", 0.1, None, None]]
    frames[3]["ev"] = [["victim", 8, 0.2, {"kind": "horny-mushroom", "x": -179, "platform": 15, "gain": 0.2}]]
    frames[21]["tr"] = frames[3]["tr"]
    frames[21]["ev"] = [["let_go", 8, 1.4, {"kind": "horny-mushroom", "gain": 0.2, "why": "seen alive"}],
                        ["victim", 5, 1.4, {"kind": "horny-mushroom", "x": -67, "platform": 15, "gain": 0.2}]]
    items = [{"gain": 0.2, "frame": 3, "victim": [-67.0, -1047.0], "how": "frames"},
             {"gain": 1.9, "frame": 28, "victim": None, "how": "unsure"}]
    got = replay_score.score_kills(frames, t0, items)
    assert got["strong"] == {"labelled": 1, "first_right": 0, "final_right": 1, "unbound": 0}
    assert got["unsure"] == 1 and got["let_go"] == 1


def test_under_her_and_players():
    frames = [frame(0, 0.0, m=[["H", 12.0, 5.0, 10, 3, None, 1, None, 0, 0, 56, 51, 0.9, "move/0"]],
                    pl=[[1, "stranger", 30.0, 0.0, 10], [2, "stranger", 400.0, -300.0, None]]),
              frame(15, 1.0, pl=[[3, "stranger", 500.0, 0.0, None]])]
    assert replay_score.score_under_her(frames, [{"frame": 0, "kind": "H"}, {"frame": 15, "kind": "G"}]) == \
        {"labelled": 2, "seen": 1}
    got = replay_score.score_players(frames, [[0, 1, 1], [15, 1, 1]])
    assert (got["more_than_dots"], got["by_her_dot"], got["by_her_held"]) == (1, 2, 1)


# --- the dumps and the cache's key ------------------------------------------------------------------------------------

def write_dump(path: Path, lines: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")


def test_check_says_where_two_dumps_part(tmp_path, capsys):
    a, b, c = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz", tmp_path / "c.jsonl.gz"
    lines = [{"head": True, "t0": 1.0}, frame(0, 0.0), frame(1, 0.067), {"end": True}]
    for p in (a, b):
        replay_run_dump(p, lines)
    replay_run_dump(c, lines[:2] + [frame(1, 0.067, sg=[sg(4, 10.0, 0.04, 1, 0.0)])] + lines[3:])
    assert replay.check(a, b) == 0
    assert replay.check(a, c) == 1
    assert "first differs at frame 1: sg" in capsys.readouterr().out


def replay_run_dump(path: Path, lines: list[dict]) -> None:
    """A dump as replay_run writes it (its _Dump: gzip without a time or a name, so the same lines are the same bytes)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("replay_run_for_test", replay.RUNNER)
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    d = run._Dump(path)
    for r in lines:
        d.write(r)
    d.close()


def test_a_dump_written_twice_is_the_same_to_the_byte(tmp_path):
    lines = [{"head": True, "t0": 1.0}, frame(0, 0.0), {"end": True}]
    replay_run_dump(tmp_path / "a.jsonl.gz", lines)
    replay_run_dump(tmp_path / "b.jsonl.gz", lines)
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (tmp_path / "b.jsonl.gz").read_bytes()
    assert not list(tmp_path.glob("*.part"))


def test_lab_files_the_key_doesnt_cover_are_named(tmp_path):
    assert replay.covered({"characters/magician/poses_1366x768.npz": "x", "mapart/10002073.npz": "y",
                           "stat_window.json": "z"}) == []
    assert replay.covered({"characters/magician/touches.jsonl": "x"}) == ["characters/magician/touches.jsonl"]
    (tmp_path / "characters" / "magician").mkdir(parents=True)
    (tmp_path / "characters" / "magician" / "calibration_1366x768.json").write_text("{}")
    (tmp_path / "characters" / "magician" / "hits.jsonl").write_text("")
    got = replay.lab_inputs(tmp_path)
    assert got["character.json"] is None and "characters/magician/calibration_1366x768.json" in got
    assert "characters/magician/hits.jsonl" not in got


def test_the_runner_imports_none_of_this_package_at_the_top():
    """replay_run.py runs another tree's gamelab: nothing of gamelab is imported before its --root is first on the
    path (every gamelab import inside main, after it)."""
    import ast
    tree = ast.parse(replay.RUNNER.read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top for a in n.names] + [n.module or "" for n in top if isinstance(n, ast.ImportFrom)]
    assert not any(name.startswith(("gamelab", "gamedata")) for name in names)


# --- on the recordings -----------------------------------------------------------------------------------------------

@needs_clips
def test_the_clock_and_her_throws_from_the_records():
    """R2's frames on the service's clock (106 of its 107 frame times named in the telemetry within 2 ms of a frame,
    the offset 1793544509.301 s on all three) and her 139 throws, each with its side by the fight's stance."""
    f = replay.feed(replay.CLIPS["R2"])
    assert abs(f["clock"]["offset"] - 1793544509.301) <= 0.002
    assert f["clock"]["matched"] >= 105
    assert f["how"]["attacks"] == 139 and f["how"]["side by stance"] == 139 and f["how"]["no side"] == 0
    assert all(th["told"] <= th["t"] <= th["told"] + replay.TURN_S + 1e-9 for th in f["throws"])
    assert f["asks"] and all(set(a) == {"loot", "loot_wide", "mesos_only"} for a in f["asks"].values())


@needs_clips
@pytest.mark.slow
def test_a_few_frames_twice_to_the_byte(tmp_path):
    """R2's frames 118-130 (a newborn at point 5 from f121) through this tree's frame step twice, unpaced: the same
    dump to the byte, with the onset at point 5 on f121."""
    c = replay.CLIPS["R2"]
    for side in ("a", "b"):
        p = replay.run_one(c, replay.REPO, tmp_path / side, 118, 13, paced=False)
        assert p.wait(timeout=300) == 0
    a, b = (tmp_path / s / f"{c.folder}.jsonl.gz" for s in "ab")
    assert a.read_bytes() == b.read_bytes()
    head, frames, end = replay.load(a)
    assert [(o["point"], o["frame"]) for o in replay_score.onsets_of(frames)] == [(5, 121)]
    assert end["frames"] == 13 and head["character"] == "magician"
