"""The Lie Detector mini-game's target followed live on the perception service's frames (control.lie_tracking): only
while the alarm is up and the board shows, the newest frame sent when the tracker is ready for it, its desktop answers
followed by trail-based cursor movement, a round's file and the same answers in the service's telemetry; the tracker's
process, its loop and the real one, on a drawn round (tests.test_lie_target)."""
import io
import json
import pickle
import threading
import time
from types import SimpleNamespace

import numpy as np

from gamelab import telemetry
from gamelab.control import lie_tracking as lt
from gamelab.perception.lie_target import BOARD_AT_NATIVE, RIM, TargetTracker
from tests.test_lie_target import FPS, round_frames


def screen(board_bgr):
    """A 1366x768 RGB frame of the game with the board where it shows."""
    full = np.zeros((768, 1366, 3), np.uint8)
    x, y, w, h = BOARD_AT_NATIVE
    full[y:y + h, x:x + w] = board_bgr[..., ::-1]
    return full


class Spotter:
    def __init__(self):
        self.board = True

    def over_play(self, full):
        return "mini-game" if self.board else None


class FakeChild:
    """The tracker's process as the tests see it: what it was sent; its replies none until it's stopped (the real one's
    ending stream means it's gone)."""

    def __init__(self):
        self.sent = []
        self.done = threading.Event()

    def send(self, msg):
        self.sent.append(msg)

    def replies(self):
        self.done.wait(30)
        yield from ()

    def stop(self):
        self.sent.append(("stopped",))
        self.done.set()


def tracking(tmp_path, alarm_up=True):
    alarm = tmp_path / "ALARM.json"
    if alarm_up:
        alarm.write_text("{}", encoding="utf-8")
    children, moves = [], []

    def child():
        children.append(FakeChild())
        return children[-1]

    def move(x, y, wait):
        moves.append((x, y))
        return True

    t = lt.LieTracking(folder=tmp_path / "out", alarm=alarm, spotter=Spotter(), child=child,
                       origin=lambda: (100, 50), mover=move)
    t.moves = moves
    return t, children, alarm


def item(n, t, full):
    return (t, full, n, 1, 1700000000.0 + t)


def test_no_alarm_nothing_kept_and_no_tracker_started(tmp_path):
    t, children, _ = tracking(tmp_path, alarm_up=False)
    t.saw(SimpleNamespace(seq=1), SimpleNamespace(full=np.zeros((768, 1366, 3), np.uint8), t=1.0, visible=True), 1)
    assert t._newest is None
    t.step(1.0, None)
    assert children == [] and t.round is None


def test_with_the_alarm_up_a_round_begins_at_the_board_and_frames_go_only_when_the_tracker_is_ready(tmp_path):
    t, children, alarm = tracking(tmp_path)
    full = screen(round_frames(n=1)[0][1])
    t.spotter.board = False
    t.step(0.0, item(0, 0.0, full))  # the alarm up, the board not yet: the tracker's process started, no round
    assert len(children) == 1 and t.round is None and children[0].sent == []
    t.spotter.board = True
    t.step(0.3, item(1, 0.3, full))  # the board seen: the round begins; the tracker not ready yet: skipped
    child = children[0]
    assert child.sent[0][0] == "begin" and t.round["skipped"] == 1 and t.round["sent"] == 0
    t.ready.set()
    t.step(0.6, item(2, 0.6, full))
    kind, tt, seq, run, clock, board, corner = child.sent[-1]
    assert (kind, tt, seq, run, corner) == ("frame", 0.6, 2, 1, (100, 50)) and board.shape == (500, 750, 3)
    assert np.array_equal(board, full[180:680, 308:1058]) and not t.ready.is_set()
    # the board gone for BOARD_GONE_S: the round is over
    t.spotter.board = False
    t.ready.set()
    t.step(0.9, item(3, 0.9, full))
    t.step(0.9 + lt.BOARD_GONE_S + 0.1, item(4, 0.9 + lt.BOARD_GONE_S + 0.1, full))
    assert child.sent[-1][0] == "end" and child.sent[-1][1] == "board gone" and t.round is None


def test_after_a_look_without_the_board_no_frame_is_sent_and_the_cursor_stops_until_the_board_shows_again(tmp_path):
    """The user, 2026-11-03: "Fix lie detector code": the 00:01 round's answers went on 2.9 s into the Truth Booth
    (BOARD_GONE_S), the cursor following them; looked for once a second ("1 in 15 frames, every literal second")."""
    assert lt.BOARD_ROUND_S == 1.0
    t, children, _ = tracking(tmp_path)
    full = screen(round_frames(n=1)[0][1])
    t.step(0.0, item(0, 0.0, full))  # the board: the round begins (the tracker's process just started: not sent)
    child = children[0]

    def frame(n, at, board):
        t.spotter.board = board
        t.ready.set()
        t.step(at, item(n, at, full))
        return [m[2] for m in child.sent if m[0] == "frame"]

    assert frame(1, 0.5, False) == [1] and t._cursor_round == t.round["name"]  # (not looked at: as the last look)
    t._cursor_target = (t.round["name"], 1, 2, 0)  # an answer not yet taken by the cursor
    assert frame(2, 1.0, False) == [1]  # looked at, the board gone: not sent, the cursor off
    assert t.round["off_board"] == 1 and t._cursor_round is None and t._cursor_target is None
    t._queue_cursor((5, 6), seq=1)  # the tracker's answer to the frame sent before: not followed
    assert t._cursor_target is None
    assert frame(3, 1.5, True) == [1]  # not looked at: still as the last look
    assert frame(4, 2.0, True) == [1, 4] and t._cursor_round == t.round["name"]  # seen again: the same round goes on
    assert frame(5, 2.5, False) == [1, 4, 5]  # not looked at: as the last look, the board's
    assert frame(6, 3.0, False) == [1, 4, 5]
    t.step(2.0 + lt.BOARD_GONE_S + 0.1, item(7, 2.0 + lt.BOARD_GONE_S + 0.1, full))
    assert child.sent[-1] == ("end", "board gone", 1, 3) and t.round is None  # (skipped: the first, not ready)
    t.close()


def test_saw_keeps_only_the_newest_frame_and_counts_the_ones_it_replaces_in_a_round(tmp_path):
    t, _, _ = tracking(tmp_path)
    t.round = {"skipped": 0}
    for n in range(3):
        t.saw(SimpleNamespace(seq=n), SimpleNamespace(full=np.zeros((768, 1366, 3), np.uint8), t=n * 0.07,
                                                     visible=True), 1)
    assert t._newest[2] == 2 and t.round["skipped"] == 2


def test_the_trackers_loop_writes_a_rounds_answers_on_the_target_in_three_coordinates_and_its_end(tmp_path):
    frames = round_frames(n=30)
    inp = io.BytesIO()
    pickle.dump(("begin", "20261102-120000", {"fps": FPS}), inp, protocol=5)
    for n, (t, board, _) in enumerate(frames):
        corner = None if n == 10 else (100, 50)  # (one frame with the window not found)
        pickle.dump(("frame", t, n, 1, 1700000000.0 + t, board[..., ::-1].copy(), corner), inp, protocol=5)
    pickle.dump(("end", "board gone", 2), inp, protocol=5)
    pickle.dump(("quit",), inp, protocol=5)
    inp.seek(0)
    out = io.BytesIO()
    lt.serve(inp, out, tmp_path)
    out.seek(0)
    replies = []
    while True:
        try:
            replies.append(pickle.load(out))
        except EOFError:
            break
    rows = [json.loads(line) for line in open(tmp_path / "20261102-120000.jsonl", encoding="utf-8")]
    head, answers, end = rows[0], [r for r in rows[1:-1] if "angle" in r], rows[-1]
    assert replies == [("ready",)] + [("ready", a) for a in answers] + [("ended", end)]
    assert head["round"] == "20261102-120000" and head["board"] == list(BOARD_AT_NATIVE)
    assert len(answers) == len(frames) and answers[0]["white"] and answers[0]["seq"] == 0
    for n, (a, (_, _, z)) in enumerate(zip(answers, frames)):
        if n >= 6:
            assert np.hypot(a["board"][0] - z[0], a["board"][1] - z[1]) < 12, a
        assert a["client"] == [round(a["board"][0] + 308, 1), round(a["board"][1] + 180, 1)]
        assert a["desktop"] == (None if n == 10 else [round(a["client"][0] + 100, 1), round(a["client"][1] + 50, 1)])
    assert end["end"] == "board gone" and end["frames"] == len(frames) and end["skipped"] == 2


def test_the_trackers_loop_writes_each_template_it_takes_before_its_answer_and_each_answer_names_its_template(tmp_path):
    """The user, 2026-11-03: "Does the telemetry include which template it used to match? If not, we should add that?"
    (the review of 00:01's round drew its answers with another frame's template than the tracker's): before each white
    answer the template taken from its frame, as the tracker has it, and each answer's `template` its seq."""
    frames = round_frames(n=12)
    board = frames[0][1]
    board[:236][(board[:236] > 240).all(-1)] = (60, 120, 180)  # the first frame's star's top covered: a bigger one next
    inp = io.BytesIO()
    pickle.dump(("begin", "20261102-120002", {"fps": FPS}), inp, protocol=5)
    for n, (t, board, _) in enumerate(frames):
        pickle.dump(("frame", t, 100 + n, 1, 1700000000.0 + t, board[..., ::-1].copy(), None), inp, protocol=5)
    pickle.dump(("end", "board gone", 0), inp, protocol=5)
    pickle.dump(("quit",), inp, protocol=5)
    inp.seek(0)
    lt.serve(inp, io.BytesIO(), tmp_path)
    rows = [json.loads(line) for line in open(tmp_path / "20261102-120002.jsonl", encoding="utf-8")][1:-1]
    tracker = TargetTracker(fps=FPS)
    taken, template = [], None
    for k, row in enumerate(rows):
        if "points" in row:  # a template: right before its own frame's white answer, the tracker's own
            nxt = rows[k + 1]
            assert nxt["white"] and nxt["seq"] == row["template"] and row["board"] == nxt["board"]
            template = row["template"]
            taken.append(row)
            continue
        assert row["template"] == template  # an answer: the last template's
        a = tracker.step(frames[row["seq"] - 100][1], frames[row["seq"] - 100][0])
        assert a["white"] == (row["seq"] == template)
        if a["white"]:
            pts, nrm, _, area = tracker.template
            assert taken[-1]["area"] == area and taken[-1]["rim"] == RIM and len(taken[-1]["points"]) == len(pts)
            assert np.abs(np.array(taken[-1]["points"]) - pts).max() <= 0.05
            assert np.abs(np.array(taken[-1]["normals"]) - nrm).max() <= 0.0005
    assert [r["template"] for r in taken] == [100, 101] and rows[0] is taken[0] and taken[1]["area"] > taken[0]["area"]
    assert {r["template"] for r in rows if "angle" in r} == {100, 101}


def test_the_trackers_own_process_answers_through_its_pipes(tmp_path):
    frames = round_frames(n=8)
    child = lt.Child(tmp_path, FPS)
    replies = child.replies()
    assert next(replies) == ("ready",)
    child.send(("begin", "20261102-120001", {"fps": FPS}))
    for n, (t, board, _) in enumerate(frames):
        child.send(("frame", t, n, 1, 1700000000.0 + t, board[..., ::-1].copy(), (100, 50)))
        kind, answer = next(replies)
        assert kind == "ready" and answer["seq"] == n and len(answer["desktop"]) == 2
    child.send(("end", "board gone", 0))
    child.stop()
    rows = [json.loads(line) for line in open(tmp_path / "20261102-120001.jsonl", encoding="utf-8")]
    answers = [r for r in rows if "angle" in r]
    assert len(answers) == len(frames) and rows[-1]["frames"] == len(frames)


class Answering(FakeChild):
    """A tracker's process that answers each frame at once, as the real one's replies come."""

    def __init__(self):
        super().__init__()
        self.out = []
        self.more = threading.Condition()

    def send(self, msg):
        super().send(msg)
        with self.more:
            if msg[0] == "frame":
                self.out.append(("ready", {"seq": msg[2], "board": [1.0, 2.0], "client": [309.0, 182.0],
                                           "desktop": [409.0, 232.0]}))
            elif msg[0] == "end":
                self.out.append(("ended", {"end": msg[1], "frames": 1}))
            self.more.notify()

    def replies(self):
        yield ("ready",)
        while not self.done.is_set():
            with self.more:
                if not self.out:
                    self.more.wait(0.05)
                    continue
                msg = self.out.pop(0)
            yield msg


def test_each_answer_and_the_rounds_begin_and_end_go_into_the_services_telemetry(tmp_path):
    alarm = tmp_path / "ALARM.json"
    alarm.write_text("{}", encoding="utf-8")
    child, moves = Answering(), []

    def move(x, y, wait):
        moves.append((x, y))
        return True

    t = lt.LieTracking(folder=tmp_path / "out", alarm=alarm, spotter=Spotter(), child=lambda: child,
                       origin=lambda: (100, 50), mover=move)
    full = screen(round_frames(n=1)[0][1])
    with telemetry.captured() as records:
        t.step(0.0, item(0, 0.0, full))
        assert t.ready.wait(5)
        t.step(0.3, item(1, 0.3, full))
        deadline = time.monotonic() + 5
        while (not any(r["topic"] == "lie_target" for r in records) or not moves
               or not any(r.get("event") == "move finished" for r in records)) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert moves == [(409, 232)]  # the answer's desktop target really goes to the cursor mover
        t._end("board gone")
        while not any(r.get("event") == "end" for r in records) and time.monotonic() < deadline:
            time.sleep(0.01)
        child.stop()
    t.close()
    begin = next(r for r in records if r["topic"] == "lie_target_round" and r["event"] == "begin")
    answer = next(r for r in records if r["topic"] == "lie_target")
    end = next(r for r in records if r["topic"] == "lie_target_round" and r["event"] == "end")
    assert begin["file"].endswith(begin["round"] + ".jsonl")
    first = next(m[2] for m in child.sent if m[0] == "frame")  # (the first frame it was ready for: 0 or 1)
    assert answer["round"] == begin["round"] and answer["seq"] == first and answer["desktop"] == [409.0, 232.0]
    assert end["round"] == begin["round"] and end["end"] == "board gone"


def test_the_default_mover_uses_the_elevated_players_trail_based_cursor_move(monkeypatch, tmp_path):
    from gamelab.control import client
    calls = []

    def move(x, y, active):
        calls.append((x, y))
        assert active() is True
        return True

    monkeypatch.setattr(client, "move_lie_cursor", move)
    t, _, _ = tracking(tmp_path)
    assert t._move_cursor(409, 232, wait=lambda due: True) is True
    assert calls == [(309, 182)]  # desktop target converted to the player plan's client coordinates


def test_the_cursor_thread_takes_only_the_newest_queued_target(tmp_path):
    t, _, _ = tracking(tmp_path)
    calls, started, release = [], threading.Event(), threading.Event()

    def move(x, y, wait):
        calls.append((x, y))
        if len(calls) == 1:
            started.set()
            release.wait(5)
        return True

    t.mover = move
    t._queue_cursor((1, 1))       # no round: nothing sent
    t._queue_cursor(None)         # no desktop place: nothing sent
    assert t._cursor_thread is None
    t._cursor_round = "round"
    t._queue_cursor((1, 1))
    assert started.wait(5)
    t._queue_cursor((2, 2))
    t._queue_cursor((3, 3))       # replaces the answer not yet taken while the first trail runs
    release.set()
    deadline = time.monotonic() + 5
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert calls == [(1, 1), (3, 3)]
    t.close()


def test_ending_the_round_interrupts_the_current_cursor_trail(tmp_path):
    t, _, _ = tracking(tmp_path)
    started, done, result = threading.Event(), threading.Event(), []

    def move(x, y, wait):
        started.set()
        result.append(wait(time.perf_counter() + 5))
        done.set()
        return result[-1]

    t.mover = move
    t.round = {"name": "round", "started": 0.0, "board_at": 0.0, "sent": 1, "skipped": 0}
    t._cursor_round = "round"
    t._queue_cursor((100, 200))
    assert started.wait(5)
    t._end("board gone")
    assert done.wait(1) and result == [False]
    assert t.cursor_why_off == ""
    t.close()


def test_a_cursor_failure_is_logged_and_recorded_with_its_target(tmp_path):
    t, _, _ = tracking(tmp_path)
    said, error = [], OSError(5, "blocked")
    t.log = said.append

    def denied(x, y, wait):
        raise error

    t.mover = denied
    t._cursor_round = "round"
    with telemetry.captured() as records:
        t._queue_cursor((409, 232), seq=7)
        deadline = time.monotonic() + 5
        while not t.cursor_why_off and time.monotonic() < deadline:
            time.sleep(0.01)
        expected = f"OSError: {error}"
        assert t.cursor_why_off == expected
        assert said == [f"the Lie Detector's cursor movement is off: {expected}"]
        failed = next(r for r in records if r.get("event") == "failed")
        assert failed["topic"] == "lie_target_cursor" and failed["round"] == "round"
        assert (failed["x"], failed["y"], failed["seq"], failed["error"]) == (409, 232, 7, expected)
    t.close()
