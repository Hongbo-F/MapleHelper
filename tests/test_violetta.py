"""The Lie Detector's Violetta game: the rules solver, the answer window's cue, the pass's dialog, the model on real
frames, the live watcher, and the pass's OK click through the elevated player (perception.violetta,
control.violetta_live, alarm.LIE_OK)."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gamelab.control import alarm as alarm_mod
from gamelab.control import client
from gamelab.perception import violetta

FIXTURES = Path(__file__).parent / "fixtures" / "violetta"
SLOTS = [526.0, 628.0, 730.0, 834.0]


def frame(name):
    return np.asarray(Image.open(FIXTURES / f"2026-11-03_013920_{name}.png").convert("RGB"))


def boxes_for(slots, hops, ts, lunge=0):
    """Each frame's boxes as the rules draw them: `hops` [(t0, T, perm), ...] in turn (a staying one's lunge the
    `lunge`th size: 0 in front, 1 behind), every Violetta seen."""
    out = []
    timeline = list(hops)
    for t in ts:
        boxes = []
        xs = np.asarray(slots, dtype=float)
        sizes = np.ones(4)
        cur = list(range(4))  # each Violetta's slot as the hops go
        for t0, T, perm in timeline:
            px, pls, _ = violetta._predict(slots, t0, T, np.array([t]))
            k = next(i for i, p in enumerate(violetta.PERMS) if tuple(p) == tuple(perm))
            if t >= t0:
                xs_new = px[k, 0]
                sz = np.exp(pls[lunge][k, 0])
                # Violetta v is at slot cur[v] when this hop starts: it moves as that slot does
                xs = np.array([xs_new[cur[v]] for v in range(4)])
                sizes = np.array([sz[cur[v]] for v in range(4)])
                if t >= t0 + T:
                    cur = [perm[c] for c in cur]
        for v in range(4):
            boxes.append({"x": float(xs[v]), "y": 400.0, "size": float(sizes[v]), "score": 0.9, "cls": 0})
        out.append(boxes)
    return out


@pytest.mark.parametrize("perm", [(1, 0, 3, 2), (2, 0, 3, 1), (3, 2, 1, 0), (1, 2, 3, 0), (0, 2, 1, 3)])
def test_a_one_hop_round_is_solved_from_the_boxes_the_rules_predict(perm):
    ts = np.arange(0, 7.0, 1 / 15)
    frames = boxes_for(SLOTS, [(1.0, 4.4, perm)], ts)
    got, kind, info = violetta.solve_round(ts, frames, SLOTS)
    assert got == perm and kind == "one"
    assert info["one"][3] == 4.4 and abs(info["one"][2] - 1.0) < 0.15


def test_a_two_hop_round_is_solved_whole():
    """Two hops of D/2 (a random middle arrangement): the made-up one's slot is where both take it."""
    ts = np.arange(0, 7.0, 1 / 15)
    p1, p2 = (1, 0, 3, 2), (3, 2, 1, 0)
    frames = boxes_for(SLOTS, [(1.0, 2.05, p1), (3.05, 2.05, p2)], ts)
    got, kind, info = violetta.solve_round(ts, frames, SLOTS)
    assert kind == "two" and got == tuple(p2[p1[i]] for i in range(4))


def test_the_lanes_are_the_clients():
    assert violetta.lane(2, 1) > 100 > violetta.lane(1, 2)       # one left in front, one right behind
    assert violetta.lane(0, 3) > violetta.lane(2, 1)              # two or more right: in front, bigger
    assert violetta.lane(3, 0) < violetta.lane(1, 2)              # two or more left: behind, smaller
    assert violetta.lane(2, 2) == 100.0


def test_the_slots_and_the_made_up_ones_start_come_from_the_standing_boxes():
    standing = [[{"x": s + 4, "y": 400, "size": 1.0, "score": 0.9, "cls": 1 if k == 2 else 0}
                 for k, s in enumerate(SLOTS)] for _ in range(5)]
    slots = violetta.slots_of(standing)
    assert slots == [s + 4 for s in SLOTS]
    assert violetta.start_slot(standing, slots) == 2
    assert violetta.start_slot([[{"x": 526, "y": 400, "size": 1, "score": 0.9, "cls": 0}]], slots) is None
    assert violetta.slots_of([]) == list(violetta.SLOTS)


def test_the_answer_window_is_the_buttons_lit():
    assert violetta.buttons_lit(frame("answer_window"))
    for name in ("standing", "shuffling", "passed"):
        assert not violetta.buttons_lit(frame(name)), name


def test_the_pass_s_dialog_gives_its_ok_button_and_nothing_else_does():
    ok = violetta.OkDialog()
    assert ok(frame("passed")) == (784, 430)
    shifted = np.roll(frame("passed"), (6, -9), axis=(0, 1))
    assert ok(shifted) == (775, 436)
    for name in ("standing", "shuffling", "answer_window"):
        assert ok(frame(name)) is None, name
    x, y = ok(frame("passed"))
    x0, y0, x1, y1 = alarm_mod.LIE_OK_BOX
    assert x0 <= x < x1 and y0 <= y < y1  # the click the player will take
    assert not (y0 <= violetta.BUTTONS[1] < y1 or y0 <= violetta.BUTTONS[3] < y1)  # never an answer button's row


def test_the_model_finds_the_four_and_the_made_up_one_on_a_real_frame():
    pytest.importorskip("ultralytics")
    det = violetta.Detector(device="cpu")
    standing, shuffling = det.detect([frame("standing"), frame("shuffling")])
    assert len([b for b in standing if b["cls"] == 0]) == 3 and len([b for b in standing if b["cls"] == 1]) == 1
    slots = violetta.slots_of([standing])
    assert violetta.start_slot([standing], slots) == 0  # the made-up one at slot 1 (the 01:39 game)
    assert all(0.8 < b["size"] < 1.25 for b in standing)
    assert len(shuffling) >= 3  # (under the spotlights, spinning)


# --- the live watcher ---------------------------------------------------------------------------------------------

class _Frame:
    def __init__(self, full, t):
        self.full, self.t, self.visible = full, t, True


class _Spotter:
    def __init__(self):
        self.window = True

    def over_play(self, full):
        return "countdown" if self.window else None


def test_the_live_watcher_says_each_rounds_slot_and_clicks_the_pass_s_ok(tmp_path, monkeypatch):
    from gamelab.control import violetta_live as vl
    from gamelab import telemetry
    alarm = tmp_path / "ALARM.json"
    alarm.write_text("{}", encoding="utf-8")
    events = []
    monkeypatch.setattr(telemetry, "emit", lambda topic, **kw: events.append((topic, kw)))
    lit, dark = frame("answer_window"), frame("shuffling")
    # two rounds: the made-up one starts at slot 2 (its face before the first window), swaps 2<->3 then 1<->3 ...
    ts = np.arange(0, 16.0, 1 / 15)
    hops = [(2.0, 4.1, (0, 2, 1, 3)), (9.0, 4.1, (2, 1, 0, 3))]
    boxes = boxes_for(SLOTS, hops, ts)
    for k, t in enumerate(ts):
        if t < 1.5:
            boxes[k][1]["cls"] = 1
    windows = {int(round(7.0 * 15)), int(round(14.0 * 15))}  # the buttons lit from these frames for a second
    lit_at = {k for w in windows for k in range(w, w + 15)}
    calls = {"det": 0}

    def detector(frames):
        k = calls["det"]
        calls["det"] += 1
        return [boxes[k] if k < len(boxes) else []]

    pushed, clicks, pointed = [], [], []
    ok_at = {"on": False}

    def pointer(x, y, active):  # (the player's move: only while the buttons are lit)
        pointed.append((x, y, active()))
        return True

    w = vl.VioletaLive(alarm=alarm, spotter=_Spotter(), log=lambda m: None, detector=detector,
                       ok=lambda full: (784, 430) if ok_at["on"] else None,
                       clicker=lambda x, y: clicks.append((x, y)) or True, push=lambda t, x: pushed.append(t),
                       pointer=pointer)
    now = 1000.0
    for k, t in enumerate(ts):
        now = 1000.0 + t
        w.step(now, (t, lit if k in lit_at else dark, k, 1))
        if w.pointing is not None:
            w.pointing.join(5)
    assert pushed == ["Violetta round 1: slot 3", "Violetta round 2: slot 1"]
    # the pointer onto each answer's button (slots 3 and 1), never a click there
    assert pointed == [(735, 505, True), (527, 505, True)]
    assert all(not (vl_y0 <= 505 < vl_y1) for vl_y0, vl_y1 in [(alarm_mod.LIE_OK_BOX[1], alarm_mod.LIE_OK_BOX[3])])
    x0, y0, x1, y1 = alarm_mod.LIE_CURSOR_BOX
    assert all(x0 <= x < x1 and y0 <= y < y1 for x, y in violetta.BUTTON_AT)  # (the move-only plan the player takes)
    rounds = [kw for topic, kw in events if topic == "violetta" and kw["event"] == "round"]
    assert [r["slot"] for r in rounds] == [3, 1] and rounds[0]["start"] == 2
    # the window gone: the game over; the pass's dialog seen twice in a row: its OK clicked once, not again at once
    w.spotter.window = False
    ok_at["on"] = True
    for k in range(1, 30):
        now += 0.25
        w.step(now, (ts[-1] + k * 0.25, dark, 10000 + k, 1))
    assert w.game is None and any(kw["event"] == "end" for topic, kw in events if topic == "violetta")
    assert clicks == [(784, 430), (784, 430), (784, 430)]  # (seen still: tried again OK_RETRY_S apart, 3 at most)
    assert [kw["clicked"] for topic, kw in events if kw.get("event") == "ok"] == [True, True, True]


def test_the_live_watcher_ignores_frames_with_no_alarm_and_no_game(tmp_path):
    from gamelab.control import violetta_live as vl
    w = vl.VioletaLive(alarm=tmp_path / "none.json", spotter=_Spotter(), detector=lambda f: [[]])
    w.saw(None, _Frame(frame("standing"), 1.0), 1)
    assert not w._queue


def test_together_starts_feeds_and_closes_every_watcher():
    from gamelab.control.violetta_live import Together
    seen = []

    class W:
        def __init__(self, name):
            self.name = name

        def start(self):
            seen.append(("start", self.name))

        def saw(self, st, frame, run):
            seen.append(("saw", self.name))

        def line(self):
            return f"; {self.name}"

        def close(self, timeout=10.0):
            seen.append(("close", self.name))

    t = Together(W("a"), W("b"))
    t.start()
    t.saw(None, None, 1)
    assert t.line() == "; a; b"
    t.close()
    assert seen == [("start", "a"), ("start", "b"), ("saw", "a"), ("saw", "b"), ("close", "a"), ("close", "b")]


# --- the pass's OK click ------------------------------------------------------------------------------------------

def _ok_plan(events=None, **kw):
    return {"kind": alarm_mod.LIE_OK, "focus": False, "capture": {"lead_ms": 0},
            "events": [[0, "mouse", 784, 430], [1000, "lbutton", "down"], [1080, "lbutton", "up"]]
            if events is None else events, **kw}


def test_the_pass_s_ok_click_is_one_click_on_its_button_and_nothing_else():
    from gamelab.control.player import alarm_refusal, lie_ok_refusal
    assert lie_ok_refusal(_ok_plan()) is None
    assert alarm_refusal(_ok_plan(), None, None, [], 100.0) is None
    for x, y in ((757, 430), (812, 430), (784, 418), (784, 442), (680, 520)):  # off the button; an answer button
        assert "stays on its button" in lie_ok_refusal(_ok_plan([[0, "mouse", x, y], [1000, "lbutton", "down"],
                                                                  [1080, "lbutton", "up"]]))
    for events in ([[0, "mouse", 784, 430], [1000, "lbutton", "down"]],
                   [[0, "mouse", 784, 430], [1000, "rbutton", "down"], [1080, "rbutton", "up"]],
                   [[0, "mouse", 784, 430], [1000, "lbutton", "down"], [1080, "lbutton", "up"], [1100, "1", "down"]],
                   [[0, "1", "down"], [1000, "lbutton", "down"], [1080, "lbutton", "up"]],
                   [[5, "mouse", 784, 430], [1000, "lbutton", "down"], [1080, "lbutton", "up"]]):
        assert "nothing else" in lie_ok_refusal(_ok_plan(events)), events
    for events in ([[0, "mouse", 784, 430], [0, "lbutton", "down"], [80, "lbutton", "up"]],
                   [[0, "mouse", 784, 430], [1600, "lbutton", "down"], [1680, "lbutton", "up"]],
                   [[0, "mouse", 784, 430], [1000, "lbutton", "down"], [1005, "lbutton", "up"]],
                   [[0, "mouse", 784, 430], [1000, "lbutton", "down"], [1400, "lbutton", "up"]]):
        assert "pressed" in lie_ok_refusal(_ok_plan(events)), events
    for plan in (_ok_plan(focus=True), _ok_plan(capture={"lead_ms": 50}), _ok_plan(end_ms=2000),
                 _ok_plan(capture={"lead_ms": 0, "fps": 1}), _ok_plan(window_client=[1366, 768])):
        assert lie_ok_refusal(plan) is not None, plan


def test_the_player_clicks_the_ok_only_with_the_game_under_the_pointer(monkeypatch, tmp_path):
    from test_alarm import _Game, _alarmed_player
    from gamelab.control import player as player_mod
    game = _Game(monkeypatch, tmp_path)
    p, said = _alarmed_player()
    monkeypatch.setattr(player_mod.window, "move_cursor", lambda x, y, wait, played: True)
    monkeypatch.setattr(player_mod.window, "pointer_on", lambda hwnd: True)
    plan = dict(_ok_plan(), id="ok1")
    result = p.run(plan)
    assert "error" not in result and game.sent == [("lbutton", True), ("lbutton", False)] and game.fronted == []
    game.sent.clear()
    monkeypatch.setattr(player_mod.window, "pointer_on", lambda hwnd: False)  # another window over the dialog
    result = p.run(dict(_ok_plan(), id="ok2"))
    assert game.sent == [] and result["problems"] == ["another window is under the pointer; the OK not clicked"]
    bad = dict(_ok_plan([[0, "mouse", 680, 520], [1000, "lbutton", "down"], [1080, "lbutton", "up"]]), id="ok3")
    result = p.run(bad)
    assert "error" in result and game.sent == []


def test_the_player_holds_an_ok_click_to_its_rules_with_no_alarm_too(monkeypatch, tmp_path):
    from test_alarm import _Game
    from gamelab.control import player as player_mod
    from gamelab.control.player import Player, allowed_keys
    from test_alarm import _keys
    game = _Game(monkeypatch, tmp_path)
    monkeypatch.setattr(player_mod, "alarm_up", lambda: False)
    keys = _keys()
    p = Player(allowed_keys(keys), "Game", threading.Event(), keys=keys, log=lambda m: None)
    bad = dict(_ok_plan([[0, "mouse", 680, 520], [1000, "lbutton", "down"], [1080, "lbutton", "up"]]), id="ok4")
    assert "stays on its button" in p.run(bad)["error"] and game.sent == []


def test_the_client_hands_the_ok_click_to_the_player(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "INBOX", tmp_path)
    monkeypatch.setattr(client, "wait", lambda pid, timeout: {
        "sent": [[0, "mouse", 784, 430], [1001, "lbutton", "down"], [1081, "lbutton", "up"]], "problems": []})
    assert client.click_lie_ok(784, 430) is True
    plan = json.loads(next(tmp_path.glob("*lie_ok*.json")).read_text(encoding="utf-8"))
    assert plan == {"events": [[0, "mouse", 784, 430], [1000, "lbutton", "down"], [1080, "lbutton", "up"]],
                    "capture": {"fps": 0, "duration_ms": 0, "scale": 1, "lead_ms": 0},
                    "focus": False, "kind": alarm_mod.LIE_OK}
    from gamelab.control.player import lie_ok_refusal
    assert lie_ok_refusal(plan) is None  # the player takes what the client sends
    monkeypatch.setattr(client, "wait", lambda pid, timeout: {"error": "Lie Detector: refused"})
    with pytest.raises(client.PlayerError, match="refused"):
        client.click_lie_ok(784, 430)
