"""The top of a rope (2026-10-26, Tree Dungeon, Forest Up North II, 1366x768): a touch cut her climb up the rope at x
111 near its top and she hung about 30 under platform 18, where her dot snapped her onto it; every Lucky Seven press was
refused there ("That cannot be used while you're on a rope or ladder." on the message bar), her MP stayed put, and out
of darts (behavior.darts) stopped the hunt after 20 of them, with her stars in her bag (22:58:32, 23:11:49, 23:12:42).
Its 12-press read of the pickup log read the party window ("7BugBaby / RideBaby").

The saved recording's frames (lab/recent/20261026-225844-out-of-darts, 15 fps, 22:57:14-22:58:44) as fixtures
(tests/fixtures/rope_top): her looks at every 3rd frame from 780 to 1310 as gamelab's look read them (her dot, her name
tag, her MP and HP), each look's message bar as one of four crops (a line redrawn has the very same pink px: on all 436
frames of the refusal, 673 px), the tour's 21 Lucky Seven presses (telemetry, 22:58:08.9-31.8), and two crops of the
pickup log's box. In the simulator, end to end: loop_tree_grind_rope_top (grinding as live: master's stack stops the
hunt there), loop_tree_rope_top, loop_tree_rope_top_15, nav_tree_rope_top_15 and nav_tree_rope_hit (tests/test_sim.py
runs every scenario)."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gamelab.behavior.darts import CHAT_EVERY, READ_EVERY, READ_FIRST, DartWatch, OutOfDarts
from gamelab.behavior.hunter import Hunter
from gamelab.behavior.safety import HpWatch
from gamelab.localization.pose import ROPE_TOP_GAP, ROPE_TOP_X, Localizer
from gamelab.motion.route import hung_from
from gamelab.perception.loot import says_no_darts
from gamelab.perception.messages import MIN_TEXT, ChatLine, ChatReader, chat_box, game_text, log_rows, refusal
from gamelab.screen import NATIVE
from gamelab.sim import scenarios

FIX = Path(__file__).parent / "fixtures" / "rope_top"
TREE = 10002073
ROPE = "That cannot be used while you're on a rope or ladder."
CROPS = {"rope": "bar_rope_00822.png", "played": "bar_played_01002.png", "megaphone": "bar_megaphone_00942.png",
         "notice": "bar_notice_00700.png"}


def tree_map():
    m = scenarios.real_map(TREE)
    if m is None:
        pytest.skip("needs Tree Dungeon, Forest Up North II's game data (gamedata/data)")
    return m


def _secs(at: str) -> float:
    h, m, s = at.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _fixture() -> dict:
    return json.loads((FIX / "looks.json").read_text(encoding="utf-8"))


def _frame(which: str) -> np.ndarray:
    """A whole 1366x768 window (RGB), black but for the message bar showing crop `which` ('none': nothing)."""
    f = np.zeros((768, 1366, 3), np.uint8)
    if which != "none":
        x, y, w, h = chat_box(1366, 768)
        f[y:y + h, x:x + w] = cv2.imread(str(FIX / CROPS[which]))[:, :, ::-1]
    return f


# --- her place: under a platform by a rope is on the rope ---------------------------------------------------------

def test_her_dot_more_than_17_under_a_platform_by_a_rope_hanging_from_it_is_her_on_the_rope():
    m = tree_map()
    loc = Localizer(m, geo=NATIVE)
    p18 = m.platform(18)
    surface = p18.y_at(111)
    # The rope at x 111 (19 -> 18): standing, her dot read a median 3.9 and at most 14.0 under the platform she stood
    # on (3,039 chain starts); hanging 19.3-30.8 under it at a rope's top. 15 under: still on 18 (navigator._rope_top's
    # case).
    for gap, on in ((-2.0, 18), (4.7, 18), (14.0, 18), (15.0, 18), (ROPE_TOP_GAP - 0.1, 18), (ROPE_TOP_GAP + 0.5, None),
                    (22.2, None), (30.8, None), (33.0, None)):
        w = loc.where_dot(loc.to_dot(111.0, surface + gap))
        assert (None if w.platform is None else w.platform.id) == on, gap
    hanging = loc.where_dot(loc.to_dot(101.3, -861.1))  # her dot hanging at 22:58:08-33 (map 101.3, -861.1)
    assert hanging.platform is None and (hanging.x, hanging.y) == pytest.approx((101.3, -861.1), abs=0.01)
    # Away from any rope (x 40: 48 from the rope at -8, 71 from 111's) a dot 30 under 18 snaps onto it as before.
    assert loc.where_dot(loc.to_dot(40.0, p18.y_at(40.0) + 30)).platform.id == 18
    assert loc.where_dot(loc.to_dot(111.0 + ROPE_TOP_X + 1, p18.y_at(128.0) + 30)).platform.id == 18
    # Every rope hanging from a platform on the map: 20 and 30 under its top's platform, on the rope; 10, on it.
    hung = [(r, hung_from(m, r)) for r in m.ropes if r.from_platform and hung_from(m, r) is not None]
    assert len(hung) >= 5
    for r, p in hung:
        for gap, on in ((10.0, p.id), (20.0, None), (30.0, None)):
            if p.y_at(r.x) + gap > r.bottom:
                continue
            w = loc.where_dot(loc.to_dot(r.x, p.y_at(r.x) + gap))
            assert (None if w.platform is None else w.platform.id) == on, (r, gap)


def test_the_saved_frames_hang_reads_her_on_the_rope_not_on_18():
    """Her looks off the recording: from 22:58:08.759 (frame 816) to 22:58:33.9 her dot stood at (101.3, -861.1), and
    was placed on 18 at x 113.3 (the service and the hunt alike); now on the rope, every look of it. Before, on 19
    (frames 780-792); climbing (795-813) and after the retreat's presses climbed her down (from 34.3), on the rope."""
    m = tree_map()
    loc = Localizer(m, geo=NATIVE)
    looks = _fixture()["looks"]
    placed = {d["i"]: loc.where_dot(tuple(d["dot"])) for d in looks}
    hang = [i for i in placed if 816 <= i <= 1194]
    assert len(hang) == 127 and all(placed[i].platform is None for i in hang)
    assert all(placed[i].platform is not None and placed[i].platform.id == 19 for i in placed if i <= 792)
    assert all(placed[i].platform is None for i in placed if 795 <= i <= 1290)
    # (her dot alone snapped her onto 18 at every look of the hang: what DOT_SNAP_LIFT alone gives)
    assert all(m.platform_under(*_raw(loc, looks, i), reach=50).id == 18 for i in hang)


def _raw(loc, looks, i):
    d = next(d for d in looks if d["i"] == i)
    x, y = loc.to_map(tuple(d["dot"]))
    return x, y - 25


# --- the message bar ----------------------------------------------------------------------------------------------

def test_the_game_s_lines_are_told_by_their_pink_and_the_refusals_by_their_words():
    pink = {name: int(np.count_nonzero(game_text(cv2.imread(str(FIX / crop))[:, :, ::-1])))
            for name, crop in CROPS.items()}
    assert pink["rope"] == 673 and pink["played"] == 531  # the game's own lines
    assert pink["megaphone"] < MIN_TEXT and pink["notice"] < MIN_TEXT  # a megaphone's purple, a notice's yellow
    # What the game said on the saved frames (676 lines read: D:/MapleHelper-scratch/rope/diag/msgbar_scan.py)
    assert refusal(ROPE) == "rope"
    assert refusal("You don't have enough MP to use this skill.") == "refused"
    # Only the refusals seen: the game's line for no stars isn't known and may say "cannot be used" (it counts), and a
    # line naming stars is the game saying she has none, never a refusal (the review's: "... without throwing stars.")
    assert refusal("That cannot be used here.") is None
    assert refusal("That cannot be used without throwing stars.") is None
    assert says_no_darts("That cannot be used without throwing stars.")
    for line in ("You are lacking the MP Potion that your pet is supposed to use.",
                 "Your pet is running low on the allotted MP potions.", "You have played MapleStory for 2 hour(s).",
                 "You cannot move to the requested channel at this time. Please try again later.",
                 "When a hit from an enemy, equipment swap, or use of certain skills, triggers", "", "Notice"):
        assert refusal(line) is None, line


def test_the_bar_is_read_by_ocr_only_when_it_shows_a_game_line_not_read_before():
    reads = []
    reader = ChatReader(ocr=lambda bgr: reads.append(bgr.shape) or ROPE)
    first = reader.read(_frame("rope"), 1.0)
    assert first.text == ROPE and first.refusal == "rope" and reads == [(24, 540, 3)]
    again = reader.read(_frame("rope"), 2.0)  # redrawn or left up: the same px, the text as read
    assert again.text == ROPE and again.same(first) and len(reads) == 1
    none = reader.read(_frame("megaphone"), 3.0)  # no game line: no OCR
    assert none.text == "" and none.mask is None and not none.same(first) and len(reads) == 1
    other = reader.read(_frame("played"), 4.0)
    assert len(reads) == 2 and not other.same(first)


def test_the_bar_s_refusal_read_by_gamelab_s_ocr():
    pytest.importorskip("rapidocr")
    line = ChatReader().read(_frame("rope"))
    assert line.refusal == "rope" and "rope or ladder" in line.text



def test_the_ocr_s_models_load_at_a_hunt_s_start_on_a_thread_of_their_own(monkeypatch):
    """The review's: the bar's first read (a stretch's 3rd press, mostly a monster standing on her) loaded the OCR's
    models then, 1.3 s in the hunt's own thread; a command that hunts loads them at its start (gamelab.__main__)."""
    import threading
    import maplehelper.ocr as ocr
    from gamelab.perception.messages import warm
    loaded = []
    monkeypatch.setattr(ocr, "shared", lambda: SimpleNamespace(load=lambda: loaded.append(
        threading.current_thread().name)))
    t = warm()
    t.join(5)
    assert loaded == ["ocr-warm"] and t.daemon

# --- the pickup log: its rows, not the party window's -------------------------------------------------------------

def test_the_pickup_log_keeps_its_rows_against_the_window_s_right_edge():
    rows = [SimpleNamespace(text="BugBaby", left=145.0, right=201.0), SimpleNamespace(text="RiceBaby", left=132.0,
            right=205.0), SimpleNamespace(text="You received EXP (+26)", left=164.0, right=296.0)]
    assert [r.text for r in log_rows(rows, 1366, 1065)] == ["You received EXP (+26)"]  # (ends at 1361: 1356 needed)


def test_the_pickup_log_read_off_the_saved_frames_leaves_the_party_window_out():
    pytest.importorskip("rapidocr")
    from maplehelper.ocr import LOOT, shared
    x, _, _, _ = LOOT.region.box((0, 0, 1366, 768))

    def read(name):
        crop = cv2.imread(str(FIX / name))
        rows = shared().read(np.ascontiguousarray(crop), min_height=LOOT.min_height) or []
        return [r.text for r in rows], [r.text for r in log_rows(rows, 1366, x)]
    seen, kept = read("pickup_01065.png")  # 22:58:25: no pickup for 4 s, the party window alone
    assert any("BugBaby" in t for t in seen) and kept == []
    seen, kept = read("pickup_ld_01615.png")  # 01:01: the log's lines, some behind the party window
    assert sum("mesos" in t for t in kept) == 2 and any("earned" in t for t in kept)


# --- out of darts: refused presses don't count --------------------------------------------------------------------

def _replay(chat: bool, ocr=lambda bgr: ROPE, still: bool = False):
    """DartWatch over the fixture's looks (her MP as read) and the tour's presses, the message bar read when it asks
    (`chat`) off each look's crop, with `still` as the hunter tells it (her dot where it was at the last refusal:
    Hunter._hear_chat): (the stop's time and presses, or None), the reads [(time, why)], the watch."""
    data = _fixture()
    texts = {"rope": ROPE, "played": "You have played MapleStory for 2 hour(s)"}
    w = DartWatch(20)
    reader = ChatReader(ocr=lambda bgr: "")
    events = sorted([(_secs(d["at"]), 0, d) for d in data["looks"]] + [(_secs(p["at"]), 1, p) for p in data["presses"]],
                    key=lambda e: e[:2])
    stop, reads, refused_at = None, [], None
    for t, press, d in events:
        if press:
            w.cast(t)
            continue
        w.see(t, d["mp"])
        if chat and w.asks_chat():
            line = reader.read(_frame(d["bar"]), t)
            line.text = texts.get(d["bar"], "")  # (what the OCR reads of each: the test above)
            here = tuple(d["dot"])
            same = still and refused_at is not None and max(abs(a - b) for a, b in zip(here, refused_at)) <= 0.6
            why = w.hear_chat(line, t, still=same)
            if why in ("rope", "refused"):
                refused_at = here
            reads.append((d["at"], why))
        if stop is None and (s := w.due(t)) is not None:
            stop = (d["at"], s.presses)
    return stop, reads, w


def test_on_the_saved_frames_the_stop_came_after_20_refused_presses_and_no_longer_does():
    """Master's watch (no bar read): the stop at the 20th press, 22:58:31.6 (live 22:58:32.3). Read at the 3rd press of
    a stretch and each 3rd after, the bar says she's on a rope at 22:58:11.4, 2.4 s after her first refused press:
    the stretch starts again (the hunter gets her off the rope then: up, onto 18). Left on it as she was (the
    recording), the line stays up the same px: not a new refusal, its presses count, until another line comes between
    (her 2 hours played at 22:58:21.1) and the refusal's back: again. No stop. With her dot where it was at the
    refusal (the hunter's `still`: the try didn't get her off), the line still up starts it again at every read."""
    assert (READ_FIRST, CHAT_EVERY) == (3, 3)
    stop, reads, w = _replay(chat=False)
    assert stop == ("22:58:31.552", 20) and w.longest[0] == 21
    stop, reads, w = _replay(chat=True)
    assert stop is None and reads[0] == ("22:58:11.360", "rope")
    assert [why for _, why in reads].count("rope") == 2 and w.refused == 2 and w.refused_presses == 15
    assert max(n for n, _ in [w.longest]) < 20
    # Her dot where it was at the refusal (the hunter's `still`: she hangs there yet), the line still up starts the
    # stretch again at every read, 6 of the 7 (the 4th read the 2 hours played): 6 presses counted in a row at the most.
    stop, reads, w = _replay(chat=True, still=True)
    assert stop is None and [why for _, why in reads].count("rope") == w.refused == 6 and reads[3] == ("22:58:21.347",
                                                                                                         None)
    assert w.longest[0] == READ_FIRST + CHAT_EVERY


def test_a_refusal_starts_a_stretch_again_once_a_line_left_up_doesn_t():
    rope = ChatLine(ROPE, np.ones((24, 540), bool), 1.0)
    w = DartWatch(20)
    w.see(0.0, 0.4)
    t = 0.0
    for _ in range(READ_FIRST):
        t += 0.9
        w.cast(t)
        w.see(t + 0.6, 0.4)
    assert w.shown == READ_FIRST and w.asks_chat()
    assert w.hear_chat(rope, t + 0.6) == "rope" and w.shown == 0 and not w.asks_chat()
    for _ in range(READ_FIRST):
        t += 0.9
        w.cast(t)
        w.see(t + 0.6, 0.4)
    assert w.hear_chat(ChatLine(ROPE, rope.mask.copy(), t), t + 0.6) is None and w.shown == READ_FIRST  # left up
    w.hear_chat(ChatLine("", None, t), t + 0.7)  # another line between (a megaphone over it)
    assert w.hear_chat(ChatLine(ROPE, rope.mask.copy(), t), t + 0.8) == "rope" and w.shown == 0
    # Her MP refused: a refusal too. Presses keyed while her look has her on a rope don't count at all.
    mp = np.zeros((24, 540), bool)
    mp[5:15, 10:300] = True
    assert w.hear_chat(ChatLine("You don't have enough MP to use this skill.", mp), t + 1) == "refused"
    w.roped = True
    w.cast(t + 2.0)
    assert w.keys == [] and w.roped_presses == 1
    assert "game refused" in w.summary() and "1 pressed on a rope" in w.summary()


def _presses(n: int, line_at, still_at=lambda t: False, every: float = 0.9):
    """DartWatch over `n` Lucky Seven presses `every` s apart that take no MP and kill nothing, a look 0.6 s after
    each (her MP the same), the bar read when it asks (asks_chat) showing `line_at(t)` (a ChatLine), with `still_at(t)`
    as the hunter tells it: (the stop or None, the watch, the reads' answers)."""
    w = DartWatch(20)
    w.see(0.0, 0.4)
    answers = []
    for k in range(n):
        t = every * (k + 1)
        w.cast(t)
        w.see(t + 0.6, 0.4)
        if w.asks_chat():
            answers.append(w.hear_chat(line_at(t + 0.6), t + 0.6, still=still_at(t + 0.6)))
        if (stop := w.due(t + 0.6)) is not None:
            return stop, w, answers
    return None, w, answers


def test_a_long_hang_with_the_refusal_left_up_never_stops_while_her_dot_stays_put():
    """The review's: one try to get her off that didn't (her dot where it was), the same line up all along (no other
    line between: a line left up and one redrawn are the same px), a hang of 45 presses as at 21:48:50 (38 s at the
    rope at x 111's top): the rope's refusal starts the stretch again at every read, her dot still there. Off the rope
    (her dot elsewhere), the line left up is an old one: her presses count, and out of darts comes at 20."""
    rope = np.ones((24, 540), bool)
    up = lambda t: ChatLine(ROPE, rope.copy(), t)  # noqa: E731
    stop, w, answers = _presses(45, up, still_at=lambda t: True)
    assert stop is None and answers.count("rope") == w.refused == 15 and w.longest[0] == READ_FIRST
    stop, w, answers = _presses(45, up, still_at=lambda t: False)
    assert stop is not None and stop.presses == 20 and w.refused == 1 and answers.count(None) >= 5


def test_a_line_on_the_bar_that_says_she_has_no_stars_stops_the_hunt_at_12_megaphones_or_not():
    """The review's: the game's line for no stars isn't known; worded "cannot be used" it was a refusal until
    2026-10-27, and with a megaphone between two reads of it (4 in the hang's 25 s on 10-26) each read started the
    stretch again, the stop at 71-249 s (master's 17.7 s). Now a line that says she has none
    (perception.loot.says_no_darts) has the stop come at READ_EVERY presses, as the pickup log's does, and no "cannot be
    used" is a refusal."""
    px = np.zeros((24, 540), bool)
    px[4:18, 20:400] = True
    for words in ("That cannot be used without throwing stars.", "You don't have enough throwing stars."):
        line = lambda t, words=words: ChatLine(words, px.copy(), t) if int(t) % 6 else ChatLine("", None, t)  # noqa
        stop, w, answers = _presses(40, line)
        assert stop is not None and stop.told and stop.presses == READ_EVERY and w.refused == 0, words
        assert words in str(stop) and "darts" in answers
    other = ChatLine("That cannot be used here.", px.copy())  # (no stars named: not a refusal now; it counts)
    stop, w, _ = _presses(40, lambda t: other)
    assert stop is not None and stop.presses == 20 and not stop.told and w.refused == 0


def test_the_stop_says_what_the_bar_showed():
    stop = OutOfDarts(20, 19.0, 0.1, bar="")
    assert str(stop).endswith("; the message bar then: no line of the game")
    assert "message bar" not in str(OutOfDarts(20, 19.0, 0.1))


def _hunter(line: ChatLine | None):
    h = object.__new__(Hunter)
    h.drink_if_low, h.hp_watch, h.darts = (lambda s: None), HpWatch(), DartWatch(20)
    h.logged, h.offs, h.refused_dot, h.off_tries = [], [], None, 0
    h.log = h.logged.append
    h.senses = SimpleNamespace(eyes=SimpleNamespace(messages=lambda: "", chat_line=lambda: line))
    h._off_rope = lambda s, down=False: h.offs.append((s, down))
    return h


def test_her_health_check_reads_the_bar_at_3_presses_and_gets_her_off_the_rope_it_says_she_s_on():
    h = _hunter(ChatLine(ROPE, np.ones((24, 540), bool), 5.0))
    h.darts.see(0.0, 0.4)
    for k in range(READ_FIRST):
        h.darts.cast(0.9 * (k + 1))
        h.darts.see(0.9 * (k + 1) + 0.6, 0.4)
    s = SimpleNamespace(t=4.0, hp=1.0, where=None)
    h.health_check(s)
    assert h.offs == [(s, False)] and h.darts.shown == 0
    assert h.logged == [f"  the game refused her presses: \"{ROPE}\" (3 in a row that took no MP: not counted)"]
    # No game line on it (or eyes that don't read it): nothing.
    h = _hunter(ChatLine("", None, 5.0))
    h.darts.see(0.0, 0.4)
    for k in range(READ_FIRST):
        h.darts.cast(0.9 * (k + 1))
        h.darts.see(0.9 * (k + 1) + 0.6, 0.4)
    h.health_check(s)
    assert h.offs == [] and h.logged == [] and h.darts.shown == READ_FIRST and h.darts.chat_read == READ_FIRST


def test_the_regular_attack_s_presses_at_a_rope_s_top_have_the_bar_read_and_get_her_off_the_rope():
    """From 2026-10-27: the simulator's tour_tree_run (the scenarios throw the regular attack) hung 31.9 s on the rope at
    x 146 under 30, placed on 30 at its top, 37 presses refused and none heard (DartWatch watched Lucky Seven's alone).
    Her regular presses keyed with her look placed at a rope's top (DartWatch.at_top), 3 in a row with no kill, have the
    bar read; elsewhere, or after a kill, they start again; none of out of darts' presses."""
    h = _hunter(ChatLine(ROPE, np.ones((24, 540), bool), 5.0))
    s = SimpleNamespace(t=4.0, hp=1.0, where=None)
    h.darts.at_top = True
    for k in range(READ_FIRST - 1):
        h.darts.regular(0.9 * (k + 1))
    h.health_check(s)
    assert h.offs == [] and h.darts.top_presses == READ_FIRST - 1
    h.darts.kill(2.0)  # (a kill of hers: they did something)
    for k in range(READ_FIRST):
        h.darts.regular(2.0 + 0.9 * (k + 1))
    assert h.darts.keys == [] and h.darts.shown == 0  # (no press of the stretch: out of darts never counts them)
    h.health_check(s)
    assert h.offs == [(s, False)] and h.darts.top_presses == 0
    assert h.logged == [f"  the game refused her presses: \"{ROPE}\" (3 in a row at a rope's top, the regular attack's or the claw's)"]
    # The claw's swipes there (no press of out of darts' either: DartWatch.cast's `swipe`) are watched so too: with them
    # left out, loop_tree_rope_top_15 hung 14.8 s at the rope at x 111's top, its presses at a Green Mushroom 73 off.
    h = _hunter(ChatLine(ROPE, np.ones((24, 540), bool), 5.0))
    h.darts.at_top = True
    for k in range(READ_FIRST):
        h.darts.cast(0.9 * (k + 1), swipe=True)
    h.health_check(s)
    assert h.offs == [(s, False)] and h.darts.keys == [] and h.darts.swiped_presses == READ_FIRST
    # Keyed away from a rope's top: they start again, and no read.
    h = _hunter(ChatLine(ROPE, np.ones((24, 540), bool), 5.0))
    for k in range(2 * READ_FIRST):
        h.darts.at_top = k % 2 == 0
        h.darts.regular(0.9 * (k + 1))
    h.health_check(s)
    assert h.offs == [] and h.logged == [] and h.darts.top_presses == 0


def test_her_dot_where_it_was_at_the_refusal_has_her_got_off_again_down_the_rope_the_second_time():
    """The review's: the one try up didn't free her (her dot where it was at the refusal, the line still up the same
    px): the stretch starts again and she's got off again, down the rope (Mover.off_rope's `down`), then up again. Her
    dot elsewhere: the line left up is an old one, nothing."""
    h = _hunter(None)
    t = 0.0
    h.senses.eyes.chat_line = lambda: ChatLine(ROPE, np.ones((24, 540), bool), t + 0.6)  # (redrawn: the same px)
    hang = SimpleNamespace(t=4.0, hp=1.0, where=SimpleNamespace(dot=(83.478, 151.522)))
    h.darts.see(0.0, 0.4)
    for tries in range(3):
        for _ in range(READ_FIRST):
            t += 0.9
            h.darts.cast(t)
            h.darts.see(t + 0.6, 0.4)
        h.health_check(hang)
        assert h.darts.shown == 0 and len(h.offs) == tries + 1
    assert [down for _, down in h.offs] == [False, True, False] and h.off_tries == 2
    assert h.logged[1].endswith("; her dot where it was then: try 2")
    moved = SimpleNamespace(t=9.0, hp=1.0, where=SimpleNamespace(dot=(83.478, 149.5)))  # (2 px up: on 18)
    for _ in range(READ_FIRST):
        t += 0.9
        h.darts.cast(t)
        h.darts.see(t + 0.6, 0.4)
    h.health_check(moved)
    assert len(h.offs) == 3 and h.darts.shown == READ_FIRST


# --- the navigator: a move that went nowhere, her climb on after a touch -----------------------------------------

def _navigator(m):
    from gamelab.control.navigator import Navigator
    nav = object.__new__(Navigator)
    nav.model, nav.logged, nav.offs = m, [], []
    nav.log = nav.logged.append
    nav.senses = SimpleNamespace(localizer=Localizer(m, geo=NATIVE))
    nav.mover = SimpleNamespace(off_rope=lambda s, rope, at: nav.offs.append((rope.x, at)) or "off")
    return nav


def test_a_move_that_never_moved_her_dot_by_a_rope_s_top_is_the_rope_only_more_than_12_under():
    """The review's: a jump_up straight up from the rope's top that falls back where it began reads as a move that
    never moved her dot too; standing there her glances read 9.2 under the platform at the most, the hang of 22:19
    14.8 (navigator.STUCK_GAP)."""
    from gamelab.control.navigator import STUCK_GAP
    m = tree_map()
    nav = _navigator(m)
    surface = m.platform(18).y_at(111.0)
    for gap, off in ((4.0, False), (9.2, False), (STUCK_GAP - 0.1, False), (14.8, True), (16.5, True)):
        nav.offs.clear()
        where = nav.senses.localizer.where_dot(nav.senses.localizer.to_dot(111.0, surface + gap))
        assert where.platform is not None and where.platform.id == 18, gap
        s = SimpleNamespace(where=where)
        got = nav._rope_top(SimpleNamespace(fresh=1), s, s)
        assert (got == "off") == off and bool(nav.offs) == off, gap


def test_her_climb_on_after_a_touch_only_under_the_rope_s_top():
    """The review's: up again went at -2 and -12 under the top, her off the rope already, and its hold kept her by the
    mushroom on the platform: under the top by CLIMB_ON_UNDER at least, by her dot or (none) the plan's height; and
    held as a chain's climb holds it (52 under: 0.75 s, where climb_ms had 0.945 s)."""
    import time
    from gamelab.control.navigator import CLIMB_ON_UNDER, _Running
    from gamelab.motion.bodies import Climb
    m = tree_map()
    nav = _navigator(m)
    loc = nav.senses.localizer
    top, bottom = -880.0, -760.0
    running = _Running("p", [SimpleNamespace(rope=[Climb(0.0, 2.0, 111.0, bottom, top)])], time.monotonic() - 1.0,
                       False)
    assert nav._climbing(running, loc.to_dot(111.0, top + 30)) == (top, pytest.approx(top + 30))
    assert nav._climbing(running, loc.to_dot(111.0, top + CLIMB_ON_UNDER)) == (top, pytest.approx(top + CLIMB_ON_UNDER))
    for under in (CLIMB_ON_UNDER - 1, 0.0, -2.0, -12.0):
        assert nav._climbing(running, loc.to_dot(111.0, top + under)) is None, under
    assert nav._climbing(running, None) == (top, pytest.approx(top + 60, abs=0.5))  # (the plan's: halfway up)
    late = running._replace(start=time.monotonic() - 2.2)  # (the plan's past the top, in UP_EXTRA)
    assert nav._climbing(late, None) is None
    played = []
    nav.actions = SimpleNamespace(keys=SimpleNamespace(up=SimpleNamespace(name="up")),
                                  play=lambda events, name: played.append((events, name)))
    nav._climb_on(top, top + 52)
    assert played == [([[0, "up", "down"], [750, "up", "up"]], "climb_on")]
    assert nav.logged[-1] == "    hit on the rope 52 under its top: up again (750 ms)"
