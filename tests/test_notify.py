"""The Lie Detector pushed to the user's phone through ntfy (control.notify; the user, 2026-10-29: "I need a better
notification method, I didn't catch the notification in time to solve this"): a cycle's pushes, the topic's file, a
push that fails, and the player's watchdog sending them. Nothing here reaches the network but a port nobody listens on."""
import json

from gamelab.control import notify
from gamelab.control.notify import Pusher

from tests.test_lie_recording import Screen, make_dog, play


def pusher(every_s=30.0):
    sent = []
    return Pusher(lambda title, text, priority: sent.append((title, priority, text)), every_s=every_s,
                  clock=lambda fmt: "01:35:32"), sent


def test_a_cycle_pushes_at_its_first_sight_again_every_30_s_while_it_can_be_answered_and_once_at_its_end():
    p, sent = pusher()
    p.seen(0.0, "mini-game notice")
    assert sent == [("Lie Detector up", "urgent", "mini-game notice seen at 01:35:32: answer it in the game now")]
    for t in (0.25, 10.0, 29.9):
        p.seen(t, "mini-game")
    assert len(sent) == 1
    p.seen(30.0, "mini-game")
    p.seen(45.0, "mini-game")
    p.seen(60.0, "quiz")
    assert [s[:2] for s in sent[1:]] == [("Lie Detector still up", "urgent")] * 2
    assert sent[1][2] == "mini-game, up 30 s: answer it in the game now"
    p.ended(75.0)
    assert sent[-1] == ("Lie Detector gone", "default", "off the screen at 01:35:32 after 75 s")
    p.ended(80.0)  # (once)
    p.seen(100.0, "countdown")  # a new cycle: pushed afresh
    assert len(sent) == 5 and sent[-1][:2] == ("Lie Detector up", "urgent")


def test_the_truth_booth_is_pushed_once_as_missed_and_not_repeated():
    p, sent = pusher()
    p.seen(0.0, "mini-game notice")
    p.seen(200.0, "truth booth")
    for t in range(201, 500, 10):
        p.seen(float(t), "truth booth")
    assert [s[:2] for s in sent] == [("Lie Detector up", "urgent"), ("Lie Detector missed", "high")]
    assert sent[1][2] == "the Truth Booth at 01:35:32, 200 s after it was first seen"
    q, sent = pusher()
    q.seen(0.0, "truth booth")  # first seen there (the player started with her in it)
    q.seen(40.0, "truth booth")
    assert sent == [("Lie Detector missed", "urgent", "truth booth seen at 01:35:32: she's in the Truth Booth")]


def test_the_topic_lives_in_the_labs_file_made_once_with_a_long_random_topic(tmp_path):
    path = tmp_path / "lab" / "ntfy.json"
    assert notify.load(path) is None
    cfg = notify.setup(path)
    assert cfg["server"] == "https://ntfy.sh" and cfg["topic"].startswith("gamelab-") and len(cfg["topic"]) >= 30
    assert notify.setup(path) == cfg == notify.load(path)  # (kept, not made again)
    assert notify.setup(tmp_path / "other.json")["topic"] != cfg["topic"]
    path.write_text(json.dumps({"topic": "t", "server": "https://example.test/"}), encoding="utf-8")
    assert notify.load(path) == {"server": "https://example.test", "topic": "t"}
    path.write_text("{", encoding="utf-8")
    assert notify.load(path) is None


def test_no_file_no_push_and_the_file_is_read_at_each_push(tmp_path, monkeypatch):
    path = tmp_path / "ntfy.json"
    went = []
    monkeypatch.setattr(notify, "send", lambda cfg, title, text, priority="urgent", tags="": went.append(cfg["topic"]))
    p = Pusher.from_lab(path=path)
    p.seen(0.0, "countdown")
    assert went == []  # no file yet
    p.ended(5.0)
    notify.setup(path)  # set up with the player running: its next push goes
    p.seen(10.0, "countdown")
    for _ in range(100):
        if went:
            break
        import time
        time.sleep(0.01)
    assert went == [notify.load(path)["topic"]]


def test_a_push_that_fails_is_said_and_never_raised():
    said = []
    t = notify.push({"server": "http://127.0.0.1:9", "topic": "x"}, "t", "x", log=said.append)
    t.join(timeout=notify.SEND_TIMEOUT_S + 5)
    assert not t.is_alive() and len(said) == 1 and said[0].startswith("the phone push failed: ")
    assert notify.push(None, "t", "x") is None


def test_the_watchdog_pushes_a_cycle_from_its_first_sight_to_its_end(tmp_path):
    screen = Screen()
    p, sent = pusher(every_s=0.5)
    dog, flag, said, alarms = make_dog(tmp_path, screen, pusher=p)
    t = play(dog, screen, [0, 0])
    assert sent == []
    t = play(dog, screen, [1, 1, 1, 2, 2], t)  # the countdown 0.75 s, then the mini-game's notice
    assert flag.is_set() and [s[0] for s in sent] == ["Lie Detector up", "Lie Detector still up",
                                                       "Lie Detector still up"]
    t = play(dog, screen, [0] * 5, t)  # gone 1 s (make_dog's end_s): the cycle's over
    assert dog.cycle_t0 is None and sent[-1][0] == "Lie Detector gone" and len(sent) == 4


def test_a_watchdog_without_a_pusher_pushes_nothing_and_a_pusher_that_raises_is_only_said(tmp_path):
    screen = Screen()
    dog, flag, said, alarms = make_dog(tmp_path, screen)
    assert dog.pusher is None
    play(dog, screen, [1, 0, 0, 0, 0, 0])

    class Broken:
        def seen(self, now, form):
            raise RuntimeError("no network")

        def ended(self, now):
            raise RuntimeError("no network")
    screen = Screen()
    dog, flag, said, alarms = make_dog(tmp_path / "b", screen, pusher=Broken())
    play(dog, screen, [1, 1, 0, 0, 0, 0, 0])
    assert flag.is_set() and dog.cycle_t0 is None and dog.troubles.get("the phone push failed", 0) >= 2
