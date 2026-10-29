"""The Lie Detector pushed to the user's phone through ntfy (control.notify; the user, 2026-10-29: "I need a better
notification method, I didn't catch the notification in time to solve this"): a cycle's pushes, the topic's file, a
push that fails, and the player's watchdog sending them. Nothing here reaches the network but a port nobody listens on."""
import json

from gamelab.control import notify
from gamelab.control.notify import Pusher

from tests.test_lie_recording import Screen, make_dog, play


def pusher(every_s=30.0):
    sent = []
    return Pusher(lambda title, text, priority, text_too=False: sent.append((title, priority, text) + (("text",)
                                                                                                    if text_too else ())),
                  every_s=every_s, clock=lambda fmt: "01:35:32"), sent


def test_a_cycle_pushes_at_its_first_sight_again_every_30_s_while_it_can_be_answered_and_once_at_its_end():
    p, sent = pusher()
    p.seen(0.0, "mini-game notice")
    assert sent == [("Lie Detector up", "urgent", "mini-game notice seen at 01:35:32: answer it in the game now",
                     "text")]
    for t in (0.25, 10.0, 29.9):
        p.seen(t, "mini-game")
    assert len(sent) == 1
    p.seen(30.0, "mini-game")
    p.seen(45.0, "mini-game")
    p.seen(60.0, "quiz")
    assert [s[:2] for s in sent[1:]] == [("Lie Detector still up", "urgent")] * 2
    assert sent[1][2] == "mini-game, up 30 s: answer it in the game now"
    assert all(len(s) == 3 for s in sent[1:])  # (the repeats: no text)
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
    assert [s[3:] for s in sent] == [("text",), ("text",)]  # (a text each: the first sight, the Truth Booth)
    assert sent[1][2] == "the Truth Booth at 01:35:32, 200 s after it was first seen"
    q, sent = pusher()
    q.seen(0.0, "truth booth")  # first seen there (the player started with her in it)
    q.seen(40.0, "truth booth")
    assert sent == [("Lie Detector missed", "urgent", "truth booth seen at 01:35:32: she's in the Truth Booth",
                     "text")]


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


def test_the_tests_never_push_or_text_from_the_labs_own_files(tmp_path, monkeypatch):
    # The suite in the main checkout (2026-10-29 ~05:35, the game closed) texted the user "countdown seen at 05:35:28"
    # from the test above, its mail_path the real lab/smtp.json by default. Under conftest's GAMELAB_NO_PUSH the lab's
    # files, wherever they are, are never read for a push; named ones still are.
    lab_push, lab_mail = tmp_path / "ntfy.json", tmp_path / "smtp.json"
    lab_push.write_text(json.dumps({"topic": "t"}), encoding="utf-8")
    lab_mail.write_text(json.dumps({"host": "h", "user": "u", "password": "p", "to": ["a"]}), encoding="utf-8")
    monkeypatch.setattr(notify, "CONFIG", lab_push)
    monkeypatch.setattr(notify, "MAIL_CONFIG", lab_mail)
    went = []
    monkeypatch.setattr(notify, "push", lambda cfg, *a, **k: went.append(("push", cfg)))
    monkeypatch.setattr(notify, "mail", lambda cfg, *a, **k: went.append(("mail", cfg)))
    assert notify.NO_PUSH_ENV == "GAMELAB_NO_PUSH"
    assert notify.load() is None and notify.load_mail() is None
    Pusher.from_lab().seen(0.0, "countdown")
    assert went == [("push", None), ("mail", None)]
    assert notify.load(lab_push) == {"server": "https://ntfy.sh", "topic": "t"}  # (named: read)
    monkeypatch.delenv(notify.NO_PUSH_ENV)  # the player's own: the lab's files, looked up at each push
    went.clear()
    Pusher.from_lab().seen(0.0, "countdown")
    assert [w[0] for w in went] == ["push", "mail"] and went[0][1]["topic"] == "t" and went[1][1]["to"] == ["a"]


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


def test_a_text_is_mailed_through_the_labs_account_at_a_cycles_first_sight_only_once_its_login_is_filled_in(
        tmp_path, monkeypatch):
    """The user, 2026-10-29: "Send lie detector alert to <their email-to-text address> also"; asked how, "My email via
    SMTP": lab/smtp.json made with the address and the login left for them; read at each push; mailed on its own
    thread; a failure said, never raised."""
    sent, logins = [], []

    class SMTP:
        def __init__(self, host, port, timeout):
            self.host, self.port = host, port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            logins.append("tls")

        def login(self, user, password):
            logins.append((user, password))

        def send_message(self, msg):
            sent.append((self.host, self.port, msg["To"], msg["Subject"], msg.get_content().strip()))
    monkeypatch.setattr(notify.smtplib, "SMTP", SMTP)
    monkeypatch.setattr(notify, "send", lambda *a, **k: None)  # (the push: not here)
    path = tmp_path / "smtp.json"
    cfg, ready = notify.setup_mail("5550001111@sms.example", path)
    assert not ready and cfg["to"] == ["5550001111@sms.example"] and cfg["user"] == cfg["password"] == ""
    assert notify.load_mail(path) is None  # (no login yet: nothing goes)
    said = []
    p = Pusher.from_lab(log=said.append, path=tmp_path / "ntfy.json", mail_path=path)
    p.seen(0.0, "countdown")
    p.ended(10.0)
    cfg.update(user="me@example.com", password="app-password")
    path.write_text(json.dumps(cfg), encoding="utf-8")
    assert notify.setup_mail("5550001111@sms.example", path)[1]  # (kept, not doubled)
    p.seen(20.0, "mini-game notice")
    p.seen(60.0, "mini-game")  # a repeat: no text
    import time
    for _ in range(200):
        if sent:
            break
        time.sleep(0.01)
    time.sleep(0.05)
    assert sent == [("smtp.gmail.com", 587, "5550001111@sms.example", "Lie Detector up",
                     "mini-game notice seen at " + sent[0][4].split("seen at ")[1])]
    assert logins == ["tls", ("me@example.com", "app-password")] and said == []

    def refused(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr(notify, "send_mail", refused)
    t = notify.mail(notify.load_mail(path), "t", "x", log=said.append)
    t.join(timeout=5)
    assert said == ["the text's mail failed: OSError: refused"]
