"""What gamelab leaves on disk stays bounded (README: Memory and disk): a long run's telemetry goes on in
parts, the oldest telemetry is deleted when a run starts, the player's log is moved aside when the player
starts, and no cancel file stays in the inbox once its plan is over. All in temp folders."""

import os
import time

from gamelab import telemetry


def _make(folder, name, days_old, size, now):
    path = folder / name
    path.write_bytes(b"x" * size)
    t = now - days_old * 86400
    os.utime(path, (t, t))
    return path


def test_a_long_run_goes_on_in_parts_that_sort_after_it_and_say_what_they_continue(tmp_path):
    path = telemetry.record_to("hunt", folder=tmp_path, max_bytes=400)
    try:
        telemetry.emit("session", map=1, command="hunt", mode="live")
        for i in range(20):
            telemetry.emit("log", text=f"line {i:02d} " + "x" * 30)
    finally:
        telemetry.stop()
    parts = sorted(tmp_path.glob("*.jsonl"))  # by name, as the newest is found
    assert parts[0] == path and len(parts) >= 3 and all(p.stat().st_size <= 400 for p in parts)
    records = [telemetry.read(p) for p in parts]
    for n, (before, part, rs) in enumerate(zip(parts, parts[1:], records[1:]), start=2):
        # Each part starts with the run's session record, marked as the continuation of the part before.
        assert part.name.endswith(f"-hunt-part{n}.jsonl")
        assert rs[0]["topic"] == "session" and rs[0]["continues"] == before.name and rs[0]["part"] == n
        assert (rs[0]["map"], rs[0]["command"], rs[0]["mode"]) == (1, "hunt", "live") and "t" in rs[0]
    lines = [r["text"] for rs in records for r in rs if r["topic"] == "log"]
    assert lines == [f"line {i:02d} " + "x" * 30 for i in range(20)]  # none lost, none twice, in order
    assert telemetry.newest(tmp_path) == parts[-1]


def test_the_oldest_telemetry_goes_first_by_age_then_past_the_size_cap(tmp_path):
    now = time.time()
    old = _make(tmp_path, "20260901-120000-tour.jsonl", 20, 100, now)
    a = _make(tmp_path, "20261010-120000-nav.jsonl", 5, 300, now)
    b = _make(tmp_path, "20261011-120000-nav.jsonl", 4, 300, now)
    c = _make(tmp_path, "20261012-120000-tour.jsonl", 3, 300, now)
    other = _make(tmp_path, "notes.txt", 30, 10, now)  # not telemetry: never touched
    assert telemetry.prune(tmp_path, keep_bytes=700, now=now, dry_run=True) == [(old, 100), (a, 300)]
    assert old.exists() and a.exists()
    assert telemetry.prune(tmp_path, keep_bytes=700, now=now) == [(old, 100), (a, 300)]
    assert sorted(tmp_path.iterdir()) == sorted([b, c, other])
    assert telemetry.prune(tmp_path, keep_bytes=700, now=now) == []
    assert telemetry.prune(tmp_path / "none") == []  # no folder yet


def test_a_run_starts_by_deleting_telemetry_older_than_two_weeks(tmp_path, capsys):
    now = time.time()
    old = _make(tmp_path, "20260901-120000-tour.jsonl", telemetry.KEEP_DAYS + 1, 10, now)
    recent = _make(tmp_path, "20261019-120000-tour.jsonl", telemetry.KEEP_DAYS - 1, 10, now)
    path = telemetry.record_to("nav", folder=tmp_path)
    telemetry.stop()
    assert not old.exists() and recent.exists() and path.exists()
    assert "deleted the 1 oldest" in capsys.readouterr().err


def test_the_player_moves_its_log_aside_when_it_starts_past_5_mb(tmp_path, monkeypatch):
    from gamelab.control import player
    log, older = tmp_path / "player.log", tmp_path / "player.log.1"
    monkeypatch.setattr(player, "PLAYER_LOG", log)
    player._rotate_log()  # no log yet: nothing to do
    log.write_text("12:00:00 player started\n", encoding="utf-8")
    player._rotate_log()
    assert log.exists() and not older.exists()
    older.write_text("the one before\n", encoding="utf-8")
    log.write_bytes(b"x" * (player.LOG_MAX_BYTES + 1))
    player._rotate_log()
    assert not log.exists() and older.stat().st_size == player.LOG_MAX_BYTES + 1
    player._log("player started")
    assert log.read_text(encoding="utf-8").endswith(" player started\n")


def test_no_cancel_file_stays_in_the_inbox_once_its_plan_is_over():
    from gamelab.control import client  # tests/conftest.py points its inbox and outbox at a temp folder
    client.INBOX.mkdir(parents=True)
    client.OUTBOX.mkdir(parents=True)
    # Cut short while it runs: the player stops and answers; once the result is read, the cancel goes too.
    (client.INBOX / "p1.running").touch()
    assert client.cancel("p1") is False and (client.INBOX / "p1.cancel").exists()
    (client.INBOX / "p1.running").unlink()  # answered (control.player._answer)
    (client.OUTBOX / "p1.json").write_text('{"sent": [], "problems": ["cancelled"]}', encoding="utf-8")
    assert client.wait("p1", timeout=1)["problems"] == ["cancelled"]
    assert list(client.INBOX.iterdir()) == [] and list(client.OUTBOX.iterdir()) == []
    # Cut short once it's over (its result read already): nothing to tell the player.
    assert client.cancel("p1") is False and list(client.INBOX.iterdir()) == []
    # Not taken yet: withdrawn.
    (client.INBOX / "p2.json").write_text("{}", encoding="utf-8")
    assert client.cancel("p2") is True and list(client.INBOX.iterdir()) == []
