"""What gamelab leaves on disk stays bounded (README: Memory and disk): a long run's telemetry goes on in
parts, the oldest telemetry past the size cap is deleted when a run starts (none for its age, unless asked),
the player's log is moved aside when the player starts, no cancel file stays in the inbox once its plan is
over, and `lab-usage` tells what's there. All in temp folders."""

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


def test_a_run_left_going_deletes_the_oldest_telemetry_as_it_goes_on_in_new_parts(tmp_path, monkeypatch, capsys):
    # watch writes about 16 MB an hour for as long as it's left running: pruning only when a run starts
    # would let the folder grow without bound.
    monkeypatch.setattr(telemetry, "KEEP_BYTES", 1000)
    path = telemetry.record_to("watch", folder=tmp_path, max_bytes=400)
    old = _make(tmp_path, "20260901-120000-tour.jsonl", 30, 10, time.time())
    try:
        telemetry.emit("session", map=1, command="watch", mode="watch")
        for i in range(40):
            telemetry.emit("log", text=f"line {i:02d} " + "x" * 30)
        writing = telemetry._sink.path
    finally:
        telemetry.stop()
    parts = sorted(tmp_path.glob("*.jsonl"))
    assert not old.exists() and not path.exists()  # the old run's file, then this run's first parts
    assert writing in parts and sum(p.stat().st_size for p in parts) <= 1000 + 400
    assert "oldest files" in capsys.readouterr().err
    # The part being written is never deleted, even past the cap.
    assert telemetry.prune(tmp_path, keep_bytes=0, spare=writing) and parts[-1].exists()


def test_the_oldest_telemetry_goes_past_the_size_cap_and_by_age_only_when_asked(tmp_path):
    now = time.time()
    old = _make(tmp_path, "20260901-120000-tour.jsonl", 400, 100, now)
    a = _make(tmp_path, "20261010-120000-nav.jsonl", 5, 300, now)
    b = _make(tmp_path, "20261011-120000-nav.jsonl", 4, 300, now)
    c = _make(tmp_path, "20261012-120000-tour.jsonl", 3, 300, now)
    other = _make(tmp_path, "notes.txt", 30, 10, now)  # not telemetry: never touched
    assert telemetry.KEEP_DAYS is None and telemetry.prune(tmp_path, now=now) == []  # a year old, within the cap
    assert telemetry.prune(tmp_path, keep_bytes=700, now=now, dry_run=True) == [(old, 100), (a, 300)]
    assert old.exists() and a.exists()
    # By age, only when it's asked for: those last written longer ago go first, then the cap as before.
    assert telemetry.prune(tmp_path, keep_days=4.5, now=now, dry_run=True) == [(old, 100), (a, 300)]
    assert telemetry.prune(tmp_path, keep_days=30, keep_bytes=700, now=now, dry_run=True) == [(old, 100), (a, 300)]
    assert telemetry.prune(tmp_path, keep_days=30, now=now, dry_run=True) == [(old, 100)]
    assert telemetry.prune(tmp_path, keep_bytes=700, now=now) == [(old, 100), (a, 300)]
    assert sorted(tmp_path.iterdir()) == sorted([b, c, other])
    assert telemetry.prune(tmp_path, keep_bytes=700, now=now) == []
    assert telemetry.prune(tmp_path / "none") == []  # no folder yet


def test_a_run_starts_by_deleting_the_oldest_telemetry_past_the_size_cap_but_none_for_its_age(tmp_path, capsys,
                                                                                               monkeypatch):
    # Live telemetry is evidence later fits use (the terrain and geometry fits of 2026-10-20 came from the live
    # runs of 2026-10-19 and 20): a file isn't deleted for its age, only to keep the folder under KEEP_BYTES.
    now = time.time()
    old = _make(tmp_path, "20250901-120000-tour.jsonl", 400, 10, now)
    recent = _make(tmp_path, "20261019-120000-tour.jsonl", 1, 10, now)
    path = telemetry.record_to("nav", folder=tmp_path)
    telemetry.stop()
    assert old.exists() and recent.exists() and path.exists() and capsys.readouterr().err == ""
    monkeypatch.setattr(telemetry, "KEEP_BYTES", 15)
    path = telemetry.record_to("nav", folder=tmp_path)
    telemetry.stop()
    assert not old.exists() and recent.exists() and path.exists()
    assert "deleted the 1 oldest files" in capsys.readouterr().err and "newest" in telemetry.kept()
    monkeypatch.setattr(telemetry, "KEEP_BYTES", 10 ** 9)
    monkeypatch.setattr(telemetry, "KEEP_DAYS", 14)  # with an age rule set, the old go for their age too
    older = _make(tmp_path, "20260901-120000-tour.jsonl", 15, 10, now)
    path = telemetry.record_to("nav", folder=tmp_path)
    telemetry.stop()
    assert not older.exists() and recent.exists() and "14 days" in capsys.readouterr().err


def test_the_part_being_written_is_the_newest_run_by_its_last_write(tmp_path):
    # telemetry.newest (the mini-map window's and the web page's LiveSource) goes by last write, not by name: in
    # the hour the clocks go back a finished run's name sorts after the part being written, which is newest.
    from gamelab.ui.sources import LiveSource
    now = time.time()
    finished = _make(tmp_path, "20261101-013000-tour.jsonl", 0.01, 10, now)  # written 15 minutes ago
    writing = tmp_path / "20261101-012000-watch-part2.jsonl"
    writing.write_text('{"t": 1, "topic": "session", "continues": "20261101-010000-watch.jsonl", "part": 2}\n',
                       encoding="utf-8")
    assert telemetry.newest(tmp_path) == writing
    clock = [0.0]
    source = LiveSource(tmp_path, clock=lambda: clock[0])
    fresh, records = source.poll()
    assert fresh and source.path == writing and records[0]["part"] == 2
    part3 = tmp_path / "20261101-020000-watch-part3.jsonl"
    part3.write_text('{"t": 2, "topic": "session", "part": 3}\n', encoding="utf-8")
    os.utime(writing, (now - 1, now - 1))
    clock[0] = 1.0  # the folder looked through again a second later: on to the next part
    fresh, records = source.poll()
    assert fresh and source.path == part3 and records[0]["part"] == 3


class _Listing:
    """os.scandir's listing, with the directory's copy of some files' last write out of date (as NTFS keeps
    it for a file still being written that nothing else opened): name -> the last write it says."""

    def __init__(self, listing, stale: dict):
        self.listing, self.stale = listing, stale

    def __enter__(self):
        self.listing.__enter__()
        return self

    def __exit__(self, *exc):
        return self.listing.__exit__(*exc)

    def __iter__(self):
        for e in self.listing:
            yield _Entry(e, self.stale.get(e.name))


class _Entry:
    def __init__(self, entry, stale):
        self._entry, self._stale, self.name, self.path = entry, stale, entry.name, entry.path

    def is_file(self):
        return self._entry.is_file()

    def stat(self):
        if self._stale is None:
            return self._entry.stat()
        return os.stat_result((0o100666, 0, 0, 0, 0, 0, 0, self._stale, self._stale, self._stale))


def test_a_part_still_being_written_is_the_newest_though_the_directory_says_it_was_written_when_made(tmp_path,
                                                                                                       monkeypatch):
    # On NTFS os.scandir reads the directory's copy of a file's last write and size, brought up to date only when
    # the file is opened or closed: a part `watch` is still writing, that nothing opens, shows as written when it
    # was made, 0 bytes long. A tour that finished after `watch` began stayed the newest, and the page froze on
    # it until that part closed (hours). newest asks os.stat of the files the directory says are recent, and
    # files_in (prune, lab-usage) asks it of every file.
    now = time.time()
    watch = _make(tmp_path, "20261020-100000-watch.jsonl", 1 / 86400, 5000, now)   # made an hour ago, written 1 s ago
    tour = _make(tmp_path, "20261020-103000-tour.jsonl", 600 / 86400, 300, now)   # finished 10 minutes ago
    old = _make(tmp_path, "20260901-100000-watch.jsonl", 40, 200, now)
    real = os.scandir
    monkeypatch.setattr(os, "scandir", lambda path: _Listing(real(path), {watch.name: now - 3600}))
    assert telemetry.newest(tmp_path, now=now) == watch
    assert {p.name: st.st_size for p, st in telemetry.files_in(tmp_path)} == {watch.name: 5000, tour.name: 300,
                                                                            old.name: 200}
    # Past RECENT_S the directory's word is taken (a part is never open that long): no os.stat per old file.
    monkeypatch.setattr(os, "scandir", lambda path: _Listing(real(path), {watch.name: now - telemetry.RECENT_S - 1}))
    assert telemetry.newest(tmp_path, now=now) == tour


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


def test_lab_usage_sizes_each_kind_and_deletes_telemetry_past_the_cap_only_when_asked(tmp_path, monkeypatch, capsys):
    from gamelab import paths
    from gamelab.__main__ import main
    from gamelab.analysis import disk
    now = time.time()
    for folder in ("recordings", "telemetry", "harness", "golden", "inbox"):
        (tmp_path / folder).mkdir()
    _make(tmp_path / "recordings", "chain.npz", 1, 5000, now)
    _make(tmp_path, "snap.png", 1, 3000, now)
    old = _make(tmp_path / "telemetry", "20260901-120000-tour.jsonl", 20, 100, now)
    new = _make(tmp_path / "telemetry", "20261019-120000-tour.jsonl", 1, 200, now)
    _make(tmp_path / "golden", "nav_up.json", 1, 50, now)
    _make(tmp_path / "inbox", "120000_plan_ab12.cancel", 1, 0, now)
    _make(tmp_path, "player.log", 1, 40, now)
    got = {u.kind: (u.files, u.bytes) for u in disk.usage(tmp_path)}
    assert got == {"recordings": (1, 5000), "screenshots": (1, 3000), "telemetry": (2, 300), "harness runs": (0, 0),
                   "golden traces": (1, 50), "player inbox, outbox": (1, 0), "other": (1, 40)}
    monkeypatch.setattr(paths, "LAB", tmp_path)
    monkeypatch.setattr(telemetry, "KEEP_BYTES", 250)  # the old one is past the cap (none goes for its age)
    assert main(["lab-usage"]) == 0
    out = capsys.readouterr().out
    assert "due to go now: 1 files (100 B)" in out and old.exists()  # told, not deleted
    assert main(["lab-usage", "--prune-telemetry"]) == 0
    assert "deleted 1 telemetry files (100 B)" in capsys.readouterr().out
    assert not old.exists() and new.exists() and (tmp_path / "recordings" / "chain.npz").exists()
