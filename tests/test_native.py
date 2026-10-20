"""gamelab's C pieces (gamelab/native.py): built once per source, old builds cleaned up, and turned off cleanly.
What each piece computes is compared with the Python it stands in for where that Python is tested
(test_perception: the monster search; test_navigation: her flight)."""

import subprocess

from gamelab import native


def test_a_build_is_named_for_its_source_and_replaces_older_ones(monkeypatch, tmp_path):
    monkeypatch.setattr(native, "BUILD", tmp_path / "build")
    monkeypatch.setattr(native, "VCVARS", tmp_path / "vcvars64.bat")
    (tmp_path / "vcvars64.bat").write_text("")
    (tmp_path / "build").mkdir()
    old = tmp_path / "build" / "native-000000000000.dll"
    old.write_bytes(b"an earlier source's")
    (tmp_path / "build" / "notes.txt").write_text("not a build")
    calls = []

    def compile_(command, cwd, **kw):  # stands in for cl: writes the DLL /Fe names
        calls.append(command)
        dll = command.split('/Fe:"')[1].split('"')[0]
        open(dll, "wb").write(b"built")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(native.subprocess, "run", compile_)
    out = native.build()
    assert out == native.dll_path() and out.read_bytes() == b"built" and out.parent == tmp_path / "build"
    assert not old.exists() and (tmp_path / "build" / "notes.txt").exists()
    assert "/fp:precise" in calls[0] and str(native.SOURCE) in calls[0]
    assert native.build() == out and len(calls) == 1  # built once
    assert [p.name for p in (tmp_path / "build").iterdir() if p.is_dir()] == []  # no temp folder left


def test_a_failed_build_says_why(monkeypatch, tmp_path):
    monkeypatch.setattr(native, "BUILD", tmp_path / "build")
    monkeypatch.setattr(native, "VCVARS", tmp_path / "vcvars64.bat")
    (tmp_path / "vcvars64.bat").write_text("")
    monkeypatch.setattr(native.subprocess, "run",
                        lambda command, **kw: subprocess.CompletedProcess(command, 2, "native.c(3): error C2143", ""))
    monkeypatch.setattr(native, "_tried", False)
    monkeypatch.setattr(native, "_lib", None)
    assert native.lib() is None and native.why.startswith("cl failed (2)")


def test_turned_off_it_answers_nothing(monkeypatch):
    monkeypatch.setattr(native, "USE", False)
    assert native.lib() is None and "USE" in native.why
    assert native.fly(None, None, 0.0, 0.0, 0.0, 0.0, 0, 0.005, 4.0, None, -1) is None
    assert native.peaks(None, None, 1, 1, 0.1, 0.35) is None
