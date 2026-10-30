"""A claimed plan's file that Windows refuses for a moment (another program has it open) is read again, not failed
(2026-10-29 17:04:41: a walk's plan answered "PermissionError ... .running" and the grind ended)."""
import json

import pytest

from gamelab.control import player as player_mod


def test_a_plan_refused_a_moment_is_read_again_and_one_refused_throughout_still_raises(tmp_path, monkeypatch):
    path = tmp_path / "170440_walk_a709.running"
    path.write_text(json.dumps({"events": []}), encoding="utf-8")
    real, refused = type(path).read_text, [3]

    def read_text(self, *a, **k):
        if refused[0]:
            refused[0] -= 1
            raise PermissionError(13, "Permission denied", str(self))
        return real(self, *a, **k)
    monkeypatch.setattr(type(path), "read_text", read_text)
    assert player_mod._read_plan(path) == {"events": []} and refused[0] == 0
    refused[0] = 10 ** 9
    monkeypatch.setattr(player_mod, "READ_RETRY_S", 0.05)
    with pytest.raises(PermissionError):
        player_mod._read_plan(path)
