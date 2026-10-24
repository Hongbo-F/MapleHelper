"""Feeding her pet (it drinks her potions for her since 2026-10-23): when it's due, and the plan's name."""

import json
from types import SimpleNamespace

from maplehelper import keys

from gamelab import paths
from gamelab.behavior import hunter as hunter_mod
from gamelab.behavior.hunter import Hunter
from gamelab.control import client


class _Actions:
    def __init__(self, pet_food):
        self.keys = SimpleNamespace(pet_food=pet_food)
        self.fed = 0

    def feed_pet(self):
        self.fed += 1
        return True


def _hunter(pet_food):
    h = Hunter.__new__(Hunter)  # only what feed_pet_if_due uses
    h.actions, h.pet_feed_s, h.log = _Actions(pet_food), hunter_mod.PET_FEED_S, lambda message: None
    return h


def test_she_feeds_her_pet_when_it_is_due_and_not_again_until_the_next_time(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PET", tmp_path / "pet.json")
    now = [1_000_000.0]
    monkeypatch.setattr(hunter_mod.time, "time", lambda: now[0])
    monkeypatch.setattr(hunter_mod.telemetry, "emit", lambda *a, **k: None)
    h = _hunter(keys.parse("u"))
    assert h.feed_pet_if_due() and h.actions.fed == 1  # never fed yet: now
    assert json.loads(paths.PET.read_text())["fed"] == now[0]
    now[0] += hunter_mod.PET_FEED_S - 1
    assert not h.feed_pet_if_due() and h.actions.fed == 1
    now[0] += 1
    assert h.feed_pet_if_due() and h.actions.fed == 2


def test_no_pet_food_key_means_no_feeding(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PET", tmp_path / "pet.json")
    h = _hunter(None)
    assert not h.feed_pet_if_due() and h.actions.fed == 0 and not paths.PET.exists()


def test_a_plan_name_keeps_only_what_the_player_keeps(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "INBOX", tmp_path)
    plan_id = client.send([], name="pet food")
    assert "_petfood_" in plan_id and (tmp_path / f"{plan_id}.json").exists()
