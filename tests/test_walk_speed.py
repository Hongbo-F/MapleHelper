"""Her walk as planned against her walk as measured (ROADMAP item 1 of the recordings of 2026-11-02, P01): this client
walks a character at Physics.img's walkSpeed 125 x (SPEED + 20)/100 (motion.body.SPEED_BONUS), SPEED her stat window's.
The source: meowdb's guide "Speed, Jump and Movement" (meowdb.com/msclassic/guides/speed-jump-and-movement, Nia Meow,
updated 2026-10-27, measured off the client's video frame by frame): "Walking speed is 125 × (Speed + 20) / 100 pixels
per second", "At the base Speed of 100, your character walks 150 pixels per second"; its formulas page and calculator
the same. The measurements: tests/fixtures/walks_tree_2026.json, both characters' steady walks on Tree Dungeon (1 unit
1 px at 1366x768), each walk's braking tail left out and Tree's slopes taken out by the v83 slope rule (README "Her walk
speed"). The magician, planned at 125 until 2026-11-02 (v83's 125 x SPEED/100 on her stat window's 100), walked 148.2
raw (81 steady walks of the hour 00:17:48-00:53:06: plans/motion-work/profile_checks.py, walkfit).
"""

import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

from maplehelper import config

from gamelab import character
from gamelab.behavior import statcheck
from gamelab.classes import magician, thief
from gamelab.motion import body, physics

WALKS = json.loads((Path(__file__).parent / "fixtures" / "walks_tree_2026.json").read_text(encoding="utf-8"))
HER_RAW = 148.2         # the magician's raw steady walk, 81 walks of the hour (se 0.4): the analysis', walkfit's
THIEF_BAND = (148.0, 151.5)  # the thief's measured walks: 148-151 at 1920x1080 (config.Character's old comment), 151.2
                             # on Tree Dungeon at 1366x768 (the fixture's 243)


def _limits(who: str) -> list[float]:
    return [row[-1] for row in WALKS["characters"][who]["walks"]]


def test_her_planned_walk_is_her_measured_one():
    """The magician's profile (her stat window's SPEED 100) plans her at 150 units/s, 1.2% over her raw 148.2 and 0.4%
    over her 149.4 with the braking tails and slopes out; v83's formula planned 125, 16% under (her takeoffs came late
    by 0.19 x the walk before them, 15 of the hour's 210 'arrived' trips walked off their platform's end)."""
    planned = physics.shoe_of(magician.FIXED_STATS).vmax
    assert magician.FIXED_STATS.speed == 100.0 and planned == 150.0
    assert planned == pytest.approx(HER_RAW, rel=0.015)
    assert planned == pytest.approx(statistics.median(_limits("magician")), rel=0.005)
    v83 = body.Shoe.from_physics(body.PHYSICS_IMG, magician.FIXED_STATS.speed, magician.FIXED_STATS.jump).vmax
    assert v83 == 125.0 and HER_RAW / v83 > 1.15


def test_the_thiefs_planned_walk_is_unchanged_and_within_hers_measured():
    """The thief's profile walks her at 149.0 as before the formula (99.2 + 20 = 119.2 exactly: her goldens and the
    harness's baselines unchanged), inside her measured 148-151.5 and 1.5% under her 151.2 on Tree Dungeon, which her
    stat window's 101 gives (151.25: the move to it is for the simulator's next re-baselining)."""
    planned = physics.shoe_of(thief.FIXED_STATS).vmax
    assert thief.FIXED_STATS == config.Character() and planned == 149.0 == physics.WALK_SPEED  # (the tests play her)
    assert THIEF_BAND[0] <= planned <= THIEF_BAND[1]
    assert planned == pytest.approx(statistics.median(_limits("thief")), rel=0.015)
    window = physics.shoe_of(config.Character(speed=WALKS["characters"]["thief"]["stat_window_speed"])).vmax
    assert window == 151.25 and window == pytest.approx(statistics.median(_limits("thief")), rel=0.005)


@pytest.mark.parametrize("who", ["magician", "thief"])
def test_the_recorded_walks_are_the_clients_formula_not_v83s(who):
    """Each character's measured walk (the fixture's walk limits) within 1% of 125 x (stat window SPEED + 20)/100 and
    more than 15% off v83's 125 x SPEED/100: the magician 149.44 (66 walks) against 150 and 125, the thief 151.19 (243)
    against 151.25 and 126.25."""
    c = WALKS["characters"][who]
    got = statistics.median(_limits(who))
    assert got == c["summary"]["limit_median"] and len(c["walks"]) == c["summary"]["n"]
    speed = c["stat_window_speed"]
    client = body.Shoe.from_physics(body.PHYSICS_IMG, speed, 100, bonus=body.SPEED_BONUS).vmax
    v83 = body.Shoe.from_physics(body.PHYSICS_IMG, speed, 100).vmax
    assert got == pytest.approx(client, rel=0.01)
    assert got / v83 > 1.15
    # ... and the raw speeds, slopes and braking tails in: still 1% (the walk is the limit: no overshoot on the step)
    assert statistics.median(row[5] for row in c["walks"]) == pytest.approx(client, rel=0.01)


def test_her_stat_window_said_only_where_it_isnt_her_walk():
    """The hunter's note on her stat window's SPEED and JUMP (statcheck.moves_notes): before 2026-11-02 it compared the
    window's SPEED with config.toml's fitted 119.2 and said so at every read for the thief. Now like for like: the
    magician's 100 against her profile's 100 says nothing; the thief's 101 against her profile's 99.2 says the walk
    each gives; a JUMP apart too."""
    reading = lambda **v: SimpleNamespace(get=v.get)
    where = lambda stat: "her profile"
    assert statcheck.moves_notes(reading(speed=100, jump=100), magician.FIXED_STATS, where) == []
    assert statcheck.moves_notes(reading(speed=99, jump=100), thief.FIXED_STATS, where) == []  # (under a point)
    assert statcheck.moves_notes(reading(speed=101, jump=100), thief.FIXED_STATS, where) == [
        "her SPEED 101% by her stat window walks her at 151.25 units/s; her moves are planned at 149 (her profile "
        "speed 99.2)"]
    assert statcheck.moves_notes(reading(speed=None, jump=110), magician.FIXED_STATS, where) == [
        "her JUMP 110% by her stat window jumps her at 610.5 units/s; her moves are planned at 555 (her profile jump "
        "100)"]


def test_where_her_stats_come_from(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[characters.magician]\nclass = \"magician\"\n[characters.magician.stats]\njump = 105\n",
                   encoding="utf-8")
    assert character.stats_from("speed", cfg, {character.FIXED_STATS: "1"}) == \
        "her class's fixed stats (GAMELAB_FIXED_STATS)"
    with character.using(character.load("thief", cfg, {})):
        assert character.stats_from("speed", cfg, {}) == "config.toml [character]"
    with character.using(character.load("magician", cfg, {})):
        assert character.stats_from("speed", cfg, {}) == "her profile (gamelab/classes/magician.py FIXED_STATS)"
        assert character.stats_from("jump", cfg, {}) == "config.toml [characters.magician.stats]"
