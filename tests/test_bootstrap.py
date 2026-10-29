"""Her stats from perception (behavior.bootstrap): her max HP off her HUD's HP number, her level off its "LV." digits
(else her EXP number), her DEF off her stat window (else fitted to her touches, else the prior), each said with where it
came from; and her safety stop's reserve worked out again from them whenever one changes (Hunter._reserve_check): a map
change, a level-up, a DEF change, a new kind seen, a bigger touch measured. The user, 2026-10-28: "I thought hp and
level will come from perception, please remove them completely from config.toml", and "please make sure the '3 hits'
is dynamically calculated from the map's strongest mob, not calculated once and hard-coded"."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maplehelper.config import GameKeys

from gamelab.attack.danger import TOP_U, touch_range, touch_taken
from gamelab.behavior import bootstrap
from gamelab.behavior.bootstrap import (FIT_AGREE, FIT_MIN, FIT_N, PLAN_DEF, PRIOR_DEF, UNREAD_LEVEL, UNREAD_SHARE,
                                        Defence, HerStats, fit_defence)
from gamelab.behavior.hunter import Hunter
from gamelab.behavior.safety import MARGIN, Touches, mob_touch
from gamelab.localization import mapdata
from gamelab.localization.tracking import Event
from gamelab.perception import hp_number, lv_number
from gamelab.perception.eyes import Observation
from gamelab.screen import STRETCHED

FIXTURES = Path(__file__).parent / "fixtures"
EMPTY = {"info": {"VRLeft": 0, "VRTop": 0, "VRRight": 2000, "VRBottom": 1000},
         "miniMap": {"centerX": 0, "centerY": 0, "mag": 4}}
TREE_MOBS = {"19": "Horny Mushroom", "13": "Green Mushroom", "7": "Slime"}
TREE_PADS = {"19": 110, "13": 54, "7": 13}
ROCKY_MOBS = {"15": "Axe Stump", "6": "Dark Stump", "5": "Stump"}
ROCKY_PADS = {"15": 75, "6": 34, "5": 11}


def _map(map_id=10002073, mobs=TREE_MOBS, pads=TREE_PADS, name="Tree"):
    return mapdata.parse(map_id, EMPTY, name, {}, dict(mobs), mob_damage=dict(pads))


def _rows():
    rows = Touches(FIXTURES / "touches_2026-10-28.jsonl").rows
    return [r for r in rows if "hp" in r]


def _top(pad, defence, level):
    return touch_range(pad, defence, level)[1]


def test_her_def_is_fitted_to_her_touches_as_the_game_works_a_touch_out():
    """Her 178 exact touches of 2026-10-24 to 28 (her HUD's HP number off the 17 recordings) fit the game's formula
    at one DEF a gear: at level 26 61.5-70.25, at 28 106.5-115.25 (new gear between); each fit's low end is the
    safety stop's (the most a touch takes of what they allow), its middle the positioning model's where the prior
    (PLAN_DEF, 75) isn't among them."""
    rows = _rows()
    assert len(rows) == 178 and (FIT_MIN, FIT_N, FIT_AGREE) == (8, 40, 0.8)
    by_level = {L: fit_defence([r for r in rows if r["level"] == L]) for L in (24, 25, 26, 28)}
    assert [(f.low, f.high, f.agree, f.n) for f in by_level.values()] == [
        (48.0, 59.5, 15, 15), (59.75, 64.75, 40, 40), (61.5, 70.25, 35, 35), (106.5, 115.25, 40, 40)]
    assert by_level[28].best == 111.0 and by_level[28].source == "touches"
    assert by_level[26].best == 66.0  # (75 isn't among what they allow either: their middle)
    # Her last FIT_N, whatever level: today's gear.
    assert fit_defence(rows) == by_level[28]
    # At its low end a Horny Mushroom's top is under her largest measured touch there with MARGIN: the measured counts.
    top = _top(110, by_level[28].low, 28)
    assert top == 123 and max(r["hp"] for r in rows if r["level"] == 28 and r["hp"] > 90) == 122
    assert -(-122 * (1 + MARGIN) // 1) == 129 > top
    assert by_level[28].words() == "DEF 106.5-115.25 (fitted to 40 of her last 40 touches)"


def test_too_few_touches_or_none_agreeing_leave_the_prior_and_say_why():
    rows = _rows()
    few = fit_defence(rows[:FIT_MIN - 1])
    assert (few.source, few.low, few.best, few.n) == ("prior", PRIOR_DEF, PLAN_DEF, FIT_MIN - 1) and PRIOR_DEF == 0
    assert few.words() == ("DEF not known (7 of the 8 touches to fit it kept, no stat window read): a touch at DEF 0, "
                           "the most it takes")
    # Her gear changed mid-way (half her touches at DEF 64, half at 111, all at level 28): no one DEF fits 80%.
    early = [dict(r, level=28) for r in rows if r["level"] == 25][:20]
    late = [r for r in rows if r["level"] == 28][:20]
    mixed = fit_defence(early + late)
    assert mixed.source == "prior" and "don't agree on one" in mixed.why and mixed.agree < FIT_AGREE * mixed.n
    # A touch of no level (taken before her level was read), or on a map whose monsters' PADamage wasn't kept, isn't
    # fitted.
    assert not bootstrap.usable({"hp": 100, "level": None, "pads": [110]})
    assert not bootstrap.usable({"hp": 100, "level": 28})


def test_her_stat_window_is_over_her_touches_and_kept(tmp_path):
    path = tmp_path / "her_stats.json"
    rows = [r for r in _rows() if r["level"] == 28]
    st = HerStats(path, rows)
    assert st.defence.source == "touches" and st.defence.low == 106.5
    note = st.see_window(111, at=1000.0, rows=rows)
    assert st.defence == Defence(111.0, 111.0, 111.0, "stat window", 40, 40, 1000.0)
    assert note.startswith("her DEF 111 (her stat window's W.DEF, read ")
    # Kept: the next run's starts from it, checked against her touches (most of DEF 64's don't fit it: said).
    again = HerStats(path, [r for r in _rows() if r["level"] == 25])
    assert again.defence.source == "stat window" and again.defence.low == 111.0
    assert again.defence.agree < FIT_AGREE * again.defence.n
    kept = json.loads(path.read_text(encoding="utf-8"))
    assert kept["window"] == {"low": 111.0, "at": 1000.0} and kept["defence"]["source"] == "stat window"
    note = HerStats(path).see_window(64, rows=[r for r in _rows() if r["level"] == 28])
    assert "of her last 40 touches fit it (her measured touches raise her reserve past it)" in note


def test_her_level_is_her_digits_else_her_exp_numbers():
    st = HerStats()
    assert st.level is None and st.touch_level == UNREAD_LEVEL == 1
    assert st.see_level(27, HerStats.EXP_FROM) == "her level 27 by her EXP number"
    assert st.see_level(28, HerStats.LV_FROM) == "her level 28 by her HUD's LV. digits (was 27)"
    assert st.see_level(27, HerStats.EXP_FROM) == "her EXP number says level 27, her HUD's LV. digits 28: the digits' taken"
    assert st.see_level(27, HerStats.EXP_FROM) is None and st.level == 28  # (said once)
    assert st.see_level(28, HerStats.LV_FROM) is None and st.see_max_hp(840) == "her max HP 840 by her HUD's HP number"
    assert st.see_max_hp(862) == "her max HP 862 by her HUD's HP number (was 840)" and st.see_max_hp(None) is None
    # The positioning model's her: as read; before her max HP is, PLAN_TOUCHES of the map's largest touch.
    assert st.her().hp == 862 and HerStats().her(120).hp == bootstrap.PLAN_TOUCHES * 120


def _look(t, hp=1.0, number=None, level=None):
    obs = Observation(t=t, frame=np.zeros((10, 10, 3), np.uint8), tag=None, hp=hp, mp=1.0,
                      exp=np.zeros((26, 140), bool))
    if number is not None:
        obs.hp_box = hp_number.draw(hp_number.shown(*number), STRETCHED)
    if level is not None:
        obs.lv_box = lv_number.draw(level, STRETCHED)
    return obs


class _Eyes:
    want_ropes = False

    def __init__(self, looks):
        self.looks = iter(looks)

    def look(self):
        return next(self.looks)


def _hunter(looks=(), model=None, log=None, feed=None):
    return Hunter(_Eyes(looks), GameKeys.__new__(GameKeys), model or _map(),
                  log=(log.append if log is not None else (lambda m: None)), feed=feed)


def test_the_bootstrap_order_and_its_log():
    """Nothing read at the start: the reserve a share of her bar, her level and DEF the priors, each said; her first
    look's HP number and LV. digits make it HP, at her level (the formula's top at DEF 0 until her DEF is known); none of
    it config.toml's."""
    said = []
    h = _hunter([_look(1.0, 1.0, (774, 774), 25), _look(1.1, 1.0, (774, 774), 25)], log=said)
    assert h.max_hp is None and h.level is None and h.stats.defence.source == "prior"
    assert said[0] == ("safety stop: under 50% of her HP bar (her max HP not read yet); at once under 17% of her bar "
                       "two looks in a row; her max HP not read yet (her HUD's HP number); level not read yet (a touch "
                       "at level 1, the most it takes); DEF not known (0 of the 8 touches to fit it kept, no stat "
                       "window read): a touch at DEF 0, the most it takes")
    assert h.hp_watch.threshold() == UNREAD_SHARE
    h.look(mobs=False)
    want = 3 * _top(110, 0.0, 25)
    assert h.max_hp == 774 and h.level == 25 and h.hp_watch.threshold() == want
    assert said[-1] == (f"  the safety stop now under {want} HP of 774 ({want / 774:.0%}: 3 Horny Mushroom touches of "
                        f"{want // 3}) (was 50% of her HP bar (her max HP not read yet)): her max HP 774 by her HUD's HP "
                        "number; her level 25 by her HUD's LV. digits")
    n = len(said)
    h.look(mobs=False)  # nothing new: nothing said, nothing made again
    assert len(said) == n
    # The positioning model's her is the read one, her DEF the prior's best.
    assert (h.her.hp, h.her.level, h.her.defence) == (774.0, 25.0, PLAN_DEF)


def test_a_hunt_started_from_a_running_perception_service_reads_them_at_once():
    """Live, the perception service has been reading her HUD before the command starts: its newest state's hp_of and
    level_of are hers from the first line."""
    newest = SimpleNamespace(hp_of=(700, 840), level_of=28)
    said = []
    _hunter(log=said, feed=SimpleNamespace(newest=lambda after: newest, throw=lambda *a: None))
    assert said[0].startswith(f"safety stop: under {3 * _top(110, 0.0, 28)} HP of 840 (")
    assert "her max HP 840 (her HUD's HP number); level 28 (her HUD's LV. digits)" in said[0]


def test_the_reserve_is_made_again_at_a_map_change_a_level_up_a_def_change_a_new_kind_and_a_bigger_touch():
    said = []
    h = _hunter([_look(1.0, 1.0, (840, 840), 28), _look(2.0, 1.0, (840, 840), 29), _look(3.0, 1.0, (840, 840), 31)],
                log=said)
    h.look(mobs=False)
    assert h.hp_watch.threshold() == 3 * _top(110, 0.0, 28)
    # A level-up (her LV. digits): the formula at it (a level takes a tenth of an HP off a touch: the same whole HP
    # here, said so), and at 31 a whole HP less.
    h.look(mobs=False)
    assert h.level == 29 and h.hp_watch.reserve.level == 29 and h.hp_watch.threshold() == 3 * _top(110, 0.0, 29)
    assert said[-1].startswith("  her level 29 by her HUD's LV. digits (was 28) (the safety stop as it was: under ")
    h.look(mobs=False)
    assert h.hp_watch.threshold() == 3 * _top(110, 0.0, 31) < 3 * _top(110, 0.0, 29)
    assert said[-1].startswith("  the safety stop now under ") and said[-1].endswith(
        "her level 31 by her HUD's LV. digits (was 29)")
    # Her DEF read off her stat window: the formula at it.
    h._notes.append(h.stats.see_window(111, rows=h.touches.rows))
    h._reserve_check()
    assert h.hp_watch.threshold() == 3 * _top(110, 111.0, 31) and "her DEF 111 (her stat window's" in said[-1]
    # A bigger touch measured (her HP number's drop, the game hitting harder than the formula at her W.DEF says):
    # MARGIN over it, from the first one.
    assert _top(110, 111.0, 31) == 122
    h._hp_number(SimpleNamespace(events=[Event("hp_read", 3.0, data={"cur": 840, "max": 840, "before": None}),
                                         Event("hp_read", 3.5, data={"cur": 710, "max": 840, "before": 840})],
                                 hp_of=(710, 840), hp_box=None))
    h._reserve_check()
    assert h.hp_watch.reserve.touch.measured == 130 and h.hp_watch.threshold() == 3 * 137
    assert said[-1].startswith("  the safety stop now under 411 HP of 840") and "a touch of 130 HP (840 -> 710" in said[-1]
    # A kind seen on the map that its spawn list hasn't (the game data's monster table gives its PADamage).
    if (big := mob_touch("zombie-mushroom")) is None:
        pytest.skip("needs the game data's monster table (gamedata/data/out/tables)")
    assert big == (204.0, "Zombie Mushroom")  # (the largest of its name's: 124 at level 24, 204 at 30)
    h._kinds_on(SimpleNamespace(monsters=[SimpleNamespace(kind="zombie-mushroom", x=300.0)]))
    h._reserve_check()
    zombie = 3 * _top(big[0], 111.0, 31)
    assert "zombie-mushroom" in h.kinds_seen and h.hp_watch.threshold() == max(zombie, 3 * 137)
    assert "a Zombie Mushroom seen on the map at x 300 (not in its spawn list; PADamage" in said[-1]
    # Another map: its own kinds (and those seen on it are this map's: a new hunter's).
    h.model, h.kinds_seen = _map(10004021, ROCKY_MOBS, ROCKY_PADS, "Rocky"), {}
    h._reserve_check()
    assert h.hp_watch.threshold() == 3 * _top(75, 111.0, 31) and said[-1].endswith(": the map (Rocky), the kinds counted")


def _touch(h, t, before, cur, most=840):
    """Her HP number's drop from `before` to `cur` at t (a touch, as the perception service's events say it), and the
    look's reserve check."""
    h._hp_number(SimpleNamespace(events=[Event("hp_read", t, data={"cur": before, "max": most, "before": None}),
                                         Event("hp_read", t + 0.5, data={"cur": cur, "max": most, "before": before})],
                                 hp_of=(cur, most), hp_box=None))
    h._reserve_check()


def test_a_level_up_keeps_what_the_game_hit_her_over_the_formula():
    """The review of 2026-10-28: at level 31 (DEF 111 by her stat window) a measured 130 over the formula's 122 raised
    her reserve to 411; her LV. digits' 32 dropped it back to 366 until a touch of that kind was measured again. Now her
    last CARRY_N touches count at another level, carried by the formula's level term (a tenth of an HP a level)."""
    from gamelab.behavior.safety import CARRY_N, carried
    said = []
    h = _hunter([_look(1.0, 1.0, (840, 840), 31), _look(9.0, 1.0, (840, 840), 32)], log=said)
    h.look(mobs=False)
    h._notes.append(h.stats.see_window(111, rows=h.touches.rows))
    h._reserve_check()
    assert h.hp_watch.threshold() == 3 * _top(110, 111.0, 31) == 366
    _touch(h, 3.0, 840, 710)  # the game hitting harder than the formula: 130 over its 122
    assert h.hp_watch.threshold() == 3 * 137 == 411
    h.look(mobs=False)  # her LV. digits: 32 (the same monster, the same gear)
    assert h.level == 32 and h.hp_watch.threshold() == 411
    t = h.hp_watch.reserve.touch
    assert (t.measured, t.taken_at, t.prior) == (130, 31, _top(110, 111.0, 32))
    assert "taken at level 31, carried by the formula's level term" in h.hp_watch.reserve.describe(840)
    assert said[-1].startswith("  her level 32 by her HUD's LV. digits (was 31) (the safety stop as it was: under 411")
    # The level term: a whole HP off for every 10 levels up, one more for a level down (the most it can be there).
    assert [carried(130, 31, lv) for lv in (31, 32, 40, 41, 51, 30)] == [130, 130, 130, 129, 128, 131]
    # One of another level older than her last CARRY_N touches goes (another gear, as her DEF's fit has it); hers at
    # her level count however old.
    kept = Touches(None)
    kept.rows = ([{"level": 31, "hp": 130, "kind": "horny-mushroom"}, {"level": 32, "hp": 118, "kind": "horny-mushroom"}]
                 + [{"level": 32, "hp": 50, "kind": "green-mushroom"}] * CARRY_N)
    assert kept.measured(32)["horny-mushroom"] == (118, 1, 32) and kept.measured(31)["horny-mushroom"] == (130, 1, 31)


def test_her_touches_file_seeded_with_the_fixture_starts_her_at_the_fit_and_her_measured_touch():
    """The merge note's seed (tests/fixtures/touches_2026-10-28.jsonl copied to lab/touches.jsonl, her gear as on
    10-28): at 840 and level 28 her DEF the fit's (106.5-115.25), her largest measured Horny Mushroom touch there (122)
    over the formula's top at its low end (123 less the margin): 3 of 129, 387. The review found 369: the fixture's rows
    had no kind, and a row of no kind counts for none; they're tagged now as the hunter tags a touch live."""
    import shutil
    from gamelab import paths
    shutil.copy(FIXTURES / "touches_2026-10-28.jsonl", paths.TOUCHES)
    said = []
    newest = SimpleNamespace(hp_of=(840, 840), level_of=28)
    h = _hunter(log=said, feed=SimpleNamespace(newest=lambda after: newest, throw=lambda *a: None))
    t = h.hp_watch.reserve.touch
    assert (h.stats.defence.low, h.stats.defence.high) == (106.5, 115.25)
    assert (t.kind, t.measured, t.n, t.hp) == ("horny-mushroom", 122, 38, 129) and h.hp_watch.threshold() == 387
    assert said[0].startswith("safety stop: under 387 HP of 840 (46%: 3 Horny Mushroom touches of 129); Horny Mushroom "
                              "129: 38 measured at level 28, the largest 122, 5% over it")
    # Each row's kind: Horny Mushroom 117-133 at level 25 (17), 103-122 at 28 (38), the touch and potion of 16 none's.
    rows = _rows()
    horny = {L: sorted(r["hp"] for r in rows if r["level"] == L and r["kind"] == "horny-mushroom") for L in (25, 28)}
    assert (len(horny[25]), horny[25][0], horny[25][-1]) == (17, 117, 133)
    assert (len(horny[28]), horny[28][0], horny[28][-1]) == (38, 103, 122)
    assert [r["hp"] for r in rows if r["kind"] is None] == [16]


def test_her_stats_file_keeps_her_max_hp_and_level_as_last_read(tmp_path):
    """The review of 2026-10-28: every HerStats() wrote her stats file's max HP and level as null (a hunter's start,
    `gamelab stats` itself), and an offline `patrol` had nothing to fall back on. A run that hasn't read them keeps
    them as the file had them, with when they were read; one that reads them writes its own."""
    path = tmp_path / "her_stats.json"
    st = HerStats(path)
    st.see_max_hp(840)
    st.see_level(28, HerStats.LV_FROM)
    kept = json.loads(path.read_text(encoding="utf-8"))
    assert (kept["max_hp"], kept["level"], kept["level_from"]) == (840, 28, HerStats.LV_FROM)
    read_at = (kept["max_hp_at"], kept["level_at"])
    again = HerStats(path, [r for r in _rows() if r["level"] == 28])  # (a fit to save: the file written again)
    assert again.max_hp is None and again.level is None  # (this run's: not read yet; the file's aren't hers now)
    kept = json.loads(path.read_text(encoding="utf-8"))
    assert (kept["max_hp"], kept["level"], (kept["max_hp_at"], kept["level_at"])) == (840, 28, read_at)
    assert kept["defence"]["source"] == "touches"
    again.see_max_hp(862)
    kept = json.loads(path.read_text(encoding="utf-8"))
    assert (kept["max_hp"], kept["level"]) == (862, 28) and kept["max_hp_at"] >= read_at[0]


def test_her_level_is_her_exp_numbers_once_her_digits_stop_reading():
    """The review of 2026-10-28: once her LV. digits had read, her EXP number's level was ignored for the rest of the
    run, the digits read or not, and their disagreement said once ever. Now her EXP number's is taken once the digits
    haven't read for LV_UNREAD_S (said), the digits' again when they read; a disagreement said once for each pair."""
    from gamelab.behavior.bootstrap import LV_UNREAD_S
    st = HerStats()
    assert st.see_level(28, HerStats.LV_FROM, 0.0) is not None
    assert st.see_level(None, HerStats.LV_FROM, 1.0) is None  # (the digits not read since 1.0)
    assert st.see_level(29, HerStats.EXP_FROM, 1.5) == ("her EXP number says level 29, her HUD's LV. digits 28: the "
                                                        "digits' taken")
    assert st.see_level(29, HerStats.EXP_FROM, 2.0) is None and st.level == 28  # (said once for the pair)
    assert st.see_level(None, HerStats.LV_FROM, 2.5) is None  # (still not read: since 1.0)
    assert st.see_level(29, HerStats.EXP_FROM, 1.0 + LV_UNREAD_S) == (
        f"her level 29 by her EXP number (her HUD's LV. digits not read for {LV_UNREAD_S:g} s or more; they said 28)")
    assert st.level == 29 and st.level_from == HerStats.EXP_FROM
    assert st.see_level(30, HerStats.EXP_FROM, 3.5) == "her level 30 by her EXP number (was 29)"
    # The digits again: theirs, and another disagreement is said.
    assert st.see_level(30, HerStats.LV_FROM, 4.0) is None and st.level_from == HerStats.LV_FROM
    assert st.see_level(31, HerStats.EXP_FROM, 4.1) == ("her EXP number says level 31, her HUD's LV. digits 30: the "
                                                        "digits' taken")
    # A digit read missed for a look (a frame failing its checks) doesn't hand it to the EXP number.
    assert st.see_level(None, HerStats.LV_FROM, 5.0) is None and st.see_level(30, HerStats.LV_FROM, 5.1) is None
    assert st.see_level(31, HerStats.EXP_FROM, 5.1 + LV_UNREAD_S) is None and st.level == 30


def test_a_def_that_drops_is_followed_by_her_newest_touches():
    """The review of 2026-10-28, her 178 touches in reverse (DEF about 111 at level 28, then about 64 at 26: gear taken
    off): the fit of her last 40 stayed at 105.75 for 9 touches after the drop, then 91.25, 77.5, 73.75, reaching 69.25
    about 32 in, a Horny Mushroom's top 124-131 meanwhile against the 134 she could take. Now her newest 8 not fitting
    it take the prior (DEF 0, said) until they agree on a DEF, then the lower of the two lows."""
    rows = list(reversed(_rows()))
    drop = next(i for i, r in enumerate(rows) if r["level"] == 26)  # (the first touch at the lower DEF)
    assert fit_defence(rows[:drop]).low == 105.75
    after = [fit_defence(rows[:n]) for n in range(drop + 1, drop + 23)]
    assert after[0].source == "prior" and "her newest 8 touches fit neither the DEF her last 40 do" in after[0].why
    assert all(f.low <= 70.0 for f in after)  # (the level's own fit 61.5-70.25)
    assert after[4].source == "touches" and "her newest 8 fit" in after[4].words() and "the lower taken" in after[4].why
    # In time order (DEF going up: new gear) her last 40 at level 28 are as before, nothing said of her newest.
    assert fit_defence(list(reversed(rows))).why == ""


def test_a_hunting_command_waits_for_her_max_hp_and_hunts_nothing_without_it(monkeypatch):
    """The review of 2026-10-28: before her max HP read the stop was 50% of her bar on every map (under 3 Horny Mushroom
    touches at levels 24-26, about 2 on a map whose largest is PADamage 150) and no TooStrong could be told. A command
    that hunts now waits HP_WAIT_S for it (the perception service's newest states) and hunts nothing without it
    (HpNotRead), unless --low-hp gives a share of her bar."""
    from gamelab import __main__ as cli
    from gamelab.behavior import hunter as hunter_mod
    from gamelab.behavior.safety import EXIT, HP_WAIT_S, HpNotRead
    # The hunter: the feed's states without her HP number, then with it.
    states = iter([SimpleNamespace(hp_of=None, level_of=None)] * 3 + [SimpleNamespace(hp_of=(700, 840), level_of=28)])
    last = [None]

    def newest(after):
        last[0] = next(states, last[0])
        return last[0]
    said = []
    h = _hunter(log=said, feed=SimpleNamespace(newest=newest, throw=lambda *a: None))
    assert h.max_hp is None and h.wait_max_hp(seconds=5.0, poll=0.001) and h.max_hp == 840 and h.level == 28
    assert said[-1].startswith(f"  the safety stop now under {3 * _top(110, 0.0, 28)} HP of 840 (") and (
        "(was 50% of her HP bar (her max HP not read yet)): her max HP 840 by her HUD's HP number; her level 28" in said[-1])
    never = _hunter(feed=SimpleNamespace(newest=lambda after: SimpleNamespace(hp_of=None, level_of=28),
                                         throw=lambda *a: None))
    assert not never.wait_max_hp(seconds=0.02, poll=0.001)
    assert "the perception service's states carry no HP number" in never.hp_why
    stop = HpNotRead(HP_WAIT_S, never.hp_why)
    assert stop.exit_code == EXIT and stop.after == "Nothing was hunted." and str(stop).startswith(
        "her max HP not read off her HUD's HP number in 5 s (the perception service's states carry no HP number")
    # The command line: `tour --loop` refused; with --low-hp a share hunts, in HP it waits too.
    from gamelab.localization import mapdata, service
    from test_worldstate import small_map

    class Client:
        def __init__(self, name, **kw):
            self.welcome, self.why = {"pid": 1, "epoch": 1, "map": kw["map_id"]}, ""

        def ready(self, timeout):
            return True
    waited = []
    monkeypatch.setattr(service, "ServiceClient", Client)
    monkeypatch.setattr(cli, "_eyes", lambda model, title, perception, restore: SimpleNamespace(log=print,
                                                                                               geo=STRETCHED))
    monkeypatch.setattr(cli, "_fit_start", lambda title, fix: None)
    monkeypatch.setattr(mapdata, "load", lambda map_id: small_map())
    monkeypatch.setattr(hunter_mod, "Hunter", lambda eyes, keys, model, log, feed, **kw: SimpleNamespace(
        wait_max_hp=lambda: waited.append(1) or False, hp_why="why", too_strong=lambda: None,
        senses=SimpleNamespace(window_fix=None)))
    try:
        with pytest.raises(HpNotRead):
            cli._hunter(cli.build_parser().parse_args(["tour", "--loop"]))
        cli._hunter(cli.build_parser().parse_args(["tour", "--loop", "--low-hp", "0.5"]))
        assert len(waited) == 1
        with pytest.raises(HpNotRead):
            cli._hunter(cli.build_parser().parse_args(["tour", "--loop", "--low-hp", "400"]))
        cli._hunter(cli.build_parser().parse_args(["tour", "--loop"]), throws=False)  # (the pet: it hunts nothing)
        assert len(waited) == 2
    finally:
        cli._PERCEPTION.clear()


def test_no_constant_stands_for_her_touch():
    """The reserve is the formula's at the read level and DEF, nothing written down: the same map at two levels and two
    DEFs gives four thresholds, each the formula's."""
    from gamelab import paths
    for level, d in ((25, 0.0), (25, 111.0), (28, 0.0), (28, 111.0)):
        paths.STATS.unlink(missing_ok=True)  # (a stat window's DEF is kept: each its own run)
        h = _hunter()
        h.stats.see_max_hp(840)
        h.stats.see_level(level, HerStats.LV_FROM)
        if d:
            h.stats.see_window(d)
        h._reserve_check()
        assert h.hp_watch.threshold() == 3 * touch_taken(110, d, level, TOP_U)
