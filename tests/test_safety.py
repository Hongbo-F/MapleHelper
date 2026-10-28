"""The safety stop (behavior.safety): her HP, in HP, under her reserve (her profile's hits of the map's largest touch:
the thief 2, the magician 3) on LOOKS looks DEBOUNCE_S apart, or under one touch FLOOR_LOOKS looks in a row, ends the
hunt; she goes to a safe place as on the Lie Detector (behavior.retreat.on_alarm), the log says so in capitals with the
numbers, whoever is at the PC is paged, and the command exits with its own code. No sitting and no waiting for HP (the
rests went on 2026-10-26). Until 2026-10-28 it was 65% of her HP for 2 s over 3 looks, or 25% two looks in a row (the
tests of that rule went with it: what replaced each is below). In the simulator, end to end: loop_tree_safety_stop
(tests/test_sim.py runs every scenario)."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from gamelab.attack import danger
from gamelab.behavior import retreat, safety
from gamelab.behavior.darts import DartWatch
from gamelab.behavior.hunter import Hunter
from gamelab.behavior.safety import (BAR_SLACK, DEBOUNCE_EXACT_S, DEBOUNCE_S, EXIT, FALLBACK_SHARE, FLOOR_LOOKS,
                                     FLOOR_TOUCHES, HITS, LOOKS, MARGIN, PRIOR_DEF, HpWatch, Reserve, SafetyStop,
                                     TooStrong, Touch, Touches, kind_of_touch, prior_def, reserve_of)

FIXTURES = Path(__file__).parent / "fixtures"
TREE = SimpleNamespace(id=10002073, mob_damage={"Green Mushroom": 54.0, "Horny Mushroom": 110.0, "Slime": 13.0})
ROCKY = SimpleNamespace(id=10004021, mob_damage={"Stump": 11.0, "Red Snail": 9.0, "Axe Stump": 75.0,
                                                 "Dark Stump": 34.0})
NONE = SimpleNamespace(id=1, mob_damage={})


def tree(level=28.0, hits=HITS, defence=75.0, touches=()):
    return reserve_of(TREE, hits, defence, level, touches)


def _watch(looks, reserve=None, max_hp=840.0, **kw):
    """An HpWatch fed `looks` ([t, share, exact HP or None]...); the stop if it came (and at which look), else None."""
    w = HpWatch(reserve or tree(), **kw)
    for i, (t, share, *exact) in enumerate(looks):
        w.see(t, share, max_hp, exact[0] if exact else None)
        if (stop := w.due()) is not None:
            return i, stop
    return None


def test_its_numbers():
    assert (HITS, DEBOUNCE_S, LOOKS, FLOOR_TOUCHES, FLOOR_LOOKS, MARGIN, PRIOR_DEF, FALLBACK_SHARE, EXIT) == \
           (2, 0.6, 2, 1, 2, 0.05, 64.0, 0.35, 4)
    assert BAR_SLACK == 1 / 104  # a px of her HP bar at 1366x768


def test_the_reserve_is_her_hits_of_the_maps_largest_touch_in_hp():
    """The user, 2026-10-28: "it should be HP under the amount of taking 3 hits from the monsters on the map ... for
    thief with high evasion this should be 2 hits, 3 for magician". The touch: the formula's top at her DEF and level
    (attack.danger.touch_range), the largest kind's."""
    r = tree()
    assert [(t.kind, t.name, t.prior, t.hp) for t in r.touches] == [
        ("horny-mushroom", "Horny Mushroom", 133, 133), ("green-mushroom", "Green Mushroom", 60, 60),
        ("slime", "Slime", 9, 9)]
    assert r.hp(840) == 266 and round(266 / 840, 3) == 0.317  # the thief at level 28
    assert tree(hits=3).hp(840) == 399  # a magician
    assert r.words(840) == "2 Horny Mushroom touches of 133"
    assert r.describe(840).startswith("under 266 HP of 840 (32%: 2 Horny Mushroom touches of 133); Horny Mushroom 133: "
                                      "the formula's top at DEF 64, level 28")
    # Absolute HP: her max HP doesn't move it (the simulator's level 25, 774: 268, 34.6%), her level does a little.
    assert tree(level=25).hp(774) == 268
    # Her DEF isn't known (her profile's 75: config.toml has none): the prior's at PRIOR_DEF at the most, the lowest her
    # touches fitted (64, levels 24-26: Horny Mushroom 113-134, she took 132 and 133; at 75 its top is 131). A DEF
    # over it doesn't lower it (111, as fitted at level 28, would give 122); one under it raises it.
    assert prior_def(75) == prior_def(111) == PRIOR_DEF == 64 and prior_def(50) == 50
    assert tree(defence=111).hp(840) == 266 and tree(defence=50).hp(840) == 2 * danger.touch_range(110, 50, 28)[1]
    assert danger.touch_range(110, 75, 25)[1] == 131 < 133 <= danger.touch_range(110, 64, 25)[1] == 134
    assert reserve_of(ROCKY, 2, 75, 28).hp(840) == 174  # an Axe Stump's 87: 21%
    # A map whose monsters' touches the game data doesn't give: FALLBACK_SHARE of her max HP, said so.
    fb = reserve_of(NONE, 2, 75, 28)
    assert fb.touch is None and fb.hp(840) == pytest.approx(294) and fb.floor(840) == pytest.approx(147)
    assert "no game data for the map's monsters' touches" in fb.words(840)


def test_on_her_maps_at_level_28():
    from gamelab.localization import mapdata
    from gamelab.sim import scenarios
    want = {10002073: 266, 10004021: 174, 10000012: 28}  # Tree Dungeon, West Rocky Mountain II, Thicket II
    have = {m: v for m, v in want.items() if scenarios.available(m)}
    if not have:
        pytest.skip("needs the maps' game data (gamedata/data)")
    for map_id, hp in have.items():
        assert reserve_of(mapdata.load(map_id), 2, 75.0, 28).hp(840) == hp, map_id


def test_her_measured_touches_count_from_the_first_at_her_level_with_a_margin():
    """A measured touch only ever raises a kind's touch (the larger of the prior and MARGIN over the largest measured),
    so none is waited for: the review of 2026-10-28 found one over the prior ignored until 10 were kept at her level and
    the hunter started again."""
    def rows(sizes, level=28, kind="horny-mushroom", map_id=TREE.id):
        return [{"level": level, "hp": h, "kind": kind, "map": map_id} for h in sizes]
    # Level 28's live touches (103-122: her DEF about 111 now) are under the formula's top at 64: the prior stays.
    t = tree(touches=rows([110] * 37 + [122])).touch
    assert (t.measured, t.n, t.hp) == (122, 38, 133)
    # At level 25 (DEF about 64) the largest, 133, x 1.05: 140, the thief's threshold 280, from the one.
    r = tree(level=25, touches=rows([133], level=25))
    assert r.touch.hp == math.ceil(1.05 * 133) == 140 and r.hp(774) == 280
    assert "1 measured at level 25, the largest 133, 5% over it" in r.describe(774)
    # Other levels' don't count, nor one at no level (her EXP number not read when it was taken), nor one of no kind
    # (a drop its size fits no kind of the map's by: a stretch not read, a kind the data hasn't).
    assert tree(touches=rows([140] * 20, level=25)).touch.hp == 133
    assert tree(touches=rows([140], level=None)).touch.hp == 133
    assert tree(touches=rows([234] + [140], kind=None)).touch.hp == 133
    # Green Mushroom's measured are its own.
    g = tree(touches=rows([60], kind="green-mushroom"))
    assert [(x.kind, x.hp) for x in g.touches][:2] == [("horny-mushroom", 133), ("green-mushroom", 63)]


def test_a_touchs_kind_is_told_by_its_size():
    # Her DEF isn't known for sure (64 at levels 24-26, 111 at 28): the formula's range over half to twice hers.
    assert [kind_of_touch(h, TREE, 75.0, 28) for h in (103, 122, 133, 46, 62, 11, 16, 200)] == [
        "horny-mushroom", "horny-mushroom", "horny-mushroom", "green-mushroom", "green-mushroom", "slime", None, None]
    # Kinds whose ranges overlap: the larger.
    both = SimpleNamespace(id=2, mob_damage={"Big": 60.0, "Small": 50.0})
    assert kind_of_touch(50, both, 75.0, 28) == "big"


def test_her_touches_are_kept_across_runs(tmp_path):
    path = tmp_path / "touches.jsonl"
    t = Touches(path)
    assert t.rows == []
    t.add({"t": 1.0, "map": TREE.id, "level": 28, "hp": 117, "kind": "horny-mushroom", "max": 840})
    path.write_text(path.read_text(encoding="utf-8") + "not json\n", encoding="utf-8")
    assert Touches(path).rows == t.rows and Touches(None).rows == []


def test_two_looks_under_it_debounce_s_apart_stop_the_hunt():
    # Her bar alone (her HP number not read): a look every 0.25 s under 266 (over the floor, 133): the look 0.6 s after
    # the first stops it (her pet's potion, 0.04-0.20 s after a touch, and the bar's settling, 0.37 s, waited for).
    under = [[k / 4, 0.3] for k in range(8)]
    i, stop = _watch(under)
    assert under[i][0] == 0.75 and stop.seconds == 0.75 and stop.looks == 4 and not stop.exact
    assert stop.hp == pytest.approx((0.3 - BAR_SLACK) * 840) and stop.threshold == 266 and stop.floor is None
    assert stop.max_hp == 840 and DEBOUNCE_S == 0.6
    assert str(stop) == ("HP about 244 of 840, under 2 Horny Mushroom touches of 133 (266) for 0.8 s (4 looks): her "
                         "pet's potions aren't keeping her up (none left? the pet hungry or not out?)")
    title, text = stop.page()
    assert title == "MapleStory: her HP stayed low" and text == (
        "gamelab stopped the hunt: HP about 244 of 840, under 2 Horny Mushroom touches of 133 (266) for 0.8 s. She went "
        "to a safe place. Check her potions and her pet.")
    # Her HP number's: 0.25 s (DEBOUNCE_EXACT_S: the potion only; the number doesn't lag), on two looks at least.
    under = [[k / 10, 0.3, 250] for k in range(8)]
    i, stop = _watch(under)
    assert DEBOUNCE_EXACT_S == 0.25 and under[i][0] == 0.3 and stop.looks == 4 and stop.hp == 250 and stop.exact
    assert str(stop).startswith("HP 250 of 840, under 2 Horny Mushroom touches of 133 (266) for 0.3 s (4 looks)")
    # Two looks that far apart are enough, however few between them.
    assert _watch([[0.0, 0.3], [0.6, 0.3]])[0] == 1 and _watch([[0.0, 0.3, 250], [0.25, 0.3, 250]])[0] == 1
    # Once: what follows it (her way to a safe place) goes on, under the threshold still.
    w = HpWatch(tree())
    for t, share, hp in under:
        w.see(t, share, 840, hp)
    assert w.due() is not None and w.due() is None
    w.see(10.0, 0.1, 840, 50)
    assert w.due() is None and w.fired


def test_her_pets_potion_a_misread_or_a_look_without_hp_doesnt_stop_it():
    # A touch to 240, her pet's potion 0.2 s later (+100: live 0.04-0.20 s after): back over it, the stretch over.
    assert _watch([[0.0, 0.29, 240], [0.1, 0.29, 240], [0.2, 0.4, 340], [0.9, 0.4, 340]]) is None
    # Under it for less than 0.6 s by the bar, however many looks (the bar's settling, 0.37 s at most): not yet; nor
    # 0.25 s by her HP number.
    assert _watch([[0.0, 0.3], [0.2, 0.3], [0.4, 0.3], [0.55, 0.33]]) is None
    assert _watch([[0.0, 0.3, 250], [0.1, 0.3, 250], [0.2, 0.3, 250], [0.24, 0.32, 270]]) is None
    # A look that read nothing neither starts nor ends a stretch.
    w = HpWatch(tree())
    w.see(0.0, 0.3, 840, 250)
    w.see(0.2, None, 840)
    assert w.looks == 1 and w.due() is None
    w.see(0.3, 0.3, 840, 250)
    assert w.due() is not None
    # A frame older than the last one seen (a look's after a glance's) is left out.
    w = HpWatch(tree())
    w.see(1.0, 0.7, 840, 590)
    w.see(0.9, 0.1, 840, 80)
    assert w.hp == 590 and w.at == 1.0 and w.since is None


def test_under_one_touch_two_looks_in_a_row_stop_it_at_once():
    # Out of a gap in her looks (a trip, a loot run) under 133: the next look under it too stops the hunt, the 0.6 s not
    # waited.
    i, stop = _watch([[0.0, 0.7, 590], [3.4, 0.13, 110], [3.45, 0.13, 110]])
    assert i == 2 and stop.floor == 133 and stop.looks == 2 and stop.seconds == pytest.approx(0.05)
    assert ("HP 110 of 840, under 2 Horny Mushroom touches of 133 (266) for 0.1 s (2 looks), under one Horny Mushroom "
            "touch (133) on 2 looks in a row: her pet") in str(stop)
    # One look under it, the next over it (a misread, a potion): not.
    assert _watch([[0.0, 0.1, 100], [0.1, 0.4, 340], [0.2, 0.1, 100], [0.3, 0.7, 590]]) is None
    # Dead (HP 0): it stops too, and says she may be.
    i, stop = _watch([[0.0, 0.0, 0], [0.1, 0.0, 0]])
    assert i == 1 and "she may be dead" in str(stop)


def test_her_hp_is_her_hp_numbers_else_the_bar_less_a_px():
    w = HpWatch(tree())
    w.see(0.0, 0.31, 840, 267)  # the number says 267: over 266, though the bar (lagging it) says 31%
    assert w.hp == 267 and w.since is None
    w.see(0.1, 0.32, 840)  # no number: the bar's 32% less a px, 260.7
    assert w.hp == pytest.approx((0.32 - 1 / 104) * 840) and w.since == 0.1 and not w.exact
    w.see(0.8, 0.32, 840)
    stop = w.due()
    assert stop.her_hp == "HP about 261 of 840" and not stop.exact


def test_low_hp_overrides_it_as_a_share_or_in_hp_and_a_potion_key_of_her_own_caps_it():
    assert HpWatch(tree(), low_hp=0.5).threshold(840) == 420 and HpWatch(tree(), low_hp=300).threshold(840) == 300
    assert HpWatch(tree(), low_hp=0.5).words(840) == "--low-hp's 50% of 840"
    # With --low-hp under one touch the floor is the threshold.
    assert HpWatch(tree(), low_hp=100).floor(840) == 100 and HpWatch(tree(), low_hp=0.5).floor(840) == 133
    assert _watch([[0.0, 0.6, 500], [0.7, 0.6, 500]], low_hp=0.65) is not None
    assert _watch([[0.0, 0.6, 500], [0.7, 0.6, 500]]) is None
    # Her own potion key drinking under --potion-hp (none bound live since 2026-10-23; the first map's simulator binds
    # one, no pet): five points under it, if that's lower, so her key keeps her HP between the two.
    assert HpWatch(tree(), potion_hp=0.30).threshold(840) == pytest.approx(210)
    assert HpWatch(tree(), potion_hp=0.30).words(840) == "her potion key's 25% of 840"
    assert HpWatch(tree(), potion_hp=0.45).threshold(840) == 266


def test_magic_guard_takes_its_share_of_a_touch_while_her_mp_pays():
    r = reserve_of(TREE, 3, 75.0, 28, guard=0.3)
    assert r.hp(400, mp=500) == pytest.approx(3 * 133 * 0.7) and r.hp(400, mp=50) == 399 and r.hp(400) == 399
    assert "less Magic Guard's 30%" in r.words(400, 500)


def test_a_magician_too_strong_for_the_map_is_refused_up_front_unless_low_hp():
    """Her placeholder 400 HP: 3 Horny Mushroom touches (393) are over her pet's 70% (280), where her pet keeps her:
    the stop would end every hunt at once. The thief's 262 of 840 isn't."""
    def hunter(hits, max_hp, low_hp=None):
        h = object.__new__(Hunter)
        h.hp_watch = HpWatch(reserve_of(TREE, hits, 75.0, 28), low_hp=low_hp)
        h.her, h.max_hp_read, h.pet_warning, h.max_mp = SimpleNamespace(hp=max_hp), None, 0.7, 500.0
        return h
    stop = hunter(3, 400.0).too_strong()
    assert isinstance(stop, TooStrong) and stop.exit_code == EXIT and stop.after == "Nothing was hunted."
    assert str(stop) == ("the map's monsters are too strong for her here: 3 Horny Mushroom touches of 133 (399 HP) is at "
                         "or over her pet's warning, 70% of her 400 HP (280), where her pet keeps her: the safety stop "
                         "would end the hunt at once")
    assert hunter(3, 400.0, low_hp=0.3).too_strong() is None and hunter(2, 840.0).too_strong() is None


def _old_rule(looks, threshold=0.65, seconds=2.0, n=3, floor=0.25, floor_n=2):
    """The rule until 2026-10-28 (65% for 2 s over 3 looks, 25% two looks in a row): the s of its first stop, or None."""
    since, count, low = None, 0, 0
    for t, hp in looks:
        low = low + 1 if hp < floor else 0
        if hp >= threshold:
            since, count = None, 0
            continue
        since = t if since is None else since
        count += 1
        if low >= floor_n or (count >= n and t - since >= seconds):
            return t
    return None


def test_on_the_live_hunts_of_2026_10_28_the_old_rule_stopped_once_for_nothing_this_one_never():
    """Her HP as the three live tours of 2026-10-28 read it (tests/fixtures/hp_live_2026-10-28.json: level 28, max HP
    840, her pet's potions +100 against touches of 103-122: every look under 0.75 and its neighbours). The old rule
    stops the 00:53 run where it stopped live, at 01:00:55 (64% for 2 s): nothing was wrong (the user: "the pet is still
    drinking HP potion on each hit"). This one never: her lowest look 461 HP (the bar's 55.8% less a px), 199 over 262."""
    data = json.loads((FIXTURES / "hp_live_2026-10-28.json").read_text(encoding="utf-8"))
    runs, max_hp = data["runs"], data["max_hp"]
    assert max_hp == 840 and sum(r["looks"] for r in runs.values()) == 6955
    stopped = {name: _old_rule(r["low"]) for name, r in runs.items()}
    live = runs["20261028-005324-tour.jsonl"]["stop"]
    assert {n for n, s in stopped.items() if s is not None} == {"20261028-005324-tour.jsonl"}
    assert abs(stopped["20261028-005324-tour.jsonl"] - live) < 0.5  # (its look; the event's a moment after)
    lowest = min(hp for r in runs.values() for _, hp in r["low"])
    assert lowest == 0.558 and round((lowest - BAR_SLACK) * max_hp) == 461
    for name, r in runs.items():
        assert _watch([[t, hp] for t, hp in r["low"]], max_hp=max_hp) is None, name
        assert _watch([[t, hp] for t, hp in r["low"]], reserve=tree(hits=3), max_hp=max_hp) is None, name


def test_on_the_live_hunts_of_2026_10_25_it_never_stops():
    """Her HP as the five live hunts of 2026-10-25 23:43 to 2026-10-26 01:28 read it (79 minutes, level 25, max HP 774,
    her pet's potions +250: tests/fixtures/hp_live_2026-10-25.json): never under 53.8%, 409 HP over the bar's px: the
    old rule never stopped them, nor does this one, with 2 hits or 3."""
    runs = json.loads((FIXTURES / "hp_live_2026-10-25.json").read_text(encoding="utf-8"))["runs"]
    assert sum(r["looks"] for r in runs.values()) == 10548 and round(sum(r["minutes"] for r in runs.values())) == 79
    assert min(hp for r in runs.values() for _, hp in r["low"]) == 0.538
    for name, r in runs.items():
        assert _old_rule(r["low"]) is None, name
        for hits in (2, 3):
            assert _watch([[t, hp] for t, hp in r["low"]], reserve=tree(25, hits), max_hp=774.0) is None, name


def _hunter(log=None, level=28.0, touches=None):
    """A hunter as far as her HP number goes (Hunter._hp_number, _max_hp_seen, _touched, _level_check)."""
    from gamelab.perception import hp_number
    h = object.__new__(Hunter)
    h.model, h.log = TREE, (log.append if log is not None else (lambda line: None))
    h.her = SimpleNamespace(hp=774.0, level=25.0, defence=75.0)
    h.max_hp_read, h.hp_follow, h._last_touch, h._hp_box_t = None, hp_number.Follow(), None, None
    h._first_look, h._unread_said, h.asked, h.retreating = None, False, False, False
    h.touches = touches or Touches(None)
    h.safety_hits, h.guard, h.level = 2, 0.0, level
    h.hp_watch = HpWatch(h._reserve())
    h.meter = SimpleNamespace(at=None)
    return h


def test_her_hp_number_is_her_hp_her_max_hp_and_her_touches(tmp_path):
    from gamelab.localization.tracking import Event
    said = []
    h = _hunter(said, touches=Touches(tmp_path / "touches.jsonl"))
    assert h.max_hp == 774  # her profile's until the HUD's number is read
    look = SimpleNamespace(events=[Event("hp_read", 1.0, data={"cur": 840, "max": 840, "before": None}),
                                   Event("hp_read", 2.0, data={"cur": 723, "max": 840, "before": 840}),
                                   Event("hp_read", 2.1, data={"cur": 823, "max": 840, "before": 723})],
                           hp_of=(823, 840), hp_box=None)
    assert h._hp_number(look) == (823, 840) and h.max_hp == 840
    assert said == ["  her max HP 840 by the HUD's number; her profile's (config.toml [character] hp) is 774: stale"]
    rows = Touches(tmp_path / "touches.jsonl").rows
    # Her level not known yet (her EXP number not read: her profile's 25 isn't taken for it): kept at none.
    assert [(r["hp"], r["kind"], r["level"], r["max"], r["map"]) for r in rows] == [(117, "horny-mushroom", None, 840,
                                                                                     TREE.id)]
    assert h._last_touch == (2.0, 840, 723)
    # Her level read: a read whose change before didn't read (before None) isn't a touch; a touch over the reserve's
    # raises it at once (MARGIN over it), said; one of no kind is said and counts for none.
    h.meter.at = (28, 1000)
    h._hp_number(SimpleNamespace(events=[Event("hp_read", 9.0, data={"cur": 600, "max": 840, "before": None}),
                                         Event("hp_read", 9.9, data={"cur": 460, "max": 840, "before": 600})],
                                 hp_of=None, hp_box=None))
    assert [(r["hp"], r["level"]) for r in h.touches.rows] == [(117, None), (140, 28)]
    assert h.hp_watch.reserve.hp(840) == 2 * 147 and said[-1].startswith(
        "  a touch of 140 HP (600 -> 460, a horny-mushroom's) raises the safety stop: now under 294 HP of 840")
    h._hp_number(SimpleNamespace(events=[Event("hp_read", 12.0, data={"cur": 200, "max": 840, "before": 460})],
                                 hp_of=None, hp_box=None))
    assert h.touches.rows[-1]["kind"] is None and h.hp_watch.reserve.hp(840) == 294
    assert said[-1] == ("  a touch of 260 HP (460 -> 200) fits no kind of the map's: kept, counted for none (the safety "
                        "stop's Horny Mushroom touch 147)")
    # A new max (a level up): hers.
    h._hp_number(SimpleNamespace(events=[], hp_of=(862, 862), hp_box=None))
    assert h.max_hp == 862 and said[-1] == "  her max HP 862 by the HUD's number (was 840)"


def test_her_reserve_is_made_again_when_she_levels(tmp_path):
    said = []
    rows = [{"level": 29, "hp": 140, "kind": "horny-mushroom", "map": TREE.id}] * 10
    h = _hunter(said, touches=Touches(None))
    h.touches.rows = rows
    h._level_check()
    assert said == [] and h.hp_watch.reserve.hp(840) == 266
    h.meter.at = (29, 120)
    h._level_check()
    assert h.level == 29 and h.hp_watch.reserve.hp(840) == 2 * 147
    assert said[-1].startswith("  her level 29 by her EXP number (was 28): the safety stop now under 294 HP")


def test_her_health_check_drinks_and_raises_the_stop_once():
    h = object.__new__(Hunter)
    drunk = []
    h.drink_if_low = drunk.append
    h.hp_watch, h.darts = HpWatch(tree()), DartWatch()
    looks = [SimpleNamespace(t=t / 4, hp=0.3) for t in range(5)]
    for s in looks[:3]:
        h.hp_watch.see(s.t, s.hp, 840)
        assert h.health_check(s) is None  # 0.5 s under it (her bar's)
    h.hp_watch.see(looks[3].t, looks[3].hp, 840)
    with pytest.raises(SafetyStop):
        h.health_check(looks[3])
    h.hp_watch.see(looks[4].t, looks[4].hp, 840)
    assert h.health_check(looks[4]) is None  # once
    assert drunk == looks[:5]  # her potions every time, first


def test_at_the_stop_she_retreats_from_it_its_said_in_capitals_she_is_paged_and_it_goes_on(monkeypatch):
    said, paged, went = [], [], []
    h = SimpleNamespace(log=said.append)

    def away(hunter, seen, why, log=None, rng=None):
        went.append((hunter, why))
        return retreat.Retreat(0.0, result="arrived")
    monkeypatch.setattr(retreat, "away", away)
    monkeypatch.setattr(retreat.alarm, "page", lambda title, text: paged.append(title))
    stop = SafetyStop(250, 0.7, 4, 262, max_hp=840, why="2 Horny Mushroom touches of 131")
    with pytest.raises(SafetyStop):
        with retreat.on_alarm(h) as watch:
            raise stop
    assert went == [(h, retreat.SAFETY)] and watch.retreat.result == "arrived"
    assert said[0].startswith("SAFETY STOP: HP 250 of 840, under 2 Horny Mushroom touches of 131 (262) for 0.7 s") \
        and "no sitting" in said[0]
    assert paged == ["MapleStory: her HP stayed low"]
    # A retreat that fails is said, and she's paged all the same.

    def broken(*a, **k):
        raise RuntimeError("no dot")
    monkeypatch.setattr(retreat, "away", broken)
    with pytest.raises(SafetyStop):
        with retreat.on_alarm(h):
            raise stop
    assert said[-1] == "retreat: failed (RuntimeError: no dot)" and len(paged) == 2
    # The Lie Detector's retreat is as it was: its own words, from its sighting.
    assert (retreat.LIE_DETECTOR.label, retreat.LIE_DETECTOR.after) == ("LIE DETECTOR", "after it was seen")


def test_a_hunting_command_stopped_by_it_exits_with_its_own_code(monkeypatch, capsys):
    from gamelab import __main__ as cli

    def tour(args):
        raise SafetyStop(200, 0.7, 3, 262, max_hp=840, why="2 Horny Mushroom touches of 131")
    monkeypatch.setattr(cli, "cmd_tour", tour)
    assert cli.main(["tour", "--loop"]) == EXIT
    out = capsys.readouterr().out
    assert out.startswith("SAFETY STOP: HP 200 of 840, under 2 Horny Mushroom touches of 131 (262) for 0.7 s") \
        and "The hunt ended; she went to a safe place" in out and "Check her HP potions" in out

    def strong(args):
        raise TooStrong(393, 400, 0.7, "3 Horny Mushroom touches of 131")
    monkeypatch.setattr(cli, "cmd_tour", strong)
    assert cli.main(["tour", "--loop"]) == EXIT
    out = capsys.readouterr().out
    assert out.startswith("TOO STRONG: the map's monsters are too strong for her here") and "Nothing was hunted." in out
    # `--low-hp` sets its threshold (None: her reserve), a share or HP, on every command that hunts; the rests' flags are
    # gone.
    p = cli.build_parser()
    for argv in (["tour"], ["hunt", "clear"], ["probe", "run", "top-60"]):
        assert p.parse_args(argv).low_hp is None and p.parse_args(argv + ["--low-hp", "0.5"]).low_hp == 0.5
        assert p.parse_args(argv + ["--low-hp", "300"]).low_hp == 300
    for gone in ("--rest-below", "--rest-avoid"):
        with pytest.raises(SystemExit):
            p.parse_args(["tour", gone, "0.5"] if gone == "--rest-below" else ["tour", gone])


def test_a_touch_counts_as_the_largest_of_its_prior_and_its_measured_with_the_margin():
    assert Touch("k", 131).hp == 131 and Touch("k", 131, 140, 1).hp == 147
    assert Touch("k", 131, 100, 50).hp == 131 and Touch("k", 131, 125, 1).hp == 132
    assert Reserve().hp(840) == pytest.approx(FALLBACK_SHARE * 840) and safety.HpWatch().threshold(840) == \
        pytest.approx(294)


def _looker(hps):
    """A hunter (test_fight's: an empty map, so her reserve is the fallback's, 35% of her 774 HP) whose looks read her
    bar's `hps` in turn, 0.1 s apart (Observations through the inline feed's step, as test_fight's EXP looks)."""
    import numpy as np

    from gamelab.perception.eyes import Observation
    from test_fight import _Eyes, hunter
    looks = [Observation(t=1.0 + k / 10, frame=np.zeros((10, 10, 3), np.uint8), tag=None, hp=hp, mp=1.0,
                         exp=np.zeros((26, 140), bool)) for k, hp in enumerate(hps)]
    return hunter(_Eyes(looks))


def test_the_stop_is_raised_at_the_look_its_due_at_whoever_takes_it():
    """The review of 2026-10-28: due only at health_check, the stop came up to 1.4 s late in the simulator (a mover's
    settle, a throw's looks ask none), a touch in that time in 3 of 42 runs with her pet out of potions. Now at the look
    it's due at (Hunter._stop_now, Senses.checks), once a hunt has asked after her health."""
    h = _looker([0.9, 0.05, 0.05, 0.05, 0.05])
    assert h.hp_watch.floor() == pytest.approx(0.35 * 774 / 2 * FLOOR_TOUCHES)
    h.health_check(h.look(mobs=False))  # (the hunt asks: from here on, any look)
    h.senses.look(mobs=False)  # under the floor: one look
    with pytest.raises(SafetyStop) as stop:
        h.senses.look(mobs=False)  # (as Mover.locate takes them: no health_check)
    assert stop.value.floor is not None and stop.value.looks == 2
    h.senses.look(mobs=False)  # once
    # Not before a hunt has asked after her health (a command that doesn't hunt: the pet's care)
    h = _looker([0.05] * 4)
    for _ in range(4):
        h.senses.look(mobs=False)
    assert h.hp_watch.due() is not None  # (due all along: never raised at a look)


def test_not_while_she_retreats_nor_with_stop_on_look_off(monkeypatch):
    from gamelab.behavior import hunter as hunter_mod
    h = _looker([0.9, 0.05, 0.05, 0.05])
    h.health_check(h.look(mobs=False))
    h.retreating = True  # (behavior.retreat.away: the Lie Detector's way to a safe place isn't cut short)
    h.senses.look(mobs=False)
    h.senses.look(mobs=False)
    h.retreating = False
    with pytest.raises(SafetyStop):
        h.senses.look(mobs=False)
    monkeypatch.setattr(hunter_mod, "STOP_ON_LOOK", False)
    h = _looker([0.9, 0.05, 0.05])
    h.health_check(h.look(mobs=False))
    h.senses.look(mobs=False)
    h.senses.look(mobs=False)
    with pytest.raises(SafetyStop):
        h.health_check(SimpleNamespace(hp=0.05, where=None, monsters=[]))


def test_a_retreat_holds_it_and_lets_it_go_after(monkeypatch):
    h = SimpleNamespace(retreating=False)
    seen = []
    monkeypatch.setattr(retreat, "_away", lambda hunter, *a, **k: seen.append(hunter.retreating) or "went")
    assert retreat.away(h, 0.0, retreat.SAFETY) == "went" and seen == [True] and h.retreating is False

    def broken(hunter, *a, **k):
        raise RuntimeError("no dot")
    monkeypatch.setattr(retreat, "_away", broken)
    with pytest.raises(RuntimeError):
        retreat.away(h, 0.0, retreat.SAFETY)
    assert h.retreating is False


def test_magic_guard_isnt_taken_off_her_reserve_while_nothing_reads_its_up(monkeypatch):
    """The review of 2026-10-28: Magic Guard is a toggle (pressed again it's off) and nothing reads whether it's up, so
    her reserve doesn't count on it (a magician with it at 80% would have stopped under 3 x 133 x 0.2 = 80 HP while a
    touch takes 133). Reserve still takes a share (`guard`) for when something does."""
    import dataclasses

    from gamelab import character as profiles
    from gamelab.behavior import hunter as hunter_mod
    from test_fight import hunter
    me = profiles.active()
    monkeypatch.setattr(hunter_mod.profiles, "active", lambda: dataclasses.replace(me, magic_guard=0.8))
    h = hunter()
    assert h.guard == 0.0 and h.hp_watch.reserve.guard == 0.0


def test_her_hp_number_not_read_is_said_once():
    said = []
    h = _hunter(said)
    for t in (0.0, 5.0, 10.5, 20.0):
        h._unread(SimpleNamespace(t=t), None)
    assert said == ["  her HP number not read in 10 s of looks (the perception service not restarted since it reads "
                    "it? a client size it reads none at?): the safety stop goes by her HP bar at her max HP 774"]
    said.clear()
    h = _hunter(said)
    h._unread(SimpleNamespace(t=0.0), None)
    h._unread(SimpleNamespace(t=1.0), (800, 840))  # read: nothing to say, ever
    h._unread(SimpleNamespace(t=30.0), None)
    assert said == []


def test_a_glance_feeds_it_and_raises_it_too():
    """A chain's watch glances at her every 30 ms or so and takes no look for up to seconds: in the simulator (the
    review's crowd, her pet out of potions) a stop's first look under her reserve came after a second touch, at 49-96
    HP. Her glances feed the watch (Hunter._glanced, Senses.glance_checks) and raise the stop as a look does."""
    h = _looker([0.9])
    h.health_check(h.look(mobs=False))
    states = iter([SimpleNamespace(t=2.0 + k / 30, hp=0.05, hp_of=None, mp=None, dot=(1.0, 1.0)) for k in range(2)])
    h.senses._feed = SimpleNamespace(glance=lambda after: next(states))
    assert h.senses.glance() == ((1.0, 1.0), 0.05)  # under the floor: one
    with pytest.raises(SafetyStop) as stop:
        h.senses.glance()
    assert stop.value.floor is not None and not stop.value.exact
    # Her HP number's, as the glance's state has it; a glance's HP 0 right after a reading of 30% or more isn't taken
    # (a look tells: Hunter._doubt_hp).
    h = _looker([0.9])
    h.health_check(h.look(mobs=False))
    h._glanced(SimpleNamespace(t=2.0, hp=0.0, hp_of=None, mp=None))
    assert h.hp_watch.share == 0.9 and h.hp_watch.at == 1.0
    h._glanced(SimpleNamespace(t=2.1, hp=0.5, hp_of=(400, 840), mp=None))
    assert (h.hp_watch.hp, h.hp_watch.exact, h.max_hp) == (400, True, 840)
