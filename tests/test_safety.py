"""The safety stop (behavior.safety): her HP, in HP, under her reserve (her profile's hits of the map's largest touch:
the thief 3 for now, the magician 3) on LOOKS looks DEBOUNCE_S apart, or under one touch FLOOR_LOOKS looks in a row,
ends the hunt; she goes to a safe place as on the Lie Detector (behavior.retreat.on_alarm), the log says so in capitals
with the numbers, whoever is at the PC is paged, and the command exits with its own code. No sitting and no waiting for
HP (the rests went on 2026-10-26). Until 2026-10-28 it was 65% of her HP for 2 s over 3 looks, or 25% two looks in a
row (the tests of that rule went with it: what replaced each is below). Her max HP, level and DEF are perception's
(tests/test_bootstrap.py: the reserve made again whenever one changes); here they're given. In the simulator, end to
end: loop_tree_safety_stop (tests/test_sim.py runs every scenario)."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from gamelab.attack import danger
from gamelab.behavior import bootstrap, retreat, safety
from gamelab.behavior.darts import DartWatch
from gamelab.behavior.hunter import Hunter
from gamelab.behavior.safety import (BAR_SLACK, DEBOUNCE_EXACT_S, DEBOUNCE_S, EXIT, FALLBACK_SHARE, FLOOR_LOOKS,
                                     FLOOR_TOUCHES, HITS, LOOKS, MARGIN, HpWatch, Reserve, SafetyStop, TooStrong, Touch,
                                     Touches, kind_of_touch, kinds_of, reserve_of)

FIXTURES = Path(__file__).parent / "fixtures"
TREE = SimpleNamespace(id=10002073, mob_damage={"Green Mushroom": 54.0, "Horny Mushroom": 110.0, "Slime": 13.0})
ROCKY = SimpleNamespace(id=10004021, mob_damage={"Stump": 11.0, "Red Snail": 9.0, "Axe Stump": 75.0,
                                                 "Dark Stump": 34.0})
NONE = SimpleNamespace(id=1, mob_damage={})
FIT_28 = 106.5  # her DEF as her level-28 touches fit it, the low end (tests/test_bootstrap.py)


def top(pad, defence=FIT_28, level=28):
    """The formula's top: what a touch by a monster of PADamage `pad` takes at most (attack.danger.touch_range)."""
    return danger.touch_range(pad, defence, level)[1]


def tree(level=28.0, hits=HITS, defence=FIT_28, measured=None, guard=0.0):
    return reserve_of(kinds_of(TREE), hits, defence, level, measured, guard, TREE.id)


HORNY = top(110)  # (the formula's, at the fit: no number of its own)


def _watch(looks, reserve=None, max_hp=840.0, **kw):
    """An HpWatch fed `looks` ([t, share, exact HP or None]...); the stop if it came (and at which look), else None."""
    w = HpWatch(reserve or tree(), **kw)
    for i, (t, share, *exact) in enumerate(looks):
        w.see(t, share, max_hp, exact[0] if exact else None)
        if (stop := w.due()) is not None:
            return i, stop
    return None


def test_its_numbers():
    assert (HITS, DEBOUNCE_S, LOOKS, FLOOR_TOUCHES, FLOOR_LOOKS, MARGIN, FALLBACK_SHARE, EXIT) == \
           (3, 0.6, 2, 1, 2, 0.05, 0.35, 4)
    assert BAR_SLACK == 1 / 104  # a px of her HP bar at 1366x768
    assert (bootstrap.PRIOR_DEF, bootstrap.UNREAD_SHARE, bootstrap.UNREAD_LEVEL) == (0.0, 0.5, 1)


def test_the_reserve_is_her_hits_of_the_maps_largest_touch_in_hp():
    """The user, 2026-10-28: "it should be HP under the amount of taking 3 hits from the monsters on the map ... for
    thief with high evasion this should be 2 hits, 3 for magician"; "3 hits for now" for the thief. The touch: the
    game's formula at the kind's PADamage (the game data's), her DEF and level, the roll's top, the largest kind's."""
    r = tree()
    assert [(t.kind, t.name, t.pad, t.prior, t.hp) for t in r.touches] == [
        ("horny-mushroom", "Horny Mushroom", 110, HORNY, HORNY), ("green-mushroom", "Green Mushroom", 54, top(54),
                                                                    top(54)), ("slime", "Slime", 13, top(13), top(13))]
    assert r.hp(840) == 3 * HORNY and HORNY == danger.touch_taken(110, FIT_28, 28, danger.TOP_U)
    assert tree(hits=2).hp(840) == 2 * HORNY  # (the thief's 2, once her retreat dodges)
    assert r.words(840) == f"3 Horny Mushroom touches of {HORNY}"
    assert r.describe(840).startswith(f"under {3 * HORNY} HP of 840 ({3 * HORNY / 840:.0%}: 3 Horny Mushroom touches "
                                      f"of {HORNY}); Horny Mushroom {HORNY}: the formula's top at PADamage 110, DEF "
                                      f"{FIT_28:g}, level 28")
    # Absolute HP: her max HP doesn't move it, her level and DEF do (DEF 0, none known: the most a touch takes).
    assert tree(level=25).hp(774) == 3 * top(110, FIT_28, 25)
    assert tree(defence=0.0).hp(840) == 3 * top(110, 0.0, 28) > tree().hp(840)
    assert reserve_of(kinds_of(ROCKY), 3, FIT_28, 28).hp(840) == 3 * top(75)  # an Axe Stump's
    # A map whose monsters' touches the game data doesn't give: FALLBACK_SHARE of her max HP, said so.
    fb = reserve_of(kinds_of(NONE), 3, FIT_28, 28)
    assert fb.touch is None and fb.hp(840) == pytest.approx(294) and fb.floor(840) == pytest.approx(98)
    assert "no game data for the map's monsters' touches" in fb.words(840)
    # Her max HP not read: UNREAD_SHARE of her bar, a share (HpWatch counts in shares then).
    assert r.hp(None) == 0.5 and r.floor(None) == pytest.approx(0.5 / 3) and r.words(None) == (
        "50% of her HP bar (her max HP not read yet)")


def test_on_her_maps_from_their_game_data():
    from gamelab.localization import mapdata
    from gamelab.sim import scenarios
    want = {10002073: ("Horny Mushroom", 110), 10004021: ("Axe Stump", 75), 10000012: ("Orange Mushroom", 17)}
    have = {m: v for m, v in want.items() if scenarios.available(m)}
    if not have:
        pytest.skip("needs the maps' game data (gamedata/data)")
    for map_id, (name, pad) in have.items():
        r = reserve_of(kinds_of(mapdata.load(map_id)), 3, FIT_28, 28)
        assert (r.touch.name, r.touch.pad, r.hp(840)) == (name, pad, 3 * top(pad)), map_id


def test_her_measured_touches_count_from_the_first_at_her_level_with_a_margin():
    """A measured touch only ever raises a kind's touch (the larger of the prior and MARGIN over the largest measured),
    so none is waited for: the review of 2026-10-28 found one over the prior ignored until 10 were kept at her level and
    the hunter started again."""
    def rows(sizes, level=28, kind="horny-mushroom"):
        kept = Touches(None)
        kept.rows = [{"level": level, "hp": h, "kind": kind} for h in sizes]
        return kept.measured(28)
    # Level 28's live touches (103-122) over the formula's top at the fit's low end with the margin: the measured.
    t = tree(measured=rows([110] * 37 + [122])).touch
    assert (t.measured, t.n, t.hp) == (122, 38, math.ceil(1.05 * 122)) and t.hp > HORNY
    assert "1 measured at level 28, the largest 122, 5% over it" in tree(measured=rows([122])).describe(840)
    # Under the prior: the prior.
    assert tree(measured=rows([100])).touch.hp == HORNY
    # Other levels' don't count, nor one at no level (her level not read when it was taken), nor one of no kind.
    assert rows([140], level=25) == {} and rows([140], level=None) == {} and rows([140], kind=None) == {}
    # Green Mushroom's measured are its own.
    g = tree(measured=rows([60], kind="green-mushroom"))
    assert [(x.kind, x.hp) for x in g.touches][:2] == [("horny-mushroom", HORNY), ("green-mushroom", 63)]


def test_a_touchs_kind_is_told_by_its_size():
    # Her DEF as known (the fit's ends, half to twice) or, not known, any DEF up to KIND_DEF_TOP.
    kinds = kinds_of(TREE)
    assert [kind_of_touch(h, kinds, (FIT_28, 115.25), 28) for h in (103, 122, 133, 46, 62, 8, 11, 16, 200)] == [
        "horny-mushroom", "horny-mushroom", "horny-mushroom", "green-mushroom", "green-mushroom", "slime", None, None,
        None]  # (11 and 16: a touch and a potion in one frame, over a Slime's most at half her DEF; 200: none's)
    assert kind_of_touch(150, kinds, (0.0, None), 28) == "horny-mushroom"  # (DEF 0: up to the formula's most)
    # Kinds whose ranges overlap: the larger.
    both = kinds_of(SimpleNamespace(id=2, mob_damage={"Big": 60.0, "Small": 50.0}))
    assert kind_of_touch(50, both, (75.0, 75.0), 28) == "big"


def test_her_touches_are_kept_across_runs(tmp_path):
    path = tmp_path / "touches.jsonl"
    t = Touches(path)
    assert t.rows == []
    t.add({"t": 1.0, "map": TREE.id, "level": 28, "hp": 117, "kind": "horny-mushroom", "max": 840, "pads": [110]})
    path.write_text(path.read_text(encoding="utf-8") + "not json\n", encoding="utf-8")
    assert Touches(path).rows == t.rows and Touches(None).rows == []
    assert t.measured(28) == {"horny-mushroom": (117, 1)} and t.measured(27) == {} and t.measured(None) == {}


def test_two_looks_under_it_debounce_s_apart_stop_the_hunt():
    # Her bar alone (her HP number not read): a look every 0.25 s under the threshold (over the floor, a touch): the look
    # 0.6 s after the first stops it (her pet's potion, 0.04-0.20 s after a touch, and the bar's settling, 0.37 s,
    # waited for).
    th = 3 * HORNY
    under = [[k / 4, 0.3] for k in range(8)]
    i, stop = _watch(under)
    assert under[i][0] == 0.75 and stop.seconds == 0.75 and stop.looks == 4 and not stop.exact
    assert stop.hp == pytest.approx((0.3 - BAR_SLACK) * 840) and stop.threshold == th and stop.floor is None
    assert stop.max_hp == 840 and DEBOUNCE_S == 0.6
    assert str(stop) == (f"HP about 244 of 840, under 3 Horny Mushroom touches of {HORNY} ({th}) for 0.8 s (4 looks): "
                         "her pet's potions aren't keeping her up (none left? the pet hungry or not out?)")
    title, text = stop.page()
    assert title == "MapleStory: her HP stayed low" and text == (
        f"gamelab stopped the hunt: HP about 244 of 840, under 3 Horny Mushroom touches of {HORNY} ({th}) for 0.8 s. "
        "She went to a safe place. Check her potions and her pet.")
    # Her HP number's: 0.25 s (DEBOUNCE_EXACT_S: the potion only; the number doesn't lag), on two looks at least.
    under = [[k / 10, 0.3, 250] for k in range(8)]
    i, stop = _watch(under)
    assert DEBOUNCE_EXACT_S == 0.25 and under[i][0] == 0.3 and stop.looks == 4 and stop.hp == 250 and stop.exact
    assert str(stop).startswith(f"HP 250 of 840, under 3 Horny Mushroom touches of {HORNY} ({th}) for 0.3 s (4 looks)")
    # Two looks that far apart are enough, however few between them.
    assert _watch([[0.0, 0.3], [0.6, 0.3]])[0] == 1 and _watch([[0.0, 0.3, 250], [0.25, 0.3, 250]])[0] == 1
    # Once: what follows it (her way to a safe place) goes on, under the threshold still.
    w = HpWatch(tree())
    for t, share, hp in under:
        w.see(t, share, 840, hp)
    assert w.due() is not None and w.due() is None
    w.see(10.0, 0.1, 840, 50)
    assert w.due() is None and w.fired


def test_with_her_max_hp_not_read_it_goes_by_her_bar_in_shares():
    """Her HP number not read yet (bootstrap: the perception service not restarted on this code, or a client size it
    reads none at): under UNREAD_SHARE of her bar, said in shares; a touch's floor a third of it."""
    i, stop = _watch([[0.0, 0.45], [0.6, 0.45]], max_hp=None)
    assert i == 1 and stop.max_hp is None and stop.threshold == 0.5
    assert str(stop).startswith("HP about 44% of her bar, under 50% of her HP bar (her max HP not read yet) (50%) for "
                                "0.6 s (2 looks)")
    assert _watch([[0.0, 0.55], [0.7, 0.55]], max_hp=None) is None
    # An HP number with no max (never) or a look without the bar: nothing.
    w = HpWatch(tree())
    w.see(0.0, None, None, 300)
    assert w.hp is None
    i, stop = _watch([[0.0, 0.1], [0.1, 0.1]], max_hp=None)
    assert i == 1 and stop.floor == pytest.approx(0.5 / 3) and "(17% of her bar) on 2 looks in a row" in str(stop)


def test_her_pets_potion_a_misread_or_a_look_without_hp_doesnt_stop_it():
    # A touch under it, her pet's potion 0.2 s later: back over it, the stretch over.
    assert _watch([[0.0, 0.29, 240], [0.1, 0.29, 240], [0.2, 0.5, 420], [0.9, 0.5, 420]]) is None
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
    # Out of a gap in her looks (a trip, a loot run) under one touch: the next look under it too stops the hunt, the
    # debounce not waited.
    i, stop = _watch([[0.0, 0.7, 590], [3.4, 0.13, 110], [3.45, 0.13, 110]])
    assert i == 2 and stop.floor == HORNY and stop.looks == 2 and stop.seconds == pytest.approx(0.05)
    assert (f"HP 110 of 840, under 3 Horny Mushroom touches of {HORNY} ({3 * HORNY}) for 0.1 s (2 looks), under one "
            f"Horny Mushroom touch ({HORNY}) on 2 looks in a row: her pet") in str(stop)
    # One look under it, the next over it (a misread, a potion): not.
    assert _watch([[0.0, 0.1, 100], [0.1, 0.4, 340], [0.2, 0.1, 100], [0.3, 0.7, 590]]) is None
    # Dead (HP 0): it stops too, and says she may be.
    i, stop = _watch([[0.0, 0.0, 0], [0.1, 0.0, 0]])
    assert i == 1 and "she may be dead" in str(stop)


def test_her_hp_is_her_hp_numbers_else_the_bar_less_a_px():
    w = HpWatch(tree())
    w.see(0.0, 0.43, 840, 3 * HORNY + 1)  # the number says one over it, though the bar (lagging it) says 43%
    assert w.hp == 3 * HORNY + 1 and w.since is None
    w.see(0.1, 0.44, 840)  # no number: the bar's 44% less a px
    assert w.hp == pytest.approx((0.44 - 1 / 104) * 840) and w.since == 0.1 and not w.exact
    w.see(0.8, 0.44, 840)
    stop = w.due()
    assert stop.her_hp == "HP about 362 of 840" and not stop.exact


def test_low_hp_overrides_it_as_a_share_or_in_hp_and_a_potion_key_of_her_own_caps_it():
    assert HpWatch(tree(), low_hp=0.5).threshold(840) == 420 and HpWatch(tree(), low_hp=300).threshold(840) == 300
    assert HpWatch(tree(), low_hp=0.5).words(840) == "--low-hp's 50% of 840"
    # With --low-hp under one touch the floor is the threshold.
    assert HpWatch(tree(), low_hp=100).floor(840) == 100 and HpWatch(tree(), low_hp=0.5).floor(840) == HORNY
    assert _watch([[0.0, 0.6, 500], [0.7, 0.6, 500]], low_hp=0.65) is not None
    assert _watch([[0.0, 0.6, 500], [0.7, 0.6, 500]]) is None
    # Her own potion key drinking under --potion-hp (none bound live since 2026-10-23; the first map's simulator binds
    # one, no pet): five points under it, if that's lower, so her key keeps her HP between the two.
    assert HpWatch(tree(), potion_hp=0.30).threshold(840) == pytest.approx(210)
    assert HpWatch(tree(), potion_hp=0.30).words(840) == "her potion key's 25% of 840"
    assert HpWatch(tree(), potion_hp=0.50).threshold(840) == 3 * HORNY


def test_magic_guard_takes_its_share_of_a_touch_while_her_mp_pays():
    r = tree(guard=0.3)
    assert r.hp(400, mp=500) == pytest.approx(3 * HORNY * 0.7) and r.hp(400, mp=50) == 3 * HORNY
    assert r.hp(400) == 3 * HORNY and "less Magic Guard's 30%" in r.words(400, 500)


def test_a_map_too_strong_for_her_is_refused_unless_low_hp():
    """Her reserve at or over her pet's 70% at her max HP, where her pet keeps her: the stop would end every hunt at
    once. Not known while her max HP isn't read."""
    def hunter(max_hp, low_hp=None):
        h = object.__new__(Hunter)
        h.hp_watch = HpWatch(tree(), low_hp=low_hp)
        h.stats, h.pet_warning, h.max_mp = bootstrap.HerStats(), 0.7, 500.0
        h.stats.see_max_hp(max_hp)
        return h
    stop = hunter(400).too_strong()
    assert isinstance(stop, TooStrong) and stop.exit_code == EXIT and stop.after == "Nothing was hunted."
    assert str(stop) == (f"the map's monsters are too strong for her here: 3 Horny Mushroom touches of {HORNY} "
                         f"({3 * HORNY} HP) is at or over her pet's warning, 70% of her 400 HP (280), where her pet "
                         "keeps her: the safety stop would end the hunt at once")
    assert hunter(400, low_hp=0.3).too_strong() is None and hunter(840).too_strong() is None
    assert hunter(None).too_strong() is None


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
    drinking HP potion on each hit"). This one never, at 2 hits or 3, her DEF as her touches fit it then; nor at the
    prior's DEF 0 (none known yet), by a few HP: her lowest look 461 (the bar's 55.8% less a px)."""
    data = json.loads((FIXTURES / "hp_live_2026-10-28.json").read_text(encoding="utf-8"))
    runs, max_hp = data["runs"], data["max_hp"]
    assert max_hp == 840 and sum(r["looks"] for r in runs.values()) == 6955
    stopped = {name: _old_rule(r["low"]) for name, r in runs.items()}
    live = runs["20261028-005324-tour.jsonl"]["stop"]
    assert {n for n, s in stopped.items() if s is not None} == {"20261028-005324-tour.jsonl"}
    assert abs(stopped["20261028-005324-tour.jsonl"] - live) < 0.5  # (its look; the event's a moment after)
    lowest = min(hp for r in runs.values() for _, hp in r["low"])
    assert lowest == 0.558 and round((lowest - BAR_SLACK) * max_hp) == 461 > tree(defence=0.0).hp(840)
    for name, r in runs.items():
        for reserve in (tree(), tree(hits=2), tree(defence=0.0)):
            assert _watch([[t, hp] for t, hp in r["low"]], reserve=reserve, max_hp=max_hp) is None, name


def test_on_the_live_hunts_of_2026_10_25_it_never_stops_at_her_def_then():
    """Her HP as the five live hunts of 2026-10-25 23:43 to 2026-10-26 01:28 read it (79 minutes, level 25, max HP 774,
    her pet's potions +250: tests/fixtures/hp_live_2026-10-25.json): never under 53.8%, 409 HP over the bar's px: the
    old rule never stopped them, nor does this one at 2 hits or 3 with her DEF as her level-25 touches fit it (59.75 at
    the low end: tests/test_bootstrap.py). At the prior's DEF 0 (her first touches of a run with none kept) one look
    of 10,548 is under it (409 against 453), and one look doesn't stop it."""
    runs = json.loads((FIXTURES / "hp_live_2026-10-25.json").read_text(encoding="utf-8"))["runs"]
    assert sum(r["looks"] for r in runs.values()) == 10548 and round(sum(r["minutes"] for r in runs.values())) == 79
    assert min(hp for r in runs.values() for _, hp in r["low"]) == 0.538
    for name, r in runs.items():
        assert _old_rule(r["low"]) is None, name
        for reserve in (tree(25, 2, 59.75), tree(25, 3, 59.75), tree(25, 3, 0.0)):
            assert _watch([[t, hp] for t, hp in r["low"]], reserve=reserve, max_hp=774.0) is None, name
    under = [hp for r in runs.values() for _, hp in r["low"] if (hp - BAR_SLACK) * 774 < tree(25, 3, 0.0).hp(774)]
    assert under == [0.538]


def test_her_health_check_drinks_and_raises_the_stop_once():
    h = object.__new__(Hunter)
    drunk = []
    h.drink_if_low = drunk.append
    h.hp_watch, h.darts, h._strong = HpWatch(tree()), DartWatch(), None
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
    # The map found too strong at the look that read her max HP (none read before the command started): raised once.
    h._strong = TooStrong(400.0, 500.0, 0.7, "3 touches")
    with pytest.raises(TooStrong):
        h.health_check(looks[4])
    assert h._strong is None and h.health_check(looks[4]) is None


def test_at_the_stop_she_retreats_from_it_its_said_in_capitals_she_is_paged_and_it_goes_on(monkeypatch):
    said, paged, went = [], [], []
    h = SimpleNamespace(log=said.append)

    def away(hunter, seen, why, log=None, rng=None):
        went.append((hunter, why))
        return retreat.Retreat(0.0, result="arrived")
    monkeypatch.setattr(retreat, "away", away)
    monkeypatch.setattr(retreat.alarm, "page", lambda title, text: paged.append(title))
    stop = SafetyStop(250, 0.7, 4, 3 * HORNY, max_hp=840, why=f"3 Horny Mushroom touches of {HORNY}")
    with pytest.raises(SafetyStop):
        with retreat.on_alarm(h) as watch:
            raise stop
    assert went == [(h, retreat.SAFETY)] and watch.retreat.result == "arrived"
    assert said[0].startswith(f"SAFETY STOP: HP 250 of 840, under 3 Horny Mushroom touches of {HORNY} ({3 * HORNY}) "
                              "for 0.7 s") and "no sitting" in said[0]
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
        raise SafetyStop(200, 0.7, 3, 3 * HORNY, max_hp=840, why=f"3 Horny Mushroom touches of {HORNY}")
    monkeypatch.setattr(cli, "cmd_tour", tour)
    assert cli.main(["tour", "--loop"]) == EXIT
    out = capsys.readouterr().out
    assert out.startswith(f"SAFETY STOP: HP 200 of 840, under 3 Horny Mushroom touches of {HORNY} ({3 * HORNY}) for "
                          "0.7 s") and "The hunt ended; she went to a safe place" in out and "Check her HP potions" in out

    def strong(args):
        raise TooStrong(3 * HORNY, 400, 0.7, f"3 Horny Mushroom touches of {HORNY}")
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
    """A hunter (test_fight's: an empty map, so her reserve is the fallback's, 35% of her max HP once read) whose looks
    read her bar's `hps` in turn, 0.1 s apart, her HP number unread (Observations through the inline feed's step, as
    test_fight's EXP looks): in shares of her bar, under UNREAD_SHARE."""
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
    assert h.hp_watch.floor() == pytest.approx(bootstrap.UNREAD_SHARE / HITS * FLOOR_TOUCHES)
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
    her reserve doesn't count on it (a magician with it at 80% would have stopped under a fifth of her 3 touches while
    a touch takes all of itself). Reserve still takes a share (`guard`) for when something does."""
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
    h = _looker([0.9])
    h.log = said.append
    for t in (0.0, 5.0, 10.5, 20.0):
        h._unread(SimpleNamespace(t=t), None)
    assert said == ["  her HP number not read in 10 s of looks (the perception service not restarted since it reads "
                    "it? a client size it reads none at?): the safety stop goes by her HP bar, under 50% of her HP bar "
                    "(her max HP not read yet)"]
    said.clear()
    h = _looker([0.9])
    h.log = said.append
    h._unread(SimpleNamespace(t=0.0), None)
    h._unread(SimpleNamespace(t=1.0), (800, 840))  # read: nothing to say, ever
    h._unread(SimpleNamespace(t=30.0), None)
    assert said == []


def test_her_hp_number_gives_her_max_hp_and_her_touches_kept_with_her_level_and_the_maps_pads(tmp_path):
    from gamelab.localization.tracking import Event
    from test_bootstrap import _hunter
    said = []
    h = _hunter(log=said)
    h.touches = Touches(tmp_path / "touches.jsonl")
    look = SimpleNamespace(events=[Event("hp_read", 1.0, data={"cur": 840, "max": 840, "before": None}),
                                   Event("hp_read", 2.0, data={"cur": 723, "max": 840, "before": 840}),
                                   Event("hp_read", 2.1, data={"cur": 823, "max": 840, "before": 723})],
                           hp_of=(823, 840), hp_box=None)
    assert h._hp_number(look) == (823, 840) and h.max_hp == 840
    assert h._notes == ["her max HP 840 by her HUD's HP number"]
    rows = Touches(tmp_path / "touches.jsonl").rows
    # Her level not known yet: kept at none (counted at no level, not fitted); the map's PADamages with it.
    assert [(r["hp"], r["kind"], r["level"], r["max"], r["map"], r["pads"]) for r in rows] == [
        (117, "horny-mushroom", None, 840, 10002073, [13.0, 54.0, 110.0])]
    assert h._last_touch == (2.0, 840, 723)
    # Her level read: a read whose change before didn't read (before None) isn't a touch; one of no kind is said and
    # counts for none.
    h._level_seen(28, "her HUD's LV. digits")
    h._reserve_check()
    h._hp_number(SimpleNamespace(events=[Event("hp_read", 9.0, data={"cur": 600, "max": 840, "before": None}),
                                         Event("hp_read", 9.9, data={"cur": 480, "max": 840, "before": 600})],
                                 hp_of=None, hp_box=None))
    assert [(r["hp"], r["level"]) for r in h.touches.rows] == [(117, None), (120, 28)]
    h._hp_number(SimpleNamespace(events=[Event("hp_read", 12.0, data={"cur": 100, "max": 840, "before": 480})],
                                 hp_of=None, hp_box=None))
    assert h.touches.rows[-1]["kind"] is None
    assert said[-1].startswith("  a touch of 380 HP (480 -> 100) fits no kind of the map's: kept, counted for none")


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
    # Her HP number's, as the glance's state has it (her max HP and level with it: the reserve made again); a glance's
    # HP 0 right after a reading of 30% or more isn't taken (a look tells: Hunter._doubt_hp).
    h = _looker([0.9])
    h.health_check(h.look(mobs=False))
    h._glanced(SimpleNamespace(t=2.0, hp=0.0, hp_of=None, mp=None))
    assert h.hp_watch.share == 0.9 and h.hp_watch.at == 1.0
    h._glanced(SimpleNamespace(t=2.1, hp=0.5, hp_of=(400, 840), level_of=28, mp=None))
    assert (h.hp_watch.hp, h.hp_watch.exact, h.max_hp, h.level) == (400, True, 840, 28)
