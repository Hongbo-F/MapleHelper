"""Her EXP gains from her EXP number read (localization.gains): a gain's size names the kind killed, a gain of two kills
is split, one no kill explains is flagged; jumps and falls wait for a second reading; level-ups; changes that can't be
read wait for a reading to say what they were (a cursor passing over the number is no kill); sizes that aren't the
game data's (another EXP rate) turn the sizes off. And in the simulator, where FakeEyes draws her number with the
game's glyphs: the hunter's count of her EXP is the world's, every kill named by its kind."""

import dataclasses

import pytest

from gamedata.paths import COPY
from gamelab.localization import gains
from gamelab.localization.gains import ExpMeter, Gain
from gamelab.perception import exp_number
from gamelab.perception.exp_number import ExpReading
from gamelab.sim import fakes, scenarios
from gamelab.sim.world import Noise

THICKET = {"Orange Mushroom": 15, "Slime": 12, "Snail": 2, "Pig": 13}  # MapModel.mob_exp on Thicket Around the Beach II


def r(exp: int, *levels: int, percent: float = 0.0) -> ExpReading:
    return ExpReading(exp, percent, levels or (19,))


def seen() -> ExpMeter:
    """A meter whose gains have been seen to be the game data's sizes (the rate check passed)."""
    m = ExpMeter(THICKET)
    m.sized = True
    return m


def test_a_gains_size_names_the_kind_and_two_kills_are_split():
    m = ExpMeter(THICKET)
    assert m.split(13) == ("pig",) and m.split(12) == ("slime",) and m.split(2) == ("snail",)
    assert m.split(15) == ("orange-mushroom",)
    assert m.split(25) == ("pig", "slime") and m.split(26) == ("pig", "pig") and m.split(4) == ("snail", "snail")
    assert m.split(39) == (None, None, None)  # three pigs, or two slimes and an Orange Mushroom
    assert m.split(7) == () and m.split(1) == () and m.split(61) == ()  # no kills add up to these


def test_the_saved_frames_gains_were_a_pig_two_slimes_and_two_pigs():
    m = ExpMeter(THICKET)
    assert m.update(0.0, r(453), changed=False) == []  # the first reading: where she starts from
    gains = [g for t, exp in enumerate((466, 478, 490, 503, 516), 1) for g in m.update(float(t), r(exp), True)]
    assert [g.kinds for g in gains] == [("pig",), ("slime",), ("slime",), ("pig",), ("pig",)]
    assert m.total == 63 and m.counts["kills"] == 5
    assert m.update(6.0, r(516), changed=True) == []  # the box changed but not the number: no gain


def test_a_size_no_kill_gives_is_flagged_and_isnt_a_kill():
    m = ExpMeter(THICKET)
    m.update(0.0, r(453), False)
    (g,) = m.update(1.0, r(460), True)  # +7: a party member's share, say
    assert g == Gain(1.0, 7, ()) and g.odd and g.kills == 0
    assert m.total == 7 and m.counts["odd"] == 1


def test_a_jump_waits_for_a_second_reading_and_a_misread_one_is_dropped():
    m = ExpMeter(THICKET)
    m.update(0.0, r(453), False)
    assert m.update(1.0, r(653), True) == [] and m.held is not None  # +200: more than 4 Orange Mushrooms
    assert m.wants_reading(1.2)  # the next look reads it again ...
    (g,) = m.update(1.2, r(653), False)  # ... and it agrees: a quest's EXP, say
    assert g.t == 1.0 and g.exp == 200 and g.odd and m.at == (19, 653)
    assert m.update(2.0, r(1653), True) == []  # a 1 misread for a 6, say
    assert m.update(2.5, r(666), False) == [Gain(2.5, 13, ("pig",))]  # the next reading says otherwise
    assert m.counts["dropped"] == 1 and m.total == 213


def test_a_fall_waits_for_a_second_reading_too_and_isnt_a_gain():
    m = ExpMeter(THICKET)
    m.update(0.0, r(5000), False)
    assert m.update(1.0, r(3332), True) == []  # 10% of level 19's 16,680 lost: a death
    (g,) = m.update(1.6, r(3332), False)
    assert g.exp == -1668 and g.odd and m.counts["fell"] == 1 and m.total == 0
    assert m.update(2.0, r(3345), True) == [Gain(2.0, 13, ("pig",))]


def test_a_level_up_is_a_gain_of_what_was_left_and_what_came():
    m = ExpMeter(THICKET)
    m.update(0.0, r(16670, 19), False)
    # 3 EXP into level 20 shows 0.01%: several levels fit that percent; she was 19, so it's 20
    (g,) = m.update(1.0, ExpReading(3, 0.01, (20, 21, 22)), True)
    assert g.kinds == ("pig",) and m.at == (20, 3) and m.total == 13


def test_levels_that_dont_follow_are_no_gain_and_not_a_fall():
    m = ExpMeter(THICKET)
    m.update(0.0, r(453, 19), False)
    assert m.update(1.0, r(100, 21), True) == []  # two levels up at once: misread, or held for a second reading
    (g,) = m.update(1.5, r(100, 21), False)
    assert g == Gain(1.0, None, ()) and g.odd and m.counts["levels"] == 1 and m.counts["fell"] == 0


def test_a_cursor_passing_over_the_number_is_no_kill():
    m = ExpMeter(THICKET)
    assert m.wants_reading(0.0)  # nothing read yet
    m.update(0.0, r(453), False)
    assert not m.wants_reading(0.1)
    assert m.update(1.0, None, True) == []  # the cursor comes over the number: a change it can't read ...
    assert m.update(1.1, None, True) == [] and m.unread == 2  # ... and moves over it
    assert m.due(2.9) == []  # a burst of changes (0.1 s apart) is a cursor moving, never a kill on its own
    assert m.update(3.0, r(453), True) == []  # it goes: the number reads as before, so nothing came
    assert m.counts["covered"] == 1 and m.counts["kills"] == 0 and m.total == 0 and m.due(10.0) == []


def test_a_kill_under_the_cursor_counts_when_the_number_reads_again():
    m = seen()
    m.update(0.0, r(453), False)
    assert m.update(1.0, None, True) == []  # the cursor comes
    assert m.update(2.0, None, True) == []  # a pig dies under it: the pixels that show change
    assert m.update(2.5, r(466), True) == [Gain(2.0, 13, ("pig",), (2.0,))]  # it goes: 13, a pig, when it changed
    assert m.total == 13 and m.counts["kills"] == 1 and m.unread == 0
    # two kills under it of different kinds: two kills, their kinds not known (which change was which isn't)
    m.update(3.0, None, True), m.update(4.0, None, True), m.update(5.0, None, True)
    assert m.update(5.4, r(491), True) == [Gain(4.0, 25, (None, None), (4.0, 5.0))]


def test_a_change_left_unread_is_a_kill_after_a_while_but_the_covers_coming_isnt():
    m = ExpMeter(THICKET)
    m.update(0.0, r(453), False)
    m.update(1.0, None, True)  # the cursor comes and stays
    assert m.due(9.0) == []  # the first burst is the cover coming: never a kill on its own
    m.update(10.0, None, True)  # a kill under it
    assert m.due(11.0) == [] and m.due(11.6) == [Gain(10.0, None, (None,))]  # unexplained UNREAD_WAIT_S later
    assert m.due(30.0) == []
    assert m.update(30.0, r(466), True) == []  # the cursor goes: 13, the one kill already counted
    assert m.counts["kills"] == 1 and m.counts["late"] == 1 and m.counts["over"] == 0 and m.total == 13
    # A cursor that comes and goes and comes back (two bursts) with no kill: one counted by then, one too many
    m.update(31.0, None, True), m.update(33.0, None, True)
    assert m.due(35.0) == [Gain(33.0, None, (None,))]
    assert m.update(36.0, r(466), True) == [] and m.counts["over"] == 1


def test_a_reading_after_changes_it_couldnt_read_is_still_held_when_it_jumps():
    m = seen()
    m.update(0.0, r(453), False)
    m.update(1.0, None, True)
    assert m.update(1.5, r(153), True) == [] and m.held is not None  # misread: -300 waits for a second reading
    assert m.update(2.0, r(466), False) == [Gain(1.0, 13, ("pig",), (1.0,))]  # which says otherwise: one pig
    assert m.counts["dropped"] == 1 and m.total == 13


def test_before_the_first_reading_each_change_is_a_kill_as_before():
    m = ExpMeter(THICKET)
    assert m.update(0.0, None, True) == [Gain(0.0, None, (None,))]  # nothing read yet to tell it by
    m.update(0.5, r(466), False)
    assert m.update(1.0, r(478), True) == [Gain(1.0, 12, ("slime",))] and m.total == 12


def test_its_exp_a_minute_and_reading_less_often_while_it_cant_be_read():
    m = ExpMeter(THICKET)
    m.update(0.0, r(453), False)
    m.update(1.0, r(466), True)
    assert m.per_minute(5.0) is None and m.per_minute(60.0) == pytest.approx(13.0)
    m.update(61.0, None, True)  # a change it can't read: read again RETRY_S later ...
    tried = [k / 2 for k in range(123, 200) if m.wants_reading(k / 2) and not m.update(k / 2, None, False)]
    assert tried[:5] == [61.5, 62.5, 64.5, 68.5, 76.5]  # ... and then each time twice as long, to RETRY_MAX_S
    assert all(b - a == pytest.approx(gains.RETRY_MAX_S) for a, b in zip(tried[3:], tried[4:]))


def test_two_kills_in_one_gain_are_split_once_the_sizes_are_seen_to_be_the_game_datas():
    m = ExpMeter(THICKET)
    m.update(0.0, r(453), False)
    assert m.update(1.0, r(479), True) == [Gain(1.0, 26, (None,))]  # two pigs, or one at rate 2: one kill for now
    exp = 479
    for t, size in enumerate((13, 12, 13, 2, 15, 12, 13), 2):
        exp += size
        m.update(float(t), r(exp), True)
    assert m.sized and not m.notes  # 7 of the last 8 one kill
    assert m.update(9.0, r(exp + 26), True) == [Gain(9.0, 26, ("pig", "pig"))]


def test_another_exp_rate_turns_the_sizes_off():
    """At an event's rate 2 every kill reads as two of its kind (a pig's 26 two pigs): each counted as one kill until
    RATE_GAINS of them turn the sizes off (each change one kill of a kind not known), until they're the game data's
    again."""
    m = ExpMeter(THICKET)
    m.update(0.0, r(453), False)
    exp = 453
    for t, size in enumerate((26, 24, 26, 30, 26, 4, 24, 26), 1):
        exp += size
        (g,) = m.update(float(t), r(exp), True)
        assert g.kinds == (None,)  # one kill each, not split
    assert m.sized is False and m.notes and "they fit rate 2 " in m.notes[0] and m.counts["kills"] == 8
    exp += 26
    assert m.update(9.0, r(exp), True) == [Gain(9.0, 26, (None,))]  # one kill
    for t, size in enumerate((13, 12, 13, 13, 15, 2), 10):  # back at rate 1
        exp += size
        m.update(float(t), r(exp), True)
    assert m.sized and len(m.notes) == 2 and m.counts["unsized"] == 7  # (the eighth, and 6 more)
    exp += 13
    assert m.update(20.0, r(exp), True) == [Gain(20.0, 13, ("pig",))]


def test_with_the_kinds_off_every_change_is_one_kill(monkeypatch):
    monkeypatch.setattr(gains, "NAME_KINDS", False)
    m = ExpMeter(THICKET)
    m.update(0.0, r(453), False)
    assert m.update(1.0, r(479), True) == [Gain(1.0, 26, (None,))]  # not two pigs: one change, one kill
    assert m.update(2.0, r(486), True) == [Gain(2.0, 7, (None,))]  # a size no kill gives: still a kill, not flagged
    assert m.counts["odd"] == 0 and m.counts["kills"] == 2 and m.total == 33


# --- in the simulator --------------------------------------------------------------------------------------------

needs_map = pytest.mark.skipif(not (COPY / "UI" / "UI_000.wz").is_file() or scenarios.real_map() is None,
                               reason="needs the game data copy (the map and the bar's glyphs)")


@needs_map
def test_the_simulators_number_reads_as_her_exp_and_level_ups_carry_over():
    assert fakes.shown_exp((19, 421), 0) == (19, 421)
    assert fakes.shown_exp((19, 16670), 23) == (20, 13)  # level 19 needs 16,680
    for level, exp in ((19, 421), (20, 13), (19, 16679)):
        reading = exp_number.read(exp_number.draw(exp_number.shown(exp, level)))
        assert reading.exp == exp and level in reading.levels


def _hunt(name: str, **changes):
    made = []
    real = scenarios.Sim.hunter

    def hunter(self, **kw):
        made.append(real(self, **kw))
        return made[-1]
    scenarios.Sim.hunter = hunter
    try:
        trace, sim = scenarios.run_sim(dataclasses.replace(scenarios.SCENARIOS[name], **changes))
    finally:
        scenarios.Sim.hunter = real
    return made[0], sim


@needs_map
def test_in_the_simulator_her_exp_off_the_number_is_the_worlds_and_each_kill_is_named():
    h, sim = _hunt("hunt_clear_basin")
    world = [k for k in sim.world.kill_log if k[3] == "her"]
    assert h.meter.total == sim.world.exp and h.meter.counts["unread"] == 0 and h.meter.counts["odd"] == 0
    named = [k for _, k in h.told]
    assert len(named) == len(world) and None not in named
    assert sorted(named) == sorted(kind for _, kind, _, _ in world)


@needs_map
def test_in_the_simulator_a_number_covered_for_a_while_counts_its_kills_and_not_the_cover():
    """The number covered for 20 s (its pixels changing as the cover comes and goes, as in game): every kill counted
    once, the cover's coming and going none; those under it a kind not known, at the time its change came."""
    h, sim = _hunt("hunt_clear_basin", noise=Noise(exp_hidden=((6.0, 24.5),)))
    world = [k for k in sim.world.kill_log if k[3] == "her"]
    hidden = [k for k in world if 6.0 <= k[0] < 24.5]
    unknown = [t for t, k in h.told if k is None]
    assert hidden and len(unknown) == len(hidden) and h.meter.counts["late"] == len(hidden) - 1  # (the last: when it's read)
    for (t, *_), told in zip(hidden, unknown):
        assert 0 <= told - fakes.START - t < 0.6  # (the look after the kill)
    assert h.exp_gains == len(world) and h.meter.total == sim.world.exp  # every kill counted once, and the EXP
    assert h.meter.counts["over"] == 0
