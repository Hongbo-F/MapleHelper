"""Her EXP gains from her EXP number read (localization.gains): a gain's size names the kind killed, a gain of two kills
is split, one no kill explains is flagged; jumps and falls wait for a second reading; level-ups; changes that can't be
read. And in the simulator, where FakeEyes draws her number with the game's glyphs: the hunter's count of her EXP is
the world's, every kill named by its kind."""

import dataclasses

import pytest

from gamedata.paths import COPY
from gamelab.localization.gains import ExpMeter, Gain
from gamelab.perception import exp_number
from gamelab.perception.exp_number import ExpReading
from gamelab.sim import fakes, scenarios
from gamelab.sim.world import Noise

THICKET = {"Orange Mushroom": 15, "Slime": 12, "Snail": 2, "Pig": 13}  # MapModel.mob_exp on Thicket Around the Beach II


def r(exp: int, *levels: int, percent: float = 0.0) -> ExpReading:
    return ExpReading(exp, percent, levels or (19,))


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


def test_a_change_it_cant_read_is_one_kill_and_its_exp_still_counts():
    m = ExpMeter(THICKET)
    assert m.wants_reading(0.0)  # nothing read yet
    m.update(0.0, r(453), False)
    assert not m.wants_reading(0.1)
    assert m.update(1.0, None, True) == [Gain(1.0, None, (None,))]  # the cursor over the number
    assert m.wants_reading(1.1) and not m.wants_reading(1.2)  # read again, RETRY_S apart
    assert m.update(1.6, r(478), False) == []  # read again: its kill was counted; the 25 EXP count from here
    assert m.total == 25 and m.update(2.0, r(491), True) == [Gain(2.0, 13, ("pig",))]
    assert m.per_minute(2.0) is None and m.per_minute(60.0) == pytest.approx(38 * 60 / 60)


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
def test_in_the_simulator_a_number_covered_for_a_while_still_counts_each_change_as_a_kill():
    h, sim = _hunt("hunt_clear_basin", noise=Noise(exp_hidden=((5.0, 25.0),)))
    world = [k for k in sim.world.kill_log if k[3] == "her"]
    hidden = [k for k in world if 5.0 <= k[0] < 25.0]
    unknown = [k for _, k in h.told].count(None)  # each change while covered a kill of a kind not known, and the
    assert hidden and len(hidden) <= unknown <= len(hidden) + 1  # first seen uncovered (its size not known either)
    assert h.exp_gains == len(world) and h.meter.total == sim.world.exp  # every kill counted, and the EXP
