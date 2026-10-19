import threading
import time
from dataclasses import fields

from maplehelper import keys
from maplehelper.combos import COMBOS
from maplehelper.config import GameKeys, Timing
from maplehelper.engine import DROPPED, LEFT, QUEUED, RIGHT, STARTED, Combo, Runner

K = keys.parse
ZERO = Timing(**{f.name: 0 for f in fields(Timing)})


class Recorder:
    """Stands in for both the keyboard and the coach overlay."""

    def __init__(self):
        self.events: list[tuple[str, str]] = []
        self.steps: list[str] = []
        self.dones: list[tuple[str, list[str]]] = []

    def send(self, key, down):
        self.events.append(("down" if down else "up", key.name))

    def step(self, title, text):
        self.steps.append(text)

    def done(self, title, text, recipe, linger_ms):
        self.dones.append((text, list(recipe)))


def game_keys(**overrides):
    base = dict(left=K("left"), right=K("right"), jump=K("alt"), attack=K("ctrl"),
                lucky_seven=K("shift"))
    return GameKeys(**(base | overrides))


def make_runner(rec, gk=None, timing=ZERO, combos=COMBOS):
    return Runner(rec.send, gk or game_keys(), timing, rec, combos)


def tap(name):
    return [("down", name), ("up", name)]


def run(runner, name):
    runner.run(runner.combos[name])
    return runner


def combo_keys(rec):
    assert rec.events[:2] == [("up", "left"), ("up", "right")]  # clean start
    return rec.events[2:]


def wait_until(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


def test_back_off_facing_right_walks_left_turns_and_throws_twice():
    rec = Recorder()
    runner = run(make_runner(rec), "back_off")
    assert combo_keys(rec) == tap("left") + tap("right") + tap("shift") * 2
    assert runner.facing == RIGHT
    assert rec.steps == ["Hold ← to walk away", "Tap → to turn around", "SHIFT (Lucky Seven) ×2"]
    assert rec.dones == [("Done ✓", rec.steps)]


def test_back_off_facing_left_is_mirrored():
    rec = Recorder()
    runner = make_runner(rec)
    runner.facing = LEFT
    run(runner, "back_off")
    assert combo_keys(rec)[:4] == tap("right") + tap("left")
    assert runner.facing == LEFT


def test_without_lucky_seven_stars_are_thrown_by_jump_attack():
    rec = Recorder()
    run(make_runner(rec, game_keys(lucky_seven=None)), "back_off")
    assert combo_keys(rec)[4:] == (tap("alt") + tap("ctrl")) * 2
    assert rec.steps[-1] == "ALT + CTRL in mid-air ×2"


def test_jump_throw():
    rec = Recorder()
    run(make_runner(rec), "jump_throw")
    assert combo_keys(rec) == tap("alt") + tap("ctrl")


# Facing right: hold ← through the jump, then hold → (the turn) while the attack goes in.
SHOT_FACING_RIGHT = [("down", "left"), ("down", "alt"), ("up", "alt"), ("up", "left"),
                     ("down", "right"), ("down", "shift"), ("up", "shift"), ("up", "right")]


def test_hop_back_is_a_jump_back_shot_plus_one_more_throw():
    rec = Recorder()
    run(make_runner(rec), "hop_back")
    assert combo_keys(rec) == SHOT_FACING_RIGHT + tap("shift")


def test_jump_back_shot_jumps_away_turns_in_the_air_and_throws_at_the_monster():
    rec = Recorder()
    runner = run(make_runner(rec), "jump_back_shot")
    assert combo_keys(rec) == SHOT_FACING_RIGHT
    assert runner.facing == RIGHT
    assert rec.steps == ["Hold ← + ALT to jump away", "In the air: hold → to face them",
                         "SHIFT (Lucky Seven) while still holding →"]


def test_jump_back_shot_without_lucky_seven_throws_with_plain_attack():
    rec = Recorder()
    run(make_runner(rec, game_keys(lucky_seven=None)), "jump_back_shot")
    assert combo_keys(rec)[-3:] == tap("ctrl") + [("up", "right")]  # already airborne, so no second jump
    assert rec.steps[-1] == "CTRL while still holding →"


def test_jump_back_shot_keeps_the_turn_arrow_held_while_the_attack_starts():
    recorded = []
    t = Timing()
    rec = Recorder()
    runner = make_runner(rec, timing=t)
    # Replay on a virtual clock to check the hold lengths that in-game testing settled on.
    from maplehelper.engine import Ctx, _Silent
    class Clock(Ctx):
        now = 0
        def wait(self, ms):
            Clock.now += ms
    ctx = Clock(lambda k, d: recorded.append((Clock.now, k.name, d)), game_keys(), t, RIGHT, False,
                threading.Event(), _Silent(), "")
    runner.combos["jump_back_shot"].script(ctx)
    at = {(name, down): ms for ms, name, down in recorded}
    assert at[("alt", True)] - at[("left", True)] == t.run_up
    assert at[("left", False)] - at[("alt", True)] == t.key_press + t.air_delay
    assert at[("shift", True)] - at[("right", True)] == t.turn_lead
    assert at[("right", False)] >= at[("shift", False)]


def test_slow_mode_never_pauses_mid_air():
    rec = Recorder()
    runner = make_runner(rec, timing=Timing(**{**vars(ZERO), "slow_pause": 150}))
    runner.slow = True
    started = time.perf_counter()
    run(runner, "jump_back_shot")
    # One pause (before the jump only); the margin covers Windows timers waking a little early or late.
    assert 0.13 <= time.perf_counter() - started < 0.28


def test_abort_releases_held_keys_and_reports_stopped():
    rec = Recorder()
    runner = make_runner(rec, timing=Timing(**{**vars(ZERO), "walk_away": 5000}))
    threading.Timer(0.05, runner.abort).start()
    started = time.perf_counter()
    run(runner, "back_off")
    assert time.perf_counter() - started < 1
    assert combo_keys(rec) == tap("left")  # released, and nothing after it ran
    assert rec.dones[-1][0] == "Stopped"


def test_slow_mode_pauses_before_each_step():
    rec = Recorder()
    runner = make_runner(rec, timing=Timing(**{**vars(ZERO), "slow_pause": 30}))
    runner.slow = True
    started = time.perf_counter()
    run(runner, "back_off")
    assert time.perf_counter() - started >= 3 * 0.03 * 0.9  # margin: Windows timers can wake a little early


def hold_left(ms):
    return lambda c: c.tap(c.keys.left, ms)


def with_timing(**overrides):
    return Timing(**{**vars(ZERO), **overrides})


QUICK_COMBOS = [Combo("quick", "Quick", lambda c: c.say("quick")), Combo("other", "Other", lambda c: c.say("other"))]


def test_expected_length_adds_up_the_combo_waits():
    t = Timing()
    runner = make_runner(Recorder(), timing=t)
    two_throws_after_walking = t.walk_away + t.turn_tap + 2 * (t.key_press + t.throw_gap)
    assert runner.expected_ms(runner.combos["back_off"]) == two_throws_after_walking
    runner.slow = True  # one pause per step, 3 steps
    assert runner.expected_ms(runner.combos["back_off"]) == two_throws_after_walking + 3 * t.slow_pause


def test_default_timing_clears_the_measured_attack_lock():
    t = Timing()
    assert t.key_press + t.throw_gap >= 730 + 50  # measured lock + margin for frame jitter


def test_press_early_in_a_combo_is_dropped():
    rec = Recorder()
    runner = make_runner(rec, timing=with_timing(buffer_window=300),
                         combos=[Combo("long", "Long", hold_left(800)), *QUICK_COMBOS])
    runner.start()
    try:
        assert runner.trigger("long") == STARTED
        assert runner.trigger("quick") == DROPPED  # before the combo even started
        wait_until(lambda: ("down", "left") in rec.events)
        assert runner.trigger("quick") == DROPPED
        wait_until(lambda: len(rec.dones) == 1)
        time.sleep(0.05)
        assert rec.dones == [("Done ✓", [])]
    finally:
        runner.stop()


def test_press_near_the_end_is_queued_and_the_newest_press_wins():
    rec = Recorder()
    runner = make_runner(rec, timing=with_timing(buffer_window=300),
                         combos=[Combo("long", "Long", hold_left(600)), *QUICK_COMBOS])
    runner.start()
    try:
        runner.trigger("long")
        wait_until(lambda: ("down", "left") in rec.events)
        time.sleep(0.4)  # ~200 ms left
        assert runner.trigger("quick") == QUEUED
        assert runner.trigger("other") == QUEUED  # replaces "quick"
        wait_until(lambda: len(rec.dones) == 2)
        time.sleep(0.05)  # nothing else may run
        assert rec.dones == [("Done ✓", []), ("Done ✓", ["other"])]
    finally:
        runner.stop()


def test_esc_stops_the_combo_and_forgets_the_queued_press():
    rec = Recorder()
    runner = make_runner(rec, timing=with_timing(buffer_window=5000),
                         combos=[Combo("long", "Long", hold_left(3000)), *QUICK_COMBOS])
    runner.start()
    try:
        runner.trigger("long")
        wait_until(lambda: ("down", "left") in rec.events)
        assert runner.trigger("quick") == QUEUED
        runner.abort()
        wait_until(lambda: len(rec.dones) == 1)
        time.sleep(0.05)
        assert rec.dones == [("Stopped", [])]
        assert runner.trigger("quick") == STARTED  # idle again
        wait_until(lambda: len(rec.dones) == 2)
        assert rec.dones[-1] == ("Done ✓", ["quick"])
    finally:
        runner.stop()


def test_chained_jump_back_shots_both_jump_away_from_the_monster():
    rec = Recorder()
    runner = make_runner(rec, timing=with_timing(throw_gap=300, buffer_window=200))
    runner.start()
    try:
        runner.trigger("jump_back_shot")
        time.sleep(0.2)  # pressed again as she lands
        assert runner.trigger("jump_back_shot") == QUEUED
        wait_until(lambda: len(rec.dones) == 2)
        clean_start = [("up", "left"), ("up", "right")]
        assert rec.events == (clean_start + SHOT_FACING_RIGHT) * 2
    finally:
        runner.stop()


def turn_left_then_wait(c):
    c.tap(c.keys.left)
    c.wait(300)


def wait_then_turn_left(c):
    c.wait(300)
    c.tap(c.keys.left)


def test_combo_that_turns_her_updates_facing():
    rec = Recorder()
    runner = make_runner(rec, combos=[Combo("turn", "Turn", turn_left_then_wait)])
    runner.run(runner.combos["turn"])
    assert runner.facing == LEFT


def test_her_arrow_press_during_a_combo_wins_over_an_earlier_combo_turn():
    rec = Recorder()
    runner = make_runner(rec, combos=[Combo("turn", "Turn", turn_left_then_wait)])
    threading.Timer(0.1, runner.note_facing, args=(RIGHT,)).start()
    runner.run(runner.combos["turn"])
    assert runner.facing == RIGHT


def test_combo_turn_after_her_arrow_press_wins():
    rec = Recorder()
    runner = make_runner(rec, combos=[Combo("turn", "Turn", wait_then_turn_left)])
    threading.Timer(0.1, runner.note_facing, args=(RIGHT,)).start()
    runner.run(runner.combos["turn"])
    assert runner.facing == LEFT


def test_her_arrow_press_during_a_combo_that_never_turns_is_kept():
    rec = Recorder()
    runner = make_runner(rec, combos=[Combo("wait", "Wait", lambda c: c.wait(300))])
    threading.Timer(0.1, runner.note_facing, args=(LEFT,)).start()
    runner.run(runner.combos["wait"])
    assert runner.facing == LEFT


class FailingKeyboard(Recorder):
    """Fails when pressing attack, like SendInput failing mid-combo."""

    def send(self, key, down):
        if key.name == "ctrl" and down:
            raise OSError("SendInput failed")
        super().send(key, down)


def test_a_combo_failing_mid_way_still_releases_keys_and_the_runner_keeps_working():
    rec = FailingKeyboard()
    runner = make_runner(rec, combos=[Combo("jump_throw", "Jump", lambda c: c.jump_attack()),
                                      Combo("ok", "OK", lambda c: None)])
    runner.start()
    try:
        runner.trigger("jump_throw")
        wait_until(lambda: len(rec.dones) == 1)
        assert combo_keys(rec) == tap("alt")  # jump went out and was released; attack failed
        assert rec.dones[0][0] == "Error, see console"
        time.sleep(0.05)
        assert runner.trigger("ok") == STARTED
        wait_until(lambda: len(rec.dones) == 2)
    finally:
        runner.stop()


def test_a_broken_combo_script_fails_before_pressing_any_key():
    rec = Recorder()

    def broken(c):
        c.down(c.keys.jump)
        raise RuntimeError("boom")

    runner = make_runner(rec, combos=[Combo("broken", "Broken", broken), Combo("ok", "OK", lambda c: None)])
    runner.start()
    try:
        runner.trigger("broken")
        wait_until(lambda: len(rec.dones) == 1)
        assert combo_keys(rec) == []  # caught by the timing dry run
        assert rec.dones[0][0] == "Error, see console"
    finally:
        runner.stop()
