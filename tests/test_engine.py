import threading
import time
from dataclasses import fields

from maplehelper import keys
from maplehelper.combos import COMBOS
from maplehelper.config import GameKeys, Timing
from maplehelper.engine import LEFT, RIGHT, Combo, Runner

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


def test_hop_back_holds_away_arrow_through_the_jump():
    rec = Recorder()
    run(make_runner(rec), "hop_back")
    assert combo_keys(rec)[:4] == [("down", "left"), ("down", "alt"), ("up", "alt"), ("up", "left")]


def test_jump_back_shot_jumps_away_turns_in_the_air_and_throws_at_the_monster():
    rec = Recorder()
    runner = run(make_runner(rec), "jump_back_shot")
    assert combo_keys(rec) == [("down", "left"), ("down", "alt"), ("up", "alt"), ("up", "left"),
                               ("down", "right"), ("up", "right"), ("down", "shift"), ("up", "shift")]
    assert runner.facing == RIGHT
    assert rec.steps == ["Hold ← + ALT to jump away", "In the air: tap → to face them",
                         "SHIFT (Lucky Seven) to throw before landing"]


def test_jump_back_shot_without_lucky_seven_throws_with_plain_attack():
    rec = Recorder()
    run(make_runner(rec, game_keys(lucky_seven=None)), "jump_back_shot")
    assert combo_keys(rec)[-2:] == tap("ctrl")  # already airborne, so no second jump
    assert rec.steps[-1] == "CTRL to throw before landing"


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


def test_busy_runner_ignores_new_combos_until_the_current_one_ends():
    rec = Recorder()
    combos = [
        Combo("long", "Long", lambda c: c.tap(c.keys.left, 5000)),
        Combo("quick", "Quick", lambda c: c.say("quick")),
    ]
    runner = make_runner(rec, combos=combos)
    runner.start()
    try:
        assert runner.trigger("long")
        wait_until(lambda: ("down", "left") in rec.events)
        assert not runner.trigger("quick")
        assert not runner.trigger("long")
        runner.abort()
        wait_until(lambda: len(rec.dones) == 1)
        assert rec.dones == [("Stopped", [])]
        wait_until(lambda: runner.trigger("quick"))  # idle again
        wait_until(lambda: len(rec.dones) == 2)
        assert rec.dones[-1] == ("Done ✓", ["quick"])
    finally:
        runner.stop()


def test_a_failing_combo_still_releases_keys_and_the_runner_keeps_working():
    rec = Recorder()

    def broken(c):
        c.down(c.keys.jump)
        raise RuntimeError("boom")

    runner = make_runner(rec, combos=[Combo("broken", "Broken", broken), Combo("ok", "OK", lambda c: None)])
    runner.start()
    try:
        runner.trigger("broken")
        wait_until(lambda: len(rec.dones) == 1)
        assert ("up", "alt") in rec.events
        assert rec.dones[0][0] == "Error, see console"
        wait_until(lambda: runner.trigger("ok"))
        wait_until(lambda: len(rec.dones) == 2)
    finally:
        runner.stop()
