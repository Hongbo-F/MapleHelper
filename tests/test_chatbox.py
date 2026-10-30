"""Her chat box collapsed at a hunt's start if it's expanded (the user, 2026-10-30 ~12:45: "At the start of grind/hunt,
you should collapse the chat window if it's open"). The reader (perception.chatbox) on real crops of its input row
(tests/fixtures/chatbox: BugBaby's grind of 2026-10-30 08:17:50 expanded then collapsed by the user, her play by hand,
RiceBaby's HUD capture, two Lie Detector recordings' JPEG frames), its button deciding whatever is over the box; a look
and the world state carrying it. The check (behavior.chatcheck) with a made-up hunter and box: her chat key once when
expanded at a quiet look, never when collapsed, not told or its input line typed in, never with the Lie Detector alarm
up, once more at most; put off near a monster. In the simulator (sim.world.Realism.chat): nothing read by default, the
start's collapse, the hunt loop's after a start put off, the key that does nothing pressed twice."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from maplehelper.config import GameKeys
from maplehelper.keys import parse as parse_key

from gamelab import paths
from gamelab.behavior import chatcheck
from gamelab.behavior.chatcheck import ChatCheck, ChatDue, chat_key
from gamelab.control import client
from gamelab.localization.worldstate import WorldState, to_scene
from gamelab.perception import chatbox
from gamelab.perception.chatbox import COLLAPSED, EXPANDED, UNKNOWN, ChatBox
from gamelab.sim import harness, scenarios
from gamelab.sim.world import LEVEL_25, Realism

FIXTURES = Path(__file__).parent / "fixtures" / "chatbox"
NOTES = json.loads((FIXTURES / "chatbox.json").read_text(encoding="utf-8"))
W, H = chatbox.FRAME


def real(name: str, base: np.ndarray | None = None) -> np.ndarray:
    """A fixture crop of the box's input row put back where it was in a frame of the game's own size (mid gray around
    it, or `base`)."""
    n = NOTES[name]
    crop = np.ascontiguousarray(cv2.imread(str(FIXTURES / n["file"]))[..., ::-1])
    fr = np.full((H, W, 3), 90, np.uint8) if base is None else base.copy()
    x0, y0 = n["at"]
    fr[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return fr


# --- the reader -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(NOTES))
def test_each_real_crop_reads_as_its_frame_did(name):
    """Expanded (four lines of megaphones; the party invite over the box; the Lie Detector's window over its lines in a
    JPEG; the magician's), collapsed (the bot's grind after the user's ', her play by hand), not told (the mouse pointer
    over the button; the UI greyed in a JPEG)."""
    got = chatbox.read(real(name))
    assert got.state == NOTES[name]["state"] and not got.typing, (name, got)


def test_the_button_decides_not_the_box_above_it():
    """What's over the box's lines (a megaphone's teal and pink backing painted where the expanded box would be, a
    notice's white flash, a black screen) doesn't change what its button and input row say."""
    for name, want in (("collapsed_bugbaby", COLLAPSED), ("expanded_bugbaby", EXPANDED)):
        for paint in ((90, 210, 190), (230, 160, 190), (255, 255, 255), (0, 0, 0)):
            fr = real(name)
            fr[560:700, 280:852] = paint
            fr[633:700, 835:849] = (200, 200, 210)  # (a scrollbar's track where the expanded box has one)
            assert chatbox.read(fr).state == want, (name, paint)


def test_text_or_a_caret_in_its_input_line_is_typing():
    """A caret (a dark column 12 px high) or letters in the expanded box's white field: its input line may have the
    keyboard (`typing`); collapsed with the field white (the input line open without the box): typing too."""
    fr = real("expanded_bugbaby")
    fr[709:721, 380] = (0, 0, 0)
    assert chatbox.read(fr) == ChatBox(EXPANDED, typing=True)
    fr = real("expanded_bugbaby")
    fr[711:719, 400:460:3] = (30, 30, 30)
    assert chatbox.read(fr).typing
    fr = real("collapsed_bugbaby")
    fr[705:726, 370:810] = 255
    assert chatbox.read(fr) == ChatBox(COLLAPSED, typing=True)
    assert chatbox.read(real("expanded_bugbaby")) == ChatBox(EXPANDED)


def test_not_told_at_another_size_or_without_a_frame_or_its_row():
    assert chatbox.read(np.zeros((1080, 1920, 3), np.uint8)) == ChatBox(
        UNKNOWN, why="the client is 1920x1080, not 1366x768")
    assert chatbox.read(None).state == UNKNOWN
    assert chatbox.read(np.full((H, W, 3), 90, np.uint8)) == ChatBox(UNKNOWN, why="its button not seen")
    fr = real("expanded_bugbaby")
    fr[705:726, 370:810] = (119, 119, 119)  # (the minus over a gray row: neither state's)
    got = chatbox.read(fr)
    assert got.state == UNKNOWN and got.why.startswith("its button's minus, its input row not seen")
    assert ChatBox(EXPANDED, typing=True).words() == "expanded, its input line with text or a caret"


def test_a_look_and_the_world_state_carry_it(monkeypatch):
    """Eyes.look_at reads it off every frame (a saved frame of the game's own size: collapsed; its input row expanded),
    the world state's step keeps it (WorldState.chat) and to_scene gives it back (Scene.chat)."""
    from PIL import Image

    from gamelab.analysis import bench
    from gamelab.screen import NATIVE
    native = Path(__file__).parent / "fixtures" / "native"
    tag = np.load(native / "tag_template_1366x768.npy")
    eyes = bench.SavedEyes(tag, detector=object(), geo=NATIVE)
    monkeypatch.setattr(eyes.name_tag, "full", tag)
    full = np.asarray(Image.open(native / "n28_000.png").convert("RGB"))
    assert eyes.look_at(full).chat == ChatBox(COLLAPSED)
    obs = eyes.look_at(real("expanded_bugbaby", full))
    assert obs.chat == ChatBox(EXPANDED) and obs.bad is None
    st = WorldState(1, 1.0, True, chat=obs.chat)
    assert to_scene(st, None, (100, 100), mobs=False).chat == ChatBox(EXPANDED)
    assert to_scene(WorldState(2, 2.0, True), None, (100, 100), mobs=False).chat is None


def test_the_world_step_keeps_each_frames():
    from test_worldstate import Drawn, small_map

    from gamelab.localization.senses import Senses
    from gamelab.localization.worldstate import WorldStep
    model = small_map()
    eyes = Drawn(model)
    step = WorldStep(Senses(eyes, model, band=(100, 100)))
    box = ChatBox(EXPANDED)
    st = step.step(lambda ropes: dataclasses.replace(eyes.frame(100.0, ropes), chat=box), 100.0)
    assert st.chat == box and to_scene(st, model, (100, 100)).chat == box
    assert step.step(lambda ropes: eyes.frame(100.1, ropes), 100.1).chat is None


# --- the check, with a made-up hunter -------------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += s

    def strftime(self, *a):  # (telemetry's, should it ask)
        return "00:00:00"


class Box:
    """A chat box the made-up hunter's looks see: `state`, toggled by her chat key unless `stuck`; `typing`; `unknown`
    for this many looks first; `after_key`: what the key leaves it as, if not toggled."""

    def __init__(self, state, stuck=False, typing=False, unknown=0, after_key=None):
        self.state, self.stuck, self.typing, self.unknown, self.after_key = state, stuck, typing, unknown, after_key

    def key(self):
        if self.after_key is not None:
            self.state, self.typing = self.after_key
        elif not self.stuck:
            self.state = COLLAPSED if self.state == EXPANDED else EXPANDED

    def read(self):
        if self.unknown:
            self.unknown -= 1
            return ChatBox(UNKNOWN, why="its button not seen")
        return None if self.state is None else ChatBox(self.state, typing=self.typing)


class Hunter:
    def __init__(self, box, clock, quiet=(None,), near=False, alarm_at=None):
        self.box, self.clock = box, clock
        self.quiet = list(quiet)  # unquiet's answers look by look (the last kept)
        self.near = near
        self.alarm_at = alarm_at  # the alarm up from this look on (its index), or from the key (-1)
        self.looks, self.lines, self.keys = 0, [], []
        self.actions = SimpleNamespace(keys=None, play=self._play)
        self.doubts_raise = False
        self.model = None
        self.hp_watch = None

    def log(self, text):
        self.lines.append(text)

    def look(self):
        if self.alarm_at is not None and 0 <= self.alarm_at <= self.looks:
            raise client.LieDetectorAlarm("Lie Detector seen at 00:00:00")
        self.looks += 1
        self.clock.now += 0.034
        return SimpleNamespace(chat=self.box.read(), t=self.clock.now, where=object())

    def _play(self, events, name, kind=None):
        assert name == "chat_collapse" and kind is None
        self.keys.append([e[1] for e in events if e[2] == "down"])
        self.looks_at_key = self.looks
        self.clock.now += 0.05
        self.box.key()


@pytest.fixture
def made(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(chatcheck, "time", clock)
    events = []
    monkeypatch.setattr(chatcheck.telemetry, "emit", lambda *a, **k: events.append(k))
    alarm = {"up": False}

    def check_alarm():
        if alarm["up"]:
            raise client.LieDetectorAlarm("Lie Detector seen at 00:00:00")
    monkeypatch.setattr(client, "check_alarm", check_alarm)

    def make(box, **kw):
        h = Hunter(box, clock, **kw)
        answers = h.quiet
        monkeypatch.setattr(chatcheck, "unquiet", lambda hh, s: answers.pop(0) if len(answers) > 1 else answers[0])
        monkeypatch.setattr(chatcheck, "near_her", lambda s, model=None: ["green-mushroom"] if h.near else [])
        return h, ChatCheck(h)
    return SimpleNamespace(make=make, clock=clock, events=events, alarm=alarm)


def test_expanded_her_chat_key_once_and_seen_collapsed(made):
    h, c = made.make(Box(EXPANDED))
    out = c.run(at_start=True)
    assert out.what == "collapsed" and out.sent == ("'",) and h.keys == [["'"]]
    assert h.lines == ["  the chat box expanded: collapsed (')"]
    assert made.events[-1]["kind"] == "chat_box" and made.events[-1]["what"] == "collapsed"
    assert c.run().what == "skipped" and h.keys == [["'"]]  # (once a run)


def test_collapsed_or_not_told_or_not_read_nothing_pressed(made):
    """Collapsed: said, nothing sent (the key toggles: pressed it would expand the box). Not told on any look within
    QUIET_S (its button covered): said, nothing sent. No state of it (the simulator's eyes): nothing said."""
    h, c = made.make(Box(COLLAPSED))
    assert c.run(at_start=True).what == "was collapsed" and h.keys == [] and h.lines == ["  the chat box: collapsed"]
    h, c = made.make(Box(EXPANDED, unknown=10 ** 6))
    out = c.run(at_start=True)
    assert out.what == "not told" and h.keys == []
    assert h.lines == ["  the chat box not told (its button not seen): nothing sent"]
    h, c = made.make(Box(None))
    assert c.run(at_start=True).what == "not read" and h.keys == [] and h.lines == []
    # Not told at first, expanded on a later look: pressed then
    h, c = made.make(Box(EXPANDED, unknown=2))
    assert c.run(at_start=True).what == "collapsed" and h.keys == [["'"]]


def test_its_input_line_typed_in_nothing_pressed(made):
    """Its input line with text or a caret (it may have the keyboard): left so, nothing sent (no Esc, no Enter)."""
    h, c = made.make(Box(EXPANDED, typing=True))
    out = c.run(at_start=True)
    assert out.what == "left" and h.keys == []
    assert "its input line with text or a caret" in h.lines[0] and h.lines[0].endswith("left so, nothing sent")
    # ... and after the key (the key typed into it): left so, not pressed again
    h, c = made.make(Box(EXPANDED, after_key=(EXPANDED, True)))
    out = c.run(at_start=True)
    assert out.what == "left" and h.keys == [["'"]] and "the key typed into it?" in h.lines[-1]


def test_still_expanded_once_more_then_left_so(made):
    """The key doing nothing: once more at the next quiet look, then "still expanded after ' twice: left so"; never a
    third."""
    h, c = made.make(Box(EXPANDED, stuck=True))
    out = c.run(at_start=True)
    assert out.what == "left" and h.keys == [["'"], ["'"]] and out.sent == ("'", "'")
    assert h.lines == ["  the chat box still expanded after ': again",
                       "  the chat box still expanded after ' twice: left so"]
    assert c.run().what == "skipped" and len(h.keys) == 2


def test_judged_only_on_a_frame_from_after_the_key(made):
    """A look whose frame is from under SETTLE_S after the key isn't taken as what the key did (the box redrawn on the
    next frame, a frame from before it still expanded): no second press for it."""
    h, c = made.make(Box(EXPANDED))
    first = h.box.read
    shown = {"n": 0}

    def late_read():  # (the first look after the key still shows it expanded, as a frame from before it would)
        if h.keys and shown["n"] < 1:
            shown["n"] += 1
            return ChatBox(EXPANDED)
        return first()
    h.box.read = late_read
    assert c.run(at_start=True).what == "collapsed" and h.keys == [["'"]]


def test_never_with_the_lie_detector_alarm_up(made):
    """The alarm up at the start (Hunter.look raises it) or just before the key (client.check_alarm): nothing sent,
    LieDetectorAlarm raised (the hunt's retreat, or the command's stop at its start)."""
    h, c = made.make(Box(EXPANDED), alarm_at=0)
    with pytest.raises(client.LieDetectorAlarm):
        c.run(at_start=True)
    assert h.keys == []
    made.alarm["up"] = True
    h, c = made.make(Box(EXPANDED))
    with pytest.raises(client.LieDetectorAlarm):
        c.run(at_start=True)
    assert h.keys == []


def test_put_off_near_a_monster_then_at_a_quiet_look_in_the_hunt_loop(made):
    """A monster within PLACE_CLEAR at the start: put off at once, nothing sent; due again DEFER_GAP_S on (Hunter.
    health_check raises ChatDue at a quiet look in the hunt loop), then collapsed. Not quiet for QUIET_S (in the air):
    put off too. After DEFERS_MAX put-offs, left so."""
    h, c = made.make(Box(EXPANDED), quiet=("green-mushroom within 250 of her",), near=True)
    out = c.run(at_start=True)
    assert out.what == "put off" and h.keys == [] and c.pending and h.looks == 1
    assert h.lines == ["  the chat box expanded: its collapse put off (green-mushroom within 250 of her): again at a "
                       "quiet moment in the hunt"]
    assert not c.again(made.clock.now + 0.5) and c.again(made.clock.now + chatcheck.DEFER_GAP_S)
    h.near, h.quiet[:] = False, [None]
    made.clock.now += chatcheck.DEFER_GAP_S
    looks = h.looks
    assert c.run(look=SimpleNamespace(chat=ChatBox(EXPANDED), t=made.clock.now)).what == "collapsed"
    assert h.keys == [["'"]] and not c.pending and h.looks_at_key == looks  # (ChatDue's look first: the key at once)
    # not on a platform the whole QUIET_S: put off
    h, c = made.make(Box(EXPANDED), quiet=("not on a platform",))
    assert c.run(at_start=True).what == "put off" and h.keys == []
    h.doubts_raise = True  # (the hunt loop's runs)
    for _ in range(chatcheck.DEFERS_MAX):
        made.clock.now += chatcheck.DEFER_GAP_S
        c.run()
    assert c.done and not c.pending and h.keys == [] and h.lines[-1].endswith(
        f"; {chatcheck.DEFERS_MAX} times put off): left so this run")


def test_her_chat_key_from_her_remembered_bindings_else_the_quote(tmp_path, monkeypatch):
    """' where nothing remembered says otherwise; CHAT+'s key in her remembered key bindings; refused where it's one of
    her own keys or her bindings have CHAT+ on no key (statcheck.bound_key, as her stats key)."""
    monkeypatch.setattr(paths, "LAB", tmp_path)
    keys = GameKeys(left=parse_key("left"), right=parse_key("right"), jump=parse_key("alt"), attack=parse_key("ctrl"))
    assert chat_key(keys) == ("'", "', CHAT+'s in her key bindings of 2026-10-28 (none remembered since)")
    (tmp_path / "keybindings-thief.json").write_text(json.dumps({"read_at": "2026-10-30 08:00:00", "functions": {
        "ctrl": "ATTACK", "'": "CHAT+", "s": "ABILITY"}}), encoding="utf-8")
    assert chat_key(keys) == ("'", "CHAT+'s in her remembered key bindings (keybindings-thief.json, read 2026-10-30 "
                                   "08:00:00)")
    (tmp_path / "keybindings-thief.json").write_text(json.dumps({"functions": {"ctrl": "CHAT+"}}), encoding="utf-8")
    key, why = chat_key(keys)
    assert key is None and why.endswith("is her attack key too: not pressed for it")
    (tmp_path / "keybindings-thief.json").write_text(json.dumps({"functions": {"ctrl": "ATTACK"}}), encoding="utf-8")
    assert chat_key(keys) == (None, "CHAT+ on no key in her remembered key bindings (keybindings-thief.json, read ?)")


def test_the_quote_reaches_the_game():
    """The elevated player presses it (every key config.toml can name: control.player.ALL_KEYS) as the game's quote key
    (VK_OEM_7); not one the alarm lets through."""
    from maplehelper.keys import KEY_CODES

    from gamelab.control import alarm, player
    assert "'" in player.allowed_keys() and KEY_CODES["'"] == (0xDE, False)
    player.validate({"events": [[0, "'", "down"], [40, "'", "up"]]}, player.allowed_keys())
    assert player.alarm_refusal({"events": [[0, "'", "down"], [40, "'", "up"]], "kind": None}, None, 1.0, [], 0.0)
    assert "chat" not in alarm.PLAN_KINDS


# --- in the simulator -----------------------------------------------------------------------------------------------

TREE = scenarios.TREE
REAL = pytest.mark.skipif(not scenarios.available(TREE), reason="needs Tree Dungeon's game data (gamedata/data)")
QUIET = set(range(0, 17))  # (the map's monsters on 0-16 only: nothing near her on 23 at the start)


def _sim(fn, monsters=QUIET, **realism):
    """`fn(sim)` in the simulator on Tree Dungeon from 23: what it gave, her chat key's presses, the log's lines of the
    chat box, the world's summary (its "chat_box" and "chat_presses" with Realism.chat set)."""
    got = {}

    def run(sim):
        got.update(fn(sim) or {})
        sim.world.advance(sim.clock.now)
        got["keys"] = [k for _, name, events, _ in sim.trace.plans if name == "chat_collapse"
                       for k in (e[1] for e in events if len(e) == 3 and e[2] == "down")]
        got["log"] = [line for line in sim.trace.log if "chat box" in line]
        got["world"] = sim.world.summary()
        return {}
    trace, _ = scenarios.run_sim(scenarios.Scenario(
        "chat_test", run, (23, 73.0), monsters=monsters, seed=1, limit=300, map_id=TREE, attack="choice",
        her=dataclasses.replace(LEVEL_25, **harness.ATTACK_HER), realism=Realism(zero_walls=True, **realism)))
    assert trace["result"] == {}, trace["result"]
    return got


@REAL
def test_in_the_simulator_nothing_by_default():
    """The simulator's eyes read no chat box (Realism.chat None): the start's check does nothing, says nothing."""
    got = _sim(lambda sim: {"out": sim.tour_stack().h.collapse_chat(at_start=True).what})
    assert got["out"] == "not read" and got["keys"] == [] and got["log"] == [] and "chat_box" not in got["world"]


@REAL
def test_in_the_simulator_collapsed_at_the_start():
    got = _sim(lambda sim: {"out": sim.tour_stack().h.collapse_chat(at_start=True).what}, chat=EXPANDED)
    assert got["out"] == "collapsed" and got["keys"] == ["'"]
    assert got["world"]["chat_box"] == COLLAPSED and got["world"]["chat_presses"] == 1
    assert any(line.endswith("the chat box expanded: collapsed (')") for line in got["log"])


@REAL
def test_in_the_simulator_stuck_twice_then_left():
    got = _sim(lambda sim: {"out": sim.tour_stack().h.collapse_chat(at_start=True).what}, chat=EXPANDED,
               chat_stuck=True)
    assert got["out"] == "left" and got["keys"] == ["'", "'"] and got["world"]["chat_presses"] == 2
    assert got["log"][-1].endswith("the chat box still expanded after ' twice: left so")


@REAL
def test_in_the_simulator_put_off_at_the_start_collapsed_in_the_hunt_loop():
    """Every spawn point filled: a monster within 250 of her at the start, put off; the grind's hunt loop collapses it
    at its first quiet look DEFER_GAP_S on (ChatDue from health_check, that look the collapse's first: 11.0 s in, the
    key at 11.4; with its own look taken anew a Green Mushroom had come within 250, put off again till 67 s)."""
    from gamelab.behavior.grind import Grind
    from gamelab.behavior.retreat import on_alarm

    def fn(sim):
        tour = sim.tour_stack()
        start = tour.h.collapse_chat(at_start=True).what
        with on_alarm(tour.h):
            Grind(tour, log=sim.log).run(minutes=0.4)
        return {"start": start, "last": tour.h.chat_check.last.what}
    got = _sim(fn, monsters="all", chat=EXPANDED)
    assert got["start"] == "put off" and got["last"] == "collapsed" and got["keys"] == ["'"]
    assert any("the chat box's collapse put off: a quiet moment now" in line for line in got["log"])
    assert got["world"]["chat_box"] == COLLAPSED


def test_chat_due_is_not_a_hunt_stop():
    from gamelab.behavior.safety import HuntStop
    assert not issubclass(ChatDue, HuntStop)
