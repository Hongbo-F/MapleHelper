"""Her chat box collapsed at a hunt's start if it's expanded (the user, 2026-10-30 ~12:45: "At the start of grind/hunt,
you should collapse the chat window if it's open"). The reader (perception.chatbox) on real crops of the box and its
input row (tests/fixtures/chatbox: BugBaby's grind of 2026-10-30 08:17:50 expanded then collapsed by the user, her play
by hand, RiceBaby's HUD capture, a hunt's start with it expanded, two Lie Detector recordings' JPEG frames; the mouse
pointer cut from frame 592), its button deciding whatever is over the box; a look and the world state carrying it. The
check (behavior.chatcheck) with a made-up hunter and box: her chat key once when expanded at a quiet look after its
field was seen unmarked a while, never when collapsed, not told or its input line typed in, never with the Lie Detector
alarm up, once more at most and not on a look the game may not have redrawn; the mouse moved off the box once where it
doesn't tell; put off near a monster, not told or with no look, and left so where no hunt loop follows; a perception
service of the code before said. In the simulator (sim.world.Realism.chat): nothing read by default, the start's
collapse, the hunt loop's after a start put off, the key that does nothing pressed twice."""

import dataclasses
import hashlib
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
from gamelab.perception.framecheck import LookLost, Problem
from gamelab.sim import harness, scenarios
from gamelab.sim.world import LEVEL_25, Realism

FIXTURES = Path(__file__).parent / "fixtures" / "chatbox"
NOTES = json.loads((FIXTURES / "chatbox.json").read_text(encoding="utf-8"))
CROPS = sorted(k for k in NOTES if not k.startswith("_"))
W, H = chatbox.FRAME


def real(name: str, base: np.ndarray | None = None) -> np.ndarray:
    """A fixture crop of the box and its input row put back where it was in a frame of the game's own size (mid gray
    around it, or `base`)."""
    n = NOTES[name]
    crop = np.ascontiguousarray(cv2.imread(str(FIXTURES / n["file"]))[..., ::-1])
    fr = np.full((H, W, 3), 90, np.uint8) if base is None else base.copy()
    x0, y0 = n["at"]
    fr[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return fr


def pointer(fr: np.ndarray, x: int, y: int) -> np.ndarray:
    """The mouse pointer of frame 592 (its RGBA cut) drawn on a copy of `fr`, its top-left at (x, y)."""
    p = cv2.imread(str(FIXTURES / NOTES["_pointer"]["file"]), cv2.IMREAD_UNCHANGED)
    rgb, on = np.ascontiguousarray(p[..., 2::-1]), p[..., 3] > 0
    fr = fr.copy()
    region = fr[y:y + rgb.shape[0], x:x + rgb.shape[1]]
    m = on[:region.shape[0], :region.shape[1]]
    region[m] = rgb[:region.shape[0], :region.shape[1]][m]
    return fr


# --- the reader -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", CROPS)
def test_each_real_crop_reads_as_its_frame_did(name):
    """Expanded (megaphones' lines; the party invite over the box twice; the magician's; a hunt's start; the Lie
    Detector's window over its lines in a JPEG), collapsed (the bot's grind after the user's ', her play by hand), not
    told (the mouse pointer over the button; the UI greyed in a JPEG). Each crop's field unmarked."""
    got = chatbox.read(real(name))
    assert got.state == NOTES[name]["state"] and not got.typing and got.marks == 0, (name, got)


def test_the_crops_are_each_their_own_frames():
    """The review of 2026-10-30: crops of the input row alone were byte-identical across frames whose field was empty
    (the row is the UI's own px), so the "four ways" were two; the box's lines above make each its frame's."""
    digests = {hashlib.md5((FIXTURES / NOTES[n]["file"]).read_bytes()).hexdigest() for n in CROPS}
    assert len(digests) == len(CROPS)
    rows = {real(n)[709:728, 278:872].tobytes() for n in CROPS if NOTES[n]["state"] == EXPANDED
            and "jpeg" not in n}
    assert len(rows) == 1  # (and the input rows of the lossless expanded ones are indeed one picture, the invite's
    #                        bottom edge over its top two rows aside)


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
    """A caret (a dark column 12 px high), letters, a lone apostrophe (1 x 3 px) or gray anti-aliased letters in the
    expanded box's white field: its input line may have the keyboard (`typing`; `marks` its px not white); one stray px
    is not. Collapsed with the field white (the input line open without the box): typing too."""
    def read(paint):
        fr = real("expanded_bugbaby")
        paint(fr)
        return chatbox.read(fr)
    caret = read(lambda fr: fr.__setitem__((slice(709, 721), 380), 0))
    assert caret.state == EXPANDED and caret.typing and caret.marks == 12
    assert read(lambda fr: fr.__setitem__((slice(711, 719), slice(400, 460, 3)), 30)).typing
    assert read(lambda fr: fr.__setitem__((slice(710, 713), 380), 0)).marks == 3  # (a lone apostrophe)
    assert read(lambda fr: fr.__setitem__((slice(710, 713), 380), 0)).typing
    assert read(lambda fr: fr.__setitem__((slice(711, 719), slice(400, 460, 3)), 140)).typing  # (gray letters)
    one = read(lambda fr: fr.__setitem__((715, 500), 0))
    assert one.state == EXPANDED and not one.typing and one.marks == 1
    fr = real("collapsed_bugbaby")
    fr[705:726, 370:810] = 255
    assert chatbox.read(fr) == ChatBox(COLLAPSED, typing=True)
    assert chatbox.read(real("expanded_bugbaby")) == ChatBox(EXPANDED)


def test_the_mouse_pointer_over_its_field_or_button():
    """The real pointer (frame 592) over the expanded field marks it (typing: the check moves the mouse off before it
    believes it), over the button hides its glyph (unknown); over the box's lines it changes nothing."""
    exp = real("expanded_bugbaby")
    over_field = [chatbox.read(pointer(exp, x, y)) for x in range(372, 790, 24) for y in (700, 706, 712)]
    assert all(r.state == EXPANDED and r.typing for r in over_field)
    assert chatbox.read(pointer(exp, 816, 707)) == ChatBox(UNKNOWN, why="its button not seen")
    for x in range(300, 800, 50):
        assert chatbox.read(pointer(exp, x, 640)) == ChatBox(EXPANDED)
        assert chatbox.read(pointer(real("collapsed_bugbaby"), x, 640)) == ChatBox(COLLAPSED)
    x, y = NOTES["_pointer"]["at"]  # (put back where it was on a frame with the box collapsed: 592's reading)
    assert chatbox.read(pointer(real("collapsed_by_hand"), x, y)).state == UNKNOWN


def test_not_told_at_another_size_or_without_a_frame_or_its_row():
    assert chatbox.read(np.zeros((1080, 1920, 3), np.uint8)) == ChatBox(
        UNKNOWN, why="the client is 1920x1080, not 1366x768", readable=False)
    assert chatbox.read(None) == ChatBox(UNKNOWN, why="no frame", readable=False)
    assert chatbox.read(np.full((H, W, 3), 90, np.uint8)) == ChatBox(UNKNOWN, why="its button not seen")
    fr = real("expanded_bugbaby")
    fr[705:726, 370:810] = (119, 119, 119)  # (the minus over a gray row: neither state's)
    got = chatbox.read(fr)
    assert got.state == UNKNOWN and got.readable and got.why.startswith("its button's minus, its input row not seen")
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
    expanded = full.copy()
    expanded[700:728, 278:872] = real("expanded_bugbaby")[700:728, 278:872]  # (its input row: her tag left as it is)
    obs = eyes.look_at(expanded)
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
    """A chat box the made-up hunter's looks see: `state`, toggled by her chat key unless `stuck` (`delay` s after
    it: a game slow to act on it); `typing` (or `marks`); `unknown` for this many looks first; `covered`: unknown, or
    with `typing` its field marked, until the mouse moves off it (the pointer over it); `after_key`: what the key leaves
    it as (state, typing, marks), if not toggled; `blink`: a caret on every look whose index is odd."""

    def __init__(self, state, stuck=False, typing=False, unknown=0, after_key=None, covered=False, delay=0.0,
                 marks=0, blink=False):
        self.state, self.stuck, self.typing, self.unknown, self.after_key = state, stuck, typing, unknown, after_key
        self.covered, self.delay, self.marks, self.blink = covered, delay, marks, blink
        self.toggle_at = None
        self.n = 0

    def key(self, now):
        if self.after_key is not None:
            self.state, self.typing, self.marks = self.after_key
        elif not self.stuck:
            self.toggle_at = now + self.delay

    def mouse(self):
        self.covered = False

    def read(self, now):
        self.n += 1
        if self.toggle_at is not None and now >= self.toggle_at:
            self.state, self.toggle_at = COLLAPSED if self.state == EXPANDED else EXPANDED, None
        if self.unknown:
            self.unknown -= 1
            return ChatBox(UNKNOWN, why="its button not seen")
        if self.covered and not self.typing:
            return ChatBox(UNKNOWN, why="its button not seen")
        if self.state is None:
            return None
        if self.state == EXPANDED and self.blink and self.n % 2 == 0:
            return ChatBox(EXPANDED, typing=True, marks=12)
        return ChatBox(self.state, typing=self.typing, marks=self.marks if not self.typing else max(self.marks, 12))


class Hunter:
    def __init__(self, box, clock, quiet=(None,), near=False, alarm_at=None, lost=0, welcome=None):
        self.box, self.clock = box, clock
        self.quiet = list(quiet)  # unquiet's answers look by look (the last kept)
        self.near = near
        self.alarm_at = alarm_at  # the alarm up from this look on (its index), or from the key (-1)
        self.lost = lost  # looks that raise LookLost first
        self.looks, self.lines, self.keys, self.plans = 0, [], [], []
        self.actions = SimpleNamespace(keys=None, play=self._play)
        self.doubts_raise = False
        self.model = None
        self.hp_watch = None
        self.senses = SimpleNamespace(feed=SimpleNamespace(welcome=welcome))

    def log(self, text):
        self.lines.append(text)

    def look(self):
        if self.alarm_at is not None and 0 <= self.alarm_at <= self.looks:
            raise client.LieDetectorAlarm("Lie Detector seen at 00:00:00")
        if self.lost:
            self.lost -= 1
            self.clock.now += 3.0
            raise LookLost(Problem("drawn", "the game draws off its window"))
        self.looks += 1
        self.clock.now += 0.034
        return SimpleNamespace(chat=self.box.read(self.clock.now), t=self.clock.now, where=object())

    def _play(self, events, name, kind=None):
        assert kind is None and name in ("chat_collapse", "chat_box_mouse")
        self.plans.append(name)
        if name == "chat_box_mouse":
            assert events == [[0, "mouse", *chatcheck.MOUSE_AT]]
            self.clock.now += 0.02
            self.box.mouse()
            return
        self.keys.append([e[1] for e in events if e[2] == "down"])
        self.looks_at_key = self.looks
        self.clock.now += 0.05
        self.box.key(self.clock.now)


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
    """The key at the first quiet look after the box was seen expanded with its field unmarked over CLEAR_S (three
    looks: a caret blinking off on one shows on another), then seen collapsed."""
    h, c = made.make(Box(EXPANDED))
    out = c.run(at_start=True)
    assert out.what == "collapsed" and out.sent == ("'",) and h.keys == [["'"]] and h.plans == ["chat_collapse"]
    assert h.looks_at_key == 3 and h.lines == ["  the chat box expanded: collapsed (')"]
    assert made.events[-1]["kind"] == "chat_box" and made.events[-1]["what"] == "collapsed"
    assert c.run().what == "skipped" and h.keys == [["'"]]  # (once a run)


def test_collapsed_or_not_told_or_not_read_nothing_pressed(made):
    """Collapsed: said, nothing sent (the key toggles: pressed it would expand the box). Not told on any look within
    QUIET_S (its button covered, the mouse moved off it once): said, the key not sent. Another client size: said at
    once, nothing sent. No state of it (the simulator's eyes): nothing said."""
    h, c = made.make(Box(COLLAPSED))
    assert c.run(at_start=True).what == "was collapsed" and h.plans == [] and h.lines == ["  the chat box: collapsed"]
    h, c = made.make(Box(EXPANDED, unknown=10 ** 6))
    out = c.run(at_start=True, loop=False)
    assert out.what == "not told" and h.keys == [] and h.plans == ["chat_box_mouse"] and out.sent == ("mouse",)
    assert h.lines == ["  the chat box not told (its button not seen): the mouse moved off it (the pointer may be "
                       "what's over it), looked at again",
                       "  the chat box not told (its button not seen): ' not sent"]
    h, c = made.make(Box(EXPANDED))
    h.box.read = lambda now: ChatBox(UNKNOWN, why="the client is 1920x1080, not 1366x768", readable=False)
    out = c.run(at_start=True)
    assert out.what == "not told" and h.plans == [] and h.looks == 1 and c.done
    assert h.lines == ["  the chat box not told (the client is 1920x1080, not 1366x768): nothing sent"]
    h, c = made.make(Box(None))
    assert c.run(at_start=True).what == "not read" and h.keys == [] and h.lines == []
    # Not told at first, expanded on a later look: pressed then
    h, c = made.make(Box(EXPANDED, unknown=2))
    assert c.run(at_start=True).what == "collapsed" and h.keys == [["'"]]


def test_the_mouse_pointer_over_the_box_moved_off_it_once(made):
    """The review of 2026-10-30: the pointer resting over its button (unknown) or its field (typing) had the check give
    up for the run, blaming the input line. The mouse moved off the box (MOUSE_AT) at the first such look, the looks
    from MOUSE_SETTLE_S after it read again: collapsed then. Still marked after it (text really there): left so."""
    h, c = made.make(Box(EXPANDED, covered=True))
    out = c.run(at_start=True)
    assert out.what == "collapsed" and h.plans == ["chat_box_mouse", "chat_collapse"] and out.sent == ("mouse", "'")
    assert h.lines[0].startswith("  the chat box not told (its button not seen): the mouse moved off it")
    h, c = made.make(Box(EXPANDED, covered=True, typing=True))
    h.box.mouse = lambda: vars(h.box).update(typing=False, covered=False)  # (the pointer was what marked its field)
    assert c.run(at_start=True).what == "collapsed" and h.plans == ["chat_box_mouse", "chat_collapse"]
    assert h.lines[0].startswith("  the chat box expanded, its input line with text or a caret: the mouse moved off it")
    h, c = made.make(Box(EXPANDED, typing=True))  # (text there: still marked with the pointer away)
    out = c.run(at_start=True)
    assert out.what == "left" and h.plans == ["chat_box_mouse"] and h.keys == []
    assert h.lines[-1].endswith("(it may have the keyboard: a key would be typed into it): left so, ' not sent")
    assert c.moves == 1


def test_a_caret_blinking_is_never_pressed_into(made):
    """The review's: the typing guard decided on one look, so a caret in its off phase let the key go into a focused
    input line. A caret on every other look: never three unmarked looks in a row over CLEAR_S, never pressed."""
    h, c = made.make(Box(EXPANDED, blink=True))
    out = c.run(at_start=True)
    assert h.keys == [] and out.what == "left" and "its input line with text or a caret" in h.lines[-1]
    assert h.plans == ["chat_box_mouse"]  # (moved off at the first marked look, as the pointer would mark it)


def test_its_input_line_typed_in_nothing_pressed(made):
    """Its input line with text or a caret (it may have the keyboard): left so, nothing sent (no Esc, no Enter). After
    the key its field marked more than before it (the key typed into it: a lone apostrophe is 3 px, one is 1 of a
    dotted font's): left so, not pressed again."""
    h, c = made.make(Box(EXPANDED, typing=True))
    out = c.run(at_start=True)
    assert out.what == "left" and h.keys == []
    assert "its input line with text or a caret" in h.lines[-1] and "left so" in h.lines[-1]
    for after in ((EXPANDED, True, 3), (EXPANDED, False, 1)):
        h, c = made.make(Box(EXPANDED, after_key=after))
        out = c.run(at_start=True)
        assert out.what == "left" and h.keys == [["'"]] and "the key typed into it?" in h.lines[-1], after


def test_still_expanded_once_more_then_left_so(made):
    """The key doing nothing: once more at a quiet look (STILL_S after the first at the soonest, and its field seen
    unmarked again), then "still expanded after ' twice: left so"; never a third."""
    h, c = made.make(Box(EXPANDED, stuck=True))
    out = c.run(at_start=True)
    assert out.what == "left" and h.keys == [["'"], ["'"]] and out.sent == ("'", "'")
    assert h.lines == ["  the chat box still expanded after ': again",
                       "  the chat box still expanded after ' twice: left so"]
    assert c.run().what == "skipped" and len(h.keys) == 2


def test_a_game_slow_to_act_on_the_key_gets_no_second(made):
    """The review's: "still expanded" was judged on the first look 0.3 s after the key and the next, so a game that
    acted on it 0.4 s late had it pressed twice (expanded again). Expanded is believed only STILL_S after the key."""
    for delay in (0.4, 0.9):
        h, c = made.make(Box(EXPANDED, delay=delay))
        out = c.run(at_start=True)
        assert out.what == "collapsed" and h.keys == [["'"]], delay
        assert h.lines == ["  the chat box expanded: collapsed (')"]


def test_judged_only_on_a_frame_from_after_the_key(made):
    """A look whose frame is from under SETTLE_S after the key isn't taken as what the key did (the box redrawn on the
    next frame, a frame from before it still expanded): no second press for it."""
    h, c = made.make(Box(EXPANDED))
    first = h.box.read
    shown = {"n": 0}

    def late_read(now):  # (the first look after the key still shows it expanded, as a frame from before it would)
        if h.keys and shown["n"] < 1:
            shown["n"] += 1
            return ChatBox(EXPANDED)
        return first(now)
    h.box.read = late_read
    assert c.run(at_start=True).what == "collapsed" and h.keys == [["'"]]


def test_never_with_the_lie_detector_alarm_up(made):
    """The alarm up at the start (Hunter.look raises it) or just before the key or the mouse (client.check_alarm):
    nothing sent, LieDetectorAlarm raised (the hunt's retreat, or the command's stop at its start)."""
    h, c = made.make(Box(EXPANDED), alarm_at=0)
    with pytest.raises(client.LieDetectorAlarm):
        c.run(at_start=True)
    assert h.plans == []
    made.alarm["up"] = True
    for box in (Box(EXPANDED), Box(EXPANDED, covered=True)):
        h, c = made.make(box)
        with pytest.raises(client.LieDetectorAlarm):
            c.run(at_start=True)
        assert h.plans == []


def test_put_off_near_a_monster_then_at_a_quiet_look_in_the_hunt_loop(made):
    """A monster within PLACE_CLEAR at the start: put off at once, nothing sent; due again DEFER_GAP_S on at a look
    that tells it (Hunter.health_check raises ChatDue at a quiet look in the hunt loop; the hunt's looks before it
    watched: its field unmarked on them over CLEAR_S, the key at once at ChatDue's look). Not quiet for QUIET_S (in the
    air): put off too. After DEFERS_MAX put-offs, left so."""
    h, c = made.make(Box(EXPANDED), quiet=("green-mushroom within 250 of her",), near=True)
    out = c.run(at_start=True, loop=True)
    assert out.what == "put off" and h.keys == [] and c.pending and h.looks == 1
    assert h.lines == ["  the chat box expanded: its collapse put off (green-mushroom within 250 of her): again at a "
                       "quiet moment in the hunt"]
    assert not c.again(made.clock.now + 0.5) and c.again(made.clock.now + chatcheck.DEFER_GAP_S)
    t = made.clock.now + chatcheck.DEFER_GAP_S
    assert not c.again(t, SimpleNamespace(chat=None))  # (a look that doesn't tell it: not that one)
    assert c.again(t, SimpleNamespace(chat=ChatBox(UNKNOWN, why="its button not seen")))  # (the mouse not moved yet)
    h.near, h.quiet[:] = False, [None]
    for i in range(4):  # (the hunt's looks, health_check's: its field unmarked on them)
        made.clock.now += 0.3
        c.watch(SimpleNamespace(chat=ChatBox(EXPANDED), t=made.clock.now))
    looks = h.looks
    assert c.run(look=SimpleNamespace(chat=ChatBox(EXPANDED), t=made.clock.now)).what == "collapsed"
    assert h.keys == [["'"]] and not c.pending and h.looks_at_key == looks  # (ChatDue's look first: the key at once)
    # not on a platform the whole QUIET_S: put off
    h, c = made.make(Box(EXPANDED), quiet=("not on a platform",))
    assert c.run(at_start=True, loop=True).what == "put off" and h.keys == []
    h.doubts_raise = True  # (the hunt loop's runs)
    for _ in range(chatcheck.DEFERS_MAX):
        made.clock.now += chatcheck.DEFER_GAP_S
        c.run()
    assert c.done and not c.pending and h.keys == [] and h.lines[-1].endswith(
        f"; {chatcheck.DEFERS_MAX} times put off): left so this run")


def test_a_watch_broken_by_a_marked_look_starts_again(made):
    """The stretch of unmarked looks: a marked one (a caret on), an unknown one or none at all ends it; the same frame
    twice (a look and a glance of it) counts once; CLEAR_LOOKS of them over CLEAR_S let the key go."""
    h, c = made.make(Box(EXPANDED))
    for t, box in ((1.0, ChatBox(EXPANDED)), (1.3, ChatBox(EXPANDED)), (1.6, ChatBox(EXPANDED, typing=True, marks=12)),
                   (1.9, ChatBox(EXPANDED)), (2.2, ChatBox(EXPANDED)), (2.2, ChatBox(EXPANDED))):
        c.watch(SimpleNamespace(chat=box, t=t))
    assert c._clear == (1.9, 2.2, 2) and not c._clear_seen(2.4)  # (3 looks, 0.5 s)
    assert c._clear_seen(2.5)
    c.watch(SimpleNamespace(chat=ChatBox(UNKNOWN, why="its button not seen"), t=2.8))
    assert c._clear is None
    c.watch(SimpleNamespace(chat=ChatBox(EXPANDED), t=3.0))
    c.watch(SimpleNamespace(chat=None, t=3.1))
    assert c._clear is None


def test_where_no_hunt_loop_follows_a_put_off_is_left_so(made):
    """The review's: at a start `loop` came out true always, so `hunt kill`, a plain `tour` and `probe run` said
    "again at a quiet moment in the hunt" and nothing tried again, nor said it was left. With `loop` False (gamelab.
    __main__'s _hunter: not `tour --loop`): left so this run, said, its event sent."""
    h, c = made.make(Box(EXPANDED), quiet=("green-mushroom within 250 of her",), near=True)
    out = c.run(at_start=True, loop=False)
    assert out.what == "left" and c.done and not c.pending and h.keys == []
    assert h.lines == ["  the chat box expanded: no quiet moment to collapse it (green-mushroom within 250 of her): "
                       "left so this run"]
    assert made.events[-1]["what"] == "left"
    h.doubts_raise = True  # (loop kept for the run: a later run, were there one, as the start said)
    assert c.run().what == "skipped"


def test_not_told_or_no_look_put_off_in_the_hunt_loop(made):
    """The review's: not told at the start's end (the last look unknown though one said expanded, or none told) ended
    the check for the run; a look lost (LookLost: the window fix not yet given at the start) ended the command. Both are
    put off in a hunt loop (tried again at a look that tells it), left so elsewhere; LookLost never raised."""
    h, c = made.make(Box(EXPANDED, unknown=10 ** 6))
    c.moves = chatcheck.MOUSE_MOVES_MAX  # (the mouse moved off it already)
    out = c.run(at_start=True, loop=True)
    assert out.what == "put off" and c.pending and h.plans == []
    assert h.lines == ["  the chat box not told (its button not seen): its collapse put off: again at a quiet moment "
                       "in the hunt"]
    assert not c.again(made.clock.now + chatcheck.DEFER_GAP_S,
                       SimpleNamespace(chat=ChatBox(UNKNOWN, why="its button not seen")))
    h.box.unknown = 0
    made.clock.now += chatcheck.DEFER_GAP_S
    assert c.run().what == "collapsed" and h.keys == [["'"]]
    # Seen expanded, not quiet, then its button hidden on the last looks: put off as expanded
    h, c = made.make(Box(EXPANDED), quiet=("not on a platform",))
    orig = h.box.read
    h.box.read = lambda now: orig(now) if h.looks < 3 else ChatBox(UNKNOWN, why="its button not seen")
    c.moves = chatcheck.MOUSE_MOVES_MAX
    out = c.run(at_start=True, loop=True)
    assert out.what == "put off" and out.text.startswith("the chat box expanded: its collapse put off (not on a")
    # No look for LOST_S: put off in the hunt loop, left so elsewhere; never raised
    h, c = made.make(Box(EXPANDED), lost=1)
    out = c.run(at_start=True, loop=True)
    assert out.what == "put off" and h.lines == [
        "  the chat box not told (no look: the game draws off its window): its collapse put off: again at a quiet "
        "moment in the hunt"]
    h, c = made.make(Box(EXPANDED), lost=1)
    out = c.run(at_start=True, loop=False)
    assert out.what == "not told" and c.done and h.lines == [
        "  the chat box not told (no look: the game draws off its window): nothing sent"]


def test_a_perception_service_of_the_code_before_is_said(made):
    """The review's: with the perception service started from the code before this, the looks carry no chat box and
    the check ended silently. Said where the looks are a service's (its welcome's pid), once; the simulator's eyes and
    a test's: nothing said."""
    h, c = made.make(Box(None), welcome={"pid": 4242, "epoch": 1})
    out = c.run(at_start=True)
    assert out.what == "not read" and c.done and h.plans == []
    assert h.lines == ["  the chat box not read (the perception service's looks carry none: one started from the code "
                       "before it? restart it to have it collapsed): nothing sent"]
    assert made.events[-1]["what"] == "not read"
    h, c = made.make(Box(None))
    assert c.run(at_start=True).what == "not read" and h.lines == []


def test_a_hunting_commands_start_collapses_it_before_her_stat_window(monkeypatch):
    """gamelab.__main__'s _hunter: her max HP waited for, her chat box (at_start, `loop` as `tour --loop` says), then
    her stat window (the same `loop`); not a command that hunts (the pet), neither."""
    from gamelab import __main__ as cli
    from gamelab.behavior import hunter as hunter_mod
    from gamelab.localization import mapdata, service
    from gamelab.screen import STRETCHED
    from test_worldstate import small_map

    class Client:
        def __init__(self, name, **kw):
            self.welcome, self.why = {"pid": 1, "epoch": 1, "map": kw["map_id"]}, ""

        def ready(self, timeout):
            return True
    order = []

    def made(eyes, keys, model, log, feed, **kw):
        h = SimpleNamespace(hp_why="why", too_strong=lambda: order.append("strong"), max_hp=840,
                            senses=SimpleNamespace(window_fix=None))
        h.wait_max_hp = lambda: order.append("wait") or True
        h.collapse_chat = lambda at_start, loop=None: order.append(("chat", at_start, loop))
        h.read_stats = lambda at_start, loop=None: order.append(("read", at_start, loop))
        return h
    monkeypatch.setattr(service, "ServiceClient", Client)
    monkeypatch.setattr(cli, "_eyes", lambda model, title, perception, restore: SimpleNamespace(log=print,
                                                                                               geo=STRETCHED))
    monkeypatch.setattr(cli, "_fit_start", lambda title, fix: None)
    monkeypatch.setattr(mapdata, "load", lambda map_id: small_map())
    monkeypatch.setattr(hunter_mod, "Hunter", made)
    try:
        cli._hunter(cli.build_parser().parse_args(["tour", "--loop"]))
        assert order == ["wait", ("chat", True, True), ("read", True, True), "strong"]
        order[:] = []
        cli._hunter(cli.build_parser().parse_args(["tour"]))  # (a plain tour: no hunt loop to try either again in)
        assert order == ["wait", ("chat", True, False), ("read", True, False), "strong"]
        order[:] = []
        cli._hunter(cli.build_parser().parse_args(["tour", "--loop"]), throws=False)  # (the pet: it hunts nothing)
        assert order == []
    finally:
        cli._PERCEPTION.clear()


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
    (VK_OEM_7); not one the alarm lets through, nor the mouse's move."""
    from maplehelper.keys import KEY_CODES

    from gamelab.control import alarm, player
    assert "'" in player.allowed_keys() and KEY_CODES["'"] == (0xDE, False)
    player.validate({"events": [[0, "'", "down"], [40, "'", "up"]]}, player.allowed_keys())
    assert player.alarm_refusal({"events": [[0, "'", "down"], [40, "'", "up"]], "kind": None}, None, 1.0, [], 0.0)
    assert player.alarm_refusal({"events": [[0, "mouse", *chatcheck.MOUSE_AT]], "kind": None}, None, 1.0, [], 0.0)
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
    at its first quiet look DEFER_GAP_S on that tells it (ChatDue from health_check, that look the collapse's first, the
    hunt's looks before it watched: its field unmarked on them; with its own look taken anew a Green Mushroom had come
    within 250, put off again till 67 s). The monsters walk as with nobody there, hit or not (Realism.chase off): with
    their chase of her on, the collapse was still put off at the 0.4 minutes' end."""
    from gamelab.behavior.grind import Grind
    from gamelab.behavior.retreat import on_alarm

    def fn(sim):
        tour = sim.tour_stack()
        start = tour.h.collapse_chat(at_start=True, loop=True).what
        with on_alarm(tour.h):
            Grind(tour, log=sim.log).run(minutes=0.4)
        return {"start": start, "last": tour.h.chat_check.last.what}
    got = _sim(fn, monsters="all", chat=EXPANDED, chase=False)
    assert got["start"] == "put off" and got["last"] == "collapsed" and got["keys"] == ["'"]
    assert any("the chat box's collapse put off: a quiet moment now" in line for line in got["log"])
    assert got["world"]["chat_box"] == COLLAPSED


def test_chat_due_is_not_a_hunt_stop():
    from gamelab.behavior.safety import HuntStop
    assert not issubclass(ChatDue, HuntStop)
