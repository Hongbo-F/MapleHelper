"""The perception's pass (localization.worldstate) on made-up frames, one a frame at the one rate (localization.rate:
15 a second): every frame's state with its own frame's whole screen, each detection with its track's id and its speed
the game data's, what's asked done on the next frame (the drops), the drops' rotation over the strips in view, the
bars read by time, her EXP gains and hits as events, her look and the search at once giving what one after the other
gives, and the Scene a look would have given."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gamelab.localization import mapdata, tracking, worldstate
from gamelab.localization.pose import Localizer
from gamelab.localization.rate import HZ
from gamelab.localization.senses import Senses
from gamelab.localization.worldstate import Asks, Schedule, WorldStep, to_scene
from gamelab.perception.eyes import Eyes, Observation
from gamelab.perception.loot import Item
from gamelab.perception.minimap import OTHER_SHIFT, Marker
from gamelab.perception.mobs import Mob
from gamelab.screen import STRETCHED, UNITS_PER_HALF_PX as U
from gamelab.sim.fakes import FakeDetector, FakeLoot, Frame, Screen

FPS = HZ
SKY = (60, 140, 230)


def small_map():
    tree = {
        "info": {"VRLeft": 0, "VRTop": 0, "VRRight": 3000, "VRBottom": 1000},
        "foothold": {"1": {
            "1": {"1": {"x1": 400, "y1": 500, "x2": 1600, "y2": 500, "prev": 0, "next": 0}},   # hers
            "2": {"2": {"x1": 500, "y1": 300, "x2": 1500, "y2": 300, "prev": 0, "next": 0}},   # 200 above
        }},
        "miniMap": {"width": 3000, "height": 1000, "centerX": 0, "centerY": 0, "mag": 4},
    }
    return mapdata.parse(1, tree, "Test", {})


class Drawn(Eyes):
    """Eyes on frames drawn as data (sim.fakes: monsters and drops listed, not painted), her standing at map
    (1000, 500), on screen at column 480 with her feet on row 300."""

    def __init__(self, model):
        self.detector, self._loot = FakeDetector(), FakeLoot()
        self.loc = Localizer(model)
        self.want_ropes, self.rope_columns = False, None
        self.mobs: list[Mob] = []
        self.drops: list[Item] = []
        self.others: list[Marker] = []  # other players' minimap dots
        self.hp, self.exp = 1.0, 0
        self.tag_at, self.grass = (480, 308), (300, 700)  # her tag on screen, the grass under her ((x, x): in the air)

    def look(self) -> Observation:
        self.looks = getattr(self, "looks", 0) + 1
        return self.frame(time.monotonic(), self.want_ropes)

    def glance(self):
        self.glances = getattr(self, "glances", 0) + 1
        return self.loc.to_dot(1000, 500), self.hp

    def picture(self) -> np.ndarray:
        frame = np.empty((540, 960, 3), np.uint8).view(Frame)
        frame[:] = SKY
        frame.screen = Screen(list(self.mobs), list(self.drops))
        return frame

    def frame(self, t: float, ropes: bool, picture: np.ndarray | None = None) -> Observation:
        exp = np.zeros((26, 140), bool)
        exp[5:15, :self.exp] = True
        return Observation(t=t, frame=self.picture() if picture is None else picture, tag=self.tag_at, hp=self.hp,
                           mp=1.0, ground=self.grass, dot=self.loc.to_dot(1000, 500), others=list(self.others), exp=exp)


def pig(x: float, y: float = 500.0, facing: int = 1, pose: str = "move") -> Mob:
    """A pig with its feet at map (x, y), as drawn on screen."""
    sx, feet = round(480 + (x - 1000) / U), round(300 + (y - 500) / U)
    return Mob("pig", sx, feet - 16, 48, 32, 0.9, float(sx), float(feet), facing, pose)


@pytest.fixture
def step():
    model = small_map()
    return WorldStep(Senses(Drawn(model), model, band=(100, 100)), schedule=Schedule(drops_strips=0))


def frames(step: WorldStep, n: int, start: float = 100.0, asks=None, each=None) -> list:
    """`n` frames, FPS a second: their states."""
    out = []
    for k in range(n):
        t = start + k / FPS
        if each:
            each(k)
        out.append(step.step(lambda ropes, t=t: step.eyes.frame(t, ropes), t, asks=(asks or {}).get(k)))
    return out


def test_every_frame_has_its_own_frames_whole_screen(step):
    """One pass a frame (the user, 2026-10-30: "make perception output the 15fps whole screen only"): each state's
    monsters are its own frame's whole screen, her band's and the screen's searches stamped with its own time, no state
    given again later."""
    step.eyes.mobs = [pig(1100), pig(1200, 300)]  # (one on the platform 200 above: off her band)
    states = frames(step, 3 * round(FPS))
    assert step.counts["frames"] == step.counts["placed"] == step.counts["wide"] == len(states) == 45
    assert all(s.band_at == s.wide_at == s.t and not s.late for s in states)
    assert all(sorted(round(m.y) for m in s.monsters) == [300, 500] for s in states)
    assert all("wide s" in s.work and s.work["detections"] == 2 for s in states)
    assert [s.work["matched"] for s in states] == [0] + [2] * 44  # (the first frame's made the two tracks)
    assert all({m.track for m in s.monsters} == {1, 2} for s in states)


def test_what_is_asked_is_done_on_the_next_frame(step):
    """The drops a look asks for are searched on the frame asked for (one shown at the time asked or later), her band
    and the whole screen being every frame's: asking for them changes nothing."""
    states = frames(step, 12, asks={2: Asks(band=True), 3: Asks(wide=True), 4: Asks(loot=True),
                                    6: Asks(1e9, loot_wide=True)})
    assert [s.seq for s in states if s.loot is not None] == [5]  # (the drops only when asked: no rotation here)
    assert step.owed is not None and step.owed.loot_wide  # (asked of a frame not shown yet: owed)
    assert all(s.monsters == [] and s.work["detections"] == 0 for s in states)


def test_a_monster_walking_is_one_track_every_frame_walking_its_kinds_speed_the_way_it_faces(step):
    """Its heading from the detection (facing and pose: the template that matched), its speed the game data's
    (monsters.walk_vx: a pig's walkSpeed 125, the tables'), from its very first frame; a stand frame stands."""
    def each(k):
        step.eyes.mobs = [pig(900 + 125 * k / FPS, pose="move" if k < 30 else "stand")]
    states = frames(step, 40, each=each)
    assert step.tracker.counts["tracks"] == 1 and all([m.track for m in s.monsters] == [1] for s in states)
    assert states[0].tracks[0].vx == 125.0 and states[29].tracks[0].vx == 125.0 and states[30].tracks[0].vx == 0.0
    assert states[0].tracks[0].facing == 1
    assert [(e.kind, e.t) for s in states for e in s.events] == [("appeared", states[0].t)]


def test_her_exp_gains_and_hits_come_as_events_at_their_frames(step):
    def each(k):
        step.eyes.exp = 10 if k >= 20 else 0
        step.eyes.hp = 0.9 if k >= 40 else 1.0
    states = frames(step, 60, each=each)
    events = [(s.seq, e.kind) for s in states for e in s.events]
    # (her HP dropping with no popup of hers told, these eyes reading none, is a touch too: perception.touches.HP)
    assert events == [(21, "exp_gain"), (41, "hp_drop"), (41, "touch")]
    assert states[40].events[0].data == {"lost": 0.1, "hp": 0.9, "new": True}
    assert states[40].events[1].data["how"] == "hp" and states[-1].touch["kind"] == "damage"


def test_a_change_of_her_exp_number_tells_what_it_said_before(step):
    """Each change tells the number its last reading said (`before`, from 2026-11-02), which a hunt attaching later
    starts from (behavior.hunter.feed_reads): before any reading, none."""
    def each(k):
        step.eyes.exp = 0 if k < 20 else 10 if k < 30 else 20
        if k == 25:
            step.meter.at = (31, 106302)  # (as a reading would have it)
    states = frames(step, 40, each=each)
    gains = [e for s in states for e in s.events if e.kind == "exp_gain"]
    assert len(gains) == 2 and "before" not in gains[0].data and gains[1].data["before"] == [31, 106302]


def test_a_scene_from_a_state_is_the_look_it_stands_for(step):
    above = pig(1200, 300)  # on the platform 200 above: out of her band (100 up)
    step.eyes.mobs = [pig(1100), above]
    step.eyes.drops = [Item("mesos", 480 + round(50 / U), 300 - 6, 13, 13, 0.9)]
    (s,) = frames(step, 1, asks={0: Asks(wide=True, loot=True)})
    model = step.model
    band = to_scene(s, model, (100, 100))
    assert [m.x for m in band.monsters] == [pytest.approx(1100, abs=2)] and band.searched and not band.wide
    assert band.seq == s.seq and band.record()["seq"] == s.seq  # (its frame number: a recording's frames by it)
    whole = to_scene(s, model, (100, 100), wide=True)
    assert sorted(round(m.y) for m in whole.monsters) == [300, 500] and whole.wide
    assert band.where.platform.id == model.platform_under(1000, 500).id and band.where.x == pytest.approx(1000, abs=12)
    looted = to_scene(s, model, (100, 100), mobs=False, loot=True)
    assert [d.item.kind for d in looted.loot] == ["mesos"] and not looted.searched
    assert to_scene(s, model, (100, 100), loot=True, mesos_only=True).loot == looted.loot
    assert np.array_equal(band.exp, step.eyes.frame(0, False).exp)


def test_nothing_is_read_from_a_frame_the_game_isnt_showing(step):
    s = step.step(lambda ropes: pytest.fail("a frame not visible was read"), 100.0, visible=False, why="not in front")
    assert not s.visible and s.why == "not in front" and step.counts["not visible"] == 1


def test_the_drops_rotation_searches_every_strip_in_view_her_platforms_most_often():
    """worldstate.DROPS_STRIPS (1) every DROPS_EVERY_S (0.2 s: every third frame at 15), unasked, the one searched
    longest ago first (her platform's counted DROPS_HER times as old): every strip in view in turn, each one's last
    search in every state (`drops`: when, and what it had), the state's `loot` still only what was asked for."""
    model = small_map()
    step = WorldStep(Senses(Drawn(model), model, band=(100, 100)), throws=False)
    assert (step.schedule.drops_strips, step.schedule.drops_every_s) == (worldstate.DROPS_STRIPS,
                                                                         worldstate.DROPS_EVERY_S) == (1, 0.2)
    step.eyes.drops = [Item("mesos", 480 + round(50 / U), 300 - 6, 13, 13, 0.9)]  # on her platform
    states = frames(step, 9)
    assert all(s.loot is None for s in states) and [s.work["strips"] for s in states] == [1, 0, 0] * 3
    assert len(states[0].drops) == 2 and {d.platform for d in states[0].drops} == {p.id for p in model.places}
    mine = model.platform_under(1000, 500).id
    first = {d.platform: d.at for d in states[0].drops}
    assert first[mine] == states[0].t and set(first.values()) == {states[0].t, None}  # (hers first, the other not yet)
    later = {d.platform: d.at for d in states[3].drops}
    assert later[mine] == states[0].t and None not in later.values()  # (then the other, searched longest ago)
    (strip,) = [d for d in states[-1].drops if d.platform == mine]
    assert [d.item.kind for d in strip.loot] == ["mesos"]
    # A look's drops asked for: searched whole on their frame, before the rotation, which takes another when it's due.
    (asked,) = frames(step, 1, start=101.0, asks={0: Asks(loot=True)})
    assert [d.item.kind for d in asked.loot] == ["mesos"] and asked.work["asked strips"] == 1
    assert asked.work["strips"] == 2  # (the other platform's by rotation)


def test_the_rotation_waits_for_a_frame_placed_firmly_and_a_strip_cut_short_is_not_sure():
    """The review of 2026-10-30: a frame placed by her offset in the air (no grass under her, no view registered) puts
    the strips as high above their ground as she is, and searched so they had the drop store see her piles gone with
    their coins there. The rotation waits for a frame placed firmly in height (worldstate._firm: her standing, or the
    view registered) and takes its turn there; a look's drops asked for are searched on its frame, not sure of anything
    gone (DropStrip.sure); a strip cut short by the bottom bar (scene.Strip.whole) isn't sure either."""
    model = small_map()
    step = WorldStep(Senses(Drawn(model), model, band=(100, 100)), throws=False)
    mine = model.platform_under(1000, 500).id
    step.eyes.grass = (480, 480)  # (in the air)
    states = frames(step, 6)
    assert [s.work["strips"] for s in states] == [0] * 6 and all(d.at is None for d in states[-1].drops)
    (asked,) = frames(step, 1, start=100.4, asks={0: Asks(loot=True)})
    assert asked.work["asked strips"] == asked.work["strips"] == 1  # (the look's own: searched, no rotation)
    assert [(d.platform, d.sure) for d in asked.drops if d.at is not None] == [(mine, False)]
    step.eyes.grass = (300, 700)  # (standing again: the rotation's turn at once)
    (firm,) = frames(step, 1, start=100.4 + 1 / FPS)
    assert firm.work["strips"] == 1 and all(d.sure for d in firm.drops if d.at == firm.t)
    step.eyes.tag_at = (480, 480)  # her feet a few rows over the bottom bar: her platform's strip cut short by it
    states = frames(step, 9, start=101.0)
    last = {d.platform: d for d in states[-1].drops}
    assert last[mine].at is not None and last[mine].at >= 101.0 and not last[mine].sure
    (other,) = [d for p, d in last.items() if p != mine]
    assert other.at is not None and other.at >= 101.0 and other.sure


def test_a_frame_that_fails_its_checks_says_why_in_its_state_and_is_neither_placed_nor_usable(step):
    # The frame's own checks (perception.framecheck) and her place off the map (localization.placecheck) as the step
    # meets them: the state carries the Problem, nothing is read from the frame, and no look may be given it.
    from gamelab.perception.framecheck import Problem
    good = frames(step, 1)[0]
    assert good.bad is None and good.placed and good.usable
    for kind in ("offset", "minimap", "hud", "picture", "tag"):
        problem = Problem(kind, f"the {kind} check failed")

        def refused(ropes, t=good.t + 0.1, problem=problem):
            return Observation(t=t, frame=step.eyes.frame(t, ropes).frame, tag=None, hp=1.0, mp=1.0, bad=problem)
        st = step.step(refused, good.t + 0.1)
        assert st.bad == problem and not st.placed and not st.usable and st.events == [] and st.monsters == []
    off = step.eyes.frame(good.t + 0.2, False)
    off.dot = (off.dot[0] + 400.0, off.dot[1])  # her dot far right of the map's end
    st = step.step(lambda ropes: off, good.t + 0.2)
    assert st.bad is not None and st.bad.kind == "place" and not st.usable
    # The game not wholly on screen: its window's problem, or (none given) why in words.
    hidden = step.step(None, good.t + 0.3, False, why="the game isn't in front",
                       bad=Problem("window", "the game isn't in front"))
    assert not hidden.visible and hidden.bad.kind == "window" and not hidden.usable
    assert step.step(None, good.t + 0.4, False, why="gone").bad == Problem("window", "gone")


def test_a_monsters_bar_is_read_while_it_shows_one_or_she_threw_its_way_else_every_bar_every_s(monkeypatch):
    # worldstate.BAR_EVERY_S (0.2 s: every third frame at 15 a second), BAR_SHOWN_S: in between, its track's bar as last
    # read. A new one's is read on its first frame. BAR_UNSEEN_S: seen again after a while unseen, its bar is read in the
    # frame it's back in (a pause between the batches below counts as one).
    from gamelab.localization.worldstate import BAR_EVERY_S
    every = round(BAR_EVERY_S * FPS)
    assert every == 3
    model = small_map()
    step = WorldStep(Senses(Drawn(model), model, band=(100, 100)), schedule=Schedule(drops_strips=0))
    reads, bar = [], {"left": None}

    def read(obs, mob):
        reads.append(obs.t)
        return bar["left"]
    step.eyes.bar = read
    step.eyes.mobs = [pig(1100, pose="stand")]
    frames(step, 1)
    assert len(reads) == 1  # (new: read on its first frame)
    frames(step, 14, start=100.0 + 1 / FPS)
    (tr,) = step.tracker.tracks.values()
    reads.clear()
    states = frames(step, 3 * every, start=101.0)
    assert len(reads) == 3  # unhurt, nobody throwing at it: every third frame
    bar["left"] = 0.6  # hit by someone: its next read shows it, and from then on it's read every frame
    states = frames(step, 2 * every, start=101.0 + 3 * every / FPS)
    first = next(i for i, st in enumerate(states) if st.monsters[0].bar == 0.6)
    assert first < every and len(reads) == 3 + 1 + (2 * every - first - 1)
    assert all(st.monsters[0].bar == 0.6 for st in states[first:])
    # A track whose bar went (none shown for BAR_SHOWN_S) is read every third frame again, its last bar kept meanwhile;
    # one she threw at is read every frame.
    bar["left"] = None
    frames(step, 15, start=103.0)
    reads.clear()
    states = frames(step, 2 * every, start=104.0)
    assert len(reads) == 2 and all(st.monsters[0].bar == (None if st.t in reads else 0.6) for st in states)
    step.throw(105.0, "right")
    reads.clear()
    frames(step, every, start=105.01)
    assert len(reads) == every
    # Hidden 0.3 s (under her damage numbers, say), then back: read in the frame it's back in, whatever BAR_EVERY_S
    # says (its bar as last read is from before it hid), and not in the next.
    monkeypatch.setattr(worldstate, "BAR_EVERY_S", 10_000)
    frames(step, 15, start=110.0)
    step.eyes.mobs = []
    frames(step, 5, start=111.0)
    step.eyes.mobs = [pig(1100, pose="stand")]
    reads.clear()
    back = frames(step, 3, start=111.4)
    assert len(reads) == 1 and reads[0] == back[0].t and back[0].monsters


def test_her_look_and_the_whole_screen_at_once_give_what_one_after_the_other_gives(monkeypatch):
    """worldstate.HALVES: with the real monster detector and the frame's picture handed with its look (the feed's Look,
    at the game's own size), the whole screen is searched on the halves' thread while her look is taken, and the states
    are the same, frame by frame, as with the search after her look (HALVES off)."""
    from gamelab.perception.mobs import MobDetector

    class Real(FakeDetector, MobDetector):  # (a real detector as the step tells them, searching as the fake does)
        def detect_wide(self, frame, rows, band=120, step=90):
            self.threads.add(threading.current_thread().name)
            return FakeDetector.detect_wide(self, frame, rows)

    class Look:
        def __init__(self, eyes, t):
            self.eyes, self.t, self.full = eyes, t, eyes.picture()

        def __call__(self, ropes):
            return self.eyes.frame(self.t, ropes, self.full)

    model = small_map()
    runs = {}
    for halves in (True, False):
        monkeypatch.setattr(worldstate, "HALVES", halves)
        eyes = Drawn(model)
        eyes.detector = Real()
        eyes.detector.threads = set()
        step = WorldStep(Senses(eyes, model, band=(100, 100)), schedule=Schedule(drops_strips=0))
        eyes.geo = SimpleNamespace(look_scale=1, client=(960, 540), hud_top=STRETCHED.hud_top)  # (the look frame the
        #                                                                                          capture itself)
        assert (step._halves is not None) == halves
        states = []
        for k in range(30):
            t = 100.0 + k / FPS
            eyes.mobs = [pig(700 + 100 * k / FPS), pig(1300 - 100 * k / FPS, facing=-1)]
            states.append(step.step(Look(eyes, t), t))
        runs[halves] = [(repr(st.tracks), st.monsters, st.events) for st in states]
        names = eyes.detector.threads
        assert all(n.startswith("wide") for n in names) if halves else names == {threading.current_thread().name}
        step.close()
    assert runs[True] == runs[False]


def test_a_track_out_of_sight_too_long_with_another_player_by_it_is_their_kill(step):
    """Hidden past tracking.HIDE_S with no gain of hers and another player's dot by it: someone else's kill, judged on
    the first frame after it's due (until 2026-10-30 a search out of a frame after its last sighting held it back)."""
    step.eyes.mobs = [pig(1100, pose="stand")]
    x, y = step.eyes.loc.to_dot(1250, 500)
    step.eyes.others = [Marker("stranger", x + OTHER_SHIFT, y + OTHER_SHIFT)]
    frames(step, 8)
    (tr,) = step.tracker.tracks.values()
    step.eyes.mobs = []  # out of sight: hidden, then due HIDE_S after it was last seen
    due = tr.last_seen + tracking.HIDE_S
    states = frames(step, round((due - 100.0) * FPS) + 2 - 8, start=100.0 + 8 / FPS)
    dead = next(st for st in states if any(e.kind == "died" for e in st.events))
    assert dead.t > due and dead.t - due <= 1 / FPS + 1e-9 and step.tracker.tracks[tr.id].cause == "other"
