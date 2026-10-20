"""The continuous feed's step (localization.worldstate) on made-up frames: what it searches when (her band every
few frames, the whole screen once a second, what's asked on the next frame), the freshness stamps, the
monsters re-found every frame, her EXP gains and hits as events, and the Scene a look would have given."""

import time

import numpy as np
import pytest

from gamelab.localization import mapdata, worldstate
from gamelab.localization.pose import Localizer
from gamelab.localization.senses import Senses
from gamelab.localization.worldstate import Asks, Schedule, WorldStep, to_scene
from gamelab.perception.eyes import Eyes, Observation
from gamelab.perception.loot import Item
from gamelab.perception.mobs import Mob
from gamelab.screen import UNITS_PER_HALF_PX as U
from gamelab.sim.fakes import FakeDetector, FakeLoot, Frame, Screen

FPS = 30
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
        self.hp, self.exp = 1.0, 0

    def look(self) -> Observation:
        self.looks = getattr(self, "looks", 0) + 1
        return self.frame(time.monotonic(), self.want_ropes)

    def glance(self):
        self.glances = getattr(self, "glances", 0) + 1
        return self.loc.to_dot(1000, 500), self.hp

    def frame(self, t: float, ropes: bool) -> Observation:
        frame = np.empty((540, 960, 3), np.uint8).view(Frame)
        frame[:] = SKY
        frame.screen = Screen(list(self.mobs), list(self.drops))
        exp = np.zeros((26, 140), bool)
        exp[5:15, :self.exp] = True
        return Observation(t=t, frame=frame, tag=(480, 308), hp=self.hp, mp=1.0, ground=(300, 700),
                           dot=self.loc.to_dot(1000, 500), exp=exp)


def pig(x: float, y: float = 500.0, facing: int = 1) -> Mob:
    """A pig with its feet at map (x, y), as drawn on screen."""
    sx, feet = round(480 + (x - 1000) / U), round(300 + (y - 500) / U)
    return Mob("pig", sx, feet - 16, 48, 32, 0.9, float(sx), float(feet), facing, "move")


@pytest.fixture
def step():
    model = small_map()
    eyes = Drawn(model)
    return WorldStep(Senses(eyes, model, band=(100, 100)), schedule=Schedule())


def frames(step: WorldStep, n: int, start: float = 100.0, asks=None, each=None) -> list:
    out = []
    for k in range(n):
        t = start + k / FPS
        if each:
            each(k)
        out.append(step.step(lambda ropes, t=t: step.eyes.frame(t, ropes), t, asks=(asks or {}).get(k)))
    return out


def test_her_band_every_6th_frame_the_whole_screen_once_a_second(step):
    states = frames(step, 3 * FPS)
    assert step.counts["frames"] == 90 and step.counts["placed"] == 90
    wide = [s.seq for s in states if s.work["wide"]]
    band = [s.seq for s in states if s.work["band"]]
    assert wide == [1, 31, 61]  # the first frame, then every 1 s
    assert band == [k for k in range(6, 91, 6) if k not in wide]
    for s in states:  # each state says when her band and the screen were last searched
        assert s.band_at == max(states[k - 1].t for k in wide + band if k <= s.seq)
        assert s.wide_at == max(states[k - 1].t for k in wide if k <= s.seq)


def test_what_is_asked_is_done_on_the_next_frame(step):
    states = frames(step, 12, asks={2: Asks(band=True), 3: Asks(wide=True), 4: Asks(loot=True)})
    assert [s.seq for s in states if s.work["band"]] == [3, 6, 12]
    assert [s.seq for s in states if s.work["wide"]] == [1, 4]
    assert [s.seq for s in states if s.loot is not None] == [5]  # drops only when asked (no schedule here)


def test_a_monster_walking_is_refound_every_frame_with_one_id(step):
    states = frames(step, 60, each=lambda k: setattr(step.eyes, "mobs", [pig(900 + 125 * k / FPS)]))
    assert step.counts["re-found"] == 59 and step.tracker.counts["tracks"] == 1
    assert all([m.kind for m in s.monsters] == ["pig"] for s in states)
    tr = states[-1].tracks[0]
    assert tr.id == 1 and tr.vx == pytest.approx(125, rel=0.2) and tr.facing == 1
    assert [e.kind for s in states for e in s.events] == ["appeared"]


def test_her_exp_gains_and_hits_come_as_events_at_their_frames(step):
    def each(k):
        step.eyes.exp = 10 if k >= 20 else 0
        step.eyes.hp = 0.9 if k >= 40 else 1.0
    states = frames(step, 60, each=each)
    events = [(s.seq, e.kind) for s in states for e in s.events]
    assert events == [(21, "exp_gain"), (41, "hp_drop")]
    assert states[40].events[0].data == {"lost": 0.1, "hp": 0.9, "new": True}


def test_a_scene_from_a_state_is_the_look_it_stands_for(step):
    above = pig(1200, 300)  # on the platform 200 above: out of her band (100 up)
    step.eyes.mobs = [pig(1100), above]
    step.eyes.drops = [Item("mesos", 480 + round(50 / U), 300 - 6, 13, 13, 0.9)]
    s = frames(step, 1, asks={0: Asks(loot=True)})[0]
    model = step.model
    band = to_scene(s, model, (100, 100))
    assert [m.x for m in band.monsters] == [pytest.approx(1100, abs=2)] and band.searched and not band.wide
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


def test_watch_searches_the_drops_on_its_own_schedule():
    model = small_map()
    step = WorldStep(Senses(Drawn(model), model, band=(100, 100)), throws=False, schedule=worldstate.WATCH)
    states = frames(step, 3 * FPS)
    assert [s.seq for s in states if s.loot is not None and not s.loot_wide] == [16, 31, 46, 76]  # every 0.5 s
    assert [s.seq for s in states if s.loot_wide] == [1, 61]
