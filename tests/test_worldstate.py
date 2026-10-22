"""The continuous feed's step (localization.worldstate) on made-up frames: what it searches when (her band every
few frames, the whole screen once a second, what's asked on the next frame), the freshness stamps, the
monsters re-found every frame, her EXP gains and hits as events, and the Scene a look would have given. And
the searches off the frame loop: an answer back frames after its own, merged as of its frame (one track, re-found
from the next frame on, a new one's appearance at its own frame), a look's ask owed while one is out."""

import time

import numpy as np
import pytest

from gamelab.localization import mapdata, tracking, worldstate
from gamelab.localization.pose import Localizer
from gamelab.localization.senses import Senses
from gamelab.localization.worldstate import Asks, Schedule, Searches, WorldStep, search, to_scene
from gamelab.perception.eyes import Eyes, Observation
from gamelab.perception.loot import Item
from gamelab.perception.minimap import OTHER_SHIFT, Marker
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
        self.others: list[Marker] = []  # other players' minimap dots
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
                           dot=self.loc.to_dot(1000, 500), others=list(self.others), exp=exp)


def pig(x: float, y: float = 500.0, facing: int = 1) -> Mob:
    """A pig with its feet at map (x, y), as drawn on screen."""
    sx, feet = round(480 + (x - 1000) / U), round(300 + (y - 500) / U)
    return Mob("pig", sx, feet - 16, 48, 32, 0.9, float(sx), float(feet), facing, "move")


@pytest.fixture
def step():
    model = small_map()
    eyes = Drawn(model)
    return WorldStep(Senses(eyes, model, band=(100, 100)), schedule=Schedule())


def frames(step: WorldStep, n: int, start: float = 100.0, asks=None, each=None, late=None) -> list:
    """`n` frames, 30 a second: their states; the late states of the frames searched go to `late`, as they come
    back (the step's searches answer at once by default, so each right after its frame)."""
    out = []
    for k in range(n):
        t = start + k / FPS
        if each:
            each(k)
        out.append(step.step(lambda ropes, t=t: step.eyes.frame(t, ropes), t, asks=(asks or {}).get(k)))
        got = step.came_back()
        if late is not None:
            late += got
    return out


class Held(Searches):
    """Searches whose answers come back only when let go (`back`): a search taking frames, as on the feed's
    thread."""

    def __init__(self, senses):
        self.senses, self.out, self.ready = senses, None, False

    def free(self):
        return self.out is None

    def submit(self, job):
        self.out, self.ready = search(self.senses, job), False

    def back(self):
        self.ready = True

    def take(self, by=None):
        if not self.ready:
            return None
        out, self.out, self.ready = self.out, None, False
        return out


def test_her_band_5_times_a_second_the_whole_screen_once_a_second(step):
    late = []
    states = frames(step, 3 * FPS, late=late)
    assert step.counts["frames"] == 90 and step.counts["placed"] == 90
    assert all(s.late for s in late) and not any(s.late for s in states)
    wide = [s.seq for s in late if s.work["wide"]]
    band = [s.seq for s in late if s.work["band"]]
    assert wide == [1, 31, 61]  # the first frame, then every 1 s
    assert band == [k for k in range(7, 91, 6) if k not in wide]  # 0.2 s after the last search of it
    slow = WorldStep(Senses(step.eyes, step.model, band=(100, 100)))
    tenth = []
    for t in (100 + k / 10 for k in range(20)):
        slow.step(lambda ropes, t=t: step.eyes.frame(t, ropes), t)
        tenth += slow.came_back()
    assert [s.seq for s in tenth if s.work["band"]] == [k for k in range(3, 21, 2) if k != 11]  # (11: the screen)
    for s in late:  # a frame searched, given again: its own frame's time
        assert s.t == states[s.seq - 1].t and s.band_at == s.t and (s.wide_at == s.t) == bool(s.work["wide"])
    for s in states:  # each frame's state says when her band and the screen were last searched (merged by then)
        assert s.band_at == max((states[k - 1].t for k in wide + band if k < s.seq), default=None)
        assert s.wide_at == max((states[k - 1].t for k in wide if k < s.seq), default=None)


def test_what_is_asked_is_done_on_the_next_frame(step):
    late = []
    frames(step, 12, asks={2: Asks(band=True), 3: Asks(wide=True), 4: Asks(loot=True)}, late=late)
    assert [s.seq for s in late if s.work["band"]] == [3, 10]  # asked, then 0.2 s after the screen's (4)
    assert [s.seq for s in late if s.work["wide"]] == [1, 4]
    assert [s.seq for s in late if s.loot is not None] == [5]  # drops only when asked (no schedule here)


def test_a_monster_walking_is_refound_every_frame_with_one_id(step):
    late = []
    states = frames(step, 60, each=lambda k: setattr(step.eyes, "mobs", [pig(900 + 125 * k / FPS)]), late=late)
    assert step.counts["re-found"] == 59 and step.tracker.counts["tracks"] == 1
    assert states[0].monsters == [] and [m.kind for m in late[0].monsters] == ["pig"]  # found by the first search
    assert all([m.kind for m in s.monsters] == ["pig"] for s in states[1:] + late)
    tr = states[-1].tracks[0]
    assert tr.id == 1 and tr.vx == pytest.approx(125, rel=0.2) and tr.facing == 1
    assert [(e.kind, e.t) for s in states + late for e in s.events] == [("appeared", states[0].t)]


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
    late = []
    frames(step, 1, asks={0: Asks(loot=True)}, late=late)
    (s,) = late  # the frame searched, with what was found in it
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
    late = []
    states = frames(step, 3 * FPS, late=late)
    assert not any(s.loot is not None for s in states)  # the drops come with the frames searched, given again
    assert [s.seq for s in late if s.loot is not None and not s.loot_wide] == [16, 31, 46, 76]  # every 0.5 s
    assert [s.seq for s in late if s.loot_wide] == [1, 61]


# --- the searches off the frame loop: answers that come back frames later ---------------------------------

def held_step():
    model = small_map()
    eyes = Drawn(model)
    senses = Senses(eyes, model, band=(100, 100))
    searches = Held(senses)
    return WorldStep(senses, schedule=Schedule(band_s=None, wide_s=None), searches=searches), searches


def one(step, k, asks=None):
    t = 100.0 + k / FPS
    return step.step(lambda ropes: step.eyes.frame(t, ropes), t, asks=asks)


def test_a_monster_a_late_search_found_is_one_track_from_its_frame_re_found_from_the_next():
    step, searches = held_step()
    step.eyes.mobs = [pig(1100)]
    first = one(step, 0, Asks(band=True))  # searched on frame 1 ...
    assert first.monsters == [] and not searches.free()
    for k in (1, 2):
        step.eyes.mobs = [pig(1100 + 4 * k)]
        assert one(step, k).monsters == [] and step.came_back() == []  # ... still out: nothing followed yet
    searches.back()
    step.eyes.mobs = [pig(1112)]
    fourth = one(step, 3)  # back by frame 4: merged, and re-found in it
    (late,) = step.came_back()
    assert late.late and late.seq == 1 and late.t == first.t and late.band_at == first.t
    assert [round(m.x) for m in late.monsters] == [1100]  # where it was in its own frame
    assert [(e.kind, e.t) for e in late.events] == [("appeared", first.t)]
    assert [round(m.x) for m in fourth.monsters] == [1112] and fourth.band_at == first.t and fourth.events == []
    (tr,) = step.tracker.tracks.values()
    assert tr.first_seen == first.t and tr.last_seen == fourth.t and tr.seen == 2
    # Searched again, with it re-found in that frame: it's that track, no new one and no event
    one(step, 4, Asks(band=True))
    searches.back()
    one(step, 5)
    (again,) = step.came_back()
    assert again.seq == 5 and [m.kind for m in again.monsters] == ["pig"] and again.events == []
    assert step.tracker.counts["tracks"] == 1


def test_a_search_out_when_one_is_asked_for_owes_it_and_hands_it_over_once_back_on_a_frame_at_its_time():
    step, searches = held_step()
    one(step, 0, Asks(wide=True))
    assert not searches.free()
    one(step, 1, Asks(1e9, band=True))  # asked of a frame after this one: not yet, whatever
    assert step.owed is not None and step.owed.band
    second = one(step, 2, Asks(band=True, loot=True))  # (merged with what was owed)
    searches.back()
    assert [s.seq for s in step.came_back()] == [1]  # the screen's, back between frames ...
    assert searches.out is not None and searches.out.job.seq == 3 and searches.out.job.band  # ... then hers, at once
    assert searches.out.job.loot and step.owed is None and second.seq == 3


def test_a_late_search_doesnt_bring_back_a_monster_found_dead_since_nor_make_it_anew():
    step, searches = held_step()
    step.eyes.mobs = [pig(1100)]
    one(step, 0, Asks(band=True))
    searches.back()
    one(step, 1)
    step.came_back()
    (tr,) = step.tracker.tracks.values()
    one(step, 2, Asks(band=True))  # searched with the pig in view (and re-found in that frame) ...
    tr.state, tr.judged, tr.died_at, tr.cause = "dead", 100.0 + 2.5 / FPS, 100.0 + 2.5 / FPS, "other"
    known = searches.out.job.known
    searches.out.job.known = {}  # ... as if its re-find had missed it in that frame
    step.eyes.mobs = []
    searches.back()
    one(step, 3)
    (late,) = step.came_back()
    assert late.events == [] and step.tracker.counts["tracks"] == 1 and known
    assert step.tracker.counts["late sightings of the dead"] == 1 and step.tracker.tracks[tr.id].state == "dead"


def test_a_re_find_the_tracker_turned_down_doesnt_hide_that_monster_from_a_late_search():
    # A fading pig's window catches another walking by, 18 units off: the tracker turns that re-find down (a dying
    # one doesn't follow a walker through), so the search of that frame shows the walker as inline it would have:
    # it's not taken for the fading one's re-find.
    step, searches = held_step()
    step.eyes.mobs = [pig(1100)]
    one(step, 0, Asks(band=True))
    searches.back()
    for k in (1, 2, 3, 4, 5):
        one(step, k)
    step.came_back()
    (tr,) = step.tracker.tracks.values()
    tr.state = "dying"
    step.eyes.mobs = [pig(1118)]  # the fading one gone, a walker in its window
    one(step, 6, Asks(band=True))
    assert tr.id not in step.tracker.refound_seen
    searches.back()
    one(step, 7)
    (late,) = step.came_back()
    assert [round(m.x) for m in late.monsters] == [1118]  # the walker, seen in that frame
    # ... and matched to a track as the search's sighting (the nearest, the one that had been fading: as inline),
    # not dropped as one of the frame's re-finds
    (merged,) = [t for t in late.tracks if t.id == tr.id]
    assert merged.last_seen == pytest.approx(100.0 + 6 / FPS)


@pytest.mark.parametrize("out", ["drops", "band"])
def test_a_hidden_track_waits_to_be_judged_for_a_search_for_monsters_out_and_not_for_one_for_the_drops(out):
    # Its frame's search for monsters, inline, would have been merged before the track was judged; one for the
    # drops alone can't see it. Once due, the track is judged at the next frame with only that out: someone else's
    # kill, another player by it on the minimap (with nobody by it, it would be let go: tracking.OTHERS_BY_PLAYER).
    step, searches = held_step()
    step.eyes.mobs = [pig(1100)]
    x, y = step.eyes.loc.to_dot(1250, 500)
    step.eyes.others = [Marker("stranger", x + OTHER_SHIFT, y + OTHER_SHIFT)]
    one(step, 0, Asks(band=True))
    searches.back()
    for k in range(1, 8):
        one(step, k)
        step.came_back()
    (tr,) = step.tracker.tracks.values()
    step.eyes.mobs = []  # out of sight: hidden, then due HIDE_S after it was last seen
    due = tr.last_seen + tracking.HIDE_S
    k = 8
    while 100.0 + (k + 1) / FPS <= due:
        one(step, k)
        k += 1
    one(step, k, Asks(loot=True) if out == "drops" else Asks(band=True))  # handed over on the frame before it's due
    assert not searches.free()
    while 100.0 + k / FPS < due + 0.2:
        k += 1
        one(step, k)
    if out == "drops":
        assert step.tracker.tracks[tr.id].state == "dead" and step.tracker.counts["died other"] == 1
        assert step.tracker.tracks[tr.id].cause == "other"
    else:
        assert step.tracker.tracks[tr.id].state == "hidden" and step.tracker.counts["died other"] == 0
        searches.back()
        one(step, k + 1)  # it wasn't there: judged as soon as that search is merged
        assert step.tracker.tracks[tr.id].state == "dead"
