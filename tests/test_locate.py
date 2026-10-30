"""Her place from the scene (gamelab/localization/locate.py): the camera registered against the map's art, her feet by
her tag's gray match, else her plate's colours, else her body, else carried by the filter, and the perception wired to
it (Eyes.look_at with a locator, Senses placing her by it, the Scene's screen <-> map by the view, the WorldState and
to_scene carrying it), behind its switch (config.toml [perception] scene), and nothing changed where it can't be used
(1920x1080, no art, the simulator's eyes). Frames are Tree Dungeon's art (gamedata's copy) at a view, her plate (made
up, as tests/test_plate.py's) and her body (tests/test_poses.py's bank) drawn on it. The evaluation set's numbers:
README, Her place from the scene."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from gamelab.analysis.bench import SavedEyes
from gamelab.localization import locate, mapart, mapdata
from gamelab.localization.camreg import Registration
from gamelab.localization.locate import DOT_OFF, Locator
from gamelab.localization.senses import Senses
from gamelab.localization.worldstate import WorldStep, to_scene
from gamelab.perception import plate as P
from gamelab.perception.poses import Background, BodyFinder
from gamelab.screen import NATIVE, STRETCHED
from tests.test_plate import model as plate_model
from tests.test_poses import bank as pose_bank
from tests.test_poses import draw as draw_pose

TREE = 10002073
VIEW = (-658, -308)       # the camera's lowest view on Tree Dungeon: platforms 28-33 in it
FEET = (701.5, None)      # her feet across on screen (her plate's centre 700); down: on platform 33's line


def needs_game_data():
    from gamedata.paths import COPY
    if not (COPY / "Map" / "Map_000.wz").is_file():
        pytest.skip("needs the game data copy (gamedata/data/copy)")


@pytest.fixture(scope="module")
def art(tmp_path_factory):
    needs_game_data()
    return mapart.load(TREE, cache=tmp_path_factory.mktemp("mapart"))


@pytest.fixture(scope="module")
def model():
    needs_game_data()
    return mapdata.load(TREE)


@pytest.fixture(scope="module")
def plate():
    return plate_model(0)


def template_of(m: P.PlateModel) -> np.ndarray:
    """Her tag's gray template as NameTag cuts it: 64 x 15 with its centre (32, 7) on the plate's."""
    gray = cv2.cvtColor(np.ascontiguousarray(m.picture), cv2.COLOR_RGB2GRAY)
    x0, _, y0, _ = m.box
    return np.ascontiguousarray(gray[-7 - y0:-7 - y0 + 15, -32 - x0:-32 - x0 + 64])


def scene(art, model, m: P.PlateModel, mx: float, view=VIEW, plate_drawn: str = "whole", body=None):
    """A frame of the art at `view` with her standing at map x `mx` on the lowest platform there in the play area: her
    plate drawn whole, partly ("part": its right part, from 8 px right of its centre, covered), or not at all ("none");
    her body's pose drawn at her feet if given. -> (frame, her feet on screen, her tag's centre, the platform)."""
    frame, _ = art.window(*view)
    frame = frame.copy()
    p = max((q for q in model.places if q.covers(mx) and view[1] + 100 < q.y_at(mx) < view[1] + 680),
            key=lambda q: q.y_at(mx))
    fx, fy = mx - view[0], p.y_at(mx) - view[1]
    cx, cy = int(round(fx - m.feet[0])), int(round(fy - m.feet[1]))
    if body is not None:
        draw_pose(frame, body, fx, fy)
    if plate_drawn != "none":
        x0, x1, y0, y1 = m.box
        frame[cy + y0:cy + y1, cx + x0:cx + x1] = m.picture
        if plate_drawn == "part":
            rng = np.random.default_rng(3)
            frame[cy + y0:cy + y1, cx + 8:cx + x1] = rng.integers(0, 256, (y1 - y0, x1 - 8, 3), dtype=np.uint8)
    return frame, (fx, fy), (cx, cy), p


def locator(art, model, m, bank=None) -> Locator:
    body = None if bank is None else BodyFinder(bank, Background(art.rgb, (art.x0, art.y0), art.static))
    return Locator(model, Registration(art), template_of(m), P.Plate(m), body)


def dot_of(loc: Locator, mx: float, my: float) -> tuple[float, float]:
    """Her minimap dot (full-size px, as the eyes give it) with her feet at map (mx, my)."""
    return loc.dots_reader.to_dot(mx + DOT_OFF[0], my + DOT_OFF[1])


def test_her_tag_places_her_and_the_view_is_the_maps(art, model, plate):
    loc = locator(art, model, plate)
    frame, (fx, fy), centre, p = scene(art, model, plate, 43.5)
    got = loc.see(frame, 0.0, dot_of(loc, fx + VIEW[0], fy + VIEW[1]))
    assert got.view == VIEW and got.view_sure
    assert got.how == "tag" and got.tag == centre and got.confidence > 0.95
    assert got.feet == pytest.approx((fx, fy)) and (got.x, got.y) == pytest.approx((43.5, p.y_at(43.5)))
    assert got.mode == "ground" and got.platform is p


def test_her_plate_places_her_where_her_tags_match_isnt_sure(art, model, plate):
    loc = locator(art, model, plate)
    for k, how in enumerate(("whole", "part", "part")):
        frame, (fx, fy), centre, p = scene(art, model, plate, 43.5 + 2 * k, plate_drawn=how)
        got = loc.see(frame, k / 15, dot_of(loc, fx + VIEW[0], fy + VIEW[1]))
    assert plates_gray(frame, plate, centre) < locate.SURE
    assert got.how == "plate" and got.tag == centre
    assert got.feet[0] == pytest.approx(fx, abs=1.0) and got.platform is p


def plates_gray(frame, m, centre) -> float:
    return P.gray_near(frame, template_of(m), centre, (2, 2))[0]


def test_her_body_places_her_where_her_plate_doesnt_show(art, model, plate):
    b = pose_bank(0, 3)
    loc = locator(art, model, plate, b)
    frame, (fx, fy), _, p = scene(art, model, plate, 43.5, body=b.poses[0])
    loc.see(frame, 0.0, dot_of(loc, fx + VIEW[0], fy + VIEW[1]))
    frame, (fx, fy), _, p = scene(art, model, plate, 45.5, plate_drawn="none", body=b.poses[0])
    got = loc.see(frame, 1 / 15, dot_of(loc, fx + VIEW[0], fy + VIEW[1]))
    assert got.how == "body" and got.feet[0] == pytest.approx(fx, abs=1.0) and got.platform is p


def test_carried_where_nothing_places_her_and_back_to_her_dot_when_it_disagrees_long(art, model, plate):
    loc = locator(art, model, plate)
    frame, (fx, fy), _, p = scene(art, model, plate, 43.5)
    loc.see(frame, 0.0, dot_of(loc, fx + VIEW[0], fy + VIEW[1]))
    bare, _ = art.window(*VIEW)
    confidences = []
    for k in range(1, 4):   # she's gone from the screen; her dot says she's where she was
        got = loc.see(bare, k / 15, dot_of(loc, fx + VIEW[0], fy + VIEW[1]))
        confidences.append(got.confidence)
    assert got.how is None and got.feet == pytest.approx((fx, fy), abs=0.5)
    assert confidences == sorted(confidences, reverse=True) and confidences[0] < 0.5
    far = (fx + VIEW[0] - 150, fy + VIEW[1])   # ... then 150 units left, for long: she's put there
    for k in range(4, 16):
        got = loc.see(bare, k / 15, dot_of(loc, *far))
    assert got.x == pytest.approx(far[0], abs=15)


def test_the_counts_go_by_the_frames_gaps(art, model, plate):
    loc = locator(art, model, plate)
    frame, (fx, fy), _, _ = scene(art, model, plate, 43.5)
    for k in range(10):
        loc.see(frame, k / 30, None)
    assert loc.k == 2   # the service's 30 a second: two frames to one of the prototype's 15
    for k in range(10, 40):
        loc.see(frame, 10 / 30 + (k - 10) / 15, None)
    assert loc.k == 1


def saved_eyes(model, m, art, geo=NATIVE):
    eyes = SavedEyes(np.zeros((8, 32), np.uint8), geo=geo, check_frames=False)
    eyes.name_tag.full = template_of(m)
    eyes.locator = Locator(model, Registration(art), lambda: eyes.name_tag.full, P.Plate(m))
    return eyes


def test_senses_places_her_by_the_scene_and_the_scene_converts_by_its_view(art, model, plate):
    eyes = saved_eyes(model, plate, art)
    senses = Senses(eyes, model, (300, 100))
    assert senses.locator is eyes.locator   # (kept: one for this map already)
    frame, (fx, fy), centre, p = scene(art, model, plate, 43.5)
    s = senses.place(eyes.look_at(frame, 0.0))
    assert s.view == VIEW and s.view_sure and s.how == "tag" and s.tag_seen and s.tag == centre
    assert (s.where.x, s.where.y) == pytest.approx((43.5, p.y_at(43.5))) and s.platform is p
    assert s.floor_y == round(fy) and s.mode == "ground" and s.grounded
    assert s.map_x(0) == VIEW[0] and s.map_y(10) == VIEW[1] + 10 and s.screen_x(43.5) == pytest.approx(fx)
    her = s.record()["her"]
    assert her["how"] == "tag" and her["mode"] == "ground" and her["conf"] > 0.9 and her["view_sure"]
    bare, _ = art.window(*VIEW)   # her gone: carried, not seen
    s = senses.place(eyes.look_at(bare, 1 / 15))
    assert s.how == "carried" and not s.tag_seen and s.view == VIEW and not s.grounded


def test_the_world_state_carries_her_place_by_the_scene(art, model, plate):
    eyes = saved_eyes(model, plate, art)
    senses = Senses(eyes, model, (300, 100))
    step = WorldStep(senses)
    frame, (fx, fy), centre, p = scene(art, model, plate, 43.5)
    st = step.step(lambda ropes: eyes.look_at(frame, 0.0, ropes), 0.0)
    assert st.view == VIEW and st.view_sure and st.feet == pytest.approx((fx, fy)) and st.mode == "ground"
    assert st.how == "tag" and st.confidence > 0.9 and st.where[2] == p.id
    s = to_scene(st, model, (300, 100))
    assert s.view == VIEW and s.feet == st.feet and s.map_x(100) == VIEW[0] + 100 and s.grounded
    step.close()


def test_switched_off_or_where_it_cant_be_used_nothing_changes(art, model, plate, tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[perception]\nscene = false\n")
    assert locate.configured(cfg) is False and locate.configured(tmp_path / "none.toml") is locate.ON
    monkeypatch.setattr(locate, "configured", lambda path=None: False)
    assert locate.build(model, template_of(plate), tmp_path) is None
    monkeypatch.setattr(locate, "configured", lambda path=None: True)
    assert locate.build(model, template_of(plate), tmp_path) is None   # no plate model in her lab folder
    plate.save(P.model_path(tmp_path))
    assert locate.build(model, template_of(plate), tmp_path, STRETCHED) is None   # 1920x1080: none
    monkeypatch.setattr(mapart, "load", lambda map_id: (_ for _ in ()).throw(OSError("no game data")))
    assert locate.build(model, template_of(plate), tmp_path, log=lambda text: None) is None   # no art
    eyes = SavedEyes(np.zeros((8, 32), np.uint8), geo=STRETCHED, check_frames=False)
    assert Senses(eyes, model, (300, 100)).locator is None and eyes.locator is None


def test_a_locator_from_her_lab_folder(art, model, plate, tmp_path, monkeypatch):
    monkeypatch.setattr(locate, "configured", lambda path=None: True)
    monkeypatch.setattr(mapart, "load", lambda map_id: art)
    plate.save(P.model_path(tmp_path))
    pose_bank(0, 3).save(Path(tmp_path) / "poses_1366x768.npz")
    loc = locate.build(model, template_of(plate), tmp_path)
    assert loc is not None and loc.body is not None and (loc.feet_dx, loc.feet_dy) == plate.feet
    assert loc.feet_map_x == (-233, 323)   # her feet across: Tree Dungeon's platforms in its VR box, 13 more


def test_after_a_gap_without_frames_she_is_placed_afresh(art, model, plate):
    loc = locator(art, model, plate)
    frame, (fx, fy), _, _ = scene(art, model, plate, 43.5)
    loc.see(frame, 0.0, None)
    bare, _ = art.window(*VIEW)
    assert loc.see(bare, 1 / 15, None).feet is not None   # carried a frame
    got = loc.see(bare, 1 / 15 + 2 * locate.GAP_S, None)   # ... but not across a gap: nothing known of her
    assert got.feet is None and got.how is None and got.view == VIEW
    frame, (fx, fy), centre, _ = scene(art, model, plate, 120.5)
    got = loc.see(frame, 1 / 15 + 2 * locate.GAP_S + 1 / 15, None)
    assert got.how == "tag" and got.tag == centre
