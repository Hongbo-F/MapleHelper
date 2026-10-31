"""Her place from the scene (gamelab/localization/locate.py): the camera registered against the map's art, her feet by
her tag's gray match, else her plate's colours, else her body, else carried by the filter, and the perception wired to
it (Eyes.look_at with a locator, Senses placing her by it, the Scene's screen <-> map by the view, the WorldState and
to_scene carrying it), behind its switch (config.toml [perception] scene), and nothing changed where it can't be used
(1920x1080, no art, the simulator's eyes); after BugBaby's review, her dot's lag allowed for in a fast fall, her body
alone keeping her on her platform (no take-off, drop or walk-off on a look-alike pose's few px), carried far from her
dot put there within DOT_TO_N frames. Frames are Tree Dungeon's art (gamedata's copy) at a view, her plate (made
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
    # (no minimap on the frame: her dot not read, the place's dot where her place puts it, as what reads it expects)
    assert s.dot is None and s.where.dot == pytest.approx(
        senses.localizer.to_dot(43.5 + DOT_OFF[0], p.y_at(43.5) + DOT_OFF[1]))
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


def test_her_dot_hints_the_cameras_cold_search_whenever_its_read(art, model, plate):
    """(Only before the first registered view, it was: on a map the camera moves across too, every lost frame after
    searched the whole map, 80-900 ms.)"""
    loc = locator(art, model, plate)
    hints = []
    see = loc.reg.see
    loc.reg.see = lambda frame, t, hint=None: hints.append(hint) or see(frame, t, hint)
    frame, _, _, p = scene(art, model, plate, 43.5)
    dot = dot_of(loc, 43.5, p.y_at(43.5))
    for k in range(3):
        loc.see(frame, k / 30, dot)
    loc.see(frame, 3 / 30, None)
    assert hints == [loc.camera.corner(*loc.dot_map(dot))] * 3 + [None]


def test_whatever_building_her_locator_raises_gives_her_place_as_before(art, model, plate, tmp_path, monkeypatch):
    """Not the service's run for the map failing ("can't look at map"): a model file cut short, the art's drawing out
    of memory (a big map's first: 0.4-0.5 GB at its peak), anything; said."""
    monkeypatch.setattr(locate, "configured", lambda path=None: True)
    monkeypatch.setattr(mapart, "load", lambda map_id: art)
    plate.save(P.model_path(tmp_path))
    bank = Path(tmp_path) / "poses_1366x768.npz"
    pose_bank(0, 3).save(bank)
    said = []
    for path in (bank, Path(P.model_path(tmp_path))):
        whole = path.read_bytes()
        path.write_bytes(whole[:len(whole) // 2])
        assert locate.build(model, template_of(plate), tmp_path, log=said.append) is None
        path.write_bytes(whole)
    monkeypatch.setattr(mapart, "load", lambda map_id: (_ for _ in ()).throw(MemoryError("Unable to allocate")))
    assert locate.build(model, template_of(plate), tmp_path, log=said.append) is None
    assert len(said) == 3 and all("can't be built" in s for s in said) and "MemoryError" in said[2]
    monkeypatch.setattr(mapart, "load", lambda map_id: art)
    assert locate.build(model, template_of(plate), tmp_path, log=said.append) is not None


def test_a_locator_that_raises_is_put_away_and_she_is_placed_as_before(art, model, plate):
    eyes = saved_eyes(model, plate, art)
    said = []
    eyes.log = said.append

    def broken(*a, **k):
        raise ValueError("cr_costs refused its arguments")
    eyes.locator.see = broken
    frame, _, _, _ = scene(art, model, plate, 43.5)
    obs = eyes.look_at(frame, 0.0)
    assert obs.bad is None and obs.located is None and eyes.locator is None
    assert len(said) == 1 and "ValueError" in said[0]
    assert eyes.look_at(frame, 1 / 30).bad is None and len(said) == 1


# --- after BugBaby's review (2026-10-30; D:/MapleHelper-scratch/loc2build/bugbaby/final.txt, v2/) -------------------

def placed(art, model, plate, mx: float = 43.5):
    """A locator that has her standing on the lowest platform at map x `mx`, her tag seen once."""
    loc = locator(art, model, plate)
    frame, (fx, fy), _, p = scene(art, model, plate, mx)
    loc.see(frame, 0.0, dot_of(loc, fx + VIEW[0], fy + VIEW[1]))
    assert loc.mode == "ground" and loc.plat is p
    return loc, p


def test_her_dot_agrees_with_her_place_in_a_fast_fall_by_its_lag(art, model, plate):
    """Falling at FALL_SPEED her dot shows her where she was DOT_LAG_S before, 44 units higher: her plate was refused
    where it was exactly hers (BugBaby's held-out frames 2827, 3068)."""
    loc, p = placed(art, model, plate)
    y = p.y_at(43.5) + 100
    loc.mode, loc.plat, loc.ky = "air", None, locate.KF1(y, locate.FALL_SPEED)
    loc.kx.v = 0.0
    fx, fy = 43.5 - VIEW[0], y - VIEW[1]
    loc.dots = [(43.5, y - locate.FALL_SPEED * locate.DOT_LAG_S)]      # (map units, as `see` keeps them)
    assert loc.dot_agrees(fx, fy) and loc.dot_agrees(fx, fy - 15)
    assert not loc.dot_agrees(fx, fy + 70) and not loc.dot_agrees(fx + 50, fy)
    loc.mode, loc.ky = "ground", None                                   # standing: no lag allowed for
    assert not loc.dot_agrees(fx, fy) and loc.dot_agrees(fx, fy - 44 + 39)


def test_her_body_alone_keeps_her_on_her_platform_unless_it_surely_says_she_left(art, model, plate):
    """A look-alike pose's feet a few px over or under her platform's line were a take-off or a drop: the filter had
    her in the air, gravity carrying her through her platform (BugBaby's held-out 2706, 2830-2852, her plate covered)."""
    stands, air = {"grounded": 1.0, "score": 0.9}, {"grounded": 0.1, "score": 0.9}
    for how, extra, dd, mode in (("body", stands, -6, "ground"),                       # kept
                                 ("body", dict(air, score=0.7), -6, "ground"),          # an air pose, weakly: kept
                                 ("body", stands, 6, "ground"),                         # a few px under it: kept
                                 ("body", air, -6, "air"),                              # surely in the air: off it
                                 ("body", stands, -locate.BODY_OFF_LINE - 4, "air"),    # a jump's rise: off it
                                 ("body", stands, locate.BODY_OFF_LINE + 4, "air"),     # dropped through it
                                 ("tag", None, -6, "air"),                              # her tag: as before
                                 ("tag", None, 6, "air")):
        loc, p = placed(art, model, plate)
        info = {}
        loc.update_y(p.y_at(loc.kx.p) + dd, 1.0, how, info, extra)
        assert (loc.mode == "ground") == (mode == "ground"), (how, extra, dd)   # ("air": off it, a rope's top too)
        assert ("body_kept" in info) == (mode == "ground" and abs(dd) > locate.TAKEOFF)


def test_carried_past_her_platforms_end_she_stays_on_it_unless_her_tag_or_plate_saw_her_lately(art, model, plate):
    """(Standing at platform 33's end with her plate covered, BugBaby's held-out frames walked off it again and again on
    her body's x 5 px off, 270 units of fall each before her dot's alarm.)"""
    for seen, mode in ((None, "ground"), (-1.0, "ground"), (-0.1, "air")):
        loc, p = placed(art, model, plate)
        loc.kx.p, loc.kx.v = p.right - 1, 100.0
        loc.last_sure = None if seen is None else loc.t + seen
        loc.predict(0.1)
        assert loc.mode == mode, seen
        if mode == "ground":
            assert loc.plat is p and loc.kx.p <= p.right + 2 and loc.kx.v == 0.0


def test_her_body_alone_a_few_px_under_a_line_she_fell_onto_lands_her_on_it(art, model, plate):
    loc, p = placed(art, model, plate)
    line = p.y_at(loc.kx.p)
    stands = {"grounded": 1.0, "score": 0.9}
    for before, how, extra, landed in ((line - 4, "body", stands, True),     # over it, then a few px under: landed
                                       (line + 20, "body", stands, False),   # never over it (from under it)
                                       (line - 4, "tag", None, False)):      # her tag 5 under it: past it, as before
        loc.mode, loc.plat, loc.ky = "air", None, locate.KF1(before, 200.0, 1.0, 100.0)
        loc.last_y = (before, loc.t)                                       # (measured there a frame before)
        info = {}
        loc.update_y(line + 5, 1.0, how, info, extra)
        assert (loc.mode == "ground" and loc.plat is p) == landed, (before, how)


def test_carried_far_from_her_dot_she_is_put_there_within_dot_to_n_frames(art, model, plate):
    loc = locator(art, model, plate)
    frame, (fx, fy), _, _ = scene(art, model, plate, 43.5)
    loc.see(frame, 0.0, dot_of(loc, fx + VIEW[0], fy + VIEW[1]))
    bare, _ = art.window(*VIEW)
    far = (fx + VIEW[0] - 150, fy + VIEW[1])
    for k in range(1, locate.DOT_TO_N + 1):
        got = loc.see(bare, k / 15, dot_of(loc, *far))
        assert got.info.get("to_dot", False) == (k == locate.DOT_TO_N)
    assert got.x == pytest.approx(far[0], abs=1)


# --- after the carried frames' investigation (2026-10-30; D:/MapleHelper-scratch/carried/verdict.txt, ad/) ---------

@pytest.mark.parametrize("tall,up,kept", [((44, 46), 0, True),     # lying down on her platform's line: kept
                                          ((44, 46), 25, False),   # ... 25 px over it (her head behind the HUD)
                                          ((70, 90), 25, True)])   # a pose of her full height there: as before
def test_her_lying_down_pose_places_her_only_on_a_platforms_line(art, model, plate, tall, up, kept):
    """Behind the HUD in a drop her lying-down pose (BugBaby's pose 31, 45 rows) matched her head over its top: her feet
    put 13-34 px off, the frames carried after drifting to 43 (the live clip of 17:54:33, 769-771, 3566, 3569)."""
    b = pose_bank(0, 3, tall=tall)
    assert all((-q.dy0 < locate.PRONE_H) == (tall[1] <= locate.PRONE_H) for q in b.poses)
    loc = locator(art, model, plate, b)
    frame, (fx, fy), _, p = scene(art, model, plate, 43.5, body=b.poses[0])
    loc.see(frame, 0.0, dot_of(loc, fx + VIEW[0], fy + VIEW[1]))
    frame, (fx, fy), _, p = scene(art, model, plate, 45.5, plate_drawn="none")
    draw_pose(frame, b.poses[0], fx, fy - up)
    assert (loc.platform_near(fx + VIEW[0], fy - up + VIEW[1], up=locate.PRONE_LINE, down=locate.PRONE_LINE)
            is None) == (up > 0)
    got = loc.see(frame, 1 / 15, dot_of(loc, fx + VIEW[0], fy + VIEW[1]))
    assert got.info["body"] >= locate.SPRITE_OK   # (found either way)
    if kept:
        assert got.how == "body" and got.info["meas"] == pytest.approx([fx, fy - up], abs=1.0)
        assert "body_prone_off_line" not in got.info
    else:
        assert got.how is None and got.info["body_prone_off_line"][2] == b.poses[0].id
        assert got.feet == pytest.approx((fx, fy), abs=3.0) and got.platform is p   # carried on her platform


def stat_window_cal(main=(587, 241), detail=(768, 299)):
    """Her stat window's calibration as the live capture of 2026-10-28 left it (the game data's pictures; its Main panel
    x 587-772, y 241-522: her feet at (701.5, 540) and her tag at y 548-564 just under it), or its panels elsewhere."""
    from gamelab.perception import stat_window as sw
    return sw.Calibration(panels={"main": {"at": list(main), "size": [185, 281]},
                                  "detail": {"at": list(detail), "size": [187, 223]}}, detail_with_main=True)


OVER = {"main": (587, 300), "detail": (768, 358)}   # her stat window's panels with its Main panel over her feet and tag
BOARD = (200, 60, 1166, 690)                          # the Lie Detector's board over the play area: no view registered


def board(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    x0, y0, x1, y1 = BOARD
    out[y0:y1, x0:x1] = (90, 120, 200)
    return out


def test_carried_under_the_lie_detector_or_her_stat_window_counts_covered_and_places_nothing_otherwise(
        art, model, plate, tmp_path):
    """The user, 2026-10-30 ~20:35, of counting the stat window's and the Lie Detector's frames as covered, not lost:
    "yes please". The Lie Detector's alarm file looked at once in COVER_POLL_S, with one of its windows over the play
    area: the view not registered (its board), or a window seen (none here: with the view registered she's lost as ever,
    the alarm up or not; its file is there from the notice, 10 s before the board, till 10 s after it, or till
    clear-alarm); her stat window where its calibration has it open, over her; neither changes where she's placed, how
    sure, or what's searched."""
    from gamelab.perception import stat_window as sw
    alarm = tmp_path / "ALARM.json"
    plain, loc = locator(art, model, plate), locator(art, model, plate)
    loc.cover = locate.Cover(alarm, stat_window_cal(**OVER))
    frame, (fx, fy), _, _ = scene(art, model, plate, 43.5)
    bare, _ = art.window(*VIEW)
    hidden = board(bare)
    window = sw.compose(base=bare, at=OVER).frame
    moved = sw.compose(base=bare, at={k: (x + 60, y + 30) for k, (x, y) in OVER.items()}).frame   # (dragged off)
    dot = dot_of(plain, fx + VIEW[0], fy + VIEW[1])
    # (each frame, whether the alarm's file is there as it's shown, what it counts as)
    frames = ([(frame, False, "tag")] + [(hidden, False, "carried")] * 3
              + [(hidden, True, "carried")] * 5   # the alarm's file written: looked at again 0.5 s on (k = 9)
              + [(hidden, True, "covered")] * 3
              + [(bare, True, "carried")]         # the view registered, no window seen, the alarm up: lost as ever
              + [(frame, True, "tag")]
              + [(hidden, False, "covered")] * 3  # ... and taken away: still up till the next look (k = 17)
              + [(hidden, False, "carried")] * 2
              + [(window, False, "covered"), (frame, False, "tag"), (window, False, "covered"),
                 (moved, False, "carried"), (window, False, "covered")])
    for k, (f, up, want) in enumerate(frames):
        if up:
            alarm.write_text("{}")
        else:
            alarm.unlink(missing_ok=True)
        a, b = plain.see(f, k / 15, dot), loc.see(f, k / 15, dot)
        assert b.view_sure == (f is not hidden), k
        assert (b.how or ("covered" if b.covered else "carried")) == want, k
        assert b.covered == (None if want != "covered" else locate.LIE_DETECTOR if k < 19 else locate.STAT_WINDOW), k
        assert (a.how, a.feet, a.x, a.y, a.mode, a.confidence, a.unseen) == (b.how, b.feet, b.x, b.y, b.mode,
                                                                             b.confidence, b.unseen), k
        assert a.covered is None
    assert (loc.counts["tag"], loc.counts["carried"], loc.counts["covered"]) == (3, 12, 9)
    assert (plain.counts["carried"], plain.counts["covered"]) == (21, 0)
    assert loc.line().startswith("her place by the scene: tag 12.5%, carried 50.0%, covered 37.5% (the Lie Detector, "
                                 "her stat window); views registered 33.3%, ")
    assert plain.line().startswith("her place by the scene: tag 12.5%, carried 87.5%; views registered")
    loc.see(bare, len(frames) / 15, dot)
    assert loc.line().startswith("her place by the scene: carried 100.0%; ")   # (what covered her since alone)


def test_the_lie_detectors_window_seen_with_the_view_registered_counts_covered(art, model, plate, tmp_path):
    """The countdown's window (the four Violettas: on its clip of 21:17:17 over her, the view registered about it) seen
    at its place (LieDetectorSpotter.over_play, looked at once in COVER_POLL_S), the alarm up: covered; gone,
    carried."""
    alarm = tmp_path / "ALARM.json"
    alarm.write_text("{}")
    frame, (fx, fy), _, _ = scene(art, model, plate, 43.5)
    bare, _ = art.window(*VIEW)
    shown = bare.copy()

    class Spotter:
        looks = 0

        def over_play(self, f):
            Spotter.looks += 1
            return "countdown" if f is shown else None
    loc = locator(art, model, plate)
    loc.cover = locate.Cover(alarm, None, Spotter())
    dot = dot_of(loc, fx + VIEW[0], fy + VIEW[1])
    got = [loc.see(f, k / 15, dot) for k, f in enumerate([frame] + [shown] * 8 + [bare] * 3)]
    assert [g.how or g.covered or "carried" for g in got] == (["tag"] + [locate.LIE_DETECTOR] * 8 + ["carried"] * 3)
    assert all(g.view_sure for g in got) and Spotter.looks == 2   # (at 1/15 s and 0.6 s: COVER_POLL_S on)
    alarm.unlink()
    loc = locator(art, model, plate)
    loc.cover = locate.Cover(alarm, None, Spotter())
    got = [loc.see(f, k / 15, dot) for k, f in enumerate([frame, shown, shown])]
    assert [g.covered for g in got] == [None] * 3 and Spotter.looks == 2   # (no alarm: not looked at)


def test_her_stat_window_open_away_from_her_leaves_a_loss_carried(art, model, plate, tmp_path):
    """The review of 2026-10-30: any carried frame counted covered while a panel was at its place, wherever she was (S
    left open, or the hunt's read, while she's lost at platform 12's right end). Now only with her feet or her tag,
    where the filter has them, under one of its panels (on the live clip of 17:54:33 its 34 frames had her feet in its
    Main panel)."""
    from gamelab.perception import stat_window as sw
    frame, (fx, fy), centre, _ = scene(art, model, plate, 43.5)
    bare, _ = art.window(*VIEW)
    for main, want in (((587, 241), None),                    # its bottom 522: her feet at 540, her tag 548-564
                       ((587, 262), locate.STAT_WINDOW),      # its bottom 543: her feet under it
                       ((587, 290), locate.STAT_WINDOW),      # her tag's top rows under it too
                       ((300, 241), None)):                   # away across
        at = {"main": main, "detail": (main[0] + 181, main[1] + 58)}
        loc = locator(art, model, plate)
        loc.cover = locate.Cover(tmp_path / "ALARM.json", stat_window_cal(**at))
        dot = dot_of(loc, fx + VIEW[0], fy + VIEW[1])
        assert loc.see(frame, 0.0, dot).how == "tag"
        if main == (587, 241):
            assert loc.see(sw.compose(base=frame, at=at).frame, 1 / 15, dot).how == "tag"   # (her tag shows below it)
        got = loc.see(sw.compose(base=bare, at=at).frame, 2 / 15, dot)
        assert got.how is None and got.covered == want, main
        assert got.feet == pytest.approx((fx, fy), abs=0.5)
    box = loc.her_box()
    assert box[0] < fx < box[2] and box[1] == pytest.approx(fy, abs=0.5)
    assert box[3] == pytest.approx(centre[1] + 7.5, abs=1.0)


def test_covered_in_the_scene_and_its_record(art, model, plate, tmp_path):
    from gamelab.perception import stat_window as sw
    eyes = saved_eyes(model, plate, art)
    eyes.locator.cover = locate.Cover(tmp_path / "ALARM.json", stat_window_cal(**OVER))
    senses = Senses(eyes, model, (300, 100))
    frame, (fx, fy), centre, p = scene(art, model, plate, 43.5)
    senses.place(eyes.look_at(frame, 0.0))
    bare, _ = art.window(*VIEW)
    s = senses.place(eyes.look_at(sw.compose(base=bare, at=OVER).frame, 1 / 15))
    assert s.how == "covered" and not s.tag_seen and s.located.covered == locate.STAT_WINDOW
    assert (s.where.x, s.where.y) == pytest.approx((43.5, p.y_at(43.5)), abs=0.5) and s.mode == "ground"
    assert s.record()["her"]["how"] == "covered"
    s = senses.place(eyes.look_at(bare, 2 / 15))
    assert s.how == "carried" and s.record()["her"]["how"] == "carried"


def test_the_alarm_counts_only_without_a_view_and_the_stat_window_only_over_her(tmp_path, monkeypatch):
    """The review of 2026-10-30: the alarm's file is there from the notice (10 s before the mini-game; the quiz's
    warning 30 s) till 10 s after it, or till clear-alarm: on its clip of 17:37:22 another player's tag over her 30 s
    after the board had gone (956-957) counted covered. Now with one of its windows over the play area alone: the view
    not registered (the board's 200 frames there were exactly its 200 not registered) or a window seen (none here);
    her stat window looked at only with her place known."""
    from gamelab.perception import stat_window as sw

    class Spotter:
        def over_play(self, frame):
            return None
    (tmp_path / "ALARM.json").write_text("{}")
    c = locate.Cover(tmp_path / "ALARM.json", stat_window_cal(), Spotter())
    seen = []
    monkeypatch.setattr(sw, "shown", lambda frame, cal, over=None: seen.append(over) or "main")
    frame = np.zeros((768, 1366, 3), np.uint8)
    assert c(frame, 0.0, False, None) == locate.LIE_DETECTOR and not seen
    assert c(frame, 0.0, True, None) is None and not seen                        # (her place not known: not looked)
    assert c(frame, 0.0, True, (680, 540, 720, 564)) == locate.STAT_WINDOW and seen == [(680, 540, 720, 564)]
    assert locate.Cover(tmp_path / "ALARM.json", None, Spotter())(frame, 0.0, True, (680, 540, 720, 564)) is None


def test_the_alarms_file_unreadable_is_up_and_the_stat_windows_look_failing_is_turned_off(tmp_path, monkeypatch):
    c = locate.Cover(tmp_path / "ALARM.json", stat_window_cal())
    assert not c.alarm_up(0.0)
    (tmp_path / "ALARM.json").write_text("{}")
    assert not c.alarm_up(0.1) and c.alarm_up(locate.COVER_POLL_S)   # (looked at again COVER_POLL_S on)
    assert c.alarm_up(0.2) and not locate.Cover(tmp_path / "none.json").alarm_up(5.0)   # (the time back: again)

    def denied(path, *a, **k):
        raise PermissionError(path)
    with monkeypatch.context() as m:
        m.setattr(locate.os, "stat", denied)
        assert locate.Cover(tmp_path / "none.json").alarm_up(0.0)          # (control.player.alarm_up's: up)
    from gamelab.perception import stat_window as sw
    monkeypatch.setattr(sw, "shown", lambda frame, cal, over=None: (_ for _ in ()).throw(OSError("no UI_000.wz")))
    c = locate.Cover(tmp_path / "none.json", stat_window_cal())
    assert c(np.zeros((768, 1366, 3), np.uint8), 0.0, True, (680, 300, 720, 400)) is None
    assert c.cal is None and "UI_000" in c.why

    class Broken:
        def over_play(self, frame):
            raise OSError("no lie_detector_title.png")
    c = locate.Cover(tmp_path / "ALARM.json", None, Broken())   # (the alarm up)
    frame = np.zeros((768, 1366, 3), np.uint8)
    assert c(frame, 0.0, True, None) is None and "lie_detector_title" in c.why_windows
    assert c(frame, 1.0, True, None) is None and c(frame, 1.0, False, None) == locate.LIE_DETECTOR


def test_a_locators_cover_from_her_stat_windows_calibration(art, model, plate, tmp_path, monkeypatch):
    """Its stat window looked for with a calibration of the client's size alone; one not as it's written (the review:
    a client of one number raised as it was said, and the whole locator wasn't built) turns the stat window off
    alone."""
    import json

    from gamelab import paths
    monkeypatch.setattr(locate, "configured", lambda path=None: True)
    monkeypatch.setattr(mapart, "load", lambda map_id: art)
    plate.save(P.model_path(tmp_path))
    said = []
    loc = locate.build(model, template_of(plate), tmp_path, log=said.append)
    assert loc.cover is not None and loc.cover.cal is None and loc.cover.alarm is None   # (paths.ALARM's, when asked)
    assert ("covered under the Lie Detector's windows (its alarm up) (her stat window not looked for: no calibration "
            "of it); in a portal's light (out00, out01)") in said[0]
    stat_window_cal().save(paths.STAT_WINDOW)
    said.clear()
    loc = locate.build(model, template_of(plate), tmp_path, log=said.append)
    assert loc.cover.cal is not None and loc.cover.cal.place("main") == (587, 241)
    assert said[0].endswith("covered under the Lie Detector's windows (its alarm up) and her stat window over her; in a "
                            "portal's light (out00, out01)")
    for data, why in (({"panels": {"main": {"at": [587, 241]}}, "client": [1366]}, "calibrated at 1366)"),
                      ({"panels": {"main": {"at": [587, 241]}}, "client": [1920, 1080]}, "calibrated at 1920x1080)"),
                      ({"panels": {"main": {"at": [587]}}}, "its calibration or pictures: IndexError")):
        paths.STAT_WINDOW.write_text(json.dumps(data))
        said.clear()
        loc = locate.build(model, template_of(plate), tmp_path, log=said.append)
        assert loc is not None and loc.cover.cal is None and why in said[0], data


# --- a portal's light (the user, 2026-10-30 ~20:47, of the portal's losses: "let's do 2 for the portal") ---------------

V12 = (-658, -1650)   # a view with Tree Dungeon's platform 12 (y -1190) the lowest in the play area, out01 at its right end
LIGHT = ("out01", (240, -1383, 323, -1176), (240, -1293, 323, -1176))   # out01's (281, -1183): lit, dense
ON_SCREEN = (900, 440, 970, 484)      # her box on screen (Locator.her_box) where her tag is searched


def test_a_portals_light_from_the_game_data(model):
    """The client draws an exit (a changeable or a script portal too) with MapHelper.img portal/game/pv/default's 8
    frames (84-88 x 182-256 px) each with its origin on the portal's point, a hidden one ph/default's portalStart and
    portalContinue (a script one psh's): its light the px lit (alpha over LIGHT_ALPHA) in any of them, its lit box and
    its dense rows' (lit across half its width or more), not the pictures' union (-44, -248, 45, 9): a fainter glow and a
    few streaks go up to its top (its review: her feet counted 175 px over the point). The picture a map names (`image`:
    2 of the game data's 466 exits) is its own. Tree Dungeon's two exits; its six spawn points none."""
    assert mapdata.portal_light() == ((-41, -200, 42, 7), (-41, -110, 42, 7))
    assert mapdata.portal_light("2") == ((-53, -127, 51, 8), (-53, -114, 51, 6))
    assert mapdata.portal_light("", 10) == mapdata.portal_light("", 11) == ((-45, -116, 46, 2), (-45, -105, 46, 1))
    lights = {p.name: (lit, dense) for p, lit, dense in mapdata.portal_lights(model)}
    assert lights == {"out00": ((241, -249, 324, -42), (241, -159, 324, -42)), "out01": LIGHT[1:]}
    assert sum(p.kind == 0 for p in model.portals) == 6 and all(p.image == "" for p in model.portals)
    tree = {"portal": {str(i): {"pn": f"p{kind}", "pt": kind, "x": 10, "y": 20, "tm": 2}
                       for i, kind in enumerate((0, 1, 2, 3, 4, 6, 7, 8, 10, 11))}}
    tree["portal"]["10"] = {"pn": "named", "pt": 2, "x": 10, "y": 20, "tm": 2, "image": "2"}
    m = mapdata.parse(1, tree, "", {})
    got = {p.name: (lit, dense) for p, lit, dense in mapdata.portal_lights(m)}
    assert set(got) == {"p2", "p4", "p7", "p10", "p11", "named"}
    assert got["p2"] == got["p4"] == got["p7"] == ((-31, -180, 52, 27), (-31, -90, 52, 27))
    assert got["p10"] == got["p11"] == ((-35, -96, 56, 22), (-35, -85, 56, 21))
    assert m.portals[-1].image == "2" and got["named"] == ((-43, -107, 61, 28), (-43, -94, 61, 26))


def test_carried_in_a_portals_light_counts_covered_and_places_nothing_otherwise(art, model, plate, tmp_path):
    """RiceBaby standing in out01's light at platform 12's right end (x 280) was lost on about 18 of 24 frames (its
    pulsing pillar over her body, its bottom over her plate's top rows), the filter holding her at her place; 18 px
    either side (x 262, 298) her tag placed her every frame. A frame no cue placed her in counts covered with her body's
    box (BODY) all inside a portal's light where the filter has her feet, her feet in its dense part, her tag where it's
    searched; where she's placed, how sure, what's searched: as without it."""
    lights = [(p.name, lit, dense) for p, lit, dense in mapdata.portal_lights(model)]
    bare, _ = art.window(*V12)
    for mx, want in ((280.5, locate.PORTAL), (262.5, None), (297.5, None)):
        plain, loc = locator(art, model, plate), locator(art, model, plate)
        loc.cover = locate.Cover(tmp_path / "ALARM.json", None, None, lights)
        frame, (fx, fy), _, p = scene(art, model, plate, mx, view=V12)
        assert p.id == 12
        dot = dot_of(plain, mx, p.y_at(mx))
        for k, f in enumerate([frame] + [bare] * 4):
            a, b = plain.see(f, k / 15, dot), loc.see(f, k / 15, dot)
            assert (a.how, a.feet, a.x, a.y, a.mode, a.platform, a.confidence, a.unseen) == (
                b.how, b.feet, b.x, b.y, b.mode, b.platform, b.confidence, b.unseen), (mx, k)
            assert b.how == ("tag" if k == 0 else None) and b.covered == (None if k == 0 else want), (mx, k)
            assert a.covered is None
        assert b.x == pytest.approx(mx, abs=0.5) and b.platform is p
        assert locate.searched(loc.her_box(), bare)
        assert loc.line().startswith("her place by the scene: tag 20.0%, " + (
            "covered 80.0% (a portal); views registered 100.0%, " if want else "carried 80.0%; views registered 100.0%"))
    eyes = saved_eyes(model, plate, art)                  # the scene's how, and its record
    eyes.locator.cover = locate.Cover(tmp_path / "ALARM.json", None, None, lights)
    senses = Senses(eyes, model, (300, 100))
    frame, _, _, p = scene(art, model, plate, 280.5, view=V12)
    assert senses.place(eyes.look_at(frame, 0.0)).how == "tag"
    s = senses.place(eyes.look_at(bare, 1 / 15))
    assert s.how == "covered" and s.located.covered == locate.PORTAL and s.record()["her"]["how"] == "covered"
    assert s.platform is p and s.where.x == pytest.approx(280.5, abs=0.5)


def test_a_portals_light_counts_her_in_it_with_her_tag_where_its_searched_after_the_lie_detector_and_her_stat_window(
        tmp_path, monkeypatch):
    """Her body's box all inside the light's lit box, her feet in its dense rows: standing on platform 12 from x 268.5
    to 293.5, jumping in it; not over its dense part (its review: Forest Up North IV's platform 8 is 156 over its out01,
    its pictures' union had had her counted there), nor under its foot. Her box on screen (Locator.her_box) off the
    frame or under the HUD's top (ROWS_MAX: her tag behind the HUD, falling into out00's light or through out01's, 5 of
    the telemetry's lost scenes): carried."""
    from gamelab.perception import stat_window as sw
    c = locate.Cover(tmp_path / "ALARM.json", None, None, [LIGHT])
    y = -1189.6                                                       # (platform 12's line there)
    assert [c.light((x, y)) for x in (268.5, 280.5, 293.5)] == ["out01"] * 3
    assert [c.light((x, y)) for x in (262.5, 268.0, 294.0, 310.0)] == [None] * 4   # (her body out of it, either side)
    assert c.light((280.5, y - 80)) == c.light((280.5, -1293.0)) == "out01"        # (jumping in it)
    assert c.light((280.5, -1294.0)) is None and c.light((280.5, -1183 - 155.7)) is None   # (over its dense part)
    assert c.light((280.5, -1175.0)) is None                          # (under its foot)
    wide = locate.Cover(tmp_path / "ALARM.json", None, None, [("p", (240, -1383, 323, -1176), (200, -1500, 400, -1100))])
    assert wide.light((280.5, -1300.0)) == "p" and wide.light((280.5, -1310.5)) is None   # (her body in the lit box)
    frame = np.zeros((768, 1366, 3), np.uint8)
    assert c(frame, 0.0, True, ON_SCREEN, (280.5, y)) == locate.PORTAL
    assert c(frame, 0.0, True, ON_SCREEN, (262.5, y)) is None and c(frame, 0.0, True, None, None) is None
    for her in ((900, 680, 970, 701), (900, 690, 970, 717.9), (-10, 440, 60, 484), (1320, 440, 1390, 484),
                (900, -5, 970, 39)):                                  # (her tag behind the HUD, off the frame)
        assert not locate.searched(her, frame) and c(frame, 0.0, True, her, (280.5, y)) is None, her
    assert locate.searched((900, 676.5, 970, 700), frame) and locate.searched((0, 0, 1366, 700), frame)
    assert locate.Cover(tmp_path / "ALARM.json")(frame, 0.0, True, ON_SCREEN, (280.5, y)) is None   # (no lights: none)
    monkeypatch.setattr(sw, "shown", lambda frame, cal, over=None: "main")
    c.cal = stat_window_cal()
    assert c(frame, 0.0, True, (680, 540, 720, 564), (280.5, y)) == locate.STAT_WINDOW   # (each as before it)
    (tmp_path / "ALARM.json").write_text("{}")
    assert c(frame, 1.0, False, ON_SCREEN, (280.5, y)) == locate.LIE_DETECTOR


def test_a_locators_portals_lights_from_its_map_and_their_pictures_not_to_be_had(art, model, plate, tmp_path,
                                                                                  monkeypatch):
    """Every portal with a light on the locator's map (any map's, from the game data); their pictures not to be had:
    none counted, said, the locator built as ever."""
    monkeypatch.setattr(locate, "configured", lambda path=None: True)
    monkeypatch.setattr(mapart, "load", lambda map_id: art)
    plate.save(P.model_path(tmp_path))
    loc = locate.build(model, template_of(plate), tmp_path)
    assert loc.cover.lights == [("out00", (241, -249, 324, -42), (241, -159, 324, -42)), LIGHT]
    assert not loc.cover.why_lights
    assert locate.cover().lights == []                                 # (no map: none)
    monkeypatch.setattr(mapdata, "portal_lights", lambda model: (_ for _ in ()).throw(KeyError("pv")))
    said = []
    loc = locate.build(model, template_of(plate), tmp_path, log=said.append)
    assert loc is not None and loc.cover.lights == [] and "KeyError" in loc.cover.why_lights
    assert said[0].endswith("; no portal's light (its pictures: KeyError('pv'))")


# --- at a rope's top (the simulator's nav_tree_rope_top_15: her dot alone puts her on the platform over it) -----------

def hanging(art, m: P.PlateModel, mx: float, my: float, view) -> np.ndarray:
    """A frame of the art at `view` with her plate drawn where her feet are at map (mx, my)."""
    frame, _ = art.window(*view)
    frame = frame.copy()
    cx, cy = int(round(mx - view[0] - m.feet[0])), int(round(my - view[1] - m.feet[1]))
    x0, x1, y0, y1 = m.box
    frame[cy + y0:cy + y1, cx + x0:cx + x1] = m.picture
    return frame


def test_hanging_under_a_platform_on_the_rope_from_it_her_tag_has_her_on_the_rope_not_on_the_platform(art, model,
                                                                                                       plate):
    """Her feet 15 and 25 under platform 18 on the rope at x 111 that hangs from it, still (2026-10-26's hang was 19-31
    under; 15 her dot alone puts on 18, tests/test_rope_top.py): her tag places her there, never on 18, and on the
    rope from her second frame (the first in the air: a rope is told by her moves)."""
    p18 = model.platform(18)
    rope = next(r for r in model.ropes if r.x == 111)
    for gap in (15.0, 25.0):
        loc = locator(art, model, plate)
        my = p18.y_at(rope.x) + gap
        view = (VIEW[0], int(my) - 400)
        for k in range(4):
            got = loc.see(hanging(art, plate, rope.x, my, view), k / 15, dot_of(loc, rope.x, my))
            assert got.how == "tag" and got.platform is None, (gap, k)
            assert (got.x, got.y) == pytest.approx((rope.x, my), abs=1.0)
            assert got.mode == ("air" if k == 0 else "rope"), (gap, k)
