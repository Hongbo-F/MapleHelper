"""Lie Detector auto-stop: spotting it, stopping the player, and refusing to run until cleared."""

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gamelab.control import alarm, client
from gamelab.perception import framecheck, lie_detector
from gamelab.perception.lie_detector import LieDetectorSpotter
from gamelab.__main__ import LIE_DETECTOR_EXIT, main
from gamelab.control.player import LIE_DETECTOR_ERROR, Player, Watchdog, _sleep_until

FIXTURES = Path(__file__).parent / "fixtures"


def img(name):
    return np.asarray(Image.open(FIXTURES / name).convert("RGB"))


SCENE = img("game_scene.png")            # daffodils and other yellow scenery
WINDOW = img("lie_detector_window.png")  # the real Lie Detector countdown window (top part)
NEAR_MISS = img("near_miss.png")         # the key bindings window's orange CANCEL button


def scene_with_window():
    frame = np.zeros((700, 1100, 3), np.uint8)
    frame[:SCENE.shape[0], :SCENE.shape[1]] = SCENE
    frame[300:300 + WINDOW.shape[0], 600:600 + WINDOW.shape[1]] = WINDOW
    return frame


@pytest.fixture
def alarm_file(tmp_path, monkeypatch):
    from gamelab.control import player
    path = tmp_path / "ALARM.json"
    for mod in (alarm, client, player):
        monkeypatch.setattr(mod, "ALARM", path)
    return path


def test_spots_the_lie_detector_anywhere_on_screen():
    spotter = LieDetectorSpotter()
    assert spotter.seen(scene_with_window())
    assert spotter.score(WINDOW) > 0.95


def test_spots_the_mouse_mini_game_and_the_truth_booth():
    # The mini-game was only captured at half size: enlarge it back to the game's size.
    half = img("lie_detector_minigame_half.png")
    mini_game = np.asarray(Image.fromarray(half).resize((half.shape[1] * 2, half.shape[0] * 2), Image.BILINEAR))
    spotter = LieDetectorSpotter()
    assert spotter.scores(mini_game)["mini-game"] >= 1
    assert spotter.seen(mini_game)
    assert spotter.scores(img("lie_detector/2026-10-18_204759_truth_booth.png"))["truth booth"] >= 1
    # the clock alone is every timed map's: the Truth Booth is the clock and the Maple Administrator under it
    assert spotter.scores(img("truth_booth_timer.png"))["truth booth"] < 1


# <date>_<time>_<form>.png (.jpg: the rolling recorder's frames, as recorded), see its README
SEEN_IN_GAME = sorted((FIXTURES / "lie_detector").glob("*.png")) + sorted((FIXTURES / "lie_detector").glob("*.jpg"))
WITHOUT = sorted((FIXTURES / "lie_detector" / "without").glob("*.jpg"))  # the same scenes just before or after
TIMED = sorted((FIXTURES / "lie_detector" / "timed").glob("*.png"))  # maps with the game's clock, not the Truth Booth
FORMS = {"countdown": "countdown", "mini_game_notice": "mini-game notice", "mini_game": "mini-game",
         "quiz_warning": "quiz warning", "quiz": "quiz", "truth_booth": "truth booth"}


@pytest.mark.parametrize("png", SEEN_IN_GAME, ids=lambda p: p.stem)
def test_spots_every_lie_detector_seen_in_game(png):
    form = FORMS[png.stem.split("_", 2)[2]]
    assert LieDetectorSpotter().scores(img(png))[form] >= 1


@pytest.mark.parametrize("png", SEEN_IN_GAME, ids=lambda p: p.stem)
def test_spots_every_lie_detector_seen_in_game_at_the_game_s_own_size_too(png):
    # Since 2026-10-23 the game's window is 1366x768, pixel for pixel: the 1920x1080 screens shrunk to it.
    form = FORMS[png.stem.split("_", 2)[2]]
    native = np.asarray(Image.fromarray(img(png)).resize((1366, 768), Image.BOX))
    assert LieDetectorSpotter().scores(native)[form] >= 1


def test_the_lie_detectors_seen_in_game_are_all_there():
    assert len(SEEN_IN_GAME) >= 15 and len(WITHOUT) >= 4
    assert {FORMS[png.stem.split("_", 2)[2]] for png in SEEN_IN_GAME} == set(FORMS.values())


@pytest.mark.parametrize("png", SEEN_IN_GAME, ids=lambda p: p.stem)
def test_each_lie_detector_seen_in_game_is_seen_as_its_own_form(png):
    """The notice and the board share the portrait and the title (the notice's template scores the board 0.80 of
    its 0.9): each is told for what it is, as are the quiz's warning and the quiz."""
    scores = LieDetectorSpotter().scores(img(png))
    assert max(scores, key=scores.get) == FORMS[png.stem.split("_", 2)[2]]


@pytest.mark.parametrize("name, left", [
    ("lie_detector/2026-10-18_204759_truth_booth.png", 4 * 60 + 44),
    ("lie_detector/2026-11-03_000250_truth_booth.png", 4 * 60 + 27),
    ("lie_detector/timed/2026-10-31_235840_kerning_pq_first_stage.png", 29 * 60 + 59),
    ("lie_detector/2026-10-25_002828_quiz.jpg", None), ("game_scene.png", None)])
def test_the_clock_s_time_left_is_read_off_its_digits(name, left):
    """The Truth Booth's stay read off its clock (`clock_left`; the user, 2026-11-03: "When you enter the truth booth
    you can see how much count down it has"): 4:44 (1920x1080, the game's stretch), 4:27 (1366x768), the party
    quest's 29:59 (two minutes' digits), at both sizes; none with no clock, nor on a frame smaller than the game's."""
    spotter = LieDetectorSpotter()
    frame = img(name)
    assert spotter.clock_left(frame) == left
    if frame.shape[:2] == (1080, 1920):
        assert spotter.clock_left(native(frame)) == left


@pytest.mark.parametrize("png", TIMED, ids=lambda p: p.stem)
def test_a_timed_map_s_clock_is_not_the_truth_booth(png):
    """The game's "Time Left" clock shows on every map with a time limit, at the Truth Booth's place (2026-10-31 23:58,
    the Kerning party quest's first stage: the alarm raised as the Truth Booth, at 1.02 of the clock's threshold). The
    clock is seen there, at both sizes, and with no Maple Administrator under it the Truth Booth isn't, nor anything."""
    spotter = LieDetectorSpotter()
    clock, threshold = spotter.half["truth booth"]
    for frame in (img(png), native(img(png))):
        grey = spotter._halved(spotter._stretched(frame))
        assert cv2.matchTemplate(grey, clock, cv2.TM_CCOEFF_NORMED).max() >= threshold
        for scores in (spotter.scores(frame), spotter.near(frame)):
            assert scores["truth booth"] < 0.8 and max(scores.values()) < 0.8, scores


def test_the_lie_detectors_timed_maps_are_there():
    assert len(TIMED) >= 1


@pytest.mark.parametrize("frame", [SCENE, NEAR_MISS, np.zeros((540, 960, 3), np.uint8)])
def test_ignores_yellow_scenery_and_other_windows(frame):
    assert not LieDetectorSpotter().seen(frame)


@pytest.mark.parametrize("jpg", WITHOUT, ids=lambda p: p.stem)
def test_sees_nothing_in_the_same_scenes_just_before_the_notice_or_the_quiz_s_warning_or_after_the_quiz(jpg):
    scores = LieDetectorSpotter().scores(img(jpg))
    assert max(scores.values()) < 0.8, scores


def numpy_yellow(rgb):
    """lie_detector.yellow as it was: numpy's tests on int16 copies."""
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    return ((r > 180) & (g > 140) & (b < 110) & (r - b > 110)).astype(np.float32)


def test_the_title_yellow_is_the_same_as_numpys_on_every_colour():
    g, b = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8), indexing="ij")
    for r0 in range(0, 256, 16):
        picture = np.empty((16 * 256, 256, 3), np.uint8)
        picture[..., 0] = np.repeat(np.arange(r0, r0 + 16, dtype=np.uint8), 256)[:, None]
        picture[..., 1], picture[..., 2] = np.tile(g, (16, 1)), np.tile(b, (16, 1))
        new = lie_detector.yellow(picture)
        assert new.dtype == np.float32 and np.array_equal(new, numpy_yellow(picture))


def test_spotter_scores_are_the_same_as_with_numpys_yellow(monkeypatch):
    half = img("lie_detector_minigame_half.png")
    mini_game = np.asarray(Image.fromarray(half).resize((half.shape[1] * 2, half.shape[0] * 2), Image.BILINEAR))
    frames = [scene_with_window(), WINDOW, mini_game, img("truth_booth_timer.png"), SCENE, NEAR_MISS,
              np.zeros((540, 960, 3), np.uint8)]
    new = [LieDetectorSpotter().scores(f) for f in frames]
    monkeypatch.setattr(lie_detector, "yellow", numpy_yellow)
    old = LieDetectorSpotter()
    assert new == [old.scores(f) for f in frames]


def native(frame):
    """A 1920x1080 frame shrunk to the game's own 1366x768, as test_spots_every_lie_detector_seen_in_game_at_the_game_s_
    own_size_too shrinks it; a 1366x768 one as it is."""
    return frame if frame.shape[:2] == (768, 1366) else np.asarray(Image.fromarray(frame).resize((1366, 768), Image.BOX))


@pytest.mark.parametrize("png", SEEN_IN_GAME + WITHOUT + TIMED, ids=lambda p: p.stem)
def test_the_look_near_the_forms_places_scores_every_lie_detector_seen_in_game_as_the_whole_frame_does(png):
    """The watchdog's look (`near`): each form matched only around where it has always shown, on the pictures the
    whole frame's look matches, scores it as that look does where it's seen, at both sizes, and nothing higher."""
    spotter = LieDetectorSpotter()
    for frame in (img(png), native(img(png))):
        whole, near = spotter.scores(frame), spotter.near(frame)
        assert near.keys() == whole.keys() and all(near[f] <= whole[f] + 1e-4 for f in whole)
        if png.parent.name not in ("without", "timed"):
            form = FORMS[png.stem.split("_", 2)[2]]
            assert near[form] == pytest.approx(whole[form], abs=1e-4) and near[form] >= 1


@pytest.mark.parametrize("png", SEEN_IN_GAME + WITHOUT + TIMED, ids=lambda p: p.stem)
def test_its_windows_over_the_play_area_are_seen_at_their_places_and_nothing_else(png):
    """`over_play` (her place by the scene counted covered under them: localization.locate.Cover): the countdown's
    window, the mini-game's board and the quiz, at both sizes; the boxes that come before them (the mini-game's notice,
    the quiz's warning), the Truth Booth's timer and the scenes without them: none; another size: none."""
    form = None if png.parent.name in ("without", "timed") else FORMS[png.stem.split("_", 2)[2]]
    want = form if form in lie_detector.OVER_PLAY else None
    spotter = LieDetectorSpotter()
    for frame in (img(png), native(img(png))):
        assert spotter.over_play(frame) == want
    assert spotter.over_play(np.asarray(Image.fromarray(img(png)).resize((1600, 900), Image.BOX))) is None


CLEAN = "lie_detector/without/2026-10-25_005500_before_the_notice.jpg"  # 1366x768, its capture anchors in place


def test_the_spotter_s_kept_pictures_change_no_score():
    """Its stretched frame, its half and its grey are kept from look to look (8 MB of fresh memory a look before, page
    faults the kernel zeroes): a frame's scores are those of a spotter that never looked, whatever size came before."""
    countdown = img("lie_detector/2026-10-19_004720_countdown.png")
    frames = [img(CLEAN), countdown, native(countdown), SCENE, img("lie_detector/2026-10-25_002828_quiz.jpg")]
    spotter = LieDetectorSpotter()
    for frame in frames + frames[::-1]:
        assert spotter.scores(frame) == LieDetectorSpotter().scores(frame)
        assert spotter.near(frame) == LieDetectorSpotter().near(frame)


def pasted(png: str, form: str, to: tuple[int, int]) -> np.ndarray:
    """A clean scene of the game's own size (its capture anchors in place) with `form`'s window, cut around its place
    from a saved Lie Detector screen brought to that size (`native`), pasted at `to` (x, y, game px): a form somewhere
    it has never shown."""
    scene = img(CLEAN).copy()
    k = lie_detector.TEMPLATE_FRAME[0] / lie_detector.NATIVE_FRAME[0]
    (x, y), = lie_detector.PLACES[form][:1]
    if form == "countdown":
        h, w = LieDetectorSpotter().template.shape
    else:
        h, w = (2 * n for n in LieDetectorSpotter().half[form][0].shape)
    x0, y0, x1, y1 = (round(v / k) for v in (x - 8, y - 8, x + w + 8, y + h + 8))
    cut = native(img(f"lie_detector/{png}"))[y0:y1, x0:x1]
    scene[to[1]:to[1] + cut.shape[0], to[0]:to[0] + cut.shape[1]] = cut
    return scene


def drawn_off(frame: np.ndarray, dx: int = -18, dy: int = -85) -> np.ndarray:
    """A 1366x768 frame as the game drawing its picture off its client shows it (2026-10-25 23:40: 18 px left and 85
    up, black where nothing is drawn)."""
    out = np.zeros_like(frame)
    out[:frame.shape[0] + dy, :frame.shape[1] + dx] = frame[-dy:, -dx:]
    return out


def watching(tmp_path, frames, spotter=None) -> tuple[Watchdog, int]:
    """A Watchdog on `frames` in turn, looked with until its alarm: (it, how many looks that took)."""
    frames = iter(frames)
    flag = threading.Event()
    dog = Watchdog("MapleStory", flag, spotter=spotter or LieDetectorSpotter(), grab=lambda: next(frames),
                   on_alarm=lambda f: None, folder=tmp_path, log=lambda m: None)
    looks = 0
    while not flag.is_set():
        dog.tick(looks * 0.25)
        looks += 1
    dog.recording.finish("stopped")
    return dog, looks


@pytest.mark.parametrize("png, form, to", [("2026-10-19_004720_countdown.png", "countdown", (200, 480)),
                                           ("2026-10-19_015828_mini_game_notice.png", "mini-game notice", (140, 210))])
def test_a_lie_detector_somewhere_it_never_showed_is_seen_by_the_watchdog_s_whole_frame_look(tmp_path, png, form, to):
    """The look near the forms' places doesn't see a form elsewhere; the whole frame's, every WHOLE_EVERY-th look, does:
    the alarm within that many looks of its first showing (2 s at 4 a second)."""
    frame = pasted(png, form, to)
    spotter = LieDetectorSpotter()
    assert spotter.scores(frame)[form] >= 1 and max(spotter.near(frame).values()) < 1
    assert not framecheck.anchors_off(frame)  # (the picture in its place: `near` is what most looks ask)
    dog, looks = watching(tmp_path, [img(CLEAN)] + [frame] * 20)  # (the first look, of the whole frame, sees none)
    assert looks == 1 + lie_detector.WHOLE_EVERY and dog.form == form  # seen at its 8th look, the 9th in all


@pytest.mark.parametrize("png, form", [("2026-10-25_002828_quiz.jpg", "quiz"),
                                       ("2026-10-25_002758_quiz_warning.jpg", "quiz warning"),
                                       ("2026-10-25_005501_mini_game_notice.jpg", "mini-game notice")])
def test_a_lie_detector_on_a_picture_drawn_off_the_client_is_seen_at_its_first_look(tmp_path, png, form):
    """The game drawing its picture off its client moves every form off its places, where `near` looks: such a frame
    (neither capture anchor in its place) is looked at whole, so the form is seen at the first look of it."""
    frame = drawn_off(img(f"lie_detector/{png}"))
    spotter = LieDetectorSpotter()
    assert spotter.scores(frame)[form] >= 1 and max(spotter.near(frame).values()) < 1
    assert framecheck.anchors_off(frame) and not framecheck.anchors_off(img(f"lie_detector/{png}"))
    dog, looks = watching(tmp_path, [img(CLEAN)] * 3 + [frame] * 20)
    assert looks == 4 and dog.form == form


def test_the_watchdog_looks_at_the_whole_frame_every_whole_every_th_look_and_in_a_cycle_and_off_the_places(tmp_path):
    class Spy:
        def __init__(self):
            self.calls, self.up = [], False

        def scores(self, frame):
            self.calls.append("whole")
            return {"countdown": 1.5 if self.up else 0.1}

        def near(self, frame):
            self.calls.append("near")
            return {"countdown": 1.5 if self.up else 0.1}

    spy, clean = Spy(), img(CLEAN)
    shown = {"frame": clean}
    dog = Watchdog("MapleStory", threading.Event(), spotter=spy, grab=lambda: shown["frame"], on_alarm=lambda f: None,
                   folder=tmp_path, log=lambda m: None)
    looks = iter(range(100))

    def look(n: int, frame) -> list[str]:
        spy.calls.clear()
        shown["frame"] = frame
        for _ in range(n):
            dog.tick(next(looks) * 0.25)
        return spy.calls

    assert look(17, clean) == (["whole"] + ["near"] * 7) * 2 + ["whole"]  # (looks 1-17)
    import cv2
    assert look(2, cv2.resize(clean, lie_detector.TEMPLATE_FRAME)) == ["whole"] * 2  # (no anchors at another size)
    assert look(1, SCENE) == ["whole"]
    assert look(2, drawn_off(clean)) == ["whole"] * 2  # (its picture off the client: neither anchor in its place)
    minimap_closed = clean.copy()
    minimap_closed[:30, :100] = 0  # (one anchor is enough: the HUD's border in its place)
    assert look(2, minimap_closed) == ["near"] * 2  # (looks 23 and 24)
    assert look(2, clean) == ["whole", "near"]
    dog.anchors_off = lambda frame: 1 / 0  # (a check that fails: the whole frame)
    assert look(1, clean) == ["whole"]
    dog.anchors_off = framecheck.anchors_off
    spy.up = True
    assert look(4, clean) == ["near", "whole", "whole", "whole"] and dog.cycle_t0 is not None  # in a cycle: whole
    dog.recording.finish("stopped")


def test_watchdog_grabs_the_game_as_an_rgb_picture_of_its_own(monkeypatch):
    from gamelab.control import player as player_mod
    rng = np.random.default_rng(0)
    bgra = rng.integers(0, 256, (1080, 1920, 4), np.uint8)
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "minimized", lambda hwnd: False)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 1920, "height": 1080})
    on_top = {"game": True}
    monkeypatch.setattr(player_mod.window, "shows", lambda hwnd, area: on_top["game"])

    class Screen:
        def grab(self, box):
            return bgra

    dog = Watchdog("MapleStory", threading.Event(), spotter=object())
    frame = dog._grab(Screen())
    assert frame.flags.c_contiguous and np.array_equal(frame, bgra[:, :, 2::-1])
    # Another window over some of the game: what's on top is still handed over (checked for the Lie Detector,
    # which may show round it), said to be covered.
    on_top["game"] = False
    frame, why = dog._grab(Screen())
    assert why == "covered" and np.array_equal(frame, bgra[:, :, 2::-1])


@pytest.mark.parametrize("size", [(3840, 2160), (1600, 900)])
def test_watchdog_brings_a_client_of_another_size_to_the_templates_size_and_sees_the_lie_detector_there(monkeypatch,
                                                                                                       size):
    """The game full screen at 3840x2160 (Alt+Enter twice puts it there on purpose: control.window_size), or resized:
    grabbed as it was, no saved Lie Detector screen was seen at such a size (0.53-0.78 of the thresholds), and nothing
    raised the alarm. Brought to 1920x1080 (lie_detector.to_template) it's seen; the game's own size and 1920x1080 are
    handed over as they are."""
    import cv2

    from gamelab.control import player as player_mod
    quiz = img("lie_detector/2026-10-25_002828_quiz.jpg")
    shown = cv2.resize(quiz, size, interpolation=cv2.INTER_LINEAR)
    area = {"left": 0, "top": 0, "width": size[0], "height": size[1]}
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "minimized", lambda hwnd: False)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: dict(area))
    monkeypatch.setattr(player_mod.window, "shows", lambda hwnd, area: True)

    class Screen:
        def grab(self, box):
            return cv2.cvtColor(shown, cv2.COLOR_RGB2BGRA)

    spotter = LieDetectorSpotter()
    assert max(spotter.scores(shown).values()) < 1  # (as it was grabbed before)
    dog = Watchdog("MapleStory", threading.Event(), spotter=spotter)
    frame = dog._grab(Screen())
    assert frame.shape == (1080, 1920, 3) and spotter.seen(frame)
    shown = quiz  # (1366x768: as it is, the spotter stretching it itself)
    area.update(width=1366, height=768)
    assert dog._grab(Screen()).shape == (768, 1366, 3)


def test_watchdog_says_why_it_has_no_picture_the_game_minimized_or_gone(monkeypatch):
    # Minimized, the game's client area isn't 0 x 0 (432x54 on 2026-10-24): Windows is asked.
    from gamelab.control import player as player_mod
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "minimized", lambda hwnd: True)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": -32000, "top": -32000, "width": 432,
                                                                        "height": 54})
    dog = Watchdog("MapleStory", threading.Event(), spotter=object())
    assert dog._grab(None) == "minimized"

    def no_game(title):
        raise RuntimeError("no window titled 'MapleStory'")

    monkeypatch.setattr(player_mod.window, "find_game", no_game)
    assert dog._grab(None) == "no window"


def test_watchdog_sets_the_alarm_first_then_reports_once(tmp_path):
    frames = iter([SCENE, SCENE, scene_with_window(), scene_with_window(), SCENE])
    flag, reports, said = threading.Event(), [], []
    dog = Watchdog("MapleStory", flag, grab=lambda: next(frames), on_alarm=lambda frame: reports.append(flag.is_set()),
                   folder=tmp_path, log=said.append)
    for t in range(5):
        dog.tick(t * 0.25)
    assert flag.is_set() and reports == [True]  # the alarm was already up when it reported, and once a cycle
    assert dog.recording is not None and dog.recording.grabs == 3  # and the watchdog goes on, recording it
    dog.recording.finish("stopped")
    assert dog.recording.join(timeout=10)


@pytest.mark.parametrize("before, seen, form, title", [
    ("2026-10-25_005500_before_the_notice", "2026-10-25_005501_mini_game_notice.jpg", "mini-game notice",
     "MapleStory Lie Detector mini-game in 10 s"),
    ("2026-10-25_002757_before_the_quiz_warning", "2026-10-25_002758_quiz_warning.jpg", "quiz warning",
     "MapleStory image quiz in 30 s"),
    ("2026-10-25_002757_before_the_quiz_warning", "2026-10-25_002828_quiz.jpg", "quiz",
     "MapleStory image quiz on screen")])
def test_the_notice_and_the_quiz_raise_the_alarm_named_for_what_was_seen(alarm_file, monkeypatch, tmp_path, before,
                                                                         seen, form, title):
    """On 2026-10-24 and 25 the alarm went off at the mini-game's board, 10 s after its notice, and not at all for the
    image quiz (30 s of warning, then 10 s to answer): the hunt went on through both. Now the first of them seen is
    the alarm, as the board is: input stopped, paged by name, the cycle recorded."""
    pages, said = [], []
    monkeypatch.setattr(alarm, "page", lambda title, text: pages.append(title))
    frames = iter([img(f"lie_detector/without/{before}.jpg"), img(f"lie_detector/{seen}"), img(f"lie_detector/{seen}")])
    flag = threading.Event()
    dog = Watchdog("MapleStory", flag, grab=lambda: next(frames), folder=tmp_path / "lie_detector", log=said.append)
    dog.tick(0.0)
    assert not flag.is_set() and not alarm_file.exists()
    dog.tick(0.25)
    info = json.loads(alarm_file.read_text())
    assert flag.is_set() and info["reason"] == "lie_detector" and info["form"] == form and pages == [title]
    assert any(m.startswith(f"LIE DETECTOR seen ({form}): input stopped") for m in said)
    dog.tick(0.5)
    dog.recording.finish("stopped")
    assert dog.recording.join(timeout=10) and dog.recording.forms[form]["frames"] == 2
    Path(info["screenshot"]).unlink()


def test_the_alarm_without_a_form_is_paged_as_before(alarm_file, monkeypatch):
    pages = []
    monkeypatch.setattr(alarm, "page", lambda title, text: pages.append((title, text)))
    for form in (None, "countdown", "truth booth"):
        info = alarm.raise_alarm(None, form=form)
        assert info.get("form") == form and "cause" not in info
    assert pages == [("MapleStory Lie Detector on screen", "gamelab stopped all input. Answer it in the game "
                                                            "yourself.")] * 3


def test_the_alarm_records_its_cause(alarm_file, monkeypatch):
    """The Truth Booth with no test seen before it (the watchdog's lie_recorder.UNKNOWN_CAUSE): in the alarm file from
    its first write, for the rolling recorder (it saves the 5 minutes before it at once)."""
    monkeypatch.setattr(alarm, "page", lambda title, text: None)
    writes = []
    real = alarm._write
    monkeypatch.setattr(alarm, "_write", lambda info: (writes.append(dict(info)), real(info)))
    info = alarm.raise_alarm(None, form="truth booth", cause="unknown")
    assert info["cause"] == "unknown" and writes[0]["cause"] == "unknown"
    assert alarm.alarm_info()["cause"] == "unknown" and alarm.alarm_info()["form"] == "truth booth"


def test_watchdog_keeps_watching_while_all_is_well(tmp_path):
    flag = threading.Event()
    dog = Watchdog("MapleStory", flag, grab=lambda: SCENE, interval=0.01, on_alarm=lambda f: None, folder=tmp_path,
                   log=lambda m: None)
    dog.start()
    time.sleep(0.1)
    assert dog.is_alive() and not flag.is_set()
    dog.stopped.set()
    dog.join(timeout=2)


def test_player_refuses_plans_once_the_alarm_is_up():
    flag = threading.Event()
    flag.set()
    # Returns before looking for the game window or pressing anything.
    assert Player({}, "no such window", flag).run({"events": [[0, "f", "down"]]}) == {"error": LIE_DETECTOR_ERROR}


def test_waits_end_early_when_the_alarm_goes_off():
    flag = threading.Event()
    threading.Timer(0.05, flag.set).start()
    started = time.perf_counter()
    assert _sleep_until(time.perf_counter() + 5, flag) is False
    assert time.perf_counter() - started < 0.5
    assert _sleep_until(time.perf_counter() + 0.01) is True


def _player_sending(monkeypatch, tmp_path, flag=None):
    from gamelab.control import player as player_mod
    from maplehelper.keys import parse
    monkeypatch.setattr(player_mod, "INBOX", tmp_path)
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 1366, "height": 768})
    monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "Game")
    monkeypatch.setattr(player_mod.win32, "send_key", lambda key, down: None)
    return Player({"f": parse("f"), "g": parse("g")}, "Game", flag)


def test_a_plan_s_keys_go_out_at_their_times(monkeypatch, tmp_path):
    """The key thread sleeps between its checks and spins only before a key (player.FINAL_S): each key goes out at its
    time, not before, and to within a few ms (here without the 1 ms timer serve asks for: a sleep on the alarm can
    end a tick, 15.6 ms, late)."""
    events = [[0, "f", "down"], [30, "f", "up"], [95, "g", "down"], [160, "g", "up"], [233, "f", "down"],
              [420, "f", "up"]]
    p = _player_sending(monkeypatch, tmp_path)
    result = p.run({"id": "keys", "events": events, "capture": {"lead_ms": 0}})
    late = [sent[0] - due[0] for sent, due in zip(result["sent"], events)]
    assert len(late) == len(events) and min(late) >= 0 and np.median(late) < 5, late


def test_a_wait_for_a_key_ends_at_once_when_the_alarm_goes_off(monkeypatch, tmp_path):
    """Between its checks the key thread sleeps on the alarm itself: the alarm wakes it (it was seen within 10 ms)."""
    from gamelab.control.player import Alarm
    flag = Alarm()
    p = _player_sending(monkeypatch, tmp_path, flag)
    threading.Timer(0.1, flag.set).start()
    result = p.run({"id": "long", "events": [[0, "f", "down"], [3000, "f", "up"]], "capture": {"lead_ms": 0}})
    assert result["error"] == LIE_DETECTOR_ERROR and time.monotonic() - flag.at < 0.05


def test_raised_alarm_blocks_gamelab_until_cleared(alarm_file, monkeypatch, tmp_path):
    from gamelab.control import lie_recorder
    monkeypatch.setattr(lie_recorder, "LIE_RECORDINGS", tmp_path / "lie_detector")  # (clear-alarm reads them)
    pages = []
    monkeypatch.setattr(alarm, "page", lambda title, text: pages.append(title))
    info = alarm.raise_alarm(scene_with_window())
    assert pages == ["MapleStory Lie Detector on screen"]
    assert json.loads(alarm_file.read_text())["reason"] == "lie_detector"
    assert Path(info["screenshot"]).exists()
    Path(info["screenshot"]).unlink()

    with pytest.raises(client.LieDetectorAlarm):
        client.submit([], name="x")
    assert main(["act", "100"]) == LIE_DETECTOR_EXIT  # refused before touching the game
    assert main(["player", "--elevate"]) == LIE_DETECTOR_EXIT  # nor is the player started again

    assert main(["clear-alarm"]) == 0
    assert not alarm_file.exists()
    client.check_alarm()  # no longer raises


def test_a_plan_can_be_cut_short_and_lets_go_of_its_keys(tmp_path, monkeypatch):
    from gamelab.control import player as player_mod
    from maplehelper.keys import parse
    sent = []
    monkeypatch.setattr(player_mod, "INBOX", tmp_path)
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 1920, "height": 1080})
    monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "Game")
    monkeypatch.setattr(player_mod.win32, "send_key", lambda key, down: sent.append((key.name, down)))
    p = Player({"f": parse("f")}, "Game")
    threading.Timer(0.1, lambda: (tmp_path / "plan1.cancel").touch()).start()
    started = time.perf_counter()
    result = p.run({"id": "plan1", "events": [[0, "f", "down"], [3000, "f", "up"]], "capture": {"lead_ms": 0}})
    assert time.perf_counter() - started < 1.0
    assert result["problems"] == ["cancelled"]
    assert sent == [("f", True), ("f", False)]           # pressed, then let go when cut short
    assert not (tmp_path / "plan1.cancel").exists()       # the signal is used up


def test_a_plan_can_last_past_its_last_key(tmp_path, monkeypatch):
    from gamelab.control import player as player_mod
    from maplehelper.keys import parse
    monkeypatch.setattr(player_mod, "INBOX", tmp_path)
    monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
    monkeypatch.setattr(player_mod.window, "client_rect", lambda hwnd: {"left": 0, "top": 0, "width": 1920, "height": 1080})
    monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "Game")
    monkeypatch.setattr(player_mod.win32, "send_key", lambda key, down: None)
    p = Player({"f": parse("f")}, "Game")
    started = time.perf_counter()
    p.run({"id": "p2", "events": [[0, "f", "down"], [10, "f", "up"]], "end_ms": 300, "capture": {"lead_ms": 0}})
    assert 0.28 <= time.perf_counter() - started < 0.6
    with pytest.raises(ValueError):
        player_mod.validate({"events": [[0, "f", "down"], [500, "f", "up"]], "end_ms": 100}, {"f": parse("f")})


# --- what still goes with the alarm up: her pet's food and her retreat (control.alarm.PLAN_KINDS) -------------------

def _keys(pet="u"):
    from maplehelper.config import GameKeys
    from maplehelper.keys import parse
    return GameKeys(left=parse("left"), right=parse("right"), jump=parse("alt"), attack=parse("ctrl"),
                    lucky_seven=parse("shift"), up=parse("up"), down=parse("down"), pick_up=parse("z"),
                    pet_food=parse(pet) if pet else None)


def _tap(key, t=0):
    return [[t, key, "down"], [t + 40, key, "up"]]


def test_with_the_alarm_up_a_pet_food_plan_is_one_tap_of_its_key_and_nothing_else():
    from gamelab.control.player import alarm_refusal
    keys = _keys()

    def food(events, **kw):
        return {"kind": "pet_food", "events": events, **kw}
    assert alarm_refusal(food(_tap("u")), keys, 30.0, [], 100.0) is None
    for events in (_tap("u") + _tap("u", 100), _tap("u") + _tap("left", 100), _tap("ctrl"), [[0, "u", "down"]],
                   [[0, "u", "down"], [500, "u", "up"]], [[0, "mouse", 5, 5]] + _tap("u"), _tap("lbutton"), []):
        assert alarm_refusal(food(events), keys, 30.0, [], 100.0) is not None, events
    assert "no mouse" in alarm_refusal(food([[0, "mouse", 5, 5]] + _tap("u")), keys, 30.0, [], 100.0)
    assert alarm_refusal(food(_tap("u"), capture={"fps": 30}), keys, 30.0, [], 100.0) is not None
    assert alarm_refusal(food(_tap("u"), window_client=[1366, 768]), keys, 30.0, [], 100.0) is not None
    assert alarm_refusal(food(_tap("u")), _keys(pet=None), 30.0, [], 100.0) == "no pet food key is bound"
    # A few a minute: the fourth within 60 s is refused, the next once the first is a minute old.
    assert "a minute at most" in alarm_refusal(food(_tap("u")), keys, 30.0, [50.0, 70.0, 90.0], 100.0)
    assert alarm_refusal(food(_tap("u")), keys, 30.0, [40.0, 70.0, 90.0], 100.0) is None
    # Any plan without a kind, or of another, is refused as before.
    assert alarm_refusal({"events": _tap("u")}, keys, 1.0, [], 100.0) is not None
    assert alarm_refusal({"kind": "throw", "events": _tap("u")}, keys, 1.0, [], 100.0) is not None


def test_with_the_alarm_up_a_plan_let_through_holds_the_player_no_longer_than_it_has_to():
    """Review, 2026-10-24: a pet food plan with a lead of 1.5 s was taken and tapped at 1.5 s, one with end_ms 2000
    held the player 2 s (and `validate` bounds a lead only by the frames recorded). Whole within TAP_MAX_MS, from its
    start: its lead, its tap, its end; a retreat's lead too (its moves end with its window)."""
    from gamelab.control.player import TAP_MAX_MS, alarm_refusal
    keys = _keys()

    def food(events=None, **kw):
        return {"kind": "pet_food", "events": events or _tap("u"), **kw}
    for ok in (food(), food(capture={"lead_ms": 0}), food(capture={"lead_ms": TAP_MAX_MS}), food(end_ms=TAP_MAX_MS),
               food(_tap("u", TAP_MAX_MS - 40))):
        assert alarm_refusal(ok, keys, 30.0, [], 100.0) is None, ok
    for late in (food(capture={"lead_ms": 1500}), food(capture={"lead_ms": -1}), food(capture={"lead_ms": "0"}),
                 food(end_ms=2000), food(_tap("u", 19_000))):
        assert alarm_refusal(late, keys, 30.0, [], 100.0) is not None, late
    run = [[0, "left", "down"], [1500, "left", "up"]]
    assert alarm_refusal({"kind": "retreat", "events": run, "end_ms": 3000}, keys, 1.0, [], 100.0) is None
    assert "starts within" in alarm_refusal({"kind": "retreat", "events": run, "capture": {"lead_ms": 5000}}, keys,
                                            1.0, [], 100.0)


def test_with_the_alarm_up_a_retreat_is_her_moves_alone_for_its_window_with_every_key_let_go():
    from gamelab.control import alarm as alarm_mod
    from gamelab.control.player import alarm_refusal
    keys = _keys()
    run = [[0, "left", "down"], [100, "alt", "down"], [140, "alt", "up"], [600, "left", "up"],
           [700, "up", "down"], [1500, "up", "up"], [1600, "down", "down"], [1700, "down", "up"]]

    def retreat(events):
        return {"kind": "retreat", "events": events}
    assert alarm_refusal(retreat(run), keys, 0.5, [], 100.0) is None
    assert alarm_refusal(retreat(run), keys, alarm_mod.RETREAT_WINDOW_S - 0.01, [], 100.0) is None
    assert "are over" in alarm_refusal(retreat(run), keys, alarm_mod.RETREAT_WINDOW_S + 0.01, [], 100.0)
    assert "isn't known" in alarm_refusal(retreat(run), keys, None, [], 100.0)
    for other in ("ctrl", "shift", "z", "u", "lbutton"):  # no attack, Lucky Seven, pick-up, pet food or click
        assert alarm_refusal(retreat(run + _tap(other, 1800)), keys, 1.0, [], 100.0) is not None, other
    assert "held at its end" in alarm_refusal(retreat(run[:-1]), keys, 1.0, [], 100.0)
    assert "no mouse" in alarm_refusal(retreat([[0, "mouse", 10, 10]] + run), keys, 1.0, [], 100.0)
    assert alarm_mod.RETREAT_WINDOW_S > alarm_mod.RETREAT_S  # her 10 s retreat fits


def test_with_the_alarm_up_the_lie_cursor_is_one_mouse_move_inside_the_mini_game_board():
    from gamelab.control import alarm as alarm_mod
    from gamelab.control.player import alarm_refusal
    keys = _keys()

    def cursor(events=None, **kw):
        return {"kind": alarm_mod.LIE_CURSOR,
                "events": [[0, "mouse", 309, 182]] if events is None else events,
                "focus": False, "capture": {"lead_ms": 0}, **kw}

    assert alarm_refusal(cursor(), keys, None, [], 100.0) is None
    assert alarm_refusal(cursor([[0, "mouse", 1057, 679]]), keys, None, [], 100.0) is None
    for x, y in ((307, 180), (1058, 180), (308, 179), (308, 680), (10, 10)):
        assert "inside its board" in alarm_refusal(cursor([[0, "mouse", x, y]]), keys, None, [], 100.0)
    for events in ([[10, "mouse", 309, 182]], [[0, "mouse", 309, 182], [0, "mouse", 400, 300]], []):
        assert "one mouse move" in alarm_refusal(cursor(events), keys, None, [], 100.0), events
    for plan in (cursor(focus=True), cursor(capture={"lead_ms": 1}), cursor(end_ms=100),
                 cursor(capture={"lead_ms": 0, "fps": 1}), cursor(window_client=[1366, 768])):
        assert alarm_refusal(plan, keys, None, [], 100.0) is not None, plan

    # Any mouse-button event selects Violetta's strict click form, never the move-only form's board-wide allowance.
    for events in ([[0, "lbutton", "down"]], [[0, "mouse", 309, 182], [10, "lbutton", "down"]],
                   [[0, "mouse", 580, 505], [1000, "lbutton", "down"], [1080, "lbutton", "up"]]):
        assert alarm_refusal(cursor(events), keys, None, [], 100.0) is not None, events
    click = [[0, "mouse", 527, 505], [1000, "lbutton", "down"], [1080, "lbutton", "up"]]
    assert alarm_refusal(cursor(click), keys, None, [], 100.0) is None


def test_the_alarm_keeps_when_the_lie_detector_was_first_seen():
    from gamelab.control.player import Alarm
    flag = Alarm()
    assert flag.at is None and not flag.is_set()
    before = time.monotonic()
    flag.set()
    first = flag.at
    assert flag.is_set() and before <= first <= time.monotonic()
    flag.set()  # seen again: the sighting it counts from stays the first
    assert flag.at == first


class _Game:
    """The player's way to the game faked: the window there and in front, the keys it sends kept."""

    def __init__(self, monkeypatch, tmp_path):
        from gamelab.control import player as player_mod
        self.sent, self.fronted = [], []
        monkeypatch.setattr(player_mod, "INBOX", tmp_path)
        monkeypatch.setattr(player_mod.window, "find_game", lambda title: 1)
        monkeypatch.setattr(player_mod.window, "client_rect",
                            lambda hwnd: {"left": 0, "top": 0, "width": 1366, "height": 768})
        monkeypatch.setattr(player_mod.window, "bring_to_front", lambda title: self.fronted.append(title))
        monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "Game")
        monkeypatch.setattr(player_mod.win32, "send_key", lambda key, down: self.sent.append((key.name, down)))
        monkeypatch.setattr(player_mod.win32, "send_mouse_button", lambda name, down: self.sent.append((name, down)))


def _alarmed_player(seen_ago: float = 1.0):
    from gamelab.control.player import Alarm, allowed_keys
    flag, said = Alarm(), []
    flag.set()
    flag.at = time.monotonic() - seen_ago
    keys = _keys()
    return Player(allowed_keys(keys), "Game", flag, keys=keys, log=said.append), said


def _plan(pid, events, kind="pet_food"):
    return {"id": pid, "kind": kind, "events": events, "focus": True, "capture": {"lead_ms": 0}}


def test_the_elevated_player_runs_the_lie_cursor_trail_and_reports_windows_failures(monkeypatch, tmp_path):
    from gamelab.control import alarm as alarm_mod
    from gamelab.control import player as player_mod
    game = _Game(monkeypatch, tmp_path)
    p, said = _alarmed_player()
    moves = []
    monkeypatch.setattr(player_mod.window, "client_rect",
                        lambda hwnd: {"left": 100, "top": 50, "width": 1366, "height": 768})

    def move(x, y, wait, played):
        moves.append((x, y, wait(time.perf_counter())))
        played.extend([[1700000000.0, 380, 220], [1700000000.008, x, y]])  # (screen px)
        return True

    monkeypatch.setattr(player_mod.window, "move_cursor", move)
    plan = _plan("cursor", [[0, "mouse", 309, 182]], alarm_mod.LIE_CURSOR)
    plan["focus"] = False
    result = p.run(plan)
    assert "error" not in result and result["sent"][0][1:] == ["mouse", 309, 182]
    assert moves == [(409, 232, True)] and game.sent == [] and game.fronted == [] and said == []
    assert result["trail"] == [[1700000000.0, 280, 170], [1700000000.008, 309, 182]]  # where it went, in client px

    def denied(x, y, wait, played):
        played.append([1700000000.0, 380, 220])  # (the start, then SendInput refused)
        raise OSError(5, "blocked")

    monkeypatch.setattr(player_mod.window, "move_cursor", denied)
    plan["id"] = "cursor_denied"
    result = p.run(plan)
    assert result["sent"] == [] and result["trail"] == [[1700000000.0, 280, 170]]
    assert result["problems"] == [f"cursor movement failed (OSError: {OSError(5, 'blocked')})"]
    assert game.sent == [] and game.fronted == []


def test_with_the_alarm_up_the_player_feeds_her_pet_and_refuses_the_rest(monkeypatch, tmp_path):
    game = _Game(monkeypatch, tmp_path)
    p, said = _alarmed_player()
    result = p.run(_plan("food1", _tap("u")))
    assert "error" not in result and game.sent == [("u", True), ("u", False)]
    assert game.fronted == []  # never brought forward with the alarm up: whoever is at the PC is answering it
    assert said == ["Lie Detector alarm up: her pet's food (food1), 1 in the last minute"]
    # The pet food key with anything else, a hunt's plan, a click: refused before anything is pressed.
    for pid, events, kind in (("food2", _tap("u") + _tap("ctrl", 100), "pet_food"), ("throw", _tap("ctrl"), None),
                              ("click", _tap("lbutton"), "pet_food")):
        assert p.run(_plan(pid, events, kind))["error"].startswith(LIE_DETECTOR_ERROR), pid
    assert p.run({"id": "throw", "events": _tap("ctrl")}) == {"error": LIE_DETECTOR_ERROR}  # as before
    assert game.sent == [("u", True), ("u", False)]
    assert [line.split(": ")[1] for line in said[1:]] == ["refused food2 (pet_food)", "refused click (pet_food)"]
    # A few a minute.
    for i in range(2, 4):
        assert "error" not in p.run(_plan(f"food{i + 1}", _tap("u")))
    assert "a minute at most" in p.run(_plan("food5", _tap("u")))["error"]


def test_with_the_alarm_up_a_pet_food_plan_that_would_hold_the_player_is_refused_at_once(monkeypatch, tmp_path):
    game = _Game(monkeypatch, tmp_path)
    p, said = _alarmed_player()
    started = time.perf_counter()
    late = dict(_plan("food1", _tap("u")), capture={"lead_ms": 1500})
    held = dict(_plan("food2", _tap("u")), end_ms=2000)
    for plan in (late, held):
        assert p.run(plan)["error"].startswith(LIE_DETECTOR_ERROR)
    assert time.perf_counter() - started < 0.2 and game.sent == [] and p.fed == []


def test_with_the_alarm_up_her_retreat_is_played_until_its_window_ends_and_refused_after(monkeypatch, tmp_path):
    from gamelab.control import alarm as alarm_mod
    from gamelab.control.player import RETREAT_OVER
    game = _Game(monkeypatch, tmp_path)
    p, said = _alarmed_player(seen_ago=alarm_mod.RETREAT_WINDOW_S - 0.2)  # 0.2 s of it left
    started = time.perf_counter()
    result = p.run(_plan("flow1", [[0, "left", "down"], [1500, "left", "up"]], "retreat"))
    assert 0.1 < time.perf_counter() - started < 0.6  # cut at the window's end, not played its 1.5 s
    assert "error" not in result and result["problems"] == [RETREAT_OVER]
    assert game.sent == [("left", True), ("left", False)]  # let go of when cut
    assert game.fronted == [] and "her retreat's plan flow1" in said[0]
    late = p.run(_plan("flow2", _tap("left"), "retreat"))
    assert late["error"].startswith(LIE_DETECTOR_ERROR) and "are over" in late["error"]
    assert game.sent == [("left", True), ("left", False)]


def test_with_the_alarm_file_up_but_unseen_by_the_player_its_kinds_go_by_the_alarm_s_rules(monkeypatch, tmp_path,
                                                                                           alarm_file):
    """Review, 2026-10-24: a plan with a kind gets past the client's check of the alarm file, and a player whose own
    watchdog hadn't seen the Lie Detector (`watch --lie-detector` raised it) played it as any: the game brought to the
    front while the test was answered, no limit a minute, no window, no key check."""
    from gamelab.control import alarm as alarm_mod
    from gamelab.control import player as player_mod
    from gamelab.control.player import Alarm, allowed_keys
    game = _Game(monkeypatch, tmp_path)
    keys, said = _keys(), []
    p = Player(allowed_keys(keys), "Game", Alarm(), keys=keys, log=said.append)  # its own alarm not set
    alarm_file.write_text(json.dumps({"reason": "lie_detector (seen by watch)", "at": time.time() - 2.0}),
                          encoding="utf-8")
    for i in range(3):
        assert "error" not in p.run(_plan(f"food{i}", _tap("u")))
    assert "a minute at most" in p.run(_plan("food3", _tap("u")))["error"]
    assert "one tap of 'u'" in p.run(_plan("food4", _tap("u") + _tap("ctrl", 100)))["error"]
    assert "error" not in p.run(_plan("flow", [[0, "left", "down"], [100, "left", "up"]], "retreat"))
    assert "isn't one of her moves' keys" in p.run(_plan("flow2", _tap("ctrl"), "retreat"))["error"]
    assert game.fronted == [] and game.sent == [("u", True), ("u", False)] * 3 + [("left", True), ("left", False)]
    assert said[0] == "Lie Detector alarm up (its file's; this player hasn't seen it): her pet's food (food0), 1 in " \
                      "the last minute"
    # Seen past the window by the file's `at`, or when isn't known: no retreat.
    alarm_file.write_text(json.dumps({"at": time.time() - alarm_mod.RETREAT_WINDOW_S - 1}), encoding="utf-8")
    assert "are over" in p.run(_plan("flow3", _tap("left"), "retreat"))["error"]
    alarm_file.write_text(json.dumps({"reason": "lie_detector"}), encoding="utf-8")
    assert "isn't known" in p.run(_plan("flow4", _tap("left"), "retreat"))["error"]
    # The file gone, a plan of a kind plays as any (the game brought forward if it asks).
    alarm_file.unlink()
    monkeypatch.setattr(player_mod.win32, "foreground_title", lambda: "Game" if game.fronted else "other")
    assert "error" not in p.run(_plan("food5", _tap("u")))
    assert game.fronted == ["Game"]


def test_a_plan_running_when_the_alarm_goes_off_is_stopped_whatever_its_kind(monkeypatch, tmp_path):
    from gamelab.control.player import Alarm, allowed_keys
    game = _Game(monkeypatch, tmp_path)
    flag, keys = Alarm(), _keys()
    p = Player(allowed_keys(keys), "Game", flag, keys=keys, log=lambda m: None)
    threading.Timer(0.1, flag.set).start()
    started = time.perf_counter()
    result = p.run(_plan("flow", [[0, "left", "down"], [3000, "left", "up"]], "retreat"))
    assert time.perf_counter() - started < 1.0 and result["error"] == LIE_DETECTOR_ERROR
    assert game.sent == [("left", True), ("left", False)]


def test_the_client_hands_the_lie_cursor_to_the_player_without_focus_capture_or_input_else(alarm_file, monkeypatch):
    from gamelab.control import alarm as alarm_mod
    alarm_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(client, "done", lambda pid: True)
    monkeypatch.setattr(client, "wait",
                        lambda pid, timeout: {"sent": [[0, "mouse", 309, 182]], "problems": [],
                                              "trail": [[1700000000.0, 300, 170], [1700000000.008, 309, 182]]})
    trail = []
    assert client.move_lie_cursor(309, 182, active=lambda: True, trail=trail) is True
    assert trail == [[1700000000.0, 300, 170], [1700000000.008, 309, 182]]  # the player's, handed back
    path = next(client.INBOX.glob("*lie_cursor*.json"))
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert plan == {"events": [[0, "mouse", 309, 182]],
                    "capture": {"fps": 0, "duration_ms": 0, "scale": 1, "lead_ms": 0},
                    "focus": False, "kind": alarm_mod.LIE_CURSOR}
    assert client.cancel(path.stem)  # test plan only: no player is running


def test_an_inactive_lie_cursor_is_withdrawn_without_waiting_for_a_result(alarm_file, monkeypatch):
    alarm_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(client, "done", lambda pid: False)

    def no_wait(pid, timeout):
        raise AssertionError("a plan never picked up has no result to wait for")

    monkeypatch.setattr(client, "wait", no_wait)
    assert client.move_lie_cursor(309, 182, active=lambda: False) is False
    assert not list(client.INBOX.glob("*lie_cursor*.json"))


def test_an_inactive_lie_cursor_already_running_is_cancelled_and_collected(monkeypatch):
    calls = []
    monkeypatch.setattr(client, "send", lambda *a, **kw: "cursor1")
    monkeypatch.setattr(client, "done", lambda pid: False)
    monkeypatch.setattr(client, "cancel", lambda pid: calls.append(("cancel", pid)) or False)
    monkeypatch.setattr(client, "wait", lambda pid, timeout: calls.append(("wait", pid)) or {
        "problems": ["cancelled"], "trail": [[1700000000.0, 300, 170]]})
    trail = []
    assert client.move_lie_cursor(309, 182, active=lambda: False, trail=trail) is False
    assert calls == [("cancel", "cursor1"), ("wait", "cursor1")]
    assert trail == [[1700000000.0, 300, 170]]  # as far as it went


def test_the_client_sends_only_the_named_kinds_with_the_alarm_up(alarm_file):
    alarm_file.write_text(json.dumps({"reason": "lie_detector", "time": "2026-10-24 21:02:00",
                                      "at": time.time() - 2.0}), encoding="utf-8")
    with pytest.raises(client.LieDetectorAlarm):
        client.send(_tap("ctrl"), name="throw")
    with pytest.raises(ValueError):
        client.send(_tap("u"), name="x", kind="throw")
    pid = client.send(_tap("u"), name="pet_food", kind="pet_food")
    plan = json.loads((client.INBOX / f"{pid}.json").read_text(encoding="utf-8"))
    assert plan["kind"] == "pet_food" and plan["events"] == _tap("u")
    assert client.alarm_up() and 1.9 < client.alarm_age() < 5.0
    # Waited for, it isn't withdrawn as a plan without a kind is (the player decides): no player, no answer.
    with pytest.raises(client.PlayerError, match="no result"):
        client.wait(pid, timeout=0.05)
    alarm_file.unlink()
    assert not client.alarm_up() and client.alarm_age() is None


def test_a_plan_refused_for_the_lie_detector_raises_its_alarm_even_before_its_file_is_written(alarm_file):
    for error in (LIE_DETECTOR_ERROR, f"{LIE_DETECTOR_ERROR}: 3 pet foods a minute at most with the alarm up"):
        pid = client.send([], name="x")
        client.OUTBOX.mkdir(parents=True, exist_ok=True)
        (client.OUTBOX / f"{pid}.json").write_text(json.dumps({"error": error}), encoding="utf-8")
        with pytest.raises(client.LieDetectorAlarm, match="Lie Detector on screen"):
            client.wait(pid, timeout=1)


# --- her keys followed to where the game has them (control.keymap): the alarm's allowance by the file's keys ----------

def _followed_player(tmp_path, names: dict | None, seen_ago: float = 1.0, unsafe=None):
    """A player with the alarm up 1 s ago, config.toml's keys (_keys), and an alarm keys file holding `names` (None: no
    file) that it reads again whenever it changes (alarm.AlarmKeys)."""
    from gamelab.control import alarm as alarm_mod
    from gamelab.control.player import Alarm, allowed_keys
    path = tmp_path / "alarm_keys.json"
    if names is not None:
        path.write_text(json.dumps({"keys": names, "character": "thief", "at": "2026-10-28 16:00:00"}),
                        encoding="utf-8")
    flag, said = Alarm(), []
    flag.set()
    flag.at = time.monotonic() - seen_ago
    keys = _keys()
    p = Player(allowed_keys(keys), "Game", flag, keys=keys, log=said.append, keys_file=path,
               unsafe=alarm_mod.ALARM_UNSAFE if unsafe is None else unsafe)
    return p, said, path


def test_with_her_keys_followed_the_alarm_still_lets_through_only_her_moves_and_pet_food(monkeypatch, tmp_path):
    """Her jump followed onto C and her pet food onto I (the game's Key Bindings read: control.keymap.publish wrote the
    file): with the alarm up the player takes her retreat's moves by C and a tap of I, and still nothing else: not her
    old jump (Alt now does something else in the game), not her attack, Lucky Seven, pick-up or old pet food key, no
    mouse."""
    game = _Game(monkeypatch, tmp_path)
    p, said, _ = _followed_player(tmp_path, {"jump": "c", "pet_food": "i"})
    run = [[0, "left", "down"], [100, "c", "down"], [140, "c", "up"], [600, "left", "up"]]
    assert "error" not in p.run(_plan("flow", run, "retreat"))
    assert "error" not in p.run(_plan("food", _tap("i")))
    assert game.sent == [("left", True), ("c", True), ("c", False), ("left", False), ("i", True), ("i", False)]
    for pid, events, kind in (("old_jump", _tap("alt"), "retreat"), ("attack", _tap("ctrl"), "retreat"),
                              ("l7", _tap("shift"), "retreat"), ("pick", _tap("z"), "retreat"),
                              ("old_food", _tap("u"), "pet_food"), ("click", _tap("lbutton"), "retreat"),
                              ("food_jump", _tap("c"), "pet_food"), ("hunt", _tap("c"), None)):
        assert p.run(_plan(pid, events, kind))["error"].startswith(LIE_DETECTOR_ERROR), pid
    assert len(game.sent) == 6
    assert any("from alarm_keys.json: left left, right right, jump c, up up, down down, pet_food i" in line
               for line in said)


@pytest.mark.parametrize("names, why", [
    ({"jump": "c", "attack": "a"}, "not only her moves and pet food"),        # another of her keys: never the file's
    ({"jump": "1"}, "a key not pressed with the alarm up"),                    # a digit (the Violetta quiz's buttons)
    ({"pet_food": "enter"}, "a key not pressed with the alarm up"),            # Enter (a form's)
    ({"jump": "esc"}, "a key not pressed with the alarm up"),
    ({"jump": "f2"}, "a key not pressed with the alarm up"),                   # maplehelper's hotkey (the player's)
    ({"jump": "left"}, "two of her moves on one key"),
    ({"pet_food": "alt"}, "her pet food on one of her moves' keys"),
    ({"jump": None}, "her arrows or jump on no key"),
    ({"jump": "lbutton"}, "not a key config.toml can name"),
    ({"jump": 7}, "not a key config.toml can name"),
])
def test_an_alarm_keys_file_not_as_her_keys_can_be_is_left_out(monkeypatch, tmp_path, names, why):
    """A file that doesn't hold only her moves' and pet food's keys as they can be (alarm.alarm_keys_problem) is left
    out, said once, and config.toml's keys stand: what goes with the alarm up never widens."""
    from gamelab.control import alarm as alarm_mod
    game = _Game(monkeypatch, tmp_path)
    p, said, _ = _followed_player(tmp_path, names, unsafe=(*alarm_mod.ALARM_UNSAFE, "f1", "f2", "f3", "f4", "f12"))
    assert p.allowance() == _keys() and why in p.alarm_keys.why
    assert "error" not in p.run(_plan("flow", _tap("alt"), "retreat"))   # (config.toml's moves)
    assert p.run(_plan("x", _tap("a"), "retreat"))["error"].startswith(LIE_DETECTOR_ERROR)
    assert sum("left out" in line for line in said) == 1 and game.sent == [("alt", True), ("alt", False)]


def test_the_player_reads_the_alarm_keys_file_again_when_it_changes(monkeypatch, tmp_path):
    """No file: config.toml's. Written (a hunt followed her jump): the file's, at the next plan, without the player
    started again (its UAC click). Broken or gone: config.toml's again."""
    _Game(monkeypatch, tmp_path)
    p, said, path = _followed_player(tmp_path, None)
    assert p.allowance().jump.name == "alt"
    path.write_text(json.dumps({"keys": {"jump": "c"}}), encoding="utf-8")
    assert p.allowance().jump.name == "c" and p.allowance().left.name == "left"
    path.write_text("{not json", encoding="utf-8")
    assert p.allowance().jump.name == "alt" and "not read" in p.alarm_keys.why
    path.unlink()
    assert p.allowance().jump.name == "alt" and p.alarm_keys.why is None


def test_a_player_without_the_file_takes_config_toml_s_keys_as_before():
    from gamelab.control.player import Alarm, allowed_keys
    keys = _keys()
    p = Player(allowed_keys(keys), "Game", Alarm(), keys=keys)
    assert p.alarm_keys is None and p.allowance() is keys
