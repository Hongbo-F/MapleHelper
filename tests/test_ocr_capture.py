from maplehelper.ocr.capture import visible_part

SCREEN = [{"left": 0, "top": 0, "width": 1920, "height": 1080}]
TWO = SCREEN + [{"left": 1920, "top": 0, "width": 1280, "height": 1024}]


def test_box_on_screen_is_unchanged():
    assert visible_part((100, 100, 800, 600), SCREEN, 10) == (100, 100, 800, 600)


def test_a_few_pixels_off_screen_are_cut_off():
    assert visible_part((-8, 100, 800, 600), SCREEN, 10) == (0, 100, 792, 600)
    assert visible_part((1200, 500, 720, 588), SCREEN, 10) == (1200, 500, 720, 580)  # 8 px past the bottom


def test_more_than_the_slack_off_screen_is_refused():
    assert visible_part((-11, 100, 800, 600), SCREEN, 10) is None
    assert visible_part((0, 0, 800, 600), SCREEN, 0) == (0, 0, 800, 600)
    assert visible_part((-1, 0, 800, 600), SCREEN, 0) is None
    assert visible_part((5000, 0, 100, 100), SCREEN, 10) is None


def test_box_spanning_two_monitors_uses_the_one_showing_most():
    # 600 px on the left monitor, 200 px on the right: far more than the slack is off "its" monitor.
    assert visible_part((1320, 100, 800, 600), TWO, 10) is None
    assert visible_part((1925, 100, 800, 600), TWO, 10) == (1925, 100, 800, 600)
    assert visible_part((1915, 100, 800, 600), TWO, 10) == (1920, 100, 795, 600)
