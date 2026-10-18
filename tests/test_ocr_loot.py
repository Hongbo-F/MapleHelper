import numpy as np

from maplehelper.ocr.engine import Row
from maplehelper.ocr.loot import LootReader, StaticText, same_line, similar_text

STATIC, RECHECK = StaticText.STATIC_S, StaticText.RECHECK_S
TEXT, BACKGROUND = 255, 128


def row(text, y, left=20, right=120, h=20):
    return Row(text, 1.0, (left + right) / 2, y, h, left, right)


class FakeEngine:
    """Draws the rows on a grey screen as white boxes and "reads" those still white (not painted over)."""

    def __init__(self):
        self.on_screen: list[Row] = []

    def screen(self) -> np.ndarray:
        img = np.full((300, 400, 3), BACKGROUND, np.uint8)
        for r in self.on_screen:
            img[int(r.y - r.height / 2) + 2:int(r.y + r.height / 2) - 2, int(r.left) + 2:int(r.right) - 2] = TEXT
        return img

    def read(self, image, min_height=0):
        return [r for r in self.on_screen if (image[int(r.y), int(r.x)] == TEXT).all()]


def reader():
    engine = FakeEngine()
    loot = LootReader(engine)
    return engine, lambda t: loot.read(engine.screen(), now=float(t)), loot


PARTY = row("Name1", 170, left=200, right=280)
EXP = row("You received EXP (+21)", 280, left=180, right=390)


def test_similar_text_survives_ocr_jitter():
    assert similar_text("PlayerOne", "PlayerOn")
    assert similar_text("* PlayerTwo", "PlayerTwo")
    assert not similar_text("PlayerTwo", "You received EXP (+21)")
    assert same_line(row("PlayerOne", 172.7), row("PlayerOm", 173.5))
    assert not same_line(row("PlayerOne", 172), row("PlayerOne", 220))                  # another line
    assert not same_line(row("PlayerOne", 172, 0, 50), row("PlayerOne", 172, 100, 150))  # elsewhere on it


def test_text_that_stays_put_is_skipped_after_a_while():
    engine, read, _ = reader()
    engine.on_screen = [PARTY, EXP]
    for t in range(int(STATIC)):
        r = read(t)
        assert [x.text for x in r.ignored] == ["Name1"] and not r.interface
    r = read(STATIC)                     # stayed put long enough: interface from now on
    assert [x.text for x in r.interface] == ["Name1"]
    r = read(STATIC + 1)                 # painted over: not read any more
    assert r.ignored == [] and [x.text for x in r.interface] == ["Name1"]
    assert [line.kind for _, line in r.messages] == ["exp"]  # the pickup message still reads


def test_pickup_messages_are_never_skipped():
    engine, read, _ = reader()
    engine.on_screen = [EXP, row("You have gained mes", 260, left=180, right=330)]  # one fading, read in part
    for t in range(int(STATIC) * 3):
        r = read(t)
    assert not r.interface
    assert [line.kind for _, line in r.messages] == ["exp"]


def test_recheck_frees_spots_whose_text_is_gone():
    engine, read, loot = reader()
    engine.on_screen = [PARTY]
    for t in range(int(STATIC) + 1):
        read(t)
    assert loot.static.masked_rows()
    engine.on_screen = []                # left the party
    r = read(STATIC + 5)                 # still skipped: nothing to notice yet
    assert r.interface and not r.recheck
    r = read(STATIC + RECHECK)
    assert r.recheck and not r.interface  # the recheck found it gone


def test_recheck_keeps_interface_that_is_still_there():
    engine, read, _ = reader()
    engine.on_screen = [PARTY]
    for t in range(int(STATIC) + 1):
        read(t)
    r = read(STATIC + RECHECK)
    assert r.recheck and [x.text for x in r.ignored] == ["Name1"] and r.interface
    r = read(STATIC + RECHECK + 1)
    assert not r.recheck and r.ignored == []


def test_recheck_keeps_an_icon_that_only_shows_once_its_name_is_skipped():
    # With the name painted over, OCR found the leader's icon beside it, which became a spot too. Unmasked,
    # OCR reads only the name: the icon's spot is right next to it, so it stays skipped.
    engine, read, loot = reader()
    star, name = row("★", 170, left=180, right=196), row("Name1", 170, left=200, right=280)
    engine.on_screen = [name]
    for t in range(int(STATIC) + 1):
        read(t)
    engine.on_screen = [star, name]
    for t in range(int(STATIC) + 1, int(2 * STATIC) + 3):
        read(t)
    assert sorted(x.text for x in loot.static.masked_rows()) == ["Name1", "★"]
    engine.on_screen = [name]
    r = read(STATIC + RECHECK)
    assert r.recheck and sorted(x.text for x in r.interface) == ["Name1", "★"]


def test_text_that_moves_is_not_interface():
    engine, read, _ = reader()
    for t in range(int(STATIC) * 2):
        engine.on_screen = [row("Name1", 100 + 25 * (t % 5), left=200, right=280)]  # never long in one place
        r = read(t)
    assert not r.interface
