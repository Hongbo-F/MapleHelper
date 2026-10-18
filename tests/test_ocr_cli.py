import io

from maplehelper.ocr.__main__ import Console, LootReport, RowMemory
from maplehelper.ocr.engine import Row
from maplehelper.ocr.loot import LootRead
from maplehelper.ocr.parse import parse_loot


def loot_read(rows, interface=()):
    parsed = [(r, parse_loot(r.text)) for r in rows]
    return LootRead(list(rows), [(r, l) for r, l in parsed if l], [r for r, l in parsed if not l], list(interface), False)


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_terminal_rewrites_repeats_in_place():
    out = Terminal()
    c = Console(out)
    c.update("missing", "12:00:00  no game window")
    c.update("missing", "12:00:01  no game")
    c.update("found", "12:00:02  HP 50/50")
    c.end()
    assert out.getvalue() == "12:00:00  no game window\r12:00:01  no game       \n12:00:02  HP 50/50\n"


def test_lines_stay_and_close_the_rewritten_line():
    out = Terminal()
    c = Console(out)
    c.update("status", "waiting")
    c.line("You received EXP (+21)")
    c.update("status", "waiting")
    c.end()
    assert out.getvalue() == "waiting\nYou received EXP (+21)\nwaiting\n"


def test_file_output_skips_repeats():
    out = io.StringIO()
    c = Console(out)
    for _ in range(3):
        c.update("missing", "no game window")
    c.update("found", "HP 50/50")
    c.end()
    assert out.getvalue() == "no game window\nHP 50/50\n"


def rows(*texts):
    return [Row(t, 1.0, 0, i * 20.0, 12, 0, 0) for i, t in enumerate(texts)]


def at(text, y, h=24.0):
    return Row(text, 1.0, 0, y, h, 0, 0)


def test_row_memory_ignores_ocr_jitter_on_a_line_that_stays():
    # Party panel lines as OCR read them on consecutive reads in the game (names replaced).
    reads = [[at("Name1", 172.7), at("* Name2", 197.7)], [at("Name1", 172.5), at("*Name2", 197.2)],
             [at("Name", 173.5), at("Name2", 197.9)], [at("米", 196.1, 15.3), at("Name2", 197.7)],
             [at("Name1", 173.5), at("米", 196.1, 15.3), at("Name2", 197.7)]]
    memory = RowMemory()
    new = [row.text for t, read in enumerate(reads) for row in read if memory.is_new(row, now=float(t))]
    assert new == ["Name1", "* Name2", "米"]


def test_row_memory_forgets_lines_gone_for_a_while():
    memory = RowMemory()
    assert memory.is_new(at("Name1", 100), now=0.0)
    assert not memory.is_new(at("Name1", 100), now=5.0)
    assert memory.is_new(at("Name1", 100), now=5.0 + RowMemory.FORGET_S + 1)


def test_row_memory_tells_different_lines_apart():
    memory = RowMemory()
    assert memory.is_new(at("You received EXP (+21)", 100), now=0.0)
    assert memory.is_new(at("Name1", 100), now=0.5)          # same place, different text
    assert memory.is_new(at("You received EXP (+21)", 160), now=1.0)  # same text, moved up a few lines


def test_rows_output_shows_each_line_once():
    out = io.StringIO()
    report = LootReport(Console(out), show_rows=True)
    for t in range(5):
        report(loot_read([at("Name1" if t % 2 else "Name", 172.7), at("* Name2", 197.7)]), "t")
    shown = [line for line in out.getvalue().splitlines() if "ignored" in line]
    assert len(shown) == 2


def test_loot_lists_each_message_when_it_shows_up():
    out = io.StringIO()
    report = LootReport(Console(out), show_rows=False)
    report(loot_read(rows("You received EXP (+21)", "Party: Rogue Lv.19")), "t")
    report(loot_read(rows("You received EXP (+21)")), "t")                           # still on screen: not again
    report(loot_read(rows("You received EXP (+21)", "You received EXP (+21)")), "t")  # the same gain again
    listed = [line for line in out.getvalue().splitlines() if "exp" in line and "message" not in line]
    assert len(listed) == 2


def test_loot_shows_a_name_left_cut_off():
    out = io.StringIO()
    LootReport(Console(out), show_rows=False)(loot_read(rows("Bubbling's Hug...x1 earned. (Etc)")), "t")
    assert "item  Bubbling's Hug... x1 " in out.getvalue()


def test_interface_lines_are_announced_once():
    out = io.StringIO()
    report = LootReport(Console(out), show_rows=False)
    party = at("Name1", 172.7)
    for _ in range(3):
        report(loot_read([], interface=[party]), "t")
    text = out.getvalue()
    assert text.count("skipping \"Name1\"") == 1
    assert "1 interface line(s) skipped" in text
