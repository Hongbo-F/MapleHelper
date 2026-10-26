# Captures that are and aren't of the game

Frames of the game's client area at 1366x768 on Tree Dungeon, Forest Up North II (map 10002073), saved by the hunter
on 2026-10-24 when her HP read 0 (`gamelab/lab/shots/hp0_*.png`, `Hunter._doubt_hp`). `tests/test_framecheck.py`
reads them.

| File | What's on it |
|---|---|
| `tree_135315.png` | A good frame (13:53:15): the minimap window at the top left, the HUD's border on row 700; her on platform 23 at x 31 by the map's minimap picture (its corner at (87, 17)), at (678, -1341) by her dot alone, off the map. The minimap window's title strip in `gamelab/assets/minimap_title.png` is cut from it |
| `tree_164845_offset.png` | 16:48:45, while the user moved the game window: the capture is offset by +26,+26 from the client area (the window's own title bar and border in the top and left rows) |
| `tree_234040_drawn.png` | 2026-10-25 23:40-41: the client the right size, 1366x768 at (2454, 107) desktop px (DPI 300%, the window 2436,22..3838,893), the game drawing its picture at the window's corner, 18 px left and 85 up (the HUD's border on row 615, x 267-1062; the minimap's title cut off; under the HUD and right of x 1348 the desktop behind). Cut, in real px, from a desktop grab round the window (`D:/MapleHelper-scratch/winfix/offset_2340_desktop_grab.png`, 20 px margin: the client at (38, 105) in it). Alt+Enter twice set it right |
