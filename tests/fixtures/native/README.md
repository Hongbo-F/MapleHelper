# The game at its own 1366x768

Frames of the game's client area at 1366x768 (Options, Graphics, Resolution "1366 x 768", windowed), where the game
draws its view pixel for pixel, saved on 2026-10-23 with no keys sent (`D:/MapleHelper-scratch/newmap/native`). She
stands on stair 28 of West Rocky Mountain II (map 10004021), level 23, 730/730 HP, 584/584 MP, EXP 31,736 [92.47%];
stumps on the bridge above, other players around. `tests/test_resolution.py` reads them.

| File | What's on it |
|---|---|
| `n28_000.png` | Her name tag at (685, 528), her minimap dot at (94.5, 172.5); two strangers' markers, the two portals; ropes in view at screen x 941 (map x -536) and 1174 (-303) |
| `n28_008.png` | The same, three strangers, two of them touching (85, 153 and 161) |
| `native_first.png` | The first frame at this size: three strangers elsewhere, another player's name tag (Magaodha) under the minimap |
| `tag_template_1366x768.npy` | Her name tag at this size (15 x 55, grayscale): the median of her tag's box on all 31 saved frames (the box is see-through, the clouds behind it move), matched at 0.92 or better on every one of them and at 0.39 at most anywhere else (the party list's "BugBaby" is another font). Calibration keeps it as `gamelab/lab/tag_template_1366x768.npy` and `tag_template_full_1366x768.npy` |
