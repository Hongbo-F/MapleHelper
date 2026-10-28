# Where OpenCV's noise decides step 1

Searches whose answer was master's float32 noise's: inputs the review of work/search-fast made (2026-10-28,
`D:/MapleHelper-scratch/sfast/review-exact/runs/maps`: Tree Dungeon's saved frames of 10-26 with another map's sprites
blended in, searched with that map's detector), on which step 1 in C as first built found other monsters than master,
or the same in another order, because it took the exact scores where two of OpenCV's differ by its noise alone (a run
of windows whose exact scores tie, which OpenCV's noise breaks into a few peaks). `tests/test_search_c.py` checks the
C gives master's answers on them, and that without its fix (`mobs._InC._master`) it doesn't: each still meets the
noise. Cut from the review's saved inputs with `D:/MapleHelper-scratch/sfast/finish/probe/crops.py`.

| File | The search | Master's answer |
|---|---|---|
| `thicket_plateau_a.png` | West Rocky Mountain II's detector (10004021) at 1366x768, one band of the whole screen (rows 512-700 of 23:15's 02040): a run of tied windows took 10 of a red snail's 12 places | 2 stumps, a red snail (109, 100) at 1.0 the C as first built missed |
| `thicket_plateau_b.png` | The same detector, rows 384-614 of 23:15's 03030 | a red snail (18, 149) at 0.913, missed |
| `thicket_window.png` | The same detector, a re-find's fallback window (red snails only) of 01:01's 00960, x 67-226, y 184-340 | a red snail (79, 78) at 1.0, missed |
| `thicket_half.png` | The same map at the 1920x1080 window's half size (`screen.STRETCHED`), rows 0-161 of 01:01's 01144 | a dark stump (180, 19) at 0.625, missed, and an orange mushroom |
| `tree_half_order.png` | Tree Dungeon's detector (10002073) at half size, the whole frame (01:01's 01084) | 9 monsters, two orange mushrooms at 0.906 each in the other order |
