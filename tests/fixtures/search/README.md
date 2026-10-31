# Where OpenCV's noise decides step 1

Searches whose answer was master's float32 noise's: inputs the review of work/search-fast made (2026-10-28,
`D:/MapleHelper-scratch/sfast/review-exact/runs/maps`: Tree Dungeon's saved frames of 10-26 with another map's sprites
blended in, searched with that map's detector), on which step 1 in C as first built found other monsters than master,
or the same in another order, because it took the exact scores where two of OpenCV's differ by its noise alone (a run
of windows whose exact scores tie, which OpenCV's noise breaks into a few peaks). `tests/test_search_c.py` checks the
C gives master's answers on them, and that without its fix (`mobs._InC._master`) it doesn't: each still meets the
noise. Cut from the review's saved inputs with `D:/MapleHelper-scratch/sfast/finish/probe/crops.py`. (The four with
West Rocky Mountain II's detector, `thicket_*.png`, went with that map's tests on 2026-10-31.)

| File | The search | Master's answer |
|---|---|---|
| `tree_half_order.png` | Tree Dungeon's detector (10002073) at half size, the whole frame (01:01's 01084) | 9 monsters, two orange mushrooms at 0.906 each in the other order |
