# Monster HP bars

Crops of frames of the game, each around a monster with the bar over it labelled by hand (2026-10-25): where the bar's
frame is, and the share of its fill green by pixel count (46 px at 1366x768; at half size of 1920x1080 from the
frame's place to the green's blended end), or where something covers its end the range the end is in, or no
monster's bar to read. `tests/test_perception.py` reads them (`perception.hud.hp_bar`).

- `n_*`: 1366x768, full-size looks: `*_png` from the frames the hunter saved on 2026-10-24
  (`gamelab/lab/shots/hp0_*.png`, `D:/MapleHelper-scratch/live/hpcheck_000.png`), the rest from the rolling
  recorder's clips of 2026-10-24 and 25 (`gamelab/lab/recent/*/NNNNN.jpg`, JPEG quality 75: the colours shift, the
  green 160 and up).
- `h_*`: 1920x1080, half-size looks (960x540) of the first map: the recordings' frames
  (`gamelab/lab/recordings/*.npz`) and `z2_shoot1.png` shrunk as a look shrinks it.

`bars.json` has each crop's size, the monster's centre `x` and top `top` in it (the detector's, or for a bar whose
monster it missed, placed from the bar: `hud.BAR_AT`), `share` (None: nothing to read), `hi` (the end hidden: the
share is between `share` and `hi`), where it's from and what's on it. `labelled.json` is the whole labelled set, by
frame (a path in `gamelab/lab`), as `[x, top, share, hi]`: at 1366x768 574 bars over their monsters (8 of them a
monster's as it dies, no green: nothing to read) and 400 monsters with none of their own; at 1920x1080 178 bars and 131
monsters with none. Its test skips where those frames aren't. Two labels were put right on review (2026-10-25):
`recent/20261025-005643-lie-detector-0055/00235.jpg`'s Green Mushroom at (629, 539), labelled full with its end hidden
(a vine's mint green over the end counted as green), is 38 of 46 px green with the HP lost's black after it, as in
00230, 00240 and 00245 (0.8261); `recordings/do_attack5.npz#36`'s pig at (296, 119), labelled a bar with no green, is
green to 0.675 under a damage number's purple halo (114-177 over red and blue, from the fill's left at 280.1 to about
302.0 of 32.3 half-size px), the red after it.

Where the frames are since 2026-10-28 (test data, `tests/testdata.py`; the test fails where a set's frames aren't, or
skips with `GAMELAB_NO_TESTDATA=1`): the names stay where the frames were in `gamelab/lab`, and the test reads them out
of the sets. At 1366x768 the rolling recorder's clips (`recent/...`, 609 frames) are gone: the recorder prunes
`lab/recent` to its newest 5 GB (it held only 2026-10-28's saves then), so that case skips, saying it needs a new
capture (the clips of a hunt of hers: a bar shows only over a monster she hit), into `gamelab/lab/testdata/
tree_dungeon_clips`; the 3 saved in `lab/shots` are in the set `tree_dungeon_drops` (2 bars, 4 monsters with none). At
1920x1080 the set `hp_bars_1920x1080` holds the 9 screenshots as saved in `gamelab/lab` and the 20 recordings of
`lab/recordings` it names with only their labelled frames (237: `frames`, and `index`, each one's number in the
recording; 93 MB for their 517). `labelled.json`'s `tree_dungeon_native` (2026-10-28) is the 118 monsters of the set
`tree_dungeon_native` (`tests/fixtures/tree_dungeon_native_monsters.json`), each at its picture's centre and top as
`eyes.bar` asks (its labelled frame's sprite placed on its feet; a stand one's for the 14 dying or with no frame
fitted), none with a bar: she didn't attack, and the bar over a monster other players hit isn't shown to her (a search
of the 24 frames for the bar's frame anywhere found none, and none by eye over the 18 flinching or dying under their
hits), with damage numbers, skills, players and their tags where a bar would be.
