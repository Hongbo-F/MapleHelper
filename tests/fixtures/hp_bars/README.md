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
monster's as it dies, no green: nothing to read) and 400 monsters with none of their own; at 1920x1080 178 bars (1 with
no green) and 131 monsters with none. Its test skips where those frames aren't.
