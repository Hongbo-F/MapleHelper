# Her, facing a known way

Crops of her around her name tag, side by side in a strip, for `tests/test_heading.py` (perception.heading: which way
she faces, off her picture above the tag). `labels.json` says each crop's size, where her tag's centre is in it, the
window size it's of, and which way she faces in each, in order.

| File | What's on it |
|---|---|
| `native.png` | 20 frames at 1366x768 (the game's own size), her outfit of 2026-10-24, 58x72 px each, her tag's centre at (29, 78): `D:/MapleHelper-scratch/newmap/native` (`n28_*`, West Rocky Mountain II, 2026-10-23), `map3/native` (`first_000`, `here_*`, Tree Dungeon), and `gamelab/lab/shots` (`hp0_*`: hit, on her chair; `nm_*`: by the portal, behind its glow). 15 face left, 5 right, labelled by eye (her eye patch is on the side she faces) and, on the three behind the portal's glow (`nm_equip`, `nm_keys`, `nm_verify`), where the eye had them wrong, by the method's leave-one-out (the eye patch shows on the right when looked at again). The tag positions are the 2026-10-24 study's (`D:/MapleHelper-scratch/heading/explore/tags.json`); `nm_pet7` is left out, its "tag" there being the Character Info window's label under her portrait. Five scenes (`n28`, Tree Dungeon's, `hp0_1353*`, `hp0_1648*`, `nm_*`): a test reads each with pictures of the others. The seed in `gamelab/assets` is these but `n28_*` |
| `old.png` | 23 frames of the old recordings (the first map, 1920x1080 looks at half size, her outfit of 2026-10-18 to 21), one per recording and way, 42x52 px each, her tag's centre at (21, 56), labelled by the keys sent: the last arrow pressed 130 ms or more before the frame (`gamelab/lab/recordings`: walks, the jump backs' turns in mid-air, taps, throws, `knockback`, chains) |
