# Every Lie Detector seen in game

The screenshots gamelab saved when MapleStory's Lie Detector came up (`gamelab/lab/lie_detector_<time>.png`, which git
ignores), kept here unchanged, and since 2026-10-25 frames of the rolling recorder's (`.jpg`, as it recorded them: the
notice and the image quiz, which raised no alarm then). Each file is named `<date>_<time>_<form>.png` (or `.jpg`), the
form being the one `perception.lie_detector` spots it as; `tests/test_alarm.py` checks every one is still spotted, as
that form. `without/` has frames of the same scenes just before or after, where nothing may be seen.

| Date | Time | Form | Map | What happened |
|---|---|---|---|---|
| 2026-10-18 | 20:47:59 | truth booth | Truth Booth | The mouse mini-game wasn't known yet. It went unanswered and she was sent here ("entry count is 1", 4:44 left). |
| 2026-10-19 | 00:47:20 | countdown | Thicket Around the Beach II | The four Violettas under spotlights, picked with buttons 1 to 4. The alarm went off at 00:46:56, during `jump_throw`; its own screenshot came out empty, so this one is 24 s later. |
| 2026-10-19 | 01:58:28 | mini-game notice | Thicket Around the Beach II | The mouse mini-game's notice, during `flow`. |
| 2026-10-19 | 02:54:39 | mini-game notice | Kerning City Construction Site | The same notice. |
| 2026-10-19 | 21:46:38 | mini-game notice | Thicket Around the Beach II | The same notice. |
| 2026-10-21 | 01:25:04 | mini-game notice | Thicket Around the Beach II | The same notice, at level 20. |
| 2026-10-24 | 21:02:00 | mini-game | Tree Dungeon, Forest Up North II | The board, at the game's own 1366x768: the alarm. The test failed over Chrome Remote Desktop's lag (the Truth Booth). |
| 2026-10-24 | 23:51:21 | mini-game notice | Tree Dungeon, Forest Up North II | The notice at 1366x768, 2 s into its 10 (the rolling recorder's frame); unseen then. |
| 2026-10-24 | 23:51:29 | mini-game | Tree Dungeon, Forest Up North II | Its board, a star to follow: the alarm, 10 s after the notice. Answered. |
| 2026-10-25 | 00:27:58 | quiz warning | Tree Dungeon, Forest Up North II | The image quiz's warning, "Question in 29 seconds"; unseen then, the hunt going on. |
| 2026-10-25 | 00:28:26 | quiz warning | Tree Dungeon, Forest Up North II | The same, "Question in 1 second". |
| 2026-10-25 | 00:28:28 | quiz | Tree Dungeon, Forest Up North II | "Click all matching images": two magnifiers and a wizard hat, "Remain : 9 Sec". Answered. |
| 2026-10-25 | 00:28:33 | quiz | Tree Dungeon, Forest Up North II | The same, both magnifiers picked (green), Confirm lit. |
| 2026-10-25 | 00:55:01 | mini-game notice | Tree Dungeon, Forest Up North II | The notice, 1 s into its 10; unseen then. |
| 2026-10-25 | 00:55:10 | mini-game | Tree Dungeon, Forest Up North II | Its board, a square to follow: the alarm. Answered. |

The `spot` scores (match / threshold, 1 = seen):

| Frame | countdown | mini-game notice | mini-game | quiz warning | quiz | truth booth |
|---|---|---|---|---|---|---|
| 2026-10-18 20:47:59 | 0.59 | 0.30 | 0.50 | 0.49 | 0.53 | **1.18** |
| 2026-10-19 00:47:20 | **1.31** | 0.30 | 0.57 | 0.58 | 0.58 | 0.70 |
| 2026-10-19 01:58:28 | 0.57 | **1.11** | 1.00 | 0.54 | 0.58 | 0.58 |
| 2026-10-19 02:54:39 | 0.57 | **1.11** | 1.01 | 0.57 | 0.60 | 0.51 |
| 2026-10-19 21:46:38 | 0.57 | **1.11** | 1.01 | 0.59 | 0.59 | 0.51 |
| 2026-10-21 01:25:04 | 0.57 | **1.11** | 1.01 | 0.55 | 0.58 | 0.54 |
| 2026-10-24 21:02:00 | 0.47 | 0.89 | **1.25** | 0.61 | 0.61 | 0.56 |
| 2026-10-24 23:51:21 | 0.51 | **1.11** | 0.93 | 0.62 | 0.61 | 0.56 |
| 2026-10-24 23:51:29 | 0.46 | 0.89 | **1.25** | 0.59 | 0.61 | 0.56 |
| 2026-10-25 00:27:58 | 0.40 | 0.31 | 0.48 | **1.37** | 0.61 | 0.56 |
| 2026-10-25 00:28:26 | 0.44 | 0.29 | 0.48 | **1.25** | 0.61 | 0.56 |
| 2026-10-25 00:28:28 | 0.47 | 0.33 | 0.48 | 0.60 | **1.43** | 0.56 |
| 2026-10-25 00:28:33 | 0.47 | 0.31 | 0.48 | 0.60 | **1.42** | 0.56 |
| 2026-10-25 00:55:01 | 0.51 | **1.11** | 0.93 | 0.58 | 0.61 | 0.56 |
| 2026-10-25 00:55:10 | 0.37 | 0.89 | **1.25** | 0.58 | 0.61 | 0.56 |
| without/ (the four) | 0.41-0.47 | 0.33-0.34 | 0.48 | 0.56-0.62 | 0.61 | 0.56-0.63 |

The notice and the board share the Maple Administrator's portrait and the title "The Lie Detector Mini-game". At
1920x1080 the board's template caught the notices only just (0.80 to 0.81 against its 0.8); at 1366x768 it scored
them 0.75, and the notices of 2026-10-24 and 25 went unseen. Since 2026-10-25 the notice has its own template, which
scores the board 0.89 of its threshold (the board is told by its own, 1.25).

The game's own pictures of all three Lie Detector games (the Violettas, the mouse mini-game and a third, read aloud,
not seen yet) can be extracted with `python -m gamedata pictures UI/UI_000.wz AntiMacro.img LieDetector.img`
(see gamedata's README). The image quiz's box isn't among these.
