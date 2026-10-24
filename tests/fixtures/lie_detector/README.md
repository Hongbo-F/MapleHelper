# Every Lie Detector seen in game

The screenshots gamelab saved when MapleStory's Lie Detector came up (`gamelab/lab/lie_detector_<time>.png`, which git
ignores), kept here unchanged. Each file is named `<date>_<time>_<form>.png`, the form being the one
`perception.lie_detector` spots it as; `tests/test_alarm.py` checks every one is still spotted.

| Date | Time | Form | Map | What happened |
|---|---|---|---|---|
| 2026-10-18 | 20:47:59 | truth booth | Truth Booth | The mouse mini-game wasn't known yet. It went unanswered and she was sent here ("entry count is 1", 4:44 left). |
| 2026-10-19 | 00:47:20 | countdown | Thicket Around the Beach II | The four Violettas under spotlights, picked with buttons 1 to 4. The alarm went off at 00:46:56, during `jump_throw`; its own screenshot came out empty, so this one is 24 s later. |
| 2026-10-19 | 01:58:28 | mini-game | Thicket Around the Beach II | The mouse mini-game's notice, during `flow`. |
| 2026-10-19 | 02:54:39 | mini-game | Kerning City Construction Site | The same notice. |
| 2026-10-19 | 21:46:38 | mini-game | Thicket Around the Beach II | The same notice. |
| 2026-10-21 | 01:25:04 | mini-game | Thicket Around the Beach II | The same notice, at level 20. |

The `spot` scores (match / threshold, 1 = seen):

| Screenshot | countdown | mini-game | truth booth |
|---|---|---|---|
| 2026-10-18 20:47:59 | 0.59 | 0.50 | **1.18** |
| 2026-10-19 00:47:20 | **1.31** | 0.57 | 0.70 |
| 2026-10-19 01:58:28 | 0.57 | **1.00** | 0.58 |
| 2026-10-19 02:54:39 | 0.57 | **1.01** | 0.51 |
| 2026-10-19 21:46:38 | 0.57 | **1.01** | 0.51 |
| 2026-10-21 01:25:04 | 0.57 | **1.01** | 0.54 |

The mini-game's notices only just pass: its half-size template matches them at 0.80 to 0.81, against a threshold of 0.8.

The game's own pictures of all three Lie Detector games (the Violettas, the mouse mini-game and a third, read aloud,
not seen yet) can be extracted with `python -m gamedata pictures UI/UI_000.wz AntiMacro.img LieDetector.img`
(see gamedata's README).
