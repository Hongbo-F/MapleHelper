# The Lie Detector and Violetta review pages

The builders of the web UI's review pages (`/review`, `gamelab/ui/review.py`) for the Lie Detector's two games, from the
alarm clips under `gamelab/lab/alarm_clips`. They began as the two studies' scripts; each is that script with its paths
taken out, so the pages come out as they did (the user, 2026-11-06: "looks like we should commit the page builders? The
built page can stay outside of git").

## The groups

Mini-game (`ld`, clips named `*mini-game-notice*`), each page from the white target to just before the SUCCESS banner:

- **ld-reference**: the reference track to check by eye: the best tracker with each frame judged on the 8 frames after it,
  3 runs, their median (`ld/reference.py`). The ring turns yellow where the runs disagree by over 15 px.
- **ld-outline-pf3**: the live tracker's outline (red), its particles (dots), the reference (thin green), how far off it
  is (`ld/review_pf3.py`).
- **ld-speed-compare**: the tracker before the speed work (c04c1bb, `ld/lie_target_c04c1bb.py`) and as it is today
  (`gamelab/perception/lie_target.py`) side by side, seed 0, each one's px off the reference and ms a frame
  (`ld/review_compare.py`). The ms are measured as it builds, so this group's text differs from one build to the next.

Violetta (`violetta`, clips named `*countdown*`), each page the stage as recorded and with the spotlights taken off, from
the four standing to the end, with the rules solver's four drawn on them (`violetta/build_review.py`):

- **violetta**: fitted to the sprite matches (`violetta/track5.py`).
- **violetta-yolo**: fitted to the YOLO detector's boxes (`violetta/yolo_detect.py`).

## Running it

From any folder, with the project's Python:

    python tools/review_pages/build.py ld --list                what has no page, in each group; builds nothing
    python tools/review_pages/build.py ld --missing             build those
    python tools/review_pages/build.py violetta 20261103-175221 one clip, by its stamp
    python tools/review_pages/build.py ld 20261104-180343 --force   build its pages again

A page that is there is never written over: its clip is skipped with a line saying so, unless `--force` names that clip,
and even then an approved clip's reference is not rebuilt. Steps whose output is already in the work folder are skipped,
but for the unchecked reference: building a clip's ld-reference page makes it again, over any hand fix in it.

- `--root DIR`: where the pages go. Default: where the web UI looks (`GAMELAB_REVIEW_ROOT`, else the scratch folder's
  `golden-tree` beside the repo).
- `GAMELAB_REVIEW_WORK`: the work folder. Default: `gamelab/lab/review_work`.
- `--weights FILE`: the Violetta detector. Default: `gamelab/assets/violetta.pt`, the one the game uses live. YOLO runs on
  the GPU (`device=0` in `yolo_detect.py`).

Needs the alarm clips, the live tracker's round files (`gamelab/lab/lie_target`) and the game's pictures
(`gamedata/data/out/pictures/UI/AntiMacro.img/lieDetector`).

## A clip's range

`ld/clips.json` holds the 11 clips from before the live tracker wrote round files (their board run, rate, START and end,
as first worked out). Any other clip's come from `ld/clip_range.py`: the round begun during the clip's recording gives the
first and last answers, the live tracker's own look for the board, going back from the first answer, gives the board's
first frame, and the SUCCESS banner's yellow glow gives the end. No round file or no banner: it says so and skips the clip
rather than guess. A range worked out is kept in the work folder (`ld/ranges/`), its frames in `ld/cache/`: to work one
out again, delete the clip's files in both.

## References: approved and unchecked

A reference is what the tracker is scored against on the pages. Approved ones are in git, in `ld/references/<key>.json`:
the 11 clips the user checked on 2026-11-02, each frame's centre and angle. Their centres agree with
`tests/fixtures/lie_detector/references/<key>.json`, the record of what the user approved (the tests hold them to it),
which `ld-speed-compare` reads for the centres as it always has.

Any other clip's reference is made by `ld/reference.py`, and goes to `<work>/ld/references-unchecked/<key>.json`, the only
place it writes. The pages find the approved one first, else the unchecked one, and say which they show: "the new
reference (not yet checked by you)". **The references of the 5 clips of Nov 4 and 5 (20261104-180343, -185153, -194712,
20261105-080220, -181523) are unchecked** until the user approves them.

To approve one, once the user has checked its ld-reference page and any hand fixes are in:

1. move `<work>/ld/references-unchecked/<key>.json` to `ld/references/<key>.json` (a work folder without it makes it
   again with `build.py ld <key> --force`: the same seeds gave the same file, measured on 20261104-194712);
2. add `tests/fixtures/lie_detector/references/<key>.json`, in the shape the others have: `clip`, `fps`, `first`, `start`,
   `end`, `fixed_by_hand` (the frames fixed by hand) and `centre` (frame number -> [x, y], from START up to the banner);
3. commit both. The clip's ld-outline-pf3 and ld-speed-compare pages still say "not yet checked" until built again with
   `--force`.

## In git, and not

In git: the code, `ld/clips.json`, `ld/references/`. Coordinates and frame numbers only, nothing taken from the game.

Not in git:

- the built pages (under the page root);
- the clips and the game's pictures;
- everything in the work folder: `ld/cache/` (frames and board scrolls), `ld/ranges/`, `ld/references-unchecked/`,
  `violetta/dets/` (first pass), `violetta/solved/`, `violetta/dets5/` (sprite matches), `violetta/detsY/` (YOLO).

The scripts in `violetta/` read and write those relative folders, so `build.py` runs them with `<work>/violetta` as their
working folder; run by hand, do the same.

Left in the studies' folders, as the build doesn't use them: `run_all.py` (`build.py` does its first pass, a clip at a
time), `synth.py`, its `cone_samples.npz` and the training scripts (the detector's weights are the only thing used from
training, and `yolo_detect.py` took only `synth.CROP` from `synth.py`).
