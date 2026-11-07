"""A mini-game clip's frame range, for a clip newer than clips.json, from what the live tracker wrote (its round file,
paths.LIE_TARGETS, a line an answer with the frame numbers the alarm clip's index has too) and the SUCCESS banner's
yellow glow on the board. On the 11 clips with recorded ends the glow's first frame is that end, on all 11."""
import json
from datetime import datetime

import cv2
import numpy as np

START_S = 4.6  # s from the board's first frame (the frame before the target's first white answer) to START
PAST_ANSWERS = 25  # frames past the last answer the banner is looked for in
AFTER_END = 21  # frames kept past the banner's first (the sets stop before it, the frame cache a little after)
GLOW = 300  # px: yellow over the frames' usual, in two frames running, that is the banner (its fade-in's first two
#            frames were 467 px at least on the 17 clips with a banner, every frame before them 0)


def round_file(clip, rounds):
    """The live tracker's file for this alarm clip: the last round begun within its recording (summary.json's times,
    in the files' own stamp form), or None."""
    s = json.load(open(clip / "summary.json", encoding="utf-8"))
    lo, hi = (datetime.fromisoformat(s[k]).strftime("%Y%m%d-%H%M%S") for k in ("from", "to"))
    return max((p for p in rounds.glob("*.jsonl") if lo <= p.stem <= hi), key=lambda p: p.stem, default=None)


def clip_range(clip, rounds):
    """The clip's {clip, round, fps, first, last, start, end} (frame numbers; `end`: the banner's first frame), or None,
    having said why: no guessing."""
    key = clip.name[:15]
    path = round_file(clip, rounds)
    if path is None:
        print(f"{key}: no live round file in its recording: no range, skipped", flush=True)
        return None
    head, *rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    answers = [r["seq"] for r in rows if "seq" in r]
    index = [json.loads(line) for line in open(clip / "index.jsonl", encoding="utf-8")]
    files = {r["i"]: r["file"] for r in index if "file" in r}
    number = {r["seq"]: r["i"] for r in index if "file" in r and "seq" in r}
    if not answers or answers[0] not in number:
        print(f"{key}: round {path.stem}'s first answer is not in the clip: no range, skipped", flush=True)
        return None
    fps = round(head["fps"])
    first = number[answers[0]] - 1
    start = first + round(START_S * fps)
    glow = []
    for i in range(start, max(number[s] for s in answers if s in number) + PAST_ANSWERS):
        if i in files:
            hsv = cv2.cvtColor(cv2.imread(str(clip / files[i]))[180:500, 308:750], cv2.COLOR_BGR2HSV)
            yellow = (hsv[..., 0] > 22) & (hsv[..., 0] < 34) & (hsv[..., 1] > 110) & (hsv[..., 2] > 210)
            glow.append((i, int(yellow.sum())))
    base = np.median([y for _, y in glow[:30]])
    end = next((i for (i, y), (_, y2) in zip(glow, glow[1:]) if y > base + GLOW and y2 > base + GLOW), None)
    if end is None:
        print(f"{key}: no SUCCESS banner after START in round {path.stem}: no range, skipped", flush=True)
        return None
    return {"clip": clip.name, "round": path.stem, "fps": fps, "first": first, "last": min(end + AFTER_END, max(files)),
            "start": start, "end": end}
