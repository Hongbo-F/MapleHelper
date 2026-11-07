"""Review sets of the Violetta Lie Detector games for the web UI (gamelab/ui/review.py): one set a game, in the group
<review root>/violetta/<clip>-<fps>fps, each frame the stage as recorded (left) and with the spotlights taken off (right),
the four Violettas as the rules solver has them drawn on both (rules.py, chained from the start), the slots numbered, the
round and the answers in the banner. README.md says how to read it; problems.json lists each round.
usage: build_review.py [clip name ...]   (default: every clip with pose-locked detections, dets3/)"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # (common.py, a folder up)
import common  # noqa: E402
import mot  # noqa: E402
import track2  # noqa: E402
import unveil2  # noqa: E402

ROOT = common.root() / "violetta"
DETS = "dets5"     # the detections drawn and solved on (--dets detsY --group violetta-yolo: YOLO's)
SOURCE = "sprite matches"
CLIPS = common.paths.ALARM_CLIPS
X0, Y0, X1, Y1 = 465, 285, 905, 515  # the crop (native px)
SC = 1.5
PW, PH = int((X1 - X0) * SC), int((Y1 - Y0) * SC)
BANNER = 64
COLOURS = [(230, 60, 60), (40, 170, 60), (50, 110, 240), (240, 150, 20)]
NAMES = "ABCD"
MAGENTA = (255, 0, 255)
TRAIL_S = 0.6
# A Violetta's middle on screen by its drawn size (the detections while the cones are on, score >= 0.75, 861 of them
# over the 8 games: residuals 9.4 px): lower in front, higher behind.
Y_AT, Y_PER_LN = 390.4, 57.2
try:
    FONT = ImageFont.truetype("consola.ttf", 15)
    SMALL = ImageFont.truetype("consola.ttf", 12)
except OSError:
    FONT = SMALL = ImageFont.load_default()


def P(x, y):
    return (x - X0) * SC, (y - Y0) * SC


def overlay(d, state, slots, trail, made_up_slot=None, dets=()):
    for k, s in enumerate(slots):
        x, _ = P(s, 0)
        d.line((x, PH - 18, x, PH), fill=(90, 90, 90), width=1)
        d.text((x - 3, PH - 16), str(k + 1), fill=(255, 255, 255), font=FONT)
    for det in dets:
        if "w" in det:  # (YOLO's own box; the made-up face's magenta)
            w, h = det["w"] * SC, det["h"] * SC
        else:
            w, h = 86 * det["scale"] * SC, 95 * det["scale"] * SC
        x, y = P(det["x"], det["y"])
        col = (255, 120, 255) if det.get("cls") == 1 else (170, 170, 170)
        d.rectangle((x - w / 2, y - h / 2, x + w / 2, y + h / 2), outline=col, width=1)
        if "w" in det:
            d.text((x - w / 2 + 2, y + h / 2 - 12), f"{det['score']:.2f}", fill=col, font=SMALL)
    if state is None:
        if made_up_slot is not None:
            x, y = P(slots[made_up_slot], 400)
            d.text((x - 6, y - 70), "*", fill=MAGENTA, font=FONT)
        return
    for k, pts in enumerate(trail):
        if len(pts) > 1:
            d.line([P(px, py) for px, py in pts], fill=COLOURS[k], width=2)
    for k, (tx, ty, ts, unseen) in enumerate(state["tracks"]):
        x, y = P(tx, ty)
        r = 26 * ts * SC
        mine = k == state["made_up"]
        col = COLOURS[k]
        d.ellipse((x - r, y - r, x + r, y + r), outline=MAGENTA if mine else col, width=4 if mine else 2)
        label = NAMES[k] + ("*" if mine else "") + ("?" if unseen else "")
        d.text((x - r + 3, y - r + 2), label, fill=MAGENTA if mine else col, font=FONT)


def rules_log(name, rows, res):
    """Each frame's four Violettas as the rules solver has them (rules.py: the round's hop(s), start and D), chained
    from the made-up one's start slot (no answer used): x, y (its standing y, lower in front, higher behind), size,
    and which is the made-up one; and each round's (round, predicted slot, answer)."""
    import numpy as np
    import rules
    slots = res["slots"]
    first_lit = next(r["t"] for r in rows if r["lit"])
    windows = [r["window_t"] for r in res["rounds"]]
    starts = [first_lit] + windows[:-1]
    stand_y = float(np.median([d["y"] for r in rows if r["t"] < first_lit for d in r["dets"] if d["score"] > 0.6]))
    pos = list(range(4))  # Violetta k's slot
    made_up = res["made_up_start"] - 1  # (a Violetta's index: it starts at its own slot)
    log, out = [], []
    for k, (ta, tb) in enumerate(zip(starts, windows)):
        seg = [r for r in rows if ta <= r["t"] <= tb]
        perm, kind, info = rules.solve_round(seg, slots)
        hops = ([(info["one"][2], info["one"][3], [q - 1 for q in info["one"][1]])] if kind == "one" else
                [(info["two"][3], info["two"][4] / 2, [q - 1 for q in info["two"][1]]),
                 (info["two"][3] + info["two"][4] / 2, info["two"][4] / 2, [q - 1 for q in info["two"][2]])])
        for r in seg:
            t = r["t"]
            tracks = []
            for v in range(4):
                at = pos[v]
                x, size = slots[at], 100.0
                for t0, T, hp in hops:
                    to = hp[at]
                    u = (t - t0) / T
                    if u >= 1:
                        at = to
                        x = slots[at]
                        continue
                    if u <= 0:
                        break
                    lift = min(max(u / 0.25, 0), 1)
                    settle = min(max((u - 0.75) / 0.25, 0), 1)
                    travel = min(max((u - 0.25) / 0.5, 0), 1)
                    x = slots[at] + (slots[to] - slots[at]) * travel
                    pct = rules.lane(at, to) if to != at else 100.0
                    size = 100 + (pct - 100) * lift * (1 - settle)
                    break
                sc = 1.05 * size / 100  # (the detector's units: a standing one ~1.0-1.05)
                tracks.append((x, Y_AT + Y_PER_LN * math.log(sc), sc, 0))
            log.append({"t": t, "i": r["i"], "round": k + 1, "made_up": made_up, "tracks": tracks})
        pos = [perm[p] for p in pos]
        pred = pos[made_up] + 1
        out.append((k + 1, pred, res["rounds"][k]["answered"]))
    if log:
        log[-1]["snapped"] = True
    return log, out


def build(job):
    """One game's set; `job`: (clip name, detections folder, root, what the boxes are) (the workers don't see the
    main process's settings on Windows)."""
    global DETS, ROOT, SOURCE
    name, DETS, ROOT, SOURCE = job
    res = json.load(open(f"solved/{name}.json"))
    mot.PER_ROUND, mot.DETS = True, DETS  # (set before the read: the study's copy read dets3 for a worker's first game)
    rows = [json.loads(l) for l in open(f"{mot.DETS}/{name}.jsonl")]
    slots = res["slots"]
    log, rounds = rules_log(name, rows, res)
    by_i = {e["i"]: e for e in log}
    idx = {r["i"]: r for r in (json.loads(l) for l in open(CLIPS / name / "index.jsonl")) if "file" in r}
    fps = 10 if name.startswith("20261026-0026") else 15
    dest = ROOT / f"{name.replace('-lie-detector-countdown', '')}-{fps}fps"
    (dest / "frames").mkdir(parents=True, exist_ok=False)
    windows = [r["window_t"] for r in res["rounds"]]
    result = {r[0]: (r[1], r[2]) for r in rounds}
    problems, round_first = [], {}
    clip_from = json.load(open(CLIPS / name / "summary.json"))["from"]
    hist = []
    for n, r in enumerate(rows):
        f = np.asarray(Image.open(CLIPS / name / idx[r["i"]]["file"]).convert("RGB"))
        raw = f[Y0:Y1, X0:X1]
        clean = raw.copy().astype(np.float32)
        img, lit = unveil2.stage(f.astype(np.float32))
        clean[track2.OY - Y0:track2.OY - Y0 + 194, track2.OX - X0:track2.OX - X0 + 412] = img
        panels = []
        state = by_i.get(r["i"])
        if state is not None:
            state = dict(state, dets=[d for d in r["dets"] if d["score"] >= mot.MIN_SCORE and d["scale"] >= mot.MIN_SCALE])
            if state.get("snapped"):
                hist = []  # (snapped to the slots: no line from where it was)
            hist.append((r["t"], state["tracks"]))
            hist = [h for h in hist if h[0] >= r["t"] - TRAIL_S]
        trail = [[(h[1][k][0], h[1][k][1]) for h in hist] for k in range(4)] if state else []
        for pic in (raw, np.clip(clean, 0, 255).astype(np.uint8)):
            im = Image.fromarray(pic).resize((PW, PH), Image.BILINEAR)
            shown = [dd for dd in r["dets"] if dd["score"] >= mot.MIN_SCORE and dd["scale"] >= mot.MIN_SCALE]
            overlay(ImageDraw.Draw(im), state, slots, trail,
                    res["made_up_start"] - 1 if (not log or r["t"] < log[0]["t"]) else None, shown)
            panels.append(np.asarray(im))
        canvas = Image.new("RGB", (PW * 2 + 8, PH + BANNER), (24, 24, 24))
        canvas.paste(Image.fromarray(panels[0]), (0, BANNER))
        canvas.paste(Image.fromarray(panels[1]), (PW + 8, BANNER))
        d = ImageDraw.Draw(canvas)
        rnd = state["round"] if state else 0
        if state and rnd not in round_first:
            round_first[rnd] = n
        if state is None and (not log or r["t"] < log[0]["t"]):
            phase = f"before the shuffles: the made-up one at slot {res['made_up_start']} (*)"
        elif state is None:
            phase = "after the last answer"
        elif rnd <= len(windows):
            phase = f"round {rnd} of {len(windows)}: shuffling to the answer window at {windows[rnd - 1]:.1f} s"
        else:
            phase = "after the last answer"
        line2 = ""
        for k, wt in enumerate(windows):
            if abs(r["t"] - wt) < 1.5 or (wt <= r["t"] < wt + 1.5):
                pred, ans = result.get(k + 1, (None, None))
                line2 = (f"round {k + 1}'s answer window: solver says slot {pred}, you answered {ans}"
                         + ("" if pred == ans else "   <-- differs"))
        d.text((8, 6), f"{idx[r['i']]['at']}  frame {n}  t {r['t']:.2f} s  |  {phase}", fill=(255, 255, 255), font=FONT)
        d.text((8, 26), line2 or "left: as recorded   right: spotlights taken off   circles: the solver's four "
               "(the client's shuffle rules fitted; size = depth), * = the made-up one",
               fill=(255, 220, 120) if line2 else (190, 190, 190), font=SMALL if not line2 else FONT)
        d.text((8, 46), f"grey boxes: {SOURCE} (magenta: the made-up face)  lines: the last 0.6 s  numbers: the slots",
               fill=(150, 150, 150), font=SMALL)
        canvas.save(dest / "frames" / f"{n:05d}.jpg", quality=85)
    for k, wt in enumerate(windows):
        pred, ans = result.get(k + 1, (None, None))
        first = round_first.get(k + 1, 0)
        last = next((n for n, r in enumerate(rows) if r["t"] >= wt), len(rows) - 1)
        problems.append({"frames": [first, min(last + 8, len(rows) - 1)],
                         "what": f"round {k + 1}: solver says slot {pred}, you answered {ans}"
                                 + (" (differs)" if pred != ans else "")})
    (dest / "problems.json").write_text(json.dumps(problems, indent=1), encoding="utf-8")
    table = "\n".join(f"| {k + 1} | {result.get(k + 1, (None,))[0]} | {res['rounds'][k]['answered']} |"
                      for k in range(len(windows)))
    (dest / "README.md").write_text(f"""# Violetta game {name}

The Lie Detector's "find the Violetta wearing make-up" game, recorded {clip_from} ({fps} frames a second), from the
four standing before the shuffles to the end. Made-up one at the start: slot {res['made_up_start']} (the game's
made-up sprite against the plain one).

Each frame: left the stage as recorded, right the same with the spotlights taken off (the game's own lightEffect
pictures, re-coloured as the game draws them, subtracted). On both, the four Violettas as the rules solver has them
(A red, B green, C blue, D orange; * the made-up one): each round fitted to the client's own shuffle rules (from its
decompile: the motion lasts 3.8-5.0 s, all four together, in one hop or two; each hop lift, travel, settle; the lane
by direction and distance: one slot left in front, one right behind, two or more right in front at ~1.5x, two or more
left behind at ~0.6x; one staying lunges and returns). The circles are where and how big the fitted hops put each,
chained from the made-up one's start slot with no answer used. Grey boxes: {SOURCE}, which the fit was scored
against (magenta: the made-up one's face, before the shuffles); the lines are each one's last 0.6 s. Slots 1 to 4 = the answer buttons.

| Round | Solver | You answered |
|---|---|---|
{table}

problems.json lists the rounds: step to each and check the circles follow the Violettas.
""", encoding="utf-8")
    return name, len(rows), dest


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--dets" in args:
        k = args.index("--dets")
        DETS, args = args[k + 1], args[:k] + args[k + 2:]
    if "--group" in args:
        k = args.index("--group")
        ROOT, args = common.root() / args[k + 1], args[:k] + args[k + 2:]
    if DETS.startswith("detsY"):
        SOURCE = "YOLO11n's boxes (trained on synthetic frames only), their confidence"
    names = args or sorted(p.stem for p in Path(DETS).glob("*.jsonl"))
    ROOT.mkdir(parents=True, exist_ok=True)
    with Pool(2) as pool:
        for name, n, dest in pool.imap_unordered(build, [(nm, DETS, ROOT, SOURCE) for nm in names]):
            print(name, n, "frames ->", dest, flush=True)
