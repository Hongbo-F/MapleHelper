"""The YOLO model's Violettas on every frame of the recorded games (the same frames and crop as the synthetic set),
written as the detectors' lines (dets/<clip>.jsonl's t, i, lit; dets with x, y in native px, a size in the detector's
units from the box's width (a sprite's ~87 px at 1.0: width barely cut at the stage's bottom), its confidence as the
score, its class) for rules.py. Timing on the GPU, batched.
usage: yolo_detect.py <weights> <out dir> [clip names...]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # (common.py, a folder up)
from common import paths  # noqa: E402

CLIPS = paths.ALARM_CLIPS
CROP = (465, 285, 905, 515)  # x0, y0, x1, y1 (native): the window around the stage, as the synthetic training set's
SPRITE_W = 75.7  # (a box is the visible pixels: 0.87x the sprite matcher's size at 87, on 6,241 matched pairs)
BATCH = 32


def main(weights, out_dir, names):
    from ultralytics import YOLO
    model = YOLO(weights)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = CROP
    total_frames, total_s = 0, 0.0
    for name in names:
        rows = [json.loads(l) for l in open(f"dets5/{name}.jsonl")]
        idx = {r["i"]: r for r in (json.loads(l) for l in open(CLIPS / name / "index.jsonl")) if "file" in r}
        with open(out_dir / f"{name}.jsonl", "w") as fh:
            for b in range(0, len(rows), BATCH):
                chunk = rows[b:b + BATCH]
                imgs = [np.asarray(Image.open(CLIPS / name / idx[r["i"]]["file"]).convert("RGB"))[y0:y1, x0:x1][..., ::-1]
                        for r in chunk]  # (BGR, as Ultralytics takes arrays)
                t = time.perf_counter()
                res = model.predict(imgs, imgsz=448, conf=0.5, device=0, verbose=False)
                total_s += time.perf_counter() - t
                total_frames += len(chunk)
                for r, rr in zip(chunk, res):
                    dets = []
                    for (bx0, by0, bx1, by1), conf, cls in zip(rr.boxes.xyxy.cpu().numpy(), rr.boxes.conf.cpu().numpy(),
                                                               rr.boxes.cls.cpu().numpy()):
                        w = bx1 - bx0
                        dets.append({"score": round(float(conf), 3), "x": round(float((bx0 + bx1) / 2 + x0), 1),
                                     "y": round(float((by0 + by1) / 2 + y0), 1),
                                     "scale": round(float(max(0.3, w / SPRITE_W)), 3), "pose": -1,
                                     "cls": int(cls), "w": round(float(w), 1), "h": round(float(by1 - by0), 1)})
                    fh.write(json.dumps({"t": r["t"], "i": r["i"], "lit": r["lit"], "dets": dets}) + "\n")
    print(f"{total_frames} frames, model time {total_s:.1f} s: {1000 * total_s / total_frames:.1f} ms a frame "
          f"({total_frames / total_s:.0f} frames a second)")


if __name__ == "__main__":
    weights, out_dir = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or sorted(p.stem for p in Path("dets5").glob("*.jsonl"))
    main(weights, out_dir, names)
