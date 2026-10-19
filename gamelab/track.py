"""Following her through recorded frames by her name tag (white text on a dark box under her feet),
and spotting jumps and claw swings."""

import cv2
import numpy as np

# Claw swing effect: bright white/lavender pixels next to her (both regular attack and Lucky Seven).
SWING_MIN_PIXELS = 40


def find_name_tag(frame: np.ndarray, near: tuple[int, int] | None = None):
    """Centre (x, y) of the name tag closest to `near` (default: the frame centre), or None.

    Looks for clusters of white text sitting on a dark box, the way MapleStory draws name tags.
    """
    h, w = frame.shape[:2]
    nx, ny = near or (w // 2, h // 2)
    white = (frame.min(axis=2) > 215).astype(np.uint8)
    dark = frame.max(axis=2) < 80
    count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(white, np.ones((3, 9), np.uint8)))
    best = None
    for x, y, bw, bh, _ in stats[1:count]:
        if not (25 <= bw <= 180 and 8 <= bh <= 26):
            continue
        box = dark[max(0, y - 3):y + bh + 3, max(0, x - 4):x + bw + 4]
        if box.size == 0 or box.mean() < 0.3:  # text on a dark box, not on scenery
            continue
        cx, cy = x + bw // 2, y + bh // 2
        dist = abs(cx - nx) + abs(cy - ny)
        if best is None or dist < best[0]:
            best = (dist, cx, cy)
    return (best[1], best[2]) if best else None


def make_template(frame: np.ndarray, centre: tuple[int, int]) -> np.ndarray:
    """Grayscale name-tag template cut tightly around the white text near `centre`."""
    gx, gy = centre
    x0, y0 = max(0, gx - 60), max(0, gy - 15)
    patch = frame[y0:gy + 15, x0:gx + 60]
    ys, xs = np.nonzero(patch.min(axis=2) > 200)
    if len(xs) < 10:
        raise ValueError(f"no name tag text near {centre}")
    tag = patch[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 3):xs.max() + 4]
    return cv2.cvtColor(np.ascontiguousarray(tag), cv2.COLOR_RGB2GRAY)


def track(frames, times, template: np.ndarray) -> list[tuple[float, int, int, float]]:
    """(t_ms, x, y, match score) of the name tag's top-left corner in every frame."""
    rows = []
    for f, t in zip(frames, times):
        gray = cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY)
        _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED))
        rows.append((float(t), int(x), int(y), float(score)))
    return rows


def airborne(rows, lift: int = 6, min_ms: float = 150, merge_ms: float = 60) -> list[tuple[float, float | None]]:
    """(takeoff, landing) spans where the tag is at least `lift` px above where she stood at the start.

    A mis-matched frame can make one jump look like two (merged when less than `merge_ms` apart)
    or make a blip that isn't a jump (dropped when shorter than `min_ms`; a jump lasts ~530 ms).
    """
    rest_y = rows[0][2]
    raw, start = [], None
    for t, _, y, _ in rows:
        up = rest_y - y >= lift
        if up and start is None:
            start = t
        elif not up and start is not None:
            raw.append([start, t])
            start = None
    if start is not None:
        raw.append([start, None])
    merged = []
    for span in raw:
        if merged and merged[-1][1] is not None and span[0] - merged[-1][1] < merge_ms:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    return [(a, b) for a, b in merged if b is None or b - a >= min_ms]


def _bright(region: np.ndarray) -> int:
    r = region.astype(np.int16)
    return int(((r.min(axis=2) > 170) & (r[..., 2] >= r[..., 1])).sum())


def swing_side(frame, x: int, y: int, tag_width: int) -> tuple[int, int]:
    """Swing-effect pixels to her left and to her right (x, y: name tag's top-left)."""
    left = _bright(frame[max(0, y - 45):y - 2, max(0, x - 70):x])
    right = _bright(frame[max(0, y - 45):y - 2, x + tag_width:x + tag_width + 70])
    return left, right


def swings(frames, rows, times, tag_width: int) -> list[dict]:
    """Claw swings as {'t': start ms, 'side': 'L'|'R'}, one per burst of swing frames."""
    counts = [swing_side(f, x, y, tag_width) for f, (_, x, y, _) in zip(frames, rows)]
    base = max((l + r for (l, r), t in zip(counts, times) if t < 0), default=0)
    out, last_t = [], None
    for (left, right), t in zip(counts, times):
        if left + right <= base + SWING_MIN_PIXELS:
            continue
        if last_t is None or t - last_t > 150:
            out.append({"t": float(t), "left": left, "right": right})
        elif left + right > out[-1]["left"] + out[-1]["right"]:
            out[-1].update(left=left, right=right)  # keep the strongest frame of the burst
        last_t = t
    for s in out:
        s["side"] = "L" if s.pop("left") > s.pop("right") else "R"
    return out
