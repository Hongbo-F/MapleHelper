"""Following her through recorded frames by her name tag (white text on a dark box under her feet),
and spotting jumps and claw swings."""

import cv2
import numpy as np

# Claw swing effect: bright white/lavender pixels next to her (both regular attack and Lucky Seven).
SWING_MIN_PIXELS = 40


def _tag_boxes(frame: np.ndarray, scale: int, white_min: int = 215, box_max: int = 110):
    """(x, y, w, h) of every cluster of white text that sits on a dark box, like a name tag.

    `scale` is how much the frame was downscaled (1 = full size). Text is dilated sideways so a
    name's letters join into one cluster. A line of text is wide and short, and behind a name the
    box is dark (median brightest channel under `box_max`); her face, gloves, clouds and other
    bright scenery fail those checks.
    """
    white = (frame.min(axis=2) > white_min).astype(np.uint8)
    kernel = np.ones((3, max(3, 13 // scale)), np.uint8)  # wide enough to bridge most letter gaps
    count, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(white, kernel))
    for x, y, bw, bh, _ in stats[1:count]:
        if not (25 // scale <= bw <= 180 // scale and 8 // scale <= bh <= 26 // scale and bw >= 2.2 * bh):
            continue
        inside = frame[y:y + bh, x:x + bw].reshape(-1, 3)
        behind = inside[inside.min(axis=1) <= white_min]
        if len(behind) and np.median(behind.max(axis=1)) < box_max:
            yield x, y, bw, bh


def find_name_tag(frame: np.ndarray, near: tuple[int, int] | None = None, scale: int = 1):
    """Centre (x, y) of the name tag closest to `near` (default: the frame centre), or None.

    The bottom tenth of the game window is the HUD (level, HP/MP, chat), never a character.
    """
    h, w = frame.shape[:2]
    nx, ny = near or (w // 2, h // 2)
    centres = [(x + bw // 2, y + bh // 2) for x, y, bw, bh in _tag_boxes(frame, scale) if y < 0.9 * h]
    return min(centres, key=lambda c: abs(c[0] - nx) + abs(c[1] - ny), default=None)


def make_template(frame: np.ndarray, centre: tuple[int, int], scale: int = 2) -> np.ndarray:
    """Grayscale name-tag template: the white text in a thin band around `centre`, cut tightly.

    The band (one text line high) keeps her glove, a rope or anything else above or below out.
    """
    gx, gy = centre
    half_h, half_w = 12 // scale, 90 // scale
    y0, x0 = max(0, gy - half_h), max(0, gx - half_w)
    band = frame[y0:gy + half_h + 1, x0:gx + half_w + 1]
    white = band.min(axis=2) > 200
    rows, cols = np.nonzero(white.any(axis=1))[0], np.nonzero(white.any(axis=0))[0]
    if len(cols) < 10 // scale:
        raise ValueError(f"no name tag text near {centre}")
    tag = frame[y0 + max(0, rows.min() - 2):y0 + rows.max() + 3, x0 + max(0, cols.min() - 2):x0 + cols.max() + 3]
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
