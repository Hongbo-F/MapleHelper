"""The Lie Detector target tracker (perception.lie_target): the outline score with off-board points typical, the white
target's template, the board's scroll from the optical flow, and a drawn round (a spinning star white through the
countdown, then only its embossed rim, turning back near the left edge, among look-alikes printed on the board)
followed from a recorded clip's frames."""
import json

import cv2
import numpy as np
import pytest

from gamelab.perception import lie_target as lt
from gamelab.perception.lie_detector_yolo import BOARD_AT_NATIVE

FPS = 15


def star(r_out=55, r_in=30):
    a = np.arange(10) * np.pi / 5 - np.pi / 2
    r = np.where(np.arange(10) % 2 == 0, r_out, r_in)
    return np.c_[r * np.cos(a), r * np.sin(a)].astype(np.float32)


def texture(seed=0):
    g = cv2.GaussianBlur(np.random.default_rng(seed).uniform(0, 1, (500, 750)).astype(np.float32), (0, 0), 2.5)
    g = (g - g.min()) / (g.max() - g.min())
    tan = np.array([90, 140, 190], np.float32)
    return np.clip(tan * (0.75 + 0.5 * g[..., None]), 0, 255).astype(np.uint8)


def draw(board, centre, theta, white=False):
    """The star at centre turned by theta (the tracker's convention), its fill white if asked, and the embossed rim
    RIM px outside the fill: a dark line with a light one just outside it."""
    c, s = np.cos(theta), np.sin(theta)
    poly = (star() @ np.array([[c, s], [-s, c]], np.float32) + centre).astype(np.int32)
    mask = np.zeros(board.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    grown = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * lt.RIM + 1, 2 * lt.RIM + 1)))
    cs, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.polylines(board, cs, True, (215, 230, 245), 2, cv2.LINE_AA)
    inner = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * lt.RIM - 3, 2 * lt.RIM - 3)))
    cs, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.polylines(board, cs, True, (40, 60, 90), 2, cv2.LINE_AA)
    if white:
        board[mask > 0] = 255


def round_frames(n=45, white=6):
    """A drawn round: the target still and white for `white` frames, then moving left at 95 px/s, back right once its
    centre is 90 px from the left edge; spinning at lt.SPIN; three look-alikes printed at fixed angles."""
    base = texture()
    for z, a in (((420, 140), 0.4), ((560, 360), 1.3), ((300, 400), 2.2)):
        draw(base, np.array(z, np.float32), a)
    x, vx, out = 200.0, -95.0, []
    for k in range(n):
        t = k / FPS
        if k >= white:
            x += vx / FPS
            if x < 90:
                vx = abs(vx)
        board = base.copy()
        z = np.array([x, 250.0], np.float32)
        draw(board, z, -lt.SPIN * t, white=k < white)
        out.append((t, board, z))
    return out


def test_an_off_board_point_counts_as_a_typical_point_neither_hit_nor_miss():
    pts = star()
    nrm = np.zeros(len(pts), np.float32)
    resp = np.ones((lt.NB, 500, 750), np.float32)
    on, half = np.float32([[300, 250]]), np.float32([[0, 250]])
    assert lt.score(resp, pts, nrm, on, 0.0, 0.4)[0] == pytest.approx(1.0)
    _, ok = lt._lookup(resp, pts, nrm, half, 0.0)
    seen = ok.mean()
    assert 0.3 < seen < 0.7
    assert lt.score(resp, pts, nrm, half, 0.0, 0.4)[0] == pytest.approx(seen + (1 - seen) * 0.4)


def test_the_boards_scroll_is_its_optical_flows_median_over_the_boards_own_pixels():
    board = texture()
    moved = cv2.warpAffine(board, np.float32([[1, 0, 3.4], [0, 1, -2.2]]), (750, 500), borderMode=cv2.BORDER_REFLECT)
    moved[200:300, 300:420] = (60, 255, 60)  # a green patch (the cursor's colour): foreign, left out
    grey = [cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) for b in (board, moved)]
    flow = cv2.DISOpticalFlow_create(2).calc(grey[0], grey[1], None)
    clean = (lt.foreign_mask(board) == 0) & (lt.foreign_mask(moved) == 0)
    assert not clean[250, 360]
    assert lt.flow_shift(flow, clean) == pytest.approx([3.4, -2.2], abs=0.3)


def test_the_white_target_gives_its_template_the_rim_outside_its_fill():
    t, board, z = round_frames(n=1)[0]
    blob = lt.target_blob(board)
    assert blob is not None and np.hypot(*(blob["c"] - z)) < 1.5
    fill, _ = lt.template_from(blob["mask"])
    rim, nrm = lt.template_from(blob["mask"], rim=lt.RIM)
    assert np.hypot(*rim.T).max() == pytest.approx(np.hypot(*fill.T).max() + lt.RIM, abs=2.5)
    assert len(nrm) == len(rim) and nrm.min() >= 0 and nrm.max() < np.pi


def test_a_drawn_round_is_followed_through_the_fade_and_a_turn_near_the_edge_from_a_clips_frames(tmp_path):
    frames = round_frames()
    clip = tmp_path / "clip"
    clip.mkdir()
    x, y, w, h = BOARD_AT_NATIVE
    with open(clip / "index.jsonl", "w", encoding="utf-8") as f:
        for i, (t, board, _) in enumerate(frames):
            screen = np.zeros((768, 1366, 3), np.uint8)
            screen[y:y + h, x:x + w] = board
            cv2.imwrite(str(clip / f"{i:05d}.png"), screen)
            f.write(json.dumps({"i": i, "file": f"{i:05d}.png", "t": t}) + "\n")
    answers = lt.track_clip(clip, first=0, fps=FPS, start_s=6 / FPS)
    # the first sight of the white target is the template frame (and a bigger blob before START another); the filter
    # answers every frame after it
    assert answers[0]["white"] and not any(answers[i]["white"] for i in range(6, len(frames)))
    ref = {"centre": {str(i): frames[i][2].tolist() for i in range(6, len(frames))}}
    r = lt.against_reference(answers, ref)
    assert r["frames"] == len(frames) - 6 and r["mean"] < 4 and r["worst"] < 12, r
    turn = [i for i in range(1, len(frames)) if frames[i][2][0] > frames[i - 1][2][0]][0]
    assert all(np.hypot(answers[i]["x"] - frames[i][2][0], answers[i]["y"] - frames[i][2][1]) < 12
               for i in range(turn - 2, turn + 4))
