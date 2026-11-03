"""A review set's frames as a video to download (the user, 2026-11-02: "Can you make the WebUI provide a button to
download a review clip as video?"): H.264 in MP4 with its index first, so a phone plays it as it comes (an iPhone takes no
MPEG-4 Part 2, which is all OpenCV here writes: 'mp4v').

OpenCV here has no H.264 encoder (its FFmpeg asks for OpenH264's DLL, not on this machine), so the frames go into a
temporary MJPEG AVI, Windows' own Media Foundation H.264 encoder re-encodes it (h264.ps1 beside this: WinRT
MediaTranscoder, run by PowerShell; nothing downloaded or installed), and `faststart` moves the index (moov) before the
media data, as ffmpeg's -movflags +faststart does. Where Windows' step fails, the 'mp4v' file OpenCV writes is the
video, and the result says so.
"""

import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np

H264_PS1 = Path(__file__).with_name("h264.ps1")
BITS_PER_PIXEL = 8.0  # the H.264 bitrate: this many bits a pixel a second (750 x 554 at 15 fps: 3.3 Mbit/s)
TIMEOUT_S = 600
CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta", b"mvex"}


def frames_video(frames: list[Path], fps: float, dst: Path) -> str:
    """The frames (image files, in order) as an MP4 at `fps`, written to `dst`: "h264", or "mp4v" where Windows' encoder
    failed. Frames of another size than the first's are fitted to it; an odd width or height gets a black column or row
    (H.264 takes even sizes only)."""
    if not frames:
        raise ValueError("no frames")
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"{frames[0]} can't be read")
    h, w = first.shape[:2]
    H, W = h + h % 2, w + w % 2
    dst = Path(dst)
    avi, raw = dst.with_suffix(".mjpg.avi"), dst.with_suffix(".h264.mp4")
    try:
        writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
        for f in frames:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            img = np.zeros((h, w, 3), np.uint8) if img is None else img
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            if (H, W) != (h, w):
                img = cv2.copyMakeBorder(img, 0, H - h, 0, W - w, cv2.BORDER_CONSTANT)
            writer.write(img)
        writer.release()
        ok, _ = h264(avi, raw, W, H, fps, int(BITS_PER_PIXEL * W * H))
        if ok:
            faststart(raw, dst)
            return "h264"
        mp4v(avi, dst, fps)
        return "mp4v"
    finally:
        for p in (avi, raw):
            p.unlink(missing_ok=True)


def h264(src: Path, dst: Path, width: int, height: int, fps: float, bitrate: int) -> tuple[bool, str]:
    """Windows' H.264 encoder on `src` (any video Media Foundation reads) to `dst`: whether it worked, and what it said."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(H264_PS1),
                            "-In", str(src), "-Out", str(dst), "-Width", str(width), "-Height", str(height),
                            "-Fps", str(int(round(fps))), "-Bitrate", str(bitrate)],
                           capture_output=True, text=True, timeout=TIMEOUT_S,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"{type(e).__name__}: {e}"
    ok = r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    return ok, (r.stdout + r.stderr).strip()[-600:]


def mp4v(src: Path, dst: Path, fps: float) -> None:
    """The video re-written as OpenCV's own MPEG-4 Part 2 MP4 (the fallback: no H.264 to be had)."""
    cap = cv2.VideoCapture(str(src))
    writer = None
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if writer is None:
                writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
            writer.write(img)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def _boxes(b: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        size, kind = struct.unpack(">I4s", b[i:i + 8])
        head = 8
        if size == 1:
            size, head = struct.unpack(">Q", b[i + 8:i + 16])[0], 16
        elif size == 0:
            size = end - i
        if size < head:
            raise ValueError(f"a bad box {kind!r} at {i}")
        yield kind, i, size, head
        i += size


def _shift(moov: bytearray, delta: int) -> None:
    """Every chunk offset (stco, co64) in the index moved by delta."""
    def walk(start, end):
        for kind, i, size, head in _boxes(moov, start, end):
            if kind in CONTAINERS:
                walk(i + head, i + size)
            elif kind in (b"stco", b"co64"):
                n = struct.unpack(">I", moov[i + head + 4:i + head + 8])[0]
                width, fmt = (4, ">I") if kind == b"stco" else (8, ">Q")
                for k in range(n):
                    o = i + head + 8 + width * k
                    v = struct.unpack(fmt, moov[o:o + width])[0] + delta
                    if kind == b"stco" and v >= 2 ** 32:
                        raise OverflowError("a chunk offset past 4 GB in stco")
                    moov[o:o + width] = struct.pack(fmt, v)
    walk(0, len(moov))


def faststart(src: Path, dst: Path) -> bool:
    """`src` (an MP4) written to `dst` with its index (moov) right after ftyp: whether it had to be moved."""
    b = Path(src).read_bytes()
    top = list(_boxes(b, 0, len(b)))
    kinds = [k for k, *_ in top]
    if b"moov" not in kinds or b"mdat" not in kinds:
        raise ValueError(f"not an MP4 with moov and mdat: {kinds}")
    if kinds.index(b"moov") < kinds.index(b"mdat"):
        Path(dst).write_bytes(b)
        return False
    _, mi, ms, _ = top[kinds.index(b"moov")]
    moov = bytearray(b[mi:mi + ms])
    _shift(moov, ms)
    parts = []
    for kind, i, size, _ in top:
        if kind == b"moov":
            continue
        parts.append(b[i:i + size])
        if kind == b"ftyp":
            parts.append(bytes(moov))
    if kinds[0] != b"ftyp":
        parts.insert(0, bytes(moov))
    tmp = Path(str(dst) + ".tmp")
    with open(tmp, "wb") as f:
        for part in parts:
            f.write(part)
    tmp.replace(dst)
    return True
