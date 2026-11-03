"""The web UI's overlaid recordings (ui.review): the sets, a set's viewer and meta, its frames and files, a video's byte
ranges, and nothing outside a set."""
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from gamelab.ui import review, web


@pytest.fixture
def served(tmp_path, monkeypatch):
    s = tmp_path / "golden" / "overlay-a"
    (s / "frames").mkdir(parents=True)
    for n in range(3):
        (s / "frames" / f"{n:05d}.jpg").write_bytes(b"jpg%d" % n)
    (s / "overlay.mp4").write_bytes(bytes(range(256)) * 4)
    (s / "problems.json").write_text(json.dumps([{"frames": [1, 2], "what": "a miss"}]), encoding="utf-8")
    (s / "secret.txt").write_text("no", encoding="utf-8")
    (tmp_path / "golden" / "no-frames").mkdir()
    g = tmp_path / "golden" / "group-a" / "clip-1"
    (g / "frames").mkdir(parents=True)
    (g / "frames" / "00000.jpg").write_bytes(b"g0")
    (g / "README.md").write_text("clip 1", encoding="utf-8")
    (g.parent / "loose.txt").write_text("no", encoding="utf-8")
    r = tmp_path / "golden" / "real"
    (r / "frames").mkdir(parents=True)
    noise = np.random.default_rng(0).integers(0, 256, (200, 300, 3), np.uint8)
    (r / "frames" / "00000.jpg").write_bytes(cv2.imencode(".jpg", noise, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    monkeypatch.setattr(review, "ROOT", tmp_path / "golden")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web._Handler)
    srv.feed = None  # (only the review's routes are asked)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url, headers=None):
    r = urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}))
    return r.status, r.headers, r.read()


def test_the_sets_a_viewer_its_meta_frames_and_a_videos_byte_ranges(served):
    assert b'href="/review/overlay-a/"' in get(served + "/review")[2] and b"no-frames" not in get(served + "/review")[2]
    assert b"overlay-a" in get(served + "/review/overlay-a/")[2]
    meta = json.loads(get(served + "/review/overlay-a/meta.json")[2])
    assert int(meta.pop("version"), 16) > 0
    assert meta == {"name": "overlay-a", "frames": 3, "last": 2, "problems": [{"frames": [1, 2], "what": "a miss"}],
                    "files": ["overlay.mp4"], "fps": 15.0}  # (no video in that overlay.mp4: the default)
    status, headers, body = get(served + "/review/overlay-a/frames/00001.jpg")
    assert (status, headers["Content-Type"], body) == (200, "image/jpeg", b"jpg1")
    status, headers, body = get(served + "/review/overlay-a/overlay.mp4", {"Range": "bytes=10-19"})
    assert status == 206 and headers["Content-Range"] == "bytes 10-19/1024" and body == bytes(range(10, 20))
    assert get(served + "/review/overlay-a/overlay.mp4", {"Range": "bytes=-4"})[2] == bytes(range(252, 256))
    assert len(get(served + "/review/overlay-a/overlay.mp4")[2]) == 1024


def test_a_frame_asked_smaller_comes_narrower_and_at_the_quality_asked_never_wider_and_else_as_it_is(served):
    """The user, 2026-11-02: "can the server send compressed image to reduce the network transfer?"."""
    url = served + "/review/real/frames/00000.jpg"
    whole = get(url)[2]

    def picture(q):
        status, headers, body = get(url + q)
        assert status == 200 and headers["Content-Type"] == "image/jpeg" and headers["Cache-Control"] == "max-age=86400"
        return body, cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)

    body, img = picture("?w=150&q=60")
    assert img.shape == (100, 150, 3) and len(body) < len(whole) / 3
    body, img = picture("?q=40")
    assert img.shape == (200, 300, 3) and len(body) < len(whole)
    assert picture("?w=1000")[1].shape == (200, 300, 3)  # never wider than the frame
    assert picture("?w=1&q=1")[1].shape == (43, 64, 3)  # clamped: 64 px wide at least
    assert get(url + "?w=wide")[2] == whole  # nothing sensible asked: as it is
    assert get(served + "/review/overlay-a/frames/00001.jpg?w=100")[2] == b"jpg1"  # not a picture: as it is


def test_the_viewer_asks_for_smaller_frames_unless_its_data_saver_is_off_and_preloads_from_the_start(served):
    """Preloading: the user, 2026-11-02, "can ui preload the clip in browser, not wait until play button"."""
    page = get(served + "/review/real/")[2].decode()
    assert 'id="saver"' in page and '"?w=" + w + "&q=60"' in page and "review-saver" in page
    assert "const AHEAD = 600, BEHIND = 60, PARALLEL = 8;" in page and 'id="load"' in page


def test_a_sets_frame_rate_is_its_videos_else_its_names_else_15_and_the_viewer_plays_at_it(served, tmp_path):
    """The user, 2026-11-02: "default to the recorded fps instead of 5fps"."""
    g = tmp_path / "golden"
    for name in ("clip-10fps", "cmp-8fps-0-compare", "video-12fps"):
        (g / name / "frames").mkdir(parents=True)
    w = cv2.VideoWriter(str(g / "video-12fps" / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 7, (64, 48))
    for _ in range(3):
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    rate = lambda name: json.loads(get(served + f"/review/{name}/meta.json")[2])["fps"]
    assert (rate("video-12fps"), rate("clip-10fps"), rate("cmp-8fps-0-compare"), rate("real")) == (7.0, 10.0, 8.0, 15.0)
    assert json.loads(get(served + "/review/group-a/clip-1/meta.json")[2])["fps"] == 15.0
    page = get(served + "/review/real/")[2].decode()
    assert 'value="5" selected' not in page and "rate.value = fps" in page and '" (recorded)"' in page


def test_a_sets_version_changes_when_a_frame_is_drawn_again_and_the_viewer_puts_it_in_each_frames_link(served, tmp_path):
    """The user, 2026-11-02, saw a redrawn set's old frames: the browser keeps frames for a day."""
    before = json.loads(get(served + "/review/overlay-a/meta.json")[2])["version"]
    f = tmp_path / "golden" / "overlay-a" / "frames" / "00001.jpg"
    f.write_bytes(b"jpg1 again")
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert json.loads(get(served + "/review/overlay-a/meta.json")[2])["version"] != before
    assert get(served + "/review/overlay-a/frames/00001.jpg?v=" + before)[2] == b"jpg1 again"  # (v: ignored)
    assert '"v=" + M.version' in get(served + "/review/overlay-a/")[2].decode()


def test_a_group_lists_its_sets_and_serves_each_as_a_set_one_level_down(served):
    top = get(served + "/review")[2]
    assert b'href="/review/group-a/"' in top and b"clip-1" not in top
    assert b'href="/review/group-a/clip-1/"' in get(served + "/review/group-a/")[2]
    assert b"group-a/clip-1" in get(served + "/review/group-a/clip-1/")[2]
    meta = json.loads(get(served + "/review/group-a/clip-1/meta.json")[2])
    assert (meta["name"], meta["frames"], meta["files"]) == ("clip-1", 1, ["README.md"])
    assert get(served + "/review/group-a/clip-1/frames/00000.jpg")[2] == b"g0"
    assert get(served + "/review/group-a/clip-1/README.md")[2] == b"clip 1"


@pytest.mark.parametrize("path", ["/review/../golden/overlay-a/secret.txt", "/review/overlay-a/secret.txt",
                                  "/review/overlay-a/frames/..%2fsecret.txt", "/review/no-frames/",
                                  "/review/overlay-a/frames/1.jpg", "/reviewx", "/review/group-a/loose.txt",
                                  "/review/group-a/clip-2/", "/review/group-a/group-a/clip-1/",
                                  "/review/overlay-a/clip-1/", "/review/group-a/../overlay-a/secret.txt"])
def test_nothing_outside_a_set_is_served(served, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(served + path)
    assert e.value.code == 404


def test_a_set_comes_as_a_video_to_download_its_own_overlay_or_one_made_from_its_frames_and_kept(served, tmp_path,
                                                                                                    monkeypatch):
    """The user, 2026-11-02: "Can you make the WebUI provide a button to download a review clip as video?"."""
    from gamelab.ui import video
    status, headers, body = get(served + "/review/overlay-a/video.mp4")
    assert body == bytes(range(256)) * 4 and headers["Content-Disposition"] == 'attachment; filename="overlay-a.mp4"'
    made = []
    real = video.frames_video
    monkeypatch.setattr(video, "h264", lambda *a: (False, "no Windows encoder in this test"))  # (mp4v: quick)
    monkeypatch.setattr(video, "frames_video", lambda *a: (made.append(a), real(*a))[1])
    status, headers, body = get(served + "/review/real/video.mp4")
    assert status == 200 and headers["Content-Type"] == "video/mp4" and body[4:8] == b"ftyp"
    assert headers["Content-Disposition"] == 'attachment; filename="real.mp4"' and len(made) == 1
    assert get(served + "/review/real/video.mp4")[2] == body and len(made) == 1  # kept: not made again
    kept = list((tmp_path / "golden" / ".videos").glob("real-*.mp4"))
    assert len(kept) == 1 and b".videos" not in get(served + "/review")[2]
    page = get(served + "/review/real/")[2].decode()
    assert 'id="video"' in page and 'base + "video.mp4"' in page and "Download video" in page


def test_faststart_puts_the_index_before_the_media_and_the_video_plays_the_same(tmp_path):
    from gamelab.ui import video
    src = tmp_path / "a.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    for n in range(10):
        w.write(np.full((48, 64, 3), n * 20, np.uint8))
    w.release()
    b = src.read_bytes()
    assert b.index(b"moov") > b.index(b"mdat")  # (OpenCV writes the index last)
    assert video.faststart(src, tmp_path / "b.mp4")
    c = (tmp_path / "b.mp4").read_bytes()
    assert c.index(b"moov") < c.index(b"mdat") and len(c) == len(b)
    caps = [cv2.VideoCapture(str(p)) for p in (src, tmp_path / "b.mp4")]
    a_frames, b_frames = ([cap.read()[1] for _ in range(10)] for cap in caps)
    assert all(np.array_equal(x, y) for x, y in zip(a_frames, b_frames))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' own H.264 encoder")
def test_frames_become_an_h264_video_with_windows_own_encoder(tmp_path):
    from gamelab.ui import video
    frames = []
    for n in range(8):
        f = tmp_path / f"{n:05d}.jpg"
        cv2.imwrite(str(f), np.random.default_rng(n).integers(0, 256, (101, 151, 3), np.uint8))  # (odd: padded)
        frames.append(f)
    out = tmp_path / "v.mp4"
    assert video.frames_video(frames, 15, out) == "h264"
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").lower() in (b"h264", b"avc1")
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (152, 102)
    b = out.read_bytes()
    assert b.index(b"moov") < b.index(b"mdat")
