"""The re-finds' matches on the GPU (gamelab/gpumatch.c, gpumatch.hlsl: Direct3D 11 behind native.c's rf_*): each
place's integer sums on the GPU, the answers the CPU's to the bit. The one call's (rf_run: the near window, the whole
one, the memo, the stages) against the CPU's context's, every field of every answer; each picture's best over a
window (gm_bests) against mm_best's, among them those the GPU hands back to the CPU (a flat window: every place a
candidate; a flat picture; one past 128 px); ties; the approximate scores within the band's half. A late answer, a
device lost (as a driver reset: between calls, or while a call is on it), an error sending a call and a device that
can't be made (rf_gpu_fault): the CPU's answers at once, that very call. Skipped without a Direct3D 11 hardware device
(never WARP: that's the CPU again)."""

import ctypes
import time

import numpy as np
import pytest

from gamelab import native
from gamelab.perception import match

from test_refind import BOUNDS, asks_of, detector, scene  # noqa: F401 (detector: the fixture)

pytestmark = pytest.mark.skipif(native.lib() is None, reason=f"gamelab's C pieces aren't built: {native.why}")

OUT_FIELDS = 56  # an RfOut's bytes but its time (`us`)


def gpu_refinds(pictures: list, threads: int = 4, fault: int = 0) -> native.Refinds:
    """A Refinds with the GPU backend (`fault` 3: its device can't be made), once it's made or not."""
    f = native.lib()
    was = native.gpu()
    native.gpu(True)
    f.rf_gpu_fault(None, fault)
    try:
        rf = native.Refinds(pictures, threads)
    finally:
        native.gpu(was)
        f.rf_gpu_fault(None, 0)
    t0 = time.monotonic()
    while rf.gpu_state()[0] == "starting" and time.monotonic() - t0 < 20:
        time.sleep(0.01)
    return rf


def need(rf: native.Refinds) -> None:
    state, stats = rf.gpu_state()
    if state != "on":
        pytest.skip(f"no Direct3D 11 hardware device here (the GPU backend {state}, why {stats.get('why')})")


def ready_of(detector) -> list:  # noqa: F811
    return [match.prepared(t.rgb, t.mask3) for t in detector._rf_table]


@pytest.fixture(scope="module", autouse=True)
def patient():
    """Every call the GPU's here, however long an idle GPU takes (gm_waits: no limit, no rest after a late one): the
    answers are the same either way, but the tests that the GPU answered need it to."""
    was = (ctypes.c_longlong * 2)()
    native.lib().gm_waits(2_000_000, 0, was)
    yield
    native.lib().gm_waits(was[0], was[1], None)


@pytest.fixture(scope="module")
def on_gpu(detector):  # noqa: F811
    rf = gpu_refinds(ready_of(detector))
    need(rf)
    yield rf
    rf.close()


def test_all_in_one_call_on_the_gpu_the_same_answers_as_each_on_its_own(detector, on_gpu, monkeypatch):  # noqa: F811
    monkeypatch.setitem(detector._rf, 4, on_gpu)
    before = on_gpu.gpu_state()[1]
    for name in ("pig_move0", "pig_move1"):
        frame_, box = scene(name)
        asks = asks_of(frame_, box)
        memos = [{} for _ in asks]
        alone = [detector.refind(frame_, b, kind, template, facing, BOUNDS, fallback, near, memo)
                 for (b, kind, template, facing, fallback, near, _), memo in zip(asks, memos)]
        c_memos = [{} for _ in asks]
        together = detector.refind_all(frame_, [(*a[:6], memo, a[6]) for a, memo in zip(asks, c_memos)], BOUNDS, 4)
        assert together == alone and any(m is not None for m in together) and any(m is None for m in together)
        assert [m.get("far") for m in c_memos] == [m.get("far") for m in memos]
    after = on_gpu.gpu_state()[1]
    assert after["answered"] > before["answered"] and after["pairs"] > before["pairs"]  # (the GPU's answers)


def jobs_of(frame_, box, rng, n: int, lists_of) -> ctypes.Array:
    """n jobs around `box`: windows near it, cut by the frame, a near window inside the full one or not, or none."""
    h, w = frame_.shape[:2]
    jobs = (native.RfJob * n)()
    for i in range(n):
        dx, dy = (int(v) for v in rng.integers(-30, 31, 2))
        pad, near = int(rng.integers(8, 24)), int(rng.integers(0, 14))
        x0, y0, x1, y1 = box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy
        full = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
        jobs[i].id = 100 + i
        jobs[i].key, jobs[i].first_at, jobs[i].first_n, jobs[i].next_at, jobs[i].next_n = lists_of(i)
        jobs[i].wfull[:] = full
        if i % 7 == 0:  # (now and then a near window past the full one: a window of its own)
            near = pad + 4
        if near and i % 5:
            jobs[i].wnear[:] = (max(0, x0 - near), max(0, y0 - near), min(w, x1 + near), min(h, y1 + near))
    return jobs


def test_rf_run_on_the_gpu_gives_every_field_of_the_cpus_answers(detector, on_gpu):  # noqa: F811
    # (a call after another with the same ids: the memos' answers too; `keep`; the stages' stops; THRESHOLD)
    cpu = native.Refinds(ready_of(detector), 4)
    lists = [detector._lists_of("pig", t, -1) for t in ("pig_move0", "pig_move1", "")]
    rng = np.random.default_rng(5)
    flat = detector._rf_lists
    looked, found = 0, 0
    for step in range(8):
        frame_, box = scene(("pig_move0", "pig_move1")[step // 3 % 2], 600 + 3 * (step // 2), 300 - step // 2)
        if step == 5:
            frame_ = frame_.copy()
            frame_[box[1] + 5, box[0] + 5] ^= 7
        if step % 2 == 0:
            jobs = jobs_of(frame_, box, rng, 12, lambda i: lists[i % 3])
        keep = step == 6
        a = cpu.run(frame_, jobs, flat, 0.6, 0.9, keep, 0.95)
        b = on_gpu.run(frame_, jobs, flat, 0.6, 0.9, keep, 0.95)
        assert a is not None and b is not None
        ab, bb = bytes(a), bytes(b)
        for i in range(len(jobs)):
            assert ab[64 * i:64 * i + OUT_FIELDS] == bb[64 * i:64 * i + OUT_FIELDS], (step, i)
        looked |= sum(o.looked for o in a)
        found += sum(o.found == 1 for o in a) + 10 * sum(o.found == 2 for o in a)
    assert looked & 1 and looked & 2 and looked & 4 and looked & 8  # (near and full matched, and from the memo)
    assert found % 10 and found // 10  # (found near, and in the whole window)
    cpu.close()


def random_pictures(rng, sizes) -> list:
    out = []
    for th, tw, cn in sizes:
        t = rng.integers(0, 256, (th, tw, cn) if cn == 3 else (th, tw), dtype=np.uint8)
        m = (rng.random((th, tw)) < 0.8).astype(np.uint8) * 255
        out.append((t, m, native.prepare_masked(t, m)))
    return out


def bests(rf, wins, pairs, cn):
    n, m = len(pairs), len(wins)
    sc, xs, ys, how = (ctypes.c_float * n)(), (ctypes.c_int * n)(), (ctypes.c_int * n)(), (ctypes.c_int * n)()
    ints = lambda v: (ctypes.c_int * len(v))(*v)  # noqa: E731
    r = rf._lib.gm_bests(rf.ctx, m, (ctypes.c_void_p * m)(*[w.ctypes.data for w in wins]),
                         ints([w.shape[0] for w in wins]), ints([w.shape[1] for w in wins]),
                         (ctypes.c_longlong * m)(*[w.strides[0] for w in wins]), cn, n,
                         (ctypes.c_int * n)(*[w for w, _ in pairs]), (ctypes.c_int * n)(*[p for _, p in pairs]),
                         sc, xs, ys, how)
    assert r == 0
    return [(sc[k], xs[k], ys[k]) for k in range(n)], list(how)


@pytest.mark.parametrize("cn", [3, 1])
def test_each_pictures_best_on_the_gpu_is_mm_bests_to_the_bit(cn):
    rng = np.random.default_rng(cn)
    sizes = [(int(rng.integers(12, 70)), int(rng.integers(12, 70)), cn) for _ in range(6)] + [(130, 40, cn)]
    pics = random_pictures(rng, sizes)
    flat_t = np.full((20, 24, 3) if cn == 3 else (20, 24), 77, np.uint8)  # (A 0: every place -1, the CPU's)
    pics.append((flat_t, np.full((20, 24), 255, np.uint8), native.prepare_masked(flat_t, np.full((20, 24), 255,
                                                                                                    np.uint8))))
    rf = gpu_refinds([p for _, _, p in pics])
    need(rf)
    wins = []
    for k in range(5):
        h, w = int(rng.integers(80, 150)), int(rng.integers(80, 150))
        img = rng.integers(0, 256, (h, w, cn) if cn == 3 else (h, w), dtype=np.uint8)
        t, m, _ = pics[k % 4]
        y, x = int(rng.integers(0, h - t.shape[0])), int(rng.integers(0, w - t.shape[1]))
        img[y:y + t.shape[0], x:x + t.shape[1]][m > 0] = t[m > 0]
        if k == 3:  # (the picture twice: equal best scores, the first row by row)
            y2, x2 = y + (h - t.shape[0] - y) // 2 + 1, x
            img[y2:y2 + t.shape[0], x2:x2 + t.shape[1]][m > 0] = t[m > 0]
        wins.append(np.ascontiguousarray(img))
    wins.append(np.full((90, 90, 3) if cn == 3 else (90, 90), 40, np.uint8))  # (a flat window: all -1, all in band)
    pairs = [(wi, j) for wi, w in enumerate(wins) for j, (t, _, _) in enumerate(pics)
             if t.shape[0] <= w.shape[0] and t.shape[1] <= w.shape[1]]
    got, how = bests(rf, wins, pairs, cn)
    for (wi, j), g in zip(pairs, got):
        want = native.best_matches(wins[wi], [pics[j][2]])[0]
        assert np.float32(g[0]).tobytes() == np.float32(want[0]).tobytes() and g[1:] == want[1:], (wi, j)
    by = {p: h for p, h in zip(pairs, how)}
    assert all(by[(5, j)] == 0 for j in range(len(pics)) if (5, j) in by)  # (the flat window: the CPU's)
    assert all(by[(wi, len(pics) - 1)] == 0 for wi in range(5))  # (the flat picture)
    assert all(by[(wi, 6)] == 0 for wi in range(5) if (wi, 6) in by)  # (past 128 px)
    assert sum(how) >= len([p for p in pairs if p[0] < 5 and p[1] < 6])  # (the rest the GPU's)
    rf.close()


def test_the_gpus_approximate_scores_are_well_within_the_band(detector, on_gpu):  # noqa: F811
    # (gpumatch.hlsl: s~ within GM_EPS = 2^-14 of the float32 score, by its arithmetic's error, about 2^-21 here)
    frame_, box = scene("pig_move1")
    win = np.ascontiguousarray(frame_[box[1] - 20:box[3] + 20, box[0] - 20:box[2] + 20])
    for k, ready in enumerate(ready_of(detector)):
        exact = native.masked_match(win, ready)
        out = np.empty(exact.size + 4, np.float32)
        assert on_gpu._lib.gm_approx(on_gpu.ctx, win.ctypes.data, win.shape[0], win.shape[1], win.strides[0], 3, k,
                                     out.ctypes.data) == 0
        assert np.abs(out[:exact.size].reshape(exact.shape).astype(np.float64) - exact).max() < 2e-6


@pytest.mark.parametrize("fault, state, why, sent", [
    (1, "on", 0, 1),   # late
    (2, "off", 7, 0),  # lost between calls (as a driver reset): seen at the next call, before it's sent
    (4, "off", 7, 1),  # lost while the call is on it: that very call the CPU's
    (5, "off", 8, 1),  # an error sending it
])
def test_a_late_answer_or_a_lost_device_leaves_the_call_to_the_cpu_at_once(detector, fault, state, why,  # noqa: F811
                                                                             sent):
    rf = gpu_refinds(ready_of(detector))
    need(rf)
    cpu = native.Refinds(ready_of(detector), 4)
    lists = [detector._lists_of("pig", t, -1) for t in ("pig_move0", "pig_move1")]
    frame_, box = scene()
    jobs = jobs_of(frame_, box, np.random.default_rng(9), 10, lambda i: lists[i % 2])
    rf._lib.rf_gpu_fault(rf.ctx, fault)
    t0 = time.perf_counter()
    b = rf.run(frame_, jobs, detector._rf_lists, 0.6, 0.9, False, 0.95)
    took = time.perf_counter() - t0
    a = cpu.run(frame_, jobs, detector._rf_lists, 0.6, 0.9, False, 0.95)
    assert bytes(a)[:OUT_FIELDS] == bytes(b)[:OUT_FIELDS] and all(
        bytes(a)[64 * i:64 * i + OUT_FIELDS] == bytes(b)[64 * i:64 * i + OUT_FIELDS] for i in range(len(jobs)))
    st, stats = rf.gpu_state()
    assert st == state and stats["why"] == why and stats["calls"] == sent and stats["answered"] == 0
    assert stats["late"] == (fault == 1)
    assert took < 0.5  # (no waiting on it)
    rf._lib.rf_gpu_fault(rf.ctx, 0)
    c = rf.run(frame_, jobs, detector._rf_lists, 0.6, 0.9, False, 0.95)  # (late: the next on the GPU again)
    a = cpu.run(frame_, jobs, detector._rf_lists, 0.6, 0.9, False, 0.95)  # (the second time: the memos' answers)
    assert all(bytes(a)[64 * i:64 * i + OUT_FIELDS] == bytes(c)[64 * i:64 * i + OUT_FIELDS] for i in range(len(jobs)))
    rf.close()
    cpu.close()


def test_a_device_that_cant_be_made_leaves_every_call_to_the_cpu(detector):  # noqa: F811
    rf = gpu_refinds(ready_of(detector), fault=3)
    assert rf.gpu_state()[0] == "off" and rf.gpu_state()[1]["why"] == 1
    frame_, box = scene()
    lists = detector._lists_of("pig", "pig_move0", -1)
    jobs = jobs_of(frame_, box, np.random.default_rng(2), 4, lambda i: lists)
    cpu = native.Refinds(ready_of(detector), 4)
    a = cpu.run(frame_, jobs, detector._rf_lists, 0.6, 0.9, False, 0.95)
    b = rf.run(frame_, jobs, detector._rf_lists, 0.6, 0.9, False, 0.95)
    assert all(bytes(a)[64 * i:64 * i + OUT_FIELDS] == bytes(b)[64 * i:64 * i + OUT_FIELDS] for i in range(len(jobs)))
    rf.close()
    cpu.close()
