// The masked match's integer sums on the GPU (Direct3D 11, cs_5_0), for gpumatch.c: native.c's contract above rf_*
// says what a GPU can give to the bit (D3D11 has no int64 and no double sqrt). Per place of a window and a picture:
//   STI  the picture's dot product with the window under it (every channel; the picture is 0 outside its mask),
//   S1_c the window's sum under the mask (each channel), S2 its sum of squares (the channels summed),
// each under 2^32 for a picture up to 128 x 128 (gpumatch.c takes no bigger), exact: STI in float32 partial sums of at
// most 64 pixels of a mask run (each product an integer at most 255^2, a partial sum at most 64 x 3 x 255^2 =
// 12,484,800 < 2^24, so every float32 step is an exact integer), each added to a uint; S1_c and S2 from each window
// row's prefix sums (uint, exact), a mask run at a time. From them, in 64-bit integers made of two uints (exact), N and
// B as native.c's score_at has them, and an approximate score s~ = N / sqrt(A B) in float32 (-1 where B is 0, exactly
// as score_at). The CPU makes the answer exact (gpumatch.c, gm_answer): s~ is within EPS (gpumatch.c's GM_EPS, 2^-14)
// of the float32 score at every place (N and B exact, then about five float roundings: at most 1.8e-7 at the 42.7 M
// places of the crowd's call set, D:/MapleHelper-scratch/gpu/tools/approx_check.py), so the places that can be the best
// are those within `band` = 2 EPS of the highest s~; their sums come back (at most `kcand`, GM_CAND, a region), the CPU
// scores them as score_at and takes the first highest row by row. A frame sent ahead (`ahead`: gpumatch.c's rf_ahead,
// the windows to be asked not known yet) has no choose pass: match writes only each place's key (`keys`), which alone
// comes back, and the CPU works out the candidates' sums itself from the frame (gpumatch.c's gm_ahead_best).
//
// Three passes a frame, one dispatch each, in one submission:
//   prep:   each window's rows as packed pixels (its bytes as the CPU copied them, a word at a time) and uint4 prefix
//           sums, laid out in RUN planes (`planes`); the tasks' records cleared.
//   match:  RUN places side by side a thread, GROUP threads a group (`groups`: a task and its first thread slot): their
//           sums, s~, and the highest s~ of the task's places and of its near sub-rectangle (InterlockedMax on an
//           orderable key). Reads prep's buffers through views of their own (read-only, so cached on the way).
//   choose: the same groups: each place within BAND of its region's highest goes to that region's candidates.
// At the RTX 3080's idle clocks (P8: 210 MHz, the frame loop's pace) the crowd's 599 recorded calls took 5.15 ms of the
// GPU's a call as first built (match 4.76), 2.37 with the planes, the packed pixels and the picture's four at a load
// (match 2.08, prep 0.21; clocked up 0.34 against 0.74): D:/MapleHelper-scratch/gsmooth/hide/kern.

#define GROUP 64  // (gpumatch.c's GM_GROUP)
#define CHUNK 64  // a float32 partial sum's pixels at most
#define RUN 4     // places side by side a thread (GM_RUN: 8 and 1 were slower, D:/MapleHelper-scratch/gpu/kern)

cbuffer Frame : register(b0)
{
    uint ntask, nwin, cn, kcand;
    float band;
    uint ahead;  // (1: a frame sent ahead: match writes the keys alone)
    uint2 unused;
};

struct Pic { uint th, tw, nruns, run_off, px_off, n, st0, st1, st2; float af; uint u0, u1; };
struct Win { uint ww, wh, raw_off, px_off, pre_off, rowa, u1, u2; };  // (rowa: a row's bytes in raw, a word's multiple)
struct Task { uint win, pic, rw, rh, nx0, ny0, nx1, ny1, place_off, u0, u1, u2; };

StructuredBuffer<Pic> pics : register(t0);
StructuredBuffer<uint> runs : register(t1);      // row | from << 8 | to << 16
ByteAddressBuffer tpx : register(t2);            // the pictures' pixels, rows tw apart: R | G << 8 | B << 16
ByteAddressBuffer raw : register(t3);            // the windows' bytes, rows ww * cn apart
StructuredBuffer<Win> wins : register(t4);
StructuredBuffer<Task> tasks : register(t5);
StructuredBuffer<uint2> groups : register(t6);   // (task, first place)
StructuredBuffer<uint> pxr : register(t7);       // pxf and pre as match reads them (prep's, read-only: cached)
StructuredBuffer<uint4> prer : register(t8);

RWStructuredBuffer<uint> pxf : register(u0);     // the windows' pixels, R | G << 8 | B << 16
RWStructuredBuffer<uint4> pre : register(u1);    // each window row's prefix sums: S1_0, S1_1, S1_2, S2; ww + 1 a row
RWStructuredBuffer<uint4> sums : register(u2);   // a place's STI, S1_0, S1_1, S1_2
RWStructuredBuffer<uint2> sc : register(u3);     // a place's S2, the key of its s~
RWByteAddressBuffer rec : register(u4);          // a task's: highest key (all, near), candidates (all, near)
RWStructuredBuffer<uint> cand : register(u5);    // a task's candidates: [2][kcand] of (place, STI, S1_0..2, S2)
RWStructuredBuffer<uint> keys : register(u6);    // a frame sent ahead's: each place's key alone (4 bytes, not 24)

uint raw_byte(uint at) { return (raw.Load(at & ~3u) >> ((at & 3u) * 8u)) & 255u; }

// A window row's pixels (pxf) and prefix sums (pre) in RUN (4) planes, column c at plane c % 4, c / 4 along it: a
// warp's threads' places are 4 apart, so each column's loads are side by side, in a quarter of the cache lines a row
// had them in. Q a plane's length: the columns 0..ww+2 (the places past rw read past it, theirs never kept). A pixel
// packed R | G << 8 | B << 16 (a quarter of a float4's bytes), as the pictures' are.
uint planes(uint ww) { return (ww + 6) >> 2; }
float3 unpacked(uint p) { return float3(p & 255u, (p >> 8) & 255u, p >> 16); }  // (a picture pixel's channels)
uint in_planes(uint c, uint Q) { return (c & 3u) * Q + (c >> 2); }

[numthreads(128, 1, 1)]
void prep(uint3 g : SV_GroupID, uint3 l : SV_GroupThreadID)
{
    if (g.x >= nwin) {  // the tasks' records
        uint t = (g.x - nwin) * 128 + l.x;
        if (t < ntask)
            rec.Store4(t * 16, uint4(0, 0, 0, 0));
        return;
    }
    Win w = wins[g.x];
    uint Q = planes(w.ww);
    for (uint y = l.x; y < w.wh; y += 128) {
        uint at = w.raw_off + y * w.rowa, o = w.px_off + y * 4 * Q, p = w.pre_off + y * 4 * Q;
        uint4 s = uint4(0, 0, 0, 0);
        pre[p] = s;
        uint x = 0;
        if (cn == 3) {  // (4 pixels from 3 words)
            for (; x + 4 <= w.ww; x += 4, at += 12) {
                uint3 d = raw.Load3(at);
                uint px[4] = {d.x & 0xffffffu, d.x >> 24 | (d.y & 0xffffu) << 8, d.y >> 16 | (d.z & 0xffu) << 16,
                              d.z >> 8};
                [unroll] for (uint k = 0; k < 4; k++) {
                    uint c0 = px[k] & 255u, c1 = (px[k] >> 8) & 255u, c2 = px[k] >> 16;
                    pxf[o + in_planes(x + k, Q)] = px[k];
                    s += uint4(c0, c1, c2, c0 * c0 + c1 * c1 + c2 * c2);
                    pre[p + in_planes(x + k + 1, Q)] = s;
                }
            }
        }
        for (; x < w.ww; x++, at += cn) {
            uint c0 = raw_byte(at), c1 = cn > 1 ? raw_byte(at + 1) : 0, c2 = cn > 2 ? raw_byte(at + 2) : 0;
            pxf[o + in_planes(x, Q)] = c0 | c1 << 8 | c2 << 16;
            s += uint4(c0, c1, c2, c0 * c0 + c1 * c1 + c2 * c2);
            pre[p + in_planes(x + 1, Q)] = s;
        }
        for (uint x = w.ww; x < 4 * Q; x++) {  // (the columns past it the last places' slide reads: theirs aren't kept)
            pxf[o + in_planes(x, Q)] = 0;
            if (x > w.ww)
                pre[p + in_planes(x, Q)] = s;
        }
    }
}

// 64-bit integers as (low, high) uints.
uint2 mul64(uint a, uint b)
{
    uint al = a & 0xffffu, ah = a >> 16, bl = b & 0xffffu, bh = b >> 16;
    uint ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    uint mid = lh + hl, cm = mid < lh ? 1u : 0u;
    uint lo = ll + (mid << 16), cl = lo < ll ? 1u : 0u;
    return uint2(lo, hh + (mid >> 16) + (cm << 16) + cl);
}
uint2 add64(uint2 a, uint2 b) { uint lo = a.x + b.x; return uint2(lo, a.y + b.y + (lo < a.x ? 1u : 0u)); }
uint2 sub64(uint2 a, uint2 b) { return uint2(a.x - b.x, a.y - b.y - (a.x < b.x ? 1u : 0u)); }
float f64(uint2 m) { return (float)m.y * 4294967296.0f + (float)m.x; }  // (m under 2^47: m.y exact)

uint key_of(float s) { uint k = asuint(s); return (k & 0x80000000u) ? ~k : (k | 0x80000000u); }
float of_key(uint k) { return asfloat((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k); }

// A thread's places: RUN of them side by side in a row (each template pixel loaded once for them, each window pixel
// once as it slides past them in registers), GROUP threads a group; a task's thread slots row by row, ceil(rw / RUN) a
// row (`groups`: a task and its first slot).
[numthreads(GROUP, 1, 1)]
void match(uint3 g : SV_GroupID, uint3 l : SV_GroupThreadID)
{
    uint2 gt = groups[g.x];
    Task t = tasks[gt.x];
    uint xb = (t.rw + RUN - 1) / RUN, slot = gt.y + l.x;
    uint py = slot / xb, px0 = (slot - py * xb) * RUN;
    uint kall = 0, knear = 0;
    if (py < t.rh) {
        Pic P = pics[t.pic];
        Win w = wins[t.win];
        uint sti[RUN];
        uint4 s[RUN];
        [unroll] for (uint i = 0; i < RUN; i++) {
            sti[i] = 0;
            s[i] = uint4(0, 0, 0, 0);
        }
        for (uint q = 0; q < P.nruns; q++) {
            uint rr = runs[P.run_off + q];
            uint r = rr & 255u, a = (rr >> 8) & 255u, b = rr >> 16;
            uint Q = planes(w.ww), j = px0 >> 2, trow = P.px_off + r * P.tw;
            uint wrow = w.px_off + (py + r) * 4 * Q + j, prow = w.pre_off + (py + r) * 4 * Q + j;
            [unroll] for (uint i = 0; i < RUN; i++)
                s[i] += prer[prow + in_planes(i + b, Q)] - prer[prow + in_planes(i + a, Q)];
            // The window's pixels under the RUN places at the picture's column x: w0..w3 (RUN 4), four columns a step
            // (each a register's turn, no moves within it), the picture's pixels four at a time.
            float3 w0 = unpacked(pxr[wrow + in_planes(a, Q)]), w1 = unpacked(pxr[wrow + in_planes(a + 1, Q)]);
            float3 w2 = unpacked(pxr[wrow + in_planes(a + 2, Q)]), w3;
            for (uint x = a; x < b;) {
                uint e = min(b, x + CHUNK);
                float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                for (; x + 4 <= e; x += 4) {
                    uint4 tp = tpx.Load4(4 * (trow + x));
                    float3 t0 = unpacked(tp.x), t1 = unpacked(tp.y), t2 = unpacked(tp.z), t3 = unpacked(tp.w);
                    w3 = unpacked(pxr[wrow + in_planes(x + 3, Q)]);
                    float3 w4 = unpacked(pxr[wrow + in_planes(x + 4, Q)]);
                    float3 w5 = unpacked(pxr[wrow + in_planes(x + 5, Q)]);
                    float3 w6 = unpacked(pxr[wrow + in_planes(x + 6, Q)]);
                    a0 += dot(t0, w0); a1 += dot(t0, w1); a2 += dot(t0, w2); a3 += dot(t0, w3);
                    a0 += dot(t1, w1); a1 += dot(t1, w2); a2 += dot(t1, w3); a3 += dot(t1, w4);
                    a0 += dot(t2, w2); a1 += dot(t2, w3); a2 += dot(t2, w4); a3 += dot(t2, w5);
                    a0 += dot(t3, w3); a1 += dot(t3, w4); a2 += dot(t3, w5); a3 += dot(t3, w6);
                    w0 = w4; w1 = w5; w2 = w6;
                }
                for (; x < e; x++) {
                    float3 tv = unpacked(tpx.Load(4 * (trow + x)));
                    w3 = unpacked(pxr[wrow + in_planes(x + 3, Q)]);
                    a0 += dot(tv, w0); a1 += dot(tv, w1); a2 += dot(tv, w2); a3 += dot(tv, w3);
                    w0 = w1; w1 = w2; w2 = w3;
                }
                sti[0] += (uint)a0;
                sti[1] += (uint)a1;
                sti[2] += (uint)a2;
                sti[3] += (uint)a3;
            }
        }
        [unroll] for (uint i = 0; i < RUN; i++) {
            uint px = px0 + i;
            if (px < t.rw) {
                uint4 u = s[i];
                uint2 N = sub64(mul64(P.n, sti[i]), add64(add64(mul64(P.st0, u.x), mul64(P.st1, u.y)),
                                                          mul64(P.st2, u.z)));
                uint2 B = sub64(mul64(P.n, u.w), add64(add64(mul64(u.x, u.x), mul64(u.y, u.y)), mul64(u.z, u.z)));
                float score = -1.0f;
                if (B.x != 0 || B.y != 0) {
                    bool neg = (N.y & 0x80000000u) != 0;
                    float nf = f64(neg ? sub64(uint2(0, 0), N) : N);
                    score = (neg ? -nf : nf) / sqrt(P.af * f64(B));
                }
                uint k = key_of(score), p = py * t.rw + px;
                kall = max(kall, k);
                if (px >= t.nx0 && px < t.nx1 && py >= t.ny0 && py < t.ny1)
                    knear = max(knear, k);
                if (ahead != 0) {
                    keys[t.place_off + p] = k;
                } else {
                    sums[t.place_off + p] = uint4(sti[i], u.x, u.y, u.z);
                    sc[t.place_off + p] = uint2(u.w, k);
                }
            }
        }
    }
    // Each thread's own (a group's reduced in groupshared first lost a group's highest now and then, with RUN 4 and 8,
    // not with 1: on the crowd's call set, with 8, 142 of 34,601 pairs then had more candidates than kcand; never fewer
    // than the best, so never another answer. D:/MapleHelper-scratch/gpu/check/dbg_det.py)
    uint old;
    if (kall)
        rec.InterlockedMax(gt.x * 16, kall, old);
    if (knear)
        rec.InterlockedMax(gt.x * 16 + 4, knear, old);
}

[numthreads(GROUP, 1, 1)]
void choose(uint3 g : SV_GroupID, uint3 l : SV_GroupThreadID)
{
    uint2 gt = groups[g.x];
    Task t = tasks[gt.x];
    uint xb = (t.rw + RUN - 1) / RUN, slot = gt.y + l.x;
    uint py = slot / xb, px0 = (slot - py * xb) * RUN;
    if (py >= t.rh)
        return;
    uint2 most = rec.Load2(gt.x * 16);
    float top[2] = {of_key(most.x) - band, of_key(most.y) - band};
    for (uint i = 0; i < RUN; i++) {
        uint px = px0 + i;
        if (px >= t.rw)
            break;
        uint p = py * t.rw + px;
        uint2 v = sc[t.place_off + p];
        float score = of_key(v.y);
        for (uint side = 0; side < 2; side++) {
            if (side == 1 && !(most.y != 0 && px >= t.nx0 && px < t.nx1 && py >= t.ny0 && py < t.ny1))
                break;
            if (score >= top[side]) {
                uint at;
                rec.InterlockedAdd(gt.x * 16 + 8 + 4 * side, 1, at);
                if (at < kcand) {
                    uint4 u = sums[t.place_off + p];
                    at = ((gt.x * 2 + side) * kcand + at) * 6;
                    cand[at] = p;
                    cand[at + 1] = u.x;
                    cand[at + 2] = u.y;
                    cand[at + 3] = u.z;
                    cand[at + 4] = u.w;
                    cand[at + 5] = v.x;
                }
            }
        }
    }
}
