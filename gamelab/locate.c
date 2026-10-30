/* Her place on screen in C: the inner loops of the two cues that find her where her name tag's gray match isn't sure
 * (the localization prototype of 2026-10-29, D:/MapleHelper-scratch/loc2/combined, ported): her name plate's exact
 * colours counted (perception/plate.py) and her body's poses checked pixel by pixel (perception/poses.py). Each gives
 * what the Python beside it gives, to the bit (everything here is integer counts, the shares count / n in doubles):
 * the Python is the fallback where this DLL isn't (no compiler, no AVX2), and the tests compare the two. Built apart
 * from native.c (its own DLL: native.py's `locate_lib`), with the same compiler and flags; AVX2's intrinsics, only
 * where the CPU has it (lc_ready: native.py asks first).
 *
 * What each gives:
 *   lc_plate: a window's plate candidates (plate.Plate.candidates): every pixel's palette index (the plate's own
 *     colours, a perfect multiplicative hash of their 24-bit codes; rows from the HUD down none), the count at each
 *     place of the plate's stable pixels whose exact colour the frame has there (32 places a step: a byte compare
 *     per pixel of the plate, summed a row of the plate at a time in bytes), places in the minimap zeroed, the best
 *     `topk` with their neighbourhoods suppressed (cv2.minMaxLoc's order: the first best row by row), and at each
 *     (the first one +-1 px) the evidence: the plate's frame px matched, her letters white, white where her name has
 *     none, and her letters in column bins all shown (`clean`).
 *   lc_pack + lc_agree: her body (poses.BodyFinder): a box of the frame packed a pixel an int32 (lc_pack, the calling
 *     thread's own), then for each candidate (a pose's pixels at her feet's column and row) and each offset within
 *     +-R of it, the share of its pixels whose colour is within the tolerance on every channel, as drawn or as the
 *     blink after a touch darkens her (the better of the two); pixels off the box don't agree. Eight offsets across
 *     at once where every pixel of the candidate stays in the box; one at a time with the box's bounds otherwise.
 * Scratch memory is the calling thread's own, kept from call to call (freed as the thread ends: DllMain). */
#include <immintrin.h>
#include <intrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EXPORT __declspec(dllexport)
#define NONE 255  /* a pixel of none of the plate's colours (its palette has at most 254) */

/* --- the CPU: AVX2's intrinsics are used, so the caller asks first (built without /arch:AVX2: this runs anywhere) */

EXPORT int lc_ready(void)
{
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7)
        return 0;
    __cpuid(r, 1);
    if (!(r[2] & (1 << 27)) || !(r[2] & (1 << 28)))  /* OSXSAVE, AVX */
        return 0;
    if ((_xgetbv(0) & 6) != 6)  /* the OS saves the YMM registers */
        return 0;
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;  /* AVX2 */
}

/* --- the calling thread's scratch memory ------------------------------------------------------------------------- */

typedef struct {
    void *at;
    size_t room;
} Room;

static __declspec(thread) Room idx_mem, res_mem, acc_mem, plane_mem;
static __declspec(thread) int plane_w, plane_h, plane_pw;  /* lc_pack's box: its size, and its rows' px (w + 8) */

/* At least `need` bytes of `r`'s, 32-byte aligned (its contents not kept); NULL without memory. */
static void *room(Room *r, size_t need)
{
    if (need > r->room) {
        _aligned_free(r->at);
        need += need / 4 + 64;
        r->at = _aligned_malloc(need, 32);
        r->room = r->at ? need : 0;
    }
    return r->at;
}

static void forget(Room *r)
{
    _aligned_free(r->at);
    r->at = 0;
    r->room = 0;
}

int __stdcall DllMain(void *instance, unsigned long reason, void *reserved)
{
    (void)instance;
    (void)reserved;
    if (reason == 3 || reason == 0) {  /* a thread ends, or the DLL goes */
        forget(&idx_mem);
        forget(&res_mem);
        forget(&acc_mem);
        forget(&plane_mem);
    }
    return 1;
}

/* --- her name plate ----------------------------------------------------------------------------------------------- */

typedef struct {
    uint32_t mul;             /* the palette's hash: (code * mul) >> shift (mod 2^32) indexes its tables */
    int shift;
    const uint8_t *h_idx;     /* [1 << (32 - shift)]: a palette colour's index there, NONE elsewhere */
    const int32_t *h_code;    /* [the same]: that colour's code (R << 16 | G << 8 | B), -1 elsewhere */
    int th, tw;               /* the count template: the box of the plate's stable px */
    int n;                    /* its stable px, row by row (rs: where each of its rows starts, th + 1 of them) */
    const int32_t *rs;
    const int16_t *ty, *tx;   /* each one's row and column in the box */
    const uint8_t *ti;        /* and its palette index */
    int ox, oy;               /* the box's top-left from the plate's centre */
    int bx, by, bh, bw;       /* the plate's whole picture (its evidence): top-left from the centre, height, width */
    int nf;                   /* its frame's px but the black outline: row, column in the picture, palette index */
    const int16_t *fy, *fx;
    const uint8_t *fi;
    int nl;                   /* her name's letters (white): row, column, column bin (-1: none) */
    const int16_t *ly, *lx;
    const int8_t *lb;
    int ng;                   /* the name's translucent background (white there: another name's letters) */
    const int16_t *gy, *gx;
    const int8_t *gb;
    int white;                /* white's palette index */
    int nbins;                /* the bins: her letters in each, the least a bin counts with, the share of them that
                                 must show and the stray white it may have */
    const int32_t *ltot;
    int min_let;
    double recall;
    int bad_max;
} LcPlate;

/* The palette index of every pixel of the frame's box (x0, y0)-(x1, y1), rows of `iw` (the box's width and 32 more,
 * NONE: the counts' 32-byte loads reach past it), rows from `hud` down NONE. */
static void plate_index(const LcPlate *m, const uint8_t *frame, long long stride, int x0, int y0, int x1, int y1,
                        int hud, uint8_t *idx, int iw)
{
    int ww = x1 - x0, wh = y1 - y0;
    for (int y = 0; y < wh; y++) {
        uint8_t *row = idx + (size_t)y * iw;
        memset(row + ww, NONE, (size_t)(iw - ww));
        if (y0 + y >= hud) {
            memset(row, NONE, (size_t)ww);
            continue;
        }
        const uint8_t *p = frame + (long long)(y0 + y) * stride + (long long)x0 * 3;
        for (int x = 0; x < ww; x++, p += 3) {
            uint32_t code = ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2];
            uint32_t h = (code * m->mul) >> m->shift;
            row[x] = m->h_code[h] == (int32_t)code ? m->h_idx[h] : NONE;
        }
    }
}

/* The counts (res, rh x rw, int32): at each place, the stable px whose palette index the frame's has there. */
static void plate_counts(const LcPlate *m, const uint8_t *idx, int iw, int rh, int rw, int32_t *res, uint8_t *acc)
{
    int nv = (rw + 31) / 32;
    uint16_t *sum = (uint16_t *)(acc + (size_t)nv * 32);  /* a row's counts in 16 bits (1,395 at most here) */
    for (int ry = 0; ry < rh; ry++) {
        memset(sum, 0, (size_t)nv * 32 * sizeof(uint16_t));
        for (int ty = 0; ty < m->th; ty++) {
            int e0 = m->rs[ty], e1 = m->rs[ty + 1];
            if (e1 <= e0)
                continue;
            const uint8_t *row = idx + (size_t)(ry + ty) * iw;
            for (int base = 0; base < nv; base += 4) {  /* 4 vectors (128 places) at a time, in registers */
                __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
                const uint8_t *r0 = row + (size_t)base * 32;
                for (int e = e0; e < e1; e++) {
                    __m256i v = _mm256_set1_epi8((char)m->ti[e]);
                    const uint8_t *p = r0 + m->tx[e];
                    a0 = _mm256_sub_epi8(a0, _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)p), v));
                    a1 = _mm256_sub_epi8(a1, _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)(p + 32)), v));
                    a2 = _mm256_sub_epi8(a2, _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)(p + 64)), v));
                    a3 = _mm256_sub_epi8(a3, _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)(p + 96)), v));
                }
                /* a row of the plate counts 255 at most (its px in a row: e1 - e0, under its width) */
                __m256i a[4] = {a0, a1, a2, a3};
                for (int j = 0; j < 4 && base + j < nv; j++) {
                    uint16_t *s = sum + (size_t)(base + j) * 32;
                    __m256i lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a[j]));
                    __m256i hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a[j], 1));
                    _mm256_storeu_si256((__m256i *)s, _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)s), lo));
                    _mm256_storeu_si256((__m256i *)(s + 16),
                                        _mm256_add_epi16(_mm256_loadu_si256((const __m256i *)(s + 16)), hi));
                }
            }
        }
        int32_t *out = res + (size_t)ry * rw;
        for (int rx = 0; rx < rw; rx++)
            out[rx] = sum[rx];
    }
}

/* The evidence at plate centre (cx, cy): its frame's px matched, her letters white, white on the name's background,
 * and the letters of the bins that show them all with no stray white (`clean`). Zero where the prototype's padded
 * window can't hold the plate's picture there. */
static void plate_evidence(const LcPlate *m, const uint8_t *idx, int iw, int ww, int wh, int x0, int y0, int cx, int cy,
                           int *frame, int *letters, int *bad, int *clean)
{
    *frame = *letters = *bad = 0;
    if (clean)
        *clean = 0;
    int oy = cy + m->by - y0 + m->bh, ox = cx + m->bx - x0 + m->bw;
    if (oy < 0 || ox < 0 || oy + m->bh > wh + 2 * m->bh || ox + m->bw > ww + 2 * m->bw)
        return;
    int top = cy + m->by - y0, left = cx + m->bx - x0;
    int lmb[64], badb[64];
    memset(lmb, 0, sizeof(lmb));
    memset(badb, 0, sizeof(badb));
    for (int i = 0; i < m->nf; i++) {
        int y = top + m->fy[i], x = left + m->fx[i];
        if (y >= 0 && y < wh && x >= 0 && x < ww && idx[(size_t)y * iw + x] == m->fi[i])
            (*frame)++;
    }
    for (int i = 0; i < m->nl; i++) {
        int y = top + m->ly[i], x = left + m->lx[i];
        if (y >= 0 && y < wh && x >= 0 && x < ww && idx[(size_t)y * iw + x] == m->white) {
            (*letters)++;
            if (m->lb[i] >= 0)
                lmb[m->lb[i]]++;
        }
    }
    for (int i = 0; i < m->ng; i++) {
        int y = top + m->gy[i], x = left + m->gx[i];
        if (y >= 0 && y < wh && x >= 0 && x < ww && idx[(size_t)y * iw + x] == m->white) {
            (*bad)++;
            if (m->gb[i] >= 0)
                badb[m->gb[i]]++;
        }
    }
    if (clean)
        for (int k = 0; k < m->nbins && k < 64; k++)
            if (m->ltot[k] >= m->min_let && (double)lmb[k] >= m->recall * (double)m->ltot[k] && badb[k] <= m->bad_max)
                *clean += lmb[k];
}

/* plate.Plate.candidates in C: the window (x0, y0)-(x1, y1) of the frame (already within it), its best `topk`
 * places by the counts (over `floor`, each suppressing +-sup_y rows and +-sup_x columns of the counts around it), the
 * first refined +-1 px (when `around0`) by frame + w_letters letters - 2 bad. out[k * 7 ...]: that score, the centre's
 * x and y, frame, letters, bad, clean. Returns how many, or -1 without memory. */
EXPORT int lc_plate(const LcPlate *m, const uint8_t *frame, long long stride, int x0, int y0, int x1, int y1, int hud,
                    int mm_rows, int mm_cols, int topk, int sup_y, int sup_x, int floor, int w_letters, int around0,
                    int *out)
{
    int ww = x1 - x0, wh = y1 - y0;
    if (ww < m->tw || wh < m->th)
        return 0;
    int iw = ww + 32;
    uint8_t *idx = (uint8_t *)room(&idx_mem, (size_t)iw * wh + 256);
    int rh = wh - m->th + 1, rw = ww - m->tw + 1;
    int32_t *res = (int32_t *)room(&res_mem, (size_t)rh * rw * sizeof(int32_t) + 64);
    int nv = (rw + 31) / 32;
    uint8_t *acc = (uint8_t *)room(&acc_mem, (size_t)nv * 32 * 3 + 64);
    if (!idx || !res || !acc)
        return -1;
    plate_index(m, frame, stride, x0, y0, x1, y1, hud, idx, iw);
    plate_counts(m, idx, iw, rh, rw, res, acc);
    int c0x = x0 - m->ox, c0y = y0 - m->oy;
    if (mm_cols - c0x > 0 && mm_rows - c0y > 0) {
        int zr = mm_rows - c0y < rh ? mm_rows - c0y : rh, zc = mm_cols - c0x < rw ? mm_cols - c0x : rw;
        for (int y = 0; y < zr; y++)
            memset(res + (size_t)y * rw, 0, (size_t)zc * sizeof(int32_t));
    }
    int found = 0;
    for (int k = 0; k < topk; k++) {
        int best = INT32_MIN, by = 0, bx = 0;
        for (int y = 0; y < rh; y++) {
            const int32_t *r = res + (size_t)y * rw;
            for (int x = 0; x < rw; x++)
                if (r[x] > best) {
                    best = r[x];
                    by = y;
                    bx = x;
                }
        }
        if (best <= floor)
            break;
        for (int y = by - sup_y < 0 ? 0 : by - sup_y; y <= by + sup_y && y < rh; y++)
            for (int x = bx - sup_x < 0 ? 0 : bx - sup_x; x <= bx + sup_x && x < rw; x++)
                res[(size_t)y * rw + x] = -1000000000;
        int px = c0x + bx, py = c0y + by, r = (k == 0 && around0) ? 1 : 0;
        int bs = 0, bxx = 0, byy = 0, bf = 0, bl = 0, bb = 0, have = 0;
        for (int dy = -r; dy <= r; dy++)
            for (int dx = -r; dx <= r; dx++) {
                int f, l, b;
                plate_evidence(m, idx, iw, ww, wh, x0, y0, px + dx, py + dy, &f, &l, &b, NULL);
                int s = f + w_letters * l - 2 * b;
                if (!have || s > bs) {
                    have = 1;
                    bs = s;
                    bxx = px + dx;
                    byy = py + dy;
                    bf = f;
                    bl = l;
                    bb = b;
                }
            }
        int f, l, b, c;
        plate_evidence(m, idx, iw, ww, wh, x0, y0, bxx, byy, &f, &l, &b, &c);
        int *o = out + (size_t)found * 7;
        o[0] = bs;
        o[1] = bxx;
        o[2] = byy;
        o[3] = bf;
        o[4] = bl;
        o[5] = bb;
        o[6] = c;
        found++;
    }
    return found;
}

/* The counts alone (the tests' check of plate_counts against numpy's): res (rh x rw) for the window. */
EXPORT int lc_plate_counts(const LcPlate *m, const uint8_t *frame, long long stride, int x0, int y0, int x1, int y1,
                           int hud, int32_t *res)
{
    int ww = x1 - x0, wh = y1 - y0;
    if (ww < m->tw || wh < m->th)
        return 0;
    int iw = ww + 32, rw = ww - m->tw + 1, rh = wh - m->th + 1, nv = (rw + 31) / 32;
    uint8_t *idx = (uint8_t *)room(&idx_mem, (size_t)iw * wh + 256);
    uint8_t *acc = (uint8_t *)room(&acc_mem, (size_t)nv * 32 * 3 + 64);
    if (!idx || !acc)
        return -1;
    plate_index(m, frame, stride, x0, y0, x1, y1, hud, idx, iw);
    plate_counts(m, idx, iw, rh, rw, res, acc);
    return 1;
}

/* --- her body ------------------------------------------------------------------------------------------------------ */

/* The box (h x w) of an RGB picture, a pixel an int32 (R | G << 8 | B << 16), into the calling thread's plane, its rows
 * w + 8 px (lc_agree's loads reach 7 px past a candidate's last offset). 0, or -1 without memory. */
EXPORT int lc_pack(const uint8_t *pic, int h, int w, long long stride)
{
    int pw = w + 8;
    uint32_t *plane = (uint32_t *)room(&plane_mem, (size_t)pw * (h + 1) * sizeof(uint32_t) + 64);
    if (!plane)
        return -1;
    for (int y = 0; y < h; y++) {
        const uint8_t *p = pic + (long long)y * stride;
        uint32_t *row = plane + (size_t)y * pw;
        for (int x = 0; x < w; x++, p += 3)
            row[x] = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
        memset(row + w, 0, 8 * sizeof(uint32_t));
    }
    memset(plane + (size_t)h * pw, 0, (size_t)pw * sizeof(uint32_t));
    plane_w = w;
    plane_h = h;
    plane_pw = pw;
    return 0;
}

static int within(uint32_t a, uint32_t c, int t)
{
    int dr = (int)(a & 255) - (int)(c & 255), dg = (int)((a >> 8) & 255) - (int)((c >> 8) & 255),
        db = (int)((a >> 16) & 255) - (int)((c >> 16) & 255);
    return dr <= t && dr >= -t && dg <= t && dg >= -t && db <= t && db >= -t;
}

/* For each candidate c (a pixel set: ys, xs from its feet, colours col0 as drawn, col1 darkened, within t0 and t1; sets'
 * [first, n, ymin, ymax, xmin, xmax] in `sets`; the candidate's set cset[c] at feet column cfx[c] and row cfy[c] of the
 * packed box) and each offset (oy, ox) within +-R (oy the slower), the share of its px agreeing in the better colour
 * mode: out[c * (2R+1)^2 + ...]. 0, or -1 (R over 3, nothing packed). */
EXPORT int lc_agree(const int32_t *ys, const int32_t *xs, const uint32_t *col0, const uint32_t *col1, int t0, int t1,
                    const int32_t *sets, int nc, const int32_t *cset, const int32_t *cfx, const int32_t *cfy, int R,
                    double *out)
{
    const uint32_t *plane = (const uint32_t *)plane_mem.at;
    if (R < 0 || R > 3 || !plane)
        return -1;
    int w = plane_w, h = plane_h, pw = plane_pw, side = 2 * R + 1, nout = side * side;
    __m256i tol0 = _mm256_set1_epi32(t0 | (t0 << 8) | (t0 << 16)), tol1 = _mm256_set1_epi32(t1 | (t1 << 8) | (t1 << 16));
    __m256i ones = _mm256_set1_epi32(-1);
    for (int c = 0; c < nc; c++) {
        const int32_t *s = sets + (size_t)cset[c] * 6;
        int first = s[0], n = s[1], fx = cfx[c], fy = cfy[c];
        double *o = out + (size_t)c * nout;
        if (n <= 0) {
            for (int k = 0; k < nout; k++)
                o[k] = 0.0;
            continue;
        }
        const int32_t *py = ys + first, *px = xs + first;
        const uint32_t *p0 = col0 + first, *p1 = col1 + first;
        if (fy - R + s[2] >= 0 && fy + R + s[3] < h && fx - R + s[4] >= 0 && fx + R + s[5] < w) {
            /* every px in the box: 8 offsets across a load (ox -R .. -R + 7; those past +R unused) */
            for (int oyi = 0; oyi < side; oyi++) {
                __m256i n0 = _mm256_setzero_si256(), n1 = _mm256_setzero_si256();
                const uint32_t *base = plane + (size_t)(fy - R + oyi) * pw + (fx - R);
                for (int i = 0; i < n; i++) {
                    __m256i a = _mm256_loadu_si256((const __m256i *)(base + (ptrdiff_t)py[i] * pw + px[i]));
                    __m256i c0 = _mm256_set1_epi32((int)p0[i]), c1 = _mm256_set1_epi32((int)p1[i]);
                    __m256i d0 = _mm256_max_epu8(_mm256_subs_epu8(a, c0), _mm256_subs_epu8(c0, a));
                    __m256i d1 = _mm256_max_epu8(_mm256_subs_epu8(a, c1), _mm256_subs_epu8(c1, a));
                    __m256i k0 = _mm256_cmpeq_epi32(_mm256_cmpeq_epi8(_mm256_min_epu8(d0, tol0), d0), ones);
                    __m256i k1 = _mm256_cmpeq_epi32(_mm256_cmpeq_epi8(_mm256_min_epu8(d1, tol1), d1), ones);
                    n0 = _mm256_sub_epi32(n0, k0);
                    n1 = _mm256_sub_epi32(n1, k1);
                }
                __declspec(align(32)) int32_t a0[8], a1[8];
                _mm256_store_si256((__m256i *)a0, n0);
                _mm256_store_si256((__m256i *)a1, n1);
                for (int oxi = 0; oxi < side; oxi++) {
                    double s0 = (double)a0[oxi] / n, s1 = (double)a1[oxi] / n;
                    o[oyi * side + oxi] = s0 >= s1 ? s0 : s1;
                }
            }
        } else {
            for (int oyi = 0; oyi < side; oyi++)
                for (int oxi = 0; oxi < side; oxi++) {
                    int k0 = 0, k1 = 0, by = fy - R + oyi, bx = fx - R + oxi;
                    for (int i = 0; i < n; i++) {
                        int y = by + py[i], x = bx + px[i];
                        if (y < 0 || y >= h || x < 0 || x >= w)
                            continue;
                        uint32_t a = plane[(size_t)y * pw + x];
                        k0 += within(a, p0[i], t0);
                        k1 += within(a, p1[i], t1);
                    }
                    double s0 = (double)k0 / n, s1 = (double)k1 / n;
                    o[oyi * side + oxi] = s0 >= s1 ? s0 : s1;
                }
        }
    }
    return 0;
}

/* Her colours' mass picture (poses.BodyFinder.mass_picture) of an h x w box of an RGB picture: 1 where the pixel is one
 * of her colours (lut: 64 levels a channel, R << 12 | G << 6 | B) and in the foreground (over `fg` from the art `bg`, an
 * int16 RGB picture of the box, on some channel, or where the art isn't known), -0.5 other foreground, 0 the art;
 * without art (bg NULL), 1 her colours and -0.25 the rest. out: h x w float32. */
EXPORT int lc_mass(const uint8_t *pic, int h, int w, long long stride, const int16_t *bg, long long bg_stride,
                   const uint8_t *known, long long known_stride, const uint8_t *lut, int fg, float *out)
{
    for (int y = 0; y < h; y++) {
        const uint8_t *p = pic + (long long)y * stride;
        float *o = out + (size_t)y * w;
        const int16_t *b = bg ? bg + (long long)y * bg_stride : 0;
        const uint8_t *k = bg ? known + (long long)y * known_stride : 0;
        for (int x = 0; x < w; x++, p += 3) {
            int her = lut[((p[0] >> 2) << 12) | ((p[1] >> 2) << 6) | (p[2] >> 2)] != 0;
            if (!b) {
                o[x] = her ? 1.0f : -0.25f;
                continue;
            }
            int d0 = p[0] - b[3 * x], d1 = p[1] - b[3 * x + 1], d2 = p[2] - b[3 * x + 2];
            d0 = d0 < 0 ? -d0 : d0;
            d1 = d1 < 0 ? -d1 : d1;
            d2 = d2 < 0 ? -d2 : d2;
            int d = d0 > d1 ? d0 : d1;
            d = d > d2 ? d : d2;
            int fore = d > fg || !k[x];
            o[x] = fore ? (her ? 1.0f : -0.5f) : 0.0f;
        }
    }
    return 0;
}
