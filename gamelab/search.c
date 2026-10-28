/* The monster search's step 1 in C (perception/mobs.py, MobDetector._detect), a band in a few calls, Python's lock let
 * go for each (ctypes): the band shrunk to step 1's scale (sr_shrink), then every kind's colour test, its colour at
 * 1/COARSE size and its integral images, each silhouette's match where the cover test keeps a window, the peaks
 * (sr_step1), and the places step 2 checks (sr_places). In numpy, OpenCV and Python they took 116 ms of CPU a whole
 * screen of a crowd on one thread, 82 of it OpenCV's matchTemplate over every window (the profile of 2026-10-27,
 * D:/MapleHelper-scratch/sfast/profile: 4.07 M windows scored a screen, 3.6% of them kept by the cover test, the
 * others' scores zeroed before the peaks); here the whole search takes 20 (README, A cheaper whole screen). Built
 * apart from native.c (its own
 * DLL: native.py's `search_lib`), with the same compiler and flags; AVX2's intrinsics, only where the CPU has it
 * (sr_ready: native.py asks first).
 *
 * What each gives:
 *   sr_shrink: OpenCV's INTER_AREA shrink (cv2.resize of a uint8 RGB picture to a smaller non-integer size: the band
 *     to SEARCH_SCALE, mobs.step1_frame), its float32 arithmetic in its order (OpenCV 5.0.0's resizeArea: each
 *     destination pixel's source weights from computeResizeAreaTab, summed across in the table's order, then down,
 *     rounded to the nearest, ties to even, and saturated): the same bytes (tests compare it with cv2.resize).
 *   sr_step1: master's step 1 (mobs._detect up to _spots), each silhouette's scores worked out exactly instead of in
 *     OpenCV's float32 DFT: a kind's colour as I4, the count of its colour's pixels in each 2x2 cell (its coarse
 *     picture, cv2.resize's INTER_AREA of the 0/1 mask, is I4 / 4 to the bit), and a silhouette's values (float32,
 *     every one a multiple of 2^-shift) as integers T_int = T x 2^shift; a window's TM_CCOEFF_NORMED score is then
 *     N / sqrt(A B) with N = n S_I4T - S_I4 S_T, A = n S_TT - S_T^2, B = n S_I4I4 - S_I4^2 (n its cells), every sum an
 *     exact integer (the products in doubles: each under 2^53, which the caller checks a silhouette's bits for), A
 *     rounded once to a double, the score rounded to float32. OpenCV's scores differ from these by at most 1.1e-6 (its
 *     DFT's roundings), so a peak can come and go where two neighbours' exact scores tie (OpenCV's noise picks one) or
 *     where one is within that of the threshold: on 589,478 silhouette matches of the saved frames 465 had another peak
 *     set, nearly all a tie's second peak, and no answer of the search was another (README). A window is scored only
 *     where the cover test keeps it (master zeroes the others' scores: MIN_COVER), never where its colour is flat (B 0:
 *     OpenCV's score is 0 there, its `diff2` test), and left out where its score can't reach thr: bounded from the
 *     window's colour by rows, then by columns, then by its exact products a few rows at a time (the rows of most
 *     template mass first), each bound an upper bound of N, the window left out where it falls under (thr - 1e-6)
 *     sqrt(A B), a margin far over every rounding (float32's spacing at 0.35 is 3e-8, the doubles' 1e-16): such a
 *     window scores under thr, so it's no peak and stops no neighbour scoring thr or more from being one. The peaks:
 *     master's rule (native.c's silhouette_peaks): at least thr and no less than any of the 8 neighbours.
 *   sr_places: master's mobs._places of a unit's spots (Python's sort of the (score, cx, cy, i) tuples, _to_check,
 *     _places), its places' spots one after another.
 * Scratch memory is the calling thread's own, kept from call to call (freed as the thread ends: DllMain). */
#include <immintrin.h>
#include <intrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define EXPORT __declspec(dllexport)

/* --- the CPU: AVX2's intrinsics are used, so the caller asks first (built without /arch:AVX2: this runs anywhere) */

EXPORT int sr_ready(void)
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

static __declspec(thread) Room bands_mem, planes_mem, hots_mem, peaks_mem, values_mem, start_mem, spots_mem, shrink_mem;

/* At least `need` bytes of `r`'s (its contents not kept); NULL without memory. `zero`: made zeroed (calloc), for the
 * score planes, which are kept zero between calls (sr_step1 clears what it wrote). */
static void *room(Room *r, size_t need, int zero)
{
    if (need > r->room) {
        free(r->at);
        need += need / 4;
        r->at = zero ? calloc(need, 1) : malloc(need);
        r->room = r->at ? need : 0;
    }
    return r->at;
}

typedef struct {
    int sw, dw, nb;
    int *base;        /* each 8 destinations' first source pixel */
    int *perm;        /* [b][k][8]: each lane's k-th source pixel - base, 0..15 */
    float *alpha;     /* [b][k][8]: its weight */
} XTab;

static __declspec(thread) XTab xt = {0};

static void forget(Room *r)
{
    free(r->at);
    r->at = 0;
    r->room = 0;
}

static void forget_xtab(void)
{
    free(xt.base);
    free(xt.perm);
    free(xt.alpha);
    memset(&xt, 0, sizeof(xt));
}

int __stdcall DllMain(void *instance, unsigned long reason, void *reserved)
{
    (void)instance;
    (void)reserved;
    if (reason == 3 || reason == 0) {  /* a thread ends, or the DLL goes */
        forget(&bands_mem);
        forget(&planes_mem);
        forget(&hots_mem);
        forget(&peaks_mem);
        forget(&values_mem);
        forget(&start_mem);
        forget(&spots_mem);
        forget(&shrink_mem);
        forget_xtab();
    }
    return 1;
}

#define A64(n) (((n) + 63) / 64 * 64)

/* --- the shrink: OpenCV's INTER_AREA ------------------------------------------------------------------------------ */

typedef struct {
    int di, si;
    float alpha;
} Alpha;

/* OpenCV's computeResizeAreaTab, for one channel: each destination's source pixels and their weights, in order. */
static int area_tab(int ssize, int dsize, double scale, Alpha *tab)
{
    int k = 0;
    for (int dx = 0; dx < dsize; dx++) {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cell = scale < ssize - fsx1 ? scale : ssize - fsx1;
        int sx1 = (int)ceil(fsx1), sx2 = (int)floor(fsx2);
        sx2 = sx2 < ssize - 1 ? sx2 : ssize - 1;
        sx1 = sx1 < sx2 ? sx1 : sx2;
        if (sx1 - fsx1 > 1e-3) {
            tab[k].di = dx;
            tab[k].si = sx1 - 1;
            tab[k++].alpha = (float)((sx1 - fsx1) / cell);
        }
        for (int sx = sx1; sx < sx2; sx++) {
            tab[k].di = dx;
            tab[k].si = sx;
            tab[k++].alpha = (float)(1.0 / cell);
        }
        if (fsx2 - sx2 > 1e-3) {
            double a = fsx2 - sx2 < 1.0 ? fsx2 - sx2 : 1.0;
            a = a < cell ? a : cell;
            tab[k].di = dx;
            tab[k].si = sx2;
            tab[k++].alpha = (float)(a / cell);
        }
    }
    return k;
}

#define AREA_K 3  /* source pixels a destination takes at most (a scale under 2) */

/* The tables across for a width (sw -> dw), kept for the next call of that width: 8 destinations at a time take their
 * (up to 3) source pixels from 16 consecutive ones (at a scale under 1.75). -1 where they don't fit that. */
static int make_xtab(int sw, int dw)
{
    if (xt.sw == sw && xt.dw == dw && xt.base)
        return 0;
    forget_xtab();
    double scale = 1. / ((double)dw / sw);  /* (OpenCV's inv_scale_x, as resize works it out) */
    int nb = (dw + 7) / 8;
    Alpha *tab = (Alpha *)malloc(sizeof(Alpha) * ((size_t)sw * 2 + 8));
    int *base = (int *)malloc(sizeof(int) * nb), *perm = (int *)malloc(sizeof(int) * nb * AREA_K * 8);
    float *alpha = (float *)malloc(sizeof(float) * nb * AREA_K * 8);
    int *cnt = (int *)calloc((size_t)dw + 8, sizeof(int));
    int *src = (int *)malloc(sizeof(int) * ((size_t)dw + 8) * AREA_K);
    float *al = (float *)malloc(sizeof(float) * ((size_t)dw + 8) * AREA_K);
    int ok = tab && base && perm && alpha && cnt && src && al;
    if (ok) {
        int n = area_tab(sw, dw, scale, tab);
        for (int k = 0; k < n && ok; k++) {
            int d = tab[k].di;
            if (cnt[d] >= AREA_K)
                ok = 0;
            else {
                src[d * AREA_K + cnt[d]] = tab[k].si;
                al[d * AREA_K + cnt[d]] = tab[k].alpha;
                cnt[d]++;
            }
        }
        for (int d = 0; d < dw && ok; d++)
            ok = cnt[d] > 0;
        for (int b = 0; b < nb && ok; b++) {
            int d0 = 8 * b, lo = src[d0 * AREA_K];
            base[b] = lo;
            for (int l = 0; l < 8; l++) {
                int d = d0 + l < dw ? d0 + l : dw - 1;  /* (past the last destination: its copy, not stored) */
                for (int k = 0; k < AREA_K; k++) {
                    int s = k < cnt[d] ? src[d * AREA_K + k] : src[d * AREA_K];
                    float a = k < cnt[d] ? al[d * AREA_K + k] : 0.0f;
                    if (s - lo < 0 || s - lo > 15)
                        ok = 0;
                    perm[(b * AREA_K + k) * 8 + l] = s - lo;
                    alpha[(b * AREA_K + k) * 8 + l] = a;
                }
            }
        }
    }
    free(tab);
    free(cnt);
    free(src);
    free(al);
    if (!ok) {
        free(base);
        free(perm);
        free(alpha);
        return -1;
    }
    xt.sw = sw;
    xt.dw = dw;
    xt.nb = nb;
    xt.base = base;
    xt.perm = perm;
    xt.alpha = alpha;
    return 0;
}

/* 16 RGB pixels (48 bytes at p) as three planes' bytes. */
static __forceinline void rgb_bytes(const unsigned char *p, __m128i *r, __m128i *g, __m128i *b)
{
    __m128i a = _mm_loadu_si128((const __m128i *)p), m = _mm_loadu_si128((const __m128i *)(p + 16)),
            c = _mm_loadu_si128((const __m128i *)(p + 32));
    const __m128i ra = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i rb = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    const __m128i rc = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
    const __m128i ga = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i gb = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    const __m128i gc = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
    const __m128i ba = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i bb = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    const __m128i bc = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);
    *r = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, ra), _mm_shuffle_epi8(m, rb)), _mm_shuffle_epi8(c, rc));
    *g = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, ga), _mm_shuffle_epi8(m, gb)), _mm_shuffle_epi8(c, gc));
    *b = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, ba), _mm_shuffle_epi8(m, bb)), _mm_shuffle_epi8(c, bc));
}

/* A source row as three float planes (each with 16 floats' room past its end, zeros). */
static void row_planes(const unsigned char *row, int sw, float *R, float *G, float *B)
{
    int x = 0;
    for (; x + 16 <= sw; x += 16) {
        __m128i r, g, b;
        rgb_bytes(row + 3 * x, &r, &g, &b);
        _mm256_storeu_ps(R + x, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r)));
        _mm256_storeu_ps(R + x + 8, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(r, 8))));
        _mm256_storeu_ps(G + x, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g)));
        _mm256_storeu_ps(G + x + 8, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(g, 8))));
        _mm256_storeu_ps(B + x, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b)));
        _mm256_storeu_ps(B + x + 8, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(b, 8))));
    }
    for (; x < sw; x++) {
        R[x] = (float)row[3 * x];
        G[x] = (float)row[3 * x + 1];
        B[x] = (float)row[3 * x + 2];
    }
    for (int k = 0; k < 16; k++)
        R[sw + k] = G[sw + k] = B[sw + k] = 0.0f;
}

/* One plane's pass across: out[dx] = ((S0 a0) + S1 a1) + S2 a2 (OpenCV's buf[dx], from 0 in the table's order), 8
 * destinations at a time. */
static void across(const float *P, float *out)
{
    for (int b = 0; b < xt.nb; b++) {
        const float *at = P + xt.base[b];
        __m256 lo = _mm256_loadu_ps(at), hi = _mm256_loadu_ps(at + 8);
        const int *pm = xt.perm + b * AREA_K * 8;
        const float *al = xt.alpha + b * AREA_K * 8;
        __m256 v = _mm256_setzero_ps();
        for (int k = 0; k < AREA_K; k++) {
            __m256i p = _mm256_loadu_si256((const __m256i *)(pm + 8 * k));
            __m256 s = _mm256_blendv_ps(_mm256_permutevar8x32_ps(lo, p), _mm256_permutevar8x32_ps(hi, p),
                                        _mm256_castsi256_ps(_mm256_slli_epi32(p, 28)));
            v = _mm256_add_ps(v, _mm256_mul_ps(s, _mm256_loadu_ps(al + 8 * k)));  /* (0 + x is x: the first as is) */
        }
        _mm256_storeu_ps(out + 8 * b, v);
    }
}

/* 8 floats rounded to the nearest (ties to even: cvRound) and saturated to uint8, as 8 bytes. */
static __forceinline unsigned long long bytes8(__m256 v)
{
    __m256i r = _mm256_cvtps_epi32(v);
    __m256i p16 = _mm256_packus_epi32(r, r);
    __m256i p8 = _mm256_packus_epi16(p16, p16);
    unsigned lo = (unsigned)_mm256_extract_epi32(p8, 0), hi = (unsigned)_mm256_extract_epi32(p8, 4);
    return ((unsigned long long)hi << 32) | lo;
}

/* A destination row from its three planes' sums, interleaved. */
static void put_row(unsigned char *D, const float *sr, const float *sg, const float *sb, int dw)
{
    char a[16], b[16], c[16], d[16];
    for (int j = 0; j < 16; j++) {
        int k = j / 3, ch = j % 3, jj = 16 + j, kk = jj / 3, cc = jj % 3;
        a[j] = ch == 0 ? (char)k : ch == 1 ? (char)(8 + k) : (char)-1;
        b[j] = ch == 2 ? (char)k : (char)-1;
        c[j] = j < 8 ? (cc == 0 ? (char)kk : cc == 1 ? (char)(8 + kk) : (char)-1) : (char)-1;
        d[j] = j < 8 && cc == 2 ? (char)kk : (char)-1;
    }
    const __m128i m0rg = _mm_loadu_si128((const __m128i *)a), m0b = _mm_loadu_si128((const __m128i *)b),
                  m1rg = _mm_loadu_si128((const __m128i *)c), m1b = _mm_loadu_si128((const __m128i *)d);
    int x = 0;
    for (; x + 8 <= dw; x += 8) {
        unsigned long long R = bytes8(_mm256_loadu_ps(sr + x)), G = bytes8(_mm256_loadu_ps(sg + x)),
                           B = bytes8(_mm256_loadu_ps(sb + x));
        __m128i rg = _mm_set_epi64x((long long)G, (long long)R), bb = _mm_cvtsi64_si128((long long)B);
        __m128i o0 = _mm_or_si128(_mm_shuffle_epi8(rg, m0rg), _mm_shuffle_epi8(bb, m0b));
        __m128i o1 = _mm_or_si128(_mm_shuffle_epi8(rg, m1rg), _mm_shuffle_epi8(bb, m1b));
        _mm_storeu_si128((__m128i *)(D + 3 * x), o0);
        _mm_storel_epi64((__m128i *)(D + 3 * x + 16), o1);
    }
    for (; x < dw; x++) {
        int v[3] = {_mm_cvtss_si32(_mm_set_ss(sr[x])), _mm_cvtss_si32(_mm_set_ss(sg[x])),
                    _mm_cvtss_si32(_mm_set_ss(sb[x]))};
        for (int k = 0; k < 3; k++)
            D[3 * x + k] = (unsigned char)(v[k] < 0 ? 0 : v[k] > 255 ? 255 : v[k]);
    }
}

/* cv2.resize(src, (dw, dh), interpolation=cv2.INTER_AREA) of a uint8 RGB picture (pixel (y, x) channel c at
 * src[y * srow + 3 x + c]) into dst, for a destination smaller each way at a scale under 1.75 across (each destination
 * pixel at most 3 source pixels each way): 0, or -1 where it declines (the caller has cv2.resize do it). OpenCV's
 * resizeArea for a non-integer scale (ResizeArea_Invoker, the float32 work type): a source row's sums across into
 * buf, then sum = beta buf (a destination row's first source row) or sum + beta buf, rounded and saturated; its
 * stripes start each destination row from zero, so how it splits them changes nothing. */
EXPORT int sr_shrink(const unsigned char *src, int sh, int sw, long long srow, unsigned char *dst, int dh, int dw,
                     long long drow)
{
    /* (a scale of 2 or more, or 1, one way: OpenCV's other paths, resizeAreaFast among them) */
    if (sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0 || dw >= sw || dh >= sh || 4LL * sw >= 7LL * dw
        || 4LL * sh >= 7LL * dh)
        return -1;
    if (make_xtab(sw, dw) != 0)
        return -1;
    double scale_y = 1. / ((double)dh / sh);
    int D8 = xt.nb * 8;
    size_t need = A64(sizeof(Alpha) * ((size_t)sh * 2 + 8)) + 3 * A64(sizeof(float) * ((size_t)sw + 16))
                  + 6 * A64(sizeof(float) * (size_t)D8);
    unsigned char *m = (unsigned char *)room(&shrink_mem, need, 0);
    if (!m)
        return -1;
    Alpha *ytab = (Alpha *)m;
    m += A64(sizeof(Alpha) * ((size_t)sh * 2 + 8));
    float *P[3], *buf[3], *sum[3];
    for (int c = 0; c < 3; c++) {
        P[c] = (float *)m;
        m += A64(sizeof(float) * ((size_t)sw + 16));
    }
    for (int c = 0; c < 3; c++) {
        buf[c] = (float *)m;
        m += A64(sizeof(float) * (size_t)D8);
        sum[c] = (float *)m;
        m += A64(sizeof(float) * (size_t)D8);
    }
    int yn = area_tab(sh, dh, scale_y, ytab);
    for (int j = 0, dy = 0; j < yn; j++) {  /* each destination row's source rows: at most AREA_K */
        int k = 0;
        while (j + k < yn && ytab[j + k].di == ytab[j].di)
            k++;
        if (k > AREA_K || ytab[j].di != dy)
            return -1;
        j += k - 1;
        dy++;
    }
    int last_sy = -1, prev_dy = ytab[0].di;
    for (int c = 0; c < 3; c++)
        memset(sum[c], 0, sizeof(float) * D8);
    for (int j = 0; j < yn; j++) {
        int dy = ytab[j].di, sy = ytab[j].si;
        __m256 vb = _mm256_set1_ps(ytab[j].alpha);
        if (sy != last_sy) {
            row_planes(src + (long long)sy * srow, sw, P[0], P[1], P[2]);
            for (int c = 0; c < 3; c++)
                across(P[c], buf[c]);
            last_sy = sy;
        }
        if (dy != prev_dy) {
            put_row(dst + (long long)prev_dy * drow, sum[0], sum[1], sum[2], dw);
            for (int c = 0; c < 3; c++)
                for (int q = 0; q < D8; q += 8)
                    _mm256_storeu_ps(sum[c] + q, _mm256_mul_ps(vb, _mm256_loadu_ps(buf[c] + q)));
            prev_dy = dy;
        } else {
            for (int c = 0; c < 3; c++)
                for (int q = 0; q < D8; q += 8)
                    _mm256_storeu_ps(sum[c] + q, _mm256_add_ps(_mm256_loadu_ps(sum[c] + q),
                                                               _mm256_mul_ps(vb, _mm256_loadu_ps(buf[c] + q))));
        }
    }
    put_row(dst + (long long)prev_dy * drow, sum[0], sum[1], sum[2], dw);
    return 0;
}

/* --- the colour tests: mobs.PALETTES' hand-made ones, and the learned gates' table (perception/gates.py) --------- */

typedef struct {
    int test;             /* 0: learned (its bits in the table); a hand-made test: 1 slime, 2 pig, 3 snail, 4 orange
                           * mushroom (mobs._slime, _pig, _snail, _orange_mushroom) */
    unsigned bits;        /* a learned unit's bits in the table (its kinds': any of them) */
    double least;         /* the pixels of its colour it needs to be looked for (MIN_BODY_PIXELS at step 1's px) */
    int first, n;         /* its shapes: shapes[first .. first + n) */
    int gfirst, gn;       /* its groups: groups[gfirst .. gfirst + gn) */
} SrUnit;

#define MAX_UNITS 16  /* (and at most 15 learned kinds: their bits in int16 lanes) */

static __forceinline int colour_of(int test, int r, int g, int b)
{
    switch (test) {
    case 1:  /* mobs._slime */
        return g > 170 && g - r > 30 && g - b > 90;
    case 2:  /* mobs._pig */
        return r > 190 && g > 120 && b > 110 && r - g > 18 && r - b > 15;
    case 3: { /* mobs._snail */
        int d = g - b < 0 ? b - g : g - b;
        int teal = r < 90 && g > 80 && b > 80 && g < 170 && d < 40;
        int lav = r > 140 && b > 190 && g > 130 && b - g > 15;
        return teal || lav;
    }
    case 4:  /* mobs._orange_mushroom */
        return r > 215 && g > 100 && g < 175 && b < 90 && r - g > 55;
    }
    return 0;
}

static __forceinline __m256i gt16(__m256i a, int v) { return _mm256_cmpgt_epi16(a, _mm256_set1_epi16((short)v)); }
static __forceinline __m256i lt16(__m256i a, int v) { return _mm256_cmpgt_epi16(_mm256_set1_epi16((short)v), a); }

/* 16 pixels' (int16 lanes) colour by hand-made test `test`: -1 where it's the kind's. */
static __forceinline __m256i colour16(int test, __m256i r, __m256i g, __m256i b)
{
    switch (test) {
    case 1:
        return _mm256_and_si256(_mm256_and_si256(gt16(g, 170), gt16(_mm256_sub_epi16(g, r), 30)),
                                gt16(_mm256_sub_epi16(g, b), 90));
    case 2:
        return _mm256_and_si256(_mm256_and_si256(_mm256_and_si256(gt16(r, 190), gt16(g, 120)),
                                                 _mm256_and_si256(gt16(b, 110), gt16(_mm256_sub_epi16(r, g), 18))),
                                gt16(_mm256_sub_epi16(r, b), 15));
    case 3: {
        __m256i teal = _mm256_and_si256(_mm256_and_si256(_mm256_and_si256(lt16(r, 90), gt16(g, 80)),
                                                         _mm256_and_si256(gt16(b, 80), lt16(g, 170))),
                                        lt16(_mm256_abs_epi16(_mm256_sub_epi16(g, b)), 40));
        __m256i lav = _mm256_and_si256(_mm256_and_si256(gt16(r, 140), gt16(b, 190)),
                                       _mm256_and_si256(gt16(g, 130), gt16(_mm256_sub_epi16(b, g), 15)));
        return _mm256_or_si256(teal, lav);
    }
    case 4:
        return _mm256_and_si256(_mm256_and_si256(_mm256_and_si256(gt16(r, 215), gt16(g, 100)),
                                                 _mm256_and_si256(lt16(g, 175), lt16(b, 90))),
                                gt16(_mm256_sub_epi16(r, g), 55));
    }
    return _mm256_setzero_si256();
}

/* 16 RGB pixels (48 bytes at p) as int16 r, g, b. */
static __forceinline void rgb16(const unsigned char *p, __m256i *r, __m256i *g, __m256i *b)
{
    __m128i R, G, B;
    rgb_bytes(p, &R, &G, &B);
    *r = _mm256_cvtepu8_epi16(R);
    *g = _mm256_cvtepu8_epi16(G);
    *b = _mm256_cvtepu8_epi16(B);
}

/* The 16 pixels' learned bits (gathered from the table, uint32 entries: gates.Gates.lut widened), int16 lanes in pixel
 * order. The bin: 5 bits a channel (gates.BITS), r's highest. */
static __forceinline __m256i bits16(const unsigned *lut, __m256i r, __m256i g, __m256i b)
{
    __m256i idx = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi16(_mm256_srli_epi16(r, 3), 10),
                                                  _mm256_slli_epi16(_mm256_srli_epi16(g, 3), 5)),
                                  _mm256_srli_epi16(b, 3));
    __m256i lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(idx));
    __m256i hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(idx, 1));
    __m256i vlo = _mm256_i32gather_epi32((const int *)lut, lo, 4);
    __m256i vhi = _mm256_i32gather_epi32((const int *)lut, hi, 4);
    return _mm256_permute4x64_epi64(_mm256_packs_epi32(vlo, vhi), 0xD8);  /* (15 bits at most: packs keeps them) */
}

/* Every unit's I4 picture (ch x cw cells: its colour's pixels in each 2x2 cell, 0..4) and its pixels' count, in one
 * pass over the band's pixels (h x w, both even: mobs._detect cuts step 1's frame so). */
static void colours(const unsigned char *rgb, int h, int w, long long row, const unsigned *lut, const SrUnit *units,
                    int nu, unsigned char **I4, long long *count)
{
    int ch = h / 2, cw = w / 2, learned = 0;
    for (int u = 0; u < nu; u++) {
        count[u] = 0;
        learned |= units[u].test == 0;
    }
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i pick = _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                          0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i cnt[MAX_UNITS];
    for (int u = 0; u < nu; u++)
        cnt[u] = _mm256_setzero_si256();
    int xs = (2 * cw) / 16 * 16;  /* pixels done 16 at a time */
    for (int y = 0; y < ch; y++) {
        const unsigned char *r0 = rgb + (long long)(2 * y) * row, *r1 = r0 + row;
        for (int x = 0; x < xs; x += 16) {
            __m256i R0, G0, B0, R1, G1, B1, L0 = _mm256_setzero_si256(), L1 = _mm256_setzero_si256();
            rgb16(r0 + 3 * x, &R0, &G0, &B0);
            rgb16(r1 + 3 * x, &R1, &G1, &B1);
            if (learned) {
                L0 = bits16(lut, R0, G0, B0);
                L1 = bits16(lut, R1, G1, B1);
            }
            for (int u = 0; u < nu; u++) {
                __m256i m0, m1;
                if (units[u].test == 0) {
                    __m256i bits = _mm256_set1_epi16((short)units[u].bits);
                    m0 = _mm256_cmpgt_epi16(_mm256_and_si256(L0, bits), _mm256_setzero_si256());
                    m1 = _mm256_cmpgt_epi16(_mm256_and_si256(L1, bits), _mm256_setzero_si256());
                } else {
                    m0 = colour16(units[u].test, R0, G0, B0);
                    m1 = colour16(units[u].test, R1, G1, B1);
                }
                __m256i s = _mm256_sub_epi16(_mm256_setzero_si256(), _mm256_add_epi16(m0, m1));  /* 0..2 a column */
                __m256i v = _mm256_madd_epi16(s, ones);  /* 8 cells: 0..4 */
                cnt[u] = _mm256_add_epi32(cnt[u], v);
                __m256i p = _mm256_shuffle_epi8(v, pick);
                unsigned char *dst = I4[u] + (long long)y * cw + x / 2;
                int lo = _mm256_extract_epi32(p, 0), hi = _mm256_extract_epi32(p, 4);
                memcpy(dst, &lo, 4);
                memcpy(dst + 4, &hi, 4);
            }
        }
        for (int u = 0; u < nu; u++)
            memset(I4[u] + (long long)y * cw + xs / 2, 0, cw - xs / 2);
        for (int x = xs; x < 2 * cw; x++) {
            const unsigned char *p[2] = {r0 + 3 * x, r1 + 3 * x};
            for (int k = 0; k < 2; k++) {
                int r = p[k][0], g = p[k][1], b = p[k][2];
                unsigned bits = learned ? lut[((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)] : 0;
                for (int u = 0; u < nu; u++) {
                    int on = units[u].test == 0 ? (bits & units[u].bits) != 0 : colour_of(units[u].test, r, g, b);
                    if (on) {
                        I4[u][(long long)y * cw + (x >> 1)]++;
                        count[u]++;
                    }
                }
            }
        }
    }
    for (int u = 0; u < nu; u++) {
        int lanes[8];
        _mm256_storeu_si256((__m256i *)lanes, cnt[u]);
        for (int k = 0; k < 8; k++)
            count[u] += lanes[k];
    }
}

/* Each unit's I4 picture and count, as sr_step1 makes them (for the tests: mobs' colour tests and the coarse picture
 * compared with them). I4 out: nu pictures of (h / 2) x (w / 2) bytes one after another. 0, or -2 past MAX_UNITS. */
EXPORT int sr_colours(const unsigned char *rgb, int h, int w, long long row, const unsigned *lut, const SrUnit *units,
                      int nu, unsigned char *out, long long *count)
{
    if (nu > MAX_UNITS || nu < 0)
        return -2;
    unsigned char *I4[MAX_UNITS];
    for (int u = 0; u < nu; u++)
        I4[u] = out + (size_t)u * (h / 2) * (w / 2);
    colours(rgb, h, w, row, lut, units, nu, I4, count);
    return 0;
}

/* --- the silhouettes' scores ------------------------------------------------------------------------------------ */

typedef struct {
    int th, tw, tw4;      /* the silhouette's size (cells); its rows' doubles in `rows`, tw4 a row (zeros after tw) */
    int kmin;             /* the least S_I4 that passes the cover test (MIN_COVER of its mean, as native.peaks) */
    long long st;         /* S_T in T_int units (exact) */
    double A;             /* n S_TT - S_T^2 in T_int units, rounded once */
    double scale;         /* 2^shift: T_int = T x scale */
    const double *rows;   /* T, th rows of tw4 doubles */
    const long long *ub;  /* row r's U[s4] at ub[r * (4 tw + 1) + s4], T_int units: the most a row's sum of I4 x T
                           * can be with s4 quarters of colour in it (4s on the row's largest values, the rest on the
                           * next) */
    const int *order;     /* the rows, most of T's mass first */
    int check;            /* rows multiplied between the early tests */
    int unused;
} SrShape;

#define GL 16  /* a group's shapes: one int16 lane each */

typedef struct {
    int ng;               /* its shapes (of one unit, one size, in order), up to GL */
    int idx[GL];          /* their indices in `shapes` */
    const short *ug;      /* [th][4 tw + 1][GL]: each shape's U, rounded up to units of 2^-ku of T (still bounds) */
    const short *vg;      /* [tw][4 th + 1][GL]: its column bound V, to units of 2^-kv */
    double su[GL];        /* 2^(shift_j - ku): a U sum in shape j's T_int units */
    double sv[GL];
} SrGroup;

#define MAX_TH 64         /* a silhouette's rows at most (cells); its columns at most 4 MAX_TH */

static __forceinline int cut(double L, double m2AB)
{
    return L <= 0 || L * L < m2AB;
}

/* A group's shapes' scores over a unit's I4 picture (its doubles `pd`, integral `ii`, integral of squares `sq`, each
 * row's running sums `rpt` (by column, then row), each column's `cp`): each window kept by the cover test whose score
 * can reach thr scored into its shape's plane (rh x rw, zero elsewhere), its place in `hots`. */
static void group_scores(const SrGroup *G, const SrShape *shapes, int ch, int cw, const double *pd, const int *ii,
                         const int *sq, const int *rpt, const int *cp, float thr, float *planes, int *hots, int *nhot,
                         long long *stats)
{
    const int ng = G->ng;
    const SrShape *g[GL];
    for (int j = 0; j < ng; j++)
        g[j] = &shapes[G->idx[j]];
    const int th = g[0]->th, tw = g[0]->tw, rh = ch - th + 1, rw = cw - tw + 1, iw = cw + 1, n = th * tw;
    const size_t plane = (size_t)rh * rw;
    int kmin = g[0]->kmin;
    for (int j = 0; j < ng; j++) {
        nhot[j] = 0;
        if (g[j]->kmin < kmin)
            kmin = g[j]->kmin;
    }
    long long tested = 0, kept = 0, rowcut = 0, colcut = 0, early = 0, scored = 0, flat = 0;
    const double margin = (double)thr - 1e-6, m2 = margin * margin;
    const long long ugw = (4LL * tw + 1) * GL, vgw = (4LL * th + 1) * GL;
    const __m256i vk = _mm256_set1_epi32(kmin - 1);
    int rs[MAX_TH + 8], cs[MAX_TH * 4 + 8];
    long long ur[MAX_TH];
    __declspec(align(32)) int au[GL], av[GL];
    for (int y = 0; y < rh; y++) {
        const int *top = ii + (long long)y * iw, *bot = ii + (long long)(y + th) * iw;
        for (int x = 0; x < rw; x += 8) {
            int m, sums[8];
            if (x + 8 <= rw) {  /* 8 windows' colour at once: the cover test */
                __m256i br = _mm256_loadu_si256((const __m256i *)(bot + x + tw));
                __m256i tr = _mm256_loadu_si256((const __m256i *)(top + x + tw));
                __m256i bl = _mm256_loadu_si256((const __m256i *)(bot + x));
                __m256i tl = _mm256_loadu_si256((const __m256i *)(top + x));
                __m256i s = _mm256_add_epi32(_mm256_sub_epi32(_mm256_sub_epi32(br, tr), bl), tl);
                m = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(s, vk)));
                _mm256_storeu_si256((__m256i *)sums, s);
                tested += 8;
            } else {
                m = 0;
                for (int k = 0; k < rw - x; k++) {
                    sums[k] = bot[x + k + tw] - top[x + k + tw] - bot[x + k] + top[x + k];
                    if (sums[k] >= kmin)
                        m |= 1 << k;
                }
                tested += rw - x;
            }
            while (m) {
                unsigned long k;
                _BitScanForward(&k, (unsigned long)m);
                m &= m - 1;
                const int xx = x + (int)k;
                const long long si = sums[k];
                const int *qt = sq + (long long)y * iw + xx, *qb = sq + (long long)(y + th) * iw + xx;
                const long long sii = (long long)qb[tw] - qt[tw] - qb[0] + qt[0];
                const long long B = (long long)n * sii - si * si;
                if (B <= 0) {  /* flat: OpenCV's score is 0 */
                    flat++;
                    continue;
                }
                {   /* the window's row sums and column sums of I4 */
                    const int *a0 = rpt + (long long)xx * (ch + 1) + y, *a1 = rpt + (long long)(xx + tw) * (ch + 1) + y;
                    int r = 0;
                    for (; r + 8 <= th; r += 8)
                        _mm256_storeu_si256((__m256i *)(rs + r),
                                            _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)(a1 + r)),
                                                             _mm256_loadu_si256((const __m256i *)(a0 + r))));
                    for (; r < th; r++)
                        rs[r] = a1[r] - a0[r];
                    const int *c0 = cp + (long long)y * cw + xx, *c1 = cp + (long long)(y + th) * cw + xx;
                    int c = 0;
                    for (; c + 8 <= tw; c += 8)
                        _mm256_storeu_si256((__m256i *)(cs + c),
                                            _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)(c1 + c)),
                                                             _mm256_loadu_si256((const __m256i *)(c0 + c))));
                    for (; c < tw; c++)
                        cs[c] = c1[c] - c0[c];
                }
                {   /* every shape's row bound and column bound at once */
                    __m256i lo = _mm256_setzero_si256(), hi = _mm256_setzero_si256();
                    const short *U = G->ug;
                    for (int r = 0; r < th; r++, U += ugw) {
                        __m256i v = _mm256_loadu_si256((const __m256i *)(U + (long long)rs[r] * GL));
                        lo = _mm256_add_epi32(lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v)));
                        hi = _mm256_add_epi32(hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1)));
                    }
                    _mm256_store_si256((__m256i *)au, lo);
                    _mm256_store_si256((__m256i *)(au + 8), hi);
                    lo = _mm256_setzero_si256();
                    hi = _mm256_setzero_si256();
                    const short *V = G->vg;
                    for (int c = 0; c < tw; c++, V += vgw) {
                        __m256i v = _mm256_loadu_si256((const __m256i *)(V + (long long)cs[c] * GL));
                        lo = _mm256_add_epi32(lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v)));
                        hi = _mm256_add_epi32(hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1)));
                    }
                    _mm256_store_si256((__m256i *)av, lo);
                    _mm256_store_si256((__m256i *)(av + 8), hi);
                }
                for (int j = 0; j < ng; j++) {
                    const SrShape *t = g[j];
                    if (si < t->kmin)
                        continue;
                    kept++;
                    const double m2AB = m2 * t->A * (double)B;
                    const double sist = (double)si * (double)t->st;
                    if (cut((double)n * ((double)au[j] * G->su[j]) - sist, m2AB)) {
                        rowcut++;
                        continue;
                    }
                    if (cut((double)n * ((double)av[j] * G->sv[j]) - sist, m2AB)) {
                        colcut++;
                        continue;
                    }
                    long long rest = 0;
                    {
                        const long long *Ue = t->ub;
                        const long long ubw = 4LL * tw + 1;
                        for (int r = 0; r < th; r++, Ue += ubw) {
                            ur[r] = Ue[rs[r]];
                            rest += ur[r];
                        }
                    }
                    long long sit = 0;  /* S_I4T, T_int units */
                    int out = 0, done = 0;
                    const double *base = pd + (long long)y * cw + xx;
                    const int check = t->check, tw4 = t->tw4;
                    const double scale = t->scale;
                    while (done < th) {
                        int upto = done + check < th ? done + check : th;
                        __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
                        for (int q = done; q < upto; q++) {
                            int r = t->order[q];
                            rest -= ur[r];
                            if (!rs[r])
                                continue;  /* (no colour in it: adds nothing) */
                            const double *irow = base + (long long)r * cw, *trow = t->rows + (long long)r * tw4;
                            int c = 0;
                            for (; c + 8 <= tw4; c += 8) {
                                __m256d p0 = _mm256_mul_pd(_mm256_loadu_pd(irow + c), _mm256_loadu_pd(trow + c));
                                __m256d p1 = _mm256_mul_pd(_mm256_loadu_pd(irow + c + 4),
                                                           _mm256_loadu_pd(trow + c + 4));
                                a0 = _mm256_add_pd(a0, p0);
                                a1 = _mm256_add_pd(a1, p1);
                            }
                            if (c < tw4)
                                a0 = _mm256_add_pd(a0, _mm256_mul_pd(_mm256_loadu_pd(irow + c),
                                                                     _mm256_loadu_pd(trow + c)));
                        }
                        double lanes[4];
                        _mm256_storeu_pd(lanes, _mm256_add_pd(a0, a1));
                        sit += (long long)(((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) * scale);  /* (exact) */
                        done = upto;
                        if (done < th && cut((double)((long long)n * (sit + rest) - si * t->st), m2AB)) {
                            out = 1;
                            break;
                        }
                    }
                    if (out) {
                        early++;
                        continue;
                    }
                    scored++;
                    const long long N = (long long)n * sit - si * t->st;
                    const int at = y * rw + xx;
                    planes[j * plane + at] = (float)((double)N / sqrt(t->A * (double)B));
                    hots[j * plane + nhot[j]++] = at;
                }
            }
        }
    }
    if (stats) {
        stats[0] += tested;
        stats[1] += kept;
        stats[2] += rowcut;
        stats[3] += colcut;
        stats[4] += early;
        stats[5] += scored;
        stats[6] += flat;
    }
}

/* A shape's peaks in its plane (its scored windows `hot`, row by row): at least thr and no less than any of the 8
 * neighbours (native.c's silhouette_peaks), into pyx (row, column) and pv, at most `cap`; the plane zeroed again where
 * it was written. Their count, or -1 past cap. */
static int shape_peaks(float *res, const int *hot, int nhot, int rh, int rw, float thr, long long *pyx, float *pv,
                       int cap)
{
    int found = 0;
    for (int h = 0; h < nhot && found >= 0; h++) {
        int at = hot[h], y = at / rw, x = at % rw;
        float v = res[at];
        if (!(v >= thr))
            continue;
        const float *row = res + (long long)y * rw;
        const float *up = y > 0 ? row - rw : 0, *down = y + 1 < rh ? row + rw : 0;
        int x0 = x > 0 ? x - 1 : x, x1 = x + 1 < rw ? x + 1 : x, peak = 1;
        for (int i = x0; i <= x1 && peak; i++) {
            if (row[i] > v || (up && up[i] > v) || (down && down[i] > v))
                peak = 0;
        }
        if (!peak)
            continue;
        if (found >= cap) {
            found = -1;
            break;
        }
        pyx[2 * found] = y;
        pyx[2 * found + 1] = x;
        pv[found] = v;
        found++;
    }
    for (int h = 0; h < nhot; h++)
        res[hot[h]] = 0.0f;
    return found;
}

/* Step 1 of a band (mobs._detect's: its colour tests, each kind's coarse picture and cover, every shape's silhouette
 * matched, the cover test, the peaks), on step 1's frame (uint8 RGB, h x w both even; pixel (y, x) at rgb[y * row +
 * 3 x]). `lut`: the learned gates' table as uint32 (bits of at most 15 kinds); `units` (nu), `shapes` (nshapes) and
 * `groups` as above, each unit's shapes in its groups. For each shape (by index), out_n its peaks' count (-1: its
 * unit's colour too little to be looked for, MIN_BODY_PIXELS), its peaks in ys, xs (row and column of the window's
 * top-left, in cells) and vs (the score), shape after shape, row by row. Returns the peaks in all, -1 past `cap`, -2
 * where it can't (too many units, a silhouette too big, no memory): the caller then does it its own way. `stats`
 * (NULL: none): adds windows tested, kept by the cover test (a shape's), left out by rows, by columns, early,
 * scored, flat. */
EXPORT int sr_step1(const unsigned char *rgb, int h, int w, long long row, const unsigned *lut, const SrUnit *units,
                    int nu, const SrShape *shapes, int nshapes, const SrGroup *groups, float thr, int *out_n,
                    long long *ys, long long *xs, float *vs, int cap, long long *stats)
{
    if (nu > MAX_UNITS || nu < 0 || h < 0 || w < 0 || (h & 1) || (w & 1))
        return -2;
    for (int u = 0; u < nu; u++) {
        if (units[u].first < 0 || units[u].n < 0 || units[u].first + units[u].n > nshapes)
            return -2;
        for (int gi = 0; gi < units[u].gn; gi++) {
            const SrGroup *G = groups + units[u].gfirst + gi;
            if (G->ng < 1 || G->ng > GL)
                return -2;
            for (int q = 0; q < G->ng; q++) {
                int k = G->idx[q];
                if (k < units[u].first || k >= units[u].first + units[u].n || shapes[k].th != shapes[G->idx[0]].th
                    || shapes[k].tw != shapes[G->idx[0]].tw || shapes[k].th > MAX_TH || shapes[k].tw > 4 * MAX_TH
                    || shapes[k].th < 1 || shapes[k].tw < 1)
                    return -2;
            }
        }
    }
    const int ch = h / 2, cw = w / 2, iw = cw + 1;
    const size_t plane = (size_t)ch * cw, iplane = (size_t)(ch + 1) * iw, cplane = (size_t)(ch + 1) * cw;
    const size_t tplane = (size_t)(cw + 1) * (ch + 1) + 16;
    const size_t per = A64(plane + 32) + 2 * A64(iplane * sizeof(int)) + A64(tplane * sizeof(int)) +
                       A64(cplane * sizeof(int)) + A64((plane + 8) * sizeof(double));
    unsigned char *mem = (unsigned char *)room(&bands_mem, per * (nu ? nu : 1) + 64, 0);
    long long *pyx = (long long *)room(&peaks_mem, 2 * sizeof(long long) * ((size_t)cap + 1), 0);
    float *pv = (float *)room(&values_mem, sizeof(float) * ((size_t)cap + 1), 0);
    if (!mem || !pyx || !pv)
        return -2;
    unsigned char *I4[MAX_UNITS];
    int *II[MAX_UNITS], *SQ[MAX_UNITS], *RPT[MAX_UNITS], *CP[MAX_UNITS];
    double *PD[MAX_UNITS];
    for (int u = 0; u < nu; u++) {
        unsigned char *p = mem + per * u;
        I4[u] = p;
        p += A64(plane + 32);
        II[u] = (int *)p;
        p += A64(iplane * sizeof(int));
        SQ[u] = (int *)p;
        p += A64(iplane * sizeof(int));
        RPT[u] = (int *)p;
        p += A64(tplane * sizeof(int));
        CP[u] = (int *)p;
        p += A64(cplane * sizeof(int));
        PD[u] = (double *)p;
    }
    int *start = (int *)room(&start_mem, sizeof(int) * ((size_t)nshapes + 1), 0);  /* each shape's first peak */
    if (!start)
        return -2;
    long long count[MAX_UNITS];
    colours(rgb, h, w, row, lut, units, nu, I4, count);
    for (int k = 0; k < nshapes; k++)
        out_n[k] = start[k] = 0;
    int total = 0;
    for (int u = 0; u < nu; u++) {
        if ((double)count[u] < units[u].least) {
            for (int j = 0; j < units[u].n; j++)
                out_n[units[u].first + j] = -1;
            continue;
        }
        int *ii = II[u], *sq = SQ[u], *rpt = RPT[u], *cp = CP[u];
        double *pd = PD[u];
        const unsigned char *a = I4[u];
        for (int x = 0; x <= cw; x++)
            ii[x] = sq[x] = 0;
        for (int x = 0; x < cw; x++)
            cp[x] = 0;
        for (int y = 0; y < ch; y++) {
            int run = 0, run2 = 0;
            int *ri = ii + (long long)(y + 1) * iw, *ui = ii + (long long)y * iw;
            int *rq = sq + (long long)(y + 1) * iw, *uq = sq + (long long)y * iw;
            int *cn = cp + (long long)(y + 1) * cw, *cu = cp + (long long)y * cw;
            const unsigned char *ar = a + (long long)y * cw;
            double *pr = pd + (long long)y * cw;
            ri[0] = rq[0] = 0;
            rpt[y] = 0;
            for (int x = 0; x < cw; x++) {
                int v = ar[x];
                run += v;
                run2 += v * v;
                ri[x + 1] = ui[x + 1] + run;
                rq[x + 1] = uq[x + 1] + run2;
                rpt[(long long)(x + 1) * (ch + 1) + y] = run;
                cn[x] = cu[x] + v;
                pr[x] = (double)v;
            }
        }
        for (int i = 0; i < 8; i++)
            pd[plane + i] = 0.0;
        for (int gi = 0; gi < units[u].gn; gi++) {
            const SrGroup *G = groups + units[u].gfirst + gi;
            const SrShape *t0 = &shapes[G->idx[0]];
            int rh = ch - t0->th + 1, rw = cw - t0->tw + 1;
            if (rh <= 0 || rw <= 0)
                continue;  /* (the band smaller than the shape: no peaks) */
            size_t pl = (size_t)rh * rw;
            float *planes = (float *)room(&planes_mem, sizeof(float) * pl * G->ng, 1);
            int *hots = (int *)room(&hots_mem, sizeof(int) * pl * G->ng, 0);
            if (!planes || !hots)
                return -2;
            int nhot[GL];
            group_scores(G, shapes, ch, cw, pd, ii, sq, rpt, cp, thr, planes, hots, nhot, stats);
            int over = 0;
            for (int q = 0; q < G->ng; q++) {  /* (every plane zeroed again, even past cap) */
                int k = shape_peaks(planes + q * pl, hots + q * pl, nhot[q], rh, rw, thr, pyx + 2 * total, pv + total,
                                    over ? 0 : cap - total);
                if (k < 0 || over) {
                    over = 1;
                    continue;
                }
                start[G->idx[q]] = total;
                out_n[G->idx[q]] = k;
                total += k;
            }
            if (over)
                return -1;
        }
    }
    int at = 0;  /* each shape's peaks, shape after shape (the groups wrote them group by group) */
    for (int k = 0; k < nshapes; k++) {
        for (int q = 0; q < out_n[k]; q++) {
            ys[at + q] = pyx[2 * (start[k] + q)];
            xs[at + q] = pyx[2 * (start[k] + q) + 1];
            vs[at + q] = pv[start[k] + q];
        }
        at += out_n[k] > 0 ? out_n[k] : 0;
    }
    return at;
}

/* --- the places ------------------------------------------------------------------------------------------------ */

typedef struct {
    double v;
    long long cx, cy;
    int i;
} Spot;

/* Python's sorted(reverse=True) of the (score, cx, cy, i) tuples: no two are the same (a shape's peaks are apart). */
static int desc(const void *pa, const void *pb)
{
    const Spot *a = (const Spot *)pa, *b = (const Spot *)pb;
    if (a->v != b->v)
        return a->v > b->v ? -1 : 1;
    if (a->cx != b->cx)
        return a->cx > b->cx ? -1 : 1;
    if (a->cy != b->cy)
        return a->cy > b->cy ? -1 : 1;
    if (a->i != b->i)
        return a->i > b->i ? -1 : 1;
    return 0;
}

#define MAXP 64  /* places a unit at most (MAX_CHECKS) */
#define MAXE 16  /* a place's spots at most (FINE_TEMPLATES) */

/* Each unit's places from sr_step1's peaks (ys, xs, vs and out_n as it gave them; sh, sw each shape's size in cells;
 * ufirst[u], un[u] a unit's shapes, ufirst -1 for one not looked for, which gets nothing), as mobs._places makes them
 * of the unit's spots (mobs._spots: score, cx = COARSE x + COARSE sw // 2, cy likewise, i the shape's index in its
 * unit; COARSE 2): out, for each unit looked for, its places' count, then each place's spots' count and its spots
 * (cx, cy, i), one after another. The ints written, -1 past `cap`, -2 past MAXP or MAXE. */
EXPORT int sr_places(const long long *ys, const long long *xs, const float *vs, const int *out_n, const int *sh,
                     const int *sw, const int *ufirst, const int *un, int nu, long long slack, int most_checks,
                     int per_place, int *out, int cap)
{
    int w = 0, at = 0;
    if (most_checks > MAXP || per_place > MAXE || most_checks < 0 || per_place < 1)
        return -2;
    for (int u = 0; u < nu; u++) {
        if (ufirst[u] < 0)
            continue;
        int ns = 0;
        for (int j = 0; j < un[u]; j++)
            ns += out_n[ufirst[u] + j] > 0 ? out_n[ufirst[u] + j] : 0;
        Spot *s = (Spot *)room(&spots_mem, sizeof(Spot) * ((size_t)ns + 1), 0);
        if (!s)
            return -2;
        int k = 0;
        for (int j = 0; j < un[u]; j++) {
            int shape = ufirst[u] + j, c = out_n[shape] > 0 ? out_n[shape] : 0;
            for (int q = 0; q < c; q++, at++) {
                s[k].v = (double)vs[at];
                s[k].cx = 2 * xs[at] + (2 * sw[shape]) / 2;
                s[k].cy = 2 * ys[at] + (2 * sh[shape]) / 2;
                s[k].i = j;
                k++;
            }
        }
        qsort(s, ns, sizeof(Spot), desc);
        long long px[MAXP], py[MAXP];  /* _to_check */
        int pi[MAXP], np = 0;
        for (int q = 0; q < ns && np < most_checks; q++) {
            int near = 0;
            for (int p = 0; p < np && !near; p++)
                near = llabs(s[q].cx - px[p]) < slack && llabs(s[q].cy - py[p]) < slack;
            if (near)
                continue;
            px[np] = s[q].cx;
            py[np] = s[q].cy;
            pi[np] = s[q].i;
            np++;
        }
        long long ex[MAXP][MAXE], ey[MAXP][MAXE];  /* _places: every spot, best first, joins the first place near it */
        int ei[MAXP][MAXE], ne[MAXP];
        for (int p = 0; p < np; p++) {
            ex[p][0] = px[p];
            ey[p][0] = py[p];
            ei[p][0] = pi[p];
            ne[p] = 1;
        }
        for (int q = 0; q < ns && np; q++) {
            int which = -1;
            for (int p = 0; p < np && which < 0; p++)
                if (llabs(s[q].cx - px[p]) < slack && llabs(s[q].cy - py[p]) < slack)
                    which = p;
            if (which < 0 || ne[which] >= per_place)
                continue;
            int dup = 0;
            for (int e = 0; e < ne[which] && !dup; e++)
                dup = ei[which][e] == s[q].i;
            if (dup)
                continue;
            ex[which][ne[which]] = s[q].cx;
            ey[which][ne[which]] = s[q].cy;
            ei[which][ne[which]] = s[q].i;
            ne[which]++;
        }
        if (w + 1 > cap)
            return -1;
        out[w++] = np;
        for (int p = 0; p < np; p++) {
            if (w + 1 + 3 * ne[p] > cap)
                return -1;
            out[w++] = ne[p];
            for (int e = 0; e < ne[p]; e++) {
                out[w++] = (int)ex[p][e];
                out[w++] = (int)ey[p][e];
                out[w++] = ei[p][e];
            }
        }
    }
    return w;
}
