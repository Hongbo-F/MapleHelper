/* Monsters fading in at their spawn points, in C: the sums of the spawn watch's blend fit (perception/spawns.py), where
 * numpy took 13.6 ms a frame at the median on the user's own play (the prototype of 2026-11-01, D:/MapleHelper-scratch/
 * spawn-fade/proto) and 132 at p99. Each gives what the Python beside it gives, to the bit: everything here is integer
 * sums (the game's alpha as 0..255, the colours as they are), the scores worked out from them in Python. The Python is
 * the fallback where this DLL isn't (no compiler, no AVX2), and the tests compare the two. Built apart from native.c
 * (its own DLL: native.py's `spawns_lib`), with the same compiler and flags; AVX2's intrinsics, only where the CPU has
 * it (sp_ready: native.py asks first).
 *
 * The fit (spawns.py's top): a newborn of opacity a over the map's art B is drawn F = B + a (A / 255) (S - B), A the
 * game's alpha of each of its picture's pixels, S its colour. With D = F - B and G = A (S - B), each channel of each
 * pixel sampled, the sums D.G, G.G, D.D and the pixels kept (n) give a = 255 D.G / G.G, the share of D.D explained
 * (D.G)^2 / (G.G D.D) and the residual. |D| <= 255, |G| <= 65025: D.G and D.D a pixel fit int32 (three channels:
 * 4.97e7, 195075), G.G doesn't (1.27e10); the sums go to int64.
 *
 * What each gives:
 *   sp_empty: a window's pixels whose largest channel differs from the art's by more than `levels` (the empty test).
 *   sp_sums: each hypothesis's sums. A hypothesis is a picture with its top-left at a window pixel; they come in runs
 *     of up to 8 a row, one column apart (its feet a unit apart along a platform's flat stretch), whose pixels lie
 *     side by side in memory: the window and its art are first copied into planes of their own (the calling thread's
 *     scratch), and a run's 8 hypotheses are 8 lanes of one load a pixel and channel. A run of 1 goes one at a time.
 * Scratch memory is the calling thread's own, kept from call to call (freed as the thread ends: DllMain). */
#include <immintrin.h>
#include <intrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define EXPORT __declspec(dllexport)
#define PAD 32       /* bytes past the last plane's end a run's lanes may read (and drop) */
#define BLOCK 32     /* pixels summed in int32 before going to int64: 32 x 3 x 255 x 65025 < 2^31 */
#define MOST_K 8192  /* pixels a picture at most (D.D's int32 lanes: 8192 x 195075 < 2^31) */

/* --- the CPU: AVX2's intrinsics are used, so the caller asks first (built without /arch:AVX2: this runs anywhere) */

EXPORT int sp_ready(void)
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

static __declspec(thread) Room plane_mem;

static void *room(Room *r, size_t need)
{
    if (r->room >= need)
        return r->at;
    _aligned_free(r->at);
    r->at = _aligned_malloc(need, 32);
    r->room = r->at ? need : 0;
    return r->at;
}

BOOL WINAPI DllMain(HINSTANCE dll, DWORD why, LPVOID reserved)
{
    (void)dll;
    (void)reserved;
    if (why == DLL_THREAD_DETACH || why == DLL_PROCESS_DETACH) {
        _aligned_free(plane_mem.at);
        plane_mem.at = NULL;
        plane_mem.room = 0;
    }
    return TRUE;
}

/* --- the empty test ----------------------------------------------------------------------------------------------- */

/* The pixels of a w x h window of `frame` (rows `fstride` bytes apart, `fpx` bytes a pixel, RGB first) whose largest
   channel differs from `plate`'s (the art's same box: rows `pstride` apart, 3 bytes a pixel) by more than `levels`. */
EXPORT int sp_empty(const uint8_t *frame, int64_t fstride, int fpx, const uint8_t *plate, int64_t pstride, int w, int h,
                    int levels)
{
    int n = 0;
    for (int y = 0; y < h; y++) {
        const uint8_t *f = frame + y * fstride, *b = plate + y * pstride;
        for (int x = 0; x < w; x++, f += fpx, b += 3) {
            int m = 0;
            for (int c = 0; c < 3; c++) {
                int d = (int)f[c] - (int)b[c];
                if (d < 0)
                    d = -d;
                if (d > m)
                    m = d;
            }
            n += m > levels;
        }
    }
    return n;
}

/* --- the sums -------------------------------------------------------------------------------------------------- */

/* One hypothesis, one pixel at a time: its window pixel o0 + off[j] (planes F, B of `hw` px each; K: keep, or NULL). */
static void one(const uint8_t *F, const uint8_t *B, const uint8_t *K, int64_t hw, int64_t o0, int k, const int32_t *off,
                const uint8_t *S, const uint8_t *A, int64_t *out)
{
    int64_t dg = 0, gg = 0, dd = 0, n = 0;
    for (int j = 0; j < k; j++) {
        int64_t o = o0 + off[j];
        if (K && !K[o])
            continue;
        n++;
        for (int c = 0; c < 3; c++) {
            int d = (int)F[c * hw + o] - (int)B[c * hw + o];
            int g = (int)A[j] * ((int)S[3 * j + c] - (int)B[c * hw + o]);
            dg += (int64_t)d * g;
            gg += (int64_t)g * g;
            dd += (int64_t)d * d;
        }
    }
    out[0] = dg;
    out[1] = gg;
    out[2] = dd;
    out[3] = n;
}

static inline __m256i lanes_u8(const uint8_t *p)
{
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p));
}

/* Its int64 lanes' sum into acc[0..3] (lanes 0-3) and acc[4..7] (4-7), from the 8 int32 lanes v. */
static inline void widen_add(int64_t *acc, __m256i v)
{
    int64_t t[8];
    _mm256_storeu_si256((__m256i *)t, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v)));
    _mm256_storeu_si256((__m256i *)(t + 4), _mm256_cvtepi32_epi64(_mm256_extracti128_si256(v, 1)));
    for (int i = 0; i < 8; i++)
        acc[i] += t[i];
}

/* A run's `lanes` hypotheses (up to 8), the first's top-left window pixel o0, the others one column on each. */
static void run8(const uint8_t *F, const uint8_t *B, const uint8_t *K, int64_t hw, int64_t o0, int lanes, int k,
                 const int32_t *off, const uint8_t *S, const uint8_t *A, int64_t *out)
{
    int64_t dg[8] = {0}, dd[8] = {0}, n[8] = {0};
    __m256i gg_even = _mm256_setzero_si256(), gg_odd = _mm256_setzero_si256();  /* lanes 0 2 4 6, 1 3 5 7 (int64) */
    const __m256i one_ = _mm256_set1_epi32(1);
    for (int j0 = 0; j0 < k; j0 += BLOCK) {
        int j1 = j0 + BLOCK < k ? j0 + BLOCK : k;
        __m256i vdg = _mm256_setzero_si256(), vdd = _mm256_setzero_si256(), vn = _mm256_setzero_si256();
        for (int j = j0; j < j1; j++) {
            int64_t o = o0 + off[j];
            __m256i keep = K ? _mm256_cmpeq_epi32(lanes_u8(K + o), one_) : _mm256_set1_epi32(-1);
            __m256i a = _mm256_set1_epi32(A[j]);
            vn = _mm256_sub_epi32(vn, keep);  /* (keep is -1 where kept) */
            for (int c = 0; c < 3; c++) {
                __m256i f = lanes_u8(F + c * hw + o), b = lanes_u8(B + c * hw + o);
                __m256i d = _mm256_and_si256(_mm256_sub_epi32(f, b), keep);
                __m256i s = _mm256_set1_epi32(S[3 * j + c]);
                __m256i g = _mm256_and_si256(_mm256_mullo_epi32(a, _mm256_sub_epi32(s, b)), keep);
                vdg = _mm256_add_epi32(vdg, _mm256_mullo_epi32(d, g));
                vdd = _mm256_add_epi32(vdd, _mm256_mullo_epi32(d, d));
                gg_even = _mm256_add_epi64(gg_even, _mm256_mul_epi32(g, g));
                __m256i godd = _mm256_srli_epi64(g, 32);
                gg_odd = _mm256_add_epi64(gg_odd, _mm256_mul_epi32(godd, godd));
            }
        }
        widen_add(dg, vdg);
        widen_add(dd, vdd);
        widen_add(n, vn);
    }
    int64_t ge[4], go[4];
    _mm256_storeu_si256((__m256i *)ge, gg_even);
    _mm256_storeu_si256((__m256i *)go, gg_odd);
    for (int i = 0; i < lanes; i++) {
        int64_t *q = out + 4 * i;
        q[0] = dg[i];
        q[1] = (i & 1) ? go[i >> 1] : ge[i >> 1];
        q[2] = dd[i];
        q[3] = n[i];
    }
}

/* The sums of every hypothesis (out: 4 int64 each, in the runs' order, a run's lanes in turn: D.G, G.G, D.D, n) of a
   w x h window of `frame` (rows `fstride` bytes apart, `fpx` bytes a pixel, RGB first) over the art's same box
   (`plate`: rows `pstride` apart, 3 bytes a pixel), `keep` (w x h bytes, 1 where a pixel counts; NULL: all; copied).
   runs[4r..]: its picture, the top-left's row and column in the window, its lanes (1-8: columns col .. col + lanes -
   1). off[p k + j]: picture p's j-th pixel's window offset from its top-left (its row times w plus its column);
   S[(p k + j) 3 + c] its colour, A[p k + j] its alpha (1-255). Every hypothesis's pixels must lie inside the window
   (the caller's). Returns the hypotheses summed, or -1 (no memory, a picture too big). */
EXPORT int sp_sums(const uint8_t *frame, int64_t fstride, int fpx, const uint8_t *plate, int64_t pstride,
                   const uint8_t *keep, int w, int h, int n_runs, const int32_t *runs, int k, const int32_t *off,
                   const uint8_t *S, const uint8_t *A, int64_t *out)
{
    if (k > MOST_K || w <= 0 || h <= 0)
        return -1;
    int64_t hw = (int64_t)w * h;
    uint8_t *planes = room(&plane_mem, (size_t)(7 * hw + PAD));
    if (!planes)
        return -1;
    uint8_t *F = planes, *B = planes + 3 * hw, *K = keep ? planes + 6 * hw : NULL;
    for (int y = 0; y < h; y++) {
        const uint8_t *f = frame + y * fstride, *b = plate + y * pstride;
        int64_t r = (int64_t)y * w;
        for (int x = 0; x < w; x++, f += fpx, b += 3)
            for (int c = 0; c < 3; c++) {
                F[c * hw + r + x] = f[c];
                B[c * hw + r + x] = b[c];
            }
    }
    if (K)
        memcpy(K, keep, (size_t)hw);
    memset(planes + (K ? 7 : 6) * hw, 0, PAD);
    int done = 0;
    for (int r = 0; r < n_runs; r++) {
        int p = runs[4 * r], row = runs[4 * r + 1], col = runs[4 * r + 2], lanes = runs[4 * r + 3];
        int64_t o0 = (int64_t)row * w + col;
        const int32_t *po = off + (int64_t)p * k;
        const uint8_t *ps = S + (int64_t)p * k * 3, *pa = A + (int64_t)p * k;
        if (lanes == 1)
            one(F, B, K, hw, o0, k, po, ps, pa, out + 4 * (int64_t)done);
        else
            run8(F, B, K, hw, o0, lanes, k, po, ps, pa, out + 4 * (int64_t)done);
        done += lanes;
    }
    return done;
}
