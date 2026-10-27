/* gamelab's pieces in C (native.py builds and loads this; each caller does the same in Python when it isn't
 * there). Called through ctypes, which lets go of Python's lock for the call. Each but the masked match does exactly
 * what the Python it stands in for does, in the same order and with the same IEEE double and float arithmetic (built
 * with /fp:precise: no fused multiply-adds, no reordering), so the answers are the same to the bit. The masked match
 * works out OpenCV's score exactly, in integers, where OpenCV rounds its float sums (see there); a frame's re-finds in
 * one call (rf_*) give what the re-finds' Python gave with it, to the bit (their contract there).
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/* --- the monster search (perception/mobs.py, _spots) ----------------------------------------------------
 * Step 1's bookkeeping after a silhouette's match: in numpy it held Python's lock, and the search's threads
 * took turns at it (18 ms of a band search's 34).
 *   1. a window with too little of the kind's colour scores 0: its colour, summed from the integral image as
 *      ((ii[y+h][x+w] - ii[y][x+w]) - ii[y+h][x]) + ii[y][x] (numpy's order), divided by the template's size,
 *      below `cover` (in double, as numpy compares float64 with a Python float);
 *   2. a peak is a window scoring at least `thr` and no less than any of its 8 neighbours (a 3x3 dilation's
 *      equality; outside the result counts as nothing), compared as float32 (numpy 2 casts the Python float to
 *      the array's float32);
 *   3. the peaks in row-major order, as 64-bit ints (np.nonzero's).
 *
 * `res`: the match's scores (rh x rw floats, row by row), zeroed in place where the cover is too low; `ii`: the
 * integral image of the kind's colour at that size ((rh + th) x (rw + tw) doubles); `th`, `tw`: the template's
 * size; `size`: th * tw; `cover`: the least colour a window needs (MIN_COVER * the template's mean); `thr`: the
 * least a peak scores. The peaks' rows, columns and scores go to `ys`, `xs` and `vs` (room for `cap`). Returns
 * how many, or -1 if more than `cap`. */
EXPORT int silhouette_peaks(float *res, int rh, int rw, const double *ii, int th, int tw, double size,
                            double cover, float thr, long long *ys, long long *xs, float *vs, int cap)
{
    int iw = rw + tw, n = 0;
    for (int y = 0; y < rh; y++) {
        const double *top = ii + (long long)y * iw, *bottom = ii + (long long)(y + th) * iw;
        float *row = res + (long long)y * rw;
        for (int x = 0; x < rw; x++) {
            double s = ((bottom[x + tw] - top[x + tw]) - bottom[x]) + top[x];
            if (s / size < cover)
                row[x] = 0.0f;
        }
    }
    for (int y = 0; y < rh; y++) {
        const float *row = res + (long long)y * rw;
        const float *up = y > 0 ? row - rw : 0, *down = y + 1 < rh ? row + rw : 0;
        for (int x = 0; x < rw; x++) {
            float v = row[x];
            if (!(v >= thr))
                continue;
            int x0 = x > 0 ? x - 1 : x, x1 = x + 1 < rw ? x + 1 : x, peak = 1;
            for (int i = x0; i <= x1 && peak; i++) {
                if (row[i] > v || (up && up[i] > v) || (down && down[i] > v))
                    peak = 0;
            }
            if (!peak)
                continue;
            if (n >= cap)
                return -1;
            ys[n] = y;
            xs[n] = x;
            vs[n] = v;
            n++;
        }
    }
    return n;
}


/* --- her flight (motion/body.py, fly, as motion/physics.py's flight calls it) ----------------------------
 * Where she lands from (x, y) moving at (vx, vy) with `arrow` held all the way: stepped dt at a time until
 * `limit` by the engine's air step (body.air_step: the fall cap's drag, gravity up to the cap, the arrow steering
 * toward the air speed, or with none the air's drag), walls stopping her sideways (the walls of the foothold group
 * she leaves, as physics hands them over; a wall at exactly her start, at the end of the platform she stands on,
 * stops her only going into it: body._side_at), the first platform (in the map's order) whose surface her feet
 * cross on the way down, through its corner too (body._lands_at), of those she could reach in the time (not
 * `skip`, within `reach` sideways, an end at or below `top`). Not airborne (a flight leaving the ground can't land
 * on its first step) and no map sides: what physics.flight asks of body.fly. In Python a flight takes about 1 ms;
 * the planner flies some 25,000 to build the map's moves.
 *
 * `shoe`: her shoe's fall, air_drag_fall, gravity, air_acc, air_speed and air_drag (body.Shoe). The walls:
 * `wall_xs` (sorted), `wall_lo`, `wall_hi` (each wall's top - 1 and bottom + 1: body.vertical_walls). Each of the
 * `n` platforms' ends `left[q]`, `right[q]`, its lower end's y `low[q]` (the larger y_at of its two ends) and its
 * segments, `seg_n[q]` of them from `seg_at[q]` in `segs` (x1, y1, x2, y2 each). Returns the platform landed on
 * (-1: none within the time), with where, when and her sideways speed then in `out` (x, t, vx), and in out[3]
 * whether that x (1) and that speed (2) are still the ones she started with, untouched by any step (Python then
 * hands back the very objects it was given: an int stays an int); -2 if `n` is more than it has room for.
 * Python's min and max are written out (min(a, b) is b only if b < a) so that even the sign of a zero comes out
 * as Python's. */
#define MAX_PLATFORMS 4096

static double segment_y(const double *s, double x)  /* mapdata.Segment.y_at */
{
    if (s[2] == s[0])
        return s[1];
    double t = (x - s[0]) / (s[2] - s[0]);
    return s[1] + t * (s[3] - s[1]);
}

typedef struct {
    int n;
    const double *left, *right, *low, *segs;
    const int *seg_at, *seg_n;
} Platforms;

static double y_at(const Platforms *m, int q, double x)  /* mapdata.Platform.y_at */
{
    const double *s = m->segs + 4 * (long long)m->seg_at[q];
    for (int i = 0; i < m->seg_n[q]; i++, s += 4) {
        if (s[0] <= x && x <= s[2])
            return segment_y(s, x);
    }
    const double *first = m->segs + 4 * (long long)m->seg_at[q];
    return x < m->left[q] ? first[1] : first[4 * (m->seg_n[q] - 1) + 3];
}

static int covers(const Platforms *m, int q, double x)  /* mapdata.Platform.covers */
{
    return m->left[q] <= x && x <= m->right[q];
}

/* body._accelerate and _decelerate; `*same` is cleared when the answer isn't v itself. */
static double accelerate(double v, double a, double limit, double dt, int *same)
{
    if (a > 0) {
        if (v < limit) {
            double w = v + a * dt;
            *same = 0;
            return limit < w ? limit : w;      /* min(w, limit) */
        }
        return v;
    }
    if (a < 0) {
        double low = -limit;
        if (v > low) {
            double w = v + a * dt;
            *same = 0;
            return low > w ? low : w;          /* max(w, -limit) */
        }
        return v;
    }
    return v;
}

static double decelerate(double v, double limit, double drag, double dt, int *same)
{
    if (v > limit) {
        double w = v - drag * dt;
        *same = 0;
        return limit > w ? limit : w;          /* max(w, limit) */
    }
    double low = -limit;
    if (v < low) {
        double w = v + drag * dt;
        *same = 0;
        return low < w ? low : w;              /* min(w, -limit) */
    }
    return v;
}

/* body._lands_at: where a step down from (x, y) to (nx, ny) puts her feet on platform q (2: at nx itself, 1:
 * through its corner); 0 if it doesn't. */
static int lands_at(const Platforms *m, int q, double x, double y, double nx, double ny, double *at)
{
    if (covers(m, q, nx)) {
        double surface = y_at(m, q, nx);
        if ((y <= surface || y <= y_at(m, q, x)) && surface <= ny + 0.01) {  /* either end's reading */
            *at = nx;
            return 2;
        }
        return 0;
    }
    if (nx == x || !covers(m, q, x))
        return 0;
    double end = nx > x ? m->right[q] : m->left[q];
    double before = y - y_at(m, q, x);
    double after = y + (ny - y) * (end - x) / (nx - x) - y_at(m, q, end);
    if (before > 0 || after < -0.01)
        return 0;
    *at = x + (end - x) * (after > before ? -before / (after - before) : 1.0);
    return 1;
}

static int side_at(const Platforms *m, double x, double y)  /* body._side_at, over all of the map's platforms */
{
    for (int q = 0; q < m->n; q++) {
        if (covers(m, q, x) && fabs(y_at(m, q, x) - y) <= 1.5) {
            if (x == m->right[q])
                return -1;
            if (x == m->left[q])
                return 1;
        }
    }
    return 0;
}

static int bisect_left(const double *xs, int n, double v)
{
    int a = 0, b = n;
    while (a < b) {
        int mid = (a + b) / 2;
        if (xs[mid] < v)
            a = mid + 1;
        else
            b = mid;
    }
    return a;
}

EXPORT int fly(double x, double y, double vx, double vy, int arrow, double dt, double limit, const double *shoe,
               int skip, const double *wall_xs, const double *wall_lo, const double *wall_hi, int walls, int n,
               const double *left, const double *right, const double *low, const int *seg_at, const int *seg_n,
               const double *segs, double *out)
{
    double fall = shoe[0], air_drag_fall = shoe[1], gravity = shoe[2], air_acc = shoe[3], air_speed = shoe[4],
           air_drag = shoe[5];
    Platforms m = {n, left, right, low, segs, seg_at, seg_n};
    int plats[MAX_PLATFORMS], n_plats = 0;
    if (n > MAX_PLATFORMS)
        return -2;
    double fastest = fabs(vx), steer = arrow ? air_speed : 0.0;
    double reach = (steer > fastest ? steer : fastest) * limit + 1;  /* max(abs(vx), ...) */
    double top = y - (vy < 0 ? vy * vy / (2 * gravity) : 0.0) - 1;
    for (int q = 0; q < n; q++) {
        if (q != skip && right[q] >= x - reach && left[q] <= x + reach && low[q] >= top)
            plats[n_plats++] = q;
    }
    double x0 = x;
    int i = bisect_left(wall_xs, walls, x);
    int side = i < walls && wall_xs[i] == x ? side_at(&m, x, y) : 0;
    int x_same = 1, vx_same = 1, ignored = 1;  /* x and vx still the start's own (see above) */
    double t = 0.0;
    while (t < limit) {
        /* body.air_step */
        double nvy = decelerate(vy, fall, air_drag_fall, dt, &ignored);
        nvy = accelerate(nvy, gravity, fall, dt, &ignored);
        int nvx_same = vx_same, nx_same = 0;
        double nvx = arrow ? accelerate(vx, arrow * air_acc, air_speed, dt, &nvx_same)
                           : decelerate(vx, 0.0, nvy < fall ? air_drag : air_drag_fall, dt, &nvx_same);
        double dx = (vx + nvx) / 2 * dt, dy = (vy + nvy) / 2 * dt;
        double nx = x + dx, ny = y + dy;
        if (nx != x) {
            double lo = nx > x ? x : nx, hi = nx > x ? nx : x;
            for (i = bisect_left(wall_xs, walls, lo); i < walls && wall_xs[i] <= hi; i++) {
                if (wall_lo[i] <= ny && ny <= wall_hi[i]
                    && !(side && wall_xs[i] == x && x == x0 && (nx - x) * side > 0)) {
                    nx = x;  /* bumped into a wall */
                    nx_same = x_same;
                    nvx = 0.0;
                    nvx_same = 0;
                    break;
                }
            }
        }
        if ((vy > 0 || nvy > 0) && t > 0 && ny >= y) {
            for (int k = 0; k < n_plats; k++) {
                double at;
                int how = lands_at(&m, plats[k], x, y, nx, ny, &at);
                if (how) {
                    out[0] = at;
                    out[1] = t + dt;
                    out[2] = nvx;
                    out[3] = (how == 2 && nx_same ? 1 : 0) + (nvx_same ? 2 : 0);
                    return plats[k];
                }
            }
        }
        x = nx;
        x_same = nx_same;
        y = ny;
        vx = nvx;
        vx_same = nvx_same;
        vy = nvy;
        t = t + dt;
    }
    out[0] = x;
    out[1] = t;
    out[2] = vx;
    out[3] = (x_same ? 1 : 0) + (vx_same ? 2 : 0);
    return -1;
}


/* --- the masked colour match (perception/match.py, masked_match and best_match) ----------------------------
 * OpenCV's matchTemplate(image, template, TM_CCOEFF_NORMED, mask) for uint8 pictures (1 or 3 channels) and a binary
 * mask, worked out exactly: integer sums, one rounding at the end. With n the mask's pixels and, per channel c, ST_c
 * and STT_c the template's sum and sum of squares over the mask (once a template: mm_prepare), at each place of the
 * image STI is the template's dot product with the image under it (every channel; the template is zero outside its
 * mask), S1_c the image's sum under the mask and S2 its sum of squares (every channel). Then
 *     N = n STI - sum_c ST_c S1_c,   A = sum_c (n STT_c - ST_c^2),   B = n S2 - sum_c S1_c^2   (all exact, int64)
 *     score = N / sqrt(A B)   (in double, then float), and -1 where A B is 0: a flat template or window, where
 *                             OpenCV gives NaN or inf (which masked_match makes -1)
 * OpenCV's formula (templmatch.cpp, matchTemplateMask: T' = M (T - the mean under M), each channel's own, the channels
 * summed), whose sums it works out in float32, through Fourier transforms; so its scores aren't these to the bit. On
 * the 1,293 masked matches of 60 frames of a crowd (2026-10-26 01:01) and 1,601 with the drops' (the planning step's
 * D:/MapleHelper-scratch/pconst/plan, bench_mmatch.py) OpenCV's were within 9.5e-7 of these in colour and 2.5e-5 on
 * brightness, the best place and the 0.55, 0.6 and 0.9 thresholds' decisions the same on every call; in windows a few
 * levels from flat its float32 sums lose more (up to 1.1e-5 and 1.0e-4 on the first map's pictures drawn on frames of
 * West Rocky Mountain II, at scores of 0.24-0.32: tests/test_native_match.py). A monster's re-find window took 0.45 ms
 * against OpenCV's 1.57 (a 102 x 97 window, a 56 x 52 picture), a step-2 check 0.05 against 0.58, a drop's strip 0.28
 * against 0.68 (D:/MapleHelper-scratch/pconst/pnative/cmp_calls.py: one thread, best of 5).
 * The dot products are worked out one of two ways, both exact integers, so the scores are the same to the bit:
 *  - place by place (`dots_avx2`): 16 int16 products at a time (AVX2's madd), 8 places at once sharing each load of the
 *    template, and only over each template row's 16-wide chunks the mask is in (its rows' `k0`, `k1`), where the CPU
 *    has AVX2 (checked at run time: the flags stay /fp:precise's, without /arch:AVX2, so the pieces above are compiled
 *    as ever); else in plain C (`dots_c`), the same integers. Its int32 lanes hold `flush` template rows of tw x cn
 *    products of at most 255^2 (each group then added in int64): every row at once for a 3-channel template up to
 *    about 105 x 105. This runs at the madd's own pace (0.42 ns a madd in registers, 0.49-0.56 with the loads, on
 *    CPUs 12-23 of the Ryzen 9 5900X, half of them busy: D:/MapleHelper-scratch/pfast/kernel/peak.c), so it can only be
 *    made cheaper by doing fewer: two result rows sharing each image load (7% faster, 2026-10-26) and a Cauchy-Schwarz
 *    bound on the rows not yet summed (on 150 of the crowd's re-find matches, numpy: 68% of the products left with the
 *    best known in advance; prune_sim.py) were measured and left out.
 *  - every place at once (`ntt_*`), where it's cheaper (`ntt_least`: a re-find's window): as the correlation of the
 *    window's channels with the template's, the products of their number-theoretic transforms (below). On the 10,698
 *    re-find calls of the crowd's 600 frames (2026-10-26 01:01 [700:1300], recorded from master's step with
 *    D:/MapleHelper-scratch/pfast/kernel/record_calls.py), through native.best_matches, ctypes and all (time_py.py: one
 *    thread, best of 3, two rounds alternated with master's DLL, CPUs 12-23 57-67% busy), a call took 0.31-0.32 ms
 *    against 0.61-0.62 (median), 0.60-0.66 against 1.89-1.94 (p90), 0.76-0.83 against 2.45-2.47 (p99), a frame's
 *    re-finds 6.2-6.5 ms against 17.3-17.6 (mean), every answer the same to the bit (README, A cheaper masked match).
 * The mask's sums come from each image row's prefix sums (each channel's apart), a run of mask pixels at a time (`runs`:
 * row, from, to), 24 places at once in AVX2's registers. The image, those sums and the transforms' planes go into
 * memory of the calling thread's own, kept from call to call (`scratch`): no allocation per match. */

#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#define MM_AVX2 1
#define THREAD_LOCAL __declspec(thread)
#else
#define MM_AVX2 0
#define THREAD_LOCAL _Thread_local
#endif

#define INT32_LIMIT 2147483647LL
#define SQ255 65025LL

typedef struct {
    int th, tw, cn;      /* the template's size and channels */
    int tpad;            /* a template row's int16s, padded to a multiple of 16 */
    int nruns;           /* the mask's runs (row, from, to: x from <= x < to) */
    int flush;           /* template rows summed in the int32 lanes before going to int64 */
    int wide;            /* 1: the image's sum of squares under the mask needs int64 */
    int t_off;           /* where the template's rows are, in bytes from the start of this */
    long long spec_off;  /* where its transform is (cn planes of NTT_PLANE values), or 0: none */
    long long chunks;    /* the 16-wide chunks its rows' dot products take a place (dots_avx2's work) */
    long long n, A;      /* the mask's pixels; sum_c (n STT_c - ST_c^2) */
    long long ST[3];
    long long STT[3];
    /* then: runs (3 ints each), k0 and k1 (an int each a row), the template's rows at t_off (int16, tpad each), its
     * transform at spec_off (64-byte aligned) */
} Masked;

static int *masked_runs(const Masked *p) { return (int *)(p + 1); }
static int *masked_k0(const Masked *p) { return masked_runs(p) + 3 * (long long)p->th * (p->tw / 2 + 1); }
static int *masked_k1(const Masked *p) { return masked_k0(p) + p->th; }
static const short *masked_rows(const Masked *p) { return (const short *)((const char *)p + p->t_off); }

static int simd = -1;  /* -1: not checked yet; 0: plain C; 1: AVX2 (mm_simd) */

static int has_avx2(void)
{
#if MM_AVX2
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7)
        return 0;
    __cpuid(info, 1);
    if (!(info[2] & (1 << 27)) || !(info[2] & (1 << 28)))  /* OSXSAVE, AVX */
        return 0;
    if ((_xgetbv(0) & 6) != 6)  /* the OS saves the YMM registers */
        return 0;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;  /* AVX2 */
#else
    return 0;
#endif
}

/* Whether the dot product runs on AVX2: `use` 1 asks for it (where the CPU has it), 0 for plain C (the tests compare
 * the two), -1 leaves it. Returns what's in use. */
EXPORT int mm_simd(int use)
{
    if (simd < 0 || use >= 0)
        simd = use != 0 && has_avx2();
    return simd;
}

/* --- the dot products by number-theoretic transform (AVX2) ---
 * A window's dot products with a template at every place at once: the correlation of the window's channels with the
 * template's, as the products of their transforms modulo the prime NTT_P (119 * 2^23 + 1: its multiplicative group has
 * the 2^23rd roots of unity), on NTT_N x NTT_N planes, each channel zero-padded, the template's picture its cyclic
 * reversal (so a place's products never wrap round: y + r < ih <= NTT_N). All in integers, so the sums are exact: a dot
 * product is of non-negative values, at most 255 x the template's sum over its mask and channels, which mm_prepare
 * checks is under NTT_P (else the template has no transform and is matched place by place). The forward transform
 * decimates in frequency (natural order in, bit-reversed out), the inverse in time (bit-reversed in, natural out), so
 * there's no reordering pass: the products are taken in the bit-reversed order both sides share. Each direction goes
 * down a plane's columns, 8 at a time in AVX2's lanes, with a transpose between the two. Values are kept in
 * [0, NTT_P); a product is Montgomery's (R = 2^32), the twiddles and the template's transform in Montgomery form, the
 * template's scaled by NTT_N^-2 (the inverse's factor) as mm_prepare makes it. The window's transform is made once a
 * call, for all its templates; each template then takes its products and one inverse, only the columns of its places.
 * On the crowd's re-finds (above; pfast/kernel/bench.c with PROF, before the scores went 4 at a time): the window's
 * transform 0.39 M cycles a call, a template's products and inverse 0.13 M, its sums and scores 0.24 M. */

#define NTT_LOG 7
#define NTT_N (1 << NTT_LOG)          /* a plane's side: windows up to this each way */
#define NTT_PLANE (NTT_N * NTT_N)     /* its values (unsigned 32-bit) */
#define NTT_G (NTT_N / 8)             /* its column groups of 8 */
#define NTT_P 998244353u              /* 119 * 2^23 + 1; 3 generates its group */
/* A template's dot products go by transform where its places x its 16-wide chunks (dots_avx2's work) come to this
 * many. On the crowd's 10,698 re-find calls, their windows as recorded and cropped by 8, 16 and 20 px each side (42,530
 * calls; pfast/kernel/bench2.c, both ways on each, best of 2): at a work of 200,000-300,000 the transform was faster
 * in 24% of the calls, at 300,000-400,000 in 97%, past 600,000 in all; with the windows as they are (REFIND_PAD 16:
 * 98 x 102, a 52 x 56 picture: 2,209 places, 413 chunks) 0.25 ms a call against 0.46 (median). */
#define NTT_LEAST 300000LL

static long long ntt_least = NTT_LEAST;           /* mm_ntt */
static unsigned ntt_pinv;                         /* -NTT_P^-1 mod 2^32 */
static unsigned ntt_fw[NTT_N], ntt_iw[NTT_N];     /* [half + i]: w^i, w^-i, w of order 2 half, in Montgomery form */
static unsigned ntt_scale;                        /* NTT_N^-2 R^2 mod p: a template's transform into form, scaled */
static volatile long ntt_ready;

/* The least work (a template's places x its chunks) whose dot products go by transform: 0 wherever it can, a huge
 * number never (the tests compare the two ways), -1 leaves it. Returns what's in use. */
EXPORT long long mm_ntt(long long least)
{
    if (least >= 0)
        ntt_least = least;
    return ntt_least;
}

/* Whether template `tpl` (mm_prepare's) has its transform (the tests'). */
EXPORT int mm_transformed(const void *tpl) { return ((const Masked *)tpl)->spec_off != 0; }

static unsigned ntt_mul(unsigned a, unsigned b) { return (unsigned)((unsigned long long)a * b % NTT_P); }

static unsigned ntt_pow(unsigned b, unsigned e)
{
    unsigned r = 1;
    for (; e; e >>= 1, b = ntt_mul(b, b))
        if (e & 1)
            r = ntt_mul(r, b);
    return r;
}

static unsigned ntt_form(unsigned a) { return (unsigned)(((unsigned long long)a << 32) % NTT_P); }  /* a R mod p */

/* The tables, once (threads that race here write the same values, and x86 keeps the stores in order). */
static void ntt_init(void)
{
    if (ntt_ready)
        return;
    unsigned inv = NTT_P;  /* NTT_P^-1 mod 2^32 by Newton's steps (3 bits right to start, twice as many each) */
    for (int i = 0; i < 5; i++)
        inv *= 2u - NTT_P * inv;
    ntt_pinv = 0u - inv;
    for (int half = 1; half < NTT_N; half <<= 1) {
        unsigned w = ntt_pow(3, (NTT_P - 1) / (2 * half)), wi = ntt_pow(w, NTT_P - 2), a = 1, b = 1;
        for (int i = 0; i < half; i++) {
            ntt_fw[half + i] = ntt_form(a);
            ntt_iw[half + i] = ntt_form(b);
            a = ntt_mul(a, w);
            b = ntt_mul(b, wi);
        }
    }
    ntt_scale = ntt_form(ntt_form(ntt_pow(ntt_mul(NTT_N, NTT_N), NTT_P - 2)));
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
    ntt_ready = 1;
}

#if MM_AVX2
/* a b R^-1 mod p in 8 lanes, for a b < 2^32 p (`bodd`: b's odd lanes in the even ones; b itself if all lanes alike). */
static __forceinline __m256i ntt_mont(__m256i a, __m256i b, __m256i bodd, __m256i P, __m256i PINV)
{
    __m256i t0 = _mm256_mul_epu32(a, b);
    __m256i t1 = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), bodd);
    __m256i u0 = _mm256_add_epi64(t0, _mm256_mul_epu32(_mm256_mul_epu32(t0, PINV), P));
    __m256i u1 = _mm256_add_epi64(t1, _mm256_mul_epu32(_mm256_mul_epu32(t1, PINV), P));
    __m256i r = _mm256_blend_epi32(_mm256_srli_epi64(u0, 32), u1, 0xAA);  /* in [0, 2p) */
    return _mm256_min_epu32(r, _mm256_sub_epi32(r, P));
}

static __forceinline __m256i ntt_less(__m256i s, __m256i P) { return _mm256_min_epu32(s, _mm256_sub_epi32(s, P)); }

#define NTT_LD(q) _mm256_loadu_si256((const __m256i *)(q))
#define NTT_ST(q, v) _mm256_storeu_si256((__m256i *)(q), v)

/* The forward transform down the columns of plane X (rows NTT_N apart) of column groups [g0, g1). A butterfly whose
 * twiddle is 1 (the first of each block: 127 of 448) takes no product. */
static void ntt_cols(unsigned *X, int g0, int g1)
{
    const __m256i P = _mm256_set1_epi32((int)NTT_P), PINV = _mm256_set1_epi32((int)ntt_pinv);
    for (int half = NTT_N / 2; half >= 1; half >>= 1)
        for (int j = 0; j < NTT_N; j += 2 * half)
            for (int i = 0; i < half; i++) {
                __m256i w = _mm256_set1_epi32((int)ntt_fw[half + i]);
                unsigned *a = X + (long long)(j + i) * NTT_N, *b = a + (long long)half * NTT_N;
                for (int g = g0; g < g1; g++) {
                    __m256i u = NTT_LD(a + 8 * g), v = NTT_LD(b + 8 * g);
                    __m256i d = _mm256_add_epi32(_mm256_sub_epi32(u, v), P);  /* in (0, 2p) */
                    NTT_ST(a + 8 * g, ntt_less(_mm256_add_epi32(u, v), P));
                    NTT_ST(b + 8 * g, i ? ntt_mont(d, w, w, P, PINV) : ntt_less(d, P));
                }
            }
}

/* The inverse (unscaled) down the columns of X of column groups [g0, g1). */
static void ntt_icols(unsigned *X, int g0, int g1)
{
    const __m256i P = _mm256_set1_epi32((int)NTT_P), PINV = _mm256_set1_epi32((int)ntt_pinv);
    for (int half = 1; half < NTT_N; half <<= 1)
        for (int j = 0; j < NTT_N; j += 2 * half)
            for (int i = 0; i < half; i++) {
                __m256i w = _mm256_set1_epi32((int)ntt_iw[half + i]);
                unsigned *a = X + (long long)(j + i) * NTT_N, *b = a + (long long)half * NTT_N;
                for (int g = g0; g < g1; g++) {
                    __m256i u = NTT_LD(a + 8 * g), v = NTT_LD(b + 8 * g);
                    if (i)
                        v = ntt_mont(v, w, w, P, PINV);
                    NTT_ST(a + 8 * g, ntt_less(_mm256_add_epi32(u, v), P));
                    NTT_ST(b + 8 * g, ntt_less(_mm256_add_epi32(_mm256_sub_epi32(u, v), P), P));
                }
            }
}

/* Rows [8 r0, 8 r1) of plane S into columns [8 r0, 8 r1) of plane D, 8 x 8 at a time. */
static void ntt_transpose(const unsigned *S, unsigned *D, int r0, int r1)
{
    for (int bi = r0; bi < r1; bi++)
        for (int bj = 0; bj < NTT_G; bj++) {
            const unsigned *s = S + (long long)8 * bi * NTT_N + 8 * bj;
            __m256i r[8], t[8], u[8];
            for (int k = 0; k < 8; k++)
                r[k] = NTT_LD(s + (long long)k * NTT_N);
            for (int k = 0; k < 8; k += 2) {
                t[k] = _mm256_unpacklo_epi32(r[k], r[k + 1]);
                t[k + 1] = _mm256_unpackhi_epi32(r[k], r[k + 1]);
            }
            for (int k = 0; k < 8; k += 4) {
                u[k] = _mm256_unpacklo_epi64(t[k], t[k + 2]);
                u[k + 1] = _mm256_unpackhi_epi64(t[k], t[k + 2]);
                u[k + 2] = _mm256_unpacklo_epi64(t[k + 1], t[k + 3]);
                u[k + 3] = _mm256_unpackhi_epi64(t[k + 1], t[k + 3]);
            }
            unsigned *d = D + (long long)8 * bj * NTT_N + 8 * bi;
            for (int k = 0; k < 4; k++) {
                NTT_ST(d + (long long)k * NTT_N, _mm256_permute2x128_si256(u[k], u[k + 4], 0x20));
                NTT_ST(d + (long long)(k + 4) * NTT_N, _mm256_permute2x128_si256(u[k], u[k + 4], 0x31));
            }
        }
}

/* Plane X's 2-D transform into W (X is used up): down its columns of groups [0, g) (the rest are zero), then across. */
static void ntt_forward(unsigned *X, unsigned *W, int g)
{
    ntt_cols(X, 0, g);
    ntt_transpose(X, W, 0, NTT_G);
    ntt_cols(W, 0, NTT_G);
}

/* ntt_forward's undone from Z into W (Z is used up), only W's columns of groups [0, g). */
static void ntt_inverse(unsigned *Z, unsigned *W, int g)
{
    ntt_icols(Z, 0, NTT_G);
    ntt_transpose(Z, W, 0, g);
    ntt_icols(W, 0, g);
}

/* Z = sum over the cn channels of F_c G_c R^-1 mod p (each a plane, one after another). */
static void ntt_products(const unsigned *F, const unsigned *G, int cn, unsigned *Z)
{
    const __m256i P = _mm256_set1_epi32((int)NTT_P), PINV = _mm256_set1_epi32((int)ntt_pinv);
    for (long long k = 0; k < NTT_PLANE; k += 8) {
        __m256i acc = _mm256_setzero_si256();
        for (int c = 0; c < cn; c++) {
            __m256i f = NTT_LD(F + c * (long long)NTT_PLANE + k), g = NTT_LD(G + c * (long long)NTT_PLANE + k);
            acc = ntt_less(_mm256_add_epi32(acc, ntt_mont(f, g, _mm256_srli_epi64(g, 32), P, PINV)), P);
        }
        NTT_ST(Z + k, acc);
    }
}

/* Plane V times NTT_N^-2 R: a template's transform scaled, in Montgomery form. */
static void ntt_scaled(unsigned *V)
{
    const __m256i P = _mm256_set1_epi32((int)NTT_P), PINV = _mm256_set1_epi32((int)ntt_pinv);
    const __m256i s = _mm256_set1_epi32((int)ntt_scale);
    for (long long k = 0; k < NTT_PLANE; k += 8)
        NTT_ST(V + k, ntt_mont(NTT_LD(V + k), s, s, P, PINV));
}
#endif

/* The bytes mm_prepare needs for a th x tw template of cn channels (with room for its transform if it fits a plane). */
EXPORT long long mm_prepare_size(int th, int tw, int cn)
{
    long long tpad = ((long long)tw * cn + 15) / 16 * 16;
    long long ints = 3LL * th * (tw / 2 + 1) + 2LL * th;
    long long spec = th <= NTT_N && tw <= NTT_N ? 4LL * cn * NTT_PLANE + 64 : 0;
    return (long long)sizeof(Masked) + 4 * ints + 32 + 2 * tpad * th + spec;
}

/* A template made ready for mm_match and mm_best, into `buf` (`size` bytes: mm_prepare_size): its colours `tpl`
 * (pixel (y, x) channel c at tpl[y * trow + x * cn + c]) and its mask (mask[y * mrow + x * mpix], nonzero: in); and,
 * where the CPU has AVX2 and the template fits a plane with its dot products under NTT_P, its transform (3 transforms
 * of a plane, 0.1 ms). `buf` mustn't move afterwards (the transform is 64-byte aligned where it lies). Returns 0, or -1
 * for what it doesn't take (another channel count; a template row too wide for the int32 lanes), -2 if `size` is too
 * small. */
EXPORT int mm_prepare(const unsigned char *tpl, long long trow, const unsigned char *mask, long long mrow, int mpix,
                      int th, int tw, int cn, void *buf, long long size)
{
    if ((cn != 1 && cn != 3) || th <= 0 || tw <= 0 || (long long)tw * cn * SQ255 > INT32_LIMIT)
        return -1;
    if (size < mm_prepare_size(th, tw, cn))
        return -2;
    Masked *p = (Masked *)buf;
    memset(p, 0, sizeof(Masked));
    p->th = th;
    p->tw = tw;
    p->cn = cn;
    p->tpad = (tw * cn + 15) / 16 * 16;
    int *runs = masked_runs(p), *k0 = masked_k0(p), *k1 = masked_k1(p);
    long long t_off = (long long)((char *)(k1 + th) - (char *)p);
    p->t_off = (int)((t_off + 31) / 32 * 32);
    short *T = (short *)((char *)p + p->t_off);
    int nruns = 0;
    for (int y = 0; y < th; y++) {
        short *row = T + (long long)y * p->tpad;
        memset(row, 0, sizeof(short) * p->tpad);
        int in = 0, a = 0, lo = -1, hi = -1;
        for (int x = 0; x <= tw; x++) {
            int m = x < tw && mask[y * mrow + (long long)x * mpix] != 0;
            if (m) {
                p->n++;
                for (int c = 0; c < cn; c++) {
                    int v = tpl[y * trow + (long long)x * cn + c];
                    row[x * cn + c] = (short)v;
                    p->ST[c] += v;
                    p->STT[c] += (long long)v * v;
                }
                if (!in) {
                    in = 1;
                    a = x;
                }
            } else if (in) {
                in = 0;
                runs[3 * nruns] = y;
                runs[3 * nruns + 1] = a;
                runs[3 * nruns + 2] = x;
                nruns++;
                if (lo < 0)
                    lo = a;
                hi = x;
            }
        }
        k0[y] = lo < 0 ? 0 : lo * cn / 16;          /* the 16-wide chunks of the row the mask is in */
        k1[y] = lo < 0 ? 0 : (hi * cn + 15) / 16;
        p->chunks += k1[y] - k0[y];
    }
    p->nruns = nruns;
    for (int c = 0; c < cn; c++)
        p->A += p->n * p->STT[c] - p->ST[c] * p->ST[c];
    long long per_row = (long long)tw * cn * SQ255;
    long long rows = INT32_LIMIT / per_row;
    p->flush = rows >= th ? th : (int)rows;
    p->wide = p->n * cn * SQ255 > INT32_LIMIT;
    p->spec_off = 0;
#if MM_AVX2
    long long most = 0;  /* the most a dot product can be: 255 x the template's sum */
    for (int c = 0; c < cn; c++)
        most += 255 * p->ST[c];
    if (th <= NTT_N && tw <= NTT_N && most < NTT_P && has_avx2()) {
        unsigned *X = (unsigned *)malloc(4 * (size_t)NTT_PLANE);
        if (X) {
            ntt_init();
            long long at = (long long)p->t_off + 2LL * p->tpad * th;
            unsigned char *base = (unsigned char *)p + at;
            long long skip = (long long)((64 - (size_t)base % 64) % 64);
            unsigned *G = (unsigned *)(base + skip);
            for (int c = 0; c < cn; c++) {
                memset(X, 0, 4 * (size_t)NTT_PLANE);
                for (int y = 0; y < th; y++)  /* the picture's cyclic reversal: (y, x) at (-y, -x) */
                    for (int x = 0; x < tw; x++)
                        X[((NTT_N - y) & (NTT_N - 1)) * NTT_N + ((NTT_N - x) & (NTT_N - 1))] =
                            (unsigned)T[(long long)y * p->tpad + x * cn + c];
                ntt_forward(X, G + (long long)c * NTT_PLANE, NTT_G);
                ntt_scaled(G + (long long)c * NTT_PLANE);
            }
            free(X);
            p->spec_off = at + skip;
        }
    }
#endif
    return 0;
}

/* --- the calling thread's scratch memory (freed as the thread ends: DllMain) */
static THREAD_LOCAL unsigned char *scratch_at;
static THREAD_LOCAL size_t scratch_size;

static unsigned char *scratch(size_t need)
{
    if (need > scratch_size) {
        free(scratch_at);
        need += need / 4;  /* (room to grow: a few sizes come and go) */
        scratch_at = (unsigned char *)malloc(need);
        scratch_size = scratch_at ? need : 0;
    }
    return scratch_at;
}

#ifdef _WIN32
int __stdcall DllMain(void *instance, unsigned long reason, void *reserved)
{
    (void)instance;
    (void)reserved;
    if (reason == 3 || reason == 0) {  /* DLL_THREAD_DETACH, DLL_PROCESS_DETACH: the thread's scratch */
        free(scratch_at);
        scratch_at = 0;
        scratch_size = 0;
    }
    return 1;
}
#endif

#define BLOCK 8      /* places a dot-product pass does at once */
#define LANES 24     /* places the mask's sums do at once (3 AVX2 registers), and the room past a row they read */
#define MAX_RUNS 512 /* runs the AVX2 sums take (more: in plain C) */

typedef struct {
    int ih, iw, cn, ipad;
    const unsigned char *img;
    long long irow;
    short *I;           /* the image, int16, rows ipad apart, zeros past each row's end */
    int *P1;            /* each row's prefix sums of each channel: P1[(c * ih + row) * (iw + 1) + x] */
    int *P2;            /* ... and of the squares, the channels summed: P2[row * (iw + 1) + x] */
    int *S1;            /* a result row's sums under the mask: S1[c * s1s + x] */
    int s1s;
    long long *S2;      /* ... of the squares (int32 in its first half unless the template is `wide`) */
    long long *DOT;     /* ... its dot products */
    float *out;         /* a result, for mm_best */
    int ntt, spec;      /* room for the transforms; the image's (F) made */
    unsigned *F;        /* the image's transform: a plane a channel */
    unsigned *X, *Z, *W;/* planes to work in */
} Picture;

static size_t align64(size_t n) { return (n + 63) / 64 * 64; }

/* The image (pixel (y, x) channel c at img[y * irow + x * cn + c]) loaded into the thread's scratch: its rows as
 * int16 and their prefix sums; with `ntt`, room for its transform and the planes to work in. 0, or -1 if it can't be
 * (a row too wide for the int32 sums, no memory). */
static int load(Picture *im, const unsigned char *img, int ih, int iw, long long irow, int cn, int ntt)
{
    if ((long long)iw * cn * SQ255 > INT32_LIMIT || ih <= 0 || iw <= 0)
        return -1;
    int ipad = iw * cn + BLOCK * cn + 32, s1s = iw + LANES;
    size_t sI = align64(sizeof(short) * (size_t)ipad * ih);
    size_t sP1 = align64(sizeof(int) * ((size_t)(iw + 1) * cn * ih + LANES));
    size_t sP2 = align64(sizeof(int) * ((size_t)(iw + 1) * ih + LANES)), sS1 = align64(sizeof(int) * (size_t)s1s * cn);
    size_t sS2 = align64(sizeof(long long) * (size_t)(iw + LANES)), sOut = align64(sizeof(float) * (size_t)iw * ih);
    size_t sPl = ntt ? 4 * (size_t)NTT_PLANE * (cn + 3) + 64 : 0;
    unsigned char *s = scratch(sI + sP1 + sP2 + sS1 + 2 * sS2 + sOut + sPl);
    if (!s)
        return -1;
    im->ih = ih;
    im->iw = iw;
    im->cn = cn;
    im->ipad = ipad;
    im->img = img;
    im->irow = irow;
    im->I = (short *)s;
    im->P1 = (int *)(s + sI);
    im->P2 = (int *)(s + sI + sP1);
    im->S1 = (int *)(s + sI + sP1 + sP2);
    im->s1s = s1s;
    im->S2 = (long long *)(s + sI + sP1 + sP2 + sS1);
    im->DOT = (long long *)(s + sI + sP1 + sP2 + sS1 + sS2);
    im->out = (float *)(s + sI + sP1 + sP2 + sS1 + 2 * sS2);
    im->ntt = ntt;
    im->spec = 0;
    if (ntt) {
        unsigned char *pl = s + sI + sP1 + sP2 + sS1 + 2 * sS2 + sOut;
        im->F = (unsigned *)(pl + (64 - (size_t)pl % 64) % 64);
        im->X = im->F + (size_t)NTT_PLANE * cn;
        im->Z = im->X + NTT_PLANE;
        im->W = im->Z + NTT_PLANE;
    }
    int irow_n = iw * cn;
    long long cs = (long long)ih * (iw + 1);  /* a channel's prefix sums */
    for (int y = 0; y < ih; y++) {
        const unsigned char *src = img + y * irow;
        short *dst = im->I + (long long)y * ipad;
        for (int k = 0; k < irow_n; k++)
            dst[k] = src[k];
        for (int k = irow_n; k < ipad; k++)
            dst[k] = 0;
        int *p1 = im->P1 + (long long)y * (iw + 1), *p2 = im->P2 + (long long)y * (iw + 1);
        for (int c = 0; c < cn; c++)
            p1[c * cs] = 0;
        p2[0] = 0;
        for (int x = 0; x < iw; x++) {
            int sq = 0;
            for (int c = 0; c < cn; c++) {
                int v = src[x * cn + c];
                p1[c * cs + x + 1] = p1[c * cs + x] + v;
                sq += v * v;
            }
            p2[x + 1] = p2[x] + sq;
        }
    }
    return 0;
}

#if MM_AVX2
static long long hsum_epi32(__m256i v)
{
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return (long long)_mm_cvtsi128_si32(s);
}

/* One result row's dot products, BLOCK places at a time (DOT has room for the last block's places past rw). */
static void dots_avx2(const Picture *im, const Masked *p, int y, int rw, long long *DOT)
{
    const short *T = masked_rows(p);
    const int *k0 = masked_k0(p), *k1 = masked_k1(p);
    int cn = p->cn, th = p->th;
    for (int x = 0; x < rw; x += BLOCK) {
        long long d[BLOCK] = {0};
        for (int r0 = 0; r0 < th; r0 += p->flush) {
            int r1 = r0 + p->flush < th ? r0 + p->flush : th;
            __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
            for (int r = r0; r < r1; r++) {
                const short *ip = im->I + (long long)(y + r) * im->ipad + (long long)x * cn;
                const short *tp = T + (long long)r * p->tpad;
                for (int k = k0[r]; k < k1[r]; k++) {
                    __m256i t = _mm256_loadu_si256((const __m256i *)(tp + 16 * k));
                    const short *q = ip + 16 * k;
#define STEP(aj, j) aj = _mm256_add_epi32(aj, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(q + (j) * cn)), t))
                    STEP(a0, 0); STEP(a1, 1); STEP(a2, 2); STEP(a3, 3);
                    STEP(a4, 4); STEP(a5, 5); STEP(a6, 6); STEP(a7, 7);
#undef STEP
                }
            }
            d[0] += hsum_epi32(a0); d[1] += hsum_epi32(a1); d[2] += hsum_epi32(a2); d[3] += hsum_epi32(a3);
            d[4] += hsum_epi32(a4); d[5] += hsum_epi32(a5); d[6] += hsum_epi32(a6); d[7] += hsum_epi32(a7);
        }
        for (int j = 0; j < BLOCK; j++)
            DOT[x + j] = d[j];
    }
    _mm256_zeroupper();
}
#endif

/* The same in plain C. */
static void dots_c(const Picture *im, const Masked *p, int y, int rw, long long *DOT)
{
    const short *T = masked_rows(p);
    const int *k0 = masked_k0(p), *k1 = masked_k1(p);
    int cn = p->cn;
    for (int x = 0; x < rw; x++) {
        long long d = 0;
        for (int r = 0; r < p->th; r++) {
            const short *ip = im->I + (long long)(y + r) * im->ipad + (long long)x * cn;
            const short *tp = T + (long long)r * p->tpad;
            for (int k = 16 * k0[r]; k < 16 * k1[r]; k++)
                d += (long long)ip[k] * tp[k];
        }
        DOT[x] = d;
    }
}

/* Result row y's sums under the mask: S1 (each channel's) and S2 (int32, or int64 for a `wide` template). */
static void sums(const Picture *im, const Masked *p, int y, int rw)
{
    int cn = p->cn, iw = im->iw, s1s = im->s1s;
    long long cs = (long long)im->ih * (iw + 1);
    const int *runs = masked_runs(p);
    int *S1 = im->S1;
    long long *S2 = im->S2;
    int *S2i = (int *)S2;
#if MM_AVX2
    if (simd && !p->wide && p->nruns <= MAX_RUNS) {  /* LANES sums at a time in registers, over all the runs */
        const int *pa[MAX_RUNS], *pb[MAX_RUNS], *qa[MAX_RUNS], *qb[MAX_RUNS];
        int nr = p->nruns;
        for (int q = 0; q < nr; q++) {
            int ry = runs[3 * q], a = runs[3 * q + 1], b = runs[3 * q + 2];
            const int *p1 = im->P1 + (long long)(y + ry) * (iw + 1), *p2 = im->P2 + (long long)(y + ry) * (iw + 1);
            pa[q] = p1 + a;
            pb[q] = p1 + b;
            qa[q] = p2 + a;
            qb[q] = p2 + b;
        }
#define DIFF(u, v, k) _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)((u) + (k))), \
                                       _mm256_loadu_si256((const __m256i *)((v) + (k))))
#define RUNS(hi, lo, off, to) \
        for (int x = 0; x < rw; x += LANES) { \
            __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0; \
            for (int q = 0; q < nr; q++) { \
                a0 = _mm256_add_epi32(a0, DIFF(hi[q] + (off), lo[q] + (off), x)); \
                a1 = _mm256_add_epi32(a1, DIFF(hi[q] + (off), lo[q] + (off), x + 8)); \
                a2 = _mm256_add_epi32(a2, DIFF(hi[q] + (off), lo[q] + (off), x + 16)); \
            } \
            _mm256_storeu_si256((__m256i *)((to) + x), a0); \
            _mm256_storeu_si256((__m256i *)((to) + x + 8), a1); \
            _mm256_storeu_si256((__m256i *)((to) + x + 16), a2); \
        }
        for (int c = 0; c < cn; c++)
            RUNS(pb, pa, c * cs, S1 + (long long)c * s1s)
        RUNS(qb, qa, 0, S2i)
#undef RUNS
#undef DIFF
        _mm256_zeroupper();
        return;
    }
#endif
    for (int c = 0; c < cn; c++)
        memset(S1 + (long long)c * s1s, 0, sizeof(int) * (size_t)rw);
    memset(S2, 0, sizeof(long long) * (size_t)rw);
    for (int q = 0; q < p->nruns; q++) {
        int ry = runs[3 * q], a = runs[3 * q + 1], b = runs[3 * q + 2];
        for (int c = 0; c < cn; c++) {
            const int *p1 = im->P1 + c * cs + (long long)(y + ry) * (iw + 1);
            int *s1 = S1 + (long long)c * s1s;
            for (int x = 0; x < rw; x++)
                s1[x] += p1[b + x] - p1[a + x];
        }
        const int *p2 = im->P2 + (long long)(y + ry) * (iw + 1);
        const int *qa = p2 + a, *qb = p2 + b;
        if (p->wide) {
            for (int x = 0; x < rw; x++)
                S2[x] += qb[x] - qa[x];
        } else {
            for (int x = 0; x < rw; x++)
                S2i[x] += qb[x] - qa[x];
        }
    }
}

/* Place x's score from its dot product and the row's sums (sums'). */
static float score_at(const Picture *im, const Masked *p, int x, long long dot)
{
    long long N = p->n * dot, B = p->n * (p->wide ? im->S2[x] : (long long)((const int *)im->S2)[x]);
    for (int c = 0; c < p->cn; c++) {
        long long s1 = im->S1[(long long)c * im->s1s + x];
        N -= p->ST[c] * s1;
        B -= s1 * s1;
    }
    double den = (double)p->A * (double)B;
    return den > 0 ? (float)((double)N / sqrt(den)) : -1.0f;
}

/* Result row y's scores into `o` from its dot products: DOT (int64), or D32 (the transform's, which are under NTT_P).
 * With D32 and a template that isn't `wide`, 4 places at a time in double, the same arithmetic to the bit: N, B and A
 * are integers under 2^53 there (n <= NTT_N^2, a dot product under NTT_P, each S1_c at most 255 n and S2 under 2^31,
 * A at most n x 255 x the template's sum < n x 255 x NTT_P / 255), so each is the int64's exactly, and A B, the square
 * root, the quotient and the float are each rounded once, as score_at's (IEEE, round to nearest). */
static void row_scores(const Picture *im, const Masked *p, int y, int rw, const long long *DOT,
                       const unsigned *D32, float *o)
{
    sums(im, p, y, rw);
    int x = 0;
#if MM_AVX2
    if (D32 && simd && !p->wide) {
        const int *S2i = (const int *)im->S2;
        __m256d n = _mm256_set1_pd((double)p->n), A = _mm256_set1_pd((double)p->A), none = _mm256_set1_pd(-1.0);
        __m256d zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1.0);
        for (; x + 4 <= rw; x += 4) {
            __m256d dot = _mm256_cvtepi32_pd(_mm_loadu_si128((const __m128i *)(D32 + x)));  /* (under 2^31) */
            __m256d N = _mm256_mul_pd(n, dot);
            __m256d B = _mm256_mul_pd(n, _mm256_cvtepi32_pd(_mm_loadu_si128((const __m128i *)(S2i + x))));
            for (int c = 0; c < p->cn; c++) {
                __m256d s1 = _mm256_cvtepi32_pd(_mm_loadu_si128((const __m128i *)(im->S1 + (long long)c * im->s1s + x)));
                N = _mm256_sub_pd(N, _mm256_mul_pd(_mm256_set1_pd((double)p->ST[c]), s1));
                B = _mm256_sub_pd(B, _mm256_mul_pd(s1, s1));
            }
            __m256d den = _mm256_mul_pd(A, B), some = _mm256_cmp_pd(den, zero, _CMP_GT_OQ);
            __m256d r = _mm256_div_pd(N, _mm256_sqrt_pd(_mm256_blendv_pd(one, den, some)));  /* (no 0/0 where it's -1) */
            r = _mm256_blendv_pd(none, r, some);
            _mm_storeu_ps(o + x, _mm256_cvtpd_ps(r));
        }
        _mm256_zeroupper();
    }
#endif
    for (; x < rw; x++)
        o[x] = score_at(im, p, x, D32 ? (long long)D32[x] : DOT[x]);
}

/* The scores of template `p` at every place of the loaded image, into `out` (rh x rw): by transform where it's cheaper
 * (ntt_least) and there's room for it, else place by place. */
static void match_loaded(Picture *im, const Masked *p, float *out)
{
    int rh = im->ih - p->th + 1, rw = im->iw - p->tw + 1;
#if MM_AVX2
    if (simd && im->ntt && p->spec_off && (long long)rh * rw * p->chunks >= ntt_least) {
        int cn = im->cn;
        if (!im->spec) {  /* the window's transform, once for the call's templates */
            for (int c = 0; c < cn; c++) {
                memset(im->X, 0, 4 * (size_t)NTT_PLANE);
                for (int y = 0; y < im->ih; y++) {
                    const unsigned char *src = im->img + y * im->irow + c;
                    unsigned *dst = im->X + (long long)y * NTT_N;
                    for (int x = 0; x < im->iw; x++)
                        dst[x] = src[x * cn];
                }
                ntt_forward(im->X, im->F + (long long)c * NTT_PLANE, (im->iw + 7) / 8);
            }
            im->spec = 1;
        }
        ntt_products(im->F, (const unsigned *)((const char *)p + p->spec_off), cn, im->Z);
        ntt_inverse(im->Z, im->W, (rw + 7) / 8);
        for (int y = 0; y < rh; y++)
            row_scores(im, p, y, rw, 0, im->W + (long long)y * NTT_N, out + (long long)y * rw);
        return;
    }
#endif
    for (int y = 0; y < rh; y++) {
#if MM_AVX2
        if (simd)
            dots_avx2(im, p, y, rw, im->DOT);
        else
#endif
            dots_c(im, p, y, rw, im->DOT);
        row_scores(im, p, y, rw, im->DOT, 0, out + (long long)y * rw);
    }
}

/* Whether template p would go by transform over an ih x iw image (match_loaded's test, before the image is loaded). */
static int ntt_wanted(const Masked *p, int ih, int iw)
{
#if MM_AVX2
    return simd && ih <= NTT_N && iw <= NTT_N && p->spec_off &&
           (long long)(ih - p->th + 1) * (iw - p->tw + 1) * p->chunks >= ntt_least;
#else
    (void)p; (void)ih; (void)iw;
    return 0;
#endif
}

/* Whether any of the n templates would go by transform over an ih x iw image (load's room for it). */
static int wants_ntt(const void *const *tpls, int n, int ih, int iw)
{
    for (int i = 0; i < n; i++) {
        if (ntt_wanted((const Masked *)tpls[i], ih, iw))
            return 1;
    }
    return 0;
}

/* Template `tpl` (mm_prepare's) matched at every place of the image (pixel (y, x) channel c at img[y * irow + x * cn +
 * c], cn the template's): the scores into `out` ((ih - th + 1) x (iw - tw + 1) floats). 0, or -1 if it can't be (the
 * image smaller than the template, other channels, a row too wide). */
EXPORT int mm_match(const unsigned char *img, int ih, int iw, long long irow, int cn, const void *tpl, float *out)
{
    const Masked *p = (const Masked *)tpl;
    Picture im;
    if (simd < 0)
        mm_simd(1);
    if (cn != p->cn || ih < p->th || iw < p->tw || load(&im, img, ih, iw, irow, cn, wants_ntt(&tpl, 1, ih, iw)) != 0)
        return -1;
    match_loaded(&im, p, out);
    return 0;
}

/* Each of `n` templates in turn (mm_prepare's) matched over the image (as mm_match's), and its best score and place
 * as cv2.minMaxLoc gives them: the first highest in row-major order, into scores, xs and ys; after the first whose
 * best is `stop` or more, no more. Returns how many were matched, or -1 if one can't be (as mm_match). The image is
 * loaded once for them all, and its transform made once if any goes by transform. */
EXPORT int mm_best(const unsigned char *img, int ih, int iw, long long irow, int cn, const void *const *tpls, int n,
                   double stop, float *scores, int *xs, int *ys)
{
    Picture im;
    if (simd < 0)
        mm_simd(1);
    for (int i = 0; i < n; i++) {
        const Masked *p = (const Masked *)tpls[i];
        if (cn != p->cn || ih < p->th || iw < p->tw)
            return -1;
    }
    if (load(&im, img, ih, iw, irow, cn, wants_ntt(tpls, n, ih, iw)) != 0)
        return -1;
    for (int i = 0; i < n; i++) {
        const Masked *p = (const Masked *)tpls[i];
        int rh = ih - p->th + 1, rw = iw - p->tw + 1;
        match_loaded(&im, p, im.out);
        long long at = 0, size = (long long)rh * rw;
        float best = im.out[0];
        for (long long k = 1; k < size; k++) {
            if (im.out[k] > best) {
                best = im.out[k];
                at = k;
            }
        }
        scores[i] = best;
        xs[i] = (int)(at % rw);
        ys[i] = (int)(at / rw);
        if ((double)best >= stop)
            return i + 1;
    }
    return n;
}


/* --- a frame's re-finds in one call (perception/mobs.py, MobDetector.refind_all) --------------------------------------
 * The frame loop's re-finds (localization/worldstate.py: every monster followed, every frame), their masked matches all
 * in one call, on threads of its own, Python's lock let go once for them all: they were a Python thread's each, the
 * matches a call a picture list, with numpy's slices and copies and the pool's hand-offs between (README, One call for
 * the re-finds).
 *
 * THE CONTRACT: this CPU backend's, and any other's behind the same calls (a Direct3D 11 compute shader's is being built
 * apart, to slot in later). A backend gives these answers to the bit; how it gets them is its own.
 *   rf_open(tpls, n, threads): a context, or NULL if it can't make one. `tpls[0..n)`: the detector's pictures, each a
 *     template and its mask as mm_prepare made them ready (Masked), the same for the context's life; a picture is named
 *     in a job by its index there (a backend may copy them: a GPU's, uploaded once, here). `threads`: how many threads
 *     work at a call, the caller's own among them (a pool made here, once: none made a call).
 *   rf_run(ctx, frame, fh, fw, frow, cn, jobs, njobs, lists, thr, good, keep, outs): 0, or -1 if it did nothing (the
 *     caller then does it all its own way). The frame: pixel (y, x) channel c at frame[y * frow + x * cn + c], uint8,
 *     cn the pictures' channels. Each job (RfJob), a monster followed:
 *       its windows, x0, y0, x1, y1 in frame px, within the frame: `wnear` (empty, x1 <= x0 or y1 <= y0: none) and
 *       `wfull` (empty: nothing to do but keep its memo);
 *       its pictures, indices into tpls: `first_n` from lists[first_at] (the picture that matched it last; both ways
 *       for a cut-out that doesn't know its facing) and `next_n` from lists[next_at] (the frames its animation most
 *       likely went to: mobs._next_frames; none where the look frame is step 1's own).
 *     In a window W looked at, its best is what mobs.MobDetector._matched finds there:
 *       a picture is matched only if it fits W (its height and width at most W's); its best is the first highest
 *       score of the masked match over W, row by row (mm_best's: the score a float32, worked out exactly, mm_match);
 *       the first pictures in turn: one becomes the best if its score is at least `thr` (THRESHOLD) and above the
 *       best's so far as rounded, R(s) (s rounded to 3 decimals as Python's round(s, 3): the double nearest that
 *       decimal, ties to even; the score a Mob keeps);
 *       then, if it has next pictures and there's no best yet or the best's R is under `good` (REFIND_GOOD), those in
 *       turn the same way, stopping after the first that scores `good` or more (whether it became the best or not).
 *     The near window first, if the job has one: where its best's R is over `good` (strictly) the job is done there
 *     (found 1); else the full window (found 2, or 0 where it has no best either).
 *     The memo: a window at the same place, looked at with the same pictures (`key`: the caller's number for them), its
 *     pixels the very same as at the last call with that job `id` (byte for byte), may be answered as then without
 *     matching. A backend may keep none: the answers are the same. The memos of ids not in a call go after it, unless
 *     `keep` (a call for some of the last call's jobs again).
 *   Each job's answer in outs[i] (RfOut): for each window looked at ([0] near, [1] full) the best picture's index (-1:
 *     none scored `thr`), its score (the float32) and its top-left in the frame; `found`; what was done (`looked`'s
 *     bits, the matches, their places x picture pixels, the job's own time).
 *   rf_start(the same as rf_run), rf_next(ctx), rf_end(ctx): rf_run in three, so the caller does its own part of each
 *     job (below) while the others are matched. rf_start: 0, or -1 as rf_run (then no rf_end); rf_next: the index of a
 *     job whose answer is in, each once, in any order, waiting (or, here, doing one) until one is; -1 once all have
 *     been handed over; rf_end: after rf_start's 0, always, once the caller is done asking (it may stop early): every
 *     job done and nothing of the backend's still reading the call's memory when it returns. One call at a time per
 *     context: rf_start again only after rf_end.
 *   rf_close(ctx): its threads ended, its memory freed.
 * The caller's own: whether the picture found is the monster's kind (mobs._its_kind: twins told apart side by side),
 * the kind's whole set where these find nothing (the fallback), the HP bars, the tracks.
 *
 * Here: each job on one thread, the jobs taken in turn from a counter (largest first) by the pool's threads and by the
 * caller's in rf_next when none is in yet; each window loaded once (`load`) for its pictures; the matches through
 * `rf_best`, the one place this calls the matcher. A GPU backend would match every job's windows in one dispatch
 * (pictures uploaded at rf_open, the frame's windows at rf_start; each place's score, then each picture's first best
 * place and the stages' order above on the CPU, or on the GPU in one more pass), rf_next handing the jobs over as they
 * come back. */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

typedef struct {
    long long id;       /* the track's: its memo */
    long long key;      /* the caller's number for its pictures (its lists): the memo's */
    int wnear[4];       /* the near window: x0, y0, x1, y1 */
    int wfull[4];       /* the full window */
    int first_at, first_n, next_at, next_n;
} RfJob;

typedef struct {
    int tpl[2];         /* [0] the near window's best picture, [1] the full window's: its index, -1 none */
    float score[2];
    int x[2], y[2];     /* its top-left, frame px */
    int looked;         /* 1 near matched, 2 near from the memo, 4 full matched, 8 full from the memo, 16 failed */
    int found;          /* 0 none, 1 near, 2 full */
    int matches;        /* pictures matched */
    int unused;
    long long work;     /* their places x picture pixels */
    double us;          /* the job's own time, microseconds */
} RfOut;

typedef struct {
    int win[4];
    long long key;
    int tpl, x, y, valid;
    float score;
    size_t cap;
    unsigned char *px;  /* the window's pixels then, rows (x1 - x0) * cn bytes */
} RfMemo;

typedef struct {
    long long id;
    int seen;
    RfMemo m[2];        /* near, full */
} RfEntry;

#define RF_MAX_JOBS 1024

typedef struct {
    const unsigned char *frame;
    long long frow;
    int cn;
    const RfJob *jobs;
    const int *lists;
    RfOut *outs;
    int count;
    double thr, good;
    volatile long next;               /* the next of `order` to take */
    int nfinished, reported;          /* jobs done (in `finished`, in the order they were), and handed to the caller */
    RfEntry *memo[RF_MAX_JOBS];       /* each job's */
    int order[RF_MAX_JOBS];           /* the jobs, largest first */
    double cost[RF_MAX_JOBS];
    int finished[RF_MAX_JOBS];
} RfBatch;

typedef struct {
    const Masked **tpls;
    int ntpl, threads, running;
    RfEntry *memo;
    int nmemo, capmemo;
    double tick_us;
    RfBatch b;          /* the call's (one at a time) */
#ifdef _WIN32
    HANDLE *workers;
    int nworkers;
    SRWLOCK lock;
    CONDITION_VARIABLE wake, idle;
    long long batch;    /* the batch number: the pool's threads wake for a new one */
    int open, inside, quit;
#endif
} RfCtx;

static double rf_round3(float s)  /* Python's round(s, 3) (s * 1000 is exact in double; rint ties to even) */
{
    return rint((double)s * 1000.0) / 1000.0;
}

/* R(s), for the tests (Python's round(s, 3) of every float32 they try). */
EXPORT double rf_rounded(float s)
{
    return rf_round3(s);
}

/* The best place of picture p over the loaded window: the first highest score, row by row, as mm_best has it. The one
 * place the re-find step calls the matcher: match_loaded's, by transform where that's cheaper (ntt_least), else place
 * by place, the same scores to the bit either way. */
static void rf_best(Picture *im, const Masked *p, float *score, int *x, int *y)
{
    int rh = im->ih - p->th + 1, rw = im->iw - p->tw + 1;
    match_loaded(im, p, im->out);
    long long at = 0, size = (long long)rh * rw;
    float best = im->out[0];
    for (long long k = 1; k < size; k++) {
        if (im->out[k] > best) {
            best = im->out[k];
            at = k;
        }
    }
    *score = best;
    *x = (int)(at % rw);
    *y = (int)(at / rw);
}

static int rf_empty(const int *w) { return w[2] <= w[0] || w[3] <= w[1]; }

static int rf_recall(const RfBatch *b, const RfJob *j, const int *w, const RfMemo *m)
{
    if (!m->valid || m->key != j->key || memcmp(m->win, w, sizeof(m->win)) != 0)
        return 0;
    size_t row = (size_t)(w[2] - w[0]) * b->cn;
    const unsigned char *at = b->frame + (long long)w[1] * b->frow + (long long)w[0] * b->cn;
    for (int y = 0; y < w[3] - w[1]; y++) {
        if (memcmp(m->px + row * y, at + (long long)y * b->frow, row) != 0)
            return 0;
    }
    return 1;
}

static void rf_keep(const RfBatch *b, const RfJob *j, const int *w, RfMemo *m, int tpl, float score, int x, int y)
{
    size_t row = (size_t)(w[2] - w[0]) * b->cn, need = row * (size_t)(w[3] - w[1]);
    m->valid = 0;
    if (need > m->cap) {
        free(m->px);
        m->px = (unsigned char *)malloc(need + need / 4);
        m->cap = m->px ? need + need / 4 : 0;
        if (!m->px)
            return;
    }
    const unsigned char *at = b->frame + (long long)w[1] * b->frow + (long long)w[0] * b->cn;
    for (int r = 0; r < w[3] - w[1]; r++)
        memcpy(m->px + row * r, at + (long long)r * b->frow, row);
    memcpy(m->win, w, sizeof(m->win));
    m->key = j->key;
    m->tpl = tpl;
    m->score = score;
    m->x = x;
    m->y = y;
    m->valid = 1;
}

/* Whether any of job j's pictures (both stages') that fit a wh x ww window would go by transform there: the window is
 * loaded once for both stages, its transform (if made) shared by them. Which way a picture goes changes no answer. */
static int rf_wants_ntt(const RfCtx *c, const RfBatch *b, const RfJob *j, int wh, int ww)
{
    for (int stage = 0; stage < 2; stage++) {
        const int *list = b->lists + (stage ? j->next_at : j->first_at);
        int n = stage ? j->next_n : j->first_n;
        for (int k = 0; k < n; k++) {
            const Masked *p = c->tpls[list[k]];
            if (wh >= p->th && ww >= p->tw && ntt_wanted(p, wh, ww))
                return 1;
        }
    }
    return 0;
}

/* Window `w` of job j (`side` 0 near, 1 full): its best (above) into o's [side]. 0, or -1 if it couldn't be loaded. */
static int rf_window(const RfCtx *c, const RfBatch *b, const RfJob *j, const int *w, int side, RfMemo *m, RfOut *o)
{
    if (m && rf_recall(b, j, w, m)) {
        o->tpl[side] = m->tpl;
        o->score[side] = m->score;
        o->x[side] = m->x;
        o->y[side] = m->y;
        o->looked |= side ? 8 : 2;
        return 0;
    }
    int ww = w[2] - w[0], wh = w[3] - w[1], loaded = 0, best = -1, bx = 0, by = 0;
    float bs = 0.0f;
    double br = 0.0;
    Picture im;
    const unsigned char *at = b->frame + (long long)w[1] * b->frow + (long long)w[0] * b->cn;
    for (int stage = 0; stage < 2; stage++) {
        const int *list = b->lists + (stage ? j->next_at : j->first_at);
        int n = stage ? j->next_n : j->first_n;
        if (stage && !(n > 0 && (best < 0 || br < b->good)))
            break;
        for (int k = 0; k < n; k++) {
            const Masked *p = c->tpls[list[k]];
            if (wh < p->th || ww < p->tw)
                continue;  /* (doesn't fit) */
            if (!loaded) {  /* once for both stages: room for the transform if any of its pictures that fit wants it */
                if (load(&im, at, wh, ww, b->frow, b->cn, rf_wants_ntt(c, b, j, wh, ww)) != 0)
                    return -1;
                loaded = 1;
            }
            float s;
            int px, py;
            rf_best(&im, p, &s, &px, &py);
            o->matches++;
            o->work += (long long)(wh - p->th + 1) * (ww - p->tw + 1) * p->th * p->tw;
            if ((double)s >= b->thr && (best < 0 || (double)s > br)) {
                best = list[k];
                bs = s;
                br = rf_round3(s);
                bx = px;
                by = py;
            }
            if (stage && (double)s >= b->good)
                break;
        }
    }
    o->tpl[side] = best;
    o->score[side] = bs;
    o->x[side] = w[0] + bx;
    o->y[side] = w[1] + by;
    o->looked |= side ? 4 : 1;
    if (m)
        rf_keep(b, j, w, m, best, bs, w[0] + bx, w[1] + by);
    return 0;
}

static double rf_now_us(const RfCtx *c)
{
#ifdef _WIN32
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * c->tick_us;
#else
    (void)c;
    return 0.0;
#endif
}

static void rf_job(const RfCtx *c, const RfBatch *b, int i)
{
    const RfJob *j = b->jobs + i;
    RfOut *o = b->outs + i;
    RfEntry *e = b->memo[i];
    double t0 = rf_now_us(c);
    memset(o, 0, sizeof(*o));
    o->tpl[0] = o->tpl[1] = -1;
    if (!rf_empty(j->wfull)) {
        if (!rf_empty(j->wnear)) {
            if (rf_window(c, b, j, j->wnear, 0, e ? &e->m[0] : 0, o) != 0)
                o->looked |= 16;
            else if (o->tpl[0] >= 0 && rf_round3(o->score[0]) > b->good)
                o->found = 1;
        }
        if (!o->found && !(o->looked & 16)) {
            if (rf_window(c, b, j, j->wfull, 1, e ? &e->m[1] : 0, o) != 0)
                o->looked |= 16;
            else
                o->found = o->tpl[1] >= 0 ? 2 : 0;
        }
    }
    o->us = rf_now_us(c) - t0;
}

static int rf_take(RfBatch *b)  /* the next job to do, or -1 if every one is taken */
{
#ifdef _WIN32
    long k = InterlockedIncrement(&b->next) - 1;
#else
    long k = b->next++;
#endif
    return k < b->count ? b->order[k] : -1;
}

static void rf_lock(RfCtx *c)
{
#ifdef _WIN32
    AcquireSRWLockExclusive(&c->lock);
#else
    (void)c;
#endif
}

static void rf_unlock(RfCtx *c)
{
#ifdef _WIN32
    ReleaseSRWLockExclusive(&c->lock);
#else
    (void)c;
#endif
}

static void rf_done(RfCtx *c, int i)  /* job i done: handed over by the next rf_next */
{
    rf_lock(c);
    c->b.finished[c->b.nfinished++] = i;
#ifdef _WIN32
    WakeAllConditionVariable(&c->idle);
#endif
    rf_unlock(c);
}

#ifdef _WIN32
static DWORD WINAPI rf_worker(void *arg)
{
    RfCtx *c = (RfCtx *)arg;
    long long seen = 0;
    AcquireSRWLockExclusive(&c->lock);
    for (;;) {
        while (!c->quit && (c->batch == seen || !c->open))
            SleepConditionVariableSRW(&c->wake, &c->lock, INFINITE, 0);
        if (c->quit)
            break;
        seen = c->batch;
        c->inside++;
        ReleaseSRWLockExclusive(&c->lock);
        for (int i; (i = rf_take(&c->b)) >= 0;) {
            rf_job(c, &c->b, i);
            rf_done(c, i);
        }
        AcquireSRWLockExclusive(&c->lock);
        c->inside--;
        WakeAllConditionVariable(&c->idle);
    }
    ReleaseSRWLockExclusive(&c->lock);
    return 0;
}
#endif

EXPORT void rf_close(void *ctx)
{
    RfCtx *c = (RfCtx *)ctx;
    if (!c)
        return;
#ifdef _WIN32
    if (c->workers) {
        AcquireSRWLockExclusive(&c->lock);
        c->quit = 1;
        WakeAllConditionVariable(&c->wake);
        ReleaseSRWLockExclusive(&c->lock);
        for (int i = 0; i < c->nworkers; i++) {
            WaitForSingleObject(c->workers[i], INFINITE);
            CloseHandle(c->workers[i]);
        }
        free(c->workers);
    }
#endif
    for (int i = 0; i < c->nmemo; i++) {
        free(c->memo[i].m[0].px);
        free(c->memo[i].m[1].px);
    }
    free(c->memo);
    free((void *)c->tpls);
    free(c);
}

EXPORT void *rf_open(const void *const *tpls, int n, int threads)
{
    if (n <= 0 || threads <= 0)
        return 0;
    if (simd < 0)
        mm_simd(1);
    RfCtx *c = (RfCtx *)calloc(1, sizeof(RfCtx));
    if (!c)
        return 0;
    c->tpls = (const Masked **)malloc(sizeof(Masked *) * (size_t)n);
    if (!c->tpls) {
        free(c);
        return 0;
    }
    for (int i = 0; i < n; i++)
        c->tpls[i] = (const Masked *)tpls[i];
    c->ntpl = n;
    c->threads = threads;
#ifdef _WIN32
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    c->tick_us = 1e6 / (double)f.QuadPart;
    InitializeSRWLock(&c->lock);
    InitializeConditionVariable(&c->wake);
    InitializeConditionVariable(&c->idle);
    if (threads > 1) {
        c->workers = (HANDLE *)calloc((size_t)threads - 1, sizeof(HANDLE));
        if (!c->workers) {
            rf_close(c);
            return 0;
        }
        for (int i = 0; i < threads - 1; i++) {
            c->workers[i] = CreateThread(0, 0, rf_worker, c, 0, 0);
            if (!c->workers[i]) {
                rf_close(c);
                return 0;
            }
            c->nworkers++;
        }
    }
#endif
    return c;
}

/* Job id's memo entry (made if it has none); NULL without memory. */
static RfEntry *rf_entry(RfCtx *c, long long id)
{
    for (int i = 0; i < c->nmemo; i++) {
        if (c->memo[i].id == id)
            return &c->memo[i];
    }
    if (c->nmemo == c->capmemo) {
        int cap = c->capmemo ? 2 * c->capmemo : 32;
        RfEntry *more = (RfEntry *)realloc(c->memo, sizeof(RfEntry) * (size_t)cap);
        if (!more)
            return 0;
        c->memo = more;
        c->capmemo = cap;
    }
    RfEntry *e = &c->memo[c->nmemo++];
    memset(e, 0, sizeof(*e));
    e->id = id;
    return e;
}

/* rf_run in three: rf_start hands the jobs to the pool's threads (0, or -1 if it did nothing: a call already running,
 * a window outside the frame, a picture not the context's); rf_next hands the jobs over one at a time as they're done,
 * the caller's thread doing one itself where none is done yet and some aren't taken (its index, or -1 when every job
 * has been handed over), so the caller can do its own part of each meanwhile; rf_end waits for the pool's threads to
 * be out of the call (always called after rf_start's 0, whatever was handed over). rf_run: all three at once. */
EXPORT int rf_start(void *ctx, const unsigned char *frame, int fh, int fw, long long frow, int cn, const RfJob *jobs,
                    int njobs, const int *lists, double thr, double good, int keep, RfOut *outs)
{
    RfCtx *c = (RfCtx *)ctx;
    if (!c || c->running || njobs < 0 || njobs > RF_MAX_JOBS)
        return -1;
    for (int i = 0; i < njobs; i++) {  /* every window within the frame, every picture one of the context's */
        const RfJob *j = jobs + i;
        for (int side = 0; side < 2; side++) {
            const int *w = side ? j->wfull : j->wnear;
            if (!rf_empty(w) && (w[0] < 0 || w[1] < 0 || w[2] > fw || w[3] > fh))
                return -1;
        }
        for (int k = 0; k < j->first_n + j->next_n; k++) {
            int t = k < j->first_n ? lists[j->first_at + k] : lists[j->next_at + k - j->first_n];
            if (t < 0 || t >= c->ntpl || c->tpls[t]->cn != cn)
                return -1;
        }
    }
    RfBatch *b = &c->b;
    /* the memos: each job's entry (all made first: the array moves as it grows), the others' gone (unless `keep`) */
    for (int i = 0; i < c->nmemo; i++)
        c->memo[i].seen = 0;
    for (int i = 0; i < njobs; i++) {
        RfEntry *e = rf_entry(c, jobs[i].id);
        if (e)
            e->seen = 1;
    }
    if (!keep) {
        int kept = 0;
        for (int i = 0; i < c->nmemo; i++) {
            if (c->memo[i].seen)
                c->memo[kept++] = c->memo[i];
            else {
                free(c->memo[i].m[0].px);
                free(c->memo[i].m[1].px);
            }
        }
        c->nmemo = kept;
    }
    for (int i = 0; i < njobs; i++) {
        b->memo[i] = 0;
        for (int k = 0; k < c->nmemo; k++) {
            if (c->memo[k].id == jobs[i].id) {
                b->memo[i] = &c->memo[k];
                break;
            }
        }
        const int *w = rf_empty(jobs[i].wnear) ? jobs[i].wfull : jobs[i].wnear;
        b->cost[i] = rf_empty(jobs[i].wfull) ? 0.0 : (double)(w[2] - w[0]) * (w[3] - w[1]) * (jobs[i].first_n + 1);
        b->order[i] = i;
    }
    for (int i = 1; i < njobs; i++) {  /* largest first (an insertion sort: a dozen jobs) */
        int v = b->order[i], k = i;
        while (k > 0 && b->cost[b->order[k - 1]] < b->cost[v]) {
            b->order[k] = b->order[k - 1];
            k--;
        }
        b->order[k] = v;
    }
    b->frame = frame;
    b->frow = frow;
    b->cn = cn;
    b->jobs = jobs;
    b->lists = lists;
    b->outs = outs;
    b->count = njobs;
    b->thr = thr;
    b->good = good;
    b->next = 0;
    b->nfinished = 0;
    b->reported = 0;
    c->running = 1;
#ifdef _WIN32
    if (c->nworkers > 0 && njobs > 1) {
        AcquireSRWLockExclusive(&c->lock);
        c->open = 1;
        c->batch++;
        WakeAllConditionVariable(&c->wake);
        ReleaseSRWLockExclusive(&c->lock);
    }
#endif
    return 0;
}

EXPORT int rf_next(void *ctx)
{
    RfCtx *c = (RfCtx *)ctx;
    if (!c || !c->running)
        return -1;
    RfBatch *b = &c->b;
    for (;;) {
        rf_lock(c);
        if (b->reported < b->nfinished) {
            int i = b->finished[b->reported++];
            rf_unlock(c);
            return i;
        }
        if (b->reported >= b->count) {
            rf_unlock(c);
            return -1;
        }
        rf_unlock(c);
        int i = rf_take(b);
        if (i >= 0) {  /* none done yet, one not taken: this thread does it */
            rf_job(c, b, i);
            rf_done(c, i);
            continue;
        }
#ifdef _WIN32
        AcquireSRWLockExclusive(&c->lock);
        while (b->reported >= b->nfinished && b->reported < b->count)
            SleepConditionVariableSRW(&c->idle, &c->lock, INFINITE, 0);
        ReleaseSRWLockExclusive(&c->lock);
#endif
    }
}

EXPORT void rf_end(void *ctx)
{
    RfCtx *c = (RfCtx *)ctx;
    if (!c || !c->running)
        return;
    RfBatch *b = &c->b;
    for (int i; (i = rf_take(b)) >= 0;)  /* (the jobs not taken yet, if the caller stopped asking) */
    {
        rf_job(c, b, i);
        rf_done(c, i);
    }
#ifdef _WIN32
    AcquireSRWLockExclusive(&c->lock);
    while (c->inside > 0 || b->nfinished < b->count)
        SleepConditionVariableSRW(&c->idle, &c->lock, INFINITE, 0);
    c->open = 0;
    ReleaseSRWLockExclusive(&c->lock);
#endif
    c->running = 0;
}

EXPORT int rf_run(void *ctx, const unsigned char *frame, int fh, int fw, long long frow, int cn, const RfJob *jobs,
                  int njobs, const int *lists, double thr, double good, int keep, RfOut *outs)
{
    if (rf_start(ctx, frame, fh, fw, frow, cn, jobs, njobs, lists, thr, good, keep, outs) != 0)
        return -1;
    while (rf_next(ctx) >= 0)
        ;
    rf_end(ctx);
    return 0;
}
