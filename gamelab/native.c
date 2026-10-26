/* gamelab's pieces in C (native.py builds and loads this; each caller does the same in Python when it isn't
 * there). Called through ctypes, which lets go of Python's lock for the call. Each but the masked match does exactly
 * what the Python it stands in for does, in the same order and with the same IEEE double and float arithmetic (built
 * with /fp:precise: no fused multiply-adds, no reordering), so the answers are the same to the bit. The masked match
 * works out OpenCV's score exactly, in integers, where OpenCV rounds its float sums (see there).
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
 * The dot product goes 16 int16 products at a time (AVX2's madd), 8 places at once sharing each load of the template,
 * and only over each template row's 16-wide chunks the mask is in (its rows' `k0`, `k1`), where the CPU has AVX2
 * (checked at run time: the flags stay /fp:precise's, without /arch:AVX2, so the pieces above are compiled as ever);
 * else in plain C, the same integers. Its int32 lanes hold `flush` template rows of tw x cn products of at most 255^2
 * (each group then added in int64): every row at once for a 3-channel template up to about 105 x 105. The mask's sums
 * come from each image row's prefix sums, a run of mask pixels at a time (`runs`: row, from, to). The image and those
 * sums go into memory of the calling thread's own, kept from call to call (`scratch`): no allocation per match. */

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
    long long n, A;      /* the mask's pixels; sum_c (n STT_c - ST_c^2) */
    long long ST[3];
    long long STT[3];
    /* then: runs (3 ints each), k0 and k1 (an int each a row), the template's rows at t_off (int16, tpad each) */
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

/* The bytes mm_prepare needs for a th x tw template of cn channels. */
EXPORT long long mm_prepare_size(int th, int tw, int cn)
{
    long long tpad = ((long long)tw * cn + 15) / 16 * 16;
    long long ints = 3LL * th * (tw / 2 + 1) + 2LL * th;
    return (long long)sizeof(Masked) + 4 * ints + 32 + 2 * tpad * th;
}

/* A template made ready for mm_match and mm_best, into `buf` (`size` bytes: mm_prepare_size): its colours `tpl`
 * (pixel (y, x) channel c at tpl[y * trow + x * cn + c]) and its mask (mask[y * mrow + x * mpix], nonzero: in).
 * Returns 0, or -1 for what it doesn't take (another channel count; a template row too wide for the int32 lanes),
 * -2 if `size` is too small. */
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
    }
    p->nruns = nruns;
    for (int c = 0; c < cn; c++)
        p->A += p->n * p->STT[c] - p->ST[c] * p->ST[c];
    long long per_row = (long long)tw * cn * SQ255;
    long long rows = INT32_LIMIT / per_row;
    p->flush = rows >= th ? th : (int)rows;
    p->wide = p->n * cn * SQ255 > INT32_LIMIT;
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

#define BLOCK 8  /* places a dot-product pass does at once */

typedef struct {
    int ih, iw, cn, ipad;
    short *I;           /* the image, int16, rows ipad apart, zeros past each row's end */
    int *P1;            /* each row's prefix sums of each channel: P1[(row * (iw + 1) + x) * cn + c] */
    int *P2;            /* ... and of the squares, the channels summed: P2[row * (iw + 1) + x] */
    int *S1;            /* a result row's sums under the mask: S1[x * cn + c] */
    long long *S2;      /* ... of the squares (int32 in its first half unless the template is `wide`) */
    long long *DOT;     /* ... its dot products */
    float *out;         /* a result, for mm_best */
} Picture;

static size_t align64(size_t n) { return (n + 63) / 64 * 64; }

/* The image (pixel (y, x) channel c at img[y * irow + x * cn + c]) loaded into the thread's scratch: its rows as
 * int16 and their prefix sums. 0, or -1 if it can't be (a row too wide for the int32 sums, no memory). */
static int load(Picture *im, const unsigned char *img, int ih, int iw, long long irow, int cn)
{
    if ((long long)iw * cn * SQ255 > INT32_LIMIT || ih <= 0 || iw <= 0)
        return -1;
    int ipad = iw * cn + BLOCK * cn + 32;
    size_t sI = align64(sizeof(short) * (size_t)ipad * ih), sP1 = align64(sizeof(int) * (size_t)(iw + 1) * cn * ih);
    size_t sP2 = align64(sizeof(int) * (size_t)(iw + 1) * ih), sS1 = align64(sizeof(int) * (size_t)iw * cn);
    size_t sS2 = align64(sizeof(long long) * (size_t)(iw + BLOCK)), sOut = align64(sizeof(float) * (size_t)iw * ih);
    unsigned char *s = scratch(sI + sP1 + sP2 + sS1 + 2 * sS2 + sOut);
    if (!s)
        return -1;
    im->ih = ih;
    im->iw = iw;
    im->cn = cn;
    im->ipad = ipad;
    im->I = (short *)s;
    im->P1 = (int *)(s + sI);
    im->P2 = (int *)(s + sI + sP1);
    im->S1 = (int *)(s + sI + sP1 + sP2);
    im->S2 = (long long *)(s + sI + sP1 + sP2 + sS1);
    im->DOT = (long long *)(s + sI + sP1 + sP2 + sS1 + sS2);
    im->out = (float *)(s + sI + sP1 + sP2 + sS1 + 2 * sS2);
    int irow_n = iw * cn;
    for (int y = 0; y < ih; y++) {
        const unsigned char *src = img + y * irow;
        short *dst = im->I + (long long)y * ipad;
        for (int k = 0; k < irow_n; k++)
            dst[k] = src[k];
        for (int k = irow_n; k < ipad; k++)
            dst[k] = 0;
        int *p1 = im->P1 + (long long)y * (iw + 1) * cn, *p2 = im->P2 + (long long)y * (iw + 1);
        for (int c = 0; c < cn; c++)
            p1[c] = 0;
        p2[0] = 0;
        for (int x = 0; x < iw; x++) {
            int sq = 0;
            for (int c = 0; c < cn; c++) {
                int v = src[x * cn + c];
                p1[(x + 1) * cn + c] = p1[x * cn + c] + v;
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

/* The scores of template `p` at every place of the loaded image, into `out` (rh x rw). */
static void match_loaded(const Picture *im, const Masked *p, float *out)
{
    int cn = p->cn, iw = im->iw, rh = im->ih - p->th + 1, rw = iw - p->tw + 1;
    const int *runs = masked_runs(p);
    int *S1 = im->S1;
    long long *S2 = im->S2, *DOT = im->DOT;
    int *S2i = (int *)S2;
    for (int y = 0; y < rh; y++) {
        memset(S1, 0, sizeof(int) * (size_t)rw * cn);
        memset(S2, 0, sizeof(long long) * (size_t)rw);
        for (int q = 0; q < p->nruns; q++) {
            int ry = runs[3 * q], a = runs[3 * q + 1], b = runs[3 * q + 2];
            const int *p1 = im->P1 + (long long)(y + ry) * (iw + 1) * cn;
            const int *pa = p1 + (long long)a * cn, *pb = p1 + (long long)b * cn;
            int m = rw * cn;
            for (int k = 0; k < m; k++)
                S1[k] += pb[k] - pa[k];
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
#if MM_AVX2
        if (simd)
            dots_avx2(im, p, y, rw, DOT);
        else
#endif
            dots_c(im, p, y, rw, DOT);
        float *o = out + (long long)y * rw;
        for (int x = 0; x < rw; x++) {
            long long N = p->n * DOT[x], B = p->n * (p->wide ? S2[x] : (long long)S2i[x]);
            for (int c = 0; c < cn; c++) {
                long long s1 = S1[(long long)x * cn + c];
                N -= p->ST[c] * s1;
                B -= s1 * s1;
            }
            double den = (double)p->A * (double)B;
            o[x] = den > 0 ? (float)((double)N / sqrt(den)) : -1.0f;
        }
    }
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
    if (cn != p->cn || ih < p->th || iw < p->tw || load(&im, img, ih, iw, irow, cn) != 0)
        return -1;
    match_loaded(&im, p, out);
    return 0;
}

/* Each of `n` templates in turn (mm_prepare's) matched over the image (as mm_match's), and its best score and place
 * as cv2.minMaxLoc gives them: the first highest in row-major order, into scores, xs and ys; after the first whose
 * best is `stop` or more, no more. Returns how many were matched, or -1 if one can't be (as mm_match). The image is
 * loaded once for them all. */
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
    if (load(&im, img, ih, iw, irow, cn) != 0)
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
