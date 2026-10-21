/* gamelab's pieces in C (native.py builds and loads this; each caller does the same in Python when it isn't
 * there). Called through ctypes, which lets go of Python's lock for the call. Each does exactly what the Python
 * it stands in for does, in the same order and with the same IEEE double and float arithmetic (built with
 * /fp:precise: no fused multiply-adds, no reordering), so the answers are the same to the bit.
 */

#include <math.h>

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
        if (y <= surface && surface <= ny + 0.01) {
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
