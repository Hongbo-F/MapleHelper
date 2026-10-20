/* The monster search's step-1 bookkeeping in C (perception/kernels.py builds and loads this; mobs.py uses it
 * when it's there). Python called through ctypes lets go of its lock for the call, so the pool's threads do
 * this at once; in numpy it held the lock and took 18 ms of a band search's 35.
 *
 * It does exactly what mobs._spots does after its match, in the same order and the same arithmetic (no fused
 * multiply-adds: built with /fp:precise), so the answers are the same to the bit:
 *   1. a window with too little of the kind's colour scores 0: its colour, summed from the integral image as
 *      ((ii[y+h][x+w] - ii[y][x+w]) - ii[y+h][x]) + ii[y][x] (numpy's order), divided by the template's size,
 *      below `cover` (in double, as numpy compares float64 with a Python float);
 *   2. a peak is a window scoring at least `thr` and no less than any of its 8 neighbours (a 3x3 dilation's
 *      equality; outside the result counts as nothing), compared as float32 (numpy 2 casts the Python float to
 *      the array's float32);
 *   3. the peaks in row-major order, as 64-bit ints (np.nonzero's).
 */

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/* `res`: the match's scores (rh x rw floats, row by row), zeroed in place where the cover is too low; `ii`: the
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
