/* The camera's registration against the map's art in C (localization/camreg.py: what it does and why; mapart.py draws
 * the art from the game data), its own DLL (camreg.py's `lib`), built by native.build like native.c's, with the same
 * compiler and flags. Each function gives exactly the integers camreg.py's numpy gives where the DLL isn't there
 * (tests compare the two); SSSE3's and AVX2's intrinsics where the CPU has AVX2 (cr_simd), else plain C, the same
 * answers.
 *
 *   cr_gray: a frame's sample rows as gray, avg(avg(p0, p2), p1) of each pixel's first three bytes, avg(a, b) being
 *     (a + b + 1) >> 1 (SSE2's pavgb): the same for RGB and BGR frames, so the service's RGB frames and saved BGR
 *     pictures register alike.
 *   cr_shrink: a gray picture in k x k blocks' means, (sum + k^2 / 2) / k^2 (a cold search's first pass).
 *   cr_costs: for each candidate view (vx[i], vy[i]: the map point at the frame's top-left), over the sample rows'
 *     pixels where both the frame's mask (`gm`: 255 in play, 0 under the HUD, the minimap, ...) and the art's (`am`:
 *     255 where the map's fixed art alone is drawn, mapart's `static`) are set, the sum of min(|frame - art|, trunc)
 *     and how many pixels: the art's pixel for the frame's (row rows[r], column c) is the map point (vx + c,
 *     vy + rows[r]), at art[(vy + rows[r] - ay0) * arow + vx + c - ax0]; outside the art, no pixel.
 */
#include <immintrin.h>
#include <intrin.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static int simd = -1;  /* -1: not asked yet; 0: plain C; 1: AVX2 (and SSSE3) */

static int has_avx2(void)
{
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7)
        return 0;
    __cpuid(r, 1);
    if (!(r[2] & (1 << 27)) || !(r[2] & (1 << 28)) || !(r[2] & (1 << 9)))  /* OSXSAVE, AVX, SSSE3 */
        return 0;
    if ((_xgetbv(0) & 6) != 6)  /* the OS saves the YMM registers */
        return 0;
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;  /* AVX2 */
}

/* Whether they run on AVX2: `use` 1 asks for it (where the CPU has it), 0 for plain C (the tests compare the two), -1
 * leaves it. Returns what's in use. */
EXPORT int cr_simd(int use)
{
    if (simd < 0 || use >= 0)
        simd = use != 0 && has_avx2();
    return simd;
}

/* --- gray --- */

static unsigned char avg8(unsigned a, unsigned b) { return (unsigned char)((a + b + 1) >> 1); }

/* pshufb masks: channel k of 16 pixels from the 3 (cn 3) or 4 (cn 4) 16-byte loads that hold them */
static __declspec(align(16)) unsigned char masks3[3][3][16], masks4[4][3][16];
static volatile long masks_ready;

static void make_masks(void)
{
    for (int k = 0; k < 3; k++)
        for (int j = 0; j < 3; j++)
            for (int i = 0; i < 16; i++) {
                int at = 3 * i + k - 16 * j;
                masks3[j][k][i] = (unsigned char)(at >= 0 && at < 16 ? at : 0x80);
            }
    for (int k = 0; k < 3; k++)
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 16; i++) {
                int at = 4 * i + k - 16 * j;
                masks4[j][k][i] = (unsigned char)(at >= 0 && at < 16 ? at : 0x80);
            }
    masks_ready = 1;
}

static void gray_row_plain(const unsigned char *p, int fw, int cn, unsigned char *o)
{
    for (int c = 0; c < fw; c++, p += cn)
        o[c] = avg8(avg8(p[0], p[2]), p[1]);
}

static void gray_row_simd(const unsigned char *p, int fw, int cn, unsigned char *o)
{
    int c = 0;
    if (cn == 3) {
        for (; c + 16 <= fw && (c + 16) * 3 <= fw * 3; c += 16) {
            const unsigned char *q = p + 3 * c;
            __m128i a = _mm_loadu_si128((const __m128i *)q), b = _mm_loadu_si128((const __m128i *)(q + 16)),
                    d = _mm_loadu_si128((const __m128i *)(q + 32)), ch[3];
            for (int k = 0; k < 3; k++)
                ch[k] = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, *(const __m128i *)masks3[0][k]),
                                                  _mm_shuffle_epi8(b, *(const __m128i *)masks3[1][k])),
                                     _mm_shuffle_epi8(d, *(const __m128i *)masks3[2][k]));
            _mm_storeu_si128((__m128i *)(o + c), _mm_avg_epu8(_mm_avg_epu8(ch[0], ch[2]), ch[1]));
        }
    } else if (cn == 4) {
        for (; c + 16 <= fw; c += 16) {
            const unsigned char *q = p + 4 * c;
            __m128i v[4], ch[3];
            for (int j = 0; j < 4; j++)
                v[j] = _mm_loadu_si128((const __m128i *)(q + 16 * j));
            for (int k = 0; k < 3; k++)
                ch[k] = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(v[0], *(const __m128i *)masks4[0][k]),
                                                  _mm_shuffle_epi8(v[1], *(const __m128i *)masks4[1][k])),
                                     _mm_or_si128(_mm_shuffle_epi8(v[2], *(const __m128i *)masks4[2][k]),
                                                  _mm_shuffle_epi8(v[3], *(const __m128i *)masks4[3][k])));
            _mm_storeu_si128((__m128i *)(o + c), _mm_avg_epu8(_mm_avg_epu8(ch[0], ch[2]), ch[1]));
        }
    }
    if (c < fw)
        gray_row_plain(p + (long long)c * cn, fw - c, cn, o + c);
}

/* The frame's rows `rows` (nrows of them) as gray, each fw wide, into `out` (nrows x fw); the frame fh x fw, `frow`
 * bytes a row, `cn` (3 or 4) bytes a pixel. Returns 0, or -1 if a row is outside the frame or cn isn't 3 or 4. */
EXPORT int cr_gray(const unsigned char *frame, int fh, int fw, long long frow, int cn, const int *rows, int nrows,
                   unsigned char *out)
{
    if (cn != 3 && cn != 4)
        return -1;
    if (simd < 0)
        cr_simd(1);
    if (!masks_ready)
        make_masks();
    for (int r = 0; r < nrows; r++) {
        if (rows[r] < 0 || rows[r] >= fh)
            return -1;
        const unsigned char *p = frame + (long long)rows[r] * frow;
        unsigned char *o = out + (long long)r * fw;
        if (simd)
            gray_row_simd(p, fw, cn, o);
        else
            gray_row_plain(p, fw, cn, o);
    }
    return 0;
}

/* A gray picture (h x w, `grow` bytes a row) in k x k blocks' means, (sum + k^2 / 2) / k^2, into `out` ((h / k) x
 * (w / k)): the part past a whole block left out (camreg.py's shrink). Returns 0, or -1 for a k out of 1-16. */
EXPORT int cr_shrink(const unsigned char *g, int h, int w, long long grow, int k, unsigned char *out)
{
    if (k < 1 || k > 16)
        return -1;
    int sh = h / k, sw = w / k, half = k * k / 2, kk = k * k;
    unsigned sums[4096];
    for (int r = 0; r < sh; r++) {
        unsigned char *o = out + (long long)r * sw;
        for (int c0 = 0; c0 < sw; c0 += 4096) {
            int n = sw - c0 < 4096 ? sw - c0 : 4096;
            for (int c = 0; c < n; c++)
                sums[c] = 0;
            for (int i = 0; i < k; i++) {
                const unsigned char *p = g + (long long)(r * k + i) * grow + (long long)c0 * k;
                for (int c = 0; c < n; c++, p += k) {
                    unsigned s = 0;
                    for (int j = 0; j < k; j++)
                        s += p[j];
                    sums[c] += s;
                }
            }
            for (int c = 0; c < n; c++)
                o[c0 + c] = (unsigned char)((sums[c] + half) / kk);
        }
    }
    return 0;
}

/* --- costs --- */

static void row_plain(const unsigned char *g, const unsigned char *gm, const unsigned char *a, const unsigned char *m,
                      int c0, int c1, int trunc, long long *sum, long long *count)
{
    long long s = 0, k = 0;
    for (int c = c0; c < c1; c++) {
        if (gm[c] & m[c]) {
            int d = g[c] > a[c] ? g[c] - a[c] : a[c] - g[c];
            s += d < trunc ? d : trunc;
            k++;
        }
    }
    *sum += s;
    *count += k;
}

static const __declspec(align(32)) unsigned char tail_mask[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

static __forceinline __m256i chunk(const unsigned char *g, const unsigned char *gm, const unsigned char *a,
                                   const unsigned char *m, int c, __m256i cap, __m256i keep, __m256i *on)
{
    __m256i f = _mm256_loadu_si256((const __m256i *)(g + c)), v = _mm256_loadu_si256((const __m256i *)(a + c));
    *on = _mm256_and_si256(_mm256_and_si256(_mm256_loadu_si256((const __m256i *)(gm + c)),
                                            _mm256_loadu_si256((const __m256i *)(m + c))), keep);
    __m256i d = _mm256_or_si256(_mm256_subs_epu8(f, v), _mm256_subs_epu8(v, f));
    return _mm256_and_si256(_mm256_min_epu8(d, cap), *on);
}

static void row_avx2(const unsigned char *g, const unsigned char *gm, const unsigned char *a, const unsigned char *m,
                     int c0, int c1, int trunc, long long *sum, long long *count)
{
    if (c1 - c0 < 32) {
        row_plain(g, gm, a, m, c0, c1, trunc, sum, count);
        return;
    }
    const __m256i zero = _mm256_setzero_si256(), one = _mm256_set1_epi8(1), cap = _mm256_set1_epi8((char)trunc),
                  all = _mm256_set1_epi8((char)0xFF);
    __m256i s = zero, k = zero, on;
    int c = c0;
    for (; c + 32 <= c1; c += 32) {
        __m256i d = chunk(g, gm, a, m, c, cap, all, &on);
        s = _mm256_add_epi64(s, _mm256_sad_epu8(d, zero));
        k = _mm256_add_epi64(k, _mm256_sad_epu8(_mm256_and_si256(on, one), zero));
    }
    if (c < c1) {  /* the last 32 again, but for the lanes already counted */
        int rest = c1 - c;
        __m256i keep = _mm256_loadu_si256((const __m256i *)(tail_mask + rest));
        __m256i d = chunk(g, gm, a, m, c1 - 32, cap, keep, &on);
        s = _mm256_add_epi64(s, _mm256_sad_epu8(d, zero));
        k = _mm256_add_epi64(k, _mm256_sad_epu8(_mm256_and_si256(on, one), zero));
    }
    long long ls[4], lk[4];
    _mm256_storeu_si256((__m256i *)ls, s);
    _mm256_storeu_si256((__m256i *)lk, k);
    *sum += ls[0] + ls[1] + ls[2] + ls[3];
    *count += lk[0] + lk[1] + lk[2] + lk[3];
}

/* The candidates' sums and counts (above). `g`, `gm`: the frame's sample rows' gray and mask (nrows x gw); `rows`: each
 * one's row in the frame; `art`, `am`: the art's gray and static mask (ah x aw, `arow` bytes a row), its pixel [0][0]
 * the map point (ax0, ay0); `trunc` 1-255. Masks are 0 or 255. Returns 0, or -1 for a `trunc` out of range. */
EXPORT int cr_costs(const unsigned char *g, const unsigned char *gm, int nrows, int gw, const int *rows,
                    const unsigned char *art, const unsigned char *am, int ah, int aw, long long arow, int ax0, int ay0,
                    const int *vx, const int *vy, int n, int trunc, long long *sums, long long *counts)
{
    if (trunc < 1 || trunc > 255)
        return -1;
    if (simd < 0)
        cr_simd(1);
    void (*row)(const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, int, int,
                int, long long *, long long *) = simd ? row_avx2 : row_plain;
    for (int i = 0; i < n; i++) {
        long long s = 0, k = 0;
        int dx = vx[i] - ax0;              /* the art's column for the frame's column 0 */
        int c0 = dx < 0 ? -dx : 0, c1 = aw - dx < gw ? aw - dx : gw;
        if (c0 < c1) {
            for (int r = 0; r < nrows; r++) {
                int ar = vy[i] + rows[r] - ay0;
                if (ar < 0 || ar >= ah)
                    continue;
                const unsigned char *ap = art + (long long)ar * arow + dx, *mp = am + (long long)ar * arow + dx;
                const long long gr = (long long)r * gw;
                row(g + gr, gm + gr, ap, mp, c0, c1, trunc, &s, &k);
            }
        }
        sums[i] = s;
        counts[i] = k;
    }
    return 0;
}
