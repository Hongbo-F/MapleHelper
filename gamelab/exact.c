/* Monsters found by the exact pixels of their pictures (perception.exact): the game draws a picture's opaque pixels as they
   are, so on a frame of the game's own size the pixels of a monster that show are exactly its picture's, however much of it
   is covered.

   Anchors: pairs of side-by-side pixels of a picture, chosen by the caller (rare in the picture, their first colour sparse
   on the map). A frame's pixel pair equal to an anchor's votes for that picture at that offset (`ex_find`); a (picture,
   offset) with min_votes votes has its pixels compared one by one; those with min_exact exact pixels come back, the most
   exact first, each frame pixel explained by one find only (a picture and its neighbour frame of the animation share most
   of their pixels: the best takes them, and the other has too few left).

   No intrinsics: plain C, built as native.py builds the others. Frames and pictures are RGB, 3 bytes a pixel. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define EXPORT __declspec(dllexport)

#define OFF 256              /* a picture's left or top may lie this far off the frame */
#define MAX_PICS 1024        /* pictures at most: 10 bits of a vote's id (11 bits each for x and y + OFF) */
#define VOTE_BITS 18         /* the votes' table: 262,144 (picture, offset) pairs a frame at most */
#define MAX_CANDS 4096       /* (picture, offset) pairs a frame that reach min_votes */

typedef struct { uint64_t key; uint32_t first, n; } Slot;
typedef struct { uint16_t pic, x, y; } Anchor;
typedef struct { int w, h, n; uint16_t *xy; uint8_t *c; } Pic;
typedef struct { uint32_t id; uint16_t gen, cnt; } Vote;
typedef struct {
    int npics; Pic *pics;
    Slot *slots; uint32_t smask; Anchor *anchors;
    uint8_t lut[1 << 15];    /* whether a 15-bit colour starts any anchor: most of a frame's pixels stop here */
    Vote *votes; uint32_t vmask; uint16_t gen;
    uint8_t *explained; int ew, eh;
    uint32_t cand[MAX_CANDS];
} Ctx;
typedef struct { int pic, ox, oy, exact, fresh; } Found;

static uint32_t hash64(uint64_t k) { return (uint32_t)((k * 0x9E3779B97F4A7C15ull) >> 40); }
static uint32_t hash32(uint32_t k) { return (k * 2654435761u) >> 8; }
static uint64_t pair_key(const uint8_t *a) {
    return (uint64_t)a[0] | (uint64_t)a[1] << 8 | (uint64_t)a[2] << 16 | (uint64_t)a[3] << 24 | (uint64_t)a[4] << 32
         | (uint64_t)a[5] << 40;
}

/* npics pictures: rgbs[p] and masks[p] (w x h, rows contiguous; dims[2p], dims[2p + 1]; copied: the caller may let them
   go); nanch anchors, (picture, x, y) each. NULL with too many pictures. */
EXPORT void *ex_open(int npics, const uint8_t **rgbs, const uint8_t **masks, const int *dims, int nanch, const int *anch) {
    if (npics > MAX_PICS) return NULL;
    Ctx *c = (Ctx *)calloc(1, sizeof(Ctx));
    c->npics = npics; c->pics = (Pic *)calloc(npics, sizeof(Pic));
    for (int p = 0; p < npics; p++) {
        Pic *P = &c->pics[p]; P->w = dims[2 * p]; P->h = dims[2 * p + 1];
        int n = 0; for (int i = 0; i < P->w * P->h; i++) n += masks[p][i] != 0;
        P->n = n; P->xy = (uint16_t *)malloc(4 * (size_t)(n + 1)); P->c = (uint8_t *)malloc(3 * (size_t)(n + 1));
        int k = 0;
        for (int y = 0; y < P->h; y++) for (int x = 0; x < P->w; x++) if (masks[p][y * P->w + x]) {
            P->xy[2 * k] = (uint16_t)x; P->xy[2 * k + 1] = (uint16_t)y;
            memcpy(P->c + 3 * k, rgbs[p] + 3 * (y * P->w + x), 3); k++;
        }
    }
    uint64_t *keys = (uint64_t *)malloc(8 * (size_t)(nanch + 1));
    for (int i = 0; i < nanch; i++) {
        int p = anch[3 * i], x = anch[3 * i + 1], y = anch[3 * i + 2];
        const uint8_t *a = rgbs[p] + 3 * (y * dims[2 * p] + x);
        keys[i] = pair_key(a);
        c->lut[(a[0] >> 3) << 10 | (a[1] >> 3) << 5 | (a[2] >> 3)] = 1;
    }
    uint32_t size = 1 << 16; while (size < (uint32_t)nanch * 4) size <<= 1;
    c->smask = size - 1; c->slots = (Slot *)calloc(size, sizeof(Slot));
    c->anchors = (Anchor *)malloc(sizeof(Anchor) * (size_t)(nanch + 1));
    for (int i = 0; i < nanch; i++) {
        uint32_t h = hash64(keys[i]) & c->smask;
        while (c->slots[h].n && c->slots[h].key != keys[i]) h = (h + 1) & c->smask;
        c->slots[h].key = keys[i]; c->slots[h].n++;
    }
    uint32_t at = 0;
    for (uint32_t h = 0; h < size; h++) if (c->slots[h].n) { c->slots[h].first = at; at += c->slots[h].n; c->slots[h].n = 0; }
    for (int i = 0; i < nanch; i++) {
        uint32_t h = hash64(keys[i]) & c->smask;
        while (c->slots[h].key != keys[i]) h = (h + 1) & c->smask;
        Anchor *A = &c->anchors[c->slots[h].first + c->slots[h].n++];
        A->pic = (uint16_t)anch[3 * i]; A->x = (uint16_t)anch[3 * i + 1]; A->y = (uint16_t)anch[3 * i + 2];
    }
    free(keys);
    c->vmask = (1u << VOTE_BITS) - 1; c->votes = (Vote *)calloc((size_t)c->vmask + 1, sizeof(Vote)); c->gen = 0;
    return c;
}

EXPORT void ex_close(void *ctx) {
    Ctx *c = (Ctx *)ctx;
    if (c == NULL) return;
    for (int p = 0; p < c->npics; p++) { free(c->pics[p].xy); free(c->pics[p].c); }
    free(c->pics); free(c->slots); free(c->anchors); free(c->votes); free(c->explained); free(c);
}

static int cmp_found(const void *a, const void *b) { return ((const Found *)b)->exact - ((const Found *)a)->exact; }

/* How many of picture P's pixels at (ox, oy) are the frame's exactly. */
static int exact_at(const Pic *P, int ox, int oy, const uint8_t *f, int H, int W, long long stride) {
    int n = 0;
    for (int k = 0; k < P->n; k++) {
        int X = ox + P->xy[2 * k], Y = oy + P->xy[2 * k + 1];
        if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
        const uint8_t *q = f + Y * stride + 3 * X, *t = P->c + 3 * k;
        n += q[0] == t[0] && q[1] == t[1] && q[2] == t[2];
    }
    return n;
}

/* The pictures on frame f (H x W, RGB, stride bytes a row) with their anchors' first pixels in the box x0..x1, y0..y1:
   5 ints each in out (picture, ox, oy, exact pixels, exact pixels no find before it explained), cap at most; how many.
   Not to be called on one context from two threads at once. */
EXPORT int ex_find(void *ctx, const uint8_t *f, int H, int W, long long stride, int x0, int y0, int x1, int y1,
                   int min_votes, int min_exact, int *out, int cap) {
    Ctx *c = (Ctx *)ctx;
    c->gen++; if (c->gen == 0) { memset(c->votes, 0, sizeof(Vote) * ((size_t)c->vmask + 1)); c->gen = 1; }
    if (!c->explained || c->ew != W || c->eh != H) {
        free(c->explained); c->explained = (uint8_t *)calloc((size_t)W * H, 1); c->ew = W; c->eh = H;
    }
    int nc = 0; uint32_t used = 0, most = (c->vmask + 1) / 4 * 3;
    if (x1 > W - 1) x1 = W - 1;
    if (y1 > H) y1 = H;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    for (int y = y0; y < y1; y++) {
        const uint8_t *row = f + y * stride;
        for (int x = x0; x < x1; x++) {
            const uint8_t *a = row + 3 * x;
            if (!c->lut[(a[0] >> 3) << 10 | (a[1] >> 3) << 5 | (a[2] >> 3)]) continue;
            uint64_t k = pair_key(a);
            uint32_t h = hash64(k) & c->smask;
            while (c->slots[h].n && c->slots[h].key != k) h = (h + 1) & c->smask;
            if (!c->slots[h].n) continue;
            const Slot *s = &c->slots[h];
            for (uint32_t i = 0; i < s->n; i++) {
                const Anchor *A = &c->anchors[s->first + i];
                int ox = x - A->x + OFF, oy = y - A->y + OFF;
                if (ox < 0 || oy < 0 || ox >= 2048 || oy >= 2048) continue;
                uint32_t id = (uint32_t)A->pic << 22 | (uint32_t)oy << 11 | (uint32_t)ox;
                uint32_t v = hash32(id) & c->vmask;
                for (;;) {
                    Vote *V = &c->votes[v];
                    if (V->gen != c->gen) {
                        if (used >= most) break;  /* (the table all but full: the frame isn't the game's) */
                        used++; V->gen = c->gen; V->id = id; V->cnt = 1;
                        if (min_votes == 1 && nc < MAX_CANDS) c->cand[nc++] = id;
                        break;
                    }
                    if (V->id == id) { if (++V->cnt == (uint16_t)min_votes && nc < MAX_CANDS) c->cand[nc++] = id; break; }
                    v = (v + 1) & c->vmask;
                }
            }
        }
    }
    Found *F = (Found *)malloc(sizeof(Found) * (size_t)(nc + 1)); int nf = 0;
    for (int i = 0; i < nc; i++) {
        int pic = c->cand[i] >> 22, oy = (int)((c->cand[i] >> 11) & 0x7FF) - OFF, ox = (int)(c->cand[i] & 0x7FF) - OFF;
        int n = exact_at(&c->pics[pic], ox, oy, f, H, W, stride);
        if (n >= min_exact) { F[nf].pic = pic; F[nf].ox = ox; F[nf].oy = oy; F[nf].exact = n; nf++; }
    }
    qsort(F, nf, sizeof(Found), cmp_found);
    int kept = 0;
    for (int i = 0; i < nf && kept < cap; i++) {
        const Pic *P = &c->pics[F[i].pic]; int fresh = 0;
        for (int k = 0; k < P->n; k++) {
            int X = F[i].ox + P->xy[2 * k], Y = F[i].oy + P->xy[2 * k + 1];
            if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
            const uint8_t *q = f + Y * stride + 3 * X, *t = P->c + 3 * k;
            if (q[0] == t[0] && q[1] == t[1] && q[2] == t[2] && !c->explained[Y * W + X]) fresh++;
        }
        F[i].fresh = fresh;
        if (fresh < min_exact) continue;
        for (int k = 0; k < P->n; k++) {
            int X = F[i].ox + P->xy[2 * k], Y = F[i].oy + P->xy[2 * k + 1];
            if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
            const uint8_t *q = f + Y * stride + 3 * X, *t = P->c + 3 * k;
            if (q[0] == t[0] && q[1] == t[1] && q[2] == t[2]) c->explained[Y * W + X] = 1;
        }
        out[5 * kept] = F[i].pic; out[5 * kept + 1] = F[i].ox; out[5 * kept + 2] = F[i].oy;
        out[5 * kept + 3] = F[i].exact; out[5 * kept + 4] = fresh; kept++;
    }
    for (int i = 0; i < kept; i++) {  /* (the explained map cleared where it was written: a frame's cost, not a screen's) */
        const Pic *P = &c->pics[out[5 * i]];
        for (int k = 0; k < P->n; k++) {
            int X = out[5 * i + 1] + P->xy[2 * k], Y = out[5 * i + 2] + P->xy[2 * k + 1];
            if (X < 0 || Y < 0 || X >= W || Y >= H) continue;
            c->explained[Y * W + X] = 0;
        }
    }
    free(F);
    return kept;
}
