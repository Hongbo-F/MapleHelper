/* --- a frame's re-find matches on the GPU (Direct3D 11): the host side of gpumatch.hlsl -------------------------------
 * Included by native.c after its re-find types (it reads Masked, RfJob, RfBatch as they are there): a backend behind
 * rf_* that gives the CPU's answers to the bit (the contract above rf_*, its GPU paragraph). The GPU works out each
 * place's integer sums (STI, S1_c, S2: exact, uint32) and an approximate score s~ (float32, within GM_EPS of the
 * float32 score); of each window's places it hands back those within 2 GM_EPS of the highest s~ (the only ones that
 * can be the best), with their sums; the CPU scores those as score_at does (int64 N and B, the double, the float32)
 * and takes the first highest row by row (gm_answer). A region with more candidates than GM_CAND, a picture past 128 x
 * 128 or flat (A 0), or any failure: the CPU matches that one as before (rf_best). So every answer is the CPU's.
 *
 * Per call (rf_start): each job's full window with every picture of its two lists that fits it (both stages at once:
 * which of them the stages look at is decided on the CPU afterwards, as rf_window decides it), the near window as a
 * sub-rectangle of the full one's places (a place's sums are its pixels', whichever window it's counted in); the
 * windows' bytes copied into one upload buffer; three dispatches (gpumatch.hlsl's prep, match, choose) and the
 * candidates copied to a staging buffer, in one submission; rf_next waits for its fence (an event, no spinning), at
 * most GM_WAIT_US: past that, or on any error, the call goes to the CPU's threads as before, at once, and so do the
 * next ones for GM_REST_US and until the GPU has caught up; past a limit (GM_MAX_*), that call; a device removed (a
 * driver reset) turns the GPU off for the context's life.
 * The device is made on a thread of its own (rf_open doesn't wait: 0.7 s to on, the device 0.2-0.35 s of it, the
 * three shaders compiled at run time by d3dcompiler_47), the pictures uploaded once, every buffer made once (no
 * allocation per call; about 35 MB of video memory). Only a hardware adapter (never WARP, the Basic Render Driver:
 * that's the CPU again). d3d11, dxgi and d3dcompiler_47 are loaded at run time, so the DLL's imports are as before and
 * a machine without them just has no GPU path. Measured where the service runs (an RTX 3080, which idles at P8 between
 * the frame loop's calls, a crowd's frame of matches 5 ms there, against 2.1 on the C's 4 threads): off by default
 * (native.GPU), as it doesn't pay; D:/MapleHelper-scratch/gpu; README, The re-finds' matches on the GPU. */

#ifdef _WIN32
#define COBJMACROS
#include <initguid.h>
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#if defined(__has_include)
#if __has_include("gpumatch_hlsl.h")
#include "gpumatch_hlsl.h"  /* GM_HLSL: gpumatch.hlsl as a string (native.py writes it at the build) */
#endif
#endif
#endif

#define GM_GROUP 64                       /* threads a group (gpumatch.hlsl's GROUP) */
#define GM_RUN 4                          /* places side by side a thread (gpumatch.hlsl's RUN) */
#define GM_CAND 32                        /* candidates a region hands back (more: that one on the CPU) */
#define GM_EPS (1.0f / 16384)             /* how far s~ may be from the float32 score: 2^-14 (gpumatch.hlsl) */
#define GM_MAX_TASKS 1024                 /* (window, picture) pairs a call (a crowd's frame: about 45) */
#define GM_MAX_WINS (2 * RF_MAX_JOBS)
#define GM_MAX_PX (1 << 19)               /* window pixels a call (a crowd's frame: about 100,000) */
#define GM_MAX_PLACES (1 << 19)           /* places a call, over its tasks (a crowd's frame: 95,000, at most 190,000) */
#define GM_MAX_PRE (GM_MAX_PX + GM_MAX_PX / 8)  /* their rows' prefix sums */
#define GM_MAX_GROUPS (GM_MAX_PLACES / GM_GROUP + GM_MAX_TASKS)  /* (a thread at least a place) */
#define GM_MAX_RAW (3 * GM_MAX_PX + 4 * GM_MAX_WINS)
#define GM_WAIT_US 6000                   /* the longest rf_next waits for the GPU before the CPU does the call */
#define GM_REST_US 1000000                /* after a call that came back late, the calls the CPU's this long */

enum { GM_STARTING = 0, GM_READY = 1, GM_OFF = 2 };

typedef struct { unsigned th, tw, nruns, run_off, px_off, n, st0, st1, st2; float af; unsigned u0, u1; } GmPic;
typedef struct { unsigned ww, wh, raw_off, px_off, pre_off, u0, u1, u2; } GmWinGpu;
typedef struct { unsigned win, pic, rw, rh, nx0, ny0, nx1, ny1, place_off, u0, u1, u2; } GmTaskGpu;
typedef struct { unsigned ntask, nwin, cn, kcand; float band; unsigned u[3]; } GmFrame;

typedef struct { const unsigned char *at; long long row; int ww, wh; } GmWin;  /* a window's pixels, CPU side */

/* What rf_gpu reports (stats[...]). */
enum { GS_CALLS, GS_GPU, GS_LATE, GS_BUSY, GS_FULL, GS_PAIRS, GS_CPU_PAIRS, GS_WAIT_US, GS_WAIT_MAX_US, GS_TASKS,
       GS_PLACES, GS_WHY, GS_GPU_US, GS_GPU_MAX_US, GS_MANY, GS_NONE, GS_PASS_US, GS_N = GS_PASS_US + 4 };

typedef struct GmCtx {
    volatile LONG state;
    int why;                       /* GS_WHY: what turned it off (gm_off's codes) */
    int fault;                     /* the tests' (rf_gpu_fault) */
    const Masked **tpls;
    int ntpl;
    unsigned char *takes;          /* each picture: whether the GPU matches it */
#ifdef _WIN32
    HANDLE starter;
    HMODULE d3d, dxgi, compiler;
    ID3D11Device *dev;
    ID3D11DeviceContext *ctx;
    ID3D11Device5 *dev5;
    ID3D11DeviceContext4 *ctx4;
    ID3D11Fence *fence;
    HANDLE done, timer;
    UINT64 fv;
    ID3D11ComputeShader *cs[3];
    ID3D11Query *q_disjoint, *q_t[5];  /* the GPU's own time of a call's passes: before prep, after each, the copies */
    ID3D11Buffer *pics, *runs, *tpx, *raw, *wins, *tasks, *groups, *frame;
    ID3D11Buffer *pxf, *pre, *sums, *sc, *rec, *cand, *s_rec, *s_cand;
    ID3D11ShaderResourceView *srv[7];
    ID3D11UnorderedAccessView *uav[6];
#endif
    /* the call's */
    GmWin win[GM_MAX_WINS];
    int win_rect[GM_MAX_WINS][4];         /* (rf's: each window's place in the frame) */
    GmWinGpu wgpu[GM_MAX_WINS];
    GmTaskGpu task[GM_MAX_TASKS];
    unsigned group[2 * GM_MAX_GROUPS];
    int nwin, ntask, ngroup, nplace, npx, npre, cn;
    const unsigned *got_rec, *got_cand;   /* the staging buffers mapped: the answers */
    double rest_until;                    /* (gm_now_us) the calls the CPU's till then: one came back late */
    unsigned last_rec[4 * GM_MAX_TASKS];  /* (the tests': the last call's records) */
    double t_sent;
    /* rf's: each job's tasks [job_at, job_at + job_n), whether its near window is a sub-rectangle of the full one */
    int job_at[RF_MAX_JOBS], job_n[RF_MAX_JOBS], job_sub[RF_MAX_JOBS];
    unsigned char task_side[GM_MAX_TASKS];  /* 0: the near window's own, 1: the full window's */
    long long stats[GS_N];
    double tick_us;
} GmCtx;

static int gm_use = 0;         /* gm_on(): contexts rf_open makes get the GPU */
static long long gm_wait_most = GM_WAIT_US, gm_rest = GM_REST_US;  /* gm_waits() */
static int gm_fault_open = 0;  /* rf_gpu_fault(NULL, 3): the next contexts' device fails to be made */

/* Whether rf_open's contexts get the GPU backend: 1 on, 0 off, -1 leaves it. Returns what's in use. */
EXPORT int gm_on(int use)
{
    if (use >= 0)
        gm_use = use != 0;
    return gm_use;
}

/* The longest rf_next waits for the GPU's answers (us), and how long the calls are the CPU's after one that came back
 * late (us): the tests' and the measurements' (a GPU that's slow, measured as it is: a long wait, no rest); -1 leaves
 * either; `was` (2, or NULL): the two before. */
EXPORT void gm_waits(long long wait_us, long long rest_us, long long *was)
{
    if (was) {
        was[0] = gm_wait_most;
        was[1] = gm_rest;
    }
    if (wait_us >= 0)
        gm_wait_most = wait_us;
    if (rest_us >= 0)
        gm_rest = rest_us;
}

static double gm_now_us(const GmCtx *g)
{
#ifdef _WIN32
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * g->tick_us;
#else
    (void)g;
    return 0.0;
#endif
}

/* The float32 score of a place from its sums: score_at's arithmetic, to the bit. */
static float gm_score(const Masked *p, unsigned dot, const unsigned *s1, unsigned s2)
{
    long long N = p->n * (long long)dot, B = p->n * (long long)s2;
    for (int c = 0; c < p->cn; c++) {
        long long v = s1[c];
        N -= p->ST[c] * v;
        B -= v * v;
    }
    double den = (double)p->A * (double)B;
    return den > 0 ? (float)((double)N / sqrt(den)) : -1.0f;
}

/* Whether the GPU takes picture p: up to 128 x 128 (its sums under 2^32, its runs' ends in a byte), not flat. */
static int gm_takes(const Masked *p) { return p->th <= 128 && p->tw <= 128 && p->A > 0 && p->cn <= 3; }

#ifdef _WIN32

#define GM_RELEASE(x) do { if (x) { IUnknown_Release((IUnknown *)(x)); (x) = 0; } } while (0)

static void gm_free_device(GmCtx *g)
{
    for (int i = 0; i < 7; i++)
        GM_RELEASE(g->srv[i]);
    for (int i = 0; i < 6; i++)
        GM_RELEASE(g->uav[i]);
    ID3D11Buffer **bs[] = {&g->pics, &g->runs, &g->tpx, &g->raw, &g->wins, &g->tasks, &g->groups, &g->frame, &g->pxf,
                           &g->pre, &g->sums, &g->sc, &g->rec, &g->cand, &g->s_rec, &g->s_cand};
    for (size_t i = 0; i < sizeof(bs) / sizeof(bs[0]); i++)
        GM_RELEASE(*bs[i]);
    for (int i = 0; i < 3; i++)
        GM_RELEASE(g->cs[i]);
    GM_RELEASE(g->fence);
    GM_RELEASE(g->q_disjoint);
    for (int i = 0; i < 5; i++)
        GM_RELEASE(g->q_t[i]);
    GM_RELEASE(g->ctx4);
    GM_RELEASE(g->dev5);
    if (g->ctx) {
        ID3D11DeviceContext_ClearState(g->ctx);
        ID3D11DeviceContext_Flush(g->ctx);
    }
    GM_RELEASE(g->ctx);
    GM_RELEASE(g->dev);
}

/* Off for good (`why`: 1 no hardware adapter, 2 no device, 3 no fence, 4 the shader, 5 the buffers, 6 no
 * d3d11/dxgi/d3dcompiler, 7 removed (a driver reset, or rf_gpu_fault's 2 and 4), 8 an error at a call (or its 5)). */
static int gm_off(GmCtx *g, int why)
{
    g->why = why;
    g->stats[GS_WHY] = why;
    InterlockedExchange(&g->state, GM_OFF);
    return -1;
}

static ID3D11Buffer *gm_buffer(GmCtx *g, size_t size, UINT stride, UINT bind, D3D11_USAGE usage, UINT cpu, UINT misc,
                               const void *init)
{
    D3D11_BUFFER_DESC d;
    memset(&d, 0, sizeof d);
    d.ByteWidth = (UINT)(stride ? size : (size + 15) / 16 * 16);  /* (a structured one: its elements' bytes) */
    d.Usage = usage;
    d.BindFlags = bind;
    d.CPUAccessFlags = cpu;
    d.MiscFlags = misc;
    d.StructureByteStride = stride;
    D3D11_SUBRESOURCE_DATA s;
    memset(&s, 0, sizeof s);
    s.pSysMem = init;
    ID3D11Buffer *b = 0;
    if (FAILED(ID3D11Device_CreateBuffer(g->dev, &d, init ? &s : 0, &b)))
        return 0;
    return b;
}

static ID3D11Buffer *gm_structured(GmCtx *g, size_t n, UINT stride, D3D11_USAGE usage, UINT bind, const void *init,
                                   ID3D11ShaderResourceView **srv, ID3D11UnorderedAccessView **uav)
{
    UINT cpu = usage == D3D11_USAGE_DYNAMIC ? D3D11_CPU_ACCESS_WRITE : 0;
    ID3D11Buffer *b = gm_buffer(g, n * stride, stride, bind, usage, cpu, D3D11_RESOURCE_MISC_BUFFER_STRUCTURED, init);
    if (!b)
        return 0;
    if (srv && FAILED(ID3D11Device_CreateShaderResourceView(g->dev, (ID3D11Resource *)b, 0, srv)))
        return b;  /* (the caller sees the view missing) */
    if (uav && FAILED(ID3D11Device_CreateUnorderedAccessView(g->dev, (ID3D11Resource *)b, 0, uav)))
        return b;
    return b;
}

typedef HRESULT(WINAPI *GmCreateFactory)(REFIID, void **);
typedef HRESULT(WINAPI *GmCompile)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO *, ID3DInclude *, LPCSTR, LPCSTR,
                                   UINT, UINT, ID3DBlob **, ID3DBlob **);

/* The device, the shaders, the pictures and every buffer: on gm_open's thread. */
static int gm_make(GmCtx *g)
{
#ifndef GM_HLSL
    return gm_off(g, 4);
#else
    if (g->fault == 3)
        return gm_off(g, 1);
    g->d3d = LoadLibraryA("d3d11.dll");
    g->dxgi = LoadLibraryA("dxgi.dll");
    g->compiler = LoadLibraryA("d3dcompiler_47.dll");
    PFN_D3D11_CREATE_DEVICE create = g->d3d ? (PFN_D3D11_CREATE_DEVICE)(void *)GetProcAddress(g->d3d,
                                                                                           "D3D11CreateDevice") : 0;
    GmCreateFactory factory = g->dxgi ? (GmCreateFactory)(void *)GetProcAddress(g->dxgi, "CreateDXGIFactory1") : 0;
    GmCompile compile = g->compiler ? (GmCompile)(void *)GetProcAddress(g->compiler, "D3DCompile") : 0;
    if (!create || !factory || !compile)
        return gm_off(g, 6);
    IDXGIFactory1 *fac = 0;
    IDXGIAdapter1 *pick = 0;
    if (FAILED(factory(&IID_IDXGIFactory1, (void **)&fac)))
        return gm_off(g, 6);
    for (UINT i = 0; !pick; i++) {  /* the first hardware adapter (never the Basic Render Driver: the CPU's) */
        IDXGIAdapter1 *a = 0;
        if (IDXGIFactory1_EnumAdapters1(fac, i, &a) != S_OK)
            break;
        DXGI_ADAPTER_DESC1 d;
        if (SUCCEEDED(IDXGIAdapter1_GetDesc1(a, &d)) && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && d.VendorId != 0x1414)
            pick = a;
        else
            IDXGIAdapter1_Release(a);
    }
    IDXGIFactory1_Release(fac);
    if (!pick)
        return gm_off(g, 1);
    D3D_FEATURE_LEVEL want[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0}, got;
    HRESULT hr = create((IDXGIAdapter *)pick, D3D_DRIVER_TYPE_UNKNOWN, 0, 0, want, 2, D3D11_SDK_VERSION, &g->dev, &got,
                        &g->ctx);
    IDXGIAdapter1_Release(pick);
    if (FAILED(hr))
        return gm_off(g, 2);
    if (FAILED(ID3D11Device_QueryInterface(g->dev, &IID_ID3D11Device5, (void **)&g->dev5)) ||
        FAILED(ID3D11DeviceContext_QueryInterface(g->ctx, &IID_ID3D11DeviceContext4, (void **)&g->ctx4)) ||
        FAILED(ID3D11Device5_CreateFence(g->dev5, 0, D3D11_FENCE_FLAG_NONE, &IID_ID3D11Fence, (void **)&g->fence)))
        return gm_off(g, 3);
    D3D11_QUERY_DESC qd = {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
    ID3D11Device_CreateQuery(g->dev, &qd, &g->q_disjoint);
    qd.Query = D3D11_QUERY_TIMESTAMP;
    for (int i = 0; i < 5; i++)
        ID3D11Device_CreateQuery(g->dev, &qd, &g->q_t[i]);
    g->done = CreateEventA(0, FALSE, FALSE, 0);
    g->timer = CreateWaitableTimerExW(0, 0, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!g->timer)  /* (before Windows 10 1803: a timer of the system's resolution) */
        g->timer = CreateWaitableTimerExW(0, 0, 0, TIMER_ALL_ACCESS);
    if (!g->done || !g->timer)
        return gm_off(g, 3);
    const char *entry[3] = {"prep", "match", "choose"};
    for (int i = 0; i < 3; i++) {
        ID3DBlob *code = 0, *err = 0;
        hr = compile(GM_HLSL, strlen(GM_HLSL), "gpumatch.hlsl", 0, 0, entry[i], "cs_5_0",
                     D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_IEEE_STRICTNESS, 0, &code, &err);
        if (err)
            ID3D10Blob_Release(err);
        if (FAILED(hr) || !code)
            return gm_off(g, 4);
        hr = ID3D11Device_CreateComputeShader(g->dev, ID3D10Blob_GetBufferPointer(code),
                                              ID3D10Blob_GetBufferSize(code), 0, &g->cs[i]);
        ID3D10Blob_Release(code);
        if (FAILED(hr))
            return gm_off(g, 4);
    }
    /* the pictures, once */
    size_t nruns = 0, npx = 0;
    for (int i = 0; i < g->ntpl; i++) {
        const Masked *p = g->tpls[i];
        g->takes[i] = (unsigned char)gm_takes(p);
        if (g->takes[i]) {
            nruns += p->nruns;
            npx += (size_t)p->th * p->tw;
        }
    }
    GmPic *pics = (GmPic *)calloc((size_t)g->ntpl + 1, sizeof(GmPic));
    unsigned *runs = (unsigned *)calloc(nruns + 1, sizeof(unsigned));
    float *tpx = (float *)calloc(4 * (npx + 1), sizeof(float));
    int ok = pics && runs && tpx;
    for (int i = 0, r = 0, x = 0; ok && i < g->ntpl; i++) {
        const Masked *p = g->tpls[i];
        if (!g->takes[i])
            continue;
        GmPic *q = pics + i;
        q->th = (unsigned)p->th;
        q->tw = (unsigned)p->tw;
        q->nruns = (unsigned)p->nruns;
        q->run_off = (unsigned)r;
        q->px_off = (unsigned)x;
        q->n = (unsigned)p->n;
        q->st0 = (unsigned)p->ST[0];
        q->st1 = p->cn > 1 ? (unsigned)p->ST[1] : 0;
        q->st2 = p->cn > 2 ? (unsigned)p->ST[2] : 0;
        q->af = (float)p->A;
        const int *pr = masked_runs(p);
        for (int k = 0; k < p->nruns; k++)
            runs[r++] = (unsigned)pr[3 * k] | (unsigned)pr[3 * k + 1] << 8 | (unsigned)pr[3 * k + 2] << 16;
        const short *T = masked_rows(p);
        for (int y = 0; y < p->th; y++)
            for (int xx = 0; xx < p->tw; xx++, x++)
                for (int c = 0; c < p->cn; c++)
                    tpx[4 * (size_t)x + c] = (float)T[(long long)y * p->tpad + xx * p->cn + c];
    }
    if (ok) {
        g->pics = gm_structured(g, (size_t)g->ntpl + 1, sizeof(GmPic), D3D11_USAGE_IMMUTABLE,
                                D3D11_BIND_SHADER_RESOURCE, pics, &g->srv[0], 0);
        g->runs = gm_structured(g, nruns + 1, 4, D3D11_USAGE_IMMUTABLE, D3D11_BIND_SHADER_RESOURCE, runs, &g->srv[1],
                                0);
        g->tpx = gm_structured(g, npx + 1, 16, D3D11_USAGE_IMMUTABLE, D3D11_BIND_SHADER_RESOURCE, tpx, &g->srv[2], 0);
    }
    free(pics);
    free(runs);
    free(tpx);
    if (!ok || !g->srv[0] || !g->srv[1] || !g->srv[2])
        return gm_off(g, 5);
    /* the calls' buffers, once */
    g->raw = gm_buffer(g, GM_MAX_RAW, 0, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE,
                       D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS, 0);
    if (g->raw) {
        D3D11_SHADER_RESOURCE_VIEW_DESC v;
        memset(&v, 0, sizeof v);
        v.Format = DXGI_FORMAT_R32_TYPELESS;
        v.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
        v.BufferEx.NumElements = (GM_MAX_RAW + 15) / 16 * 4;
        v.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
        ID3D11Device_CreateShaderResourceView(g->dev, (ID3D11Resource *)g->raw, &v, &g->srv[3]);
    }
    g->wins = gm_structured(g, GM_MAX_WINS, sizeof(GmWinGpu), D3D11_USAGE_DYNAMIC, D3D11_BIND_SHADER_RESOURCE, 0,
                            &g->srv[4], 0);
    g->tasks = gm_structured(g, GM_MAX_TASKS, sizeof(GmTaskGpu), D3D11_USAGE_DYNAMIC, D3D11_BIND_SHADER_RESOURCE, 0,
                             &g->srv[5], 0);
    g->groups = gm_structured(g, GM_MAX_GROUPS, 8, D3D11_USAGE_DYNAMIC, D3D11_BIND_SHADER_RESOURCE, 0, &g->srv[6], 0);
    g->frame = gm_buffer(g, sizeof(GmFrame), 0, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC,
                         D3D11_CPU_ACCESS_WRITE, 0, 0);
    UINT rw = D3D11_BIND_UNORDERED_ACCESS;
    g->pxf = gm_structured(g, GM_MAX_PX, 16, D3D11_USAGE_DEFAULT, rw, 0, 0, &g->uav[0]);
    g->pre = gm_structured(g, GM_MAX_PRE, 16, D3D11_USAGE_DEFAULT, rw, 0, 0, &g->uav[1]);
    g->sums = gm_structured(g, GM_MAX_PLACES, 16, D3D11_USAGE_DEFAULT, rw, 0, 0, &g->uav[2]);
    g->sc = gm_structured(g, GM_MAX_PLACES, 8, D3D11_USAGE_DEFAULT, rw, 0, 0, &g->uav[3]);
    g->cand = gm_structured(g, (size_t)GM_MAX_TASKS * 2 * GM_CAND * 6, 4, D3D11_USAGE_DEFAULT, rw, 0, 0, &g->uav[5]);
    g->rec = gm_buffer(g, (size_t)GM_MAX_TASKS * 16, 0, rw, D3D11_USAGE_DEFAULT, 0,
                       D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS, 0);
    if (g->rec) {
        D3D11_UNORDERED_ACCESS_VIEW_DESC v;
        memset(&v, 0, sizeof v);
        v.Format = DXGI_FORMAT_R32_TYPELESS;
        v.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        v.Buffer.NumElements = GM_MAX_TASKS * 4;
        v.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        ID3D11Device_CreateUnorderedAccessView(g->dev, (ID3D11Resource *)g->rec, &v, &g->uav[4]);
    }
    g->s_rec = gm_buffer(g, (size_t)GM_MAX_TASKS * 16, 0, 0, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ, 0, 0);
    g->s_cand = gm_buffer(g, (size_t)GM_MAX_TASKS * 2 * GM_CAND * 6 * 4, 0, 0, D3D11_USAGE_STAGING,
                          D3D11_CPU_ACCESS_READ, 0, 0);
    for (int i = 0; i < 7; i++)
        if (!g->srv[i])
            return gm_off(g, 5);
    for (int i = 0; i < 6; i++)
        if (!g->uav[i])
            return gm_off(g, 5);
    if (!g->frame || !g->s_rec || !g->s_cand)
        return gm_off(g, 5);
    ID3D11DeviceContext_CSSetShaderResources(g->ctx, 0, 7, g->srv);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(g->ctx, 0, 6, g->uav, 0);
    ID3D11DeviceContext_CSSetConstantBuffers(g->ctx, 0, 1, &g->frame);
    InterlockedExchange(&g->state, GM_READY);
    return 0;
#endif
}

static DWORD WINAPI gm_starter(void *arg)
{
    GmCtx *g = (GmCtx *)arg;
    if (gm_make(g) != 0)
        gm_free_device(g);
    return 0;
}

#endif /* _WIN32 */

/* A GPU backend for the n pictures (rf_open's), made ready on a thread of its own; NULL without memory. */
static GmCtx *gm_open(const Masked **tpls, int n, int fault)
{
    GmCtx *g = (GmCtx *)calloc(1, sizeof(GmCtx));
    if (!g)
        return 0;
    g->tpls = tpls;
    g->ntpl = n;
    g->fault = fault;
    g->takes = (unsigned char *)calloc((size_t)n, 1);
    if (!g->takes) {
        free(g);
        return 0;
    }
#ifdef _WIN32
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    g->tick_us = 1e6 / (double)f.QuadPart;
    g->starter = CreateThread(0, 0, gm_starter, g, 0, 0);
    if (!g->starter)
        gm_off(g, 2);
#else
    g->state = GM_OFF;
#endif
    return g;
}

static void gm_close(GmCtx *g)
{
    if (!g)
        return;
#ifdef _WIN32
    if (g->starter) {
        WaitForSingleObject(g->starter, INFINITE);
        CloseHandle(g->starter);
    }
    if (g->got_rec && g->ctx) {
        ID3D11DeviceContext_Unmap(g->ctx, (ID3D11Resource *)g->s_rec, 0);
        ID3D11DeviceContext_Unmap(g->ctx, (ID3D11Resource *)g->s_cand, 0);
    }
    gm_free_device(g);
    if (g->done)
        CloseHandle(g->done);
    if (g->timer)
        CloseHandle(g->timer);
    if (g->d3d)
        FreeLibrary(g->d3d);
    if (g->dxgi)
        FreeLibrary(g->dxgi);
    if (g->compiler)
        FreeLibrary(g->compiler);
#endif
    free(g->takes);
    free(g);
}

/* A window for the call (its pixels at `at`, rows `row` bytes apart): its index, or -1 past the limits. */
static int gm_add_window(GmCtx *g, const unsigned char *at, long long row, int ww, int wh)
{
    if (g->nwin >= GM_MAX_WINS || g->npx + (long long)ww * wh > GM_MAX_PX ||
        g->npre + (long long)(ww + 1) * wh > GM_MAX_PRE)
        return -1;
    GmWin *w = g->win + g->nwin;
    w->at = at;
    w->row = row;
    w->ww = ww;
    w->wh = wh;
    g->npx += ww * wh;
    g->npre += (ww + 1) * wh;
    return g->nwin++;
}

/* A task: picture `pic` over window `win`, its near sub-rectangle of places [nx0, nx1) x [ny0, ny1) (none: empty).
 * Its index, or -1 past the limits. */
static int gm_add_task(GmCtx *g, int win, int pic, int nx0, int ny0, int nx1, int ny1)
{
    const Masked *p = g->tpls[pic];
    const GmWin *w = g->win + win;
    int rw = w->ww - p->tw + 1, rh = w->wh - p->th + 1;
    if (g->ntask >= GM_MAX_TASKS || rw <= 0 || rh <= 0 || g->nplace + (long long)rw * rh > GM_MAX_PLACES)
        return -1;
    GmTaskGpu *t = g->task + g->ntask;
    memset(t, 0, sizeof *t);
    t->win = (unsigned)win;
    t->pic = (unsigned)pic;
    t->rw = (unsigned)rw;
    t->rh = (unsigned)rh;
    if (nx1 > nx0 && ny1 > ny0) {
        t->nx0 = (unsigned)nx0;
        t->ny0 = (unsigned)ny0;
        t->nx1 = (unsigned)nx1;
        t->ny1 = (unsigned)ny1;
    }
    t->place_off = (unsigned)g->nplace;
    g->nplace += rw * rh;
    return g->ntask++;
}

/* The call's windows and tasks (gm_add_*) to the GPU, in one submission: 0, or -1 (nothing sent: the CPU's). */
static int gm_send(GmCtx *g)
{
#ifdef _WIN32
    if (g->ntask == 0)
        return -1;
    g->ngroup = 0;
    for (int k = 0; k < g->ntask; k++) {  /* a task's thread slots: ceil(rw / GM_RUN) a row */
        unsigned slots = (g->task[k].rw + GM_RUN - 1) / GM_RUN * g->task[k].rh;
        for (unsigned first = 0; first < slots; first += GM_GROUP) {
            if (g->ngroup >= GM_MAX_GROUPS)
                return -1;
            g->group[2 * g->ngroup] = (unsigned)k;
            g->group[2 * g->ngroup + 1] = first;
            g->ngroup++;
        }
    }
    ID3D11DeviceContext *x = g->ctx;
    D3D11_MAPPED_SUBRESOURCE m;
    if (g->fault == 5 || FAILED(ID3D11DeviceContext_Map(x, (ID3D11Resource *)g->raw, 0, D3D11_MAP_WRITE_DISCARD, 0,
                                                        &m)))  /* (5: the tests') */
        return gm_off(g, 8);
    GmWinGpu *ws = g->wgpu;
    unsigned raw = 0, px = 0, pre = 0;
    for (int i = 0; i < g->nwin; i++) {  /* each window's rows, side by side */
        const GmWin *w = g->win + i;
        size_t rowb = (size_t)w->ww * g->cn;
        ws[i].ww = (unsigned)w->ww;
        ws[i].wh = (unsigned)w->wh;
        ws[i].raw_off = raw;
        ws[i].px_off = px;
        ws[i].pre_off = pre;
        for (int y = 0; y < w->wh; y++)
            memcpy((unsigned char *)m.pData + raw + rowb * y, w->at + w->row * y, rowb);
        raw += (unsigned)((rowb * w->wh + 3) / 4 * 4);
        px += (unsigned)(w->ww * w->wh);
        pre += (unsigned)((w->ww + 1) * w->wh);
    }
    ID3D11DeviceContext_Unmap(x, (ID3D11Resource *)g->raw, 0);
    struct { ID3D11Buffer *b; const void *src; size_t size; } up[4] = {
        {g->wins, ws, sizeof(GmWinGpu) * (size_t)g->nwin},
        {g->tasks, g->task, sizeof(GmTaskGpu) * (size_t)g->ntask},
        {g->groups, g->group, 8 * (size_t)g->ngroup},
        {g->frame, 0, sizeof(GmFrame)}};
    GmFrame fr;
    memset(&fr, 0, sizeof fr);
    fr.ntask = (unsigned)g->ntask;
    fr.nwin = (unsigned)g->nwin;
    fr.cn = (unsigned)g->cn;
    fr.kcand = GM_CAND;
    fr.band = 2 * GM_EPS;
    up[3].src = &fr;
    for (int i = 0; i < 4; i++) {
        if (FAILED(ID3D11DeviceContext_Map(x, (ID3D11Resource *)up[i].b, 0, D3D11_MAP_WRITE_DISCARD, 0, &m)))
            return gm_off(g, 8);
        memcpy(m.pData, up[i].src, up[i].size);
        ID3D11DeviceContext_Unmap(x, (ID3D11Resource *)up[i].b, 0);
    }
    int timed = g->q_disjoint && g->q_t[0] && g->q_t[1] && g->q_t[2] && g->q_t[3] && g->q_t[4];
    if (timed) {
        ID3D11DeviceContext_Begin(x, (ID3D11Asynchronous *)g->q_disjoint);
        ID3D11DeviceContext_End(x, (ID3D11Asynchronous *)g->q_t[0]);
    }
    ID3D11DeviceContext_CSSetShader(x, g->cs[0], 0, 0);
    ID3D11DeviceContext_Dispatch(x, (UINT)(g->nwin + (g->ntask + 127) / 128), 1, 1);
    if (timed)
        ID3D11DeviceContext_End(x, (ID3D11Asynchronous *)g->q_t[1]);
    ID3D11DeviceContext_CSSetShader(x, g->cs[1], 0, 0);
    ID3D11DeviceContext_Dispatch(x, (UINT)g->ngroup, 1, 1);
    if (timed)
        ID3D11DeviceContext_End(x, (ID3D11Asynchronous *)g->q_t[2]);
    ID3D11DeviceContext_CSSetShader(x, g->cs[2], 0, 0);
    ID3D11DeviceContext_Dispatch(x, (UINT)g->ngroup, 1, 1);
    if (timed)
        ID3D11DeviceContext_End(x, (ID3D11Asynchronous *)g->q_t[3]);
    D3D11_BOX box = {0, 0, 0, (UINT)(16 * g->ntask), 1, 1};
    ID3D11DeviceContext_CopySubresourceRegion(x, (ID3D11Resource *)g->s_rec, 0, 0, 0, 0, (ID3D11Resource *)g->rec, 0,
                                              &box);
    box.right = (UINT)(g->ntask * 2 * GM_CAND * 6 * 4);
    ID3D11DeviceContext_CopySubresourceRegion(x, (ID3D11Resource *)g->s_cand, 0, 0, 0, 0, (ID3D11Resource *)g->cand,
                                              0, &box);
    if (timed) {
        ID3D11DeviceContext_End(x, (ID3D11Asynchronous *)g->q_t[4]);
        ID3D11DeviceContext_End(x, (ID3D11Asynchronous *)g->q_disjoint);
    }
    g->fv++;
    ResetEvent(g->done);
    if (FAILED(ID3D11DeviceContext4_Signal(g->ctx4, g->fence, g->fv)) ||
        FAILED(ID3D11Fence_SetEventOnCompletion(g->fence, g->fv, g->done)))
        return gm_off(g, 8);
    ID3D11DeviceContext_Flush(x);
    g->t_sent = gm_now_us(g);
    g->stats[GS_TASKS] += g->ntask;
    g->stats[GS_PLACES] += g->nplace;
    return 0;
#else
    (void)g;
    return -1;
#endif
}

/* Whether a call can go to the GPU now (ready, and the last one it was sent done: one that came back late keeps
 * the next calls on the CPU until it's in), the call's lists emptied. */
static int gm_ready(GmCtx *g, int cn)
{
    if (!g || g->state != GM_READY)
        return 0;
#ifdef _WIN32
    UINT64 done = ID3D11Fence_GetCompletedValue(g->fence);
    if (g->fault == 2 || done == (UINT64)-1 || ID3D11Device_GetDeviceRemovedReason(g->dev) != S_OK) {
        /* (2: the tests', as a driver reset between calls. The device is let go at rf_close, not here: its release
         * took 25 ms on the frame's thread, 2026-10-27, gpu/integ/fault) */
        gm_off(g, 7);
        return 0;
    }
    if (done < g->fv || gm_now_us(g) < g->rest_until) {
        g->stats[GS_BUSY]++;
        return 0;
    }
#endif
    g->nwin = g->ntask = g->nplace = g->npx = g->npre = 0;
    g->cn = cn;
    return 1;
}

/* The answers of the call gm_send sent: 0 when they're in (mapped, for gm_answer), -1 if they're not within
 * GM_WAIT_US or the device failed (the call is then the CPU's). Never longer than that. */
static int gm_wait(GmCtx *g)
{
#ifdef _WIN32
    LARGE_INTEGER due;
    due.QuadPart = -(LONGLONG)gm_wait_most * 10;  /* (relative, in 100 ns) */
    int lost = g->fault == 4;  /* (the tests': removed while the call is on it, as a driver reset mid-call; the
                                * fence then reads UINT64_MAX at once) */
    int late = g->fault == 1 || !SetWaitableTimer(g->timer, &due, 0, 0, 0, FALSE);
    while (!late && !lost && ID3D11Fence_GetCompletedValue(g->fence) < g->fv) {
        HANDLE hs[2] = {g->done, g->timer};
        DWORD w = WaitForMultipleObjects(2, hs, FALSE, (DWORD)(gm_wait_most / 1000 + 50));
        if (w != WAIT_OBJECT_0)
            late = 1;
    }
    CancelWaitableTimer(g->timer);
    double us = gm_now_us(g) - g->t_sent;
    g->stats[GS_WAIT_US] += (long long)us;
    if ((long long)us > g->stats[GS_WAIT_MAX_US])
        g->stats[GS_WAIT_MAX_US] = (long long)us;
    if (lost || ID3D11Device_GetDeviceRemovedReason(g->dev) != S_OK)
        return gm_off(g, 7);
    if (late) {
        g->stats[GS_LATE]++;
        g->rest_until = gm_now_us(g) + (double)gm_rest;
        return -1;
    }
    /* (the fence is past the copies, so these don't wait; with DO_NOT_WAIT the runtime, which doesn't follow the
     * fence, said WAS_STILL_DRAWING) */
    D3D11_MAPPED_SUBRESOURCE a, b;
    if (FAILED(ID3D11DeviceContext_Map(g->ctx, (ID3D11Resource *)g->s_rec, 0, D3D11_MAP_READ, 0, &a))) {
        g->stats[GS_LATE]++;
        return -1;
    }
    if (FAILED(ID3D11DeviceContext_Map(g->ctx, (ID3D11Resource *)g->s_cand, 0, D3D11_MAP_READ, 0, &b))) {
        ID3D11DeviceContext_Unmap(g->ctx, (ID3D11Resource *)g->s_rec, 0);
        g->stats[GS_LATE]++;
        return -1;
    }
    g->got_rec = (const unsigned *)a.pData;
    memcpy(g->last_rec, g->got_rec, 16 * (size_t)g->ntask);
    g->got_cand = (const unsigned *)b.pData;
    g->stats[GS_GPU]++;
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj;
    UINT64 ts[5];
    int ok = g->q_disjoint && ID3D11DeviceContext_GetData(g->ctx, (ID3D11Asynchronous *)g->q_disjoint, &dj, sizeof dj,
                                                        D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK &&
             !dj.Disjoint && dj.Frequency;
    for (int i = 0; ok && i < 5; i++)
        ok = ID3D11DeviceContext_GetData(g->ctx, (ID3D11Asynchronous *)g->q_t[i], ts + i, sizeof ts[i],
                                         D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK;
    if (ok) {
        long long gus = (long long)((double)(ts[4] - ts[0]) * 1e6 / (double)dj.Frequency);
        g->stats[GS_GPU_US] += gus;
        if (gus > g->stats[GS_GPU_MAX_US])
            g->stats[GS_GPU_MAX_US] = gus;
        for (int i = 0; i < 4; i++)
            g->stats[GS_PASS_US + i] += (long long)((double)(ts[i + 1] - ts[i]) * 1e6 / (double)dj.Frequency);
    }
    return 0;
#else
    (void)g;
    return -1;
#endif
}

/* After gm_wait's 0, once the answers have been read. */
static void gm_done(GmCtx *g)
{
#ifdef _WIN32
    if (g->got_rec) {
        ID3D11DeviceContext_Unmap(g->ctx, (ID3D11Resource *)g->s_rec, 0);
        ID3D11DeviceContext_Unmap(g->ctx, (ID3D11Resource *)g->s_cand, 0);
    }
#endif
    g->got_rec = g->got_cand = 0;
}

/* Task k's best (`sub` 1: of its near sub-rectangle's places, else of all): the first highest float32 score row by
 * row and its place (in the task's window), as rf_best gives it; 1, or 0 where the GPU hasn't it (too many places
 * within the band: the caller matches it itself). */
static int gm_answer(GmCtx *g, int k, int sub, float *score, int *x, int *y)
{
    const unsigned *r = g->got_rec + 4 * k;
    unsigned n = r[2 + sub];
    if (n == 0 || n > GM_CAND) {
        g->stats[GS_CPU_PAIRS]++;
        g->stats[n ? GS_MANY : GS_NONE]++;
        return 0;
    }
    const GmTaskGpu *t = g->task + k;
    const Masked *p = g->tpls[t->pic];
    const unsigned *c = g->got_cand + ((size_t)(2 * k + sub) * GM_CAND) * 6;
    float best = 0.0f;
    unsigned at = 0xffffffffu;
    for (unsigned i = 0; i < n; i++, c += 6) {
        float s = gm_score(p, c[1], c + 2, c[5]);
        if (at == 0xffffffffu || s > best || (s == best && c[0] < at)) {
            best = s;
            at = c[0];
        }
    }
    *score = best;
    *x = (int)(at % t->rw);
    *y = (int)(at / t->rw);
    g->stats[GS_PAIRS]++;
    return 1;
}

/* --- rf_*'s use of it (native.c: rf_start, rf_window, rf_next, rf_end) */

/* Window `w` (x0, y0, x1, y1 in the frame) of the call's: its index, the same for the same rectangle in a call; -1
 * past the limits. */
static int gm_rf_window(GmCtx *g, const RfBatch *b, const int *w)
{
    for (int i = 0; i < g->nwin; i++)
        if (memcmp(g->win_rect[i], w, sizeof g->win_rect[i]) == 0)
            return i;
    int k = gm_add_window(g, b->frame + (long long)w[1] * b->frow + (long long)w[0] * b->cn, b->frow, w[2] - w[0],
                          w[3] - w[1]);
    if (k >= 0)
        memcpy(g->win_rect[k], w, sizeof g->win_rect[k]);
    return k;
}

/* The call's jobs to the GPU (rf_start): each job's full window with every picture of its lists that fits it and the
 * GPU takes, its near window a sub-rectangle of those places where it lies inside the full one (else a window of its
 * own); a window only with a picture to match there, once a call for two jobs' alike. 0: sent; -1: the CPU's (not
 * ready, busy, past the limits). */
static int gm_rf_send(GmCtx *g, const RfBatch *b)
{
    if (!gm_ready(g, b->cn))
        return -1;
    g->stats[GS_CALLS]++;
    for (int i = 0; i < b->count; i++) {
        const RfJob *j = b->jobs + i;
        g->job_at[i] = g->ntask;
        g->job_n[i] = 0;
        g->job_sub[i] = 0;
        if (rf_empty(j->wfull))
            continue;
        const int *f = j->wfull, *nw = j->wnear;
        int has_near = !rf_empty(nw);
        int sub = has_near && nw[0] >= f[0] && nw[1] >= f[1] && nw[2] <= f[2] && nw[3] <= f[3];
        g->job_sub[i] = sub;
        for (int side = 1; side >= 0; side--) {  /* the full window's tasks, then a near window's own */
            const int *w = side ? f : nw;
            if (!side && (!has_near || sub))
                continue;
            int win = -1, ww = w[2] - w[0], wh = w[3] - w[1];  /* (the window added with its first task) */
            for (int k = 0; k < j->first_n + j->next_n; k++) {
                int t = k < j->first_n ? b->lists[j->first_at + k] : b->lists[j->next_at + k - j->first_n];
                const Masked *p = g->tpls[t];
                if (!g->takes[t] || p->th > wh || p->tw > ww)
                    continue;
                int seen = 0;
                for (int q = g->job_at[i]; q < g->ntask && !seen; q++)
                    seen = (int)g->task[q].pic == t && g->task_side[q] == side;
                if (seen)
                    continue;
                int nx0 = 0, ny0 = 0, nx1 = 0, ny1 = 0;
                if (side && sub && p->th <= nw[3] - nw[1] && p->tw <= nw[2] - nw[0]) {
                    nx0 = nw[0] - f[0];
                    ny0 = nw[1] - f[1];
                    nx1 = nx0 + (nw[2] - nw[0]) - p->tw + 1;
                    ny1 = ny0 + (nw[3] - nw[1]) - p->th + 1;
                }
                if (win < 0 && (win = gm_rf_window(g, b, w)) < 0) {
                    g->stats[GS_FULL]++;
                    return -1;
                }
                int q = gm_add_task(g, win, t, nx0, ny0, nx1, ny1);
                if (q < 0) {
                    g->stats[GS_FULL]++;
                    return -1;
                }
                g->task_side[q] = (unsigned char)side;
                g->job_n[i]++;
            }
        }
    }
    return gm_send(g);
}

/* Picture `pic`'s best over job i's window `side` (0 near, 1 full) as rf_best gives it (x, y in that window), from
 * the GPU's answers: 1, or 0 where it hasn't one (the caller matches it itself). */
static int gm_rf_best(GmCtx *g, const RfBatch *b, int i, int side, int pic, float *score, int *x, int *y)
{
    int want = side == 1 || g->job_sub[i];  /* the task's side: the full window's (its near sub-rectangle for near) */
    for (int q = g->job_at[i]; q < g->job_at[i] + g->job_n[i]; q++) {
        if ((int)g->task[q].pic != pic || g->task_side[q] != want)
            continue;
        int sub = side == 0 && g->job_sub[i];
        if (!gm_answer(g, q, sub, score, x, y))
            return 0;
        if (sub) {
            const RfJob *j = b->jobs + i;
            *x -= j->wnear[0] - j->wfull[0];
            *y -= j->wnear[1] - j->wfull[1];
        }
        return 1;
    }
    g->stats[GS_CPU_PAIRS]++;
    return 0;
}

/* --- what rf_* and the tests see of it */


/* Context ctx's GPU backend: 0 none (gm_on off when it was opened), 1 being made, 2 in use, 3 off (couldn't be made,
 * or lost: stats[GS_WHY] says why); `stats` (GS_N long longs, or NULL): calls asked, sent and answered by it, late
 * (past GM_WAIT_US: the CPU's), busy (one before still on it), past its limits, pairs answered, pairs the CPU matched,
 * the waits' total and longest us, tasks and places sent, why it's off. */
EXPORT int rf_gpu(void *ctx, long long *stats)
{
    RfCtx *c = (RfCtx *)ctx;
    GmCtx *g = c ? c->gpu : 0;
    if (!g)
        return 0;
    if (stats)
        memcpy(stats, g->stats, sizeof g->stats);
    return g->state == GM_READY ? 2 : g->state == GM_OFF ? 3 : 1;
}

/* The tests': `what` 1, ctx's next waits come back late (the CPU's); 2, its device lost at the next call (as a driver
 * reset between calls); 4, lost while the next call is on it (a reset mid-call: the wait sees it at once); 5, an
 * error sending the next call; 0, none. With ctx NULL and 3: the next contexts' device can't be made. */
EXPORT void rf_gpu_fault(void *ctx, int what)
{
    RfCtx *c = (RfCtx *)ctx;
    if (!c)
        gm_fault_open = what;
    else if (c->gpu)
        c->gpu->fault = what;
}

/* The tests' and benchmarks': each of `ntask` (window, picture) pairs' best over its window as rf_best gives it (the
 * first highest score row by row, its place), worked out on ctx's GPU in one submission: windows imgs[i], ihs[i] x
 * iws[i], rows irows[i] bytes apart, cn channels; pair k picture task_pic[k] (an index into rf_open's) over window
 * task_win[k]; how[k] 1 where the GPU answered, 0 where the CPU matched it (more candidates than GM_CAND, a picture
 * the GPU doesn't take). 0, or -1 if the GPU couldn't (not in use, past its limits, late: nothing written). */
EXPORT int gm_bests(void *ctx, int nwin, const unsigned char *const *imgs, const int *ihs, const int *iws,
                    const long long *irows, int cn, int ntask, const int *task_win, const int *task_pic, float *scores,
                    int *xs, int *ys, int *how)
{
    RfCtx *c = (RfCtx *)ctx;
    GmCtx *g = c ? c->gpu : 0;
    if (!g || c->running || !gm_ready(g, cn))
        return -1;
    for (int i = 0; i < nwin; i++)
        if (gm_add_window(g, imgs[i], irows[i], iws[i], ihs[i]) < 0)
            return -1;
    int *task_of = (int *)malloc(sizeof(int) * (size_t)(ntask + 1));
    if (!task_of)
        return -1;
    for (int k = 0; k < ntask; k++) {
        int t = task_pic[k];
        const Masked *p = g->tpls[t];
        task_of[k] = t < 0 || t >= g->ntpl || p->cn != cn || !g->takes[t] ? -1 :
                     gm_add_task(g, task_win[k], t, 0, 0, 0, 0);
    }
    int ok = gm_send(g) == 0 && gm_wait(g) == 0;
    for (int k = 0; ok && k < ntask; k++) {
        how[k] = task_of[k] >= 0 && gm_answer(g, task_of[k], 0, scores + k, xs + k, ys + k);
        if (!how[k]) {
            int w = task_win[k];
            const Masked *p = g->tpls[task_pic[k]];
            Picture im;
            if (p->cn != cn || ihs[w] < p->th || iws[w] < p->tw ||
                load(&im, imgs[w], ihs[w], iws[w], irows[w], cn, ntt_wanted(p, ihs[w], iws[w])) != 0) {
                ok = 0;
                break;
            }
            rf_best(&im, p, scores + k, xs + k, ys + k);
        }
    }
    if (g->got_rec)
        gm_done(g);
    free(task_of);
    return ok ? 0 : -1;
}

#ifdef _WIN32
/* The s~ of task k's places of the last submission (its sc), into out. */
static int gm_read_sc(GmCtx *g, int k, float *out)
{
    unsigned places = g->task[k].rw * g->task[k].rh;
    ID3D11Buffer *s = gm_buffer(g, 8 * (size_t)places, 0, 0, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ, 0, 0);
    if (!s)
        return -1;
    D3D11_BOX box = {8 * g->task[k].place_off, 0, 0, 8 * (g->task[k].place_off + places), 1, 1};
    ID3D11DeviceContext_CopySubresourceRegion(g->ctx, (ID3D11Resource *)s, 0, 0, 0, 0, (ID3D11Resource *)g->sc, 0,
                                              &box);
    D3D11_MAPPED_SUBRESOURCE m;
    int ok = SUCCEEDED(ID3D11DeviceContext_Map(g->ctx, (ID3D11Resource *)s, 0, D3D11_MAP_READ, 0, &m));
    if (ok) {
        const unsigned *v = (const unsigned *)m.pData;
        for (unsigned q = 0; q < places; q++) {
            unsigned key = v[2 * q + 1], u = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
            memcpy(out + q, &u, 4);
        }
        ID3D11DeviceContext_Unmap(g->ctx, (ID3D11Resource *)s, 0);
    }
    ID3D11Buffer_Release(s);
    return ok ? 0 : -1;
}
#endif

/* The tests': picture `pic`'s approximate score s~ at every place of the image as the GPU has it (`out`: (ih - th + 1)
 * x (iw - tw + 1) floats), to check GM_EPS; with img NULL, task `pic` of the last submission's (gm_bests'), and its
 * record (highest keys, candidates) into out after them. 0, or -1 if it couldn't. */
EXPORT int gm_approx(void *ctx, const unsigned char *img, int ih, int iw, long long irow, int cn, int pic, float *out)
{
#ifdef _WIN32
    RfCtx *c = (RfCtx *)ctx;
    GmCtx *g = c ? c->gpu : 0;
    if (!g || c->running)
        return -1;
    if (!img) {
        if (pic < 0 || pic >= g->ntask || gm_read_sc(g, pic, out) != 0)
            return -1;
        memcpy(out + g->task[pic].rw * g->task[pic].rh, g->last_rec + 4 * pic, 16);
        return 0;
    }
    if (pic < 0 || pic >= g->ntpl || !g->takes[pic] || !gm_ready(g, cn) ||
        g->tpls[pic]->cn != cn || gm_add_window(g, img, irow, iw, ih) < 0 || gm_add_task(g, 0, pic, 0, 0, 0, 0) < 0)
        return -1;
    if (gm_send(g) != 0 || gm_wait(g) != 0)
        return -1;
    gm_done(g);
    return gm_read_sc(g, 0, out);
#else
    (void)ctx; (void)img; (void)ih; (void)iw; (void)irow; (void)cn; (void)pic; (void)out;
    return -1;
#endif
}
