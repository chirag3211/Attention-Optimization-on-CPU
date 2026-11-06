/*
 * attn_omp.c -- Q4: attn_best.c parallelised with OpenMP.
 *
 * Rows of the output are independent, so the row blocks of attn_best.c are
 * distributed over threads (schedule(dynamic): cheap, and robust to
 * performance/efficiency cores). K and V are packed once, in parallel, and
 * shared read-only; every thread owns its Q-panel / score-block scratch so
 * there is no write sharing. The block size is reduced for small N so that
 * there are at least 4 blocks per thread. Number of threads: OMP_NUM_THREADS.
 *
 * Keep the driver (main) below unchanged. See attn_naive.c for the input and
 * output format.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- register-tiled FMA micro-kernel (portable: GCC/Clang vector extensions) ---- */
#if defined(__AVX512F__)
#define VL 16 /* floats per vector: 512 bit */
#else
#define VL 8 /* 256 bit: AVX2, or 2 x 128-bit NEON registers */
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
#define MR 6 /* rows of the register tile    */
#else
#define MR 4
#endif
#define NV 2           /* vectors per tile row */
#define NR (NV * VL)   /* columns of the register tile: MR x NR floats live in registers */

typedef float vf __attribute__((vector_size(VL * sizeof(float)), aligned(4)));

#define INLINE static inline __attribute__((always_inline))

/*
 * C[0:mr, 0:nr] (+)= alpha * A * B   over kc terms.
 *   A(r,p) = A[r*rs + p*cs]           (packed panel: rs=1, cs=MR; row-major: rs=lda, cs=1)
 *   B(p,c) = Bp[p*NR + c]             (packed panel, NR contiguous columns per p)
 * The MR x NR accumulator tile stays in vector registers for the whole kc loop.
 */
INLINE void ukernel(int kc, const float *A, long rs, long cs, const float *Bp, float *C, long ldc,
                    int mr, int nr, int acc, float alpha)
{
    vf c[MR][NV];
    for (int r = 0; r < MR; r++)
        for (int v = 0; v < NV; v++)
            c[r][v] = (vf){0};
    for (int p = 0; p < kc; p++) {
        vf b[NV];
        for (int v = 0; v < NV; v++)
            b[v] = *(const vf *)(Bp + (long)p * NR + v * VL);
        for (int r = 0; r < MR; r++) {
            const float a = A[r * rs + p * cs];
            for (int v = 0; v < NV; v++)
                c[r][v] += b[v] * a;
        }
    }
    if (mr == MR && nr == NR) {
        for (int r = 0; r < MR; r++)
            for (int v = 0; v < NV; v++) {
                vf *dst = (vf *)(C + r * ldc + v * VL);
                *dst = acc ? *dst + c[r][v] * alpha : c[r][v] * alpha;
            }
    } else { /* edge tile: spill, copy the valid part */
        float t[MR][NR] __attribute__((aligned(64)));
        for (int r = 0; r < MR; r++)
            for (int v = 0; v < NV; v++)
                *(vf *)(&t[r][v * VL]) = c[r][v];
        for (int r = 0; r < mr; r++)
            for (int j = 0; j < nr; j++) {
                float *dst = C + r * ldc + j;
                *dst = acc ? *dst + alpha * t[r][j] : alpha * t[r][j];
            }
    }
}

/* Pack rows [i0,i0+mc) x cols [p0,p0+kc) of row-major A into MR-row panels: Ap[panel][p][r]. */
static void pack_A(const float *A, long lda, int i0, int mc, int p0, int kc, float *Ap)
{
    for (int ir = 0; ir < mc; ir += MR) {
        const int mr = mc - ir < MR ? mc - ir : MR;
        float *dst = Ap + (long)ir * kc;
        for (int p = 0; p < kc; p++)
            for (int r = 0; r < MR; r++)
                dst[p * MR + r] = r < mr ? A[(long)(i0 + ir + r) * lda + p0 + p] : 0.0f;
    }
}

/*
 * Pack B(p,j), p in [p0,p0+kc), j in [j0,j0+nc), into NR-column panels: Bp[panel][p][c].
 * trans == 0: B(p,j) = B[p*ldb + j];  trans == 1: B(p,j) = B[j*ldb + p]  (B^T of a row-major matrix).
 */
static void pack_B(const float *B, long ldb, int trans, int p0, int kc, int j0, int nc, float *Bp)
{
    for (int jr = 0; jr < nc; jr += NR) {
        const int nr = nc - jr < NR ? nc - jr : NR;
        float *dst = Bp + (long)jr * kc;
        if (!trans) {
            for (int p = 0; p < kc; p++) {
                for (int c = 0; c < nr; c++)
                    dst[p * NR + c] = B[(long)(p0 + p) * ldb + j0 + jr + c];
                for (int c = nr; c < NR; c++)
                    dst[p * NR + c] = 0.0f;
            }
        } else {
            for (int c = 0; c < NR; c++)
                for (int p = 0; p < kc; p++)
                    dst[p * NR + c] = c < nr ? B[(long)(j0 + jr + c) * ldb + p0 + p] : 0.0f;
        }
    }
}

static inline long round_up(long x, long m) { return (x + m - 1) / m * m; }
/* portable 64-byte aligned allocation (aligned_alloc does not exist in the Windows C runtime) */
static inline void *alloc64(size_t bytes)
{
    void *raw = malloc(bytes + 64 + sizeof(void *));
    if (!raw) return NULL;
    uintptr_t a = ((uintptr_t)raw + sizeof(void *) + 63) & ~(uintptr_t)63;
    ((void **)a)[-1] = raw;
    return (void *)a;
}
static inline void free64(void *p) { if (p) free(((void **)p)[-1]); }
/* ---- fast vectorised exp for x <= 0 (Cephes polynomial, rel. error ~1e-7) ---- */
typedef int vi __attribute__((vector_size(VL * sizeof(float)), aligned(4)));

INLINE vf vexp(vf x)
{
    x = __builtin_elementwise_max(x, (vf){0} + -87.0f);      /* avoid denormal results */
    const vf t = x * 1.44269504f;                            /* x / ln 2 */
    const vf n = (t + 12582912.0f) - 12582912.0f;            /* round to nearest integer */
    vf r = x - n * 0.693359375f;                             /* x - n ln 2, in two pieces */
    r = r - n * -2.12194440e-4f;
    vf y = (vf){0} + 1.9875691500e-4f;
    y = y * r + 1.3981999507e-3f;
    y = y * r + 8.3334519073e-3f;
    y = y * r + 4.1665795894e-2f;
    y = y * r + 1.6666665459e-1f;
    y = y * r + 5.0000001201e-1f;
    y = y * (r * r) + r + 1.0f;                              /* exp(r) */
    const vi e = (__builtin_convertvector(n, vi) + 127) << 23; /* 2^n built in the exponent bits */
    return y * (vf)e;
}

/* In place: x[j] = exp(x[j] - max(x)) for j < n; returns sum(x). */
static float softmax_row(float *x, int n)
{
    const int nv = n / VL * VL;
    vf mv = (vf){0} + x[0];
    for (int j = 0; j < nv; j += VL)
        mv = __builtin_elementwise_max(mv, *(const vf *)(x + j));
    float m = mv[0];
    for (int l = 1; l < VL; l++)
        m = m > mv[l] ? m : mv[l];
    for (int j = nv; j < n; j++)
        m = m > x[j] ? m : x[j];

    vf sv = {0};
    for (int j = 0; j < nv; j += VL) {
        const vf e = vexp(*(const vf *)(x + j) - m);
        *(vf *)(x + j) = e;
        sv += e;
    }
    float sum = 0.0f;
    for (int l = 0; l < VL; l++)
        sum += sv[l];
    for (int j = nv; j < n; j++) {
        x[j] = expf(x[j] - m);
        sum += x[j];
    }
    return sum;
}

#define DKC 256 /* depth chunk of the Q K^T product   */
#define PKC 128 /* depth chunk of the P V product     */

/* V (N x d) -> panels of NR columns: Vp[panel][j][c], rows j0 <= j < j1 */
static void pack_V(const float *V, int N, int d, int j0, int j1, float *Vp)
{
    const int np = (d + NR - 1) / NR;
    for (int j = j0; j < j1; j++)
        for (int dp = 0; dp < np; dp++)
            for (int c = 0; c < NR; c++)
                Vp[(long)dp * NR * N + (long)j * NR + c] =
                    dp * NR + c < d ? V[(long)j * d + dp * NR + c] : 0.0f;
}

/*
 * Rows [i0, i0+mb) of the output, mb <= MR * (tiles per block):
 *   S_blk = scale * Q_blk K^T      (mb x Npad, stays in L2; K already packed as Kp)
 *   P_blk = exp(S_blk - rowmax)    (vectorised, in place), l = rowsum
 *   O_blk = (P_blk V) / l          (V already packed as Vp)
 * The N x N matrix never exists: only this block of rows does.
 */
static void attn_block(const float *Q, const float *Kp, const float *Vp, float *O, int N, int d,
                       int Npad, int i0, int mb, float scale, float *Qp, float *Sb, float *sums)
{
    const int nt = (mb + MR - 1) / MR;
    pack_A(Q, d, i0, mb, 0, d, Qp);

    for (int pc = 0; pc < d; pc += DKC) {
        const int kc = d - pc < DKC ? d - pc : DKC;
        for (int jp = 0; jp < Npad / NR; jp++)
            for (int t = 0; t < nt; t++)
                ukernel(kc, Qp + (long)t * MR * d + (long)pc * MR, 1, MR,
                        Kp + (long)jp * NR * d + (long)pc * NR, Sb + (long)t * MR * Npad + jp * NR,
                        Npad, MR, NR, pc > 0, scale);
    }

    for (int r = 0; r < mb; r++)
        sums[r] = softmax_row(Sb + (long)r * Npad, N);

    const int np = (d + NR - 1) / NR;
    for (int pc = 0; pc < N; pc += PKC) {
        const int kc = N - pc < PKC ? N - pc : PKC;
        for (int dp = 0; dp < np; dp++)
            for (int t = 0; t < nt; t++)
                ukernel(kc, Sb + (long)t * MR * Npad + pc, Npad, 1,
                        Vp + (long)dp * NR * N + (long)pc * NR, O + (long)(i0 + t * MR) * d + dp * NR,
                        d, mb - t * MR < MR ? mb - t * MR : MR, d - dp * NR < NR ? d - dp * NR : NR,
                        pc > 0, 1.0f);
    }

    for (int r = 0; r < mb; r++) {
        const float inv = 1.0f / sums[r];
        float *o = O + (long)(i0 + r) * d;
        for (int k = 0; k < d; k++)
            o[k] *= inv;
    }
}

/* number of MR-row tiles per block: keep the score block (tiles*MR x Npad floats) <= 512 KB, and
 * leave at least `min_blocks` blocks so that every thread has work */
static int tiles_per_block(int N, int Npad, int min_blocks)
{
    int nt = (int)(131072 / ((long)Npad * MR));
    const int cap = (N + MR * min_blocks - 1) / (MR * min_blocks);
    if (nt > cap) nt = cap;
    return nt < 1 ? 1 : nt;
}

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    const int Npad = (int)round_up(N, NR);
    const int np = (d + NR - 1) / NR;
    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif
    const int nt = tiles_per_block(N, Npad, 4 * T), Br = nt * MR;
    const int nblocks = (N + Br - 1) / Br;

    float *Kp = alloc64((size_t)Npad * d * sizeof(float));
    float *Vp = alloc64((size_t)np * NR * N * sizeof(float));

    #pragma omp parallel
    {
        float *Qp = alloc64((size_t)Br * d * sizeof(float));
        float *Sb = alloc64((size_t)Br * Npad * sizeof(float));
        float *sums = malloc((size_t)Br * sizeof(float));

        #pragma omp for schedule(static)
        for (int jp = 0; jp < Npad / NR; jp++) { /* pack K, one panel per iteration */
            const int j0 = jp * NR;
            pack_B(K, d, 1, 0, d, j0, N - j0 < NR ? N - j0 : NR, Kp + (long)j0 * d);
        }
        #pragma omp for schedule(static)
        for (int c = 0; c < 64; c++) /* pack V in 64 row chunks */
            pack_V(V, N, d, (int)((long)N * c / 64), (int)((long)N * (c + 1) / 64), Vp);
        /* implicit barrier: K and V are fully packed before any block starts */

        #pragma omp for schedule(dynamic, 1)
        for (int b = 0; b < nblocks; b++) {
            const int i0 = b * Br;
            attn_block(Q, Kp, Vp, O, N, d, Npad, i0, N - i0 < Br ? N - i0 : Br, scale, Qp, Sb, sums);
        }
        free64(Qp); free64(Sb); free(sums);
    }
    free64(Kp); free64(Vp);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
