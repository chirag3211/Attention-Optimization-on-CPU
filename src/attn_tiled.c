/*
 * attn_tiled.c -- Q3b: cache and register tiling.
 *
 * Both matrix products use one BLIS-style blocked GEMM written from scratch
 * (no BLAS):
 *   - register tile: a MR x NR block of C is accumulated in vector registers
 *     over the whole k loop (MR*NV accumulators, NV B-vectors, 1 broadcast);
 *     every loaded B vector is reused MR times and every broadcast NV times;
 *   - cache blocking: A is packed in MC x KC blocks (L2), B in KC x NC blocks,
 *     and one KC x NR micro-panel of B (16 KB) stays in L1 while the micro-kernel
 *     sweeps down the rows of the A block. Packing also removes the power-of-two
 *     leading dimensions (4096 floats = 16 KB) that would make rows collide in
 *     the same cache sets.
 * The N x N score matrix is still materialised and softmax is a separate
 * (scalar libm expf) pass, as in the naive code.
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

#define KC 128  /* depth of a block: KC x NR micro-panel of B = 16 KB fits L1 */
#define MC 96   /* rows of an A block (multiple of MR): MC x KC = 48 KB, lives in L2 */
#define NC 1024 /* columns of a B block: KC x NC = 512 KB */

/* C (M x Nn) = alpha * A (M x K, row-major) * B   where B(p,j) is read as in pack_B(). */
static void gemm(int M, int Nn, int K, const float *A, long lda, const float *B, long ldb,
                 int transB, float *C, long ldc, float alpha, float *Abuf, float *Bbuf)
{
    for (int jc = 0; jc < Nn; jc += NC) {
        const int nc = Nn - jc < NC ? Nn - jc : NC;
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = K - pc < KC ? K - pc : KC;
            pack_B(B, ldb, transB, pc, kc, jc, nc, Bbuf);
            for (int ic = 0; ic < M; ic += MC) {
                const int mc = M - ic < MC ? M - ic : MC;
                pack_A(A, lda, ic, mc, pc, kc, Abuf);
                for (int jr = 0; jr < nc; jr += NR)
                    for (int ir = 0; ir < mc; ir += MR)
                        ukernel(kc, Abuf + (long)ir * kc, 1, MR, Bbuf + (long)jr * kc,
                                C + (long)(ic + ir) * ldc + jc + jr, ldc,
                                mc - ir < MR ? mc - ir : MR, nc - jr < NR ? nc - jr : NR, pc > 0,
                                alpha);
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *S = malloc((size_t)N * N * sizeof(float));
    float *Abuf = alloc64((size_t)MC * KC * sizeof(float));
    float *Bbuf = alloc64((size_t)KC * NC * sizeof(float));

    /* 1. S = scale * Q K^T */
    gemm(N, N, d, Q, d, K, d, 1, S, N, scale, Abuf, Bbuf);

    /* 2. softmax of each row of S */
    for (int i = 0; i < N; i++) {
        float *row = S + (size_t)i * N;
        float max = row[0];
        for (int j = 1; j < N; j++)
            if (row[j] > max)
                max = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - max);
            sum += row[j];
        }
        const float inv = 1.0f / sum;
        for (int j = 0; j < N; j++)
            row[j] *= inv;
    }

    /* 3. O = S V */
    gemm(N, d, N, S, N, V, d, 0, O, d, 1.0f, Abuf, Bbuf);

    free64(Abuf);
    free64(Bbuf);
    free(S);
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
