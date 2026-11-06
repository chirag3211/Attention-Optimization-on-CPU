/*
 * attn_fused.c -- Q3c: the three steps fused row by row.
 *
 * For each query row i we compute the N scores, softmax them and multiply by V
 * before moving on to row i+1. Only one row of scores (N floats, 16 KB for
 * N = 4096) is ever live, so it stays in L1 and the N x N matrix is never
 * stored. The inner loops are the vectorisable forms from attn_loops.c; the
 * softmax still uses libm expf. Each row streams all of K^T and V once (from L2).
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

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
void attention(const float *restrict Q, const float *restrict K, const float *restrict V,
               float *restrict O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *Kt = malloc((size_t)N * d * sizeof(float)); /* d x N */
    float *s = malloc((size_t)N * sizeof(float));      /* one row of scores */

    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kt[(size_t)k * N + j] = K[(size_t)j * d + k];

    for (int i = 0; i < N; i++) {
        /* 1. one row of scores */
        for (int j = 0; j < N; j++)
            s[j] = 0.0f;
        for (int k = 0; k < d; k++) {
            const float q = Q[(size_t)i * d + k] * scale;
            const float *restrict kt = Kt + (size_t)k * N;
            for (int j = 0; j < N; j++)
                s[j] += q * kt[j];
        }

        /* 2. softmax numerators; the 1/sum is applied once to the output row */
        float max = s[0];
        for (int j = 1; j < N; j++)
            if (s[j] > max)
                max = s[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            s[j] = expf(s[j] - max);
            sum += s[j];
        }

        /* 3. output row */
        float *restrict o = O + (size_t)i * d;
        for (int k = 0; k < d; k++)
            o[k] = 0.0f;
        for (int j = 0; j < N; j++) {
            const float p = s[j];
            const float *restrict v = V + (size_t)j * d;
            for (int k = 0; k < d; k++)
                o[k] += p * v[k];
        }
        const float inv = 1.0f / sum;
        for (int k = 0; k < d; k++)
            o[k] *= inv;
    }

    free(s);
    free(Kt);
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
