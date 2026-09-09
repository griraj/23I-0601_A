/* task2_dgemv.c — Assignment #1, Task 2: BLAS Level 2 — DGEMV
 * Student: Gri Raj | Roll: 23I-0601 | Section: A
 *
 * Computes y = alpha*A*x + beta*y, A is m x n, row-major.
 * Decomposition: contiguous blocks of OUTPUT ROWS across threads
 * (embarrassingly parallel — each y[i] written by exactly one thread).
 * V1 = serial, V2 = pthreads, V3 = pthreads + CPU affinity,
 * BLAS = OpenBLAS cblas_dgemv reference.
 *
 * Usage:
 *   ./task2_dgemv --version V1|V2|V3|BLAS [--threads N] [--trials N] [--size N] [--csv file]
 */
#include "common.h"
#include <cblas.h>

typedef struct { int tid, threads, m, n, affinity; double alpha, beta; double *A, *x, *y, *yin; } job_t;

static void *worker(void *p) {
    job_t *q = (job_t *)p;
    if (q->affinity) pin_cpu(q->tid);
    int lo, hi; split_range(q->tid, q->threads, q->m, &lo, &hi);
    for (int i = lo; i < hi; i++) {
        double s = 0;
        for (int j = 0; j < q->n; j++) s += q->A[(size_t)i * q->n + j] * q->x[j];
        q->y[i] = q->alpha * s + q->beta * q->yin[i];
    }
    return NULL;
}

static void serial(job_t *q) { q->tid = 0; q->threads = 1; q->affinity = 0; worker(q); }

static void parallel(job_t *base, int affinity) {
    pthread_t *th = calloc(base->threads, sizeof *th);
    job_t *q = calloc(base->threads, sizeof *q);
    for (int t = 0; t < base->threads; t++) {
        q[t] = *base; q[t].tid = t; q[t].affinity = affinity;
        pthread_create(&th[t], NULL, worker, &q[t]);
    }
    for (int t = 0; t < base->threads; t++) pthread_join(th[t], NULL);
    free(q); free(th);
}

static void blas_ref(job_t *q) {
    memcpy(q->y, q->yin, (size_t)q->m * sizeof(double));
    cblas_dgemv(CblasRowMajor, CblasNoTrans, q->m, q->n, q->alpha, q->A, q->n, q->x, 1, q->beta, q->y, 1);
}

int main(int argc, char **argv) {
    int threads = 1, trials = 5, size = 1600;
    char *ver = "V1", *csv = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--version") && i + 1 < argc) ver = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trials") && i + 1 < argc) trials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else { fprintf(stderr, "Usage: %s --version V1|V2|V3|BLAS [--threads N] [--trials N] [--size N] [--csv file]\n", argv[0]); return 2; }
    }
    if (threads < 1) { fprintf(stderr, "threads must be >= 1\n"); return 2; }

    job_t q = { .m = size, .n = size, .alpha = 1.25, .beta = 0.75, .threads = threads };
    uint64_t seed = 42;
    q.A = malloc((size_t)size * size * sizeof *q.A);
    q.x = malloc((size_t)size * sizeof *q.x);
    q.yin = malloc((size_t)size * sizeof *q.yin);
    q.y = malloc((size_t)size * sizeof *q.y);
    fill(q.A, (size_t)size * size, &seed);
    fill(q.x, size, &seed);
    fill(q.yin, size, &seed);

    double *ref = malloc((size_t)size * sizeof *ref);
    blas_ref(&q);
    memcpy(ref, q.y, (size_t)size * sizeof(double));

    FILE *out = csv_open(csv);
    if (!out) return 1;

    for (int r = 1; r <= trials; r++) {
        double t0 = now_ms();
        if (!strcmp(ver, "V1")) serial(&q);
        else if (!strcmp(ver, "BLAS")) blas_ref(&q);
        else if (!strcmp(ver, "V2")) parallel(&q, 0);
        else if (!strcmp(ver, "V3")) parallel(&q, 1);
        else { fprintf(stderr, "Unknown version '%s'\n", ver); return 2; }
        double dt = now_ms() - t0;

        double err = maxerr(ref, q.y, size);
        int chunk = (size + threads - 1) / threads; /* rows per thread */
        csv_row(out, 2, ver, threads, chunk, r, dt, err);
    }

    if (csv) fclose(out);
    free(q.A); free(q.x); free(q.y); free(q.yin); free(ref);
    return 0;
}
