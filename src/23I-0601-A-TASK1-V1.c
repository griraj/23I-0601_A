/* task1_daxpy.c — Assignment #1, Task 1: BLAS Level 1 — DAXPY
 * Student: Gri Raj | Roll: 23I-0601 | Section: A
 *
 * Computes y = alpha*x + y.
 * V1 = serial, V2 = pthreads (default OS scheduling), V3 = pthreads +
 * CPU affinity, BLAS = OpenBLAS cblas_daxpy reference.
 *
 * Usage:
 *   ./task1_daxpy --version V1|V2|V3|BLAS [--threads N] [--trials N] [--size N] [--csv file]
 */
#include "common.h"
#include <cblas.h>

typedef struct { int tid, threads, n, affinity; double alpha; double *x, *y, *yin; } job_t;

static void *worker(void *p) {
    job_t *q = (job_t *)p;
    if (q->affinity) pin_cpu(q->tid); /* pin THIS thread before doing any work */
    int lo, hi; split_range(q->tid, q->threads, q->n, &lo, &hi);
    for (int i = lo; i < hi; i++) q->y[i] = q->alpha * q->x[i] + q->yin[i];
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

static void blas_ref(job_t *q) { cblas_daxpy(q->n, q->alpha, q->x, 1, q->y, 1); }

int main(int argc, char **argv) {
    int threads = 1, trials = 5, size = 20000000; /* larger N so real compute time dominates thread-creation overhead */
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

    job_t q = { .n = size, .alpha = 1.25, .threads = threads };
    uint64_t seed = 42;
    q.x = malloc((size_t)size * sizeof *q.x);
    q.yin = malloc((size_t)size * sizeof *q.yin);
    q.y = malloc((size_t)size * sizeof *q.y);
    fill(q.x, size, &seed);
    fill(q.yin, size, &seed);

    /* Independent reference for correctness checking: computed once via BLAS. */
    double *ref = malloc((size_t)size * sizeof *ref);
    memcpy(q.y, q.yin, (size_t)size * sizeof(double));
    blas_ref(&q);
    memcpy(ref, q.y, (size_t)size * sizeof(double));

    FILE *out = csv_open(csv);
    if (!out) return 1;

    for (int r = 1; r <= trials; r++) {
        memcpy(q.y, q.yin, (size_t)size * sizeof(double)); /* reset output each trial */
        double t0 = now_ms();
        if (!strcmp(ver, "V1")) serial(&q);
        else if (!strcmp(ver, "BLAS")) blas_ref(&q);
        else if (!strcmp(ver, "V2")) parallel(&q, 0);
        else if (!strcmp(ver, "V3")) parallel(&q, 1);
        else { fprintf(stderr, "Unknown version '%s'\n", ver); return 2; }
        double dt = now_ms() - t0;

        double err = maxerr(ref, q.y, size);
        int chunk = (size + threads - 1) / threads; /* elements per thread */
        csv_row(out, 1, ver, threads, chunk, r, dt, err);
    }

    if (csv) fclose(out);
    free(q.x); free(q.y); free(q.yin); free(ref);
    return 0;
}
