/* task3_dgemm.c — Assignment #1, Task 3: BLAS Level 3 — DGEMM
 * Student: Gri Raj | Roll: 23I-0601 | Section: A
 *
 * Computes C = alpha*A*B + beta*C, all n x n, row-major.
 * V1      = serial, naive i-j-k loop order (poor locality on B).
 * V1-ikj  = serial, i-k-j loop order (row-major-friendly: inner loop
 *           strides contiguously through B's row and C's row).
 * V2      = pthreads, row-blocks of C, each thread uses ikj internally.
 * V3      = V2 + CPU affinity.
 * BLAS    = OpenBLAS cblas_dgemm reference.
 *
 * Usage:
 *   ./task3_dgemm --version V1|V1-ikj|V2|V3|BLAS [--threads N] [--trials N] [--size N] [--csv file]
 */
#include "common.h"
#include <cblas.h>

typedef struct { int tid, threads, m, n, k, affinity, ikj; double alpha, beta; double *A, *B, *C, *Cin; } job_t;

static void *worker(void *p) {
    job_t *q = (job_t *)p;
    if (q->affinity) pin_cpu(q->tid);
    int lo, hi; split_range(q->tid, q->threads, q->m, &lo, &hi);

    /* start each thread's C rows from beta*Cin */
    for (int i = lo; i < hi; i++)
        for (int j = 0; j < q->n; j++)
            q->C[(size_t)i * q->n + j] = q->beta * q->Cin[(size_t)i * q->n + j];

    if (q->ikj) {
        for (int i = lo; i < hi; i++)
            for (int z = 0; z < q->k; z++) {
                double a = q->alpha * q->A[(size_t)i * q->k + z];
                const double *b_row = &q->B[(size_t)z * q->n];
                double *c_row = &q->C[(size_t)i * q->n];
                for (int j = 0; j < q->n; j++) c_row[j] += a * b_row[j];
            }
    } else {
        for (int i = lo; i < hi; i++)
            for (int j = 0; j < q->n; j++) {
                double s = 0;
                for (int z = 0; z < q->k; z++) s += q->A[(size_t)i * q->k + z] * q->B[(size_t)z * q->n + j];
                q->C[(size_t)i * q->n + j] += q->alpha * s;
            }
    }
    return NULL;
}

static void serial(job_t *q, int ikj) { q->tid = 0; q->threads = 1; q->affinity = 0; q->ikj = ikj; worker(q); }

static void parallel(job_t *base, int affinity) {
    pthread_t *th = calloc(base->threads, sizeof *th);
    job_t *q = calloc(base->threads, sizeof *q);
    for (int t = 0; t < base->threads; t++) {
        q[t] = *base; q[t].tid = t; q[t].affinity = affinity; q[t].ikj = 1; /* V2/V3 always use the faster ikj order internally */
        pthread_create(&th[t], NULL, worker, &q[t]);
    }
    for (int t = 0; t < base->threads; t++) pthread_join(th[t], NULL);
    free(q); free(th);
}

static void blas_ref(job_t *q) {
    memcpy(q->C, q->Cin, (size_t)q->m * q->n * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, q->m, q->n, q->k,
                q->alpha, q->A, q->k, q->B, q->n, q->beta, q->C, q->n);
}

int main(int argc, char **argv) {
    int threads = 1, trials = 5, size = 384;
    char *ver = "V1", *csv = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--version") && i + 1 < argc) ver = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trials") && i + 1 < argc) trials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else { fprintf(stderr, "Usage: %s --version V1|V1-ikj|V2|V3|BLAS [--threads N] [--trials N] [--size N] [--csv file]\n", argv[0]); return 2; }
    }
    if (threads < 1) { fprintf(stderr, "threads must be >= 1\n"); return 2; }

    job_t q = { .m = size, .n = size, .k = size, .alpha = 1.25, .beta = 0.75, .threads = threads };
    uint64_t seed = 42;
    q.A = malloc((size_t)size * size * sizeof *q.A);
    q.B = malloc((size_t)size * size * sizeof *q.B);
    q.C = malloc((size_t)size * size * sizeof *q.C);
    q.Cin = malloc((size_t)size * size * sizeof *q.Cin);
    fill(q.A, (size_t)size * size, &seed);
    fill(q.B, (size_t)size * size, &seed);
    fill(q.Cin, (size_t)size * size, &seed);

    double *ref = malloc((size_t)size * size * sizeof *ref);
    blas_ref(&q);
    memcpy(ref, q.C, (size_t)size * size * sizeof(double));

    FILE *out = csv_open(csv);
    if (!out) return 1;

    for (int r = 1; r <= trials; r++) {
        double t0 = now_ms();
        if (!strcmp(ver, "V1")) serial(&q, 0);
        else if (!strcmp(ver, "V1-ikj")) serial(&q, 1);
        else if (!strcmp(ver, "BLAS")) blas_ref(&q);
        else if (!strcmp(ver, "V2")) parallel(&q, 0);
        else if (!strcmp(ver, "V3")) parallel(&q, 1);
        else { fprintf(stderr, "Unknown version '%s'\n", ver); return 2; }
        double dt = now_ms() - t0;

        double err = maxerr(ref, q.C, (size_t)size * size);
        int chunk = (size + threads - 1) / threads; /* C rows per thread */
        csv_row(out, 3, ver, threads, chunk, r, dt, err);
    }

    if (csv) fclose(out);
    free(q.A); free(q.B); free(q.C); free(q.Cin); free(ref);
    return 0;
}
