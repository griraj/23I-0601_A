/* task4_spmv.c — Assignment #1, Task 4: Sparse Matrix-Vector Multiplication (CSR)
 * Student: Gri Raj | Roll: 23I-0601 | Section: A
 *
 * Computes y = A*x where A is a sparse m x n matrix in CSR format.
 * V1 = serial, V2 = pthreads (row-blocks), V3 = pthreads + CPU affinity.
 * No BLAS comparison for this task (no CBLAS sparse routine used) — the
 * correctness reference is an independently-computed serial CSR pass.
 *
 * IMPORTANT: if your course provided a specific Matrix Market (.mtx)
 * file for this task, use it via --matrix path/to/file.mtx. Without
 * --matrix, this falls back to a deterministic synthetic banded CSR
 * matrix so the program is always runnable, but a banded matrix has a
 * very regular, cache-friendly sparsity pattern — for a meaningful
 * "load imbalance" and "cache miss" discussion in your report, a real
 * irregular matrix (the provided one) is much more informative.
 *
 * Usage:
 *   ./task4_spmv --version V1|V2|V3 [--threads N] [--trials N] [--size N] [--matrix file] [--csv file]
 */
#include "common.h"

typedef struct { int tid, threads, m, n, affinity; size_t nnz; int *rp, *ci; double *val, *x, *y; } job_t;

static void *worker(void *p) {
    job_t *q = (job_t *)p;
    if (q->affinity) pin_cpu(q->tid);
    int lo, hi; split_range(q->tid, q->threads, q->m, &lo, &hi);
    for (int i = lo; i < hi; i++) {
        double s = 0;
        for (int z = q->rp[i]; z < q->rp[i + 1]; z++) s += q->val[z] * q->x[q->ci[z]];
        q->y[i] = s;
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

/* Deterministic synthetic fallback: banded matrix, half-bandwidth w=3. */
static void csr_banded(job_t *q) {
    int w = 3;
    q->rp = calloc(q->m + 1, sizeof *q->rp);
    for (int i = 0; i < q->m; i++) {
        int a = i - w < 0 ? 0 : i - w;
        int b = i + w >= q->n ? q->n - 1 : i + w;
        q->rp[i + 1] = q->rp[i] + b - a + 1;
    }
    q->nnz = q->rp[q->m];
    q->ci = malloc(q->nnz * sizeof *q->ci);
    q->val = malloc(q->nnz * sizeof *q->val);
    uint64_t s = 9;
    for (int i = 0; i < q->m; i++) {
        int z = q->rp[i];
        for (int j = (i - w < 0 ? 0 : i - w); j <= i + w && j < q->n; j++, z++) {
            q->ci[z] = j;
            q->val[z] = rnd(&s);
        }
    }
}

/* Matrix Market coordinate reader. General matrices only; symmetric
 * entries are mirrored into both (r,c) and (c,r). */
static int csr_mm(job_t *q, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char l[512];
    int sym = 0, m, n; size_t nz;
    if (!fgets(l, sizeof l, f)) { fclose(f); return -1; }
    sym = strstr(l, "symmetric") != 0;
    do { if (!fgets(l, sizeof l, f)) { fclose(f); return -1; } } while (l[0] == '%');
    if (sscanf(l, "%d %d %zu", &m, &n, &nz) != 3) { fclose(f); return -1; }

    size_t cap = nz * (sym ? 2 : 1);
    int *rr = malloc(cap * sizeof *rr), *cc = malloc(cap * sizeof *cc);
    double *vv = malloc(cap * sizeof *vv);
    size_t used = 0;
    for (size_t e = 0; e < nz && fgets(l, sizeof l, f); e++) {
        int r, c; double v;
        if (sscanf(l, "%d %d %lf", &r, &c, &v) != 3) continue;
        rr[used] = --r; cc[used] = --c; vv[used++] = v;
        if (sym && r != c) { rr[used] = c; cc[used] = r; vv[used++] = v; }
    }
    fclose(f);

    q->m = m; q->n = n;
    q->rp = calloc(m + 1, sizeof *q->rp);
    for (size_t e = 0; e < used; e++) if (rr[e] >= 0 && rr[e] < m) q->rp[rr[e] + 1]++;
    for (int i = 0; i < m; i++) q->rp[i + 1] += q->rp[i];
    q->nnz = q->rp[m];
    q->ci = malloc(q->nnz * sizeof *q->ci);
    q->val = malloc(q->nnz * sizeof *q->val);
    int *at = malloc(m * sizeof *at);
    memcpy(at, q->rp, m * sizeof *at);
    for (size_t e = 0; e < used; e++) {
        int r = rr[e];
        int z = at[r]++;
        q->ci[z] = cc[e];
        q->val[z] = vv[e];
    }
    free(at); free(rr); free(cc); free(vv);
    return 0;
}

int main(int argc, char **argv) {
    int threads = 1, trials = 5, size = 20000;
    char *ver = "V1", *csv = NULL, *matrix = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--version") && i + 1 < argc) ver = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trials") && i + 1 < argc) trials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--matrix") && i + 1 < argc) matrix = argv[++i];
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else { fprintf(stderr, "Usage: %s --version V1|V2|V3 [--threads N] [--trials N] [--size N] [--matrix file] [--csv file]\n", argv[0]); return 2; }
    }
    if (threads < 1) { fprintf(stderr, "threads must be >= 1\n"); return 2; }

    job_t q = { .m = size, .n = size, .threads = threads };
    if (!matrix || csr_mm(&q, matrix) != 0) {
        if (matrix) fprintf(stderr, "Warning: could not read '%s', falling back to synthetic banded matrix\n", matrix);
        csr_banded(&q);
    }

    uint64_t seed = 42;
    q.x = malloc((size_t)q.n * sizeof *q.x);
    q.y = malloc((size_t)q.m * sizeof *q.y);
    fill(q.x, q.n, &seed);

    /* Independent reference: plain serial CSR pass (no threading, no affinity). */
    double *ref = malloc((size_t)q.m * sizeof *ref);
    for (int i = 0; i < q.m; i++) {
        double s = 0;
        for (int z = q.rp[i]; z < q.rp[i + 1]; z++) s += q.val[z] * q.x[q.ci[z]];
        ref[i] = s;
    }

    FILE *out = csv_open(csv);
    if (!out) return 1;

    for (int r = 1; r <= trials; r++) {
        double t0 = now_ms();
        if (!strcmp(ver, "V1")) serial(&q);
        else if (!strcmp(ver, "V2")) parallel(&q, 0);
        else if (!strcmp(ver, "V3")) parallel(&q, 1);
        else { fprintf(stderr, "Unknown version '%s' (BLAS not applicable to Task 4)\n", ver); return 2; }
        double dt = now_ms() - t0;

        double err = maxerr(ref, q.y, q.m);
        int chunk = (q.m + threads - 1) / threads; /* rows per thread */
        csv_row(out, 4, ver, threads, chunk, r, dt, err);
    }

    if (csv) fclose(out);
    free(q.x); free(q.y); free(q.rp); free(q.ci); free(q.val); free(ref);
    return 0;
}
