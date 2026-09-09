/* common.h — shared helpers for Assignment #1 (PDC)
 * Student: Gri Raj | Roll: 23I-0601 | Section: A
 *
 * Split out of the original single-file 23I-0601-A-TASKS-V1-V3.c so that
 * each task has its own readable source file, per the submission
 * guideline of clearly labeled per-task sources. Behavior/semantics are
 * unchanged from the original implementation.
 */
#ifndef PDCA_COMMON_H
#define PDCA_COMMON_H

#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Monotonic wall-clock time in milliseconds. */
static inline double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return 1e3 * (double)t.tv_sec + 1e-6 * (double)t.tv_nsec;
}

/* Small fast xorshift-style PRNG (deterministic given a seed) so every
 * version (serial/pthreads/affinity/BLAS) operates on IDENTICAL input
 * data -> correctness comparisons are meaningful. */
static inline double rnd(uint64_t *s) {
    *s = *s * 6364136223846793005ULL + 1;
    return 1.0 + (double)(*s >> 11) * (9.0 / 9007199254740992.0);
}

static inline void fill(double *a, size_t n, uint64_t *s) {
    for (size_t i = 0; i < n; i++) a[i] = rnd(s);
}

/* Pin the calling thread to logical CPU (tid mod online_cpus). */
static inline void pin_cpu(int tid) {
    cpu_set_t set;
    CPU_ZERO(&set);
    int cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    CPU_SET(tid % (cpus > 0 ? cpus : 1), &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

/* Relative infinity-norm-style error: different valid floating-point
 * summation orders are expected to differ slightly in absolute terms
 * for DGEMV/DGEMM/SpMV, so we compare relative to the larger magnitude. */
static inline double maxerr(const double *a, const double *b, size_t n) {
    double e = 0;
    for (size_t i = 0; i < n; i++) {
        double scale = fmax(1.0, fmax(fabs(a[i]), fabs(b[i])));
        double d = fabs(a[i] - b[i]) / scale;
        if (d > e) e = d;
    }
    return e;
}

/* Compute [lo, hi) row/element range for thread `tid` of `nthreads`
 * splitting `total` items into contiguous static blocks. */
static inline void split_range(int tid, int nthreads, int total, int *lo, int *hi) {
    *lo = (int)((long long)tid * total / nthreads);
    *hi = (int)((long long)(tid + 1) * total / nthreads);
}

/* Open CSV in append mode, writing the header if the file is new/empty.
 * Columns match the assignment's minimum requirement plus a validation
 * column: task,version,threads,chunk,trial,time_ms,validation_max_relative_error */
static inline FILE *csv_open(const char *path) {
    FILE *f = path ? fopen(path, "a") : stdout;
    if (!f) { perror("csv_open"); return NULL; }
    if (path) {
        fseek(f, 0, SEEK_END);
        if (ftell(f) == 0)
            fprintf(f, "task,version,threads,chunk,trial,time_ms,validation_max_relative_error\n");
    }
    return f;
}

static inline void csv_row(FILE *f, int task, const char *version, int threads,
                            int chunk, int trial, double time_ms, double err) {
    fprintf(f, "%d,%s,%d,%d,%d,%.6f,%.17g\n", task, version, threads, chunk, trial, time_ms, err);
    printf("Task %d %s trial %d (threads=%d): %.3f ms, max error %.3e\n",
           task, version, trial, threads, time_ms, err);
}

#endif /* PDCA_COMMON_H */
