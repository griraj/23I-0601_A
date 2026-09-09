# Assignment 1 - Parallel and Distributed Computing

Prepared for Gri Raj, roll number 23I-0601, Section A. Do not claim a result
that you did not collect on the target Linux machine.

## Contents

- `src/common.h` — shared timing, RNG, affinity, and error-checking helpers used by every task.
- `src/task1_daxpy.c` — Task 1 (BLAS Level 1: DAXPY): V1 serial, V2 pthreads, V3 pthreads+affinity, BLAS reference.
- `src/task2_dgemv.c` — Task 2 (BLAS Level 2: DGEMV): V1/V2/V3/BLAS, row-block decomposition.
- `src/task3_dgemm.c` — Task 3 (BLAS Level 3: DGEMM): V1 (ijk), V1-ikj, V2/V3 (row-blocks, ikj internally), BLAS.
- `src/task4_spmv.c` — Task 4 (Sparse Matrix-Vector, CSR): V1/V2/V3, Matrix Market reader + synthetic banded fallback, no BLAS comparison (no CBLAS sparse routine used).
- `run_experiments.sh` — five-trial benchmark sweep across thread counts, OpenBLAS fixed at one internal thread.
- `summarize_results.py` — builds `results/task5.csv` (average, minimum, speedup, efficiency) from the raw per-task CSVs.
- `report.pdf` — report with implementation and analysis framework. Complete its measured-results placeholders after running the experiments.

### What changed from the original single-file version

1. **Split into one source file per task** (`src/task1_daxpy.c` ... `src/task4_spmv.c`) plus a shared `src/common.h`, per the submission guideline of clearly labeled per-task sources. Behavior is unchanged except for the two fixes below.
2. **Fixed a real bug in `summarize_results.py`**: the old baseline-selection line matched `version in ('V1', 'V1-ikj')`, which let Task 3's `V1-ikj` row silently overwrite the true `V1` serial baseline (non-deterministically, depending on file/dict iteration order). This produced an impossible <1.0 "speedup" for `V1` itself and a self-referential 1.0 for `V1-ikj`. Fixed to match `version == 'V1'` only, so T1 is always the plain serial baseline as the report text already claims.
3. **Increased default problem sizes for Task 1 (DAXPY) and Task 4 (SpMV)** in `run_experiments.sh` (20,000 → 20,000,000 and 2,000,000 respectively). At the original size, total runtime was well under a millisecond, so per-trial thread-creation overhead completely dominated the pthreads timings and no real scaling could be observed. **You must re-run `make run` to regenerate `results/*.csv` and `results/task5.csv` with these new sizes** — the old CSVs are gone; don't submit stale/mismatched data.
4. **Removed the compiled binary from the submission folder** (the old zip shipped `pdca_bench`, which violates the "no binaries" submission rule).

## Linux prerequisites and build

Ubuntu/Debian example:

```bash
sudo apt-get install build-essential libopenblas-dev linux-tools-common linux-tools-$(uname -r)
make
```

This builds four separate binaries: `task1_daxpy`, `task2_dgemv`, `task3_dgemm`, `task4_spmv`.

Documented compilation command (per-task, same flags as before):

```bash
gcc -O3 -march=native -std=c11 -Wall -Wextra -Wpedantic -pthread -g -fno-omit-frame-pointer -Isrc src/task1_daxpy.c -o task1_daxpy -lopenblas -lm -pthread
# (repeat for task2_dgemv.c, task3_dgemm.c, task4_spmv.c)
```

## Reproduce measurements

Run on the intended Linux system:

```bash
lscpu | tee results/lscpu.txt
export OPENBLAS_NUM_THREADS=1 OPENBLAS_DYNAMIC=0
make run
```

`run_experiments.sh` tests powers of two through `nproc`, runs five trials per configuration, and writes Tasks 1-4. It then derives Task 5 from the raw observations via `summarize_results.py`. To profile a representative configuration:

```bash
perf stat -r 5 -e task-clock,cycles,instructions,cache-references,cache-misses,context-switches,cpu-migrations \
  ./task3_dgemm --version V3 --threads 4 --trials 1 --size 384
```

If an event is unavailable, record `NA`, copy perf's reason into the report, and do not substitute an estimate.

`V3` uses static contiguous row/vector blocks and pins logical worker `t` to logical CPU `t mod online_CPUs` (set from *inside* the worker thread via `pthread_setaffinity_np`, so pinning happens regardless of `pthread_create` ordering). It has no shared output writes and uses `pthread_join` as its completion synchronization.

**Task 4 / Matrix Market data:** check whether your course provided a specific `.mtx` file for this task — the assignment text says "convert the *provided* Matrix Market data." If so, place it under `data/` and run with `MTX_FILE=data/yourfile.mtx bash run_experiments.sh` (or pass `--matrix data/yourfile.mtx` directly to `./task4_spmv`). Without a provided file, the program falls back to a deterministic synthetic banded CSR matrix so it's always runnable — but a banded matrix has a very regular sparsity pattern, so your "load imbalance" / "cache miss" discussion will be much more meaningful against a real irregular matrix if one was supplied.

## Correctness

Tasks 1-3 compare every non-BLAS run against OpenBLAS and write the maximum relative error in each raw CSV row. Acceptance is `<= 1e-12`; relative validation is appropriate because mathematically equivalent floating-point summation orders can differ in absolute magnitude. Task 4 validates against an independently calculated serial CSR reference.

## Submission check

After replacing the identity fields and collecting fresh data (see "What changed" above — you must re-run experiments), update the report's machine table and results figures, then run `make clean`. Ensure the ZIP contains only source (`src/`), `README.md`, `Makefile`, `run_experiments.sh`, `summarize_results.py`, `results/*.csv`, `report.pdf`, and any required Matrix Market input — **no binaries, objects, or profiling output.**
