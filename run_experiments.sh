#!/usr/bin/env bash
# Reproducible Linux data collection. Uses at least five trials per configuration.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OPENBLAS_DYNAMIC=0
mkdir -p results

for task in 1 2 3 4; do
  printf 'task,version,threads,chunk,trial,time_ms,validation_max_relative_error\n' > "results/task${task}.csv"
done

# Problem sizes: Task 1 (DAXPY) and Task 4 (SpMV) were bumped up from the
# original 20000 -- at that size the whole computation finishes in well
# under a millisecond, so per-trial thread-creation overhead completely
# dominated the pthreads timings and no real scaling could be observed.
# Re-run and confirm these new sizes give you a clean speedup trend before
# writing up Task 5; adjust further if your machine is much faster/slower.
declare -A SIZE=( [1]=20000000 [2]=1600 [3]=384 [4]=2000000 )

# If your course provided a specific Matrix Market file for Task 4, point
# this at it (place the file in data/ first). Leave empty to use the
# synthetic banded CSR fallback.
MTX_FILE="${MTX_FILE:-}"

binary_for() {
  case "$1" in
    1) echo "./task1_daxpy" ;;
    2) echo "./task2_dgemv" ;;
    3) echo "./task3_dgemm" ;;
    4) echo "./task4_spmv" ;;
  esac
}

for ((t=1; t<=$(nproc); t*=2)); do
  for task in 1 2 3 4; do
    file="results/task${task}.csv"
    bin="$(binary_for "$task")"
    versions=(V1 V2 V3 BLAS)
    [[ "$task" == 3 ]] && versions=(V1 V1-ikj V2 V3 BLAS)
    [[ "$task" == 4 ]] && versions=(V1 V2 V3)   # no BLAS comparison for sparse
    for v in "${versions[@]}"; do
      [[ ( "$v" == V1 || "$v" == V1-ikj ) && "$t" != 1 ]] && continue
      [[ "$v" == BLAS && "$t" != 1 ]] && continue
      size="${SIZE[$task]}"
      extra=()
      [[ "$task" == 4 && -n "$MTX_FILE" ]] && extra=(--matrix "$MTX_FILE")
      "$bin" --version "$v" --threads "$t" --trials 5 --size "$size" --csv "$file" "${extra[@]}"
    done
  done
done

python3 summarize_results.py
