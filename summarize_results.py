#!/usr/bin/env python3
"""Create Task 5 (results/task5.csv) from the measured task1-4 CSVs.
Never invents benchmark observations -- only aggregates real rows.

BUGFIX (vs. original version): the baseline T1 for each task must always
be the plain serial 'V1' average, per the report's stated methodology
("Use the V1 serial average as T1"). The original code matched on
`version in ('V1', 'V1-ikj')`, which let Task 3's V1-ikj row silently
overwrite the true V1 baseline depending on dict/file iteration order
(non-deterministic!). That produced a wrong, self-referential speedup
of 1.0 for V1-ikj and an impossible <1.0 "speedup" for V1 itself.
Fixed here by matching ONLY on version == 'V1'.
"""
import csv
import glob
import statistics

rows = []
for name in sorted(glob.glob('results/task[1-4].csv')):
    with open(name, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)

groups = {}
for r in rows:
    key = (r['task'], r['version'], r['threads'], r['chunk'])
    groups.setdefault(key, []).append(float(r['time_ms']))

base = {}
for (task, version, threads, chunk), xs in groups.items():
    if version == 'V1' and threads == '1':
        base[task] = statistics.mean(xs)

with open('results/task5.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['task', 'version', 'threads', 'chunk', 'trial', 'time_ms',
                'average_ms', 'minimum_ms', 'speedup', 'efficiency'])
    for key, xs in sorted(groups.items()):
        task, version, threads, chunk = key
        avg = statistics.mean(xs)
        b = base.get(task, float('nan'))
        sp = b / avg if b == b else float('nan')
        ef = sp / int(threads) if sp == sp else float('nan')
        for i, x in enumerate(xs, 1):
            w.writerow([task, version, threads, chunk, i, f'{x:.6f}',
                        f'{avg:.6f}', f'{min(xs):.6f}', f'{sp:.6f}', f'{ef:.6f}'])

missing = [t for t in ('1', '2', '3', '4') if t not in base]
if missing:
    print(f"WARNING: no V1 (threads=1) baseline found for task(s) {missing} "
          f"-- speedup/efficiency for those tasks will be NaN. "
          f"Make sure run_experiments.sh actually ran V1 at threads=1 for every task.")
else:
    print("Task 5 summary written to results/task5.csv. "
          f"Baselines used (T1, ms): " +
          ", ".join(f"task{t}={base[t]:.4f}" for t in sorted(base)))
