# causet_lab scaling study: does dimension level off or keep rising?

The three best (still-failing) configs from `results/followup_report.md` grown as large as feasible: N = 2000, 4000, 8000, 16000, 3 seeds each. Enabled by a column-mirror cache optimization in rules.py (~20x speedup over the naive implementation -- the dominant prior cost was cache-hostile fancy-indexed column slicing on a row-major boolean matrix, not the matmul FLOPs themselves).

## Numbers

| N | local_parents_adaptive(c=4,beta=0.33) (MM / mp) | local_parents_adaptive(c=2,beta=0.5) (MM / mp) | local_parents(k=12) (MM / mp) | sprinkle d=3 (MM) | sprinkle d=4 (MM) |
|---|---|---|---|---|---|
| 2000 | 3.135 / 3.358 | 3.047 / 3.335 | 2.967 / 3.178 | 3.048 | 4.197 |
| 4000 | 3.397 / 3.727 | 3.311 / 3.655 | 3.132 / 3.458 | 3.031 | 4.132 |
| 8000 | 3.602 / 3.986 | 3.600 / 3.986 | 3.363 / 3.757 | 3.019 | 4.139 |
| 16000 | 3.805 / 4.194 | 3.879 / 4.264 | 3.503 / 3.917 | 3.041 | 4.120 |

## Verdict: leveling off, or still rising?

- local_parents_adaptive(c=4,beta=0.33): mm deltas per doubling: +0.262 -> +0.204 -> +0.203 (first=+0.262, last=+0.203) => STILL RISING
- local_parents_adaptive(c=2,beta=0.5): mm deltas per doubling: +0.264 -> +0.289 -> +0.279 (first=+0.264, last=+0.279) => STILL RISING
- local_parents(k=12): mm deltas per doubling: +0.166 -> +0.230 -> +0.140 (first=+0.166, last=+0.140) => STILL RISING

"Deltas per doubling" are the change in the Myrheim-Meyer estimate between consecutive N values (2000->4000, 4000->8000, 8000->16000). LEVELING OFF means the last delta shrank to under 40% of the first delta and under 0.15 in absolute size; STILL RISING means it didn't.

Plot: `scaling_dimension_vs_logN.png`
