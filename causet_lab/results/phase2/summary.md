# causet_lab summary

mm = Myrheim-Meyer dimension estimate; mp = midpoint dimension estimate.
"whole" applies the estimator directly to the entire causal set, which is
only meaningful when the whole set is itself an interval -- true for the
sprinkles, not for grown or junk orders. "sampled" applies it to randomly
sampled sub-intervals, which is the fair comparison across every case.

| case | N | ordering fraction | mm (whole) | mp (whole) | mm (sampled) | mp (sampled) | height | seeds |
|---|---|---|---|---|---|---|---|---|
| kleitman_rothschild | 500 | 0.376 | 2.37 | 2.80 | 10.00 | nan | 3.0 | 5 |
| kleitman_rothschild | 1000 | 0.375 | 2.38 | 2.84 | 10.00 | nan | 3.0 | 5 |
| kleitman_rothschild | 2000 | 0.375 | 2.38 | 2.86 | 10.00 | nan | 3.0 | 5 |
| percolation p=0.01 | 2000 | 0.559 | 1.85 | 1.72 | 2.92 | 3.10 | 46.0 | 3 |
| percolation p=0.05 | 2000 | 0.933 | 1.11 | 1.07 | 2.10 | 2.14 | 185.7 | 3 |
| percolation p=0.1 | 2000 | 0.973 | 1.04 | 1.03 | 1.63 | 1.62 | 326.7 | 3 |
| percolation p=0.3 | 2000 | 0.995 | 1.01 | 1.00 | 1.15 | 1.15 | 791.3 | 3 |
| sprinkle d=2 | 500 | 0.505 | 1.99 | 1.99 | 2.02 | 2.16 | 40.2 | 5 |
| sprinkle d=2 | 1000 | 0.502 | 2.00 | 1.99 | 2.01 | 2.11 | 59.2 | 5 |
| sprinkle d=2 | 2000 | 0.502 | 1.99 | 2.01 | 2.01 | 2.11 | 83.8 | 5 |
| sprinkle d=3 | 500 | 0.227 | 3.01 | 3.09 | 3.07 | 3.35 | 14.2 | 5 |
| sprinkle d=3 | 1000 | 0.221 | 3.04 | 3.15 | 3.05 | 3.32 | 20.0 | 5 |
| sprinkle d=3 | 2000 | 0.231 | 2.99 | 3.03 | 3.05 | 3.31 | 23.6 | 5 |
| sprinkle d=4 | 500 | 0.103 | 3.97 | 4.22 | 4.26 | 4.24 | 8.2 | 5 |
| sprinkle d=4 | 1000 | 0.099 | 4.01 | 4.09 | 4.17 | 4.29 | 9.8 | 5 |
| sprinkle d=4 | 2000 | 0.102 | 3.98 | 4.08 | 4.20 | 4.35 | 12.6 | 5 |

Plots are saved alongside this file in `causet_lab/results/`.