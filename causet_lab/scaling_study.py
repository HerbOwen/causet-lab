"""Does dimension level off or keep rising as N grows?

The three best (still-failing) configs from results/followup_report.md --
local_parents_adaptive(c=4,beta=0.33), local_parents_adaptive(c=2,beta=0.5),
and local_parents(k=12) -- are grown as large as is feasible (N up to
16000, enabled by the column-mirror cache optimization in rules.py, an
~20x speedup over the naive implementation) to answer one question:
does the sampled-interval dimension estimate converge to a fixed value,
or does it keep climbing indefinitely as the universe grows? A rule whose
dimension never stabilizes cannot be modeling a fixed-dimensional
spacetime no matter how good it looks at any single N.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .generators import sprinkle, grow
from .rules import local_parents, local_parents_adaptive
from .battery import analyze_matrix, nanmean, nanstd
from . import plots

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SCALING_NS = [2000, 4000, 8000, 16000]
SEEDS = [0, 1, 2]
SPRINKLE_SEEDS = [0, 1, 2, 3, 4]

N_SAMPLES = 200
MIN_INTERVAL = 10
MAX_INTERVAL = 300
KMAX = 10

TARGET_CONFIGS = [
    ("local_parents_adaptive(c=4,beta=0.33)", lambda: local_parents_adaptive(4, 0.33)),
    ("local_parents_adaptive(c=2,beta=0.5)", lambda: local_parents_adaptive(2, 0.5)),
    ("local_parents(k=12)", lambda: local_parents(12)),
]


def _run_config(label: str, factory, study_ns=None, seeds=None) -> dict:
    study_ns = study_ns if study_ns is not None else SCALING_NS
    seeds = seeds if seeds is not None else SEEDS
    t0 = time.time()
    by_N = {}
    for N in study_ns:
        mm_list, mp_list = [], []
        for seed in seeds:
            rule = factory()
            cset = grow(N, rule, seed=seed)
            res = analyze_matrix(
                cset.C, seed=seed, n_samples=N_SAMPLES, min_size=MIN_INTERVAL,
                max_size=MAX_INTERVAL, kmax=KMAX,
            )
            mm_list.append(res["mm_dim_sampled_mean"])
            mp_list.append(res["mp_dim_sampled_mean"])
        by_N[N] = {
            "mm_mean": nanmean(mm_list), "mm_std": nanstd(mm_list),
            "mp_mean": nanmean(mp_list), "mp_std": nanstd(mp_list),
        }
        print(f"[scaling-study] {label:40s} N={N:6d} done ({time.time() - t0:.1f}s elapsed) "
              f"mm={by_N[N]['mm_mean']:.3f} mp={by_N[N]['mp_mean']:.3f}")
    return by_N


def _run_sprinkle_reference(d: int, study_ns=None, seeds=None) -> dict:
    study_ns = study_ns if study_ns is not None else SCALING_NS
    seeds = seeds if seeds is not None else SPRINKLE_SEEDS
    t0 = time.time()
    by_N = {}
    for N in study_ns:
        mm_list = []
        for seed in seeds:
            cset, _coords = sprinkle(N, d, seed=seed * 100_000 + d * 10_000 + N)
            res = analyze_matrix(
                cset.C, seed=seed, n_samples=N_SAMPLES, min_size=MIN_INTERVAL,
                max_size=min(MAX_INTERVAL, max(20, N // 4)), kmax=KMAX,
            )
            mm_list.append(res["mm_dim_sampled_mean"])
        by_N[N] = {"mm_mean": nanmean(mm_list), "mm_std": nanstd(mm_list)}
        print(f"[scaling-study] sprinkle d={d:<28d} N={N:6d} done ({time.time() - t0:.1f}s elapsed) "
              f"mm={by_N[N]['mm_mean']:.3f}")
    return by_N


def _verdict_line(label: str, by_N: dict, study_ns: list) -> str:
    mm = [by_N[N]["mm_mean"] for N in study_ns]
    mp = [by_N[N]["mp_mean"] for N in study_ns]
    mm_deltas = [mm[i + 1] - mm[i] for i in range(len(mm) - 1)]
    mp_deltas = [mp[i + 1] - mp[i] for i in range(len(mp) - 1)]
    first_mm, last_mm = mm_deltas[0], mm_deltas[-1]
    decelerating = (
        np.isfinite(first_mm) and np.isfinite(last_mm)
        and abs(first_mm) > 1e-9 and abs(last_mm) < 0.4 * abs(first_mm) and abs(last_mm) < 0.15
    )
    verdict = "LEVELING OFF" if decelerating else "STILL RISING"
    deltas_str = " -> ".join(f"{d:+.3f}" for d in mm_deltas)
    return (
        f"{label}: mm deltas per doubling: {deltas_str} "
        f"(first={first_mm:+.3f}, last={last_mm:+.3f}) => {verdict}"
    )


def run_scaling_study(study_ns=None, seeds=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    study_ns = study_ns if study_ns is not None else SCALING_NS
    seeds = seeds if seeds is not None else SEEDS
    t0 = time.time()

    print(f"[scaling-study] sprinkle references d=3,4 over N={study_ns}...")
    sprinkle_d3 = _run_sprinkle_reference(3, study_ns=study_ns)
    sprinkle_d4 = _run_sprinkle_reference(4, study_ns=study_ns)

    print(f"[scaling-study] running {len(TARGET_CONFIGS)} target configs over N={study_ns}, seeds={seeds}...")
    results = {}
    for label, factory in TARGET_CONFIGS:
        results[label] = _run_config(label, factory, study_ns=study_ns, seeds=seeds)

    # Plot: dimension vs log N, sprinkle d=3/d=4 bands behind.
    cases = {}
    for label, by_N in results.items():
        cases[label] = {
            "N": study_ns,
            "mm_mean": [results[label][N]["mm_mean"] for N in study_ns],
            "mm_std": [results[label][N]["mm_std"] for N in study_ns],
            "mp_mean": [results[label][N]["mp_mean"] for N in study_ns],
            "mp_std": [results[label][N]["mp_std"] for N in study_ns],
        }
    bands = {
        3: {"N": study_ns, "mm_mean": [sprinkle_d3[N]["mm_mean"] for N in study_ns],
            "mm_std": [sprinkle_d3[N]["mm_std"] for N in study_ns]},
        4: {"N": study_ns, "mm_mean": [sprinkle_d4[N]["mm_mean"] for N in study_ns],
            "mm_std": [sprinkle_d4[N]["mm_std"] for N in study_ns]},
    }
    plot_path = RESULTS_DIR / "scaling_dimension_vs_logN.png"
    plots.plot_dimension_vs_N_bands(
        cases, bands, str(plot_path),
        title="Does dimension plateau or keep rising? (sprinkle d=3,4 bands behind)",
        legend_cols=1,
    )

    # Report.
    lines = []
    lines.append("# causet_lab scaling study: does dimension level off or keep rising?")
    lines.append("")
    lines.append(
        "The three best (still-failing) configs from `results/followup_report.md` grown as "
        f"large as feasible: N = {', '.join(str(n) for n in study_ns)}, {len(seeds)} seeds each. "
        "Enabled by a column-mirror cache optimization in rules.py (~20x speedup over the naive "
        "implementation -- the dominant prior cost was cache-hostile fancy-indexed column slicing "
        "on a row-major boolean matrix, not the matmul FLOPs themselves)."
    )
    lines.append("")
    lines.append("## Numbers")
    lines.append("")
    header = "| N | " + " | ".join(f"{label} (MM / mp)" for label, _ in TARGET_CONFIGS) + " | sprinkle d=3 (MM) | sprinkle d=4 (MM) |"
    lines.append(header)
    lines.append("|" + "---|" * (len(TARGET_CONFIGS) + 3))
    for N in study_ns:
        row = [str(N)]
        for label, _ in TARGET_CONFIGS:
            by_N = results[label][N]
            row.append(f"{by_N['mm_mean']:.3f} / {by_N['mp_mean']:.3f}")
        row.append(f"{sprinkle_d3[N]['mm_mean']:.3f}")
        row.append(f"{sprinkle_d4[N]['mm_mean']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Verdict: leveling off, or still rising?")
    lines.append("")
    verdict_lines = []
    for label, _ in TARGET_CONFIGS:
        vline = _verdict_line(label, results[label], study_ns)
        verdict_lines.append(vline)
        lines.append(f"- {vline}")
    lines.append("")
    step_str = ", ".join(f"{a}->{b}" for a, b in zip(study_ns, study_ns[1:]))
    lines.append(
        f"\"Deltas per doubling\" are the change in the Myrheim-Meyer estimate between consecutive "
        f"N values ({step_str}). LEVELING OFF means the last delta shrank "
        "to under 40% of the first delta and under 0.15 in absolute size; STILL RISING means it "
        "didn't."
    )
    lines.append("")
    lines.append(f"Plot: `{plot_path.name}`")
    lines.append("")

    report_path = RESULTS_DIR / "scaling_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[scaling-study] done in {time.time() - t0:.1f}s. wrote {report_path}")
    for vline in verdict_lines:
        print(f"[scaling-study] {vline}")
    return {"results": results, "sprinkle_d3": sprinkle_d3, "sprinkle_d4": sprinkle_d4}
