"""CLI entry point for causet_lab.

    python -m causet_lab.run controls
    python -m causet_lab.run junk
    python -m causet_lab.run percolation --p 0.01 0.05 0.1 0.3 --N 2000
    python -m causet_lab.run rule --name transitive_percolation --N 2000 --param p=0.1
    python -m causet_lab.run all
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .generators import sprinkle, kleitman_rothschild, grow
from .rules import RULE_FACTORIES, transitive_percolation
from .battery import analyze_matrix, nanmean, nanstd
from . import plots

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_CONTROL_NS = [500, 1000, 2000]
DEFAULT_CONTROL_DS = [2, 3, 4]
DEFAULT_CONTROL_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_JUNK_NS = [500, 1000, 2000]
DEFAULT_PERCOLATION_P = [0.01, 0.05, 0.1, 0.3]
DEFAULT_PERCOLATION_SEEDS = [0, 1, 2]


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def summarize_rows(rows: list) -> list:
    """Group per-(case, N) raw rows (one per seed) into mean summary rows."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["case"], r["N"])].append(r)
    summary = []
    for (case, N), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        summary.append({
            "case": case,
            "N": N,
            "ordering_fraction": nanmean([r["ordering_fraction"] for r in rs]),
            "mm_dim_whole": nanmean([r["mm_dim_whole"] for r in rs]),
            "mp_dim_whole": nanmean([r["mp_dim_whole"] for r in rs]),
            "mm_dim_sampled": nanmean([r["mm_dim_sampled_mean"] for r in rs]),
            "mp_dim_sampled": nanmean([r["mp_dim_sampled_mean"] for r in rs]),
            "height": nanmean([r["height"] for r in rs]),
            "n_seeds": len(rs),
        })
    return summary


def print_table(summary: list) -> None:
    header = f"{'case':<30}{'N':>7}{'f':>8}{'mm(whole)':>11}{'mp(whole)':>11}{'mm(samp)':>10}{'mp(samp)':>10}{'height':>9}{'seeds':>7}"
    print(header)
    print("-" * len(header))
    for s in summary:
        print(
            f"{s['case']:<30}{s['N']:>7}{s['ordering_fraction']:>8.3f}"
            f"{s['mm_dim_whole']:>11.2f}{s['mp_dim_whole']:>11.2f}"
            f"{s['mm_dim_sampled']:>10.2f}{s['mp_dim_sampled']:>10.2f}"
            f"{s['height']:>9.1f}{s['n_seeds']:>7}"
        )


def write_summary_md(summary: list, path: Path) -> None:
    lines = [
        "# causet_lab summary",
        "",
        "mm = Myrheim-Meyer dimension estimate; mp = midpoint dimension estimate.",
        "\"whole\" applies the estimator directly to the entire causal set, which is",
        "only meaningful when the whole set is itself an interval -- true for the",
        "sprinkles, not for grown or junk orders. \"sampled\" applies it to randomly",
        "sampled sub-intervals, which is the fair comparison across every case.",
        "",
        "| case | N | ordering fraction | mm (whole) | mp (whole) | mm (sampled) | mp (sampled) | height | seeds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['case']} | {s['N']} | {s['ordering_fraction']:.3f} | {s['mm_dim_whole']:.2f} | "
            f"{s['mp_dim_whole']:.2f} | {s['mm_dim_sampled']:.2f} | {s['mp_dim_sampled']:.2f} | "
            f"{s['height']:.1f} | {s['n_seeds']} |"
        )
    lines.append("")
    lines.append("Plots are saved alongside this file in `causet_lab/results/`.")
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_controls(args=None) -> list:
    """Sprinkle d = 2, 3, 4 at N = 500, 1000, 2000 (5 seeds each): the
    known-answer controls."""
    _ensure_results_dir()
    t0 = time.time()
    rows = []
    dim_cases = {}
    height_cases = {}
    abundance_profiles = {}

    for d in DEFAULT_CONTROL_DS:
        mm_by_N = {N: [] for N in DEFAULT_CONTROL_NS}
        mp_by_N = {N: [] for N in DEFAULT_CONTROL_NS}
        h_by_N = {N: [] for N in DEFAULT_CONTROL_NS}
        rep_profile = None
        for N in DEFAULT_CONTROL_NS:
            for seed in DEFAULT_CONTROL_SEEDS:
                cset, coords = sprinkle(N, d, seed=seed * 100_000 + d * 10_000 + N)
                res = analyze_matrix(cset.C, seed=seed, min_size=10, max_size=min(300, max(20, N // 4)))
                mm_by_N[N].append(res["mm_dim_whole"])
                mp_by_N[N].append(res["mp_dim_whole"])
                h_by_N[N].append(res["height"])
                rows.append({"case": f"sprinkle d={d}", "N": N, "seed": seed, **res})
                if N == DEFAULT_CONTROL_NS[-1] and seed == DEFAULT_CONTROL_SEEDS[0]:
                    rep_profile = res["abundance_profile"]
            print(f"[controls] d={d} N={N} done ({time.time() - t0:.1f}s elapsed)")
        dim_cases[f"d={d}"] = {
            "x": DEFAULT_CONTROL_NS,
            "mm_mean": [nanmean(mm_by_N[N]) for N in DEFAULT_CONTROL_NS],
            "mm_std": [nanstd(mm_by_N[N]) for N in DEFAULT_CONTROL_NS],
            "mp_mean": [nanmean(mp_by_N[N]) for N in DEFAULT_CONTROL_NS],
            "mp_std": [nanstd(mp_by_N[N]) for N in DEFAULT_CONTROL_NS],
        }
        height_cases[f"d={d}"] = {
            "N": DEFAULT_CONTROL_NS,
            "height_mean": [nanmean(h_by_N[N]) for N in DEFAULT_CONTROL_NS],
            "height_std": [nanstd(h_by_N[N]) for N in DEFAULT_CONTROL_NS],
        }
        abundance_profiles[f"sprinkle d={d}"] = rep_profile

        small, small_coords = sprinkle(100, d, seed=42)
        plots.plot_hasse(
            small.C, str(RESULTS_DIR / f"hasse_sprinkle_d{d}.png"), coords=small_coords,
            title=f"Hasse diagram (links only): sprinkle d={d}, N=100",
        )

    summary = summarize_rows(rows)
    print_table(summary)
    plots.plot_dimension_vs_x(
        dim_cases, "N", str(RESULTS_DIR / "controls_dimension_vs_N.png"),
        title="Controls: dimension estimates vs N", ref_lines=DEFAULT_CONTROL_DS,
    )
    plots.plot_height_vs_N(
        height_cases, str(RESULTS_DIR / "controls_height_vs_N.png"),
        ref_slopes=[1 / d for d in DEFAULT_CONTROL_DS], title="Controls: height vs N",
    )
    plots.plot_abundance_profiles(
        abundance_profiles, str(RESULTS_DIR / "controls_abundance_profiles.png"),
        title="Controls: interval abundance profiles",
    )
    print(f"[controls] total time {time.time() - t0:.1f}s")
    return rows


def cmd_junk(args) -> list:
    """Kleitman-Rothschild junk orders: what 'no geometry' looks like."""
    _ensure_results_dir()
    t0 = time.time()
    Ns = args.N
    rows = []
    profile = None
    for N in Ns:
        for seed in range(5):
            cset = kleitman_rothschild(N, seed=seed)
            res = analyze_matrix(cset.C, seed=seed, min_size=0, max_size=max(50, N // 4))
            rows.append({"case": "kleitman_rothschild", "N": N, "seed": seed, **res})
            if profile is None:
                profile = res["abundance_profile"]
        print(f"[junk] N={N} done ({time.time() - t0:.1f}s elapsed)")

    summary = summarize_rows(rows)
    print_table(summary)
    plots.plot_abundance_profiles(
        {"kleitman_rothschild": profile}, str(RESULTS_DIR / "junk_abundance_profile.png"),
        title="Junk: interval abundance profile",
    )
    small = kleitman_rothschild(100, seed=42)
    plots.plot_hasse(
        small.C, str(RESULTS_DIR / "hasse_junk.png"), coords=None,
        title="Hasse diagram (links only): Kleitman-Rothschild, N=100",
    )
    print(f"[junk] total time {time.time() - t0:.1f}s")
    return rows


def cmd_percolation(args) -> list:
    """Grow universes under transitive percolation and measure sampled
    intervals."""
    _ensure_results_dir()
    t0 = time.time()
    rows = []
    mm_mean, mm_std, mp_mean, mp_std = [], [], [], []
    profiles = {}

    for p in args.p:
        rule = transitive_percolation(p)
        mm_vals, mp_vals = [], []
        profile = None
        for seed in args.seeds:
            cset = grow(args.N, rule, seed=seed)
            res = analyze_matrix(cset.C, seed=seed, min_size=10, max_size=300)
            rows.append({"case": f"percolation p={p}", "N": args.N, "seed": seed, **res})
            mm_vals.append(res["mm_dim_sampled_mean"])
            mp_vals.append(res["mp_dim_sampled_mean"])
            if profile is None:
                profile = res["abundance_profile"]
            print(f"[percolation] p={p} seed={seed} done ({time.time() - t0:.1f}s elapsed)")
        mm_mean.append(nanmean(mm_vals))
        mm_std.append(nanstd(mm_vals))
        mp_mean.append(nanmean(mp_vals))
        mp_std.append(nanstd(mp_vals))
        profiles[f"p={p}"] = profile

        small = grow(100, rule, seed=999)
        tag = f"{p:g}".replace(".", "p")
        plots.plot_hasse(
            small.C, str(RESULTS_DIR / f"hasse_percolation_p{tag}.png"), coords=None,
            title=f"Hasse diagram (links only): percolation p={p}, N=100",
        )

    summary = summarize_rows(rows)
    print_table(summary)
    dim_cases = {
        "percolation": {"x": args.p, "mm_mean": mm_mean, "mm_std": mm_std, "mp_mean": mp_mean, "mp_std": mp_std},
    }
    plots.plot_dimension_vs_x(
        dim_cases, "p", str(RESULTS_DIR / "percolation_dimension_vs_p.png"),
        title=f"Transitive percolation: dimension vs p (N={args.N})",
    )
    plots.plot_abundance_profiles(
        profiles, str(RESULTS_DIR / "percolation_abundance_profiles.png"),
        title="Percolation: interval abundance profiles",
    )
    print(f"[percolation] total time {time.time() - t0:.1f}s")
    return rows


def cmd_rule(args) -> list:
    """Run any rule from rules.py through the same measurement battery."""
    _ensure_results_dir()
    t0 = time.time()
    factory = RULE_FACTORIES[args.name]
    kwargs = {}
    for kv in args.param:
        key, _, val = kv.partition("=")
        try:
            val = float(val)
        except ValueError:
            pass
        kwargs[key] = val
    rule = factory(**kwargs)

    rows = []
    profile = None
    for seed in args.seeds:
        cset = grow(args.N, rule, seed=seed)
        res = analyze_matrix(cset.C, seed=seed, min_size=10, max_size=300)
        rows.append({"case": f"rule:{args.name}", "N": args.N, "seed": seed, **res})
        if profile is None:
            profile = res["abundance_profile"]
        print(f"[rule] {args.name} seed={seed} done ({time.time() - t0:.1f}s elapsed)")

    summary = summarize_rows(rows)
    print_table(summary)
    plots.plot_abundance_profiles(
        {f"rule:{args.name}": profile}, str(RESULTS_DIR / f"rule_{args.name}_abundance.png"),
        title=f"Rule '{args.name}': interval abundance profile",
    )
    small = grow(100, rule, seed=999)
    plots.plot_hasse(
        small.C, str(RESULTS_DIR / f"hasse_rule_{args.name}.png"), coords=None,
        title=f"Hasse diagram (links only): rule={args.name}, N=100",
    )
    print(f"[rule] total time {time.time() - t0:.1f}s")
    return rows


def cmd_all() -> None:
    """Run controls, junk, and percolation, then write results/summary.md
    comparing every case side by side."""
    _ensure_results_dir()
    t0 = time.time()

    print("=== controls ===")
    controls_rows = cmd_controls()

    print("=== junk ===")
    junk_rows = cmd_junk(argparse.Namespace(N=DEFAULT_JUNK_NS))

    print("=== percolation ===")
    percolation_rows = cmd_percolation(
        argparse.Namespace(p=DEFAULT_PERCOLATION_P, N=2000, seeds=DEFAULT_PERCOLATION_SEEDS)
    )

    all_rows = controls_rows + junk_rows + percolation_rows
    summary = summarize_rows(all_rows)
    print("\n=== summary (all cases) ===")
    print_table(summary)
    write_summary_md(summary, RESULTS_DIR / "summary.md")
    print(f"\n[all] wrote {RESULTS_DIR / 'summary.md'}")
    print(f"[all] total time {time.time() - t0:.1f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m causet_lab.run",
        description="Explore whether spacetime can emerge from pure causal order.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("controls", help="Sprinkle d=2,3,4 controls (known dimension) and measure everything.")

    p_junk = sub.add_parser("junk", help="Kleitman-Rothschild junk orders (no geometry).")
    p_junk.add_argument("--N", type=int, nargs="+", default=DEFAULT_JUNK_NS)

    p_perc = sub.add_parser("percolation", help="Grow universes via transitive percolation.")
    p_perc.add_argument("--p", type=float, nargs="+", default=DEFAULT_PERCOLATION_P)
    p_perc.add_argument("--N", type=int, default=2000)
    p_perc.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_PERCOLATION_SEEDS)

    p_rule = sub.add_parser("rule", help="Run any rule from rules.py through the same battery.")
    p_rule.add_argument("--name", required=True, choices=sorted(RULE_FACTORIES.keys()))
    p_rule.add_argument("--N", type=int, default=2000)
    p_rule.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p_rule.add_argument("--param", nargs="*", default=[], help="key=value pairs for the rule factory, e.g. p=0.1")

    sub.add_parser("all", help="Run controls, junk, percolation; write results/summary.md.")

    p_study = sub.add_parser(
        "rules-study",
        help="Batch-test the R1-R6 growth-rule hypotheses, score against sprinkle-calibrated "
             "verdict thresholds, write results/rules_report.md.",
    )
    p_study.add_argument("--N", type=int, nargs="+", default=None,
                          help="override the N grid (default 500 1000 2000 4000); smaller values speed up smoke tests")
    p_study.add_argument("--seeds", type=int, nargs="+", default=None,
                          help="override the seed list (default 0 1 2 3 4)")

    p_followup = sub.add_parser(
        "rules-followup",
        help="Follow-up batch: local_parents at larger k, local_parents_adaptive, "
             "frontier_weighted, tightened inner-structure check, write results/followup_report.md.",
    )
    p_followup.add_argument("--N", type=int, nargs="+", default=None,
                             help="override the N grid (default 500 1000 2000 4000); smaller values speed up smoke tests")
    p_followup.add_argument("--seeds", type=int, nargs="+", default=None,
                             help="override the seed list (default 0 1 2 3 4)")

    p_scaling = sub.add_parser(
        "scaling-study",
        help="Grow the 3 best followup configs as large as feasible (default N up to 16000) to "
             "see whether their dimension estimate levels off or keeps rising; write "
             "results/scaling_report.md.",
    )
    p_scaling.add_argument("--N", type=int, nargs="+", default=None,
                            help="override the N grid (default 2000 4000 8000 16000)")
    p_scaling.add_argument("--seeds", type=int, nargs="+", default=None,
                            help="override the seed list (default 0 1 2)")

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "controls":
        cmd_controls()
    elif args.command == "junk":
        cmd_junk(args)
    elif args.command == "percolation":
        cmd_percolation(args)
    elif args.command == "rule":
        cmd_rule(args)
    elif args.command == "all":
        cmd_all()
    elif args.command == "rules-study":
        from . import rules_study
        rules_study.run_study(study_ns=args.N, seeds=args.seeds)
    elif args.command == "rules-followup":
        from . import followup_study
        followup_study.run_followup_study(study_ns=args.N, seeds=args.seeds)
    elif args.command == "scaling-study":
        from . import scaling_study
        scaling_study.run_scaling_study(study_ns=args.N, seeds=args.seeds)


if __name__ == "__main__":
    main()
