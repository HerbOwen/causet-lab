"""Batch study: which growth rules produce universes that pass the full
measurement battery?

This runs every rule in the R1-R6 batch (see rules.py) across a grid of N,
with repeated seeds, through the same measurement battery used elsewhere
in causet_lab, then scores each rule against five checks calibrated from
the sprinkle controls (never hand-tuned) and writes results/rules_report.md.

See rules.py's module docstring for the frontier-collapse story: the first
version of R1-R4 and R6 picked parents only from the current frontier
(maximal elements), which turned out to deterministically collapse every
one of those rules into a plain chain, for any parameter choice. The
rules in this file are the fixed versions, which pick parents from all
existing elements instead.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .generators import sprinkle, grow
from .rules import (
    random_parents,
    local_parents,
    preferential_parents,
    bounded_valence,
    recency_cheat,
    width_forced,
)
from .battery import analyze_matrix, nanmean, nanstd, nanmean_axis0
from .measures import height_exponent, dimension_drift, abundance_distance
from . import plots

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "phase2"

STUDY_NS = [500, 1000, 2000, 4000]
SPRINKLE_CALIBRATION_NS = [500, 1000, 2000, 3000, 4000]
SEEDS = [0, 1, 2, 3, 4]

SLOW_THRESHOLD_SECONDS = 15.0
FALLBACK_N = 3000

KMAX = 10
MIN_INTERVAL = 10
MAX_INTERVAL = 300
N_SAMPLES = 200

HEIGHT_CONSISTENCY_BAND = (0.8, 1.25)
VALENCE_GROWTH_RATIO_THRESHOLD = 1.15  # valence(N_max) / valence(N_min) >= this to count as "grows"

HASSE_N = 100
HASSE_SEED = 123


# ---------------------------------------------------------------------------
# Sprinkle baselines: every verdict threshold (except the two given directly
# in the spec, height-consistency's [0.8, 1.25] band and the valence-growth
# ratio) is calibrated from these, never hand-tuned per rule.
# ---------------------------------------------------------------------------

def compute_sprinkle_baselines(calibration_ns=None):
    calibration_ns = calibration_ns if calibration_ns is not None else SPRINKLE_CALIBRATION_NS
    ref_N = max(calibration_ns)
    print(f"[rules-study] computing sprinkle baselines (d=2,3,4) over N={calibration_ns}...")
    t0 = time.time()
    baselines = {}
    all_agreement = []
    all_drift = []
    all_pairwise_abundance = []

    for d in (2, 3, 4):
        mm_by_N, mp_by_N, mm_std_by_N = {}, {}, {}
        height_by_N, height_std_by_N, valence_by_N, fw_by_N = {}, {}, {}, {}
        profile_by_N = {}
        for N in calibration_ns:
            mm_list, mp_list, h_list, val_list, fw_list = [], [], [], [], []
            profiles = []
            for seed in SEEDS:
                cset, _coords = sprinkle(N, d, seed=seed * 100_000 + d * 10_000 + N)
                res = analyze_matrix(
                    cset.C, seed=seed, n_samples=N_SAMPLES, min_size=MIN_INTERVAL,
                    max_size=min(MAX_INTERVAL, max(20, N // 4)), kmax=KMAX,
                )
                mm_list.append(res["mm_dim_sampled_mean"])
                mp_list.append(res["mp_dim_sampled_mean"])
                h_list.append(res["height"])
                val_list.append(res["valence"])
                fw_list.append(res["frontier_width"])
                profiles.append(res["abundance_profile"])
                agree = abs(res["mm_dim_sampled_mean"] - res["mp_dim_sampled_mean"])
                if np.isfinite(agree):
                    all_agreement.append(agree)
            mm_by_N[N] = nanmean(mm_list)
            mp_by_N[N] = nanmean(mp_list)
            mm_std_by_N[N] = nanstd(mm_list)
            height_by_N[N] = nanmean(h_list)
            height_std_by_N[N] = nanstd(h_list)
            valence_by_N[N] = nanmean(val_list)
            fw_by_N[N] = nanmean(fw_list)
            profile_by_N[N] = nanmean_axis0(profiles)
            if N == ref_N:
                for i in range(len(profiles)):
                    for j in range(i + 1, len(profiles)):
                        dist = abundance_distance(profiles[i], profiles[j])
                        if np.isfinite(dist):
                            all_pairwise_abundance.append(dist)

        Ns_arr = np.array(calibration_ns, dtype=float)
        mm_arr = np.array([mm_by_N[N] for N in calibration_ns])
        h_arr = np.array([height_by_N[N] for N in calibration_ns])
        drift = dimension_drift(Ns_arr, mm_arr)
        alpha = height_exponent(Ns_arr, h_arr)
        if np.isfinite(drift):
            all_drift.append(abs(drift))

        baselines[d] = {
            "mm_by_N": mm_by_N, "mp_by_N": mp_by_N, "mm_std_by_N": mm_std_by_N,
            "height_by_N": height_by_N, "height_std_by_N": height_std_by_N,
            "valence_by_N": valence_by_N, "fw_by_N": fw_by_N,
            "profile_by_N": profile_by_N, "drift": drift, "alpha": alpha,
        }
        print(f"[rules-study]   d={d} baseline done ({time.time() - t0:.1f}s elapsed)")

    thresholds = {
        "agreement_baseline": nanmean(all_agreement),
        "drift_baseline": nanmean(all_drift) if all_drift else 0.01,
        "abundance_baseline": nanmean(all_pairwise_abundance),
    }
    print(f"[rules-study] sprinkle baselines complete ({time.time() - t0:.1f}s): {thresholds}")
    return baselines, thresholds


def calibrate_width_forced_c(baselines: dict, d_target: int, N_ref: int = 2000) -> float:
    """c such that c * N_ref**((d-1)/d) matches the real sprinkle's final
    frontier width at N_ref -- the "shape" width_forced is told to target."""
    fw = baselines[d_target]["fw_by_N"][N_ref]
    return float(fw / (N_ref ** ((d_target - 1.0) / d_target)))


# ---------------------------------------------------------------------------
# Rule configuration grid
# ---------------------------------------------------------------------------

def build_rule_configs(width_forced_c: dict) -> list:
    configs = []
    for k in (1, 2, 3, 5):
        configs.append({
            "family": "random_parents", "label": f"random_parents(k={k})",
            "params": {"k": k}, "factory": (lambda k=k: random_parents(k)), "flag": None,
        })
    for k in (2, 3, 5, 8):
        configs.append({
            "family": "local_parents", "label": f"local_parents(k={k})",
            "params": {"k": k}, "factory": (lambda k=k: local_parents(k)), "flag": None,
        })
    for k in (2, 3, 5):
        configs.append({
            "family": "preferential_parents", "label": f"preferential_parents(k={k})",
            "params": {"k": k}, "factory": (lambda k=k: preferential_parents(k)), "flag": None,
        })
    for k in (2, 3):
        for vmax in (2, 4, 8):
            configs.append({
                "family": "bounded_valence", "label": f"bounded_valence(k={k},vmax={vmax})",
                "params": {"k": k, "vmax": vmax},
                "factory": (lambda k=k, vmax=vmax: bounded_valence(k, vmax)), "flag": None,
            })
    for L in (20, 50):
        for p in (0.1, 0.3):
            configs.append({
                "family": "recency_cheat", "label": f"recency_cheat(L={L},p={p})",
                "params": {"L": L, "p": p},
                "factory": (lambda L=L, p=p: recency_cheat(L, p)), "flag": "illegal",
            })
    for d_target in (2, 4):
        c = width_forced_c[d_target]
        configs.append({
            "family": "width_forced", "label": f"width_forced(d_target={d_target})",
            "params": {"k": 3, "d_target": float(d_target), "c": c},
            "factory": (lambda d_target=d_target, c=c: width_forced(3, float(d_target), c)),
            "flag": "cheat",
        })
    return configs


# ---------------------------------------------------------------------------
# Running one rule configuration across the full N grid
# ---------------------------------------------------------------------------

def _summarize_by_N(rows: list) -> list:
    by_N = {}
    for r in rows:
        by_N.setdefault(r["N"], []).append(r)
    summary = []
    for N in sorted(by_N):
        rs = by_N[N]
        summary.append({
            "N": N,
            "mm_mean": nanmean([r["mm_dim_sampled_mean"] for r in rs]),
            "mm_std": nanstd([r["mm_dim_sampled_mean"] for r in rs]),
            "mp_mean": nanmean([r["mp_dim_sampled_mean"] for r in rs]),
            "mp_std": nanstd([r["mp_dim_sampled_mean"] for r in rs]),
            "height_mean": nanmean([r["height"] for r in rs]),
            "height_std": nanstd([r["height"] for r in rs]),
            "valence_mean": nanmean([r["valence"] for r in rs]),
            "ordering_fraction_mean": nanmean([r["ordering_fraction"] for r in rs]),
            "n_seeds": len(rs),
        })
    return summary


def run_rule_config(config: dict, study_ns=None, seeds=None) -> dict:
    study_ns = study_ns if study_ns is not None else STUDY_NS
    seeds = seeds if seeds is not None else SEEDS
    rows = []
    notes = []
    frontier_traj = None
    widen_fraction = None
    frontier_traj_N = None
    cap_hits_total = 0
    cap_calls_total = 0
    has_cap_stats = False
    t0 = time.time()

    for N in study_ns:
        N_eff = N
        if N == study_ns[-1] and N >= 4000:
            probe_rule = config["factory"]()
            probe_t0 = time.time()
            grow(N, probe_rule, seed=0)
            probe_dt = time.time() - probe_t0
            if probe_dt > SLOW_THRESHOLD_SECONDS:
                notes.append(
                    f"N={N} too slow ({probe_dt:.1f}s for a single seed); "
                    f"dropped to N={FALLBACK_N} for this config"
                )
                N_eff = FALLBACK_N

        for seed in seeds:
            rule = config["factory"]()
            track = (N == study_ns[-1]) and (seed == seeds[0])
            if track:
                cset, info = grow(N_eff, rule, seed=seed, track_frontier=True, track_branching=True)
                frontier_traj = (info["frontier_history"][:, 0], info["frontier_history"][:, 1])
                widen_fraction = info["widen_fraction"]
                frontier_traj_N = N_eff
            else:
                cset = grow(N_eff, rule, seed=seed)
            rule_state = getattr(rule, "state", None)
            if rule_state is not None and "cap_hits" in rule_state:
                has_cap_stats = True
                cap_hits_total += rule_state["cap_hits"]
                cap_calls_total += rule_state["total_calls"]
            res = analyze_matrix(
                cset.C, seed=seed, n_samples=N_SAMPLES, min_size=MIN_INTERVAL,
                max_size=MAX_INTERVAL, kmax=KMAX,
            )
            rows.append({"N": N_eff, "seed": seed, **res})
        print(f"[rules-study] {config['label']:32s} N={N_eff:5d} done ({time.time() - t0:.1f}s elapsed)")

    return {
        "label": config["label"], "family": config["family"], "params": config["params"],
        "flag": config.get("flag"), "rows": rows, "summary_by_N": _summarize_by_N(rows),
        "frontier_trajectory": frontier_traj, "frontier_trajectory_N": frontier_traj_N,
        "widen_fraction": widen_fraction, "notes": notes, "total_time": time.time() - t0,
        "cap_hit_stats": {"hits": cap_hits_total, "calls": cap_calls_total} if has_cap_stats else None,
    }


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def compute_verdict(result: dict, baselines: dict, thresholds: dict) -> dict:
    summary = result["summary_by_N"]
    Ns = np.array([s["N"] for s in summary], dtype=float)
    mm_means = np.array([s["mm_mean"] for s in summary])
    mp_means = np.array([s["mp_mean"] for s in summary])
    height_means = np.array([s["height_mean"] for s in summary])
    valence_means = np.array([s["valence_mean"] for s in summary])

    agreement = nanmean(np.abs(mm_means - mp_means))
    drift = dimension_drift(Ns, mm_means)
    alpha = height_exponent(Ns, height_means)
    dMM = nanmean(mm_means)
    height_consistency = alpha * dMM if np.isfinite(alpha) and np.isfinite(dMM) else float("nan")

    N_last = summary[-1]["N"]
    rows_at_last_N = [r for r in result["rows"] if r["N"] == N_last]
    profile = nanmean_axis0([r["abundance_profile"] for r in rows_at_last_N])
    distances = {}
    for d in (2, 3, 4):
        sprinkle_profile = baselines[d]["profile_by_N"].get(N_last)
        distances[d] = abundance_distance(profile, sprinkle_profile) if sprinkle_profile is not None else float("nan")
    finite_distances = {d: v for d, v in distances.items() if np.isfinite(v)}
    if finite_distances:
        best_d = min(finite_distances, key=finite_distances.get)
        best_distance = finite_distances[best_d]
    else:
        best_d, best_distance = None, float("nan")

    v0, v1 = valence_means[0], valence_means[-1]
    valence_ratio = (v1 / v0) if np.isfinite(v0) and np.isfinite(v1) and v0 > 0 else float("nan")

    agreement_threshold = 2 * thresholds["agreement_baseline"]
    drift_threshold = 2 * thresholds["drift_baseline"]
    abundance_threshold = 2 * thresholds["abundance_baseline"]

    checks = {
        "agreement": {
            "value": agreement, "threshold": agreement_threshold,
            "pass": bool(np.isfinite(agreement) and agreement <= agreement_threshold),
        },
        "stability": {
            "value": abs(drift) if np.isfinite(drift) else float("nan"), "threshold": drift_threshold,
            "pass": bool(np.isfinite(drift) and abs(drift) <= drift_threshold),
        },
        "height_consistency": {
            "value": height_consistency, "threshold": HEIGHT_CONSISTENCY_BAND,
            "pass": bool(np.isfinite(height_consistency)
                         and HEIGHT_CONSISTENCY_BAND[0] <= height_consistency <= HEIGHT_CONSISTENCY_BAND[1]),
        },
        "inner_structure": {
            "value": best_distance, "threshold": abundance_threshold, "closest_d": best_d,
            "pass": bool(np.isfinite(best_distance) and best_distance <= abundance_threshold),
        },
        "valence_growth": {
            "value": valence_ratio, "threshold": VALENCE_GROWTH_RATIO_THRESHOLD,
            "pass": bool(np.isfinite(valence_ratio) and valence_ratio >= VALENCE_GROWTH_RATIO_THRESHOLD),
        },
    }
    overall_pass = all(c["pass"] for c in checks.values())
    return {
        "checks": checks, "overall_pass": overall_pass, "agreement": agreement, "drift": drift,
        "alpha": alpha, "dMM": dMM, "height_consistency": height_consistency,
        "valence_ratio": valence_ratio, "closest_d": best_d, "abundance_distance": best_distance,
        "abundance_distances_by_d": distances,
    }


def failure_signature(verdict: dict) -> str:
    checks = verdict["checks"]
    failed = [name for name, c in checks.items() if not c["pass"]]
    if not failed:
        return "passes all checks"
    parts = []
    if "agreement" in failed:
        parts.append(f"estimators disagree (|MM-mp|={verdict['agreement']:.2f} > {checks['agreement']['threshold']:.2f})")
    if "stability" in failed:
        d = verdict["drift"]
        d_str = f"{d:.3f}" if np.isfinite(d) else "nan"
        parts.append(f"dimension drifts with N (slope={d_str}, threshold {checks['stability']['threshold']:.3f})")
    if "height_consistency" in failed:
        hc = verdict["height_consistency"]
        hc_str = f"{hc:.2f}" if np.isfinite(hc) else "nan"
        parts.append(f"height scaling inconsistent (alpha*dMM={hc_str}, want 0.8-1.25)")
    if "inner_structure" in failed:
        d = verdict["closest_d"]
        parts.append(
            f"abundance profile far from any sprinkle (closest d={d}, "
            f"distance={verdict['abundance_distance']:.2f} > {checks['inner_structure']['threshold']:.2f})"
        )
    if "valence_growth" in failed:
        vr = verdict["valence_ratio"]
        vr_str = f"{vr:.2f}" if np.isfinite(vr) else "nan"
        parts.append(f"valence flat (ratio={vr_str} < {VALENCE_GROWTH_RATIO_THRESHOLD})")
    return "; ".join(parts)


def pick_best_per_family(results: list) -> dict:
    by_family = {}
    for r in results:
        by_family.setdefault(r["family"], []).append(r)
    best = {}
    for family, items in by_family.items():
        def score(r):
            n_pass = sum(c["pass"] for c in r["verdict"]["checks"].values())
            agreement = r["verdict"]["agreement"]
            agreement = agreement if np.isfinite(agreement) else float("inf")
            return (-n_pass, agreement)
        best[family] = min(items, key=score)
    return best


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _sprinkle_bands(baselines: dict) -> dict:
    bands = {}
    for d in (2, 3, 4):
        Ns = SPRINKLE_CALIBRATION_NS
        bands[d] = {
            "N": Ns,
            "mm_mean": [baselines[d]["mm_by_N"][N] for N in Ns],
            "mm_std": [baselines[d]["mm_std_by_N"][N] for N in Ns],
        }
    return bands


def generate_plots(results: list, baselines: dict, best_per_family: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bands = _sprinkle_bands(baselines)

    def dim_cases(items):
        cases = {}
        for r in items:
            s = r["summary_by_N"]
            cases[r["label"]] = {
                "N": [x["N"] for x in s], "mm_mean": [x["mm_mean"] for x in s], "mm_std": [x["mm_std"] for x in s],
                "mp_mean": [x["mp_mean"] for x in s], "mp_std": [x["mp_std"] for x in s],
            }
        return cases

    plots.plot_dimension_vs_N_bands(
        dim_cases(results), bands, str(RESULTS_DIR / "rules_dimension_vs_N_overview.png"),
        title="All rules: dimension vs N (sprinkle bands behind)", legend_cols=3,
    )

    by_family = {}
    for r in results:
        by_family.setdefault(r["family"], []).append(r)

    for family, items in by_family.items():
        plots.plot_dimension_vs_N_bands(
            dim_cases(items), bands, str(RESULTS_DIR / f"rules_dimension_vs_N_{family}.png"),
            title=f"{family}: dimension vs N (sprinkle bands behind)",
        )
        trajectories = {r["label"]: r["frontier_trajectory"] for r in items if r["frontier_trajectory"] is not None}
        if trajectories:
            plots.plot_frontier_width_vs_n(
                trajectories, str(RESULTS_DIR / f"rules_frontier_width_{family}.png"),
                title=f"{family}: frontier width vs n during growth",
            )

    valence_cases = {}
    for r in results:
        s = r["summary_by_N"]
        valence_cases[r["label"]] = {"N": [x["N"] for x in s], "valence_mean": [x["valence_mean"] for x in s]}
    for d in (2, 3, 4):
        Ns = SPRINKLE_CALIBRATION_NS
        valence_cases[f"sprinkle d={d}"] = {"N": Ns, "valence_mean": [baselines[d]["valence_by_N"][N] for N in Ns]}
    plots.plot_valence_vs_N(
        valence_cases, str(RESULTS_DIR / "rules_valence_vs_N_overview.png"),
        title="All rules and sprinkles: valence vs N",
    )

    for family, r in best_per_family.items():
        rule = r["factory"]()
        small = grow(HASSE_N, rule, seed=HASSE_SEED)
        flag = r.get("flag")
        tag = f" [{flag.upper()}]" if flag else ""
        plots.plot_hasse(
            small.C, str(RESULTS_DIR / f"hasse_best_{family}.png"), coords=None,
            title=f"Best {family}: {r['label']}{tag}, N={HASSE_N}",
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(x, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, float) and not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def write_report(results: list, baselines: dict, thresholds: dict, width_forced_c: dict,
                  best_per_family: dict, output_path: Path) -> None:
    lines = []
    lines.append("# causet_lab growth-rule battery report")
    lines.append("")
    lines.append(
        "Tests whether any growth rule produces universes that pass the full measurement "
        "battery: do the two dimension estimators agree, does the estimate stay stable as "
        "N grows, does height scale the way a sprinkle's does, does the inner interval "
        "structure resemble a sprinkle of the matching dimension, and does link valence "
        "grow with N instead of staying flat like a lattice."
    )
    lines.append("")
    lines.append("## The frontier-collapse finding")
    lines.append("")
    lines.append(
        "The first version of R1 (`frontier_random`), R2 (`frontier_local`), R3 "
        "(`frontier_preferential`), R4 (`bounded_valence`), and R6 (`width_forced`) picked "
        "parents only from the current *maximal* elements (the frontier). That is fatally "
        "self-defeating: growth always starts from a single seed element, so the frontier "
        "starts at width 1, and \"pick k, or all of them if fewer than k exist\" always fires "
        "the collapse branch at width 1 for any k >= 1 -- every one of those rules "
        "deterministically produced a plain chain (height == N) regardless of parameters or "
        "seed. The fix, applied throughout this report: parents are chosen from *all* "
        "existing elements, not just the frontier, so frontier width becomes an emergent "
        "outcome (tracked below as widen vs deepen fraction) instead of something the "
        "selection rule can single-handedly destroy. See `rules.py` for the full "
        "implementation and reasoning."
    )
    lines.append("")

    lines.append("## Calibration (thresholds derived from the sprinkle controls, not hand-tuned)")
    lines.append("")
    lines.append(
        f"- Agreement baseline (mean |MM - midpoint| across sprinkle d=2,3,4): "
        f"{thresholds['agreement_baseline']:.3f} -> pass threshold (2x) = {2 * thresholds['agreement_baseline']:.3f}"
    )
    lines.append(
        f"- Drift baseline (mean |dimension drift vs log N| across sprinkle d=2,3,4): "
        f"{thresholds['drift_baseline']:.4f} -> pass threshold (2x) = {2 * thresholds['drift_baseline']:.4f}"
    )
    lines.append(
        f"- Abundance baseline (mean seed-to-seed L1 distance within same-d sprinkles at N=4000): "
        f"{thresholds['abundance_baseline']:.3f} -> pass threshold (2x) = {2 * thresholds['abundance_baseline']:.3f}"
    )
    lines.append(
        f"- Height-consistency band (given directly in the spec, not calibrated): "
        f"alpha * d_MM in [{HEIGHT_CONSISTENCY_BAND[0]}, {HEIGHT_CONSISTENCY_BAND[1]}]"
    )
    lines.append(
        f"- Valence-growth threshold (judgment call, documented rather than calibrated): "
        f"valence(N_max) / valence(N_min) >= {VALENCE_GROWTH_RATIO_THRESHOLD}"
    )
    lines.append(
        f"- width_forced calibration constant c (fit to real sprinkle frontier width at N=2000): "
        f"d_target=2 -> c={width_forced_c[2]:.3f}, d_target=4 -> c={width_forced_c[4]:.3f}"
    )
    lines.append("")
    lines.append("A rule PASSES only if its best parameter setting satisfies all five checks.")
    lines.append("")

    lines.append("## Verdict summary (best setting per rule family)")
    lines.append("")
    lines.append("| rule family | best setting | flag | verdict | checks passed | failure signature |")
    lines.append("|---|---|---|---|---|---|")
    family_order = list(best_per_family.keys())
    for family in family_order:
        r = best_per_family[family]
        v = r["verdict"]
        n_pass = sum(c["pass"] for c in v["checks"].values())
        flag = r.get("flag")
        flag_str = flag.upper() if flag else "-"
        verdict_str = "PASS" if v["overall_pass"] else "FAIL"
        sig = failure_signature(v)
        lines.append(f"| {family} | {r['label']} | {flag_str} | {verdict_str} | {n_pass}/5 | {sig} |")
    lines.append("")

    lines.append("## All configurations tried")
    lines.append("")
    lines.append(
        "| rule | agreement | stability (drift) | height consistency | "
        "inner structure (closest d) | valence ratio | checks | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        v = r["verdict"]
        n_pass = sum(c["pass"] for c in v["checks"].values())
        marks = {name: ("ok" if c["pass"] else "FAIL") for name, c in v["checks"].items()}
        notes = "; ".join(r["notes"]) if r["notes"] else ""
        lines.append(
            f"| {r['label']} | {_fmt(v['agreement'])} ({marks['agreement']}) | "
            f"{_fmt(v['drift'], 3)} ({marks['stability']}) | "
            f"{_fmt(v['height_consistency'])} ({marks['height_consistency']}) | "
            f"d={v['closest_d']}, {_fmt(v['abundance_distance'])} ({marks['inner_structure']}) | "
            f"{_fmt(v['valence_ratio'])} ({marks['valence_growth']}) | {n_pass}/5 | {notes} |"
        )
    lines.append("")

    lines.append("## Per-family detail")
    lines.append("")
    for family in family_order:
        items = [r for r in results if r["family"] == family]
        best = best_per_family[family]
        lines.append(f"### {family}")
        lines.append("")
        flag = best.get("flag")
        if flag == "illegal":
            lines.append(
                "**ILLEGAL CONTROL**: this rule reads raw label/index order, which violates "
                "discrete general covariance. Included only to show what cheating with labels buys you."
            )
            lines.append("")
        elif flag == "cheat":
            lines.append(
                "**CHEAT CONTROL**: this rule has the target dimension baked in by hand (the "
                "frontier-width constant c is calibrated from real sprinkles). Included only to "
                "test whether getting the frontier's shape right is sufficient on its own."
            )
            lines.append("")
        lines.append(f"Best setting: **{best['label']}** -- {'PASS' if best['verdict']['overall_pass'] else 'FAIL'}")
        lines.append("")
        lines.append(f"Signature: {failure_signature(best['verdict'])}")
        if best["widen_fraction"] is not None:
            lines.append("")
            lines.append(
                f"Widen fraction at N={best['frontier_trajectory_N']}: {best['widen_fraction']:.3f} "
                f"(fraction of growth steps that branched off below the tip rather than extending it)"
            )
        if best["notes"]:
            lines.append("")
            lines.append(f"Notes: {'; '.join(best['notes'])}")
        lines.append("")
        lines.append("| setting | agreement | stability | height consistency | inner structure | valence ratio | checks |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in items:
            v = r["verdict"]
            n_pass = sum(c["pass"] for c in v["checks"].values())
            lines.append(
                f"| {r['label']} | {_fmt(v['agreement'])} | {_fmt(v['drift'], 3)} | "
                f"{_fmt(v['height_consistency'])} | d={v['closest_d']}, {_fmt(v['abundance_distance'])} | "
                f"{_fmt(v['valence_ratio'])} | {n_pass}/5 |"
            )
        lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append("- `rules_dimension_vs_N_overview.png` -- every rule's dimension estimates vs N, sprinkle d=2,3,4 bands behind")
    lines.append("- `rules_dimension_vs_N_<family>.png` -- per-family detail")
    lines.append("- `rules_frontier_width_<family>.png` -- frontier width vs n during growth, log-log, per family")
    lines.append("- `rules_valence_vs_N_overview.png` -- link valence vs N, every rule and sprinkle")
    lines.append("- `hasse_best_<family>.png` -- Hasse diagram (links only) for the best setting of each family, N=100")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_study(study_ns=None, seeds=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    baselines, thresholds = compute_sprinkle_baselines()
    width_forced_c = {
        2: calibrate_width_forced_c(baselines, 2),
        4: calibrate_width_forced_c(baselines, 4),
    }
    configs = build_rule_configs(width_forced_c)
    print(f"[rules-study] running {len(configs)} rule configurations...")

    results = []
    for config in configs:
        r = run_rule_config(config, study_ns=study_ns, seeds=seeds)
        r["factory"] = config["factory"]
        r["verdict"] = compute_verdict(r, baselines, thresholds)
        results.append(r)

    best_per_family = pick_best_per_family(results)
    generate_plots(results, baselines, best_per_family)
    report_path = RESULTS_DIR / "rules_report.md"
    write_report(results, baselines, thresholds, width_forced_c, best_per_family, report_path)

    print(f"\n[rules-study] done in {time.time() - t0:.1f}s. wrote {report_path}")
    return {
        "results": results, "baselines": baselines, "thresholds": thresholds,
        "width_forced_c": width_forced_c, "best_per_family": best_per_family,
    }
