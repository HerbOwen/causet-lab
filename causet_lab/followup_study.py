"""Follow-up to rules_study.py / results/rules_report.md.

local_parents showed inner-structure distance falling steadily with k
(0.42, 0.24, 0.11, 0.07 at k=2,3,5,8; pass threshold 0.059, closest d=3)
while every fixed-k rule in that batch had structurally flat valence. This
module tests three ideas built on that observation:

1. local_parents at larger fixed k (12, 16, 24, 32): does the
   inner-structure distance keep falling and cross the threshold, and what
   happens to dimension drift as k grows?
2. local_parents_adaptive(c, beta): the number of parents grows with the
   picked element's own past size, k = max(1, round(c * (1+|past(x)|)^beta)).
   This lets valence grow with the universe using only local, label-free
   information -- no total size N, no target dimension baked in.
3. frontier_weighted(a, k): picks the anchor element with probability
   weighted toward small |future(x)| (near the causal "present") instead of
   relying on frontier/maximal-element bookkeeping, which is what caused
   the original R1-R4/R6 collapse in rules_study.py.

Check 4 (inner structure) is tightened here: a config only passes it if the
closest-matching sprinkle dimension equals round(d_MM), AND round(d_midpoint)
agrees with that same integer. Any config passing 4 or 5 of the 5 checks at
the main N grid gets a confirmation rerun: 5 fresh seeds at N=5000, checked
against sprinkle controls also generated at N=5000, before being called a
pass.
"""
from __future__ import annotations

import time

import numpy as np

from .generators import grow
from .rules import local_parents, local_parents_adaptive, frontier_weighted
from .battery import analyze_matrix, nanmean, nanmean_axis0
from .measures import dimension_drift, height_exponent, abundance_distance
from . import plots
from .rules_study import (
    RESULTS_DIR, STUDY_NS, SEEDS, SPRINKLE_CALIBRATION_NS,
    N_SAMPLES, MIN_INTERVAL, MAX_INTERVAL, KMAX,
    HEIGHT_CONSISTENCY_BAND, VALENCE_GROWTH_RATIO_THRESHOLD, HASSE_N, HASSE_SEED,
    compute_sprinkle_baselines, run_rule_config, pick_best_per_family, _sprinkle_bands,
)

CONFIRM_N = 5000
CONFIRM_SEEDS = [1000, 1001, 1002, 1003, 1004]


# ---------------------------------------------------------------------------
# Rule configuration grid
# ---------------------------------------------------------------------------

def build_followup_configs() -> list:
    configs = []
    for k in (2, 3, 5, 8, 12, 16, 24, 32):
        configs.append({
            "family": "local_parents", "label": f"local_parents(k={k})",
            "params": {"k": k}, "factory": (lambda k=k: local_parents(k)), "flag": None,
        })
    for beta in (0.25, 0.33, 0.5):
        for c in (1, 2, 4):
            configs.append({
                "family": "local_parents_adaptive", "label": f"local_parents_adaptive(c={c},beta={beta})",
                "params": {"c": c, "beta": beta},
                "factory": (lambda c=c, beta=beta: local_parents_adaptive(c, beta)), "flag": None,
            })
    for a in (0.5, 1, 2):
        for k in (5, 8):
            configs.append({
                "family": "frontier_weighted", "label": f"frontier_weighted(a={a},k={k})",
                "params": {"a": a, "k": k},
                "factory": (lambda a=a, k=k: frontier_weighted(a, k)), "flag": None,
            })
    return configs


# ---------------------------------------------------------------------------
# Verdicts (tightened check 4)
# ---------------------------------------------------------------------------

def compute_verdict_tightened(result: dict, baselines: dict, thresholds: dict) -> dict:
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
    dMP = nanmean(mp_means)
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

    dMM_round = int(round(dMM)) if np.isfinite(dMM) else None
    dMP_round = int(round(dMP)) if np.isfinite(dMP) else None
    dims_match = bool(dMM_round is not None and dMP_round is not None
                       and dMM_round == dMP_round and best_d == dMM_round)

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
            "dMM_round": dMM_round, "dMP_round": dMP_round, "dims_match": dims_match,
            "pass": bool(np.isfinite(best_distance) and best_distance <= abundance_threshold and dims_match),
        },
        "valence_growth": {
            "value": valence_ratio, "threshold": VALENCE_GROWTH_RATIO_THRESHOLD,
            "pass": bool(np.isfinite(valence_ratio) and valence_ratio >= VALENCE_GROWTH_RATIO_THRESHOLD),
        },
    }
    overall_pass = all(c["pass"] for c in checks.values())
    return {
        "checks": checks, "overall_pass": overall_pass, "agreement": agreement, "drift": drift,
        "alpha": alpha, "dMM": dMM, "dMP": dMP, "dMM_round": dMM_round, "dMP_round": dMP_round,
        "height_consistency": height_consistency, "valence_ratio": valence_ratio,
        "closest_d": best_d, "abundance_distance": best_distance, "dims_match": dims_match,
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
        ic = checks["inner_structure"]
        if not ic["dims_match"]:
            parts.append(
                f"dimension estimates don't round-agree (d_MM~{ic['dMM_round']}, d_mp~{ic['dMP_round']}, "
                f"closest sprinkle d={ic['closest_d']})"
            )
        else:
            parts.append(
                f"abundance profile far from matching sprinkle (distance={verdict['abundance_distance']:.2f} "
                f"> {ic['threshold']:.2f})"
            )
    if "valence_growth" in failed:
        vr = verdict["valence_ratio"]
        vr_str = f"{vr:.2f}" if np.isfinite(vr) else "nan"
        parts.append(f"valence flat (ratio={vr_str} < {VALENCE_GROWTH_RATIO_THRESHOLD})")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Confirmation rerun at N=5000 with fresh seeds
# ---------------------------------------------------------------------------

def run_confirmation(config: dict, baselines: dict, thresholds: dict) -> dict:
    t0 = time.time()
    rows = []
    for seed in CONFIRM_SEEDS:
        rule = config["factory"]()
        cset = grow(CONFIRM_N, rule, seed=seed)
        res = analyze_matrix(
            cset.C, seed=seed, n_samples=N_SAMPLES, min_size=MIN_INTERVAL,
            max_size=MAX_INTERVAL, kmax=KMAX,
        )
        rows.append({"N": CONFIRM_N, "seed": seed, **res})

    mm = np.array([r["mm_dim_sampled_mean"] for r in rows])
    mp = np.array([r["mp_dim_sampled_mean"] for r in rows])
    agreement = nanmean(np.abs(mm - mp))
    dMM = nanmean(mm)
    dMP = nanmean(mp)
    profile = nanmean_axis0([r["abundance_profile"] for r in rows])
    distances = {}
    for d in (2, 3, 4):
        sp = baselines[d]["profile_by_N"].get(CONFIRM_N)
        distances[d] = abundance_distance(profile, sp) if sp is not None else float("nan")
    finite = {d: v for d, v in distances.items() if np.isfinite(v)}
    if finite:
        best_d = min(finite, key=finite.get)
        best_distance = finite[best_d]
    else:
        best_d, best_distance = None, float("nan")

    dMM_round = int(round(dMM)) if np.isfinite(dMM) else None
    dMP_round = int(round(dMP)) if np.isfinite(dMP) else None
    dims_match = bool(dMM_round is not None and dMP_round is not None
                       and dMM_round == dMP_round and best_d == dMM_round)

    agreement_threshold = 2 * thresholds["agreement_baseline"]
    abundance_threshold = 2 * thresholds["abundance_baseline"]
    agreement_pass = bool(np.isfinite(agreement) and agreement <= agreement_threshold)
    inner_pass = bool(np.isfinite(best_distance) and best_distance <= abundance_threshold and dims_match)
    confirmed = agreement_pass and inner_pass

    print(
        f"[rules-study] confirmation rerun {config['label']} at N={CONFIRM_N} "
        f"done ({time.time() - t0:.1f}s): agreement={agreement:.3f} "
        f"({'pass' if agreement_pass else 'FAIL'}), d_MM~{dMM_round} d_mp~{dMP_round} "
        f"closest_d={best_d} dist={best_distance:.3f} ({'pass' if inner_pass else 'FAIL'}) "
        f"-> {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}"
    )
    return {
        "N": CONFIRM_N, "agreement": agreement, "agreement_pass": agreement_pass,
        "dMM": dMM, "dMP": dMP, "dMM_round": dMM_round, "dMP_round": dMP_round,
        "closest_d": best_d, "abundance_distance": best_distance, "inner_pass": inner_pass,
        "confirmed": confirmed,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_followup_plots(results: list, baselines: dict, thresholds: dict, best_per_family: dict) -> None:
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
        dim_cases(results), bands, str(RESULTS_DIR / "followup_dimension_vs_N_overview.png"),
        title="Follow-up rules: dimension vs N (sprinkle bands behind)", legend_cols=2,
    )

    by_family = {}
    for r in results:
        by_family.setdefault(r["family"], []).append(r)

    for family, items in by_family.items():
        plots.plot_dimension_vs_N_bands(
            dim_cases(items), bands, str(RESULTS_DIR / f"followup_dimension_vs_N_{family}.png"),
            title=f"{family}: dimension vs N (sprinkle bands behind)",
        )
        trajectories = {r["label"]: r["frontier_trajectory"] for r in items if r["frontier_trajectory"] is not None}
        if trajectories:
            plots.plot_frontier_width_vs_n(
                trajectories, str(RESULTS_DIR / f"followup_frontier_width_{family}.png"),
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
        valence_cases, str(RESULTS_DIR / "followup_valence_vs_N_overview.png"),
        title="Follow-up rules and sprinkles: valence vs N",
    )

    lp_items = [r for r in results if r["family"] == "local_parents"]
    points = [(r["params"]["k"], r["verdict"]["abundance_distance"], r["verdict"]["closest_d"]) for r in lp_items]
    plots.plot_distance_vs_x(
        points, str(RESULTS_DIR / "followup_local_parents_distance_vs_k.png"),
        xlabel="k", threshold=2 * thresholds["abundance_baseline"],
        title="local_parents: inner-structure distance vs k (includes k=2,3,5,8 from rules_report.md)",
    )

    for family, r in best_per_family.items():
        rule = r["factory"]()
        small = grow(HASSE_N, rule, seed=HASSE_SEED)
        plots.plot_hasse(
            small.C, str(RESULTS_DIR / f"hasse_followup_best_{family}.png"), coords=None,
            title=f"Best {family}: {r['label']}, N={HASSE_N}",
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


def write_followup_report(results: list, baselines: dict, thresholds: dict, best_per_family: dict, output_path) -> None:
    lines = []
    lines.append("# causet_lab follow-up report: locality + growing valence")
    lines.append("")
    lines.append(
        "Follow-up to `results/rules_report.md`. `local_parents` showed inner-structure distance "
        "falling steadily with k (0.42, 0.24, 0.11, 0.07 at k=2,3,5,8; pass threshold 0.059, closest "
        "d=3) while every fixed-k rule had structurally flat valence. Tested here: (1) larger fixed "
        "k for local_parents, (2) `local_parents_adaptive`, where the parent count grows with the "
        "picked element's own past size (no labels, no N, no target dimension), and (3) "
        "`frontier_weighted`, preferring elements near the causal present (small future) measured "
        "purely from order."
    )
    lines.append("")
    lines.append("## Tightened inner-structure check")
    lines.append("")
    lines.append(
        "Check 4 now requires three numbers to agree, not just a distance threshold: the closest "
        "matching sprinkle dimension must equal round(d_MM), and round(d_midpoint) must equal that "
        "same integer too. A config can have a small abundance distance and still fail this check "
        "if the two dimension estimators round to different integers, or round to a different "
        "integer than the sprinkle profile it's closest to."
    )
    lines.append("")
    lines.append(f"- Agreement threshold (2x sprinkle baseline): {2 * thresholds['agreement_baseline']:.3f}")
    lines.append(f"- Drift threshold (2x sprinkle baseline): {2 * thresholds['drift_baseline']:.4f}")
    lines.append(f"- Abundance threshold (2x sprinkle baseline): {2 * thresholds['abundance_baseline']:.3f}")
    lines.append(f"- Height-consistency band: alpha * d_MM in [{HEIGHT_CONSISTENCY_BAND[0]}, {HEIGHT_CONSISTENCY_BAND[1]}]")
    lines.append(f"- Valence-growth threshold: valence(N_max) / valence(N_min) >= {VALENCE_GROWTH_RATIO_THRESHOLD}")
    lines.append(f"- Confirmation rerun: N={CONFIRM_N}, 5 fresh seeds, triggered for any config passing >= 4/5 checks")
    lines.append("")

    lines.append("## Verdict summary (best setting per family)")
    lines.append("")
    lines.append("| family | best setting | verdict | checks | d_MM~ / d_mp~ / closest d | confirmation @N=5000 | failure signature |")
    lines.append("|---|---|---|---|---|---|---|")
    family_order = list(best_per_family.keys())
    for family in family_order:
        r = best_per_family[family]
        v = r["verdict"]
        n_pass = sum(c["pass"] for c in v["checks"].values())
        verdict_str = "PASS" if v["overall_pass"] else "FAIL"
        conf = r.get("confirmation")
        conf_str = "-" if conf is None else ("CONFIRMED" if conf["confirmed"] else "not confirmed")
        dims_str = f"{v['dMM_round']} / {v['dMP_round']} / {v['closest_d']}"
        sig = failure_signature(v)
        lines.append(f"| {family} | {r['label']} | {verdict_str} | {n_pass}/5 | {dims_str} | {conf_str} | {sig} |")
    lines.append("")

    lines.append("## All configurations tried")
    lines.append("")
    lines.append(
        "| rule | agreement | stability | height consist. | d_MM~/d_mp~/closest | abundance dist | "
        "valence ratio | checks | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        v = r["verdict"]
        n_pass = sum(c["pass"] for c in v["checks"].values())
        notes = list(r["notes"]) if r["notes"] else []
        if r.get("cap_hit_stats"):
            ch = r["cap_hit_stats"]
            rate = (ch["hits"] / ch["calls"]) if ch["calls"] else 0.0
            notes.append(f"candidate-sample cap hit {ch['hits']}/{ch['calls']} steps ({rate:.1%})")
        notes_str = "; ".join(notes)
        lines.append(
            f"| {r['label']} | {_fmt(v['agreement'])} | {_fmt(v['drift'], 3)} | {_fmt(v['height_consistency'])} | "
            f"{v['dMM_round']}/{v['dMP_round']}/{v['closest_d']} | {_fmt(v['abundance_distance'])} | "
            f"{_fmt(v['valence_ratio'])} | {n_pass}/5 | {notes_str} |"
        )
    lines.append("")

    lines.append("## Confirmation reruns (configs passing >= 4/5 checks at the main N grid)")
    lines.append("")
    confirmed_any = [r for r in results if r.get("confirmation") is not None]
    if not confirmed_any:
        lines.append(
            "No configuration passed 4 or more checks at the main N grid, so no confirmation "
            "rerun was triggered."
        )
    else:
        lines.append(f"| rule | agreement @N={CONFIRM_N} | d_MM~ / d_mp~ / closest d | abundance dist | confirmed |")
        lines.append("|---|---|---|---|---|")
        for r in confirmed_any:
            conf = r["confirmation"]
            lines.append(
                f"| {r['label']} | {_fmt(conf['agreement'])} | "
                f"{conf['dMM_round']} / {conf['dMP_round']} / {conf['closest_d']} | "
                f"{_fmt(conf['abundance_distance'])} | {'YES' if conf['confirmed'] else 'no'} |"
            )
    lines.append("")

    lines.append("## Per-family detail")
    lines.append("")
    for family in family_order:
        items = [r for r in results if r["family"] == family]
        best = best_per_family[family]
        lines.append(f"### {family}")
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
        if best.get("cap_hit_stats"):
            ch = best["cap_hit_stats"]
            rate = (ch["hits"] / ch["calls"]) if ch["calls"] else 0.0
            lines.append("")
            lines.append(f"Candidate-sample cap hit {ch['hits']}/{ch['calls']} growth steps ({rate:.1%}).")
        if best.get("notes"):
            lines.append("")
            lines.append(f"Notes: {'; '.join(best['notes'])}")
        lines.append("")
        lines.append(
            "| setting | agreement | stability | height consist. | d_MM~/d_mp~/closest | abundance dist | "
            "valence ratio | checks |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in items:
            v = r["verdict"]
            n_pass = sum(c["pass"] for c in v["checks"].values())
            lines.append(
                f"| {r['label']} | {_fmt(v['agreement'])} | {_fmt(v['drift'], 3)} | {_fmt(v['height_consistency'])} | "
                f"{v['dMM_round']}/{v['dMP_round']}/{v['closest_d']} | {_fmt(v['abundance_distance'])} | "
                f"{_fmt(v['valence_ratio'])} | {n_pass}/5 |"
            )
        lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append("- `followup_dimension_vs_N_overview.png` / `followup_dimension_vs_N_<family>.png`")
    lines.append("- `followup_frontier_width_<family>.png` -- frontier width vs n during growth, log-log")
    lines.append("- `followup_valence_vs_N_overview.png` -- link valence vs N, every rule and sprinkle")
    lines.append(
        "- `followup_local_parents_distance_vs_k.png` -- inner-structure distance vs k, "
        "k=2..32, extending the k=2,3,5,8 points from rules_report.md"
    )
    lines.append("- `hasse_followup_best_<family>.png` -- Hasse diagram for the best setting of each family, N=100")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_followup_study(study_ns=None, seeds=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    calibration_ns = sorted(set(SPRINKLE_CALIBRATION_NS) | {CONFIRM_N})
    baselines, thresholds = compute_sprinkle_baselines(calibration_ns=calibration_ns)

    configs = build_followup_configs()
    print(f"[rules-study] running {len(configs)} follow-up rule configurations...")

    results = []
    for config in configs:
        r = run_rule_config(config, study_ns=study_ns, seeds=seeds)
        r["factory"] = config["factory"]
        r["verdict"] = compute_verdict_tightened(r, baselines, thresholds)
        n_pass = sum(c["pass"] for c in r["verdict"]["checks"].values())
        r["confirmation"] = run_confirmation(config, baselines, thresholds) if n_pass >= 4 else None
        results.append(r)

    best_per_family = pick_best_per_family(results)
    generate_followup_plots(results, baselines, thresholds, best_per_family)
    report_path = RESULTS_DIR / "followup_report.md"
    write_followup_report(results, baselines, thresholds, best_per_family, report_path)

    print(f"\n[rules-study] follow-up done in {time.time() - t0:.1f}s. wrote {report_path}")
    return {
        "results": results, "baselines": baselines, "thresholds": thresholds,
        "best_per_family": best_per_family,
    }
