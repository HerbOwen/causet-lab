"""All plotting for causet_lab. Every function saves a PNG and returns the
output path. Uses the non-interactive Agg backend since this is a batch
CLI tool, not a notebook.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .causet import CausalSet  # noqa: E402

_COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _color_for(index: int) -> str:
    return _COLOR_CYCLE[index % len(_COLOR_CYCLE)]


def plot_dimension_vs_x(
    cases: dict,
    xlabel: str,
    outfile: str,
    ref_lines=(2, 3, 4),
    title: str = "Dimension estimates",
    xscale: str = "linear",
) -> str:
    """Plot both dimension estimators vs x for each case, with error bars.

    cases: {label: {"x": array, "mm_mean": array, "mm_std": array,
                     "mp_mean": array, "mp_std": array}}
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (label, d) in enumerate(cases.items()):
        color = _color_for(i)
        x = np.asarray(d["x"], dtype=float)
        ax.errorbar(
            x, d["mm_mean"], yerr=d.get("mm_std"), marker="o", linestyle="-",
            color=color, label=f"{label} (Myrheim-Meyer)", capsize=3,
        )
        ax.errorbar(
            x * 1.0, d["mp_mean"], yerr=d.get("mp_std"), marker="s", linestyle="--",
            color=color, alpha=0.6, label=f"{label} (midpoint)", capsize=3,
        )
    for ref in ref_lines:
        ax.axhline(ref, color="gray", linestyle=":", linewidth=1)
        ax.text(ax.get_xlim()[1] if xscale == "linear" else ax.get_xlim()[1], ref,
                 f" d={ref}", va="center", fontsize=8, color="gray")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("estimated dimension")
    ax.set_title(title)
    if xscale == "log":
        ax.set_xscale("log")
    ax.legend(fontsize=7, loc="best", ncol=1)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_abundance_profiles(
    profiles: dict,
    outfile: str,
    kmax: int = 10,
    title: str = "Interval abundance profiles",
) -> str:
    """Overlay interval-abundance profiles for several cases.

    profiles: {label: array of length kmax + 1}
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ks = np.arange(kmax + 1)
    width = 0.8 / max(len(profiles), 1)
    for i, (label, prof) in enumerate(profiles.items()):
        prof = np.asarray(prof, dtype=float)
        ax.plot(ks, prof, marker="o", label=label, color=_color_for(i))
    ax.set_xlabel("interval size k")
    ax.set_ylabel("fraction of related pairs with interval size k")
    ax.set_title(title)
    ax.set_xticks(ks)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_height_vs_N(
    cases: dict,
    outfile: str,
    ref_slopes=(0.5, 1 / 3, 0.25),
    title: str = "Height vs N",
) -> str:
    """Log-log plot of height vs N with reference power-law slopes.

    cases: {label: {"N": array, "height_mean": array, "height_std": array}}
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    all_N = []
    all_h = []
    for i, (label, d) in enumerate(cases.items()):
        N = np.asarray(d["N"], dtype=float)
        h = np.asarray(d["height_mean"], dtype=float)
        ax.errorbar(
            N, h, yerr=d.get("height_std"), marker="o", linestyle="-",
            color=_color_for(i), label=label, capsize=3,
        )
        all_N.append(N)
        all_h.append(h)
    ax.set_xscale("log")
    ax.set_yscale("log")

    if all_N:
        N_ref = np.concatenate(all_N)
        h_ref = np.concatenate(all_h)
        valid = h_ref > 0
        if valid.any():
            n0 = N_ref[valid].min()
            h0 = h_ref[valid][np.argmin(N_ref[valid])]
            n_line = np.array([N_ref[valid].min(), N_ref[valid].max()])
            for slope in ref_slopes:
                h_line = h0 * (n_line / n0) ** slope
                ax.plot(n_line, h_line, linestyle=":", color="gray", linewidth=1)
                ax.text(n_line[-1], h_line[-1], f" slope 1/{round(1/slope)}",
                         fontsize=7, color="gray", va="center")

    ax.set_xlabel("N")
    ax.set_ylabel("height (longest chain)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_hasse(
    C: np.ndarray,
    outfile: str,
    coords: np.ndarray = None,
    title: str = "Hasse diagram",
    seed: int = 0,
) -> str:
    """Draw a Hasse diagram showing only link (covering) relations.

    If ``coords`` is given (sprinkled sets), the true (x, t) coordinates
    are used: time on the vertical axis, the first spatial coordinate on
    the horizontal axis. Otherwise (grown sets), the vertical axis is each
    element's height (length of the longest chain ending there) and the
    horizontal position is an arbitrary jittered spread used only to keep
    points from overlapping -- it carries no physical meaning.
    """
    cset = CausalSet(np.asarray(C, dtype=bool))
    links = cset.links()
    N = cset.N
    rng = np.random.default_rng(seed)

    arbitrary_x = coords is None
    if coords is not None:
        y = coords[:, 0]
        x = coords[:, 1] if coords.shape[1] > 1 else np.zeros(N)
    else:
        levels = cset.chain_lengths()
        y = levels.astype(float)
        x = np.zeros(N)
        for lvl in np.unique(levels):
            idx = np.flatnonzero(levels == lvl)
            k = len(idx)
            spread = (np.arange(k) - (k - 1) / 2) / max(k, 1)
            jitter = rng.normal(0, 0.03, size=k)
            x[idx] = spread + jitter

    fig, ax = plt.subplots(figsize=(7, 7))
    ii, jj = np.nonzero(links)
    for i, j in zip(ii, jj):
        ax.plot([x[i], x[j]], [y[i], y[j]], color="steelblue", linewidth=0.7, alpha=0.7, zorder=1)
    ax.scatter(x, y, color="darkred", s=18, zorder=2)

    ax.set_ylabel("time t" if not arbitrary_x else "height (longest chain to element)")
    if arbitrary_x:
        ax.set_xlabel("horizontal position is ARBITRARY (layout only, no physical meaning)")
    else:
        ax.set_xlabel("spatial coordinate x (first spatial dimension only, if d > 2)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_dimension_vs_N_bands(
    cases: dict,
    sprinkle_bands: dict,
    outfile: str,
    title: str = "Dimension vs N",
    legend_cols: int = 2,
) -> str:
    """Dimension (both estimators) vs N for one or more rules, with
    sprinkle d=2,3,4 bands (mean +/- std across seeds) shaded behind.

    cases: {label: {"N": array, "mm_mean": array, "mm_std": array,
                     "mp_mean": array, "mp_std": array}}
    sprinkle_bands: {d: {"N": array, "mm_mean": array, "mm_std": array}}
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for d, band in sprinkle_bands.items():
        N = np.asarray(band["N"], dtype=float)
        mean = np.asarray(band["mm_mean"], dtype=float)
        std = np.nan_to_num(np.asarray(band["mm_std"], dtype=float))
        ax.fill_between(N, mean - std, mean + std, color="gray", alpha=0.12, zorder=0)
        ax.plot(N, mean, color="gray", linestyle=":", linewidth=1, zorder=0)
        ax.annotate(f"d={d}", (N[-1], mean[-1]), color="gray", fontsize=8,
                    xytext=(4, 0), textcoords="offset points", va="center")

    for i, (label, d) in enumerate(cases.items()):
        color = _color_for(i)
        N = np.asarray(d["N"], dtype=float)
        ax.errorbar(N, d["mm_mean"], yerr=d.get("mm_std"), marker="o", linestyle="-",
                     color=color, label=f"{label} (MM)", capsize=2, markersize=4, linewidth=1)
        ax.errorbar(N, d["mp_mean"], yerr=d.get("mp_std"), marker="s", linestyle="--",
                     color=color, alpha=0.55, label=f"{label} (mp)", capsize=2, markersize=4, linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("estimated dimension (sampled intervals)")
    ax.set_title(title)
    ax.legend(fontsize=6.5, ncol=legend_cols, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_frontier_width_vs_n(
    trajectories: dict,
    outfile: str,
    ref_slopes=(0.5, 2 / 3, 0.75),
    ref_labels=("(d-1)/d, d=2", "(d-1)/d, d=3", "(d-1)/d, d=4"),
    title: str = "Frontier width vs n",
) -> str:
    """Log-log plot of frontier (maximal-element) width vs universe size n
    during growth, for one or more rules, with reference power-law slopes.

    trajectories: {label: (n_array, width_array)}
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    all_n, all_w = [], []
    for i, (label, (n, w)) in enumerate(trajectories.items()):
        n = np.asarray(n, dtype=float)
        w = np.asarray(w, dtype=float)
        ax.plot(n, w, color=_color_for(i), label=label, linewidth=1, alpha=0.85)
        all_n.append(n)
        all_w.append(w)
    ax.set_xscale("log")
    ax.set_yscale("log")

    if all_n:
        n_ref = np.concatenate(all_n)
        w_ref = np.concatenate(all_w)
        valid = w_ref > 0
        if valid.any():
            n0 = n_ref[valid].min()
            w0 = w_ref[valid][np.argmin(n_ref[valid])]
            n_line = np.array([n_ref[valid].min(), n_ref[valid].max()])
            for slope, slabel in zip(ref_slopes, ref_labels):
                w_line = w0 * (n_line / n0) ** slope
                ax.plot(n_line, w_line, linestyle=":", color="gray", linewidth=1)
                ax.annotate(slabel, (n_line[-1], w_line[-1]), color="gray", fontsize=7,
                            xytext=(4, 0), textcoords="offset points", va="center")

    ax.set_xlabel("n (universe size during growth)")
    ax.set_ylabel("frontier width (# maximal elements)")
    ax.set_title(title)
    ax.legend(fontsize=6.5, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_valence_vs_N(
    cases: dict,
    outfile: str,
    title: str = "Link valence vs N",
) -> str:
    """Mean link valence vs N for rules and sprinkles together.

    cases: {label: {"N": array, "valence_mean": array}}
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (label, d) in enumerate(cases.items()):
        N = np.asarray(d["N"], dtype=float)
        v = np.asarray(d["valence_mean"], dtype=float)
        style = "--" if label.startswith("sprinkle") else "-"
        ax.plot(N, v, marker="o", linestyle=style, color=_color_for(i),
                 label=label, markersize=4, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("mean link valence")
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_distance_vs_x(
    points: list,
    outfile: str,
    xlabel: str = "k",
    threshold: float = None,
    title: str = "Inner-structure distance",
    xscale: str = "linear",
) -> str:
    """Inner-structure (abundance) distance to the closest-matching sprinkle
    vs some control parameter (e.g. k), with each point annotated by which
    sprinkle dimension it was closest to.

    points: list of (x, distance, closest_d) tuples, any order.
    """
    points = sorted(points, key=lambda p: p[0])
    xs = [p[0] for p in points]
    dists = [p[1] for p in points]
    closest = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xs, dists, marker="o", color=_color_for(0), linewidth=1.5)
    for x, d, cd in zip(xs, dists, closest):
        if d is not None and np.isfinite(d):
            ax.annotate(f"d={cd}", (x, d), fontsize=7, xytext=(4, 4),
                        textcoords="offset points", color="gray")
    if threshold is not None:
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=1)
        ax.annotate("pass threshold", (xs[-1], threshold), fontsize=8, color="gray",
                    xytext=(4, 2), textcoords="offset points")
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("abundance L1 distance to closest-matching sprinkle")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile
