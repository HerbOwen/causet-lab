"""Scan every ever_visited bin-to-bin ln_g step for all 10 n=50 WL
checkpoints from the v2 (fixed-code) production run, flagging any step
that's wildly larger than the typical neighbor-to-neighbor step --
exactly the signature that trapped bg0 before the hot-end-cap fix.
Combinatorial sanity bound (per the user): ln(C(m,n)) for n=50,
m=4*50^2=10000 is ~several hundred at most, so any single adjacent-bin
ln_g step should be nowhere near that, let alone O(1000+).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from causet_lab.mcmc.checkpoint import load_checkpoint

CKPT_DIR = str(Path(__file__).resolve().parents[2] / "data" / "checkpoints")

files = {
    "randombg bg0": "randombg_wl_n50_bg0.pkl",
    "randombg bg1 (FAILED)": "randombg_wl_n50_bg1.pkl",
    "randombg bg2": "randombg_wl_n50_bg2.pkl",
    "randombg bg3": "randombg_wl_n50_bg3.pkl",
    "randombg bg4": "randombg_wl_n50_bg4.pkl",
    "reglat cs0": "reglat_via_rbg_wl_n50_cs0.pkl",
    "reglat cs1": "reglat_via_rbg_wl_n50_cs1.pkl",
    "reglat cs2 (FAILED)": "reglat_via_rbg_wl_n50_cs2.pkl",
    "reglat cs3 (FAILED)": "reglat_via_rbg_wl_n50_cs3.pkl",
    "reglat cs4": "reglat_via_rbg_wl_n50_cs4.pkl",
}

for label, fname in files.items():
    ck = load_checkpoint(f"{CKPT_DIR}/{fname}")
    ln_g = ck["ln_g"]
    ever_visited = ck["ever_visited"]
    idx = np.flatnonzero(ever_visited)
    lo, hi = idx[0], idx[-1]
    # consecutive ever_visited bins within [lo, hi] (treat gaps in
    # ever_visited itself -- unvisited bins sandwiched between visited
    # ones -- as a separate thing to flag, not silently skip)
    full_range = np.arange(lo, hi + 1)
    unvisited_holes = full_range[~ever_visited[full_range]]
    steps = np.diff(ln_g[full_range])
    abs_steps = np.abs(steps)
    worst_idx = np.argmax(abs_steps)
    median_step = np.median(abs_steps[abs_steps > 0]) if np.any(abs_steps > 0) else 0.0
    print(f"{label}: n_bins={ck['n_bins']} ever_visited=[{lo},{hi}] ({int(ever_visited.sum())} bins) "
          f"stage={ck['stage']} f_mod={ck['f_mod']:.2e}")
    print(f"  holes (unvisited within span): {len(unvisited_holes)} {list(unvisited_holes) if len(unvisited_holes) <= 10 else '(>10)'}")
    print(f"  median |step|={median_step:.3f}  worst |step|={abs_steps[worst_idx]:.3f} "
          f"at bin {full_range[worst_idx]}->{full_range[worst_idx]+1} "
          f"(ln_g: {ln_g[full_range[worst_idx]]:.2f} -> {ln_g[full_range[worst_idx]+1]:.2f})")
    flag = "  *** ANOMALOUS (>50x median, or >100 absolute) ***" if (abs_steps[worst_idx] > 100 or (median_step > 0 and abs_steps[worst_idx] > 50 * median_step)) else ""
    print(f"  {flag}")
    print()

print("SCAN_DONE")
