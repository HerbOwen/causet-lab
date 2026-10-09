"""P_beta_c(S) double-peak check for all 10 n=50 seeds (5 randombg + 5 reglat).

5 of the 10 seeds were re-run under the fixed-window method (see
phase5_rerun_failed_n50_v2.py) and their raw WL/MUCA checkpoints are not
committed to this repo (tens of MB each). For those seeds this script
falls back to data/phase5_n50_pbetac_curves.pkl, a small (~25KB) cache of
the already-reweighted P_beta_c(S) curve and peak diagnostic extracted
from those checkpoints. If you have regenerated the checkpoints yourself
(see scripts/phase5/phase5_rerun_failed_n50_v2.py), place them under
data/checkpoints/newmethod/ and this script will use them directly
instead of the cache.
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from causet_lab.mcmc import muca
from causet_lab.mcmc.checkpoint import load_checkpoint

DATA = str(Path(__file__).resolve().parents[2] / "data")
NEW_CKPT_DIR = f"{DATA}/checkpoints/newmethod"
CURVE_CACHE = Path(DATA) / "phase5_n50_pbetac_curves.pkl"

with open(f"{DATA}/phase5_n50_final_merged.pkl", "rb") as f:
    d = pickle.load(f)

cache = pickle.loads(CURVE_CACHE.read_bytes()) if CURVE_CACHE.exists() else {}

def get_p_and_diag(model, seed, source, beta_c):
    if source == "v2_old_method":
        r = d[model][seed]
        ln_g, H_muca, bin_lo, bin_width, n_bins = r["ln_g"], r["H_muca"], r["bin_lo"], r["bin_width"], r["n_bins"]
    else:
        ckpt_names = (
            (f"randombg_wl_n50_bg{seed}.pkl", f"randombg_muca_n50_bg{seed}.pkl")
            if model == "randombg"
            else (f"reglat_via_rbg_wl_n50_cs{seed}.pkl", f"reglat_via_rbg_muca_n50_cs{seed}.pkl")
        )
        wl_path, muca_path = (Path(NEW_CKPT_DIR) / n for n in ckpt_names)
        if not (wl_path.exists() and muca_path.exists()):
            cached = cache.get((model, seed))
            if cached is None:
                raise FileNotFoundError(
                    f"No checkpoint and no cached curve for {model} seed {seed}; "
                    "regenerate via phase5_rerun_failed_n50_v2.py or restore the cache file."
                )
            return cached["P_beta_c_S"], cached["double_peak_diag"], True
        wl_ck, muca_ck = load_checkpoint(str(wl_path)), load_checkpoint(str(muca_path))
        ln_g, H_muca, bin_lo, bin_width, n_bins = wl_ck["ln_g"], muca_ck["H"], wl_ck["bin_lo"], wl_ck["bin_width"], wl_ck["n_bins"]
    p, centers = muca.reweight_P_beta_corrected(ln_g, H_muca, bin_lo, bin_width, n_bins, beta_c)
    return p, muca._peak_diagnostics(p), False

for model in ["randombg", "reglat"]:
    print(f"=== {model} P_beta_c(S) shape (n=50) ===")
    for seed in range(5):
        r = d[model][seed]
        beta_c = r["beta_c"]
        source = r["_source"]
        p, diag, from_cache = get_p_and_diag(model, seed, source, beta_c)
        tag = f"{source}, from cache" if from_cache else source
        if diag is None:
            print(f"  seed={seed} ({tag}): single broad hump (no double-peak structure)")
        else:
            print(f"  seed={seed} ({tag}): DOUBLE PEAK flagged, barrier={diag['barrier']:.3f}, peaks={diag['peaks']}")
print("PBETAC_CHECK_DONE")
