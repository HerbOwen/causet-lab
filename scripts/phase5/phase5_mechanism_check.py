import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from causet_lab.mcmc import random_bg as rbgm
from causet_lab.mcmc import muca

CK = str(Path(__file__).resolve().parents[2] / "data" / "checkpoints")
SCRATCH = str(Path(__file__).resolve().parents[2] / "data")

n, eps = 30, 0.1
w, h, m = rbgm.background_dims(n)


def greedy_packed_floor(site_t, site_x, n, w, eps):
    """Deterministic, non-MCMC 'best hand-built' construction: pick the
    globally earliest site as an apex, then take the (n-1) sites in its
    forward lightcone with smallest t -- a compact, densely-interrelated
    cluster by construction (not claimed globally optimal)."""
    m = len(site_t)
    apex = int(np.argmin(site_t))
    t_a, x_a = site_t[apex], site_x[apex]
    dt = site_t - t_a
    dx = (site_x - x_a) % w
    dx = np.minimum(dx, w - dx)
    in_future = (dt > 0) & (dx <= dt)
    in_future[apex] = False
    future_ids = np.flatnonzero(in_future)
    order = np.argsort(site_t[future_ids])
    chosen_future = future_ids[order[: n - 1]]
    selected = np.concatenate([[apex], chosen_future])
    rest = np.setdiff1d(np.arange(m), selected, assume_unique=False)
    L = np.concatenate([selected, rest]).astype(np.int64)
    S, _Nk = rbgm.randombg_action_full(L, n, site_t, site_x, w, eps)
    return S


# ---- Regular lattice (Phase 4 original checkpoint) ----
with open(f"{CK}/lattice_wl_n30.pkl", "rb") as f:
    reg_wl = pickle.load(f)
with open(f"{CK}/lattice_muca_n30.pkl", "rb") as f:
    reg_muca = pickle.load(f)

ever_visited = reg_wl["ever_visited"]
lo_idx = int(np.flatnonzero(ever_visited)[0])
reg_lowest_S = reg_wl["bin_lo"] + lo_idx * reg_wl["bin_width"]
reg_mean_S_hot, _ = muca.reweight_mean_var_S(
    reg_wl["ln_g"], reg_muca["H"], reg_muca["bin_lo"], reg_muca["bin_width"], reg_muca["n_bins"], 0.0,
)

site_ids = np.arange(m)
reg_site_t = (site_ids // w).astype(np.float64)
reg_site_x = (site_ids % w).astype(np.float64)
reg_floor_S = greedy_packed_floor(reg_site_t, reg_site_x, n, w, eps)

print(f"REGULAR LATTICE: lowest_S_reached(WL)={reg_lowest_S:.4f}  greedy_floor_S={reg_floor_S:.4f}  "
      f"hot_mean_S(beta=0)={reg_mean_S_hot:.4f}")

# ---- Random backgrounds (5 realizations) ----
print()
rows = []
for seed in range(5):
    with open(f"{CK}/randombg_wl_n30_bg{seed}.pkl", "rb") as f:
        wl = pickle.load(f)
    with open(f"{CK}/randombg_muca_n30_bg{seed}.pkl", "rb") as f:
        mc = pickle.load(f)
    ever_visited = wl["ever_visited"]
    lo_idx = int(np.flatnonzero(ever_visited)[0])
    lowest_S = wl["bin_lo"] + lo_idx * wl["bin_width"]
    mean_S_hot, _ = muca.reweight_mean_var_S(wl["ln_g"], mc["H"], mc["bin_lo"], mc["bin_width"], mc["n_bins"], 0.0)

    site_t, site_x = rbgm.sprinkle_background(m, w, h, seed=seed)  # deterministic regen, same seed
    floor_S = greedy_packed_floor(site_t, site_x, n, w, eps)

    print(f"RANDOM BG seed={seed}: lowest_S_reached(WL)={lowest_S:.4f}  greedy_floor_S={floor_S:.4f}  "
          f"hot_mean_S(beta=0)={mean_S_hot:.4f}")
    rows.append((lowest_S, floor_S, mean_S_hot))

rows = np.array(rows)
print()
print(f"RANDOM BG mean +/- std: lowest_S_reached={rows[:,0].mean():.4f}+/-{rows[:,0].std(ddof=1):.4f}  "
      f"greedy_floor_S={rows[:,1].mean():.4f}+/-{rows[:,1].std(ddof=1):.4f}  "
      f"hot_mean_S={rows[:,2].mean():.4f}+/-{rows[:,2].std(ddof=1):.4f}")

with open(f"{SCRATCH}/phase5_mechanism_check_results.pkl", "wb") as f:
    pickle.dump({
        "regular": {"lowest_S": reg_lowest_S, "floor_S": reg_floor_S, "hot_mean_S": reg_mean_S_hot},
        "random_bg_rows": rows.tolist(),
    }, f)
print("MECHANISM_CHECK_DONE")
