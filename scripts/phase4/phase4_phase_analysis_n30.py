import time, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from causet_lab.mcmc import lattice_gas as lgm
from causet_lab.generators import sprinkle
from causet_lab.battery import analyze_matrix, nanmean, nanmean_axis0
from causet_lab.measures import abundance_distance

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")

n = 30
eps = lgm.EPS_DEFAULT
w, h, m = lgm.lattice_dims(n)

with open(f"{SCRATCH}/phase4_pilot_n30_result.pkl", "rb") as f:
    prev = pickle.load(f)
beta_c = prev["beta_c_res"]["beta_c"]
print(f"[analysis] n={n} eps={eps} beta_c(located)={beta_c:.4f}", flush=True)

kmax = 6
N_SEEDS = 10

def measure_sample(L, ref):
    C = lgm.lattice_to_matrix(L, n, w)
    res = analyze_matrix(C, seed=0, n_samples=300, min_size=2, max_size=n, kmax=kmax)
    of_whole = res["ordering_fraction"]
    h_whole = res["height"]
    layers = lgm.layer_count(L, n, w)
    return res["mm_dim_sampled_mean"], res["abundance_profile"], of_whole, h_whole, layers

ref_cset, _ = sprinkle(3000, 2, seed=0)
ref = analyze_matrix(ref_cset.C, seed=0, n_samples=300, min_size=2, max_size=n, kmax=kmax)

# ---- Low beta (hot): uniformly random fillings, beta=0 ----
t0 = time.time()
mm_dims_lo, profiles_lo, of_lo, h_lo, layers_lo = [], [], [], [], []
for seed in range(N_SEEDS):
    L = lgm.random_filling(n, m, seed=seed)
    mm, prof, of_w, hh, ly = measure_sample(L, ref)
    mm_dims_lo.append(mm); profiles_lo.append(prof); of_lo.append(of_w); h_lo.append(hh); layers_lo.append(ly)
t_lo = time.time() - t0
print(f"[analysis] low-beta (beta=0) done in {t_lo:.1f}s", flush=True)

# ---- High beta (cold/layered): equilibrated chains at beta = 4*beta_c ----
beta_hi = 4.0 * beta_c
N_SWEEPS = 400
t0 = time.time()
mm_dims_hi, profiles_hi, of_hi, h_hi, layers_hi = [], [], [], [], []
for seed in range(N_SEEDS):
    actions, L, future, past, counts = lgm.run_pilot_chain(
        n, w, h, m, eps, beta=beta_hi, seed=1000 + seed,
        n_sweeps=N_SWEEPS, burn_in=N_SWEEPS - 1, measure_every=1, anneal_from=0.0,
    )
    mm, prof, of_w, hh, ly = measure_sample(L, ref)
    mm_dims_hi.append(mm); profiles_hi.append(prof); of_hi.append(of_w); h_hi.append(hh); layers_hi.append(ly)
t_hi = time.time() - t0
print(f"[analysis] high-beta (beta={beta_hi:.2f}) done in {t_hi:.1f}s", flush=True)

mean_profile_lo = nanmean_axis0(np.array(profiles_lo))
mean_profile_hi = nanmean_axis0(np.array(profiles_hi))
dist_lo = abundance_distance(mean_profile_lo, ref["abundance_profile"])
dist_hi = abundance_distance(mean_profile_hi, ref["abundance_profile"])

summary = {
    "n": n, "eps": eps, "beta_c": beta_c, "beta_hi": beta_hi,
    "lo": {"mm_dim": nanmean(mm_dims_lo), "of_whole": nanmean(of_lo), "height_whole": nanmean(h_lo),
           "layers": nanmean(layers_lo), "abundance_dist_to_sprinkle": dist_lo,
           "mm_dims_raw": mm_dims_lo, "of_raw": of_lo, "height_raw": h_lo, "layers_raw": layers_lo},
    "hi": {"mm_dim": nanmean(mm_dims_hi), "of_whole": nanmean(of_hi), "height_whole": nanmean(h_hi),
           "layers": nanmean(layers_hi), "abundance_dist_to_sprinkle": dist_hi,
           "mm_dims_raw": mm_dims_hi, "of_raw": of_hi, "height_raw": h_hi, "layers_raw": layers_hi},
    "t_lo": t_lo, "t_hi": t_hi,
}
print("\n=== SUMMARY ===")
for k in ["lo", "hi"]:
    s = summary[k]
    print(f"{k}: mm_dim={s['mm_dim']:.3f} of_whole={s['of_whole']:.4f} height_whole={s['height_whole']:.2f} "
          f"layers={s['layers']:.2f} abundance_dist={s['abundance_dist_to_sprinkle']:.3f}")

with open(f"{SCRATCH}/phase4_phase_analysis_n30_result.pkl", "wb") as f:
    pickle.dump(summary, f)
print("ANALYSIS DONE", flush=True)
