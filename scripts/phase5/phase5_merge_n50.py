"""Merge the final n=50 dataset: for seeds re-run under the new
fixed-window method (randombg 0,1; reglat 0,2,3), use those results
(seed0 in both models is a fairness cross-check, already confirmed
consistent with its old v2 value). For untouched seeds (randombg
2,3,4; reglat 1,4), keep the original v2 (old adaptive-widen method)
results, which showed no anomalies and reliable round-trip counts.
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")

with open(f"{SCRATCH}/phase5_randombg_n50_results.pkl", "rb") as f:
    rb_old = pickle.load(f)
with open(f"{SCRATCH}/phase5_reglat_via_rbg_n50_results.pkl", "rb") as f:
    rl_old = pickle.load(f)
with open(f"{SCRATCH}/phase5_newmethod_n50_results.pkl", "rb") as f:
    newm = pickle.load(f)

randombg_final = {}
for seed in range(5):
    if ("randombg", seed) in newm:
        randombg_final[seed] = newm[("randombg", seed)]
        randombg_final[seed]["_source"] = "newmethod"
    else:
        randombg_final[seed] = rb_old[seed]
        randombg_final[seed]["_source"] = "v2_old_method"

reglat_final = {}
for seed in range(5):
    if ("reglat", seed) in newm:
        reglat_final[seed] = newm[("reglat", seed)]
        reglat_final[seed]["_source"] = "newmethod"
    else:
        reglat_final[seed] = rl_old[seed]
        reglat_final[seed]["_source"] = "v2_old_method"

print("=== Final n=50 randombg dataset ===")
betas_rb = []
for seed in range(5):
    r = randombg_final[seed]
    betas_rb.append(r["beta_c"])
    print(f"  seed={seed} source={r['_source']} beta_c={r['beta_c']:.4f}")
betas_rb = np.array(betas_rb)
print(f"  MEAN={betas_rb.mean():.4f} STD={betas_rb.std(ddof=1):.4f} SEM={betas_rb.std(ddof=1)/np.sqrt(5):.4f}")

print("\n=== Final n=50 reglat (regular lattice via rbg) dataset ===")
betas_rl = []
for seed in range(5):
    r = reglat_final[seed]
    betas_rl.append(r["beta_c"])
    print(f"  seed={seed} source={r['_source']} beta_c={r['beta_c']:.4f}")
betas_rl = np.array(betas_rl)
print(f"  MEAN={betas_rl.mean():.4f} STD={betas_rl.std(ddof=1):.4f} SEM={betas_rl.std(ddof=1)/np.sqrt(5):.4f}")

gap = betas_rl.mean() - betas_rb.mean()
sem_rb = betas_rb.std(ddof=1) / np.sqrt(5)
sem_rl = betas_rl.std(ddof=1) / np.sqrt(5)
combined_sem = float(np.sqrt(sem_rb**2 + sem_rl**2))
z = gap / combined_sem
pct_shift = 100.0 * gap / betas_rl.mean()
print(f"\nGap (reglat - randombg) = {gap:.4f}  combined_SEM={combined_sem:.4f}  z={z:.2f}  pct_shift={pct_shift:.2f}%")

# n=30 comparison (from the committed Phase 5 stage-1 report)
n30_rl_mean, n30_rl_sem = 11.095, 0.005
n30_rb_mean, n30_rb_sem = 9.723, 0.026
n30_gap = n30_rl_mean - n30_rb_mean
n30_pct = 100.0 * n30_gap / n30_rl_mean
print(f"\nn=30 comparison: reglat={n30_rl_mean} randombg={n30_rb_mean} gap={n30_gap:.3f} pct={n30_pct:.2f}%")
print(f"n=50 pct shift ({pct_shift:.2f}%) vs n=30 pct shift ({n30_pct:.2f}%)")

with open(f"{SCRATCH}/phase5_n50_final_merged.pkl", "wb") as f:
    pickle.dump({"randombg": randombg_final, "reglat": reglat_final,
                 "betas_rb": betas_rb, "betas_rl": betas_rl,
                 "gap": gap, "combined_sem": combined_sem, "z": z, "pct_shift": pct_shift}, f)
print("\nMERGE_DONE")
