"""Diagnose the n=50 random-background (bg0) WL stall before trusting
any stall-shortcut mechanism. Questions to answer:
1. Is "9/87 visited" the current-stage H (reset each stage) or
   ever_visited? (Answer from code: current-stage H; ever_visited=87.)
2. Which action region is bin_idx=114 in (hot/cold/middle)?
3. Is the walker's configuration literally frozen (near-zero
   acceptance), or just revisiting a small set of bins?
4. Is ln_g still being updated/rising in that region (WL should push
   the walk out if it's tracking properly)?
5. What's the empirical move-acceptance rate from this exact state,
   measured directly (not inferred)?
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from causet_lab.mcmc import random_bg as rbgm
from causet_lab.mcmc import random_bg_core as rbg
from causet_lab.mcmc.checkpoint import load_checkpoint
from causet_lab.mcmc.action import f2_smear_table
from causet_lab.mcmc import rng as rngm

CKPT = str(Path(__file__).resolve().parents[2] / "data" / "checkpoints" / "randombg_wl_n50_bg0.pkl")

ck = load_checkpoint(CKPT)
n, eps = 50, 0.1
w, h, m = rbgm.background_dims(n)
site_t, site_x = ck["site_t"], ck["site_x"]
L = ck["L"]
future, past, counts = ck["future"], ck["past"], ck["counts"]
ln_g, H = ck["ln_g"], ck["H"]
bin_lo, bin_width, n_bins = ck["bin_lo"], ck["bin_width"], ck["n_bins"]
f_mod, stage = ck["f_mod"], ck["stage"]
bin_idx = ck["bin_idx"]
ever_visited = ck["ever_visited"]
rng_state = np.uint64(ck["rng_state"])

f2_table = f2_smear_table(max(n - 2, 0), eps)

print(f"=== Checkpoint state: bg0, n=50 ===")
print(f"stage={stage} f_mod={f_mod:.3e} n_bins={n_bins} bin_lo={bin_lo:.3f} bin_width={bin_width:.4f}")
print(f"bin_idx={bin_idx}  ever_visited count={int(ever_visited.sum())}/{n_bins}")
print(f"ever_visited range: [{np.flatnonzero(ever_visited)[0]}, {np.flatnonzero(ever_visited)[-1]}]")

S_current = rbg.lattice_gas_action_from_counts(n, counts, f2_table, eps)
bin_from_S = rbg.bin_index(S_current, bin_lo, bin_width, n_bins)
S_at_bin_lo = bin_lo + bin_idx * bin_width
S_at_bin_hi = bin_lo + (bin_idx + 1) * bin_width
print(f"\nActual recomputed action S = {S_current:.4f} -> bin {bin_from_S} (checkpoint says bin_idx={bin_idx})")
print(f"Bin {bin_idx} covers S in [{S_at_bin_lo:.4f}, {S_at_bin_hi:.4f}]")
print(f"Full explored range: bin_lo={bin_lo:.3f} to bin_hi={bin_lo + n_bins*bin_width:.3f}")
print(f"ever_visited extremes in S: [{bin_lo + np.flatnonzero(ever_visited)[0]*bin_width:.3f}, "
      f"{bin_lo + (np.flatnonzero(ever_visited)[-1]+1)*bin_width:.3f}]")

# Where does S=0 fall (rough hot/cold midpoint reference)? And compare
# to known hot-phase mean S and cold floor from n=30 (same model class,
# scaled): just report bin_idx's position as a fraction of the explored span.
frac = (bin_idx - np.flatnonzero(ever_visited)[0]) / max(1, (np.flatnonzero(ever_visited)[-1] - np.flatnonzero(ever_visited)[0]))
print(f"bin_idx={bin_idx} sits at {frac:.1%} of the way from the coldest to hottest EVER-VISITED bin "
      f"(0%=coldest/most-ordered extreme, 100%=hottest/most-disordered extreme)")

print(f"\n=== ln_g near bin_idx (window +/- 5 bins) ===")
for b in range(max(0, bin_idx - 5), min(n_bins, bin_idx + 6)):
    marker = " <== current" if b == bin_idx else ("  (ever-visited)" if ever_visited[b] else "  (never visited)")
    print(f"  bin {b:3d}: ln_g={ln_g[b]:14.4f}  H(this stage)={H[b]:6d}{marker}")

print(f"\n=== ln_g over the full ever-visited span (every 5th bin) ===")
lo_ev, hi_ev = int(np.flatnonzero(ever_visited)[0]), int(np.flatnonzero(ever_visited)[-1])
for b in range(lo_ev, hi_ev + 1, 5):
    print(f"  bin {b:3d}: ln_g={ln_g[b]:14.4f}  ever_visited={ever_visited[b]}")

# ---- Empirical acceptance rate from this exact state, measured directly ----
# Run several independent short bursts, re-loading the same starting
# state each time (not letting one burst's drift contaminate the next),
# and count actual accept/reject at the move level by instrumenting a
# plain-Python version of the WL move step (slower, but transparent).

def wl_accept_prob(ln_g_old, ln_g_new):
    d = ln_g_old - ln_g_new
    if d >= 0:
        return 1.0
    return float(np.exp(d))

print(f"\n=== Direct candidate-move scan from the frozen state ===")
rng_np = np.random.default_rng(12345)
n_trials = 2000
dS_list = []
new_bin_list = []
accept_prob_list = []
L_work = L.copy()
future_work, past_work, counts_work = future.copy(), past.copy(), counts.copy()
for trial in range(n_trials):
    i = int(rng_np.integers(0, n))
    j = int(rng_np.integers(n, m))
    rbg.apply_relocate_randombg_jit(L_work, future_work, past_work, counts_work, n, site_t, site_x, w, i, j)
    S_new = rbg.lattice_gas_action_from_counts(n, counts_work, f2_table, eps)
    bin_new = rbg.bin_index(S_new, bin_lo, bin_width, n_bins)
    dS = S_new - S_current
    ln_g_new_val = ln_g[bin_new] if 0 <= bin_new < n_bins else np.inf
    ln_g_old_val = ln_g[bin_idx]
    p_acc = wl_accept_prob(ln_g_old_val, ln_g_new_val) if 0 <= bin_new < n_bins else 0.0
    dS_list.append(dS)
    new_bin_list.append(bin_new)
    accept_prob_list.append(p_acc)
    # revert
    rbg.apply_relocate_randombg_jit(L_work, future_work, past_work, counts_work, n, site_t, site_x, w, i, j)

dS_arr = np.array(dS_list)
bin_arr = np.array(new_bin_list)
pacc_arr = np.array(accept_prob_list)
print(f"{n_trials} candidate single-site relocations from the frozen configuration:")
print(f"  dS: min={dS_arr.min():.3f} max={dS_arr.max():.3f} mean={dS_arr.mean():.3f} std={dS_arr.std():.3f}")
print(f"  resulting bin spread: {bin_arr.min()} to {bin_arr.max()} (current bin={bin_idx})")
print(f"  fraction landing in-bounds (0<=bin<{n_bins}): {np.mean((bin_arr>=0)&(bin_arr<n_bins)):.3f}")
print(f"  mean WL accept prob (ln_g[old]-ln_g[new] rule): {pacc_arr.mean():.6f}")
print(f"  max WL accept prob across all {n_trials} candidates: {pacc_arr.max():.6f}")
print(f"  fraction with accept prob > 0.01: {np.mean(pacc_arr > 0.01):.4f}")
print(f"  fraction with accept prob > 0.5: {np.mean(pacc_arr > 0.5):.4f}")

# Histogram of which bins candidate moves would land in, vs ever_visited
unique_bins, counts_per_bin = np.unique(bin_arr[(bin_arr >= 0) & (bin_arr < n_bins)], return_counts=True)
print(f"\n  Candidate moves land in {len(unique_bins)} distinct bins; of these, "
      f"{int(ever_visited[unique_bins].sum())} are in ever_visited, "
      f"{len(unique_bins) - int(ever_visited[unique_bins].sum())} are NEW (never visited before).")
print(f"  Top 10 most-landed-in bins (bin: count, ln_g, ever_visited):")
order = np.argsort(-counts_per_bin)[:10]
for idx in order:
    b = int(unique_bins[idx])
    print(f"    bin {b:3d}: {counts_per_bin[idx]:4d} candidates, ln_g={ln_g[b]:.3f}, ever_visited={ever_visited[b]}")

# ---- Empirical move acceptance: actually run the real WL kernel for a
# short burst starting from this exact state, and count real accept/reject
# by comparing bin_idx before/after many small chunks.
print(f"\n=== Running the real WL kernel for 200 chunks (2,000,000 moves) from this exact state ===")
L2, future2, past2, counts2 = L.copy(), future.copy(), past.copy(), counts.copy()
ln_g2, H2 = ln_g.copy(), H.copy()
extreme_state = np.zeros(1, dtype=np.int64)
half_trips = np.zeros(1, dtype=np.int64)
bin_idx2 = bin_idx
rng_state2 = rng_state
bins_seen = set()
for chunk in range(200):
    rng_state2, bin_idx2, edge_hits = rbg.wl_or_muca_sweep_chunk_randombg_jit(
        L2, future2, past2, counts2, n, m, site_t, site_x, w, eps, f2_table, ln_g2, H2,
        bin_lo, bin_width, n_bins, f_mod, True, 10000,
        rng_state2, bin_idx2, extreme_state, half_trips, 0, lo_ev, hi_ev,
    )
    bins_seen.add(int(bin_idx2))
print(f"After 2,000,000 further moves with the REAL kernel (same f_mod, starting ln_g): "
      f"distinct bins visited = {len(bins_seen)}, final bin_idx={bin_idx2}")
print(f"bins seen: {sorted(bins_seen)}")
print(f"ln_g[current bin] grew from {ln_g[bin_idx]:.3f} to {ln_g2[bin_idx2]:.3f}")

print("\nDIAGNOSIS_DONE")
