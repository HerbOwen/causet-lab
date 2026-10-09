"""Follow-up: is the bg0 bin-114 trap merely slow (WL eventually
equalizes ln_g and escapes) or pathological? Run many more real-kernel
moves from the exact frozen checkpoint state and watch ln_g[114] vs.
its neighbors, and whether/when bin_idx actually changes.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from causet_lab.mcmc import random_bg as rbgm
from causet_lab.mcmc import random_bg_core as rbg
from causet_lab.mcmc.checkpoint import load_checkpoint
from causet_lab.mcmc.action import f2_smear_table

CKPT = str(Path(__file__).resolve().parents[2] / "data" / "checkpoints" / "randombg_wl_n50_bg0.pkl")
ck = load_checkpoint(CKPT)
n, eps = 50, 0.1
w, h, m = rbgm.background_dims(n)
site_t, site_x = ck["site_t"], ck["site_x"]
L, future, past, counts = ck["L"].copy(), ck["future"].copy(), ck["past"].copy(), ck["counts"].copy()
ln_g, H = ck["ln_g"].copy(), ck["H"].copy()
bin_lo, bin_width, n_bins = ck["bin_lo"], ck["bin_width"], ck["n_bins"]
f_mod = ck["f_mod"]
bin_idx = ck["bin_idx"]
ever_visited = ck["ever_visited"]
rng_state = np.uint64(ck["rng_state"])
f2_table = f2_smear_table(max(n - 2, 0), eps)
lo_ev, hi_ev = int(np.flatnonzero(ever_visited)[0]), int(np.flatnonzero(ever_visited)[-1])

extreme_state = np.zeros(1, dtype=np.int64)
half_trips = np.zeros(1, dtype=np.int64)

print(f"Starting bin_idx={bin_idx}, ln_g[114]={ln_g[114]:.3f}, ln_g[113]={ln_g[113]:.3f} "
      f"(gap={ln_g[113]-ln_g[114]:.3f}, need gap<=0 roughly to escape with decent odds)")
print(f"f_mod={f_mod:.3e} -> gap closes by ~f_mod per move stuck (~{f_mod:.3e}/move)")

t0 = time.time()
CHUNK = 10000
N_CHUNKS = 6000  # 60,000,000 moves -- much further than the original 600s-equivalent window
escaped_at = None
bins_seen_total = set()
report_every = 200
for c in range(N_CHUNKS):
    rng_state, bin_idx, edge_hits = rbg.wl_or_muca_sweep_chunk_randombg_jit(
        L, future, past, counts, n, m, site_t, site_x, w, eps, f2_table, ln_g, H,
        bin_lo, bin_width, n_bins, f_mod, True, CHUNK,
        rng_state, bin_idx, extreme_state, half_trips, 0, lo_ev, hi_ev,
    )
    bins_seen_total.add(int(bin_idx))
    if bin_idx != 114 and escaped_at is None:
        escaped_at = (c + 1) * CHUNK
        print(f"ESCAPED at move {escaped_at} (elapsed {time.time()-t0:.1f}s): new bin_idx={bin_idx}")
    if c % report_every == 0:
        print(f"  chunk {c}/{N_CHUNKS} moves={(c+1)*CHUNK} bin_idx={bin_idx} "
              f"ln_g[114]={ln_g[114]:.3f} ln_g[113]={ln_g[113]:.3f} gap={ln_g[113]-ln_g[114]:.3f} "
              f"distinct_bins_seen_so_far={len(bins_seen_total)} ({time.time()-t0:.1f}s)", flush=True)
    if escaped_at is not None and c > (escaped_at // CHUNK) + 50:
        break

print(f"\nFinal: bin_idx={bin_idx}, distinct bins seen over run: {sorted(bins_seen_total)}")
print(f"Total moves run: {(c+1)*CHUNK}, wall time: {time.time()-t0:.1f}s")
print("PART2_DONE")
