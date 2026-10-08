# Phase 4: dimensionally restricted (lattice-gas) causal sets, d=2

## Summary

Goal: implement Cunningham & Surya's (C&S) lattice-gas causal set model, arXiv:1908.11647, and check it reproduces their qualitative d=2 result -- a manifold-like phase at low beta and a layered phase at high beta -- as a known-answer check before considering higher dimensions. Laptop scale (n=30, 50) instead of their n=200.

**Result: pass.** The beta=0 control reproduces 2D-sprinkle-like local structure; the incremental engine matches full recomputation bit-for-bit over 1e5 moves; a Wang-Landau/multicanonical (WL/MUCA) run at n=30 locates a clear beta_c via the action-variance peak, and the low-beta/high-beta phases either side of it differ exactly the way C&S report (manifold-like vs layered), with the hot-phase ordering fraction landing almost exactly on their quoted value. One real engineering constraint was found: **n=80 is not reachable without extending the bitset engine** (see Known limitations #1) -- only n<=64 is supported as built.

## Part 1: construction (quoted from C&S where possible)

**Background lattice** (C&S Sec. 2, Eq. 3 and surrounding text): spacetime M_2 ~ R x S^1, metric `ds^2 = -dt^2 + dtheta^2`, `theta ~ theta + 2*pi`, "local light cones ... at 45 degrees". The lattice `L_2^(m)` has `h*w = m` sites `(t, theta)`, `t in {0,...,h}`, `theta = 2*pi*k/w`, `k in {0,...,w-1}`. This project uses C&S's own d=2 sizing convention: aspect ratio `alpha = h/w = 4`, `w = n`, so `m = alpha*n^2`, confirmed against their own number ("we use ... alpha=4 ... m ~ 1.6x10^5" at their n=200 -> `4*200^2 = 160000`, matches).

**Filling** (C&S Sec. 2): an n-site filling is represented as "a random permutation of L" where `L = {0,...,m-1}`; occupied sites are `L[0:n]`, and the move set is "a swap between a randomly chosen element in the first n entries ... with a randomly ... chosen element from the last m-n entries." Implemented exactly this way (`lattice_gas.py`/`lattice_gas_core.py`): `L` is an `(m,)` int64 array, a site id `s` decodes to `(t, k) = (s // w, s % w)` (quoted: `t_i = floor(L_i/w)`, `theta_i = (2*pi/w)(L_i mod w)`).

**Causal relation -- flagged, not a direct quote.** C&S state the causal structure follows from the metric with 45-degree lightcones and confirm (their Fig. 2(ii)) that relations wrap "around the cylinder" via the `S^1` topology, but give no explicit integer-coordinate formula. This project's own derivation:

```
(t_a, k_a) precedes (t_b, k_b)  iff  t_b > t_a
                                 and  min(dk, w - dk) <= t_b - t_a
                                 where dk = (k_b - k_a) mod w
```

`min(dk, w-dk)` is the shortest-arc distance on a cycle of circumference `w`, which already handles wraparound in either direction without extra winding-number logic (going further around a cycle is never shorter). Lightcone-boundary pairs (equality) are treated as related -- also this project's own choice, since C&S do not address it.

**Smeared action, Eq. 8/9 (quoted exactly)**:

```
S_BD^(2)(c)/hbar = 2*eps*[n - 2*eps * sum_{r=1}^{n-1} n_r * f_2(r-1, eps)]      (Eq. 8)

f_2(r, eps) = (1-eps)^r * [1 - 2*r*eps/(1-eps) + r*(r-1)*eps^2/(2*(1-eps)^2)]   (Eq. 9)
```

`f_2` is confirmed identical to this codebase's existing `action.f2_smear_table` kernel (reused directly). **The outer prefactor and summation bound genuinely differ from the Phase 3/3b order-based action**, confirmed by direct quotation from both papers, not a transcription error: Glaser-O'Connor-Surya's order-based action uses a `4*eps` prefactor and sums from `r=0` (including links/`n_0`); C&S's Eq. 8 uses `2*eps` and sums from `r=1` (excluding links entirely). Implemented as a separate function (`lattice_gas_core.lattice_gas_action_from_counts`), not by modifying the existing order-based action.

**Parameters C&S use for d=2**: `eps = 0.1` (quoted: "we fix eps=0.1"); their own located `beta_c^(2) ~= 2.344` at `n=200` ("For our simulations, beta_c(2)~=2.344"), with **no n-scaling formula** given (unlike Glaser-O'Connor-Surya 2018's `beta_c(N,eps)` for the order model). This project's `beta_c_guess(n) = 2.344*200/n` is an explicitly-flagged, untrusted 1/n heuristic, used only to size the WL pilot range -- **not treated as a prediction anywhere a result is reported** (see Part 4).

Sweep = `n*(n-1)/2` individual moves (quoted).

## Part 2: reused infrastructure

Bitset relations/incremental updates, the smeared-action kernel (`f2_smear_table`), checkpointing, and the MUCA orchestration (`reweight_P_beta_corrected`, `locate_beta_c_variance_peak`, stage-doubling WL, stall detection) are all reused from Phase 3b unchanged. `muca.py` gained a minimal, backward-compatible `production_fn` parameter (defaults to the existing order-model function) so `run_muca_recursion`/`run_muca_production_parallel` can be reused for the lattice-gas model without duplicating ~150 lines; the full test suite (51 tests) was re-run before and after this change with no regression.

New lattice-gas-specific code: `mcmc/lattice_gas_core.py` (bitset engine, action, single-element "relocate" move, sweep kernels) and `mcmc/lattice_gas.py` (sizing, random fillings, pilot chains, WL/MUCA orchestration) -- written as clones of the existing orders-model engine/orchestration rather than generalizing it in place, since the existing code is already tested and committed (Phase 3b) and the two models' moves differ (relocate changes one element; the orders model's swap changes two).

## Part 3: controls

**Incremental vs. full recomputation** (`test_lattice_gas_incremental_action_matches_full_recompute`, n in {16,30,47}, eps in {0.1,0.21}, 1e5 moves each, checked every 500 steps): the incrementally-updated action **and** the full `counts`/`n_r` histogram (not just the scalar action -- a compensating pair of errors could hide behind a correct-looking `S`) match the independent full recomputation bit-for-bit at every checked step, across all 6 (n, eps) combinations. **Pass, no bugs found.**

**Move self-inverse** (`test_lattice_gas_rejected_move_is_exactly_undone`): reapplying `apply_relocate` with the same `(i,j)` restores `(L, future, past, counts)` exactly, over 50 random moves. **Pass.**

**beta=0 (uniformly random filling) looks like a 2D sprinkle.** The whole lattice-gas filling is a cylinder of aspect ratio `h/w=4`, not an Alexandrov interval, so its *whole-matrix* ordering fraction/dimension are not directly comparable to a sprinkle's (confirmed: C&S's own hot-phase ordering fraction is ~0.88, not the causal-diamond value ~0.5). Fixed by reusing this project's existing solution for exactly this problem -- `battery.analyze_matrix`'s sub-interval sampling (already used for non-diamond-shaped grown/junk orders), since an interval is diamond-shaped by construction regardless of the embedding's global topology.

| n | MM dimension (sampled intervals) | Abundance-profile L1 distance to a sprinkle |
|---|---|---|
| 30 | **2.292** | **0.100** |
| 50 | **2.073** | **0.086** |

Both comfortably within target (dim ~2, distance well under the 0.5 threshold). **Pass.**

## Part 4: n=30 pilot

**Runtime**: Wang-Landau 1613.7s + MUCA production 948.2s = **2561.8s (~42.7 min)** end-to-end, single seed, on this laptop.

- WL: 20 stages to `f <= F_FINAL=1e-6`, 96/120 bins reachable (stall-detected and accepted after 25s with no new bin at stage 0 -- genuinely unreachable action values, not a slow-to-reach region).
- MUCA: 134.5M moves to reach the target 10 round trips; `hidden_barrier_warning=False`; `height_round_trips=716` (height round-trips far exceed the 10 S-round-trips, so height is not hiding a barrier S isn't seeing).
- **beta_c (action-variance peak) = 11.118**, interior to the scanned bracket. The 1/n heuristic guess (15.627) overshoots by ~40% -- expected, since it is a bare 1/n extrapolation of C&S's single n=200 point, and was never trusted as more than a bracket-sizing aid (per instruction, treated as a sanity check only, not a validation).

### Low beta vs. high beta

Using `muca.reweight_observable` against the frozen WL weight + MUCA histogram for height/ordering-fraction (reweighted across the full explored range), and fresh equilibrated samples at `beta = 4*beta_c = 44.47` (10 independent chains, annealed from beta=0 over 400 sweeps) for MM dimension and layer count, compared against 10 fresh beta=0 (uniformly random) samples:

| Observable | Low beta (beta=0, hot) | High beta (beta=4*beta_c, cold) | C&S d=2 (quoted/derived) |
|---|---|---|---|
| MM dimension (sampled intervals) | **2.18** | **2.87** | manifold-like (~2) vs. non-manifold |
| Ordering fraction (whole filling) | **0.882** | **0.646** | **0.88** (hot) vs. **~0.5** (cold, asymptotic bilayer limit) |
| Height (longest chain) | **7.40** | **4.20** | lower height = more layered (qualitative) |
| Occupied lattice rows ("layers") | **26.7**/120 | **12.8**/120 | ~5 layers at their n=200 |
| Abundance-profile L1 distance to a sprinkle | **0.065** | **0.794** | -- |

**Qualitative match, in the right direction on every observable**: low beta is manifold-like (dimension near 2, abundance profile close to a sprinkle, ordering fraction landing almost exactly on C&S's own quoted hot-phase value of 0.88); high beta collapses toward fewer occupied rows, lower height, lower dimension-fidelity, and an abundance profile far from any sprinkle -- the same direction as C&S's layered phase. Two caveats: (1) our cold-phase ordering fraction (0.646) does not reach their asymptotic bilayer value (~0.5) -- expected, since n=30 is well below their n=200 and we sampled at `4*beta_c` rather than the deep frozen limit; (2) our layer collapse (26.7 -> 12.8 out of 120 rows) is proportionally much milder than their ~5-layer collapse at n=200/h=800 -- plausibly a genuine finite-size effect (consistent with C&S's own reported poor thermalisation near beta_c at small n), not investigated further at this scale.

## Part 5: cross-check against the 2D orders model

C&S do **not** quantitatively compare the lattice-gas model's beta_c (or phase diagram) to the earlier "generic" causal-set orders MCMC model (Glaser, O'Connor & Surya 2018) anywhere in the paper -- the only related remark is a qualitative aside that the sample space of n-element 2-orders "is dominated by causal sets approximated by the flat causal diamond," with no quantitative beta_c comparison. Since the paper itself doesn't provide this check, this project ran its own, as an additional (not paper-reproducing) cross-check between its two validated models, at matching N=30 and **eps=0.1** (C&S's own d=2 choice, not Phase 3b's earlier eps=0.5 calibration):

| | Runtime | beta_c (variance peak) | Formula/heuristic prediction |
|---|---|---|---|
| Orders model (Phase 3b engine), N=30, eps=0.1 | WL 735.5s + MUCA 150.1s = 885.6s | **6.939** | Glaser-O'Connor-Surya (2018) formula: 6.992 (z~0.05, excellent agreement) |
| Lattice-gas model, n=30, eps=0.1 | 2561.8s (above) | **11.118** | C&S 1/n heuristic: 15.627 (not trusted, see above) |

**Caveat, important**: the two models' action normalizations genuinely differ (Part 1 -- `2*eps` vs `4*eps` prefactor, and the summation's lower bound), so their beta values are not on a directly-comparable scale by construction; a numeric ratio between the two beta_c values is not expected to equal 1 and isn't a pass/fail criterion. What this cross-check does confirm: (1) the orders-model engine, run fresh at a new eps it had not previously been calibrated at, reproduces the published Glaser-O'Connor-Surya formula almost exactly (6.939 vs. 6.992) -- a strong independent validation of that engine and of the variance-peak beta_c-location method shared by both models; (2) both located beta_c values are the same order of magnitude (~7 vs. ~11) at the same N and eps, which is at least consistent with (not a proof of) the two actions being comparably-scaled smeared BD actions on similarly-sized configuration spaces, rather than differing by some large, unexplained factor.

## Known limitations

1. **n=80 ("if feasible") is not reachable without extending the engine.** `lattice_gas_core.build_lattice_state_bitset` asserts `n <= 64` -- the engine represents each element's relations as a single `uint64` bitset (inherited from `bitset_core.py`'s Phase 3b design). Reaching n=80 would need a multi-word bitset (e.g. two uint64 words per element) or an equivalent redesign, not attempted here. **Not feasible as built; only n=30 and n=50 are in scope.**
2. **Single seed at n=30.** No seed-to-seed beta_c spread was characterized (mirrors Phase 3b's own single-seed limitation at this stage).
3. **Cold-phase sampling used fixed-beta pilot chains at `4*beta_c`, not the full MUCA reweighting**, for the MM-dimension/layer-count numbers specifically (height/ordering-fraction *were* obtained by full reweighting across the whole explored range). This is because MM dimension requires sampling sub-intervals from the actual relation matrix per configuration, which was not recorded at every MUCA structural-measurement step (only height/ordering-fraction were, to keep that per-step cost cheap); a single beta point well into the cold phase was judged sufficient for this qualitative check rather than re-running MUCA with a more expensive per-step measurement.
4. **The two flagged ambiguities (causal relation formula, action normalization) are this project's own resolutions of real gaps/differences in the source material, not independently verified against another implementation of C&S's model.**

## Pass criteria, assessed against what was actually run

1. **beta=0 control looks like a 2D sprinkle**: met (n=30, n=50).
2. **Incremental == full recomputation over 1e5 moves**: met, bit-for-bit, across all tested (n, eps).
3. **Manifold-like phase at low beta, layered phase at high beta, qualitatively matching C&S's figures**: met at n=30 -- every observable in Part 4's table moves in the correct direction, and the hot-phase ordering fraction (0.882) essentially reproduces their quoted 0.88.
4. **Approximate beta_c from the variance peak**: met (11.118, interior, n=30).
5. **Report runtime per n before scaling up**: this report -- n=30 took ~42.7 min end-to-end; n=50/80 not yet run (n=80 excluded per limitation #1 above).

**Overall: pass**, for the known-answer check this phase set out to do, at n=30. n=50 (within the n<=64 engine limit) is the natural next step if more compute is warranted; n=80 would require engine changes first.
