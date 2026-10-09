import time, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from causet_lab.mcmc import muca

SCRATCH = str(Path(__file__).resolve().parents[2] / "data")

def main():
    t0 = time.time()
    r = muca.run_muca_one(40, 0.5, seed=0, target_round_trips=10,
                           checkpoint_dir=f"{SCRATCH}/checkpoints", verbose=True)
    elapsed = time.time() - t0
    print(f"[pilot40] TOTAL elapsed={elapsed:.1f}s t_wl={r['t_wl']:.1f}s t_muca={r['t_muca']:.1f}s", flush=True)
    print(f"[pilot40] beta_c_var_res={r['beta_c_var_res']['beta_c']:.5f} "
          f"interior={r['beta_c_var_res']['is_interior']}", flush=True)
    print(f"[pilot40] beta_c_equal_height={r['beta_c_res']}", flush=True)
    print(f"[pilot40] predicted={r['predicted']:.5f}", flush=True)
    print(f"[pilot40] round_trips={r['prod']['round_trips']} hidden_barrier={r['prod']['hidden_barrier_warning']}", flush=True)
    with open(f"{SCRATCH}/pilot_n40_eps0p5_result.pkl", "wb") as f:
        pickle.dump(r, f)

if __name__ == "__main__":
    main()
