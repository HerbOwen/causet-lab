import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from causet_lab.mcmc import study


def main():
    t0 = time.time()
    for eps in (0.21, 0.5):
        print(f"=== blind center test N=40 eps={eps} ===", flush=True)
        result = study.run_blind_center_test(40, eps, center_mults=(0.6, 1.6))
        for cm, r in result["by_center"].items():
            print(
                f"  center_mult={cm} center_beta={r['center_beta']:.5f} -> "
                f"located beta_c={r['beta_c']:.5f} (bootstrap_std={r['bootstrap_std']:.5f}), "
                f"predicted={result['predicted']:.5f}, n_converged={r['n_converged']}/{r['n_points']}, "
                f"max_converged_beta={r['max_converged_beta']:.5f}, "
                f"beta_c_is_max_converged={r['beta_c_is_max_converged']}",
                flush=True,
            )
        print(f"  (elapsed so far {time.time()-t0:.1f}s)", flush=True)
    print(f"TOTAL {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
