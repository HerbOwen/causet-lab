import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from causet_lab.mcmc.action import beta_c_glaser_2018

LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[2] / "data" / "mcmc_full_run.log"
)

line_re = re.compile(
    r"N=\s*(?P<N>\d+)\s+eps=(?P<eps>[\d.]+)\s+beta=\s*(?P<beta>[-\d.]+)\s+\[(?P<tag>[^\]]+)\]"
)

points = {}  # (N, eps) -> list of (beta, converged_bool)
for line in LOG.read_text().splitlines():
    m = line_re.search(line)
    if not m:
        continue
    N = int(m.group("N"))
    eps = float(m.group("eps"))
    beta = float(m.group("beta"))
    converged = "UNCONVERGED" not in line
    points.setdefault((eps, N), []).append((beta, converged))

ns = [30, 40, 50, 60]
epss = [0.21, 0.5]

reported_beta_c = {
    (0.21, 30): 0.95652, (0.21, 40): 0.70177, (0.21, 50): 0.59210, (0.21, 60): 0.50254,
    (0.5, 30): 0.14797, (0.5, 40): 0.10226, (0.5, 50): 0.08865, (0.5, 60): 0.08667,
}

print(f"{'eps':>5} {'N':>4} {'predicted':>10} {'beta_c':>9} {'n_conv':>7} {'below(-25%)':>12} {'above(+25%)':>12} {'max_conv_beta':>14} {'IS_MAX_CONV':>12}")
for eps in epss:
    for N in ns:
        pts = points.get((eps, N), [])
        predicted = beta_c_glaser_2018(N, eps)
        lo, hi = 0.75 * predicted, 1.25 * predicted
        conv = [b for b, c in pts if c]
        below = [b for b in conv if lo <= b < predicted]
        above = [b for b in conv if predicted <= b <= hi]
        bc = reported_beta_c[(eps, N)]
        max_conv = max(conv) if conv else float("nan")
        is_max = abs(bc - max_conv) < 1e-4
        print(f"{eps:>5} {N:>4} {predicted:>10.5f} {bc:>9.5f} {len(conv):>7} {len(below):>12} {len(above):>12} {max_conv:>14.5f} {str(is_max):>12}")
