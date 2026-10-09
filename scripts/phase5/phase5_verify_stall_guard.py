"""Verify the fixed stall shortcut does NOT fire while genuinely
confined, using an extremely aggressive threshold (1s) that would fire
almost instantly under the OLD (buggy) logic. Resumes from the exact
bg0 n=50 checkpoint that was previously (wrongly) shortcut."""
import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from causet_lab.mcmc import random_bg as rbgm

CKPT_DIR = str(Path(__file__).resolve().parents[2] / "data" / "checkpoints")
SRC = f"{CKPT_DIR}/randombg_wl_n50_bg0.pkl"
TEST_CKPT = f"{CKPT_DIR}/TEST_guard_randombg_wl_n50_bg0.pkl"
shutil.copyfile(SRC, TEST_CKPT)

wl = rbgm.run_wang_landau_randombg(
    50, 0.1, None, None, seed=10000, checkpoint_path=TEST_CKPT,
    verbose=True, verbose_label="GUARD-TEST bg0 n=50",
    stall_shortcut_seconds=1.0, stall_shortcut_f=1e-2,
    checkpoint_every_s=15,
)
print(f"\nused_stall_shortcut={wl['used_stall_shortcut']}  stage={wl['stages']}  elapsed={wl['elapsed']:.1f}s")
print("GUARD_TEST_DONE")
