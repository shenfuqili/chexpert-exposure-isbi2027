"""Amendment 9: copy only the invariance scores out of the laptop's feats_eval_*.npz (step 0 and final, both runs)."""
import numpy as np

out = {}
for run, final in ((1, 18160), (2, 18171)):
    for tag, step in (("s0", 0), ("fin", final)):
        d = np.load(rf"D:\isbi\runs\run{run}\feats_eval_step{step:07d}.npz")
        out[f"r{run}_{tag}_rows"], out[f"r{run}_{tag}_inv"] = d["inv_rows"], d["inv"]
np.savez(r"D:\isbi\results\amendment9\inv_scores.npz", **out)
print({k: v.shape for k, v in out.items()})
