"""Amendment 7 summary: the single-network head score's membership AUC on the 720 bridge images under RAD-DINO and
under each proxy run (a negative control that saw neither pool), paired differences, and the registered reading rule.
Inputs: results/amendment6/raddino_head_per_image.csv, results/amendment7/control_scores.csv, size_check.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MDE = 0.567  # the bridge's minimal detectable AUC, as written in Amendment 7
N_BOOT = 10_000


def paired_auc_difference(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, pat: np.ndarray, n: int = N_BOOT,
                          seed: int = 0) -> tuple[float, list[float]]:
    """AUC(s_a) - AUC(s_b) on the same images, with a patient-clustered bootstrap 95% CI (same resample for both)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(pat)
    idx_of = {p: np.flatnonzero(pat == p) for p in uniq}
    point = float(roc_auc_score(y, s_a) - roc_auc_score(y, s_b))
    draws = []
    for _ in range(n):
        pick = np.concatenate([idx_of[p] for p in rng.choice(uniq, len(uniq))])
        if 0 < y[pick].sum() < len(pick):
            draws.append(roc_auc_score(y[pick], s_a[pick]) - roc_auc_score(y[pick], s_b[pick]))
    return point, [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def verdict(proxies: list[dict], modal_auc: float) -> str:
    """(a) every proxy AUC CI includes 0.5; (b) every paired-difference CI lies above 0; (c) modal-size AUC > MDE."""
    a = all(p["auc_ci95"][0] <= 0.5 <= p["auc_ci95"][1] for p in proxies)
    b = all(p["diff_ci95"][0] > 0 for p in proxies)
    if a and b:
        return ("memorisation: the proxies do not separate the pools and RAD-DINO does, beyond them"
                if modal_auc > MDE else "size confound unresolved: (a) and (b) hold but the modal-size AUC does not")
    return "pool difference that the gate missed; the proxy AUCs give its size"


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from bridge_raddino import boot_auc

    rad = pd.read_csv(ROOT / "results" / "amendment6" / "raddino_head_per_image.csv")
    ctl = pd.read_csv(ROOT / "results" / "amendment7" / "control_scores.csv")
    if not (rad.path.to_numpy() == ctl.path.to_numpy()).all():
        raise SystemExit("control rows do not match the RAD-DINO rows")
    size = json.loads((ROOT / "results" / "amendment7" / "size_check.json").read_text())
    y, pat, s_rad = rad.member.to_numpy(), rad.patient.to_numpy(), rad.head_score.to_numpy()
    out = {"raddino": dict(zip(("auc", "auc_ci95"), boot_auc(y, s_rad, pat))), "proxies": {}}
    for col in [c for c in ctl.columns if c.startswith("head_")]:
        auc, ci = boot_auc(y, ctl[col].to_numpy(), pat)
        diff, dci = paired_auc_difference(y, s_rad, ctl[col].to_numpy(), pat)
        out["proxies"][col[len("head_"):]] = {"auc": auc, "auc_ci95": ci, "diff": diff, "diff_ci95": dci}
    out["modal_size_auc"] = size["head_auc_modal_size"]
    out["verdict"] = verdict(list(out["proxies"].values()), size["head_auc_modal_size"])
    (ROOT / "results" / "amendment7" / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
