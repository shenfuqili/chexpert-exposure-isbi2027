"""Unattended go/no-go check after the pilot, run on the laptop before the formal runs start.

Passes only if: the training loss is finite and fell; every evaluation snapshot has a feature file with
finite, non-constant features; invariance scores lie in (0, 1]; the step-0 probe macro-AUC on H is in a
plausible range for DINOv2-S on CheXpert; and the final-snapshot probe on H has not collapsed.
Writes gate.json next to the pilot run and GATE_OK on success; exits 1 on failure so the queue stops.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
AUC_RANGE_STEP0 = (0.60, 0.95)
MAX_AUC_DROP = 0.05  # pilot-only sanity bound; the preregistered collapse check applies to the formal runs


def probe_auc(groups: pd.DataFrame, rows: np.ndarray, feats: np.ndarray) -> float:
    g = groups.loc[rows, "group"].to_numpy()
    y = groups.loc[rows, [f"{lab}__uzero" for lab in LABELS]].to_numpy()
    tr, te = g == "P_base", np.isin(g, ("H_a", "H_b"))
    mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-6
    x = (feats - mu) / sd
    aucs = []
    for j in range(len(LABELS)):
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(x[tr], y[tr, j])
        aucs.append(roc_auc_score(y[te, j], clf.predict_proba(x[te])[:, 1]))
    return float(np.mean(aucs))


def main(run_dir: str, groups_csv: str) -> int:
    run, groups = Path(run_dir), pd.read_csv(groups_csv)
    checks, info = {}, {}
    recs = [json.loads(x) for x in (run / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    losses = [r["loss"] for r in recs]
    checks["loss_finite"] = bool(np.all(np.isfinite(losses)))
    checks["loss_fell"] = bool(len(losses) >= 4 and np.mean(losses[-3:]) < np.mean(losses[:3]))
    info["loss_first_last"] = [losses[0], losses[-1]] if losses else None
    snaps = sorted(run.glob("eval_step*.pt"))
    feats_files = [run / f"feats_{s.stem}.npz" for s in snaps]
    checks["all_features_present"] = len(snaps) >= 2 and all(f.exists() for f in feats_files)
    aucs = {}
    for f in feats_files if checks["all_features_present"] else []:
        d = np.load(f)
        x = d["feats"].astype(np.float32)
        ok = bool(np.isfinite(x).all() and x.std(0).mean() > 1e-3 and np.all((d["inv"] > 0) & (d["inv"] <= 1.0001)))
        checks[f"sane_{f.stem}"] = ok
        if ok:
            aucs[f.stem] = probe_auc(groups, d["rows"], x)
    info["probe_macro_auc_H"] = aucs
    if aucs:
        first, last = aucs[min(aucs)], aucs[max(aucs)]
        checks["step0_auc_plausible"] = AUC_RANGE_STEP0[0] <= first <= AUC_RANGE_STEP0[1]
        checks["no_collapse"] = last >= first - MAX_AUC_DROP
    passed = bool(checks) and all(checks.values())
    (run / "gate.json").write_text(json.dumps({"passed": passed, "checks": checks, "info": info}, indent=2))
    print(json.dumps({"passed": passed, "checks": checks, "info": info}, indent=2))
    if passed:
        (run / "GATE_OK").write_text("ok")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
