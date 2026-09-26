"""Amendment 9 (post hoc): null-arm membership AUCs and two checks of the RAD-DINO pools.

nulls  Run 1 H_a vs H_b and Run 2 E100 vs H_b at the final snapshot (both arms unseen in that run): backbone
       invariance minus the step-0 encoder's (from --inv, extracted on the laptop from feats_eval_*.npz by
       deploy/extract_inv_a9.py) and the Amendment 6 head score (results/amendment6/proxy_head_run*.npz).
pools  RAD-DINO members vs non-members: out-of-fold AUC of a logistic classifier on intensity statistics and on the
       14 CheXpert labels (patient-grouped 5-fold cross-validation).
Both modes apply the reading rules written into PREREG.md before the checks were computed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from bridge_raddino import N_BOOT, boot_auc  # noqa: E402

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
LABELS14 = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
            "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices"]
NULL_PAIRS = ((1, "H_a", "H_b"), (2, "E100", "H_b"))  # (run, arm, comparator), both unseen in that run
SMALLEST_EXPOSED_AUDITOR_AUC, GATE = 0.52, 0.55


def intensity_stats(img: np.ndarray) -> np.ndarray:
    """Mean, SD and seven percentiles of the image scaled to [0, 1]."""
    x = np.asarray(img, dtype=np.float64)
    x = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    return np.r_[x.mean(), x.std(), np.percentile(x, PERCENTILES)]


def oof_auc(X, y, patients, n_splits: int = 5, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, list[float]]:
    """Out-of-fold AUC of a standardized logistic classifier, patient-grouped folds, patient-bootstrap CI."""
    X, y, patients = np.asarray(X, float), np.asarray(y, int), np.asarray(patients)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, patients):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        oof[te] = model.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return boot_auc(y, oof, patients, n=n_boot, seed=seed)


def null_auc(scores: pd.Series, groups: pd.DataFrame, pos_arm: str, neg_arm: str, n_boot: int = N_BOOT,
             seed: int = 0) -> dict:
    """Membership AUC of `pos_arm` against `neg_arm` for per-row scores (index = groups.csv row)."""
    g = groups.loc[scores.index]
    keep = g.group.isin([pos_arm, neg_arm]).to_numpy()
    y = (g.group.to_numpy()[keep] == pos_arm).astype(int)
    auc, ci = boot_auc(y, scores.to_numpy()[keep], g.patient.to_numpy()[keep], n=n_boot, seed=seed)
    return {"pos": pos_arm, "neg": neg_arm, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "auc": auc,
            "ci95": ci}


def nulls(groups: pd.DataFrame, inv: dict, heads: dict[int, pd.Series], n_boot: int = N_BOOT) -> dict:
    out = {"invariance_step0_reference": [], "head_score": []}
    for run, arm, comp in NULL_PAIRS:
        cur = pd.Series(inv[f"r{run}_fin_inv"], index=inv[f"r{run}_fin_rows"])
        ref = pd.Series(inv[f"r{run}_s0_inv"], index=inv[f"r{run}_s0_rows"])
        cal = (cur - ref).dropna()
        out["invariance_step0_reference"].append({"run": run, **null_auc(cal, groups, arm, comp, n_boot)})
        out["head_score"].append({"run": run, **null_auc(heads[run], groups, arm, comp, n_boot)})
    worst = max(r["auc"] for r in out["invariance_step0_reference"])
    out["verdict"] = ("auditor signal not distinguishable from differences between random arms"
                      if worst >= SMALLEST_EXPOSED_AUDITOR_AUC else "auditor signal reported as a weak exposure signal")
    return out


def _key(path: str) -> str:
    return "/".join(Path(path).parts[-3:])  # patientNNNNN/studyN/viewN_frontal.jpg


def pools(per_image: pd.DataFrame, labels: pd.DataFrame, root: Path, n_boot: int = N_BOOT) -> dict:
    lab = labels.assign(key=labels.Path.map(_key)).set_index("key")
    keys = per_image.path.map(_key)
    missing = sorted(set(keys) - set(lab.index))
    if missing:
        raise ValueError(f"{len(missing)} bridge images have no labels, e.g. {missing[:2]}")
    stats = np.stack([intensity_stats(np.asarray(Image.open(root / p).convert("L"))) for p in per_image.path])
    y, pat = per_image.member.to_numpy(), per_image.patient.to_numpy()
    res = {}
    for name, X in (("intensity_statistics", stats), ("labels14", lab.loc[keys, LABELS14].to_numpy(float))):
        auc, ci = oof_auc(X, y, pat, n_boot=n_boot)
        res[name] = {"auc": auc, "ci95": ci}
    worst = max(v["auc"] for v in res.values())
    res["verdict"] = ("a pool difference is a possible explanation of the RAD-DINO head result" if worst >= GATE
                      else "pools not separable by intensity statistics or labels")
    return res


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("nulls", "pools"))
    ap.add_argument("--out", type=Path, default=root / "results" / "amendment9")
    ap.add_argument("--inv", type=Path, default=root / "results" / "amendment9" / "inv_scores.npz")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "nulls":
        groups = pd.read_csv(root / "data" / "groups.csv")
        heads = {}
        for f in sorted((root / "results" / "amendment6").glob("proxy_head_run*.npz")):
            d = np.load(f)
            heads[int(d["run"])] = pd.Series(d["score"], index=d["rows"])
        res = nulls(groups, dict(np.load(args.inv)), heads)
    else:
        per_image = pd.read_csv(root / "results" / "bridge" / "bridge_per_image.csv")
        lab_dir = root / "data" / "chexlocalize" / "CheXpert"
        labels = pd.concat([pd.read_csv(lab_dir / "val_labels.csv"), pd.read_csv(lab_dir / "test_labels.csv")])
        res = pools(per_image, labels, root)
    (args.out / f"{args.mode}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
