"""Amendment 10 (post hoc, exploratory): the crossover premium with probes trained on images exposed as often as the
test images, as in a benchmark whose training and test splits were both in pretraining.

Within H_a and E100: two-fold patient-grouped cross-fitting (a probe trained on one half predicts the other), repeated
over 5 fixed splits shared by both runs; A_X,r is the macro-AUC of each image's mean prediction, and
Delta_m = 1/2 [(A_Ha,2 - A_Ha,1) + (A_E100,1 - A_E100,2)]. Patient-bootstrap CIs are conditional on the fitted probes.
Laptop: python amendment10_matched.py --run1 D:\\isbi\\runs\\run1 --run2 D:\\isbi\\runs\\run2
        --groups D:\\isbi\\data\\groups.csv --out D:\\isbi\\results\\amendment10 --jobs 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
import analyze  # noqa: E402

ARMS = ("H_a", "E100")
SPLIT_SEED, N_REPEATS = 20260926, 5


def half_splits(patients: np.ndarray, n_repeats: int = N_REPEATS, seed: int = SPLIT_SEED) -> list[np.ndarray]:
    """Boolean masks, one per repeat, each selecting a random half of the patients (all their images)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(patients)
    return [np.isin(patients, rng.permutation(uniq)[: len(uniq) // 2]) for _ in range(n_repeats)]


def _fit_predict(fit, feats, y, patients, train):
    predict, _ = fit(feats[train], y[train], patients[train])
    return predict(feats[~train])


def crossfit_predictions(feats: np.ndarray, y: np.ndarray, patients: np.ndarray, masks: list[np.ndarray],
                         fit=analyze.fit_probe, jobs: int = 1) -> np.ndarray:
    """Each image's prediction, averaged over repeats, always from a probe trained on the other half."""
    halves = [(k, train) for k, m in enumerate(masks) for train in (m, ~m)]
    outs = Parallel(n_jobs=jobs)(delayed(_fit_predict)(fit, feats, y, patients, train) for _, train in halves)
    preds = np.zeros((len(masks), *y.shape))
    for (k, train), p in zip(halves, outs):
        preds[k, ~train] = p
    return preds.mean(0)


def matched_premium(auc: dict) -> float:
    return 0.5 * ((auc[("H_a", 2)] - auc[("H_a", 1)]) + (auc[("E100", 1)] - auc[("E100", 2)]))


def matched_crossover(feats_by_run: dict[int, np.ndarray], y: np.ndarray, patients: np.ndarray, groups: np.ndarray,
                      n_repeats: int = N_REPEATS, n_boot: int = analyze.N_BOOT, seed: int = 0,
                      fit=analyze.fit_probe, jobs: int = 1) -> dict:
    idx = {arm: np.flatnonzero(groups == arm) for arm in ARMS}
    preds, auc = {}, {}
    for arm, rows in idx.items():
        masks = half_splits(patients[rows], n_repeats)
        for run, feats in feats_by_run.items():
            preds[(arm, run)] = crossfit_predictions(feats[rows], y[rows], patients[rows], masks, fit, jobs)
            auc[(arm, run)] = analyze.macro_auc(y[rows], preds[(arm, run)])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        boot = {}
        for arm, rows in idx.items():
            bi = analyze.patient_boot_index(patients[rows], rng)
            boot |= {(arm, run): analyze.macro_auc(y[rows][bi], preds[(arm, run)][bi]) for run in feats_by_run}
        draws[b] = matched_premium(boot)
    return {"delta": matched_premium(auc), "ci90": np.percentile(draws, [5, 95]).tolist(),
            "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
            "auc": {f"{arm}_run{run}": v for (arm, run), v in auc.items()},
            "n_images": {arm: int(len(rows)) for arm, rows in idx.items()}, "n_repeats": n_repeats, "n_boot": n_boot}


def verdict(res: dict, margin: float = analyze.MARGIN) -> str:
    lo, hi = res["ci90"]
    if hi < margin:
        return "a premium above 0.5 points is excluded also with matched probes"
    if lo > margin:
        return "a premium is reported for the matched setting"
    return "the matched result is inconclusive"


def _final_snapshot(run_dir: Path) -> dict:
    f = max(run_dir.glob("feats_eval_step*.npz"), key=lambda p: int(p.stem.split("step")[1]))
    d = np.load(f)
    return {"step": int(f.stem.split("step")[1]), "rows": d["rows"], "feats": d["feats"].astype(np.float32)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", type=Path, required=True)
    ap.add_argument("--run2", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=analyze.N_BOOT)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    groups = pd.read_csv(a.groups)
    snaps = {1: _final_snapshot(a.run1), 2: _final_snapshot(a.run2)}
    if not np.array_equal(snaps[1]["rows"], snaps[2]["rows"]):
        raise SystemExit("the two runs' feature files list different rows")
    rows = snaps[1]["rows"]
    keep = groups.loc[rows, "group"].isin(ARMS).to_numpy()
    rows = rows[keep]
    res = matched_crossover({r: s["feats"][keep] for r, s in snaps.items()}, analyze.labels_of(groups, rows, "uzero"),
                            groups.loc[rows, "patient"].to_numpy(), groups.loc[rows, "group"].to_numpy(),
                            n_boot=a.n_boot, jobs=a.jobs)
    res |= {"steps": {r: s["step"] for r, s in snaps.items()}, "verdict": verdict(res)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "matched.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
