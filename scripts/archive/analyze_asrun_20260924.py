"""Statistics for the randomized exposure study (PREREG.md Sec. 4-6).

Inputs: data/groups.csv and, per run, run_dir/feats_eval_step*.npz from extract_features.py.
Outputs (results/): arm_auc.csv (probe and k-NN macro-AUC per run x snapshot x arm),
deltas.csv (unpaired deltas vs H with cluster-bootstrap CIs), mia.csv (membership AUC and TPR@1%FPR),
primary.json (crossover estimate, 90% CI, TOST decision, permutation p), checks.json.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
C_GRID = (0.01, 0.1, 1.0, 10.0)
MARGIN = 0.005
N_BOOT = 10_000
N_PERM = 10_000
KNN_K, KNN_T = 20, 0.07
ARMS = ("H_a", "H_b", "E3", "E15", "E100", "E500")


@dataclass(frozen=True)
class Snapshot:
    run: int
    step: int
    rows: np.ndarray
    feats: np.ndarray
    inv_rows: np.ndarray
    inv: np.ndarray
    dino_ce: np.ndarray


def load_snapshots(run_dir: Path, run: int) -> list[Snapshot]:
    snaps = []
    for f in sorted(p for p in run_dir.glob("feats_eval_step*.npz") if ".tmp" not in p.name):
        d = np.load(f)
        step = int(f.stem.split("step")[1])
        ce = d["dino_ce"] if "dino_ce" in d.files else np.zeros(0)
        snaps.append(Snapshot(run, step, d["rows"], d["feats"].astype(np.float32), d["inv_rows"], d["inv"], ce))
    return snaps


def labels_of(groups: pd.DataFrame, rows: np.ndarray, policy: str) -> np.ndarray:
    return groups.loc[rows, [f"{lab}__{policy}" for lab in LABELS]].to_numpy()


def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    aucs = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(aucs))


# ----------------------------------------------------------------------------- probes

def fit_probe(x: np.ndarray, y: np.ndarray, patients: np.ndarray, seed: int = 0):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = (x - mu) / sd
    best_c, best = C_GRID[0], -1.0
    for c in C_GRID:
        scores = []
        for tr, va in GroupKFold(n_splits=5).split(xs, y, patients):
            p = np.column_stack([LogisticRegression(C=c, max_iter=2000).fit(xs[tr], y[tr, j]).predict_proba(xs[va])[:, 1]
                                 for j in range(y.shape[1])])
            scores.append(macro_auc(y[va], p))
        if np.mean(scores) > best:
            best_c, best = c, float(np.mean(scores))
    models = [LogisticRegression(C=best_c, max_iter=2000).fit(xs, y[:, j]) for j in range(y.shape[1])]
    return (lambda z: np.column_stack([m.predict_proba((z - mu) / sd)[:, 1] for m in models])), best_c


def knn_scores(bank: np.ndarray, bank_y: np.ndarray, query: np.ndarray) -> np.ndarray:
    b = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    out = np.zeros((len(q), bank_y.shape[1]))
    for s in range(0, len(q), 2048):
        sim = q[s:s + 2048] @ b.T
        idx = np.argpartition(-sim, KNN_K, axis=1)[:, :KNN_K]
        w = np.exp(np.take_along_axis(sim, idx, axis=1) / KNN_T)
        out[s:s + 2048] = np.einsum("nk,nkj->nj", w, bank_y[idx]) / w.sum(1, keepdims=True)
    return out


# ----------------------------------------------------------------------------- bootstrap helpers

def patient_boot_index(patients: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    uniq, inv = np.unique(patients, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    pick = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([members[k] for k in pick])


def boot_delta(y_a, p_a, pat_a, y_b, p_b, pat_b, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    point = macro_auc(y_a, p_a) - macro_auc(y_b, p_b)
    draws = []
    for _ in range(n):
        ia, ib = patient_boot_index(pat_a, rng), patient_boot_index(pat_b, rng)
        draws.append(macro_auc(y_a[ia], p_a[ia]) - macro_auc(y_b[ib], p_b[ib]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


# ----------------------------------------------------------------------------- per-run evaluation

def evaluate_snapshot(snap: Snapshot, groups: pd.DataFrame, policy: str, probe_set: str) -> dict:
    g = groups.loc[snap.rows, "group"].to_numpy()
    pat = groups.loc[snap.rows, "patient"].to_numpy()
    y = labels_of(groups, snap.rows, policy)
    bank = g == probe_set
    predict, c = fit_probe(snap.feats[bank], y[bank], pat[bank])
    probs = predict(snap.feats)
    knn = knn_scores(snap.feats[bank], y[bank], snap.feats)
    return {"group": g, "patient": pat, "y": y, "probe": probs, "knn": knn, "C": c}


def arm_table(ev: dict, run: int, step: int) -> list[dict]:
    rows = []
    for arm in ARMS:
        m = ev["group"] == arm
        if m.sum() == 0:
            continue
        rows.append({"run": run, "step": step, "arm": arm, "n": int(m.sum()),
                     "probe_auc": macro_auc(ev["y"][m], ev["probe"][m]),
                     "knn_auc": macro_auc(ev["y"][m], ev["knn"][m])})
    return rows


def arm_doses(groups: pd.DataFrame, run: int) -> dict:
    return groups.groupby("group")[f"dose_run{run}"].first().to_dict()


def unpaired_deltas(ev: dict, run: int, step: int, n_boot: int, doses: dict) -> list[dict]:
    """Each exposed test arm vs the pooled held-out test arms of the same run (arms chosen by dose, not name)."""
    held = [a for a in ARMS if doses.get(a, 0) == 0]
    h = np.isin(ev["group"], held)
    out = []
    for arm in [a for a in ARMS if doses.get(a, 0) > 0]:
        m = ev["group"] == arm
        for kind in ("probe", "knn"):
            rng = np.random.default_rng(0)
            point = macro_auc(ev["y"][m], ev[kind][m]) - macro_auc(ev["y"][h], ev[kind][h])
            draws = []
            for _ in range(n_boot):
                ia, ib = patient_boot_index(ev["patient"][m], rng), patient_boot_index(ev["patient"][h], rng)
                ya, pa, yb, pb = ev["y"][m][ia], ev[kind][m][ia], ev["y"][h][ib], ev[kind][h][ib]
                draws.append(macro_auc(ya, pa) - macro_auc(yb, pb))
            d = np.asarray(draws)
            p_two = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
            out.append({"run": run, "step": step, "arm": arm, "dose": int(doses[arm]), "held_out": "+".join(held),
                        "kind": kind, "delta": point, "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
                        "lo90": float(np.percentile(d, 5)), "hi90": float(np.percentile(d, 95)), "p_boot": p_two,
                        "equivalent_0.005": bool(np.percentile(d, 5) > -MARGIN and np.percentile(d, 95) < MARGIN)})
    return out


def baseline_adjusted(ev0: dict, ev1: dict, run: int, step: int, n_boot: int, doses: dict) -> list[dict]:
    """Exploratory: change in (exposed arm - held-out) probe macro-AUC from step 0 to `step`, i.e. the arm gap
    after training minus the same arm gap before any exposure. The same patient resample is used at both
    snapshots, so each draw is a within-image contrast."""
    held = [a for a in ARMS if doses.get(a, 0) == 0]
    h = np.isin(ev1["group"], held)
    out = []
    for arm in [a for a in ARMS if doses.get(a, 0) > 0]:
        m = ev1["group"] == arm
        def did(im, ih):
            gap1 = macro_auc(ev1["y"][m][im], ev1["probe"][m][im]) - macro_auc(ev1["y"][h][ih], ev1["probe"][h][ih])
            gap0 = macro_auc(ev0["y"][m][im], ev0["probe"][m][im]) - macro_auc(ev0["y"][h][ih], ev0["probe"][h][ih])
            return gap1 - gap0
        rng = np.random.default_rng(1)
        all_m, all_h = np.arange(m.sum()), np.arange(h.sum())
        point = did(all_m, all_h)
        draws = np.array([did(patient_boot_index(ev1["patient"][m], rng), patient_boot_index(ev1["patient"][h], rng))
                          for _ in range(n_boot)])
        out.append({"run": run, "step": step, "arm": arm, "dose": int(doses[arm]), "adjusted_delta": point,
                    "lo95": float(np.percentile(draws, 2.5)), "hi95": float(np.percentile(draws, 97.5))})
    return out


# ----------------------------------------------------------------------------- membership inference

def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr_max: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_max, fpr, tpr))


def _score_series(snap: Snapshot, kind: str) -> pd.Series:
    if kind == "inv":
        return pd.Series(snap.inv, index=snap.inv_rows)
    return pd.Series(-snap.dino_ce, index=snap.inv_rows) if len(snap.dino_ce) else pd.Series(dtype=float)


def mia_table(snaps: dict, groups: pd.DataFrame) -> list[dict]:
    """Membership AUC of each exposed arm vs H_b (dose 0 in both runs), PREREG 5.3.

    Scores: augmentation invariance (primary) and negative DINO cross-entropy (secondary), each calibrated by
    subtracting a reference: the partner run at the same step for arms whose dose differs between runs
    (H_a, E100), otherwise the same run's step-0 encoder.
    """
    out = []
    by_run = {r: {s.step: s for s in ss} for r, ss in snaps.items()}
    for run, run_snaps in by_run.items():
        doses = arm_doses(groups, run)
        other = by_run.get(3 - run, {})
        for step, snap in run_snaps.items():
            if step == 0:
                continue
            for kind in ("inv", "dino_ce"):
                cur = _score_series(snap, kind)
                if cur.empty:
                    continue
                for arm in [a for a in ARMS if doses.get(a, 0) > 0]:
                    partner_differs = arm_doses(groups, 3 - run).get(arm) != doses[arm] if other else False
                    ref_snap = other.get(step) if partner_differs and step in other else run_snaps[0]
                    ref = _score_series(ref_snap, kind)
                    if ref.empty:
                        continue  # reference extracted without this score
                    cal = (cur - ref).dropna()
                    g = groups.loc[cal.index, "group"]
                    pos, neg = cal[g == arm].to_numpy(), cal[g == "H_b"].to_numpy()
                    yy, ss = np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]
                    from scipy.stats import mannwhitneyu
                    out.append({"run": run, "step": step, "arm": arm, "dose": int(doses[arm]), "score": kind,
                                "reference": "partner_run" if ref_snap is not run_snaps[0] else "step0",
                                "mia_auc": roc_auc_score(yy, ss), "tpr_at_1pct_fpr": tpr_at_fpr(yy, ss),
                                "p_mwu": float(mannwhitneyu(pos, neg, alternative="greater").pvalue)})
    return out


# ----------------------------------------------------------------------------- primary crossover endpoint

def _xstat(ev1: dict, ev2: dict, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    """1/2[(AUC_A,R2 - AUC_A,R1) + (AUC_B,R1 - AUC_B,R2)]: A exposed in Run 2, B exposed in Run 1."""
    def auc(ev, i):
        return macro_auc(ev["y"][i], ev["probe"][i])
    return 0.5 * ((auc(ev2, idx_a) - auc(ev1, idx_a)) + (auc(ev1, idx_b) - auc(ev2, idx_b)))


def _take_patients(order: np.ndarray, members: dict, target: int, start: int) -> tuple[np.ndarray, int]:
    picked, n, k = [], 0, start
    while n < target and k < len(order):
        picked.append(members[order[k]]); n += len(members[order[k]]); k += 1
    return np.concatenate(picked), k


def run_diff(ev1: dict, ev2: dict, idx: np.ndarray) -> float:
    """D_S = AUC_S(Run 2) - AUC_S(Run 1) for an image set S."""
    return macro_auc(ev2["y"][idx], ev2["probe"][idx]) - macro_auc(ev1["y"][idx], ev1["probe"][idx])


def placebo_var_d(ev1: dict, ev2: dict, pools: tuple[str, ...], size: int, n_draws: int,
                  rng: np.random.Generator) -> float:
    """Superpopulation variance of D_S for patient-level random sets of `size` images.

    Sets are drawn from groups whose exposure is identical in both runs (so E[D_S] is the pure run
    effect). Draws from a finite pool of N images understate the variance by (1 - size/N); the finite
    population correction undoes that.
    """
    rows = np.flatnonzero(np.isin(ev1["group"], pools))
    pats = ev1["patient"][rows]
    members = {pid: rows[pats == pid] for pid in np.unique(pats)}
    keys = np.array(list(members))
    draws = np.empty(n_draws)
    for b in range(n_draws):
        idx, _ = _take_patients(rng.permutation(keys), members, size, 0)
        draws[b] = run_diff(ev1, ev2, idx)
    fpc = 1.0 - size / len(rows)
    return float(draws.var(ddof=1) / max(fpc, 1e-3))


def crossover(ev1: dict, ev2: dict, n_boot: int, n_perm: int, seed: int = 0,
              pools: tuple[str, ...] = ("P_clean", "H_b")) -> dict:
    """Primary endpoint (PREREG Amendments 1-3). G1 = H_a, G2 = E100 (Run-1 roles); rows align across runs.

    Delta_x = 1/2 (D_G1 - D_G2). Under no exposure effect D_G1 and D_G2 are independent draws with the same
    mean (the run effect), so Var(Delta_x) = (Var(D_n1) + Var(D_n2)) / 4, estimated from placebo sets.
    """
    from scipy.stats import norm
    g1, g2 = np.flatnonzero(ev1["group"] == "H_a"), np.flatnonzero(ev1["group"] == "E100")
    point = 0.5 * (run_diff(ev1, ev2, g1) - run_diff(ev1, ev2, g2))
    rng = np.random.default_rng(seed)
    var = (placebo_var_d(ev1, ev2, pools, len(g1), n_perm, rng) + placebo_var_d(ev1, ev2, pools, len(g2), n_perm, rng)) / 4
    se = float(np.sqrt(var))
    boot = []
    for _ in range(n_boot):
        i1 = g1[patient_boot_index(ev1["patient"][g1], rng)]
        i2 = g2[patient_boot_index(ev1["patient"][g2], rng)]
        boot.append(0.5 * (run_diff(ev1, ev2, i1) - run_diff(ev1, ev2, i2)))
    boot_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None
    lo90, hi90 = point - 1.645 * se, point + 1.645 * se
    return {"delta_x": point, "se": se, "ci90": [lo90, hi90], "ci95": [point - 1.96 * se, point + 1.96 * se],
            "ci95_boot_sampling_only": boot_ci,
            "p": float(2 * norm.sf(abs(point) / se)), "margin": MARGIN,
            "equivalent": bool(lo90 > -MARGIN and hi90 < MARGIN), "placebo_pools": list(pools)}


# ----------------------------------------------------------------------------- main

def validity_checks(arm_rows: list[dict], mia_rows: list[dict], groups: pd.DataFrame) -> dict:
    """PREREG Section 6: positive control, adaptation, collapse (per run, final vs step 0, held-out arms pooled
    as the mean of their macro-AUCs)."""
    a = pd.DataFrame(arm_rows)
    m = pd.DataFrame(mia_rows)
    out = {}
    for run in sorted(a.run.unique()):
        held = [k for k, v in arm_doses(groups, run).items() if v == 0 and k in ARMS]
        r = a[(a.run == run) & a.arm.isin(held)]
        first, last = r.step.min(), r.step.max()
        auc0, auc1 = r[r.step == first].probe_auc.mean(), r[r.step == last].probe_auc.mean()
        knn0, knn1 = r[r.step == first].knn_auc.mean(), r[r.step == last].knn_auc.mean()
        pc = m[(m.run == run) & (m.step == last) & (m.arm == "E500") & (m.score == "inv")] if len(m) else m
        out[f"run{run}"] = {
            "positive_control_E500_mia_auc": float(pc.mia_auc.iloc[0]) if len(pc) else None,
            "positive_control_passed": bool(len(pc) and pc.mia_auc.iloc[0] > 0.6),
            "adaptation_probe_auc_step0_final": [float(auc0), float(auc1)], "adaptation_passed": bool(auc1 > auc0),
            "collapse_knn_auc_step0_final": [float(knn0), float(knn1)], "no_collapse": bool(knn1 >= knn0 - 0.01)}
    return out


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--run1", type=Path, required=True)
    ap.add_argument("--run2", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", choices=("uzero", "uone"), default="uzero")
    ap.add_argument("--probe-set", choices=("P_base", "P_clean"), default="P_base")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(a.groups)
    snaps = {1: load_snapshots(a.run1, 1)}
    if a.run2:
        snaps[2] = load_snapshots(a.run2, 2)
    evals, arm_rows, delta_rows, adjusted_rows = {}, [], [], []
    for run, run_snaps in snaps.items():
        for s in run_snaps:
            ev = evaluate_snapshot(s, groups, a.policy, a.probe_set)
            evals[(run, s.step)] = ev
            arm_rows += arm_table(ev, run, s.step)
            if s.step > 0:
                delta_rows += unpaired_deltas(ev, run, s.step, a.n_boot, arm_doses(groups, run))
                adjusted_rows += baseline_adjusted(evals[(run, 0)], ev, run, s.step, min(a.n_boot, 2000), arm_doses(groups, run))
    tag = f"{a.policy}_{a.probe_set}"
    pd.DataFrame(arm_rows).to_csv(a.out / f"arm_auc_{tag}.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(a.out / f"deltas_{tag}.csv", index=False)
    pd.DataFrame(adjusted_rows).to_csv(a.out / f"deltas_adjusted_{tag}.csv", index=False)
    mia_rows = mia_table(snaps, groups)
    pd.DataFrame(mia_rows).to_csv(a.out / "mia.csv", index=False)
    (a.out / f"checks_{tag}.json").write_text(json.dumps(validity_checks(arm_rows, mia_rows, groups), indent=2))
    fam = [(f"delta_{r['kind']}_run{r['run']}_step{r['step']}_{r['arm']}", r["p_boot"]) for r in delta_rows]
    fam += [(f"mia_{r['score']}_run{r['run']}_step{r['step']}_{r['arm']}", r["p_mwu"]) for r in mia_rows]
    if fam:
        adj = holm([p for _, p in fam])
        pd.DataFrame({"test": [t for t, _ in fam], "p": [p for _, p in fam], "p_holm": adj}).to_csv(
            a.out / f"holm_{tag}.csv", index=False)
    if 2 in snaps:
        last1, last2 = max(s.step for s in snaps[1]), max(s.step for s in snaps[2])
        for run_dir, last in ((a.run1, last1), (a.run2, last2)):
            done = run_dir / "DONE"
            if done.exists() and int(done.read_text().strip()) != last:
                raise SystemExit(f"{run_dir}: last feature snapshot {last} != finished step {done.read_text()}")
        r1 = {s.step: s for s in snaps[1]}[last1].rows
        r2 = {s.step: s for s in snaps[2]}[last2].rows
        if not np.array_equal(r1, r2):
            raise SystemExit("rows differ between runs; crossover requires identical evaluation rows")
        pools = ("P_clean", "H_b") if a.probe_set == "P_base" else ("P_base", "H_b")
        primary = crossover(evals[(1, last1)], evals[(2, last2)], a.n_boot, a.n_perm, pools=pools)
        (a.out / f"primary_{tag}.json").write_text(json.dumps(primary, indent=2))
        print("primary:", primary)
    print(pd.DataFrame(arm_rows).to_string(index=False))


if __name__ == "__main__":
    main()
