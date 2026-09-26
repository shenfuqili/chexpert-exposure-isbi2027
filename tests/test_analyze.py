"""Checks that the crossover estimator recovers an injected exposure effect and stays null without one."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze as an  # noqa: E402

SIZES = {"P_base": 1500, "P_clean": 1500, "H_a": 600, "H_b": 600, "E100": 600, "E3": 300, "E15": 300, "E500": 100}


def synth(effect: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, pid = [], 0
    for grp, n in SIZES.items():
        for i in range(n):
            if i % 3 == 0:
                pid += 1
            y = (rng.random(5) < 0.3).astype(int)
            rows.append({"group": grp, "patient": pid, **{f"{lab}__uzero": y[j] for j, lab in enumerate(an.LABELS)}})
    groups = pd.DataFrame(rows)
    y = groups[[f"{lab}__uzero" for lab in an.LABELS]].to_numpy()
    w = rng.normal(size=(5, 32))
    base_noise = rng.normal(size=(len(groups), 32)) * 3.0

    def feats(exposed_arm: str) -> np.ndarray:
        gain = np.where(groups["group"].eq(exposed_arm) | groups["group"].eq("E500"), 1 + effect, 1.0)
        return (y @ w) * gain[:, None] + base_noise + rng.normal(size=base_noise.shape) * 0.3

    ev1 = {"group": groups["group"].to_numpy(), "patient": groups["patient"].to_numpy(), "y": y}
    ev2 = dict(ev1)
    for ev, arm in ((ev1, "E100"), (ev2, "H_a")):
        x = feats(arm)
        bank = ev["group"] == "P_base"
        predict, _ = an.fit_probe(x[bank], y[bank], ev["patient"][bank])
        ev["probe"] = predict(x)
    return ev1, ev2


def test_crossover_detects_injected_effect():
    ev1, ev2 = synth(effect=0.6)
    res = an.crossover(ev1, ev2, n_boot=200, n_perm=200)
    assert res["delta_x"] > 0.01 and res["ci95"][0] > 0 and not res["equivalent"]


def test_crossover_null_covers_zero():
    ev1, ev2 = synth(effect=0.0, seed=1)
    res = an.crossover(ev1, ev2, n_boot=200, n_perm=200)
    assert res["ci95"][0] < 0 < res["ci95"][1] and res["p"] > 0.01


def test_holm_monotone_and_capped():
    adj = an.holm([0.01, 0.04, 0.03, 0.5])
    assert adj[0] == 0.04 and max(adj) <= 1.0 and adj[3] == 0.5


def test_tpr_at_fpr_perfect_separation():
    y = np.r_[np.ones(100), np.zeros(100)]
    assert an.tpr_at_fpr(y, np.r_[np.ones(100) + 1, np.zeros(100)]) == 1.0


def _mia_groups(n: int = 50) -> pd.DataFrame:
    doses = {"E100": (100, 0), "H_a": (0, 100), "H_b": (0, 0), "E3": (3, 3), "E15": (15, 15), "E500": (500, 500)}
    rows = []
    for grp, (d1, d2) in doses.items():
        rows += [{"group": grp, "patient": len(rows) + i, "dose_run1": d1, "dose_run2": d2} for i in range(n)]
    return pd.DataFrame(rows)


def test_mia_uses_partner_run_reference_when_step_counts_differ():
    """E100 and H_a swap roles between runs, so PREREG 5.3 calibrates them against the partner run's snapshot at
    the same schedule position; the two runs' step numbers differ by a few steps (18160 vs 18171)."""
    groups = _mia_groups()
    rng = np.random.default_rng(0)
    idx, g = groups.index.to_numpy(), groups["group"].to_numpy()
    base = rng.normal(size=len(groups)) * 5          # per-image difficulty shared by every snapshot
    step0 = base + rng.normal(size=len(groups)) * 5  # the shared initial encoder adds unrelated noise

    def snap(run: int, step: int, inv: np.ndarray) -> an.Snapshot:
        return an.Snapshot(run, step, idx, np.zeros((len(idx), 2)), idx, inv, np.zeros(0))

    def trained(exposed: str) -> np.ndarray:
        return base + 1.0 * (g == exposed) + rng.normal(size=len(groups)) * 0.05

    snaps = {1: [snap(1, 0, step0), snap(1, 100, trained("E100")), snap(1, 300, trained("E100"))],
             2: [snap(2, 0, step0), snap(2, 101, trained("H_a")), snap(2, 303, trained("H_a"))]}
    rows = pd.DataFrame(an.mia_table(snaps, groups))
    swapped = rows[((rows.run == 1) & (rows.arm == "E100")) | ((rows.run == 2) & (rows.arm == "H_a"))]
    assert len(swapped) == 4 and (swapped.reference == "partner_run").all()
    assert (swapped.mia_auc > 0.95).all()
    assert (rows[rows.arm.isin(["E3", "E15", "E500"])].reference == "step0").all()


def test_crossover_loglik_detects_injected_effect_and_stays_null():
    """PREREG secondary 6: the crossover contrast of per-image probe log-likelihood."""
    ev1, ev2 = synth(effect=0.6)
    hit = an.crossover(ev1, ev2, n_boot=0, n_perm=200, stat=an.mean_loglik)
    assert hit["delta_x"] > 0 and hit["ci95"][0] > 0
    ev1, ev2 = synth(effect=0.0, seed=1)
    null = an.crossover(ev1, ev2, n_boot=0, n_perm=200, stat=an.mean_loglik)
    assert null["ci95"][0] < 0 < null["ci95"][1]


def test_progress_file_is_written_and_write_failures_are_ignored(tmp_path):
    import json
    prog = an.Progress(tmp_path / "progress.json", total=10)
    prog(0, "stage A")
    prog(4)
    rec = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert rec["done"] == 4 and rec["total"] == 10 and rec["stage"] == "stage A"
    prog.finish()
    rec = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert rec["done"] == rec["total"] and rec["stages"][0]["stage"] == "stage A"
    an.Progress(tmp_path / "missing_dir" / "p.json", total=1)(1, "x")  # must not raise


def test_progress_ticks_do_not_change_results():
    ev1, _ = synth(effect=0.3)
    ev1["knn"] = ev1["probe"]  # synth() builds probe scores only
    doses = {"H_a": 0, "H_b": 0, "E3": 3, "E15": 15, "E100": 100, "E500": 500}
    ticks = []
    plain = an.unpaired_deltas(ev1, 1, 10, 50, doses)
    ticked = an.unpaired_deltas(ev1, 1, 10, 50, doses, tick=ticks.append)
    assert plain == ticked and sum(ticks) == 4 * 2 * 50
