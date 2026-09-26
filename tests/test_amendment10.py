"""Amendment 10: crossover premium with probes trained on images exposed as often as the test images."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import amendment10_matched as a10  # noqa: E402


def test_half_splits_keep_patients_together_and_differ_between_repeats():
    patients = np.repeat(np.arange(300), 3)
    masks = a10.half_splits(patients, n_repeats=5, seed=0)
    assert len(masks) == 5
    for m in masks:
        per_patient = {p: set(m[patients == p]) for p in np.unique(patients)}
        assert all(len(v) == 1 for v in per_patient.values())
        assert 0.45 < m.mean() < 0.55
    assert not np.array_equal(masks[0], masks[1])
    assert all(np.array_equal(a, b) for a, b in zip(masks, a10.half_splits(patients, 5, 0)))


def test_each_image_is_predicted_by_a_probe_trained_on_the_other_half():
    feats, y = np.zeros((6, 2)), np.zeros((6, 1))
    mask = np.array([True, True, True, False, False, False])

    def fake_fit(x, yy, pat):
        n = len(x)  # rows it was trained on; tag predictions with that count's half
        tag = 1.0 if n == 3 and fake_fit.calls == 0 else 2.0
        fake_fit.calls += 1
        return (lambda z: np.full((len(z), 1), tag)), None
    fake_fit.calls = 0
    p = a10.crossfit_predictions(feats, y, np.arange(6), [mask], fit=fake_fit)
    assert np.all(p[~mask] == 1.0) and np.all(p[mask] == 2.0)  # first probe (trained on mask) scored ~mask


def test_matched_premium_formula():
    auc = {("H_a", 1): 0.70, ("H_a", 2): 0.72, ("E100", 1): 0.71, ("E100", 2): 0.70}
    assert a10.matched_premium(auc) == pytest.approx(0.015)


def test_matched_crossover_detects_an_injected_premium_and_not_a_fake_one():
    rng = np.random.default_rng(0)
    n, d = 240, 6
    groups = np.r_[["H_a"] * n, ["E100"] * n]
    patients = np.arange(2 * n) // 2
    y = (rng.random((2 * n, 2)) < 0.4).astype(int)
    base = rng.normal(size=(2 * n, d))
    signal = np.c_[y, np.zeros((2 * n, d - 2))]
    feats_same = base + 0.6 * signal
    exposed1 = (groups == "E100")[:, None]  # run 1 exposes E100, run 2 exposes H_a
    boost = 0.8 * signal
    run1 = feats_same + boost * exposed1
    run2 = feats_same + boost * ~exposed1
    res = a10.matched_crossover({1: run1, 2: run2}, y, patients, groups, n_repeats=2, n_boot=200)
    assert res["delta"] > 0.02 and res["ci90"][0] > 0
    null = a10.matched_crossover({1: feats_same, 2: feats_same}, y, patients, groups, n_repeats=2, n_boot=200)
    assert abs(null["delta"]) < 1e-12
