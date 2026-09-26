"""Amendment 9: null-arm membership AUCs and the RAD-DINO pool checks (intensity statistics, labels)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import amendment9_checks as a9  # noqa: E402


def test_intensity_statistics_are_scaled_to_unit_range():
    img = np.linspace(20, 220, 101 * 100).reshape(101, 100)
    s = a9.intensity_stats(img)
    assert s.shape == (9,)
    assert abs(s[0] - 0.5) < 1e-6  # mean
    assert np.allclose(s[2:], [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], atol=1e-3)


def test_out_of_fold_auc_finds_a_real_difference_and_not_a_fake_one():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(150), np.zeros(350)]
    pat = np.arange(500) // 2
    noise = rng.normal(size=(500, 4))
    auc_real, ci_real = a9.oof_auc(noise + 1.5 * y[:, None], y, pat, n_boot=300)
    auc_fake, ci_fake = a9.oof_auc(noise, y, pat, n_boot=300)
    assert auc_real > 0.9 and ci_real[0] > 0.8
    assert 0.4 < auc_fake < 0.6 and ci_fake[0] < 0.5 < ci_fake[1]


def test_null_auc_compares_two_arms_with_a_patient_bootstrap():
    rng = np.random.default_rng(1)
    groups = pd.DataFrame({"group": ["H_a"] * 400 + ["H_b"] * 400 + ["E3"] * 50,
                           "patient": np.arange(850) // 2})
    same = pd.Series(rng.normal(size=850), index=groups.index)
    res = a9.null_auc(same, groups, "H_a", "H_b", n_boot=300)
    assert res["n_pos"] == 400 and res["n_neg"] == 400
    assert res["ci95"][0] < 0.5 < res["ci95"][1]
    shifted = same + 1.0 * (groups.group == "H_a").to_numpy()
    assert a9.null_auc(shifted, groups, "H_a", "H_b", n_boot=300)["auc"] > 0.7
