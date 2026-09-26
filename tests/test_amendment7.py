"""Amendment 7 summary: paired bootstrap of AUC differences and the pre-registered reading rule."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import amendment7_summary as a7  # noqa: E402


def test_paired_difference_is_zero_for_identical_scores_and_positive_for_signal():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(100), np.zeros(300)]
    pat = np.arange(400) // 2
    noise = rng.normal(size=400)
    point, ci = a7.paired_auc_difference(y, noise, noise, pat, n=300)
    assert point == 0 and ci == [0.0, 0.0]
    point, ci = a7.paired_auc_difference(y, noise + 1.5 * y, noise, pat, n=300)
    assert point > 0.2 and ci[0] > 0


def test_reading_rule_follows_the_registration():
    ok_proxy = {"auc_ci95": [0.45, 0.55], "diff_ci95": [0.1, 0.3]}
    pool = {"auc_ci95": [0.62, 0.70], "diff_ci95": [-0.05, 0.05]}
    assert a7.verdict([ok_proxy, ok_proxy], modal_auc=0.70).startswith("memorisation")
    assert a7.verdict([ok_proxy, pool], modal_auc=0.70).startswith("pool difference")
    assert a7.verdict([ok_proxy, ok_proxy], modal_auc=0.55).startswith("size confound")
