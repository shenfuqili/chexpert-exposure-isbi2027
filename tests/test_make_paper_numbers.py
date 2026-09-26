"""paper/numbers.tex generator: number formatting, the Holm family, the claims the text relies on, and the values
read from the real result files."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_paper_numbers as mpn  # noqa: E402


def test_points_are_hundredths_of_auc_with_a_real_minus_and_no_negative_zero():
    assert mpn.pts(-0.0042061) == "$-0.42$"
    assert mpn.pts(0.0016096) == "0.16"
    assert mpn.pts(-0.00004) == "0.00"
    assert mpn.pts(-0.072354, nd=1) == "$-7.2$"


def test_p_values_keep_two_significant_digits_and_floor_at_one_in_a_thousand():
    assert mpn.pval(0.0919806) == "0.092"
    assert mpn.pval(0.3771398) == "0.38"
    assert mpn.pval(0.0186) == "0.019"
    assert mpn.pval(0.0901) == "0.090"
    assert mpn.pval(0.00041) == "$<$0.001"
    assert mpn.pval(1.0) == "1"


def test_interval_uses_to_when_a_bound_is_negative_and_a_dash_otherwise():
    assert mpn.ci(-0.0083122, -0.0000999, mpn.pts) == "$-0.83$ to $-0.01$"
    assert mpn.ci(0.6743757, 0.7544040, mpn.auc) == "0.67--0.75"


def test_macro_names_must_be_letters_only_and_unique():
    assert mpn.macro("PrimDelta", "$-0.42$") == r"\newcommand{\PrimDelta}{$-0.42$}"
    with pytest.raises(ValueError):
        mpn.macro("Prim2", "x")
    with pytest.raises(ValueError):
        mpn.merge({"A": "1"}, {"A": "2"})


def test_extended_holm_adds_endpoints_to_the_registered_family():
    family = {"a": 0.01, "b": 0.04}
    adj = mpn.extended_holm(family, {"c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    with pytest.raises(ValueError):
        mpn.extended_holm(family, {"a": 0.5})


def test_claim_check_fails_loudly_when_the_bootstrap_interval_is_not_the_wider_one():
    wide = {"ci95": [-0.009, 0.001], "ci95_boot_sampling_only": [-0.010, 0.002]}
    narrow = {"ci95": [-0.009, 0.001], "ci95_boot_sampling_only": [-0.008, 0.0]}
    mpn.require_bootstrap_wider({"primary": wide})
    with pytest.raises(ValueError, match="loglik"):
        mpn.require_bootstrap_wider({"primary": wide, "loglik": narrow})


def test_values_read_from_the_result_files():
    v = mpn.build(ROOT)
    assert v["PrimDelta"] == "$-0.42$"
    assert v["PrimCI"] == "$-0.83$ to $-0.01$"
    assert v["PrimP"] == "0.092"
    assert v["CleanDelta"] == "$-0.37$" and v["UoneDelta"] == "$-0.18$"
    assert v["LLDrop"] == "0.020" and v["LLHolm"] == "0.019"
    assert v["RadHead"] == "0.72" and v["RadHeadCI"] == "0.67--0.75"
    assert v["RadBackbone"] == "0.48"
    assert v["TwinOne"] == "0.65" and v["TwinTwo"] == "0.64"
    assert v["AuditorRange"] == "0.52--0.56" and v["AuditorMax"] == "0.56"
    assert v["StepsOne"] == "18{,}160" and v["StepsTwo"] == "18{,}171"
    assert (v["HeadThree"], v["HeadFifteen"], v["HeadHundred"], v["HeadFiveHundred"]) == ("0.51", "0.53", "0.61", "0.78")
    assert (v["HeadTPRHundred"], v["HeadTPRFiveHundred"]) == ("2.4", "11.1")
    assert (v["CEHundred"], v["CEFiveHundred"]) == ("0.61", "0.78")
    assert (v["CtlOne"], v["CtlTwo"], v["DiffOne"], v["DiffTwo"]) == ("0.53", "0.49", "0.19", "0.22")
    assert (v["ModalAUC"], v["ModalCI"], v["SizeWidth"], v["SizeHeight"]) == ("0.71", "0.66--0.76", "0.55", "0.55")
    text = mpn.render(v)
    assert text.count(r"\newcommand") == len(v)


def test_arm_balance_is_described_over_the_larger_arms_only():
    v = mpn.build(ROOT)
    assert v["AgeRange"] == "60--61" and v["APRange"] == "84--86"
    assert v["FemaleLarger"] == "39--43" and v["FemaleEFiveHundred"] == "53" and v["PrevSpread"] == "3"
    assert v["NFrontal"] == "79{,}543" and v["NPatients"] == "26{,}982" and v["EFiveHundredPatients"] == "184"


def test_mean_over_runs_needs_exactly_one_row_per_run_and_dose():
    ok = pd.DataFrame({"dose": [3, 3], "run": [1, 2], "mia_auc": [0.5, 0.6]})
    assert mpn.mean_over_runs(ok, "mia_auc").loc[3] == pytest.approx(0.55)
    with pytest.raises(ValueError):
        mpn.mean_over_runs(pd.DataFrame({"dose": [3, 3, 3], "run": [1, 2, 2], "mia_auc": [0.5, 0.6, 0.7]}), "mia_auc")
    with pytest.raises(ValueError):
        mpn.mean_over_runs(pd.DataFrame({"dose": [3], "run": [1], "mia_auc": [0.5]}), "mia_auc")


def test_half_up_is_for_non_negative_shares_only():
    assert mpn.half_up(38.5) == 39
    with pytest.raises(ValueError):
        mpn.half_up(-2.5)


def test_numbers_added_after_the_internal_review():
    v = mpn.build(ROOT)
    assert v["LowCIMax"] == "2.0"  # highest 95% bound of the dose-3/15 premiums
    assert v["EFiveHundredRunGap"] == "2.8" and v["HbRunGap"] == "0.1"  # same exposure, different run
    assert v["ViewShareHigh"] == "56" and v["RadNIHShare"] == "3"


def test_low_pass_check_of_the_raddino_head_result():
    v = mpn.build(ROOT)
    assert v["RadHeadLow"] == "0.71" and v["RadHeadLowCI"] == "0.67--0.75"


def test_amendment9_nulls_pool_checks_and_the_unpaired_dose100_contrasts():
    v = mpn.build(ROOT)
    assert (v["NullInvOne"], v["NullInvTwo"], v["NullHeadOne"], v["NullHeadTwo"]) == ("0.51", "0.52", "0.51", "0.50")
    assert (v["IntensityAUC"], v["LabelsAUC"]) == ("0.48", "0.52")
    assert (v["UnpairedHundredOne"], v["UnpairedHundredTwo"], v["EHundredHarder"]) == ("$-1.25$", "0.63", "1.7")


def test_amendment10_matched_exposure_premium():
    v = mpn.build(ROOT)
    assert v["MatchedDelta"] == "$-0.45$" and v["MatchedCI"] == "$-0.88$ to $-0.01$"
