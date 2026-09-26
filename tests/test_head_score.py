"""Amendment 6: the single-network head score for the RAD-DINO bridge and for the proxy calibration."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import head_score as hs  # noqa: E402


def test_self_distill_score_matches_naive_loop():
    torch.manual_seed(0)
    logits = torch.randn(3, 4, 50)
    want = torch.zeros(3)
    for b in range(3):
        ces = [-(F.softmax(logits[b, i] / 0.07, -1) * F.log_softmax(logits[b, j] / 0.1, -1)).sum()
               for i in range(4) for j in range(4) if i != j]
        want[b] = -torch.stack(ces).mean()
    assert torch.allclose(hs.self_distill_score(logits), want, atol=1e-5)


def test_self_distill_score_is_higher_for_consistent_views():
    torch.manual_seed(1)
    consistent = (torch.randn(1, 1, 50) * 3).repeat(1, 8, 1) + 0.01 * torch.randn(1, 8, 50)
    scattered = torch.randn(1, 8, 50) * 3
    assert hs.self_distill_score(consistent)[0] > hs.self_distill_score(scattered)[0]


def test_released_head_matches_torch_weight_norm():
    """RAD-DINO's dino_head.safetensors stores the last layer as weight_g / weight_v (torch.nn.utils.weight_norm)."""
    torch.manual_seed(2)
    mlp = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 8))
    last = nn.utils.weight_norm(nn.Linear(8, 20, bias=False))
    with torch.no_grad():
        last.weight_g.uniform_(0.5, 2.0)
    x = torch.randn(5, 16)
    want = last(F.normalize(mlp(x), dim=-1))
    sd = {f"mlp.{k}": v for k, v in mlp.state_dict().items()}
    sd["last_layer.weight_g"], sd["last_layer.weight_v"] = last.weight_g.detach(), last.weight_v.detach()
    assert torch.allclose(hs.ReleasedDinoHead(sd)(x), want, atol=1e-5)


def test_proxy_mia_uncalibrated_and_partner_calibrated():
    groups = pd.DataFrame({"group": ["H_b"] * 40 + ["E100"] * 40 + ["H_a"] * 40 + ["E3"] * 40,
                           "dose_run1": [0] * 40 + [100] * 40 + [0] * 40 + [3] * 40,
                           "dose_run2": [0] * 40 + [0] * 40 + [100] * 40 + [3] * 40})
    rng = np.random.default_rng(0)
    base = rng.normal(size=len(groups)) * 3  # image difficulty shared by both runs
    g = groups["group"].to_numpy()
    s1 = pd.Series(base + 1.0 * (g == "E100") + rng.normal(size=len(g)) * 0.1, index=groups.index)
    s2 = pd.Series(base + 1.0 * (g == "H_a") + rng.normal(size=len(g)) * 0.1, index=groups.index)
    res = hs.proxy_mia({1: s1, 2: s2}, groups)
    cal = res[res.calibration == "partner_run"]
    assert set(zip(cal.run, cal.arm)) == {(1, "E100"), (2, "H_a")} and (cal.mia_auc > 0.95).all()
    unc = res[(res.calibration == "none") & (res.run == 1) & (res.arm == "E100")].mia_auc.iloc[0]
    assert 0.5 < unc < cal[cal.run == 1].mia_auc.iloc[0]
    assert set(res[res.run == 1].arm) == {"E100", "E3"}  # dose-0 arms are not scored as members


@pytest.fixture()
def fake_run(tmp_path: Path) -> Path:
    """Six tiny images and an untrained snapshot in the formal runs' checkpoint format."""
    import timm
    from ssl_train import DINOHead
    rng = np.random.default_rng(0)
    rows = []
    for k, grp in enumerate(["H_b", "H_b", "E100", "E100", "H_a", "E500"]):
        rel = f"CheXpert-v1.0-small/train/patient{k:05d}/study1/view1_frontal.jpg"
        p = tmp_path / "data" / rel.split("CheXpert-v1.0-small/")[1]
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (60, 50), dtype=np.uint8)).save(p)
        rows.append({"Path": rel, "group": grp, "patient": k})
    pd.DataFrame(rows).to_csv(tmp_path / "groups.csv", index=False)
    backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0,
                                 img_size=224, dynamic_img_size=True)
    head = DINOHead(384, 64, use_bn=True).eval()

    def poisoned(sd: dict) -> dict:  # student weights are NaN, so any use of the student makes scores non-finite
        return {k: torch.full_like(v, float("nan")) if v.is_floating_point() else v for k, v in sd.items()}

    run = tmp_path / "run1"
    run.mkdir()
    for step in (0, 10):
        torch.save({"teacher": backbone.state_dict(), "student": poisoned(backbone.state_dict()),
                    "teacher_head": head.state_dict(), "student_head": poisoned(head.state_dict())},
                   run / f"eval_step{step:07d}.pt")
    return tmp_path


def test_proxy_scores_use_the_final_teacher_snapshot(fake_run: Path):
    out = fake_run / "scores.npz"
    code = hs.score_proxy_run(fake_run / "run1", 1, pd.read_csv(fake_run / "groups.csv"), fake_run / "data", out,
                              device=torch.device("cpu"), workers=0, batch=4)
    d = np.load(out)
    assert code == 0 and d["rows"].tolist() == [0, 1, 2, 3, 4, 5] and int(d["step"]) == 10 and int(d["run"]) == 1
    assert np.isfinite(d["score"]).all()  # teacher backbone and teacher head only


def test_proxy_scoring_honours_the_pause_file(fake_run: Path):
    """Watchdog contract: exit code 3 and no output (partial or final) when a game asks for the GPU."""
    out, pause = fake_run / "scores.npz", fake_run / "PAUSE"
    pause.write_text("game", encoding="utf-8")
    code = hs.score_proxy_run(fake_run / "run1", 1, pd.read_csv(fake_run / "groups.csv"), fake_run / "data", out,
                              device=torch.device("cpu"), workers=0, batch=4, pause_file=pause)
    assert code == hs.PAUSED and not out.exists() and not list(fake_run.glob("scores.npz*"))


def test_analyze_mode_matches_scores_to_runs_by_content(tmp_path: Path):
    groups = pd.DataFrame({"group": ["H_b", "E100", "H_a"] * 20, "dose_run1": [0, 100, 0] * 20,
                           "dose_run2": [0, 0, 100] * 20})
    groups.to_csv(tmp_path / "groups.csv", index=False)
    rows = groups.index.to_numpy()
    for run in (1, 2):
        np.savez(tmp_path / f"s{run}.npz", rows=rows, score=np.random.default_rng(run).normal(size=len(rows)),
                 step=10, run=run)
    np.savez(tmp_path / "dup.npz", rows=rows, score=np.zeros(len(rows)), step=10, run=1)

    def run_analyze(first: str, second: str, out: str) -> pd.DataFrame:
        hs.main(["analyze", "--groups", str(tmp_path / "groups.csv"), "--scores", str(tmp_path / first),
                 str(tmp_path / second), "--out", str(tmp_path / out)])
        return pd.read_csv(tmp_path / out)

    pd.testing.assert_frame_equal(run_analyze("s1.npz", "s2.npz", "a.csv"), run_analyze("s2.npz", "s1.npz", "b.csv"))
    with pytest.raises(SystemExit):
        run_analyze("s1.npz", "dup.npz", "c.csv")  # both files claim run 1


def test_hf_dinov2_cls_token_is_after_the_final_layernorm():
    """The bridge feeds HF last_hidden_state[:, 0] to the DINO head, which DINOv2 applies to x_norm_clstoken."""
    from transformers import Dinov2Config, Dinov2Model
    cfg = Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                       image_size=28, patch_size=14)
    model = Dinov2Model(cfg).eval()
    with torch.no_grad():
        out = model(pixel_values=torch.randn(2, 3, 28, 28), output_hidden_states=True)
        assert torch.allclose(out.last_hidden_state[:, 0], model.layernorm(out.hidden_states[-1])[:, 0], atol=1e-6)
        assert not torch.allclose(out.last_hidden_state[:, 0], out.hidden_states[-1][:, 0], atol=1e-3)


def test_bridge_control_scores_the_bridge_images_with_proxy_teachers(fake_run: Path):
    """Amendment 7 negative control: proxy teacher + teacher head on the bridge images, one column per snapshot."""
    groups = pd.read_csv(fake_run / "groups.csv")
    bridge = pd.DataFrame({"path": [str(fake_run / "data" / p.split("CheXpert-v1.0-small/")[1]) for p in groups.Path],
                           "member": [1, 1, 1, 0, 0, 0], "patient": range(6)})
    bridge.to_csv(fake_run / "bridge.csv", index=False)
    snap = fake_run / "run1" / "eval_step0000010.pt"
    res = hs.score_bridge_with_proxies(fake_run / "bridge.csv", {"run1": snap, "run2": snap}, torch.device("cpu"))
    assert list(res.columns) == ["path", "member", "patient", "head_run1", "head_run2"]
    assert np.isfinite(res[["head_run1", "head_run2"]].to_numpy()).all()  # teacher only (student is NaN)
    assert np.allclose(res.head_run1, res.head_run2)  # same snapshot, same fixed views


def test_lowpass_keeps_size_and_removes_detail_finer_than_the_target_resolution():
    """Amendment 8: shorter side down to 224 and back up; coarse structure survives, pixel-level detail does not."""
    rng = np.random.default_rng(0)
    coarse = np.kron(rng.uniform(0, 255, (13, 17)), np.ones((40, 40)))  # 520 x 680, blocks of 40 px
    fine = np.indices(coarse.shape).sum(0) % 2 * 60.0  # 1-pixel checkerboard
    img = Image.fromarray(np.clip(coarse + fine, 0, 255).astype(np.uint8)).convert("RGB")
    out = hs.lowpass(img, 224)
    assert out.size == img.size and out.mode == img.mode
    a, b = np.asarray(img, dtype=float)[..., 0], np.asarray(out, dtype=float)[..., 0]
    hf = lambda x: np.abs(np.diff(x, axis=1)).mean()  # noqa: E731
    assert hf(b) < 0.25 * hf(a)
    assert np.corrcoef(coarse.ravel(), b.ravel())[0, 1] > 0.95
    assert hs.lowpass(img, None) is img
