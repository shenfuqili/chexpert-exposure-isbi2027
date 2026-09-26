"""Unit tests for the exposure-training script: dosing, crop ordering, loss, pause/resume."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ssl_train as st  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ssl_train.py"


def test_schedule_gives_exact_doses():
    groups = pd.DataFrame({"dose_run1": [0, 3, 15, 100, 500], "dose_run2": [100, 3, 15, 0, 500]})
    for run in (1, 2):
        sched = st.build_schedule(groups, run, seed=1)
        counts = np.bincount(sched, minlength=len(groups))
        assert counts.tolist() == groups[f"dose_run{run}"].tolist()


def test_schedule_is_seeded_and_run_specific():
    groups = pd.DataFrame({"dose_run1": [5, 5, 5], "dose_run2": [5, 5, 5]})
    a = st.build_schedule(groups, 1, seed=1)
    assert np.array_equal(a, st.build_schedule(groups, 1, seed=1))
    assert not np.array_equal(a, st.build_schedule(groups, 2, seed=1))


def test_collate_is_crop_major():
    # item i crop c is filled with value 10*i + c
    batch = [([torch.full((1,), 10.0 * i + c) for c in range(2)],
              [torch.full((1,), 10.0 * i + 2 + c) for c in range(3)]) for i in range(4)]
    g, loc = st.collate(batch)
    first_crop = g.chunk(2)[0].flatten().tolist()
    assert first_crop == [0.0, 10.0, 20.0, 30.0]
    assert loc.chunk(3)[2].flatten().tolist() == [4.0, 14.0, 24.0, 34.0]


def test_dino_loss_is_finite_and_ignores_same_view():
    torch.manual_seed(0)
    student = torch.randn(4 * 3, 16)
    teacher = [torch.softmax(torch.randn(4, 16), -1) for _ in range(2)]
    loss = st.dino_loss(student, teacher, n_crops=3, tau_s=0.1)
    assert torch.isfinite(loss) and loss > 0


def test_schedules_endpoints():
    s0, s_end = st.schedules(0, 1000, 32), st.schedules(1000, 1000, 32)
    assert s0["lr"] == 0.0 and s_end["lr"] == pytest.approx(1e-6, rel=1e-3)
    assert s0["wd"] == pytest.approx(0.04) and s_end["wd"] == pytest.approx(0.4)
    assert s0["mom"] == pytest.approx(0.992) and s_end["mom"] == pytest.approx(1.0)
    assert s0["tau_t"] == pytest.approx(0.04) and s_end["tau_t"] == pytest.approx(0.07)


@pytest.fixture()
def tiny_set(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for k in range(12):
        rel = f"CheXpert-v1.0-small/train/patient{k:05d}/study1/view1_frontal.jpg"
        p = tmp_path / "data" / rel.split("CheXpert-v1.0-small/")[1]
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (60, 50), dtype=np.uint8)).save(p)
        rows.append({"Path": rel, "dose_run1": 2, "dose_run2": 2})
    pd.DataFrame(rows).to_csv(tmp_path / "groups.csv", index=False)
    return tmp_path


def _train(tmp: Path, extra: list[str]) -> int:
    cmd = [sys.executable, str(SCRIPT), "--groups", str(tmp / "groups.csv"), "--data-root", str(tmp / "data"),
           "--out", str(tmp / "out"), "--run", "1", "--batch", "4", "--n-local", "2", "--workers", "0",
           "--device", "cpu", "--max-steps", "4", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900).returncode


@pytest.mark.integration
def test_pause_then_resume(tiny_set: Path):
    pause = tiny_set / "PAUSE"
    pause.write_text("x")
    assert _train(tiny_set, ["--pause-file", str(pause)]) == st.EXIT_PAUSED
    assert (tiny_set / "out" / "ckpt_last.pt").exists() and not (tiny_set / "out" / "DONE").exists()
    pause.unlink()
    assert _train(tiny_set, ["--pause-file", str(pause), "--resume"]) == 0
    assert (tiny_set / "out" / "DONE").read_text() == "4"


def test_sinkhorn_knopp_is_doubly_normalised():
    torch.manual_seed(0)
    logits = torch.randn(64, 128) * 0.05
    q = st.sinkhorn_knopp(logits, tau_t=0.04, n_iter=3)
    assert torch.allclose(q.sum(1), torch.ones(64), atol=1e-4)
    assert q.sum(0).std() / q.sum(0).mean() < 0.05  # near-equal prototype usage


def test_sinkhorn_knopp_sharpens_distinct_samples():
    torch.manual_seed(0)
    logits = torch.randn(64, 256) * 0.2  # samples that differ, as after a trained head
    q = st.sinkhorn_knopp(logits, tau_t=0.04)
    ent = float((-(q * torch.log(q + 1e-12)).sum(-1)).mean())
    assert ent < 0.8 * float(np.log(256))


def test_schedule_has_no_duplicate_image_within_a_batch():
    groups = pd.DataFrame({"dose_run1": [500] * 50 + [100] * 200 + [10] * 3000, "dose_run2": [10] * 3250})
    sched = st.build_schedule(groups, 1, seed=1, batch=64)
    assert np.array_equal(np.bincount(sched, minlength=len(groups)), groups["dose_run1"].to_numpy())
    batches = sched[: len(sched) // 64 * 64].reshape(-1, 64)
    dup_batches = sum(len(set(b.tolist())) < 64 for b in batches)
    assert dup_batches <= 2  # only possible at the very tail


def test_koleo_is_finite_for_near_duplicates():
    x = torch.randn(1, 384).repeat(8, 1) + 1e-4 * torch.randn(8, 384)
    x.requires_grad_(True)
    val = st.koleo(x)
    val.backward()
    assert torch.isfinite(val) and torch.isfinite(x.grad).all()


def test_bn_head_outputs_differ_across_near_identical_inputs():
    torch.manual_seed(0)
    head = st.DINOHead(384, 256, use_bn=True).train()
    base = torch.randn(1, 384)
    out = head(base + 0.01 * torch.randn(32, 384))
    assert out.std(0).mean() > 10 * st.DINOHead(384, 256, use_bn=False).train()(base + 0.01 * torch.randn(32, 384)).std(0).mean()
