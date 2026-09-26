"""Frozen-teacher features and membership scores for every evaluation snapshot (PREREG.md Sec. 4-5).

For each eval_step*.pt in a run directory, writes feats_<snapshot>.npz with
  - feats: teacher CLS features (float16) for all evaluation groups, deterministic centre crop
  - inv:   augmentation invariance = mean pairwise cosine similarity among K fixed augmented views,
           for the test arms only (H_a, H_b, E3, E15, E100, E500)
Augmented views use a per-image seed, so every encoder sees exactly the same K views; this is what
makes reference-calibrated membership scores (score under model minus score under reference) valid.
Snapshots already processed are skipped, so the job can be killed and restarted at any time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

EVAL_GROUPS = ("P_base", "P_clean", "H_a", "H_b", "E3", "E15", "E100", "E500")
TEST_ARMS = ("H_a", "H_b", "E3", "E15", "E100", "E500")
K_VIEWS = 8
VIEW_SEED = 777
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def eval_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def view_transform(size: int = 224):
    """Global-crop augmentation family used in training (RandomResizedCrop 0.5-1, flip, colour, blur)."""
    bic = transforms.InterpolationMode.BICUBIC
    return transforms.Compose([
        transforms.RandomResizedCrop(size, (0.5, 1.0), interpolation=bic), transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_image(root: Path, rel_path: str) -> Image.Image:
    rel = rel_path.split("CheXpert-v1.0-small/", 1)[-1]
    with Image.open(root / rel) as im:
        return im.convert("RGB")


class CenterSet(Dataset):
    def __init__(self, paths: list[str], root: Path) -> None:
        self.paths, self.root, self.tf = paths, root, eval_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.tf(load_image(self.root, self.paths[i]))


class ViewSet(Dataset):
    """K augmented views per image; view k of row r is generated from seed (VIEW_SEED, r, k)."""

    def __init__(self, paths: list[str], rows: list[int], root: Path) -> None:
        self.paths, self.rows, self.root, self.tf = paths, rows, root, view_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        img = load_image(self.root, self.paths[i])
        views = []
        for k in range(K_VIEWS):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(VIEW_SEED * 10_000_019 + self.rows[i] * K_VIEWS + k)
                views.append(self.tf(img))
        return torch.stack(views)


def make_encoder(snapshot: Path, device: torch.device, key: str = "teacher") -> torch.nn.Module:
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0,
                              img_size=224, dynamic_img_size=True)
    state = torch.load(snapshot, map_location="cpu", weights_only=False)[key]
    model.load_state_dict(state)
    return model.to(device).eval()


def make_heads(snapshot: Path, device: torch.device):
    """Teacher and student DINO heads from a snapshot (eval mode, BatchNorm running statistics)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from ssl_train import DINOHead
    st = torch.load(snapshot, map_location="cpu", weights_only=False)
    out_dim = st["teacher_head"]["last.parametrizations.weight.original1"].shape[0]
    use_bn = any(".running_mean" in k for k in st["teacher_head"])
    heads = []
    for key in ("teacher_head", "student_head"):
        h = DINOHead(384, out_dim, use_bn=use_bn)
        h.load_state_dict(st[key])
        heads.append(h.to(device).eval())
    return heads


@torch.no_grad()
def embed(model, loader, device, amp: bool) -> np.ndarray:
    out = []
    for x in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def invariance(model, loader, device, amp: bool, student=None, heads=None) -> tuple[np.ndarray, np.ndarray]:
    """Augmentation invariance (teacher CLS) and, if heads are given, the DINO cross-entropy between the
    teacher's view-i target (softmax at tau 0.07) and the student's view-j prediction (tau 0.1), i != j."""
    scores, ces = [], []
    for x in loader:  # (B, K, 3, H, W)
        b = x.shape[0]
        xs = x.flatten(0, 1).to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            z = model(xs).float()
            if heads is not None:
                t_logit = heads[0](z).float().view(b, K_VIEWS, -1)
                s_logit = heads[1](student(xs).float()).float().view(b, K_VIEWS, -1)
        zn = F.normalize(z, dim=-1).view(b, K_VIEWS, -1)
        sim = zn @ zn.transpose(1, 2)
        scores.append(((sim.sum(dim=(1, 2)) - K_VIEWS) / (K_VIEWS * (K_VIEWS - 1))).cpu())
        if heads is not None:
            q = F.softmax(t_logit / 0.07, dim=-1)
            logp = F.log_softmax(s_logit / 0.1, dim=-1)
            ce = -(q.unsqueeze(2) * logp.unsqueeze(1)).sum(-1)  # (B, K_teacher, K_student)
            mask = ~torch.eye(K_VIEWS, dtype=torch.bool, device=ce.device)
            ces.append(ce[:, mask].mean(-1).cpu())
    return torch.cat(scores).numpy(), (torch.cat(ces).numpy() if ces else np.zeros(0))


def process(snapshot: Path, groups: pd.DataFrame, root: Path, out: Path, workers: int, device) -> None:
    amp = device.type == "cuda"
    ev = groups[groups["group"].isin(EVAL_GROUPS)]
    arms = groups[groups["group"].isin(TEST_ARMS)]
    model = make_encoder(snapshot, device)
    student = make_encoder(snapshot, device, key="student")
    heads = make_heads(snapshot, device)
    kw = dict(num_workers=workers, pin_memory=amp, persistent_workers=False)
    feats = embed(model, DataLoader(CenterSet(ev["Path"].tolist(), root), batch_size=256, **kw), device, amp)
    inv, ce = invariance(model, DataLoader(ViewSet(arms["Path"].tolist(), arms.index.tolist(), root),
                                           batch_size=32, **kw), device, amp, student=student, heads=heads)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, rows=ev.index.to_numpy(), feats=feats.astype(np.float16),
             inv_rows=arms.index.to_numpy(), inv=inv.astype(np.float32), dino_ce=ce.astype(np.float32))
    tmp.replace(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--done-marker", type=Path, default=None)
    ap.add_argument("--pause-file", type=Path, default=None)
    a = ap.parse_args()
    device = torch.device(a.device)
    groups = pd.read_csv(a.groups)
    snaps = sorted(a.run_dir.glob("eval_step*.pt"))
    if not snaps:
        print("no snapshots found", file=sys.stderr)
        return 1
    for snap in snaps:
        out = a.run_dir / f"feats_{snap.stem}.npz"
        if out.exists():
            continue
        if a.pause_file and a.pause_file.exists():
            print("[paused]", flush=True)
            return 3
        print(f"processing {snap.name}", flush=True)
        process(snap, groups, a.data_root, out, a.workers, device)
    if a.done_marker:
        a.done_marker.write_text(str(len(snaps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
