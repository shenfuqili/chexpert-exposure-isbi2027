"""Bridge to the released RAD-DINO (PREREG.md Sec. 7): members = CheXpert valid, non-members = CheXpert test.

Steps: (1) preprocess every full-resolution image exactly like RAD-DINO's pipeline (B-spline resize so the
shorter side is 518, min-max scaling to [0, 255]); (2) exchangeability gate: can a classifier on DINOv2-B
features (the network RAD-DINO started from) tell the two pools apart? (3) membership score: augmentation
invariance under RAD-DINO minus the same score under DINOv2-B, on identical fixed augmented views;
(4) membership AUC with a patient-clustered bootstrap CI and the minimal detectable effect.
"""
from __future__ import annotations

import argparse
import json
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from torchvision import transforms
from transformers import AutoModel

SHORT_SIDE, K_VIEWS, VIEW_SEED, N_BOOT = 518, 8, 777, 10_000
MODELS = {"rad_dino": "microsoft/rad-dino", "dinov2_b": "facebook/dinov2-base"}
NORM = {"rad_dino": ((0.5307,) * 3, (0.2583,) * 3), "dinov2_b": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}


def raddino_preprocess(path: str) -> Image.Image:
    """Shorter side to 518 with cubic B-spline interpolation, then min-max to [0, 255] (RAD-DINO model card)."""
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    factor = SHORT_SIDE / min(arr.shape)
    arr = zoom(arr, factor, order=3)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6) * 255.0
    return Image.fromarray(arr.round().clip(0, 255).astype(np.uint8)).convert("RGB")


def views_of(img: Image.Image, row: int) -> list[Image.Image]:
    bic = transforms.InterpolationMode.BICUBIC
    tf = transforms.Compose([
        transforms.RandomResizedCrop(SHORT_SIDE, (0.5, 1.0), interpolation=bic), transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5)])
    out = []
    for k in range(K_VIEWS):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(VIEW_SEED * 10_000_019 + row * K_VIEWS + k)
            out.append(tf(img))
    return out


def centre_view(img: Image.Image) -> Image.Image:
    return transforms.CenterCrop(SHORT_SIDE)(img)


def to_tensor(imgs: list[Image.Image], model_key: str) -> torch.Tensor:
    mean, std = NORM[model_key]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return torch.stack([tf(i) for i in imgs])


@torch.no_grad()
def cls(model, x: torch.Tensor, device) -> torch.Tensor:
    return model(pixel_values=x.to(device)).last_hidden_state[:, 0].float().cpu()


def patient_of(path: str) -> int:
    m = re.search(r"patient(\d+)", path)
    return int(m.group(1)) if m else -1


def boot_auc(y: np.ndarray, s: np.ndarray, pat: np.ndarray, n: int = N_BOOT, seed: int = 0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(pat)
    idx_of = {p: np.flatnonzero(pat == p) for p in uniq}
    draws = []
    for _ in range(n):
        pick = np.concatenate([idx_of[p] for p in rng.choice(uniq, len(uniq))])
        if 0 < y[pick].sum() < len(pick):
            draws.append(roc_auc_score(y[pick], s[pick]))
    return float(roc_auc_score(y, s)), [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def mde_auc(n_pos: int, n_neg: int) -> float:
    """AUC detectable with 80% power at two-sided alpha 0.05 (null SE of the Mann-Whitney AUC)."""
    se0 = np.sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))
    return float(0.5 + (1.96 + 0.84) * se0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members-glob", required=True)
    ap.add_argument("--nonmembers-glob", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob(a.members_glob, recursive=True)) + sorted(glob(a.nonmembers_glob, recursive=True))
    n_mem = len(glob(a.members_glob, recursive=True))
    y = np.r_[np.ones(n_mem), np.zeros(len(paths) - n_mem)]
    pat = np.array([patient_of(p) for p in paths])
    device = torch.device(a.device)
    models = {k: AutoModel.from_pretrained(v).to(device).eval() for k, v in MODELS.items()}
    inv = {k: np.zeros(len(paths)) for k in models}
    centre_feats, sizes = [], []
    for row, path in enumerate(paths):
        img = raddino_preprocess(path)
        sizes.append(Image.open(path).size)
        views = views_of(img, row)
        for key, model in models.items():
            z = F.normalize(cls(model, to_tensor(views, key), device), dim=-1)
            sim = z @ z.t()
            inv[key][row] = float((sim.sum() - K_VIEWS) / (K_VIEWS * (K_VIEWS - 1)))
        centre_feats.append(cls(models["dinov2_b"], to_tensor([centre_view(img)], "dinov2_b"), device)[0].numpy())
    feats = np.stack(centre_feats)
    gate_p = cross_val_predict(LogisticRegression(C=0.1, max_iter=5000), feats, y, groups=pat,
                               cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
    gate_auc, gate_ci = boot_auc(y, gate_p, pat)
    calibrated = inv["rad_dino"] - inv["dinov2_b"]
    mia_auc, mia_ci = boot_auc(y, calibrated, pat)
    raw_auc, raw_ci = boot_auc(y, inv["rad_dino"], pat)
    result = {"n_members": int(y.sum()), "n_nonmembers": int((1 - y).sum()),
              "gate_auc": gate_auc, "gate_ci95": gate_ci, "gate_passed": bool(gate_auc < 0.55),
              "mia_auc_calibrated": mia_auc, "mia_ci95": mia_ci, "mia_auc_raw_raddino": raw_auc, "raw_ci95": raw_ci,
              "mde_auc": mde_auc(int(y.sum()), int((1 - y).sum()))}
    (a.out / "bridge.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame({"path": paths, "member": y, "patient": pat, "width": [s[0] for s in sizes],
                  "height": [s[1] for s in sizes], "inv_raddino": inv["rad_dino"], "inv_dinov2b": inv["dinov2_b"],
                  "gate_prob": gate_p}).to_csv(a.out / "bridge_per_image.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
