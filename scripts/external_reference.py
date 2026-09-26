"""Exploratory (not pre-registered): how good is the proxy encoder relative to RAD-DINO?

Linear probes (5 CheXpert competition labels) trained on 6,000 P_base images and tested on the CheXpert
official test set (frontal views; never seen by any of the four encoders). Encoders at their native
resolution: DINOv2-S step 0 and the Run-1 proxy at 224 px, DINOv2-B and RAD-DINO at 518 px.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torchvision import transforms
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
IMN = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
DEV = torch.device("mps")


def tf(size: int, norm) -> transforms.Compose:
    bic = transforms.InterpolationMode.BICUBIC
    return transforms.Compose([transforms.Resize(size, interpolation=bic), transforms.CenterCrop(size),
                               transforms.ToTensor(), transforms.Normalize(*norm)])


def load_models(proxy_ckpt: Path) -> dict:
    w = str(ROOT / "data" / "dinov2_s_lvd142m.safetensors")
    s0 = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224,
                           dynamic_img_size=True, pretrained_cfg_overlay={"file": w})
    px = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, img_size=224,
                           dynamic_img_size=True)
    px.load_state_dict(torch.load(proxy_ckpt, map_location="cpu", weights_only=False)["teacher"])
    return {"DINOv2-S (step 0)": (s0, tf(224, IMN), False), "Proxy after Run 1": (px, tf(224, IMN), False),
            "DINOv2-B": (AutoModel.from_pretrained("facebook/dinov2-base"), tf(518, IMN), True),
            "RAD-DINO": (AutoModel.from_pretrained("microsoft/rad-dino"), tf(518, ((0.5307,) * 3, (0.2583,) * 3)), True)}


@torch.no_grad()
def embed(model, hf: bool, imgs: list[Image.Image], transform) -> np.ndarray:
    model = model.to(DEV).eval()
    out = []
    for i in range(0, len(imgs), 32):
        x = torch.stack([transform(im) for im in imgs[i:i + 32]]).to(DEV)
        z = model(pixel_values=x).last_hidden_state[:, 0] if hf else model(x)
        out.append(z.float().cpu())
    model.to("cpu")
    return torch.cat(out).numpy()


def probe(xtr, ytr, gtr, xte, yte, pte, n_boot: int = 1000) -> dict:
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    preds = []
    for j in range(len(LABELS)):
        best_c, best = 0.1, -1
        for c in (0.01, 0.1, 1.0):
            s = [roc_auc_score(ytr[va, j], LogisticRegression(C=c, max_iter=3000).fit(xtr[tr], ytr[tr, j]).predict_proba(xtr[va])[:, 1])
                 for tr, va in GroupKFold(5).split(xtr, ytr[:, j], gtr)]
            if np.mean(s) > best:
                best_c, best = c, np.mean(s)
        preds.append(LogisticRegression(C=best_c, max_iter=3000).fit(xtr, ytr[:, j]).predict_proba(xte)[:, 1])
    p = np.column_stack(preds)
    per = [roc_auc_score(yte[:, j], p[:, j]) for j in range(len(LABELS))]
    rng, uniq = np.random.default_rng(0), np.unique(pte)
    idx = {u: np.flatnonzero(pte == u) for u in uniq}
    boots = []
    for _ in range(n_boot):
        b = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
        if all(0 < yte[b, j].sum() < len(b) for j in range(len(LABELS))):
            boots.append(np.mean([roc_auc_score(yte[b, j], p[b, j]) for j in range(len(LABELS))]))
    return {"macro_auc": float(np.mean(per)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "per_label": dict(zip(LABELS, map(float, per)))}


def main(proxy_ckpt: str) -> None:
    g = pd.read_csv(ROOT / "data" / "groups.csv")
    tr = g[g.group == "P_base"].sample(6000, random_state=0)
    z = zipfile.ZipFile(ROOT / "data" / "raw" / "chexpert_subset.zip")
    tr_imgs = [Image.open(io.BytesIO(z.read(p.split("CheXpert-v1.0-small/")[1]))).convert("RGB") for p in tr.Path]
    ytr = tr[[f"{l}__uzero" for l in LABELS]].to_numpy()
    te = pd.read_csv(ROOT / "data" / "chexlocalize" / "CheXpert" / "test_labels.csv")
    te = te[te.Path.str.contains("frontal")].reset_index(drop=True)
    base = ROOT / "data" / "chexlocalize" / "CheXpert"
    te_imgs = [Image.open(base / p).convert("RGB") for p in te.Path]
    yte = te[LABELS].to_numpy().astype(int)
    pte = te.Path.str.extract(r"patient(\d+)")[0].astype(int).to_numpy()
    res = {}
    for name, (m, transform, hf) in load_models(Path(proxy_ckpt)).items():
        xtr, xte = embed(m, hf, tr_imgs, transform), embed(m, hf, te_imgs, transform)
        res[name] = probe(xtr, ytr, tr.patient.to_numpy(), xte, yte, pte)
        print(f"{name:20s} macro-AUC {res[name]['macro_auc']:.4f} [{res[name]['ci95'][0]:.4f}, {res[name]['ci95'][1]:.4f}]", flush=True)
    out = ROOT / "results" / "external_reference.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n_train": len(tr), "n_test_images": len(te), "n_test_patients": int(len(np.unique(pte))), **res}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
