"""Amendment 6 (exploratory): a membership score from one backbone plus its released DINO head.

RAD-DINO released one backbone and one DINO head, so the score uses a single network in both roles. For each
image's K fixed augmented views: CE_ij = cross-entropy between softmax(h(v_i) / 0.07) and log-softmax(h(v_j) / 0.1),
and the score is minus the mean of CE_ij over i != j (higher = more self-consistent). No centring and no reference
model: no DINO head is published for DINOv2-B.

Modes:
  raddino  RAD-DINO on the Section 7 bridge images (same images, order and views), on the Mac;
           writes <out>/raddino_head.json and raddino_head_per_image.csv.
  proxy    teacher backbone + teacher head of a run's final snapshot on the test arms (views as in
           extract_features.py), on the laptop GPU; writes one .npz per run, recording the run number.
           Exits with code 3 and writes nothing when the pause file appears.
  analyze  membership AUC of each exposed arm vs H_b from the proxy scores, uncalibrated and, for arms whose
           dose differs between runs, calibrated by the partner run. Runs are matched by the stored run number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

T_TEACHER, T_STUDENT = 0.07, 0.1
TEST_ARMS = ("H_a", "H_b", "E3", "E15", "E100", "E500")
PAUSED = 3


def self_distill_score(logits: torch.Tensor, t_teacher: float = T_TEACHER, t_student: float = T_STUDENT) -> torch.Tensor:
    """logits: (B, K, P) from one network for K views of each image; returns (B,) = -mean_{i != j} CE(q_i, p_j)."""
    logits = logits.float()
    q = F.softmax(logits / t_teacher, dim=-1)
    logp = F.log_softmax(logits / t_student, dim=-1)
    ce = -torch.einsum("bip,bjp->bij", q, logp)
    off = ~torch.eye(logits.shape[1], dtype=torch.bool, device=logits.device)
    return -ce[:, off].mean(-1)


class ReleasedDinoHead(nn.Module):
    """DINOv2's DINOHead as stored in RAD-DINO's dino_head.safetensors: Linear-GELU-Linear-GELU-Linear, L2
    normalisation, then a weight-normalised linear layer saved as weight_g / weight_v. Sizes come from the tensors."""

    def __init__(self, sd: dict[str, torch.Tensor]) -> None:
        super().__init__()
        w0, w2, w4 = sd["mlp.0.weight"], sd["mlp.2.weight"], sd["mlp.4.weight"]
        self.mlp = nn.Sequential(nn.Linear(w0.shape[1], w0.shape[0]), nn.GELU(), nn.Linear(w2.shape[1], w2.shape[0]),
                                 nn.GELU(), nn.Linear(w4.shape[1], w4.shape[0]))
        self.mlp.load_state_dict({k[len("mlp."):]: v for k, v in sd.items() if k.startswith("mlp.")})
        v, g = sd["last_layer.weight_v"].float(), sd["last_layer.weight_g"].float()
        self.register_buffer("weight", g * v / v.norm(dim=1, keepdim=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1) @ self.weight.t()


# ----------------------------------------------------------------------------- RAD-DINO (Mac)

def lowpass(img: Image.Image, side: int | None) -> Image.Image:
    """Amendment 8: shorter side down to `side` pixels (Pillow's bicubic resize antialiases when shrinking), then
    back up to the original size, so only detail visible at `side`-pixel resolution remains. None = unchanged."""
    if side is None:
        return img
    w, h = img.size
    f = side / min(w, h)
    small = img.resize((max(1, round(w * f)), max(1, round(h * f))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def score_raddino(per_image_csv: Path, head_path: Path, out_dir: Path, device: torch.device,
                  lowpass_side: int | None = None, root: Path | None = None,
                  model_id: str = "microsoft/rad-dino") -> dict:
    """Amendment 6 score on the bridge images; with `lowpass_side`, the Amendment 8 low-pass check (same images,
    order and view seeds; outputs get a _lowpass<side> suffix). `root` re-bases the relative bridge paths."""
    from bridge_raddino import boot_auc, cls, mde_auc, raddino_preprocess, to_tensor, views_of
    from safetensors.torch import load_file
    from transformers import AutoModel

    df = pd.read_csv(per_image_csv)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    head = ReleasedDinoHead(load_file(str(head_path))).to(device).eval()
    scores = np.zeros(len(df))
    with torch.no_grad():
        for row, path in enumerate(df["path"]):  # row indexes the view seeds, exactly as in the bridge
            img = lowpass(raddino_preprocess(str(root / path) if root else path), lowpass_side)
            z = cls(model, to_tensor(views_of(img, row), "rad_dino"), device)
            scores[row] = float(self_distill_score(head(z.to(device)).unsqueeze(0))[0].cpu())
    y, pat = df["member"].to_numpy(), df["patient"].to_numpy()
    auc, ci = boot_auc(y, scores, pat)
    tag = f"_lowpass{lowpass_side}" if lowpass_side else ""
    what = "single-network DINO-head self-distillation (Amendment 6)"
    res = {"score": what + (f", low-passed to {lowpass_side} px (Amendment 8)" if lowpass_side else ""),
           "n_members": int(y.sum()), "n_nonmembers": int((1 - y).sum()), "mia_auc": auc, "mia_ci95": ci,
           "mde_auc": mde_auc(int(y.sum()), int((1 - y).sum()))}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"raddino_head{tag}.json").write_text(json.dumps(res, indent=2))
    df.assign(head_score=scores)[["path", "member", "patient", "head_score"]].to_csv(
        out_dir / f"raddino_head{tag}_per_image.csv", index=False)
    return res


# ----------------------------------------------------------------------------- Amendment 7 negative control

def score_bridge_with_proxies(per_image_csv: Path, snapshots: dict[str, Path], device: torch.device,
                              root: Path | None = None, pause_file: Path | None = None) -> pd.DataFrame | None:
    """The head score on the bridge images under proxy teachers (backbone + head) that saw neither pool: RAD-DINO's
    preprocessing, then the proxy's 224-pixel view transform with the bridge's per-image seeds. One column per
    snapshot; each image is preprocessed once. `root` re-bases the relative bridge paths (laptop copy)."""
    from bridge_raddino import raddino_preprocess
    from extract_features import K_VIEWS, VIEW_SEED, make_encoder, make_heads, view_transform

    df = pd.read_csv(per_image_csv)
    nets = {name: (make_encoder(snap, device), make_heads(snap, device)[0]) for name, snap in snapshots.items()}
    tf = view_transform(224)
    scores = {name: np.zeros(len(df)) for name in nets}
    with torch.no_grad():
        for row, path in enumerate(df["path"]):
            if pause_file and pause_file.exists():
                return None
            img = raddino_preprocess(str(root / path) if root else path)
            views = []
            for k in range(K_VIEWS):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(VIEW_SEED * 10_000_019 + row * K_VIEWS + k)
                    views.append(tf(img))
            x = torch.stack(views).to(device)
            for name, (model, head) in nets.items():
                scores[name][row] = float(self_distill_score(head(model(x).float()).unsqueeze(0))[0].cpu())
    out = df[["path", "member", "patient"]].copy()
    for name in nets:
        out[f"head_{name}"] = scores[name]
    return out


# ----------------------------------------------------------------------------- proxy (laptop GPU)

def score_proxy_run(run_dir: Path, run: int, groups: pd.DataFrame, data_root: Path, out: Path, device: torch.device,
                    workers: int = 6, batch: int = 32, pause_file: Path | None = None) -> int:
    from extract_features import ViewSet, make_encoder, make_heads
    from torch.utils.data import DataLoader

    snap = max(run_dir.glob("eval_step*.pt"), key=lambda p: int(p.stem.split("step")[1]))
    arms = groups[groups["group"].isin(TEST_ARMS)]
    model, head = make_encoder(snap, device), make_heads(snap, device)[0]  # teacher backbone, teacher head
    amp = device.type == "cuda"
    loader = DataLoader(ViewSet(arms["Path"].tolist(), arms.index.tolist(), data_root), batch_size=batch,
                        num_workers=workers, pin_memory=amp)
    scores = []
    with torch.no_grad():
        for x in loader:  # (B, K, 3, H, W)
            if pause_file and pause_file.exists():
                return PAUSED
            b, k = x.shape[:2]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = head(model(x.flatten(0, 1).to(device, non_blocking=True)).float())
            scores.append(self_distill_score(logits.float().view(b, k, -1)).cpu())
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, rows=arms.index.to_numpy(), score=torch.cat(scores).numpy(), step=int(snap.stem.split("step")[1]),
             run=run)
    os.replace(tmp, out)
    return 0


def proxy_mia(scores: dict[int, pd.Series], groups: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score

    from analyze import tpr_at_fpr

    rows = []
    for run, s in sorted(scores.items()):
        dose = groups.groupby("group")[f"dose_run{run}"].first()
        other = groups.groupby("group")[f"dose_run{3 - run}"].first()
        for arm in [a for a in TEST_ARMS if dose.get(a, 0) > 0]:
            variants = [("none", s)]
            if (3 - run) in scores and other.get(arm) != dose[arm]:
                variants.append(("partner_run", (s - scores[3 - run]).dropna()))
            for calibration, sc in variants:
                g = groups.loc[sc.index, "group"]
                pos, neg = sc[g == arm].to_numpy(), sc[g == "H_b"].to_numpy()
                y, v = np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]
                rows.append({"run": run, "arm": arm, "dose": int(dose[arm]), "calibration": calibration,
                             "mia_auc": roc_auc_score(y, v), "tpr_at_1pct_fpr": tpr_at_fpr(y, v),
                             "p_mwu": float(mannwhitneyu(pos, neg, alternative="greater").pvalue),
                             "n_pos": len(pos), "n_neg": len(neg)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("raddino")
    r.add_argument("--per-image", type=Path, required=True, help="results/bridge/bridge_per_image.csv")
    r.add_argument("--head", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--device", default="mps")
    r.add_argument("--lowpass", type=int, default=None, help="Amendment 8: low-pass to this shorter side first")
    r.add_argument("--root", type=Path, default=None)
    r.add_argument("--model", default="microsoft/rad-dino", help="hub id or local folder with RAD-DINO weights")
    p = sub.add_parser("proxy")
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--run", type=int, choices=(1, 2), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--pause-file", type=Path, default=None)
    c = sub.add_parser("control", help="Amendment 7 negative control on the bridge images")
    c.add_argument("--per-image", type=Path, required=True)
    c.add_argument("--snapshot", action="append", required=True, help="NAME=PATH, repeatable")
    c.add_argument("--root", type=Path, default=None)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--device", default="cuda")
    c.add_argument("--pause-file", type=Path, default=None)
    a = sub.add_parser("analyze")
    a.add_argument("--groups", type=Path, required=True)
    a.add_argument("--scores", type=Path, nargs=2, required=True, help="the two .npz files from proxy mode")
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.mode == "raddino":
        print(json.dumps(score_raddino(args.per_image, args.head, args.out, torch.device(args.device), args.lowpass,
                                       args.root, args.model), indent=2))
        return 0
    if args.mode == "control":
        snaps = dict(item.split("=", 1) for item in args.snapshot)
        res = score_bridge_with_proxies(args.per_image, {k: Path(v) for k, v in snaps.items()},
                                        torch.device(args.device), args.root, args.pause_file)
        if res is None:
            return PAUSED
        tmp = args.out.with_name(args.out.name + ".tmp")
        res.to_csv(tmp, index=False)
        os.replace(tmp, args.out)
        return 0
    groups = pd.read_csv(args.groups)
    if args.mode == "proxy":
        if args.pause_file and args.pause_file.exists():
            return PAUSED
        return score_proxy_run(args.run_dir, args.run, groups, args.data_root, args.out, torch.device(args.device),
                               args.workers, pause_file=args.pause_file)
    scores = {}
    for f in args.scores:
        d = np.load(f)
        scores[int(d["run"])] = pd.Series(d["score"], index=d["rows"])
    if sorted(scores) != [1, 2]:
        raise SystemExit(f"need one score file per run, got runs {[int(np.load(f)['run']) for f in args.scores]}")
    res = proxy_mia(scores, groups)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
