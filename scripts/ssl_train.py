"""Continued DINO pretraining of DINOv2-S/14 on CheXpert with dosed exposure arms (PREREG.md Sec. 3).

Built for a 6 GB laptop GPU: bf16 autocast, gradient checkpointing, a resumable checkpoint every
few minutes, and a pause file that makes the run save and exit cleanly (used by watchdog.py so
training steps aside when the owner starts a game).

Hyperparameters follow RAD-DINO's released config scaled to ViT-S: lr = 1e-3 * sqrt(batch/1024),
layer-wise decay 0.9, patch-embed lr x0.2, weight decay 0.04 -> 0.4, teacher momentum
0.992 -> 1, teacher temperature 0.04 -> 0.07 over the first half, KoLeo 0.1, clip 3.0, no iBOT.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

EXIT_PAUSED = 3
BASE_DOSE = 10  # views of background and P_base images over the whole schedule
EVAL_FRACTIONS = (0.35, 1.0)  # schedule points saved for evaluation (RAD-DINO's checkpoint sits at 35%)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Config:
    groups_csv: Path
    data_root: Path
    out_dir: Path
    run: int
    batch: int = 32
    n_local: int = 6
    global_size: int = 224
    local_size: int = 98
    out_dim: int = 65536
    workers: int = 6
    seed: int = 1
    ckpt_minutes: float = 20.0
    pause_file: Path | None = None
    bench_steps: int = 0
    max_steps: int = 0
    head_warmup_frac: float = 0.10
    head_lr_mult: float = 10.0
    lr_mult: float = 1.0
    drop_path: float = 0.1
    device: str = "cuda"
    init_weights: Path | None = None
    grad_ckpt: bool = True
    centering: str = "sk"
    koleo_weight: float = 0.1


# ----------------------------------------------------------------------------- schedule

def build_schedule(groups: pd.DataFrame, run: int, seed: int, batch: int = 64) -> np.ndarray:
    """Every image index repeated by its dose for this run, shuffled, with no image twice in one batch.

    Duplicates within a batch would let KoLeo push two views of the same exposed image apart, which does
    not happen at RAD-DINO's scale. They are removed by swapping with a later position (deterministic).
    """
    dose = groups[f"dose_run{run}"].to_numpy()
    rng = np.random.default_rng(seed * 1000 + run)
    idx = rng.permutation(np.repeat(np.arange(len(groups)), dose)).astype(np.int32)
    n = len(idx)
    for start in range(0, n - batch + 1, batch):
        seen = set()
        for pos in range(start, start + batch):
            if idx[pos] in seen:
                j = pos + batch
                while j < n and (idx[j] in seen or idx[j] == idx[pos]):
                    j += 1
                if j < n:
                    idx[pos], idx[j] = idx[j], idx[pos]
            seen.add(int(idx[pos]))
    return idx


class ScheduleSampler(Sampler[int]):
    """Yields the precomputed schedule from a given offset (for resuming)."""

    def __init__(self, schedule: np.ndarray, start: int) -> None:
        self.schedule, self.start = schedule, start

    def __iter__(self):
        return iter(self.schedule[self.start:].tolist())

    def __len__(self) -> int:
        return len(self.schedule) - self.start


# ----------------------------------------------------------------------------- data

class GaussianBlur(transforms.RandomApply):
    def __init__(self, p: float) -> None:
        super().__init__([transforms.GaussianBlur(9, sigma=(0.1, 2.0))], p=p)


class MultiCrop:
    """DINOv2 multi-crop as in RAD-DINO's augmentations.py (no solarization). Picklable for Windows workers."""

    def __init__(self, global_size: int, local_size: int, n_local: int) -> None:
        norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        color = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        bic = transforms.InterpolationMode.BICUBIC
        self.geo_g = transforms.Compose([transforms.RandomResizedCrop(global_size, (0.5, 1.0), interpolation=bic),
                                         transforms.RandomHorizontalFlip()])
        self.geo_l = transforms.Compose([transforms.RandomResizedCrop(local_size, (0.2, 0.5), interpolation=bic),
                                         transforms.RandomHorizontalFlip()])
        self.g1 = transforms.Compose([color, GaussianBlur(0.5), norm])
        self.g2 = transforms.Compose([color, GaussianBlur(0.1), norm])
        self.loc = transforms.Compose([color, GaussianBlur(0.5), norm])
        self.n_local = n_local

    def __call__(self, img: Image.Image):
        globals_ = [self.g1(self.geo_g(img)), self.g2(self.geo_g(img))]
        locals_ = [self.loc(self.geo_l(img)) for _ in range(self.n_local)]
        return globals_, locals_


class CXRDataset(Dataset):
    def __init__(self, paths: list[str], data_root: Path, transform) -> None:
        self.paths, self.root, self.transform = paths, data_root, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        rel = self.paths[i].split("CheXpert-v1.0-small/", 1)[-1]
        with Image.open(self.root / rel) as im:
            img = im.convert("RGB")
        return self.transform(img)


def collate(batch):
    """Crop-major stacking: [crop0 of every image, crop1 of every image, ...] so .chunk(n_crops) splits by crop."""
    n_g, n_l = len(batch[0][0]), len(batch[0][1])
    g = torch.cat([torch.stack([item[0][c] for item in batch]) for c in range(n_g)])
    loc = torch.cat([torch.stack([item[1][c] for item in batch]) for c in range(n_l)])
    return g, loc


# ----------------------------------------------------------------------------- model

class DINOHead(nn.Module):
    """DINO projection head. use_bn adds BatchNorm after the hidden layers (DINO's use_bn_in_head option);
    it removes the component shared by all images in a batch, which near-identical chest X-ray features need."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 2048, bottleneck: int = 256, use_bn: bool = True) -> None:
        super().__init__()
        norm = (lambda d: nn.BatchNorm1d(d)) if use_bn else (lambda d: nn.Identity())
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), norm(hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), norm(hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1.0)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.last(F.normalize(self.mlp(x), dim=-1))


def make_backbone(drop_path: float, weights: Path | None = None) -> nn.Module:
    """DINOv2-S/14 at 224 px with dynamic image size; `weights` points to a local safetensors copy if given."""
    overlay = {"file": str(weights)} if weights else None
    return timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                             img_size=224, dynamic_img_size=True, drop_path_rate=drop_path,
                             pretrained_cfg_overlay=overlay)


def param_groups(backbone: nn.Module, head: nn.Module, layer_decay: float = 0.9):
    n_blocks = len(backbone.blocks)
    groups = []
    for name, p in backbone.named_parameters():
        if name.startswith("blocks."):
            depth = int(name.split(".")[1]) + 1
        elif name.startswith(("norm", "fc_norm")):
            depth = n_blocks + 1
        else:
            depth = 0
        scale = layer_decay ** (n_blocks + 1 - depth)
        if name.startswith("patch_embed"):
            scale *= 0.2
        no_wd = p.ndim == 1 or name in ("cls_token", "pos_embed", "reg_token", "mask_token")
        groups.append({"params": [p], "lr_scale": scale, "wd_scale": 0.0 if no_wd else 1.0, "is_backbone": True})
    for name, p in head.named_parameters():
        if p.requires_grad:
            groups.append({"params": [p], "lr_scale": 1.0, "wd_scale": 0.0 if p.ndim == 1 else 1.0,
                           "is_backbone": False, "is_last": name.startswith("last")})
    return groups


# ----------------------------------------------------------------------------- loss

def dino_loss(student_out: torch.Tensor, teacher_probs: list[torch.Tensor], n_crops: int, tau_s: float):
    s_chunks = (student_out / tau_s).chunk(n_crops)
    total, n_terms = 0.0, 0
    for iq, q in enumerate(teacher_probs):
        for v, s in enumerate(s_chunks):
            if v == iq:
                continue
            total = total + torch.sum(-q * F.log_softmax(s, dim=-1), dim=-1).mean()
            n_terms += 1
    return total / n_terms


@torch.no_grad()
def sinkhorn_knopp(teacher_logits: torch.Tensor, tau_t: float, n_iter: int = 3) -> torch.Tensor:
    """DINOv2's Sinkhorn-Knopp teacher targets: equipartition over prototypes within the batch."""
    q = torch.exp((teacher_logits.float() - teacher_logits.float().max()) / tau_t).t()  # K x B
    k, b = q.shape
    q /= q.sum()
    for _ in range(n_iter):
        q /= q.sum(dim=1, keepdim=True); q /= k
        q /= q.sum(dim=0, keepdim=True); q /= b
    return (q * b).t()


@torch.no_grad()
def diagnostics(t_logits: torch.Tensor, probs: torch.Tensor, s_out: torch.Tensor, s_cls: torch.Tensor) -> dict:
    """Collapse monitors: teacher/student entropies, teacher logit spread, mean pairwise CLS cosine."""
    ent = lambda p: float((-(p * torch.log(p + 1e-12)).sum(-1)).mean())
    z = F.normalize(s_cls.float(), dim=-1)
    n = z.shape[0]
    cos = (z @ z.t()).sum() - n
    return {"t_ent": ent(probs), "t_std": float(t_logits.std(-1).mean()), "s_ent": ent(F.softmax(s_out / 0.1, -1)),
            "cls_cos": float(cos / (n * (n - 1)))}


def koleo(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KoLeo regularizer (DINOv2 form) computed in fp32 with autocast off, so near-duplicate pairs stay finite."""
    with torch.autocast(x.device.type, enabled=False):
        x = F.normalize(x.float(), eps=eps, dim=-1)
        dots = x @ x.t()
        dots.fill_diagonal_(-1.0)
        nn_idx = dots.argmax(dim=1)
        dist = F.pairwise_distance(x, x[nn_idx], eps=eps)
        return -torch.log(dist + eps).mean()


def cosine(step: int, total: int, start: float, end: float) -> float:
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(step, total) / max(total, 1)))


def schedules(step: int, total: int, batch: int) -> dict:
    """Per-step lr, weight decay, teacher momentum and teacher temperature (RAD-DINO config, sqrt lr scaling)."""
    peak = 1e-3 * math.sqrt(batch / 1024)
    warm = max(1, int(0.05 * total))
    lr = peak * step / warm if step < warm else cosine(step - warm, total - warm, peak, 1e-6)
    half = max(1, total // 2)
    return {
        "lr": lr,
        "wd": cosine(step, total, 0.04, 0.4),
        "mom": cosine(step, total, 0.992, 1.0),
        "tau_t": 0.04 + (0.07 - 0.04) * min(step, half) / half,
    }


# ----------------------------------------------------------------------------- checkpointing

def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_eval_snapshot(state: dict, out_dir: Path, step: int) -> None:
    snap = {k: state[k] for k in ("teacher", "teacher_head", "student", "student_head", "center")}
    save_atomic({**snap, "step": step}, out_dir / f"eval_step{step:07d}.pt")


# ----------------------------------------------------------------------------- training

class Trainer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.dev = torch.device(cfg.device)
        groups = pd.read_csv(cfg.groups_csv)
        self.paths = groups["Path"].tolist()
        self.schedule = build_schedule(groups, cfg.run, cfg.seed, cfg.batch)
        self.total = len(self.schedule) // cfg.batch
        if cfg.max_steps:
            self.total = min(self.total, cfg.max_steps)
        self.eval_steps = sorted({int(round(f * self.total)) for f in EVAL_FRACTIONS})
        self._build_models()

    def _build_models(self) -> None:
        cfg = self.cfg
        self.student = make_backbone(cfg.drop_path, cfg.init_weights).to(self.dev)
        self.teacher = make_backbone(0.0, cfg.init_weights).to(self.dev)
        self.teacher.load_state_dict(self.student.state_dict())
        self.student.set_grad_checkpointing(cfg.grad_ckpt)
        dim = self.student.num_features
        self.s_head = DINOHead(dim, cfg.out_dim).to(self.dev)
        self.t_head = DINOHead(dim, cfg.out_dim).to(self.dev)
        self.t_head.load_state_dict(self.s_head.state_dict())
        for p in list(self.teacher.parameters()) + list(self.t_head.parameters()):
            p.requires_grad = False
        self.opt = torch.optim.AdamW(param_groups(self.student, self.s_head), lr=0.0, weight_decay=0.0)
        self.center = torch.zeros(1, cfg.out_dim, device=self.dev)
        self.step = 0
        self.skipped = 0

    def state(self) -> dict:
        return {"teacher": self.teacher.state_dict(), "teacher_head": self.t_head.state_dict(),
                "student": self.student.state_dict(), "student_head": self.s_head.state_dict(),
                "center": self.center, "opt": self.opt.state_dict(), "step": self.step,
                "rng_torch": torch.get_rng_state(), "config": {k: str(v) for k, v in vars(self.cfg).items()}}

    def load(self, path: Path) -> None:
        st = torch.load(path, map_location=self.dev, weights_only=False)
        self.teacher.load_state_dict(st["teacher"]); self.t_head.load_state_dict(st["teacher_head"])
        self.student.load_state_dict(st["student"]); self.s_head.load_state_dict(st["student_head"])
        self.center, self.step = st["center"], st["step"]
        self.opt.load_state_dict(st["opt"]); torch.set_rng_state(st["rng_torch"].cpu())

    def loader(self) -> DataLoader:
        cfg = self.cfg
        ds = CXRDataset(self.paths, cfg.data_root, MultiCrop(cfg.global_size, cfg.local_size, cfg.n_local))
        sampler = ScheduleSampler(self.schedule, self.step * cfg.batch)
        return DataLoader(ds, batch_size=cfg.batch, sampler=sampler, num_workers=cfg.workers, collate_fn=collate,
                          drop_last=True, pin_memory=self.dev.type == "cuda", persistent_workers=cfg.workers > 0)

    def _set_hparams(self, sch: dict) -> None:
        """Head warm-up: backbone frozen and the new head trained at head_lr_mult x lr, then joint training."""
        warm = self.step < int(self.cfg.head_warmup_frac * self.total)
        head_lr = self.cfg.head_lr_mult * max(sch["lr"], 1e-3 * math.sqrt(self.cfg.batch / 1024) * 0.1) if warm else sch["lr"]
        for g in self.opt.param_groups:
            if g["is_backbone"]:
                g["lr"] = 0.0 if warm else sch["lr"] * g["lr_scale"]
            else:
                g["lr"] = head_lr * g["lr_scale"]
            g["weight_decay"] = sch["wd"] * g["wd_scale"]

    def train_step(self, g: torch.Tensor, loc: torch.Tensor) -> dict:
        cfg, sch = self.cfg, schedules(self.step, self.total, self.cfg.batch)
        sch = {**sch, "lr": sch["lr"] * cfg.lr_mult}
        self._set_hparams(sch)
        g, loc = g.to(self.dev, non_blocking=True), loc.to(self.dev, non_blocking=True)
        n_crops = 2 + cfg.n_local
        with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            with torch.no_grad():
                t_out = self.t_head(self.teacher(g)).float()
                if cfg.centering == "sk":
                    probs_all = sinkhorn_knopp(t_out, sch["tau_t"])
                else:
                    probs_all = F.softmax((t_out - self.center) / sch["tau_t"], dim=-1)
                probs = probs_all.chunk(2)
            s_g = self.student(g)
            s_l = self.student(loc)
            s_out = self.s_head(torch.cat([s_g, s_l])).float()
            dino = dino_loss(s_out, list(probs), n_crops, tau_s=0.1)
            ko = sum(koleo(c) for c in s_g.chunk(2))
            loss = dino + cfg.koleo_weight * ko
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.step < max(1, int(0.01 * self.total)):  # freeze the last head layer early, as DINOv2 does
            for p in self.s_head.last.parameters():
                p.grad = None
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in list(self.student.parameters()) + list(self.s_head.parameters()) if p.requires_grad], 3.0)
        if not (torch.isfinite(loss) and torch.isfinite(gnorm)):
            self.opt.zero_grad(set_to_none=True)
            self.skipped += 1
            return {"loss": float("nan"), "skipped": self.skipped, **sch}
        self.opt.step()
        with torch.no_grad():
            m = sch["mom"]
            for ps, pt in zip(list(self.student.parameters()) + list(self.s_head.parameters()),
                              list(self.teacher.parameters()) + list(self.t_head.parameters())):
                pt.mul_(m).add_(ps.detach(), alpha=1 - m)
            self.center.mul_(0.9).add_(t_out.mean(dim=0, keepdim=True), alpha=0.1)
        diag = diagnostics(t_out, probs_all, s_out.detach(), s_g.detach()) if (self.step + 1) % 50 == 0 else {}
        return {"loss": float(loss.detach()), "dino": float(dino.detach()), "koleo": float(ko.detach()), **diag, **sch}

    def run(self, resume: bool) -> int:
        cfg, out = self.cfg, self.cfg.out_dir
        out.mkdir(parents=True, exist_ok=True)
        last = out / "ckpt_last.pt"
        if resume and last.exists():
            self.load(last)
        if self.step == 0 and not (out / "eval_step0000000.pt").exists() and not cfg.bench_steps:
            save_eval_snapshot(self.state(), out, 0)
        log = open(out / "train_log.jsonl", "a", encoding="utf-8")
        t_ckpt, t0, seen0 = time.time(), time.time(), self.step
        for g, loc in self.loader():
            if self.step >= self.total:
                break
            if cfg.pause_file and self.step % 10 == 0 and cfg.pause_file.exists():
                save_atomic(self.state(), last)
                print(f"[paused] step {self.step}", flush=True)
                return EXIT_PAUSED
            info = self.train_step(g, loc)
            self.step += 1
            if self.step in self.eval_steps:
                save_eval_snapshot(self.state(), out, self.step)
            if self.step % 50 == 0 or cfg.bench_steps:
                ips = (self.step - seen0) * cfg.batch / (time.time() - t0)
                mem = torch.cuda.max_memory_allocated() / 2**30 if self.dev.type == "cuda" else 0.0
                rec = {"step": self.step, "total": self.total, "img_per_s": round(ips, 2), "mem_gib": round(mem, 2), "elapsed_s": round(time.time() - t0, 1),
                       **{k: round(v, 6) for k, v in info.items()}}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(rec, flush=True)
            if cfg.bench_steps and self.step - seen0 >= cfg.bench_steps:
                return 0
            if time.time() - t_ckpt > cfg.ckpt_minutes * 60:
                save_atomic(self.state(), last); t_ckpt = time.time()
        save_atomic(self.state(), last)
        (out / "DONE").write_text(str(self.step))
        return 0


def parse_args() -> tuple[Config, bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", type=int, choices=(1, 2), required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-local", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pause-file", type=Path, default=None)
    ap.add_argument("--bench-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-weights", type=Path, default=None)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--out-dim", type=int, default=4096)
    ap.add_argument("--centering", choices=("sk", "mean"), default="sk")
    ap.add_argument("--koleo", type=float, default=0.1)
    ap.add_argument("--head-warmup", type=float, default=0.10)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--lr-mult", type=float, default=1.0)
    a = ap.parse_args()
    cfg = Config(groups_csv=a.groups, data_root=a.data_root, out_dir=a.out, run=a.run, batch=a.batch,
                 n_local=a.n_local, workers=a.workers, seed=a.seed, pause_file=a.pause_file,
                 bench_steps=a.bench_steps, max_steps=a.max_steps, device=a.device, init_weights=a.init_weights,
                 grad_ckpt=not a.no_grad_ckpt, out_dim=a.out_dim, centering=a.centering, koleo_weight=a.koleo,
                 head_warmup_frac=a.head_warmup, head_lr_mult=a.head_lr_mult,
                 lr_mult=a.lr_mult)
    return cfg, a.resume


if __name__ == "__main__":
    config, resume_flag = parse_args()
    sys.exit(Trainer(config).run(resume=resume_flag))
