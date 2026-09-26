"""Writes D:/isbi/queue.json. Usage: python make_queue.py <batch> <n_local> [--full]"""
import json
import sys

PY = r"D:\bench_tmp\venv\Scripts\python.exe"
ROOT = r"D:\isbi"
COMMON = ["--groups", rf"{ROOT}\data\groups.csv", "--data-root", rf"{ROOT}\data\chexpert"]
PAUSE = rf"{ROOT}\PAUSE"


def arg(name: str, default: str) -> str:
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


TRAIN_FLAGS = ["--out-dim", "4096", "--centering", "sk", "--koleo", arg("koleo", "0.1"), "--lr-mult", arg("lr-mult", "1.0"),
               "--head-warmup", "0.1", "--head-lr-mult", "10"]


def train(name: str, run: int, batch: str, n_local: str, max_steps: int = 0) -> dict:
    cmd = [PY, rf"{ROOT}\scripts\ssl_train.py", *COMMON, "--out", rf"{ROOT}\runs\{name}", "--run", str(run),
           "--seed", str(run), *TRAIN_FLAGS,
           "--batch", batch, "--n-local", n_local, "--workers", "6", "--pause-file", PAUSE, "--resume",
           "--init-weights", rf"{ROOT}\data\dinov2_s_lvd142m.safetensors"]
    if max_steps:
        cmd += ["--max-steps", str(max_steps)]
    return {"name": f"{name}_train", "cmd": cmd, "done": rf"{ROOT}\runs\{name}\DONE"}


def extract(name: str) -> dict:
    cmd = [PY, rf"{ROOT}\scripts\extract_features.py", *COMMON, "--run-dir", rf"{ROOT}\runs\{name}",
           "--workers", "6", "--pause-file", PAUSE, "--done-marker", rf"{ROOT}\runs\{name}\FEATS_DONE"]
    return {"name": f"{name}_extract", "cmd": cmd, "done": rf"{ROOT}\runs\{name}\FEATS_DONE"}


if __name__ == "__main__":
    batch, n_local, full = sys.argv[1], sys.argv[2], "--full" in sys.argv
    jobs = []
    if full:
        rep = lambda mode: {"name": f"report_{mode}", "done": rf"{ROOT}\results\REPORT_{mode}_SENT",
                            "cmd": [PY, rf"{ROOT}\scripts\report.py", mode]}
        jobs += [train("run1", 1, batch, n_local), extract("run1"), rep("interim"),
                 train("run2", 2, batch, n_local), extract("run2"), rep("final")]
    text = json.dumps({"pause_file": PAUSE, "jobs": jobs}, indent=1)
    out = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    if out:
        with open(out, "w", encoding="utf-8") as f:  # no BOM
            f.write(text)
    print(text)
