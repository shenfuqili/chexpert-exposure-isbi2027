"""Writes D:/isbi/queue.json for the pre-registered tuning sweep (Amendment 5): 4 variants x 1000 steps + probe."""
import json

PY = r"D:\bench_tmp\venv\Scripts\python.exe"
ROOT = r"D:\isbi"
VARIANTS = {"sw_k010_lr100": ("0.1", "1.0"), "sw_k010_lr030": ("0.1", "0.3"),
            "sw_k002_lr100": ("0.02", "1.0"), "sw_k000_lr100": ("0", "1.0")}
jobs = []
for name, (koleo, lr) in VARIANTS.items():
    out = rf"{ROOT}\runs\{name}"
    jobs.append({"name": f"{name}_train", "done": rf"{out}\DONE", "cmd": [
        PY, rf"{ROOT}\scripts\ssl_train.py", "--groups", rf"{ROOT}\data\groups.csv", "--data-root", rf"{ROOT}\data\chexpert",
        "--out", out, "--run", "1", "--seed", "1", "--out-dim", "4096", "--centering", "sk", "--koleo", koleo,
        "--lr-mult", lr, "--head-warmup", "0.1", "--head-lr-mult", "10", "--batch", "64", "--n-local", "8",
        "--workers", "6", "--max-steps", "1000", "--pause-file", rf"{ROOT}\PAUSE", "--resume",
        "--init-weights", rf"{ROOT}\data\dinov2_s_lvd142m.safetensors"]})
    jobs.append({"name": f"{name}_probe", "done": rf"{out}\probe.json", "cmd": [
        PY, rf"{ROOT}\scripts\quick_probe.py", out, rf"{ROOT}\data\groups.csv", rf"{ROOT}\data\chexpert", rf"{out}\probe.json", "P_clean"]})
with open(rf"{ROOT}\queue.json", "w", encoding="utf-8") as f:
    f.write(json.dumps({"pause_file": rf"{ROOT}\PAUSE", "jobs": jobs}, indent=1))
print([j["name"] for j in jobs])
