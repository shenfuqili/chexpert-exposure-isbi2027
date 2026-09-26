"""Patient-level random assignment of CheXpert-small frontal training images to study arms.

Implements Section 2 of PREREG.md: seed 20260923, all images of a patient in one group,
groups filled in a fixed order until each image-count target is reached.
Writes data/groups.csv and data/group_summary.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260923
ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

# (group, target images, dose in Run 1, dose in Run 2). Order fixes how patients are dealt out.
ARMS = [
    ("H_a", 4000, 0, 100),
    ("H_b", 4000, 0, 0),
    ("E100", 4000, 100, 0),
    ("E500", 500, 500, 500),
    ("E3", 6000, 3, 3),
    ("E15", 6000, 15, 15),
    ("P_base", 15000, 10, 10),
    ("P_clean", 15000, 0, 0),
    ("B", 25000, 10, 10),
]


def load_frontal(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["Frontal/Lateral"] == "Frontal"].copy()
    df["patient"] = df["Path"].str.extract(r"patient(\d+)")[0].astype(int)
    for lab in LABELS:
        raw = df[lab]
        df[f"{lab}__uzero"] = (raw == 1.0).astype(np.int8)
        df[f"{lab}__uone"] = raw.isin([1.0, -1.0]).astype(np.int8)
    return df


def assign(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    patients = df["patient"].unique()
    order = rng.permutation(patients)
    counts = df.groupby("patient").size()

    group_of = {}
    arm_idx, filled = 0, 0
    for pid in order:
        if arm_idx >= len(ARMS):
            break
        name, target, _, _ = ARMS[arm_idx]
        group_of[pid] = name
        filled += int(counts[pid])
        if filled >= target:
            arm_idx, filled = arm_idx + 1, 0

    out = df[df["patient"].isin(group_of)].copy()
    out["group"] = out["patient"].map(group_of)
    dose1 = {a[0]: a[2] for a in ARMS}
    dose2 = {a[0]: a[3] for a in ARMS}
    out["dose_run1"] = out["group"].map(dose1)
    out["dose_run2"] = out["group"].map(dose2)
    return out


def summarize(out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, *_ in ARMS:
        g = out[out["group"] == name]
        row = {
            "group": name,
            "images": len(g),
            "patients": g["patient"].nunique(),
            "dose_run1": int(g["dose_run1"].iloc[0]),
            "dose_run2": int(g["dose_run2"].iloc[0]),
            "age_mean": round(g["Age"].mean(), 1),
            "female_pct": round(100 * (g["Sex"] == "Female").mean(), 1),
            "AP_pct": round(100 * (g["AP/PA"] == "AP").mean(), 1),
        }
        for lab in LABELS:
            row[f"prev_{lab}"] = round(100 * g[f"{lab}__uzero"].mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = load_frontal(ROOT / "data" / "train.csv")
    out = assign(df)
    cols = ["Path", "patient", "group", "dose_run1", "dose_run2", "Sex", "Age", "AP/PA"]
    cols += [c for c in out.columns if c.endswith("__uzero") or c.endswith("__uone")]
    out[cols].to_csv(ROOT / "data" / "groups.csv", index=False)
    summary = summarize(out)
    summary.to_csv(ROOT / "data" / "group_summary.csv", index=False)
    print(f"frontal images available: {len(df)}, patients: {df['patient'].nunique()}")
    print(f"assigned images: {len(out)}, patients: {out['patient'].nunique()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
