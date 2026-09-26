"""Fast validity check for a run directory: centre-crop teacher features for P_base (probe training) and H
(held-out test) at every snapshot, then 5-label probe macro-AUC on H. No augmented views, so it takes minutes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import CenterSet, make_encoder  # noqa: E402

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


def main(run_dir: str, groups_csv: str, data_root: str, out_json: str = "", test_group: str = "P_clean",
         n_train: int = 6000) -> None:
    import json
    g = pd.read_csv(groups_csv)
    tr = g[g.group == "P_base"].sample(n_train, random_state=0)
    te = g[g.group == test_group].sample(min(6000, int((g.group == test_group).sum())), random_state=0) \
        if test_group != "H" else g[g.group.isin(["H_a", "H_b"])]
    results = {}
    dev = torch.device("cuda")
    for snap in sorted(Path(run_dir).glob("eval_step*.pt")):
        model = make_encoder(snap, dev)
        feats = {}
        for name, df in (("tr", tr), ("te", te)):
            out = []
            with torch.no_grad():
                for x in DataLoader(CenterSet(df.Path.tolist(), Path(data_root)), batch_size=256, num_workers=6):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out.append(model(x.to(dev)).float().cpu())
            feats[name] = torch.cat(out).numpy()
        mu, sd = feats["tr"].mean(0), feats["tr"].std(0) + 1e-6
        aucs = []
        for lab in LABELS:
            ytr, yte = (tr[f"{lab}__uzero"].to_numpy(), te[f"{lab}__uzero"].to_numpy())
            clf = LogisticRegression(C=0.1, max_iter=3000).fit((feats["tr"] - mu) / sd, ytr)
            aucs.append(roc_auc_score(yte, clf.predict_proba((feats["te"] - mu) / sd)[:, 1]))
        z = torch.nn.functional.normalize(torch.from_numpy(feats["te"][:512]), dim=-1)
        results[snap.stem] = {"macro_auc": float(np.mean(aucs)), "per_label": [float(a) for a in aucs],
                              "mean_cos": float((z @ z.T).mean())}
        print(f"{snap.name}: {test_group} macro-AUC {np.mean(aucs):.4f} | per label {np.round(aucs, 3).tolist()} | mean pairwise cos {float((z @ z.T).mean()):.3f}", flush=True)
    if out_json:
        Path(out_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
