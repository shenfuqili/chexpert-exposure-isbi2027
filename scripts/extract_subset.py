"""Extract only the images listed in groups.csv from the Kaggle CheXpert-small zip (skips macOS '._' junk)."""
import sys
import zipfile
from pathlib import Path

import pandas as pd


def main(zip_path: str, groups_csv: str, dest: str) -> int:
    wanted = {p.split("CheXpert-v1.0-small/", 1)[-1] for p in pd.read_csv(groups_csv)["Path"]}
    dest_dir = Path(dest)
    done = 0
    with zipfile.ZipFile(zip_path) as z:
        names = {n.split("CheXpert-v1.0-small/", 1)[-1]: n for n in z.namelist() if "/._" not in n}
        missing = sorted(wanted - names.keys())
        for rel in sorted(wanted & names.keys()):
            target = dest_dir / rel
            if target.exists():
                done += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(names[rel]))
            done += 1
    print(f"extracted or present: {done}, missing from zip: {len(missing)}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
