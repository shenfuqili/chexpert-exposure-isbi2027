"""Writes paper/tables/audit.tex (Table 1) from the matrix in audit/exposure_matrix.md, so the paper never
diverges from the audited source. Partial cells keep their share, "+sel" becomes a dagger."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "audit" / "exposure_matrix.md", ROOT / "paper" / "tables" / "audit.tex"
SHORT = {"RAD-DINO (HF release)": "RAD-DINO", "MedSigLIP (MedGemma)": "MedSigLIP", "CXR Foundation / ELIXR": "ELIXR",
         "Ark+ (Ark-6)": "Ark+", "CXR-CLIP (M,C,C14)": "CXR-CLIP", "BioViL-T (added)": "BioViL-T",
         "DINOv2 LVD-142M (control)": "DINOv2"}
CITE = {"RAD-DINO": "perezgarcia2025raddino", "EVA-X": "yao2025evax", "MedSigLIP": "sellergren2025medgemma",
        "MedImageInsight": "codella2024medimageinsight", "ELIXR": "xu2023elixr", "CheXzero": "tiu2022chexzero",
        "BiomedCLIP": "zhang2025biomedclip", "Ark+": "ma2025arkplus", "CXR-CLIP": "you2023cxrclip",
        "KAD": "zhang2023kad", "BioViL-T": "bannur2023biovilt", "DINOv2": "oquab2024dinov2"}
NOTE_MARK = {"ᵃ": "a", "ᵇ": "b", "ᶜ": "c"}
DROP = {"BioViL-T", "KAD", "MedSigLIP", "BiomedCLIP", "DINOv2"}  # no documented exposure on KEEP_COLS; kept in the released audit/
KEEP_COLS = ["NIH test", "RSNA", "CheX val", "CheX test", "MIMIC test", "VinDr test"]
SHORT_HEAD = {"NIH test": "NIH", "RSNA": "RSNA", "CheX val": "CX val", "CheX test": "CX test", "MIMIC test": "MIMIC",
              "VinDr test": "VinDr"}
HEAD = ["NIH test", "RSNA", "SIIM", "ChestX-Det", "CheX val", "CheX test", "MIMIC test", "MS-CXR", "VinDr test",
        "PadChest"]


def matrix_rows(md: str) -> list[list[str]]:
    block = md.split("## Matrix", 1)[1].split("## Legend", 1)[0]
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in block.splitlines() if ln.startswith("|")]
    return [r for r in rows[2:] if r]  # drop header and separator


def cell(raw: str) -> str:
    sel = "+sel" in raw
    notes = "".join(NOTE_MARK[m] for m in re.findall(r"[ᵃᵇᶜ]", raw))
    txt = re.sub(r"[ᵃᵇᶜ]", "", raw.replace("+sel", "")).strip()
    share = re.search(r"\((≥?\d[\d–%.≥ ]*%?)", txt)
    code = re.sub(r"\(.*?\)", "", txt).replace("\\*", "*").replace("**", "").strip()
    code = code.replace("-part", "p").replace("S-train", "St")
    out = code  # no bold: the letters already grade, and bold made the one full manifest look worst
    if share and "p" in code:
        out += r"\,{\tiny " + share.group(1).replace("–", "--").replace("≥", r"$\geq$").replace("%", r"\%") + "}"
    out = out.replace("*", r"$^*$") + (r"$^\dagger$" if sel else "")
    return out + (rf"$^{{{notes}}}$" if notes else "")


def main() -> None:
    rows = matrix_rows(SRC.read_text(encoding="utf-8"))
    cols = [HEAD.index(h) for h in KEEP_COLS]
    kept = [r for r in rows if SHORT.get(r[0], r[0]) not in DROP]
    lines = [r"\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{2.4pt}",
             r"\caption{Benchmark exposure of public chest X-ray encoders. M: exposed, verified against a released manifest; "
             r"S: exposed per stated training data; D: exposed via derivation (RSNA images are NIH images); p: partly "
             r"(share if computable); St: only the source's training split used (not exposure); N: not a stated "
             r"source; ?: unclear. $^*$ checked at image level; $^\dagger$ benchmark validation data selected or "
             r"tuned the checkpoint; $^c$ Ark+ uses the VinDr test list for validation. RAD-DINO and Ark+ released "
             r"image-level lists, so their rows are the most complete. Other NIH subsets follow the RSNA column; five "
             r"other encoders show no documented exposure on these benchmarks. The full audit, with sources, is "
             r"released. Benchmarks: NIH ChestX-ray14~\cite{wang2017chestxray8}, RSNA Pneumonia~\cite{shih2019rsna}, "
             r"CheXpert (CX)~\cite{irvin2019chexpert}, MIMIC-CXR~\cite{johnson2019mimiccxr}, "
             r"VinDr-CXR~\cite{nguyen2022vindrcxr}.}\label{tab:audit}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Encoder & " + " & ".join(SHORT_HEAD[h] for h in KEEP_COLS) + r" \\", r"\midrule"]
    for r in kept:
        name = SHORT.get(r[0], r[0])
        label = name.replace("+", r"{+}") + rf"~\cite{{{CITE[name]}}}"
        lines.append(label + " & " + " & ".join(cell(r[1 + i]) for i in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(kept)} of {len(rows)} encoders, {len(cols)} of {len(HEAD)} benchmarks)")


if __name__ == "__main__":
    main()
