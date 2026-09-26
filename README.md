# Seen in pretraining: randomized exposure in chest X-ray encoders

Code, analysis plan, audit and results for the ISBI 2027 submission of the same title (Zhibin Wang, Xi'an
Jiaotong-Liverpool University).

Public chest X-ray encoders are often evaluated on benchmarks whose test images were in their pretraining data.
We continue DINO pretraining of DINOv2-S on CheXpert with patients assigned at random to arms seen 0 to 500 times,
run a crossover that swaps the dose-100 arm with a held-out arm, and measure (1) the frozen-probe AUC premium of
seen over unseen images and (2) whether exposure can be detected, including on the released RAD-DINO.

## Contents

| Path | What |
|---|---|
| `PREREG.md` | Analysis plan written before training, with dated amendments 1-10 and analysis logs |
| `audit/REGISTRATION_HASHES.txt` | SHA-256 of every plan version and of code and results at each step, with UTC times |
| `audit/exposure_matrix.md`, `.csv` | Full benchmark-exposure audit (12 encoders x 10 benchmarks) with sources; Table 1 shows a subset |
| `scripts/` | Group assignment, continued DINO pretraining, feature extraction, analysis, RAD-DINO analyses, amendment checks, and the generators for Table 1, Figs. 1-2 and every number in the text |
| `scripts/archive/` | The analysis script exactly as first run (see the analysis log in `PREREG.md`) |
| `tests/` | pytest suite |
| `results/` | All analysis outputs used in the paper (scores and statistics only; no images or labels) |
| `data/group_summary.csv` | Per-arm counts and aggregate statistics |
| `paper/` | LaTeX source; `paper/numbers.tex` is written by `scripts/make_paper_numbers.py` |
| `deploy/` | Job files and launch scripts used on the training machine (Windows laptop, RTX 4050, 6 GB) |

## Data (not included)

- CheXpert (Stanford AIMI): the downsampled training split for pretraining and probes.
- CheXlocalize (Stanford AIMI): full-resolution CheXpert validation and test images for the RAD-DINO analyses.
- RAD-DINO weights, DINO head and training manifest: Hugging Face `microsoft/rad-dino`.
- DINOv2-S/14 (timm `vit_small_patch14_dinov2.lvd142m`) and DINOv2-B (`facebook/dinov2-base`).

The CheXpert license does not allow redistribution, so the patient-to-arm assignment with labels
(`data/groups.csv`) is not included. `scripts/assign_groups.py` regenerates it from CheXpert's `train.csv`
with the fixed seed (20260923); `data/group_summary.csv` holds per-arm aggregates only.

## Reproducing

Python 3.10 (`pip install -r requirements.txt`). Each script documents its options (`--help`). In order:

1. `scripts/assign_groups.py` - patient-level random assignment.
2. `scripts/ssl_train.py` - two runs (`--run 1`, `--run 2`); about 7 h each on an RTX 4050 laptop GPU.
3. `scripts/extract_features.py` - frozen features and membership scores for each saved snapshot.
4. `scripts/analyze.py` - probes, premiums, crossover endpoint, membership AUCs, validity checks
   (the three pre-specified variants are listed in `deploy/run_reanalysis.ps1`).
5. `scripts/bridge_raddino.py`, `scripts/head_score.py`, `scripts/amendment7_summary.py`,
   `scripts/amendment9_checks.py`, `scripts/amendment10_matched.py` - RAD-DINO analyses and later amendments.
6. `scripts/make_table1.py`, `scripts/make_fig1.py`, `scripts/make_fig2.py`, `scripts/make_paper_numbers.py`,
   then `latexmk -pdf main.tex` in `paper/`. `make_paper_numbers.py` stops if a claim made in the text no longer
   holds for the result files.

`pytest` runs the test suite; the tests that read `results/` need no data.

## Use of AI

Code, figures, the exposure audit and the draft text were produced with Claude (Anthropic); the author checked
all content and takes full responsibility for it.

## License

Code: MIT (see `LICENSE`). The datasets and models above keep their own licenses.
