# Table 1: benchmark-exposure matrix for public CXR encoders (evidence-graded)

Compiled 2026-09-24. One row per released checkpoint. "Exposure" means the benchmark's images were in the encoder's pretraining data. Using a benchmark's validation data to select or tune a checkpoint is flagged separately as `+sel`.

Per-cell quotes, computed counts and sources (URL plus section, table or page) are in `exposure_matrix.csv` (120 rows).

## Matrix

| Encoder (release) | NIH test | RSNA | SIIM-ACR | ChestX-Det | CheXpert val | CheXpert test | MIMIC test | MS-CXR | VinDr test | PadChest |
|---|---|---|---|---|---|---|---|---|---|---|
| RAD-DINO (HF release) | **M** (25,596/25,596) | **D\*** (30,000/30,000) +sel | **D** | **D** | **M** (234/234) | S-train\* | S-train\* | M-part | N\* +sel | M-part (≥38%)ᵃ |
| EVA-X | S-train | D-part\* (63–73%) | D-part | D-part | S-train | S-train | ? | D-part | N | N |
| MedSigLIP (MedGemma) | Nᵇ | Nᵇ | Nᵇ | Nᵇ | N | N | S-train | D-part | N | N |
| MedImageInsight | **S-part\*** (45.4%) | D-part\* (48.5%) | D-part | D-part | N | N | S-train | D-part | N | N |
| CXR Foundation / ELIXR | ? | D-part | D-part | D-part | N +sel | N | S-train | D-part | N | N |
| CheXzero | N | N | N | N | N +sel | N | **S** | **D** | N | N |
| BiomedCLIP | N | N | N | N | N | N | N | N | N | N |
| Ark+ (Ark-6) | S-train\* | D-part\* (89%; 64% of Ark's own RSNA test) | D-part | D-part | S-train\* | S-train\* | S-train\* | M-part | S-train\*ᶜ | N |
| CXR-CLIP (M,C,C14) | ? | D-part (59–80%) | D-part | D-part | S-train +sel | S-train | S-train | D-part | N | N |
| KAD | N +sel | N | N | N | N +sel | N | ? | D-part | N | N |
| BioViL-T (added) | N | N | N | N | N | N | ? | N | N | N |
| DINOv2 LVD-142M (control) | N | N | N | N | N | N | N | N | N | N |

RSNA percentages are shares of all 30,000 RSNA challenge images that are exposed through NIH. Shares of the 26,684-image Kaggle stage-2 training pool, which most papers split, are within 0.5 percentage points of these, except Ark+ (92.7%).

- ᵃ Of the 39,053-image physician-labelled PadChest subset that CheXzero and KAD use as a test set. This is a counting lower bound: 136,787 − (160,868 − 39,053) = 14,972. In total, 85% of PadChest images are in the manifest.
- ᵇ The MedSigLIP training source SLAKE contains 179 CXRs sampled from NIH. Their NIH IDs are unknown, so at most 179 images could overlap.
- ᶜ Ark+'s pretraining config uses the VinDr **test** list as its `val_list`.

## Legend

| Code | Meaning |
|---|---|
| M | Exposed. Verified image by image against a released training-image list (RAD-DINO `training_images.csv`; Ark+ pretraining split lists). |
| S | Exposed according to the stated training data (paper or model card). |
| S-train | The source dataset was used for training, but per the statement only its training split, so this evaluation split is not exposed. |
| D | Exposed through a documented derived relation: the benchmark's images are a subset of an exposed parent dataset. |
| -part | Only part of the benchmark is exposed. The fraction is given when it can be computed, otherwise it is unknown. |
| N | Not in the stated training data, or explicitly excluded. |
| \* | Checked at image level against released ID lists: manifests, official NIH, MIMIC and CheXpert splits, and RSNA's RSNA→NIH mapping. N\* and S-train\* mean the images are verified absent. |
| ? | Unknown, or the sources conflict. |
| +sel | The benchmark's validation data, not its test split, was used to select, ensemble or threshold the released checkpoint. |

Web or literature-trained encoders (DINOv2, BiomedCLIP, MedSigLIP's WebLI and PMC data, MedImageInsight's PMC-15M, ROCO and 1.83M proprietary images) could include individual benchmark images by accident. For those, N means "not a stated source", not "proven absent".

## Other released manifests

Beyond RAD-DINO, I checked every encoder for an image-level list of its training data.

- **Ark+** publishes the image-level split lists that its pretraining config reads. Graded M or \*.
- **MedImageInsight** trained on a public split (the NIH-CXR-LT train list), so its exposure can be computed from that list.
- **CXR-CLIP** publishes code that reproduces its NIH split, but the result conflicts with the paper (see the uncertain cells below).
- **CheXzero**'s released preprocessing code uses every MIMIC-CXR-JPG file, with no split filter.
- **None** was found for EVA-X (only downstream CXR14 splits), MedSigLIP, CXR Foundation, BiomedCLIP, KAD, BioViL-T or DINOv2. KAD's Google Drive folder holds only `best_valid.pt` checkpoints; its Baidu link was not checked.

## Verified derived-dataset relations

1. **RSNA Pneumonia ⊂ NIH ChestX-ray14.**
   - RSNA's dataset description: "30,000 frontal view chest radiographs from the 112,000-image public National Institutes of Health (NIH) CXR8 dataset".
   - RSNA's official mapping file ([JSON](https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json), linked from the [challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)) maps all 30,000 RSNA IDs to NIH filenames.
   - Computed: 21,804 of them are in the NIH `train_val_list`, 8,196 in the official `test_list`.
   - Reference: Shih et al., *Radiol AI* 2019, e180041.
2. **SIIM-ACR ⊂ NIH.**
   - [SIIM](https://siim.org/research-journal/siim-machine-learning-challenges/pneumothorax-kaggle-challenge/): the challenge used "augmented annotations on the public chest radiograph dataset from the National Institutes of Health (NIH)".
   - Filice et al., *J Digit Imaging* 2020;33:490–496, "…pneumothorax annotations … on the NIH chest X-ray dataset". I could only read the title and abstract; the full text was blocked.
   - No public SIIM→NIH ID mapping exists, so which NIH split each image came from is unknown.
3. **ChestX-Det ⊂ NIH.**
   - [README](https://github.com/Deepwise-AILab/ChestX-Det-Dataset): "ChestX-Det consists of 3578 images from NIH ChestX-14".
   - Lian et al., arXiv:2104.10326 §4.1: random split of 3,025 training and 553 test images.
   - ChestX-Det10 (arXiv:2006.10550v2, p1–2): 3,543 images, randomly split 3,001/542.
   - The released files are renamed (e.g. `36200.png`), so NIH split membership is unknown.
4. **MS-CXR ⊂ MIMIC-CXR, and not confined to MIMIC's test split.**
   - BioViL §3.1, p8: "All the benchmark samples are chosen from the public MIMIC-CXR dataset".
   - PhysioNet v1.1.0 defines MS-CXR's own 70:15:15 patient-level split.
   - The one MS-CXR image shown publicly (`c436cddb-…`, patient p15928453) is in the MIMIC **train** split. It appears in both the RAD-DINO and Ark train lists.
   - Of the 354 images in MS-CXR's documented source set MIMIC-CXR-Annotations (Tam et al. 2020), 348 are train-split.
5. **NIH-CXR-LT re-splits NIH.** Its training split (68,058 images; Box folder "LongTailCXR") contains 11,616 of the 25,596 official NIH test images.
6. **Google's CXR14 adjudicated test set lies inside the NIH official test split.** All 1,962 images (Majkowska 2020), which are the CXR14 evaluation set used by ELIXR and MedGemma, are in the official test split.
7. **SLAKE's CXRs come from NIH.** "From [8] [NIH], we randomly select 179 chest X-Ray images" (arXiv:2102.09542 §2.1).
8. **Reference split sizes.**
   - CheXpert validation: 200 studies, 234 images (patients 64541–64740). CheXpert test: 500 studies, 668 images (patients ≥64741) (Irvin 2019; CheXlocalize release; local files).
   - MIMIC-CXR-JPG: train 368,960, validate 2,991, test 5,159 images (arXiv:1901.07042v5, Table 3, p4).
   - VinDr-CXR: train 15,000, test 3,000.
   - PadChest: 160,868 images and no official test split.

**How the image-level numbers were computed.** Every number above comes from exact set intersections of public ID lists:

- NIH `test_list.txt` and `train_val_list.txt`, from the NIH Box.
- The RSNA mapping JSON.
- The NIH-CXR-LT split CSVs.
- The Ark+ `Ark_Plus/dataset/*` lists.
- EVA-X's `cxr14` split files.
- The RAD-DINO manifest.
- Google's NIH labels (torchxrayvision copy).
- The CXR-CLIP README split, reproduced with the same patient-level split code (`train_test_split`, `random_state=0`, on `Data_Entry_2017_v2020.csv`).
- For MIMIC, the official test and validate lists as copied in the public Ark repository. The original PhysioNet file requires credentialed access and was not consulted.

## Cells I am least sure about

1. **CheXzero × MIMIC test = S.**
   - Supporting S: the paper says "377,110 pairs" and "all images … stored in a single HDF5 file", and 377,110 is the whole-dataset count. The released code applies no split filter.
   - Against: one sentence says "jointly train on the MIMIC-CXR training dataset", and another says one AP/PA image was chosen per study.
2. **CXR-CLIP × NIH test = ?.**
   - The text says "20% of the original training set" was held out for validation, and the README splits only `train_val_list`.
   - But Table 1 lists 89,696 training and 22,423 validation images. These add up to 112,119, i.e. essentially all of NIH, which would put about 80% of the official test split in pretraining.
3. **CXR Foundation × NIH test = ?, and its NIH-derived D-part cells.**
   - Its SupCon initialisation used ChestX-ray14 (Sellergren 2022), but the split is not stated in the text I could reach. The paper's tables are behind a 403 on pubs.rsna.org.
   - The HAI-DEF model card's training-data list omits ChestX-ray14.
4. **Every MS-CXR cell.**
   - Only one MS-CXR image ID is public, so fractions are unknown.
   - With PhysioNet credentials, intersecting the MS-CXR IDs with the RAD-DINO and Ark lists would settle both rows exactly.
5. **SIIM and ChestX-Det D-part fractions.** Neither has an NIH ID mapping. RAD-DINO's cells are still full D, because it contains all of NIH.
6. **MedSigLIP NIH-column cells = N.** SLAKE's 179 NIH images could overlap.
7. **MIMIC test = ? for EVA-X, KAD and BioViL-T.** None states the split: EVA-X and KAD give no split, and BioViL-T uses its own split with a 2,971-image held-out set. All three still get D-part for MS-CXR, because any use of MIMIC exposes its training-split images.
8. **RAD-DINO +sel scope.** The card says selection used the "validation sets of the evaluation datasets described in the paper". I read that as the paper's linear-probe sets: VinDr-CXR, CANDID-PTX and RSNA.
9. **Ark+ VinDr test.**
   - The pretraining config's `val_list` is the VinDr test list. With the default cosine schedule it is only logged.
   - How the released `ep50` checkpoint was chosen is not documented.
10. **EVA-X RSNA range.**
    - 63% if the NIH val split was excluded from pretraining, 73% if it was included; the paper does not say which.
    - EVA-X's cxr14 train list is identical to Ark's.

## Corrections to the task brief

- **RAD-DINO card and RSNA.** No version of the RAD-DINO model card says RSNA-Pneumonia is an NIH subset. I checked all 26 README revisions (2024-05-17 to 2026-05-12); none mentions RSNA, pneumonia or "external".
  - The "external" label comes from the paper: §2.1.1, p4 and §3.5, p9.
  - The NIH derivation comes from RSNA's own description and mapping file.
  - The card does say two relevant things: MAIRA validation and test images were excluded, and the checkpoint was chosen by linear probing on validation sets.
- **Which checkpoint the manifest covers.** The manifest describes the released checkpoint only. The paper's checkpoint also used private data, 210,491 frontal MIMIC images and 160,817 PadChest images (Table D.1, p29).
