# refs.bib verification log

Checked on 2026-09-25. 33 requested entries: **33 verified and included, 0 unverified.**

Sources used:
- Crossref REST (`https://api.crossref.org/works/<DOI>`) for every DOI.
- Publisher and proceedings pages: nature.com meta tags, CVF open access, NeurIPS proceedings, PMLR, and Springer book pages (for LNCS volume numbers).
- arXiv abstract pages.
- OpenReview API (`api2.openreview.net/notes/search`), including the DBLP records that OpenReview mirrors.

dblp.org itself blocked automated access (HTTP 429, then a bot challenge), so DBLP data came from those OpenReview mirror records. ai.nejm.org also blocked access (HTTP 403 / Cloudflare).

Compile check: `bibtex` with the paper's `IEEEbib.bst` ran on all 33 keys with 0 warnings. It was run in a temporary directory that was then deleted.

## Included entries

| # | key | verified as | arXiv | URLs used |
|---|-----|-------------|-------|-----------|
| 1 | perezgarcia2025raddino | Nat. Mach. Intell. 7(1):119–130, 2025; 15 authors | 2401.10815 | api.crossref.org/works/10.1038/s42256-024-00965-w ; nature.com/articles/s42256-024-00965-w ; arxiv.org/abs/2401.10815 |
| 2 | oquab2024dinov2 | TMLR 2024 (accepted; published 2024-01-11; Featured Certification); 26 authors | 2304.07193 | openreview.net/forum?id=a68SUt6zFt (via API; also DBLP record journals/tmlr/OquabDMVSKFHMEA24) ; arxiv.org/abs/2304.07193 |
| 3 | caron2021dino | ICCV 2021, pp. 9630–9640 (IEEE Xplore pagination); 7 authors | 2104.14294 | api.crossref.org/works/10.1109/ICCV48922.2021.00951 ; openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html ; arxiv.org/abs/2104.14294 |
| 4 | sablayrolles2019koleo | ICLR 2019 (poster); 4 authors | 1806.03198 | openreview.net/forum?id=SkGuG2R5tm (via API; also DBLP conf/iclr/SablayrollesDSJ19) ; arxiv.org/abs/1806.03198 |
| 5 | irvin2019chexpert | Proc. AAAI 33(01):590–597, 2019; 20 authors | 1901.07031 | api.crossref.org/works/10.1609/aaai.v33i01.3301590 ; arxiv.org/abs/1901.07031 |
| 6 | wang2017chestxray8 | CVPR 2017, pp. 3462–3471 (IEEE Xplore pagination); 6 authors | 1705.02315 | api.crossref.org/works/10.1109/CVPR.2017.369 ; openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html ; arxiv.org/abs/1705.02315 |
| 7 | shih2019rsna | Radiol. Artif. Intell. 1(1):e180041, 2019; 19 authors | none found | api.crossref.org/works/10.1148/ryai.2019180041 |
| 8 | johnson2019mimiccxr | Sci. Data 6(1):317, 2019; 8 authors | none found | api.crossref.org/works/10.1038/s41597-019-0322-0 ; nature.com/articles/s41597-019-0322-0 |
| 9 | saporta2022chexlocalize | Nat. Mach. Intell. 4(10):867–878, 2022; 12 authors | none found | api.crossref.org/works/10.1038/s42256-022-00536-x ; nature.com/articles/s42256-022-00536-x |
| 10 | nguyen2022vindrcxr | Sci. Data 9(1):429, 2022; 24 authors | 2012.15029 | api.crossref.org/works/10.1038/s41597-022-01498-w ; nature.com/articles/s41597-022-01498-w ; arxiv.org/abs/2012.15029 |
| 11 | bustos2020padchest | Med. Image Anal. 66:101797, 2020; 4 authors | 1901.07441 | api.crossref.org/works/10.1016/j.media.2020.101797 ; arxiv.org/abs/1901.07441 |
| 12 | boecking2022biovil | ECCV 2022, LNCS 13696, pp. 1–21; 12 authors | 2204.09817 | api.crossref.org/works/10.1007/978-3-031-20059-5_1 ; link.springer.com/book/10.1007/978-3-031-20059-5 (gives LNCS 13696) ; citation-needed.springer.com/v2/references/10.1007/978-3-031-20059-5_1 ; arxiv.org/abs/2204.09817 |
| 13 | lian2021chestxdet | IEEE TMI 40(8):2042–2052, 2021; 7 authors | 2104.10326 | api.crossref.org/works/10.1109/TMI.2021.3070847 ; arxiv.org/abs/2104.10326 ; github.com/Deepwise-AILab/ChestX-Det-Dataset (README) |
| 14 | holste2022nihcxrlt | DALI 2022 (MICCAI workshop), LNCS 13567, pp. 22–32; 8 authors | 2208.13365 | api.crossref.org/works/10.1007/978-3-031-17027-0_3 ; link.springer.com/book/10.1007/978-3-031-17027-0 (gives LNCS 13567) ; arxiv.org/abs/2208.13365 |
| 15 | yao2025evax | npj Digit. Med. 8(1):678, 2025; 8 authors | 2405.05237 | api.crossref.org/works/10.1038/s41746-025-02032-z ; nature.com/articles/s41746-025-02032-z ; arxiv.org/abs/2405.05237 |
| 16 | sellergren2025medgemma | arXiv tech report, 2025 (latest version v4, 2026-04-06); 81 authors | 2507.05201 | arxiv.org/abs/2507.05201 |
| 17 | codella2024medimageinsight | arXiv, 2024; 31 authors | 2410.06542 | arxiv.org/abs/2410.06542 ; microsoft.com/en-us/research/publication/medimageinsight-an-open-source-embedding-model-for-general-domain-medical-imaging/ (lists arXiv only) |
| 18 | xu2023elixr | arXiv, 2023; 28 authors | 2308.01317 | arxiv.org/abs/2308.01317 ; research.google/pubs/elixr-towards-a-general-purpose-x-ray-artificial-intelligence-system-through-alignment-of-large-language-models-and-radiology-vision-encoders/ (lists "arxiv (2023)") |
| 19 | tiu2022chexzero | Nat. Biomed. Eng. 6(12):1399–1406, 2022; 6 authors | none found | api.crossref.org/works/10.1038/s41551-022-00936-9 ; nature.com/articles/s41551-022-00936-9 |
| 20 | zhang2025biomedclip | NEJM AI 2(1), 2025 (issue dated 2025-01); 24 authors | 2303.00915 | api.crossref.org/works/10.1056/AIoa2400640 ; arxiv.org/abs/2303.00915 ; huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (cross-check only) |
| 21 | ma2025arkplus | Nature 643(8071):488–498, 2025 (published online 2025-06-11); 4 authors | none found | api.crossref.org/works/10.1038/s41586-025-09079-8 ; nature.com/articles/s41586-025-09079-8 |
| 22 | you2023cxrclip | MICCAI 2023, LNCS 14221, pp. 101–111; 8 authors | 2310.13292 | api.crossref.org/works/10.1007/978-3-031-43895-0_10 ; link.springer.com/book/10.1007/978-3-031-43895-0 (gives LNCS 14221) ; arxiv.org/abs/2310.13292 |
| 23 | zhang2023kad | Nat. Commun. 14(1):4542, 2023; 5 authors | 2302.14042 | api.crossref.org/works/10.1038/s41467-023-40260-7 ; nature.com/articles/s41467-023-40260-7 ; arxiv.org/abs/2302.14042 |
| 24 | bannur2023biovilt | CVPR 2023, pp. 15016–15027; 16 authors | 2301.04558 | api.crossref.org/works/10.1109/CVPR52729.2023.01442 ; openaccess.thecvf.com/content/CVPR2023/html/Bannur_Learning_To_Exploit_Temporal_Structure_for_Biomedical_Vision-Language_Processing_CVPR_2023_paper.html ; arxiv.org/abs/2301.04558 |
| 25 | meehan2023dejavu | NeurIPS 2023 (Adv. NeurIPS 36), pp. 42775–42798; 5 authors | 2304.13850 | proceedings.neurips.cc/paper_files/paper/2023/hash/854b6ec839294bf332db0d86e2f83c3f-Abstract-Conference.html ; api.crossref.org/works/10.52202/075280-1854 ; openreview.net/forum?id=lkBygTc0SI ; arxiv.org/abs/2304.13850 |
| 26 | wang2024sslmem | ICLR 2024 (poster); 6 authors | 2401.12233 | openreview.net/forum?id=KSjPaXtxP8 (via API; also DBLP conf/iclr/WangKD0PB24) ; arxiv.org/abs/2401.12233 |
| 27 | magar2022contamination | ACL 2022 (Vol. 2: Short Papers), pp. 157–165; 2 authors | 2203.08242 | api.crossref.org/works/10.18653/v1/2022.acl-short.18 ; arxiv.org/abs/2203.08242 |
| 28 | jiang2024contamination | arXiv preprint, 2024; 7 authors | 2401.06059 | arxiv.org/abs/2401.06059 ; OpenReview API search (see note) |
| 29 | bordt2025forget | ICML 2025, PMLR 267, pp. 4998–5016; 4 authors (**ICML 2025 confirmed**) | 2410.03249 | proceedings.mlr.press/v267/bordt25a.html ; openreview.net/forum?id=Pf0PaYS9KG ; arxiv.org/abs/2410.03249 |
| 30 | carlini2022lira | IEEE S&P 2022, pp. 1897–1914; 6 authors | 2112.03570 | api.crossref.org/works/10.1109/SP46214.2022.9833649 ; arxiv.org/abs/2112.03570 |
| 31 | shokri2017membership | IEEE S&P 2017, pp. 3–18; 4 authors | 1610.05820 | api.crossref.org/works/10.1109/SP.2017.41 ; arxiv.org/abs/1610.05820 |
| 32 | lakens2017equivalence | Soc. Psychol. Personal. Sci. 8(4):355–362, 2017 | — | api.crossref.org/works/10.1177/1948550617697177 (title "Equivalence Tests" + subtitle "A Practical Primer for t Tests, Correlations, and Meta-Analyses") |
| 33 | schuirmann1987tost | J. Pharmacokinet. Biopharm. 15(6):657–680, 1987 | — | api.crossref.org/works/10.1007/BF01068419 |

"none found" = an arXiv title search (arxiv.org/search, searchtype=title) returned no match.

## Decisions and discrepancies

- **Pagination (DINO, ChestX-ray8).** Two paginations exist for each. I used the IEEE Xplore/Crossref pages, which go with the DOI.
  - DINO: 9630–9640 in IEEE Xplore vs 9650–9660 in the CVF open-access version.
  - ChestX-ray8: 3462–3471 in IEEE Xplore vs 2097–2106 in CVF (the arXiv journal reference also gives 2097–2106).
  - For BioViL-T, both versions give 15016–15027.
- **KAD author order.** The published version (Crossref/Nature) lists "…, Weidi Xie, Yanfeng Wang"; arXiv lists "…, Yanfeng Wang, Weidi Xie". I used the published order.
- **EVA-X.** A peer-reviewed version exists (npj Digital Medicine, 2025), so the entry uses it instead of the arXiv preprint.
- **BiomedCLIP.**
  - Year: Crossref dates vol. 2 no. 1 to 2025-01, so I used 2025. The HF model card's BibTeX says 2024, probably the online-first date, which I could not confirm.
  - Article locator: the DOI suffix looks like one ("AIoa2400640"), but ai.nejm.org was blocked and Crossref has no page or article number. I left the `pages` field out rather than guess.
  - The arXiv 2303.00915 title differs ("BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs").
- **Jiang et al. 2024.** There is no archival peer-reviewed version. OpenReview shows only non-archival ICLR 2024 workshop posters (R2-FM, DPFM, ME-FoMo) under a different title ("Does Data Contamination Make a Difference? Insights from Intentionally Contaminating Pre-training Data For Language Models") and a TMLR submission marked "Rejected". I cited the arXiv version.
- **MedGemma, MedImageInsight, ELIXR.** No peer-reviewed version found, so these are cited as `journal = {arXiv preprint arXiv:…}`. That is the form IEEEbib prints; it ignores the `eprint` field.
- **ChestX-Det.** It was introduced by the SAR-Net paper (IEEE TMI 2021). The arXiv abstract says "we also provide ChestX-Det … ~3500 images of 13 common disease categories". Its 10-class predecessor, ChestX-Det10 (Liu, Lian, Yu; arXiv 2006.10550, 2020), is a separate paper and is not in refs.bib. If the paper uses the 10-class version, cite that one instead.
- **AAAI (CheXpert).** I left out issue "01": IEEEbib's `@inproceedings` cannot print both volume and number and warns if both are set.
- **Title style edits** (IEEEbib sentence-cases titles; braces protect acronyms):
  - PadChest and EVA-X titles use "x-ray" as published. I normalised these to {X}-ray, and capitalised the "A" after the EVA-X colon to match the arXiv title.
  - VinDr-CXR: the published curly apostrophe became an ASCII `'`.
- **Article numbers** (Sci. Data, Nat. Commun., npj Digit. Med., MedIA, Radiol. AI) go in `pages`. IEEEbib prints them as "pp. 317".
- **Metadata-only fields.** IEEEbib ignores `doi`, `eprint`, `archivePrefix` and `url`; they are kept for reference.
- **Name fixes.** `{Chih-ying}` is braced; without the braces BibTeX treats "ying" as a von-particle and prints "Chih ying Deng".
- **Omitted by choice** (not needed for IEEE style): month, editors, publisher/address.

## UNVERIFIED

None.

## Related work for novelty check (not in refs.bib)

**Directly measure contamination or overlap for medical-imaging foundation models or VLMs:**
- B. C. Xu, L. Wu, A. Ryu, "A Controlled Audit of Pretraining Contamination in Public Medical Vision-Language Benchmarks," arXiv 2606.10066, June 2026 (preprint). https://arxiv.org/abs/2606.10066
  - Audits open medical VLMs on SLAKE-En, PathVQA, VQA-RAD and an OmniMedVQA mirror, using image near-neighbour overlap against PMC-OA plus text-side detectors.
  - 4.2–19.8% of SLAKE-En images are flagged, but manual checks judge this source/distribution overlap rather than exact duplicates.
  - Shows that Min-K%++ and cross-model detectors are unreliable.
  - This is the closest prior work. It covers VQA/VLMs, not CXR encoders.
- W. Zhang, Z. Zhou, J. Kang, S. Li, "Auditing Data Leakage in Whole-Slide Image Multimodal Benchmarks," arXiv 2607.12278, July 2026. https://arxiv.org/abs/2607.12278
  - Traces slide, case and tissue-source-site IDs across TCGA-derived WSI-VQA benchmarks and finds 92.3–100% case-level train/test overlap.
  - The leakage is linearly decodable from foundation-model features, and accuracy differs between leaked and clean cases.

**Adjacent (not imaging):**
- S. Ali, "Auditing pretraining contamination in single-cell foundation model benchmarks," arXiv 2607.20572, July 2026. https://arxiv.org/abs/2607.20572
  - Combines MinHash fingerprints against the pretraining corpus with a loss-based membership inference attack.
  - Includes a controlled re-pretraining experiment, a design similar to randomized exposure.
- Z. Wang et al., "Membership Inference Attacks Expose Participation Privacy in ECG Foundation Encoders," arXiv 2604.10424, April 2026. https://arxiv.org/abs/2604.10424
  - Subject-level membership-inference audit of self-supervised ECG encoders.

**Tangential (does not measure overlap):**
- Y. Kim et al., "Encoding Versus Linear Use of Patient Characteristics in Chest X-Ray Foundation Models on MIMIC-CXR," Diagnostics, 2026, doi:10.3390/diagnostics16132030.
  - Uses only "overlap-free" encoders for its main MIMIC-CXR analysis. RAD-DINO, CheXzero and CheSS are limited to encoding and fairness analyses because their pretraining overlaps MIMIC-CXR.

No work was found that measures pretraining exposure or benchmark overlap for **chest X-ray encoders** specifically, including with randomized-exposure experiments. Searches covered CXR foundation-model contamination, pretraining/test overlap, leakage, memorization and membership inference.

## Added 2026-09-26

| key | verified as | source |
|-----|-------------|--------|
| xu2026medvlm | arXiv 2606.10066 (8 Jun 2026); B. C. Xu, L. Wu, A. Ryu | arxiv.org/abs/2606.10066 |
| zhang2026wsileak | arXiv 2607.12278 (14 Jul 2026); W. Zhang, Z. Zhou, J. Kang, S. Li | arxiv.org/abs/2607.12278 |
| ali2026scfm | arXiv 2607.20572 (21 Jul 2026); S. Ali | arxiv.org/abs/2607.20572 |
| wang2026ecgmia | arXiv 2604.10424 (12 Apr 2026); Z. Wang, E. Khatibi, A. Sharma, K. Chakrabarty, S. R. Moosavi, F. Firouzi, A. Rahmani | arxiv.org/abs/2604.10424 |
| liu2021encodermi | ACM CCS 2021, pp. 2081–2095; 4 authors | api.crossref.org/works/10.1145/3460120.3484749 |
| mayilvahanan2024clip | ICLR 2024; 5 authors | arxiv.org/abs/2310.09562 |
| microsoft2024raddinocard | Hugging Face model card of microsoft/rad-dino (source of the 882,775-image manifest, step 35,000 of 100,000 and 2,560 images per step) | huggingface.co/microsoft/rad-dino |

Author lists with more than three names are shortened to three plus "et al." in refs.bib to fit the reference page; the full lists are in archive/refs_full_authors.bib.
