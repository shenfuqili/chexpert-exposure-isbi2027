# Pre-registration: randomized benchmark-exposure study (ISBI 2027)

Registered before any training run. The registration timestamp is the SHA-256 record in audit/REGISTRATION_HASHES.txt.
Any later change to this file is recorded as a dated amendment at the bottom, with the reason, before the affected analysis is run.

## 1. Questions and estimand

- Q1 (detectability): at a given exposure dose (number of times an image is seen during continued self-supervised pretraining), how well does membership inference separate exposed from held-out images?
- Q2 (exposure premium): how much higher is the frozen-encoder linear-probe AUC on exposed test images than on held-out test images drawn from the same distribution?
- Estimand: the instance-level premium. Exposed and held-out images come from the same source, patients and preprocessing pipeline, assigned at random at the patient level, so domain familiarity is equal across arms.

## 2. Data and randomization

- Source: CheXpert-small training split, frontal views only.
- Randomization: patients are assigned to groups with `numpy.random.default_rng(20260923)`, stratified by nothing; all images of a patient go to one group.
- Groups (image counts are targets; exact counts are fixed by the assignment script and logged):
  - B background: 25,000 images, dose 10
  - P_base probe-training: 15,000 images, dose 10
  - P_clean probe-training: 15,000 images, dose 0
  - H held-out test: 8,000 images (H_a 4,000 + H_b 4,000), dose 0
  - E3: 6,000 images, dose ~3
  - E15: 6,000 images, dose ~15
  - E100: 4,000 images, dose ~100
  - E500: 500 images, dose ~500 (positive control)
- Doses are implemented with a weighted sampler (per-image sampling weight = dose / 10 relative to the background), not by using intermediate checkpoints as doses.

## 3. Training

- Initialization: DINOv2-S/14 (timm `vit_small_patch14_dinov2.lvd142m`).
- Objective: DINO loss with an EMA teacher, plus KoLeo regularization; no iBOT.
- Crops: 2 global crops at 224 px, local crops at 98 px (count fixed after the throughput benchmark, between 4 and 8, before Run 1 starts).
- Precision bf16 with gradient checkpointing; the same hyperparameters for both runs.
- Run 1: arms as in Section 2. Run 2 (crossover): H_a and E100 swap roles; everything else identical except the augmentation seed.
- Evaluation checkpoints: step 0 (initial weights), 35% of the schedule, 100% of the schedule.

## 4. Primary endpoint

- Delta = macro-AUC(E100) - macro-AUC(H) at 100% of the schedule, Run 1.
- Macro-AUC over the 5 CheXpert competition labels: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion. Uncertain labels (-1) count as negative; blank labels count as negative.
- Features: CLS token of the teacher encoder, standardized with statistics from the probe-training set.
- Probe: one-vs-rest logistic regression trained on P_base; the inverse regularization strength C is chosen by 5-fold patient-grouped cross-validation within P_base, from {0.01, 0.1, 1, 10}.
- Smallest effect size of interest: 0.005 macro-AUC.
- Test: two one-sided tests (TOST) for equivalence within +/-0.005 at alpha = 0.05, using a patient-clustered bootstrap (10,000 resamples) for the confidence interval; a patient-level permutation test of arm labels (10,000 permutations) for the difference from zero.

## 5. Secondary endpoints (Holm correction across this family)

1. Delta at doses 3, 15 and 500, and at 35% of the schedule.
2. Delta for a k-NN classifier (k = 20, cosine similarity, reference bank = P_base).
3. Membership-inference AUC and TPR at 1% FPR for each exposed arm versus H. Primary score: augmentation invariance (mean pairwise cosine similarity among 8 augmented views), calibrated by subtracting the same score under a reference encoder (the crossover partner run for E100 and H_a; the step-0 encoder for other arms). Secondary score: DINO loss under the run's own heads.
4. Delta with the probe trained on P_clean.
5. Delta with uncertain labels counted as positive.
6. Crossover paired analysis: per-image difference in probe log-likelihood when exposed versus held-out, across Run 1 and Run 2.

## 6. Validity checks (reported whatever they show)

- Positive control: membership AUC for E500 at 100% should exceed 0.6. If it does not, the membership analysis is reported as insensitive.
- Adaptation: macro-AUC on H at 100% should exceed macro-AUC on H at step 0. If it does not, the continued pretraining is reported as not having adapted the encoder.
- Collapse: k-NN macro-AUC on H at 100% should not fall more than 0.01 below step 0.

## 7. Bridge to the released RAD-DINO (secondary, descriptive)

- Members: CheXpert validation images (234, listed in RAD-DINO's training_images.csv). Non-members: CheXpert test images (668, not listed). Full-resolution sources, preprocessed identically (shorter side 518 with B-spline resizing, min-max scaling).
- Exchangeability gate: a logistic classifier on DINOv2-B features, patient-grouped 5-fold cross-validation, must give AUC < 0.55 at separating the two pools. If it does not, the bridge is reported as confounded and repeated on the subset of images with the modal size.
- Score: augmentation invariance under RAD-DINO minus the same score under DINOv2-B. Report membership AUC with a patient-clustered bootstrap CI and the minimal detectable effect.

## 8. What will not change after seeing results

Group definitions and sizes, doses, the random seed for assignment, the primary endpoint, the equivalence margin, the label policy, the probe protocol and the statistical tests above.

## Amendments

### Amendment 1 (2026-09-23, before any data was downloaded or any model was trained)

Reason: a power check showed that the unpaired Run-1 contrast in Section 4 (4,000 vs 8,000 images, macro-AUC over 5 labels) has a standard error of roughly 0.008-0.009 for the difference, so its 90% interval would be about +/-0.015 and could never establish equivalence within +/-0.005. The crossover already planned in Section 3 removes image-difficulty variance.

Change to Section 4 (primary endpoint):
- Let G1 = the images of H_a and G2 = the images of E100 in Run 1. In Run 2 their roles swap, so G1 is exposed at dose ~100 and G2 is held out.
- Primary estimate: Delta_x = 1/2 [ (AUC_G1,Run2 - AUC_G1,Run1) + (AUC_G2,Run1 - AUC_G2,Run2) ], where each AUC is the 5-label macro-AUC of the probe trained on P_base within that run, evaluated at 100% of the schedule. Averaging the two groups cancels the run (period) effect.
- Inference: patient-clustered bootstrap resampling patients within G1 and within G2 jointly (10,000 resamples); TOST within +/-0.005 at alpha = 0.05; paired permutation test that flips the exposure label per patient across runs.
- Pre-specified fallback: if Run 2 has not finished evaluation by 2026-10-10, the primary endpoint reverts to the unpaired Run-1 contrast of Section 4, and the paper reports that the equivalence test is underpowered.

The unpaired Run-1 contrast of Section 4 becomes secondary endpoint 0 in Section 5. Nothing else changes.

### Amendment 2 (2026-09-23, before any data was downloaded or any model was trained)

Reason: testing the analysis code on synthetic data with no exposure effect showed that the inference in Amendment 1 is anti-conservative. A patient bootstrap within fixed image sets ignores the run-by-image-set interaction (the two training runs differ randomly, and that difference is not the same on every image set); a per-patient permutation is invalid because exposure is assigned to whole groups, not to patients.

Change to the inference for the primary endpoint (point estimate unchanged):
- Write D_S = AUC_S(Run 2) - AUC_S(Run 1) for an image set S, so Delta_x = 1/2 (D_G1 - D_G2). Under no exposure effect, D_G1 and D_G2 are independent draws with the same mean (the run effect), so Var(Delta_x) = [Var(D at |G1|) + Var(D at |G2|)] / 4.
- Var(D at n images) is estimated from 2,000 patient-level random sets of n images drawn from groups whose exposure is identical in both runs and which are not used to train the probe (P_clean and H_b; 19,000 images), multiplied by 1 / (1 - n/N) to correct for drawing from a finite pool of N images.
- 90% CI = Delta_x +/- 1.645 SE; equivalence is declared if it lies inside +/-0.005 (TOST at alpha = 0.05); two-sided p from the normal approximation.
- Calibration check done on synthetic null data before registration: 0 of 30 null simulations rejected at alpha = 0.05, and all 30 95% intervals covered zero; an injected small effect was detected.
- The patient-bootstrap interval (image sampling only) is still reported, labelled as such.

### Amendment 3 (2026-09-23, clerical, before any data was downloaded or any model was trained)

The header sentence said the registration timestamp was a git commit; no commit was made. It now points to the SHA-256 record in audit/REGISTRATION_HASHES.txt, which holds a hash of every version of this file. No analysis content changed.

### Amendment 4 (2026-09-23, before the bridge analysis was run)

The bridge in Section 7 is restricted to frontal views: 202 validation images (200 patients) as members and 518 test images (500 patients) as non-members. Reason: lateral views make up 14% of the validation set and 22% of the test set, a view-mix difference that the exchangeability gate would pick up and that is unrelated to membership; the randomized experiment also uses frontal views only. The minimal detectable membership AUC becomes about 0.567.

### Decision record (2026-09-23, planned in Section 3, made before Run 1)

Throughput benchmark on the RTX 4050 Laptop (60 steps, batch 32, steady state from step 20 to 60): 6 local crops 50 img/s; 8 local crops 45 img/s, peak allocated memory 1.9 GiB with gradient checkpointing; without gradient checkpointing 6.45 GiB exceeds the 6 GB card and throughput falls to 8.7 img/s. Chosen for both runs: batch 32, 8 local crops (matching RAD-DINO's 8), gradient checkpointing on. Expected 36,320 steps (about 7.2 h) per run.

### Amendment 5 (2026-09-24, after the pilot and before any formal run; no exposure outcome was examined)

Reason: the pilot (2,000 steps, the Section 3 recipe with batch 32 and 65,536 prototypes) collapsed to the uniform solution. Diagnostics on 64 held-out images: teacher logit spread across prototypes fell from 0.062 at step 0 to 0.0014; teacher entropy and the DINO cross-entropy both equalled ln K (11.090); mean pairwise cosine between different images' CLS features fell from 0.90 to 0.035, i.e. the KoLeo term alone was reshaping the encoder. Chest X-ray CLS features are so similar to one another that a randomly initialised head maps every image to nearly the same output, so the centred teacher targets start almost uniform; with a small batch this is an absorbing state. A collapsed run would manufacture instance-level separation and invalidate an exposure study. A second trial with Sinkhorn-Knopp targets and 4,096 prototypes but no head normalisation also collapsed (DINO loss fixed at ln 4096 = 8.318). Only training dynamics and the probe AUC on held-out images were looked at; no exposure contrast or membership score was computed on either pilot.

Changes to Section 3 (training), applied identically to both formal runs:
- Projection head with BatchNorm after the two hidden layers (DINO's use_bn_in_head option), in both student and teacher heads.
- Teacher targets by Sinkhorn-Knopp normalisation (DINOv2's default for its released models) instead of centring; 4,096 prototypes instead of 65,536; batch 64 instead of 32 (learning rate follows the same square-root rule: 1e-3 x sqrt(64/1024)).
- Head warm-up: for the first 10% of steps the backbone is frozen and the head is trained at 10x the scheduled learning rate.
- KoLeo computed in fp32 outside autocast (DINOv2's pairwise-distance form); non-finite steps are skipped and counted.
- The shuffled schedule is repaired so that no image appears twice within one batch (keeps KoLeo from pushing two views of one exposed image apart, which cannot happen at RAD-DINO's scale); per-image doses are unchanged.
- Run 2 uses seed 2 (augmentation and schedule), Run 1 seed 1.
- Collapse monitors are logged every 50 steps (teacher and student entropy, teacher logit spread, mean pairwise CLS cosine).

Verification trial (600 steps, new recipe): 600 steps did not collapse: DINO loss 8.13 at step 300 (ln 4096 = 8.318), teacher logit spread rose from 0.054 to 0.060, mean pairwise CLS cosine fell from 0.91 to 0.02-0.04. For reference, on 96 CheXpert images the mean pairwise CLS cosine is 0.922 for DINOv2-S, 0.949 for DINOv2-B (RAD-DINO's initialisation) and 0.045 for the released RAD-DINO, so this spreading matches RAD-DINO's end state. However, a linear probe (trained on 6,000 P_base images, tested on H) fell from macro-AUC 0.685 at step 0 to 0.636 at step 600. H was used here only for this validity quantity, never for an exposure contrast; from now on tuning uses P_clean.

Pre-specified tuning sweep, run before any formal run (1,000 steps each, same recipe otherwise, Run-1 arms, seed 1):
- V1 KoLeo 0.1, lr x1 (recipe above); V2 KoLeo 0.1, lr x0.3; V3 KoLeo 0.02, lr x1; V4 KoLeo 0, lr x1.
- Evaluation: 5-label probe trained on 6,000 P_base images, macro-AUC on 6,000 P_clean images, at the final snapshot.
- Collapse exclusion: final logged DINO loss not below ln(4096) - 0.05, or final teacher logit spread below 0.01.
- Selection: among non-collapsed variants, the highest P_clean macro-AUC; if another variant is within 0.005 of the best, prefer KoLeo 0.1 (RAD-DINO's value), then lr x1.
- The chosen variant is used unchanged for Run 1 and Run 2 (full 18,160-step schedules). If every variant ends more than 0.02 below its own step-0 AUC, the best one is still used and the adaptation check is reported as failed.


Note (2026-09-24, before the sweep finished): an end-to-end code test ran extract_features.py and analyze.py on the 600-step verification-trial snapshots (a recipe that will not be used) with a 300-images-per-group subset. The script printed per-arm probe AUCs; with 300 images per arm they are noise, the recipe is not the one selected, and they were not used for any decision.

### Decision record: tuning sweep result (2026-09-24, applying the Amendment 5 rule, before any formal run)

| Variant | Final DINO loss | Final teacher logit spread | P_clean macro-AUC step 0 / 350 / 1000 | Mean pairwise CLS cosine at 1000 |
|---|---|---|---|---|
| V1 KoLeo 0.1, lr x1 | 7.625 | 0.135 | 0.6845 / 0.6606 / 0.6512 | 0.025 |
| V2 KoLeo 0.1, lr x0.3 | 7.781 | 0.109 | 0.6845 / 0.6630 / 0.6581 | 0.016 |
| V3 KoLeo 0.02, lr x1 | 7.377 | 0.158 | 0.6845 / 0.6959 / 0.6923 | 0.044 |
| V4 KoLeo 0, lr x1 | 7.243 | 0.172 | 0.6845 / 0.7265 / 0.7334 | 0.649 |

No variant met the collapse exclusion (all final DINO losses below 8.268 and spreads above 0.01). V4 has the highest P_clean macro-AUC (0.7334; next best 0.6923, a gap of 0.041, above the 0.005 tie margin), so V4 is selected for Run 1 and Run 2: KoLeo weight 0, learning rate x1, all other Amendment 5 settings unchanged. V4 improves on its step-0 probe (+0.049), whereas the KoLeo-0.1 variants degrade it. Limitation recorded now: V4 keeps images' CLS features closer together (mean pairwise cosine 0.65) than the released RAD-DINO (0.045); the proxy matches RAD-DINO in improving the encoder for chest X-ray findings, not in feature-space uniformity.

### Analysis log (2026-09-25, after the final analysis; no plan content changes)

- The final analysis (finished on the laptop 2026-09-25 05:19) ran scripts/analyze.py with SHA-256 39d3fa06…afb4, archived as scripts/archive/analyze_asrun_20260924.py. This is not the version hashed at Amendment 2 (16f9bd01…); the edits made on 2026-09-24 were not hash-recorded. The baseline-adjusted contrast in deltas_adjusted_*.csv was added on 2026-09-24 and is exploratory.
- Deviation found and corrected. Section 5.3 calibrates the membership scores of E100 and H_a against the crossover partner run. The code looked the partner snapshot up by step number; the two runs' schedules differ by a few steps (Run 1: 6,356 and 18,160; Run 2: 6,360 and 18,171), so every arm silently fell back to the step-0 reference. The code now pairs snapshots by schedule position (test: tests/test_analyze.py::test_mia_uses_partner_run_reference_when_step_counts_differ). Both versions will be reported. The positive control (E500, step-0 reference, as specified) is unaffected.
- Secondary endpoints 4 (probe trained on P_clean), 5 (uncertain labels counted as positive) and 6 (crossover contrast of per-image probe log-likelihood) were not part of the final analysis. Endpoint 6 is implemented as the primary's crossover contrast with the per-image log-likelihood in place of macro-AUC, with the same placebo-variance inference. All three are being run with the corrected code into results/final_v2/.
- The p-values of the unpaired contrasts come from the patient bootstrap (two-sided), not from the patient-level permutation test named in Section 4.

### Amendment 6 (2026-09-25; exploratory; written after the randomized results were seen, before any computation with RAD-DINO's DINO head)

Reason: in the randomized experiment the backbone-only score (augmentation invariance calibrated by the step-0 encoder) failed the positive control (E500 membership AUC 0.557 in Run 1 and 0.553 in Run 2), while the secondary score, the DINO loss under the run's own heads, separated exposed from held-out images (E100 0.61, E500 0.78 at the final snapshot). RAD-DINO's release includes its DINO head (dino_head.safetensors), so a head-based score can be computed on the released model. We found no published DINO head for DINOv2-B, so this score has no reference-model calibration.

Plan:
- Score (single network, because only one backbone and one head are released): for each image's 8 fixed augmented views (the Section 7 bridge views, identical seeds), CE_ij = cross-entropy between softmax(h(v_i) / 0.07) and log-softmax(h(v_j) / 0.1), where h is backbone plus DINO head; the score is minus the mean of CE_ij over i != j. No centring, no reference subtraction.
- RAD-DINO: members and non-members as in Amendment 4 (202 and 518 frontal images); membership AUC with a patient-clustered bootstrap 95% CI (10,000 resamples); the minimal detectable AUC stays about 0.567.
- Calibration in the randomized experiment: the same single-network score (teacher backbone plus teacher head) on the test arms at the final snapshot of both runs; membership AUC of each exposed arm vs H_b, uncalibrated (the setting available for RAD-DINO) and, for E100 and H_a, also calibrated by the crossover partner run.
- Reading rules, fixed now: RAD-DINO's AUC is read against the proxy's uncalibrated AUC at dose ~100. If that proxy AUC is below 0.567, a null RAD-DINO result is reported as uninformative rather than as absence of memorisation. All results of this amendment are reported as exploratory, whatever they show.

### Amendment 7 (2026-09-25; post hoc; written after seeing the RAD-DINO head-score result and before any of the checks below was computed)

Result that prompted it: under Amendment 6, RAD-DINO's single-network head score separated its member pool (CheXpert validation, 202 frontal images) from its non-member pool (CheXpert test, 518) with AUC 0.716 [0.674, 0.754], far above the detectable 0.567. The exchangeability gate used DINOv2-B features; a difference between the pools that those features do not register but the head score does would produce the same result without memorisation.

Checks, all reported as post hoc:
1. Negative control. The same single-network head score under the final teacher backbone and teacher head of each proxy run, which saw neither pool, on the same 720 images: RAD-DINO's preprocessing (shorter side 518, min-max), then the proxy's 224-pixel view transform with the same per-image seeds. Membership AUC with a patient-clustered bootstrap 95% CI, and the paired difference AUC(RAD-DINO) - AUC(proxy run r) with a patient-clustered bootstrap CI over the same images.
2. Image size. AUC of image width and of image height for pool membership; the RAD-DINO head-score AUC within the modal image size (the Amendment 4 fallback subset).

Reading rule, fixed now: the RAD-DINO result is described as evidence of memorisation only if (a) both proxy AUCs have 95% CIs that include 0.5, (b) both paired differences have 95% CIs above 0, and (c) the modal-size AUC stays above 0.567. If (a) or (b) fails, it is reported as a pool difference that the gate missed, with the proxy AUCs as its size; if only (c) fails, the size confound is reported as unresolved.

### Analysis log 2 (2026-09-26; after all results, while writing the paper; no plan content changes)

- Deviation not logged before: membership AUCs (Section 5.3) were computed for each exposed arm against H_b only, not against all of H. In Run 2, H_a is the dose-100 arm, so H_b is the only test arm unexposed in both runs; using it in both runs keeps the runs comparable. scripts/analyze.py implemented 5.3 this way from the start (before any formal run).
- The Section 5 Holm family is completed with the three endpoints computed on 2026-09-25 (endpoints 4, 5 and 6, each as a crossover contrast like the primary): the 64 tests written by analyze.py plus these three, 67 tests in all (scripts/make_paper_numbers.py).
- Never registered, reported as exploratory: the probe AUC of the proxy, DINOv2-S, DINOv2-B and RAD-DINO on the official CheXpert test set (scripts/external_reference.py, 2026-09-24), and the baseline-adjusted premium (already noted in the first analysis log).
- In every crossover contrast the patient-bootstrap interval (image sampling only) was wider than the placebo-set interval of Amendment 2. Both are reported; no conclusion changes.

### Amendment 8 (2026-09-26; post hoc; written after an internal review of the draft and before the check below was computed)

Reason: the Amendment 7 negative control scored the bridge images with the proxies' 224-pixel views, while RAD-DINO's head score used 518-pixel views. A difference between the validation and test releases that is visible only in fine detail would then separate the pools under RAD-DINO but not under the proxies. Both pools are JPEG files with identical quantization tables, mode and header (checked 2026-09-26), which rules out different JPEG encoding settings but not other fine-scale differences.

Check: recompute RAD-DINO's Amendment 6 head score on the same 720 images, views and seeds after low-pass filtering each preprocessed image: shorter side down to 224 pixels (bicubic, antialiased), then back up to its original size (bicubic). Membership AUC with the patient-clustered bootstrap 95% CI, as in Amendment 6.

Reading rule, fixed now: if the low-passed AUC is above the minimal detectable AUC (0.567) and its 95% CI excludes 0.5, the separation is reported as not explained by image content finer than 224-pixel resolution; otherwise the release-difference explanation is reported as unresolved. Either way the result is post hoc and exploratory.

### Amendment 9 (2026-09-26; post hoc; written after a second internal review and before any of the checks below was computed)

Reason: reviewers asked (1) whether the weak signal of the backbone score with the step-0 reference (membership AUC 0.52-0.56 at every dose) exceeds what two never-seen arms give, since every arm is compared with the same H_b; and (2) whether the RAD-DINO member and non-member pools differ in intensity distribution or case mix, which the DINOv2-B gate may not register.

Checks:
1. Null membership AUCs at the final snapshot: Run 1, H_a vs H_b (both never seen in Run 1); Run 2, E100 vs H_b (both never seen in Run 2). Scores: backbone augmentation invariance minus that of the step-0 encoder (the outside auditor's version), and the Amendment 6 head score (no reference). Patient-clustered bootstrap 95% CI, 10,000 resamples.
2. RAD-DINO pools (202 members, 518 non-members): out-of-fold AUC of a logistic classifier (patient-grouped 5-fold cross-validation, standardized inputs) on (a) intensity statistics of each image scaled to [0, 1] (mean, standard deviation, and the 1st, 5th, 25th, 50th, 75th, 95th and 99th percentiles) and (b) the 14 CheXpert labels; patient-clustered bootstrap 95% CI of each AUC.

Reading rules, fixed now:
1. If either null AUC of the invariance score reaches 0.52 (the smallest exposed-arm value), the auditor's signal is reported as not distinguishable from differences between random arms; otherwise it is reported as a weak exposure signal. The head-score null AUCs are reported next to its dose curve.
2. If either pool classifier reaches AUC 0.55 (the gate threshold), that difference is reported as a possible explanation of the RAD-DINO head result; otherwise the pools are reported as not separable by intensity statistics or labels.

### Amendment 10 (2026-09-26; post hoc; exploratory; written after the second internal review and before the analysis below was run)

Reason: in RAD-DINO's NIH case the probe-training and test splits of a benchmark were exposed equally often, whereas the primary contrast uses probes trained on P_base (10 exposures) or P_clean (none). Reviewers asked for the contrast with probes trained on images exposed as often as the test images.

Analysis (final snapshot of each run; features, labels and probe protocol as in Section 4; uncertain labels negative): within each of H_a and E100, patients are split into two halves; a probe trained on one half predicts the other and vice versa (two-fold patient-grouped cross-fitting), repeated over 5 fixed random splits that are the same in both runs; each image's prediction is its mean over the 5 repeats. A_X,r is the macro-AUC of these predictions for arm X in run r, so probe-training and test images always share the arm's exposure (both about 100 or both 0). Matched premium: Delta_m = 1/2 [(A_Ha,2 - A_Ha,1) + (A_E100,1 - A_E100,2)]. 90% and 95% CIs from a patient bootstrap within H_a and E100 (10,000 resamples), conditional on the fitted probes.

Reading rule, fixed now: if the 90% CI lies below +0.005 (0.5 points), a premium above 0.5 points is reported as excluded also with matched probes; if it lies above +0.005, a premium is reported for the matched setting; otherwise the matched result is reported as inconclusive. The result is exploratory whatever it shows.
