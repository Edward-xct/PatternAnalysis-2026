# Patient-Level Alzheimer’s Disease Classification from ADNI MRI with ConvNeXt

**Student:** Chongtao Xie  
**Student number:** 49859083  
**Course:** COMP3710 Pattern Recognition  
**Project:** ADNI AD-versus-CN classification — Hard difficulty

## Executive summary

This project classifies Alzheimer’s disease (AD) versus cognitively normal controls (CN) from preprocessed ADNI sagittal MRI slices. The project’s ConvNeXt-Tiny implementation uses ImageNet initialisation and ADNI fine-tuning. Mean logits across all slices and visits produce one prediction per patient; validation-fitted temperature scaling and a triage rule provide calibrated probabilities and referral decisions. SmallCNN and randomly initialised ConvNeXt-Tiny provide baselines on the same patient split.

![Pipeline and ConvNeXt-Tiny architecture](figures/report/pipeline_overview.png)

The final model correctly classified **85 of 102 held-out patients (83.33% accuracy; unadjusted 95% CI 74.9–89.3%)**, exceeding the 0.80 target as a point estimate, with **0.7931 macro-F1** and **0.8546 AUROC**. These exploratory results follow adaptive test-set reuse (Section 7). The unadjusted paired bootstrap favours its AUROC over both baselines; its accuracy gain over scratch ConvNeXt is not significant (McNemar p = 0.30).

The ImageNet model had the most overconfident raw outputs among the three evaluated models: 12 of its 17 errors had confidence of at least 0.9. Temperature scaling (T = 3.23) reduced test ECE from 0.144 to 0.063, the lowest of the three. Confidence separated correct from incorrect decisions modestly across models (AUROC 0.67–0.70). The validation-fitted reject option decided 83% of patients automatically at 85.9% accuracy and cleared 8 of 33 AD patients as normal without review. These findings support research and clinician-facing decision support, with **clinician review of NC predictions**.

![Patient-level test comparison](figures/report/test_metric_comparison.png)

## 1. Problem and research questions

ADNI provides multiple MRI scans per patient and 20 two-dimensional slices per scan. This creates two important engineering challenges:

1. slices and repeated visits from the same patient are correlated, so a slice-level random split would leak identity information;
2. a high aggregate accuracy can conceal overconfident or clinically costly errors.

The project therefore asks:

- Does ConvNeXt-Tiny improve patient-level performance over SmallCNN, and are the gains statistically meaningful on 102 test patients?
- Does ImageNet initialisation improve ConvNeXt optimisation and generalisation?
- Does ConvNeXt quantify uncertainty better than the baseline: are its probabilities calibrated, and does its confidence separate correct from incorrect decisions?
- Can a validation-selected reject option safely automate confident cases and route ambiguous patients to human review?
- Is the performance improvement justified by the additional compute cost?

In the source code, the normal class is named `NC`; it is equivalent to CN in this report.

## 2. Feasibility review

This is the written feasibility review presented during the compulsory tutor check-off. The tutor confirmed that the scope, patient-level split, model comparison, and proposed evaluation were appropriate. The “Final outcome” column was added after the check-off.

### User need, scope, and acceptance criteria

The intended user is a research team investigating an MRI-based screening assistant. The prototype returns one AD/CN prediction, calibrated confidence and an optional referral decision per patient to support clinician review.

| Acceptance criterion | Target | Final outcome |
|---|---:|---:|
| Patient leakage | zero subject overlap | achieved |
| Held-out patient accuracy | at least 0.80 | **0.8333** (point estimate; exploratory, see Section 7) |
| Evaluation | accuracy, class metrics, macro-F1, AUROC, calibration | achieved |
| GPU feasibility | peak training VRAM below 6 GB | **4.12 GB** |
| Ambiguity handling | validation-only calibration and referral thresholds | procedure achieved; safety not achieved: on test, 8 of 33 AD patients were cleared without review and accepted-case AD sensitivity was 0.71 against the 0.90 validation target |

### Model choice and course concepts

The project uses a hard-difficulty ConvNeXt-Tiny classifier. ConvNeXt modernises convolutional networks with a patchifying stem, depthwise convolutions, LayerNorm, inverted bottlenecks, GELU activations, residual connections, stochastic depth, and layer scale. These elements connect directly to course concepts including convolutional feature extraction, normalisation, regularisation, transfer learning, class imbalance, calibration, and rigorous held-out evaluation. SmallCNN provides a transparent baseline, while the scratch and ImageNet-initialised ConvNeXt runs separate architecture capacity from transfer-learning effects.

### Preliminary feasibility evidence

The data audit found 30,520 valid JPEG slices, 1,526 scans, and 680 patients, with exactly 20 slices per scan. It also found 216 patients shared between the supplied directory-level train and test sets. This made a new patient-level split necessary. CPU and GPU smoke tests confirmed valid `(N, 3, 224, 224)` batches, two-class logits, finite gradients, checkpoint saving, and full patient aggregation. ConvNeXt used approximately 4.12 GB peak training VRAM on an A100, within the resource budget.

### Risks, compute budget, and fallback

The main risks were patient leakage, class imbalance, overfitting of a 27.8M-parameter model, validation overfitting during calibration, and long GPU queue times. The planned budget was one A100 GPU, batch size 64, and at most 30 epochs per main experiment. The fallback was the 0.39M-parameter SmallCNN, which had already completed end-to-end training and inference. Additional safeguards included a deterministic split, patient-aware balanced sampling, validation-only checkpoint selection, explicit test locking, automated tests, and preservation of every experiment log.

**Next planned experiment.** A failure autopsy of 3–5 test errors of the frozen ImageNet-initialised model, followed by completion of the report, with no further tuning based on test results.

## 3. Data audit and patient-level split

The course dataset is located at:

```text
/home/groups/comp3710/ADNI
```

JPEG filenames encode a scan identifier and slice index. `meta_data_with_label.json` maps each scan to a subject and diagnosis. Metadata label 2 is mapped to AD (model target 1), and label 0 is mapped to NC (model target 0).

The supplied directories had no repeated scan IDs, but **216 patients occurred in both directories**. Reusing that split would allow scans from one person to influence both training and evaluation. `scripts/audit_data.py` therefore pools the scans and calls `stratified_subject_split()` in `dataset.py`, which performs a deterministic, diagnosis-stratified 70/15/15 patient split using seed 3710. Every slice, scan, and visit from one subject remains in exactly one split.

| Split | Patients | AD | NC | Scans | Slices |
|---|---:|---:|---:|---:|---:|
| Train | 476 | 155 | 321 | 1,081 | 21,620 |
| Validation | 102 | 33 | 69 | 228 | 4,560 |
| Test | 102 | 33 | 69 | 217 | 4,340 |
| **Total** | **680** | **221** | **459** | **1,526** | **30,520** |

The 70/15/15 split balances training data and held-out evaluation: 476 patients train the network, 102 support checkpoint selection, temperature fitting and threshold selection, and 102 form the test set. Subject overlap is zero. Increasing the test share would reduce the training cohort; patient-level k-fold cross-validation would increase GPU cost beyond the planned budget. Test-set use is documented in Section 7. Generated manifests contain machine-specific paths and are excluded from Git.

## 4. Preprocessing and patient aggregation

**Example input.** Each patient contributes one to five MRI visits, with 20 consecutive skull-stripped sagittal grayscale JPEG slices per visit. Each scan spans a 19-slice window within original indices 78–114, centred on the midline. The montages show the corpus callosum, lateral ventricles, brainstem, cerebellum and medial cortex. Section 10 discusses the possible limits of hippocampal coverage. Figure 1 illustrates the input format; course MRI images remain private.

`dataset.py` validates the manifest and rejects any patient overlap. Each grayscale slice is resized to 224 × 224, copied into three channels, converted to a tensor, and normalised with ImageNet channel statistics. Training applies conservative random affine augmentation (±5° rotation, 2% translation, 0.95–1.05 scale); validation and test transforms are deterministic.

Training is slice-based for computational efficiency. A patient-and-class-aware weighted sampler prevents the larger NC class and patients with many visits from dominating updates. During evaluation, the model processes all slices and all scans in the selected split. For patient `p`, logits are aggregated before softmax:

```text
patient_logit(p) = mean(logit(slice) for every slice from patient p)
patient_probability(p) = softmax(patient_logit(p))
```

All headline metrics are patient-level.

## 5. Models

### 5.1 SmallCNN baseline

SmallCNN stacks four blocks of 3 × 3 convolution, BatchNorm, ReLU, and 2 × 2 max pooling (32, 64, 128 and 256 channels), followed by global average pooling, dropout, and a two-class linear head. Each 3 × 3 convolution only sees a small neighbourhood, so large anatomical patterns must be assembled through repeated pooling. It contains **389,410 trainable parameters** and supplies a low-cost reference for the complete pipeline.

### 5.2 ConvNeXt-Tiny: working principles

`ConvNeXtTiny` in `modules.py` implements the architecture of Liu et al. [1], which redesigns a ResNet using choices borrowed from vision transformers (Figure 1b–c).

- **Patchify stem.** A 4 × 4 convolution with stride 4 turns the 224 × 224 slice into a 56 × 56 grid of 96-channel patch embeddings, followed by LayerNorm.
- **Four stages.** Stage depths are `[3, 3, 9, 3]` with `[96, 192, 384, 768]` channels at 56, 28, 14 and 7 pixels resolution. Between stages, LayerNorm and a 2 × 2 stride-2 convolution halve the resolution and double the channels.
- **ConvNeXt block.** Each block computes

```text
y = x + DropPath( γ * W2 · GELU( W1 · LayerNorm( DWConv7x7(x) ) ) )      (* = per-channel scaling)
```

  The 7 × 7 depthwise convolution mixes information spatially, one channel at a time, which gives a large receptive field cheaply, similar to a local attention window. LayerNorm and the two pointwise linear layers (`W1`: C → 4C, `W2`: 4C → C) form an inverted bottleneck that mixes channels. The learnable per-channel layer scale `γ` starts at 1e-6, so every block begins close to the identity and the 18-block network trains stably. Stochastic depth (DropPath) randomly skips whole residual branches during training, with a rate rising linearly from 0 to 0.1 across the blocks, which regularises the deep model.
- **Head.** Global average pooling, LayerNorm, dropout 0.3, and a linear layer produce the NC/AD logits.

The model contains **27,821,666 trainable parameters**, about 71 times more than SmallCNN. Section 5.3 explains how the same class is initialised from ImageNet.

### 5.3 Final ImageNet-initialised ConvNeXt-Tiny

The final model is the same `ConvNeXtTiny` class from `modules.py` described in Section 5.2; only its starting weights differ. `load_imagenet_weights()` downloads torchvision’s `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, renames every tensor to this project’s layer names with `convert_torchvision_convnext_state_dict()`, discards the 1,000-class ImageNet head, and keeps a freshly initialised two-class head. Loading is strict: every one of the 180 backbone tensors must match, otherwise training stops. The whole network is then fine-tuned on ADNI, so the architecture and parameter count (27,821,666) are identical to the scratch model.

The first ImageNet run (job 650741) used torchvision’s ConvNeXt class. Its frozen checkpoint was converted into `ConvNeXtTiny` and re-evaluated without retraining. Two unit tests check backbone mapping and logit agreement within floating-point tolerance (`rtol = 1e-4`, `atol = 1e-5`). For the real checkpoint, `scripts/compare_predictions.py` found exactly equal stored patient logits for all 102 validation and 102 test patients, unchanged predicted classes, and exact agreement across all 129 numeric entries of `evaluation.json`. The comparison summaries are in `figures/report/checkpoint_equivalence_validation.json` and `checkpoint_equivalence_test.json`. Reported final-model results use this converted checkpoint in the project’s own implementation.

**Reproducing the reported results.** Re-evaluation uses the frozen checkpoint. Retraining with `scripts/train_convnext_imagenet.slurm` creates a new run whose results can differ: the torchvision and project constructors consume the random-number stream differently, giving different initial two-class heads even with seed 3710, and mixed-precision GPU execution is not guaranteed to be deterministic. `figures/report/final_checkpoint_provenance.json`, generated by `scripts/checkpoint_provenance.py`, records the original job ID, Git commit, timestamps, training configuration, stored epoch and checkpoint SHA-256.

ImageNet initialisation provides reusable low-level edge, texture and shape filters for fine-tuning the 27.8M-parameter network. The pretrained weights are attributed in Section 15.

## 6. Training and model selection

**What training optimises.** Each training example is one slice, labelled with its patient’s diagnosis, and every model minimises the unweighted two-class cross-entropy between the slice logits and that label. Class imbalance and repeated visits are handled by the sampler rather than the loss. `make_subject_and_class_balanced_sampler()` in `dataset.py` gives each slice the weight 1 / (number of slices of its patient × number of patients in its class) and draws 21,620 slices with replacement per epoch. In expectation, half of the draws are AD and half NC, and every patient within a class contributes equally, however many visits they have.

Validation and test metrics use patient-level averages of all slice logits (Section 4). The slice-level training objective assigns the patient diagnosis to each correlated slice, including slices with no visible disease change. Patient-level averaging combines evidence across slices and visits. Training curves therefore show slice-level loss, while checkpoint selection uses patient-level macro-F1.

All three models use the same patient split, transforms, balanced sampler, patient aggregation, seed 3710, AdamW optimiser, cosine learning-rate decay, weight decay `1e-4`, dropout `0.30`, batch size 64, and mixed precision on CUDA. The best checkpoint is selected by **validation macro-F1**, not test accuracy.

| Model | Initial LR | Epoch budget | Best epoch | Validation accuracy | Validation macro-F1 |
|---|---:|---:|---:|---:|---:|
| SmallCNN | 3e-4 | 30 | 9 | 0.8529 | 0.8133 |
| ConvNeXt-Tiny, scratch | 3e-4 | 30 | 16 | 0.8333 | 0.8081 |
| ConvNeXt-Tiny, ImageNet | 5e-5 | 20 | 9 | **0.8824** | **0.8583** |

The pretrained network uses a smaller learning rate and shorter schedule to preserve useful starting features. This engineering comparison jointly varies initialisation, learning rate and epoch budget. Fixed budgets cap cluster usage, and validation macro-F1 selects the checkpoint.

`train.py` redraws `training_curves.png` after every epoch (training and validation loss, patient-level validation accuracy, macro-F1 and AUROC, and the learning rate), so progress can be inspected while a job is running. For the ImageNet model, training loss falls close to zero while validation loss rises from epoch 2, even though validation AUROC keeps improving. The network becomes increasingly overconfident rather than less discriminative, which anticipates the large calibration temperature in Section 8.

Scratch ConvNeXt was also trained with seeds 2026 and 4985908 on the same split to assess seed sensitivity. Best validation macro-F1 was 0.7760 and 0.7973, respectively, compared with 0.8081 for seed 3710, spanning about 0.03 across these runs. The two additional runs were evaluated on validation only.

![Training curves](figures/report/training_curves.png)

## 7. Held-out test results

The following results include all 102 patient-disjoint test subjects. Smoke-test outputs are excluded.

**Test-set use.** Evaluation proceeded in this order: (1) SmallCNN and scratch ConvNeXt were tested once each and fell below 0.80 accuracy. (2) A SmallCNN + scratch-ConvNeXt logit ensemble, fitted on validation, was tested once and reached 0.8039 accuracy (82/102), 0.7591 macro-F1 and 0.8019 AUROC. (3) After seeing the scratch model’s test result, an ImageNet-initialised ConvNeXt was trained, selected on validation and tested once. (4) The frozen ImageNet checkpoint was re-evaluated in `ConvNeXtTiny` to verify implementation equivalence. Checkpoints, temperatures and thresholds were selected on validation. Adapting model development after earlier test evaluations makes the final results exploratory, with selection bias of unknown size. The reported intervals and tests are unadjusted for this reuse.

| Test metric | SmallCNN | ConvNeXt scratch | ConvNeXt ImageNet |
|---|---:|---:|---:|
| Accuracy | 0.7549 (77/102) | 0.7843 (80/102) | **0.8333 (85/102)** |
| Macro-F1 | 0.6888 | 0.7536 | **0.7931** |
| AUROC | 0.7910 | 0.8081 | **0.8546** |
| AD precision | 0.6818 | 0.6667 | **0.8333** |
| AD recall | 0.4545 | **0.6667** | 0.6061 |
| NC precision | 0.7750 | **0.8406** | 0.8333 |
| NC recall | 0.8986 | 0.8406 | **0.9420** |

![Test confusion matrices](figures/report/test_confusion_matrices.png)

The final model’s confusion matrix is TN = 65, FP = 4, FN = 13, TP = 20. The scratch model retained the highest AD recall, illustrating that model selection depends on the operational cost assigned to false negatives rather than accuracy alone.

**Statistical uncertainty.** Accuracy and AD-recall intervals use Wilson scores. AUROC intervals and paired differences use a class-stratified patient bootstrap with 5,000 resamples, evaluating all three models on the same resampled patients (`uncertainty_analysis.py`). These unadjusted analyses quantify uncertainty within the 102-patient test set.

| Model | Accuracy (95% Wilson CI) | AUROC (95% bootstrap CI) | AD recall (95% Wilson CI) |
|---|---:|---:|---:|
| SmallCNN | 0.755 (0.663–0.828) | 0.791 (0.694–0.880) | 0.455 (0.298–0.620) |
| ConvNeXt scratch | 0.784 (0.695–0.853) | 0.808 (0.711–0.894) | 0.667 (0.496–0.802) |
| ConvNeXt ImageNet | 0.833 (0.749–0.893) | 0.855 (0.769–0.929) | 0.606 (0.437–0.753) |

| ImageNet ConvNeXt minus | Accuracy difference (95% CI) | Exact McNemar p | AUROC difference (95% CI) |
|---|---:|---:|---:|
| SmallCNN | +0.078 (+0.010 to +0.147) | 0.057 (11 vs 3 discordant) | +0.064 (+0.009 to +0.119) |
| ConvNeXt scratch | +0.049 (−0.020 to +0.118) | 0.302 (10 vs 5 discordant) | +0.047 (+0.002 to +0.097) |

The unadjusted bootstrap intervals for both AUROC gains exclude zero, narrowly for the comparison with scratch ConvNeXt. The accuracy gain over SmallCNN is borderline: its bootstrap interval excludes zero, while McNemar p = 0.057. The accuracy difference from scratch ConvNeXt remains within sampling uncertainty. The observed accuracy exceeds 0.80; its 95% interval extends down to 74.9%.

![Validation-to-test gap](figures/report/validation_test_gap.png)

All three models performed worse on test than validation. Selection optimism, sampling variation and case differences are possible explanations. SmallCNN had the largest accuracy drop (0.853 → 0.755). The ImageNet model dropped 0.049 in accuracy and 0.099 in AUROC (0.954 → 0.855).

## 8. Uncertainty, calibration and reject option

### 8.1 Does the larger model quantify uncertainty better?

A single temperature `T` is fitted per model by minimising the negative log-likelihood of the validation patients’ logits (LBFGS on log T). On test, the calibrated probability and the expected calibration error (ECE) are

```text
p_T(AD) = softmax(patient_logit / T)[AD]
ECE     = Σ_b (n_b / N) · | accuracy_b − mean_confidence_b |    (10 equal-width confidence bins)
```

For two classes, dividing logits by `T` changes how extreme the probabilities are but never which class is predicted, so accuracy and AUROC are unchanged. The last two columns below test the central question of the dilemma: whether a model’s confidence is higher for its correct decisions than for its errors, and how many errors are made with confidence of at least 0.9.

| Model | T | ECE raw → cal | NLL raw → cal | Raw conf ≥ 0.9: patients (accuracy) | Confidence AUROC for correctness | Errors with conf ≥ 0.9, raw → cal |
|---|---:|---:|---:|---:|---:|---:|
| SmallCNN | 0.87 | 0.098 → 0.123 | 0.566 → 0.594 | 36 (0.861) | 0.690 | 5 → 7 of 25 |
| ConvNeXt scratch | 1.67 | 0.128 → 0.108 | 0.619 → 0.508 | 65 (0.877) | 0.673 | 8 → 6 of 22 |
| ConvNeXt ImageNet | 3.23 | 0.144 → 0.063 | 0.997 → 0.441 | 89 (0.865) | 0.701 | 12 → 7 of 17 |

![Reliability diagrams for the three models](figures/report/model_reliability.png)

Three findings answer the dilemma.

1. **ImageNet ConvNeXt had the most overconfident raw outputs in these runs.** It assigned confidence of at least 0.9 to 89 of 102 patients, with 86.5% accuracy in that group; 12 of its 17 errors met that confidence level. It had the largest fitted temperature (3.23) and worst raw NLL, consistent with its near-zero training loss (Section 6).
2. **Temperature scaling improved ImageNet calibration and sharpened SmallCNN probabilities.** After scaling, ImageNet ConvNeXt had the lowest ECE (0.063), NLL (0.441) and Brier score (0.131, from 0.145). SmallCNN was slightly underconfident on validation (T = 0.87); scaling sharpened its probabilities and worsened test ECE (0.098 → 0.123). Validation-fitted temperatures transferred unevenly to test.
3. **Confidence-based error detection remained modest.** Correctness AUROC was 0.67–0.70 across the three models, based on 17–25 errors per model. Even after calibration, 7 of the final model’s 17 errors had confidence of at least 0.9. These small error counts limit the precision of model comparisons. ImageNet ConvNeXt achieved better observed AD-versus-NC ranking, with no clear improvement in recognising its own errors.

### 8.2 Reject option

The three-way rule refers a patient to a clinician unless the calibrated probability is decisive:

```text
P(AD) ≤ τ_NC  → accept as NC
P(AD) ≥ τ_AD  → accept as AD          (τ_NC ≤ 0.5 ≤ τ_AD)
otherwise     → refer
```

`evaluate.py` searches every validation probability as a candidate threshold and keeps the pair with the **highest coverage** that still reaches 0.90 AD sensitivity and 0.90 NC specificity among accepted validation patients, with at least 50% coverage. For the final model this gives τ_NC = 0.1373 and τ_AD = 0.5.

| Final model, selective result | Validation | Test |
|---|---:|---:|
| Accepted patients | 88/102 | 85/102 |
| Referred patients | 14/102 | 17/102 |
| Coverage | 0.8627 | 0.8333 |
| Selective accuracy | 0.9432 | 0.8588 |
| Accepted-case AD sensitivity | 0.9231 | 0.7143 |
| Accepted-case NC specificity | 0.9516 | 0.9298 |

The same procedure applied to each model gives the following frozen operating points on test. “AD auto-cleared” counts AD patients who were accepted as NC without review, the most costly outcome in screening.

| Model | τ_NC | τ_AD | Coverage | Selective accuracy | AD auto-cleared | NC auto-flagged | Errors referred |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmallCNN | 0.141 | 0.500 | 0.725 | 0.811 | 7/33 (21.2%) | 7/69 (10.1%) | 11/25 |
| ConvNeXt scratch | 0.138 | 0.674 | 0.667 | 0.824 | 6/33 (18.2%) | 6/69 (8.7%) | 10/22 |
| ConvNeXt ImageNet | 0.137 | 0.500 | 0.833 | 0.859 | 8/33 (24.2%) | 4/69 (5.8%) | 5/17 |

![Risk-coverage curves for the three models](figures/report/model_risk_coverage.png)

- **The ImageNet model gives the best automation trade-off on average.** It decides 83% of patients automatically at the highest selective accuracy, and its risk-coverage curve lies on or below both baselines at every coverage.
- **It also auto-clears the most AD patients.** Eight of its 13 false negatives had calibrated P(AD) below 0.137 and were never referred. The five that were referred had P(AD) between 0.150 and 0.496. Accepted-case AD sensitivity fell from 0.92 on validation to 0.71 on test.
- **The rule is one-sided for two of the three models.** Because the search maximises coverage, it lowers τ_AD to 0.5 whenever validation false positives are rare. Every patient predicted as AD is then accepted, which is why the final model’s false positives (including Case B in Section 10) were never referred.
- **Fixed thresholds cost accuracy at equal coverage.** For the scratch ConvNeXt and the final model, the frozen operating point lies above the plain confidence-ranked curve (0.141 versus 0.129 selective risk for the final model at 83% coverage). The class-specific thresholds accept some less confident patients and refer some more confident ones, and the sensitivity and specificity targets met on validation did not hold on test. SmallCNN’s frozen point lies on its curve.

Clinician review remains necessary alongside confidence-based referral.

## 9. Resource profile

Measurements use Rangpur A100 jobs. Training time sums the train-plus-validation loops in each saved history. Inference latency measures batched GPU forward passes with mixed precision and batch size 64, excluding image loading, preprocessing and scheduler queue time. Per-patient latency is an amortised forward-pass estimate: per-slice time multiplied by the average of 42.5 slices per test patient.

| Model | Trainable parameters | Peak training VRAM | Peak inference VRAM | Time per epoch | Training loop | Forward ms/slice | Forward ms/patient (amortised) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmallCNN | 0.389M | **1.27 GB** | **0.48 GB** | **31.6 s** | 15.80 min (30 epochs) | **0.577** | **24.6** |
| ConvNeXt scratch | 27.822M | 4.12 GB | 0.58 GB | 35.3 s | 17.63 min (30 epochs) | 0.791 | 33.7 |
| ConvNeXt ImageNet | 27.822M | 4.12 GB | 0.58 GB | 34.1 s | **11.35 min (20 epochs)** | 0.769 | 32.7 |

![Resource profile](figures/report/resource_profile.png)

ConvNeXt has about 71 times more parameters, while epoch times range from 31.6 to 35.3 s. Similar times suggest substantial JPEG-decoding and augmentation overhead on four CPU workers; the dominant cost was not profiled. The ImageNet run used 20 epochs versus 30 for the baselines. Its added cost over SmallCNN is 2.85 GB of training memory and about 8 ms of amortised forward time per patient. Training remained within the 6 GB feasibility budget.

## 10. Qualitative failure autopsies

The final model made 17 errors: 13 false negatives (AD → NC) and 4 false positives (NC → AD). `failure_cases.py` selected the four most confident false negatives and the most confident false positive, creating a private nine-slice montage of one scan per case. Subject identifiers and course MRI images remain private. Cases A–E follow the selection order; cross-model values come from `uncertainty_analysis.py`.

| Case | Error | Scans / slices; slices voted AD | Calibrated P(AD): SmallCNN / scratch / ImageNet | Final-model triage |
|---|---|---|---:|---|
| A | AD → NC | 4 / 80; 0 | 0.042 / 0.044 / 0.029 | accepted as NC |
| B | NC → AD | 3 / 60; 48 | 0.803 / 0.888 / 0.778 | accepted as AD |
| C | AD → NC | 1 / 20; 0 | 0.230 (referred) / 0.082 / 0.034 | accepted as NC |
| D | AD → NC | 2 / 40; 0 | 0.080 / 0.038 / 0.035 | accepted as NC |
| E | AD → NC | 1 / 20; 0 | 0.068 / 0.082 / 0.049 | accepted as NC |

**Errors are shared across architectures.** Both baselines also misclassified every selected case; SmallCNN referred Case C. Across the test set, 10 of the final model’s 17 errors and 9 of its 13 false negatives were shared by all three models, while 1 error was unique to ImageNet ConvNeXt. These shared errors identify patients that were difficult across the three evaluated models.

**Failure hypotheses.** A non-radiologist inspected the montages. The observations below motivate engineering tests; anatomical coverage and causal explanations remain unverified.

- **Limited field of view (all cases).** Every scan contributes near-midline sagittal slices (Section 4). Hippocampal and medial temporal lobe atrophy is an established structural MRI marker of AD [8]. Slice indices and montages suggest partial hippocampal coverage in this window. If key lateral cues are absent, the model must rely on ventricles, the corpus callosum and medial cortex, where AD-related change may be less conspicuous. In Cases A, C, D and E, every slice from every scan voted NC: coherent patient-level misses.
- **Normal-ageing confounder (Case B).** The false positive shows relatively large lateral ventricles in the midline slices. Ventricular enlargement and hippocampal atrophy occur in AD and normal ageing [9], providing a plausible explanation for 48 of its 60 slices voting AD.
- **Intensity artefacts (Cases C and E).** Several slices show a hazy, brighter band over part of the cortex. Possible explanations include acquisition differences or intensity non-uniformity of the kind addressed by N4 bias-field correction [10]. The artefacts were not quantified or tested as causes of error. Both ConvNeXts stayed confident on Case C while SmallCNN referred it.
- **Repeated visits do not fix a systematic miss (Case A).** Four scans and 80 slices all pointed the same wrong way. Averaging more visits reduces random slice-level noise, but it cannot recover a cue that the model fails to detect in every visit.

**Engineering responses.** Test the field-of-view hypothesis with lateral and whole-volume slices or a 3D model that includes the medial temporal lobes, and check whether these patients are classified correctly. Test whether bias-field correction or intensity augmentation changes predictions for Cases C and E. Replace equal slice averaging with attention pooling that identifies influential slices, and flag intensity artefacts or disagreement between visits for mandatory review. All five selected cases were accepted by the frozen rule, showing that average calibration gains leave confident errors unflagged.

## 11. Recommendation to the project manager

**Use ImageNet-initialised ConvNeXt for research and clinician-facing decision support.** Compared with SmallCNN, it provides:

- better ranking of AD versus NC (AUROC +0.064; the unadjusted bootstrap interval excludes zero);
- the best calibrated probabilities after temperature scaling (ECE 0.063);
- the best automation trade-off (83% coverage at 85.9% accuracy, against 73% at 81.1% for SmallCNN).

The added cost is 2.85 GB of training memory and about 8 ms of amortised forward time per patient. Section 7 reports the exploratory statistical comparisons.

**Require clinician review of NC predictions.** The frozen rule auto-cleared 8 of 33 AD patients. The operating policy is:

1. Use the model as a second reader and prioritisation aid. Review AD predictions and referred patients first; keep NC predictions in the normal clinician-review workflow.
2. Before any wider use, re-select τ_NC on a larger validation cohort (for example with cross-validation) to meet an explicit target for AD patients cleared without review, and make the rule two-sided so confident AD calls can also be audited.
3. Automatically refer scans with intensity artefacts or with disagreement between visits.

Prioritise sensitivity and robustness in the next experiments:

- include lateral slices or a 3D model so that the hippocampi are certainly included;
- train with a cost-sensitive patient-level objective;
- recalibrate on more patients;
- validate on an external site, with subgroup and scanner analysis.

## 12. Usage and reproducibility

### Environment

The tested Rangpur environment is Python 3.11.15, PyTorch 2.13.0, torchvision 0.28.0, Pillow 12.3.0, NumPy 2.4.6, and matplotlib 3.11.2. Exact versions are pinned in `requirements.txt`.

Set up once, from the repository root on a Rangpur login node:

```bash
cd "$HOME/PatternAnalysis-2026"
source "$HOME/miniconda3/bin/activate"
conda create -n torch python=3.11 -y
conda activate torch
python -m pip install -r recognition/ConvNeXt_ADNI_s4985908/requirements.txt
mkdir -p recognition/ConvNeXt_ADNI_s4985908/logs
python -c "from torchvision.models import ConvNeXt_Tiny_Weights as W; W.IMAGENET1K_V1.get_state_dict(progress=True)"
```

- **Use one environment everywhere.** Every later command assumes `conda activate torch` in the current shell and the repository root as the working directory. The Slurm scripts activate the same environment themselves with `source "$HOME/miniconda3/bin/activate"` and `conda activate torch`, so the environment must be named `torch` and Miniconda must be installed in `$HOME/miniconda3`; otherwise edit those two lines in `scripts/*.slurm`.
- **Create `logs/` before the first `sbatch`.** The Slurm scripts write their output to `recognition/ConvNeXt_ADNI_s4985908/logs/`. The folder is Git-ignored and therefore absent from a fresh clone, and Slurm cannot open a log file in a missing folder.
- **Cache the ImageNet weights once.** The last line downloads them into the PyTorch cache on the login node, so the ImageNet training job does not depend on internet access from a compute node.

### Audit, tests, and training

```bash
python recognition/ConvNeXt_ADNI_s4985908/scripts/audit_data.py
sbatch recognition/ConvNeXt_ADNI_s4985908/scripts/run_tests.slurm
sbatch recognition/ConvNeXt_ADNI_s4985908/scripts/train_smallcnn.slurm
sbatch recognition/ConvNeXt_ADNI_s4985908/scripts/train_convnext.slurm
sbatch recognition/ConvNeXt_ADNI_s4985908/scripts/train_convnext_imagenet.slurm
```

The latest automated run passed all **fourteen tests**. They cover the split function on a synthetic cohort (sizes, determinism, invalid ratios), patient-disjoint splits on the real manifest, label consistency, 20 slices per scan, data shapes, all three model constructors and forward passes, equivalence within floating-point tolerance between a randomly weighted torchvision ConvNeXt and the converted `ConvNeXtTiny`, and invalid-input rejection.

Training curves can also be redrawn from any saved run without retraining:

```bash
python recognition/ConvNeXt_ADNI_s4985908/train.py \
  --plot-history <OUTPUT_DIR>/history.json
```

### Example inference

```bash
python recognition/ConvNeXt_ADNI_s4985908/predict.py \
  --checkpoint <OUTPUT_DIR>/best.pt \
  --split validation \
  --output-dir <PREDICTION_DIR> \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda \
  --plots
```

The output includes a row per slice, a row per patient, and a JSON summary. A typical patient row contains the subject key, target class, aggregated NC/AD logits, AD probability, predicted class, number of scans, and number of slices. With `--plots`, the script also saves `<split>_patient_predictions.png`, a non-identifying figure of every patient’s P(AD) by true class next to the confusion matrix. It then prints the patient accuracy and a few example rows; for the test split:

```text
patient_accuracy=85/102 (0.8333)
example patients (target -> predicted, P(AD), scans, slices):
  patient 1: NC -> NC, P(AD)=0.015, scans=3, slices=60
  patient 4: AD -> AD, P(AD)=1.000, scans=5, slices=100
```

Test inference requires `--allow-test` and rejects partial test runs.

### Calibration and final test evaluation

```bash
MPLCONFIGDIR=$HOME/tmp/matplotlib \
python recognition/ConvNeXt_ADNI_s4985908/evaluate.py \
  --validation-csv <VALIDATION_DIR>/validation_patient_predictions.csv \
  --test-csv <TEST_DIR>/test_patient_predictions.csv \
  --output-dir <FINAL_RESULTS_DIR> \
  --expected-validation-patients 102 \
  --expected-test-patients 102 \
  --plots \
  --allow-test
```

### Cross-model analysis and report figures

`uncertainty_analysis.py` reads the three `evaluate.py` output folders and writes the calibration, reject-option, significance, error-overlap and failure-case tables (`model_uncertainty.md` and `.json`) and two figures. It never writes subject identifiers. `scripts/make_pipeline_figure.py` draws Figure 1, and `report_figures.py` draws the remaining aggregate figures.

```bash
P=recognition/ConvNeXt_ADNI_s4985908
python $P/uncertainty_analysis.py \
  --smallcnn-results <SMALLCNN_RESULTS_DIR> \
  --scratch-results <SCRATCH_RESULTS_DIR> \
  --pretrained-results <IMAGENET_RESULTS_DIR> \
  --output-dir $P/figures/report
python $P/scripts/make_pipeline_figure.py --output-dir $P/figures/report
```

`scripts/compare_predictions.py` compares two patient-level prediction runs (and optionally two `evaluation.json` files). It reports the largest logit and probability differences in scientific notation, the number of patients with exactly equal stored logits, and any change in predicted class; `--output-json` saves this summary without subject identifiers.

Slurm scripts record job ID, host, Git commit, timestamps, GPU details, and output directory. `scripts/checkpoint_provenance.py` turns that log, the run’s `config.json` and the checkpoint’s SHA-256 hash into a committed provenance record (Section 5.3). Seed 3710 is used for the split, the reported primary experiments, and the bootstrap. Raw data, manifests with machine paths, checkpoints, logs, patient prediction files, and private MRI montages are excluded from Git.

## 13. Repository structure

```text
ConvNeXt_ADNI_s4985908/
|-- README.md
|-- dataset.py
|-- modules.py
|-- train.py
|-- predict.py
|-- evaluate.py
|-- uncertainty_analysis.py
|-- ensemble.py
|-- convnext_ensemble.py
|-- failure_cases.py
|-- report_figures.py
|-- requirements.txt
|-- scripts/
|   |-- audit_data.py
|   |-- checkpoint_provenance.py
|   |-- compare_predictions.py
|   |-- make_pipeline_figure.py
|   |-- run_tests.slurm
|   |-- train_smallcnn.slurm
|   |-- train_convnext.slurm
|   `-- train_convnext_imagenet.slurm
|-- tests/
|   |-- test_patient_split.py
|   |-- test_dataset_shapes.py
|   `-- test_model_forward.py
|-- figures/report/         # aggregate, non-identifying report figures
|-- artifacts/              # generated locally; Git-ignored
|-- logs/                   # generated locally; Git-ignored
`-- outputs/                # checkpoints and patient results; Git-ignored
```

The ensemble scripts are retained as documented secondary experiments. The final reported system is the single ImageNet-initialised ConvNeXt-Tiny, because it outperformed the validation-fitted ensemble (0.8039 test accuracy) without requiring multiple models at inference.

## 14. Limitations and ethics

The evaluation covers the ADNI research cohort. Generalisation across clinical prevalence, scanners, demographics, comorbidities and external institutions is uncertain. Twenty near-midline sagittal slices per scan provide limited volumetric context and may omit much of the medial temporal lobes. Repeated visits are correlated within patients, and diagnosis labels are not pathology ground truth. ImageNet features come from natural images.

The 102-patient test cohort gives wide confidence intervals (Section 7). Calibration and referral thresholds fitted on 102 validation patients transferred incompletely to test. Section 7 documents adaptive test-set reuse and the unadjusted statistical analyses. Failure-case interpretations are based on non-radiologist observations. Clinical validity has not been established.

## 15. References and attribution

1. Z. Liu et al., “A ConvNet for the 2020s,” *CVPR*, 2022. <https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html>
2. PyTorch Contributors, “ConvNeXt model and pretrained weights,” torchvision documentation. <https://docs.pytorch.org/vision/main/models/convnext.html>
3. C. Guo et al., “On Calibration of Modern Neural Networks,” *ICML*, 2017. <https://proceedings.mlr.press/v70/guo17a.html>
4. I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” *ICLR*, 2019. <https://openreview.net/forum?id=Bkg6RiCqY7>
5. Alzheimer’s Disease Neuroimaging Initiative, ADNI data resource. <https://adni.loni.usc.edu/>
6. Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” *Psychometrika*, 12(2), 1947.
7. E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” *Journal of the American Statistical Association*, 22(158), 1927.
8. G. B. Frisoni, N. C. Fox, C. R. Jack Jr, P. Scheltens, and P. M. Thompson, “The clinical use of structural MRI in Alzheimer disease,” *Nature Reviews Neurology*, 6(2), 67–77, 2010.
9. L. G. Apostolova et al., “Hippocampal atrophy and ventricular enlargement in normal aging, mild cognitive impairment (MCI), and Alzheimer disease,” *Alzheimer Disease and Associated Disorders*, 26(1), 17–27, 2012. <https://doi.org/10.1097/WAD.0b013e3182163b62>
10. N. J. Tustison et al., “N4ITK: Improved N3 bias correction,” *IEEE Transactions on Medical Imaging*, 29(6), 1310–1320, 2010. <https://doi.org/10.1109/TMI.2010.2046908>

The final model’s starting weights are torchvision’s `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`; they are third-party pretrained material rather than student-created weights. The parameter-name mapping in `convert_torchvision_convnext_state_dict()` was written by consulting the layer layout of torchvision’s ConvNeXt source (`torchvision/models/convnext.py`, reference 2); no torchvision model code is used at inference. The student-authored contribution is the ADNI data audit and patient split, model integration, custom training and evaluation loops, experiments, calibration/reject analysis, failure analysis, and report.

## Artificial Intelligence Usage Disclosure

**Tools used.** OpenAI ChatGPT and Codex were used throughout the project. Anthropic Claude was used during the final revision to review the completed code and report.

**What AI assisted with.**

- *Understanding the task and planning (ChatGPT/Codex):* translating and interpreting the assignment specification, and discussing how to structure the experiments.
- *Code (ChatGPT/Codex):* debugging errors in the Slurm scripts, data loading and training loop; explaining concepts such as temperature scaling, expected calibration error and reject options; and suggesting ideas for improving accuracy. I wrote the implementation myself.
- *Code (Claude, final revision):* after reviewing the finished project, Claude drafted the following changes. I applied, tested and committed each one myself:
  - the torchvision-to-`ConvNeXtTiny` weight-name mapping and checkpoint conversion in `modules.py`, with its two equivalence tests;
  - moving `stratified_subject_split()` into `dataset.py`, with synthetic split tests;
  - training-curve plotting in `train.py` and the `--plots` option in `predict.py`;
  - `scripts/compare_predictions.py` and `scripts/checkpoint_provenance.py`;
  - `uncertainty_analysis.py` (cross-model calibration, reject-option, significance and error-overlap analysis) and `scripts/make_pipeline_figure.py`;
  - the redesigned figures in `report_figures.py`.
- *Report:* I wrote the original report and used ChatGPT to check grammar and wording. During the final revision, Claude reviewed it against the assignment specification and drafted the revised text, based on its review of my code and result files, for:
  - the executive summary and research questions;
  - the working-principles subsection (5.2) and Section 5.3;
  - the test-set-use, significance and generalisation-gap paragraphs in Section 7;
  - Section 8, Sections 9–11, the environment set-up in Section 12, and the additions to Sections 2, 3, 4, 6 and 14 (the next planned experiment in Section 2 restates the plan in my check-off document).

  I checked every number in these sections against my result files, and the visual observations in Section 10 against the montages.

**How AI output was verified.**

- Every change was run on Rangpur before it was committed, and the full unit-test suite (14 tests) passes.
- Data leakage: the patient split is checked by unit tests and by `validate_patient_level_split()` on every manifest load. After the split function was moved, I regenerated the manifests and confirmed they were byte-identical to the frozen ones.
- Model correctness: unit tests check output shapes, parameter counts, and that converted torchvision weights produce the same logits in `ConvNeXtTiny` within floating-point tolerance. Re-evaluating the frozen final checkpoint with `ConvNeXtTiny` gave stored patient logits exactly equal to the original run for all 102 validation and 102 test patients, and all 129 numeric entries of `evaluation.json` were exactly equal. The comparison summaries are committed in `figures/report/checkpoint_equivalence_*.json`.
- Analysis: `uncertainty_analysis.py` recomputes accuracy, AUROC and AD recall from the patient CSVs; these match `evaluation.json` exactly, and its outputs were regenerated on Rangpur from the final result folders.
- Results: I checked the confusion-matrix counts, accuracies and per-class metrics against the patient-level prediction CSVs.
- I read every module and can explain what it does and why.

I take full responsibility for all code, results and claims in this submission.
