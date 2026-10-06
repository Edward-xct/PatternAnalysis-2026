# ConvNeXt for Patient-Level AD vs CN Classification on ADNI MRI

**Student:** s4985908  
**Course:** COMP3710 Pattern Recognition  
**Difficulty:** Hard

## 1. Project overview

This project classifies Alzheimer's Disease (AD) and cognitively normal
controls (CN) from preprocessed ADNI T1-weighted MRI slices. The main model is
a from-scratch ConvNeXt-Tiny implementation. A compact convolutional neural
network (SmallCNN) is used as the baseline.

The optimisation unit is a 2D slice, but the decision unit is a patient. Slice
logits from all scans belonging to one subject are averaged before softmax, and
all primary metrics are calculated at patient level. In the code, the normal
class is named `NC`; it has the same meaning as CN in this report.

The investigation asks:

1. Does ConvNeXt-Tiny improve patient-level classification over SmallCNN?
2. Are the predicted probabilities calibrated, especially for errors?
3. Can a validation-selected reject option refer uncertain patients while
   retaining useful coverage?
4. Is the additional ConvNeXt computation justified by its performance?

## 2. Data audit and leakage prevention

The course dataset is read from:

```text
/home/groups/comp3710/ADNI
```

It contains JPEG slices under `AD_NC/train/{AD,NC}` and
`AD_NC/test/{AD,NC}`, together with `meta_data_with_label.json`. JPEG names
have the form `<scan_id>_<slice_index>.jpeg`. The metadata maps each scan ID to
an ADNI subject ID and diagnosis. Metadata label `2` is AD and label `0` is NC;
these are remapped to model targets `1` and `0`, respectively.

The audit found:

- 30,520 JPEG slices;
- 1,526 MRI scan IDs;
- 680 unique subjects;
- exactly 20 slices per scan;
- no inconsistent subject labels;
- no scan overlap between the supplied train and test directories; and
- **216 subjects shared between the supplied train and test directories**.

The supplied directory split therefore leaks patients and is not used for
model selection or final evaluation. `scripts/audit_data.py` combines all
available scans and creates a deterministic, label-stratified 70/15/15 split
using seed `3710`. Every scan and every visit from one subject remains in one
split.

| Split | Subjects | AD | NC | Scans | Slices |
|---|---:|---:|---:|---:|---:|
| Train | 476 | 155 | 321 | 1,081 | 21,620 |
| Validation | 102 | 33 | 69 | 228 | 4,560 |
| Test | 102 | 33 | 69 | 217 | 4,340 |
| **Total** | **680** | **221** | **459** | **1,526** | **30,520** |

The resulting subject overlap is zero. The script writes reproducible local
artifacts to `artifacts/subject_split.csv`, `artifacts/slice_manifest.csv`, and
`artifacts/split_summary.json`. These generated files contain machine-specific
paths and are intentionally ignored by Git.

Generate the manifests from the repository root with:

```bash
python recognition/ConvNeXt_ADNI_s4985908/scripts/audit_data.py
```

## 3. Input pipeline

`dataset.py` loads the manifest and validates that no subject occurs in more
than one split. Each grayscale slice is resized to 224 x 224, copied to three
channels, converted to a tensor, and normalised with ImageNet channel
statistics. Training uses conservative affine augmentation; validation and
test transformations are deterministic.

The training sampler balances both diagnosis and subject identity so that the
larger NC class and subjects with more scans do not dominate optimisation.
Evaluation always uses every available slice and averages a subject's slice
logits before applying softmax.

## 4. Models

### 4.1 SmallCNN baseline

SmallCNN uses repeated convolution, batch normalisation, ReLU, and pooling,
followed by global average pooling, dropout, and a two-class linear head. It
has **389,410 trainable parameters**.

### 4.2 ConvNeXt-Tiny

The main model is implemented in `modules.py` using core PyTorch operations.
It is randomly initialised rather than loaded from a pretrained model. Its
configuration is:

- patchify stem: 4 x 4 convolution with stride 4;
- stage depths: `[3, 3, 9, 3]`;
- stage channels: `[96, 192, 384, 768]`;
- 7 x 7 depthwise convolution in each block;
- channels-last LayerNorm and a 4x pointwise MLP with GELU;
- layer scale, stochastic depth, and residual connections; and
- global average pooling, LayerNorm, dropout, and a two-class head.

It has **27,821,666 trainable parameters**.

Both models consume tensors of shape `(batch, 3, 224, 224)` and output two
logits ordered as NC and AD.

## 5. Training and model selection

Both models use the same manifest, input pipeline, balanced sampler, seed, and
patient-level validation code. The default configuration is:

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| Epochs | 30 |
| Batch size | 64 |
| Initial learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Dropout | 0.30 |
| Scheduler | cosine decay |
| Random seed | 3710 |
| Model-selection metric | validation macro-F1 |
| Mixed precision | enabled on CUDA |

`train.py` saves the best checkpoint, configuration, epoch history, validation
metrics, runtime, and peak GPU memory. The test split is not evaluated during
ordinary training.

From the repository root, submit the reproducible Rangpur jobs with:

```bash
sbatch recognition/ConvNeXt_ADNI_s4985908/scripts/train_smallcnn.slurm
sbatch recognition/ConvNeXt_ADNI_s4985908/scripts/train_convnext.slurm
```

The scripts record the job ID, host, Git commit, timestamps, GPU information,
and an output directory containing the Slurm job ID.

## 6. Inference, calibration, and reject option

`predict.py` exports both slice-level and patient-level CSV files. Test
inference is locked behind `--allow-test`, and partial test inference is
forbidden. Validation inference should be completed before the final model and
decision rule are frozen.

Example validation inference:

```bash
python recognition/ConvNeXt_ADNI_s4985908/predict.py \
  --checkpoint <CHECKPOINT>/best.pt \
  --split validation \
  --output-dir <OUTPUT>/validation_predictions \
  --device cuda
```

`evaluate.py` fits one temperature parameter using validation logits only. It
also chooses lower and upper probability thresholds for a three-way decision:

```text
P(AD) <= tau_NC                 -> accept as NC
tau_NC < P(AD) < tau_AD         -> refer for human review
P(AD) >= tau_AD                 -> accept as AD
```

The validation analysis reports accuracy, per-class precision and recall,
macro-F1, AUROC, NLL, Brier score, expected calibration error (ECE), coverage,
referral rate, and selective accuracy. It can also produce a reliability
diagram and risk-coverage curve.

```bash
MPLCONFIGDIR=$HOME/tmp/matplotlib \
python recognition/ConvNeXt_ADNI_s4985908/evaluate.py \
  --validation-csv <OUTPUT>/validation_predictions/validation_patient_predictions.csv \
  --output-dir <OUTPUT>/validation_evaluation \
  --expected-validation-patients 102 \
  --plots
```

Only after the model, temperature method, and reject thresholds are frozen is
the held-out test split unlocked:

```bash
python recognition/ConvNeXt_ADNI_s4985908/predict.py \
  --checkpoint <CHECKPOINT>/best.pt \
  --split test \
  --output-dir <OUTPUT>/test_predictions \
  --device cuda \
  --allow-test

MPLCONFIGDIR=$HOME/tmp/matplotlib \
python recognition/ConvNeXt_ADNI_s4985908/evaluate.py \
  --validation-csv <OUTPUT>/validation_predictions/validation_patient_predictions.csv \
  --test-csv <OUTPUT>/test_predictions/test_patient_predictions.csv \
  --output-dir <OUTPUT>/final_evaluation \
  --expected-validation-patients 102 \
  --expected-test-patients 102 \
  --plots \
  --allow-test
```

## 7. Results

The table below is intentionally left incomplete until both full training jobs
finish. Smoke-test results are not reported as experimental results.

| Patient-level test metric | SmallCNN | ConvNeXt-Tiny |
|---|---:|---:|
| Accuracy | TBD | TBD |
| Macro-F1 | TBD | TBD |
| AUROC | TBD | TBD |
| Precision (AD) | TBD | TBD |
| Recall (AD) | TBD | TBD |
| Precision (NC) | TBD | TBD |
| Recall (NC) | TBD | TBD |
| ECE before calibration | TBD | TBD |
| ECE after calibration | TBD | TBD |
| Brier score after calibration | TBD | TBD |
| Selective accuracy | TBD | TBD |
| Coverage | TBD | TBD |
| Referral rate | TBD | TBD |
| Peak GPU memory | TBD | TBD |
| Inference latency per patient | TBD | TBD |

The course target is patient-level test accuracy of at least 0.80. If the
target is not reached, the measured result will still be reported and analysed
rather than hidden.

## 8. Failure-case analysis

After the one-time final test evaluation, 3-5 representative patient-level
errors will be selected programmatically. The analysis will include both false
negatives and false positives and will record:

- true and predicted diagnosis;
- raw and calibrated confidence;
- whether SmallCNN and ConvNeXt agree;
- whether the reject option refers the case; and
- visible acquisition, preprocessing, contrast, or slice-position issues.

Any proposed medical explanation will be labelled as a hypothesis because the
model and this analysis are not clinical diagnostic tools.

## 9. Verification

The automated test suite covers patient-disjoint splits, label consistency,
20 slices per scan, data shapes and types, both model forward passes, and
rejection of invalid non-RGB input.

Run it on Rangpur with:

```bash
sbatch recognition/ConvNeXt_ADNI_s4985908/scripts/run_tests.slurm
```

The latest verification run completed all eight tests successfully. Separate
CPU smoke tests also verified the data loader, SmallCNN forward/backward pass,
ConvNeXt forward/backward pass, training pipeline, full validation inference,
calibration output, and plot generation. These checks demonstrate software
correctness only; they are not substitutes for full experiments.

## 10. Repository structure

```text
ConvNeXt_ADNI_s4985908/
|-- README.md
|-- dataset.py
|-- modules.py
|-- train.py
|-- predict.py
|-- evaluate.py
|-- requirements.txt
|-- scripts/
|   |-- audit_data.py
|   |-- run_tests.slurm
|   |-- train_smallcnn.slurm
|   `-- train_convnext.slurm
|-- tests/
|   |-- test_patient_split.py
|   |-- test_dataset_shapes.py
|   `-- test_model_forward.py
|-- figures/                 # small report figures only
|-- artifacts/               # generated locally; ignored by Git
|-- logs/                    # generated locally; ignored by Git
`-- outputs/                 # checkpoints/results; ignored by Git
```

## 11. Environment

The tested Rangpur environment is Python 3.11.15 with PyTorch 2.13.0,
torchvision 0.28.0, Pillow 12.3.0, NumPy 2.4.6, and matplotlib 3.11.2. Exact
versions are recorded in `requirements.txt`.

## 12. Reproducibility and data policy

- Seed `3710` is used for splitting and experiments.
- Slurm logs record the exact Git commit and execution host.
- Generated manifests, raw ADNI images, checkpoints, and large outputs are not
  committed.
- The held-out test set remains locked until all modelling and threshold
  choices are frozen.
- Results are reported at patient level, not only at slice level.

## 13. Limitations and ethical considerations

ADNI is a research cohort and may not represent the prevalence, scanner
variation, comorbidities, and demographics of a deployment population. Slices
from repeated scans are correlated, and a 2D model does not use the full 3D
anatomical context. Dataset labels are not equivalent to ground-truth
pathology. High test performance would therefore not establish clinical
validity. The reject option is an experimental decision-support mechanism, not
a substitute for expert review.

## 14. AI assistance disclosure

Generative AI was used to help design the project structure, draft and review
code, explain Git/Rangpur commands, and prepare documentation. The student
executed the code, inspected outputs, verified hashes and tests, and remains
responsible for understanding, validating, and presenting every submitted
component. No ADNI image data or patient-level records were uploaded to an AI
service.
