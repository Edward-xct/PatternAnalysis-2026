# ConvNeXt for Patient-Level AD vs CN Classification on ADNI MRI

**Student:** s4985908  
**Course:** COMP3710 Pattern Recognition  
**Difficulty:** Hard

## Project overview

This project develops a ConvNeXt-based classifier for distinguishing
Alzheimer's Disease (AD) from Cognitively Normal (CN) cases using
preprocessed ADNI brain MRI slices.

A small convolutional neural network will be implemented as the baseline.
Both models will use identical leakage-free patient-level data splits.

## Preliminary data audit

The course dataset contains:

- 30,520 JPEG slices
- 1,526 MRI scan IDs
- approximately 680 unique subjects
- 20 slices per scan
- JSON label 2 for AD and label 0 for CN

The supplied directory split has no repeated scan IDs, but 216 subjects
appear in both the original training and test directories. Therefore, the
supplied split will not be used for final evaluation. All scans belonging
to one subject will be assigned to exactly one new split.

## Planned investigation

The project will compare the baseline CNN and ConvNeXt using patient-level
accuracy, per-class precision and recall, macro-F1, and AUROC. It will also
investigate confidence calibration, overconfident errors, and a reject-option
rule for routing ambiguous predictions to human reviewers.


