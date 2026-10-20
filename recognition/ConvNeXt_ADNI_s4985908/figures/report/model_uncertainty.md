## Calibration and confidence

| Model | T | ECE raw → cal | NLL raw → cal | Raw conf ≥ 0.9: patients (accuracy) | Confidence AUROC for correctness | Errors with conf ≥ 0.9, raw → cal |
|---|---:|---:|---:|---:|---:|---:|
| SmallCNN | 0.87 | 0.098 → 0.123 | 0.566 → 0.594 | 36 (0.861) | 0.690 | 5 → 7 of 25 |
| ConvNeXt (scratch) | 1.67 | 0.128 → 0.108 | 0.619 → 0.508 | 65 (0.877) | 0.673 | 8 → 6 of 22 |
| ConvNeXt (ImageNet) | 3.23 | 0.144 → 0.063 | 0.997 → 0.441 | 89 (0.865) | 0.701 | 12 → 7 of 17 |

## Frozen reject option

| Model | NC threshold | AD threshold | Coverage | Selective accuracy | AD auto-cleared | NC auto-flagged | Errors referred |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmallCNN | 0.141 | 0.500 | 0.725 | 0.811 | 7/33 (21.2%) | 7/69 (10.1%) | 11/25 |
| ConvNeXt (scratch) | 0.138 | 0.674 | 0.667 | 0.824 | 6/33 (18.2%) | 6/69 (8.7%) | 10/22 |
| ConvNeXt (ImageNet) | 0.137 | 0.500 | 0.833 | 0.859 | 8/33 (24.2%) | 4/69 (5.8%) | 5/17 |

## Uncertainty of the test metrics

| Model | Accuracy (95% Wilson CI) | AUROC (95% bootstrap CI) | AD recall (95% Wilson CI) |
|---|---:|---:|---:|
| SmallCNN | 0.755 (0.663–0.828) | 0.791 (0.694–0.880) | 0.455 (0.298–0.620) |
| ConvNeXt (scratch) | 0.784 (0.695–0.853) | 0.808 (0.711–0.894) | 0.667 (0.496–0.802) |
| ConvNeXt (ImageNet) | 0.833 (0.749–0.893) | 0.855 (0.769–0.929) | 0.606 (0.437–0.753) |

| ImageNet ConvNeXt minus | Accuracy difference (95% CI) | McNemar p | AUROC difference (95% CI) |
|---|---:|---:|---:|
| SmallCNN | +0.078 (+0.010 to +0.147) | 0.057 (11 vs 3 discordant) | +0.064 (+0.009 to +0.119) |
| ConvNeXt (scratch) | +0.049 (-0.020 to +0.118) | 0.302 (10 vs 5 discordant) | +0.047 (+0.002 to +0.097) |

## Error overlap

Errors: SmallCNN 25, scratch 22, ImageNet 17. ImageNet errors also made by both baselines: 10; ImageNet false negatives missed by all three models: 9/13; errors unique to ImageNet: 1.

## Failure cases across models (calibrated P(AD), triage)

| Case | Error | SmallCNN | ConvNeXt scratch | ConvNeXt ImageNet |
|---|---|---:|---:|---:|
| A | AD -> NC | 0.042 (NC) | 0.044 (NC) | 0.029 (NC) |
| B | NC -> AD | 0.803 (AD) | 0.888 (AD) | 0.778 (AD) |
| C | AD -> NC | 0.230 (REFER) | 0.082 (NC) | 0.034 (NC) |
| D | AD -> NC | 0.080 (NC) | 0.038 (NC) | 0.035 (NC) |
| E | AD -> NC | 0.068 (NC) | 0.082 (NC) | 0.049 (NC) |

Referred ImageNet false negatives, calibrated P(AD): 0.150, 0.164, 0.329, 0.363, 0.496
