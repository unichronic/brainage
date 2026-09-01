# Exp2 cortical-thickness experiment

## Status

Cortical thickness is an exploratory branch of the Exp2 workflow. It is not
part of the current production GM feature vector and no cortical-thickness
prediction head was promoted.

The pipeline keeps this branch label-blind during feature extraction. It uses
the same downloaded raw T1 files as the GM and CSF branches, while ages and
base-model predictions are kept out of the cortical pilot inputs.

## Methods

- CortexODE: global and distributional cortical-thickness summaries;
- DL+DiReCT: segmentation/probability-map front end; and
- CortexMorph: regional thickness summaries from the DL+DiReCT outputs.

DL+DiReCT and CortexMorph require their own external repositories and
checkpoints. They are optional and are not bundled in this compact checkout.

## Exp2 result

On the 554-row calibration split, repeated group-safe checks gave:

| Representation | MAE (years) |
|---|---:|
| Raw Exp2 prediction | 4.812 |
| Raw Exp2 + whole-GM summary | 4.788 |
| Raw Exp2 + whole-GM + CortexODE global thickness | 4.799 |

The thickness fusion was therefore slightly worse than the comparison
baseline. The locked-test extraction was completed label-blind, and a
calibration-selected residual thickness head improved some age-band bias
measures but did not improve the overall downstream objective consistently.
It remains a documented sensitivity result, not a production model.

The thickness representation itself does contain age-related structure: on
calibration and locked-test data, bilateral p95 thickness had Pearson
correlations of -0.453 and -0.527 with chronological age. The global mean is
nonlinear across the lifespan and varies by cohort, so raw thickness values
should not be pooled without a predeclared harmonization plan.

## Reproduce

After the main pipeline has run `prepare` and `download`, start the default
calibration-only pilot with:

```bash
python pipeline/run_exp2_pipeline.py cortical --methods cortexode
```

Use `--include-locked-test` only for a predeclared label-blind extraction, and
provide `--dl-direct-dir` when running the optional DL+DiReCT/CortexMorph
methods.
