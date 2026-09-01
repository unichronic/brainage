# Preprocessing compatibility

There are two valid ways to use this distribution, with different guarantees.

## Existing FastSPM maps: exact released-model path

Use `extract-features` on supplied `mwc1` maps and then `predict`. The
packaged Python implementation was checked against the existing 50-map
archive: it produced the same `(50, 29852)` feature matrix, and the Docker
model produced the same 5.183-year confirmation MAE as the research run.

## Raw T1 images: Docker/Octave path

Use `preprocess INPUT OUTPUT` to run SPM12 r7771 Unified Segmentation under
Octave. This requires no MATLAB license and is the portable replacement for
the old Singularity/MATLAB Runtime execution path.

The released Exp2 maps, however, were created by SPM12 r7771's standalone
MATLAB Runtime. Although the MATLAB and Octave drivers use the same
`fastspm_v1.m` batch recipe, the numerical engines can produce small map
differences. In the smoke test, the same subject had a mean absolute map
difference of about 0.0025 in the generated `mwc1` image and a 0.14-year
change when the released model was applied. This is not evidence that one
engine is better; it is evidence that the preprocessing engine is part of the
model input definition.

Therefore:

- use the released model with maps from the released preprocessing run when
  reproducing the reported result;
- if raw images are processed with this Docker image, use it consistently for
  every training, validation, and test subject; and
- retrain a new model if the Docker/Octave preprocessing path becomes the
  production representation.

This keeps the core BrainAgeR principle intact while avoiding an unmeasured
cross-runtime domain shift.
