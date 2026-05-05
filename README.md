# walkability_predictive_spatial_workflow_Kigali
Python code for a spatially validated walkability prediction workflow using Kigali participatory micro-audit data. It supports reproducible modelling of perceived walkability and walking discretion with spatial validation, calibration, conformal uncertainty, feature analysis, and decision-unit targeting.

This script implements the modelling workflow used in the associated manuscript:
"Spatially validated prediction of perceived walkability and perceived walking
discretion in Kigali: A participatory micro-audit workflow for urban planning."

Purpose
-------
The code evaluates whether participatory micro-audit observations contain
transferable predictive information for two walking-related outcomes:

1. Perceived walkability:
   an ordered three-class outcome coded as Good = 0, Concern = 1, Problem = 2.

2. Perceived walking discretion:
   a binary outcome coded as 1 for walking by choice and 0 for walking by necessity.

The workflow is designed for planning-oriented predictive assessment, not causal
effect estimation. Its main objective is to evaluate spatial transferability,
calibration quality, uncertainty, and decision-unit prioritisation value under
leakage-aware validation.

Main workflow
-------------
The script performs the following operations:

1. Loads the cleaned wide-format walkability dataset.
2. Constructs outcome variables when needed from the original survey fields.
3. Infers predictor blocks:
   - micro-scale environmental issue indicators,
   - respondent and trip-context variables,
   - location-related variables.
4. Projects WGS84 latitude/longitude coordinates to a local metric coordinate
   system and constructs metre-grid spatial groupings.
5. Optionally assigns observations to external polygon groups when a polygon
   file is provided.
6. Builds spatial and grouped validation protocols, including:
   - leave-one-area-out,
   - polygon/block holdout,
   - metre-grid blocking,
   - respondent-group validation,
   - random validation as a diagnostic baseline.
7. Fits five candidate model families:
   - elastic-net logistic regression,
   - histogram gradient boosting,
   - random forest,
   - extra-trees,
   - gradient boosting.
8. Performs nested hyperparameter tuning inside the outer-training folds.
9. Splits each outer-training fold into disjoint subsets for:
   - model fitting,
   - probability calibration,
   - conformal uncertainty estimation.
10. Produces out-of-fold calibrated probabilities and conformal prediction sets.
11. Evaluates discrimination, error, calibration, ordinal performance, and
    conformal coverage.
12. Computes original-variable permutation importance under primary spatial
    validation protocols.
13. Aggregates predicted risk to decision units and evaluates targeting
    performance against issue-count and random baselines.
14. Exports paper-facing summary tables, figures, maps, and reproducibility
    manifests.

Important interpretation
------------------------
The outputs are intended to support cautious planning diagnostics in sampled or
analytically similar walking environments. They should not be interpreted as
city-wide prevalence estimates, causal intervention effects, or deterministic
classifications of pedestrian conditions.

Author
------
[Insert author names]

License
-------
[Insert license]

Citation
--------
If using this code, please cite the associated paper and the archived software
release DOI.
"""
