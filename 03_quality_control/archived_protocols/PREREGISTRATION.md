# Locked analysis plan before outcome unblinding

Date locked: 2026-08-29 (Asia/Shanghai)

Source dataset SHA-256:
`efac55a276a3c11c47b15aaab06f945b955d8c3c43abe98b21d8f75f21828c64`

## Information inspected before locking

Only the column names, data types, total sample size, center-specific sample
sizes, global uniqueness of `id`, and feature-missingness counts were inspected.
No MPR90 class count, validation outcome, model prediction, performance metric,
or outcome-stratified characteristic was calculated or displayed before this
plan was locked.

## Cohorts and endpoint

- Development cohort: center `yz`, 128 patients.
- External validation cohort: center `zy`, 65 patients.
- Independent unit: one patient identified by globally unique `id`.
- Endpoint: MPR90, defined as `mpr >= 0.90`.
- Predictors: BMI, baseline maximum tumour diameter, AST and TBIL.
- Missing predictors: development-cohort medians are fitted once and then
  applied unchanged to both cohorts. No outcome-informed imputation is allowed.

## Primary analysis

- Primary LLM condition: 20-shot in-context learning.
- Demonstrations: 10 MPR90-positive and 10 MPR90-negative development patients,
  sampled without replacement and randomly ordered.
- Prompt replicates: 20 independently sampled demonstration sets.
- Validation query: one patient per prompt.
- Primary metric: AUROC.
- Primary comparison: mean AUROC across the 20 demonstration replicates minus
  the AUROC of the frozen four-feature L2-penalized logistic-regression model.
- Uncertainty: stratified patient-level bootstrap combined with bootstrap
  resampling of demonstration replicates. The interval is conditional on the
  observed development and validation cohorts and the frozen model versions.

The primary comparison will be interpreted from its effect estimate and 95%
bootstrap confidence interval. No significance-star claim is planned.

## Secondary and exploratory analyses

- Secondary metrics: AUPRC, Brier score, log loss, accuracy, sensitivity and
  specificity at a threshold of 0.5.
- Secondary shot conditions: 0, 4, 8, 10, 40 and 80 shots when class counts make
  balanced sampling feasible.
- Full development-pool prompting is a sensitivity analysis because it uses
  the natural development prevalence rather than balanced demonstrations.
- Shot-curve comparisons other than the 20-shot primary comparison are
  exploratory. No family-wise confirmatory claim will be made from them.
- Subgroup, missingness-pattern, failure-case and label-permutation analyses are
  exploratory and must be labelled as post-primary unless separately locked
  before outcome access.

## Blinding and leakage controls

Raw patient identifiers, continuous `mpr`, validation labels, `status`, `RFS`,
the pseudonym key and the identifier map remain in the private custodian
directory. The prediction operator receives pseudonymous IDs and the four
allowed predictors only. Every prediction file must have complete prespecified
coverage and a matching freeze hash before unblinding. Code, inputs, prompts,
model settings and outputs are hashed and retained.

This local workflow cannot establish that a third-party model never encountered
the cohort during pretraining. A publication must either document a temporally
post-cutoff/prospective validation cohort or state pretraining contamination as
not excludable. Sending patient features to an external endpoint additionally
requires documented ethics/data-use authorization and an approved no-training
and retention arrangement.
