# MPR90 benchmark protocol

## Claim and boundary

This benchmark tests whether a language model supplied with four pre-treatment
features can discriminate MPR90 in an external-center cohort, compared with a
fixed four-feature logistic-regression baseline. It does not establish clinical
utility, causality, calibration in a deployment population, or safety for
patient care.

The independent evaluation unit is one validation patient. The development
center supplies training data and in-context demonstrations; the validation
center supplies features only until all predictions are frozen. Each validation
patient is queried in a separate prompt so other validation patients cannot
change its prediction through within-prompt comparison or ordering effects.

## Data-leakage threat model

The preparation step blocks duplicate global subject identifiers and exact
four-feature records spanning development and validation centers. It fits
imputation medians on the development center only. Operator files contain only
pseudonymous IDs, the four declared features, and—only for development
demonstrations—the binary target. Raw IDs, continuous `mpr`, validation labels,
the pseudonym key, integrity records and the source-data audit stay in the
private directory.

This protects the benchmark pipeline from accidental local label and identifier
leakage. It cannot prove that an externally hosted model has never seen the
cohort during pretraining and cannot control a provider's retention policy.
Before using any model endpoint, the investigator must document all of the
following:

- ethics/IRB and data-use authorization for sending the four clinical features;
- an institution-approved deployment or contract that disables provider
  training and satisfies the required retention period (preferably no
  retention), or a locally hosted model with network isolation;
- disabled conversational memory and tools/connectors that could retrieve
  external patient information;
- model name, immutable version/snapshot, access date, endpoint, temperature,
  seed if supported, and every retry;
- evidence that the validation cohort was unavailable before the model's
  training cutoff, or an explicit statement that pretraining contamination
  cannot be excluded.

Pseudonymization is not anonymization. Only the `operator` directory may leave
the controlled data environment, and only after institutional review.

## Frozen workflow

1. Choose the primary shot condition before inspecting validation outcomes. If
   it is not declared with `--primary-shot`, every shot comparison is labelled
   exploratory.
2. Prepare a fresh run:

   `python llm_icl_benchmark.py prepare --data zh.csv --run-dir RUN_001 --subject-id-column id --primary-shot 20`

3. Keep `RUN_001/private` with the data custodian. The prediction operator may
   use `RUN_001/operator`, but an external model endpoint should receive only
   each `system_prompt` and `user_prompt` string—not the directory or manifest.
4. Generate the blinded traditional baseline:

   `python llm_icl_benchmark.py baseline --run-dir RUN_001`

5. Run each record in `operator/prompts.jsonl` with the frozen model settings.
   Convert the returned JSON arrays to one CSV with columns
   `condition_id,public_id,pred_label,prob_mpr90`. Record failed calls and retry
   them under the identical frozen settings; never select among multiple model
   answers based on apparent quality.
6. Validate coverage and freeze the LLM output without opening ground truth:

   `python llm_icl_benchmark.py validate-predictions --run-dir RUN_001 --input llm_predictions.csv`

7. After both prediction files are complete and signed off, the data custodian
   performs the only unblinding step:

   `python llm_icl_benchmark.py evaluate --run-dir RUN_001`

Do not rerun or tune prompts after seeing validation metrics. Any later change
creates a new, explicitly exploratory run with a new directory.

## Analysis plan

The primary metric is AUROC. Secondary metrics are AUPRC, Brier score, log
loss, accuracy, sensitivity and specificity at the predeclared 0.5 threshold.
Patient-level stratified percentile bootstrap intervals quantify validation-set
sampling uncertainty. Intermediate shot conditions use 20 independently drawn,
balanced demonstration sets; results are reported per replicate and summarized
across prompt draws. The full development-pool condition is a sensitivity
analysis because its class prevalence differs from balanced demonstrations.

Report effect estimates and confidence intervals, not significance stars alone.
The repeated prompt draws are not independent patients and must not be used to
inflate `n`. Comparisons across every shot size are multiple exploratory
analyses unless one condition was preregistered as primary.

## Required robustness and negative controls

For a publication-grade study, retain the zero-shot condition, the fixed ML
baseline, all planned demonstration replicates, and the full-pool sensitivity
analysis. Additionally run, as separately labelled exploratory controls:

- a temporally held-out or prospectively collected validation cohort if
  pretraining contamination is plausible;
- a label-permuted demonstration control to test whether gains depend on the
  demonstration mapping (never mix this control into the main shot curve);
- subgroup and missingness-pattern analyses only when sample sizes and the
  analysis plan support them;
- calibration plots and calibration slope/intercept; do not describe raw LLM
  scores as calibrated clinical probabilities without external evidence;
- a complete failure-case review conducted after the primary analysis, with no
  post-hoc deletion of difficult cases.

## Reporting checklist

Archive the source-file hash, operator/private integrity manifest, source code,
software environment, exact prompts, all raw model responses, prediction freeze
hashes, validated
prediction CSV, exclusions (normally none), retry log and final evaluation
files. State center-specific sample sizes, outcome prevalence after unblinding,
the independent unit, the imputation rule, demonstration sampling, number of
prompt replicates, model settings, metric definitions and uncertainty method.
Any deviation from this protocol must be timestamped and labelled as occurring
before or after unblinding.
