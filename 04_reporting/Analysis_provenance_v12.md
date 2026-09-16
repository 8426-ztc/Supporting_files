# Analysis provenance and interpretation

The original LLM protocol and the subsequent integration with ExtraTrees answer related but distinct questions. This note maps their analysis status and interval sources without changing historical outputs.

## Analysis sequence

The locally archived plan dated 29 August 2026 specified a primary 20-shot LLM condition, 20 prompt replicates and a frozen four-feature L2 logistic-regression reference. Addendum 06 retained that comparison when execution changed to cohort-batched inference. The plan is evidence of local specification before outcome access, not a public registry record.

ExtraTrees was selected within the conventional-model development workflow and locked before external evaluation. Its subsequent pairing with the LLM was an integrated secondary comparison. The historical label “primary paired contrast” identifies the main contrast within that integration; it does not establish prespecification in the original LLM protocol. Section S7 retains the original logistic-regression comparison.

The LLM-protocol logistic reference is a separate fitted model from the ridge-logistic candidate in the nine-algorithm conventional development comparison. Their external AUROCs should not be interchanged.

## Estimands and interval sources

The primary LLM quantity is the mean of 20 run-specific AUROCs. The AUROC of probabilities averaged for each patient is a different quantity used for descriptive displays. The same distinction applies to mean run-specific Brier score and the Brier score of averaged probabilities. No ensemble was fitted or independently evaluated.

The standalone ExtraTrees AUROC interval uses 1,000 patient-level bootstrap resamples. Original LLM and LLM-versus-logistic intervals use 1,000 hierarchical outcome-stratified resamples. The integrated LLM-versus-ExtraTrees intervals use 10,000 paired hierarchical outcome-stratified resamples. Marginal intervals recalculated during the integration need not equal the standalone intervals and are not substituted for the designated headline intervals.

The headline ExtraTrees estimate and interval are in `01_tables_and_source_data/PRIMARY_REPORT_OOF_AND_EXTERNAL_AUC.csv`; the paired comparison is in `01_tables_and_source_data/ML_LLM_PAIRED_METRICS_SUMMARY_20260831_v1.csv`. The v18 manuscript and Figure 2 use the source AUROC of 0.6675. No model or statistical analysis was rerun for this documentation correction.

## Meaning of audit

Audit denotes checks of recorded execution, expected case coverage, output schema, case ordering, value ranges, threshold consistency, artifact identity and recorded outcome access. These checks do not independently establish all historical governance actions, the provider's internal state or performance in untested populations. The current manuscript provides the interpretation; historical code and generated prose remain unchanged for provenance.
