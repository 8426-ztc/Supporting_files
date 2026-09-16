# Interpretation of the public aggregate results

The external cohort contains 57 patients. The 122-row LLM table contains accepted runs, not patient records. Twenty runs were evaluated at each intermediate budget (4, 8, 10, 20, 40 and 80); 0 and 109 demonstrations were each evaluated once. Their exported SD=0 is a single-run placeholder, not an estimate of repeatability.

The primary LLM estimate averages run-specific AUROCs: 0.65178125 at 20 demonstrations. Averaging probabilities before computing AUROC gives a different, descriptive estimate (0.66625). These estimands must not be interchanged. The same distinction applies to Brier scores.

The prespecified LLM logistic reference has AUROC 0.655. The RidgeLogistic candidate in conventional development is a different fitted pipeline (AUROC 0.6525). LLM columns labelled baseline refer to the former. The ExtraTrees integration is a secondary comparison; historical primary paired contrast labels identify its internal main contrast.

The headline ExtraTrees AUROC is 0.6675, with its standalone interval in PRIMARY_REPORT_OOF_AND_EXTERNAL_AUC.csv. Marginal intervals in the paired summary use a different resampling workflow and do not replace headline intervals. SVM has a higher descriptive external AUROC (0.67625); ExtraTrees was selected using development performance, not the external maximum. All nine algorithm results are retained.

The frequency export covers 28 candidate variables; the final selection manifest contains 20 eligible candidates. Four variables met the stability threshold. These counts describe different stages. Detailed clinical distributions, exclusion categories and figure files are supplied with the journal materials rather than duplicated here.

The recorded clinical exclusions total 27 among 193 reviewed records, leaving 166 analysed patients. The historical processing label missing pathological necrosis reflects the author-confirmed exclusion coding; it does not identify a second excluded group. The retrospective eligibility decisions remain a design limitation.

This release preserves numerical outputs, historical source code and protocols. It adds documentation and selects a smaller public subset; no inference, fitting, patient-level bootstrap or clinical source correction was performed.
