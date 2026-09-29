# TRIPOD+AI reporting cross-reference

This evidence map corresponds to Online Resource 2 of manuscript v25. Concise topic labels identify the relevant manuscript locations and unresolved reporting gaps; the map is not a certification of complete adherence. Source: https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf

| Item | Topic | Location | Evidence or remaining gap |
|---|---|---|---|
| 1 | Title | Title | Target, prediction task and evaluation identified. |
| 2 | Abstract | Abstract | Design, sample sizes, prespecified LLM condition, secondary ExtraTrees comparison and uncertainty reported. |
| 3a | Clinical rationale | Introduction | Clinical context, existing prediction approaches and rationale for external evaluation are described. |
| 3b | Intended population and users | Introduction; Online Resource 1, Appendix A | Preoperative response assessment for multidisciplinary teams is the intended context. Clinical deployment was not evaluated. |
| 3c | Health inequalities | Limitations; Online Resource 1, Appendix A | Health inequalities were not specifically investigated; cohort representativeness and untested subgroups are identified as limitations. |
| 4 | Objectives | Introduction, final paragraph | Conventional development and external LLM evaluation. |
| 5a | Data source | Study design and participants; Limitations | Two retrospective institutional cohorts; selection of surgical patients limits representativeness. |
| 5b | Dates | Study design and participants | Surgery dates were 26 December 2017 to 27 December 2024; exact predictor measurement windows were not established. |
| 6a | Setting | Study design and participants | Two named centres and countries/cities. |
| 6b | Eligibility | Study design and participants; Online Resource 1, Table S1 | 27 recorded retrospective exclusions reconciled in Online Resource 1, Table S1; these are clinical screening decisions |
| 6c | Treatment | Outcome and candidate predictors; Study population and between-centre heterogeneity; Table 1 | Centre-specific exposure; inputs not uniformly treatment-naïve. |
| 7 | Preparation | Missing data handling and cohort comparisons; ML Model Development, Locking, and Performance Evaluation; Online Resource 1, Supplementary Methods 1–2 | Training-only preprocessing; no group-specific preprocessing audit. |
| 8a | Outcome | Outcome and candidate predictors; Online Resource 1, Table S1 | Necrosis ≥90% from surgical pathology; patient-level label. |
| 8b | Outcome assessors | Outcome and candidate predictors | Pathologists assessed resection specimens. Assessor numbers and demographic characteristics were not reported. |
| 8c | Outcome blinding | Outcome and candidate predictors; Limitations | Blinding of the original pathology readers to predictor information was not established. Outcome blinding during model inference is a separate procedure. |
| 9a | Predictor candidates | Outcome and candidate predictors; Online Resource 1, Supplementary Methods 1; Online Resource 1, Table S2 | Candidate dictionary and fold-local selection supplied. |
| 9b | Predictor definitions | Outcome and candidate predictors; Online Resource 1, Supplementary Methods 2; Online Resource 1, Table S2 | Same earliest-available preoperative record rule at both centres; prior therapy allowed; exact measurement windows not established. |
| 9c | Predictor assessors | Outcome and candidate predictors | Qualifications and demographic characteristics of the original tumour-diameter assessors were not reported. |
| 10 | Study size | Study design and participants; ML Model Development, Locking, and Performance Evaluation; Discussion | Available retained cohort; no formal precision/power calculation. |
| 11 | Missingness | Study design and participants; Missing data handling and cohort comparisons; Online Resource 1, Table S1; Online Resource 1, Table S2 | Exclusion flag distinguished from clinical decisions; training-derived imputation. |
| 12a | Data partitioning | Study design and participants; ML Model Development, Locking, and Performance Evaluation | One centre supplied development data and the other external validation data; repeated nested cross-validation was restricted to development data. |
| 12b | Predictor handling | Outcome and candidate predictors; Missing data handling and cohort comparisons; Online Resource 1, Supplementary Methods 1–2; Online Resource 1, Table S2 | Clinical units, transformations, training-derived imputation and LLM input construction are described. |
| 12c | Model development | ML Model Development, Locking, and Performance Evaluation; Online Resource 1, Supplementary Methods 1 | Nine algorithms, fold-local feature selection, hyperparameter tuning, internal evaluation and the development-only selection rule are described. |
| 12d | Clustering | Study design and participants; Study population and between-centre heterogeneity | Site-separated evaluation; no multi-cluster random-effects model. |
| 12e | Evaluation measures | ML Model Development, Locking, and Performance Evaluation; LLM experiment; Paired comparisons with ML models; Statistical analysis and reproducibility; Figures 2–4 | Discrimination, probability error and calibration; paired performance differences with confidence intervals. |
| 12f | Updating | ML Model Development, Locking, and Performance Evaluation; Online Resource 1, Supplementary Methods 4 | No external recalibration or model refitting was performed. |
| 12g | Prediction procedure | ML Model Development, Locking, and Performance Evaluation; LLM experiment; Paired comparisons with ML models; Statistical analysis and reproducibility; Online Resource 1, Supplementary Methods 4; Online Resource 1, Supplementary Methods 1–3 | Locked conventional preprocessing and hosted cohort-batched inference. |
| 13 | Class balance | Online Resource 1, Supplementary Methods 2; archived source | Label-balanced intermediate demonstrations; natural full-pool composition. Conventional settings preserved in source. |
| 14 | Fairness | Discussion; Limitations | Not evaluated; no fairness claim. |
| 15 | Outputs | ML Model Development, Locking, and Performance Evaluation; LLM experiment; Online Resource 1, Supplementary Methods 3 | Probabilities; thresholds 0.49 and 0.500. |
| 16 | Between-centre differences | Outcome and candidate predictors; Study population and between-centre heterogeneity; Table 1 | Case-mix and treatment differences acknowledged; a common baseline definition does not imply identical measurement windows. |
| 17 | Ethics | Ethics approval; Data protection | Committee, approval 25-005/0005, dated 23 January 2025; the approving committee belongs to the development institution. |
| 18a | Funding | Funding | Funding sources and grant numbers are reported; funder roles were not reported. |
| 18b | Competing interests | Competing interests | The manuscript reports no competing interests. |
| 18c | Protocol | Online Resource 1, Supplementary Methods 3–4; supporting repository, 03_quality_control/archived_protocols | The dated local protocol and its amendments are publicly available; the executed design is distinguished from the earlier pilot. |
| 18d | Registration | Online Resource 1, Supplementary Methods 4 | The local analysis plan documents specification before external-outcome access; no public registration identifier is reported. |
| 18e | Data access | Data availability; Online Resource 1, Table S2; supporting repository | Selected aggregate results are public. Individual-level data are restricted and may be requested from the corresponding author subject to institutional and ethics approval and an appropriate agreement. |
| 18f | Code access | Code and prompt availability; Statistical analysis and reproducibility; supporting repository | Analysis code, prompt templates and software versions are public. Patient-level inputs are required for reanalysis. The repository does not specify a reuse licence. |
| 19 | Patient/public involvement | Not reported | Patient and public involvement was not reported. |
| 20a | Flow | Study design and participants; Study population and between-centre heterogeneity; Figure 1; Online Resource 1, Table S1 | 2395 screened, 193 reviewed, 27 excluded, 166 retained. |
| 20b | Cohort characteristics | Study population and between-centre heterogeneity; Table 1; Online Resource 1, Table S2 | Sample sizes, outcome counts, treatment exposure and predictor distributions are reported by centre; variable-level missingness is provided in Table S2. |
| 20c | Development–validation comparison | Study population and between-centre heterogeneity; Table 1; Online Resource 1, Table S2 | Development and external validation distributions are presented side by side; between-centre differences are described. |
| 21 | Analysis sample sizes | ML Model Development, Locking, and Performance Evaluation; LLM experiment; Study population and between-centre heterogeneity; source tables | Development 109/55 positive; external 57/25; round-specific evaluable counts in source. |
| 22 | Model specification | ML Model Development, Locking, and Performance Evaluation; LLM experiment; Online Resource 1, Supplementary Methods 1–4; Code and prompt availability | Algorithms, tuned settings, prompts and source code are supplied. Refitting requires restricted inputs, and an immutable hosted-model snapshot is unavailable. |
| 23a | Performance | ML model selection and external performance; LLM performance and repeatability across demonstration budgets; Paired external comparisons with ML models; Figures 2–4 | Primary estimates and intervals reported; subgroup metrics not evaluated. |
| 23b | Performance heterogeneity | Study population and between-centre heterogeneity; Discussion | No additional multicentre heterogeneity analysis. |
| 24 | Model updating results | ML Model Development, Locking, and Performance Evaluation; Online Resource 1, Supplementary Methods 4 | Not applicable. Models were not updated using external validation data. |
| 25 | Interpretation | Discussion | Paired differences do not establish advantage or equivalence. |
| 26 | Limitations | Limitations; Online Resource 1, Table S1 | Selection, timing, sample size, calibration and cohort-batched limitations visible. |
| 27a | Input quality in clinical use | Discussion; Online Resource 1, Appendix A | Input checks and handling of unavailable measurements are identified as requirements for future implementation; prospective performance of these safeguards was not evaluated. |
| 27b | User interaction and expertise | Discussion; Online Resource 1, Appendix A | Intended use by multidisciplinary clinical teams is described. User training requirements and interaction with a clinical interface were not evaluated. |
| 27c | Further evaluation | Discussion; Limitations | Larger external cohorts, patient-wise evaluation and workflow monitoring are proposed before clinical use. |
