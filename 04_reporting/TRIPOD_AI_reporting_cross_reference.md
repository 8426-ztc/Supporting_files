# TRIPOD+AI reporting cross-reference

This evidence map uses abbreviated topic labels and section locations. It identifies partial or unavailable reporting explicitly; it is not a certification of complete adherence. The official checklist remains the reference for submission. Source: https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf

| Item | Topic | Location | Evidence or remaining gap |
|---|---|---|---|
| 1 | Title | Title | Target, prediction task and evaluation identified. |
| 2 | Abstract | Abstract | Design, sample sizes, prespecified LLM condition, secondary ExtraTrees comparison and uncertainty reported. |
| 3a–b | Context and users | Introduction; S6 | Offline preoperative research setting; clinical deployment not tested. |
| 3c | Health inequalities | S6 | Group representativeness limited; no dedicated inequality investigation. |
| 4 | Objectives | Introduction, final paragraph | Conventional development and external LLM evaluation. |
| 5a | Data source | 2.1; S1; S6 | Two retrospective institutional cohorts. |
| 5b | Dates | 2.1; S6 | Years 2018–2024 reported; exact accrual dates not supplied. |
| 6a | Setting | 2.1 | Two named centres and countries/cities. |
| 6b | Eligibility | 2.2; S1; Table S2 | 27 recorded retrospective exclusions reconciled in Table S2; these are clinical screening decisions, not a prospective eligibility protocol. |
| 6c | Treatment | 2.4; 3.1; Table 1 | Centre-specific exposure; inputs not uniformly treatment-naïve. |
| 7 | Preparation | 2.5–2.6; S2–S3 | Training-only preprocessing; no group-specific preprocessing audit. |
| 8a | Outcome | 2.4; S1 | Necrosis ≥90% from surgical pathology; patient-level label. |
| 8b | Outcome assessors | 2.4 | Professional pathologists; exact assessor numbers and demographics not supplied. |
| 8c | Outcome blinding | 2.4; 2.8; 3.6 | External-label access during inference audited. Blinding of original pathology readers to clinical measurements not established. |
| 9a | Predictor candidates | 2.4; S2; Table S1 | Candidate dictionary and fold-local selection supplied. |
| 9b | Predictor definitions | 2.4; S3; Table S1 | Same earliest-available preoperative record rule at both centres; prior therapy allowed; exact measurement windows not established. |
| 9c | Predictor assessors | 2.4 | Qualifications of original tumour-diameter assessors not supplied. |
| 10 | Study size | 2.2; 2.6; Discussion | Available retained cohort; no formal precision/power calculation. |
| 11 | Missingness | 2.2; 2.5; S1; Table S1 | Exclusion flag distinguished from clinical decisions; training-derived imputation. |
| 12a–c | Model development | 2.6; S2 | Site split, tuning, selection, transformations and internal evaluation described. |
| 12d | Clustering | 2.1; 3.1 | Site-separated evaluation; no multi-cluster random-effects model. |
| 12e | Evaluation measures | 2.7–2.10; Figures 2–4 | Discrimination, probability error and calibration; paired performance differences with confidence intervals. |
| 12f | Updating | 2.6; S6 | No external recalibration or refitting. |
| 12g | Prediction procedure | 2.6; 2.8–2.10; S7; S2–S4 | Locked conventional preprocessing and hosted cohort-batched inference. |
| 13 | Class balance | S3; archived source | Label-balanced intermediate demonstrations; natural full-pool composition. Conventional settings preserved in source. |
| 14 | Fairness | Discussion; S6 | Not evaluated; no fairness claim. |
| 15 | Outputs | 2.6; 2.8; S4 | Probabilities; thresholds 0.490142 and 0.500. |
| 16 | Between-centre differences | 2.4; 3.1; Table 1 | Case-mix and treatment differences acknowledged; a common baseline definition does not imply identical measurement windows. |
| 17 | Ethics | Declarations; 2.3 (data protection) | Committee, approval 2026140, multicentre scope and consent waiver are reported from author confirmation. |
| 18a–b | Funding and interests | Declarations | Author fields remain to be completed. |
| 18c | Protocol | S4; archived protocols | Local initial protocol and amendments retained; no invented public registration. |
| 18d | Registration | Archived protocols | Local protocol and amendments are archived; no public registration identifier is claimed. |
| 18e–f | Data and code | Declarations; supplementary files | Code and aggregates accompany the article; a GitHub upload package is prepared. The public repository is https://github.com/8426-ztc/Supporting_files; restricted-data requests go through the corresponding author, subject to institutional approval. No reuse licence has been assigned. |
| 19 | Patient/public involvement | Author completion | Not documented in supplied sources; no absence or involvement invented. |
| 20a | Flow | 2.2; 3.1; Figure 1; S1 | 2395 screened, 193 reviewed, 27 excluded, 166 retained. |
| 20b–c | Cohort characteristics | 3.1; Tables 1 and S1 | Per-centre distributions, outcomes and missingness reported. |
| 21 | Analysis sample sizes | 2.6; 2.8; 3.1; source tables | Development 109/55 positive; external 57/25; round-specific evaluable counts in source. |
| 22 | Model specification | 2.6; S2–S4; S6; code | Detailed source supplied; full independent regeneration needs approved inputs. No portable hosted snapshot. |
| 23a | Performance | 3.2–3.5; Figures 2–4 | Primary estimates and intervals reported; subgroup metrics not evaluated. |
| 23b | Performance heterogeneity | 3.1; Discussion | No additional multicentre heterogeneity analysis. |
| 24 | Model updating results | S6 | Not applicable: models not updated on external data. |
| 25 | Interpretation | Discussion | Paired differences do not establish advantage or equivalence. |
| 26 | Limitations | Discussion; S1; S6 | Selection, timing, sample size, calibration and cohort-batched limitations visible. |
| 27a–c | Use and next validation | Discussion; S6 | Input checks, patient-wise evaluation and workflow monitoring are future requirements, not implemented controls. |
