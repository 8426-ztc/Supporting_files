# Adapting the prompt and evaluation workflow to a new clinical task

This guide accompanies the executed prompt template in Supplementary Section S3. It describes a proposed adaptation workflow, not an experiment performed in the present study. Keep the original executed prompt unchanged as the record of the HCC benchmark and save any adaptation as a new version.

## Define the task before writing the prompt

Specify the clinical population, prediction time, binary outcome and reference assessment. State which information is available at prediction time. Replace every occurrence of the original endpoint, including its label, probability field and examples; changing only the task title leaves an inconsistent prompt. In particular, do not carry the original prompt's historical “pre-treatment” wording into a task with a different observation window.

## Construct the inputs from development data

Select predictors and fit imputation or transformation rules using development data only. Document names, units, transformations, missing-value handling and measurement times. The four predictors selected for this HCC cohort are not a validated feature set for other tasks. Exclude variables that reveal the outcome or occur after the intended prediction time. Apply the saved preprocessing rules to validation inputs without refitting.

## Replace the task-specific fields

| Template element | Adaptation required |
|---|---|
| Clinical instruction | New population, endpoint, prediction time and probability meaning |
| Predictor fields | New development-selected variables with explicit units and transformations |
| Labelled examples | Development cases only, with labels defined for the new task |
| Query records | Unlabelled records in the chosen evaluation setting, using pseudonymous IDs |
| Output schema | Consistent ID field, outcome probability and any prespecified label rule |
| Checks | Expected IDs, completeness, uniqueness, finite probabilities in [0,1] and any label–probability consistency rule |

## Prespecify the evaluation

Choose the demonstration budget, number of repeated runs, sampling rules, primary metric and comparator before inspecting validation outcomes. Twenty examples per prompt was the primary condition for this benchmark; it is not an established optimum for a new task. Select and lock conventional comparators using development data. Decide whether queries will be presented individually or jointly. The executed HCC study used cohort-batched queries; switching to patient-wise inference creates a different procedure that requires evaluation.

Record the prompt version, available model identifier and settings, sampled demonstrations, accepted output and failed attempts. Use a prespecified retry rule that does not select responses by predictive performance. Validate and freeze predictions before accessing the held-out outcomes. Retain patient-level pairing when comparing methods and account for repeated inference without treating the number of output rows as the number of independent patients.

## Assess what transferred

Evaluate discrimination, probability accuracy, calibration and variability across runs in the new validation data. Mean run-specific AUROC and AUROC of averaged patient probabilities answer different questions; define the intended target. Reusing a prompt structure does not establish transfer of its measured performance, nor guarantee exact regeneration of a hosted-model response. Clinical use requires validation in the intended setting and the relevant institutional data-use permissions.

The present revision provides editable research materials. Repository deposition remains pending in the manuscript availability statement; this guide does not claim that a public repository has already been published.
