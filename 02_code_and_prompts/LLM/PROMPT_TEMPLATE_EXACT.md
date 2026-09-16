# Exact prompt structure used in the frozen experiment

This file reports the exact static instructions and the exact placeholder structure. Patient-level values and demonstration labels are intentionally omitted from the public version. The 122 exact payloads are represented by SHA-256 hashes in `prompt_condition_manifest.csv` and are retained in the separate controlled-review package.

## 1. System prompt

```text
You are evaluating a binary prediction task. Predict whether MPR90 is achieved from four pre-treatment features.

MPR90 means MPR >= 0.90. Use the labeled reference cases only as in-context demonstrations. Do not use outside patient information and do not infer validation labels.

Return exactly one JSON array with 166 objects, in the same order as the input cases. Each object must contain only:
  id: the provided case_id string
  pred_label: 0 or 1
  prob_mpr90: number from 0 to 1, rounded to three decimals

Set pred_label=1 when prob_mpr90 >= 0.500. Do not output explanations, confidence scores, markdown, or additional fields.
```

## 2. User-prompt template

For 0-shot:

```text
No labeled demonstrations are provided. This is the zero-shot condition.
```

For k-shot conditions:

```text
Labeled demonstrations:
1. case_id={DEVELOPMENT_PSEUDONYM}: BMI (kg/m^2)={VALUE_3DP}, baseline tumor maximum diameter (mm)={VALUE_3DP}, AST (U/L)={VALUE_3DP}, TBIL (umol/L)={VALUE_3DP}; observed_label={0_OR_1}
...
k. {SAME_FORMAT}
```

This was followed by both frozen query blocks:

```text
Unlabeled development query cases:
1. case_id={DEVELOPMENT_PSEUDONYM}: BMI (kg/m^2)={VALUE_3DP}, baseline tumor maximum diameter (mm)={VALUE_3DP}, AST (U/L)={VALUE_3DP}, TBIL (umol/L)={VALUE_3DP}
...
109. {SAME_FORMAT_WITHOUT_LABEL}

Unlabeled external-validation query cases:
110. case_id={VALIDATION_PSEUDONYM}: BMI (kg/m^2)={VALUE_3DP}, baseline tumor maximum diameter (mm)={VALUE_3DP}, AST (U/L)={VALUE_3DP}, TBIL (umol/L)={VALUE_3DP}
...
166. {SAME_FORMAT_WITHOUT_LABEL}

Return predictions for all 166 query cases in the displayed order. Return only the required JSON array.
```

Validation query rows never contained `observed_label`. Development query rows also omitted the label; only the separate demonstration block contained labeled development examples.

## 3. Terminal runtime suffix appended to every call

```text
This is a frozen, outcome-blind cohort-batched evaluation. Do not use tools, files, web search, memory, connectors, or external information. Do not omit, add, reorder, or rename any query ID. For transport-schema compatibility, wrap the required JSON array in an object with the single key predictions. Return only that JSON object.
```

The terminal suffix explains the apparent transport difference between the earlier request for a JSON array and the recorded final schema: the accepted last message was an object with the single key `predictions`, whose value was the ordered 166-object array.

## 4. Accepted output schema

```json
{
  "predictions": [
    {
      "id": "{PROVIDED_CASE_ID}",
      "pred_label": 0,
      "prob_mpr90": 0.000
    }
  ]
}
```

Exactly 166 unique expected IDs were required, in the displayed order. No additional keys were accepted. `pred_label` had to equal 1 if and only if `prob_mpr90 >= 0.500`.

## 5. Execution context

- One independent ephemeral call per shot × replicate condition.
- Model: `gpt-5.6-sol`; reasoning effort: `high`.
- Read-only empty workspace; no tools, files, web, memory, connectors or external information.
- Shot conditions: 0 and 109 once; 4, 8, 10, 20, 40 and 80 each repeated 20 times.
- Provider sampling seed: unavailable. Demonstration-set sampling used a fixed documented seed schedule.
- Full prompt payload archive SHA-256: `f2d205522755f5d4bc7940ca7d07481ecd6adee23c58cc6e453203b327f5f84b`.

