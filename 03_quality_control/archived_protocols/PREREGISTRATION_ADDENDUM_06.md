# Preregistration addendum 06 — cohort-batched rounds

Date added: 2026-08-29 (Asia/Shanghai)

Status at addition: 37 of 6,954 predictions from the superseded one-patient-per-
call pilot had completed. No validation outcome had been accessed and no
validation performance metric had been calculated. The pilot artifacts remain
archived in `publication_run_20260829_v2` and will not be pooled with the new
analysis.

## User-directed operational redesign

One round is now one model call for one frozen `shot × replicate` condition.
Each round requests predictions for all 109 pseudonymous development patients
and all 57 pseudonymous external-validation patients in a single structured
response. The frozen demonstration sets, shot sizes, replicate counts, model,
reasoning effort and decision threshold are unchanged. There are 122 rounds:
one zero-shot round, 20 rounds for each of 4, 8, 10, 20, 40 and 80 shots, and
one full-pool 109-shot round.

The new formal run directory is `publication_run_20260829_v3_batched`. It is a
new estimand and must not be combined numerically with the superseded pilot.

## Leakage and interpretation boundary

Validation labels are never included in prompts and remain locked until all 122
rounds and 20,252 patient-level outputs are complete, audited and frozen.

For development patients, a patient used as a labeled demonstration in a round
has direct label exposure. Its prediction is retained because the user requires
all development outputs, but is marked `label_exposed=1` and prohibited from
development performance calculations for that round. Consequently:

- zero-shot development performance can use all 109 development patients;
- intermediate-shot development performance may use only non-demonstration
  patients and is exploratory because the evaluable subset varies by round;
- 109-shot development performance is not estimable without direct label
  leakage and will be reported as not applicable;
- no resubstitution result may be described as training discrimination.

Because all query features are visible together, this design is a
cohort-batched or transductive evaluation. Predictions within a round may depend
on other patients' feature distributions and are not equivalent to prospective
single-patient inference. This limitation must appear in Methods, Discussion
and reviewer materials.

## Frozen execution and audit

Every round uses a new ephemeral `gpt-5.6-sol` session with high reasoning, a
read-only empty workspace, a fixed structured-output schema and explicit
prohibition of tools, files, web search, memory and external information. The
response must contain exactly 166 unique pseudonymous IDs in frozen query order.
All raw responses, JSONL events, stderr, prompts, settings, retries and hashes
are retained. A failed round may be retried only under identical settings; no
answer selection or prompt adaptation is allowed.

The first and only unblinding occurs after complete coverage, zero duplicate or
unexpected IDs, correct thresholding, matching prompt/response/event hashes,
valid ledger chains and zero forbidden tool events. The primary external-
validation analysis remains the preregistered 20-shot AUROC comparison; other
shot curves and all development results are exploratory.

