# Preregistration Addendum 08: operational timeout epoch

## Timing and blinding status

This addendum was written before any execution-epoch-3 model call and while validation outcomes remained locked. At this boundary, 114 of 122 prespecified cohort-batched conditions (18,924 of 20,252 ordered patient predictions) had passed the outcome-blind audit; eight conditions remained. The most recent audit was `PASS` with report SHA-256 `cb51227802242081f8daa4c3810dc1b2c9c89ced0be6dfed6c06c2bbb8a95733`.

## Trigger

During execution epoch 2, two 80-shot subprocesses exceeded the 900-second local orchestration timeout:

- `shot080_rep08`, shard 0; batch stderr SHA-256 `60b5b6b4375da775634a4ae5cbadf031f7e4b03711409a5a04a168e2644a0057`.
- `shot080_rep07`, shard 3; batch stderr SHA-256 `2a263aecba86bf924607279b07d5899f0f5be843872a282ec2480bd679620052`.

These attempts produced no accepted round file and were not used for performance selection. Their orchestration logs are retained. The epoch-2 runner did not convert operating-system `TimeoutExpired` exceptions into inference-ledger records, creating an operational audit gap even though the batch logs preserved the failures.

## Frozen operational correction for execution epoch 3

Execution epoch 3 is limited to the eight still-incomplete prespecified conditions. It preserves the exact frozen prompts, expected ordered IDs, JSON schema, model (`gpt-5.6-sol`), reasoning effort (`high`), ephemeral execution, read-only empty workspace, no-tool rule, decision threshold, shard assignment and maximum of two attempts. The only intentional operational change is:

- per-attempt orchestration timeout: 1,800 seconds, frozen in every epoch-3 state manifest.

The epoch-3 runner also records any future operating-system timeout as a hash-chained `PROMPT_ATTEMPT_FAILED` event, including the timeout value and hashes of the captured event/stderr artifacts. This correction does not inspect validation outcomes, alter prompts, choose among successful answers or change the scientific estimand.

Execution-epoch-3 state and ledger files are separate from epochs 1 and 2. Completion remains defined only by an exact 166-row ordered response that passes schema, uniqueness, ID-order, threshold, model-setting and forbidden-tool checks. Final unblinding remains prohibited until all 122 conditions pass the consolidated multi-epoch audit and the 20,252-row prediction file is frozen.
