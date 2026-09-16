# Preregistration addendum 07 — Codex CLI execution epoch 2

Date added: 2026-08-30 (Asia/Shanghai)

Status at discovery: 37 of 122 cohort-batched rounds (6,142 of 20,252
patient-level outputs) had completed. The outcome-blind inference audit passed,
no forbidden tool event had occurred, no validation outcome had been accessed,
and no validation performance metric had been calculated.

## Trigger

An attempted continuation stopped before sending a new prompt because the Codex
desktop application had automatically replaced the frozen command-line runtime.
The epoch-1 runtime was `codex-cli 0.150.0-alpha.12.2` at
`...\fac60c5e9a2ae3df\codex.exe`; that binary is no longer present. The available
runtime is `codex-cli 0.151.0-alpha.7.1`, SHA-256
`3052f7887c10e97f6cfe4941353bd0763300c4907e49a8688958cb40d4159d89`.
The existing runner correctly blocked execution because its immutable state
manifest detected both path and version changes.

Immediately before this update, four pending 8-shot rounds had each exhausted
two attempts because the account usage limit was reached. These eight failed
attempts produced no accepted predictions; their event and stderr artifacts
were preserved by hash. The outcome-blind audit still passed at 37 rounds.

## Frozen operational continuation

Continuation is permitted as execution epoch 2 only. Epoch 1 manifests,
ledgers, raw responses, events and round outputs remain immutable. Epoch 2 uses
new state manifests and new hash-chained ledgers while writing successful rounds
to the same condition-keyed output store. Completed condition keys from either
epoch are skipped, so no completed round is rerun or pooled twice.

The following scientific and inference settings are unchanged:

- `gpt-5.6-sol` with high reasoning;
- one independent ephemeral call per `shot × replicate` round;
- the exact frozen prompts, 166-query order and demonstration sets;
- the exact structured-output schema and 0.500 threshold;
- read-only empty workspace and prohibition of tools, files, web search,
  memory, connectors and external information;
- retry, label-exposure, blinding, statistical and unblinding rules.

The CLI transition is an unavoidable operational dependency change, not a
planned model comparison. Runtime epoch will be retained in audit materials and
reported as a limitation/sensitivity concern. No output may be selected or
excluded based on apparent predictive quality.

