# Reproducibility code guide

## Core workflow

1. `llm_icl_benchmark.py` — cohort preparation, pseudonymization, demonstration sampling, baseline model and core metric utilities.
2. `prepare_batched_rounds.py` — converts the frozen study into 122 cohort-batched prompt conditions.
3. `codex_round_inference.py`, `codex_round_inference_epoch02.py`, `codex_round_inference_epoch03.py` — immutable execution epochs. Epoch 2 and 3 exist only because the desktop CLI changed and the local timeout was amended while outcomes remained locked.
4. `archive_failed_attempts.py` — preserves failed attempt artifacts rather than selecting or silently discarding them.
5. `audit_round_inference.py` — validates prompts, expected IDs, state, ledgers, events, settings and completed responses across epochs.
6. `freeze_round_predictions.py` — creates and hashes the complete prediction file before outcome access.
7. `evaluate_batched_rounds.py` — performs the first and only unblinding and computes patient- and prompt-replicate-aware statistics.
8. `generate_publication_package.py`, `plot_publication_figures.py`, `plot_development_figure.py` and workbook scripts — generate publication outputs and QA artifacts.

## Important reproducibility boundary

The public code and prompt template can reproduce the workflow structure, but exact model outputs may not be bitwise reproducible because the hosted model is proprietary, a provider sampling seed was unavailable and service-side model behaviour can change. The retained prompt hashes, output hashes, runtime states, failure logs and frozen prediction file allow the reported run to be authenticated.

## Data required for an authorized rerun

An authorized rerun requires the four pre-treatment variables, study-specific pseudonyms and binary development labels. Validation labels must remain outside all prompt/operator files until predictions are complete and frozen. Do not reconstruct or release the raw identifier mapping or pseudonymization key in the public repository.

