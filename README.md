# Supporting files for the outcome-blinded in-context prediction benchmark

Repository: https://github.com/8426-ztc/Supporting_files

This compact release accompanies manuscript versions. It provides executed code, a patient-free prompt template, selected aggregate results and non-identifying execution records. It contains no patient-level clinical dataset, patient-level predictions, instantiated prompts, linkage key or figure files.

## 📁 Directory Structure 

- `01_tables_and_source_data`: 11 aggregate result/configuration files and an interpretation guide, covering the headline estimates, all nine conventional models, demonstration budgets, run variation and the paired comparison.
- `02_code_and_prompts`: conventional modelling, LLM inference/evaluation and paired-analysis source, prompt template, condition manifest and adaptation guidance. Figure-only and publication-packaging utilities are omitted.
- `03_quality_control`: aggregate execution summary, outcome-access record and archived protocol/addenda. Local dated protocols document recorded planning; they are not public preregistration records.
- `04_reporting`: reporting cross-reference and analysis provenance.
- `MANIFEST_SHA256.csv` and `verify_package.py`: byte-level release integrity verification.

Read `01_tables_and_source_data/README.md` and `04_reporting/Analysis_provenance_v12.md` before interpreting results. Detailed clinical tables and figures accompany the article. No additional clinical data are implied by the folder name source_data.

## Reproducibility 
All data and results can be traced via the SHA256 manifest.Code and aggregates support methodological inspection and aggregate consistency checks. Refitting models or recomputing patient-level metrics requires approved inputs not distributed here. Historical scripts may refer to the original local directory layout; they are archived research source, not a turnkey installation. Hosted-model responses are not guaranteed to regenerate exactly. Figure-generation helpers mentioned in historical code readmes are retained in the journal package, not this compact release.

## access

Patient-level data, predictions and instantiated prompts are restricted because they contain sensitive human-participant information. Qualified researchers may contact the corresponding author listed on the manuscript title page; access remains subject to institutional approval, an appropriate ethics determination and a data-use agreement. No unrestricted data-access commitment or reuse licence is implied.

## Integrity

The conventional modelling script retains its frozen SHA-256: 1f3c5ef4604c933ca59d1c9385b6ce63a8580a9663efbb9234dff1b68b8ca95b. Historical hashes in protocols identify archived analysis artifacts, including restricted artifacts; the root manifest identifies files actually distributed in this release. `.gitattributes` disables automatic text conversion to preserve these bytes. Run `python verify_package.py` after downloading or cloning the release.

This release changes documentation and distribution scope only; model results are unchanged. It is intended to replace the previous repository tree, not to be merged on top of it while leaving superseded files in place.



