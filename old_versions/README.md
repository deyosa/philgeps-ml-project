# Old versions (archived, do not use)

Everything in this folder is **superseded** and kept only as a record of earlier work. None of it is part of the current project: nothing in `notebooks/` reads these files or their outputs, and their numbers do not describe the current model.

- Do not run, edit or cite them as current results.
- **The current project** is `notebooks/00_data_cleaning.ipynb` (cleaned dataset) followed by `notebooks/05_final_modeling.ipynb` (final model). See the repository [README](../README.md).

## Versions, oldest first

| Version | What | Original author | Why it is superseded |
|---|---|---|---|
| 1 | [`LT8-ML_Govt_Bidding_Documented.ipynb`](LT8-ML_Govt_Bidding_Documented.ipynb): the original analytical run on 14 converted quarterly files (Public Bidding, Failed vs Awarded, train 2023–2024, validate 2025) | JenQ | Re-run on the merged 1.5 + 2.0 data, then replaced by notebook 05 |
| – | `philgeps_dataset_initial_analysis.ipynb`, `philgeps_dataset_initial_analysis_updated.ipynb`: initial dataset profiling | johnlagac | Replaced by the data preparation in later versions |
| – | `2.0 and 1.5 Data Exploration.ipynb`, `2.0 and 1.5 Data Exploration_Cleaned.ipynb`: first data-exploration pipeline | Diluni_Peiris | Replaced by `notebooks/00_data_cleaning.ipynb` |
| 2 | [`public_bidding_pipeline/`](public_bidding_pipeline/): harmonisation script and notebooks 01–04 (Public Bidding only, gradient boosting) | Deborah Argayosa | Replaced by `notebooks/05_final_modeling.ipynb` |

## Version 2: the Public Bidding pipeline

[`public_bidding_pipeline/`](public_bidding_pipeline/) holds the second version of the project, archived as a whole because notebook 05 uses a different scope, split, feature set and model:

| | Version 2 (archived) | Notebook 05 (current) |
|---|---|---|
| Input | Cleaned CSV → `scripts/harmonise_cleaned.py` → Parquet | Cleaned CSV, prepared inside the notebook |
| Scope | Public Bidding only, 2025 Q3–Q4 left out | All procurement modes, 2023 Q1 to 2026 Q2 |
| Split | Random 70/15/15 by notice group (re-advertisements kept together) | Random 80/20 by bid, plus a time-based split at 1 July 2025 |
| Models | 4 algorithms × 2 feature sets; calibrated gradient boosting | 7 algorithms × 3 scenarios; XGBoost |
| Test result | ROC-AUC 0.772, PR-AUC 0.522 (random: 0.192) | ROC-AUC 0.869, PR-AUC 0.658 (random: 0.151) |

Contents:

- `notebooks/01_data_preparation_and_eda.ipynb`: raw-file inventory and checks, bid-level table, outcome rule, the 2025 Q3–Q4 exclusion, re-advertised notices
- `notebooks/02_modeling.ipynb`: grouped split, model comparison, calibration, confusion matrices, sensitivity checks
- `notebooks/03_diagnostics_and_interpretation.ipynb`: out-of-time check, drift, importance, budget patterns, findings
- `notebooks/04_project_summary.ipynb`: auto-updating summary of 01–03
- `notebooks/procurement_failure.py`: definitions shared by 01–04
- `scripts/harmonise_cleaned.py`, `scripts/run_pipeline.py`, `scripts/combine_raw.py`: harmonisation, pipeline runner, raw-file cross-check
- `data/reference/public_bidding_notices_per_year.csv`: yearly Public Bidding volume including Closed notices, used by notebook 04

The files were moved without changing their code, so their paths (for example `scripts/harmonise_cleaned.py`, `data/processed/`) refer to the repository layout before archiving. They are not maintained and may not run from here.

Several checks from version 2 are still relevant to the current cleaned data, and can be read in these notebooks' saved outputs:
- The cleaned CSV matches an independent rebuild from the raw files (2,093,210 line items).
- The `2.0_2023_*` to `2.0_2025_*` raw files are byte-for-byte copies of the 1.5 files.
- In the 1.5 exports for 2025 Q3–Q4, award records are largely missing, so labels for those quarters lean towards cancellations. Re-downloading gave identical files.
- Notebook 00 decodes `1.5_2025_Q4.csv` as Latin-1 instead of Windows-1252, which garbles a few thousand titles.
