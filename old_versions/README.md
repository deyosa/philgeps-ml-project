# Old versions (archived, do not use)

These notebooks are **superseded** and kept only as a record of earlier work. They are **not part of the pipeline**: nothing in `scripts/` or `notebooks/` reads them or their outputs, and their numbers do not describe the current dataset, target or model.

- Do not run, edit or cite them as current results.
- They read files that are no longer the main input (the old raw file names, or `data/processed/philgeps_combined.parquet`), and they predate the current outcome rule (Failed or Cancelled = 1, Awarded = 0, Closed removed, 2023 Q1 to 2026 Q2).

| Notebook | Original author | Superseded by |
|---|---|---|
| `philgeps_dataset_initial_analysis.ipynb` | johnlagac | `notebooks/01_data_preparation_and_eda.ipynb` |
| `philgeps_dataset_initial_analysis_updated.ipynb` | johnlagac | `notebooks/01_data_preparation_and_eda.ipynb` |
| `2.0 and 1.5 Data Exploration.ipynb` | Diluni_Peiris | `notebooks/00_data_cleaning.ipynb` and `notebooks/01_data_preparation_and_eda.ipynb` |
| `2.0 and 1.5 Data Exploration_Cleaned.ipynb` | Diluni_Peiris | `notebooks/00_data_cleaning.ipynb` and `notebooks/01_data_preparation_and_eda.ipynb` |

The current process is described in the repository [README](../README.md).
