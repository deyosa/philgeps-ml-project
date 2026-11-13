# philgeps-ml-project
Predicting the likelihood of procurement failure for PhilGEPS public biddings: a notice-level risk score, estimated from information available when the bid notice is published.

The model estimates whether a published procurement notice is likely to end in a **failed bidding or a cancellation** rather than an award. It is intended for risk screening and review prioritisation by agencies, oversight bodies and analysts. It is **not** a model of whether a particular supplier will win a contract, and it does not recommend which bids suppliers should pursue.

Outcome definition (one record per `Bid Reference No.`): `Target_Failed = 1` when every line item of the bid is `Failed` or every line item is `Cancelled`, and `0` when every line item is `Awarded`. Bids whose status is `Closed` are **removed** from the data. Bids that are `Active`, `Shortlisted`, `Concluded`, missing or conflicting are left unlabelled and excluded from supervised modelling. Only fields that exist at publication are used as predictors; award and contract fields are excluded.

Study window: **2023 Q1 to 2026 Q2** (by source quarter). The 2026 Q3 files are in `data/raw/` and in the combined file, but the analysis notebooks filter them out (`STUDY_QUARTERS` in [`notebooks/procurement_failure.py`](notebooks/procurement_failure.py)).

## Dataset

Raw PhilGEPS bid notice and award data is in [`data/raw/`](data/raw/). It comes from the [PhilGEPS Open Data](https://philgeps.gov.ph/CmsHomePages/open-data) page. Each row is one bid line item, joined with its award details if there is one (`Bid Notice Status`, award status, `Contract Amount`, awardee info, etc.). Missing values are written as the literal string `NULL`.

The files are tracked with [Git LFS](https://git-lfs.com/). Install Git LFS before cloning, or run `git lfs pull` after cloning to download the actual data.

### File naming

Files are named `<system>_<year>_Q<quarter>.csv`, where `<system>` is the PhilGEPS platform the data came from:

| Files | System | Coverage | Size |
|---|---|---|---|
| `1.5_2023_Q1.csv` to `1.5_2026_Q3.csv` (15 files) | PhilGEPS 1.5 (legacy) | Jan 2023 to Sep 2026 | about 4.2 GB |
| `2.0_2026_Q1.csv` to `2.0_2026_Q3.csv` (3 files) | PhilGEPS 2.0 (Modernized PhilGEPS) | Jan 2026 to Sep 2026 | about 110 MB |

**Use the `1.5_*` files plus the three `2.0_2026_*` files.** The `2.0_2023_*` to `2.0_2025_*` files are byte-for-byte copies of the matching `1.5_*` files (same checksums and the 1.5 column layout), so loading both double-counts 2023 to 2025. PhilGEPS 2.0 data only starts in 2026, when both systems ran side by side. In 2026 Q1 the 1.5 file has about 247k line items and the 2.0 file about 19k.

### Formats

The files are **not all in the same format**:

| Files | Delimiter | Encoding | Columns |
|---|---|---|---|
| `1.5_2023_Q1` to `1.5_2025_Q3` (11 files) | comma | UTF-8 with BOM | 46, original 1.5 names |
| `1.5_2025_Q4` | **tab** | **Windows-1252** (with a few invalid bytes) | 46, original 1.5 names |
| `1.5_2026_Q1` to `1.5_2026_Q3` (3 files) | comma | **Windows-1252** | 47, updated 1.5 names (see below) |
| `2.0_2026_Q1` to `2.0_2026_Q3` (3 files) | comma, quoted | UTF-8 | 45, 2.0 names (see below) |

Column changes in the **1.5 2026** files compared with 1.5 2023 to 2025:

- Added `Solicitation No.` and `Awardee OrgID`. Removed `Calendar Type`.
- Renamed `Business Category` to `Procurement Category`, `Funding Source` to `Source of Funds`, `Trade Agreement` to `Applicable Procurement Rules`, `Area of Delivery` to `Area of Delivery/Project Location`, and `Contract Duration` to `Delivery Period`.
- Dates are still day first, but not always zero-padded (`7/4/2026` as well as `17/04/2026`).

Column changes in the **2.0 2026** files compared with 1.5 2023 to 2025:

- `Funding Instrument` is missing. `Trade Agreement` became `Applicable Procurement Rules`, and `Funding Source` became `Source of Funds` with different values (e.g. `Regular Agency Fund (01000000)` instead of `Government of the Philippines (GOP)`).
- `Calendar Type` / `Line Item No` became `Lot Type` / `Item Code`. Other renames include `Approved Budget for the Contract (ABC)`, `Unit of Measure`, `Award Notice Status (Item)` and `Awardee Size (MSME Category)`.
- `UNSPSC Description` is labelled `Item/Lot Name`, which duplicates an earlier column name (pandas renames it `Item/Lot Name.1`).
- Dates are `24-Jan-2026` (`Closing Date` includes a time) instead of `20/01/2023` (day first).
- `Classification` is coarser (`Goods` instead of `Goods - General Support Services`).

### Loading

```python
import pandas as pd

kw = dict(na_values="NULL", low_memory=False)
cp1252 = dict(encoding="cp1252", encoding_errors="replace")  # skips a few invalid bytes in 1.5_2025_Q4

df_2023 = pd.read_csv("data/raw/1.5_2023_Q1.csv", encoding="utf-8-sig", **kw)
df_q4_2025 = pd.read_csv("data/raw/1.5_2025_Q4.csv", sep="\t", **cp1252, **kw)
df_2026_v15 = pd.read_csv("data/raw/1.5_2026_Q1.csv", **cp1252, **kw)
df_2026_v20 = pd.read_csv("data/raw/2.0_2026_Q1.csv", **kw)
```

## Combined dataset

[`scripts/combine_raw.py`](scripts/combine_raw.py) reads the 15 `1.5_*` files and the three `2.0_2026_*` files, repairs the shifted columns in `1.5_2025_Q4.csv`, maps the three column layouts onto the 1.5 (2023–2025) column names, harmonises spelling-only differences (statuses, `Region`, `Classification`, duration units) and writes one Parquet file:

```bash
python scripts/combine_raw.py      # from the repository root; about 15 minutes
# -> data/processed/philgeps_combined.parquet  (6,059,201 line items, about 420 MB)
```

Every row keeps `Source System` (`1.5` or `2.0`), `Source File` and `Source Quarter`, so it can be traced back to its raw file. The raw files are never modified. `data/processed/` is ignored by Git; rebuild the file locally instead of committing it.

Values that genuinely differ between the systems are left as they are: `Business Category`, `Funding Source` and `Award Notice Status` use different vocabularies in PhilGEPS 2.0. Split by `Source System` before comparing them. The 1.5 and 2.0 exports for 2026 contain different notices (no shared bid references), so both are kept.

Bid-level counts used for modelling (Public Bidding only, 2023 Q1 to 2026 Q2): 1,198,528 bids, of which 568,505 `Closed` bids are removed and 103 more are excluded for data-quality reasons, leaving 629,920. Of these, 615,397 are labelled: 492,282 Awarded (0) and 123,115 Failed or Cancelled (1; 66,250 Failed and 56,865 Cancelled), a 20.0% failure rate. The other 14,523 are unlabelled (`Active`, `Shortlisted`, `Concluded`).

## Analysis notebooks

The analytical roadmap runs in three notebooks in [`notebooks/`](notebooks/), which share their definitions through [`notebooks/procurement_failure.py`](notebooks/procurement_failure.py). Run them in order after `scripts/combine_raw.py`. They write their outputs (the bid table, models and metrics) to `data/processed/`, which Git ignores.

| Notebook | Roadmap steps | Contents |
|---|---|---|
| [`01_data_preparation_and_eda.ipynb`](notebooks/01_data_preparation_and_eda.ipynb) | 1–5 | Source inventory and duplicate check, raw layouts, procurement modes and statuses, unit of observation, 2025 Q4 repair check, harmonisation, consolidation, bid-level table with exception log, outcome rule, label coverage, field availability, class balance |
| [`02_modeling.ipynb`](notebooks/02_modeling.ipynb) | 6 | Random 70/15/15 split of the merged bid table (stratified by outcome and system), baselines, 4 models × 2 feature sets (LT8's original 5, then expanded), calibration, test evaluation, sensitivity with only failed biddings counted as failures |
| [`03_diagnostics_and_interpretation.ipynb`](notebooks/03_diagnostics_and_interpretation.ipynb) | 7–9 | Out-of-time check (train 2023–2024, test 2025/2026 by system), drift, segment performance, label coverage, permutation importance, budget and classification patterns, review-queue simulation, findings |

Headline result: gradient boosting on the expanded at-publication features reaches test ROC-AUC 0.79 and PR-AUC 0.52 (random ranking: 0.20). Reviewing the 10% of notices with the highest predicted risk would catch 31% of failed or cancelled procurements, at 3.1 times the random hit rate. Counting only failed biddings (Cancelled left out) gives ROC-AUC 0.84.

## Other notebooks

- [`philgeps_dataset_initial_analysis_updated.ipynb`](philgeps_dataset_initial_analysis_updated.ipynb): loads the combined dataset, profiles every field, defines the bid-level outcome (how ongoing, closed and cancelled notices are handled, class balance by system and year), classifies fields by availability at publication, and charts every low-cardinality field. Run `scripts/combine_raw.py` first.
- [`LT8-ML_Govt_Bidding_Documented.ipynb`](LT8-ML_Govt_Bidding_Documented.ipynb): the original run of the roadmap on the earlier 14-file set (2023 Q1 to 2026 Q2), kept as a record. Its header gives the procurement-failure framing and answers the reviewer's questions with current numbers, and its last cell points to the `notebooks/` series.
- [`philgeps_dataset_initial_analysis.ipynb`](philgeps_dataset_initial_analysis.ipynb): earlier version of the initial analysis, kept for reference. It reads the raw CSVs directly with the old file names.
- [`data_exploration/2.0 and 1.5 Data Exploration.ipynb`](data_exploration/2.0%20and%201.5%20Data%20Exploration.ipynb): exploration pipeline on the combined dataset: source summary, exact-duplicate rows, missing values, count plots of selected fields (saved to `data/processed/figures/`), and data-scoping notes.
