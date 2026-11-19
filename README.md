# philgeps-ml-project
Predicting government procurement failure on PhilGEPS: at the time a bid notice is posted, how likely is it that the procurement will **fail or be cancelled** instead of being **awarded**?

An early warning lets procuring entities fix a bid (budget, timeline, specifications) before it fails, which avoids re-bidding delays and wasted procurement effort. The model is intended for risk screening and review prioritisation. It is **not** a model of whether a particular supplier will win a contract.

Outcome definition (one record per bid): `1` when the bid ended `Failed` or `Cancelled`, `0` when it was `Awarded`. Bids whose status is `Closed` are removed during cleaning, and bids still `Active`, `Shortlisted`, `Concluded` or `Published` have no outcome yet and are left out. All procurement modes are included. Award and awardee fields are excluded from the predictors, because they are only filled in after the outcome.

Study window: **2023 Q1 to 2026 Q2**. The 2026 Q3 files are in `data/raw/`, but the cleaning notebook does not read them.

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

## Pipeline

The project runs in two notebooks in [`notebooks/`](notebooks/):

| Step | Notebook | Output |
|---|---|---|
| 0 | [`00_data_cleaning.ipynb`](notebooks/00_data_cleaning.ipynb): reads the 28 raw files from 2023 Q1 to 2026 Q2, repairs the 2025 Q4 column shift, renames the 2.0 headers, merges everything, and drops exact duplicate rows and `Closed` rows | `data/raw/cleaned_data/PhilGeps_Cleaned.csv` (2,093,210 line items, 1.6 GB). **This is the main dataset.** |
| 1 | [`05_final_modeling.ipynb`](notebooks/05_final_modeling.ipynb): **the final model.** Merges the 1.5 and 2.0 column layouts, keeps bids with a final outcome, builds one row per bid, removes leaky fields, engineers features, tunes hyperparameters with 5-fold cross-validation, and trains 7 algorithms in 3 scenarios, followed by feature importance and a look at the drivers of failure | models, metrics and charts in the notebook |

The cleaned CSV is committed with Git LFS, like the raw files, so `git lfs pull` downloads it and you can go straight to notebook 05. Notebook 00 only needs re-running if the raw data changes; it holds all 11 million raw rows in memory at once and needs well over 16 GB of RAM.

Notebook 05 needs `xgboost` and `imbalanced-learn` besides the usual scientific Python stack. A full run takes about 1 to 1.5 hours and about 8 GB of free RAM.

To download only what notebook 05 needs, and save Git LFS bandwidth:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/deyosa/philgeps-ml-project.git
cd philgeps-ml-project
git lfs pull --include="data/raw/cleaned_data/*"
```

The exact-duplicate step in notebook 00 removes 5,868,193 rows: every row of the twelve `2.0_2023_*` to `2.0_2025_*` files (byte-for-byte copies of the matching `1.5_*` files), plus repeated identical line items within the 1.5 exports.

## Final model

Notebook 05 compares Logistic Regression, Ridge (L2), Lasso (L1), KNN, Random Forest, XGBoost and HistGB in three scenarios:

| Scenario | Training data | Test data |
|---|---|---|
| All features (unbalanced) | Random 80% of bids (1.28M), stratified by outcome; no bid in both sets | Remaining 20% (320k bids) |
| Undersampling 60/40 | Same training bids, undersampled to 60% Awarded / 40% Failed inside the pipeline | Same test set |
| Time-based split | Bids published before 1 July 2025 | Bids from 1 July 2025 to June 2026 |

Results (test set, positive class = Failed or Cancelled):

- **Best model: XGBoost** (`max_depth` = 10) in the main scenario, with **PR AUC 0.658** (no-skill baseline 0.151) and **ROC AUC 0.869**. At a 0.5 threshold, precision is 0.835 and recall 0.362; the F1-maximising threshold (about 0.28) gives precision 0.584 and recall 0.566.
- The tree ensembles beat the logistic regressions (PR AUC about 0.577) and KNN (0.532). The penalty makes no practical difference to the logistic regressions.
- **Undersampling** does not improve any model; it only shifts the predicted probabilities.
- **Over time** performance drops: on the time-based split XGBoost reaches PR AUC 0.561 and ROC AUC 0.793, and Random Forest is best (0.589 / 0.806). These numbers are the more realistic estimate for deployment.
- **Key drivers:** the procuring entity's track record (its historical failure rate), then procurement mode, location, whether the notice has a bidding window, and very small or very large budgets.

## Data caveats

Found while building the earlier version of the project (see [`old_versions/README.md`](old_versions/README.md)); they apply to the cleaned dataset that notebook 05 uses:

- In the PhilGEPS 1.5 exports for **2025 Q3–Q4**, award records are largely missing. Labelled bids from those quarters lean towards cancellations, which affects the time-based test period (from July 2025). Re-downloading the files from PhilGEPS gave identical files.
- Notebook 00 decodes `1.5_2025_Q4.csv` as Latin-1 instead of Windows-1252, so a few thousand notice titles from 2025 Q4 contain garbled characters (for example `Multi\x96Purpose`).
- A failed or cancelled procurement is often re-advertised under a new bid reference with the same procuring entity, title and budget. A random split by bid can put the original and the re-advertisement on different sides.

## Archive

Earlier versions are kept read-only in [`old_versions/`](old_versions/): the original documented run (`LT8-ML_Govt_Bidding_Documented.ipynb`), the initial dataset analyses, the first data-exploration notebooks, and the second version of the project, the Public Bidding pipeline (`public_bidding_pipeline/`). They are not part of the current project and their numbers are out of date; see [`old_versions/README.md`](old_versions/README.md).
