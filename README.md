# philgeps-ml-project
Predicting which PhilGEPS public biddings end without a contract: an ML risk score to help suppliers decide which government tenders to bid on

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

## Notebooks

- [`data_exploration/2.0 and 1.5 Data Exploration.ipynb`](data_exploration/2.0%20and%201.5%20Data%20Exploration.ipynb): data analysis pipeline that reads every file in `data/raw/`, merges them, drops duplicate rows, and checks missing values
- [`philgeps_dataset_initial_analysis.ipynb`](philgeps_dataset_initial_analysis.ipynb) and [`philgeps_dataset_initial_analysis_updated.ipynb`](philgeps_dataset_initial_analysis_updated.ipynb): consolidate every CSV in `data/raw/` into one DataFrame and profile fields, data types, missing values and distributions
- [`LT8-ML_Govt_Bidding_Documented.ipynb`](LT8-ML_Govt_Bidding_Documented.ipynb): data preparation, exploratory analysis and failure-risk modelling for Public Bidding (written for the earlier 14-file set, 2023 Q1 to 2026 Q2)

The notebooks that read every file in `data/raw/` will pick up the duplicate `2.0_2023_*` to `2.0_2025_*` files. Exclude them or deduplicate after loading.
