# philgeps-ml-project
Predicting which PhilGEPS public biddings end without a contract: an ML risk score to help suppliers decide which government tenders to bid on

## Dataset

Raw PhilGEPS bid notice and award data is in [`data/raw/`](data/raw/). It covers January 2023 to September 2026 in 15 quarterly files (about 3.6 GB in total). Each row is one bid line item, joined with its award details if there is one (`Bid Notice Status`, award status, `Contract Amount`, awardee info, etc.). Missing values are written as the literal string `NULL`.

The files are tracked with [Git LFS](https://git-lfs.com/). Install Git LFS before cloning, or run `git lfs pull` after cloning to download the actual data.

The files are **not all in the same format**:

| Files | Delimiter | Encoding | Columns |
|---|---|---|---|
| `2023_*` to `2025_(JUL-SEPT).csv` (11 files) | comma | UTF-8 with BOM | 46, original names |
| `2025_Oct-Dec.csv` | **tab** | **Windows-1252** | 45 + one empty trailing column, original names |
| `DATASETS_MPHILGEPS_2026_*.csv` (3 files) | comma, quoted | UTF-8 | 45, **renamed** (e.g. `Approved Budget for the Contract (ABC)`, `Procurement Category`, `Source of Funds`, `Awardee Size (MSME Category)`) |

Differences between the 2026 files and the 2023–2025 files that matter when combining them:

- `Funding Instrument` is missing. `Trade Agreement` became `Applicable Procurement Rules`, and `Funding Source` became `Source of Funds` with different values (e.g. `Regular Agency Fund (01000000)` instead of `Government of the Philippines (GOP)`).
- `Calendar Type` / `Line Item No` became `Lot Type` / `Item Code`. `UNSPSC Description` is labelled `Item/Lot Name`, which duplicates an earlier column name (pandas renames it `Item/Lot Name.1`).
- Dates are `24-Jan-2026` (`Closing Date` includes a time) instead of `20/01/2023` (day first).
- `Classification` is coarser (`Goods` instead of `Goods - General Support Services`).

```python
import pandas as pd

kw = dict(na_values="NULL", low_memory=False)
df_2023 = pd.read_csv("data/raw/2023_(JAN-MAR).csv", encoding="utf-8-sig", **kw)
df_q4_2025 = pd.read_csv("data/raw/2025_Oct-Dec.csv", sep="\t", encoding="cp1252", **kw)
df_2026 = pd.read_csv("data/raw/DATASETS_MPHILGEPS_2026_JAN-MAR.csv", **kw)
```
