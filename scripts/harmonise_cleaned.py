"""Turn the cleaned CSV from notebook 00 into the main modelling file.

Usage (from the repository root):

    python scripts/harmonise_cleaned.py

Input:  data/raw/cleaned_data/PhilGeps_Cleaned.csv   (written by notebooks/00_data_cleaning.ipynb)
Output: data/processed/philgeps_cleaned.parquet      (one row per bid line item; read by notebooks 01-03)

Notebook 00 already selects the 2023 Q1 to 2026 Q2 files, repairs the 2025 Q4
column shift, drops exact duplicate rows (which also removes the 2.0_2023-2025
copies of the 1.5 files) and drops Closed rows. This script finishes the
harmonisation that the models need:

  * Adds Source System. A row is PhilGEPS 2.0 when Lot Type is filled; only the
    2.0 layout has that column, and it is filled on every 2.0 row.
  * Merges column pairs that hold the same field under the 2026 names
    (Procurement Category -> Business Category, Source of Funds -> Funding Source,
    Delivery Period -> Contract Duration, ...).
  * Splits durations such as '30 Day/s' into Contract Duration and Calendar Type.
  * Maps PhilGEPS 2.0 statuses, Classification and Trade Agreement labels, and
    Region spellings, onto the 1.5 values, and adds Procurement Mode (Grouped).
  * Parses dates and amounts, and turns float-formatted identifiers ('10346394.0')
    back into plain strings.

The value mappings are shared with scripts/combine_raw.py, which builds the same
line items directly from the raw files and serves as an independent cross-check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combine_raw import (  # noqa: E402
    CLASSIFICATION_20, COLUMNS, DATE_COLS, NUM_COLS, REGION_MAP, STATUS_20, TRADE_AGREEMENT_20,
    group_mode, parse_dates, split_duration,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "cleaned_data" / "PhilGeps_Cleaned.csv"
OUT = ROOT / "data" / "processed" / "philgeps_cleaned.parquet"
CHUNK = 200_000

# The cleaned file has no file lineage, and notebook 00 renames the 2.0 Item Code to Line Item No.
CLEAN_COLUMNS = [c for c in COLUMNS if c not in ("Source File", "Source Quarter", "Item Code")]
SCHEMA = pa.schema([
    (c, pa.timestamp("s") if c in DATE_COLS else pa.float64() if c in NUM_COLS else pa.string())
    for c in CLEAN_COLUMNS
])

# (target column, column holding the same field in the 2026 / 2.0 layouts)
MERGE_PAIRS = [
    ("Business Category", "Procurement Category"),
    ("Funding Source", "Source of Funds"),
    ("Trade Agreement", "Applicable Procurement Rules"),
    ("Area of Delivery", "Area of Delivery/Project Location"),
    ("Contract Duration", "Delivery Period"),
    ("UNSPSC Description", "Item/Lot Name.1"),  # second 'Item/Lot Name' column in 2.0 holds the UNSPSC description
    ("Award Notice Status", "Award Notice Status (Item)"),
]
ID_COLS = ["Bid Reference No.", "Award Reference No.", "UNSPSC Code", "Line Item No", "Awardee OrgID"]
DATE_FORMATS = ["%d/%m/%Y", "%d-%b-%Y", "%d-%b-%Y %I:%M %p"]


def transform(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    is20 = df["Lot Type"].notna()
    df["Source System"] = is20.map({True: "2.0", False: "1.5"})

    for target, other in MERGE_PAIRS:
        both = df[target].notna() & df[other].notna() & (df[target] != df[other])
        stats["merge_conflicts"][target] = stats["merge_conflicts"].get(target, 0) + int(both.sum())
        df[target] = df[target].fillna(df[other])
        df = df.drop(columns=other)

    # Durations written as '30 Day/s' (2025 Q4, 2026 and 2.0 rows) have no Calendar Type yet.
    needs_split = df["Calendar Type"].isna() & df["Contract Duration"].notna()
    if needs_split.any():
        sub = df.loc[needs_split, ["Contract Duration", "Calendar Type"]].copy()
        split_duration(sub)
        df.loc[needs_split, ["Contract Duration", "Calendar Type"]] = sub.to_numpy()

    df.loc[is20, "Bid Notice Status"] = df.loc[is20, "Bid Notice Status"].replace(STATUS_20)
    df.loc[is20, "Classification"] = df.loc[is20, "Classification"].replace(CLASSIFICATION_20)
    df.loc[is20, "Trade Agreement"] = df.loc[is20, "Trade Agreement"].replace(TRADE_AGREEMENT_20)
    df.loc[is20, "Awardee Joint Venture"] = df.loc[is20, "Awardee Joint Venture"].replace({"No JV": pd.NA})
    df["Region"] = df["Region"].map(lambda r: REGION_MAP.get(r.strip().upper(), r.strip()) if isinstance(r, str) else r)
    df["Procurement Mode (Grouped)"] = group_mode(df["Procurement Mode"].astype("string"))

    for c in ID_COLS:
        df[c] = df[c].astype("string").str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    for c in DATE_COLS:
        raw = df[c]
        df[c] = parse_dates(raw, DATE_FORMATS)
        stats["bad_dates"][c] = stats["bad_dates"].get(c, 0) + int((raw.notna() & df[c].isna()).sum())
    for c in NUM_COLS:
        raw = df[c]
        df[c] = pd.to_numeric(raw.astype("string").str.replace(",", "", regex=False), errors="coerce")
        stats["bad_numbers"][c] = stats["bad_numbers"].get(c, 0) + int((raw.notna() & df[c].isna()).sum())

    extra = set(df.columns) - set(CLEAN_COLUMNS)
    if extra:
        sys.exit(f"Unmapped columns in the cleaned file: {sorted(extra)}")
    for c in CLEAN_COLUMNS:
        if c not in df:
            df[c] = pd.NA
    return df[CLEAN_COLUMNS]


def main() -> None:
    if not SRC.exists():
        sys.exit(f"{SRC.relative_to(ROOT)} not found. Run notebooks/00_data_cleaning.ipynb first, "
                 "or copy the cleaned CSV there.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".parquet.tmp")
    stats = {"merge_conflicts": {}, "bad_dates": {}, "bad_numbers": {}}
    counts: dict[tuple[str, str], int] = {}
    n = 0
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
        for chunk in pd.read_csv(SRC, dtype=str, chunksize=CHUNK, low_memory=False):
            df = transform(chunk, stats)
            writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
            n += len(df)
            for k, v in df.groupby(["Source System", df["Bid Notice Status"].fillna("(missing)")]).size().items():
                counts[k] = counts.get(k, 0) + int(v)
            print(f"  {n:>10,} rows", flush=True)
    tmp.replace(OUT)

    print(f"\nTotal: {n:,} rows -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:,.0f} MB)")
    print("\nBid Notice Status (line items):")
    for (system, s), v in sorted(counts.items()):
        print(f"  {system}  {s:<12} {v:>10,}")
    print("\nMerged column pairs with conflicting values (target kept):")
    for c, v in stats["merge_conflicts"].items():
        print(f"  {c:<32} {v:>8,}")
    print("\nValues that could not be parsed (left empty):")
    for kind in ("bad_dates", "bad_numbers"):
        for c, v in stats[kind].items():
            if v:
                print(f"  {c:<32} {v:>8,}")


if __name__ == "__main__":
    main()
