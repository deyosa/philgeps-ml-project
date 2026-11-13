"""Combine the raw PhilGEPS 1.5 and 2.0 quarterly files into one Parquet file.

Usage (from the repository root):

    python scripts/combine_raw.py

Output: data/processed/philgeps_combined.parquet (one row per bid line item).

What it does:
  * Reads every 1.5_*.csv file and only the 2.0_2026_*.csv files. The
    2.0_2023_* to 2.0_2025_* files are byte-for-byte copies of the 1.5 files
    and are skipped to avoid double counting.
  * Repairs 1.5_2025_Q4.csv, where the Calendar Type value is missing and every
    later value is shifted one column to the left.
  * Renames columns to the 1.5 (2023-2025) names and adds Source System,
    Source File and Source Quarter.
  * Harmonises labels that only differ in spelling (statuses, Classification,
    Region, Trade Agreement, duration units) and adds Procurement Mode (Grouped).
  * Parses dates and amounts.

Value sets that are NOT comparable between systems (Business Category,
Funding Source, Award Notice Status in 2.0) are kept as they are. Use
Source System to tell them apart.

Files are read in chunks so the script runs with a few GB of free RAM.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "philgeps_combined.parquet"
CHUNK = 100_000

# --------------------------------------------------------------------------
# Output schema (1.5 names, plus columns only some sources have)
# --------------------------------------------------------------------------
DATE_COLS = [
    "Published Date", "Closing Date", "Published Date(Award)", "Award Date",
    "Notice to Proceed Date", "Contract Effectivity Date", "Contract End Date",
]
NUM_COLS = ["Approved Budget of the Contract", "Quantity", "Item Budget", "Contract Amount", "Contract Duration"]

COLUMNS = [
    "Source System", "Source File", "Source Quarter",
    "Procuring Entity (PE)", "Region", "Province", "City/Municipality",
    "Government Branch", "PE Organization Type", "PE Organization Type (Grouped)",
    "Bid Reference No.", "Solicitation No.", "Notice Title", "Classification",
    "Procurement Mode", "Procurement Mode (Grouped)", "Business Category",
    "Funding Source", "Funding Instrument", "Trade Agreement",
    "Approved Budget of the Contract", "Published Date", "Closing Date",
    "Area of Delivery", "Contract Duration", "Calendar Type", "Lot Type",
    "Line Item No", "Item Code", "Item Name", "Item Description", "Quantity",
    "UOM", "Item Budget", "Bid Notice Status", "Award Reference No.",
    "Award Title", "UNSPSC Code", "UNSPSC Description", "Published Date(Award)",
    "Award Date", "Contract Amount", "Award Notice Status",
    "Notice to Proceed Date", "Contract Effectivity Date", "Contract End Date",
    "Awardee OrgID", "Awardee Organization Name", "Country of Awardee",
    "Region of Awardee", "Province of Awardee", "City/Municipality of Awardee",
    "Awardee Size", "Awardee Joint Venture",
]
SCHEMA = pa.schema(
    [
        (c, pa.timestamp("s") if c in DATE_COLS else pa.float64() if c in NUM_COLS else pa.string())
        for c in COLUMNS
    ]
)

# --------------------------------------------------------------------------
# Column renames per layout
# --------------------------------------------------------------------------
RENAME_15_2026 = {  # 1.5 files from 2026 Q1 (and the repaired 2025 Q4 layout is unchanged)
    "Procurement Category": "Business Category",
    "Source of Funds": "Funding Source",
    "Applicable Procurement Rules": "Trade Agreement",
    "Area of Delivery/Project Location": "Area of Delivery",
    "Delivery Period": "Contract Duration",
}
RENAME_20 = {
    "Procurement Category": "Business Category",
    "Source of Funds": "Funding Source",
    "Applicable Procurement Rules": "Trade Agreement",
    "Approved Budget for the Contract (ABC)": "Approved Budget of the Contract",
    "Area of Delivery/Project Location": "Area of Delivery",
    "Delivery Period/Contract Duration": "Contract Duration",
    "Item/Lot Name": "Item Name",
    "Item/Lot Description": "Item Description",
    "Unit of Measure": "UOM",
    "Item/Lot Name.1": "UNSPSC Description",  # second "Item/Lot Name" column, filled with UNSPSC Code
    "Published Date (Award)": "Published Date(Award)",
    "Award Notice Status (Item)": "Award Notice Status",
    "Awardee Size (MSME Category)": "Awardee Size",
}

# --------------------------------------------------------------------------
# Value harmonisation
# --------------------------------------------------------------------------
STATUS_20 = {"Published": "Active", "CANCEL_APPROVED": "Cancelled", "CONCLUDED": "Concluded"}
CLASSIFICATION_20 = {"Civil Works - Infra Project": "Civil Works"}
TRADE_AGREEMENT_20 = {
    "World Bank Procurement Guidelines": "WB Procurement Guidelines",
    "International Executive Agreement": "International / Executive Agreement",
    "Other": "Others",
}
REGIONS_15 = [
    "BARMM", "CAR", "Caraga", "NCR", "Negros Island Region", "Region I", "Region II",
    "Region III", "Region IV-A", "Region IV-B", "Region IX", "Region V", "Region VI",
    "Region VII", "Region VIII", "Region X", "Region XI", "Region XII",
]
REGION_MAP = {r.upper(): r for r in REGIONS_15} | {"ARMM": "BARMM"}
UNIT_MAP = {"day": "Day/s", "days": "Day/s", "day/s": "Day/s",
            "month": "Month/s", "months": "Month/s", "month/s": "Month/s",
            "year": "Year/s", "years": "Year/s", "year/s": "Year/s"}

# Procurement Mode (Grouped): drop the "(Sec. ...)" suffix, then map 2.0 names to 1.5 names.
MODE_GROUP = {
    "Competitive Bidding (Public Bidding)": "Public Bidding",
    "Negotiated Procurement - Small Value Procurement": "Small Value Procurement",
    "Negotiated Procurement - Lease of Real Property": "Negotiated Procurement - Lease of Real Property and Venue",
}
SEC_SUFFIX = re.compile(r"\s*\((?:Sec\.\s*)?[\d.a-z]+\)$")


def group_mode(s: pd.Series) -> pd.Series:
    out = s.str.replace(SEC_SUFFIX, "", regex=True).str.strip()
    return out.replace(MODE_GROUP)


def split_duration(df: pd.DataFrame) -> None:
    """Turn '30 Day/s' or '1 years' into Contract Duration=30/1 and Calendar Type."""
    m = df["Contract Duration"].astype("string").str.extract(r"^\s*([\d.,]+)\s*(.*?)\s*$")
    df["Contract Duration"] = m[0].str.replace(",", "", regex=False)
    unit = m[1].str.lower().map(UNIT_MAP)
    df["Calendar Type"] = unit.where(unit.notna(), m[1].replace("", pd.NA))


def parse_dates(s: pd.Series, formats: list[str]) -> pd.Series:
    s = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[s]")
    for fmt in formats:
        todo = out.isna() & s.notna()
        if not todo.any():
            break
        out[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    return out


# --------------------------------------------------------------------------
# Source files
# --------------------------------------------------------------------------
def sources() -> list[dict]:
    out = []
    for f in sorted(RAW.glob("1.5_*.csv")):
        year = int(f.stem.split("_")[1])
        if f.name == "1.5_2025_Q4.csv":
            layout = dict(sep="\t", encoding="cp1252", fix_shift=True, rename=RENAME_15_2026)
        elif year >= 2026:
            layout = dict(sep=",", encoding="cp1252", rename=RENAME_15_2026)
        else:
            layout = dict(sep=",", encoding="utf-8-sig", rename={})
        out.append(dict(path=f, system="1.5", date_formats=["%d/%m/%Y"], **layout))
    for f in sorted(RAW.glob("2.0_*.csv")):
        if int(f.stem.split("_")[1]) < 2026:
            continue  # duplicates of the 1.5 files
        out.append(dict(path=f, system="2.0", sep=",", encoding="utf-8", rename=RENAME_20,
                        date_formats=["%d-%b-%Y", "%d-%b-%Y %I:%M %p"]))
    return out


def read_chunks(src: dict):
    kw = dict(sep=src["sep"], encoding=src["encoding"], encoding_errors="replace",
              dtype=str, na_values=["NULL"], chunksize=CHUNK, low_memory=False)
    if src.get("fix_shift"):
        header = pd.read_csv(src["path"], nrows=0, **{k: v for k, v in kw.items() if k != "chunksize"}).columns
        i = list(header).index("Calendar Type")
        names = list(header[:i]) + list(header[i + 1:]) + ["__trailing"]
        kw |= dict(header=0, names=names)
    yield from pd.read_csv(src["path"], **kw)


def transform(df: pd.DataFrame, src: dict, stats: dict) -> pd.DataFrame:
    if "__trailing" in df:
        if df["__trailing"].notna().any():
            sys.exit(f"{src['path'].name}: shifted-column repair found data in the trailing column; stopping.")
        df = df.drop(columns="__trailing")
    df = df.rename(columns=src["rename"])

    if "Calendar Type" not in df:  # duration and unit are combined in one field
        split_duration(df)

    if src["system"] == "2.0":
        df["Bid Notice Status"] = df["Bid Notice Status"].replace(STATUS_20)
        df["Classification"] = df["Classification"].replace(CLASSIFICATION_20)
        df["Trade Agreement"] = df["Trade Agreement"].replace(TRADE_AGREEMENT_20)
        df["Awardee Joint Venture"] = df["Awardee Joint Venture"].replace({"No JV": pd.NA})

    df["Region"] = df["Region"].map(lambda r: REGION_MAP.get(r.strip().upper(), r.strip()) if isinstance(r, str) else r)
    df["Procurement Mode (Grouped)"] = group_mode(df["Procurement Mode"].astype("string"))

    for c in DATE_COLS:
        raw = df[c]
        df[c] = parse_dates(raw, src["date_formats"])
        stats["bad_dates"][c] = stats["bad_dates"].get(c, 0) + int((raw.notna() & df[c].isna()).sum())
    for c in NUM_COLS:
        raw = df[c]
        df[c] = pd.to_numeric(raw.astype("string").str.replace(",", "", regex=False), errors="coerce")
        stats["bad_numbers"][c] = stats["bad_numbers"].get(c, 0) + int((raw.notna() & df[c].isna()).sum())

    year, q = src["path"].stem.split("_")[1:]
    df["Source System"] = src["system"]
    df["Source File"] = src["path"].name
    df["Source Quarter"] = f"{year}{q}"

    for c in COLUMNS:
        if c not in df:
            df[c] = pd.NA
    extra = set(df.columns) - set(COLUMNS)
    if extra:
        sys.exit(f"{src['path'].name}: unmapped columns {sorted(extra)}")
    return df[COLUMNS]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".parquet.tmp")
    stats = {"bad_dates": {}, "bad_numbers": {}}
    per_file, status = [], {}
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
        for src in sources():
            n = 0
            for chunk in read_chunks(src):
                df = transform(chunk, src, stats)
                writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
                n += len(df)
                for k, v in df["Bid Notice Status"].fillna("(missing)").value_counts().items():
                    status[(src["system"], k)] = status.get((src["system"], k), 0) + int(v)
            per_file.append((src["path"].name, n))
            print(f"{src['path'].name:<18} {n:>10,} rows", flush=True)
    tmp.replace(OUT)

    total = sum(n for _, n in per_file)
    print(f"\nTotal: {total:,} rows -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:,.0f} MB)")
    print("\nBid Notice Status (line items):")
    for (system, s), v in sorted(status.items()):
        print(f"  {system}  {s:<12} {v:>10,}")
    print("\nValues that could not be parsed (left empty):")
    for kind in ("bad_dates", "bad_numbers"):
        for c, v in stats[kind].items():
            if v:
                print(f"  {c:<32} {v:>8,}")


if __name__ == "__main__":
    main()
