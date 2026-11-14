"""Shared definitions for the procurement-failure notebooks (01, 02, 03).

Keeping the outcome rule, the bid-level feature build, the feature sets and the
ranking metrics in one place means every notebook uses the same definitions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def find_root() -> Path:
    """Repository root: the first parent folder that contains data/raw."""
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "raw").is_dir():
            return p
    raise FileNotFoundError("Could not find the repository root (a folder containing data/raw).")


ROOT = find_root()
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
COMBINED = PROCESSED / "philgeps_combined.parquet"
BIDS = PROCESSED / "bids_public_bidding.parquet"
EXCEPTIONS = PROCESSED / "bids_public_bidding_exceptions.csv"
MODELS = PROCESSED / "models"
SPLITS = PROCESSED / "split_assignment.parquet"

RANDOM_STATE = 42
KEY = ["Source System", "Bid Reference No."]

# Study window: 2023 Q1 to 2026 Q2 (Source Quarter of the raw file). 2026 Q3 is left out.
STUDY_QUARTERS = [f"{y}Q{q}" for y in range(2023, 2027) for q in range(1, 5) if (y, q) <= (2026, 2)]

# --------------------------------------------------------------------------
# Outcome rule
# --------------------------------------------------------------------------
STATUS_ORDER = ["Awarded", "Failed", "Cancelled", "Active", "Shortlisted",
                "Concluded", "Conflicting", "(missing)"]
# Bids with these statuses are removed from the bid table altogether.
REMOVED_STATUSES = ["Closed"]


def bid_status(items: pd.DataFrame) -> pd.Series:
    """One status per bid.

    A bid takes a status only when every one of its line items has that status.
    Bids whose line items disagree are 'Conflicting'.
    """
    counts = pd.crosstab([items[k] for k in KEY], items["Bid Notice Status"].astype("string").fillna("(missing)"))
    n_statuses = (counts > 0).sum(axis=1)
    label = pd.Series("Other", index=counts.index, dtype="object")
    for status in counts.columns:
        label[(n_statuses == 1) & (counts[status] > 0)] = status
    label[n_statuses > 1] = "Conflicting"
    return label.rename("status")


def target_failed(status: pd.Series) -> pd.Series:
    """Main target (procurement failure): 1 = Failed or Cancelled, 0 = Awarded, NaN = unlabelled."""
    return status.map({"Failed": 1.0, "Cancelled": 1.0, "Awarded": 0.0})


def target_failed_only(status: pd.Series) -> pd.Series:
    """Sensitivity target: 1 = Failed, 0 = Awarded, NaN = everything else (Cancelled included)."""
    return status.map({"Failed": 1.0, "Awarded": 0.0})


# --------------------------------------------------------------------------
# Bid-level table
# --------------------------------------------------------------------------
ITEM_COLUMNS = [
    "Source System", "Source Quarter", "Bid Reference No.", "Bid Notice Status",
    "Approved Budget of the Contract", "Published Date", "Closing Date",
    "Contract Duration", "Calendar Type", "Business Category", "Funding Source",
    "Funding Instrument", "Trade Agreement", "Region", "Classification",
    "PE Organization Type (Grouped)", "Government Branch", "Lot Type",
]
CATEGORY_FIELDS = ["Business Category", "Funding Source", "Funding Instrument", "Trade Agreement",
                   "Region", "Classification", "PE Organization Type (Grouped)", "Government Branch",
                   "Lot Type"]
# Fields whose disagreement inside one bid makes the bid unusable (as in LT8 §5.6–5.7).
STRICT_FIELDS = ["Business Category", "Region", "Classification"]
DAYS_PER_UNIT = {"Day/s": 1, "Month/s": 30, "Year/s": 365}


STUDY_FILTER = [("Source Quarter", "in", STUDY_QUARTERS)]


def load_public_bidding_items(columns: list[str] | None = None) -> pd.DataFrame:
    """Public Bidding line items from the study window (filters applied while reading)."""
    columns = columns or ITEM_COLUMNS
    table = pq.read_table(COMBINED, columns=columns,
                          filters=[("Procurement Mode (Grouped)", "=", "Public Bidding")] + STUDY_FILTER)
    text = [c for c in columns if str(table.schema.field(c).type) == "string" and c != "Bid Reference No."]
    return table.to_pandas(categories=text)


def _mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else np.nan


def build_bid_table(items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate line items to one row per bid.

    Returns (bids, exceptions). `bids` holds every bid that passed the quality rules,
    labelled or not, except bids with a status in REMOVED_STATUSES (Closed), which are
    dropped before the quality rules. `exceptions` lists excluded bids with the reason.
    """
    items = items.dropna(subset=["Bid Reference No."]).copy()
    duration_days = items["Contract Duration"] * items["Calendar Type"].astype("string").map(DAYS_PER_UNIT).astype(float)
    items["duration_days"] = duration_days

    g = items.groupby(KEY, observed=True, sort=False)
    bids = g.agg(
        source_quarter=("Source Quarter", "first"),
        budget_min=("Approved Budget of the Contract", "min"),
        budget_max=("Approved Budget of the Contract", "max"),
        published_min=("Published Date", "min"),
        published_max=("Published Date", "max"),
        closing=("Closing Date", "max"),
        n_line_items=("Bid Notice Status", "size"),
        contract_duration_days=("duration_days", "max"),
    )
    bids = bids.join(bid_status(items))

    # Category fields: value per bid and number of distinct values (version count).
    for c in CATEGORY_FIELDS:
        nunique = g[c].nunique()
        bids[c + " (versions)"] = nunique
        first = g[c].first().astype("object")
        multi = nunique > 1
        if multi.any():
            # Bids with several values: use the most frequent one (flagged via the version count).
            keys = multi[multi].index.to_frame(index=False)
            sub = items[KEY + [c]].merge(keys, on=KEY)
            sub[c] = sub[c].astype("object")
            modes = sub.groupby(KEY, observed=True)[c].agg(_mode)
            first.loc[modes.index] = modes
        bids[c] = first

    bids = bids.reset_index()
    bids = bids[~bids["status"].isin(REMOVED_STATUSES)].reset_index(drop=True)

    # Quality rules -> exceptions log.
    reasons = pd.Series("", index=bids.index)
    def flag(mask, text):
        reasons[mask] = reasons[mask].where(reasons[mask] == "", reasons[mask] + "; ") + text
    flag(bids["budget_max"].isna(), "missing approved budget")
    flag(bids["budget_max"] <= 0, "non-positive approved budget")
    flag(bids["budget_max"].notna() & (bids["budget_min"] != bids["budget_max"]), "more than one approved budget")
    flag(bids["published_min"].isna(), "missing published date")
    flag(bids["published_max"].notna() & (bids["published_min"] != bids["published_max"]), "more than one published date")
    for c in STRICT_FIELDS:
        flag(bids[c + " (versions)"] > 1, f"more than one {c}")
    excluded = reasons != ""
    exceptions = bids.loc[excluded, KEY + ["status", "source_quarter"]].assign(reason=reasons[excluded])
    bids = bids.loc[~excluded].copy()

    # Features available at publication.
    bids["approved_budget"] = bids["budget_max"]
    bids["log_approved_budget"] = np.log1p(bids["approved_budget"])
    bids["published_date"] = bids["published_min"]
    bids["published_year"] = bids["published_date"].dt.year
    bids["published_month"] = bids["published_date"].dt.month
    bids["published_dayofweek"] = bids["published_date"].dt.dayofweek
    window = (bids["closing"].dt.normalize() - bids["published_date"]).dt.days
    bids["bidding_window_days"] = window.where((window >= 0) & (window <= 365))
    bids["log_n_line_items"] = np.log1p(bids["n_line_items"])
    dur = bids["contract_duration_days"]
    bids["log_contract_duration_days"] = np.log1p(dur.where(dur <= 3650))
    bids["multi_funding_source"] = (bids["Funding Source (versions)"] > 1).astype(int)

    bids["Target_Failed"] = target_failed(bids["status"])
    bids["Target_Failed_Only"] = target_failed_only(bids["status"])

    keep = (KEY + ["source_quarter", "status", "Target_Failed", "Target_Failed_Only",
                   "approved_budget", "log_approved_budget", "published_date", "published_year",
                   "published_month", "published_dayofweek", "closing", "bidding_window_days",
                   "n_line_items", "log_n_line_items", "contract_duration_days",
                   "log_contract_duration_days", "multi_funding_source"] + CATEGORY_FIELDS)
    return bids[keep].reset_index(drop=True), exceptions.reset_index(drop=True)


# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------
# LT8 §6.1: log approved budget, funding source, procurement category, region, publication month.
ORIGINAL_NUMERIC = ["log_approved_budget"]
ORIGINAL_CATEGORICAL = ["Funding Source", "Business Category", "Region", "published_month"]

EXPANDED_NUMERIC = ORIGINAL_NUMERIC + ["bidding_window_days", "log_n_line_items",
                                       "log_contract_duration_days", "multi_funding_source"]
EXPANDED_CATEGORICAL = ORIGINAL_CATEGORICAL + [
    "Classification", "PE Organization Type (Grouped)", "Government Branch", "Funding Instrument",
    "Trade Agreement", "Lot Type", "published_dayofweek", "Source System"]

FEATURE_SETS = {
    "original_5": (ORIGINAL_NUMERIC, ORIGINAL_CATEGORICAL),
    "expanded": (EXPANDED_NUMERIC, EXPANDED_CATEGORICAL),
}


def model_frame(bids: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Feature columns with categoricals as plain strings ('Missing' for gaps)."""
    X = bids[numeric + categorical].copy()
    for c in categorical:
        X[c] = X[c].astype("string").fillna("Missing").astype(str)
    return X


def one_hot_preprocessor(numeric: list[str], categorical: list[str], min_frequency: int = 50) -> ColumnTransformer:
    """For logistic regression, decision tree and random forest."""
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                              sparse_output=True), categorical),
    ])


def ordinal_preprocessor(numeric: list[str], categorical: list[str], min_frequency: int = 50) -> ColumnTransformer:
    """For histogram gradient boosting with native categorical support."""
    return ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1, min_frequency=min_frequency,
                               max_categories=250), categorical),
    ], verbose_feature_names_out=False)


# --------------------------------------------------------------------------
# Ranking metrics for a review queue (LT8 §6.2)
# --------------------------------------------------------------------------
REVIEW_SHARES = (0.05, 0.10, 0.20, 0.30, 0.50)


def queue_metrics(y_true, score, shares=REVIEW_SHARES) -> pd.DataFrame:
    """Precision, recall and lift when reviewing the top share of notices by predicted risk."""
    y = np.asarray(y_true, dtype=float)
    order = np.argsort(-np.asarray(score), kind="stable")
    base = y.mean()
    rows = []
    for s in shares:
        k = max(1, int(round(len(y) * s)))
        hits = y[order[:k]].sum()
        precision = hits / k
        rows.append({"Review share": f"top {int(s * 100)}%", "Notices reviewed": k,
                     "Precision@k": precision, "Recall@k": hits / y.sum() if y.sum() else np.nan,
                     "Lift@k": precision / base if base else np.nan})
    return pd.DataFrame(rows)


def summary_metrics(y_true, score) -> dict:
    y = np.asarray(y_true, dtype=float)
    q = queue_metrics(y, score, shares=(0.10,)).iloc[0]
    return {
        "n": len(y), "failure rate": y.mean(),
        "ROC-AUC": roc_auc_score(y, score) if 0 < y.mean() < 1 else np.nan,
        "PR-AUC": average_precision_score(y, score) if 0 < y.mean() < 1 else np.nan,
        "Precision@10%": q["Precision@k"], "Recall@10%": q["Recall@k"], "Lift@10%": q["Lift@k"],
    }
