from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amgm import config as amgm_config
from amgm.data.loading import load_sebx_am_data

DATA_DIR = Path(amgm_config.am_dataset_dir)
OUTPUT_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "phase1"

SECURITY_TABLE_NAME = "security_data"
REQUIRED_SECURITY_COLUMNS = {"IssueId", "Date", "ClAdjLoc"}

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_subheader(title):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

def safe_string(value):
    if pd.isna(value): return "<NA>"
    return str(value)

def load_dataset():
    print_header("1. DATASET LOCATION")
    print(f"Repository root : {REPO_ROOT}")
    print(f"Dataset path    : {DATA_DIR}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            "\nDataset directory does not exist.\n"
            f"Configured path: {DATA_DIR}\n\n"
            "Chech amgm/config.py and make sure am_dataset_dir points to your Data directory"
        )

    print("\nLoading dataset...")

    data = load_sebx_am_data(DATA_DIR, cached=True, force=False)
    if not data:
        raise RuntimeError(
            f"No tables were loaded from {DATA_DIR}"
        )

    print(f"Loaded {len(data)} table(s).")

    return data

def inspect_tables(data):
    print_header("2. RAW TABLES")

    for table_name in sorted(data.keys()):
        df = data[table_name]
        print_subheader(f"TABLE: {table_name}")

        print(f"Rows    : {len(df):,}")
        print(f"Columns : {len(df.columns):,}")

        print("\nColumn names:")
        for column in df.columns: print(f"  - {column}")

        print("\nFirst 3 rows:")
        if len(df) == 0: print("<EMPTY TABLE>")
        else: print(df.head(3).to_string(index=False))

def prepare_security_data(data):
    print_header("3. SECURITY DATA VALIDATION")
    if SECURITY_TABLE_NAME not in data:
        raise KeyError(
            f"Expected table '{SECURITY_TABLE_NAME}' was not found.\n"
            f"Available tables: {sorted(data.keys())}"
        )

    securities = data[SECURITY_TABLE_NAME].copy()
    missing_columns = (REQUIRED_SECURITY_COLUMNS - set(securities.columns))

    if missing_columns:
        raise KeyError(
            f"security_data is missing required column(s): {sorted(missing_columns)}"
        )

    securities["IssueId"] = securities["IssueId"].astype(str)
    securities["Date"] = pd.to_datetime(securities["Date"], errors="coerce")
    securities["ClAdjLoc"] = pd.to_numeric(securities["ClAdjLoc"], errors="coerce")

    print(f"Rows                   : {len(securities):,}")
    print(f"Unique IssueIds        : {securities['IssueId'].nunique():,}")

    valid_dates = securities["Date"].dropna()

    if len(valid_dates) > 0:
        print(f"Earliest date          : {valid_dates.min().date()}")
        print(f"Latest date            : {valid_dates.max().date()}")

    print(f"Missing dates          : {securities['Date'].isna().sum():,}")
    print(f"Missing ClAdjLoc       : {securities['ClAdjLoc'].isna().sum():,}")

    non_positive = (securities["ClAdjLoc"].notna() & (securities["ClAdjLoc"] <= 0)).sum()
    print(f"Non-positive prices    : {non_positive:,}")

    duplicate_rows = securities.duplicated(subset=["IssueId", "Date"]).sum()
    print(f"Duplicate ID/date rows : {duplicate_rows:,}")

    return securities

def calculate_issue_coverage(securities):
    print_header("4. HISTORY AVAILABLE FOR EACH ISSUE ID")

    grouped = securities.groupby("IssueId", sort=True, observed=True)
    coverage = grouped.agg(
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        total_rows=("Date", "size"),
        valid_dates=("Date", "count"),
        valid_prices=("ClAdjLoc", "count"),
        missing_prices=("ClAdjLoc", lambda x: x.isna().sum())
    ).reset_index()

    coverage["history_calendar_days"] = (coverage["last_date"] - coverage["first_date"]).dt.days
    coverage["history_years_approx"] = coverage["history_calendar_days"] / 365.25
    coverage["price_coverage_pct"] = 100.0 * coverage["valid_prices"] / coverage["total_rows"].replace(0, np.nan)

    coverage = coverage.sort_values(by=["valid_prices", "history_calendar_days"], ascending = [False, False]).reset_index(drop=True)

    print("\nTop 20 IssueIds by number of valid price observations:\n")

    columns_to_print = [
        "IssueId",
        "first_date",
        "last_date",
        "valid_prices",
        "missing_prices",
        "history_years_approx",
        "price_coverage_pct"
    ]

    print(coverage[columns_to_print].head(20).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return coverage


def find_issue_id_metadata_tables(data):
    print("5. ISSUE ID METADATA TABLES")

    candidates = {}
    for table_name, df in data.items():
        if table_name == SECURITY_TABLE_NAME: continue
        if "IssueId" not in df.columns: continue

        candidate = df.copy()
        candidate["IssueId"] = candidate["IssueId"].astype(str)
        candidates[table_name] = candidate

        print_subheader(f"Metadata candidate: {table_name}")
        print(f"Rows                        : {len(candidate):,}")
        print(f"Unique IssueIds             : {candidate['IssueId'].nunique():,}")
        print("\nColumns:")

        for column in candidate.columns:
            print(f"  - {column}")

        print("\nSample rows:")
        print(candidate.head(5).to_string(index=False))

    if not candidates:
        print("\nNo table other than security_data conatins an 'IssueId' column")

    return candidates


def find_interesting_metadata_columns(df):

    keywords = [
        "ticker",
        "symbol",
        "name",
        "company",
        "security",
        "sector",
        "industry",
        "country",
        "exchange",
        "isin",
        "sedol",
        "cusip",
        "issue"
    ]

    selected = []

    for column in df.columns:
        column_lower = str(column).lower()
        if column == "IssueId" or any(keyword in column_lower for keyword in keywords):
            selected.append(column)

    return selected

def inspect_metadata_columns(metadata_tables):

    print_header("6. POSSIBLE SECURITY IDENTIFIERS")

    if not metadata_tables:
        print("No IssueId metadata tables available")
        return

    for table_name, df in metadata_tables.items():

        interesting_columns = find_interesting_metadata_columns(df)
        print_subheader(table_name)

        if not interesting_columns:
            print("No obviously descriptive metadata columns found")
            continue

        print("Potenially useful columns:\n")

        for column in interesting_columns:
            print(f"  - {column}")

        preview_columns = interesting_columns[:12]
        print("\nPreview\n")
        print(df[preview_columns].drop_duplicates().head(10).to_string(index=False))


def calculate_common_date_statistics(securities, coverage, number_of_candidates):
    
    print_header("7. COMMON HISTORY CHECK")

    candidate_issue_ids = coverage["IssueId"].head(number_of_candidates).tolist()
    candidate_data = securities[securities["IssueId"].isin(candidate_issue_ids)].copy()
    price_matrix = candidate_data.pivot_table(index="Date", columns="IssueId", values="ClAdjLoc", aggfunc="last")
    completely_overlapping = price_matrix.dropna()

    print(f"Candidate securities checked.  : {len(candidate_issue_ids)}")
    print(f"Total distinct dates           : {len(price_matrix):,}")
    print(f"Dates common to ALL candidates : {len(completely_overlapping):,}")

    if not completely_overlapping.empty:
        print(f"Common-history start           : {completely_overlapping.index.min().date()}")
        print(f"Common-history end             : {completely_overlapping.index.max().date()}")

    print("\nThese are the longest-history IssueIds, not our final portfolio:")
    for issue_id in candidate_issue_ids:
        print(f"  - {issue_id}")

def save_outputs(data, coverage, metadata_tables):
    print_header("8. SAVING PHASE 1 OUTPUTS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage_file = OUTPUT_DIR/ "issue_coverage.csv"
    coverage.to_csv(coverage_file, index=False)
    print(f"Saved: {coverage_file}")

    for table_name, df in metadata_tables.items():

        columns = find_interesting_metadata_columns(df)

        if not columns: 
            columns = list(df.columns)

        if "IssueId" in df.columns and "IssueId" not in columns: 
            columns = ["IssueId", *columns]

        output_file = OUTPUT_DIR / f"{table_name}_metadata_preview.csv"
        df[columns].drop_duplicates().to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

    inventory_rows = []
    for table_name, df in data.items():
        inventory_rows.append(
            {
                "table_name": table_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": " | ".join(map(str, df.columns))
            }
        )

    inventory = pd.DataFrame(inventory_rows)
    inventory_file = OUTPUT_DIR / "table_inventory.csv"
    inventory.to_csv(inventory_file, index=False)

    print(f"Saved: {inventory_file}")


def print_final_summary(coverage):
    print_header("PHASE 1 COMPLETE")
    print(f"Securities inspected : {len(coverage):,}")
    print("\nNext step is to inspect metadata so we can choose 3-5 securities")

def main():

    data = load_dataset()
    inspect_tables(data)
    securities = prepare_security_data(data)
    coverage = calculate_issue_coverage(securities)
    metadata_tables = find_issue_id_metadata_tables(data)
    inspect_metadata_columns(metadata_tables)
    calculate_common_date_statistics(securities, coverage, number_of_candidates=10)
    save_outputs(data, coverage, metadata_tables)
    print_final_summary(coverage)


if __name__ == "__main__":
    main()