from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Define Paths 

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amgm import config as amgm_config
from amgm.data.loading import load_sebx_am_data

DATA_DIR = Path(amgm_config.am_dataset_dir)

OUTPUT_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "data"

# Define Portfolios

SECTOR_NAMES = {
    10: "Energy",
    35: "Health Care",
    40: "Financials",
    45: "Information Technology"
}

DIVERSIFIED_PORTFOLIO = {
    "00138001": "Energy",
    "00107801": "Health Care",
    "00141402": "Financials",
    "00116101": "Information Technology"
}

CONCENTRATED_PORTFOLIO = {
    "00116101": "Information Technology",
    "00169001": "Information Technology",
    "00177301": "Information Technology",
    "00187801": "Information Technology"
}


PORTFOLIOS = {
    "diversified": DIVERSIFIED_PORTFOLIO,
    "concentrated": CONCENTRATED_PORTFOLIO
}


def load_data():

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    print(f"Dataset directory: {DATA_DIR}")

    data = load_sebx_am_data(DATA_DIR, cached=True, force=False)
    required_tables = {
        "security_data",
        "constituents"
    }

    missing_tables = required_tables - set(data.keys())

    if missing_tables:
        raise KeyError(
            f"Missing required table(s): {sorted(missing_tables)}"
        )

    print("Dataset loaded successfully")

    return data


def prepare_tables(data):

    security_data = data["security_data"].copy()
    constituents = data["constituents"].copy()

    security_data["IssueId"] = security_data["IssueId"].astype(str)
    constituents["IssueId"] = constituents["IssueId"].astype(str)

    security_data["Date"] = pd.to_datetime(security_data["Date"], errors="coerce")
    security_data["ClAdjLoc"] = pd.to_numeric(security_data["ClAdjLoc"], errors="coerce")

    return security_data, constituents


def validate_portfolio(portfolio_name, portfolio, security_data, constituents):

    print("\n" + "=" * 80)
    print(f"VALIDATING PORTFOLIO: {portfolio_name.upper()}")
    print("=" * 80)

    rows = []
    for issue_id, intended_sector_name in portfolio.items():
        prices = security_data[security_data["IssueId"] == issue_id].copy()

        if prices.empty:
            raise ValueError(
                f"{issue_id} has no historical price data"
            )


        metadata = constituents[constituents["IssueId"] == issue_id].copy()

        if metadata.empty:
            raise ValueError(
                f"{issue_id} has no metadata in constituents."
            )

        metadata_row = metadata.iloc[0]
        country = metadata_row["Country"]
        sector_code = int(metadata_row["SectorCode"])
        industry_code = int(metadata_row["IndustryCode"])
        subindustry_code = int(metadata_row["SubIndustryCode"])
        actual_sector_name = SECTOR_NAMES.get(sector_code)

        if country != "US":
            raise ValueError(
                f"{issue_id} is not a US security, country={country}"
            )

        if actual_sector_name != intended_sector_name:
            raise ValueError(
                f"{issue_id}: expected sector : {intended_sector_name}, metadata shows : {actual_sector_name}"
            )

        missing_prices = prices["ClAdjLoc"].isna().sum()
        duplicate_dates = prices["Date"].duplicated().sum()
        first_date = prices["Date"].min()
        last_date = prices["Date"].max()
        valid_prices = prices["ClAdjLoc"].notna().sum()

        rows.append({
            "portfolio": portfolio_name,
            "IssueId": issue_id,
            "Country": country,
            "SectorCode": sector_code,
            "Sector": actual_sector_name,
            "IndustryCode": industry_code,
            "SubIndustryCode": subindustry_code,
            "first_date": first_date,
            "last_date": last_date,
            "valid_prices": valid_prices,
            "missing_prices": missing_prices,
            "duplicate_dates": duplicate_dates
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    return summary


def build_price_matrix(portfolio, security_data):

    issue_ids = list(portfolio.keys())

    selected = security_data[security_data["IssueId"].isin(issue_ids)][["IssueId", "Date", "ClAdjLoc"]].copy()
    selected = selected.dropna(subset=["Date", "ClAdjLoc"])

    price_matrix = selected.pivot(index="Date", columns="IssueId", values="ClAdjLoc")
    price_matrix = price_matrix[issue_ids]
    price_matrix = price_matrix.dropna(how="any")
    price_matrix = price_matrix.sort_index()

    return price_matrix

def print_price_matrix_summary(portfolio_name, price_matrix):

    print("\n" + "=" * 80)
    print(f"ALIGNED PRICES: {portfolio_name.upper()}")
    print("=" * 80)

    print(f"Number of securities : {price_matrix.shape[1]}")
    print(f"Common observations  : {price_matrix.shape[0]:,}")
    print(f"First common date    : {price_matrix.index.min().date()}")
    print(f"Last common date     : {price_matrix.index.max().date()}")

    print("\nFirst 5 rows:")
    print(price_matrix.head())

    print("\nLast 5 rows:")
    print(price_matrix.tail())

def save_outputs(portfolio_summaries, price_matrices):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    portfolio_metadata = pd.concat(portfolio_summaries, ignore_index=True)
    metadata_file = OUTPUT_DIR / "portfolio_definitions.csv"
    portfolio_metadata.to_csv(metadata_file, index=False)

    print(f"\nSaved portfolio definitions:  {metadata_file}")

    for portfolio_name, price_matrix in price_matrices.items():
        output_file = OUTPUT_DIR / f"{portfolio_name}_prices.csv"
        price_matrix.to_csv(output_file, index=True)

        print(f"\nSaved {portfolio_name} prices:  {output_file}")

def main():

    data = load_data()
    security_data, constituents = prepare_tables(data)

    portfolio_summaries = []
    price_matrices = {}

    for portfolio_name, portfolio in PORTFOLIOS.items():
        summary = validate_portfolio(portfolio_name, portfolio, security_data, constituents)
        portfolio_summaries.append(summary)

        price_matrix = build_price_matrix(portfolio, security_data)
        price_matrices[portfolio_name] = price_matrix
        print_price_matrix_summary(portfolio_name, price_matrix)

    save_outputs(portfolio_summaries, price_matrices)

    print("\n" + "=" * 80)
    print("PORTFOLIO DATA PREPARATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()