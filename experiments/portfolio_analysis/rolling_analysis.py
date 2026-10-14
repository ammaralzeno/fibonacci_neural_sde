from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "data"
OUTPUT_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "results"

DIVERSIFIED_RETURN_FILE = DATA_DIR / "diversified_returns.csv"
CONCENTRATED_RETURN_FILE = DATA_DIR / "concentrated_returns.csv"

ROLLING_WINDOW = 60

def load_return_matrix(file_path):

    if not file_path.exists():
        raise FileNotFoundError(f"Return file not found:\n{file_path}")

    returns = pd.read_csv(file_path, parse_dates=["Date"])
    returns = returns.set_index("Date")
    returns = returns.sort_index()

    for column in returns.columns:
        returns[column] = pd.to_numeric(returns[column], errors="coerce")

    return returns


def validate_returns(portfolio_name, returns):

    print("\n" + "=" * 80)
    print(f"VALIDATING RETURNS: {portfolio_name.upper()}")
    print("=" * 80)

    print(f"Observations   : {len(returns):,}")
    print(f"Securities     : {returns.shape[1]}")
    print(f"First date     : {returns.index.min().date()}")
    print(f"Last date      : {returns.index.max().date()}")

    missing_values = returns.isna().sum().sum()
    non_finite_values = (~np.isfinite(returns.to_numpy())).sum()

    print(f"Missing values : {missing_values:,}")
    print(f"Non-finite     : {non_finite_values:,}")

    if missing_values > 0:
        raise ValueError(f"{portfolio_name}: missing return values found")

    if non_finite_values > 0:
        raise ValueError(f"{portfolio_name}: non-finite return values found")


def calculate_rolling_correlations(returns, window):

    rolling_data = {}
    securities = returns.columns.tolist()

    for asset_a, asset_b in combinations(securities, 2):
        pair_name = f"{asset_a}__{asset_b}"
        rolling_corr = returns[asset_a].rolling(window=window).corr(returns[asset_b])
        rolling_data[pair_name] = rolling_corr

    rolling_correlations = pd.DataFrame(rolling_data, index=returns.index)

    return rolling_correlations


def clean_rolling_correlations(rolling_correlations):

    cleaned = rolling_correlations.dropna(how="any")

    return cleaned


def summarize_rolling_correlations(rolling_correlations):

    summary = pd.DataFrame(
        {
            "mean": rolling_correlations.mean(),
            "std": rolling_correlations.std(),
            "min": rolling_correlations.min(),
            "max": rolling_correlations.max(),
            "latest": rolling_correlations.iloc[-1]
        }
    )
    summary.index.name = "pair"

    return summary


def print_rolling_summary(portfolio_name, rolling_correlations, summary):

    print("\n" + "=" * 80)
    print(f"ROLLING CORRELATION SUMMARY: {portfolio_name.upper()}")
    print("=" * 80)

    print(f"Rolling window         : {ROLLING_WINDOW} trading days")
    print(f"Valid rolling dates    : {len(rolling_correlations):,}")
    print(f"First rolling date     : {rolling_correlations.index.min().date()}")
    print(f"Last rolling date      : {rolling_correlations.index.max().date()}")
    print(f"Number of asset pairs  : {rolling_correlations.shape[1]}")

    print("\nPair summary:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nFirst 5 rolling-correlation rows:")
    print(rolling_correlations.head().to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nLast 5 rolling-correlation rows:")
    print(rolling_correlations.tail().to_string(float_format=lambda x: f"{x:.4f}"))


def save_results(portfolio_name, rolling_correlations, summary):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rolling_file = OUTPUT_DIR / f"{portfolio_name}_rolling_correlations.csv"
    summary_file = OUTPUT_DIR / f"{portfolio_name}_rolling_correlation_summary.csv"

    rolling_correlations.to_csv(rolling_file, index=True)
    summary.to_csv(summary_file, index=True)


    print(f"\nSaved rolling correlations:  {rolling_file}")
    print(f"\nSaved rolling summary:  {summary_file}")


def process_portfolio(portfolio_name, return_file):

    returns = load_return_matrix(return_file)
    validate_returns(portfolio_name, returns)

    rolling_correlations = calculate_rolling_correlations(returns, window=ROLLING_WINDOW)
    rolling_correlations = clean_rolling_correlations(rolling_correlations)

    summary = summarize_rolling_correlations(rolling_correlations)
    print_rolling_summary(portfolio_name, rolling_correlations, summary)
    save_results(portfolio_name, rolling_correlations, summary)


def main():

    print("=" * 80)
    print("ROLLING CORRELATION ANALYSIS")
    print("=" * 80)

    process_portfolio(portfolio_name="diversified", return_file=DIVERSIFIED_RETURN_FILE)
    process_portfolio(portfolio_name="concentrated", return_file=CONCENTRATED_RETURN_FILE)

    print("\n" + "=" * 80)
    print("ROLLING CORRELATION ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()