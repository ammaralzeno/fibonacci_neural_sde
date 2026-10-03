from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "data"
OUTPUT_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "results"

DIVERSIFIED_RETURN_FILE = DATA_DIR / "diversified_returns.csv"
CONCENTRATED_RETURN_FILE = DATA_DIR / "concentrated_returns.csv"

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


def calculate_correlation(returns):

    correlation = returns.corr(method="pearson")

    return correlation


def calculate_covariance(returns):
    covariance = returns.cov()
    return covariance


def print_correlation_matrix(portfolio_name, correlation):

    print("\n" + "=" * 80)
    print(f"CORRELATION MATRIX: {portfolio_name.upper()}")
    print("=" * 80)

    print(correlation.to_string(float_format=lambda x: f"{x:.4f}"))


def print_covariance_matrix(portfolio_name, covariance):

    print("\n" + "=" * 80)
    print(f"COVARIANCE MATRIX: {portfolio_name.upper()}")
    print("=" * 80)

    print(covariance.to_string(float_format=lambda x: f"{x:.8f}"))


def summarize_pairwise_correlations(portfolio_name, correlation):

    rows = []
    columns = correlation.columns
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            asset_a = columns[i]
            asset_b = columns[j]

            rows.append(
                {
                    "portfolio": portfolio_name,
                    "asset_1": asset_a,
                    "asset_2": asset_b,
                    "correlation": correlation.loc[asset_a, asset_b]
                }
            )

    pairwise = pd.DataFrame(rows)
    pairwise = pairwise.sort_values(by="correlation", ascending=False).reset_index(drop=True)
    print("\nPairwise correlations:")
    print(pairwise.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return pairwise

def save_results(portfolio_name, correlation, covariance, pairwise):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    correlation_file = OUTPUT_DIR / f"{portfolio_name}_correlation.csv"
    covariance_file = OUTPUT_DIR / f"{portfolio_name}_covariance.csv"
    pairwise_file = OUTPUT_DIR / f"{portfolio_name}_pairwise_correlations.csv"

    correlation.to_csv(correlation_file)
    covariance.to_csv(covariance_file)
    pairwise.to_csv(pairwise_file, index=False)

    print(f"\nSaved correlation matrix:  {correlation_file}")
    print(f"\nSaved covariance matrix:  {covariance_file}")
    print(f"\nSaved pairwise correlations:  {pairwise_file}")

def process_portfolio(portfolio_name, return_file):

    returns = load_return_matrix(return_file)
    validate_returns(portfolio_name, returns)

    correlation = calculate_correlation(returns)
    covariance = calculate_covariance(returns)
    print_correlation_matrix(portfolio_name, correlation)
    print_covariance_matrix(portfolio_name, covariance)

    pairwise = summarize_pairwise_correlations(portfolio_name, correlation)
    save_results(portfolio_name, correlation, covariance, pairwise)


def main():

    print("=" * 80)
    print("STATIC PORTFOLIO RELATIONSHIP ANALYSIS")
    print("=" * 80)

    process_portfolio(portfolio_name="diversified", return_file=DIVERSIFIED_RETURN_FILE)
    process_portfolio(portfolio_name="concentrated", return_file=CONCENTRATED_RETURN_FILE)

    print("\n" + "=" * 80)
    print("STATIC ANALYSIS COMPLETE")
    print("=" * 80)



if __name__ == "__main__":
    main()

