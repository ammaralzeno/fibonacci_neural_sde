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

MIN_LAG = -5
MAX_LAG = 5
LAGS = list(range(MIN_LAG, MAX_LAG + 1))

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


def correlation_at_lag(series_1, series_2, lag):

    aligned = pd.concat([
        series_1.rename("asset_1"), series_2.shift(lag).rename("asset_2_shifted")], axis=1).dropna()

    n_obs = len(aligned)

    if n_obs < 2: return np.nan, n_obs

    correlation = aligned["asset_1"].corr(aligned["asset_2_shifted"])
    return correlation, n_obs


def calculate_lagged_correlations(returns, lags):

    rows = []
    securities = returns.columns.tolist()

    for asset_1, asset_2 in combinations(securities, 2):
        for lag in lags:
            correlation, n_obs = correlation_at_lag(returns[asset_1], returns[asset_2], lag)
            rows.append(
                {
                    "asset_1": asset_1,
                    "asset_2": asset_2,
                    "pair": f"{asset_1}__{asset_2}",
                    "lag": lag,
                    "correlation": correlation,
                    "n_obs": n_obs
                }
            )

    lagged = pd.DataFrame(rows)
    return lagged


def build_lag_matrix(lagged_correlations):

    lag_matrix = lagged_correlations.pivot(index="lag", columns="pair", values="correlation")
    lag_matrix = lag_matrix.sort_index()
    return lag_matrix


def summarize_strongest_lags(lagged_correlations):

    rows = []

    for pair, group in lagged_correlations.groupby("pair", sort=True):

        group = group.sort_values("lag").reset_index(drop=True)

        zero_lag_row = group[group["lag"] == 0]
        zero_lag_corr = zero_lag_row["correlation"].iloc[0]

        strongest_idx = group["correlation"].abs().idxmax()
        strongest_row = group.loc[strongest_idx]

        rows.append(
            {
                "pair": pair,
                "asset_1": strongest_row["asset_1"],
                "asset_2": strongest_row["asset_2"],
                "strongest_lag": int(strongest_row["lag"]),
                "strongest_correlation": strongest_row["correlation"],
                "abs_strongest_correlation": abs(strongest_row["correlation"]),
                "zero_lag_correlation": zero_lag_corr,
                "difference_vs_zero_lag": strongest_row["correlation"] - zero_lag_corr
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(by="abs_strongest_correlation", ascending=False).reset_index(drop=True)

    return summary


def print_lag_matrix(portfolio_name, lag_matrix):

    print("\n" + "=" * 80)
    print(f"LAGGED CORRELATION MATRIX: {portfolio_name.upper()}")
    print("=" * 80)

    print(
        "Convention: corr(asset_1(t), asset_2(t - lag))\n"
        "  lag > 0  => asset_2 leads asset_1\n"
        "  lag < 0  => asset_1 leads asset_2\n"
    )

    print(lag_matrix.to_string(float_format=lambda x: f"{x:.4f}"))


def print_strongest_lag_summary(portfolio_name, summary):

    print("\n" + "=" * 80)
    print(f"STRONGEST LAG SUMMARY: {portfolio_name.upper()}")
    print("=" * 80)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def save_results(portfolio_name, lagged_correlations, lag_matrix, summary):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lagged_file = OUTPUT_DIR / f"{portfolio_name}_lagged_correlations_long.csv"
    matrix_file = OUTPUT_DIR / f"{portfolio_name}_lagged_correlation_matrix.csv"
    summary_file = OUTPUT_DIR / f"{portfolio_name}_strongest_lags.csv"

    lagged_correlations.to_csv(lagged_file, index=False)
    lag_matrix.to_csv(matrix_file, index=True)
    summary.to_csv(summary_file, index=False)

    print(f"\nSaved long-format lagged correlations:  {lagged_file}")
    print(f"Saved lagged correlation matrix:  {matrix_file}")
    print(f"Saved strongest-lag summary:  {summary_file}")


def process_portfolio(portfolio_name, return_file):

    returns = load_return_matrix(return_file)
    validate_returns(portfolio_name, returns)

    lagged_correlations = calculate_lagged_correlations(returns, lags=LAGS)
    lag_matrix = build_lag_matrix(lagged_correlations)
    summary = summarize_strongest_lags(lagged_correlations)

    print_lag_matrix(portfolio_name, lag_matrix)
    print_strongest_lag_summary(portfolio_name, summary)
    save_results(portfolio_name, lagged_correlations, lag_matrix, summary)


def main():

    print("=" * 80)
    print("LAGGED CROSS-CORRELATION ANALYSIS")
    print("=" * 80)

    process_portfolio(portfolio_name="diversified", return_file=DIVERSIFIED_RETURN_FILE)
    process_portfolio(portfolio_name="concentrated", return_file=CONCENTRATED_RETURN_FILE)

    print("\n" + "=" * 80)
    print("LAGGED CROSS-CORRELATION ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()