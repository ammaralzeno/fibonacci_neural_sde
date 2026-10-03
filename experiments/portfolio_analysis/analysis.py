from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "data"

DIVERSIFIED_PRICE_FILE = DATA_DIR / "diversified_prices.csv"
CONCENTRATED_PRICE_FILE = DATA_DIR / "concentrated_prices.csv"

DIVERSIFIED_RETURN_FILE = DATA_DIR / "diversified_returns.csv"
CONCENTRATED_RETURN_FILE = DATA_DIR / "concentrated_returns.csv"

def load_price_matrix(file_path):

    if not file_path.exists():
        raise FileNotFoundError(f"Price file not found:\n{file_path}")

    prices = pd.read_csv(file_path, parse_dates=["Date"])
    prices = prices.set_index("Date")
    prices = prices.sort_index()

    for column in prices.columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    return prices

def validate_prices(portfolio_name, prices):

    print("\n" + "=" * 80)
    print(f"PRICE DATA: {portfolio_name.upper()}")
    print("=" * 80)

    print(f"Observations : {len(prices):,}")
    print(f"Securities   : {prices.shape[1]}")

    print(f"First date   : {prices.index.min().date()}")
    print(f"Last date    : {prices.index.max().date()}")
    print(f"Missing values: {prices.isna().sum().sum():,}")

    duplicate_dates = prices.index.duplicated().sum()
    print(f"Duplicate dates: {duplicate_dates:,}")

    non_positive = (prices <= 0).sum().sum()
    print(f"Non-positive prices: {non_positive:,}")

    if prices.isna().any().any():
        raise ValueError(f"{portfolio_name}: price data contains missing values")

    if duplicate_dates > 0:
        raise ValueError(f"{portfolio_name}: duplicate dates found")

    if non_positive > 0:
        raise ValueError(f"{portfolio_name}: non-positive prices found")

def calculate_returns(prices):

    returns = prices.pct_change(fill_method=None)
    returns = returns.iloc[1:].copy()

    return returns

def validate_returns(portfolio_name, returns):

    print("\n" + "=" * 80)
    print(f"DAILY RETURNS: {portfolio_name.upper()}")
    print("=" * 80)

    print(f"Observations : {len(returns):,}")
    print(f"Securities   : {returns.shape[1]}")

    print(f"First date   : {returns.index.min().date()}")
    print(f"Last date    : {returns.index.max().date()}")

    missing_values = returns.isna().sum().sum()
    print(f"Missing values: {missing_values:,}")

    values = returns.to_numpy()
    non_finite = (~np.isfinite(values)).sum()
    print(f"Non-finite values: {non_finite:,}")

    if missing_values > 0:
        raise ValueError(f"{portfolio_name}: returns contain missing values")

    if non_finite > 0:
        raise ValueError(f"{portfolio_name}: returns contain NaN or infinite values")

    print("\nFirst 5 return rows:")
    print(returns.head())

    print("\nLast 5 return rows:")
    print(returns.tail())

def print_return_summary(portfolio_name, returns):

    summary = pd.DataFrame(
        {
            "mean_daily_return": returns.mean(),
            "daily_std": returns.std(),
            "min_return": returns.min(),
            "max_return": returns.max()
        }
    )

    summary["mean_daily_return_pct"] = summary["mean_daily_return"] * 100
    summary["daily_std_pct"] = summary["daily_std"] * 100
    summary["min_return_pct"] = summary["min_return"] * 100
    summary["max_return_pct"] = summary["max_return"] * 100

    print("\n" + "=" * 80)
    print(f"RETURN SUMMARY: {portfolio_name.upper()}")
    print("=" * 80)

    columns_to_show = ["mean_daily_return_pct", "daily_std_pct", "min_return_pct", "max_return_pct"]
    print(summary[columns_to_show].to_string(float_format=lambda x: f"{x:.4f}"))

def save_returns(returns, output_file):

    returns.to_csv(output_file, index=True)
    print(f"\nSaved returns:\n{output_file}")

def process_portfolio(portfolio_name, price_file, return_file):

    prices = load_price_matrix(price_file)
    validate_prices(portfolio_name, prices)
    returns = calculate_returns(prices)
    validate_returns(portfolio_name, returns)
    print_return_summary(portfolio_name, returns)
    save_returns(returns, return_file)

    return returns

def main():

    print("=" * 80)
    print("PORTFOLIO RETURN PREPARATION")
    print("=" * 80)

    diversified_returns = process_portfolio(portfolio_name="diversified", price_file=DIVERSIFIED_PRICE_FILE, return_file=DIVERSIFIED_RETURN_FILE)
    concentrated_returns = process_portfolio(portfolio_name="concentrated", price_file=CONCENTRATED_PRICE_FILE, return_file=CONCENTRATED_RETURN_FILE)

    print("\n" + "=" * 80)
    print("RETURN PREPARATION COMPLETE")
    print("=" * 80)

    print(f"\nDiversified return observations: {len(diversified_returns):,}")
    print(f"Concentrated return observations: {len(concentrated_returns):,}")

if __name__ == "__main__":
    main()