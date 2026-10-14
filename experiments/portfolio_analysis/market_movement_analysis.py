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

LOWER_QUANTILE = 0.05
UPPER_QUANTILE = 0.95

def load_return_matrix(file_path):

    if not file_path.exists():
        raise FileNotFoundError(f"Return file not found:\n{file_path}")

    returns = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    returns = returns.sort_index()

    for column in returns.columns:
        returns[column] = pd.to_numeric(returns[column], errors="coerce")

    return returns


def validate_returns(portfolio_name, returns):

    missing = returns.isna().sum().sum()
    non_finite = (~np.isfinite(returns.to_numpy())).sum()

    if missing > 0:
        raise ValueError(f"{portfolio_name}: missing return values found")

    if non_finite > 0:
        raise ValueError(f"{portfolio_name}: non-finite return values found")


def build_reference_basket(diversified, concentrated):

    """
    Internal equal weight reference basket, 
    using unique securities from both portfolios
    """

    combined = pd.concat([diversified, concentrated], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    common_dates = (diversified.index.intersection(concentrated.index).sort_values())
    combined = combined.loc[common_dates]

    basket_return = combined.mean(axis=1)
    basket_return.name = "reference_basket_return"

    print("\n" + "=" * 80)
    print("REFERENCE BASKET")
    print("=" * 80)

    print(f"Unique securities  : {combined.shape[1]}")
    print(f"Observations       : {len(combined):,}")
    print(f"First date         : {combined.index.min().date()}")
    print(f"Last date          : {combined.index.max().date()}")

    return basket_return


def classify_market_movements(basket_return):

    lower_threshold = basket_return.quantile(LOWER_QUANTILE)
    upper_threshold = basket_return.quantile(UPPER_QUANTILE)

    regime = pd.Series("normal", index=basket_return.index, dtype="object")
    regime.loc[basket_return <= lower_threshold] = "large_down"
    regime.loc[basket_return >= upper_threshold] = "large_up"

    print("\n" + "=" * 80)
    print("MOVEMENT THRESHOLDS")
    print("=" * 80)

    print(f"Bottom 5% threshold : {lower_threshold * 100:.3f}%")
    print(f"Top 5% threshold    : {upper_threshold * 100:.3f}%")

    print("\nNumber of dates:")
    print(regime.value_counts().reindex(["large_down", "normal", "large_up"]).to_string())

    return regime


def average_pairwise_correlation(returns):

    correlation = returns.corr()
    values = correlation.to_numpy()

    upper_triangle = values[np.triu_indices_from(values, k=1)]
    return upper_triangle.mean()


def average_sign_agreement(returns):

    agreements = []
    for asset_1, asset_2 in combinations(returns.columns, 2):
        same_direction = (np.sign(returns[asset_1]) == np.sign(returns[asset_2]))
        agreements.append(same_direction.mean())

    return np.mean(agreements)


def analyse_regime(portfolio_name, returns, regime_name, regime):

    selected_dates = regime[regime == regime_name].index
    subset = returns.loc[returns.index.intersection(selected_dates)]

    result = {
        "portfolio": portfolio_name,
        "regime": regime_name,
        "n_days": len(subset),
        "average_pairwise_correlation": average_pairwise_correlation(subset),
        "average_sign_agreement": average_sign_agreement(subset),
        "average_absolute_return": subset.abs().mean().mean()
    }
    return result


def analyse_portfolio(portfolio_name, returns, regime):

    rows = []
    for regime_name in ["large_down", "normal", "large_up"]:
        rows.append(analyse_regime(portfolio_name, returns, regime_name, regime))

    return pd.DataFrame(rows)


def print_results(results):

    display = results.copy()

    display["average_sign_agreement"] *= 100
    display["average_absolute_return"] *= 100

    print("\n" + "=" * 80)
    print("CO-MOVEMENT DURING LARGE MOVEMENTS")
    print("=" * 80)

    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSign agreement and absolute return shown as percentage")


def save_results(results, basket_return, regime):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "large_movement_comovement.csv"
    dates_file = OUTPUT_DIR / "market_movement_regimes.csv"

    results.to_csv(results_file, index=False)

    regime_data = pd.DataFrame(
        {
            "reference_basket_return": basket_return,
            "regime": regime
        }
    )
    regime_data.to_csv(dates_file, index=True)

    print(f"\nSaved co-movement results:  {results_file}")
    print(f"Saved movement regimes:  {dates_file}")


def main():

    print("=" * 80)
    print("LARGE MARKET MOVEMENT ANALYSIS")
    print("=" * 80)

    diversified = load_return_matrix(DIVERSIFIED_RETURN_FILE)
    concentrated = load_return_matrix(CONCENTRATED_RETURN_FILE)

    validate_returns("diversified", diversified)
    validate_returns("concentrated", concentrated)

    basket_return = build_reference_basket(diversified, concentrated)
    regime = classify_market_movements(basket_return)

    diversified_results = analyse_portfolio("diversified", diversified, regime)
    concentrated_results = analyse_portfolio("concentrated", concentrated, regime)

    results = pd.concat([diversified_results, concentrated_results], ignore_index=True)
    print_results(results)
    save_results(results, basket_return, regime)

    print("\n" + "=" * 80)
    print("LARGE MOVEMENT ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()