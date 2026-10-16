from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from amgm.utils.common import calculate_fibLevels

LOOKBACK_WINDOW = 252

PRICE_FILE = REPO_ROOT / "workspace" / "portfolio_analysis" / "data" / "diversified_prices.csv"
OUTPUT_DIR = REPO_ROOT / "workspace" / "fibonacci_multiasset" / "data"
OUTPUT_FILE = OUTPUT_DIR / "multi_asset_fibonacci_features.csv"
VALIDATION_FILE = OUTPUT_DIR / "fibonacci_feature_validation.csv"

ASSET_NAMES = {
    "00138001": "Energy",
    "00107801": "Health Care",
    "00141402": "Financials",
    "00116101": "Information Technology"
}

FIB_RATIOS = np.array([0.000, 0.236, 0.382, 0.500, 0.618, 0.786, 1.000], dtype=float)
FIB_LABELS = ["000", "236", "382", "500", "618", "786", "1000"]


def load_prices():

    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Diversified price file not found:\n{PRICE_FILE}")

    prices = pd.read_csv(PRICE_FILE, parse_dates=["Date"])
    prices = prices.set_index("Date").sort_index()

    missing_assets = set(ASSET_NAMES) - set(prices.columns)
    if missing_assets:
        raise ValueError(f"Missing required assets: {sorted(missing_assets)}")

    prices = prices[list(ASSET_NAMES.keys())].copy()

    for column in prices.columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    return prices


def validate_prices(prices):

    print("\n" + "=" * 80)
    print("SOURCE PRICE DATA")
    print("=" * 80)

    print(f"Observations : {len(prices):,}")
    print(f"Assets       : {prices.shape[1]}")
    print(f"First date   : {prices.index.min().date()}")
    print(f"Last date    : {prices.index.max().date()}")

    missing = int(prices.isna().sum().sum())
    non_positive = int((prices <= 0).sum().sum())
    duplicate_dates = int(prices.index.duplicated().sum()) 

    print(f"Missing values      : {missing:,}")
    print(f"Non-positive prices : {non_positive:,}")
    print(f"Duplicate dates     : {duplicate_dates:,}")
    print(f"Lookback window     : {LOOKBACK_WINDOW} trading days")

    if missing > 0:
        raise ValueError("Price matrix contains missing values")

    if non_positive > 0:
        raise ValueError("Price matrix contains non-positive values")

    if duplicate_dates > 0:
        raise ValueError("Price matrix contains duplicate dates")

    if len(prices) < LOOKBACK_WINDOW:
        raise ValueError("Not enough observations for the Fibonacci lookback")


def calculate_asset_features(issue_id, asset_name, prices):

    values = prices.to_numpy(dtype=float)
    dates = prices.index.to_numpy()

    windows = sliding_window_view(values, window_shape=LOOKBACK_WINDOW)
    feature_dates = dates[LOOKBACK_WINDOW - 1:]
    expected_windows = len(values) - LOOKBACK_WINDOW + 1

    if len(windows) != expected_windows:
        raise RuntimeError("Unexpected rolling-window count")

    fib_levels, delta = calculate_fibLevels(windows)
    delta = np.asarray(delta, dtype=float)
    current_prices = windows[:, -1].reshape(-1, 1)
    safe_delta = np.maximum(delta, 1e-8)
    fib_features = (current_prices - fib_levels) / safe_delta

    window_min = windows.min(axis=1)
    window_max = windows.max(axis=1)
    window_range = window_max - window_min

    nearest_index = np.argmin(np.abs(fib_features), axis=1)
    row_indices = np.arange(len(windows))
    nearest_ratio = FIB_RATIOS[nearest_index]
    nearest_level_price = fib_levels[row_indices, nearest_index]
    nearest_feature = fib_features[row_indices, nearest_index]

    output = pd.DataFrame(
        {
            "Date": feature_dates,
            "IssueId": issue_id,
            "Asset": asset_name,
            "Price": current_prices[:, 0],
            "WindowMin": window_min,
            "WindowMax": window_max,
            "WindowRange": window_range,
            "NearestFibRatio": nearest_ratio,
            "NearestFibPrice": nearest_level_price,
            "NearestFibFeature": nearest_feature,
            "AbsDistanceToNearestFib": np.abs(nearest_feature)
        }
    )

    for index, label in enumerate(FIB_LABELS):
        output[f"Fib_{label}"] = fib_levels[:, index]

    for index, label in enumerate(FIB_LABELS):
        output[f"Feature_{label}"] = fib_features[:, index]

    return output, windows, fib_features


def validate_repository_equivalence(issue_id, windows, raw_features):

    window_min = windows.min(axis=1, keepdims=True)
    window_max = windows.max(axis=1, keepdims=True)
    window_range = np.maximum(window_max - window_min, 1e-8)

    normalized_windows = (windows - window_min) / window_range
    normalized_levels, normalized_delta = calculate_fibLevels(normalized_windows)
    normalized_current_price = normalized_windows[:, -1:]
    normalized_features = (normalized_current_price - normalized_levels) / np.maximum(normalized_delta, 1e-8)

    differences = np.abs(raw_features - normalized_features)
    max_difference = float(np.max(differences))
    mean_difference = float(np.mean(differences))

    equivalent = bool(np.allclose(raw_features, normalized_features, atol=1e-6, rtol=1e-6))

    return {
        "IssueId": issue_id,
        "n_windows": len(windows),
        "max_feature_difference": max_difference,
        "mean_feature_difference": mean_difference,
        "repository_equivalent": equivalent
    }


def validate_fibonacci_output(issue_id, features, expected_rows):

    print("\n" + "=" * 80)
    print(f"FIBONACCI FEATURES: {ASSET_NAMES[issue_id].upper()} ({issue_id})")
    print("=" * 80)

    print(f"Feature observations : {len(features):,}")
    print(f"First feature date   : {features['Date'].min().date()}")
    print(f"Last feature date    : {features['Date'].max().date()}")

    if len(features) != expected_rows:
        raise ValueError(f"{issue_id}: expected {expected_rows} feature rows, found {len(features)}")

    fib_columns = [f"Fib_{label}" for label in FIB_LABELS]
    feature_columns = [f"Feature_{label}" for label in FIB_LABELS]

    if features[fib_columns].isna().any().any():
        raise ValueError(f"{issue_id}: missing Fibonacci levels found")
    
    if features[feature_columns].isna().any().any():
        raise ValueError(f"{issue_id}: missing Fibonacci features found")

    fib_values = features[fib_columns].to_numpy()
    if np.any(np.diff(fib_values, axis=1) < 0):
        raise ValueError(f"{issue_id}: Fibonacci levels are not ordered")

    if not np.allclose(features["Fib_000"], features["WindowMin"]):
        raise ValueError(f"{issue_id}: Fib_000 != rolling minimum")

    if not np.allclose(features["Fib_1000"], features["WindowMax"]):
        raise ValueError(f"{issue_id}: Fib_1000 != rolling maximum")

    print(f"Mean normalized distance to nearest level   : {features['AbsDistanceToNearestFib'].mean():.4f}")
    print(f"Median normalized distance to nearest level : {features['AbsDistanceToNearestFib'].median():.4f}")

    print("\nFirst 3 rows:")

    display_columns = [
        "Date",
        "Price",
        "WindowMin",
        "WindowMax",
        "NearestFibRatio",
        "NearestFibPrice",
        "NearestFibFeature"
    ]

    print(features[display_columns].head(3).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():

    print("=" * 80)
    print("MULTI-ASSET FIBONACCI FEATURE CONSTRUCTION")
    print("=" * 80)

    prices = load_prices()
    validate_prices(prices)
    expected_rows = len(prices) - LOOKBACK_WINDOW + 1

    all_features = []
    validation_rows = []

    for issue_id, asset_name in ASSET_NAMES.items():

        asset_features, windows, fib_features = calculate_asset_features(issue_id=issue_id, asset_name=asset_name, prices=prices[issue_id])
        validate_fibonacci_output(issue_id, asset_features, expected_rows)

        equivalence = validate_repository_equivalence(issue_id, windows, fib_features)
        print("\nRepository-equivalence check:")

        print(f"  Equivalent             : {equivalence['repository_equivalent']}")
        print(f"  Maximum feature error  : {equivalence['max_feature_difference']:.10f}")
        
        validation_rows.append(equivalence)
        all_features.append(asset_features)


    features = pd.concat(all_features, ignore_index=True)
    features = features.sort_values(["Date", "IssueId"]).reset_index(drop=True)
    validation = pd.DataFrame(validation_rows)

    date_counts = (features.groupby("Date")["IssueId"].nunique())

    if not (date_counts == len(ASSET_NAMES)).all():
        raise ValueError("Not every Fibonacci feature date contains all four assets")
    
    if not validation["repository_equivalent"].all():
        raise ValueError("At least one asset did not reproduce the repository Fibonacci representation")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_FILE, index=False)
    validation.to_csv(VALIDATION_FILE, index=False)

    print("\n" + "=" * 80)
    print("FINAL MULTI-ASSET DATASET")
    print("=" * 80)

    print(f"Total rows       : {len(features):,}")
    print(f"Assets           : {features['IssueId'].nunique()}")
    print(f"Common dates     : {features['Date'].nunique():,}")

    print(f"First date       : {features['Date'].min().date()}")
    print(f"Last date        : {features['Date'].max().date()}")

    print(f"\nSaved Fibonacci features:\n{OUTPUT_FILE}")
    print(f"\nSaved validation summary:\n{VALIDATION_FILE}")

    print("\n" + "=" * 80)
    print("FIBONACCI FEATURE CONSTRUCTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()