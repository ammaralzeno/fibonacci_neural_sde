from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = REPO_ROOT / "workspace" / "fibonacci_multiasset" / "data" / "multi_asset_fibonacci_features.csv"
FIGURES_DIR = REPO_ROOT / "workspace" / "fibonacci_multiasset" / "figures"
OUTPUT_FILE = FIGURES_DIR / "multi_asset_fibonacci_representation.png"

ASSET_ORDER = ["Energy", "Health Care", "Financials", "Information Technology"]
FIB_RATIOS = [0.000, 0.236, 0.382, 0.500, 0.618, 0.786, 1.000]


def load_features():

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Fibonacci feature file not found:\n{DATA_FILE}")

    features = pd.read_csv(DATA_FILE, parse_dates=["Date"], dtype={"IssueId": str})
    features = features.sort_values(["Asset", "Date"])

    required_columns = {"Date", "IssueId", "Asset", "Feature_000"}
    missing = required_columns - set(features.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if features[list(required_columns)].isna().any().any():
        raise ValueError("Missing values found in required Fibonacci features")

    return features


def plot_multi_asset_fibonacci(features):

    fig, axes = plt.subplots(nrows=len(ASSET_ORDER), ncols=1, figsize=(14, 11), sharex=True, sharey=True)

    for ax, asset_name in zip(axes, ASSET_ORDER):

        asset_data = features[features["Asset"] == asset_name].sort_values("Date")

        if asset_data.empty:
            raise ValueError(f"No data found for {asset_name}")


        normalized_position = asset_data["Feature_000"]
        ax.plot(asset_data["Date"], normalized_position, linewidth=1.1)

        for ratio in FIB_RATIOS:
            ax.axhline(ratio, linestyle="--", linewidth=0.8, alpha=0.65)
            ax.text(1.003, ratio, f"{ratio:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)

        ax.set_title(asset_name)
        ax.set_ylabel("252-day\nrange position")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Multi-Asset Fibonacci Representation — 252-Day Rolling Price Range", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.97, 0.97])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved:\n{OUTPUT_FILE}")


def main():

    print("=" * 80)
    print("MULTI-ASSET FIBONACCI VISUALIZATION")
    print("=" * 80)

    features = load_features()

    print(f"Rows        : {len(features):,}")
    print(f"Assets      : {features['Asset'].nunique()}")
    print(f"Dates       : {features['Date'].nunique():,}")

    plot_multi_asset_fibonacci(features)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()