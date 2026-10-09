from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "results"
FIGURES_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "figures"

DIVERSIFIED_FILE = RESULTS_DIR / "diversified_rolling_correlations.csv"
CONCENTRATED_FILE = RESULTS_DIR / "concentrated_rolling_correlations.csv"

DIVERSIFIED_LABELS = {
    "00138001": "Energy",
    "00107801": "Health Care",
    "00141402": "Financials",
    "00116101": "Information Technology"
}


def load_rolling_correlations(file_path):

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    data = data.sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")

    if data.empty:
        raise ValueError(f"Empty dataset: {file_path}")

    if data.index.has_duplicates:
        raise ValueError(f"Duplicate dates: {file_path}")

    if not np.isfinite(data.to_numpy()).all():
        raise ValueError(f"Invalid correlation values: {file_path}")

    if data.shape[1] != 6:
        raise ValueError(f"Expected 6 asset pairs, found {data.shape[1]}")

    return data


def plot_pairwise_correlations(portfolio_name, rolling_data):

    fig, ax = plt.subplots(figsize=(13, 6))

    for pair in rolling_data.columns:
        asset_1, asset_2 = pair.split("__")

        if portfolio_name == "diversified":
            label = (f"{DIVERSIFIED_LABELS[asset_1]} / "f"{DIVERSIFIED_LABELS[asset_2]}")
        else:
            label = f"{asset_1} / {asset_2}"

        ax.plot(rolling_data.index, rolling_data[pair], label=label, linewidth=1.3, alpha=0.85)

        
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"{portfolio_name.capitalize()} Portfolio — 60-Day Rolling Correlations")

    ax.set_xlabel("Date")
    ax.set_ylabel("Pearson Correlation")

    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.2)

    ax.legend(title="Asset pairs", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)

    output_file = FIGURES_DIR / f"{portfolio_name}_rolling_correlations.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved: {output_file}")


def calculate_average_rolling_correlation(rolling_data):

    return rolling_data.mean(axis=1)


def plot_portfolio_comparison(diversified_data, concentrated_data):

    common_dates = diversified_data.index.intersection(concentrated_data.index).sort_values()
    if common_dates.empty:
        raise ValueError("The portfolios have no common rolling-correlation dates")

    diversified_avg = calculate_average_rolling_correlation(diversified_data.loc[common_dates])
    concentrated_avg = calculate_average_rolling_correlation(concentrated_data.loc[common_dates])

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(common_dates, diversified_avg, label="Diversified", linewidth=1.7)
    ax.plot(common_dates, concentrated_avg, label="Concentrated", linewidth=1.7)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Diversified vs Concentrated — Average 60-Day Rolling Correlation")

    ax.set_xlabel("Date")
    ax.set_ylabel("Average Pairwise Correlation")

    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.2)
    ax.legend()

    output_file = FIGURES_DIR / "average_rolling_correlation_comparison.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved: {output_file}")

    #Summary
    print("\nPORTFOLIO COMPARISON")
    print("-" * 60)

    print(f"Diversified mean rolling correlation :  {diversified_avg.mean():.4f}")
    print(f"Concentrated mean rolling correlation:  {concentrated_avg.mean():.4f}")

    print(f"Dates concentrated > diversified     :  {(concentrated_avg > diversified_avg).sum():,}")
    print(f"Dates diversified > concentrated     :  {(diversified_avg > concentrated_avg).sum():,}")

    print(f"Total comparison dates               : {len(common_dates):,}")


def main():

    print("=" * 80)
    print("ROLLING CORRELATION VISUALIZATION")
    print("=" * 80)


    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    diversified = load_rolling_correlations(DIVERSIFIED_FILE)
    concentrated = load_rolling_correlations(CONCENTRATED_FILE)

    print(f"\nDiversified shape : {diversified.shape}")
    print(f"Concentrated shape: {concentrated.shape}")

    plot_pairwise_correlations("diversified", diversified)
    plot_pairwise_correlations("concentrated", concentrated)
    plot_portfolio_comparison(diversified, concentrated)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()