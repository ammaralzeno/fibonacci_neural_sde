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

DIVERSIFIED_CORRELATION_FILE = RESULTS_DIR / "diversified_correlation.csv"
CONCENTRATED_CORRELATION_FILE = RESULTS_DIR / "concentrated_correlation.csv"

DIVERSIFIED_ROLLING_FILE = RESULTS_DIR / "diversified_rolling_correlations.csv"
CONCENTRATED_ROLLING_FILE = RESULTS_DIR / "concentrated_rolling_correlations.csv"

DIVERSIFIED_LAG_FILE = RESULTS_DIR / "diversified_lagged_correlation_matrix.csv"
CONCENTRATED_LAG_FILE = RESULTS_DIR / "concentrated_lagged_correlation_matrix.csv"

LARGE_MOVEMENT_FILE = RESULTS_DIR / "large_movement_comovement.csv"
DIVERSIFIED_PCA_FILE = RESULTS_DIR / "diversified_pca_summary.csv"
CONCENTRATED_PCA_FILE = RESULTS_DIR / "concentrated_pca_summary.csv"

DIVERSIFIED_LABELS = {
    "00138001": "Energy",
    "00107801": "Health Care",
    "00141402": "Financials",
    "00116101": "Information Technology"
}


def asset_label(portfolio_name, issue_id):

    if portfolio_name == "diversified":
        return DIVERSIFIED_LABELS.get(issue_id, issue_id)

    return issue_id


def pair_label(portfolio_name, pair):

    asset_1, asset_2 = pair.split("__")
    return f"{asset_label(portfolio_name, asset_1)} / {asset_label(portfolio_name, asset_2)}"


def check_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Required result file not found:\n{file_path}")


def load_correlation_matrix(file_path):

    check_file(file_path)
    matrix = pd.read_csv(file_path, converters={0: str})
    matrix = matrix.set_index(matrix.columns[0])

    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    if matrix.isna().any().any():
        raise ValueError(f"Invalid values in correlation matrix:\n{file_path}")

    return matrix


def load_rolling_correlations(file_path):

    check_file(file_path)
    data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

    data = data.sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")

    if data.isna().any().any():
        raise ValueError(f"Invalid rolling correlations:\n{file_path}")

    return data


def load_lag_matrix(file_path):

    check_file(file_path)
    data = pd.read_csv(file_path, index_col="lag")

    data.index = pd.to_numeric(data.index, errors="raise").astype(int)
    data = data.sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")

    if data.isna().any().any():
        raise ValueError(f"Invalid lag correlations:\n{file_path}")

    return data


def plot_correlation_heatmap(portfolio_name, correlation):

    display = correlation.copy()

    display.index = [asset_label(portfolio_name, asset) for asset in display.index]
    display.columns = [asset_label(portfolio_name, asset) for asset in display.columns]

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(display.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(np.arange(len(display.columns)))
    ax.set_xticklabels(display.columns, rotation=30, ha="right")

    ax.set_yticks(np.arange(len(display.index)))
    ax.set_yticklabels(display.index)

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            value = display.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center")

    ax.set_title(f"{portfolio_name.capitalize()} Portfolio — Return Correlation Matrix")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Pearson correlation")

    output_file = FIGURES_DIR / f"{portfolio_name}_correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved: {output_file}")


def plot_pairwise_correlations(portfolio_name, rolling_data):

    fig, ax = plt.subplots(figsize=(13, 6))
    for pair in rolling_data.columns:
        ax.plot(rolling_data.index, rolling_data[pair], label=pair_label(portfolio_name, pair), linewidth=1.3, alpha=0.85)

    ax.axhline(y=0, linestyle="--", linewidth=0.8)
    ax.set_title(f"{portfolio_name.capitalize()} Portfolio — 60-Day Rolling Correlations")

    ax.set_xlabel("Date")
    ax.set_ylabel("Pearson correlation")

    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.2)
    ax.legend(title="Asset pairs", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)

    output_file = FIGURES_DIR / f"{portfolio_name}_rolling_correlations.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved: {output_file}")


def calculate_average_rolling_correlation(rolling_data):

    return rolling_data.mean(axis=1)


def plot_average_rolling_comparison(diversified, concentrated):

    common_dates = diversified.index.intersection(concentrated.index).sort_values()
    if common_dates.empty:
        raise ValueError("The portfolios have no common rolling-correlation dates")

    diversified_avg = calculate_average_rolling_correlation(diversified.loc[common_dates])
    concentrated_avg = calculate_average_rolling_correlation(concentrated.loc[common_dates])

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
    print("\nROLLING CORRELATION COMPARISON")
    print("-" * 60)

    print(f"Diversified mean rolling correlation :  {diversified_avg.mean():.4f}")
    print(f"Concentrated mean rolling correlation:  {concentrated_avg.mean():.4f}")

    print(f"Dates concentrated > diversified     :  {(concentrated_avg > diversified_avg).sum():,}")
    print(f"Dates diversified > concentrated     :  {(diversified_avg > concentrated_avg).sum():,}")

    print(f"Total comparison dates               : {len(common_dates):,}")


def plot_lag_heatmap(portfolio_name, lag_matrix, shared_limit):

    display = lag_matrix.copy()
    display.columns = [pair_label(portfolio_name, pair) for pair in display.columns]

    fig, ax = plt.subplots(figsize=(13, 7))
    image = ax.imshow(display.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-shared_limit, vmax=shared_limit)

    ax.set_xticks(np.arange(len(display.columns)))
    ax.set_xticklabels(display.columns, rotation=35, ha="right")

    ax.set_yticks(np.arange(len(display.index)))
    ax.set_yticklabels(display.index)

    for row in range(display.shape[0]):
        for column in range(display.shape[1]):
            value = display.iloc[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7)

    # Highlight zero lag row.
    lag_values = list(display.index)
    zero_position = (lag_values.index(0))
    ax.axhline(zero_position - 0.5, linewidth=1.5)
    ax.axhline(zero_position + 0.5, linewidth=1.5)

    ax.set_title(f"{portfolio_name.capitalize()} Portfolio — Lagged Cross-Correlations")
    ax.set_xlabel("Asset pair")
    ax.set_ylabel("Lag (trading days)")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Pearson correlation")

    output_file = FIGURES_DIR / f"{portfolio_name}_lag_correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved: {output_file}")


def plot_market_movement_sign_agreement(results):

    regime_order = ["large_down", "normal", "large_up"]
    regime_labels = {"large_down": "Large down", "normal": "Normal", "large_up": "Large up"}

    pivot = results.pivot(index="regime", columns="portfolio", values="average_sign_agreement")
    pivot = pivot.loc[regime_order]

    diversified = pivot["diversified"].to_numpy() * 100
    concentrated = pivot["concentrated"].to_numpy() * 100

    positions = np.arange(len(regime_order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 6))

    bars_diversified = ax.bar(positions - width / 2, diversified, width, label="Diversified")
    bars_concentrated = ax.bar(positions + width / 2, concentrated, width, label="Concentrated")

    ax.set_xticks(positions)
    ax.set_xticklabels([regime_labels[regime] for regime in regime_order])

    ax.set_ylim(0, 100)
    ax.set_ylabel("Same-direction pairwise agreement (%)")
    ax.set_xlabel("Reference-basket movement regime")

    ax.set_title("Directional Co-Movement During Large Market Movements")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    ax.bar_label(bars_diversified, fmt="%.1f%%", padding=3)
    ax.bar_label(bars_concentrated, fmt="%.1f%%", padding=3)

    output_file = FIGURES_DIR / "market_movement_sign_agreement.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved: {output_file}")


def plot_pca_explained_variance(diversified, concentrated):

    components = diversified["component"].tolist()
    diversified_variance = diversified["explained_variance_ratio"].to_numpy()* 100
    concentrated_variance = concentrated["explained_variance_ratio"].to_numpy() * 100

    positions = np.arange(len(components))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 6))
    diversified_bars = ax.bar(positions - width / 2, diversified_variance, width, label="Diversified")
    concentrated_bars = ax.bar(positions + width / 2, concentrated_variance, width, label="Concentrated")

    ax.set_xticks(positions)
    ax.set_xticklabels(components)
    ax.set_xlabel("Principal component")

    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA Explained Variance — Diversified vs Concentrated")

    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    ax.bar_label(diversified_bars, fmt="%.1f%%", padding=3)
    ax.bar_label(concentrated_bars, fmt="%.1f%%", padding=3)

    output_file = FIGURES_DIR / "pca_explained_variance_comparison.png"
    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved: {output_file}")


def main():

    print("=" * 80)
    print("FINAL PORTFOLIO ANALYSIS VISUALIZATIONS")
    print("=" * 80)


    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    diversified_correlation = load_correlation_matrix(DIVERSIFIED_CORRELATION_FILE)
    concentrated_correlation = load_correlation_matrix(CONCENTRATED_CORRELATION_FILE)
    plot_correlation_heatmap("diversified", diversified_correlation)
    plot_correlation_heatmap("concentrated", concentrated_correlation)

    diversified_rolling = load_rolling_correlations(DIVERSIFIED_ROLLING_FILE)
    concentrated_rolling = load_rolling_correlations(CONCENTRATED_ROLLING_FILE)
    plot_pairwise_correlations("diversified", diversified_rolling)
    plot_pairwise_correlations("concentrated", concentrated_rolling)
    plot_average_rolling_comparison(diversified_rolling, concentrated_rolling)

    diversified_lag = load_lag_matrix(DIVERSIFIED_LAG_FILE)
    concentrated_lag = load_lag_matrix(CONCENTRATED_LAG_FILE)
    shared_lag_limit = max(np.abs(diversified_lag.to_numpy()).max(), np.abs(concentrated_lag.to_numpy()).max())
    plot_lag_heatmap("diversified", diversified_lag, shared_limit=shared_lag_limit)
    plot_lag_heatmap("concentrated", concentrated_lag, shared_limit=shared_lag_limit)

    check_file(LARGE_MOVEMENT_FILE)
    movement_results = pd.read_csv(LARGE_MOVEMENT_FILE)
    plot_market_movement_sign_agreement(movement_results)

    check_file(DIVERSIFIED_PCA_FILE)
    check_file(CONCENTRATED_PCA_FILE)
    diversified_pca = pd.read_csv(DIVERSIFIED_PCA_FILE)
    concentrated_pca = pd.read_csv(CONCENTRATED_PCA_FILE)
    plot_pca_explained_variance(diversified_pca, concentrated_pca)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()