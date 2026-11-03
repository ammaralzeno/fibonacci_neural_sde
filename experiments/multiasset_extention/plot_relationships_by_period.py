from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "workspace" / "multiasset_extensions" / "relationships"
FIG_DIR = ROOT / "workspace" / "multiasset_extensions" / "figures" / "relationships"

PORTFOLIOS = [
    "diversified",
    "concentrated",
    "ultra_concentrated"
]

PERIODS = [
    "2016-2017",
    "2018-2019",
    "2020"
]

PORTFOLIO_LABELS = {
    "diversified": "Diversified",
    "concentrated": "Concentrated IT",
    "ultra_concentrated": "Homebuilders"
}

PERIOD_LABELS = {
    "2016-2017": "2016–2017",
    "2018-2019": "2018–2019",
    "2020": "2020"
}

ASSET_LABELS = {
    "diversified": {
        "00138001": "Energy",
        "00107801": "Health",
        "00141402": "Financials",
        "00116101": "IT"
    },
    "concentrated": {
        "00116101": "IT 1161",
        "00169001": "IT 1690",
        "00177301": "IT 1773",
        "00187801": "IT 1878"
    },
    "ultra_concentrated": {
        "00666901": "LEN",
        "00882301": "PHM",
        "01239501": "TOL",
        "01268901": "KBH"
    }
}

COLORS = {
    "diversified": "#4477AA",
    "concentrated": "#EE9944",
    "ultra_concentrated": "#228866"
}

def load_results():

    summary = pd.read_csv(DATA_DIR / "relationship_period_summary.csv")
    pairs = pd.read_csv(DATA_DIR / "relationship_period_pairwise.csv", dtype={"IssueId1": str, "IssueId2": str})
    lags = pd.read_csv(DATA_DIR / "relationship_period_lagged.csv", dtype={"IssueId1": str, "IssueId2": str})
    rolling = pd.read_csv(DATA_DIR / "relationship_rolling_60d.csv", parse_dates=["Date"])

    if len(summary) != 12:
        raise ValueError("Expected 12 summary rows")

    if len(pairs) != 72:
        raise ValueError("Expected 72 pairwise rows")

    if len(lags) != 792:
        raise ValueError("Expected 792 lag rows")

    if len(rolling) != 4356:
        raise ValueError("Expected 4356 rolling observations")

    for name, data in {"summary": summary, "pairs": pairs, "lags": lags, "rolling": rolling}.items():
        if data.isna().any().any():
            raise ValueError(f"Missing values in {name}")

    if summary.duplicated(["Portfolio", "Period"]).any():
        raise ValueError("Duplicate portfolio-period summaries")

    if pairs.duplicated(["Portfolio", "Period", "IssueId1", "IssueId2"]).any():
        raise ValueError("Duplicate pairwise results")

    if lags.duplicated(["Portfolio", "Period", "IssueId1", "IssueId2", "LagDays"]).any():
        raise ValueError("Duplicate lag measurements")

    if rolling.duplicated(["Portfolio", "Date"]).any():
        raise ValueError("Duplicate rolling dates")

    for portfolio in PORTFOLIOS:
        for period in PERIODS:

            pair_subset = pairs[(pairs["Portfolio"] == portfolio) & (pairs["Period"] == period)]
            lag_subset = lags[(lags["Portfolio"] == portfolio) & (lags["Period"] == period)]

            if len(pair_subset) != 6:
                raise ValueError(f"Expected 6 pairs: {portfolio}/{period}")

            if len(lag_subset) != 66:
                raise ValueError(f"Expected 66 lag values: {portfolio}/{period}")


    print("Plot input validation: PASSED")
    return summary, pairs, lags, rolling


def plot_correlation_matrices(pairs):

    fig, axes = plt.subplots(3, 3, figsize=(16, 15), constrained_layout=True)
    for i, portfolio in enumerate(PORTFOLIOS):

        labels = ASSET_LABELS[portfolio]
        ids = list(labels)
        lookup = {iid: position for position, iid in enumerate(ids)}

        for j, period in enumerate(PERIODS):

            subset = pairs[(pairs["Portfolio"] == portfolio) & (pairs["Period"] == period)]
            matrix = np.eye(4)

            for row in subset.itertuples(index=False):

                if row.IssueId1 not in lookup or row.IssueId2 not in lookup:
                    raise ValueError(f"Unexpected asset: {portfolio}")

                a = lookup[row.IssueId1]
                b = lookup[row.IssueId2]

                matrix[a, b] = row.Correlation
                matrix[b, a] = row.Correlation

            ax = axes[i, j]
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)

            ax.set_xticks(range(4), list(labels.values()), rotation=35, ha="right")
            ax.set_yticks(range(4), list(labels.values()))
            ax.set_title(f"{PORTFOLIO_LABELS[portfolio]}· {PERIOD_LABELS[period]}", fontsize=11)

            for a in range(4):
                for b in range(4):

                    value = matrix[a, b]
                    ax.text(b, a, f"{value:.2f}", ha="center", va="center", color=("white" if value >= 0.65 else "black"), fontsize=10)

    fig.colorbar(image, ax=axes, shrink=0.65, label="Pearson return correlation")
    fig.suptitle("Pairwise Return Correlations Across Portfolios and Historical Periods", fontsize=15)

    path = FIG_DIR / "01_period_correlation_matrices.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved: {path}")


def plot_rolling_correlations(rolling):

    matrix = rolling.pivot(index="Date", columns="Portfolio", values="RollingMeanPairCorrelation").sort_index()
    matrix = matrix.loc["2016-01-01":"2020-12-31"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), color="gray", alpha=0.10, label="2020")

    for portfolio in PORTFOLIOS:
        ax.plot(matrix.index, matrix[portfolio], linewidth=1.6, color=COLORS[portfolio], label=PORTFOLIO_LABELS[portfolio])

    for boundary in ["2018-01-01", "2020-01-01"]:
        ax.axvline(pd.Timestamp(boundary), color="black", linestyle="--", alpha=0.35, linewidth=1)

    ax.set_title("Rolling 60-Day Average Pairwise Return Correlation")

    ax.set_xlabel("Date")
    ax.set_ylabel("Mean of six pairwise correlations")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    fig.tight_layout()
    path = FIG_DIR / "02_rolling_correlation_comparison.png"

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_pca(summary):

    sample = summary[summary["Period"].isin(PERIODS)].copy()

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(3)
    width = 0.24

    for i, portfolio in enumerate(PORTFOLIOS):
        values = sample[sample["Portfolio"] == portfolio].set_index("Period").reindex(PERIODS)["PC1ExplainedVariancePct"].to_numpy()
        bars = ax.bar(x + (i - 1) * width, values, width, color=COLORS[portfolio], label=PORTFOLIO_LABELS[portfolio])
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)

    ax.set_xticks(x, [PERIOD_LABELS[p] for p in PERIODS])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Variance explained by PC1 (%)")

    ax.set_title("Common-Factor Strength Across Portfolios")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    fig.tight_layout()
    path = FIG_DIR / "03_pca_common_factor_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {path}")


def plot_lag_profiles(lags):

    mean_lags = lags[lags["Period"].isin(PERIODS)].groupby(["Portfolio", "Period", "LagDays"], as_index=False)["Correlation"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)

    for ax, period in zip(axes, PERIODS):

        for portfolio in PORTFOLIOS:
            subset = mean_lags[(mean_lags["Portfolio"] == portfolio) & (mean_lags["Period"] == period)].sort_values("LagDays")
            ax.plot(subset["LagDays"], subset["Correlation"], marker="o", markersize=4, linewidth=1.7, color=COLORS[portfolio], label=PORTFOLIO_LABELS[portfolio])

        ax.axvline(0, linestyle="--", color="black", linewidth=1, alpha=0.6)
        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)

        ax.set_title(PERIOD_LABELS[period])
        ax.set_xlabel("Lag (trading days)")
        ax.set_xticks(range(-5, 6, 2))
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Mean pairwise lagged-return correlation")
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Cross-Asset Lag Profiles by Historical Period", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.87])

    path = FIG_DIR / "04_lag_correlation_profiles.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {path}")


def main():

    print("=" * 80)
    print("HISTORICAL RELATIONSHIP VISUALIZATIONS")
    print("=" * 80)

    summary, pairs, lags, rolling = load_results()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_correlation_matrices(pairs)
    plot_rolling_correlations(rolling)
    plot_pca(summary)
    plot_lag_profiles(lags)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()