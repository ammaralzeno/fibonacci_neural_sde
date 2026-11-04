from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]

REL_DIR = ROOT / "workspace" / "multiasset_extensions" / "relationships"
FIB_DIR = ROOT / "workspace" / "multiasset_extensions" / "fibonacci"
FIG_DIR = ROOT / "workspace" / "multiasset_extensions" / "figures" / "fibonacci"

PORTFOLIOS = ["diversified", "concentrated", "ultra_concentrated"]
PERIODS = ["2016-2017", "2018-2019", "2020"]
HORIZONS = [1, 5, 20]

LABELS = {
    "diversified": "Diversified",
    "concentrated": "Concentrated IT",
    "ultra_concentrated": "Homebuilders"
}

PERIOD_LABELS = ["2016–17", "2018–19", "2020"]

COLORS = {
    "diversified": "#4477AA",
    "concentrated": "#EE9944",
    "ultra_concentrated": "#228866"
}

def load_results():

    files = {
        "relationships": REL_DIR / "relationship_period_summary.csv",
        "simultaneity": FIB_DIR / "portfolio_fibonacci_simultaneity_summary.csv",
        "individual": FIB_DIR / "portfolio_fibonacci_forward_returns_by_period.csv",
        "cross": FIB_DIR / "portfolio_fibonacci_cross_asset_by_period.csv"
    }

    data = {}

    for name, path in files.items():

        if not path.exists():
            raise FileNotFoundError(path)

        data[name] = pd.read_csv(path)

    expected = {
        "relationships": 12,
        "simultaneity": 12,
        "individual": 864,
        "cross": 576
    }

    for name, count in expected.items():

        if len(data[name]) != count:
            raise ValueError(f"Unexpected row count in {name}: {len(data[name])}")

    for name in ["relationships", "simultaneity"]:
        if data[name].duplicated(["Portfolio", "Period"]).any():
            raise ValueError(f"Duplicate portfolio-period rows: {name}")

    print("Plot input validation: PASSED")

    return data


def period_values(data, portfolio, column):

    subset = data[(data["Portfolio"] == portfolio) & (data["Period"].isin(PERIODS))].set_index("Period")
    values = subset.reindex(PERIODS)[column].to_numpy(float)

    if len(subset) != 3 or not np.isfinite(values).all():
        raise ValueError(f"Invalid period values: {portfolio}/{column}")

    return values


def plot_correlation_vs_synchronization(data):

    relationships = data["relationships"]
    simultaneity = data["simultaneity"]

    measures = [
        (
            relationships,
            "MeanPairCorrelation",
            "Daily Return Correlation",
            "Mean pairwise correlation"
        ),
        (
            simultaneity,
            "PooledCoincidenceLift",
            "Fibonacci Coincidence Lift",
            "Observed / expected coincidences"
        ),
        (
            simultaneity,
            "MeanEventPhi",
            "Fibonacci Event Correlation",
            "Mean binary event phi"
        )
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    x = np.arange(len(PERIODS))

    for ax, (table, column, title, ylabel) in zip(axes, measures):

        for portfolio in PORTFOLIOS:
            values = period_values(table, portfolio, column)
            ax.plot(x, values, marker="o", linewidth=2, markersize=6, color=COLORS[portfolio], label=LABELS[portfolio])

        ax.set_xticks(x, PERIOD_LABELS)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    axes[0].set_ylim(0, 1)
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    axes[0].legend(fontsize=9)
    fig.suptitle("Ordinary Return Dependence vs Fibonacci-Event Synchronization", fontsize=14)
    path = FIG_DIR / "01_correlation_vs_fibonacci_synchronization.png"

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_same_day_responses(data):

    cross = data["cross"]
    same_day = cross[(cross["Period"].isin(PERIODS)) & (cross["HorizonDays"] == 0)].copy()

    if len(same_day) != 108:
        raise ValueError("Expected 108 same-day directed-pair rows")

    if (same_day["BaselineMeanAbsReturn"] <= 0).any():
        raise ValueError("Invalid absolute-return baseline")

    same_day["AbsMultiplier"] = same_day["EventMeanAbsReturn"] / same_day["BaselineMeanAbsReturn"]
    same_day["SignDifferencePP"] = 100 * same_day["SignAgreementDifference"]
    summary = same_day.groupby(["Portfolio", "Period"], as_index=False)[["AbsMultiplier", "SignDifferencePP"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    x = np.arange(len(PERIODS))
    width = 0.24

    for i, portfolio in enumerate(PORTFOLIOS):

        position = x + (i - 1) * width

        abs_values = period_values(summary, portfolio, "AbsMultiplier")
        sign_values = period_values(summary, portfolio, "SignDifferencePP")

        axes[0].bar(position, abs_values, width, color=COLORS[portfolio], label=LABELS[portfolio])
        axes[1].bar(position, sign_values, width, color=COLORS[portfolio], label=LABELS[portfolio])

    axes[0].axhline(1, color="black", linewidth=1, linestyle="--")
    axes[1].axhline(0, color="black", linewidth=1)

    axes[0].set_title("Target Absolute-Return Multiplier")
    axes[0].set_ylabel("Event-day / non-event absolute return")

    axes[1].set_title("Extra Same-Day Directional Agreement")
    axes[1].set_ylabel("Event − non-event (percentage points)")

    for ax in axes:
        ax.set_xticks(x, PERIOD_LABELS)
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend(fontsize=9)

    fig.suptitle("Same-Day Cross-Asset Responses to Fibonacci Events", fontsize=14)

    path = FIG_DIR / "02_same_day_cross_asset_comparison.png"

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def make_forward_matrix(table, level_filter=None):

    subset = table[(table["Period"].isin(PERIODS)) & (table["HorizonDays"].isin(HORIZONS))].copy()

    if level_filter is not None:
        subset = subset[subset["FibLevel"] == level_filter]

    subset["DifferencePP"] = 100 * subset["MeanReturnDifference"]
    rows = pd.MultiIndex.from_product([PORTFOLIOS, PERIODS], names=["Portfolio", "Period"])

    matrix = (
        subset
        .groupby(["Portfolio", "Period", "HorizonDays"])["DifferencePP"]
        .mean()
        .unstack("HorizonDays")
        .reindex(index=rows, columns=HORIZONS)
    )

    if (matrix.shape != (9, 3) or not np.isfinite(matrix.to_numpy()).all()):
        raise ValueError("Incomplete forward-return matrix")

    return matrix


def plot_forward_responses(data):

    individual = make_forward_matrix(data["individual"], level_filter="ALL")
    cross = make_forward_matrix(data["cross"])

    maximum = max(np.abs(individual.to_numpy()).max(), np.abs(cross.to_numpy()).max())
    scale = max(1.0, float(np.ceil(maximum)))
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)

    fig, axes = plt.subplots(1, 2, figsize=(15, 9), constrained_layout=True)

    titles = ["Individual Asset Forward Returns", "Cross-Asset Forward Returns"]

    row_labels = [
        f"{LABELS[p]} · {period.replace('2016-2017', '2016–17').replace('2018-2019', '2018–19')}"
        for p in PORTFOLIOS
        for period in PERIODS
    ]

    for ax, matrix, title in zip(axes, [individual, cross], titles):
        image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")

        ax.set_xticks(range(3), ["1 day", "5 days", "20 days"])
        ax.set_yticks(range(9), row_labels)

        ax.set_title(title)
        ax.set_xlabel("Forward-return horizon")

        for i in range(9):
            for j in range(3):
                value = matrix.iat[i, j]
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=10, color=("white" if abs(value) > 0.65 * scale else "black"))

        for boundary in [2.5, 5.5]:
            ax.axhline(boundary, color="black", linewidth=1.5)

    axes[1].tick_params(axis="y", labelleft=False)
    fig.colorbar(image, ax=axes, shrink=0.70, label="Event − non-event mean return (percentage points)")
    fig.suptitle("Fibonacci-Conditioned Forward Returns by Portfolio and Period", fontsize=14)

    path = FIG_DIR / "03_forward_return_comparison.png"

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def main():

    print("=" * 85)
    print("THREE-PORTFOLIO FIBONACCI COMPARATIVE FIGURES")
    print("=" * 85)

    data = load_results()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_correlation_vs_synchronization(data)
    plot_same_day_responses(data)
    plot_forward_responses(data)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()