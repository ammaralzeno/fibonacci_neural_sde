from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "workspace" / "fibonacci_multiasset" / "data"
FIG_DIR = REPO_ROOT / "workspace" / "fibonacci_multiasset" / "figures"

ASSETS = ["Energy", "Health Care", "Financials", "Information Technology"]
HORIZONS = [1, 5, 20]

def load_results():

    files = {
        "individual": "diversified_fibonacci_individual_asset_summary.csv",
        "pairwise": "diversified_fibonacci_pairwise_simultaneity.csv",
        "cross": "diversified_fibonacci_cross_asset_summary.csv",
        "detail": "diversified_fibonacci_cross_asset_event_responses.csv"
    }

    results = {}

    for key, filename in files.items():
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(path)

        results[key] = pd.read_csv(path)

    return results


def validate_results(results):

    individual = results["individual"]
    pairwise = results["pairwise"]
    cross = results["cross"]
    detail = results["detail"]

    pooled = individual[individual["FibLevel"] == "ALL"]

    if len(pooled) != 12:
        raise ValueError("Expected 12 pooled individual-asset results")

    if len(pairwise) != 6:
        raise ValueError("Expected 6 unordered asset pairs")

    if len(cross) != 48:
        raise ValueError("Expected 48 directed cross-asset summary rows")

    same_day = cross[cross["HorizonDays"] == 0]
    if len(same_day) != 12:
        raise ValueError("Expected 12 same-day directed pairs")

    expected_detail = 3 * int(same_day.groupby("SourceAsset")["EventN"].first().sum())
    if len(detail) != expected_detail:
        raise ValueError(f"Expected {expected_detail} detailed event-response rows, found {len(detail)}")

    if individual["MeanReturnDifference"].isna().any():
        raise ValueError("Missing individual return differences")

    if cross["MeanReturnDifference"].isna().any():
        raise ValueError("Missing cross-asset return differences")

    print("Result validation passed")
    print(f"Individual summary rows : {len(individual)}")
    print(f"Pairwise summary rows   : {len(pairwise)}")
    print(f"Cross-asset summary rows: {len(cross)}")
    print(f"Detailed response rows  : {len(detail)}")


def make_matrix(data, value_column, symmetric=False):

    matrix = pd.DataFrame(np.nan, index=ASSETS, columns=ASSETS)

    for row in data.to_dict("records"):

        if symmetric:
            source = row["Asset_1"]
            target = row["Asset_2"]
        else:
            source = row["SourceAsset"]
            target = row["TargetAsset"]

        matrix.loc[source, target] = row[value_column]

        if symmetric:
            matrix.loc[target, source] = row[value_column]

    return matrix.astype(float)


def draw_heatmap(ax, matrix, title, norm, annotation_format):

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#EFEFEF")

    image = ax.imshow(matrix.to_numpy(), cmap=cmap, norm=norm, aspect="equal")

    ax.set_xticks(range(len(ASSETS)), ASSETS, rotation=35, ha="right")
    ax.set_yticks(range(len(ASSETS)), ASSETS)

    ax.set_xlabel("Target asset")
    ax.set_ylabel("Source asset")
    ax.set_title(title)

    for i in range(len(ASSETS)):
        for j in range(len(ASSETS)):

            value = matrix.iloc[i, j]

            if np.isfinite(value):
                ax.text(j, i, annotation_format(value), ha="center", va="center", fontsize=9)

    return image


def plot_individual_effects(individual):

    pooled = individual[individual["FibLevel"] == "ALL"].copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = np.arange(len(ASSETS))
    width = 0.24

    for i, horizon in enumerate(HORIZONS):

        subset = pooled[pooled["HorizonDays"] == horizon].set_index("Asset").reindex(ASSETS)
        values = 100 * subset["MeanReturnDifference"].to_numpy()
        ax.bar(x + (i - 1) * width, values, width, label=f"{horizon}-day")

    ax.axhline(0, color="black", linewidth=0.9)

    ax.set_xticks(x, ASSETS)
    ax.set_ylabel("Event − non-event mean return (percentage points)")
    ax.set_title("Individual-Asset Forward Returns After Fibonacci Entry Events")

    ax.legend(title="Forward horizon")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    
    path = FIG_DIR / "01_individual_fibonacci_effects.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_simultaneity(pairwise):

    lift = make_matrix(pairwise, "CoincidenceLift", symmetric=True)
    counts = make_matrix(pairwise, "SimultaneousEvents", symmetric=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.0)
    image = draw_heatmap(ax, lift, "Same-Day Fibonacci Event Coincidence Lift", norm, lambda value: f"{value:.2f}×")

    for i in range(len(ASSETS)):
        for j in range(len(ASSETS)):

            if i != j:
                count = counts.iloc[i, j]
                ax.text(j, i + 0.25, f"n={int(count)}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Observed / independence-expected coincidences", shrink=0.75)
    fig.tight_layout()

    path = FIG_DIR / "02_fibonacci_simultaneity_lift.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_same_day_effects(cross):

    same_day = cross[cross["HorizonDays"] == 0].copy()
    same_day["AbsReturnMultiplier"] = same_day["EventMeanAbsReturn"] / same_day["BaselineMeanAbsReturn"]
    same_day["SignAgreementDifferencePP"] = 100 * same_day["SignAgreementDifference"]

    abs_matrix = make_matrix(same_day, "AbsReturnMultiplier")
    sign_matrix = make_matrix(same_day, "SignAgreementDifferencePP")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    image_1 = draw_heatmap(axes[0], abs_matrix, "Target Absolute-Return Multiplier", TwoSlopeNorm(vmin=0.8, vcenter=1.0, vmax=1.8), lambda value: f"{value:.2f}×")
    image_2 = draw_heatmap(axes[1], sign_matrix, "Change in Same-Day Directional Agreement", TwoSlopeNorm(vmin=-5, vcenter=0, vmax=20), lambda value: f"{value:+.1f}")

    fig.colorbar(image_1, ax=axes[0], shrink=0.65, label="Event-day / baseline absolute return")
    fig.colorbar(image_2, ax=axes[1], shrink=0.65, label="Percentage-point difference")

    fig.suptitle("Cross-Asset Co-Movement on Source Fibonacci Event Days", fontsize=14)
    fig.tight_layout()

    path = FIG_DIR / "03_fibonacci_same_day_cross_asset.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def plot_forward_effects(cross):

    forward = cross[cross["HorizonDays"].isin(HORIZONS)].copy()
    forward["MeanDifferencePP"] = 100 * forward["MeanReturnDifference"]

    max_abs = float(forward["MeanDifferencePP"].abs().max())
    scale = max(0.5, np.ceil(max_abs * 2) / 2)
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, horizon in zip(axes, HORIZONS):

        subset = forward[forward["HorizonDays"] == horizon]
        matrix = make_matrix(subset, "MeanDifferencePP")
        image = draw_heatmap(ax, matrix, f"{horizon}-Day Forward Return", norm, lambda value: f"{value:+.2f}")

    fig.colorbar(image, ax=axes, shrink=0.65, label="Event − non-event return (percentage points)")
    fig.suptitle("Target Forward Returns Conditional on Source Fibonacci Events", fontsize=14)

    path = FIG_DIR / "04_fibonacci_cross_asset_forward_returns.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


def main():

    print("=" * 80)
    print("FIBONACCI MULTI-ASSET RESULTS VISUALIZATION")
    print("=" * 80)

    results = load_results()
    validate_results(results)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_individual_effects(results["individual"])
    plot_simultaneity(results["pairwise"])
    plot_same_day_effects(results["cross"])
    plot_forward_effects(results["cross"])

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()