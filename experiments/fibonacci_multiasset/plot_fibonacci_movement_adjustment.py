from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "workspace" / "fibonacci_multiasset" / "data" / "diversified_fibonacci_movement_adjusted.csv"
OUTPUT_FILE = ROOT / "workspace" / "fibonacci_multiasset" / "figures" / "05_fibonacci_movement_adjustment.png"

SHORT_NAMES = {
    "Energy": "Energy",
    "Health Care": "Health Care",
    "Financials": "Financials",
    "Information Technology": "IT"
}

def load_results():

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    data = pd.read_csv(DATA_FILE, dtype={"SourceIssueId": str, "TargetIssueId": str})
    required = [
        "SourceAsset",
        "TargetAsset",
        "TargetAbsMultiplierRaw",
        "TargetAbsMultiplierAdjusted",
        "SignAgreementDiffPPRaw",
        "SignAgreementDiffPPAdjusted"
    ]

    if data[required].isna().any().any():
        raise ValueError("Missing movement-adjustment results")

    if len(data) != 12:
        raise ValueError("Expected 12 directed asset pairs")

    if data.duplicated(["SourceIssueId", "TargetIssueId"]).any():
        raise ValueError("Duplicate directed asset pairs")

    data = data.sort_values(["SourceAsset", "TargetAsset"]).reset_index(drop=True)
    data["Pair"] = data["SourceAsset"].map(SHORT_NAMES) + " → " + data["TargetAsset"].map(SHORT_NAMES)

    return data


def draw_comparison(ax, data, raw_column, adjusted_column, title, xlabel, reference):

    y = np.arange(len(data))

    raw = data[raw_column].to_numpy()
    adjusted = data[adjusted_column].to_numpy()

    ax.hlines(y, np.minimum(raw, adjusted), np.maximum(raw, adjusted), color="0.65", linewidth=1.5, zorder=1)
    ax.scatter(raw, y, s=55, color="#377eb8", label="Unadjusted", zorder=3)
    ax.scatter(adjusted, y, s=55, marker="s", color="#e69f00", label="Movement-adjusted", zorder=3)
    ax.axvline(reference, color="black", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(data["Pair"])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.2)


def main():

    data = load_results()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    draw_comparison(axes[0], data, "TargetAbsMultiplierRaw", "TargetAbsMultiplierAdjusted", "Target Absolute-Return Multiplier", "Event-day / baseline absolute return", 1.0)
    draw_comparison(axes[1], data, "SignAgreementDiffPPRaw", "SignAgreementDiffPPAdjusted", "Same-Day Directional Agreement", "Event − baseline (percentage points)", 0.0)
    axes[0].invert_yaxis()

    axes[0].legend(loc="lower right", frameon=True)
    fig.suptitle("Fibonacci Event Co-Movement: Before and After Source-Movement Adjustment", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")

    plt.close(fig)

    print(f"Directed pairs: {len(data)}")
    print(f"Mean absolute-return multiplier: {data['TargetAbsMultiplierRaw'].mean():.3f} -> {data['TargetAbsMultiplierAdjusted'].mean():.3f}")
    print(f"Mean directional agreement difference (pp): {data['SignAgreementDiffPPRaw'].mean():.3f} -> {data['SignAgreementDiffPPAdjusted'].mean():.3f}")

    print(f"\nSaved:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()