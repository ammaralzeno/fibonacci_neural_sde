from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "workspace" / "multiasset_extensions" / "fibonacci"
EVENT_FILE = DATA_DIR / "portfolio_fibonacci_event_state.csv"
SUMMARY_FILE = DATA_DIR / "portfolio_fibonacci_simultaneity_summary.csv"
PAIR_FILE = DATA_DIR / "portfolio_fibonacci_simultaneity_pairwise.csv"

PORTFOLIOS = [
    "diversified",
    "concentrated",
    "ultra_concentrated"
]

PERIODS = {
    "2016-2017": ("2016-01-01", "2017-12-31"),
    "2018-2019": ("2018-01-01", "2019-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "full_2015-2020": ("2015-01-01", "2020-12-31")
}

def load_events():

    if not EVENT_FILE.exists():
        raise FileNotFoundError(EVENT_FILE)

    data = pd.read_csv(EVENT_FILE, parse_dates=["Date"], dtype={"IssueId": str})
    
    required = [
        "Date",
        "Portfolio",
        "IssueId",
        "InteriorFibHit",
        "InteriorFibEventStart",
        "InteriorFibRatio"
    ]

    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if data.duplicated(["Portfolio", "IssueId", "Date"]).any():
        raise ValueError("Duplicate portfolio/asset/date records")

    for column in ["InteriorFibHit", "InteriorFibEventStart"]:

        parsed = data[column].astype(str).str.lower().map({"true": True, "false": False})

        if parsed.isna().any():
            raise ValueError(f"Invalid boolean column: {column}")

        data[column] = parsed.astype(bool)

    data["InteriorFibRatio"] = pd.to_numeric(data["InteriorFibRatio"], errors="coerce")

    if (data["InteriorFibEventStart"] & ~data["InteriorFibHit"]).any():
        raise ValueError("An event start occurs outside a level zone")

    if data.loc[data["InteriorFibEventStart"], "InteriorFibRatio"].isna().any():
        raise ValueError("Event start has no Fibonacci ratio")

    if len(data) != 15132:
        raise ValueError("Unexpected total event-state rows")

    dates = pd.DatetimeIndex(sorted(data["Date"].unique()))

    if len(dates) != 1261:
        raise ValueError("Unexpected Fibonacci calendar length")

    for portfolio in PORTFOLIOS:

        subset = data[data["Portfolio"] == portfolio]
        ids = sorted(subset["IssueId"].unique())

        if len(ids) != 4:
            raise ValueError(f"Expected four assets: {portfolio}")

        expected = pd.MultiIndex.from_product([dates, ids], names=["Date", "IssueId"])
        actual = subset.set_index(["Date", "IssueId"]).index.sort_values()

        if not actual.equals(expected):
            raise ValueError(f"Unbalanced Fibonacci panel: {portfolio}")

    print("Event-state validation: PASSED")

    return data


def analyze_period(portfolio, period, subset):

    event_matrix = subset.pivot(index="Date", columns="IssueId", values="InteriorFibEventStart").sort_index()
    level_matrix = subset.pivot(index="Date", columns="IssueId", values="InteriorFibRatio").reindex(index=event_matrix.index, columns=event_matrix.columns)

    if event_matrix.shape[1] != 4:
        raise ValueError("Expected four portfolio assets")

    if event_matrix.isna().any().any():
        raise ValueError("Missing event flags")

    event_matrix = event_matrix.astype(bool)

    n = len(event_matrix)

    if n < 30:
        raise ValueError(f"Insufficient dates: {portfolio}/{period}")

    event_count_by_date = event_matrix.sum(axis=1)

    pair_rows = []

    for a, b in combinations(event_matrix.columns, 2):

        a_events = event_matrix[a]
        b_events = event_matrix[b]

        n_a = int(a_events.sum())
        n_b = int(b_events.sum())

        simultaneous = a_events & b_events
        observed = int(simultaneous.sum())

        expected = n_a * n_b / n

        lift = observed / expected if expected > 0 else np.nan
        union_count = int((a_events | b_events).sum())

        jaccard = observed / union_count if union_count > 0 else np.nan

        denominator = np.sqrt(n_a * (n - n_a) * n_b * (n - n_b))
        phi = (n * observed - n_a * n_b) / denominator if denominator > 0 else np.nan

        levels_a = level_matrix.loc[simultaneous, a].to_numpy(dtype=float)
        levels_b = level_matrix.loc[simultaneous, b].to_numpy(dtype=float)

        if not (np.isfinite(levels_a).all() and np.isfinite(levels_b).all()):
            raise ValueError("Missing level at simultaneous event")

        same_level = int(np.isclose(levels_a, levels_b, rtol=0, atol=1e-10).sum())

        pair_rows.append({
            "Portfolio": portfolio,
            "Period": period,
            "IssueId1": a,
            "IssueId2": b,
            "N_Dates": n,
            "Events1": n_a,
            "Events2": n_b,
            "ObservedCoincidences": observed,
            "ExpectedCoincidences": expected,
            "CoincidenceLift": lift,
            "EventPhi": phi,
            "EventJaccard": jaccard,
            "SameLevelCoincidences": same_level,
            "SameLevelFraction": same_level / observed if observed > 0 else np.nan,
        })

    pairs = pd.DataFrame(pair_rows)

    if len(pairs) != 6:
        raise ValueError("Expected six unordered pairs")

    daily_counts = event_count_by_date.to_numpy()

    pair_incidences = int(np.sum(daily_counts * (daily_counts - 1) // 2))
    observed_total = int(pairs["ObservedCoincidences"].sum())

    if observed_total != pair_incidences:
        raise ValueError("Pairwise coincidence counting mismatch")

    total_expected = float(pairs["ExpectedCoincidences"].sum())

    summary = {
        "Portfolio": portfolio,
        "Period": period,
        "N_Dates": n,
        "EventStarts": int(event_matrix.to_numpy().sum()),
        "EventRatePct": 100 * event_matrix.to_numpy().mean(),
        "DatesWith2PlusEvents": int((event_count_by_date >= 2).sum()),
        "DatesWith3PlusEvents": int((event_count_by_date >= 3).sum()),
        "DatesWith4Events": int((event_count_by_date == 4).sum()),
        "MultiEventDateRatePct": 100 * (event_count_by_date >= 2).mean(),
        "ObservedPairCoincidences": observed_total,
        "ExpectedPairCoincidences": total_expected,
        "PooledCoincidenceLift": observed_total / total_expected if total_expected > 0 else np.nan,
        "MeanPairwiseLift": float(pairs["CoincidenceLift"].mean()),
        "MeanEventPhi": float(pairs["EventPhi"].mean()),
        "SameLevelCoincidences": int(pairs["SameLevelCoincidences"].sum()),
    }

    return summary, pairs


def validate_original(summary, pairs):

    original = summary[(summary["Portfolio"] == "diversified") & (summary["Period"] == "full_2015-2020")]

    if len(original) != 1:
        raise ValueError("Missing original diversified summary")

    row = original.iloc[0]

    if (
        row["N_Dates"] != 1261
        or row["EventStarts"] != 530
        or row["DatesWith2PlusEvents"] != 82
        or row["DatesWith3PlusEvents"] != 10
        or row["DatesWith4Events"] != 0
    ):
        raise ValueError("Original diversified event totals were not reproduced")

    original_pairs = pairs[(pairs["Portfolio"] == "diversified") & (pairs["Period"] == "full_2015-2020")]
    observed_counts = sorted(original_pairs["ObservedCoincidences"].tolist())
    expected_counts = [12, 13, 16, 20, 20, 21]

    if observed_counts != expected_counts:
        raise ValueError("Original diversified pairwise coincidences differ")

    print("Original Fibonacci simultaneity reproduced: PASSED")


def main():

    print("=" * 90)
    print("FIBONACCI SIMULTANEITY BY PORTFOLIO AND PERIOD")
    print("=" * 90)

    data = load_events()

    summaries = []
    pair_tables = []

    for portfolio in PORTFOLIOS:

        portfolio_data = data[data["Portfolio"] == portfolio]

        for period, (start, end) in PERIODS.items():

            subset = portfolio_data[portfolio_data["Date"].between(start, end)]
            summary, pairs = analyze_period(portfolio, period, subset)
            summaries.append(summary)
            pair_tables.append(pairs)

    summary_df = pd.DataFrame(summaries)
    pair_df = pd.concat(pair_tables, ignore_index=True)
    validate_original(summary_df, pair_df)

    print("\nSIMULTANEITY SUMMARY")
    print(summary_df[
        [
            "Portfolio",
            "Period",
            "N_Dates",
            "EventStarts",
            "EventRatePct",
            "MultiEventDateRatePct",
            "PooledCoincidenceLift",
            "MeanEventPhi",
            "SameLevelCoincidences"
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n2020 PAIRWISE COINCIDENCE LIFTS")
    print(pair_df[pair_df["Period"] == "2020"][
        [
            "Portfolio",
            "IssueId1",
            "IssueId2",
            "ObservedCoincidences",
            "ExpectedCoincidences",
            "CoincidenceLift",
            "EventPhi"
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    pair_df.to_csv(PAIR_FILE, index=False)

    print("\nSaved:")
    print(SUMMARY_FILE)
    print(PAIR_FILE)

    print("\nSIMULTANEITY ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()