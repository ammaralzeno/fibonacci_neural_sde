from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "workspace" / "fibonacci_multiasset" / "data"
EVENT_STATE_FILE = DATA_DIR / "diversified_fibonacci_event_state.csv"
EVENT_MATRIX_FILE = DATA_DIR / "diversified_fibonacci_event_matrix.csv"
SIMULTANEOUS_DATES_FILE = DATA_DIR / "diversified_fibonacci_simultaneous_dates.csv"
PAIRWISE_FILE = DATA_DIR / "diversified_fibonacci_pairwise_simultaneity.csv"
SUMMARY_FILE = DATA_DIR / "diversified_fibonacci_simultaneity_summary.csv"


def load_event_state():

    if not EVENT_STATE_FILE.exists():
        raise FileNotFoundError(f"Event-state file not found:\n{EVENT_STATE_FILE}")

    data = pd.read_csv(EVENT_STATE_FILE, parse_dates=["Date"], dtype={"IssueId": str, "InteriorFibLabel": str})
    data = data.sort_values(["Date", "IssueId"]).reset_index(drop=True)

    return data


def validate_event_state(data):

    required_columns = {"Date", "IssueId", "Asset", "InteriorFibEventStart", "InteriorFibLabel", "InteriorFibRatio"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    duplicates = data.duplicated(subset=["Date", "IssueId"]).sum()
    if duplicates > 0:
        raise ValueError("Duplicate IssueId/date rows found")

    asset_date_counts = data.groupby("IssueId")["Date"].nunique()
    if asset_date_counts.nunique() != 1:
        raise ValueError("Assets do not have the same number of dates")

    common_dates = data["Date"].nunique()
    expected_rows = common_dates * data["IssueId"].nunique()
    if len(data) != expected_rows:
        raise ValueError("Event-state panel is not fully balanced")

    print("\n" + "=" * 100)
    print("INPUT VALIDATION")
    print("=" * 100)

    print(f"Assets       :  {data['IssueId'].nunique()}")
    print(f"Common dates :  {common_dates:,}")
    print(f"Total rows   :  {len(data):,}")
    print(f"First date   :  {data['Date'].min().date()}")
    print(f"Last date    :  {data['Date'].max().date()}")


def build_event_matrix(data):

    matrix = data.pivot(index="Date", columns="IssueId", values="InteriorFibEventStart").sort_index()
    matrix = matrix.astype(bool)
    matrix.columns = matrix.columns.astype(str)

    return matrix


def build_level_matrix(data):

    event_rows = data[data["InteriorFibEventStart"]]
    level_matrix = event_rows.pivot(index="Date", columns="IssueId", values="InteriorFibRatio")

    all_dates = data["Date"].drop_duplicates().sort_values()
    all_issue_ids = sorted(data["IssueId"].unique())

    level_matrix = level_matrix.reindex(index=all_dates, columns=all_issue_ids)

    return level_matrix


def get_asset_names(data):

    mapping = data[["IssueId", "Asset"]].drop_duplicates().set_index("IssueId")["Asset"].to_dict()
    return mapping


def build_simultaneous_dates(event_matrix, level_matrix, asset_names):

    rows = []

    number_of_events = event_matrix.sum(axis=1)
    simultaneous_dates = number_of_events[number_of_events >= 2].index

    for date in simultaneous_dates:
        active_ids = event_matrix.columns[event_matrix.loc[date].to_numpy()].tolist()
        active_assets = [asset_names[issue_id] for issue_id in active_ids]

        levels = [level_matrix.loc[date, issue_id] for issue_id in active_ids]
        rows.append(
            {
                "Date": date,
                "NumberOfEvents": len(active_ids),
                "IssueIds": ";".join(active_ids),
                "Assets": ";".join(active_assets),
                "FibRatios": ";".join(f"{float(level):.3f}" for level in levels),
                "AllSameFibLevel": np.allclose(levels, levels[0])
            }
        )

    return pd.DataFrame(rows)


def build_pairwise_summary(event_matrix, level_matrix, asset_names):

    rows = []
    n_dates = len(event_matrix)

    for issue_1, issue_2 in combinations(event_matrix.columns, 2):

        event_1 = event_matrix[issue_1].to_numpy(dtype=bool)
        event_2 = event_matrix[issue_2].to_numpy(dtype=bool)

        count_1 = int(event_1.sum())
        count_2 = int(event_2.sum())

        joint = event_1 & event_2
        joint_count = int(joint.sum())

        rate_1 = count_1 / n_dates
        rate_2 = count_2 / n_dates

        observed_joint_rate = joint_count / n_dates
        expected_joint_rate = rate_1 * rate_2
        expected_joint_count = expected_joint_rate * n_dates

        if expected_joint_count > 0:
            coincidence_lift = joint_count / expected_joint_count
        else:
            coincidence_lift = np.nan

        union_count = int((event_1 | event_2).sum())
        if union_count > 0:
            jaccard = joint_count / union_count
        else:
            jaccard = np.nan

        event_1_float = event_1.astype(float)
        event_2_float = event_2.astype(float)

        if (event_1_float.std() > 0 and event_2_float.std() > 0):
            phi = float(np.corrcoef(event_1_float, event_2_float)[0, 1])
        else:
            phi = np.nan

        joint_dates = event_matrix.index[joint]
        same_level_count = 0

        for date in joint_dates:
            level_1 = level_matrix.loc[date, issue_1]
            level_2 = level_matrix.loc[date, issue_2]

            if np.isclose(level_1, level_2):
                same_level_count += 1

        if joint_count > 0:
            same_level_share = same_level_count / joint_count

        else:
            same_level_share = np.nan

        rows.append(
            {
                "IssueId_1": issue_1,
                "Asset_1": asset_names[issue_1],
                "IssueId_2": issue_2,
                "Asset_2": asset_names[issue_2],
                "TotalDates": n_dates,
                "Events_1": count_1,
                "Events_2": count_2,
                "SimultaneousEvents": joint_count,
                "ObservedSimultaneousRate": observed_joint_rate,
                "ExpectedSimultaneousRate": expected_joint_rate,
                "ExpectedSimultaneousEvents": expected_joint_count,
                "CoincidenceLift": coincidence_lift,
                "BinaryEventCorrelation": phi,
                "JaccardSimilarity": jaccard,
                "SameLevelSimultaneousEvents": same_level_count,
                "SameLevelShare": same_level_share
            }
        )

    return pd.DataFrame(rows)


def build_overall_summary(event_matrix):

    number_of_events = event_matrix.sum(axis=1)
    rows = []

    n_dates = len(event_matrix)
    n_assets = event_matrix.shape[1]

    for count in range(n_assets + 1):
        days = int((number_of_events == count).sum())
        rows.append(
            {
                "NumberOfSimultaneousEvents": count,
                "NumberOfDates": days,
                "FractionOfDates": days / n_dates
            }
        )

    return pd.DataFrame(rows)


def print_overall_summary(summary):

    display = summary.copy()
    display["FractionOfDates"] *= 100

    print("\n" + "=" * 100)
    print("NUMBER OF FIBONACCI EVENT STARTS PER DATE")
    print("=" * 100)

    print(display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    simultaneous_days = int(summary.loc[summary["NumberOfSimultaneousEvents"] >= 2, "NumberOfDates"].sum())
    print(f"\nDates with >=2 simultaneous events: {simultaneous_days}")


def print_pairwise_summary(pairwise):

    display = pairwise.copy()
    display["SameLevelShare"] *= 100

    columns = [
        "Asset_1",
        "Asset_2",
        "Events_1",
        "Events_2",
        "SimultaneousEvents",
        "ExpectedSimultaneousEvents",
        "CoincidenceLift",
        "BinaryEventCorrelation",
        "SameLevelSimultaneousEvents",
        "SameLevelShare"
    ]

    print("\n" + "=" * 130)
    print("PAIRWISE FIBONACCI EVENT SIMULTANEITY")
    print("=" * 130)

    print(display[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nCoincidenceLift = observed simultaneous events / expected simultaneous events under independence")


def main():

    print("=" * 100)
    print("MULTI-ASSET FIBONACCI EVENT SIMULTANEITY")
    print("=" * 100)

    data = load_event_state()
    validate_event_state(data)

    asset_names = get_asset_names(data)
    event_matrix = build_event_matrix(data)
    level_matrix = build_level_matrix(data)
    simultaneous_dates = build_simultaneous_dates(event_matrix, level_matrix, asset_names)

    pairwise = build_pairwise_summary(event_matrix, level_matrix, asset_names)
    summary = build_overall_summary(event_matrix)
    print_overall_summary(summary)
    print_pairwise_summary(pairwise)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    event_matrix.to_csv(EVENT_MATRIX_FILE, index=True)
    simultaneous_dates.to_csv(SIMULTANEOUS_DATES_FILE, index=False)
    pairwise.to_csv(PAIRWISE_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nSaved:")
    print(f"  Event matrix:\n  {EVENT_MATRIX_FILE}")
    print(f"  Simultaneous dates:\n  {SIMULTANEOUS_DATES_FILE}")
    print(f"  Pairwise summary:\n  {PAIRWISE_FILE}")
    print(f"  Overall summary:\n  {SUMMARY_FILE}")

    print("\n" + "=" * 100)
    print("SIMULTANEITY ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()