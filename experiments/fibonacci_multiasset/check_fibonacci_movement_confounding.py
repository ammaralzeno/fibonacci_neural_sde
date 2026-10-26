from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "workspace" / "fibonacci_multiasset" / "data"
PRICE_FILE = ROOT / "workspace" / "portfolio_analysis" / "data" / "diversified_prices.csv"
STATE_FILE = DATA_DIR / "diversified_fibonacci_event_state.csv"
PREVIOUS_FILE = DATA_DIR / "diversified_fibonacci_cross_asset_summary.csv"
OUTPUT_FILE = DATA_DIR / "diversified_fibonacci_movement_adjusted.csv"


def load_data():

    prices = pd.read_csv(PRICE_FILE, parse_dates=["Date"]).set_index("Date").sort_index()
    prices.columns = prices.columns.astype(str)

    state = pd.read_csv(STATE_FILE, parse_dates=["Date"], dtype={"IssueId": str})
    previous = pd.read_csv(PREVIOUS_FILE, dtype={"SourceIssueId": str, "TargetIssueId": str})

    if prices.index.has_duplicates:
        raise ValueError("Duplicate price dates")

    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Invalid price data")

    if state.duplicated(["Date", "IssueId"]).any():
        raise ValueError("Duplicate event-state records")

    flag = state["InteriorFibEventStart"]
    if flag.dtype != bool:
        flag = flag.astype(str).str.lower().map({"true": True, "false": False})

        if flag.isna().any():
            raise ValueError("Invalid event-start values")

        state["InteriorFibEventStart"] = flag.astype(bool)

    dates = pd.DatetimeIndex(sorted(state["Date"].unique()), name="Date")
    ids = sorted(state["IssueId"].unique())

    expected = pd.MultiIndex.from_product([dates, ids], names=["Date", "IssueId"])
    actual = state.set_index(["Date", "IssueId"]).index.sort_values()

    if not actual.equals(expected):
        raise ValueError("Event-state panel is not balanced")

    if not dates.isin(prices.index).all():
        raise ValueError("Missing event dates in source prices")

    if not set(ids).issubset(prices.columns):
        raise ValueError("Missing asset prices")

    events = state.pivot(index="Date", columns="IssueId", values="InteriorFibEventStart").reindex(index=dates, columns=ids).astype(bool)
    names = state[["IssueId", "Asset"]].drop_duplicates().set_index("IssueId")["Asset"].to_dict()

    if len(names) != len(ids):
        raise ValueError("Inconsistent asset names")

    daily_returns = prices.pct_change(fill_method=None).reindex(dates)[ids]

    if daily_returns.isna().any().any():
        raise ValueError("Missing daily returns")

    return events, daily_returns, names, previous


def analyze_pair(source, target, events, returns, names):

    event_mask = events[source]

    source_returns = returns[source]
    target_returns = returns[target]

    sign_agreement = np.sign(source_returns) == np.sign(target_returns)

    # Original, unadjusted comparison.
    event_target_abs = target_returns.loc[event_mask].abs().mean()
    normal_target_abs = target_returns.loc[~event_mask].abs().mean()

    event_sign = sign_agreement.loc[event_mask].mean()
    normal_sign = sign_agreement.loc[~event_mask].mean()

    raw_abs_multiplier = event_target_abs / normal_target_abs
    raw_sign_difference = 100 * (event_sign - normal_sign)

    source_movement_multiplier = source_returns.loc[event_mask].abs().mean() / source_returns.loc[~event_mask].abs().mean()
    magnitude_bucket = pd.qcut(source_returns.abs(), q=5, labels=False, duplicates="drop")

    strata = pd.DataFrame(
        {
            "MagnitudeBucket": magnitude_bucket,
            "SourceDirection": np.sign(source_returns).astype(int),
        },
        index=returns.index
    )

    matched_abs_sum = 0.0
    matched_sign_sum = 0.0
    matched_event_count = 0
    matched_strata = 0

    for _, group in strata.groupby(["MagnitudeBucket", "SourceDirection"]):

        group_dates = group.index
        group_event_mask = event_mask.loc[group_dates]

        event_count = int(group_event_mask.sum())
        baseline_count = int((~group_event_mask).sum())

        if event_count == 0: 
            continue

        if baseline_count == 0:
            raise ValueError(f"No matched baseline for {names[source]} -> {names[target]}")

        baseline_dates = group_dates[~group_event_mask.to_numpy()]
        baseline_abs = target_returns.loc[baseline_dates].abs().mean()
        baseline_sign = sign_agreement.loc[baseline_dates].mean()

        matched_abs_sum += event_count * baseline_abs
        matched_sign_sum += event_count * baseline_sign

        matched_event_count += event_count
        matched_strata += 1

    total_events = int(event_mask.sum())

    if matched_event_count != total_events:
        raise ValueError("Not all event days were matched")

    adjusted_baseline_abs = matched_abs_sum / matched_event_count
    adjusted_baseline_sign = matched_sign_sum / matched_event_count

    if adjusted_baseline_abs <= 0:
        raise ValueError("Invalid adjusted absolute-return baseline")

    adjusted_abs_multiplier = event_target_abs / adjusted_baseline_abs
    adjusted_sign_difference = 100 * (event_sign - adjusted_baseline_sign)

    return {
        "SourceIssueId": source,
        "SourceAsset": names[source],
        "TargetIssueId": target,
        "TargetAsset": names[target],
        "EventN": total_events,
        "MatchedStrata": matched_strata,
        "SourceAbsMoveMultiplier": source_movement_multiplier,
        "TargetAbsMultiplierRaw": raw_abs_multiplier,
        "TargetAbsMultiplierAdjusted": adjusted_abs_multiplier,
        "SignAgreementDiffPPRaw": raw_sign_difference,
        "SignAgreementDiffPPAdjusted": adjusted_sign_difference
    }


def validate_against_previous(results, previous):

    original = previous[previous["HorizonDays"] == 0]

    if len(original) != len(results):
        raise ValueError("Previous summary has the wrong number of pairs")

    for row in results.itertuples(index=False):

        match = original[(original["SourceIssueId"] == row.SourceIssueId) & (original["TargetIssueId"] == row.TargetIssueId)]

        if len(match) != 1:
            raise ValueError("Missing previous pairwise result")

        old = match.iloc[0]
        old_abs_multiplier = old["EventMeanAbsReturn"] / old["BaselineMeanAbsReturn"]
        old_sign_difference = 100 * old["SignAgreementDifference"]

        if not np.isclose(old_abs_multiplier, row.TargetAbsMultiplierRaw, atol=1e-8):
            raise ValueError("Absolute-return comparison mismatch")

        if not np.isclose(old_sign_difference, row.SignAgreementDiffPPRaw, atol=1e-8):
            raise ValueError("Sign-agreement comparison mismatch")

        if old["EventN"] != row.EventN:
            raise ValueError("Event-count mismatch")


def main():

    print("=" * 90)
    print("FIBONACCI SAME-DAY MOVEMENT CONFOUNDING CHECK")
    print("=" * 90)

    events, returns, names, previous = load_data()

    print(f"Common dates : {len(events):,}")
    print(f"Assets       : {len(events.columns)}")
    print(f"Event starts : {int(events.to_numpy().sum()):,}")

    results = pd.DataFrame([analyze_pair(source, target, events, returns, names) for source, target in permutations(events.columns, 2)])
    validate_against_previous(results, previous)

    print("\nPrevious result reproduction: PASSED")
    columns = [
        "SourceAsset",
        "TargetAsset",
        "EventN",
        "SourceAbsMoveMultiplier",
        "TargetAbsMultiplierRaw",
        "TargetAbsMultiplierAdjusted",
        "SignAgreementDiffPPRaw",
        "SignAgreementDiffPPAdjusted"
    ]

    print("\nPAIRWISE RESULTS")
    print(results[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    measures = [
        "SourceAbsMoveMultiplier",
        "TargetAbsMultiplierRaw",
        "TargetAbsMultiplierAdjusted",
        "SignAgreementDiffPPRaw",
        "SignAgreementDiffPPAdjusted"
    ]

    print(f"\nMEAN ACROSS {len(results)} DIRECTED PAIRS")
    for measure in measures:
        print(f"{measure:32s}: {results[measure].mean():.3f}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved:\n{OUTPUT_FILE}")
    print("\nMOVEMENT CONFOUNDING CHECK COMPLETE")


if __name__ == "__main__":
    main()