from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

PORTFOLIO_DATA_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "data"
FIB_DATA_DIR = REPO_ROOT / "workspace" / "fibonacci_multiasset" / "data"
PRICE_FILE = PORTFOLIO_DATA_DIR / "diversified_prices.csv"
EVENT_STATE_FILE = FIB_DATA_DIR / "diversified_fibonacci_event_state.csv"
DETAIL_FILE = FIB_DATA_DIR / "diversified_fibonacci_cross_asset_event_responses.csv"
SUMMARY_FILE = FIB_DATA_DIR / "diversified_fibonacci_cross_asset_summary.csv"

FORWARD_HORIZONS = [1, 5, 20]

def load_prices():

    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Price file not found:\n{PRICE_FILE}")

    prices = pd.read_csv(PRICE_FILE, parse_dates=["Date"])
    prices = prices.set_index("Date").sort_index()
    prices.columns = prices.columns.astype(str)

    for column in prices.columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    return prices


def load_event_state():

    if not EVENT_STATE_FILE.exists():
        raise FileNotFoundError(f"Event-state file not found:\n{EVENT_STATE_FILE}")

    data = pd.read_csv(EVENT_STATE_FILE, parse_dates=["Date"], dtype={"IssueId": str, "InteriorFibLabel": str})
    data = data.sort_values(["Date", "IssueId"]).reset_index(drop=True)

    # Explicit safety handling in case pandas loads bool col as str
    if data["InteriorFibEventStart"].dtype != bool:
        converted = data["InteriorFibEventStart"].astype(str).str.lower().map({"true": True, "false": False})

        if converted.isna().any():
            raise ValueError("Could not parse InteriorFibEventStart as boolean")
        
        data["InteriorFibEventStart"] = converted.astype(bool)

    return data


def validate_inputs(prices, event_state):

    print("\n" + "=" * 100)
    print("INPUT VALIDATION")
    print("=" * 100)

    if prices.index.duplicated().any():
        raise ValueError("Duplicate dates found in price matrix")

    if prices.isna().any().any():
        raise ValueError("Missing prices found")

    if (prices <= 0).any().any():
        raise ValueError("Non-positive prices found")

    duplicates = event_state.duplicated(subset=["Date", "IssueId"]).sum()
    if duplicates > 0:
        raise ValueError("Duplicate IssueId/date rows in event state")

    issue_ids = set(event_state["IssueId"])
    missing_assets = issue_ids - set(prices.columns)
    if missing_assets:
        raise ValueError(f"Missing price assets: {sorted(missing_assets)}")

    event_dates = set(event_state["Date"])
    missing_dates = event_dates - set(prices.index)
    if missing_dates:
        raise ValueError("Event-state dates are missing from the price matrix")

    common_dates = event_state["Date"].nunique()
    n_assets = event_state["IssueId"].nunique()
    expected_rows = common_dates * n_assets
    if len(event_state) != expected_rows:
        raise ValueError("Event-state panel is not balanced")

    total_events = int(event_state["InteriorFibEventStart"].sum())
    print(f"Assets             : {n_assets}")
    print(f"Common dates       : {common_dates:,}")
    print(f"Total event starts : {total_events:,}")
    print(f"Analysis period    : {event_state['Date'].min().date()} to {event_state['Date'].max().date()}")


def calculate_return_matrices(prices):

    daily_returns = prices.pct_change(fill_method=None)
    forward_returns = {}

    for horizon in FORWARD_HORIZONS:
        forward_returns[horizon] = prices.shift(-horizon) / prices - 1

    return daily_returns, forward_returns


def get_asset_names(event_state):

    mapping = event_state[["IssueId", "Asset"]].drop_duplicates().set_index("IssueId")["Asset"].to_dict()

    return mapping


def build_event_responses(event_state, daily_returns, forward_returns, asset_names):

    rows = []
    event_rows = event_state[event_state["InteriorFibEventStart"]]
    issue_ids = sorted(event_state["IssueId"].unique())

    for source_id, target_id in permutations(issue_ids, 2):

        source_events = event_rows[event_rows["IssueId"] == source_id].sort_values("Date")
        
        for event in source_events.itertuples(index=False):

            date = event.Date

            row = {
                "Date": date,
                "SourceIssueId": source_id,
                "SourceAsset": asset_names[source_id],
                "TargetIssueId": target_id,
                "TargetAsset": asset_names[target_id],
                "SourceFibRatio": float(event.InteriorFibRatio),
                "SourceFibLabel": event.InteriorFibLabel,
                "SourceSameDayReturn": daily_returns.loc[date, source_id],
                "TargetSameDayReturn": daily_returns.loc[date, target_id]
            }

            for horizon in FORWARD_HORIZONS:

                row[f"TargetForwardReturn_{horizon}d"] = forward_returns[horizon].loc[date, target_id]

            rows.append(row)

    responses = pd.DataFrame(rows)

    source_sign = np.sign(responses["SourceSameDayReturn"])
    target_sign = np.sign(responses["TargetSameDayReturn"])
    responses["SameDaySignAgreement"] = (source_sign == target_sign)

    return responses


def calculate_statistics(values):

    values = pd.Series(values).dropna().astype(float)

    if len(values) == 0:
        return {
            "N": 0,
            "MeanReturn": np.nan,
            "MedianReturn": np.nan,
            "PositiveRate": np.nan,
            "MeanAbsReturn": np.nan
        }

    return {
        "N": len(values),
        "MeanReturn": float(values.mean()),
        "MedianReturn": float(values.median()),
        "PositiveRate": float((values > 0).mean()),
        "MeanAbsReturn": float(values.abs().mean())
    }


def build_summary(event_state, daily_returns, forward_returns, asset_names):

    rows = []
    issue_ids = sorted(event_state["IssueId"].unique())
    analysis_dates = event_state["Date"].drop_duplicates().sort_values()

    for source_id, target_id in permutations(issue_ids, 2):

        source_state = event_state[event_state["IssueId"] == source_id].sort_values("Date")
        source_event_dates = source_state.loc[source_state["InteriorFibEventStart"], "Date"]
        source_event_date_set = set(source_event_dates)

        non_event_dates = analysis_dates[~analysis_dates.isin(source_event_date_set)]
        event_target_returns = daily_returns[target_id].reindex(source_event_dates)
        non_event_target_returns = daily_returns[target_id].reindex(non_event_dates)

        event_stats = calculate_statistics(event_target_returns)
        baseline_stats = calculate_statistics(non_event_target_returns)
        source_event_returns = daily_returns[source_id].reindex(source_event_dates)

        valid_event_sign = (source_event_returns.notna() & event_target_returns.notna())
        event_sign_agreement = float((np.sign(source_event_returns[valid_event_sign]) == np.sign(event_target_returns[valid_event_sign])).mean())
        source_baseline_returns = daily_returns[source_id].reindex(non_event_dates)

        valid_baseline_sign = (source_baseline_returns.notna() & non_event_target_returns.notna())
        baseline_sign_agreement = float((np.sign(source_baseline_returns[valid_baseline_sign]) == np.sign(non_event_target_returns[valid_baseline_sign])).mean())

        rows.append(
            {
                "SourceIssueId": source_id,
                "SourceAsset": asset_names[source_id],
                "TargetIssueId": target_id,
                "TargetAsset": asset_names[target_id],
                "ResponseWindow": "same_day",
                "HorizonDays": 0,
                "EventN": event_stats["N"],
                "EventMeanReturn": event_stats["MeanReturn"],
                "BaselineMeanReturn": baseline_stats["MeanReturn"],
                "MeanReturnDifference": event_stats["MeanReturn"] - baseline_stats["MeanReturn"],
                "EventMedianReturn": event_stats["MedianReturn"],
                "BaselineMedianReturn": baseline_stats["MedianReturn"],
                "EventPositiveRate": event_stats["PositiveRate"],
                "BaselinePositiveRate": baseline_stats["PositiveRate"],
                "PositiveRateDifference": event_stats["PositiveRate"] - baseline_stats["PositiveRate"],
                "EventMeanAbsReturn": event_stats["MeanAbsReturn"],
                "BaselineMeanAbsReturn": baseline_stats["MeanAbsReturn"],
                "MeanAbsReturnDifference": event_stats["MeanAbsReturn"] - baseline_stats["MeanAbsReturn"],
                "EventSignAgreement": event_sign_agreement,
                "BaselineSignAgreement": baseline_sign_agreement,
                "SignAgreementDifference": event_sign_agreement - baseline_sign_agreement
            }
        )

        for horizon in FORWARD_HORIZONS:

            target_forward = forward_returns[horizon][target_id]
            event_values = target_forward.reindex(source_event_dates)
            baseline_values = target_forward.reindex(non_event_dates)

            event_stats = calculate_statistics(event_values)
            baseline_stats = calculate_statistics(baseline_values)

            rows.append(
                {
                    "SourceIssueId": source_id,
                    "SourceAsset": asset_names[source_id],
                    "TargetIssueId": target_id,
                    "TargetAsset": asset_names[target_id],
                    "ResponseWindow": f"forward_{horizon}d",
                    "HorizonDays": horizon,
                    "EventN": event_stats["N"],
                    "EventMeanReturn": event_stats["MeanReturn"],
                    "BaselineMeanReturn": baseline_stats["MeanReturn"],
                    "MeanReturnDifference": event_stats["MeanReturn"] - baseline_stats["MeanReturn"],
                    "EventMedianReturn": event_stats["MedianReturn"],
                    "BaselineMedianReturn": baseline_stats["MedianReturn"],
                    "EventPositiveRate": event_stats["PositiveRate"],
                    "BaselinePositiveRate": baseline_stats["PositiveRate"],
                    "PositiveRateDifference": event_stats["PositiveRate"] - baseline_stats["PositiveRate"],
                    "EventMeanAbsReturn": event_stats["MeanAbsReturn"],
                    "BaselineMeanAbsReturn": baseline_stats["MeanAbsReturn"],
                    "MeanAbsReturnDifference": event_stats["MeanAbsReturn"] - baseline_stats["MeanAbsReturn"],
                    "EventSignAgreement": np.nan,
                    "BaselineSignAgreement": np.nan,
                    "SignAgreementDifference": np.nan
                }
            )

    return pd.DataFrame(rows)


def print_same_day_summary(summary):

    display = summary[summary["HorizonDays"] == 0].copy()
    percentage_columns = [
        "EventMeanReturn",
        "BaselineMeanReturn",
        "MeanReturnDifference",
        "EventMeanAbsReturn",
        "BaselineMeanAbsReturn",
        "MeanAbsReturnDifference",
        "EventSignAgreement",
        "BaselineSignAgreement",
        "SignAgreementDifference"
    ]

    display[percentage_columns] *= 100
    columns = [
        "SourceAsset",
        "TargetAsset",
        "EventN",
        "EventMeanReturn",
        "BaselineMeanReturn",
        "MeanReturnDifference",
        "EventMeanAbsReturn",
        "BaselineMeanAbsReturn",
        "EventSignAgreement",
        "BaselineSignAgreement",
        "SignAgreementDifference"
    ]

    print("\n" + "=" * 150)
    print("CROSS-ASSET RESPONSE ON THE FIBONACCI EVENT DAY")
    print("=" * 150)

    print(display[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nReturns and sign-agreement values are shown as percentages")


def print_forward_summary(summary):

    display = summary[summary["HorizonDays"] > 0].copy()
    percentage_columns = [
        "EventMeanReturn",
        "BaselineMeanReturn",
        "MeanReturnDifference",
        "EventPositiveRate",
        "BaselinePositiveRate",
        "PositiveRateDifference",
        "EventMeanAbsReturn",
        "BaselineMeanAbsReturn",
        "MeanAbsReturnDifference"
    ]

    display[percentage_columns] *= 100
    columns = [
        "SourceAsset",
        "TargetAsset",
        "HorizonDays",
        "EventN",
        "EventMeanReturn",
        "BaselineMeanReturn",
        "MeanReturnDifference",
        "EventPositiveRate",
        "BaselinePositiveRate",
        "PositiveRateDifference"
    ]

    print("\n" + "=" * 150)
    print("CROSS-ASSET FORWARD RESPONSE AFTER A SOURCE FIBONACCI EVENT")
    print("=" * 150)

    print(display[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nReturns and positive rates are shown as percentages")


def main():

    print("=" * 100)
    print("CROSS-ASSET FIBONACCI RESPONSE ANALYSIS")
    print("=" * 100)

    prices = load_prices()
    event_state = load_event_state()
    validate_inputs(prices, event_state)

    daily_returns, forward_returns = calculate_return_matrices(prices)
    asset_names = get_asset_names(event_state)
    responses = build_event_responses(event_state, daily_returns, forward_returns, asset_names)
    summary = build_summary(event_state, daily_returns, forward_returns, asset_names)

    print_same_day_summary(summary)
    print_forward_summary(summary)

    FIB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    responses.to_csv(DETAIL_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nSaved:")
    print(f"  Detailed event responses:\n  {DETAIL_FILE}")
    print(f"  Cross-asset summary:\n  {SUMMARY_FILE}")

    print("\n" + "=" * 100)
    print("CROSS-ASSET RESPONSE ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()