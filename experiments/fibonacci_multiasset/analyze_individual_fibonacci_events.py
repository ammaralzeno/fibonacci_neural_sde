from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

PORTFOLIO_DATA_DIR = REPO_ROOT / "workspace" / "portfolio_analysis" / "data"
FIB_DATA_DIR = REPO_ROOT / "workspace" / "fibonacci_multiasset" / "data"
PRICE_FILE = PORTFOLIO_DATA_DIR / "diversified_prices.csv"
EVENT_FILE = FIB_DATA_DIR / "diversified_fibonacci_events.csv"
EVENT_STATE_FILE = FIB_DATA_DIR / "diversified_fibonacci_event_state.csv"
EVENT_RETURN_FILE = FIB_DATA_DIR / "diversified_fibonacci_event_forward_returns.csv"
SUMMARY_FILE = FIB_DATA_DIR / "diversified_fibonacci_individual_asset_summary.csv"

FORWARD_HORIZONS = [1, 5, 20]
INTERIOR_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]

def load_prices():

    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Price file not found:\n{PRICE_FILE}")

    prices = pd.read_csv(PRICE_FILE, parse_dates=["Date"])
    prices = prices.set_index("Date").sort_index()
    prices.columns = prices.columns.astype(str)

    for column in prices.columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    return prices


def load_events():

    if not EVENT_FILE.exists():
        raise FileNotFoundError(f"Event file not found:\n{EVENT_FILE}")

    events = pd.read_csv(EVENT_FILE, parse_dates=["StartDate", "EndDate"], dtype={"IssueId": str, "FibLabel": str})

    return events


def load_event_state():

    if not EVENT_STATE_FILE.exists():
        raise FileNotFoundError(f"Event-state file not found:\n{EVENT_STATE_FILE}")

    event_state = pd.read_csv(EVENT_STATE_FILE, parse_dates=["Date"], dtype={"IssueId": str})

    return event_state


def validate_inputs(prices, events, event_state):

    print("\n" + "=" * 100)
    print("INPUT VALIDATION")
    print("=" * 100)

    if prices.index.duplicated().any():
        raise ValueError("Price matrix contains duplicate dates")

    if prices.isna().any().any():
        raise ValueError("Price matrix contains missing prices")

    if (prices <= 0).any().any():
        raise ValueError("Price matrix contains non-positive prices")

    if events.empty:
        raise ValueError("No Fibonacci events found")

    if event_state.empty:
        raise ValueError("Event-state data is empty")

    required_event_columns = {"IssueId", "Asset", "EventId", "FibRatio", "StartDate", "StartPrice"}
    missing = required_event_columns - set(events.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")

    duplicate_events = events.duplicated(subset=["IssueId", "EventId"]).sum()
    if duplicate_events > 0:
        raise ValueError("Duplicate Fibonacci events found")

    event_issue_ids = set(events["IssueId"])
    missing_price_assets = event_issue_ids - set(prices.columns)
    if missing_price_assets:
        raise ValueError(f"Events contain IssueIds missing from price matrix: {sorted(missing_price_assets)}")

    for row in events.itertuples(index=False):

        if row.StartDate not in prices.index:
            raise ValueError(f"Event date {row.StartDate} not found for {row.IssueId}")

        source_price = float(prices.loc[row.StartDate, row.IssueId])
        if not np.isclose(source_price, float(row.StartPrice), atol=1e-8, rtol=1e-8):
            raise ValueError(f"Start-price mismatch for {row.IssueId}, event {row.EventId}: {row.StartPrice} != {source_price}")

    print(f"Price observations : {len(prices):,}")
    print(f"Assets             : {prices.shape[1]}")
    print(f"Fibonacci events   : {len(events):,}")
    print(f"Feature dates      : {event_state['Date'].nunique():,}")
    print(f"Analysis period    : {event_state['Date'].min().date()} to {event_state['Date'].max().date()}")
    print("Event prices match source prices: True")


def calculate_forward_returns(prices):

    output = {}
    for issue_id in prices.columns:
        asset = pd.DataFrame({"Price": prices[issue_id]})
        for horizon in FORWARD_HORIZONS:
            asset[f"ForwardReturn_{horizon}d"] = asset["Price"].shift(-horizon) / asset["Price"] - 1

        output[issue_id] = asset

    return output


def attach_event_forward_returns(events, forward_returns):

    output = events.copy()
    for horizon in FORWARD_HORIZONS:
        output[f"ForwardReturn_{horizon}d"] = np.nan

    for issue_id, indices in output.groupby("IssueId").groups.items():
        asset_returns = forward_returns[issue_id]
        start_dates = output.loc[indices, "StartDate"]

        for horizon in FORWARD_HORIZONS:
            column = f"ForwardReturn_{horizon}d"
            output.loc[indices, column] = asset_returns[column].reindex(start_dates).to_numpy()

    return output


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


def build_summary(event_returns, event_state, forward_returns):

    rows = []
    asset_info = event_state[["IssueId", "Asset"]].drop_duplicates()
    for asset_row in asset_info.itertuples(index=False):

        issue_id = asset_row.IssueId
        asset_name = asset_row.Asset
        asset_events = event_returns[event_returns["IssueId"] == issue_id]
        asset_state = event_state[event_state["IssueId"] == issue_id].sort_values("Date")
        analysis_dates = asset_state["Date"].drop_duplicates().sort_values()
        event_dates = set(asset_events["StartDate"])
        asset_forward = forward_returns[issue_id]

        level_groups = [("ALL", np.nan, asset_events)]

        for ratio in INTERIOR_LEVELS:
            level_events = asset_events[np.isclose(asset_events["FibRatio"], ratio)]
            level_groups.append((f"{ratio:.3f}", ratio, level_events))

        for level_label, level_ratio, selected_events in level_groups:
            for horizon in FORWARD_HORIZONS:
                return_column = f"ForwardReturn_{horizon}d"
                event_stats = calculate_statistics(selected_events[return_column])

                baseline_series = asset_forward[return_column].reindex(analysis_dates)
                unconditional_stats = calculate_statistics(baseline_series)

                non_event_dates = [date for date in analysis_dates if date not in event_dates]
                non_event_series = asset_forward[return_column].reindex(non_event_dates)
                non_event_stats = calculate_statistics(non_event_series)

                rows.append(
                    {
                        "IssueId": issue_id,
                        "Asset": asset_name,
                        "FibLevel": level_label,
                        "FibRatio": level_ratio,
                        "HorizonDays": horizon,
                        "EventN": event_stats["N"],
                        "EventMeanReturn": event_stats["MeanReturn"],
                        "EventMedianReturn": event_stats["MedianReturn"],
                        "EventPositiveRate": event_stats["PositiveRate"],
                        "EventMeanAbsReturn": event_stats["MeanAbsReturn"],
                        "NonEventN": non_event_stats["N"],
                        "NonEventMeanReturn": non_event_stats["MeanReturn"],
                        "NonEventMedianReturn": non_event_stats["MedianReturn"],
                        "NonEventPositiveRate": non_event_stats["PositiveRate"],
                        "NonEventMeanAbsReturn": non_event_stats["MeanAbsReturn"],
                        "UnconditionalN": unconditional_stats["N"],
                        "UnconditionalMeanReturn": unconditional_stats["MeanReturn"],
                        "MeanReturnDifference": event_stats["MeanReturn"] - non_event_stats["MeanReturn"],
                        "MedianReturnDifference": event_stats["MedianReturn"] - non_event_stats["MedianReturn"],
                        "PositiveRateDifference": event_stats["PositiveRate"]- non_event_stats["PositiveRate"],
                        "MeanAbsReturnDifference": event_stats["MeanAbsReturn"] - non_event_stats["MeanAbsReturn"]
                    }
                )

    return pd.DataFrame(rows)


def print_overall_summary(summary):

    overall = summary[summary["FibLevel"] == "ALL"].copy()
    percentage_columns = [
        "EventMeanReturn",
        "EventMedianReturn",
        "EventPositiveRate",
        "EventMeanAbsReturn",
        "NonEventMeanReturn",
        "NonEventMedianReturn",
        "NonEventPositiveRate",
        "NonEventMeanAbsReturn",
        "MeanReturnDifference",
        "MedianReturnDifference",
        "PositiveRateDifference",
        "MeanAbsReturnDifference",
    ]

    overall[percentage_columns] *= 100
    display_columns = [
        "Asset",
        "HorizonDays",
        "EventN",
        "EventMeanReturn",
        "NonEventMeanReturn",
        "MeanReturnDifference",
        "EventPositiveRate",
        "NonEventPositiveRate",
        "PositiveRateDifference",
        "EventMeanAbsReturn",
        "NonEventMeanAbsReturn",
    ]

    print("\n" + "=" * 130)
    print("ALL INTERIOR FIBONACCI EVENTS — FORWARD RETURN COMPARISON")
    print("=" * 130)

    print(overall[display_columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nReturns and positive rates are shown as percentages")


def print_level_summary(summary):

    level_results = summary[summary["FibLevel"] != "ALL"].copy()
    level_results["EventMeanReturn"] *= 100
    level_results["NonEventMeanReturn"] *= 100
    level_results["MeanReturnDifference"] *= 100

    display_columns = [
        "Asset",
        "FibRatio",
        "HorizonDays",
        "EventN",
        "EventMeanReturn",
        "NonEventMeanReturn",
        "MeanReturnDifference"
    ]

    print("\n" + "=" * 130)
    print("RESULTS BY INDIVIDUAL FIBONACCI LEVEL")
    print("=" * 130)
    print(level_results[display_columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nReturns are shown as percentages")


def main():

    print("=" * 100)
    print("INDIVIDUAL-ASSET FIBONACCI EVENT ANALYSIS")
    print("=" * 100)
    print("Forward horizons: " + ", ".join(f"{h} trading day" + ("" if h == 1 else "s") for h in FORWARD_HORIZONS))

    prices = load_prices()
    events = load_events()
    event_state = load_event_state()
    validate_inputs(prices, events, event_state)

    forward_returns = calculate_forward_returns(prices)
    event_returns = attach_event_forward_returns(events, forward_returns)

    summary = build_summary(event_returns, event_state, forward_returns)
    print_overall_summary(summary)
    print_level_summary(summary)

    FIB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    event_returns.to_csv(EVENT_RETURN_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nSaved:")
    print(f"  Event forward returns:\n  {EVENT_RETURN_FILE}")
    print(f"  Individual-asset summary:\n  {SUMMARY_FILE}")

    print("\n" + "=" * 100)
    print("INDIVIDUAL-ASSET ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()