from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "workspace" / "fibonacci_multiasset" / "data"
FEATURE_FILE = DATA_DIR / "multi_asset_fibonacci_features.csv"
EVENT_STATE_FILE = DATA_DIR / "diversified_fibonacci_event_state.csv"
EVENT_FILE = DATA_DIR / "diversified_fibonacci_events.csv"
EVENT_SUMMARY_FILE = DATA_DIR / "diversified_fibonacci_event_summary.csv"
LEVEL_SUMMARY_FILE = DATA_DIR / "diversified_fibonacci_level_summary.csv"

TOLERANCE = 0.02

INTERIOR_LEVELS = {
    "236": 0.236,
    "382": 0.382,
    "500": 0.500,
    "618": 0.618,
    "786": 0.786
}

ENDPOINT_LEVELS = {
    "000": 0.000,
    "1000": 1.000
}


def load_features():

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Fibonacci feature file not found:\n{FEATURE_FILE}")

    features = pd.read_csv(FEATURE_FILE, parse_dates=["Date"], dtype={"IssueId": str})
    features = features.sort_values(["IssueId", "Date"]).reset_index(drop=True)

    required_columns = {"Date", "IssueId", "Asset", "Price"}
    required_columns.update({f"Feature_{label}" for label in (list(INTERIOR_LEVELS) + list(ENDPOINT_LEVELS))})

    missing = required_columns - set(features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if features[list(required_columns)].isna().any().any():
        raise ValueError("Missing values found in Fibonacci feature data")

    return features


def calculate_nearest_level(asset_data, levels):

    labels = list(levels.keys())
    ratios = np.array([levels[label] for label in labels], dtype=float)

    columns = [f"Feature_{label}" for label in labels]
    distances = asset_data[columns].to_numpy(dtype=float)
    absolute_distances = np.abs(distances)
    nearest_index = np.argmin(absolute_distances, axis=1)

    rows = np.arange(len(asset_data))
    nearest_signed_distance = distances[rows, nearest_index]
    nearest_abs_distance = np.abs(nearest_signed_distance)
    nearest_labels = np.array(labels, dtype=object)[nearest_index]
    nearest_ratios = ratios[nearest_index]

    return nearest_labels, nearest_ratios, nearest_signed_distance, nearest_abs_distance


def detect_asset_events(asset_data):

    asset_data = asset_data.sort_values("Date").copy().reset_index(drop=True)
    interior_label, interior_ratio, interior_signed_distance, interior_abs_distance = calculate_nearest_level(asset_data, INTERIOR_LEVELS)
    endpoint_label, endpoint_ratio, _, endpoint_abs_distance = calculate_nearest_level(asset_data, ENDPOINT_LEVELS)

    interior_hit = (interior_abs_distance <= TOLERANCE)
    asset_data["InteriorFibHit"] = interior_hit
    asset_data["InteriorFibLabel"] = np.where(interior_hit, interior_label, None)
    asset_data["InteriorFibRatio"] = np.where(interior_hit, interior_ratio, np.nan)
    asset_data["InteriorFibSignedDistance"] = np.where(interior_hit, interior_signed_distance, np.nan)
    asset_data["InteriorFibAbsDistance"] = np.where(interior_hit, interior_abs_distance, np.nan)

    endpoint_hit = (endpoint_abs_distance <= TOLERANCE)
    asset_data["EndpointHit"] = endpoint_hit
    asset_data["EndpointLabel"] = np.where(endpoint_hit, endpoint_label, None)
    asset_data["EndpointRatio"] = np.where(endpoint_hit, endpoint_ratio, np.nan)
    asset_data["EndpointAbsDistance"] = np.where(endpoint_hit, endpoint_abs_distance, np.nan)

    previous_hit = asset_data["InteriorFibHit"].shift(1, fill_value=False)
    previous_label = asset_data["InteriorFibLabel"].shift(1)

    same_level_as_previous = asset_data["InteriorFibLabel"] == previous_label 
    asset_data["InteriorFibEventStart"] = asset_data["InteriorFibHit"] & (~previous_hit | ~same_level_as_previous)

    event_number = asset_data["InteriorFibEventStart"].cumsum()
    asset_data["InteriorFibEventId"] = np.where(asset_data["InteriorFibHit"], event_number, np.nan)

    return asset_data


def build_event_table(event_state):

    rows = []
    hit_rows = event_state[event_state["InteriorFibHit"]].copy()

    for (issue_id, event_id), group in hit_rows.groupby(["IssueId", "InteriorFibEventId"]):
        
        group = group.sort_values("Date")

        first = group.iloc[0]
        last = group.iloc[-1]

        levels = group["InteriorFibLabel"].dropna().unique()

        if len(levels) != 1:
            raise ValueError(f"Event {event_id} for {issue_id} contains multiple Fibonacci levels")
       
        rows.append(
            {
                "IssueId": issue_id,
                "Asset": first["Asset"],
                "EventId": int(event_id),
                "FibLabel": first["InteriorFibLabel"],
                "FibRatio": float(first["InteriorFibRatio"]),
                "StartDate": first["Date"],
                "EndDate": last["Date"],
                "DurationTradingDays": len(group),
                "StartPrice": float(first["Price"]),
                "EntrySignedDistance": float(first["InteriorFibSignedDistance"]),
                "MinimumAbsDistance": float(group["InteriorFibAbsDistance"].min())
            }
        )

    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values(["StartDate", "IssueId"]).reset_index(drop=True)

    return events


def build_event_summary(event_state, events):

    rows = []
    for (issue_id, asset_name), group in event_state.groupby(["IssueId", "Asset"]):

        asset_events = events[events["IssueId"] == issue_id]
        observations = len(group)
        hit_days = int(group["InteriorFibHit"].sum())

        endpoint_hit_days = int(group["EndpointHit"].sum())
        event_count = len(asset_events)

        if event_count > 0:
            median_duration = float(asset_events["DurationTradingDays"].median())
            mean_duration = float(asset_events["DurationTradingDays"].mean())

        else:
            median_duration = np.nan
            mean_duration = np.nan

        rows.append(
            {
                "IssueId": issue_id,
                "Asset": asset_name,
                "Observations": observations,
                "InteriorHitDays": hit_days,
                "InteriorHitRate": hit_days / observations,
                "InteriorEvents": event_count,
                "EventsPer100Days": event_count / observations * 100,
                "MeanEventDuration": mean_duration,
                "MedianEventDuration": median_duration,
                "EndpointHitDays": endpoint_hit_days,
                "EndpointHitRate": endpoint_hit_days / observations
            }
        )

    return pd.DataFrame(rows)


def build_level_summary(event_state, events):

    rows = []

    for (issue_id, asset_name), group in event_state.groupby(["IssueId", "Asset"]):
        for label, ratio in INTERIOR_LEVELS.items():
            hit_days = int((group["InteriorFibLabel"] == label).sum())
            asset_events = events[(events["IssueId"] == issue_id) & (events["FibLabel"] == label)]

            rows.append(
                {
                    "IssueId": issue_id,
                    "Asset": asset_name,
                    "FibLabel": label,
                    "FibRatio": ratio,
                    "HitDays": hit_days,
                    "EventCount": len(asset_events)
                }
            )

    return pd.DataFrame(rows)


def print_event_summary(summary):

    display = summary.copy()

    display["InteriorHitRate"] *= 100
    display["EndpointHitRate"] *= 100

    print("\n" + "=" * 100)
    print("FIBONACCI EVENT SUMMARY")
    print("=" * 100)

    print(display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nHit rates shown as percentages")


def print_level_summary(level_summary):

    print("\n" + "=" * 100)
    print("EVENTS BY INTERIOR FIBONACCI LEVEL")
    print("=" * 100)

    display = level_summary[["Asset", "FibRatio", "HitDays", "EventCount"]]

    print(display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():

    print("=" * 100)
    print("MULTI-ASSET FIBONACCI EVENT DETECTION")
    print("=" * 100)

    print(f"Interior-zone tolerance : {TOLERANCE:.3f}")
    print("Interior levels          : " + ", ".join(f"{ratio:.3f}" for ratio in INTERIOR_LEVELS.values()))

    features = load_features()
    asset_outputs = []

    for _, group in features.groupby("IssueId", sort=False):
        detected = detect_asset_events(group)
        asset_outputs.append(detected)

    event_state = pd.concat(asset_outputs, ignore_index=True)
    event_state = event_state.sort_values(["Date", "IssueId"]).reset_index(drop=True)
    events = build_event_table(event_state)

    summary = build_event_summary(event_state, events)
    level_summary = build_level_summary(event_state, events)
    print_event_summary(summary)
    print_level_summary(level_summary)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    event_state.to_csv(EVENT_STATE_FILE, index=False)
    events.to_csv(EVENT_FILE, index=False)
    summary.to_csv(EVENT_SUMMARY_FILE, index=False)
    level_summary.to_csv(LEVEL_SUMMARY_FILE, index=False)

    print("\nSaved:")
    print(f"  Event state:\n  {EVENT_STATE_FILE}")
    print(f"  Event episodes:\n  {EVENT_FILE}")
    print(f"  Asset summary:\n  {EVENT_SUMMARY_FILE}")
    print(f"  Level summary:\n  {LEVEL_SUMMARY_FILE}")

    print("\n" + "=" * 100)
    print("FIBONACCI EVENT DETECTION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()