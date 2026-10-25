from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "workspace" / "fibonacci_multiasset" / "data"
PRICE_FILE = REPO_ROOT / "workspace" / "portfolio_analysis" / "data" / "diversified_prices.csv"
FEATURE_FILE = DATA_DIR / "multi_asset_fibonacci_features.csv"
STATE_FILE = DATA_DIR / "diversified_fibonacci_event_state.csv"

RATIOS = np.array([0.236, 0.382, 0.500, 0.618, 0.786])
LABELS = ["236", "382", "500", "618", "786"]

SHIFTS = [-0.08, -0.055, 0.0, 0.055, 0.08]
TOLERANCE = 0.02
HORIZONS = [1, 5, 20]

def load_inputs():

    features = pd.read_csv(FEATURE_FILE, parse_dates=["Date"], dtype={"IssueId": str})
    state = pd.read_csv(STATE_FILE, parse_dates=["Date"], dtype={"IssueId": str})

    prices = pd.read_csv(PRICE_FILE, parse_dates=["Date"]).set_index("Date").sort_index()
    prices.columns = prices.columns.astype(str)

    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Invalid source prices")

    if (prices.index.has_duplicates or features.duplicated(["Date", "IssueId"]).any() or state.duplicated(["Date", "IssueId"]).any()):
        raise ValueError("Duplicate dates or asset-date records")

    needed = ["Feature_000"] + [f"Feature_{label}" for label in LABELS]
    if features[needed].isna().any().any():
        raise ValueError("Missing Fibonacci feature values")

    if state["InteriorFibEventStart"].dtype != bool:
        parsed = state["InteriorFibEventStart"].astype(str).str.lower().map({"true": True, "false": False})

        if parsed.isna().any():
            raise ValueError("Invalid event-start flag")

        state["InteriorFibEventStart"] = parsed.astype(bool)

    dates = pd.DatetimeIndex(sorted(features["Date"].unique()), name="Date")
    ids = sorted(features["IssueId"].unique())
    expected = pd.MultiIndex.from_product([dates, ids], names=["Date", "IssueId"])

    if not (features.set_index(["Date", "IssueId"]).index.sort_values().equals(expected)):
        raise ValueError("Fibonacci feature panel is not balanced")

    if not (state.set_index(["Date", "IssueId"]).index.sort_values().equals(expected)):
        raise ValueError("Event-state panel differs from feature panel")

    if (not dates.isin(prices.index).all() or not set(ids).issubset(prices.columns)):
        raise ValueError("Features do not align with price matrix")

    names = state[["IssueId", "Asset"]].drop_duplicates().set_index("IssueId")["Asset"].to_dict()

    if len(names) != len(ids):
        raise ValueError("Asset-name mapping is inconsistent")

    return features, state, prices, dates, ids, names


def detect_event_starts(features, dates, ids, shift):

    events = pd.DataFrame(False, index=dates, columns=ids)

    for iid in ids:

        asset = features[features["IssueId"] == iid].sort_values("Date")

        if not pd.DatetimeIndex(asset["Date"]).equals(dates):
            raise ValueError(f"Date alignment failed for {iid}")

        if shift == 0.0:
            distances = asset[[f"Feature_{label}" for label in LABELS]].to_numpy(float)

        else:
            position = asset["Feature_000"].to_numpy(float)
            distances = position[:, None] - (RATIOS + shift)[None, :]

        nearest = np.abs(distances).argmin(axis=1)
        hit = np.abs(distances[np.arange(len(asset)), nearest]) <= TOLERANCE

        previous_hit = np.r_[False, hit[:-1]]
        previous_level = np.r_[-1, nearest[:-1]]

        starts = hit & (~previous_hit | (nearest != previous_level))
        events[iid] = starts

    return events


def describe_simultaneity(events):

    n = len(events)

    lifts = []
    for a, b in combinations(events.columns, 2):
        observed = int((events[a] & events[b]).sum())
        expected = events[a].sum() * events[b].sum() / n
        lifts.append(observed / expected if expected > 0 else np.nan)

    return {
        "TotalEvents": int(events.to_numpy().sum()),
        "DatesAtLeastTwo": int((events.sum(axis=1) >= 2).sum()),
        "MeanPairwiseLift": float(np.nanmean(lifts))
    }


def describe_individual(events, prices, names, label):

    records = []
    for iid in events.columns:

        for horizon in HORIZONS:

            future = (prices[iid].shift(-horizon) / prices[iid] - 1).reindex(events.index)
            event_values = future.loc[events[iid]].dropna()
            ordinary_values = future.loc[~events[iid]].dropna()

            records.append(
                {
                    "LevelSet": label,
                    "Asset": names[iid],
                    "HorizonDays": horizon,
                    "EventN": len(event_values),
                    "NonEventN": len(ordinary_values),
                    "MeanReturnDifferencePP": 100 * (event_values.mean() - ordinary_values.mean()),
                    "PositiveRateDifferencePP": 100 * ((event_values > 0).mean() - (ordinary_values > 0).mean())
                }
            )

    return records


def describe_same_day(events, prices, names, label):

    daily = prices.pct_change(fill_method=None).reindex(events.index)

    records = []
    for source, target in permutations(events.columns, 2):

        event_mask = events[source]
        target_event = daily.loc[event_mask, target].dropna()
        target_other = daily.loc[~event_mask, target].dropna()

        source_returns = daily[source]
        target_returns = daily[target]

        agrees = np.sign(source_returns) == np.sign(target_returns)
        valid = source_returns.notna() & target_returns.notna()

        on_event = event_mask & valid
        off_event = ~event_mask & valid

        records.append(
            {
                "LevelSet": label,
                "Source": names[source],
                "Target": names[target],
                "EventN": len(target_event),
                "AbsReturnMultiplier": target_event.abs().mean() / target_other.abs().mean(),
                "SignAgreementDifferencePP": 100 * (agrees.loc[on_event].mean() - agrees.loc[off_event].mean())
            }
        )

    return records


def main():

    features, state, prices, dates, ids, names = load_inputs()
    recorded = state.pivot(index="Date", columns="IssueId", values="InteriorFibEventStart").reindex(index=dates, columns=ids)

    overview = []
    individual = []
    cross = []

    for shift in SHIFTS:

        label = "Fibonacci" if shift == 0 else f"Shift {shift:+.3f}"
        events = detect_event_starts(features, dates, ids, shift)

        if shift == 0:
            if not events.equals(recorded):
                differences = int((events != recorded).to_numpy().sum())
                raise ValueError(f"Original Fibonacci event reproduction failed: {differences} differences")

            print(f"Original event-start reproduction passed: {int(events.to_numpy().sum())} events")

        overview.append(
            {
                "LevelSet": label,
                "Shift": shift,
                **describe_simultaneity(events)
            }
        )

        individual.extend(describe_individual(events, prices, names, label))
        cross.extend(describe_same_day(events, prices, names, label))

    overall_df = pd.DataFrame(overview)
    individual_df = pd.DataFrame(individual)
    cross_df = pd.DataFrame(cross)

    cross_aggregate = cross_df.groupby("LevelSet", as_index=False)[["AbsReturnMultiplier", "SignAgreementDifferencePP"]].mean()
    overall_df = overall_df.merge(cross_aggregate, on="LevelSet", validate="one_to_one")

    order = ["Fibonacci" if shift == 0 else f"Shift {shift:+.3f}" for shift in SHIFTS]
    overall_df["LevelSet"] = pd.Categorical(overall_df["LevelSet"], categories=order, ordered=True)
    overall_df = overall_df.sort_values("LevelSet")

    print("\nPLACEBO COMPARISON")
    print("(Cross-asset measures average 12 directed pairs)")
    print(overall_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    for horizon in HORIZONS:

        by_asset = individual_df.loc[individual_df["HorizonDays"] == horizon].pivot(index="LevelSet", columns="Asset", values="MeanReturnDifferencePP").reindex(order)

        print(f"\n{horizon}-DAY INDIVIDUAL EVENT MINUS NON-EVENT MEAN RETURNS")
        print("(Percentage points)")
        print(by_asset.to_string(float_format=lambda v: f"{v:+.3f}"))

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    overall_df.to_csv(DATA_DIR / "diversified_fibonacci_placebo_overview.csv", index=False)
    individual_df.to_csv(DATA_DIR / "diversified_fibonacci_placebo_individual.csv", index=False)
    cross_df.to_csv(DATA_DIR / "diversified_fibonacci_placebo_same_day_cross_asset.csv", index=False)

    print("\nSaved three placebo comparison CSV files.")


if __name__ == "__main__":
    main()