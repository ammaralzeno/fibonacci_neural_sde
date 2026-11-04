from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_PRICES = ROOT / "workspace" / "portfolio_analysis" / "data"
EXTENSION_DATA = ROOT / "workspace" / "multiasset_extensions" / "data"
ORIGINAL_FIB = ROOT / "workspace" / "fibonacci_multiasset" / "data"
OUTPUT = ROOT / "workspace" / "multiasset_extensions" / "fibonacci"

PORTFOLIO_FILES = {
    "diversified": ORIGINAL_PRICES / "diversified_prices.csv",
    "concentrated": ORIGINAL_PRICES / "concentrated_prices.csv",
    "ultra_concentrated": EXTENSION_DATA / "ultra_concentrated_prices.csv"
}

PERIODS = {
    "2016-2017": ("2016-01-01", "2017-12-31"),
    "2018-2019": ("2018-01-01", "2019-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "full_2015-2020": ("2015-01-01", "2020-12-31")
}

LOOKBACK = 252
TOLERANCE = 0.02

RATIOS = np.array([0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0])

FEATURE_COLUMNS = [
    "Feature_000",
    "Feature_236",
    "Feature_382",
    "Feature_500",
    "Feature_618",
    "Feature_786",
    "Feature_1000"
]


def load_portfolios():

    portfolios = {}

    for name, path in PORTFOLIO_FILES.items():
        prices = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        prices.columns = prices.columns.astype(str)

        if prices.shape[1] != 4 or prices.index.has_duplicates:
            raise ValueError(f"Invalid price panel: {name}")

        if prices.columns.duplicated().any():
            raise ValueError(f"Duplicate IssueIds: {name}")

        values = prices.to_numpy(dtype=float)

        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Invalid prices: {name}")

        portfolios[name] = prices

    reference = portfolios["diversified"].index

    if len(reference) != 1512:
        raise ValueError("Unexpected reference calendar length")

    for name, prices in portfolios.items():
        if not prices.index.equals(reference):
            raise ValueError(f"Calendar mismatch: {name}")

    print(f"Aligned portfolios: 3 x 4 assets, {len(reference):,} price dates")

    return portfolios


def make_features_and_events(portfolios):

    feature_frames = []
    event_frames = []

    for portfolio, prices in portfolios.items():

        dates = prices.index[LOOKBACK - 1:]

        for iid in prices.columns:

            values = prices[iid].to_numpy(dtype=float)
            windows = np.lib.stride_tricks.sliding_window_view(values, LOOKBACK)

            minimum = windows.min(axis=1)
            maximum = windows.max(axis=1)
            spread = maximum - minimum

            if (spread <= 0).any():
                raise ValueError(f"Zero price range: {portfolio}/{iid}")

            position = (windows[:, -1] - minimum) / spread
            distances = position[:, None] - RATIOS[None, :]
            features = pd.DataFrame(distances, columns=FEATURE_COLUMNS)

            features.insert(0, "IssueId", iid)
            features.insert(0, "Portfolio", portfolio)
            features.insert(0, "Date", dates)
            feature_frames.append(features)

            interior = distances[:, 1:-1]
            nearest = np.abs(interior).argmin(axis=1)
            nearest_abs = np.abs(interior[np.arange(len(dates)), nearest])

            hit = nearest_abs <= TOLERANCE
            previous_hit = np.r_[False, hit[:-1]]
            previous_level = np.r_[-1, nearest[:-1]]
            start = hit & (~previous_hit | (nearest != previous_level))

            event_frames.append(
                pd.DataFrame({
                    "Date": dates,
                    "Portfolio": portfolio,
                    "IssueId": iid,
                    "InteriorFibHit": hit,
                    "InteriorFibEventStart": start,
                    "InteriorFibRatio": np.where(hit, RATIOS[nearest + 1], np.nan),
                    "NearestInteriorDistance": nearest_abs
                })
            )

    all_features = pd.concat(feature_frames, ignore_index=True)
    all_events = pd.concat(event_frames, ignore_index=True)

    for frame in [all_features, all_events]:
        if frame.duplicated(["Portfolio", "IssueId", "Date"]).any():
            raise ValueError("Duplicate panel records")

    expected = len(portfolios) * 4 * (len(next(iter(portfolios.values()))) - LOOKBACK + 1)

    if len(all_features) != expected or len(all_events) != expected:
        raise ValueError("Unexpected panel size")

    return all_features, all_events


def validate_original_diversified(features, events):

    old_features = pd.read_csv(ORIGINAL_FIB / "multi_asset_fibonacci_features.csv", parse_dates=["Date"], dtype={"IssueId": str})
    old_state = pd.read_csv(ORIGINAL_FIB / "diversified_fibonacci_event_state.csv", parse_dates=["Date"], dtype={"IssueId": str})

    new_features = features.loc[features["Portfolio"] == "diversified"]
    new_events = events.loc[events["Portfolio"] == "diversified"]

    check_columns = FEATURE_COLUMNS[:-1]
    missing_columns = set(check_columns) - set(old_features.columns)

    if missing_columns:
        raise ValueError(f"Missing original feature columns: {sorted(missing_columns)}")

    if len(old_features) != len(new_features) or len(old_state) != len(new_events):
        raise ValueError("Original diversified row counts differ")

    for old, new in [(old_features, new_features), (old_state, new_events)]:

        if old.duplicated(["Date", "IssueId"]).any():
            raise ValueError("Duplicate original records")

        old_keys = pd.MultiIndex.from_frame(old[["Date", "IssueId"]])
        new_keys = pd.MultiIndex.from_frame(new[["Date", "IssueId"]])

        if set(old_keys) != set(new_keys):
            raise ValueError("Original and extended diversified dates differ")

    old_f = old_features.set_index(["Date", "IssueId"]).sort_index()
    new_f = new_features.set_index(["Date", "IssueId"]).sort_index()

    errors = np.abs(old_f[check_columns].to_numpy(float) - new_f[check_columns].to_numpy(float))
    max_error = float(errors.max())

    if not np.isfinite(max_error) or max_error > 5e-6:
        raise ValueError(f"Original feature reproduction failed: {max_error}")

    old_e = old_state.set_index(["Date", "IssueId"]).sort_index()
    new_e = new_events.set_index(["Date", "IssueId"]).sort_index()
    old_flags = old_e["InteriorFibEventStart"].astype(str).str.lower().map({"true": True, "false": False})

    if old_flags.isna().any():
        raise ValueError("Invalid original event flags")

    mismatches = int((old_flags.to_numpy(bool)!= new_e["InteriorFibEventStart"].to_numpy(bool)).sum())

    if mismatches:
        raise ValueError(f"Original event reproduction failed: {mismatches} differences")

    event_count = int(old_flags.sum())

    if event_count != 530:
        raise ValueError("Unexpected original event count")

    print(f"Original diversified features reproduced: maximum error {max_error:.2e}")
    print(f"Original diversified events reproduced: {event_count}")


def summarize_events(events):

    rows = []

    for portfolio in PORTFOLIO_FILES:

        data = events.loc[events["Portfolio"] == portfolio]

        for period, (start, end) in PERIODS.items():

            sample = data.loc[data["Date"].between(start, end)]
            daily = sample.groupby("Date")["InteriorFibEventStart"].sum()
            n_dates = len(daily)

            if len(sample) != 4 * n_dates or n_dates == 0:
                raise ValueError(f"Unbalanced panel: {portfolio}/{period}")

            rows.append({
                "Portfolio": portfolio,
                "Period": period,
                "N_Dates": n_dates,
                "EventStarts": int(sample["InteriorFibEventStart"].sum()),
                "HitDays": int(sample["InteriorFibHit"].sum()),
                "DatesWith2PlusEvents": int((daily >= 2).sum()),
                "DatesWith3PlusEvents": int((daily >= 3).sum())
            })

    return pd.DataFrame(rows)


def main():

    print("=" * 85)
    print("THREE-PORTFOLIO FIBONACCI FEATURES AND EVENTS")
    print("=" * 85)

    portfolios = load_portfolios()
    features, events = make_features_and_events(portfolios)
    validate_original_diversified(features, events)
    summary = summarize_events(events)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT / "portfolio_fibonacci_features.csv", index=False)
    events.to_csv(OUTPUT / "portfolio_fibonacci_event_state.csv", index=False)
    summary.to_csv(OUTPUT / "portfolio_fibonacci_event_count_preview.csv", index=False)

    print(f"\nFeature records: {len(features):,}")
    print(f"Event-state records: {len(events):,}")

    print("\nEVENT COUNTS BY PORTFOLIO AND PERIOD")
    print(summary.to_string(index=False))

    print(f"\nSaved three CSVs under:\n{OUTPUT}")


if __name__ == "__main__":
    main()