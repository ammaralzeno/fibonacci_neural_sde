from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_PRICES = ROOT / "workspace" / "portfolio_analysis" / "data"
EXTENSION_DATA = ROOT / "workspace" / "multiasset_extensions" / "data"
FIB_DIR = ROOT / "workspace" / "multiasset_extensions" / "fibonacci"
ORIGINAL_FIB = ROOT / "workspace" / "fibonacci_multiasset" / "data"

PRICE_FILES = {
    "diversified": ORIGINAL_PRICES / "diversified_prices.csv",
    "concentrated": ORIGINAL_PRICES / "concentrated_prices.csv",
    "ultra_concentrated": EXTENSION_DATA / "ultra_concentrated_prices.csv"
}

EVENT_FILE = FIB_DIR / "portfolio_fibonacci_event_state.csv"

OUTPUT_FILE = FIB_DIR / "portfolio_fibonacci_forward_returns_by_period.csv"

PERIODS = {
    "2016-2017": ("2016-01-01", "2017-12-31"),
    "2018-2019": ("2018-01-01", "2019-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "full_2015-2020": ("2015-01-01", "2020-12-31")
}

HORIZONS = [1, 5, 20]
LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]

def load_data():

    portfolios = {}

    for portfolio, path in PRICE_FILES.items():

        if not path.exists():
            raise FileNotFoundError(path)

        prices = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        prices.columns = prices.columns.astype(str)

        if (
            prices.shape[1] != 4
            or prices.index.has_duplicates
            or prices.columns.has_duplicates
        ):
            raise ValueError(f"Invalid price structure: {portfolio}")

        values = prices.to_numpy(dtype=float)

        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Invalid prices: {portfolio}")

        portfolios[portfolio] = prices

    reference_dates = portfolios["diversified"].index

    for portfolio, prices in portfolios.items():

        if not prices.index.equals(reference_dates):
            raise ValueError(f"Calendar mismatch: {portfolio}")

    events = pd.read_csv(EVENT_FILE, parse_dates=["Date"], dtype={"IssueId": str})

    if events.duplicated(["Portfolio", "IssueId", "Date"]).any():
        raise ValueError("Duplicate event-state records")

    parsed = events["InteriorFibEventStart"].astype(str).str.lower().map({"true": True, "false": False})

    if parsed.isna().any():
        raise ValueError("Invalid event-start flags")

    events["InteriorFibEventStart"] = parsed.astype(bool)

    events["InteriorFibRatio"] = pd.to_numeric(events["InteriorFibRatio"], errors="coerce")

    if events.loc[events["InteriorFibEventStart"], "InteriorFibRatio"].isna().any():
        raise ValueError("Missing level at an event start")

    if len(events) != 15132:
        raise ValueError("Unexpected event-state row count")

    expected_dates = reference_dates[251:]

    if len(expected_dates) != 1261:
        raise ValueError("Unexpected Fibonacci calendar")

    for portfolio, prices in portfolios.items():

        subset = events[events["Portfolio"] == portfolio]
        expected = pd.MultiIndex.from_product([expected_dates, prices.columns], names=["Date", "IssueId"]).sort_values()
        actual = subset.set_index(["Date", "IssueId"]).index.sort_values()

        if not actual.equals(expected):
            raise ValueError(f"Event-state alignment mismatch: {portfolio}")

    print("Portfolio and event alignment: PASSED")

    return portfolios, events


def statistics(values):

    values = pd.Series(values).dropna().astype(float)

    if values.empty:
        return {
            "N": 0,
            "Mean": np.nan,
            "Median": np.nan,
            "PositiveRate": np.nan,
            "MeanAbs": np.nan
        }

    return {
        "N": int(len(values)),
        "Mean": float(values.mean()),
        "Median": float(values.median()),
        "PositiveRate": float((values > 0).mean()),
        "MeanAbs": float(values.abs().mean())
    }


def analyze_portfolio(portfolio, prices, events):

    rows = []
    state = events[events["Portfolio"] == portfolio].copy()

    for period, (start, end) in PERIODS.items():

        period_prices = prices.loc[start:end]
        period_state = state[state["Date"].between(start, end)]

        for iid in prices.columns:

            asset_state = period_state[period_state["IssueId"] == iid].set_index("Date").sort_index()

            if (
                not asset_state.index.is_unique
                or not asset_state.index.isin(period_prices.index).all()
            ):
                raise ValueError(f"State/date mismatch: {portfolio}/{period}/{iid}")

            event_start = asset_state["InteriorFibEventStart"]
            event_level = asset_state["InteriorFibRatio"]

            for horizon in HORIZONS:

                forward = (period_prices[iid].shift(-horizon) / period_prices[iid] - 1).reindex(asset_state.index)
                valid = forward.notna()

                baseline_values = forward[valid & ~event_start]
                baseline = statistics(baseline_values)

                for level in ["ALL"] + LEVELS:

                    if level == "ALL":
                        selected = event_start

                    else:
                        selected = pd.Series(np.isclose(event_level.to_numpy(float), level, rtol=0, atol=1e-10), index=asset_state.index) & event_start

                    event_values = forward[valid & selected]
                    event_stats = statistics(event_values)

                    if event_stats["N"] == 0:
                        mean_diff = np.nan
                        positive_diff = np.nan
                        absolute_diff = np.nan

                    else:
                        mean_diff = event_stats["Mean"] - baseline["Mean"]
                        positive_diff = event_stats["PositiveRate"] - baseline["PositiveRate"]
                        absolute_diff = event_stats["MeanAbs"] - baseline["MeanAbs"]

                    rows.append({
                        "Portfolio": portfolio,
                        "Period": period,
                        "IssueId": iid,
                        "FibLevel": str(level),
                        "HorizonDays": horizon,
                        "EventN": event_stats["N"],
                        "BaselineN": baseline["N"],
                        "EventMeanReturn": event_stats["Mean"],
                        "BaselineMeanReturn": baseline["Mean"],
                        "MeanReturnDifference": mean_diff,
                        "EventMedianReturn": event_stats["Median"],
                        "EventPositiveRate": event_stats["PositiveRate"],
                        "BaselinePositiveRate": baseline["PositiveRate"],
                        "PositiveRateDifference": positive_diff,
                        "EventMeanAbsReturn": event_stats["MeanAbs"],
                        "BaselineMeanAbsReturn": baseline["MeanAbs"],
                        "MeanAbsReturnDifference": absolute_diff
                    })

    return rows


def validate_original(results):

    old_path = ORIGINAL_FIB / "diversified_fibonacci_individual_asset_summary.csv"
    state_path = ORIGINAL_FIB / "diversified_fibonacci_event_state.csv"

    old = pd.read_csv(old_path)
    old_state = pd.read_csv(state_path, dtype={"IssueId": str})

    mapping = old_state[["IssueId", "Asset"]].drop_duplicates()

    if mapping["IssueId"].duplicated().any():
        raise ValueError("Ambiguous original asset mapping")

    old = old[old["FibLevel"] == "ALL"].copy()
    new = results[
        (results["Portfolio"] == "diversified")
        & (results["Period"] == "full_2015-2020")
        & (results["FibLevel"] == "ALL")
    ].copy()

    new = new.merge(mapping, on="IssueId", validate="many_to_one")

    if len(old) != 12 or len(new) != 12:
        raise ValueError("Expected 12 original pooled results")

    merged = new.merge(
        old[
            [
                "Asset",
                "HorizonDays",
                "EventN",
                "MeanReturnDifference"
            ]
        ],
        on=["Asset", "HorizonDays"],
        suffixes=("", "_Original"),
        validate="one_to_one"
    )

    if len(merged) != 12:
        raise ValueError("Original comparison is incomplete")

    if not (merged["EventN"] == merged["EventN_Original"]).all():
        raise ValueError("Original event counts differ")

    differences = np.abs(merged["MeanReturnDifference"] - merged["MeanReturnDifference_Original"])
    max_error = float(differences.max())

    if not np.isfinite(max_error) or max_error > 1e-10:
        raise ValueError(f"Original forward-return reproduction failed: {max_error}")

    print(f"Original diversified forward-return results reproduced: maximum error {max_error:.2e}")


def main():

    print("=" * 90)
    print("FIBONACCI INDIVIDUAL FORWARD RETURNS BY PERIOD")
    print("=" * 90)

    portfolios, events = load_data()
    rows = []

    for portfolio, prices in portfolios.items():
        rows.extend(analyze_portfolio(portfolio, prices, events))

    results = pd.DataFrame(rows)
    expected_rows = 3 * 4 * 4 * 3 * 6

    if len(results) != expected_rows:
        raise ValueError(f"Expected {expected_rows} results, got {len(results)}")

    validate_original(results)

    pooled = results[(results["FibLevel"] == "ALL") & (results["Period"] != "full_2015-2020")].copy()
    pooled["MeanReturnDifferencePP"] = 100 * pooled["MeanReturnDifference"]

    summary = (pooled
        .groupby(["Portfolio", "Period", "HorizonDays"], as_index=False)
        .agg(MeanDifferencePP=("MeanReturnDifferencePP", "mean"), MinEventN=("EventN", "min")))

    print("\nMEAN EVENT MINUS NON-EVENT RETURN DIFFERENCE")
    print("(Percentage points; equal-weight mean across four assets)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    small = results[(results["FibLevel"] != "ALL") & (results["EventN"] < 10)]
    print(f"\nLevel-specific rows with fewer than 10 events: {len(small)}")

    FIB_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(results)} results:")
    print(OUTPUT_FILE)

    print("\nFORWARD-RETURN ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()